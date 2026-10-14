"""Streaming motions into the run_isg policy loop, and the blocking robot API for code agents.

``MotionSession`` turns one prepared motion into one 23-D action per policy call. ``ThreadBridge`` + ``RobotAPI`` let
a program run in a worker thread (a CaP-X-style code agent): each API call hands a :class:`MotionRequest` to the policy
thread and blocks until that motion has been streamed, then returns its :class:`MotionOutcome`; state and images are
the latest observation the policy thread published. Both planners therefore share one motion layer and one set of
bounds."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

from ispatialgym.harness.core.types import MotionOutcome, MotionPlan, MotionRejected, MotionRequest, Observation


class MotionSession:
    def __init__(self, layer):
        self.layer = layer
        self.gen = None
        self.request: Optional[MotionRequest] = None
        self.plan: Optional[MotionPlan] = None
        self.steps = 0

    @property
    def busy(self) -> bool:
        return self.gen is not None

    def start(self, req: MotionRequest) -> MotionPlan:
        """Plan the motion (raises :class:`MotionRejected` before anything moves)."""
        plan, gen = self.layer.prepare(req)
        self.gen, self.request, self.plan, self.steps = gen, req, plan, 0
        return plan

    def step(self) -> Tuple[Optional[Any], Optional[MotionOutcome]]:
        """``(action, None)`` while the motion runs, ``(None, outcome)`` when it has finished."""
        if self.gen is None:
            return None, None
        try:
            a = next(self.gen)
        except StopIteration as s:
            outcome = s.value if isinstance(s.value, MotionOutcome) else MotionOutcome(True, "done")
            outcome.steps = self.steps
            self.gen = None
            return None, outcome
        except Exception as e:  # noqa: BLE001 - a failing controller ends this motion, not the episode
            outcome = MotionOutcome(False, f"motion failed: {type(e).__name__}: {e}", steps=self.steps)
            self.gen = None
            return None, outcome
        self.steps += 1
        return a, None


class ThreadBridge:
    """One pending request at a time between a worker thread (``submit``) and the policy thread (``take`` / ``done``)."""

    def __init__(self):
        self._req: "queue.Queue[MotionRequest]" = queue.Queue(maxsize=1)
        self._res: "queue.Queue[MotionOutcome]" = queue.Queue(maxsize=1)
        self.latest: Optional[Observation] = None
        self.closed = threading.Event()

    def submit(self, req: MotionRequest, timeout: Optional[float] = None) -> MotionOutcome:
        """Blocks until the motion has run; raises ``RuntimeError`` once the episode has ended (so a worker blocked here
        cannot keep the process alive) and ``TimeoutError`` after ``timeout`` seconds."""
        if self.closed.is_set():
            raise RuntimeError("the episode has ended")
        self._req.put(req, timeout=timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self._res.get(timeout=0.1)
            except queue.Empty:
                if self.closed.is_set():
                    raise RuntimeError("the episode has ended") from None
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"no outcome for {req.kind} after {timeout} s") from None

    def take(self, timeout: float = 0.0) -> Optional[MotionRequest]:
        try:
            return self._req.get(timeout=timeout) if timeout > 0 else self._req.get_nowait()
        except queue.Empty:
            return None

    def done(self, outcome: MotionOutcome) -> None:
        self._res.put(outcome)

    def publish(self, obs: Observation) -> None:
        self.latest = obs

    def close(self) -> None:
        self.closed.set()


class RobotAPI:
    """Blocking robot calls for generated programs; the bounds, planner and controllers are the motion layer's."""

    def __init__(self, bridge: ThreadBridge):
        self._b = bridge

    def _do(self, kind: str, **params: Any) -> MotionOutcome:
        return self._b.submit(MotionRequest(kind, {k: v for k, v in params.items() if v is not None}))

    def move_eef(self, targets: Dict[str, float]) -> MotionOutcome:
        """Named absolute targets (``left_x`` ... ``right_gripper``, :mod:`core.pose`); unnamed dimensions hold."""
        return self._do("move_eef", targets=dict(targets))

    def move_base(self, forward_m: float = 0.0, left_m: float = 0.0, turn_deg: float = 0.0) -> MotionOutcome:
        return self._do("move_base", forward_m=forward_m, left_m=left_m, turn_deg=turn_deg)

    def move_torso(self, translate: Optional[float] = None, tilt_deg: Optional[float] = None) -> MotionOutcome:
        return self._do("move_torso", translate=translate, tilt_deg=tilt_deg)

    def wait(self, steps: int = 30) -> MotionOutcome:
        return self._do("hold", steps=steps)

    def get_state(self) -> Dict[str, Any]:
        return self._b.latest.state.to_dict() if self._b.latest else {}

    def get_images(self) -> Dict[str, Any]:
        return dict(self._b.latest.images) if self._b.latest else {}


def serve_bridge(session: MotionSession, bridge: ThreadBridge, wait_s: float = 0.0) -> Tuple[Optional[Any], bool]:
    """One policy-thread step for a bridged program: start the pending request when idle, stream it, hand back the
    outcome. Returns ``(action, idle)``; ``idle`` means no motion is running (the caller emits a hold action).
    ``wait_s`` lets the policy thread wait for the program's next call instead of stepping the simulation meanwhile."""
    if not session.busy:
        req = bridge.take(wait_s)
        if req is None:
            return None, True
        try:
            session.start(req)
        except MotionRejected as e:
            bridge.done(MotionOutcome(False, f"rejected (nothing moved): {e}"))
            return None, True
    a, outcome = session.step()
    if outcome is not None:
        bridge.done(outcome)
        return None, True
    return a, False
