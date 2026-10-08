"""``VecStreamExecutor``: the executor's stream + epilogue (``Executor.run_stream`` / ``Executor.epilogue``) for N environments of
one vectorized ``og.Environment``, every environment with its own stream, step counters, QA monitors, recorder, settle
windows and verdict.

One ``env.step`` per control step for the whole batch (BEHAVIOR's ``omnigibson/eval/evaluator.py``): the ``(N, action_dim)``
action holds, per environment, the action its stream produced this step, or - once the stream has ended - the executor's
stream no-op (its **last streamed action with the base stopped**) during the settle windows and after its verdict until the
batch ends. Everything that decides an episode is per environment and reuses the single-env executor's rules unchanged:

* the stream ends at the source's ``None`` (horizon / stop-on-success) or at that env's first QA verdict (timeout);
* the epilogue windows: ``max(settle_min_steps, ceil(settle_fraction * n_action_steps))``, then a second window of
  ``settle_second_window_factor`` x when the robot / a tracked object moved or the binary goal is lost at its end
  (``Executor._moved``), the lab's never-held rule (``never_held_window`` steps, no second window) when the goal never held
  during the stream;
* the verdicts and the result dict of ``Executor.epilogue`` (``not_reached`` / ``tolerance`` / ``not_settled`` / timeout).

A finished environment's robot keeps receiving its no-op; nothing of it is recorded any more. The per-env ``timings`` are wall
times of the shared loop (every environment advances together).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch as th

from ispatialgym.oracle.executor import Executor, ExecutorConfig, MotionSnapshot, _StreamAction, _hold_actions, _jsonable, _out_of_tolerance, _pose_reached_of, _unsatisfied
from ispatialgym.oracle.qa import QAStats, Verdict
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE

ActionSource = Callable[[Any, Any], Optional[Any]]  # (obs, info) -> action | None


def isg_of(info: Any) -> Dict[str, Any]:
    info = info or {}
    return (info.get("done") or {}).get("ispatialgym") or info.get("ispatialgym") or {}


class _Rules:
    """The single-env executor's epilogue rules bound to one environment (``Executor._moved`` / ``orientation_targets`` /
    the D-007 branch log read the instance's ``cfg``, ``task`` and ``isg``)."""

    _moved = Executor._moved
    orientation_targets = Executor.orientation_targets
    _reset_predicate_branches = Executor._reset_predicate_branches
    _predicate_branches = Executor._predicate_branches

    def __init__(self, cfg: ExecutorConfig, task):
        self.cfg, self.task = cfg, task
        self.isg: Dict[str, Any] = {}


@dataclass
class Slot:
    """One environment of the batch and its episode state."""

    idx: int
    robot: Any
    source: ActionSource
    cfg: ExecutorConfig = field(default_factory=ExecutorConfig)
    task: Any = None
    monitors: Any = None
    recorder: Any = None
    tracked: Dict[str, Any] = field(default_factory=dict)
    held: Callable[[], list] = lambda: []
    never_held_window: Optional[int] = 10  # the lab's stream rule; None = the executor's default windows in every case
    # -- state --
    stage: str = "stream"  # stream | settle1 | settle2 | done
    n: int = 0
    n_action_steps: int = 0
    phase_steps: Dict[int, int] = field(default_factory=lambda: {PHASE_ORACLE: 0, PHASE_SETTLE: 0})
    last_action: Optional[th.Tensor] = None
    last_obs: Any = None
    last_info: Any = None
    verdict: Optional[Verdict] = None
    settle_verdict: Optional[Verdict] = None
    window: int = 0
    windows: List[int] = field(default_factory=list)
    window_left: int = 0
    before: Optional[MotionSnapshot] = None
    motion: Dict[str, Any] = field(default_factory=dict)
    motion_first: Optional[Dict[str, Any]] = None
    relax_info: Dict[str, Any] = field(default_factory=dict)
    stats: QAStats = field(default_factory=QAStats)
    result: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    t_stream0: float = 0.0
    t_settle0: float = 0.0
    rules: Any = None
    _hold_iter: Any = None

    @property
    def isg(self) -> Dict[str, Any]:
        return isg_of(self.last_info)

    @property
    def done(self) -> bool:
        return self.stage == "done"


class VecStreamExecutor:
    """Streams for N environments stepped together (see the module docstring).

    ``env.step(actions, n_render_iterations)`` must return per-env lists (``obs_list, rewards, terminateds, truncateds,
    infos``); ``sim`` provides ``current_time`` for the recorders (None: not recorded)."""

    def __init__(self, env, slots: List[Slot], sim=None, n_render_iterations: int = 1):
        self.env = env
        self.slots = list(slots)
        self.sim = sim
        self.n_render_iterations = int(n_render_iterations)
        for s in self.slots:
            if s.monitors is not None:
                s.stats = s.monitors.stats
            s.rules = _Rules(s.cfg, s.task)
        self.steps = 0  # batch steps
        self.action_dim = int(getattr(self.slots[0].robot, "action_dim", 23)) if self.slots else 23

    # ------------------------------------------------------------------ per-slot actions

    def _noop(self, s: Slot) -> th.Tensor:
        """The stream no-op of ``Executor._noop_actions``: the last streamed action with the base stopped (a stream that
        produced no action: the executor's hold of the current joints)."""
        if s.last_action is not None:
            hold = s.last_action.clone()
            idx = getattr(s.robot, "controller_action_idx", {}).get("base")
            if idx is not None:
                hold[idx] = 0.0
            return hold
        if s._hold_iter is None:
            s._hold_iter = _hold_actions(s.robot, 10**9)
        return th.as_tensor(next(s._hold_iter), dtype=th.float32).reshape(-1)

    # ------------------------------------------------------------------ run

    def run(self, obs_list: Iterable[Any], info_list: Iterable[Any]) -> List[Dict[str, Any]]:
        """Run every slot's stream and epilogue; returns the per-slot result dicts (``Executor.epilogue``'s)."""
        obs_list, info_list = list(obs_list), list(info_list)
        for s in self.slots:
            s.last_obs, s.last_info = obs_list[s.idx], info_list[s.idx]
            s.t_stream0 = time.time()
            if s.monitors is not None:
                s.monitors.begin_action(_StreamAction(), 0, 0, intended=(), held=s.held(), manipulation=False)
        while any(not s.done for s in self.slots):
            actions = th.zeros((len(self.slots), self.action_dim), dtype=th.float32)
            phases: Dict[int, int] = {}
            for s in self.slots:
                if s.stage == "stream":
                    a = s.source(s.last_obs, s.last_info)
                    if a is None:
                        self._end_stream(s, None)
                    else:
                        a = th.as_tensor(a, dtype=th.float32).reshape(-1)
                        s.last_action = a.clone()
                        actions[s.idx] = a
                        phases[s.idx] = PHASE_ORACLE
                        continue
                if s.stage in ("settle1", "settle2"):
                    actions[s.idx] = self._noop(s)
                    phases[s.idx] = PHASE_SETTLE
                elif s.stage == "done":
                    actions[s.idx] = self._noop(s)
            if all(s.done for s in self.slots):
                break
            obs_list, rewards, terminateds, truncateds, infos = self.env.step(actions, n_render_iterations=self.n_render_iterations)
            self.steps += 1
            sim_time = None
            if self.sim is not None:
                try:
                    sim_time = float(self.sim.current_time)
                except Exception:  # noqa: BLE001
                    pass
            for s in self.slots:
                if s.idx not in phases:
                    continue  # finished: its no-op was stepped, nothing recorded
                phase = phases[s.idx]
                s.n += 1
                s.phase_steps[phase] = s.phase_steps.get(phase, 0) + 1
                s.last_obs, s.last_info = obs_list[s.idx], infos[s.idx]
                if s.recorder is not None:
                    try:
                        rew = float(rewards[s.idx])
                    except Exception:  # noqa: BLE001
                        rew = 0.0
                    s.recorder.record(s.n, actions[s.idx], s.last_obs, s.last_info, state=None, phase=phase, action_index=0, attempt=0, reward=rew,
                                      terminated=bool(terminateds[s.idx]), truncated=bool(truncateds[s.idx]), sim_time=sim_time)
                verdict = s.monitors.check(s.n) if s.monitors is not None else None
                if verdict is None and s.n >= s.cfg.max_steps:
                    verdict = Verdict("timeout", f"episode exceeded max_steps {s.cfg.max_steps}", step=s.n)
                    s.stats.record(verdict)
                if phase == PHASE_ORACLE:
                    if verdict is not None:
                        self._end_stream(s, verdict)
                else:
                    if verdict is not None:
                        s.settle_verdict = verdict
                        self._end_window(s)
                    else:
                        s.window_left -= 1
                        if s.window_left <= 0:
                            self._end_window(s)
        return [s.result for s in self.slots]

    # ------------------------------------------------------------------ stream end -> epilogue

    def _end_stream(self, s: Slot, verdict: Optional[Verdict]) -> None:
        s.verdict = verdict
        if s.monitors is not None:
            s.monitors.end_action()
        if s.recorder is not None:
            s.recorder.commit()
        s.timings["actions_s"] = time.time() - s.t_stream0
        s.n_action_steps = s.n
        self._begin_epilogue(s)

    def _begin_epilogue(self, s: Slot) -> None:
        s.t_settle0 = time.time()
        if s.monitors is not None:
            s.monitors.begin_epilogue(held=s.held())
        isg = s.isg
        never_held = not isg.get("binary_success_ever", isg.get("binary_success", False))
        if s.never_held_window is not None and never_held:
            s.window = int(s.never_held_window)
            s._second_factor = 0.0
        else:
            s.window = max(int(s.cfg.settle_min_steps), int(math.ceil(s.cfg.settle_fraction * s.n_action_steps)))
            s._second_factor = float(s.cfg.settle_second_window_factor)
        s.before = MotionSnapshot(s.robot, s.tracked)
        s.windows = [s.window]
        s.rules._reset_predicate_branches()
        if s.verdict is None or s.verdict.category != "timeout":
            s.stage = "settle1"
            s.window_left = s.window
        else:
            s.motion = s.before.delta(MotionSnapshot(s.robot, s.tracked))
            self._finalize(s)

    def _end_window(self, s: Slot) -> None:
        if s.recorder is not None:
            s.recorder.commit()
        after = MotionSnapshot(s.robot, s.tracked)
        motion = s.before.delta(after)
        isg = s.isg
        s.rules.isg = isg
        moved = s.rules._moved(motion, isg)
        binary_final = bool(isg.get("binary_success", False))
        if s.stage == "settle1":
            s.motion = motion
            if s.verdict is None and s.settle_verdict is None and (moved or not binary_final) and s._second_factor > 0:
                window2 = int(math.ceil(s._second_factor * s.window))
                s.motion_first = motion
                s.before = MotionSnapshot(s.robot, s.tracked)
                s.rules._reset_predicate_branches()
                s.stage = "settle2"
                s.window_left = window2
                s.windows.append(window2)
                return
        else:
            motion["first_window"] = s.motion_first
            s.motion = motion
        self._finalize(s)

    # ------------------------------------------------------------------ verdict (Executor.epilogue's)

    def _finalize(self, s: Slot) -> None:
        isg = s.isg
        s.rules.isg = isg
        motion = s.motion
        moved = s.rules._moved(motion, isg)
        binary_final = bool(isg.get("binary_success", False))
        binary_ever = bool(isg.get("binary_success_ever", isg.get("binary_success", False)))
        settled_task = bool(isg.get("settled_success", False))
        binary_branch, branch_log = s.rules._predicate_branches()
        out_of_tol = _out_of_tolerance(isg)
        verdict = s.verdict
        if verdict is None and s.settle_verdict is not None:
            verdict = s.settle_verdict
        if verdict is None and not (binary_ever or binary_final):
            unsatisfied = _unsatisfied(isg)
            only_tolerance = bool(out_of_tol) and all(_pose_reached_of(c) in out_of_tol for c in unsatisfied)
            if only_tolerance:
                verdict = Verdict("tolerance", "measured error above tolerance for " + ", ".join(out_of_tol) + " (the goal never held)", step=s.n,
                                  details={"targets": out_of_tol, "never_reached": True})
            else:
                verdict = Verdict("not_reached", "binary goal not reached; unsatisfied: " + ", ".join(unsatisfied), step=s.n)
            s.stats.record(verdict)
        if verdict is None and out_of_tol:
            verdict = Verdict("tolerance", "measured error above tolerance for " + ", ".join(out_of_tol), step=s.n, details={"targets": out_of_tol})
            s.stats.record(verdict)
        if verdict is None and (moved or not binary_final or (s.cfg.require_task_settled and not settled_task)):
            if moved:
                msg = "robot / object motion after the final no-op window"
            elif not binary_final:
                msg = "the binary goal no longer holds at the last step"
            else:
                msg = "the evaluator's pose_settled atoms do not hold at the last step"
            verdict = Verdict("not_settled", msg, step=s.n, details={"motion": motion, "task_settled_success": settled_task})
            s.stats.record(verdict)
        s.timings["settle_s"] = time.time() - s.t_settle0
        s.verdict = verdict
        errors = isg.get("final_errors") or isg.get("errors") or {}
        last_errors = isg.get("errors") or {}
        s.result = {
            "success": verdict is None,
            "verdict": None if verdict is None else verdict.to_dict(),
            "reason": None if verdict is None else verdict.category,
            "binary_success_ever": binary_ever,
            "binary_success_final": binary_final,
            "binary_branch": binary_branch,
            "binary_branch_fallback": sorted(k for k, v in binary_branch.items() if v not in ("omnigibson",)),
            "binary_branch_log": branch_log,
            "support_disturbance": {},
            "articulated_root": {},
            "settled_success": settled_task,
            "pose_settled_config": isg.get("pose_settled_config"),
            "settled_success_ever": int(isg.get("first_settled_success_step", -1)) >= 0,
            "first_success_step": isg.get("first_success_step"),
            "first_settled_success_step": isg.get("first_settled_success_step"),
            "settle_latency_steps": isg.get("settle_latency_steps"),
            "errors": errors,
            "errors_final_step": last_errors,
            "conditions": isg.get("conditions"),
            "measured": isg.get("measured"),
            "group_assignments": isg.get("group_assignments"),
            "n_steps": s.n,
            "n_action_steps": s.n_action_steps,
            "n_settle_steps": s.phase_steps.get(PHASE_SETTLE, 0),
            "settle_window": s.window,
            "settle_windows": list(s.windows),
            "epilogue_relax": dict(s.relax_info),
            "kept_hold_arms": [],
            "jams": [],
            "witness_branch_accepted": [],
            "protection": {"protected": [], "plan_checks": 0, "fired": list(getattr(s.monitors, "protect_events", []) or [])},
            "demo_gating": {"enabled": []},
            "motion_delta": motion,
            "actions": [],
            "qa": s.monitors.report() if s.monitors is not None else {"stats": s.stats.to_dict()},
            "checkpoint_reports": [],
            "brace_retry": _jsonable([]),
            "goal_observation": None,
            "timings": dict(s.timings),
            "env_idx": s.idx,
            "batch_steps": self.steps,
        }
        s.stage = "done"


__all__ = ["VecStreamExecutor", "Slot", "isg_of"]
