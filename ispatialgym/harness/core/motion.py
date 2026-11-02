"""The non-learned motion layer (needs the simulator): bounded targets -> 23-D action chunks.

``prepare(request)`` validates and plans **before** anything moves (raising :class:`MotionRejected`) and returns the
:class:`MotionPlan` the tool result is written from and a generator that yields one 23-D action per control step and
returns a :class:`MotionOutcome`.

* ``move_eef`` (RoboProbe Inspect EEF): named absolute grasp-point targets for one or both arms; unnamed dimensions hold
  their measured value. cuRobo plans the joint path to the eef pose (:mod:`planning`, both arms as one unit); the path is
  resampled by its joint travel so no joint moves more than its per-step budget (its velocity limit x
  ``arm_velocity_scale`` x dt) and streamed open loop; a gripper value moves the jaw only once the arms have arrived,
  through the oracle's gripper routine padded to its planned length. A motion longer than ``max_duration_s`` is refused.
* ``move_base``: a straight base move relative to the current pose, tracked by ``Motions.track_path``.
* ``move_torso``: ``Motions.trunk_to`` (robot-level torso controller).
* ``hold``: hold actions (code agents waiting).
"""

from __future__ import annotations

import math
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.harness.config import MotionLimits
from ispatialgym.harness.core import pose
from ispatialgym.harness.core.approver import TargetApprover, parse_eef_targets
from ispatialgym.harness.core.types import MotionOutcome, MotionPlan, MotionRejected, MotionRequest
from ispatialgym.robot import conventions as C

MotionGen = Generator[th.Tensor, None, MotionOutcome]
NpPose = Tuple[np.ndarray, np.ndarray]

#: gripper targets at or beyond these use the full close (assisted grasp) / full open (release) routines
CLOSED, OPEN = 0.02, 0.98


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v) + "]"


def joint_travel(path: th.Tensor, inv_limit: np.ndarray) -> np.ndarray:
    """Cumulative travel along ``path`` in units of the per-step budget: each segment counts its largest
    ``|dq_j| * inv_limit_j`` (0 for joints that are not paced)."""
    p = path.detach().cpu().double().numpy()
    seg = (np.abs(np.diff(p, axis=0)) * inv_limit[None, :]).max(axis=1) if len(p) > 1 else np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_path(path: th.Tensor, steps: int, inv_limit: np.ndarray) -> th.Tensor:
    """``steps`` waypoints along ``path`` evenly spaced in :func:`joint_travel`, ending on its last point, so no step moves
    a paced joint by more than its budget once ``steps >= travel``. The first point is where the arm already is, so
    sampling starts one interval in."""
    p = path.detach().cpu().double().numpy()
    s = joint_travel(path, inv_limit)
    src = s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(p))
    dst = np.linspace(0.0, 1.0, int(steps) + 1)[1:]
    return th.as_tensor(np.stack([np.interp(dst, src, p[:, j]) for j in range(p.shape[1])], axis=1), dtype=th.float32)


class MotionLayer:
    def __init__(self, emb, planner, limits: MotionLimits, approver: TargetApprover):
        self.emb = emb
        self.planner = planner
        self.limits = limits
        self.approver = approver
        self._step_limits: Dict[str, np.ndarray] = {}

    @property
    def max_steps(self) -> int:
        return max(1, int(math.ceil(self.limits.max_duration_s / self.emb.dt)))

    def step_limits(self, arm: str) -> np.ndarray:
        if arm not in self._step_limits:
            self._step_limits[arm] = self.emb.arm_step_limits(arm, self.limits.arm_velocity_scale)
        return self._step_limits[arm]

    def prepare(self, req: MotionRequest) -> Tuple[MotionPlan, MotionGen]:
        p = req.params
        if req.kind == "move_eef":
            return self._prepare_eef(p)
        if req.kind == "move_base":
            return self._prepare_base(p)
        if req.kind == "move_torso":
            return self._prepare_torso(p)
        if req.kind == "hold":
            n = max(1, int(p.get("steps", 30)))
            return MotionPlan("hold", n, self.emb.dt), self._run_hold(n)
        raise MotionRejected(f"unknown motion {req.kind!r}")

    # -- move_eef -------------------------------------------------------------------------------------------------
    def _prepare_eef(self, p) -> Tuple[MotionPlan, MotionGen]:
        by_arm, error = parse_eef_targets(p.get("targets"))
        if error:
            raise MotionRejected(error)
        assert by_arm is not None
        emb = self.emb
        jitter = np.asarray(p.get("jitter_m") or (0.0, 0.0, 0.0), dtype=float)
        requested: Dict[str, Tuple[Dict[str, float], NpPose]] = {}
        executed: Dict[str, NpPose] = {}
        for arm in sorted(by_arm):
            axes = {k: v for k, v in by_arm[arm].items() if k != "gripper"}
            if not axes:
                continue
            measured = emb.grasp_values(arm)
            merged = {**measured, **axes}
            requested[arm] = (merged, pose.values_to_pose(merged))
            exe = dict(merged, x=merged["x"] + jitter[0], y=merged["y"] + jitter[1], z=merged["z"] + jitter[2])
            exe_pose = pose.values_to_pose(exe)
            cur_pos, cur_quat = pose.values_to_pose(measured)
            if np.linalg.norm(exe_pose[0] - cur_pos) > 1e-3 or pose.orientation_error_deg(exe_pose[1], cur_quat) > 0.1:
                executed[arm] = exe_pose
        path = None
        inv_limit = np.zeros(len(emb.q_full()))
        if executed:
            targets = {arm: emb.to_world(tuple(th.as_tensor(x, dtype=th.float32) for x in target)) for arm, target in executed.items()}
            path, status = self.planner.plan(targets, emb.backend.q_full())
            if path is None:
                who = "left and right arm poses" if len(executed) == 2 else f"{next(iter(executed))} arm pose"
                raise MotionRejected(f"{who} unreachable (planner status: {status}). The robot did not move; choose another pose. Neither arm moved.",
                                     repairable=False, detail=getattr(self.planner, "last_detail", None))
            for arm in executed:
                inv_limit[[int(i) for i in emb.robot.arm_control_idx[arm]]] = 1.0 / self.step_limits(arm)
        arm_steps = int(math.ceil(float(joint_travel(path, inv_limit)[-1]) - 1e-9)) if path is not None else 0
        grips = {}
        for arm, axes in by_arm.items():
            if axes.get("gripper") is not None:
                n = self._gripper_steps(arm, float(axes["gripper"]))
                if n:
                    grips[arm] = (float(axes["gripper"]), n)
        gripper_steps = max((n for _, n in grips.values()), default=0)
        steps = arm_steps + gripper_steps
        if steps > self.max_steps:
            raise MotionRejected(f"this motion needs {steps * emb.dt:.1f}s at the safe speed, over the {self.limits.max_duration_s:g}s playout cap; "
                                 "split it into smaller motions")
        q_traj = resample_path(path, arm_steps, inv_limit) if arm_steps else None
        plan = MotionPlan("move_eef", max(steps, 1), emb.dt, gripper_steps=gripper_steps, reports="grasp point" if requested else None)
        return plan, self._run_eef(q_traj, sorted(executed), grips, gripper_steps, requested, hold=steps == 0)

    def _gripper_steps(self, arm: str, target: float) -> int:
        """Control steps of the jaw phase for ``target`` (0 when the jaw is already there)."""
        m = self.emb.motions
        o = self.emb.gripper_opening(arm)
        wait = int(self.limits.gripper_wait_steps)
        if target <= CLOSED:
            if o <= CLOSED:
                return 0
            rate = m.cfg.gripper_close_rate
            lo, _ = m._gripper_limits(arm)
            q_now = float(self.emb.robot.get_joint_positions()[self.emb.robot.gripper_control_idx[arm]].float().mean())
            ramp = int(math.ceil(max(0.0, q_now - lo) / float(rate))) if rate else 0
            return ramp + 2 * wait + 5  # ramp, contact hold, full-close fallback, settle
        if target >= OPEN:
            return 0 if o >= OPEN else int(m.cfg.unload_steps) + wait + 5
        return 0 if abs(target - o) < 0.02 else int(self.limits.gripper_partial_steps)

    def _gripper_gen(self, arm: str, target: float) -> Iterator[th.Tensor]:
        m = self.emb.motions
        wait = int(self.limits.gripper_wait_steps)
        if target <= CLOSED:
            return m.gripper(arm, True, steps=wait, rate=m.cfg.gripper_close_rate)
        if target >= OPEN:
            return m.gripper(arm, False, steps=wait)
        lo, hi = m._gripper_limits(arm)
        return m.gripper_opening(arm, lo + target * (hi - lo), steps=int(self.limits.gripper_partial_steps))

    def _zip(self, gens: List[Iterator[th.Tensor]], steps: int) -> Iterator[th.Tensor]:
        """Run per-arm generators side by side, one control step each per action, then hold to exactly ``steps``. Every
        generator writes its arm's command into the shared action state, so the action of the last one carries all."""
        live, n = list(gens), 0
        while live and n < steps:
            a = None
            for g in list(live):
                try:
                    a = next(g)
                except StopIteration:
                    live.remove(g)
            if a is None:
                break
            n += 1
            yield a
        if n < steps:
            yield from self.emb.motions.hold(steps - n)

    def _run_eef(self, q_traj: Optional[th.Tensor], arms: List[str], grips, gripper_steps: int, requested, hold: bool) -> MotionGen:
        from ispatialgym.oracle.motions import SkillError

        m = self.emb.motions
        stopped = None
        n = 0
        try:
            if q_traj is not None:
                for a in m.execute_joint_trajectory(q_traj, arms=arms, converge=False):
                    n += 1
                    yield a
            if grips:
                for a in self._zip([self._gripper_gen(arm, target) for arm, (target, _) in sorted(grips.items())], gripper_steps):
                    n += 1
                    yield a
            if hold and n == 0:
                yield from m.hold(1)
                n = 1
        except SkillError as e:
            stopped = str(e)
        arrival = {}
        for arm, (values, (_, req_quat)) in requested.items():
            got = self.emb.grasp_values(arm)
            _, got_quat = pose.values_to_pose(got)
            err = float(np.linalg.norm([got[k] - values[k] for k in pose.POSITION_AXES]))
            arrival[arm] = {"pos_mm": round(1000.0 * err, 1), "rot_deg": round(pose.orientation_error_deg(got_quat, req_quat), 2),
                            "requested": {k: round(v, 4) for k, v in values.items()}, "reached": {k: round(v, 4) for k, v in got.items()}}
        details = {"arrival": {"kind": "move_eef", "arms": arrival}, "grippers": {arm: round(self.emb.gripper_opening(arm), 3) for arm in grips}}
        parts = [f"{arm} {v['pos_mm']:.0f} mm and {v['rot_deg']:.1f} deg from its target" for arm, v in sorted(arrival.items())]
        parts += [f"{arm} gripper opening {self.emb.gripper_opening(arm):.2f}" for arm in sorted(grips)]
        msg = "; ".join(parts) or "held"
        if stopped:
            return MotionOutcome(False, f"motion stopped ({stopped}); {msg}", details=details)
        return MotionOutcome(True, msg, details=details)

    def _run_hold(self, steps: int) -> MotionGen:
        yield from self.emb.motions.hold(max(1, steps))
        return MotionOutcome(True, f"held for {steps} control steps")

    # -- move_torso -----------------------------------------------------------------------------------------------
    def _prepare_torso(self, p) -> Tuple[MotionPlan, MotionGen]:
        tr = None if p.get("translate") is None else float(p["translate"])
        td = None if p.get("tilt_deg") is None else float(p["tilt_deg"])
        self.approver.check_torso(tr, td)
        m = self.emb.motions
        t0, k0 = self.emb.torso()
        d_t = 0.0 if tr is None else abs(tr - t0)
        d_k = 0.0 if td is None else abs(math.radians(td) - k0)
        secs = max(d_t / m.cfg.trunk_translate_rate, d_k / m.cfg.trunk_tilt_rate)
        plan = MotionPlan("move_torso", max(1, int(math.ceil(secs / self.emb.dt))), self.emb.dt, approximate=True, reports="torso")
        return plan, self._run_torso(tr, td)

    def _run_torso(self, translate, tilt_deg) -> MotionGen:
        from ispatialgym.oracle.motions import SkillError

        stopped = None
        try:
            yield from self.emb.motions.trunk_to(translate=translate, tilt=None if tilt_deg is None else math.radians(tilt_deg))
        except SkillError as e:
            stopped = str(e)
        t, k = self.emb.torso()
        arrival = {"kind": "move_torso", "translate": None if translate is None else round(abs(t - translate), 3),
                   "tilt_deg": None if tilt_deg is None else round(abs(math.degrees(k) - tilt_deg), 2)}
        msg = f"torso at translate {t:.2f}, tilt {math.degrees(k):+.1f} deg"
        return MotionOutcome(stopped is None, msg + (f"; stopped: {stopped}" if stopped else ""), details={"arrival": arrival})

    # -- move_base ------------------------------------------------------------------------------------------------
    def _prepare_base(self, p) -> Tuple[MotionPlan, MotionGen]:
        f, l, t = float(p.get("forward_m") or 0.0), float(p.get("left_m") or 0.0), float(p.get("turn_deg") or 0.0)
        self.approver.check_base(f, l, t)
        m = self.emb.motions
        pos, quat = self.emb.base_pose_world()
        yaw = C.yaw_of(quat)
        wp = [float(pos[0]) + math.cos(yaw) * f - math.sin(yaw) * l, float(pos[1]) + math.sin(yaw) * f + math.cos(yaw) * l,
              C.wrap_angle(yaw + math.radians(t))]
        cfg = m.cfg.base_tracker
        v_lin = m.cfg.path_speed_fraction * min(cfg.v_max[0], cfg.v_max[1])
        v_ang = m.cfg.path_speed_fraction * cfg.v_max[2]
        ref = m._base_reference([wp], v_lin, v_ang, a_max=min(cfg.a_max[0], cfg.a_max[1]))
        plan = MotionPlan("move_base", len(ref) + 3, self.emb.dt, approximate=True, reports="base")
        return plan, self._run_base((f, l, t), wp, (pos, quat))

    def _run_base(self, req, wp, start) -> MotionGen:
        from ispatialgym.oracle.motions import SkillError

        stopped = None
        try:
            yield from self.emb.motions.track_path([wp])
        except SkillError as e:
            stopped = str(e)
        p, q = C.relative(self.emb.base_pose_world(), start)
        got = (float(p[0]), float(p[1]), math.degrees(C.yaw_of(q)))
        err = math.hypot(got[0] - req[0], got[1] - req[1])
        dyaw = abs(math.degrees(C.wrap_angle(math.radians(got[2] - req[2]))))
        arrival = {"kind": "move_base", "pos_mm": round(1000.0 * err, 1), "yaw_deg": round(dyaw, 2), "moved": got, "requested": list(req)}
        msg = f"base moved {got[0]:+.2f} m forward, {got[1]:+.2f} m left and turned {got[2]:+.1f} deg (requested {_fmt(req)})"
        return MotionOutcome(stopped is None, msg + (f"; stopped early: {stopped}" if stopped else ""), details={"arrival": arrival})
