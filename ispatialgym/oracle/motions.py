"""Layer 1 of the v2 oracle: decision-free motions (``oracle_primitives_design.md`` §3.2, §3.4; TODO F2).

Eight motions, each a generator of 23-D JoyLo actions (one per environment step) that receives **one fully specified
target** and either reaches it or raises :class:`SkillError` with a QA category of §6.4:

==================================  ==========================================================================
``track_path(waypoints)``           the v1 base tracker follows a recorded ``[x, y, yaw]`` waypoint list
``trunk_to(translate, tilt)``       the trunk along the JoyLo keyframe curve at bounded rates, both scalars
                                    interpolated jointly (the sweep the sampler checked, conventions
                                    ``trunk_sweep``)
``arm_to(arms, pose | q)``          cuRobo trajopt with the trunk locked — the one planner left at run time
                                    (§3.9); ``plan_failed`` when no trajectory is found
``arm_along(arm, waypoints)``       differential IK through a **given** eef waypoint list (densified to the
                                    tracking resolution, retimed to the Cartesian speed); no collision veto —
                                    contact motions must touch, the QA collision monitor judges contacts
``gripper(arm, close)``             ±1 until the fingers stop or the assisted grasp engages
``base_follow(base_poses, arm)``    the arm locked at its current joints, the base tracks the recorded ``b_k``
``hold(steps)``                     zero base velocity, current targets
``settle()``                        hold until every joint velocity is below a threshold
==================================  ==========================================================================

What was cut from v1 ``skills.Skills`` (kept as the record of what works): every decision — stance search
(``stance_for`` / ``approach`` / ``go_to_stance``), grasp candidates (``grasp_candidates`` / ``grasp_pose_for``),
the base planner and its trav-map / straight-line cascade (``plan_base`` / ``navigate_to``), the swept trunk check
and its ``ready_arms`` fallback, ``torso=True`` embodiments, the reposition / planned-descent fallbacks of ``place``,
the composites (``grasp`` / ``place`` / ``push`` / ``brace`` — now Layer-2 actions in ``oracle/execute.py``) and the
``pose`` / ``delta_pos`` forms of ``move_eef_cartesian`` (the conventions produce the waypoint lists). What was kept
verbatim in spirit: ``track_base`` / ``_wait_converged`` / ``execute_joint_trajectory`` / ``_retime``, the
``set_torso`` rate limiting, ``move_eef``'s trajopt with the trunk locked, ``move_eef_cartesian``'s differential IK
and time parameterisation, ``set_gripper``, ``hold``, ``settle``.

Executor state (§3.8) lives on the instance and is exported / imported for the checkpoints: :class:`ActionState`
hold targets and gripper commands, :class:`TrunkState`, the last tracker's reference index. The cuRobo backend's
attached objects are the executor's too (``backend.attach_object`` / ``detach_object`` are called by the actions).

This module imports no simulator: ``env`` / ``planner`` are duck-typed (an OmniGibson ``Environment`` and an
:class:`~ispatialgym.oracle.curobo_planner.R1ProCuRobo`), so the executor tests run it against fakes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.action import ActionState
from ispatialgym.oracle.base_tracker import BaseTrackerConfig, BaseTrajectoryTracker
from ispatialgym.oracle.trunk import TrunkState
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)

Pose = Tuple[th.Tensor, th.Tensor]

#: QA categories of §6.4 a motion may raise (the executor maps everything else to ``execution``).
MOTION_CATEGORIES = ("plan_failed", "execution", "timeout", "failed_grasp", "collision")


class SkillError(RuntimeError):
    """A motion could not be planned or did not converge. ``reason`` is a QA category of §6.4."""

    def __init__(self, reason: str, message: str = "", **details):
        super().__init__(f"[{reason}] {message}" + (f" {details}" if details else ""))
        self.reason = reason
        self.message = message
        self.details = details


@dataclass
class MotionsConfig:
    #: joint-space convergence tolerance for streamed trajectories (rad) and the loose bound that fails the motion
    joint_tol: float = 0.02
    joint_tol_loose: float = 0.08
    #: how long to keep holding the final waypoint while waiting for convergence (steps)
    settle_steps: int = 90
    #: eef speed of ``arm_along`` (m/s, rad/s) and the densification resolution of its waypoint lists (m, rad)
    cartesian_speed: float = 0.10
    cartesian_ang_speed: float = 0.6
    cartesian_step: float = 0.01
    cartesian_rot_step: float = 0.05
    #: differential-IK tolerances (see ``R1ProCuRobo.solve_ik_path``)
    diff_ik_pos_tol: float = 2e-3
    diff_ik_rot_tol: float = 1e-2
    #: gripper actuation wait (steps)
    gripper_steps: int = 60
    #: base tracking (paths) and base-follow (articulation) tracker configurations
    base_tracker: BaseTrackerConfig = field(default_factory=BaseTrackerConfig)
    base_follow_speed: float = 0.15  # m/s along the recorded b_k
    base_follow_yaw_rate: float = 0.4  # rad/s
    #: fraction of the tracker's v_max used to time-parameterise a recorded path
    path_speed_fraction: float = 0.8
    #: JoyLo trunk rates (translate units / s, rad / s); JoyLo defaults are 0.1 / 0.05, faster here for shorter demos
    trunk_translate_rate: float = 0.25
    trunk_tilt_rate: float = 0.15
    #: torso more than this off the keyframe curve is ramped onto it first (rad), at ``trunk_ramp_rate`` (rad / s)
    trunk_oncurve_tol: float = 0.05
    trunk_ramp_rate: float = 0.5
    #: trajopt attempts of ``arm_to``
    plan_attempts: int = 4
    #: step budgets (a motion exceeding one raises ``timeout``)
    trunk_max_steps: int = 2000
    trajectory_max_steps: int = 6000
    settle_max_steps: int = 120
    settle_vel_tol: float = 0.02


# ==========================================================================================
# pose helpers
# ==========================================================================================


def as_world_pose(x: Any) -> Pose:
    """``(pos (3,), quat_xyzw (4,))`` float32 CPU tensors from a sidecar ``Pose`` (``position`` / ``orientation``), a
    conventions ``FramedPose`` (``pos`` / ``quat``), a ``{"position", "orientation"}`` dict, a 7-vector or a
    ``(pos, quat)`` pair."""
    if hasattr(x, "tensors") and hasattr(x, "position"):
        return x.tensors()
    if hasattr(x, "pos") and hasattr(x, "quat"):
        return C.as_tensor(x.pos), C.as_tensor(x.quat)
    if hasattr(x, "position") and hasattr(x, "orientation"):
        return C.as_tensor(x.position), C.as_tensor(x.orientation)
    if isinstance(x, dict):
        return C.as_tensor(x["position"]), C.as_tensor(x["orientation"])
    if isinstance(x, (tuple, list)) and len(x) == 2 and not isinstance(x[0], (int, float)):
        return C.as_tensor(x[0]), C.as_tensor(x[1])
    flat = C.as_tensor(x).reshape(-1)
    if flat.numel() == 7:
        return flat[:3].clone(), flat[3:].clone()
    raise TypeError(f"cannot interpret {type(x).__name__} as a world pose")


def as_pose2d(x: Any) -> Tuple[float, float, float]:
    v = [float(a) for a in C.as_tensor(x).reshape(-1).tolist()]
    if len(v) != 3:
        raise TypeError(f"expected [x, y, yaw], got {len(v)} values")
    return v[0], v[1], v[2]


# ==========================================================================================
# Motions
# ==========================================================================================


class Motions:
    """The eight Layer-1 motions bound to one R1Pro (``env.robots[0]``) and one cuRobo backend."""

    def __init__(self, env, planner, cfg: Optional[MotionsConfig] = None, robot=None):
        self.env = env
        self.robot = robot if robot is not None else env.robots[0]
        self.backend = planner
        self.cfg = cfg or MotionsConfig()
        self.dt = float(getattr(planner, "control_dt", 1.0 / 30.0))
        self.state = ActionState(self.robot)
        self._joint_limits = (
            self.robot.joint_lower_limits[self.robot.trunk_control_idx],
            self.robot.joint_upper_limits[self.robot.trunk_control_idx],
        )
        self.trunk = TrunkState.from_q(
            self.robot.get_joint_positions()[self.robot.trunk_control_idx],
            translate_rate=self.cfg.trunk_translate_rate,
            tilt_rate=self.cfg.trunk_tilt_rate,
        )
        self.tracker: Optional[BaseTrajectoryTracker] = None
        #: steps emitted by the last motion (diagnostics)
        self.last_steps = 0

    # ------------------------------------------------------------------ privileged reads (state, never decisions)

    def q_full(self) -> th.Tensor:
        return self.robot.get_joint_positions().clone().float()

    def eef_pose(self, arm: str) -> Pose:
        pos, quat = self.robot.eef_links[arm].get_position_orientation()
        return pos.float().clone(), quat.float().clone()

    def base_pose2d(self) -> Tuple[float, float, float]:
        pos, quat = self.robot.get_position_orientation()
        return float(pos[0]), float(pos[1]), C.yaw_of(quat.float())

    def held_object(self, arm: str):
        return self.robot._ag_obj_in_hand.get(arm)

    def gripper_closed(self, arm: str) -> bool:
        return self.state.gripper_cmd[arm] < 0.0

    # ------------------------------------------------------------------ executor state (checkpoints, §3.8)

    def export_state(self) -> Dict[str, Any]:
        """Hold targets, gripper commands, trunk scalars and the tracker's reference index (plain tensors / numbers)."""
        return {
            "trunk_target": self.state.trunk_target.clone(),
            "arm_target": {arm: q.clone() for arm, q in self.state.arm_target.items()},
            "gripper_cmd": dict(self.state.gripper_cmd),
            "trunk": (float(self.trunk.translate), float(self.trunk.tilt)),
            "tracker_index": None if self.tracker is None else int(self.tracker.k),
        }

    def import_state(self, d: Dict[str, Any]) -> None:
        self.state.trunk_target = d["trunk_target"].clone()
        self.state.arm_target = {arm: q.clone() for arm, q in d["arm_target"].items()}
        self.state.gripper_cmd = dict(d["gripper_cmd"])
        self.state.stop_base()
        t, k = d["trunk"]
        self.trunk = TrunkState(float(t), float(k), translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        self.tracker = None

    # ------------------------------------------------------------------ emitters

    def _emit(self) -> th.Tensor:
        self.last_steps += 1
        return self.state.action()

    def hold(self, steps: int = 1) -> Iterator[th.Tensor]:
        """Emit hold actions (zero base velocity, current targets)."""
        self.state.stop_base()
        for _ in range(int(steps)):
            yield self._emit()

    def settle(self, max_steps: Optional[int] = None, vel_tol: Optional[float] = None) -> Iterator[th.Tensor]:
        """Hold until every joint velocity is below ``vel_tol`` (or ``max_steps`` elapsed; never raises)."""
        self.state.stop_base()
        max_steps = self.cfg.settle_max_steps if max_steps is None else int(max_steps)
        vel_tol = self.cfg.settle_vel_tol if vel_tol is None else float(vel_tol)
        for _ in range(max_steps):
            yield self._emit()
            if float(self.robot.get_joint_velocities().abs().max()) < vel_tol:
                return

    def _wait_converged(self, arms: Sequence[str], trunk: bool, tol: float, max_steps: int) -> Iterator[th.Tensor]:
        def err() -> float:
            return max([self.state.arm_error(a) for a in arms] + ([self.state.trunk_error()] if trunk else [0.0]))

        for _ in range(max_steps):
            if err() < tol:
                return
            yield self._emit()
        e = err()
        if e > self.cfg.joint_tol_loose:
            raise SkillError("execution", f"joints did not converge (max err {e:.3f} rad)", arms=list(arms), trunk=trunk, err_rad=e)

    # ------------------------------------------------------------------ gripper

    def gripper(self, arm: str, close: bool, steps: Optional[int] = None) -> Iterator[th.Tensor]:
        """Open / close the gripper and wait until the fingers stop moving (or an assisted grasp engages)."""
        self.state.set_gripper(arm, close)
        self.state.stop_base()
        steps = self.cfg.gripper_steps if steps is None else int(steps)
        idx = self.robot.gripper_control_idx[arm]
        for i in range(steps):
            yield self._emit()
            if i > 10:
                if close and self.held_object(arm) is not None:
                    break
                if float(self.robot.get_joint_velocities()[idx].abs().max()) < 1e-3:
                    break
        yield from self.hold(5)

    # ------------------------------------------------------------------ trunk

    def trunk_to(self, translate: Optional[float] = None, tilt: Optional[float] = None) -> Iterator[th.Tensor]:
        """Move the trunk to the JoyLo pair along the keyframe curve, both scalars at bounded rates and finishing
        together (the straight (translate, tilt) sweep of conventions ``trunk_sweep`` / spec §3.8). A torso that sits
        off the curve (a camera-relocalization init) is first ramped onto it. No collision check, no fallback."""
        idx = self.robot.trunk_control_idx
        q_now = self.robot.get_joint_positions()[idx].clone().float()
        ts = TrunkState.from_q(q_now, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        self.state.stop_base()
        # 1) onto the curve
        q_curve = ts.q(self._joint_limits)
        dev = float(th.max(th.abs(q_curve - q_now)))
        if dev > self.cfg.trunk_oncurve_tol:
            n = max(1, int(math.ceil(dev / (self.cfg.trunk_ramp_rate * self.dt))))
            log.info("trunk_to: torso %.2f rad off the JoyLo curve; ramping onto it over %d steps", dev, n)
            for k in range(1, n + 1):
                self.state.trunk_target = q_now + (q_curve - q_now) * (k / n)
                yield self._emit()
        # 2) joint interpolation: scale the rates so both scalars arrive together (never faster than configured)
        lo, hi = M.TRUNK_TRANSLATE_RANGE
        t_goal = ts.translate if translate is None else min(max(float(translate), lo), hi)
        k_goal = ts.tilt if tilt is None else float(tilt)
        d_t, d_k = abs(t_goal - ts.translate), abs(k_goal - ts.tilt)
        T = max(d_t / self.cfg.trunk_translate_rate, d_k / self.cfg.trunk_tilt_rate)
        if T > 1e-9:
            ts.translate_rate = max(d_t / T, 1e-6)
            ts.tilt_rate = max(d_k / T, 1e-6)
        for _ in range(self.cfg.trunk_max_steps):
            done = ts.step_towards(t_goal, k_goal, self.dt)
            self.state.trunk_target = ts.q(self._joint_limits)
            yield self._emit()
            if done:
                break
        else:
            raise SkillError("timeout", "trunk_to did not finish", translate=t_goal, tilt=k_goal)
        self.trunk = TrunkState(ts.translate, ts.tilt, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        yield from self._wait_converged([], trunk=True, tol=self.cfg.joint_tol, max_steps=self.cfg.settle_steps)

    # ------------------------------------------------------------------ joint trajectory streaming

    def execute_joint_trajectory(self, q_traj: th.Tensor, arms: Sequence[str], trunk: bool = False, tol: Optional[float] = None) -> Iterator[th.Tensor]:
        """Stream a ``(T, 28)`` joint trajectory (one waypoint per control step) for the given groups, then hold the
        last waypoint until the joints converge."""
        q_traj = C.as_tensor(q_traj)
        if len(q_traj) > self.cfg.trajectory_max_steps:
            raise SkillError("timeout", f"trajectory of {len(q_traj)} steps exceeds the budget {self.cfg.trajectory_max_steps}")
        self.state.stop_base()
        for q in q_traj:
            self.state.set_from_q(q, arms=arms, trunk=trunk)
            yield self._emit()
        yield from self._wait_converged(arms, trunk, tol or self.cfg.joint_tol, self.cfg.settle_steps)
        if trunk:
            self.trunk = TrunkState.from_q(self.state.trunk_target, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)

    @staticmethod
    def _retime(q_traj: th.Tensor, steps: Sequence[int]) -> th.Tensor:
        """Linearly interpolate consecutive waypoints so segment ``i`` takes ``steps[i]`` control steps."""
        out = [q_traj[0]]
        for a, b, n in zip(q_traj[:-1], q_traj[1:], steps):
            n = max(1, int(n))
            for k in range(1, n + 1):
                out.append(a + (b - a) * (k / n))
        return th.stack(out)

    # ------------------------------------------------------------------ arm: trajopt (§3.9)

    @staticmethod
    def _emb_for(arms: Sequence[str]) -> str:
        return "both_arms" if len(arms) == 2 else f"{arms[0]}_arm"

    def arm_to(self, arms: Union[str, Sequence[str]], pose: Any = None, q: Any = None, ignore_objects=()) -> Iterator[th.Tensor]:
        """cuRobo trajopt of the arm(s) with the trunk locked, then streamed. Exactly one of ``pose`` (a world pose
        for one arm, or ``{arm: pose}``) or ``q`` (a 28-D vector, ``{joint_name: value}`` or ``{arm: q7}``) is the
        target. Raises ``plan_failed`` when no trajectory is found (the sampler proved the key poses, not the
        trajectory between them, §3.9)."""
        arms = (arms,) if isinstance(arms, str) else tuple(arms)
        emb = self._emb_for(arms)
        self.backend.update_world(ignore_objects=ignore_objects)
        q_start = self.backend.q_full()
        if (pose is None) == (q is None):
            raise ValueError("arm_to needs exactly one of pose / q")
        if pose is not None:
            poses = pose if isinstance(pose, dict) and set(pose) <= set(M.ARMS) else {arms[0]: pose}
            targets = {M.EEF_LINK[a]: as_world_pose(p) for a, p in poses.items()}
            res = self.backend.plan_pose(emb, targets, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            what = "pose"
        else:
            q_goal = q_start.clone()
            if isinstance(q, dict) and set(q) <= set(M.ARMS):
                for a, q7 in q.items():
                    q_goal[self.robot.arm_control_idx[a]] = C.as_tensor(q7)
            elif isinstance(q, dict):
                for name, v in q.items():
                    q_goal[self.backend._jidx[name]] = float(v)
            else:
                q28 = C.as_tensor(q).reshape(-1)
                for a in arms:
                    q_goal[self.robot.arm_control_idx[a]] = q28[self.robot.arm_control_idx[a]]
            res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            what = "joint"
        if not res.success:
            raise SkillError("plan_failed", f"arm_to({','.join(arms)}, {what}): {res.status}", diagnostics=dict(res.diagnostics), stage="trajopt")
        log.info("arm_to(%s): %d waypoints (%.1fs), planned in %.2fs", ",".join(arms), len(res.q_traj), res.duration, res.plan_time)
        yield from self.execute_joint_trajectory(res.q_traj, arms=arms)

    # ------------------------------------------------------------------ arm: differential IK along waypoints

    def _densify(self, start: Pose, waypoints: Sequence[Any]) -> List[Pose]:
        """Straight segments between the given waypoints (positions linear, orientations SLERPed) at the tracking
        resolution; the given waypoints are kept as segment ends."""
        dense: List[Pose] = []
        prev = start
        for wp in waypoints:
            p1, q1 = as_world_pose(wp)
            dist = float(th.linalg.norm(p1 - prev[0]))
            ang = C.quat_angle(prev[1], q1)
            n = max(1, int(math.ceil(max(dist / self.cfg.cartesian_step, ang / self.cfg.cartesian_rot_step))))
            dense.extend(C.line_waypoints(prev, (p1, q1), n))
            prev = (p1, q1)
        return dense

    def arm_along(self, arm: str, waypoints: Sequence[Any], speed: Optional[float] = None) -> Iterator[th.Tensor]:
        """Track a given eef waypoint list with damped-least-squares differential IK on the arm's joints (trunk and
        base locked), streamed at ``speed`` (m/s; ``cartesian_speed``). The list is densified to the tracking
        resolution and every waypoint is reached in order; no collision veto (contact motions must touch — the QA
        collision monitor judges contacts). Raises ``plan_failed`` (stage ``diff_ik``) when a waypoint cannot be
        reached from the current configuration."""
        waypoints = list(waypoints)
        if not waypoints:
            return
        start = self.eef_pose(arm)
        dense = self._densify(start, waypoints)
        frame = M.EEF_LINK[arm]
        res = self.backend.solve_ik_path(
            f"{arm}_arm",
            [{frame: wp} for wp in dense],
            pos_tol=self.cfg.diff_ik_pos_tol,
            rot_tol=self.cfg.diff_ik_rot_tol,
            check_collisions=False,
            sync_world=False,
        )
        if not res.success:
            raise SkillError("plan_failed", f"arm_along({arm}): {res.status}", stage="diff_ik", n_waypoints=len(waypoints))
        speed = self.cfg.cartesian_speed if speed is None else float(speed)
        steps = []
        prev = start
        for wp in dense:
            seg = float(th.linalg.norm(wp[0] - prev[0]))
            ang = C.quat_angle(prev[1], wp[1])
            steps.append(max(1, int(round(max(seg / (speed * self.dt), ang / (self.cfg.cartesian_ang_speed * self.dt))))))
            prev = wp
        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps), arms=[arm])

    # ------------------------------------------------------------------ base

    def _base_reference(self, waypoints: Sequence[Any], v_lin: float, v_ang: float) -> th.Tensor:
        """Time-parameterise the polyline current pose → waypoints at ``v_lin`` (arc length) with the yaw blended
        linearly along every segment toward that waypoint's recorded yaw (``v_ang`` bounds pure rotations)."""
        pts = [self.base_pose2d()] + [as_pose2d(w) for w in waypoints]
        xy = th.tensor([[p[0], p[1]] for p in pts], dtype=th.float32)
        yaws = [pts[0][2]]
        for p in pts[1:]:
            yaws.append(yaws[-1] + C.wrap_angle(p[2] - yaws[-1]))
        seg = th.linalg.norm(xy[1:] - xy[:-1], dim=1)
        dyaw = th.tensor([abs(yaws[i + 1] - yaws[i]) for i in range(len(seg))], dtype=th.float32)
        # per-segment duration: the slower of translation and rotation
        dur = th.maximum(seg / v_lin, dyaw / v_ang)
        total = float(dur.sum())
        n = max(2, int(math.ceil(total / self.dt)) + 1)
        cum = th.cat([th.zeros(1), th.cumsum(dur, 0)])
        out = []
        for ti in th.linspace(0.0, total, n).tolist():
            k = int(th.searchsorted(cum, th.tensor(ti), right=True)) - 1
            k = min(max(k, 0), len(seg) - 1)
            f = 0.0 if float(dur[k]) < 1e-9 else min(1.0, (ti - float(cum[k])) / float(dur[k]))
            p = xy[k] + (xy[k + 1] - xy[k]) * f
            out.append([float(p[0]), float(p[1]), yaws[k] + (yaws[k + 1] - yaws[k]) * f])
        out[-1] = [float(xy[-1][0]), float(xy[-1][1]), yaws[-1]]
        return th.tensor(out, dtype=th.float32)

    def track_path(self, waypoints: Sequence[Any], posture: Optional[str] = None, tracker_cfg: Optional[BaseTrackerConfig] = None) -> Iterator[th.Tensor]:
        """Follow a recorded ``[x, y, yaw]`` waypoint list with the joystick-like tracker (feed-forward + pose
        feedback, velocity / acceleration limited, smoothed; translation and rotation simultaneous). ``posture`` is the
        recorded navigation posture (informational: Layer 1 changes no posture). Arm and trunk hold targets stay as
        they are. Raises ``execution`` when the tracker times out."""
        waypoints = list(waypoints)
        if not waypoints:
            return
        cfg = tracker_cfg or self.cfg.base_tracker
        v_lin = self.cfg.path_speed_fraction * min(cfg.v_max[0], cfg.v_max[1])
        v_ang = self.cfg.path_speed_fraction * cfg.v_max[2]
        ref = self._base_reference(waypoints, v_lin, v_ang)
        tracker = BaseTrajectoryTracker(ref, self.dt, cfg)
        self.tracker = tracker
        try:
            while True:
                v, done = tracker.step(self.base_pose2d())
                self.state.set_base_velocity(float(v[0]), float(v[1]), float(v[2]))
                yield self._emit()
                if done:
                    break
                if tracker.timed_out:
                    d, dyaw = tracker.error(self.base_pose2d())
                    raise SkillError("execution", f"base tracking timed out (pos err {d:.3f} m, yaw err {dyaw:.3f} rad)", pos_err_m=d, yaw_err_rad=dyaw, posture=posture)
        finally:
            self.state.stop_base()
        yield from self.hold(3)

    def base_follow(self, base_poses: Sequence[Any], arm: str, speed: Optional[float] = None) -> Iterator[th.Tensor]:
        """Articulation with the handle grasped: lock ``arm`` at its current joints (the grasp solution q*) and track
        the recorded base poses ``b_k`` in order at the base-follow speed, so the eef stays on the handle while the
        joint moves (spec §5.3)."""
        q = self.robot.get_joint_positions()
        self.state.arm_target[arm] = q[self.robot.arm_control_idx[arm]].clone().float()
        s = self.cfg.base_follow_speed if speed is None else float(speed)
        cfg = replace(self.cfg.base_tracker, v_max=(s, s, self.cfg.base_follow_yaw_rate))
        yield from self.track_path(base_poses, tracker_cfg=cfg)


__all__ = ["SkillError", "MotionsConfig", "Motions", "MOTION_CATEGORIES", "as_world_pose", "as_pose2d"]
