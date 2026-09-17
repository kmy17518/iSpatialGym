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
    #: a contact leg (a push arc, a brace approach) ends with the pad pressing on the object: the position-controlled
    #: joints settle with a steady error that reflects the contact force, not a tracking failure
    joint_tol_loaded: float = 0.25
    #: how long to keep holding the final waypoint while waiting for convergence (steps)
    settle_steps: int = 90
    #: eef speed of ``arm_along`` (m/s, rad/s) and the densification resolution of its waypoint lists (m, rad)
    cartesian_speed: float = 0.10
    cartesian_ang_speed: float = 0.6
    cartesian_step: float = 0.01
    cartesian_rot_step: float = 0.05
    #: differential-IK tolerances and iteration budget per waypoint (see ``R1ProCuRobo.solve_ik_path``; v1 used 30
    #: iterations, too few near a stretched-arm configuration where the damped steps shrink)
    diff_ik_pos_tol: float = 2e-3
    diff_ik_rot_tol: float = 1e-2
    diff_ik_max_iters: int = 100
    #: a waypoint the iterations leave within these residuals is accepted (a straight approach stalling 2.1 mm short
    #: of the 2 mm tolerance is not a failed approach; the eef postcondition is 2 cm)
    diff_ik_soft_pos_tol: float = 5e-3
    diff_ik_soft_rot_tol: float = 3e-2
    #: soft residual accepted on free-space legs (lift / retreat: the eef postcondition is 2 cm)
    diff_ik_soft_pos_tol_free: float = 1e-2
    #: joint speed bound of a streamed differential-IK path (rad/s; a nullspace escape near a singularity moves the
    #: joints far for a small eef step)
    diff_ik_joint_speed: float = 1.0
    #: gripper actuation wait (steps)
    gripper_steps: int = 60
    #: base tracking (paths) and base-follow (articulation) tracker configurations
    base_tracker: BaseTrackerConfig = field(default_factory=BaseTrackerConfig)
    base_follow_speed: float = 0.15  # m/s along the recorded b_k
    base_follow_yaw_rate: float = 0.4  # rad/s
    #: fraction of the tracker's v_max used to time-parameterise a recorded path
    path_speed_fraction: float = 0.8
    #: corner cutting the reference allows at a heading change (m): the speed through a corner of angle θ is
    #: sqrt(2·a_max·path_corner_cut) / sin(θ/2), so the velocity-vector change fits the tracker's acceleration limit
    #: without leaving the path by more than this (a 90° corner at a_max 0.6 → 0.16 m/s; straights at v_max)
    path_corner_cut: float = 0.01
    #: the reference index never runs more than this ahead of the base (m): a lagging base is waited for instead of
    #: chased toward a point past the corner (the corridor of an obstructed path leaves a few cm per side)
    path_max_lag: float = 0.04
    #: arc-length resolution of the speed profile (m)
    path_profile_ds: float = 0.005
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


@dataclass
class ApproachFinish:
    """How ``Motions.arm_along`` may finish a stalled straight segment: plan the remainder with ``arm_to`` to
    ``pose`` (``q_hint`` = the sidecar's IK witness of that pose when recorded), with ``ignore_objects`` out of the
    collision world (the grasped object itself); ``used`` / ``info`` report what happened."""

    pose: Any
    q_hint: Any = None
    ignore_objects: Sequence[Any] = ()
    max_stall: float = 0.015
    max_remaining: float = 0.06
    #: largest joint change the finishing plan may make when cuRobo picks the branch (no witness); rad
    max_excursion: float = 0.6
    used: bool = False
    info: Optional[Dict[str, Any]] = None


class PathTracker(BaseTrajectoryTracker):
    """The v1 joystick-like tracker (feed-forward + pose feedback, velocity / acceleration limits, exponential
    smoothing, simultaneous translation and rotation) with three changes found necessary on the pilot's obstructed
    paths (5 / 12 demos touched the obstacle while the recorded path itself cleared it by 3–6 cm):

    * **lag guard** — the reference index never runs more than ``max_lag`` metres ahead of the base; v1 advanced one
      sample per step whatever the base did, so a base that fell behind at a corner was pulled toward a target
      already past it and cut the corner;
    * **world-frame smoothing** — v1 smoothed and acceleration-limited the command in the *base* frame, so while the
      base turned (0.8 rad/s) a stale command kept rotating with it and drove the base sideways in the world (the
      25 cm drift after the 90° turns); the filter state lives in the world frame here, the per-axis velocity limits
      stay in the base frame;
    * the smoothing is applied **before** the acceleration clamp, so the effective acceleration is ``a_max`` (v1's
      order made it ``alpha · a_max`` = 0.21 m/s², far below what its constant-speed reference asked at corners).
    """

    def __init__(self, ref: th.Tensor, dt: float, cfg: Optional[BaseTrackerConfig] = None, max_lag: float = 0.04):
        super().__init__(ref, dt, cfg)
        self.max_lag = float(max_lag)
        self.max_steps = int(self.cfg.timeout_factor * len(self.ref) + self.cfg.timeout_extra)
        self.last_w = th.zeros(3)  # filter state: the last command in the world frame (vx, vy, wz)

    def _advance(self, cur: Tuple[float, float, float]):
        T = len(self.ref)
        if self.k >= T - 1:
            return
        window = self.ref[self.k : min(T, self.k + 30), :2]
        d = th.linalg.norm(window - th.tensor(cur[:2], dtype=window.dtype), dim=1)
        k_near = self.k + int(th.argmin(d))
        if k_near > self.k:
            self.k = min(k_near, T - 1)  # the base is nearer a later sample: catch up
        elif float(d[0]) <= self.max_lag:
            self.k = min(self.k + 1, T - 1)  # on the reference: one sample per step
        # else: the base lags the current sample by more than max_lag -> hold the reference

    def step(self, cur: Tuple[float, float, float]) -> Tuple[th.Tensor, bool]:
        self.steps += 1
        self._advance(cur)
        T = len(self.ref)
        k = self.k
        cfg = self.cfg
        tgt = self.ref[min(k + cfg.lookahead_steps, T - 1)]
        v_ff_w = (self.ref[k + 1] - self.ref[k]) / self.dt if k < T - 1 else th.zeros(3)
        ex, ey = float(tgt[0]) - cur[0], float(tgt[1]) - cur[1]
        eyaw = C.wrap_angle(float(tgt[2]) - cur[2])
        v_w = th.tensor([float(v_ff_w[0]) + cfg.kp_pos * ex, float(v_ff_w[1]) + cfg.kp_pos * ey, float(v_ff_w[2]) + cfg.kp_yaw * eyaw])
        # smoothing (JoyLo stick filter) and acceleration limit in the world frame
        v_w = cfg.alpha * v_w + (1.0 - cfg.alpha) * self.last_w
        a_max = th.tensor(cfg.a_max) * self.dt
        v_w = self.last_w + th.clamp(v_w - self.last_w, -a_max, a_max)
        # per-axis velocity limits in the base frame
        c, s = math.cos(cur[2]), math.sin(cur[2])
        v_b = th.tensor([c * float(v_w[0]) + s * float(v_w[1]), -s * float(v_w[0]) + c * float(v_w[1]), float(v_w[2])])
        v_max = th.tensor(cfg.v_max)
        v_b = th.clamp(v_b, -v_max, v_max)
        # termination
        d_goal, dyaw_goal = self.error(cur)
        at_goal = k >= T - 1 and d_goal < cfg.pos_tol and dyaw_goal < cfg.yaw_tol
        self.settled = self.settled + 1 if at_goal else 0
        done = self.settled >= cfg.settle_steps
        if done:
            v_b = th.zeros(3)
        self.last_w = th.tensor([c * float(v_b[0]) - s * float(v_b[1]), s * float(v_b[0]) + c * float(v_b[1]), float(v_b[2])])
        self.last_cmd = v_b
        return v_b, done


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
        self.last_relaxed_links: List[str] = []  # held-arm links taken out of the sphere model by the last arm_to plan
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

    def _wait_converged(self, arms: Sequence[str], trunk: bool, tol: float, max_steps: int, loose: Optional[float] = None) -> Iterator[th.Tensor]:
        def err() -> float:
            return max([self.state.arm_error(a) for a in arms] + ([self.state.trunk_error()] if trunk else [0.0]))

        for _ in range(max_steps):
            if err() < tol:
                return
            yield self._emit()
        e = err()
        if e > (self.cfg.joint_tol_loose if loose is None else loose):
            raise SkillError("execution", f"joints did not converge (max err {e:.3f} rad)", arms=list(arms), trunk=trunk, err_rad=e)
        if e >= tol:
            log.info("joints settled %.3f rad off their targets (accepted%s)", e, ", contact leg" if loose is not None else "")

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

    def execute_joint_trajectory(self, q_traj: th.Tensor, arms: Sequence[str], trunk: bool = False, tol: Optional[float] = None, loaded: bool = False) -> Iterator[th.Tensor]:
        """Stream a ``(T, 28)`` joint trajectory (one waypoint per control step) for the given groups, then hold the
        last waypoint until the joints converge (``loaded``: a contact leg, the steady error under the contact force
        is accepted up to ``joint_tol_loaded``)."""
        q_traj = C.as_tensor(q_traj)
        if len(q_traj) > self.cfg.trajectory_max_steps:
            raise SkillError("timeout", f"trajectory of {len(q_traj)} steps exceeds the budget {self.cfg.trajectory_max_steps}")
        self.state.stop_base()
        for q in q_traj:
            self.state.set_from_q(q, arms=arms, trunk=trunk)
            yield self._emit()
        yield from self._wait_converged(arms, trunk, tol or self.cfg.joint_tol, self.cfg.settle_steps, loose=self.cfg.joint_tol_loaded if loaded else None)
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

    def arm_to(self, arms: Union[str, Sequence[str]], pose: Any = None, q: Any = None, ignore_objects=(), q_hint: Any = None,
               max_excursion: Optional[float] = None) -> Iterator[th.Tensor]:
        """cuRobo trajopt of the arm(s) with the trunk locked, then streamed. Exactly one of ``pose`` (a world pose
        for one arm, or ``{arm: pose}``) or ``q`` (a 28-D vector, ``{joint_name: value}`` or ``{arm: q7}``) is the
        target. Raises ``plan_failed`` when no trajectory is found (the sampler proved the key poses, not the
        trajectory between them, §3.9).

        ``q_hint`` (a 28-D configuration, the sampler's IK witness of ``pose``): the arm is planned in **joint space**
        to the witness's arm joints — the IK branch the sampler validated the following approach / descent from —
        and the residual to ``pose`` (the object settled a few mm away from its recorded pose) is closed with a short
        ``arm_along``. Without a hint cuRobo picks its own IK branch for the pose goal; ``max_excursion`` (rad) then
        bounds the largest joint change of the plan — a finishing motion of a few centimetres must not become an elbow
        flip (the pilot's french-fries finish swung the hand out of the camera frame)."""
        arms = (arms,) if isinstance(arms, str) else tuple(arms)
        emb = self._emb_for(arms)
        self.backend.update_world(ignore_objects=ignore_objects)
        q_start = self.backend.q_full()
        if (pose is None) == (q is None):
            raise ValueError("arm_to needs exactly one of pose / q")
        relax = getattr(self.backend, "static_links_relaxed", None)
        if pose is not None and q_hint is not None and len(arms) == 1:
            arm = arms[0]
            q28 = C.as_tensor(q_hint).reshape(-1)
            q_goal = q_start.clone()
            q_goal[self.robot.arm_control_idx[arm]] = q28[self.robot.arm_control_idx[arm]]
            if relax is not None:
                with relax(emb, q_start, arms) as relaxed:
                    self.last_relaxed_links = list(relaxed)
                    res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            else:
                res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            if not res.success:
                raise SkillError("plan_failed", f"arm_to({arm}, witness q): {res.status}", diagnostics=dict(res.diagnostics), stage="trajopt")
            if max_excursion is not None:
                idx = self.robot.arm_control_idx[arm]
                exc = float(th.max(th.abs(q_goal[idx] - q_start[idx])))
                if exc > float(max_excursion):
                    raise SkillError("plan_failed", f"arm_to({arm}, witness q): the recorded IK witness lies {exc:.2f} rad away in joint space (> {max_excursion:.2f} allowed for a "
                                     f"finishing motion): the sidecar's pre-pose and end-pose witnesses are in different IK branches", stage="trajopt", excursion=exc)
            log.info("arm_to(%s): %d waypoints to the recorded IK witness (%.1fs), planned in %.2fs", arm, len(res.q_traj), res.duration, res.plan_time)
            yield from self.execute_joint_trajectory(res.q_traj, arms=arms)
            target = as_world_pose(pose)
            cur = self.eef_pose(arm)
            dp, da = float(th.linalg.norm(cur[0] - target[0])), C.quat_angle(cur[1], target[1])
            if dp > 2e-3 or da > 0.01:
                log.info("arm_to(%s): witness residual %.1f mm / %.3f rad, closing it along a straight segment", arm, dp * 1000, da)
                yield from self.arm_along(arm, [target])
            return
        if pose is not None:
            poses = pose if isinstance(pose, dict) and set(pose) <= set(M.ARMS) else {arms[0]: pose}
            targets = {M.EEF_LINK[a]: as_world_pose(p) for a, p in poses.items()}
            if relax is not None:
                with relax(emb, q_start, arms) as relaxed:
                    self.last_relaxed_links = list(relaxed)
                    res = self.backend.plan_pose(emb, targets, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            else:
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
        if max_excursion is not None:
            idx = th.cat([self.robot.arm_control_idx[a] for a in arms])
            exc = float(th.max(th.abs(res.q_traj[:, idx] - q_start[idx])))
            if exc > float(max_excursion):
                raise SkillError("plan_failed", f"arm_to({','.join(arms)}, {what}): the plan swings a joint by {exc:.2f} rad (> {max_excursion:.2f} allowed for a finishing motion; "
                                 f"cuRobo's IK branch differs from the current one, no IK witness in the sidecar)", stage="trajopt", excursion=exc)
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

    def arm_along(self, arm: str, waypoints: Sequence[Any], speed: Optional[float] = None, finish: Optional["ApproachFinish"] = None,
                  soft_pos_tol: Optional[float] = None, loaded: bool = False) -> Iterator[th.Tensor]:
        """Track a given eef waypoint list with damped-least-squares differential IK on the arm's joints (trunk and
        base locked), streamed at ``speed`` (m/s; ``cartesian_speed``). The list is densified to the tracking
        resolution and every waypoint is reached in order; no collision veto (contact motions must touch — the QA
        collision monitor judges contacts). Raises ``plan_failed`` (stage ``diff_ik``) when a waypoint cannot be
        reached from the current configuration.

        ``finish`` (owner rule 2026-09-16): when the differential IK **stalls** — no joint at a limit, the residual
        below ``finish.max_stall`` — with at most ``finish.max_remaining`` of the segment left, the converged prefix
        is executed and the rest is planned by ``arm_to`` to ``finish.pose`` (to the sidecar's IK witness
        ``finish.q_hint`` when there is one, else cuRobo's own IK branch) instead of failing: the stall is the reach
        boundary of the *branch* the straight segment started in (more iterations do not help: 30 → 100 moved a
        7.2 mm stall to 7.0 mm), while the pose itself is reachable (the sampler's exact IK)."""
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
            max_iters=self.cfg.diff_ik_max_iters,
            check_collisions=False,
            sync_world=False,
            soft_pos_tol=self.cfg.diff_ik_soft_pos_tol if soft_pos_tol is None else float(soft_pos_tol),
            soft_rot_tol=self.cfg.diff_ik_soft_rot_tol,
        )
        if res.success and (res.diagnostics or {}).get("soft_waypoints"):
            log.info("arm_along(%s): %s", arm, res.status)
        speed = self.cfg.cartesian_speed if speed is None else float(speed)
        steps = []
        prev = start
        q_traj = res.q_traj if res.q_traj is not None else None
        for i, wp in enumerate(dense):
            seg = float(th.linalg.norm(wp[0] - prev[0]))
            ang = C.quat_angle(prev[1], wp[1])
            n_steps = max(seg / (speed * self.dt), ang / (self.cfg.cartesian_ang_speed * self.dt))
            if q_traj is not None and i + 1 < len(q_traj):
                # a singularity escape may move the joints a lot for a small eef step: keep the joint speed bounded
                dq = float(th.max(th.abs(q_traj[i + 1] - q_traj[i])))
                n_steps = max(n_steps, dq / (self.cfg.diff_ik_joint_speed * self.dt))
            steps.append(max(1, int(round(n_steps))))
            prev = wp
        if not res.success:
            diag = dict(getattr(res, "diagnostics", None) or {})
            k = diag.get("fail_index")
            stall = diag.get("pos_err")
            if finish is not None and diag.get("stage") == "diff_ik" and k is not None and stall is not None and not diag.get("at_lower") and not diag.get("at_upper"):
                remaining = sum(float(th.linalg.norm(dense[i][0] - dense[i - 1][0])) for i in range(max(k, 1), len(dense)))
                remaining += stall
                if stall <= finish.max_stall and remaining <= finish.max_remaining:
                    log.warning("arm_along(%s): %s -> executing the %d converged waypoints and finishing the last %.1f cm with arm_to (%s)", arm, res.status, int(diag.get("converged", 0)), remaining * 100,
                                "IK witness" if finish.q_hint is not None else "cuRobo IK")
                    if res.q_traj is not None and len(res.q_traj) > 1:
                        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps[: len(res.q_traj) - 1]), arms=[arm])
                    yield from self.arm_to(arm, pose=finish.pose, q_hint=finish.q_hint, ignore_objects=finish.ignore_objects, max_excursion=finish.max_excursion)
                    finish.used = True
                    finish.info = {"stall_m": stall, "remaining_m": remaining, "converged": int(diag.get("converged", 0)), "n_dense": len(dense),
                                   "by": "witness" if finish.q_hint is not None else "curobo_ik"}
                    return
            raise SkillError("plan_failed", f"arm_along({arm}): {res.status}", stage="diff_ik", n_waypoints=len(waypoints), **{k_: v for k_, v in diag.items() if k_ in ("fail_index", "pos_err", "rot_err")})
        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps), arms=[arm], loaded=loaded)

    # ------------------------------------------------------------------ base

    def _base_reference(self, waypoints: Sequence[Any], v_lin: float, v_ang: float, a_max: Optional[float] = None) -> th.Tensor:
        """Time-parameterise the polyline current pose → waypoints with a **corner-aware trapezoidal speed profile**:
        ``v_lin`` on the straights, ``sqrt(2·a_max·path_corner_cut) / sin(θ/2)`` through a heading change θ, zero at
        the end, ramps at ``a_max`` (the tracker's own limit), every segment slow enough for the yaw change toward the
        waypoint's recorded heading to stay within ``v_ang``; the yaw is blended linearly along each segment. A
        reference the tracker can follow within its acceleration limit is what keeps the base inside the corridor of
        a recorded path (the pilot's obstructed demos cut 45–63° corners by 5–8 cm at constant speed and touched the
        obstacle; 25 cm off after a 90° turn)."""
        a_max = float(self.cfg.base_tracker.a_max[0] if a_max is None else a_max)
        pts = [self.base_pose2d()] + [as_pose2d(w) for w in waypoints]
        xy = th.tensor([[p[0], p[1]] for p in pts], dtype=th.float64)
        yaws = [pts[0][2]]
        for p in pts[1:]:
            yaws.append(yaws[-1] + C.wrap_angle(p[2] - yaws[-1]))
        seg = th.linalg.norm(xy[1:] - xy[:-1], dim=1)
        n_seg = len(seg)
        # -- speed caps: per segment (the yaw change toward the waypoint's heading needs time: smooth-step blend, peak
        # rate 1.5 × mean ≤ v_ang, peak acceleration 6·Δyaw/T² ≤ the tracker's yaw acceleration) and per interior
        # corner (heading change)
        a_yaw = float(self.cfg.base_tracker.a_max[2])
        v_seg = []
        for i in range(n_seg):
            dy = abs(yaws[i + 1] - yaws[i])
            if dy < 1e-6:
                cap = v_lin
            else:
                t_yaw = max(1.5 * dy / v_ang, math.sqrt(6.0 * dy / a_yaw))
                cap = min(v_lin, float(seg[i]) / t_yaw)
            v_seg.append(max(cap, 1e-3))
        v_corner = [v_lin] * (n_seg + 1)
        v_corner[-1] = 0.0
        heads = []
        for i in range(n_seg):
            d = xy[i + 1] - xy[i]
            heads.append(None if float(seg[i]) < 1e-6 else math.atan2(float(d[1]), float(d[0])))
        v_turn = math.sqrt(2.0 * a_max * self.cfg.path_corner_cut)
        for i in range(1, n_seg):
            if heads[i - 1] is None or heads[i] is None:
                v_corner[i] = 0.0  # a pure rotation at this node: stop, turn, go on
                continue
            theta = abs(C.wrap_angle(heads[i] - heads[i - 1]))
            if theta > math.radians(5.0):
                v_corner[i] = min(v_lin, v_turn / math.sin(theta / 2.0))
        # -- arc-length grid, forward / backward passes (v² = v0² + 2 a s)
        ds = float(self.cfg.path_profile_ds)
        s_nodes = [0.0]
        for i in range(n_seg):
            s_nodes.append(s_nodes[-1] + float(seg[i]))
        total_len = s_nodes[-1]
        grid = [0.0]
        while grid[-1] < total_len - 1e-9:
            grid.append(min(total_len, grid[-1] + ds))
        grid_t = th.tensor(grid, dtype=th.float64)
        nodes_t = th.tensor(s_nodes, dtype=th.float64)

        def seg_of(s: float) -> int:
            """Segment index of arc length ``s`` (zero-length segments are skipped: the later segment wins)."""
            k = int(th.searchsorted(nodes_t, th.tensor(s, dtype=th.float64), right=True)) - 1
            return min(max(k, 0), n_seg - 1)

        v_max_at = th.tensor([v_seg[seg_of(s)] for s in grid], dtype=th.float64)
        for i, s_node in enumerate(s_nodes):
            j = int(th.argmin(th.abs(grid_t - s_node)))
            v_max_at[j] = min(float(v_max_at[j]), v_corner[i])
        v = v_max_at.clone()
        v[0] = min(float(v[0]), 0.05)  # start from (nearly) rest
        for j in range(1, len(grid)):
            v[j] = min(float(v[j]), math.sqrt(float(v[j - 1]) ** 2 + 2.0 * a_max * (grid[j] - grid[j - 1])))
        for j in range(len(grid) - 2, -1, -1):
            v[j] = min(float(v[j]), math.sqrt(float(v[j + 1]) ** 2 + 2.0 * a_max * (grid[j + 1] - grid[j])))
        # -- time along the grid (trapezoid on 1/v)
        t_grid = [0.0]
        for j in range(1, len(grid)):
            vm = max(0.5 * (float(v[j]) + float(v[j - 1])), 0.02)
            t_grid.append(t_grid[-1] + (grid[j] - grid[j - 1]) / vm)
        tg = th.tensor(t_grid, dtype=th.float64)

        def smooth(f: float) -> float:
            return f * f * (3.0 - 2.0 * f)

        def pose_at_s(s: float) -> List[float]:
            k = seg_of(s)
            L = float(seg[k])
            f = 0.0 if L < 1e-9 else min(1.0, max(0.0, (s - s_nodes[k]) / L))
            p = xy[k] + (xy[k + 1] - xy[k]) * f
            return [float(p[0]), float(p[1]), yaws[k] + (yaws[k + 1] - yaws[k]) * smooth(f)]

        def dwell(k: int) -> List[List[float]]:
            """Samples of a pure rotation at node ``k`` (segment ``k`` has zero length): a smooth-step yaw blend whose
            peak rate stays within ``v_ang`` and peak acceleration within the tracker's, position fixed."""
            dy = abs(yaws[k + 1] - yaws[k])
            t_yaw = max(1.5 * dy / v_ang, math.sqrt(6.0 * dy / a_yaw))
            n_d = max(1, int(math.ceil(t_yaw / self.dt)))
            return [[float(xy[k][0]), float(xy[k][1]), yaws[k] + (yaws[k + 1] - yaws[k]) * smooth(m / n_d)] for m in range(1, n_d + 1)]

        # -- sample at dt; the pure-rotation segments are inserted as dwell time where the arc length passes them
        samples: List[List[float]] = []
        total_t = t_grid[-1]
        n = max(1, int(math.ceil(total_t / self.dt))) if total_len > 1e-9 else 0
        k_prev = -1
        for i in range(1, n + 1):
            ti = min(total_t, i * self.dt)
            jj = int(th.searchsorted(tg, th.tensor(ti, dtype=th.float64), right=True)) - 1
            jj = min(max(jj, 0), len(grid) - 2)
            f = 0.0 if tg[jj + 1] - tg[jj] < 1e-12 else float((ti - tg[jj]) / (tg[jj + 1] - tg[jj]))
            s = grid[jj] + (grid[jj + 1] - grid[jj]) * f
            k_new = seg_of(s)
            for m in range(k_prev + 1, k_new):
                if float(seg[m]) < 1e-6:
                    samples.extend(dwell(m))
            k_prev = max(k_prev, k_new)
            samples.append(pose_at_s(s))
        for m in range(k_prev + 1, n_seg):
            if float(seg[m]) < 1e-6:
                samples.extend(dwell(m))
        samples.append([float(xy[-1][0]), float(xy[-1][1]), yaws[-1]])
        return th.tensor(samples, dtype=th.float32)

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
        ref = self._base_reference(waypoints, v_lin, v_ang, a_max=min(cfg.a_max[0], cfg.a_max[1]))
        tracker = PathTracker(ref, self.dt, cfg, max_lag=self.cfg.path_max_lag)
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


__all__ = ["SkillError", "MotionsConfig", "Motions", "MOTION_CATEGORIES", "as_world_pose", "as_pose2d", "ApproachFinish", "PathTracker"]
