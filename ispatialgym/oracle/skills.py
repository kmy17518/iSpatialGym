"""Layer-1 motor skills for R1Pro.

Every public skill is a generator that yields 23-D JoyLo actions (one per environment step) and
returns when the motion is complete. Skills raise :class:`SkillError` when planning or execution
fails so that task oracles can retry with different parameters.

Typical use::

    skills = Skills(env, robot)                       # builds the cuRobo backend
    run(env, skills.navigate_to((1.0, 0.5, 0.0)))     # drive somewhere (joystick-like)
    run(env, skills.set_torso(translate=1.0))         # lower the torso along the JoyLo curve
    run(env, skills.grasp("right", obj))              # pre-grasp -> approach -> close -> lift
    run(env, skills.place("right", FramedPose.create([...], frame="obj:tray")))

Frames: pose arguments accept :class:`~ispatialgym.oracle.poses.FramedPose` (``world``, ``base``,
``eef:<arm>``, ``obj:<name>``) or a plain ``(pos, quat_xyzw)`` world pose.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch as th

from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: F401  (import before omnigibson: warp pinning)

import omnigibson.utils.transform_utils as T  # noqa: E402

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402
from ispatialgym.oracle.base_tracker import BaseTrackerConfig, BaseTrajectoryTracker  # noqa: E402
from ispatialgym.oracle.curobo_planner import PlanResult  # noqa: E402
from ispatialgym.oracle.poses import (  # noqa: E402
    FramedPose,
    Pose,
    as_tensor,
    compose,
    invert,
    pose2d_to_pose,
    pose_to_pose2d,
    quat_angle,
    quat_with_z_axis,
    slerp,
    top_down_quat,
    wrap_angle,
    yaw_of,
)
from ispatialgym.oracle.trunk import TrunkState  # noqa: E402

log = logging.getLogger(__name__)

PoseLike = Union[FramedPose, Pose]


class SkillError(RuntimeError):
    """A skill could not be planned or executed."""

    def __init__(self, reason: str, message: str, **details):
        super().__init__(f"[{reason}] {message}" + (f" {details}" if details else ""))
        self.reason = reason
        self.details = details


@dataclass
class SkillsConfig:
    arm: str = "right"
    #: joint-space convergence tolerance for streamed trajectories (rad)
    joint_tol: float = 0.02
    joint_tol_loose: float = 0.08
    #: how long to keep holding the final waypoint while waiting for convergence (steps)
    settle_steps: int = 90
    #: speed for Cartesian (differential-IK) motions (m/s and rad/s of the end-effector)
    cartesian_speed: float = 0.10
    cartesian_ang_speed: float = 0.6
    cartesian_step: float = 0.01
    cartesian_rot_step: float = 0.05
    #: gripper actuation wait (steps)
    gripper_steps: int = 60
    #: default approach / retreat distances (m)
    pregrasp_dist: float = 0.10
    lift_dist: float = 0.08
    place_approach_dist: float = 0.08
    #: base tracking
    base_tracker: BaseTrackerConfig = field(default_factory=BaseTrackerConfig)
    #: trunk rates (JoyLo defaults are 0.1 / 0.05; slightly faster here for shorter demos)
    trunk_translate_rate: float = 0.25
    trunk_tilt_rate: float = 0.15
    #: planning attempts
    plan_attempts: int = 4


# ==========================================================================================
# runner helpers
# ==========================================================================================


def run(env, gen: Iterator[th.Tensor], on_step=None, max_steps: int = 20000) -> int:
    """Step ``env`` with every action produced by a skill generator. Returns number of steps."""
    n = 0
    for action in gen:
        env.step(action)
        n += 1
        if on_step is not None:
            on_step(n, action)
        if n >= max_steps:
            raise SkillError("timeout", f"skill exceeded {max_steps} env steps")
    return n


# ==========================================================================================
# Skills
# ==========================================================================================


class Skills:
    def __init__(self, env, robot=None, backend: Optional[R1ProCuRobo] = None, cfg: Optional[SkillsConfig] = None, prebuild=("right_arm", "base")):
        self.env = env
        self.robot = robot if robot is not None else env.robots[0]
        self.cfg = cfg or SkillsConfig()
        self.arm = self.cfg.arm
        self.backend = backend if backend is not None else R1ProCuRobo(self.robot, prebuild=prebuild)
        self.dt = self.backend.control_dt
        self.state = ActionState(self.robot)
        self.trunk = TrunkState.from_q(self.robot.get_joint_positions()[self.robot.trunk_control_idx], translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        self._joint_limits = (self.robot.joint_lower_limits[self.robot.trunk_control_idx], self.robot.joint_upper_limits[self.robot.trunk_control_idx])

    # ------------------------------------------------------------------ frames & state

    def eef_pose(self, arm: Optional[str] = None) -> Pose:
        arm = arm or self.arm
        pos, quat = self.robot.eef_links[arm].get_position_orientation()
        return pos.float(), quat.float()

    def base_pose2d(self) -> Tuple[float, float, float]:
        pos, quat = self.robot.get_position_orientation()
        return float(pos[0]), float(pos[1]), yaw_of(quat)

    def object(self, name: str):
        obj = self.env.scene.object_registry("name", name)
        if obj is None:
            raise SkillError("unknown_object", f"no object named '{name}' in the scene")
        return obj

    def resolve(self, pose: PoseLike) -> Pose:
        """Convert a (possibly framed) pose to a world-frame ``(pos, quat_xyzw)``."""
        if isinstance(pose, FramedPose):
            local = (as_tensor(pose.pos), as_tensor(pose.quat))
            frame = pose.frame
            if frame == "world":
                return local
            if frame == "base":
                bp, bq = self.robot.get_position_orientation()
                return compose((bp.float(), bq.float()), local)
            if frame.startswith("eef:"):
                return compose(self.eef_pose(frame.split(":", 1)[1]), local)
            if frame.startswith("obj:"):
                obj = self.object(frame.split(":", 1)[1])
                op, oq = obj.get_position_orientation()
                return compose((op.float(), oq.float()), local)
            raise SkillError("bad_frame", f"unknown frame '{frame}'")
        pos, quat = pose
        return as_tensor(pos), as_tensor(quat)

    def held_object(self, arm: Optional[str] = None):
        return self.robot._ag_obj_in_hand[arm or self.arm]

    def _emb_for(self, arm: str, torso: bool = False) -> str:
        return f"{arm}_arm_torso" if torso else f"{arm}_arm"

    # ------------------------------------------------------------------ primitive emitters

    def _emit(self) -> th.Tensor:
        return self.state.action()

    def hold(self, steps: int = 1) -> Iterator[th.Tensor]:
        """Emit hold actions (zero base velocity, current targets)."""
        self.state.stop_base()
        for _ in range(int(steps)):
            yield self._emit()

    def settle(self, max_steps: int = 120, vel_tol: float = 0.02) -> Iterator[th.Tensor]:
        """Hold until every joint velocity is below ``vel_tol``."""
        self.state.stop_base()
        for _ in range(max_steps):
            yield self._emit()
            if float(self.robot.get_joint_velocities().abs().max()) < vel_tol:
                return

    def _wait_converged(self, arms: Sequence[str], trunk: bool, tol: float, max_steps: int) -> Iterator[th.Tensor]:
        for _ in range(max_steps):
            err = max([self.state.arm_error(a) for a in arms] + ([self.state.trunk_error()] if trunk else [0.0]))
            if err < tol:
                return
            yield self._emit()
        err = max([self.state.arm_error(a) for a in arms] + ([self.state.trunk_error()] if trunk else [0.0]))
        if err > self.cfg.joint_tol_loose:
            raise SkillError("execution", f"joints did not converge (max err {err:.3f} rad)")

    # ------------------------------------------------------------------ gripper

    def set_gripper(self, arm: Optional[str] = None, close: bool = True, steps: Optional[int] = None) -> Iterator[th.Tensor]:
        """Open / close the gripper and wait until the fingers stop moving (or an object is grasped)."""
        arm = arm or self.arm
        self.state.set_gripper(arm, close)
        self.state.stop_base()
        steps = steps or self.cfg.gripper_steps
        idx = self.robot.gripper_control_idx[arm]
        for i in range(steps):
            yield self._emit()
            if i > 10:
                if close and self.held_object(arm) is not None:
                    break
                if float(self.robot.get_joint_velocities()[idx].abs().max()) < 1e-3:
                    break
        yield from self.hold(5)

    def open_gripper(self, arm: Optional[str] = None) -> Iterator[th.Tensor]:
        yield from self.set_gripper(arm, close=False)

    def close_gripper(self, arm: Optional[str] = None) -> Iterator[th.Tensor]:
        yield from self.set_gripper(arm, close=True)

    # ------------------------------------------------------------------ trunk

    def _trunk_path_collides(self, translate: Optional[float], tilt: Optional[float]) -> Optional[int]:
        """Sample the trunk motion and return the index of the first colliding sample (or None)."""
        ts = TrunkState(self.trunk.translate, self.trunk.tilt, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        q = self.backend.q_full()
        qs = []
        for _ in range(400):
            done = ts.step_towards(translate, tilt, 0.1)
            qk = q.clone()
            qk[self.robot.trunk_control_idx] = ts.q(self._joint_limits)
            qs.append(qk)
            if done:
                break
        # only collisions *introduced* by the motion matter: drop samples at (or before) the current state
        qs = [qk for qk in qs if float(th.max(th.abs(qk[self.robot.trunk_control_idx] - q[self.robot.trunk_control_idx]))) > 0.02]
        if not qs:
            return None
        coll = self.backend.check_collision(th.stack(qs), emb_name="whole_body", sync_world=True)
        if bool(coll.any()):
            return int(th.nonzero(coll).view(-1)[0]) + 1
        return None

    def ready_arms(self, arms: Sequence[str] = ("left", "right"), heights: Sequence[float] = (0.0, 0.12, 0.24, 0.36)) -> Iterator[th.Tensor]:
        """Planned motion of the arms to a natural 'ready' configuration (hands in front, elbows bent).

        First tries the canonical joint configuration :data:`ARM_READY_Q`; if that is not plannable
        (e.g. the hands would end up inside a table in front of the robot) it falls back to IK-based
        candidates: hands 0.45 m ahead of the base at increasing ``heights`` above the canonical one.
        """
        arms = [a for a in arms if float(th.max(th.abs(self.robot.get_joint_positions()[self.robot.arm_control_idx[a]] - th.tensor(M.ARM_READY_Q[a])))) > self.cfg.joint_tol_loose]
        if not arms:
            return
        emb = "both_arms" if len(arms) == 2 else self._emb_for(arms[0])
        for a in arms:
            self._attached(a)
        self.backend.update_world()
        q_goal = self.backend.q_full().clone()
        for a in arms:
            for n, v in zip(M.ARM_JOINTS[a], M.ARM_READY_Q[a]):
                q_goal[self.backend._jidx[n]] = float(v)
        res = self.backend.plan_joint(emb, q_goal, max_attempts=self.cfg.plan_attempts, sync_world=False)
        if res.success:
            yield from self.execute_joint_trajectory(res.q_traj, arms=arms)
            return
        goal_coll = bool(self.backend.check_collision(q_goal, emb_name="whole_body")[0])
        log.info("ready_arms: joint-space plan failed (%s; goal in collision: %s); lifting hands first", res.status, goal_coll)
        # fallback A: hanging hands next to furniture block the joint-space path -> lift them straight up first
        if not goal_coll:
            lifted = False
            for a in arms:
                try:
                    yield from self.move_eef_cartesian(a, delta_pos=[0.0, 0.0, 0.25], check_collisions=True)
                    lifted = True
                except SkillError as e:
                    log.info("ready_arms: lifting %s hand failed: %s", a, e)
            if lifted:
                res = self.backend.plan_joint(emb, q_goal, max_attempts=self.cfg.plan_attempts)
                if res.success:
                    yield from self.execute_joint_trajectory(res.q_traj, arms=arms)
                    return
        # fallback B: Cartesian ready poses in the base frame at increasing heights
        bp, bq = self.robot.get_position_orientation()
        base = (bp.float(), bq.float())
        Rb = T.quat2mat(base[1])
        z_axis = Rb @ th.tensor([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])
        for h in heights:
            targets = {}
            for a in arms:
                side = 1.0 if a == "left" else -1.0
                pos = compose(base, (th.tensor([0.45, side * 0.18, 0.78 + h]), th.tensor([0.0, 0.0, 0.0, 1.0])))[0]
                targets[M.EEF_LINK[a]] = (pos, quat_with_z_axis(z_axis, self.eef_pose(a)[1]))
            res = self.backend.plan_pose(emb, targets, max_attempts=2, sync_world=False)
            if res.success:
                log.info("ready_arms: using Cartesian fallback at +%.2f m", h)
                yield from self.execute_joint_trajectory(res.q_traj, arms=arms)
                return
            ik = {f: self.backend.solve_ik(emb, {f: p}, sync_world=False) is not None for f, p in targets.items()}
            log.info("ready_arms: fallback +%.2f m failed (%s); per-arm IK feasible: %s", h, res.status, ik)
        raise SkillError("planning", "ready_arms: no plannable ready configuration")

    def set_torso(self, translate: Optional[float] = None, tilt: Optional[float] = None, delta: bool = False, max_steps: int = 2000, auto_ready_arms: bool = True) -> Iterator[th.Tensor]:
        """Move the trunk along the JoyLo keyframe curve (``translate`` in [0, 2]) and/or apply a
        ``tilt`` offset on ``torso_joint3``, at bounded rates exactly like holding the JoyLo controls.

        ``delta=True`` interprets the arguments relative to the current values. The swept trunk motion
        is collision-checked (whole body); if it collides (typically hanging arms hitting the base)
        and ``auto_ready_arms`` is set, the arms are first moved to :data:`ARM_READY_Q`.
        """
        self.trunk = TrunkState.from_q(self.robot.get_joint_positions()[self.robot.trunk_control_idx], translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        if delta:
            translate = None if translate is None else self.trunk.translate + float(translate)
            tilt = None if tilt is None else self.trunk.tilt + float(tilt)
        hit = self._trunk_path_collides(translate, tilt)
        if hit is not None and auto_ready_arms:
            log.info("set_torso: trunk path collides at sample %d; moving arms to the ready pose first", hit)
            yield from self.ready_arms()
            hit = self._trunk_path_collides(translate, tilt)
        if hit is not None:
            raise SkillError("collision", f"trunk motion would collide (sample {hit})", translate=translate, tilt=tilt)
        self.state.stop_base()
        for _ in range(max_steps):
            done = self.trunk.step_towards(translate, tilt, self.dt)
            self.state.trunk_target = self.trunk.q(self._joint_limits)
            yield self._emit()
            if done:
                break
        else:
            raise SkillError("timeout", "set_torso did not finish")
        yield from self._wait_converged([], trunk=True, tol=self.cfg.joint_tol, max_steps=self.cfg.settle_steps)

    # ------------------------------------------------------------------ joint trajectory execution

    def execute_joint_trajectory(self, q_traj: th.Tensor, arms: Sequence[str], trunk: bool = False, tol: Optional[float] = None) -> Iterator[th.Tensor]:
        """Stream a ``(T, 28)`` joint trajectory (one waypoint per control step) for the given groups,
        then hold the last waypoint until the joints converge."""
        q_traj = as_tensor(q_traj)
        self.state.stop_base()
        for q in q_traj:
            self.state.set_from_q(q, arms=arms, trunk=trunk)
            yield self._emit()
        yield from self._wait_converged(arms, trunk, tol or self.cfg.joint_tol, self.cfg.settle_steps)
        if trunk:
            self.trunk = TrunkState.from_q(self.state.trunk_target, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)

    def _retime(self, q_traj: th.Tensor, steps_per_segment: int) -> th.Tensor:
        """Linearly interpolate between consecutive waypoints so each segment takes ``steps_per_segment`` steps."""
        if steps_per_segment <= 1 or len(q_traj) < 2:
            return q_traj
        out = [q_traj[0]]
        for a, b in zip(q_traj[:-1], q_traj[1:]):
            for k in range(1, steps_per_segment + 1):
                out.append(a + (b - a) * (k / steps_per_segment))
        return th.stack(out)

    # ------------------------------------------------------------------ arm: planned motions

    def _attached(self, arm: str):
        obj = self.held_object(arm)
        if obj is not None and arm not in self.backend._attached:
            self.backend.attach_object(arm, obj)
        elif obj is None and arm in self.backend._attached:
            self.backend.detach_object(arm)

    def plan_eef(self, arm: str, pose: PoseLike, torso: bool = False, ignore_objects=()) -> PlanResult:
        pos, quat = self.resolve(pose)
        self._attached(arm)
        emb = self._emb_for(arm, torso)
        self.backend.update_world(ignore_objects=ignore_objects)
        return self.backend.plan_pose(emb, {M.EEF_LINK[arm]: (pos, quat)}, max_attempts=self.cfg.plan_attempts, sync_world=False)

    def move_eef(self, arm: Optional[str] = None, pose: PoseLike = None, torso: bool = False, ignore_objects=(), tol: Optional[float] = None) -> Iterator[th.Tensor]:
        """Plan (cuRobo, collision-free, only the arm's joints - plus torso if ``torso``) and execute a
        motion bringing the end-effector to ``pose``."""
        arm = arm or self.arm
        res = self.plan_eef(arm, pose, torso=torso, ignore_objects=ignore_objects)
        if not res.success:
            raise SkillError("planning", f"move_eef({arm}) failed: {res.status}")
        log.info("move_eef(%s): %d waypoints (%.1fs), planned in %.2fs", arm, len(res.q_traj), res.duration, res.plan_time)
        yield from self.execute_joint_trajectory(res.q_traj, arms=[arm], trunk=torso, tol=tol)

    def move_eefs(self, poses: Dict[str, PoseLike], ignore_objects=()) -> Iterator[th.Tensor]:
        """Bimanual planned motion: ``poses`` maps arm name -> target pose (arms not listed stay put)."""
        for arm in poses:
            self._attached(arm)
        self.backend.update_world(ignore_objects=ignore_objects)
        targets = {M.EEF_LINK[a]: self.resolve(p) for a, p in poses.items()}
        res = self.backend.plan_pose("both_arms", targets, max_attempts=self.cfg.plan_attempts, sync_world=False)
        if not res.success:
            raise SkillError("planning", f"move_eefs failed: {res.status}")
        yield from self.execute_joint_trajectory(res.q_traj, arms=list(poses.keys()))

    def move_joints(self, q_target: Dict[str, float] | th.Tensor, emb: str = "right_arm") -> Iterator[th.Tensor]:
        """Planned joint-space motion (collision-free) for the active joints of ``emb``."""
        q_goal = self.backend.q_full().clone()
        if isinstance(q_target, dict):
            for n, v in q_target.items():
                q_goal[self.backend._jidx[n]] = float(v)
        else:
            q_goal = as_tensor(q_target)
        res = self.backend.plan_joint(emb, q_goal, max_attempts=self.cfg.plan_attempts)
        if not res.success:
            raise SkillError("planning", f"move_joints failed: {res.status}")
        e = M.EMBODIMENTS[emb]
        arms = [a for a in M.ARMS if set(M.ARM_JOINTS[a]) & set(e.active_joints)]
        trunk = bool(set(M.TORSO_JOINTS) & set(e.active_joints))
        yield from self.execute_joint_trajectory(res.q_traj, arms=arms, trunk=trunk)

    # ------------------------------------------------------------------ arm: Cartesian motions

    def _dense_waypoints(self, start: Pose, goal: Pose) -> List[Pose]:
        """Interpolate a straight line (position) + slerp (orientation) from start to goal."""
        p0, q0 = as_tensor(start[0]), as_tensor(start[1])
        p1, q1 = as_tensor(goal[0]), as_tensor(goal[1])
        n = max(1, int(math.ceil(max(float(th.linalg.norm(p1 - p0)) / self.cfg.cartesian_step, quat_angle(q0, q1) / self.cfg.cartesian_rot_step))))
        return [(p0 + (p1 - p0) * (k / n), slerp(q0, q1, k / n)) for k in range(1, n + 1)]

    def move_eef_cartesian(
        self,
        arm: Optional[str] = None,
        pose: Optional[PoseLike] = None,
        delta_pos=None,
        delta_frame: str = "world",
        waypoints: Optional[Sequence[Pose]] = None,
        speed: Optional[float] = None,
        check_collisions: bool = True,
        ignore_objects=(),
        torso: bool = False,
    ) -> Iterator[th.Tensor]:
        """Move the end-effector along a straight line (or explicit dense ``waypoints``) with fixed
        orientation interpolation, using differential IK on the arm joints only (plus the torso
        joints when ``torso`` is set).

        Exactly one of ``pose`` (goal pose), ``delta_pos`` (translation in ``delta_frame``: ``world``,
        ``base`` or ``eef``) or ``waypoints`` (world poses) must be given.
        """
        arm = arm or self.arm
        start = self.eef_pose(arm)
        if waypoints is None:
            if pose is not None:
                goal = self.resolve(pose)
            elif delta_pos is not None:
                d = as_tensor(delta_pos)
                if delta_frame == "eef":
                    d = T.quat2mat(start[1]) @ d
                elif delta_frame == "base":
                    d = T.quat2mat(self.robot.get_position_orientation()[1].float()) @ d
                goal = (start[0] + d, start[1])
            else:
                raise ValueError("move_eef_cartesian needs pose, delta_pos or waypoints")
            waypoints = self._dense_waypoints(start, goal)
        self._attached(arm)
        self.backend.update_world(ignore_objects=ignore_objects)
        res = self.backend.solve_ik_path(self._emb_for(arm, torso), [{M.EEF_LINK[arm]: wp} for wp in waypoints], check_collisions=check_collisions, sync_world=False)
        if not res.success:
            raise SkillError("planning", f"move_eef_cartesian({arm}) failed: {res.status}")
        # time parameterisation: waypoint spacing / speed
        seg = float(th.linalg.norm(as_tensor(waypoints[0][0]) - start[0])) if len(waypoints) else self.cfg.cartesian_step
        if len(waypoints) > 1:
            seg = max(seg, float(th.linalg.norm(as_tensor(waypoints[1][0]) - as_tensor(waypoints[0][0]))))
        ang = quat_angle(start[1], waypoints[0][1]) if len(waypoints) else 0.0
        speed = speed or self.cfg.cartesian_speed
        steps = max(1, int(round(max(seg / (speed * self.dt), ang / (self.cfg.cartesian_ang_speed * self.dt)))))
        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps), arms=[arm], trunk=torso)

    def rotate_eef(self, arm: Optional[str] = None, delta_quat=None, pivot: Union[str, th.Tensor] = "eef", frame: str = "world", speed: Optional[float] = None, torso: bool = False) -> Iterator[th.Tensor]:
        """Rotate the end-effector by ``delta_quat`` (xyzw) about ``pivot``.

        ``pivot``: ``"eef"`` (rotate in place), ``"object"`` (centre of the held object, keeps it in
        view), or a world position. ``frame``: rotation expressed in ``world`` or ``eef`` axes.
        """
        arm = arm or self.arm
        p0, q0 = self.eef_pose(arm)
        dq = as_tensor(delta_quat)
        R0 = T.quat2mat(q0)
        Rd = T.quat2mat(dq)
        R1 = (Rd @ R0) if frame == "world" else (R0 @ Rd)
        q1 = T.mat2quat(R1)
        if isinstance(pivot, str) and pivot == "eef":
            piv = p0.clone()
        elif isinstance(pivot, str) and pivot == "object":
            obj = self.held_object(arm)
            if obj is None:
                raise SkillError("no_object", "rotate_eef(pivot='object') requires a held object")
            piv = obj.aabb_center.float()
        else:
            piv = as_tensor(pivot)
        ang = quat_angle(q0, q1)
        n = max(1, int(math.ceil(ang / self.cfg.cartesian_rot_step)))
        wps = []
        r = p0 - piv
        for k in range(1, n + 1):
            qk = slerp(q0, q1, k / n)
            Rk = T.quat2mat(qk) @ R0.T  # incremental rotation from the start
            wps.append((piv + Rk @ r, qk))
        yield from self.move_eef_cartesian(arm, waypoints=wps, speed=speed, torso=torso)

    # ------------------------------------------------------------------ base

    def _base_reference_from_plan(self, res: PlanResult) -> th.Tensor:
        """World-frame ``(x, y, yaw)`` reference from the base joints of a planned trajectory."""
        j = self.backend._jidx
        q = res.q_traj
        rp, rq = self.backend.root_pose()
        ryaw = yaw_of(rq)
        c, s = math.cos(ryaw), math.sin(ryaw)
        x = q[:, j["base_footprint_x_joint"]]
        y = q[:, j["base_footprint_y_joint"]]
        yaw = q[:, j["base_footprint_rz_joint"]] + ryaw
        wx = rp[0] + c * x - s * y
        wy = rp[1] + s * x + c * y
        return th.stack([wx, wy, yaw], dim=1)

    def _base_reference_from_waypoints(self, pts: th.Tensor, yaw_goal: float) -> th.Tensor:
        """Time-parameterise 2-D path ``pts`` at the tracker's max speed with yaw blended linearly."""
        x0, y0, yaw0 = self.base_pose2d()
        pts = th.cat([th.tensor([[x0, y0]]), as_tensor(pts).reshape(-1, 2)], dim=0)
        seg = th.linalg.norm(pts[1:] - pts[:-1], dim=1)
        total = float(seg.sum())
        v = 0.8 * min(self.cfg.base_tracker.v_max[0], self.cfg.base_tracker.v_max[1])
        n = max(2, int(total / (v * self.dt)))
        cum = th.cat([th.zeros(1), th.cumsum(seg, 0)])
        s = th.linspace(0, total, n)
        # linear interpolation along arc length
        out = []
        dyaw = wrap_angle(yaw_goal - yaw0)
        for si in s.tolist():
            k = int(th.searchsorted(cum, th.tensor(si), right=True)) - 1
            k = min(max(k, 0), len(seg) - 1)
            t = 0.0 if float(seg[k]) < 1e-9 else (si - float(cum[k])) / float(seg[k])
            p = pts[k] + (pts[k + 1] - pts[k]) * t
            frac = si / total if total > 1e-9 else 1.0
            out.append([float(p[0]), float(p[1]), yaw0 + dyaw * frac])
        return th.tensor(out)

    def plan_base(self, pose2d: Tuple[float, float, float], ignore_objects=()) -> th.Tensor:
        """Collision-free ``(x, y, yaw)`` reference to ``pose2d`` (world): cuRobo whole-body base plan,
        falling back to the scene traversability map for long / cluttered routes."""
        x, y, yaw = pose2d
        self.backend.update_world(ignore_objects=ignore_objects)
        zb = float(self.robot.get_position_orientation()[0][2])
        target = pose2d_to_pose(x, y, yaw, z=zb)
        res = self.backend.plan_pose("base", {M.BASE_LINK: target}, max_attempts=self.cfg.plan_attempts, sync_world=False)
        if res.success:
            return self._base_reference_from_plan(res)
        log.warning("cuRobo base plan failed (%s); falling back to traversability-map path", res.status)
        trav = getattr(self.env.scene, "trav_map", None)
        x0, y0, _ = self.base_pose2d()
        ref = None
        if trav is not None and getattr(trav, "floor_map", None) is not None:
            try:
                path, _ = trav.get_shortest_path(0, th.tensor([x0, y0]), th.tensor([x, y]), entire_path=True, robot=self.robot)
            except Exception as e:  # map may not cover the query
                log.warning("traversability-map query failed: %s", e)
                path = None
            if path is not None:
                ref = self._base_reference_from_waypoints(path, yaw)
        if ref is None:
            ref = self._base_reference_from_waypoints(th.tensor([[x, y]]), yaw)  # straight line
        # the fallback must never drive the (whole) body into obstacles: validate every ~10 cm
        if not self.base_reference_collision_free(ref):
            raise SkillError("planning", f"no collision-free base path to {pose2d}", curobo_status=res.status)
        return ref

    def q_with_base_pose(self, q: th.Tensor, x: float, y: float, yaw: float) -> th.Tensor:
        """Copy of ``q`` with the virtual base joints set so that base_link is at world ``(x, y, yaw)``."""
        j = self.backend._jidx
        rp, rq = self.backend.root_pose()
        ryaw = yaw_of(rq)
        c, s = math.cos(ryaw), math.sin(ryaw)
        dx, dy = float(x - rp[0]), float(y - rp[1])
        qk = as_tensor(q).clone()
        qk[j["base_footprint_x_joint"]] = c * dx + s * dy
        qk[j["base_footprint_y_joint"]] = -s * dx + c * dy
        qk[j["base_footprint_rz_joint"]] = wrap_angle(float(yaw) - ryaw)
        return qk

    def base_reference_collision_free(self, ref: th.Tensor, stride_m: float = 0.1) -> bool:
        """Whole-body collision check of a world-frame ``(x, y, yaw)`` reference at the current arm/torso pose."""
        q = self.backend.q_full()
        pts = [ref[0]]
        for r in ref[1:]:
            if float(th.linalg.norm(r[:2] - pts[-1][:2])) >= stride_m or r is ref[-1]:
                pts.append(r)
        qs = [self.q_with_base_pose(q, float(r[0]), float(r[1]), float(r[2])) for r in pts]
        return not bool(self.backend.check_collision(th.stack(qs), emb_name="whole_body").any())

    # ------------------------------------------------------------------ stance selection

    def stance_for(
        self,
        target,
        arm: Optional[str] = None,
        approach: str = "top",
        distances: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.8),
        translates: Sequence[float] = (0.5, 0.75, 1.0, 0.25, 0.0),
        lateral: Optional[float] = None,
        pregrasp_dist: Optional[float] = None,
        ignore_objects=(),
        eef_candidates: Optional[Sequence[Pose]] = None,
    ) -> Tuple[Tuple[float, float, float], float]:
        """Choose a base pose ``(x, y, yaw)`` and trunk ``translate`` from which ``target`` (an object or
        a world position) can be grasped with ``arm``.

        Candidates face the target from the direction of the current base position, at increasing
        distances and for several JoyLo trunk levels; the first one that is whole-body collision-free
        and has IK-feasible grasp + pre-grasp poses wins. Mirrors how a JoyLo operator positions the
        robot before reaching: stand close, lower the trunk to bring the shoulder above the object.

        ``eef_candidates`` (world eef poses that must all be reachable, e.g. a place pose and the
        pose above it) replaces the automatic grasp candidates.
        """
        arm = arm or self.arm
        obj = None
        if not isinstance(target, (tuple, list, th.Tensor)):
            obj = target
            tpos = obj.aabb_center.float()
        else:
            tpos = as_tensor(target)
        pregrasp_dist = self.cfg.pregrasp_dist if pregrasp_dist is None else pregrasp_dist
        lateral = (-0.2 if arm == "right" else 0.2) if lateral is None else lateral  # target on the arm's side
        x0, y0, _ = self.base_pose2d()
        d0 = th.tensor([float(tpos[0]) - x0, float(tpos[1]) - y0])
        theta = math.atan2(float(d0[1]), float(d0[0])) if float(th.linalg.norm(d0)) > 1e-3 else self.base_pose2d()[2]
        ignore_grasp = list(ignore_objects) + ([obj] if obj is not None else [])
        self._attached(arm)
        q = self.backend.q_full()
        emb = self._emb_for(arm)
        if eef_candidates is not None:
            cand_sets = [[self.resolve(p) for p in eef_candidates]]  # one set: all poses must be reachable
        else:
            cands = self.grasp_candidates(obj, arm, approach, n_yaw=4) if obj is not None else [(tpos, top_down_quat(self.eef_pose(arm)[1]))]
            cand_sets = []
            for pos, quat in cands:
                axis = T.quat2mat(quat)[:, 2]
                cand_sets.append([(pos - axis * pregrasp_dist, quat), (pos, quat)])  # pre-grasp and grasp
        # 1) stance must be collision-free with the *full* world (hands must not rest on the target)
        stances = []
        self.backend.update_world(ignore_objects=ignore_objects)
        for translate in translates:
            for d in distances:
                # base centre at distance d behind the target along theta, shifted so the target sits at base-frame y = lateral
                bx = float(tpos[0]) - d * math.cos(theta) - lateral * (-math.sin(theta))
                by = float(tpos[1]) - d * math.sin(theta) - lateral * math.cos(theta)
                qk = self.q_with_base_pose(q, bx, by, theta)
                qk[self.robot.trunk_control_idx] = TrunkState(translate, 0.0).q(self._joint_limits)
                stances.append((translate, d, bx, by, qk))
        coll = self.backend.check_collision(th.stack([s[4] for s in stances]), emb_name="whole_body", margin=0.01)
        stances = [s for s, c in zip(stances, coll.tolist()) if not c]
        # 2) grasp and pre-grasp must be IK-feasible from the stance (target object ignored for contact)
        self.backend.update_world(ignore_objects=ignore_grasp)
        for translate, d, bx, by, qk in stances:
            for poses in cand_sets:
                if all(self.backend.solve_ik(emb, {M.EEF_LINK[arm]: p}, q_seed=qk, sync_world=False) is not None for p in poses):
                    log.info("stance_for: base (%.2f, %.2f, %.2f) translate %.2f distance %.2f", bx, by, theta, translate, d)
                    return (bx, by, theta), translate
        raise SkillError("planning", "stance_for: no feasible stance found", target=tpos.tolist())

    def approach(self, target, arm: Optional[str] = None, approach: str = "top", **kw) -> Iterator[th.Tensor]:
        """Navigate + set the trunk so that ``target`` is graspable (see :meth:`stance_for`).

        The trunk is changed before driving when the new base pose would collide at the current trunk
        level (e.g. hands too low for the table in front), otherwise after arriving.
        """
        arm = arm or self.arm
        stance = self.stance_for(target, arm=arm, approach=approach, **kw)
        yield from self.go_to_stance(stance)

    def go_to_stance(self, stance) -> Iterator[th.Tensor]:
        """Execute a stance from :meth:`stance_for`: trunk first if the new base pose collides at the
        current trunk level, otherwise drive first."""
        (bx, by, yaw), translate = stance
        q = self.backend.q_full()
        q_new_base = self.q_with_base_pose(q, bx, by, yaw)
        torso_first = bool(self.backend.check_collision(q_new_base, emb_name="whole_body", sync_world=True)[0])
        if torso_first:
            yield from self.set_torso(translate=translate)
        yield from self.navigate_to((bx, by, yaw))
        if not torso_first:
            yield from self.set_torso(translate=translate)

    def navigate_to(self, pose: Union[Tuple[float, float, float], PoseLike], ignore_objects=(), tracker_cfg: Optional[BaseTrackerConfig] = None) -> Iterator[th.Tensor]:
        """Drive the base to a 2-D pose ``(x, y, yaw)`` (world) or a framed pose, joystick-style:
        plan a smooth collision-free base path with cuRobo, then track it with a holonomic velocity
        controller (translation and rotation simultaneously, acceleration limited, smoothed)."""
        if isinstance(pose, FramedPose) or (isinstance(pose, (tuple, list)) and len(pose) == 2):
            pose2d = pose_to_pose2d(self.resolve(pose))
        else:
            pose2d = tuple(float(v) for v in as_tensor(pose).reshape(-1)[:3])
        ref = self.plan_base(pose2d, ignore_objects=ignore_objects)
        yield from self.track_base(ref, tracker_cfg)

    def track_base(self, ref: th.Tensor, tracker_cfg: Optional[BaseTrackerConfig] = None) -> Iterator[th.Tensor]:
        """Follow a world-frame ``(x, y, yaw)(t)`` reference with the joystick-like tracker."""
        tracker = BaseTrajectoryTracker(ref, self.dt, tracker_cfg or self.cfg.base_tracker)
        self.state.hold_current(arms=True, trunk=True)
        while True:
            v, done = tracker.step(self.base_pose2d())
            self.state.set_base_velocity(float(v[0]), float(v[1]), float(v[2]))
            yield self._emit()
            if done:
                break
            if tracker.timed_out:
                d, dyaw = tracker.error(self.base_pose2d())
                raise SkillError("execution", f"base tracking timed out (pos err {d:.3f} m, yaw err {dyaw:.3f} rad)")
        self.state.stop_base()
        yield from self.hold(3)

    # ------------------------------------------------------------------ composites

    def fingertip_length(self, arm: str) -> float:
        """Distance from the eef frame origin to the fingertips along the approach axis (m)."""
        try:
            lens = self.robot.eef_to_fingertip_lengths[arm]
            return float(max(abs(float(v)) for v in lens.values()))
        except Exception:
            return 0.03

    def grasp_candidates(self, obj, arm: Optional[str] = None, approach: str = "top", offset: Optional[th.Tensor] = None, n_yaw: int = 8, tilts: Sequence[float] = (0.0, 30.0, 45.0)) -> List[Pose]:
        """Candidate grasp poses on an object, ordered by preference.

        ``top``: eef approach axis pointing down at the object's AABB centre (clamped so that the
        fingertips stay above the object's bottom), for each ``tilt`` (degrees the approach axis is
        pitched from vertical toward the robot, i.e. reaching over the object from the robot's side)
        and ``n_yaw`` wrist yaws. ``front``: approach along the robot's forward direction with a few
        rolls. Within a tilt the first candidate keeps the current wrist yaw; the others rotate away
        from it in increasing steps.
        """
        arm = arm or self.arm
        centre = obj.aabb_center.float().clone()
        lo, hi = obj.aabb
        if offset is not None:
            centre = centre + as_tensor(offset)
        _, eq = self.eef_pose(arm)
        cands = []
        if approach == "top":
            centre[2] = max(float(centre[2]), float(lo[2]) + self.fingertip_length(arm) + 0.01)
            fwd = T.quat2mat(self.robot.get_position_orientation()[1].float())[:, 0]
            fwd = fwd / th.linalg.norm(fwd)
            for tilt in tilts:
                t = math.radians(tilt)
                z_axis = th.tensor([0.0, 0.0, -1.0]) * math.cos(t) + fwd * math.sin(t)  # down, leaning away from the robot
                base = quat_with_z_axis(z_axis, eq)
                for k in range(n_yaw):
                    # 0, +45, -45, +90, -90, ... degrees about the approach axis
                    ang = (math.pi / 4) * ((k + 1) // 2) * (1 if k % 2 == 1 else -1) if k else 0.0
                    cands.append((centre.clone(), T.quat_multiply(base, T.euler2quat(th.tensor([0.0, 0.0, ang])))))
        elif approach == "front":
            fwd = T.quat2mat(self.robot.get_position_orientation()[1].float())[:, 0]
            base = quat_with_z_axis(fwd, eq)
            for k in range(n_yaw):
                ang = (math.pi / 4) * ((k + 1) // 2) * (1 if k % 2 == 1 else -1) if k else 0.0
                cands.append((centre.clone(), T.quat_multiply(base, T.euler2quat(th.tensor([0.0, 0.0, ang])))))
        else:
            raise ValueError(approach)
        return cands

    def grasp_pose_for(self, obj, arm: Optional[str] = None, approach: str = "top", offset: Optional[th.Tensor] = None, pregrasp_dist: Optional[float] = None, torso: bool = False) -> Pose:
        """First grasp candidate (see :meth:`grasp_candidates`) whose grasp *and* pre-grasp poses are
        IK-feasible; falls back to the nominal candidate if none is."""
        arm = arm or self.arm
        pregrasp_dist = self.cfg.pregrasp_dist if pregrasp_dist is None else pregrasp_dist
        cands = self.grasp_candidates(obj, arm, approach, offset)
        self._attached(arm)
        self.backend.update_world(ignore_objects=[obj])
        emb = self._emb_for(arm, torso)
        for pos, quat in cands:
            axis = T.quat2mat(quat)[:, 2]
            pre = (pos - axis * pregrasp_dist, quat)
            if self.backend.solve_ik(emb, {M.EEF_LINK[arm]: pre}, sync_world=False) is None:
                continue
            if self.backend.solve_ik(emb, {M.EEF_LINK[arm]: (pos, quat)}, sync_world=False) is None:
                continue
            return pos, quat
        log.warning("grasp_pose_for(%s): no IK-feasible candidate; returning the nominal one", obj.name)
        return cands[0]

    def grasp(self, arm: Optional[str] = None, obj=None, approach: str = "top", grasp_pose: Optional[PoseLike] = None, pregrasp_dist: Optional[float] = None, lift_dist: Optional[float] = None, torso: bool = False) -> Iterator[th.Tensor]:
        """Pre-grasp (planned) -> open -> straight approach -> close -> lift. Uses assisted grasping."""
        arm = arm or self.arm
        obj = self.object(obj) if isinstance(obj, str) else obj
        pregrasp_dist = self.cfg.pregrasp_dist if pregrasp_dist is None else pregrasp_dist
        gp = self.resolve(grasp_pose) if grasp_pose is not None else self.grasp_pose_for(obj, arm, approach, pregrasp_dist=pregrasp_dist, torso=torso)
        lift_dist = self.cfg.lift_dist if lift_dist is None else lift_dist
        approach_axis = T.quat2mat(gp[1])[:, 2]  # eef z = approach direction
        pre = (gp[0] - approach_axis * pregrasp_dist, gp[1])
        yield from self.open_gripper(arm)
        yield from self.move_eef(arm, pre, torso=torso, ignore_objects=[obj])
        yield from self.move_eef_cartesian(arm, pose=gp, check_collisions=False, ignore_objects=[obj])
        yield from self.close_gripper(arm)
        if self.held_object(arm) is not obj:
            raise SkillError("grasp", f"{obj.name} not in hand after closing gripper", in_hand=None if self.held_object(arm) is None else self.held_object(arm).name)
        self.backend.attach_object(arm, obj)
        if lift_dist > 0:
            yield from self.move_eef_cartesian(arm, delta_pos=[0.0, 0.0, lift_dist], check_collisions=False)

    def place(self, arm: Optional[str] = None, pose: PoseLike = None, approach_dist: Optional[float] = None, retreat: bool = True, torso: bool = False, reposition: bool = True) -> Iterator[th.Tensor]:
        """Bring the held object to ``pose`` (pose of the *end-effector*): planned move above, straight
        descent, release, retreat. With ``reposition`` the robot first moves to a stance from which the
        place pose is reachable when it is not from the current one."""
        arm = arm or self.arm
        obj = self.held_object(arm)
        goal = self.resolve(pose)
        approach_dist = self.cfg.place_approach_dist if approach_dist is None else approach_dist
        above = (goal[0] + th.tensor([0.0, 0.0, approach_dist]), goal[1])
        self._attached(arm)
        self.backend.update_world()
        emb = self._emb_for(arm, torso)
        reachable = all(self.backend.solve_ik(emb, {M.EEF_LINK[arm]: p}, sync_world=False) is not None for p in (above, goal))
        if not reachable:
            if not reposition:
                raise SkillError("planning", "place pose not reachable from the current stance")
            log.info("place: place pose not reachable from here; repositioning")
            stance = self.stance_for(goal[0], arm=arm, eef_candidates=[above, goal], lateral=None)
            yield from self.go_to_stance(stance)
        yield from self.move_eef(arm, above, torso=torso)
        try:
            yield from self.move_eef_cartesian(arm, pose=goal, check_collisions=False)
        except SkillError as e:
            # straight descent not trackable (workspace boundary): fall back to a planned motion
            log.info("place: straight descent failed (%s); using a planned motion instead", e)
            yield from self.move_eef(arm, goal, torso=torso, ignore_objects=[obj] if obj is not None else ())
        yield from self.open_gripper(arm)
        if obj is not None:
            self.robot.release_grasp_immediately(arm) if self.held_object(arm) is obj else None
            self.backend.detach_object(arm)
        if retreat:
            yield from self.move_eef_cartesian(arm, delta_pos=[0.0, 0.0, approach_dist], check_collisions=False, ignore_objects=[obj] if obj is not None else ())

    def place_object_at(self, arm: Optional[str] = None, object_pose: PoseLike = None, drop_height: float = 0.01, **kw) -> Iterator[th.Tensor]:
        """Place so that the *held object* ends at ``object_pose`` (uses the current grasp transform).

        The object is released ``drop_height`` metres above the requested pose so that the grasped
        geometry does not have to be pushed into the support surface.
        """
        arm = arm or self.arm
        obj = self.held_object(arm)
        if obj is None:
            raise SkillError("no_object", "place_object_at requires a held object")
        op, oq = obj.get_position_orientation()
        eef_in_obj = compose(invert((op.float(), oq.float())), self.eef_pose(arm))  # eef pose in the object frame
        goal_obj = self.resolve(object_pose)
        goal_obj = (goal_obj[0] + th.tensor([0.0, 0.0, float(drop_height)]), goal_obj[1])
        goal_eef = compose(goal_obj, eef_in_obj)
        yield from self.place(arm, goal_eef, **kw)

    def push(self, arm: Optional[str] = None, contact_pose: PoseLike = None, direction=None, distance: float = 0.1, frame: str = "world", standoff: float = 0.05, speed: Optional[float] = None) -> Iterator[th.Tensor]:
        """Closed-gripper push: go to a standoff before ``contact_pose`` (planned), then move
        straight through the contact along ``direction`` for ``distance`` metres."""
        arm = arm or self.arm
        cp = self.resolve(contact_pose)
        d = as_tensor(direction)
        if frame == "base":
            d = T.quat2mat(self.robot.get_position_orientation()[1].float()) @ d
        d = d / th.linalg.norm(d)
        yield from self.close_gripper(arm)
        yield from self.move_eef(arm, (cp[0] - d * standoff, cp[1]))
        yield from self.move_eef_cartesian(arm, pose=(cp[0] + d * distance, cp[1]), check_collisions=False, speed=speed)

    def brace(self, arm: str, pose: PoseLike) -> Iterator[th.Tensor]:
        """Support hand: close the gripper and bring it to ``pose`` (planned)."""
        yield from self.close_gripper(arm)
        yield from self.move_eef(arm, pose)

    def retract_arm(self, arm: Optional[str] = None, q_arm: Optional[th.Tensor] = None) -> Iterator[th.Tensor]:
        """Planned joint-space motion of the arm back to a rest configuration (default: zeros)."""
        arm = arm or self.arm
        q_arm = th.zeros(7) if q_arm is None else as_tensor(q_arm)
        yield from self.move_joints({n: float(v) for n, v in zip(M.ARM_JOINTS[arm], q_arm)}, emb=self._emb_for(arm))

    # ------------------------------------------------------------------ queries

    def reachable(self, arm: str, pose: PoseLike, torso: bool = False) -> bool:
        pos, quat = self.resolve(pose)
        self._attached(arm)
        return self.backend.solve_ik(self._emb_for(arm, torso), {M.EEF_LINK[arm]: (pos, quat)}) is not None
