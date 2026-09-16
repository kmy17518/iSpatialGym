"""Layer-2 execution of the action vocabulary (``oracle_primitives_design.md`` §3.6 table, §6.1; TODO F2).

``oracle/actions.py`` (F6) declares the action classes with ``targets()`` / ``annotate()``; their ``execute()`` /
``postcondition()`` raise ``NotImplementedError`` there. This module implements both per class as a **dispatch
table** ``EXECUTORS[ActionClass] = (execute_fn, postcondition_fn)`` that the executor consults, so the two files
never collide::

    execute_fn(action, motions, state, targets, ctx) -> Iterator[23-D action]     # Layer-1 motions only
    postcondition_fn(action, state, targets, ctx) -> Optional[str]                 # None = ok, else a §6.4 category

``targets`` is ``action.targets(sidecar, state)`` evaluated by the executor **at run time** (again on a retry, from
the restored state, §3.8); ``ctx`` is a per-attempt dict an execute function may fill for its postcondition (object
pose before a push, joint value before a base-follow). ``state`` is the executor's privileged view
(:class:`PrivilegedState`): read between yields only for the postconditions and to close the loop on tracking
error, never to choose between alternatives (§3 rules).

Behaviour per action (§3.6 / §6.1):

* ``Navigate``   ``track_path(waypoints)``; post: base within tolerance of the last waypoint.
* ``Transition`` ``track_path(drive_back)`` → ``trunk_to`` → ``track_path(drive_in)``; post: trunk pair and base.
* ``SetTorso``   (``arm_to(untucked)`` when ``goal.arms`` says so) → ``trunk_to``; head-camera residual above the
  correction threshold → one corrective ``track_path`` to the recorded goal base (correction, not fallback); post:
  trunk converged, camera pose within the spec §8 tolerance (``tolerance``).
* ``Grasp``      ``gripper(open)`` → ``arm_to(pre_grasp)`` → ``arm_along(approach)`` → ``gripper(close)``; the
  assisted grasp must hold the object (``failed_grasp``), then backend ``attach_object``.
* ``Lift`` / ``Reorient``  ``arm_along`` over the recorded segment; post: still held, eef at the segment end.
* ``Place``      ``arm_to(pre_place)`` → ``arm_along(descent)`` → ``gripper(open)`` (+ ``release_grasp_immediately``,
  backend ``detach_object``) → ``arm_along(retreat)`` → settle; post: released, object at rest (``not_settled``) or
  ``attached`` for ``kind="attach"``.
* ``Push``       ``gripper(close)`` → ``arm_to(pre)`` → ``arm_along(init → mid → goal)``; post: eef at the goal
  contact, the pushed object moved (``no_motion``).
* ``Brace``      ``gripper(close)`` → ``arm_to(pre)`` → ``arm_along(pre → contact)``; held (no further command).
* ``BaseFollow`` ``base_follow(b_k)`` with the arm locked, or ``arm_along(h(q_k))`` in arm mode; post: joint at
  ``q_goal`` (spec §8 tolerance) or ``passive_from`` reached.
* ``Retreat``    ``gripper(open)`` (+ release / detach) when ``release`` → ``arm_along(retreat)``.

Tolerance categories: an eef / base / trunk that did not reach its recorded target is ``execution`` (a motion did
not converge); a task quantity off its target (camera pose, joint value) is ``tolerance``; a push / articulation
that moved nothing is ``no_motion``; a grasp the assisted grasp did not attach is ``failed_grasp``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, Tuple

import torch as th

from ispatialgym.oracle.actions import (
    Action,
    BaseFollow,
    Brace,
    Grasp,
    Lift,
    Navigate,
    Place,
    Push,
    Reorient,
    Retreat,
    SetTorso,
    Transition,
    VOCABULARY,
)
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.motions import Motions, SkillError, as_pose2d, as_world_pose
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)


class PrivilegedState(Protocol):
    """What the execute / postcondition functions read (implemented by ``executor.SimState``)."""

    robot: Any
    cam_in_zed: Any

    def object(self, name: str) -> Any: ...  # scene object of a bddl instance
    def object_pose(self, name: str) -> Tuple[th.Tensor, th.Tensor]: ...
    def object_velocity(self, name: str) -> float: ...  # |linear velocity| (m/s)
    def eef_pose(self, arm: str) -> Tuple[th.Tensor, th.Tensor]: ...
    def base_pose2d(self) -> Tuple[float, float, float]: ...
    def held_object(self, arm: str) -> Any: ...
    def joint_position(self, name: str, joint: str) -> float: ...
    def camera_pose(self) -> Tuple[th.Tensor, th.Tensor]: ...
    def trunk(self) -> Tuple[float, float]: ...
    def attached(self, child: str, parent: str) -> bool: ...


@dataclass
class ExecuteConfig:
    #: base at the last path waypoint / drive-in pose (m, rad)
    nav_pos_tol: float = 0.05
    nav_yaw_tol: float = 0.08
    #: trunk (translate, tilt) at the recorded pair
    trunk_tol: float = 0.03
    #: eef at the end of a tracked segment (m, rad)
    eef_pos_tol: float = 0.02
    eef_rot_tol: float = 0.10
    #: camera relocalization (spec §8) and the residual above which the corrective ``track_path`` runs
    camera_pos_tol: float = 0.05
    camera_rot_tol: float = 0.10
    camera_correct_pos: float = 0.02
    camera_correct_rot: float = 0.03
    #: a pushed / articulated object counts as moved above these (m, rad)
    moved_pos: float = 0.005
    moved_rot: float = 0.02
    #: a placed object counts as settled below this linear speed (m/s)
    object_settle_vel: float = 0.05
    #: settle windows after a place / base-follow (steps)
    place_settle_steps: int = 60
    follow_settle_steps: int = 45
    set_torso_settle_steps: int = 45


CONFIG = ExecuteConfig()

ExecuteFn = Callable[[Action, Motions, PrivilegedState, Dict[str, Any], Dict[str, Any]], Iterator[th.Tensor]]
PostconditionFn = Callable[[Action, PrivilegedState, Dict[str, Any], Dict[str, Any]], Optional[str]]


# ==========================================================================================
# small helpers
# ==========================================================================================


def _cfg(ctx: Dict[str, Any]) -> ExecuteConfig:
    return ctx.get("cfg") or CONFIG


def _hook(ctx: Dict[str, Any], name: str):
    """An executor hook (``ctx["hooks"][name]``; a no-op without an executor): ``fov(enabled)`` enables / disables
    the gripper-in-FOV monitor around the checked segment of the action (``qa.Monitors`` arming rule)."""
    return (ctx.get("hooks") or {}).get(name) or (lambda *a, **k: None)


def _witness(ctx: Dict[str, Any], key: str):
    """The sampler's IK witness (28-D q) of the action target ``key`` when the sidecar recorded one (``arm_to`` then
    plans to that configuration, i.e. into the branch the following approach / descent was validated from)."""
    return (ctx.get("witnesses") or {}).get(key)


def _pose_error(a, b) -> Tuple[float, float]:
    (pa, qa), (pb, qb) = as_world_pose(a), as_world_pose(b)
    return float(th.linalg.norm(pa - pb)), C.quat_angle(qa, qb)


def _base_error(cur: Tuple[float, float, float], goal) -> Tuple[float, float]:
    gx, gy, gyaw = as_pose2d(goal)
    return math.hypot(gx - cur[0], gy - cur[1]), abs(C.wrap_angle(gyaw - cur[2]))


def _eef_reached(state: PrivilegedState, arm: str, target, cfg: ExecuteConfig, what: str) -> Optional[str]:
    dp, da = _pose_error(state.eef_pose(arm), target)
    if dp > cfg.eef_pos_tol or da > cfg.eef_rot_tol:
        return "execution"
    return None


def _scene_name(state: PrivilegedState, inst: str) -> Optional[str]:
    try:
        return state.object(inst).name
    except Exception:  # noqa: BLE001  (an instance without a scene object: nothing to ignore)
        return None


#: recorded idle-arm posture names -> the 7 arm joint values (A6; ``untucked`` is the robot constant)
IDLE_POSTURES = ("untucked", "tucked", "reset", "ready")
#: an arm already within this of its recorded posture is not moved (rad)
IDLE_ARM_TOL = 0.05


def posture_q(robot, posture: str, arm: str) -> th.Tensor:
    if posture == "untucked":
        return C.as_tensor(robot.untucked_default_joint_pos)[robot.arm_control_idx[arm]].clone()
    if posture in ("tucked", "reset"):
        return th.zeros(7)
    if posture == "ready":
        return th.tensor(M.ARM_READY_Q[arm], dtype=th.float32)
    raise ValueError(f"unknown idle-arm posture {posture!r}")


def idle_arm_targets(record, state) -> Dict[str, th.Tensor]:
    """A6: the recorded joint values of the **idle** arms of a stance / goal record — ``arms_q = {arm: [7]}`` written
    by the sampler (``SamplerV2.annotate_stance_arms``), else the named ``idle_posture`` — for every arm that is not
    a working arm of the record (a posture-string ``arms`` = both arms idle). Both fields live in the F6 record's
    ``extra`` until the sidecar dataclass names them; ``{}`` when nothing is recorded."""
    extra = getattr(record, "extra", None) or {}
    arms_q = getattr(record, "arms_q", None) or extra.get("arms_q")
    posture = getattr(record, "idle_posture", None) or extra.get("idle_posture")
    working = set(getattr(record, "working_arms", []) or [])
    out: Dict[str, th.Tensor] = {}
    for arm in M.ARMS:
        if arm in working:
            continue
        if isinstance(arms_q, dict) and arms_q.get(arm) is not None:
            q = C.as_tensor(arms_q[arm]).reshape(-1)
            out[arm] = q if q.numel() == 7 else q[state.robot.arm_control_idx[arm]]
        elif posture and posture != "same":
            out[arm] = posture_q(state.robot, posture, arm)
    return out


def move_idle_arms(motions: Motions, state, record, ctx: Dict[str, Any]) -> Iterator[th.Tensor]:
    """``arm_to`` the idle arms of ``record`` to their recorded posture (A6) — before the trunk motion, so the swept
    trunk motion is the one the sampler checked with those arm values. Arms already there are left alone."""
    targets = idle_arm_targets(record, state)
    q = state.robot.get_joint_positions()
    move = {arm: qa for arm, qa in targets.items() if float(th.max(th.abs(q[state.robot.arm_control_idx[arm]].float() - qa))) > IDLE_ARM_TOL}
    ctx["idle_arms_moved"] = sorted(move)
    if move:
        yield from motions.arm_to(tuple(sorted(move)), q=move)


def camera_goal(targets: Dict[str, Any], cam_in_zed=None) -> Tuple[th.Tensor, th.Tensor]:
    """World head-camera pose of a ``SetTorso`` target (``base`` pose2d + ``trunk`` pair) through the conventions."""
    t, k = targets["trunk"]
    q4 = C.torso_q(float(t), float(k), C.TORSO_JOINT_LIMITS)
    return C.camera_pose(q4, base_pose=tuple(as_pose2d(targets["base"])), cam_in_zed=cam_in_zed)


# ==========================================================================================
# Navigate / Transition / SetTorso
# ==========================================================================================


def execute_navigate(action: Navigate, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    yield from motions.track_path(targets["waypoints"], posture=targets.get("posture"))


def post_navigate(action: Navigate, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    d, dyaw = _base_error(state.base_pose2d(), targets["goal"])
    ctx["base_error"] = (d, dyaw)
    return None if d <= cfg.nav_pos_tol and dyaw <= cfg.nav_yaw_tol else "execution"


def execute_transition(action: Transition, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    yield from motions.track_path([targets["drive_back"]])
    # A6: the arriving stance's idle arm(s) to their recorded posture before the trunk moves (at the stand-off)
    yield from move_idle_arms(motions, state, action.stance_to, ctx)
    t, k = targets["trunk"]
    yield from motions.trunk_to(float(t), float(k))
    yield from motions.track_path([targets["drive_in"]])


def post_transition(action: Transition, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    t, k = targets["trunk"]
    t_now, k_now = state.trunk()
    ctx["trunk_error"] = (abs(t_now - float(t)), abs(k_now - float(k)))
    if max(ctx["trunk_error"]) > cfg.trunk_tol:
        return "execution"
    d, dyaw = _base_error(state.base_pose2d(), targets["drive_in"])
    ctx["base_error"] = (d, dyaw)
    return None if d <= cfg.nav_pos_tol and dyaw <= cfg.nav_yaw_tol else "execution"


def execute_set_torso(action: SetTorso, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    if targets["arms"] == "untucked":
        yield from motions.arm_to(("left", "right"), q=state.robot.untucked_default_joint_pos)
    else:
        yield from move_idle_arms(motions, state, action.goal, ctx)  # A6: a recorded idle posture / arms_q on the goal
    t, k = targets["trunk"]
    yield from motions.trunk_to(float(t), float(k))
    goal = camera_goal(targets, getattr(state, "cam_in_zed", None))
    dp, da = _pose_error(state.camera_pose(), goal)
    ctx["camera_error_before_correction"] = (dp, da)
    if dp > cfg.camera_correct_pos or da > cfg.camera_correct_rot:
        # correction, not fallback: close the loop on the base tracking error with one drive to the recorded goal base
        log.info("set_torso: camera residual %.3f m / %.3f rad; one corrective track_path to the goal base", dp, da)
        yield from motions.track_path([targets["base"]])
        ctx["corrected"] = True
    yield from motions.settle(max_steps=cfg.set_torso_settle_steps)


def post_set_torso(action: SetTorso, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    t, k = targets["trunk"]
    t_now, k_now = state.trunk()
    ctx["trunk_error"] = (abs(t_now - float(t)), abs(k_now - float(k)))
    if max(ctx["trunk_error"]) > cfg.trunk_tol:
        return "execution"
    dp, da = _pose_error(state.camera_pose(), camera_goal(targets, getattr(state, "cam_in_zed", None)))
    ctx["camera_error"] = (dp, da)
    return None if dp <= cfg.camera_pos_tol and da <= cfg.camera_rot_tol else "tolerance"


# ==========================================================================================
# Grasp / Lift / Reorient / Place / Retreat
# ==========================================================================================


def execute_grasp(action: Grasp, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.arm
    _hook(ctx, "fov")(False)  # the idle posture / the transit to the pre-grasp are not the sampler's to guarantee
    yield from motions.gripper(arm, close=False)
    yield from motions.arm_to(arm, pose=targets["pre_grasp"], q_hint=_witness(ctx, "pre_grasp"))
    _hook(ctx, "fov")(True)  # from the pre-grasp on the fingertips enter the frame and must stay (recorded grasp pose in frame)
    yield from motions.arm_along(arm, targets["approach"])
    yield from motions.gripper(arm, close=True)
    obj = state.object(action.object)
    held = state.held_object(arm)
    if held is not obj:
        raise SkillError("failed_grasp", f"{action.object} not in the {arm} hand after closing", in_hand=None if held is None else held.name)
    motions.backend.attach_object(arm, obj)
    ctx["attached"] = obj.name


def post_grasp(action: Grasp, state, targets, ctx) -> Optional[str]:
    return None if state.held_object(action.arm) is state.object(action.object) else "failed_grasp"


def execute_lift(action: Lift, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    ctx["object_quat0"] = state.object_pose(action.object)[1]
    yield from motions.arm_along(action.arm, targets["lift"])


def post_lift(action: Lift, state, targets, ctx) -> Optional[str]:
    if state.held_object(action.arm) is not state.object(action.object):
        return "failed_grasp"
    return _eef_reached(state, action.arm, targets["lift"][-1], _cfg(ctx), "lift")


def execute_reorient(action: Reorient, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    yield from motions.arm_along(action.arm, targets["waypoints"])


def post_reorient(action: Reorient, state, targets, ctx) -> Optional[str]:
    if state.held_object(action.arm) is not state.object(action.object):
        return "failed_grasp"
    return _eef_reached(state, action.arm, targets["waypoints"][-1], _cfg(ctx), "reorient")


def _release(motions: Motions, state, arm: str) -> None:
    """Release whatever the arm holds (assisted grasp + planner attachment); idempotent."""
    if state.held_object(arm) is not None:
        state.robot.release_grasp_immediately(arm)
    motions.backend.detach_object(arm)


def execute_place(action: Place, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.arm
    _hook(ctx, "fov")(False)  # the transit to the pre-place (10 cm back along a tilted approach) may leave the margin
    yield from motions.arm_to(arm, pose=targets["pre_place"], q_hint=_witness(ctx, "pre_place"))
    _hook(ctx, "fov")(True)  # the descent ends at the recorded (in-frame) place pose
    yield from motions.arm_along(arm, targets["descent"])
    yield from motions.gripper(arm, close=False)
    _release(motions, state, arm)
    _hook(ctx, "fov")(False)  # the retreat leg is the same as a Retreat action: exempt
    yield from motions.arm_along(arm, targets["retreat"])
    yield from motions.settle(max_steps=cfg.place_settle_steps)


def post_place(action: Place, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    if state.held_object(action.arm) is not None:
        return "execution"
    if action.kind == "attach":
        return None if action.on and state.attached(action.object, action.on) else "execution"
    if action.kind == "support":
        v = state.object_velocity(action.object)
        ctx["object_speed"] = v
        if v > cfg.object_settle_vel:
            return "not_settled"
    return None


def execute_retreat(action: Retreat, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.arm
    if action.release:
        yield from motions.gripper(arm, close=False)
        _release(motions, state, arm)
    yield from motions.arm_along(arm, targets["retreat"])


def post_retreat(action: Retreat, state, targets, ctx) -> Optional[str]:
    if action.release and state.held_object(action.arm) is not None:
        return "execution"
    return _eef_reached(state, action.arm, targets["retreat"][-1], _cfg(ctx), "retreat")


# ==========================================================================================
# Push / Brace / BaseFollow
# ==========================================================================================


def execute_push(action: Push, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.contact.arm
    _hook(ctx, "fov")(False)
    yield from motions.gripper(arm, close=True)
    yield from motions.arm_to(arm, pose=targets["pre"], q_hint=_witness(ctx, "pre"))
    _hook(ctx, "fov")(True)
    ctx["object_pose0"] = tuple(x.clone() for x in state.object_pose(action.contact.object))
    yield from motions.arm_along(arm, targets["waypoints"])


def post_push(action: Push, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    verdict = _eef_reached(state, action.contact.arm, targets["waypoints"][-1], cfg, "push")
    if verdict is not None:
        return verdict
    before = ctx.get("object_pose0")
    if before is None:
        return None
    dp, da = _pose_error(state.object_pose(action.contact.object), before)
    ctx["object_motion"] = (dp, da)
    return None if dp > cfg.moved_pos or da > cfg.moved_rot else "no_motion"


def execute_brace(action: Brace, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.contact.arm
    _hook(ctx, "fov")(False)
    yield from motions.gripper(arm, close=True)
    yield from motions.arm_to(arm, pose=targets["pre"], q_hint=_witness(ctx, "pre"))
    _hook(ctx, "fov")(True)
    yield from motions.arm_along(arm, targets["approach"])


def post_brace(action: Brace, state, targets, ctx) -> Optional[str]:
    return _eef_reached(state, action.contact.arm, targets["contact"], _cfg(ctx), "brace")


def joint_tolerance(art) -> float:
    """spec §8: revolute max(0.05 rad, 0.10·|Δq|); prismatic max(0.01 m, 0.10·|Δq|)."""
    span = abs(art.q_goal - art.q_init)
    return max(0.05 if art.joint_type == "revolute" else 0.01, 0.10 * span)


def execute_base_follow(action: BaseFollow, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    art = action.articulation
    ctx["q0"] = state.joint_position(art.object, art.joint)
    if targets["mode"] == "base":
        if not targets.get("base_poses"):
            raise SkillError("plan_failed", "base_follow: no recorded base poses in base mode", stage="targets")
        yield from motions.base_follow(targets["base_poses"], action.arm)
    else:
        yield from motions.arm_along(action.arm, targets["handle_poses"])
    yield from motions.settle(max_steps=cfg.follow_settle_steps)


def post_base_follow(action: BaseFollow, state, targets, ctx) -> Optional[str]:
    art = action.articulation
    if state.held_object(action.arm) is not state.object(art.object):
        return "failed_grasp"
    q = state.joint_position(art.object, art.joint)
    ctx["q"] = q
    tol = joint_tolerance(art)
    span = art.q_goal - art.q_init
    sign = 1.0 if span >= 0 else -1.0
    q_target = art.q_at(art.passive_from) if art.passive_from is not None else art.q_goal
    progress = sign * (q - art.q_init)
    if progress >= sign * (q_target - art.q_init) - tol:
        return None
    return "no_motion" if progress < 0.10 * abs(span) else "tolerance"


# ==========================================================================================
# dispatch
# ==========================================================================================

EXECUTORS: Dict[type, Tuple[ExecuteFn, PostconditionFn]] = {
    Navigate: (execute_navigate, post_navigate),
    Transition: (execute_transition, post_transition),
    SetTorso: (execute_set_torso, post_set_torso),
    Grasp: (execute_grasp, post_grasp),
    Lift: (execute_lift, post_lift),
    Reorient: (execute_reorient, post_reorient),
    Place: (execute_place, post_place),
    Push: (execute_push, post_push),
    Brace: (execute_brace, post_brace),
    BaseFollow: (execute_base_follow, post_base_follow),
    Retreat: (execute_retreat, post_retreat),
}

#: actions during which the failed-grasp / contact rules of §6.4 apply (finger contacts are what manipulation is)
MANIPULATION_ACTIONS: Tuple[type, ...] = (Grasp, Lift, Reorient, Place, Push, Brace, BaseFollow, Retreat)
#: manipulation actions the gripper-in-FOV monitor does not watch: the hand backing off may leave the frame
FOV_EXEMPT_ACTIONS: Tuple[type, ...] = (Retreat,)


def executor_for(action: Action) -> Tuple[ExecuteFn, PostconditionFn]:
    """The ``(execute, postcondition)`` pair of ``action``'s class (exact class first, then its MRO)."""
    for cls in type(action).__mro__:
        if cls in EXECUTORS:
            return EXECUTORS[cls]
    raise KeyError(f"no executor for action class {type(action).__name__}")


def is_manipulation(action: Action) -> bool:
    return isinstance(action, MANIPULATION_ACTIONS)


def fov_monitored(action: Action) -> bool:
    """Whether the gripper-in-FOV monitor watches ``action`` (manipulation actions except the retreats; the monitor
    itself arms once the fingertips first enter the frame, ``qa.Monitors``)."""
    return isinstance(action, MANIPULATION_ACTIONS) and not isinstance(action, FOV_EXEMPT_ACTIONS)


def run_action(action: Action, motions: Motions, state, targets: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Iterator[th.Tensor]:
    """Yield the 23-D actions of ``action`` (Layer-1 motions only)."""
    fn, _ = executor_for(action)
    yield from fn(action, motions, state, targets, {} if ctx is None else ctx)


def check_postcondition(action: Action, state, targets: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """``None`` when the action achieved its recorded target, else the QA category of §6.4."""
    _, fn = executor_for(action)
    return fn(action, state, targets, {} if ctx is None else ctx)


def coverage() -> Dict[str, bool]:
    """Which classes of the F6 vocabulary have an executor (tests assert every one does)."""
    return {cls.__name__: cls in EXECUTORS for cls in VOCABULARY}


__all__ = [
    "PrivilegedState", "ExecuteConfig", "CONFIG", "EXECUTORS", "MANIPULATION_ACTIONS", "FOV_EXEMPT_ACTIONS", "executor_for",
    "is_manipulation", "fov_monitored", "run_action", "check_postcondition", "coverage", "camera_goal", "joint_tolerance",
    "idle_arm_targets", "move_idle_arms", "posture_q", "IDLE_POSTURES", "IDLE_ARM_TOL",
]
