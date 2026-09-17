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
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

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
from ispatialgym.oracle.motions import ApproachFinish, Motions, SkillError, as_pose2d, as_world_pose
from ispatialgym.oracle.sidecar import Pose
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
    #: the pad at the end of a push / brace is loaded by the object it holds (a swung painting rests on the pad, a
    #: braced board preloads the stop): the joint controller yields 2-4 cm under that load (the painting push: 31 mm
    #: one run, 43 mm the next), so their eef checks are only a sanity bound on the pad having stayed with its object;
    #: whether the object reached its goal is the epilogue's judgement
    contact_eef_pos_tol: float = 0.08
    contact_eef_rot_tol: float = 0.30
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
    #: settle windows after a place / base-follow (steps); a drop holds this long for the object to come to rest
    place_settle_steps: int = 60
    drop_settle_steps: int = 30
    follow_settle_steps: int = 45
    set_torso_settle_steps: int = 45
    #: push closure (owner rule 2026-09-16, execution accuracy): after the recorded arc / line the pad holds in contact
    #: until the object is still (at least ``push_hold_min_steps``, at most ``push_settle_max_steps``; still = below
    #: ``push_still_pos`` / ``push_still_rot`` per step for ``push_still_steps`` steps), the residual to the goal is
    #: measured and pushed once more along the extended arc / line when it lies between ``push_residual_min_*`` and
    #: ``push_residual_max_fraction`` of the recorded motion; then the pad backs off ``push_backoff_m`` along the
    #: reversed push direction before the retreat lifts it (a pad rubbing up the face dragged the object 0.004 rad)
    push_hold_min_steps: int = 10
    push_settle_max_steps: int = 45
    push_still_pos: float = 2e-4
    push_still_rot: float = 1e-3
    push_still_steps: int = 5
    push_residual_min_rad: float = 0.02
    push_residual_min_m: float = 0.005
    push_residual_max_fraction: float = 0.25
    push_backoff_m: float = 0.02
    #: approach / descent finish (owner rule 2026-09-16): when the straight differential-IK segment stalls with a
    #: residual below ``approach_finish_max_stall`` (m) and no more than ``approach_finish_max_remaining`` (m) of the
    #: segment left, the converged prefix is executed and the last part is planned by ``arm_to`` to the end pose — to
    #: the sidecar's IK witness when it has one, else cuRobo's own IK branch — instead of failing ``plan_failed``
    approach_finish_max_stall: float = 0.015
    approach_finish_max_remaining: float = 0.06
    #: free-space legs (lift / retreat) are finished by ``arm_to`` to their end pose from any stall (no contact made)
    free_leg_finish_max_stall: float = 0.03
    free_leg_finish_max_remaining: float = 0.6
    #: a contact pre-pose next to an ignored support (wall, table) is reached via a standoff this far away from the
    #: support, planned with the support present, then a straight leg (``reach_pre_pose``)
    pre_pose_standoff_m: float = 0.08


CONFIG = ExecuteConfig()

ExecuteFn = Callable[[Action, Motions, PrivilegedState, Dict[str, Any], Dict[str, Any]], Iterator[th.Tensor]]
PostconditionFn = Callable[[Action, PrivilegedState, Dict[str, Any], Dict[str, Any]], Optional[str]]


# ==========================================================================================
# small helpers
# ==========================================================================================


def _cfg(ctx: Dict[str, Any]) -> ExecuteConfig:
    return ctx.get("cfg") or CONFIG


def contact_ignores(state, name: str, extra: Sequence[str] = ()) -> List[Any]:
    """Scene objects left out of the planner's collision world for the pre-pose / finishing plans of a contact
    action on ``name`` (the sampler's own rule, spec §3.6 / owner 2026-09-16): the container or support the object
    rests in / on / against (``SimState.contact_supports``: the source tray of disembedding, the table, the wall a
    painting hangs on) plus the instances in ``extra`` (a recorded ``grasps[].grasp_from``, the place ``on``)."""
    out: List[Any] = []
    getter = getattr(state, "contact_supports", None) or getattr(state, "touching", None)
    if getter is not None:
        try:
            out.extend(getter(name))
        except Exception:  # noqa: BLE001
            pass
    for inst in extra:
        if not inst:
            continue
        try:
            o = state.object(inst)
        except Exception:  # noqa: BLE001
            continue
        if not any(o is x for x in out):
            out.append(o)
    return out


def escape_direction(state, pose, objects: Sequence[Any], near: float = 0.15) -> Optional[th.Tensor]:
    """Unit direction from the ignored supports **near** ``pose`` (within ``near`` of its position; a floor 1.6 m below
    a wall brace does not count) to the pose: the sum of the vectors from the nearest point of each object's AABB to
    the pose position — away from a wall horizontally, up from a table; None when no support is near."""
    p = as_world_pose(pose)[0]
    total = th.zeros(3)
    for o in objects:
        try:
            lo, hi = o.aabb
        except Exception:  # noqa: BLE001
            continue
        lo, hi = C.as_tensor(lo).float(), C.as_tensor(hi).float()
        nearest = th.minimum(th.maximum(p, lo), hi)
        v = p - nearest
        dist = float(th.linalg.norm(v))
        if dist > near:
            continue
        if dist < 1e-4:  # the pose sits inside the AABB: leave through the nearest face
            gaps = th.cat([p - lo, hi - p])
            k = int(th.argmin(gaps))
            v = th.zeros(3)
            v[k % 3] = -1.0 if k < 3 else 1.0
        total = total + v / max(float(th.linalg.norm(v)), 1e-9)
    n = float(th.linalg.norm(total))
    return None if n < 1e-6 else total / n


def reach_pre_pose(motions: Motions, state, arm: str, pre, supports: Sequence[Any], through: Sequence[Any], q_hint, ctx: Dict[str, Any], cfg: "ExecuteConfig") -> Iterator[th.Tensor]:
    """Reach a contact pre-pose that the sampler checked with its supports ignored (a wall / table 2 cm from the
    pads). The supports leave the collision world only for the **last straight leg**: the trajopt goes to a standoff
    ``pre + standoff · n`` (``n`` away from the supports) with the supports present — a trajopt with the wall ignored
    planned the hand *through* the wall region and the arm jammed against it (1.39 rad joint error on the painting
    push) — then a straight ``arm_along`` closes the standoff. ``through`` (the pushed target) is ignored for both."""
    if not supports:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through))
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    n = escape_direction(state, pre, supports)
    if n is None:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(supports) + list(through))
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    p, q = as_world_pose(pre)
    standoff = (p + n * cfg.pre_pose_standoff_m, q)
    ctx["pre_pose_standoff"] = {"direction": [round(float(v), 3) for v in n.tolist()], "distance_m": cfg.pre_pose_standoff_m}
    yield from motions.arm_to(arm, pose=standoff, ignore_objects=list(through))
    if getattr(motions, "last_relaxed_links", None):
        ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
    yield from motions.arm_along(arm, [pre], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)


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


def _eef_reached(state: PrivilegedState, arm: str, target, cfg: ExecuteConfig, what: str, loaded: bool = False) -> Optional[str]:
    dp, da = _pose_error(state.eef_pose(arm), target)
    pos_tol, rot_tol = (cfg.contact_eef_pos_tol, cfg.contact_eef_rot_tol) if loaded else (cfg.eef_pos_tol, cfg.eef_rot_tol)
    if dp > pos_tol or da > rot_tol:
        log.warning("%s: the %s hand is %.1f mm / %.3f rad off its end pose (tolerance %.0f mm / %.2f rad)", what, arm, dp * 1000, da, pos_tol * 1000, rot_tol)
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


def stance_arm_targets(record, state, include_working: bool = True) -> Dict[str, th.Tensor]:
    """The recorded arm joint values of a stance / goal record — ``arms_q = {arm: [7]}`` written by the sampler
    (``SamplerV2.annotate_stance_arms``), else the named posture (``arm_posture`` / ``idle_posture``: untucked /
    tucked / reset / ready) — for every arm (owner decision 2026-09-16 10:47: the working arms too — the wall two-arm
    stance records ``arm_posture: tucked`` for both arms and the drive-in must happen in that posture; the working arm
    reaches its pre-pose from there in the next action), or only for the idle arms with ``include_working=False``.
    The fields live in the F6 record's ``extra``; ``{}`` when nothing is recorded."""
    extra = getattr(record, "extra", None) or {}
    arms_q = getattr(record, "arms_q", None) or extra.get("arms_q")
    posture = getattr(record, "idle_posture", None) or extra.get("arm_posture") or extra.get("idle_posture")
    working = set(getattr(record, "working_arms", []) or [])
    out: Dict[str, th.Tensor] = {}
    for arm in M.ARMS:
        if arm in working and not include_working:
            continue
        if isinstance(arms_q, dict) and arms_q.get(arm) is not None:
            q = C.as_tensor(arms_q[arm]).reshape(-1)
            out[arm] = q if q.numel() == 7 else q[state.robot.arm_control_idx[arm]]
        elif posture and posture != "same":
            out[arm] = posture_q(state.robot, posture, arm)
    return out


def idle_arm_targets(record, state) -> Dict[str, th.Tensor]:
    """The recorded joint values of the **idle** arms only (see :func:`stance_arm_targets`)."""
    return stance_arm_targets(record, state, include_working=False)


def stance_fingers(record) -> Optional[float]:
    """``stances.<name>.fingers``: the finger opening the stance's checks used (written when it is not the open
    default; closed pads for the wall two-arm stance), None when not recorded."""
    extra = getattr(record, "extra", None) or {}
    f = getattr(record, "fingers", None)
    if f is None:
        f = extra.get("fingers")
    return None if f is None else float(f)


def move_idle_arms(motions: Motions, state, record, ctx: Dict[str, Any]) -> Iterator[th.Tensor]:
    """Put the arms into the stance's recorded drive-in posture before the trunk motion / drive-in (A6 + owner
    decision 2026-09-16 10:47): the recorded finger opening (``fingers``: closed pads for the wall two-arm stance) and
    the recorded joints (``arms_q`` / ``arm_posture``) on **every arm that holds nothing** — working arms included —
    so the swept trunk motion and the drive-in happen in the posture the sampler checked (the wall stance at
    d = 0.55 m was rejected untucked and recorded tucked). A carried object keeps its grasp and its arm. Arms already
    there are left alone. Notes: ``arms_posture`` = {moved, kept (holding), source}, ``fingers_set``."""
    holding = [arm for arm in M.ARMS if state.held_object(arm) is not None]
    fingers = stance_fingers(record)
    set_fingers = []
    if fingers is not None:
        want_close = fingers <= 0.5 * float(C.FINGER_OPEN)
        for arm in M.ARMS:
            if arm in holding:
                continue  # a carried object keeps its grasp
            if motions.gripper_closed(arm) != want_close:
                set_fingers.append(arm)
                yield from motions.gripper(arm, close=want_close)
    ctx["fingers_set"] = {"arms": set_fingers, "closed": (fingers is not None and fingers <= 0.5 * float(C.FINGER_OPEN))} if fingers is not None else None
    targets = stance_arm_targets(record, state, include_working=True)
    q = state.robot.get_joint_positions()
    move = {arm: qa for arm, qa in targets.items() if arm not in holding and float(th.max(th.abs(q[state.robot.arm_control_idx[arm]].float() - qa))) > IDLE_ARM_TOL}
    extra = getattr(record, "extra", None) or {}
    source = "arms_q" if extra.get("arms_q") else (extra.get("arm_posture") or extra.get("idle_posture") or None)
    ctx["arms_posture"] = {"moved": sorted(move), "kept_holding": [a for a in holding if a in targets], "source": source}
    ctx["idle_arms_moved"] = sorted(a for a in move if a not in set(getattr(record, "working_arms", []) or []))
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


def grasp_pinch_width(action: Grasp) -> Optional[float]:
    """The pinch width (m) of a library grasp — ``Grasp.meta["width"]`` of the library candidate (15.9 mm for
    ``marker__xakoic/grasp_1``); None for a cube / handle grasp or an older library entry without it (fully open)."""
    if action.kind != "library" or action.grasp is None:
        return None
    try:
        width = (C.grasp_entry(action.grasp.grasp_key, action.grasp.grasp_name).meta or {}).get("width")
    except Exception:  # noqa: BLE001  (a library entry the run cannot resolve keeps the full opening)
        return None
    try:
        width = float(width)
    except (TypeError, ValueError):
        return None
    return width if math.isfinite(width) and width > 0 else None


def execute_grasp(action: Grasp, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.arm
    _hook(ctx, "fov")(False)  # the idle posture / the transit to the pre-grasp are not the sampler's to guarantee
    yield from motions.gripper(arm, close=False)
    # the container / support the object rests in (the source tray of disembedding, the table; ``grasps[].grasp_from``
    # when recorded) leaves the collision world for the grasp poses, like the sampler's check (spec §3.6)
    grasp_from = (getattr(action.grasp, "extra", None) or {}).get("grasp_from") if action.grasp is not None else None
    supports = contact_ignores(state, action.object, extra=(grasp_from,) if grasp_from else ())
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in supports]
    yield from motions.arm_to(arm, pose=targets["pre_grasp"], q_hint=_witness(ctx, "pre_grasp"), ignore_objects=supports)
    _hook(ctx, "fov")(True)  # from the pre-grasp on the fingertips enter the frame and must stay (recorded grasp pose in frame)
    obj = state.object(action.object)
    cfg = _cfg(ctx)
    finish = ApproachFinish(targets["grasp"], q_hint=_witness(ctx, "grasp"), ignore_objects=[obj] + list(supports), max_stall=cfg.approach_finish_max_stall,
                            max_remaining=cfg.approach_finish_max_remaining)
    # pre-shaped opening (owner 2026-09-16 17:10 item 4, closure probe): pads at pinch width / 2 + margin before the descent,
    # so the lower pad of a tilted grasp clears the table (the fully open pad stood on it: approach 9 mm short, one pad 3
    # steps late, the cylinder rolled 19-77 deg) and the tap-to-pinch gap is <= 1 cm per side
    width = grasp_pinch_width(action)
    if width is not None and motions.cfg.gripper_preshape_margin is not None:
        opening = 0.5 * width + float(motions.cfg.gripper_preshape_margin)
        yield from motions.gripper_opening(arm, opening)
        ctx["preshape_opening_m"] = opening
    yield from motions.arm_along(arm, targets["approach"], finish=finish)
    if finish.used:
        ctx["approach_finish"] = finish.info
    # ramped closure (2.25 mm / step, a quarter of the free speed): both pads reach the object in the same step
    yield from motions.gripper(arm, close=True, rate=motions.cfg.gripper_close_rate)
    ctx["closure_rate_m_per_step"] = motions.cfg.gripper_close_rate
    held = state.held_object(arm)
    if held is not obj:
        raise SkillError("failed_grasp", f"{action.object} not in the {arm} hand after closing", in_hand=None if held is None else held.name)
    motions.backend.attach_object(arm, obj)
    ctx["attached"] = obj.name


def post_grasp(action: Grasp, state, targets, ctx) -> Optional[str]:
    return None if state.held_object(action.arm) is state.object(action.object) else "failed_grasp"


def _free_leg_finish(state, end_pose, witness_q, ignore: Sequence[Any], cfg: ExecuteConfig) -> ApproachFinish:
    """The finish of a free-space leg (lift, retreat): a differential-IK stall anywhere on it is completed by
    ``arm_to`` to the leg's end pose (no contact is being made, a curved finish is fine)."""
    return ApproachFinish(end_pose, q_hint=witness_q, ignore_objects=list(ignore), max_stall=cfg.free_leg_finish_max_stall, max_remaining=cfg.free_leg_finish_max_remaining)


def execute_lift(action: Lift, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    ctx["object_quat0"] = state.object_pose(action.object)[1]
    # the supports the object leaves (the table, a tray) may sit within the planner's activation distance of the
    # fingers at the grasp: out of the collision world for the finish, like the sampler's lift check
    finish = _free_leg_finish(state, targets["lift"][-1], _witness(ctx, "lift"), contact_ignores(state, action.object), cfg)
    yield from motions.arm_along(action.arm, targets["lift"], finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if finish.used:
        ctx["lift_finish"] = finish.info


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


def offset_brace(brace: Brace, offset_m: float) -> Brace:
    """The brace with its ``pre`` / ``contact`` poses moved ``offset_m`` **outward** along −``direction`` (the recorded
    approach points into the face): +5 / +10 mm let a short target rotate further, −5 / −10 mm stop an overshoot
    earlier (the brace retry ladder, owner 2026-09-16)."""
    c = brace.contact
    d = C.as_tensor(c.direction).reshape(-1)[:3].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    shift = -d * float(offset_m)

    def moved(pose):
        p, q = as_world_pose(pose)
        return Pose(position=[float(v) for v in (p + shift).tolist()], orientation=[float(v) for v in q.tolist()])

    contact = replace(c, pre=moved(c.pre), contact=moved(c.contact), extra={**c.extra, "retry_offset_m": float(offset_m)})
    return Brace(contact=contact, base_yaw=brace.base_yaw)


def intended_contacts(action: Action) -> Tuple[str, ...]:
    """The bddl instances an action may touch (the QA collision whitelist): its ``objects`` plus ``touches`` (the floor
    of a push-under), except that a ``Place(kind="drop")`` must not touch its container (owner decision 2026-09-16:
    release above the rim)."""
    objs = tuple(getattr(action, "objects", ()) or ()) + tuple(getattr(action, "touches", ()) or ())
    forbidden = set(forbidden_contacts(action))
    return tuple(o for o in objs if o not in forbidden)


def forbidden_contacts(action: Action) -> Tuple[str, ...]:
    """Instances no robot link — fingers included — and no held object may touch during the action, whatever the
    per-link exemptions: the container of a drop (a bowl lifted by the opening fingers is a ``collision``) and the
    action's ``avoid`` list (the partner of a push-under, during the drop and the push)."""
    out: List[str] = []
    if isinstance(action, Place) and action.kind == "drop" and action.on:
        out.append(action.on)
    out.extend(getattr(action, "avoid", ()) or ())
    return tuple(dict.fromkeys(out))


def execute_place(action: Place, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.arm
    drop = action.kind == "drop"
    _hook(ctx, "fov")(False)  # the transit to the pre-place (10 cm back along a tilted approach) may leave the margin
    on = []
    if getattr(action, "on", None) and not drop:  # a drop keeps its container in the collision world: it must not be touched
        if action.relation in ("on", "into", "under") or action.kind == "attach":
            on = contact_ignores(state, action.on, extra=(action.on,))  # the destination support / container and what it rests on
        else:
            on = contact_ignores(state, action.on)  # next to / touching: the partner stays an obstacle, only the shared support leaves
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in on]
    yield from motions.arm_to(arm, pose=targets["pre_place"], q_hint=_witness(ctx, "pre_place"), ignore_objects=on)
    _hook(ctx, "fov")(True)  # the descent ends at the recorded (in-frame) place pose
    finish = ApproachFinish(targets["place"], q_hint=_witness(ctx, "place"), ignore_objects=on, max_stall=cfg.approach_finish_max_stall, max_remaining=cfg.approach_finish_max_remaining)
    yield from motions.arm_along(arm, targets["descent"], finish=finish)
    if finish.used:
        ctx["descent_finish"] = finish.info
    # open at the recorded place pose: a drop releases right there (above the rim, no contact with the container), a
    # support place sets the object down
    yield from motions.gripper(arm, close=False)
    _release(motions, state, arm)
    _hook(ctx, "fov")(False)  # the retreat leg is the same as a Retreat action: exempt
    # retreat straight back along -a (up for a top-down pose) before any lateral motion
    yield from motions.arm_along(arm, targets["retreat"])
    if drop:
        yield from motions.hold(cfg.drop_settle_steps)  # the object falls and comes to rest in the container
    else:
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
    cfg = _cfg(ctx)
    arm = action.arm
    if action.release:
        yield from motions.gripper(arm, close=False)
        _release(motions, state, arm)
    ignore: List[Any] = []
    if getattr(action, "object", None):
        try:
            ignore = [state.object(action.object)] + contact_ignores(state, action.object)  # the released object and its support
        except Exception:  # noqa: BLE001
            ignore = []
    finish = _free_leg_finish(state, targets["retreat"][-1], _witness(ctx, "retreat"), ignore, cfg)
    yield from motions.arm_along(arm, targets["retreat"], finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if finish.used:
        ctx["retreat_finish"] = finish.info


def post_retreat(action: Retreat, state, targets, ctx) -> Optional[str]:
    if action.release and state.held_object(action.arm) is not None:
        return "execution"
    # a pad leaving a brace / push contact springs back by the joint error it carried under the load (2-3 cm)
    loaded = isinstance(getattr(action, "after", None), (Brace, Push))
    return _eef_reached(state, action.arm, targets["retreat"][-1], _cfg(ctx), "retreat", loaded=loaded)


# ==========================================================================================
# Push / Brace / BaseFollow
# ==========================================================================================


@dataclass
class PushGeometry:
    """The recorded push as a motion of the pad: an ``arc`` about the axis through ``pivot`` along ``axis`` (the
    sampler's hinge: a vertical axis at the settled COM for a tabletop yaw correction, spec §6.10; the horizontal hook
    axis of a hanging painting) through ``init, mid, goal`` — ``angle`` the signed swept angle about ``axis``,
    ``radius`` — or a straight ``line`` of ``distance`` along ``direction``. The axis is oriented +z-most (a tabletop
    arc keeps the yaw sign convention), a horizontal axis so that the swept angle is positive."""

    kind: str
    direction: th.Tensor
    pivot: Optional[th.Tensor] = None
    radius: float = 0.0
    angle: float = 0.0
    distance: float = 0.0
    axis: th.Tensor = field(default_factory=lambda: th.tensor([0.0, 0.0, 1.0]))

    @property
    def total(self) -> float:
        return abs(self.angle) if self.kind == "arc" else self.distance


def _circle_through(p0: th.Tensor, p1: th.Tensor, p2: th.Tensor, min_sagitta: float = 5e-4) -> Optional[Tuple[th.Tensor, float, th.Tensor]]:
    """Circumcentre, radius and unit plane normal of three points in 3-D; None when they are (nearly) collinear (the
    middle point less than ``min_sagitta`` off the chord)."""
    a, b = p1 - p0, p2 - p0
    n = th.linalg.cross(a, b)
    nn = float(th.linalg.norm(n))
    chord = float(th.linalg.norm(b))
    if chord < 1e-9 or nn / chord < min_sagitta:
        return None
    c = p0 + th.linalg.cross(float(th.dot(a, a)) * b - float(th.dot(b, b)) * a, n) / (2.0 * nn * nn)
    return c, float(th.linalg.norm(c - p0)), n / nn


def _signed_angle_about(u: th.Tensor, v: th.Tensor, axis: th.Tensor) -> float:
    """Signed angle from ``u`` to ``v`` about ``axis`` (both projected onto the plane normal to the axis)."""
    u = u - axis * float(th.dot(u, axis))
    v = v - axis * float(th.dot(v, axis))
    return math.atan2(float(th.dot(axis, th.linalg.cross(u, v))), float(th.dot(u, v)))


def push_geometry(waypoints: Sequence[Any], direction: Sequence[float]) -> PushGeometry:
    pts = [as_world_pose(w)[0].float() for w in waypoints]
    direction = C.as_tensor(direction).reshape(-1)[:3].float()
    if len(pts) >= 3:
        circ = _circle_through(pts[0], pts[len(pts) // 2], pts[-1])
        if circ is not None:
            pivot, r, axis = circ
            if abs(float(axis[2])) > 1e-3:
                if float(axis[2]) < 0:
                    axis = -axis
            angle = sum(_signed_angle_about(a - pivot, b - pivot, axis) for a, b in zip(pts[:-1], pts[1:]))
            if abs(float(axis[2])) <= 1e-3 and angle < 0:
                axis, angle = -axis, -angle
            return PushGeometry("arc", direction, pivot=pivot, radius=r, angle=angle, axis=axis)
    dist = sum(float(th.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
    return PushGeometry("line", direction, distance=dist)


def _rotate_pose_about(pose, pivot: th.Tensor, angle: float, axis: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
    """A world pose rotated by ``angle`` (rad) about the axis through ``pivot`` along ``axis`` (+z by default)."""
    p, q = as_world_pose(pose)
    axis = th.tensor([0.0, 0.0, 1.0]) if axis is None else axis / max(float(th.linalg.norm(axis)), 1e-9)
    rot = C.quat_from_axis_angle(axis, angle)
    pivot = C.as_tensor(pivot).float().reshape(-1)
    if pivot.numel() == 2:
        pivot = th.tensor([float(pivot[0]), float(pivot[1]), float(p[2])])
    return pivot + C.quat_rotate(rot, p - pivot), C.qmul(rot, q)


def _rotation_about(from_q: th.Tensor, to_q: th.Tensor, axis: th.Tensor) -> float:
    """The rotation (rad, signed about ``axis``) of the world rotation taking orientation ``from_q`` to ``to_q``:
    the axis-angle vector of ``to_q * from_q^-1`` projected on the axis."""
    r = C.qmul(to_q, C.qinv(from_q))
    r = r / max(float(th.linalg.norm(r)), 1e-9)
    if float(r[3]) < 0:
        r = -r
    s = float(th.linalg.norm(r[:3]))
    if s < 1e-9:
        return 0.0
    ang = 2.0 * math.atan2(s, float(r[3]))
    return float(th.dot(r[:3] / s, axis)) * ang


def _axis_residual(goal_q: th.Tensor, cur_q: th.Tensor, axis: th.Tensor, equivalences: Sequence[Any] = ()) -> float:
    """Signed rotation about ``axis`` from ``cur_q`` to the nearest equivalent of ``goal_q`` (rad)."""
    best, best_full = None, None
    for e in [None] + list(equivalences or []):
        g = goal_q if e is None else C.qmul(goal_q, C.as_tensor(e).float())
        full = C.quat_angle(g, cur_q)
        if best_full is None or full < best_full:
            best_full, best = full, _rotation_about(cur_q, g, axis)
    return float(best)


def _yaw_residual(goal_q: th.Tensor, cur_q: th.Tensor, equivalences: Sequence[Any] = ()) -> float:
    """Signed rotation about +z from ``cur_q`` to the nearest equivalent of ``goal_q`` (rad)."""
    return _axis_residual(goal_q, cur_q, th.tensor([0.0, 0.0, 1.0]), equivalences)


def push_residual(geom: PushGeometry, state, obj: str, pose0) -> Tuple[Optional[float], str]:
    """How much of the push is still missing, signed in the push's own sense: for an arc the object rotation still
    needed about +z (rad), for a line the displacement still needed along ``direction`` (m). Measured against the
    task's **goal** (the TRO pose target of the pushed object, world frame, nearest orientation equivalence) when it
    has one, else against the recorded motion (swept angle / distance minus what the object did since the push
    began). Returns ``(residual, source)``; ``(None, reason)`` when nothing can be measured."""
    p, q = state.object_pose(obj)
    p, q = C.as_tensor(p).float(), C.as_tensor(q).float()
    tgt = getattr(state, "pose_target", lambda name: None)(obj)
    world = tgt is not None and str(getattr(tgt, "frame", "world")) == "world"
    if geom.kind == "arc":
        if world and getattr(tgt, "orientation", None) is not None:
            return _axis_residual(C.as_tensor(tgt.orientation).float(), q, geom.axis, getattr(tgt, "orientation_equivalences", ())), "goal_orientation"
        if pose0 is None:
            return None, "no_reference"
        done = _rotation_about(as_world_pose(pose0)[1].float(), q, geom.axis)
        return float(geom.angle - done), "arc_angle"
    d = geom.direction / max(float(th.linalg.norm(geom.direction)), 1e-9)
    if world and getattr(tgt, "position", None) is not None:
        gp = C.as_tensor(tgt.position).float()
        return float(th.dot(gp - p, d)), "goal_position"
    if pose0 is None:
        return None, "no_reference"
    done = float(th.dot(p - as_world_pose(pose0)[0].float(), d))
    return float(geom.distance - done), "line_distance"


def extend_push(geom: PushGeometry, end_pose, residual: float, step: float = 0.05) -> List[Tuple[th.Tensor, th.Tensor]]:
    """The pad poses that continue the recorded push by ``residual`` (rad for an arc about its pivot, m for a line
    along its direction) from ``end_pose``; the arc is densified at ``step`` rad so the pad stays on the circle."""
    if geom.kind == "arc":
        n = max(1, int(math.ceil(abs(residual) / step)))
        return [_rotate_pose_about(end_pose, geom.pivot, residual * (i / n), geom.axis) for i in range(1, n + 1)]
    p, q = as_world_pose(end_pose)
    d = geom.direction / max(float(th.linalg.norm(geom.direction)), 1e-9)
    return [(p + d * residual, q)]


def push_tangent(geom: PushGeometry, pose) -> th.Tensor:
    """Unit push direction of the pad at ``pose`` (the arc's tangent in the swept sense, or the line direction)."""
    if geom.kind == "arc":
        p = as_world_pose(pose)[0].float()
        radial = p - geom.pivot
        radial = radial - geom.axis * float(th.dot(radial, geom.axis))
        t = th.linalg.cross(geom.axis, radial) * (1.0 if geom.angle >= 0 else -1.0)
        return t / max(float(th.linalg.norm(t)), 1e-9)
    d = geom.direction.clone()
    return d / max(float(th.linalg.norm(d)), 1e-9)


def _hold_until_still(motions: Motions, state, obj: str, cfg: ExecuteConfig) -> Iterator[th.Tensor]:
    """Hold the arm (pad in contact) until the object's pose stops changing between steps."""
    prev = state.object_pose(obj)
    still = 0
    for i in range(int(cfg.push_settle_max_steps)):
        yield from motions.hold(1)
        cur = state.object_pose(obj)
        dp, da = _pose_error(cur, prev)
        prev = cur
        still = still + 1 if (dp < cfg.push_still_pos and da < cfg.push_still_rot) else 0
        if i + 1 >= cfg.push_hold_min_steps and still >= cfg.push_still_steps:
            return


def execute_push(action: Push, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.contact.arm
    obj = action.contact.object
    _hook(ctx, "fov")(False)
    yield from motions.gripper(arm, close=True)
    supports = contact_ignores(state, obj)  # the wall / table the target rests against (sampler rule: ignored along the chain)
    try:
        through = [state.object(obj)]  # the pushed object leaves the collision world for the push plan (program_check._ignore_for)
    except Exception:  # noqa: BLE001
        through = []
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in list(supports) + through]
    yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, through, _witness(ctx, "pre"), ctx, cfg)
    _hook(ctx, "fov")(True)
    pose0 = tuple(x.clone() for x in state.object_pose(obj))
    ctx["object_pose0"] = pose0
    geom = push_geometry(targets["waypoints"], targets["direction"])
    if geom.kind == "arc":
        off = C.as_tensor(pose0[0]).float() - geom.pivot
        off = off - geom.axis * float(th.dot(off, geom.axis))
        ctx["push_pivot_offset_m"] = float(th.linalg.norm(off))  # the arc's axis vs the object's COM
        ctx["push_axis"] = [round(float(v), 3) for v in geom.axis.tolist()]
    yield from motions.arm_along(arm, targets["waypoints"], loaded=True)
    # 1. hold with the pad in contact until the object is still, then measure the residual to the goal
    yield from _hold_until_still(motions, state, obj, cfg)
    residual, source = push_residual(geom, state, obj, pose0)
    ctx["push_residual"] = {"kind": geom.kind, "value": residual, "source": source, "recorded": geom.total,
                            "angle_sign": (math.copysign(1.0, geom.angle) if geom.kind == "arc" else 1.0)}
    end_pose = as_world_pose(targets["waypoints"][-1])
    # 2. re-push the residual once when it is in the push's sense and the recorded arc / line allows the extension
    if residual is not None:
        forward = residual * (1.0 if geom.kind == "line" else math.copysign(1.0, geom.angle)) > 0
        ctx["push_residual"]["short"] = bool(forward)
        rmin = cfg.push_residual_min_rad if geom.kind == "arc" else cfg.push_residual_min_m
        if forward and rmin <= abs(residual) <= cfg.push_residual_max_fraction * geom.total:
            ext = extend_push(geom, end_pose, residual)
            log.info("push(%s): residual %.4f %s (%s) -> pushing on along the extended %s", obj, residual, "rad" if geom.kind == "arc" else "m", source, geom.kind)
            yield from motions.arm_along(arm, ext, loaded=True)
            yield from _hold_until_still(motions, state, obj, cfg)
            end_pose = as_world_pose(ext[-1])
            after, _ = push_residual(geom, state, obj, pose0)
            ctx["push_residual"]["extended_by"] = residual
            ctx["push_residual"]["after"] = after
        elif forward and abs(residual) > cfg.push_residual_max_fraction * geom.total:
            ctx["push_residual"]["skipped"] = "beyond the recorded arc"
        elif not forward and abs(residual) >= rmin:
            ctx["push_residual"]["skipped"] = "overshoot (a pad cannot pull)"
    # 3. back off along the reversed push direction so the retreat does not rub the pad up the object's face, then
    #    measure the residual the object is left with (a swung painting settles back onto the backed-off pad)
    tangent = push_tangent(geom, end_pose)
    back = (end_pose[0] - tangent * cfg.push_backoff_m, end_pose[1])
    yield from motions.arm_along(arm, [back], loaded=True)
    ctx["push_end_pose"] = back
    yield from _hold_until_still(motions, state, obj, cfg)
    final, _ = push_residual(geom, state, obj, pose0)
    ctx["push_residual"]["final"] = final
    p_now = C.as_tensor(state.object_pose(obj)[0]).float()
    ctx["push_object_drift_m"] = float(th.linalg.norm(p_now - C.as_tensor(pose0[0]).float()))


def post_push(action: Push, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    verdict = _eef_reached(state, action.contact.arm, ctx.get("push_end_pose", targets["waypoints"][-1]), cfg, "push", loaded=True)
    if verdict is not None:
        return verdict
    before = ctx.get("object_pose0")
    if before is None:
        return None
    dp, da = _pose_error(state.object_pose(action.contact.object), before)
    ctx["object_motion"] = (dp, da)
    return None if dp > cfg.moved_pos or da > cfg.moved_rot else "no_motion"


def execute_brace(action: Brace, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.contact.arm
    _hook(ctx, "fov")(False)
    yield from motions.gripper(arm, close=True)
    supports = contact_ignores(state, action.contact.object)  # the wall / table the braced object rests against
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in supports]
    yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, [], _witness(ctx, "pre"), ctx, cfg)
    _hook(ctx, "fov")(True)
    yield from motions.arm_along(arm, targets["approach"], loaded=True)


def post_brace(action: Brace, state, targets, ctx) -> Optional[str]:
    return _eef_reached(state, action.contact.arm, targets["contact"], _cfg(ctx), "brace", loaded=True)


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
    "idle_arm_targets", "stance_arm_targets", "move_idle_arms", "stance_fingers", "posture_q", "contact_ignores", "escape_direction", "reach_pre_pose", "IDLE_POSTURES", "IDLE_ARM_TOL", "intended_contacts", "forbidden_contacts", "offset_brace",
    "push_geometry", "push_residual", "extend_push", "push_tangent", "PushGeometry",
]
