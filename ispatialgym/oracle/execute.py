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
import os
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
    #: (owner 2026-09-17 23:39) a push leg stops once the object's residual to its TRO goal is within this fraction of the
    #: goal tolerance (arcs / lines) instead of driving the pad to the recorded end
    push_stop_at_goal: bool = True
    push_stop_frac_arc: float = 0.15
    push_stop_frac_line: float = 0.0
    #: (rush 2026-09-17) a line push still short after the arm's extension is closed by driving the base along the push
    #: direction with the arm locked, up to this far (m; 0 disables), at this speed (m/s), to this tracker tolerance (m)
    push_base_nudge_max_m: float = 0.06
    push_base_nudge_min_m: float = 0.003
    push_base_nudge_speed: float = 0.05
    push_base_nudge_tol_m: float = 0.008
    push_base_nudge_overshoot_m: float = 0.006
    #: further nudge rounds when the object rocks back after the pad's back-off (a blade high on a box tilts it)
    push_renudge_rounds: int = 2
    #: (rush) a straight push's pad poses follow the object's actual planar offset from its recorded start, up to this (m; 0 = off)
    push_recentre_max_m: float = 0.06
    #: (owner 2026-09-18 01:33) a recorded top-down blade is raised so its palm stands this far above the object's top (m; < 0 = off),
    #: as long as the finger section still covers this much of the face
    push_palm_clear_m: float = 0.005
    push_palm_min_face_m: float = 0.03
    #: first choice for a palm overlap: pitch the blade about its fingertips by this (deg; 0 = never), palm leaning away from the face
    push_palm_pitch_deg: float = 0.0  # 30 deg pitched the napkins box 15 deg (0918-023627); the raise keeps it within 1-6 deg
    #: (rush) a place whose hand ends horizontally short of the recorded place pose by (min, max] is closed with the base, arm locked
    place_base_nudge_min_m: float = 0.005
    place_base_nudge_max_m: float = 0.04
    place_base_nudge_tol_m: float = 0.004
    #: approach / descent finish (owner rule 2026-09-16): when the straight differential-IK segment stalls with a
    #: residual below ``approach_finish_max_stall`` (m) and no more than ``approach_finish_max_remaining`` (m) of the
    #: segment left, the converged prefix is executed and the last part is planned by ``arm_to`` to the end pose — to
    #: the sidecar's IK witness when it has one, else cuRobo's own IK branch — instead of failing ``plan_failed``
    approach_finish_max_stall: float = 0.015
    #: the whole 10 cm approach / descent may be finished (the touching descent stalled after 2 cm against joint 2's
    #: limit with 8 cm left; the finish plans to the recorded end witness, collision-checked but for the ignored supports)
    approach_finish_max_remaining: float = 0.10
    #: free-space legs (lift / retreat) are finished by ``arm_to`` to their end pose from any stall (no contact made)
    free_leg_finish_max_stall: float = 0.03
    free_leg_finish_max_remaining: float = 0.6
    #: a free-space retreat whose finish is rejected is accepted when it ends within this of its end pose (m)
    retreat_partial_accept_m: float = 0.04
    #: a pad leaving a brace / push contact first moves this far straight away from the supports (escape_direction) (m; 0 = off)
    contact_retreat_escape_m: float = 0.05
    #: a contact pre-pose next to an ignored support (wall, table) is reached via a standoff this far away from the
    #: support, planned with the support present, then a straight leg (``reach_pre_pose``)
    pre_pose_standoff_m: float = 0.08
    #: the carry leg lift end -> pre-place (FOV rule (b), owner 2026-09-16 17:10): ``"straight"`` = a straight Cartesian
    #: leg (through the recorded ``grasps[k].carry`` waypoint when the sampler wrote one), differential IK with the
    #: free-leg finish, both ends recorded in frame; ``"trajopt"`` = the former free trajopt transit (the regression
    #: reference: nextto 0916-152503 swung 40 cm back toward the robot and left the frame for 106 steps)
    carry_leg: str = "straight"


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
    q_standoff = standoff_configuration(motions, arm, pre, standoff, q_hint) if q_hint is not None else None
    if q_standoff is not None:
        # the standoff in the **witness's IK branch**: differential IK from the recorded pre-pose witness back along the
        # short leg, then a joint-space plan there - a pose plan let cuRobo pick its own branch and the whole contact
        # chain (arc, back-off, retreat) ran 3.18 rad away from the sampler's witnesses (axial retreat 0916-202403)
        ctx["pre_pose_standoff"]["branch"] = "witness"
        yield from motions.arm_to(arm, q=q_standoff, ignore_objects=list(through))
    else:
        ctx["pre_pose_standoff"]["branch"] = "planner"
        yield from motions.arm_to(arm, pose=standoff, ignore_objects=list(through))
    if getattr(motions, "last_relaxed_links", None):
        ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
    yield from motions.arm_along(arm, [pre], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)


def standoff_configuration(motions: Motions, arm: str, pre, standoff, q_hint) -> Optional[th.Tensor]:
    """The joint configuration (28-D) of ``standoff`` in the branch of the pre-pose witness ``q_hint``: differential IK
    from the witness along the straight leg pre -> standoff (no collision check: the leg is the one the hand will
    retrace). None when the backend cannot track it (the caller plans the pose instead)."""
    backend = getattr(motions, "backend", None)
    solve = getattr(backend, "solve_ik_path", None)
    if solve is None:
        return None
    try:
        # the witness's base / torso entries are the sampler session's own (odometry frame): only its arm joints are
        # the witness, everything else is the live configuration (as arm_to does)
        q0 = C.as_tensor(backend.q_full()).reshape(-1).float().clone()
        idx = motions.robot.arm_control_idx[arm]
        q0[idx] = C.as_tensor(q_hint).reshape(-1).float()[idx]
        dense = motions._densify(as_world_pose(pre), [standoff])
        frame = M.EEF_LINK[arm]
        res = solve(f"{arm}_arm", [{frame: wp} for wp in dense], q_start=q0, pos_tol=motions.cfg.diff_ik_pos_tol, rot_tol=motions.cfg.diff_ik_rot_tol,
                    max_iters=motions.cfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
        if not res.success or res.q_traj is None or len(res.q_traj) == 0:
            log.warning("standoff configuration from the witness not trackable (%s): planning the standoff pose instead", res.status)
            return None
        return C.as_tensor(res.q_traj[-1]).reshape(-1).float()
    except Exception as e:  # noqa: BLE001
        log.warning("standoff configuration from the witness failed (%r): planning the standoff pose instead", e)
        return None


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


def shorten_push(push: Push, shorten_rad: float) -> Push:
    """(rush 2026-09-17) The arc push with its swept angle reduced by ``shorten_rad``: init stays, mid / goal are the
    recorded init rotated about the recorded arc's pivot by half / all of the shortened angle (painting: the pad's arc
    is about the hook while the painting hinges at its centre, so it rotates ~0.25 rad more than the pad sweeps; an
    overshoot with no brace catch is corrected by pushing less, not by moving the brace)."""
    c = push.contact
    if c.mid is None or shorten_rad <= 0:
        return push
    geom = push_geometry(c.waypoints, c.direction)
    if geom.kind != "arc":
        return push
    angle = float(geom.angle)
    new_angle = math.copysign(max(abs(angle) - float(shorten_rad), 0.05), angle)

    def rotated(frac: float) -> Pose:
        p, q = _rotate_pose_about(c.init, geom.pivot, new_angle * frac, geom.axis)
        return Pose(position=[float(v) for v in p.tolist()], orientation=[float(v) for v in q.tolist()])

    contact = replace(c, mid=rotated(0.5), goal=rotated(1.0), extra={**c.extra, "shortened_rad": float(shorten_rad), "swept_rad": new_angle})
    return Push(contact=contact, base_yaw=push.base_yaw)


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
    # the FOV span opened at the pre-grasp stays open through the carry (rule (b)): no hook here
    on = []
    if getattr(action, "on", None) and not drop:  # a drop keeps its container in the collision world: it must not be touched
        if action.relation in ("on", "into", "under") or action.kind == "attach":
            on = contact_ignores(state, action.on, extra=(action.on,))  # the destination support / container and what it rests on
        else:
            on = contact_ignores(state, action.on)  # next to / touching: the partner stays an obstacle, only the shared support leaves
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in on]
    carry = list(targets.get("carry") or ())
    if cfg.carry_leg == "straight" and carry:
        # the carry leg: a straight Cartesian leg from where the hand is (the lift end) to the recorded pre-place, through
        # the recorded carry waypoint when there is one; both ends are in frame, so the fingertips stay in frame along it
        # (the free trajopt transit swung the nextto marker 40 cm back toward the robot and out of the frame). A stall
        # is finished by ``arm_to`` to the pre-place, as for the lift / retreat legs.
        finish = _free_leg_finish(state, targets["pre_place"], _witness(ctx, "pre_place"), on, cfg)
        yield from motions.arm_along(arm, carry, finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
        ctx["carry_leg"] = {"kind": "straight", "waypoints": len(carry), "via_recorded": len(targets.get("carry_via") or ()), "finish": finish.info if finish.used else None}
    else:
        yield from motions.arm_to(arm, pose=targets["pre_place"], q_hint=_witness(ctx, "pre_place"), ignore_objects=on)
        ctx["carry_leg"] = {"kind": "trajopt"}
    finish = ApproachFinish(targets["place"], q_hint=_witness(ctx, "place"), ignore_objects=on, max_stall=cfg.approach_finish_max_stall, max_remaining=cfg.approach_finish_max_remaining)
    yield from motions.arm_along(arm, targets["descent"], finish=finish)
    if finish.used:
        ctx["descent_finish"] = finish.info
    # rush 2026-09-17: the hand ends the descent short of the recorded place pose by a systematic ~1-1.5 cm (the witness
    # configuration's reach in the simulator is less than the sampler's model: no joint limit, no singularity, the DLS
    # converges onto the reach sphere); the shortfall is closed with the BASE, arm locked, before the release - the
    # recorded push (gap + tolerance + 5 mm) then covers the gap (touching 0917-223915: place 1.45 cm short, push
    # extension at the reach boundary yawed the box 30 deg)
    if cfg.place_base_nudge_max_m > 0:
        short = as_world_pose(targets["place"])[0].float() - as_world_pose(state.eef_pose(arm))[0].float()
        short[2] = 0.0
        dist = float(th.linalg.norm(short))
        if cfg.place_base_nudge_min_m < dist <= cfg.place_base_nudge_max_m:
            x, y, yaw = state.base_pose2d()
            goal2d = [x + float(short[0]), y + float(short[1]), yaw]
            log.info("place(%s): the hand is %.1f mm short of the recorded place pose -> nudging the base by that, arm locked", action.object, dist * 1000)
            q = motions.robot.get_joint_positions()
            motions.state.arm_target[arm] = q[motions.robot.arm_control_idx[arm]].clone().float()
            tcfg = replace(motions.cfg.base_tracker, v_max=(cfg.push_base_nudge_speed, cfg.push_base_nudge_speed, 0.2), pos_tol=cfg.place_base_nudge_tol_m)
            try:
                yield from motions.track_path([goal2d], tracker_cfg=tcfg)
            except SkillError as e:
                log.warning("place(%s): base nudge did not settle (%s); keeping what it achieved", action.object, e.message)
            yield from motions.settle(max_steps=cfg.place_settle_steps // 2)
            after = as_world_pose(targets["place"])[0].float() - as_world_pose(state.eef_pose(arm))[0].float()
            ctx["place_base_nudge"] = {"short_m": [round(float(v), 4) for v in short.tolist()], "after_m": round(float(th.linalg.norm(after[:2])), 4)}
    # open at the recorded place pose: a drop releases right there (above the rim, no contact with the container), a
    # support place sets the object down
    yield from motions.gripper(arm, close=False)
    _release(motions, state, arm)
    _hook(ctx, "fov")(False)  # the span closes with the release: the retreat legs are unmonitored (rule (b))
    # retreat straight back along -a (up for a top-down pose) before any lateral motion; a free-space leg after the
    # release: a differential-IK stall is finished by arm_to like the Retreat action's leg (touching 0917-203045: the
    # place retreat stalled at waypoint 2/12 with a 7 mm residual, no joint at a limit, and failed the place)
    ignore_retreat: List[Any] = list(on)
    try:
        ignore_retreat.append(state.object(action.object))  # the object just set down sits right under the pads
    except Exception:  # noqa: BLE001
        pass
    finish = _free_leg_finish(state, targets["retreat"][-1], _witness(ctx, "retreat"), ignore_retreat, cfg)
    yield from motions.arm_along(arm, targets["retreat"], finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if finish.used:
        ctx["retreat_finish"] = finish.info
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
        held = state.held_object(arm)
        if held is not None:
            articulated = bool(getattr(held, "n_joints", 0) or len(getattr(held, "joints", {}) or {}))
            # a handle of an articulated object is released with the full opening at once (no staged release: the assisted
            # grasp would re-grasp the handle during the retreat and the door cannot follow the hand)
            yield from motions.gripper(arm, close=False, staged=not articulated)
            _release(motions, state, arm)
        # a brace holds nothing: its closed fist leaves as it is (owner 2026-09-17 23:39: no gripper opening at the
        # brace retreat); the contact span still closes here
        _hook(ctx, "fov")(False)
    ignore: List[Any] = []
    if getattr(action, "object", None):
        try:
            ignore = [state.object(action.object)] + contact_ignores(state, action.object)  # the released object and its support
        except Exception:  # noqa: BLE001
            ignore = []
    finish = _free_leg_finish(state, targets["retreat"][-1], _witness(ctx, "retreat"), ignore, cfg)
    leg = list(targets["retreat"])
    if isinstance(getattr(action, "after", None), (Brace, Push)) and cfg.contact_retreat_escape_m > 0:
        # rush 2026-09-17: a pad leaving a contact first moves straight away from the supports (the wall normal / up from a
        # table: escape_direction, as the pre-pose standoff) before the leg to the recorded clear pose - that single
        # straight leg has a component along the face and rubbed the painting from +0.053 to +0.135 rad (0917-213345;
        # a back-off along the pad's own approach axis slid 9 cm down the face instead, 0917-214548)
        supports = contact_ignores(state, action.object) if getattr(action, "object", None) else []
        here = as_world_pose(state.eef_pose(arm))
        n = escape_direction(state, here, supports)
        # owner 2026-09-17 23:39: a pad leaves its contact RADIALLY - away from the object's centre in the support plane
        # (the pad at the painting's left edge goes left, the one at the top edge goes up; a table push backs off
        # opposite the push) - then off the support along the escape normal. The recorded clear pose of the wall push
        # lies 8 cm DOWN the face (the pad's approach axis is vertical there) and the transit to it dragged the
        # painting's edge (0917-224633, +0.13 rad); the postcondition is judged on the new end pose
        radial = None
        try:
            centre = C.as_tensor(state.object_pose(action.object)[0]).reshape(-1)[:3].float()
            radial = here[0].float() - centre
            if n is not None:
                radial = radial - n * float(th.dot(radial, n))
            radial = radial / float(th.linalg.norm(radial)) if float(th.linalg.norm(radial)) > 1e-3 else None
        except Exception:  # noqa: BLE001
            radial = None
        if radial is not None or n is not None:
            d1 = radial if radial is not None else n
            out = cfg.contact_retreat_escape_m
            far = max(float(getattr(action, "distance", 0.0) or 0.0), 2 * out)
            end = here[0] + d1 * far + (n * out if (n is not None and radial is not None) else 0.0)
            leg = [(here[0] + d1 * out, here[1]), (end, here[1])]
            targets["retreat"] = leg
            finish = _free_leg_finish(state, leg[-1], None, ignore, cfg)
            ctx["retreat_escape"] = {"radial": None if radial is None else [round(float(v), 3) for v in radial.tolist()],
                                     "normal": None if n is None else [round(float(v), 3) for v in n.tolist()], "distance_m": far}
    try:
        yield from motions.arm_along(arm, leg, finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    except SkillError as e:
        # rush 2026-09-17 (touching 0917-204511): the converged prefix of the leg is executed before the finish is
        # rejected (witness and cuRobo IK both in another branch); a free-space retreat that already stands within
        # retreat_partial_accept_m of its end pose is accepted as partial and judged with the loaded tolerance
        if e.reason != "plan_failed":
            raise
        dp, _ = _pose_error(state.eef_pose(arm), targets["retreat"][-1])
        if dp > cfg.retreat_partial_accept_m:
            raise
        log.warning("retreat: the leg ends %.1f mm short of its end pose (finish rejected: %s) - accepted as a partial free-space retreat", dp * 1000, e.message)
        ctx["retreat_partial_m"] = dp
    if finish.used:
        ctx["retreat_finish"] = finish.info


def post_retreat(action: Retreat, state, targets, ctx) -> Optional[str]:
    if action.release and state.held_object(action.arm) is not None:
        return "execution"
    # a pad leaving a brace / push contact springs back by the joint error it carried under the load (2-3 cm)
    loaded = isinstance(getattr(action, "after", None), (Brace, Push)) or ctx.get("retreat_partial_m") is not None
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


#: length of the closed fingers' section from the tip to the palm (m; ``sampling.v2.tabletop.PALM_FROM_TIP`` = FINGER_PROFILE[-1][0])
PALM_FROM_TIP_M = 0.07


def _pitched_blade(motions: Motions, state, arm: str, targets: Dict[str, Any], pitch: float) -> Optional[Dict[str, Any]]:
    """The recorded top-down blade push pitched by ``pitch`` (rad) about the fingertip contact line so the palm leans
    away from the pushed face: the approach axis a' = cos·(recorded a) + sin·(push direction) (from the palm toward the
    tips: down and toward the object), every eef origin re-derived from its recorded fingertip point, the closing axis
    kept, the pre-pose 10 cm back along −a'. Returned only when the collision-aware IK reaches both the pre-pose and the
    first contact pose from the current configuration (``R1ProCuRobo.solve_ik``); None otherwise."""
    try:
        d = C.as_tensor(targets["direction"]).reshape(-1)[:3].float()
        d = d / max(float(th.linalg.norm(d)), 1e-9)
        p0, q0 = as_world_pose(targets["waypoints"][0])
        a = C.approach_axis(q0)
        k = th.linalg.cross(a, d)  # horizontal axis perpendicular to the approach and the push: rotating about it leans the palm back
        if float(th.linalg.norm(k)) < 1e-6:
            return None
        rot = C.quat_from_axis_angle(k / th.linalg.norm(k), pitch)  # rot(a) = a cos + d sin: tips toward the object, palm away
        a2 = C.unit(C.quat_rotate(rot, a))
        quat = C.qmul(rot, q0)  # the closing axis (along the push for a blade) turns with it

        def re_origin(pose):
            p, q = as_world_pose(pose)
            tip = p + C.FINGERTIP_LENGTH * C.approach_axis(q)
            return tip - C.FINGERTIP_LENGTH * a2

        wps = [Pose(position=[float(v) for v in re_origin(w).tolist()], orientation=[float(v) for v in quat.tolist()]) for w in targets["waypoints"]]
        pre_p = re_origin(targets["waypoints"][0]) - a2 * float(C.PRE_DIST)
        pre = Pose(position=[float(v) for v in pre_p.tolist()], orientation=[float(v) for v in quat.tolist()])
        q_seed = motions.robot.get_joint_positions()
        for pose in (pre, wps[0]):
            sol = motions.backend.solve_ik(f"{arm}_arm", {M.EEF_LINK[arm]: as_world_pose(pose)}, q_seed=q_seed, sync_world=True)
            if sol is None:
                log.info("push: pitched blade (%.0f deg) not reachable here (%s) - falling back to the raised blade", math.degrees(pitch), "pre-pose" if pose is pre else "contact")
                return None
        out = dict(targets)
        out["pre"], out["waypoints"] = pre, wps
        return out
    except Exception as e:  # noqa: BLE001 - the pitch is an option, never a failure
        log.info("push: pitched blade unavailable (%r)", e)
        return None


def _base_nudge(motions: Motions, state, arm: str, obj: str, direction, left: float, cfg: ExecuteConfig, label: str) -> Iterator[th.Tensor]:
    """Drive the base ``left`` (+ overshoot) along the push ``direction`` with the arm locked at its joints (the door
    base-follow mechanism): the pad in contact pushes the object the rest of the way when the arm sits at its reach
    boundary. The nudge goes a little past the goal (a touching goal is contact) and the tracker's tolerance shrinks with
    the distance (0918-014040: a 7 mm nudge inside the 8 mm tolerance never moved the base)."""
    d = C.as_tensor(direction).reshape(-1)[:3].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    move = float(left) + cfg.push_base_nudge_overshoot_m
    tol = min(cfg.push_base_nudge_tol_m, max(0.002, 0.5 * float(left)))
    x, y, yaw = state.base_pose2d()
    goal2d = [x + float(d[0]) * move, y + float(d[1]) * move, yaw]
    log.info("%s(%s): %.1f mm still missing with the arm at its reach boundary -> nudging the base %.1f mm along the push direction (tol %.0f mm), arm locked",
             label, obj, float(left) * 1000, move * 1000, tol * 1000)
    q = motions.robot.get_joint_positions()
    motions.state.arm_target[arm] = q[motions.robot.arm_control_idx[arm]].clone().float()
    tcfg = replace(motions.cfg.base_tracker, v_max=(cfg.push_base_nudge_speed, cfg.push_base_nudge_speed, 0.2), pos_tol=tol)
    try:
        yield from motions.track_path([goal2d], tracker_cfg=tcfg)
    except SkillError as e:  # a tracker time-out keeps what the nudge achieved
        log.warning("%s(%s): base nudge did not settle (%s); keeping what it achieved", label, obj, e.message)


def _push_leg(motions: Motions, state, arm: str, obj: str, waypoints: Sequence[Any], geom: "PushGeometry", pose0, cfg: ExecuteConfig,
              ctx: Dict[str, Any], label: str, partial: bool = False) -> Iterator[th.Tensor]:
    """A pad leg with the pushed object watched (owner 2026-09-17 23:39): the leg **stops as soon as the object has
    reached its goal** (``push_stop_at_goal``; residual measured against the TRO target within ``push_stop_frac_*`` of
    its tolerance) instead of driving the pad to the recorded end - the recorded arc rotates the painting further than
    the goal (0917-224633: -0.25 rad), and a pad driven on against a blocked target loads the arm and the base (the
    base sprang forward when the pad left) and squeezes the target against the brace (+0.1 rad relaxation)."""
    sign = 1.0 if geom.kind == "line" else math.copysign(1.0, geom.angle)
    tgt = state.pose_target(obj) if hasattr(state, "pose_target") else None
    tol = None
    if tgt is not None:
        tol = getattr(tgt, "orientation_tolerance" if geom.kind == "arc" else "position_tolerance", None)
    margin = None if tol is None else float(tol) * (cfg.push_stop_frac_arc if geom.kind == "arc" else cfg.push_stop_frac_line)
    gen = motions.arm_along(arm, waypoints, loaded=True, partial=partial)
    k = 0
    for a in gen:
        yield a
        k += 1
        if not cfg.push_stop_at_goal or margin is None:
            continue
        res, src = push_residual(geom, state, obj, pose0)
        if res is None or src not in ("goal_position", "goal_orientation"):
            continue
        if float(res) * sign <= margin:
            log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d steps -> the pad stops here", label, obj, src, float(res) * sign, margin, k)
            ctx.setdefault("push_stopped", []).append({"leg": label, "steps": k, "residual": float(res) * sign})
            gen.close()
            return


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
    # rush 2026-09-17: a straight push is re-centred on the object's ACTUAL planar position - the recorded pad poses
    # assume the object at goal - direction x distance_m (the recorded place); a box placed 1-3 cm off is met off-centre
    # and yaws under the push (touching 0917-215342: 35 deg, orientation tolerance 0.2 rad)
    tgt0 = state.pose_target(obj) if hasattr(state, "pose_target") else None
    if action.contact.mid is None and cfg.push_recentre_max_m > 0 and tgt0 is not None and getattr(tgt0, "position", None) is not None:
        d0 = C.as_tensor(targets["direction"]).reshape(-1)[:3].float()
        d0 = d0 / max(float(th.linalg.norm(d0)), 1e-9)
        rec0 = C.as_tensor(tgt0.position).reshape(-1)[:3].float() - d0 * float(action.contact.distance_m)
        off = C.as_tensor(state.object_pose(obj)[0]).reshape(-1)[:3].float() - rec0
        off[2] = 0.0
        if 0.003 < float(th.linalg.norm(off)) <= cfg.push_recentre_max_m:
            def _shifted(pose):
                p, q = as_world_pose(pose)
                return Pose(position=[float(v) for v in (p + off).tolist()], orientation=[float(v) for v in q.tolist()])
            targets = dict(targets)
            targets["pre"] = _shifted(targets["pre"])
            targets["waypoints"] = [_shifted(w) for w in targets["waypoints"]]
            ctx["push_recentre_m"] = [round(float(v), 4) for v in off.tolist()]
            log.info("push(%s): object %.1f cm off its recorded start position -> pad pre-pose and waypoints shifted by %s", obj, float(th.linalg.norm(off)) * 100, ctx["push_recentre_m"])
    # (owner 2026-09-18 01:33, (b)) palm clearance of a recorded top-down blade: the fingers' section is PALM_FROM_TIP
    # (7 cm) long, above it the palm is wider than the blade stand-off assumed - when the object's top rises above the
    # palm line the pad (pre-pose + waypoints) is raised until the palm stands push_palm_clear_m above the top, as long as
    # the finger section still covers >= push_palm_min_face_m of the face (touching 0917-22*: the palm pressed the box
    # and shook the table)
    extra = getattr(action.contact, "extra", None) or {}
    tip_z = extra.get("fingertip_z")
    if cfg.push_palm_clear_m >= 0 and tip_z is not None and extra.get("blade_half_thickness_m") is not None and not extra.get("palm_clear"):
        try:
            top_z = float(C.as_tensor(state.object(obj).aabb[1]).reshape(-1)[2])
        except Exception:  # noqa: BLE001
            top_z = None
        palm_z = float(tip_z) + PALM_FROM_TIP_M
        if top_z is not None and palm_z < top_z + cfg.push_palm_clear_m:
            dz = top_z + cfg.push_palm_clear_m - palm_z
            pitched = None
            if cfg.push_palm_pitch_deg > 0:
                # first choice: PITCH the blade about its fingertip contact line, palm leaning away from the face (the
                # sampler's pitched classes) - the fingers keep pushing LOW on the face (a vertical blade raised for palm
                # clearance pushes 7.8 cm up an 11 cm box, tilts it, and the box rocks back when the pad leaves:
                # 0918-015654 / 0918-021730); adopted only when its pre-pose and contact pose are reachable here
                pitched = _pitched_blade(motions, state, arm, targets, math.radians(cfg.push_palm_pitch_deg))
            if pitched is not None:
                targets = pitched
                ctx["push_palm_pitch_deg"] = cfg.push_palm_pitch_deg
                log.info("push(%s): the blade's palm line (%.3f) would overlap the object's top (%.3f) -> blade pitched %.0f deg about the fingertips, palm leaning away from the face",
                         obj, palm_z, top_z, cfg.push_palm_pitch_deg)
            elif float(tip_z) + dz <= top_z - cfg.push_palm_min_face_m:
                lift = th.tensor([0.0, 0.0, dz])

                def _lifted(pose):
                    p, q = as_world_pose(pose)
                    return Pose(position=[float(v) for v in (p + lift).tolist()], orientation=[float(v) for v in q.tolist()])

                targets = dict(targets)
                targets["pre"] = _lifted(targets["pre"])
                targets["waypoints"] = [_lifted(w) for w in targets["waypoints"]]
                ctx["push_palm_lift_m"] = round(dz, 4)
                log.info("push(%s): the blade's palm line (%.3f) would overlap the object's top (%.3f) -> pad raised %.1f cm, palm %.0f mm clear", obj, palm_z, top_z, dz * 100, cfg.push_palm_clear_m * 1000)
            else:
                log.warning("push(%s): the object's top (%.3f) reaches the blade's palm section (palm line %.3f) and raising the pad would leave < %.0f mm of face - pushing as recorded",
                            obj, top_z, palm_z, cfg.push_palm_min_face_m * 1000)
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
    yield from _push_leg(motions, state, arm, obj, targets["waypoints"], geom, pose0, cfg, ctx, "push")
    # 1. hold with the pad in contact until the object is still, then measure the residual to the goal
    yield from _hold_until_still(motions, state, obj, cfg)
    residual, source = push_residual(geom, state, obj, pose0)
    ctx["push_residual"] = {"kind": geom.kind, "value": residual, "source": source, "recorded": geom.total,
                            "angle_sign": (math.copysign(1.0, geom.angle) if geom.kind == "arc" else 1.0)}
    # the pad's end pose: where it actually stopped when the leg ended at the goal (0917-235415: judged against the
    # recorded arc end, 64 mm / 0.47 rad away -> 'postcondition of push failed'), else the recorded end
    end_pose = as_world_pose(state.eef_pose(arm)) if ctx.get("push_stopped") else as_world_pose(targets["waypoints"][-1])
    # 2. re-push the residual once when it is in the push's sense and the recorded arc / line allows the extension
    if residual is not None:
        forward = residual * (1.0 if geom.kind == "line" else math.copysign(1.0, geom.angle)) > 0
        ctx["push_residual"]["short"] = bool(forward)
        rmin = cfg.push_residual_min_rad if geom.kind == "arc" else cfg.push_residual_min_m
        # the re-push cap: a fraction of the recorded motion, never below twice the goal tolerance (a recorded push of
        # gap + tolerance, core a6d7123, may leave a residual of a tolerance on a short gap)
        cap = cfg.push_residual_max_fraction * geom.total
        tgt = getattr(state, "pose_target", lambda name: None)(obj)
        tol = getattr(tgt, "orientation_tolerance" if geom.kind == "arc" else "position_tolerance", None) if tgt is not None else None
        if tol is not None:
            cap = max(cap, 2.0 * float(tol))
        ctx["push_residual"]["repush_cap"] = cap
        if forward and rmin <= abs(residual) <= cap:
            ext = extend_push(geom, end_pose, residual)
            log.info("push(%s): residual %.4f %s (%s) -> pushing on along the extended %s", obj, residual, "rad" if geom.kind == "arc" else "m", source, geom.kind)
            yield from _push_leg(motions, state, arm, obj, ext, geom, pose0, cfg, ctx, "re-push", partial=True)  # best-effort: a stall keeps what it achieved
            yield from _hold_until_still(motions, state, obj, cfg)
            end_pose = as_world_pose(state.eef_pose(arm))  # where the pad actually is (a partial extension ends short of ext[-1])
            after, _ = push_residual(geom, state, obj, pose0)
            ctx["push_residual"]["extended_by"] = residual
            ctx["push_residual"]["after"] = after
        elif forward and abs(residual) > cap:
            ctx["push_residual"]["skipped"] = "beyond the recorded arc"
        elif not forward and abs(residual) >= rmin:
            ctx["push_residual"]["skipped"] = "overshoot (a pad cannot pull)"
        # 2b. (rush 2026-09-17) a line push whose arm sits at its reach boundary (touching 0917-211018: the extension
        #     moved the box 0 mm, 3 cm short) is closed with the base: the arm locked at its joints, the base driven
        #     along the push direction by what is still missing (the door base-follow mechanism), pad in contact
        left, _ = push_residual(geom, state, obj, pose0)
        if geom.kind == "line" and cfg.push_base_nudge_max_m > 0 and left is not None and cfg.push_base_nudge_min_m <= left <= cfg.push_base_nudge_max_m:
            yield from _base_nudge(motions, state, arm, obj, targets["direction"], left, cfg, "push")
            yield from _hold_until_still(motions, state, obj, cfg)
            end_pose = as_world_pose(state.eef_pose(arm))
            after_nudge, _ = push_residual(geom, state, obj, pose0)
            ctx["push_residual"]["base_nudge_m"] = left
            ctx["push_residual"]["after_nudge"] = after_nudge
    # 3. back off along the reversed push direction so the retreat does not rub the pad up the object's face, then
    #    measure the residual the object is left with (a swung painting settles back onto the backed-off pad)
    tangent = push_tangent(geom, end_pose)
    back = (end_pose[0] - tangent * cfg.push_backoff_m, end_pose[1])
    if cfg.push_backoff_m > 0:
        yield from motions.arm_along(arm, [back], loaded=True)
    else:
        back = end_pose  # rush: ISG_PUSH_BACKOFF_M=0 keeps the pad on the target (the brace retreats first, ISG_BRACE_RETREATS_FIRST)
    ctx["push_end_pose"] = back
    _hook(ctx, "fov")(False)  # the contact span ends with the contact leg
    yield from _hold_until_still(motions, state, obj, cfg)
    final, _ = push_residual(geom, state, obj, pose0)
    # 3b. (owner 2026-09-18 01:33) a line push whose object rocks back once the pad leaves (0918-015654: pushed 1 mm past
    #     the touching position, +8 mm back at the back-off - a blade high on a box tilts it) is closed in further rounds:
    #     re-approach along the push line to the object, nudge the base by what reopened (the settled, upright object
    #     needs less push each round), back off, measure; up to push_renudge_rounds
    rounds = 0
    while (geom.kind == "line" and cfg.push_renudge_rounds > rounds and final is not None and cfg.push_base_nudge_min_m <= final <= cfg.push_base_nudge_max_m
           and cfg.push_backoff_m > 0):
        rounds += 1
        log.info("push(%s): %.1f mm reopened after the back-off -> re-nudge round %d", obj, final * 1000, rounds)
        _hook(ctx, "fov")(True)
        yield from motions.arm_along(arm, [end_pose], loaded=True, partial=True)  # back onto the face along the push line
        yield from _base_nudge(motions, state, arm, obj, targets["direction"], final, cfg, f"re-nudge {rounds}")
        yield from _hold_until_still(motions, state, obj, cfg)
        end_pose = as_world_pose(state.eef_pose(arm))
        back = (end_pose[0] - tangent * cfg.push_backoff_m, end_pose[1])
        yield from motions.arm_along(arm, [back], loaded=True)
        ctx["push_end_pose"] = back
        _hook(ctx, "fov")(False)
        yield from _hold_until_still(motions, state, obj, cfg)
        final, _ = push_residual(geom, state, obj, pose0)
        ctx["push_residual"].setdefault("renudge", []).append({"round": rounds, "final": final})
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
    _hook(ctx, "fov")(False)  # the contact span ends with the contact leg (the brace then holds still)


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
    ignore_passive = os.environ.get("ISG_FOLLOW_IGNORE_PASSIVE", "0") == "1"  # rush: the follow drove to q_goal (sidecar.followed_fractions)
    q_target = art.q_at(art.passive_from) if (art.passive_from is not None and not ignore_passive) else art.q_goal
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
#: actions the gripper-in-FOV monitor never runs during: none — rule (b) decides per arm by its open span (a retreat
#: after a release has no open span; a ``Retreat(release=True)`` is watched until it releases)
FOV_EXEMPT_ACTIONS: Tuple[type, ...] = ()


def executor_for(action: Action) -> Tuple[ExecuteFn, PostconditionFn]:
    """The ``(execute, postcondition)`` pair of ``action``'s class (exact class first, then its MRO)."""
    for cls in type(action).__mro__:
        if cls in EXECUTORS:
            return EXECUTORS[cls]
    raise KeyError(f"no executor for action class {type(action).__name__}")


def is_manipulation(action: Action) -> bool:
    return isinstance(action, MANIPULATION_ACTIONS)


def fov_monitored(action: Action) -> bool:
    """Whether the gripper-in-FOV monitor runs during ``action``: every action of a program (rule (b): which arms are
    watched is decided by the spans the execute functions open at the pre-grasp / pre-contact and close at the
    release / end of the contact leg — a carried object's hand is watched through a Transition as well)."""
    return isinstance(action, Action) and not isinstance(action, FOV_EXEMPT_ACTIONS)


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
