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

* ``Navigate``   ``track_path(waypoints, style)``; post: base within tolerance of the last waypoint.
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
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Protocol, Sequence, Tuple

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
from ispatialgym.oracle.motions import ApproachFinish, Motions, SkillError, arc_between, as_pose2d, as_world_pose, joint_axis_from_contacts
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
    def attachment_offset(self, child: str, parent: str) -> Optional[Dict[str, Any]]: ...  # live meta-link offset + correcting transform
    def attach(self, child: str, parent: str) -> bool: ...  # AttachedTo.set_value(True) at the current poses
    def init_pose(self, name: str) -> Optional[Tuple[th.Tensor, th.Tensor]]: ...  # the object's pose at the program start


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
    #: closed-loop articulation follow (2026-09-18, dishwasher 0917-213723): the follow is judged on the MOVED JOINT, not
    #: on the arm - it stops as soon as the joint is within ``follow_stop_frac`` of its tolerance from q_goal
    #: (``follow_stop_at_goal``), the arm's stored deflection is unwound (``follow_unwind``: hold target := measured
    #: joints, so the drives stop pressing and the release shoves nothing) and the hold ends once the joint is still
    #: (``follow_still_rad`` per step for ``follow_still_steps`` steps, at most ``follow_settle_steps``). The former
    #: arm-side waits (90 steps for a joint error the door's friction never let close + 45 steps for phantom joint
    #: velocities of saturated drives) paused the demo 4.5 s with the door already open.
    follow_stop_at_goal: bool = True
    follow_stop_frac: float = 0.25
    follow_unwind: bool = True
    #: the unwind is spread over this many steps (hold target -> measured joints, re-read every step): a one-step unwind
    #: let the wound-up arm snap 0.85 rad/s against the door (dishwasher_wide 0920-002330) and the strain it could not
    #: shed stayed in the grasp constraint - the release then rocked the torso (base pitch 0.24 rad/s)
    follow_unwind_steps: int = 12
    follow_still_rad: float = 5e-4
    follow_still_steps: int = 5
    #: sequential arm segments track the handle arc CLOSED-LOOP (``Motions.arm_arc_closed_loop``: one differential-IK
    #: step per control step from the measured joints, the target never more than one step ahead of where the door lets
    #: the arm be) instead of streaming an open-loop joint path between the recorded poses. With the robot's 1e7-stiff,
    #: effort-capped drives an open-loop path saturates the drives on any mismatch with the grasp constraint (~100 N at
    #: the hand): the 47.6 kg chassis tipped 0.8 deg, the evaluator's 250 kg one bent torso_joint1 2.6 deg, both snapping
    #: back at the release (dishwasher_wide 0921-050314 / 0921-235247). The compliance is in the recorded commands, so a
    #: policy trained on the demo reproduces it on the evaluator's stock drives (owner 2026-09-22 11:19).
    #: measured 0922-1126xx on the same instance: the per-step closed loop turned the steady wrench into a 30 Hz pulse train
    #: (every 1.7 mm correction a saturated burst): torso 4.3 deg, eef bursts 0.39 m/s, release snap 38 deg/s - worse than
    #: the streamed path (2.6 deg / 0.17 / 25). "reanchored" streams the path open-loop within each recorded pose but starts
    #: every pose's arc from the MEASURED hand pose rotated by the joint delta about the axis, instead of chaining the
    #: recorded target poses: the grasp offset otherwise becomes a permanent radial mismatch from the second pose on.
    follow_arm_tracking: str = "reanchored"  # "recorded" (chain the recorded poses) | "reanchored" | "closed_loop"
    follow_arm_band_rad: float = 0.005  # a recorded pose counts as reached when the joint is within this of its q
    #: steps held after a handle / object is released before the retreat leg moves (the constraint-removal transient)
    release_settle_steps: int = 12
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
    #: handle grasps (articulation) attach the assisted grasp with a FixedJoint instead of OmniGibson's SphericalJoint for
    #: links behind a movable joint (2026-09-18: the lid did not turn with the hand, ghost_hand 29 deg on a 4 cm lever)
    handle_grasp_fixed_ag: bool = True
    push_stop_frac_line: float = 0.0
    #: (2026-09-22, alignment-world_vertical 0918-000750) pad speed along a push ARC about a hinge (m/s; the approach to the
    #: first contact keeps ``cartesian_speed``). The hinge torque is friction + damping * omega, so at the former 0.10 m/s
    #: (0.6 rad/s on the painting) the viscous term alone was 3.6 of the 6.8 N.m the pad had to supply (~40 N at 0.17 m):
    #: the elbow drive saturated at its 25 N.m cap, the wrist drives went bang-bang against the contact and the painting
    #: shook. 3.5 cm/s (0.2 rad/s) roughly halves the required force. ``push_speed_line``: straight pushes (None = default).
    push_speed_arc: float = 0.035
    push_speed_line: Optional[float] = None
    #: the arc of a hinged target is tracked CLOSED-LOOP (``Motions.arm_arc_closed_loop``: one differential-IK step per
    #: control step from the MEASURED joints, the pad target never more than one step ahead of where the hinge lets the
    #: pad be) instead of streaming the open-loop joint path of the recorded arc: with the 1e7-stiff, effort-capped drives
    #: an open-loop path accumulates its mismatch with the constraint as drive saturation (0918-000750: joint4 0.097 rad
    #: behind its target for the whole arc, released as a kick at the stop). ``"streamed"`` = the former open-loop leg.
    #: Articulated pushes (a lid on its joint, ``contacts.push.articulation``) keep the streamed leg.
    push_arc_tracking: str = "closed_loop"  # "closed_loop" | "streamed"
    #: the closed-loop arc stops when the pad has been commanded this many times the recorded arc without the object
    #: following (a blocked target is not pushed harder)
    push_arc_lag_factor: float = 1.3
    #: the object counts as touched by the approaching pad once it has turned this much (rad; logged and recorded, the
    #: approach continues to the recorded contact pose - see ``_push_arc_closed_loop``)
    push_contact_detect_rad: float = 0.01
    #: (2026-09-22) the brace of a HINGED target (``contacts.brace.kind`` side_high / below_high / above_low) stands where
    #: the braced point of the object will be AT THE GOAL, this far back along the brace direction (m; < 0 = the recorded
    #: contact at the initial pose). Placed at the initial pose the fist sits in the path of the last degrees of the
    #: rotation (0918-000750: the painting's edge swept 4.7 cm into the fist while levelling, the pads clamped the painting
    #: and the stop kicked it back 0.8 deg = the whole final error); at the goal edge + gap it only catches an overshoot.
    brace_at_goal_gap_m: float = 0.01
    #: the fractions of the goal rotation tried for the goal-edge brace, in turn, until its pre and contact poses are within
    #: the arm's reach (collision-aware IK); none reachable = the recorded contact. Only the full goal edge by default: a
    #: partial fraction still puts the fist in the edge's path before the goal, and the 33 % pose of 0922-234254 stood in the
    #: push arm's transit (the left arm's trajopt to its pre-pose stand-off failed with start and goal collision-free) while
    #: the recorded contact is the sampler-validated pair
    brace_at_goal_fractions: Tuple[float, ...] = (1.0,)
    #: (2026-09-22) the radial contact escape of a retreat: its end pose is probed for reachability (collision-aware IK) and
    #: the radial distance shortened by these factors in turn (never below the escape itself) until a solution exists, the
    #: leg is collision-checked (self + world minus the released object and its supports) with this joint-limit margin
    #: (rad), and a colliding / stalling straight leg is finished by ``arm_to`` - instead of executing a partial straight
    #: leg toward an unreachable end pose (0918-000750: the leg folded the elbow into its own upper arm 3 mrad short of
    #: the joint limit and the drive fought that contact through the settle window)
    retreat_reach_factors: Tuple[float, ...] = (1.0, 0.75, 0.5)
    retreat_check_collisions: bool = True
    retreat_limit_margin_rad: float = 0.05
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
    #: (2026-09-19, attachment) at an attach place the male / female meta-link offset is read from the live world; above these
    #: the hand is corrected by the male -> female transform (a short Cartesian leg, at most ``attach_correct_max_m``) before
    #: the assisted grasp is dropped and ``AttachedTo.set_value(True)`` snaps the residual and creates the fixed joint
    attach_correct_min_m: float = 0.003
    attach_correct_min_rad: float = 0.03
    attach_correct_max_m: float = 0.06
    attach_settle_steps: int = 20
    #: a correction pass lifts the part by this before the lateral / rotational move and descends again (0920-213721: the pass
    #: moved a cap hovering 4.6 mm high DOWN and 4.4 mm sideways in one straight leg; its 5 deg-tilted rim caught the neck edge and
    #: pushed the lemonade bottle 10 mm / 5 deg, which then rocked for 145 steps after the release)
    attach_correct_lift_m: float = 0.006
    #: (2026-09-20 22:30) the place is re-aimed for the part's LIVE in-hand pose at the pre-place: the male -> female fix measured
    #: there gives the hand pose that aligns the links (the fingertip pinch of a 1.1 cm cap lets it pivot 3-5 deg / 5 mm about the
    #: closing axis before the assisted grasp locks, and the carry leg's soft tolerance adds up to 1 cm); applied when the shift
    #: from the recorded place is within these bounds
    inhand_reaim_max_m: float = 0.03
    inhand_reaim_max_rad: float = 0.35
    #: (owner 2026-09-20 18:25) the snap AttachedTo makes when the joint is created must stay invisible: a residual above these after
    #: the correction gets one more correction pass, and above them still the part is NOT attached (the place fails, the executor's
    #: retry re-aims at the partner's live pose) - 0920-164504 snapped the cap 17 mm
    attach_snap_max_m: float = 0.008
    attach_snap_max_rad: float = 0.10
    #: an attach place re-aims at the partner's LIVE pose: the recorded pre-place / place / descent / retreat / carry are moved by the
    #: partner's displacement since the program start when it exceeds this (0920-164504: the bottle knocked 4.7 cm during the grasp
    #: phase, the descent came down 4 cm beside the neck)
    partner_shift_min_m: float = 0.005
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
    #: owner 2026-09-22 11:27: a pinching hand that comes from its own brace beside the object reaches the pre-grasp by a tracked
    #: Cartesian detour (up ``grasp_transit_lift_m``, across, down) instead of a joint-space plan that swung it out of the frame
    grasp_transit_cartesian: bool = True
    grasp_transit_lift_m: float = 0.06
    #: owner 2026-09-22 11:27: a brace is a safety stop, not a force - its approach leg stops as soon as the braced object has moved
    #: this far (m; 0 = off) and then backs off ``brace_touch_backoff_m`` along its approach
    brace_touch_stop_m: float = 0.002
    brace_touch_backoff_m: float = 0.004
    #: a contact pre-pose next to an ignored support (wall, table) is reached via a standoff this far away from the
    #: support, planned with the support present, then a straight leg (``reach_pre_pose``)
    pre_pose_standoff_m: float = 0.08
    #: (owner 2026-09-18 12:42) no standoff when the pre-pose is already this clear of every support (m): planned directly
    pre_pose_standoff_skip_m: float = 0.08
    #: (owner 2026-09-18 12:42) a straight push's pre-pose sits this far above / before its first contact pose (m; None = the record's 10 cm)
    push_pre_dist_m: Optional[float] = 0.04
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


def support_clearance(state, pose, objects: Sequence[Any]) -> Optional[float]:
    """Smallest distance (m) from ``pose``'s position to the AABBs of ``objects``; None without a measurable object."""
    p = as_world_pose(pose)[0]
    best = None
    for o in objects:
        try:
            lo, hi = o.aabb
        except Exception:  # noqa: BLE001
            continue
        lo, hi = C.as_tensor(lo).float(), C.as_tensor(hi).float()
        nearest = th.minimum(th.maximum(p, lo), hi)
        d = float(th.linalg.norm(p - nearest))
        best = d if best is None else min(best, d)
    return best


def reach_pre_pose(motions: Motions, state, arm: str, pre, supports: Sequence[Any], through: Sequence[Any], q_hint, ctx: Dict[str, Any], cfg: "ExecuteConfig",
                   ignore_links: Sequence[Any] = ()) -> Iterator[th.Tensor]:
    """Reach a contact pre-pose that the sampler checked with its supports ignored (a wall / table 2 cm from the
    pads). The supports leave the collision world only for the **last straight leg**: the trajopt goes to a standoff
    ``pre + standoff · n`` (``n`` away from the supports) with the supports present — a trajopt with the wall ignored
    planned the hand *through* the wall region and the arm jammed against it (1.39 rad joint error on the painting
    push) — then a straight ``arm_along`` closes the standoff. ``through`` (the pushed target) is ignored for both."""
    if not supports:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through), ignore_links=list(ignore_links))
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    n = escape_direction(state, pre, supports)
    if n is None:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(supports) + list(through), ignore_links=list(ignore_links))
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    clearance = support_clearance(state, pre, supports)
    if clearance is not None and clearance >= cfg.pre_pose_standoff_skip_m:
        # owner 2026-09-18 12:42 (1): a pre-pose already this clear of its supports (a table push's pre-pose 10-16 cm
        # above the table) is planned directly with the supports present - the standoff excursion (up 8 cm, then down
        # again) was a redundant raise before the descent; the wall contacts (pre-pose 2-5 cm off the wall) keep it
        ctx["pre_pose_standoff"] = {"skipped": True, "clearance_m": round(clearance, 3)}
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through), ignore_links=list(ignore_links))
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
        yield from motions.arm_to(arm, q=q_standoff, ignore_objects=list(through), ignore_links=list(ignore_links))
    else:
        ctx["pre_pose_standoff"]["branch"] = "planner"
        yield from motions.arm_to(arm, pose=standoff, ignore_objects=list(through), ignore_links=list(ignore_links))
    if getattr(motions, "last_relaxed_links", None):
        ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
    yield from motions.arm_along(arm, [pre], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)


def _grasp_extra_float(action: Grasp, key: str) -> Optional[float]:
    """A positive float fact of the grasp record (``grasps[].<key>``), None otherwise."""
    extra = getattr(getattr(action, "grasp", None), "extra", None)
    v = extra.get(key) if isinstance(extra, dict) else None
    try:
        v = float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
    return v if v is not None and math.isfinite(v) and v > 0 else None


def reach_via_standoff(motions: Motions, arm: str, pre, distance: float, q_hint, ignore: Sequence[Any], ctx: Dict[str, Any]) -> Iterator[th.Tensor]:
    """Over-the-top approach of a grasp pre-pose (attachment, owner 2026-09-19 20:42; ``grasps[].approach_standoff_m`` recorded
    by the sampler when the partner stands near the male): the trajopt goes to a standoff ``distance`` further back along the
    approach axis (above a top-down grasp) with the scene present, then a straight leg closes the standoff — the free trajopt to
    the pre-grasp swung the open hand at table height past the partner and the wrist camera grazed the 29 cm teapot's spout
    (pwgbsb / jlalfc 0919-173115 / 181512). The standoff is taken in the pre-grasp witness's IK branch when it can be tracked
    from it (``standoff_configuration``); a standoff plan that fails falls back to the direct plan."""
    p, q = as_world_pose(pre)
    standoff = (p - C.approach_axis(q) * float(distance), q)
    rec: Dict[str, Any] = {"distance_m": float(distance)}
    ctx["approach_standoff"] = rec
    q_standoff = standoff_configuration(motions, arm, pre, standoff, q_hint) if q_hint is not None else None
    # the transit to the standoff is planned with the supports PRESENT (owner 2026-09-20 18:25, as the direct transit of
    # 8254a8fc): the standoff stands 15-20 cm above the table, nothing needs ignoring - planned with the table ignored, trajopt
    # dragged the open hand along the table for 162 steps and brushed the lemonade bottle 14 mm / 16 deg (cap 0920-193713);
    # the supports leave the collision world only for the fallbacks
    for ignore_now, tag in (([], "supports present"), (list(ignore), "supports ignored (fallback)")):
        try:
            if q_standoff is not None:
                rec["branch"] = "witness"
                yield from motions.arm_to(arm, q=q_standoff, ignore_objects=ignore_now)
            else:
                rec["branch"] = "planner"
                yield from motions.arm_to(arm, pose=standoff, ignore_objects=ignore_now)
            ctx["pre_grasp_transit"] = f"{tag} (standoff {distance:.2f} m)"
            break
        except SkillError as e:
            log.warning("grasp(%s): the standoff plan with %s failed (%s)", arm, tag, e.message)
            rec.setdefault("failures", []).append(f"{tag}: {e.message[:100]}")
    else:
        log.warning("grasp(%s): no plan to the standoff; direct plan to the pre-grasp with the supports ignored", arm)
        rec["fallback"] = "direct"
        ctx["pre_grasp_transit"] = "supports ignored (direct fallback)"
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(ignore))
        return
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


def _patch_ag_joint_type(robot, joint_type: Optional[str]) -> None:
    """Force the assisted-grasp joint type OmniGibson picks for the next attachment (``robot._get_assisted_grasp_joint_type``:
    FixedJoint for rigid bodies, SphericalJoint for a link behind a movable joint). ``joint_type`` = ``"FixedJoint"`` installs
    the override (objects the stock rule refuses - too heavy - stay refused), ``None`` restores the stock method."""
    if joint_type is None:
        orig = getattr(robot, "_isg_ag_joint_type_orig", None)
        if orig is not None:
            robot._get_assisted_grasp_joint_type = orig
            del robot._isg_ag_joint_type_orig
        return
    if getattr(robot, "_isg_ag_joint_type_orig", None) is None:
        robot._isg_ag_joint_type_orig = robot._get_assisted_grasp_joint_type

    def forced(target_obj, target_link_name, _orig=robot._isg_ag_joint_type_orig, _kind=joint_type):
        stock = _orig(target_obj, target_link_name)
        return None if stock is None else _kind

    robot._get_assisted_grasp_joint_type = forced


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
    # owner 2026-09-22 11:27: a stance may record the opening PER ARM (``fingers_by_arm``, the push-close re-stance: the fist closed,
    # the idle hand as it is) - the single ``fingers`` value applies to every free arm as before (item 5, the wall two-arm stance)
    by_arm = (getattr(record, "extra", None) or {}).get("fingers_by_arm")
    if isinstance(by_arm, dict) and by_arm:
        for arm in M.ARMS:
            if arm in holding or by_arm.get(arm) is None:
                continue
            want_close = float(by_arm[arm]) <= 0.5 * float(C.FINGER_OPEN)
            if motions.gripper_closed(arm) != want_close:
                set_fingers.append(arm)
                yield from motions.gripper(arm, close=want_close)
        ctx["fingers_set"] = {"arms": set_fingers, "by_arm": {a: float(v) for a, v in by_arm.items() if v is not None}}
    elif fingers is not None:
        want_close = fingers <= 0.5 * float(C.FINGER_OPEN)
        for arm in M.ARMS:
            if arm in holding:
                continue  # a carried object keeps its grasp
            if motions.gripper_closed(arm) != want_close:
                set_fingers.append(arm)
                yield from motions.gripper(arm, close=want_close)
        ctx["fingers_set"] = {"arms": set_fingers, "closed": want_close}
    else:
        ctx["fingers_set"] = None
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
    # a path driven with EMPTY hands was validated by the sampler at the departing stance's arm posture (``paths[].posture``,
    # ``untucked``); after a place + retreat the working arm still hangs over the table, so every arm holding nothing is
    # brought back to that posture before the base moves (block design's per-block stances, owner 2026-09-19). A carried
    # object keeps its grasp and arm (the carried paths were swept with the lift configuration).
    posture = targets.get("posture")
    if posture == "untucked" and not targets.get("carried") and all(state.held_object(arm) is None for arm in M.ARMS):
        q_now = state.robot.get_joint_positions()
        want = state.robot.untucked_default_joint_pos
        move = {}
        for arm in M.ARMS:
            idx = state.robot.arm_control_idx[arm]
            qa = C.as_tensor(want[idx]).reshape(-1).float()
            if float(th.max(th.abs(q_now[idx].float() - qa))) > IDLE_ARM_TOL:
                move[arm] = qa
        ctx["navigate_arms_reposed"] = sorted(move)
        if move:
            yield from motions.arm_to(tuple(sorted(move)), q=move)
    yield from motions.track_path(targets["waypoints"], posture=posture, style=targets.get("style"))


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


def achievable_trunk(t: float, k: float) -> Tuple[float, float]:
    """The (translate, tilt) the torso actually reaches for a nominal pair: ``conventions.torso_q`` clamps the tilt joint to
    its limit, so a deep-crouch tilt of the reach-band grid ((0.95, -1.2): joint3 -0.89 - 1.2 < -1.83) lands at -0.94.
    Sampler and oracle command the same clamped configuration; the post-conditions compare against it (0920-011106: the
    A -> B transition 'failed' with trunk_error 0.26 rad although the torso stood exactly where the sampler had checked)."""
    q = C.torso_q(float(t), float(k), C.TORSO_JOINT_LIMITS)
    return C.torso_q_to_translate_tilt(q)


def post_transition(action: Transition, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    t, k = achievable_trunk(*targets["trunk"])
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
    t, k = achievable_trunk(*targets["trunk"])
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
    if action.kind == "handle" and action.articulation is not None:
        # 2026-09-19 (two-flap carton): the sampler records the handle's pinch width (articulation.handle_width_m) so the approach
        # is pre-shaped like a library grasp - a fully open lower pad (5 cm below the flap) passes through the neighbouring flap
        w = (action.articulation.extra or {}).get("handle_width_m")
        try:
            w = float(w)
        except (TypeError, ValueError):
            return None
        return w if math.isfinite(w) and w > 0 else None
    if action.kind == "cube":
        # block design (owner 2026-09-19): the pinch width is the cube's edge (``blocks[].edge_m``, else the conventions'
        # default) - without a pre-shape the fully open release beside a placed neighbour 6 mm away shoved it 4 cm along
        # the table (demo 0919-121912: both left-column blocks 3.6 cm off, 'not_reached')
        w = (getattr(getattr(action, "block", None), "extra", None) or {}).get("edge_m")
        try:
            w = float(w) if w is not None else float(C.BLOCK_EDGE)
        except (TypeError, ValueError):
            return None
        return w if math.isfinite(w) and w > 0 else None
    if action.kind != "library" or action.grasp is None:
        return None
    # the sampler's own measurement of the object's width at the pads (``grasps[].pinch_width_m``, attachment 2026-09-19) wins over
    # the library's ``width``: lid/pwgbsb's library grasps are one-finger wedges whose recorded width (6-12 mm) is a third of the
    # 24-30 mm knob - the pads pre-shaped to a 26 mm gap landed on the knob and the closure caught nothing (0919-210926)
    recorded = _grasp_extra_float(action, "pinch_width_m")
    if recorded is not None:
        return recorded
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
    hold = getattr(motions, "press_hold", None)  # ctx is per action (executor.attempt): the held press lives on the Motions
    ignore_links = []
    if hold and hold.get("object") == action.object and hold.get("arm") != arm:
        # the other hand is pressing this very object (a two-flap carton's prep press, 2026-09-19): its fist in contact makes
        # the start configuration "in collision" for cuRobo (0919-061328: offender left_lid). Only the PRESSED LINK leaves the
        # world for the transit (0920-021937: with the whole box gone the plan ran through its body, the arm did not converge)
        obj_h = state.object(action.object)
        if hold.get("link"):
            # the pressed link and every other MOVING link of the object (the sheets a fist may touch: the other flap's edge at
            # the seam); the base link (the box body) stays an obstacle
            moving = [ln for ln in getattr(obj_h, "links", {}) if ln != getattr(getattr(obj_h, "root_link", None), "body_name", None) and ln != "base_link"]
            ignore_links = [(obj_h, ln) for ln in ({hold["link"]} | set(moving))]
            log.info("grasp(%s): links %s of %s leave the collision world for the transit (the %s fist presses %s)", arm, sorted(ln for _, ln in ignore_links), action.object,
                     hold.get("arm"), hold["link"])
        else:
            supports = list(supports) + [obj_h]
            log.info("grasp(%s): %s leaves the collision world for the transit (the %s hand holds its press)", arm, action.object, hold.get("arm"))
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in supports] + [f"{getattr(o, 'name', o)}:{ln}" for o, ln in ignore_links]
    standoff_m = _grasp_extra_float(action, "approach_standoff_m")
    if standoff_m:
        yield from reach_via_standoff(motions, arm, targets["pre_grasp"], standoff_m, _witness(ctx, "pre_grasp"), supports, ctx)
    elif ignore_links:
        # the other hand presses the object (the lever): the pinching hand comes from its brace beside the box. Owner 2026-09-22 11:27
        # (0921-232806 / 0922-113512: the joint-space plan from the blade pose swung the hand 70 cm to the right and out of the frame):
        # first a tracked Cartesian detour - straight up, across at that height, down onto the pre-grasp - the planner only when
        # the tracking has no solution
        done = False
        if _cfg(ctx).grasp_transit_cartesian:
            try:
                here = as_world_pose(state.eef_pose(arm))
                pre_p, pre_q = as_world_pose(targets["pre_grasp"])
                lift = float(_cfg(ctx).grasp_transit_lift_m)
                z_top = max(float(here[0][2]), float(pre_p[2])) + lift
                up = (th.tensor([float(here[0][0]), float(here[0][1]), z_top]), here[1])
                over = (th.tensor([float(pre_p[0]), float(pre_p[1]), z_top]), pre_q)
                legs = [here, up, over, (pre_p, pre_q)]
                yield from motions.arm_along(arm, legs)
                ctx["pre_grasp_transit"] = f"cartesian detour {lift * 100:.0f} cm above"
                done = True
            except SkillError as e:
                log.info("grasp(%s): the Cartesian detour to the pre-grasp has no solution (%s) - planning it", arm, e.message[:80])
        if not done:
            # the pose goal in cuRobo's OWN branch first (trajopt picks the goal configuration with the path: the branch nearest the blade
            # pose; 0922-150937: the joint-space plan to the sampler's witness - another branch - swung the hand 70 cm out to the right),
            # the witness branch when that fails
            try:
                yield from motions.arm_to(arm, pose=targets["pre_grasp"], ignore_objects=supports, ignore_links=ignore_links)
                ctx["pre_grasp_transit"] = "pose goal, cuRobo's own branch"
            except SkillError as e:
                log.info("grasp(%s): the pose-goal plan to the pre-grasp failed (%s) - planning to the witness branch", arm, e.message[:80])
                yield from motions.arm_to(arm, pose=targets["pre_grasp"], q_hint=_witness(ctx, "pre_grasp"), ignore_objects=supports, ignore_links=ignore_links)
                ctx["pre_grasp_transit"] = "witness branch"
    else:
        # the transit to the pre-grasp is planned with the supports PRESENT (owner 2026-09-20 18:25; the contact pre-poses' rule
        # ``reach_pre_pose``: a low pre-pose is reached through a standoff away from the support, the support leaves the collision
        # world only for the last straight leg) - planned with the table ignored, trajopt swept the open hand 55 cm at table
        # height, the fingertips dragging the table for 94 steps at 135 N and knocking the partner bottle 4.7 cm (cap 0920-164504).
        # The former plan (supports ignored throughout) is the fallback when no plan exists with them present
        try:
            yield from reach_pre_pose(motions, state, arm, targets["pre_grasp"], supports, [], _witness(ctx, "pre_grasp"), ctx, _cfg(ctx))
            ctx["pre_grasp_transit"] = "supports present"
        except SkillError as e:
            log.warning("grasp(%s): no transit to the pre-grasp with the supports present (%s); planning with them ignored", arm, e.message)
            ctx["pre_grasp_transit"] = f"supports ignored (fallback: {e.message[:80]})"
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
    margin = motions.cfg.gripper_preshape_margin
    if action.kind == "handle" and action.articulation is not None and (action.articulation.extra or {}).get("preshape_margin_m") is not None:
        # a record's own pre-shape margin (two-flap carton 2026-09-20: a pinch approaching ACROSS a sheet's edge at 25 deg needs a
        # wider gap than 1 cm - the edge met the lower pad head-on and the approach stalled 6 cm short, 0920-011416)
        margin = float(action.articulation.extra["preshape_margin_m"])
    if width is not None and margin is not None:
        opening = 0.5 * width + float(margin)
        yield from motions.gripper_opening(arm, opening)
        ctx["preshape_opening_m"] = opening
    yield from motions.arm_along(arm, targets["approach"], finish=finish)
    if finish.used:
        ctx["approach_finish"] = finish.info
    # ramped closure (2.25 mm / step, a quarter of the free speed): both pads reach the object in the same step
    # 2026-09-18 (blender cwkvib 0918-225935, ghost_hand "drifted 2 cm / 29 deg" at the follow): OmniGibson attaches an assisted
    # grasp on a link behind a movable joint (a lid, a door leaf) with a SPHERICAL joint at the contact point - the hand can
    # turn without turning the link, and the arm's arc-tracking slack (1 cm) is tens of degrees on a 4 cm handle lever. A
    # handle pinched across a 13 mm tab transmits torque in reality: handle grasps attach with a FIXED joint (the only
    # kind the follow was designed for; the dishwasher survived the spherical one only through its 30 cm lever)
    fixed_ag = cfg.handle_grasp_fixed_ag and action.kind == "handle"
    if fixed_ag:
        _patch_ag_joint_type(motions.robot, "FixedJoint")
    try:
        yield from motions.gripper(arm, close=True, rate=motions.cfg.gripper_close_rate)
    finally:
        if fixed_ag:
            _patch_ag_joint_type(motions.robot, None)
    ctx["closure_rate_m_per_step"] = motions.cfg.gripper_close_rate
    if fixed_ag:
        params = (getattr(motions.robot, "_ag_obj_constraint_params", None) or {}).get(arm) or {}
        ctx["ag_joint_type"] = params.get("joint_type")
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
    # a free-space in-hand leg (the SLERP carry of attachment, the held-object reorientations): a differential-IK stall
    # anywhere on it is finished by ``arm_to`` to the recorded end pose (its ``end`` witness first), like the lift / retreat /
    # carry legs — attachment 0919-185350 (lid pwgbsb -> teapot jlalfc, pre-attach 25 cm from the lift): the SLERP stalled at
    # waypoint 24/29 with a 13.7 mm residual, no joint at a limit, no singularity (the reach boundary of the start branch)
    cfg = _cfg(ctx)
    finish = _free_leg_finish(state, targets["waypoints"][-1], _witness(ctx, "end") if _witness(ctx, "end") is not None else _witness(ctx, "waypoints"),
                              contact_ignores(state, action.object), cfg)
    yield from motions.arm_along(action.arm, targets["waypoints"], finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if finish.used:
        ctx["reorient_finish"] = finish.info


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
    if action.kind == "attach" and action.on:
        targets = _reaim_attach_targets(action, state, targets, ctx, cfg)
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
    if action.kind == "attach" and action.on:
        # the part is in the air above the partner: re-aim the descent for its live in-hand pose (and the carry's residual)
        targets = yield from _reaim_for_inhand_pose(action, motions, state, targets, ctx, cfg)
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
    if action.kind == "attach" and action.on:
        yield from _attach_at_place(action, motions, state, ctx)
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


def _reaim_attach_targets(action: Place, state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: ExecuteConfig) -> Dict[str, Any]:
    """The attach place re-aimed at the partner's LIVE pose (owner 2026-09-20 18:25): the recorded pre-place / place / descent /
    retreat / carry were computed for the partner at its init pose; when the partner has moved since the program start by more
    than ``partner_shift_min_m`` (0920-164504: the bottle knocked 4.7 cm during the grasp phase; the descent came down 4 cm
    beside the neck, the correction then dragged the cap into it), every target moves with it. A child with a (near-)continuous
    symmetry about the insertion axis (>= 12 recorded equivalences, a zinf cap / cork / lid) follows the partner's TRANSLATION
    only - its yaw is free, and turning the hand to a spun bottle's yaw is wasted motion; an asymmetric child follows the full
    planar rigid motion (translation + yaw about the partner's position). ``ctx["partner_shift"]`` records the facts."""
    getter = getattr(state, "init_pose", None)
    init = getter(action.on) if getter is not None else None
    if init is None:
        return targets
    try:
        live = state.object_pose(action.on)
    except Exception:  # noqa: BLE001
        return targets
    p0, q0 = as_world_pose(init)
    p1, q1 = as_world_pose(live)
    d = (p1 - p0).float()
    dist = float(th.linalg.norm(d))
    dyaw = C.wrap_angle(C.yaw_of(q1) - C.yaw_of(q0)) if hasattr(C, "yaw_of") else 0.0
    rec: Dict[str, Any] = {"partner": action.on, "shift_m": round(dist, 4), "delta": [round(float(v), 4) for v in d.tolist()], "yaw_rad": round(float(dyaw), 4)}
    ctx["partner_shift"] = rec
    if dist < cfg.partner_shift_min_m and abs(float(dyaw)) < 0.02:
        rec["applied"] = False
        return targets
    n_eq = 0
    try:
        from ispatialgym.sampling import predicates as PRED

        n_eq = len((PRED.attachment_context(state.object(action.object), state.object(action.on)) or {}).get("equivalences") or [])
    except Exception:  # noqa: BLE001
        n_eq = 0
    symmetric = n_eq >= 12
    rec.update({"child_equivalences": n_eq, "mode": "translation" if symmetric else "translation+yaw"})

    def moved(pose):
        p, q = as_world_pose(pose)
        if symmetric or abs(float(dyaw)) < 1e-6:
            return Pose(position=[float(v) for v in (p.float() + d).tolist()], orientation=[float(v) for v in q.tolist()])
        qz = C.quat_from_yaw(float(dyaw))
        p_new = p1.float() + C.quat_rotate(qz, p.float() - p0.float())  # the planar rigid motion of the partner applied to the eef target
        return Pose(position=[float(v) for v in p_new.tolist()], orientation=[float(v) for v in C.qmul(qz, q.float()).tolist()])

    out = dict(targets)
    for key in ("pre_place", "place"):
        if out.get(key) is not None:
            out[key] = moved(out[key])
    for key in ("descent", "retreat", "carry"):
        if out.get(key):
            out[key] = [moved(w) for w in out[key]]
    rec["applied"] = True
    log.info("place(%s): %s moved %.1f mm / %.1f deg since the start -> attach targets re-aimed (%s)", action.object, action.on, dist * 1000, math.degrees(float(dyaw)), rec["mode"])
    return out


def _reaim_for_inhand_pose(action: Place, motions: Motions, state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: ExecuteConfig):
    """At the pre-place (the part in the air above the partner) the live male -> female fix (modulo the child's symmetry) applied
    to the live hand pose IS the hand pose that aligns the meta links: ``place`` becomes that pose, ``pre_place`` the same pose
    raised by the recorded descent vector, ``descent`` the straight leg between them, ``retreat`` the recorded retreat carried
    along; the hand first moves to the new pre-place (a few mm, free space). This absorbs the part's in-hand pose (cap/xxqyrt
    0920-213721: 4.6 mm / 5.3 deg off the sampler's ``eef_in_object`` - the fingertip pinch of an 1.1 cm disc pivots about the
    closing axis before the assisted grasp locks) and the carry leg's residual (soft tolerance 1 cm; cork 0920-212757 arrived
    12.8 mm off after a perfect grasp), so the descent lands aligned instead of landing off and correcting in contact.
    ``ctx["inhand_reaim"]`` records the facts. A generator: yields the pre-place move's steps and returns the targets."""
    arm = action.arm
    getter = getattr(state, "attachment_offset", None)
    off = getter(action.object, action.on) if getter is not None else None
    rec: Dict[str, Any] = {"applied": False}
    ctx["inhand_reaim"] = rec
    if off is None or not off.get("fix") or targets.get("place") is None or targets.get("pre_place") is None:
        rec["reason"] = "no meta-link pair / no place targets"
        return targets
    p_now, q_now = (x.float() for x in state.eef_pose(arm))
    p_live, q_live = C.compose(off["fix"], (p_now, q_now))
    p_rec, q_rec = as_world_pose(targets["place"])
    p_pre_rec, _ = as_world_pose(targets["pre_place"])
    d = (p_live - p_rec.float())
    dist, drot = float(th.linalg.norm(d)), C.quat_angle(q_live, q_rec.float())
    rec.update({"shift_m": round(dist, 4), "delta": [round(float(v), 4) for v in d.tolist()], "rot_rad": round(float(drot), 4),
                "inhand_offset_m": round(float(off["pos_m"]), 4), "inhand_offset_rad": round(float(off["rot_rad"]), 4)})
    if dist < 1e-3 and drot < 0.01:
        rec["reason"] = "within 1 mm / 0.01 rad of the recorded place"
        return targets
    if dist > cfg.inhand_reaim_max_m or drot > cfg.inhand_reaim_max_rad:
        rec["reason"] = f"shift {dist * 1000:.1f} mm / {math.degrees(drot):.1f} deg exceeds the bounds; recorded targets kept"
        log.warning("place(%s): in-hand re-aim skipped: %s", action.object, rec["reason"])
        return targets
    rise = (p_pre_rec.float() - p_rec.float())  # the recorded descent vector, reversed
    place_live = Pose(position=[float(v) for v in p_live.tolist()], orientation=[float(v) for v in q_live.tolist()])
    pre_live = Pose(position=[float(v) for v in (p_live + rise).tolist()], orientation=[float(v) for v in q_live.tolist()])
    out = dict(targets)
    out["place"], out["pre_place"], out["descent"] = place_live, pre_live, [place_live]
    if out.get("retreat"):
        out["retreat"] = [Pose(position=[float(v) for v in (p_live + (as_world_pose(w)[0].float() - p_rec.float())).tolist()], orientation=[float(v) for v in q_live.tolist()])
                          for w in out["retreat"]]
    log.info("place(%s): re-aimed for the in-hand pose: the aligned hand pose is %.1f mm / %.1f deg from the recorded place (meta links %.1f mm / %.1f deg off in the hand)",
             action.object, dist * 1000, math.degrees(drot), off["pos_m"] * 1000, math.degrees(off["rot_rad"]))
    try:
        yield from motions.arm_along(arm, [pre_live], soft_pos_tol=cfg.attach_correct_min_m)
    except SkillError as e:
        rec["pre_place_move_error"] = e.message
        log.warning("place(%s): the move to the re-aimed pre-place did not complete (%s); descending from where the hand is", action.object, e.message)
    rec["applied"] = True
    return out


@dataclass
class AttachCorrection:
    """The attach correction regime (owner 2026-09-20 23:39, attachment only): read from the sidecar's ``attach_correction``
    (the sampler copies ``generation.attach_correction`` of the alias), else from the customs entry of the sidecar's alias
    (a demo-only re-run of an older instance), else these defaults = the behaviour before the knob (two passes at
    ``attach_correct_min_m``). Each pass: settle -> re-measure the male / female offset modulo the symmetry -> lift /
    lateral / descend leg at ``soft_pos_tol_m`` -> settle; the loop stops when the residual is within the targets, the
    improvement falls below ``stop_when_improvement_below_m`` or the passes are exhausted."""

    soft_pos_tol_m: float = 0.003
    settle_steps: int = 20
    max_passes: int = 2
    stop_when_improvement_below_m: float = 0.0
    target_pos_m: float = 0.003
    target_rot_rad: float = 0.03
    source: str = "defaults"

    def as_note(self) -> Dict[str, Any]:
        return {"soft_pos_tol_m": self.soft_pos_tol_m, "settle_steps": self.settle_steps, "max_passes": self.max_passes,
                "stop_when_improvement_below_m": self.stop_when_improvement_below_m, "target_pos_m": self.target_pos_m,
                "target_rot_rad": self.target_rot_rad, "source": self.source}


def attach_correction_regime(ctx: Dict[str, Any], cfg: ExecuteConfig) -> AttachCorrection:
    """See :class:`AttachCorrection`."""
    reg = AttachCorrection(soft_pos_tol_m=cfg.attach_correct_min_m, settle_steps=cfg.attach_settle_steps, target_pos_m=cfg.attach_correct_min_m,
                           target_rot_rad=cfg.attach_correct_min_rad)
    sidecar = ctx.get("sidecar")
    d = (getattr(sidecar, "extra", None) or {}).get("attach_correction") if sidecar is not None else None
    source = "sidecar"
    if not isinstance(d, dict) or not d:
        d, source = None, "defaults"
        alias = getattr(sidecar, "alias", None) if sidecar is not None else None
        if alias:
            try:
                import json

                from ispatialgym.sampling.customs import DEFAULT_CUSTOMS_PATH

                # a plain read (not the schema-validating Customs.load: an unrelated entry mid-edit must not cost the regime)
                with open(DEFAULT_CUSTOMS_PATH) as f:
                    entry = json.load(f).get(alias) or {}
                g = entry.get("generation") or {}
                if isinstance(g.get("attach_correction"), dict) and g["attach_correction"]:
                    d, source = g["attach_correction"], f"customs[{alias}]"
            except Exception as e:  # noqa: BLE001 - the customs are optional here
                log.warning("attach correction: customs lookup for %s failed (%s); defaults", alias, e)
    if d:
        for k in ("soft_pos_tol_m", "stop_when_improvement_below_m", "target_pos_m", "target_rot_rad"):
            if k in d:
                setattr(reg, k, float(d[k]))
        for k in ("settle_steps", "max_passes"):
            if k in d:
                setattr(reg, k, int(d[k]))
    reg.source = source
    return reg


def _attach_at_place(action: Place, motions: Motions, state, ctx) -> Iterator[th.Tensor]:
    """The BEHAVIOR attachment at the place pose (2026-09-19; attachment lid / teapot 0916-205838: the hand reached the
    recorded attach pose within a millimetre, opened, and the lid fell through the collision-filtered teapot onto the
    table — nothing on the demo side ever created the attachment, and OmniGibson's contact-triggered auto-attach cannot
    fire between the filtered members).

    The male / female meta-link offset is read from the live world (the recorded pose does not know the in-hand slip of
    the closure or the descent's shortfall). Above ``attach_correct_min_*`` the hand is moved by the male -> female
    transform — the part is rigid in the hand, so ``T_eef' = T_female @ inv(T_male) @ T_eef`` — as a short Cartesian leg
    (closed-loop correction on the object, not a fallback). Then the assisted grasp is dropped and, in the same step,
    ``AttachedTo.set_value(True)`` snaps the millimetre residual and creates the fixed joint, so the part never
    free-falls; the fingers open afterwards. A residual beyond OmniGibson's 5 cm / 15 deg does not attach and the
    postcondition fails as before (``ctx["attach"]`` records the offsets for the audit)."""
    cfg = _cfg(ctx)
    arm = action.arm
    getter = getattr(state, "attachment_offset", None)
    off = getter(action.object, action.on) if getter is not None else None
    rec: Dict[str, Any] = {"offset_before": None if off is None else {k: v for k, v in off.items() if k != "fix"}}
    ctx["attach"] = rec
    if off is None:
        log.warning("place(%s): no attachment meta-link pair with %s; releasing without an explicit attach", action.object, action.on)
        return
    log.info("place(%s): attach offset %.1f mm / %.1f deg modulo the part's symmetry (raw %.1f deg; male %s -> female %s of %s)", action.object, off["pos_m"] * 1000,
             math.degrees(off["rot_rad"]), math.degrees(off.get("rot_raw_rad", off["rot_rad"])), off["male_link"], off["female_link"], action.on)
    # closed-loop correction passes under the task's regime (owner 2026-09-20 23:39, attachment only; AttachCorrection): each
    # pass moves the hand by the transform carrying the (equivalent) male frame onto the female one, until the residual is
    # within the regime's targets, the improvement falls below its floor or the passes are exhausted; a residual above
    # attach_snap_max_* after that does NOT attach. The joint itself is created at the current relative pose
    # (predicates.attach_in_place), so whatever residual remains is a physical misalignment, never a snap
    reg = attach_correction_regime(ctx, cfg)
    rec["regime"] = reg.as_note()
    passes: List[Dict[str, Any]] = []
    partner0 = None
    try:
        partner0 = tuple(x.float().clone() for x in as_world_pose(state.object_pose(action.on)))
    except Exception:  # noqa: BLE001
        pass
    for k in range(max(0, int(reg.max_passes))):
        need = off["pos_m"] > reg.target_pos_m or off["rot_rad"] > reg.target_rot_rad
        if not need or off["pos_m"] > cfg.attach_correct_max_m:
            break
        p, q = state.eef_pose(arm)
        target = C.compose(off["fix"], (p.float(), q.float()))
        # lift -> lateral / rotational move at height -> descend: the part is clear of the partner while it moves sideways
        # (0920-213721: a straight down-and-sideways leg caught the tilted rim on the neck and pushed the bottle); the leg's
        # tolerance is the regime's (the free-space 1 cm left cork 0920-212757 at 7.8 mm after two passes)
        top = max(float(p[2]), float(target[0][2])) + cfg.attach_correct_lift_m
        via1 = (th.tensor([float(p[0]), float(p[1]), top]), q.float())
        via2 = (th.tensor([float(target[0][0]), float(target[0][1]), top]), target[1])
        try:
            yield from motions.arm_along(arm, [via1, via2, target], soft_pos_tol=reg.soft_pos_tol_m)
            yield from motions.settle(max_steps=reg.settle_steps)
        except SkillError as e:
            log.warning("place(%s): attach correction pass %d did not complete (%s)", action.object, k + 1, e.message)
            rec["correction_error"] = e.message
        off2 = getter(action.object, action.on)
        if off2 is None:
            break
        improvement = float(off["pos_m"]) - float(off2["pos_m"])
        passes.append({**{k2: v for k2, v in off2.items() if k2 != "fix"}, "improvement_m": round(improvement, 5)})
        log.info("place(%s): after correction pass %d: %.1f mm / %.1f deg (improvement %.1f mm)", action.object, k + 1, off2["pos_m"] * 1000,
                 math.degrees(off2["rot_rad"]), improvement * 1000)
        off = off2
        if improvement < reg.stop_when_improvement_below_m and (off["pos_m"] > reg.target_pos_m or off["rot_rad"] > reg.target_rot_rad):
            rec["stopped"] = f"pass {k + 1} improved the residual by {improvement * 1000:.2f} mm < {reg.stop_when_improvement_below_m * 1000:.2f} mm"
            log.info("place(%s): correction stopped: %s", action.object, rec["stopped"])
            break
    rec["offset_after_correction"] = passes[-1] if passes else None
    rec["correction_passes"] = len(passes)
    rec["passes"] = passes
    if partner0 is not None:
        try:
            p1, q1 = (x.float() for x in as_world_pose(state.object_pose(action.on)))
            rec["partner_moved_during_correction"] = {"pos_m": round(float(th.linalg.norm(p1 - partner0[0])), 5), "rot_rad": round(float(C.quat_angle(q1, partner0[1])), 5)}
        except Exception:  # noqa: BLE001
            pass
    if off["pos_m"] > cfg.attach_snap_max_m or off["rot_rad"] > cfg.attach_snap_max_rad:
        # the part would sit visibly off its seat (0920-164504: 17 mm): leave it unattached, the postcondition fails and the executor's
        # retry re-aims the place at the partner's live pose
        rec.update({"attached": False, "refused": f"residual {off['pos_m'] * 1000:.1f} mm / {math.degrees(off['rot_rad']):.1f} deg exceeds the misalignment bound "
                                                    f"({cfg.attach_snap_max_m * 1000:.0f} mm / {math.degrees(cfg.attach_snap_max_rad):.0f} deg)"})
        log.warning("place(%s): %s - not attached", action.object, rec["refused"])
        raise SkillError("execution", f"attach refused: {rec['refused']}")  # still held: the checkpoint rollback + retry re-aim the place
    # drop the assisted grasp and attach in one step (no physics in between): the fixed joint is created at the current relative
    # pose of the meta links (predicates.attach_in_place), so the part does not move when it appears nor when the fingers open
    if state.held_object(arm) is not None:
        state.robot.release_grasp_immediately(arm)
    motions.backend.detach_object(arm)
    ok = bool(state.attach(action.object, action.on))
    rec["attached"] = ok
    rec["alignment"] = getattr(state, "last_attach", None)  # the evaluator's alignment record (modulo the part's symmetry)
    if ok:
        log.info("place(%s): attached to %s (fixed attachment joint at the current relative pose, %s)", action.object, action.on,
                 (rec["alignment"] or {}).get("joint", "?"))
    else:
        log.warning("place(%s): AttachedTo.set_value(True) refused at the place pose (meta links beyond 5 cm / 15 deg?)", action.object)


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


def _retreat_reach(motions: Motions, arm: str, here, d1: th.Tensor, normal_out: th.Tensor, far: float, out: float, ignore: Sequence[Any],
                   cfg: ExecuteConfig) -> Tuple[float, Dict[str, Any]]:
    """The radial distance of a contact escape whose end pose ``here + d1 * far + normal_out`` the collision-aware IK
    (``backend.solve_ik``, world minus ``ignore``) reaches from the current configuration: ``far`` itself, else
    ``far`` shortened by ``retreat_reach_factors`` in turn (never below the escape ``out`` itself, the leg's first
    segment); the shortest tried when none is reachable (the leg's finish / partial acceptance then judge it).
    Returns ``(far, probe record)``; ``far`` unchanged without a backend IK (the test stubs)."""
    backend = getattr(motions, "backend", None)
    solve_ik = getattr(backend, "solve_ik", None)
    robot = getattr(motions, "robot", None)
    factors = tuple(cfg.retreat_reach_factors) or (1.0,)
    if solve_ik is None or robot is None:
        return far, {}
    try:
        backend.update_world(ignore_objects=ignore)
    except Exception as e:  # noqa: BLE001
        log.debug("retreat: world update for the reach probe failed (%r)", e)
    q_seed = robot.get_joint_positions()
    tried: List[Dict[str, Any]] = []
    for f in factors:
        d = max(float(far) * float(f), float(out))
        if tried and abs(d - tried[-1]["far_m"]) < 1e-6:
            break  # the floor was reached already
        end = here[0] + d1 * d + normal_out
        try:
            sol = solve_ik(f"{arm}_arm", {M.EEF_LINK[arm]: (end, here[1])}, q_seed=q_seed, sync_world=False)
        except Exception as e:  # noqa: BLE001 - a probe failure keeps the recorded distance
            log.warning("retreat: reach probe of the escape end pose failed (%r) - keeping %.0f cm", e, far * 100)
            return far, {"error": repr(e)[:160]}
        tried.append({"far_m": round(d, 4), "reachable": sol is not None})
        if sol is not None:
            if d < far - 1e-6:
                log.warning("retreat(%s): the escape end pose %.0f cm out is not reachable from here - shortened to %.0f cm", arm, far * 100, d * 100)
            return d, {"tried": tried}
    log.warning("retreat(%s): no reachable end pose along the escape (%s) - keeping the shortest, %.0f cm", arm, tried, tried[-1]["far_m"] * 100)
    return float(tried[-1]["far_m"]), {"tried": tried, "reachable": False}


def execute_retreat(action: Retreat, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.arm
    if action.release:
        held = state.held_object(arm)
        if held is not None:
            articulated = bool(getattr(held, "n_joints", 0) or len(getattr(held, "joints", {}) or {}))
            # a handle of an articulated object is released with the full opening at once (no staged release: the assisted
            # grasp would re-grasp the handle during the retreat and the door cannot follow the hand) - except a SHEET pinched
            # across its thickness (handle_kind edge_lid, a carton flap 2026-09-19): the full 10 cm opening swept a finger
            # against the 0.8 mm flap and pressed it down 0.25 rad in lockstep with the fingers (0919-070252); it opens to the
            # pre-shape width, slides off the free edge (Retreat._edge_escape) and completes the opening once clear
            art = getattr(getattr(action, "after", None), "articulation", None)
            sheet = bool(art is not None and (getattr(art, "extra", None) or {}).get("handle_kind") == "edge_lid")
            yield from motions.gripper(arm, close=False, staged=(not articulated) or sheet)
            _release(motions, state, arm)
            if articulated and cfg.release_settle_steps > 0:
                # let the constraint-removal transient die before the arm moves off the handle (the leaf keeps its angle
                # through the counterbalance / friction; the torso rock of 0920-002330 happened while the leg started)
                yield from motions.hold(int(cfg.release_settle_steps))
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
    after_x = (getattr(getattr(getattr(action, "after", None), "contact", None), "extra", None) or {}) if isinstance(getattr(action, "after", None), (Push, Brace)) else {}
    if after_x.get("hold"):
        motions.press_hold = None
        held_press = True  # the pressing fist leaves a springy flap straight back along its approach, no radial slide on the paper
    else:
        held_press = bool(after_x.get("straight_retreat"))  # a push-close segment: the fist leaves the tip straight back (no radial slide on the link)
    if after_x.get("kind") == "push_under":
        # a FLOOR blade (IORS push-under) leaves along its own approach axis - straight UP to the recorded pre pose (the default leg
        # already in ``targets``). The radial escape below combined the partner (the table the object went under is in the
        # contact's support list) and the floor into a diagonal escape normal (back + up) and projected the back-off
        # perpendicular to it: 10 cm back AND 7 cm DOWN, the fingertips into the floor within 0.4 s, the arm pressing the
        # floor for ~100 steps and rocking the base 0.4-0.9 deg (0921-021333 / 021337: the wobble at the end of the push)
        held_press = True
        ctx["retreat_escape"] = {"skipped": "push_under: straight up along -a"}
    if isinstance(getattr(action, "after", None), (Brace, Push)) and cfg.contact_retreat_escape_m > 0 and not held_press:
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
            if getattr(action, "escape_far", None):  # a pack push's shorter escape (Retreat.escape_far)
                far = float(action.escape_far)
                out = min(out, far)
            normal_out = n * out if (n is not None and radial is not None) else th.zeros(3)
            # 2026-09-22: the end pose is probed for reachability first and the radial distance shortened when the arm
            # cannot get there (0918-000750: 13 mm unreachable, the straight leg toward it folded the elbow into the upper arm)
            far_used, probe = _retreat_reach(motions, arm, here, d1, normal_out, far, out, ignore, cfg)
            end = here[0] + d1 * far_used + normal_out
            leg = [(here[0] + d1 * out, here[1]), (end, here[1])]
            targets["retreat"] = leg
            finish = _free_leg_finish(state, leg[-1], None, ignore, cfg)
            ctx["retreat_escape"] = {"radial": None if radial is None else [round(float(v), 3) for v in radial.tolist()],
                                     "normal": None if n is None else [round(float(v), 3) for v in n.tolist()], "distance_m": far_used}
            if far_used < far - 1e-6:
                ctx["retreat_escape"]["recorded_distance_m"] = far
            if probe:
                ctx["retreat_escape"]["reach_probe"] = probe
    start_pose = as_world_pose(state.eef_pose(arm))
    try:
        yield from motions.arm_along(arm, leg, finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free,
                                     check_collisions=bool(cfg.retreat_check_collisions), limit_margin=float(cfg.retreat_limit_margin_rad), ignore_objects=ignore)
    except SkillError as e:
        # rush 2026-09-17 (touching 0917-204511): the converged prefix of the leg is executed before the finish is
        # rejected (witness and cuRobo IK both in another branch); a free-space retreat that already stands within
        # retreat_partial_accept_m of its end pose is accepted as partial and judged with the loaded tolerance.
        # 2026-09-22 (0922-235343): a contact escape whose pad already stands the escape distance away from where it left
        # the object is a retreat that did its job, however far the (unreachable) end pose still is - the left pad of the
        # wall push sits at the arm's reach boundary and no end pose along its escape was reachable at all
        if e.reason != "plan_failed":
            raise
        dp, _ = _pose_error(state.eef_pose(arm), targets["retreat"][-1])
        moved = float(th.linalg.norm(as_world_pose(state.eef_pose(arm))[0] - start_pose[0]))
        escape = ctx.get("retreat_escape") or {}
        clear = bool(escape) and "skipped" not in escape and moved >= cfg.contact_retreat_escape_m - 0.005
        if dp > cfg.retreat_partial_accept_m and not clear:
            raise
        log.warning("retreat: the leg ends %.1f mm short of its end pose, the pad %.1f mm from where it left the object (finish rejected: %s) - accepted as a partial free-space retreat",
                    dp * 1000, moved * 1000, e.message)
        ctx["retreat_partial_m"] = dp
        ctx["retreat_moved_m"] = moved
        if clear and dp > cfg.retreat_partial_accept_m:
            targets["retreat"] = list(targets["retreat"][:-1]) + [as_world_pose(state.eef_pose(arm))]  # the postcondition judges the pose the escape ended at
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


def push_residual(geom: PushGeometry, state, obj: str, pose0, art: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], str]:
    """How much of the push is still missing, signed in the push's own sense: for an arc the object rotation still
    needed about +z (rad), for a line the displacement still needed along ``direction`` (m). Measured against the
    task's **goal** (the TRO pose target of the pushed object, world frame, nearest orientation equivalence) when it
    has one, else against the recorded motion (swept angle / distance minus what the object did since the push
    began). Returns ``(residual, source)``; ``(None, reason)`` when nothing can be measured.

    ``art`` (small articulation stage 2, 2026-09-18: ``contacts.push.articulation`` = ``{joint, q_to, ...}``): the pushed
    body is a **link** of ``obj`` on its own joint - the root does not move, the residual is the joint's remaining travel
    ``q_to − q`` in the arc's sense (the pad and the lid turn about the same hinge by the same angle)."""
    if art is not None and geom.kind == "arc":
        # the pad's swept angle maps onto the joint's travel q_from -> q_to (either direction: a lid opened, a flap pressed
        # down); the residual is the unswept part of the recorded arc, in the arc's own sense
        q_now = float(state.joint_position(obj, art["joint"]))
        q_from, q_to = float(art.get("q_from", 0.0)), float(art["q_to"])
        q_arc = float(art.get("q_arc_end", q_to))  # the recorded arc may run past the goal (a push-close's final segment)
        if abs(q_arc - q_from) < 1e-9:
            return 0.0, "goal_orientation"
        return float(geom.angle) * (q_to - q_now) / (q_arc - q_from), "goal_orientation"
    if art is not None:
        # a straight pad path that moves a link on its joint (the lever's vertical lift under a flap edge, 0920-195611: judged by the
        # root's position target the push 'reached its goal' after 7 steps and the flap never moved): the pad's distance maps onto
        # the joint's travel q_from -> q_to; the residual is the unswept part in the line's own sense (metres)
        q_now = float(state.joint_position(obj, art["joint"]))
        q_from, q_to = float(art.get("q_from", 0.0)), float(art["q_to"])
        if abs(q_to - q_from) < 1e-9:
            return 0.0, "goal_orientation"
        return float(geom.distance) * (q_to - q_now) / (q_to - q_from), "goal_orientation"
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
              ctx: Dict[str, Any], label: str, partial: bool = False, speed: Optional[float] = None) -> Iterator[th.Tensor]:
    """A pad leg with the pushed object watched (owner 2026-09-17 23:39): the leg **stops as soon as the object has
    reached its goal** (``push_stop_at_goal``; residual measured against the TRO target within ``push_stop_frac_*`` of
    its tolerance) instead of driving the pad to the recorded end - the recorded arc rotates the painting further than
    the goal (0917-224633: -0.25 rad), and a pad driven on against a blocked target loads the arm and the base (the
    base sprang forward when the pad left) and squeezes the target against the brace (+0.1 rad relaxation)."""
    sign = 1.0 if geom.kind == "line" else math.copysign(1.0, geom.angle)
    margin = _push_stop_margin(state, obj, geom, cfg, ctx)
    if speed is None:
        speed = cfg.push_speed_arc if geom.kind == "arc" else cfg.push_speed_line
    gen = motions.arm_along(arm, waypoints, loaded=True, partial=partial, speed=speed)
    k = 0
    for a in gen:
        yield a
        k += 1
        if not cfg.push_stop_at_goal or margin is None:
            continue
        res, src = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
        if res is None or src not in ("goal_position", "goal_orientation"):
            continue
        if float(res) * sign <= margin:
            log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d steps -> the pad stops here", label, obj, src, float(res) * sign, margin, k)
            ctx.setdefault("push_stopped", []).append({"leg": label, "steps": k, "residual": float(res) * sign})
            gen.close()
            return


def _push_stop_margin(state, obj: str, geom: "PushGeometry", cfg: ExecuteConfig, ctx: Dict[str, Any]) -> Optional[float]:
    """The residual (rad for an arc, m for a line, in the push's sense) at which a push leg stops: ``push_stop_frac_*``
    of the goal tolerance of the pushed object (the TRO target, or the joint target of an articulated push)."""
    tgt = state.pose_target(obj) if hasattr(state, "pose_target") else None
    tol = None
    if tgt is not None:
        tol = getattr(tgt, "orientation_tolerance" if geom.kind == "arc" else "position_tolerance", None)
    art = ctx.get("push_articulation")
    if art is not None and art.get("tolerance") is not None:
        tol = float(art["tolerance"])  # the joint target's tolerance (small articulation stage 2)
        if geom.kind == "line":
            # a straight pad path moving a link (the lever's lift): the residual is in metres of pad travel - so is the tolerance
            span = abs(float(art["q_to"]) - float(art.get("q_from", 0.0)))
            tol = tol * float(geom.distance) / span if span > 1e-9 else tol
    return None if tol is None else float(tol) * (cfg.push_stop_frac_arc if geom.kind == "arc" else cfg.push_stop_frac_line)


def _push_arc_closed_loop(motions: Motions, state, arm: str, obj: str, waypoints: Sequence[Any], geom: "PushGeometry", pose0, cfg: ExecuteConfig,
                          ctx: Dict[str, Any], label: str = "push") -> Iterator[th.Tensor]:
    """The recorded arc of a hinged target, tracked closed-loop (``push_arc_tracking = "closed_loop"``, 2026-09-22): the pad
    first reaches the first contact pose along a straight leg (watched like any push leg), then follows the circle about
    the arc's axis with ``Motions.arm_arc_closed_loop`` - every control step the FK of the MEASURED joints is rotated by
    one step's angle at ``push_speed_arc`` and one differential-IK step gives the command, so the target never runs more
    than a step ahead of where the hinge lets the pad be. The object's rotation about the axis since the push began plays
    the part of the door joint; the leg stops as soon as the object's residual to its TRO goal is within the stop margin
    (``push_stop_at_goal``), when the recorded arc has been swept, or when the object stops following (``push_arc_lag_factor``)."""
    sign = math.copysign(1.0, geom.angle)
    margin = _push_stop_margin(state, obj, geom, cfg, ctx)
    q0_obj = as_world_pose(pose0)[1].float()
    axis = geom.axis

    def swept() -> float:  # the object's rotation about the hinge axis since the push began (the arc's sense)
        return _rotation_about(q0_obj, C.as_tensor(state.object_pose(obj)[1]).float(), axis)

    # the approach to the recorded first contact pose, at the arc's speed and watched. The recorded pre -> init leg is the
    # first part of the push itself: the fist comes up diagonally (up and into the wall) under the slanted low edge, its
    # top meets the edge within the first centimetres and the straight leg carries the object most of the way (0918-000750 /
    # 0922-180242: the whole rotation happened on that leg at 0.10 m/s). The closed-loop arc takes over only AT ``init`` -
    # the pose the sampler put the pad centre on the edge, on the recorded circle: handed over at the first touch
    # (0922-235343) the pad orbited the hinge from a point still in front of the face plane, the orbit has no component
    # toward the wall, and the edge slipped off the fist (the object turned 2 deg in 86 steps).
    gen = motions.arm_along(arm, [waypoints[0]], loaded=True, speed=cfg.push_speed_arc)
    k = 0
    first_contact: Optional[int] = None
    for a in gen:
        yield a
        k += 1
        if first_contact is None and abs(swept()) > cfg.push_contact_detect_rad:
            first_contact = k
            log.info("%s(%s): first contact after %d approach steps (the object turned %.3f rad)", label, obj, k, abs(swept()))
        if cfg.push_stop_at_goal and margin is not None:
            res, src = push_residual(geom, state, obj, pose0, None)
            if res is not None and src in ("goal_position", "goal_orientation") and float(res) * sign <= margin:
                log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d approach steps -> the pad stops here", label, obj, src, float(res) * sign, margin, k)
                ctx.setdefault("push_stopped", []).append({"leg": f"{label}-approach", "steps": k, "residual": float(res) * sign})
                gen.close()
                return
    ctx["push_first_contact"] = {"approach_steps": first_contact, "swept_rad": round(abs(swept()), 4), "approach_total_steps": k}

    motions.stop_requested = False
    gen = motions.arm_arc_closed_loop(arm, geom.pivot, axis, as_world_pose(waypoints[-1]), swept, float(geom.angle), speed=cfg.push_speed_arc,
                                      band_rad=float(margin) if margin is not None else 0.005, lag_factor=cfg.push_arc_lag_factor)
    k = 0
    info: Optional[Dict[str, Any]] = None
    try:
        while True:
            try:
                a = next(gen)
            except StopIteration as e:
                info = e.value
                break
            yield a
            k += 1
            if not cfg.push_stop_at_goal or margin is None:
                continue
            res, src = push_residual(geom, state, obj, pose0, None)
            if res is None or src not in ("goal_position", "goal_orientation"):
                continue
            if float(res) * sign <= margin:
                log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d closed-loop steps -> the pad stops here", label, obj, src, float(res) * sign, margin, k)
                ctx.setdefault("push_stopped", []).append({"leg": label, "steps": k, "residual": float(res) * sign, "closed_loop": True})
                motions.stop_requested = True
    finally:
        motions.stop_requested = False
    ctx["push_arc_closed_loop"] = {"steps": k, "speed_m_s": cfg.push_speed_arc, **({} if info is None else {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in info.items()})}
    log.info("%s(%s): closed-loop arc ended after %d steps (%s)", label, obj, k, None if info is None else info.get("stop"))


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
    extra0 = getattr(action.contact, "extra", None) or {}
    try:
        # the pushed object leaves the collision world for the push plan (program_check._ignore_for) - except an articulated
        # lid (stage 2): the hand travels from the released handle on one side of the standing lid to the underside pre-pose on
        # the other, and a plan through the lid would knock it off its brake
        through = [] if isinstance(extra0.get("articulation"), dict) else [state.object(obj)]
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
        # along the push line the pad only moves BACK (the object farther from the goal than recorded); an object that
        # landed nearer the partner keeps the recorded start - the pad pushes air first and the closed-loop stop ends
        # the push (0918-150927: a 45 deg blade shifted 3.5 cm toward the box descended onto it, joints blocked 0.16 rad)
        along = float(th.dot(off, d0))
        if along > 0:
            off = off - d0 * along
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
    # small articulation stage 2 (2026-09-18): the pushed body is the lid link on its joint - residuals / stop-at-goal /
    # the motion check read the joint (push_residual(..., art)), not the root pose
    ctx["push_articulation"] = dict(extra["articulation"]) if isinstance(extra.get("articulation"), dict) else None
    if ctx["push_articulation"] is not None:
        ctx["push_q0"] = float(state.joint_position(obj, ctx["push_articulation"]["joint"]))
    tip_z = extra.get("fingertip_z")
    # a pitched blade class (push_blade_30 / 45 / 60, the sampler's palm-away candidates) already keeps its palm off the face
    pitched_class = False
    try:
        cls = str(extra.get("pose_class") or "")
        pitched_class = cls.startswith("push_blade_") and float(cls[len("push_blade_"):].rstrip("-")) >= 30.0
    except (TypeError, ValueError):
        pitched_class = False
    if cfg.push_palm_clear_m >= 0 and tip_z is not None and extra.get("blade_half_thickness_m") is not None and not extra.get("palm_clear") and not pitched_class:
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
    # owner 2026-09-18 12:42 (1): a straight table push's pre-pose is brought down to push_pre_dist_m above the first
    # contact pose along the recorded approach (the record's 10 cm) - the hand arrives from the place retreat high above
    # the box, so the long pre-pose descent was a redundant "down, then down again" (0918-014040)
    pre_pose = targets["pre"]
    q_hint_pre = _witness(ctx, "pre")
    if action.contact.mid is None and cfg.push_pre_dist_m is not None and not pitched_class:  # a pitched blade keeps its recorded pre-pose (the shortened one sits at the table)
        p_pre, q_pre = as_world_pose(targets["pre"])
        p_init, _ = as_world_pose(targets["waypoints"][0])
        a_dir = C.approach_axis(q_pre)
        along = float(th.dot(p_init - p_pre, a_dir))  # the recorded pre distance along the approach (positive: pre before init)
        if along > cfg.push_pre_dist_m + 0.005:
            pre_pose = Pose(position=[float(v) for v in (p_init - a_dir * cfg.push_pre_dist_m).tolist()], orientation=[float(v) for v in q_pre.tolist()])
            q_hint_pre = None  # the recorded witness belongs to the recorded pre-pose; cuRobo plans the lowered one
            ctx["push_pre_dist_m"] = cfg.push_pre_dist_m
    hold_here = getattr(motions, "press_hold", None)
    if extra.get("approach_is_slide") and hold_here and hold_here.get("arm") == arm:
        # the lever's lift (two-flap v4, owner 2026-09-20 15:21): this pre-contact IS the press's end pose where the fist already is,
        # in contact with the flap (a cuRobo transit would start "in collision"); the approach leg below slides it into the gap
        log.info("push(%s): the %s fist starts the slide from the press it holds (no transit)", obj, arm)
        ctx["push_pre_skipped"] = True
    else:
        link_ign = []
        if ctx.get("push_articulation") is not None:
            # a push that moves a link on its joint (two-flap carton: the fist on a flap while the other hand's blade stands beside the
            # box): the object's MOVING links (the sheets) leave the planner's world for the transit - 0921-013757: 'start configuration
            # is in collision: right_lid' with the braced blade's padded spheres at the flap's hinge edge; the box body stays an obstacle
            try:
                obj_p = state.object(obj)
                root = getattr(getattr(obj_p, "root_link", None), "body_name", None) or "base_link"
                link_ign = [(obj_p, ln) for ln in getattr(obj_p, "links", {}) if ln != root and ln != "base_link"]
                # the body pieces the OTHER hand's held brace stands in (its padded spheres inside the wall read as a start collision,
                # 0921-015434): the world meshes of this object whose removal clears the current configuration
                try:
                    q_now = state.robot.get_joint_positions()
                    backend = getattr(motions, "backend", None)
                    if backend is not None and hasattr(backend, "world_collision_offenders"):
                        backend.update_world(ignore_objects=list(supports) + list(through), ignore_links=link_ign)
                        offs = backend.world_collision_offenders(q_now, limit=4)
                        for o_ in offs:
                            if f"/{getattr(obj_p, 'name', obj)}/" in str(o_):
                                parts = str(o_).split("/")
                                ln_i = parts.index(getattr(obj_p, "name", obj)) + 1 if getattr(obj_p, "name", obj) in parts else None
                                if ln_i is not None and ln_i < len(parts):
                                    link_ign.append((obj_p, parts[ln_i], parts[-1]))
                except Exception as e:  # noqa: BLE001
                    log.info("push(%s): start-collision offenders not resolved (%r)", obj, e)
                if link_ign:
                    ctx["plan_ignored_links"] = sorted(str(e[1]) + ("/" + str(e[2]) if len(e) > 2 else "") for e in link_ign)
            except Exception:  # noqa: BLE001
                link_ign = []
        yield from reach_pre_pose(motions, state, arm, pre_pose, supports, through, q_hint_pre, ctx, cfg, ignore_links=link_ign)
    _hook(ctx, "fov")(True)
    pose0 = tuple(x.clone() for x in state.object_pose(obj))
    ctx["object_pose0"] = pose0
    geom = push_geometry(targets["waypoints"], targets["direction"])
    if geom.kind == "arc":
        off = C.as_tensor(pose0[0]).float() - geom.pivot
        off = off - geom.axis * float(th.dot(off, geom.axis))
        ctx["push_pivot_offset_m"] = float(th.linalg.norm(off))  # the arc's axis vs the object's COM
        ctx["push_axis"] = [round(float(v), 3) for v in geom.axis.tolist()]
    closed_loop = (geom.kind == "arc" and cfg.push_arc_tracking == "closed_loop" and ctx.get("push_articulation") is None
                   and getattr(getattr(motions, "backend", None), "fk", None) is not None and hasattr(motions, "arm_arc_closed_loop"))
    if closed_loop:
        yield from _push_arc_closed_loop(motions, state, arm, obj, targets["waypoints"], geom, pose0, cfg, ctx, "push")
    else:
        yield from _push_leg(motions, state, arm, obj, targets["waypoints"], geom, pose0, cfg, ctx, "push")
    # 1. hold with the pad in contact until the object is still, then measure the residual to the goal
    yield from _hold_until_still(motions, state, obj, cfg)
    residual, source = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
    ctx["push_residual"] = {"kind": geom.kind, "value": residual, "source": source, "recorded": geom.total,
                            "angle_sign": (math.copysign(1.0, geom.angle) if geom.kind == "arc" else 1.0)}
    # the pad's end pose: where it actually stopped when the leg ended at the goal (0917-235415: judged against the
    # recorded arc end, 64 mm / 0.47 rad away -> 'postcondition of push failed'), else the recorded end
    end_pose = as_world_pose(state.eef_pose(arm)) if (ctx.get("push_stopped") or ctx.get("push_arc_closed_loop")) else as_world_pose(targets["waypoints"][-1])
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
        art_c = ctx.get("push_articulation")
        # a straight pad path moving a link (the lever's lift): the joint quantities (rad) map onto pad metres by distance / span
        m_per_rad = 1.0
        if art_c is not None and geom.kind == "line":
            span = abs(float(art_c["q_to"]) - float(art_c.get("q_from", 0.0)))
            m_per_rad = float(geom.distance) / span if span > 1e-9 else 1.0
        if art_c is not None and art_c.get("tolerance") is not None:
            tol = float(art_c["tolerance"]) * m_per_rad
        if tol is not None:
            cap = max(cap, 2.0 * float(tol))
        if art_c is not None and art_c.get("repush_cap_rad") is not None:
            cap = max(cap, float(art_c["repush_cap_rad"]) * m_per_rad)  # a push-close segment: the link lags the pad ~0.12 rad
        ctx["push_residual"]["repush_cap"] = cap
        if forward and rmin <= abs(residual) <= cap:
            ext = extend_push(geom, end_pose, residual)
            log.info("push(%s): residual %.4f %s (%s) -> pushing on along the extended %s", obj, residual, "rad" if geom.kind == "arc" else "m", source, geom.kind)
            yield from _push_leg(motions, state, arm, obj, ext, geom, pose0, cfg, ctx, "re-push", partial=True)  # best-effort: a stall keeps what it achieved
            yield from _hold_until_still(motions, state, obj, cfg)
            end_pose = as_world_pose(state.eef_pose(arm))  # where the pad actually is (a partial extension ends short of ext[-1])
            after, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
            ctx["push_residual"]["extended_by"] = residual
            ctx["push_residual"]["after"] = after
        elif forward and abs(residual) > cap:
            ctx["push_residual"]["skipped"] = "beyond the recorded arc"
        elif not forward and abs(residual) >= rmin:
            ctx["push_residual"]["skipped"] = "overshoot (a pad cannot pull)"
        # 2b. (rush 2026-09-17) a line push whose arm sits at its reach boundary (touching 0917-211018: the extension
        #     moved the box 0 mm, 3 cm short) is closed with the base: the arm locked at its joints, the base driven
        #     along the push direction by what is still missing (the door base-follow mechanism), pad in contact
        left, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
        if geom.kind == "line" and cfg.push_base_nudge_max_m > 0 and left is not None and cfg.push_base_nudge_min_m <= left <= cfg.push_base_nudge_max_m and ctx.get("push_articulation") is None:
            # (no base nudge for a link on its joint: the lever's vertical lift 0920-201652 drove the base 51 mm with the fist under the flap)
            yield from _base_nudge(motions, state, arm, obj, targets["direction"], left, cfg, "push")
            yield from _hold_until_still(motions, state, obj, cfg)
            end_pose = as_world_pose(state.eef_pose(arm))
            after_nudge, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
            ctx["push_residual"]["base_nudge_m"] = left
            ctx["push_residual"]["after_nudge"] = after_nudge
    # 3. back off along the reversed push direction so the retreat does not rub the pad up the object's face, then
    #    measure the residual the object is left with (a swung painting settles back onto the backed-off pad)
    tangent = push_tangent(geom, end_pose)
    back = (end_pose[0] - tangent * cfg.push_backoff_m, end_pose[1])
    if extra.get("hold"):
        back = end_pose  # a preparatory press (two-flap carton, 2026-09-19): the pad stays on the flap until its Retreat, after the other hand's grasp
        ctx["push_hold"] = True
        motions.press_hold = {"arm": arm, "object": action.contact.object, "link": (extra.get("articulation") or {}).get("link")}
    elif cfg.push_backoff_m > 0:
        yield from motions.arm_along(arm, [back], loaded=True)
    else:
        back = end_pose  # rush: ISG_PUSH_BACKOFF_M=0 keeps the pad on the target (the brace retreats first, ISG_BRACE_RETREATS_FIRST)
    ctx["push_end_pose"] = back
    _hook(ctx, "fov")(False)  # the contact span ends with the contact leg
    yield from _hold_until_still(motions, state, obj, cfg)
    final, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
    # 3b. (owner 2026-09-18 01:33) a line push whose object rocks back once the pad leaves (0918-015654: pushed 1 mm past
    #     the touching position, +8 mm back at the back-off - a blade high on a box tilts it) is closed in further rounds:
    #     re-approach along the push line to the object, nudge the base by what reopened (the settled, upright object
    #     needs less push each round), back off, measure; up to push_renudge_rounds
    rounds = 0
    while (geom.kind == "line" and cfg.push_renudge_rounds > rounds and final is not None and cfg.push_base_nudge_min_m <= final <= cfg.push_base_nudge_max_m
           and cfg.push_backoff_m > 0 and not extra.get("hold")):
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
        final, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
        ctx["push_residual"].setdefault("renudge", []).append({"round": rounds, "final": final})
    ctx["push_residual"]["final"] = final
    p_now = C.as_tensor(state.object_pose(obj)[0]).float()
    ctx["push_object_drift_m"] = float(th.linalg.norm(p_now - C.as_tensor(pose0[0]).float()))


def post_push(action: Push, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    verdict = _eef_reached(state, action.contact.arm, ctx.get("push_end_pose", targets["waypoints"][-1]), cfg, "push", loaded=True)
    if verdict is not None:
        return verdict
    art = ctx.get("push_articulation")
    if art is not None and ctx.get("push_q0") is not None:
        dq = abs(float(state.joint_position(action.contact.object, art["joint"])) - float(ctx["push_q0"]))
        ctx["object_motion"] = (0.0, dq)
        return None if dq > cfg.moved_rot else "no_motion"
    before = ctx.get("object_pose0")
    if before is None:
        return None
    dp, da = _pose_error(state.object_pose(action.contact.object), before)
    ctx["object_motion"] = (dp, da)
    return None if dp > cfg.moved_pos or da > cfg.moved_rot else "no_motion"


#: ``contacts.brace.kind`` of the hinged (wall / axial alignment) targets whose brace stands at the GOAL edge (``brace_targets_at_goal``)
HINGE_BRACE_KINDS = ("side_high", "below_high", "above_low")


def _pose_reachable(motions: Motions, arm: str, pose, ignore: Sequence[Any] = ()) -> Optional[bool]:
    """Whether the collision-aware IK of the backend (world minus ``ignore``) reaches ``pose`` with ``arm`` from the current
    configuration; None without a backend IK (the test stubs - the plan decides then)."""
    backend = getattr(motions, "backend", None)
    solve_ik = getattr(backend, "solve_ik", None)
    robot = getattr(motions, "robot", None)
    if solve_ik is None or robot is None:
        return None
    try:
        backend.update_world(ignore_objects=ignore)
        sol = solve_ik(f"{arm}_arm", {M.EEF_LINK[arm]: as_world_pose(pose)}, q_seed=robot.get_joint_positions(), sync_world=False)
    except Exception as e:  # noqa: BLE001
        log.debug("reach probe of %s failed (%r)", arm, e)
        return None
    return sol is not None


def brace_targets_at_goal(contact, state, cfg: ExecuteConfig, fraction: float = 1.0) -> Optional[Dict[str, Any]]:
    """The brace ``{"pre", "contact", "approach"}`` of a hinged target moved to where the braced point of the object will
    be AT THE GOAL, ``brace_at_goal_gap_m`` back along the brace direction (2026-09-22, owner: "a brace is a safety stop,
    not a force"). The recorded contact was placed against the object at its initial pose; the object rotates about
    its hinge (at its COM: the sampler's constraint anchor = the object's position) by the rotation from its current
    orientation to its TRO goal orientation, and the braced point (the pad centre, ``CONTACT_CENTER_Z`` along the
    approach from the eef origin) moves with it - at the initial pose the fist sat in the path of the last degrees of
    the rotation (0918-000750: 4.7 cm of edge travel into the fist, the two fists clamped the painting, the stop kicked it
    back 0.8 deg). The fist orientation is kept (the brace direction is the pressing direction); the pre pose keeps its
    recorded stand-off along that direction. None when not applicable (not a hinged brace kind, no world orientation
    goal, the object already within 0.01 rad of it, or the gap disabled) - the recorded targets are used then.

    ``fraction`` < 1 places the fist at the edge position of that fraction of the goal rotation (the ladder of
    ``brace_at_goal_fractions`` when the full goal-edge pose is out of the arm's reach from the stance, 0922-180242:
    14 mm short for the right arm) - still ahead of the recorded contact, so the edge meets the fist later than at the
    initial pose, though before the goal."""
    if cfg.brace_at_goal_gap_m < 0 or getattr(contact, "kind", None) not in HINGE_BRACE_KINDS:
        return None
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return None
    obj = contact.object
    tgt = state.pose_target(obj) if hasattr(state, "pose_target") else None
    if tgt is None or getattr(tgt, "orientation", None) is None or str(getattr(tgt, "frame", "world")) != "world":
        return None
    p_obj, q_obj = state.object_pose(obj)
    p_obj, q_obj = C.as_tensor(p_obj).reshape(-1)[:3].float(), C.as_tensor(q_obj).reshape(-1)[:4].float()
    goal_q = C.as_tensor(tgt.orientation).reshape(-1)[:4].float()
    rel = C.qmul(goal_q, C.qinv(q_obj))  # the world rotation taking the object from where it is to its goal
    rel = rel / max(float(th.linalg.norm(rel)), 1e-9)
    if float(rel[3]) < 0:
        rel = -rel
    angle = 2.0 * math.atan2(float(th.linalg.norm(rel[:3])), float(rel[3]))
    if angle < 0.01:
        return None
    if fraction < 1.0:
        rel = C.quat_from_axis_angle(rel[:3] / max(float(th.linalg.norm(rel[:3])), 1e-9), angle * fraction)
    a = C.as_tensor(contact.direction).reshape(-1)[:3].float()
    a = a / max(float(th.linalg.norm(a)), 1e-9)
    c_p, c_q = as_world_pose(contact.contact)
    pre_p, pre_q = as_world_pose(contact.pre)
    pad = c_p + a * C.CONTACT_CENTER_Z  # the pad centre = the braced point of the object at its initial pose
    pad_goal = p_obj + C.quat_rotate(rel, pad - p_obj)  # that material point at the goal (rotation about the hinge at the COM)
    gap = float(cfg.brace_at_goal_gap_m)
    new_c = pad_goal - a * C.CONTACT_CENTER_Z - a * gap
    standoff = float(th.dot(c_p - pre_p, a))
    if standoff <= 1e-3:
        standoff = float(C.PRE_DIST)
    new_pre = new_c - a * standoff
    n_wp = 3  # actions.N_WAYPOINTS: 2 + the goal
    approach = [Pose(position=[float(v) for v in (new_pre + (new_c - new_pre) * (k / n_wp)).tolist()], orientation=[float(v) for v in c_q.tolist()]) for k in range(1, n_wp + 1)]
    return {
        "pre": Pose(position=[float(v) for v in new_pre.tolist()], orientation=[float(v) for v in pre_q.tolist()]),
        "contact": Pose(position=[float(v) for v in new_c.tolist()], orientation=[float(v) for v in c_q.tolist()]),
        "approach": approach,
        "shift_m": float(th.linalg.norm(new_c - c_p)),
        "goal_rotation_rad": angle,
        "fraction": fraction,
        "gap_m": gap,
    }


def execute_brace(action: Brace, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.contact.arm
    _hook(ctx, "fov")(False)
    recorded = {k: targets[k] for k in ("pre", "contact", "approach") if k in targets}
    q_hint_pre = _witness(ctx, "pre")
    supports = contact_ignores(state, action.contact.object)  # the wall / table the braced object rests against
    at_goal = None
    probes: List[Dict[str, Any]] = []
    try:
        # the goal-edge brace, at the largest fraction of the goal rotation whose pre and contact poses the collision-aware IK
        # reaches from this stance (0922-180242: the full goal edge lay 14 mm beyond the right arm's reach)
        for frac in tuple(cfg.brace_at_goal_fractions) or (1.0,):
            cand = brace_targets_at_goal(action.contact, state, cfg, fraction=float(frac))
            if cand is None:
                break
            ok = [_pose_reachable(motions, arm, cand[k], supports) for k in ("pre", "contact")]
            probes.append({"fraction": float(frac), "shift_m": round(cand["shift_m"], 4), "reachable": None if any(o is None for o in ok) else all(ok)})
            if all(o is not False for o in ok):
                at_goal = cand
                break
    except Exception as e:  # noqa: BLE001 - the recorded brace stands when the goal edge cannot be computed
        log.warning("brace(%s): goal-edge brace not computed (%r) - bracing at the recorded contact", action.contact.object, e)
        at_goal = None
    if probes and at_goal is None:
        log.warning("brace(%s): no goal-edge brace pose within reach (%s) - bracing at the recorded contact", action.contact.object, probes)
        ctx["brace_at_goal"] = {"fallback": "recorded contact", "reason": "no reachable goal-edge pose", "probes": probes}
    if at_goal is not None:
        targets.update({k: at_goal[k] for k in ("pre", "contact", "approach")})
        q_hint_pre = None  # the recorded witness belongs to the recorded pre-pose; cuRobo plans the moved one
        ctx["brace_at_goal"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at_goal.items() if k in ("shift_m", "goal_rotation_rad", "fraction", "gap_m")}
        if len(probes) > 1:
            ctx["brace_at_goal"]["probes"] = probes
        log.info("brace(%s): the fist stands where the braced edge will be at %.0f %% of the goal rotation (%.1f deg) + %.0f mm gap: contact moved %.1f cm",
                 action.contact.object, at_goal["fraction"] * 100, math.degrees(at_goal["goal_rotation_rad"]), at_goal["gap_m"] * 1000, at_goal["shift_m"] * 100)
    opening = (getattr(action.contact, "extra", None) or {}).get("fingers_opening_m")
    if opening is not None:
        # a WIDE stop (owner 2026-09-19 00:49): the fingers pre-shaped to the recorded opening so both tips land on the braced
        # face, spread along it - a lever arm against the pivot a single closed tip cannot resist (touching 0918-224622: the
        # partner yawed 0.8 deg about the point stop and the pair lost contact)
        yield from motions.gripper_opening(arm, float(opening))
    else:
        yield from motions.gripper(arm, close=True)
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in supports]
    try:
        yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, [], q_hint_pre, ctx, cfg)
    except SkillError as e:
        if at_goal is None or e.reason != "plan_failed":
            raise
        # the goal-edge pre pose is out of this stance's reach: the recorded brace (validated by the sampler) stands instead
        log.warning("brace(%s): the goal-edge pre pose is not reachable (%s) - bracing at the recorded contact", action.contact.object, e.message)
        targets.update(recorded)
        ctx["brace_at_goal"] = {**ctx.get("brace_at_goal", {}), "fallback": "recorded contact", "reason": e.message[:160]}
        yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, [], _witness(ctx, "pre"), ctx, cfg)
    _hook(ctx, "fov")(True)
    obj = action.contact.object
    if cfg.brace_touch_stop_m > 0:
        # owner 2026-09-22 11:27: the brace must not act on the object - the approach leg is watched and stops the moment the braced
        # object moves (0921-234836: the stop's approach shoved the box 2.4 cm; 0921-030848: the pads 5 mm into the face rocked the
        # base), then backs off a little along the approach so the pads rest just off the face
        p0 = C.as_tensor(state.object_pose(obj)[0]).float().clone()
        gen = motions.arm_along(arm, targets["approach"], loaded=True)
        k = 0
        for a in gen:
            yield a
            k += 1
            moved = float(th.linalg.norm(C.as_tensor(state.object_pose(obj)[0]).float() - p0))
            if moved > cfg.brace_touch_stop_m:
                gen.close()
                here = as_world_pose(state.eef_pose(arm))
                a_dir = C.approach_axis(here[1])
                back = (here[0] - a_dir * cfg.brace_touch_backoff_m, here[1])
                log.info("brace(%s): the object moved %.1f mm after %d steps of the approach - the stop is in touch; backing off %.0f mm", obj, moved * 1000, k, cfg.brace_touch_backoff_m * 1000)
                ctx["brace_touch_stop"] = {"steps": k, "moved_m": moved, "backoff_m": cfg.brace_touch_backoff_m}
                yield from motions.arm_along(arm, [here, back])
                break
    else:
        yield from motions.arm_along(arm, targets["approach"], loaded=True)
    _hook(ctx, "fov")(False)  # the contact span ends with the contact leg (the brace then holds still)


def post_brace(action: Brace, state, targets, ctx) -> Optional[str]:
    if ctx.get("brace_touch_stop"):
        return None  # the stop met the object before its recorded pose: it stands where the object is
    return _eef_reached(state, action.contact.arm, targets["contact"], _cfg(ctx), "brace", loaded=True)


def joint_tolerance(art) -> float:
    """spec §8: revolute max(0.05 rad, 0.10·|Δq|); prismatic max(0.01 m, 0.10·|Δq|)."""
    span = abs(art.q_goal - art.q_init)
    return max(0.05 if art.joint_type == "revolute" else 0.01, 0.10 * span)


def _hold_until_joint_still(motions: Motions, state, obj: str, joint: str, cfg: ExecuteConfig) -> Generator[th.Tensor, None, Tuple[int, float]]:
    """Hold (arm targets as they are) until the articulated joint stops changing between steps — ``follow_still_rad``
    per step for ``follow_still_steps`` consecutive steps —, at most ``follow_settle_steps``. Returns ``(steps, q)``
    (read it with ``yield from``)."""
    prev = state.joint_position(obj, joint)
    still, n = 0, 0
    for n in range(1, int(cfg.follow_settle_steps) + 1):
        yield from motions.hold(1)
        q = state.joint_position(obj, joint)
        still = still + 1 if abs(q - prev) < cfg.follow_still_rad else 0
        prev = q
        if still >= cfg.follow_still_steps:
            break
    return n, prev


def execute_base_follow(action: BaseFollow, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    """Move the joint with the handle grasped, **judged on the joint** (the door), not on the arm: the recorded handle
    path (arm mode) / base poses (base mode) are followed until the joint is within ``follow_stop_frac`` of its
    tolerance from q_goal, or the recorded motion ends; the arm's stored deflection is then unwound (the drives stop
    pressing on a door its friction already holds) and the hold ends once the joint is still."""
    cfg = _cfg(ctx)
    art = action.articulation
    q0 = state.joint_position(art.object, art.joint)
    ctx["q0"] = q0
    tol = joint_tolerance(art)
    sign = 1.0 if art.q_goal - art.q_init >= 0 else -1.0
    band = cfg.follow_stop_frac * tol
    note: Dict[str, Any] = {"mode": targets["mode"], "q0": q0, "stop_band_rad": band}

    def reached() -> bool:
        return sign * (state.joint_position(art.object, art.joint) - art.q_goal) >= -band

    if targets["mode"] == "base":
        if not targets.get("base_poses"):
            raise SkillError("plan_failed", "base_follow: no recorded base poses in base mode", stage="targets")
        gen = motions.base_follow(targets["base_poses"], action.arm)
    elif targets["mode"] == "hybrid":
        # D-011 C: the recorded dense path (base pose + arm joints per <= 10 deg step) is driven and tracked together
        if not targets.get("follow_path") or len(targets["follow_path"]) < 2:
            raise SkillError("plan_failed", "base_follow: no recorded follow path in hybrid mode", stage="targets")
        # the joint axis from the recorded handle contacts: the arm-only steps then follow the true arc between poses
        contacts = [w.contact if hasattr(w, "contact") else w["contact"] for _, w in sorted(art.waypoints.items(), key=lambda kv: float(kv[0]))] if art.waypoints else []
        axis = joint_axis_from_contacts(contacts) if len(contacts) >= 3 else None
        note["arc_axis"] = None if axis is None else {"centre": [round(float(v), 3) for v in axis[0].tolist()], "axis": [round(float(v), 3) for v in axis[1].tolist()]}
        if targets.get("follow_segments"):
            note["policy"] = "sequential"
            gen = _follow_sequential(action, motions, state, targets["follow_segments"], ctx, axis, note)
        else:
            gen = motions.base_follow_hybrid(targets["follow_path"], action.arm, axis=axis)
    else:
        gen = motions.arm_along(action.arm, targets["handle_poses"], converge=False)
    k = 0
    stopping = 0
    q_phase0 = None
    for a in gen:
        yield a
        k += 1
        if getattr(motions, "follow_phase", "arm") == "base":
            # sequential policy: the door must stand still while the base drives (anchored move: the hand stays on the
            # handle; release / re-grasp: the hand is off it) - a moving joint here is a broken assumption, abort
            q_now = state.joint_position(art.object, art.joint)
            if q_phase0 is None:
                q_phase0 = q_now
            elif abs(q_now - q_phase0) > max(cfg.follow_still_rad * 20, 0.02):
                gen.close()
                raise SkillError("execution", f"base_follow: {art.object}/{art.joint} moved {abs(q_now - q_phase0):.3f} rad while the base drove (door not stationary)", stage="base_move")
            continue
        q_phase0 = None
        if stopping:
            stopping += 1
            if stopping > 60:  # the motion did not wind down on its own: cut it
                gen.close()
                break
            continue
        if cfg.follow_stop_at_goal and reached():
            q = state.joint_position(art.object, art.joint)
            log.info("base_follow(%s): %s/%s at %.4f reached the goal band (%.4f ± %.4f) after %d steps -> the follow stops here", action.arm, art.object, art.joint, q,
                     art.q_goal, band, k)
            note["stopped_at_goal"] = {"steps": k, "q": q}
            if targets["mode"] == "hybrid" and hasattr(motions, "stop_requested"):
                # cooperative stop: the hybrid follow decelerates the base and freezes the arm target, then returns
                # (closing the generator cut the base velocity in one step: a 2 m/s^2 jolt at the goal, 0920-164632)
                motions.stop_requested = True
                stopping = 1
                continue
            gen.close()
            break
    note["motion_steps"] = k
    if targets["mode"] == "arm" and cfg.follow_stop_at_goal and not reached() and (getattr(art, "extra", None) or {}).get("hinge_world") is not None:
        # owner 2026-09-21: the recorded motion ended short of the goal band (0921-021042: 0.06 rad, the hand a few mm short of the last
        # pose) - the hand rides ON along the hinge arc by what is missing (+ the band), the live hinge (the object may have shifted),
        # until the joint reaches the band; the remaining shortfall stays in the record
        try:
            ex = art.extra or {}
            hinge = C.as_tensor(ex["hinge_world"]).reshape(-1)[:3].float()
            axis = C.unit(C.as_tensor(ex["axis_world"]).reshape(-1)[:3].float())
            rec = ex.get("object_pose")
            if isinstance(rec, dict):
                # the hinge moves with the object: T_live o T_rec^-1 applied to the recorded hinge point / axis
                live = as_world_pose(state.object_pose(art.object))
                rec_pose = (C.as_tensor(rec["position"]).float(), C.as_tensor(rec["orientation"]).float())
                d_pose = C.compose(live, C.invert(rec_pose))
                dp_, dq_ = C.as_tensor(d_pose[0]).float(), C.as_tensor(d_pose[1]).float()
                hinge = C.quat_rotate(dq_, hinge) + dp_
                axis = C.unit(C.quat_rotate(dq_, axis))
            q_now = float(state.joint_position(art.object, art.joint))
            d_ext = sign * (abs(art.q_goal - q_now) + band + 0.02)
            here = as_world_pose(state.eef_pose(action.arm))
            ext = C.orbit_waypoints(here, hinge, C.quat_from_axis_angle(axis, d_ext), n=3, frame="world")
            log.info("base_follow(%s): the recorded motion ended at q %.3f (goal %.3f): riding on along the hinge arc by %.3f rad", action.arm, q_now, art.q_goal, d_ext)
            gen2 = motions.arm_along(action.arm, list(ext), converge=False)
            k2 = 0
            for a in gen2:
                yield a
                k2 += 1
                if reached():
                    gen2.close()
                    break
            note["extended"] = {"rad": float(d_ext), "steps": k2, "q": float(state.joint_position(art.object, art.joint))}
        except Exception as e:  # noqa: BLE001
            log.warning("base_follow(%s): the extension past the recorded motion failed: %r", action.arm, e)
    if cfg.follow_unwind:
        # the drives stop pressing on the door: the hold target becomes the measured configuration (the wind-up under
        # the friction load stays in the record), re-read every step for follow_unwind_steps so the elastic strain bleeds
        # off gradually instead of snapping; the release then imparts nothing
        total = motions.unwind(action.arm)
        for _ in range(max(0, int(cfg.follow_unwind_steps) - 1)):
            yield from motions.hold(1)
            total += motions.unwind(action.arm)
        note["unwind_rad"] = total
    hold_steps, q_end = yield from _hold_until_joint_still(motions, state, art.object, art.joint, cfg)
    note.update({"hold_steps": hold_steps, "q_end": q_end, "shortfall_rad": sign * (art.q_goal - q_end)})
    ctx["follow"] = note


def _ramp_arm_to_q(motions: Motions, arm: str, q_arm: Sequence[float], speed: Optional[float] = None, loaded: bool = False) -> Generator[th.Tensor, None, float]:
    """A straight joint-space ramp of ``arm`` onto the 7-joint configuration ``q_arm`` at ``speed`` rad/s (the follow's
    arm speed by default), no planner. For a target the sampler recorded and collision-checked a short distance away
    (the retreat witness after a differential-IK retreat, the pre-grasp witness when the planner refuses a start
    configuration that is 9 mm inside its self-collision padding: dishwasher_wide 0921-041230). Returns the joint
    distance (rad)."""
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    q_now = robot.get_joint_positions().clone().float()
    q_goal = q_now.clone()
    q_goal[idx] = C.as_tensor(q_arm).reshape(-1).float()
    d = float(th.max(th.abs(q_goal[idx] - q_now[idx])))
    rate = float(speed if speed is not None else motions.cfg.follow_arm_speed)
    n = max(2, int(math.ceil(d / max(rate * motions.dt, 1e-6))))
    q_start = q_now.clone()
    q_start[idx] = motions.state.arm_target[arm].clone().float() if motions.state.arm_target.get(arm) is not None else q_now[idx]
    traj = th.stack([q_start + (q_goal - q_start) * (k / n) for k in range(1, n + 1)])
    yield from motions.execute_joint_trajectory(traj, arms=[arm], loaded=loaded, converge=True)
    return d


def _follow_sequential(action: BaseFollow, motions: Motions, state, segments: List[Dict[str, Any]], ctx: Dict[str, Any], axis, note: Dict[str, Any]) -> Iterator[th.Tensor]:
    """Owner policy 2026-09-20 20:19, strictly one thing at a time: ``arm`` segments track the handle arc between the
    recorded poses (``arc_between`` about the joint axis, differential IK, no convergence wait); ``release_move_regrasp``
    segments (the only base move, owner 2026-09-20 23:58) unwind, release, back off, drive with the arm
    frozen (or in the idle posture), re-grasp from the pre-grasp witness and re-attach. The goal-band stop is
    cooperative (``motions.stop_requested``) and only honoured between arm poses."""
    cfg = _cfg(ctx)
    arm = action.arm
    art = action.articulation
    obj = state.object(art.object)
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    log_moves: List[Dict[str, Any]] = []
    note["segments"] = log_moves
    motions.stop_requested = False
    motions.follow_phase = "arm"
    # the sense of the fitted hinge axis relative to the joint's positive direction, from two recorded poses of distinct q
    # (the fit returns a direction only): +1 when a positive joint delta rotates the hand positively about the axis
    axis_sign = 0.0
    if axis is not None:
        rec = [(float(p["q"]), as_world_pose(p["eef"])[0]) for s in segments if s.get("type") == "arm" for p in (s.get("poses") or []) if p.get("eef") is not None and p.get("q") is not None]
        for (qa, pa), (qb, pb) in zip(rec[:-1], rec[1:]):
            if abs(qb - qa) > 1e-3:
                ra = (pa - axis[0]) - axis[1] * float((pa - axis[0]) @ axis[1])
                rb = (pb - axis[0]) - axis[1] * float((pb - axis[0]) @ axis[1])
                ang = math.atan2(float(th.cross(ra, rb, dim=0) @ axis[1]), float(ra @ rb))
                if abs(ang) > 1e-3:
                    axis_sign = 1.0 if (ang > 0) == (qb > qa) else -1.0
                    break
        note["arc_axis_sign"] = axis_sign
    for j, seg in enumerate(segments):
        kind = seg.get("type")
        if kind == "arm":
            eef_prev = motions.eef_pose(arm)
            for p in seg.get("poses") or []:
                if motions.stop_requested:
                    return
                if p.get("eef") is None:
                    continue
                target = as_world_pose(p["eef"])
                try:
                    if cfg.follow_arm_tracking == "closed_loop" and axis is not None and p.get("q") is not None:
                        info = yield from motions.arm_arc_closed_loop(arm, axis[0], axis[1], target, lambda: float(state.joint_position(art.object, art.joint)), float(p["q"]),
                                                                     speed=motions.cfg.follow_eef_speed, band_rad=cfg.follow_arm_band_rad)
                        log_moves.append({"type": "arm_pose_closed_loop", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}})
                        if info.get("stop") == "door_lags":
                            log.warning("base_follow: arm pose q = %.3f: the joint did not follow the commanded arc (%.3f rad commanded, joint %.3f -> %.3f)", float(p["q"]),
                                        info["commanded_rad"], info["q_from"], info["q_end"])
                            if p.get("arm_q") is not None:  # the recorded IK witness puts the hand on h(q): the open-loop way, as a fallback
                                d_w = yield from _ramp_arm_to_q(motions, arm, p["arm_q"], loaded=True)
                                log_moves.append({"type": "arm_witness_ramp", "q": p.get("q"), "rad": round(d_w, 4), "after": "door_lags"})
                    elif cfg.follow_arm_tracking == "reanchored" and axis is not None and axis_sign != 0.0 and p.get("q") is not None:
                        # the arc of this pose is the hand's ACTUAL pose (the FK of the measured joints - the wound-up drives'
                        # own frame) rotated about the fitted hinge by the measured joint delta to the pose's q. The recorded
                        # target positions are not used: the fingers meet the bar before the recorded grasp pose (dishwasher
                        # 0921-235247: the hand attached 20 mm above it, radius 0.612 vs 0.592 m), so every recorded follow
                        # target sat 2 cm off the true handle circle and the drives stayed saturated through the whole pull
                        q_now = float(state.joint_position(art.object, art.joint))
                        dq = float(p["q"]) - q_now
                        q_full = robot.get_joint_positions().clone().float()
                        pose_fk = as_world_pose(motions.backend.fk([M.EEF_LINK[arm]], q_full)[M.EEF_LINK[arm]])
                        n_wp = max(1, int(math.ceil(abs(dq) / 0.03)))
                        wps = C.orbit_waypoints(pose_fk, axis[0], C.quat_from_axis_angle(axis[1], axis_sign * dq), n=n_wp, frame="world") if abs(dq) > 1e-4 else []
                        if wps:
                            yield from motions.arm_along(arm, wps, loaded=True, speed=motions.cfg.follow_eef_speed, converge=False)
                        r_vec = (pose_fk[0] - axis[0]) - axis[1] * float((pose_fk[0] - axis[0]) @ axis[1])
                        log_moves.append({"type": "arm_pose_reanchored", "q": p.get("q"), "q_from": round(q_now, 4), "n_wps": len(wps), "radius_m": round(float(th.linalg.norm(r_vec)), 4),
                                          "recorded_target_offset_mm": round(1000 * float(th.linalg.norm(wps[-1][0] - target[0])), 1) if wps else None})
                    else:
                        wps = arc_between(eef_prev, target, axis[0], axis[1]) if axis is not None else [target]
                        yield from motions.arm_along(arm, wps, loaded=True, speed=motions.cfg.follow_eef_speed, converge=False)
                except SkillError as e:
                    if e.details.get("stage") != "diff_ik" or p.get("arm_q") is None:
                        raise
                    # the differential IK from the loaded (wound-up) configuration gave up on the arc (door 0921-000629:
                    # 10 mm at waypoint 4/7 with no joint at a limit); the sampler's IK witness of this pose is the
                    # configuration that puts the hand on h(q) - ramp onto it, the attached handle follows
                    d_w = yield from _ramp_arm_to_q(motions, arm, p["arm_q"], loaded=True)
                    log.warning("base_follow: arm segment pose q = %.3f: %s -> ramped onto the recorded IK witness (%.3f rad)", float(p.get("q") or 0.0), (e.message or e.reason)[:140], d_w)
                    log_moves.append({"type": "arm_witness_ramp", "q": p.get("q"), "rad": round(d_w, 4)})
                eef_prev = target
            log_moves.append({"type": "arm", "q_from": seg.get("q_from"), "q_to": seg.get("q_to")})
        elif kind == "release_move_regrasp":
            # 1. unwind gradually, hold until the joint is still, release (full opening for a handle), settle
            total = motions.unwind(arm)
            for _ in range(max(0, int(cfg.follow_unwind_steps) - 1)):
                yield from motions.hold(1)
                total += motions.unwind(arm)
            yield from _hold_until_joint_still(motions, state, art.object, art.joint, cfg)
            yield from motions.gripper(arm, close=False, staged=False)
            _release(motions, state, arm)
            # the gripper-in-FOV span closes at the release and re-opens at the next pre-grasp (qa.py rule (b): "the
            # retreat legs and the transit to the next pre-grasp are unmonitored; every grasp -> release pair is its own
            # span") - door 0921-045829: the retreated hand, 0.32 m below the camera, tripped the monitor during the drive
            fov_hook = (ctx.get("hooks") or {}).get("fov")
            if fov_hook is not None:
                fov_hook(False)
            yield from motions.hold(int(cfg.release_settle_steps))
            # 2. back off to the retreat pose (differential IK), then onto the sampler's retreat witness - the same hand
            # pose, but the configuration the search collision-checked; the differential IK drifts in the null space
            # and left link5 9 mm inside the trunk's self-collision padding, where no plan may start (0921-041230)
            yield from motions.arm_along(arm, [as_world_pose(seg["retreat_eef"])], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
            d_ret = 0.0
            if seg.get("arm_q_retreat"):
                d_ret = yield from _ramp_arm_to_q(motions, arm, seg["arm_q_retreat"])
                log.info("base_follow: release at q = %.3f: retreat %.2f m, arm ramped onto the recorded retreat witness (%.3f rad)", float(seg.get("q") or 0.0), float(seg.get("retreat_m") or C.PRE_DIST), d_ret)
            if seg.get("hold_posture") == "idle" and seg.get("arm_q_hold"):
                yield from motions.arm_to(arm, q={arm: C.as_tensor(seg["arm_q_hold"]).reshape(-1).float()})
            motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
            motions.follow_phase = "base"
            try:
                yield from motions.track_path([as_pose2d(b) for b in seg["base_path"]], tracker_cfg=replace(motions.cfg.base_tracker, v_max=(motions.cfg.base_follow_speed, motions.cfg.base_follow_speed, motions.cfg.base_follow_yaw_rate)))
            finally:
                motions.follow_phase = "arm"
            yield from motions.settle(max_steps=20)
            # 3. re-grasp: pre-grasp (witness), straight approach, ramped closure, attach
            pre, grasp = as_world_pose(seg["pre_grasp_eef"]), as_world_pose(seg["grasp_eef"])
            yield from motions.gripper(arm, close=False)
            try:
                yield from motions.arm_to(arm, pose=pre, q_hint=_witness(ctx, f"regrasp_pre[{j}]"), ignore_objects=[obj])
            except SkillError as e:
                if "start configuration is in collision" not in str(e) or not seg.get("arm_q_pre"):
                    raise
                # the planner refuses the start state (its self-collision padding); the recorded pre-grasp witness is a
                # short free-space move away - ramp onto it in joint space, at the follow's arm speed
                d_pre = yield from _ramp_arm_to_q(motions, arm, seg["arm_q_pre"])
                log.warning("base_follow: re-grasp pre-grasp plan refused its start state (%s); ramped onto the recorded pre-grasp witness (%.3f rad) instead", str(e)[:160], d_pre)
            if fov_hook is not None:
                fov_hook(True)  # the pre-grasp is reached: the next grasp -> release span opens
            finish = ApproachFinish(grasp, q_hint=_witness(ctx, f"regrasp[{j}]"), ignore_objects=[obj], max_stall=cfg.approach_finish_max_stall, max_remaining=cfg.approach_finish_max_remaining)
            yield from motions.arm_along(arm, [grasp], finish=finish)
            fixed_ag = cfg.handle_grasp_fixed_ag
            if fixed_ag:
                _patch_ag_joint_type(robot, "FixedJoint")
            try:
                yield from motions.gripper(arm, close=True, rate=motions.cfg.gripper_close_rate)
            finally:
                if fixed_ag:
                    _patch_ag_joint_type(robot, None)
            if state.held_object(arm) is not obj:
                raise SkillError("failed_grasp", f"{art.object} not in the {arm} hand after the re-grasp of segment {j}", stage="regrasp")
            motions.backend.attach_object(arm, obj)
            motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
            log_moves.append({"type": "release_move_regrasp", "q": seg.get("q"), "strides": len(seg["base_path"]), "hold_posture": seg.get("hold_posture"), "unwind_rad": round(total, 4),
                              "m": round(math.hypot(seg["base_to"][0] - seg["base_from"][0], seg["base_to"][1] - seg["base_from"][1]), 3)})
        else:
            raise SkillError("plan_failed", f"base_follow: unknown follow segment type {kind!r}", stage="targets")
    yield from motions.hold(3)


def post_base_follow(action: BaseFollow, state, targets, ctx) -> Optional[str]:
    art = action.articulation
    if state.held_object(action.arm) is not state.object(art.object):
        return "failed_grasp"
    q = state.joint_position(art.object, art.joint)
    ctx["q"] = q
    tol = joint_tolerance(art)
    span = art.q_goal - art.q_init
    sign = 1.0 if span >= 0 else -1.0
    # a recorded passive continuation (the sampler's probe: released at that waypoint the link swings on to q_goal by
    # itself) ends the hand's part there, as does a stage-2 ``follow_to`` (the fist push takes the joint on); otherwise
    # the hand takes the joint to q_goal
    stop = art.follow_to
    q_target = art.q_at(stop) if stop is not None else art.q_goal
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
