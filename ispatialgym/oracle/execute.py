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
  assisted grasp must hold the object (``failed_grasp``), then backend ``attach_object``; the instances of
  ``grasps[].may_touch`` must still rest on their support (``distractor_displaced``; after the Lift as well).
* ``Lift`` / ``Reorient``  ``arm_along`` over the recorded segment; post: still held, eef at the segment end.
* ``Place``      ``arm_to(pre_place)`` → ``arm_along(descent)`` → ``gripper(open)`` (the open command +1 at once, OmniGibson's
  assisted grasp lets go; backend ``detach_object``) → ``arm_along(retreat)`` → settle; post: released, object at rest
  (``not_settled``) or ``attached`` for ``kind="attach"``: the descent ends at a hover pose above the seat, the hand aligns
  the part laterally and in yaw there, touches it down on its partner, opens in one step, and the task's attach rule joins the
  part where it rests on its partner.
* ``Push``       ``gripper(close)`` → ``arm_to(pre)`` → ``arm_along(init → mid → goal)``; post: eef at the goal
  contact, the pushed object moved (``no_motion``). A record with a stroke plan (``strokes``) pushes in strokes, the object measured and the
  blade re-aimed before each.
* ``Brace``      ``gripper(close)`` → ``arm_to(pre)`` → ``arm_along(pre → contact)``; held (no further command).
* ``BaseFollow`` ``base_follow(b_k)`` with the arm locked, or ``arm_along(h(q_k))`` in arm mode; post: joint at
  ``q_goal`` (spec §8 tolerance) or ``passive_from`` reached.
* ``PushClose``  per chunk: stance by backend IK → ``track_path`` → ``gripper(close)`` → ``arm_to(pre)`` → guarded
  approach → ``arm_along(arc)`` → back off (a planned push, a cabinet door's push finish: the recorded stance and contact);
  post: joint within the spec §8 tolerance of the lower limit.
* ``BasePush``   ``gripper(close)`` → ``arm_to(pre)`` → guarded approach → ``track_path`` along the recorded base line with the
  arm held (stopped at NextTo + a push-on, a sideways drift, or the end; a lagging object re-contacted, a turning one realigned
  onto the line from its centre to its goal) → unload; a two-hand push does each step with both hands; post: the object moved
  (``no_motion``).
* ``Retreat``    ``gripper(open)`` (+ release / detach) when ``release`` → ``arm_along(retreat)``.

Tolerance categories: an eef / base / trunk that did not reach its recorded target is ``execution`` (a motion did
not converge); a task quantity off its target (camera pose, joint value) is ``tolerance``; a push / articulation
that moved nothing is ``no_motion``; a grasp the assisted grasp did not attach is ``failed_grasp``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Protocol, Sequence, Tuple

import torch as th

from ispatialgym.oracle.actions import (
    Action,
    BaseFollow,
    BasePush,
    Brace,
    Grasp,
    Lift,
    Navigate,
    Place,
    Push,
    PushClose,
    Reorient,
    Retreat,
    SetTorso,
    Transition,
    VOCABULARY,
)
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.motions import ApproachFinish, Motions, SkillError, arc_between, as_pose2d, as_world_pose, joint_axis_from_contacts, links_kw
from ispatialgym.oracle.sidecar import Pose
from ispatialgym.robot import base_path as BP
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
    #: an axis-aligned leg (``Navigate.axis_leg``, owner 2026-10-13): the driven heading within this of the leg's (rad) and the
    #: base within this of the recorded polyline (m) at every step, else the postcondition fails
    axis_leg_yaw_tol: float = math.radians(2.0)
    axis_leg_corridor_m: float = 0.03
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
    push_stop_frac_line: float = 0.0
    #: (2026-09-22, alignment-world_vertical 0918-000750) pad speed along a push ARC about a hinge (m/s; the approach to the
    #: first contact keeps ``cartesian_speed``). The hinge torque is friction + damping * omega, so at the former 0.10 m/s
    #: (0.6 rad/s on the painting) the viscous term alone was 3.6 of the 6.8 N.m the pad had to supply (~40 N at 0.17 m):
    #: the elbow drive saturated at its 25 N.m cap, the wrist drives went bang-bang against the contact and the painting
    #: shook. 3.5 cm/s (0.2 rad/s) roughly halves the required force. ``push_speed_line``: straight pushes (None = default).
    #: A push contact's recorded ``push_speed_m_s`` (:func:`push_speed_of`) replaces both for that push's pad legs.
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
    #: (owner 2026-10-14 ~04:25, touching#02) a base nudge carries every arm with the base: a brace standing on the far side of
    #: the pair (``ctx["braces_held"]``) left the braced object with the base and the nudge pushed the pair past the goal
    #: (pair1013b_c8_k10: the partner 26.5 mm, a brace-retry rung in every demo). On: the brace arm is re-solved every control
    #: step so its commanded hand pose stays where it was in the world (``_NudgeBraceHold``), and the nudge stops once the
    #: pushed object is at its goal or the base has advanced less than ``push_nudge_stall_m`` over ``push_nudge_stall_steps``
    #: steps (pressing the pair against the brace). Off: the former nudge (the brace moves with the base).
    push_nudge_hold_brace: bool = True
    push_nudge_stall_steps: int = 10
    push_nudge_stall_m: float = 0.001
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
    #: (2026-09-19, attachment) the re-aim move to the pre-place stops within this of its target
    attach_correct_min_m: float = 0.003
    #: owner 2026-09-25 (cork 0925-195316 'attach refused 16.6 mm / 4.9 deg'): the attach's insertion legs - the re-aim move to the
    #: pre-place, the descent, the hover alignment - track the rotation to this soft tolerance (the default 0.03 rad let the male
    #: arrive 4.4 deg tilted)
    attach_leg_soft_rot_tol: float = 0.01
    #: the attach's insertion legs keep the differential IK this far from the joint limits (the cork's correction passes swung the
    #: residual by +-10 mm / +-3 deg between passes: commanded millimetre moves at a joint limit produce uncommanded rotations)
    attach_leg_limit_margin_rad: float = 0.05
    attach_correct_min_rad: float = 0.03
    attach_settle_steps: int = 20
    #: (2026-09-20 22:30) the place is re-aimed for the part's LIVE in-hand pose at the pre-place: the male -> female fix measured
    #: there gives the hand pose that aligns the links (the fingertip pinch of a 1.1 cm cap lets it pivot 3-5 deg / 5 mm about the
    #: closing axis before the assisted grasp locks, and the carry leg's soft tolerance adds up to 1 cm); applied when the shift
    #: from the recorded place is within these bounds
    inhand_reaim_max_m: float = 0.03
    inhand_reaim_max_rad: float = 0.35
    #: an attach place re-aims at the partner's LIVE pose: the recorded pre-place / place / descent / retreat / carry are moved by the
    #: partner's displacement since the program start when it exceeds this (0920-164504: the bottle knocked 4.7 cm during the grasp
    #: phase, the descent came down 4 cm beside the neck)
    partner_shift_min_m: float = 0.005
    #: (owner 2026-10-10) the attach place holds the part at the contact hold down to a HOVER pose - the seated hand pose raised along
    #: the seat axis until the lowest point of both the part and the fingertips is this far above the partner's top under them
    #: (the partner's collision convexes whose footprint meets theirs, widened by ``attach_hover_footprint_margin_m``) ...
    attach_hover_clear_m: float = 0.003
    attach_hover_footprint_margin_m: float = 0.01
    #: ... aligns the male link there with the female one - laterally (across the seat axis) and in yaw about it modulo the part's
    #: symmetry, the height and the tilt kept - in the closed-loop passes of the task's correction regime (``AttachCorrection``),
    #: then TOUCHES DOWN (owner 2026-10-10: released at the hover, the contact hold's squeeze let go at once kicked the cap 8 mm
    #: aside): the hand lowered along the seat axis at ``attach_touchdown_speed_m_s`` until PhysX reports the part in contact with
    #: its partner, at most the hover's part clearance + ``attach_touchdown_extra_m``, then ``attach_touchdown_hold_steps`` still;
    #: a touchdown that fails releases where the hand is (a plain drop) - never a partial opening ...
    attach_touchdown_speed_m_s: float = 0.01
    attach_touchdown_extra_m: float = 0.004
    attach_touchdown_hold_steps: int = 3
    attach_touchdown_soft_pos_tol_m: float = 0.0005
    #: ... opens in one step to +1 and holds still until the task's attach rule has joined the part (in contact), at most this many
    #: steps, before the retreat
    attach_join_wait_steps: int = 45
    #: (owner 2026-09-27, first-demo v2.1 A3, bar soap 0927-034001: "the robot wobbles at the grasp and the release") a library pinch
    #: whose finger envelope (both finger meshes, closed..open) at the grasp pose reaches below the grasped object's lowest point + this
    #: is lifted by the deficit along world z, the pads' fingertip-side end kept ``grasp_support_pad_margin_m`` below the object's top
    #: (the attachment rule's FINGER_SUPPORT_CLEARANCE_M): the recorded bar-soap pinch aimed the fingertips 1 cm into the tray floor,
    #: the hand stopped on it and the arm pressed ~100 N into it for the whole closure, bending torso_joint1 by 7.6 mrad ...
    grasp_support_clearance_m: float = 0.004
    grasp_support_pad_margin_m: float = 0.005
    #: ... unless the faces at the lifted pads' fingertip-side end slope more than this (|normal z|, sin 20 deg): a curved side above
    #: its widest section - the milk bottle lying on its side, pinched 13 mm higher, rolled 14 deg under the closure (0927-210646)
    grasp_support_face_slope_max: float = 0.34
    #: ... and a support place whose object hangs lower in the hand than the recorded ``eef_in_object`` by more than this is raised by
    #: the difference, so the object meets its support instead of being driven into it (the same pinch pressed the destination tray)
    place_inhand_raise_min_m: float = 0.002
    #: owner 2026-10-11 (objects on an INTERIOR support surface - a shelf inside a bookcase / wardrobe / hall tree, ``interior_split``):
    #: the object rests on the recorded board when its bottom (at the grasp, or where the place sets it down) lies within
    #: ``interior_surface_tol_m`` of the board height; the board pieces under it are the support's collision meshes whose top face is
    #: within ``interior_board_z_margin_m`` of that height and at most ``interior_board_max_thickness_m`` tall
    #: (``SimWorld.support_interior``'s rule), within ``interior_board_margin_m`` of the object's footprint (the open fingers beside it)
    interior_surface_tol_m: float = 0.06
    interior_board_z_margin_m: float = 0.03
    interior_board_max_thickness_m: float = 0.12
    interior_board_margin_m: float = 0.10
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
    #: a retreat leg that fails (plan_failed) short of both the partial acceptance and the contact escape is replaced by a planned
    #: transit to its end pose from where the hand is, before the action fails (owner 2026-09-26, axial 0926-061151 retreat[2])
    retreat_plan_fallback: bool = True
    #: a slide-then-lift retreat's recorded witness is used only within this of the leg's end pose (m; ``_retreat_witness``)
    retreat_witness_pose_tol_m: float = 0.02
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
    #: a following brace (``Brace.follow``) moves its recorded targets with the braced object once it has moved this far since the
    #: program start (m): the recorded stop stands only ``stop_clearance_m`` (5 mm) off the face
    brace_follow_min_m: float = 0.001
    #: a following push (``Push.follow``) moves the pushed object's goal with that instance once it has moved this far since the
    #: program start (m): the touching goal is contact with it
    push_goal_follow_min_m: float = 0.001
    #: a contact pre-pose next to an ignored support (wall, table) is reached via a standoff this far away from the
    #: support, planned with the support present, then a straight leg (``reach_pre_pose``); a standoff that does not track from
    #: the pre-pose witness: the pre-pose is planned directly with the witness first (owner 2026-10-12)
    pre_pose_standoff_m: float = 0.08
    #: (owner 2026-09-18 12:42) no standoff when the pre-pose is already this clear of every support (m): planned directly
    pre_pose_standoff_skip_m: float = 0.08
    #: (owner 2026-10-12 item 2b, axial#02) no standoff either for a brace / push pre-contact standing at least this far ABOVE every
    #: support near it - inside their footprints, over their tops (a table / desk contact, not a wall) - planned directly as above
    #: (m; None = off, only ``pre_pose_standoff_skip_m``; the owner's value 0.06)
    pre_pose_standoff_skip_above_m: Optional[float] = None
    #: (owner 2026-09-18 12:42) a straight push's pre-pose sits this far above / before its first contact pose (m; None = the record's 10 cm)
    push_pre_dist_m: Optional[float] = 0.04
    #: (touching#04 2026-10-12) the transit to a free object's push pre-contact is planned with the pushed object in the collision world -
    #: the leg protects it, and the planner leaves out only what a motion touches on the leg where it touches it - and with it left out
    #: only when no plan exists with it present (``reach_push_pre``). False: left out throughout, as before
    push_transit_target_present: bool = True
    #: ... a recorded pre-contact witness (solved by the sampler with the object left out) whose arm spheres stand closer than this to the
    #: object's box is planned with it left out at once: the planner refuses that goal only after its witness rounds and posture detours
    push_transit_witness_clear_m: float = 0.01
    #: the carry leg lift end -> pre-place (FOV rule (b), owner 2026-09-16 17:10): ``"straight"`` = a straight Cartesian
    #: leg (through the recorded ``grasps[k].carry`` waypoint when the sampler wrote one), differential IK with the
    #: free-leg finish, both ends recorded in frame; ``"trajopt"`` = the former free trajopt transit (the regression
    #: reference: nextto 0916-152503 swung 40 cm back toward the robot and left the frame for 106 steps)
    carry_leg: str = "straight"
    #: owner 2026-09-30 (first-demo v3.0 F2, door open 0928-042225): a ``release_move_regrasp`` ramps onto the recorded
    #: ``arm_q_retreat`` only when it lies within this of the measured joints after the back-off (rad, largest arm joint). The ramp
    #: absorbs the differential IK's null-space drift (0921-041230); 70 ramps of 38 door / dishwasher runs measured 0.012-0.595 rad,
    #: the far IK solution of the v2.2 door instance 3.505 rad (4 runs). A farther witness keeps the arm where the back-off left it
    #: when the drive with it is clear (whole body, ``follow_keep_drive_margin_m`` = the sampler's drive check) and it keeps
    #: ``follow_keep_self_clearance_m`` from cuRobo's self-collision padding (the sampler's ``hybrid_follow.self_clearance_m``)
    follow_retreat_ramp_max_rad: float = 1.0
    follow_keep_drive_margin_m: float = 0.01
    follow_keep_self_clearance_m: float = 0.02
    #: owner 2026-10-09 (audit of the v3.0-reconciled demos: "too much push at the end makes the robot wobble ... door case is
    #: unacceptable"): a push-close pushes only until the leaf is closed - the arm's arc and the base drag stop once the joint is within
    #: ``push_close_stop_rad`` of ``q_lower`` or the trunk is bent ``push_close_stop_trunk_rad`` off its target by a leaf that no longer
    #: gives, and the hold targets are unloaded before the fist backs off. 1008: the drag drove the fist 6.7 cm on into the closed leaf,
    #: torso_joint1 ended 4.2 deg behind its target and sprang forward at the back-off - the head lunged 78 mm
    push_close_stop_rad: float = 0.004
    push_close_stop_trunk_rad: float = math.radians(0.3)
    #: owner 2026-10-12 ~04:20 (a cabinet door's push finish, ``PushClose.plan``): the base drives back to the planned stance only when it
    #: stands farther than this from it (m / rad); the pre-contact is planned to the sampler's arm witness while the leaf is within
    #: ``push_finish_hint_max_rad`` of the angle the chunk was planned at; a leaf within ``push_finish_landing_tol_rad`` of the planned
    #: landing is not pushed again
    push_finish_restance_m: float = 0.05
    push_finish_restance_rad: float = 0.10
    push_finish_hint_max_rad: float = 0.05
    push_finish_landing_tol_rad: float = math.radians(1.0)
    #: ... a grasp approach stops once a finger presses the support the object rests on - the hand touches it and the trunk is bent
    #: ``grasp_support_stop_trunk_rad`` off its target - and is unloaded before the closure (brussels sprouts: a pinch of a 4.4 cm
    #: sphere aims the fingertips 24 mm below its bottom; the hand stopped on the table and pressed it up to 196 N, torso_joint1 0.8 deg,
    #: the head 17.5 mm, springing back at the lift)
    grasp_support_stop_trunk_rad: float = math.radians(0.1)
    #: ... a dropped object is let go of (the open command at once, like every release) after it was brought straight down from
    #: the recorded drop pose (``lower_drop``) until its bottom is ``drop_lower_clear_m``
    #: above whatever is under it - at most ``drop_lower_max_m``, a collision-checked leg that ends where the arm stalls, this far from
    #: its joint limits - so it does not fall far: the sandal released 24 cm above its rest at once tumbled 7 deg, bounced 14 deg and
    #: landed 14 mm aside, and the push under the table then turned it 60 deg. Below ``drop_lower_min_m`` of planned lowering the
    #: recorded drop pose is kept
    drop_lower_max_m: float = 0.20
    drop_lower_clear_m: float = 0.02
    drop_lower_min_m: float = 0.01
    drop_lower_limit_margin_rad: float = 0.05
    #: ... and the fully opened fingers keep this much above whatever is under them (a container's rim). A leg whose plan jumps is tried
    #: again shorter, at these fractions of the planned lowering
    drop_lower_finger_clear_m: float = 0.015
    drop_lower_fractions: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15)
    #: ... a next-to / touching place keeps the fully open fingers ``place_partner_clear_m`` from the partner at the release: the
    #: assisted grasp lets go on the full open command only, and the pad on the partner's side swept 15 mm toward it (touching:
    #: the box of tissues shoved 6 mm). The place (and the push that follows it) is moved away from the partner by what is missing,
    #: up to ``place_partner_clear_max_m``
    place_partner_clear_m: float = 0.005
    place_partner_clear_max_m: float = 0.03
    place_partner_clear_spacing_m: float = 0.004
    #: owner 2026-10-09 (blender open: the free-standing body tipped 0.9 deg and slid 2 mm under the arm-mode follow while the torso bent
    #: 0.41 deg): the arm-mode follow of a free-standing object turns the hand rigidly about the hinge as it stands when the follow begins
    #: (``follow_arc_step_rad`` per waypoint; <= 0 = the recorded chords), streamed in chunks of ``follow_reaim_chunk`` waypoints, the
    #: next chunk at ``follow_slow_factor`` of the pace while the torso is bent more than ``follow_slow_trunk_rad``
    follow_reaim_chunk: int = 4
    follow_arc_step_rad: float = 0.03
    follow_slow_trunk_rad: float = math.radians(0.15)
    follow_slow_factor: float = 0.5


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


#: a ``grasps[].may_touch`` object counts as lifted with the target when its AABB centre rose more than this during the lift
MAY_TOUCH_LIFT_RISE_M = 0.02


def _may_touch_of(action) -> Tuple[Tuple[str, ...], Optional[str]]:
    """``(objects, stay_on)`` of the ``grasps[].may_touch`` record behind a Grasp (or the Grasp a Lift follows)."""
    grasp = getattr(action, "grasp", None) if isinstance(action, Grasp) else getattr(getattr(action, "after", None), "grasp", None)
    return grasp.may_touch if grasp is not None and hasattr(grasp, "may_touch") else ((), None)


def may_touch_objects(state, action, pre_grasp: bool = False) -> Dict[str, Any]:
    """``{instance: scene object}`` the grasp may touch or displace (``grasps[].may_touch``, owner 2026-10-12 A2b: disembedding's parts
    and distractors): out of the planner's world for the grasp's approach finish and the lift's, as in the sampler's grasp check.
    ``pre_grasp``: only those the pre-grasp may overlap too (``may_touch.pre_grasp``, the puzzle's fixed-layout parts) - supports of
    the transit to the pre-grasp."""
    grasp = getattr(action, "grasp", None)
    names = getattr(grasp, "may_touch_pre_grasp", ()) if pre_grasp else _may_touch_of(action)[0]
    out: Dict[str, Any] = {}
    for inst in names:
        try:
            out[inst] = state.object(inst)
        except Exception:  # noqa: BLE001
            continue
    return out


def _aabb_centre(obj) -> Optional[th.Tensor]:
    try:
        lo, hi = obj.aabb
        return 0.5 * (th.as_tensor(lo).float() + th.as_tensor(hi).float())
    except Exception:  # noqa: BLE001
        return None


def rests_over(obj, support) -> Optional[bool]:
    """Whether ``obj``'s AABB centre lies over ``support``'s footprint: inside its bounding box aligned with its own yaw
    (``get_base_aligned_bbox``) in x / y, else inside its world AABB; None when an AABB is unavailable (the test stubs)."""
    c = _aabb_centre(obj)
    if c is None:
        return None
    fn = getattr(support, "get_base_aligned_bbox", None)
    if callable(fn):
        try:
            centre, orn, extent, _ = fn(xy_aligned=True)
            local = C.transform_points(C.invert((th.as_tensor(centre).float(), th.as_tensor(orn).float())), c.reshape(1, 3))[0]
            half = 0.5 * th.as_tensor(extent).float()
            return bool(abs(float(local[0])) <= float(half[0]) and abs(float(local[1])) <= float(half[1]))
        except Exception:  # noqa: BLE001 - the world AABB below
            pass
    try:
        lo, hi = (th.as_tensor(v).float() for v in support.aabb)
    except Exception:  # noqa: BLE001
        return None
    return bool(float(lo[0]) <= float(c[0]) <= float(hi[0]) and float(lo[1]) <= float(c[1]) <= float(hi[1]))


def may_touch_verdict(action, state, ctx: Dict[str, Any], stage: str) -> Optional[str]:
    """Owner 2026-10-12 (A2b): after the grasp and after the lift every ``grasps[].may_touch`` object must still rest on its ``stay_on``
    support (disembedding: each part / distractor on the source tray) - its AABB centre over the support's footprint
    (:func:`rests_over`) and, after the lift, not risen more than ``MAY_TOUCH_LIFT_RISE_M`` (lifted with the target; ``ctx
    ["may_touch_z0"]`` from the lift's start). ``ctx["may_touch"]`` records per object whether it is over the support, its rise and
    how far it moved since the program start; ``"distractor_displaced"`` when one is off, else None (also without a record)."""
    objs, on = _may_touch_of(action)
    if not objs or on is None:
        return None
    try:
        support = state.object(on)
    except Exception:  # noqa: BLE001
        return None
    z0 = ctx.get("may_touch_z0") or {}
    init = getattr(state, "init_poses", None) or {}
    per: Dict[str, Any] = {}
    off: List[str] = []
    for inst in objs:
        try:
            obj = state.object(inst)
        except Exception:  # noqa: BLE001
            continue
        over = rests_over(obj, support)
        c = _aabb_centre(obj)
        r: Dict[str, Any] = {"over_support": over}
        if c is not None and inst in z0:
            r["rise_m"] = round(float(c[2]) - float(z0[inst]), 4)
        if inst in init:
            try:
                r["moved_m"] = round(float(th.linalg.norm(state.object_pose(inst)[0] - init[inst][0])), 4)
            except Exception:  # noqa: BLE001
                pass
        per[inst] = r
        if over is False or r.get("rise_m", 0.0) > MAY_TOUCH_LIFT_RISE_M:
            off.append(inst)
    ctx.setdefault("may_touch", {}).update({"stage": stage, "stay_on": on, "objects": per, "displaced": off})
    if off:
        log.warning("%s(%s): %s %s off %s (owner A2b: the parts / distractors must stay on it)", stage, getattr(action, "object", "?"), off,
                    "is" if len(off) == 1 else "are", on)
        return "distractor_displaced"
    return None


def _interior_fact(srf: Any) -> Optional[Dict[str, Any]]:
    """``srf`` when it is the record of an interior support surface (``surface: "interior"`` with its support and board height)."""
    if not isinstance(srf, dict) or srf.get("surface") != "interior" or not srf.get("support") or srf.get("z_world") is None:
        return None
    return srf


def interior_support_surface(template_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The interior support surface the sampler worked on for the instance's template: ``template_meta.json``
    ``template_facts.support_surface`` (``tabletop.support_surface``: a shelf inside a bookcase / wardrobe / hall tree - the
    ``support`` scene object and the board height ``z_world``) when its surface is ``interior``; None for a support top or a
    template without the record. The executor hands it to every action (``ctx["support_surface"]``, ``interior_split``)."""
    facts = (template_meta or {}).get("template_facts")
    return _interior_fact(facts.get("support_surface")) if isinstance(facts, dict) else None


@dataclass
class InteriorSplit:
    """The collision world of one action on an object resting on an interior support surface (``interior_split``)."""

    support: Any
    #: the action's ignore set without the support
    objects: List[Any]
    #: ``(support, link name, collision mesh name)`` of the board pieces under the object - the only part of the support left out of
    #: the legs that start or end at the contact
    links: List[Tuple[Any, str, str]]
    note: Dict[str, Any]


def board_pieces(support, z: float, lo_xy: Sequence[float], hi_xy: Sequence[float], cfg: "ExecuteConfig") -> List[Tuple[Any, str, str]]:
    """``(support, link name, collision mesh name)`` of the support's board at world height ``z`` under the xy box ``lo_xy`` ..
    ``hi_xy``: its collision meshes whose top face lies within ``interior_board_z_margin_m`` of ``z``, at most
    ``interior_board_max_thickness_m`` tall (the board pieces of ``SimWorld.support_interior``), overlapping the box."""
    out: List[Tuple[Any, str, str]] = []
    for link_name, link in (getattr(support, "links", None) or {}).items():
        for cm in (getattr(link, "collision_meshes", None) or {}).values():
            try:
                plo, phi = cm.aabb
                plo, phi = C.as_tensor(plo).float().reshape(-1), C.as_tensor(phi).float().reshape(-1)
            except Exception:  # noqa: BLE001 - a piece without an AABB stays an obstacle
                continue
            if abs(float(phi[2]) - float(z)) > cfg.interior_board_z_margin_m or float(phi[2] - plo[2]) > cfg.interior_board_max_thickness_m:
                continue
            if float(plo[0]) > float(hi_xy[0]) or float(phi[0]) < float(lo_xy[0]) or float(plo[1]) > float(hi_xy[1]) or float(phi[1]) < float(lo_xy[1]):
                continue
            out.append((support, str(link_name), str(cm.prim_path).rsplit("/", 1)[-1]))
    return out


def interior_split(state, ctx: Dict[str, Any], name: str, ignore: Sequence[Any], place=None, arm: Optional[str] = None) -> Optional[InteriorSplit]:
    """Owner 2026-10-11 (objects on an interior support surface): the collision world of an action on ``name`` when the sampler
    recorded an interior surface for the instance (``ctx["support_surface"]``, ``interior_support_surface``) and the object rests on
    it - its bottom within ``interior_surface_tol_m`` of the board height, at its live pose or, with ``place`` (an eef pose) and
    ``arm``, where the hand that holds it sets it down. The support leaves ``ignore``: the board above, the side walls and the back
    panel stay obstacles for every leg of the action, and only the board pieces under the object (``board_pieces``, its footprint
    widened by ``interior_board_margin_m``) are left out of the legs that start or end at the contact - the reach bands' rule for
    the grasp pose (``reach_bands --surface-z``). None outside that situation (no record, the object elsewhere, the support neither
    in ``ignore`` nor under the object): the action's collision world as it was."""
    srf = _interior_fact(ctx.get("support_surface"))
    if srf is None:
        return None
    cfg = _cfg(ctx)
    try:
        support = state.object(str(srf["support"]))
        lo, hi = state.object(name).aabb
        lo, hi = C.as_tensor(lo).float().reshape(-1)[:3].clone(), C.as_tensor(hi).float().reshape(-1)[:3].clone()
        if place is not None:
            shift = as_world_pose(place)[0].float() - as_world_pose(state.eef_pose(arm))[0].float()
            lo, hi = lo + shift, hi + shift
        z = float(srf["z_world"])
    except Exception as e:  # noqa: BLE001 - no scene object / AABB / hand pose: the collision world as it was
        log.debug("interior split of %s unavailable (%r)", name, e)
        return None
    if abs(float(lo[2]) - z) > cfg.interior_surface_tol_m:
        return None
    m = float(cfg.interior_board_margin_m)
    links = board_pieces(support, z, (float(lo[0]) - m, float(lo[1]) - m), (float(hi[0]) + m, float(hi[1]) + m), cfg)
    was = any(o is support for o in ignore)
    if not was and not links:
        return None
    note = {"support": getattr(support, "name", str(srf["support"])), "z_world": round(z, 4), "object_bottom_z": round(float(lo[2]), 4),
            "support_was_ignored": was, "board_pieces": [f"{ln}/{mesh}" for _, ln, mesh in links]}
    if not links:
        note["no_board_piece"] = "no board piece of the support under the object: the whole support stays an obstacle"
        log.warning("%s: no board piece of %s at z %.3f under it - the whole support stays an obstacle", name, note["support"], z)
    return InteriorSplit(support, [o for o in ignore if o is not support], links, note)


def _plan_ignored(objects: Sequence[Any], links: Sequence[Tuple[Any, ...]] = ()) -> List[str]:
    """``ctx["plan_ignored"]``: the objects and collision pieces left out of the planner's world (``object:link[:mesh]``)."""
    return [getattr(o, "name", str(o)) for o in objects] + [":".join([str(getattr(e[0], "name", e[0]))] + [str(v) for v in e[1:]]) for e in links]


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


def height_above_supports(pose, objects: Sequence[Any], near: float = 0.15) -> Optional[float]:
    """How far ``pose``'s position stands above the ``objects`` near it (within ``near`` of their AABBs, ``escape_direction``'s
    rule): the smallest height over their tops when it stands inside every such footprint, above the top; None when one of them
    is beside or over it (a wall, a board overhead) or none is near."""
    p = as_world_pose(pose)[0]
    best = None
    for o in objects:
        try:
            lo, hi = o.aabb
        except Exception:  # noqa: BLE001
            continue
        lo, hi = C.as_tensor(lo).float(), C.as_tensor(hi).float()
        if float(th.linalg.norm(p - th.minimum(th.maximum(p, lo), hi))) > near:
            continue
        if not (bool(th.all(p[:2] >= lo[:2])) and bool(th.all(p[:2] <= hi[:2])) and float(p[2]) > float(hi[2])):
            return None
        h = float(p[2] - hi[2])
        best = h if best is None else min(best, h)
    return best


def _counted(gen: Iterator[th.Tensor], n: List[int]) -> Iterator[th.Tensor]:
    """``gen``'s actions, counted in ``n[0]`` (did a motion that raised move the robot first?)."""
    try:
        for a in gen:
            n[0] += 1
            yield a
    finally:
        gen.close()


def reach_pre_pose(motions: Motions, state, arm: str, pre, supports: Sequence[Any], through: Sequence[Any], q_hint, ctx: Dict[str, Any], cfg: "ExecuteConfig",
                   ignore_links: Sequence[Any] = (), accept_other_branch: bool = False, skip_above_m: Optional[float] = None) -> Iterator[th.Tensor]:
    """Reach a contact pre-pose that the sampler checked with its supports ignored (a wall / table 2 cm from the
    pads). The supports leave the collision world only for the **last straight leg**: the trajopt goes to a standoff
    ``pre + standoff · n`` (``n`` away from the supports) with the supports present — a trajopt with the wall ignored
    planned the hand *through* the wall region and the arm jammed against it (1.39 rad joint error on the painting
    push) — then a straight ``arm_along`` closes the standoff. ``through`` (the pushed target) is ignored for both.
    A standoff that does not track from the witness ``q_hint`` (owner 2026-10-12, axial#02) is not planned first: the pre-pose
    is planned straight to with the witness, the supports present (the skip branch's call); the bare standoff only when that
    plan fails before the robot moved. ``ctx["pre_pose_standoff"]["branch"]``: witness / direct / planner.
    ``accept_other_branch``: passed to ``Motions.arm_to`` for the planned legs to the pre-pose (a push, owner 2026-09-26).
    ``skip_above_m``: a pre-pose at least this far above every support near it is planned directly too (the brace / push callers
    pass ``pre_pose_standoff_skip_above_m``)."""
    if not supports:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through), ignore_links=list(ignore_links), accept_other_branch=accept_other_branch)
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    n = escape_direction(state, pre, supports)
    if n is None:
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(supports) + list(through), ignore_links=list(ignore_links), accept_other_branch=accept_other_branch)
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    clearance = support_clearance(state, pre, supports)
    above = height_above_supports(pre, supports) if skip_above_m is not None else None
    if (clearance is not None and clearance >= cfg.pre_pose_standoff_skip_m) or (above is not None and above >= skip_above_m):
        # owner 2026-09-18 12:42 (1): a pre-pose already this clear of its supports (a table push's pre-pose 10-16 cm
        # above the table) is planned directly with the supports present - the standoff excursion (up 8 cm, then down
        # again) was a redundant raise before the descent; the wall contacts (pre-pose 2-5 cm off the wall) keep it
        ctx["pre_pose_standoff"] = {"skipped": True, "clearance_m": None if clearance is None else round(clearance, 3),
                                    **({"above_m": round(above, 3)} if above is not None else {})}
        yield from motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through), ignore_links=list(ignore_links), accept_other_branch=accept_other_branch)
        if getattr(motions, "last_relaxed_links", None):
            ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
        return
    p, q = as_world_pose(pre)
    standoff = (p + n * cfg.pre_pose_standoff_m, q)
    ctx["pre_pose_standoff"] = {"direction": [round(float(v), 3) for v in n.tolist()], "distance_m": cfg.pre_pose_standoff_m,
                                **({"clearance_m": round(clearance, 3)} if clearance is not None else {})}
    q_standoff = standoff_configuration(motions, arm, pre, standoff, q_hint) if q_hint is not None else None
    if q_standoff is not None:
        # the standoff in the **witness's IK branch**: differential IK from the recorded pre-pose witness back along the
        # short leg, then a joint-space plan there - a pose plan let cuRobo pick its own branch and the whole contact
        # chain (arc, back-off, retreat) ran 3.18 rad away from the sampler's witnesses (axial retreat 0916-202403)
        ctx["pre_pose_standoff"]["branch"] = "witness"
        yield from motions.arm_to(arm, q=q_standoff, ignore_objects=list(through), ignore_links=list(ignore_links))
    else:
        if q_hint is not None:
            # owner 2026-10-12 (axial#02 c3_k0: the push pre-contact 7.2 cm above the desk, right_arm_joint4 0.05 rad from its limit -
            # ~1 cm of the 8 cm rise reachable there, the bare standoff failed IK): a standoff the witness cannot track to may be
            # out of this stance's reach; the witness itself is the sampler's validated pre-pose configuration
            ctx["pre_pose_standoff"]["branch"] = "direct"
            log.warning("reach_pre_pose(%s): the standoff does not track from the witness - planning straight to the pre-pose with the witness, supports present", arm)
            moved = [0]
            try:
                yield from _counted(motions.arm_to(arm, pose=pre, q_hint=q_hint, ignore_objects=list(through), ignore_links=list(ignore_links),
                                                   accept_other_branch=accept_other_branch), moved)
            except SkillError as e:
                if moved[0] or e.reason != "plan_failed":
                    raise
                log.warning("reach_pre_pose(%s): the direct plan to the pre-pose failed before the robot moved (%s) - planning the bare standoff", arm, e.message[:120])
                ctx["pre_pose_standoff"]["direct_failed"] = e.message[:160]
            else:
                if getattr(motions, "last_relaxed_links", None):
                    ctx["plan_relaxed_links"] = list(motions.last_relaxed_links)
                return
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
    retrace). None when the backend cannot track it (``reach_pre_pose`` then plans the pre-pose with the witness first,
    ``reach_via_standoff`` the standoff pose)."""
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
            log.warning("standoff configuration from the witness not trackable (%s)", res.status)
            return None
        return C.as_tensor(res.q_traj[-1]).reshape(-1).float()
    except Exception as e:  # noqa: BLE001
        log.warning("standoff configuration from the witness failed (%r)", e)
        return None


def _hook(ctx: Dict[str, Any], name: str):
    """An executor hook (``ctx["hooks"][name]``; a no-op without an executor): ``fov(enabled)`` enables / disables
    the gripper-in-FOV monitor around the checked segment of the action (``qa.Monitors`` arming rule)."""
    return (ctx.get("hooks") or {}).get(name) or (lambda *a, **k: None)


def _patch_ag_joint_type(robot, joint_type: Optional[str]) -> None:
    """Compatibility entry point of the lab evaluator (``lab_tools/evaluator_patches.py``, ``oracle/vec_loader.py``): the
    assisted-grasp joint of a link behind a movable joint is the task's rule (``tasks.ispatialgym_task``), not the oracle's.
    ``"FixedJoint"`` applies that rule to ``robot`` again; ``None`` leaves the robot as it is. OmniGibson's joint choice is
    not overridden here."""
    if joint_type is None:
        return
    if joint_type != "FixedJoint":
        raise ValueError(f"_patch_ag_joint_type: the task's rule is 'FixedJoint' (or None: no-op), got {joint_type!r}")
    from ispatialgym.tasks.ispatialgym_task import apply_ag_joint_rule

    apply_ag_joint_rule(robot)


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


#: rad: a kept branch's chain may come this close to a joint limit where the recorded chain came closer (the sampler's
#: ``IKChecker.JOINT_LIMIT_MARGIN``); clearances within ``orient.CLEARANCE_EQ`` of each other count as equal
KEEP_BRANCH_LIMIT_MARGIN = 0.05


def _arm_limit_margin(robot, q28, arm: str) -> float:
    """Smallest distance (rad) of ``arm``'s joints in ``q28`` to their limits."""
    idx = robot.arm_control_idx[arm]
    q = C.as_tensor(q28).reshape(-1).float()[idx]
    lo, hi = C.as_tensor(robot.joint_lower_limits).float()[idx], C.as_tensor(robot.joint_upper_limits).float()[idx]
    return float(th.min(th.minimum(q - lo, hi - q)))


def keep_arm_branch(motions: Motions, state, arm: str, q_rec, sweep: Sequence[th.Tensor], nxt: Optional[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Owner 2026-09-27 (first-demo v2.1 A2, block design 0927-040243: before the third grasp the base stood still while the transition
    swung the hand 41 cm out to the right and back, 382 steps): a travel leg brings an empty arm to the joints the sampler swept it
    with (``arms_q``, "the arm as the previous motion left it") - but the arm can hold that hand pose in another IK branch (grasp 2's
    via-above planned the pose goal in cuRobo's own branch and the Cartesian actions after it stayed there). The arm then stays where it
    is when (1) its hand is at the recorded one (``witness_residual_accept_pos`` or the retreat's own acceptance ``retreat_partial_accept_m``
    / ``witness_residual_accept_rot``), (2) the leg's configurations ``sweep`` with the arm as it is are collision-free and keep the
    protected objects' distance (``oracle.protection``), and (3) the arm's next contact ``nxt`` (``Executor.next_contact``) is reached
    from here: the IK of its first pose seeded at the arm on the contact's stance lands nearer than the recorded witness, and the
    contact's chain up to the arm's next travel leg tracks from there no closer to a joint limit than the recorded chain (or
    ``KEEP_BRANCH_LIMIT_MARGIN``). Returns the facts; ``keep`` True carries ``witnesses`` ({action index: {key: 28-D q}}): the chain's
    configurations from here, which replace the sampler's witnesses of those actions so the next contact goes straight to its pre-pose."""
    from ispatialgym.sampling.v2.orient import CLEARANCE_EQ

    backend, cfg = motions.backend, motions.cfg
    idx, eef = state.robot.arm_control_idx[arm], M.EEF_LINK[arm]
    q_now = C.as_tensor(backend.q_full()).reshape(-1).float().clone()
    q_r = q_now.clone()
    q_r[idx] = C.as_tensor(q_rec).reshape(-1).float()
    dp, da = _pose_error(backend.fk([eef], q_now)[eef], backend.fk([eef], q_r)[eef])
    f: Dict[str, Any] = {"keep": False, "joint_gap_rad": round(float(th.max(th.abs(q_now[idx] - q_r[idx]))), 3), "hand_gap_m": round(dp, 4), "hand_gap_rad": round(da, 4)}
    # the recorded joints are where the sampler's previous motion ended (``choose_departure_arms``); the executor accepts that motion's end
    # within ``retreat_partial_accept_m`` (0927-040243 leg 16: the recorded retreat stalled at right_arm_joint2's limit, the kept branch
    # retreated the full 10 cm - the hands 22 mm apart)
    if dp > max(cfg.witness_residual_accept_pos, _cfg(ctx).retreat_partial_accept_m) or da > cfg.witness_residual_accept_rot:
        return {**f, "why": "the hand is not at the recorded pose"}
    if not nxt or not nxt.get("chain"):
        return {**f, "why": (nxt or {}).get("why") or "no grasp of the arm follows the travel legs"}
    f["next"] = nxt["index"]
    qs = th.stack([C.as_tensor(q).reshape(-1).float() for q in sweep]) if sweep else q_now.unsqueeze(0)
    if bool(C.as_tensor(backend.check_collision(qs, sync_world=True)).any()):
        return {**f, "why": "the arm as it is collides during the leg"}
    objs = _protected_for(ctx, state, nxt.get("object"))
    if objs and len(qs) >= 3 and hasattr(motions, "_protect_violations"):
        from ispatialgym.oracle import protection as PROT

        viol = motions._protect_violations([arm], qs, "whole_body", {"objects": objs, "inflation": PROT.INFLATION_M})
        if viol:
            return {**f, "why": f"the arm as it is comes within the protection distance of {viol[0].get('object')} during the leg"}
    base, trunk = nxt["stance"]
    q0 = _q_at_base(motions, q_now, *base)
    q0[state.robot.trunk_control_idx] = C.as_tensor(trunk).reshape(-1).float()
    first = nxt["chain"][0]
    q_first = backend.solve_ik(f"{arm}_arm", {eef: as_world_pose(first["pose"])}, q_seed=q0, sync_world=False)
    if q_first is None:
        return {**f, "why": f"no IK of the next {nxt['name']}'s {first['key']} from the arm as it is"}
    q_first = C.as_tensor(q_first).reshape(-1).float()
    reach = float(th.max(th.abs(q_first[idx] - q0[idx])))
    reach_rec = float(th.max(th.abs(C.as_tensor(first["q"]).reshape(-1).float()[idx] - q0[idx])))
    f.update({"reach_rad": round(reach, 3), "recorded_reach_rad": round(reach_rec, 3)})
    if reach >= reach_rec:
        return {**f, "why": "the IK from here is no nearer than the recorded witness"}
    out: Dict[int, Dict[str, th.Tensor]] = {first["action"]: {first["key"]: q_first}}
    q, prev = q_first, first["pose"]
    for w in nxt["chain"][1:]:
        dense = motions._densify(as_world_pose(prev), [w["pose"]])
        res = backend.solve_ik_path(f"{arm}_arm", [{eef: p} for p in dense], q_start=q, pos_tol=cfg.diff_ik_pos_tol, rot_tol=cfg.diff_ik_rot_tol,
                                    max_iters=cfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=cfg.diff_ik_soft_pos_tol_free)
        if not res.success or res.q_traj is None or len(res.q_traj) == 0:
            return {**f, "why": f"the next chain does not track from here at {w['action']}:{w['key']} ({res.status})"[:200]}
        q = C.as_tensor(res.q_traj[-1]).reshape(-1).float()
        out.setdefault(w["action"], {})[w["key"]] = q
        prev = w["pose"]
    margin = min(_arm_limit_margin(state.robot, qq, arm) for d in out.values() for qq in d.values())
    floor = min(nxt["margin"], KEEP_BRANCH_LIMIT_MARGIN) if nxt.get("margin") is not None else KEEP_BRANCH_LIMIT_MARGIN
    f.update({"chain_margin_rad": round(margin, 4), "recorded_margin_rad": nxt.get("margin")})
    if margin < floor - CLEARANCE_EQ:
        return {**f, "why": f"the chain from here comes {margin:.4f} rad from a joint limit (recorded {nxt.get('margin')})"}
    return {**f, "keep": True, "chain": [f"{a}:{k}" for a, d in sorted(out.items()) for k in d], "witnesses": out}


def _base_legs(motions: Motions, q: th.Tensor, start, goal, step_m: float = 0.05, step_rad: float = 0.1) -> List[th.Tensor]:
    """Configurations along a straight base leg ``start`` -> ``goal`` ([x, y, yaw] world) with the joints of ``q`` otherwise."""
    (x0, y0, a0), (x1, y1, a1) = (float(v) for v in start), (float(v) for v in goal)
    da = C.wrap_angle(a1 - a0)
    n = max(1, int(math.ceil(max(math.hypot(x1 - x0, y1 - y0) / step_m, abs(da) / step_rad))))
    return [_q_at_base(motions, q, x0 + (x1 - x0) * s / n, y0 + (y1 - y0) * s / n, a0 + da * s / n) for s in range(1, n + 1)]


def transition_sweep(motions: Motions, state, targets) -> List[th.Tensor]:
    """A transition's configurations after the drive-back with the arms as they are: the trunk swept to its target, then the drive-in."""
    q = C.as_tensor(motions.backend.q_full()).reshape(-1).float().clone()
    tidx = state.robot.trunk_control_idx
    t, k = targets["trunk"]
    q_t = C.as_tensor(C.torso_q(float(t), float(k), C.TORSO_JOINT_LIMITS)).reshape(-1).float()
    out = [q]
    for s in range(1, 9):
        qq = q.clone()
        qq[tidx] = q[tidx] + (q_t - q[tidx]) * (s / 8)
        out.append(qq)
    return out + _base_legs(motions, out[-1], state.base_pose2d(), as_pose2d(targets["drive_in"]))


def path_sweep(motions: Motions, state, waypoints) -> List[th.Tensor]:
    """A recorded base path's configurations with the arms and the trunk as they are."""
    q = C.as_tensor(motions.backend.q_full()).reshape(-1).float().clone()
    out, prev = [q], state.base_pose2d()
    for w in waypoints:
        w = as_pose2d(w)
        out += _base_legs(motions, q, prev, w)
        prev = w
    return out


def _keep_branches(motions: Motions, state, ctx: Dict[str, Any], move: Dict[str, th.Tensor], sweep_fn: Callable[[], List[th.Tensor]]) -> Dict[str, th.Tensor]:
    """``keep_arm_branch`` for every arm a travel leg would re-pose to its recorded joints: the kept arms leave ``move``; the facts go to
    ``ctx["branch_kept"]`` and the kept arms' chain configurations to ``ctx["witness_overrides"]`` (the executor applies them to the next
    contact's actions once the leg passes)."""
    nxt = ctx.get("next_contact") or {}
    kept: Dict[str, Any] = {}
    sweep: Optional[List[th.Tensor]] = None
    for arm in sorted(move):
        try:
            if sweep is None:
                sweep = sweep_fn()
            facts = keep_arm_branch(motions, state, arm, move[arm], sweep, nxt.get(arm), ctx)
        except Exception as e:  # noqa: BLE001 - a backend without FK / IK / frames (the test stubs): the recorded joints
            facts = {"keep": False, "why": f"no branch check ({e!r})"[:160]}
        kept[arm] = {k: v for k, v in facts.items() if k != "witnesses"}
        if facts.get("keep"):
            ctx.setdefault("witness_overrides", {})[arm] = facts["witnesses"]
            log.info("travel leg: the %s arm holds the recorded hand pose %.1f mm off in another IK branch (%.2f rad): kept, the next %s runs from here",
                     arm, facts["hand_gap_m"] * 1000, facts["joint_gap_rad"], (nxt.get(arm) or {}).get("name"))
    ctx["branch_kept"] = kept
    return {arm: q for arm, q in move.items() if not kept[arm].get("keep")}


def _q_at_stance(motions: Motions, targets, q: th.Tensor) -> th.Tensor:
    """``q`` at a transition's arriving stance: the trunk at ``targets["trunk"]``, the base at ``targets["drive_in"]``."""
    out = q.clone()
    t, k = targets["trunk"]
    out[motions.robot.trunk_control_idx] = C.as_tensor(C.torso_q(float(t), float(k), C.TORSO_JOINT_LIMITS)).reshape(-1).float()
    return _q_at_base(motions, out, *as_pose2d(targets["drive_in"]))


def _transition_sweep_from(motions: Motions, state, targets, q: th.Tensor) -> List[th.Tensor]:
    """:func:`transition_sweep` from the configuration ``q`` (the arms as given) instead of the measured one."""
    tidx = motions.robot.trunk_control_idx
    t, k = targets["trunk"]
    q_t = C.as_tensor(C.torso_q(float(t), float(k), C.TORSO_JOINT_LIMITS)).reshape(-1).float()
    out = [q.clone()]
    for s in range(1, 9):
        qq = q.clone()
        qq[tidx] = q[tidx] + (q_t - q[tidx]) * (s / 8)
        out.append(qq)
    return out + _base_legs(motions, out[-1], state.base_pose2d(), as_pose2d(targets["drive_in"]))


def _joint_line(q_from: th.Tensor, q_to: th.Tensor, idx, step: float = 0.05) -> List[th.Tensor]:
    """Configurations along the straight joint-space line ``q_from`` -> ``q_to`` of the joints ``idx`` (both ends included)."""
    d = float(th.max(th.abs(q_to[idx] - q_from[idx]))) if len(idx) else 0.0
    n = max(1, int(math.ceil(d / step)))
    out = []
    for s in range(n + 1):
        q = q_from.clone()
        q[idx] = q_from[idx] + (q_to[idx] - q_from[idx]) * (s / n)
        out.append(q)
    return out


def contact_route_goal(motions: Motions, state, arm: str, contact: Dict[str, Any], q_ctx: th.Tensor, cfg: "ExecuteConfig") -> Tuple[th.Tensor, str]:
    """Where a free arm's transit to its brace / push pre-pose ends (``reach_pre_pose``'s rule) with the robot at ``q_ctx``: the standoff
    ``pre_pose_standoff_m`` off the supports in the branch of the contact's first witness when the pre-pose sits closer than
    ``pre_pose_standoff_skip_m`` to them and the standoff tracks from that witness, else that witness (a standoff that does not
    track: the transit is planned straight to the pre-pose with the witness). Differential IK only (deterministic); ``(28-D q, label)``."""
    w = contact["witnesses"]
    key = "pre" if "pre" in w else sorted(w)[0]
    idx = motions.robot.arm_control_idx[arm]
    q_w = q_ctx.clone()
    q_w[idx] = C.as_tensor(w[key]).reshape(-1).float()[idx]
    pre = (contact.get("targets") or {}).get("pre")
    supports = contact_ignores(state, contact["object"]) if pre is not None else []
    n = escape_direction(state, pre, supports) if supports else None
    if n is None:
        return q_w, f"the {contact['name']}'s {key} witness"
    clearance = support_clearance(state, pre, supports)
    if clearance is not None and clearance >= cfg.pre_pose_standoff_skip_m:
        return q_w, f"the {contact['name']}'s {key} witness ({clearance:.3f} m clear of its supports)"
    above = height_above_supports(pre, supports) if cfg.pre_pose_standoff_skip_above_m is not None else None
    if above is not None and above >= cfg.pre_pose_standoff_skip_above_m:
        return q_w, f"the {contact['name']}'s {key} witness ({above:.3f} m above its supports)"
    p, q = as_world_pose(pre)
    dense = motions._densify((p, q), [(p + n * cfg.pre_pose_standoff_m, q)])
    res = motions.backend.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: wp} for wp in dense], q_start=q_w, pos_tol=motions.cfg.diff_ik_pos_tol,
                                        rot_tol=motions.cfg.diff_ik_rot_tol, max_iters=motions.cfg.diff_ik_max_iters, check_collisions=False, sync_world=False,
                                        soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if not res.success or res.q_traj is None or len(res.q_traj) == 0:
        return q_w, f"the {contact['name']}'s {key} witness (its standoff does not track: planned straight to the pre-pose)"
    return C.as_tensor(res.q_traj[-1]).reshape(-1).float(), f"the {contact['name']}'s standoff in the {key} witness's branch"


def keep_contact_arms(motions: Motions, state, targets, ctx: Dict[str, Any], move: Dict[str, th.Tensor], record=None) -> Dict[str, th.Tensor]:
    """Owner 2026-09-30 (first-demo v3.0 F1, world vertical 0928-043205: the transition tucked both untucked arms - the wall stance
    records ``arm_posture: tucked`` - and the brace / push then started from tucked). A transition leaves a free arm as it is
    instead of putting it into the arriving stance's recorded posture when it is one of that stance's working arms, the stance
    ``record`` names that posture (``arm_posture``: tucked / ready / reset / untucked - a fixed default, not a configuration of the
    arm's own motion), the transition record has no ``arms_q`` of its own, the arm's next action at the arriving stance is a brace /
    push (``ctx["stance_contacts"]``, ``Executor.stance_contact``) and, the arms kept as they are and the others at their targets:
    (1) the transition's sweep (trunk, then drive-in) is collision-free (whole body) and keeps the protected objects' distance;
    (2) the arm's straight joint-space route from its joints to where its transit ends (:func:`contact_route_goal`) at the arriving
    stance is collision-free and keeps that distance - with an arm that acts before it standing at that arm's contact;
    (3) an arm that acts later stays collision-free as it is while the earlier arm stands at each of its contact's witnesses.
    An arm failing (2) or (3) is re-posed and the rest re-checked; (1) failing re-poses every arm (the recorded sequence).
    Collision checks, FK and differential IK only (no planner, no cuRobo random stream). ``ctx["contact_arms_kept"]`` records
    fired / not fired and why; returns ``move`` without the kept arms."""
    cfg = _cfg(ctx)
    contacts = ctx.get("stance_contacts") or {}
    rec: Dict[str, Any] = {"fired": False, "arms": {}}
    ctx["contact_arms_kept"] = rec
    posture = (getattr(record, "extra", None) or {}).get("arm_posture")
    working = set(getattr(record, "working_arms", []) or [])
    if not move:
        rec["why"] = "no arm to re-pose"
        return move
    if targets.get("arms_q"):
        rec["why"] = "the transition re-poses the arms to the joints its record was swept with, not to the stance's posture"
        return move
    if posture not in IDLE_POSTURES:
        rec["why"] = "the arriving stance names no arm posture for its working arms"
        return move
    cand = []
    for arm in sorted(move):
        c = contacts.get(arm)
        if arm not in working:
            rec["arms"][arm] = {"kept": False, "why": f"not a working arm of the arriving stance (its idle arm goes to {posture})"}
        elif not c:
            rec["arms"][arm] = {"kept": False, "why": "its next action at the arriving stance is not a brace / push"}
        elif not c.get("witnesses"):
            rec["arms"][arm] = {"kept": False, "why": f"no recorded witness of its {c['name']}"}
        else:
            cand.append(arm)
    if not cand:
        rec["why"] = f"no working arm put into the stance's {posture} posture has a brace / push next at the arriving stance"
        return move
    rec["posture"] = posture
    from ispatialgym.oracle import protection as PROT

    backend = motions.backend
    kept = sorted(cand, key=lambda a: contacts[a]["index"])
    order = {a: contacts[a]["index"] for a in kept}

    def protected(arms):
        objs: List[Tuple[str, Any]] = []
        for a in arms:
            for n_, o in _protected_for(ctx, state, contacts[a]["object"]):
                if not any(o is x for _, x in objs):
                    objs.append((n_, o))
        return objs

    def collides(qs) -> Optional[int]:
        hits = C.as_tensor(backend.check_collision(th.stack(qs), sync_world=True)).reshape(-1)
        return int(th.nonzero(hits)[0]) if bool(hits.any()) else None

    def too_close(arms, qs, objs) -> Optional[Dict[str, Any]]:
        if not objs or len(qs) < 3 or not hasattr(motions, "_protect_violations"):
            return None
        viol = motions._protect_violations(list(arms), th.stack(qs), "whole_body", {"objects": objs, "inflation": PROT.INFLATION_M})
        return viol[0] if viol else None

    try:
        for _ in range(len(cand)):
            q_base = C.as_tensor(backend.q_full()).reshape(-1).float().clone()
            for arm, qa in move.items():
                if arm not in kept:
                    q_base[motions.robot.arm_control_idx[arm]] = C.as_tensor(qa).reshape(-1).float()
            sweep = _transition_sweep_from(motions, state, targets, q_base)
            hit = collides(sweep)
            viol = None if hit is not None else too_close(kept, sweep, protected(kept))
            if hit is not None or viol is not None:
                why = (f"the sweep with the arms as they are collides (configuration {hit} of {len(sweep)})" if hit is not None else
                       f"the sweep with the arms as they are comes {viol.get('clearance_m')} m from {viol.get('object')}")
                rec["why"] = why + ": every arm re-posed"
                for arm in kept:
                    rec["arms"][arm] = {"kept": False, "why": why}
                return move
            q_st = _q_at_stance(motions, targets, q_base)
            failed: Dict[str, str] = {}
            for arm in kept:
                idx = motions.robot.arm_control_idx[arm]
                earlier = [b for b in kept if b != arm and order[b] < order[arm]]
                q_ctx = q_st.clone()
                for b in earlier:  # the earlier arm stands at its contact (its last witness) while this arm moves
                    wb = contacts[b]["witnesses"]
                    q_ctx[motions.robot.arm_control_idx[b]] = C.as_tensor(wb[list(wb)[-1]]).reshape(-1).float()[motions.robot.arm_control_idx[b]]
                goal, label = contact_route_goal(motions, state, arm, contacts[arm], q_ctx, cfg)
                route = _joint_line(q_ctx, goal, idx)
                hit = collides(route)
                viol = None if hit is not None else too_close([arm], route, protected([arm]))
                facts = {"route_to": label, "route_rad": round(float(th.max(th.abs(goal[idx] - q_ctx[idx]))), 3), "route_configurations": len(route)}
                if hit is not None or viol is not None:
                    failed[arm] = (f"its route to {label} collides (configuration {hit} of {len(route)})" if hit is not None else
                                   f"its route to {label} comes {viol.get('clearance_m')} m from {viol.get('object')}")
                    rec["arms"][arm] = {"kept": False, "why": failed[arm], **facts}
                    continue
                holds = []
                for b in earlier:  # this arm, as it is, while the earlier arm goes through its contact
                    ib = motions.robot.arm_control_idx[b]
                    for q_b in contacts[b]["witnesses"].values():
                        q_h = q_st.clone()
                        q_h[ib] = C.as_tensor(q_b).reshape(-1).float()[ib]
                        holds.append(q_h)
                if holds and collides(holds) is not None:
                    failed[arm] = f"as it is it collides with the {earlier[0]} arm's {contacts[earlier[0]]['name']}"
                    rec["arms"][arm] = {"kept": False, "why": failed[arm], **facts}
                    continue
                rec["arms"][arm] = {"kept": True, "contact": f"{contacts[arm]['name']}[{contacts[arm]['index']}]", **facts,
                                    **({"held_beside": f"{earlier[0]} {contacts[earlier[0]]['name']}", "hold_configurations": len(holds)} if holds else {})}
            if not failed:
                break
            kept = [a for a in kept if a not in failed]
            if not kept:
                break
    except Exception as e:  # noqa: BLE001 - a backend without the checks (the test stubs): the recorded sequence
        rec["why"] = f"no collision check ({e!r}): every arm re-posed"[:200]
        for arm in cand:
            rec["arms"][arm] = {"kept": False, "why": rec["why"]}
        return move
    rec["fired"] = bool(kept)
    rec["why"] = (f"{', '.join(kept)} kept as they are: the sweep, the route to the contact and the later arm's hold are clear" if kept else
                  "every candidate arm failed its route / hold check: re-posed")
    if kept:
        log.info("transition: %s kept as they are (next: %s); the sweep, the routes to the contacts and the holds are collision-free", "/".join(kept),
                 ", ".join(f"{a} {contacts[a]['name']}" for a in kept))
    return {arm: q for arm, q in move.items() if arm not in kept}


def move_idle_arms(motions: Motions, state, record, ctx: Dict[str, Any], arms_q: Optional[Dict[str, Sequence[float]]] = None,
                   keep_sweep: Optional[Callable[[], List[th.Tensor]]] = None,
                   keep_contacts: Optional[Callable[[Dict[str, th.Tensor]], Dict[str, th.Tensor]]] = None) -> Iterator[th.Tensor]:
    """Put the arms into the stance's recorded drive-in posture before the trunk motion / drive-in (A6 + owner
    decision 2026-09-16 10:47): the recorded finger opening (``fingers``: closed pads for the wall two-arm stance) and
    the recorded joints (``arms_q`` / ``arm_posture``) on **every arm that holds nothing** — working arms included —
    so the swept trunk motion and the drive-in happen in the posture the sampler checked (the wall stance at
    d = 0.55 m was rejected untucked and recorded tucked). A carried object keeps its grasp and its arm. Arms already
    there are left alone. Notes: ``arms_posture`` = {moved, kept (holding), source}, ``fingers_set``.

    ``arms_q`` (owner 2026-09-24): the arm joints the *transition record* itself was swept with (``transitions[].arms_q``
    — the arm left where the previous motion ended it, ``program_check.choose_departure_arms``); they replace the stance's
    arm targets, so an arm already there does not move (the fingers still follow the stance record). ``keep_sweep`` (the leg's
    configurations with the arms as they are): an arm that holds the recorded hand pose in another IK branch may stay
    (``keep_arm_branch``, first-demo v2.1 A2). ``keep_contacts`` (a transition, first-demo v3.0 F1): an arm whose next action at
    the arriving stance is a brace / push may stay as it is (:func:`keep_contact_arms`)."""
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
    if arms_q:
        targets = {arm: C.as_tensor(list(arms_q[arm])).reshape(-1).float() for arm in M.ARMS if arms_q.get(arm) is not None}
        source = "record arms_q (the arm as the previous motion left it)"
    else:
        targets = stance_arm_targets(record, state, include_working=True)
        extra = getattr(record, "extra", None) or {}
        source = "arms_q" if extra.get("arms_q") else (extra.get("arm_posture") or extra.get("idle_posture") or None)
    q = state.robot.get_joint_positions()
    move = {arm: qa for arm, qa in targets.items() if arm not in holding and float(th.max(th.abs(q[state.robot.arm_control_idx[arm]].float() - qa))) > IDLE_ARM_TOL}
    if arms_q and keep_sweep is not None and move:
        move = _keep_branches(motions, state, ctx, move, keep_sweep)
    if keep_contacts is not None:
        move = keep_contacts(move)
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
    arms_q = targets.get("arms_q")
    if arms_q and not targets.get("carried"):
        # owner 2026-09-24: the record was swept with the arm where the previous motion left it (``paths[].arms_q``): an arm
        # that is there stays; one that deviates (a retreat that ended short in another branch) is brought to the swept joints
        q_now = state.robot.get_joint_positions()
        move = {}
        for arm in M.ARMS:
            if arms_q.get(arm) is None or state.held_object(arm) is not None:
                continue
            idx = state.robot.arm_control_idx[arm]
            qa = C.as_tensor(list(arms_q[arm])).reshape(-1).float()
            if float(th.max(th.abs(q_now[idx].float() - qa))) > IDLE_ARM_TOL:
                move[arm] = qa
        if move and ctx.get("next_contact"):
            # first-demo v2.1 A2: an arm holding the recorded hand pose in another IK branch drives as it is when the path clears it
            move = _keep_branches(motions, state, ctx, move, lambda: path_sweep(motions, state, targets["waypoints"]))
        ctx["navigate_arms_reposed"] = sorted(move)
        ctx["navigate_arms_source"] = "record arms_q"
        if move:
            yield from motions.arm_to(tuple(sorted(move)), q=move)
    elif posture == "untucked" and not targets.get("carried") and all(state.held_object(arm) is None for arm in M.ARMS):
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
    if targets.get("axis_leg"):
        # owner 2026-10-13 (disembedding): the leg into the place pose keeps the heading and moves forward / backward or left / right
        # one after the other - a stop at every corner, the driven poses kept for post_navigate's check
        poses = ctx["axis_leg_poses"] = [tuple(float(v) for v in state.base_pose2d())]
        for a in motions.track_path(targets["waypoints"], posture=posture, style=BP.RECORDED, corner_stop=True):
            yield a
            poses.append(tuple(float(v) for v in state.base_pose2d()))
        return
    yield from motions.track_path(targets["waypoints"], posture=posture, style=targets.get("style"))


def post_navigate(action: Navigate, state, targets, ctx) -> Optional[str]:
    cfg = _cfg(ctx)
    d, dyaw = _base_error(state.base_pose2d(), targets["goal"])
    ctx["base_error"] = (d, dyaw)
    if targets.get("axis_leg"):
        tr = BP.axis_leg_tracking(ctx.pop("axis_leg_poses", None) or [], targets["waypoints"])
        ok = tr["max_yaw_dev_rad"] <= cfg.axis_leg_yaw_tol and tr["max_corridor_m"] <= cfg.axis_leg_corridor_m
        ctx["axis_leg"] = {"max_yaw_dev_deg": round(math.degrees(tr["max_yaw_dev_rad"]), 3), "max_corridor_mm": round(1000.0 * tr["max_corridor_m"], 1),
                           "n_poses": tr["n_poses"], "moves": [[a, round(m, 3)] for a, m in BP.axis_leg_moves(targets["waypoints"])], "ok": ok}
        if not ok:
            return "execution"
    return None if d <= cfg.nav_pos_tol and dyaw <= cfg.nav_yaw_tol else "execution"


def execute_transition(action: Transition, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    yield from motions.track_path([targets["drive_back"]])
    # A6: the arriving stance's idle arm(s) to their recorded posture before the trunk moves (at the stand-off) - or, when the
    # record was swept with the arm where the previous motion left it (``arms_q``, owner 2026-09-24), to exactly that; an arm
    # holding that hand pose in another IK branch may stay (first-demo v2.1 A2)
    keep_sweep = (lambda: transition_sweep(motions, state, targets)) if ctx.get("next_contact") else None
    # first-demo v3.0 F1: an arm whose next action at the arriving stance is a brace / push may go there from where it is
    keep_contacts = (lambda move: keep_contact_arms(motions, state, targets, ctx, move, record=action.stance_to)) if "stance_contacts" in ctx else None
    yield from move_idle_arms(motions, state, action.stance_to, ctx, arms_q=targets.get("arms_q"), keep_sweep=keep_sweep, keep_contacts=keep_contacts)
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


def grasp_closure(action: Grasp, width: Optional[float], cfg) -> Tuple[Optional[float], Optional[float], bool]:
    """``(rate, squeeze, thin)`` of a grasp's closure (:meth:`Motions.gripper`): a library pinch at most ``cfg.thin_pinch_width_m`` wide
    closes at ``thin_pinch_close_rate`` and holds at the touch width less ``thin_pinch_squeeze_m``; every other grasp at
    ``gripper_close_rate`` with the configured squeeze (``squeeze`` None)."""
    thin_w = getattr(cfg, "thin_pinch_width_m", None)
    if action.kind == "library" and width is not None and thin_w is not None and float(width) <= float(thin_w):
        return float(cfg.thin_pinch_close_rate), float(cfg.thin_pinch_squeeze_m), True
    return cfg.gripper_close_rate, None, False


def _grasp_transit(action: Grasp, motions: Motions, state, targets, ctx, arm: str, supports: Sequence[Any], ignore_links: Sequence[Any]) -> Iterator[th.Tensor]:
    """The free move to the pre-grasp (``execute_grasp``): through a recorded standoff, a lever grasp's Cartesian detour, or
    the supports-present plan of ``reach_pre_pose`` (the supports-ignored plan as its fallback)."""
    standoff_m = _grasp_extra_float(action, "approach_standoff_m")
    if standoff_m:
        yield from reach_via_standoff(motions, arm, targets["pre_grasp"], standoff_m, _witness(ctx, "pre_grasp"), supports, ctx)
    elif ignore_links:
        # the other hand presses the object (the lever): the pinching hand comes from its brace beside the box. Owner 2026-09-22 11:27
        # (0921-232806 / 0922-113512: the joint-space plan from the blade pose swung the hand 70 cm to the right and out of the frame):
        # first a tracked Cartesian detour - straight up, across at that height, down onto the pre-grasp - the planner only when
        # the tracking has no solution
        done = False
        branch_short = False
        if _cfg(ctx).grasp_transit_cartesian:
            try:
                here = as_world_pose(state.eef_pose(arm))
                pre_p, pre_q = as_world_pose(targets["pre_grasp"])
                lift = float(_cfg(ctx).grasp_transit_lift_m)
                z_top = max(float(here[0][2]), float(pre_p[2])) + lift
                up = (th.tensor([float(here[0][0]), float(here[0][1]), z_top]), here[1])
                over = (th.tensor([float(pre_p[0]), float(pre_p[1]), z_top]), pre_q)
                legs = [here, up, over, (pre_p, pre_q)]
                # the detour keeps the brace's IK branch; the sampler checked the approach in the pre-grasp witness's. The detour is
                # taken only when that branch also reaches the grasp (0927-023811: the brace branch stood at right_arm_joint2's limit
                # and stalled 11 mm short of a pinch 2 cm further along the flap edge; the witness branch was never used)
                if hasattr(motions, "along_reaches") and targets.get("grasp") is not None and not motions.along_reaches(arm, legs[1:] + [as_world_pose(targets["grasp"])]):
                    branch_short = True
                    raise SkillError("plan_failed", "the detour's IK branch does not reach the grasp", stage="diff_ik")
                yield from motions.arm_along(arm, legs)
                ctx["pre_grasp_transit"] = f"cartesian detour {lift * 100:.0f} cm above"
                done = True
            except SkillError as e:
                log.info("grasp(%s): the Cartesian detour to the pre-grasp has no solution (%s) - planning it", arm, e.message[:80])
        if not done and branch_short:
            # cuRobo's own branch is the one nearest the hand - the branch that does not reach the grasp: the witness branch directly
            yield from motions.arm_to(arm, pose=targets["pre_grasp"], q_hint=_witness(ctx, "pre_grasp"), ignore_objects=supports, ignore_links=ignore_links)
            ctx["pre_grasp_transit"] = "witness branch (the detour's branch does not reach the grasp)"
        elif not done:
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
        # The former plan (supports ignored throughout) is the fallback when no plan exists with them present.
        # first-demo v3.0 F5 (nextto marker 0928-050212): a planned transit that dips below both of its ends runs as a Cartesian leg
        # when one qualifies (``Motions.transit_dip_gate``); ``ctx["transit_dip"]`` records the last plan's verdict
        gate = motions.transit_dip_gate(arm, targets.get("approach") or ()) if hasattr(motions, "transit_dip_gate") else nullcontext(None)
        try:
            with gate as dip:
                try:
                    yield from reach_pre_pose(motions, state, arm, targets["pre_grasp"], supports, [], _witness(ctx, "pre_grasp"), ctx, _cfg(ctx))
                finally:
                    if dip is not None:
                        ctx["transit_dip"] = dict(dip)
            ctx["pre_grasp_transit"] = "supports present"
        except SkillError as e:
            if e.details.get("stage") == "protected":
                raise
            log.warning("grasp(%s): no transit to the pre-grasp with the supports present (%s); planning with them ignored", arm, e.message)
            ctx["pre_grasp_transit"] = f"supports ignored (fallback: {e.message[:80]})"
            yield from motions.arm_to(arm, pose=targets["pre_grasp"], q_hint=_witness(ctx, "pre_grasp"), ignore_objects=supports)


def _protected_for(ctx, state, target: Optional[str] = None) -> List[Tuple[str, Any]]:
    """The objects protected during this action's transit (oracle.protection): the task's goal objects (``ctx["protect"]
    ["goal"]``, set by run_isg) plus the action's own ``target`` when the task placed it - protected until its intended
    contact begins - minus whatever is in hand (a carried object is part of the robot)."""
    P = ctx.get("protect") or {}
    objs: List[Tuple[str, Any]] = list(P.get("goal") or [])
    if target:
        from ispatialgym.oracle import protection as PROT

        try:
            o = state.object(target)
        except Exception:  # noqa: BLE001
            o = None
        if o is not None and PROT.is_task_object(o) and not any(o is x for _, x in objs):
            objs.append((target, o))
    try:
        held = list(state.held_objects())
    except Exception:  # noqa: BLE001
        held = []
    return [(n, o) for n, o in objs if not any(o is h for h in held)]


@contextmanager
def _transit(ctx, motions: Motions, state, action, leg: str, target: Optional[str] = None, along: Optional[str] = None):
    """A transit of ``action`` (``leg``: pre_grasp / carry / pre_contact): the plan-time protect context of ``motions`` and the
    QA transit span over the protected objects (``_protected_for``); no protected object - a no-op, the motion unchanged."""
    objs = _protected_for(ctx, state, target)
    P = ctx.get("protect") or {}
    label = {"task": P.get("task"), "action": getattr(action, "name", ""), "index": P.get("index"), "leg": leg}
    if not objs or not hasattr(motions, "protecting"):
        yield
        return
    hold = getattr(motions, "press_hold", None) or {}
    if hold.get("arm") and hold.get("object") and hold.get("arm") != getattr(action, "arm", None):
        # the other hand holds a press on a protected object (the two-flap lever's fist under the flap the transit's hand goes to
        # pinch): its finger / gripper links touch that object by design, a contact that comes and goes as the flap settles is not
        # the transit's (0927-022029: left_gripper_finger_link2 on the propped flap two steps into the pinch's pre-grasp transit)
        try:
            label["hold_exempt"] = {"links": list(M.FINGER_LINKS[hold["arm"]]) + [M.GRIPPER_LINK[hold["arm"]]], "object": state.object(hold["object"]).name}
        except Exception:  # noqa: BLE001
            pass
    ctx.setdefault("protection", []).append({"leg": leg, "protected": [getattr(o, "name", n) for n, o in objs], "forced_via_above": bool(P.get("force")),
                                             **({"hold_exempt": label["hold_exempt"]} if label.get("hold_exempt") else {})})
    hook = _hook(ctx, "transit")
    try:
        hook(True, objs, label)
    except TypeError:
        pass
    try:
        with motions.protecting(objs, leg=leg, task=label["task"] or "", action=label["action"], index=label["index"], force_via_above=bool(P.get("force")), along=along):
            yield
    finally:
        try:
            hook(False)
        except TypeError:
            pass


def _finger_low_z(pos: th.Tensor, quat: th.Tensor, arm: str) -> Optional[float]:
    """Lowest world z of the finger meshes at the eef pose (pos, quat) over the closure (closed and open), None without the assets."""
    R = C.quat_to_mat(quat.float()).numpy()
    p = pos.float().numpy()
    lows = []
    for q in (C.FINGER_CLOSED, C.FINGER_OPEN):
        pts = M.finger_points_eef(q, arm)
        if pts is None:
            return None
        lows.append(float((pts @ R.T + p)[:, 2].min()))
    return min(lows)


def _raised(pose, dz: float) -> Pose:
    p, q = as_world_pose(pose)
    return Pose(position=[float(p[0]), float(p[1]), float(p[2]) + float(dz)], orientation=[float(v) for v in q.tolist()])


def _ramped(start, waypoints: Sequence[Any], end, dz: float) -> List[Pose]:
    """``waypoints`` between ``start`` (kept) and ``end`` (raised by ``dz``) raised by ``dz`` x their path fraction from ``start``."""
    pts = [as_world_pose(p)[0] for p in [start] + list(waypoints) + [end]]
    cum = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        cum.append(cum[-1] + float(th.linalg.norm(b.float() - a.float())))
    total = cum[-1]
    return [_raised(p, dz * (cum[i + 1] / total if total > 1e-9 else 1.0)) for i, p in enumerate(waypoints)]


def _pinch_face_slope(obj, centre, closing, reach: float = 0.15) -> Optional[float]:
    """The larger |normal z| of the two faces of ``obj`` a pinch centred at ``centre`` closes on along ``closing`` (a raycast toward
    ``centre`` from each side, the first hit on the object); None when a side misses it."""
    root = str(getattr(obj, "prim_path", "") or "")
    c = C.as_tensor(centre).reshape(-1).float()
    d = C.as_tensor(closing).reshape(-1).float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    if not root:
        return None
    slopes = []
    for ray in _raycast_all([(c + reach * d).tolist(), (c - reach * d).tolist()], [c.tolist(), c.tolist()]):
        hits = [h for h in (ray if isinstance(ray, list) else [ray]) if h.get("hit") and str(h.get("rigidBody") or "").startswith(root)]
        if not hits:
            return None
        n = C.as_tensor(min(hits, key=lambda h: float(h.get("distance", reach)))["normal"]).reshape(-1).float()
        slopes.append(abs(float(n[2])) / max(float(th.linalg.norm(n)), 1e-9))
    return max(slopes)


def _approach_tracks(motions: Optional[Motions], arm: str, q_hint, poses: Sequence[Any]) -> Optional[bool]:
    """Whether the differential IK tracks the straight legs through ``poses`` (the pre-grasp, its approach, the lifted grasp)
    from ``q_hint``'s arm joints - the pre-grasp witness the transit ends in and the approach leaves from (the current configuration
    without a hint), as ``Motions.arm_along`` will; None without a backend (the test stubs)."""
    backend = getattr(motions, "backend", None)
    solve = getattr(backend, "solve_ik_path", None)
    robot = getattr(motions, "robot", None)
    if solve is None or robot is None or not poses:
        return None
    cfg = motions.cfg
    try:
        q0 = C.as_tensor(backend.q_full()).reshape(-1).float().clone()
        idx = robot.arm_control_idx[arm]
        if q_hint is not None:
            q0[idx] = C.as_tensor(q_hint).reshape(-1).float()[idx]
        start = backend.fk([M.EEF_LINK[arm]], q0)[M.EEF_LINK[arm]]
        dense = motions._densify(as_world_pose(start), [as_world_pose(p) for p in poses])
        res = solve(f"{arm}_arm", [{M.EEF_LINK[arm]: p} for p in dense], q_start=q0, pos_tol=cfg.diff_ik_pos_tol, rot_tol=cfg.diff_ik_rot_tol,
                    max_iters=cfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=cfg.diff_ik_soft_pos_tol_free)
    except Exception as e:  # noqa: BLE001
        log.debug("approach probe of %s failed (%r)", arm, e)
        return None
    return bool(res.success)


def lift_grasp_off_support(action: Grasp, motions: Optional[Motions], state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: "ExecuteConfig") -> Dict[str, Any]:
    """Owner 2026-09-27 (first-demo v2.1 A3, bar soap 0927-034001: "the wrist camera contacts the table during grasp & release and
    pushes against it, causing robot to wobble" - a replay's contact dump showed only the two finger links, pressing 91-154 N into the
    tray floor while the hand stood still, and torso_joint1 bent 7.6 mrad; the camera never touched): a library pinch of an object
    resting on a support must not put a finger below the object's lowest point - nothing of the hand fits there. When the finger
    envelope at the grasp pose reaches below the object's bottom + ``grasp_support_clearance_m``, the grasp is lifted by the deficit
    along world z and the approach ramps up to it from the recorded pre-grasp, provided the pads' fingertip-side end stays
    ``grasp_support_pad_margin_m`` below the object's top and that approach tracks from the pre-grasp witness (``_approach_tracks``) -
    else the recorded grasp runs; a lifted grasp whose attempt failed is not lifted on the retry (``Executor.run_action`` keeps
    ``support_lift_refused``). The pre-grasp stays where it was recorded, and with it the transit there: the sandal 0927-030120 with
    its pre-grasp lifted too took the transit's via-above into a posture its approach stalled from (13 mm short 4.7 cm before the grasp,
    the finish toward the grasp witness 3.06 rad away refused - 8b572a12 on a pod and locally, both attempts). A vertical approach (the
    bar soap's) is the same line either way. Only an object set down or attached next is lifted (``ctx["next_place_kind"]``): those
    places follow the in-hand pose (``raise_place_for_inhand_pose``, the attach re-aim); a drop does not - the sandal lifted 14 mm
    landed 9.6 deg / 7 mm off its recorded drop and the push under the table then turned it 78 deg (503be065, local 0927-163758;
    v2.0: 2 deg). ``ctx["grasp_support_lift"]`` records the facts."""
    if action.kind != "library" or targets.get("grasp") is None:
        return targets
    g_extra = getattr(getattr(action, "grasp", None), "extra", None)
    if isinstance(g_extra, dict) and g_extra.get("pick_overhang"):
        # the overhang pick (owner 2026-10-13 ~03:15 item 4): the object was pushed past its support's edge so that the finger below it
        # stands in free space - lifting the grasp would only take the pads off the rim
        ctx["grasp_support_lift"] = {"applied": False, "reason": "an overhang grasp: the finger below the object stands beyond the support's edge"}
        return targets
    try:
        lo, hi = state.object(action.object).aabb
        bottom, top = float(lo[2]), float(hi[2])
        pos, quat = (C.as_tensor(x).float() for x in as_world_pose(targets["grasp"]))
        low = _finger_low_z(pos, quat, action.arm)
    except Exception as e:  # noqa: BLE001 - no aabb / no finger meshes: the recorded grasp
        ctx["grasp_support_lift"] = {"applied": False, "reason": f"{e!r}"[:120]}
        return targets
    if low is None:
        return targets
    need = bottom + cfg.grasp_support_clearance_m - low
    pad_end = float((pos + C.PAD_Z_RANGE[1] * C.approach_axis(quat))[2])
    rec: Dict[str, Any] = {"applied": False, "finger_low_z": round(low, 4), "object_bottom_z": round(bottom, 4), "deficit_m": round(need, 4), "object_top_z": round(top, 4)}
    ctx["grasp_support_lift"] = rec
    if need <= 0.0:
        return targets
    if pad_end + need > top - cfg.grasp_support_pad_margin_m:
        rec["reason"] = f"a {need * 1000:.1f} mm lift leaves the pads {(top - pad_end - need) * 1000:.1f} mm below the object's top"
        return targets
    nxt = ctx.get("next_place_kind")
    if nxt not in ("support", "attach"):
        rec["reason"] = (f"the object's next place ({nxt}) does not follow its in-hand pose (a drop lands as the release throws it): the "
                         f"recorded grasp runs")
        return targets
    try:
        # the lifted pads' fingertip-side end: what grips an object the pads overhang (the milk: the lifted pads reached down only to
        # 12 mm above the bottle's widest section; v2.0's pinch had them there)
        centre = pos + th.tensor([0.0, 0.0, need]) + (C.PAD_Z_RANGE[1] - 0.001) * C.approach_axis(quat)
        slope = _pinch_face_slope(state.object(action.object), centre, C.closing_axis(quat))
    except Exception as e:  # noqa: BLE001 - no scene queries (the test stubs): the faces are not judged
        log.debug("pinch face probe of %s failed (%r)", action.object, e)
        slope = None
    if slope is not None:
        rec["face_slope"] = round(slope, 3)
        if slope > cfg.grasp_support_face_slope_max:
            rec["reason"] = (f"the lifted pads would close on faces sloped {math.degrees(math.asin(min(slope, 1.0))):.0f} deg from vertical (a curved "
                             f"side above its widest section rolls under the closure): the recorded grasp runs")
            return targets
    index = (ctx.get("protect") or {}).get("index")
    if index is not None and index in getattr(motions, "support_lift_refused", ()):
        rec["reason"] = "the lifted grasp failed the previous attempt of this action: the recorded grasp runs"
        return targets
    out = dict(targets)
    out["grasp"] = _raised(targets["grasp"], need)
    out["approach"] = _ramped(targets["pre_grasp"], list(targets["approach"]), targets["grasp"], need)
    if _approach_tracks(motions, action.arm, _witness(ctx, "pre_grasp"), [out["pre_grasp"]] + list(out["approach"]) + [out["grasp"]]) is False:
        rec["reason"] = f"the approach to the {need * 1000:.1f} mm lifted grasp does not track from the pre-grasp witness: the recorded grasp runs"
        return targets
    rec.update({"applied": True, "lift_m": round(need, 4)})
    log.info("grasp(%s): the finger envelope reaches %.1f mm below the object's bottom + %.0f mm: the grasp is lifted by that", action.object,
             need * 1000, cfg.grasp_support_clearance_m * 1000)
    return out


def raise_place_for_inhand_pose(action: Place, state, sidecar, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: "ExecuteConfig") -> Dict[str, Any]:
    """Owner 2026-09-27 (first-demo v2.1 A3): the recorded place pose sets the object down only with the recorded in-hand transform
    (``grasps[].eef_in_object``); an object that hangs LOWER in the hand (a grasp that ended higher on it - stopped short on the
    support, 1 cm on the bar soap's tray floor, or lifted by ``lift_grasp_off_support``) was driven into its support by that much, the
    fingertips pressing ~100 N into the destination tray through the release. The pre-place, place, descent and retreat are raised by
    the difference (z only, never lowered) and the carry ramps up to the raised pre-place. ``ctx["place_inhand_raise"]`` records it."""
    if action.kind != "support" or targets.get("place") is None:
        return targets
    try:
        obj = state.object(action.object)
        if state.held_object(action.arm) is not obj:
            return targets
        g = next((g for g in (getattr(sidecar, "grasps", None) or []) if g.object == action.object and getattr(g, "arm", None) in (None, action.arm)), None)
        eio = (getattr(g, "extra", None) or {}).get("eef_in_object") if g is not None else None
        if not isinstance(eio, dict) or eio.get("position") is None or eio.get("orientation") is None:
            return targets
        eio_rec = (th.tensor([float(v) for v in eio["position"]]), th.tensor([float(v) for v in eio["orientation"]]))
        eio_live = C.relative(tuple(C.as_tensor(x).float() for x in state.eef_pose(action.arm)), tuple(C.as_tensor(x).float() for x in state.object_pose(action.object)))
        place = tuple(C.as_tensor(x).float() for x in as_world_pose(targets["place"]))
        live = C.compose(C.compose(place, C.invert(eio_rec)), eio_live)  # the hand pose that sets the object where the recorded place does
        dz = float(live[0][2] - place[0][2])
    except Exception as e:  # noqa: BLE001
        ctx["place_inhand_raise"] = {"applied": False, "reason": f"{e!r}"[:120]}
        return targets
    rec: Dict[str, Any] = {"applied": False, "dz_m": round(dz, 4)}
    ctx["place_inhand_raise"] = rec
    if dz <= cfg.place_inhand_raise_min_m:
        return targets
    out = dict(targets)
    for k in ("pre_place", "place"):
        out[k] = _raised(targets[k], dz)
    out["descent"] = [_raised(p, dz) for p in targets["descent"]]
    out["retreat"] = [_raised(p, dz) for p in targets["retreat"]]
    carry = list(targets.get("carry") or ())
    if carry:
        out["carry"] = [_raised(p, dz * (i + 1) / len(carry)) for i, p in enumerate(carry)]
    rec.update({"applied": True, "raised_m": round(dz, 4)})
    log.info("place(%s): the object hangs %.1f mm lower in the hand than recorded: the place is raised by that", action.object, dz * 1000)
    return out


def _into_witness_posture(motions: Motions, targets, ctx, arm: str) -> Iterator[th.Tensor]:
    """The transit to the pre-grasp may leave the arm in another IK branch than the pre-grasp witness: the via-above's pose-goal point
    above descends in its own branch (sandal on pods, 0927-154128 / 0927-165511: at the pre-grasp the arm stood 2.91 rad from the
    witness, shoulder and elbow flipped - v2.0 0927-030120 and the local replays 0.91 rad, the same branch - and the approach stalled
    56 mm short of the grasp in both attempts). When the approach does not track from where the arm is but does from the pre-grasp
    witness, where the sampler checked it, the arm first moves into the witness's posture: a joint-space plan to the witness refined
    for the pre-grasp pose, checked against the protected objects of the enclosing transit; with no such plan nothing moves.
    ``ctx["witness_posture"]`` records it."""
    w = _witness(ctx, "pre_grasp")
    backend = getattr(motions, "backend", None)
    if w is None or backend is None or not hasattr(motions, "along_reaches"):
        return
    try:
        if motions.along_reaches(arm, targets["approach"]):
            return
        if _approach_tracks(motions, arm, w, [targets["pre_grasp"]] + list(targets["approach"])) is not True:
            return
        emb = f"{arm}_arm"
        idx = motions.robot.arm_control_idx[arm]
        q_now = C.as_tensor(backend.q_full()).reshape(-1).float()
        q_w = q_now.clone()
        q_w[idx] = C.as_tensor(w).reshape(-1).float()[idx]
        q_ik = backend.solve_ik(emb, {M.EEF_LINK[arm]: as_world_pose(targets["pre_grasp"])}, q_seed=q_w, sync_world=False)
        if q_ik is not None:
            q_ik = C.as_tensor(q_ik).reshape(-1).float()
            if float(th.max(th.abs(q_ik[idx] - q_w[idx]))) <= motions.cfg.witness_refine_max_jump:
                q_w[idx] = q_ik[idx]
        rec: Dict[str, Any] = {"gap_rad": round(float(th.max(th.abs(q_w[idx] - q_now[idx]))), 3)}
        ctx["witness_posture"] = rec
        backend.update_world()  # the supports the transit left out of the world are obstacles again
        res = backend.plan_joint(emb, q_w, q_start=q_now, max_attempts=motions.cfg.plan_attempts, sync_world=False)
    except Exception as e:  # noqa: BLE001 - the approach from where the arm is, as before
        log.debug("witness posture check of %s failed (%r)", arm, e)
        return
    if not getattr(res, "success", False) or res.q_traj is None:
        rec["result"] = f"no plan ({getattr(res, 'status', '?')})"
        return
    viol = motions._protect_violations([arm], res.q_traj, emb)
    if viol:
        rec["result"] = f"the plan passes {viol[0]['object']} at {viol[0]['clearance_m']} m"
        return
    log.info("grasp(%s): the approach does not track from where the transit left the arm (%.2f rad from the pre-grasp witness) but "
             "does from the witness: the arm moves into its posture first", arm, rec["gap_rad"])
    yield from motions.execute_joint_trajectory(res.q_traj, arms=[arm])
    rec["result"] = "moved into the witness posture"


def repose_arm(action: Grasp, motions: Motions, ctx) -> Iterator[th.Tensor]:
    """``Grasp.repose`` (block design after a between-block base move, owner 2026-10-12 item 2): the working arm's planned joint-space
    move (cuRobo, the world present: the blocks, the board, the table) from where the travel left it to the recorded pre-grasp
    configuration - the configuration the program check tracked the block's legs from. A plan that fails before the robot moved is
    planned again (``MotionsConfig.witness_plan_retries`` rounds, as a joint-space plan to a witness). ``ctx["repose"]`` records it."""
    rp = action.repose
    arm = action.arm
    q7 = C.as_tensor(list(rp["q"])).reshape(-1).float()
    rounds = 1 + int(getattr(getattr(motions, "cfg", None), "witness_plan_retries", 0))
    for k in range(rounds):
        moved = [0]
        try:
            yield from _counted(motions.arm_to(arm, q={arm: q7}), moved)
        except SkillError as e:
            if moved[0] or e.reason != "plan_failed" or k == rounds - 1:
                raise
            log.info("grasp(%s): the re-pose plan failed (%s) - planning it again", arm, e.message[:120])
            continue
        ctx["repose"] = {"target": rp.get("target"), "from_stance": rp.get("from_stance"), "rounds": k + 1}
        return


def execute_grasp(action: Grasp, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    arm = action.arm
    targets = lift_grasp_off_support(action, motions, state, targets, ctx, _cfg(ctx))
    # owner 2026-10-12 (A2b): the parts / distractors the grasp may touch leave the world for the approach finish (the transit plans
    # around them as before) and count as supports the fingers may stand on; how far the target was pushed before the pads close
    touch = list(may_touch_objects(state, action).values())
    obj_pos0 = state.object_pose(action.object)[0] if touch else None
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
    # owner 2026-10-11: an object on an interior shelf keeps its support in the world - the transit to the pre-grasp plans around
    # the whole support (the reach bands check the pre-grasp so), the approach finish leaves out only the board pieces under the object
    split = interior_split(state, ctx, action.object, supports)
    ignored_supports = supports if split is None else split.objects
    if split is not None:
        ctx["interior_support"] = {**split.note, "transit": "support present", "approach_finish": "support present except the board pieces"}
        ctx["plan_ignored"] = _plan_ignored(split.objects, list(ignore_links) + list(split.links))
    # the transit to the pre-grasp: the goal objects and the grasp's own target are protected until the approach begins
    # (owner 2026-09-26, oracle.protection: teacup 0925-222212 / 235559 - the hand passed low over the saucer and shoved it 20 cm)
    # owner 2026-10-12 (A2b): the parts the pre-grasp may overlap (``may_touch.pre_grasp``) are supports of the transit - planned around,
    # out of the world for its last leg and for the supports-ignored fallback
    pre_touch = [o for o in may_touch_objects(state, action, pre_grasp=True).values() if all(o is not s for s in ignored_supports)]
    if pre_touch:
        ctx["plan_ignored"] = list(ctx["plan_ignored"]) + [getattr(o, "name", str(o)) for o in pre_touch]
    with _transit(ctx, motions, state, action, "pre_grasp", target=action.object):
        if action.repose is not None:
            yield from repose_arm(action, motions, ctx)
        yield from _grasp_transit(action, motions, state, targets, ctx, arm, list(ignored_supports) + pre_touch, ignore_links)
        yield from _into_witness_posture(motions, targets, ctx, arm)
    _hook(ctx, "fov")(True)  # from the pre-grasp on the fingertips enter the frame and must stay (recorded grasp pose in frame)
    obj = state.object(action.object)
    cfg = _cfg(ctx)
    finish = ApproachFinish(targets["grasp"], q_hint=_witness(ctx, "grasp"), ignore_objects=[obj] + list(ignored_supports) + touch,
                            max_stall=cfg.approach_finish_max_stall, max_remaining=cfg.approach_finish_max_remaining)
    if split is not None and split.links:
        finish.ignore_links = list(split.links)
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
    support_objs = list(supports) + touch
    if split is not None and not split.note["support_was_ignored"]:
        support_objs.append(split.support)  # the board it rests on is what the fingers can stand on (no contact report found it)
    pressed: List[Any] = []
    trunk0 = motions.state.trunk_error()

    def presses_support() -> bool:
        # the fingers on the support with the trunk bent off its target: the hand stands on it and the arm presses on
        contacts = getattr(state, "hand_contacts", None)
        if contacts is None or motions.state.trunk_error() - trunk0 < cfg.grasp_support_stop_trunk_rad:
            return False
        pressed[:] = [o for o in contacts(arm) if any(o is s for s in support_objs)]
        return bool(pressed)

    if support_objs and cfg.grasp_support_stop_trunk_rad > 0:
        stopped = yield from _until(motions.arm_along(arm, targets["approach"], finish=finish), presses_support)
    else:
        stopped = False
        yield from motions.arm_along(arm, targets["approach"], finish=finish)
    if finish.used:
        ctx["approach_finish"] = finish.info
    if stopped:
        short = float(th.linalg.norm(as_world_pose(targets["grasp"])[0].float() - as_world_pose(state.eef_pose(arm))[0].float()))
        trunk_deg = math.degrees(motions.state.trunk_error())
        un = yield from motions.unload([arm])
        ctx["approach_support_stop"] = {"on": [getattr(o, "name", str(o)) for o in pressed], "trunk_deg": round(trunk_deg, 3), "short_m": round(short, 4),
                                        "unload_rad": {k: round(v, 4) for k, v in un.items()}}
        log.info("grasp(%s): the fingers stand on %s %.1f mm short of the grasp pose (trunk %.2f deg off its target): the approach stops there, unloaded",
                 action.object, ctx["approach_support_stop"]["on"], short * 1000, trunk_deg)
    if obj_pos0 is not None:
        ctx.setdefault("may_touch", {})["target_shift_before_close_m"] = round(float(th.linalg.norm(state.object_pose(action.object)[0] - obj_pos0)), 4)
    # ramped closure (2.25 mm / step, a quarter of the free speed): both pads reach the object in the same step. A handle's
    # link is attached with a FIXED joint by the task's rule (ISpatialGymTask: AG_MOVING_LINK_JOINT_TYPE), the only kind the
    # follow was designed for - the same rule an evaluated policy's grasp meets
    rate, squeeze, thin = grasp_closure(action, width, motions.cfg)
    yield from motions.gripper(arm, close=True, rate=rate, squeeze=squeeze)
    ctx["closure_rate_m_per_step"] = rate
    if thin:
        ctx["thin_pinch"] = {"width_m": round(float(width), 4), "rate_m_per_step": rate, "squeeze_m": squeeze}
    if action.kind == "handle":
        params = (getattr(motions.robot, "_ag_obj_constraint_params", None) or {}).get(arm) or {}
        ctx["ag_joint_type"] = params.get("joint_type")
    held = state.held_object(arm)
    if held is not obj:
        raise SkillError("failed_grasp", f"{action.object} not in the {arm} hand after closing", in_hand=None if held is None else held.name)
    motions.backend.attach_object(arm, obj)
    ctx["attached"] = obj.name


def post_grasp(action: Grasp, state, targets, ctx) -> Optional[str]:
    if state.held_object(action.arm) is not state.object(action.object):
        return "failed_grasp"
    return may_touch_verdict(action, state, ctx, "grasp")


def _free_leg_finish(state, end_pose, witness_q, ignore: Sequence[Any], cfg: ExecuteConfig) -> ApproachFinish:
    """The finish of a free-space leg (lift, retreat): a differential-IK stall anywhere on it is completed by
    ``arm_to`` to the leg's end pose (no contact is being made, a curved finish is fine)."""
    return ApproachFinish(end_pose, q_hint=witness_q, ignore_objects=list(ignore), max_stall=cfg.free_leg_finish_max_stall, max_remaining=cfg.free_leg_finish_max_remaining)


def execute_lift(action: Lift, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    ctx["object_quat0"] = state.object_pose(action.object)[1]
    # the supports the object leaves (the table, a tray) may sit within the planner's activation distance of the
    # fingers at the grasp: out of the collision world for the finish, like the sampler's lift check
    supports = contact_ignores(state, action.object)
    # owner 2026-10-11: off an interior shelf only the board pieces under the object leave it (the board above stays an obstacle)
    split = interior_split(state, ctx, action.object, supports)
    # owner 2026-10-12 (A2b): the lift leaves the parts / distractors the grasp may touch (the fingers start among them); where each
    # stands now, so the postcondition sees one carried along
    touch = may_touch_objects(state, action)
    centres = {inst: _aabb_centre(o) for inst, o in touch.items()}
    ctx["may_touch_z0"] = {inst: float(c[2]) for inst, c in centres.items() if c is not None}
    finish = _free_leg_finish(state, targets["lift"][-1], _witness(ctx, "lift"), list(supports if split is None else split.objects) + list(touch.values()), cfg)
    if split is not None:
        finish.ignore_links = list(split.links)
        ctx["interior_support"] = {**split.note, "lift_finish": "support present except the board pieces"}
    yield from motions.arm_along(action.arm, targets["lift"], finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
    if finish.used:
        ctx["lift_finish"] = finish.info


def post_lift(action: Lift, state, targets, ctx) -> Optional[str]:
    if state.held_object(action.arm) is not state.object(action.object):
        return "failed_grasp"
    return _eef_reached(state, action.arm, targets["lift"][-1], _cfg(ctx), "lift") or may_touch_verdict(action, state, ctx, "lift")


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


RELEASE_WAIT_STEPS = 30


def _until(gen: Iterator[th.Tensor], stop: Callable[[], bool]) -> Generator[th.Tensor, None, bool]:
    """Stream ``gen`` until ``stop()`` holds after a step (the simulator has applied the action), then close it. Returns whether it
    stopped early (read it with ``yield from``)."""
    for a in gen:
        yield a
        if stop():
            close = getattr(gen, "close", None)
            if close is not None:
                close()
            return True
    return False


def _release(motions: Motions, state, arm: str) -> Iterator[th.Tensor]:
    """After the open command: hold until OmniGibson's assisted grasp has let go of what the arm held (its rule acts at the
    open command, the in-hand registry clears after its release window), then drop the planner attachment. Nothing is
    released directly - a demo must release the way a policy does; a grasp that does not let go fails the action."""
    for _ in range(RELEASE_WAIT_STEPS):
        if state.held_object(arm) is None:
            break
        yield from motions.hold(1)
    if state.held_object(arm) is not None:
        raise SkillError("execution", f"the assisted grasp of {arm} did not let go at the open command", arm=arm)
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


def aim_push_past_goal(push: Push, extend_rad: float) -> Push:
    """(owner 2026-09-26, the braced-push ladder) The arc push aimed ``extend_rad`` PAST its goal: mid / goal are the recorded
    init rotated about the recorded arc's pivot by half / all of the lengthened angle and the pad's goal stop moves past the
    goal by the same angle (``contact.extra["aim_past_goal_rad"]``, read by the push legs). The mirror of ``shorten_push``:
    a painting that sags back ~0.015-0.035 rad once the pad leaves (brace untouched) ends short by that sag - pushing past the
    goal by the measured sag is the remedy; moving the brace outward is not (it was never touched)."""
    c = push.contact
    if c.mid is None or extend_rad <= 0:
        return push
    geom = push_geometry(c.waypoints, c.direction)
    if geom.kind != "arc":
        return push
    angle = float(geom.angle)
    new_angle = math.copysign(abs(angle) + float(extend_rad), angle)

    def rotated(frac: float) -> Pose:
        p, q = _rotate_pose_about(c.init, geom.pivot, new_angle * frac, geom.axis)
        return Pose(position=[float(v) for v in p.tolist()], orientation=[float(v) for v in q.tolist()])

    contact = replace(c, mid=rotated(0.5), goal=rotated(1.0), extra={**c.extra, "aim_past_goal_rad": float(extend_rad), "swept_rad": new_angle})
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


@contextmanager
def attach_hold_kept(action: Place, motions: Motions, state, ctx: Dict[str, Any]):
    """Owner 2026-09-30 (first-demo v3.0 F4, cork 0928-041525: the descent seated the cork within 0.7 mm / 0.4 deg, then the jam
    probe relaxed the hold twice - the loaded wrist reports 0.10-0.12 rad/s while moving < 1 mrad, its only contacts the fingers on
    the cork - and joints 5-7 crept 8-21 mrad: 3.9 mm before the offset was measured, 3.3 mm after the correction pass). In an attach
    place with the part in hand the holding arm joins ``keep_hold_arms`` from the descent until the attachment exists, so
    ``Motions._relax_if_jammed`` leaves it alone there; every other place, and this one before the descent, is unchanged. The
    window spans the descent to the hover pose, the hover alignment, the release and the wait for the joint.
    ``ctx["attach_hold"]`` records fired / not fired and why."""
    arm = action.arm
    kept = getattr(motions, "keep_hold_arms", None)
    if action.kind != "attach" or not action.on:
        why = "not an attach place"
    elif kept is None:
        why = "no jam probe to suspend"
    elif state.held_object(arm) is None:
        why = "the part is not in hand"
    else:
        why = None
    if why is not None:
        ctx["attach_hold"] = {"fired": False, "why": why}
        yield
        return
    added = arm not in kept
    kept.add(arm)
    ctx["attach_hold"] = {"fired": True, "arm": arm, "why": "attach place with the part in hand: no jam relax of the holding arm until the attachment exists"}
    try:
        yield
    finally:
        if added:
            kept.discard(arm)


def execute_place(action: Place, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.arm
    drop = action.kind == "drop"
    # the FOV span opened at the pre-grasp stays open through the carry (rule (b)): no hook here
    on = []
    if getattr(action, "on", None) and not drop:  # a drop keeps its container in the collision world: it must not be touched
        if action.relation in ("on", "into") or action.kind == "attach":
            # owner 2026-09-26: a place ignores the partner it is set on / into / attached to - only that. The partner's own supports
            # (the table, the floor) came in through ``contact_supports`` whenever the partner was awake and reporting contacts: the
            # teacup's place ignored breakfast_table + floors in wave 3 (the transit had shoved the saucer awake), only the saucer in
            # wave 2 (the saucer asleep) - the ignore list must not depend on that
            try:
                on = [state.object(action.on)]
            except Exception:  # noqa: BLE001
                on = contact_ignores(state, action.on, extra=(action.on,))
        else:
            # next to / touching: the partner stays an obstacle, only the shared support leaves; under (the large-scale place-under,
            # owner 2026-10-12 item 2c): the partner stays an obstacle above the hand, its support - the floor the object is set on - leaves
            on = contact_ignores(state, action.on)
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in on]
    if action.kind == "attach" and action.on:
        targets = _reaim_attach_targets(action, state, targets, ctx, cfg)
    elif action.kind == "support":
        targets = raise_place_for_inhand_pose(action, state, ctx.get("sidecar"), targets, ctx, cfg)
        if action.on and action.relation not in ("on", "into", "under"):
            targets = clear_release_of_partner(action, motions, state, targets, ctx, cfg)
    # owner 2026-10-11: set down on an interior shelf, the object's support stays in the world - whole for the carry to the pre-place
    # (the sampler checks the pre-place so), without the board pieces under the object for the descent and the retreat finishes
    split = interior_split(state, ctx, action.object, on, place=targets.get("place"), arm=arm)
    if split is not None:
        on = split.objects
        ctx["interior_support"] = {**split.note, "carry": "support present", "descent_finish": "support present except the board pieces",
                                   "retreat_finish": "support present except the board pieces"}
        ctx["plan_ignored"] = _plan_ignored(on, split.links)
    carry = list(targets.get("carry") or ())
    if cfg.carry_leg == "straight" and carry:
        # the carry leg: a straight Cartesian leg from where the hand is (the lift end) to the recorded pre-place, through
        # the recorded carry waypoint when there is one; both ends are in frame, so the fingertips stay in frame along it
        # (the free trajopt transit swung the nextto marker 40 cm back toward the robot and out of the frame). A stall
        # is finished by ``arm_to`` to the pre-place, as for the lift / retreat legs.
        finish = _free_leg_finish(state, targets["pre_place"], _witness(ctx, "pre_place"), on, cfg)
        # the carry is a transit (owner 2026-09-26, oracle.protection): its tracked path, the carried object included, is checked
        # against the protected goal objects before it runs; when it passes too close it is re-run via the point above the pre-place
        with _transit(ctx, motions, state, action, "carry", along="raise"):
            try:
                yield from motions.arm_along(arm, carry, finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
                ctx["carry_leg"] = {"kind": "straight", "waypoints": len(carry), "via_recorded": len(targets.get("carry_via") or ()), "finish": finish.info if finish.used else None}
            except SkillError as e:
                if e.details.get("stage") != "protected":
                    raise
                here = as_world_pose(state.eef_pose(arm))
                pre_p, pre_q = as_world_pose(targets["pre_place"])
                z_top = max(float(here[0][2]), float(pre_p[2])) + float(motions.cfg.protect_above_m)
                legs = [(th.tensor([float(here[0][0]), float(here[0][1]), z_top]), here[1]), (th.tensor([float(pre_p[0]), float(pre_p[1]), z_top]), pre_q), (pre_p, pre_q)]
                P = getattr(motions, "_protect", None)
                if P is not None:
                    P["along"] = "log"  # the via-above leg is executed; a remaining near pass is logged, the run-time check stays
                finish2 = _free_leg_finish(state, targets["pre_place"], _witness(ctx, "pre_place"), on, cfg)
                yield from motions.arm_along(arm, legs, finish=finish2, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
                ctx["carry_leg"] = {"kind": "straight via above", "lift_m": round(z_top - max(float(here[0][2]), float(pre_p[2])), 3), "finish": finish2.info if finish2.used else None,
                                    "protected": e.details.get("protected")}
    else:
        with _transit(ctx, motions, state, action, "carry"):
            yield from motions.arm_to(arm, pose=targets["pre_place"], q_hint=_witness(ctx, "pre_place"), ignore_objects=on)
        ctx["carry_leg"] = {"kind": "trajopt"}
    if action.kind == "attach" and action.on:
        # the part is in the air above the partner: re-aim the descent for its live in-hand pose (and the carry's residual), then
        # end it at the hover pose above the seat - the contact hold keeps the part through the carry and the descent
        targets = yield from _reaim_for_inhand_pose(action, motions, state, targets, ctx, cfg)
        targets = attach_hover_targets(action, motions, state, targets, ctx, cfg)
    finish = ApproachFinish(targets["place"], q_hint=_witness(ctx, "place"), ignore_objects=on, max_stall=cfg.approach_finish_max_stall, max_remaining=cfg.approach_finish_max_remaining)
    if split is not None and split.links:
        finish.ignore_links = list(split.links)
    with attach_hold_kept(action, motions, state, ctx):
        # an attach's insertion leg tracks the rotation to the attach tolerance (owner 2026-09-25, cork 0925-195316: the male arrived
        # tilted 4.4 deg with the default 0.03 rad soft tolerance on the re-aim move and the descent); other places keep the default
        yield from motions.arm_along(arm, targets["descent"], finish=finish, soft_rot_tol=cfg.attach_leg_soft_rot_tol if action.kind == "attach" else None,
                                     limit_margin=cfg.attach_leg_limit_margin_rad if action.kind == "attach" else 0.0)
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
            # aligned laterally and in yaw at the hover pose, touched down on the partner, released in one step; the task's
            # attach rule joins the part there
            yield from _align_at_hover(action, motions, state, ctx, cfg)
            yield from _touch_down(action, motions, state, ctx, cfg)
            yield from _release_into_join(action, motions, state, ctx, cfg)
        else:
            # open at the recorded place pose: a drop releases right there (above the rim, no contact with the container), a
            # support place sets the object down
            if drop and cfg.drop_lower_max_m > 0:
                yield from lower_drop(action, motions, state, ctx, cfg)
            yield from motions.gripper(arm, close=False)
            yield from _release(motions, state, arm)
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
    if split is not None and split.links:
        finish.ignore_links = list(split.links)
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
        yield from motions.arm_along(arm, [pre_live], soft_pos_tol=cfg.attach_correct_min_m, soft_rot_tol=cfg.attach_leg_soft_rot_tol, limit_margin=cfg.attach_leg_limit_margin_rad)
    except SkillError as e:
        rec["pre_place_move_error"] = e.message
        log.warning("place(%s): the move to the re-aimed pre-place did not complete (%s); descending from where the hand is", action.object, e.message)
    rec["applied"] = True
    return out


_FINGER_SURFACE: Dict[Tuple[str, float], Optional[List[Tuple[Any, Any]]]] = {}


def finger_surface_points(arm: str, spacing: float) -> Optional[List[Tuple[Any, Any]]]:
    """Points on the two finger meshes' convex hulls, at most ``spacing`` apart, in the eef frame at finger joint 0, with each finger's
    slide axis (``r1pro_model.finger_hulls_eef`` gives the hull vertices only: a rim edge under a finger's flat underside passes
    between rays cast from its corners). ``[(points (N, 3) ndarray, axis (3,) ndarray), ...]``; None without the assets."""
    key = (arm, round(float(spacing), 5))
    if key in _FINGER_SURFACE:
        return _FINGER_SURFACE[key]
    hulls = M.finger_hulls_eef(arm)
    out = None
    if hulls:
        import numpy as np
        from scipy.spatial import ConvexHull

        out = []
        for pts, axis in hulls:
            samples = [pts]
            for tri in ConvexHull(pts).simplices:
                a, b, c = pts[tri]
                n = max(1, int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing)))
                ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], dtype=float) / n
                samples.append(a + ij[:, :1] * (b - a) + ij[:, 1:] * (c - a))
            out.append((np.unique(np.vstack(samples).round(5), axis=0), axis))
    _FINGER_SURFACE[key] = out
    return out


def _raycast_all(starts, ends):
    """Every hit along each ray (OmniGibson ``raytest_batch``, PhysX scene queries): a list per ray of hit dicts."""
    from omnigibson.utils.sampling_utils import raytest_batch

    return raytest_batch(starts, ends, only_closest=False)


def surface_below(obj, robot=None, inset: float = 0.25, depth: float = 0.6) -> Tuple[Optional[float], int]:
    """The highest surface under ``obj``'s footprint (rays straight down from its AABB bottom at the centre and four inset corners,
    ``depth`` long; hits on the object itself and on ``robot`` do not count). ``(world z or None when nothing is hit, rays cast)``."""
    lo, hi = (C.as_tensor(v).reshape(-1)[:3].float() for v in obj.aabb)
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo) * (1.0 - float(inset))
    z = float(lo[2]) - 1e-3
    pts = [(float(c[0]) + sx * float(h[0]), float(c[1]) + sy * float(h[1])) for sx, sy in ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))]
    starts = [[x, y, z] for x, y in pts]
    ends = [[x, y, z - float(depth)] for x, y in pts]
    skip = [str(getattr(o, "prim_path", "") or "") for o in (obj, robot) if o is not None]
    best = None
    for ray in _raycast_all(starts, ends):
        for hit in sorted((h_ for h_ in (ray if isinstance(ray, list) else [ray]) if h_.get("hit")), key=lambda h_: float(h_.get("distance", 0.0))):
            body = str(hit.get("rigidBody") or "")
            if any(s and body.startswith(s) for s in skip):
                continue
            zz = z - float(hit.get("distance", 0.0))
            best = zz if best is None else max(best, zz)
            break
    return best, len(starts)


def open_fingers_clearance_below(arm: str, eef_pose, q_open: float, skip: Sequence[Any], depth: float, spacing: float) -> Optional[float]:
    """How far the hand at ``eef_pose`` can come straight down before its fingers, fully open (``q_open`` per finger), meet anything
    other than ``skip`` (rays down from the finger surface, ``depth`` long). None without finger meshes; ``depth`` when nothing is met."""
    import numpy as np

    surf = finger_surface_points(arm, spacing)
    if not surf:
        return None
    p, q = as_world_pose(eef_pose)
    R = C.quat_to_mat(C.as_tensor(q).float()).numpy()
    pts = np.vstack([(np.asarray(pp, dtype=float) + float(q_open) * np.asarray(ax, dtype=float)) for pp, ax in surf]) @ R.T + C.as_tensor(p).float().numpy()
    starts = pts.tolist()
    ends = (pts - np.array([0.0, 0.0, float(depth)])).tolist()
    roots = [str(getattr(o, "prim_path", "") or "") for o in skip if o is not None]
    best = float(depth)
    for ray in _raycast_all(starts, ends):
        for h in (ray if isinstance(ray, list) else [ray]):
            body = str(h.get("rigidBody") or "")
            if not h.get("hit") or any(r and body.startswith(r) for r in roots):
                continue
            best = min(best, float(h.get("distance", depth)))
    return best


def lower_drop(action: Place, motions: Motions, state, ctx: Dict[str, Any], cfg: ExecuteConfig) -> Iterator[th.Tensor]:
    """Owner 2026-10-09 (sandal: "the problem is the long fall and bounce, not the instant open"): before a drop's release the held
    object comes straight down from the recorded drop pose until its bottom is ``drop_lower_clear_m`` above the surface under it
    (``surface_below``), at most ``drop_lower_max_m`` - a differential-IK leg checked against the world with only the held object left
    out, kept ``drop_lower_limit_margin_rad`` from the joint limits, that ends where it collides or stalls. ``ctx["drop_lowered"]``
    records the planned and the achieved lowering."""
    arm = action.arm
    rec: Dict[str, Any] = {"applied": False}
    ctx["drop_lowered"] = rec
    try:
        obj = state.object(action.object)
        bottom = float(C.as_tensor(obj.aabb[0]).reshape(-1)[2])
        surface, n_rays = surface_below(obj, getattr(motions, "robot", None))
    except Exception as e:  # noqa: BLE001 - no raycasts / aabb (the test stubs): the recorded drop
        rec["reason"] = f"no surface probe ({e!r})"[:160]
        return
    want = float(cfg.drop_lower_max_m) if surface is None else min(float(cfg.drop_lower_max_m), bottom - surface - float(cfg.drop_lower_clear_m))
    rec.update({"bottom_z": round(bottom, 4), "surface_z": None if surface is None else round(surface, 4), "rays": n_rays})
    # the fingers open fully at the release: they must not come down onto what stands around the object (a bowl's rim - the brussels
    # sprouts lowered into the bowl touched it with the opened fingers)
    try:
        _lo, hi = motions._gripper_limits(arm)
        fingers = open_fingers_clearance_below(arm, state.eef_pose(arm), hi, [obj, getattr(motions, "robot", None)], max(want, 0.0) + float(cfg.drop_lower_finger_clear_m),
                                               float(cfg.place_partner_clear_spacing_m))
    except Exception as e:  # noqa: BLE001
        fingers = None
        rec["finger_probe"] = f"{e!r}"[:120]
    if fingers is not None:
        rec["open_fingers_free_m"] = round(fingers, 4)
        want = min(want, fingers - float(cfg.drop_lower_finger_clear_m))
    rec["planned_m"] = round(want, 4)
    if want < float(cfg.drop_lower_min_m):
        rec["reason"] = f"{max(want, 0.0) * 1000:.0f} mm to lower (the object or its opened fingers already near what is under them)"
        return
    p, q = (x.float() for x in as_world_pose(state.eef_pose(arm)))
    tried: List[Dict[str, Any]] = []
    rec["tried"] = tried
    for frac in cfg.drop_lower_fractions:
        d = want * float(frac)
        if d < float(cfg.drop_lower_min_m):
            break
        try:
            yield from motions.arm_along(arm, [(p - th.tensor([0.0, 0.0, d]), q)], check_collisions=True, ignore_objects=[obj], partial=True,
                                         limit_margin=float(cfg.drop_lower_limit_margin_rad))
            tried.append({"m": round(d, 4), "result": "ran"})
            break
        except SkillError as e:
            # a jump (the straight line crosses a branch flip: the sandal's wrist 5-9 cm down) plans nothing and moves nothing - a
            # shorter leg may stay on this side of it; any other failure ends the lowering where the hand is
            tried.append({"m": round(d, 4), "result": (e.message or e.reason)[:120]})
            if "joint jump" not in str(e.message or ""):
                break
    yield from motions.hold(3)
    p1 = as_world_pose(state.eef_pose(arm))[0].float()
    bottom1 = float(C.as_tensor(obj.aabb[0]).reshape(-1)[2])
    rec.update({"applied": True, "lowered_m": round(float(p[2] - p1[2]), 4), "bottom_above_surface_m": None if surface is None else round(bottom1 - surface, 4)})
    log.info("place(%s): the drop came %.1f cm straight down (planned %.1f cm); its bottom is %s above what is under it", action.object, float(p[2] - p1[2]) * 100,
             want * 100, "?" if surface is None else f"{(bottom1 - surface) * 100:.1f} cm")


def _open_fingers_meet(arm: str, finger_q: Sequence[float], q_open: float, place, partner, cfg: ExecuteConfig) -> Tuple[float, int, List[str], Optional[th.Tensor]]:
    """How far the fingers opening from ``finger_q`` to ``q_open`` (per finger) at the hand pose ``place`` reach into
    ``place_partner_clear_m`` of ``partner``: rays from the finger surface along each finger's slide axis over its opening plus that
    margin, against the partner's collision geometry. ``(largest overlap along a slide axis (m; 0 = clear), rays cast, partner bodies
    hit, the world slide axis of the worst ray or None)``."""
    import numpy as np

    surf = finger_surface_points(arm, cfg.place_partner_clear_spacing_m)
    if not surf:
        return 0.0, 0, [], None
    p1, q1 = as_world_pose(place)
    R = C.quat_to_mat(C.as_tensor(q1).float()).numpy()
    t = C.as_tensor(p1).float().numpy()
    starts, ends, axes, lengths = [], [], [], []
    for (pts, ax), q in zip(surf, finger_q):
        ax = np.asarray(ax, dtype=float)
        ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
        travel = max(float(q_open) - float(q), 0.0) + float(cfg.place_partner_clear_m)
        a_w = R @ ax
        s = (np.asarray(pts, dtype=float) + float(q) * ax) @ R.T + t
        starts.append(s)
        ends.append(s + travel * a_w)
        axes += [a_w] * len(s)
        lengths += [travel] * len(s)
    root = str(getattr(partner, "prim_path", "") or "")
    worst, worst_axis, bodies = 0.0, None, set()
    for i, ray in enumerate(_raycast_all(np.vstack(starts).tolist(), np.vstack(ends).tolist())):
        for h in (ray if isinstance(ray, list) else [ray]):
            body = str(h.get("rigidBody") or "")
            if not h.get("hit") or not root or not body.startswith(root):
                continue
            over = lengths[i] - float(h.get("distance", lengths[i]))
            if over > worst:
                worst, worst_axis = over, axes[i]
            bodies.add(body[len(root):] or body)
    return worst, len(lengths), sorted(bodies), None if worst_axis is None else th.as_tensor(worst_axis).float()


PLACE_TARGET_POSES = ("pre_place", "place")
PLACE_TARGET_PATHS = ("descent", "retreat", "carry", "carry_via")


def shift_place_targets(targets: Dict[str, Any], shift: th.Tensor) -> Dict[str, Any]:
    """Every place target (pre-place, place, descent, retreat, carry) moved by ``shift`` (world, m), orientations kept."""
    d = C.as_tensor(shift).reshape(-1)[:3].float()

    def moved(pose):
        p, q = as_world_pose(pose)
        return Pose(position=[float(v) for v in (p.float() + d).tolist()], orientation=[float(v) for v in q.float().tolist()])

    out = dict(targets)
    for k in PLACE_TARGET_POSES:
        if out.get(k) is not None:
            out[k] = moved(out[k])
    for k in PLACE_TARGET_PATHS:
        if out.get(k):
            out[k] = [moved(w) for w in out[k]]
    return out


def partner_clear_shift(overlap_m: float, slide_axis: Optional[th.Tensor], place_xyz, partner_centre, max_m: float) -> Tuple[Optional[th.Tensor], float]:
    """The horizontal move of a place away from its partner that takes an opening finger ``overlap_m`` (along its world slide axis)
    off it: along the line from the partner's centre to the place, by the overlap over the axis' share of that direction (at least
    0.3), capped at ``max_m``. ``(shift or None when there is no direction away, the move needed (m))``."""
    u = C.as_tensor(place_xyz).reshape(-1)[:3].float() - C.as_tensor(partner_centre).reshape(-1)[:3].float()
    u[2] = 0.0
    n = float(th.linalg.norm(u))
    if n < 1e-6:
        return None, float(overlap_m)
    u = u / n
    cos = abs(float(th.dot(C.as_tensor(slide_axis).reshape(-1)[:3].float(), u))) if slide_axis is not None else 1.0
    need = float(overlap_m) / max(cos, 0.3)
    return u * min(need, float(max_m)), need


def clear_release_of_partner(action: Place, motions: Motions, state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: ExecuteConfig) -> Dict[str, Any]:
    """Owner 2026-10-09 ("touching: can we make the grasp release a bit farther away from the partner, so when releasing the gripper
    fingers don't touch the partner"): the assisted grasp lets go on the full open command only, so at a next-to / touching place the
    pad on the partner's side sweeps its whole opening toward the partner (touching 1008: 15 mm; the box of tissues was shoved 6 mm).
    The opening at the recorded place pose is swept against the partner (``_open_fingers_meet``); when it comes within
    ``place_partner_clear_m`` of it, every place target moves away from the partner by what is missing (``partner_clear_shift``, up to
    ``place_partner_clear_max_m``) and the move is kept for the push that closes the gap after the release (``Motions.place_shifts``:
    ``execute_push`` extends its line by it). ``ctx["partner_release_clearance"]`` records the facts."""
    rec: Dict[str, Any] = {"applied": False}
    ctx["partner_release_clearance"] = rec
    if not cfg.place_partner_clear_max_m > 0 or targets.get("place") is None:
        return targets
    arm = action.arm
    try:
        partner = state.object(action.on)
        idx = motions.robot.gripper_control_idx[arm]
        q_now = [float(v) for v in motions.robot.get_joint_positions()[idx].tolist()]
        _lo, hi = motions._gripper_limits(arm)
        over, n_rays, bodies, axis_w = _open_fingers_meet(arm, q_now, hi, targets["place"], partner, cfg)
        lo_p, hi_p = partner.aabb
        centre = 0.5 * (C.as_tensor(lo_p).float() + C.as_tensor(hi_p).float())
    except Exception as e:  # noqa: BLE001 - no raycasts / finger meshes: the recorded place runs
        rec["reason"] = f"no finger sweep ({e!r})"[:160]
        return targets
    rec.update({"overlap_m": round(over, 4), "rays": n_rays, "partner_bodies": bodies, "finger_q_m": [round(v, 4) for v in q_now]})
    if over <= 0.0:
        return targets
    shift, need = partner_clear_shift(over, axis_w, as_world_pose(targets["place"])[0], centre, cfg.place_partner_clear_max_m)
    rec["needed_m"] = round(need, 4)
    if shift is None:
        rec["reason"] = "the place stands over the partner's centre: no direction away from it"
        return targets
    out = shift_place_targets(targets, shift)
    try:
        over2, _, _, _ = _open_fingers_meet(arm, q_now, hi, out["place"], partner, cfg)
    except Exception:  # noqa: BLE001
        over2 = None
    rec.update({"applied": True, "shift_m": [round(float(v), 4) for v in shift.tolist()], "overlap_after_m": None if over2 is None else round(over2, 4)})
    shifts = getattr(motions, "place_shifts", None)
    if shifts is None:
        shifts = {}
        motions.place_shifts = shifts
    shifts[action.object] = shift.clone()
    log.info("place(%s): the opening fingers would come %.1f mm into %.0f mm of %s at the release: the place moves %.1f mm away from it",
             action.object, over * 1000, cfg.place_partner_clear_m * 1000, action.on, float(th.linalg.norm(shift)) * 1000)
    return out


@dataclass
class AttachCorrection:
    """The attach correction regime (owner 2026-09-20 23:39, attachment only): read from the sidecar's ``attach_correction``
    (the sampler copies ``generation.attach_correction`` of the alias), else from the customs entry of the sidecar's alias
    (a demo-only re-run of an older instance), else these defaults (two passes at ``attach_correct_min_m``). It drives the
    hover alignment (``_align_at_hover``, owner 2026-10-10): settle on arrival (at most ``settle_steps``) -> measure; each pass a
    lateral + yaw leg at ``soft_pos_tol_m`` -> settle -> re-measure; the loop stops when the lateral offset is within
    ``target_pos_m`` and the yaw within ``target_rot_rad``, the lateral improvement falls below ``stop_when_improvement_below_m``
    or the passes are exhausted."""

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


def _collision_points(links: Sequence[Any]) -> Optional[th.Tensor]:
    """World points of the collision geometry's convex hull of every link in ``links`` (None without any)."""
    pts = [th.as_tensor(p).float() for p in (getattr(link, "collision_boundary_points_world", None) for link in links) if p is not None and len(p)]
    return th.cat(pts, dim=0) if pts else None


def _seat_axis(state, action: Place) -> Optional[th.Tensor]:
    """The pair's insertion axis in the world, pointing out of the seat (``predicates.seat_axis``, the attach rule's own)."""
    from ispatialgym.sampling import predicates as PRED

    links = state.attachment_links(action.object, action.on) if hasattr(state, "attachment_links") else None
    if links is None:
        return None
    return PRED.seat_axis(state.object(action.object), state.object(action.on), links[1].body_name).float()


def attach_hover_targets(action: Place, motions: Motions, state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: ExecuteConfig) -> Dict[str, Any]:
    """The attach place's hover pose (owner 2026-10-10): ``place`` - the hand pose that seats the part, re-aimed for its live in-hand
    pose - raised along the seat axis until the lowest point of both the part and the fingertips is ``attach_hover_clear_m`` above
    the partner's top under them (the highest vertex of the partner's collision convexes whose footprint meets theirs, widened by
    ``attach_hover_footprint_margin_m``). ``place`` and ``descent`` end there and the retreat is raised with it. The part and the
    closed fingers are rigid with the hand, so their collision points are carried from where they are now to the seated hand pose.
    ``ctx["attach"]["hover"]`` records the height above the seat, the clearances and the partner's top."""
    rec: Dict[str, Any] = {"applied": False}
    ctx.setdefault("attach", {})["hover"] = rec
    if targets.get("place") is None:
        rec["reason"] = "no place target"
        return targets
    try:
        import omnigibson as og

        og.sim.sync_physx_to_fabric()  # the collision points read the render-side transforms
    except Exception:  # noqa: BLE001
        pass
    child, partner = state.object(action.object), state.object(action.on)
    part = _collision_points(list(child.links.values()))
    fingers = _collision_points(list((getattr(motions.robot, "finger_links", None) or {}).get(action.arm, [])))
    meshes = []
    for link in partner.links.values():
        for mesh in (getattr(link, "collision_meshes", None) or {}).values():
            pts = mesh.points_in_world_frame
            if pts is not None and len(pts):
                meshes.append(th.as_tensor(pts).float())
    axis = _seat_axis(state, action)
    if part is None or not meshes or axis is None:
        rec["reason"] = "no collision geometry / attachment pair"
        log.warning("place(%s): %s - descending to the seated pose", action.object, rec["reason"])
        return targets
    p_now, q_now = (x.float() for x in state.eef_pose(action.arm))
    p_seat, q_seat = (x.float() for x in as_world_pose(targets["place"]))
    carry = C.compose((p_seat, q_seat), C.invert((p_now, q_now)))
    part_s = C.transform_points(carry, part)
    fing_s = C.transform_points(carry, fingers) if fingers is not None else None
    moving = th.cat([part_s] + ([fing_s] if fing_s is not None else []), dim=0)
    lo = moving[:, :2].min(dim=0).values - cfg.attach_hover_footprint_margin_m
    hi = moving[:, :2].max(dim=0).values + cfg.attach_hover_footprint_margin_m
    under = [m for m in meshes if bool((m[:, :2].min(dim=0).values <= hi).all() and (m[:, :2].max(dim=0).values >= lo).all())]
    top = max(float(m[:, 2].max()) for m in (under or meshes))
    low_part = float(part_s[:, 2].min())
    low_fingers = float(fing_s[:, 2].min()) if fing_s is not None else float("inf")
    h = max(0.0, top + cfg.attach_hover_clear_m - min(low_part, low_fingers)) / max(float(axis[2]), 0.5)
    shift = axis * h
    hover = Pose(position=[float(v) for v in (p_seat + shift).tolist()], orientation=[float(v) for v in q_seat.tolist()])
    out = dict(targets)
    out["place"], out["descent"] = hover, [hover]
    if out.get("retreat"):
        out["retreat"] = [Pose(position=[float(v) for v in (as_world_pose(w)[0].float() + shift).tolist()], orientation=[float(v) for v in as_world_pose(w)[1].float().tolist()])
                          for w in out["retreat"]]
    dz = float(shift[2])
    rec.update({"applied": True, "height_above_seat_m": round(h, 4), "part_clearance_m": round(low_part + dz - top, 4),
                "finger_clearance_m": None if fing_s is None else round(low_fingers + dz - top, 4), "partner_top_z": round(top, 4),
                "partner_convexes_under": len(under), "partner_convexes": len(meshes), "seat_axis": [round(float(v), 4) for v in axis.tolist()],
                "lowest": "fingertips" if low_fingers < low_part else "part"})
    log.info("place(%s): hover %.1f mm above the seat - the part %.1f mm and the fingertips %s mm above the top of %s", action.object, h * 1000,
             rec["part_clearance_m"] * 1000, "n/a" if rec["finger_clearance_m"] is None else f"{rec['finger_clearance_m'] * 1000:.1f}", action.on)
    return out


def _hover_offset(state, action: Place) -> Optional[Dict[str, Any]]:
    """The male link against the female one at the hover pose: the vector onto the seat axis (``lateral``, its length
    ``lateral_m``), the height along the axis (``vertical_m``), the yaw about it to the nearest symmetric equivalent of the female
    frame (``yaw_rad``, the twist of the alignment's fix) and the whole angle modulo the symmetry (``rot_rad``)."""
    links = state.attachment_links(action.object, action.on) if hasattr(state, "attachment_links") else None
    off = state.attachment_offset(action.object, action.on) if hasattr(state, "attachment_offset") else None
    axis = _seat_axis(state, action)
    if links is None or off is None or axis is None:
        return None
    pm = links[0].get_position_orientation()[0].float()
    pf = links[1].get_position_orientation()[0].float()
    d = pf - pm
    lateral = d - float(th.dot(d, axis)) * axis
    q_fix = C.as_tensor(off["fix"][1]).float()
    if float(q_fix[3]) < 0.0:
        q_fix = -q_fix
    yaw = 2.0 * math.atan2(float(th.dot(q_fix[:3], axis)), float(q_fix[3]))
    return {"lateral": lateral, "lateral_m": float(th.linalg.norm(lateral)), "vertical_m": float(th.dot(pm - pf, axis)), "yaw_rad": C.wrap_angle(yaw),
            "rot_rad": float(off["rot_rad"]), "male": pm, "axis": axis}


def _hover_note(off: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return None if off is None else {k: round(float(off[k]), 5) for k in ("lateral_m", "vertical_m", "yaw_rad", "rot_rad")}


def _align_at_hover(action: Place, motions: Motions, state, ctx: Dict[str, Any], cfg: ExecuteConfig) -> Iterator[th.Tensor]:
    """Closed-loop alignment at the hover pose (owner 2026-10-10): the hand first settles there - the descent arrives at its full
    speed, and a part released on arrival leaves at that speed - then each pass moves the hand so the male link comes onto the seat
    axis and turns about that axis by the yaw to the nearest symmetric equivalent of the female frame - the hover height and the
    tilt stay - then settles and measures again. The task's correction regime (``AttachCorrection``) gives the passes, the leg's
    soft tolerance, the settle and the targets (lateral ``target_pos_m``, yaw ``target_rot_rad``); the loop also stops when a pass
    improves the lateral offset by less than ``stop_when_improvement_below_m``. ``ctx["attach"]["alignment"]`` records the passes."""
    arm = action.arm
    rec = ctx.setdefault("attach", {}).setdefault("alignment", {})
    reg = attach_correction_regime(ctx, cfg)
    rec["regime"] = reg.as_note()
    n = 0
    for a in motions.settle(max_steps=reg.settle_steps):
        n += 1
        yield a
    rec["settle_steps"] = n
    off = _hover_offset(state, action)
    rec["before"] = _hover_note(off)
    passes: List[Dict[str, Any]] = []
    for k in range(max(0, int(reg.max_passes))):
        if off is None or (off["lateral_m"] <= reg.target_pos_m and abs(off["yaw_rad"]) <= reg.target_rot_rad):
            break
        q_t = C.quat_from_axis_angle(off["axis"], off["yaw_rad"])
        p_e, q_e = (x.float() for x in state.eef_pose(arm))
        target = (C.quat_rotate(q_t, p_e - off["male"]) + off["male"] + off["lateral"], C.qmul(q_t, q_e))
        try:
            yield from motions.arm_along(arm, [target], soft_pos_tol=reg.soft_pos_tol_m, soft_rot_tol=min(0.5 * reg.target_rot_rad, cfg.attach_leg_soft_rot_tol),
                                         limit_margin=cfg.attach_leg_limit_margin_rad)
            yield from motions.settle(max_steps=reg.settle_steps)
        except SkillError as e:
            rec["error"] = e.message
            log.warning("place(%s): hover alignment pass %d did not complete (%s)", action.object, k + 1, e.message)
        off2 = _hover_offset(state, action)
        if off2 is None:
            break
        improvement = off["lateral_m"] - off2["lateral_m"]
        passes.append({**_hover_note(off2), "improvement_m": round(improvement, 5)})
        log.info("place(%s): hover alignment pass %d: lateral %.1f mm, yaw %.2f deg (height %.1f mm)", action.object, k + 1, off2["lateral_m"] * 1000,
                 math.degrees(off2["yaw_rad"]), off2["vertical_m"] * 1000)
        off = off2
        if improvement < reg.stop_when_improvement_below_m and abs(off["yaw_rad"]) <= reg.target_rot_rad and off["lateral_m"] > reg.target_pos_m:
            rec["stopped"] = f"pass {k + 1} improved the lateral offset by {improvement * 1000:.2f} mm"
            break
    rec["passes"] = passes
    rec["after"] = _hover_note(off)


def _touch_down(action: Place, motions: Motions, state, ctx: Dict[str, Any], cfg: ExecuteConfig) -> Iterator[th.Tensor]:
    """The attach touchdown (owner 2026-10-10): after the hover alignment the hand comes straight down the seat axis at
    ``attach_touchdown_speed_m_s`` until PhysX reports the part in contact with its partner (at most the hover's part clearance +
    ``attach_touchdown_extra_m``), then holds still, so the part rests on its partner when the fingers open in one step and the
    attach rule joins it in that step (the release hook leaves a touching part to the step's own check). Released at the hover,
    the contact hold's squeeze let go at once shot the cap 8 mm aside - never joined, the place rolled back and its replay
    failed; touched down it moved < 1.5 mm. A touchdown that fails - its leg raises, or no contact within the depth - releases
    where the hand is: a plain drop. The gripper keeps the contact hold throughout (never a partial opening).
    ``ctx["attach"]["touchdown"]`` records it."""
    from ispatialgym.sampling import predicates as PRED

    arm = action.arm
    child, partner = state.object(action.object), state.object(action.on)
    rec: Dict[str, Any] = {"touched": False, "steps": 0, "fallback": None}
    ctx.setdefault("attach", {})["touchdown"] = rec

    def touching() -> bool:
        try:
            return bool(PRED.pair_in_contact(child, partner))
        except Exception:  # noqa: BLE001 - no contact matrix (the test stubs)
            return False

    axis = _seat_axis(state, action)
    hover = ctx["attach"].get("hover") or {}
    depth = float(hover.get("part_clearance_m") or cfg.attach_hover_clear_m) + float(cfg.attach_touchdown_extra_m)
    p0, q0 = (x.float() for x in state.eef_pose(arm))
    rec["depth_max_m"] = round(depth, 4)
    if touching():
        rec["touched"] = True
    elif axis is None:
        rec["fallback"] = "no seat axis: released at the hover"
    else:
        leg = motions.arm_along(arm, [(p0 - depth * axis, q0)], speed=cfg.attach_touchdown_speed_m_s, converge=False,
                                soft_pos_tol=cfg.attach_touchdown_soft_pos_tol_m, limit_margin=cfg.attach_leg_limit_margin_rad)
        try:
            for a in leg:
                yield a
                rec["steps"] += 1
                if touching():
                    rec["touched"] = True
                    break
        except SkillError as e:
            rec["fallback"] = f"the touchdown leg failed ({e.message}): released where the hand is"
        finally:
            leg.close()
        if not rec["touched"] and rec["fallback"] is None:
            rec["fallback"] = f"no contact within {depth * 1000:.1f} mm: released where the hand is"
    for a in motions.hold(int(cfg.attach_touchdown_hold_steps)):
        yield a
    if axis is not None:
        rec["lowered_m"] = round(float(th.dot(p0 - state.eef_pose(arm)[0].float(), axis)), 5)
    rec["contact_at_release"] = touching()
    if rec["fallback"]:
        log.warning("place(%s): touchdown on %s: %s", action.object, action.on, rec["fallback"])
    else:
        log.info("place(%s): touched down on %s after %d steps (%.1f mm)", action.object, action.on, rec["steps"], 1000 * float(rec.get("lowered_m") or 0.0))


def _release_into_join(action: Place, motions: Motions, state, ctx: Dict[str, Any], cfg: ExecuteConfig) -> Iterator[th.Tensor]:
    """The attach release (owner 2026-10-10): the open command at once (+1 in one step, like every release); the part
    rests on its partner after the touchdown (after a failed one it drops the rest of the gap) and the task's attach rule joins it
    there - in contact, released and aligned; never in mid-air. The hand holds still until the joint exists, at
    most ``attach_join_wait_steps``; the retreat follows. ``ctx["attach"]["release"]`` records the steps from the release to the first
    contact and to the join, the join record (trigger, alignment, snap size), the release hook's verdict and the part's bounce."""
    from ispatialgym.sampling import predicates as PRED

    arm = action.arm
    child, partner = state.object(action.object), state.object(action.on)
    rec: Dict[str, Any] = {"instant": True}
    ctx.setdefault("attach", {})["release"] = rec
    n0 = len(PRED.JOINS)
    PRED.LAST_RELEASE_ATTACH.clear()
    obs: Dict[str, Any] = {"k": 0, "contact": None, "joined": None, "trace": []}

    def watch():
        obs["k"] += 1
        try:
            touching = PRED.pair_in_contact(child, partner)
        except Exception:  # noqa: BLE001 - no contact matrix (the test stubs)
            touching = False
        joined = bool(state.attached(action.object, action.on))
        if touching and obs["contact"] is None:
            obs["contact"] = obs["k"]
        if joined and obs["joined"] is None:
            obs["joined"] = obs["k"]
        h = None if joined else _hover_offset(state, action)
        if h is not None:
            obs["trace"].append((obs["k"], h["vertical_m"], h["lateral_m"], h["rot_rad"], touching))

    for a in motions.gripper(arm, close=False):
        yield a
        watch()
    for a in _release(motions, state, arm):
        yield a
        watch()
    for _ in range(int(cfg.attach_join_wait_steps)):
        if obs["joined"] is not None:
            break
        for a in motions.hold(1):
            yield a
        watch()
    joins = PRED.JOINS[n0:]
    hook = {k: v for k, v in PRED.LAST_RELEASE_ATTACH.items() if k in ("attached", "reason", "trigger", "contact", "lateral_m", "vertical_m", "rot_rad")} or None
    after = [t for t in obs["trace"] if obs["contact"] is not None and t[0] >= obs["contact"]]
    rec.update({"steps_to_first_contact": obs["contact"], "steps_to_join": obs["joined"], "join": joins[0] if joins else None, "release_hook": hook,
                "bounce_m": round(max(t[1] for t in after) - after[0][1], 5) if after else None,
                "max_rot_rad_before_join": round(max(t[3] for t in obs["trace"]), 5) if obs["trace"] else None,
                "attached": bool(state.attached(action.object, action.on))})
    if rec["attached"]:
        j = rec["join"] or {}
        log.info("place(%s): released at hover, first contact after %s steps, joined to %s after %s steps (%s, %s)", action.object, obs["contact"], action.on,
                 obs["joined"], j.get("trigger"), j.get("mode"))
    else:
        log.warning("place(%s): released at hover but not joined to %s within %d steps (first contact after %s steps)", action.object, action.on,
                    cfg.attach_join_wait_steps, obs["contact"])



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
                   cfg: ExecuteConfig, ignore_links: Sequence[Any] = ()) -> Tuple[float, Dict[str, Any]]:
    """The radial distance of a contact escape whose end pose ``here + d1 * far + normal_out`` the collision-aware IK
    (``backend.solve_ik``, world minus ``ignore`` and the collision pieces ``ignore_links``) reaches from the current configuration: ``far`` itself, else
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
        backend.update_world(ignore_objects=ignore, **links_kw(ignore_links))
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


def _start_world_offenders(motions: Motions, state, arm: str, ignore: Sequence[Any], limit: int = 4, ignore_links: Sequence[Any] = ()) -> List[Any]:
    """The scene objects whose collision meshes the arm's CURRENT configuration already penetrates in the planner's
    world with ``ignore`` (and the collision pieces ``ignore_links``) applied (``R1ProCuRobo.start_world_offenders`` →
    ``objects_of_world_meshes``); ``[]`` when the start is clear or the backend has no world model (the test stubs). Never
    raises: a failure here leaves the leg's ignore set as it was."""
    backend = getattr(motions, "backend", None)
    if backend is None or not hasattr(backend, "start_world_offenders") or not hasattr(backend, "objects_of_world_meshes"):
        return []
    try:
        backend.update_world(ignore_objects=list(ignore), **links_kw(ignore_links))
        q_now = backend.q_full() if hasattr(backend, "q_full") else state.robot.get_joint_positions()
        names = backend.start_world_offenders(q_now, emb_name=f"{arm}_arm", limit=limit)
        if not names:
            return []
        objs = [o for o in backend.objects_of_world_meshes(names) if not any(o is x for x in ignore)]
        log.info("retreat(%s): the start configuration already penetrates %s in the planner's world -> %s exempted for this leg (new collisions still count)",
                 arm, [str(n).rsplit("/", 1)[-1] for n in names], [getattr(o, "name", str(o)) for o in objs] or "no scene object (ground slab / unresolved)")
        return objs
    except Exception as e:  # noqa: BLE001
        log.info("retreat(%s): start-collision offenders not resolved (%r)", arm, e)
        return []


def _retreat_witness(action: Retreat, targets, ctx: Dict[str, Any], cfg: ExecuteConfig):
    """The recorded IK witness the retreat's finish aims at (``_witness``), except for a slide-then-lift retreat (``Retreat.lift_after_m``)
    whose witness was recorded for another pose than the leg's end (``ctx["witness_poses"]``, beyond ``retreat_witness_pose_tol_m``): a
    record sampled before the push-under retreat backed out first carries the witness of the straight lift off the push goal, and a
    finish planned to it would lift the hand into the partner's edge. Dropped: the finish takes cuRobo's IK (``ctx["retreat_witness_dropped"]``)."""
    q = _witness(ctx, "retreat")
    if q is None or getattr(action, "lift_after_m", None) is None:
        return q
    rec = (ctx.get("witness_poses") or {}).get("retreat")
    off = None if rec is None else float(th.linalg.norm(C.as_tensor(rec[0]).reshape(-1)[:3].float() - as_world_pose(targets["retreat"][-1])[0].float()))
    if off is not None and off <= cfg.retreat_witness_pose_tol_m:
        return q
    ctx["retreat_witness_dropped"] = {"recorded_pose_off_m": None if off is None else round(off, 4), "tolerance_m": cfg.retreat_witness_pose_tol_m}
    log.info("retreat(%s): the recorded witness belongs to a pose %s from this leg's end (the record predates the slide-then-lift retreat) - not used",
             action.arm, "of unknown distance" if off is None else f"{off * 100:.1f} cm")
    return None


def execute_retreat(action: Retreat, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.arm
    if action.release:
        held = state.held_object(arm)
        if held is not None:
            articulated = bool(getattr(held, "n_joints", 0) or len(getattr(held, "joints", {}) or {}))
            # a handle of an articulated object - a SHEET pinched across its thickness (handle_kind edge_lid, a carton flap) as
            # well - is let go of here, by the open command; the open pads then slide off a flap's free edge (Retreat._edge_escape)
            yield from motions.gripper(arm, close=False)
            yield from _release(motions, state, arm)
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
    # owner 2026-10-11: off an object on an interior shelf the support stays in the world but for the board pieces under the object
    split = interior_split(state, ctx, action.object, ignore) if getattr(action, "object", None) else None
    links: List[Any] = []
    if split is not None:
        ignore, links = split.objects, list(split.links)
        ctx["interior_support"] = {**split.note, "retreat": "support present except the board pieces"}
    # owner 2026-09-24 item 2 (a): a retreat leaves a contact - every object the retreating hand touches where the leg STARTS is
    # out of the collision world for this leg (the partner the push pad ended against, touching 0924-200220 retreat[8]: 'start
    # configuration is in collision: world; offenders: partner_box_of_tissues'); everything else the leg meets is still a collision
    start_touch = [o for o in (getattr(state, "hand_contacts", lambda a: [])(arm) or []) if not any(o is x for x in ignore)]
    if start_touch:
        ctx["retreat_start_contacts"] = sorted(getattr(o, "name", str(o)) for o in start_touch)
        ignore = ignore + start_touch
    # owner 2026-09-25 (touching 0925-134446 / 140113 retreat[8]): the start-in-collision set comes from the PLANNER as well -
    # the physics contact report saw no pad-partner contact while cuRobo's padded spheres put the start configuration inside
    # the partner's mesh ('IK failed [start configuration is in collision: world; offenders: ntbrtz__base_link__collision__0]'),
    # so neither the collision-checked leg nor the arm_to finish had a collision-free start. The objects owning the world
    # meshes the start already penetrates (the leg's ignore set applied) join the leg's and the finish's ignore_objects
    start_off = _start_world_offenders(motions, state, arm, ignore, **links_kw(links))
    if start_off:
        ctx["retreat_start_offenders"] = sorted(getattr(o, "name", str(o)) for o in start_off)
        ignore = ignore + start_off
    finish = _free_leg_finish(state, targets["retreat"][-1], _retreat_witness(action, targets, ctx, cfg), ignore, cfg)
    if links:
        finish.ignore_links = list(links)
    leg = list(targets["retreat"])
    after_x = (getattr(getattr(getattr(action, "after", None), "contact", None), "extra", None) or {}) if isinstance(getattr(action, "after", None), (Push, Brace)) else {}
    if after_x.get("hold"):
        hold = getattr(motions, "press_hold", None) or {}
        motions.press_hold = None
        held_press = True  # the pressing fist leaves a springy flap straight back along its approach, no radial slide on the paper
        other = next((a for a in M.ARMS if a != arm), None)
        held_by_other = state.held_object(other) if other is not None and hasattr(state, "held_object") else None
        if (action.lift_first_m is not None and action.slide_dir is not None and hold.get("arm") == arm and hold.get("object") == action.object
                and held_by_other is not None and held_by_other is state.object(action.object)):
            # the fist sits under the link it lifted and the other hand now holds that object (two-flap lever: the right hand pinches the
            # flap the left fist raised). Straight up first drove the fingers into the held flap's underside: the holding arm yielded 4 mm
            # and the box tipped 1.5 deg / 16 mm through the hinge (0926-073750 retreat[6]). The fist slides out past the flap's free edge
            # first - the flap rises away from it - and lifts after: the same end pose
            here = as_world_pose(state.eef_pose(arm))
            out = here[0] + C.unit(C.as_tensor(action.slide_dir).reshape(-1)[:3].float()) * float(action.distance)
            leg = [(out, here[1]), (out + th.tensor([0.0, 0.0, float(action.lift_first_m)]), here[1])]
            targets["retreat"] = leg
            ctx["retreat_slide_first"] = {"held_by": other, "link": hold.get("link"), "slide_m": float(action.distance), "lift_m": float(action.lift_first_m)}
    else:
        held_press = bool(after_x.get("straight_retreat"))  # a push-close segment: the fist leaves the tip straight back (no radial slide on the link)
    if after_x.get("kind") in ("push_under", "push_touching"):
        # a FLOOR blade (IORS push-under / large-scale touching) leaves along its own approach axis - straight UP to the recorded pre
        # pose (the default leg already in ``targets``); the push-under retreat backs out along the push at its height first, then
        # lifts (``Retreat.lift_after_m``, the leg in ``targets``). The radial escape below combined the partner (the table the object
        # went under is in the contact's support list) and the floor into a diagonal escape normal (back + up) and projected the
        # back-off perpendicular to it: 10 cm back AND 7 cm DOWN, the fingertips into the floor within 0.4 s, the arm pressing the
        # floor for ~100 steps and rocking the base 0.4-0.9 deg (0921-021333 / 021337: the wobble at the end of the push)
        held_press = True
        how = "back along the push at its height, then up" if getattr(action, "lift_after_m", None) is not None else "straight up along -a"
        ctx["retreat_escape"] = {"skipped": f"{after_x.get('kind')}: {how}"}
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
            far_used, probe = _retreat_reach(motions, arm, here, d1, normal_out, far, out, ignore, cfg, **links_kw(links))
            end = here[0] + d1 * far_used + normal_out
            leg = [(here[0] + d1 * out, here[1]), (end, here[1])]
            targets["retreat"] = leg
            finish = _free_leg_finish(state, leg[-1], None, ignore, cfg)
            if links:
                finish.ignore_links = list(links)
            ctx["retreat_escape"] = {"radial": None if radial is None else [round(float(v), 3) for v in radial.tolist()],
                                     "normal": None if n is None else [round(float(v), 3) for v in n.tolist()], "distance_m": far_used}
            if far_used < far - 1e-6:
                ctx["retreat_escape"]["recorded_distance_m"] = far
            if probe:
                ctx["retreat_escape"]["reach_probe"] = probe
    start_pose = as_world_pose(state.eef_pose(arm))
    try:
        yield from motions.arm_along(arm, leg, finish=finish, soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free,
                                     check_collisions=bool(cfg.retreat_check_collisions), limit_margin=float(cfg.retreat_limit_margin_rad), ignore_objects=ignore,
                                     **links_kw(links))
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
            if not cfg.retreat_plan_fallback:
                raise
            # owner 2026-09-26 (axial 0926-061151 retreat[2]: from the push's end configuration the escape leg reached right_arm_joint4's
            # lower limit at waypoint 3/11, both attempts): only here, where the action fails otherwise - a planned transit to the leg's
            # end pose from where the hand is (the same ignore set: the released object, its support, what the start touches)
            try:
                yield from motions.arm_to(arm, pose=targets["retreat"][-1], ignore_objects=list(ignore), **links_kw(links))
            except SkillError as e2:
                raise SkillError(e.reason, f"{e.message}; planned transit to the retreat's end pose: {e2.message[:160]}", **dict(getattr(e, "details", {}) or {})) from e
            ctx["retreat_planned_fallback"] = {"leg_failure": e.message[:200], "moved_m": round(moved, 4), "short_m": round(dp, 4)}
            log.warning("retreat: the leg failed %.1f mm short of its end pose (the pad %.1f mm from where it left the object: %s) - PLANNED transit to the end pose instead",
                        dp * 1000, moved * 1000, e.message[:160])
            if finish.used:
                ctx["retreat_finish"] = finish.info
            return
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
        if "init" in targets:
            out["init"] = Pose(position=[float(v) for v in re_origin(targets["init"]).tolist()], orientation=[float(v) for v in quat.tolist()])
        return out
    except Exception as e:  # noqa: BLE001 - the pitch is an option, never a failure
        log.info("push: pitched blade unavailable (%r)", e)
        return None


def held_brace_arms(ctx: Dict[str, Any], push_arm: str, cfg: ExecuteConfig) -> List[str]:
    """The arms whose braces stand through a push (``ctx["braces_held"]``: the executor's ``braces_held``, read off the program), other
    than the push arm: held in the world during its base nudges when ``push_nudge_hold_brace`` is on, else none."""
    if not cfg.push_nudge_hold_brace:
        return []
    return sorted({b["arm"] for b in (ctx.get("braces_held") or []) if b.get("arm") and b["arm"] != push_arm})


class _NudgeBraceHold:
    """The braces of ``arms`` held where they stand in the world while the base nudges (``push_nudge_hold_brace``; owner 2026-10-14
    ~04:25: "the brace hand must stay where it is in the world while the base nudges"). At the start the world pose of each brace
    hand's COMMANDED target is taken (the FK of the measured joints with the arm at its hold target: a loaded brace keeps its
    wind-up, so its push on the object stays as it was); every control step one IK solve from that target, with the base where it
    is now, moves the arm's target so the commanded hand stays on that pose (as ``Motions.arm_arc_closed_loop``, the joint step
    bounded by ``diff_ik_joint_speed``). Called as ``Motions.track_path``'s ``on_step``: returns True to stop the base when
    ``stop_when()`` says the pushed object is at its goal, or when the base, still more than ``tol`` short of the nudge's ``move``,
    advanced less than ``cfg.push_nudge_stall_m`` over the last ``cfg.push_nudge_stall_steps`` steps (it presses the pair against
    the brace). ``record``: per arm the measured hand's largest / final drift from its start, the IK misses, the steps, the base's
    travel and why the nudge stopped."""

    def __init__(self, motions: Motions, arms: Sequence[str], direction, cfg: ExecuteConfig, stop_when: Optional[Callable[[], bool]] = None,
                 move: float = float("inf"), tol: float = 0.0):
        self.m, self.arms, self.cfg, self.stop_when = motions, list(arms), cfg, stop_when
        self.move, self.tol = float(move), float(tol)
        d = C.as_tensor(direction).reshape(-1)[:3].float()
        self.d = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
        q = motions.robot.get_joint_positions().clone().float()
        self.frames = {a: M.EEF_LINK[a] for a in self.arms}
        self.targets: Dict[str, Any] = {}
        for a in self.arms:
            qt = q.clone()
            qt[motions.robot.arm_control_idx[a]] = motions.state.arm_target[a].float()
            self.targets[a] = as_world_pose(motions.backend.fk([self.frames[a]], qt)[self.frames[a]])
        self.start = {a: C.as_tensor(motions.eef_pose(a)[0]).float().clone() for a in self.arms}
        self.along: List[float] = [float(C.as_tensor(motions.base_pose2d()[:2]).float() @ self.d)]
        self.record: Dict[str, Any] = {"arms": {a: {"max_drift_m": 0.0, "final_drift_m": 0.0, "ik_failures": 0} for a in self.arms}, "steps": 0, "stop": None}

    def __call__(self) -> bool:
        m = self.m
        q = m.robot.get_joint_positions().clone().float()
        for a in self.arms:
            idx = m.robot.arm_control_idx[a]
            q_s = q.clone()
            q_s[idx] = m.state.arm_target[a].float()
            res = m.backend.solve_ik_path(f"{a}_arm", [{self.frames[a]: self.targets[a]}], q_start=q_s, pos_tol=1e-4, rot_tol=5e-4, max_iters=20,
                                          check_collisions=False, sync_world=False, soft_pos_tol=5e-4, soft_rot_tol=2e-3)
            rec = self.record["arms"][a]
            if res.success and res.q_traj is not None and len(res.q_traj) > 0:
                q_cmd = C.as_tensor(res.q_traj[-1]).reshape(-1).float()
                dq = float(th.max(th.abs(q_cmd[idx] - q_s[idx])))
                step = m.cfg.diff_ik_joint_speed * m.dt
                if dq > step:
                    q_cmd = q_s.clone()
                    q_cmd[idx] = q_s[idx] + (C.as_tensor(res.q_traj[-1]).reshape(-1).float()[idx] - q_s[idx]) * (step / dq)
                m.state.set_from_q(q_cmd, arms=[a], trunk=False)
            else:
                rec["ik_failures"] += 1
            drift = float(th.linalg.norm(C.as_tensor(m.eef_pose(a)[0]).float() - self.start[a]))
            rec["max_drift_m"], rec["final_drift_m"] = max(rec["max_drift_m"], drift), drift
        self.record["steps"] += 1
        self.along.append(float(C.as_tensor(m.base_pose2d()[:2]).float() @ self.d))
        self.record["base_moved_m"] = round(self.along[-1] - self.along[0], 4)
        if self.stop_when is not None and self.stop_when():
            self.record["stop"] = "object_at_goal"
            return True
        n = int(self.cfg.push_nudge_stall_steps)
        short = self.move - (self.along[-1] - self.along[0]) > self.tol  # a base settling into the tracker's tolerance is no stall
        if short and len(self.along) > n and self.along[-1] - self.along[-1 - n] < float(self.cfg.push_nudge_stall_m):
            self.record["stop"] = "stalled_against_the_brace"
            return True
        return False


def _base_nudge(motions: Motions, state, arm: str, obj: str, direction, left: float, cfg: ExecuteConfig, label: str,
                speed: Optional[float] = None, hold_arms: Sequence[str] = (), stop_when: Optional[Callable[[], bool]] = None) -> Generator[th.Tensor, None, Optional[Dict[str, Any]]]:
    """Drive the base ``left`` (+ overshoot) along the push ``direction`` with the arm locked at its joints (the door
    base-follow mechanism): the pad in contact pushes the object the rest of the way when the arm sits at its reach
    boundary. The nudge goes a little past the goal (a touching goal is contact) and the tracker's tolerance shrinks with
    the distance (0918-014040: a 7 mm nudge inside the 8 mm tolerance never moved the base). ``speed``: the push's recorded
    pad speed (:func:`push_speed_of`) caps ``push_base_nudge_speed``. ``hold_arms``: the arms bracing the object's partner
    (``ctx["braces_held"]``), held in the world through the nudge (:class:`_NudgeBraceHold`, which also stops the base on
    ``stop_when()`` - the object at its goal - or a stall); returns that hold's record (None without one)."""
    d = C.as_tensor(direction).reshape(-1)[:3].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    move = float(left) + cfg.push_base_nudge_overshoot_m
    tol = min(cfg.push_base_nudge_tol_m, max(0.002, 0.5 * float(left)))
    x, y, yaw = state.base_pose2d()
    goal2d = [x + float(d[0]) * move, y + float(d[1]) * move, yaw]
    hold = _NudgeBraceHold(motions, hold_arms, d, cfg, stop_when=stop_when, move=move, tol=tol) if hold_arms else None
    log.info("%s(%s): %.1f mm still missing with the arm at its reach boundary -> nudging the base %.1f mm along the push direction (tol %.0f mm), arm locked%s",
             label, obj, float(left) * 1000, move * 1000, tol * 1000, f"; the {'/'.join(hold_arms)} brace held in the world" if hold is not None else "")
    q = motions.robot.get_joint_positions()
    motions.state.arm_target[arm] = q[motions.robot.arm_control_idx[arm]].clone().float()
    v = cfg.push_base_nudge_speed if speed is None else min(cfg.push_base_nudge_speed, float(speed))
    tcfg = replace(motions.cfg.base_tracker, v_max=(v, v, 0.2), pos_tol=tol)
    try:
        yield from motions.track_path([goal2d], tracker_cfg=tcfg, on_step=hold)
    except SkillError as e:  # a tracker time-out keeps what the nudge achieved
        log.warning("%s(%s): base nudge did not settle (%s); keeping what it achieved", label, obj, e.message)
        if hold is not None:
            hold.record["stop"] = "tracker_timeout"
    if hold is None:
        return None
    rec = dict(hold.record, label=label, stop=hold.record["stop"] or "nudge_goal")
    rec["arms"] = {a: {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in r.items()} for a, r in rec["arms"].items()}
    log.info("%s(%s): the brace held through the nudge (%s): base %.1f mm, hand drift %s, IK misses %s", label, obj, rec["stop"], 1000 * float(rec.get("base_moved_m") or 0.0),
             {a: f"{1000 * r['max_drift_m']:.1f} mm" for a, r in rec["arms"].items()}, {a: r["ik_failures"] for a, r in rec["arms"].items()})
    return rec


def push_speed_of(contact) -> Optional[float]:
    """The pad speed (m/s) a push contact records (``contacts.push.push_speed_m_s``, the sampler's ``generation.push_speed_m_s``, owner
    2026-10-14 ~02:04: "slow push should be parameter of pushes"): every pad leg of that push runs at it - the approach from the lowered
    pre-contact and the push, the re-push, the strokes, the closed-loop arc, the back-offs, the rock-back round's return onto the face -
    and a base nudge at most at it; None (absent / not positive) = ``push_speed_line`` / ``push_speed_arc`` / the arm's speed."""
    v = (getattr(contact, "extra", None) or {}).get("push_speed_m_s")
    try:
        v = None if v is None else float(v)
    except (TypeError, ValueError):
        return None
    return v if v is not None and v > 0.0 else None


def _speed_kw(ctx: Dict[str, Any]) -> Dict[str, float]:
    """``arm_along`` keyword of the push's recorded pad speed (``ctx["push_speed_m_s"]``), empty without one (the arm's own speed)."""
    v = ctx.get("push_speed_m_s")
    return {} if v is None else {"speed": float(v)}


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
        speed = ctx.get("push_speed_m_s")  # the push's recorded pad speed (push_speed_of)
    if speed is None and ctx.get("push_articulation") is None:
        # ``push_speed_arc`` answers a viscous hinge (the painting's); a link on its joint is held by the loader's joint brake, a
        # Coulomb torque (constraints.joint_brake_spec) that does not grow with the pad's speed - its legs keep the arm's Cartesian speed
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
        if float(res) * sign <= margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0):
            log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d steps -> the pad stops here", label, obj, src, float(res) * sign,
                     margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0), k)
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
    arc_speed = float(ctx.get("push_speed_m_s") or cfg.push_speed_arc)  # the push's recorded pad speed (push_speed_of) first

    def swept() -> float:  # the object's rotation about the hinge axis since the push began (the arc's sense)
        return _rotation_about(q0_obj, C.as_tensor(state.object_pose(obj)[1]).float(), axis)

    # the approach to the recorded first contact pose, at the arc's speed and watched. The recorded pre -> init leg is the
    # first part of the push itself: the fist comes up diagonally (up and into the wall) under the slanted low edge, its
    # top meets the edge within the first centimetres and the straight leg carries the object most of the way (0918-000750 /
    # 0922-180242: the whole rotation happened on that leg at 0.10 m/s). The closed-loop arc takes over only AT ``init`` -
    # the pose the sampler put the pad centre on the edge, on the recorded circle: handed over at the first touch
    # (0922-235343) the pad orbited the hinge from a point still in front of the face plane, the orbit has no component
    # toward the wall, and the edge slipped off the fist (the object turned 2 deg in 86 steps).
    gen = motions.arm_along(arm, [waypoints[0]], loaded=True, speed=arc_speed)
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
            if res is not None and src in ("goal_position", "goal_orientation") and float(res) * sign <= margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0):
                log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d approach steps -> the pad stops here", label, obj, src, float(res) * sign,
                         margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0), k)
                ctx.setdefault("push_stopped", []).append({"leg": f"{label}-approach", "steps": k, "residual": float(res) * sign})
                gen.close()
                return
    ctx["push_first_contact"] = {"approach_steps": first_contact, "swept_rad": round(abs(swept()), 4), "approach_total_steps": k}

    motions.stop_requested = False
    gen = motions.arm_arc_closed_loop(arm, geom.pivot, axis, as_world_pose(waypoints[-1]), swept, float(geom.angle), speed=arc_speed,
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
            if float(res) * sign <= margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0):
                log.info("%s(%s): the object reached its goal (%s residual %.4f <= %.4f) after %d closed-loop steps -> the pad stops here", label, obj, src, float(res) * sign,
                         margin - float(ctx.get("push_aim_past_goal", 0.0) or 0.0), k)
                ctx.setdefault("push_stopped", []).append({"leg": label, "steps": k, "residual": float(res) * sign, "closed_loop": True})
                motions.stop_requested = True
    finally:
        motions.stop_requested = False
    ctx["push_arc_closed_loop"] = {"steps": k, "speed_m_s": arc_speed, **({} if info is None else {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in info.items()})}
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


class _GoalShiftedState:
    """``state`` whose world pose target of ``obj`` is moved by ``shift`` (a following push); everything else is ``state``'s."""

    def __init__(self, state, obj: str, shift: th.Tensor):
        self._state, self._obj, self._shift = state, obj, shift

    def __getattr__(self, name):
        return getattr(self._state, name)

    def pose_target(self, name: str):
        tgt = self._state.pose_target(name)
        if name != self._obj or tgt is None or getattr(tgt, "position", None) is None or str(getattr(tgt, "frame", "world")) != "world":
            return tgt
        return replace(tgt, position=C.as_tensor(tgt.position).reshape(-1)[:3].float() + self._shift)


def _push_goal_follow(action: Push, state, ctx: Dict[str, Any], cfg: ExecuteConfig) -> Optional[th.Tensor]:
    """The planar displacement of a following push's instance (``Push.follow``, the touching partner) since the program start,
    or None below ``push_goal_follow_min_m`` / without a reference. The pushed object's goal was sampled against that instance
    at its init pose (touching: contact with the partner); release validation 2026-10-03: the gripper opening fully at the
    place pose shoved the partner 5.5-6.5 mm away from the target, the brace held it there, and the push to the recorded goal
    ended 6 mm short of touching. ``ctx["push_goal_follow"]`` records the facts."""
    getter = getattr(state, "init_pose", None)
    init = getter(action.follow) if getter is not None else None
    if init is None:
        return None
    try:
        live = state.object_pose(action.follow)
    except Exception:  # noqa: BLE001
        return None
    d = as_world_pose(live)[0].float() - as_world_pose(init)[0].float()
    d[2] = 0.0
    dist = float(th.linalg.norm(d))
    ctx["push_goal_follow"] = {"follow": action.follow, "shift_m": round(dist, 4), "delta": [round(float(v), 4) for v in d.tolist()], "applied": dist >= cfg.push_goal_follow_min_m}
    return d if dist >= cfg.push_goal_follow_min_m else None


def _pitched_push_class(contact) -> bool:
    """A recorded pitched blade class (``push_blade_30`` and steeper, the sampler's palm-away candidates)."""
    try:
        cls = str((getattr(contact, "extra", None) or {}).get("pose_class") or "")
        return cls.startswith("push_blade_") and float(cls[len("push_blade_"):].rstrip("-")) >= 30.0
    except (TypeError, ValueError):
        return False


def push_lowered_pre(contact, pre, first, cfg: ExecuteConfig) -> Optional[Pose]:
    """The pre-contact a straight push is reached at instead of the recorded ``pre`` (owner 2026-09-18 12:42 (1)): ``push_pre_dist_m``
    before the first contact pose ``first`` along the recorded approach, when ``pre`` lies farther than that (+ 5 mm) before it.
    None keeps the recorded pre-contact: an arc (``mid``), a pitched blade class (its shortened pre-pose sits at the table),
    ``push_pre_dist_m`` None, no ``first``. ``first`` must be the recorded ``contact.init``, not the first push waypoint: a straight
    push's waypoints (``actions._line``) leave the start out and begin a third of the push past it, which put a lowered top-down
    blade that far toward the object - over its top once the third exceeds the blade's stand-off from the face."""
    if getattr(contact, "mid", None) is not None or cfg.push_pre_dist_m is None or first is None or _pitched_push_class(contact):
        return None
    p_pre, q_pre = as_world_pose(pre)
    p_first, _ = as_world_pose(first)
    a_dir = C.approach_axis(q_pre)
    if float(th.dot(p_first - p_pre, a_dir)) <= cfg.push_pre_dist_m + 0.005:
        return None
    return Pose(position=[float(v) for v in (p_first - a_dir * cfg.push_pre_dist_m).tolist()], orientation=[float(v) for v in q_pre.tolist()])


# ---- push-under strokes (owner 2026-10-13 ~04:30 item 2): a push that does not turn the object --------------------------------

#: the knobs of a push record's stroke plan (``contacts.push.strokes``); the plan's own values override each. ``backoff_m``: the hand backs
#: off this far along the push between strokes; ``reaim_max_m``: the re-aimed blade stays this close to the recorded line;
#: ``yaw_gain_m_per_rad`` / ``yaw_deadband_rad``: the contact's sideways move per radian of yaw error beyond the deadband (+ = left of the
#: root's line for a counter-clockwise error). Sim check 17838968 (the sandal's round heel, 5.1 deg off): 0 turned it back to 3.1 deg, +0.15
#: put the blade on the heel's flank and turned it 12 deg the wrong way, -0.15 turned it 69 deg
PUSH_STROKE_DEFAULTS: Dict[str, float] = {"backoff_m": 0.03, "reaim_max_m": 0.03, "yaw_gain_m_per_rad": 0.0, "yaw_deadband_rad": math.radians(2.0)}
#: the closed blade of today's push-under (closing axis along the push): the front finger's face 8.4 mm ahead of the eef origin
#: (``tabletop.FINGER_PROFILE`` at the tip), 14 mm wide (``FINGER_SIDE_PROFILE``) - the record's ``face_half_m`` / ``contact_strips_m``
#: describe another hand (a two-point or turned blade)
PUSH_BLADE_FACE_HALF_M = 0.0084
PUSH_BLADE_STRIPS_M: Tuple[Tuple[float, float], ...] = ((0.0, 0.007),)
#: the height band above the fingertips a floor blade meets the object in (``iors_large.IORSLarge.BLADE_SPAN_M``)
PUSH_BLADE_BAND_M = 0.05


def push_stroke_plan(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The push record's stroke plan (``extra["strokes"]``: ``n``, ``ends_m`` - the object's progress along the push after each stroke,
    cumulative - and the :data:`PUSH_STROKE_DEFAULTS` knobs) with the defaults filled in; None without one or with a malformed one."""
    raw = (extra or {}).get("strokes")
    if not isinstance(raw, dict):
        return None
    try:
        n = int(raw.get("n", 0))
        ends = [float(v) for v in (raw.get("ends_m") or [])]
    except (TypeError, ValueError):
        return None
    if n < 1 or len(ends) != n or ends[-1] <= 0.0 or any(b < a for a, b in zip(ends[:-1], ends[1:])):
        return None
    out: Dict[str, Any] = {"n": n, "ends_m": ends}
    for k, v in PUSH_STROKE_DEFAULTS.items():
        r = raw.get(k)
        out[k] = float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(float(r)) else v
    return out


def push_stroke_aim(root_xy, line_xy, direction, yaw_err: float, plan: Dict[str, Any]) -> Tuple[float, float, float]:
    """Which point of the object's back a stroke pushes, across the push (owner 2026-10-13 ~04:30 item 2): ``(target, off, term)`` in metres
    from the recorded line (the line through ``line_xy`` along the unit ``direction``; + = left of the push). ``off``: the object's root off
    that line - the stroke pushes where the line through the root along the push meets the back; ``term``: ``yaw_gain_m_per_rad`` x
    ``yaw_err`` (the object's yaw minus its reference, rad) once ``|yaw_err|`` exceeds ``yaw_deadband_rad``, else 0; ``target`` = ``off +
    term``. The blade's aim that meets the back there is :func:`push_contact_aim`'s, within ``reaim_max_m`` of the line."""
    d = C.as_tensor(direction).reshape(-1)[:2].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    n = th.tensor([-float(d[1]), float(d[0])])
    off = float((C.as_tensor(root_xy).reshape(-1)[:2].float() - C.as_tensor(line_xy).reshape(-1)[:2].float()) @ n)
    term = float(plan["yaw_gain_m_per_rad"]) * float(yaw_err) if abs(float(yaw_err)) > float(plan["yaw_deadband_rad"]) else 0.0
    return off + term, off, term


def _hull_2d(pts: th.Tensor) -> th.Tensor:
    """Convex hull (counter-clockwise, no repeated end) of planar points (N, 2); fewer than three distinct points come back as they are."""
    p = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in pts.tolist()})
    if len(p) < 3:
        return th.tensor(p, dtype=th.float32).reshape(-1, 2)

    def half(seq):
        out: List[Tuple[float, float]] = []
        for q in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(q)
        return out

    lo, hi = half(p), half(list(reversed(p)))
    return th.tensor(lo[:-1] + hi[:-1], dtype=th.float32)


def push_back_contact(pts_world, line_xy, direction, aim: float, strips: Sequence[Sequence[float]], z_lo: float, z_hi: float) -> Optional[Tuple[float, float]]:
    """Where the blade first meets the object moving along the push: ``(along, lateral)`` (from ``line_xy``, along the unit ``direction`` /
    + = left of it) of the rearmost point of the object's footprint within the blade's parts - ``strips`` = ``(lateral centre from the blade's
    aim, half width)`` per part touching the object (today's blade one 14 mm part, a two-point hand its two fingertips) - with the blade at
    ``aim`` across the push. The footprint is the convex hull of the points ``pts_world`` (N, 3) whose height lies in ``[z_lo, z_hi]`` (the
    blade's band), so a part narrower than the vertex spacing still meets the edges it crosses. None when fewer than three points lie in the
    band or no part meets the footprint."""
    P = C.as_tensor(pts_world).reshape(-1, 3).float()
    P = P[(P[:, 2] >= float(z_lo)) & (P[:, 2] <= float(z_hi))]
    if len(P) < 3:
        return None
    d = C.as_tensor(direction).reshape(-1)[:2].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    n = th.tensor([-float(d[1]), float(d[0])])
    rel = P[:, :2] - C.as_tensor(line_xy).reshape(-1)[:2].float()
    hull = _hull_2d(th.stack([rel @ d, rel @ n], dim=1))  # (along, lateral)
    if len(hull) < 3:
        return None
    best: Optional[Tuple[float, float]] = None
    for c, hw in strips:
        lo, hi = float(aim) + float(c) - float(hw), float(aim) + float(c) + float(hw)
        cand = [(float(v[0]), float(v[1])) for v in hull if lo <= float(v[1]) <= hi]
        for a, b in zip(hull, th.roll(hull, -1, dims=0)):
            for lat in (lo, hi):
                da, db = float(a[1]) - lat, float(b[1]) - lat
                if da * db < 0.0:
                    t = da / (da - db)
                    cand.append((float(a[0]) + t * (float(b[0]) - float(a[0])), lat))
        for hit in cand:
            if best is None or hit[0] < best[0]:
                best = hit
    return best


def push_back_along(pts_world, line_xy, direction, aim: float, strips: Sequence[Sequence[float]], z_lo: float, z_hi: float) -> Optional[float]:
    """The along coordinate of :func:`push_back_contact` (None when it has none)."""
    hit = push_back_contact(pts_world, line_xy, direction, aim, strips, z_lo, z_hi)
    return None if hit is None else hit[0]


def push_contact_aim(pts_world, line_xy, direction, target: float, strips: Sequence[Sequence[float]], z_lo: float, z_hi: float,
                     steps: int = 29) -> Optional[Tuple[float, float, float]]:
    """The blade's aim across the push whose first contact with the object (:func:`push_back_contact`) lies nearest the lateral ``target``
    (the point of the object's back a stroke pushes): a blade of one part is tried at ``steps`` aims within its half span of the target - on a
    round heel the part's edge, not its centre, meets the back when the heel's rearmost point lies beside the target -, a hand of several
    parts (the two-point fingertips) straddles it. ``(aim, contact along, contact lateral)``; None when the blade meets nothing."""
    if len(strips) > 1:
        hit = push_back_contact(pts_world, line_xy, direction, target, strips, z_lo, z_hi)
        return None if hit is None else (float(target), hit[0], hit[1])
    span = abs(float(strips[0][0])) + float(strips[0][1])
    best: Optional[Tuple[Tuple[float, float], float, Tuple[float, float]]] = None
    for k in range(int(steps)):
        a = float(target) - span + 2.0 * span * k / max(1, int(steps) - 1)
        hit = push_back_contact(pts_world, line_xy, direction, a, strips, z_lo, z_hi)
        if hit is None:
            continue
        key = (int(abs(hit[1] - float(target)) / 1e-3), abs(a - float(target)))  # the contact within 1 mm: the blade nearest centred on it
        if best is None or key < best[0]:
            best = (key, a, hit)
    return None if best is None else (best[1], best[2][0], best[2][1])


def _object_points_world(state, obj: str) -> Optional[th.Tensor]:
    """The pushed object's collision boundary points in the world now (every link's, ``sim_utils.local_hull``'s source); None when the
    state's object has none (the test stubs)."""
    try:
        o = state.object(obj)
        pts = [p for p in (getattr(link, "collision_boundary_points_world", None) for link in o.links.values()) if p is not None and len(p) > 0]
    except Exception:  # noqa: BLE001 - no live geometry: the caller uses the record's contact offset
        return None
    return th.cat([C.as_tensor(p).reshape(-1, 3).float() for p in pts], dim=0) if pts else None


def push_yaw_error(state, obj: str, pose0) -> Tuple[float, str]:
    """``(error, reference)``: the pushed object's yaw minus the goal's (the nearest orientation equivalence) when its TRO target is a world
    orientation measured in full, else minus its yaw at the push start ``pose0`` (rad)."""
    q = C.as_tensor(state.object_pose(obj)[1]).reshape(-1)[:4].float()
    tgt = getattr(state, "pose_target", lambda name: None)(obj)
    if (tgt is not None and str(getattr(tgt, "frame", "world")) == "world" and getattr(tgt, "orientation", None) is not None
            and str(getattr(tgt, "orientation_metric", "full") or "full") == "full"):
        return -_yaw_residual(C.as_tensor(tgt.orientation).reshape(-1)[:4].float(), q, getattr(tgt, "orientation_equivalences", ()) or ()), "goal"
    return C.wrap_angle(C.yaw_of(q) - C.yaw_of(as_world_pose(pose0)[1].float())), "push start"


def _push_strokes(motions: Motions, state, arm: str, obj: str, targets, contact, geom: "PushGeometry", pose0, cfg: ExecuteConfig, ctx: Dict[str, Any],
                  plan: Dict[str, Any]) -> Iterator[th.Tensor]:
    """The recorded straight push in ``plan["n"]`` strokes (owner 2026-10-13 ~04:30 item 2): before each stroke the object is held still and
    measured, and the blade re-aims where the line through its root along the push direction meets its back (:func:`push_stroke_aim` /
    :func:`push_back_along` over its live collision points; the record's contact offset without them), at the recorded blade height and
    orientation; the stroke then pushes along the push until the object has advanced to ``ends_m[k]`` of the push (the last stroke to the
    recorded goal line), watched as every push leg (stop at the goal). Between strokes the hand backs off ``backoff_m`` along the push. A
    re-aim move that fails falls back to the recorded line for that stroke. ``ctx["push_strokes"]`` records every stroke."""
    extra = getattr(contact, "extra", None) or {}
    wps = targets["waypoints"]
    p_init, q_blade = (x.float() for x in as_world_pose(wps[0]))
    p_goal = as_world_pose(wps[-1])[0].float()
    d3 = geom.direction / max(float(th.linalg.norm(geom.direction)), 1e-9)
    d3 = th.tensor([float(d3[0]), float(d3[1]), 0.0])
    d3 = d3 / max(float(th.linalg.norm(d3)), 1e-9)
    n3 = th.tensor([-float(d3[1]), float(d3[0]), 0.0])
    goal_along = float((p_goal - p_init) @ d3)
    standoff = float(extra.get("blade_standoff_m", 0.0) or 0.0)
    face = float(extra.get("face_half_m", PUSH_BLADE_FACE_HALF_M))
    strips = [tuple(float(v) for v in s) for s in (extra.get("contact_strips_m") or PUSH_BLADE_STRIPS_M)]
    tip_z = float(C.fingertip((p_init, q_blade))[2])
    band = (tip_z, tip_z + PUSH_BLADE_BAND_M)
    res0, _ = push_residual(geom, state, obj, pose0, None)
    total = float(plan["ends_m"][-1])
    rep: Dict[str, Any] = {"n": int(plan["n"]), "plan": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in plan.items()}, "strokes": []}
    ctx["push_strokes"] = rep
    e0, ref = push_yaw_error(state, obj, pose0)
    rep.update({"yaw_reference": ref, "yaw_start_deg": round(math.degrees(e0), 2)})
    for k in range(1, int(plan["n"]) + 1):
        if k > 1:
            p_e, q_e = (C.as_tensor(x).float() for x in motions.eef_pose(arm))
            yield from motions.arm_along(arm, [(p_e - d3 * float(plan["backoff_m"]), q_e)], loaded=True, **_speed_kw(ctx))
            yield from _hold_until_still(motions, state, obj, cfg)
        p_o = C.as_tensor(state.object_pose(obj)[0]).reshape(-1)[:3].float()
        e, _ = push_yaw_error(state, obj, pose0)
        target, off, term = push_stroke_aim(p_o[:2], p_init[:2], d3[:2], e, plan)
        pts = _object_points_world(state, obj)
        hit = push_contact_aim(pts, p_init[:2], d3[:2], target, strips, *band) if pts is not None else None
        cap = abs(float(plan["reaim_max_m"]))
        if hit is not None:
            aim = max(-cap, min(cap, hit[0]))
            back = push_back_contact(pts, p_init[:2], d3[:2], aim, strips, *band) if aim != hit[0] else hit[1:]
            hit = (aim, *back) if back is not None else None
        source = "the object's back" if hit is not None else "the recorded contact offset"
        if hit is None:
            aim = max(-cap, min(cap, target))
            rear, contact_lat = float((p_o - p_init) @ d3) - float(extra.get("half_extent_m", 0.0) or 0.0), aim
        else:
            aim, rear, contact_lat = hit
        res_now, _ = push_residual(geom, state, obj, pose0, None)
        start_along = rear - standoff
        if k == int(plan["n"]) or res0 is None or res_now is None:
            end_along = goal_along
        else:
            want = float(res0) * (1.0 - float(plan["ends_m"][k - 1]) / total)
            end_along = min(goal_along, rear + max(0.0, float(res_now) - want) - face)
        end_along = max(end_along, start_along)
        pose_at = lambda along, lat: (p_init + d3 * along + n3 * lat, q_blade)  # noqa: E731
        rec: Dict[str, Any] = {"stroke": k, "yaw_before_deg": round(math.degrees(e), 2), "root_off_line_m": round(off, 4), "yaw_term_m": round(term, 4),
                               "target_m": round(target, 4), "aim_m": round(aim, 4), "contact_lateral_m": round(contact_lat, 4), "contact": source,
                               "start_along_m": round(start_along, 4), "end_along_m": round(end_along, 4)}
        try:
            yield from motions.arm_along(arm, [pose_at(start_along, aim)], loaded=False)
        except SkillError as err:
            rec["reaim_failed"] = (err.message or err.reason)[:160]
            log.warning("push(%s): stroke %d/%d re-aim %+.1f mm not reached (%s) - the recorded line for this stroke", obj, k, int(plan["n"]), aim * 1000, rec["reaim_failed"])
            aim = 0.0
            rec["aim_m"] = 0.0
            yield from motions.arm_along(arm, [pose_at(start_along, 0.0)], loaded=False)
        s_before = float((C.as_tensor(state.object_pose(obj)[0]).reshape(-1)[:2].float() - pose0[0].reshape(-1)[:2].float()) @ d3[:2])
        n_stop = len(ctx.get("push_stopped") or [])
        yield from _push_leg(motions, state, arm, obj, [pose_at(end_along, aim)], geom, pose0, cfg, ctx, f"stroke {k}/{int(plan['n'])}")
        yield from _hold_until_still(motions, state, obj, cfg)
        p_after = C.as_tensor(state.object_pose(obj)[0]).reshape(-1)[:2].float()
        e1, _ = push_yaw_error(state, obj, pose0)
        rec.update({"pushed_m": round(float((p_after - pose0[0].reshape(-1)[:2].float()) @ d3[:2]) - s_before, 4), "yaw_after_deg": round(math.degrees(e1), 2)})
        rep["strokes"].append(rec)
        log.info("push(%s): stroke %d/%d: yaw %+.1f deg vs the %s, root %+.1f mm off the line (yaw term %+.1f mm) -> contact %+.1f mm, blade %+.1f mm across "
                 "(contact from %s); pushed %.1f mm, yaw %+.1f deg", obj, k, int(plan["n"]), math.degrees(e), ref, off * 1000, term * 1000, contact_lat * 1000, aim * 1000,
                 source, rec["pushed_m"] * 1000, math.degrees(e1))
        if len(ctx.get("push_stopped") or []) > n_stop:
            break
    e_end, _ = push_yaw_error(state, obj, pose0)
    rep["yaw_end_deg"] = round(math.degrees(e_end), 2)
    log.info("push(%s): %d of %d strokes, yaw %+.1f -> %+.1f deg vs the %s", obj, len(rep["strokes"]), int(plan["n"]), rep["yaw_start_deg"], rep["yaw_end_deg"], ref)


def push_transit_witness_clear(motions: Motions, arm: str, objects: Sequence[Any], q_hint, cfg: ExecuteConfig) -> Tuple[bool, str]:
    """Whether the transit to a push pre-contact tries the pushed ``objects`` in the collision world first (``reach_push_pre``):
    ``(False, why)`` when the recorded witness ``q_hint`` puts the arm's spheres closer than ``push_transit_witness_clear_m`` to an
    object's box (``protection.sphere_box_clearance``, the plan check's measure). Without a witness (a lowered pre-contact: cuRobo's IK
    picks the configuration) or a sphere model to measure it, ``(True, why)``."""
    if q_hint is None:
        return True, "no witness (the planner's IK picks the pre-contact's configuration)"
    backend = getattr(motions, "backend", None)
    fn = getattr(backend, "robot_spheres_world", None)
    if fn is None:
        return True, "no sphere model to measure the witness"
    from ispatialgym.oracle import protection as PROT

    try:
        q = C.as_tensor(backend.q_full()).reshape(-1).float().clone()
        idx = motions.robot.arm_control_idx[arm]
        q[idx] = C.as_tensor(q_hint).reshape(-1).float()[idx]
        spheres, links = fn(q.reshape(1, -1), f"{arm}_arm")
        mask = PROT.moving_sphere_mask(links, [arm])
        worst: Optional[Tuple[float, str]] = None
        for o in objects:
            box = PROT.object_box(o)
            if box is None or not bool(mask.any()):
                continue
            clear = float(PROT.sphere_box_clearance(C.as_tensor(spheres)[:, mask], box)[0].reshape(-1)[0])
            if worst is None or clear < worst[0]:
                worst = (clear, str(getattr(o, "name", o)))
    except Exception as e:  # noqa: BLE001 - the measure only decides which world is tried first
        return True, f"the witness not measured ({e!r})"[:160]
    if worst is None:
        return True, "the witness not measured (no box)"
    if worst[0] < cfg.push_transit_witness_clear_m:
        return False, f"the pre-contact witness stands {worst[0]:.4f} m from {worst[1]}'s box (< {cfg.push_transit_witness_clear_m:.3f})"
    return True, f"the pre-contact witness stands {worst[0]:.4f} m from {worst[1]}'s box"


def reach_push_pre(motions: Motions, state, arm: str, pre, supports: Sequence[Any], through: Sequence[Any], q_hint, ctx: Dict[str, Any],
                   cfg: ExecuteConfig, ignore_links: Sequence[Any] = ()) -> Iterator[th.Tensor]:
    """The transit to a push's pre-contact (``reach_pre_pose``). The planner leaves out only what a motion touches, on the leg where it
    touches it (oracle doc 6.4): the pushed objects ``through`` - protected on this leg (``_transit``) - stay in the collision world, and so
    they do for the protect check's via-above and re-plans (touching#04 m3_k0: with the spray bottle left out every plan from the place's
    retreat, 10 cm above the 27 cm bottle, to the pre-contact at its side went through its top, -5.2 cm against its box, the via-above's
    plan and the re-plans too). The world without them - the sampler's for the push chain - is the fallback when no plan exists with them present
    (a ``plan_failed`` that is not the protect check's own), or at once when the recorded witness stands within
    ``push_transit_witness_clear_m`` of one (``push_transit_witness_clear``). ``ctx["push_transit"]`` records the world planned in."""
    kw = dict(ignore_links=ignore_links, accept_other_branch=True, skip_above_m=cfg.pre_pose_standoff_skip_above_m)
    if through and cfg.push_transit_target_present:
        names = [str(getattr(o, "name", o)) for o in through]
        keep, why = push_transit_witness_clear(motions, arm, through, q_hint, cfg)
        if keep:
            try:
                yield from reach_pre_pose(motions, state, arm, pre, supports, [], q_hint, ctx, cfg, **kw)
            except SkillError as e:
                if e.reason != "plan_failed" or e.details.get("stage") == "protected":
                    raise
                why = f"no plan with it present: {e.message[:160]}"
                log.warning("push(%s): no transit to the pre-contact with %s in the collision world (%s) - planning with it left out", arm, names, e.message[:160])
            else:
                ctx["push_transit"] = {"target": "present", "why": why}
                ctx["plan_ignored"] = [n for n in ctx.get("plan_ignored", []) if n not in names]
                return
        ctx["push_transit"] = {"target": "left out", "why": why}
    yield from reach_pre_pose(motions, state, arm, pre, supports, through, q_hint, ctx, cfg, **kw)


class _RecordedPushState:
    """``state`` without the pushed object's TRO pose target: the overhang pick's push (``contacts.push_pick``, kind ``pick_overhang``)
    moves the object toward its support's edge, not toward the task's goal - its stop / residual is the recorded displacement."""

    def __init__(self, state, obj: str):
        self._state, self._obj = state, obj

    def __getattr__(self, name):
        return getattr(self._state, name)

    def pose_target(self, name: str):
        return None if name == self._obj else self._state.pose_target(name)


def pick_push_geometry(geom: "PushGeometry", distance_m: float) -> "PushGeometry":
    """The overhang pick's push as the object's motion: the recorded line with the object's displacement (``contacts.push_pick.distance_m``)
    as its length - the pad travels that plus its start clearance, the residual is measured against what the object should do."""
    return PushGeometry("line", geom.direction, distance=float(distance_m))


def execute_push(action: Push, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    cfg = _cfg(ctx)
    arm = action.contact.arm
    obj = action.contact.object
    extra0 = getattr(action.contact, "extra", None) or {}
    pad_speed = push_speed_of(action.contact)
    if pad_speed is not None:
        ctx["push_speed_m_s"] = pad_speed  # every pad leg below (_push_leg, _push_arc_closed_loop, _speed_kw) and the base nudges' cap
    _hook(ctx, "fov")(False)
    if extra0.get("fingers_opening_m") is not None:
        # an open-hand push (the overhang pick's: the open fingers straddle the object's far rim, a cradle it cannot slide off)
        yield from motions.gripper_opening(arm, float(extra0["fingers_opening_m"]))
    else:
        yield from motions.gripper(arm, close=True)
    pick_push = str(extra0.get("kind") or "") == "pick_overhang"
    if pick_push:
        # owner 2026-10-13 ~03:15 item 4: the object's TRO goal is the task's (the sink) - this push stops on its own recorded displacement,
        # never re-centres on the goal and never nudges the base (the grasp follows from the same stance)
        state = _RecordedPushState(state, obj)
        ctx["push_pick"] = {"distance_m": float(action.contact.distance_m), "support": extra0.get("support"), "fingers": extra0.get("fingers")}
    supports = contact_ignores(state, obj)  # the wall / table the target rests against (sampler rule: ignored along the chain)
    # a re-push after a free short (owner 2026-09-26, braced-push ladder): the pad stops that far PAST the goal (``aim_push_past_goal``)
    ctx["push_aim_past_goal"] = float(extra0.get("aim_past_goal_rad", 0.0) or 0.0)
    if extra0.get("push_contact_form") == "two_point" and extra0.get("fingers_opening_m") is not None:
        # a two-point push-under (owner 2026-10-13 ~04:30 item 2): the fingertips opened above, spread across the object's back
        ctx["push_contact_form"] = {"form": "two_point", "fingers_opening_m": float(extra0["fingers_opening_m"])}
    try:
        # the pushed object leaves the collision world for the push plan (program_check._ignore_for) - except an articulated
        # lid (stage 2): the hand travels from the released handle on one side of the standing lid to the underside pre-pose on
        # the other, and a plan through the lid would knock it off its brake
        through = [] if isinstance(extra0.get("articulation"), dict) else [state.object(obj)]
    except Exception:  # noqa: BLE001
        through = []
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in list(supports) + through]
    if action.contact.mid is None and getattr(action.contact, "init", None) is not None:
        # the first contact pose, moved with the pad below like pre / waypoints (a straight push's waypoints start past it)
        targets = dict(targets)
        targets["init"] = action.contact.init
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
            if "init" in targets:
                targets["init"] = _shifted(targets["init"])
            ctx["push_recentre_m"] = [round(float(v), 4) for v in off.tolist()]
            log.info("push(%s): object %.1f cm off its recorded start position -> pad pre-pose and waypoints shifted by %s", obj, float(th.linalg.norm(off)) * 100, ctx["push_recentre_m"])
            # a place moved away from the partner for the release (clear_release_of_partner) moved the start back along the line by
            # design: the line keeps its recorded end, extended by that move
            place_shift = (getattr(motions, "place_shifts", None) or {}).get(obj)
            back = -float(th.dot(C.as_tensor(place_shift).reshape(-1)[:3].float(), d0)) if place_shift is not None else 0.0
            if back > 0.0 and along < 0.0:
                p_end, q_end = as_world_pose(targets["waypoints"][-1])
                ext = min(back, -along)
                targets["waypoints"] = list(targets["waypoints"]) + [Pose(position=[float(v) for v in (p_end + d0 * ext).tolist()], orientation=[float(v) for v in q_end.tolist()])]
                ctx["push_place_shift_extended_m"] = round(ext, 4)
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
    pitched_class = _pitched_push_class(action.contact)
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
                if "init" in targets:
                    targets["init"] = _lifted(targets["init"])
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
    lowered = push_lowered_pre(action.contact, targets["pre"], targets.get("init"), cfg)
    if lowered is not None:
        pre_pose = lowered
        q_hint_pre = None  # the recorded witness belongs to the recorded pre-pose; cuRobo plans the lowered one
        ctx["push_pre_dist_m"] = cfg.push_pre_dist_m
        ctx["push_pre_lowered"] = {"pre": [round(float(v), 4) for v in as_world_pose(lowered)[0].tolist()],
                                   "init": [round(float(v), 4) for v in as_world_pose(targets["init"])[0].tolist()], "dist_m": cfg.push_pre_dist_m}
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
                        # by prim path, not name: the vectorized OmniGibson (BEHAVIOR-1K #2329) names the object's prim
                        # articulated__<name>
                        obj_path = str(getattr(obj_p, "prim_path", "") or "")
                        for o_ in offs:
                            if obj_path and str(o_).startswith(obj_path + "/"):
                                rel = str(o_)[len(obj_path) + 1:].split("/")
                                link_ign.append((obj_p, rel[0], rel[-1]))
                except Exception as e:  # noqa: BLE001
                    log.info("push(%s): start-collision offenders not resolved (%r)", obj, e)
                if link_ign:
                    ctx["plan_ignored_links"] = sorted(str(e[1]) + ("/" + str(e[2]) if len(e) > 2 else "") for e in link_ign)
            except Exception:  # noqa: BLE001
                link_ign = []
        # the transit to the pre-contact (owner 2026-09-26, oracle.protection): the goal objects and the pushed target are protected
        with _transit(ctx, motions, state, action, "pre_contact", target=obj):
            # owner 2026-09-26 (axial 0926-043254 / 051919): a push's legs are Cartesian and closed-loop from wherever the hand is -
            # a pre-contact witness trajopt cannot reach (another branch) is replaced by cuRobo's own branch instead of failing
            yield from reach_push_pre(motions, state, arm, pre_pose, supports, through, q_hint_pre, ctx, cfg, ignore_links=link_ign)
    shift = _push_goal_follow(action, state, ctx, cfg) if getattr(action, "follow", None) else None
    if shift is not None:
        # after the pad's start was placed against the object where it is: the goal (every stop, residual and re-push below
        # reads it) moves with the partner, and a straight push line is extended by what the partner moved along it
        state = _GoalShiftedState(state, obj, shift)
        d0 = C.as_tensor(targets["direction"]).reshape(-1)[:3].float()
        d0 = d0 / max(float(th.linalg.norm(d0)), 1e-9)
        along = float(th.dot(shift, d0))
        if action.contact.mid is None and along > 0:
            p_end, q_end = as_world_pose(targets["waypoints"][-1])
            targets = dict(targets)
            targets["waypoints"] = list(targets["waypoints"]) + [Pose(position=[float(v) for v in (p_end + d0 * along).tolist()], orientation=[float(v) for v in q_end.tolist()])]
            ctx["push_goal_follow"]["extended_m"] = round(along, 4)
        log.info("push(%s): %s moved %.1f mm since the start -> the goal follows it (line extended %.1f mm)", obj, action.follow, float(th.linalg.norm(shift)) * 1000, max(along, 0.0) * 1000)
    _hook(ctx, "fov")(True)
    pose0 = tuple(x.clone() for x in state.object_pose(obj))
    ctx["object_pose0"] = pose0
    geom = push_geometry(targets["waypoints"], targets["direction"])
    if pick_push:
        geom = pick_push_geometry(geom, float(action.contact.distance_m))
    if geom.kind == "arc":
        off = C.as_tensor(pose0[0]).float() - geom.pivot
        off = off - geom.axis * float(th.dot(off, geom.axis))
        ctx["push_pivot_offset_m"] = float(th.linalg.norm(off))  # the arc's axis vs the object's COM
        ctx["push_axis"] = [round(float(v), 3) for v in geom.axis.tolist()]
    closed_loop = (geom.kind == "arc" and cfg.push_arc_tracking == "closed_loop" and ctx.get("push_articulation") is None
                   and getattr(getattr(motions, "backend", None), "fk", None) is not None and hasattr(motions, "arm_arc_closed_loop"))
    strokes = push_stroke_plan(extra) if geom.kind == "line" and ctx.get("push_articulation") is None else None
    if closed_loop:
        yield from _push_arc_closed_loop(motions, state, arm, obj, targets["waypoints"], geom, pose0, cfg, ctx, "push")
    elif strokes is not None:
        yield from _push_strokes(motions, state, arm, obj, targets, action.contact, geom, pose0, cfg, ctx, strokes)
    else:
        # from a lowered pre-contact the pad finishes the recorded approach at the first contact pose, then pushes: a top-down blade
        # sent from there straight to the first push waypoint would sweep down onto the object's top edge
        leg = list(targets["waypoints"])
        if lowered is not None and not ctx.get("push_pre_skipped"):
            leg = [targets["init"]] + leg
        yield from _push_leg(motions, state, arm, obj, leg, geom, pose0, cfg, ctx, "push")
    nudge_hold_arms = held_brace_arms(ctx, arm, cfg)

    def _object_at_goal() -> bool:
        res_, src_ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
        return res_ is not None and src_ in ("goal_position", "goal_orientation") and float(res_) <= 0.0

    # 1. hold with the pad in contact until the object is still, then measure the residual to the goal
    yield from _hold_until_still(motions, state, obj, cfg)
    residual, source = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
    ctx["push_residual"] = {"kind": geom.kind, "value": residual, "source": source, "recorded": geom.total,
                            "angle_sign": (math.copysign(1.0, geom.angle) if geom.kind == "arc" else 1.0)}
    # the pad's end pose: where it actually stopped when the leg ended at the goal (0917-235415: judged against the
    # recorded arc end, 64 mm / 0.47 rad away -> 'postcondition of push failed'), else the recorded end
    end_pose = (as_world_pose(state.eef_pose(arm)) if (ctx.get("push_stopped") or ctx.get("push_arc_closed_loop") or ctx.get("push_strokes"))
                else as_world_pose(targets["waypoints"][-1]))
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
        if (geom.kind == "line" and cfg.push_base_nudge_max_m > 0 and left is not None and cfg.push_base_nudge_min_m <= left <= cfg.push_base_nudge_max_m
                and ctx.get("push_articulation") is None and not pick_push):
            # (no base nudge for a link on its joint: the lever's vertical lift 0920-201652 drove the base 51 mm with the fist under the flap)
            hold_rec = yield from _base_nudge(motions, state, arm, obj, targets["direction"], left, cfg, "push", speed=ctx.get("push_speed_m_s"),
                                              hold_arms=nudge_hold_arms, stop_when=_object_at_goal if nudge_hold_arms else None)
            if hold_rec is not None:
                ctx.setdefault("push_nudge_brace_hold", []).append(hold_rec)
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
        yield from motions.arm_along(arm, [back], loaded=True, **_speed_kw(ctx))
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
           and cfg.push_backoff_m > 0 and not extra.get("hold") and not pick_push):
        rounds += 1
        log.info("push(%s): %.1f mm reopened after the back-off -> re-nudge round %d", obj, final * 1000, rounds)
        _hook(ctx, "fov")(True)
        yield from motions.arm_along(arm, [end_pose], loaded=True, partial=True, **_speed_kw(ctx))  # back onto the face along the push line
        hold_rec = yield from _base_nudge(motions, state, arm, obj, targets["direction"], final, cfg, f"re-nudge {rounds}", speed=ctx.get("push_speed_m_s"),
                                          hold_arms=nudge_hold_arms, stop_when=_object_at_goal if nudge_hold_arms else None)
        if hold_rec is not None:
            ctx.setdefault("push_nudge_brace_hold", []).append(hold_rec)
        yield from _hold_until_still(motions, state, obj, cfg)
        end_pose = as_world_pose(state.eef_pose(arm))
        back = (end_pose[0] - tangent * cfg.push_backoff_m, end_pose[1])
        yield from motions.arm_along(arm, [back], loaded=True, **_speed_kw(ctx))
        ctx["push_end_pose"] = back
        _hook(ctx, "fov")(False)
        yield from _hold_until_still(motions, state, obj, cfg)
        final, _ = push_residual(geom, state, obj, pose0, ctx.get("push_articulation"))
        ctx["push_residual"].setdefault("renudge", []).append({"round": rounds, "final": final})
    ctx["push_residual"]["final"] = final
    p_now = C.as_tensor(state.object_pose(obj)[0]).float()
    ctx["push_object_drift_m"] = float(th.linalg.norm(p_now - C.as_tensor(pose0[0]).float()))
    ctx["push_stop"] = push_stop_facts(action, state, ctx, p_now, final)


def push_stop_facts(action: Push, state, ctx: Dict[str, Any], p_now, final) -> Dict[str, Any]:
    """Where a push left its object (owner 2026-10-12, IORS touching large-scale: "where the push stopped and how far the partner
    moved"): ``stop`` = ``goal_reached`` (a pad leg stopped at the goal, ``ctx["push_stopped"]``) or ``leg_end``, the object's
    position and residual after the back-off, and for a following push (``Push.follow``) the followed instance's displacement since
    the program start (planar ``moved_m``, ``moved_z_m``)."""
    rec: Dict[str, Any] = {"stop": "goal_reached" if ctx.get("push_stopped") else "leg_end", "legs_stopped": ctx.get("push_stopped") or [],
                           "object_position": [round(float(v), 4) for v in C.as_tensor(p_now).reshape(-1)[:3].tolist()],
                           "residual_final": None if final is None else round(float(final), 4)}
    follow = getattr(action, "follow", None)
    if follow:
        getter = getattr(state, "init_pose", None)
        try:
            init = getter(follow) if getter is not None else None
            live = state.object_pose(follow) if init is not None else None
        except Exception:  # noqa: BLE001
            init = live = None
        if init is not None and live is not None:
            d = as_world_pose(live)[0].float() - as_world_pose(init)[0].float()
            rec["follow"] = {"name": follow, "moved_m": round(float(th.linalg.norm(d[:2])), 4), "moved_z_m": round(float(d[2]), 4)}
    moved = rec.get("follow")
    log.info("push(%s): stopped (%s) with the object at %s, residual %s%s", action.contact.object, rec["stop"], rec["object_position"], rec["residual_final"],
             f"; {moved['name']} moved {1000 * moved['moved_m']:.1f} mm since the program start" if moved else "")
    return rec


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
    if (getattr(contact, "extra", None) or {}).get("placed_at") == "goal":
        return None  # the sampler placed the stop at the goal edge already (a second rotation would move it off the target)
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


def _brace_follow_targets(action: Brace, state, targets: Dict[str, Any], ctx: Dict[str, Any], cfg: ExecuteConfig) -> Dict[str, Any]:
    """A following brace (``Brace.follow``) re-aimed at the braced object's LIVE pose: the recorded pre / contact / approach stand
    ``stop_clearance_m`` off the face of the object at its init pose; when it has moved since the program start by more than
    ``brace_follow_min_m`` (touching, release validation 2026-10-03: the gripper opening fully at the place pose shoved the
    partner 5.5-6.5 mm toward the stop, the stop pressed into it and the push's start configuration was in collision), every
    target moves with the object's planar rigid motion (translation + yaw about its position). ``ctx["brace_follow"]`` records
    the facts; the same targets are returned when nothing moves."""
    obj = action.contact.object
    getter = getattr(state, "init_pose", None)
    init = getter(obj) if getter is not None else None
    if init is None:
        return targets
    try:
        live = state.object_pose(obj)
    except Exception:  # noqa: BLE001
        return targets
    p0, q0 = as_world_pose(init)
    p1, q1 = as_world_pose(live)
    p0, p1 = p0.float(), p1.float()
    d = p1 - p0
    dist = float(th.linalg.norm(d))
    dyaw = float(C.wrap_angle(C.yaw_of(q1) - C.yaw_of(q0)))
    rec: Dict[str, Any] = {"object": obj, "shift_m": round(dist, 4), "delta": [round(float(v), 4) for v in d.tolist()], "yaw_rad": round(dyaw, 4)}
    ctx["brace_follow"] = rec
    if dist < cfg.brace_follow_min_m and abs(dyaw) < 0.005:
        rec["applied"] = False
        return targets
    qz = C.quat_from_yaw(dyaw)

    def moved(pose):
        p, q = as_world_pose(pose)
        p_new = p1 + C.quat_rotate(qz, p.float() - p0)
        return Pose(position=[float(v) for v in p_new.tolist()], orientation=[float(v) for v in C.qmul(qz, q.float()).tolist()])

    out = {"pre": moved(targets["pre"]), "contact": moved(targets["contact"]), "approach": [moved(w) for w in targets["approach"]]}
    rec["applied"] = True
    log.info("brace(%s): moved %.1f mm / %.1f deg since the start -> the stop follows it", obj, dist * 1000, math.degrees(dyaw))
    return out


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
    elif getattr(action, "follow", False):
        followed = _brace_follow_targets(action, state, targets, ctx, cfg)
        if followed is not targets:
            targets.update(followed)
            q_hint_pre = None  # the recorded witness belongs to the recorded pre-pose; cuRobo plans the moved one
    opening = (getattr(action.contact, "extra", None) or {}).get("fingers_opening_m")
    if opening is not None:
        # a WIDE stop (owner 2026-09-19 00:49): the fingers pre-shaped to the recorded opening so both tips land on the braced
        # face, spread along it - a lever arm against the pivot a single closed tip cannot resist (touching 0918-224622: the
        # partner yawed 0.8 deg about the point stop and the pair lost contact)
        yield from motions.gripper_opening(arm, float(opening))
    else:
        yield from motions.gripper(arm, close=True)
    ctx["plan_ignored"] = [getattr(o, "name", str(o)) for o in supports]
    # the transit to the brace's pre-contact (owner 2026-09-26, oracle.protection): the goal objects and the braced object are protected
    with _transit(ctx, motions, state, action, "pre_contact", target=action.contact.object):
        try:
            yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, [], q_hint_pre, ctx, cfg, skip_above_m=cfg.pre_pose_standoff_skip_above_m)
        except SkillError as e:
            if at_goal is None or e.reason != "plan_failed":
                raise
            # the goal-edge pre pose is out of this stance's reach: the recorded brace (validated by the sampler) stands instead
            log.warning("brace(%s): the goal-edge pre pose is not reachable (%s) - bracing at the recorded contact", action.contact.object, e.message)
            targets.update(recorded)
            ctx["brace_at_goal"] = {**ctx.get("brace_at_goal", {}), "fallback": "recorded contact", "reason": e.message[:160]}
            yield from reach_pre_pose(motions, state, arm, targets["pre"], supports, [], _witness(ctx, "pre"), ctx, cfg, skip_above_m=cfg.pre_pose_standoff_skip_above_m)
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


def _live_hinge(state, art) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """``(point, unit axis)`` of the recorded hinge (``articulation.hinge_world`` / ``axis_world``) where it is now: it moves with
    the object (T_live o T_rec^-1 applied to the recorded point / axis when the record carries the object's pose). None without
    a recorded hinge."""
    ex = getattr(art, "extra", None) or {}
    if ex.get("hinge_world") is None or ex.get("axis_world") is None:
        return None
    hinge = C.as_tensor(ex["hinge_world"]).reshape(-1)[:3].float()
    axis = C.unit(C.as_tensor(ex["axis_world"]).reshape(-1)[:3].float())
    rec = ex.get("object_pose")
    if isinstance(rec, dict):
        live = as_world_pose(state.object_pose(art.object))
        rec_pose = (C.as_tensor(rec["position"]).float(), C.as_tensor(rec["orientation"]).float())
        d_pose = C.compose(live, C.invert(rec_pose))
        dp_, dq_ = C.as_tensor(d_pose[0]).float(), C.as_tensor(d_pose[1]).float()
        hinge = C.quat_rotate(dq_, hinge) + dp_
        axis = C.unit(C.quat_rotate(dq_, axis))
    return hinge, axis


def _follow_handle_poses(action: BaseFollow, motions: Motions, state, targets: Dict[str, Any], cfg: ExecuteConfig, note: Dict[str, Any]) -> Iterator[th.Tensor]:
    """The arm-mode follow: the hand drives the handle through the recorded poses. A fixed-base articulation streams them as recorded
    (straight between poses). For a free-standing body the hand turns rigidly about the hinge as it stands when the follow begins
    (``_live_hinge``) through the recorded joint values (``targets["q_values"]``), one waypoint per ``follow_arc_step_rad``: the chords
    between the five recorded poses pulled the handle off its circle (blender open 1008: torso 0.41 deg, the body tipped 0.9 deg), an
    arc snapped onto each recorded pose jerked the hand by the grasp's offset from it, and a hinge re-aimed for the body's live pose
    followed a tipping blender until it ran away (4.6 deg, 14 mm). After a chunk of ``follow_reaim_chunk`` waypoints that left the torso
    bent beyond ``follow_slow_trunk_rad`` the next runs at ``follow_slow_factor`` of the pace. ``note["handle_arc"]`` records it."""
    poses = targets["handle_poses"]
    art = action.articulation
    try:
        fixed = getattr(state.object(art.object), "fixed_base", None)
        free = fixed is not None and not bool(fixed)
        hinge = _live_hinge(state, art) if free else None
        hand0 = as_world_pose(motions.eef_pose(action.arm)) if free else None
        q0 = float(state.joint_position(art.object, art.joint)) if free else None
        q_vals = [float(v) for v in (targets.get("q_values") or [])]
    except Exception:  # noqa: BLE001 - no live pose (the test stubs): the recorded chords
        free = False
    if not free or hinge is None or len(q_vals) != len(poses) or not q_vals or float(cfg.follow_arc_step_rad) <= 0:
        yield from motions.arm_along(action.arm, poses, converge=False)
        return
    centre, axis = (C.as_tensor(v).reshape(-1)[:3].float() for v in hinge)
    p0, r0 = (C.as_tensor(v).float() for v in hand0)

    def turned(ang: float) -> Tuple[th.Tensor, th.Tensor]:
        rq = C.quat_from_axis_angle(axis, float(ang))
        return centre + C.quat_rotate(rq, p0 - centre), C.qmul(rq, r0)

    # the sense of a positive joint delta about the axis: the one that carries the hand toward the first recorded pose
    d1, first = q_vals[0] - q0, as_world_pose(poses[0])[0].float()
    sense = 1.0 if float(th.linalg.norm(turned(d1)[0] - first)) <= float(th.linalg.norm(turned(-d1)[0] - first)) else -1.0
    total = q_vals[-1] - q0
    n = max(1, int(math.ceil(abs(total) / float(cfg.follow_arc_step_rad))))
    dense = [turned(sense * total * k / n) for k in range(1, n + 1)]
    info: Dict[str, Any] = {"waypoints": n, "rad": round(total, 4), "sense": sense, "chunks": 0, "slowed": 0,
                            "end_off_recorded_m": round(float(th.linalg.norm(dense[-1][0] - as_world_pose(poses[-1])[0].float())), 4)}
    note["handle_arc"] = info
    speed: Optional[float] = None
    K = max(1, int(cfg.follow_reaim_chunk))
    for k in range(0, len(dense), K):
        yield from motions.arm_along(action.arm, dense[k:k + K], converge=False, speed=speed)
        info["chunks"] += 1
        if motions.state.trunk_error() > float(cfg.follow_slow_trunk_rad):
            speed = float(motions.cfg.cartesian_speed) * float(cfg.follow_slow_factor)
            info["slowed"] += 1
        else:
            speed = None


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
        gen = _follow_handle_poses(action, motions, state, targets, cfg, note)
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
    # a stage-2 hand-over (``articulation.follow_to``, sampler ``stage2_push``): the program's fist push takes the joint on from
    # q(follow_to), where the sampler found the hand's reach ends - the hand's part ends there, as post_base_follow judges it
    handover = (getattr(art, "extra", None) or {}).get("follow_to")
    q_hand = art.q_at(float(handover)) if handover is not None else art.q_goal

    def hand_reached() -> bool:
        return sign * (state.joint_position(art.object, art.joint) - q_hand) >= -band

    if handover is not None:
        note["handover"] = {"follow_to": float(handover), "q": q_hand}
    if targets["mode"] == "arm" and cfg.follow_stop_at_goal and not hand_reached() and (getattr(art, "extra", None) or {}).get("hinge_world") is not None:
        # owner 2026-09-21: the recorded motion ended short of the goal band (0921-021042: 0.06 rad, the hand a few mm short of the last
        # pose) - the hand rides ON along the hinge arc by what is missing (+ the band), the live hinge (the object may have shifted),
        # until the joint reaches the band; the remaining shortfall stays in the record
        try:
            hinge, axis = _live_hinge(state, art)
            q_now = float(state.joint_position(art.object, art.joint))
            d_ext = sign * (abs(q_hand - q_now) + band + 0.02)
            here = as_world_pose(state.eef_pose(action.arm))
            ext = C.orbit_waypoints(here, hinge, C.quat_from_axis_angle(axis, d_ext), n=3, frame="world")
            log.info("base_follow(%s): the recorded motion ended at q %.3f (the hand's part ends at %.3f): riding on along the hinge arc by %.3f rad", action.arm, q_now, q_hand, d_ext)
            gen2 = motions.arm_along(action.arm, list(ext), converge=False)
            k2 = 0
            for a in gen2:
                yield a
                k2 += 1
                if hand_reached():
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
        # the torso the hand's load bent goes with the arm (owner 2026-10-09: dishwasher close 0.24 deg, door open 0.38 deg kept bent
        # through the arm's unwind - the unwound arm is a rigid link - and sprung back when the gripper opened)
        trunk_total = _unwind_trunk(motions)
        prev = state.object_pose(art.object)
        for k_u in range(1, max(0, int(cfg.follow_unwind_steps) - 1) + 1):
            yield from motions.hold(1)
            cur = state.object_pose(art.object)
            dp_u, da_u = _pose_error(cur, prev)
            prev = cur
            if dp_u > cfg.push_still_pos or da_u > cfg.push_still_rot:
                # the object's body is the loop's only free link (a free-standing carton, a blender): re-read every step, the loaded arm
                # creeps and the body turns with it (carton lever 0926-073750: +0.97 deg over the re-reads at 0.08 deg per step, still on
                # the first step with the targets frozen). The targets stay where the last re-read put them; a fixed-base root never
                # moves, so doors and appliances unwind as before
                note["unwind_stopped"] = {"after_steps": k_u, "root_step_m": round(dp_u, 5), "root_step_rad": round(da_u, 5)}
                break
            total += motions.unwind(action.arm)
            trunk_total += _unwind_trunk(motions, threshold=trunk_total == 0.0)  # the unwound arm hands its load to the torso
        note["unwind_rad"] = total
        if trunk_total > 0.0:
            note["unwind_trunk_rad"] = round(trunk_total, 5)
    hold_steps, q_end = yield from _hold_until_joint_still(motions, state, art.object, art.joint, cfg)
    note.update({"hold_steps": hold_steps, "q_end": q_end, "shortfall_rad": sign * (art.q_goal - q_end)})
    ctx["follow"] = note


def _unwind_trunk(motions, threshold: bool = True) -> float:
    """``Motions.unwind_trunk``: from ``cfg.unload_min_trunk_rad`` on for the first re-target of an unwind (``threshold``), every re-read
    after it; 0 for a motions object without a trunk."""
    fn = getattr(motions, "unwind_trunk", None)
    if fn is None:
        return 0.0
    return fn(float(getattr(getattr(motions, "cfg", None), "unload_min_trunk_rad", 0.0)) if threshold else 0.0)


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


def keep_retreat_arm(motions: Motions, state, arm: str, seg: Dict[str, Any], cfg: ExecuteConfig) -> Dict[str, Any]:
    """Owner 2026-09-30 (first-demo v3.0 F2, door open 0928-042225: after the first release and the 10 cm back-off the ramp onto the
    recorded ``arm_q_retreat`` turned joint 1 by 3.4 rad and raised the hand 1.04 -> 1.69 m and back, then the re-grasp plan swung it
    back to the pre-grasp witness - the sampler's retreat IK had taken the largest-margin solution of the pose anywhere). Whether a
    ``release_move_regrasp`` keeps the arm where the back-off left it instead of ramping onto ``arm_q_retreat``: only when the
    recorded witness lies farther than ``follow_retreat_ramp_max_rad`` from the measured joints (another IK solution of the same hand
    pose, not null-space drift), the configurations of the drive along ``seg["base_path"]`` with the arm as it is are collision-free
    (whole body, ``follow_keep_drive_margin_m``) and the arm keeps ``follow_keep_self_clearance_m`` from cuRobo's self-collision
    padding (a re-grasp plan may start from it). Reads only; ``fired`` False = ramp as before, with the reason."""
    idx = motions.robot.arm_control_idx[arm]
    q_now = C.as_tensor(motions.robot.get_joint_positions()).reshape(-1).float().clone()
    gap = float(th.max(th.abs(q_now[idx] - C.as_tensor(seg["arm_q_retreat"]).reshape(-1).float())))
    f: Dict[str, Any] = {"fired": False, "gap_rad": round(gap, 3), "threshold_rad": cfg.follow_retreat_ramp_max_rad}
    if gap <= cfg.follow_retreat_ramp_max_rad:
        return {**f, "why": "the recorded retreat witness is near the measured joints: ramped onto it"}
    if seg.get("hold_posture") == "idle":
        return {**f, "why": "the recorded drive holds the idle posture: ramped onto the recorded witness"}
    backend = motions.backend
    try:
        qs = path_sweep(motions, state, seg.get("base_path") or [])
        hits = C.as_tensor(backend.check_collision(th.stack(qs), sync_world=True, margin=cfg.follow_keep_drive_margin_m)).reshape(-1)
        if bool(hits.any()):
            return {**f, "why": f"the arm as it is collides on the drive (configuration {int(th.nonzero(hits)[0])} of {len(qs)}): ramped onto the recorded witness"}
        pairs = backend.self_collision_pairs(q_now, f"{arm}_arm", margin=cfg.follow_keep_self_clearance_m)
        if pairs:
            return {**f, "why": f"the arm as it is lies within {cfg.follow_keep_self_clearance_m * 1000:.0f} mm of the self-collision padding "
                                f"({'-'.join(pairs[0]['links'])}): ramped onto the recorded witness"}
    except Exception as e:  # noqa: BLE001 - a backend without the checks (the test stubs): ramp as before
        return {**f, "why": f"no drive / self-clearance check ({e!r}): ramped onto the recorded witness"[:200]}
    return {**f, "fired": True, "drive_configurations": len(qs),
            "why": "the recorded retreat witness is another IK solution; the drive and the self-clearance check out with the arm as it is: kept"}


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
            # 1. unwind gradually (the arm and the torso it bent), hold until the joint is still, release (full opening for a handle), settle
            total = motions.unwind(arm)
            trunk_total = _unwind_trunk(motions)
            for _ in range(max(0, int(cfg.follow_unwind_steps) - 1)):
                yield from motions.hold(1)
                total += motions.unwind(arm)
                trunk_total += _unwind_trunk(motions, threshold=trunk_total == 0.0)
            yield from _hold_until_joint_still(motions, state, art.object, art.joint, cfg)
            late = _unwind_trunk(motions)  # the leaf's swing went on loading the torso while the joint came to rest (door open)
            if late > 0.0:
                trunk_total += late
                yield from motions.hold(int(cfg.follow_still_steps))
            yield from motions.gripper(arm, close=False)
            yield from _release(motions, state, arm)
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
            retreat_kept = None
            if seg.get("arm_q_retreat"):
                retreat_kept = keep_retreat_arm(motions, state, arm, seg, cfg)
                if retreat_kept["fired"]:
                    log.info("base_follow: release at q = %.3f: retreat %.2f m; the recorded retreat witness lies %.3f rad from the arm (> %.2f, another IK solution): "
                             "the arm stays where the back-off left it", float(seg.get("q") or 0.0), float(seg.get("retreat_m") or C.PRE_DIST), retreat_kept["gap_rad"],
                             cfg.follow_retreat_ramp_max_rad)
                else:
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
            yield from motions.gripper(arm, close=True, rate=motions.cfg.gripper_close_rate)
            if state.held_object(arm) is not obj:
                raise SkillError("failed_grasp", f"{art.object} not in the {arm} hand after the re-grasp of segment {j}", stage="regrasp")
            motions.backend.attach_object(arm, obj)
            motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
            log_moves.append({"type": "release_move_regrasp", "q": seg.get("q"), "strides": len(seg["base_path"]), "hold_posture": seg.get("hold_posture"), "unwind_rad": round(total, 4),
                              **({"unwind_trunk_rad": round(trunk_total, 5)} if trunk_total > 0.0 else {}),
                              "m": round(math.hypot(seg["base_to"][0] - seg["base_from"][0], seg["base_to"][1] - seg["base_from"][1]), 3),
                              **({"retreat_kept": retreat_kept} if retreat_kept is not None else {})})
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
# push-close (room door, outer face)
# ==========================================================================================

HANDLE_STANDOFF_M = 0.06  # a bar / lever handle's grasp contact sits about this far in front of the leaf's outer face


def rot_z(v: th.Tensor, ang: float) -> th.Tensor:
    c, s = math.cos(ang), math.sin(ang)
    return th.tensor([c * float(v[0]) - s * float(v[1]), s * float(v[0]) + c * float(v[1]), float(v[2])], dtype=th.float32)


def leaf_frame(hinge_xy: Sequence[float], handle_xy_closed: Sequence[float], approach_closed: Sequence[float], standoff_m: float = HANDLE_STANDOFF_M):
    """Unit horizontal directions of the closed leaf: ``d0`` from the hinge along the **leaf plane** toward the free edge,
    ``m0`` = the outer-face normal (toward the robot: −approach), and the swing sign ``s`` such that the leaf direction at
    opening angle q is ``rot(s·q)·d0`` about +z (the outer face swings toward the robot). The handle contact lies
    ``standoff_m`` in front of the outer face, so the hinge→handle direction is tilted toward m by atan(standoff / r);
    it is rotated back (0920-024158 … 051131: the uncorrected plane put every push 5–6 cm in front of the leaf)."""
    dh = th.tensor([handle_xy_closed[0] - hinge_xy[0], handle_xy_closed[1] - hinge_xy[1], 0.0], dtype=th.float32)
    r = float(dh.norm())
    dh = dh / r
    m_ref = -th.tensor([approach_closed[0], approach_closed[1], 0.0], dtype=th.float32)
    m_ref = m_ref / m_ref.norm()
    plus = th.tensor([-dh[1], dh[0], 0.0])  # dh rotated +90° about z
    s = 1.0 if float(th.dot(plus, m_ref)) > 0 else -1.0
    alpha = math.atan2(float(standoff_m), max(r, 1e-6))
    d0 = rot_z(dh, -s * alpha)
    m0 = rot_z(d0, s * math.pi / 2)
    return d0, m0, s


def leaf_width(handle_radius_m: float, standoff_m: float = HANDLE_STANDOFF_M, edge_m: float = 0.035) -> float:
    """Hinge → free edge along the leaf: the handle's in-plane radius plus its distance to the edge (the leaf link's AABB
    at an open angle is inflated by the thickness projection)."""
    return math.sqrt(max(handle_radius_m ** 2 - standoff_m ** 2, 0.0)) + edge_m


def _leaf_dims(obj, hinge_xy, d0: th.Tensor, m0: th.Tensor, leaf_link: str) -> Tuple[float, float]:
    """(width from the hinge to the free edge, thickness) of the leaf from its link AABB, in the frame ``d0`` / ``m0``."""
    link = (getattr(obj, "links", {}) or {}).get(leaf_link)
    if link is None:
        return 0.0, 0.04
    lo, hi = link.aabb
    corners = th.tensor([[float(x), float(y)] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    rel = corners - th.tensor([[hinge_xy[0], hinge_xy[1]]])
    width = float(th.max(rel @ d0[:2]))
    thick = float(th.max(rel @ m0[:2]) - th.min(rel @ m0[:2]))
    return width, min(max(thick, 0.03), 0.08)


def _arms_ready(motions: Motions, rep: Dict[str, Any], ignore_objects=()) -> Iterator[th.Tensor]:
    """Both arms to "ready" before the base drives beside the leaf ("tucked" self-collides with the base on this robot,
    owner 2026-09-20 18:43)."""
    try:
        yield from motions.arm_to(("left", "right"), q={"left": posture_q(motions.robot, "ready", "left"), "right": posture_q(motions.robot, "ready", "right")},
                                  ignore_objects=list(ignore_objects))
        rep["tucked"] = "ready"
    except SkillError as e:
        rep["tucked"] = f"ready failed: {(e.message or e.reason)[:120]}"
        log.warning("push_close: arms to ready failed: %s", e)


def guarded_approach(motions: Motions, arm: str, approach: th.Tensor, q_of, step_m: float = 0.02, max_m: float = 0.30,
                     move_rad: float = math.radians(0.4), deflect_m: float = 0.012):
    """Advance the hand along ``approach`` in ``step_m`` increments until the leaf starts to move (``q_of`` changes by
    ``move_rad``) or the eef deflects ``deflect_m`` against the surface (loaded arm): the *actual* contact pose — the
    leaf's face plane is only known to the handle stand-off. Yields actions; returns ``(contact_pose or None,
    travelled_m, how)`` through the generator's return value."""
    a = C.unit(approach)
    q0 = float(q_of())
    travelled = 0.0
    how = "no contact"
    while travelled < max_m - 1e-6:
        eef = motions.eef_pose(arm)
        target = (C.as_tensor(eef[0]) + step_m * a, C.as_tensor(eef[1]))
        try:
            yield from motions.arm_along(arm, [target], loaded=True)
        except SkillError as e:
            if _blocked_at_surface(e):
                return motions.eef_pose(arm), travelled, f"arm blocked at the surface ({e.message or e.reason})"
            how = f"arm stalled: {e.message or e.reason}"
            break
        travelled += step_m
        yield from motions.hold(2)
        if abs(float(q_of()) - q0) >= move_rad:
            return motions.eef_pose(arm), travelled, "leaf moved"
        d = motions._deflection(arm)
        if d is not None and d >= deflect_m:
            return motions.eef_pose(arm), travelled, f"eef deflected {d * 1000:.0f} mm"
    return None, travelled, how


def _blocked_at_surface(e: SkillError) -> bool:
    """A guarded step's ``arm_along`` failure that means the hand presses on the surface: the differential IK stalled at least 6 mm
    short with no joint at a limit."""
    det = getattr(e, "details", {}) or {}
    msg = str(e.message or "")
    pos_err = float(det.get("pos_err", 0.0) or 0.0)
    pos_err = pos_err / 1000.0 if pos_err > 1.0 else pos_err  # mm in some diagnostics
    limit_free = ("at_lower=[]" in msg and "at_upper=[]" in msg) or ("at_lower" not in msg)
    return limit_free and pos_err >= 0.006


def guarded_approach_hands(motions: Motions, arms: Sequence[str], approach: th.Tensor, moved_of: Dict[str, Callable[[], float]], step_m: float = 0.02,
                           fine_m: float = 0.005, fine_from_m: float = 0.0, max_m: float = 0.30, move_m: float = 0.003, deflect_m: float = 0.012):
    """The guarded approach of a two-hand push (owner 2026-10-12 ~20:17 item C): the hands advance along ``approach`` in turn, ``step_m``
    each up to ``fine_from_m`` of travel and ``fine_m`` beyond, until one touches - during its step the object moves ``move_m`` at that
    hand's spot (``moved_of[arm]()``, a coordinate along the push), or the hand is blocked at the surface, or it deflects ``deflect_m`` -
    and holds there; the others go on in ``fine_m`` steps until they touch too (at most ``max_m`` of travel each), so the second touches a
    few mm after the first. Yields actions; returns ``{arm: (contact pose or None, travelled_m, how)}`` through the generator's return
    value."""
    a = C.unit(approach)
    out: Dict[str, Tuple[Any, float, str]] = {}
    travelled = {arm: 0.0 for arm in arms}
    while len(out) < len(arms):
        for arm in [x for x in arms if x not in out]:
            coarse = fine_from_m - travelled[arm]
            step = fine_m if coarse <= 1e-9 or any(r[0] is not None for r in out.values()) else min(step_m, coarse)
            if travelled[arm] + step > max_m + 1e-6:
                out[arm] = (None, travelled[arm], "no contact")
                continue
            s0 = float(moved_of[arm]())
            eef = motions.eef_pose(arm)
            try:
                yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) + step * a, C.as_tensor(eef[1]))], loaded=True)
            except SkillError as e:
                blocked = _blocked_at_surface(e)
                out[arm] = (motions.eef_pose(arm) if blocked else None, travelled[arm], f"arm {'blocked at the surface' if blocked else 'stalled'} ({e.message or e.reason})")
                continue
            travelled[arm] += step
            yield from motions.hold(2)
            if float(moved_of[arm]()) - s0 >= move_m:
                out[arm] = (motions.eef_pose(arm), travelled[arm], "object moved")
                continue
            d = motions._deflection(arm)
            if d is not None and d >= deflect_m:
                out[arm] = (motions.eef_pose(arm), travelled[arm], f"eef deflected {d * 1000:.0f} mm")
    return out


def _q_at_base(motions: Motions, q_full: th.Tensor, x: float, y: float, yaw: float) -> th.Tensor:
    """``q_full`` with the base virtual joints set to the world pose (x, y, yaw)."""
    q = q_full.clone()
    jidx = motions.backend._jidx
    p_root, q_root = motions.backend.world_to_root((th.tensor([x, y, float(q[jidx["base_footprint_z_joint"]])]), C.quat_from_yaw(yaw)))
    q[jidx["base_footprint_x_joint"]] = float(p_root[0])
    q[jidx["base_footprint_y_joint"]] = float(p_root[1])
    q[jidx["base_footprint_rz_joint"]] = C.yaw_of(q_root)
    return q


def pushable_arc_from(motions: Motions, arm: str, base: Tuple[float, float, float], pre, contact, arc: List[Any]) -> Tuple[int, Optional[th.Tensor]]:
    """The backend's collision-aware IK from a candidate base pose BEFORE driving there: the pre pose, the contact pose,
    then the arc waypoints with chained seeds. ``(n reachable arc waypoints, q at the pre pose)``, ``(-1, None)`` when
    the pre or the contact pose is unreachable."""
    emb = f"{arm}_arm"
    link = M.EEF_LINK[arm]
    q_seed = _q_at_base(motions, motions.backend.q_full(), *base)
    q_pre = motions.backend.solve_ik(emb, {link: (C.as_tensor(pre[0]), C.as_tensor(pre[1]))}, q_seed=q_seed, sync_world=False)
    if q_pre is None:
        return -1, None
    q = motions.backend.solve_ik(emb, {link: (C.as_tensor(contact[0]), C.as_tensor(contact[1]))}, q_seed=q_pre, sync_world=False)
    if q is None:
        return -1, None
    n = 0
    for wp in arc:
        q_w = motions.backend.solve_ik(emb, {link: (C.as_tensor(wp[0]), C.as_tensor(wp[1]))}, q_seed=q, sync_world=False)
        if q_w is None:
            break
        q = q_w
        n += 1
    return n, q_pre


def planned_push(plan: Dict[str, Any], q: float, live=None) -> Dict[str, Any]:
    """A planned push-close chunk (``PushClose.plan``, sampler ``articulation_large.plan_push_finish``) with the leaf at joint value
    ``q``: the recorded fist contact (planned with the leaf at ``q_from``) turned with the leaf about the vertical hinge line by
    s·(q − q_from) - the same point of the leaf -, its pre-contact ``pre_m`` back along the approach, the approach (into the face),
    the outer-face normal (horizontal, toward the robot), the arc centre (the hinge at the contact's height) and the signed angle
    about +z that turns the leaf from ``q`` to the planned landing ``q_to`` (``remaining`` = how far, 0 once there). ``live`` = the
    object's pose now: everything moves with the object when it moved since the record (``object_pose``)."""
    s = float(plan["swing_sign"])
    hinge = C.as_tensor(plan["hinge_world"]).reshape(-1)[:3].float()
    p_c, q_c = (v.float() for v in as_world_pose(plan["contact"]))
    normal = C.as_tensor(plan["normal"]).reshape(-1)[:3].float()
    if live is not None and isinstance(plan.get("object_pose"), dict):
        d_pos, d_quat = C.compose(as_world_pose(live), C.invert(as_world_pose(plan["object_pose"])))
        hinge, p_c = C.quat_rotate(d_quat, hinge) + d_pos, C.quat_rotate(d_quat, p_c) + d_pos
        q_c, normal = C.qmul(d_quat, q_c), C.quat_rotate(d_quat, normal)
    turn = C.quat_from_axis_angle([0.0, 0.0, 1.0], s * (float(q) - float(plan["q_from"])))
    centre = th.tensor([float(hinge[0]), float(hinge[1]), float(p_c[2])])
    contact = (centre + C.quat_rotate(turn, p_c - centre), C.qmul(turn, q_c))
    n = C.quat_rotate(turn, th.tensor([float(normal[0]), float(normal[1]), 0.0]))
    remaining = max(float(q) - float(plan["q_to"]), 0.0)
    return {"contact": contact, "pre": C.pre_pose(contact, float(plan.get("pre_m", 0.10))), "approach": C.approach_axis(contact[1]), "normal": C.unit(n),
            "centre": centre, "remaining": remaining, "delta": -s * remaining}


def _execute_push_close_planned(action: PushClose, motions: Motions, state, ctx) -> Iterator[th.Tensor]:
    """A cabinet door's push finish (owner 2026-10-12 ~04:20): the hand has followed the handle as far as the arm reaches and let
    go; the closed hand pushes the leaf's outer face the rest of the way shut with the stance and the contact the sampler planned
    (``PushClose.plan``, :func:`planned_push`) - drive back to the planned stance only when the base stands off it, the pre-contact
    planned to the sampler's arm witness (the object present; then cuRobo's branch; then the object ignored), the guarded approach
    along the planned approach, the arc about the hinge to the planned landing (stopped once the leaf is closed or the trunk is
    pressed, unloaded), the fist back along its approach; no base drag. The last chunk is repeated while it still gains."""
    arm, name = action.arm, action.object
    cfg = _cfg(ctx)
    rep: Dict[str, Any] = {"chunks": [], "planned": len(action.plan or [])}
    ctx["push_close"] = rep
    _hook(ctx, "fov")(False)

    def q_now() -> float:
        return float(state.joint_position(name, action.joint))

    def leaf_stop(trunk0: float) -> Optional[str]:
        if q_now() - action.q_lower <= cfg.push_close_stop_rad:
            return f"leaf closed (q {math.degrees(q_now()):.2f} deg)"
        if motions.state.trunk_error() - trunk0 >= cfg.push_close_stop_trunk_rad:
            return f"trunk pressed {math.degrees(motions.state.trunk_error()):.2f} deg off its target (q {math.degrees(q_now()):.2f} deg)"
        return None

    def live_pose():
        try:
            return as_world_pose(state.object_pose(name))
        except Exception:  # noqa: BLE001 - no live pose (the test stubs): the record as it is
            return None

    obj = state.object(name)
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    plans = [dict(p) for p in (action.plan or [])]
    q0 = q_now()
    rep["geometry"] = {"hinge_world": [round(float(v), 3) for v in plans[0]["hinge_world"]], "q_start_deg": round(math.degrees(q0), 2),
                       "q_planned_from_deg": round(math.degrees(float(plans[0]["q_from"])), 2), "q_to_deg": [round(math.degrees(float(p["q_to"])), 2) for p in plans]}
    motions.backend.update_world()
    for attempt in range(len(plans) + max(0, int(action.plan_repeats))):
        q = q_now()
        if q - action.q_lower <= action.closed_tol_rad:
            break
        k = min(attempt, len(plans) - 1)
        plan = plans[k]
        g = planned_push(plan, q, live_pose())
        rec: Dict[str, Any] = {"chunk": k, "repeat": attempt >= len(plans), "q_before_deg": round(math.degrees(q), 2), "contact": [round(float(v), 3) for v in g["contact"][0]],
                               "delta_deg": round(math.degrees(g["remaining"]), 1)}
        if g["remaining"] <= cfg.push_finish_landing_tol_rad:
            rec["result"] = f"the leaf is at the planned landing ({math.degrees(float(plan['q_to'])):.1f} deg)"
            rep["chunks"].append(rec)
            break
        target = as_pose2d(plan["base_pose2d"])
        off_m, off_rad = _base_error(motions.base_pose2d(), target)
        rec["stance_off"] = {"m": round(off_m, 3), "rad": round(off_rad, 3)}
        log.info("push_close (planned chunk %d%s): leaf %.1f deg, %.1f deg to the planned landing; the base stands %.3f m / %.3f rad off the planned stance",
                 k, ", repeated" if rec["repeat"] else "", math.degrees(q), math.degrees(g["remaining"]), off_m, off_rad)
        if off_m > cfg.push_finish_restance_m or off_rad > cfg.push_finish_restance_rad:
            yield from _arms_ready(motions, rec)
            try:
                yield from motions.track_path([target])
                yield from motions.settle(max_steps=20)
            except SkillError as e:
                rec["result"] = f"drive to the planned stance failed: {e.message or e.reason}"
                rep["chunks"].append(rec)
                break
            rec["stance"] = [round(v, 3) for v in target]
        motions.backend.update_world()
        yield from motions.gripper(arm, close=True)
        tries: List[Tuple[str, Dict[str, Any]]] = []
        if plan.get("arm_q_pre") is not None and abs(q - float(plan["q_from"])) <= cfg.push_finish_hint_max_rad:
            hint = C.as_tensor(motions.backend.q_full()).reshape(-1).float().clone()
            hint[idx] = C.as_tensor(plan["arm_q_pre"]).reshape(-1).float()
            tries.append(("the recorded witness", {"q_hint": hint, "accept_other_branch": True}))
        tries += [("the object present", {}), ("the object ignored", {"ignore_objects": [obj]})]
        rec["pre"] = None
        for how, kw in tries:
            try:
                yield from motions.arm_to(arm, pose=g["pre"], **kw)
                rec["pre"] = how
                break
            except SkillError as e:
                rec.setdefault("pre_failures", []).append(f"{how}: {(e.message or e.reason)[:140]}")
        if rec["pre"] is None:
            rec["result"] = "pre-contact unreachable"
            rep["chunks"].append(rec)
            log.warning("push_close (planned chunk %d): %s (%s)", k, rec["result"], rec.get("pre_failures"))
            break
        a = g["approach"]
        found, travelled, how = yield from guarded_approach(motions, arm, a, q_now)
        rec["approach"] = {"travelled_m": round(travelled, 3), "how": how}
        if found is not None and "blocked" in how:
            # contact evidence, as for a room door: a further 1 cm push must move the leaf, else the stall was the reach limit
            q_a = q_now()
            try:
                eef = motions.eef_pose(arm)
                yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) + 0.01 * a, C.as_tensor(eef[1]))], loaded=True)
            except SkillError:
                pass
            yield from motions.hold(3)
            moved = abs(q_now() - q_a) >= math.radians(0.3)
            rec["approach"]["evidence"] = {"door_moved_deg": round(math.degrees(q_now() - q_a), 2), "contact": moved}
            if not moved:
                found = None
                how = "stall at the reach limit (door did not move under a 1 cm push)"
        if found is None:
            rec["result"] = f"no contact ({how})"
            rep["chunks"].append(rec)
            log.warning("push_close (planned chunk %d): %s", k, rec["result"])
            break
        cpose = motions.eef_pose(arm)
        cpose = (C.as_tensor(cpose[0]), C.as_tensor(cpose[1]))
        g = planned_push(plan, q_now(), live_pose())  # the approach may have moved the leaf a little
        why: List[Optional[str]] = [None]
        trunk0 = motions.state.trunk_error()

        def stop() -> bool:
            why[0] = leaf_stop(trunk0)
            return why[0] is not None

        if g["remaining"] <= cfg.push_finish_landing_tol_rad:
            rec["arc"] = "none (the approach brought the leaf to the planned landing)"
        else:
            n_exec = max(int(round(g["remaining"] / math.radians(2.0))), 1)
            centre = th.tensor([float(g["centre"][0]), float(g["centre"][1]), float(cpose[0][2])])
            arc = C.arc_waypoints(cpose, centre, th.tensor([0.0, 0.0, 1.0]), g["delta"], n=n_exec)
            try:
                stopped = yield from _until(motions.arm_along(arm, list(arc), loaded=True, partial=True), stop)
                rec["arc"] = f"stopped: {why[0]}" if stopped else "complete"
            except SkillError as e:
                rec["arc"] = f"stopped: {e.message or e.reason}"
            if why[0] is not None:
                un = yield from motions.unload([arm])
                rec["arc_unload_rad"] = {kk: round(v, 4) for kk, v in un.items()}
        yield from motions.settle(max_steps=40)
        q_after = q_now()
        rec["q_after_deg"] = round(math.degrees(q_after), 2)
        un = yield from motions.unload([arm])  # the fist leaves a leaf it may still press: nothing springs back when it goes
        if any(v > 0.0 for v in un.values()):
            rec["backoff_unload_rad"] = {kk: round(v, 4) for kk, v in un.items()}
        try:  # off the face, back along the fist's approach, then let it settle
            eef = motions.eef_pose(arm)
            a_now = C.approach_axis(C.as_tensor(eef[1]))
            yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) - 0.10 * a_now, C.as_tensor(eef[1]))], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
        except SkillError as e:
            rec["back_off_failed"] = e.message or e.reason
        yield from motions.settle(max_steps=40)
        rec["q_settled_deg"] = round(math.degrees(q_now()), 2)
        motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
        rep["chunks"].append(rec)
        log.info("push_close (planned chunk %d): door %.1f -> %.1f deg (settled %.1f), %s", k, math.degrees(q), math.degrees(q_after), math.degrees(q_now()), rec.get("arc"))
        if q - q_now() < math.radians(0.5):
            rec["result"] = "no progress"
            break
    q_end = q_now()
    rep["final_q_deg"] = round(math.degrees(q_end), 2)
    rep["closed"] = bool(q_end - action.q_lower <= action.closed_tol_rad)
    rep["verdict"] = "closed by pushing" if rep["closed"] else f"stopped at {math.degrees(q_end):.1f} deg"
    mon = ctx.get("monitors")
    if mon is not None:
        rep["contacts"] = [dict(e) for e in getattr(mon, "collision_events", [])[-40:]]
    log.info("push_close (planned) done: %s (%d chunks); contacts %s", rep["verdict"], len(rep["chunks"]), rep.get("contacts"))


def execute_push_close(action: PushClose, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    """Owner decision B (2026-09-20 20:19), sequential: per chunk - pick the stance on the +m side of the leaf by the
    backend's IK of the contact pose and the arc waypoints (front and side candidates, longest pushable arc wins), drive
    there (arms "ready"), then arm only: guarded approach along −m onto the outer face (a diff-IK stall counts as contact
    only if the door moves under a 1 cm further push), push along the arc about the hinge to the arm's limit (the base
    drags along −m when the arc gains < 2°), back off; repeat until closed or a chunk gains nothing. A planned push (a
    cabinet door's push finish, ``action.plan``) runs :func:`_execute_push_close_planned` instead. A cabinet door closed
    room-door style (``programs.push_close_options``) brings the leaf the sampler measured - width, outer face, handle stand-off,
    closed joint value - in place of the room door's estimates."""
    if action.plan:
        yield from _execute_push_close_planned(action, motions, state, ctx)
        return
    arm, name = action.arm, action.object
    cfg = _cfg(ctx)
    rep: Dict[str, Any] = {"chunks": []}
    ctx["push_close"] = rep
    _hook(ctx, "fov")(False)  # the hand leaves the camera frame at the side stances: the contact span is not a grasp span

    def q_now() -> float:
        return float(state.joint_position(name, action.joint))

    def leaf_stop(trunk0: float) -> Optional[str]:
        if q_now() - action.q_lower <= cfg.push_close_stop_rad:
            return f"leaf closed (q {math.degrees(q_now()):.2f} deg)"
        if motions.state.trunk_error() - trunk0 >= cfg.push_close_stop_trunk_rad:
            return f"trunk pressed {math.degrees(motions.state.trunk_error()):.2f} deg off its target (q {math.degrees(q_now()):.2f} deg)"
        return None

    obj = state.object(name)
    standoff = HANDLE_STANDOFF_M if action.handle_standoff_m is None else float(action.handle_standoff_m)
    d0, m0, sgn = leaf_frame(action.hinge_xy, action.handle_xy_closed, action.approach_closed, standoff)
    q0 = q_now()
    _w_aabb, thick = _leaf_dims(obj, action.hinge_xy, rot_z(d0, sgn * (q0 - action.q_closed)), rot_z(m0, sgn * (q0 - action.q_closed)), action.leaf_link)
    width = leaf_width(action.handle_radius_m) if action.leaf_width_m is None else float(action.leaf_width_m)
    face = (0.5 * thick if action.face_offset_m is None else float(action.face_offset_m)) + action.contact_clearance_m
    rep["geometry"] = {"hinge_xy": list(action.hinge_xy), "leaf_width_m": round(width, 3), "thickness_m": round(thick, 3), "q_start_deg": round(math.degrees(q0), 2)}
    if action.face_offset_m is not None or action.handle_standoff_m is not None:
        rep["geometry"].update({"face_offset_m": action.face_offset_m, "contact_clearance_m": action.contact_clearance_m, "handle_standoff_m": round(standoff, 4),
                                "swing_sign": sgn})
    hinge = th.tensor([action.hinge_xy[0], action.hinge_xy[1], 0.0])
    z_c = action.handle_z - action.push_height_drop_m
    robot = motions.robot
    idx = robot.arm_control_idx[arm]
    motions.backend.update_world()
    for chunk in range(action.max_chunks):
        q = q_now()
        if q - action.q_lower <= action.closed_tol_rad:
            break
        d1, m1 = rot_z(d0, sgn * (q - action.q_closed)), rot_z(m0, sgn * (q - action.q_closed))
        edge = hinge + width * d1
        p_contact = hinge + (width - action.push_inset_m) * d1 + face * m1 + th.tensor([0.0, 0.0, z_c])
        contact_pose = C.contact_eef(p_contact, -m1, closing=th.tensor([0.0, 0.0, 1.0]))  # a = -m: pushing toward the inside (closing)
        pre = C.pre_pose(contact_pose, action.push_pre_m)
        delta = -sgn * min(math.radians(action.chunk_deg), max(q - action.q_lower, 0.0))
        n_arc = max(int(round(abs(delta) / math.radians(2.0))), 4)  # scored at the execution's density (2° per waypoint)
        arc = C.arc_waypoints(contact_pose, hinge + th.tensor([0.0, 0.0, z_c]), th.tensor([0.0, 0.0, 1.0]), delta, n=n_arc)
        rec: Dict[str, Any] = {"q_before_deg": round(math.degrees(q), 2), "contact": [round(float(v), 3) for v in p_contact], "free_edge_xy": [round(float(v), 3) for v in edge[:2]],
                               "delta_deg": round(math.degrees(abs(delta)), 1)}
        # front stances (base on the +m side facing the contact) and side stances (beside the leaf's swing area, heading
        # along the wall, the arm extended laterally toward the outer face), scored before driving
        perp = th.tensor([-float(m1[1]), float(m1[0]), 0.0])
        cands: List[Tuple[str, th.Tensor, float]] = []
        for dist in (0.50, 0.60, 0.44):
            for lat in (0.0, -0.20, 0.20, -0.35):
                b = C.as_tensor(pre[0]) + dist * m1 + lat * perp
                aim = C.as_tensor(pre[0]) - b
                cands.append(("front", b, math.atan2(float(aim[1]), float(aim[0]))))
        for side_dir in (perp, -perp):
            for dist in (0.55, 0.65):
                for out in (0.10, 0.25):
                    b = C.as_tensor(pre[0]) + dist * side_dir + out * m1
                    cands.append(("side", b, math.atan2(float(-side_dir[1]), float(-side_dir[0]))))
        scored = []
        here = motions.base_pose2d()
        for kind_c, b, yaw in cands:
            base = (float(b[0]), float(b[1]), yaw)
            n_ok, _ = pushable_arc_from(motions, arm, base, pre, contact_pose, arc)
            scored.append((n_ok, -math.hypot(base[0] - here[0], base[1] - here[1]), base, kind_c))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        rec["stance_scores"] = [{"base": [round(v, 2) for v in t[2]], "arc_ok": f"{t[0]}/{len(arc)}", "kind": t[3]} for t in scored[:6]]
        best = scored[0]
        if best[0] < 0:
            rec["result"] = "no stance reaches the contact pose (backend IK)"
            rep["chunks"].append(rec)
            log.warning("push_close chunk %d: %s", chunk, rec["result"])
            break
        target = best[2]
        n_reach = max(best[0], 0)
        delta_exec = delta * (n_reach / len(arc)) if n_reach > 0 else delta * (1.0 / len(arc))
        rec["plan"] = {"stance_kind": best[3], "waypoints_reachable": f"{n_reach}/{len(arc)}", "predicted_landing_deg": round(math.degrees(q + delta_exec), 2)}
        log.info("push_close chunk %d: %s stance %s reaches the contact and %d/%d arc waypoints (best of %d); predicted landing %.1f deg", chunk, best[3],
                 [round(v, 2) for v in target], n_reach, len(arc), len(scored), math.degrees(q + delta_exec))
        yield from _arms_ready(motions, rec)
        try:
            yield from motions.track_path([target])
            yield from motions.settle(max_steps=20)
        except SkillError as e:
            rec["result"] = f"drive to the stance failed: {e.message or e.reason}"
            rep["chunks"].append(rec)
            break
        rec["stance"] = [round(v, 3) for v in target]
        motions.backend.update_world()
        yield from motions.gripper(arm, close=True)
        try:
            yield from motions.arm_to(arm, pose=pre, ignore_objects=[obj])
        except SkillError as e:
            rec["result"] = f"pre pose unreachable after the drive: {(e.message or e.reason)[:140]}"
            rep["chunks"].append(rec)
            break
        found, travelled, how = yield from guarded_approach(motions, arm, -m1, q_now)
        rec["approach"] = {"travelled_m": round(travelled, 3), "how": how}
        if found is not None and "blocked" in how:
            # contact evidence: a further 1 cm push must move the door, else the stall was the reach limit
            q_a = q_now()
            try:
                eef = motions.eef_pose(arm)
                yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) - 0.01 * m1, C.as_tensor(eef[1]))], loaded=True)
            except SkillError:
                pass
            yield from motions.hold(3)
            moved = abs(q_now() - q_a) >= math.radians(0.3)
            rec["approach"]["evidence"] = {"door_moved_deg": round(math.degrees(q_now() - q_a), 2), "contact": moved}
            if not moved:
                found = None
                how = "stall at the reach limit (door did not move under a 1 cm push)"
        if found is None:
            rec["result"] = f"no contact ({how})"
            rep["chunks"].append(rec)
            log.warning("push_close chunk %d: %s", chunk, rec["result"])
            break
        cpose = motions.eef_pose(arm)
        cpose = (C.as_tensor(cpose[0]), C.as_tensor(cpose[1]))
        n_exec = max(int(round(abs(delta_exec) / math.radians(2.0))), 1)
        arc2 = C.arc_waypoints(cpose, hinge + th.tensor([0.0, 0.0, float(cpose[0][2])]), th.tensor([0.0, 0.0, 1.0]), delta_exec, n=n_exec)
        why: List[Optional[str]] = [None]
        trunk0 = [motions.state.trunk_error()]

        def stop() -> bool:
            why[0] = leaf_stop(trunk0[0])
            return why[0] is not None

        try:
            stopped = yield from _until(motions.arm_along(arm, list(arc2), loaded=True, partial=True), stop)
            rec["arc"] = f"stopped: {why[0]}" if stopped else "complete"
        except SkillError as e:
            rec["arc"] = f"stopped: {e.message or e.reason}"
        if why[0] is not None:
            un = yield from motions.unload([arm])
            rec["arc_unload_rad"] = {k: round(v, 4) for k, v in un.items()}
        yield from motions.settle(max_steps=40)
        q_after = q_now()
        rec["q_after_deg"] = round(math.degrees(q_after), 2)
        rec["plan"]["actual_landing_deg"] = round(math.degrees(q_after), 2)
        # progress is q falling toward q_lower whatever the swing sign (sgn only turns the leaf in the world frame)
        if q - q_after < math.radians(2.0):
            # the arm's arc gained little: drive the base along -m with the fist on the face (the base does the pushing)
            motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
            bx, by, byaw = motions.base_pose2d()
            tgt = (bx - action.drag_m * float(m1[0]), by - action.drag_m * float(m1[1]), byaw)
            motions.follow_phase = "base"
            why[0] = None
            trunk0[0] = motions.state.trunk_error()
            try:
                stopped = yield from _until(motions.track_path([tgt], tracker_cfg=replace(motions.cfg.base_tracker, v_max=(0.06, 0.06, 0.3))), stop)
                if stopped:
                    bx2, by2, _ = motions.base_pose2d()
                    rec["base_drag"] = f"stopped after {math.hypot(bx2 - bx, by2 - by):.2f} m: {why[0]}"
                else:
                    rec["base_drag"] = f"{action.drag_m:.2f} m toward the frame"
            except SkillError as e:
                rec["base_drag"] = f"stopped: {(e.message or e.reason)[:120]}"
            finally:
                motions.follow_phase = "arm"
            if why[0] is not None:
                un = yield from motions.unload([arm])
                rec["drag_unload_rad"] = {k: round(v, 4) for k, v in un.items()}
            yield from motions.settle(max_steps=40)
            q_after = q_now()
            rec["q_after_drag_deg"] = round(math.degrees(q_after), 2)
        un = yield from motions.unload([arm])  # the fist leaves a leaf it may still press: nothing springs back when it goes
        if any(v > 0.0 for v in un.values()):
            rec["backoff_unload_rad"] = {k: round(v, 4) for k, v in un.items()}
        try:  # off the face (+m), then let it settle
            eef = motions.eef_pose(arm)
            yield from motions.arm_along(arm, [(C.as_tensor(eef[0]) + 0.10 * m1, C.as_tensor(eef[1]))], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
        except SkillError as e:
            rec["back_off_failed"] = e.message or e.reason
        yield from motions.settle(max_steps=40)
        rec["q_settled_deg"] = round(math.degrees(q_now()), 2)
        motions.state.arm_target[arm] = robot.get_joint_positions()[idx].clone().float()
        rep["chunks"].append(rec)
        log.info("push_close chunk %d: door %.1f -> %.1f deg (settled %.1f), %s", chunk, math.degrees(q), math.degrees(q_after), math.degrees(q_now()), rec.get("arc"))
        if q - q_now() < math.radians(0.5):
            rec["result"] = "no progress"
            break
    q_end = q_now()
    rep["final_q_deg"] = round(math.degrees(q_end), 2)
    rep["closed"] = bool(q_end - action.q_lower <= action.closed_tol_rad)
    rep["verdict"] = "closed by pushing" if rep["closed"] else f"stopped at {math.degrees(q_end):.1f} deg"
    mon = ctx.get("monitors")
    if mon is not None:
        rep["contacts"] = [dict(e) for e in getattr(mon, "collision_events", [])[-40:]]
    log.info("push_close done: %s (%d chunks); contacts %s", rep["verdict"], len(rep["chunks"]), rep.get("contacts"))


def post_push_close(action: PushClose, state, targets, ctx) -> Optional[str]:
    """Judged on the joint, as ``post_base_follow``: within the spec §8 tolerance of ``q_lower`` (revolute max(0.05 rad,
    0.10·|Δq|), Δq = ``tol_span_rad`` when recorded - a push finish is judged on the instance's span -, else the push's own),
    ``no_motion`` below 10 % of the push's span, else ``tolerance``."""
    q = state.joint_position(action.object, action.joint)
    ctx["q"] = q
    span = abs(action.q_from - action.q_lower)
    tol_span = abs(action.tol_span_rad) if action.tol_span_rad is not None else span
    closed = abs(q - action.q_lower)
    if closed <= max(0.05, 0.10 * tol_span):
        return None
    return "no_motion" if span - closed < 0.10 * span else "tolerance"


# ==========================================================================================
# base push (IORS nextto large-scale: a floor-standing object pushed along the floor by the base)
# ==========================================================================================

#: knobs of a base push; the sampler's record (``contacts.push`` extras of the same names) overrides each
BASE_PUSH_DEFAULTS: Dict[str, float] = {
    "push_speed_m_s": 0.08,  # base speed along the line (the room door's base drag: 0.06)
    "push_yaw_rate": 0.2,
    "moved_m": 0.003,  # the object counts as touched by the approaching hand once it moved this far along the push
    "evidence_m": 0.002,  # a blocked hand is a contact only when a further 1 cm push moves the object this far (the room door's rule)
    "approach_extra_m": 0.10,  # the guarded approach runs at most the pre-contact distance + this
    "lag_m": 0.04,  # the object lags the hand's progress by more than this: the drive stops for a re-contact
    "recontacts": 2,
    "drift_tol_m": 0.10,  # sideways drift of the object off the line that stops the drive
    "yaw_tol_rad": 0.61,  # turn of the object that stops the drive (35 deg; a single-hand push turns the cart ~20 deg by mid-push)
    "nextto_push_on_m": 0.05,  # once NextTo holds the drive goes on this far (the object well inside the bound), then stops
    "nextto_hold_checks": 2,  # NextTo must read true at this many consecutive checks
    "nextto_check_every": 5,  # steps between NextTo reads
    # the object turned this far since the push start / the last realign: the drive stops for a realign (yaw_tol_rad, on the same
    # turn, stops the drive only once the realigns are used)
    "realign_turn_rad": 0.26,
    "realigns": 3,
    "realign_clear_m": 0.10,  # the hand backs off this far before the realign's base move (it shifts ~9 cm along a face turned 15 deg, ~2.5 cm closer)
    "realign_margin_m": 0.01,  # whole-body clearance of the realign's base move (the sampler's push sweep margin)
    # two hands (item C): the last centimetres before the recorded contact and every step after the first hand touched are this short
    "hands_fine_before_m": 0.03,
    "hands_fine_step_m": 0.005,
}


def base_push_knobs(extra: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """:data:`BASE_PUSH_DEFAULTS` with the record's numeric overrides."""
    out = dict(BASE_PUSH_DEFAULTS)
    for k in out:
        v = (extra or {}).get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def push_frame_offset(p0, p, direction) -> Tuple[float, float]:
    """``(along, sideways)``: the planar displacement ``p0 -> p`` along the unit push ``direction`` and across it (+ = left of it)."""
    d = C.as_tensor(direction).reshape(-1)[:2].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    v = C.as_tensor(p).reshape(-1)[:2].float() - C.as_tensor(p0).reshape(-1)[:2].float()
    return float(v[0] * d[0] + v[1] * d[1]), float(-v[0] * d[1] + v[1] * d[0])


def nextto_now(state, a: str, b: str) -> Optional[bool]:
    """OmniGibson's ``NextTo(a, b)`` - the state the task's ``(nextto ?a ?b)`` goal reads - now; ``state.nextto(a, b)`` when the
    state provides one (the test stubs). None when it cannot be read."""
    fn = getattr(state, "nextto", None)
    try:
        if callable(fn):
            return bool(fn(a, b))
        from omnigibson.object_states import NextTo

        oa, ob = state.object(a), state.object(b)
        if NextTo not in getattr(oa, "states", {}):
            return None
        return bool(oa.states[NextTo].get_value(ob))
    except Exception as e:  # noqa: BLE001 - a read failure is no verdict
        log.debug("NextTo(%s, %s) unreadable (%r)", a, b, e)
        return None


def realign_line(pose0, pose, contact_local, direction, distance_m: float) -> Optional[Dict[str, Any]]:
    """Where a base push's realign puts the hand (owner 2026-10-12 ~20:17, nextto_large): the object was pushed from ``pose0`` along the
    unit ``direction`` and stands at ``pose`` now; ``contact_local`` is the record's contact point in its root frame (``contact_in_object``)
    on the push face, whose outward normal was -``direction`` at ``pose0``, and on the face's centre line (the sampler's push face: the
    line across the face through the footprint centre). The footprint centre is the root's foot on that line. The push line runs from
    the centre now to the centre's goal (its place at ``pose0`` moved ``distance_m`` along ``direction``); the hand goes where that line
    enters the face at the object's pose now, at the contact's height. ``{"point", "direction" (unit, z 0), "remaining_m", "line_rad"
    (the line's angle from ``direction``), "offset_m" (the point along the face from the recorded contact, + = left of the push),
    "centre", "goal"}``; None when less than 2 cm remain or the line meets the face more than 60 deg off its normal (or from the front)."""
    d = C.as_tensor(direction).reshape(-1)[:2].float()
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    p0, q0 = (C.as_tensor(v).reshape(-1).float() for v in pose0)
    p, q = (C.as_tensor(v).reshape(-1).float() for v in pose)
    loc = C.as_tensor(contact_local).reshape(-1)[:3].float()
    turn = C.wrap_angle(C.yaw_of(q) - C.yaw_of(q0))
    n0 = -d
    n = th.tensor([math.cos(turn) * float(n0[0]) - math.sin(turn) * float(n0[1]), math.sin(turn) * float(n0[0]) + math.cos(turn) * float(n0[1])])
    c0, c1 = p0[:3] + C.quat_rotate(q0, loc), p[:3] + C.quat_rotate(q, loc)
    h = float((c1[:2] - p[:2]) @ n)  # the face ahead of the root along its outward normal
    centre = c1[:2] - n * h
    goal = c0[:2] - n0 * float((c0[:2] - p0[:2]) @ n0) + float(distance_m) * d
    remaining = float(th.linalg.norm(goal - centre))
    if remaining < 0.02 or h <= 0.0:
        return None
    u = (goal - centre) / remaining
    cos_in = -float(u @ n)
    if cos_in < 0.5:
        return None
    pt = centre - (h / cos_in) * u
    left = th.tensor([float(n[1]), -float(n[0])])  # left of the push into the face
    return {"point": th.tensor([float(pt[0]), float(pt[1]), float(c1[2])]), "direction": th.tensor([float(u[0]), float(u[1]), 0.0]), "remaining_m": remaining,
            "line_rad": C.wrap_angle(math.atan2(float(u[1]), float(u[0])) - math.atan2(float(d[1]), float(d[0]))), "offset_m": float((pt - c1[:2]) @ left),
            "centre": centre, "goal": goal}


def realign_base_clear(motions: Motions, b_from, b_to, margin: float) -> Optional[str]:
    """Why a realign's base move ``b_from -> b_to`` ([x, y, yaw]) is not clear, None when it is: the whole body with the arms and the
    trunk as they are at every 5 cm / 0.1 rad of the straight leg (:func:`_base_legs`), cuRobo's check against the synchronised world
    (the pushed object in it) with ``margin``. A backend that cannot check does not clear the move."""
    try:
        q = C.as_tensor(motions.backend.q_full()).reshape(-1).float().clone()
        qs = _base_legs(motions, q, b_from, b_to)
        hits = C.as_tensor(motions.backend.check_collision(th.stack(qs), sync_world=True, margin=float(margin))).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return f"no collision check of the base move ({e!r})"[:200]
    if bool(hits.any()):
        return f"the base move collides at configuration {int(th.nonzero(hits)[0]) + 1} of {len(qs)} (margin {margin:.2f} m)"
    return None


def execute_base_push(action: BasePush, motions: Motions, state, targets, ctx) -> Iterator[th.Tensor]:
    """The closed hand on the object's push face, then the base drives the object along the recorded line (owner 2026-10-12,
    nextto_large): the pre-contact (planned to the sampler's witness, then with the object present, then ignored), a guarded
    approach along the push direction until the object moves or the hand is blocked (a blocked hand counts only when a further 1 cm
    push moves the object), the arm held at its joints while the base drives to the recorded end of the line - stopped once NextTo
    holds and the object went ``nextto_push_on_m`` further, when it drifts sideways beyond ``drift_tol_m`` or turns beyond
    ``yaw_tol_rad``, re-contacted (unload, 2 cm back, guarded approach) when it lags the hand by ``lag_m`` - the arm unloaded at the
    end. A turn of ``realign_turn_rad`` since the push start or the last realign realigns the push (owner 2026-10-12 ~20:17): unload, the
    hand ``realign_clear_m`` back, its new contact where the line from the object's footprint centre to its goal enters the push face
    (:func:`realign_line`), the base moved - collision-checked (:func:`realign_base_clear`) - to stand to that contact as the record
    stands to its contact, the hand to the pre-contact there, the guarded approach, then the drive along that line for the remaining
    distance; at most ``realigns`` times, the ``yaw_tol_rad`` stop on the same turn only after them (none without the record's
    ``contact_in_object``). ``ctx["base_push"]`` records the pushed distance, the sideways drift, NextTo at the end, why the drive
    stopped and each realign.

    Two hands (``BasePush.second``, owner 2026-10-12 ~20:17 item C): both grippers closed, both hands to their pre-contacts (the first
    hand first, each planned to its own witness ``pre`` / ``second.pre``; a second hand that cannot get there leaves the push to the
    first), the guarded approach of :func:`guarded_approach_hands` until both touch (the evidence push per blocked hand), both arms held
    through the drive, both unloaded, backed off and approached again on a lag or a realign, both unloaded at the end. The realign's line
    enters the face at the record's ``face_centre_in_object`` (between the spots) and the stance is carried onto it from the midpoint of
    the two recorded contacts, so both hands keep their places relative to the base."""
    c = action.contact
    arm, obj = c.arm, c.object
    knobs = base_push_knobs(c.extra)
    start, end = action.base_line
    d3 = th.tensor([float(c.direction[0]), float(c.direction[1]), 0.0])
    d3 = d3 / max(float(th.linalg.norm(d3)), 1e-9)
    rep: Dict[str, Any] = {"distance_m": round(float(c.distance_m), 4), "drift_tol_m": knobs["drift_tol_m"], "partner": action.partner, "recontacts": 0, "realigns": 0}
    ctx["base_push"] = rep
    _hook(ctx, "fov")(False)
    for h in action.contacts:
        yield from motions.gripper(h.arm, close=True)
    try:
        target_obj = [state.object(obj)]
    except Exception:  # noqa: BLE001
        target_obj = []
    pose0 = tuple(x.clone() for x in state.object_pose(obj))
    ctx["object_pose0"] = pose0
    loc = c.extra.get("contact_in_object")
    n_realign = int(knobs["realigns"]) if isinstance(loc, (list, tuple)) and len(loc) == 3 else 0
    # the line the hand pushes along (the recorded pre-contact -> contact line is horizontal along the push, whatever the hand's pitch)
    # and the base's end on it; a realign replaces both
    leg: Dict[str, Any] = {"a": d3, "end": end}

    def along() -> float:
        return push_frame_offset(pose0[0], state.object_pose(obj)[0], leg["a"])[0]

    def to_pre(hand: str, pose, key: str) -> Generator[th.Tensor, None, Tuple[Optional[str], List[str]]]:
        tries: List[Tuple[str, Dict[str, Any]]] = []
        if _witness(ctx, key) is not None:
            tries.append(("the recorded witness", {"q_hint": _witness(ctx, key), "accept_other_branch": True}))
        tries += [("the object present", {"accept_other_branch": True}), ("the object ignored", {"ignore_objects": target_obj, "accept_other_branch": True})]
        fails: List[str] = []
        with _transit(ctx, motions, state, action, "pre_contact", target=obj):
            for how, kw in tries:
                try:
                    yield from motions.arm_to(hand, pose=pose, **kw)
                    return how, fails
                except SkillError as e:
                    fails.append(f"{how}: {(e.message or e.reason)[:140]}")
        return None, fails

    rep["pre"], fails = yield from to_pre(arm, targets["pre"], "pre")
    if fails:
        rep["pre_failures"] = fails
    if rep["pre"] is None:
        rep["result"] = "pre-contact unreachable"
        log.warning("base_push(%s): %s (%s)", obj, rep["result"], rep.get("pre_failures"))
        return
    arms = [arm]
    sec = targets.get("second") if action.second is not None else None
    if sec is not None:
        rep["second_pre"], fails = yield from to_pre(sec["arm"], sec["pre"], "second.pre")
        if fails:
            rep["second_pre_failures"] = fails
        if rep["second_pre"] is None:
            rep["second_hand"] = f"pre-contact unreachable: the {arm} hand pushes alone"
            log.warning("base_push(%s): the %s hand's %s (%s)", obj, sec["arm"], rep["second_hand"], fails)
        else:
            arms.append(str(sec["arm"]))
        rep["hands"] = list(arms)
    _hook(ctx, "fov")(True)
    pre_dist = float(th.linalg.norm(as_world_pose(targets["contact"])[0] - as_world_pose(targets["pre"])[0]))
    # where each hand meets the face, in the object's frame (the record's contact_in_object; a realign moves them with the hands)
    spots = {h.arm: C.as_tensor(h.extra["contact_in_object"]).reshape(-1)[:3].float() for h in action.contacts
             if isinstance(h.extra.get("contact_in_object"), (list, tuple)) and len(h.extra["contact_in_object"]) == 3}

    def spot_along(hand: str) -> float:
        """Where the object's face is at ``hand``'s spot, along the push (its root without the record's ``contact_in_object``)."""
        p, q = (C.as_tensor(v).reshape(-1).float() for v in state.object_pose(obj))
        w = p[:3] + (C.quat_rotate(q, spots[hand]) if hand in spots else 0.0)
        return float(w[:2] @ leg["a"][:2])

    def evidence(hand: str, moved_of: Callable[[], float], rec: Dict[str, Any]) -> Generator[th.Tensor, None, bool]:
        """A blocked hand is a contact only when a further 1 cm push moves the object (the room door's rule)."""
        s0 = moved_of()
        try:
            eef = motions.eef_pose(hand)
            yield from motions.arm_along(hand, [(C.as_tensor(eef[0]) + 0.01 * leg["a"], C.as_tensor(eef[1]))], loaded=True)
        except SkillError:
            pass
        yield from motions.hold(3)
        moved = moved_of() - s0 >= knobs["evidence_m"]
        rec["evidence"] = {"object_moved_m": round(moved_of() - s0, 4), "contact": moved}
        if not moved:
            rec["how"] = "stall at the reach limit (the object did not move under a 1 cm push)"
        return moved

    def contact(label: str, back_m: Optional[float] = None) -> Generator[th.Tensor, None, bool]:
        """The guarded approach from ``back_m`` behind the face (default: the pre-contact's distance) and the evidence of a blocked hand."""
        if len(arms) > 1:
            return (yield from contact_hands(label, pre_dist if back_m is None else back_m))
        found, travelled, how = yield from guarded_approach(motions, arm, leg["a"], along, max_m=pre_dist + knobs["approach_extra_m"], move_rad=knobs["moved_m"])
        rec: Dict[str, Any] = {"travelled_m": round(travelled, 3), "how": "object moved" if how == "leaf moved" else how}
        if found is not None and how != "leaf moved" and not (yield from evidence(arm, along, rec)):
            found = None
        rep.setdefault("approaches", []).append({"leg": label, **rec})
        return found is not None

    def contact_hands(label: str, back_m: float) -> Generator[th.Tensor, None, bool]:
        res = yield from guarded_approach_hands(motions, arms, leg["a"], {h: (lambda h=h: spot_along(h)) for h in arms}, fine_m=knobs["hands_fine_step_m"],
                                                fine_from_m=max(0.0, back_m - knobs["hands_fine_before_m"]), max_m=pre_dist + knobs["approach_extra_m"],
                                                move_m=knobs["moved_m"])
        recs: Dict[str, Dict[str, Any]] = {}
        for h in arms:
            found, travelled, how = res[h]
            recs[h] = {"travelled_m": round(travelled, 3), "how": how}
            if found is not None and how != "object moved" and not (yield from evidence(h, lambda h=h: spot_along(h), recs[h])):
                found = None
            recs[h]["contact"] = found is not None
        touching = [h for h in arms if recs[h]["contact"]]
        if len(touching) == len(arms):
            how = "both hands touch"
        elif touching:
            how = "; ".join([f"the {touching[0]} hand touches"] + [f"the {h} hand: {recs[h]['how']}" for h in arms if h not in touching])
        else:
            how = "; ".join(f"{h}: {recs[h]['how']}" for h in arms)
        rep.setdefault("approaches", []).append({"leg": label, "how": how, "hands": recs})
        return bool(touching)

    if not (yield from contact("approach")):
        rep["result"] = f"no contact ({rep['approaches'][-1]['how']})"
        log.warning("base_push(%s): %s", obj, rep["result"])
        _hook(ctx, "fov")(False)
        return
    tcfg = replace(motions.cfg.base_tracker, v_max=(knobs["push_speed_m_s"], knobs["push_speed_m_s"], knobs["push_yaw_rate"]))
    idx = {h: motions.robot.arm_control_idx[h] for h in arms}
    yaw0 = C.yaw_of(C.as_tensor(pose0[1]).float())
    leg["yaw"] = yaw0  # the object's yaw its turn is measured from: the push start, then each realign's re-contact

    def realign(k: int) -> Generator[th.Tensor, None, Optional[str]]:
        """Realign ``k`` (the docstring above); None once the hands push again on the new line, else why not."""
        q_was = C.as_tensor(state.object_pose(obj)[1]).float()
        rec: Dict[str, Any] = {"realign": k, "turn_deg": round(math.degrees(abs(C.wrap_angle(C.yaw_of(q_was) - yaw0))), 2),
                               "leg_turn_deg": round(math.degrees(abs(C.wrap_angle(C.yaw_of(q_was) - leg["yaw"]))), 2)}
        rep.setdefault("realign_moves", []).append(rec)
        yield from motions.unload(arms)
        for h in arms:
            try:
                eef = motions.eef_pose(h)
                yield from motions.arm_along(h, [(C.as_tensor(eef[0]) - knobs["realign_clear_m"] * leg["a"], C.as_tensor(eef[1]))],
                                             soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
            except SkillError as e:
                rec["back_off_failed" if h == arm else f"{h}_back_off_failed"] = (e.message or e.reason)[:120]
        yield from motions.settle(max_steps=30)
        p, q = state.object_pose(obj)
        rec["object_pose2d"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(C.yaw_of(C.as_tensor(q).float()), 4)]
        # two hands: the line enters the face between the spots, and the stance is carried from between the two recorded contacts
        centre = c.extra.get("face_centre_in_object")
        two = len(arms) > 1 and isinstance(centre, (list, tuple)) and len(centre) == 3
        g = realign_line(pose0, (p, q), centre if two else loc, d3, float(c.distance_m))
        if g is None:
            return "no push line into the face (under 2 cm to the goal, or the face more than 60 deg off the line)"
        hands = {arm: (targets["contact"], targets["pre"])}
        if len(arms) > 1:
            hands[arms[1]] = (sec["contact"], sec["pre"])
        tips = {h: as_world_pose(cp)[0] + C.FINGERTIP_LENGTH * C.approach_axis(as_world_pose(cp)[1]) for h, (cp, _) in hands.items()}  # on the face
        rot = C.quat_from_yaw(g["line_rad"])
        p_rec = 0.5 * (tips[arms[0]] + tips[arms[1]]) if two else tips[arm]

        def moved(v) -> th.Tensor:  # a point of the recorded push stance carried with its contact: p_rec onto the new point, turned about +z
            return g["point"] + C.quat_rotate(rot, C.as_tensor(v).reshape(-1)[:3].float() - p_rec)

        b_xy = moved([start[0], start[1], float(p_rec[2])])
        b_from, b_to = motions.base_pose2d(), (float(b_xy[0]), float(b_xy[1]), C.wrap_angle(start[2] + g["line_rad"]))
        u = g["direction"]
        end_new = (b_to[0] + g["remaining_m"] * float(u[0]), b_to[1] + g["remaining_m"] * float(u[1]), b_to[2])
        pres = {h: (moved(as_world_pose(pp)[0]), C.qmul(rot, as_world_pose(pp)[1])) for h, (_, pp) in hands.items()}
        pre = pres[arm]
        move_m, move_deg = _base_error(b_from, b_to)[0], math.degrees(C.wrap_angle(b_to[2] - b_from[2]))
        rec.update({"point": [round(float(v), 4) for v in g["point"]], "line_deg": round(math.degrees(g["line_rad"]), 2), "hand_offset_m": round(g["offset_m"], 4),
                    "remaining_m": round(g["remaining_m"], 4), "pre": [round(float(v), 4) for v in pre[0]],
                    "base_move": {"from": [round(float(v), 4) for v in b_from], "to": [round(v, 4) for v in b_to], "m": round(move_m, 4), "deg": round(move_deg, 2)}})
        if len(arms) > 1:
            rec["pres"] = {h: [round(float(v), 4) for v in pp[0]] for h, pp in pres.items()}
        blocked = realign_base_clear(motions, b_from, b_to, knobs["realign_margin_m"])
        if blocked is not None:
            return blocked
        motions.follow_phase = "base"
        try:
            yield from motions.track_path([b_to], tracker_cfg=tcfg)
        except SkillError as e:
            return f"the base move: {(e.message or e.reason)[:140]}"
        finally:
            motions.follow_phase = "arm"
        for h in arms:
            try:
                yield from motions.arm_along(h, [pres[h]], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
            except SkillError as e:
                rec["pre_failed" if h == arm else f"{h}_pre_failed"] = (e.message or e.reason)[:120]
        q_inv = C.qinv(C.as_tensor(q).reshape(-1).float())
        for h in arms:  # each hand now meets the face where its recorded contact was carried to
            spots[h] = C.quat_rotate(q_inv, moved(tips[h]) - C.as_tensor(p).reshape(-1)[:3].float())
        leg["a"] = u
        if not (yield from contact(f"realign {k}")):
            return f"no contact ({rep['approaches'][-1]['how']})"
        leg.update({"yaw": C.yaw_of(C.as_tensor(state.object_pose(obj)[1]).float()), "end": end_new})
        rec.update({"turn_after_deg": round(math.degrees(abs(C.wrap_angle(leg["yaw"] - yaw0))), 2), "result": "pushing on the new line"})
        log.info("base_push(%s): realign %d/%d: turned %.1f deg on the line (%.1f deg since the push start) -> %.1f deg after the re-contact; the hand %+.3f m "
                 "along the face from the recorded contact, the line %+.1f deg off the recorded one; base move %.3f m / %+.1f deg; %.3f m to the goal",
                 obj, k, n_realign, rec["leg_turn_deg"], rec["turn_deg"], rec["turn_after_deg"], g["offset_m"], math.degrees(g["line_rad"]), move_m, move_deg,
                 g["remaining_m"])
        return None

    why: List[Optional[str]] = [None]
    nt = {"first_along": None, "streak": 0, "steps": 0}
    stop_reason = None
    while True:
        q_now = motions.robot.get_joint_positions()
        for h in arms:
            motions.state.arm_target[h] = q_now[idx[h]].clone().float()  # the arms held fixed relative to the base
        b0 = motions.base_pose2d()
        s_obj0 = along()

        def stop() -> bool:
            p, q = state.object_pose(obj)
            s, side = push_frame_offset(pose0[0], p, d3)
            turn = abs(C.wrap_angle(C.yaw_of(C.as_tensor(q).float()) - leg["yaw"]))
            spent = rep["realigns"] >= n_realign
            since = f" since realign {rep['realigns']}" if rep["realigns"] else ""
            if abs(side) > knobs["drift_tol_m"] or (spent and turn > knobs["yaw_tol_rad"]):
                why[0] = (f"drift: {side:+.3f} m sideways, {math.degrees(turn):.1f} deg turned{since} (tolerance {knobs['drift_tol_m']:.2f} m / "
                          f"{math.degrees(knobs['yaw_tol_rad']):.0f} deg)")
                return True
            if not spent and turn > knobs["realign_turn_rad"]:
                why[0] = f"turn: {math.degrees(turn):.1f} deg{since} (realign at {math.degrees(knobs['realign_turn_rad']):.0f} deg)"
                return True
            base_s = push_frame_offset(b0[:2], motions.base_pose2d()[:2], leg["a"])[0]
            s_obj = push_frame_offset(pose0[0], p, leg["a"])[0] - s_obj0
            if base_s - s_obj > knobs["lag_m"]:
                why[0] = f"lag: the base went {base_s:.3f} m, the object {s_obj:.3f} m"
                return True
            if action.partner:
                nt["steps"] += 1
                if nt["steps"] % max(1, int(knobs["nextto_check_every"])) == 0:
                    v = nextto_now(state, obj, action.partner)
                    nt["streak"] = nt["streak"] + 1 if v else 0
                    if nt["streak"] >= int(knobs["nextto_hold_checks"]) and nt["first_along"] is None:
                        nt["first_along"] = s
                        rep["nextto_first_at_m"] = round(s, 4)
                if nt["first_along"] is not None and s >= nt["first_along"] + knobs["nextto_push_on_m"]:
                    why[0] = f"nextto: held since {nt['first_along']:.3f} m, pushed on to {s:.3f} m"
                    return True
            return False

        why[0] = None
        motions.follow_phase = "base"
        try:
            stopped = yield from _until(motions.track_path([leg["end"]], tracker_cfg=tcfg), stop)
        except SkillError as e:
            stopped, why[0] = True, f"drive: {(e.message or e.reason)[:160]}"
        finally:
            motions.follow_phase = "arm"
        if not stopped:
            stop_reason = "distance done"
            break
        if why[0] is not None and why[0].startswith("turn"):
            rep["realigns"] += 1
            failed = yield from realign(rep["realigns"])
            if failed is None:
                continue
            rep["realign_moves"][-1]["result"] = f"failed: {failed}"
            log.warning("base_push(%s): %s -> realign %d/%d failed: %s", obj, why[0], rep["realigns"], n_realign, failed)
            stop_reason = f"{why[0]}; realign {rep['realigns']} failed: {failed}"
            break
        if why[0] is not None and why[0].startswith("lag") and rep["recontacts"] < int(knobs["recontacts"]):
            rep["recontacts"] += 1
            rep.setdefault("lags", []).append(why[0])
            log.info("base_push(%s): %s -> re-contact %d", obj, why[0], rep["recontacts"])
            yield from motions.unload(arms)
            for h in arms:
                try:
                    eef = motions.eef_pose(h)
                    yield from motions.arm_along(h, [(C.as_tensor(eef[0]) - 0.02 * leg["a"], C.as_tensor(eef[1]))], soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free)
                except SkillError:
                    pass
            yield from contact(f"re-contact {rep['recontacts']}", back_m=0.02)
            continue
        stop_reason = why[0]
        break
    un = yield from motions.unload(arms)  # nothing springs back when the hands leave
    if any(v > 0.0 for v in un.values()):
        rep["unload_rad"] = {k: round(v, 4) for k, v in un.items()}
    yield from motions.settle(max_steps=30)
    p, q = state.object_pose(obj)
    s, side = push_frame_offset(pose0[0], p, d3)
    bx, by, _ = motions.base_pose2d()
    rep.update({"stop": stop_reason, "pushed_m": round(s, 4), "drift_m": round(side, 4), "turn_deg": round(math.degrees(abs(C.wrap_angle(C.yaw_of(C.as_tensor(q).float()) - yaw0))), 2),
                "base_travel_m": round(push_frame_offset(start[:2], (bx, by), d3)[0], 4)})
    if action.partner:
        rep["nextto_end"] = nextto_now(state, obj, action.partner)
    rep["result"] = "pushed"
    _hook(ctx, "fov")(False)
    log.info("base_push(%s): pushed %.3f m of %.3f m, sideways drift %+.3f m (tolerance %.2f), turned %.1f deg, NextTo(%s) at the end: %s; stop: %s; %d re-contacts, "
             "%d realigns%s", obj, s, float(c.distance_m), side, knobs["drift_tol_m"], rep["turn_deg"], action.partner, rep.get("nextto_end"), stop_reason, rep["recontacts"],
             rep["realigns"], "" if action.second is None else f"; hands {' + '.join(arms)}")


def post_base_push(action: BasePush, state, targets, ctx) -> Optional[str]:
    """The pushed object moved (``moved_pos`` / ``moved_rot``), else ``no_motion``; whether it ended NextTo its partner is the
    epilogue's judgement (``ctx["base_push"]`` records it)."""
    cfg = _cfg(ctx)
    before = ctx.get("object_pose0")
    if before is None:
        return "no_motion"
    dp, da = _pose_error(state.object_pose(action.contact.object), before)
    ctx["object_motion"] = (dp, da)
    return None if dp > cfg.moved_pos or da > cfg.moved_rot else "no_motion"


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
    PushClose: (execute_push_close, post_push_close),
    BasePush: (execute_base_push, post_base_push),
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
    "push_geometry", "push_residual", "extend_push", "push_tangent", "PushGeometry", "push_stop_facts", "BASE_PUSH_DEFAULTS", "base_push_knobs",
    "push_frame_offset", "nextto_now", "realign_line", "realign_base_clear",
]
