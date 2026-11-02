"""Layer 1 of the v2 oracle: decision-free motions (``oracle_primitives_design.md`` §3.2, §3.4; TODO F2).

Eight motions, each a generator of 23-D JoyLo actions (one per environment step) that receives **one fully specified
target** and either reaches it or raises :class:`SkillError` with a QA category of §6.4:

==================================  ==========================================================================
``track_path(waypoints, style)``    the v1 base tracker follows a recorded ``[x, y, yaw]`` waypoint list (as
                                    recorded, or as the shared ``robot.base_path`` ``face_travel`` schedule)
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
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.action import ActionState
from ispatialgym.oracle.base_tracker import BaseTrackerConfig, BaseTrajectoryTracker
from ispatialgym.oracle.trunk import TrunkState
from ispatialgym.robot import base_path as BP
from ispatialgym.robot import conventions as C

log = logging.getLogger(__name__)


# train85 (owner 2026-10-14 ~22:00): the oracle's waits end once nothing moves any more, so a demo carries no long standstill
# (thresholds of reconcile83's generation.oracle_still_exit). ISPATIALGYM_ORACLE_STILL_EXIT=1 turns it on; unset, nothing changes.
STILL_EXIT_MOVE = 1e-4  # rad or m per control step: did not move
STILL_EXIT_STEPS = 5  # consecutive still steps that end a settle / a gripper wait
STILL_EXIT_OBJECT_SPEED = 0.005  # m/s: the object released last must be slower than this too (settle)
STILL_EXIT_CONVERGE_WINDOW = 10  # steps: _wait_converged stops when its error dropped less than STILL_EXIT_MOVE over this many


def oracle_still_exit() -> bool:
    return os.environ.get("ISPATIALGYM_ORACLE_STILL_EXIT") == "1"

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
    #: ``arm_to`` to an IK witness of a *pre-pose*: a residual to the recorded pose up to these is closed best-effort (a
    #: witness on a joint limit cannot move the last millimetres along a straight segment; the hand stays at the
    #: sampler-validated configuration and the following leg starts from where the hand is - the touching pre-grasp
    #: witness stood 10.5 mm off its pose), a larger one and every finishing motion must close
    witness_residual_accept_pos: float = 0.02
    witness_residual_accept_rot: float = 0.05
    #: a finishing motion (the grasp / place pose itself) tolerates a smaller unclosed residual: the touching grasp
    #: witnesses stand 11 mm off their poses (the sampler's frame) and the 20:24 demo grasped the box from exactly
    #: that configuration; the assisted-grasp check / the placed-pose tolerance judge the outcome
    witness_residual_accept_pos_finish: float = 0.015
    #: the witness is refined by the collision-aware IK of the recorded pose seeded at it when that solution stays
    #: within this joint distance (the same branch); the sampler's witnesses miss their poses by 3-15 mm
    witness_refine_max_jump: float = 0.6
    #: owner 2026-09-25 (teacup pod demo 0925-172723): rounds of ``plan_attempts`` joint-space attempts to a recorded witness after the
    #: first round fails (trajopt is stochastic), before the pose-goal fallback; 4 since the cap 0925-204950 grasp: all 3 rounds failed,
    #: the pose-goal fallback landed 2.73 rad away (refused) and the executor's next attempt planned the same leg in one round
    witness_plan_retries: int = 4
    #: the pose-goal fallback is accepted only within this joint distance of the witness; further is another IK branch, and the
    #: chained legs that follow (tracked from the witness's branch) would fail - the attempt fails instead and the executor retries
    witness_branch_max_jump: float = 0.6
    #: first-demo v2.2 (kielbasa 0928-025936 / 050633 / 050708, cap 0927-221834): when every joint-space round to a witness fails
    #: and the pose-goal fallback lands in another IK branch (where ``plan_failed`` was raised), the witness is planned to through
    #: these arm postures in turn (start -> posture -> witness, two joint-space plans) before the attempt fails
    witness_via_postures: Tuple[str, ...] = ("ready", "untucked")
    #: the protected-object transit remedy (owner 2026-09-26, ``Motions.protecting``): the hand first rises this much where it
    #: is, the plan goes to the point this much above the transit's goal, then straight down; direct re-plans when that fails
    protect_lift_m: float = 0.12
    protect_above_m: float = 0.12
    protect_replans: int = 3
    #: a pre-grasp transit plan that clears the protected objects only the long way round - its hand path longer than this
    #: times the via-above's nominal one (``protect_lift_m`` + the straight distance + ``protect_above_m``) and its hand higher
    #: than ``protect_lift_m`` above both ends - takes the via-above too (kielbasa 0926-074242: the direct plan to the witness
    #: failed, the graph-seeded retry arched 1.27 m / 19 cm high for a 0.53 m move; the via-above on the same instance: 0.90 m / 12 cm)
    protect_detour_ratio: float = 1.4
    #: owner 2026-09-30 (first-demo v3.0 F5, nextto marker 0928-050212: the plan to the pre-grasp lowered the hand from 1.013 to 0.837 m
    #: beside the robot and rose to the pre-grasp at 0.926 m - 8.9 cm below its lower end - before the approach descended again): a
    #: planned grasp transit (``transit_dip_gate``) whose hand path dips more than this below both of its ends runs as a Cartesian leg
    #: instead - straight to its goal, else up ``transit_dip_lift_m`` above the higher end, across, down - when that leg tracks with
    #: collisions and the protected objects checked, ends in the plan's IK branch and the approach tracks from it. The v2.2 grasp
    #: transits dipped 0-5.6 cm (free-object pinches; milk 4.95, cork 5.6), the marker 9.0, the handle grasps of the door / dishwasher
    #: 13-25 cm.
    transit_dip_max_m: float = 0.07
    transit_dip_lift_m: float = 0.06
    #: a Cartesian leg tracked back from the plan's end reaches the hand's pose in other joints: an in-place adjustment of at most this
    #: (rad, largest arm joint; the marker: 0.32 rad of joints 1 / 3 at the idle arm's shoulder singularity) moving the hand at most
    #: ``transit_dip_adjust_hand_m`` precedes it, at ``transit_dip_adjust_rate`` (rad/s)
    transit_dip_adjust_max_rad: float = 0.6
    transit_dip_adjust_hand_m: float = 0.02
    transit_dip_adjust_rate: float = 0.5
    #: a leg that starts with the eef link this far from the FK of the reported joints (the arm wound up under a
    #: contact load) first re-targets the joints onto the link's pose (``arm_along`` unwind)
    unwind_min_deflection: float = 0.005
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
    #: (rush 2026-09-17: 1 cm - the touching runs 0917-2030xx..2051xx stall at 5-7 mm on descents / pushes with no joint
    #: at a limit and no singularity; the eef postcondition is 2 cm and push outcomes are measured on the object)
    diff_ik_soft_pos_tol: float = 1e-2
    diff_ik_soft_rot_tol: float = 3e-2
    #: soft residual accepted on free-space legs (lift / retreat: the eef postcondition is 2 cm)
    diff_ik_soft_pos_tol_free: float = 1e-2
    #: joint speed bound of a streamed differential-IK path (rad/s; a nullspace escape near a singularity moves the
    #: joints far for a small eef step)
    diff_ik_joint_speed: float = 1.0
    #: gripper actuation wait (steps)
    gripper_steps: int = 60
    #: grasp closure ramp (owner decision 2026-09-16 17:10 item 4, closure probe ``runlogs/closure_probe_20260916``): the
    #: smooth-mode gripper command is ramped so the finger target moves at most this far per control step (m) from the
    #: current opening to closed. The stock step command drives both fingers at the joint's 0.25 m/s limit; on a tilted
    #: tabletop grasp the lower pad rubs the table and arrives 3 steps after the other pad, which knocks the un-pinched
    #: cylinder along the table and rolls it like a wheel (marker: 19-77 deg, demo 0916-152503: 24.6 deg). At 2.25 mm/step
    #: (a quarter of the free closing speed) both pads track the target and a centred object is pinched before it rolls:
    #: 0.1 / 2.9 / -3.2 deg on the marker (4.5 mm/step: -0.8 / -1.8 / 13.1 deg). None = the step command.
    gripper_close_rate: Optional[float] = 0.00225
    #: pre-shaped opening before a grasp descent: fingers at pinch width / 2 + this margin (m) so the lower pad of a
    #: tilted grasp clears the table (the fully open pad stood on it); None = fully open
    gripper_preshape_margin: Optional[float] = 0.010
    #: hold steps of the pre-shaped opening before the descent
    gripper_preshape_steps: int = 12
    #: symmetric closure (core 2026-09-16 22:09, closure probe): once both pads touch the object the ramp stops and the
    #: command holds the fingers where they are minus this squeeze (m). The two position drives (stiffness 1e7, max
    #: effort 100 N: any target inside the object saturates them) otherwise keep driving to 0 through the object and
    #: PhysX lets one of them win - finger 2 shoved the un-attached marker 8.5 mm and the centred napkins box 15.4 mm
    #: toward finger 1 (0916-202403) before the assisted grasp attached. With the targets frozen at the contact width the
    #: drift is bounded by the squeeze. None = the stock command (the fight).
    gripper_contact_squeeze_m: Optional[float] = 0.001
    #: a thin pinch (a library pinch at most this wide, m: a tray's wall) closes at ``thin_pinch_close_rate`` and holds the pads at the
    #: touch width less ``thin_pinch_squeeze_m``: at the ramp and squeeze above the pads strike a 3.2 mm wall at ~0.27 m/s and the
    #: light tray rocks on it (+-17 mm) until the assisted grasp's window of uninterrupted contact completes. None = every pinch as above
    thin_pinch_width_m: Optional[float] = 0.006
    thin_pinch_close_rate: float = 0.0005
    thin_pinch_squeeze_m: float = 0.0
    #: base tracking (paths) and base-follow (articulation) tracker configurations
    base_tracker: BaseTrackerConfig = field(default_factory=BaseTrackerConfig)
    base_follow_speed: float = 0.15  # m/s along the recorded b_k
    base_follow_yaw_rate: float = 0.4  # rad/s
    #: hybrid follow: the arm's hold target ramps between consecutive recorded poses by the base's progress along the
    #: segment; a short base step with a large arm change (dishwasher_wide 0920-002330: 0.1 m at 0.15 m/s carrying a
    #: 0.5 rad wrist change - 3 rad/s, the robot wobbled) is stretched so no arm joint ramps faster than this (rad/s)
    follow_arm_speed: float = 0.35
    #: hybrid follow, arm-only dense steps: eef speed (m/s) of the differential-IK tracking of the recorded handle pose
    follow_eef_speed: float = 0.05
    #: hybrid follow: base acceleration bound per segment (m/s^2; the stock tracker's 2.3 m/s^2 start/stop rocked the torso)
    follow_base_accel: float = 0.4
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
    #: ``track_path(corner_stop=True)`` (owner 2026-10-13, an axis-aligned leg): the reference holds each interior waypoint this
    #: long after stopping there, so the base settles before the next move along the other axis (s)
    corner_dwell_s: float = 0.4
    #: ... and brakes toward every stop at this deceleration (m/s^2): the tracker's command smoothing runs the base ahead of a
    #: reference braking at the tracker's own limit (a 0.5 m slide stopped 4 cm past its corner at 0.6 m/s^2 in the fake world)
    corner_stop_decel: float = 0.2
    #: ``track_path(style="face_travel")`` (camera relocalization, owner 2026-09-22): the shared driven schedule of
    #: ``robot.base_path`` — the polyline simplified within one map cell, corners filleted under a 5 cm deviation budget
    #: with the body turning over a window that keeps cruise speed, the end turns on the spot above 90° and blended
    #: below; the speed profile adds the lateral-acceleration and yaw limits through the arcs. The sampler validates
    #: the same schedule with the same (default) configuration: change it on both sides or not at all.
    face_travel: BP.BasePathConfig = field(default_factory=BP.BasePathConfig)
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
    #: (2026-09-22, 0918-000750) a converged FREE-SPACE leg whose joints keep reporting velocities above ``jam_vel_tol``
    #: for ``jam_probe_steps`` steps while their positions change by less than ``jam_pos_eps`` (rad) is jammed: a drive
    #: fighting a contact the differential IK did not see (the elbow folded onto its own upper arm: 0.3-2.2 rad/s reported,
    #: 0.5 mrad of motion, for the whole 185-step settle window - the evaluator's ``pose_settled`` of the robot never held).
    #: The hold targets are then re-targeted onto the measured joints over ``jam_relax_steps`` steps (``relax_arms``).
    #: ``jam_vel_tol`` sits well above the 0.02-0.05 rad/s a freshly converged free arm still reports (0922-180242).
    jam_relax: bool = True
    jam_vel_tol: float = 0.1
    jam_probe_steps: int = 3
    jam_pos_eps: float = 1e-3
    jam_relax_steps: int = 6
    #: (owner 2026-10-09, audit of the v3.0-reconciled demos: "too much push at the end makes the robot wobble") a contact that bears
    #: load is let go of only once the drives stopped pressing on it (``unload``): the hold targets of the arm and of the trunk follow
    #: the measured joints for ``unload_steps`` steps. Nothing is emitted unless an arm is more than ``unload_min_arm_rad`` or the
    #: trunk more than ``unload_min_trunk_rad`` off its target (a free arm converges within ~1 mrad, a free torso within 0.2 mrad;
    #: the door push-close left torso_joint1 74 mrad behind its target against the closed leaf)
    unload_steps: int = 12
    unload_min_arm_rad: float = 0.005
    unload_min_trunk_rad: float = 0.001


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


def joint_axis_from_contacts(contacts: Sequence[Sequence[float]]) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """``(centre, unit axis)`` of the circle through the handle contact points recorded at several joint values (the
    articulation's ``waypoints``): the hinge line the hand moves about. None with fewer than three distinct points."""
    pts = [C.as_tensor(c).reshape(-1)[:3].double() for c in contacts]
    if len(pts) < 3:
        return None
    p0, p1, p2 = pts[0], pts[len(pts) // 2], pts[-1]
    u, v = p1 - p0, p2 - p0
    n = th.cross(u, v, dim=0)
    if float(n.norm()) < 1e-9:
        return None
    n = n / n.norm()
    uu, vv, uv = float(u @ u), float(v @ v), float(u @ v)
    d = 2.0 * (uu * vv - uv * uv)
    if abs(d) < 1e-12:
        return None
    a = (vv * (uu - uv)) / d
    b = (uu * (vv - uv)) / d
    centre = p0 + a * u + b * v
    return centre.float(), n.float()


def arc_between(eef_a: Pose, eef_b: Pose, centre: th.Tensor, axis: th.Tensor, step_rad: float = 0.03) -> List[Pose]:
    """Dense poses from ``eef_a`` to ``eef_b`` along the circle about (``centre``, ``axis``): the rigid rotation that
    carries ``eef_a``'s position onto ``eef_b``'s (angle and sign from the two radial vectors), sampled every
    ``step_rad``; falls back to ``[eef_b]`` when the two poses are not on one circle about that axis (> 2 cm off)."""
    pa, qa = C.as_pose(eef_a)
    pb, qb = C.as_pose(eef_b)
    ax = C.unit(axis)
    ra = (pa - centre) - ax * float((pa - centre) @ ax)
    rb = (pb - centre) - ax * float((pb - centre) @ ax)
    if float(ra.norm()) < 1e-6 or float(rb.norm()) < 1e-6 or abs(float(ra.norm()) - float(rb.norm())) > 0.02:
        return [(pb, qb)]
    cosang = max(-1.0, min(1.0, float(ra @ rb) / (float(ra.norm()) * float(rb.norm()))))
    ang = math.acos(cosang)
    if float(th.cross(ra, rb, dim=0) @ ax) < 0:
        ang = -ang
    n = max(1, int(math.ceil(abs(ang) / max(step_rad, 1e-6))))
    out = C.orbit_waypoints((pa, qa), centre, C.quat_from_axis_angle(ax, ang), n=n, frame="world")
    out[-1] = (pb, qb)  # land exactly on the recorded pose
    return out


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
    #: single collision pieces ``(object, link, mesh)`` out of the finishing plan's world as well (``R1ProCuRobo.update_world``):
    #: the board pieces under an object on an interior shelf, whose support stays an obstacle otherwise (``execute.interior_split``)
    ignore_links: Sequence[Any] = ()


def links_kw(links: Sequence[Any]) -> Dict[str, Any]:
    """The ``ignore_links`` keyword of a world update / ``arm_to`` for ``links``; empty when there are none, so that call is the one
    made without them (a backend without link-level ignores never sees the keyword)."""
    return {"ignore_links": list(links)} if links else {}


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
    """The eight Layer-1 motions bound to one R1Pro (``env.scene.robots[0]``) and one cuRobo backend."""

    def __init__(self, env, planner, cfg: Optional[MotionsConfig] = None, robot=None):
        self.env = env
        self.robot = robot if robot is not None else env.scene.robots[0]
        self.backend = planner
        self.cfg = cfg or MotionsConfig()
        self.dt = float(getattr(planner, "control_dt", 1.0 / 30.0))
        self.state = ActionState(self.robot)
        self.stop_requested = False  # cooperative stop of a running follow (execute_base_follow at the goal band)
        self.follow_phase = "arm"  # sequential follow: "base" while the base drives (the caller watches the door then)
        self.last_relaxed_links: List[str] = []  # held-arm links taken out of the sphere model by the last arm_to plan
        self.last_witness_residual: Optional[Tuple[float, float]] = None  # a witness residual arm_to left unclosed (m, rad)
        self.last_jam: Optional[Dict[str, Any]] = None  # the last jammed free-space leg relaxed by _relax_if_jammed
        self.jams: List[Dict[str, Any]] = []
        # arms that keep their object to the end of the episode (set by the executor for the program's last action): a
        # wrist carrying a payload sits a steady offset below its target, which the jam probe reads as a jam; relaxing
        # it only lets the payload sag in front of the camera
        self.keep_hold_arms: Set[str] = set()
        # the protected-object transit check (owner 2026-09-26, oracle.protection): the active context (set by the execute
        # functions around a transit) and every firing of it this episode (task, action, object, clearance, remedy)
        self._protect: Optional[Dict[str, Any]] = None
        self.protect_events: List[Dict[str, Any]] = []
        self.protect_checks = 0
        # every witness whose other-branch pose-goal plan was kept (``arm_to(accept_other_branch=True)``, a push's pre-contact)
        self.witness_branch_accepted_events: List[Dict[str, Any]] = []
        # every witness reached through an arm posture (``cfg.witness_via_postures``) instead of failing the attempt
        self.witness_via_posture_events: List[Dict[str, Any]] = []
        # the dipping-transit gate of a grasp transit (first-demo v3.0 F5, ``transit_dip_gate``): the active context, None outside it
        self._dip: Optional[Dict[str, Any]] = None
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
        #: gripper bookkeeping (executor state, exported with the checkpoints): the intended open / closed state per arm (a
        #: pre-shaped opening emits a command < 0 while the fingers are meant to be open) and the release events for the record
        self.gripper_intent_closed: Dict[str, bool] = {}
        self.release_events: List[Dict[str, Any]] = []
        #: object -> the world move a next-to / touching place made away from its partner so the opening fingers clear it
        #: (``execute.clear_release_of_partner``); the push that follows extends its line by it
        self.place_shifts: Dict[str, th.Tensor] = {}

    # ------------------------------------------------------------------ privileged reads (state, never decisions)

    def q_full(self) -> th.Tensor:
        return self.robot.get_joint_positions().clone().float()

    def eef_pose(self, arm: str) -> Pose:
        pos, quat = self.robot.eef_links[arm].get_position_orientation()
        return pos.float().clone(), quat.float().clone()

    def _deflection(self, arm: str) -> Optional[float]:
        """Distance (m) between the eef link's pose and the FK of the reported joint positions - the elastic wind-up
        of a loaded arm (0 when unloaded; None without a backend FK)."""
        fk = getattr(self.backend, "fk", None)
        if fk is None:
            return None
        try:
            pose = fk([M.EEF_LINK[arm]])[M.EEF_LINK[arm]]
        except Exception:  # noqa: BLE001
            return None
        return float(th.linalg.norm(as_world_pose(pose)[0] - self.eef_pose(arm)[0]))

    def base_pose2d(self) -> Tuple[float, float, float]:
        pos, quat = self.robot.get_position_orientation()
        return float(pos[0]), float(pos[1]), C.yaw_of(quat.float())

    def held_object(self, arm: str):
        return self.robot._ag_obj_in_hand.get(arm)

    def object_between_pads(self, arm: str) -> bool:
        """Whether both finger pads of ``arm`` touch one assisted-graspable object right now (OmniGibson's own in-hand
        rule ``_calculate_in_hand_object``: contact on at least two finger links + the object on the ray between the pads
        - the condition whose 0.3 s window attaches the assisted grasp). False when the robot has no such rule."""
        fn = getattr(self.robot, "_calculate_in_hand_object", None)
        if not callable(fn):
            return False
        try:
            return fn(arm=arm) is not None
        except Exception:  # noqa: BLE001
            return False

    def gripper_closed(self, arm: str) -> bool:
        """Whether the gripper is meant to be closed (the last ``gripper`` intent; a pre-shaped opening emits a command
        below zero while the fingers are meant to be open)."""
        intent = self.gripper_intent_closed.get(arm)
        return bool(intent) if intent is not None else self.state.gripper_cmd[arm] < 0.0

    # ------------------------------------------------------------------ executor state (checkpoints, §3.8)

    def export_state(self) -> Dict[str, Any]:
        """Hold targets, gripper commands, trunk scalars and the tracker's reference index (plain tensors / numbers)."""
        return {
            "trunk_target": self.state.trunk_target.clone(),
            "arm_target": {arm: q.clone() for arm, q in self.state.arm_target.items()},
            "gripper_cmd": dict(self.state.gripper_cmd),
            "trunk": (float(self.trunk.translate), float(self.trunk.tilt)),
            "tracker_index": None if self.tracker is None else int(self.tracker.k),
            "gripper_intent_closed": dict(self.gripper_intent_closed),
        }

    def import_state(self, d: Dict[str, Any]) -> None:
        self.state.trunk_target = d["trunk_target"].clone()
        self.state.arm_target = {arm: q.clone() for arm, q in d["arm_target"].items()}
        self.state.gripper_cmd = dict(d["gripper_cmd"])
        self.state.stop_base()
        t, k = d["trunk"]
        self.trunk = TrunkState(float(t), float(k), translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)
        self.tracker = None
        self.gripper_intent_closed = dict(d.get("gripper_intent_closed") or {})

    # ------------------------------------------------------------------ emitters

    def _emit(self) -> th.Tensor:
        self.last_steps += 1
        return self.state.action()

    def hold(self, steps: int = 1) -> Iterator[th.Tensor]:
        """Emit hold actions (zero base velocity, current targets)."""
        self.state.stop_base()
        for _ in range(int(steps)):
            yield self._emit()

    def _still_watch(self) -> Callable[[], bool]:
        """The still exit of ``settle``: a check to call after every emitted step, True once every robot joint moved less than
        STILL_EXIT_MOVE per step and the object released last (``release_events``; none: no object check) moved slower than
        STILL_EXIT_OBJECT_SPEED, STILL_EXIT_STEPS steps in a row."""
        name = next((e.get("object") for e in reversed(self.release_events) if e.get("event") == "instant_open" and e.get("object")), None)
        scene = getattr(self.env, "scene", None)
        obj = scene.object_registry("name", name, None) if (name is not None and scene is not None) else None
        prev = [self.robot.get_joint_positions().float().clone()]
        run = [0]

        def still() -> bool:
            q = self.robot.get_joint_positions().float()
            ok = float((q - prev[0]).abs().max()) < STILL_EXIT_MOVE
            prev[0] = q.clone()
            if ok and obj is not None:
                try:
                    ok = float(th.linalg.norm(obj.get_linear_velocity().float())) < STILL_EXIT_OBJECT_SPEED
                except Exception:  # noqa: BLE001  (as SimState.object_velocity: no reading = at rest)
                    pass
            run[0] = run[0] + 1 if ok else 0
            return run[0] >= STILL_EXIT_STEPS

        return still

    def _trim_still_waypoints(self, q_traj: th.Tensor, arms: Sequence[str], trunk: bool) -> th.Tensor:
        """The still exit of ``execute_joint_trajectory``, on the joints the trajectory drives: streaming starts at the last
        waypoint still within STILL_EXIT_MOVE of the first, and the trailing waypoints within it of the last are dropped but
        the last."""
        idx = [self.robot.arm_control_idx[a] for a in arms if a in self.robot.arm_control_idx] + ([self.robot.trunk_control_idx] if trunk else [])
        T = len(q_traj)
        if T < 2 or not idx:
            return q_traj
        sub = q_traj[:, th.cat([th.as_tensor(i).long().flatten() for i in idx])].float()
        moving = th.nonzero((sub - sub[0]).abs().max(dim=1).values > STILL_EXIT_MOVE).flatten()
        start = int(moving[0]) - 1 if len(moving) else T - 1
        away = th.nonzero((sub - sub[-1]).abs().max(dim=1).values > STILL_EXIT_MOVE).flatten()
        end = int(away[-1]) + 1 if len(away) else 0  # the first of the trailing waypoints within STILL_EXIT_MOVE of the last
        keep = list(range(start, max(start, end))) + [T - 1]
        return q_traj if len(keep) == T else q_traj[keep]

    def settle(self, max_steps: Optional[int] = None, vel_tol: Optional[float] = None) -> Iterator[th.Tensor]:
        """Hold until every joint velocity is below ``vel_tol`` (or ``max_steps`` elapsed; never raises)."""
        self.state.stop_base()
        max_steps = self.cfg.settle_max_steps if max_steps is None else int(max_steps)
        vel_tol = self.cfg.settle_vel_tol if vel_tol is None else float(vel_tol)
        still = self._still_watch() if oracle_still_exit() else None
        for _ in range(max_steps):
            yield self._emit()
            if float(self.robot.get_joint_velocities().abs().max()) < vel_tol:
                return
            if still is not None and still():
                return

    def _wait_converged(self, arms: Sequence[str], trunk: bool, tol: float, max_steps: int, loose: Optional[float] = None) -> Iterator[th.Tensor]:
        def err() -> float:
            return max([self.state.arm_error(a) for a in arms] + ([self.state.trunk_error()] if trunk else [0.0]))

        still_exit = oracle_still_exit()
        errs: List[float] = []
        for _ in range(max_steps):
            if err() < tol:
                return
            if still_exit:
                errs.append(err())
                if len(errs) > STILL_EXIT_CONVERGE_WINDOW and errs[-1 - STILL_EXIT_CONVERGE_WINDOW] - errs[-1] < STILL_EXIT_MOVE:
                    break
            yield self._emit()
        e = err()
        if e > (self.cfg.joint_tol_loose if loose is None else loose):
            raise SkillError("execution", f"joints did not converge (max err {e:.3f} rad)", arms=list(arms), trunk=trunk, err_rad=e)
        if e >= tol:
            log.info("joints settled %.3f rad off their targets (accepted%s)", e, ", contact leg" if loose is not None else "")

    # ------------------------------------------------------------------ gripper

    def _gripper_limits(self, arm: str) -> Tuple[float, float]:
        idx = self.robot.gripper_control_idx[arm]
        return float(self.robot.joint_lower_limits[idx].float().mean()), float(self.robot.joint_upper_limits[idx].float().mean())

    def gripper_command(self, arm: str, q_m: float) -> float:
        """The smooth-mode gripper command holding the fingers at opening ``q_m`` per finger (linear in [-1, 1] between
        the joint limits; the closure probe's ``2·(q − lo)/(hi − lo) − 1``)."""
        lo, hi = self._gripper_limits(arm)
        q_m = min(hi, max(lo, float(q_m)))
        return max(-1.0, min(1.0, 2.0 * (q_m - lo) / (hi - lo) - 1.0))

    def gripper_opening(self, arm: str, q_m: float, steps: Optional[int] = None) -> Iterator[th.Tensor]:
        """Hold the fingers at opening ``q_m`` per finger (smooth-mode command; the pre-shaped opening of a grasp,
        ``execute_grasp``: pinch width / 2 + ``gripper_preshape_margin`` before the descent so no pad stands on the
        table)."""
        lo, hi = self._gripper_limits(arm)
        q_m = min(hi, max(lo, float(q_m)))
        self.gripper_intent_closed[arm] = False
        self.state.gripper_cmd[arm] = self.gripper_command(arm, q_m)
        self.state.stop_base()
        for _ in range(int(self.cfg.gripper_preshape_steps if steps is None else steps)):
            yield self._emit()

    def gripper(self, arm: str, close: bool, steps: Optional[int] = None, rate: Optional[float] = None,
                squeeze: Optional[float] = None) -> Iterator[th.Tensor]:
        """Open / close the gripper and wait until the fingers stop moving (or an assisted grasp engages).

        Closing: ``rate`` (m per control step; ``execute_grasp`` passes ``cfg.gripper_close_rate``) ramps the smooth-mode
        command from the current opening to closed so both pads track the same target and reach the object in the same
        step even while one pad rubs the table (closure probe 2026-09-16: the stock step command let the free pad knock
        the un-pinched cylinder along the table and roll it 19-77 deg); the stock step command follows the ramp (and is
        the whole command without a rate: the push / brace closures). ``squeeze`` replaces ``cfg.gripper_contact_squeeze_m``
        (a thin pinch's ``thin_pinch_squeeze_m``).

        Opening (owner 2026-10-10): exactly the open command (+1) in one step, every release, the hand held where it is -
        no ramp, no partial or staged opening. A held object is released by that command alone: OmniGibson's assisted
        grasp lets go when the finger target reaches the open limit (and wakes the object), as it does for a policy
        imitating the demo; the retreat follows. A handle of a fixed-base body is unloaded first."""
        self.state.stop_base()
        idx = self.robot.gripper_control_idx[arm]
        lo, hi = self._gripper_limits(arm)
        self.gripper_intent_closed[arm] = bool(close)
        q_now = float(self.robot.get_joint_positions()[idx].float().mean())
        held_before = None if close else self.held_object(arm)
        if close:
            contact_hold = None
            sq = self.cfg.gripper_contact_squeeze_m if squeeze is None else float(squeeze)
            if rate:
                q_target = q_now
                n = 0
                while q_target > lo + 1e-6:
                    q_target = max(lo, q_target - float(rate))
                    self.state.gripper_cmd[arm] = self.gripper_command(arm, q_target)
                    yield self._emit()
                    n += 1
                    if n > 10 and self.held_object(arm) is not None:
                        break  # attached: finish with the full command below
                    if sq is not None and self.object_between_pads(arm):
                        # symmetric closure: both pads touch the object - stop the ramp and hold the fingers at the contact
                        # width minus the squeeze, so the two saturated drives do not fight through the object (finger 2
                        # would win and shove the un-attached object toward finger 1: marker 8.5 mm, napkins box 15.4 mm)
                        q_touch = float(self.robot.get_joint_positions()[idx].float().mean())
                        contact_hold = max(lo, q_touch - float(sq))
                        self.state.gripper_cmd[arm] = self.gripper_command(arm, contact_hold)
                        self.release_events.append({"arm": arm, "event": "contact_hold", "hold_m": round(contact_hold, 4), "touch_m": round(q_touch, 4),
                                                    "fingers_m": [round(float(v), 4) for v in self.robot.get_joint_positions()[idx].float().tolist()], "ramp_steps": n})
                        break
            if contact_hold is not None:
                # wait for the assisted grasp with the pads holding the object; without it in the step budget the stock
                # command takes over (the closure then behaves as before)
                steps = self.cfg.gripper_steps if steps is None else int(steps)
                for i in range(steps):
                    yield self._emit()
                    if self.held_object(arm) is not None:
                        self.release_events[-1]["attached_after_steps"] = i + 1
                        self.release_events[-1]["fingers_at_attach_m"] = [round(float(v), 4) for v in self.robot.get_joint_positions()[idx].float().tolist()]
                        yield from self.hold(5)
                        return
                log.info("gripper(%s): both pads on the object but no assisted grasp within %d steps at the contact hold %.1f mm; closing fully", arm, steps, contact_hold * 1000)
                self.release_events[-1]["fallback"] = "full_close"
            self.state.set_gripper(arm, True)
        else:
            articulated = held_before is not None and bool(getattr(held_before, "n_joints", 0) or len(getattr(held_before, "joints", {}) or {}))
            if articulated and bool(getattr(held_before, "fixed_base", False)):
                # a handle of a door / appliance: the drives stop pressing before the fingers open (dishwasher close 1008: the
                # torso bent 0.24 deg against the closed door sprang back at the release). A free-standing body (a blender,
                # a carton) is left as it is: re-reading the loaded arm turns it (execute_base_follow's unwind)
                yield from self.unload([arm])
            if held_before is not None:
                self.release_events.append({"arm": arm, "event": "instant_open", "from_m": round(q_now, 4), "object": getattr(held_before, "name", None)})
            self.state.set_gripper(arm, False)  # exactly the open command (the assisted grasp's release condition)
        steps = self.cfg.gripper_steps if steps is None else int(steps)
        still_exit = oracle_still_exit() and not (close and rate)
        q_fingers = self.robot.get_joint_positions()[idx].float().clone() if still_exit else None
        n_still = 0
        for i in range(steps):
            yield self._emit()
            if still_exit:
                q_now_f = self.robot.get_joint_positions()[idx].float()
                n_still = n_still + 1 if float((q_now_f - q_fingers).abs().max()) < STILL_EXIT_MOVE else 0
                q_fingers = q_now_f.clone()
            if i > 10:
                if close and self.held_object(arm) is not None:
                    break
                if float(self.robot.get_joint_velocities()[idx].abs().max()) < 1e-3:
                    break
                if still_exit and n_still >= STILL_EXIT_STEPS:
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

    def execute_joint_trajectory(self, q_traj: th.Tensor, arms: Sequence[str], trunk: bool = False, tol: Optional[float] = None, loaded: bool = False,
                                 converge: bool = True) -> Iterator[th.Tensor]:
        """Stream a ``(T, 28)`` joint trajectory (one waypoint per control step) for the given groups, then hold the
        last waypoint until the joints converge (``loaded``: a contact leg, the steady error under the contact force
        is accepted up to ``joint_tol_loaded``). ``converge=False`` returns right after the last waypoint: the caller
        judges the outcome on the object it moves (a door held by its joint friction never lets the loaded arm
        converge - the follow waited the full 90 steps for a 0.052 rad error that could not change)."""
        q_traj = C.as_tensor(q_traj)
        if len(q_traj) > self.cfg.trajectory_max_steps:
            raise SkillError("timeout", f"trajectory of {len(q_traj)} steps exceeds the budget {self.cfg.trajectory_max_steps}")
        if oracle_still_exit():
            q_traj = self._trim_still_waypoints(q_traj, arms, trunk)
        self.state.stop_base()
        for q in q_traj:
            self.state.set_from_q(q, arms=arms, trunk=trunk)
            yield self._emit()
        if converge:
            yield from self._wait_converged(arms, trunk, tol or self.cfg.joint_tol, self.cfg.settle_steps, loose=self.cfg.joint_tol_loaded if loaded else None)
            if not loaded:
                yield from self._relax_if_jammed(arms)
        if trunk:
            self.trunk = TrunkState.from_q(self.state.trunk_target, translate_rate=self.cfg.trunk_translate_rate, tilt_rate=self.cfg.trunk_tilt_rate)

    def relax_arms(self, arms: Sequence[str], steps: Optional[int] = None) -> Generator[th.Tensor, None, Dict[str, float]]:
        """Re-target the hold targets of ``arms`` onto their measured joints over ``steps`` control steps (the measured
        joints re-read every step, as ``follow_unwind`` does; a one-step unwind let a wound-up arm snap 0.85 rad/s), so
        the position drives stop pressing against whatever holds the arm. Returns ``{arm: largest target change (rad)}``
        (read it with ``yield from``); the base is stopped."""
        steps = self.cfg.jam_relax_steps if steps is None else int(steps)
        steps = max(int(steps), 1)
        arms = [a for a in arms if a in M.ARMS]
        start = {a: (None if self.state.arm_target.get(a) is None else self.state.arm_target[a].clone().float()) for a in arms}
        out: Dict[str, float] = {a: 0.0 for a in arms}
        self.state.stop_base()
        for k in range(1, steps + 1):
            f = k / steps
            for a in arms:
                q = self.robot.get_joint_positions()[self.robot.arm_control_idx[a]].clone().float()
                q0 = q if start[a] is None else start[a]
                self.state.arm_target[a] = q0 + (q - q0) * f
                out[a] = max(out[a], float(th.max(th.abs(q - q0))))
            yield self._emit()
        return out

    def _relax_if_jammed(self, arms: Sequence[str]) -> Iterator[th.Tensor]:
        """After a free-space leg converged: joints still reporting velocities above ``jam_vel_tol`` for
        ``jam_probe_steps`` steps without moving (``jam_pos_eps``) are a drive fighting an unseen contact - the hold
        targets are relaxed onto the measured joints (``relax_arms``; the jam is kept in ``last_jam`` for the notes).
        Arms in ``keep_hold_arms`` are never relaxed."""
        arms = [a for a in arms if a in M.ARMS and a not in self.keep_hold_arms]
        if not self.cfg.jam_relax or not arms:
            return
        idx = th.cat([self.robot.arm_control_idx[a] for a in arms])
        q0 = self.robot.get_joint_positions()[idx].clone().float()
        for _ in range(int(self.cfg.jam_probe_steps)):
            if float(self.robot.get_joint_velocities()[idx].abs().max()) <= self.cfg.jam_vel_tol:
                return
            yield self._emit()
        v = float(self.robot.get_joint_velocities()[idx].abs().max())
        dq = float((self.robot.get_joint_positions()[idx].float() - q0).abs().max())
        if v <= self.cfg.jam_vel_tol or dq > self.cfg.jam_pos_eps:
            return
        worst = self._jam_worst_joint(arms)
        contacts = self._jam_contacts(arms)
        log.warning("%s arm joints report %.2f rad/s but moved %.2f mrad over %d steps after converging: a drive is fighting a contact the leg did not see - "
                    "relaxing the hold targets onto the measured joints%s%s", "/".join(arms), v, dq * 1000, int(self.cfg.jam_probe_steps),
                    "" if not worst else f" (worst joint {worst['joint']}: {worst['error_rad']:+.4f} rad off its target, {worst['to_limit_rad']:.4f} rad from its {worst['limit']} limit{' - AT THE LIMIT' if worst['at_limit'] else ''})",
                    f"; contacts of the arm's links right now: {contacts}" if contacts else "; no contact reported on the arm's links (the held object's contacts included)")
        relaxed = yield from self.relax_arms(arms)
        self.last_jam = {"arms": list(arms), "reported_vel_rad_s": round(v, 4), "moved_rad": round(dq, 6), "relaxed_rad": {a: round(d, 5) for a, d in relaxed.items()},
                         "worst_joint": worst, "contacts": contacts}
        self.jams.append(self.last_jam)

    def _jam_contacts(self, arms: Sequence[str], limit: int = 8) -> List[str]:
        """``'<own link> ~ <other body>'`` for every current contact pair of the robot whose own body is one of ``arms``' links
        (or an object held in that hand): robot-robot pairs included, so a self-contact (the elbow folded into the torso) is
        named too - QA and ``hand_contacts`` leave those out."""
        try:
            from omnigibson.utils.usd_utils import RigidContactAPI
        except Exception:  # noqa: BLE001
            return []
        scene = getattr(self.robot, "scene", None)
        if scene is None or not hasattr(self.robot, "prim_path"):
            return []
        prefix = self.robot.prim_path + "/"
        try:
            pairs = RigidContactAPI.get_contact_pairs(scene.idx, [self.robot], None, current_only=True)
        except Exception as e:  # noqa: BLE001
            log.debug("jam contacts unavailable: %r", e)
            return []
        held = {a: getattr(self.robot, "_ag_obj_in_hand", {}).get(a) for a in arms}
        held_prefixes = {o.prim_path + "/" for o in held.values() if o is not None and hasattr(o, "prim_path")}
        out: List[str] = []
        for own, other in pairs:
            own_s, other_s = str(own), str(other)
            own_link = own_s.rsplit("/", 1)[-1]
            mine = own_s.startswith(prefix) and any(own_link.startswith(a + "_") or own_link == f"attached_object_{a}_eef_link" for a in arms)
            if not mine and any(own_s.startswith(p) for p in held_prefixes):
                mine, own_link = True, "held:" + own_link
            if not mine:
                continue
            if other_s.startswith(prefix):
                other_name = "robot:" + other_s.rsplit("/", 1)[-1]
            else:
                other_name = other_s.rsplit("/", 2)[-2] + "/" + other_s.rsplit("/", 1)[-1] if other_s.count("/") >= 2 else other_s
            item = f"{own_link} ~ {other_name}"
            if item not in out:
                out.append(item)
            if len(out) >= limit:
                break
        return out

    def _jam_worst_joint(self, arms: Sequence[str]) -> Optional[Dict[str, Any]]:
        """The arm joint furthest from its hold target and how far it stands from its joint limits (cork 0925-195316 / 205211:
        the re-aim leg 'settled 0.056 rad off' then jammed - a joint pressed against its limit and a hand blocked by a contact
        look alike in the velocities; this names the joint so the notes tell them apart)."""
        try:
            q_all = self.robot.get_joint_positions().float()
            lo_all, hi_all = self.robot.joint_lower_limits.float(), self.robot.joint_upper_limits.float()
            names = list(self.robot.joints.keys())
            best: Optional[Dict[str, Any]] = None
            for a in arms:
                tgt = self.state.arm_target.get(a)
                if tgt is None:
                    continue
                idx = self.robot.arm_control_idx[a]
                for k, i in enumerate(idx.tolist()):
                    err = float(tgt[k]) - float(q_all[i])
                    d_lo, d_hi = float(q_all[i] - lo_all[i]), float(hi_all[i] - q_all[i])
                    limit = "lower" if d_lo < d_hi else "upper"
                    to_limit = min(d_lo, d_hi)
                    toward = (err < 0 and limit == "lower") or (err > 0 and limit == "upper")  # the target lies beyond the near limit
                    rec = {"arm": a, "joint": names[i] if i < len(names) else str(i), "error_rad": round(err, 5), "to_limit_rad": round(to_limit, 5),
                           "limit": limit, "at_limit": bool(to_limit < 0.02 and toward)}
                    if best is None or abs(err) > abs(best["error_rad"]):
                        best = rec
            return best
        except Exception as e:  # noqa: BLE001
            log.debug("jam diagnostics unavailable: %r", e)
            return None

    def unwind(self, arm: str) -> float:
        """Re-target ``arm``'s hold target onto its measured joints so the drives stop pushing. A loaded arm (the hand
        on a door held by its joint friction) is wound up against its target: the position drives (stiffness 1e7,
        18-55 N·m caps) keep pressing until the load is gone, and the release then shoves the object (dishwasher
        0917-213723: joint4 0.052 rad off its target, the door moved 0.009 rad while the fingers opened). Returns the
        largest joint change (rad); emits nothing - the caller holds."""
        q = self.robot.get_joint_positions()[self.robot.arm_control_idx[arm]].clone().float()
        prev = self.state.arm_target.get(arm)
        d = float(th.max(th.abs(q - prev))) if prev is not None else 0.0
        self.state.arm_target[arm] = q
        return d

    def unwind_trunk(self, min_rad: float = 0.0) -> float:
        """The trunk's :meth:`unwind`: its hold target becomes the measured torso when that lies more than ``min_rad`` off it. A load the
        hand bears bends the torso against its effort-capped drives - the leaf a fist presses, a door held by its friction, the
        support under the fingertips - and an arm unwound alone is then a rigid link that keeps it bent, until the contact goes and
        the drives spring it back (door open: torso_joint1 6.6 mrad past its target through the unwind, released by the gripper).
        Returns the change (rad); emits nothing - the caller holds."""
        q = self.robot.get_joint_positions()[self.robot.trunk_control_idx].clone().float()
        d = float(th.max(th.abs(q - self.state.trunk_target)))
        if d <= float(min_rad):
            return 0.0
        self.state.trunk_target = q
        return d

    def unload(self, arms: Sequence[str], steps: Optional[int] = None, trunk: bool = True) -> Generator[th.Tensor, None, Dict[str, float]]:
        """Stop pressing before a contact that bears load is let go of (``unload_steps``): every step the hold targets of ``arms`` and
        (``trunk``) the torso's follow the measured joints, the base stopped, so the strain bleeds off while the contact still holds
        and nothing springs back when it goes. Emits nothing when no arm is more than ``unload_min_arm_rad`` and the trunk no more
        than ``unload_min_trunk_rad`` off its target. Returns the largest change per arm and ``"trunk"`` (rad; ``yield from``)."""
        arms = [a for a in arms if a in M.ARMS and self.state.arm_target.get(a) is not None]
        out: Dict[str, float] = {a: 0.0 for a in arms}
        out["trunk"] = 0.0
        loaded = any(self.state.arm_error(a) > self.cfg.unload_min_arm_rad for a in arms)
        loaded = loaded or (trunk and self.state.trunk_error() > self.cfg.unload_min_trunk_rad)
        if not loaded:
            return out
        self.state.stop_base()
        for _ in range(max(1, int(self.cfg.unload_steps if steps is None else steps))):
            for a in arms:
                out[a] = max(out[a], self.unwind(a))
            if trunk:
                out["trunk"] = max(out["trunk"], self.unwind_trunk())
            yield self._emit()
        return out

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

    # ------------------------------------------------------------------ protected-object transit check (oracle.protection)

    @contextmanager
    def protecting(self, objects: Sequence[Tuple[str, Any]], leg: str, task: str = "", action: str = "", index: Optional[int] = None,
                   force_via_above: bool = False, along: Optional[str] = None, inflation: Optional[float] = None):
        """Owner 2026-09-26 (teacup 0925-222212 / 235559: the transit to the pre-grasp shoved the saucer 20 cm): inside this
        context every trajectory ``arm_to`` plans (and, with ``along="raise"`` / ``"log"``, the dense path ``arm_along``
        tracks: the carry) is checked with the moving arm's spheres - a carried object's included - against the protected
        ``objects`` (``[(bddl name, scene object)]``) inflated by ``inflation`` (``protection.INFLATION_M``), threshold
        lowered to the endpoints' own clearance. A plan that clears runs unchanged; one that does not is re-planned via the
        point above its goal (``arm_to``) or raises ``plan_failed`` stage ``protected`` for the caller's via-above
        (``along="raise"``). ``force_via_above``: the retry after a run-time protected verdict takes the via-above at once.
        No objects: a no-op (the motion is exactly as before). Every firing lands in ``protect_events`` and the log."""
        from ispatialgym.oracle import protection as PROT

        prev = self._protect
        self._protect = None if not objects else {"objects": list(objects), "leg": leg, "task": task, "action": action, "index": index,
                                                   "force": bool(force_via_above), "along": along,
                                                   "inflation": PROT.INFLATION_M if inflation is None else float(inflation)}
        try:
            yield self._protect
        finally:
            self._protect = prev

    def _protect_violations(self, arms: Sequence[str], q_traj: Any, emb: str, P: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """The protected objects a planned trajectory ``q_traj`` ((T, 28) full configurations) comes too close to (``[]`` when it
        clears, when no context is active or when the backend cannot compute spheres)."""
        from ispatialgym.oracle import protection as PROT

        P = self._protect if P is None else P
        fn = getattr(self.backend, "robot_spheres_world", None)
        if P is None or fn is None or q_traj is None:
            return []
        try:
            q = C.as_tensor(q_traj)
            if q.ndim == 1 or len(q) < 3:
                return []
            spheres, links = fn(q, emb)
            self.protect_checks += 1
            return PROT.check_trajectory(spheres, links, arms, P["objects"], P["inflation"])
        except Exception as e:  # noqa: BLE001 - the check is advisory; the run-time monitor stays
            log.warning("protected-object plan check unavailable (%r)", e)
            return []

    def _protect_record(self, P: Dict[str, Any], arm: str, viol: Sequence[Dict[str, Any]], remedy: str, result: str) -> None:
        ev = {"check": "plan", "task": P.get("task"), "action": P.get("action"), "index": P.get("index"), "leg": P.get("leg"), "arm": arm,
              "objects": [dict(v) for v in viol], "remedy": remedy, "result": result}
        self.protect_events.append(ev)
        for v in viol or [{}]:
            log.warning("PROTECT plan-check fired: task=%s action=%s[%s] leg=%s object=%s clearance=%s m (threshold %s m, link %s, waypoint %s/%s) -> %s: %s",
                        P.get("task"), P.get("action"), P.get("index"), P.get("leg"), v.get("object", "-"), v.get("clearance_m", "-"), v.get("threshold_m", "-"),
                        v.get("link", "-"), v.get("index", "-"), v.get("n", "-"), remedy, result)

    def _via_above(self, arm: str, emb: str, goal_pose: Any, q_goal: Optional[th.Tensor], P: Dict[str, Any]) -> Generator[th.Tensor, None, Tuple[bool, str]]:
        """The via-above transit: the hand rises ``protect_lift_m`` straight up where it is (collision-checked; skipped when
        it cannot), a joint-space plan goes to the point ``protect_above_m`` above ``goal_pose`` (its IK seeded at ``q_goal``
        - the goal's branch - when given), checked against the protected objects, then a straight descent reaches
        ``goal_pose``. Returns ``(done, why)``; nothing after the lift has moved when it returns False."""
        up = th.tensor([0.0, 0.0, float(self.cfg.protect_lift_m)])
        here = self.eef_pose(arm)
        ign_objs, ign_links = getattr(self, "_protect_ignores", ((), ()))
        try:
            yield from self.arm_along(arm, [(here[0] + up, here[1])], check_collisions=True, ignore_objects=list(ign_objs), soft_pos_tol=self.cfg.diff_ik_soft_pos_tol_free,
                                      refine_jumps=False)
            lifted = "lifted"
        except SkillError as e:
            # the tracked straight rise jumps near the singular hanging posture (teacup 0926-010042: 'joint jump 0.22 rad at waypoint 1'):
            # a short joint-space plan to the raised hand pose instead (12 cm, else 6 cm), checked like any transit
            lifted = f"no lift ({e.message[:60]})"
            for dz in (float(self.cfg.protect_lift_m), 0.5 * float(self.cfg.protect_lift_m)):
                here = self.eef_pose(arm)
                try:
                    r_up = self.backend.plan_pose(emb, {M.EEF_LINK[arm]: (here[0] + th.tensor([0.0, 0.0, dz]), here[1])}, q_start=self.backend.q_full(),
                                                  max_attempts=self.cfg.plan_attempts, sync_world=False)
                except SkillError:
                    continue
                if not getattr(r_up, "success", False) or r_up.q_traj is None:
                    continue
                idx_up = self.robot.arm_control_idx[arm]
                q_up = C.as_tensor(r_up.q_traj)
                if float(th.max(th.abs(q_up[-1][idx_up] - q_up[0][idx_up]))) > 1.2 or self._protect_violations([arm], r_up.q_traj, emb, P):
                    continue  # a rise that swings the arm or passes a protected object is no remedy
                yield from self.execute_joint_trajectory(r_up.q_traj, arms=[arm])
                lifted = f"lifted {dz:.2f} m (planned)"
                break
        try:  # the enclosing arm_to's collision world (its ignores) for the plan to the point above
            self.backend.update_world(ignore_objects=ign_objs, ignore_links=ign_links)
        except TypeError:
            self.backend.update_world(ignore_objects=ign_objs)
        gp, gq = as_world_pose(goal_pose)
        above = (gp.float() + th.tensor([0.0, 0.0, float(self.cfg.protect_above_m)]), gq.float())
        q_start = self.backend.q_full()
        idx = self.robot.arm_control_idx[arm]
        q_above = None
        solve_ik = getattr(self.backend, "solve_ik", None)
        if q_goal is not None and solve_ik is not None:
            try:
                q_ik = solve_ik(emb, {M.EEF_LINK[arm]: above}, q_seed=q_goal, sync_world=False)
            except Exception:  # noqa: BLE001
                q_ik = None
            if q_ik is not None:
                q_ik = C.as_tensor(q_ik).reshape(-1)
                if float(th.max(th.abs(q_ik[idx] - q_goal[idx]))) <= self.cfg.witness_refine_max_jump:
                    q_above = q_start.clone()
                    q_above[idx] = q_ik[idx]
        if q_above is not None:
            res = self.backend.plan_joint(emb, q_above, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
        else:
            res = self.backend.plan_pose(emb, {M.EEF_LINK[arm]: above}, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
        if not res.success:
            return False, f"{lifted}; no plan to the point above the goal ({res.status})"
        viol = self._protect_violations([arm], res.q_traj, emb, P)
        if viol:
            return False, f"{lifted}; the plan to the point above still passes {viol[0]['object']} at {viol[0]['clearance_m']} m"
        yield from self.execute_joint_trajectory(res.q_traj, arms=[arm])
        yield from self.arm_along(arm, [(gp.float(), gq.float())], soft_pos_tol=self.cfg.diff_ik_soft_pos_tol_free)
        return True, f"{lifted}; above the goal by {self.cfg.protect_above_m:.2f} m ({'goal branch' if q_above is not None else 'pose goal'}), straight down"

    def _plan_detour(self, arm: str, q_traj: Any) -> Optional[Dict[str, Any]]:
        """The hand path of a planned transit that detours past the via-above's - longer than ``protect_detour_ratio`` x
        (``protect_lift_m`` + the straight distance + ``protect_above_m``) and higher than ``protect_lift_m`` above both of its
        ends: ``{"path_m", "straight_m", "via_above_m", "rise_m"}``; None otherwise or without a batched FK."""
        fk_batch = getattr(self.backend, "fk_batch", None)
        if fk_batch is None or q_traj is None:
            return None
        try:
            q = C.as_tensor(q_traj)
            if q.ndim != 2 or len(q) < 3:
                return None
            p = C.as_tensor(fk_batch(M.EEF_LINK[arm], q, f"{arm}_arm")[0]).float()
        except Exception as e:  # noqa: BLE001 - advisory, like the clearance check
            log.debug("transit detour check unavailable (%r)", e)
            return None
        path = float(th.linalg.norm(p[1:] - p[:-1], dim=1).sum())
        straight = float(th.linalg.norm(p[-1] - p[0]))
        via = float(self.cfg.protect_lift_m) + straight + float(self.cfg.protect_above_m)
        rise = float(p[:, 2].max()) - max(float(p[0, 2]), float(p[-1, 2]))
        if path <= float(self.cfg.protect_detour_ratio) * via or rise <= float(self.cfg.protect_lift_m):
            return None
        return {"object": "-", "path_m": round(path, 3), "straight_m": round(straight, 3), "via_above_m": round(via, 3), "rise_m": round(rise, 3)}

    @contextmanager
    def transit_dip_gate(self, arm: str, approach: Sequence[Any] = ()):
        """The planned transits of ``arm`` inside this block (a grasp transit, first-demo v3.0 F5) pass :meth:`_dip_gate` before they
        stream; ``approach`` (the grasp's approach waypoints) must track from where a Cartesian replacement ends. Yields the record:
        ``{"fired", "why", "dip_m", ...}`` of the last plan checked (``{"fired": False, "why": "no planned transit"}`` when none)."""
        prev = self._dip
        rec: Dict[str, Any] = {"fired": False, "why": "no planned transit"}
        self._dip = {"arm": arm, "approach": list(approach), "record": rec}
        try:
            yield rec
        finally:
            self._dip = prev

    def _plan_dip(self, arm: str, q_traj: Any) -> Optional[Tuple[float, float, float, float]]:
        """``(dip, z_start, z_end, z_min)`` of a planned trajectory's hand path - how far it runs below the lower of its ends (m); the
        mirror of :meth:`_plan_detour`. None without a batched FK."""
        fk_batch = getattr(self.backend, "fk_batch", None)
        if fk_batch is None or q_traj is None:
            return None
        try:
            q = C.as_tensor(q_traj)
            if q.ndim != 2 or len(q) < 3:
                return None
            p = C.as_tensor(fk_batch(M.EEF_LINK[arm], q, f"{arm}_arm")[0]).float()
        except Exception as e:  # noqa: BLE001 - advisory: without FK the plan runs as before
            log.debug("transit dip check unavailable (%r)", e)
            return None
        z0, z1, zmin = float(p[0, 2]), float(p[-1, 2]), float(p[:, 2].min())
        return min(z0, z1) - zmin, z0, z1, zmin

    def _track_leg(self, arm: str, start: Pose, waypoints: Sequence[Any], q_start: th.Tensor) -> Tuple[Any, List[Pose]]:
        """``solve_ik_path`` of the straight segments ``start`` -> ``waypoints`` from ``q_start`` with collisions checked (the current world),
        at the free legs' tolerances, solved once more at half the spacing on a joint jump (as ``arm_along``). ``(result, poses)`` - the
        poses aligned with ``result.q_traj`` (``start`` first)."""
        frame = M.EEF_LINK[arm]
        kw = dict(pos_tol=self.cfg.diff_ik_pos_tol, rot_tol=self.cfg.diff_ik_rot_tol, max_iters=self.cfg.diff_ik_max_iters, check_collisions=True, sync_world=False,
                  soft_pos_tol=self.cfg.diff_ik_soft_pos_tol_free, soft_rot_tol=self.cfg.diff_ik_soft_rot_tol)
        dense = self._densify(start, waypoints)
        res = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp} for wp in dense], q_start=q_start, **kw)
        if not res.success and str(res.status).startswith("joint jump"):
            finer: List[Pose] = []
            prev = start
            for wp in dense:
                finer.extend(C.line_waypoints(prev, wp, 2))
                prev = wp
            res2 = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp} for wp in finer], q_start=q_start, **kw)
            if res2.success:
                res, dense = res2, finer
        return res, [start] + list(dense)

    def _leg_steps(self, poses: Sequence[Pose], q_traj: th.Tensor) -> List[int]:
        """Control steps per segment of a tracked leg, as ``arm_along`` times it (eef speed, eef rotation speed, joint speed bound)."""
        steps = []
        for i in range(len(poses) - 1):
            seg = float(th.linalg.norm(as_world_pose(poses[i + 1])[0] - as_world_pose(poses[i])[0]))
            ang = C.quat_angle(as_world_pose(poses[i])[1], as_world_pose(poses[i + 1])[1])
            n = max(seg / (self.cfg.cartesian_speed * self.dt), ang / (self.cfg.cartesian_ang_speed * self.dt))
            if i + 1 < len(q_traj):
                n = max(n, float(th.max(th.abs(q_traj[i + 1] - q_traj[i]))) / (self.cfg.diff_ik_joint_speed * self.dt))
            steps.append(max(1, int(round(n))))
        return steps

    def _cartesian_transit(self, arm: str, waypoints: Sequence[Any], q_plan_end: th.Tensor) -> Tuple[Optional[List[Tuple[th.Tensor, List[int]]]], str]:
        """The joint path of a Cartesian leg replacing a dipping plan (:meth:`_dip_gate`): the hand along ``waypoints`` from where it is,
        tracked with collisions checked, keeping the protected objects' distance, ending within ``witness_refine_max_jump`` of the plan's
        last configuration (its IK branch: the chain after the transit runs as validated), the gate's approach tracking from its end.
        Tracked forward from the current configuration, else backward from the plan's last configuration to the hand's pose (the marker
        0928-050212: the idle arm stands at a shoulder singularity - joints 1 and 3 aligned - and the forward differential IK drifts into
        the flipped-shoulder family and stalls short; tracked back from the plan's end the same line reaches the hand 0.32 rad of in-place
        joint 1 / 3 motion from the idle joints): then an in-place adjustment of at most ``transit_dip_adjust_max_rad`` that moves the hand
        at most ``transit_dip_adjust_hand_m`` comes first. Differential IK, FK and collision checks only; nothing moves. Returns the
        segments ``[(q_traj, steps), ...]`` to stream in order, or None, and why."""
        D = self._dip or {}
        idx = self.robot.arm_control_idx[arm]
        q_now = C.as_tensor(self.backend.q_full()).reshape(-1).float()
        q_end_plan = C.as_tensor(q_plan_end).reshape(-1).float()
        here = self.eef_pose(arm)
        goal = as_world_pose(waypoints[-1])
        whys = []
        for direction in ("forward", "backward"):
            adjust = None
            if direction == "forward":
                res, poses = self._track_leg(arm, here, waypoints, q_now)
                if not res.success or res.q_traj is None or len(res.q_traj) == 0:
                    whys.append(f"forward: {res.status}"[:120])
                    continue
                q_traj = C.as_tensor(res.q_traj).float()
                segments = [(q_traj, self._leg_steps(poses, q_traj))]
            else:
                q_pe = q_now.clone()
                q_pe[idx] = q_end_plan[idx]
                back = [as_world_pose(w) for w in list(waypoints)[:-1]][::-1] + [here]
                res, poses = self._track_leg(arm, goal, back, q_pe)
                if not res.success or res.q_traj is None or len(res.q_traj) == 0:
                    whys.append(f"backward: {res.status}"[:120])
                    continue
                q_traj, poses = C.as_tensor(res.q_traj).float().flip(0), poses[::-1]
                gap0 = float(th.max(th.abs(q_traj[0][idx] - q_now[idx])))
                if gap0 > self.cfg.transit_dip_adjust_max_rad:
                    whys.append(f"backward: reaches the hand {gap0:.2f} rad from its joints (> {self.cfg.transit_dip_adjust_max_rad:.2f})")
                    continue
                q_adj = q_now.clone()
                q_adj[idx] = q_traj[0][idx]
                n_adj = max(2, int(math.ceil(gap0 / 0.05)))
                ramp = th.stack([q_now + (q_adj - q_now) * (k / n_adj) for k in range(n_adj + 1)])
                if bool(C.as_tensor(self.backend.check_collision(ramp, sync_world=False)).any()):
                    whys.append("backward: the in-place adjustment collides")
                    continue
                fk_batch = getattr(self.backend, "fk_batch", None)
                moved = 0.0
                if fk_batch is not None:
                    p_adj = C.as_tensor(fk_batch(M.EEF_LINK[arm], ramp, f"{arm}_arm")[0]).float()
                    moved = float(th.linalg.norm(p_adj - here[0].float(), dim=1).max())
                if moved > self.cfg.transit_dip_adjust_hand_m:
                    whys.append(f"backward: the in-place adjustment moves the hand {moved * 1000:.0f} mm")
                    continue
                q_traj[0] = q_adj
                segments = [(th.stack([q_now, q_adj]), [max(1, int(math.ceil(gap0 / (self.cfg.transit_dip_adjust_rate * self.dt))))]),
                            (q_traj, self._leg_steps(poses, q_traj))]
                adjust = {"rad": round(gap0, 3), "hand_mm": round(moved * 1000, 1)}
            full = th.cat([seg[0] for seg in segments])
            if self._protect is not None:
                viol = self._protect_violations([arm], full, f"{arm}_arm", self._protect)
                if viol:
                    whys.append(f"{direction}: passes {viol[0].get('object')} at {viol[0].get('clearance_m')} m")
                    continue
            q_last = full[-1]
            gap = float(th.max(th.abs(q_last[idx] - q_end_plan[idx])))
            if gap > self.cfg.witness_refine_max_jump:
                whys.append(f"{direction}: ends {gap:.2f} rad from the plan's last configuration (another IK branch)")
                continue
            approach = list(D.get("approach") or [])
            if approach:
                frame = M.EEF_LINK[arm]
                dense_a = self._densify(goal, approach)
                res_a = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp} for wp in dense_a], q_start=q_last, pos_tol=self.cfg.diff_ik_pos_tol,
                                                   rot_tol=self.cfg.diff_ik_rot_tol, max_iters=self.cfg.diff_ik_max_iters, check_collisions=False, sync_world=False,
                                                   soft_pos_tol=self.cfg.diff_ik_soft_pos_tol, soft_rot_tol=self.cfg.diff_ik_soft_rot_tol)
                if not res_a.success:
                    whys.append(f"{direction}: the approach does not track from its end ({res_a.status})"[:160])
                    continue
            why = f"{direction}, ends {gap:.2f} rad from the plan's end" + (f", in-place adjustment {adjust['rad']} rad / {adjust['hand_mm']} mm first" if adjust else "")
            return segments, why
        return None, "; ".join(whys)

    def _dip_gate(self, arms: Sequence[str], res: Any, goal_pose: Any, q_goal: Optional[th.Tensor]) -> Generator[th.Tensor, None, bool]:
        """Owner 2026-09-30 (first-demo v3.0 F5): inside :meth:`transit_dip_gate`, a planned transit whose hand path dips more than
        ``transit_dip_max_m`` below both of its ends (:meth:`_plan_dip`, FK of the plan before it streams) is replaced by a Cartesian
        leg - straight to the goal, else up ``transit_dip_lift_m`` above the higher end, across, down - when :meth:`_cartesian_transit`
        accepts it. Returns True when the leg ran (nothing of the plan streams); False = the plan runs as before. Outside the gate, or
        for a plan that does not dip, nothing is computed beyond the FK."""
        D = self._dip
        if D is None or len(arms) != 1 or arms[0] != D["arm"] or res is None or getattr(res, "q_traj", None) is None:
            return False
        arm = arms[0]
        rec: Dict[str, Any] = {"fired": False, "threshold_m": self.cfg.transit_dip_max_m}
        D["record"].clear()
        D["record"].update(rec)
        rec = D["record"]
        dip = self._plan_dip(arm, res.q_traj)
        if dip is None:
            rec["why"] = "no FK of the plan"
            return False
        rec.update({"dip_m": round(dip[0], 4), "z_start": round(dip[1], 3), "z_end": round(dip[2], 3), "z_min": round(dip[3], 3)})
        if dip[0] <= self.cfg.transit_dip_max_m:
            rec["why"] = f"the plan dips {dip[0] * 100:.1f} cm below its lower end (<= {self.cfg.transit_dip_max_m * 100:.0f} cm): planned as before"
            return False
        q_end = C.as_tensor(res.q_traj[-1]).reshape(-1).float() if q_goal is None else C.as_tensor(q_goal).reshape(-1).float()
        if goal_pose is None:
            try:
                goal_pose = self.backend.fk([M.EEF_LINK[arm]], q_end)[M.EEF_LINK[arm]]
            except Exception as e:  # noqa: BLE001
                rec["why"] = f"no goal pose ({e!r})"[:160]
                return False
        gp, gq = (x.float() for x in as_world_pose(goal_pose))
        here = self.eef_pose(arm)
        z_top = max(float(here[0][2]), float(gp[2])) + float(self.cfg.transit_dip_lift_m)
        legs = [("straight", [(gp, gq)]),
                ("up-over-down", [(th.tensor([float(here[0][0]), float(here[0][1]), z_top]), here[1]), (th.tensor([float(gp[0]), float(gp[1]), z_top]), gq), (gp, gq)])]
        tried = []
        for name, wps in legs:
            try:
                segments, why = self._cartesian_transit(arm, wps, q_end)
            except Exception as e:  # noqa: BLE001 - a backend without the checks: the plan runs as before
                segments, why = None, f"no check ({e!r})"[:160]
            tried.append(f"{name}: {why}")
            if segments is None:
                continue
            for q_traj, steps in segments:
                yield from self.execute_joint_trajectory(self._retime(q_traj, steps), arms=[arm])
            rec.update({"fired": True, "leg": name, "tried": tried,
                        "why": f"the plan dipped {dip[0] * 100:.1f} cm below its lower end: replaced by the {name} Cartesian leg"})
            log.info("arm_to(%s): the planned transit dips %.1f cm below its lower end -> %s Cartesian leg instead (%s)", arm, dip[0] * 100, name, "; ".join(tried))
            return True
        rec.update({"tried": tried, "why": f"the plan dips {dip[0] * 100:.1f} cm but no Cartesian leg qualifies: planned as before"})
        log.info("arm_to(%s): the planned transit dips %.1f cm below its lower end; no Cartesian leg qualifies (%s): the plan runs", arm, dip[0] * 100, "; ".join(tried))
        return False

    def _protect_gate(self, arms: Sequence[str], emb: str, res: Any, goal_pose: Any, q_goal: Optional[th.Tensor],
                      replan: Callable[[], Any]) -> Generator[th.Tensor, None, Tuple[bool, Any]]:
        """Before ``arm_to`` streams ``res``: inside a protect context, check it; a plan that clears is kept (``(False, res)``).
        One that does not (or every plan, after a run-time protected verdict) is re-planned via the point above the goal
        (``(True, res)``: done, nothing left to execute), else re-planned directly up to ``protect_replans`` times (the first
        plan that clears, else the one that keeps farthest from the protected objects - the run-time check is the backstop).
        A pre-grasp plan that clears but detours past the via-above (``_plan_detour``) takes the via-above as well; when that
        fails before the arm moved, the clearing plan is used."""
        P = self._protect
        if P is None or len(arms) != 1:
            return False, res
        arm = arms[0]
        viol = [] if P["force"] else self._protect_violations(arms, res.q_traj, emb, P)
        detour = self._plan_detour(arm, res.q_traj) if not viol and not P["force"] and P.get("leg") == "pre_grasp" else None
        if not viol and not P["force"] and detour is None:
            return False, res
        if goal_pose is None and q_goal is not None:
            try:
                goal_pose = self.backend.fk([M.EEF_LINK[arm]], q_goal)[M.EEF_LINK[arm]]
            except Exception:  # noqa: BLE001
                goal_pose = None
        self._protect = None  # the remedy's own motions are checked explicitly, never recursively
        try:
            why = "no goal pose"
            if goal_pose is not None:
                done, why = yield from self._via_above(arm, emb, goal_pose, q_goal, P)
                if done:
                    self._protect_record(P, arm, viol or ([detour] if detour else []), "via-above" + (" (forced after a run-time protected verdict)" if P["force"] else "")
                                         + (f" (the clearing plan detoured: hand path {detour['path_m']} m for {detour['straight_m']} m, {detour['rise_m']} m above both ends)" if detour else ""), why)
                    return True, res
            best, best_clear = res, (min(v["clearance_m"] for v in viol) if viol else -1.0)
            idx_a = self.robot.arm_control_idx[arm]
            moved = float(th.max(th.abs(self.backend.q_full()[idx_a] - C.as_tensor(res.q_traj[0]).reshape(-1)[idx_a]))) > 1e-3
            if detour is not None and not moved:
                self._protect_record(P, arm, [detour], "kept", f"via-above failed ({why}); the detouring plan clears the protected objects and is used")
                return False, res
            if moved:
                best, best_clear = None, float("-inf")  # the via-above lift moved the arm: the original plan no longer starts here
            for k in range(int(self.cfg.protect_replans)):
                try:
                    r2 = replan()
                except SkillError:
                    r2 = None
                if r2 is None or not getattr(r2, "success", False):
                    continue
                v2 = self._protect_violations(arms, r2.q_traj, emb, P)
                if not v2:
                    self._protect_record(P, arm, viol, f"re-plan {k + 1}", f"via-above failed ({why}); the direct re-plan clears")
                    return False, r2
                c2 = min(v["clearance_m"] for v in v2)
                if c2 > best_clear:
                    best, best_clear = r2, c2
            if best is None:
                self._protect_record(P, arm, viol, "failed", f"via-above failed ({why}) after the lift; no re-plan from there")
                raise SkillError("plan_failed", f"arm_to({arm}): protected-object re-plan failed ({why}); no plan from the lifted hand", stage="protected", protected=list(viol))
            self._protect_record(P, arm, viol, "kept", f"via-above failed ({why}); no re-plan clears - the plan keeping {best_clear:.4f} m is used, the run-time check stays")
            return False, best
        finally:
            self._protect = P

    def _plan_via_posture(self, emb: str, arm: str, q_start: th.Tensor, q_goal: th.Tensor, relax, why: str):
        """The witness ``q_goal`` reached through an arm posture (``cfg.witness_via_postures``, in turn): start -> posture -> witness
        as two joint-space plans; the first posture both legs plan through wins and is recorded (``witness_via_posture_events``).
        None when none does - the caller then raises ``plan_failed`` as before."""
        import copy

        idx = self.robot.arm_control_idx[arm]
        tried: List[str] = []
        for name in self.cfg.witness_via_postures:
            q_via = q_start.clone()
            if name == "ready":
                q_via[idx] = th.tensor(M.ARM_READY_Q[arm], dtype=q_via.dtype)
            elif name == "untucked":
                q_via[idx] = C.as_tensor(self.robot.untucked_default_joint_pos)[idx].to(q_via.dtype)
            else:
                continue
            legs: List[th.Tensor] = []
            q_from, plan_time, r = q_start, 0.0, None
            for q_to in (q_via, q_goal):
                with (relax(emb, q_from, (arm,)) if relax is not None else nullcontext([])) as relaxed:
                    if relax is not None:
                        self.last_relaxed_links = list(relaxed)
                    r = self.backend.plan_joint(emb, q_to, q_start=q_from, max_attempts=self.cfg.plan_attempts, sync_world=False)
                if not r.success:
                    break
                legs.append(C.as_tensor(r.q_traj))
                plan_time += float(getattr(r, "plan_time", 0.0) or 0.0)
                q_from = legs[-1][-1].reshape(-1)
            tried.append(f"{name}: " + ("ok" if len(legs) == 2 else f"leg {len(legs) + 1} failed"))
            if len(legs) == 2:
                self.witness_via_posture_events.append({"arm": arm, "posture": name, "why": why, "tried": list(tried)})
                log.warning("arm_to(%s): VIA POSTURE - the joint-space plan to the witness failed every round and %s; the witness is reached through "
                            "the %s posture (start -> %s -> witness)", arm, why, name, name)
                out = copy.copy(r)
                out.success, out.q_traj, out.status, out.plan_time = True, th.cat([legs[0], legs[1][1:]]), f"via the {name} posture", plan_time
                return out
        log.info("arm_to(%s): no posture carries the witness transit (%s)", arm, "; ".join(tried))
        return None

    def arm_to(self, arms: Union[str, Sequence[str]], pose: Any = None, q: Any = None, ignore_objects=(), q_hint: Any = None,
               max_excursion: Optional[float] = None, ignore_links=(), accept_other_branch: bool = False) -> Iterator[th.Tensor]:
        """cuRobo trajopt of the arm(s) with the trunk locked, then streamed. Exactly one of ``pose`` (a world pose
        for one arm, or ``{arm: pose}``) or ``q`` (a 28-D vector, ``{joint_name: value}`` or ``{arm: q7}``) is the
        target. Raises ``plan_failed`` when no trajectory is found (the sampler proved the key poses, not the
        trajectory between them, §3.9).

        ``q_hint`` (a 28-D configuration, the sampler's IK witness of ``pose``): the arm is planned in **joint space**
        to the witness's arm joints — the IK branch the sampler validated the following approach / descent from —
        and the residual to ``pose`` (the object settled a few mm away from its recorded pose) is closed with a short
        ``arm_along``. Without a hint cuRobo picks its own IK branch for the pose goal; ``max_excursion`` (rad) then
        bounds the largest joint change of the plan — a finishing motion of a few centimetres must not become an elbow
        flip (the pilot's french-fries finish swung the hand out of the camera frame).

        ``accept_other_branch`` (owner 2026-09-26, a push's pre-contact: its legs are Cartesian, tracked from wherever the hand is):
        when every joint-space round to the witness fails and the pose-goal plan lands in another IK branch, that plan is kept
        instead of failing the attempt - only in that situation, which raised ``plan_failed`` before."""
        arms = (arms,) if isinstance(arms, str) else tuple(arms)
        emb = self._emb_for(arms)
        self._protect_ignores = (tuple(ignore_objects or ()), tuple(ignore_links or ()))  # the via-above re-plan keeps this world
        try:
            self.backend.update_world(ignore_objects=ignore_objects, ignore_links=ignore_links)
        except TypeError:  # a backend without link-level ignores (the test stubs)
            self.backend.update_world(ignore_objects=ignore_objects)
        q_start = self.backend.q_full()
        if (pose is None) == (q is None):
            raise ValueError("arm_to needs exactly one of pose / q")
        relax = getattr(self.backend, "static_links_relaxed", None)
        if pose is not None and q_hint is not None and len(arms) == 1:
            arm = arms[0]
            q28 = C.as_tensor(q_hint).reshape(-1)
            q_goal = q_start.clone()
            idx = self.robot.arm_control_idx[arm]
            q_goal[idx] = q28[idx]
            # the witness is the sampler's IK solution in *its* frame (the recorded stance base, its session's settle):
            # its FK misses the recorded pose by 3-15 mm per instance (nextto 3-5, touching 9-13, axial 14.6 mm, a
            # constant world offset per sidecar). Refine it: the collision-aware IK of the exact pose seeded at the
            # witness (same branch when the joint distance stays small), else the witness as recorded
            solve_ik = getattr(self.backend, "solve_ik", None)
            if solve_ik is not None:
                try:
                    q_ik = solve_ik(emb, {M.EEF_LINK[arm]: as_world_pose(pose)}, q_seed=q_goal, sync_world=False)
                except Exception as e:  # noqa: BLE001
                    log.info("arm_to(%s): witness refinement IK unavailable (%r)", arm, e)
                    q_ik = None
                if q_ik is not None:
                    q_ik = C.as_tensor(q_ik).reshape(-1)
                    jump = float(th.max(th.abs(q_ik[idx] - q_goal[idx])))
                    if jump <= self.cfg.witness_refine_max_jump:
                        log.info("arm_to(%s): witness refined by IK of the recorded pose (joint distance %.3f rad)", arm, jump)
                        q_goal[idx] = q_ik[idx]
                    else:
                        log.info("arm_to(%s): IK of the recorded pose lies %.2f rad from the witness (> %.2f): keeping the witness", arm, jump, self.cfg.witness_refine_max_jump)
            if relax is not None:
                with relax(emb, q_start, arms) as relaxed:
                    self.last_relaxed_links = list(relaxed)
                    res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            else:
                res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
            if not res.success:
                # owner 2026-09-25 (teacup pod demo 0925-172723): the joint-space plan is stochastic (trajopt seeds) - the same witness
                # planned fine on the local GPU; a second round of attempts comes before any change of branch
                for k_retry in range(self.cfg.witness_plan_retries):
                    if relax is not None:
                        with relax(emb, q_start, arms) as relaxed:
                            self.last_relaxed_links = list(relaxed)
                            res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
                    else:
                        res = self.backend.plan_joint(emb, q_goal, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
                    if res.success:
                        log.info("arm_to(%s): joint-space plan to the witness succeeded on retry %d", arm, k_retry + 1)
                        break
            if not res.success:
                # 2026-09-18 (blender eyedvd pod demo 0918-222128): the joint-space plan to the witness failed twice with a final pose
                # error of 0.0 mm - the goal is fine, the path to that configuration is not (a witness on a joint limit under the
                # margin-0 test setting, or a sweep through the target). Fall back to the pose goal in cuRobo's own branch before
                # giving up; the legs that follow start from wherever the hand is (arm_along tracks from the current configuration)
                diag_js = dict(res.diagnostics)
                q_goal_js = q_goal.clone()
                targets = {M.EEF_LINK[arm]: as_world_pose(pose)}
                if relax is not None:
                    with relax(emb, q_start, arms) as relaxed:
                        self.last_relaxed_links = list(relaxed)
                        res = self.backend.plan_pose(emb, targets, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
                else:
                    res = self.backend.plan_pose(emb, targets, q_start=q_start, max_attempts=self.cfg.plan_attempts, sync_world=False)
                via = None
                if not res.success:
                    via = self._plan_via_posture(emb, arm, q_start, q_goal_js, relax, f"the pose-goal fallback failed too ({res.status})")
                    if via is None:
                        raise SkillError("plan_failed", f"arm_to({arm}, witness q): {diag_js.get('trajopt_verdict') or 'joint-space plan failed'}; pose-goal fallback: {res.status}",
                                         diagnostics={"joint_goal": diag_js, "pose_goal": dict(res.diagnostics)}, stage="trajopt")
                else:
                    q_goal = C.as_tensor(res.q_traj[-1]).reshape(-1) if res.q_traj is not None and len(res.q_traj) else q_goal
                    branch_gap = float(th.max(th.abs(q_goal[idx] - q28[idx])))
                    if branch_gap > self.cfg.witness_branch_max_jump and accept_other_branch:
                        # owner 2026-09-26 (axial 0926-043254 wave 10, 0926-050248 wave 11, 0926-051919 wave 12: the push pre-contact witness
                        # 3.10 rad from cuRobo's branch - from the seeded IK too - trajopt reaches it only through the board and the push
                        # arc is untrackable from it even when it is reached): the caller's legs start from wherever the hand is (a push:
                        # Cartesian, closed-loop), so cuRobo's own branch is kept - where this raised before
                        ev = {"arm": arm, "branch_gap_rad": round(branch_gap, 3), "rounds": 1 + self.cfg.witness_plan_retries}
                        self.witness_branch_accepted_events.append(ev)
                        log.warning("arm_to(%s): OTHER BRANCH accepted - the joint-space plan to the witness failed %d rounds; the pose-goal plan ends %.2f rad "
                                    "from it and is kept (the caller's legs are tracked from wherever the hand is)", arm, ev["rounds"], branch_gap)
                    elif branch_gap > self.cfg.witness_branch_max_jump:
                        # owner 2026-09-25: with the sidecar's legs chained from this witness (the approach tracked from the pre-grasp, the
                        # finish aimed at the grasp witness), a pre pose reached in another branch fails the leg that follows anyway
                        # (0925-172723: pre-grasp 3.13 rad off the witness, the approach stalled, its finish 3.46 rad away, 'the plan swings
                        # a joint by 3.21 rad'); failing here lets the executor's attempt retry re-plan from the checkpoint instead
                        via = self._plan_via_posture(emb, arm, q_start, q_goal_js, relax, f"the pose-goal plan ends {branch_gap:.2f} rad from it")
                        if via is None:
                            raise SkillError("plan_failed", f"arm_to({arm}, witness q): {diag_js.get('trajopt_verdict') or 'joint-space plan failed'} after {1 + self.cfg.witness_plan_retries} rounds; "
                                             f"the pose-goal plan ends {branch_gap:.2f} rad from the witness (> {self.cfg.witness_branch_max_jump:.2f}: another IK branch, the chained legs would fail)",
                                             diagnostics={"joint_goal": diag_js, "pose_goal": dict(res.diagnostics)}, stage="trajopt", excursion=branch_gap)
                    if via is None:
                        log.warning("arm_to(%s): joint-space plan to the witness failed (%s); planned to the pose in cuRobo's own branch instead (joint distance to the witness %.2f rad)",
                                    arm, diag_js.get("trajopt_verdict"), branch_gap)
                if via is not None:
                    res, q_goal = via, q_goal_js
            if max_excursion is not None:
                idx = self.robot.arm_control_idx[arm]
                exc = float(th.max(th.abs(q_goal[idx] - q_start[idx])))
                if exc > float(max_excursion):
                    raise SkillError("plan_failed", f"arm_to({arm}, witness q): the recorded IK witness lies {exc:.2f} rad away in joint space (> {max_excursion:.2f} allowed for a "
                                     f"finishing motion): the sidecar's pre-pose and end-pose witnesses are in different IK branches", stage="trajopt", excursion=exc)
            log.info("arm_to(%s): %d waypoints to the recorded IK witness (%.1fs), planned in %.2fs", arm, len(res.q_traj), res.duration, res.plan_time)
            q_goal_w = q_goal
            if (yield from self._dip_gate(arms, res, as_world_pose(pose), q_goal_w)):
                return
            handled, res = yield from self._protect_gate(arms, emb, res, as_world_pose(pose), q_goal_w,
                                                         lambda: self.backend.plan_joint(emb, q_goal_w, q_start=self.backend.q_full(), max_attempts=self.cfg.plan_attempts, sync_world=False))
            if handled:
                return
            yield from self.execute_joint_trajectory(res.q_traj, arms=arms)
            target = as_world_pose(pose)
            cur = self.eef_pose(arm)
            dp, da = float(th.linalg.norm(cur[0] - target[0])), C.quat_angle(cur[1], target[1])
            if dp > 2e-3 or da > 0.01:
                log.info("arm_to(%s): witness residual %.1f mm / %.3f rad, closing it along a straight segment", arm, dp * 1000, da)
                # a finishing motion (``max_excursion`` given: the end pose of an approach / descent) tolerates less
                accept_pos = self.cfg.witness_residual_accept_pos if max_excursion is None else self.cfg.witness_residual_accept_pos_finish
                if dp <= accept_pos and da <= self.cfg.witness_residual_accept_rot:
                    # the witness *is* the sampler's exact-IK solution of this pose (its tolerance); closing the last
                    # millimetres is a refinement that a witness sitting on a joint limit cannot make (touching box:
                    # right_arm_joint2 2.8 mrad from its limit, the first DLS step jumped 0.25 rad) - the hand stays at
                    # the validated configuration and the following leg starts from where the hand is
                    try:
                        yield from self.arm_along(arm, [target])
                    except SkillError as e:
                        log.warning("arm_to(%s): the %.1f mm / %.3f rad witness residual is left as is (%s)", arm, dp * 1000, da, e)
                        self.last_witness_residual = (dp, da)
                else:
                    try:
                        yield from self.arm_along(arm, [target])
                    except SkillError as e:
                        raise SkillError(e.reason, f"arm_to({arm}, witness q): closing the {dp * 1000:.1f} mm / {da:.3f} rad residual between the recorded witness and its pose failed: {e.message}"
                                         f" (the recorded pose is not reachable from the witness's configuration; the sampler's witnesses miss their poses by a constant offset per sidecar)",
                                         stage="witness_residual", residual_m=dp, residual_rad=da) from e
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
        if self._dip is not None and len(arms) == 1:
            if (yield from self._dip_gate(arms, res, targets[M.EEF_LINK[arms[0]]] if pose is not None else None, None if pose is not None else q_goal)):
                return
        if self._protect is not None and len(arms) == 1:
            if pose is not None:
                goal_g = targets[M.EEF_LINK[arms[0]]]
                q_goal_g = C.as_tensor(res.q_traj[-1]).reshape(-1)
                replan = lambda: self.backend.plan_pose(emb, targets, q_start=self.backend.q_full(), max_attempts=self.cfg.plan_attempts, sync_world=False)  # noqa: E731
            else:
                goal_g, q_goal_g = None, q_goal
                replan = lambda: self.backend.plan_joint(emb, q_goal_g, q_start=self.backend.q_full(), max_attempts=self.cfg.plan_attempts, sync_world=False)  # noqa: E731
            handled, res = yield from self._protect_gate(arms, emb, res, goal_g, q_goal_g, replan)
            if handled:
                return
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

    def along_reaches(self, arm: str, waypoints: Sequence[Any]) -> bool:
        """Whether :meth:`arm_along` would track ``waypoints`` from the current configuration to the last one (the same
        densification and differential-IK settings, nothing executed): the Cartesian leg keeps the IK branch it starts in."""
        waypoints = list(waypoints)
        if not waypoints:
            return True
        dense = self._densify(self.eef_pose(arm), waypoints)
        res = self.backend.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: wp} for wp in dense], pos_tol=self.cfg.diff_ik_pos_tol, rot_tol=self.cfg.diff_ik_rot_tol,
                                         max_iters=self.cfg.diff_ik_max_iters, check_collisions=False, sync_world=False,
                                         soft_pos_tol=self.cfg.diff_ik_soft_pos_tol, soft_rot_tol=self.cfg.diff_ik_soft_rot_tol)
        return bool(res.success)

    def arm_along(self, arm: str, waypoints: Sequence[Any], speed: Optional[float] = None, finish: Optional["ApproachFinish"] = None,
                  soft_pos_tol: Optional[float] = None, loaded: bool = False, partial: bool = False, converge: bool = True,
                  check_collisions: bool = False, limit_margin: float = 0.0, ignore_objects: Sequence[Any] = (), soft_rot_tol: Optional[float] = None,
                  refine_jumps: bool = True, ignore_links: Sequence[Any] = ()) -> Iterator[th.Tensor]:
        """Track a given eef waypoint list with damped-least-squares differential IK on the arm's joints (trunk and
        base locked), streamed at ``speed`` (m/s; ``cartesian_speed``). The list is densified to the tracking
        resolution and every waypoint is reached in order; no collision veto by default (contact motions must touch —
        the QA collision monitor judges contacts). Raises ``plan_failed`` (stage ``diff_ik``) when a waypoint cannot be
        reached from the current configuration.

        ``check_collisions`` (free-space legs, 2026-09-22): the joint path is validated against the world (minus
        ``ignore_objects`` and the collision pieces ``ignore_links``) and the robot itself; the collision-free prefix is executed and the rest finished by
        ``arm_to`` when a ``finish`` is given (or kept as is for a ``partial`` leg), else the leg fails ``plan_failed``
        (stage ``collision``). ``limit_margin`` (rad) keeps the differential IK this far from the joint limits (the
        0918-000750 retreat folded the elbow into its own upper arm 3 mrad short of the limit and the drive fought that
        contact for the rest of the episode - the robot never counted as settled).

        ``finish`` (owner rule 2026-09-16): when the differential IK **stalls** — no joint at a limit, the residual
        below ``finish.max_stall`` — with at most ``finish.max_remaining`` of the segment left, the converged prefix
        is executed and the rest is planned by ``arm_to`` to ``finish.pose`` (to the sidecar's IK witness
        ``finish.q_hint`` when there is one, else cuRobo's own IK branch) instead of failing: the stall is the reach
        boundary of the *branch* the straight segment started in (more iterations do not help: 30 → 100 moved a
        7.2 mm stall to 7.0 mm), while the pose itself is reachable (the sampler's exact IK).

        ``converge=False``: the streamed trajectory returns after its last waypoint without the joint-convergence wait
        (:meth:`execute_joint_trajectory`); the caller watches the object it moves.

        ``refine_jumps``: a leg failing on a joint jump (> ``max_joint_step`` between two waypoints) is solved once more at half
        the waypoint spacing; off for a caller that handles a jump itself."""
        waypoints = list(waypoints)
        if not waypoints:
            return
        start = self.eef_pose(arm)
        dense = self._densify(start, waypoints)
        frame = M.EEF_LINK[arm]
        # a loaded arm (a pad pressing on its object) is wound up: PhysX reports joint positions whose FK lies past the
        # eef link (touching push 0916-202403: 2.9 cm deeper into the box). A leg planned from those joints snaps the
        # link onto their FK the moment the load releases - the retreat pushed the box another 27.6 mm. Unwind first:
        # the first waypoint is the link's own pose, so the joints are re-targeted onto where the link actually is.
        deflection = self._deflection(arm)
        if deflection is not None and deflection > self.cfg.unwind_min_deflection:
            log.info("arm_along(%s): the arm is wound up %.1f mm under load (FK of the joints vs the eef link): unwinding onto the link pose first", arm, deflection * 1000)
            dense = [start] + dense
        if check_collisions:
            try:
                self.backend.update_world(ignore_objects=ignore_objects, **links_kw(ignore_links))
            except Exception as e:  # noqa: BLE001 - a backend without a world model (the test stubs) checks nothing
                log.debug("arm_along(%s): world update for the collision check failed (%r)", arm, e)
        extra_kw: Dict[str, Any] = {"limit_margin": float(limit_margin)} if limit_margin > 0.0 else {}
        # a collision-checked leg fails on NEW collisions only (owner 2026-09-24 item 2): the self-collision pairs its start
        # already violates against another arm's link / attached object leave the sphere model for this leg
        relax_start = getattr(self.backend, "start_collisions_relaxed", None) if check_collisions else None
        relax_ctx = relax_start(f"{arm}_arm", self.backend.q_full(), [arm]) if relax_start is not None else nullcontext([])
        with relax_ctx as relaxed_start:
            if relaxed_start:
                self.last_relaxed_links = list(relaxed_start)
            solve_kw = dict(pos_tol=self.cfg.diff_ik_pos_tol, rot_tol=self.cfg.diff_ik_rot_tol, max_iters=self.cfg.diff_ik_max_iters, check_collisions=bool(check_collisions),
                            sync_world=False, soft_pos_tol=self.cfg.diff_ik_soft_pos_tol if soft_pos_tol is None else float(soft_pos_tol),
                            soft_rot_tol=self.cfg.diff_ik_soft_rot_tol if soft_rot_tol is None else float(soft_rot_tol), **extra_kw)
            res = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp} for wp in dense], **solve_kw)
            if refine_jumps and not res.success and str(res.status).startswith("joint jump"):
                # a fast but continuous turn near a singularity exceeds max_joint_step between two waypoints (shampoo 0927-183304 on
                # 052de5eb: the carry to the floor jumped 0.23 rad at waypoint 44, both attempts); a branch flip jumps at any spacing.
                # ``refine_jumps=False``: the caller handles a jump itself (the via-above's rise plans a lift instead - on the sandal
                # that plan left the arm where the approach tracks; tracking the rise at half spacing did not, 0927-192757 on 5174c989)
                finer: List[Pose] = []
                prev_wp = start
                for wp in dense:
                    finer.extend(C.line_waypoints(prev_wp, wp, 2))
                    prev_wp = wp
                res2 = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp} for wp in finer], **solve_kw)
                if res2.success:
                    log.info("arm_along(%s): %s at the tracking resolution; the leg tracks at half the waypoint spacing (%d waypoints)", arm, res.status, len(finer))
                    res, dense = res2, finer
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
            at_limit = bool(diag.get("at_lower") or diag.get("at_upper"))
            if diag.get("stage") == "collision":
                # a straight free-space leg whose later waypoints collide (the robot with itself or the world): the
                # collision-free prefix is executed; the rest is planned around by arm_to when a finish is given
                n_ok = int(diag.get("converged", 0))
                pairs = ", ".join("-".join(p["links"]) for p in (diag.get("self_collision") or []))
                if n_ok > 0 and res.q_traj is not None and len(res.q_traj) > 1:
                    yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps[: len(res.q_traj) - 1]), arms=[arm], loaded=loaded)
                if finish is not None:
                    log.warning("arm_along(%s): %s -> executed the %d collision-free waypoints, finishing with arm_to (cuRobo, collision-checked)%s", arm, res.status, n_ok,
                                f" [{pairs}]" if pairs else "")
                    by = "witness" if finish.q_hint is not None else "curobo_ik"
                    try:
                        yield from self.arm_to(arm, pose=finish.pose, q_hint=finish.q_hint, ignore_objects=finish.ignore_objects, max_excursion=finish.max_excursion,
                                               **links_kw(getattr(finish, "ignore_links", ())))
                    except SkillError as e:
                        if finish.q_hint is None or e.details.get("excursion") is None:
                            raise
                        log.warning("arm_along(%s): finish toward the witness rejected (%s) -> finishing by cuRobo IK in the current branch", arm, e.message)
                        by = "curobo_ik_fallback"
                        yield from self.arm_to(arm, pose=finish.pose, q_hint=None, ignore_objects=finish.ignore_objects, max_excursion=finish.max_excursion,
                                               **links_kw(getattr(finish, "ignore_links", ())))
                    finish.used = True
                    finish.info = {"collision": diag.get("kind"), "self_collision": diag.get("self_collision"), "converged": n_ok, "n_dense": len(dense), "by": by}
                    return
                if partial:
                    log.warning("arm_along(%s): %s -> executed the %d collision-free waypoints only (best-effort leg)", arm, res.status, n_ok)
                    return
                raise SkillError("plan_failed", f"arm_along({arm}): {res.status}", stage="collision", n_waypoints=len(waypoints), fail_index=k, self_collision=diag.get("self_collision"))
            # a stall at a joint limit is finished only toward a recorded witness (a known configuration of the end
            # pose; the touching approach slides along right_arm_joint2's limit, as the sampler's own approach witnesses
            # do), a free finish from a limit stall would just pick another branch
            if finish is not None and diag.get("stage") == "diff_ik" and k is not None and stall is not None and (not at_limit or finish.q_hint is not None):
                remaining = sum(float(th.linalg.norm(dense[i][0] - dense[i - 1][0])) for i in range(max(k, 1), len(dense)))
                remaining += stall
                if stall <= finish.max_stall and remaining <= finish.max_remaining:
                    log.warning("arm_along(%s): %s -> executing the %d converged waypoints and finishing the last %.1f cm with arm_to (%s)", arm, res.status, int(diag.get("converged", 0)), remaining * 100,
                                "IK witness" if finish.q_hint is not None else "cuRobo IK")
                    if res.q_traj is not None and len(res.q_traj) > 1:
                        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps[: len(res.q_traj) - 1]), arms=[arm])
                    by = "witness" if finish.q_hint is not None else "curobo_ik"
                    try:
                        yield from self.arm_to(arm, pose=finish.pose, q_hint=finish.q_hint, ignore_objects=finish.ignore_objects, max_excursion=finish.max_excursion,
                                               **links_kw(getattr(finish, "ignore_links", ())))
                    except SkillError as e:
                        # the recorded witness sits in another IK branch (touching 0917-203700 retreat: 3.29 rad away): the
                        # plan is rejected before any motion, so finish without the hint - cuRobo's IK from the current
                        # configuration, still bounded by max_excursion (a stall at a limit keeps the witness-only rule)
                        if finish.q_hint is None or e.details.get("excursion") is None or at_limit:
                            raise
                        log.warning("arm_along(%s): finish toward the witness rejected (%s) -> finishing by cuRobo IK in the current branch", arm, e.message)
                        by = "curobo_ik_fallback"
                        yield from self.arm_to(arm, pose=finish.pose, q_hint=None, ignore_objects=finish.ignore_objects, max_excursion=finish.max_excursion,
                                               **links_kw(getattr(finish, "ignore_links", ())))
                    finish.used = True
                    finish.info = {"stall_m": stall, "remaining_m": remaining, "converged": int(diag.get("converged", 0)), "n_dense": len(dense), "by": by}
                    return
            if partial and diag.get("stage") == "diff_ik" and res.q_traj is not None and len(res.q_traj) > 1:
                # a best-effort leg (the residual re-push, rush 2026-09-17): the converged prefix is executed, the
                # caller measures what it achieved (touching 0917-205741: the extension stalled 3/5 at the reach limit)
                log.warning("arm_along(%s): %s -> executing the %d converged waypoints only (best-effort leg)", arm, res.status, int(diag.get("converged", 0)))
                yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps[: len(res.q_traj) - 1]), arms=[arm], loaded=loaded)
                return
            if finish is not None and diag.get("stage") == "diff_ik" and at_limit and finish.q_hint is None and res.q_traj is not None and len(res.q_traj) > 1:
                # a finishing leg that stalls AT a joint limit with no recorded witness (owner 2026-09-24, block design 0925-062801
                # retreat[15]: right_arm_joint2 at its lower limit 13 mm short, no in-branch configuration of the end pose exists
                # - the sampler's own tracking of that leg stalled the same way and recorded no witness): the converged prefix is
                # executed before the leg fails, exactly as after a rejected finish, so a free-space retreat's caller can accept
                # it as the partial retreat it is (``execute_retreat``, ``retreat_partial_accept_m``) instead of failing the action
                log.warning("arm_along(%s): %s -> a stall at a joint limit with no witness: executing the %d converged waypoints, then failing the leg (a retreat caller may accept the partial leg)",
                            arm, res.status, int(diag.get("converged", 0)))
                yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps[: len(res.q_traj) - 1]), arms=[arm], loaded=loaded)
            raise SkillError("plan_failed", f"arm_along({arm}): {res.status}", stage="diff_ik", n_waypoints=len(waypoints), **{k_: v for k_, v in diag.items() if k_ in ("fail_index", "pos_err", "rot_err")})
        P = self._protect
        if P is not None and P.get("along") in ("raise", "log") and res.q_traj is not None:
            # the carry (owner 2026-09-26): the tracked path with the carried object checked against the protected objects
            viol = self._protect_violations([arm], res.q_traj, f"{arm}_arm", P)
            if viol:
                if P["along"] == "raise":
                    self._protect_record(P, arm, viol, "via-above (caller)", "the tracked leg is not executed")
                    raise SkillError("plan_failed", f"arm_along({arm}): the {P.get('leg')} passes {viol[0]['object']} at {viol[0]['clearance_m']} m (threshold {viol[0]['threshold_m']} m)",
                                     stage="protected", protected=viol)
                self._protect_record(P, arm, viol, "kept", "the via-above leg still passes near; executed, the run-time check stays")
        yield from self.execute_joint_trajectory(self._retime(res.q_traj, steps), arms=[arm], loaded=loaded, converge=converge)

    def arm_arc_closed_loop(self, arm: str, centre: th.Tensor, axis: th.Tensor, target: Pose, joint_value: Callable[[], float], q_to: float,
                            speed: Optional[float] = None, band_rad: float = 0.005, lag_factor: float = 1.3, max_ik_failures: int = 10,
                            max_iters: int = 20) -> Generator[th.Tensor, None, Dict[str, Any]]:
        """Closed-loop follow of a rigidly held handle along the circle about (``centre``, ``axis``) until the joint it
        moves reads ``q_to`` (``joint_value()``, within ``band_rad``) or the hand reaches ``target``'s angle on the arc.

        Every control step: the FK pose of the MEASURED arm joints is rotated about the axis by one step's angle
        (``speed`` m/s at the hand's radius) and one differential-IK step from the MEASURED joints gives the commanded
        joint target. The target therefore never runs more than one step ahead of where the door lets the arm be:
        with 1e7-stiff, effort-capped drives (the BEHAVIOR-1K robot's; a policy cannot change them) any accumulated
        mismatch between an open-loop target path and the constraint saturates the drives at 18-55 N.m - ~100 N at
        the hand, which tipped the 47.6 kg chassis 0.8 deg and bent torso_joint1 2.6 deg on the 250 kg one
        (dishwasher_wide 0921-050314 / 0921-235247). The commanded lead is what the demo records and a policy imitates,
        so this compliance survives the evaluator's stock drives. Stops early on ``stop_requested`` (the goal band),
        when the commanded angle exceeds ``lag_factor`` x the expected one (the door does not follow: a blocked leaf
        is not pushed harder), or after ``max_ik_failures`` consecutive IK misses (``plan_failed`` / ``diff_ik``)."""
        frame = M.EEF_LINK[arm]
        ax = C.unit(C.as_tensor(axis).float())
        cen = C.as_tensor(centre).float()
        speed = self.cfg.follow_eef_speed if speed is None else float(speed)
        p_t, _ = C.as_pose(target)
        q0 = float(joint_value())
        expected = abs(float(q_to) - q0)
        sgn_joint = 1.0 if float(q_to) >= q0 else -1.0

        def radial(p):
            r = p - cen
            return r - ax * float(r @ ax)

        def signed_angle(ra, rb) -> float:  # both radial (perpendicular to the axis): the cross product lies along +-axis
            return math.atan2(float(th.cross(ra, rb, dim=0) @ ax), float(ra @ rb))

        q_full = self.robot.get_joint_positions().clone().float()
        p_fk, _ = as_world_pose(self.backend.fk([frame], q_full)[frame])
        r_now = radial(p_fk)
        radius = max(float(th.linalg.norm(r_now)), 0.05)
        omega = speed / radius
        step_ang = omega * self.dt
        arc_dir = signed_angle(r_now, radial(p_t))  # signed arc angle from the hand to the recorded target about the axis
        sgn = 1.0 if arc_dir >= 0 else -1.0
        info: Dict[str, Any] = {"steps": 0, "commanded_rad": 0.0, "ik_failures": 0, "stop": None, "q_from": q0, "q_to": float(q_to), "arc_to_target_rad": round(arc_dir, 4)}
        consecutive = 0
        max_steps = int(max(expected, abs(arc_dir)) / max(step_ang, 1e-6) * 3) + 60
        self.state.stop_base()
        while True:
            if self.stop_requested:
                info["stop"] = "goal_band"
                break
            q_now = float(joint_value())
            if sgn_joint * (q_now - float(q_to)) >= -band_rad:
                info["stop"] = "joint_at_target"
                break
            if info["commanded_rad"] >= lag_factor * max(expected, 1e-3) + 2 * step_ang:
                info["stop"] = "door_lags"  # the leaf did not follow the commanded angle: do not push harder
                break
            if info["steps"] >= max_steps:
                info["stop"] = "max_steps"
                break
            q_full = self.robot.get_joint_positions().clone().float()
            pose_fk = as_world_pose(self.backend.fk([frame], q_full)[frame])
            remaining = signed_angle(radial(pose_fk[0]), radial(p_t))
            d_ang = sgn * min(step_ang, max(abs(remaining), 0.25 * step_ang))  # never below a quarter step: the joint band decides the end
            wp = C.orbit_waypoints(pose_fk, cen, C.quat_from_axis_angle(ax, d_ang), n=1, frame="world")[-1]
            # tolerances far below the step (1.7 mm at 0.05 m/s): the path solver's 2 mm / 1 cm-soft acceptance took the
            # step's target as already reached and commanded no lead at all (0922-112650: 0.169 rad "commanded", door still)
            step_m = abs(d_ang) * radius
            res = self.backend.solve_ik_path(f"{arm}_arm", [{frame: wp}], q_start=q_full, pos_tol=min(1e-4, 0.05 * step_m), rot_tol=5e-4,
                                             max_iters=max_iters, check_collisions=False, sync_world=False, soft_pos_tol=0.25 * step_m, soft_rot_tol=2e-3)
            if res.success and res.q_traj is not None and len(res.q_traj) > 0:
                q_cmd = C.as_tensor(res.q_traj[-1]).reshape(-1).float()
                idx = self.robot.arm_control_idx[arm]
                dq = float(th.max(th.abs(q_cmd[idx] - q_full[idx])))
                if dq > self.cfg.diff_ik_joint_speed * self.dt:  # a singularity escape: bound the joint step, keep the direction
                    q_cmd = q_full.clone()
                    q_cmd[idx] = q_full[idx] + (C.as_tensor(res.q_traj[-1]).reshape(-1).float()[idx] - q_full[idx]) * (self.cfg.diff_ik_joint_speed * self.dt / dq)
                self.state.set_from_q(q_cmd, arms=[arm], trunk=False)
                info["commanded_rad"] += abs(d_ang)
                consecutive = 0
            else:
                consecutive += 1
                info["ik_failures"] += 1
                if consecutive >= max_ik_failures:
                    raise SkillError("plan_failed", f"arm_arc_closed_loop({arm}): {consecutive} consecutive differential-IK misses at q = {q_now:.3f} ({getattr(res, 'status', '')[:120]})",
                                     stage="diff_ik", n_waypoints=1)
            info["steps"] += 1
            yield self._emit()
        info["q_end"] = float(joint_value())
        info["deflection_m"] = self._deflection(arm)
        return info

    # ------------------------------------------------------------------ base

    def _base_reference(self, waypoints: Sequence[Any], v_lin: float, v_ang: float, a_max: Optional[float] = None, style: Optional[str] = None,
                        corner_stop: bool = False) -> th.Tensor:
        """Time-parameterise the polyline current pose → waypoints.

        ``style = "face_travel"`` (owner 2026-09-22, camera relocalization): the shared ``robot.base_path`` schedule —
        the polyline simplified within one map cell (the pilot's dense grid records: 28 cells, headings alternating
        0° / 45°, a speed dip and a zig-zag at every stair), fillets at the corners, the body facing the way with its
        turns spread over windows that keep the cruise speed, the end turns on the spot above 90° and blended below —
        with the lateral-acceleration and yaw limits through the arcs (``BP.time_parameterise``; the 0916-141035
        record: 36.8 s and 18 % of the time at cruise → 25 s with the drive 90 % at cruise).

        Otherwise (the recorded style) the **corner-aware trapezoidal speed profile** on the polyline as recorded:
        ``v_lin`` on the straights, ``sqrt(2·a_max·path_corner_cut) / sin(θ/2)`` through a heading change θ, zero at
        the end, ramps at ``a_max`` (the tracker's own limit), every segment slow enough for the yaw change toward the
        waypoint's recorded heading to stay within ``v_ang``; the yaw is blended linearly along each segment. A
        reference the tracker can follow within its acceleration limit is what keeps the base inside the corridor of
        a recorded path (the pilot's obstructed demos cut 45–63° corners by 5–8 cm at constant speed and touched the
        obstacle; 25 cm off after a 90° turn). ``corner_stop`` (owner 2026-10-13, an axis-aligned leg: forward / backward
        or left / right moves one after the other): zero speed at every interior waypoint, held ``corner_dwell_s``."""
        a_max = float(self.cfg.base_tracker.a_max[0] if a_max is None else a_max)
        pts = [self.base_pose2d()] + [as_pose2d(w) for w in waypoints]
        if style == BP.FACE_TRAVEL:
            if corner_stop:
                raise SkillError("execution", "track_path: corner_stop drives the recorded polyline, not the face_travel schedule")
            path = BP.BasePath.build(pts, self.cfg.face_travel)
            return BP.time_parameterise(path, self.dt, v_lin, v_ang, a_lin=a_max, a_yaw=float(self.cfg.base_tracker.a_max[2]), ds=float(self.cfg.path_profile_ds))
        if style not in (None, BP.RECORDED):
            raise SkillError("execution", f"track_path: unknown style {style!r}")
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
            if heads[i - 1] is None or heads[i] is None or corner_stop:
                v_corner[i] = 0.0  # a pure rotation at this node (or a corner stop): stop, turn, go on
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
        a_dec = min(a_max, float(self.cfg.corner_stop_decel)) if corner_stop else a_max
        for j in range(1, len(grid)):
            v[j] = min(float(v[j]), math.sqrt(float(v[j - 1]) ** 2 + 2.0 * a_max * (grid[j] - grid[j - 1])))
        for j in range(len(grid) - 2, -1, -1):
            v[j] = min(float(v[j]), math.sqrt(float(v[j + 1]) ** 2 + 2.0 * a_dec * (grid[j + 1] - grid[j])))
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

        n_hold = int(math.ceil(float(self.cfg.corner_dwell_s) / self.dt)) if corner_stop else 0

        # -- sample at dt; the pure-rotation segments are inserted as dwell time where the arc length passes them, the corner
        # holds of ``corner_stop`` where it passes an interior waypoint
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
            for m in range(max(k_prev + 1, 1), k_new + 1):
                samples.extend([[float(xy[m][0]), float(xy[m][1]), yaws[m]]] * n_hold)
            k_prev = max(k_prev, k_new)
            samples.append(pose_at_s(s))
        for m in range(k_prev + 1, n_seg):
            if float(seg[m]) < 1e-6:
                samples.extend(dwell(m))
        samples.append([float(xy[-1][0]), float(xy[-1][1]), yaws[-1]])
        return th.tensor(samples, dtype=th.float32)

    def track_path(self, waypoints: Sequence[Any], posture: Optional[str] = None, tracker_cfg: Optional[BaseTrackerConfig] = None,
                   style: Optional[str] = None, corner_stop: bool = False, on_step: Optional[Callable[[], bool]] = None) -> Iterator[th.Tensor]:
        """Follow a recorded ``[x, y, yaw]`` waypoint list with the joystick-like tracker (feed-forward + pose
        feedback, velocity / acceleration limited, smoothed; translation and rotation simultaneous). ``posture`` is the
        recorded navigation posture (informational: Layer 1 changes no posture). ``style`` (``Navigate.style``): the
        polyline as recorded (default) or the ``face_travel`` schedule of ``robot.base_path`` (``_base_reference``);
        ``corner_stop``: the recorded polyline with a stop and a hold at every interior waypoint (an axis-aligned leg).
        Arm and trunk hold targets stay as they are, unless ``on_step`` (called every control step after the base command,
        before it is emitted) changes them; ``on_step`` returning True stops the base there. Raises ``execution`` when the
        tracker times out."""
        waypoints = list(waypoints)
        if not waypoints:
            return
        cfg = tracker_cfg or self.cfg.base_tracker
        v_lin = self.cfg.path_speed_fraction * min(cfg.v_max[0], cfg.v_max[1])
        v_ang = self.cfg.path_speed_fraction * cfg.v_max[2]
        ref = self._base_reference(waypoints, v_lin, v_ang, a_max=min(cfg.a_max[0], cfg.a_max[1]), style=style, corner_stop=corner_stop)
        tracker = PathTracker(ref, self.dt, cfg, max_lag=self.cfg.path_max_lag)
        self.tracker = tracker
        try:
            while True:
                v, done = tracker.step(self.base_pose2d())
                self.state.set_base_velocity(float(v[0]), float(v[1]), float(v[2]))
                if on_step is not None and on_step():
                    break
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

    def drive_with_arm_schedule(self, base_path: Sequence[Any], arm: str, arm_schedule: Sequence[Sequence[float]], speed: Optional[float] = None) -> Iterator[th.Tensor]:
        """Sequential follow, anchored move: the base tracks ``base_path`` (strides) at the base-follow speed while the
        ``arm``'s hold target follows the recorded per-stride joints ``arm_schedule`` (one 7-vector per stride, the last
        for the path's end) by the tracker's progress, slew-limited at ``follow_arm_speed``, and both wind down together
        at the end (no velocity cut). The trunk stays; the fingers stay closed."""
        path = [as_pose2d(b) for b in base_path]
        sched = [C.as_tensor(a).reshape(-1).float() for a in arm_schedule]
        if not path or not sched:
            return
        s = self.cfg.base_follow_speed if speed is None else float(speed)
        v_lin = self.cfg.path_speed_fraction * s
        v_ang = self.cfg.path_speed_fraction * self.cfg.base_follow_yaw_rate
        a_max = min(self.cfg.base_tracker.a_max[0], self.cfg.base_tracker.a_max[1], self.cfg.follow_base_accel)
        cfg = replace(self.cfg.base_tracker, v_max=(s, s, self.cfg.base_follow_yaw_rate), a_max=(a_max, a_max, self.cfg.base_tracker.a_max[2]))
        q0 = self.state.arm_target[arm].clone()
        pts = [q0] + sched
        m = len(pts) - 1

        def at(f: float) -> th.Tensor:
            x = min(max(f, 0.0), 1.0) * m
            j = min(int(math.floor(x)), m - 1)
            t = x - j
            return pts[j] + (pts[j + 1] - pts[j]) * t

        dq_max = max(float(th.max(th.abs(pts[j + 1] - pts[j]))) for j in range(m))
        ref = self._base_reference(path, v_lin, v_ang, a_max=a_max)
        tracker = PathTracker(ref, self.dt, cfg, max_lag=self.cfg.path_max_lag)
        self.tracker = tracker
        n_ref = max(1, len(ref) - 1)
        f_step = (self.cfg.follow_arm_speed * self.dt) / (dq_max * m) if dq_max > 1e-6 else 1.0
        v_prev, f_prev = th.zeros(3), 0.0
        self.follow_phase = "base"
        try:
            while True:
                v, done = tracker.step(self.base_pose2d())
                if done:
                    break
                v_prev = th.as_tensor(v).reshape(-1).float().clone()
                self.state.set_base_velocity(float(v[0]), float(v[1]), float(v[2]))
                f_prev = min(1.0, min(tracker.k / n_ref, f_prev + f_step))
                self.state.arm_target[arm] = at(f_prev)
                yield self._emit()
                if tracker.timed_out:
                    d, dyaw = tracker.error(self.base_pose2d())
                    raise SkillError("execution", f"drive_with_arm_schedule: base tracking timed out (pos err {d:.3f} m, yaw err {dyaw:.3f} rad)", pos_err_m=d, yaw_err_rad=dyaw)
            n_dec = int(math.ceil(float(v_prev[:2].norm()) / max(self.cfg.follow_base_accel * self.dt, 1e-6)))
            n_arm = int(math.ceil((1.0 - f_prev) / max(f_step, 1e-9)))
            n_fin = max(1, n_dec, n_arm)
            for k in range(1, n_fin + 1):
                g = max(0.0, 1.0 - k / max(n_dec, 1))
                self.state.set_base_velocity(float(v_prev[0]) * g, float(v_prev[1]) * g, float(v_prev[2]) * g)
                self.state.arm_target[arm] = at(f_prev + (1.0 - f_prev) * k / n_fin)
                yield self._emit()
            self.state.arm_target[arm] = sched[-1].clone()
        finally:
            self.state.stop_base()
            self.follow_phase = "arm"

    def base_follow_hybrid(self, follow_path: Sequence[Any], arm: str, speed: Optional[float] = None,
                           axis: Optional[Tuple[th.Tensor, th.Tensor]] = None) -> Iterator[th.Tensor]:
        """Articulation with the handle grasped, **hybrid** mode (sampler ``articulation_hybrid``, D-011 C): the recorded
        dense path ``poses[]`` = ``{"q", "base_pose2d": [x, y, yaw], "arm_q": [7 joints]}`` from the grasp (k = 0) to the
        last followed waypoint. Per consecutive pair the base drives ``b_{i-1} -> b_i`` (the joystick tracker at the
        base-follow speed) while ``arm``'s hold target ramps from where it is to ``arm_q_i`` **by the tracker's progress
        along the segment** (not by time), so the eef stays on the handle arc the sampler verified at every pose
        (consecutive poses are <= 10 deg of joint motion apart); an arm-only step (base unchanged) ramps the joints at
        the joint speed bound. The trunk stays; the fingers stay closed."""
        poses = [dict(p) for p in follow_path]
        if len(poses) < 2:
            return
        self.stop_requested = False  # set by the caller (the goal band reached): wind the base down and return
        idx = self.robot.arm_control_idx[arm]
        s = self.cfg.base_follow_speed if speed is None else float(speed)
        cfg = replace(self.cfg.base_tracker, v_max=(s, s, self.cfg.base_follow_yaw_rate))
        v_lin = self.cfg.path_speed_fraction * s
        v_ang = self.cfg.path_speed_fraction * self.cfg.base_follow_yaw_rate
        self.state.arm_target[arm] = self.robot.get_joint_positions()[idx].clone().float()
        try:
            for i in range(1, len(poses)):
                b1 = as_pose2d(poses[i]["base_pose2d"])
                q0 = self.state.arm_target[arm].clone()
                q1 = C.as_tensor(poses[i]["arm_q"]).reshape(-1).float()
                if q1.numel() != q0.numel():
                    raise SkillError("plan_failed", f"base_follow_hybrid: pose {i} has {q1.numel()} arm joints, the arm has {q0.numel()}", stage="targets")
                here = self.base_pose2d()
                dist = math.hypot(b1[0] - here[0], b1[1] - here[1])
                dyaw = abs(C.wrap_angle(b1[2] - here[2]))
                moves = dist > 2e-3 or dyaw > 5e-3
                if moves:
                    # the segment lasts at least as long as the arm change needs at follow_arm_speed: the base slows down
                    # for a pose pair with a big arm change, the arm ramp (by progress) inherits the slower pace
                    dq = float(th.max(th.abs(q1 - q0)))
                    t_base = max(dist / max(v_lin, 1e-6), dyaw / max(v_ang, 1e-6))
                    t_arm = dq / max(self.cfg.follow_arm_speed, 1e-6)
                    scale = min(1.0, t_base / t_arm) if t_arm > 1e-6 else 1.0
                    v_lin_i, v_ang_i = max(v_lin * scale, 0.01), max(v_ang * scale, 0.03)
                    a_max = min(cfg.a_max[0], cfg.a_max[1], self.cfg.follow_base_accel)
                    cfg_i = replace(cfg, v_max=(v_lin_i / self.cfg.path_speed_fraction, v_lin_i / self.cfg.path_speed_fraction, v_ang_i / self.cfg.path_speed_fraction),
                                    a_max=(a_max, a_max, cfg.a_max[2]) if len(cfg.a_max) >= 3 else cfg.a_max)
                    ref = self._base_reference([b1], v_lin_i, v_ang_i, a_max=a_max)
                    tracker = PathTracker(ref, self.dt, cfg_i, max_lag=self.cfg.path_max_lag)
                    self.tracker = tracker
                    n_ref = max(1, len(ref) - 1)
                    v_prev = th.zeros(3)
                    f_prev = 0.0
                    dq_seg = float(th.max(th.abs(q1 - q0)))
                    f_step = (self.cfg.follow_arm_speed * self.dt) / dq_seg if dq_seg > 1e-6 else 1.0  # slew limit on the ramp fraction
                    while True:
                        v, done = tracker.step(self.base_pose2d())
                        if done or self.stop_requested:
                            break  # the final command is already ~0: ramp down from the previous one below
                        v_prev = th.as_tensor(v).reshape(-1).float().clone()
                        self.state.set_base_velocity(float(v[0]), float(v[1]), float(v[2]))
                        # the arm follows the tracker's progress, but never faster than follow_arm_speed: the reference index
                        # runs ahead of the base while a segment starts with its yaw blend (door 0920-173049: 0.4 rad in 6 steps)
                        f_prev = min(1.0, min(tracker.k / n_ref, f_prev + f_step))
                        self.state.arm_target[arm] = q0 + (q1 - q0) * f_prev
                        yield self._emit()
                        if tracker.timed_out:
                            d, dyaw = tracker.error(self.base_pose2d())
                            raise SkillError("execution", f"base_follow_hybrid: base tracking timed out at pose {i} (pos err {d:.3f} m, yaw err {dyaw:.3f} rad)",
                                             pos_err_m=d, yaw_err_rad=dyaw, pose_index=i)
                    # decelerate the base from its last command at follow_base_accel and finish the arm ramp at follow_arm_speed
                    # over the same steps: no velocity cut and no jump to the final joints when the tracker reports done
                    # (0920-160045: a 0.07 -> 0 m/s cut and a 0.13 m/s eef jump at the segment end)
                    stopping = bool(self.stop_requested)
                    q_hold = self.state.arm_target[arm].clone()
                    dq_left = 0.0 if stopping else float(th.max(th.abs(q1 - q0))) * (1.0 - f_prev)
                    n_dec = int(math.ceil(float(v_prev[:2].norm()) / max(self.cfg.follow_base_accel * self.dt, 1e-6)))
                    n_arm = int(math.ceil(dq_left / max(self.cfg.follow_arm_speed * self.dt, 1e-6)))
                    n_fin = max(1, n_dec, n_arm)
                    for k in range(1, n_fin + 1):
                        g = max(0.0, 1.0 - k / max(n_dec, 1))
                        self.state.set_base_velocity(float(v_prev[0]) * g, float(v_prev[1]) * g, float(v_prev[2]) * g)
                        self.state.arm_target[arm] = q_hold if stopping else q0 + (q1 - q0) * (f_prev + (1.0 - f_prev) * k / n_fin)
                        yield self._emit()
                    self.state.stop_base()
                    if stopping:
                        return
                else:
                    # an arm-only dense step: track the recorded eef pose with the differential IK (the arm-mode follow's
                    # method) instead of a straight joint-space ramp between two IK solutions - that line leaves the handle
                    # arc by a few mm and the loaded wrist winds up against the door's friction and slips (dishwasher_wide
                    # 0920-140730: joint 7 at 2.1-2.6 rad/s in the arm-only steps, the arm-mode follow stays under 0.7)
                    eef_i = poses[i].get("eef")
                    tracked = False
                    if eef_i is not None:
                        try:
                            target_i = as_world_pose(eef_i)
                            wps: List[Any] = [target_i]
                            if axis is not None and poses[i - 1].get("eef") is not None:
                                # the true handle arc between the two recorded poses (rotation about the joint axis) instead
                                # of the chord: no kink at every 10 deg pose (owner 2026-09-20 15:54: "smooth like a good teleoperator")
                                wps = arc_between(as_world_pose(poses[i - 1]["eef"]), target_i, axis[0], axis[1])
                            # converge=False as in the arm-mode follow: a loaded arm never meets the joint-convergence
                            # tolerance and waited out the 90-step timeout at every pose (the 3 s pauses, owner 2026-09-20 18:43)
                            yield from self.arm_along(arm, wps, loaded=True, speed=self.cfg.follow_eef_speed, converge=False)
                            tracked = True
                        except SkillError as e:
                            log.warning("base_follow_hybrid: eef tracking of pose %d failed (%s): joint ramp instead", i, e.message or e.reason)
                    if not tracked:
                        dq = float(th.max(th.abs(q1 - q0)))
                        n = max(1, int(math.ceil(dq / (min(self.cfg.diff_ik_joint_speed, self.cfg.follow_arm_speed) * self.dt))))
                        for k in range(1, n + 1):
                            self.state.arm_target[arm] = q0 + (q1 - q0) * (k / n)
                            yield self._emit()
                            if self.stop_requested:
                                return
                        self.state.arm_target[arm] = q1.clone()
                    if self.stop_requested:
                        return
                    continue
                self.state.arm_target[arm] = q1.clone()
        finally:
            self.state.stop_base()
        yield from self.hold(3)


__all__ = ["SkillError", "MotionsConfig", "Motions", "MOTION_CATEGORIES", "as_world_pose", "as_pose2d", "ApproachFinish", "PathTracker"]
