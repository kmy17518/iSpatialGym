"""IORS small-scale (spec §6.11): ontop, under, inside, nextto, touching — the reference v2 task module.

Init per §4 through the shared tabletop phases (``v2/tabletop.py``: 1a edge + stand-off, 1b init trunk, 1c lateral with
the IORS edge rule, 2 manipulation stance survivors, 3 target from the feasible set F, partner placement); this module
adds the §6.11 rules only: pair admissibility per template, the start region and the partner side / gap rule of §4.6 step 2
(A1, owner 2026-10-12: the side is fixed before the target is placed - ``partner_side`` for eval -, the target starts on a cell
whose partner fits at some gap of its interval), the goal pose per relation, the place / pre-place IK from the manipulation
stance after the goal settled, the touching push + brace, the constraints it learned. Every number comes from the customs
``generation`` block (D1 ``generation_defaults``): ``table_end_margin_m``, ``start_region_lateral_draws``, ``partner_on_meta_link``
(A5: the partner centred on one of the support's meta links of that type, e.g. a burner's ``heatsource``), ``start_region_on_neighbors``
(with ``partner_on_meta_link``: the target starts on a neighbouring support's top, A4), ``init_align_support`` (the target and the
partner start with their footprints along the support's edges, owner 2026-10-13), ``partner_aligned_to_target`` (the partner's long
side along the target's long axis), ``pair_goal_align`` (nextto / touching: the target ends parallel to its partner, its front pointing
the way the key says, side by side across the long axes, owner 2026-10-13 ~22:23), ``partner_flat_top`` (the partner's footprint on a
part of the top seen from above as one flat face), ``brace_open_when_wide`` / ``push_speed_m_s`` (touching: the open-finger stop on a
partner face wide enough for it, the push's pad speed, owner 2026-10-14 ~02:04),
``partner_footprint_margin_m``, ``footprint_gap_min_m``, ``partner_gap_extra_m``, ``partner_moved_max_m``, ``goal_drop_m``,
``fit_tiers``, ``opening_margin_m`` / ``drop_height_m`` (floor) / ``release_clearance_m`` / ``max_protrusion_fraction``
(inside), ``handedness`` (nextto mirror pairs: the target ends on its own side of the partner), ``target_up_axis`` (inside: the
model axis the target stands on in its partner - it starts at its grasp key's rest and the goal stands it on that axis),
``tolerances``. A target bigger than its partner is shrunk by ``ground_isg`` before sampling (``roles.target.scale``,
``sampling/pair_rules.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.fastflags import fast_on, fast_value  # train-fast 2026-10-31: SPA_CACHE
from ispatialgym.oracle.programs import TOUCHING_SHELF_LAYOUT
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import alias_memory as AM
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import campaign_pair_goals as CPG
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put, cache_root  # train-fast 2026-10-31: the SPA lateral-cell memo on disk
from ispatialgym.sampling.sim_utils import PARK_POSITION, Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import contact_poses as CPZ  # i7-contact 2026-11-07: the touching push / brace pose rules (shared with the contact maps)
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import alias_switches as _AS
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import overhang_fallback as OF
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, TemplateExhausted, annotate_height_extra, contact_height_extra, register, requested_height_extra
from ispatialgym.sampling.v2.core import start_per_attempt  # noqa: E402  (train85)
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance, pose_from_sidecar, pose_to_sidecar

RETREAT_DIST_M = 0.10  # oracle ``actions.RETREAT_DIST``: a place ends this far back along the approach (release + lift-off)
RETREAT_LITTLE_M = 0.05  # oracle ``actions.RETREAT_LITTLE``: the retreat after a place
#: oracle-fix 2026-11-06 (``generation.shelf_backoff_transit_m``): the program check's pre-grasp witness may lie this far (largest arm joint, rad)
#: from the joints the backed-off transition carries the arm to - the settled target's few-mm shift; beyond it the grasp's transit is a swing
SHELF_BACKOFF_WITNESS_GAP_RAD = 0.3
#: oracle-fix 2026-11-06 (``generation.shelf_start_draw``, touching#03): the layouts (compartment, leans, spine depth) one attempt draws until
#: both books' leaning starts pass the side-board rays, the board and the free span (``shelf_wall`` / ``shelf_board`` / ``shelf_width``)
SHELF_DRAW_TRIES = 8
SHELF_DRAW_REJECTS = ("shelf_wall", "shelf_board", "shelf_width")
#: i7-t03 2026-11-08 (owner request item 1, DEFAULT OFF until the owner decides): when the shelf layout's fixed start fails the grasp or the
#: transit check at a manipulation candidate, the base is moved along the edge - first to the COMPUTED stance (the book at
#: :data:`SHELF_STANCE_BOOK_LATERAL_M` from the robot's centre line, same stand-off / trunk), then a scan (shifts x ``tabletop.STAND_OFFS``,
#: the computed stance first) - every stance through the same exact checks (``IORSSmall.shelf_fixed_start``). "1" on, else off.
SHELF_STANCE_SHIFT_ENV = "ISPATIALGYM_SHELF_STANCE_SHIFT"
#: the book's lateral from the robot's centre line (m, + = the robot's left) for the LEFT working arm in every written touching_03 demo:
#: fix1_t03nr x 3 (bay 0, base lateral -0.065) +0.185, fix2_t03init x 4 (bay -0.37, base lateral -0.465) +0.189 / +0.191 / +0.190 / +0.200 -
#: all at stand-off 0.55, trunk (0.25, 0.0), the book 0.687 m base-forward (``generation.shelf_stance_book_lateral_m``); the right arm mirrored
SHELF_STANCE_BOOK_LATERAL_M = 0.19
#: the scan's shifts along the edge (m) relative to the computed stance (``generation.shelf_stance_scan_shifts_m``), x the stand-offs
SHELF_STANCE_SCAN_SHIFTS_M = (0.0, 0.05, -0.05, 0.10, -0.10)
#: stances the scan tries per attempt beyond the computed ones (``generation.shelf_stance_scan_max``): each costs a fixed-start grasp check
SHELF_STANCE_SCAN_MAX = 6
#: a pre-grasp witness farther than this (largest arm joint, rad) from the transit's start posture is reported as the FLIPPED branch
SHELF_FLIPPED_REACH_RAD = 2.5


#: i7-t03 round 2 2026-11-08 (owner (a), DEFAULT OFF): the shelf layout's start falls back to IORSLarge's free-floor start
#: (``iors_large.free_floor_stance``, ``generation.free_start``) when the edge init stance does not frame the target AND the partner
SHELF_FREE_START_ENV = "ISPATIALGYM_SHELF_FREE_START"


#: i7-t03 round 2 2026-11-08 (owner (b), DEFAULT OFF): every fixed-start stance the base-shift search judges (the candidate, the computed
#: stance, the scan) is judged on the grasp AND the tip release at the drafted goal (``place_infeasible_why``); a stance passing both is
#: taken first (no carried drive), a grasp-only stance is the fallback. Needs ``ISPATIALGYM_SHELF_STANCE_SHIFT=1`` for the shifted stances.
SHELF_JOINT_STANCE_ENV = "ISPATIALGYM_SHELF_JOINT_STANCE"


def shelf_joint_stance_on(s=None) -> bool:
    return _AS.switch_on(SHELF_JOINT_STANCE_ENV, s)


#: i7-t03 round 2 2026-11-08 (owner (d), APPROVED oracle + contact-rule change, DEFAULT OFF): the shelf tip release's lift turns the leaning
#: book upright while it rises (``grasps[i].lift_turn``, the oracle's ``Lift.turn``; the sampler's lift end via ``program_check.lift_end_pose``)
#: when the upright book stays under the board above, and at the lift ONLY the attached book may touch the side board it leaned on (clearance
#: >= 0 on its hull, never penetrate); every robot link keeps the obstacle rule
SHELF_LIFT_STRAIGHTEN_ENV = "ISPATIALGYM_SHELF_LIFT_STRAIGHTEN"
#: the leaned-on board is the split-kept piece within this of the book at its start (m)
LEANED_BOARD_MAX_GAP_M = 0.005
#: the upright book's top keeps this below the board above along the lift (m)
LIFT_HEADROOM_MARGIN_M = 0.005
#: (d) touch-rule fix 2026-11-08 (LEAD DECISION, owner away; same switch): at lift pose 0 ONLY (the grasp pose = the book's resting lean on the
#: board) the book's hull may read down to this below the leaned-on board (m) - the resting contact, measured < 0.05 mm in every one of the 53
#: i7t03r4_v / r5_diag refusals; every later lift pose keeps clearance >= 0 (:data:`LIFT_TOUCH_CLEARANCE_TOL_M`) and every robot link today's rule
LIFT_START_CONTACT_TOL_M = 1e-4
#: the clearance tolerance of the leaned-board touch rule (m; numerical zero)
LIFT_TOUCH_CLEARANCE_TOL_M = 1e-9
#: (d) fix 2026-11-08: the book's standing axis (:meth:`IORSSmall.shelf_standing_axis`) is used only while it stands within this of vertical at
#: the attempt's start (deg; the drawn leans are 11-20 deg) - past it the lean cannot be told from another axis and the lift stays today's
SHELF_LIFT_AXIS_MAX_TILT_DEG = 40.0


def shelf_lift_straighten_on(s=None) -> bool:
    return _AS.switch_on(SHELF_LIFT_STRAIGHTEN_ENV, s)


#: i7-t03 2026-11-08 (owner 5k.1, DEFAULT OFF): the shelf layout's free span between the two leaning books is judged against a COMPUTED
#: requirement instead of the fixed ``generation.shelf_free_span_min_m`` (0.12 in the customs, fitted to kmuacd's 0.342 m cube; code default
#: 0.15): the carried book's extent along the edge (at the lean the attempt drew; upright = its thickness when the straightening lift (d) is
#: on) plus the open jaw's excess over the pinch on both sides (the oracle's pre-shape opening + both fingers' thickness - the pinch width).
#: Checked per attempt at the drawn leans (:meth:`IORSSmall.draw_shelf_layout`) and at the template stage at the most favourable drawable
#: leans (a template no draw can pass still stops there). Kit ``shelf_span.passed`` / ``.rejected`` (+ ``.template_rejected``)
SHELF_SPAN_RULE_ENV = "ISPATIALGYM_SHELF_SPAN_RULE"


def shelf_span_rule_on() -> bool:
    return os.environ.get(SHELF_SPAN_RULE_ENV, "").strip().lower() in ("1", "on", "true")


#: i7-t03 2026-11-08 (owner 5p.1, an approved oracle change, DEFAULT OFF): the shelf tip release's clear pose (15 cm back along the approach:
#: the place's 10 cm release lift-off + the 5 cm retreat) is out of the arm's reach from the release stance (diagnostic 0: it fails with the
#: shelf and the partner removed). When it fails, the base backs off instead of the arm: the place's own end (10 cm back) from the release
#: stance, then a backward drive of RETREAT_LITTLE_M along the stance's axis with the arms frozen there (the shelf back-off rule's drive-back,
#: ``navigation.path_collision_free`` strides) - recorded as the stance ``retreat`` + the transition into it, which the oracle program drives
#: in place of the final 5 cm retreat (``programs.program_iors_touching``). Kit ``shelf_retreat.backoff_passed`` / ``.backoff_failed.<why>``
#: / ``.recorded``
SHELF_RETREAT_BACKOFF_ENV = "ISPATIALGYM_SHELF_RETREAT_BACKOFF"
#: the stance the backed-off retreat ends at (``sidecar.stances``; the oracle's program reads it)
RETREAT_STANCE = "retreat"


def shelf_retreat_backoff_on(s=None) -> bool:
    return _AS.switch_on(SHELF_RETREAT_BACKOFF_ENV, s)


#: i7-t03 round 7 (LEAD DECISION #37, owner away; DEFAULT OFF): the shelf tip's fixed-start grasp is judged at the shelf stance with the goal
#: check's own frame rule (:meth:`IORSSmall.require_in_frame` on the pre-grasp + grasp at that stance's camera, same 5 % margin) - a stance
#: whose pre-grasp fingertip leaves the image is a stance-shift need ("frame") instead of a goal-check reject after the lift (i7t03r6_v: 35
#: x "pre-grasp fingertip outside the manipulation image", all at stand-off 0.45). Kit ``shelf_pregrasp_frame.checked / .passed / .refused``
SHELF_PREGRASP_FRAME_ENV = "ISPATIALGYM_SHELF_PREGRASP_FRAME"
#: i7-t03 round 7 (LEAD DECISION #37; DEFAULT OFF, the shelf tip release only): ``release_clear_why`` checks the retreat TRACKED from the place
#: configuration (place -> 10 cm -> 15 cm back along the approach, one branch, as the program check runs the oracle's legs) instead of a fresh
#: 32-seed solve of the 15 cm pose (i7t03r6_v: 12 x program_check "retreat: the tracked chain's end configuration collides" after a fresh
#: solve passed). Kit ``shelf_release_tracked.checked / .passed / .failed``
SHELF_RELEASE_TRACKED_ENV = "ISPATIALGYM_SHELF_RELEASE_TRACKED"


def _pair_mesh_on(s=None) -> bool:
    from ispatialgym.sampling.v2 import pair_mesh as PM

    return PM.pair_mesh_on(s)


def shelf_pregrasp_frame_on(s=None) -> bool:
    return _AS.switch_on(SHELF_PREGRASP_FRAME_ENV, s)


def shelf_release_tracked_on(s=None) -> bool:
    return _AS.switch_on(SHELF_RELEASE_TRACKED_ENV, s)


def shortest_arc(u: th.Tensor, v: th.Tensor) -> th.Tensor:
    """The smallest rotation (xyzw) taking unit vector ``u`` onto ``v``."""
    u = G.t(u).float() / float(th.linalg.norm(G.t(u).float()))
    v = G.t(v).float() / float(th.linalg.norm(G.t(v).float()))
    c = float(th.dot(u, v))
    if c > 1.0 - 1e-9:
        return th.tensor([0.0, 0.0, 0.0, 1.0])
    axis = th.linalg.cross(u, v)
    if float(th.linalg.norm(axis)) < 1e-9:  # opposite: any perpendicular axis
        axis = th.linalg.cross(u, th.tensor([1.0, 0.0, 0.0]) if abs(float(u[0])) < 0.9 else th.tensor([0.0, 1.0, 0.0]))
    axis = axis / float(th.linalg.norm(axis))
    ang = math.acos(max(-1.0, min(1.0, c)))
    return th.cat([axis * math.sin(ang / 2), th.tensor([math.cos(ang / 2)])])


def _slerp(q0: th.Tensor, q1: th.Tensor, a: float) -> th.Tensor:
    """Spherical interpolation of xyzw quaternions (the shorter arc)."""
    d = float(th.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / float(th.linalg.norm(q))
    th0 = math.acos(d)
    return (math.sin((1 - a) * th0) * q0 + math.sin(a * th0) * q1) / math.sin(th0)


def lift_object_poses(grasp_pose: G.Pose, lift_end: G.Pose, eef_in_object: G.Pose, n: int = 10) -> List[G.Pose]:
    """The held object's poses along a lift (the eef lerped / slerped from ``grasp_pose`` to ``lift_end``, the object = eef o T_obj->eef^-1)."""
    inv = G.invert((G.t(eef_in_object[0]).float(), G.t(eef_in_object[1]).float()))
    p0, p1 = G.t(grasp_pose[0]).float(), G.t(lift_end[0]).float()
    q0, q1 = G.t(grasp_pose[1]).float(), G.t(lift_end[1]).float()
    out = []
    for k in range(n + 1):
        a = k / n
        out.append(G.compose((p0 + a * (p1 - p0), _slerp(q0, q1, a)), inv))
    return out


def lift_headroom_m(hull_local: th.Tensor, poses: Sequence[G.Pose], ceiling_z: float) -> float:
    """The smallest distance (m) from the held object's top to ``ceiling_z`` over ``poses`` (negative: above it)."""
    return min(float(ceiling_z) - float(G.local_points_world(hull_local, (G.t(p[0]).float(), G.t(p[1]).float()))[:, 2].max()) for p in poses)


def lift_touch_clearance_why(cls: Sequence[float]) -> Optional[str]:
    """(d): the leaned-board touch rule on the attached book's hull clearances (m) to that board at each lift pose (index 0 = the grasp pose).
    None = allowed: pose 0 (the resting lean contact) >= -:data:`LIFT_START_CONTACT_TOL_M`, every later pose >= -:data:`LIFT_TOUCH_CLEARANCE_TOL_M`.
    Else ``"penetration_start"`` (pose 0 deeper than the resting tolerance) or ``"penetration"`` (a later pose below 0)."""
    if not cls:
        return "penetration"
    if any(c < -LIFT_TOUCH_CLEARANCE_TOL_M for c in cls[1:]):
        return "penetration"
    if cls[0] < -LIFT_START_CONTACT_TOL_M:
        return "penetration_start"
    return None


class _PieceShim:
    """One collision piece of an object as an ignorable object (``IKChecker.ignoring`` switches off its prim path; no split plan)."""

    def __init__(self, name: str, prim: str, cm: Any):
        self.name = name
        self.links = {"piece": type("L", (), {"collision_meshes": {prim: cm}})()}


#: i7-t03 round 2 2026-11-08 (owner (c), DEFAULT OFF): the shelf tip release's second base pose is not chosen by the push-inherited contact
#: laterals but as a drive along the edge toward the partner that keeps the lifted book inside the free gap to the partner (minus the carried
#: path margin): the drives SHELF_RELEASE_DRIVE_FRACTIONS of that room, the first whose release passes the same exact place checks
SHELF_RELEASE_IN_GAP_ENV = "ISPATIALGYM_SHELF_RELEASE_IN_GAP"
SHELF_RELEASE_DRIVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def shelf_release_in_gap_on(s=None) -> bool:
    return _AS.switch_on(SHELF_RELEASE_IN_GAP_ENV, s)


def shelf_free_start_on(s=None) -> bool:
    """``ISPATIALGYM_SHELF_FREE_START`` = "1" (i7-t03 round 2 item (a); default off)."""
    return _AS.switch_on(SHELF_FREE_START_ENV, s)


def shelf_stance_shift_on(s=None) -> bool:
    """``ISPATIALGYM_SHELF_STANCE_SHIFT`` = "1" (i7-t03 item 1; default off)."""
    return _AS.switch_on(SHELF_STANCE_SHIFT_ENV, s)


#: i7-t03both 2026-11-09 (owner decision A5, DEFAULT OFF): with ``generation.shelf_start_draw`` and the v2 library's lean rests, each attempt
#: draws the compartment AND the board the target leans against among every layout of the support whose board has a lean rest with an interior
#: reach map (``tabletop.shelf_lean_pool``), then that board's lean rest (its angle) uniformly - the grasps, the rest, the band and the working
#: arm follow the drawn rest (:meth:`IORSSmall.draw_shelf_side`). Today (off): the template's ONE lean rest pins the board (and its angle).
#: Needs start_per_attempt (each attempt's Phase 2 with its arm and band); eval keeps the board opposite ``generation.partner_side``.
SHELF_BOARD_DRAW_ENV = "ISPATIALGYM_SHELF_BOARD_DRAW"


def shelf_board_draw_on(s=None) -> bool:
    """``ISPATIALGYM_SHELF_BOARD_DRAW`` (i7-t03both, owner A5; default off - not in ``alias_switches.SHELF_TIP_DEFAULTS``)."""
    return _AS.switch_on(SHELF_BOARD_DRAW_ENV, s)


#: i7-t03both r2 2026-11-09 (lead #84b, DEFAULT OFF): the shelf layout's free-floor start (``SHELF_FREE_START``) keeps only a base position in the
#: same connected region of the eroded traversability map as the attempt's manipulation candidates - T03B_PATH_DIAG: every init -> drive-back
#: ``no_path`` of the board draw (60 / 76) left from a free start in another region (1221 cells) than the bookcase front (472 cells)
SHELF_FREE_START_CONNECTED_ENV = "ISPATIALGYM_SHELF_FREE_START_CONNECTED"


def shelf_free_start_connected_on(s=None) -> bool:
    return _AS.switch_on(SHELF_FREE_START_CONNECTED_ENV, s)


#: i7-t03both r2 2026-11-09 (lead #84b, DEFAULT OFF): at the fixed shelf start the working arm's interior reach map is read AT THE BOOK - a
#: Phase-2 candidate whose trunk / stand-off cell under the book holds no feasible grasp row skips the exact checks and goes to the stance
#: shift, which tries first the MAP stances: the base moved along the edge / to another map stand-off and trunk so the book lands on a cell the
#: map calls feasible (T03B_GRASP_DIAG: the candidate's focus is any raster cell, the books are placed per attempt after it, so the selected
#: candidate's map verdict says nothing about the book: 'exact IK failed at pose 1 of 2' with the map's cell ik_pre / the trunk not in the map)
SHELF_MAP_STANCE_ENV = "ISPATIALGYM_SHELF_MAP_STANCE"
#: map stances tried per candidate (each one a full fixed-start check)
SHELF_MAP_STANCE_MAX = 6


def shelf_map_stance_on(s=None) -> bool:
    return _AS.switch_on(SHELF_MAP_STANCE_ENV, s)


def shelf_need_of(err: Reject) -> Optional[str]:
    """Why a candidate's fixed start failed, for the stance shift: ``"transit"`` (the transit / back-off checks), ``"grasp"`` (no library
    grasp reaches the fixed start pose), None for anything the base position cannot change (the shift is not tried)."""
    msg = str(err)
    if msg.startswith(f"[{err.reason}] "):
        msg = msg[len(err.reason) + 3:]
    if err.reason == "grasp_unreachable" and msg.startswith("transit check"):
        return "transit"
    if err.reason == "no_transition" and msg.startswith("shelf back-off"):
        return "transit"
    if err.reason == "grasp_unreachable":
        return "grasp"
    if err.reason == "gripper_not_in_frame" and msg.startswith(SHELF_PREGRASP_FRAME_TAG):
        return "frame"
    return None


#: the message prefix of the round-7 shelf-stance frame reject (:data:`SHELF_PREGRASP_FRAME_ENV`)
SHELF_PREGRASP_FRAME_TAG = "shelf stance frame:"


def reach_branch(reach: Optional[float]) -> str:
    """``"flipped"`` / ``"near"`` / ``"-"`` of a transit reach (rad, :data:`SHELF_FLIPPED_REACH_RAD`)."""
    if reach is None or not math.isfinite(float(reach)):
        return "-"
    return "flipped" if float(reach) > SHELF_FLIPPED_REACH_RAD else "near"


# i7-contact 2026-11-07: the touching pose rules' constants and hull helpers live in ``contact_poses`` (the sampler and the contact maps share
# them); the names stay importable from here
from ispatialgym.sampling.v2.contact_poses import (BRACE_OPEN_END_MARGIN_M, BRACE_OPEN_FACE_DEPTH_TOL_M, CLOSED_TIP_SPREAD_M, HAND_SPHERES_EEF,  # noqa: E402,F401
                                                   OPEN_TIP_CENTRE_M, PUSH_OPEN_OPENING_M, PUSH_OPEN_SUPPORT_CLEARANCE_M, PUSH_RAISED_MAX_ASPECT,
                                                   PUSH_SWEEP_BASE_CUT_M, PUSH_SWEEP_MIN_HEIGHT_M, PUSH_SWEEP_WINDOW_M, PUSH_WRIST_TURNS, STOP_CLEARANCE,
                                                   HullSections, face_run_at, hull_cut_points, min_area_rect, open_fingers_drop_below,
                                                   point_triangle_distances, polygon_centroid, push_face_point)

log = logging.getLogger("ispatialgym.sampling.v2")


def _with_orient(extras: Dict[str, Any], decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The contact record's extras with the orientation decision (``orient.choose``) when the class had two wrist turns."""
    if decision is None or len([o for o in decision.get("options", []) if o.get("source") == "ik"]) < 2:
        return dict(extras)
    return dict(extras, orient=decision)

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_small_scale": "ontop",
    "inter_object_relation_satisfaction-under_small_scale": "under",
    "inter_object_relation_satisfaction-inside_small_scale": "inside",
    "inter_object_relation_satisfaction-nextto_small_scale": "nextto",
    "inter_object_relation_satisfaction-touching_small_scale": "touching",
}
#: v1 ``bddl_writer`` definition ids of the relations (facade rendering when the variant has no grounded BDDL)
V1_DEFINITION_ID = {"ontop": 0, "under": 1, "inside": 2, "nextto": 3, "touching": 4}


def _tabletop():
    """The shared tabletop phases module (``v2/tabletop.py``); kept as an accessor because ``iors_large`` imports it."""
    return TT


# ----------------------------------------------------------------------------------------------------------------
# A1 (owner decision 2026-09-16): joint placement of a work point (right arm) and a brace point (left arm)
# ----------------------------------------------------------------------------------------------------------------

from ispatialgym.sampling.v2.stances import BAND_CELL_TOL, brace_placement_admissible, in_band, nearer_side_lateral  # noqa: E402,F401  (pure rules, tested there)

#: i7-plateqtf2 (owner decision 2026-11-08 02:35 item 3): the draw's pre-place frame rule - ``fingertip`` (default: the program check's
#: fingertip point + 2 cm slack, ``IORSSmall.pre_place_frame_ok``) or ``box`` (the eef box, the rule before)
PREPLACE_FRAME_RULE_ENV = "ISPATIALGYM_PREPLACE_FRAME_RULE"


def pre_place_frame_rule() -> str:
    v = (os.environ.get(PREPLACE_FRAME_RULE_ENV, "") or "fingertip").strip().lower()
    if v not in ("fingertip", "box"):
        raise ValueError(f"{PREPLACE_FRAME_RULE_ENV}={v!r}: expected fingertip or box")
    return v


#: i7-framing (owner decision 2026-11-08 5.2): the frame rule of the PRE-GRASP and the touching PRE-CONTACTS (push / brace) - ``box`` (the
#: eef box, the rule before) or ``fingertip`` (DEFAULT since the count: the pre-place's rule: ``program_check.fingertip_frame_margin``, the fingertip point inside
#: QA's 5 % plus 2 cm slack). Both rules are evaluated either way and the disagreements counted (``precontact_frame.<pose>.removed``: the
#: box fails and the fingertip passes; ``.added``: the box passes and the fingertip fails; ``.removed_object_out``: a removed rejection
#: whose object - the target at the pre-grasp / push, the partner at the brace - is not wholly in that image). The contact poses (grasp,
#: push contact / goal, brace contact) keep the box, as the program check does (``CONTACT_POSE_KEYS``).
PRECONTACT_FRAME_RULE_ENV = "ISPATIALGYM_PRECONTACT_FRAME_RULE"
PRECONTACT_FRAME_POSES = ("pre-grasp", "push pre-contact", "brace pre-contact")


def precontact_frame_rule() -> str:
    v = (os.environ.get(PRECONTACT_FRAME_RULE_ENV, "") or "fingertip").strip().lower()  # i7-framing: DEFAULT fingertip (owner 5.2, after the count)
    if v not in ("fingertip", "box"):
        raise ValueError(f"{PRECONTACT_FRAME_RULE_ENV}={v!r}: expected fingertip or box")
    return v


def left_contact_band_cells(s, trunk: Tuple[float, float], height: float, stance=None):
    """Base-frame cells of the left-arm generic CONTACT band (closed gripper, brace approach; one band per object height
    class — B3 A1) at ``trunk``, through ``tabletop.load_left_contact_band`` (B3's ``ReachBands.load(scene, support,
    arm="left", contact_set="generic_contact", height=h)``, nearest class within ±4 cm, in-process compute as the last
    resort; cached on ``s.bands_left``). None only when no band can be had — then the key looked for and the left bands
    present for the support are logged and the caller records the gate as unavailable."""
    bands = None
    try:
        bands = TT.load_left_contact_band(s, float(height), top=s.top, edge=s.edge, compute=True)
    except Exception as e:  # noqa: BLE001 - loader / compute failure: recorded, the exact two-arm IK stays the gate
        log.warning("left contact band for height %.3f unavailable: %r", float(height), e)
    if bands is None:
        import glob
        import os

        cls = TT.height_class(float(height))
        root = os.environ.get("ISPATIALGYM_REACH_BANDS", "runlogs/reach_bands_20260915/bands")
        stem = os.path.join(root, f"contact__{s.world.scene_model}__{s.top.name}")
        present = sorted(os.path.basename(p) for pat in ("__left_*.npz", "-interior-*__left_*.npz") for p in glob.glob(stem + pat))
        log.warning("no left contact band: key scene=%s support=%s arm=left contact_set=%s height=%.3f (class %.2f) surface_z=%s; left bands present: %s",
                    s.world.scene_model, s.top.name, getattr(TT, "CONTACT_SET", "generic_contact"), float(height), cls, getattr(s, "surface_z", None),
                    present or "none")
        return None
    cells = bands.band(trunk[0], trunk[1], **TT.band_kw(bands, stance))  # a multi-distance band at ``stance``'s stand-off
    log.info("left contact band %s: %d cells at trunk (%.2f, %.2f)", getattr(bands, "path", "?"), len(cells), trunk[0], trunk[1])
    return cells


#: §6.11 touching admissibility numbers (pushability / stability; ``generation`` overrides)
PUSH_MAX_PINCH = 0.089
PUSH_MAX_LONG = 0.30
PUSH_MAX_HEIGHT = 0.30
PARTNER_MAX_ASPECT = 2.5
#: 7g (owner 2026-10-12), the flat base: free spots of the top the target is settled on alone at the template before it counts as
#: trembling (``generation.tremble_screen_spots``; the default for touching, 0 = off for the other relations)
TREMBLE_SCREEN_SPOTS = 2
#: inside (owner decision 2026-09-16): the open gripper's lowest point clears the partner's rim by this at the release
RELEASE_CLEARANCE = 0.02
#: ontop / under (owner item 3): the object is released at most this far above the actual surface under its footprint
#: centre — half the well depth, at least ``RELEASE_ABOVE_SURFACE_MIN`` (a flat plate)
RELEASE_ABOVE_SURFACE = 0.01
RELEASE_ABOVE_SURFACE_MIN = 0.003
#: library grasps whose closing axis is inclined more than this to the horizontal (and the vertical) in the target's rest
#: frame rotate the object in the hand until the pinch is flush (dropped while a flush grasp remains)
CLOSING_TILT_MAX_DEG = 12.0
#: a target that rolled more than this while settling gets its grasp re-selected at the settled pose
REGRASP_AFTER_ROLL_DEG = 10.0
#: the open gripper at the grasp configuration keeps this clear of the support (closure probe: a pad on the table makes
#: the closure asymmetric and knocks the object)
PAD_SUPPORT_CLEARANCE = 0.002
#: the pinch centre may sit this far from the object's centre along the closing axis; farther pinches are re-centred
PINCH_OFFSET_MAX = 0.005
#: touching brace (axial stop-brace rule): the pads' / blade's outer face this far outside the partner's far face, and the
#: partner height below which the top-down blade is tried before the pad contacts
THIN_PARTNER_HEIGHT = 0.04
#: R1Pro finger envelope beyond the pad faces (m): pad / finger thickness outward of the open pads along ±c, finger
#: half width along eef x (conservative; only the open-gripper lowest point is derived from them)
FINGER_THICKNESS = 0.012
FINGER_HALF_WIDTH = 0.012
#: ``generation.raised_release`` (nextto; owner 2026-10-14 ~01:51, nextto#03): the open jaw clears the partner by ``partner_clearance_m``
#: at the release, raised in ``step_m`` steps up to ``max_raise_m`` (the owner: about 2 cm, "not too high")
RAISED_RELEASE_DEFAULTS = {"partner_clearance_m": 0.005, "max_raise_m": 0.03, "step_m": 0.001}
#: touching#03 shelf layout: the lift of the target before it is slid along the board to the partner's side (it never passes over
#: the partner); 5 mm under the headroom the interior raster keeps above the tallest book at every free cell
SHELF_CARRY_HEIGHT_M = TT.INTERIOR_HEADROOM_MARGIN_M - 0.005
#: ``grasps[0].place_kind`` of the shelf layout's tip release (``generation.shelf_tip_release``, oracle-fix 2026-11-04 round 4): the book is
#: released at the goal check's start pose and tips onto the partner - no push
SHELF_TIP_PLACE_KIND = "shelf_tip"
#: owner 2026-10-12 item 4A: the lifted target's bottom (on the table at init) clears what the carry passes over by this much
#: (``generation.carry_clearance_m``)
CARRY_CLEARANCE_M = 0.03
#: 4A: ``generation.carry_clears`` - the partner's top ("partner"), only the table ("table"), or the table unless the carried drive to
#: the touching push stance takes the box over the partner ("auto"); default "auto" for touching, "partner" for the other relations
CARRY_CLEARS = ("partner", "table", "auto")
#: owner 2026-10-13 (touching#02): ``generation.widest_faces`` - the target's and the partner's widest upright faces parallel, their
#: normals along the edge tangent or the stance's depth axis ("parallel"), along the tangent only ("tangent") or the depth only ("depth")
WIDEST_FACES = ("parallel", "tangent", "depth")
#: owner 2026-10-13 ~22:23 (pair goal alignment): ``generation.pair_goal_align.facing`` - where the target's front points in the stance's
#: edge frame ("away" = along the edge's inward normal, "toward" against it, "left" / "right" = along / against the edge tangent, "any")
PAIR_FACINGS = ("away", "toward", "left", "right", "any")
#: the object axes ``pair_goal_align.front`` may name (the model's pointing end, in its own frame)
PAIR_FRONT_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}
#: ``pair_goal_align`` defaults: the settled goal's largest front / long-axis angle (deg) and stagger along the partner's long axis (m)
PAIR_TOL_DEG = 5.0
PAIR_STAGGER_MAX_M = 0.03
#: A1 start region (owner 2026-10-12): Phase-1c lateral draws of the template before it is rejected when no partner side the template
#: may draw has a start cell (``generation.start_region_lateral_draws``)
START_REGION_LATERAL_DRAWS = 4
#: A1: partner orientations tried per attempt, each with the start region's gap interval re-evaluated at the drawn target pose
PARTNER_YAW_DRAWS = 16
#: owner A1 (2026-11-08): the stand-offs a touching push stance may take (the touch_push maps' edge distances 0.35 / 0.45 / 0.55 and closer
#: ones: the placed target sits 0.66-0.93 m ahead, the map reaches ~0.75 m); the whole-body collision check is the lower limit
PUSH_STANCE_STAND_OFFS_M = (0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55)
#: owner A1: at most this many map-feasible push stances get the exact push-chain checks
PUSH_STANCE_MAX_EXACT = 8
#: owner 3.6 (2026-11-08, i7-contact3): the touching place depth capped by the touch_push map's reach (:meth:`IORSSmall.place_depth_caps`) -
#: "0" off, "shadow" (the cap computed and counted, no candidate removed: sampling as with it off), "1" on (a manipulation candidate whose
#: target - placed at its start depth - no collision-free push stance's map reaches leaves the attempt's draws). Default "1" since the
#: 2026-11-08 A/B (the owner decided the rule, 3.6): of the would-be removals that reached a push test none was reachable (touching_02 85 / 85,
#: touching_04 58 / 58), none was written; on vs shadow 0.33 vs 0.53 GPU-h per success
PLACE_DEPTH_CAP_ENV = "ISPATIALGYM_PLACE_DEPTH_CAP"
PLACE_DEPTH_CAP_DEFAULT = "1"
#: i7-contact3 touching_04 follow-up (c) (lead 2026-11-08, within owner 3.6): "1" = the attempt's start is re-drawn (start per attempt) while every
#: candidate of the partner side lies beyond the cap, at most :data:`PLACE_DEPTH_CAP_REDRAWS` times - the place depth DRAWN within the cap
#: instead of an attempt that loses every candidate. Default "0" (the owner decides after the A/B).
PLACE_DEPTH_CAP_REDRAW_ENV = "ISPATIALGYM_PLACE_DEPTH_CAP_REDRAW"
PLACE_DEPTH_CAP_REDRAWS = 8
#: follow-up (c), the draw itself: "1" = Phase 2's (stance, position) pairs are cut to the cap (``keep_pairs``: the start / place positions
#: the attempt draws from lie within the push's reach; an empty class falls back as Phase 2 does) - the redraw above re-draws whole starts,
#: this draws the position within the cap. Enters the start-per-attempt cell cache key when on. Default "1" since the owner's 5r.1
#: (2026-11-08; A/B c3_fu_c2: attempts losing every candidate 47 -> 0, written per attempt touching_04 0.044 -> 0.11, touching_02 0.045 ->
#: 0.16); "0" = the draw of before
PLACE_DEPTH_CAP_DRAW_ENV = "ISPATIALGYM_PLACE_DEPTH_CAP_DRAW"
PLACE_DEPTH_CAP_DRAW_DEFAULT = "1"
#: follow-up (b): the cap's basis - "map" (default: the deepest feasible map cell at any yaw / lateral from the base stand-offs that are
#: collision-free at the candidate's own lateral) or "a1" (A1's own push stance test, :meth:`IORSSmall.push_stance_candidates` with the
#: collision check, for the target at the candidate's focus at each map yaw slot: the same basis as the partner-draw gate)
PLACE_DEPTH_CAP_BASIS_ENV = "ISPATIALGYM_PLACE_DEPTH_CAP_BASIS"
#: follow-up (a) (lead-approved, sampler-only): the carried drive's lift (the box over the partner's top, ``carry_clears`` auto) screened at
#: the draw - the goal check's own test, made where it is certain the goal check makes it (every A1 push stance is a real second base pose
#: whose drive passes over the partner): "0" off, "shadow" (computed and counted, nothing rejected), "1" on. Default "1" since the owner's
#: 5r.1 (2026-11-08; A/B c3_fu_a: goal_check lift_unreachable touching_04 6 -> 2, touching_02 12 -> 5; in shadow all 18 would-be rejections
#: failed at or before the goal check)
TOUCH_LIFT_SCREEN_ENV = "ISPATIALGYM_TOUCH_LIFT_SCREEN"
TOUCH_LIFT_SCREEN_DEFAULT = "1"
#: owner 5r.1 touching_04 follow-up (2026-11-08): the goal check's contact_unreachable screened at the draw against the push map - the goal
#: check's own A1 push stances (:meth:`IORSSmall.push_stance_candidates`: touch_push map + whole-body collision, the same call and inputs) must
#: hold one stance from which the carried box's place passes (:meth:`IORSSmall.place_infeasible_why`, the goal check's own "place:" verdict);
#: when none does, the goal check rejects contact_unreachable ~130 s later, after the draft. "0" off, "shadow" (counted, nothing rejected),
#: "1" on. Only where the goal check is certain to use those stances (A1 non-empty; not the shelf layout / tip release). Default "1" since
#: the lead's decision 7 (2026-11-08, owner away; A/B c3_r1: shadow 3 / 3 would-be rejections failed at the goal check, on: goal-check
#: contact_unreachable 8 -> 5, touching_04 1.33 -> 0.99 GPU-h per success)
TOUCH_A1_SCREEN_ENV = "ISPATIALGYM_TOUCH_A1_SCREEN"
TOUCH_A1_SCREEN_DEFAULT = "1"
#: owner 5r.1 touching_04 follow-up (2026-11-08): "1" = the touching Phase 2 scans the trunk tilts the touch_push map is built at (its
#: eligible trunks at each translate, within the torso limit) instead of ``tabletop.tilt_scan``'s continuous samples, so the manipulation
#: and push stances (same trunk) read built map columns, not the nearest / inherited ones (c2's place-screen and A1 failures sat at
#: (0.25, -0.53) and (0.0, -0.61)). Only with a touch_push map; enters the start-per-attempt cell cache key when on. Default "0"
TOUCH_BUILT_TILTS_ENV = "ISPATIALGYM_TOUCH_BUILT_TILTS"
#: lead decision 9 (2026-11-08, owner away; i7-contact3's symbolic_only diagnosis: the touching_04 spray bottle turns during the push, the
#: push line ~4 mm off its root, the turn's sign that of an off-centre push in 37 / 46): "1" = the touching push's contact line (the swept
#: and the open-gripper classes) runs through the target's MASS CENTRE (PhysX link COMs, mass-weighted; the AABB centre without masses) at
#: its place instead of its footprint centre (contact_poses.push_line_t0). An oracle push change (the recorded push poses); the touch_push
#: maps / library keep the footprint line. Default "0"
PUSH_LINE_COM_ENV = "ISPATIALGYM_PUSH_LINE_COM"
#: i7-nextto 2026-11-09 (lead round 4, nextto_02's place screening): K > 0 = when the segment draw of a candidate whose grasp already passed
#: dies on a rejection the partner pose decides - the release sweep (:meth:`IORSSmall.avoid_release_sweep`), the place / pre-place frame
#: test, the init-image test of the partner, or the place screening's exact IK - the partner is drawn again (fresh :meth:`place_partner`
#: draw, every rule unchanged) up to K more times with the same grasp, before the candidate is given up (int2_10b nextto_02: 50 of the
#: logged candidate deaths were the place screening's exact IK, each after a full grasp search). Only for the relations of
#: :data:`PLACE_PARTNER_RETRY_RELATIONS_ENV` (default "nextto"). Default "0" (off). Kit ``place_retry.armed`` / ``.retried.<reason>`` /
#: ``.rescued`` / ``.exhausted``
#: i7-nextto 2026-11-09 (round 2): "1" = nextto's grasp tiers are each split into the top-down grasps (approach within 45 deg of straight
#: down at the target's rest pose) first and the other grasps after them, so F is searched over the top grasps and only when they have no
#: feasible (yaw, grasp) at the focus over the rest (the flush / inclined tiers keep their order). Every rule unchanged; which grasp a
#: candidate takes changes. Measured on nextto_02 (salt shaker): every written instance of int1_1 / int2_10b (7 / 7) holds a top grasp,
#: and every place-screening failure of i7nx_n02_off / _on (20 / 20, 74 / 74) a side grasp. "0" = off. DEFAULT "1" since the i7-nextto
#: A/B (2026-11-09, nextto_02 300119 N 4, svl titanrtx, per-arm empty caches: off 41 / 3 / 3 0.229 GPU-h per ok, on 14 / 4 / 4 0.097, the top
#: tier taken 7 times; control nextto_04 300137: every grasp top-down, the same 26 attempts / 4 / 4 in both arms). Kit ``top_first.top`` /
#: ``.rest`` (the tier the grasp came from), ``.no_top`` (a tier without a top grasp)
NEXTTO_TOP_FIRST_ENV = "ISPATIALGYM_NEXTTO_TOP_FIRST"
NEXTTO_TOP_FIRST_DEFAULT = "1"
PLACE_PARTNER_RETRY_ENV = "ISPATIALGYM_PLACE_PARTNER_RETRY"
PLACE_PARTNER_RETRY_RELATIONS_ENV = "ISPATIALGYM_PLACE_PARTNER_RETRY_RELATIONS"
#: the segment-draw rejections a fresh partner pose can change (the place screening's own codes are marked by ``_place_screen_reject``)
PLACE_PARTNER_RETRY_REASONS = ("release_sweep", "gripper_not_in_frame", "not_in_frame")
#: i7-symonly 2026-11-09 (touching_04 symbolic_only: 50 of its 58 recorded symbolic_only demos and 10 of 11 not_settled pushed the spray
#: bottle at 7-13 cm - it tilted 11-22 deg and turned past the 0.1 rad tolerance; at 5-6 cm ok 33 / symbolic_only 8 of 50): the swept
#: touching push's contact height capped at the target's tip height (``contact_poses.push_tip_height_cap``: ``generation.
#: push_tip_height_ratio`` x the footprint's leading half-depth along the push, never below the lowest fingertip push) - in the exact
#: choice (:meth:`IORSSmall._touching_push_choice`: the push stance search, the goal check's push, the recorded push) and in the touch_push
#: map lookups of the push stance candidates (the A1 screen reads the same). A draw with no low enough contact fails contact_unreachable
#: and another is drawn. "0" off (default), "shadow" (computed and counted, nothing dropped), "1" on. Only under the push height sweep
#: (touching tabletop). Kit ``push_tip_cap.checked`` / ``.dropped`` (candidates above the cap skipped) / ``.chosen_above`` (shadow: the
#: chosen class is above the cap) / ``.map_lost`` (a map lookup feasible only above the cap)
PUSH_TIP_CAP_ENV = "ISPATIALGYM_PUSH_TIP_CAP"
PUSH_TIP_CAP_DEFAULT = "0"
#: i7-t04push 2026-11-09 (owner A7 (1): "apply the push tip cap at the start draw"): the push the goal check will look for, screened when the
#: candidate is drawn - the swept / open classes for the drawn place pose and the edge's push direction (capped at the tip height when
#: :data:`PUSH_TIP_CAP_ENV` is on), each tried against :meth:`IORSSmall.push_hand_overlap` (geometry only: the hand beyond the fingers vs the
#: target's hull, at the contact and at the push goal). No class clearing it = the goal check's "no push class passes from any push stance
#: lateral" (the overlap does not depend on the stance): sym_t04_on (TIP_CAP=1) lost 17 attempts there, 1076 s, every one on the knuckle /
#: gripper body 11-16 mm inside the hull. Rejected here the next candidate is drawn in the same attempt. "0" off (default), "shadow" (counted,
#: nothing rejected), "1" on. Touching tabletop under the push height sweep only; not with ISPATIALGYM_HAND_OVERLAP_RANK (overlaps are then
#: tried, not refused). r2 (tp_t04_capscr: the overlap alone checked 28, rejected 0 - the 5 / 6 cm tips cleared the hull and then failed the
#: exact IK from every A1 push stance): the goal check's exact push chain + place from its A1 stances is screened too. Kit
#: ``push_tip_screen.checked`` / ``.passed`` / ``.rejected`` (+ ``_overlap`` / ``_exact``) / ``.shadow_would_reject`` / ``.uncertain``
PUSH_TIP_SCREEN_ENV = "ISPATIALGYM_PUSH_TIP_SCREEN"
PUSH_TIP_SCREEN_DEFAULT = "0"
#: i7-t04push 2026-11-09 (owner A7 (2): "try short strokes for the tabletop touching push"): ``ISPATIALGYM_TOUCH_PUSH_STROKE_M=<m>`` records a
#: fingertip push (``push_tip_h`` / ``push_tip_v-``, closed hand) as strokes of at most that length for the oracle's ``execute._push_strokes``
#: (today only the floor push's ``ISPATIALGYM_FLOORPUSH_STROKE_M``): before each stroke the target is held still and measured, the tips
#: re-aim where the line through its root meets its back (with a yaw term, within :data:`TOUCH_PUSH_STROKE_REAIM_MAX_M`), between strokes
#: the hand backs off :data:`TOUCH_PUSH_STROKE_BACKOFF_M`. The record adds the tips' geometry (``face_half_m`` = FINGERTIP_LENGTH,
#: ``blade_standoff_m`` = that + the stop clearance, ``contact_strips_m``, ``contact_band_m`` +-3 cm about the tips - r1's +-1 cm held too few
#: hull vertices and the stroke fell back to the root -, ``half_extent_m`` the root's depth behind the contact). Unset / "off" / "0" =
#: one stroke (default). Kit ``touch_push_strokes.recorded`` / ``.skipped_class``
TOUCH_PUSH_STROKE_ENV = "ISPATIALGYM_TOUCH_PUSH_STROKE_M"
TOUCH_PUSH_STROKE_BACKOFF_M = 0.02
TOUCH_PUSH_STROKE_REAIM_MAX_M = 0.015
TOUCH_PUSH_STROKE_YAW_GAIN_M_PER_RAD = 0.15
TOUCH_PUSH_STROKES_MAX = 6
TOUCH_PUSH_STROKE_BAND_M = (-0.03, 0.03)
#: oracle-fix 2026-11-07 (owner prompt E 3, ISPATIALGYM_FRAME_RANK): valid partner poses ranked by the place frame margin
FRAME_RANK_K = 4
#: A1: band cells within this depth of the target count for its reach rule (as the partner placement always cut the band)
BAND_DEPTH_WINDOW_M = 0.12
#: A1: extras key of a Phase-2 candidate: ``{"left" | "right": [arm, ...]}``, the arms whose band meets the side's reach rule there
REGION_ARMS = "start_region_arms"
#: A5: extras key of a Phase-2 candidate: ``{"left" | "right": [spot index, ...]}``, the meta-link spots the partner may take there
META_SPOTS = "meta_link_spots"
#: 7e (owner 2026-10-12): the rejections of the init -> manipulation travel (``SamplerV2.record_stance_travel``: transition, departure,
#: edge path) that repeat on every redraw of a candidate (:func:`travel_failure_deterministic`)
TRAVEL_REASONS = ("no_transition", "no_path", "path_collision", "path_too_long")
#: 7e: extras key of a Phase-2 candidate dropped after such a failure (the failure)
TRAVEL_DROPPED = "travel_dropped"
#: train-fast 2026-10-31 (``ISPATIALGYM_FAST_SPA_CACHE``): the cache kind of a start-per-attempt lateral cell (:meth:`IORSSmall._spa_cell`); bump
#: when what a cell holds or how it is computed changes
SPA_CELL_KIND = "spa_cell_v1"
#: the rejections of one lateral cell that the per-attempt draw retries on another cell / edge (every other rejection ends the attempt)
SPA_CELL_REJECTS = ("no_workspace", "edge_too_short", "stance_collision", "shelf_width", "no_manipulation_stance", "no_start_region")
_SPA_CACHE_ANNOUNCED = False


def place_depth_cap_redraw_on() -> bool:
    return (os.environ.get(PLACE_DEPTH_CAP_REDRAW_ENV) or "0").strip() == "1"


def place_depth_cap_draw_on() -> bool:
    return (os.environ.get(PLACE_DEPTH_CAP_DRAW_ENV) or PLACE_DEPTH_CAP_DRAW_DEFAULT).strip() == "1"


def place_depth_cap_basis() -> str:
    return "a1" if (os.environ.get(PLACE_DEPTH_CAP_BASIS_ENV) or "map").strip().lower() == "a1" else "map"


def touch_lift_screen_mode() -> str:
    v = (os.environ.get(TOUCH_LIFT_SCREEN_ENV) or TOUCH_LIFT_SCREEN_DEFAULT).strip().lower()
    return "on" if v in ("1", "on") else ("shadow" if v == "shadow" else "off")


def touch_a1_screen_mode() -> str:
    v = (os.environ.get(TOUCH_A1_SCREEN_ENV) or TOUCH_A1_SCREEN_DEFAULT).strip().lower()
    return "on" if v in ("1", "on") else ("shadow" if v == "shadow" else "off")


def push_line_com_on() -> bool:
    if (os.environ.get(PUSH_LINE_COM_ENV) or "0").strip() == "1":
        return True
    from ispatialgym.sampling import campaign_runtime
    return campaign_runtime.contact_recovery('touching_push_mass_center')


def _tip_keep(cap: float):
    """A ``ContactMap.lookup`` ``keep`` for :data:`PUSH_TIP_CAP_ENV`: a class row whose label carries a contact height
    (``push_tip_h-@5.0cm``) at most ``cap`` m; a row without one is kept."""
    import re

    def keep(row) -> bool:
        m = re.search(r"@([0-9.]+)cm", str((row or {}).get("class") or ""))
        return m is None or float(m.group(1)) / 100.0 <= cap + 1e-6
    return keep


def touch_push_stroke_m() -> Optional[float]:
    """:data:`TOUCH_PUSH_STROKE_ENV` as a length (m), None when unset / "off" / "0"; ``ValueError`` for a length that is not positive."""
    v = (os.environ.get(TOUCH_PUSH_STROKE_ENV) or "").strip().lower()
    if v in ("", "off", "0", "none"):
        return None
    m = float(v)
    if not m > 0.0:
        raise ValueError(f"{TOUCH_PUSH_STROKE_ENV}={v!r}: a positive length (m)")
    return m


def push_tip_screen_mode() -> str:
    """``"off"`` / ``"shadow"`` / ``"on"`` from :data:`PUSH_TIP_SCREEN_ENV` (unset or empty: :data:`PUSH_TIP_SCREEN_DEFAULT`)."""
    v = (os.environ.get(PUSH_TIP_SCREEN_ENV) or PUSH_TIP_SCREEN_DEFAULT).strip().lower()
    return "on" if v in ("1", "on") else ("shadow" if v == "shadow" else "off")


def push_tip_cap_mode() -> str:
    """``"off"`` / ``"shadow"`` / ``"on"`` from :data:`PUSH_TIP_CAP_ENV` (unset or empty: :data:`PUSH_TIP_CAP_DEFAULT`)."""
    v = (os.environ.get(PUSH_TIP_CAP_ENV) or PUSH_TIP_CAP_DEFAULT).strip().lower()
    return "on" if v in ("1", "on") else ("shadow" if v == "shadow" else "off")


def touch_built_tilts_on() -> bool:
    return (os.environ.get(TOUCH_BUILT_TILTS_ENV) or "0").strip() == "1"


def nextto_top_first_on() -> bool:
    return (os.environ.get(NEXTTO_TOP_FIRST_ENV) or NEXTTO_TOP_FIRST_DEFAULT).strip() == "1"


def top_down_grasp(eef_in_object, rest_quat, cos_max: float = -math.sqrt(0.5)) -> bool:
    """The grasp's approach axis (eef z) points within 45 deg of straight down when the target rests at ``rest_quat``; False when the
    pose cannot be read."""
    try:
        a_obj = C.quat_to_mat(G.t(eef_in_object[1]).float())[:, 2]
        a_world = G.rotate(G.t(rest_quat).float(), a_obj)
    except Exception:  # noqa: BLE001 - an unreadable grasp pose is not a top grasp
        return False
    return float(a_world[2]) <= cos_max


def place_partner_retries(relation: Optional[str]) -> int:
    """:data:`PLACE_PARTNER_RETRY_ENV` for ``relation`` (0 = off, also for a relation outside :data:`PLACE_PARTNER_RETRY_RELATIONS_ENV`)."""
    try:
        k = int((os.environ.get(PLACE_PARTNER_RETRY_ENV) or "0").strip())
    except ValueError:
        return 0
    rels = {r.strip() for r in (os.environ.get(PLACE_PARTNER_RETRY_RELATIONS_ENV) or "nextto").split(",") if r.strip()}
    return max(0, k) if relation in rels else 0


@contextmanager
def phase_timer(s, name: str):
    """i7-nextto 2026-11-09 (measurement only): the wall time of one candidate phase into the kit counters ``<name>_calls`` / ``<name>_s``
    (the attempt's kit line prints them as ``<name> N calls Xs``)."""
    t0 = time.time()
    try:
        yield
    finally:
        st = getattr(getattr(s, "kit", None), "stats", None)
        if isinstance(st, dict):
            st[name + "_calls"] = st.get(name + "_calls", 0) + 1
            st[name + "_s"] = st.get(name + "_s", 0.0) + (time.time() - t0)


def place_depth_cap_mode() -> str:
    """``"off"`` / ``"shadow"`` / ``"on"`` from :data:`PLACE_DEPTH_CAP_ENV` (unset or empty: :data:`PLACE_DEPTH_CAP_DEFAULT`)."""
    v = (os.environ.get(PLACE_DEPTH_CAP_ENV) or PLACE_DEPTH_CAP_DEFAULT).strip().lower()
    if v in ("1", "on", "true", "yes"):
        return "on"
    return "shadow" if v == "shadow" else "off"


def hand_overlap_rank_on() -> bool:
    """i7-contact 2026-11-07, owner item 3 (``ISPATIALGYM_HAND_OVERLAP_RANK``, default "0" = OFF; the owner decides): the touching PUSH's hand-overlap
    screen (:meth:`IORSSmall.push_hand_overlap`: wrist camera / knuckles / gripper body within 5 mm of the target) is a RANKING hint instead of
    a rejection - the candidates without an overlap are tried first, the overlapping ones after them in their order, each decided by the
    exact chain IK (and, in the library, by physics validation: ``validate_contacts --screen-as-rank``); the chosen push's extras name the
    overlap (``hand_overlap_hint``). The brace's screen is unchanged. Measured (screened diagnostic, contact_libs_20261107e_screendiag): the
    hand model is the real gripper (its base plate spans +-54 mm, the knuckle spheres +-57 mm), the overlaps are real: physically valid
    napkins 42 / 48, cereal 2 / 9, spray 1 / 23, and the napkins' screened pushes press the box into the slab with 143 N median at its first
    motion (live pushes 4 N)."""
    return os.environ.get("ISPATIALGYM_HAND_OVERLAP_RANK", "0").strip() == "1"


def push_fingers(extras: Optional[Dict[str, Any]]) -> float:
    """oracle-fix 2026-11-06 (``generation.push_gripper_open``): the finger joints a push candidate / record is checked with - its
    ``fingers_opening_m`` (the open push: the oracle holds the gripper there, ``execute_push``; the program check replays with it), else
    closed."""
    v = (extras or {}).get("fingers_opening_m")
    return IK.FINGER_CLOSED if v is None else float(v)


def _spa_json_default(o):
    """``json.dumps`` fallback of a lateral cell: numpy scalars, tensors, sets and records (never a crash of the train job over a type)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, th.Tensor):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "to_sidecar"):
        return o.to_sidecar()
    return str(o)


def travel_failure_deterministic(err: Reject, world) -> bool:
    """7e: whether a failed init -> manipulation travel of a candidate fails the same way on every redraw. The check is a pure function of
    the init stance (fixed by ``sample_template``), the candidate's work stance (base pose, trunk, stand-off, the template's working arm,
    the idle arm's Phase-2 posture), the scene objects (reset to the scene file before every attempt), the traversability maps and the
    task objects; it draws nothing (no IK: fixed s_T list, trunk-sweep samples and drive strides, grid A* on the maps, collision checks).
    So it repeats when ``err`` is one of :data:`TRAVEL_REASONS` and every spawned object stands at its park slot (``SimWorld.park``: every
    rejected attempt parks them) - after a written instance they stand where it left them, and that failure is not taken as final."""
    if err.reason not in TRAVEL_REASONS:
        return False
    for i, obj in enumerate(world.spawned.values()):
        slot = PARK_POSITION + th.tensor([float(i), 0.0, 0.0])
        if float(th.linalg.norm(G.t(obj.get_position_orientation()[0]).float() - slot)) > 0.05:
            return False
    return True


def merge_shelf_pose_facts(start: Dict[str, Any], facts: Dict[str, Any]) -> None:
    """``tabletop.shelf_lean_pose``'s facts into the shelf start. They name the side (``"left"`` / ``"right"``); it is kept as
    ``lean_side_name`` so ``lean_side`` stays the sign (+1 = the robot's left) the push, the partner's lean and the goal side
    compute with."""
    start.update({("lean_side_name" if k == "lean_side" else k): v for k, v in facts.items()})


def shelf_free_span(t_hull_w: th.Tensor, p_hull_w: th.Tensor, edge: EdgeFrame, side: float) -> float:
    """The free span (m) along ``edge``'s tangent between the two leaning books (world hull points) of the shelf layout, the target
    against the board on ``side`` (+1 = the robot's left) and the partner against the other - ``sample_template``'s measure."""
    t_lat = (t_hull_w[:, :2] - edge.mid) @ edge.tangent
    p_lat = (p_hull_w[:, :2] - edge.mid) @ edge.tangent
    return float(t_lat.min() - p_lat.max()) if side > 0 else float(p_lat.min() - t_lat.max())


def shelf_draw_on(s) -> bool:
    """``generation.shelf_start_draw`` is on for ``s``'s template (``IORSSmall.shelf_start_draw``: the shelf layout with the knob). False for an
    object without a shelf start or the knob - the methods that ask also run on stand-ins."""
    if getattr(s, "shelf_start", None) is None or not callable(getattr(s, "gen", None)) or s.gen("shelf_start_draw") is None:
        return False
    return IORSSmall.shelf_start_draw(s) is not None


def shelf_goal_dz(start: Optional[Dict[str, Any]], hull: th.Tensor, t_pose: G.Pose, goal_quat) -> float:
    """oracle-fix 2026-11-06 (``generation.shelf_start_draw``): the height change that keeps the target's lowest hull point at its start's when
    the drafted goal leans another angle than the start (``start["partner_lean_deg"]``: the goal is drafted parallel to the partner) - at
    11 vs 20 deg the book's bottom would sit 2.3 mm above or inside the board; 0 without a drawn partner lean (the mirrored lean is the
    start's own: the goal keeps the start's height, as before)."""
    if not start or start.get("partner_lean_deg") is None:
        return 0.0
    z0 = float(G.local_points_world(hull, (th.zeros(3), G.t(t_pose[1])))[:, 2].min())
    z1 = float(G.local_points_world(hull, (th.zeros(3), G.t(goal_quat)))[:, 2].min())
    return z0 - z1


def partner_sides(relation: str, handed: Optional[Dict[str, str]] = None, shelf_lean: Optional[float] = None) -> List[float]:
    """A1 step 3 (owner 2026-10-12): the partner sides (+1 = the robot's left of the target, -1 = its right) a pair admits - touching the
    robot's left (the right hand pushes toward it, the left braces its far side, §6.11), in the shelf layout (``shelf_lean`` = the
    target's lean side) the side opposite the lean (the partner leans against the other board), a handed nextto pair
    (``handedness_pair``) the side that leaves the target on its own hand's side, otherwise both."""
    from ispatialgym.sampling import pair_rules as PR

    if relation == "touching":
        return [-float(shelf_lean)] if shelf_lean is not None else [1.0]
    if handed:
        return [PR.partner_side_for(handed["target_side"])]
    return [1.0, -1.0]


def region_sides(allowed: Sequence[float], assigned: Optional[float], cells: Dict[float, int]) -> List[float]:
    """A1 step 3: the partner sides an attempt draws from - eval (``assigned`` = ``generation.partner_side``) that side alone when its
    start region has a cell, train every allowed side whose region has one (``cells``: side -> start cells), in ``allowed``'s order.
    Empty: the robot's lateral is redrawn, then the template rejected."""
    if assigned is not None:
        return [assigned] if cells.get(assigned, 0) > 0 else []
    return [sd for sd in allowed if cells.get(sd, 0) > 0]


def footprint_laterals(pts: th.Tensor, tangent) -> Tuple[float, float]:
    """(min, max) of points (a footprint polygon, a hull) along the horizontal ``tangent``."""
    v = G.t(pts)[:, :2].float() @ G.t(tangent)[:2].float()
    return float(v.min()), float(v.max())


def target_aligned_partner_yaws(t_fp, p_fp_rest, across: bool) -> List[float]:
    """``generation.partner_aligned_to_target``: the partner yaws (rad, about world z, applied to its rest orientation as ``draw_partner_pose``
    applies them) that lay the longer side of its footprint along the target's long axis - ``t_fp`` the target's footprint at its init pose,
    ``p_fp_rest`` the partner's at its rest orientation: the two half a turn apart, then with ``across`` the two a quarter turn from them (the
    partner's long side across the target)."""
    y = G.wrap_angle(min_area_rect(t_fp)[0] - min_area_rect(p_fp_rest)[0])
    out = [y, G.wrap_angle(y + math.pi)]
    if across:
        out += [G.wrap_angle(y + math.pi / 2), G.wrap_angle(y - math.pi / 2)]
    return out


def long_axes_angle_deg(a_fp, b_fp) -> float:
    """The angle (deg, 0-90) between the long axes of two footprints' minimum-area rectangles (0 = parallel)."""
    d = abs(min_area_rect(a_fp)[0] - min_area_rect(b_fp)[0]) % math.pi
    return math.degrees(min(d, math.pi - d))


def axes_angle_deg(a: float, b: float) -> float:
    """The angle (deg, 0-90) between two horizontal axes given by their world angles (rad, either direction)."""
    d = (float(a) - float(b)) % math.pi
    return math.degrees(min(d, math.pi - d))


def obb_long_axis(metadata: Dict[str, Any]) -> Optional[th.Tensor]:
    """The long axis (unit, object frame) of a model's oriented bounding box - ``link_bounding_boxes.base_link.collision.oriented`` of its
    ``misc/metadata.json``: the box axis (a column of its ``transform``) of the largest ``extent``; None without the box, or when the next
    extent is within 10 % of it (no long axis)."""
    box = (((metadata.get("link_bounding_boxes") or {}).get("base_link") or {}).get("collision") or {}).get("oriented") or {}
    if "extent" not in box or "transform" not in box:
        return None
    ext = [float(v) for v in box["extent"]]
    i, j = sorted(range(3), key=lambda k: -ext[k])[:2]
    if ext[j] >= 0.9 * ext[i]:
        return None
    return th.tensor(box["transform"], dtype=th.float32)[:3, i]


def raster_from_above(tri: np.ndarray, x0: float, y0: float, nx: int, ny: int, res: float) -> Tuple[np.ndarray, np.ndarray]:
    """What is seen from straight above: per cell centre ``(x0 + (i + 0.5) res, y0 + (j + 0.5) res)`` the z of the highest triangle of ``tri``
    (M, 3, 3; world) over it and that triangle's |normal z|, NaN where none covers it (vertical triangles are not seen)."""
    zbuf, nbuf = np.full((nx, ny), np.nan), np.full((nx, ny), np.nan)
    tri = np.asarray(tri, dtype=float).reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1)
    for k in np.nonzero(nn > 1e-12)[0]:
        anz = abs(float(n[k, 2])) / float(nn[k])
        if anz < 1e-3:
            continue
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri[k]
        lo, hi = tri[k, :, :2].min(axis=0), tri[k, :, :2].max(axis=0)
        i0, i1 = max(int(math.floor((lo[0] - x0) / res - 0.5)), 0), min(int(math.ceil((hi[0] - x0) / res - 0.5)), nx - 1)
        j0, j1 = max(int(math.floor((lo[1] - y0) / res - 0.5)), 0), min(int(math.ceil((hi[1] - y0) / res - 0.5)), ny - 1)
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if i1 < i0 or j1 < j0 or abs(d) < 1e-15:
            continue
        X, Y = np.meshgrid(x0 + (np.arange(i0, i1 + 1) + 0.5) * res, y0 + (np.arange(j0, j1 + 1) + 0.5) * res, indexing="ij")
        w1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / d
        w2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / d
        w3 = 1.0 - w1 - w2
        z = w1 * az + w2 * bz + w3 * cz
        sub_z, sub_n = zbuf[i0:i1 + 1, j0:j1 + 1], nbuf[i0:i1 + 1, j0:j1 + 1]
        better = (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9) & (np.isnan(sub_z) | (z > sub_z))
        sub_z[better], sub_n[better] = z[better], anz
    return zbuf, nbuf


def polygon_signed_distance(pts, poly) -> np.ndarray:
    """Signed distance (m) of the points ``pts`` (N, 2) from the convex CCW polygon ``poly`` (V, 2): the largest distance beyond its edge lines
    (positive outside, negative inside; beyond a corner the distance to the nearer edge line)."""
    pts, poly = np.asarray(pts, dtype=float).reshape(-1, 2), np.asarray(poly, dtype=float).reshape(-1, 2)
    e = np.roll(poly, -1, axis=0) - poly
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)  # outward for a CCW polygon
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return ((pts[:, None, :] - poly[None, :, :]) * nrm[None, :, :]).sum(axis=2).max(axis=1)


class FlatTop:
    """``generation.partner_flat_top`` (under_small#04, owner 2026-10-13 ~18:40: "move the sheet to a flat part of the support"): the cells of a
    support's top that are seen from straight above as one flat face - the highest face over the cell centre horizontal (|normal z| >=
    ``normal_min``) within ``tol`` of the top's height ``z_ref``. A seam between two table units, an edge, a rim, a lip or an opening is not
    flat. Cell (i, j) is centred at ``(x0 + (i + 0.5) res, y0 + (j + 0.5) res)``; ``z`` the seen height (NaN where nothing is seen)."""

    def __init__(self, x0: float, y0: float, res: float, z: np.ndarray, flat: np.ndarray, z_ref: float):
        self.x0, self.y0, self.res, self.z, self.flat, self.z_ref = float(x0), float(y0), float(res), np.asarray(z), np.asarray(flat, dtype=bool), float(z_ref)

    @classmethod
    def from_raster(cls, x0: float, y0: float, res: float, zbuf: np.ndarray, nbuf: np.ndarray, z_hint: float, tol: float, normal_min: float,
                    band: float = 0.03) -> "FlatTop":
        """From :func:`raster_from_above`: ``z_ref`` = the most common height (1 mm bins) of the horizontal cells within ``band`` of ``z_hint``
        (the collision top's height), else ``z_hint``."""
        horiz = ~np.isnan(zbuf) & (np.nan_to_num(nbuf) >= normal_min) & (np.abs(np.nan_to_num(zbuf, nan=-1e9) - z_hint) <= band)
        z_ref = float(z_hint)
        if horiz.any():
            bins, counts = np.unique(np.round(zbuf[horiz] * 1000.0).astype(np.int64), return_counts=True)
            mode = int(bins[int(np.argmax(counts))])
            z_ref = float(np.median(zbuf[horiz][np.round(zbuf[horiz] * 1000.0).astype(np.int64) == mode]))
        flat = horiz & (np.abs(np.nan_to_num(zbuf, nan=-1e9) - z_ref) <= tol)
        return cls(x0, y0, res, zbuf, flat, z_ref)

    def _window(self, lo, hi) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The cells centred within the world box ``lo`` .. ``hi``: their centres' x and y, and their flat / z sub-arrays (a cell outside the
        raster is not flat, its z NaN)."""
        i = np.arange(math.ceil((float(lo[0]) - self.x0) / self.res - 0.5), math.floor((float(hi[0]) - self.x0) / self.res - 0.5) + 1)
        j = np.arange(math.ceil((float(lo[1]) - self.y0) / self.res - 0.5), math.floor((float(hi[1]) - self.y0) / self.res - 0.5) + 1)
        flat, z = np.zeros((len(i), len(j)), dtype=bool), np.full((len(i), len(j)), np.nan)
        ii, jj = (i >= 0) & (i < self.flat.shape[0]), (j >= 0) & (j < self.flat.shape[1])
        if ii.any() and jj.any():
            flat[np.ix_(ii, jj)] = self.flat[np.ix_(i[ii], j[jj])]
            z[np.ix_(ii, jj)] = self.z[np.ix_(i[ii], j[jj])]
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res, flat, z

    def clearance(self, poly, reach: float) -> float:
        """The distance (m) from the footprint polygon ``poly`` (world xy, convex CCW) to the nearest cell centre that is not flat, looked for
        within ``reach`` of it; ``reach`` when none is nearer, negative when one lies inside it."""
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        xs, ys, flat, _ = self._window(p.min(axis=0) - reach, p.max(axis=0) + reach)
        a, b = np.nonzero(~flat)
        if not len(a):
            return float(reach)
        return min(float(reach), float(polygon_signed_distance(np.stack([xs[a], ys[b]], axis=1), p).min()))

    def heights(self, poly) -> Optional[Tuple[float, float]]:
        """(min, max) of the seen height over the cells whose centres lie inside ``poly``; None when it covers none."""
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        xs, ys, _, z = self._window(p.min(axis=0), p.max(axis=0))
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        inside = (polygon_signed_distance(np.stack([X.ravel(), Y.ravel()], axis=1), p) <= 0.0).reshape(X.shape) & ~np.isnan(z)
        return (float(z[inside].min()), float(z[inside].max())) if inside.any() else None

    def facts(self) -> Dict[str, Any]:
        return {"res_m": self.res, "z_ref": round(self.z_ref, 5), "n_cells": int(self.flat.size), "n_flat": int(self.flat.sum()),
                "n_seen": int((~np.isnan(self.z)).sum())}


class ContainerFloor:
    """oracle-fix 2026-11-06 (``generation.ontop_centre: "container"``, owner on the ontop_small#02 steak demo): the floor of an open
    container seen from straight above - the convex collision pieces (points, an upright frame: z up) as triangles (their convex hulls),
    rastered (:func:`raster_from_above`) at ``res``; the floor = the horizontal cells (|n_z| >= ``normal_min``) within ``tol`` of the lowest
    1 mm height that holds at least ``min_frac`` of the horizontal cells (a pan's rim top and handle lie above it); its centre = the centre of
    the largest disc of floor cells (the floor cells farthest from every other cell, :func:`scipy.ndimage.distance_transform_edt`), its
    radius that disc's. :meth:`radius_at` - the disc of floor around another centre (a fillable meta link's). ``build`` returns None when no
    floor is seen."""

    def __init__(self, x0: float, y0: float, res: float, z: float, dist: np.ndarray, centre_xy: np.ndarray, n_flat: int):
        self.x0, self.y0, self.res, self.z, self.dist, self.centre_xy, self.n_flat = float(x0), float(y0), float(res), float(z), dist, np.asarray(centre_xy, dtype=float), int(n_flat)

    @classmethod
    def build(cls, pieces: Sequence[Any], res: float = 0.002, tol: float = 0.002, normal_min: float = 0.99, min_frac: float = 0.1) -> Optional["ContainerFloor"]:
        from scipy import ndimage
        from scipy.spatial import ConvexHull

        tris = []
        for P in pieces:
            P = np.asarray(G.t(P).double().numpy() if isinstance(P, th.Tensor) else P, dtype=float).reshape(-1, 3)
            if len(P) < 4:
                continue
            try:
                tris.append(P[ConvexHull(P).simplices])
            except Exception:  # noqa: BLE001 - a flat piece: nothing seen from above that its neighbours do not show
                continue
        if not tris:
            return None
        tri = np.concatenate(tris)
        lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
        x0, y0 = float(lo[0]) - 2 * res, float(lo[1]) - 2 * res
        nx, ny = int(math.ceil((float(hi[0]) - x0) / res)) + 3, int(math.ceil((float(hi[1]) - y0) / res)) + 3
        zbuf, nbuf = raster_from_above(tri, x0, y0, nx, ny, res)
        horiz = ~np.isnan(zbuf) & (np.nan_to_num(nbuf) >= normal_min)
        if not horiz.any():
            return None
        zn = np.nan_to_num(zbuf, nan=-1e9)
        flat = None
        for b in np.unique(np.round(zbuf[horiz] * 1000.0).astype(np.int64)):  # ascending: the lowest band of ``tol`` that holds enough of the floor
            band = horiz & (zn >= (b - 0.5) / 1000.0) & (zn <= (b - 0.5) / 1000.0 + tol)
            if band.sum() >= max(4, min_frac * horiz.sum()):
                flat = band
                break
        if flat is None:
            return None
        z_ref = float(np.median(zbuf[flat]))
        dist = ndimage.distance_transform_edt(flat) * res
        top = dist >= float(dist.max()) - 0.5 * res
        ii, jj = np.nonzero(top)
        centre = np.array([x0 + (float(ii.mean()) + 0.5) * res, y0 + (float(jj.mean()) + 0.5) * res])
        return cls(x0, y0, res, z_ref, dist, centre, int(flat.sum()))

    def radius_at(self, xy) -> float:
        """The radius (m) of the floor disc around ``xy`` (the upright frame): the distance from its cell to the nearest cell that is not
        floor, less half a cell; 0 off the floor."""
        i, j = int(math.floor((float(xy[0]) - self.x0) / self.res)), int(math.floor((float(xy[1]) - self.y0) / self.res))
        if not (0 <= i < self.dist.shape[0] and 0 <= j < self.dist.shape[1]):
            return 0.0
        return max(0.0, float(self.dist[i, j]) - 0.5 * self.res)

    @property
    def radius(self) -> float:
        return self.radius_at(self.centre_xy)


def gap_along(behind_w: th.Tensor, ahead_w: th.Tensor, direction) -> float:
    """Gap (m) along ``direction`` (world, horizontal) from a hull to the hull ahead of it, both world points: the nearest extent of
    ``ahead_w`` minus the farthest of ``behind_w`` (negative when they overlap along it)."""
    d = th.tensor([float(direction[0]), float(direction[1])])
    d = d / th.linalg.norm(d)
    return float((ahead_w[:, :2] @ d).min() - (behind_w[:, :2] @ d).max())


def tremble_screen_spots(top: SupportTop, fp: th.Tensor, n: int, margin: float = 0.02) -> List[th.Tensor]:
    """7g: up to ``n`` free cells of ``top`` (world xy) that take the footprint ``fp`` (about the cell, (V, 2)) on free cells with
    ``margin`` (``tabletop.polygons_supported``): the one nearest the middle of those cells first, then each next one farthest from the
    spots already taken. Empty when none does."""
    cells = top.cells(free_only=True)
    if n <= 0 or len(cells) == 0:
        return []
    cells = cells[TT.polygons_supported(top, cells[:, None, :] + G.t(fp).float()[None, :, :2], margin)]
    if len(cells) == 0:
        return []
    spots = [cells[int(th.argmin(th.linalg.norm(cells - cells.mean(dim=0), dim=1)))]]
    while len(spots) < min(int(n), len(cells)):
        d = th.stack([th.linalg.norm(cells - sp, dim=1) for sp in spots]).min(dim=0).values
        spots.append(cells[int(th.argmax(d))])
    return spots


def gripper_corners_world(eef: G.Pose, open_width: float = C.GRIPPER_OPEN_M, thickness: float = FINGER_THICKNESS,
                          half_width: float = FINGER_HALF_WIDTH) -> th.Tensor:
    """World corners (8, 3) of the open gripper's finger envelope for an eef pose: ``PAD_Z_RANGE[0]`` .. ``FINGERTIP_LENGTH``
    along the approach axis a, ±(``open_width`` / 2 + ``thickness``) along the closing axis c, ±``half_width`` along eef x.
    ``min(..., 2)`` is the lowest point of the open gripper (release-height rule of §6.11 inside)."""
    pos, quat = G.t(eef[0]), G.t(eef[1])
    R = C.quat_to_mat(quat)
    a, c, x = R[:, 2], R[:, 1], R[:, 0]
    pts = []
    for za in (float(C.PAD_Z_RANGE[0]), float(C.FINGERTIP_LENGTH)):
        for sc in (-1.0, 1.0):
            for sx in (-1.0, 1.0):
                pts.append(pos + za * a + sc * (0.5 * float(open_width) + float(thickness)) * c + sx * float(half_width) * x)
    return th.stack(pts, dim=0)


def hull_edge_points(pts: th.Tensor, spacing: float = 0.003) -> th.Tensor:
    """The convex hull's vertices of ``pts`` (N, 3) plus points along its edges at most ``spacing`` apart: a slender hull (a chopstick's
    rings at its ends and middle) has no vertex where the open jaw stands, though its edges pass through it."""
    from scipy.spatial import ConvexHull

    p = G.t(pts).double().numpy()
    try:
        hull = ConvexHull(p)
    except Exception:  # noqa: BLE001 - a flat / degenerate set: the points as they are
        return G.t(pts).float()
    edges = {tuple(sorted((int(s[i]), int(s[(i + 1) % 3])))) for s in hull.simplices for i in range(3)}
    out = [p[hull.vertices]]
    for i, j in edges:
        n = int(math.ceil(float(np.linalg.norm(p[j] - p[i])) / float(spacing)))
        if n > 1:
            out.append(p[i] + np.linspace(0.0, 1.0, n + 1)[1:-1, None] * (p[j] - p[i]))
    return th.as_tensor(np.vstack(out), dtype=th.float32)


def place_approached(s, inside: Optional[bool] = None) -> List[Any]:
    """i7-split 2026-11-07: the objects an IORS small place descent closes in on (removed along the whole tracked descent, ``IKChecker.exact``'s
    ``approached``): the partner placed on / next to. The supports were passed too (``approached=ign``), so the support left the world along
    the whole descent and its split (rims / walls / rails above its plane) never applied; they stay in the descent's last-pose ignore (their
    split there), as the program check's place has them (``program_check._approached_for``). ``inside`` (default: ``s.relation``): an inside
    release keeps the partner in the world - nothing approached."""
    inside = (getattr(s, "relation", None) == "inside") if inside is None else bool(inside)
    return [] if inside else [s.obj("partner")]


@register(*RELATION_OF_SUBTASK)
class IORSSmall(SamplerV2):
    """One class for the five small-scale relations; the relation comes from the subtask id."""
    START_DRAW_FAMILY = True  # i7-startdraw: the start base per attempt under ISPATIALGYM_START_DRAW (spa_redraw_start)

    @property
    def overhang_fallback_enabled(self) -> bool:
        """i7-plateqtf2 (owner 2026-11-08 02:35 item 2): the side grasp's overhang fallback in ``tabletop._phase3_draw``
        (``v2/overhang_fallback.py``) - for the relations whose program is ``programs.program_iors_small`` (its pick takes the recorded
        push: ``_pick_actions``); touching's program picks without it."""
        from ispatialgym.sampling.v2 import overhang_fallback as OF

        return OF.on() and getattr(self, "relation", None) != "touching"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = RELATION_OF_SUBTASK[self.subtask]
        if self.gp.bddl is None:  # facade: the v1 writer's definition id of the relation
            self.definition_id = V1_DEFINITION_ID[self.relation]
        self.top: Optional[SupportTop] = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.edge: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.pair: Dict[str, Any] = {}
        self.bands = None
        self.segment_arms: List[str] = ["right"]  # the arms offered for the target's segment (sample_template)
        self.arm_offer_why = ""  # why the other arm is not offered for the template (``tabletop.arm_offer``)
        self.arm_bands: Dict[str, Any] = {}
        self.goal_side: Optional[str] = None  # nextto / touching: partner side the target goes to ("left" / "right", robot frame)
        self.release: Optional[Dict[str, Any]] = None  # inside: release-height facts of :meth:`release_pose`
        self.standing: Optional[Dict[str, Any]] = None  # inside with target_up_axis: the goal orientation's facts (:meth:`standing_goal_quat`)
        self.release_surface_z: Optional[float] = None  # ontop / under: the raycast surface under the goal footprint centre
        self.release_above_m: Optional[float] = None  # ontop / under: the release height above that surface
        self._touching_place_pos: Optional[th.Tensor] = None  # touching: the nextto position the program places at
        self._brace_gate: Optional[Dict[str, Any]] = None  # touching: the A1 joint band gate verdict of the drawn partner position
        self.grasps_flush: List[Any] = []  # the flush-pinch tier of ``self.grasps`` (tried first at every focus)
        self._reselected_q: Optional[th.Tensor] = None  # grasp configuration of the last re-selected grasp
        self.release_sweep: Optional[Dict[str, Any]] = None  # nextto / touching: open-jaw footprint facts of the place
        self.shelf_start: Optional[Dict[str, Any]] = None  # touching with an upright-shelf key: the leaning start (tabletop.shelf_lean_pose)
        self.shelf_pose: Optional[G.Pose] = None
        self.shelf_partner_pose: Optional[G.Pose] = None  # the partner leaning against the opposite board
        self.region_sides: List[float] = []  # A1: the partner sides whose start region has a cell (sample_template)
        self.region_side: Optional[float] = None  # A1: the partner side of the current attempt (+1 = the robot's left of the target)
        self.partner_draw: Optional[Dict[str, Any]] = None  # A1: the drawn partner's side, gap and the gap interval it came from
        self.meta_spots: Optional[List[Tuple[str, th.Tensor]]] = None  # A5: (meta link, world xy) the partner may be centred on
        self.flat_top: Optional[FlatTop] = None  # generation.partner_flat_top: the flat cells of the support's top (build_flat_top)

    # ------------------------------------------------------------------ numbers (customs generation)

    @property
    def margin_support(self) -> float:
        return float(self.gen("support_margin", 0.02))

    @property
    def table_end_margin(self) -> float:
        return float(self.gen("table_end_margin_m", TT.TABLE_END_MARGIN))

    @property
    def gap_min(self) -> float:
        return float(self.gen("footprint_gap_min_m", 0.03))

    @property
    def partner_gap_extra(self) -> float:
        return float(self.gen("partner_gap_extra_m", 0.03))

    @property
    def nextto_gap_extra(self) -> float:
        """oracle-fix 2026-11-06 (owner: "salt shaker / pepper shaker too close together at init - just a little nudge makes the condition
        true"): the nextto init gap's margin over g_NT, ``generation.nextto_gap_extra_m`` (the key the defaults always documented for
        g in [g_NT + 0.03, D_p]; the bound read partner_gap_extra_m, also 0.03 everywhere). Set to 0.12 the target moves >= 0.13 m to its
        goal - beyond the policy position tolerance 0.09 - so a nudge of the partner cannot satisfy the goal at init."""
        return float(self.gen("nextto_gap_extra_m", self.partner_gap_extra))

    @property
    def goal_drop(self) -> float:
        return float(self.gen("goal_drop_m", 0.01))

    def carry_rule(self) -> str:
        """What the carry clears (owner 2026-10-12 item 4A): ``generation.carry_clears`` of :data:`CARRY_CLEARS`; default "auto" for
        touching - the box is set down beside the partner, not over it (touching#02 c2_k0: 7 of 11 attempts failed only the 0.27 m
        lift) -, "partner" for the other relations."""
        rule = str(self.gen("carry_clears", "auto" if self.relation == "touching" else "partner"))
        if rule not in CARRY_CLEARS:
            raise Reject("spec", f"{self.gp.variant}: generation.carry_clears must be one of {list(CARRY_CLEARS)}, got {rule!r}")
        return rule

    def carry_height_over(self, partner: bool) -> float:
        """The lift above the grasp pose (the target's bottom on the table at init) that clears by ``carry_clearance_m`` the partner's
        top (``partner``), else only the table."""
        c = float(self.gen("carry_clearance_m", CARRY_CLEARANCE_M))
        return max(c, TT.height_of(self.hull["partner"], self.rest["partner"]) + c) if partner else c

    # ------------------------------------------------------------------ roles

    def v1_role_variants(self) -> Dict[str, str]:
        return {"partner_support": "support"}

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.variant}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    @property
    def support(self):
        return self.obj("support")

    # ------------------------------------------------------------------ template (§6.11 pair admissibility + §4 phases 1a–2)

    def sample_template(self):
        tgt, prt = self.obj("target"), self.obj("partner")
        self.hull["target"], self.hull["partner"] = self.world.local_hull(tgt), self.world.local_hull(prt)
        # i7-lean 2026-11-07 (owner decision): touching starts an upright-shelf book at the v2 library's LEAN rest when it holds one (the key
        # becomes <key>@<lean tag>: its grasps, its measured rest, its interior map - tabletop.shelf_lean_key); otherwise as before
        self.grasps, self.grasp_key, self.rest["target"] = TT.library_grasps(self, "target", lean=self.relation == "touching")
        # grasps whose attached object self-collides with the gripper palm / wrist (sphere fit vs the robot model) can
        # never pass an attached check (lift, carry, place): they leave the grasp set here, once per template
        attachable = [g for g in self.grasps if self.ikc.attachable(tgt, g.pose, "right")]
        self.template_facts["grasps_dropped_not_attachable"] = [g.name for g in self.grasps if g not in attachable]
        if not attachable:
            raise Reject("no_attachable_grasp", f"{self.grasp_key}: every library grasp self-collides once the object is attached")
        self.grasps = attachable
        # oracle-fix 2026-11-04 round 3 (touching#03): opt-in generation.approach_finger_support_touch here too (IORSLarge.setup_finger_touch's rule):
        # the approach checked with the fingers at the oracle's pre-shape instead of fully open (the board-side finger filled the ~6 cm to the side board)
        p_touch = IK.finger_touch_params(self.gen("approach_finger_support_touch"))
        if p_touch:
            ws = [TT.grasp_pinch_width(g) for g in self.grasps]
            op = None if not ws or None in ws else min(C.FINGER_OPEN, max(0.5 * w + IK.ORACLE_PRESHAPE_MARGIN_M for w in ws))
            self._shelf_finger_open_template = op
            self.ikc.set_finger_touch(IK.FingerTouch(p_touch["margin_m"], p_touch["max_penetration_m"], p_touch["preshape"], supports=self.support_objects,
                                                     opening=lambda: self.__dict__.get("_shelf_finger_open", op)))  # i7-t03both: the drawn rest's
        # pinch centring (oracle-B, run 0916-202403: a pinch off the object's centre along the closing axis lets one pad reach
        # the object first and drag it until the other lands - +15.4 mm at the closure): the pinch centre is moved onto
        # the hull's centre along c when it sits farther than ``pinch_offset_max_m`` from it (the pads close until contact,
        # so the shift along c changes nothing else); the original offset is recorded per grasp
        self.grasps, offsets = self.recentre_pinches(self.grasps, tgt)
        self.template_facts["grasp_pinch_offset_m"] = {k: round(v, 4) for k, v in offsets.items()}
        # the grasps whose pinch WAS moved (owner 2026-09-24 item 3: an off-centre pinch whose re-centring would change the width
        # between the pads - a rim pinch of a thin-walled cup - keeps its offset and is listed with its reason)
        self.template_facts["grasps_recentred"] = [k for k, f in (self.template_facts.get("grasp_pinch_recentre") or {}).items() if f.get("applied")]
        # flush pinch (teacup demo 2026-09-16 14:44: grasp_3's closing axis was inclined 28 deg to the horizontal; the flat
        # pads squeezed the vertical cup wall at two heights, the cup rotated in the hand until the pinch was flush - 16-32
        # deg in the hand, 21 deg at the release, at rest 10 deg on the well slope). The target is placed at its rest
        # orientation, so a grasp's closing axis must be horizontal (or vertical) in the rest frame within
        # ``closing_tilt_max_deg``; inclined pinches are dropped while a flush grasp remains. The pinch line's distance to the
        # centroid (``pivot_arm``, gravity torque about the pinch) is recorded per grasp as well.
        tilts = {g.name: self.closing_tilt_deg(g.pose) for g in self.grasps}
        self.template_facts["grasp_closing_tilt_deg"] = {k: round(v, 1) for k, v in tilts.items()}
        self.template_facts["grasp_pivot_arms_m"] = {g.name: round(self.pivot_arm(g.pose), 4) for g in self.grasps}
        self.template_facts["grasp_pinch_vs_long_axis_deg"] = {g.name: round(self.pinch_vs_long_axis_deg(g.pose), 1) for g in self.grasps}
        tilt_max = float(self.gen("closing_tilt_max_deg", CLOSING_TILT_MAX_DEG))
        # two-tier grasp set: the flush pinches are tried first at every focus (``sample_instance``), the inclined ones only
        # when the flush set has no feasible (yaw, grasp) there - a round apple keeps its grasps (the earlier hard filter left
        # F empty), a cup gets a flush pinch whenever one is reachable (the 18:41 teacup drew grasp_3, 28 deg, and only the
        # corrected release kept it upright)
        flush = [g for g in self.grasps if tilts[g.name] <= tilt_max]
        self.grasps_flush = flush if flush else list(self.grasps)
        self.template_facts["grasps_flush_first"] = [g.name for g in self.grasps_flush]
        self.template_facts["grasps_inclined_fallback"] = [g.name for g in self.grasps if g not in self.grasps_flush]
        if not flush:
            log.info("%s: no library grasp pinches flush (closing tilts %s deg > %.0f); single tier", self.grasp_key,
                     {k: round(v, 1) for k, v in tilts.items()}, tilt_max)
        self.rest["partner"] = self._partner_rest()
        # owner 2026-10-11 item 3 / 11:46 mirrored layout (touching#03): a book picked with its upright-shelf key starts leaning
        # against the side board its rest orientation leans to; the partner book leans against the opposite board
        self.shelf_start = TT.upright_shelf_facts(self.grasp_key, self.rest["target"]) if self.relation == "touching" else None
        if self.shelf_start is not None:
            self.rest["partner"] = TT.upright_lean_quat(self.hull["partner"], -self.shelf_start["lean_side"], self.shelf_start["lean_deg"])
        self.pair = self.pair_admissibility()
        self.template_facts["pair"] = self.pair
        self.template_facts["carry_clears"] = self.carry_rule()  # 4A: Reject("spec") before any phase
        self.template_facts["widest_faces"] = self.widest_faces()  # Reject("spec") before any phase
        self.template_facts["brace_open_when_wide"] = self.brace_open_when_wide()  # Reject("spec") before any phase
        self.template_facts["push_speed_m_s"] = self.push_speed()  # Reject("spec") before any phase
        self.template_facts["partner_sides"] = [TT.side_name(sd) for sd in self.template_sides()]  # A1: Reject("partner_side") before any phase
        self.template_facts["partner_on_meta_link"] = self.meta_link()
        self.template_facts["partner_aligned_to_target"] = self.partner_aligned_to_target()
        pair = self.pair_goal_align()  # Reject("spec") before any phase
        self.template_facts["pair_goal_align"] = None if pair is None else {k: v for k, v in pair.items() if k != "front"}
        H_max = max(TT.height_of(self.hull["target"], self.rest["target"]), TT.height_of(self.hull["partner"], self.rest["partner"]))
        self.top = TT.support_surface(self, self.support, height=H_max)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{self.support.name}: too little free top")
        self.flat_top = self.build_flat_top()
        self.target_tremble_screen()  # 7g: before the bands and the stances
        nb_top = self.neighbor_start_top()  # start_region_on_neighbors (rejected here without a neighbour top): Phase 2's foci on its cells
        margins = self.lateral_margins()
        rng_t = np.random.default_rng(self.ctx.base_seed)
        errors: List[str] = []
        self.init_stance = None
        init_trunk = None
        # Phases 1a / 1b / 1c per edge: §4.2 has no preference among the edges and §4.4 checks floor + collision per
        # lateral position, so every edge with a free-floor run in front of it (a third of its 10 cm lateral raster at
        # s = 0.45 on free floor) is tried; an edge too short for both objects is skipped (§4.4 step 2, A1)
        for edge, init in TT.edge_frames(self, self.top, rng_t):
            lats = th.linspace(-edge.length / 2 + 0.1, edge.length / 2 - 0.1, max(2, int(edge.length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if self.kit.trav_free(edge.base_xy(init.stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                errors.append(f"{edge.name}: free floor at {n_free}/{len(lats)} lateral positions")
                continue
            try:
                init_trunk = TT.phase1b_init_trunk(self, init, H_max, top=self.top)
                self.init_stance = TT.phase1c_lateral(self, init_trunk, margins, rng_t, top=self.top)
                self.edge = edge
                break
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e}")
                    continue
                raise
        if self.init_stance is None:
            raise Reject("no_free_edge", "; ".join(errors) or f"{self.support.name}: no edge with a free, collision-free stance")
        self.init_stance.name = "init"
        # owner 2026-10-11 item 3 (touching#03): a book picked with its upright-shelf key starts standing on its bottom edge, spine
        # toward the robot, leaning against the side board of the compartment its rest orientation leans to (no drawn position)
        self.shelf_pose = self.shelf_partner_pose = None
        if self.shelf_start is not None:
            side = self.shelf_start["lean_side"]
            self.shelf_pose, facts = TT.shelf_lean_pose(self, self.hull["target"], self.rest["target"], self.top, self.edge, self.support, side)
            self.shelf_partner_pose, pfacts = TT.shelf_lean_pose(self, self.hull["partner"], self.rest["partner"], self.top, self.edge, self.support, -side)
            # room between the two books for the place and the push: the free span along the edge must exceed
            # generation.shelf_free_span_min_m (default 0.15: the carried book's lean extent + the open jaw on both sides)
            t_lat = (self.world.hull_at(self.hull["target"], self.shelf_pose)[:, :2] - self.edge.mid) @ self.edge.tangent
            p_lat = (self.world.hull_at(self.hull["partner"], self.shelf_partner_pose)[:, :2] - self.edge.mid) @ self.edge.tangent
            free = float(t_lat.min() - p_lat.max()) if side > 0 else float(p_lat.min() - t_lat.max())
            merge_shelf_pose_facts(self.shelf_start, facts)
            self.shelf_start.update({"partner": pfacts, "free_span_m": round(free, 4)})
            self.template_facts["shelf_start"] = self.shelf_start
            if not shelf_span_rule_on() and free < float(self.gen("shelf_free_span_min_m", 0.15)):
                raise Reject("shelf_width", f"{self.support.name}: {free:.2f} m between the two leaning books (< {float(self.gen('shelf_free_span_min_m', 0.15)):.2f})")
            if self.shelf_start_draw() is not None:
                # oracle-fix 2026-11-06 (generation.shelf_start_draw): the template's start (the key's lean and side, its compartment) is what
                # every attempt draws its own from (:meth:`draw_shelf_side`, :meth:`draw_shelf_layout`); the template check above stays
                self._shelf_template = {k: self.shelf_start.get(k) for k in ("key", "lean_side", "lean_deg", "context")}
                self._shelf_rest0 = {"target": self.rest["target"].clone(), "partner": self.rest["partner"].clone()}
                self._board_keys = self.shelf_board_pool() if shelf_board_draw_on(self) else None  # i7-t03both (owner A5)
                self.template_facts["shelf_start_draw"] = self.shelf_draw_facts()
            if shelf_span_rule_on():  # 5k.1: the computed requirement at the most favourable drawable leans
                self.shelf_template_span_check()
        # the arms offered for the grasp -> carry -> place -> release segment (owner 2026-09-27, first-demo v2.2): the other arm when
        # it has no role besides the segment and the target is free; its band is its own or the default arm's mirrored
        phase2_band: Dict[str, Any] = {}
        if self.shelf_start is not None:
            # owner 2026-10-11 11:46 (touching#03): the arm on the target's side grasps, places and pushes (the left one for a partner on
            # the right board), the other arm has no role; Phase 2's foci come from that arm's band (its own, else the mirror)
            arm = self.shelf_push_arm()
            self.shelf_use_arm(arm)
            phase2_band = {"bands": self.band_of_arm(arm)}
        else:
            self.segment_arms, self.arm_offer_why = TT.arm_offer(self, other_arm_roles=self.other_arm_roles())
            self.arm_bands = TT.arm_bands_for(self, self.segment_arms, bands=self.bands) if len(self.segment_arms) > 1 else {}
        # Phase 2: manipulation stance survivors of the best angle class (each with its focus), cut to the A1 start region (owner 2026-10-12):
        # the foci that admit the partner on a side the template may draw. Without one the robot's lateral is redrawn (Phase 1c) a few
        # times before the template is rejected. The shelf layout has no region: the boards fix both books.
        failed: List[str] = []  # the lateral draws without a start cell
        phase2_any = False
        for i in range(max(1, int(self.gen("start_region_lateral_draws", START_REGION_LATERAL_DRAWS)))):
            if i:
                try:
                    self.init_stance = TT.phase1c_lateral(self, init_trunk, margins, rng_t, top=self.top)
                except Reject as e:
                    failed.append(f"redraw: [{e.reason}] {e}")
                    break
                self.init_stance.name = "init"
            cands = TT.phase2_manipulation_stance(self, self.init_stance, TT.half_extent(self.hull["target"], self.rest["target"]),
                                                  target_height=TT.height_of(self.hull["target"], self.rest["target"]), top=self.top if nb_top is None else nb_top,
                                                  arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self), keep_pairs=self._pdc_keep_pairs(),
                                                  **phase2_band)
            phase2_any = phase2_any or bool(cands)
            self.manip_candidates = cands if self.shelf_start is not None else self.start_region(cands)
            if self.manip_candidates:
                break
            failed.append(f"lateral {self.init_stance.lateral:+.2f}: " + ("Phase 2 left no survivor" if not cands else "no start cell for the partner on the "
                                                                          + " / ".join(TT.side_name(sd) for sd in self.template_sides())))
        if "start_region" in self.template_facts:
            self.template_facts["start_region"]["failed_lateral_draws"] = failed
        if not self.manip_candidates and start_per_attempt(self):
            self.spa_template_start(failed)
        if not self.manip_candidates:
            if not phase2_any:
                raise Reject("no_manipulation_stance", "Phase 2 left no survivor: " + "; ".join(failed))
            raise Reject("no_start_region", f"{self.support.name} edge {self.edge.name} (L {self.edge.length:.2f}): " + "; ".join(failed))
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "bands": getattr(getattr(self, "bands", None), "path", None), "H_max": H_max, "segment_arms": list(self.segment_arms)})

    def other_arm_roles(self) -> Tuple[str, ...]:
        """Contacts planned besides the grasp -> carry -> place -> release segment: touching pushes the placed target on with the
        grasping hand while the other hand braces the partner (``touching_contacts``); the other relations plan none."""
        return ("push", "brace") if self.relation == "touching" else ()

    def band_of_arm(self, arm: str):
        """The reach band of ``arm`` (the default arm's own; the other arm's own or mirrored, ``tabletop.arm_band``)."""
        return (getattr(self, "arm_bands", None) or {}).get(arm) or self.bands

    def apply_height_extra_to_grasp(self, grasp: GraspRecord, t_pose: G.Pose, manip: Stance) -> None:
        """A/B knob ``generation.contact_height_extra_m`` on the grasp (owner 2026-09-16 23:00): the pinch moves up the object
        by the effective lift — the fingertip only until it is >= x above the support, the pad centre never above the hull
        top minus 5 mm (``core.contact_height_extra(kind="grasp")``) — by shifting ``eef_in_object`` along world up in the
        object frame, so every derived pose (grasp / pre-grasp at the settled pose, place = goal ∘ T_obj→eef) carries it;
        the lifted pre-grasp / grasp must still pass the exact IK, else the library height is kept. Facts on the record."""
        x = requested_height_extra(self)
        if x <= 0.0 or grasp.eef_in_object is None or grasp.grasp is None:
            grasp.extras.update({"height_extra_m": x, "height_extra_effective_m": 0.0, "height_extra_clip_reason": "off" if x <= 0 else "no grasp pose"})
            return
        tip_z = float(C.fingertip((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
        pad_z = float(C.pad_center((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
        hull_w = self.world.hull_at(self.hull["target"], t_pose)
        eff, why = contact_height_extra(self, "grasp", fingertip_z=tip_z, support_z=float(self.top.z), band_top_z=float(hull_w[:, 2].max()), pad_center_z=pad_z)
        if eff > 0.0:
            R_obj = C.quat_to_mat(G.t(t_pose[1]))
            up_obj = R_obj.T @ th.tensor([0.0, 0.0, 1.0])  # world up in the object frame
            eef_in_object = (G.t(grasp.eef_in_object[0]) + eff * up_obj, G.t(grasp.eef_in_object[1]))
            eef = IK.grasp_pose(t_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            res = self.ikc.exact([pre, eef], manip, grasp.arm, ignore_for_last=self.support_objects(), scale_key=self.choice.models["target"].scale, cache=False,
                                 approached=[self.obj("target")])
            if res.ok:
                grasp.eef_in_object, grasp.grasp, grasp.pre_grasp = eef_in_object, eef, pre
            else:
                eff, why = 0.0, f"IK failed with the lift ({res.reason}); library height kept"
        grasp.extras.update({"height_extra_m": x, "height_extra_effective_m": round(eff, 4), "height_extra_clip_reason": why})

    def phase3_two_tier(self, manip: Stance, rng: np.random.Generator, arms: Optional[Sequence[str]] = None):
        """§4.6 step 1 with the flush pinches first: ``tabletop.phase3_target`` over ``grasps_flush``; when that finds no
        feasible (yaw, grasp) at the focus and inclined grasps exist, the same draw over every attachable grasp. ``arms``: the arms
        offered for the segment at this focus (the chosen one is the record's ``arm``). :meth:`pair_goal_align`: F over
        :meth:`pair_target_yaws` with the footprint ``support_margin`` (default 0.02, the start's usual margin) inside the top, else
        :meth:`widest_faces`: over :meth:`widest_faces_target_yaws`, else with :meth:`init_align_support` over the four
        :meth:`support_aligned_yaws`; drawn without jitter."""
        tiers = [self.grasps_flush]
        if len(self.grasps_flush) < len(self.grasps):
            tiers.append(self.grasps)
        kinds: List[Optional[str]] = [None] * len(tiers)
        if self.relation == "nextto" and nextto_top_first_on():
            # i7-nextto (ISPATIALGYM_NEXTTO_TOP_FIRST): each tier's top-down grasps first, its other grasps after them
            split, kinds = [], []
            for gs in tiers:
                top = [g for g in gs if top_down_grasp(getattr(g, "pose", None), (getattr(self, "rest", None) or {}).get("target"))]
                rest = [g for g in gs if not any(g is t for t in top)]
                if not top:
                    TT._count(self, "top_first.no_top")
                if not getattr(self, "_top_first_logged", False):
                    log.info("%s: NEXTTO_TOP_FIRST top %s, then %s", self.gp.variant, [getattr(g, "name", g) for g in top], [getattr(g, "name", g) for g in rest])
                    self._top_first_logged = True
                for kind, sub in (("top", top), ("rest", rest)):
                    if sub:
                        split.append(sub)
                        kinds.append(kind)
            tiers = split or tiers
            kinds = kinds if split else [None] * len(tiers)
        nb_top = self.neighbor_start_top()  # start_region_on_neighbors: the target is drawn on the neighbours' cells
        if getattr(self, "_campaign_pair_active", None) is not None:
            yaws = None  # The fixed goal constrains its orientation; fresh starts keep their full yaw range.
        elif self.pair_goal_align() is not None:
            yaws = self.pair_target_yaws(manip)
        else:
            yaws = self.widest_faces_target_yaws(manip) if self.widest_faces() else self.support_aligned_yaws("target", manip) if self.init_align_support() else None
        yaw_kw = {"yaws": yaws, "jitter": 0.0} if yaws is not None else {}
        if self.pair_goal_align() is not None and getattr(self, "_campaign_pair_active", None) is None:
            yaw_kw["margin"] = self.margin_support  # the long axis across the edge may need the support's whole depth (the partner keeps it too)
        last = None
        for i, gs in enumerate(tiers):
            try:
                t_pose, grasp = TT.phase3_target(self, manip, rng, gs, self.hull["target"], self.rest["target"], self.support_objects(),
                                                 top=self.top if nb_top is None else nb_top, scale_key=self.choice.models["target"].scale, arms=arms,
                                                 arm_bands=getattr(self, "arm_bands", None) or None, **yaw_kw)
                if i > 0 and kinds[i] is None:
                    log.info("%s: no flush pinch feasible at the focus; inclined grasp %s drawn", self.gp.variant, grasp.grasp_name)
                if kinds[i] is not None:
                    TT._count(self, f"top_first.{kinds[i]}")
                return t_pose, grasp
            except Reject as e:
                last = e
                if e.reason != "grasp_unreachable":
                    raise
        raise last

    def pinch_offset(self, eef_in_object: G.Pose) -> float:
        """Signed offset (m) of the pinch centre (pad centre) from the target hull's centre along the closing axis c, in the
        object frame: the pads close symmetrically about it, so an offset lets the nearer pad touch first and drag the object."""
        pos, quat = G.t(eef_in_object[0]), G.t(eef_in_object[1])
        R = C.quat_to_mat(quat)
        a, c = R[:, 2], R[:, 1]
        pc = pos + C.CONTACT_CENTER_Z * a
        proj = self.hull["target"] @ c
        mid = 0.5 * float(proj.min() + proj.max())
        return float(th.dot(pc, c)) - mid

    def pinch_chord(self, eef_in_object: G.Pose) -> Optional[float]:
        """The target's width across the closing axis between the pads at a grasp pose: the hull points inside the pads' band
        along the approach axis (``PAD_Z_RANGE[0]`` .. the fingertips) and within the pads' lateral reach, measured along the
        closing axis in the eef frame (the attachment module's ``pinch_width_m``). None when fewer than two hull points lie
        in the band (the pads close on nothing there)."""
        eef = (G.t(eef_in_object[0]), G.t(eef_in_object[1]))
        loc = C.transform_points(C.invert(eef), G.t(self.hull["target"]))
        in_z = (loc[:, 2] >= C.PAD_Z_RANGE[0]) & (loc[:, 2] <= C.FINGERTIP_LENGTH + 0.002)
        band = in_z & (loc[:, 0].abs() <= 0.03)
        if int(band.sum()) < 2:
            # a sparse collision hull (a prism marker: vertices at its two end rings only) has no boundary point within the pads'
            # lateral reach - its cross-section is the same all along, so every point in the pads' height band counts
            band = in_z
        if int(band.sum()) < 2:
            return None
        return float(loc[band, 1].max() - loc[band, 1].min())

    def recentre_pinches(self, grasps: Sequence[Any], tgt) -> Tuple[List[Any], Dict[str, float]]:
        """Library grasps with the pinch centre moved onto the hull centre along the closing axis when the offset exceeds
        ``pinch_offset_max_m`` (a copy of the library record with the shifted position; the attachability screen is re-run on
        it, a grasp that stops being attachable keeps its original pose). Returns ``(grasps, {name: original offset})``.

        Owner 2026-09-24 item 3: the shift is applied only when the pinch geometry survives it — the hull chord across the
        closing axis between the pads at the re-centred pose (:meth:`pinch_chord`) must match the library pinch width (else the
        chord at the original pose) within ``pinch_recentre_width_tol_m`` / ``pinch_recentre_width_tol_frac``. A solid body
        (the nextto marker, 453b3065) spans the same chord anywhere along c and is re-centred; a thin-walled cup's rim pinch
        moved onto the cup's centre puts the pads over the interior, the 13.7 mm pre-shape lands on the rim and the assisted
        grasp never attaches (teacup 0925-075114 / 114310, ``grasp_4`` 45.4 mm -> 0): such a grasp keeps its offset. The
        decision per grasp is recorded in ``template_facts["grasp_pinch_recentre"]``."""
        import copy

        limit = float(self.gen("pinch_offset_max_m", PINCH_OFFSET_MAX))
        tol_abs = float(self.gen("pinch_recentre_width_tol_m", 0.005))
        tol_frac = float(self.gen("pinch_recentre_width_tol_frac", 0.25))
        out: List[Any] = []
        offsets: Dict[str, float] = {}
        facts: Dict[str, Dict[str, Any]] = {}
        v2_keep = os.environ.get("ISPATIALGYM_V2_RECENTRE", "") != "1"
        for g in grasps:
            off = self.pinch_offset(g.pose)
            offsets[g.name] = off
            if abs(off) <= limit:
                out.append(g)
                continue
            if v2_keep and (getattr(g, "meta", None) or {}).get("library") == "v2":
                # of-v2maps 2026-11-07 (prompt I item 2; the isg1020 spec P1 §3: a grasp is used as physics-validated): a v2 grasp's
                # T_object_eef passed the closure / lift / shake validation at THIS pinch centre - moving it makes an unvalidated grasp
                # (abv2_base: 5 v2 grasps moved by 6.8-14.6 mm, steak / salt shaker / plate). ISPATIALGYM_V2_RECENTRE=1: as before
                facts[g.name] = {"offset_m": round(off, 4), "applied": False, "reason": "v2 grasp: kept at its validated pose"}
                out.append(g)
                continue
            c = C.quat_to_mat(G.t(g.pose[1]))[:, 1]
            shifted = copy.copy(g)
            shifted.position = G.t(g.pose[0]) - off * c
            w0, w1 = self.pinch_chord(g.pose), self.pinch_chord(shifted.pose)
            lib = None
            try:
                lib = (getattr(g, "meta", None) or {}).get("width")
                lib = None if lib is None else float(lib)
            except (TypeError, ValueError):
                lib = None
            ref = lib if lib is not None and lib > 0 else w0
            fact: Dict[str, Any] = {"offset_m": round(off, 4), "chord_original_m": None if w0 is None else round(w0, 4), "chord_recentred_m": None if w1 is None else round(w1, 4),
                                    "library_width_m": None if lib is None else round(lib, 4)}
            if w0 is None and w1 is None:
                comparable = False  # no boundary point between the pads at either pose: nothing says the geometry changes (453b3065 as before)
                fact["note"] = "no hull points in the pads' band at either pose: re-centred as before"
            else:
                comparable = True
            if comparable and (ref is None or w1 is None or abs(w1 - ref) > max(tol_abs, tol_frac * ref)):
                fact.update({"applied": False, "reason": "the re-centred pads would close on a different width" if w1 is not None and ref is not None else "no hull between the re-centred pads"})
                facts[g.name] = fact
                log.info("%s: grasp %s pinch %+.1f mm off centre; re-centring would change the pinch width from %s to %s mm (library %s) - kept as is",
                         self.grasp_key, g.name, off * 1000, None if ref is None else round(ref * 1000, 1), None if w1 is None else round(w1 * 1000, 1), None if lib is None else round(lib * 1000, 1))
                out.append(g)
                offsets[g.name] = off
                continue
            if self.ikc.attachable(tgt, shifted.pose, "right"):
                log.info("%s: grasp %s pinch centre %+.1f mm off the object's centre along the closing axis - re-centred (pinch width %.1f -> %.1f mm)", self.grasp_key, g.name, off * 1000,
                         (w0 or 0.0) * 1000, w1 * 1000)
                fact.update({"applied": True})
                out.append(shifted)
            else:
                log.info("%s: grasp %s pinch %+.1f mm off centre; the re-centred pose is not attachable - kept as is", self.grasp_key, g.name, off * 1000)
                fact.update({"applied": False, "reason": "the re-centred pose is not attachable"})
                out.append(g)
            facts[g.name] = fact
        self.template_facts["grasp_pinch_recentre"] = facts
        return out, offsets

    def pinch_vs_long_axis_deg(self, eef_in_object: G.Pose) -> float:
        """Angle (deg, 0-90) between a library grasp's pinch line (closing axis c) and the target's long axis — the largest
        AABB extent in the rest frame (a lying cylinder's axis). 90 = the pads close across the cylinder, 0 = along it."""
        c_obj = C.quat_to_mat(G.t(eef_in_object[1]))[:, 1]
        c_rest = G.rotate(G.t(self.rest["target"]), c_obj)
        ext = TT.hull_extents(self.hull["target"], self.rest["target"])
        axis = th.zeros(3)
        axis[int(th.argmax(ext))] = 1.0
        d = min(1.0, abs(float(th.dot(c_rest / th.linalg.norm(c_rest).clamp(min=1e-9), axis))))
        return math.degrees(math.acos(d))

    def grasp_facts(self, grasp: GraspRecord) -> Dict[str, Any]:
        """Facts of the recorded grasp for the closure investigation (agent 52b4fb5f): ``pinch_vs_long_axis_deg``,
        ``pad_height_above_support_m`` (pad centre z at the grasp pose - support top z), ``pad_height_above_axis_m`` (the same
        minus the object's radius = half its rest-frame height: a lying cylinder's axis), ``object_radius_m``,
        ``closing_tilt_deg`` and ``pivot_arm_m``. None where a value is not available."""
        out: Dict[str, Any] = {}
        if grasp.eef_in_object is None:
            return out
        out["pinch_vs_long_axis_deg"] = round(self.pinch_vs_long_axis_deg(grasp.eef_in_object), 1)
        out["pinch_offset_m"] = round(self.pinch_offset(grasp.eef_in_object), 4)
        row_name, flipped = C.split_twin(grasp.grasp_name or "")
        orig = (self.template_facts.get("grasp_pinch_offset_m") or {}).get(row_name)
        # the offset is measured along the closing axis, which a twin negates
        out["pinch_offset_original_m"] = round(float(out["pinch_offset_m"] if orig is None else (-orig if flipped else orig)), 4)
        out["closing_tilt_deg"] = round(self.closing_tilt_deg(grasp.eef_in_object), 1)
        out["pivot_arm_m"] = round(self.pivot_arm(grasp.eef_in_object), 4)
        ext = TT.hull_extents(self.hull["target"], self.rest["target"])
        radius = 0.5 * float(ext[2])
        out["object_radius_m"] = round(radius, 4)
        if grasp.grasp is not None and self.top is not None:
            pad_z = float(C.pad_center((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
            out["pad_height_above_support_m"] = round(pad_z - float(self.top.z), 4)
            out["pad_height_above_axis_m"] = round(pad_z - float(self.top.z) - radius, 4)
        return out

    def closing_tilt_deg(self, eef_in_object: G.Pose) -> float:
        """Inclination (deg) of a library grasp's closing axis to the horizontal *or* the vertical, whichever is nearer, in the
        target's rest frame: 0 = a flush pinch of a vertical wall (c horizontal) or of the top / bottom faces (c vertical)."""
        c_obj = C.quat_to_mat(G.t(eef_in_object[1]))[:, 1]
        c_world = G.rotate(G.t(self.rest["target"]), c_obj)
        cz = min(1.0, abs(float(c_world[2])))
        from_horizontal = math.degrees(math.asin(cz))
        return min(from_horizontal, 90.0 - from_horizontal)

    def pivot_arm(self, eef_in_object: G.Pose) -> float:
        """Distance (m) from the target hull's centroid to the pinch line of a library grasp: the pad centre
        (``eef origin + CONTACT_CENTER_Z * a``) along the closing axis c, both in the object frame. Small = the pinch carries
        the object without a gravity torque about the closing axis (the object does not pivot in the hand)."""
        pos, quat = G.t(eef_in_object[0]), G.t(eef_in_object[1])
        R = C.quat_to_mat(quat)
        a, c = R[:, 2], R[:, 1]
        pc = pos + C.CONTACT_CENTER_Z * a
        hull = self.hull["target"]
        centroid = 0.5 * (hull.min(dim=0).values + hull.max(dim=0).values)
        d = centroid - pc
        perp = d - th.dot(d, c) * c
        return float(th.linalg.norm(perp))

    def _partner_rest(self) -> th.Tensor:
        choice = self.choice.models["partner"]
        key = choice.grasp_key or self.gp.roles["partner"].grasp_key_of(choice.category, choice.model)
        if key:
            try:
                return C.rest_quat(key)
            except Exception:
                pass
        return C.IDENTITY_QUAT.clone()

    def pair_admissibility(self) -> Dict[str, Any]:
        """§6.11 per (target, partner): ontop / under -> the partner's top admits the target footprint with >= 1 cm
        (tier any_yaw / aligned / centred, else ``partner_too_small``); nextto -> the centre heights overlap
        (``nextto_height_mismatch``); touching -> partner stability / target pushability; inside -> opening >= footprint
        + 1 cm at the rest orientation - standing on :meth:`up_axis` when the variant names one (the drop test runs in the goal check).
        Ontop with ``generation.ontop_centre: "container"`` (oracle-fix 2026-11-06): the target's footprint (its farthest point from its
        centre, any yaw) + the margin within the partner's floor disc (:meth:`container_floor`), tier ``floor_disc``; the box tiers when the
        partner shows no floor disc."""
        up = self.up_axis() if self.gen("target_up_axis") is not None else None
        # i7-lean 2026-11-07: a v2 LEAN rest is measured at the pose the legacy upright-shelf path measured (the base key's own rest:
        # tabletop.lean_admissibility_rest) - the lean rest stays the start / grasps / map rest
        adm = TT.lean_admissibility_rest(getattr(self, "grasp_key", None)) if up is None else None
        t_ext = TT.hull_extents(self.hull["target"], self.standing_quats()[0] if up is not None else (adm if adm is not None else self.rest["target"]))
        p_ext = TT.hull_extents(self.hull["partner"], self.rest["partner"])
        out: Dict[str, Any] = {"target_extent": t_ext.tolist(), "partner_extent": p_ext.tolist(), "relation": self.relation}
        if adm is not None:
            out["target_extent_at"] = "the upright-shelf key's own rest (the legacy pair check's pose; the v2 lean rest is the start)"
        if up is not None:
            out["target_up_axis"] = up
        tx, ty = sorted([float(t_ext[0]), float(t_ext[1])])
        px, py = sorted([float(p_ext[0]), float(p_ext[1])])
        m = float(self.gen("partner_footprint_margin_m", 0.01))
        container = self.relation == "ontop" and self.gen("ontop_centre", "bbox") not in (None, "bbox") and self.ontop_centre() == "container"
        floor = self.container_floor() if container else None
        if floor is not None and floor.get("radius_m") is not None:
            # oracle-fix 2026-11-06 (ontop_small#02: the steak's x_down rest 25 x 6.3 cm passed the pan's 0.29 x 0.47 m box, handle included,
            # and stood on its edge at the AABB centre, 9 cm from the floor's): the floor disc admits the footprint at any yaw about its centre
            fp = G.local_points_world(self.hull["target"], (th.zeros(3), G.t(self.rest["target"])))[:, :2].float()
            reach = float(th.linalg.norm(fp - min_area_rect(fp)[3], dim=1).max())
            out.update(tier="floor_disc", container_floor=floor, target_reach_m=round(reach, 4))
            self.template_facts["container_floor"] = dict(floor, target_reach_m=round(reach, 4), margin_m=m)
            if reach + m > float(floor["radius_m"]) + 1e-9:
                raise Reject("partner_too_small", f"target footprint reaches {reach:.3f} m from its centre: with {m:.2f} margin it leaves the partner's floor "
                                                  f"disc (radius {float(floor['radius_m']):.3f} m, generation.ontop_centre 'container')")
            return out
        if self.relation in ("ontop", "under"):
            tiers = list(self.gen("fit_tiers", ["any_yaw", "aligned", "centred"]))
            diag = math.hypot(tx, ty)
            tier = None
            if "any_yaw" in tiers and diag + 2 * m <= px:
                tier = "any_yaw"
            elif "aligned" in tiers and tx + 2 * m <= px and ty + 2 * m <= py:
                tier = "aligned"
            elif "centred" in tiers and tx + 2 * m <= px and ty <= py:
                tier = "centred"
            if tier is None:
                raise Reject("partner_too_small", f"target {tx:.2f}x{ty:.2f} does not fit on partner {px:.2f}x{py:.2f} with {m:.2f} margin")
            out["tier"] = tier
        elif self.relation == "nextto":
            t_h, p_h = float(t_ext[2]), float(p_ext[2])
            if not (min(t_h, p_h) > 0):
                raise Reject("nextto_height_mismatch", "degenerate heights")
            out["g_NT"] = float((t_ext + p_ext).mean() / 6)  # OmniGibson NextTo threshold
            handed = self.handedness_pair()
            if handed is not None:
                out["handedness"] = handed
        elif self.relation == "touching":
            if float(p_ext[2]) > float(self.gen("partner_max_aspect", PARTNER_MAX_ASPECT)) * min(px, py) or px < 0.03:
                raise Reject("partner_unstable", f"partner {px:.2f}x{py:.2f}x{float(p_ext[2]):.2f}")
            if tx > float(self.gen("push_max_pinch_m", PUSH_MAX_PINCH)) or ty > float(self.gen("push_max_long_m", PUSH_MAX_LONG)) or float(t_ext[2]) > float(self.gen("push_max_height_m", PUSH_MAX_HEIGHT)):
                raise Reject("target_not_pushable", f"target {tx:.2f}x{ty:.2f}x{float(t_ext[2]):.2f}")
            out["g_NT"] = float((t_ext + p_ext).mean() / 6)
        elif self.relation == "inside":
            om = float(self.gen("opening_margin_m", 0.01))
            if tx + om > px or ty + om > py:
                raise Reject("does_not_fit_inside", f"target {tx:.2f}x{ty:.2f} vs opening {px:.2f}x{py:.2f}")
        return out

    def target_tremble_screen(self) -> None:
        """7g (owner 2026-10-12): touching's target stability gate - :meth:`pair_admissibility` checks sizes only and the spec's flat base
        was not implemented (touching#04: 96 ``trembling`` + 29 ``unstable_init`` rejections, every one the target spray_bottle/asztxi, each
        ~50 s into its attempt). The target alone, at its rest orientation, is set 3 mm above free spots of the top (:func:`tremble_screen_spots`,
        ``pose_centred_on``) and judged as ``settle_and_validate`` judges an instance: ``SimWorld.settle`` (``settle_steps`` + 10 physics
        steps), then ``SimWorld.tremble_after`` with the ``tremble_*`` limits. The first still spot passes; trembling on all of
        ``generation.tremble_screen_spots`` (touching: :data:`TREMBLE_SCREEN_SPOTS`, the other relations 0 = off) rejects the template
        (``target_unstable``). Not run for the shelf layout (the book leans against a board) nor for a template continued with instances
        written. The target is parked again; the facts go to ``template_facts["tremble_screen"]``."""
        n = int(self.gen("tremble_screen_spots", TREMBLE_SCREEN_SPOTS if self.relation == "touching" else 0))
        facts: Dict[str, Any] = {"spots_max": n, "spots": []}
        self.template_facts["tremble_screen"] = facts
        tgt, W = self.obj("target"), self.world
        if n <= 0:
            facts["skipped"] = "off"
            return
        if self.shelf_start is not None:
            facts["skipped"] = "shelf layout: the target leans against a board"
            return
        if int(self.template_meta.get("n_written_before", 0) or 0) > 0 or tgt.name not in W.spawned:
            facts["skipped"] = "template continued with instances written" if tgt.name in W.spawned else "the target is not a spawned object"
            return
        fp = W.footprint(self.hull["target"], (th.zeros(3), G.t(self.rest["target"])))
        spots = tremble_screen_spots(self.top, fp - 0.5 * (fp.max(dim=0).values + fp.min(dim=0).values), n, margin=self.margin_support)
        if not spots:
            facts["skipped"] = "no free spot of the top takes the target's footprint"
            return
        limits = {"frames": int(self.gen("tremble_frames", W.TREMBLE_FRAMES)), "pos_tol": float(self.gen("tremble_pos_tol_m", W.TREMBLE_POS_TOL)),
                  "ang_tol": float(self.gen("tremble_ang_tol_rad", W.TREMBLE_ANG_TOL)), "drift_pos_tol": float(self.gen("tremble_drift_pos_tol_m", W.TREMBLE_DRIFT_POS_TOL)),
                  "drift_ang_tol": float(self.gen("tremble_drift_ang_tol_rad", W.TREMBLE_DRIFT_ANG_TOL))}
        err = None
        try:
            for xy in spots:
                tgt.set_position_orientation(*TT.pose_centred_on(self, self.hull["target"], self.rest["target"], xy, top=self.top))
                W.activate(tgt)
                vel = W.settle([tgt], steps=int(self.gen("settle_steps", self.DEFAULT_SETTLE_STEPS)))
                err = W.tremble_after([tgt], **limits)
                facts["spots"].append({"xy": [round(float(v), 3) for v in xy], "still": err is None, "residual_velocity": vel,
                                       "tremble": dict((getattr(W, "last_tremble", None) or {}).get(tgt.name) or {})})
                if err is None:
                    return
        finally:
            W.park(tgt, list(W.spawned).index(tgt.name))
            self.kit.sync_world()
        raise Reject("target_unstable", f"{tgt.name} at its rest orientation trembles alone on each of {len(spots)} free spots of {self.top.name}: {err}")

    def handedness_pair(self) -> Optional[Dict[str, str]]:
        """Owner 2026-10-11 (nextto shoe pair): ``generation.handedness`` (``{"<category>/<model>": "left" | "right"}``, written by
        ``ground_isg``) names the hand of a mirror-pair model; when the drawn target and partner both have one the target must end
        on its own side of the partner, robot frame (the right shoe right of the left shoe): ``{"target", "partner",
        "target_side"}``. None when either hand is unknown; ``Reject("handedness_pair")`` for two objects of one hand."""
        from ispatialgym.sampling import pair_rules as PR

        hands = self.gen("handedness") or {}
        if not isinstance(hands, dict):
            return None
        t, p = self.choice.models["target"], self.choice.models["partner"]
        ht, hp = hands.get(f"{t.category}/{t.model}"), hands.get(f"{p.category}/{p.model}")
        try:
            side = PR.handed_target_side(ht, hp)
        except ValueError:
            raise Reject("handedness_pair", f"target {t.category}/{t.model} and partner {p.category}/{p.model} are both {ht}")
        return None if side is None else {"target": ht, "partner": hp, "target_side": side}

    # ------------------------------------------------------------------ A1 start region (owner 2026-10-12)

    def assigned_side(self) -> Optional[float]:
        """A1 step 3, eval: ``generation.partner_side`` (``"left"`` / ``"right"``: the partner's side of the target, robot frame;
        ``multiply_isg`` drops it for train) as +1 / -1, None without it; ``Reject("partner_side")`` for a side the pair does not admit."""
        name = self.gen("partner_side")
        if name is None:
            return None
        if name not in ("left", "right"):
            raise Reject("spec", f"{self.gp.variant}: generation.partner_side must be 'left' or 'right', got {name!r}")
        side = 1.0 if name == "left" else -1.0
        allowed = self.admitted_sides()
        if side not in allowed:
            who = "the shelf layout (the partner leans against the board opposite the target's lean)" if self.shelf_lean() is not None else "the pair"
            raise Reject("partner_side", f"{self.gp.variant}: generation.partner_side {name} - {who} admits the partner on the "
                                         f"{' / '.join(TT.side_name(sd) for sd in allowed)} only")
        return side

    def shelf_lean(self) -> Optional[float]:
        """The target's lean side in the shelf layout (+1 = the robot's left), None on the tabletop."""
        start = getattr(self, "shelf_start", None)
        return None if start is None else float(start["lean_side"])

    def admitted_sides(self) -> List[float]:
        """:func:`partner_sides` of this template: the relation, the pair's handedness and the shelf layout's lean."""
        return partner_sides(self.relation, (self.pair or {}).get("handedness"), self.shelf_lean())

    def template_sides(self) -> List[float]:
        """A1 step 3: the partner sides the template may draw - the eval side alone (:meth:`assigned_side`), else every side the pair admits."""
        assigned = self.assigned_side()
        return [assigned] if assigned is not None else self.admitted_sides()

    def reach_rule(self) -> Optional[str]:
        """A1 step 2 (c), the band cut the partner placement always made (``tabletop.lateral_gap_masks``): ontop / under / inside the partner's
        centre (the goal is on / in it), nextto - and touching without its push stance - the target's place position, touching with the push
        stance nothing (the place and push screens of the draw judge it from the second base pose)."""
        if self.relation in ("ontop", "under", "inside"):
            return "partner_centre"
        if self.relation == "touching" and bool(self.gen("push_stance", True)) and bool(self.gen("reach_screen_place", True)):
            return None
        return "target_goal"

    def band_near(self, st: Stance, depth: float, arms: Sequence[str]) -> Optional[np.ndarray]:
        """Base-frame band cells (N, 2) within ``BAND_DEPTH_WINDOW_M`` of base-frame ``depth`` at ``st``'s trunk, over the bands of ``arms``;
        None when none of them has a band."""
        bands = list({id(b): b for b in (self.band_of_arm(a) for a in arms) if b is not None}.values())
        if not bands:
            return None
        cells = np.concatenate([np.asarray(b.band(st.translate, st.tilt, **TT.band_kw(b, st)), dtype=float).reshape(-1, 2) for b in bands])
        return cells[np.abs(cells[:, 0] - float(depth)) <= BAND_DEPTH_WINDOW_M]

    def band_rule(self, st: Stance, depth: float, arms: Sequence[str]) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
        """``(reach, band)`` for ``tabletop.lateral_gap_masks`` at ``st``: :meth:`reach_rule` and the laterals (λ_min, λ_max) of :meth:`band_near`
        (None without a cell there: the rule passes nothing); no rule without a band."""
        near = self.band_near(st, depth, arms)
        if near is None:
            return None, None
        return self.reach_rule(), ((float(near[:, 1].min()), float(near[:, 1].max())) if len(near) else None)

    def edge_partner_quats(self) -> List[th.Tensor]:
        """The partner's rest orientation turned so its footprint lies along the edge, and a quarter turn from there (only the first when
        both footprints span the same along the edge): the orientations the start region tries. :meth:`pair_goal_align`: the partner
        orientations the key admits beside a target at any of its :meth:`pair_target_yaws` (the first of equal spans)."""
        if self.pair_goal_align() is not None:
            rest = G.t(self.rest["target"])
            quats = [G.qmul(G.quat_from_yaw(self.pair_partner_yaw((th.zeros(3), G.qmul(G.quat_from_yaw(y), rest)))), G.t(self.rest["partner"]))
                     for y in self.pair_target_yaws(self.init_stance)]
            out, spans = [], []
            for q in quats:
                a, b = footprint_laterals(self.world.footprint(self.hull["partner"], (th.zeros(3), q)), self.init_stance.tangent)
                if all(abs((b - a) - s) >= 0.01 for s in spans):
                    out.append(q)
                    spans.append(b - a)
            return out
        q0 = self.edge_aligned_quat(self.hull["partner"], self.rest["partner"], self.init_stance)
        q1 = G.qmul(G.quat_from_yaw(math.pi / 2), q0)
        (a0, b0), (a1, b1) = (footprint_laterals(self.world.footprint(self.hull["partner"], (th.zeros(3), q)), self.init_stance.tangent) for q in (q0, q1))
        return [q0] if abs((b0 - a0) - (b1 - a1)) < 0.01 else [q0, q1]

    def meta_link_points(self, link: str) -> List[Tuple[str, th.Tensor]]:
        """A5: the support's meta links of type ``link`` as ``(id, world xyz)`` - its loaded meta-link prims, else ``meta_links.base_link.<link>``
        of its model's ``misc/metadata.json`` (object frame) at its pose and scale (``tabletop.meta_link_world_points``)."""
        import json
        import os

        from omnigibson.utils.asset_utils import get_dataset_path

        obj = self.support
        pts = [(str(lk.meta_link_id), G.t(lk.get_position_orientation()[0]).float()) for lk in obj.links.values() if lk.is_meta_link and lk.meta_link_type == link]
        if pts:
            return pts
        with open(os.path.join(get_dataset_path("behavior-1k-assets"), "objects", obj.category, obj.model, "misc", "metadata.json")) as f:
            entries = ((json.load(f).get("meta_links") or {}).get("base_link") or {}).get(link) or {}
        return TT.meta_link_world_points(entries, obj.get_position_orientation(), obj.scale)

    def meta_link(self) -> Optional[str]:
        """A5 (owner 2026-10-12): ``generation.partner_on_meta_link``, the meta-link type the partner is centred on (e.g. ``"heatsource"``),
        None without it; ``Reject("spec")`` for touching (its partner is braced beside the push)."""
        link = self.gen("partner_on_meta_link")
        if link and self.relation == "touching":
            raise Reject("spec", f"{self.gp.variant}: generation.partner_on_meta_link is for ontop / under / inside / nextto (touching braces its partner)")
        return str(link) if link else None

    def partner_aligned_to_target(self) -> bool:
        """under_small#04 (owner 2026-10-13 ~18:40: "align the spatula with the paper sheet"), ``generation.partner_aligned_to_target``
        (default off): the partner is drawn only at the yaws that lay its footprint's long side along the target's long axis at the drawn
        target pose (:meth:`aligned_partner_yaws`) instead of random yaws. The target keeps its init yaw to the goal (§6.11), so it lies
        parallel to the partner's edges at init and at the goal - for ontop / under of tier ``aligned`` / ``centred`` the long axes
        parallel, the only yaws at which the target fits on the partner. ``Reject("spec")`` with touching (edge-aligned, :meth:`edge_aligned`)
        or ``partner_on_meta_link`` (a random yaw per spot)."""
        on = bool(self.gen("partner_aligned_to_target", False))
        if on and (self.relation == "touching" or self.meta_link() is not None):
            raise Reject("spec", f"{self.gp.variant}: generation.partner_aligned_to_target is for the plain partner draw of ontop / under / inside / nextto "
                                 f"(not touching, not partner_on_meta_link)")
        return on

    def aligned_partner_yaws(self, t_pose: G.Pose, rng: np.random.Generator) -> List[float]:
        """The partner yaws an attempt tries with :meth:`partner_aligned_to_target`: :func:`target_aligned_partner_yaws` at the drawn target
        pose - the long side across the target too only when the pair fits at any yaw (no fit tier, or ``any_yaw``) -, shuffled and repeated
        up to ``PARTNER_YAW_DRAWS`` tries (each try draws its own gap)."""
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        p_fp = self.world.footprint(self.hull["partner"], (th.zeros(3), G.t(self.rest["partner"])))
        base = target_aligned_partner_yaws(t_fp, p_fp, across=(self.pair or {}).get("tier") in (None, "any_yaw"))
        return [base[int(i)] for _ in range(max(1, PARTNER_YAW_DRAWS // len(base))) for i in rng.permutation(len(base))]

    @property
    def flat_margin(self) -> float:
        return float(self.gen("partner_flat_margin_m", 0.01))

    def build_flat_top(self) -> Optional[FlatTop]:
        """under_small#04 (owner 2026-10-13 ~18:40: "move the sheet to a flat part of the support"), ``generation.partner_flat_top`` (default
        off): the partner's footprint, grown by ``partner_flat_margin_m`` (0.01), must lie on cells of the support's top that are seen from
        straight above as one flat face (:class:`FlatTop`). Built once per template from the support's visual meshes (the faces reaching
        within 5 cm of the top), rasterised at ``partner_flat_res_m`` (0.002) with ``partner_flat_tol_m`` (0.003) and
        ``partner_flat_normal_min`` (0.99). The visual meshes, not raycasts: commercial_kitchen_table_vxvtec_0's collision top is one flat
        hull (z 0.8705 across its whole top), while its render shows the seam between its two table units, which c3_k1's sheet covered.
        None without the key."""
        if not self.gen("partner_flat_top", False):
            return None
        import time

        t0 = time.time()
        res, tol = float(self.gen("partner_flat_res_m", 0.002)), float(self.gen("partner_flat_tol_m", 0.003))
        normal_min = float(self.gen("partner_flat_normal_min", 0.99))
        z_top = float(self.top.z)
        self.world.sync_render_transforms()
        tris = []
        for link in self.support.links.values():
            for mesh in link.visual_meshes.values():
                tri = mesh.points_in_world_frame.float().numpy()[mesh.faces.long().numpy()]
                tris.append(tri[tri[:, :, 2].max(axis=1) >= z_top - 0.05])
        tri = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        lo, hi = (G.t(v).float() for v in self.support.aabb)
        x0, y0 = float(lo[0]) - res, float(lo[1]) - res
        nx, ny = int(math.ceil((float(hi[0]) - x0) / res)) + 2, int(math.ceil((float(hi[1]) - y0) / res)) + 2
        zbuf, nbuf = raster_from_above(tri, x0, y0, nx, ny, res)
        flat = FlatTop.from_raster(x0, y0, res, zbuf, nbuf, z_top, tol, normal_min)
        facts = dict(flat.facts(), tol_m=tol, normal_min=normal_min, margin_m=self.flat_margin, collision_top_z=round(z_top, 5), n_triangles=int(len(tri)),
                     build_s=round(time.time() - t0, 2))
        self.template_facts["partner_flat_top"] = facts
        log.info("%s: partner_flat_top on %s: %d of %d cells flat at z %.4f (collision top %.4f), %d faces, %.1f s", self.gp.variant, self.support.name, facts["n_flat"],
                 facts["n_cells"], flat.z_ref, z_top, len(tri), facts["build_s"])
        if not facts["n_flat"]:
            raise Reject("support_flat", f"{self.support.name}: no flat cell on its top seen from above ({len(tri)} faces near the top)")
        return flat

    def flat_gap_mask(self, side: float, t_fp: th.Tensor, depth: float, p_quat, manip: Stance, gs: Sequence[float]) -> np.ndarray:
        """``generation.partner_flat_top``: per gap of ``gs``, the partner footprint at ``p_quat`` laid out as ``tabletop.partner_gap_masks``
        lays it out keeps more than ``partner_flat_margin_m`` from every cell of the top that is not flat."""
        t_lat = (G.t(t_fp)[:, :2].float() - manip.base_xy) @ manip.tangent
        fp = self.world.footprint(self.hull["partner"], (th.zeros(3), G.t(p_quat))).float()
        p_lat = fp @ manip.tangent
        xy = TT.partner_origin_xy(side, (float(t_lat.min()), float(t_lat.max())), (float(p_lat.min()), float(p_lat.max())), depth, manip, gs)
        reach = self.flat_margin + 2 * self.flat_top.res
        return np.array([self.flat_top.clearance(fp + xy[k], reach) > self.flat_margin for k in range(len(xy))], dtype=bool)

    def flat_pose_why(self, p_pose: G.Pose) -> Optional[str]:
        """``generation.partner_flat_top`` at a drawn partner pose: why its footprint is too near a part of the top that is not flat, else None."""
        c = self.flat_top.clearance(self.world.footprint(self.hull["partner"], p_pose), self.flat_margin + 2 * self.flat_top.res)
        if c <= self.flat_margin:
            return f"partner footprint {100 * c:.1f} cm from a part of the top that is not flat (needs > {100 * self.flat_margin:.1f} cm)"
        return None

    def neighbor_start_top(self) -> Optional["NB.UnionTop"]:
        """Owner 2026-10-12 (ontop_small#02 6d), ``generation.start_region_on_neighbors``: the partner stands on a meta link of the support
        (:meth:`meta_link`) and the target starts on a neighbouring support's top (A4) - the union top with the support's own cells taken
        out (``UnionTop.neighbors_only``), on which Phase 2 takes its foci, the start region keeps them and Phase 3 draws the target. None
        without the key; ``Reject("spec")`` without ``partner_on_meta_link``, ``Reject("no_start_region")`` when no neighbour top is united
        with the support."""
        if not self.gen("start_region_on_neighbors", False):
            return None
        if self.meta_link() is None:
            raise Reject("spec", f"{self.gp.variant}: generation.start_region_on_neighbors needs generation.partner_on_meta_link (the partner on the support)")
        if not isinstance(self.top, NB.UnionTop):
            raise Reject("no_start_region", f"{self.top.name}: generation.start_region_on_neighbors, but no neighbour top is united with the support")
        return self.top.neighbors_only()

    def find_meta_spots(self, cam: Optional[G.Pose]) -> Optional[List[Tuple[str, th.Tensor]]]:
        """A5: the support's meta links of the :meth:`meta_link` type (:meth:`meta_link_points`) whose position is on the top (a free cell;
        the partner centred there may overhang it - a pot's handle off the stove, the settle and the validation judge it) with the partner
        centred there fully in the init image (in either edge-aligned orientation) - ``(id, world xy)``; None without the key. Every meta
        link with its verdict goes to ``template_facts["meta_link_spots"]``."""
        link = self.meta_link()
        if link is None:
            return None
        spots, facts = [], []
        quats = self.edge_partner_quats()
        for mid, p in self.meta_link_points(link):
            poses = [TT.pose_centred_on(self, self.hull["partner"], q, p[:2], top=self.top) for q in quats]
            on_top = self.top.contains(p[:2])
            in_frame = cam is None or any(TT.in_frame(self, cam, TT.hull_corners_world(self.hull["partner"], pose)) for pose in poses)
            facts.append({"id": mid, "xy": [round(float(p[0]), 4), round(float(p[1]), 4)], "on_top": on_top, "in_frame": in_frame})
            if on_top and in_frame:
                spots.append((mid, p[:2].clone()))
        self.template_facts["meta_link_spots"] = {"link": link, "spots": facts, "usable": [mid for mid, _ in spots]}
        return spots

    def meta_spot_region(self, c: Stance, side: float, w_t: float, gap_range: Tuple[float, float], reach: Optional[str],
                         band: Optional[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], List[int]]:
        """A5 start region of candidate ``c`` on ``side``: the spots of ``meta_spots`` beside the target's focus at a gap of ``gap_range``
        (``tabletop.meta_spot_gaps``, the target's footprint ``table_end_margin_m`` from the edge's ends) whose partner centre meets the reach
        rule (``reach`` / ``band`` of :meth:`band_rule`, as for a drawn gap) - ``((min gap, max gap), spot indices)``, ``(None, [])`` when none."""
        depth, t_c = c.local(c.focus[:2])
        w_p = TT.half_extent(self.hull["partner"], self.rest["partner"])
        mid = float((self.edge.mid - c.base_xy) @ c.tangent)
        ends = (mid - 0.5 * float(self.edge.length), mid + 0.5 * float(self.edge.length))
        ok = []
        for k, g in TT.meta_spot_gaps(side, (depth, t_c), w_t, [c.local(xy) for _, xy in self.meta_spots], w_p, gap_range, ends=ends, m_end=self.table_end_margin):
            if reach is None or TT.lateral_gap_masks([g], side, (t_c - w_t, t_c + w_t), (-w_p, w_p), band=band, reach=reach, t_origin=t_c,
                                                     place_gap=self.place_gap())["reach"][0]:
                ok.append((k, g))
        if not ok:
            return None, []
        return (min(g for _, g in ok), max(g for _, g in ok)), [k for k, _ in ok]

    def start_region(self, cands: Sequence[Stance]) -> List[Stance]:
        """A1 step 2 at the template, after Phase 2: the candidates whose focus is a valid target start for a partner side the template may
        draw (:meth:`template_sides`) - some gap of [g_min, D_p] puts the partner (a) on the top with both footprints at least
        ``table_end_margin_m`` from the edge's ends, (b) fully in the init image, (c) meeting :meth:`reach_rule` at the candidate's trunk with
        the band of an arm that may make the segment there; the target nominally a ``w_t`` square at the focus, the partner in either
        edge-aligned orientation (:meth:`edge_partner_quats`). Each survivor's ``extras["start_region"]`` holds the gap interval per side and
        ``extras["start_region_arms"]`` the arms that meet (c) per side; ``region_sides`` the sides with a cell; the region's size goes to
        ``template_facts["start_region"]``. With ``partner_on_meta_link`` (A5) the partner stands on a meta-link spot (:meth:`find_meta_spots`)
        and the region follows from those spots (:meth:`meta_spot_region`; the candidate's spots per side in ``extras["meta_link_spots"]``);
        with ``start_region_on_neighbors`` too, only the foci whose Phase-2 footprint (``w_t`` + 0.02 around them) lies on a neighbour's top
        (:meth:`neighbor_start_top`)."""
        allowed = self.admitted_sides()
        assigned = self.assigned_side()
        sides = [assigned] if assigned is not None else allowed
        g_lo, g_hi = self.gap_range()
        gs = TT.gap_grid(g_lo, g_hi)
        cam = self.camera_pose_at(self.init_stance)
        self.meta_spots = self.find_meta_spots(cam)
        w_t = TT.half_extent(self.hull["target"], self.rest["target"])
        on_nb = self.neighbor_start_top()
        n_off = 0
        place_gap = self.place_gap()
        quats = [(q, footprint_laterals(self.world.footprint(self.hull["partner"], (th.zeros(3), q)), self.init_stance.tangent)) for q in self.edge_partner_quats()]
        focus_masks: Dict[Tuple, np.ndarray] = {}  # (focus, side, orientation) -> the rules of the focus alone: ends, top, frame
        cells: Dict[float, set] = {sd: set() for sd in sides}
        n_cands: Dict[float, int] = {sd: 0 for sd in sides}
        gaps: Dict[float, List[float]] = {sd: [] for sd in sides}
        out: List[Stance] = []
        for c in cands:
            if on_nb is not None and not on_nb.contains(c.focus[:2], w_t + 0.02):  # Phase 2's focus rule, on the neighbours' cells
                n_off += 1
                continue
            depth, t_c = c.local(c.focus[:2])
            t_fp = th.stack([c.world(depth + dd, t_c + dl) for dd in (-w_t, w_t) for dl in (-w_t, w_t)])
            # the reach rule per arm that may make the segment here: the attempt offers the arms whose band admits the drawn side
            rules = {a: self.band_rule(c, float(c.world_to_base(c.focus[:2])[0]), [a]) for a in (c.extras.get("focus_arms") or self.segment_arms[:1])}
            fkey = (round(float(c.focus[0]), 3), round(float(c.focus[1]), 3))
            region: Dict[str, List[float]] = {}
            arms: Dict[str, List[str]] = {}
            for sd in sides:
                if self.meta_spots is not None:
                    runs, spots = [], set()
                    for arm, (reach, band) in rules.items():
                        run_arm, idx = self.meta_spot_region(c, sd, w_t, (g_lo, g_hi), reach, band)
                        if idx:
                            arms.setdefault(TT.side_name(sd), []).append(arm)
                            runs.append(run_arm)
                            spots |= set(idx)
                    run = (min(r[0] for r in runs), max(r[1] for r in runs)) if runs else None
                    if spots:
                        c.extras.setdefault(META_SPOTS, {})[TT.side_name(sd)] = sorted(spots)
                else:
                    ok = np.zeros(len(gs), dtype=bool)
                    for arm, (reach, band) in rules.items():
                        ok_arm = np.zeros(len(gs), dtype=bool)
                        for qi, (q, p_lat) in enumerate(quats):
                            key = (fkey, sd, qi)
                            if key not in focus_masks:
                                masks = TT.partner_gap_masks(self, sd, t_fp, depth, self.hull["partner"], q, c, gs, top=self.top, cam=cam,
                                                             margin=self.margin_support, m_end=self.table_end_margin)
                                focus_masks[key] = np.logical_and.reduce([np.ones(len(gs), dtype=bool)] + list(masks.values()))
                            m = focus_masks[key]
                            if reach is not None:
                                m = m & TT.lateral_gap_masks(gs, sd, (t_c - w_t, t_c + w_t), p_lat, band=band, reach=reach, t_origin=t_c,
                                                             place_gap=place_gap)["reach"]
                            ok_arm |= m
                        if ok_arm.any():
                            arms.setdefault(TT.side_name(sd), []).append(arm)
                        ok |= ok_arm
                    run = TT.longest_run(gs, ok)
                if run is not None:
                    region[TT.side_name(sd)] = [round(run[0], 4), round(run[1], 4)]
                    cells[sd].add(tuple(c.extras.get("_position") or fkey))
                    n_cands[sd] += 1
                    gaps[sd] += list(run)
            if region:
                c.extras[TT.START_REGION] = region
                c.extras[REGION_ARMS] = arms
                out.append(c)
        self.region_sides = region_sides(allowed, assigned, {sd: len(v) for sd, v in cells.items()})
        margins = self.init_stance.extras.get("margins") or {}
        self.template_facts["start_region"] = {
            "edge": self.edge.name, "edge_length_m": round(float(self.edge.length), 3), "L_min_m": round(float(margins.get("L_min", 0.0)), 3),
            "m_end_m": self.table_end_margin, "lateral_m": round(float(self.init_stance.lateral), 3),
            "assigned": None if assigned is None else TT.side_name(assigned), "sides": [TT.side_name(sd) for sd in sides],
            "drawn_sides": [TT.side_name(sd) for sd in self.region_sides], "gap_range_m": [round(g_lo, 4), round(g_hi, 4)], "phase2_candidates": len(cands),
            "cells": {TT.side_name(sd): len(cells[sd]) for sd in sides}, "candidates": {TT.side_name(sd): n_cands[sd] for sd in sides},
            "gap_m": {TT.side_name(sd): ([round(min(gaps[sd]), 4), round(max(gaps[sd]), 4)] if gaps[sd] else None) for sd in sides},
            "meta_link": None if self.meta_spots is None else [mid for mid, _ in self.meta_spots]}
        log.info("%s: start region on edge %s (L %.2f, lateral %+.2f): %s of %d Phase-2 candidates", self.gp.variant, self.edge.name, self.edge.length,
                 self.init_stance.lateral, ", ".join(f"{TT.side_name(sd)} {len(cells[sd])} cells ({n_cands[sd]} candidates)" for sd in sides), len(cands))
        if on_nb is not None:
            names = [m.name for m in on_nb.members[1:]]
            self.template_facts["start_region"].update({"on_neighbors": names, "off_neighbors": n_off})
            log.info("%s: start_region_on_neighbors: %d of %d Phase-2 candidates have their focus off the neighbour tops %s", self.gp.variant, n_off, len(cands), names)
        return out

    def draw_partner_side(self, rng: np.random.Generator) -> float:
        """A1 step 3, once per attempt before the target is placed: uniform among ``region_sides`` (eval: the assigned side, the only one)."""
        sides = self.region_sides
        if not sides:
            raise Reject("no_start_region", "no partner side has a start cell")
        return sides[int(rng.integers(len(sides)))] if len(sides) > 1 else sides[0]

    def sample(self, rng: np.random.Generator) -> Draft:
        from ispatialgym.sampling.v2.core import start_per_attempt

        shelf_draw = shelf_draw_on(self)
        if shelf_draw:
            # oracle-fix 2026-11-06 (generation.shelf_start_draw): the board the target leans against first - it names the working arm, whose
            # band this attempt's Phase 2 reads
            self.draw_shelf_side(rng)
        if start_per_attempt(self):  # this attempt's start first: the partner sides come from its start region
            self.spa_draw(rng)  # timed (_spa_draw_s); i7-startdraw: the START_DRAW counters
            self._spa_drawn = True
        if shelf_draw:
            self.draw_shelf_layout(rng)  # the compartment and the leans: the two books' poses follow the attempt's edge
        # A1 step 3: the partner's side is fixed before the target is placed - before the split cone draws the start position too
        self.region_side = None if self.shelf_start is not None else self.draw_partner_side(rng)
        if start_per_attempt() and place_depth_cap_redraw_on() and place_depth_cap_mode() == "on" and self.place_depth_cap_applies():
            # follow-up (c): the start (= place) depth drawn within the cap - re-drawn while every candidate of the side lies beyond it
            for i in range(PLACE_DEPTH_CAP_REDRAWS + 1):
                cands = [c for c in self.manip_candidates if TT.in_start_region(c, self.region_side) and TRAVEL_DROPPED not in c.extras]
                facts = self.place_depth_caps(cands) if cands else {}
                if not cands or not facts or any(not f["remove"] for f in facts.values()):
                    if i > 0:
                        TT._count(self, "place_depth_cap.redraw_found")
                    break
                if i == PLACE_DEPTH_CAP_REDRAWS:
                    TT._count(self, "place_depth_cap.redraw_exhausted")
                    break
                TT._count(self, "place_depth_cap.redraws")
                self.spa_redraw_start(rng)
                self.region_side = self.draw_partner_side(rng)
        return super().sample(rng)

    def live_candidates(self, side: Optional[float]) -> List[Stance]:
        """The template's candidates for ``side`` not dropped for their travel (7e): every split-cone position's (``tabletop.split_cone_draw``
        replaces ``manip_candidates`` per attempt), else ``manip_candidates``."""
        cands = getattr(self, "_split_cone_cands", None)
        out = [c for c in (self.manip_candidates if cands is None else cands) if TT.in_start_region(c, side) and TRAVEL_DROPPED not in c.extras]
        # i7-plateqtf 2026-11-07 (ISPATIALGYM_FAST_MAP_TILT_INHERIT): a candidate whose focus stands on an inherited map cell invalidated for
        # its tilt meanwhile (v2/map_tilt.py) leaves the draws - a start cell cached before the invalidation still lists it
        live = [c for c in out if not TT.map_tilt_focus_dead(self, c)]
        if len(live) < len(out):
            TT._countn(self, "map_tilt.candidates_dropped", len(out) - len(live))
        return live

    def drop_candidate(self, cand: Stance, err: Reject) -> None:
        """7e (owner 2026-10-12; touching#02 c5_k0 redrew its only candidate 407 times, each one ``no_transition``): ``cand``'s init ->
        manipulation travel failed deterministically (:func:`travel_failure_deterministic`), so it leaves the draws for the rest of the
        template - marked, out of ``manip_candidates`` and of the split cone (a side's position pool left empty is rebuilt from the live
        candidates at its next draw, a lower init-view class included) - and a partner side without a live candidate leaves
        ``region_sides``. Recorded in ``template_meta.travel_dropped`` and the counter ``travel_dropped``."""
        why = f"[{err.reason}] {err}"
        cand.extras[TRAVEL_DROPPED] = why[:400]
        if start_per_attempt(self) and fast_on("SPA_CACHE"):
            # train-fast 2026-10-31: a reused lateral cell hands out fresh copies of its candidates, so the mark above does not carry; the
            # candidate's identity does (:meth:`_spa_apply_cell` filters it out of every later reuse in this process)
            self.__dict__.setdefault("_spa_dropped", set()).add(self._spa_drop_key(cand))
        self.manip_candidates = [c for c in self.manip_candidates if c is not cand]
        if getattr(self, "_split_cone_cands", None):
            self._split_cone_cands = [c for c in self._split_cone_cands if c is not cand]
        pools = getattr(self, "_split_cone_pools", None) or {}
        for side in list(pools):
            pool = {key: [c for c in cs if c is not cand] for key, cs in pools[side].items()}
            pool = {key: cs for key, cs in pool.items() if cs}
            if pool:
                pools[side] = pool
            else:
                del pools[side]
        self.region_sides = [sd for sd in self.region_sides if self.live_candidates(sd)]
        self.counters["travel_dropped"] = self.counters.get("travel_dropped", 0) + 1
        self.template_meta.setdefault("travel_dropped", []).append({"base_pose2d": [round(float(v), 4) for v in cand.base_pose2d], "trunk": [cand.translate, round(float(cand.tilt), 4)],
                                                                   "focus": None if cand.focus is None else [round(float(v), 4) for v in cand.focus.tolist()], "why": why[:300]})
        log.info("%s: candidate at %s (trunk %.2f / %+.2f) dropped for the template: its init -> manipulation travel fails on every redraw (%s)", self.gp.variant,
                 [round(float(v), 2) for v in cand.base_pose2d], cand.translate, cand.tilt, why[:200])

    def start_region_exhausted(self) -> bool:
        """7e: no candidate is left for any partner side the attempts may draw (the shelf layout: none at all) - every one was dropped for
        its travel; no attempt of the template can succeed any more."""
        if self.shelf_start is not None:
            return not self.live_candidates(None)
        return not self.region_sides

    def lateral_margins(self) -> Dict[str, Any]:
        """The IORS row of §4.4 for ``tabletop.lateral_margins`` (λ, b, R_img are filled by Phase 1b / 1c): w_t, w_p, W_p,
        D_p_max = W_p + 0.03, safety, relation, g_NT, g_min, m_end (A1: the edge holds both objects at the smallest gap)."""
        w_t = TT.half_extent(self.hull["target"], self.rest["target"])
        w_p = TT.half_extent(self.hull["partner"], self.rest["partner"])
        W_p = 2 * w_p
        # a nextto pair's D_p ceiling moves with its gap margin (the interval [g_NT + margin, W_p + margin] keeps its width)
        extra = max(self.partner_gap_extra, self.nextto_gap_extra) if self.relation == "nextto" else self.partner_gap_extra
        return {"kind": "iors", "w_t": w_t, "w_p": w_p, "W_p": W_p, "D_p_max": W_p + extra, "safety": 0.02, "relation": self.relation,
                "g_NT": self.pair.get("g_NT"), "nextto_gap_extra": self.nextto_gap_extra, "g_min": self.gap_min, "m_end": self.table_end_margin}

    # ------------------------------------------------------------------ instance (§4.6 Phase 3 + §6.11)

    def _spa_edge_trunk(self, edge, init, H_max: float):
        """start_per_attempt: the edge's init trunk (Phase 1b) after its free-floor test, once per process, or the Reject the edge raised."""
        cache = self.__dict__.setdefault("_spa_edges", {})
        if edge.name not in cache:
            lats = th.linspace(-edge.length / 2 + 0.1, edge.length / 2 - 0.1, max(2, int(edge.length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if self.kit.trav_free(edge.base_xy(init.stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                cache[edge.name] = Reject("no_free_edge", f"{edge.name}: free floor at {n_free}/{len(lats)} lateral positions")
            else:
                try:
                    cache[edge.name] = TT.phase1b_init_trunk(self, init, H_max, top=self.top)
                except Reject as e:
                    if e.reason not in ("no_workspace", "edge_too_short", "stance_collision"):
                        raise
                    cache[edge.name] = e
        return cache[edge.name]

    # ------------------------------------------------------------------ start_per_attempt: the lateral-cell memo (train-fast 2026-10-31, SPA_CACHE)
    # Nothing the per-attempt draw computes depends on the attempt's object placement (the spawned objects are parked): Phase 1c's checks,
    # the shelf layout's book poses, Phase 2 (split cone) and the start region depend on template-fixed facts plus the drawn (edge, lateral).
    # So the lateral is discretised into cells of ISPATIALGYM_FAST_SPA_CELL_M over Phase 1c's bounds and the whole result of a cell (init
    # stance, candidates after the start region, region sides, shelf poses, meta spots) is memoised in-process and on disk; an attempt
    # draws the lateral as before, snaps it to its cell and reuses the cell. Off (ISPATIALGYM_FAST_SPA_CACHE=0): the draw below, unchanged.

    def _spa_cache_on(self) -> bool:
        """Whether the lateral-cell memo is live (``ISPATIALGYM_FAST_SPA_CACHE``, default on; announced once per process)."""
        global _SPA_CACHE_ANNOUNCED
        on = fast_on("SPA_CACHE")
        if on and not _SPA_CACHE_ANNOUNCED:
            _SPA_CACHE_ANNOUNCED = True
            log.info("%s: start_per_attempt: SPA_CACHE on - the start is drawn per lateral cell of %.3f m (ISPATIALGYM_FAST_SPA_CELL_M), up to %d cells "
                     "nearest the drawn lateral tried per edge (ISPATIALGYM_FAST_SPA_CELL_TRIES); cells memoised in-process and on disk under %s",
                     self.gp.variant, float(fast_value("SPA_CELL_M", 0.02)), int(fast_value("SPA_CELL_TRIES", 6)), cache_root() or "(no disk cache)")
        return on

    @staticmethod
    def _spa_cand_key(c: Stance) -> Tuple:
        """A Phase-2 candidate's identity across the fresh copies a reused cell hands out: base pose, trunk and focus, rounded."""
        f = c.focus
        return (tuple(round(float(v), 4) for v in c.base_pose2d), round(float(c.translate), 3), round(float(c.tilt), 4),
                None if f is None else tuple(round(float(v), 3) for v in f.flatten().tolist()[:3]))

    def _spa_drop_key(self, c: Stance) -> Tuple:
        """:meth:`_spa_cand_key`; with ``generation.shelf_start_draw`` (oracle-fix 2026-11-06) also the attempt's working arm - a travel that
        fails with one arm working (the other idling) says nothing about the same stance with the arms swapped."""
        key = self._spa_cand_key(c)
        return key + (tuple(self.segment_arms),) if shelf_draw_on(self) else key

    @staticmethod
    def _spa_top_signature(top) -> Dict[str, Any]:
        """What identifies a top raster in a cell's cache key: name, height, grid (resolution, shape, origin) and a hash of its free cells."""
        free = top.free
        return {"name": str(top.name), "z": round(float(top.z), 4), "resolution": round(float(top.resolution), 4), "shape": [int(v) for v in free.shape],
                "origin": [round(float(top.xs[0]), 4), round(float(top.ys[0]), 4)],
                "free_sha": hashlib.sha1(np.ascontiguousarray(free.detach().cpu().numpy()).tobytes()).hexdigest()[:16]}

    @staticmethod
    def _spa_band_signature(bands, sampler=None) -> Optional[Dict[str, Any]]:
        """:func:`tabletop.band_facts` plus the band's key / lattice / trunk count: a band recomputed under the same path changes the key."""
        if bands is None:
            return None
        meta = getattr(bands, "meta", None) or {}
        # of-v2maps 2026-11-07: a map view has no "key" / path - its map key and file name it (a rebuilt map must not reuse stale cells),
        # and the support-first switch changes the cells Phase 2 keeps
        return {**(TT.band_facts(bands) or {}), "key": meta.get("key") or meta.get("map_key"), "map_path": os.path.basename(str(meta.get("map_path") or "")) or None,
                "lattice": meta.get("lattice"), "n_trunks": len(meta.get("trunks") or []),
                "support_first": TT.support_first_tag(sampler, bands),  # i7-unblock: forced on / off, or the automatic rule's tag
                **({"map_tilt": meta["map_tilt"]} if meta.get("map_tilt") else {})}  # i7-plateqtf: the expanding map's rule + overlay version

    @staticmethod
    def _spa_cell_centre(lo: float, hi: float, cell_m: float, i: int) -> float:
        """The lateral a cell stands for: the midpoint of its interval ``[lo + i w, min(hi, lo + (i + 1) w)]`` (the last cell may be narrower)."""
        a = float(lo) + int(i) * float(cell_m)
        return 0.5 * (a + min(float(hi), a + float(cell_m)))

    def _spa_cell_payload(self, edge: EdgeFrame, init_trunk: Stance, margins: Dict[str, Any], H_max: float, nb_top, cell_m: float, n_cells: int, i: int,
                          lateral: float) -> Dict[str, Any]:
        """The cache key of one lateral cell: everything the cell's computation reads - the collision world (``ikc.cache_scope``: scene, rooms,
        floor), the template, the support and its top raster (and the neighbour top of ``start_region_on_neighbors``), the edge, the Phase-1b
        trunk, H_max, both hulls (half extent, height, rest quat), the margins row, the generation knobs Phase 2 / the start region read, the
        sides, the segment arms and their bands, the shelf layout, and the cell itself. With ``generation.shelf_start_draw`` (oracle-fix
        2026-11-06) the cell holds no book poses (``"shelf_poses": "per_attempt"``: :meth:`draw_shelf_layout` places them after the cell)."""
        ex = init_trunk.extras
        arms = list(self.segment_arms)
        payload = {**self.ikc.cache_scope, "template_id": self.choice.template_id, "support": self.support.name, "relation": self.relation,
                "top": self._spa_top_signature(self.top), "nb_top": None if nb_top is None else self._spa_top_signature(nb_top), "edge": edge.to_json(),
                "init_trunk": {"translate": float(init_trunk.translate), "tilt": float(init_trunk.tilt), "stand_off": init_trunk.stand_off,
                               **{k: ex.get(k) for k in ("lam_min", "lam_max", "d_ws", "d_near", "d_far", "t_range", "b", "delta_min", "delta_max")}},
                "H_max": float(H_max),
                "hull": {role: [TT.half_extent(self.hull[role], self.rest[role]), TT.height_of(self.hull[role], self.rest[role]), self.rest[role]] for role in ("target", "partner")},
                "margins": margins,
                "gen": {"split_cone": TT.split_cone_opts(self), "angle_classes": list(TT.phase2_angle_classes(self)),
                        "start_region_lateral_draws": int(self.gen("start_region_lateral_draws", START_REGION_LATERAL_DRAWS)), "gap_min": self.gap_min,
                        "table_end_margin": self.table_end_margin, "partner_gap_extra": self.partner_gap_extra, "support_margin": self.margin_support,
                        "partner_side": self.gen("partner_side"), "meta_link": self.meta_link(), "start_region_on_neighbors": bool(self.gen("start_region_on_neighbors", False)),
                        "push_stance": self.gen("push_stance", True), "reach_screen_place": self.gen("reach_screen_place", True),
                        "shelf_free_span_min_m": float(self.gen("shelf_free_span_min_m", 0.15)),
                        "shelf_spine_behind_edge_m": self.gen("shelf_spine_behind_edge_m"), "shelf_bay_lateral_m": self.gen("shelf_bay_lateral_m")},
                "sides": [float(sd) for sd in self.template_sides()], "pair": {"g_NT": (self.pair or {}).get("g_NT"), "handedness": (self.pair or {}).get("handedness")},
                "segment_arms": arms, "bands": {a: self._spa_band_signature(self.band_of_arm(a), self) for a in arms},
                "shelf_start": None if self.shelf_start is None else {k: self.shelf_start.get(k) for k in ("key", "lean_side", "lean_deg")},
                "cell": {"index": int(i), "n": int(n_cells), "width_m": float(cell_m), "lateral": float(lateral)}}
        if shelf_draw_on(self):
            payload["shelf_poses"] = "per_attempt"
        if self._built_tilts_map() is not None:  # the cell's Phase 2 on the map's built tilts (only when on: keys unchanged off)
            payload["phase2_built_tilts"] = True
        if self._pdc_keep_pairs() is not None:  # follow-up (c): the cell's Phase 2 is cut to the cap (only when on: keys unchanged off)
            payload["place_depth_cap_draw"] = {"basis": place_depth_cap_basis(), "stand_offs": self.push_stance_stand_offs(init_trunk)}
        return payload

    def _spa_cell_compute(self, edge: EdgeFrame, init_trunk: Stance, lateral: float, margins: Dict[str, Any], H_max: float, nb_top) -> Dict[str, Any]:
        """The start of one lateral cell, computed as :meth:`spa_redraw_start` computes a draw without the memo - Phase 1c at the cell's lateral
        (its floor / collision / translate-retry / trunk-sweep checks), the shelf layout's book poses, Phase 2 (split cone), the A1 start region -
        as a JSON-ready dict: ``{"ok": True, "init", "cands", "region_sides", "shelf_pose", "shelf_partner_pose", "meta_spots",
        "start_region_facts", "class1_deep", "n_phase2", "compute_s"}`` or ``{"ok": False, "reject": [reason, message], "compute_s"}`` for a
        rejection of :data:`SPA_CELL_REJECTS` (the draw tries another cell / edge); any other rejection propagates. Sets ``self.edge`` /
        ``self.init_stance`` / the shelf poses / ``meta_spots`` as the plain draw does (:meth:`_spa_apply_cell` sets them from the result)."""
        t0 = time.time()

        def rejected(reason: str, msg: str) -> Dict[str, Any]:
            return {"ok": False, "reject": [reason, msg[:400]], "compute_s": round(time.time() - t0, 2)}

        try:
            init_stance = TT.phase1c_lateral(self, init_trunk, margins, None, top=self.top, lateral=lateral)
        except Reject as e:
            if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                return rejected(e.reason, str(e))
            raise
        init_stance.name = "init"
        self.edge, self.init_stance = edge, init_stance
        self.shelf_pose = self.shelf_partner_pose = None
        phase2_band: Dict[str, Any] = {}
        if self.shelf_start is not None and not shelf_draw_on(self):  # oracle-fix 2026-11-06: shelf_start_draw places the books per attempt
            side = self.shelf_start["lean_side"]
            self.shelf_pose, _ = TT.shelf_lean_pose(self, self.hull["target"], self.rest["target"], self.top, edge, self.support, side)
            self.shelf_partner_pose, _ = TT.shelf_lean_pose(self, self.hull["partner"], self.rest["partner"], self.top, edge, self.support, -side)
            t_lat = (self.world.hull_at(self.hull["target"], self.shelf_pose)[:, :2] - edge.mid) @ edge.tangent
            p_lat = (self.world.hull_at(self.hull["partner"], self.shelf_partner_pose)[:, :2] - edge.mid) @ edge.tangent
            free = float(t_lat.min() - p_lat.max()) if side > 0 else float(p_lat.min() - t_lat.max())
            if free < self.shelf_span_min(self.shelf_pose):
                return rejected("shelf_width", f"{free:.2f} m between the two leaning books on edge {edge.name}")
        if self.shelf_start is not None:
            phase2_band = {"bands": self.band_of_arm(self.segment_arms[0])}
        cands = TT.phase2_manipulation_stance(self, init_stance, TT.half_extent(self.hull["target"], self.rest["target"]),
                                              target_height=TT.height_of(self.hull["target"], self.rest["target"]), top=self.top if nb_top is None else nb_top,
                                              arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self), keep_pairs=self._pdc_keep_pairs(),
                                              **phase2_band)
        kept = cands if self.shelf_start is not None else self.start_region(cands)
        if not kept:
            if not cands:
                return rejected("no_manipulation_stance", f"lateral {lateral:+.3f}: Phase 2 left no survivor")
            return rejected("no_start_region", f"lateral {lateral:+.3f}: no start cell for the partner on the " + " / ".join(TT.side_name(sd) for sd in self.template_sides()))
        spots = self.meta_spots
        return {"ok": True, "init": init_stance.to_cache(), "cands": [c.to_cache() for c in kept], "region_sides": [float(sd) for sd in self.region_sides],
                "shelf_pose": pose_to_sidecar(self.shelf_pose), "shelf_partner_pose": pose_to_sidecar(self.shelf_partner_pose),
                "meta_spots": None if spots is None else [[str(m), [float(xy[0]), float(xy[1])]] for m, xy in spots],
                "start_region_facts": None if self.shelf_start is not None else self.template_facts.get("start_region"),
                "class1_deep": getattr(self, "_split_cone_class1_deep", None), "n_phase2": len(cands), "compute_s": round(time.time() - t0, 2)}

    def _spa_disk_hit_valid(self, edge: EdgeFrame, hit: Any) -> bool:
        """A disk entry is trusted when it is well-formed and (for a start) its init stance is still whole-body collision-free here - one batched
        collision check; a cached rejection is trusted as it is (its key names the collision world and every input)."""
        if not isinstance(hit, dict) or "ok" not in hit:
            return False
        if not hit["ok"]:
            return isinstance(hit.get("reject"), list) and len(hit["reject"]) == 2
        try:
            if not hit.get("cands"):
                return False
            init = Stance.from_cache(hit["init"], edge=edge)
        except (KeyError, TypeError, ValueError):
            return False
        return bool(self.ikc.stance_collision_free(init))

    def _spa_cell(self, edge: EdgeFrame, init_trunk: Stance, i: int, n_cells: int, lo: float, hi: float, cell_m: float, margins: Dict[str, Any], H_max: float,
                  nb_top) -> Tuple[Dict[str, Any], str]:
        """Cell ``i`` of ``edge`` - ``(cell, how)``, ``how`` in ``memo`` (this process), ``hit`` (disk), ``miss`` (computed now and written). The memo
        keeps every cell as its JSON text, so each use parses a fresh copy: nothing an attempt mutates reaches the next one."""
        memo = self.__dict__.setdefault("_spa_cells", {})
        mkey = (edge.name, int(i))
        if shelf_draw_on(self):  # oracle-fix 2026-11-06 (shelf_start_draw): a cell per working arm (its band's candidates, the other arm idling)
            mkey = mkey + (tuple(self.segment_arms),)
        if mkey in memo:
            return json.loads(memo[mkey]), "memo"
        lateral = self._spa_cell_centre(lo, hi, cell_m, i)
        t0 = time.time()
        key = cache_key(SPA_CELL_KIND, self._spa_cell_payload(edge, init_trunk, margins, H_max, nb_top, cell_m, n_cells, i, lateral))
        hit = cache_get(key)
        if hit is not None:
            if self._spa_disk_hit_valid(edge, hit):
                memo[mkey] = json.dumps(hit, default=_spa_json_default)
                log.info("%s: start draw cache hit edge=%s cell=%d/%d lateral=%+.3f in %.2fs (%s)", self.gp.variant, edge.name, i, n_cells, lateral, time.time() - t0,
                         f"{len(hit['cands'])} candidates" if hit["ok"] else f"reject [{hit['reject'][0]}]")
                return json.loads(memo[mkey]), "hit"
            log.info("%s: start draw cache entry edge=%s cell=%d/%d is stale or malformed (its init stance collides here): recomputed", self.gp.variant, edge.name, i, n_cells)
        cell = self._spa_cell_compute(edge, init_trunk, lateral, margins, H_max, nb_top)
        text = json.dumps(cell, default=_spa_json_default)
        memo[mkey] = text
        cache_put(key, json.loads(text), round_floats=False)
        log.info("%s: start draw cache miss edge=%s cell=%d/%d lateral=%+.3f computed in %.1fs (%s)", self.gp.variant, edge.name, i, n_cells, lateral, time.time() - t0,
                 f"{len(cell['cands'])} candidates of {cell['n_phase2']} Phase-2 survivors" if cell["ok"] else f"reject [{cell['reject'][0]}] {cell['reject'][1][:120]}")
        return json.loads(text), "miss"

    def _spa_apply_cell(self, edge: EdgeFrame, cell: Dict[str, Any]) -> bool:
        """Make a (non-rejected) cell this attempt's start: fresh stances from the cell, the candidates dropped for their travel in this
        process left out (``_spa_dropped``), the region sides cut to the live candidates as :meth:`drop_candidate` cuts them. False when no
        live candidate (or, on the tabletop, no side with one) is left: the caller tries the next cell."""
        init = Stance.from_cache(cell["init"], edge=edge)
        init.name = "init"
        cands = [Stance.from_cache(d, edge=edge) for d in cell["cands"]]
        dropped = self.__dict__.get("_spa_dropped") or set()
        if dropped:
            cands = [c for c in cands if self._spa_drop_key(c) not in dropped]
        sides = [float(sd) for sd in (cell.get("region_sides") or [])]
        if self.shelf_start is None:
            sides = [sd for sd in sides if any(TT.in_start_region(c, sd) for c in cands)]
            if not sides:
                return False
        elif not cands:
            return False
        self.edge, self.init_stance = edge, init
        self.shelf_pose, self.shelf_partner_pose = pose_from_sidecar(cell.get("shelf_pose")), pose_from_sidecar(cell.get("shelf_partner_pose"))
        self.manip_candidates = cands
        self.region_sides = sides
        spots = cell.get("meta_spots")
        self.meta_spots = None if spots is None else [(str(m), th.tensor([float(xy[0]), float(xy[1])])) for m, xy in spots]
        if cell.get("start_region_facts") is not None:
            self.template_facts["start_region"] = cell["start_region_facts"]
        self._split_cone_class1_deep = cell.get("class1_deep")
        return True

    def _spa_redraw_start_cached(self, rng: np.random.Generator) -> None:
        """:meth:`spa_redraw_start` through the lateral-cell memo. The edges come in the same ``rng.permutation`` order as the plain draw; for the
        first edge with a Phase-1b trunk the lateral is drawn uniformly over Phase 1c's bounds with the same ``rng.uniform`` call, snapped to its
        cell, and the cell fetched (memo / disk / computed). A rejected cell (Phase 1c finds no stance at that lateral, no Phase-2 survivor, no
        start cell for a partner side, the shelf too narrow) sends the draw to the next cell nearest the drawn lateral, up to
        ``ISPATIALGYM_FAST_SPA_CELL_TRIES`` cells, then to the next edge. ``_spa_facts`` records the cell, the edge's cell count, where the cell
        came from (``memo`` / ``hit`` / ``miss``) and the draw's time."""
        t0 = time.time()
        H_max = max(TT.height_of(self.hull["target"], self.rest["target"]), TT.height_of(self.hull["partner"], self.rest["partner"]))
        nb_top = self.neighbor_start_top()
        margins = self.lateral_margins()
        cell_m = max(1e-3, float(fast_value("SPA_CELL_M", 0.02)))
        tries = max(1, int(fast_value("SPA_CELL_TRIES", 6)))
        errors: List[str] = []
        last_reject: Optional[str] = None
        n_tried = 0
        from ispatialgym.sampling import campaign_tabletop_starts as CTS
        plan = self.campaign_pair_plan() if os.environ.get(CPG.ENV) else None
        anchor = getattr(plan, 'anchor_base_pose2d', None)
        for edge, init in CTS.ordered_edges(TT.edge_frames(self, self.top, rng), anchor):
            trunk = self._spa_edge_trunk(edge, init, H_max)
            if isinstance(trunk, Reject):
                errors.append(f"{edge.name}: {trunk}")
                continue
            try:
                bounds, _info, _b = TT.phase1c_bounds(self, trunk, margins, top=self.top, strict=True)
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e}")
                    continue
                raise
            lo, hi = float(bounds[0]), float(bounds[1])
            n_cells = max(1, int(math.ceil((hi - lo) / cell_m - 1e-9)))
            lat = (float(rng.uniform(lo, hi)) if anchor is None else
                   CTS.near_lateral(rng, lo, hi, CTS.edge_lateral(edge, anchor)))
            order = sorted(range(n_cells), key=lambda k: (abs(self._spa_cell_centre(lo, hi, cell_m, k) - lat), k))[:tries]
            for i in order:
                n_tried += 1
                cell, how = self._spa_cell(edge, trunk, i, n_cells, lo, hi, cell_m, margins, H_max, nb_top)
                if not cell["ok"]:
                    last_reject = cell["reject"][0]
                    errors.append(f"{edge.name} cell {i}/{n_cells}: [{cell['reject'][0]}] {cell['reject'][1][:160]}")
                    continue
                if not self._spa_apply_cell(edge, cell):
                    errors.append(f"{edge.name} cell {i}/{n_cells}: every candidate dropped for its travel in this process")
                    continue
                self._spa_facts = {"edge": edge.name, "lateral_m": round(float(self.init_stance.lateral), 4),
                                   "trunk": [round(float(self.init_stance.translate), 3), round(float(self.init_stance.tilt), 3)],
                                   "base_pose2d": [round(float(v), 4) for v in self.init_stance.base_pose2d], "n_manipulation_candidates": len(self.manip_candidates),
                                   "cell": int(i), "n_cells": int(n_cells), "cell_m": cell_m, "lateral_drawn_m": round(lat, 4), "cells_tried": n_tried,
                                   "cache": how, "draw_s": round(time.time() - t0, 3)}
                log.info("%s: start_per_attempt (SPA_CACHE): edge %s cell %d/%d (%s) lateral %+.3f (drawn %+.3f), %d candidates, sides %s, %.2fs", self.gp.variant,
                         edge.name, i, n_cells, how, self.init_stance.lateral, lat, len(self.manip_candidates), [TT.side_name(sd) for sd in self.region_sides],
                         time.time() - t0)
                return
        reason = last_reject if last_reject in ("no_manipulation_stance", "no_start_region", "shelf_width") else "no_free_edge"
        raise Reject(reason, "start_per_attempt (SPA_CACHE): " + ("; ".join(errors) or f"{self.support.name}: no edge with a free, collision-free stance"))

    def spa_redraw_start(self, rng: np.random.Generator) -> None:
        """start_per_attempt (train85): this attempt's own edge, init stance and Phase-2 pool cut to the A1 start region (lateral redrawn until it has a cell) from the attempt's ``rng``, drawn as
        ``sample_template`` draws the template's (every edge with a free-floor run, random order; Phase 1c's lateral); the shelf layout's
        book poses follow the edge. The segment arms and their bands are the template's. Train-fast 2026-10-31: with ``SPA_CACHE`` on
        (:meth:`_spa_cache_on`) the draw goes through the lateral-cell memo (:meth:`_spa_redraw_start_cached`); off, the draw below."""
        if self._spa_cache_on():
            return self._spa_redraw_start_cached(rng)
        t0 = time.time()
        H_max = max(TT.height_of(self.hull["target"], self.rest["target"]), TT.height_of(self.hull["partner"], self.rest["partner"]))
        nb_top = self.neighbor_start_top()
        margins = self.lateral_margins()
        errors: List[str] = []
        init_stance = init_trunk = None
        for edge, init in TT.edge_frames(self, self.top, rng):
            trunk = self._spa_edge_trunk(edge, init, H_max)
            if isinstance(trunk, Reject):
                errors.append(f"{edge.name}: {trunk}")
                continue
            try:
                init_stance = TT.phase1c_lateral(self, trunk, margins, rng, top=self.top)
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e}")
                    continue
                raise
            init_trunk, self.edge = trunk, edge
            break
        if init_stance is None:
            raise Reject("no_free_edge", "start_per_attempt: " + ("; ".join(errors) or f"{self.support.name}: no edge with a free, collision-free stance"))
        init_stance.name = "init"
        self.init_stance = init_stance
        self.shelf_pose = self.shelf_partner_pose = None
        phase2_band: Dict[str, Any] = {}
        if self.shelf_start is not None and not shelf_draw_on(self):  # oracle-fix 2026-11-06: shelf_start_draw places the books per attempt
            side = self.shelf_start["lean_side"]
            self.shelf_pose, _ = TT.shelf_lean_pose(self, self.hull["target"], self.rest["target"], self.top, self.edge, self.support, side)
            self.shelf_partner_pose, _ = TT.shelf_lean_pose(self, self.hull["partner"], self.rest["partner"], self.top, self.edge, self.support, -side)
            t_lat = (self.world.hull_at(self.hull["target"], self.shelf_pose)[:, :2] - self.edge.mid) @ self.edge.tangent
            p_lat = (self.world.hull_at(self.hull["partner"], self.shelf_partner_pose)[:, :2] - self.edge.mid) @ self.edge.tangent
            free = float(t_lat.min() - p_lat.max()) if side > 0 else float(p_lat.min() - t_lat.max())
            if free < self.shelf_span_min(self.shelf_pose):
                raise Reject("shelf_width", f"start_per_attempt: {free:.2f} m between the two leaning books on edge {self.edge.name}")
        if self.shelf_start is not None:
            phase2_band = {"bands": self.band_of_arm(self.segment_arms[0])}
        failed: List[str] = []
        phase2_any = False
        for i in range(max(1, int(self.gen("start_region_lateral_draws", START_REGION_LATERAL_DRAWS)))):
            if i:
                try:
                    self.init_stance = TT.phase1c_lateral(self, init_trunk, margins, rng, top=self.top)
                except Reject as e:
                    failed.append(f"redraw: [{e.reason}] {e}")
                    break
                self.init_stance.name = "init"
            cands = TT.phase2_manipulation_stance(self, self.init_stance, TT.half_extent(self.hull["target"], self.rest["target"]),
                                                  target_height=TT.height_of(self.hull["target"], self.rest["target"]), top=self.top if nb_top is None else nb_top,
                                                  arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self), keep_pairs=self._pdc_keep_pairs(),
                                                  **phase2_band)
            phase2_any = phase2_any or bool(cands)
            self.manip_candidates = cands if self.shelf_start is not None else self.start_region(cands)
            if self.manip_candidates:
                break
            failed.append(f"lateral {self.init_stance.lateral:+.2f}: " + ("Phase 2 left no survivor" if not cands else "no start cell for the partner on the "
                                                                          + " / ".join(TT.side_name(sd) for sd in self.template_sides())))
        if not self.manip_candidates:
            raise Reject("no_manipulation_stance" if not phase2_any else "no_start_region", "start_per_attempt: " + "; ".join(failed))
        self._spa_facts = {"edge": self.edge.name, "lateral_m": round(float(self.init_stance.lateral), 4),
                           "trunk": [round(float(self.init_stance.translate), 3), round(float(self.init_stance.tilt), 3)],
                           "base_pose2d": [round(float(v), 4) for v in self.init_stance.base_pose2d], "n_manipulation_candidates": len(self.manip_candidates)}
        # train-fast 2026-10-31 (facts only, no behaviour): the lateral's cell at the SPA_CACHE cell width, so the logs of both paths show the
        # same distribution over the edge
        lb = self.init_stance.extras.get("lateral_bounds")
        if lb and len(lb) == 2:
            cell_m = max(1e-3, float(fast_value("SPA_CELL_M", 0.02)))
            lo, hi = float(lb[0]), float(lb[1])
            n_cells = max(1, int(math.ceil((hi - lo) / cell_m - 1e-9)))
            self._spa_facts.update({"cell": int(min(n_cells - 1, max(0, math.floor((float(self.init_stance.lateral) - lo) / cell_m)))), "n_cells": n_cells,
                                    "cell_m": cell_m, "cache": "off", "draw_s": round(time.time() - t0, 3)})

    def campaign_pair_plan(self):
        if not os.environ.get(CPG.ENV):
            return None
        if not hasattr(self, "_campaign_group_started_at"):
            self._campaign_group_started_at = time.time()
        try:
            plan = CPG.load_pair_plan(template_id=self.ctx.template_id, scene=self.scene, rooms=self.choice.rooms,
                                     relation=self.relation, activated_at=self._campaign_group_started_at,
                                     expected_sha256=getattr(self, "_campaign_pair_plan_sha", None))
            if getattr(self, "_campaign_group_written", 0) >= plan.max_members:
                raise Reject("campaign_group_complete", "pair goal-group member limit reached")
            if (plan.target_name, plan.partner_name) != (self.bindings.one("target").name, self.bindings.one("partner").name):
                raise Reject("campaign_goal_plan", "pair plan role bindings differ from template")
            if self.shelf_partner_pose is not None or self.meta_spots is not None:
                raise Reject("campaign_goal_plan", "shelf/meta-link pair goal adapter is pending")
            self._campaign_pair_plan_sha = plan.sha256
            return plan
        except CPG.GoalGroupTimeout as exc:
            raise Reject("campaign_group_timeout", str(exc)) from exc
        except (CPG.GoalPlanError, OSError, ValueError) as exc:
            raise Reject("campaign_goal_plan", str(exc)) from exc

    def campaign_group_remaining_s(self):
        plan = self.campaign_pair_plan()
        return None if plan is None else max(0.0, plan.deadline_at - time.time())

    def campaign_verify_pair(self, draft, *, settled=False):
        plan = getattr(self, "_campaign_pair_active", None)
        if plan is None:
            return
        self.campaign_group_remaining_s()
        partner = self.objects[plan.partner_name].get_position_orientation()
        dp, dq = CPG.pose_error(partner, plan.partner)
        position_tol, orientation_tol = self.tol("position", .03), self.tol("orientation", .20)
        partner_tol = min(position_tol, float(self.gen("partner_moved_max_m", .005)))
        group = draft.summary["campaign_goal_group"]
        group["partner_residual"] = {"position_m": dp, "orientation_rad": dq}
        if dp > partner_tol or dq > orientation_tol:
            raise Reject("campaign_partner_drift", f"fixed partner drifted {dp:.6f} m / {dq:.6f} rad")
        if settled:
            try:
                actual = self.objects[plan.target_name].get_position_orientation()
                dp, dq = CPG.pose_error(actual, plan.goal)
                group["settled_goal_residual"] = {"position_m": dp, "orientation_rad": dq}
                plan.verify_settled(actual, partner, position_tol=position_tol, orientation_tol=orientation_tol, partner_position_tol=partner_tol)
            except CPG.GoalPlanError as exc:
                raise Reject("campaign_goal_drift", str(exc)) from exc
        if plan.release_policy == "native_grasp_height":
            # This is the per-grasp execution release, not the fixed settled
            # goal. Keep the native rim-clearance pose already checked by IK.
            release = draft.goal_poses.get(plan.target_name)
            if release is None:
                raise Reject("campaign_release_missing", "native inside release was not recorded")
            group["actual_release"] = {"position": [float(v) for v in release[0]], "orientation": [float(v) for v in release[1]]}
            group["release_geometry"] = dict(self.release)
        else:
            draft.goal_poses[plan.target_name] = tuple(G.t(v) for v in plan.release)
        target = draft.pose_targets[plan.target_name]
        target["position"] = list(plan.goal[0])
        if plan.relation == "inside":
            target.update({"orientation": None, "orientation_metric": None, "orientation_tolerance": None})
        elif target.get("orientation_metric") is not None:
            target["orientation"] = list(plan.goal[1])
        partner_target = draft.pose_targets[plan.partner_name]
        partner_target["position"] = list(plan.partner[0])
        if partner_target.get("orientation_metric") is not None:
            partner_target["orientation"] = list(plan.partner[1])
        draft.summary["campaign_goal_group"]["phase"] = "settled_goal_verified" if settled else "settled_geometry_verified"

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        if os.environ.get(CPG.ENV):
            self._campaign_pair_active = self.campaign_pair_plan()
            plan = self._campaign_pair_active
            partner = self.obj("partner")
            partner.set_position_orientation(*(G.t(v) for v in plan.partner))
            partner.keep_still()
            if self.relation in ("ontop", "under"):
                # The release ray must see the fixed partner at its new pose.
                # sync_world below updates cuRobo meshes, not PhysX queries;
                # without a physics step the ray can still hit only the table
                # and the release silently falls back to the saucer's rim top.
                self.world.step_physics(1, still=[self.robot, partner])
                dp, dq = CPG.pose_error(partner.get_position_orientation(), plan.partner)
                partner_tol = min(self.tol("position", .03), float(self.gen("partner_moved_max_m", .005)))
                if dp > partner_tol or dq > self.tol("orientation", .20):
                    raise Reject("campaign_partner_drift", f"fixed partner drifted during query sync by {dp:.6f} m / {dq:.6f} rad")
                TT._count(self, "campaign_partner.surface_query_synced")
            self.kit.sync_world()
            self._campaign_latest_proposal = {"start_poses": {plan.partner_name: plan.partner},
                                              "goal_poses": {plan.target_name: plan.goal, plan.partner_name: plan.partner},
                                              "group_id": plan.group_id}
        d = Draft()
        self._stance_shift_budget = {"scan": 0, "seen": set()}  # i7-t03 item 1: the scan's per-attempt budget
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if shelf_free_start_on(self) and self.shelf_pose is not None:  # i7-t03 round 2 (a): the free-floor start as the edge stance's fallback
            self.init_stance = self.shelf_free_start(rng, self.init_stance)
        init = self.init_stance
        self.sidecar.add_stance(init)
        cam_init = self.camera_pose_at(init)
        # A1: the target starts in the start region of the attempt's partner side (7e: never on a candidate dropped for its travel)
        checked = self.manip_candidates
        if getattr(self, '_campaign_pair_active', None) is not None and getattr(self, '_split_cone_cands', None):
            checked = self._split_cone_cands
        cands = [c for c in checked if TT.in_start_region(c, self.region_side) and TRAVEL_DROPPED not in c.extras]
        # owner 3.6 (2026-11-08, ISPATIALGYM_PLACE_DEPTH_CAP): the touching place depth capped by the push map's reach - before the draw
        self._pdc_shadow_drafted = False
        if getattr(self, "_lift_screen_shadow", False):  # the previous attempt was drafted past a would-be lift rejection and not rejected
            log.info("TOUCH_LIFT_SCREEN shadow outcome: written")
        self._lift_screen_shadow = False
        if getattr(self, "_a1_screen_shadow", False):  # the same for the A1 place screen's shadow
            log.info("TOUCH_A1_SCREEN shadow outcome: written")
        self._a1_screen_shadow = False
        cands, pdc = self.place_depth_cap_filter(cands)
        order = rng.permutation(len(cands))
        from ispatialgym.sampling import campaign_runtime
        frame_recovery = self.relation == "inside" and campaign_runtime.inside_frame_recovery()
        if (os.environ.get("ISPATIALGYM_STANCE_FRAME_AIM", "") == "1" or frame_recovery or getattr(self, '_campaign_pair_active', None) is not None) and self.shelf_pose is None and self.relation != "touching" and cands:
            # oracle-fix 2026-11-07 (owner prompt E 2, opt-in): the candidates whose expected place sits deepest in their manipulation image first
            # (stance_frame_aim: the head camera from base + trunk, no IK); the random order breaks ties
            aim = {int(i): self.stance_frame_aim(cands[int(i)]) for i in order}
            order = sorted(order, key=lambda i: -aim[int(i)] if math.isfinite(aim[int(i)]) else 1e9)
            for i, v in aim.items():
                cands[i].extras["frame_aim_px"] = round(v, 1) if math.isfinite(v) else None
            self.counters["frame_aim.ranked"] = self.counters.get("frame_aim.ranked", 0) + 1
        if self.shelf_pose is not None:  # the start is fixed: the stances whose focus is nearest it first
            sx = self.shelf_pose[0][:2]
            order = sorted(order, key=lambda i: float(th.linalg.norm(cands[int(i)].focus[:2] - sx)) if cands[int(i)].focus is not None else 1e9)
        errors = []
        default = self.segment_arms[0]
        # i7-memory2 (owner 2026-11-08 2.3): the per-alias outcome memory's PICK hook - the candidates of a flagged (arm x trunk[, stand-off])
        # pick bin move to the end of the order above (a Thompson draw keeps them in place, never below the floor; never removed). Memory off
        # / nothing flagged: the order unchanged and no random number drawn (alias_memory.reorder_pick; kit alias_memory.pick.*)
        order = AM.reorder_pick(self, order, cands, lambda st: default, rng)
        plan = getattr(self, "_campaign_pair_active", None)
        if plan is not None and self.relation == "inside":
            from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
            written = getattr(self, "_campaign_written_inside_stances", {}).get(plan.sha256, [])
            order = prioritize_written_stances(order, cands, written)
        for k in order[:6]:
            cand = cands[int(k)]
            pdc_c = pdc.get(id(cand))
            pdc_wr = bool(pdc_c and pdc_c["remove"])  # shadow: the cap would have removed this candidate
            if pdc_wr:
                TT._count(self, "place_depth_cap.shadow.tried")
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, [default]
            manip.extras.update(cand.extras)
            if self.shelf_start is not None:
                # an idle posture Phase 2 recorded belongs to the arm the stance search idled: the shelf layout's arm works, the other
                # arm idles in its first collision-free posture
                st_arm = TT.stance_for_arm(self, cand.with_trunk(cand.translate, cand.tilt, "manipulation"), default)
                if st_arm is None:
                    errors.append(f"candidate {int(k)}: no collision-free idle posture of the other arm with the {default} arm working")
                    continue
                TT.adopt_arm(manip, st_arm)
            # the other arm is tried only where its band covers the focus (owner 2026-09-27), and only the arms that can place the partner on
            # the attempt's side from here (A1 start region)
            arms_here = [a for a in self.segment_arms if a == default or a in (cand.extras.get("focus_arms") or [])]
            if self.region_side is not None and REGION_ARMS in cand.extras:
                arms_here = [a for a in arms_here if a in cand.extras[REGION_ARMS].get(TT.side_name(self.region_side), [])]
                if not arms_here:
                    errors.append(f"candidate {int(k)}: no offered arm meets the {TT.side_name(self.region_side)} partner's reach rule")
                    continue
            memo_key = None
            try:
                ft = getattr(getattr(self, "ikc", None), "finger_touch", None)  # oracle-fix 2026-11-04 round 3: the pre-shaped opening when the preshape rule is on
                pre_open = ft.opening() if (ft is not None and getattr(ft, "preshape", False) and callable(getattr(ft, "opening", None))) else None
                per_grasp = bool(self.gen("shelf_transit_check") and self.gen("shelf_transit_per_grasp"))
                if self.shelf_pose is not None and self.gen("fixed_start_memo"):
                    # oracle-fix 2026-11-05 (speed, opt-in): the fixed start's grasp verdict per stance - the start pose is the template's and the
                    # stances come from the cached start cells, so the same exact IK / transit check failed again and again (t03tip5: 762 'no library
                    # grasp reaches the fixed start pose' at 19.5 s and 1551 transit rejections at 29.8 s, 97 % of 17.6 h)
                    memo_key = self._fixed_start_memo_key(manip, default, pre_open, per_grasp)
                    hit = self.__dict__.setdefault("_fixed_start_memo", {}).get(memo_key)
                    if hit is None:
                        hit = self._fixed_start_memo_persisted(memo_key)
                        if hit is not None:
                            TT._count(self, "fixed_start_memo.persisted_hit")
                            self._fixed_start_memo[memo_key] = hit
                    if hit is not None:
                        TT._count(self, "fixed_start_memo.hit")
                        raise Reject("grasp_unreachable", f"{hit} (remembered: this stance's fixed-start grasp failed in an earlier attempt)")
                # §4.5 step 5: transition init -> manipulation
                self.sidecar.transitions.clear()
                try:
                    with phase_timer(self, "cand_travel"):  # i7-nextto: measurement only
                        self.record_stance_travel(init, manip)
                except Reject as e:
                    if travel_failure_deterministic(e, self.world) and not init.extras.get("free_start"):
                        # i7-t03 round 2 (a): a travel from a FREE-FLOOR start (drawn per attempt) says nothing about the candidate - it is not
                        # dropped (round-2 A/B b: 8 no_path drives from free starts dropped the cell's candidates -> no_manipulation_stance x34)
                        self.drop_candidate(cand, e)
                    raise
                # §4.6 step 1: target init pose from the feasible set F at the focus (exact IK confirmation); the upright shelf start is
                # fixed, its grasp confirmed with nothing ignored (the board above and the side walls are obstacles at the grasp pose)
                if self.shelf_pose is not None:
                    t_pose = self.shelf_pose

                    def fixed_start(m: Stance, _manip=manip, _pre_open=pre_open, _per_grasp=per_grasp, _default=default, again: bool = False) -> GraspRecord:
                        if m is not _manip or again:  # i7-t03 item 1: a shifted stance travels from init itself (the candidate's travel is recorded above)
                            self.sidecar.transitions.clear()
                            self.record_stance_travel(init, m)
                        # oracle-fix 2026-11-05 (speed, opt-in generation.shelf_transit_per_grasp): the transit check judges each grasp that passes
                        # the IK - the first with a clear transit is taken - instead of failing the stance on the first grasp's transit
                        accept = (lambda rec: self.shelf_transit_why(m, rec, _default, _pre_open)) if _per_grasp else None
                        g = TT.grasp_at_pose(self, m, t_pose, self.grasps, [], self.edge.yaw, arm=_default, scale_key=self.choice.models["target"].scale,
                                             fingers=_pre_open, accept=accept, prescreen=bool(self.gen("fixed_start_prescreen")))
                        if _pre_open is not None:  # oracle-fix 2026-11-04 round 4: checked pre-shaped -> the oracle's transit to the pre-grasp pre-shaped too
                            g.extras["preshape_transit"] = True
                        why_transit = self.shelf_transit_why(m, g, _default, _pre_open) if (self.gen("shelf_transit_check") and not _per_grasp) else None
                        if why_transit is not None:
                            raise Reject("grasp_unreachable", why_transit)
                        if shelf_pregrasp_frame_on(self) and self.shelf_tip():
                            self.shelf_pregrasp_frame_check(m, g)
                        if self.shelf_backoff_m() > 0.0:
                            # oracle-fix 2026-11-06 (owner feedback on the touching#03 demo, opt-in generation.shelf_backoff_transit_m): the arrival
                            # transition recorded again backed off, the arm carried to the pre-grasp inside it (:meth:`record_shelf_backoff`)
                            self.record_shelf_backoff(m, g, _default, _pre_open)
                        return g

                    try:
                        # i7-t03 2026-11-08 (owner item 1, ISPATIALGYM_SHELF_STANCE_SHIFT, default off): a failure the base position can change
                        # tries the computed / scanned stances along the edge with the same checks (shelf_fixed_start); off = fixed_start(manip)
                        manip, grasp = self.shelf_fixed_start(init, cand, manip, default, fixed_start)
                    except Reject as e:
                        if memo_key is not None:
                            self._fixed_start_memo[memo_key] = str(e)[:300]
                            self._fixed_start_memo_persisted(memo_key, str(e)[:300])
                        raise
                else:
                    with phase_timer(self, "cand_grasp"):  # i7-nextto: measurement only
                        t_pose, grasp = self.phase3_two_tier(manip, rng, arms=arms_here)
                if grasp.arm != default:
                    self.sidecar.transitions.clear()
                    self.record_stance_travel(init, manip)  # the other arm idles now
                self.apply_height_extra_to_grasp(grasp, t_pose, manip)  # A/B knob generation.contact_height_extra_m (0 = no-op)
                # the segment's later contacts; when they fail with the chosen arm, the whole segment is retried with the other arm if
                # its grasp passed too (owner 2026-09-27: the fallback spans the segment)
                tried: List[str] = []
                retries = place_partner_retries(self.relation)  # i7-nextto (ISPATIALGYM_PLACE_PARTNER_RETRY, default 0 = off)
                if frame_recovery and getattr(self, "_campaign_pair_active", None) is None:
                    # A rejected random partner must not consume the already
                    # validated grasp without trying another bounded placement.
                    # Fixed sibling groups keep their exact partner pose.
                    retries = max(retries, 2)
                retried = 0
                while True:
                    summary0 = {kk: (list(vv) if isinstance(vv, list) else vv) for kk, vv in d.summary.items()} if retries else None
                    try:
                        with phase_timer(self, "cand_segment"):  # i7-nextto: measurement only
                            p_pose = self._segment_draw(rng, d, int(k), t_pose, grasp, manip, cam_init, pname)
                        if retried:
                            TT._count(self, "place_retry.rescued")
                        break
                    except Reject as e:
                        if retries and (e.reason in PLACE_PARTNER_RETRY_REASONS or getattr(self, "_place_screen_reject", False)):
                            # i7-nextto: the grasp passed and a partner-dependent check failed - a fresh partner draw with the same grasp first
                            # (the records of the failed segment draw undone as the caller below undoes them)
                            if not retried:
                                TT._count(self, "place_retry.armed")
                            if retried < retries:
                                retried += 1
                                TT._count(self, f"place_retry.retried.{'place_screen' if getattr(self, '_place_screen_reject', False) else e.reason}")
                                self.sidecar.stances.pop("manipulation", None)
                                self.sidecar.grasps.clear()
                                OF.drop_records(self)
                                d.summary.clear()
                                d.summary.update(summary0)
                                continue
                            TT._count(self, "place_retry.exhausted")
                        tried.append(grasp.arm)
                        others = [a for a in TT.fallback_arms(grasp) if a not in tried]
                        if not others:
                            raise
                        self.sidecar.stances.pop("manipulation", None)
                        self.sidecar.grasps.clear()
                        OF.drop_records(self)
                        g2 = TT.segment_fallback_grasp(self, manip, t_pose, grasp, self.grasps, others[0], ignore_for_grasp=self.support_objects(),
                                                       scale_key=self.choice.models["target"].scale, support_z=float(self.top.z), why=f"[{e.reason}] {e}")
                        if g2 is None:
                            raise
                        grasp = g2
                        self.sidecar.transitions.clear()
                        self.record_stance_travel(init, manip)
                        if self.shelf_pose is not None and self.shelf_backoff_m() > 0.0:
                            self.record_shelf_backoff(manip, grasp, grasp.arm, pre_open)  # oracle-fix 2026-11-06: the other arm's transit backed off too
                        self.apply_height_extra_to_grasp(grasp, t_pose, manip)
                        retried = 0  # i7-nextto: the other arm's segment gets its own partner retries
                # carry height (4A, :meth:`carry_rule`): the target (bottom at the table at init) clears the partner's top by carry_clearance_m on
                # the way; touching only the table (set down beside the partner; "auto": raised by the push stance when its drive passes over it)
                carry_rule = self.carry_rule()
                grasp.carry_height = self.carry_height_over(carry_rule == "partner")
                if self.shelf_start is not None:
                    # the shelf layout slides the book along the board to the partner's side: it clears the board, not the partner
                    # (a lift by the partner's height runs into the board above; the lift check ignores the support)
                    grasp.carry_height = float(self.gen("shelf_carry_height_m", SHELF_CARRY_HEIGHT_M))
                    carry_rule = "shelf"
                d.summary["carry_lift"] = {"rule": carry_rule, "height_m": round(float(grasp.carry_height), 4)}
                d.object_poses[tname] = t_pose
                d.object_poses[pname] = p_pose
                campaign_plan = getattr(self, "_campaign_pair_active", None)
                if campaign_plan is not None:
                    d.summary["campaign_goal_group"] = campaign_plan.summary()
                    d.summary["campaign_goal_group"]["requested_start"] = {"target": _jsonable(t_pose), "grasp": grasp.grasp_name}
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"relation": self.relation, "manipulation_candidate": int(k), "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw},
                                  "manipulation_stance": manip.to_meta(), "pair": self.pair, "arm": TT.segment_arm_summary(grasp, arms_here),
                                  "partner_draw": self.partner_draw})
                if getattr(self, "_shelf_draw", None) is not None:  # oracle-fix 2026-11-06 (shelf_start_draw): the attempt's compartment, boards, leans
                    d.summary["shelf_start"] = dict(self._shelf_draw, arm=grasp.arm)
                grasp.extras["arm_offer"] = TT.arm_offer_record(self.segment_arms, arms_here, grasp, self.arm_offer_why)
                d.pose_targets[pname] = self._pose_target("partner", p_pose)
                d.pose_targets[tname] = self._pose_target("target", t_pose)
                d.summary["target_draft_pose"] = [[float(v) for v in t_pose[0]], [float(v) for v in t_pose[1]]]
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, self.choice.models["partner"].category.replace("_", " "))
                if pdc_c is not None:
                    d.summary["place_depth_cap"] = dict(pdc_c, mode=place_depth_cap_mode(),
                                                        place_depth_m=round(float(manip.local(t_pose[0][:2])[0]), 4))
                    if pdc_wr:
                        TT._count(self, "place_depth_cap.shadow.drafted")
                        self._pdc_shadow_drafted = True
                return d
            except Reject as e:
                if getattr(self, "_lift_screen_shadow", False):  # follow-up (a) shadow: this candidate died later in the draw
                    log.info("TOUCH_LIFT_SCREEN shadow outcome: candidate [%s] %s", e.reason, str(e)[:160])
                    self._lift_screen_shadow = False
                if getattr(self, "_a1_screen_shadow", False):
                    log.info("TOUCH_A1_SCREEN shadow outcome: candidate [%s] %s", e.reason, str(e)[:160])
                    self._a1_screen_shadow = False
                if pdc_wr:
                    TT._count(self, f"place_depth_cap.shadow.candidate.{e.reason}")
                errors.append(f"candidate {int(k)}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.grasps.clear()
                OF.drop_records(self)
        if self.start_region_exhausted():
            raise TemplateExhausted("target_placement", f"start region exhausted: no candidate left after {self.counters.get('travel_dropped', 0)} deterministic "
                                                        f"init -> manipulation travel failures ({'; '.join(errors[-3:])})")
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def _segment_draw(self, rng: np.random.Generator, d: Draft, k: int, t_pose: G.Pose, grasp: GraspRecord, manip: Stance, cam_init: G.Pose, pname: str) -> G.Pose:
        """The draw's checks of the segment after its grasp, with the grasp's arm: §4.6 step 2 (the partner in that arm's band), step 4
        (frame), the place in the manipulation image and the place screening. Records the stance and the grasp; returns the partner pose."""
        self._place_screen_reject = False  # i7-nextto: set when the place screening's exact IK rejects (PLACE_PARTNER_RETRY)
        # §4.6 step 2: partner on the attempt's side (A1) at a gap of the start region's interval
        frame_score = None
        from ispatialgym.sampling import campaign_runtime
        frame_recovery = self.relation == "inside" and campaign_runtime.inside_frame_recovery()
        if (os.environ.get("ISPATIALGYM_FRAME_RANK", "") == "1" or frame_recovery) and self.relation != "touching":
            # oracle-fix 2026-11-07 (owner prompt E 3): the partner draw ranked by the place / pre-place eef box margin in this candidate's image
            cam_m = self.camera_pose_at(manip)

            def frame_score(pp, _cam=cam_m):
                place_g = IK.grasp_pose(self.goal_pose(t_pose, pp), grasp.eef_in_object)
                return min(self.ikc.eef_frame_margin(place_g, _cam), self.pre_place_frame_margin(self.pre_place_of(place_g), _cam))
            # goal_pose reads the sidecar's manipulation stance (abv2_rank 2026-11-07: KeyError 'manipulation' in 21 / 21 iors_small runs): it is
            # registered here, before the partner draw, instead of after it (the caller pops it again when this candidate is rejected)
            self.sidecar.add_stance(manip)
        from ispatialgym.sampling.v2 import map_f as MF

        if MF.release_on() and self.relation in ("nextto", "under") and MF.is_view(self.band_of_arm(grasp.arm if getattr(grasp, "arm", None) else "right")):
            # of-v2maps 2026-11-07 (owner prompt I item 6, opt-in ISPATIALGYM_MAP_RELEASE=1): the valid partner poses ranked first by whether
            # the map holds an entry at the place eef pose (map_sampling.release_candidates), then by the frame margin when FRAME_RANK is on
            view = self.band_of_arm(grasp.arm if getattr(grasp, "arm", None) else "right")
            fs0 = frame_score
            self.sidecar.add_stance(manip)

            def frame_score(pp, _fs=fs0, _view=view):
                place_g = IK.grasp_pose(self.goal_pose(t_pose, pp), grasp.eef_in_object)
                T = np.eye(4)
                T[:3, :3] = C.quat_to_mat(G.t(place_g[1])).double().numpy()
                T[:3, 3] = G.t(place_g[0]).double().numpy()
                n = MF.release_count(_view, manip, T, grasp.grasp_name, float(getattr(self.world, "floor_z", 0.0)), float(self.top.z))
                self.counters["map_release." + ("none" if n is None else ("hit" if n else "miss"))] = self.counters.get(
                    "map_release." + ("none" if n is None else ("hit" if n else "miss")), 0) + 1
                return (1e4 if n else 0.0) + (float(_fs(pp)) if _fs is not None else 0.0)
        with phase_timer(self, "cand_partner"):  # i7-nextto: measurement only
            p_pose = self.place_partner(rng, t_pose, manip, cam_init, frame_score=frame_score)
        if frame_score is not None and isinstance(self.partner_draw, dict) and self.partner_draw.get("frame_rank"):
            d.summary.setdefault("frame_rank", []).append(self.partner_draw["frame_rank"])
        if self.edge_aligned():
            d.summary["edge_alignment"] = {
                "edge_yaw_deg": round(math.degrees(math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))), 1),
                "target_inhand_yaw_deg": round(math.degrees(self.edge_yaw_delta(self.hull["target"], t_pose[1], manip)), 1),
                "partner_yaw_deg": round(math.degrees(G.yaw_of(p_pose[1])), 1),
                "partner_residual_deg": round(math.degrees(self.edge_yaw_delta(self.hull["partner"], p_pose[1], manip)), 2),
            }
        if self.widest_faces():
            d.summary["widest_faces"] = self.widest_faces_facts(t_pose, p_pose, manip)
        if self.init_align_support():
            d.summary["init_align_support"] = {
                "edge_yaw_deg": round(math.degrees(math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))), 1),
                "target_residual_deg": round(math.degrees(self.support_yaw_delta("target", t_pose[1], manip)), 2),
                "partner_residual_deg": round(math.degrees(self.support_yaw_delta("partner", p_pose[1], manip)), 2),
                "axis": {role: self.support_axis_angle(role, pose[1])[1] for role, pose in (("target", t_pose), ("partner", p_pose))},
            }
        pair = self.pair_goal_align()
        if pair is not None:
            d.summary["pair_goal_align"] = dict({k: v for k, v in pair.items() if k != "front"}, init=self.pair_facts(t_pose, p_pose, manip))
        # §4.6 step 4: every object fully in frame (5 %) in the init image
        for role, pose in (("target", t_pose), ("partner", p_pose)):
            if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull[role], pose)):
                raise Reject("not_in_frame", f"{role} not in the init image (5 %)")
        grasp.place_on = pname
        self.sidecar.add_stance(manip)
        self.sidecar.add_grasp(grasp)  # (popped again by the caller when this candidate is rejected)
        if OF.record(self, grasp):  # i7-plateqtf2: the overhang fallback's push (contacts.push_pick) and grasps[0].pick_overhang
            d.summary["pick_overhang"] = _jsonable(OF.summary(grasp))
        # the place / pre-place of this candidate must project into its manipulation image (owner 1b: the FOV
        # monitor arms from the pre-pose): checked here, before the settle, so the next candidate is tried instead of
        # the attempt dying in the goal check (inside: release rim + 4-5 cm and the pre-place 10 cm above it)
        if self.relation != "touching":
            cam_manip = self.camera_pose_at(manip)
            goal_guess = self.goal_pose(t_pose, p_pose)
            place_guess = IK.grasp_pose(goal_guess, grasp.eef_in_object)
            for what, pose in (("place", place_guess), ("pre-place", self.pre_place_of(place_guess))):
                # i7-plateqtf2 (owner 2026-11-08 02:35 item 3): the place (a contact pose) keeps the eef box; the pre-place is judged as the
                # program check judges it (the fingertip point, QA's 5 % + 2 cm slack: pre_place_frame_ok)
                if not (self.ikc.eef_in_frame(pose, cam_manip) if what == "place" else self.pre_place_frame_ok(pose, cam_manip)):
                    # of-v2maps 2026-11-06: opt-in ISPATIALGYM_FRAME_DIAG (the projected box / fingers, a rendered image for the first few)
                    TT.frame_diag(self, manip, grasp.arm, pose, what, obj=self.obj("target"), obj_pose=goal_guess, cam_pose=cam_manip,
                                  tag=f"candidate {int(k)} grasp {getattr(grasp, 'grasp_name', '')}")
                    rule = "eef pose" if what == "place" or pre_place_frame_rule() == "box" else "fingertip (+2 cm slack)"
                    raise Reject("gripper_not_in_frame", f"{what} {rule} outside the manipulation image (5 %) for candidate {int(k)} "
                                                         f"(trunk {manip.translate}, {manip.tilt:.2f})")
        # (owner 2026-09-18 01:33, (a)) reach screening of the place at the draw: the pre-place -> place descent with the
        # object attached must pass the exact IK (margin + reach quality) from THIS candidate's stance - else the
        # next candidate is tried instead of the attempt dying in the goal check 20-60 s later (touching re-samples
        # 0917-235350 .. 0918-011947: 5/8 attempts lost to ik_joint_limit / witness residual at the place descent)
        if bool(self.gen("reach_screen_place", True)):
            obj_at_place = self.place_pose_for(t_pose, p_pose, manip) if self.relation == "touching" else self.goal_pose(t_pose, p_pose)
            place_guess = IK.grasp_pose(obj_at_place, grasp.eef_in_object)
            ign_place = self.support_objects() + ([self.obj("partner")] if self.relation != "inside" else [])
            # touching (owner 2026-09-18 16:32): the place is made from the second base pose the box is carried to
            st_place = (self.push_stance_for(manip, obj_at_place) if self.relation == "touching" else None) or manip
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object), phase_timer(self, "cand_place_ik"):
                res_place = self.ikc.exact([self.pre_place_of(place_guess), place_guess], st_place, grasp.arm, ignore_for_last=ign_place,
                                           scale_key=self.choice.models["target"].scale, cache=False, approached=place_approached(self))
            if not res_place.ok:
                self._place_screen_reject = True
                try:  # i7-nextto (facts only): where the failed place lies relative to the target's init pose and the stance
                    dxy = G.t(obj_at_place[0])[:2] - G.t(t_pose[0])[:2]
                    p_dep, p_lat = st_place.local(obj_at_place[0])
                    log.info("PLACE_SCREEN_FAIL candidate %d arm %s grasp %s code %s: place %.3f m from the init (depth %+.3f lat %+.3f in the stance), "
                             "partner gap %s yaw %s, release sweep %s", int(k), grasp.arm, getattr(grasp, "grasp_name", None), res_place.code,
                             float(th.linalg.norm(dxy)), float(p_dep), float(p_lat), (self.partner_draw or {}).get("gap_m"), (self.partner_draw or {}).get("yaw_deg"),
                             (self.release_sweep or {}).get("remedy"))
                except Exception:  # noqa: BLE001 - a diagnostic never fails the draw
                    pass
                raise Reject(res_place.code or "place_unreachable", f"place screening at the draw (candidate {int(k)}, trunk {manip.translate}, "
                                                                    f"{manip.tilt:.2f}, from the {st_place.name} stance): {res_place.reason}")
            if self.relation == "touching" and os.environ.get("ISPATIALGYM_TOUCHING_PUSH_SCREEN", "") == "1" and self.shelf_start is None:
                # of-v2maps 2026-11-07 (owner prompt F, opt-in): the push screened at the draw too, from the goal check's push-stance laterals
                # (touching_push_stance's order), so a candidate whose push no class reaches gives way to the next candidate instead of the attempt
                # dying in the goal check (weak1_v2 touching_01: 11 / 16 goal checks 'no push class passes from any push stance lateral',
                # touching_02 66 / 71; the map view's foci reach deeper than the legacy band's)
                why = self.push_screen_why(manip, obj_at_place, p_pose)
                self.counters["push_screen.checked"] = self.counters.get("push_screen.checked", 0) + 1
                if why is not None:
                    self.counters["push_screen.rejected"] = self.counters.get("push_screen.rejected", 0) + 1
                    raise Reject("contact_unreachable", f"push screening at the draw (candidate {int(k)}, trunk {manip.translate}, {manip.tilt:.2f}): {why}")
        if self.relation == "touching" and touch_lift_screen_mode() != "off":
            self._touch_lift_screen(manip, grasp, t_pose, p_pose, int(k))
        if self.relation == "touching" and touch_a1_screen_mode() != "off":
            self._touch_a1_screen(manip, grasp, t_pose, p_pose, int(k))
        if self.relation == "touching" and push_tip_screen_mode() != "off":
            self._touch_tip_screen(manip, grasp, t_pose, p_pose, int(k))
        return p_pose

    def _touch_tip_screen(self, manip: Stance, grasp: GraspRecord, t_pose: G.Pose, p_pose: G.Pose, k: int) -> None:
        """:data:`PUSH_TIP_SCREEN_ENV` (i7-t04push): ``Reject("contact_unreachable")`` at the draw when the goal check's push stance search is
        certain to fail - (a) no push class it would try clears the target's hull (:meth:`push_overlap_screen_why`, geometry), or (b) none of
        its A1 push stances (:meth:`push_stance_candidates`, the first ``push_stance_max_exact``: the same call and inputs) has both a push
        chain (:meth:`push_chain_feasible`: exact IK + retreat, capped under :data:`PUSH_TIP_CAP_ENV`) and the box's place
        (:meth:`place_infeasible_why`) - the goal check's own acceptance; shadow counts. Not screened (``push_tip_screen.uncertain``): the
        shelf layout / tip release, outside the push height sweep, with ISPATIALGYM_HAND_OVERLAP_RANK, and (b) without an A1 stance. Kit
        ``push_tip_screen.rejected_overlap`` / ``.rejected_exact`` / ``.exact_stances``."""
        if self.shelf_start is not None or self.shelf_tip() or not self.push_height_sweep_on() or hand_overlap_rank_on():
            TT._count(self, "push_tip_screen.uncertain")
            return
        t_place = self.place_pose_for(t_pose, p_pose, manip)
        TT._count(self, "push_tip_screen.checked")
        why = self.push_overlap_screen_why(manip, t_place, p_pose)
        kind = "overlap"
        if why is None and TT.contact_maps_on():
            laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
            rows, _f = self.push_stance_candidates(manip, t_place, laterals, p_pose=p_pose)
            rows = rows[:int(self.gen("push_stance_max_exact", PUSH_STANCE_MAX_EXACT))]
            if not rows:
                TT._count(self, "push_tip_screen.uncertain")
            else:
                whys = []
                for lat, cand, _x in rows:
                    probe = cand if cand is not None else manip
                    TT._count(self, "push_tip_screen.exact_stances")
                    self._push_chain_fails = []
                    push_kind = self.push_chain_feasible(probe, t_place, p_pose)
                    place_why = self.place_infeasible_why(probe, t_place, grasp) if push_kind is not None else None
                    if push_kind is not None and place_why is None:
                        why = None
                        break
                    so = getattr(probe, "stand_off", None)
                    whys.append(f"{lat:+.2f}{f' s{float(so):.2f}' if so is not None else ''}: " + (f"place: {place_why}" if push_kind else f"no class [{len(self._push_chain_fails)} cands]"))
                else:
                    why, kind = "no A1 push stance has a push chain and the place (exact): " + "; ".join(whys), "exact"
        if why is None:
            TT._count(self, "push_tip_screen.passed")
            return
        TT._count(self, f"push_tip_screen.rejected_{kind}" if push_tip_screen_mode() == "on" else f"push_tip_screen.shadow_would_reject_{kind}")
        msg = f"push tip screen at the draw (candidate {k}, trunk {manip.translate}, {manip.tilt:.2f}): {why}"
        if push_tip_screen_mode() == "shadow":
            TT._count(self, "push_tip_screen.shadow_would_reject")
            log.info("PUSH_TIP_SCREEN shadow would reject: %s", msg[:400])
            return
        TT._count(self, "push_tip_screen.rejected")
        raise Reject("contact_unreachable", msg)

    def push_overlap_screen_why(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Optional[str]:
        """None when one push class of the goal check's stream (:meth:`touching_push_candidates` under the sweep, the open classes first
        under :meth:`push_gripper_open_on`, the pair's ``push_max_height_m``, the tip cap when :data:`PUSH_TIP_CAP_ENV` is on) clears the
        target's hull at its contact and at the push goal (:meth:`push_hand_overlap`, the goal check's first test); else why (the overlaps
        counted). The push along ``st``'s edge, the box at ``t_place_pose``, the partner at ``p_pose``; geometry only."""
        tangent = st.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])
        p_init, _b = self.touching_points(t_place_pose, p_pose, st)
        hull_w = self.world.hull_at(self.hull["target"], t_place_pose)
        t_face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        dist = self.push_distance(float((p_hull[:, :2] @ tangent).min() - (hull_w[:, :2] @ tangent).max()))
        line_xy = self.push_line_xy(t_place_pose)
        cands = self.touching_push_candidates(st, p_init, t_face_z, direction, t_hull_w=hull_w, line_xy=line_xy)
        if self.push_gripper_open_on():
            import itertools

            cands = itertools.chain(self.open_push_candidates(st, hull_w, t_face_z, direction, line_xy=line_xy), cands)
        pair = self.pair_goal_align()
        h_max = float(pair["push_max_height_m"]) if (pair is not None and pair["push_max_height_m"] is not None) else None
        cap = None
        if push_tip_cap_mode() == "on":
            cap = float(CPZ.push_tip_height_cap(hull_w, t_face_z[0], direction, line_xy=line_xy,
                                                ratio=float(TT.phase2_gen(self, "push_tip_height_ratio", CPZ.PUSH_TIP_HEIGHT_RATIO)))["cap_m"])
        import collections

        why: "collections.Counter[str]" = collections.Counter()
        n = 0
        for _kind, init, extras in cands:
            if h_max is not None and float(extras.get("fingertip_z") or G.t(init[0])[2]) - float(t_face_z[0]) > h_max + 1e-9:
                continue
            h = extras.get("contact_height_m")
            if cap is not None and h is not None and float(h) > cap + 1e-6:
                continue
            n += 1
            goal = (G.t(init[0]) + direction * float(dist), G.t(init[1]))
            ov = self.push_hand_overlap(init, goal, t_place_pose, direction, dist, against="hull")
            if ov is None:
                return None
            why[str(ov).split(" -")[0][:40]] += 1
        return f"no push class of {n}{f' at or below the tip cap {100 * cap:.1f} cm' if cap is not None else ''} clears the target's hull (" + \
            ", ".join(f"{c}x {w}" for w, c in why.most_common(4)) + ")"

    def _touch_a1_screen(self, manip: Stance, grasp: GraspRecord, t_pose: G.Pose, p_pose: G.Pose, k: int) -> None:
        """Follow-up :data:`TOUCH_A1_SCREEN_ENV`: the goal check's push stance search (:meth:`touching_push_stance`) at the draw, as far as it
        is certain - its plan is A1's map-feasible, collision-free stances (the same :meth:`push_stance_candidates` call, first
        ``push_stance_max_exact``), and a stance passes only when the box's place from it passes (:meth:`place_infeasible_why`). No stance
        placing = the goal check's contact_unreachable, made here before the draft (~1-2 s per stance instead of ~130 s). No A1 stance (the
        goal check then falls back to the laterals at the manipulation stand-off), the shelf layout or the tip release: not screened
        (``touch_a1_screen.uncertain``). Kit ``touch_a1_screen.checked`` / ``.passed`` / ``.rejected`` / ``.shadow_would_reject``,
        ``.place_checks``."""
        if self.shelf_start is not None or self.shelf_tip() or not TT.contact_maps_on():
            return
        t_place = self.place_pose_for(t_pose, p_pose, manip)
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        rows, _f = self.push_stance_candidates(manip, t_place, laterals, p_pose=p_pose)
        rows = rows[:int(self.gen("push_stance_max_exact", PUSH_STANCE_MAX_EXACT))]
        if not rows:
            TT._count(self, "touch_a1_screen.uncertain")
            return
        TT._count(self, "touch_a1_screen.checked")
        whys = []
        for lat, cand, _x in rows:
            probe = cand if cand is not None else manip
            TT._count(self, "touch_a1_screen.place_checks")
            why = self.place_infeasible_why(probe, t_place, grasp)
            if why is None:
                TT._count(self, "touch_a1_screen.passed")
                return
            so = getattr(probe, "stand_off", None)
            whys.append(f"{lat:+.2f}{f' s{float(so):.2f}' if so is not None else ''}: {why}")
        msg = f"no A1 push stance places the box (screened at the draw, candidate {k}, trunk {manip.translate}, {manip.tilt:.2f}): {'; '.join(whys)}"
        if touch_a1_screen_mode() == "shadow":
            TT._count(self, "touch_a1_screen.shadow_would_reject")
            self._a1_screen_shadow = True
            log.info("TOUCH_A1_SCREEN shadow would reject candidate %d: %s", k, msg[:400])
            return
        TT._count(self, "touch_a1_screen.rejected")
        raise Reject("contact_unreachable", msg)

    def _touch_lift_screen(self, manip: Stance, grasp: GraspRecord, t_pose: G.Pose, p_pose: G.Pose, k: int) -> None:
        """Follow-up (a) (:data:`TOUCH_LIFT_SCREEN_ENV`): the goal check's carried-drive lift test (:meth:`_record_push_stance`: the box
        ``carry_height`` above the grasp, exact IK at the manipulation stance with the box attached) made at the draw - only where the goal
        check is certain to make it at the partner's height: the carry rule "auto" or "partner", the push stance search will try the A1
        candidates (:meth:`push_stance_candidates` - the same call, the same inputs), every one is a real second base pose (no manipulation
        stance entry, which records no drive) and every drive passes within ``carry_clearance_m`` of the partner (:meth:`drive_partner_gap`:
        the auto rule raises the box over the partner). Then a failing lift fails the attempt there as it would ~150 s later. Shadow: the
        verdict counted (``touch_lift_screen.shadow_would_reject``) and the candidate kept. Kit ``touch_lift_screen.checked`` / ``.uncertain``
        / ``.passed`` / ``.rejected``."""
        rule = self.carry_rule()
        if self.shelf_start is not None or rule not in ("auto", "partner") or not TT.contact_maps_on():
            return
        t_place = self.place_pose_for(t_pose, p_pose, manip)
        if rule == "auto":
            laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
            rows, _f = self.push_stance_candidates(manip, t_place, laterals, p_pose=p_pose)
            rows = rows[:int(self.gen("push_stance_max_exact", PUSH_STANCE_MAX_EXACT))]
            clear = float(self.gen("carry_clearance_m", CARRY_CLEARANCE_M))
            if not rows or any(c is None for _l, c, _x in rows) or any(self.drive_partner_gap(t_pose, p_pose, manip, c) >= clear for _l, c, _x in rows):
                TT._count(self, "touch_lift_screen.uncertain")
                return
        TT._count(self, "touch_lift_screen.checked")
        saved = grasp.carry_height
        try:
            grasp.carry_height = self.carry_height_over(True)
            lift_end = PC.lift_end_pose(grasp)
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                r_lift = self.ikc.exact([lift_end], manip, grasp.arm, ignore_for_last=self.support_objects(), cache=False)
            height = float(grasp.carry_height)
        finally:
            grasp.carry_height = saved
        if r_lift.ok and r_lift.q is not None:
            TT._count(self, "touch_lift_screen.passed")
            return
        if touch_lift_screen_mode() == "shadow":
            TT._count(self, "touch_lift_screen.shadow_would_reject")
            self._lift_screen_shadow = True
            log.info("TOUCH_LIFT_SCREEN shadow would reject candidate %d: lift %.2f m: %s", k, height, r_lift.reason)
            return
        TT._count(self, "touch_lift_screen.rejected")
        raise Reject("lift_unreachable", f"lift ({height:.2f} m above the grasp) has no configuration at the {manip.name} stance for the carried drive to "
                                         f"the push stance (screened at the draw, candidate {k}): {r_lift.reason}")

    def push_screen_why(self, manip: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Optional[str]:
        """``ISPATIALGYM_TOUCHING_PUSH_SCREEN`` (of-v2maps 2026-11-07): None when a push class passes (:meth:`push_chain_feasible`, the chain +
        retreat clear pose) from the manipulation stance or one of the goal check's push-stance laterals (``push_contact_laterals_m``, default
        -0.12 / -0.16 / -0.20 / -0.08) with the box at its predicted place and the partner at its drawn pose; else the laterals' summary."""
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        tried = []
        for lat in laterals:
            cand = self.push_stance_for(manip, t_place_pose, lateral=lat)
            probe = cand if cand is not None else manip
            self._push_chain_fails = []
            self._push_chain_height = None
            kind = self.push_chain_feasible(probe, t_place_pose, p_pose)
            if kind is not None:
                return None
            tried.append(f"{lat:+.2f}{' (manipulation)' if cand is None else ''}: no class [{len(self._push_chain_fails)} cands]")
        return "no push class from any push stance lateral (" + "; ".join(tried) + ")"

    def _pose_target(self, role: str, pose: G.Pose) -> Dict[str, Any]:
        tgt = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        mc = self.choice.models[role]
        eq = self.gp.orientation_equivalences(role, mc.category, mc.model)
        if eq:
            tgt["orientation_equivalences"] = [[float(v) for v in q] for q in eq]
        tols = self.gen("tolerances") or {}
        if role == "target" and "orientation_metric" in tols and tols["orientation_metric"] is None:
            tgt["orientation_metric"] = None
            tgt["orientation_tolerance"] = None
        return tgt

    def gap_range(self) -> Tuple[float, float]:
        """§4.6 step 2: ontop / under / inside / touching g ∈ [0.03, D_p]; nextto g ∈ [g_NT + 0.03, D_p] (D_p from Phase 1c)."""
        info = (self.init_stance.extras.get("margins") or {})
        D_p = float(info.get("D_p", self.lateral_margins()["D_p_max"]))
        g_lo = self.gap_min
        if self.relation == "nextto":
            g_lo = float(self.pair["g_NT"]) + self.nextto_gap_extra
            if D_p < g_lo:
                raise Reject("pair_gap", f"D_p {D_p:.2f} < g_NT + {self.nextto_gap_extra:.2f} = {g_lo:.2f}")
        return g_lo, max(g_lo, D_p)

    def place_gap(self) -> float:
        """AABB gap between target and partner at the *place* position: nextto ``g_NT - 0.01`` (the predicate holds below
        g_NT); touching = the same nextto position — the oracle program places there and the push closes the gap to
        contact (``push_distance``) —, or ``generation.touching_place_gap_m`` when set (owner 2026-10-12 item 4: set down farther
        from the partner so that the opening fingers clear it at the release; the push covers the longer distance)."""
        if self.relation == "touching" and self.gen("touching_place_gap_m") is not None:
            return max(0.0, float(self.gen("touching_place_gap_m")))
        return max(0.0, float(self.pair.get("g_NT", 0.03)) - 0.01)

    def push_margin(self) -> float:
        """Touching: how far past the measured gap the recorded push reaches - the place position tolerance (the executed
        place may leave the target that much farther from the partner: oracle-B run 0916-202403, boxes 4.1 cm apart at a
        2.7 cm recorded push) plus 5 mm of pressing. The oracle pushes the residual it measures; the recorded arc is its cap."""
        return float(self.tol("position", 0.03)) + 0.005

    def push_distance(self, gap_at_place: Optional[float] = None) -> float:
        """Touching: the recorded push spans the gap between the target's near face at the place and the partner's face —
        measured from the hulls when given, else the nominal place gap — plus ``push_margin`` (``touching_contacts``:
        ``push_goal = push_init + direction * push_distance``)."""
        gap = self.place_gap() if gap_at_place is None else max(0.0, float(gap_at_place))
        return gap + self.push_margin()

    def draw_partner_pose(self, rng: np.random.Generator, side: float, t_pose: G.Pose, manip: Stance, cam: G.Pose, yaw: float) -> Tuple[Optional[G.Pose], str]:
        """One A1 partner draw at world ``yaw`` on ``side`` of the placed target: the start region's gap interval re-evaluated at the drawn target
        pose and this orientation (``tabletop.partner_gap_masks`` with :meth:`band_rule` at the band of the arm that makes the segment), a gap
        drawn uniformly from it, the pose checked as §4.6 step 2 checks it. ``(pose, "")`` or ``(None, why)``; the draw's facts in
        ``partner_draw``. :meth:`pair_goal_align`: the partner's footprint rectangle level with the target's along the stance's depth axis
        (its origin at the target's origin depth otherwise)."""
        p_quat = G.qmul(G.quat_from_yaw(yaw), G.t(self.rest["partner"]))
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        depth, t_c = manip.local(t_pose[0])
        if self.pair_goal_align() is not None:
            depth = self.pair_level_depth("partner", p_quat, min_area_rect(t_fp)[3], manip)
        reach, band = self.band_rule(manip, float(manip.world_to_base(t_pose[0][:2])[0]), (manip.working_arms or self.segment_arms)[:1])
        gs = TT.gap_grid(*self.gap_range(), step=0.5 * TT.START_REGION_GAP_STEP)
        masks = TT.partner_gap_masks(self, side, t_fp, depth, self.hull["partner"], p_quat, manip, gs, top=self.top, cam=cam, margin=self.margin_support,
                                     m_end=self.table_end_margin, band=band, reach=reach, t_origin=t_c, place_gap=self.place_gap())
        flat = getattr(self, "flat_top", None)
        if flat is not None and len(gs):
            masks["flat"] = self.flat_gap_mask(side, t_fp, depth, p_quat, manip, gs)
        run = TT.longest_run(gs, np.logical_and.reduce([np.ones(len(gs), dtype=bool)] + list(masks.values())))
        if run is None:
            return None, TT.gap_masks_why(gs, masks)
        g = float(rng.uniform(run[0], run[1]))
        pose = TT.partner_pose_at(self, side, t_fp, depth, self.hull["partner"], p_quat, manip, g, top=self.top)
        why = TT.partner_pose_why(self, t_fp, self.hull["partner"], pose, top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min)
        if not why and flat is not None:
            why = self.flat_pose_why(pose)
        if why:
            return None, f"gap {g:.3f}: {why}"
        self.partner_draw = {"side": TT.side_name(side), "gap_m": round(g, 4), "gap_interval_m": [round(run[0], 4), round(run[1], 4)],
                             "yaw_deg": round(math.degrees(yaw), 1), "cell_interval_m": (manip.extras.get(TT.START_REGION) or {}).get(TT.side_name(side))}
        if flat is not None:
            p_fp = self.world.footprint(self.hull["partner"], pose)
            zr = flat.heights(p_fp)
            self.partner_draw["flat_top"] = {"clearance_m": round(flat.clearance(p_fp, 0.10), 4),
                                             "seen_height_range_mm": None if zr is None else [round(1000 * (v - flat.z_ref), 2) for v in zr]}
        return pose, ""

    def draw_meta_partner_pose(self, rng: np.random.Generator, side: float, t_pose: G.Pose, manip: Stance, cam: G.Pose) -> G.Pose:
        """A5 (owner 2026-10-12): the partner centred on one of the candidate's meta-link spots for ``side`` (drawn uniformly) at a random yaw,
        checked at the drawn target pose - on ``side`` of it at a lateral gap of the gap range, the reach rule with the band of the segment's
        arm (:meth:`band_rule`), at least ``footprint_gap_min_m`` from the target, fully in the init image (the spot is on the top,
        :meth:`find_meta_spots`). Up to ``PARTNER_YAW_DRAWS`` yaws per spot (:meth:`init_align_support`: its four support-aligned ones in
        random order); ``Reject("partner_placement")`` when none passes."""
        g_lo, g_hi = self.gap_range()
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        t_lo, t_hi = footprint_laterals(t_fp - manip.base_xy, manip.tangent)
        _, t_c = manip.local(t_pose[0])
        reach, band = self.band_rule(manip, float(manip.world_to_base(t_pose[0][:2])[0]), (manip.working_arms or self.segment_arms)[:1])
        misses: List[str] = []
        spots = (manip.extras.get(META_SPOTS) or {}).get(TT.side_name(side), [])
        aligned = self.support_aligned_yaws("partner", manip) if self.init_align_support() else None
        for k in rng.permutation(len(spots)):
            mid, xy = self.meta_spots[spots[int(k)]]
            yaws = [aligned[int(j)] for j in rng.permutation(4)] if aligned else (float(rng.uniform(-math.pi, math.pi)) for _ in range(PARTNER_YAW_DRAWS))
            for yaw in yaws:
                pose = TT.pose_centred_on(self, self.hull["partner"], G.qmul(G.quat_from_yaw(yaw), G.t(self.rest["partner"])), xy, top=self.top)
                p_fp = self.world.footprint(self.hull["partner"], pose)
                p_lo, p_hi = footprint_laterals(p_fp - manip.base_xy, manip.tangent)
                g = p_lo - t_hi if side > 0 else t_lo - p_hi
                if not g_lo <= g <= g_hi:
                    why = f"lateral gap {g:.3f} outside [{g_lo:.2f}, {g_hi:.2f}]"
                elif reach is not None and not TT.lateral_gap_masks([g], side, (t_lo, t_hi), (0.0, p_hi - p_lo), band=band, reach=reach, t_origin=t_c,
                                                                    place_gap=self.place_gap())["reach"][0]:
                    why = f"no band lateral of the {(manip.working_arms or self.segment_arms)[0]} arm for the partner"
                elif TT.polygon_gap(t_fp, p_fp) < self.gap_min:
                    why = f"partner closer than {self.gap_min:.2f} m to the target"
                elif cam is not None and not TT.in_frame(self, cam, TT.hull_corners_world(self.hull["partner"], pose)):
                    why = "partner not in the init image"
                else:
                    why = ""
                if why:
                    misses.append(f"spot {mid} yaw {math.degrees(yaw):+.0f}: {why}")
                    continue
                self.partner_draw = {"side": TT.side_name(side), "gap_m": round(g, 4), "meta_link": mid, "yaw_deg": round(math.degrees(yaw), 1),
                                     "cell_interval_m": (manip.extras.get(TT.START_REGION) or {}).get(TT.side_name(side))}
                return pose
        raise Reject("partner_placement", f"no meta-link spot of {len(spots)} takes the partner on the {TT.side_name(side)} of the target ({'; '.join(misses[-3:])})")

    def stance_frame_aim(self, cand: Stance) -> float:
        """oracle-fix 2026-11-07 (owner prompt E 2, ``ISPATIALGYM_STANCE_FRAME_AIM``): the frame margin (px, ``IKChecker.eef_frame_margin``) of
        an upright eef box at the EXPECTED place - the partner's centre at the mid gap of the candidate's start-region interval on the attempt's
        side (focus + left x side x (mid gap + both half extents)), at the partner's top + the target's height - in the candidate's head
        image; -inf without a focus. A ranking proxy (the grasp's place orientation is not known yet), not a rule."""
        plan = getattr(self, '_campaign_pair_active', None)
        if plan is not None:
            aim = (G.t(plan.release[0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            return self.ikc.eef_frame_margin(aim, self.camera_pose_at(cand))
        if cand.focus is None:
            return float("-inf")
        side = self.region_side if self.region_side is not None else self.template_sides()[0]
        reg = (cand.extras.get(TT.START_REGION) or {}).get(TT.side_name(side))
        gap = 0.5 * (float(reg[0]) + float(reg[1])) if reg else 0.05
        r_t = TT.half_extent(self.hull["target"], self.rest["target"])
        r_p = TT.half_extent(self.hull["partner"], self.rest["partner"])
        xy = G.t(cand.focus)[:2] + G.t(cand.left) * float(side) * (gap + r_t + r_p)
        z = float(self.top.z) + TT.height_of(self.hull["partner"], self.rest["partner"]) + TT.height_of(self.hull["target"], self.rest["target"])
        aim = (th.tensor([float(xy[0]), float(xy[1]), z]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        return self.ikc.eef_frame_margin(aim, self.camera_pose_at(cand))

    def place_partner(self, rng: np.random.Generator, t_pose: G.Pose, manip: Stance, cam: G.Pose, frame_score=None) -> G.Pose:
        """§4.6 step 2 with the A1 start region (owner 2026-10-12): the partner on the attempt's side (``region_side``, fixed before the target
        was placed; touching the robot's left) at a gap of the start region's interval re-evaluated at the drawn target pose
        (:meth:`draw_partner_pose`: on the top, both footprints ``table_end_margin_m`` from the edge's ends, in the init image, and the reach
        rule cut to the band of the manipulation trunk - §4.1 / principle 3, the place pose of §6.11 needs its goal in the band).
        ``PARTNER_YAW_DRAWS`` random partner yaws, then the two edge-aligned ones (:meth:`edge_partner_quats`) - with :meth:`init_align_support`
        the four edge-aligned yaws only, ``PARTNER_YAW_DRAWS`` tries; with :meth:`pair_goal_align` the one yaw of :meth:`pair_partner_yaw` at
        every try (touching too); ``Reject("partner_placement")`` when none admits a gap. The shelf
        layout keeps its partner leaning against the opposite board; with ``partner_on_meta_link`` (A5) it stands on a meta-link spot
        (:meth:`draw_meta_partner_pose`)."""
        self.partner_draw = None
        campaign_plan = getattr(self, "_campaign_pair_active", None)
        if campaign_plan is not None:
            pose = tuple(G.t(v) for v in campaign_plan.partner)
            why = TT.partner_pose_why(self, self.world.footprint(self.hull["target"], t_pose), self.hull["partner"], pose,
                                      top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min)
            if not why and getattr(self, "flat_top", None) is not None:
                why = self.flat_pose_why(pose)
            if why:
                raise Reject("campaign_partner_placement", why)
            self.partner_draw = {"source": "goal_plan", "group_id": campaign_plan.group_id}
            if self.relation == "touching":
                # Keep the native side, contact-map and possible push-stance
                # checks even though the partner pose was selected first.
                p_hull = self.world.hull_at(self.hull["partner"], pose)
                p_lat = (p_hull[:, :2] - manip.base_xy) @ manip.tangent
                if float(manip.local(t_pose[0])[1]) >= .5 * float(p_lat.min() + p_lat.max()):
                    raise Reject("touching_side", "fresh target start is on the wrong side of the fixed partner")
                t_place = self.place_pose_for(t_pose, pose, manip)
                if TT.contact_maps_on():
                    self.contact_maps_of_pair()
                    self.template_facts["left_contact_band"] = "touch_brace map"
                    gate = self.contact_map_gate(manip, t_place, pose)
                    if gate["ok"]:
                        self._brace_gate = gate
                    else:
                        available = self.push_map_any(manip, t_place, pose)
                        if not available["push_any"]:
                            raise Reject("contact_unreachable", "fixed touching pair has no native feasible push stance")
                        self._brace_gate = dict(gate, ok=bool(available["pair_any"]), a1=available,
                                                at="fixed pair native push stance candidates")
                else:
                    right = self.bands.band(manip.translate, manip.tilt, **TT.band_kw(self.bands, manip)) if getattr(self, "bands", None) is not None else None
                    left = left_contact_band_cells(self, (manip.translate, manip.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]), stance=manip)
                    if right is None or left is None:
                        self._brace_gate = {"ok": None, "why": "left contact band unavailable"}
                    else:
                        work, brace = self.touching_points(t_place, pose, manip)
                        self._brace_gate = {"ok": brace_placement_admissible(manip.world_to_base(work[:2]), manip.world_to_base(brace[:2]), right, left),
                                            "why": "fixed pair native contact bands"}
            self._campaign_latest_proposal = {"start_poses": {campaign_plan.target_name: t_pose, campaign_plan.partner_name: pose},
                                              "goal_poses": {campaign_plan.target_name: campaign_plan.goal},
                                              "base_pose2d": manip.base_pose2d, "group_id": campaign_plan.group_id}
            return pose
        if self.shelf_partner_pose is not None:
            self._brace_gate = {"ok": False, "why": "shelf (owner 2026-10-11): the side board holds the partner, no brace"}
            return self.shelf_partner_pose
        side = self.region_side if self.region_side is not None else self.template_sides()[0]
        misses: List[str] = []
        if self.meta_spots is not None:
            return self.draw_meta_partner_pose(rng, side, t_pose, manip, cam)
        if self.relation != "touching":
            # random yaws, then the two edge-aligned footprints the start region was computed with (a tight top admits little else);
            # init_align_support: the four edge-aligned yaws only, in random order, each tried with fresh gap draws; partner_aligned_to_target:
            # the yaws along the target's drawn pose instead (the partner follows the target - edge-aligned too when the target is);
            # pair_goal_align: the one yaw that puts the partner's front at the required turn from the target's, fresh gap draws
            if self.pair_goal_align() is not None:
                yaws = [self.pair_partner_yaw(t_pose)] * PARTNER_YAW_DRAWS
            elif self.partner_aligned_to_target():
                yaws = self.aligned_partner_yaws(t_pose, rng)
            elif self.init_align_support():
                aligned = self.support_aligned_yaws("partner", manip)
                yaws = [aligned[int(k)] for k in rng.permutation(4)] * (PARTNER_YAW_DRAWS // 4)
            else:
                yaw0 = self.edge_yaw_delta(self.hull["partner"], self.rest["partner"], manip)
                yaws = [float(rng.uniform(-math.pi, math.pi)) for _ in range(PARTNER_YAW_DRAWS)] + [yaw0, yaw0 + math.pi / 2]
            # oracle-fix 2026-11-07 (owner prompt E 3, opt-in ISPATIALGYM_FRAME_RANK=1 via ``frame_score``): up to FRAME_RANK_K valid partner
            # poses, the one whose place / pre-place eef box sits deepest inside the manipulation image (frame_score: px margin) is kept -
            # the same draws, the same rules, the frame rule itself unchanged (it still rejects below)
            ranked: List[Tuple[float, G.Pose, Any]] = []
            for yaw in yaws:
                p_pose, why = self.draw_partner_pose(rng, side, t_pose, manip, cam, yaw)
                if p_pose is not None:
                    if self.partner_aligned_to_target():
                        self.partner_draw["long_axes_deg"] = round(long_axes_angle_deg(self.world.footprint(self.hull["target"], t_pose),
                                                                                       self.world.footprint(self.hull["partner"], p_pose)), 2)
                    if frame_score is None:
                        return p_pose
                    ranked.append((float(frame_score(p_pose)), p_pose, dict(self.partner_draw or {})))
                    if len(ranked) >= FRAME_RANK_K:
                        break
                    continue
                misses.append(f"yaw {math.degrees(yaw):+.0f}: {why}")
            if ranked:
                best = max(range(len(ranked)), key=lambda i: ranked[i][0])
                score, p_pose, facts = ranked[best]
                self.partner_draw = dict(facts, frame_rank={"n": len(ranked), "margins_px": [round(r[0], 1) if math.isfinite(r[0]) else None for r in ranked],
                                                            "chosen": best, "first_in_frame": bool(ranked[0][0] >= 0.0)})
                self.counters["frame_rank.partner"] = self.counters.get("frame_rank.partner", 0) + 1
                if ranked[0][0] < 0.0 <= score:
                    self.counters["frame_rank.partner_rescued"] = self.counters.get("frame_rank.partner_rescued", 0) + 1
                return p_pose
            raise Reject("partner_placement", f"no partner yaw of {len(yaws)} admits a gap on the {TT.side_name(side)} of the target ({'; '.join(misses[-3:])})")
        # A1 with the brace optional (owner 2026-09-16 17:10, the axial rule): a partner position whose push work point (the
        # target's right face at the *place* position) lies in the right band and whose brace point (partner's far / left
        # face) lies in the left contact band of the same trunk is preferred and enables the brace; otherwise the last
        # drawn position is kept and the instance becomes push-only (``self._brace_gate`` tells ``touching_contacts``)
        maps = TT.contact_maps_on() and self.shelf_start is None  # i7-contact: the touch_push / touch_brace maps under v2only
        if maps:
            right_cells = left_cells = None
            self.contact_maps_of_pair()  # a miss rejects missing_map (MAP_REQUEST logged)
            self.template_facts["left_contact_band"] = "touch_brace map"
        else:
            right_cells = self.bands.band(manip.translate, manip.tilt, **TT.band_kw(self.bands, manip)) if getattr(self, "bands", None) is not None else None
            left_cells = left_contact_band_cells(self, (manip.translate, manip.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]), stance=manip)
            self.template_facts["left_contact_band"] = "unavailable" if left_cells is None else int(len(left_cells))
        faces_yaws = self.widest_faces_partner_yaws(t_pose, manip) if self.widest_faces() else None  # the two turns parallel to the target
        if self.pair_goal_align() is not None:
            faces_yaws = [self.pair_partner_yaw(t_pose)]  # the front at the required turn from the target's
        errors = []
        p_pose = None
        push_only: Optional[Tuple[G.Pose, Dict[str, Any], Dict[str, Any]]] = None  # owner A1: the first position whose push fits a push stance
        # edge alignment (owner 2026-09-18 13:50): the partner's footprint axes along the support edge - the two aligned
        # footprints (long side along / across the edge) alternate over the tries; without it a random yaw per try
        yaw0 = (self.edge_yaw_delta(self.hull["partner"], self.rest["partner"], manip) if self.edge_aligned() else
                self.support_yaw_delta("partner", self.rest["partner"], manip) if self.init_align_support() else None)
        for i in range(8):
            yaw = yaw0 + (i % 2) * (math.pi / 2) if yaw0 is not None else float(rng.uniform(-math.pi, math.pi))
            if faces_yaws:
                yaw = faces_yaws[i % len(faces_yaws)]
            pose, why = self.draw_partner_pose(rng, side, t_pose, manip, cam, yaw)
            if pose is None:
                misses.append(f"yaw {math.degrees(yaw):+.0f}: {why}")
                continue
            p_pose = pose
            if maps:
                t_place = self.place_pose_for(t_pose, p_pose, manip)
                g = self.contact_map_gate(manip, t_place, p_pose)
                if g["ok"]:
                    self._brace_gate = g
                    return p_pose
                # owner A1 (2026-11-08): the push happens from the push stance, which moves along AND toward / away from the edge - the
                # maps over those candidates decide (map lookups only)
                a1 = self.push_map_any(manip, t_place, p_pose)
                if a1["pair_any"]:
                    self._brace_gate = dict(g, ok=True, why=None, at="a push stance candidate (owner A1)", a1=a1)
                    TT._count(self, "touch_push.gate_at_push_stance")
                    return p_pose
                if a1["push_any"] and push_only is None:
                    push_only = (p_pose, dict(self.partner_draw or {}), a1)
                errors.append(f"{g['why']} (push stances: {a1['n']})")
                continue
            if left_cells is None or right_cells is None:
                self._brace_gate = {"ok": None, "why": "left contact band unavailable"}  # unknown: the two-arm IK decides
                return p_pose
            work, brace = self.touching_points(self.place_pose_for(t_pose, p_pose, manip), p_pose, manip)
            w_b, b_b = manip.world_to_base(work[:2]), manip.world_to_base(brace[:2])
            if brace_placement_admissible(w_b, b_b, right_cells, left_cells):
                self._brace_gate = {"ok": True, "work_base": [round(float(v), 3) for v in w_b], "brace_base": [round(float(v), 3) for v in b_b],
                                    "right_cells": int(len(right_cells)), "left_cells": int(len(left_cells))}
                return p_pose
            errors.append(f"work {[round(float(v), 2) for v in w_b]} {'in' if in_band(w_b, right_cells) else 'OFF'} right band / brace "
                          f"{[round(float(v), 2) for v in b_b]} {'in' if in_band(b_b, left_cells) else 'OFF'} left band")
        if p_pose is None:
            raise Reject("partner_placement", f"no partner yaw of 8 admits a gap on the {TT.side_name(side)} of the target ({'; '.join(misses[-3:])})")
        if maps:
            if push_only is None:
                # owner A1 (2026-11-08): the push fails the map from every push stance at every partner position - the attempt would grasp, lift
                # and place before dying in goal_check (i7_contact2: 107 / 150 attempts); rejected here instead
                TT._count(self, "touch_push.partner_rejected")
                raise Reject("contact_unreachable", f"touching partner draw: no push stance (laterals x stand-offs) has a feasible touch_push class at any "
                                                    f"of 8 partner positions (trunk ({manip.translate}, {manip.tilt:.2f}); e.g. {errors[-2:]})")
            p_pose, self.partner_draw, a1 = push_only
            self._brace_gate = {"ok": False, "maps": True, "a1": a1, "why": f"none of 8 partner positions has the push (touch_push map) and the brace (touch_brace map) "
                                                                            f"feasible together from any push stance at trunk ({manip.translate}, {manip.tilt:.2f}); e.g. {errors[-2:]}"}
            return p_pose
        self._brace_gate = {"ok": False, "why": f"none of 8 partner positions puts the push point in the right band ({len(right_cells)} cells) and the brace point "
                                                f"in the left contact band ({len(left_cells)} cells) at trunk ({manip.translate}, {manip.tilt:.2f}); e.g. {errors[-2:]}"}
        return p_pose

    # ------------------------------------------------------------------ edge alignment (owner 2026-09-18 13:50 / 14:28)

    def edge_aligned(self) -> bool:
        """Touching (owner 2026-09-18 13:50): the partner is spawned with its footprint axes along the support edge and the
        target's goal / place orientation is the edge-aligned yaw nearest its init yaw (an in-hand rotation <= 45 deg) -
        the boxes meet face to face; ``generation.edge_aligned`` (default on for touching)."""
        return self.relation == "touching" and bool(self.gen("edge_aligned", True))

    def init_align_support(self) -> bool:
        """Owner 2026-10-13 (nextto#01 / #03 / #04): ``generation.init_align_support`` (default false) - the target and the partner start
        with their long axes along the support top's edges: the target's init yaw is one of its :meth:`support_aligned_yaws` (no jitter),
        and so is the partner's (in place of the random yaws)."""
        return bool(self.gen("init_align_support", False))

    def model_long_axis(self, role: str) -> Optional[th.Tensor]:
        """``role``'s long axis in its object frame: :func:`obb_long_axis` of its model's ``misc/metadata.json`` (beside its USD), stretched by
        its scale; None without one."""
        import json
        import os

        obj = self.obj(role)
        scale = G.t(obj.scale).float().reshape(-1)
        key = (obj.name, tuple(round(float(v), 6) for v in scale.tolist()))
        cache = self.__dict__.setdefault("_long_axes", {})
        if key not in cache:
            usd, axis = getattr(obj, "usd_path", None), None
            if usd:
                try:
                    with open(os.path.join(os.path.dirname(os.path.dirname(usd)), "misc", "metadata.json")) as f:
                        axis = obb_long_axis(json.load(f))
                except (OSError, ValueError):
                    axis = None
            cache[key] = None if axis is None else axis * scale[:3]
        return cache[key]

    def support_axis_angle(self, role: str, quat) -> Tuple[float, str]:
        """World angle (rad) of the axis :meth:`init_align_support` lays along the support's edges for ``role`` at ``quat``, and its source:
        the model's long axis (:meth:`model_long_axis`, ``"long_axis"``) unless it stands more than 60 deg off the floor, else the
        footprint's minimum-area rectangle (:meth:`footprint_axis_angle`, ``"footprint"``). The rectangle can sit a few degrees off a tapered
        model's long axis (tablespoon/yxxfdj: 3.3 deg)."""
        axis = self.model_long_axis(role)
        if axis is not None:
            v = C.quat_to_mat(G.t(quat)).float() @ axis.float()
            if float(th.linalg.norm(v[:2])) >= 0.5 * float(th.linalg.norm(v)):
                return math.atan2(float(v[1]), float(v[0])), "long_axis"
        return self.footprint_axis_angle(self.hull[role], quat), "footprint"

    def support_yaw_delta(self, role: str, quat, manip: Stance) -> float:
        """Smallest world yaw (rad, in [-pi/4, pi/4]) that turns ``role``'s :meth:`support_axis_angle` at ``quat`` onto the stance's edge frame."""
        phi = math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))
        return (phi - self.support_axis_angle(role, quat)[0] + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def support_aligned_yaws(self, role: str, manip: Stance) -> List[float]:
        """:meth:`init_align_support`'s four world yaws (applied to ``role``'s rest orientation): its :meth:`support_axis_angle` along or across
        the stance's support edge."""
        yaw0 = self.support_yaw_delta(role, self.rest[role], manip)
        return [yaw0 + k * math.pi / 2 for k in range(4)]

    def footprint_axis_angle(self, hull: th.Tensor, quat) -> float:
        """World angle (rad, modulo pi/2) of the minimum-area bounding rectangle of the hull's footprint at ``quat``."""
        pts = self.world.footprint(hull, (th.zeros(3), G.t(quat)))
        best = None
        n = len(pts)
        for i in range(n):
            e = pts[(i + 1) % n] - pts[i]
            if float(th.linalg.norm(e)) < 1e-6:
                continue
            ang = math.atan2(float(e[1]), float(e[0]))
            c, s = math.cos(ang), math.sin(ang)
            u = pts[:, 0] * c + pts[:, 1] * s
            v = -pts[:, 0] * s + pts[:, 1] * c
            area = float((u.max() - u.min()) * (v.max() - v.min()))
            if best is None or area < best[0]:
                best = (area, ang)
        return 0.0 if best is None else best[1] % (math.pi / 2)

    def edge_yaw_delta(self, hull: th.Tensor, quat, manip: Stance) -> float:
        """Smallest world yaw (rad, in [-pi/4, pi/4]) that turns the footprint at ``quat`` onto the stance's edge frame."""
        phi = math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))
        theta = self.footprint_axis_angle(hull, quat)
        return (phi - theta + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def edge_aligned_quat(self, hull: th.Tensor, quat, manip: Stance) -> th.Tensor:
        """``quat`` turned about world z by :meth:`edge_yaw_delta`."""
        return G.qmul(G.quat_from_yaw(self.edge_yaw_delta(hull, quat, manip)), G.t(quat))

    # ------------------------------------------------------------------ widest faces (owner 2026-10-13, touching#02)

    def widest_faces(self) -> Optional[str]:
        """Owner 2026-10-13 (touching#02: "aligned means their widest faces point the same way; now they are perpendicular"):
        ``generation.widest_faces`` (default None = off), one of :data:`WIDEST_FACES` - the target's and the partner's widest upright
        faces are parallel at init and at the goal. The target's init yaw is one of :meth:`widest_faces_target_yaws` (no jitter), so its
        goal yaw (the edge-aligned yaw nearest the init yaw) is the init yaw; the partner takes the two edge-aligned turns parallel to
        the target's drawn pose (:meth:`widest_faces_partner_yaws`). The faces come from the hulls at their rest orientations, whatever
        each model's local frame. Touching with the edge alignment and no shelf layout only, else ``Reject("spec")``."""
        mode = self.gen("widest_faces")
        if mode is None:
            return None
        if mode not in WIDEST_FACES or self.relation != "touching" or not self.edge_aligned() or getattr(self, "shelf_start", None) is not None:
            raise Reject("spec", f"{self.gp.variant}: generation.widest_faces {mode!r}: one of {WIDEST_FACES}, for touching with the edge alignment "
                                 f"and no shelf layout")
        return mode

    def brace_open_when_wide(self) -> bool:
        """Owner 2026-10-14 ~02:04 (touching#02: "left brace should be open gripper like touching 1 because it's wide (so make this
        situational key for the touching tasks)"): ``generation.brace_open_when_wide`` (default false = off). On, a partner whose braced
        face is wide enough for the open fingertips (:meth:`brace_open_rule`) is braced by the open-finger stop only - no closed tips,
        blade or pad fallback, a draw without one is rejected (``brace_open_required``); a narrower face keeps today's candidates.
        Touching without the shelf layout (which braces nothing) only, else ``Reject("spec")``."""
        on = self.gen("brace_open_when_wide")
        if on is None or on is False:
            return False
        if on is not True or self.relation != "touching" or getattr(self, "shelf_start", None) is not None:
            raise Reject("spec", f"{self.gp.variant}: generation.brace_open_when_wide {on!r}: true / false, for touching without the shelf layout")
        return True

    def push_speed(self) -> Optional[float]:
        """Owner 2026-10-14 ~02:04 (touching#02: "right gripper should push more gently (slowly) - slow push should be parameter of
        pushes - so that the cereal box doesn't wobble that much"): ``generation.push_speed_m_s`` (default None = the oracle's own speeds:
        the arm's Cartesian speed for a straight push, ``push_speed_arc`` for an arc), the pad speed recorded on the touching push
        (``contacts.push.push_speed_m_s``) for every pad leg of it (oracle ``execute_push``). A positive number, touching only, else
        ``Reject("spec")``."""
        v = self.gen("push_speed_m_s")
        if v is None:
            return None
        try:
            speed = float(v)
        except (TypeError, ValueError):
            speed = float("nan")
        if not speed > 0.0 or self.relation != "touching":
            raise Reject("spec", f"{self.gp.variant}: generation.push_speed_m_s {v!r}: a positive speed (m/s), for touching only")
        return speed

    def widest_face_normal_yaw(self, hull: th.Tensor, quat) -> float:
        """World yaw (rad, modulo pi) of the normal of the hull's widest upright face at ``quat``: the short axis of its footprint's
        minimum-area bounding rectangle (:meth:`footprint_axis_angle`)."""
        theta = self.footprint_axis_angle(hull, quat)
        pts = self.world.footprint(hull, (th.zeros(3), G.t(quat)))
        u = pts[:, 0] * math.cos(theta) + pts[:, 1] * math.sin(theta)
        v = -pts[:, 0] * math.sin(theta) + pts[:, 1] * math.cos(theta)
        return (theta if float(u.max() - u.min()) <= float(v.max() - v.min()) else theta + math.pi / 2) % math.pi

    def widest_face_axis(self, hull: th.Tensor, quat, manip: Stance) -> str:
        """``"tangent"`` / ``"depth"``: the axis of ``manip``'s edge frame nearer the normal of the hull's widest upright face at ``quat``."""
        d = (self.widest_face_normal_yaw(hull, quat) - math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))) % math.pi
        return "tangent" if min(d, math.pi - d) < math.pi / 4 else "depth"

    def edge_aligned_yaws(self, role: str, manip: Stance) -> List[float]:
        """The four world yaws (applied to ``role``'s rest orientation) that put its footprint along the stance's edge frame."""
        yaw0 = self.edge_yaw_delta(self.hull[role], self.rest[role], manip)
        return [yaw0 + k * math.pi / 2 for k in range(4)]

    def _yaws_with_axis(self, role: str, manip: Stance, axis: str) -> List[float]:
        """``role``'s edge-aligned yaws whose widest face's normal lies along ``axis`` (all four for a square footprint, which has none)."""
        yaws = self.edge_aligned_yaws(role, manip)
        rest = G.t(self.rest[role])
        match = [y for y in yaws if self.widest_face_axis(self.hull[role], G.qmul(G.quat_from_yaw(y), rest), manip) == axis]
        return match or yaws

    def widest_faces_target_yaws(self, manip: Stance) -> List[float]:
        """The target's init yaws under :meth:`widest_faces`: its four edge-aligned yaws ("parallel"), or the two whose widest face's
        normal lies along the edge tangent ("tangent") / the stance's depth axis ("depth")."""
        mode = self.widest_faces()
        return self.edge_aligned_yaws("target", manip) if mode == "parallel" else self._yaws_with_axis("target", manip, mode)

    def widest_faces_partner_yaws(self, t_pose: G.Pose, manip: Stance) -> List[float]:
        """The partner's two edge-aligned yaws (180 deg apart) whose widest face is parallel to the target's at ``t_pose``."""
        return self._yaws_with_axis("partner", manip, self.widest_face_axis(self.hull["target"], t_pose[1], manip))

    def widest_faces_facts(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> Dict[str, Any]:
        """``summary["widest_faces"]``: the mode, each widest face's axis and the angle (deg) between the two normals at init and at the
        goal orientation (:meth:`target_goal_quat`)."""
        def normal(role, quat):
            return self.widest_face_normal_yaw(self.hull[role], quat)

        def between(a, b):
            d = (a - b) % math.pi
            return round(math.degrees(min(d, math.pi - d)), 2)

        q_goal = self.target_goal_quat(t_pose, manip)
        p_n = normal("partner", p_pose[1])
        return {"mode": self.widest_faces(), "target_init_axis": self.widest_face_axis(self.hull["target"], t_pose[1], manip),
                "target_goal_axis": self.widest_face_axis(self.hull["target"], q_goal, manip), "partner_axis": self.widest_face_axis(self.hull["partner"], p_pose[1], manip),
                "init_deg": between(normal("target", t_pose[1]), p_n), "goal_deg": between(normal("target", q_goal), p_n)}

    # ------------------------------------------------------------------ pair goal alignment (owner 2026-10-13 ~22:23, touching#02 / nextto#01 / #03 / #04)

    def pair_goal_align(self) -> Optional[Dict[str, Any]]:
        """Owner 2026-10-13 ~22:23 (COMPLAINTS.md "Pair goal alignment": the target ends in a set orientation relative to its partner, the
        way a person would put the pair down): ``generation.pair_goal_align`` (default None = off), nextto / touching only. ``front``:
        ``{"<category>/<model>": "+x" | "-x" | "+y" | "-y"}`` - the end each model points with (a pencil's tip, a shoe's toe, a box's
        printed front), in its own frame, for the drawn target and partner; ``relative_yaw_deg`` (0): the target's front is the partner's
        turned by this about the vertical (0 = pointing the same way); ``facing`` (``"any"``): where the target's front points in the
        stance's edge frame (:data:`PAIR_FACINGS`); ``faces`` (touching, optional): ``"widest"`` - the faces the boxes meet on are their
        widest upright faces; ``tol_deg`` (5) / ``stagger_max_m`` (0.03): the settled goal's tolerances; ``push_max_height_m`` (touching,
        optional): the push contact at most this high above the target's bottom (a thin box pushed across its thin axis tips above
        b / mu). The pair stands side by side across their long axes (:meth:`pair_axis_angle`: the model's oriented box's longest side
        lying, else the footprint's long side): the target's init yaw is one of :meth:`pair_target_yaws` (long axis across the edge, the
        front per ``facing``), the partner's :meth:`pair_partner_yaw`, level with it (:meth:`pair_level_depth`), and the goal - the target
        beside the partner along the edge, as for every nextto / touching goal - keeps that orientation (:meth:`pair_goal_quat`) at the
        partner's level. ``Reject("spec")`` with another relation, the shelf layout, ``partner_on_meta_link``,
        ``partner_aligned_to_target`` (it draws the partner's yaw itself) or a model without a ``front``."""
        spec = self.gen("pair_goal_align")
        if not spec:
            return None
        where = f"{self.gp.variant}: generation.pair_goal_align"
        if not isinstance(spec, dict) or self.relation not in ("nextto", "touching"):
            raise Reject("spec", f"{where} is a dict, for nextto / touching only")
        if getattr(self, "shelf_start", None) is not None or self.meta_link() is not None or bool(self.gen("partner_aligned_to_target", False)):
            raise Reject("spec", f"{where}: not with the shelf layout, partner_on_meta_link or partner_aligned_to_target")
        fronts, names = {}, {}
        for role in ("target", "partner"):
            mc = self.choice.models[role]
            key = f"{mc.category}/{mc.model}"
            axis = (spec.get("front") or {}).get(key)
            if axis not in PAIR_FRONT_AXES:
                raise Reject("spec", f"{where}.front[{key!r}] must be one of {sorted(PAIR_FRONT_AXES)}, got {axis!r}")
            fronts[role], names[role] = th.tensor(PAIR_FRONT_AXES[axis], dtype=th.float32), f"{key} {axis}"
        facing, faces, push_h = spec.get("facing", "any"), spec.get("faces"), spec.get("push_max_height_m")
        if facing not in PAIR_FACINGS or faces not in (None, "widest") or ((faces is not None or push_h is not None) and self.relation != "touching"):
            raise Reject("spec", f"{where}: facing one of {PAIR_FACINGS}, faces 'widest' and push_max_height_m for touching only (got {facing!r}, {faces!r}, {push_h!r})")
        return {"front": fronts, "front_axes": names, "relative_yaw_deg": float(spec.get("relative_yaw_deg", 0.0)), "facing": facing, "faces": faces,
                "tol_deg": float(spec.get("tol_deg", PAIR_TOL_DEG)), "stagger_max_m": float(spec.get("stagger_max_m", PAIR_STAGGER_MAX_M)),
                "push_max_height_m": None if push_h is None else float(push_h)}

    def pair_axis_angle(self, role: str, quat) -> float:
        """World angle (rad, either direction) of ``role``'s long axis at ``quat`` for :meth:`pair_goal_align`: :meth:`support_axis_angle`'s
        model long axis when it lies, else the long side of the footprint's minimum-area rectangle (:func:`min_area_rect`) - the footprint
        branch of support_axis_angle keeps the rectangle's axes modulo 90 deg only, and a box's long side must be told from its short one."""
        angle, source = self.support_axis_angle(role, quat)
        if source == "long_axis":
            return angle
        return min_area_rect(self.world.footprint(self.hull[role], (th.zeros(3), G.t(quat))))[0]

    def front_yaw(self, role: str, quat) -> float:
        """World yaw (rad) of ``role``'s front at ``quat`` (:meth:`pair_goal_align`): its named object axis projected on the floor, snapped to
        the nearest of the four directions of its :meth:`pair_axis_angle` - along the long axis for a pencil or a shoe, across the
        footprint's long side for a box's printed face (gym_shoe/mcfcwq's long axis is 4.6 deg off its +x)."""
        v = C.quat_to_mat(G.t(quat)).float() @ self.pair_goal_align()["front"][role]
        psi = math.atan2(float(v[1]), float(v[0]))
        theta = self.pair_axis_angle(role, quat)
        return G.wrap_angle(theta + round(G.wrap_angle(psi - theta) / (math.pi / 2)) * (math.pi / 2))

    def pair_facing_yaw(self, manip: Stance) -> Optional[float]:
        """World yaw (rad) the target's front must point to under :meth:`pair_goal_align`'s ``facing`` at ``manip``; None for ``"any"``."""
        facing = self.pair_goal_align()["facing"]
        if facing == "any":
            return None
        v = {"away": manip.inward, "toward": -manip.inward, "left": manip.tangent, "right": -manip.tangent}[facing]
        return math.atan2(float(v[1]), float(v[0]))

    def pair_target_yaws(self, manip: Stance) -> List[float]:
        """The target's init yaws under :meth:`pair_goal_align` (applied to its rest orientation): those of its four :meth:`support_aligned_yaws`
        whose long axis lies across the stance's edge - the partner beside it along the edge stands across their long axes - and whose front
        points per ``facing``. The goal keeps the init yaw (the base does not turn between the grasp and the place)."""
        rest = G.t(self.rest["target"])
        inward = math.atan2(float(manip.inward[1]), float(manip.inward[0]))
        want = self.pair_facing_yaw(manip)
        out = []
        for y in self.support_aligned_yaws("target", manip):
            q = G.qmul(G.quat_from_yaw(y), rest)
            if axes_angle_deg(self.pair_axis_angle("target", q), inward) > 45.0:
                continue
            if want is not None and abs(math.degrees(G.wrap_angle(self.front_yaw("target", q) - want))) > 45.0:
                continue
            out.append(y)
        if not out:
            raise Reject("spec", f"{self.gp.variant}: generation.pair_goal_align: no support-aligned yaw lays the target's long axis across the edge with "
                                 f"its front {self.pair_goal_align()['facing']}")
        return out

    def pair_partner_yaw(self, t_pose: G.Pose) -> float:
        """The partner's yaw (applied to its rest orientation) that puts its front at the target's front turned back by ``relative_yaw_deg``."""
        rel = math.radians(self.pair_goal_align()["relative_yaw_deg"])
        return G.wrap_angle(self.front_yaw("target", t_pose[1]) - rel - self.front_yaw("partner", self.rest["partner"]))

    def pair_goal_quat(self, t_pose: G.Pose, p_pose: G.Pose) -> th.Tensor:
        """The goal orientation under :meth:`pair_goal_align`: the target's init orientation turned about the vertical until its front is the
        partner's (at ``p_pose``) turned by ``relative_yaw_deg`` - no turn at the draft (the partner was drawn so), the partner's settle
        drift after it; ``Reject("pair_goal_align")`` beyond ``tol_deg``."""
        spec = self.pair_goal_align()
        d = G.wrap_angle(self.front_yaw("partner", p_pose[1]) + math.radians(spec["relative_yaw_deg"]) - self.front_yaw("target", t_pose[1]))
        if abs(math.degrees(d)) > spec["tol_deg"]:
            raise Reject("pair_goal_align", f"the target's front is {math.degrees(-d):+.1f} deg off the partner's turned by {spec['relative_yaw_deg']:.0f} deg "
                                            f"(> {spec['tol_deg']:.0f})")
        return G.qmul(G.quat_from_yaw(d), G.t(t_pose[1]))

    def pair_level_depth(self, role: str, quat, centre_xy, manip: Stance) -> float:
        """The depth (from the stance's front edge) of ``role``'s origin at ``quat`` that puts the centre of its footprint's minimum-area
        rectangle at the depth of the world point ``centre_xy`` (the other object's rectangle centre): the long axes' midpoints level."""
        d_c, _ = manip.local(G.t(centre_xy)[:2])
        off = min_area_rect(self.world.footprint(self.hull[role], (th.zeros(3), G.t(quat))))[3]
        return d_c - float(off.float() @ G.t(manip.inward).float())

    def pair_facts(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> Dict[str, Any]:
        """:meth:`pair_goal_align`'s residuals at the given poses: each front's yaw, ``rel_yaw_deg`` (the target's front against the partner's
        turned by ``relative_yaw_deg``, signed), ``long_axes_deg`` (0 = parallel), each long axis against the stance's depth axis,
        ``facing_deg`` (the target's front against ``facing``), ``offset_deg`` (the partner -> target offset of the footprint rectangles'
        centres against the partner's long axis: 90 = side by side), ``stagger_m`` (that offset along the partner's long axis), ``side`` (the
        target's side of the partner, robot frame) and ``pair_side`` (seen along the partner's front), ``gap_m`` (footprint clearance) and
        ``faces`` (the upright face of each that faces the other: its widest one or a narrow one)."""
        rel = math.radians(self.pair_goal_align()["relative_yaw_deg"])
        q_t, q_p = G.t(t_pose[1]), G.t(p_pose[1])
        phi_t, phi_p = self.front_yaw("target", q_t), self.front_yaw("partner", q_p)
        th_t, th_p = self.pair_axis_angle("target", q_t), self.pair_axis_angle("partner", q_p)
        fp_t, fp_p = self.world.footprint(self.hull["target"], t_pose), self.world.footprint(self.hull["partner"], p_pose)
        off = (min_area_rect(fp_t)[3] - min_area_rect(fp_p)[3]).float()
        along = float(off[0]) * math.cos(th_p) + float(off[1]) * math.sin(th_p)
        across = abs(float(off[1]) * math.cos(th_p) - float(off[0]) * math.sin(th_p))
        inward = math.atan2(float(manip.inward[1]), float(manip.inward[0]))
        want = self.pair_facing_yaw(manip)
        u = math.atan2(float(off[1]), float(off[0]))

        def face(role, quat):
            return "widest" if axes_angle_deg(self.widest_face_normal_yaw(self.hull[role], quat), u) < 45.0 else "narrow"

        return {"front_yaw_deg": {"target": round(math.degrees(phi_t), 2), "partner": round(math.degrees(phi_p), 2)},
                "rel_yaw_deg": round(math.degrees(G.wrap_angle(phi_t - phi_p - rel)), 2), "long_axes_deg": round(axes_angle_deg(th_t, th_p), 2),
                "axis_vs_depth_deg": {"target": round(axes_angle_deg(th_t, inward), 2), "partner": round(axes_angle_deg(th_p, inward), 2)},
                "facing_deg": None if want is None else round(abs(math.degrees(G.wrap_angle(phi_t - want))), 2),
                "offset_deg": round(math.degrees(math.atan2(across, abs(along))), 2), "stagger_m": round(along, 4),
                "side": "left" if float(off.float() @ G.t(manip.tangent).float()) > 0 else "right",
                "pair_side": "left" if -float(off[0]) * math.sin(phi_p) + float(off[1]) * math.cos(phi_p) > 0 else "right",
                "gap_m": round(TT.polygon_gap(fp_t, fp_p), 4), "faces": {"target": face("target", q_t), "partner": face("partner", q_p)}}

    def pair_violation(self, facts: Dict[str, Any]) -> str:
        """Why residuals of :meth:`pair_facts` break :meth:`pair_goal_align` (fronts / long axes beyond ``tol_deg``, the facing beyond it, the
        target staggered more than ``stagger_max_m`` along the partner's long axis, the wrong side, other faces than ``faces``), else ``""``."""
        spec, why = self.pair_goal_align(), []
        tol = spec["tol_deg"]
        if abs(facts["rel_yaw_deg"]) > tol:
            why.append(f"fronts {facts['rel_yaw_deg']:+.1f} deg off the required turn")
        if facts["long_axes_deg"] > tol:
            why.append(f"long axes {facts['long_axes_deg']:.1f} deg apart")
        if facts["facing_deg"] is not None and facts["facing_deg"] > tol:
            why.append(f"front {facts['facing_deg']:.1f} deg off '{spec['facing']}'")
        if abs(facts["stagger_m"]) > spec["stagger_max_m"]:
            why.append(f"staggered {facts['stagger_m']:+.3f} m along the partner's long axis")
        if self.goal_side is not None and facts["side"] != self.goal_side:
            why.append(f"on the partner's {facts['side']}, not its {self.goal_side}")
        if spec["faces"] is not None and (facts["faces"]["target"] != spec["faces"] or facts["faces"]["partner"] != spec["faces"]):
            why.append(f"meeting on the {facts['faces']['target']} / {facts['faces']['partner']} faces")
        return "; ".join(why)

    def check_pair_goal(self, draft: Draft, tname: str, pname: str) -> None:
        """:meth:`pair_goal_align` at the settled goal (both objects where the goal settle left them): ``summary.pair_goal_align.goal_settled``;
        ``Reject("pair_goal_align")`` when :meth:`pair_violation` names a breach."""
        poses = {role: tuple(v.float() for v in self.objects[name].get_position_orientation()) for role, name in (("target", tname), ("partner", pname))}
        facts = self.pair_facts(poses["target"], poses["partner"], self.sidecar.stance("manipulation"))
        rec = draft.summary.get("pair_goal_align")
        if isinstance(rec, dict):
            rec["goal_settled"] = facts
        why = self.pair_violation(facts)
        if why:
            self.counters["pair_goal_align"] = self.counters.get("pair_goal_align", 0) + 1
            raise Reject("pair_goal_align", f"settled goal: {why}")

    def target_goal_quat(self, t_pose: G.Pose, manip: Stance, p_pose: Optional[G.Pose] = None) -> th.Tensor:
        """The target's goal orientation; with :meth:`pair_goal_align` and the partner's pose, :meth:`pair_goal_quat`. The shelf layout: the
        start mirrored to lean the partner's way, parallel to it - with ``generation.shelf_start_draw`` (oracle-fix 2026-11-06) the partner's
        own drawn lean (``partner_lean_deg``; the turn is the sum of the two leans, the tip release settles from there)."""
        if self.shelf_start is not None:  # touching#03: the book ends resting against the partner, leaning like it
            lean = self.shelf_start["lean_deg"]
            if self.shelf_start.get("partner_lean_deg") is not None:
                lean = 0.5 * (float(lean) + float(self.shelf_start["partner_lean_deg"]))
            return TT.mirrored_lean_quat(t_pose[1], self.edge, self.shelf_start["lean_side"], lean)
        if self.relation == "inside" and self.up_axis() is not None:
            return self.standing_goal_quat(t_pose[1], manip)
        if p_pose is not None and self.pair_goal_align() is not None:
            return self.pair_goal_quat(t_pose, p_pose)
        return self.edge_aligned_quat(self.hull["target"], t_pose[1], manip) if self.edge_aligned() else G.t(t_pose[1])

    def place_pose_for(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> G.Pose:
        """Target pose at its *place* position for a drafted partner pose: the nearer-side rule of ``goal_pose`` with the
        place gap (nextto: g_NT - 1 cm; touching: the same, the push closes it); :meth:`pair_goal_align`: at the levelled depth."""
        campaign_plan = getattr(self, "_campaign_pair_active", None)
        if campaign_plan is not None and self.relation == "touching":
            planned = tuple(G.t(v) for v in campaign_plan.place)
            native_pos = G.t(campaign_plan.goal[0]) - th.tensor([float(manip.tangent[0]), float(manip.tangent[1]), 0.0]) * self.place_gap()
            dp, dq = CPG.pose_error((native_pos, G.t(campaign_plan.goal[1])), planned)
            if dp > self.ikc.WITNESS_POS_TOL or dq > self.ikc.WITNESS_ANG_TOL:
                raise Reject("campaign_place_drift", "fixed touching place disagrees with native push direction/gap")
            if not self.top.polygon_supported(self.world.footprint(self.hull["target"], planned), margin=self.margin_support):
                raise Reject("campaign_goal_support", "fixed touching pre-push place leaves the support")
            return planned
        t_quat = self.target_goal_quat(t_pose, manip, p_pose)
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        p_lat = (p_hull_w[:, :2] - manip.base_xy) @ manip.tangent
        t_lat = t_hull[:, :2] @ manip.tangent
        depth_t, lat_init = manip.local(t_pose[0])
        if self.pair_goal_align() is not None:
            depth_t = self.pair_level_depth("target", t_quat, min_area_rect(self.world.footprint(self.hull["partner"], p_pose))[3], manip)
        _, lat = nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_lat.min()), float(t_lat.max())), self.place_gap())
        xy = manip.world(depth_t, lat)
        dz = shelf_goal_dz(getattr(self, "shelf_start", None), self.hull["target"], t_pose, t_quat)  # oracle-fix 2026-11-06 (shelf_start_draw)
        return th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2]) + dz]), t_quat

    # ------------------------------------------------------------------ touching shelf layout (touching#03, owner 2026-10-11 11:46)

    # -- the start drawn per attempt (oracle-fix 2026-11-06, owner feedback on the touching#03 demo: "touching 3 - it does need more varied
    #    start - why don't we try also initializing in different compartments' walls, and at different lean angles?") --

    def shelf_start_draw(self) -> Optional[Dict[str, Any]]:
        """Opt-in ``generation.shelf_start_draw`` (oracle-fix 2026-11-06, touching#03: the books started at one pose in every attempt, so with the
        counted eval index every completed attempt was ``repeats_eval`` - start and goal 0.00 m / 0 deg from the eval instance): the shelf
        layout's start drawn per attempt instead of fixed by the template. ``{"layouts": {support: [[bay_m, side], ...]}, "lean_deg": [lo, hi],
        "spine_behind_edge_m": [lo, hi], "tries": n}``: ``layouts`` - the compartment (its centre, m along the edge tangent from the edge
        midpoint, + = the robot's left: ``shelf_bay_lateral_m``'s lateral) and the side board the target leans against (``"left"`` /
        ``"right"``, robot frame; the partner leans against the compartment's other board and the arm on the target's side works), one drawn
        uniformly per attempt - a support without a list keeps the template's layout (``shelf_bay_lateral_m`` or 0, the key's side);
        ``lean_deg`` - the target's and the partner's lean, each drawn uniformly (omitted: the key's lean for both); ``spine_behind_edge_m``
        - the spine depth behind the board's front edge, one draw for both books (omitted: ``shelf_spine_behind_edge_m``); ``tries`` - the
        layouts an attempt draws until both starts pass (:data:`SHELF_DRAW_TRIES`). Normalised: ``{"layouts": [(bay_m, side sign)],
        "lean_deg", "spine_behind_edge_m", "tries"}`` for this template's support. None when off (every attempt starts at the template's
        start, as before) or outside the shelf layout; ``Reject("spec")`` for a malformed block."""
        if getattr(self, "shelf_start", None) is None:
            return None
        raw = self.gen("shelf_start_draw")
        if raw is None:
            return None
        cached = self.__dict__.get("_shelf_draw_spec")
        if cached is not None and cached[0] is raw:
            return cached[1]
        bad = f"{self.gp.variant}: generation.shelf_start_draw"
        if not isinstance(raw, dict):
            raise Reject("spec", f"{bad} must be a dict, got {raw!r}")

        def pair(key: str, lo_min: float, hi_max: float) -> Optional[Tuple[float, float]]:
            v = raw.get(key)
            if v is None:
                return None
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and lo_min <= float(v[0]) <= float(v[1]) <= hi_max):
                raise Reject("spec", f"{bad}.{key} must be [lo, hi] with {lo_min} <= lo <= hi <= {hi_max}, got {v!r}")
            return float(v[0]), float(v[1])

        name = getattr(self.support, "name", "")
        bays = self.gen("shelf_bay_lateral_m")
        bay0 = float(bays.get(name, 0.0)) if isinstance(bays, dict) else 0.0
        side0 = float(self.__dict__.get("_shelf_template", self.shelf_start)["lean_side"])
        per_support = raw.get("layouts") or {}
        if not isinstance(per_support, dict):
            raise Reject("spec", f"{bad}.layouts must map a support name to its layouts, got {per_support!r}")
        rows = per_support.get(name)
        layouts: List[Tuple[float, float]] = []
        for row in rows if rows is not None else [[bay0, TT.side_name(side0)]]:
            if not (isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], (int, float)) and row[1] in ("left", "right")):
                raise Reject("spec", f"{bad}.layouts.{name}: every layout is [bay_m, 'left' | 'right'], got {row!r}")
            layouts.append((float(row[0]), 1.0 if row[1] == "left" else -1.0))
        if not layouts:
            raise Reject("spec", f"{bad}.layouts.{name} is empty")
        tries = raw.get("tries", SHELF_DRAW_TRIES)
        if not isinstance(tries, int) or tries < 1:
            raise Reject("spec", f"{bad}.tries must be a positive int, got {tries!r}")
        spec = {"layouts": layouts, "lean_deg": pair("lean_deg", 0.0, 45.0), "spine_behind_edge_m": pair("spine_behind_edge_m", 0.0, 0.5), "tries": tries}
        # i7-t03both 2026-11-09 (owner A5): ``rows`` {support: {"<board height m above the floor>": [[bay_m, side], ...]}} - the compartments of the
        # support's OTHER rows; drawn only with ISPATIALGYM_SHELF_BOARD_DRAW (each its own interior surface and maps; normalised [(z, layouts)])
        rows_raw = (raw.get("rows") or {}).get(name)
        rows_n: List[Tuple[float, List[Tuple[float, float]]]] = []
        if rows_raw is not None:
            if not isinstance(rows_raw, dict):
                raise Reject("spec", f"{bad}.rows.{name} must map a board height to its layouts, got {rows_raw!r}")
            for zk, lays in rows_raw.items():
                try:
                    zf = round(float(zk), 4)
                except (TypeError, ValueError):
                    zf = None
                if zf is None:
                    raise Reject("spec", f"{bad}.rows.{name}: a row key is a board height in m, got {zk!r}")
                out_l = []
                for row in lays or []:
                    if not (isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], (int, float)) and row[1] in ("left", "right")):
                        raise Reject("spec", f"{bad}.rows.{name}.{zk}: every layout is [bay_m, 'left' | 'right'], got {row!r}")
                    out_l.append((float(row[0]), 1.0 if row[1] == "left" else -1.0))
                if out_l:
                    rows_n.append((zf, out_l))
        if rows_n:
            spec["rows"] = sorted(rows_n)
        self._shelf_draw_spec = (raw, spec)
        return spec

    def shelf_draw_facts(self) -> Dict[str, Any]:
        """``template_facts["shelf_start_draw"]``: the layouts an attempt draws from (bay, the target's board), the ranges, the tries and the
        template's own layout; the mirror (the other board, the other arm) needs start_per_attempt (each attempt's Phase 2 with that arm)."""
        spec = self.shelf_start_draw()
        bays = self.gen("shelf_bay_lateral_m")
        return {"layouts": [[bay, TT.side_name(sd)] for bay, sd in spec["layouts"]], "lean_deg": spec["lean_deg"], "spine_behind_edge_m": spec["spine_behind_edge_m"],
                "tries": spec["tries"], "template_layout": [float(bays.get(self.support.name, 0.0)) if isinstance(bays, dict) else 0.0,
                                                            TT.side_name(self._shelf_template["lean_side"])],
                "mirror": bool(start_per_attempt(self)) and any(sd != self._shelf_template["lean_side"] for _, sd in spec["layouts"]),
                **({"rows": {str(z): [[bay, TT.side_name(sd)] for bay, sd in lays] for z, lays in spec["rows"]}} if spec.get("rows") and shelf_board_draw_on(self) else {})}

    def shelf_use_arm(self, arm: str) -> None:
        """The shelf layout's working arm (owner 2026-10-11 11:46): the arm on the target's side grasps, places and pushes, the other arm has no
        role; Phase 2's foci come from that arm's band (its own, else the mirror). oracle-fix 2026-11-06: :meth:`draw_shelf_side` sets it per attempt."""
        self.segment_arms = [arm]
        if self.gen("shelf_working_arm_default"):  # oracle-fix 2026-11-04 round 3: stances checked with the pushing arm working (idle-posture rejects)
            self.ikc.default_working_arms = [arm]
        self.arm_offer_why = f"touching shelf layout: the {arm} arm on the target's side grasps, places and pushes it; nothing braces the partner"
        self.arm_bands = TT.arm_bands_for(self, self.segment_arms, bands=self.bands)

    def shelf_rests(self, side: float, lean_target: float, lean_partner: float) -> Tuple[th.Tensor, th.Tensor]:
        """The two books' rest orientations for a drawn layout: the target's library rest turned to lean ``lean_target`` deg toward ``side``
        (``tabletop.releaned_quat``), the partner leaning ``lean_partner`` deg toward the other board (``tabletop.upright_lean_quat``).
        i7-lean 2026-11-07: a v2 LEAN rest (the key ``<key>@<lean tag>``) is the target's rest as measured - never re-leaned (its grasps
        and its interior map are that rest's; :meth:`draw_shelf_side` keeps its board, :meth:`draw_shelf_layout` its angle)."""
        base = self._shelf_template
        if TT.lean_rest_side(getattr(self, "grasp_key", None)) is not None:
            return self._shelf_rest0["target"].clone(), TT.upright_lean_quat(self.hull["partner"], -float(side), float(lean_partner))
        t_rest = TT.releaned_quat(self._shelf_rest0["target"], float(base["lean_side"]), float(base["lean_deg"]), side, lean_target)
        return t_rest, TT.upright_lean_quat(self.hull["partner"], -float(side), float(lean_partner))

    def draw_shelf_side(self, rng: np.random.Generator) -> None:
        """:meth:`shelf_start_draw`, before the attempt's start: one of the support's layouts uniformly - its board names the working arm
        (:meth:`shelf_use_arm`) and the books' nominal rests (the key's lean, mirrored to that board) feed the start's Phase 2 and its cache
        key; :meth:`draw_shelf_layout` then places the books. Restricted to the key's board without start_per_attempt (the template's Phase-2
        candidates belong to its arm) and, for eval, to the board opposite ``generation.partner_side``."""
        spec = self.shelf_start_draw()
        base = self._shelf_template
        layouts, why = list(spec["layouts"]), None
        assigned = self.gen("partner_side")
        fixed = (-1.0 if assigned == "left" else 1.0) if assigned in ("left", "right") else None
        if fixed is not None:
            why = f"generation.partner_side {assigned}: the target on the {TT.side_name(fixed)} board"
        elif not start_per_attempt(self):
            fixed, why = float(base["lean_side"]), "no start_per_attempt: the template's arm and candidates"
        lean_side = TT.lean_rest_side(getattr(self, "grasp_key", None))
        pool = getattr(self, "_board_keys", None)
        board_draw = bool(pool) and bool(start_per_attempt(self))
        if board_draw:
            # i7-t03both 2026-11-09 (owner A5, ISPATIALGYM_SHELF_BOARD_DRAW): the compartment (its row too: ``shelf_start_draw.rows``) and the board
            # are drawn among the layouts whose board has a lean rest with that row's interior map; eval keeps the board opposite partner_side
            have = [sd for sd in (1.0, -1.0) if pool.get(sd)]
            if fixed is not None and assigned in ("left", "right"):
                have = [sd for sd in have if sd == fixed]
            z0 = self._shelf_row_z0
            cands = []
            for c in [(b, sd, z0) for b, sd in layouts] + [(b, sd, self.shelf_row_key(z)) for z, lays in (spec.get("rows") or []) for b, sd in lays]:
                if c not in cands and c[1] in have and any(c[2] in st["rows"] for st in pool[c[1]]):
                    cands.append(c)
            if not cands:
                raise Reject("missing_grasps:lean", f"shelf board draw: no layout of {self.support.name} on a board with a lean rest and its row's map "
                                                    f"({sorted(TT.side_name(sd) for sd in have)})")
            fixed, why = None, f"shelf board draw: the boards with a lean rest {[TT.side_name(sd) for sd in have]}"
            lean_side = None
        elif pool:
            TT._count(self, "shelf_board_draw.off_no_start_per_attempt")
        if lean_side is not None:  # i7-lean 2026-11-07: a v2 lean rest leans on one board (its grasps and its map are that rest's)
            fixed, why = lean_side, f"v2 lean rest {getattr(self, 'grasp_key', None)}: its board"
        if fixed is not None:
            bays = self.gen("shelf_bay_lateral_m")
            bay0 = float(bays.get(self.support.name, 0.0)) if isinstance(bays, dict) else 0.0
            layouts = [lay for lay in layouts if lay[1] == fixed] or [(bay0, fixed)]
        drawn_key = None
        if board_draw:
            bay, side, z = cands[int(rng.integers(len(cands)))]
            states = [st for st in pool[side] if z in st["rows"]]
            st_k = states[int(rng.integers(len(states)))] if len(states) > 1 else states[0]
            self.apply_board_key(st_k, z)
            base = self._shelf_template
            drawn_key = st_k["key"]
            layouts = [(c[0], c[1]) for c in cands]
            TT._count(self, f"shelf_board_draw.drawn.{TT.side_name(side)}")
            TT._count(self, f"shelf_board_draw.layout.{z:.4f}_{bay:+.2f}_{TT.side_name(side)}")
            TT._count(self, f"shelf_board_draw.rest.{drawn_key.split('@')[-1]}")
            log.info("%s: shelf board draw: row %.4f bay %+.2f %s board, lean rest %s (%d layouts, %d rests on that board and row)", self.gp.variant, z, bay,
                     TT.side_name(side), drawn_key, len(cands), len(states))
        else:
            k = int(rng.integers(len(layouts)))
            bay, side = layouts[k]
        self.shelf_start = {"key": base["key"], "lean_side": side, "lean_deg": base["lean_deg"], "context": base.get("context")}
        self.rest["target"], self.rest["partner"] = self.shelf_rests(side, float(base["lean_deg"]), float(base["lean_deg"]))
        self.shelf_use_arm(self.shelf_push_arm())
        self.shelf_pose = self.shelf_partner_pose = None
        self._shelf_draw = {"layout": [bay, TT.side_name(side)], "n_layouts": len(layouts), "restricted": why}
        if drawn_key is not None:
            self._shelf_draw["board_draw"] = {"grasp_key": drawn_key, "lean_deg": base["lean_deg"], "row_z": z, "n_rests": len(states)}
            self._shelf_side_layouts = [(c[0], c[1]) for c in cands if c[1] == side and c[2] == z]  # a redraw stays on the row (its top)
        else:
            self._shelf_side_layouts = [lay for lay in layouts if lay[1] == side]

    def _board_key_state(self, key: str) -> Dict[str, Any]:
        """i7-t03both (owner A5): the target's grasp set at lean key ``key`` as :meth:`sample_template` builds the template's - the library's
        rows (``tabletop.library_grasps(key=)``), the attachable ones, the pinch re-centring, the flush tier - with its rest, its shelf facts
        (lean side / angle) and its pre-shape opening. ``Reject`` as the template's would."""
        tgt = self.obj("target")
        grasps, key, rest = TT.library_grasps(self, "target", key=key)
        grasps = [g for g in grasps if self.ikc.attachable(tgt, g.pose, "right")]
        if not grasps:
            raise Reject("no_attachable_grasp", f"{key}: every library grasp self-collides once the object is attached")
        p_touch = IK.finger_touch_params(self.gen("approach_finger_support_touch"))
        op = None
        if p_touch:
            ws = [TT.grasp_pinch_width(g) for g in grasps]
            op = None if not ws or None in ws else min(C.FINGER_OPEN, max(0.5 * w + IK.ORACLE_PRESHAPE_MARGIN_M for w in ws))
        keep = {k: self.template_facts.get(k) for k in ("grasp_pinch_recentre",)}  # the template's own facts stay the template's
        try:
            grasps, offsets = self.recentre_pinches(grasps, tgt)
        finally:
            for k, v in keep.items():
                if v is None:
                    self.template_facts.pop(k, None)
                else:
                    self.template_facts[k] = v
        tilt_max = float(self.gen("closing_tilt_max_deg", CLOSING_TILT_MAX_DEG))
        flush = [g for g in grasps if self.closing_tilt_deg(g.pose) <= tilt_max]
        facts = TT.upright_shelf_facts(key, rest) or {}
        return {"key": key, "grasps": list(grasps), "grasps_flush": flush if flush else list(grasps), "rest": rest.clone(), "finger_open": op,
                "template": {"key": key, "lean_side": facts.get("lean_side"), "lean_deg": facts.get("lean_deg"), "context": facts.get("context")},
                "bands": {}, "rows": []}

    def apply_board_key(self, st: Dict[str, Any], z: Optional[float] = None) -> None:
        """i7-t03both: the attempt's lean rest ``st`` (:meth:`_board_key_state`) becomes the sampler's - grasp key, grasps (both tiers), the
        target's rest and the shelf template the draw starts from, the pre-shape opening - at the row ``z`` (board height above the floor;
        None = the current row, :meth:`apply_shelf_row`), and its band there (cached per rest and row: the band left behind is kept on its
        state; None = :func:`tabletop.ensure_bands` reads the rest's interior map at the next lookup). A cached map band re-registers the
        rest's support policy at the row's board (as its lookup did)."""
        cur, cz = getattr(self, "_board_current", None), getattr(self, "_shelf_row_z", None)
        if cur is not None:
            cur["bands"][cz] = self.bands
            if self.bands is not None:  # kept alive: tabletop.arm_band caches the other arm's band by id() of this one
                self.__dict__.setdefault("_board_band_refs", []).append(self.bands)
        if z is not None and z != cz:
            self.apply_shelf_row(z)
        self.grasp_key, self.grasps, self.grasps_flush = st["key"], list(st["grasps"]), list(st["grasps_flush"])
        self.rest["target"] = st["rest"].clone()
        self._shelf_rest0["target"] = st["rest"].clone()
        self._shelf_template = dict(st["template"])
        if st["finger_open"] is not None or "_shelf_finger_open" in self.__dict__:
            self._shelf_finger_open = st["finger_open"]
        self.bands = st["bands"].get(getattr(self, "_shelf_row_z", None))
        self._board_current = st
        if self.bands is not None and (cur is not st or z != cz):
            self._board_policy()

    def _board_policy(self) -> None:
        """The support policy of the current lean rest at the current row's board (``tabletop._register_map_policy``, as an interior map hit
        registers it) - a band served from the cache did not look the map up again."""
        from ispatialgym.sampling import grasp_library as GL

        lib = GL.v2_entry(self.grasp_key)
        rid = GL.v2_rest_id_for_key(lib, self.grasp_key) if lib is not None else None
        if lib is not None and rid is not None and getattr(self, "top", None) is not None:
            TT._register_map_policy(self, lib, rid, {self.support.name: float(self.top.z)})

    def shelf_row_key(self, z: float) -> float:
        """A ``shelf_start_draw.rows`` height as the row it names: the template's own row (``_shelf_row_z0``) within ``interiors.HEIGHT_TOL_M``
        (the rows list may repeat the template's board to add compartments there), else the height (4 decimals)."""
        from ispatialgym.sampling import interiors as IN

        z0 = getattr(self, "_shelf_row_z0", None)
        return z0 if z0 is not None and abs(float(z) - float(z0)) <= IN.HEIGHT_TOL_M else round(float(z), 4)

    def shelf_row(self, z: float) -> Dict[str, Any]:
        """i7-t03both (``shelf_start_draw.rows``): the row context at board height ``z`` (m above the floor) - its interior surface
        (``tabletop.support_surface(top_z=)`` with the template's H_max), its own start-per-attempt edge trunks - built once (the template's
        own row is :meth:`sample_template`'s). The current row's surface, support split registration and facts are left as they were."""
        rows = self.__dict__.setdefault("_shelf_rows", {})
        if z in rows:
            return rows[z]
        keep_fact, keep_top, keep_sz = self.template_facts.get("support_surface"), self.top, getattr(self, "surface_z", None)
        H_max = max(TT.height_of(self.hull["target"], self.rest["target"]), TT.height_of(self.hull["partner"], self.rest["partner"]))
        try:
            top = TT.support_surface(self, self.support, height=H_max, top_z=float(z))
            sz = self.surface_z
        finally:
            self.template_facts["support_surface"] = keep_fact
            self.top, self.surface_z = keep_top, keep_sz
            if keep_top is not None:
                from ispatialgym.sampling.v2 import support_split as SS

                SS.register_top(self, self.support, keep_top, what="interior")
        rows[z] = {"top": top, "surface_z": sz, "spa_edges": {}, "facts": {"z_world": round(float(top.z), 4), "n_free_cells": int(top.free.sum())}}
        return rows[z]

    def apply_shelf_row(self, z: float) -> None:
        """i7-t03both: the row ``z`` becomes the sampler's - its interior surface (``top``, ``surface_z``: the band lookups' height), its support
        split plane, its edge trunks; the attempt's start (edge, init, Phase 2) is then drawn on it (start_per_attempt). Kit
        ``shelf_board_draw.row_switch``."""
        from ispatialgym.sampling.v2 import support_split as SS

        cz = getattr(self, "_shelf_row_z", None)
        if cz is not None:
            self.shelf_row(cz)["spa_edges"] = self.__dict__.get("_spa_edges", {})
        row = self.shelf_row(z)
        self.top, self.surface_z = row["top"], row["surface_z"]
        self._spa_edges = row["spa_edges"]
        SS.register_top(self, self.support, self.top, what="interior")
        self._shelf_row_z = z
        TT._count(self, "shelf_board_draw.row_switch")

    # -- i7-t03both diagnostics (ISPATIALGYM_SHELF_DRAW_DIAG=1, LOG ONLY: no verdict changes) --

    def _diag_map_status(self, arm: str, stance: Stance, xy_base: Tuple[float, float]) -> Dict[str, Any]:
        """The arm's band (an interior map view) at ``stance``'s trunk / stand-off: the raw map's status per grasp row at the cell nearest the
        object's base-frame xy, that cell's offset, and the nearest cell any row calls feasible."""
        band = self.band_of_arm(arm)
        meta = dict(getattr(band, "meta", None) or {})
        path = meta.get("map_path")
        out: Dict[str, Any] = {"map": os.path.basename(str(path)) if path else None, "band_arm": getattr(band, "arm", None)}
        if not path:
            return out
        cache = self.__dict__.setdefault("_diag_maps", {})
        if path not in cache:
            js = path if str(path).endswith(".json") else str(path)[:-4] + ".json"
            with open(js) as f:
                m = json.load(f)
            cache[path] = (m, dict(np.load(js[:-5] + ".npz")))
        m, z = cache[path]
        codes = {int(v): k for k, v in (m.get("status_codes") or {}).items()}
        cells = np.asarray(z["cells"], dtype=float)
        so = float(stance.stand_off if stance.stand_off is not None else 0.0)
        eds = [float(v) for v in m.get("edge_distances") or []]
        de = min(eds, key=lambda v: abs(v - so)) if eds else so
        key = f"s{de:.2f}_t{float(stance.translate):.2f}_k{float(stance.tilt):+.1f}"
        out.update({"trunk_key": key, "stand_off": round(so, 3)})
        if key not in z:
            tks = sorted(k for k in z if k.startswith(f"s{de:.2f}_"))
            out["trunk_missing"] = tks[:12]
            return out
        a = np.asarray(z[key])
        d = np.linalg.norm(cells - np.asarray(xy_base)[None, :], axis=1)
        i = int(np.argmin(d))
        names = [g.get("name") for g in m.get("grasps") or []]
        row = a[i].reshape(-1)
        out.update({"cell": [round(float(v), 3) for v in cells[i]], "cell_off_m": round(float(d[i]), 3),
                    "status": {f"{names[j] if j < len(names) else j}#{j}": codes.get(int(v), int(v)) for j, v in enumerate(row)}})
        feas = np.where((a.reshape(len(cells), -1) == 1).any(axis=1))[0]
        if len(feas):
            j = int(feas[np.argmin(d[feas])])
            out.update({"nearest_feasible_cell": [round(float(v), 3) for v in cells[j]], "nearest_feasible_m": round(float(d[j]), 3), "n_feasible_cells": int(len(feas))})
        else:
            out["n_feasible_cells"] = 0
        return out

    def _diag_base_frame(self, stance: Stance, xyz) -> Tuple[float, float]:
        x0, y0, yaw = (float(v) for v in stance.base_pose2d)
        dx, dy = float(xyz[0]) - x0, float(xyz[1]) - y0
        return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)

    def shelf_grasp_diag(self, manip: Stance, pose, grasps, arm: str, reasons, exact_kw) -> None:
        """LOG ONLY (``T03B_GRASP_DIAG {json}``): the fixed start's failed grasp at ``manip`` - the stance, the book in the base frame, the map's
        status at that cell (:meth:`_diag_map_status`) against the live verdicts, and per grasp a probe of the pre-grasp IK with the support
        ignored whole (reach vs collision) plus the links of that solution that touch the support."""
        try:
            from ispatialgym.oracle import r1pro_model as M

            xy = self._diag_base_frame(manip, G.t(pose[0]))
            rec: Dict[str, Any] = {"variant": self.gp.variant, "stance": manip.name, "base_pose2d": [round(float(v), 4) for v in manip.base_pose2d],
                                   "trunk": [round(float(manip.translate), 3), round(float(manip.tilt), 3)], "stand_off": manip.stand_off,
                                   "lateral": round(float(manip.lateral), 4), "arm": arm, "grasp_key": self.grasp_key,
                                   "row_z": getattr(self, "_shelf_row_z", None), "board": TT.side_name(float(self.shelf_start["lean_side"])) if self.shelf_start else None,
                                   "bay": (getattr(self, "_shelf_draw", None) or {}).get("bay_m"), "book_base_xy": [round(v, 4) for v in xy],
                                   "book_z": round(float(G.t(pose[0])[2]) - float(getattr(self.world, "floor_z", 0.0)), 4),
                                   "stance_extras": {k: manip.extras.get(k) for k in ("stance_shift_trunk", "init_view_angle", "lateral_shift_m") if k in manip.extras},
                                   "live": list(reasons)[:6]}
            rec["map"] = self._diag_map_status(arm, manip, xy)
            probes = {}
            links = [ln for ln in self.ikc._non_finger_links(arm) if "attached" not in ln] + list(M.FINGER_LINKS[arm])
            for g in list(grasps)[:4]:
                eef = IK.grasp_pose(pose, g.pose)
                pre = IK.pre_pose(eef)
                pr: Dict[str, Any] = {}
                with self.ikc.ignoring(self.support_objects(), split=False):
                    q = self.ikc._solve_exact(pre, self.ikc.q_locked(manip), arm)
                pr["pre_ik_support_ignored"] = q is not None
                if q is not None:
                    clear = self.kit.links_clear(q, links, 0.0)
                    pr["all_links_clear_with_support"] = bool(clear[0])
                    if not bool(clear[0]):
                        bad = [ln for ln in links if not bool(self.kit.links_clear(q, [ln], 0.0)[0])]
                        pr["links_touching"] = bad[:8]
                    eef_b = self._diag_base_frame(manip, G.t(pre[0]))
                    pr["pre_base_xy"] = [round(v, 3) for v in eef_b]
                    pr["pre_z"] = round(float(G.t(pre[0])[2]) - float(getattr(self.world, "floor_z", 0.0)), 3)
                else:
                    with self.ikc.ignoring(self.support_objects() + list(getattr(self.world, "spawned", {}).values()), split=False):
                        q2 = self.ikc._solve_exact(pre, self.ikc.q_locked(manip), arm)
                    pr["pre_ik_all_spawned_ignored"] = q2 is not None
                # the GRASP pose: solved with everything the exact chain ignores at its last pose, then with the support ignored whole too; the
                # blame of that solution in the live world (what the grasp configuration collides with) and the touching links
                ign_last = list(exact_kw.get("ignore_for_last") or [])
                with self.ikc.ignoring(ign_last):
                    qg = self.ikc._solve_exact(eef, self.ikc.q_locked(manip), arm)
                pr["grasp_ik_live"] = qg is not None
                if qg is None:
                    with self.ikc.ignoring(ign_last + list(self.support_objects()), split=False):
                        qg2 = self.ikc._solve_exact(eef, self.ikc.q_locked(manip), arm)
                    pr["grasp_ik_support_ignored"] = qg2 is not None
                    if qg2 is not None:
                        pr["grasp_blame"] = self.ikc._collision_blame(qg2, ign_last)[:200]
                        with self.ikc.ignoring(ign_last):
                            pr["grasp_links_touching"] = [ln for ln in links if not bool(self.kit.links_clear(qg2, [ln], 0.0)[0])][:8]
                        pr["grasp_self_pairs"] = [f"{a} x {b}" for (a, b), _ in (self.ikc.self_collision_pairs(qg2) or [])[:3]]
                        try:  # the gripper body's box behind the eef (eef frame: x +-0.045, y +-0.06, z -0.13..0) vs each split-kept support piece (SDF)
                            gx, gy, gz = np.meshgrid(np.linspace(-0.045, 0.045, 4), np.linspace(-0.06, 0.06, 5), np.linspace(-0.13, 0.0, 8), indexing="ij")
                            pts = th.tensor(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1), dtype=th.float32)
                            pts_w = G.local_points_world(pts, (G.t(eef[0]).float(), G.t(eef[1]).float()))
                            rows = []
                            for prim, _cm, geom in self._kept_pieces():
                                d = geom.sdf(pts_w)
                                j = int(th.argmin(d))
                                rows.append((float(d[j]), prim.rsplit("/", 1)[-1], [round(float(v), 3) for v in pts[j].tolist()], [round(float(v), 4) for v in pts_w[j].tolist()]))
                            rows.sort(key=lambda r: r[0])
                            pr["grasp_gripper_box_pieces"] = rows[:3]  # (sdf m, piece, the eef-frame point, its world point)
                        except Exception as e:  # noqa: BLE001
                            pr["grasp_gripper_box_pieces"] = repr(e)[:120]
                probes[getattr(g, "name", "?")] = pr
            rec["probes"] = probes
            log.info("T03B_GRASP_DIAG %s", json.dumps(rec, default=str))
        except Exception as e:  # noqa: BLE001 - a diagnostic never changes the attempt
            log.info("T03B_GRASP_DIAG error %r", e)

    def shelf_path_diag(self, init: Stance, back: Stance, s_T: float, err: str) -> None:
        """LOG ONLY (``T03B_PATH_DIAG {json}``): an init -> drive-back path failure - both base poses, the init's kind (free start or the
        edge), the nearest eroded-free cells and the sizes of their connected components (a pocket vs the room)."""
        try:
            from ispatialgym.sampling.v2 import navigation as NAV

            free = NAV.eroded_map(self.kit).cpu().numpy().astype(bool)
            out: Dict[str, Any] = {"variant": self.gp.variant, "s_T": round(float(s_T), 3), "err": err[:200], "init": [round(float(v), 3) for v in init.base_pose2d],
                                   "init_free_start": "free_start_from_edge_lateral_m" in (init.extras or {}), "back": [round(float(v), 3) for v in back.base_pose2d],
                                   "row_z": getattr(self, "_shelf_row_z", None), "board": TT.side_name(float(self.shelf_start["lean_side"])) if self.shelf_start else None,
                                   "bay": (getattr(self, "_shelf_draw", None) or {}).get("bay_m"), "arm": (self.segment_arms or [None])[0]}
            try:
                from scipy import ndimage

                lab, n = ndimage.label(free)
                for nm, st in (("init", init), ("back", back)):
                    ij = NAV._to_map(self.kit, st.base_xy)
                    c = NAV.nearest_free_cell(th.as_tensor(free), ij, 20)
                    out[f"{nm}_cell"] = list(ij)
                    out[f"{nm}_free_cell_off"] = None if c is None else int(max(abs(c[0] - ij[0]), abs(c[1] - ij[1])))
                    out[f"{nm}_component_cells"] = None if c is None else int((lab == lab[c]).sum())
                    out[f"{nm}_trav_free"] = bool(self.kit.trav_free(st.base_xy))
                out["n_components"] = int(n)
            except Exception as e:  # noqa: BLE001
                out["label_error"] = repr(e)[:120]
            log.info("T03B_PATH_DIAG %s", json.dumps(out, default=str))
        except Exception as e:  # noqa: BLE001
            log.info("T03B_PATH_DIAG error %r", e)

    def shelf_sweep_diag(self, qs: th.Tensor, back_xyyaw, s_T: float, manip: Stance) -> None:
        """LOG ONLY (``T03B_SWEEP_DIAG {json}``): the first colliding configuration of a transition's trunk sweep at s_T and what it collides with."""
        try:
            ok = self.kit.collision_free(qs, margin=0.01)
            k = int((~ok).nonzero()[0][0]) if bool((~ok).any()) else -1
            blame = self.ikc._collision_blame(qs[k], [], margin=0.01) if k >= 0 else "free"
            log.info("T03B_SWEEP_DIAG %s", json.dumps({"variant": self.gp.variant, "s_T": round(float(s_T), 3), "back": [round(float(v), 3) for v in back_xyyaw],
                                                       "manip": [round(float(v), 3) for v in manip.base_pose2d], "trunk_to": [manip.translate, round(float(manip.tilt), 3)],
                                                       "sample": k, "n": int(len(qs)), "blame": blame, "row_z": getattr(self, "_shelf_row_z", None),
                                                       "board": TT.side_name(float(self.shelf_start["lean_side"])) if self.shelf_start else None,
                                                       "arm": (self.segment_arms or [None])[0]}, default=str))
        except Exception as e:  # noqa: BLE001
            log.info("T03B_SWEEP_DIAG error %r", e)

    def shelf_board_pool(self) -> Optional[Dict[float, List[Dict[str, Any]]]]:
        """i7-t03both 2026-11-09 (owner A5, :data:`SHELF_BOARD_DRAW_ENV`), once per template: ``{+1 (left board) / -1: [rest states]}`` - every
        lean rest of the target's model within ``shelf_start_draw.lean_deg`` (``tabletop.shelf_lean_pool``) whose grasp set builds, with the
        rows (the template's board and every ``shelf_start_draw.rows`` height) where its interior reach maps serve for the default arm and the
        board's working arm (the probe reads them now; ``st["rows"]``). A rest without a row is dropped (kit
        ``shelf_board_draw.rest_dropped.<reason>``), a row without a rest is never drawn. The template's own rest is its state as built.
        None when the template is not at a v2 lean rest (the switch does nothing); facts ``template_facts["shelf_board_draw"]``."""
        from ispatialgym.grasping import lean as LN

        base_key, tag = LN.split_lean_key(self.grasp_key)
        if tag is None:
            return None
        keys = TT.shelf_lean_pool(self, base_key)
        z0 = round(float(self.top.z) - float(getattr(self.world, "floor_z", 0.0)), 4)
        self._shelf_row_z0 = self._shelf_row_z = z0
        self._shelf_rows = {z0: {"top": self.top, "surface_z": getattr(self, "surface_z", None), "spa_edges": self.__dict__.get("_spa_edges", {}),
                                 "facts": {"z_world": round(float(self.top.z), 4), "n_free_cells": int(self.top.free.sum()), "template": True}}}
        rows = [z0]
        for z, _ in ((self.shelf_start_draw() or {}).get("rows") or []):
            if self.shelf_row_key(z) not in rows:
                rows.append(self.shelf_row_key(z))
        own = {"key": self.grasp_key, "grasps": list(self.grasps), "grasps_flush": list(self.grasps_flush), "rest": self._shelf_rest0["target"].clone(),
               "finger_open": self.__dict__.get("_shelf_finger_open_template"), "template": dict(self._shelf_template), "bands": {}, "rows": []}
        saved = {"bands": self.bands, "arm_bands": dict(self.arm_bands or {})}
        pool: Dict[float, List[Dict[str, Any]]] = {}
        facts: Dict[str, Any] = {"keys": {}, "template_key": self.grasp_key, "rows": {}}
        self._board_current = own
        try:
            for z in rows:
                try:
                    if z != z0:
                        self.shelf_row(z)
                except Reject as e:
                    TT._count(self, f"shelf_board_draw.row_dropped.{e.reason}")
                    facts["rows"][str(z)] = {"dropped": f"[{e.reason}] {str(e)[:200]}"}
                    continue
                facts["rows"][str(z)] = self._shelf_rows[z]["facts"]
            for side_name, ks in sorted(keys.items()):
                for k in ks:
                    try:
                        st = own if k == self.grasp_key else self._board_key_state(k)
                    except Reject as e:
                        TT._count(self, f"shelf_board_draw.rest_dropped.{e.reason}")
                        facts["keys"][k] = {"dropped": f"[{e.reason}] {str(e)[:200]}"}
                        continue
                    arm = "left" if float(st["template"]["lean_side"]) > 0 else "right"
                    misses = {}
                    for z in [z for z in rows if z in self._shelf_rows]:
                        try:
                            self.apply_board_key(st, z)
                            self.bands = None
                            st["bands"][z] = TT.ensure_bands(self, top=self.top)
                            TT.arm_band(self, arm, bands=st["bands"][z])
                            st["rows"].append(z)
                        except Reject as e:
                            st["bands"].pop(z, None)
                            self.bands = None
                            misses[str(z)] = f"[{e.reason}] {str(e)[:160]}"
                            TT._count(self, f"shelf_board_draw.row_map_missing.{e.reason}")
                    facts["keys"][k] = {"side": side_name, "lean_deg": st["template"]["lean_deg"], "n_grasps": len(st["grasps"]), "arm": arm, "rows": list(st["rows"]),
                                        "missing": misses}
                    if not st["rows"]:
                        TT._count(self, "shelf_board_draw.rest_dropped.no_row")
                        continue
                    pool.setdefault(float(st["template"]["lean_side"]), []).append(st)
        finally:
            self.apply_board_key(own, z0)  # the template's rest at its row; its probed band (if any) is the template's band from here on
            self.arm_bands = saved["arm_bands"]
            if self.bands is None:
                self.bands = saved["bands"]
        facts["boards"] = {TT.side_name(sd): {st["key"]: st["rows"] for st in v} for sd, v in pool.items()}
        self.template_facts["shelf_board_draw"] = facts
        log.info("%s: shelf board draw: lean rests by board (rows) %s; rows %s", self.gp.variant, facts["boards"], facts["rows"])
        return pool or None

    def draw_shelf_layout(self, rng: np.random.Generator) -> None:
        """:meth:`shelf_start_draw`, after the attempt's start (its edge): the books' leaning starts of the drawn layout - the target against
        the drawn board of the compartment, the partner against the other (``tabletop.shelf_lean_pose`` with the bay as the rays' start and
        the drawn spine depth), each at its drawn lean - and the free span between them (``shelf_free_span_min_m``). A layout that fails
        (``shelf_wall`` / ``shelf_board`` / ``shelf_width``) is drawn again among the layouts of the same board (the arm is the attempt's),
        up to ``tries``; then that rejection. Sets the rests, the two poses and ``shelf_start`` (the drawn facts, ``partner_lean_deg``: the
        goal is drafted parallel to the partner); the facts go to ``summary.shelf_start``."""
        spec = self.shelf_start_draw()
        base = self._shelf_template
        side = float(self.shelf_start["lean_side"])
        same = self._shelf_side_layouts
        bay = float(self._shelf_draw["layout"][0])
        span_min = float(self.gen("shelf_free_span_min_m", 0.15))
        tried: List[str] = []
        last: Optional[Reject] = None
        for k in range(int(spec["tries"])):
            if k:
                bay = float(same[int(rng.integers(len(same)))][0])
            lean_t, lean_p = (float(rng.uniform(*spec["lean_deg"])), float(rng.uniform(*spec["lean_deg"]))) if spec["lean_deg"] else (float(base["lean_deg"]),) * 2
            if TT.lean_rest_side(getattr(self, "grasp_key", None)) is not None:  # i7-lean: the v2 lean rest's own angle (the partner's lean is still drawn)
                lean_t = float(base["lean_deg"])
            spine = float(rng.uniform(*spec["spine_behind_edge_m"])) if spec["spine_behind_edge_m"] else None
            rest_t, rest_p = self.shelf_rests(side, lean_t, lean_p)
            label = f"bay {bay:+.2f} {TT.side_name(side)} board, leans {lean_t:.1f} / {lean_p:.1f} deg" + ("" if spine is None else f", spine {spine:.3f} m")
            try:
                t_pose, tf = TT.shelf_lean_pose(self, self.hull["target"], rest_t, self.top, self.edge, self.support, side, spine_behind=spine, lateral0=bay)
                p_pose, pf = TT.shelf_lean_pose(self, self.hull["partner"], rest_p, self.top, self.edge, self.support, -side, spine_behind=spine, lateral0=bay)
                free = shelf_free_span(self.world.hull_at(self.hull["target"], t_pose), self.world.hull_at(self.hull["partner"], p_pose), self.edge, side)
                span_facts = None
                if shelf_span_rule_on():  # 5k.1: the requirement at this attempt's drawn leans
                    req, span_facts = self.shelf_span_required(t_pose)
                    if free < req:
                        TT._count(self, "shelf_span.rejected")
                        raise Reject("shelf_width", f"{self.support.name}: {free:.3f} m between the two leaning books (< {req:.3f} required: carried "
                                                    f"{span_facts['carried']} {span_facts['carried_extent_m']:.3f} + jaw {span_facts['jaw_excess_m']:.3f})")
                    TT._count(self, "shelf_span.passed")
                elif free < span_min:
                    raise Reject("shelf_width", f"{self.support.name}: {free:.3f} m between the two leaning books (< {span_min:.2f})")
            except Reject as e:
                if e.reason not in SHELF_DRAW_REJECTS:
                    raise
                TT._count(self, f"shelf_start_draw.{e.reason}")
                tried.append(f"{label}: [{e.reason}] {str(e)[:160]}")
                last = e
                continue
            start = {"key": base["key"], "lean_side": side, "lean_deg": round(lean_t, 2), "context": base.get("context"), "partner_lean_deg": round(lean_p, 2),
                     "bay_m": round(bay, 4), "spine_behind_edge_m": round(float(self.gen("shelf_spine_behind_edge_m", TT.SHELF_SPINE_BEHIND_EDGE_M) if spine is None else spine), 4)}
            merge_shelf_pose_facts(start, tf)
            start.update({"partner": pf, "free_span_m": round(free, 4)})
            if span_facts is not None:
                start["span_rule"] = span_facts
            self.shelf_start = start
            self.rest["target"], self.rest["partner"] = rest_t, rest_p
            self.shelf_pose, self.shelf_partner_pose = t_pose, p_pose
            self._shelf_draw.update({"bay_m": round(bay, 4), "target_board": TT.side_name(side), "target_lean_deg": round(lean_t, 2), "partner_lean_deg": round(lean_p, 2),
                                     "spine_behind_edge_m": start["spine_behind_edge_m"], "target_position": tf.get("position"), "partner_position": pf.get("position"),
                                     "wall_lateral_m": [tf.get("wall_lateral_m"), pf.get("wall_lateral_m")], "free_span_m": round(free, 4), "tries": k + 1,
                                     "rejected": tried})
            log.info("%s: shelf start draw: %s (the %s arm), free span %.3f m, try %d of %d", self.gp.variant, label, self.segment_arms[0], free, k + 1, int(spec["tries"]))
            return
        raise Reject(last.reason if last is not None else "shelf_wall", f"shelf_start_draw: no drawn layout of the {TT.side_name(side)} board passes in "
                                                                       f"{int(spec['tries'])} tries ({'; '.join(tried[-3:])})")

    # ------------------------------------------------------------------ the stance shift at the fixed start (i7-t03 2026-11-08, item 1)

    # ------------------------------------------------------------------ the straightening lift (i7-t03 round 2 (d))

    def _kept_pieces(self) -> List[Tuple[str, Any, Any]]:
        """``[(prim, cm, ConvexPieces)]`` of the shelf support's pieces its split keeps in the world (the side boards, walls, the board above)."""
        from ispatialgym.sampling.v2 import support_split as SS

        sup = self.support
        plan = SS.plan_for(self.ikc, sup, count=False) if SS.split_on() else None
        if plan is None or not plan.kept:
            return []
        out = []
        for prim, cm in SS._pieces(sup):
            if prim in plan.off:
                continue
            g = self.ikc._pieces_geometry(sup, frozenset([prim]))
            if g is not None:
                out.append((prim, cm, g))
        return out

    def shelf_standing_axis(self, pose: Optional[G.Pose] = None) -> Tuple[Optional[th.Tensor], Dict[str, Any]]:
        """(d) fix 2026-11-08 (touching has no ``generation.target_up_axis``: :meth:`up_axis` is inside-only and stays so): the book's
        standing (bottom -> top) axis as a signed model axis (+-e_i), the rule ``tabletop.upright_shelf_facts`` reads the shelf start's lean
        from (the local axis most aligned with world +z, ``grasping.lean.up_axis_local``) applied to the v2 lean entry's upright base
        (``lean_base_wxyz``, source ``v2_lean_base``) when the key is a lean key that has one, else to the template's rest (the upright-shelf
        key's rest or the measured lean rest; source ``rest``). ``(None, facts with why)`` when it cannot be read or stands more than
        :data:`SHELF_LIFT_AXIS_MAX_TILT_DEG` from vertical at the attempt's start pose. Never raises."""
        from ispatialgym.grasping import lean as LN
        from ispatialgym.sampling import grasp_library as GL

        facts: Dict[str, Any] = {}
        try:
            ax, src = None, None
            key = getattr(self, "grasp_key", None)
            if key and GL.lean_key_parts(key)[1] is not None:
                lib = GL.v2_entry(key)
                base = None if lib is None else lib.data.get("lean_base_wxyz")
                if base is not None:
                    ax, src = LN.up_axis_local(LN.xyzw(base)), "v2_lean_base"
            if ax is None:
                rest = (self.__dict__.get("_shelf_rest0") or getattr(self, "rest", None) or {}).get("target")
                if rest is not None:
                    ax, src = LN.up_axis_local(np.asarray(G.t(rest).detach().cpu().double().numpy())), "rest"
            if ax is None:
                return None, {"why": "no_rest"}
            v = th.as_tensor(np.asarray(ax, dtype=np.float64), dtype=th.float32)
            up_w = G.rotate(G.t((self.shelf_pose if pose is None else pose)[1]).float(), v)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up_w[2])))))
            facts.update(axis=[int(round(float(x))) for x in v], axis_source=src, start_tilt_deg=round(tilt, 2))
            if not math.isfinite(tilt) or tilt > SHELF_LIFT_AXIS_MAX_TILT_DEG:
                facts["why"] = "tilted"
                return None, facts
            return v, facts
        except Exception as e:  # noqa: BLE001 - (d) must fall back to today's lift, never fail the attempt
            log.warning("shelf_standing_axis: %s: %r", getattr(getattr(self, "gp", None), "variant", "?"), e)
            facts.update(why="error", error=repr(e)[:200])
            return None, facts

    def shelf_span_required(self, t_pose: G.Pose) -> Tuple[float, Dict[str, Any]]:
        """5k.1 (:data:`SHELF_SPAN_RULE_ENV`): the free span (m) the shelf layout needs with the target at ``t_pose`` - the carried book's
        extent along the edge tangent (its hull at ``t_pose``: the drawn lean; with the straightening lift on, the book turned upright on
        :meth:`shelf_standing_axis` - its extent at the lean when no axis is read) + the open jaw's excess over the pinch, both sides:
        ``opening + 2 FINGER_THICKNESS - w`` (``opening`` = the oracle's pre-shape ``w + 2 ORACLE_PRESHAPE_MARGIN_M`` under
        ``approach_finger_support_touch.preshape``, else ``GRIPPER_OPEN_M``; ``w`` = the narrowest library pinch, else the hull's thinnest
        extent). ``(required, facts)``."""
        tan = G.t(self.edge.tangent).float()[:2]
        pose = (G.t(t_pose[0]).float(), G.t(t_pose[1]).float())

        def extent(q: th.Tensor) -> float:
            lat = self.world.hull_at(self.hull["target"], (pose[0], q))[:, :2] @ tan
            return float(lat.max() - lat.min())

        carried, how = extent(pose[1]), "lean"
        if shelf_lift_straighten_on(self):
            axis, _ = self.shelf_standing_axis(pose)
            if axis is not None:
                q_up = G.qmul(shortest_arc(G.rotate(pose[1], axis), th.tensor([0.0, 0.0, 1.0])), pose[1])
                carried, how = extent(q_up), "upright"
        ws = [w for w in (TT.grasp_pinch_width(g) for g in (getattr(self, "grasps", None) or [])) if w is not None and w > 0]
        ext = self.hull["target"].max(dim=0).values - self.hull["target"].min(dim=0).values
        w = min(ws) if ws else float(ext.min())
        p_touch = IK.finger_touch_params(self.gen("approach_finger_support_touch"))
        opening = min(float(C.GRIPPER_OPEN_M), w + 2.0 * IK.ORACLE_PRESHAPE_MARGIN_M) if (p_touch and p_touch.get("preshape")) else float(C.GRIPPER_OPEN_M)
        excess = max(0.0, opening + 2.0 * FINGER_THICKNESS - w)
        req = carried + excess
        return req, {"required_m": round(req, 4), "carried_extent_m": round(carried, 4), "carried": how, "jaw_excess_m": round(excess, 4),
                     "opening_m": round(opening, 4), "pinch_m": round(w, 4)}

    def shelf_span_min(self, t_pose: G.Pose) -> float:
        """The free span the layout needs with the target at ``t_pose``: :meth:`shelf_span_required` with the 5k.1 switch, else
        ``generation.shelf_free_span_min_m`` (default 0.15)."""
        return self.shelf_span_required(t_pose)[0] if shelf_span_rule_on() else float(self.gen("shelf_free_span_min_m", 0.15))

    def shelf_template_span_check(self) -> None:
        """5k.1 at the template stage (switch on): the free span at the most favourable leans an attempt can draw (the target's lowest drawn
        lean - a v2 lean rest keeps its own - and the partner's lowest) against :meth:`shelf_span_required` there; ``Reject("shelf_width")``
        only when no draw can pass. Without ``shelf_start_draw``: the template's own leans. Facts ``template_facts["shelf_span"]``."""
        side = float(self.shelf_start["lean_side"])
        spec = self.shelf_start_draw()
        t_pose, p_pose = self.shelf_pose, self.shelf_partner_pose
        lean = None
        if spec is not None and spec.get("lean_deg"):
            lo = float(spec["lean_deg"][0])
            lean_t = float(self._shelf_template["lean_deg"]) if TT.lean_rest_side(getattr(self, "grasp_key", None)) is not None else lo
            rest_t, rest_p = self.shelf_rests(side, lean_t, lo)
            t_pose, _ = TT.shelf_lean_pose(self, self.hull["target"], rest_t, self.top, self.edge, self.support, side)
            p_pose, _ = TT.shelf_lean_pose(self, self.hull["partner"], rest_p, self.top, self.edge, self.support, -side)
            lean = [round(lean_t, 2), round(lo, 2)]
        free = shelf_free_span(self.world.hull_at(self.hull["target"], t_pose), self.world.hull_at(self.hull["partner"], p_pose), self.edge, side)
        req, rf = self.shelf_span_required(t_pose)
        self.template_facts["shelf_span"] = {"free_span_m": round(free, 4), "leans_deg": lean, **rf}
        if free < req:
            TT._count(self, "shelf_span.template_rejected")
            raise Reject("shelf_width", f"{self.support.name}: {free:.3f} m between the two leaning books at the most favourable drawable leans {lean} "
                                        f"(< {req:.3f} required: carried {rf['carried']} {rf['carried_extent_m']:.3f} + jaw {rf['jaw_excess_m']:.3f})")

    def shelf_lift_plan(self, grasp: GraspRecord) -> Dict[str, Any]:
        """(d): ``grasps.lift_turn`` (the upright turn of the leaning book) when the upright book stays under the board above along the lift
        (:func:`lift_headroom_m` >= :data:`LIFT_HEADROOM_MARGIN_M`), and ``grasps.lift_leaned_piece`` = the split-kept piece the book leans on
        (within :data:`LEANED_BOARD_MAX_GAP_M` of its start hull). Kit ``shelf_lift.straightened`` / ``.no_headroom``. The facts.
        (d) fix 2026-11-08: the up direction is the book's standing axis (:meth:`shelf_standing_axis`); without one, or on any error here,
        the lift is today's (no turn, no leaned-board touch; kit ``shelf_lift.fallback_no_axis.<why>`` / ``.fallback_error``) - never raises."""
        for k in ("lift_turn", "lift_leaned_piece"):  # only this attempt's plan sets them
            grasp.extras.pop(k, None)
        axis, afacts = self.shelf_standing_axis()
        if axis is None:
            TT._count(self, f"shelf_lift.fallback_no_axis.{afacts.get('why', 'unknown')}")
            return {"fallback": "no_axis", **afacts}
        try:
            return self._shelf_lift_plan(grasp, axis, afacts)
        except Exception as e:  # noqa: BLE001 - as above: the attempt keeps today's lift
            log.warning("shelf_lift_plan: %s: %r (today's lift)", getattr(getattr(self, "gp", None), "variant", "?"), e, exc_info=True)
            grasp.extras.pop("lift_turn", None)
            grasp.extras.pop("lift_leaned_piece", None)
            TT._count(self, "shelf_lift.fallback_error")
            return {"fallback": "error", "error": repr(e)[:200], **afacts}

    def _shelf_lift_plan(self, grasp: GraspRecord, axis: th.Tensor, afacts: Dict[str, Any]) -> Dict[str, Any]:
        facts: Dict[str, Any] = dict(afacts)
        pieces = self._kept_pieces()
        t0 = (G.t(self.shelf_pose[0]).float(), G.t(self.shelf_pose[1]).float())
        hull_w = self.world.hull_at(self.hull["target"], t0)
        top0 = float(hull_w[:, 2].max())
        lo_xy, hi_xy = hull_w[:, :2].min(dim=0).values, hull_w[:, :2].max(dim=0).values
        ceiling, leaned = math.inf, None
        for prim, _cm, geom in pieces:
            for lo, hi in geom.boxes:
                if float(lo[2]) > top0 - 0.02 and bool((lo[:2] < hi_xy).all()) and bool((hi[:2] > lo_xy).all()):
                    ceiling = min(ceiling, float(lo[2]))
            d = float(geom.sdf(hull_w).min())
            if d <= LEANED_BOARD_MAX_GAP_M and (leaned is None or d < leaned[1]):
                leaned = (prim, d)
        up = G.rotate(t0[1], axis)
        turn = shortest_arc(up, th.tensor([0.0, 0.0, 1.0]))
        g2 = type("G2", (), {})()
        g2.grasp, g2.carry_height = grasp.grasp, grasp.carry_height
        g2.extras = dict(grasp.extras, lift_turn=[float(v) for v in turn])
        end = PC.lift_end_pose(g2)
        poses = lift_object_poses(grasp.grasp, end, grasp.eef_in_object)
        head = lift_headroom_m(self.hull["target"], poses, ceiling) if math.isfinite(ceiling) else math.inf
        facts.update(turn_deg=round(math.degrees(2 * math.acos(min(1.0, abs(float(turn[3]))))), 2), ceiling_z=None if not math.isfinite(ceiling) else round(ceiling, 4),
                     headroom_m=None if not math.isfinite(head) else round(head, 4), leaned_piece=None if leaned is None else leaned[0].rsplit("/", 2)[-2:],
                     leaned_gap_m=None if leaned is None else round(leaned[1], 4))
        if head >= LIFT_HEADROOM_MARGIN_M:
            grasp.extras["lift_turn"] = [round(float(v), 6) for v in turn]
            TT._count(self, "shelf_lift.straightened")
        else:
            TT._count(self, "shelf_lift.no_headroom")
        if leaned is not None:
            grasp.extras["lift_leaned_piece"] = leaned[0]
        return facts

    def lift_touch_ignores(self, g: Any) -> List[Any]:
        """(d): the leaned-on board as an ignorable piece for the lift (the program check's lift; :meth:`lift_with_touch`), [] otherwise."""
        ex = getattr(g, "extras", None)
        if ex is None:
            ex = getattr(g, "extra", None)
        prim = (ex or {}).get("lift_leaned_piece") if isinstance(ex, dict) else None
        if not prim or not shelf_lift_straighten_on(self):
            return []
        from ispatialgym.sampling.v2 import support_split as SS

        cm = dict(SS._pieces(self.support)).get(prim)
        return [] if cm is None else [_PieceShim(f"{self.support.name}:leaned_board", prim, cm)]

    def lift_with_touch(self, grasp: GraspRecord, check) -> Any:
        """(d): ``check()`` (the lift test, object attached) as today; when it fails and the switch is on, once more with the leaned-on board out of
        the world, kept only when the attached book's hull stays at clearance >= 0 from that board along the lift (pose 0, the resting lean, down
        to -:data:`LIFT_START_CONTACT_TOL_M`: :func:`lift_touch_clearance_why`) and every robot link is clear of it at margin 0 at the lift's end
        configuration. Kit ``shelf_lift.leaned_board_touch_allowed`` / ``.leaned_board_start_contact_allowed`` (pose 0 < 0, within the tolerance) /
        ``.leaned_board_touch_refused.<why>`` (ik / penetration / penetration_start / robot_link)."""
        r = check()
        shim = self.lift_touch_ignores(grasp)
        if getattr(r, "ok", False) or not shim:
            return r
        with self.ikc.ignoring(shim, split=False):
            r2 = check()
        if not r2.ok or r2.q is None:
            TT._count(self, "shelf_lift.leaned_board_touch_refused.ik")
            return r
        geom = self.ikc._pieces_geometry(self.support, frozenset([grasp.extras["lift_leaned_piece"]]))
        poses = lift_object_poses(grasp.grasp, PC.lift_end_pose(grasp), grasp.eef_in_object)
        cls = [float(geom.sdf(G.local_points_world(self.hull["target"], (G.t(p[0]).float(), G.t(p[1]).float()))).min()) for p in poses] if geom else []
        why = lift_touch_clearance_why(cls)
        if why is not None:
            TT._count(self, f"shelf_lift.leaned_board_touch_refused.{why}")
            log.info("SHELF_LIFT_TOUCH refused %s cl0 %s cl_rest %s", why, round(cls[0], 6) if cls else None, round(min(cls[1:]), 6) if len(cls) > 1 else None)
            return r
        links = [ln for ln in self.ikc._non_finger_links(grasp.arm) if "attached" not in ln]
        from ispatialgym.oracle import r1pro_model as M

        links += list(M.FINGER_LINKS[grasp.arm])
        with self.ikc.ignoring(self.support_objects()):
            ok_links = bool(self.kit.links_clear(r2.q, links, 0.0)[0])
        if not ok_links:
            TT._count(self, "shelf_lift.leaned_board_touch_refused.robot_link")
            return r
        TT._count(self, "shelf_lift.leaned_board_touch_allowed")
        if cls[0] < -LIFT_TOUCH_CLEARANCE_TOL_M:  # allowed with the resting contact at pose 0 (within LIFT_START_CONTACT_TOL_M)
            TT._count(self, "shelf_lift.leaned_board_start_contact_allowed")
        return r2

    def lifted_book_gap(self, manip: Stance, grasp: GraspRecord) -> float:
        """(c): the free gap (m) along the shelf push direction between the book at the lift's end (held in the hand: ``lift_end_pose`` o
        T_obj->eef^-1) and the partner at its start - how far the base may carry it toward the partner."""
        u = self.shelf_push_direction(manip)[:2]
        end = PC.lift_end_pose(grasp)
        obj = G.compose((G.t(end[0]).float(), G.t(end[1]).float()), G.invert((G.t(grasp.eef_in_object[0]).float(), G.t(grasp.eef_in_object[1]).float())))
        t_w = self.world.hull_at(self.hull["target"], obj)
        p_now = self.obj("partner").get_position_orientation()
        p_w = self.world.hull_at(self.hull["partner"], (p_now[0].float(), p_now[1].float()))
        return float((p_w[:, :2] @ u).min() - (t_w[:, :2] @ u).max())

    def shelf_release_in_gap_plan(self, manip: Stance, grasp: GraspRecord) -> List[Tuple[float, Stance]]:
        """(c): ``[(drive m, stance)]`` - the manipulation stance moved along the edge toward the partner by the fractions
        :data:`SHELF_RELEASE_DRIVE_FRACTIONS` of the room the free gap leaves (gap - ``carry_path_margin_m``), same stand-off / trunk, as the
        push stance is built (``push_stance_for``: focus, the shelf push arm, ``lateral_shift_m``). [] when there is no room (counted)."""
        room = self.lifted_book_gap(manip, grasp) - float(self.gen("carry_path_margin_m", 0.01))
        self._release_gap_facts = {"room_m": round(room, 4)}
        if room <= 0.005:
            return []
        sg = self.shelf_push_sign()
        out = []
        for f in SHELF_RELEASE_DRIVE_FRACTIONS:
            d = float(f) * room
            st = manip.shifted(lateral=float(manip.lateral) + sg * d, name="push").with_trunk(manip.translate, manip.tilt, "push")
            st.focus, st.arms = manip.focus, [self.shelf_push_arm()]
            st.extras.update({"second_base_pose": True, "lateral_shift_m": round(sg * d, 4), "drive_in_gap_m": round(d, 4), "room_m": round(room, 4),
                              "rule": "shelf tip release: the base drives the lifted book toward the partner inside the free gap (i7-t03 round 2 (c))"})
            out.append((d, st))
        return out

    def books_in_frame(self, st: Stance, cam: Optional[G.Pose] = None) -> bool:
        """The shelf books at their start poses - the target AND the partner - fully in ``st``'s head-camera image (5 %), the test the
        segment draw applies to the init image (``TT.in_frame`` of the hull corners)."""
        cam = self.camera_pose_at(st) if cam is None else cam
        return all(TT.in_frame(self, cam, TT.hull_corners_world(self.hull[role], pose))
                   for role, pose in (("target", self.shelf_pose), ("partner", self.shelf_partner_pose)) if pose is not None)

    def shelf_free_start(self, rng: np.random.Generator, init: Stance) -> Stance:
        """i7-t03 round 2 item (a) (:data:`SHELF_FREE_START_ENV`, default off): the edge init stance when it frames both books (counter
        ``free_start.edge_ok``), else IORSLarge's free-floor start (``iors_large.free_floor_stance``: 0.6-2.5 m from the target, facing it,
        ``generation.free_start``) with the same frame test - the target AND the partner in the start image - as the FALLBACK (counters
        ``free_start.used`` / ``.failed.<why>``; the edge stance stays when none is found). The manipulation candidates are the edge
        cell's; the drive to the grasp stance is the path + the shelf back-off arrival (``core.record_stance_travel``)."""
        if self.shelf_pose is None or self.shelf_partner_pose is None:
            return init
        if self.books_in_frame(init):
            TT._count(self, "free_start.edge_ok")
            return init
        from ispatialgym.sampling.v2.tasks.iors_large import free_floor_stance

        t_xy = G.t(self.shelf_pose[0])[:2].float()
        focus = 0.5 * (G.t(self.shelf_pose[0]).float() + G.t(self.shelf_partner_pose[0]).float())
        view_ok = lambda st_, cam: self.books_in_frame(st_, cam)  # noqa: E731
        if shelf_free_start_connected_on(self):  # i7-t03both r2: only a start the drive can leave from (the candidates' region)
            ref = [c for c in (self.manip_candidates or []) if TRAVEL_DROPPED not in c.extras][:1] or [init]
            comp = self.free_component(ref[0].base_xy)
            if comp is not None:
                inner = view_ok

                def view_ok(st_, cam, _inner=inner, _comp=comp):
                    if self.free_component(st_.base_xy) != _comp:
                        TT._count(self, "free_start_connected.refused")
                        return False
                    return _inner(st_, cam)
        try:
            st = free_floor_stance(self, t_xy, focus, view_ok, rng, what="the partner")
        except Reject as e:
            TT._count(self, "free_start.failed.no_stance")
            log.info("%s: free start (shelf): %s - the edge init stance stays", self.gp.variant, e)
            return init
        TT._count(self, "free_start.used")
        st.name = "init"
        st.extras["free_start_from_edge_lateral_m"] = round(float(init.lateral), 4)
        return st

    def free_component(self, xy) -> Optional[int]:
        """i7-t03both r2: the connected region (label) of the eroded traversability map the base position ``xy`` belongs to - its nearest free
        cell within 20 cells, as ``navigation.plan_path`` snaps the ends of a global path; None without one. The labels are computed once."""
        from ispatialgym.sampling.v2 import navigation as NAV

        lab = self.__dict__.get("_free_labels")
        if lab is None:
            free = NAV.eroded_map(self.kit).cpu().numpy().astype(bool)
            try:
                from scipy import ndimage

                lab = ndimage.label(free)[0]
            except ImportError:  # 4-connected flood fill
                lab = np.zeros(free.shape, dtype=np.int32)
                n = 0
                for i0, j0 in zip(*np.nonzero(free)):
                    if lab[i0, j0]:
                        continue
                    n += 1
                    stack = [(i0, j0)]
                    lab[i0, j0] = n
                    while stack:
                        i, j = stack.pop()
                        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not lab[a, b]:
                                lab[a, b] = n
                                stack.append((a, b))
            self._free_labels = lab
            self._free_map = th.as_tensor(lab > 0)
        c = NAV.nearest_free_cell(self._free_map, NAV._to_map(self.kit, xy), 20)
        return None if c is None else int(lab[c])

    def _raw_map(self, arm: str) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
        """The working arm's interior map behind its band view (json, npz arrays), cached; None for a band that is not a map view."""
        band = self.band_of_arm(arm)
        path = (getattr(band, "meta", None) or {}).get("map_path")
        if not path:
            return None
        cache = self.__dict__.setdefault("_diag_maps", {})
        if path not in cache:
            js = path if str(path).endswith(".json") else str(path)[:-4] + ".json"
            with open(js) as f:
                m = json.load(f)
            cache[path] = (m, dict(np.load(js[:-5] + ".npz")))
        return cache[path]

    @staticmethod
    def _map_trunk_key(stand_off: float, translate: float, tilt: float) -> str:
        return f"s{float(stand_off):.2f}_t{float(translate):.2f}_k{float(tilt):+.1f}"

    def map_book_feasible(self, st: Stance, arm: str) -> Optional[bool]:
        """i7-t03both r2 (:data:`SHELF_MAP_STANCE_ENV`): whether the arm's interior map holds a feasible grasp row at the cell under the
        book (the cell nearest its base-frame position, within half the lattice) for ``st``'s trunk at the nearest map stand-off - False also
        when that trunk is not in the map; None without a raw map (no verdict)."""
        rm = self._raw_map(arm)
        if rm is None or self.shelf_pose is None:
            return None
        m, z = rm
        eds = [float(v) for v in m.get("edge_distances") or []]
        so = float(st.stand_off if st.stand_off is not None else 0.0)
        if not eds:
            return None
        de = min(eds, key=lambda v: abs(v - so))
        key = self._map_trunk_key(de, st.translate, st.tilt)
        if key not in z:
            return False
        x, y = self._diag_base_frame(st, G.t(self.shelf_pose[0]))  # the book's base-frame xy (the map's cell frame)
        cells = np.asarray(z["cells"], dtype=float)
        lat = float((m.get("config") or {}).get("lattice_m") or 0.08)
        d = np.abs(cells - np.array([x + (de - so), y])[None, :]).max(axis=1)
        i = int(np.argmin(d))
        if d[i] > 0.5 * lat + 1e-6:
            return False
        return bool((np.asarray(z[key])[i] == 1).any())

    def shelf_map_stances(self, cand: Stance, arm: str) -> List[Tuple[str, Stance, float]]:
        """i7-t03both r2 (:data:`SHELF_MAP_STANCE_ENV`): ``[("map", stance, shift_m)]`` - ``cand`` moved along the edge and to each map stand-off
        so the fixed start's book lands on a cell of the arm's interior map with a feasible grasp row, at that map trunk: the book's depth fixes
        the map row (x = its depth behind the front plane + the stand-off, the nearest lattice row within half a lattice), each feasible cell
        of that row gives the lateral (the base moves by the book's offset from the cell). Ordered by the trunk's distance to ``cand``'s, then
        the lateral move; at most :data:`SHELF_MAP_STANCE_MAX`. [] without a raw map."""
        rm = self._raw_map(arm)
        if rm is None or self.shelf_pose is None or cand.edge is None:
            return []
        m, z = rm
        cells = np.asarray(z["cells"], dtype=float)
        lat = float((m.get("config") or {}).get("lattice_m") or 0.08)
        x, y = self._diag_base_frame(cand, G.t(self.shelf_pose[0]))  # the book's base-frame xy (the map's cell frame)
        so0 = float(cand.stand_off if cand.stand_off is not None else 0.0)
        out = []
        for key, a in z.items():
            mt = re.match(r"s([0-9.]+)_t([0-9.]+)_k([+-][0-9.]+)$", key)
            if not mt:
                continue
            so, tr, ti = float(mt.group(1)), float(mt.group(2)), float(mt.group(3))
            xb = x + (so - so0)  # the book's base-frame depth at that stand-off
            feas = (np.asarray(a).reshape(len(cells), -1) == 1).any(axis=1)
            for i in np.nonzero(feas & (np.abs(cells[:, 0] - xb) <= 0.5 * lat + 1e-6))[0]:
                shift = y - float(cells[i, 1])  # moving the base +d along the tangent moves the book -d in the robot's frame
                dt = abs(tr - float(cand.translate)) + abs(ti - float(cand.tilt))
                out.append((dt, abs(shift) + abs(so - so0), so, tr, ti, shift, float(cells[i, 1])))
        out.sort(key=lambda r: (r[0], r[1]))
        res = []
        for dt, _, so, tr, ti, shift, yc in out[:SHELF_MAP_STANCE_MAX]:
            st = cand.shifted(stand_off=so, lateral=float(cand.lateral) + shift, name="manipulation").with_trunk(tr, ti, "manipulation")
            st.extras["map_stance"] = {"cell_y": round(yc, 3), "stand_off": so, "trunk": [tr, ti]}
            if dt > 1e-6:
                st.extras["stance_shift_trunk"] = [tr, ti]
            res.append(("map", st, round(shift, 4)))
        TT._countn(self, "map_stance.generated", len(res))
        return res

    def shelf_shift_stances(self, cand: Stance, arm: str) -> List[Tuple[str, Stance, float]]:
        """``[(label, stance, shift_m), ...]`` the stance shift tries for the manipulation candidate ``cand`` (its trunk, the working ``arm``):
        the COMPUTED stance first - ``cand`` moved along the edge so the fixed start's book lies at ``generation.shelf_stance_book_lateral_m``
        (:data:`SHELF_STANCE_BOOK_LATERAL_M`; mirrored for the right arm) from the robot's centre line, as ``push_stance_for`` places the push
        contact, same stand-off and trunk - then the scan: ``generation.shelf_stance_scan_shifts_m`` (relative to the computed lateral) x
        ``tabletop.STAND_OFFS`` (``cand``'s stand-off first), the computed stance not repeated. ``shift_m`` = the move along the edge from
        ``cand`` (m, + = the robot's left). Pure geometry: nothing is checked here (:meth:`shelf_stance_ok`).
        i7-t03 2026-11-08 (owner 5p.2, under (b)'s :data:`SHELF_JOINT_STANCE_ENV`): every base position is tried at ``cand``'s trunk first, then
        at the other trunks of the attempt's manipulation candidates (``manip_candidates``, distinct (translate, tilt), their order) - the
        computed position's other trunks before the scan; only ``cand``'s trunk at the computed position is labelled ``computed``."""
        if self.shelf_pose is None or cand.edge is None:
            return []
        mapped = self.shelf_map_stances(cand, arm) if shelf_map_stance_on(self) else []
        if mapped:
            return mapped + self._shelf_shift_stances(cand, arm)
        return self._shelf_shift_stances(cand, arm)

    def _shelf_shift_stances(self, cand: Stance, arm: str) -> List[Tuple[str, Stance, float]]:
        """:meth:`shelf_shift_stances` without the map stances (the computed stance, then the scan)."""
        want = float(self.gen("shelf_stance_book_lateral_m", SHELF_STANCE_BOOK_LATERAL_M) or SHELF_STANCE_BOOK_LATERAL_M)
        want = want if arm == "left" else -want
        _, lat_book = cand.local(G.t(self.shelf_pose[0])[:2])
        dlat = float(lat_book) - want  # moving the base +d along the tangent moves the book -d in the robot's frame
        so0 = float(cand.stand_off if cand.stand_off is not None else TT.STAND_OFF)
        sos = [so0] + [float(v) for v in TT.STAND_OFFS if abs(float(v) - so0) > 1e-6]
        shifts = [float(v) for v in (self.gen("shelf_stance_scan_shifts_m") or SHELF_STANCE_SCAN_SHIFTS_M)]
        own = (round(float(cand.translate), 4), round(float(cand.tilt), 4))
        trunks = [own]
        if shelf_joint_stance_on(self):  # 5p.2: the trunk set of every candidate
            for c in getattr(self, "manip_candidates", None) or []:
                tk = (round(float(c.translate), 4), round(float(c.tilt), 4))
                if tk not in trunks:
                    trunks.append(tk)
        out: List[Tuple[str, Stance, float]] = []
        seen = set()
        for sh in [0.0] + shifts:
            for so in sos:
                key = (round(sh, 4), round(so, 4))
                if key in seen:
                    continue
                seen.add(key)
                for tk in trunks:
                    st = cand.shifted(stand_off=so, lateral=float(cand.lateral) + dlat + sh, name="manipulation")
                    if tk != own:
                        st = st.with_trunk(tk[0], tk[1], "manipulation")
                        st.extras["stance_shift_trunk"] = [tk[0], tk[1]]
                    out.append(("computed" if not out else "scan", st, round(dlat + sh, 4)))
        return out

    def _shelf_camera_rule(self, cand: Stance) -> Tuple[float, Any]:
        """``(max angle rad, boxed(stance, cam, focus))`` - Phase 2's work-camera rule for the class ``cand`` was admitted under: the split
        cone's work class (``phase2_angle_classes``) for a split-cone survivor (``init_view_angle`` in its extras), else the plain Phase 2's
        loose class (the split cone's ``fallback_classes_deg`` when it has them); the box = the target's half extent / height around the focus
        (``default_in_frame``, as Phase 2 without an ``in_frame_fn``)."""
        classes = TT.phase2_angle_classes(self)
        opts = TT.split_cone_opts(self)
        if "init_view_angle" not in cand.extras and opts is not None and opts.get("fallback_classes_deg"):
            classes = opts["fallback_classes_deg"]
        half = TT.half_extent(self.hull["target"], self.rest["target"])
        height = TT.height_of(self.hull["target"], self.rest["target"])

        def boxed(st, cam, focus) -> bool:
            return bool(TT.default_in_frame(self, st, cam, focus, None, float(half), float(height), TT.IN_FRAME_MARGIN))

        return math.radians(float(classes[1])), boxed

    def shelf_stance_ok(self, st: Stance, cand: Stance, arm: str) -> Tuple[Optional[Stance], str]:
        """The stance-level checks of a shifted stance, as Phase 2 checked ``cand``: the base on the free floor, whole-body collision-free with
        ``arm`` working and the other arm in its first collision-free idle posture (``stance_for_arm``), and the work camera's rule
        (:meth:`_shelf_camera_rule`) at the fixed start (the focus = the book's xy at ``cand``'s focus height). ``(stance, "")`` or
        ``(None, "floor" | "collision" | "camera")``."""
        if not self.kit.trav_free(st.base_xy):
            return None, "floor"
        st.arms, st.idle_posture = "untucked", None  # stance_for_arm checks (and idles the other arm) only a stance not yet working ``arm``
        st_arm = TT.stance_for_arm(self, st.with_trunk(st.translate, st.tilt, "manipulation"), arm)
        if st_arm is None:
            return None, "collision"
        TT.adopt_arm(st, st_arm)
        fz = float(cand.focus[2]) if cand.focus is not None else float(G.t(self.shelf_pose[0])[2])
        bx = G.t(self.shelf_pose[0])
        focus = th.tensor([float(bx[0]), float(bx[1]), fz])
        max_ang, boxed = self._shelf_camera_rule(cand)
        cam = TT.stance_cameras(self, [st])[0]
        ang = float(self.world.camera_model.angle_off_axis(cam, focus))
        if ang > max_ang or not boxed(st, cam, focus):
            return None, "camera"
        st.focus = focus
        st.extras["axis_angle"] = ang
        return st, ""

    def shelf_release_why(self, st: Stance, grasp: GraspRecord) -> Optional[str]:
        """i7-t03 round 2 (b): the tip release from ``st`` at the DRAFTED goal - ``goal_pose`` of the fixed start poses (the AABB-contact lean
        the goal check settles from; it depends on the stance only through the edge frame) - by the goal check's own place test
        (``place_infeasible_why``: pre-place -> place descent with the book attached, then the clear pose). None = it passes."""
        prev = self.sidecar.stances.get("manipulation")
        side = getattr(self, "goal_side", None)
        self.sidecar.stances["manipulation"] = st
        try:
            goal = self.goal_pose(self.shelf_pose, self.shelf_partner_pose)
        except Reject as e:
            return f"goal: [{e.reason}] {e}"
        finally:
            if prev is None:
                self.sidecar.stances.pop("manipulation", None)
            else:
                self.sidecar.stances["manipulation"] = prev
            self.goal_side = side
        return self.place_infeasible_why(st, (G.t(goal[0]).float(), G.t(goal[1]).float()), grasp)

    def _joint_ok(self, st: Stance, grasp: GraspRecord) -> bool:
        why = self.shelf_release_why(st, grasp)
        grasp.extras["joint_release"] = "ok" if why is None else why[:160]
        return why is None

    def shelf_pregrasp_frame_check(self, st: Stance, grasp: GraspRecord) -> None:
        """Round 7 (:data:`SHELF_PREGRASP_FRAME_ENV`): the goal check's frame test of the pre-grasp and grasp (:meth:`require_in_frame`, the
        customs' ``grasp_frame_rule``, 5 % margin) at ``st``'s camera, when the fixed start's grasp is chosen there. Raises
        ``Reject("gripper_not_in_frame", SHELF_PREGRASP_FRAME_TAG ...)`` (:func:`shelf_need_of` -> ``"frame"``: the stance shift tries the other
        base positions / stand-offs)."""
        TT._count(self, "shelf_pregrasp_frame.checked")
        try:
            self.require_in_frame(self.camera_pose_at(st), {"pre-grasp": grasp.pre_grasp, "grasp": grasp.grasp})
        except Reject as e:
            TT._count(self, "shelf_pregrasp_frame.refused")
            raise Reject("gripper_not_in_frame", f"{SHELF_PREGRASP_FRAME_TAG} {str(e).split('] ', 1)[-1]} at stand-off {float(st.stand_off or 0.0):.2f}")
        TT._count(self, "shelf_pregrasp_frame.passed")

    def shelf_fixed_start(self, init: Stance, cand: Stance, manip: Stance, arm: str, try_fn) -> Tuple[Stance, GraspRecord]:
        """The fixed start's checks at ``manip`` (``try_fn(stance) -> GraspRecord``: the init -> manipulation travel, ``grasp_at_pose`` with
        every grasp and twin, the transit check, the back-off - :meth:`sample_instance`'s own; ``Reject`` on failure). With the stance shift on
        (:data:`SHELF_STANCE_SHIFT_ENV`, i7-t03 item 1) a failure the base position can change (:func:`shelf_need_of`) tries the stances of
        :meth:`shelf_stance_shift` (the same ``try_fn``); a candidate that passes is never shifted. Returns ``(stance, grasp)``."""
        joint = shelf_joint_stance_on(self) and self.shelf_pose is not None
        if shelf_map_stance_on(self) and shelf_stance_shift_on(self) and self.shelf_pose is not None:
            ok = self.map_book_feasible(manip, arm)
            TT._count(self, f"map_stance.candidate.{'none' if ok is None else ('feasible' if ok else 'infeasible')}")
            if ok is False:  # i7-t03both r2: the map has no grasp at the book from this candidate - straight to the (map) stance shift
                return self.shelf_stance_shift(init, cand, manip, arm, Reject("grasp_unreachable", "map: no feasible grasp row at the book's cell from this "
                                                                                 "candidate's trunk / stand-off"), "grasp", try_fn)
        try:
            g0 = try_fn(manip)
        except Reject as e:
            need = shelf_need_of(e)
            if not shelf_stance_shift_on(self) or self.shelf_pose is None or need is None:
                raise
            return self.shelf_stance_shift(init, cand, manip, arm, e, need, try_fn)
        if not joint:
            return manip, g0
        if self._joint_ok(manip, g0):  # i7-t03 round 2 (b): the candidate grasps AND releases - no carried drive
            TT._count(self, "stance_shift.both_passed")
            return manip, g0
        if not shelf_stance_shift_on(self):
            TT._count(self, "stance_shift.grasp_only")
            return manip, g0
        # the candidate grasps but does not release: a shifted stance that does both, else the candidate (its travel recorded again)
        try:
            st, g1 = self.shelf_stance_shift(init, cand, manip, arm, Reject("grasp_unreachable", "release: " + str(g0.extras.get("joint_release"))), "release",
                                             try_fn, require_both=True)
            return st, g1
        except Reject:
            TT._count(self, "stance_shift.grasp_only")
            self.sidecar.transitions.clear()
            return manip, try_fn(manip, again=True)

    def shelf_stance_shift(self, init: Stance, cand: Stance, manip: Stance, arm: str, err: Reject, need: str, try_fn,
                           require_both: bool = False) -> Tuple[Stance, GraspRecord]:
        """i7-t03 item 1: the shifted stances of :meth:`shelf_shift_stances` in order - the computed stance, then the scan (at most
        ``generation.shelf_stance_scan_max`` scan stances per attempt, :data:`SHELF_STANCE_SCAN_MAX`; a stance tried earlier in the attempt is
        not tried again) - each through :meth:`shelf_stance_ok` and ``try_fn`` (the same exact checks as ``manip``); the first that passes is
        the manipulation stance (``extras["stance_shift"]``: offset, label, need, reach before / after and the witness branch). Kit counters
        ``stance_shift.need.<grasp|transit>``, ``.computed_tried`` / ``.computed_passed``, ``.scan_tried`` / ``.scan_passed``,
        ``.rejected.<floor|collision|camera|check>``, ``.offset.<+0.00>``. Raises ``err`` when none passes."""
        TT._count(self, f"stance_shift.need.{need}")
        m0 = re.search(r"lies ([0-9.]+) rad", str(err))
        reach0 = float(m0.group(1)) if m0 else None
        budget = self.__dict__.setdefault("_stance_shift_budget", {"scan": 0, "seen": set()})
        scan_max = int(self.gen("shelf_stance_scan_max", SHELF_STANCE_SCAN_MAX) or 0)
        tried: List[str] = []
        fallback = None  # (b): the first grasp-only stance
        for label, st, shift in self.shelf_shift_stances(cand, arm):
            key = (tuple(round(float(v), 3) for v in st.base_pose2d), round(float(st.translate), 3), round(float(st.tilt), 3), arm)
            if key in budget["seen"]:
                continue
            joint = shelf_joint_stance_on(self)  # 5p.2: the budget counts the stances that reach the exact checks (the trunk set adds cheap rejects)
            if label == "scan" and budget["scan"] >= scan_max:
                break
            if label == "map" and budget.setdefault("map", 0) >= SHELF_MAP_STANCE_MAX:
                continue
            budget["seen"].add(key)
            if label == "scan" and not joint:
                budget["scan"] += 1
            TT._count(self, f"stance_shift.{label}_tried")
            other_trunk = st.extras.get("stance_shift_trunk")
            if other_trunk is not None:
                TT._count(self, "stance_shift.trunk.other_tried")
            st.extras.update(cand.extras)
            if other_trunk is not None:
                st.extras["stance_shift_trunk"] = other_trunk
            st.extras.pop("_position", None)  # the shifted stance's travel is the axis-aligned leg (core.record_stance_travel)
            st.extras["stance_shift"] = {"label": label, "shift_m": round(float(shift), 4), "stand_off": round(float(st.stand_off or 0.0), 3), "need": need,
                                         "from_lateral": round(float(cand.lateral), 4), "reach_before_rad": reach0, "branch_before": reach_branch(reach0)}
            st_ok, why = self.shelf_stance_ok(st, cand, arm)
            if st_ok is None:
                TT._count(self, f"stance_shift.rejected.{why}")
                tried.append(f"{label} {shift:+.2f}/{float(st.stand_off or 0.0):.2f}: {why}")
                continue
            if label == "scan" and joint:
                budget["scan"] += 1
            if label == "map":
                budget["map"] = budget.get("map", 0) + 1
            try:
                grasp = try_fn(st_ok)
            except Reject as e2:
                TT._count(self, "stance_shift.rejected.check")
                self.drop_travel_records(init.name, st_ok.name)
                tried.append(f"{label} {shift:+.2f}/{float(st_ok.stand_off or 0.0):.2f}: [{e2.reason}] {str(e2)[:90]}")
                continue
            if shelf_joint_stance_on(self):  # i7-t03 round 2 (b): a stance that grasps AND releases first; a grasp-only one is the fallback
                if self._joint_ok(st_ok, grasp):
                    TT._count(self, "stance_shift.both_passed")
                    st_ok.extras["stance_shift"]["release"] = "ok"
                else:
                    st_ok.extras["stance_shift"]["release"] = grasp.extras.get("joint_release")
                    if require_both:
                        TT._count(self, "stance_shift.rejected.release")
                        self.drop_travel_records(init.name, st_ok.name)
                        tried.append(f"{label} {shift:+.2f}: release {str(grasp.extras.get('joint_release'))[:80]}")
                        continue
                    if fallback is None:
                        fallback = (st_ok, grasp, label, shift)
                    self.drop_travel_records(init.name, st_ok.name)
                    tried.append(f"{label} {shift:+.2f}: grasp only")
                    continue
            facts = grasp.extras.get("shelf_transit") or {}
            reach1 = facts.get("reach_rad")
            st_ok.extras["stance_shift"].update(reach_after_rad=reach1, branch_after=reach_branch(reach1), grasp=grasp.grasp_name,
                                                trunk=[round(float(st_ok.translate), 3), round(float(st_ok.tilt), 3)], own_trunk=other_trunk is None)
            if other_trunk is not None:
                TT._count(self, "stance_shift.trunk.other_passed")
            TT._count(self, f"stance_shift.{label}_passed")
            TT._count(self, f"stance_shift.offset.{shift:+.2f}")
            log.info("%s: stance shift (%s, need %s): base moved %+.3f m along the edge at stand-off %.2f (book lateral %+.3f), grasp %s; reach %s (%s) -> %s (%s)%s",
                     self.gp.variant, label, need, shift, float(st_ok.stand_off or 0.0), float(st_ok.local(G.t(self.shelf_pose[0])[:2])[1]), grasp.grasp_name,
                     "-" if reach0 is None else f"{reach0:.2f}", reach_branch(reach0), "-" if reach1 is None else f"{float(reach1):.2f}", reach_branch(reach1),
                     f"; tried before: {'; '.join(tried)}" if tried else "")
            return st_ok, grasp
        if fallback is not None:  # (b): no stance grasps AND releases - the first grasp-only stance (its travel and checks again)
            st_f, _g, label, shift = fallback
            TT._count(self, "stance_shift.grasp_only")
            self.sidecar.transitions.clear()
            grasp = try_fn(st_f, again=True)
            TT._count(self, f"stance_shift.{label}_passed")
            TT._count(self, f"stance_shift.offset.{shift:+.2f}")
            return st_f, grasp
        log.info("%s: stance shift (need %s) found no stance for candidate at lateral %+.3f: %s", self.gp.variant, need, float(cand.lateral), "; ".join(tried) or "none left")
        raise err

    def _fixed_start_memo_key(self, manip: Stance, arm: str, fingers: Optional[float], per_grasp: bool) -> Tuple[Any, ...]:
        """The key of the fixed start's grasp verdict at ``manip`` (``generation.fixed_start_memo``): the base pose, trunk and locked
        configuration (both arms' postures), the working arm, the pre-shaped opening and whether the transit is judged per grasp; the start
        pose and the grasp library are the template's (with ``generation.shelf_start_draw`` the attempt's start pose joins the key)."""
        q = self.ikc.q_locked(manip)
        key = (tuple(round(float(v), 3) for v in manip.base_pose2d), round(float(manip.translate), 3), round(float(manip.tilt), 3),
               tuple(round(float(v), 3) for v in G.t(q).flatten().tolist()), arm, None if fingers is None else round(float(fingers), 4), bool(per_grasp))
        if self.shelf_backoff_m() > 0.0 or self.gen("shelf_transit_max_reach_rad") is not None:  # oracle-fix 2026-11-06: another verdict
            key = key + ("backoff", round(self.shelf_backoff_m(), 4), self.gen("shelf_transit_max_reach_rad"))
        if shelf_draw_on(self) and self.shelf_pose is not None:  # oracle-fix 2026-11-06 (shelf_start_draw): the start is the attempt's
            key = key + ("start", tuple(round(float(v), 3) for v in [*G.t(self.shelf_pose[0]).tolist(), *G.t(self.shelf_pose[1]).tolist()]))
        if getattr(self, "_board_keys", None):  # i7-t03both: the attempt's lean rest (its grasps) is drawn too
            key = key + ("grasp_key", self.grasp_key)
        return key

    def _fixed_start_memo_persisted(self, key: Tuple[Any, ...], why: Optional[str] = None) -> Optional[str]:
        """Opt-in ``generation.fixed_start_memo_persist`` (oracle-fix 2026-11-05, speed): :meth:`_fixed_start_memo_key`'s failures in the
        sampling cache (job_val.sh SHARED_SAMPLING_CACHE: shared by the jobs of one alias), keyed with the IK cache scope, the variant, the
        alias's whole generation block and the template's start pose. With ``why``: stores it; without: the stored failure or None."""
        if not bool(self.gen("fixed_start_memo_persist") or False) or self.shelf_pose is None:
            return None
        from ispatialgym.sampling.cache import cache_get, cache_key, cache_put

        gen = {k: v for k, v in (getattr(self.gp, "generation", None) or {}).items() if not str(k).startswith("_")}
        pose = [G.t(self.shelf_pose[0]).tolist(), G.t(self.shelf_pose[1]).tolist()]
        ck = cache_key("fixed_start_fail_v1", {"scope": getattr(self.ikc, "cache_scope", {}), "variant": self.gp.variant, "generation": gen, "start": pose, "key": list(key)})
        if why is not None:
            cache_put(ck, {"why": str(why)[:900]})
            return why
        hit = cache_get(ck)
        return None if not isinstance(hit, dict) else hit.get("why")

    def shelf_transit_why(self, manip: Stance, grasp: GraspRecord, arm: str, fingers: Optional[float] = None) -> Optional[str]:
        """Opt-in ``generation.shelf_transit_check`` (oracle-fix 2026-11-04 round 4, touching#03: 8 / 8 demos failed the transit to the shelf
        pre-grasp - the straight ramp into bookcase_opryhp_1 / the support, via "ready" too, and trajopt after it): the oracle's first strategies
        re-checked here - the straight joint ramp from the stance's configuration to the pre-grasp witness, else the two ramps via the "ready"
        arm posture (``r1pro_model.ARM_READY_Q``), each waypoint penetration-free (self + world); None when one passes, else why.
        With ``generation.shelf_backoff_transit_m`` / ``shelf_transit_max_reach_rad`` (oracle-fix 2026-11-06): :meth:`_shelf_backoff_transit_why`."""
        from ispatialgym.oracle import r1pro_model as M

        if self.shelf_backoff_m() > 0.0 or self.gen("shelf_transit_max_reach_rad") is not None:
            return self._shelf_backoff_transit_why(manip, grasp, arm, fingers)
        kw = {"fingers": fingers} if fingers is not None else {}
        r = self.ikc.exact([grasp.pre_grasp], manip, arm, cache=False, **kw)
        if not r.ok or r.q is None:
            return f"transit check: no pre-grasp witness ({r.reason})"
        q0 = self.ikc.q_locked(manip)
        q1 = G.t(r.q).to(q0.dtype).clone()
        idx = self.kit.robot.arm_control_idx[arm]

        def clear(qa: th.Tensor, qb: th.Tensor) -> bool:
            n = max(8, int(math.ceil(float((qb[idx] - qa[idx]).abs().max()) / 0.02)))
            prof = 0.5 * (1.0 - th.cos(th.linspace(0.0, math.pi, n + 1, dtype=qa.dtype)))
            ramp = qa.unsqueeze(0) + prof.unsqueeze(1) * (qb - qa).unsqueeze(0)
            return bool(self.kit.collision_free(ramp, margin=0.0)[1:].all())

        if clear(q0, q1):
            return None
        q_ready = q0.clone()
        q_ready[idx] = th.tensor(M.ARM_READY_Q[arm], dtype=q0.dtype)
        if clear(q0, q_ready) and clear(q_ready, q1):
            return None
        return f"transit check: the straight {arm}-arm ramp from the stance to the pre-grasp witness penetrates (direct and via ready)"

    # ------------------------------------------------------------------ shelf back-off (oracle-fix 2026-11-06, touching#03)

    def shelf_backoff_m(self) -> float:
        """Opt-in ``generation.shelf_backoff_transit_m`` (oracle-fix 2026-11-06, owner feedback on the touching#03 demo: "instead of swinging
        arm backwards to reach grasp, I prefer robot moving backwards then reaching through forward movement"): the shelf layout's
        init -> manipulation transition drives back to the manipulation stand-off + this, the working arm makes its straight joint ramp to
        the pre-grasp there, then the trunk and the drive-in with the arm frozen (:meth:`record_shelf_backoff`). 0 = off (the transit to
        the pre-grasp is the grasp's own, from the stance)."""
        if self.shelf_start is None:
            return 0.0
        return max(0.0, float(self.gen("shelf_backoff_transit_m", 0.0) or 0.0))

    def _shelf_arrival(self, manip: Stance):
        """The recorded arrival transition into ``manip`` (``record_stance_travel``'s; the back-off replaces it)."""
        tr = next((t for t in self.sidecar.transitions if t.to == manip.name and not t.departure), None)
        if tr is None or self.sidecar.stances.get(tr.frm) is None:
            raise Reject("no_transition", f"shelf back-off: no recorded transition into {manip.name}")
        return tr

    def _shelf_backoff_floor(self, manip: Stance) -> float:
        """The smallest s_T of the backed-off arrival: the manipulation stand-off + ``generation.shelf_backoff_transit_m``, or the recorded
        arrival's s_T when that is larger (a path planned to a drive-back pose farther out: the arm is not carried forward first)."""
        return max(float(manip.stand_off or 0.0) + self.shelf_backoff_m(), float(self._shelf_arrival(manip).stand_off))

    def _shelf_transit_start(self, manip: Stance, s_T: Optional[float], fingers: Optional[float]) -> th.Tensor:
        """The configuration the shelf transit starts from: with the back-off (``s_T``) the drive-back pose ``NAV.departure_pose(manip, s_T)``
        with the trunk the arrival transition starts from (the oracle moves the arms after the drive-back, before the trunk) and the arms
        the robot arrives there with - the departing stance's, or the manipulation stance's posture after a departure transition re-posed
        them; without it the stance's configuration (the grasp's transit). ``fingers``: the pre-shaped opening (None = the stance's)."""
        from ispatialgym.sampling.v2 import navigation as NAV
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        if s_T is None:
            return self.ikc.q_locked(manip, fingers=fingers)
        arr = self._shelf_arrival(manip)
        frm = self.sidecar.stances.get(arr.frm)
        t, k = arr.trunk_from if arr.trunk_from is not None else (frm.translate, frm.tilt)
        r, l, f = stance_arm_vectors(self.kit, manip if self.departure_of(arr.frm, arr.to) is not None else frm)
        x, y, yaw = NAV.departure_pose(manip, float(s_T))
        q = self.kit.q_config(arms="untucked", translate=float(t), tilt=float(k), right_arm=r, left_arm=l, fingers=f if fingers is None else float(fingers))
        return self.kit.q_at_base_pose(q, x, y, yaw)

    def _shelf_backoff_arms(self, manip: Stance, arm: str, q_arm: th.Tensor) -> Dict[str, th.Tensor]:
        """The arms the backed-off transition moves to after its drive-back (its record's ``arms_q``): ``arm`` at the pre-grasp joints
        ``q_arm``, the other arm in the manipulation stance's idle posture."""
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        r, l, _ = stance_arm_vectors(self.kit, manip)
        other = "right" if arm == "left" else "left"
        return {arm: G.t(q_arm).float().reshape(-1).clone(), other: G.t(r if other == "right" else l).float().reshape(-1).clone()}

    def _shelf_ramp_why(self, manip: Stance, arm: str, q_arm: th.Tensor, s_T: Optional[float], fingers: Optional[float]) -> Optional[str]:
        """The straight joint ramp of the arms from :meth:`_shelf_transit_start` to :meth:`_shelf_backoff_arms` (``arm`` at ``q_arm``),
        else the two ramps via the "ready" posture of ``arm`` (``shelf_transit_why``'s rule), each waypoint penetration-free; None
        when one passes, else why."""
        from ispatialgym.oracle import r1pro_model as M

        q0 = self._shelf_transit_start(manip, s_T, fingers)
        q1 = q0.clone()
        idx_of = self.kit.robot.arm_control_idx
        for a, q in self._shelf_backoff_arms(manip, arm, q_arm).items():
            q1[idx_of[a]] = q.to(q0.dtype)
        idx = th.cat([th.as_tensor(idx_of[a]).reshape(-1).long() for a in ("left", "right")])

        def clear(qa: th.Tensor, qb: th.Tensor) -> bool:
            n = max(8, int(math.ceil(float((qb[idx] - qa[idx]).abs().max()) / 0.02)))
            prof = 0.5 * (1.0 - th.cos(th.linspace(0.0, math.pi, n + 1, dtype=qa.dtype)))
            ramp = qa.unsqueeze(0) + prof.unsqueeze(1) * (qb - qa).unsqueeze(0)
            return bool(self.kit.collision_free(ramp, margin=0.0)[1:].all())

        if clear(q0, q1):
            return None
        q_ready = q0.clone()
        q_ready[idx_of[arm]] = th.tensor(M.ARM_READY_Q[arm], dtype=q0.dtype)
        if clear(q0, q_ready) and clear(q_ready, q1):
            return None
        where = "the stance" if s_T is None else f"the drive-back pose at s_T {float(s_T):.2f}"
        return f"transit check: the straight {arm}-arm ramp from {where} to the pre-grasp witness penetrates (direct and via ready)"

    def _shelf_backoff_transit_why(self, manip: Stance, grasp: GraspRecord, arm: str, fingers: Optional[float] = None) -> Optional[str]:
        """:meth:`shelf_transit_why` with the back-off and / or the reach limit (oracle-fix 2026-11-06). The pre-grasp witness (exact IK); with
        ``generation.shelf_transit_max_reach_rad`` a witness farther than that (largest arm-joint change) from the posture the transit
        starts in is re-solved - the chain [pre-grasp, grasp] seeded at that posture, then at ``ARM_READY_Q`` - and rejected when no solution
        is within it (t03tip5_kmuacd_s5: the shoulder-flipped twin [-3.23, 2.63, 2.16, ...] 3.23 rad from untucked, an 11.5 s swing up to
        z 1.90 and 0.47 m behind the base; a near branch 1.37 rad away exists for the same pose); then :meth:`_shelf_ramp_why` - with the
        back-off at the drive-back pose ``s + generation.shelf_backoff_transit_m`` (at the stance the near branch's ramp passes over the
        bookcase top: 55 / 70 ramp configurations within 4 cm of it; backed off 0.15-0.25 m: 0 / 70) - and with the back-off the
        backed-off transition itself (:meth:`_shelf_backoff_transition`). A passing grasp keeps its pre-grasp joints (``_transit_q``), the
        transition (``_backoff_tr``) and the facts (``shelf_transit``); None when it passes, else why."""
        from ispatialgym.oracle import r1pro_model as M

        kw = {"fingers": fingers} if fingers is not None else {}
        r = self.ikc.exact([grasp.pre_grasp], manip, arm, cache=False, **kw)
        if not r.ok or r.q is None:
            return f"transit check: no pre-grasp witness ({r.reason})"
        back = self.shelf_backoff_m()
        s_T = self._shelf_backoff_floor(manip) if back > 0.0 else None
        q0 = self._shelf_transit_start(manip, s_T, fingers)
        idx = self.kit.robot.arm_control_idx[arm]
        q_pre = G.t(r.q).to(q0.dtype)[idx].clone()
        reach = float((q_pre - q0[idx]).abs().max())
        facts: Dict[str, Any] = {"reach_rad": round(reach, 3)}
        reach_max = self.gen("shelf_transit_max_reach_rad")
        if reach_max is not None and reach > float(reach_max):
            found = None
            for label, seed in (("the transit's start posture", q0[idx]), ("ready", th.tensor(M.ARM_READY_Q[arm], dtype=q0.dtype))):
                q_seed = self.ikc.q_locked(manip, fingers=fingers).clone()
                q_seed[idx] = seed.to(q_seed.dtype)
                with TT.record_policy_opening(self, grasp):  # of-v2maps 2026-11-07: the policy at the grasp's pre-shape
                    r2 = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, arm, q_seed=q_seed, cache=False, scale_key=self.choice.models["target"].scale, **kw)
                q2 = self.ikc.solutions.get((arm, self.ikc._pose_key(grasp.pre_grasp))) if r2.ok else None
                if q2 is None:
                    continue
                q2 = G.t(q2).to(q0.dtype)[idx].clone()
                reach2 = float((q2 - q0[idx]).abs().max())
                if reach2 <= float(reach_max):
                    found = (label, q2, reach2)
                    break
            if found is None:
                TT._count(self, "shelf_transit.reach_rejected")
                return (f"transit check: the pre-grasp witness lies {reach:.2f} rad from the transit's start posture (> generation.shelf_transit_max_reach_rad "
                        f"{float(reach_max):.2f}) and no re-solve from that posture / ready is nearer")
            TT._count(self, "shelf_transit.reach_resolved")
            facts = {"reach_rad": round(found[2], 3), "first_reach_rad": round(reach, 3), "resolved_from": found[0]}
            q_pre = found[1]
        why = self._shelf_ramp_why(manip, arm, q_pre, s_T, fingers)
        if why is not None:
            return why
        if back > 0.0:
            tr, why = self._shelf_backoff_transition(manip, arm, q_pre, fingers)
            if tr is None:
                return why
            grasp.extras["_backoff_tr"] = tr
            facts.update({"backoff_m": round(back, 4), "s_T": round(float(tr.stand_off), 4)})
        grasp.extras["_transit_q"] = q_pre
        grasp.extras["shelf_transit"] = facts
        return None

    def _shelf_backoff_transition(self, manip: Stance, arm: str, q_arm: th.Tensor, fingers: Optional[float]):
        """``(Transition, "")`` of the backed-off arrival into ``manip`` - ``NAV.transition`` from the recorded arrival's stances and trunk with
        s_T >= :meth:`_shelf_backoff_floor` (the stand-off + ``generation.shelf_backoff_transit_m``), the sweep and the drive-in with the arms at
        :meth:`_shelf_backoff_arms` (``arms_to_q``) and the pre-shaped ``fingers``, the drive-back from the path's end as recorded; the ramp
        checked again at a larger s_T - or ``(None, why)``. Nothing is recorded."""
        from ispatialgym.sampling.v2 import navigation as NAV

        back = self.shelf_backoff_m()
        arr = self._shelf_arrival(manip)
        frm = self.sidecar.stances.get(arr.frm)
        s_min = self._shelf_backoff_floor(manip)
        tr = NAV.transition(self.kit, frm, manip, min_standoff=s_min, path_end_standoff=self.path_end_standoff(frm, manip), trunk_from=arr.trunk_from,
                            extra_max=max(NAV.TRANSITION_EXTRA_MAX, s_min - float(manip.stand_off or 0.0) + 0.3), blame=self.ikc._collision_blame,
                            arms_to_q=self._shelf_backoff_arms(manip, arm, q_arm),
                            fingers=fingers)
        if tr is None:
            return None, (f"shelf back-off: no s_T >= {s_min:.2f} of {frm.name} -> {manip.name} is swept collision-free with the {arm} arm at its pre-grasp "
                          f"(generation.shelf_backoff_transit_m {back:.2f})")
        if abs(float(tr.stand_off) - s_min) > 1e-6:
            why = self._shelf_ramp_why(manip, arm, q_arm, float(tr.stand_off), fingers)
            if why is not None:
                return None, why
        return tr, ""

    def record_shelf_backoff(self, manip: Stance, grasp: GraspRecord, arm: str, fingers: Optional[float] = None):
        """Opt-in ``generation.shelf_backoff_transit_m`` (oracle-fix 2026-11-06): the arrival transition into ``manip`` replaced by the backed-off
        one :meth:`_shelf_backoff_transit_why` checked for ``grasp`` (checked here when the transit check did not run). Its record carries
        what the oracle runs (``execute_transition``): ``arms_q`` (the working arm's pre-grasp joints, the other arm's idle posture) - the
        arms go there after the drive-back, a straight ramp first (``move_idle_arms``), so the grasp's own transit is ~0 rad - and
        ``preshape_m`` (the fingers pre-shaped before the arms move, the pre-grasp checked so); ``backoff_m``. Returns the transition;
        ``Reject`` when the check fails."""
        if grasp.extras.get("_backoff_tr") is None:
            why = self._shelf_backoff_transit_why(manip, grasp, arm, fingers)
            if why is not None:
                raise Reject("no_transition" if why.startswith("shelf back-off") else "grasp_unreachable", why)
        tr = grasp.extras["_backoff_tr"]
        arr = self._shelf_arrival(manip)
        tr.holding = arr.holding
        tr.extras.update(arr.extras or {})
        tr.extras["arms_q"] = {a: [round(float(v), 5) for v in q.tolist()] for a, q in self._shelf_backoff_arms(manip, arm, grasp.extras["_transit_q"]).items()}
        if fingers is not None:
            tr.extras["preshape_m"] = {arm: round(float(fingers), 5)}
        tr.extras["backoff_m"] = round(self.shelf_backoff_m(), 4)
        tr.extras["backoff_arm"] = arm
        self.sidecar.transitions[self.sidecar.transitions.index(arr)] = tr
        if isinstance(manip.extras.get("init_travel"), dict):
            manip.extras["init_travel"]["transition_stand_off_m"] = round(float(tr.stand_off), 4)
        TT._count(self, "shelf_backoff.recorded")
        log.info("%s: shelf back-off - %s -> %s at s_T %.2f (stand-off %.2f + %.2f), the %s arm to its pre-grasp after the drive-back (reach %.2f rad)",
                 self.gp.variant, arr.frm, manip.name, float(tr.stand_off), float(manip.stand_off or 0.0), self.shelf_backoff_m(), arm,
                 float((grasp.extras.get("shelf_transit") or {}).get("reach_rad", float("nan"))))
        return tr

    def shelf_backoff_witness_why(self) -> Optional[str]:
        """The program check's pre-grasp witness of the backed-off arm (``sidecar.ik_witnesses``, what the oracle's grasp plans to) against the
        joints the backed-off transition carries that arm to (``arms_q``): farther than :data:`SHELF_BACKOFF_WITNESS_GAP_RAD` (largest arm
        joint) the grasp's own transit would be another swing (a witness re-solved in another branch after the settle) - why, else None
        (also without a back-off record)."""
        for tr in self.sidecar.transitions:
            ex = tr.extras or {}
            arm = ex.get("backoff_arm")
            if not ex.get("backoff_m") or arm not in (ex.get("arms_q") or {}):
                continue
            idx = self.kit.robot.arm_control_idx[arm]
            w = next((e for e in (getattr(self.sidecar, "ik_witnesses", None) or []) if e.get("name") == "grasp" and e.get("arm") == arm
                      and isinstance((e.get("targets") or {}).get("pre_grasp"), dict)), None)
            if w is None or w["targets"]["pre_grasp"].get("q") is None:
                return f"shelf back-off: no pre-grasp witness of the {arm} arm's grasp after the backed-off transition"
            q_w = th.as_tensor(w["targets"]["pre_grasp"]["q"], dtype=th.float32).reshape(-1)[idx]
            gap = float((q_w - th.as_tensor(ex["arms_q"][arm], dtype=th.float32)).abs().max())
            if gap > SHELF_BACKOFF_WITNESS_GAP_RAD:
                return (f"shelf back-off: the {arm} arm's pre-grasp witness lies {gap:.2f} rad from the joints the backed-off transition carries the arm to "
                        f"(> {SHELF_BACKOFF_WITNESS_GAP_RAD:.2f}: the grasp's transit would swing it into another branch)")
        return None

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """:meth:`SamplerV2.program_check`, then :meth:`shelf_backoff_witness_why` (oracle-fix 2026-11-06; a no-op without a back-off record)."""
        report = super().program_check(draft)
        why = self.shelf_backoff_witness_why()
        if why is not None:
            raise Reject("program_check_failed", why)
        return report

    def shelf_tip(self) -> bool:
        """Shelf layout with ``generation.shelf_tip_release`` (oracle-fix 2026-11-04 round 4, touching#03: every push class failed at its pre-pose
        inside the 0.33 m compartment from every lateral): the carried book is released at the drafted goal - the AABB-contact lean the goal
        check settles from - and tips onto the partner by itself (``shelf_tip_max_nudges`` 1 mm nudges at most); no push stance, no
        ``contacts.push``; ``grasps[0].place_kind`` :data:`SHELF_TIP_PLACE_KIND`."""
        return self.shelf_start is not None and bool(self.gen("shelf_tip_release", False))

    def shelf_push_sign(self) -> float:
        """The shelf layout's push along the edge tangent, toward the partner leaning against the board opposite the target's: -1
        (the robot's right) for a target leaning against the left board."""
        return -float(self.shelf_start["lean_side"])

    def shelf_push_arm(self) -> str:
        """The arm on the target's side: it grasps, places and pushes the target in the shelf layout (the left arm for a partner on
        the right board), with the right hand's push parameters mirrored."""
        return "left" if self.shelf_push_sign() < 0 else "right"

    def shelf_push_direction(self, st: Stance) -> th.Tensor:
        """The shelf layout's unit push direction (world xyz) from ``st``: :meth:`shelf_push_sign` x the edge tangent."""
        sg = self.shelf_push_sign()
        return th.tensor([sg * float(st.tangent[0]), sg * float(st.tangent[1]), 0.0])

    def touching_points(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> Tuple[th.Tensor, th.Tensor]:
        """§6.11 touching: the push contact on the target's right face (mid height) and the brace point on the partner's far
        (left) face (mid height), world xyz, from the target / partner poses. Shelf layout: the push contact on the target's face
        away from the partner along the shelf push direction (the left face for a push to the right) and the partner's far face
        on the other side (no brace uses it)."""
        if self.shelf_start is not None:
            d = self.shelf_push_direction(manip)
            return (push_face_point(self.world.hull_at(self.hull["target"], t_pose), d),
                    push_face_point(self.world.hull_at(self.hull["partner"], p_pose), -d))
        # i7-contact 2026-11-07: the rule the touch_push / touch_brace maps share (contact_poses.touching_points)
        return CPZ.touching_points(self.world.hull_at(self.hull["target"], t_pose), self.world.hull_at(self.hull["partner"], p_pose), manip.tangent)

    # ------------------------------------------------------------------ standing target (inside, owner 2026-10-11, inside#03)

    def up_axis(self) -> Optional[str]:
        """Inside: the model axis the drawn target stands on in its partner - ``generation.target_up_axis`` in the per-model dict form
        (``pair_rules.target_up_axis``: the sunflower's stem end -> head). None for every other relation and without the key."""
        from ispatialgym.sampling import pair_rules as PR

        if self.relation != "inside" or self.gen("target_up_axis") is None:
            return None
        t = self.choice.models["target"]
        try:
            return PR.target_up_axis(self.gen("target_up_axis"), t.category, t.model)
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: generation.target_up_axis of {t.category}/{t.model}: {e}")

    def up_vector(self) -> th.Tensor:
        """:meth:`up_axis` as a unit vector in the target's frame."""
        from ispatialgym.sampling import pair_rules as PR

        i, sign = PR.up_axis(self.up_axis())
        v = th.zeros(3)
        v[i] = sign
        return v

    def standing_quats(self) -> List[th.Tensor]:
        """The target standing on :meth:`up_axis`: the four orientations (xyzw), a quarter turn apart about the vertical, whose
        footprint (its minimum-area rectangle) lies along the world axes - those of the partner AABB the release fit checks."""
        from ispatialgym.sampling import pair_rules as PR

        w, x, y, z = PR.standing_quat_wxyz(self.up_axis())
        q0 = th.tensor([x, y, z, w])
        theta = self.footprint_axis_angle(self.hull["target"], q0)
        return [G.qmul(G.quat_from_yaw(k * math.pi / 2 - theta), q0) for k in range(4)]

    def standing_goal_quat(self, q_init, manip: Stance) -> th.Tensor:
        """The goal orientation of a standing target: of :meth:`standing_quats` the one nearest the start orientation ``q_init``
        (the oracle turns the target upright on the carry from the lift to the pre-place, the smallest turn) among those at which
        the drawn grasp approaches away from the robot (approach . stance forward >= 0); the four when no grasp is drawn yet. The
        first of equal turns. Facts in ``self.standing``."""
        cands = self.standing_quats()
        grasp = self.sidecar.grasp_of(self.bindings.one("target").name)
        fwd = manip.forward
        ahead: List[Optional[float]] = []
        for q in cands:
            if grasp is None or grasp.eef_in_object is None:
                ahead.append(None)
                continue
            a = G.rotate(q, C.quat_to_mat(G.t(grasp.eef_in_object[1]))[:, 2])
            ahead.append(float(a[0] * fwd[0] + a[1] * fwd[1]))
        turns = [math.degrees(G.quat_angle(q_init, q)) for q in cands]
        pool = [k for k in range(len(cands)) if ahead[k] is None or ahead[k] >= -1e-6] or list(range(len(cands)))
        k = min(pool, key=lambda i: turns[i])
        self.standing = {"up_axis": self.up_axis(), "turn_deg": round(turns[k], 1), "approach_ahead": None if ahead[k] is None else round(ahead[k], 3),
                         "turns_deg": [round(v, 1) for v in turns]}
        return cands[k]

    def shelf_pre_place(self, place: G.Pose) -> G.Pose:
        """The pre-place of the touching shelf layout's place (oracle-fix 2026-11-04 round 3, touching#03): with
        ``generation.shelf_pre_place_up_m`` the place raised that much straight up (the book carried into the compartment over its board;
        10 cm back along the level approach the attached book's sphere fit sat in the board - every place screening failed at pose 1 of 2),
        else ``IK.pre_pose``."""
        up = self.gen("shelf_pre_place_up_m") if self.shelf_start is not None else None
        if up is None:
            return IK.pre_pose(place)
        return G.t(place[0]) + th.tensor([0.0, 0.0, float(up)]), G.t(place[1])

    def pre_place_of(self, place: G.Pose) -> G.Pose:
        """The pre-place of a place eef pose: ``PRE_DIST`` back along the approach; a standing target ``PRE_DIST`` straight above, so
        the descent lowers it into its release over the opening (its grasp approaches it sideways there: along the approach the
        target would come in over the rim)."""
        if self.shelf_start is not None and self.gen("shelf_pre_place_up_m") is not None:
            return self.shelf_pre_place(place)  # oracle-fix 2026-11-04 round 3: every caller (the draw's screening too) uses the shelf's vertical pre-place
        if self.up_axis() is None:
            return IK.pre_pose(place)
        return G.t(place[0]) + th.tensor([0.0, 0.0, float(IK.PRE_DIST)]), G.t(place[1])

    def check_goal_upright(self, draft: Draft):
        """``SamplerV2.check_goal_upright``; a standing target: the angle of its up axis (:meth:`up_vector`) to the vertical after the
        goal settle - the stock check follows the object's z axis, which a spin about the vertical turns -, up to the lean the partner
        allows a target standing in it (``atan(footprint short side / height)`` of its rest extents), at least ``goal_tip_max_deg``."""
        if self.up_axis() is None:
            return super().check_goal_upright(draft)
        tname = self.bindings.one("target").name
        _, q_now = self.obj("target").get_position_orientation()
        u_now = G.rotate(q_now.float(), self.up_vector())
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(u_now[2])))))
        p_ext = TT.hull_extents(self.hull["partner"], self.rest["partner"])
        lean = math.degrees(math.atan2(min(float(p_ext[0]), float(p_ext[1])), float(p_ext[2])))
        max_deg = max(float(self.gen("goal_tip_max_deg", self.GOAL_TIP_MAX_DEG)), lean)
        draft.summary["goal_tilt_deg"] = {tname: round(tilt, 2)}
        draft.summary["goal_tilt_max_deg"] = round(max_deg, 2)
        if tilt > max_deg:
            self.counters["goal_tipped"] = self.counters.get("goal_tipped", 0) + 1
            raise Reject("goal_tipped", f"{tname}: up axis {self.up_axis()} {tilt:.1f} deg off the vertical after the goal settle (> {max_deg:.0f})")

    # ------------------------------------------------------------------ goal (§6.11)

    def goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        goal = self.goal_pose(draft.object_poses[tname], draft.object_poses[pname])
        draft.goal_poses[tname] = goal
        self._set_target_goal(draft, tname, goal)
        if self.goal_side is not None:
            draft.summary["goal_side"] = self.goal_side
        manip = self.sidecar.stance("manipulation")
        self.note_pair_goal(draft, goal, draft.object_poses[pname], manip)
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def note_pair_goal(self, draft: Draft, goal: G.Pose, p_pose: G.Pose, manip: Stance) -> None:
        """:meth:`pair_goal_align`: the residuals of the drafted goal (target at ``goal``, partner at ``p_pose``) in ``summary.pair_goal_align.goal``."""
        if self.pair_goal_align() is not None and isinstance(draft.summary.get("pair_goal_align"), dict):
            draft.summary["pair_goal_align"]["goal"] = self.pair_facts(goal, p_pose, manip)

    def _set_target_goal(self, draft: Draft, tname: str, goal: G.Pose):
        tgt = draft.pose_targets[tname]
        tgt["position"] = [float(v) for v in goal[0]]
        if tgt.get("orientation_metric") is not None:
            tgt["orientation"] = [float(v) for v in goal[1]]

    # -- ontop on a container's floor (oracle-fix 2026-11-06, owner on the ontop_small#02 steak demo: "want to scale down steak so it can be
    #    grasped lying down & fits inside the floor of the frying pan (right now it's standing up)") --

    def ontop_centre(self) -> str:
        """``generation.ontop_centre`` (oracle-fix 2026-11-06): ``"bbox"`` (default, today's: the goal centred on the partner's world AABB) or
        ``"container"`` (ontop: the target's footprint centred on the partner's floor, :meth:`container_floor`, and the pair check's fit on
        the floor's disc - frying_pan/jpzusm's AABB holds its 18 cm handle: its centre lies at object x -0.071 m, the floor's at +0.018)."""
        v = str(self.gen("ontop_centre", "bbox") or "bbox")
        if v not in ("bbox", "container"):
            raise Reject("spec", f"{self.gp.variant}: generation.ontop_centre must be 'bbox' or 'container', got {v!r}")
        return v

    def container_floor(self) -> Optional[Dict[str, Any]]:
        """:meth:`ontop_centre` ``"container"``: the partner's floor in its root frame - :class:`ContainerFloor` of its collision pieces
        (``SimWorld.local_collision_pieces``) turned to its rest orientation (``container_floor_res_m`` 0.002, ``container_floor_tol_m``
        0.005: the band above the floor's lowest level, jpzusm's floor is dished 3.6 mm; ``container_floor_normal_min`` 0.99), else the centre
        of its fillable meta link's volume (:meth:`_fillable_root`; no disc: the pair check keeps the box tiers). ``{"centre_root" (xyz),
        "radius_m", "source": "floor" | "fillable", ...facts}``, once per template (the partner's hull of the template); None when neither."""
        hull = self.hull.get("partner")
        cached = getattr(self, "_container_floor", None)
        if cached is not None and cached[0] is hull:
            return cached[1]
        prt = self.obj("partner")
        rest = (th.zeros(3), G.t(self.rest["partner"]).float())
        up = [C.transform_points(rest, G.t(P).float().reshape(-1, 3)) for P in self.world.local_collision_pieces(prt)]
        floor = ContainerFloor.build(up, res=float(self.gen("container_floor_res_m", 0.002)), tol=float(self.gen("container_floor_tol_m", 0.005)),
                                     normal_min=float(self.gen("container_floor_normal_min", 0.99)))
        fill = self._fillable_root(prt)
        fill_up = None if fill is None else C.transform_points(rest, fill.reshape(1, 3))[0]
        out: Optional[Dict[str, Any]] = None
        if floor is not None:
            c_up = th.tensor([float(floor.centre_xy[0]), float(floor.centre_xy[1]), floor.z], dtype=th.float32)
            bottom = min(float(P[:, 2].min()) for P in up if len(P))
            out = {"source": "floor", "centre_root": C.transform_points(C.invert(rest), c_up.reshape(1, 3))[0], "radius_m": round(floor.radius, 4),
                   "floor_above_bottom_m": round(floor.z - bottom, 4), "n_floor_cells": floor.n_flat}
            if fill_up is not None:  # recorded: OmniGibson's own Inside samples the fillable volume's AABB
                out["fillable_offset_m"] = round(float(th.linalg.norm(fill_up[:2] - c_up[:2])), 4)
        elif fill is not None:
            out = {"source": "fillable", "centre_root": fill, "radius_m": None}
        self._container_floor = (hull, out)
        if out is not None:
            log.info("%s: ontop_centre container: %s centre %s (root frame), disc radius %s m", self.gp.variant, out["source"],
                     [round(float(v), 4) for v in out["centre_root"]], out["radius_m"])
        return out

    def container_floor_at(self, p_pose: G.Pose) -> Optional[Dict[str, Any]]:
        """:meth:`container_floor` with the partner at ``p_pose``: ``{"xy": world xy of its centre, "radius_m", "source"}``; None without one."""
        floor = self.container_floor()
        if floor is None:
            return None
        w = C.transform_points((G.t(p_pose[0]).float(), G.t(p_pose[1]).float()), G.t(floor["centre_root"]).float().reshape(1, 3))[0]
        return {"xy": w[:2].clone(), "radius_m": floor["radius_m"], "source": floor["source"]}

    def _fillable_root(self, obj) -> Optional[th.Tensor]:
        """The centre of ``obj``'s fillable meta link's volume (its visual AABB, as OmniGibson's Inside samples it; else the link's origin)
        in ``obj``'s root frame, from the current sim poses; None without a container meta link."""
        for link in getattr(obj, "links", {}).values():
            if not getattr(link, "is_meta_link", False):
                continue
            from omnigibson.macros import macros

            if link.meta_link_type not in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                continue
            try:
                lo, hi = link.visual_aabb
                p = (G.t(lo).float() + G.t(hi).float()) / 2
            except Exception:  # noqa: BLE001 - no visual mesh: the link's own origin
                p = G.t(link.get_position_orientation()[0]).float()
            root = obj.get_position_orientation()
            return C.transform_points(C.invert((G.t(root[0]).float(), G.t(root[1]).float())), p.reshape(1, 3))[0]
        return None

    def goal_pose(self, t_pose: G.Pose, p_pose: G.Pose) -> G.Pose:
        """§6.11 goal: ontop / under(small) -> centred on the partner, z = partner top + bottom offset + goal_drop; inside
        -> the **release pose**: centred on the fillable centroid, high enough above the rim that the open gripper never
        touches the partner (:meth:`release_pose`; the drop test of the goal check settles it, the settled pose is the
        recorded goal); nextto / touching -> the position closest to the init pose that satisfies the predicate: the
        target goes on the partner **side facing its init position** (nextto: AABB gap < g_NT; touching: contact,
        nudged in the goal check). ``self.goal_side`` = ``"left"`` / ``"right"`` (robot frame) records the chosen side.
        Touching with :meth:`edge_aligned`: the goal orientation is the edge-aligned yaw nearest the init yaw. :meth:`pair_goal_align`
        (nextto / touching): the orientation of :meth:`pair_goal_quat` and the target's footprint rectangle level with the partner's
        along the stance's depth axis - beside it across their long axes."""
        campaign_plan = getattr(self, "_campaign_pair_active", None)
        if campaign_plan is not None and not getattr(self, "_campaign_native_goal", False):
            # Recompute the native release geometry at the planned goal to retain
            # rim clearance, surface and natural-pair diagnostics. The independent
            # physics and full path checks still judge the fixed release below.
            self._campaign_native_goal = True
            try:
                # Surface placement is checked at the planned release attitude.
                # The goal is the later physics-settled pose: feeding its tilt
                # back into the release calculation changes the hull's lowest
                # point and rejects a valid anchored release before any motion.
                # Keep the independent settled-goal check in campaign_verify_pair.
                reference = campaign_plan.release if self.relation in ("ontop", "under") else campaign_plan.goal
                native = IORSSmall.goal_pose(self, tuple(G.t(v) for v in reference), tuple(G.t(v) for v in campaign_plan.partner))
            finally:
                self._campaign_native_goal = False
            dp, dq = CPG.pose_error(native, campaign_plan.release)
            if campaign_plan.release_policy == "native_grasp_height":
                # The open jaw's lowest point depends on eef_in_object. The
                # unchanged native release_pose computes and checks its needed
                # clearance; a different grasp must be allowed a different z.
                dp = float(th.linalg.norm(G.t(native[0])[:2] - G.t(campaign_plan.release[0])[:2]))
            if self.relation == "nextto":
                # The goal's gap is independently sampled from the middle 80%
                # of the predicate range; native goal generation picks one gap.
                dp = abs(float(native[0][2]) - campaign_plan.release[0][2])
                if not self.top.polygon_supported(self.world.footprint(self.hull["target"], tuple(G.t(v) for v in campaign_plan.release)),
                                                   margin=self.margin_support):
                    raise Reject("campaign_goal_support", "fixed nextto goal leaves the support")
            if dp > self.ikc.WITNESS_POS_TOL or dq > self.ikc.WITNESS_ANG_TOL:
                raise Reject("campaign_release_drift", f"release differs from native pair geometry by {dp:.6f} m / {dq:.6f} rad; "
                             f"native_z={float(native[0][2]):.6f} planned_z={campaign_plan.release[0][2]:.6f} "
                             f"surface_z={getattr(self, 'release_surface_z', None)} above={getattr(self, 'release_above_m', None)}")
            if campaign_plan.release_policy == "native_grasp_height":
                return native
            return tuple(G.t(v) for v in campaign_plan.release)
        st = self.sidecar.stance("manipulation")
        t_quat = self.target_goal_quat(t_pose, st, p_pose if self.relation in ("nextto", "touching") else None)
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        self.goal_side = None
        if self.relation in ("ontop", "under"):
            xy = (p_hull_w[:, :2].max(dim=0).values + p_hull_w[:, :2].min(dim=0).values) / 2
            origin_off = th.zeros(2)
            if self.partner_aligned_to_target():
                # the target's footprint (its minimum-area rectangle), not its origin, centred on the partner's: the aligned long axes keep
                # equal margins at both ends (the spatula's origin sits 1.6 cm toward its blade, the sheet leaves 1 cm per end)
                xy = min_area_rect(p_hull_w)[3]
                origin_off = min_area_rect(t_hull)[3]
            container = self.relation == "ontop" and self.gen("ontop_centre", "bbox") not in (None, "bbox") and self.ontop_centre() == "container"
            floor = self.container_floor_at(p_pose) if container else None
            if floor is not None:
                # oracle-fix 2026-11-06 (owner on the ontop_small#02 steak demo: "fits inside the floor of the frying pan"): the target's
                # footprint centred on the partner's floor - the pan's AABB centre (handle included) lay 9 cm from its floor's centre
                xy = floor["xy"]
                origin_off = min_area_rect(t_hull)[3]
            top = float(p_hull_w[:, 2].max())
            # owner item 3 (2026-09-16 10:15): release 1 cm above the *actual* surface under the footprint centre (saucer
            # well, plate centre) — a downward raycast from above the partner, the target ignored — instead of goal_drop
            # above the partner's hull top (the teacup fell rim-to-well in the demo); usable once the partner is placed
            surface = self.release_surface(xy, p_pose, top)
            self.release_surface_z = None if surface is None else surface["z"]
            base_z = top if surface is None else float(surface["z"])
            # release height: half the well depth (rim top - well floor), clamped to [3 mm, release_above_surface_m]: a
            # shallow saucer well (1.5 cm) gets a 7.5 mm release, a flat plate 3 mm - a tall cup dropped 1 cm onto the
            # curved well landed on the slope and rested 10 deg tilted (demo 2026-09-16 14:44)
            depth = max(0.0, top - base_z)
            above = min(float(self.gen("release_above_surface_m", RELEASE_ABOVE_SURFACE)), max(RELEASE_ABOVE_SURFACE_MIN, 0.5 * depth))
            above += contact_height_extra(self, "release")[0]  # A/B knob: the release height lifted by the constant
            self.release_above_m = above
            z = base_z + above - float(t_hull[:, 2].min())
            return th.tensor([float(xy[0] - origin_off[0]), float(xy[1] - origin_off[1]), z]), t_quat
        if self.relation == "inside":
            return self.release_pose(t_quat, t_hull, p_hull_w, p_pose)
        gap = 0.0 if self.relation == "touching" else max(0.0, float(self.pair["g_NT"]) - 0.01)
        p_lat = (p_hull_w[:, :2] - st.base_xy) @ st.tangent
        t_lat = t_hull[:, :2] @ st.tangent  # the target's own lateral extent about its origin
        _, lat_init = st.local(t_pose[0])
        # owner review 2026-09-16: the partner side that faces the target's init position (no lift over the partner)
        self.goal_side, lat = nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_lat.min()), float(t_lat.max())), gap)
        if self.relation == "touching" and self.goal_side != ("left" if self.shelf_start is not None and self.shelf_start["lean_side"] > 0 else "right"):
            # §4.6 step 2 puts the partner on the robot's left of the target: the push goes toward +tangent into the
            # partner's right face, the brace holds its left face; a target starting left of the partner contradicts that
            raise Reject("touching_side", f"target init lateral {lat_init:+.2f} is left of the partner centre {0.5 * float(p_lat.min() + p_lat.max()):+.2f}")
        depth_t, _ = st.local(t_pose[0])
        if self.pair_goal_align() is not None:
            depth_t = self.pair_level_depth("target", t_quat, min_area_rect(p_hull_w)[3], st)
        raised = self.raised_release() if self.relation == "nextto" else None
        if self.relation == "nextto":
            handed = (self.pair or {}).get("handedness")
            if handed and self.goal_side != handed["target_side"]:
                # the partner was placed on the other side (``handed_ranges``): a target starting on the wrong side contradicts that
                raise Reject("handedness_side", f"the {handed['target']} target would end {self.goal_side} of the {handed['partner']} partner")
            if raised is None:
                lat, t_quat = self.avoid_release_sweep(st, depth_t, lat, t_quat, lat_init, p_lat, p_hull_w, gap)
        xy = st.world(depth_t, lat)
        pos = th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2]) + shelf_goal_dz(getattr(self, "shelf_start", None), self.hull["target"], t_pose, t_quat)])
        if raised is not None:
            pos = pos + th.tensor([0.0, 0.0, self.release_raise(pos, t_quat, p_hull_w, raised)])
        return pos, t_quat

    # -- raised release (owner 2026-10-14 ~01:51, nextto#03: "try the higher release: lift the spoon about 2 cm so the fingers open above
    #    the chopstick ... not too high - just so that the fingers don't touch") --

    def raised_release(self) -> Optional[Dict[str, float]]:
        """nextto ``generation.raised_release`` (default unset = off): ``true`` or ``{"partner_clearance_m", "max_raise_m", "step_m"}``
        (defaults :data:`RAISED_RELEASE_DEFAULTS`). The release stays the open command (+1); the place rises over the resting goal to the
        lowest height at which the fully open jaw clears the partner (:meth:`release_raise`), the goal check drops the target from there
        and the settled pose is the recorded goal. None when unset or for another relation."""
        v = self.gen("raised_release", None)
        if self.relation != "nextto" or v is None or v is False:
            return None
        out = dict(RAISED_RELEASE_DEFAULTS)
        if isinstance(v, dict):
            unknown = sorted(set(v) - set(out))
            if unknown:
                raise Reject("spec", f"{self.gp.variant}: generation.raised_release: unknown keys {unknown} (known {sorted(out)})")
            out.update({k: float(x) for k, x in v.items()})
        elif v is not True:
            raise Reject("spec", f"{self.gp.variant}: generation.raised_release: true or a dict, got {v!r}")
        return out

    def release_raise(self, pos: th.Tensor, t_quat: th.Tensor, p_hull_w: th.Tensor, raised: Dict[str, float]) -> float:
        """The lowest raise (m, ``step_m`` steps from 0) of the target's resting goal ``(pos, t_quat)`` at which the fully open jaw of the
        drawn grasp (:meth:`open_jaw_clearance`: from the fingertips up through the retreat, ``GRIPPER_OPEN_M``) clears the partner's hull
        - its edges sampled (:func:`hull_edge_points`) - by ``partner_clearance_m``; ``Reject("release_sweep")`` when not even
        ``max_raise_m`` does. 0 without a drawn grasp. ``self.raised_release_facts`` records it."""
        self.raised_release_facts = None
        grasp = self.sidecar.grasp_of(self.bindings.one("target").name)
        if grasp is None or grasp.eef_in_object is None:
            return 0.0
        pts = hull_edge_points(p_hull_w)
        need, top, step = float(raised["partner_clearance_m"]), float(raised["max_raise_m"]), max(float(raised["step_m"]), 1e-4)
        q = G.t(t_quat)

        def clearance(h: float) -> float:
            return self.open_jaw_clearance(IK.grasp_pose((G.t(pos) + th.tensor([0.0, 0.0, h]), q), grasp.eef_in_object), pts)

        c0 = clearance(0.0)
        n = int(math.floor(top / step + 1e-9))
        h, c = 0.0, c0
        for i in range(n + 1):
            h = i * step
            c = clearance(h)
            if c >= need:
                break
        place = IK.grasp_pose((G.t(pos) + th.tensor([0.0, 0.0, h]), q), grasp.eef_in_object)
        low = float(gripper_corners_world(place, open_width=C.GRIPPER_OPEN_M)[:, 2].min())
        self.raised_release_facts = {"raise_m": round(h, 4), "jaw_clearance_m": round(c, 4), "jaw_clearance_at_rest_m": round(c0, 4),
                                     "partner_clearance_m": need, "max_raise_m": top, "open_gripper_lowest_z": round(low, 4),
                                     "partner_top_z": round(float(p_hull_w[:, 2].max()), 4),
                                     "open_gripper_lowest_above_partner_top_m": round(low - float(p_hull_w[:, 2].max()), 4)}
        if c < need:
            raise Reject("release_sweep", f"raised release: the fully open jaw clears the partner by {c * 1000:.1f} mm at the highest allowed raise "
                                          f"({top * 1000:.0f} mm; {need * 1000:.0f} mm needed, {c0 * 1000:.1f} mm at rest)")
        return h

    def raised_release_record(self, draft: Draft, tname: str, place: G.Pose, p_hull_now: th.Tensor, raised: Dict[str, float]) -> Dict[str, Any]:
        """The raised release at the recorded place (``after_goal_check``): :meth:`release_raise`'s facts, the open jaw's clearance from the
        settled partner there (below 0 -> ``Reject("release_sweep")``), the lowest open-gripper point above the partner's top, and the drop
        from the release pose (``draft.goal_poses``) to the settled goal (the pose target): ``drop_m`` (height), ``slide_m`` (horizontal),
        ``turn_deg`` / ``yaw_deg``."""
        rec: Dict[str, Any] = dict(getattr(self, "raised_release_facts", None) or {}, key=dict(raised))
        c_place = self.open_jaw_clearance(place, hull_edge_points(p_hull_now))
        low = float(gripper_corners_world(place, open_width=C.GRIPPER_OPEN_M)[:, 2].min())
        rec.update({"jaw_clearance_at_place_m": round(c_place, 4), "open_gripper_lowest_above_partner_top_at_place_m": round(low - float(p_hull_now[:, 2].max()), 4)})
        rel_pos, rel_quat = (G.t(v) for v in draft.goal_poses[tname])
        tgt = draft.pose_targets[tname]
        settled_pos = th.tensor(tgt["position"], dtype=th.float32)
        rec.update({"release_z": round(float(rel_pos[2]), 4), "settled_z": round(float(settled_pos[2]), 4), "drop_m": round(float(rel_pos[2] - settled_pos[2]), 4),
                    "slide_m": round(float(th.linalg.norm(settled_pos[:2] - rel_pos[:2])), 4)})
        if tgt.get("orientation") is not None:
            q1 = th.tensor(tgt["orientation"], dtype=th.float32)
            rec.update({"turn_deg": round(math.degrees(G.quat_angle(rel_quat, q1)), 2), "yaw_deg": round(math.degrees(G.wrap_angle(G.yaw_of(q1) - G.yaw_of(rel_quat))), 2)})
        if c_place < 0:
            raise Reject("release_sweep", f"raised release: at the recorded place the fully open jaw reaches {-c_place * 1000:.1f} mm into the settled partner")
        return rec

    # -- open-jaw footprint at the release (oracle nextto re-run 0916-181440: the opening finger swept the pencil 9 cm) --

    def open_jaw_clearance(self, place_eef: G.Pose, obstacle_pts: th.Tensor, open_width: float = C.GRIPPER_OPEN_M, retreat: float = C.PRE_DIST) -> float:
        """Signed clearance (m) of ``obstacle_pts`` (world) from the **swept open-jaw footprint** of a place eef pose: the box
        the opened fingers sweep from the release up through the retreat, in the eef frame — along a from
        ``PAD_Z_RANGE[0] - retreat`` to ``FINGERTIP_LENGTH``, along c ±(``open_width`` / 2 + finger thickness + 1 cm), along
        x ±(finger half width + 1 cm). Positive = every point outside by at least that much; negative = a point inside."""
        pos, quat = G.t(place_eef[0]), G.t(place_eef[1])
        R = C.quat_to_mat(quat)
        local = (G.t(obstacle_pts) - pos) @ R  # columns of R = eef axes -> coordinates (x, c, a)
        hx = FINGER_HALF_WIDTH + 0.01
        hc = 0.5 * float(open_width) + FINGER_THICKNESS + 0.01
        a_lo, a_hi = float(C.PAD_Z_RANGE[0]) - float(retreat), float(C.FINGERTIP_LENGTH)
        dx = local[:, 0].abs() - hx
        dc = local[:, 1].abs() - hc
        da = th.maximum(a_lo - local[:, 2], local[:, 2] - a_hi)
        return float(th.stack([dx, dc, da], dim=1).max(dim=1).values.min())

    def open_jaw_opening_max(self, place_eef: G.Pose, obstacle_pts: th.Tensor, retreat: float = C.PRE_DIST) -> float:
        """Largest symmetric release opening (m) whose swept jaw keeps 1 cm from the obstacle points that lie within the
        jaw's x / a extents: ``2 * (min |c| - finger thickness - 1 cm)``; the full ``GRIPPER_OPEN_M`` when none does."""
        pos, quat = G.t(place_eef[0]), G.t(place_eef[1])
        R = C.quat_to_mat(quat)
        local = (G.t(obstacle_pts) - pos) @ R
        hx = FINGER_HALF_WIDTH + 0.01
        a_lo, a_hi = float(C.PAD_Z_RANGE[0]) - float(retreat), float(C.FINGERTIP_LENGTH)
        in_xa = (local[:, 0].abs() <= hx) & (local[:, 2] >= a_lo) & (local[:, 2] <= a_hi)
        if not bool(in_xa.any()):
            return float(C.GRIPPER_OPEN_M)
        c_min = float(local[in_xa, 1].abs().min())
        return max(0.0, min(float(C.GRIPPER_OPEN_M), 2.0 * (c_min - FINGER_THICKNESS - 0.01)))

    def avoid_release_sweep(self, st: Stance, depth_t: float, lat: float, t_quat: th.Tensor, lat_init: float, p_lat: th.Tensor, p_hull_w: th.Tensor,
                            gap: float) -> Tuple[float, th.Tensor]:
        """nextto (owner 2026-09-16 20:29): the partner must lie outside the open-jaw footprint at the place; otherwise the
        place moves along the nextto bound away from the partner (gap up to g_NT - 3 mm), then — when the target's orientation
        is free — the place yaw turns the jaw away (±90°, 180°). ``self.release_sweep`` records the clearance, the remedy and
        the largest release opening that clears the partner by 1 cm (``release_opening_max_m``, for the oracle)."""
        tname = self.bindings.one("target").name
        grasp = self.sidecar.grasp_of(tname)
        self.release_sweep = None
        if grasp is None or grasp.eef_in_object is None:
            return lat, t_quat
        g_nt = float(self.pair.get("g_NT", 0.03))
        z = float(self.top.z) if self.top is not None else 0.0
        t_hull0 = self.hull["target"]

        def place_of(lat_v: float, quat: th.Tensor) -> G.Pose:
            xy = st.world(depth_t, lat_v)
            t_hull = G.local_points_world(t_hull0, (th.zeros(3), quat))
            return IK.grasp_pose((th.tensor([float(xy[0]), float(xy[1]), z - float(t_hull[:, 2].min()) + 0.005]), quat), grasp.eef_in_object)

        def lat_of(quat: th.Tensor, gap_v: float) -> float:
            t_hull = G.local_points_world(t_hull0, (th.zeros(3), quat))
            t_l = t_hull[:, :2] @ st.tangent
            return nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_l.min()), float(t_l.max())), gap_v)[1]

        c0 = self.open_jaw_clearance(place_of(lat, t_quat), p_hull_w)
        rec: Dict[str, Any] = {"clearance_m": round(c0, 4), "remedy": None, "open_width_m": C.GRIPPER_OPEN_M}
        best_lat, best_quat, best_c = lat, t_quat, c0
        if c0 < 0:
            # (a) along the nextto bound: the AABB gap may grow to just under g_NT (the predicate: norm(gap) < g_NT + adjacency)
            for gap_v in (g_nt - 0.006, g_nt - 0.003):
                if gap_v <= gap:
                    continue
                lat_v = lat_of(t_quat, gap_v)
                c_v = self.open_jaw_clearance(place_of(lat_v, t_quat), p_hull_w)
                if c_v > best_c:
                    best_lat, best_c = lat_v, c_v
                    rec.update({"remedy": f"gap {gap_v:.3f}", "gap_m": round(gap_v, 4)})
                if best_c >= 0:
                    break
        if best_c < 0 and self.target_orientation_free() and self.pair_goal_align() is None:
            # (b) the place yaw turns the jaw away from the partner (the target's orientation is not constrained; pair_goal_align fixes it)
            for yaw_deg in (90.0, -90.0, 180.0):
                q_yaw = G.compose((th.zeros(3), G.quat_from_yaw(math.radians(yaw_deg))), (th.zeros(3), t_quat))[1]  # world yaw applied to the rest pose
                for gap_v in (gap, g_nt - 0.003):
                    lat_v = lat_of(q_yaw, gap_v)
                    c_v = self.open_jaw_clearance(place_of(lat_v, q_yaw), p_hull_w)
                    if c_v > best_c:
                        best_lat, best_quat, best_c = lat_v, q_yaw, c_v
                        rec.update({"remedy": f"yaw {yaw_deg:+.0f} deg, gap {gap_v:.3f}", "yaw_deg": yaw_deg, "gap_m": round(gap_v, 4)})
                if best_c >= 0:
                    break
        # the largest symmetric release opening that clears the partner by 1 cm at the chosen place (the oracle may open
        # only this far: closure probe 'pre-shaped opening'); the pinch width + 1 cm is the least that releases the object
        opening_max = self.open_jaw_opening_max(place_of(best_lat, best_quat), p_hull_w) if best_c < 0 else float(C.GRIPPER_OPEN_M)
        ext = TT.hull_extents(t_hull0, self.rest["target"])
        pinch = float(min(ext[0], ext[1]))
        rec.update({"clearance_after_m": round(best_c, 4), "release_opening_max_m": round(max(0.0, opening_max), 4), "pinch_width_m": round(pinch, 4),
                    "partner_inside_jaw": best_c < 0})
        self.release_sweep = rec
        if best_c < 0 and opening_max < pinch + 0.01:
            raise Reject("release_sweep", f"the partner lies {-best_c * 1000:.0f} mm inside the open-jaw footprint at every admissible place (gap up to g_NT - 3 mm"
                                          f"{', yaw turned' if self.target_orientation_free() else ''}); even an opening of pinch + 1 cm ({pinch + 0.01:.3f} m) would sweep it")
        return best_lat, best_quat

    def target_orientation_free(self) -> bool:
        tols = self.gen("tolerances") or {}
        return "orientation_metric" in tols and tols["orientation_metric"] is None

    def release_surface(self, xy: th.Tensor, p_pose: G.Pose, top_z: float) -> Optional[Dict[str, Any]]:
        """``SimWorld.surface_z_under`` at the goal xy (ray from 30 cm above the partner's top, the target ignored) when the
        partner actually stands at ``p_pose`` (it is parked while the first goal is drafted; ``after_settle`` recomputes);
        a hit above the support and within the partner's height range counts, else None (hull-top rule)."""
        p_now = self.obj("partner").get_position_orientation()[0].float()
        if float(th.linalg.norm(p_now[:2] - G.t(p_pose[0])[:2])) > 0.05:
            return None
        # a small cluster of rays (centre + 4 at +-1.5 cm): a saucer's convex-decomposed well can let the centre ray through
        # to the table (teacup run 17:24: surface 0.8127 = the table, 6 mm under the well floor - the cup was set 4 mm into
        # the saucer); only hits on the partner's own links count, the highest of them is the floor under the footprint centre
        partner = self.obj("partner")
        paths = tuple({l.prim_path for l in partner.links.values()} | {partner.prim_path})
        c = G.t(xy)[:2]
        hits = []
        for dx, dy in ((0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015)):
            h = self.world.surface_z_under(th.tensor([float(c[0]) + dx, float(c[1]) + dy]), top_z + 0.30, ignore=[self.obj("target")])
            if h is not None and str(h.get("body", "")).startswith(paths) and h["z"] <= top_z + 0.005:
                hits.append(h)
        if not hits:
            return None
        best = max(hits, key=lambda h: h["z"])
        best["n_partner_hits"] = len(hits)
        return best

    def release_pose(self, t_quat: th.Tensor, t_hull: th.Tensor, p_hull_w: th.Tensor, p_pose: G.Pose) -> G.Pose:
        """§6.11 inside, owner decision 2026-09-16: release just above the partner's rim so the gripper never contacts it.
        The object is centred on the container and raised until the lowest point of the **open** gripper (finger corners:
        ``PAD_Z_RANGE[0]`` .. ``FINGERTIP_LENGTH`` along a, ±(``GRIPPER_OPEN_M`` / 2 + finger thickness) along c,
        ±finger half width along x, from the drawn grasp's ``eef_in_object``) clears the rim ``p_hull_w[:, 2].max()`` by
        ``release_clearance_m`` (default 0.02); ``drop_height_m`` stays the floor of the object's bottom above the rim.
        The target footprint at the release pose must lie within the opening (outer extents minus ``opening_margin_m``),
        else ``Reject("does_not_fit_inside")``. A standing target (:meth:`up_axis`) has its footprint centred on the container
        instead of its origin: the sunflower's head, off its stem by 1.6 cm, passes the opening centred. Facts for the sidecar /
        summary in ``self.release``."""
        tname = self.bindings.one("target").name
        centre, rim_z = self._container_volume(self.obj("partner"), p_pose)
        t_min = float(t_hull[:, 2].min())
        drop_floor = float(self.gen("drop_height_m", 0.005)) + contact_height_extra(self, "release")[0]  # A/B knob lift
        clearance = float(self.gen("release_clearance_m", RELEASE_CLEARANCE))
        xy = centre[:2]
        if self.up_axis() is not None:
            xy = xy - 0.5 * (t_hull[:, :2].min(dim=0).values + t_hull[:, :2].max(dim=0).values)
        pos = th.tensor([float(xy[0]), float(xy[1]), rim_z + drop_floor - t_min])
        grasp = self.sidecar.grasp_of(tname)
        finger_low = None
        if grasp is not None and grasp.eef_in_object is not None:
            eef = IK.grasp_pose((pos, t_quat), grasp.eef_in_object)
            finger_low = float(gripper_corners_world(eef, open_width=C.GRIPPER_OPEN_M)[:, 2].min())
            lift = max(0.0, rim_z + clearance - finger_low)
            pos = pos + th.tensor([0.0, 0.0, lift])
            finger_low += lift
        # lateral fit of the target at the release pose inside the opening (outer extents minus the margin per side)
        om = float(self.gen("opening_margin_m", 0.01))
        t_xy = self.world.hull_at(self.hull["target"], (pos, t_quat))[:, :2]
        p_lo, p_hi = p_hull_w[:, :2].min(dim=0).values, p_hull_w[:, :2].max(dim=0).values
        if bool((t_xy.min(dim=0).values < p_lo + om / 2).any()) or bool((t_xy.max(dim=0).values > p_hi - om / 2).any()):
            raise Reject("does_not_fit_inside", f"target footprint {[round(float(v), 3) for v in (t_xy.max(dim=0).values - t_xy.min(dim=0).values)]} at the release "
                                                f"pose leaves the opening {[round(float(v), 3) for v in (p_hi - p_lo)]} (margin {om:.3f})")
        self.release = {"place_kind": "drop", "rim_z": round(rim_z, 4), "drop_height_m": round(float(pos[2]) + t_min - rim_z, 4),
                        "finger_clearance_m": None if finger_low is None else round(finger_low - rim_z, 4), "release_clearance_m": clearance}
        if self.standing is not None:
            self.release["standing"] = dict(self.standing)
        return pos, t_quat

    def after_settle(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        p, q = self.obj("partner").get_position_orientation()
        draft.pose_targets[pname] = self._pose_target("partner", (p.clone(), q.clone()))
        goal = self.goal_pose(draft.object_poses[tname], (p.clone(), q.clone()))
        draft.goal_poses[tname] = goal
        self._set_target_goal(draft, tname, goal)
        self.note_pair_goal(draft, goal, (p.clone(), q.clone()), self.sidecar.stance("manipulation"))
        # the recorded grasp poses follow the settled target pose (what the oracle re-derives from the loaded instance)
        grasp = self.sidecar.grasp_of(tname)
        if grasp is not None and (grasp.extras or {}).get("pick_overhang"):
            # i7-plateqtf2 (item 2): an overhang grasp is taken where the push leaves the object (the record's post_push_pose, settled by
            # the overhang pick's own physics settle) - not re-derived from the settled INIT pose, never re-selected there
            draft.summary["target_settle"] = self.settle_motion(draft, tname)
        elif grasp is not None and grasp.eef_in_object is not None:
            grasp.grasp = IK.grasp_pose(draft.object_poses[tname], grasp.eef_in_object)
            grasp.pre_grasp = IK.pre_pose(grasp.grasp)
            manip = self.sidecar.stance("manipulation")
            settled = self.settle_motion(draft, tname)
            draft.summary["target_settle"] = settled
            with TT.record_policy_opening(self, grasp):  # of-v2maps 2026-11-07: the policy at the grasp's pre-shape
                res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, ignore_for_last=self.support_objects(),
                                     scale_key=self.choice.models["target"].scale, cache=False, approached=[self.obj("target")])
            ok = res.ok
            if not ok and settled.get("rotation_deg", 0.0) > float(self.gen("regrasp_after_roll_deg", REGRASP_AFTER_ROLL_DEG)):
                # a round target rolled while settling: its library grasp now faces the wrong way; re-select among the
                # library grasps at the settled pose from the same stance (what the oracle would have to grasp anyway)
                chosen = self.reselect_grasp(draft, grasp, manip, tname)
                if chosen is not None:
                    log.info("%s: target rolled %.0f deg while settling; grasp re-selected at the settled pose: %s", self.gp.variant, settled["rotation_deg"], chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_after_roll": True}
                    ok = True
            if not ok:
                # diagnostics (inside regression 2026-09-16 14:29: the pre-grasp failed 6/8 at the settled pose): the pre
                # alone, with the partner removed, with every spawned object removed, and how far the target settled
                def _ok(r):
                    return "ok" if r.ok else "fail"

                everything = list(self.world.spawned.values())
                r_pre = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, cache=False)
                r_pre_np = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, ignore_for_last=[self.obj("partner")], cache=False)
                r_pre_all = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, ignore_for_last=everything + self.support_objects(), cache=False)
                moved = float(th.linalg.norm(G.t(draft.object_poses[tname][0]) - G.t(draft.pose_targets[tname]["position"])))
                raise Reject(res.code or "grasp_unreachable", f"pre-grasp / grasp at the settled target pose fail the exact IK ({res.reason}; pre alone {_ok(r_pre)}, "
                                                                f"pre with the partner removed {_ok(r_pre_np)}, pre with every object removed {_ok(r_pre_all)}; target settled "
                                                                f"{settled.get('translation_mm', float('nan')):.1f} mm / {settled.get('rotation_deg', float('nan')):.0f} deg from its draft "
                                                                f"(goal {moved * 1000:.0f} mm away); trunk ({manip.translate}, {manip.tilt:.2f}), grasp {grasp.grasp_name} yaw {grasp.yaw:.2f})")
            # closure probe 2026-09-16 (runlogs/closure_probe_20260916): the lower *open* pad standing on the table makes the
            # closure asymmetric (one finger crawls, the other knocks the object along the table: marker roll 24-77 deg, the
            # sprout knocked out of the pinch). The support is ignored for the grasp pose by §3.6, so this is checked here:
            # the open gripper at the grasp configuration must stay ``pad_support_clearance_m`` clear of the support; a grasp
            # that fails tries the other library grasps at the settled pose, then ``grasp_pad_on_support``
            q_grasp = res.q if res.ok else getattr(self, "_reselected_q", None)
            clear = None if q_grasp is None else self.open_pads_clear_of_support(q_grasp)
            if clear is False:
                # a preference, not a gate: a grasp whose open pads clear the support replaces this one when the library has
                # one at the settled pose; otherwise the grasp stays (the oracle's closure ramp covers the residual) and the
                # fact is recorded (low cylinders: the pads necessarily reach the table)
                chosen = self.reselect_grasp(draft, grasp, manip, tname, require_pad_clearance=True)
                if chosen is not None:
                    log.info("%s: open pad on the support for %s; grasp re-selected at the settled pose: %s", self.gp.variant, draft.summary.get("grasp", {}).get("name"), chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_pad_clearance": True}
                    clear = True
                else:
                    log.info("%s: open pad on the support for %s and no library grasp clears it at the settled pose; kept", self.gp.variant, grasp.grasp_name)
            grasp.extras["open_pads_clear_of_support"] = clear
            draft.summary["open_pads_clear_of_support"] = clear
        if self.up_axis() is not None:
            # the standing goal's yaw follows the grasp's approach (:meth:`standing_goal_quat`): a grasp re-selected above gets it anew
            goal = self.goal_pose(draft.object_poses[tname], (p.clone(), q.clone()))
            draft.goal_poses[tname] = goal
            self._set_target_goal(draft, tname, goal)

    def open_pads_clear_of_support(self, q: th.Tensor) -> bool:
        """Whether the robot at the grasp configuration ``q`` (fingers open as the sampler solves it) keeps its gripper
        ``pad_support_clearance_m`` (5 mm) clear of the support: whole-body proximity check with every spawned object removed,
        so only the support (and the scene) remain — the complement of the grasp pose's §3.6 support exemption."""
        margin = float(self.gen("pad_support_clearance_m", PAD_SUPPORT_CLEARANCE))
        others = list(self.world.spawned.values())
        with self.ikc.ignoring(others):
            return bool(self.kit.collision_free(q, margin=margin)[0])

    def settle_motion(self, draft: Draft, tname: str) -> Dict[str, float]:
        """Translation (mm) and rotation (deg) of the target between its drafted pose (``summary.target_draft_pose``) and
        its settled pose — a round target may roll while settling, which turns its library grasp away."""
        d = draft.summary.get("target_draft_pose")
        if not d:
            return {}
        p0, q0 = th.tensor(d[0]), th.tensor(d[1])
        p1, q1 = G.t(draft.object_poses[tname][0]), G.t(draft.object_poses[tname][1])
        return {"translation_mm": round(float(th.linalg.norm(p1 - p0)) * 1000, 1), "rotation_deg": round(math.degrees(G.quat_angle(q0, q1)), 1)}

    def reselect_grasp(self, draft: Draft, grasp: GraspRecord, manip: Stance, tname: str, require_pad_clearance: bool = False,
                       exclude: Sequence[str] = ()) -> Optional[str]:
        """Try the library grasps (flush tier first, then the rest; all attachable) at the *settled* target pose from the
        manipulation stance — pre-grasp + grasp exactly, support removed for the grasp, the target removed along the
        approach, optionally the open pads clear of the support — and rewrite the grasp record with the first that passes.
        With ``generation.grasp_twins`` each library pinch is recorded as the row or its twin (``tabletop.pick_twin``; the
        name rules - ``exclude``, the current grasp under ``require_pad_clearance`` - apply to each finger assignment).
        Returns its name or None."""
        t_pose = draft.object_poses[tname]
        order = list(self.grasps_flush) + [g for g in self.grasps if g not in self.grasps_flush]
        twins = TT.grasp_twins_on(self)
        kw = {"ignore_for_last": self.support_objects(), "scale_key": self.choice.models["target"].scale, "cache": False, "approached": [self.obj("target")]}

        def barred(name) -> bool:
            return (name == grasp.grasp_name and require_pad_clearance) or name in exclude

        for g in order:
            eef_in_object = getattr(g, "pose", None)
            name = getattr(g, "name", None)
            if eef_in_object is None:
                continue
            twin_open = twins and name is not None and not barred(C.twin_name(name))
            if barred(name) and not twin_open:
                continue
            eef = IK.grasp_pose(t_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            res = IK.IKResult(False, None, "excluded by name") if barred(name) else self.ikc.exact([pre, eef], manip, grasp.arm, **kw)
            pick = None
            if twin_open:
                pick = TT.pick_twin(self, manip, grasp.arm, g, eef_in_object, eef, pre, res, kw, support_z=float(self.top.z),
                                    accept=self.open_pads_clear_of_support if require_pad_clearance else None, where="reselect_grasp",
                                    row_allowed=not barred(name))
                if not pick.passed:
                    continue
                g, eef_in_object, eef, pre, res = pick.grasp, pick.eef_in_object, pick.eef, pick.pre, pick.res
            else:
                if not res.ok:
                    continue
                if require_pad_clearance and res.q is not None and not self.open_pads_clear_of_support(res.q):
                    continue
            self._reselected_q = res.q
            grasp.grasp_key, grasp.grasp_name = getattr(g, "key", grasp.grasp_key), getattr(g, "name", grasp.grasp_name)
            grasp.finger_q = list(getattr(g, "finger_q", []) or []) or None
            grasp.eef_in_object = eef_in_object
            grasp.grasp, grasp.pre_grasp = eef, pre
            if pick is not None:
                grasp.extras.update({"wrist_flip": pick.twin, "twin_choice": pick.facts})
            elif "wrist_flip" in grasp.extras:  # a library row replaced the drawn grasp: the draw's choice no longer describes it
                grasp.extras["wrist_flip"] = False
                grasp.extras.pop("twin_choice", None)
            self.apply_height_extra_to_grasp(grasp, t_pose, manip)  # the A/B lift applies to the re-selected library grasp too
            return str(grasp.grasp_name)
        return None

    def turn_recheck(self, draft: Draft, grasp: GraspRecord, q_first: Optional[th.Tensor] = None) -> Optional[str]:
        """``tabletop.held_turns``: this module's checks that depend on the wrist turn, for ``grasp`` just recorded as its other finger
        assignment - the pick at the settled target pose as ``after_settle`` solves it (from ``q_first``, the recorded pre-grasp
        witness wrist-flipped, else afresh) with the wrist camera clear of the support top and of the world and the open pads as clear
        as the recorded turn's; the pre-place -> place descent and the lift of ``after_goal_check`` with the object attached; touching:
        the place from the push stance with its retreat clear pose (``place_infeasible_why``) and the carried drive there swept again
        at this turn's lift configuration. None when every check passes, else the first failure."""
        if any((t.extras or {}).get("backoff_m") for t in getattr(self.sidecar, "transitions", None) or ()):
            # oracle-fix 2026-11-06: the backed-off transition carries the arm to the recorded turn's pre-grasp joints (``arms_q``)
            raise TT.TurnNotOffered("the backed-off init -> manipulation transition carries the arm to this turn's pre-grasp (generation.shelf_backoff_transit_m)")
        tgt, manip = self.obj("target"), self.sidecar.stance("manipulation")
        scale = self.choice.models["target"].scale
        kw = {"ignore_for_last": self.support_objects(), "scale_key": scale, "cache": False, "approached": [tgt]}
        with TT.record_policy_opening(self, grasp):  # of-v2maps 2026-11-07: the policy at the grasp's pre-shape
            res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, q_first=q_first, **kw)
            if not res.ok and q_first is not None:
                res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, **kw)
        if not res.ok:
            return f"pick: {res.reason}"
        if not TT.camera_top_clear(self, grasp.grasp, float(self.top.z), grasp.arm) or not O.camera_clear(self.ikc, res.q, grasp.arm):
            return "pick: the wrist camera on the support top / within 5 mm of the world"
        clear = self.open_pads_clear_of_support(res.q)
        if grasp.extras.get("open_pads_clear_of_support") and not clear:
            return "pick: the open pads on the support (the recorded turn's clear it)"
        grasp.extras["open_pads_clear_of_support"] = clear
        push = self.sidecar.stances.get("push") if self.relation == "touching" else None
        st_place = push or manip
        ign = self.support_objects() + ([self.obj("partner")] if self.relation != "inside" else [])
        lift_end = PC.lift_end_pose(grasp)
        arm_q = None
        with attached(self.kit, grasp.arm, tgt, eef_in_object=grasp.eef_in_object):
            r = self.ikc.exact([grasp.pre_place, grasp.place], st_place, grasp.arm, ignore_for_last=ign, scale_key=scale, cache=False,
                               approached=place_approached(self))
            if not r.ok:
                return f"descent: {r.reason}"
            r = self.ikc.waypoints_ok([grasp.grasp, lift_end], manip, grasp.arm, ignore_all=self.support_objects(), cache=False, approached=[])
            if not r.ok:
                return f"lift: {r.reason}"
            if push is not None:
                # the carried drive to the push stance is swept with the arm at the lift configuration (``push_stance``)
                r = self.ikc.exact([lift_end], manip, grasp.arm, ignore_for_last=self.support_objects(), cache=False)
                if not r.ok or r.q is None:
                    return f"lift configuration: {r.reason}"
                idx = self.kit.robot.arm_control_idx
                arm_q = {"right": r.q[idx["right"]].clone(), "left": r.q[idx["left"]].clone()}
        if self.relation == "touching":
            r = self.ikc.exact([IK.pre_pose(grasp.place, RETREAT_DIST_M + RETREAT_LITTLE_M)], st_place, grasp.arm, ignore_for_last=self.support_objects() + [tgt], cache=False)
            if not r.ok:
                # owner 5p.1 (ISPATIALGYM_SHELF_RETREAT_BACKOFF, the shelf tip release): the place's end + the base backing off may pass instead
                why_b = self._retreat_backoff_why(st_place, grasp.place, grasp.arm, False) if (shelf_retreat_backoff_on(self) and self.shelf_tip()) else r.reason
                if why_b is not None:
                    return f"place retreat clear pose: {r.reason}" + ("" if why_b == r.reason else f"; base back-off: {why_b}")
        if push is not None:
            # the push stance got its carried object after ``check_stances_with_objects``: the drive's sweep covers its pose
            why = self.recheck_carried_travel(self.bindings.one("target").name, arm_q=arm_q, stances=False)
            if why:
                return f"carried drive: {why}"
        return None

    def arm_switch(self, draft: Draft, grasp: GraspRecord, arm: str) -> Optional[str]:
        """``tabletop.held_arms``: the segment's stance switched to ``arm`` in place - the manipulation stance with ``arm`` working and
        the other arm in its first collision-free idle posture, the init -> manipulation transition recorded again (with the objects at
        their settled poses, as ``check_stances_with_objects`` sweeps it). Touching is not offered (its push stance and brace are the
        grasping arm's). None when the switch is made, else why not."""
        if self.relation == "touching":
            raise TT.TurnNotOffered("touching: the push and the brace belong to the arms as recorded")
        manip, init = self.sidecar.stance("manipulation"), self.sidecar.stance("init")
        st = TT.stance_for_arm(self, manip, arm)
        if st is None:
            return f"no collision-free idle posture of the {TT.RB.other_arm(arm)} arm with the {arm} arm working"
        TT.adopt_arm(manip, st)
        self.sidecar.transitions[:] = [t for t in self.sidecar.transitions if not (t.frm == init.name and t.to == manip.name)]
        self.record_stance_travel(init, manip)
        return None

    def refine_targets_from_goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if self.relation == "touching":
            # the tip release (no push follows): the settle itself must bring the book onto the partner, a few 1 mm nudges at most
            self._nudge_until_touching(tname, pname, max_steps=int(self.gen("shelf_tip_max_nudges", 3)) if self.shelf_tip() else 25)
        super().refine_targets_from_goal(draft)
        if self.relation == "inside":
            self._check_inside_protrusion(tname, pname)
        if self.relation == "nextto":
            # the nearer-side goal must still satisfy the predicate after the settle: AABB gap < g_NT (PhysX poses + hulls;
            # ``obj.aabb`` reads the render-side transform, which physics-only steps leave stale)
            gap = self._aabb_gap(tname, pname)
            draft.summary["nextto_gap_m"] = round(gap, 4)
            if gap >= float(self.pair["g_NT"]):
                raise Reject("goal_infeasible", f"nextto AABB gap {gap:.3f} >= g_NT {float(self.pair['g_NT']):.3f} after the settle")
        if self.pair_goal_align() is not None:
            self.check_pair_goal(draft, tname, pname)
        p, _ = self.obj("partner").get_position_orientation()
        tgt = draft.pose_targets[pname]
        moved = float(th.linalg.norm(p - th.tensor(tgt["position"])))
        if moved > float(self.gen("partner_moved_max_m", 0.005)):
            raise Reject("goal_partner_moved", f"partner moved {moved:.3f} m at the goal")
        if getattr(self, "_campaign_pair_active", None) is not None:
            self.campaign_verify_pair(draft, settled=True)

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        if getattr(self, "_campaign_pair_active", None) is not None:
            self.campaign_verify_pair(draft)
            draft.summary["campaign_goal_group"]["phase"] = "settled_goal_verified"
        path = super().write_instance(draft, index, seed)
        if getattr(self, "_campaign_pair_active", None) is not None:
            self._campaign_group_written = getattr(self, "_campaign_group_written", 0) + 1
            if self.relation == "inside":
                # Only a fully validated and durably written native member can
                # guide later siblings of this exact immutable goal group.
                if not hasattr(self, "_campaign_written_inside_stances"):
                    self._campaign_written_inside_stances = {}
                self._campaign_written_inside_stances.setdefault(self._campaign_pair_active.sha256, []).append(
                    draft.summary["manipulation_stance"])
        return path

    def _aabb_gap(self, tname: str, pname: str) -> float:
        """Norm of the per-axis separation vector between the two objects' hull AABBs at their current PhysX poses (0 when
        they overlap on every axis) — the distance OmniGibson's ``NextTo`` compares with g_NT = mean(dims_a + dims_b) / 6."""
        boxes = []
        for role, name in (("target", tname), ("partner", pname)):
            p, q = self.objects[name].get_position_orientation()
            h = self.world.hull_at(self.hull[role], (p.float(), q.float()))
            boxes.append((h.min(dim=0).values, h.max(dim=0).values))
        (a_lo, a_hi), (b_lo, b_hi) = boxes
        sep = th.clamp(th.maximum(a_lo - b_hi, b_lo - a_hi), min=0.0)
        return float(th.linalg.norm(sep))

    def _check_inside_protrusion(self, tname: str, pname: str):
        """§6.11 inside drop test: after the settle the target protrudes above the partner's rim by at most half its height
        (hull AABBs at the PhysX poses; ``obj.aabb`` reads the render-side transform, stale after physics-only steps)."""
        boxes = {}
        for role, name in (("target", tname), ("partner", pname)):
            p, q = self.objects[name].get_position_orientation()
            h = self.world.hull_at(self.hull[role], (p.float(), q.float()))
            boxes[role] = (h.min(dim=0).values, h.max(dim=0).values)
        (t_lo, t_hi), (_, p_hi) = boxes["target"], boxes["partner"]
        protrusion = float(t_hi[2] - p_hi[2])
        if protrusion > float(self.gen("max_protrusion_fraction", 0.5)) * float(t_hi[2] - t_lo[2]):
            raise Reject("does_not_fit_inside", f"target protrudes {protrusion:.3f} m above the rim")

    def after_goal_check(self, draft: Draft):
        """§6.11: place pose (settled goal pose ∘ T_obj→eef of the drawn grasp) and pre-place IK-checked from the
        manipulation stance with the object attached (support and partner ignored for the place pose only); the grasp
        and place eef poses in the manipulation image (§3.9); touching: push contact + left brace recorded."""
        tname = self.bindings.one("target").name
        grasp = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = draft.goal_poses[tname][1] if draft.pose_targets[tname].get("orientation") is None else th.tensor(draft.pose_targets[tname]["orientation"])
        raised = self.raised_release() if self.relation == "nextto" else None
        if self.relation == "inside":
            # the place is the release above the rim (``draft.goal_poses`` = the release pose of :meth:`release_pose`); the
            # recorded goal is what the drop settled to. The gripper never reaches into the partner, so it is not ignored.
            rel_pos, rel_quat = draft.goal_poses[tname]
            place = IK.grasp_pose((G.t(rel_pos), G.t(rel_quat)), grasp.eef_in_object)
            pre_place = self.pre_place_of(place)
            ignore = self.support_objects()
            rim_z = float(self.release["rim_z"]) if getattr(self, "release", None) else None
            low = float(gripper_corners_world(place)[:, 2].min())
            if rim_z is not None and low < rim_z + 0.5 * float(self.gen("release_clearance_m", RELEASE_CLEARANCE)):
                raise Reject("release_too_low", f"open gripper lowest point {low:.3f} vs rim {rim_z:.3f} at the recorded place")
            grasp.extras.update(self.release or {})
            grasp.extras["place_lowest_finger_z"] = round(low, 4)
            draft.summary["release"] = dict(self.release or {}, place_lowest_finger_z=round(low, 4), settled_goal_z=round(float(goal_pos[2]), 4))
        else:
            place_pos = goal_pos
            if raised is not None:
                # the place is the raised release (``draft.goal_poses``: the resting goal raised by :meth:`release_raise`); the recorded
                # goal is what the drop from there settled to
                place_pos, goal_quat = G.t(draft.goal_poses[tname][0]), G.t(draft.goal_poses[tname][1])
            if self.relation == "touching":
                # the oracle program places at the *nextto* position and pushes the target into contact from there: the
                # place is the settled contact position moved back (away from the partner, robot's right) by the push distance
                tangent = manip.tangent
                # the place sits the nominal gap (g_NT - 1 cm) short of the settled contact; the push (recorded in
                # ``touching_contacts``) spans the measured gap at that place plus the margin
                place_pos = goal_pos - th.tensor([float(tangent[0]), float(tangent[1]), 0.0]) * self.place_gap()
                if self.shelf_start is not None:  # shelf layout: back against the shelf push direction (the robot's left for a push to the right)
                    place_pos = goal_pos - self.shelf_push_direction(manip) * self.place_gap()
                self._touching_place_pos = place_pos
                draft.summary["touching"] = {"place_gap_m": round(self.place_gap(), 4)}
                if getattr(self, "_campaign_pair_active", None) is not None:
                    fixed = self.place_pose_for(draft.object_poses[tname], self.obj("partner").get_position_orientation(), manip)
                    dp, dq = CPG.pose_error((place_pos, goal_quat), fixed)
                    if dp > self.ikc.WITNESS_POS_TOL or dq > self.ikc.WITNESS_ANG_TOL:
                        raise Reject("campaign_place_drift", "settled touching place changed before exact IK")
                    place_pos, goal_quat = fixed
                    self._touching_place_pos = place_pos
                if self.shelf_tip():
                    # oracle-fix 2026-11-04 round 4: the lift moves the book shelf_lift_away_m toward the partner too - straight up it stayed against the
                    # side board it leans on (t03tip2b: the carried drive's first swept pose, attached book + left gripper on bookcase_opryhp_0)
                    away = self.shelf_push_direction(manip) * float(self.gen("shelf_lift_away_m", 0.03))
                    grasp.extras["lift_away"] = [round(float(v), 4) for v in away]
                    if shelf_lift_straighten_on(self):  # i7-t03 round 2 (d): the straightening lift + the leaned-board touch at the lift
                        draft.summary["shelf_lift"] = self.shelf_lift_plan(grasp)
                    # the release where the goal check's settle started (the drafted goal: the AABB-contact lean); that settle tipped the
                    # book onto the partner, so the oracle's release does the same and nothing pushes
                    place_pos, goal_quat = G.t(draft.goal_poses[tname][0]).float(), G.t(draft.goal_poses[tname][1]).float()
                    self._touching_place_pos = place_pos
                    draft.summary["touching"] = {"place_gap_m": 0.0, "tip_release": True, "nudge_steps": getattr(self, "_nudge_steps", None)}
            place = IK.grasp_pose((place_pos, goal_quat), grasp.eef_in_object)
            pre_place = self.shelf_pre_place(place)
            ignore = self.support_objects() + [self.obj("partner")]
            if self.relation in ("nextto", "touching"):
                # open-jaw footprint facts at the recorded place (touching: the nextto position the push starts from)
                p_now = self.obj("partner").get_position_orientation()
                p_hull_now = self.world.hull_at(self.hull["partner"], (p_now[0].float(), p_now[1].float()))
                c_place = self.open_jaw_clearance(place, p_hull_now)
                rec = dict(self.release_sweep or {})
                rec.update({"clearance_at_place_m": round(c_place, 4), "partner_inside_jaw_at_place": c_place < 0,
                            "release_opening_max_m": round(self.open_jaw_opening_max(place, p_hull_now), 4)})
                grasp.extras["release_sweep"] = rec
                draft.summary["release_sweep"] = rec
            if self.relation in ("ontop", "under"):
                grasp.extras["release_surface_z"] = None if self.release_surface_z is None else round(float(self.release_surface_z), 4)
                grasp.extras["release_above_m"] = None if self.release_above_m is None else round(float(self.release_above_m), 4)
                grasp.extras["pivot_arm_m"] = round(self.pivot_arm(grasp.eef_in_object), 4) if grasp.eef_in_object is not None else None
                grasp.extras["closing_tilt_deg"] = round(self.closing_tilt_deg(grasp.eef_in_object), 1) if grasp.eef_in_object is not None else None
                draft.summary["release_surface_z"] = grasp.extras["release_surface_z"]
                draft.summary["release_above_m"] = grasp.extras["release_above_m"]
                draft.summary["pivot_arm_m"] = grasp.extras["pivot_arm_m"]
        if self.goal_side is not None:
            draft.summary["goal_side"] = self.goal_side
        # grasp facts for the finger-closure investigation (facts only, 2026-09-16 18:16): the pinch line's angle to the
        # object's long axis and the pad-centre height above the support at the grasp pose
        grasp.extras.update(self.grasp_facts(grasp))
        draft.summary["grasp_facts"] = {k: grasp.extras.get(k) for k in ("pinch_vs_long_axis_deg", "pad_height_above_support_m", "pad_height_above_axis_m",
                                                                          "closing_tilt_deg", "pivot_arm_m")}
        approached_place = (self.support_objects() + [self.obj("partner")]) if self.relation != "inside" else self.support_objects()
        # touching (owner 2026-09-18 16:32): the box is carried to a second base pose along the edge (recorded path, in hand)
        # and placed / pushed from there - the place descent and the contacts are checked from that stance
        work = self.push_stance(draft, manip, grasp) if self.relation == "touching" else None
        st_place = work if work is not None else manip

        def descent(pre_p: G.Pose, pl: G.Pose):
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                # along the whole descent: the placed-on partner (owner A) and — as the program check's ``place`` rule and the
                # oracle's descent — the supports (the carried object's sphere fit reaches the table 4-6 cm above the place)
                return self.ikc.exact([pre_p, pl], st_place, grasp.arm, ignore_for_last=ignore, scale_key=self.choice.models["target"].scale, cache=False,
                                      approached=approached_place)

        res = descent(pre_place, place)
        if not res.ok and res.code and "self-collision (attached_object" in res.reason and not (grasp.extras or {}).get("pick_overhang"):
            # A7 relay (oracle, 2026-09-16 21:07): a collision appearing along the descent between the carried object and the
            # forearm / torso / head is a genuine infeasibility of *this* grasp yaw / approach class, not a hand-fixed pair
            # (those are not checked) — try the other library grasps at the settled init pose before rejecting
            tried = [grasp.grasp_name]
            for _ in range(len(self.grasps)):
                chosen = self.reselect_grasp(draft, grasp, manip, tname, exclude=tried)
                if chosen is None:
                    break
                tried.append(chosen)
                place = IK.grasp_pose((place_pos if self.relation != "inside" else G.t(draft.goal_poses[tname][0]), goal_quat), grasp.eef_in_object)
                pre_place = self.pre_place_of(place)
                res = descent(pre_place, place)
                if res.ok:
                    log.info("%s: the carried object hit the robot on the descent with the drawn grasp; grasp %s carries clear", self.gp.variant, chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_descent_self_collision": True,
                                              "tried": tried}
                    grasp.extras.update(self.grasp_facts(grasp))
                    break
        if not res.ok and res.code:
            raise Reject(res.code, f"pre-place -> place descent from the {st_place.name} stance with the object attached: {res.reason}")
        if not res.ok:
            # diagnostics only on failure (which pose, and whether the attached object / the world / the reach is the cause)
            def _ok(r):
                return "ok" if r.ok else "fail"

            r_pre = self.ikc.exact([pre_place], st_place, grasp.arm, cache=False)
            r_place = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
            everything = [o for o in self.world.scene.objects if self.world.obstacle_filter(o)]
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                r_pre_obj = self.ikc.exact([pre_place], st_place, grasp.arm, cache=False)
                r_pre_obj_ign = self.ikc.exact([pre_place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj_free = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=everything, cache=False)
            _, lat = st_place.local(goal_pos[:2])
            raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the {st_place.name} stance with the object attached "
                                              f"(without the object: pre-place {_ok(r_pre)}, place {_ok(r_place)}; with the object: pre-place {_ok(r_pre_obj)}, "
                                              f"pre-place ignoring support+partner {_ok(r_pre_obj_ign)}, place {_ok(r_place_obj)}, place ignoring every object "
                                              f"{_ok(r_place_obj_free)}; goal lateral {lat:+.2f} m, trunk ({manip.translate}, {manip.tilt:.2f}), stand-off {manip.stand_off})")
        grasp.pre_place, grasp.place = pre_place, place
        if raised is not None:
            # the raised release at the recorded place, with the grasp that carries the target there (a re-selected one included)
            p_now = self.obj("partner").get_position_orientation()
            drop = self.raised_release_record(draft, tname, place, self.world.hull_at(self.hull["partner"], (p_now[0].float(), p_now[1].float())), raised)
            grasp.extras["raised_release"] = drop
            draft.summary["raised_release"] = drop
        if self.relation == "touching":
            # the oracle's place ends 10 cm back along the approach (release + lift-off) and its retreat goes 5 cm further: the
            # pose 15 cm back from the place, fingers open, the released box ignored, must have a configuration (the program
            # check's 'retreat: exact IK failed / joint 6-7 at the limit (stance push)' on 6/16 attempts of 0918-175907)
            # owner 5p.1 (ISPATIALGYM_SHELF_RETREAT_BACKOFF, the shelf tip release): or the place's end + the base backing off (recorded here)
            why_clear, code_clear = self.release_clear_why(st_place, place, grasp.arm, record=True, q_place=getattr(res, "q", None))
            if why_clear is not None:
                raise Reject(code_clear or "place_unreachable", f"place retreat clear pose ({(RETREAT_DIST_M + RETREAT_LITTLE_M) * 100:.0f} cm back along the "
                                                               f"approach) from the {st_place.name} stance: {why_clear}")
        # the lift (oracle ``Lift``): grasp pose raised by carry_height with the object attached, support ignored (the
        # object leaves it); the pre-place is where the carry ends, so the lift end must be IK-feasible too
        lift_end = PC.lift_end_pose(grasp)
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            # tracked from the grasp configuration (one IK branch along the lift), support ignored throughout
            r_lift = self.lift_with_touch(grasp, lambda: self.ikc.waypoints_ok([grasp.grasp, lift_end], manip, grasp.arm, ignore_all=self.support_objects(),
                                                                                cache=False, approached=[]))
        if not r_lift.ok:
            raise Reject(r_lift.code or "lift_unreachable", f"lift ({grasp.carry_height:.2f} m above the grasp) fails with the object attached: {r_lift.reason}")
        cam = self.camera_pose_at(manip)
        cam_place = cam if work is None else self.camera_pose_at(work)
        # the pre-poses join the check (owner 2026-09-16 10:15 item 1b): the oracle's FOV monitor arms from the pre-pose
        t_now = self.obj("target").get_position_orientation()
        self.require_in_frame(cam, {"pre-grasp": grasp.pre_grasp, "grasp": grasp.grasp}, objects={"pre-grasp": ("target", (t_now[0].float(), t_now[1].float()))})
        self.require_in_frame(cam_place, {"pre-place": pre_place, "place": place})
        if PC.carry_frame_check_on() or PC.carry_torso_lift_on() or PC.carry_band_on():
            # i7-framing (owner 2026-11-08 5.1 / 5d): the lifted carry (the oracle's protective via-above) framed by the fingertip rule
            self.carry_frame_check(draft, grasp, manip, st_place, lift_end, pre_place, r_lift)
        if self.shelf_tip():
            grasp.extras["place_kind"] = SHELF_TIP_PLACE_KIND  # the program: pick, carry, place (the book tips onto the partner), retreat
            return
        if self.relation == "touching":
            # the push (+ brace) from the stance the box was placed from (the second base pose when recorded)
            cam = cam_place
            self.touching_contacts(draft, st_place)
            c = self.sidecar.contacts
            poses = {"push pre-contact": c["push"].pre, "push contact": c["push"].contact, "push goal": c["push"].goal}
            # i7-framing 5.2: the objects of the pre-contacts' removed_object_out counter (the target placed, the partner where it stands)
            p_now = self.obj("partner").get_position_orientation()
            objs = {"brace pre-contact": ("partner", (p_now[0].float(), p_now[1].float()))}
            if grasp.eef_in_object is not None:
                eio = (G.t(grasp.eef_in_object[0]).float(), G.t(grasp.eef_in_object[1]).float())
                objs["push pre-contact"] = ("target", G.compose((G.t(place[0]).float(), G.t(place[1]).float()), G.invert(eio)))
            if "brace" in c:
                # a brace whose poses leave the frame is dropped (push-only) rather than rejecting the instance
                brace_poses = {"brace pre-contact": c["brace"].pre, "brace contact": c["brace"].contact}

                def _brace_in(what, p):
                    if what in PRECONTACT_FRAME_POSES:  # i7-framing 5.2: the pre-contact by precontact_frame_rule, counted
                        box, tip = self.precontact_frame_ok(what, p, cam, objs.get(what))
                        return tip if precontact_frame_rule() == "fingertip" else box
                    return self.ikc.eef_in_frame(p, cam)
                if all(_brace_in(w, p) for w, p in brace_poses.items() if p is not None):
                    pass  # judged here (the require_in_frame below would only repeat it: the box passes => the fingertip-only rule passes)
                else:
                    self.sidecar.contacts.pop("brace", None)
                    manip.arms, manip.contacts = ["right"], ["push"]
                    draft.summary["brace"] = {"present": False, "why": "brace poses outside the manipulation image (5 %)", "gate": (self._brace_gate or {})}
            self.require_in_frame(cam, poses, objects=objs)

    @contextmanager
    def posed(self, role: str, pose: G.Pose):
        """The role's object teleported to ``pose`` (collision world synced) for the block, then put back exactly where it was."""
        obj = self.obj(role)
        prev = obj.get_position_orientation()
        obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
        obj.keep_still()
        # train-fast 2026-10-31 (SYNC_ONLY_OBJECTS): only this object moved - its meshes alone are re-posed (the kit falls back to the
        # full pass when the robot root moved); the touching push-stance search enters this context per lateral (36 full syncs = 37-170 s)
        TT.sync_moved(self.kit, [obj])
        try:
            yield
        finally:
            obj.set_position_orientation(prev[0], prev[1])
            obj.keep_still()
            TT.sync_moved(self.kit, [obj])

    def contact_maps_of_pair(self):
        """i7-contact 2026-11-07 (step 3, under v2only): the target's ``touch_push`` map (right arm) and the partner's ``touch_brace`` map
        (left arm) at the top (``tabletop.need_contact_map``: the brace gate is a REQUIRED step, owner 2026-11-07 item 5 - a miss rejects
        ``missing_map``, the ``MAP_REQUEST`` line names it)."""
        return TT.need_contact_map(self, "touch_push", "target", self.top), TT.need_contact_map(self, "touch_brace", "partner", self.top)

    def contact_map_gate(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Dict[str, Any]:
        """The A1 joint gate on the contact maps (i7-contact, under v2only): the TARGET's root at its place pose in the ``touch_push`` map
        (a feasible push class at its yaw slot) and the PARTNER's root in the ``touch_brace`` map, at stance ``st`` - the maps are keyed by
        the object ROOT and its yaw from the stance (the band gate tested face points against grasp cells); the exact chain IK decides."""
        pm, bm = self.contact_maps_of_pair()
        w = TT.contact_map_check(pm, st, t_place_pose)
        b = TT.contact_map_check(bm, st, p_pose)
        ok = bool(w["ok"] and b["ok"])
        out = {"ok": ok, "maps": True, "stance": st.name, "work": {k: w.get(k) for k in ("cell", "yaw_slot_deg", "yaw_err_deg", "feasible", "why")},
               "brace": {k: b.get(k) for k in ("cell", "yaw_slot_deg", "yaw_err_deg", "feasible", "why")}}
        if not ok:
            out["why"] = f"push {'ok' if w['ok'] else 'OFF: ' + str(w['why'])} / brace {'ok' if b['ok'] else 'OFF: ' + str(b['why'])} at {st.name}"
        return out

    def brace_gate_at(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Dict[str, Any]:
        """A1 joint band gate at stance ``st``: the push work point (target's right face at the place) in the right band and the
        brace point (partner's far face) in the left contact band of the same trunk (``brace_placement_admissible``). The shelf
        layout has no brace at any stance."""
        if self.shelf_start is not None:
            return {"ok": False, "stance": st.name, "why": "shelf layout: the side board holds the partner, no brace"}
        if TT.contact_maps_on():
            return self.contact_map_gate(st, t_place_pose, p_pose)
        right_cells = self.bands.band(st.translate, st.tilt, **TT.band_kw(self.bands, st)) if getattr(self, "bands", None) is not None else None
        left_cells = left_contact_band_cells(self, (st.translate, st.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]), stance=st)
        if left_cells is None or right_cells is None:
            return {"ok": None, "why": "left contact band unavailable"}
        work, brace = self.touching_points(t_place_pose, p_pose, st)
        w_b, b_b = st.world_to_base(work[:2]), st.world_to_base(brace[:2])
        ok = brace_placement_admissible(w_b, b_b, right_cells, left_cells)
        return {"ok": bool(ok), "stance": st.name, "work_base": [round(float(v), 3) for v in w_b], "brace_base": [round(float(v), 3) for v in b_b],
                "right_cells": int(len(right_cells)), "left_cells": int(len(left_cells)),
                **({} if ok else {"why": f"work {'in' if in_band(w_b, right_cells) else 'OFF'} right band / brace {'in' if in_band(b_b, left_cells) else 'OFF'} left band at {st.name}"})}

    def hand_overlap(self, eef_pose: G.Pose, role: str, obj_pose: G.Pose, margin: float = 0.005, against: str = "box") -> Optional[str]:
        """Which part of the hand beyond the finger profile (``HAND_SPHERES_EEF``: wrist camera, knuckles, gripper body) comes
        within ``margin`` of the role's hull box at ``obj_pose`` for the eef pose — None when all clear. The IK check ignores the
        pushed object at its contact pose, so the camera hanging over the box top was invisible to it (owner 2026-09-18 19:39).
        ``against="hull"`` (oracle-fix 2026-11-06, :meth:`push_height_sweep_on`): the distance to the hull itself (:class:`HullSections`) -
        the box in the object's frame stood ~2 cm proud of the spray bottle's surface where the leaning blades' gripper body came
        (touching#04: 'gripper body -17 mm from the box'); the box when qhull cannot build the hull."""
        # i7-contact 2026-11-07: the shared rule (contact_poses.hand_overlap; the libraries screen their entries with it)
        hull_w = self.world.hull_at(self.hull[role], obj_pose) if against == "hull" else None
        return CPZ.hand_overlap(eef_pose, self.hull[role], obj_pose, margin=margin, against=against, hull_w=hull_w)

    def push_hand_overlap(self, push_init: G.Pose, push_goal: G.Pose, t_place_pose: G.Pose, direction: th.Tensor, dist: float,
                          against: str = "box") -> Optional[str]:
        """The hand at the push contact (box at the place) and at the push goal (box pushed ``dist``): the first overlap, or None.
        ``against``: :meth:`hand_overlap`'s."""
        bad = self.hand_overlap(push_init, "target", t_place_pose, against=against)
        if bad is None:
            pushed = (G.t(t_place_pose[0]) + G.t(direction) * float(dist), G.t(t_place_pose[1]))
            bad = self.hand_overlap(push_goal, "target", pushed, against=against)
        return bad

    def push_chain_feasible(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose, with_brace: bool = False,
                            height_range: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Optional[str]:
        """The first push class whose [pre, init, goal] chain and retreat clear pose pass the exact IK from ``st`` with the box
        posed at the place (the checks ``touching_contacts`` repeats when recording); None when no class passes. With
        ``with_brace`` the class must also admit a left brace in the two-arm solve (``exact_both_arms``, the right arm locked
        at its push goal) — the first ``brace_search_candidates`` brace candidates are tried; the returned kind is then
        ``"<push kind>+<brace kind>"``. Shelf layout: the push along the shelf push direction by the arm on the target's side
        (:meth:`_touching_push_choice`), never braced. ``height_range`` (oracle-fix 2026-11-06, :meth:`push_height_sweep_on`): only
        the candidates whose contact height lies in it; the passing class's height is left in ``self._push_chain_height``.
        :meth:`push_gripper_open_on` (oracle-fix 2026-11-06): with ``with_brace`` and no brace joining the open push, today's (closed)
        classes are tried with the brace (``"<closed kind>+<brace kind>"``) before the stance counts as push-only."""
        tangent = st.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])
        p_init, b_point = self.touching_points(t_place_pose, p_pose, st)
        t_hull_w = self.world.hull_at(self.hull["target"], t_place_pose)
        t_face_z = (float(t_hull_w[:, 2].min()), float(t_hull_w[:, 2].max()))
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
        gap = float((p_hull[:, :2] @ tangent).min() - (t_hull_w[:, :2] @ tangent).max())
        dist = self.push_distance(gap)
        if self.shelf_start is not None:
            if with_brace:
                return None
            direction = self.shelf_push_direction(st)
            dist = self.push_distance(gap_along(t_hull_w, p_hull, direction))
        ignore_push = self.support_objects() + [self.obj("target")]
        ignore_brace = self.support_objects() + [self.obj("partner")]
        n_brace = int(self.gen("brace_search_candidates", 12))
        with self.posed("target", t_place_pose):
            chosen, _dec, _fails = self._touching_push_choice(st, t_place_pose, p_init, t_face_z, direction, dist, ignore_push, height_range=height_range)
            self._push_chain_fails = list(_fails)  # oracle-fix 2026-11-04: named in push_stance's "no class" entries
            if chosen is None:
                return None
            push_kind = chosen.proposal.payload[0]
            self._push_chain_height = chosen.proposal.payload[2].get("contact_height_m")
            if not with_brace:
                return push_kind
            b_chosen, _bdec, _bfails = self._touching_brace_choice(st, chosen.proposal.chain, t_place_pose, p_pose, b_point, face_z, direction, dist, p_hull, p_init,
                                                                   ignore_push, ignore_brace, limit=n_brace, diagnose=False,
                                                                   push_fingers_q=push_fingers(chosen.proposal.payload[2]))
            if b_chosen is not None:
                return f"{push_kind}+{b_chosen.proposal.payload[2]}"
            if push_fingers(chosen.proposal.payload[2]) != IK.FINGER_CLOSED:
                # oracle-fix 2026-11-06 (push_gripper_open_on): no brace joins the open push from this stance - today's classes with the brace
                # (the open push never costs the stance its brace: touching_02's wide partner face requires it, brace_open_when_wide)
                closed, _cdec, _cfails = self._touching_push_choice(st, t_place_pose, p_init, t_face_z, direction, dist, ignore_push, height_range=height_range,
                                                                    open_classes=False)
                if closed is not None:
                    b_closed, _bdec2, _bfails2 = self._touching_brace_choice(st, closed.proposal.chain, t_place_pose, p_pose, b_point, face_z, direction, dist, p_hull,
                                                                             p_init, ignore_push, ignore_brace, limit=n_brace, diagnose=False)
                    if b_closed is not None:
                        self._push_chain_height = closed.proposal.payload[2].get("contact_height_m")
                        return f"{closed.proposal.payload[0]}+{b_closed.proposal.payload[2]}"
            return None  # the push class passes but no brace joins it from this stance
        return None

    def place_infeasible_why(self, st: Stance, t_place_pose: G.Pose, grasp: GraspRecord) -> Optional[str]:
        """Why the place of the carried box from ``st`` fails (pre-place -> place descent with the box attached, then the
        15 cm clear pose of the release + retreat with the box ignored), or None when both pass."""
        place = IK.grasp_pose(t_place_pose, grasp.eef_in_object)
        ign = self.support_objects() + [self.obj("partner")]
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            r = self.ikc.exact([self.shelf_pre_place(place), place], st, grasp.arm, ignore_for_last=ign, scale_key=self.choice.models["target"].scale, cache=False,
                               approached=place_approached(self, inside=False))
        if not r.ok:
            return f"descent: {r.reason}"
        why, _code = self.release_clear_why(st, place, grasp.arm, q_place=r.q)
        if why is not None:
            return f"clear pose: {why}"
        return None

    def release_clear_why(self, st: Stance, place: G.Pose, arm: str, record: bool = False, q_place: Optional[th.Tensor] = None) -> Tuple[Optional[str], Optional[str]]:
        """The release clear pose from ``st``: the hand ``RETREAT_DIST_M + RETREAT_LITTLE_M`` (15 cm) back along the approach from ``place``,
        fingers open, the released target and the supports ignored - ``(None, None)`` when its exact IK passes, else ``(why, code)``. 5p.1
        (:data:`SHELF_RETREAT_BACKOFF_ENV`, the shelf tip release only): when it fails, :meth:`_retreat_backoff_why` (the place's end + the
        base backing off) may pass instead; ``record`` (the goal check's final release) records that retreat stance and transition."""
        tracked = shelf_release_tracked_on(self) and self.shelf_tip()
        if tracked:
            r = self.release_tracked(st, place, arm, [RETREAT_DIST_M, RETREAT_DIST_M + RETREAT_LITTLE_M], q_place)
        else:
            r = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)], st, arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if record:
            self._drop_retreat_record()
        if r.ok and r.q is not None and _pair_mesh_on(self):
            touch = self.pair_mesh_why(r.q, arm, "clear")
            if touch is not None:
                r = IK.IKResult(False, None, f"self mesh contact at the clear pose: {touch}", 1)
        if r.ok:
            return None, None
        if not (shelf_retreat_backoff_on(self) and self.shelf_tip()):
            return r.reason, getattr(r, "code", None)
        why = self._retreat_backoff_why(st, place, arm, record, q_place=q_place)
        if why is None:
            return None, None
        return f"{r.reason}; base back-off: {why}", getattr(r, "code", None)

    def release_tracked(self, st: Stance, place: G.Pose, arm: str, backs: Sequence[float], q_place: Optional[th.Tensor] = None) -> Any:
        """Round 7 (:data:`SHELF_RELEASE_TRACKED_ENV`): the retreat from the place configuration ``q_place`` (else the place solved afresh)
        through the poses ``backs`` (m back along the approach) as ONE tracked chain (``ikc.exact`` over [place, *backs], the supports and the
        released target out of the world throughout) - the branch the program check and the oracle follow. Kit ``shelf_release_tracked.*``."""
        TT._count(self, "shelf_release_tracked.checked")
        ign = self.support_objects() + [self.obj("target")]
        with self.ikc.ignoring(ign):
            r = self.ikc.exact([place] + [IK.pre_pose(place, float(b)) for b in backs], st, arm, ignore_for_last=ign, cache=False, approached=[],
                               q_first=None if q_place is None else G.t(q_place))
        TT._count(self, "shelf_release_tracked." + ("passed" if r.ok else "failed"))
        return r

    def pair_mesh_why(self, q: th.Tensor, arm: str, where: str) -> Optional[str]:
        """Round 7 (``pair_mesh.PAIR_MESH_ENV``): the ``{arm}_arm_link5`` / ``torso_link2`` collision-mesh clearance at ``q`` (the sphere model
        misses it); None when off or clear, else why. Kit ``pair_mesh.checked.<where>`` / ``.refused.<where>``."""
        from ispatialgym.sampling.v2 import pair_mesh as PM

        if not PM.pair_mesh_on(self):
            return None
        TT._count(self, f"pair_mesh.checked.{where}")
        try:
            why = PM.pair_touch_why(G.t(q).tolist(), arm)
        except Exception as e:  # noqa: BLE001
            TT._count(self, f"pair_mesh.error.{where}")
            log.warning("pair mesh check failed (%r); not judged", e)
            return None
        if why is not None:
            TT._count(self, f"pair_mesh.refused.{where}")
            log.info("%s: PAIR_MESH refused at the %s: %s", self.gp.variant, where, why)
        return why

    def _drop_retreat_record(self) -> None:
        self.sidecar.stances.pop(RETREAT_STANCE, None)
        self.sidecar.transitions[:] = [t for t in self.sidecar.transitions if t.to != RETREAT_STANCE]

    def _retreat_backoff_why(self, st: Stance, place: G.Pose, arm: str, record: bool, q_place: Optional[th.Tensor] = None) -> Optional[str]:
        """5p.1: the shelf tip release's retreat with the base backing off - (1) the place's end, ``RETREAT_DIST_M`` back along the approach
        (where the oracle's place leaves the open hand), exact IK from ``st``; (2) the drive-back ``st`` -> ``st`` at stand-off +
        ``RETREAT_LITTLE_M`` (strides of ``navigation.STRIDE_M``) whole-body collision-free at ``st``'s trunk with both arms frozen at that
        configuration and the fingers open, the released target ignored (the program check sweeps it again with the book at its release
        pose). The hand ends where the 15 cm clear pose is. ``record``: stance :data:`RETREAT_STANCE` + the transition into it (``arms_q``,
        ``retreat_backoff_m``). None when it passes, else why."""
        from ispatialgym.sampling.v2 import navigation as NAV
        from ispatialgym.sampling.v2.stances import Transition

        if shelf_release_tracked_on(self):
            r = self.release_tracked(st, place, arm, [RETREAT_DIST_M], q_place)
        else:
            r = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M)], st, arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if not r.ok or r.q is None:
            TT._count(self, "shelf_retreat.backoff_failed.place_end")
            return f"the place's end ({RETREAT_DIST_M * 100:.0f} cm back): {r.reason}"
        touch = self.pair_mesh_why(r.q, arm, "place_end") if _pair_mesh_on(self) else None
        if touch is not None:
            TT._count(self, "shelf_retreat.backoff_failed.place_end")
            return f"the place's end ({RETREAT_DIST_M * 100:.0f} cm back): {touch}"
        if st.edge is None or st.stand_off is None:
            TT._count(self, "shelf_retreat.backoff_failed.no_edge")
            return "the release stance has no edge frame to back off along"
        idx = self.kit.robot.arm_control_idx
        q_arms = {a: G.t(r.q)[idx[a]].clone() for a in ("right", "left")}
        back = st.shifted(stand_off=float(st.stand_off) + RETREAT_LITTLE_M, name=RETREAT_STANCE)
        trunk = (float(st.translate), float(st.tilt))
        drive = NAV.stride_poses([tuple(float(v) for v in st.base_pose2d), tuple(float(v) for v in back.base_pose2d)], NAV.STRIDE_M)
        with self.ikc.ignoring([self.obj("target")]):
            ok, bad = NAV.path_collision_free(self.kit, drive, trunk, "untucked", margin=0.01, q_arms=q_arms, fingers=float(C.FINGER_OPEN))
        if not ok:
            TT._count(self, "shelf_retreat.backoff_failed.drive")
            return f"the {RETREAT_LITTLE_M * 100:.0f} cm drive-back with the arms frozen collides at stride {bad}/{len(drive)}"
        TT._count(self, "shelf_retreat.backoff_passed")
        if record:
            back.arms, back.idle_posture = st.arms, st.idle_posture
            back.extras = {k: v for k, v in (st.extras or {}).items() if not str(k).startswith("_")}
            back.extras["retreat_backoff"] = {"from": st.name, "drive_m": RETREAT_LITTLE_M, "rule": "owner 5p.1: the base backs off instead of the arm's last 5 cm"}
            self.sidecar.add_stance(back)
            tr = Transition(st.name, RETREAT_STANCE, float(back.stand_off), trunk_from=trunk, trunk_to=trunk, n_samples=0)
            tr.extras["arms_q"] = {a: [round(float(v), 5) for v in q.tolist()] for a, q in q_arms.items()}
            tr.extras["retreat_backoff_m"] = RETREAT_LITTLE_M
            self.sidecar.add_transition(tr)
            TT._count(self, "shelf_retreat.recorded")
            log.info("%s: shelf retreat back-off - the %s arm stops at the place's end, the base drives back %.2f m (%s -> %s, s %.2f -> %.2f)",
                     self.gp.variant, arm, RETREAT_LITTLE_M, st.name, RETREAT_STANCE, float(st.stand_off), float(back.stand_off))
        return None

    def _push_stance_diag(self, lat: float, st: Stance, is_manip: bool, t_place_pose: G.Pose) -> Dict[str, Any]:
        """DIAGNOSTIC (i7-contact 2026-11-07, touching_02's contact_unreachable): one push-stance lateral without a push class, in numbers - the
        stance (base, trunk, stand-off), the placed target's root in its base frame and its yaw from the stance, the ``touch_push`` map's
        lookup there (under v2only) and every candidate's failure (``label: reason``)."""
        out: Dict[str, Any] = {"lateral": lat, "manipulation": is_manip, "base_pose2d": [round(float(v), 4) for v in G.t(st.base_pose2d).tolist()],
                               "translate": round(float(st.translate), 3), "tilt": round(float(st.tilt), 3), "stand_off": getattr(st, "stand_off", None),
                               "fails": list(getattr(self, "_push_chain_fails", []) or [])[:24]}
        try:
            xy = st.world_to_base(G.t(t_place_pose[0])[:2])
            out["root_base"] = [round(float(v), 4) for v in xy.tolist()]
            out["root_z"] = round(float(G.t(t_place_pose[0])[2]), 4)
            if TT.contact_maps_on():
                cm = TT.contact_map(self, "touch_push", "target", self.top)
                if cm is not None:
                    r = TT.contact_map_check(cm, st, t_place_pose)
                    out["map"] = {k: r.get(k) for k in ("ok", "why", "cell", "yaw_slot_deg", "yaw_err_deg", "feasible", "trunk", "edge_distance")}
        except Exception as e:  # noqa: BLE001 - a diagnostic never rejects
            out["diag_error"] = repr(e)[:200]
        return out

    def push_stance_for(self, manip: Stance, t_place_pose: G.Pose, lateral: Optional[float] = None, stand_off: Optional[float] = None) -> Optional[Stance]:
        """Touching (owner 2026-09-18 16:32): the second base pose ``push`` = the manipulation stance moved along the edge, same
        trunk, so that the push contact (the placed box's right face) lies at ``push_contact_lateral_m`` from the robot's centre
        line (default -0.12: every push that worked had it 7-18 cm right of centre; at -0.05 the right arm crosses the body
        and joints 2 / 3 sit on their limits, 0918-173814). The box is carried there in hand and set down from there.
        Pure geometry (nothing recorded); None when the shift is below ``push_stance_min_shift_m`` or the feature is off
        (``generation.push_stance``). Shelf layout: the contact on the target's face away from the partner and the right hand's
        lateral mirrored for the left hand (+0.12: left of centre), the stance's working arm the shelf push arm. ``stand_off`` (owner A1,
        2026-11-08, :meth:`push_stance_candidates`): the stance also moves toward / away from the edge to that stand-off (None: the
        manipulation stance's); a stand-off change alone is a push stance too."""
        if self.relation != "touching" or not bool(self.gen("push_stance", True)):
            return None
        tangent = manip.tangent
        t_hull = self.world.hull_at(self.hull["target"], t_place_pose)
        right_face_xy = t_hull[int(th.argmin(t_hull[:, :2] @ tangent)), :2]
        _, lat_work = manip.local(right_face_xy)
        want = float(self.gen("push_contact_lateral_m", -0.12)) if lateral is None else float(lateral)
        if self.shelf_start is not None:
            sg = self.shelf_push_sign()
            _, lat_work = manip.local(t_hull[int(th.argmin(t_hull[:, :2] @ (sg * tangent))), :2])
            if lateral is None:
                want = sg * float(self.gen("push_contact_lateral_m", -0.12))
        dlat = float(lat_work - want)
        so_manip = float(manip.stand_off) if manip.stand_off is not None else None
        d_so = 0.0 if (stand_off is None or so_manip is None) else float(stand_off) - so_manip
        if abs(dlat) < float(self.gen("push_stance_min_shift_m", 0.02)):
            if abs(d_so) < 1e-6:
                return None
            dlat = 0.0
        so_new = None if abs(d_so) < 1e-6 else float(stand_off)
        target_xy = G.t(manip.base_xy) + G.t(tangent) * dlat - G.t(manip.inward) * d_so
        cands = [manip.shifted(stand_off=so_new, lateral=manip.lateral + s * dlat, name="push") for s in (1.0, -1.0)]  # the edge frame's lateral sign
        st = min(cands, key=lambda c: float(th.linalg.norm(G.t(c.base_xy) - target_xy)))
        st = st.with_trunk(manip.translate, manip.tilt, "push")
        st.focus, st.arms = G.t(t_place_pose[0]).clone(), ["right"]
        if self.shelf_start is not None:
            st.arms = [self.shelf_push_arm()]
        st.extras.update({"second_base_pose": True, "lateral_shift_m": round(dlat, 4), "push_contact_lateral_m": round(float(lat_work - dlat), 4),
                          "rule": "base moved along the edge (box in hand) so the push contact lies at push_contact_lateral_m from the centre line"})
        if so_new is not None:
            st.extras.update({"stand_off_shift_m": round(d_so, 4), "stand_off_rule": "owner A1 2026-11-08: the stand-off chosen from the touch_push map"})
        return st

    def push_stance_stand_offs(self, manip: Stance) -> List[float]:
        """Owner A1 (2026-11-08): the stand-offs a touching push stance may take - ``generation.push_stance_stand_offs_m`` (default
        :data:`PUSH_STANCE_STAND_OFFS_M`: the touch_push maps' edge distances and closer ones) plus the manipulation stance's own. The lower
        limit is the base's distance to the table: every candidate passes the whole-body collision check (:meth:`push_stance_candidates`)."""
        vals = [float(v) for v in (self.gen("push_stance_stand_offs_m", None) or PUSH_STANCE_STAND_OFFS_M)]
        if manip.stand_off is not None:
            vals.append(float(manip.stand_off))
        return sorted({round(v, 4) for v in vals})

    def push_stance_candidates(self, manip: Stance, t_place_pose: G.Pose, laterals: Sequence[float], p_pose: Optional[G.Pose] = None,
                               collision: bool = True) -> Tuple[List[Tuple[float, Optional[Stance], Dict[str, Any]]], Dict[str, Any]]:
        """Owner A1 (2026-11-08, under v2only, sampler-only): the push stance moves along the edge (the contact laterals) AND toward / away
        from it (:meth:`push_stance_stand_offs`), the stand-off chosen from the target's ``touch_push`` map at its place pose: the candidates
        whose map lookup has a feasible push class, those whose partner (``p_pose``) also fits the ``touch_brace`` map first, then per
        lateral (in the laterals' order) the stand-off nearest the manipulation stance's (least base travel). With ``collision`` the
        candidates pass the whole-body collision check (the base's minimum distance to the table: the footprint against the support).
        Returns ``([(lateral, stance or None = the manipulation stance, lookup facts)], facts)``."""
        pm, bm = self.contact_maps_of_pair()
        so_m = float(manip.stand_off) if manip.stand_off is not None else 0.0
        rows: List[Tuple[Tuple[int, int, float], float, Optional[Stance], Dict[str, Any]]] = []
        n_all = n_map = 0
        # i7-symonly (PUSH_TIP_CAP_ENV): the map's classes above the target's tip height do not count (on) / are counted (shadow)
        tip_keep, tip_mode = None, push_tip_cap_mode()
        if tip_mode != "off" and self.push_height_sweep_on():
            t_hull = self.world.hull_at(self.hull["target"], t_place_pose)
            tip = self.push_tip_cap(t_place_pose, float(t_hull[:, 2].min()), th.tensor([float(manip.tangent[0]), float(manip.tangent[1]), 0.0]),
                                    hull_w=t_hull, line_xy=self.push_line_xy(t_place_pose))
            tip_keep = _tip_keep(float(tip["cap_m"])) if tip is not None else None
        for li, lat in enumerate(laterals):
            for so in self.push_stance_stand_offs(manip):
                cand = self.push_stance_for(manip, t_place_pose, lateral=lat, stand_off=so)
                probe = cand if cand is not None else manip
                n_all += 1
                w = TT.contact_map_check(pm, probe, t_place_pose, keep=tip_keep if tip_mode == "on" else None)
                if tip_keep is not None:
                    other = TT.contact_map_check(pm, probe, t_place_pose, keep=None if tip_mode == "on" else tip_keep)
                    if (other["ok"] if tip_mode == "on" else w["ok"]) and not (w["ok"] if tip_mode == "on" else other["ok"]):
                        TT._count(self, "push_tip_cap.map_lost")
                if not w["ok"]:
                    continue
                n_map += 1
                b_ok = bool(TT.contact_map_check(bm, probe, p_pose)["ok"]) if (p_pose is not None and bm is not None) else False
                facts = {"lateral": lat, "stand_off": so, "push_feasible": list(w.get("feasible") or []), "brace_map": b_ok, "cell": w.get("cell")}
                rows.append(((0 if b_ok else 1, li, abs(so - so_m)), lat, cand, facts))
        rows.sort(key=lambda r: r[0])
        seen, out = set(), []
        for _k, lat, cand, facts in rows:
            key = None if cand is None else (round(float(cand.base_pose2d[0]), 4), round(float(cand.base_pose2d[1]), 4))
            if key in seen:
                continue
            seen.add(key)
            out.append((lat, cand, facts))
        n_free = len(out)
        if collision and out:
            stances = [c for _l, c, _f in out if c is not None]
            free = self.ikc.stances_collision_free(stances).tolist() if stances else []
            ok = {id(c): bool(f) for c, f in zip(stances, free)}
            out = [r for r in out if r[1] is None or ok.get(id(r[1]), False)]
            n_free = len(out)
        return out, {"candidates": n_all, "map_feasible": n_map, "collision_free": n_free}

    def push_map_any(self, manip: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Dict[str, Any]:
        """Owner A1 (2026-11-08): the touch_push / touch_brace maps over every push stance candidate (:meth:`push_stance_candidates`: map
        lookups and the batched whole-body collision check of the map-feasible stances - the base's limit at the table; i7_contactfu on the
        bar: the map fitted only at stand-offs < 0.35 m, where the base collides, so a map-only test passed 65 partner draws that died in
        goal_check): ``{"push_any", "pair_any", "n"}``."""
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        rows, facts = self.push_stance_candidates(manip, t_place_pose, laterals, p_pose=p_pose, collision=True)
        return {"push_any": bool(rows), "pair_any": any(f["brace_map"] for _l, _c, f in rows), "n": facts}

    # -- owner 3.6 (2026-11-08, i7-contact3): the place depth capped by the push map's reach ------------------------------------------------

    def place_depth_cap_applies(self) -> bool:
        """:func:`place_depth_cap_mode` is not off and the attempt's place is followed by a mapped push: touching, not the shelf layout (its
        place is the board compartment, not a drawn depth), the push stance on, the contact maps read (v2only + a maps root)."""
        return (place_depth_cap_mode() != "off" and TT.contact_maps_on() and getattr(self, "relation", None) == "touching"
                and getattr(self, "shelf_start", None) is None and getattr(self, "shelf_pose", None) is None and bool(self.gen("push_stance", True)))

    def push_reach_x(self, pm, translate: float, tilt: float) -> Dict[float, Optional[float]]:
        """Per edge distance of the ``touch_push`` map ``pm`` at the trunk nearest (``translate``, ``tilt``) (``ContactMap.nearest_trunk``, as
        the lookup): the deepest base-frame root x that a lookup can find a feasible class at - the deepest cell with a feasible class at
        any yaw slot and any lateral, plus half the lattice (a root nearer the next cell maps to it). None: no feasible cell."""
        from ispatialgym.sampling import contact_maps as CMAP

        tk = pm.nearest_trunk(translate, tilt)
        if tk is None:
            return {}
        cache = self.__dict__.setdefault("_push_reach_x", {})
        key = (pm.path, tuple(tk))
        if key not in cache:
            present = np.array([[r is not None for r in pm.class_rows(y)] for y in range(len(pm.yaws))], dtype=bool)  # (yaw, class)
            half = 0.5 * float((pm.meta.get("config") or {}).get("lattice_m", 0.08))
            out: Dict[float, Optional[float]] = {}
            for d_e in pm.meta["edge_distances"]:
                feas = ((pm.status(float(d_e), tk[0], tk[1]) == CMAP.STATUS["feasible"]) & present[None]).any(axis=(1, 2))
                out[float(d_e)] = float(pm.cells[feas, 0].max()) + half if feas.any() else None
            cache[key] = out
        return cache[key]

    def place_depth_caps(self, cands: Sequence[Stance]) -> Dict[int, Dict[str, Any]]:
        """Owner 3.6 (2026-11-08): the deepest place a push can reach, per manipulation candidate (``id(cand)`` -> facts). The touching target is
        placed at its start depth (:meth:`place_pose_for`: ``depth_t`` of the start; level with the partner), and its push is made from a push
        stance on the same edge and trunk at one of :meth:`push_stance_stand_offs` (A1). The stand-offs the base really takes are those whose
        stance passes the whole-body collision check (the base's minimum distance to the support; one batched check per candidate base pose
        and trunk, at the candidate's lateral); from each, the target's ``touch_push`` map reaches roots up to :meth:`push_reach_x` ahead of
        the base, i.e. a place depth of ``x_reach - edge_depth``. ``cap_m`` = the largest over those stand-offs (None: no free stance or no
        feasible cell = nothing reachable); ``depth_m`` = the candidate's focus depth (the start / place position is drawn within
        ``tabletop.FOCUS_RADIUS`` of it; the rest stays with A1's partner-draw gate); ``remove`` = depth beyond the cap. Empty when the map
        misses (A1's required map rejects later, as before)."""
        pm = TT.contact_map(self, "touch_push", "target", self.top)
        if pm is None:
            TT._count(self, "place_depth_cap.no_map")
            return {}
        memo = self.__dict__.setdefault("_pdc_free", {})

        def key_of(c: Stance):
            return tuple(round(float(v), 3) for v in c.base_pose2d) + (round(float(c.translate), 3), round(float(c.tilt), 3))

        todo: Dict[Tuple, Stance] = {}
        for c in cands:
            if c.focus is not None and key_of(c) not in memo:
                todo.setdefault(key_of(c), c)
        if todo:
            stances, owners = [], []
            for k, c in todo.items():
                for so in self.push_stance_stand_offs(c):
                    st = c.shifted(stand_off=so, name="push").with_trunk(c.translate, c.tilt, "push")
                    st.arms = ["right"]  # as push_stance_for records it
                    stances.append(st)
                    owners.append((k, float(so), float(st.edge_depth)))
            free = self.ikc.stances_collision_free(stances).tolist()
            for (k, so, ed), f in zip(owners, free):
                memo.setdefault(k, []).append((so, ed, bool(f)))
            TT._countn(self, "place_depth_cap.stances_checked", len(stances))
        d_es = [float(v) for v in pm.meta["edge_distances"]]
        out: Dict[int, Dict[str, Any]] = {}
        for c in cands:
            if c.focus is None:
                continue
            rows = memo.get(key_of(c)) or []
            xr = self.push_reach_x(pm, c.translate, c.tilt)
            caps = []
            for so, ed, f in rows:
                x = xr.get(min(d_es, key=lambda v: abs(v - so))) if f else None
                if x is not None:
                    caps.append((x - ed, so))
            free_sos = [so for so, _ed, f in rows if f]
            cap = max(caps)[0] if caps else None
            depth = float(c.local(c.focus[:2])[0])
            out[id(c)] = {"cap_m": None if cap is None else round(cap, 4), "depth_m": round(depth, 4), "remove": cap is None or depth > cap,
                          "s_min": min(free_sos) if free_sos else None, "trunk": [float(c.translate), round(float(c.tilt), 3)]}
            if place_depth_cap_basis() == "a1":
                ok_a1, slot = self._a1_reach_at_focus(pm, c)
                out[id(c)].update(remove_map=out[id(c)]["remove"], remove=not ok_a1, basis="a1", a1_yaw_slot=slot)
                TT._count(self, "place_depth_cap.a1_" + ("ok" if ok_a1 else "off") + ("_map_ok" if not out[id(c)]["remove_map"] else "_map_off"))
        return out

    def _a1_reach_at_focus(self, pm, c: Stance) -> Tuple[bool, Optional[float]]:
        """Follow-up (b): A1's push stance test (:meth:`push_stance_candidates`, map + whole-body collision of the shifted / stood-off stances)
        for the target at ``c``'s focus, tried at the map's yaw slots in turn; ``(any passes, the first passing slot in degrees)``."""
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        manip = c.with_trunk(c.translate, c.tilt, "manipulation")
        manip.focus, manip.arms = c.focus, [self.segment_arms[0]]
        base_yaw = float(c.base_pose2d[2])
        for y in [float(v) for v in pm.yaws.tolist()]:
            pose = TT.object_pose_on(self, self.hull["target"], self.rest["target"], c.focus[:2], base_yaw + y, float(self.top.z))
            rows, _facts = self.push_stance_candidates(manip, pose, laterals, p_pose=None, collision=True)
            if rows:
                return True, round(math.degrees(y), 1)
        return False, None

    def _built_tilts_map(self):
        """The touch_push map :meth:`phase2_built_tilts` reads, or None where the switch does not act (off, not touching, the shelf layout,
        no contact maps or no map: kit ``touch_built_tilts.no_map``)."""
        if not (touch_built_tilts_on() and getattr(self, "relation", None) == "touching" and getattr(self, "shelf_start", None) is None
                and getattr(self, "shelf_pose", None) is None and TT.contact_maps_on()):
            return None
        pm = TT.contact_map(self, "touch_push", "target", self.top)
        if pm is None:
            TT._count(self, "touch_built_tilts.no_map")
        return pm

    def phase2_built_tilts(self, translate: float, limit: float) -> Optional[List[float]]:
        """Follow-up :data:`TOUCH_BUILT_TILTS_ENV` (``tabletop.phase2_tilts``' hook): the tilts of the touch_push map's eligible trunks at
        ``translate`` (2 cm, as ``ContactMap.nearest_trunk``) within the torso limit ``limit``, 0 first; None (the continuous scan) when off,
        not touching / the shelf layout, without the contact maps or the map, or no built tilt there."""
        pm = self._built_tilts_map()
        if pm is None:
            return None
        tilts = sorted({round(float(ti), 4) for tr, ti in pm.trunks() if abs(float(tr) - float(translate)) <= 0.02 and float(ti) >= float(limit) - 1e-6},
                       reverse=True)
        if not tilts:
            TT._count(self, "touch_built_tilts.none_at_translate")
            return None
        TT._count(self, "touch_built_tilts.scans")
        return tilts

    def _pdc_keep_pairs(self):
        """Follow-up (c) (:data:`PLACE_DEPTH_CAP_DRAW_ENV`, with the cap on): Phase 2's ``keep_pairs`` - the (stance, position) pairs whose
        focus lies within the cap (:meth:`place_depth_caps`); None when off or the cap does not apply (no map: every pair kept)."""
        if not (place_depth_cap_draw_on() and place_depth_cap_mode() == "on" and self.place_depth_cap_applies()):
            return None

        def keep(cands: List[Stance]) -> List[Stance]:
            facts = self.place_depth_caps(cands)
            if not facts:
                return cands
            kept = [c for c in cands if not (facts.get(id(c)) or {}).get("remove")]
            TT._countn(self, "place_depth_cap.phase2_removed", len(cands) - len(kept))
            log.info("%s: place depth cap in Phase 2: %d of %d (stance, position) pairs within the cap", self.gp.variant, len(kept), len(cands))
            return kept
        return keep

    def place_depth_cap_filter(self, cands: List[Stance]) -> Tuple[List[Stance], Dict[int, Dict[str, Any]]]:
        """:meth:`place_depth_caps` applied to the attempt's candidates: on - the ``remove`` ones leave the draws; shadow - all kept, the would-be
        removals counted (their outcomes are tallied per candidate in :meth:`sample_instance` and per attempt in :meth:`on_reject`). Kit counters
        ``place_depth_cap.candidates`` / ``.removed`` (on) / ``.shadow_would_remove`` (shadow) / ``.all_removed``."""
        if not self.place_depth_cap_applies() or not cands:
            return cands, {}
        facts = self.place_depth_caps(cands)
        if not facts:
            return cands, {}
        n_rm = sum(1 for f in facts.values() if f["remove"])
        TT._countn(self, "place_depth_cap.candidates", len(facts))
        log.info("%s: place depth cap (%s, basis %s): %d of %d candidates beyond it: %s", self.gp.variant, place_depth_cap_mode(), place_depth_cap_basis(), n_rm, len(facts),
                 [(f["depth_m"], f["cap_m"], f["s_min"], f["trunk"]) for f in facts.values()][:12])
        if place_depth_cap_mode() == "shadow":
            TT._countn(self, "place_depth_cap.shadow_would_remove", n_rm)
            return cands, facts
        TT._countn(self, "place_depth_cap.removed", n_rm)
        kept = [c for c in cands if not (facts.get(id(c)) or {}).get("remove")]
        if not kept:
            TT._count(self, "place_depth_cap.all_removed")
            caps = [f["cap_m"] for f in facts.values() if f["cap_m"] is not None]
            raise Reject("contact_unreachable", f"place depth cap (owner 3.6): none of {len(cands)} candidates places the target within the touch_push "
                                                f"map's reach of a collision-free push stance (depths {min(f['depth_m'] for f in facts.values()):.2f}-"
                                                f"{max(f['depth_m'] for f in facts.values()):.2f} m, caps {('%.2f-%.2f m' % (min(caps), max(caps))) if caps else 'none'})")
        return kept, facts

    def on_reject(self, err: Reject):
        """Owner 3.6 (shadow mode): an attempt drafted from a candidate the place depth cap would have removed is tallied by its rejection
        (``place_depth_cap.shadow.attempt.<reason>``)."""
        if getattr(self, "_pdc_shadow_drafted", False):
            TT._count(self, f"place_depth_cap.shadow.attempt.{err.reason}")
            self._pdc_shadow_drafted = False
        if getattr(self, "_lift_screen_shadow", False):
            log.info("TOUCH_LIFT_SCREEN shadow outcome: [%s] %s", err.reason, str(err)[:160])
            self._lift_screen_shadow = False
        if getattr(self, "_a1_screen_shadow", False):
            log.info("TOUCH_A1_SCREEN shadow outcome: [%s] %s", err.reason, str(err)[:160])
            self._a1_screen_shadow = False
        super().on_reject(err)

    def push_stance(self, draft: Draft, manip: Stance, grasp: GraspRecord) -> Optional[Stance]:
        """Record the ``push`` stance of ``push_stance_for``: whole-body check with the objects at the place, the path
        manipulation -> push carried (the box posed in the hand per the grasp record at its lift configuration; ``paths[].carried``),
        the brace gate and the goal observation stance re-evaluated there. ``Reject(stance_collision | lift_unreachable | no_path |
        path_collision)``."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        place_pos = getattr(self, "_touching_place_pos", None)
        if place_pos is None:
            place_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = G.t(draft.goal_poses[tname][1])
        t_place_pose = (G.t(place_pos), goal_quat)
        p_now = self.obj("partner").get_position_orientation()
        p_pose = (p_now[0].clone().float(), p_now[1].clone().float())
        # the contact lateral is searched (default -0.12 first, then further right / nearer): the first stance from which a push
        # class passes the chain + retreat clear pose (0918-174815: at -0.12 alone joint 4 / the clear pose failed 9/16 attempts)
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        if self.shelf_start is not None:  # the right hand's laterals mirrored for the left hand (+0.12, +0.16, +0.20, +0.08)
            laterals = [self.shelf_push_sign() * v for v in laterals]
        # owner 2026-09-18 20:48: a lateral from which the push AND a left brace pass together is preferred (the push shoves both
        # boxes; from the first push-feasible lateral the brace failed the two-arm solve while the left arm alone reached it,
        # 0918-210216); the first push-only lateral is the fallback
        st, tried, manip_ok = None, [], False
        fallback: Optional[Tuple[Optional[Stance], bool]] = None  # (stance or None for the manipulation stance, manip_ok)
        want_brace = bool(self.gen("brace_stance_search", True)) and (self._brace_gate or {}).get("ok") is not False
        # oracle-fix 2026-11-06 (owner on the touching#04 spray-bottle demo, push_height_sweep_on): every lateral is searched and the one whose
        # feasible push contact is LOWEST wins (with a brace on a tie, then the order); a later lateral only tries contacts up to the best
        # height found so far
        sweep = self.push_height_sweep_on()
        best: Optional[Tuple[Tuple[float, int], Optional[Stance], bool]] = None  # ((height, unbraced), stance or None for the manipulation stance, manip_ok)
        tip_ok: List[Optional[Stance]] = []  # the tip release's laterals whose place passes (None = the manipulation stance itself)
        diag: List[Dict[str, Any]] = []  # i7-contact 2026-11-07: the laterals without a push class, in numbers (TOUCH_PUSH_DIAG)
        plan = [(lat, self.push_stance_for(manip, t_place_pose, lateral=lat)) for lat in laterals]  # None = the manipulation stance already puts the contact there
        if self.shelf_tip() and shelf_release_in_gap_on(self):
            # i7-t03 round 2 (c): the manipulation stance first, then drives inside the free gap (no push-inherited contact laterals)
            plan = [(0.0, None)] + [(sg_d, st_d) for sg_d, st_d in self.shelf_release_in_gap_plan(manip, grasp)]
            if len(plan) == 1:
                TT._count(self, "shelf_release.no_room")
        if TT.contact_maps_on() and self.shelf_start is None and not self.shelf_tip():
            # owner A1 (2026-11-08): the push stance also moves toward / away from the edge, the stand-off chosen from the touch_push map
            # (the map-feasible (lateral, stand-off) candidates, brace-map ones first, nearest stand-off; collision-checked); none -> the laterals
            # at the manipulation stand-off as before (the exact chain decides, TOUCH_PUSH_DIAG explains)
            a1, a1_facts = self.push_stance_candidates(manip, t_place_pose, laterals, p_pose=p_pose)
            draft.summary.setdefault("touching", {})["push_stance_a1"] = dict(a1_facts, tried=[{k: f[k] for k in ("lateral", "stand_off", "brace_map")} for _l, _c, f in a1[:PUSH_STANCE_MAX_EXACT]])
            log.info("%s: A1 push stances %s: %s", self.gp.variant, json.dumps(a1_facts), [(f["lateral"], f["stand_off"], f["brace_map"]) for _l, _c, f in a1[:PUSH_STANCE_MAX_EXACT]])
            if a1:
                plan = [(lat, cand) for lat, cand, _f in a1[:int(self.gen("push_stance_max_exact", PUSH_STANCE_MAX_EXACT))]]
            else:
                TT._count(self, "touch_push.a1_no_map_stance")
        for lat, cand in plan:
            probe = cand if cand is not None else manip
            if self.shelf_tip():
                # oracle-fix 2026-11-04 round 4: the tip release keeps the second base pose for the PLACE (where the place passed under the push
                # mode) and pushes nothing - the first lateral whose place descent + clear pose pass
                place_why = self.place_infeasible_why(probe, t_place_pose, grasp)
                tried.append(f"{lat:+.2f}{' (manipulation)' if cand is None else ''}: tip release{'; place: ' + place_why if place_why else ''}")
                if place_why is None:
                    tip_ok.append(cand)
                    if cand is None:
                        break  # the manipulation stance places: no second base pose, no carried drive
                continue
            self._push_chain_fails = []
            self._push_chain_height = None
            cap = (None, best[0][0]) if (sweep and best is not None) else None
            kind = self.push_chain_feasible(probe, t_place_pose, p_pose, height_range=cap)
            h = getattr(self, "_push_chain_height", None) if kind is not None else None
            no_class = "no class"
            if kind is None:  # oracle-fix 2026-11-04: the distinct failure reasons of the classes tried (count x reason, numbers kept)
                import collections
                reasons = collections.Counter(f.split(": ", 1)[-1][:120] for f in self._push_chain_fails)
                no_class = f"no class{f' at or below {100 * cap[1]:.1f} cm' if cap else ''} [{len(self._push_chain_fails)} cands: " + \
                    " | ".join(f"{n}x {r}" for r, n in reasons.most_common(4)) + "]"
            if kind is None:
                diag.append(self._push_stance_diag(lat, probe, cand is None, t_place_pose))
            place_why = self.place_infeasible_why(probe, t_place_pose, grasp) if kind is not None else None
            both = self.push_chain_feasible(probe, t_place_pose, p_pose, with_brace=True, height_range=(h, h) if (sweep and h is not None) else None) \
                if (kind is not None and place_why is None and want_brace) else None
            # oracle-fix 2026-11-06 (push_gripper_open_on): an open class's height is named too (today's mid-height classes carry none), and a
            # brace that joined today's class instead of the open push names that class
            both_push = both.split("+", 1)[0] if both else None
            tried.append(f"{lat:+.2f}{f' s{float(probe.stand_off):.2f}' if probe.stand_off is not None else ''}{' (manipulation)' if cand is None else ''}: {kind or no_class}{f' @ {100 * h:.1f} cm' if h is not None else ''}"
                         f"{'; place: ' + place_why if place_why else ''}"
                         f"{'; brace: ' + both.split('+', 1)[1] if both else ('; brace: none' if kind and not place_why and want_brace else '')}"
                         f"{f' (with {both_push})' if both_push and both_push != kind else ''}")
            if kind is not None and place_why is None:
                if sweep:
                    rank = (round(float(h), 4) if h is not None else math.inf, 0 if (both is not None or not want_brace) else 1)
                    if best is None or rank < best[0]:
                        best = (rank, cand, cand is None)
                    continue
                if both is not None or not want_brace:
                    st, manip_ok = cand, cand is None
                    break
                if fallback is None:
                    fallback = (cand, cand is None)
        if tip_ok:
            # oracle-fix 2026-11-06 (touching#03 with the shelf back-off: 11 / 11 goal checks rejected 'local path manipulation -> push
            # collides', the book lifted inside the cube swept into its side board by the lateral drive): the manipulation stance when it
            # places, else each passing lateral in order until its stance and carried drive pass (``_record_push_stance``)
            if any(c is None for c in tip_ok):
                return None
            whys = []
            for cand in tip_ok:
                try:
                    log.info("%s: push stance search: %s", self.gp.variant, " | ".join(tried))
                    draft.summary.setdefault("touching", {})["push_stance_search"] = tried
                    st_rec = self._record_push_stance(draft, manip, grasp, cand, t_place_pose, p_pose)
                    if "drive_in_gap_m" in cand.extras:
                        TT._count(self, "shelf_release.drive_in_gap")
                    return st_rec
                except Reject as e:
                    whys.append(f"{float(cand.extras.get('lateral_shift_m', 0.0)):+.2f}: {e}")
                    tried.append(f"tip release drive {whys[-1][:160]}")
            if shelf_release_in_gap_on(self):
                TT._count(self, "shelf_release.no_stance")
            raise Reject("path_collision", f"touching tip release: no passing lateral's stance + carried drive is clear ({'; '.join(whys)})")
        if best is not None:
            st, manip_ok = best[1], best[2]
            draft.summary.setdefault("touching", {})["push_stance_lowest_contact_m"] = None if math.isinf(best[0][0]) else best[0][0]
        if st is None and fallback is not None:
            st, manip_ok = fallback
        draft.summary.setdefault("touching", {})["push_stance_search"] = tried
        if st is None:
            if manip_ok:
                return None  # the manipulation stance itself passes: no second base pose
            if self.shelf_tip():
                if shelf_release_in_gap_on(self):
                    TT._count(self, "shelf_release.no_stance")
                raise Reject("place_unreachable", f"touching tip release: the place fails from every lateral ({'; '.join(tried)})")
            log.info("TOUCH_PUSH_DIAG %s", json.dumps({"variant": self.gp.variant, "laterals": diag}, default=str))
            raise Reject("contact_unreachable", f"touching push: no push class passes from any push stance lateral ({'; '.join(tried)})")
        log.info("%s: push stance search: %s", self.gp.variant, " | ".join(tried))
        return self._record_push_stance(draft, manip, grasp, st, t_place_pose, p_pose)

    def _record_push_stance(self, draft: Draft, manip: Stance, grasp: GraspRecord, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Stance:
        """The chosen push stance ``st`` checked whole-body with the objects at the place and recorded, with the carried drive manipulation
        -> ``st`` (:meth:`push_stance`). ``Reject(stance_collision | lift_unreachable | no_path | path_collision)``."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        try:
            TT.whole_body_check_at(self, [st], {tname: t_place_pose, pname: p_pose}, require=True, reason="stance_collision")
        finally:
            # the box goes back to its init pose: the program check and the grasp checks read the live world (0918-170003: the
            # box left at the place, 3-6 cm from the init, made the checker's grasp IK fail on 4/12 attempts)
            self.obj("target").set_position_orientation(*[G.t(v) for v in draft.object_poses[tname]])
            self.obj("target").keep_still()
        self.sidecar.add_stance(st)
        TT.sync_moved(self.kit, [self.obj("target")])  # train-fast 2026-10-31: the box alone moved back
        if self.shelf_start is None and self.carry_rule() == "auto":
            self.auto_carry_height(draft, grasp, manip, st, p_pose)
        # the drive is swept with the arm at the LIFT configuration (the box carry_height above the grasp, as the oracle drives
        # it), not the stance's untucked arms - the box posed in the untucked hand stands inside the table (0918-165104)
        lift_end = PC.lift_end_pose(grasp)
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            r_lift = self.lift_with_touch(grasp, lambda: self.ikc.exact([lift_end], manip, grasp.arm, ignore_for_last=self.support_objects(), cache=False))
        if not r_lift.ok or r_lift.q is None:
            # owner 2026-10-12 item 7d: swept with the untucked hand instead, the box stood inside the table and the lift failure read
            # as path_collision (touching#02 c2_k0 attempt 7)
            raise Reject("lift_unreachable", f"lift ({grasp.carry_height:.2f} m above the grasp) has no configuration at the {manip.name} stance for the carried "
                                             f"drive to the push stance: {r_lift.reason}")
        idx = self.kit.robot.arm_control_idx
        arm_q = {"right": r_lift.q[idx["right"]].clone(), "left": r_lift.q[idx["left"]].clone()}
        path = self.record_path(manip, st, carried=tname, arm=grasp.arm, arm_q=arm_q)  # Reject(no_path | path_collision) fails the attempt
        st.carried = tname
        st.extras["path_length_m"] = getattr(path, "length_m", None)
        self._brace_gate = self.brace_gate_at(st, t_place_pose, p_pose)
        self.sidecar.set_goal(st.base_pose2d, st.translate, st.tilt, arms="same")  # the goal observation is taken from the push stance
        draft.summary["push_stance"] = {"lateral_shift_m": st.extras["lateral_shift_m"], "base_pose2d": [float(v) for v in st.base_pose2d],
                                        "path_length_m": st.extras["path_length_m"], "brace_gate": self._brace_gate,
                                        "stand_off": st.stand_off, "stand_off_shift_m": st.extras.get("stand_off_shift_m")}
        if st.extras.get("stand_off_shift_m"):
            TT._count(self, "touch_push.stand_off_moved")
        log.info("%s: push stance recorded %+.3f m along the edge (path %.2f m, box in hand)", self.gp.variant, st.extras["lateral_shift_m"],
                 float(st.extras["path_length_m"] or 0.0))
        return st

    def drive_partner_gap(self, t_pose: G.Pose, p_pose: G.Pose, frm: Stance, to: Stance) -> float:
        """4A: the clearance (m) between the partner's footprint and the target's, swept along the base move ``frm`` -> ``to`` with the box
        held rigidly in the hand (the push stance keeps the yaw: a translation), the target from ``t_pose``."""
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        move = (G.t(to.base_xy) - G.t(frm.base_xy)).float()
        swept = G.convex_hull_2d(th.cat([t_fp, t_fp + move[None, :]]))
        return TT.polygon_gap(swept, self.world.footprint(self.hull["partner"], p_pose))

    def auto_carry_height(self, draft: Draft, grasp: GraspRecord, manip: Stance, st: Stance, p_pose: G.Pose) -> None:
        """4A ``carry_clears: "auto"``: the carry clears only the table unless the carried drive manipulation -> ``st`` takes the box within
        ``carry_clearance_m`` of the partner's footprint - the base moves toward the partner by the push stance's shift (13-17 cm in
        touching#02), farther than the start gap in most of its draws -, then it clears the partner's top as before. The verdict goes to
        ``summary.carry_lift``."""
        tname = self.bindings.one("target").name
        gap = self.drive_partner_gap(draft.object_poses[tname], p_pose, manip, st)
        over = gap < float(self.gen("carry_clearance_m", CARRY_CLEARANCE_M))
        grasp.carry_height = self.carry_height_over(over)
        draft.summary["carry_lift"] = {"rule": "auto", "clears": "partner" if over else "table", "drive_gap_m": round(gap, 4), "height_m": round(float(grasp.carry_height), 4)}
        if over:
            log.info("%s: the carried drive to the push stance passes %.3f m from the partner: the lift clears its top (%.2f m)", self.gp.variant, gap, grasp.carry_height)

    def pre_place_frame_ok(self, pose: G.Pose, cam: G.Pose) -> bool:
        """i7-plateqtf2 (owner decision 2026-11-08 02:35 item 3, a camera-frame rule change): the draw's pre-place frame check is the program
        check's (``program_check.fingertip_frame_margin``: the fingertip point inside QA's 5 % margin plus 2 cm slack at the fingertip) instead
        of the eef box (``IKChecker.eef_in_frame``), which was stricter than both the program check and QA (fingertip only) there.
        ``ISPATIALGYM_PREPLACE_FRAME_RULE=box``: the box as before. Kit ``preplace_frame.checked`` / ``.rejected`` / ``.removed`` (the box
        would have rejected, the fingertip rule passes)."""
        if pre_place_frame_rule() == "box":
            return bool(self.ikc.eef_in_frame(pose, cam))
        fm = PC.fingertip_frame_margin(self.kit.world.camera_model, cam, pose)
        TT._count(self, "preplace_frame.checked")
        if not fm["inside"]:
            TT._count(self, "preplace_frame.rejected")
        elif not self.ikc.eef_in_frame(pose, cam):
            TT._count(self, "preplace_frame.removed")
        return bool(fm["inside"])

    def pre_place_frame_margin(self, pose: G.Pose, cam: G.Pose) -> float:
        """The pre-place's frame margin (px; >= 0 passes) under :func:`pre_place_frame_rule` (the frame ranking's score)."""
        if pre_place_frame_rule() == "box":
            return float(self.ikc.eef_frame_margin(pose, cam))
        return float(PC.fingertip_frame_margin(self.kit.world.camera_model, cam, pose)["margin_px"])

    def carry_frame_check(self, draft: Draft, grasp: GraspRecord, manip: Stance, st_place: Stance, lift_end: G.Pose, pre_place: G.Pose, r_lift) -> None:
        """i7-framing: the fingertips along the oracle's protective via-above of the carry (``PC.via_above_carry_poses``: rise to max(lift end,
        pre-place) + 0.12 m, across, down - taken when the straight leg passes a protected object; under_small_03 300150: gripper_out_of_fov
        there, the straight carry framed), judged by the pre-place's fingertip rule. Three switches (all default off; owner 2026-11-08):

        - ``ISPATIALGYM_CARRY_FRAME_CHECK=1`` (5.1, not a default - 5d.1): out of frame -> ``Reject("gripper_not_in_frame")``, unless the
          torso lift (below) rescues it;
        - ``ISPATIALGYM_CARRY_TORSO_LIFT=1`` (5.1 / 5d.2): where only the arm lift leaves the frame (the straight carry framed),
          :meth:`carry_torso_lift` is recorded as ``grasps[].carry_torso`` (the oracle executes it on the via-above); alone it never rejects;
        - ``ISPATIALGYM_CARRY_BAND`` (5d.3, the middle option; DEFAULT ON since 5s.1, ``0`` = off): the straight carry's NOMINAL protect margin (:meth:`carry_nominal_margin`,
          ``carry_margin.carry_protect_margin``) picks the band - ``above`` (the lift almost surely does not fire): no check; ``below``
          (it fires most often): the check, out of frame -> reject; ``near``: the torso lift when possible (it costs nothing when the
          lift does not fire and keeps the frame when it does), else the check is SKIPPED, never a reject (owner 2026-11-08 5s.1: the
          lift fires in ~5 % of the near band's carries).

        Kit ``framing.carry.checked`` / ``.failed`` / ``.skipped_drive`` / ``.torso_lift_used`` / ``.torso_lift_failed.<why>`` /
        ``.accepted_out`` (torso-only mode, nothing rescued) and ``framing.carry.band.{below,near,above}`` / ``.band.rejected`` (below
        only) / ``.band.torso_lift_used`` / ``.band.near_skipped[.<no_trunk|frame|leg_ik|collision|...|straight_out>]``."""
        if st_place is not manip:
            TT._count(self, "framing.carry.skipped_drive")  # the place is made from another base pose (a carried drive first): not modelled
            return
        check = PC.carry_frame_check_on()
        band = None
        rec: Dict[str, Any] = {}
        if PC.carry_band_on() and not check:
            m = self.carry_nominal_margin(grasp, lift_end, pre_place)
            band = PC.carry_band(m.get("margin_m"))
            TT._count(self, f"framing.carry.band.{band}")
            rec["band"] = dict(m, band=band)
            draft.summary["carry_frame"] = dict(rec)
            if band == "above":
                return
        TT._count(self, "framing.carry.checked")
        model, cam = self.kit.world.camera_model, self.camera_pose_at(manip)
        out = PC.first_out_of_frame(model, cam, PC.via_above_carry_poses(lift_end, pre_place))
        if out is None:
            return
        TT._count(self, "framing.carry.failed")
        straight = PC.first_out_of_frame(model, cam, {"straight": list(C.line_waypoints(lift_end, pre_place, PC.CARRY_FRAME_LEG_WAYPOINTS, endpoints=True))})
        rec.update({"via_above_out": out, "straight_out": straight, "torso_lift": None})
        if straight is None and (PC.carry_torso_lift_on() or band == "near"):
            ch, why = self.carry_torso_lift(grasp, manip, lift_end, pre_place, r_lift)
            rec["torso_lift"] = ch if ch is not None else {"failed": why}
            if ch is not None:
                TT._count(self, "framing.carry.torso_lift_used")
                if band is not None:
                    TT._count(self, "framing.carry.band.torso_lift_used")
                grasp.extras["carry_torso"] = ch
                draft.summary["carry_frame"] = rec
                return
            TT._count(self, f"framing.carry.torso_lift_failed.{why}")
        draft.summary["carry_frame"] = rec
        if band == "near":
            # owner 2026-11-08 5s.1: the near band never rejects - the lift fires in ~5 % of its carries (22 of 410 on the replay), so
            # where the torso lift cannot keep the frame the check is SKIPPED (only far-below carries are rejected)
            why_skip = "straight_out" if straight is not None else (rec["torso_lift"] or {}).get("failed", "torso")
            TT._count(self, "framing.carry.band.near_skipped")
            TT._count(self, f"framing.carry.band.near_skipped.{why_skip}")
            rec["near_skipped"] = why_skip
            return
        if not check and band is None:
            TT._count(self, "framing.carry.accepted_out")  # torso-only mode: nothing rescued, nothing rejected (the rule as before)
            return
        if band is not None:
            TT._count(self, "framing.carry.band.rejected")
        raise Reject("gripper_not_in_frame", f"lifted carry (the oracle's protective via-above, +{PC.ORACLE_PROTECT_ABOVE_M:.2f} m"
                                             f"{'' if band is None else ', band ' + band}): fingertip (+2 cm slack) outside "
                                             f"the manipulation image at {out['key']} {out['index'] + 1}/{out['n']} (u {out['u']}, v {out['v']}, depth {out['depth_m']} m)"
                                             f"{'' if straight is None else '; the straight carry too'}")

    def carry_nominal_margin(self, grasp: GraspRecord, lift_end: G.Pose, pre_place: G.Pose) -> Dict[str, Any]:
        """i7-framing 5d.3: the straight carry's nominal protect margin (``carry_margin.carry_protect_margin``) - the carried target's
        base-aligned collision box (``oracle.protection.object_box``, in its object frame) attached by ``eef_in_object``, the hand box,
        against the partner's base-aligned box (IORS small's protected goal object; the target is in the hand), along lift end ->
        [the recorded carry vias] -> pre-place."""
        import numpy as np

        from ispatialgym.oracle import protection as PROT
        from ispatialgym.sampling.v2 import carry_margin as CMg

        tgt, par = self.obj("target"), self.obj("partner")
        tb, pb = PROT.object_box(tgt), PROT.object_box(par)
        if tb is None or pb is None or grasp.eef_in_object is None:
            return {"margin_m": float("inf"), "why": "no box"}
        tp, tq = tgt.get_position_orientation()
        R_obj = CMg.quat_to_mat(G.t(tq).float().tolist())
        c_local = R_obj.T @ (np.asarray(G.t(tb[0]).float().tolist()) - np.asarray(G.t(tp).float().tolist()))
        obj_box = (c_local, np.asarray(G.t(tb[2]).float().tolist()))
        par_box = (np.asarray(G.t(pb[0]).float().tolist()), np.asarray(G.t(pb[1]).float().tolist()), np.asarray(G.t(pb[2]).float().tolist()))
        vias = [v for v in (getattr(grasp, "carry", None) or []) if v is not None] if isinstance(getattr(grasp, "carry", None), (list, tuple)) else []

        def pl(p):
            return [float(v) for v in G.t(p[0]).reshape(-1)], [float(v) for v in G.t(p[1]).reshape(-1)]
        eio = pl(grasp.eef_in_object)
        return CMg.carry_protect_margin([pl(lift_end)] + [pl(v) for v in vias] + [pl(pre_place)], eio, obj_box, [par_box])

    def carry_torso_lift(self, grasp: GraspRecord, manip: Stance, lift_end: G.Pose, pre_place: G.Pose, r_lift) -> Tuple[Optional[Dict[str, Any]], str]:
        """i7-framing (owner 2026-11-08 5.1, the owner's torso suggestion; ``ISPATIALGYM_CARRY_TORSO_LIFT=1``): the carry's protective rise made
        by the TORSO, arm joints held - the oracle (``execute._carry_torso_lift``) sweeps the trunk to ``carry_torso`` (the hand and the head
        camera move together: the image is unchanged), tracks the straight carry at that trunk (``Mt ∘`` the straight leg), sweeps the trunk
        back (the hand lands on the pre-place). The trunk: :func:`program_check.carry_torso_choice` (>= 0.12 m rise at both ends, <= 4 cm
        sideways, <= 0.1 rad turn). Checked here with the object attached: the straight leg at the carry trunk (tracked exact IK, framed),
        the trunk sweeps with the arm at the lift-end / pre-place configuration (whole body, supports ignored, 4 steps each).
        Returns (record, "") or (None, why)."""
        import dataclasses

        from ispatialgym.oracle import r1pro_model as M

        t0, k0 = float(manip.translate), float(manip.tilt)
        ch = PC.carry_torso_choice(manip.base_pose2d, (t0, k0), [lift_end[0], pre_place[0]], floor_z=float(self.kit.floor_z))
        if ch is None:
            return None, "no_trunk"
        if r_lift is None or getattr(r_lift, "q", None) is None:
            return None, "no_lift_q"
        carry_st = dataclasses.replace(manip, translate=float(ch["translate"]), tilt=float(ch["tilt"]))
        Mt = PC.torso_move(manip.base_pose2d, (t0, k0), (ch["translate"], ch["tilt"]), floor_z=float(self.kit.floor_z))
        a2, b2 = G.compose(Mt, lift_end), G.compose(Mt, pre_place)
        leg = [a2] + list(C.line_waypoints(a2, b2, PC.CARRY_FRAME_LEG_WAYPOINTS))
        if PC.first_out_of_frame(self.kit.world.camera_model, self.camera_pose_at(carry_st), {"torso_leg": leg}) is not None:
            return None, "frame"
        idx = [C.JOINT_INDEX[n] for n in M.ARM_JOINTS[grasp.arm]]
        supports = self.support_objects()
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            r_leg = self.ikc.waypoints_ok(leg, carry_st, grasp.arm, ignore_all=supports, cache=False, approached=[])
            if not r_leg.ok or r_leg.q is None:
                return None, "leg_ik"
            q_lift, q_pre = G.t(r_lift.q).reshape(-1)[idx], G.t(r_leg.q).reshape(-1)[idx]
            sweep = []
            for q_arm in (q_lift, q_pre):
                for s in (0.25, 0.5, 0.75, 1.0):
                    st_s = dataclasses.replace(manip, translate=t0 + s * (float(ch["translate"]) - t0), tilt=k0 + s * (float(ch["tilt"]) - k0))
                    sweep.append(self.ikc.q_locked(st_s, **{f"{grasp.arm}_arm": q_arm}))
            with self.ikc.ignoring(list(supports)):
                free = self.kit.collision_free(th.stack(sweep), margin=0.005)
        if not bool(th.all(free)):
            return None, "sweep_collision"
        return dict(ch, leg_waypoints=len(leg)), ""

    def precontact_frame_ok(self, what: str, pose: G.Pose, cam: G.Pose, obj: Optional[Tuple[str, G.Pose]] = None) -> Tuple[bool, bool]:
        """i7-framing (owner decision 2026-11-08 5.2): ``(box, fingertip)`` verdicts of a pre-grasp / touching pre-contact eef pose - the eef
        box (``IKChecker.eef_in_frame``) and the pre-place's fingertip rule (``program_check.fingertip_frame_margin``: QA's 5 % + 2 cm slack).
        Counts ``precontact_frame.<what>.checked`` / ``.removed`` (box fails, fingertip passes) / ``.added`` (box passes, fingertip fails) and,
        for a removed one whose ``obj`` = (hull key, object pose) is not wholly in the image (``TT.in_frame`` of its hull corners, 5 %),
        ``.removed_object_out``. The caller applies :func:`precontact_frame_rule`."""
        box = bool(self.ikc.eef_in_frame(pose, cam))
        tip = bool(PC.fingertip_frame_margin(self.kit.world.camera_model, cam, pose)["inside"])
        key = "precontact_frame." + what.replace(" ", "_").replace("-", "_")
        TT._count(self, key + ".checked")
        if box != tip:
            TT._count(self, key + (".removed" if tip else ".added"))
            if tip and obj is not None and not TT.in_frame(self, cam, TT.hull_corners_world(self.hull[obj[0]], obj[1])):
                TT._count(self, key + ".removed_object_out")
        return box, tip

    def require_in_frame(self, cam: G.Pose, poses: Dict[str, Optional[G.Pose]], objects: Optional[Dict[str, Tuple[str, G.Pose]]] = None):
        """Every given eef pose must project inside the manipulation image with the 5 % margin (``IKChecker.eef_in_frame``
        with the shared camera FK of ``camera_pose_at``); ``Reject("gripper_not_in_frame")`` names the first that does not. Opt-in
        ``generation.grasp_frame_rule: "fingertip"`` (oracle-fix 2026-11-04 round 4, IORSLarge's rule): QA's own criterion instead - the fingertip
        point (``conventions.fingertip``) inside the image with the 5 % margin (t03tip3 opryhp: the shelf pre-grasp's +-6 cm eef box left the image).
        i7-framing (owner 2026-11-08 5.2): the pre-grasp and the touching pre-contacts (:data:`PRECONTACT_FRAME_POSES`) take
        :func:`precontact_frame_rule` (``fingertip`` = the pre-place's rule); ``objects`` = {pose name: (hull key, object pose)} for the
        ``removed_object_out`` counter of :meth:`precontact_frame_ok`."""
        tip_rule = str(self.gen("grasp_frame_rule") or "box") == "fingertip"
        for what, pose in poses.items():
            if pose is None:
                continue
            if what in PRECONTACT_FRAME_POSES:
                _box, tip = self.precontact_frame_ok(what, pose, cam, (objects or {}).get(what))
                if precontact_frame_rule() == "fingertip":
                    if not tip:
                        raise Reject("gripper_not_in_frame", f"{what} fingertip (+2 cm slack) outside the manipulation image (5 %)")
                    continue
            if what == "pre-place" and pre_place_frame_rule() == "fingertip":
                # i7-plateqtf2 (owner 2026-11-08 02:35 item 3): the program check's criterion at the pre-place (fingertip + 2 cm slack)
                if not self.pre_place_frame_ok(pose, cam):
                    raise Reject("gripper_not_in_frame", f"{what} fingertip (+2 cm slack) outside the manipulation image (5 %)")
                continue
            if tip_rule:
                tip = C.fingertip((G.t(pose[0]).float(), G.t(pose[1]).float()))
                ok = bool(self.kit.world.camera_model.all_inside(cam, G.t(tip).reshape(1, 3), IK.IN_FRAME_MARGIN))
            else:
                ok = self.ikc.eef_in_frame(pose, cam)
            if not ok:
                raise Reject("gripper_not_in_frame", f"{what} {'fingertip' if tip_rule else 'eef pose'} outside the manipulation image (5 %)")

    def touching_contacts(self, draft: Draft, manip: Stance):
        """§6.11 touching: push contact on the target's right face (right arm) toward the partner, brace pre-contact +
        contact of the closed left gripper on the partner's far (left) side; both IK-checked (§4.6 step 6). Shelf layout
        (touching#03): the push on the target's face away from the partner along the shelf push direction by the arm on the
        target's side (the left arm pushing to the right), recorded with ``layout`` = ``TOUCHING_SHELF_LAYOUT`` for the oracle's
        program; no brace."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        # the push starts at the *place* (nextto) position the program sets the target down at and ends at the settled
        # contact position (+5 mm of pressing): ``push_distance`` (the old code pushed from the contact position onward)
        goal_quat = draft.goal_poses[tname][1]
        place_pos = getattr(self, "_touching_place_pos", None)
        if place_pos is None:
            place_pos = th.tensor(draft.pose_targets[tname]["position"])
        t_pose = (G.t(place_pos), G.t(goal_quat))
        p_now = self.obj("partner").get_position_orientation()
        p_pose = (p_now[0].clone().float(), p_now[1].clone().float())
        tangent = manip.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])  # toward the partner = robot's left
        p_init, b_point = self.touching_points(t_pose, p_pose, manip)
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
        # the push spans the *measured* gap at the place (target's left face at the place pose -> partner's right face along
        # the push direction) plus the margin; the recorded goal lies past the contact by the margin
        t_hull_place = self.world.hull_at(self.hull["target"], t_pose)
        gap_at_place = float((p_hull[:, :2] @ tangent).min() - (t_hull_place[:, :2] @ tangent).max())
        if self.shelf_start is not None:
            direction = self.shelf_push_direction(manip)
            gap_at_place = gap_along(t_hull_place, p_hull, direction)
            self._brace_gate = self.brace_gate_at(manip, t_pose, p_pose)
            draft.summary.setdefault("touching", {}).update({"layout": TOUCHING_SHELF_LAYOUT, "push_arm": self.shelf_push_arm(),
                                                             "push_direction": [round(float(v), 4) for v in direction]})
        dist = self.push_distance(gap_at_place)
        draft.summary.setdefault("touching", {}).update({"gap_at_place_m": round(gap_at_place, 4), "push_distance_m": round(dist, 4), "push_margin_m": round(self.push_margin(), 4)})
        pair = self.pair_goal_align()
        if pair is not None and pair["push_max_height_m"] is not None:
            draft.summary["touching"]["push_max_height_m"] = pair["push_max_height_m"]
        open_rule = self.brace_open_rule(b_point, face_z, -direction, p_hull, p_init) if self.shelf_start is None else None
        if open_rule is not None:
            draft.summary["touching"]["brace_open_when_wide"] = open_rule
            log.info("%s: brace_open_when_wide: the partner's braced face is %.3f m wide at z %.3f (min %.3f): %s", self.gp.variant, open_rule["face_width_m"],
                     open_rule["height_m"], open_rule["min_width_m"], f"open-finger stop required, spread {open_rule['spread_m']}" if open_rule["required"]
                     else "narrow - today's brace candidates")
        # closed pads for both contacts (§3.6); the push arm first, then the brace with the push arm locked (§4.6 step 6).
        # Push candidates: the pads closing horizontally (default) then vertically (the other wrist roll — the horizontal
        # one put the right pre-contact out of reach on 5/8 attempts of the sanitary-napkins loop); brace candidates in
        # order (pads on the far face / top-down blade beside it, ``touching_brace_candidates``)
        why: List[str] = []
        t_hull_w = self.world.hull_at(self.hull["target"], t_pose)
        t_face_z = (float(t_hull_w[:, 2].min()), float(t_hull_w[:, 2].max()))
        gate = self._brace_gate or {"ok": None, "why": "no gate verdict"}
        try_brace = gate.get("ok") is not False  # False = no joint two-arm placement at this trunk: push-only from the start
        ignore_push = self.support_objects() + [self.obj("target")]
        ignore_brace = self.support_objects() + [self.obj("partner")]
        push_chosen = None  # (kind, pre, init, goal, extras, q of the right arm)
        brace_chosen = None  # (pre, contact, kind, extras)
        open_fallback: Optional[Dict[str, Any]] = None  # oracle-fix 2026-11-06: the open push given up for today's class + the brace
        # the collision world holds the target at its INIT pose, but the push happens after the place: the box is posed at the
        # place for the chain's checks (the pre-contact of the placed box lies inside the box at its init pose otherwise -
        # 0918-170804: 'exact IK failed at pose 1 of 3' on 6/12 attempts)
        with self.posed("target", t_pose):
            chosen, dec_p, fails_p = self._touching_push_choice(manip, t_pose, p_init, t_face_z, direction, dist, ignore_push)
            why.extend(fails_p)
            if chosen is not None:
                push_kind, push_init, push_extras = chosen.proposal.payload
                push_pre, _, push_goal = chosen.proposal.chain
                push_chosen = (push_kind, push_pre, push_init, push_goal, _with_orient(push_extras, dec_p))
                # (2) a brace when the joint band gate admitted the position: the first two-arm candidate that passes
                if try_brace:
                    b_chosen, dec_b, fails_b = self._touching_brace_choice(manip, chosen.proposal.chain, t_pose, p_pose, b_point, face_z, direction, dist, p_hull, p_init,
                                                                           ignore_push, ignore_brace, push_fingers_q=push_fingers(push_extras))
                    why.extend(f"{push_kind}/{w}" for w in fails_b)
                    if b_chosen is None and push_fingers(push_extras) != IK.FINGER_CLOSED:
                        # oracle-fix 2026-11-06 (push_gripper_open_on): no brace joins the open push - today's (closed) classes with the brace
                        # before the open push is kept alone (it never costs the instance its brace; push_chain_feasible chose the stance so)
                        closed, dec_c, fails_c = self._touching_push_choice(manip, t_pose, p_init, t_face_z, direction, dist, ignore_push, open_classes=False)
                        why.extend(fails_c)
                        if closed is not None:
                            b_closed, dec_bc, fails_bc = self._touching_brace_choice(manip, closed.proposal.chain, t_pose, p_pose, b_point, face_z, direction, dist,
                                                                                     p_hull, p_init, ignore_push, ignore_brace)
                            why.extend(f"{closed.proposal.payload[0]}/{w}" for w in fails_bc)
                            if b_closed is not None:
                                open_fallback = {"open_push": push_kind, "kept": closed.proposal.payload[0], "why": "no brace joins the open push"}
                                push_kind, push_init, push_extras = closed.proposal.payload
                                push_pre, _, push_goal = closed.proposal.chain
                                push_chosen = (push_kind, push_pre, push_init, push_goal, _with_orient(push_extras, dec_c))
                                b_chosen, dec_b = b_closed, dec_bc
                    if b_chosen is not None:
                        pre, contact, kind, extras = b_chosen.proposal.payload
                        brace_chosen = (pre, contact, kind, _with_orient(extras, dec_b))
        if push_chosen is None:
            d_i, l_i = manip.local(G.t(p_init)[:2])
            raise Reject("contact_unreachable", f"touching push: no push candidate passes the exact IK from the {manip.name} stance (push init at depth {d_i:.2f} "
                                                f"lateral {l_i:+.2f} z {float(p_init[2]):.2f}, face z {t_face_z[0]:.2f}-{t_face_z[1]:.2f}, trunk {manip.translate}, "
                                                f"{manip.tilt:.2f}; {'; '.join(why[-3:])})")
        if brace_chosen is None and open_rule is not None and open_rule["required"]:
            if not try_brace:
                why_open = f"no joint two-arm placement ({gate.get('why')})"
            elif open_rule["spread_m"] is None:
                why_open = (f"the push line lies {open_rule['push_line_from_face_centre_m']:+.3f} m off the face's centre: no open spread "
                            f"{CPZ.brace_wide_spreads(self)} straddles it {BRACE_OPEN_END_MARGIN_M:.2f} m from the face's ends")
            else:
                why_open = f"no open-finger stop passed ({'; '.join([w for w in why if '/' in w][-3:]) or 'no candidate at a brace height'})"
            raise Reject("brace_open_required", f"touching: the partner's braced face is {open_rule['face_width_m']:.3f} m wide at the brace height "
                                                f"(>= {open_rule['min_width_m']:.3f} m: generation.brace_open_when_wide requires the open-finger stop) but {why_open}")
        push_kind, push_pre, push_init, push_goal, push_extras = push_chosen
        speed = self.push_speed()
        if speed is not None:
            push_extras = dict(push_extras, push_speed_m_s=speed)  # the oracle's pad speed for every leg of this push
        draft.summary.setdefault("touching", {}).update({"push_kind": push_kind, "classes_failed": why[:6], **{k: v for k, v in push_extras.items() if v is not None}})
        if self.push_gripper_open_on():
            # oracle-fix 2026-11-06: the hand the push records - the open class, or today's class after the brace fallback
            draft.summary["touching"]["push_gripper_open"] = {"open": push_fingers(push_extras) != IK.FINGER_CLOSED, "fingers_opening_m": push_extras.get("fingers_opening_m"),
                                                              **(open_fallback or {})}
        log.info("%s: touching push class %s from the %s stance%s", self.gp.variant, push_kind, manip.name, f" (failed before it: {'; '.join(why[:10])})" if why else "")
        if touch_push_stroke_m() is not None:  # i7-t04push (TOUCH_PUSH_STROKE_ENV); off: today's record
            push_extras = dict(push_extras, **self.touch_push_strokes(push_kind, push_extras, dist, root_xy=G.t(t_pose[0])[:2], direction=direction))
        push_rec = ContactRecord("push", "right", push_pre, push_init, None, push_goal, direction, dist, object=tname, extras=dict(push_extras, pose_class=push_kind))
        if self.shelf_start is not None:
            push_rec.arm = self.shelf_push_arm()
            push_rec.extras["layout"] = TOUCHING_SHELF_LAYOUT
        push_rec.stance = manip.name  # the stance the chain was checked from ("manipulation" or the second base pose "push")
        self.sidecar.add_contact(push_rec)
        if TT.contact_maps_on() and self.shelf_start is None:
            self._note_contacts(draft, manip, t_pose, p_pose, push_kind, push_extras, brace_chosen)
        if brace_chosen is not None:
            pre, contact, kind, extras = brace_chosen
            brace_rec = TT.BraceRecord("brace", "left", pre, contact, direction=-direction, brace_kind=kind, object=pname, extras=extras)
            brace_rec.stance = manip.name
            TT.count_brace_opening(self, extras)
            self.sidecar.add_contact(brace_rec)
            draft.summary["brace"] = {"present": True, "kind": kind, "gate": gate, **{k: v for k, v in extras.items() if v is not None}}
            manip.arms = ["right", "left"]
            manip.contacts = ["push", "brace"]
        else:
            # push-only touching: ``contacts.brace`` absent, the manipulation stance right-arm only; the program is
            # transition -> push -> retreat (oracle agent 7deabbed); ``summary.brace.present`` = False, why recorded
            draft.summary["brace"] = {"present": False, "gate": gate, "candidates_failed": len([w for w in why if "/" in w]),
                                      "why": "; ".join(why[-3:]) if try_brace else str(gate.get("why"))}
            manip.arms = ["right"]
            if self.shelf_start is not None:
                manip.arms = [self.shelf_push_arm()]
            manip.contacts = ["push"]
            self.sidecar.contacts.pop("brace", None)

    def _note_contacts(self, draft: Draft, st: Stance, t_pose: G.Pose, p_pose: G.Pose, push_kind: str, push_extras: Dict[str, Any], brace_chosen) -> None:
        """Owner A6 (2026-11-08): the library entries this touching instance executes (the push's class label as the library builds it -
        ``kind@<contact height cm>`` when the class records a height -, the brace's kind), noted for the contact maps' learned overlay
        (``contact_overlay.note_used``; harvested after a successful demo). Advisory: never rejects."""
        from ispatialgym.sampling import contact_overlay as CO

        try:
            pm, bm = self.contact_maps_of_pair()
            h = (push_extras or {}).get("contact_height_m")
            cls = push_kind if h is None else f"{push_kind}@{100.0 * float(h):.1f}cm"
            CO.note_used(draft.summary, pm, "touch_push", cls, TT.contact_map_check(pm, st, t_pose), "target", stance=st.name)
            if brace_chosen is not None:
                CO.note_used(draft.summary, bm, "touch_brace", str(brace_chosen[2]), TT.contact_map_check(bm, st, p_pose), "partner", stance=st.name)
        except Exception as ex:  # noqa: BLE001 - the overlay is advisory
            log.info("%s: contact overlay note skipped: %r", self.gp.variant, ex)

    def _touching_push_choice(self, st: Stance, t_pose: G.Pose, p_init: th.Tensor, t_face_z: Tuple[float, float], direction: th.Tensor, dist: float,
                              ignore_push: Sequence[Any], arm: Optional[str] = None, height_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                              open_classes: bool = True):
        """The push of the touching task (the box posed at the place by the caller): ``touching_push_candidates`` in their order,
        a class's two wrist turns chosen by ``orient.choose`` (owner 2026-09-27). A candidate passes when (0) the hand beyond the
        finger profile (wrist camera, knuckles, body) clears the box at the contact and the goal, (1) the [pre, init, goal] chain
        passes the exact IK on its own (the requirement; the brace is optional, owner 2026-09-16 17:10: the axial rule) and (2)
        the oracle's retreat after the push - 10 cm back along the approach axis from the push GOAL (above it for a blade) - has a
        configuration (a class without one dies in the program check, 9/16 of 0918-173814). ``arm``: the push arm (default the
        right arm; the shelf layout's :meth:`shelf_push_arm`). :meth:`pair_goal_align`'s ``push_max_height_m``: only the candidates whose
        contact (the fingertips of a blade, else the eef's height at the face) is at most that high above the target's bottom.
        :meth:`push_height_sweep_on` (oracle-fix 2026-11-06): the candidates by contact height (:meth:`swept_push_candidates`), the hand
        checked against the hull, within ``height_range`` (m above the target's bottom, either end None = open) when given; the labels
        carry the height (``push_tip_h-@5.0cm``). :meth:`push_gripper_open_on` (oracle-fix 2026-11-06): the open gripper's classes
        (:meth:`open_push_candidates`) before them, checked with their open fingers (:func:`push_fingers`) and the hand against the hull;
        ``open_classes=False``: today's classes alone (the brace fallback of :meth:`touching_contacts` / :meth:`push_chain_feasible`).
        Returns ``orient.choose_first``'s ``(chosen, decision, failures)``."""
        if arm is None:
            arm = self.shelf_push_arm() if self.shelf_start is not None else "right"
        sweep = t_pose is not None and self.push_height_sweep_on()
        open_push = t_pose is not None and open_classes and self.push_gripper_open_on()
        hull_w = self.world.hull_at(self.hull["target"], t_pose) if (sweep or open_push) else None
        line_xy = self.push_line_xy(t_pose) if (sweep or open_push) else None
        cands = self.touching_push_candidates(st, p_init, t_face_z, direction, t_hull_w=hull_w if sweep else None, line_xy=line_xy)
        if open_push:
            import itertools

            # oracle-fix 2026-11-06 (owner on touching#02's cereal box): the open gripper low on the face first, today's classes after it
            cands = itertools.chain(self.open_push_candidates(st, hull_w, t_face_z, direction, line_xy=line_xy), cands)
        pair = self.pair_goal_align()
        if pair is not None and pair["push_max_height_m"] is not None:
            h_max = float(pair["push_max_height_m"])
            cands = (c for c in cands if float(c[2].get("fingertip_z") or G.t(c[1][0])[2]) - float(t_face_z[0]) <= h_max + 1e-9)
        if height_range is not None:
            lo, hi = (None if v is None else float(v) for v in height_range)
            cands = (c for c in cands if c[2].get("contact_height_m") is None
                     or ((lo is None or float(c[2]["contact_height_m"]) >= lo - 1e-6) and (hi is None or float(c[2]["contact_height_m"]) <= hi + 1e-6)))
        # i7-symonly (PUSH_TIP_CAP_ENV): the contacts above the target's tip height skipped (on) / counted (shadow)
        tip = self.push_tip_cap(t_pose, float(t_face_z[0]), direction, hull_w=hull_w, line_xy=line_xy) if sweep else None
        if tip is not None and push_tip_cap_mode() == "on":
            cands = self._tip_capped(cands, float(tip["cap_m"]))

        def chain(c):
            _kind, init, extras = c
            return [IK.pre_pose(init, float(extras.get("pre_dist_m") or IK.PRE_DIST)), init, (init[0] + direction * dist, init[1])]

        def overlap_of(c):
            init, goal = chain(c)[1:]
            return self.push_hand_overlap(init, goal, t_pose, direction, dist,
                                          against="hull" if (sweep or push_fingers(c[2]) != IK.FINGER_CLOSED) else "box")

        rank = hand_overlap_rank_on()
        if rank:  # owner item 3 switch (default off): the overlapping candidates after the clean ones, not rejected
            clean, hinted = [], []
            for c in cands:
                ov = overlap_of(c)
                (clean if ov is None else hinted).append(c if ov is None else (c[0], c[1], dict(c[2], hand_overlap_hint=ov)))
            cands = clean + hinted

        def label(c):
            h = c[2].get("contact_height_m")
            return c[0] if h is None else f"{c[0]}@{100.0 * float(h):.1f}cm"

        def evaluate(p):
            pre, init, goal = p.chain
            fingers = push_fingers(p.payload[2])  # closed, or the open push's opening (oracle-fix 2026-11-06)
            if not rank:
                overlap = self.push_hand_overlap(init, goal, t_pose, direction, dist, against="hull" if (sweep or fingers != IK.FINGER_CLOSED) else "box")
                if overlap is not None:
                    return False, None, None, overlap
            r_push = self.ikc.exact(p.chain, st, arm, ignore_for_last=ignore_push, cache=False, fingers=fingers, q_first=p.q_first)
            if not r_push.ok:
                return False, r_push, None, f"{arm}: {r_push.reason}"
            qs = O.chain_qs(self.ikc, arm, p.chain)
            r_ret = self.ikc.exact([IK.pre_pose(goal)], st, arm, ignore_for_last=ignore_push, cache=False, fingers=fingers)
            if not r_ret.ok:
                return False, r_push, None, f"retreat clear pose: {r_ret.reason}"
            return True, r_push, qs, ""

        res = O.choose_first(self.ikc, arm, cands, key=lambda c: c[1], label=label,
                             chain=chain, evaluate=evaluate, q_start=self.ikc.q_locked(st), camera_idx=[1], stage="touching push")
        if tip is not None and res[0] is not None:
            h = res[0].proposal.payload[2].get("contact_height_m")
            if h is not None and float(h) > float(tip["cap_m"]) + 1e-6:
                TT._count(self, "push_tip_cap.chosen_above")  # shadow: the class the cap would have skipped
        return res

    def touch_push_strokes(self, kind: str, extras: Dict[str, Any], dist: float, root_xy: Optional[th.Tensor] = None,
                           direction: Optional[th.Tensor] = None) -> Dict[str, Any]:
        """:data:`TOUCH_PUSH_STROKE_ENV` (i7-t04push): the push record's ``strokes`` plan (``iors_large.push_stroke_plan`` over the push's
        ``dist``) and the fingertips' geometry for ``execute._push_strokes``; ``{}`` when off, for the shelf layout and for a class that is not a
        closed fingertip push (``touch_push_strokes.skipped_class``)."""
        stroke = touch_push_stroke_m()
        if stroke is None or self.shelf_start is not None:
            return {}
        if not str(kind).startswith("push_tip_") or push_fingers(extras) != IK.FINGER_CLOSED:
            TT._count(self, "touch_push_strokes.skipped_class")
            return {}
        from ispatialgym.sampling.v2.tasks.iors_large import push_contact_strips, push_stroke_plan

        plan = push_stroke_plan(float(dist), stroke, TOUCH_PUSH_STROKES_MAX, backoff_m=TOUCH_PUSH_STROKE_BACKOFF_M, reaim_max_m=TOUCH_PUSH_STROKE_REAIM_MAX_M,
                                yaw_gain_m_per_rad=TOUCH_PUSH_STROKE_YAW_GAIN_M_PER_RAD, yaw_deadband_rad=math.radians(2.0))
        strips, _face = push_contact_strips("closed_turned" if str(kind).startswith("push_tip_h") else "blade")
        clearance = float(extras.get("stop_clearance_m") or CPZ.STOP_CLEARANCE)
        TT._count(self, "touch_push_strokes.recorded")
        log.info("%s: touching push %s in %d strokes of <= %.3f m over %.3f m (TOUCH_PUSH_STROKE_M)", self.gp.variant, kind, plan["n"], stroke, float(dist))
        out = {"strokes": plan, "face_half_m": round(C.FINGERTIP_LENGTH, 5), "blade_standoff_m": round(C.FINGERTIP_LENGTH + clearance, 5),
               "contact_strips_m": [[round(c, 5), round(hw, 5)] for c, hw in strips], "contact_band_m": list(TOUCH_PUSH_STROKE_BAND_M)}
        cp = extras.get("contact_point")
        if root_xy is not None and direction is not None and cp is not None:
            # the stroke's fallback without the live hull in its band: the target's back = its root less this along the push
            d2 = G.t(direction).float()[:2]
            d2 = d2 / max(float(th.linalg.norm(d2)), 1e-9)
            out["half_extent_m"] = round(max(0.0, float((G.t(root_xy).float()[:2] - G.t(cp).float()[:2]) @ d2)), 5)
        return out

    def push_tip_cap(self, t_pose: Optional[G.Pose], z0: float, direction: th.Tensor, hull_w: Optional[th.Tensor] = None,
                     line_xy: Optional[th.Tensor] = None) -> Optional[Dict[str, float]]:
        """:data:`PUSH_TIP_CAP_ENV` (i7-symonly): the target's tip height for a push along ``direction`` at ``t_pose``
        (``contact_poses.push_tip_height_cap`` with ``generation.push_tip_height_ratio``); None when the switch is off, without a place pose
        or outside the push height sweep. Kit ``push_tip_cap.checked``; the latest is kept in ``self._push_tip_cap_last``."""
        if push_tip_cap_mode() == "off" or t_pose is None or not self.push_height_sweep_on():
            return None
        if hull_w is None:
            hull_w = self.world.hull_at(self.hull["target"], t_pose)
        ratio = float(TT.phase2_gen(self, "push_tip_height_ratio", CPZ.PUSH_TIP_HEIGHT_RATIO))
        out = CPZ.push_tip_height_cap(hull_w, float(z0), direction, line_xy=line_xy, ratio=ratio)
        TT._count(self, "push_tip_cap.checked")
        self._push_tip_cap_last = out
        return out

    def _tip_capped(self, cands, cap: float):
        """``cands`` without the contacts above ``cap`` (m above the target's bottom; a candidate without a height is kept); kit
        ``push_tip_cap.dropped`` per skipped candidate the choice reached."""
        for c in cands:
            h = c[2].get("contact_height_m")
            if h is not None and float(h) > cap + 1e-6:
                TT._count(self, "push_tip_cap.dropped")
                continue
            yield c

    def _touching_brace_choice(self, st: Stance, push_chain: List[G.Pose], t_pose: G.Pose, p_pose: G.Pose, b_point: th.Tensor, face_z: Tuple[float, float],
                               direction: th.Tensor, dist: float, p_hull: th.Tensor, p_init: th.Tensor, ignore_push: Sequence[Any], ignore_brace: Sequence[Any],
                               limit: Optional[int] = None, diagnose: bool = True, push_fingers_q: float = IK.FINGER_CLOSED):
        """The left brace joining the chosen push: ``touching_brace_candidates`` in their order (the first ``limit``), a class's two
        wrist turns chosen by ``orient.choose``. A candidate passes when the hand beyond the finger profile clears the braced
        partner, the two-arm solve holds with the right arm locked at its push GOAL, and the brace arm's retreat after its release
        (10 cm back along the approach, fingers open) has a configuration (0918-223037 attempt 0 died in the program check on
        left_arm_joint7's limit there). Owner 2026-09-18 22:29: with the box still at the place the right fingertips locked at the
        goal sit ~5 cm inside it and every whole-body configuration read 'in collision' (0918-211933 .. 220534) - the box is posed
        where the push leaves it and leaves the collision world for the two-arm solve (the tips 5 mm before its face sat inside
        the checker's margin, 0918-223037 attempt 2); the push chain itself was validated with its own ignores. ``push_fingers_q``
        (oracle-fix 2026-11-06, :meth:`push_gripper_open_on`): the finger joints of the two-arm solve - the push's (:func:`push_fingers`; the
        program check checks the push with the brace locked at that value too)."""
        import itertools

        pushed = (G.t(t_pose[0]) + direction * dist, G.t(t_pose[1]))

        def evaluate(p):
            pre, contact = p.chain
            # the brace hand at the opening the oracle braces at (tabletop.brace_fingers: a wide stop's fingers_opening_m, else closed) in the
            # two-arm solve, the diagnostic and the retreat - as the program check checks it (i7-contact 2026-11-07; i7-contactbrace 2026-11-08:
            # always, the one rule of every brace check - ISPATIALGYM_TOUCH_BRACE_AT_OPENING is gone)
            brace_q = TT.brace_fingers(p.payload[3])
            overlap = self.hand_overlap(contact, "partner", p_pose)  # camera / knuckles / body vs the braced partner
            if overlap is not None:
                return False, None, None, overlap
            with self.posed("target", pushed), self.ikc.ignoring([self.obj("target")]):
                res = self.ikc.exact_both_arms(("right", list(push_chain)), ("left", [pre, contact]), st, ignore_push, ignore_brace, fingers=push_fingers_q,
                                               q_first_other=p.q_first, fingers_other=brace_q)
            if not res.ok:
                if not diagnose:
                    return False, res, None, str(res.reason)
                # diagnostic (owner 2026-09-18 20:48): the left chain on its own, the right arm at the stance posture - a pass here names
                # the two-arm combination (self-collision / the locked right arm) as the blocker
                alone = self.ikc.exact([pre, contact], st, "left", ignore_for_last=ignore_brace, cache=False, fingers=brace_q)
                return False, res, None, f"{res.reason} [left alone: {'ok' if alone.ok else alone.reason}]"
            qs = O.chain_qs(self.ikc, "left", [pre, contact])
            r_ret = self.ikc.exact([IK.pre_pose(contact, RETREAT_DIST_M)], st, "left", ignore_for_last=ignore_brace, cache=False, fingers=brace_q)
            if not r_ret.ok:
                return False, res, None, f"brace retreat clear pose: {r_ret.reason}"
            return True, res, qs, ""

        def evaluate_counted(p):
            out = evaluate(p)
            TT.count_brace_opening(self, p.payload[3], ok=bool(out[0]))  # i7-contactbrace: the openings tried / passed (the A/B's counters)
            return out

        stream = self.touching_brace_candidates(st, b_point, face_z, -direction, partner_hull_w=p_hull, push_point=p_init)
        if limit is not None:
            stream = itertools.islice(stream, int(limit))
        return O.choose_first(self.ikc, "left", stream, key=lambda c: c[1], label=lambda c: c[2], chain=lambda c: [c[0], c[1]], evaluate=evaluate_counted,
                              q_start=self.ikc.q_locked(st), stage="touching brace")

    def push_line_xy(self, t_pose: G.Pose) -> Optional[th.Tensor]:
        """:data:`PUSH_LINE_COM_ENV`: the world xy of the target's mass centre at ``t_pose`` (the push line's point), None when off (the
        footprint centre line). The centre in the root frame is cached per target object; kit ``push_line_com.used``."""
        if not push_line_com_on() or t_pose is None:
            return None
        from ispatialgym.sampling.v2.tasks.alignment_axial import _com_local

        obj = self.obj("target")
        memo = self.__dict__.setdefault("_push_line_com", {})
        key = getattr(obj, "name", id(obj))
        if key not in memo:
            memo[key] = G.t(_com_local(obj)).float()
            log.info("%s: push line through the mass centre (ISPATIALGYM_PUSH_LINE_COM): %s in the root frame %s", self.gp.variant, key,
                     [round(float(v), 4) for v in memo[key]])
        TT._count(self, "push_line_com.used")
        return (G.t(t_pose[0]).float() + C.quat_rotate(G.t(t_pose[1]).float(), memo[key]))[:2]

    def touching_push_candidates(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                                 t_hull_w: Optional[th.Tensor] = None, line_xy: Optional[th.Tensor] = None):
        """The push record builder (``contact_poses.touching_push_candidates``, the rule the ``touch_push`` maps share): the closed right gripper's
        push poses at the target's right face (``face_point``, mid height) along ``direction``, in the order to try; ``t_hull_w`` (the target's
        hull at the place, given under :meth:`push_height_sweep_on`): :meth:`swept_push_candidates` instead. Yields ``(kind, eef pose, extras)``."""
        yield from CPZ.touching_push_candidates(self, manip, face_point, face_z, direction, t_hull_w=t_hull_w, line_xy=line_xy)

    def _leaning_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float,
                       z_tip: float, lean_deg: float):
        """``contact_poses.leaning_blade``: the closed blade leaning ``lean_deg`` along the push, its fingertips at ``z_tip``."""
        yield from CPZ.leaning_blade(self, manip, face_point, face_z, d, outward, clearance, z_tip, lean_deg)

    def _raised_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float, z_tip: float):
        """``contact_poses.raised_blade``: the vertical blade with its fingertips at ``z_tip`` = face top - PALM_FROM_TIP (+ clearance)."""
        yield from CPZ.raised_blade(self, manip, face_point, face_z, d, outward, clearance, z_tip)

    # -- push height sweep (oracle-fix 2026-11-06, owner on the touching#04 spray-bottle demo: "for small scale touching, can we make the push
    #    push the most reachable center bottom part of the target object so it doesn't wobble so much") --

    def push_height_sweep_on(self) -> bool:
        """``generation.push_height_sweep`` (oracle-fix 2026-11-06; on for touching small scale through ``generation_defaults``, ``false`` =
        the classes at today's heights): the push contact as low on the target's centre line as the hand allows
        (:meth:`swept_push_candidates`), the hand beyond the fingers checked against the hull (:meth:`hand_overlap`), the push stance at the
        lateral whose feasible contact is lowest (:meth:`push_stance`). The tabletop layout only: the shelf layout's book leans against a
        side board, and a push at its foot slides the foot from under the lean."""
        return self.relation == "touching" and self.shelf_start is None and bool(TT.phase2_gen(self, "push_height_sweep", False))

    def swept_push_candidates(self, manip: Stance, t_hull_w: th.Tensor, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor):
        """:meth:`push_height_sweep_on`: ``contact_poses.swept_push_candidates`` - today's push classes at every contact height from the class's
        lowest (``PUSH_SWEEP_MIN_HEIGHT_M``) up to ``push_max_height_fraction`` x the target's height on the line through its footprint centre,
        lowest first; the raised blade last for a target that is not tall. Yields ``(kind, eef pose, extras)`` (``contact_height_m`` ...)."""
        yield from CPZ.swept_push_candidates(self, manip, t_hull_w, face_point, face_z, direction)

    # -- the open-gripper push (oracle-fix 2026-11-06, owner on touching#02's cereal box: "cereal box - does it right now push with gripper
    #    open? if it doesn't can you try with gripper open at lower height and see if it improves yield?") --

    def push_gripper_open_on(self) -> bool:
        """``generation.push_gripper_open`` (oracle-fix 2026-11-06, default off): the touching push tries the OPEN right gripper low on the
        target's face first (:meth:`open_push_candidates`), today's (closed) classes after it; the chain, its retreat clear pose and the
        two-arm brace solve are checked with the open fingers (:func:`push_fingers`), the record's ``fingers_opening_m`` is the opening the
        oracle holds through pre-contact -> push -> retreat (``execute_push``) and the program check replays with; a brace that joins only
        today's class keeps that class (:meth:`push_chain_feasible`, :meth:`touching_contacts`). The tabletop layout only (the shelf
        layout's book leans on a side board and is pushed by the hand that carried it)."""
        return self.relation == "touching" and self.shelf_start is None and bool(TT.phase2_gen(self, "push_gripper_open", False))

    def open_push_candidates(self, manip: Stance, t_hull_w: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                             line_xy: Optional[th.Tensor] = None):
        """``generation.push_gripper_open``: ``contact_poses.open_push_candidates`` - the right gripper open ``push_open_opening_m`` per finger,
        both fingertips ``stop_clearance_m`` before the face on the footprint centre line (``push_open_v-``, ``push_open_h-`` / ``push_open_h``),
        by the lower fingertip's height. Yields ``(kind, eef pose, extras)``; the extras carry ``fingers_opening_m``."""
        yield from CPZ.open_push_candidates(self, manip, t_hull_w, face_z, direction, line_xy=line_xy)

    def wide_stop_spread(self, partner_hull_w: th.Tensor, b_point: th.Tensor, outward: th.Tensor, side_h: th.Tensor, s_push: float,
                         spreads: Sequence[float] = (0.09, 0.07, 0.05), end_margin: float = 0.01, face_depth_tol: float = 0.01, band: float = 0.03) -> Optional[float]:
        """``contact_poses.wide_stop_spread`` (owner 2026-09-19 00:49): the largest spread whose two fingertips land on the far face's flat run
        ``end_margin`` from its ends around ``s_push``; None = a narrow (closed) stop."""
        return CPZ.wide_stop_spread(partner_hull_w, b_point, outward, side_h, s_push, spreads=spreads, end_margin=end_margin, face_depth_tol=face_depth_tol, band=band)

    def brace_tip_heights(self, b_point: th.Tensor, face_z: Tuple[float, float]) -> List[float]:
        """``contact_poses.brace_tip_heights``: the fingertip-front brace heights on the partner's far face, the higher first - ``max(braced
        point, z0 + brace_tip_min_above_m)`` and ``min(z1 - 0.01, z0 + 0.075)``, those within 1 cm of the top dropped."""
        return CPZ.brace_tip_heights(self, b_point, face_z)

    def brace_open_rule(self, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: Optional[th.Tensor],
                        push_point: Optional[th.Tensor] = None) -> Optional[Dict[str, Any]]:
        """``generation.brace_open_when_wide`` (:meth:`brace_open_when_wide`): ``contact_poses.brace_open_rule`` - whether the partner's braced face
        takes the open fingertips (its flat run at the higher brace height >= the smallest open spread + 2 end margins) and the spread; None
        without the key, a partner hull or a brace height."""
        return CPZ.brace_open_rule(self, b_point, face_z, a_face, partner_hull_w, push_point, open_when_wide=self.brace_open_when_wide())

    def _open_brace_candidates(self, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: th.Tensor,
                               rule: Dict[str, Any], clearance: float):
        """``contact_poses.open_brace_candidates``: the open-finger stops of :meth:`brace_open_rule` (``brace_wide_h-`` / ``brace_wide_h`` [``_lo``])."""
        yield from CPZ.open_brace_candidates(self, b_point, face_z, a_face, partner_hull_w, rule, clearance)

    def touching_brace_candidates(self, manip: Stance, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor,
                                  partner_hull_w: Optional[th.Tensor] = None, push_point: Optional[th.Tensor] = None):
        """The brace record builder (``contact_poses.touching_brace_candidates``, the rule the ``touch_brace`` maps share): the left gripper's
        brace poses at the partner's far (robot-left) face in the order to try - the open-finger stop on a wide face under
        ``brace_open_when_wide``, else the fingertip-front classes (wide stop, ``brace_tip_h[-][_p15][_lo]``), the top-down blades, the side
        contacts. Yields ``(pre, contact, kind, extras)``."""
        yield from CPZ.touching_brace_candidates(self, manip, b_point, face_z, a_face, partner_hull_w=partner_hull_w, push_point=push_point,
                                                 open_when_wide=self.brace_open_when_wide())

    def _nudge_until_touching(self, tname: str, pname: str, step: float = 0.001, max_steps: int = 25):
        from omnigibson.object_states import Touching

        t, p = self.objects[tname], self.objects[pname]
        direction = self.sidecar.stance("manipulation").tangent
        if self.shelf_start is not None:  # toward the partner on the other side board
            direction = self.shelf_push_direction(self.sidecar.stance("manipulation"))[:2]
        for k in range(max_steps + 1):
            self.world.wake([t, p])
            if t.states[Touching].get_value(p):
                self._nudge_steps = k
                return
            if k == max_steps:
                break
            pos, quat = t.get_position_orientation()
            t.set_position_orientation(pos + th.tensor([float(direction[0]) * step, float(direction[1]) * step, 0.0]), quat)
            t.keep_still()
            self.world.step_physics(3, still=[self.robot, p])
        raise Reject("goal_infeasible", "could not establish contact for touching")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        tname = self.bindings.one("target").name
        g = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        recs: List[Dict[str, Any]] = []
        mc = self.choice.models["target"]
        if g is not None and manip.focus is not None:
            recs.append({"scope": {"target": f"{mc.category}/{mc.model}", "support_height": round(float(self.top.z - self.world.floor_z), 2)},
                         "type": "grasp_set", "trunk": [manip.translate, round(manip.tilt, 2)],
                         "cell": [round(v, 2) for v in manip.world_to_base(manip.focus[:2]).tolist()],
                         "feasible": [[round(g.yaw, 2), g.grasp_name]], "hard": False})
        pname = self.bindings.one("partner").name
        depth_t, lat_t = manip.local(draft.object_poses[tname][0])
        depth_p, lat_p = manip.local(draft.object_poses[pname][0])
        recs.append({"scope": {"scene": self.scene, "support": self.support.name, "target": f"{mc.category}/{mc.model}",
                               "partner": f"{self.choice.models['partner'].category}/{self.choice.models['partner'].model}"},
                     "type": "placement", "literal": f"({self.relation} target partner)", "depth": [round(depth_t, 2), round(depth_t, 2)],
                     "lateral": [round(lat_t, 2), round(lat_t, 2)], "partner_lateral": [round(lat_p, 2), round(lat_p, 2)], "hard": False})
        if getattr(self, "bands", None) is not None and getattr(self.bands, "path", None):
            recs.append({"scope": {"scene": self.scene, "support": self.support.name, "target": f"{mc.category}/{mc.model}"}, "type": "band_ref",
                         "path": str(self.bands.path), "hard": False})
        return recs

    # ------------------------------------------------------------------ helpers

    def _container_volume(self, obj, p_pose: G.Pose) -> Tuple[th.Tensor, float]:
        """(container centre — the fillable meta link's xy when the model has one, else the hull centre —, rim z = the
        partner hull's highest point at ``p_pose``)."""
        from omnigibson.macros import macros

        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        centre = (p_hull.max(dim=0).values + p_hull.min(dim=0).values) / 2
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                # the meta link's offset in the root frame from the current sim poses, applied at ``p_pose`` (the partner may
                # still be parked when the goal is first computed)
                root = obj.get_position_orientation()
                p, _ = link.get_position_orientation()
                local = G.invert((G.t(root[0]), G.t(root[1])))
                p_local = C.transform_points(local, G.t(p).unsqueeze(0))[0]
                p_world = C.transform_points((G.t(p_pose[0]), G.t(p_pose[1])), p_local.unsqueeze(0))[0]
                centre = th.tensor([float(p_world[0]), float(p_world[1]), float(centre[2])])
                break
        return centre, float(p_hull[:, 2].max())
