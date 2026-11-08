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
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put, cache_root  # train-fast 2026-10-31: the SPA lateral-cell memo on disk
from ispatialgym.sampling.sim_utils import PARK_POSITION, Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import contact_poses as CPZ  # i7-contact 2026-11-07: the touching push / brace pose rules (shared with the contact maps)
from ispatialgym.sampling.v2 import ik as IK
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


def shelf_joint_stance_on() -> bool:
    return os.environ.get(SHELF_JOINT_STANCE_ENV, "").strip().lower() in ("1", "on", "true")


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


def shelf_lift_straighten_on() -> bool:
    return os.environ.get(SHELF_LIFT_STRAIGHTEN_ENV, "").strip().lower() in ("1", "on", "true")


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


def shelf_retreat_backoff_on() -> bool:
    return os.environ.get(SHELF_RETREAT_BACKOFF_ENV, "").strip().lower() in ("1", "on", "true")


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


def shelf_release_in_gap_on() -> bool:
    return os.environ.get(SHELF_RELEASE_IN_GAP_ENV, "").strip().lower() in ("1", "on", "true")


def shelf_free_start_on() -> bool:
    """``ISPATIALGYM_SHELF_FREE_START`` = "1" (i7-t03 round 2 item (a); default off)."""
    return os.environ.get(SHELF_FREE_START_ENV, "").strip().lower() in ("1", "on", "true")


def shelf_stance_shift_on() -> bool:
    """``ISPATIALGYM_SHELF_STANCE_SHIFT`` = "1" (i7-t03 item 1; default off)."""
    return os.environ.get(SHELF_STANCE_SHIFT_ENV, "").strip().lower() in ("1", "on", "true")


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
    return None


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
            self.ikc.set_finger_touch(IK.FingerTouch(p_touch["margin_m"], p_touch["max_penetration_m"], p_touch["preshape"], supports=self.support_objects,
                                                     opening=lambda: op))
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
                                                  arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self),
                                                  **phase2_band)
            phase2_any = phase2_any or bool(cands)
            self.manip_candidates = cands if self.shelf_start is not None else self.start_region(cands)
            if self.manip_candidates:
                break
            failed.append(f"lateral {self.init_stance.lateral:+.2f}: " + ("Phase 2 left no survivor" if not cands else "no start cell for the partner on the "
                                                                          + " / ".join(TT.side_name(sd) for sd in self.template_sides())))
        if "start_region" in self.template_facts:
            self.template_facts["start_region"]["failed_lateral_draws"] = failed
        if not self.manip_candidates and start_per_attempt():
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
        nb_top = self.neighbor_start_top()  # start_region_on_neighbors: the target is drawn on the neighbours' cells
        if self.pair_goal_align() is not None:
            yaws = self.pair_target_yaws(manip)
        else:
            yaws = self.widest_faces_target_yaws(manip) if self.widest_faces() else self.support_aligned_yaws("target", manip) if self.init_align_support() else None
        yaw_kw = {"yaws": yaws, "jitter": 0.0} if yaws is not None else {}
        if self.pair_goal_align() is not None:
            yaw_kw["margin"] = self.margin_support  # the long axis across the edge may need the support's whole depth (the partner keeps it too)
        last = None
        for i, gs in enumerate(tiers):
            try:
                t_pose, grasp = TT.phase3_target(self, manip, rng, gs, self.hull["target"], self.rest["target"], self.support_objects(),
                                                 top=self.top if nb_top is None else nb_top, scale_key=self.choice.models["target"].scale, arms=arms,
                                                 arm_bands=getattr(self, "arm_bands", None) or None, **yaw_kw)
                if i > 0:
                    log.info("%s: no flush pinch feasible at the focus; inclined grasp %s drawn", self.gp.variant, grasp.grasp_name)
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
        if start_per_attempt():  # this attempt's start first: the partner sides come from its start region
            import time

            t_spa = time.time()
            self.spa_redraw_start(rng)
            self._spa_draw_s = time.time() - t_spa
            self._spa_drawn = True
        if shelf_draw:
            self.draw_shelf_layout(rng)  # the compartment and the leans: the two books' poses follow the attempt's edge
        # A1 step 3: the partner's side is fixed before the target is placed - before the split cone draws the start position too
        self.region_side = None if self.shelf_start is not None else self.draw_partner_side(rng)
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
        if start_per_attempt() and fast_on("SPA_CACHE"):
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
                                              arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self),
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
        for edge, init in TT.edge_frames(self, self.top, rng):
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
            lat = float(rng.uniform(lo, hi))  # the plain draw's first Phase-1c draw: the same call on the same stream
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
                                                  arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self),
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

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        self._stance_shift_budget = {"scan": 0, "seen": set()}  # i7-t03 item 1: the scan's per-attempt budget
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if shelf_free_start_on() and self.shelf_pose is not None:  # i7-t03 round 2 (a): the free-floor start as the edge stance's fallback
            self.init_stance = self.shelf_free_start(rng, self.init_stance)
        init = self.init_stance
        self.sidecar.add_stance(init)
        cam_init = self.camera_pose_at(init)
        # A1: the target starts in the start region of the attempt's partner side (7e: never on a candidate dropped for its travel)
        cands = [c for c in self.manip_candidates if TT.in_start_region(c, self.region_side) and TRAVEL_DROPPED not in c.extras]
        order = rng.permutation(len(cands))
        if os.environ.get("ISPATIALGYM_STANCE_FRAME_AIM", "") == "1" and self.shelf_pose is None and self.relation != "touching" and cands:
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
        for k in order[:6]:
            cand = cands[int(k)]
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
                    t_pose, grasp = self.phase3_two_tier(manip, rng, arms=arms_here)
                if grasp.arm != default:
                    self.sidecar.transitions.clear()
                    self.record_stance_travel(init, manip)  # the other arm idles now
                self.apply_height_extra_to_grasp(grasp, t_pose, manip)  # A/B knob generation.contact_height_extra_m (0 = no-op)
                # the segment's later contacts; when they fail with the chosen arm, the whole segment is retried with the other arm if
                # its grasp passed too (owner 2026-09-27: the fallback spans the segment)
                tried: List[str] = []
                while True:
                    try:
                        p_pose = self._segment_draw(rng, d, int(k), t_pose, grasp, manip, cam_init, pname)
                        break
                    except Reject as e:
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
                return d
            except Reject as e:
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
        # §4.6 step 2: partner on the attempt's side (A1) at a gap of the start region's interval
        frame_score = None
        if os.environ.get("ISPATIALGYM_FRAME_RANK", "") == "1" and self.relation != "touching":
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
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                res_place = self.ikc.exact([self.pre_place_of(place_guess), place_guess], st_place, grasp.arm, ignore_for_last=ign_place,
                                           scale_key=self.choice.models["target"].scale, cache=False, approached=place_approached(self))
            if not res_place.ok:
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
        return p_pose

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
                "mirror": bool(start_per_attempt()) and any(sd != self._shelf_template["lean_side"] for _, sd in spec["layouts"])}

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
        elif not start_per_attempt():
            fixed, why = float(base["lean_side"]), "no start_per_attempt: the template's arm and candidates"
        lean_side = TT.lean_rest_side(getattr(self, "grasp_key", None))
        if lean_side is not None:  # i7-lean 2026-11-07: a v2 lean rest leans on one board (its grasps and its map are that rest's)
            fixed, why = lean_side, f"v2 lean rest {getattr(self, 'grasp_key', None)}: its board"
        if fixed is not None:
            bays = self.gen("shelf_bay_lateral_m")
            bay0 = float(bays.get(self.support.name, 0.0)) if isinstance(bays, dict) else 0.0
            layouts = [lay for lay in layouts if lay[1] == fixed] or [(bay0, fixed)]
        k = int(rng.integers(len(layouts)))
        bay, side = layouts[k]
        self.shelf_start = {"key": base["key"], "lean_side": side, "lean_deg": base["lean_deg"], "context": base.get("context")}
        self.rest["target"], self.rest["partner"] = self.shelf_rests(side, float(base["lean_deg"]), float(base["lean_deg"]))
        self.shelf_use_arm(self.shelf_push_arm())
        self.shelf_pose = self.shelf_partner_pose = None
        self._shelf_draw = {"layout": [bay, TT.side_name(side)], "n_layouts": len(layouts), "restricted": why}
        self._shelf_side_layouts = [lay for lay in layouts if lay[1] == side]

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
        if shelf_lift_straighten_on():
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
        if not prim or not shelf_lift_straighten_on():
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
        try:
            st = free_floor_stance(self, t_xy, focus, lambda st_, cam: self.books_in_frame(st_, cam), rng, what="the partner")
        except Reject as e:
            TT._count(self, "free_start.failed.no_stance")
            log.info("%s: free start (shelf): %s - the edge init stance stays", self.gp.variant, e)
            return init
        TT._count(self, "free_start.used")
        st.name = "init"
        st.extras["free_start_from_edge_lateral_m"] = round(float(init.lateral), 4)
        return st

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
        want = float(self.gen("shelf_stance_book_lateral_m", SHELF_STANCE_BOOK_LATERAL_M) or SHELF_STANCE_BOOK_LATERAL_M)
        want = want if arm == "left" else -want
        _, lat_book = cand.local(G.t(self.shelf_pose[0])[:2])
        dlat = float(lat_book) - want  # moving the base +d along the tangent moves the book -d in the robot's frame
        so0 = float(cand.stand_off if cand.stand_off is not None else TT.STAND_OFF)
        sos = [so0] + [float(v) for v in TT.STAND_OFFS if abs(float(v) - so0) > 1e-6]
        shifts = [float(v) for v in (self.gen("shelf_stance_scan_shifts_m") or SHELF_STANCE_SCAN_SHIFTS_M)]
        own = (round(float(cand.translate), 4), round(float(cand.tilt), 4))
        trunks = [own]
        if shelf_joint_stance_on():  # 5p.2: the trunk set of every candidate
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

    def shelf_fixed_start(self, init: Stance, cand: Stance, manip: Stance, arm: str, try_fn) -> Tuple[Stance, GraspRecord]:
        """The fixed start's checks at ``manip`` (``try_fn(stance) -> GraspRecord``: the init -> manipulation travel, ``grasp_at_pose`` with
        every grasp and twin, the transit check, the back-off - :meth:`sample_instance`'s own; ``Reject`` on failure). With the stance shift on
        (:data:`SHELF_STANCE_SHIFT_ENV`, i7-t03 item 1) a failure the base position can change (:func:`shelf_need_of`) tries the stances of
        :meth:`shelf_stance_shift` (the same ``try_fn``); a candidate that passes is never shifted. Returns ``(stance, grasp)``."""
        joint = shelf_joint_stance_on() and self.shelf_pose is not None
        try:
            g0 = try_fn(manip)
        except Reject as e:
            need = shelf_need_of(e)
            if not shelf_stance_shift_on() or self.shelf_pose is None or need is None:
                raise
            return self.shelf_stance_shift(init, cand, manip, arm, e, need, try_fn)
        if not joint:
            return manip, g0
        if self._joint_ok(manip, g0):  # i7-t03 round 2 (b): the candidate grasps AND releases - no carried drive
            TT._count(self, "stance_shift.both_passed")
            return manip, g0
        if not shelf_stance_shift_on():
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
            joint = shelf_joint_stance_on()  # 5p.2: the budget counts the stances that reach the exact checks (the trunk set adds cheap rejects)
            if label == "scan" and budget["scan"] >= scan_max:
                break
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
            try:
                grasp = try_fn(st_ok)
            except Reject as e2:
                TT._count(self, "stance_shift.rejected.check")
                self.drop_travel_records(init.name, st_ok.name)
                tried.append(f"{label} {shift:+.2f}/{float(st_ok.stand_off or 0.0):.2f}: [{e2.reason}] {str(e2)[:90]}")
                continue
            if shelf_joint_stance_on():  # i7-t03 round 2 (b): a stance that grasps AND releases first; a grasp-only one is the fallback
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
        if TT.wrist_camera_clearance(grasp.grasp, float(self.top.z)) < TT.TWIN_CAMERA_CLEARANCE_M or not O.camera_clear(self.ikc, res.q, grasp.arm):
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
                why_b = self._retreat_backoff_why(st_place, grasp.place, grasp.arm, False) if (shelf_retreat_backoff_on() and self.shelf_tip()) else r.reason
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
                if self.shelf_tip():
                    # oracle-fix 2026-11-04 round 4: the lift moves the book shelf_lift_away_m toward the partner too - straight up it stayed against the
                    # side board it leans on (t03tip2b: the carried drive's first swept pose, attached book + left gripper on bookcase_opryhp_0)
                    away = self.shelf_push_direction(manip) * float(self.gen("shelf_lift_away_m", 0.03))
                    grasp.extras["lift_away"] = [round(float(v), 4) for v in away]
                    if shelf_lift_straighten_on():  # i7-t03 round 2 (d): the straightening lift + the leaned-board touch at the lift
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
            why_clear, code_clear = self.release_clear_why(st_place, place, grasp.arm, record=True)
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
        self.require_in_frame(cam, {"pre-grasp": grasp.pre_grasp, "grasp": grasp.grasp})
        self.require_in_frame(cam_place, {"pre-place": pre_place, "place": place})
        if self.shelf_tip():
            grasp.extras["place_kind"] = SHELF_TIP_PLACE_KIND  # the program: pick, carry, place (the book tips onto the partner), retreat
            return
        if self.relation == "touching":
            # the push (+ brace) from the stance the box was placed from (the second base pose when recorded)
            cam = cam_place
            self.touching_contacts(draft, st_place)
            c = self.sidecar.contacts
            poses = {"push pre-contact": c["push"].pre, "push contact": c["push"].contact, "push goal": c["push"].goal}
            if "brace" in c:
                # a brace whose poses leave the frame is dropped (push-only) rather than rejecting the instance
                brace_poses = {"brace pre-contact": c["brace"].pre, "brace contact": c["brace"].contact}
                if all(self.ikc.eef_in_frame(p, cam) for p in brace_poses.values() if p is not None):
                    poses.update(brace_poses)
                else:
                    self.sidecar.contacts.pop("brace", None)
                    manip.arms, manip.contacts = ["right"], ["push"]
                    draft.summary["brace"] = {"present": False, "why": "brace poses outside the manipulation image (5 %)", "gate": (self._brace_gate or {})}
            self.require_in_frame(cam, poses)

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
        why, _code = self.release_clear_why(st, place, grasp.arm)
        if why is not None:
            return f"clear pose: {why}"
        return None

    def release_clear_why(self, st: Stance, place: G.Pose, arm: str, record: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """The release clear pose from ``st``: the hand ``RETREAT_DIST_M + RETREAT_LITTLE_M`` (15 cm) back along the approach from ``place``,
        fingers open, the released target and the supports ignored - ``(None, None)`` when its exact IK passes, else ``(why, code)``. 5p.1
        (:data:`SHELF_RETREAT_BACKOFF_ENV`, the shelf tip release only): when it fails, :meth:`_retreat_backoff_why` (the place's end + the
        base backing off) may pass instead; ``record`` (the goal check's final release) records that retreat stance and transition."""
        r = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)], st, arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if record:
            self._drop_retreat_record()
        if r.ok:
            return None, None
        if not (shelf_retreat_backoff_on() and self.shelf_tip()):
            return r.reason, r.code
        why = self._retreat_backoff_why(st, place, arm, record)
        if why is None:
            return None, None
        return f"{r.reason}; base back-off: {why}", r.code

    def _drop_retreat_record(self) -> None:
        self.sidecar.stances.pop(RETREAT_STANCE, None)
        self.sidecar.transitions[:] = [t for t in self.sidecar.transitions if t.to != RETREAT_STANCE]

    def _retreat_backoff_why(self, st: Stance, place: G.Pose, arm: str, record: bool) -> Optional[str]:
        """5p.1: the shelf tip release's retreat with the base backing off - (1) the place's end, ``RETREAT_DIST_M`` back along the approach
        (where the oracle's place leaves the open hand), exact IK from ``st``; (2) the drive-back ``st`` -> ``st`` at stand-off +
        ``RETREAT_LITTLE_M`` (strides of ``navigation.STRIDE_M``) whole-body collision-free at ``st``'s trunk with both arms frozen at that
        configuration and the fingers open, the released target ignored (the program check sweeps it again with the book at its release
        pose). The hand ends where the 15 cm clear pose is. ``record``: stance :data:`RETREAT_STANCE` + the transition into it (``arms_q``,
        ``retreat_backoff_m``). None when it passes, else why."""
        from ispatialgym.sampling.v2 import navigation as NAV
        from ispatialgym.sampling.v2.stances import Transition

        r = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M)], st, arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if not r.ok or r.q is None:
            TT._count(self, "shelf_retreat.backoff_failed.place_end")
            return f"the place's end ({RETREAT_DIST_M * 100:.0f} cm back): {r.reason}"
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
        for li, lat in enumerate(laterals):
            for so in self.push_stance_stand_offs(manip):
                cand = self.push_stance_for(manip, t_place_pose, lateral=lat, stand_off=so)
                probe = cand if cand is not None else manip
                n_all += 1
                w = TT.contact_map_check(pm, probe, t_place_pose)
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
        if self.shelf_tip() and shelf_release_in_gap_on():
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
            if shelf_release_in_gap_on():
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
                if shelf_release_in_gap_on():
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

    def require_in_frame(self, cam: G.Pose, poses: Dict[str, Optional[G.Pose]]):
        """Every given eef pose must project inside the manipulation image with the 5 % margin (``IKChecker.eef_in_frame``
        with the shared camera FK of ``camera_pose_at``); ``Reject("gripper_not_in_frame")`` names the first that does not. Opt-in
        ``generation.grasp_frame_rule: "fingertip"`` (oracle-fix 2026-11-04 round 4, IORSLarge's rule): QA's own criterion instead - the fingertip
        point (``conventions.fingertip``) inside the image with the 5 % margin (t03tip3 opryhp: the shelf pre-grasp's +-6 cm eef box left the image)."""
        tip_rule = str(self.gen("grasp_frame_rule") or "box") == "fingertip"
        for what, pose in poses.items():
            if pose is None:
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
        cands = self.touching_push_candidates(st, p_init, t_face_z, direction, t_hull_w=hull_w if sweep else None)
        if open_push:
            import itertools

            # oracle-fix 2026-11-06 (owner on touching#02's cereal box): the open gripper low on the face first, today's classes after it
            cands = itertools.chain(self.open_push_candidates(st, hull_w, t_face_z, direction), cands)
        pair = self.pair_goal_align()
        if pair is not None and pair["push_max_height_m"] is not None:
            h_max = float(pair["push_max_height_m"])
            cands = (c for c in cands if float(c[2].get("fingertip_z") or G.t(c[1][0])[2]) - float(t_face_z[0]) <= h_max + 1e-9)
        if height_range is not None:
            lo, hi = (None if v is None else float(v) for v in height_range)
            cands = (c for c in cands if c[2].get("contact_height_m") is None
                     or ((lo is None or float(c[2]["contact_height_m"]) >= lo - 1e-6) and (hi is None or float(c[2]["contact_height_m"]) <= hi + 1e-6)))

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

        return O.choose_first(self.ikc, arm, cands, key=lambda c: c[1], label=label,
                              chain=chain, evaluate=evaluate, q_start=self.ikc.q_locked(st), camera_idx=[1], stage="touching push")

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

    def touching_push_candidates(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                                 t_hull_w: Optional[th.Tensor] = None):
        """The push record builder (``contact_poses.touching_push_candidates``, the rule the ``touch_push`` maps share): the closed right gripper's
        push poses at the target's right face (``face_point``, mid height) along ``direction``, in the order to try; ``t_hull_w`` (the target's
        hull at the place, given under :meth:`push_height_sweep_on`): :meth:`swept_push_candidates` instead. Yields ``(kind, eef pose, extras)``."""
        yield from CPZ.touching_push_candidates(self, manip, face_point, face_z, direction, t_hull_w=t_hull_w)

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

    def open_push_candidates(self, manip: Stance, t_hull_w: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor):
        """``generation.push_gripper_open``: ``contact_poses.open_push_candidates`` - the right gripper open ``push_open_opening_m`` per finger,
        both fingertips ``stop_clearance_m`` before the face on the footprint centre line (``push_open_v-``, ``push_open_h-`` / ``push_open_h``),
        by the lower fingertip's height. Yields ``(kind, eef pose, extras)``; the extras carry ``fingers_opening_m``."""
        yield from CPZ.open_push_candidates(self, manip, t_hull_w, face_z, direction)

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
