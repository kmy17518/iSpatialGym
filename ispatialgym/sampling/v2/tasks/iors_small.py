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
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2 import orient as O
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

#: Tip-to-tip spread of the CLOSED fingers (the finger joints sit at y = ±0.0135 in the gripper link; URDF ``gripper_finger_joint``):
#: a wide stop with tip spread ``s`` pre-shapes each finger to ``(s - CLOSED_TIP_SPREAD_M) / 2``.
CLOSED_TIP_SPREAD_M = 0.027
#: ``generation.brace_open_when_wide``: flat face kept between each open fingertip and the braced face's end, and the depth band (m)
#: within which the face's cut points count as its flat run (``wide_stop_spread``'s ``end_margin`` / ``face_depth_tol``)
BRACE_OPEN_END_MARGIN_M = 0.01
BRACE_OPEN_FACE_DEPTH_TOL_M = 0.01
#: Wrist turns of a blade push in the order to try (owner 2026-09-18 19:39): the wrist camera (``right_realsense_link``, three
#: r 1.8 cm spheres 4-7 cm to the side of the gripper at the palm-line height) lies on the eef's -y side; ``eef_quat(a, c·outward)``
#: puts eef +y = c·outward, so c = +1 hangs the camera over the pushed object's top and c = -1 turns it away from it. The camera
#: contacting the box top pushed the robot down against the table (demos 0918-190236 / 191811: robot and table shaking).
#: ``left_realsense_link`` sits at the same eef-frame position and every class derives its axes from the push direction, so the
#: left hand's push along -tangent (the touching shelf layout) keeps this order: camera away from the pushed object / up first.
PUSH_WRIST_TURNS = (-1.0, 1.0)
#: Hand geometry beyond the finger profile, sphere centres in the EEF frame (gripper-link spheres of
#: ``r1pro_description_curobo_default.yaml`` mapped by ``EEF_IN_GRIPPER`` (0, 0, -0.06; rpy 0, pi, 0): p_eef = (-x, y, -(z + 0.06)))
#: — what the IK collision check does not see against the pushed object (ignored at the contact pose, §3.6).
HAND_SPHERES_EEF: Dict[str, Tuple[Tuple[Tuple[float, float, float], ...], float]] = {
    "wrist camera": (((-0.056, -0.002, -0.076), (-0.056, -0.030, -0.076), (-0.056, -0.057, -0.076)), 0.018),
    "knuckle": (((-0.015, -0.042, -0.0675), (0.015, -0.042, -0.0675), (-0.015, 0.042, -0.0675), (0.015, 0.042, -0.0675)), 0.015),
    "gripper body": (((0.0, 0.0, -0.097),), 0.04),
}

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
STOP_CLEARANCE = 0.005
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
#: oracle-fix 2026-11-06 (owner on the touching#04 spray-bottle demo, ``generation.push_height_sweep``): the lowest contact height (m above
#: the target's bottom = the support top) of each push class at which the hand clears the support - the vertical blade's fingertips 5 mm
#: (the old blade's height), a leaning blade's fingertips ``blade_lean_tip_above_m`` (its leading edge dips below the tips), the horizontal
#: hand of the fingertip / pad pushes closing horizontally 5 cm (the gripper body, r 4 cm on the eef axis, 1 cm clear), the vertically
#: closing ones 6.2 cm (the lower knuckle 4.2 cm below the pads, r 1.5 cm, 5 mm clear); the pad pushes ``push_h`` / ``push_v`` (one turn
#: each: the wrist camera hangs 7.4 / 7.5 cm below the contact with its r 1.8 cm) 8 cm - below it the pre-pose's camera stands in the table
PUSH_SWEEP_MIN_HEIGHT_M = {"blade": 0.005, "tip_h": 0.05, "tip_v": 0.062, "pad": 0.08}
#: the push contact's window around the centre line (m, across it and in height): the outermost material within it is the face the
#: fingertips stop ``stop_clearance_m`` before (no part of them starts inside the target)
PUSH_SWEEP_WINDOW_M = 0.01
#: the footprint the push line goes through: the target's hull cut this far above its bottom (the part it stands on)
PUSH_SWEEP_BASE_CUT_M = 0.005
#: the raised blade (palm above the top) is the sweep's last resort for a target this tall over its smallest footprint side or less
#: (``generation.push_raised_max_aspect``; default: every target - the sweep tries every lower contact first): the spray bottle aliases set
#: 1.5 (it rocked 3.6 -> 20.3 deg pushed near its top); with 1.5 for every alias touching_02's cereal box lost its only feasible class
#: (fix1 2026-11-06: 1 instance in 146 attempts vs trainfast's 33 per instance, 9 goal-check contact_unreachable)
PUSH_RAISED_MAX_ASPECT = math.inf
#: oracle-fix 2026-11-06 (owner on touching#02's cereal box, ``generation.push_gripper_open``): the open push's finger joint opening (m per
#: finger, ``generation.push_open_opening_m``): the fingertip centres 7.7 cm apart - closing vertically at the knuckle's lowest eef height
#: (6.2 cm) the lower tip pushes 2.35 cm above the support, the upper one 10.05 cm (the x0.8 cereal box's middle: 10.9 cm), the lower
#: finger's base sphere 8.5 mm clear of the support
PUSH_OPEN_OPENING_M = 0.035
#: the fingertip centre's offset from the closing plane at the finger joint opening 0 (the tip sphere of ``tabletop.FINGER_SPHERES_EEF``:
#: y 3.5 mm, r 5 mm); open by ``q`` a tip's centre lies ``q`` + this from the eef axis
OPEN_TIP_CENTRE_M = 0.0035
#: the open fingers' lowest point above the support (m): the eef axis of an open push is raised until the lower finger clears it by this
PUSH_OPEN_SUPPORT_CLEARANCE_M = 0.005


def push_fingers(extras: Optional[Dict[str, Any]]) -> float:
    """oracle-fix 2026-11-06 (``generation.push_gripper_open``): the finger joints a push candidate / record is checked with - its
    ``fingers_opening_m`` (the open push: the oracle holds the gripper there, ``execute_push``; the program check replays with it), else
    closed."""
    v = (extras or {}).get("fingers_opening_m")
    return IK.FINGER_CLOSED if v is None else float(v)


def open_fingers_drop_below(eef_quat, opening: float) -> float:
    """How far the OPEN fingers (``tabletop.FINGER_SPHERES_EEF`` moved ``opening`` outward along the closing axis) reach below the eef origin
    at the eef orientation ``eef_quat`` (m; the hand above them is ``tabletop.hand_drop_below``)."""
    R = C.quat_to_mat(G.t(eef_quat).float())
    worst = -math.inf
    for (x, y, z), r in TT.FINGER_SPHERES_EEF:
        c = R @ th.tensor([x, y + math.copysign(float(opening), y), z], dtype=th.float32)
        worst = max(worst, -float(c[2]) + float(r))
    return worst


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


def push_face_point(hull_w: th.Tensor, direction) -> th.Tensor:
    """The contact of a push along ``direction`` (world, horizontal) on a hull given in world points: the face the pad meets (the
    hull's extreme against ``direction``) at mid height, on the line through the hull centre along ``direction``. Along +tangent
    this is the right-face point of ``touching_points``, along -tangent the left-face point."""
    d = th.tensor([float(direction[0]), float(direction[1])])
    d = d / th.linalg.norm(d)
    z = float(hull_w[:, 2].min()) + 0.5 * float(hull_w[:, 2].max() - hull_w[:, 2].min())
    face = float((hull_w[:, :2] @ d).min())
    centre = hull_w.mean(dim=0)
    return th.tensor([*(centre[:2] + d * (face - float(centre[:2] @ d))).tolist(), z])


def hull_cut_points(hull_w: th.Tensor, z: float) -> th.Tensor:
    """The cut of a convex hull, given by its points (world xyz), by the horizontal plane at ``z``: the points where the segment between
    every point at or below ``z`` and every point at or above it crosses the plane (N, 2; world xy). The hull's edges are among those
    segments and every crossing lies inside the hull, so their convex hull is the exact cut. Empty when the hull does not reach ``z``."""
    p = G.t(hull_w).float()
    lo, hi = p[p[:, 2] <= z], p[p[:, 2] >= z]
    if len(lo) == 0 or len(hi) == 0:
        return th.zeros(0, 2)
    a, b = lo[:, None, :], hi[None, :, :]
    da, db = z - a[..., 2], b[..., 2] - z  # both >= 0
    span = da + db
    t = th.where(span > 1e-12, da / span.clamp(min=1e-12), th.zeros_like(span))
    return (a[..., :2] + t[..., None] * (b[..., :2] - a[..., :2])).reshape(-1, 2)


def face_run_at(hull_w: th.Tensor, z: float, outward: th.Tensor, side_h: th.Tensor, face_depth_tol: float = 0.01) -> Optional[Tuple[float, float]]:
    """The flat run of the hull's face toward ``outward`` (horizontal unit) at height ``z``: ``(lo, hi)`` along ``side_h`` of the cut's
    points (:func:`hull_cut_points`) within ``face_depth_tol`` of its outermost offset - the face's width at that height, whatever the
    hull's vertices (a box hull has them at its top and bottom corners only). None when the hull does not reach ``z``."""
    cut = hull_cut_points(hull_w, z)
    if len(cut) == 0:
        return None
    off = cut @ G.t(outward)[:2].float()
    face = cut[off >= float(off.max()) - face_depth_tol]
    s = face @ G.t(side_h)[:2].float()
    return float(s.min()), float(s.max())


def polygon_centroid(poly) -> Optional[th.Tensor]:
    """The area centroid (xy) of a convex polygon (V, 2; CCW corners); the corners' mean for a degenerate one, None for an empty one."""
    p = G.t(poly).float().reshape(-1, 2)
    if len(p) == 0:
        return None
    q = th.roll(p, -1, dims=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-12:
        return p.mean(dim=0)
    return th.stack([((p[:, 0] + q[:, 0]) * cross).sum(), ((p[:, 1] + q[:, 1]) * cross).sum()]) / (6.0 * a)


def point_triangle_distances(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Euclidean distance (N, F) from every point of ``pts`` (N, 3) to every triangle of ``tris`` (F, 3, 3): the closest point on each
    triangle by its Voronoi regions (vertex, edge, face; Ericson, Real-Time Collision Detection 5.1.5), vectorised."""
    p = np.asarray(pts, dtype=float).reshape(-1, 1, 3)
    t = np.asarray(tris, dtype=float).reshape(1, -1, 3, 3)
    a, b, c = t[:, :, 0], t[:, :, 1], t[:, :, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def div(n, m):
        return n / np.where(np.abs(m) > 1e-18, m, 1e-18)

    s = va + vb + vc
    q = a + ab * div(vb, s)[..., None] + ac * div(vc, s)[..., None]  # inside the face region
    # the regions in reverse priority, each overriding the ones after it
    q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None], q)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * div(d2, d2 - d6)[..., None], q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * div(d1, d1 - d3)[..., None], q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
    return np.linalg.norm(p - q, axis=-1)


class HullSections:
    """oracle-fix 2026-11-06 (``generation.push_height_sweep``): a convex hull given by its points (world xyz) - its horizontal cuts
    (:meth:`cut`), the outermost material of a cut on a line (:meth:`outermost`), a cut's area centroid and the distance of points to the
    hull (:meth:`distance`). qhull's triangles give the hull's edges (a cut's corners are where they cross the plane: a few dozen points,
    where :func:`hull_cut_points` pairs every point below with every point above) and its faces (the exact point-to-hull distance). Without
    qhull (a flat or degenerate point set) the cuts are :func:`hull_cut_points`' and :meth:`distance` is None."""

    def __init__(self, hull_w):
        self.pts = G.t(hull_w).float().reshape(-1, 3)
        self.z_lo, self.z_hi = float(self.pts[:, 2].min()), float(self.pts[:, 2].max())
        self.edges = self.tris = self.planes = None
        self._cuts: Dict[float, th.Tensor] = {}
        try:
            from scipy.spatial import ConvexHull

            P = self.pts.double().numpy()
            ch = ConvexHull(P)
            s = ch.simplices
            e = np.unique(np.sort(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]), axis=1), axis=0)
            self.edges, self.tris, self.planes = (P[e[:, 0]], P[e[:, 1]]), P[s], ch.equations
        except Exception:  # noqa: BLE001 - a flat / degenerate point set (or no scipy): the all-pairs cut, no distance
            pass

    def cut(self, z: float) -> th.Tensor:
        """The cut at height ``z`` (CCW corners, (V, 2) world xy; empty when the hull does not reach ``z``)."""
        key = round(float(z), 5)
        poly = self._cuts.get(key)
        if poly is None:
            if self.edges is None:
                pts = hull_cut_points(self.pts, float(z))
            else:
                a, b = self.edges
                za, zb = a[:, 2] - float(z), b[:, 2] - float(z)
                hit = (np.minimum(za, zb) <= 0.0) & (np.maximum(za, zb) >= 0.0)
                a, b, za, zb = a[hit], b[hit], za[hit], zb[hit]
                span = zb - za
                t = np.where(np.abs(span) > 1e-12, -za / np.where(np.abs(span) > 1e-12, span, 1.0), 0.0)
                pts = th.tensor(a[:, :2] + t[:, None] * (b[:, :2] - a[:, :2]), dtype=th.float32).reshape(-1, 2)
            poly = G.convex_hull_2d(pts) if len(pts) >= 3 else th.zeros(0, 2)
            self._cuts[key] = poly
        return poly

    def outermost(self, z: float, normal, tangent, t0: float) -> Optional[float]:
        """The largest coordinate along ``normal`` of the cut at ``z`` on the line {p · ``tangent`` = ``t0``} (``iors_large_nextto.outermost_on_line``);
        None when the line misses it."""
        from ispatialgym.sampling.v2.tasks.iors_large_nextto import outermost_on_line

        poly = self.cut(z)
        if len(poly) < 3:
            return None
        return outermost_on_line([poly], normal, tangent, t0)

    def centroid(self, z: float) -> Optional[th.Tensor]:
        """The area centroid (xy) of the cut at ``z``; None when the hull does not reach it."""
        poly = self.cut(z)
        return polygon_centroid(poly) if len(poly) >= 3 else None

    def distance(self, pts) -> Optional[th.Tensor]:
        """Distance (m) of each point (N, 3; world) to the hull, 0 inside; None without qhull's faces."""
        if self.tris is None:
            return None
        P = G.t(pts).double().reshape(-1, 3).numpy()
        inside = (P @ self.planes[:, :3].T + self.planes[:, 3]).max(axis=1) <= 0.0
        d = point_triangle_distances(P, self.tris).min(axis=1)
        d[inside] = 0.0
        return th.tensor(d, dtype=th.float32)


def footprint_laterals(pts: th.Tensor, tangent) -> Tuple[float, float]:
    """(min, max) of points (a footprint polygon, a hull) along the horizontal ``tangent``."""
    v = G.t(pts)[:, :2].float() @ G.t(tangent)[:2].float()
    return float(v.min()), float(v.max())


def min_area_rect(pts) -> Tuple[float, float, float, th.Tensor]:
    """The minimum-area bounding rectangle of the points ``pts`` (N, 2+; world xy, e.g. a footprint polygon): ``(angle, long, short, centre)``
    - the world angle (rad, in [0, pi)) of its longer side, the lengths of its two sides (one side lies along an edge of the convex hull)
    and its centre (world xy)."""
    xy = G.t(pts)[:, :2].float()
    hull = G.convex_hull_2d(xy) if len(xy) > 2 else xy
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if float(th.linalg.norm(e)) < 1e-9:
            continue
        ang = math.atan2(float(e[1]), float(e[0]))
        c, s = math.cos(ang), math.sin(ang)
        u, v = hull[:, 0] * c + hull[:, 1] * s, -hull[:, 0] * s + hull[:, 1] * c
        a, b = float(u.max() - u.min()), float(v.max() - v.min())
        if best is None or a * b < best[0] - 1e-12:
            uc, vc = 0.5 * float(u.max() + u.min()), 0.5 * float(v.max() + v.min())
            best = (a * b, ang, a, b, th.tensor([uc * c - vc * s, uc * s + vc * c]))
    if best is None:
        return 0.0, 0.0, 0.0, (xy.mean(dim=0) if len(xy) else th.zeros(2))
    _, ang, a, b, centre = best
    if b > a:
        ang, a, b = ang + math.pi / 2, b, a
    return ang % math.pi, a, b, centre


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


@register(*RELATION_OF_SUBTASK)
class IORSSmall(SamplerV2):
    """One class for the five small-scale relations; the relation comes from the subtask id."""

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
        self.grasps, self.grasp_key, self.rest["target"] = TT.library_grasps(self, "target")
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
            if free < float(self.gen("shelf_free_span_min_m", 0.15)):
                raise Reject("shelf_width", f"{self.support.name}: {free:.2f} m between the two leaning books (< {float(self.gen('shelf_free_span_min_m', 0.15)):.2f})")
            if self.shelf_start_draw() is not None:
                # oracle-fix 2026-11-06 (generation.shelf_start_draw): the template's start (the key's lean and side, its compartment) is what
                # every attempt draws its own from (:meth:`draw_shelf_side`, :meth:`draw_shelf_layout`); the template check above stays
                self._shelf_template = {k: self.shelf_start.get(k) for k in ("key", "lean_side", "lean_deg", "context")}
                self._shelf_rest0 = {"target": self.rest["target"].clone(), "partner": self.rest["partner"].clone()}
                self.template_facts["shelf_start_draw"] = self.shelf_draw_facts()
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
        for g in grasps:
            off = self.pinch_offset(g.pose)
            offsets[g.name] = off
            if abs(off) <= limit:
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
        t_ext = TT.hull_extents(self.hull["target"], self.standing_quats()[0] if up is not None else self.rest["target"])
        p_ext = TT.hull_extents(self.hull["partner"], self.rest["partner"])
        out: Dict[str, Any] = {"target_extent": t_ext.tolist(), "partner_extent": p_ext.tolist(), "relation": self.relation}
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
        return [c for c in (self.manip_candidates if cands is None else cands) if TT.in_start_region(c, side) and TRAVEL_DROPPED not in c.extras]

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
    def _spa_band_signature(bands) -> Optional[Dict[str, Any]]:
        """:func:`tabletop.band_facts` plus the band's key / lattice / trunk count: a band recomputed under the same path changes the key."""
        if bands is None:
            return None
        meta = getattr(bands, "meta", None) or {}
        return {**(TT.band_facts(bands) or {}), "key": meta.get("key"), "lattice": meta.get("lattice"), "n_trunks": len(meta.get("trunks") or [])}

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
                "segment_arms": arms, "bands": {a: self._spa_band_signature(self.band_of_arm(a)) for a in arms},
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
            if free < float(self.gen("shelf_free_span_min_m", 0.15)):
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
            if free < float(self.gen("shelf_free_span_min_m", 0.15)):
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
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        init = self.init_stance
        self.sidecar.add_stance(init)
        cam_init = self.camera_pose_at(init)
        # A1: the target starts in the start region of the attempt's partner side (7e: never on a candidate dropped for its travel)
        cands = [c for c in self.manip_candidates if TT.in_start_region(c, self.region_side) and TRAVEL_DROPPED not in c.extras]
        order = rng.permutation(len(cands))
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
                    if travel_failure_deterministic(e, self.world):
                        self.drop_candidate(cand, e)
                    raise
                # §4.6 step 1: target init pose from the feasible set F at the focus (exact IK confirmation); the upright shelf start is
                # fixed, its grasp confirmed with nothing ignored (the board above and the side walls are obstacles at the grasp pose)
                if self.shelf_pose is not None:
                    t_pose = self.shelf_pose
                    try:
                        # oracle-fix 2026-11-05 (speed, opt-in generation.shelf_transit_per_grasp): the transit check judges each grasp that passes
                        # the IK - the first with a clear transit is taken - instead of failing the stance on the first grasp's transit
                        accept = (lambda rec: self.shelf_transit_why(manip, rec, default, pre_open)) if per_grasp else None
                        grasp = TT.grasp_at_pose(self, manip, t_pose, self.grasps, [], self.edge.yaw, arm=default, scale_key=self.choice.models["target"].scale,
                                                 fingers=pre_open, accept=accept, prescreen=bool(self.gen("fixed_start_prescreen")))
                        if pre_open is not None:  # oracle-fix 2026-11-04 round 4: checked pre-shaped -> the oracle's transit to the pre-grasp pre-shaped too
                            grasp.extras["preshape_transit"] = True
                        why_transit = self.shelf_transit_why(manip, grasp, default, pre_open) if (self.gen("shelf_transit_check") and not per_grasp) else None
                        if why_transit is not None:
                            raise Reject("grasp_unreachable", why_transit)
                        if self.shelf_backoff_m() > 0.0:
                            # oracle-fix 2026-11-06 (owner feedback on the touching#03 demo, opt-in generation.shelf_backoff_transit_m): the arrival
                            # transition recorded again backed off, the arm carried to the pre-grasp inside it (:meth:`record_shelf_backoff`)
                            self.record_shelf_backoff(manip, grasp, default, pre_open)
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
                return min(self.ikc.eef_frame_margin(place_g, _cam), self.ikc.eef_frame_margin(self.pre_place_of(place_g), _cam))
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
        # the place / pre-place of this candidate must project into its manipulation image (owner 1b: the FOV
        # monitor arms from the pre-pose): checked here, before the settle, so the next candidate is tried instead of
        # the attempt dying in the goal check (inside: release rim + 4-5 cm and the pre-place 10 cm above it)
        if self.relation != "touching":
            cam_manip = self.camera_pose_at(manip)
            goal_guess = self.goal_pose(t_pose, p_pose)
            place_guess = IK.grasp_pose(goal_guess, grasp.eef_in_object)
            for what, pose in (("place", place_guess), ("pre-place", self.pre_place_of(place_guess))):
                if not self.ikc.eef_in_frame(pose, cam_manip):
                    raise Reject("gripper_not_in_frame", f"{what} eef pose outside the manipulation image (5 %) for candidate {int(k)} "
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
                                           scale_key=self.choice.models["target"].scale, cache=False, approached=ign_place)
            if not res_place.ok:
                raise Reject(res_place.code or "place_unreachable", f"place screening at the draw (candidate {int(k)}, trunk {manip.translate}, "
                                                                    f"{manip.tilt:.2f}, from the {st_place.name} stance): {res_place.reason}")
        return p_pose

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
        right_cells = self.bands.band(manip.translate, manip.tilt, **TT.band_kw(self.bands, manip)) if getattr(self, "bands", None) is not None else None
        left_cells = left_contact_band_cells(self, (manip.translate, manip.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]), stance=manip)
        self.template_facts["left_contact_band"] = "unavailable" if left_cells is None else int(len(left_cells))
        faces_yaws = self.widest_faces_partner_yaws(t_pose, manip) if self.widest_faces() else None  # the two turns parallel to the target
        if self.pair_goal_align() is not None:
            faces_yaws = [self.pair_partner_yaw(t_pose)]  # the front at the required turn from the target's
        errors = []
        p_pose = None
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
        (``tabletop.releaned_quat``), the partner leaning ``lean_partner`` deg toward the other board (``tabletop.upright_lean_quat``)."""
        base = self._shelf_template
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
            spine = float(rng.uniform(*spec["spine_behind_edge_m"])) if spec["spine_behind_edge_m"] else None
            rest_t, rest_p = self.shelf_rests(side, lean_t, lean_p)
            label = f"bay {bay:+.2f} {TT.side_name(side)} board, leans {lean_t:.1f} / {lean_p:.1f} deg" + ("" if spine is None else f", spine {spine:.3f} m")
            try:
                t_pose, tf = TT.shelf_lean_pose(self, self.hull["target"], rest_t, self.top, self.edge, self.support, side, spine_behind=spine, lateral0=bay)
                p_pose, pf = TT.shelf_lean_pose(self, self.hull["partner"], rest_p, self.top, self.edge, self.support, -side, spine_behind=spine, lateral0=bay)
                free = shelf_free_span(self.world.hull_at(self.hull["target"], t_pose), self.world.hull_at(self.hull["partner"], p_pose), self.edge, side)
                if free < span_min:
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
        tangent = manip.tangent
        t_hull = self.world.hull_at(self.hull["target"], t_pose)
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        z_t = float(t_hull[:, 2].min()) + 0.5 * float(t_hull[:, 2].max() - t_hull[:, 2].min())
        right_face = float((t_hull[:, :2] @ tangent).min())
        centre_t = t_hull.mean(dim=0)
        work = th.tensor([*(centre_t[:2] + tangent * (right_face - float(centre_t[:2] @ tangent))).tolist(), z_t])
        left_face = float((p_hull[:, :2] @ tangent).max())
        centre_p = p_hull.mean(dim=0)
        z_p = float(p_hull[:, 2].min()) + 0.5 * float(p_hull[:, 2].max() - p_hull[:, 2].min())
        brace = th.tensor([*(centre_p[:2] + tangent * (left_face - float(centre_p[:2] @ tangent))).tolist(), z_p])
        return work, brace

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
        if grasp is not None and grasp.eef_in_object is not None:
            grasp.grasp = IK.grasp_pose(draft.object_poses[tname], grasp.eef_in_object)
            grasp.pre_grasp = IK.pre_pose(grasp.grasp)
            manip = self.sidecar.stance("manipulation")
            settled = self.settle_motion(draft, tname)
            draft.summary["target_settle"] = settled
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
            r = self.ikc.exact([grasp.pre_place, grasp.place], st_place, grasp.arm, ignore_for_last=ign, scale_key=scale, cache=False, approached=ign)
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
                return f"place retreat clear pose: {r.reason}"
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
        if not res.ok and res.code and "self-collision (attached_object" in res.reason:
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
            clear = IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)
            r_clear = self.ikc.exact([clear], st_place, grasp.arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
            if not r_clear.ok:
                raise Reject(r_clear.code or "place_unreachable", f"place retreat clear pose ({(RETREAT_DIST_M + RETREAT_LITTLE_M) * 100:.0f} cm back along the "
                                                                 f"approach) from the {st_place.name} stance: {r_clear.reason}")
        # the lift (oracle ``Lift``): grasp pose raised by carry_height with the object attached, support ignored (the
        # object leaves it); the pre-place is where the carry ends, so the lift end must be IK-feasible too
        lift_end = PC.lift_end_pose(grasp)
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            # tracked from the grasp configuration (one IK branch along the lift), support ignored throughout
            r_lift = self.ikc.waypoints_ok([grasp.grasp, lift_end], manip, grasp.arm, ignore_all=self.support_objects(), cache=False, approached=[])
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

    def brace_gate_at(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Dict[str, Any]:
        """A1 joint band gate at stance ``st``: the push work point (target's right face at the place) in the right band and the
        brace point (partner's far face) in the left contact band of the same trunk (``brace_placement_admissible``). The shelf
        layout has no brace at any stance."""
        if self.shelf_start is not None:
            return {"ok": False, "stance": st.name, "why": "shelf layout: the side board holds the partner, no brace"}
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
        hull = G.t(self.hull[role])
        eef = (G.t(eef_pose[0]), G.t(eef_pose[1]))
        if against == "hull":
            sec = HullSections(self.world.hull_at(self.hull[role], obj_pose))
            worst = None
            for part, (centres, r) in HAND_SPHERES_EEF.items():
                dist = sec.distance(C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
                if dist is None:
                    break
                dmin = float(dist.min())
                if dmin < r + margin and (worst is None or dmin - r < worst[1]):
                    worst = (part, dmin - r)
            else:
                return None if worst is None else f"{worst[0]} {worst[1] * 1000:+.0f} mm from the hull (margin {margin * 1000:.0f} mm)"
        lo, hi = hull.min(dim=0).values, hull.max(dim=0).values
        inv = C.invert((G.t(obj_pose[0]), G.t(obj_pose[1])))
        worst = None
        for part, (centres, r) in HAND_SPHERES_EEF.items():
            pts_o = C.transform_points(inv, C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
            d = th.clamp(lo - pts_o, min=0.0) + th.clamp(pts_o - hi, min=0.0)
            dist = float(th.linalg.norm(d, dim=1).min())
            if dist < r + margin and (worst is None or dist - r < worst[1]):
                worst = (part, dist - r)
        if worst is None:
            return None
        return f"{worst[0]} {worst[1] * 1000:+.0f} mm from the box (margin {margin * 1000:.0f} mm)"

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
            r = self.ikc.exact([self.shelf_pre_place(place), place], st, grasp.arm, ignore_for_last=ign, scale_key=self.choice.models["target"].scale, cache=False, approached=ign)
        if not r.ok:
            return f"descent: {r.reason}"
        r2 = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)], st, grasp.arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if not r2.ok:
            return f"clear pose: {r2.reason}"
        return None

    def push_stance_for(self, manip: Stance, t_place_pose: G.Pose, lateral: Optional[float] = None) -> Optional[Stance]:
        """Touching (owner 2026-09-18 16:32): the second base pose ``push`` = the manipulation stance moved along the edge, same
        trunk, so that the push contact (the placed box's right face) lies at ``push_contact_lateral_m`` from the robot's centre
        line (default -0.12: every push that worked had it 7-18 cm right of centre; at -0.05 the right arm crosses the body
        and joints 2 / 3 sit on their limits, 0918-173814). The box is carried there in hand and set down from there.
        Pure geometry (nothing recorded); None when the shift is below ``push_stance_min_shift_m`` or the feature is off
        (``generation.push_stance``). Shelf layout: the contact on the target's face away from the partner and the right hand's
        lateral mirrored for the left hand (+0.12: left of centre), the stance's working arm the shelf push arm."""
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
        if abs(dlat) < float(self.gen("push_stance_min_shift_m", 0.02)):
            return None
        target_xy = G.t(manip.base_xy) + G.t(tangent) * dlat
        cands = [manip.shifted(lateral=manip.lateral + s * dlat, name="push") for s in (1.0, -1.0)]  # the edge frame's lateral sign
        st = min(cands, key=lambda c: float(th.linalg.norm(G.t(c.base_xy) - target_xy)))
        st = st.with_trunk(manip.translate, manip.tilt, "push")
        st.focus, st.arms = G.t(t_place_pose[0]).clone(), ["right"]
        if self.shelf_start is not None:
            st.arms = [self.shelf_push_arm()]
        st.extras.update({"second_base_pose": True, "lateral_shift_m": round(dlat, 4), "push_contact_lateral_m": round(float(lat_work - dlat), 4),
                          "rule": "base moved along the edge (box in hand) so the push contact lies at push_contact_lateral_m from the centre line"})
        return st

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
        for lat in laterals:
            cand = self.push_stance_for(manip, t_place_pose, lateral=lat)  # None = the manipulation stance already puts the contact there
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
            place_why = self.place_infeasible_why(probe, t_place_pose, grasp) if kind is not None else None
            both = self.push_chain_feasible(probe, t_place_pose, p_pose, with_brace=True, height_range=(h, h) if (sweep and h is not None) else None) \
                if (kind is not None and place_why is None and want_brace) else None
            # oracle-fix 2026-11-06 (push_gripper_open_on): an open class's height is named too (today's mid-height classes carry none), and a
            # brace that joined today's class instead of the open push names that class
            both_push = both.split("+", 1)[0] if both else None
            tried.append(f"{lat:+.2f}{' (manipulation)' if cand is None else ''}: {kind or no_class}{f' @ {100 * h:.1f} cm' if h is not None else ''}"
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
                    return self._record_push_stance(draft, manip, grasp, cand, t_place_pose, p_pose)
                except Reject as e:
                    whys.append(f"{float(cand.extras.get('lateral_shift_m', 0.0)):+.2f}: {e}")
                    tried.append(f"tip release drive {whys[-1][:160]}")
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
                raise Reject("place_unreachable", f"touching tip release: the place fails from every lateral ({'; '.join(tried)})")
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
            r_lift = self.ikc.exact([lift_end], manip, grasp.arm, ignore_for_last=self.support_objects(), cache=False)
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
                                        "path_length_m": st.extras["path_length_m"], "brace_gate": self._brace_gate}
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

    def require_in_frame(self, cam: G.Pose, poses: Dict[str, Optional[G.Pose]]):
        """Every given eef pose must project inside the manipulation image with the 5 % margin (``IKChecker.eef_in_frame``
        with the shared camera FK of ``camera_pose_at``); ``Reject("gripper_not_in_frame")`` names the first that does not. Opt-in
        ``generation.grasp_frame_rule: "fingertip"`` (oracle-fix 2026-11-04 round 4, IORSLarge's rule): QA's own criterion instead - the fingertip
        point (``conventions.fingertip``) inside the image with the 5 % margin (t03tip3 opryhp: the shelf pre-grasp's +-6 cm eef box left the image)."""
        tip_rule = str(self.gen("grasp_frame_rule") or "box") == "fingertip"
        for what, pose in poses.items():
            if pose is None:
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
                            f"{self.gen('brace_wide_spreads_m', [0.09, 0.07, 0.05])} straddles it {BRACE_OPEN_END_MARGIN_M:.2f} m from the face's ends")
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
        if brace_chosen is not None:
            pre, contact, kind, extras = brace_chosen
            brace_rec = TT.BraceRecord("brace", "left", pre, contact, direction=-direction, brace_kind=kind, object=pname, extras=extras)
            brace_rec.stance = manip.name
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

        def label(c):
            h = c[2].get("contact_height_m")
            return c[0] if h is None else f"{c[0]}@{100.0 * float(h):.1f}cm"

        def evaluate(p):
            pre, init, goal = p.chain
            fingers = push_fingers(p.payload[2])  # closed, or the open push's opening (oracle-fix 2026-11-06)
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
            overlap = self.hand_overlap(contact, "partner", p_pose)  # camera / knuckles / body vs the braced partner
            if overlap is not None:
                return False, None, None, overlap
            with self.posed("target", pushed), self.ikc.ignoring([self.obj("target")]):
                res = self.ikc.exact_both_arms(("right", list(push_chain)), ("left", [pre, contact]), st, ignore_push, ignore_brace, fingers=push_fingers_q,
                                               q_first_other=p.q_first)
            if not res.ok:
                if not diagnose:
                    return False, res, None, str(res.reason)
                # diagnostic (owner 2026-09-18 20:48): the left chain on its own, the right arm at the stance posture - a pass here names
                # the two-arm combination (self-collision / the locked right arm) as the blocker
                alone = self.ikc.exact([pre, contact], st, "left", ignore_for_last=ignore_brace, cache=False, fingers=IK.FINGER_CLOSED)
                return False, res, None, f"{res.reason} [left alone: {'ok' if alone.ok else alone.reason}]"
            qs = O.chain_qs(self.ikc, "left", [pre, contact])
            r_ret = self.ikc.exact([IK.pre_pose(contact, RETREAT_DIST_M)], st, "left", ignore_for_last=ignore_brace, cache=False)
            if not r_ret.ok:
                return False, res, None, f"brace retreat clear pose: {r_ret.reason}"
            return True, res, qs, ""

        stream = self.touching_brace_candidates(st, b_point, face_z, -direction, partner_hull_w=p_hull, push_point=p_init)
        if limit is not None:
            stream = itertools.islice(stream, int(limit))
        return O.choose_first(self.ikc, "left", stream, key=lambda c: c[1], label=lambda c: c[2], chain=lambda c: [c[0], c[1]], evaluate=evaluate,
                              q_start=self.ikc.q_locked(st), stage="touching brace")

    def touching_push_candidates(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                                 t_hull_w: Optional[th.Tensor] = None):
        """Push poses of the closed right gripper at the target's right face (``face_point``, mid height), pushing along
        ``direction`` (robot's left, into the partner), in the order to try: (1) the pads on the face — pad centre on the
        point, approach along ``direction``, closing horizontal then vertical; (2) the top-down blade beside the face
        (``tabletop.blade_standoff`` over the target's face height above the tips, pitched per ``approach_pitch_order``,
        pads' faces parallel to the pushed face, either wrist turn) — a sideways pad push 10 cm right of the target was
        out of reach for the right arm on every attempt of the sanitary-napkins loop. Yields ``(kind, eef pose, extras)``.
        ``t_hull_w`` (the target's hull at the place, given under :meth:`push_height_sweep_on`): :meth:`swept_push_candidates` instead."""
        if t_hull_w is not None:
            yield from self.swept_push_candidates(manip, t_hull_w, face_point, face_z, direction)
            return
        d = C.unit(G.t(direction))
        outward = -d  # the pushed (right) face's outward normal
        z0, z1 = face_z
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))
        # A/B knob: pad contacts lifted by the constant (pad centre + half the finger height below the face top), blades lifted
        # with the fingertip kept 1 cm below the face top
        pad_lift, pad_why = contact_height_extra(self, "push", pad_center_z=float(face_point[2]), band_top_z=z1)
        fp = G.t(face_point) + th.tensor([0.0, 0.0, pad_lift])
        pad_facts = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}
        # owner 2026-09-18 19:39 / 20:10: FINGERTIP-FRONT pushes at the face's mid height first - the hand horizontal along the
        # push direction, the closed tips ``clearance`` before the face (``fingertip_eef``; ``contact_eef`` puts the pad centre
        # on the face, tips 3 cm inside the box). Pushing at the centre-of-mass height the tall narrow box slides; every blade
        # pushed it 9 cm up and it fell over (0918-194552: z 0.867 -> 0.848). Horizontal closing: the knuckles beside the
        # pads, the wrist camera 7.6 cm behind the tips (never over the box), the gripper body ~1.5 cm above the table at the
        # mid height; ``pre_dist`` 5 cm (the 10 cm pre 10 cm to the right of the face was out of IK reach on every attempt).
        # A vertical closing hangs the camera 6-7 cm below / above the pads: only the camera-UP turn is admissible.
        h_axis = C.default_closing_axis(d)
        tip_pre = float(self.gen("push_tip_pre_dist_m", 0.05))
        for c_sign in PUSH_WRIST_TURNS:
            pose = C.fingertip_eef(fp, d, c_sign * h_axis, clearance)
            yield f"push_tip_h{'' if c_sign > 0 else '-'}", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True,
                                                                                            "stop_clearance_m": clearance, **pad_facts}
        pose = C.fingertip_eef(fp, d, th.tensor([0.0, 0.0, -1.0]), clearance)  # eef +y = down: the camera (eef -y side) points up
        yield "push_tip_v-", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **pad_facts}
        yield "push_h", IK.contact_eef(fp, d), {"blade_half_thickness_m": None, **pad_facts}
        yield "push_v", IK.contact_eef(fp, d, closing=th.tensor([0.0, 0.0, 1.0])), {"blade_half_thickness_m": None, **pad_facts}
        raised_fallback = None
        # owner 2026-09-18 15:50: a sideways push uses the VERTICAL blade only - ``pitched_approach`` leans the blade about the
        # push axis, so a pitched finger side meets the face obliquely and yaws the box (0918-153503: 45 deg blade, 29 deg yaw,
        # contact lost); ``generation.push_blade_pitches_deg`` lists the admissible pitches
        pitches = [float(p) for p in (self.gen("push_blade_pitches_deg", None) or [0.0])]
        for pitch in [p for p in TT.approach_pitch_order(None) if p in pitches]:
            z_tip = z0 + 0.005
            if bool(self.gen("blade_toward_com", False)):  # off since 2026-09-18 16:00: the 2 cm raise put the recorded retreat's clear pose at the reach limit (0918-154421: 4/8 'retreat: exact IK failed'); the oracle raises the pad at run time for the palm
                # owner 2026-09-17 23:39: push nearer the centre of mass - the finger section (centre ~3.5 cm above the
                # tips) is centred on the face's mid height when the face allows it, else it stays at the face bottom
                z_tip = max(z_tip, min(float(face_point[2]) - 0.035, z1 - 0.03))
            blade_lift, blade_why = contact_height_extra(self, "blade", fingertip_z=z_tip, band_top_z=z1)
            z_tip += blade_lift
            # a push needs the blade on the lower part of the face only (unlike a stop brace, which the whole face may
            # meet): faces taller than the finger section are pushed over their lowest ``PALM_FROM_TIP`` (7 cm)
            # (the finger section is PALM_FROM_TIP long along the finger axis: pitched, it covers cos(pitch) of that in height)
            z_face_hi = min(z1, z_tip + (TT.PALM_FROM_TIP - 0.005) * math.cos(math.radians(pitch)))
            # owner 2026-09-17 23:39 (touching 0917-22*: the palm pressed the sanitary-napkins box and shook the table): a
            # face that reaches the palm section of a vertical blade (z1 above the fingers' 7 cm) is only pushed by a
            # PITCHED blade - the palm leans away from the face by ~ PALM_FROM_TIP·sin(pitch) - never by the 0° one,
            # whose stand-off (finger profile only) puts the wider palm inside the face
            # (default 0 = off since 2026-09-18 15:45: ``pitched_approach`` leans the blade FORWARD, about the push axis, so
            # a pitched blade does not move the palm away from a face lying to the pad's side - the 45 deg blade of
            # 0918-144517 knocked the box over; the oracle raises a vertical blade until its palm clears the face top)
            if z1 > z_tip + TT.PALM_FROM_TIP and pitch < float(self.gen("blade_palm_min_pitch_deg", 0.0)):
                # fallback (after the pitched blades, which come first in the order): the vertical blade RAISED so that its
                # palm clears the face top - the finger section pushes the top PALM_FROM_TIP of the face (re-sample
                # 0918-010325: every pitched blade fails the exact IK from the stance)
                z_hi_tip = z1 - TT.PALM_FROM_TIP + 0.005
                if bool(self.gen("blade_palm_clear_fallback", True)) and z_hi_tip <= z1 - 0.03:
                    raised_fallback = (z_hi_tip,)  # yielded after the pitched blades (below)
                continue
            leans = self.gen("blade_lean_deg", [20.0, 12.0, 8.0])
            leans = [float(v) for v in (leans if isinstance(leans, (list, tuple)) else [leans]) if float(v) > 0]
            if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and leans:
                # owner 2026-09-18 19:05: a tall face is pushed LOW by a blade leaning along the push axis (fingertips lead, the
                # palm recedes from the face by h·tan(lean)): the raised blade pushed the top 4 cm of the 11 cm box and it
                # tipped instead of sliding - each base nudge closed 10.9 mm and the same 10.9 mm reopened at the back-off
                # (0918-184752: touching held during the push, lost at the last step). Leans in order: 20 deg has no pre-pose
                # IK at these stances (0918-191811, the wrist roll at its limit); at 12 deg the thick finger section touches at
                # ~ the centre-of-mass height and the palm still recedes 1.5 cm.
                # The tips ``blade_lean_tip_above_m`` above the face bottom: leaning, the fingertip's leading edge dips below the tip
                # height and 5 mm above the table collided with it at the contact pose (0918-190236)
                z_lean_tip = z0 + float(self.gen("blade_lean_tip_above_m", 0.025))
                for lean in leans:
                    yield from self._leaning_blade(manip, face_point, (z0, z1), d, outward, clearance, z_lean_tip, lean)
            if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and bool(self.gen("blade_palm_clear_first", True)):
                # owner 2026-09-18 18:45: a face taller than the finger section is pushed by the RAISED vertical blade first (palm
                # ``blade_palm_clear_m`` above the face top, validated by the chain check) - the low blade's palm overlaps the top
                # and the oracle raised it 3.9 cm at run time to an unvalidated pose: 'joints did not converge 0.087 rad'
                # (0918-182409); with 5 mm of clearance the wider palm / knuckle came down ON the box's top edge as the blade
                # descended beside the face (0918-183535, same error) - 2.5 cm by default
                palm_clear = float(self.gen("blade_palm_clear_m", 0.025))
                yield from self._raised_blade(manip, face_point, (z0, z1), d, outward, clearance, z1 - TT.PALM_FROM_TIP + palm_clear)
            so = TT.blade_standoff((z0, z_face_hi), z_tip, pitch, clearance, side=False)
            if so is None:
                continue  # nothing of the face above the tips: no blade push
            standoff, half = so
            a_vec = TT.pitched_approach(pitch, manip.forward)
            origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
            for c_sign in PUSH_WRIST_TURNS:
                yield f"push_blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                    "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                    "height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(blade_lift, 4), "height_extra_clip_reason": blade_why}
        if raised_fallback is not None:
            yield from self._raised_blade(manip, face_point, (z0, z1), d, outward, clearance, raised_fallback[0])

    def _leaning_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float,
                       z_tip: float, lean_deg: float):
        """The closed blade with its fingertips ``z_tip`` (5 mm above the face bottom) and the hand leaning ``lean_deg`` along
        the push direction (approach = down rotated toward ``d``): the fingertips lead, every higher finger section recedes
        from the face by h·tan(lean), so only the tips touch (below the centre of mass: the box slides, no tipping) and the
        palm never reaches the face. Stand-off = the largest protrusion of the finger profile toward the face + clearance."""
        z0, z1 = face_z
        theta = math.radians(float(lean_deg))
        d3 = th.tensor([float(d[0]), float(d[1]), 0.0])
        d3 = d3 / max(float(th.linalg.norm(d3)), 1e-9)
        a_vec = C.unit(th.tensor([0.0, 0.0, -math.cos(theta)]) + d3 * math.sin(theta))
        protrude, h_prev = 0.0, 0.0
        for h, half in TT.FINGER_PROFILE:  # a section spans (h_prev, h]; its lowest point protrudes the most
            protrude = max(protrude, float(half) - h_prev * math.tan(theta))
            h_prev = float(h)
        standoff = protrude + float(clearance)
        # eef origin = fingertip - a·FINGERTIP_LENGTH, the tips ``standoff`` outside the face at ``z_tip``
        tip = th.tensor([float(face_point[0]), float(face_point[1]), float(z_tip)]) + outward * standoff
        origin = tip - a_vec * C.FINGERTIP_LENGTH
        for c_sign in PUSH_WRIST_TURNS:
            yield f"push_blade_lean{lean_deg:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                "blade_half_thickness_m": float(TT.FINGER_PROFILE[0][1]), "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": float(z_tip),
                "palm_clear": True, "lean_deg": float(lean_deg), "height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0,
                "height_extra_clip_reason": "leaning_blade"}

    def _raised_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float, z_tip: float):
        """The vertical (0 deg) blade with its fingertips at ``z_tip`` = face top − PALM_FROM_TIP: the palm stands above
        the face, the finger section covers the face's top part. Yielded after every pitched blade failed."""
        z0, z1 = face_z
        so = TT.blade_standoff((z0, z1), z_tip, 0.0, clearance, side=False)
        if so is None:
            return
        standoff, half = so
        a_vec = TT.pitched_approach(0.0, manip.forward)
        origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH]) + outward * standoff
        for c_sign in PUSH_WRIST_TURNS:
            yield f"push_blade_0_high{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                "palm_clear": True, "height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "palm_clear"}

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
        """:meth:`push_height_sweep_on` (touching#04: every class at mid height failed, so the palm-clear raised blade pushed the 27 cm
        spray bottle at 83 % of its height and it tilted 3.6 -> 20.3 deg): today's push classes - the fingertip pushes ``push_tip_h`` /
        ``push_tip_v-``, the pad pushes ``push_h`` / ``push_v``, the leaning blades ``blade_lean_deg``, the vertical blade - at every
        contact height (m above the target's bottom, the support top) from the class's lowest that clears the support
        (:data:`PUSH_SWEEP_MIN_HEIGHT_M`; a leaning blade's ``blade_lean_tip_above_m``) in ``push_height_step_m`` (0.01) steps up to
        ``push_max_height_fraction`` (0.5) x the target's height, yielded by contact height (a blade's fingertips, else the pads / tips;
        today's class order breaks ties, a class's two wrist turns stay together). The contact lies on the line through the target's
        footprint centre (the area centroid of its hull's cut :data:`PUSH_SWEEP_BASE_CUT_M` above the bottom: the part it stands on, not
        the trigger head the whole hull's outermost point was) along the push, on the hull's cut there: the outermost material within
        :data:`PUSH_SWEEP_WINDOW_M` of the line and of the height, so the tips start ``stop_clearance_m`` off everything they face; a
        blade stands off from the cut over its whole finger section (``TT.FINGER_PROFILE``). The hand above the fingers is left to
        :meth:`hand_overlap` against the hull (a vertical blade beside a tall face fails there: its palm and knuckles reach into it), so
        a blade that passes keeps its recorded height (``palm_clear``: the oracle does not raise it). The raised blade comes last and only
        for a target that is not tall (height / smallest footprint side <= ``push_raised_max_aspect``, :data:`PUSH_RAISED_MAX_ASPECT`);
        a tall target without a low contact fails ``contact_unreachable`` at the stance, and another stance is tried. Yields
        ``(kind, eef pose, extras)``; the extras carry ``contact_height_m``, ``height_fraction`` and ``contact_point`` (world)."""
        d = C.unit(G.t(direction).float())
        outward = -d
        d2 = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
        side2, out2 = th.tensor([-float(d2[1]), float(d2[0])]), -d2
        z0, z1 = float(face_z[0]), float(face_z[1])
        height = max(z1 - z0, 1e-6)
        sec = HullSections(t_hull_w)
        centre = sec.centroid(z0 + PUSH_SWEEP_BASE_CUT_M)
        if centre is None:
            centre = polygon_centroid(G.convex_hull_2d(G.t(t_hull_w)[:, :2].float()))
        t0 = float(centre @ side2)
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))
        step = max(float(TT.phase2_gen(self, "push_height_step_m", 0.01)), 0.002)
        h_cap = float(TT.phase2_gen(self, "push_max_height_fraction", 0.5)) * height
        tip_pre = float(self.gen("push_tip_pre_dist_m", 0.05))
        knob = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "push_height_sweep"}

        def face(z: float, dz: float = PUSH_SWEEP_WINDOW_M) -> Optional[float]:
            """The outermost material (along ``outward``) within the window around the centre line at height ``z``; None where the line
            itself misses the cut."""
            if sec.outermost(z, out2, side2, t0) is None:
                return None
            zs = [zz for zz in ((z - dz, z, z + dz) if dz > 0 else (z,)) if z0 <= zz <= z1] or [z]
            vals = [sec.outermost(zz, out2, side2, t0 + k * 0.5 * PUSH_SWEEP_WINDOW_M) for zz in zs for k in (-2, -1, 0, 1, 2)]
            return max(v for v in vals if v is not None)

        def point(s: float, z: float) -> th.Tensor:
            xy = side2 * t0 + out2 * float(s)
            return th.tensor([float(xy[0]), float(xy[1]), float(z)])

        def heights(h_min: float) -> List[float]:
            return [round(h_min + k * step, 4) for k in range(int(math.floor((h_cap - h_min) / step + 1e-6)) + 1)] if h_min <= h_cap + 1e-9 else []

        def facts(h: float, contact: th.Tensor) -> Dict[str, Any]:
            return {"contact_height_m": round(h, 4), "height_fraction": round(h / height, 3), "contact_point": [round(float(v), 4) for v in contact],
                    "push_height_sweep": True, **knob}

        def tips(h: float) -> List[Tuple[str, G.Pose, Dict[str, Any]]]:  # push_tip_h / push_tip_h- (closing horizontal)
            s = face(z0 + h)
            if s is None:
                return []
            fp, h_axis = point(s, z0 + h), C.default_closing_axis(d)
            return [(f"push_tip_h{'' if c > 0 else '-'}", C.fingertip_eef(fp, d, c * h_axis, clearance),
                     {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **facts(h, fp)}) for c in PUSH_WRIST_TURNS]

        def tip_v(h: float):  # push_tip_v- (closing vertical, the camera up)
            s = face(z0 + h)
            if s is None:
                return []
            fp = point(s, z0 + h)
            return [("push_tip_v-", C.fingertip_eef(fp, d, th.tensor([0.0, 0.0, -1.0]), clearance),
                     {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **facts(h, fp)})]

        def pad(h: float, vertical: bool):  # push_h / push_v (the pad centre on the face)
            s = face(z0 + h)
            if s is None:
                return []
            fp = point(s, z0 + h)
            pose = IK.contact_eef(fp, d, closing=th.tensor([0.0, 0.0, 1.0])) if vertical else IK.contact_eef(fp, d)
            return [("push_v" if vertical else "push_h", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, **facts(h, fp)})]

        def section_standoff(z_tip: float, lean: float) -> Optional[Tuple[float, float, float]]:
            """``(tip offset along outward, the face at the tips, the thickest half)``: the fingertip axis this far out that every finger
            section (``TT.FINGER_PROFILE``, receding h·sin(lean) when leaning) keeps ``clearance`` off the cut; None without material."""
            th_l = math.radians(lean)
            need, half_max, h_prev = None, 0.0, 0.0
            for h_sec, half in TT.FINGER_PROFILE:
                for hf in np.linspace(h_prev, float(h_sec), max(2, int(math.ceil((float(h_sec) - h_prev) / 0.01)) + 1)):
                    zz = z_tip + float(hf) * math.cos(th_l)
                    if not (z0 <= zz <= z1):
                        continue
                    s = face(zz, dz=0.0)
                    if s is None:
                        continue
                    v = s + float(half) - float(hf) * math.sin(th_l)
                    need, half_max = (v if need is None else max(need, v)), max(half_max, float(half))
                h_prev = float(h_sec)
            s_tip = face(z_tip)
            if need is None or s_tip is None:
                return None
            return need + clearance, s_tip, half_max

        def blade(h: float):  # push_blade_0 / push_blade_0- (vertical, beside the face)
            z_tip = z0 + h
            so = section_standoff(z_tip, 0.0)
            if so is None:
                return []
            s_o, s_tip, half = so
            a_vec = TT.pitched_approach(0.0, manip.forward)
            origin = point(s_o, z_tip + C.FINGERTIP_LENGTH)
            return [(f"push_blade_0{'' if c > 0 else '-'}", (origin, C.eef_quat(a_vec, c * outward)),
                     {"blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": round(s_o - s_tip, 4), "fingertip_z": z_tip, "palm_clear": True,
                      **facts(h, point(s_tip, z_tip))}) for c in PUSH_WRIST_TURNS]

        def leaning(h: float, lean: float):  # push_blade_lean<deg> (the tips lead, the palm recedes)
            z_tip = z0 + h
            so = section_standoff(z_tip, lean)
            if so is None:
                return []
            s_o, s_tip, _half = so
            theta = math.radians(lean)
            a_vec = C.unit(th.tensor([0.0, 0.0, -math.cos(theta)]) + th.tensor([float(d[0]), float(d[1]), 0.0]) * math.sin(theta))
            origin = point(s_o, z_tip) - a_vec * C.FINGERTIP_LENGTH
            return [(f"push_blade_lean{lean:.0f}{'' if c > 0 else '-'}", (origin, C.eef_quat(a_vec, c * outward)),
                     {"blade_half_thickness_m": float(TT.FINGER_PROFILE[0][1]), "stop_clearance_m": clearance, "blade_standoff_m": round(s_o - s_tip, 4),
                      "fingertip_z": float(z_tip), "palm_clear": True, "lean_deg": float(lean), **facts(h, point(s_tip, z_tip))}) for c in PUSH_WRIST_TURNS]

        leans = self.gen("blade_lean_deg", [20.0, 12.0, 8.0])
        leans = [float(v) for v in (leans if isinstance(leans, (list, tuple)) else [leans]) if float(v) > 0]
        pitches = [float(p) for p in (self.gen("push_blade_pitches_deg", None) or [0.0])]
        lean_tip = float(self.gen("blade_lean_tip_above_m", 0.025))
        # (height, class rank = today's order, builder): the fingertip and pad pushes, the leaning blades, the vertical blade
        groups: List[Tuple[float, int, Any]] = []
        groups += [(h, 0, (lambda h=h: tips(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["tip_h"])]
        groups += [(h, 1, (lambda h=h: tip_v(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["tip_v"])]
        groups += [(h, 2, (lambda h=h: pad(h, False))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["pad"])]
        groups += [(h, 3, (lambda h=h: pad(h, True))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["pad"])]
        for i, lean in enumerate(leans):
            groups += [(h, 4 + i, (lambda h=h, lean=lean: leaning(h, lean))) for h in heights(lean_tip)]
        if 0.0 in pitches:
            groups += [(h, 4 + len(leans), (lambda h=h: blade(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["blade"])]
        for _h, _rank, build in sorted(groups, key=lambda g: (g[0], g[1])):
            yield from build()
        # the raised blade (palm above the top: the fingers push its top PALM_FROM_TIP) - the last resort of a target that is not tall
        _ang, _long, short, _c = min_area_rect(G.convex_hull_2d(G.t(t_hull_w)[:, :2].float()))
        aspect = height / max(short, 1e-6)
        if aspect <= float(self.gen("push_raised_max_aspect", PUSH_RAISED_MAX_ASPECT)) + 1e-9 and z1 - z0 > TT.PALM_FROM_TIP:
            z_tip = z1 - TT.PALM_FROM_TIP + float(self.gen("blade_palm_clear_m", 0.025))
            for kind, pose, extras in self._raised_blade(manip, face_point, (z0, z1), d, outward, clearance, z_tip):
                yield kind, pose, dict(extras, contact_height_m=round(z_tip - z0, 4), height_fraction=round((z_tip - z0) / height, 3), push_height_sweep="raised_last",
                                       aspect=round(aspect, 2))

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
        """:meth:`push_gripper_open_on`: the right gripper open ``push_open_opening_m`` per finger (:data:`PUSH_OPEN_OPENING_M`), approaching
        along the push with both fingertips ``stop_clearance_m`` before the face on the line through the target's footprint centre (as
        :meth:`swept_push_candidates`). ``push_open_v-``: closing vertical, the wrist camera up (``push_tip_v-``'s turn) - the tips one above
        the other on the line, the lower one the contact, the hand above it by the half opening; ``push_open_h-`` / ``push_open_h``: closing
        horizontal - the tips side by side at the contact height (the wider contact). A class's eef axis runs from its lowest
        (:data:`PUSH_SWEEP_MIN_HEIGHT_M`: the lower knuckle 4.2 cm below the axis / the gripper body; raised until the open fingers clear the
        support by :data:`PUSH_OPEN_SUPPORT_CLEARANCE_M`) in ``push_height_step_m`` steps up to ``push_max_height_fraction`` x the target's
        height; both tips must land on the face's flat run :data:`BRACE_OPEN_END_MARGIN_M` from its ends, their outermost material within
        :data:`BRACE_OPEN_FACE_DEPTH_TOL_M` of each other (one tip leading would yaw the target). Yielded by the lower fingertip's height
        (``contact_height_m``), ``push_open_v-`` first on a tie; the extras carry ``fingers_opening_m`` (the oracle's opening),
        ``tip_heights_m``, ``eef_height_m``, ``tip_spread_m``, ``hand_clearance_m`` (the hand's lowest point - knuckle, gripper body, camera or
        the open fingers - above the support) and ``contact_point`` (world: the lower tip / the tips' middle on the face)."""
        d = C.unit(G.t(direction).float())
        d2 = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
        side2, out2 = th.tensor([-float(d2[1]), float(d2[0])]), -d2
        side3, out3 = th.tensor([float(side2[0]), float(side2[1]), 0.0]), th.tensor([float(out2[0]), float(out2[1]), 0.0])
        z0, z1 = float(face_z[0]), float(face_z[1])
        height = max(z1 - z0, 1e-6)
        sec = HullSections(t_hull_w)
        centre = sec.centroid(z0 + PUSH_SWEEP_BASE_CUT_M)
        if centre is None:
            centre = polygon_centroid(G.convex_hull_2d(G.t(t_hull_w)[:, :2].float()))
        t0 = float(centre @ side2)
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))
        step = max(float(TT.phase2_gen(self, "push_height_step_m", 0.01)), 0.002)
        h_cap = float(TT.phase2_gen(self, "push_max_height_fraction", 0.5)) * height
        tip_pre = float(self.gen("push_tip_pre_dist_m", 0.05))
        q_open = min(float(C.FINGER_OPEN), max(0.0, float(TT.phase2_gen(self, "push_open_opening_m", PUSH_OPEN_OPENING_M))))
        tip_off = q_open + OPEN_TIP_CENTRE_M  # a fingertip centre from the eef axis
        tip_half = float(TT.FINGER_SIDE_PROFILE[0][1])  # a fingertip's half width across the closing axis
        knob = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "push_gripper_open"}
        down = th.tensor([0.0, 0.0, -1.0])
        h_axis = C.default_closing_axis(d)

        def face_at(z: float, t: float) -> Optional[float]:
            """The outermost material (along ``outward``) within a fingertip's window around (``t``, ``z``); None where the line misses the cut."""
            if not (z0 <= z <= z1) or sec.outermost(z, out2, side2, t) is None:
                return None
            vals = [sec.outermost(zz, out2, side2, t + k * tip_half) for zz in (z - PUSH_SWEEP_WINDOW_M, z, z + PUSH_SWEEP_WINDOW_M) if z0 <= zz <= z1
                    for k in (-1, 0, 1)]
            return max(v for v in vals if v is not None)

        def on_run(z: float, lo: float, hi: float) -> bool:
            run = face_run_at(t_hull_w, z, out3, side3, BRACE_OPEN_FACE_DEPTH_TOL_M)
            return run is not None and run[0] + BRACE_OPEN_END_MARGIN_M <= lo + 1e-9 and hi <= run[1] - BRACE_OPEN_END_MARGIN_M + 1e-9

        def point(s: float, t: float, z: float) -> th.Tensor:
            xy = side2 * float(t) + out2 * float(s)
            return th.tensor([float(xy[0]), float(xy[1]), float(z)])

        def facts(h_lo: float, h_hi: float, h_eef: float, contact: th.Tensor, pose: G.Pose) -> Dict[str, Any]:
            drop = max(TT.hand_drop_below(pose[1]), open_fingers_drop_below(pose[1], q_open))
            return {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, "push_gripper_open": True,
                    "fingers_opening_m": round(q_open, 4), "tip_spread_m": round(2.0 * tip_off, 4), "contact_height_m": round(h_lo, 4),
                    "height_fraction": round(h_lo / height, 3), "tip_heights_m": [round(h_lo, 4), round(h_hi, 4)], "eef_height_m": round(h_eef, 4),
                    "hand_clearance_m": round(float(pose[0][2]) - drop - z0, 4), "contact_point": [round(float(v), 4) for v in contact], **knob}

        def open_v(h_eef: float):  # push_open_v- (closing vertical, the camera up): the tips at the axis -/+ tip_off
            z_ax = z0 + h_eef
            z_lo, z_hi = z_ax - tip_off, z_ax + tip_off
            s_lo, s_hi = face_at(z_lo, t0), face_at(z_hi, t0)
            if s_lo is None or s_hi is None or abs(s_lo - s_hi) > BRACE_OPEN_FACE_DEPTH_TOL_M:
                return []
            if not (on_run(z_lo, t0 - tip_half, t0 + tip_half) and on_run(z_hi, t0 - tip_half, t0 + tip_half)):
                return []
            pose = C.fingertip_eef(point(max(s_lo, s_hi), t0, z_ax), d, down, clearance)  # eef +y = down: the camera (eef -y side) up
            pose = (G.t(pose[0]), G.t(pose[1]))
            return [("push_open_v-", pose, facts(z_lo - z0, z_hi - z0, h_eef, point(s_lo, t0, z_lo), pose))]

        def open_h(h_eef: float):  # push_open_h- / push_open_h (closing horizontal): the tips at the line -/+ tip_off
            z = z0 + h_eef
            if not on_run(z, t0 - tip_off - tip_half, t0 + tip_off + tip_half):
                return []
            vals = [face_at(z, t) for t in (t0 - tip_off, t0 + tip_off)]
            if any(v is None for v in vals) or abs(vals[0] - vals[1]) > BRACE_OPEN_FACE_DEPTH_TOL_M:
                return []
            out = []
            for c in PUSH_WRIST_TURNS:
                pose = C.fingertip_eef(point(max(vals), t0, z), d, c * h_axis, clearance)
                pose = (G.t(pose[0]), G.t(pose[1]))
                ex = facts(h_eef, h_eef, h_eef, point(max(vals), t0, z), pose)
                if ex["hand_clearance_m"] >= PUSH_OPEN_SUPPORT_CLEARANCE_M - 1e-6:  # the camera-down turn hangs its camera 7.4 cm below the axis
                    out.append((f"push_open_h{'' if c > 0 else '-'}", pose, ex))
            return out

        def heights(h_min: float) -> List[float]:
            return [round(h_min + k * step, 4) for k in range(int(math.floor((h_cap - h_min) / step + 1e-6)) + 1)] if h_min <= h_cap + 1e-9 else []

        # the eef axis's lowest per class: the hand's (the lower knuckle / the gripper body), raised until the open lower finger clears the support
        q_v = C.eef_quat(d, down)
        v_min = max(PUSH_SWEEP_MIN_HEIGHT_M["tip_v"], open_fingers_drop_below(q_v, q_open) + PUSH_OPEN_SUPPORT_CLEARANCE_M)
        h_min = max(PUSH_SWEEP_MIN_HEIGHT_M["tip_h"], open_fingers_drop_below(C.eef_quat(d, h_axis), q_open) + PUSH_OPEN_SUPPORT_CLEARANCE_M)
        groups: List[Tuple[float, int, Any]] = [(h - tip_off, 0, (lambda h=h: open_v(h))) for h in heights(round(v_min, 4))]
        groups += [(h, 1, (lambda h=h: open_h(h))) for h in heights(round(h_min, 4))]
        for _h, _rank, build in sorted(groups, key=lambda g: (round(g[0], 4), g[1])):
            yield from build()

    def wide_stop_spread(self, partner_hull_w: th.Tensor, b_point: th.Tensor, outward: th.Tensor, side_h: th.Tensor, s_push: float,
                         spreads: Sequence[float] = (0.09, 0.07, 0.05), end_margin: float = 0.01, face_depth_tol: float = 0.01, band: float = 0.03) -> Optional[float]:
        """Owner 2026-09-19 00:49 rule for a WIDE stop (both fingertips on the braced face, spread along it): the largest
        ``spreads`` value whose two landing points ``s_push ± s/2`` lie on the far face's flat run with ``end_margin`` to its
        ends. The flat run = the side coordinates of the hull points within ``face_depth_tol`` of the face's outermost
        offset in the ``band`` around the brace height — a protrusion at one end leaves only its own points there (a short
        run: no wide stop), a face shorter than the smallest spread + margins gets none either. None = narrow (closed) stop."""
        z_b = float(b_point[2])
        pts = partner_hull_w[(partner_hull_w[:, 2] > z_b - band) & (partner_hull_w[:, 2] < z_b + band)]
        if len(pts) < 2:
            return None
        off = pts[:, :2] @ outward[:2]
        face = pts[off >= float(off.max()) - face_depth_tol]
        s = face[:, :2] @ side_h[:2]
        s_min, s_max = float(s.min()) + end_margin, float(s.max()) - end_margin
        for spread in spreads:
            if s_min <= s_push - spread / 2 and s_push + spread / 2 <= s_max:
                return float(spread)
        return None

    def brace_tip_heights(self, b_point: th.Tensor, face_z: Tuple[float, float]) -> List[float]:
        """The fingertip-front brace heights on the partner's far face, the higher first: the braced point's height raised to
        ``brace_tip_min_above_m`` above the face's bottom (the gripper body clears the table) and ``z0 + 0.075`` (at most 1 cm under the
        top); heights within 1 cm of the top dropped."""
        z0, z1 = face_z
        heights = [max(float(b_point[2]), z0 + float(self.gen("brace_tip_min_above_m", 0.055))), min(z1 - 0.01, z0 + 0.075)]
        return sorted({round(h, 4) for h in heights if h + 0.01 <= z1}, reverse=True)

    def brace_open_rule(self, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: Optional[th.Tensor],
                        push_point: Optional[th.Tensor] = None) -> Optional[Dict[str, Any]]:
        """``generation.brace_open_when_wide``: whether the partner's braced (far) face takes the open fingertips. The face's width = its
        flat run at the higher brace height (:meth:`brace_tip_heights`) in the hull's horizontal cut (:func:`face_run_at`); wide = at least
        the smallest open spread of ``brace_wide_spreads_m`` plus ``BRACE_OPEN_END_MARGIN_M`` at each end. The spread is the largest one
        whose two tips straddle the push line (centred on it) with that margin to the face's ends, as the wide stop's rule
        (:meth:`wide_stop_spread`) - which reads the run off the hull's VERTICES within 3 cm of the brace height: a box hull's lie at its
        top and bottom corners, so a tall box's mid-height band holds few or none (touching#02 pair1013_c8_k8: two of dkhgxn's 56 points on
        its 15.5 cm face, a 9.5 cm run off the push line - no spread, closed tips). None without the key, a partner hull or a brace height;
        else the record (``required`` = wide: the open-finger stop or the draw is rejected)."""
        if not self.brace_open_when_wide() or partner_hull_w is None:
            return None
        heights = self.brace_tip_heights(b_point, face_z)
        if not heights:
            return None
        a = C.unit(G.t(a_face).float())
        outward = -a
        side_h = C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), outward, dim=0))
        spreads = sorted((float(v) for v in self.gen("brace_wide_spreads_m", [0.09, 0.07, 0.05])), reverse=True)
        end = BRACE_OPEN_END_MARGIN_M
        min_width = min(spreads) + 2.0 * end
        run = face_run_at(partner_hull_w, heights[0], outward, side_h, BRACE_OPEN_FACE_DEPTH_TOL_M)
        s_b = float(G.t(b_point)[:2].float() @ side_h[:2])
        s_push = float(G.t(push_point)[:2].float() @ side_h[:2]) if push_point is not None else s_b
        width = 0.0 if run is None else run[1] - run[0]
        rec: Dict[str, Any] = {"rule": "face width at the brace height >= the smallest open spread + 2 end margins", "height_m": round(heights[0], 4),
                               "face_width_m": round(width, 4), "min_width_m": round(min_width, 4), "wide": bool(run is not None and width >= min_width - 1e-9),
                               "push_line_from_face_centre_m": None if run is None else round(s_push - 0.5 * (run[0] + run[1]), 4),
                               "push_line_side_m": round(s_push - s_b, 4), "spread_m": None, "required": False}
        if rec["wide"]:
            rec["required"] = True
            rec["spread_m"] = next((s for s in spreads if run[0] + end <= s_push - s / 2 + 1e-9 and s_push + s / 2 <= run[1] - end + 1e-9), None)
        return rec

    def _open_brace_candidates(self, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: th.Tensor,
                               rule: Dict[str, Any], clearance: float):
        """The open-finger stops of :meth:`brace_open_rule`: the fingertips ``clearance`` before the far face, pre-shaped to the rule's
        spread straddling the push line, the hand horizontal along the brace direction - at each brace height whose cut takes the spread
        (the higher first), both wrist turns (camera up first, ``brace_wide_h-``; ``brace_wide_h``; ``_lo`` at the lower height)."""
        spread = rule.get("spread_m")
        if spread is None:
            return
        a = C.unit(G.t(a_face).float())
        side_h = C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), -a, dim=0))
        heights = self.brace_tip_heights(b_point, face_z)
        tip_pre = float(self.gen("push_tip_pre_dist_m", 0.05))
        shift = float(rule["push_line_side_m"])
        s_b = float(G.t(b_point)[:2].float() @ side_h[:2])
        q_open = max(0.0, min(C.FINGER_OPEN, (spread - CLOSED_TIP_SPREAD_M) / 2))
        h_brace = C.default_closing_axis(a)
        facts = {k: rule[k] for k in ("face_width_m", "min_width_m", "spread_m", "required")}
        for z in heights:
            run = face_run_at(partner_hull_w, z, -a, side_h, BRACE_OPEN_FACE_DEPTH_TOL_M)
            if run is None or not (run[0] + BRACE_OPEN_END_MARGIN_M <= s_b + shift - spread / 2 + 1e-9 and s_b + shift + spread / 2 <= run[1] - BRACE_OPEN_END_MARGIN_M + 1e-9):
                continue
            tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), z]) + side_h * shift
            for c_sign in PUSH_WRIST_TURNS:
                pose = C.fingertip_eef(tip_pt, a, c_sign * h_brace, clearance)  # c = -1: eef +y = -h, the camera (eef -x side) up
                contact = (G.t(pose[0]), G.t(pose[1]))
                kind = f"brace_wide_h{'' if c_sign > 0 else '-'}{'' if z == heights[0] else '_lo'}"
                yield IK.pre_pose(contact, tip_pre), contact, kind, {
                    "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": kind, "tip_front": True, "pitch_deg": 0.0,
                    "pre_dist_m": tip_pre, "contact_z": round(z, 4), "fingers_opening_m": round(q_open, 4), "stop_width_m": spread,
                    "push_line_side_m": round(shift, 4), "open_when_wide": facts, "height_extra_m": requested_height_extra(self),
                    "height_extra_effective_m": round(z - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}

    def touching_brace_candidates(self, manip: Stance, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor,
                                  partner_hull_w: Optional[th.Tensor] = None, push_point: Optional[th.Tensor] = None):
        """Brace poses of the closed left gripper at the partner's far (robot-left) face, in the order to try — the axial
        stop-brace rule instead of the former 12 mm pad-slab at the face point: (1) the pads on the face, pad centre
        ``STOP_CLEARANCE`` outside it (closing axis vertical then horizontal, either wrist turn); (2) the top-down blade
        beside the face (``tabletop.blade_standoff``: eef origin ``finger half thickness over the face height above the
        tips + STOP_CLEARANCE`` outside the face, approach pitched per ``tabletop.approach_pitch_order``, the pads' faces
        parallel to the object face). Thin partners (< ``THIN_PARTNER_HEIGHT``) try the blade first. Yields ``(pre, contact,
        kind, extras)`` with the brace record fields ``blade_half_thickness_m`` / ``stop_clearance_m`` / ``blade_standoff_m`` /
        ``pose_class``."""
        a = C.unit(G.t(a_face))  # into the face
        outward = -a
        z_axis = th.tensor([0.0, 0.0, 1.0])
        side_h = C.unit(th.cross(z_axis, outward, dim=0))
        z0, z1 = face_z
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))
        rule = self.brace_open_rule(b_point, face_z, a_face, partner_hull_w, push_point)
        if rule is not None and rule["required"]:
            # generation.brace_open_when_wide on a wide face: the open-finger stop or nothing (``touching_contacts`` rejects the draw)
            yield from self._open_brace_candidates(b_point, face_z, a_face, partner_hull_w, rule, clearance)
            return

        # owner 2026-09-18 20:48: the FINGERTIP-FRONT brace first, mirrored from the push class that made 0918-200452 - the left
        # hand horizontal along -push direction, the closed tips ``clearance`` before the partner's far face, wrist camera UP
        # (closing = -default: the camera lies on the eef's -x side, the default turn hung it at the table), the contact raised
        # to ``z0 + brace_tip_min_above_m`` so the gripper body (r 4 cm, 9.7 cm behind the tips) clears the table - at the
        # partner's mid height (0.85 on a 0.81 table) it sat on it and every brace pre-pose failed the IK. 5 cm pre.
        if bool(self.gen("brace_tip_first", True)):
            tip_pre = float(self.gen("push_tip_pre_dist_m", 0.05))
            heights = self.brace_tip_heights(b_point, face_z)  # the higher first: the body further off the table
            # WIDE stop first when the rule admits it (``wide_stop_spread``): the fingers pre-shaped so both tips land on the far
            # face straddling the push line - the lever arm against the pivot a point stop cannot resist (0918-224622: the
            # partner yawed 0.8 deg about the closed tips and the pair lost contact). Camera-up wrist turn, same pose family.
            if heights and partner_hull_w is not None and bool(self.gen("brace_wide_stop", True)):
                s_b = float(G.t(b_point)[:2] @ side_h[:2])
                s_push = float(G.t(push_point)[:2] @ side_h[:2]) if push_point is not None else s_b
                spread = self.wide_stop_spread(partner_hull_w, b_point, outward, side_h, s_push, spreads=[float(v) for v in self.gen("brace_wide_spreads_m", [0.09, 0.07, 0.05])])
                if spread is not None:
                    q_open = max(0.0, min(C.FINGER_OPEN, (spread - CLOSED_TIP_SPREAD_M) / 2))  # per-finger joint value for that tip spread
                    tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), heights[0]]) + side_h * (s_push - s_b)
                    h_brace = C.default_closing_axis(a)
                    pose = C.fingertip_eef(tip_pt, a, -h_brace, clearance)  # eef +y = -h: the camera (eef -x side) up
                    contact = (G.t(pose[0]), G.t(pose[1]))
                    yield IK.pre_pose(contact, tip_pre), contact, "brace_wide_h-", {
                        "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": "brace_wide_h-", "tip_front": True, "pitch_deg": 0.0,
                        "pre_dist_m": tip_pre, "contact_z": round(heights[0], 4), "fingers_opening_m": round(q_open, 4), "stop_width_m": spread,
                        "push_line_side_m": round(s_push - s_b, 4), "height_extra_m": requested_height_extra(self),
                        "height_extra_effective_m": round(heights[0] - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}
            for pitch in [float(p) for p in (self.gen("brace_tip_pitches_deg", None) or [0.0, 15.0])]:
                # pitched: the hand comes from above-left pointing down-right; the body 9.7 cm behind the tips rises 9.7·sin(pitch) cm
                th_p = math.radians(pitch)
                a_p = C.unit(a * math.cos(th_p) + th.tensor([0.0, 0.0, -math.sin(th_p)]))
                h_brace = C.default_closing_axis(a_p)
                for z_tip_brace in heights:
                    tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), z_tip_brace])
                    for c_sign in PUSH_WRIST_TURNS:
                        pose = C.fingertip_eef(tip_pt, a_p, c_sign * h_brace, clearance)
                        contact = (G.t(pose[0]), G.t(pose[1]))
                        kind = f"brace_tip_h{'' if c_sign > 0 else '-'}{'' if pitch == 0 else f'_p{pitch:.0f}'}{'' if z_tip_brace == heights[0] else '_lo'}"
                        yield IK.pre_pose(contact, tip_pre), contact, kind, {
                            "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": kind, "tip_front": True, "pitch_deg": pitch,
                            "pre_dist_m": tip_pre, "contact_z": round(z_tip_brace, 4), "height_extra_m": requested_height_extra(self),
                            "height_extra_effective_m": round(z_tip_brace - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}
        pad_lift, pad_why = contact_height_extra(self, "brace", pad_center_z=float(b_point[2]), band_top_z=z1)  # A/B knob
        pad_facts = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}

        def side_contacts():
            point = G.t(b_point) + outward * clearance + th.tensor([0.0, 0.0, pad_lift])
            for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
                name = TT.contact_class(manip, a, c_axis) or f"side_{suffix}"
                for c_sign in (1.0, -1.0):
                    contact = IK.contact_eef(point, a, c_sign * c_axis)
                    yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": clearance, "pose_class": name, "blade_half_thickness_m": None, **pad_facts}

        def blades():
            for pitch in TT.approach_pitch_order(None):
                z_tip = z0 + 0.005
                blade_lift, blade_why = contact_height_extra(self, "blade", fingertip_z=z_tip, band_top_z=z1)
                z_tip += blade_lift  # the fingertips just above the support: the fingers span the face height
                variants = []  # (z_tip, raised)
                palm_clear = float(self.gen("blade_palm_clear_m", 0.025))
                if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP - palm_clear:
                    # the face reaches (within the palm clearance of) the palm line: the 8 cm tissue box vs the 7 cm finger section -
                    # the low blade's knuckles sit 2 mm from the face top (hand gate, 0918-214155). Owner 2026-09-18 21:20: the RAISED
                    # vertical blade first - tips at face top - PALM_FROM_TIP + palm clearance, the finger section on the face's top
                    # part, palm / knuckles / camera above it; the left arm comes from ABOVE the partner, not across the right
                    # forearm at its push goal (the horizontal brace_tip classes pass alone but collide in the two-arm solve)
                    variants.append((z1 - TT.PALM_FROM_TIP + palm_clear, True))
                variants.append((z_tip, False))
                for z_tip_v, raised in variants:
                    so = TT.blade_standoff((z0, z1), z_tip_v, pitch, clearance, side=False)
                    if so is None:
                        continue  # the face lies entirely below the fingertips
                    standoff, half = so
                    a_vec = TT.pitched_approach(pitch, manip.forward)
                    origin = th.tensor([float(b_point[0]), float(b_point[1]), z_tip_v + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
                    name = f"blade_{pitch:.0f}{'_high' if raised else ''}"
                    lift_eff, why_eff = (z_tip_v - (z0 + 0.005), "palm_clear") if raised else (blade_lift, blade_why)
                    for c_sign in PUSH_WRIST_TURNS:  # the wrist camera away from the partner first
                        contact = (origin, C.eef_quat(a_vec, c_sign * outward))
                        yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {
                            "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip_v, "pose_class": name,
                            "palm_clear": raised, "height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(lift_eff, 4), "height_extra_clip_reason": why_eff}

        # blades before the side contacts: the side classes hang the camera down / need the hand across the right forearm
        order = (blades, side_contacts)
        for gen in order:
            yield from gen()

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
