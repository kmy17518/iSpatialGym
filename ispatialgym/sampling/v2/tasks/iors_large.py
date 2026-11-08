"""IORS — ontop / under, large-scale (spec §5.6 / §6.12; oracle doc §6.2 row "IORS ontop / under (large-scale)").

Roles: ``target`` (spawned, library grasps), fixed ``partner`` (bed, sofa, desk, breakfast table, countertop), fixed
``support`` (the **init surface**: a support object within ``max_partner_geodesic`` of the partner, the partner's own
top — under only — or the floor beside the partner; the class is an explicit choice of the variant / bindings, never a
runtime fallback unless the variant sets ``generation.init_support_fallback``: owner 2026-10-12 item 2b), ``floor``. Three recorded stances: **A** init (tabletop Phase 1 at the support + the partner
visible: ≥ 4 of its 8 AABB corners and its centre in frame), **B** manipulation (tabletop Phase 2 at the support, band
from the reach bands), **C** placement (base square to the partner edge nearest the init pose, stand-off {0.35, 0.45,
0.55}, lateral along the edge, trunk from the band; ontop: goal cell = the band cell inside the partner's top face
eroded by half-extent + 0.05 nearest the init pose; under: the floor strip under the partner where an upward ray hits
it ≥ target height + 0.03 above the floor, within 0.45 m of the near edge, clearance ≥ height + 0.17 (§6.12)).

B ↔ C coupling: one grasp for pick and place — (yaw, grasp) drawn uniformly from F_init ∩ F_goal (F_init = the feasible
set at B's focus cell, F_goal = the library grasps screened at the goal cell for the init yaw from C), exact IK at B
(pre-grasp + grasp, init surface ignored for the grasp pose) and at C (pre-place + place with the object attached;
ontop: the partner's top ignored for the place pose only; under: only the floor ignored). Paths A → B (§3.7 / §3.8)
and B → C (≤ ``max_partner_geodesic`` + 1 m, object attached) and the transitions are recorded; the sidecar carries
``stances.placement`` (+ ``goal_cell`` / ``init_surface`` / ``path_length_m`` in its extras and the language block),
``grasps[0]`` with ``pre_place`` / ``place`` / ``place_on``.

Reach bands (B3, ``sampling.reach_bands``): loaded through ``ReachBands.load`` for (support, target), (partner-as-
support, target) and ``--support floor [--under partner]``; on a cache miss they are computed in-process with the
module's ``BandComputer`` (the function the CLI runs) and written to the cache directory (``ISG_REACH_BANDS_COMPUTE=0``
turns the compute-on-miss into ``Reject("missing_band")``). Under v2 only (``ISPATIALGYM_V2_ONLY=1``, i7-floor 2026-11-07):
the v2 reach maps instead (``IORSLarge.v2_band``) - the floor map (``tabletop.floor_map_bands``) for the floor init, the top map
(``tabletop.map_bands``) of the partner's top, each arm's own; a miss logs ``MAP_REQUEST`` and rejects (``missing_map`` under
``ISPATIALGYM_REACH_MAPS_STRICT``), no legacy band is loaded or computed.

Tabletop phases 1a / 1b / 1c / 2 are ``v2/tabletop.py``'s (edge candidates, init trunk, lateral draw, Phase-2 survivors
with the (support, target) band). Floor init: Stance A's trunk is the translate whose optical axis is closest to the
floor cell with the target and the partner in frame (the §4.3 Δ solve has no edge on the floor; documented stand-in).

**Push-under (owner decision 2026‑09‑16 06:50, item 7; ``generation.under_mode`` = ``"push"`` default).** The floor under
a bed / sofa is out of the arm's reach (B3: 0.24–0.64 m ahead at the deep trunks only), so the under goal is reached in
two strokes from stance C (base square to the near footprint edge, facing inward = the push direction, stand-off
0.5–0.7 m, floor trunks): (1) a **drop** of the carried target on a free floor cell ``half + 0.10`` m outside the edge
(object bottom 3 cm above the floor; ``grasps[0].place`` / ``pre_place`` = the release, ``place_kind: "drop"``,
``place_on`` = the floor), (2) a **straight floor-level push** of the closed gripper from behind the target to the
**goal cell** 7 cm inside the footprint edge (the ``under`` predicate needs the centre column only; the cell must clear
target height + 3 cm) — ``contacts.push`` with ``kind: "push_under"``, ``pre`` / ``init`` (start) / ``mid`` / ``goal`` blade
poses (blade stand-off from ``tabletop.blade_standoff`` over the target's face height, approach pitched 0–60° toward
the push), ``direction``, ``distance_m``, ``push_height_m``, ``drop_cell`` / ``goal_cell``. Exact IK on the fly from C for
the drop (floor ignored, object attached) and the push chain (floor + target ignored, partner present). The program
appends ``push → retreat`` after ``place(drop) → retreat``; ``under`` is read in the epilogue; the goal-feasibility check
teleports the target to the goal cell as before.

**The flat hand (owner 2026-10-12 ~19:40, under_large #02 / #03: a briefcase / trash can whose back face rises past the blade's 7 cm
finger section; ``generation.push_under_hand`` = ``"flat"``, default ``"blade"``).** The same contact point and straight push with the hand
turned flat, the closed fingers forward along the push (the pitched blade at 90°): the closing axis lateral or vertical (each twin), the
fingertips on the back face at the lowest height the arm reaches (owner ~20:55: the R1Pro's wrist centre, 0.22 m behind the eef on the
approach axis, never comes below 0.135 m, sim check 17833270 - a ladder of tip heights from the trunk's reach floor up to the target's top,
the first that solves; the hand's lowest collision sphere at least 1 cm above the floor), the eef origin the fingertip length + 5 mm behind
the contact, the pre-pose 4 cm behind the start; the ground slab stays in the IK; no centre-of-mass rule - the contact height is logged and
recorded against it, the goal check decides. The record carries ``pitch_deg`` 90, ``push_hand: "flat"`` and ``palm_clear`` (the oracle's
top-down palm clearance leaves it alone). With ``generation.push_under_palm_contact`` on as well, the flat hand pushes (the palm blade is not
tried). The flat hand needs the drop cell farther ahead than the blade (``push_under_drop_ahead_m`` / ``push_under_stand_offs_m``).

**The pitched hand (owner 2026-10-12 ~20:55 item C; ``generation.push_under_hand`` = ``"pitched_far"``).** The R1Pro's wrist centre sits
0.22 m behind the hand, so the flat hand never holds its tips 5-10 cm up; pitched 60-70° from straight down toward the push
(``push_under_far_pitches_deg``, the widest reach first) the closed hand puts them 8 cm up 0.50-0.90 m ahead of the base at the deepest trunk,
the palm leaning away from the face. Per pitch four wrist turns (the closing axis lateral, the wrist camera up / down, or in the push plane),
the fingertips ``push_far_tip_m`` (8 cm) up - at least the hand's lowest cuRobo sphere + 1 cm above the floor, their leading edge below the
target's centre of mass -, the leading edge touching the rearmost collision point within its height band and a palm test (every sphere of
the hand behind the object's back at its height) in place of the finger-section rule, the pre-pose 3 cm back and 6 cm up from the start (10 cm
back along the fingers put the wrist into the base); the ground slab stays in the IK. Stance C stands farther out (stand-offs 0.80-1.00 m, the
drop cell up to 0.80 m ahead, 0.68 m preferred, the trunks (2.0, 0) / (1.75, 0), the lane 0.30 / 0.25 m to the arm's side; every key
overridable). The record carries ``push_hand: "pitched_far"``, ``pitch_deg``, ``palm_clear`` and the contact / centre-of-mass heights; the palm
key is ignored as with the flat hand.

**The heel-first yaw lock (owner 2026-10-13 ~02:55, under_large#01; ``generation.push_under_yaw`` = ``"heel_first"``, default none).** The push
tries the carried yaws by the contact offset, the largest first; at a B whose F_init holds no heel-first yaw that is the sandal's toe (0.117 m
against the heel's 0.128 m), and a toe-pushed sandal turned 86° on its way in (placed_orientation 1.50 rad). With the lock the B pre-screen,
the chain-first screen and the F draws keep only the yaws at which the heel - the end of the footprint's long axis farther from the root -
points back at the robot within one yaw step (``push_under_yaw_tol_deg``) and the push meets the heel's half; the yaw drawn at B is checked
again before the place IK. One INFO line 'push-under yaw lock heel_first: kept N of M yaws' per C whose yaws it narrows.

**The push that does not turn the object (owner 2026-10-13 ~04:30 item 2, under_large#01).** A long object pushed from one end turns away from
the push: c3_k0's sandal landed 5.3 deg off the line and the heel push grew that to 34.8 deg over 0.29 m. ``generation.push_under_stroke_m``
(default none = one stroke) records a stroke plan (``strokes``: at most ``push_under_strokes_max`` equal strokes, the oracle's back-off /
re-aim / yaw-gain knobs): the oracle measures the object before each stroke and re-aims the blade so it meets the back where the line through
the root along the push does (``execute._push_strokes``; sim check 17838968 in c3_k0's scene: 3 strokes turned the sandal -2.0 deg and left
it 8 mm from the goal, the single stroke +22.3 deg and 49 mm). ``generation.push_under_contact`` (default none = today's blade) tries other
hand forms first: ``"two_point"`` (the open fingertips across the back; one stroke whatever the stroke key says) and / or ``"closed_turned"``
(the closed blade turned 90 deg), today's blade last; the record carries ``push_contact_form`` (+ ``fingers_opening_m`` for the open hand,
which the IK, the goal check and the program check use).

**The arm of the segment (owner 2026-09-27, first-demo v2.2).** Grasp at B → carry → place / drop at C (→ the push-under blade:
``programs.py`` wants the grasp arm) is one single-arm segment; the other arm idles in its recorded posture (A6, re-derived per
working arm). ``tabletop.offered_arms`` offers both arms (the other arm has no role, the target is free); the other arm's bands
are its own cached bands, else the working arm's tabletop / floor bands mirrored in memory (never written, never computed
in-process). B candidates are generated for every offered arm (floor: the lateral offsets mirrored; support: Phase 2 with both
bands), each tagged ``focus_arms`` (the bands that hold the target) and ranked by the arm that reaches better; F_init is
screened per arm (the default arm always, the other one where its band holds the target). At B one coupling runs per arm
with pairs, the arms whose band holds the target first (the default arm first among them): the grasp is
``tabletop.pick_twin`` over the arms whose F_init holds the drawn pair (the
orientation rule decides; a pair the other arm wins waits for that arm's coupling), and C, the carry poses, the place, the
push lane and every record follow the segment arm (right-side laterals mirrored for the left arm). A coupling that fails
after its grasps passed hands the pairs it won to the other arm: a regrasp at B with that arm at the same target pose and
library row (``grasps[0].extras.segment_fallback``). With one arm offered (``generation.other_arm`` false) nothing changes.

**The partner's joints at init (owner 2026-10-12 ~16:05, inside_large_scale#02: the fridge's left door open).**
``generation.partner_joint_init`` = ``{joint: position}`` (default none: the scene file's joints) is set on the partner once the
partner is bound, before anything reads its geometry (the robot re-parked away from it, a settle, the joints re-read:
``template_facts.partner_joint_init``); every attempt applies those joints with the draft (``Draft.object_joints``) and records them
after the settle as a ``joint_positions`` constraint of the TRO (``oracle.run_isg`` holds and restores them) and in
``summary.partner_joints``. The links the request moved (the door) stay an obstacle and nothing else: the parked robot keeps 1 m from
the partner, every recorded path plans around their floor footprints, and the inside mode leaves them out of the stance-C stand-off
and the partner's doors (open or closed) out of the interior raster (:mod:`ispatialgym.sampling.object_joints`).

**The overhang pick (owner 2026-10-13 ~03:15 item 4, inside_large#03's plate; ``generation.pick_overhang_push``, default off).** When
every direct exact IK of a drawn (yaw, grasp) pair fails at B, the same library grasp is tried at the object pushed toward the support
edge B stands at (:meth:`IORSLarge._overhang_pick`, :mod:`ispatialgym.sampling.v2.overhang`): the push distance from the edge (raycast)
and the grasp's finger sweep below the support top (the fingers ``finger_clearance_m`` beyond the edge, the centre of mass
``com_margin_m`` inside), the grasp's exact IK at the pushed pose with the support present, the push chain (the open / closed hand behind
the object's far side, pitched over it, fingertips ``tip_m`` above the support) and a physics settle of the pushed object. The pick
records ``contacts.push_pick`` (``kind: "pick_overhang"``) and ``grasps[0].pick_overhang``; the program pushes, retreats and grasps.

**The v2 overhang rest (i7-overhang 2026-11-07, owner decision: the overhang pick's PRIMARY path, hardback bqvvvx / inside_large_01).** With
``pick_overhang_push`` mode ``only`` and a v2 library + maps root holding an OVERHANG rest of the target's key (``grasping.overhang``: the
object settled with one face ``distance_m`` beyond a slab edge, grasps generated and validated there), the key becomes ``<key>@<overhang
tag>`` (``tabletop.overhang_key``): the grasps and the rest are that entry's, the stance search reads the OVERHANG map (``tabletop.
overhang_map_bands``: the init cells behind its feasible post-push cells), the pairs at B are the overhanging face flush with B's edge
(``overhang.flush_yaw``) whose grasp the map holds feasible at the post-push cell from B or B backed off (:meth:`_v2_overhang_map_ok`, the
map-hit step), and the push leaves the object exactly at the library rest (``overhang.v2_rest_push``: the face ``distance_m`` beyond the
edge - one rung, inside the exact finger-clearance / centre-of-mass range). Every check after the map stays exact (the grasp IK at the
pushed pose, the push chain, the settle, the program check). A map miss rejects ``missing_map`` with a MAP_REQUEST line; there is no
legacy band and no legacy grasp.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.sampling import alias_memory as AM
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import object_joints as OJ
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import contact_poses as CPZ  # i7-contact 2026-11-07: the shared push / brace pose rules
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.grasp_library import MEMBER_SEP as GL_MEMBER_SEP
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_large_scale": "ontop",
    "inter_object_relation_satisfaction-under_large_scale": "under",
    "inter_object_relation_satisfaction-inside_large_scale": "inside",
    "inter_object_relation_satisfaction-touching_large_scale": "touching",
    "inter_object_relation_satisfaction-nextto_large_scale": "nextto",
}
V1_DEFINITION_ID = {"ontop": 0, "under": 1}
#: relations registered ahead of their modes (owner 2026-10-12): relation -> its hook, which ``bind_scene_roles`` (the first stage of the
#: template setup) calls before any of the ontop / under flow; the hook returns the relation's mode class (:meth:`IORSLarge._adopt_mode`)
#: and rejects until the mode lands
MODE_HOOKS = {"inside": "_sample_inside", "touching": "_sample_touching", "nextto": "_sample_nextto"}
#: scene categories a fixed partner may bind to (B4 ``PARTNER_CATEGORIES``)
PARTNER_CATEGORIES = ("bed", "sofa", "couch", "desk", "breakfast_table", "countertop")
SUPPORT_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar", "nightstand", "console_table",
                      "side_table", "shelf", "bench")
INIT_SURFACE_CLASSES = ("support", "partner", "floor")
SCENE_INVENTORY_ROOT = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
STAND_OFFS = (0.35, 0.45, 0.55)
SUPPORT_MARGIN = 0.02
# ---- push-under (owner decision 2026‑09‑16 06:50, item 7): the under goal is reached by a drop just outside the partner's
# footprint edge and a straight floor-level push of the target under the edge ------------------------------------------------
PUSH_UNDER_INSIDE_M = 0.07          # goal = the target centre this far inside the footprint edge (5–10 cm; ``under`` needs the centre column)
PUSH_UNDER_DROP_EXTRA_M = 0.10      # drop centre = half extent along the push + this outside the edge
PUSH_UNDER_DROP_HEIGHT_M = 0.15     # object bottom above the floor at the recorded drop pose (owner 2026-10-10: the oracle releases every free drop
                                    # at once after bringing it straight down to 2 cm above the floor; from 0.25 m the sandal's wrist flipped 5-9 cm
                                    # down and the fall tumbled it, from 0.15 m at the (0.95, -1.2) drop trunk the descent comes the full way: landed
                                    # 1.2 mm aside; z < 0.08 m stays out of the arm's reach even at translate 0.95)
#: drop-stance trunks tried in turn (``generation.drop_trunks`` overrides): the deeper (0.95, -1.2) first, from which the 0.15 m drop comes down
#: to the floor, then the moderate ones (2026-09-20: (0.5, -0.6) cannot reach the sandal's drop beside the breakfast table)
PUSH_UNDER_DROP_TRUNKS = ((0.95, -1.2), (0.5, -0.6), (0.5, -0.9), (0.75, -0.6), (0.25, -0.6), (0.75, -0.9))
PUSH_UNDER_STAND_OFFS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)  # base stand-off from the edge: the drop cell 0.25–0.6 m ahead (B3: the floor is reachable 0.24–0.64 m ahead)
PUSH_UNDER_DROP_AHEAD_M = (0.25, 0.60)  # the frontal stances' drop cell this far ahead of the base (``generation.push_under_drop_ahead_m``)
PUSH_UNDER_CLEARANCE_M = 0.005      # blade clearance to the target's back face at the push start
#: ``generation.push_under_hand`` (owner 2026-10-12 ~19:40, under_large #02 / #03): ``"blade"`` (default) the closed gripper's pitched
#: floor blade, ``"flat"`` the flat hand - the closed fingers forward along the push (:meth:`IORSLarge._push_under_flat_chain`),
#: ``"pitched_far"`` the hand pitched 60-70 deg toward the push from stances farther out (owner ~20:55 item C,
#: :meth:`IORSLarge._push_under_pitched_chain`)
PUSH_UNDER_HANDS = ("blade", "flat", "pitched_far")
#: ``generation.push_under_yaw`` (owner 2026-10-13 ~02:55, under_large#01): default none = every carried yaw, in the
#: :meth:`IORSLarge.push_yaw_rank` order; ``"heel_first"`` = only the yaws at which the push meets the target's heel
#: (:func:`footprint_heel`, :meth:`IORSLarge.push_yaw_kept`)
PUSH_UNDER_YAW_LOCKS = ("heel_first",)
#: heel_first: the heel must lie at least this much farther from the root than the toe (the sandal chgcrm: 0.128 vs 0.117 m)
PUSH_HEEL_MIN_ASYM_M = 0.005
#: heel_first: the push's contact at least this fraction of the heel's extent behind the root (the heel's half, not the strap)
PUSH_HEEL_CONTACT_FRAC = 0.5
#: ``generation.push_under_contact`` (owner 2026-10-13 ~04:30 item 2): the blade hand's contact forms. ``"blade"`` = today's (closed, the
#: closing axis along the push: a 14 mm face); ``"two_point"`` = the gripper open, the closing axis across the push, both fingertips on the
#: target's back; ``"closed_turned"`` = the closed blade turned 90 deg (both fingers side by side: a 16.8 mm face)
PUSH_CONTACT_FORMS = ("two_point", "closed_turned", "blade")
#: two_point: the fingertips' gap = this fraction of the target's width across the push ``PUSH_TWO_POINT_DEPTH_M`` ahead of its rearmost point
#: in the blade's band, within ``PUSH_TWO_POINT_OPENING_M`` (``generation.push_under_two_point_opening_m`` sets the gap)
PUSH_TWO_POINT_WIDTH_FRAC = 0.6
PUSH_TWO_POINT_DEPTH_M = 0.02
PUSH_TWO_POINT_OPENING_M = (0.025, 0.08)
#: ``generation.push_under_stroke_m`` (owner 2026-10-13 ~04:30 item 2): the push in at most this many strokes (``push_under_strokes_max``)
PUSH_STROKES_MAX = 3
#: the flat hand's lowest collision sphere this far above the floor (``generation.push_flat_floor_clearance_m``)
PUSH_FLAT_FLOOR_CLEARANCE_M = 0.01
#: the flat hand's fingertip ladder (owner 2026-10-12 ~20:55): the lowest tip height the arm reaches per floor trunk - offline FK of the
#: right arm (URDF limits, the cuRobo spheres against the floor, the base and the torso; 4M configurations per trunk, sim check 17833270
#: found nothing at 5-10 cm): a collision-free horizontal hand on the frontal lane holds its tips no lower, the wrist centre sitting
#: ``FLAT_WRIST_FROM_EEF_M`` behind the eef on the approach axis. ``generation.push_flat_tip_min_m`` sets one floor for every trunk
FLAT_TIP_FLOOR_M = {(2.0, 0.0): 0.125, (1.75, 0.0): 0.15, (0.95, -1.2): 0.205, (0.95, -0.9): 0.215, (0.75, -1.2): 0.265, (0.75, -0.9): 0.30}
#: the rungs: ``generation.push_flat_tip_step_m`` apart up to ``push_flat_tip_max_m`` and ``push_flat_top_margin_m`` below the target's
#: top; at most ``push_flat_rungs_max`` rungs per stance and carried yaw are IK-solved
PUSH_FLAT_TIP_STEP_M = 0.02
PUSH_FLAT_TIP_MAX_M = 0.25
PUSH_FLAT_TOP_MARGIN_M = 0.02
PUSH_FLAT_RUNGS_MAX = 3
#: the flat hand's pre-pose this far behind the start along the push (``generation.push_flat_pre_dist_m``; the oracle's lowered pre of a
#: straight push, ``push_pre_dist_m``): the hand's wrist sits 0.22 m behind its fingertips, and a 10 cm back-off put it in the base
PUSH_FLAT_PRE_DIST_M = 0.04
#: the wrist centre (where the joint 5 / 6 / 7 axes meet) behind the eef origin on the approach axis (``conventions.twin_arm_q``)
FLAT_WRIST_FROM_EEF_M = 0.22065
#: the base's cuRobo spheres in base_link (``r1pro_description_curobo_default.yaml``) and the wrist's (``arm_link6``, r 0.04, at the wrist
#: centre) with both 2 mm buffers: a flat hand whose wrist sphere meets one collides with the base, whatever the arm does
FLAT_BASE_SPHERES = (((0.1, -0.205, 0.165), 0.15), ((0.1, 0.0, 0.165), 0.15), ((0.1, 0.205, 0.165), 0.15), ((-0.08, -0.155, 0.175), 0.165),
                     ((-0.08, 0.155, 0.175), 0.165), ((-0.23, 0.0, 0.185), 0.18))
FLAT_WRIST_SPHERE_M = 0.044
#: the shoulder (``<arm>_arm_link1`` origin) -> wrist centre reach: offline FK max 0.649 m + 2 cm
FLAT_WRIST_REACH_M = 0.67
# ---- the pitched hand (``push_under_hand`` = ``"pitched_far"``): offline IK of the R1Pro right arm (URDF + cuRobo spheres, trunk (2.0, 0),
# the lane 0.25 m to the arm's side): pitch 60 holds tips 8 cm up 0.50-0.90 m ahead (10 cm: 0.49-0.94), 65 0.51-0.88, 70 0.53-0.83;
# (0.95, -1.2) nothing, the centred lane nothing; the sim check's chains: pitch 60 from (2.0, 0) 0.51-0.91 m ahead, from (1.75, 0) 0.36-0.83
#: the pitch ladder, deg from straight down toward the push (``generation.push_under_far_pitches_deg``): the widest reach first
PUSH_FAR_PITCHES_DEG = (60.0, 65.0, 70.0)
#: the fingertips' height above the floor (``generation.push_far_tip_m``), raised to the hand's lowest sphere + the floor clearance and
#: lowered to keep the contact below the target's centre of mass
PUSH_FAR_TIP_M = 0.08
#: the pitched hand's lowest collision sphere at least this far above the floor (``generation.push_far_floor_clearance_m``)
PUSH_FAR_FLOOR_CLEARANCE_M = 0.01
#: the pitched hand's pre-pose this far back along the push and up from the start (``generation.push_far_pre_m``): 10 cm back along the
#: finger axis (``conventions.pre_pose``) put the wrist into the base's spheres at every stance that reaches; 6 cm up keeps the reach, and
#: 3 cm back leaves the fingertips' edge 3.5 cm from the face for the oracle's transit with the object present
PUSH_FAR_PRE_M = (0.03, 0.06)
#: the pitched hand's stance C: stand-offs (``generation.push_under_stand_offs_m`` overrides), the drop cell's window and preferred
#: distance ahead of the base (``generation.push_under_drop_ahead_m`` / ``push_under_drop_ahead_pref_m``), the trunks
#: (``generation.floor_trunks``) and the lane's offset to the arm's side (``generation.frontal_lane_offset_m``): the push start must lie
#: >= 0.50 m ahead, its goal <= 0.90 m (sim check 17834116 / 17834117: chains at stand-offs 0.85-0.975 from (2.0, 0), 0.80-0.85 from
#: (1.75, 0); the centred lane reaches nothing, 0.30 m more than 0.25)
PUSH_FAR_STAND_OFFS = (0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0)
PUSH_FAR_DROP_AHEAD_M = (0.25, 0.80)
PUSH_FAR_DROP_AHEAD_PREF_M = 0.68
PUSH_FAR_TRUNKS = ((2.0, 0.0), (1.75, 0.0))
PUSH_FAR_LANE_OFFSETS = (0.3, 0.25)
#: trunks that reach the floor (B3 floor bands, A6 recompute: cells only at translate ≥ 0.75, tilt ≤ −0.9) + the ground half of the
#: JoyLo curve (W1, owner 2026-09-20 22:15: (2.0, 0) puts the shoulder 8 cm lower than (0.95, -1.2) and reaches the floor blade
#: 0.40-0.85 m ahead instead of 0.40-0.70; ``eef_reach_volume.GROUND_TRUNKS``)
FLOOR_TRUNKS = ((0.75, -0.9), (0.75, -1.2), (0.95, -0.9), (0.95, -1.2)) + tuple(tuple(t) for t in RB.GROUND_TRUNKS)
#: per-attempt time budget (owner 2026‑09‑16 22:55 item 4; as attachment's 094cfe3): run 0916‑190702 spent > 2 h in one push-under
#: attempt (6 B candidates × up to 56 C candidates × the F_goal feasible sets) and hit the 7200 s run timeout with nothing
#: written and no rejection recorded — the attempt now stops with ``Reject("time_budget")`` naming the stage it reached
#: (``generation.attempt_budget_s`` overrides). 900 -> 1500 s (2026-09-20): the F_init pre-screen (8 B x 16 yaws) and the
#: carry-leg checks (departure sweep, path, arrival sweep with the object attached) add ~3 min per attempt
ATTEMPT_BUDGET_S = 1500.0
#: physics steps after ``generation.partner_joint_init`` set the partner's joints (+ the settle's own 10 and one full step)
PARTNER_JOINT_SETTLE_STEPS = 30
#: growth of the moved links' floor footprints on the path planner's local map
PARTNER_MOVED_FOOTPRINT_M = 0.02


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class PushUnderRecord(ContactRecord):
    """``contacts.push`` of the push-under (F6 ``PushContact`` + extras): ``kind: "push_under"``, straight floor-level
    push ``init → mid → goal`` of the closed gripper (blade behind the target along the push direction), from the
    placement stance; ``push_height_m``, ``pitch_deg``, ``blade_standoff_m``, ``half_extent_m`` (target half extent along
    the push), ``drop_cell`` / ``goal_cell`` (target centres, world xy); the flat hand's ``push_hand: "flat"``, ``palm_clear``,
    ``wrist_turn``, ``palm_drop_m``, ``fingertip_band_m``, ``contact_height_m``, ``target_com_height_m``; the pitched hand's ``push_hand:
    "pitched_far"``, ``palm_clear``, ``wrist_turn``, ``contact_height_m``, ``target_com_height_m``, ``palm_gap_m``, ``hand_low_m``,
    ``fingertip_band_m``; every hand's ``chain_arm_q`` / ``chain_branch`` (the sampled chain's configurations, :func:`push_chain_seed`); with
    ``generation.push_under_stroke_m`` the stroke plan ``strokes``, with ``generation.push_under_contact`` ``push_contact_form`` (+
    ``contact_strips_m`` / ``face_half_m``, and ``fingers_opening_m`` / ``contact_width_m`` for ``"two_point"``)."""

    def __init__(self, arm: str, pre, start, mid, goal, direction, distance_m: float, object: str, stance: str = "placement", **extras):
        super().__init__("push", arm, pre, start, mid, goal, direction, distance_m, None, object, stance)
        self.extras: Dict[str, Any] = dict(extras)

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d["kind"] = "push_under"
        for k, v in self.extras.items():
            if k not in d:
                d[k] = _jsonable(v)
        return d


def push_under_palm_flags(chain: Dict[str, Any]) -> Dict[str, Any]:
    """The push-under record's flags of a chain pushed by the palm (``palm_contact`` from ``_push_under_chain``): ``palm_contact`` and
    ``palm_clear`` - the oracle raises a top-down blade whose palm would meet the object's top unless its record has ``palm_clear``,
    and this push runs as recorded."""
    return {"palm_contact": True, "palm_clear": True} if chain.get("palm_contact") else {}


def push_record_fingers(extras: Dict[str, Any]) -> float:
    """The finger joints a push-under record's chain is solved with: a two-point push's ``fingers_opening_m`` (``generation.push_under_contact``,
    owner 2026-10-13 ~04:30 item 2), else the closed gripper."""
    if extras.get("push_contact_form") == "two_point" and extras.get("fingers_opening_m") is not None:
        return float(extras["fingers_opening_m"])
    return C.FINGER_CLOSED


def push_chain_seed(kit, chosen, arm: str) -> Dict[str, Any]:
    """The push-under record's seed of the chain the orientation rule chose (``orient.Scored``), which the goal check re-solves the chain
    to the settled goal from (owner 2026-10-13 ~01:05 item 1): ``chain_arm_q`` - ``arm``'s 7 joint values at the chain's poses ``[pre, start,
    mid, goal]`` as solved (the pre-pose's is the configuration the tracked chain leaves from) - and ``chain_branch``: ``"map"`` (the
    wrist-flipped configuration of the twin turn's solution) or ``"ik"`` (solved as proposed). Empty when a configuration is missing."""
    qs = list(getattr(chosen, "qs", None) or [])
    if not qs or len(qs) != len(chosen.proposal.chain) or any(q is None for q in qs):
        return {}
    idx = kit.robot.arm_control_idx[arm]
    return {"chain_arm_q": [[round(float(v), 5) for v in G.t(q).reshape(-1)[idx].tolist()] for q in qs], "chain_branch": str(chosen.proposal.source)}


TARGET_JITTER = math.radians(22.5)
FOCUS_RADIUS = 0.05
YAWS = [k * math.pi / 4 for k in range(8)]


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


#: §5.6 / owner decision A5 (2026‑09‑16): the variant-level selection order of the init surface, applied by ``ground_isg``
#: through fallback variants (primary → fallback when the primary yields nothing); a module never switches class at run time
SELECTION_ORDER = {"under": ("support", "partner", "floor"), "ontop": ("support", "floor")}


def init_surface_class(relation: str, support_binding: Any, partner_names: Sequence[str], variant_support_token: Optional[str] = None,
                       variant_partner_token: Optional[str] = None) -> str:
    """§5.6 init-surface class fixed by the variant (A5: no runtime switching): ``"floor"`` (the binding / variant token
    says floor, or the binding is a room dict), ``"partner"`` (the support binding names the partner, or the variant
    repeats the partner token ``…-<partner>-<partner>``; under only) or ``"support"`` (a support object). ``ValueError``
    for the ontop + partner combination (the goal would hold at init)."""
    names = [n for n in support_binding if isinstance(n, str)] if isinstance(support_binding, list) else []
    token = (variant_support_token or "").lower()
    ptoken = (variant_partner_token or "").lower()
    if names == ["floor"] or token in ("floor", "floors") or (isinstance(support_binding, dict) and (support_binding.get("floor") or support_binding.get("room"))):
        return "floor"
    own_top = (names and all(n in set(partner_names) for n in names)) or (not names and token and ptoken and token == ptoken)
    if own_top:
        if relation == "ontop":
            raise ValueError("ontop with the partner's own top as the init surface: the goal would hold at init")
        return "partner"
    if not names and not isinstance(support_binding, dict) and token and token in {c for c in PARTNER_CATEGORIES} and relation == "under":
        return "partner"
    return "support"


def variant_tokens(variant: str, subtask: str) -> List[str]:
    """The ``{target}-{partner}-{support}`` tokens of ``<subtask>-{target}-{partner}-{support}`` ([] without tokens)."""
    rest = variant[len(subtask):].strip("-") if variant.startswith(subtask) else ""
    return [p for p in rest.split("-") if p]


def variant_support_token(variant: str, subtask: str) -> Optional[str]:
    """The ``{support}`` token of ``<subtask>-{target}-{partner}-{support}`` (None when the variant has no tokens)."""
    parts = variant_tokens(variant, subtask)
    return parts[-1] if len(parts) >= 3 else None


def variant_partner_token(variant: str, subtask: str) -> Optional[str]:
    """The ``{partner}`` token (None when the variant has fewer than three tokens)."""
    parts = variant_tokens(variant, subtask)
    return parts[-2] if len(parts) >= 3 else None


def nearest_edge(edges: Sequence[EdgeFrame], xy) -> EdgeFrame:
    """The partner edge whose midpoint is nearest to a world xy (§5.6: the edge nearest the target's init pose)."""
    p = th.as_tensor([float(xy[0]), float(xy[1])])
    return min(edges, key=lambda e: float(th.linalg.norm(e.mid - p)))


def aabb_edges(lo, hi, overhang_from_top: Optional[Dict[str, float]] = None) -> List[EdgeFrame]:
    """The four AABB edges of a partner as EdgeFrames (inward normals), named ``+x / -x / +y / -y``."""
    lo, hi = th.as_tensor(lo, dtype=th.float32)[:2], th.as_tensor(hi, dtype=th.float32)[:2]
    cx, cy = float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2
    out = []
    for name, mid, outward, length in (("+x", (float(hi[0]), cy), (1.0, 0.0), float(hi[1] - lo[1])), ("-x", (float(lo[0]), cy), (-1.0, 0.0), float(hi[1] - lo[1])),
                                       ("+y", (cx, float(hi[1])), (0.0, 1.0), float(hi[0] - lo[0])), ("-y", (cx, float(lo[1])), (0.0, -1.0), float(hi[0] - lo[0]))):
        out.append(EdgeFrame.from_edge(name, th.tensor(mid), th.tensor(outward), length, float((overhang_from_top or {}).get(name, 0.0))))
    return out


def edge_overhang(top_edge_mid, outward, aabb_lo, aabb_hi) -> float:
    """AABB overhang beyond a top-raster edge along its outward normal (§4.2 / §5.6: ≥ 5 cm counts, else 0)."""
    corners = G.aabb_corners(th.as_tensor(aabb_lo, dtype=th.float32), th.as_tensor(aabb_hi, dtype=th.float32))[:, :2]
    n = th.as_tensor(outward, dtype=th.float32)[:2]
    over = float((corners @ n).max()) - float(th.dot(th.as_tensor(top_edge_mid, dtype=th.float32)[:2], n))
    return over if over >= 0.05 else 0.0


def erode_cells(cells: th.Tensor, boundary_xy: th.Tensor, margin: float) -> th.Tensor:
    """Cells (N, 2) farther than ``margin`` from every point of a boundary polyline / raster (the top eroded by the
    target's half-extent + 0.05)."""
    if len(cells) == 0 or len(boundary_xy) == 0:
        return cells
    d = th.cdist(cells.float(), boundary_xy.float())
    return cells[d.min(dim=1).values > margin]


def under_clearance_cells(grid: Dict[str, Any], min_clearance: float) -> th.Tensor:
    """World xy of the B4 under-clearance raster cells (``partners/<name>.json`` ``under_clearance``) whose clearance
    ≥ ``min_clearance``."""
    res = float(grid["resolution"])
    ox, oy = (float(v) for v in grid["origin"])
    g = grid["grid"]
    out = []
    for i, row in enumerate(g):
        for j, c in enumerate(row):
            if c is not None and float(c) >= min_clearance:
                out.append((ox + (i + 0.5) * res, oy + (j + 0.5) * res))
    return th.tensor(out, dtype=th.float32) if out else th.zeros(0, 2)


def under_admissible(clearance_at_goal: float, target_height: float, extra: float = 0.17) -> bool:
    """§6.12 under admissibility: clearance ≥ target height + 0.12 (gripper below the wrist) + 0.05."""
    return float(clearance_at_goal) >= float(target_height) + float(extra)


def band_back_offset(pts: th.Tensor, direction_xy, z_lo: float, z_hi: float, section: bool = False) -> Optional[float]:
    """Distance behind the origin (against ``direction_xy``) of the rearmost of the posed hull points ``pts`` (N, 3) whose height above
    their lowest point lies in ``[z_lo, z_hi]``; ``section``: the hull's cross-sections at ``z_lo`` / ``z_hi`` as well - every segment
    between two points on either side of the plane (inside the convex hull) cut there - so a band narrower than the hull's vertex spacing
    (a box's corners lie at its bottom and top only) still meets the faces it crosses. None when nothing lies in the band."""
    z = pts[:, 2] - pts[:, 2].min()
    d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])], dtype=pts.dtype)
    d = d / th.linalg.norm(d)
    sel = (z >= float(z_lo)) & (z <= float(z_hi))
    back = [-(pts[sel][:, :2] @ d)] if bool(sel.any()) else []
    if section:
        for zp in (float(z_lo), float(z_hi)):
            below, above = z < zp, z > zp
            if bool(below.any()) and bool(above.any()):
                zb, za = z[below], z[above]
                t = (zp - zb[:, None]) / (za[None, :] - zb[:, None])
                xy = pts[below][:, None, :2] + t[..., None] * (pts[above][None, :, :2] - pts[below][:, None, :2])
                back.append(-(xy.reshape(-1, 2) @ d))
    if not back:
        return None
    return float(th.cat(back).max())


def footprint_heel(pts: th.Tensor, step_deg: float = 1.0) -> Tuple[th.Tensor, float, float]:
    """The heel of a footprint (``generation.push_under_yaw`` = ``"heel_first"``): ``(h, heel, toe)`` - the unit xy direction from the root
    to the heel and how far the heel / the toe lie from the root along it. The long axis runs across the narrowest width of the points
    ``pts`` (N, 2+; xy relative to the root, searched in ``step_deg`` steps); its end farther from the root is the heel (the sandal chgcrm:
    0.128 m behind its root, the toe 0.117 m ahead)."""
    xy = th.as_tensor(pts, dtype=th.float32)[:, :2]
    ang = th.arange(0.0, 180.0, float(step_deg)) * (math.pi / 180.0)
    n = th.stack([th.cos(ang), th.sin(ang)], dim=1)
    proj = xy @ n.T
    k = int(th.argmin(proj.max(dim=0).values - proj.min(dim=0).values))
    u = th.stack([-n[k, 1], n[k, 0]])
    p = xy @ u
    ahead, behind = float(p.max()), float(-p.min())
    return (u, ahead, behind) if ahead >= behind else (-u, behind, ahead)


def push_contact_strips(form: str, opening: float = 0.0) -> Tuple[Tuple[Tuple[float, float], ...], float]:
    """The parts of a contact form's hand that meet the target's back (``generation.push_under_contact``), ``((lateral centre from the eef
    origin, half width), ...)``, and how far their faces stand ahead of the eef origin along the push (the finger profiles at the tip): today's
    blade one 14 mm part 8.4 mm ahead, the turned blade one 16.8 mm part 7 mm ahead, a two-point hand its two fingertips (8.4 mm thick each,
    their pads ``opening`` apart) 7 mm ahead."""
    t, w = float(TT.FINGER_PROFILE[0][1]), float(TT.FINGER_SIDE_PROFILE[0][1])
    if form == "two_point":
        c = 0.5 * float(opening) + 0.5 * t
        return ((-c, 0.5 * t), (c, 0.5 * t)), w
    if form == "closed_turned":
        return ((0.0, t),), w
    return ((0.0, w),), t


def finger_joint_for_opening(opening: float) -> float:
    """The per-finger joint value (``motions.gripper_opening``'s, the IK's ``fingers``) that holds the pads ``opening`` apart: linear up to the
    measured full opening (``GRIPPER_MAX_OPENING_M``, 96.9 mm) at ``FINGER_OPEN`` (0.05 per finger)."""
    inset = 2.0 * C.FINGER_OPEN - C.GRIPPER_MAX_OPENING_M
    return max(C.FINGER_CLOSED, min(C.FINGER_OPEN, 0.5 * (float(opening) + inset)))


def push_form_contact(form: str, pts: th.Tensor, d2, z_lo: float, z_hi: float, opening: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Where a contact form's hand meets the target pushed along ``d2`` (``generation.push_under_contact``): ``pts`` = the posed hull (N, 3; the
    root at the origin), the hand's band ``[z_lo, z_hi]`` above the target's lowest point. ``{"form", "off_m" (how far behind the root the
    hand's first part meets the back: the rearmost point of the band's footprint within its parts, :func:`execute.push_back_along`),
    "strips_m", "face_half_m"}`` + for ``"two_point"`` ``"opening_m"`` (``opening``, else :data:`PUSH_TWO_POINT_WIDTH_FRAC` of the footprint's
    width :data:`PUSH_TWO_POINT_DEPTH_M` ahead of its rearmost point, within :data:`PUSH_TWO_POINT_OPENING_M`) and ``"fingers_m"``
    (:func:`finger_joint_for_opening`). None when no part meets the band's footprint."""
    from ispatialgym.oracle.execute import push_back_along

    P = th.as_tensor(pts, dtype=th.float32).reshape(-1, 3).clone()
    P[:, 2] = P[:, 2] - P[:, 2].min()
    d = th.as_tensor([float(d2[0]), float(d2[1])], dtype=th.float32)
    d = d / th.linalg.norm(d)
    origin = th.zeros(2)
    if form == "two_point" and opening is None:
        rear = push_back_along(P, origin, d, 0.0, ((0.0, 10.0),), z_lo, z_hi)
        if rear is None:
            return None
        band = P[(P[:, 2] >= float(z_lo)) & (P[:, 2] <= float(z_hi))][:, :2]
        n = th.stack([-d[1], d[0]])
        hull = G.convex_hull_2d(th.stack([band @ d, band @ n], dim=1))
        cut = rear + PUSH_TWO_POINT_DEPTH_M
        lats = [float(v[1]) for v in hull if float(v[0]) <= cut]
        for a, b in zip(hull, th.roll(hull, -1, dims=0)):
            da, db = float(a[0]) - cut, float(b[0]) - cut
            if da * db < 0.0:
                lats.append(float(a[1]) + da / (da - db) * (float(b[1]) - float(a[1])))
        width = (max(lats) - min(lats)) if lats else 0.0
        lo, hi = PUSH_TWO_POINT_OPENING_M
        opening = max(lo, min(hi, PUSH_TWO_POINT_WIDTH_FRAC * width))
    strips, face = push_contact_strips(form, float(opening or 0.0))
    along = push_back_along(P, origin, d, 0.0, strips, z_lo, z_hi)
    if along is None:
        return None
    out: Dict[str, Any] = {"form": form, "off_m": -float(along), "strips_m": [[round(c, 5), round(hw, 5)] for c, hw in strips], "face_half_m": face}
    if form == "two_point":
        out.update({"opening_m": float(opening), "fingers_m": finger_joint_for_opening(float(opening))})
    return out


def push_stroke_plan(distance_m: float, stroke_m: float, n_max: int = PUSH_STROKES_MAX, **knobs: float) -> Dict[str, Any]:
    """The push record's stroke plan (``generation.push_under_stroke_m``, owner 2026-10-13 ~04:30 item 2): ``n`` = ``ceil(distance / stroke)``
    equal strokes, at most ``n_max``; ``ends_m`` the object's progress along the push after each (cumulative); ``knobs`` (``backoff_m``,
    ``reaim_max_m``, ``yaw_gain_m_per_rad``, ``yaw_deadband_rad``) as given - the oracle's ``execute.push_stroke_plan`` reads it."""
    n = max(1, min(int(n_max), int(math.ceil(float(distance_m) / max(float(stroke_m), 1e-6) - 1e-9))))
    return {"n": n, "ends_m": [round(float(distance_m) * k / n, 5) for k in range(1, n + 1)], **{k: round(float(v), 5) for k, v in knobs.items()}}


def pitched_palm_gap(pts: th.Tensor, direction_xy, eef_pos, eef_quat) -> Tuple[float, str]:
    """The pitched hand's palm test (owner 2026-10-12 ~20:55 item C): the smallest gap along the push between a sphere of the hand
    (``tabletop.hand_spheres_eef``: gripper body, knuckles, wrist camera, closed fingers) and the object's back at that sphere's height -
    the rearmost of the posed hull ``pts`` (N, 3; the object's root at the origin, heights from its lowest point) within the sphere's
    height band, cross-sections counted (:func:`band_back_offset`) - for the eef pose ``eef_pos`` (relative to the root, z above the object's
    bottom) / ``eef_quat``; and the part it belongs to. The hand leaning away from a vertical face keeps every sphere behind it; a face that
    leans back over the fingertips meets the knuckles or the camera first. A sphere above or below the object is clear. ``(gap, part)``;
    negative: that part reaches the object."""
    d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1]), 0.0], dtype=th.float32)
    d = d / th.linalg.norm(d)
    p, R = G.t(eef_pos).float(), C.quat_to_mat(G.t(eef_quat)).float()
    best: Tuple[float, str] = (math.inf, "")
    for part, c, r in TT.hand_spheres_eef():
        w = p + R @ th.as_tensor(c, dtype=th.float32)
        back = band_back_offset(pts, d[:2], float(w[2]) - float(r), float(w[2]) + float(r), section=True)
        if back is None:
            continue
        gap = -back - (float(w @ d) + float(r))
        if gap < best[0]:
            best = (gap, part)
    return best


#: owner 2026-10-12 item 2 (large-scale eval picks): generation keys that relax a rule of this module; unset (None / False) keeps the
#: rule as before. The keys a variant sets are recorded in ``template_facts["relaxations"]``.
RELAXATION_KEYS = ("partner_view_min_points", "init_support_max_geodesic_m", "init_support_fallback", "under_axis_max_deg")


def enough_in_view(inside, min_points: Optional[int] = None) -> bool:
    """Owner 2026-10-12 item 2a (``generation.partner_view_min_points``): the partner's region points in frame (``inside``: one flag per
    point, the 5 % margin) - every point when ``min_points`` is None (the default, as before), else at least ``min_points`` of them
    (every point when the region has fewer)."""
    inside = th.as_tensor(inside, dtype=th.bool).reshape(-1)
    if min_points is None:
        return bool(inside.all())
    return int(inside.sum()) >= min(int(min_points), int(inside.numel()))


def intersect_feasible(F_init: Sequence[Tuple[float, Any]], F_goal: Sequence[Tuple[float, Any]], yaw_step: float = math.pi / 8) -> List[Tuple[float, Any]]:
    """F_init ∩ F_goal over (yaw, grasp) pairs: grasps compared by name, yaws by their bucket on the screening grid
    (``yaw_step``; 16 yaws = π/8). The §3.10 feasible-set cache stores floats rounded to 4 decimals (``cache._plain``),
    so a cached F_goal came back with yaws like 0.3927 against F_init's 0.39269908 and the earlier 6-decimal comparison
    emptied the intersection on every cache hit (0920-021701: 'F_init 21, F_goal 21, F 0' next to 'F 17' for the same sets)."""
    key = lambda yg: (int(round(float(yg[0]) / yaw_step)), getattr(yg[1], "name", yg[1]))  # noqa: E731
    goal_keys = {key(yg) for yg in F_goal}
    return [yg for yg in F_init if key(yg) in goal_keys]


def trunk_first_order(c_pool: Sequence[Any], order: Sequence[int]) -> List[int]:
    """The stance C ORDER with the least extended trunk first (owner decision 2026-11-07, i7-placeorder, option 1 for inside_large_01;
    opt-in ``generation.c_order_trunk_first``): the indices ``order`` into ``c_pool`` re-sorted by (trunk translate, stand-off) ascending -
    the least trunk translate first, then the nearest stand-off; a STABLE sort, so the prior order (travel / edge rank / the seeded
    permutation) decides among equal keys. It changes the order, never the set: no candidate is dropped (the per-coupling budget
    ``c_candidates_max`` is the same); the exact IK and the QA gates judge every candidate as before, so a more extended trunk is coupled
    only after the less extended candidates before it failed. The per-alias outcome memory (``alias_memory.reorder``, i7-memory) takes this
    order as its prior and only defers the candidates of the bins it down-weighted.

    Evidence (``$OFH/placeorder_20261107/demos.jsonl``, 2340 demos): no campaign-wide link between trunk extension and place failures; the 3
    place-time robot_wobble / robot_touching_itself of the campaign are inside_large_01 at translate 0.50 / tilt -0.6 / stand-off 0.85, while
    translate 0.25 at 0.85 went 4 / 4."""
    def key(k: int) -> Tuple[float, float]:
        c = c_pool[int(k)]
        return round(float(c.translate), 4), round(float(c.stand_off or 0.0), 4)
    return sorted((int(k) for k in order), key=key)


def pair_key(yaw: float, grasp: Any, yaw_step: float = math.pi / 8) -> Tuple[int, Any]:
    """A (yaw, grasp) pair's key on the screening grid (:func:`intersect_feasible`'s): yaw bucket, grasp name."""
    return int(round(float(yaw) / yaw_step)), getattr(grasp, "name", grasp)


def dead_at_b(F_live: Sequence[Tuple[float, Any]], b_ik_cache: Dict[Any, Any], yaw_step: float, end_aligned=lambda yaw0: None,
              buckets: Optional[set] = None) -> bool:
    """oracle-fix 2026-11-06 (speed, ``generation.c_skip_dead_b``): True when every pair of ``F_live`` a draw at this C can take (the yaw
    buckets ``buckets`` only, when given: the push-under yaw lock's) ends in the draw loop's first ``continue`` - its B key (the draw
    loop's ``bk``: yaw bucket, or the aligned init yaw of ``end_aligned``, and the grasp name) holds None in ``b_ik_cache`` (the pick's
    exact IK failed at B for every arm that held it) or ``end_aligned`` gives False (no aligned goal yaw). Such a C cannot be written:
    its push chain, F_goal and release-tilt screens are skipped. False for no candidate pair (the C loop's own rules name that)."""
    n = 0
    for yaw0, g in F_live:
        b = int(round(float(yaw0) / yaw_step))
        if buckets is not None and b not in buckets:
            continue
        n += 1
        y_end = end_aligned(float(yaw0))
        if y_end is False:
            continue
        bk = (b if y_end is None else round(float(y_end), 4), getattr(g, "name", str(g)))
        if bk not in b_ik_cache or b_ik_cache[bk] is not None:
            return False
    return n > 0


def union_feasible(F_by_arm: Dict[str, Sequence[Tuple[float, Any]]], yaw_step: float = math.pi / 8) -> List[Tuple[float, Any]]:
    """The usable pairs of a B over its arms (owner 2026-09-27, first-demo v2.2): the first arm's F in order, then the pairs only
    the next arms hold."""
    out: List[Tuple[float, Any]] = []
    seen = set()
    for F in F_by_arm.values():
        for yg in F:
            k = pair_key(yg[0], yg[1], yaw_step)
            if k not in seen:
                seen.add(k)
                out.append(yg)
    return out


# ---- the arm of the segment (owner 2026-09-27, first-demo v2.2): the module's right-side laterals, laid out for the right arm,
# mirrored for the left arm ----------------------------------------------------------------------------------------------------


def arm_sign(arm: str) -> float:
    """+1 for the right arm, -1 for the left: the sign of the module's right-side laterals (floor B offsets, carry poses, frontal
    push lane, side push stance) for ``arm``."""
    return 1.0 if arm == "right" else -1.0


def mirrored_offsets(offsets: Sequence[float], arm: str) -> List[float]:
    """Stance B's lateral offsets of the floor anchor cell for ``arm``: ``floor_init_lateral_offsets_m`` put c_t to the robot's right
    (the right arm's floor band), negated for the left arm (the base turned the other way, c_t to the robot's left)."""
    return [arm_sign(arm) * float(v) for v in offsets]


def offset_yaw(yaw0: float, y_off: float, dist: float) -> float:
    """Base yaw that puts the point ``dist`` away at the bearing ``yaw0`` at the base-frame lateral ``y_off`` (+ = the robot's left)."""
    return yaw0 + math.asin(-y_off / dist)


#: i7-goalmaps 2026-11-08 (lead decision, touching_large_02's umbrella carry): opt-in ``ISPATIALGYM_CARRY_CENTRED=1`` (default off) - the carry
#: pose candidates of :meth:`IORSLarge.carry_poses_at_b` start with :data:`CARRY_CENTRED_POSES`: the hand in front of the body, its lateral
#: chosen so that the CARRIED OBJECT's centre (its hull's bounding-box centre) lies on the base's centre line. The beside-the-body poses
#: (CARRY_POSES, the hand 0.35-0.45 m to the arm's side) hold a long object held across the heading (the umbrella, 0.82 m, released along
#: the hall tree's board) out to ~0.85 m on that side: gm_t02_umb2 / umb2s + PATH_BLAME, every carry with the hand >= 0.3 m to the side
#: failed its B -> C path (~800 of 800; the umbrella or the gripper against standing_mirror_wwkuuf_0 / the walls), the hand within 0.1 m of
#: the centre line passed 8 of 15. The same exact checks judge them (the head camera frame from B, the exact IK with the object attached,
#: the in-hand line from the lift, the carried path); only the order and the extra candidates change. Kit counters carry_pose.centred_offered
#: / carry_pose.centred_used.
CARRY_CENTRED_ENV = "ISPATIALGYM_CARRY_CENTRED"
#: i7-goalmaps 2026-11-08 (lead decision, the same carry): opt-in ``ISPATIALGYM_CARRY_SIDEWAYS=1`` (default off) - when the carried B -> C path
#: collides at every variant tried today (straight arrival, plain; local then global), it is planned again with the base driving SIDEWAYS
#: (``NAV.plan_path`` heading_offset +-pi / 2, the R1Pro base is holonomic): the object, held across the robot's heading because the release
#: needs it there (the umbrella along the hall tree's board, across the robot facing the board), then lies along the way. The offset whose
#: sideways yaw on the straight B -> C line is nearer C's yaw is tried first. The same swept check (the object attached, the carry arms,
#: the yaw-lag envelope) judges it; the record is "recorded" style with the sideways yaws, which the oracle tracks as recorded. Kit counters
#: carry_path.sideways_tried / carry_path.sideways_used.
CARRY_SIDEWAYS_ENV = "ISPATIALGYM_CARRY_SIDEWAYS"


#: i7-goalmaps 2026-11-08 (lead decision, the same carry): opt-in ``ISPATIALGYM_CARRY_WIDE_PATH=1`` (default off) - after the carried B -> C path
#: collides at every variant of today, the GLOBAL path is planned again on the eroded traversability map eroded further by each of
#: :data:`CARRY_WIDE_PATH_EROSIONS_M` (``NAV.plan_path`` global_extra_erosion_m, a square kernel like the map's own erosion): the shortest path
#: no longer hugs the obstacles, so an object held across the heading (the umbrella reaches ~0.4 m beyond the base's sides) passes them -
#: the centred umbrella carries of gm_t02_cc_on collided 20-30 % along the path, base at (-0.6..-0.8, 1.4..1.5) heading north past
#: standing_mirror_wwkuuf_0. The same swept check judges the path. Kit carry_path.wide_tried / wide_used.
CARRY_WIDE_PATH_ENV = "ISPATIALGYM_CARRY_WIDE_PATH"
CARRY_WIDE_PATH_EROSIONS_M: Tuple[float, ...] = (0.15, 0.30)


def carry_wide_path_on(gp: Any = None) -> bool:
    return _switch(CARRY_WIDE_PATH_ENV, gp, "carry_wide_path")


def carry_sideways_on() -> bool:
    return os.environ.get(CARRY_SIDEWAYS_ENV, "0").strip().lower() in ("1", "on", "true", "yes")


def sideways_offsets(from_xy: Sequence[float], to_xy: Sequence[float], to_yaw: float) -> List[float]:
    """+-pi / 2, the one whose sideways yaw on the straight line ``from_xy`` -> ``to_xy`` lies nearer ``to_yaw`` first."""
    h = math.atan2(float(to_xy[1]) - float(from_xy[1]), float(to_xy[0]) - float(from_xy[0]))
    offs = [math.pi / 2, -math.pi / 2]
    return sorted(offs, key=lambda o: abs((h + o - float(to_yaw) + math.pi) % (2.0 * math.pi) - math.pi))
#: (ahead, raise above the lift) of the centred carry poses, tried in this order (0.45-0.55 m ahead: nearer folds the elbow, 0920-071230)
CARRY_CENTRED_POSES: Tuple[Tuple[float, float], ...] = ((0.50, 0.10), (0.45, 0.15), (0.55, 0.05), (0.50, 0.20))


def _switch(env: str, gp: Any, key: str) -> bool:
    """An opt-in carry switch: the env var when set (1 / on / true / yes = on, anything else off), else the alias's ``generation.<key>``
    (lead decision #42, 2026-11-08: the alias-scoped default - only touching_large_02's customs set carry_centred / carry_wide_path), else off."""
    v = os.environ.get(env)
    if v is not None and v.strip() != "":
        return v.strip().lower() in ("1", "on", "true", "yes")
    try:
        return bool(gp.gen(key, False)) if gp is not None else False
    except Exception:  # noqa: BLE001 - a stand-in gp without generation keys
        return False


def carry_centred_on(gp: Any = None) -> bool:
    return _switch(CARRY_CENTRED_ENV, gp, "carry_centred")


def centred_hand_lateral(eef: Any, eef_in_object: Any, hull_local: Optional[th.Tensor], left: th.Tensor) -> float:
    """The hand's lateral offset (along ``left``, the base's +y) that puts the carried object's centre on the base's centre line: minus the
    lateral of (object hull's bounding-box centre - eef position), the object posed from the eef by the grasp (``eef_in_object`` =
    T_obj->eef). Without a hull: the object's root."""
    obj = G.compose((G.t(eef[0]).float(), G.t(eef[1]).float()), G.invert((G.t(eef_in_object[0]).float(), G.t(eef_in_object[1]).float())))
    if hull_local is not None and len(hull_local):
        pts = G.local_points_world(hull_local.float(), obj)
        centre = 0.5 * (pts.min(dim=0).values + pts.max(dim=0).values)
    else:
        centre = G.t(obj[0]).float()
    d = (centre[:2] - G.t(eef[0]).float()[:2])
    return -float(d @ G.t(left).float()[:2])


def carry_poses_for(poses: Sequence[Sequence[Any]], arm: str) -> List[Tuple[float, Optional[float], float]]:
    """``CARRY_POSES`` / ``generation.carry_poses`` (ahead, lateral, raise) for ``arm``: the lateral beside the body on the arm's side
    (the right arm's -0.35..-0.45 m positive for the left arm); None (= the lift's lateral) kept."""
    s = arm_sign(arm)
    return [(ahead, None if lat is None else s * float(lat), raise_m) for ahead, lat, raise_m in poses]


def frontal_lane_lateral(lane_lat: float, offset: float, arm: str) -> float:
    """Base lateral (along the edge tangent, + = the robot's left facing the edge) of a frontal push-under stance whose push lane at
    ``lane_lat`` lies ``offset`` to ``arm``'s side of the base (right arm: the base left of the lane)."""
    return float(lane_lat) + arm_sign(arm) * float(offset)


def side_push_pose(mid: th.Tensor, tangent: th.Tensor, inward: th.Tensor, lane_lat: float, x_ahead: float, b: float, arm: str) -> Tuple[float, float, float]:
    """Base pose of a side push-under stance: ``b`` outside the edge, ``x_ahead`` short of the lane point along the edge, facing along
    the edge with the partner on ``arm``'s side (right arm: facing +tangent, the robot's right = inward; left arm: facing -tangent)."""
    s = arm_sign(arm)
    xy = mid + tangent * (float(lane_lat) - s * float(x_ahead)) - inward * float(b)
    return float(xy[0]), float(xy[1]), math.atan2(s * float(tangent[1]), s * float(tangent[0]))


@dataclass
class SegmentArms:
    """The arms of the segment at one stance B (owner 2026-09-27, first-demo v2.2): ``stances[arm]`` (the arm working, the other one
    in its A6 idle posture) and ``F_init[arm]``, the default arm first; ``bands[arm]`` the band that shortlisted B; ``b_ik`` the grasps
    at B shared by the arms' couplings ((yaw bucket, grasp) -> None or the pick: ``yaw``, ``t_pose``, the ``arm`` that won it, the arms
    ``viable`` there, ``opts`` per arm, ``after`` the arms holding the pair whose segment had failed); ``done`` the arms whose coupling
    failed, ``failed`` a pair's first failure after its grasp, ``why`` a coupling's error; ``leg`` the version of the recorded A -> B
    leg and ``leg_ok`` the drive's clearance of the placed target per (pair, leg)."""

    stances: Dict[str, Stance]
    F_init: Dict[str, List[Tuple[float, Any]]]
    bands: Dict[str, Any] = field(default_factory=dict)
    yaw_step: float = math.pi / 8
    b_ik: Dict[Tuple[int, Any], Optional[Dict[str, Any]]] = field(default_factory=dict)
    done: List[str] = field(default_factory=list)
    failed: Dict[Tuple[int, Any], str] = field(default_factory=dict)
    why: Dict[str, str] = field(default_factory=dict)
    leg: int = 0
    leg_ok: Dict[Tuple[Any, int], Optional[str]] = field(default_factory=dict)

    @property
    def arms(self) -> List[str]:
        return list(self.stances)

    def arms_of(self, yaw: float, grasp: Any) -> List[str]:
        """The arms whose F_init holds the pair, in order (the arms a grasp at B offers)."""
        k = pair_key(yaw, grasp, self.yaw_step)
        return [a for a in self.stances if any(pair_key(y, g, self.yaw_step) == k for y, g in self.F_init.get(a, []))]

    def role(self, chosen: str, viable: Sequence[str], arm: str) -> str:
        """What a pair whose grasp at B went to ``chosen`` is in ``arm``'s coupling: ``own`` (chosen is the arm), ``defer`` (the chosen
        arm's coupling has not run yet), ``fallback`` (the chosen arm's coupling failed and ``arm`` was viable at the grasp) or
        ``none``."""
        if chosen == arm:
            return "own"
        if chosen not in self.done:
            return "defer"
        return "fallback" if arm in viable else "none"


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*RELATION_OF_SUBTASK)
class IORSLarge(SamplerV2):
    relation_mode: Optional[type] = None  # the relation's mode class once adopted (inside / touching / nextto)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = RELATION_OF_SUBTASK[self.subtask]
        if self.gp.bddl is None and self.relation in V1_DEFINITION_ID:
            self.definition_id = V1_DEFINITION_ID[self.relation]
        self.hull: Optional[th.Tensor] = None
        self.rest: th.Tensor = G.IDENTITY_QUAT.clone()
        self.bands: Optional[RB.ReachBands] = None  # ``tabletop.ensure_bands`` caches the init-support band here
        self.band_spec: Optional[TT.BandSpec] = None  # what ``ensure_bands`` computes on a miss (set once the grasps are loaded)
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.surface: Dict[str, Any] = {}  # kind, obj, top, band, arm_bands, edge, init, manip candidates
        self.place: Dict[str, Any] = {}  # partner geometry for C: top / under cells, edges, band
        self.floor_z = 0.0
        self.seg_arm: str = TT.default_arm(self)  # the arm of the grasp -> carry -> place -> release segment being coupled
        self.partner_joints: Optional[Dict[str, float]] = None  # every partner joint once generation.partner_joint_init set them
        self.partner_moved: List[str] = []  # the partner links that request moved (the open door)
        self.finger_touch_error: Optional[str] = self.setup_finger_touch()  # a bad value: the spec rejection of bind_scene_roles
        self.v2_overhang: Optional[Dict[str, Any]] = None  # i7-overhang: the v2 overhang rest of the grasp key (tabletop.overhang_rest_of_key)

    # ------------------------------------------------------------------ roles

    def bind_scene_roles(self):
        if getattr(self, "finger_touch_error", None):
            raise Reject("spec", f"{self.gp.variant}: generation.{self.finger_touch_error}")
        if self.relation in MODE_HOOKS and self.relation_mode is None:
            self._adopt_mode(getattr(self, MODE_HOOKS[self.relation])())
            return self.bind_scene_roles()  # the mode's own, else the ontop / under flow from here on with the mode's overrides
        self.bind_floor_v2()
        req = self.choice.bindings.get("partner")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if names:
            partner = self.world.obj(names[0])
        else:
            cats = list((req or {}).get("category", PARTNER_CATEGORIES)) if isinstance(req, dict) or req is None else list(PARTNER_CATEGORIES)
            rooms = set(self.world.rooms or [])
            cands = sorted([o for o in self.world.scene.objects if o.category in cats and (not rooms or set(o.in_rooms or []) & rooms)], key=lambda o: o.name)
            if not cands:
                raise Reject("spec", f"{self.gp.variant}: no partner candidate of {cats} in {sorted(rooms)}")
            partner = cands[0]
        self.bind_scene_object("partner", partner)
        self.apply_partner_joint_init()

    def apply_partner_joint_init(self) -> None:
        """``generation.partner_joint_init`` (module docstring; default none: nothing changes): the partner's joints set (clamped to their
        limits), the robot re-parked away from the partner, a settle of :data:`PARTNER_JOINT_SETTLE_STEPS`, the joints re-read; the links
        that moved are kept in ``partner_moved``. ``Reject("spec")`` for an unknown joint or a value that is not a number."""
        self.partner_joints, self.partner_moved = None, []
        req = self.gp.gen("partner_joint_init")
        if not req:
            return
        partner = self.partner
        try:
            spec, clamped = OJ.parse_joint_spec(req, list(partner.joints), OJ.joint_limits(partner))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: generation.partner_joint_init on {partner.name}: {e}")
        before = OJ.link_poses(partner)
        q0 = OJ.joint_dict(partner)
        OJ.set_joints(partner, spec)
        self.partner_joints = OJ.joint_dict(partner)  # read by park_keep_away from here on
        self.park_robot()
        residual = self.world.settle([partner], steps=PARTNER_JOINT_SETTLE_STEPS)
        partner.set_joint_velocities(th.zeros_like(partner.get_joint_positions()))
        partner.keep_still()
        self.partner_joints = OJ.joint_dict(partner)
        self.partner_moved = OJ.moved_links(before, OJ.link_poses(partner))
        settled = {n: round(self.partner_joints[n], 4) for n in spec}
        dev = max(abs(self.partner_joints[n] - v) for n, v in spec.items())
        self.template_facts["partner_joint_init"] = {"object": partner.name, "requested": {n: float(v) for n, v in req.items()}, "applied": spec, "clamped": clamped,
                                                     "scene_file": {n: round(q0[n], 4) for n in spec}, "settled": settled, "max_settle_dev_rad": round(dev, 4),
                                                     "joints": {n: round(v, 4) for n, v in self.partner_joints.items()}, "moved_links": list(self.partner_moved),
                                                     "settle_residual": residual}
        log.info("%s: generation.partner_joint_init on %s: %s -> settled %s (max deviation %.3f%s), moved links %s", self.gp.variant, partner.name, spec, settled, dev,
                 f"; clamped {clamped}" if clamped else "", self.partner_moved)
        if dev > 0.05:
            log.warning("%s: %s's joints moved %.3f from the requested %s during the settle (settled %s)", self.gp.variant, partner.name, dev, spec, settled)

    def partner_static_aabb(self) -> Tuple[th.Tensor, th.Tensor]:
        """The partner's AABB without the links ``partner_joint_init`` moved (the open door is no part of a stand-off)."""
        return OJ.static_aabb(self.partner, self.partner_moved)

    def partner_moved_boxes(self) -> List[Tuple[th.Tensor, th.Tensor]]:
        """World AABBs of the collision pieces of the links ``partner_joint_init`` moved (none without the key)."""
        return OJ.link_piece_aabbs(self.partner, self.partner_moved) if self.partner_moved else []

    def partner_part_boxes(self) -> List[Tuple[th.Tensor, th.Tensor]]:
        """With ``partner_joint_init``: the collision pieces of the partner's articulated parts - every link but its root, the doors open
        or closed (a bin in a fridge door is no part of the shelf behind it); none without the key."""
        if not getattr(self, "partner_joints", None):
            return []
        partner = self.partner
        root = getattr(partner, "root_link_name", "base_link")
        return OJ.link_piece_aabbs(partner, [n for n in partner.links if n != root])

    def park_keep_away(self) -> List[Any]:
        """With ``partner_joint_init`` the parked robot keeps 1 m from the partner too (the open door's swing is no parking cell)."""
        return [self.partner] if getattr(self, "partner_joints", None) else []

    def path_blocked(self) -> List[Any]:
        """The moved links' floor footprints: the paths plan around the open door (its body stays in every swept check)."""
        return OJ.footprints(self.partner_moved_boxes(), PARTNER_MOVED_FOOTPRINT_M) if getattr(self, "partner_moved", None) else []

    def record_partner_joints(self, draft: Draft) -> None:
        """After the settle: the partner's joints as settled -> the TRO's ``joint_positions`` constraint (the requested joints; the loader
        sets, holds and restores them) and ``summary.partner_joints``."""
        if not getattr(self, "partner_joints", None):
            return
        partner, pname = self.partner, self.pname
        q = draft.object_joints.get(pname)
        joints = {n: float(q[i]) for i, n in enumerate(partner.joints)} if q is not None else OJ.joint_dict(partner)
        named = {n: round(joints[n], 5) for n in (self.gp.gen("partner_joint_init") or {})}
        draft.extras["constraints"] = [c for c in (draft.extras.get("constraints") or []) if not (c.get("type") == "joint_positions" and c.get("object") == pname)]
        draft.extras["constraints"].append({"type": "joint_positions", "object": pname, "joints": named, "source": "generation.partner_joint_init",
                                            "note": "the partner's joints at init as the sampler set them (owner 2026-10-12 ~16:05, inside_large_scale#02: the fridge "
                                                    "door open); the loader sets them after load_tro_instance, holds them in the settle and restores them after a reset"})
        draft.summary["partner_joints"] = {"object": partner.name, "init": {n: round(v, 4) for n, v in joints.items()}, "requested": named}

    def sample(self, rng: np.random.Generator) -> Draft:
        draft = super().sample(rng)
        if getattr(self, "partner_joints", None):
            draft.object_joints[self.pname] = th.tensor([self.partner_joints[n] for n in self.partner.joints], dtype=th.float32)
        return draft

    @property
    def partner(self):
        return self.obj("partner")

    @property
    def target(self):
        return self.obj("target")

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    @property
    def pname(self) -> str:
        return self.bindings.one("partner").name

    def support_objects(self) -> List[Any]:
        """The init surface when it is a piece of furniture (parking keep-away, grasp ignore) and the neighbouring supports its top
        is united with; the floor is handled by :meth:`grasp_ignore_objects` (its AABB would forbid every parking cell)."""
        out = []
        if self.surface.get("obj") is not None:
            out.append(self.surface["obj"])
        return out + NB.neighbor_objects(self)

    def grasp_ignore_objects(self) -> List[Any]:
        """§5.6: the init surface ignored for the grasp pose only (the floor for the floor init)."""
        return [self.obj("floor")] if self.surface.get("kind") == "floor" else self.support_objects()

    def setup_finger_touch(self) -> Optional[str]:
        """``generation.approach_finger_support_touch`` (owner 2026-10-13 ~03:15 item 3; default off = today's approach check): the IK
        checker's :class:`IK.FingerTouch` - along a tracked grasp approach (the draw at B, the program check's grasp) the acting arm's
        finger links may touch :meth:`grasp_ignore_objects` lightly, checked at :meth:`preshape_opening`. Returns the error of a bad value
        (the rule stays off; :meth:`bind_scene_roles` rejects the template ``spec``). i7-fingertouch 2026-11-07: the fingers vs the support
        by ``ISPATIALGYM_FINGER_TOUCH_RULE`` (``FingerTouch.rule``: "geometry" = the contact policy's 2 mm on the finger hulls, the default since
        the A/B of 2026-11-07; "spheres" = the key's ``max_penetration_m`` on the finger spheres); a bad switch value is the same ``spec`` rejection."""
        try:
            p = IK.finger_touch_params(self.gp.gen("approach_finger_support_touch"))
            ft = None if p is None else IK.FingerTouch(p["margin_m"], p["max_penetration_m"], p["preshape"], supports=self.grasp_ignore_objects,
                                                       opening=self.preshape_opening, rule=IK.finger_touch_rule())
        except ValueError as e:
            return str(e)
        self.ikc.set_finger_touch(ft)
        return None

    def preshape_opening(self) -> Optional[float]:
        """The widest per-finger opening the oracle pre-shapes to before approaching one of the target's library grasps (``execute_grasp``:
        pinch width / 2 + its pre-shape margin, at most fully open); None when a row has no width (the oracle approaches fully open then)."""
        widths = [TT.grasp_pinch_width(g) for g in self.grasps]
        if not widths or any(w is None for w in widths):
            return None
        return min(float(C.FINGER_OPEN), max(0.5 * w + IK.ORACLE_PRESHAPE_MARGIN_M for w in widths))

    # duck-typed attributes the shared tabletop phases read (``tabletop.py`` / the TEMP stand-in of ``iors_small``)
    @property
    def support(self):
        return self.surface.get("obj")

    @property
    def top(self) -> Optional[SupportTop]:
        return self.surface.get("top")

    def num(self, key: str, default: float) -> float:
        v = self.gp.gen(key)
        if v is None and key.endswith("_m"):
            v = self.gp.gen(key[:-2])  # v1 spelling (max_partner_geodesic)
        return float(default if v is None else v)

    # ------------------------------------------------------------------ init surface (§5.6)

    def surface_candidates(self) -> Tuple[str, List[Any]]:
        b = self.choice.bindings
        req = b.get("support")
        try:
            kind = init_surface_class(self.relation, req, [self.partner.name], variant_support_token(self.gp.variant, self.subtask),
                                      variant_partner_token(self.gp.variant, self.subtask))
        except ValueError as e:
            raise Reject("spec", str(e))
        self.template_facts["init_surface_selection_order"] = list(SELECTION_ORDER.get(self.relation, ()))
        if kind == "floor":
            return kind, [self.obj("floor")]
        if kind == "partner":
            return kind, [self.partner]
        # owner 2026-10-12 item 2b: ``generation.init_support_max_geodesic_m`` drops a support, named or by category, whose geodesic to
        # the partner is None or larger (unset: named supports are taken as bound, category ones within ``max_partner_geodesic_m``);
        # ``self.support_drops`` = {name: why} for the fallback's reason
        cap = self.gp.gen("init_support_max_geodesic_m")
        self.support_drops: Dict[str, str] = {}
        names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
        if names:
            if cap is None:
                return kind, [self.world.obj(n) for n in names]
            out = []
            for n in names:
                try:
                    o = self.world.obj(n)
                except KeyError:
                    self.support_drops[n] = "missing from the scene"
                    continue
                if self._support_near_partner(o, float(cap)):
                    out.append(o)
            self.template_facts["init_support_dropped"] = dict(self.support_drops)
            return kind, out
        cats = list((req or {}).get("category", SUPPORT_CATEGORIES)) if isinstance(req, dict) or req is None else list(SUPPORT_CATEGORIES)
        rooms = set(self.world.rooms or [])
        cands = [o for o in self.world.scene.objects if o.category in cats and o.name in self.world.scene_object_names and (not rooms or set(o.in_rooms or []) & rooms)]
        cands = [o for o in cands if o is not self.partner]
        max_geo = self.num("max_partner_geodesic_m", 3.0) if cap is None else float(cap)
        out = [o for o in sorted(cands, key=lambda o: o.name) if self._support_near_partner(o, max_geo)]
        if not cands:
            self.support_drops["category"] = f"no {cats} object in the rooms {sorted(rooms)}"
        if cap is not None:
            self.template_facts["init_support_dropped"] = dict(self.support_drops)
        return kind, out

    def _support_near_partner(self, o, max_geo: float) -> bool:
        """The support's geodesic to the partner exists and is ≤ ``max_geo`` (else noted in ``self.support_drops``)."""
        dist = self.geodesic_between(o, self.partner)
        if dist is not None and dist <= max_geo:
            return True
        self.support_drops[o.name] = "no geodesic to the partner" if dist is None else f"geodesic {dist:.2f} m > {max_geo:.2f} m"
        return False

    def init_support_fallback(self, kind: str) -> Optional[Dict[str, Any]]:
        """Owner 2026-10-12 item 2b (``generation.init_support_fallback``, default off: A5's "no runtime class switching"): when no support
        is left (too far, missing from the scene, none of the category), ontop falls back to the floor init and under to the partner's own
        top. Returns ``{from, to, reason}`` (recorded as ``template_facts.init_surface.fallback``) or None."""
        if kind != "support" or not bool(self.gp.gen("init_support_fallback", False)):
            return None
        drops = getattr(self, "support_drops", {}) or {}
        reason = "; ".join(f"{n}: {why}" for n, why in drops.items()) or "no support candidate"
        return {"from": "support", "to": "floor" if self.relation == "ontop" else "partner", "reason": reason}

    def near_side_point(self, obj, toward_xy, off: float = 0.5) -> th.Tensor:
        """A point 0.5 m outside the AABB side of ``obj`` that faces ``toward_xy`` (geodesic anchor)."""
        lo, hi = obj.aabb
        p = th.as_tensor([float(toward_xy[0]), float(toward_xy[1])])
        near = th.minimum(th.maximum(p, lo[:2].float()), hi[:2].float())
        d = p - near
        n = float(th.linalg.norm(d))
        return near + (d / n if n > 1e-6 else th.tensor([1.0, 0.0])) * off

    def geodesic_between(self, a, b) -> Optional[float]:
        """``kit.geodesic`` (OmniGibson's A* on the robot-eroded map) between the :meth:`near_side_point` anchors of ``a`` and ``b``. That
        A* finds no path when an anchor lies off the eroded map - in a support whose AABB holds the partner (a drop-in sink set in the
        counter run), on a neighbour (an L-joined counter), in the erosion band in front of a counter -, so then both anchors move to
        their nearest eroded-free cell (:meth:`free_cell_near`) and the geodesic is taken between those."""
        ca, cb = a.aabb_center[:2].float(), b.aabb_center[:2].float()
        pa, pb = self.near_side_point(a, cb), self.near_side_point(b, ca)
        dist = self._geodesic(pa, pb)
        if dist is None:
            ends = [self.free_cell_near(p) for p in (pa, pb)]
            if all(e is not None for e in ends):
                dist = self._geodesic(*ends)
        return dist

    def _geodesic(self, pa, pb) -> Optional[float]:
        try:
            _, dist = self.kit.geodesic(pa, pb)
        except Exception:  # noqa: BLE001
            return None
        return dist

    def free_cell_near(self, xy, max_cells: int = 20) -> Optional[th.Tensor]:
        """Centre (world xy) of the eroded-free traversability cell nearest ``xy`` (``navigation.nearest_free_cell``: ``xy``'s own cell when
        free, None beyond ``max_cells``). The centre, not the corner ``map_to_world`` gives: ``world_to_map`` truncates, and a corner can
        come back as the blocked neighbour."""
        ij = NAV.nearest_free_cell(NAV.eroded_map(self.kit), NAV._to_map(self.kit, xy), max_cells)
        if ij is None:
            return None
        return NAV._to_world(self.kit, ij) + 0.5 * float(self.kit.world.scene.trav_map.map_resolution)

    def prepare_bindings(self):
        try:
            self._prepare_bindings()
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def _prepare_bindings(self):
        """Target geometry and grasps, the partner's placement geometry, then the init surface: one candidate of the
        grounded class drawn uniformly, the rest of the class in random order; the support role bound (aliased to the
        partner / floor bound when it is the same object)."""
        relaxed = {k: self.gp.gen(k) for k in RELAXATION_KEYS if self.gp.gen(k) not in (None, False)}
        if relaxed:
            self.template_facts["relaxations"] = _jsonable(relaxed)
            log.info("%s: relaxed rules (owner 2026-10-12 item 2): %s", self.gp.variant, relaxed)
        tgt = self.target
        self.hull = self.world.local_hull(tgt)
        # i7-overhang 2026-11-07: the overhang pick alone (mode only) takes the key's v2 OVERHANG rest when the library holds one (every other
        # alias: the call as before)
        gen = getattr(self.gp, "gen", None)
        try:
            ov0 = OH.options(gen("pick_overhang_push")) if callable(gen) else None
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")
        if ov0 is not None and ov0["mode"] == "only":
            self.grasps, self.grasp_key, self.rest = TT.library_grasps(self, "target", overhang=True)
        else:
            self.grasps, self.grasp_key, self.rest = TT.library_grasps(self, "target")
        self.v2_overhang = TT.overhang_rest_of_key(self.grasp_key)
        if self.v2_overhang is not None:
            self.template_facts["v2_overhang"] = {k: self.v2_overhang.get(k) for k in ("tag", "side", "distance_m", "measured_distance_m", "edge_along_m",
                                                                                      "com_inside_m", "normal_xy", "entry", "rest_id")}
        keep = self.gp.gen("grasp_rows")
        if keep and self.v2_overhang is not None:
            # the rows name the LEGACY library's pinches; the v2 overhang entry's grasps are its own (generated at the overhanging face)
            log.info("%s: generation.grasp_rows %s not applied: the grasps are the v2 overhang entry's %s", self.gp.variant, keep, [g.name for g in self.grasps])
            keep = None
        if keep:  # oracle-fix 2026-11-04 round 3 (hardback): opt-in generation.grasp_rows - only these library rows (the spine edge's pinches)
            kept = [g for g in self.grasps if getattr(g, "name", None) in set(keep)]
            if not kept:
                raise Reject("spec", f"{self.gp.variant}: generation.grasp_rows {keep} keeps none of {[getattr(g, 'name', None) for g in self.grasps]}")
            log.info("%s: grasp rows %s kept of %s", self.gp.variant, [g.name for g in kept], [getattr(g, "name", None) for g in self.grasps])
            self.grasps = kept
        self.band_spec = self.make_band_spec()
        self.floor_z = float(self.obj("floor").aabb[1][2])
        self.prepare_partner()
        kind, cands = self.surface_candidates()
        self.init_fallback = self.init_support_fallback(kind) if not cands else None
        if self.init_fallback is not None:
            log.info("%s: no init support left (%s): the %s init instead (generation.init_support_fallback)", self.gp.variant, self.init_fallback["reason"],
                     self.init_fallback["to"])
            kind = self.init_fallback["to"]
            cands = [self.obj("floor")] if kind == "floor" else [self.partner]
        if not cands:
            raise Reject("no_init_surface", f"{self.gp.variant}: no candidate of the init-surface class {kind!r}")
        self.bind_init_surface(kind, cands)

    def bind_init_surface(self, kind: str, cands: List[Any]) -> None:
        """The init surface of the template: the candidates in random order (the template seed's stream), the first that admits stances
        A–C bound as the support role; ``Reject("no_init_surface")`` when none does.

        oracle-fix 2026-11-06 (opt-in ``generation.init_setup_draws``, inside_large#03): the template seed is the same in every train job
        (multiply_isg: split_seed(train, variant) * 7919, whatever the start index), so one unlucky draw of the init edge / lateral rejects
        the template in every job (bar_egwapq_0 +u, the only collision-free edge: lateral -0.03, the free top beside the sink 37-45 deg off
        the camera axis, 0 Phase-2 pairs within 30 deg in every survey1 / trainfast / train85 job); up to that many more draws from the
        derived streams ``[base_seed, k]`` (train85's spa2 setup draws, iors_small's ``SamplerV2.spa_template_start``). Draw 0 is today's."""
        errors: List[str] = []
        for k in range(1 + max(0, int(self.gp.gen("init_setup_draws", 0) or 0))):
            rng = np.random.default_rng(self.ctx.base_seed if k == 0 else [int(self.ctx.base_seed), k])
            order = [cands[int(i)] for i in rng.permutation(len(cands))]
            for cand in order:
                try:
                    self.try_surface(kind, cand, rng)
                except Reject as e:
                    errors.append(f"{f'setup draw {k}: ' if k else ''}{cand.name}: {e.reason}: {e}")
                    continue
                if kind == "floor":
                    self.variant_role("support", self.bindings.one("floor"))
                elif kind == "partner":
                    self.variant_role("support", self.bindings.one("partner"))
                else:
                    self.bind_scene_object("support", cand)
                self.template_facts["init_surface"] = {"class": kind, "name": cand.name, "tried": len(errors) + 1, **({"setup_draw": k} if k else {}),
                                                       **({"fallback": self.init_fallback} if self.init_fallback is not None else {})}
                if k:
                    log.info("%s: init surface %s found on setup draw %d (generation.init_setup_draws; %d earlier tries failed)", self.gp.variant,
                             cand.name, k, len(errors))
                return
        raise Reject("no_init_surface", f"{self.gp.variant}: no {kind} candidate admits stances A–C: " + "; ".join(errors[-3:]))

    # ------------------------------------------------------------------ partner geometry for C

    def partner_record(self) -> Optional[Dict[str, Any]]:
        p = os.path.join(SCENE_INVENTORY_ROOT, self.world.scene_model, "partners", f"{self.partner.name}.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def target_dims(self) -> Tuple[float, float]:
        pts = G.local_points_world(self.hull, (th.zeros(3), self.rest))
        ext = pts.max(dim=0).values - pts.min(dim=0).values
        return float(ext[:2].max()) / 2, float(ext[2])

    def prepare_partner(self):
        partner = self.partner
        half, height = self.target_dims()
        margin = self.num("goal_cell_margin_m", 0.05) + half
        lo, hi = partner.aabb
        if self.relation == "ontop":
            top = self.support_top(partner)
            if int(top.free.sum()) < 50:
                raise Reject("partner_top_small", f"{partner.name}: too little free top")
            # i7-split 2026-11-07: the partner's top is its support-split plane - its rails / headboard / rims stay obstacles at the release
            # (``place_ignores``) and wherever else it is ignored
            from ispatialgym.sampling.v2 import support_split as SS

            SS.register_top(self, partner, top, what="partner_top")
            cells = top.cells(free_only=True)
            # boundary = raster cells with a missing neighbour (the face outline) — eroded by half-extent + 0.05
            rim = self._raster_rim(top)
            goal_cells = erode_cells(cells, rim, margin)
            edges = []
            for name, mid, normal, length in top.edges():
                edges.append(EdgeFrame.from_edge(name, mid, normal, length, edge_overhang(mid, normal, lo, hi)))
            self.place = {"top": top, "cells": goal_cells, "edges": edges, "z": top.z, "height": height, "half": half}
            if len(goal_cells) == 0:
                raise Reject("partner_too_small", f"{partner.name}: no top cell survives the erosion by {margin:.2f} m")
        else:
            rec = self.partner_record()
            need = height + self.num("under_ray_extra_m", 0.03)
            if rec is not None and rec.get("under_clearance"):
                cells = under_clearance_cells(rec["under_clearance"], need)
                clear_map = rec["under_clearance"]
            else:
                cells, clear_map = self._under_raycast(partner, need), None
            edges = aabb_edges(lo, hi)
            if self.push_under:
                # push-under: goal cells sit at the footprint edge (centre 5–10 cm inside; the target's outer part may stick
                # out) — keep the clear cells within 0.12 m inside any AABB edge
                if len(cells):
                    depth = th.stack([th.minimum(cells[:, 0] - lo[0].float(), hi[0].float() - cells[:, 0]),
                                      th.minimum(cells[:, 1] - lo[1].float(), hi[1].float() - cells[:, 1])], dim=1).min(dim=1).values
                    cells = cells[(depth >= 0.0) & (depth <= PUSH_UNDER_INSIDE_M + 0.05)]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell within {PUSH_UNDER_INSIDE_M + 0.05:.2f} m of its footprint edge clears {need:.2f} m")
            else:
                # inside the footprint shrunk by half-extent + 0.05 and within 0.45 m of a near AABB edge
                fp_lo, fp_hi = lo[:2].float() + margin, hi[:2].float() - margin
                if len(cells):
                    inside = ((cells >= fp_lo) & (cells <= fp_hi)).all(dim=1)
                    cells = cells[inside]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell under it clears {need:.2f} m inside the footprint eroded by {margin:.2f} m")
        self.template_facts["partner"] = {"name": partner.name, "n_goal_cells": int(len(self.place["cells"])), "edges": [e.to_json() for e in self.place["edges"]]}

    def _raster_rim(self, top: SupportTop) -> th.Tensor:
        m = top.mask
        pad = th.nn.functional.pad(m.float()[None, None], (1, 1, 1, 1))[0, 0] > 0.5
        interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        rim = m & ~interior
        idx = th.nonzero(rim)
        return th.stack([top.xs[idx[:, 0]], top.ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)

    def _under_raycast(self, partner, need: float, res: float = 0.05) -> th.Tensor:
        lo, hi = partner.aabb
        xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
        ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        starts = th.cat([pts, th.full((len(pts), 1), self.floor_z + 0.005)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(hi[2]) + 0.5
        hits = self.world.raycast(starts, ends)
        prims = {l.prim_path for l in partner.links.values()}
        keep = [k for k, h in enumerate(hits) if h["hit"] and h["rigidBody"] in prims and float(h["position"][2]) - self.floor_z >= need]
        return pts[keep] if keep else th.zeros(0, 2)

    # ------------------------------------------------------------------ push-under helpers (owner decision 2026‑09‑16 item 7)

    @property
    def push_under(self) -> bool:
        """The under goal is reached by drop + straight floor push (``generation.under_mode`` = ``"push"`` default)."""
        return self.relation == "under" and str(self.gp.gen("under_mode", "push")) == "push"

    @property
    def push_hand(self) -> str:
        """The hand of the floor push (``generation.push_under_hand``, :data:`PUSH_UNDER_HANDS`): ``"blade"`` (default), ``"flat"`` or
        ``"pitched_far"``."""
        v = str(self.gp.gen("push_under_hand", "blade"))
        if v not in PUSH_UNDER_HANDS:
            raise Reject("spec", f"generation.push_under_hand {v!r} is not one of {PUSH_UNDER_HANDS}")
        return v

    #: ground-like scene categories whose AABB spans rooms (the garden lawn's AABB covers the child's room floor strip)
    GROUND_CATEGORIES = ("floors", "ceilings", "walls", "lawn", "driveway", "roof", "pool", "carpet", "rug", "mat", "doormat", "fence", "rail_fence")
    GROUND_FOOTPRINT_M2 = 15.0

    def floor_blocking_objects(self) -> List[Any]:
        """Scene objects whose AABB blocks a floor cell for the target / the drop: not ground-like (categories above, or a
        footprint > 15 m²), not the robot, not spawned, ≥ 5 cm tall (flat floor coverings are part of the floor)."""
        out = []
        for o in self.world.scene.objects:
            if o.category in self.GROUND_CATEGORIES or o is self.robot or o.name in self.world.spawned:
                continue
            lo, hi = o.aabb
            if float(hi[2] - lo[2]) < 0.05:
                continue
            if float((hi[0] - lo[0]) * (hi[1] - lo[1])) > self.GROUND_FOOTPRINT_M2:
                continue
            out.append(o)
        return out

    def _free_floor_cell(self, xy, why: Optional[Dict[str, int]] = None) -> bool:
        """Free floor for the drop: un-eroded traversable, in the task rooms, ≥ 2 cm outside every object AABB (flat
        floor coverings < 5 cm ignored; the AABB list is built once per template). ``why`` counts the failing filter."""
        p = th.as_tensor([float(xy[0]), float(xy[1])])
        if not self.kit.trav_free(p):
            if why is not None:
                why["not_traversable"] = why.get("not_traversable", 0) + 1
            return False
        if not P0.in_task_rooms(self, p):
            if why is not None:
                why["outside_rooms"] = why.get("outside_rooms", 0) + 1
            return False
        aabbs = self.__dict__.get("_floor_aabbs")
        if aabbs is None:
            aabbs = [(o.name, o.aabb[0][:2].float() - 0.02, o.aabb[1][:2].float() + 0.02) for o in self.floor_blocking_objects()]
            self._floor_aabbs = aabbs
        hit = [n for n, a, b in aabbs if bool((p >= a).all() and (p <= b).all())]
        if hit:
            if why is not None:
                why[f"aabb:{hit[0]}"] = why.get(f"aabb:{hit[0]}", 0) + 1
            return False
        return True

    def drop_pose_at(self, xy, quat) -> G.Pose:
        """Release pose of the push-under drop: the target's bottom ``PUSH_UNDER_DROP_HEIGHT_M`` above the floor at ``xy``."""
        z = self.floor_z + self.world.bottom_offset(self.hull, quat) + float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M))
        return th.tensor([float(xy[0]), float(xy[1]), z]), quat

    #: height of the blade above the fingertips that can meet the pushed object (the finger pads, PAD_Z_RANGE span)
    BLADE_SPAN_M = 0.05

    def contact_offset_along(self, quat, direction_xy, z_lo: float, z_hi: float, section: bool = False) -> Optional[float]:
        """Distance from the target's root back (against the push direction) to the rearmost collision point the blade can
        meet: the collision boundary points at orientation ``quat`` whose height above the target's bottom lies in
        ``[z_lo, z_hi]`` (the blade's band: fingertips to fingertips + ``BLADE_SPAN_M``). A sandal pushed along its length
        with the fingertips 3.6 cm up is met at its STRAP, ~3 cm behind the root, not at the heel 12 cm behind: modelled
        with the footprint's half extent the blade stopped 9-10 cm short of the goal cell (0920-143722 / 154503: the
        sandal's root ended 3 cm ahead of the blade). ``None`` when nothing of the target reaches the band. ``section``
        (the flat hand's 1.4-1.7 cm fingertip band): the hull's cross-sections at the band's ends count too
        (:func:`band_back_offset`)."""
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))
        if section:
            return band_back_offset(pts, direction_xy, z_lo, z_hi, section=True)
        z = pts[:, 2] - pts[:, 2].min()
        sel = (z >= float(z_lo)) & (z <= float(z_hi))
        if not bool(sel.any()):
            return None
        d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])])
        d = d / th.linalg.norm(d)
        proj = pts[sel][:, :2] @ d
        return float(-proj.min())

    def reach_margin(self, stance: Stance, point_world, arm: Optional[str] = None) -> float:
        """W2 (owner 2026-09-20 22:15): ``arm``'s (default: the segment arm) stretched length minus the shoulder -> ``point_world``
        distance at the stance's trunk (shoulder through the shared torso FK, ``conventions.shoulder_position``; the stretched length
        once per arm through the kit's FK with the arm joints at zero). Positive = inside the envelope; the trunk that bows the
        tallest torso forward (154503 / 154504: B at (0.0, -0.61), the grasp 0.88 m out, 250-365 steps) scores lowest. The shoulder
        height is measured from the floor object's top (``self.floor_z``; ``tabletop.reach_margin`` takes the kit's measured floor)."""
        arm = arm or self.seg_arm
        attr = f"_arm_len_{arm}"
        if not hasattr(self, attr):
            from ispatialgym.oracle import r1pro_model as M
            q = self.kit.q_config(arms="untucked", translate=0.5, tilt=0.0)
            q[self.kit.robot.arm_control_idx[arm]] = 0.0
            fk = self.kit.fk(["torso_link4", M.EEF_LINK[arm]], q)
            sh = G.compose(fk["torso_link4"], (th.as_tensor(C._SHOULDER_IN_TORSO4[arm], dtype=th.float32), G.IDENTITY_QUAT))[0]
            setattr(self, attr, float(th.linalg.norm(fk[M.EEF_LINK[arm]][0] - sh)))
        # the URDF-clamped torso (a tilt past the torso_joint3 limit, e.g. (0.95, -1.2), is what the robot actually reaches)
        sh_b = C.torso_fk(C.torso_q(float(stance.translate), float(stance.tilt), C.TORSO_JOINT_LIMITS))[f"{arm}_shoulder"][0]
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        c, s_ = math.cos(yaw), math.sin(yaw)
        sh_w = th.tensor([x + c * float(sh_b[0]) - s_ * float(sh_b[1]), y + s_ * float(sh_b[0]) + c * float(sh_b[1]), self.floor_z + C.NOMINAL_BASE_Z + float(sh_b[2])])
        p = th.as_tensor([float(v) for v in point_world], dtype=th.float32)
        return getattr(self, attr) - float(th.linalg.norm(p - sh_w))

    def best_reach_margin(self, stance: Stance, point_world, arms: Sequence[str]) -> Tuple[float, str]:
        """The largest :meth:`reach_margin` over ``arms`` and its arm, the first arm on a tie (``tabletop.best_reach_margin``'s rule):
        a stance B ranked by whichever arm reaches better (owner 2026-09-27, first-demo v2.2)."""
        best: Optional[Tuple[float, str]] = None
        for a in arms:
            m = self.reach_margin(stance, point_world, a)
            if best is None or m > best[0] + 1e-9:
                best = (m, a)
        return best if best is not None else (self.reach_margin(stance, point_world), self.seg_arm)

    def half_extent_along(self, quat, direction_xy) -> float:
        """Half extent of the target's footprint along a horizontal direction at orientation ``quat``."""
        pts = G.local_points_world(self.hull, (th.zeros(3), quat))[:, :2]
        d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])])
        d = d / th.linalg.norm(d)
        proj = pts @ d
        return float(proj.max() - proj.min()) / 2

    #: below this fingertip height above the floor the cuRobo ground slab flags the padded fingertip spheres (first-demo loop
    #: 2026-09-16: every push pre-pose failed the IK at z_push = 4.5 cm) - the slab is switched off for such a chain's IK
    SLAB_TIP_M = 0.045

    def push_height(self, pitch_deg: float) -> Tuple[float, float]:
        """``(z_push, z_tip)`` of the floor blade: the fingertips at ``push_tip_frac`` (½) of the target's height above the
        floor, at least ``push_tip_clearance_m`` (1.5 cm) and at most 5 cm up; the eef origin ``FINGERTIP_LENGTH·cos(pitch)``
        above the tips. The old rule (eef origin at ¾ of the height, fingertips ≥ 4 cm up for the cuRobo ground slab) ran
        the blade over a 4.5 cm sandal: it caught the strap near the middle and moved the sandal 17 cm of the 29 cm
        (0920-143722, 12 cm short of the goal cell). A blade below ``SLAB_TIP_M`` has its IK solved with the slab off."""
        # the fingertips at half the height (3.6 cm for the 7.2 cm sandal chgcrm: the height the exact IK admits and 0920-154503
        # executed; 1.1 cm had no IK solution at all, 0920-170805) - what the blade meets at that height is modelled per yaw by
        # ``contact_offset_along`` (pod probe 2026-09-20: the sandal's heel face reaches 12.8 cm back only up to z 3.5 cm, its toe
        # 11.4 cm forward up to z 4.5 cm, the strap 2.7 cm behind the root above - so the blade at 3.6 cm pushes it toe-first)
        # i7-contact 2026-11-07: the shared rule (contact_poses.floor_blade_height)
        return CPZ.floor_blade_height(float(self.place["height"]), self.floor_z, pitch_deg, frac=float(self.num("push_tip_frac", 0.5)),
                                      clear=float(self.num("push_tip_clearance_m", 0.015)))

    def flat_turns(self, d2) -> List[Tuple[float, float, str, th.Tensor]]:
        """The flat hand's wrist turns pushing along ``d2`` (:func:`tabletop.flat_hand_quats`), lowest fingertips first: ``(fingertip height
        above the floor, the hand's lowest collision sphere below the eef origin, label, eef quat)``. The fingertips (at the eef origin's
        height: the approach is horizontal) ride ``generation.push_flat_floor_clearance_m`` above that sphere's bottom
        (:func:`tabletop.hand_drop_below`) - the hand never scrapes the floor, and the IK keeps the ground slab."""
        clear = float(self.num("push_flat_floor_clearance_m", PUSH_FLAT_FLOOR_CLEARANCE_M))
        turns = [(TT.hand_drop_below(q), label, q) for label, q in TT.flat_hand_quats(d2)]
        return [(below + clear, below, label, q) for below, label, q in sorted(turns, key=lambda t: t[0])]

    def flat_tip_floor(self, stance: Optional[Stance] = None) -> float:
        """The lowest fingertip height a flat hand reaches from ``stance``'s trunk (:data:`FLAT_TIP_FLOOR_M`; the lowest of the table for
        another trunk or no stance), ``generation.push_flat_tip_min_m`` for every trunk when set."""
        v = self.gp.gen("push_flat_tip_min_m")
        if v is not None:
            return float(v)
        if stance is None:
            return min(FLAT_TIP_FLOOR_M.values())
        return FLAT_TIP_FLOOR_M.get((round(float(stance.translate), 2), round(float(stance.tilt), 2)), min(FLAT_TIP_FLOOR_M.values()))

    def flat_rungs(self, stance: Stance, quat) -> List[float]:
        """The flat hand's fingertip heights above the floor to try from ``stance`` for the target at orientation ``quat``, lowest first: from
        the lowest wrist turn's floor rule (:meth:`flat_turns`) or the trunk's reach floor (:meth:`flat_tip_floor`), whichever is higher,
        ``push_flat_tip_step_m`` apart up to ``push_flat_tip_max_m`` and ``push_flat_top_margin_m`` below the target's top (the tips stay
        on its back face); empty when the floor lies above that cap."""
        lo = max(self.flat_turns((1.0, 0.0))[0][0], self.flat_tip_floor(stance))
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))
        top = float(pts[:, 2].max() - pts[:, 2].min())
        hi = min(float(self.num("push_flat_tip_max_m", PUSH_FLAT_TIP_MAX_M)), top - float(self.num("push_flat_top_margin_m", PUSH_FLAT_TOP_MARGIN_M)))
        step = float(self.num("push_flat_tip_step_m", PUSH_FLAT_TIP_STEP_M))
        n = int(math.floor((hi - lo) / step + 1e-6)) + 1 if hi >= lo else 0
        return [round(lo + k * step, 4) for k in range(n)]

    def flat_chain_clash(self, stance: Stance, poses: Sequence[G.Pose], arm: str) -> Optional[str]:
        """Why the flat hand's poses cannot be reached from ``stance`` whatever the arm does, without an IK (None: let the IK decide): the wrist
        sphere (:data:`FLAT_WRIST_SPHERE_M` around the wrist centre, ``FLAT_WRIST_FROM_EEF_M`` behind an eef pose on its approach axis) in one
        of the base's spheres (:data:`FLAT_BASE_SPHERES`), or the wrist centre farther than :data:`FLAT_WRIST_REACH_M` from the arm's shoulder
        at the stance's trunk."""
        base = th.tensor([c for c, _ in FLAT_BASE_SPHERES], dtype=th.float32)
        reach = th.tensor([r for _, r in FLAT_BASE_SPHERES], dtype=th.float32) + FLAT_WRIST_SPHERE_M
        sh = C.torso_fk(C.torso_q(float(stance.translate), float(stance.tilt), C.TORSO_JOINT_LIMITS))[f"{arm}_shoulder"][0].float()
        bz = P0.base_z(self)
        for p, q in poses:
            w = P0.to_base_frame(G.t(p) - FLAT_WRIST_FROM_EEF_M * C.approach_axis(G.t(q)), stance.base_pose2d, bz)
            if bool((th.linalg.norm(base - w, dim=1) < reach).any()):
                return "wrist in the base"
            if float(th.linalg.norm(w - sh)) > FLAT_WRIST_REACH_M:
                return "wrist beyond the shoulder's reach"
        return None

    def far_pitches(self) -> List[float]:
        """The pitched hand's pitch ladder (``generation.push_under_far_pitches_deg``, default :data:`PUSH_FAR_PITCHES_DEG`)."""
        return [float(v) for v in (self.gp.gen("push_under_far_pitches_deg") or PUSH_FAR_PITCHES_DEG)]

    def pitched_turns(self, pitch_deg: float, d2, com_h: Optional[float] = None) -> List[Tuple[Optional[float], float, str, th.Tensor]]:
        """The pitched hand's wrist turns at ``pitch_deg`` pushing along ``d2`` (:func:`tabletop.pitched_hand_quats`), the hand's lowest part
        nearest the fingertips first: ``(fingertip height above the floor, the lowest collision sphere below the fingertips, label, eef quat)``.
        The fingertips stand ``generation.push_far_tip_m`` up, at least that sphere + ``push_far_floor_clearance_m`` (the hand never scrapes
        the floor, the IK keeps the ground slab) and, given the target's centre-of-mass height ``com_h``, low enough that their leading edge
        stays below it; None when the two bounds cross (the turn cannot push below the centre of mass)."""
        target = float(self.num("push_far_tip_m", PUSH_FAR_TIP_M))
        clear = float(self.num("push_far_floor_clearance_m", PUSH_FAR_FLOOR_CLEARANCE_M))
        rise = math.sin(math.radians(float(pitch_deg)))
        out = []
        for label, q in TT.pitched_hand_quats(pitch_deg, d2):
            drop = TT.pitched_hand_drop(q)
            lo = drop + clear
            hi = math.inf if com_h is None else float(com_h) - TT.flat_tip_half_height(q) * rise - 1e-3
            out.append((max(lo, min(target, hi)) if lo <= hi else None, drop, label, q))
        return sorted(out, key=lambda t: t[1])

    def push_contact_offset(self, quat, direction_xy) -> Optional[float]:
        """:meth:`contact_offset_along` in the band the push meets the target in (the yaw orders by contact offset): the blade's, its
        fingertips at the 30° pitch up to ``BLADE_SPAN_M`` above them; the flat hand's fingertips at the lowest rung any trunk reaches; the
        pitched hand's fingertip edge at its first pitch's first turn (the cross-sections counted for both)."""
        if self.push_hand == "flat":
            tip, _, _, q = self.flat_turns((1.0, 0.0))[0]
            tip = max(tip, self.flat_tip_floor())
            band = TT.flat_tip_half_height(q)
            return self.contact_offset_along(quat, direction_xy, tip - band, tip + band, section=True)
        if self.push_hand == "pitched_far":
            pitch = self.far_pitches()[0]
            tip, _, _, q = self.pitched_turns(pitch, (1.0, 0.0), self.com_height(quat))[0]
            if tip is None:
                return None
            band = TT.flat_tip_half_height(q) * math.sin(math.radians(pitch))
            return self.contact_offset_along(quat, direction_xy, tip - band, tip + band, section=True)
        z_lo = self.push_height(30.0)[1] - self.floor_z
        return self.contact_offset_along(quat, direction_xy, z_lo, z_lo + self.BLADE_SPAN_M)

    def target_com_local(self) -> th.Tensor:
        """The target's centre of mass in its root frame: PhysX's mass-weighted link centres of mass, the centre of the hull's box without
        them (``alignment_axial._com_local``'s rule); read once per bound target."""
        obj = self.target
        cached = self.__dict__.get("_com_local")
        if cached is not None and cached[0] == obj.name:
            return cached[1]
        com = None
        try:
            total, acc = 0.0, th.zeros(3)
            rp, rq = obj.get_position_orientation()
            for link in obj.links.values():
                m = float(link.mass)
                if m <= 0:
                    continue
                lp, lq = link.get_position_orientation()
                acc += m * (lp.float() + C.quat_rotate(lq.float(), G.t(link.center_of_mass)))
                total += m
            if total > 0:
                com = C.quat_rotate(C.qinv(rq.float()), acc / total - rp.float())
        except Exception:  # noqa: BLE001 - no PhysX masses: the box centre
            com = None
        if com is None:
            com = 0.5 * (self.hull.min(dim=0).values + self.hull.max(dim=0).values)
        self._com_local = (obj.name, com)
        return com

    def com_height(self, quat) -> float:
        """The target's centre of mass above its lowest collision point at orientation ``quat``."""
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))
        com = G.local_points_world(self.target_com_local().reshape(1, 3), (th.zeros(3), G.t(quat)))[0]
        return float(com[2] - pts[:, 2].min())

    #: the order of the carried yaws the push chain is tried at (named by the chain-first screen's error)
    PUSH_YAW_ORDER = "largest-offset"

    def push_yaw_rank(self, stance: Stance, quat, off: float) -> float:
        """Rank of a carried yaw for the push chain from ``stance`` (smaller first; ``quat`` = the target's orientation at C,
        ``off`` = the blade's contact offset there): the push-under's LARGEST offset first (the blade goal nearest the base,
        2026-09-20 19:20). Orders the chain-first screen's yaws and the F draws."""
        return -float(off)

    def push_yaw_lock(self) -> Optional[str]:
        """``generation.push_under_yaw`` (:data:`PUSH_UNDER_YAW_LOCKS`; default None or ``"none"`` = every carried yaw)."""
        v = self.gp.gen("push_under_yaw")
        if v is None or str(v) == "none":
            return None
        if str(v) not in PUSH_UNDER_YAW_LOCKS:
            raise Reject("spec", f"generation.push_under_yaw {v!r} is not one of {PUSH_UNDER_YAW_LOCKS}")
        return str(v)

    def push_yaw_tol_deg(self) -> float:
        """heel_first's largest angle between the heel's direction and the push's reverse (``generation.push_under_yaw_tol_deg``; default one
        step of the yaw grid, :meth:`yaws`: 22.5 deg at 16 - the two grid yaws on either side of the exact heel-first yaw)."""
        return float(self.num("push_under_yaw_tol_deg", 360.0 / max(1, int(self.num("n_yaws", 16)))))

    def push_heel(self) -> Tuple[th.Tensor, float, float]:
        """The target's heel for ``heel_first``: ``(h, heel, toe)`` - :func:`footprint_heel` of its footprint at the rest orientation, ``h``
        turned into the root frame (the yaw at C turns it with the target); memoised per bound target and recorded in
        ``template_facts.push_under_yaw``. ``Reject("spec")`` when the ends lie within :data:`PUSH_HEEL_MIN_ASYM_M` of each other."""
        name = getattr(self.target, "name", None)
        memo = self.__dict__.get("_push_heel")
        if memo is not None and memo[0] == name:
            return memo[1]
        h, heel, toe = footprint_heel(G.local_points_world(self.hull, (th.zeros(3), G.t(self.rest))))
        if heel - toe < PUSH_HEEL_MIN_ASYM_M:
            raise Reject("spec", f"generation.push_under_yaw 'heel_first': {name}'s long axis ends {heel:.3f} / {toe:.3f} m from its root - no heel")
        h_local = G.local_points_world(th.tensor([[float(h[0]), float(h[1]), 0.0]]), (th.zeros(3), G.qinv(self.rest)))[0]
        self._push_heel = (name, (h_local, heel, toe))
        self.template_facts["push_under_yaw"] = {"lock": "heel_first", "heel_m": round(heel, 4), "toe_m": round(toe, 4),
                                                 "heel_dir_local": [round(float(v), 4) for v in h_local], "tol_deg": round(self.push_yaw_tol_deg(), 2)}
        return self._push_heel[1]

    def push_yaw_kept(self, quat, d2, off: Optional[float] = None) -> bool:
        """Whether ``generation.push_under_yaw`` lets the push along ``d2`` meet the target at orientation ``quat`` (always without the key).
        ``"heel_first"`` (owner 2026-10-13 ~02:55, under_large#01): the heel (:meth:`push_heel`) points back at the robot - its direction at
        ``quat`` within :meth:`push_yaw_tol_deg` of the push's reverse: the long axis along the push, the toes ahead - and the push's contact
        (:meth:`push_contact_offset`, the rearmost point in the hand's height band; ``off`` when the caller has it) lies on the heel's half,
        at least :data:`PUSH_HEEL_CONTACT_FRAC` of the heel's extent behind the root. The yaw order alone takes the largest offset left: at a
        B whose F_init held no heel-first yaw it pushed the sandal toe first (0.117 m, c1_k1 at 27ee9069), and fingertips above the heel's
        face (3.5 cm) meet the strap 2.7 cm behind the root."""
        if self.push_yaw_lock() is None:
            return True
        h_local, heel, _ = self.push_heel()
        h = G.local_points_world(h_local.reshape(1, 3), (th.zeros(3), G.t(quat)))[0, :2]
        d = th.as_tensor([float(d2[0]), float(d2[1])], dtype=th.float32)
        cos = -float(h @ d) / max(float(th.linalg.norm(h)) * float(th.linalg.norm(d)), 1e-9)
        if cos < math.cos(math.radians(self.push_yaw_tol_deg())) - 1e-6:
            return False
        if off is None:
            off = self.push_contact_offset(quat, d2)
        return off is not None and float(off) >= PUSH_HEEL_CONTACT_FRAC * heel

    def push_yaws_locked(self, yaws: Sequence[float], d_yaw: float, d2, offs: Dict[float, float], place: Stance) -> List[float]:
        """The init yaws of ``yaws`` (the target at ``yaw + d_yaw`` at C, ``offs`` their contact offsets) that :meth:`push_yaw_kept` keeps for the
        push along ``d2`` from ``place``, in order; all of them without ``generation.push_under_yaw``. One INFO line 'push-under yaw lock
        heel_first: kept N of M yaws' when it removes any."""
        lock = self.push_yaw_lock()
        if lock is None:
            return list(yaws)
        kept = [y for y in yaws if self.push_yaw_kept(G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), d2, offs.get(y))]
        if len(kept) < len(yaws):
            TT._count(self, "push_yaw_lock.removed")
            log.info("%s: push-under yaw lock %s: kept %d of %d yaws at C (d %.2f λ %+.2f, carry yaw offset %+.2f): %s, offsets %s m", self.gp.variant, lock, len(kept),
                     len(yaws), place.stand_off or 0.0, place.lateral, d_yaw, [round(y, 2) for y in kept], [round(float(offs.get(y, -1.0)), 3) for y in kept])
        return kept

    def _c_dead_at_b(self, F_live: Sequence[Tuple[float, Any]], b_ik_cache: Dict[Any, Any], place: Stance, manip: Stance, yaw_step: float) -> bool:
        """oracle-fix 2026-11-06 (speed, opt-in ``generation.c_skip_dead_b``): C ``place`` cannot be written from B ``manip`` - every pair of
        ``F_live`` a draw there can take already failed its exact IK at B (:func:`dead_at_b`; the B side does not depend on C, ``b_ik_cache``
        is the C loop's). The push-under yaw lock's kept yaws at this C (:meth:`push_yaws_locked`'s rule, carry yaw offset as the C loop's)
        narrow the pairs. under_large_scale_01 (sandal under the bed, trainfast crop4 / full4): 41 of 42 C tries of a B exhausted by
        'B: [pre-grasp, grasp] exact IK (cached)' alone after their push chains (3.7-4.7 s each) and F_goal screens; the chains after B
        was dead were 365 of 508 in a 3005 s attempt."""
        d_yaw = 0.0 if str(self.gp.gen("carry_yaw", "relative")) == "world" else float(place.yaw - manip.yaw)
        end = lambda y0: self.end_aligned_init_yaw(y0, d_yaw)  # noqa: E731
        if dead_at_b(F_live, b_ik_cache, yaw_step, end):
            return True
        if not self.push_under or self.push_yaw_lock() is None:
            return False
        d_push = th.as_tensor(place.extras["push_dir"], dtype=th.float32)
        yaws = sorted({float(y) for y, _ in F_live})
        kept = []
        for y in yaws:
            q = G.qmul(G.quat_from_yaw(y + d_yaw), self.rest)
            o = self.push_contact_offset(q, d_push)
            if self.push_yaw_kept(q, d_push, -1.0 if o is None else o):
                kept.append(y)
        if not kept or len(kept) == len(yaws):  # none kept: the C loop's lock rule skips the C itself; all kept: judged above
            return False
        return dead_at_b(F_live, b_ik_cache, yaw_step, end, {int(round(y / yaw_step)) for y in kept})

    def push_contact_forms(self) -> List[str]:
        """``generation.push_under_contact`` (owner 2026-10-13 ~04:30 item 2; :data:`PUSH_CONTACT_FORMS`, one name or a list in preference
        order): the blade hand's contact forms in the order :meth:`_push_under_chain` tries them, today's ``"blade"`` last unless the list
        places it (so the key never loses a chain that solves today); ``["blade"]`` without the key. ``Reject("spec")`` for an unknown form."""
        v = self.gp.gen("push_under_contact")
        if v is None:
            return ["blade"]
        forms = [v] if isinstance(v, str) else list(v) if isinstance(v, (list, tuple)) else None
        if not forms or any(f not in PUSH_CONTACT_FORMS for f in forms):
            raise Reject("spec", f"generation.push_under_contact {v!r}: one of {PUSH_CONTACT_FORMS} or a list of them")
        return list(dict.fromkeys(forms + ([] if "blade" in forms else ["blade"])))

    def push_strokes_record(self, distance_m: float, form: Optional[str] = None) -> Dict[str, Any]:
        """``generation.push_under_stroke_m`` (owner 2026-10-13 ~04:30 item 2; default none = one stroke): the push record's ``strokes`` - the
        plan :func:`push_stroke_plan` makes for the push's distance (at most ``push_under_strokes_max`` strokes) with the oracle's knobs
        ``push_under_stroke_backoff_m`` (0.03), ``push_under_reaim_max_m`` (0.03), ``push_under_reaim_yaw_gain`` (m/rad, 0) and
        ``push_under_reaim_yaw_deadband_deg`` (2); ``{}`` without the key, and for a chain of the contact form ``"two_point"`` (``form``): the
        sim check 17838968 re-aimed the open hand at the line through the sandal's root and one fingertip met the round heel first - the
        sandal turned 80 deg, where the strokes with the blade turned it -2 deg. ``Reject("spec")`` for a stroke that is not a positive
        length."""
        v = self.gp.gen("push_under_stroke_m")
        if v is None:
            return {}
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v) > 0.0:
            raise Reject("spec", f"generation.push_under_stroke_m {v!r}: a positive length (m)")
        if form == "two_point":
            log.info("%s: push-under strokes (generation.push_under_stroke_m %.3f) not recorded for the two-point hand: one stroke", self.gp.variant, float(v))
            return {}
        plan = push_stroke_plan(distance_m, float(v), int(self.num("push_under_strokes_max", PUSH_STROKES_MAX)),
                                backoff_m=float(self.num("push_under_stroke_backoff_m", 0.03)), reaim_max_m=float(self.num("push_under_reaim_max_m", 0.03)),
                                yaw_gain_m_per_rad=float(self.num("push_under_reaim_yaw_gain", 0.0)),
                                yaw_deadband_rad=math.radians(float(self.num("push_under_reaim_yaw_deadband_deg", 2.0))))
        log.info("%s: push-under strokes (generation.push_under_stroke_m %.3f): %d strokes over %.3f m, the object's progress after each %s m, back-off %.3f m, "
                 "re-aim within %.3f m, yaw gain %.3f m/rad", self.gp.variant, float(v), plan["n"], float(distance_m), plan["ends_m"], plan["backoff_m"], plan["reaim_max_m"],
                 plan["yaw_gain_m_per_rad"])
        return {"strokes": plan}

    def push_drop_ahead_pref(self) -> Optional[float]:
        """The drop cell's preferred distance ahead of the base in the C order (``generation.push_under_drop_ahead_pref_m``; the pitched
        hand's :data:`PUSH_FAR_DROP_AHEAD_PREF_M` without it)."""
        v = self.gp.gen("push_under_drop_ahead_pref_m")
        if v is None and self.relation == "under" and self.push_hand == "pitched_far":
            return PUSH_FAR_DROP_AHEAD_PREF_M
        return None if v is None else float(v)

    def _push_under_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The straight floor-level push from the drop position to the goal cell, exact IK from stance C (closed gripper,
        floor and target ignored, partner present): blade behind the target along the push direction (stand-off from
        ``tabletop.blade_standoff`` over the target's face height), approach pitched toward the push direction (0 / 30 /
        45 / 60°, either wrist turn), poses ``[pre, start, mid, goal]``, with the segment arm (``programs.py``: the push-under blade
        is the grasp arm). ``generation.push_under_palm_contact``: a face past the finger section is pushed by the palm of the vertical
        blade (``palm_contact`` in the fields). Returns the record fields or None. ``generation.push_under_hand`` = ``"flat"`` / ``"pitched_far"``:
        the flat hand's (:meth:`_push_under_flat_chain`) / the pitched hand's chain (:meth:`_push_under_pitched_chain`) instead, whatever the
        blade's keys (``push_under_palm_contact`` too) say."""
        if self.push_hand == "flat":
            return self._push_under_flat_chain(place, quat, drop_xy, goal_xy, d2)
        if self.push_hand == "pitched_far":
            return self._push_under_pitched_chain(place, quat, drop_xy, goal_xy, d2)
        floor, target = self.obj("floor"), self.target
        arm = self.seg_arm
        d2 = th.as_tensor([float(d2[0]), float(d2[1])])
        d2 = d2 / th.linalg.norm(d2)
        d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
        n3 = th.tensor([-float(d2[1]), float(d2[0]), 0.0])  # across the push: the closing axis of the two-point / turned blade
        height = float(self.place["height"])
        cd = th.as_tensor([float(drop_xy[0]), float(drop_xy[1])])
        gc = th.as_tensor([float(goal_xy[0]), float(goal_xy[1])])
        distance = float(th.linalg.norm(gc - cd))
        why: List[str] = []
        half_d: Optional[float] = self.half_extent_along(quat, d2)  # the footprint's, for the log when no pitch gets a blade
        palm_ok = bool(self.gp.gen("push_under_palm_contact", False))
        forms = self.push_contact_forms()
        keyed = forms != ["blade"]
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat))) if keyed else None
        for form, pitch in ((f, p) for f in forms for p in TT.approach_pitch_order(None)):
            z_push, z_tip = self.push_height(pitch)
            tag = f"{form} " if keyed else ""
            fingers, rec = C.FINGER_CLOSED, ({"push_contact_form": "blade"} if keyed else {})
            if form == "blade":
                so = TT.blade_standoff((self.floor_z, self.floor_z + height), z_tip, pitch, PUSH_UNDER_CLEARANCE_M)
                # owner 2026-10-12 ~19:40: the goal counts, not which part of the hand meets the face - the vertical blade's palm pushes a
                # face past the finger section, standing off by the palm's half extent along the push (per wrist turn, below)
                palm = so is None and palm_ok and pitch == 0.0 and self.floor_z + height > z_tip + TT.PALM_FROM_TIP
                if so is None and not palm:
                    why.append(f"{tag}pitch {pitch:.0f}: face reaches the palm")
                    continue
                standoff = 0.0 if palm else so[0]
                # the blade meets the rearmost collision point within its height band (a sandal's strap, a box's face), not the
                # footprint's rear edge; the (generation) ``push_contact_offset_m`` overrides the model. The palm meets the face anywhere
                # above the fingertips (and comes down beside it from the pre-pose)
                off_cfg = self.gp.gen("push_contact_offset_m")
                z_hi = height if palm else z_tip - self.floor_z + self.BLADE_SPAN_M
                off_m = float(off_cfg) if off_cfg is not None else self.contact_offset_along(quat, d2, z_tip - self.floor_z, z_hi)
                closing = d3
            else:
                # owner 2026-10-13 ~04:30 item 2: the closing axis across the push - both fingertips on the back (open) or both fingers
                # side by side (closed); the hand stands where its first part meets the back (:func:`push_form_contact`)
                palm = False
                so = TT.blade_standoff((self.floor_z, self.floor_z + height), z_tip, pitch, PUSH_UNDER_CLEARANCE_M, side=True)
                if so is None:
                    why.append(f"{tag}pitch {pitch:.0f}: face reaches the palm")
                    continue
                standoff = so[0]
                tip = z_tip - self.floor_z
                fc = push_form_contact(form, pts, d2, tip, tip + self.BLADE_SPAN_M, opening=self.gp.gen("push_under_two_point_opening_m"))
                off_m = None if fc is None else fc["off_m"]
                if fc is not None:
                    rec = {"push_contact_form": form, "contact_strips_m": fc["strips_m"], "face_half_m": round(fc["face_half_m"], 5)}
                    if form == "two_point":
                        fingers = fc["fingers_m"]
                        rec.update({"fingers_opening_m": round(fingers, 5), "contact_width_m": round(fc["opening_m"], 4)})
                closing = n3
            if off_m is None:
                why.append(f"{tag}pitch {pitch:.0f}: nothing of the target within the blade's height band")
                continue
            half_d = off_m
            a_vec = TT.pitched_approach(pitch, d2)  # pitched toward the push direction (= the robot's forward for a frontal stance, its arm's side for a side stance)
            # the blade's goal runs ``push_goal_extra_m`` (3 cm) past the model's: the oracle's differential-IK leg stops 2-3 cm
            # short of its last waypoint (0920-143722 / 154503: eef 0.704-0.716 m ahead for a 0.736 m goal)
            extra = float(self.num("push_goal_extra_m", 0.03))
            low = (z_tip - self.floor_z) < self.SLAB_TIP_M
            props = []
            for rank, c_sign in enumerate((1.0, -1.0)):
                q = C.eef_quat(a_vec, c_sign * closing)
                # the palm: the wrist camera reaches 1.8 cm further toward the face than the knuckles - each turn has its stand-off
                back = d3 * (TT.palm_half_extent(q, d3) + PUSH_UNDER_CLEARANCE_M) if palm else None
                # i7-contact 2026-11-07: the shared chain (contact_poses.floor_push_chain)
                chain = CPZ.floor_push_chain(cd, gc, d2, z_push, half_d + standoff, extra, q, pre=("approach",), shift=back)
                props.append(O.Proposal(f"{tag}pitch {pitch:.0f}{'' if c_sign > 0 else '-'}", rank, c_sign, chain))

            def evaluate(p, low=low, fingers=fingers):
                with self.ikc.ignoring([floor, target], ground=not low):
                    res = self.ikc.exact(p.chain, place, arm, cache=False, fingers=fingers, q_first=p.q_first)
                return bool(res.ok), res, None, str(res.reason)

            # the two wrist turns by the orientation rule (owner 2026-09-27); the camera is checked at the blade's start, where the
            # target and the floor left the IK's world
            chosen, decision = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(place), camera_idx=[1], stage=f"push-under {tag}pitch {pitch:.0f}",
                                        downstream=lambda s, ground=not low, fingers=fingers: self.push_pinned_check(place, s, arm, ground, fingers=fingers))
            if chosen is not None:
                pre, start_p, mid_p, goal_p = chosen.proposal.chain
                if palm:
                    standoff = TT.palm_half_extent(start_p[1], d3) + PUSH_UNDER_CLEARANCE_M
                    log.info("%s: push-under by the palm (generation.push_under_palm_contact): stand-off %.3f m (wrist %+.0f), contact offset %.3f m over the face "
                             "above the tips", self.gp.variant, standoff, chosen.proposal.payload, half_d)
                if keyed:
                    TT._count(self, f"push_contact_form.{form}")
                    log.info("%s: push-under contact form %s (generation.push_under_contact %s): pitch %.0f, wrist %+.0f, stand-off %.3f m, contact offset %.3f m%s "
                             "(C d %.2f λ %+.2f)", self.gp.variant, form, forms, pitch, chosen.proposal.payload, standoff, half_d,
                             f", fingertips {rec['contact_width_m']:.3f} m apart" if form == "two_point" else "", place.stand_off or 0.0, place.lateral)
                return {"pre": pre, "start": start_p, "mid": mid_p, "goal": goal_p, "direction": d3, "distance_m": distance, "pitch_deg": float(pitch),
                        "blade_standoff_m": float(standoff), "push_height_m": float(z_push - self.floor_z), "half_extent_m": half_d, "wrist_sign": chosen.proposal.payload,
                        "fingertip_z_m": float(z_tip - self.floor_z), "ground_ignored": bool(low), "orient": decision, "seed": push_chain_seed(self.kit, chosen, arm),
                        **({"palm_contact": True} if palm else {}), **({"record": rec} if rec else {})}
            why.extend(f"{o['label']}: {o.get('why', '')}" for o in decision["options"] if not o.get("ok"))
        log.info("%s: push-under chain unreachable from stance C (d %.2f λ %+.2f trunk %s; start %s, goal %s in base): %s", self.gp.variant, place.stand_off or 0.0,
                 place.lateral, (place.translate, round(place.tilt, 2)),
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(cd[0]), float(cd[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(gc[0]), float(gc[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 "; ".join(why))
        return None

    def _push_under_flat_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The flat hand's push chain (``generation.push_under_hand`` = ``"flat"``, owner 2026-10-12 ~19:40 / ~20:55): the blade's contact point
        and straight push with the hand turned flat, the closed fingers forward along the push (the pitched blade at 90°), the fingertips at
        the lowest height the arm reaches - the rungs of :meth:`flat_rungs` lowest first (the arm's wrist centre never comes below
        :data:`FLAT_TIP_FLOOR_M`), and at each rung the wrist turns of :meth:`flat_turns` whose floor rule admits it (the closing axis lateral
        or vertical, each twin); the first rung with a solved chain wins. A turn whose wrist centre would sit in the base or beyond the
        shoulder's reach (:meth:`flat_chain_clash`) is not IK-solved; at most ``push_flat_rungs_max`` rungs are. Only the tips touch, so the
        eef origin stands ``FINGERTIP_LENGTH`` + ``PUSH_UNDER_CLEARANCE_M`` behind the rearmost collision point within the fingertips' height
        band (:meth:`contact_offset_along`) and the pre-pose ``push_flat_pre_dist_m`` behind the start. No centre-of-mass rule: the goal check
        decides, and the contact height is logged and recorded against the target's centre of mass. Exact IK from stance C, closed gripper,
        the floor object and the target ignored, the ground slab kept; the turn by the orientation rule. Returns :meth:`_push_under_chain`'s
        fields (``pitch_deg`` 90, ``blade_standoff_m`` the fingertip stand-off, ``wrist_sign`` the turn's label) + ``record`` (the push
        record's flat-hand facts and ``palm_clear``: the oracle's palm clearance of a top-down blade leaves it alone) or None."""
        floor, target = self.obj("floor"), self.target
        arm = self.seg_arm
        d2 = th.as_tensor([float(d2[0]), float(d2[1])])
        d2 = d2 / th.linalg.norm(d2)
        d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
        cd = th.as_tensor([float(drop_xy[0]), float(drop_xy[1])])
        gc = th.as_tensor([float(goal_xy[0]), float(goal_xy[1])])
        distance = float(th.linalg.norm(gc - cd))
        turns = self.flat_turns(d2)
        standoff = TT.flat_hand_standoff(PUSH_UNDER_CLEARANCE_M)
        extra = float(self.num("push_goal_extra_m", 0.03))
        pre_d = float(self.num("push_flat_pre_dist_m", PUSH_FLAT_PRE_DIST_M))
        com_h = self.com_height(quat)
        off_cfg = self.gp.gen("push_contact_offset_m")
        rungs = self.flat_rungs(place, quat)
        n_max = int(self.num("push_flat_rungs_max", PUSH_FLAT_RUNGS_MAX))

        def evaluate(p):
            with self.ikc.ignoring([floor, target]):  # the ground slab stays: the hand's lowest sphere rides above the floor
                res = self.ikc.exact(p.chain, place, arm, cache=False, fingers=C.FINGER_CLOSED, q_first=p.q_first)
            return bool(res.ok), res, None, str(res.reason)

        why: List[str] = [] if rungs else [f"no rung from the reach floor {self.flat_tip_floor(place):.3f} m to the cap"]
        solved = 0
        for k, tip in enumerate(rungs):
            props: List[O.Proposal] = []
            skipped: Dict[str, int] = {}
            for floor_tip, below, label, q in turns:
                if tip < floor_tip - 1e-9:
                    continue  # the hand's lowest sphere would come within the floor clearance
                band = TT.flat_tip_half_height(q)
                off = float(off_cfg) if off_cfg is not None else self.contact_offset_along(quat, d2, tip - band, tip + band, section=True)
                if off is None:
                    skipped["nothing of the target within the fingertips' band"] = skipped.get("nothing of the target within the fingertips' band", 0) + 1
                    continue
                z = self.floor_z + tip
                chain = CPZ.floor_push_chain(cd, gc, d2, z, off + standoff, extra, q, pre=("back", pre_d))  # i7-contact: the shared chain
                clash = self.flat_chain_clash(place, chain, arm)
                if clash is not None:
                    skipped[clash] = skipped.get(clash, 0) + 1
                    continue
                facts = {"wrist_turn": label, "fingertip_z_m": float(tip), "palm_drop_m": float(below), "half_extent_m": float(off),
                         "fingertip_band_m": [float(tip - band), float(tip + band)]}
                props.append(O.Proposal(f"flat {label}", len(props), facts, chain))
            why.extend(f"tips {tip:.3f}: {r} ({n} turn{'s' if n > 1 else ''})" for r, n in skipped.items())
            if not props:
                continue
            solved += 1
            chosen, decision = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(place), camera_idx=[1], stage=f"push-under flat hand, tips {tip:.3f}",
                                        downstream=lambda s: self.push_pinned_check(place, s, arm, True))
            if chosen is not None:
                pre, start_p, mid_p, goal_p = chosen.proposal.chain
                f = chosen.proposal.payload
                log.info("%s: flat-hand contact height %.3f m vs centre of mass %.3f m (rung %d of %d from %.3f m, wrist turn %s; C d %.2f λ %+.2f trunk %s)",
                         self.gp.variant, tip, com_h, k + 1, len(rungs), rungs[0], f["wrist_turn"], place.stand_off or 0.0, place.lateral,
                         (place.translate, round(place.tilt, 2)))
                return {"pre": pre, "start": start_p, "mid": mid_p, "goal": goal_p, "direction": d3, "distance_m": distance, "pitch_deg": 90.0,
                        "blade_standoff_m": float(standoff), "push_height_m": f["fingertip_z_m"], "half_extent_m": f["half_extent_m"], "wrist_sign": f["wrist_turn"],
                        "fingertip_z_m": f["fingertip_z_m"], "ground_ignored": False, "orient": decision, "seed": push_chain_seed(self.kit, chosen, arm),
                        "record": {"push_hand": "flat", "palm_clear": True, "wrist_turn": f["wrist_turn"], "palm_drop_m": f["palm_drop_m"],
                                   "fingertip_band_m": f["fingertip_band_m"], "contact_height_m": float(tip), "target_com_height_m": float(com_h)}}
            why.extend(f"tips {tip:.3f} {o['label']}: {o.get('why', '')}" for o in decision["options"] if not o.get("ok"))
            if solved >= n_max:
                why.append(f"{n_max} rungs IK-solved, the ladder stops")
                break
        log.info("%s: push-under flat-hand chain unreachable from stance C (d %.2f λ %+.2f trunk %s; rungs %s, centre of mass %.3f m): %s", self.gp.variant,
                 place.stand_off or 0.0, place.lateral, (place.translate, round(place.tilt, 2)), [f"{r:.3f}" for r in rungs], com_h, "; ".join(why)[:1500])
        return None

    def _push_under_pitched_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The pitched hand's push chain (``generation.push_under_hand`` = ``"pitched_far"``, owner 2026-10-12 ~20:55 item C): the blade's
        contact point and straight push with the closed hand pitched 60-70° from straight down toward the push (:meth:`far_pitches`, tried in
        turn), the palm leaning away from the face and the fingertips low on it. Per pitch the wrist turns of :meth:`pitched_turns` (the
        fingertips ``push_far_tip_m`` up, the hand's lowest sphere above the floor, their leading edge below the target's centre of mass); the
        leading edge touches the rearmost collision point within the fingertips' height band (:meth:`contact_offset_along`, cross-sections
        counted), the eef origin standing off for it (:func:`tabletop.pitched_tip_offsets`); the palm test (:func:`pitched_palm_gap`: every
        sphere of the hand behind the object's back at its height) replaces ``blade_standoff``'s finger-section rule; the pre-pose
        ``push_far_pre_m`` back and up from the start. Exact IK from stance C, closed gripper, the floor object and the target ignored, the
        ground slab kept; the turn by the orientation rule. Returns :meth:`_push_under_chain`'s fields (``wrist_sign`` the turn's label) +
        ``record`` (``push_hand``, ``palm_clear``: the oracle's palm clearance of a top-down blade leaves it alone; the contact and
        centre-of-mass heights) or None."""
        floor, target = self.obj("floor"), self.target
        arm = self.seg_arm
        d2 = th.as_tensor([float(d2[0]), float(d2[1])])
        d2 = d2 / th.linalg.norm(d2)
        d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
        cd = th.as_tensor([float(drop_xy[0]), float(drop_xy[1])])
        gc = th.as_tensor([float(goal_xy[0]), float(goal_xy[1])])
        distance = float(th.linalg.norm(gc - cd))
        extra = float(self.num("push_goal_extra_m", 0.03))
        com_h = self.com_height(quat)
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))
        off_cfg = self.gp.gen("push_contact_offset_m")
        pre_back, pre_up = (float(v) for v in (self.gp.gen("push_far_pre_m") or PUSH_FAR_PRE_M))
        why: List[str] = []
        first: Optional[O.Proposal] = None
        for pitch in self.far_pitches():
            rise_p = math.sin(math.radians(pitch))
            props: List[O.Proposal] = []
            for tip, drop, label, q in self.pitched_turns(pitch, d2, com_h):
                if tip is None:
                    why.append(f"pitch {pitch:.0f} {label}: the hand's lowest sphere {drop:.3f} m below the tips keeps their edge above the centre of mass ({com_h:.3f} m)")
                    continue
                standoff, rise, h = TT.pitched_tip_offsets(pitch, q, PUSH_UNDER_CLEARANCE_M)
                band = (tip - h * rise_p, tip + h * rise_p)
                off = float(off_cfg) if off_cfg is not None else self.contact_offset_along(quat, d2, band[0], band[1], section=True)
                if off is None:
                    why.append(f"pitch {pitch:.0f} {label}: nothing of the target within the fingertips' band {band[0]:.3f}-{band[1]:.3f} m")
                    continue
                gap, part = pitched_palm_gap(pts, d2, th.tensor([-float(d2[0]) * (off + standoff), -float(d2[1]) * (off + standoff), tip + rise]), q)
                if gap < 0.0:
                    why.append(f"pitch {pitch:.0f} {label}: the face reaches the {part} ({gap:+.3f} m)")
                    continue
                z = self.floor_z + tip + rise
                chain = CPZ.floor_push_chain(cd, gc, d2, z, off + standoff, extra, q, pre=("back_up", pre_back, pre_up))  # i7-contact: the shared chain
                facts = {"wrist_turn": label, "pitch_deg": float(pitch), "fingertip_z_m": float(tip), "push_height_m": float(tip + rise), "half_extent_m": float(off),
                         "standoff_m": float(standoff), "contact_height_m": float(band[1]), "fingertip_band_m": [float(band[0]), float(band[1])],
                         "palm_gap_m": float(gap) if math.isfinite(gap) else None, "palm_part": part or None, "hand_low_m": float(tip - drop)}
                props.append(O.Proposal(f"pitch {pitch:.0f} {label}", len(props), facts, chain))
            if not props:
                continue
            first = first or props[0]

            def evaluate(p):
                with self.ikc.ignoring([floor, target]):  # the ground slab stays: the hand's lowest sphere rides above the floor
                    res = self.ikc.exact(p.chain, place, arm, cache=False, fingers=C.FINGER_CLOSED, q_first=p.q_first)
                return bool(res.ok), res, None, str(res.reason)

            chosen, decision = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(place), camera_idx=[1], stage=f"push-under pitched hand {pitch:.0f}",
                                        downstream=lambda s: self.push_pinned_check(place, s, arm, True))
            if chosen is not None:
                pre, start_p, mid_p, goal_p = chosen.proposal.chain
                f = chosen.proposal.payload
                base_z = P0.base_z(self)
                log.info("%s: push-under pitched hand SOLVED (d %.2f λ %+.2f trunk %s): pitch %.0f %s, tips %.3f m, contact %.3f m, centre of mass %.3f m, hand low %.3f m, "
                         "palm gap %s (%s), stand-off %.3f m, contact offset %.3f m, start %s -> goal %s in base", self.gp.variant, place.stand_off or 0.0, place.lateral,
                         (place.translate, round(place.tilt, 2)), f["pitch_deg"], f["wrist_turn"], f["fingertip_z_m"], f["contact_height_m"], com_h, f["hand_low_m"],
                         "none" if f["palm_gap_m"] is None else f"{f['palm_gap_m']:.3f} m", f["palm_part"], f["standoff_m"], f["half_extent_m"],
                         [round(float(v), 3) for v in P0.to_base_frame(start_p[0], place.base_pose2d, base_z).tolist()],
                         [round(float(v), 3) for v in P0.to_base_frame(goal_p[0], place.base_pose2d, base_z).tolist()])
                return {"pre": pre, "start": start_p, "mid": mid_p, "goal": goal_p, "direction": d3, "distance_m": distance, "pitch_deg": f["pitch_deg"],
                        "blade_standoff_m": f["standoff_m"], "push_height_m": f["push_height_m"], "half_extent_m": f["half_extent_m"], "wrist_sign": f["wrist_turn"],
                        "fingertip_z_m": f["fingertip_z_m"], "ground_ignored": False, "orient": decision, "seed": push_chain_seed(self.kit, chosen, arm),
                        "record": {"push_hand": "pitched_far", "palm_clear": True, "wrist_turn": f["wrist_turn"], "contact_height_m": f["contact_height_m"],
                                   "target_com_height_m": float(com_h), "palm_gap_m": f["palm_gap_m"], "hand_low_m": f["hand_low_m"],
                                   "fingertip_band_m": f["fingertip_band_m"]}}
            why.extend(f"{o['label']}: {o.get('why', '')}" for o in decision["options"] if not o.get("ok"))
        ends = (f"start {[round(float(v), 2) for v in P0.to_base_frame(first.chain[1][0], place.base_pose2d, P0.base_z(self)).tolist()]}, goal "
                f"{[round(float(v), 2) for v in P0.to_base_frame(first.chain[3][0], place.base_pose2d, P0.base_z(self)).tolist()]} in base" if first is not None else "no turn")
        log.info("%s: push-under pitched-hand chain unreachable from stance C (d %.2f λ %+.2f trunk %s; eef %s): %s", self.gp.variant, place.stand_off or 0.0,
                 place.lateral, (place.translate, round(place.tilt, 2)), ends, "; ".join(why))
        return None

    def push_pinned_check(self, stance: Stance, scored, arm: str, ground: bool, fingers: float = C.FINGER_CLOSED) -> Tuple[bool, str]:
        """The push-under chain's ``orient.choose`` downstream (owner 2026-10-13 ~03:15 item 1): the configuration the goal check pins
        (:meth:`_push_under_goal_chain`) checked when the chain is accepted. ``scored``'s pre-pose configuration as the record keeps it
        (:func:`push_chain_seed`), on ``stance``'s locked joints with the gripper closed, must be a collision-free witness of the pre-pose
        in the chain's world (the floor object and the target out, the ground slab per ``ground``) - the goal check's own seed test, so
        an option failing it would fail that test there. A wrist-flipped option mapped from a remembered solution of the same world pose
        solved from another stance C (the C candidates share the drop cell; ``IKChecker.solutions`` is keyed by the pose alone) keeps that
        stance's base / trunk joints, and ``exact`` pins it by FK alone: under#02 pitched_c2_k0 recorded such chains on 5 of 7 attempts,
        each 'seeded failed' at the goal check. A failing option is passed over for the next one in the order. With
        ``generation.push_pinned_settle_m`` (m, default off) the goal check's seeded re-solve must also pass with the push goal that much
        farther along the push (the settled goal a few mm out, the chain's reach-limited end). A chain without every configuration
        (no seed recorded) passes as before, every chain with ``generation.push_pinned_start_check`` = ``"off"``. ``fingers``: the chain's
        finger joints (a two-point push's opening, ``generation.push_under_contact``). Returns ``(ok, why)``; one INFO line 'push-under
        pinned-start check: ...' per rejection."""
        if str(self.gp.gen("push_pinned_start_check", "on")) == "off":
            return True, ""
        seed = push_chain_seed(self.kit, scored, arm)
        if not seed:
            return True, ""
        chain = scored.proposal.chain
        q_pin = self.ikc.q_locked(stance, fingers=fingers).clone()
        q_pin[self.kit.robot.arm_control_idx[arm]] = th.as_tensor(seed["chain_arm_q"][0], dtype=q_pin.dtype)
        shift = self.gp.gen("push_pinned_settle_m")
        why = ""
        with self.ikc.ignoring([self.obj("floor"), self.target], ground=ground):
            if not self.ikc.collision_free_at(q_pin, chain[0], arm):
                fk = getattr(self.ikc, "fk_residual", None)
                off = f" (FK {1000.0 * float(fk(q_pin, chain[0], arm)):.1f} mm from the pre-pose)" if fk is not None else ""
                why = f"the recorded pre-pose configuration is off the pre-pose or collides on this stance{off}"
            elif shift is not None and float(shift) > 0.0:
                d = G.t(chain[3][0]).float() - G.t(chain[1][0]).float()
                d[2] = 0.0
                goal = G.t(chain[3][0]).float() + float(shift) * d / th.linalg.norm(d).clamp(min=1e-9)
                saved = dict(self.ikc.solutions)
                try:
                    res = self.ikc.exact([chain[0], chain[1], (0.5 * (G.t(chain[1][0]).float() + goal), chain[2][1]), (goal, chain[3][1])], stance, arm, cache=False,
                                         fingers=fingers, q_first=q_pin)
                finally:
                    self.ikc.solutions.clear()
                    self.ikc.solutions.update(saved)
                if not res.ok:
                    why = f"the seeded re-solve fails with the push goal {1000.0 * float(shift):.0f} mm farther: {str(res.reason)[:200]}"
        if not why:
            return True, ""
        TT._count(self, "push_pinned_start.rejected")
        log.info("%s: push-under pinned-start check: %s [%s, %s arm] passed over - %s (C d %.2f λ %+.2f trunk %s)", self.gp.variant, scored.proposal.label,
                 "wrist-flipped" if scored.proposal.source == "map" else "solved", arm, why, stance.stand_off or 0.0, stance.lateral,
                 (stance.translate, round(stance.tilt, 2)))
        return False, why

    def clearance_at(self, xy) -> Optional[float]:
        cm = self.place.get("clearance")
        if not cm:
            return None
        res, (ox, oy) = float(cm["resolution"]), (float(cm["origin"][0]), float(cm["origin"][1]))
        i, j = int((float(xy[0]) - ox) / res), int((float(xy[1]) - oy) / res)
        g = cm["grid"]
        if 0 <= i < len(g) and 0 <= j < len(g[i]) and g[i][j] is not None:
            return float(g[i][j])
        return None

    # ------------------------------------------------------------------ reach bands (B3)

    def make_band_spec(self) -> TT.BandSpec:
        c = self.choice.models["target"]
        return TT.BandSpec(object=f"{c.category}/{c.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull, rest_quat=self.rest,
                           scale=list(c.scale) if c.scale else None, height=self.target_dims()[1])

    def band(self, kind: str, support_obj=None, under_partner=None, arm: Optional[str] = None) -> Optional[RB.ReachBands]:
        """The reach band of a factor set (§5.6): tabletop mode (init support / the partner's top) through
        ``tabletop.ensure_bands`` / ``compute_bands`` (``ReachBands.load`` on a hit, ``BandComputer`` on a miss), the
        floor modes (``--support floor [--under partner]``) through :meth:`compute_band` on a miss; memoised per
        (kind, support, under) for the template. ``ISG_REACH_BANDS_COMPUTE=0`` turns a miss into ``Reject("missing_band")``.
        ``arm`` (owner 2026-09-27, first-demo v2.2): that arm's band - the band above when it is that arm's, else
        :meth:`other_arm_band`. Under v2 only (``reach_bands.v2_only``, i7-floor 2026-11-07): :meth:`v2_band` - the v2 reach maps, never a
        legacy band."""
        if RB.v2_only():
            return self.v2_band(kind, support_obj, under_partner, arm)
        c = self.choice.models["target"]
        obj = f"{c.category}/{c.model}"
        support = "floor" if kind == "floor" else support_obj.name
        under = under_partner.name if under_partner is not None else None
        memo = self.__dict__.setdefault("_bands", {})
        if arm is not None:
            base = self.band(kind, support_obj, under_partner)
            if base is None or base.arm == arm:
                return base
            key = (kind, support, under, arm)
            if key not in memo:
                memo[key] = self.other_arm_band(base, kind, support, under, arm)
            return memo[key]
        key = (kind, support, under)
        if key in memo:
            return memo[key]
        # the init support's interior surface (tabletop.support_surface) has its own band; the partner's top never does
        surface = TT.surface_band_kwargs(self) if (kind == "tabletop" and support_obj is self.surface.get("obj")) else {}
        try:
            rb = RB.ReachBands.load(self.world.scene_model, support, obj, self.grasp_key, under=under, scale=c.scale, **surface)
            log.info("%s: reach band %s (%s) from %s", self.gp.variant, support, under or "-", rb.path)
            # a band computed before the A6 posture retry can be empty (every trunk stance collided with the untucked
            # arms: the fleet's floor-under bands of the beds) — an empty band is a miss, recomputed with the current script
            # oracle-fix 2026-11-04 (owner item 5): a band computed for another grasp set (the legacy rows vs their v2 replacement under the
            # same legacy key, ISPATIALGYM_GRASP_LIBRARY_V2) is stale - a miss; bands without recorded grasp names are kept as before
            names = rb.meta.get("grasp_names")
            grasps_ = getattr(self, "grasps", None) or []  # (a sampler without its grasp list keeps the band, as before)
            if names and grasps_ and sorted(names) != sorted(g.name for g in grasps_) and os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") != "0":
                log.info("%s: cached reach band %s (%s) was computed for another grasp set (%d names vs %d now): recomputing", self.gp.variant, support,
                         under or "-", len(names), len(grasps_))
                raise FileNotFoundError("grasp set changed")
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0 and os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") != "0":
                log.info("%s: cached reach band %s (%s) is empty (computed with %s arms, no A6 retry): recomputing in-process", self.gp.variant, support, under or "-",
                         rb.meta.get("arms"))
                raise FileNotFoundError("empty band")
        except FileNotFoundError:
            if os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") == "0" or surface:
                raise Reject("missing_band", f"no reach band for scene={self.world.scene_model} support={support} object={obj} under={under} scale={c.scale}"
                                             + (f" surface_z={surface['surface_z']:.2f}" if surface else ""))
            if kind == "tabletop":
                top = self.surface.get("top") if (self.surface.get("obj") is support_obj and self.surface.get("top") is not None) else self.support_top(support_obj, ignore=[self.target])
                rb = TT.compute_bands(self, self.make_band_spec(), top)
            else:
                rb = self.compute_band(kind, support_obj, under_partner)
        if rb is not None and int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
            # still empty after the recompute (the band stance itself collides at every trunk — e.g. the band script's
            # under stance at a blocked AABB edge of the partner): the stance search runs without the band filter and the
            # exact IK of the coupling decides (recorded in the template facts; the band owner gets the case)
            log.warning("%s: reach band %s (%s) is empty after the recompute: stance C candidates are taken from the goal cells without the band filter",
                        self.gp.variant, support, under or "-")
            self.template_facts.setdefault("empty_bands", []).append({"support": support, "under": under, "stance": rb.meta.get("stance")})
            rb = None
        memo[key] = rb
        return rb

    def v2_band(self, kind: str, support_obj=None, under_partner=None, arm: Optional[str] = None) -> RB.ReachBands:
        """i7-floor 2026-11-07, v2 only (owner: no legacy band - v2 grasps and reach maps only): the v2 reach map of a factor set as a band
        view (``map_sampling.as_bands``) for ``arm`` (default ``tabletop.default_arm``) - each arm's OWN map, never a mirror (a map is computed
        per arm in the synthetic world):

        * the floor (``kind`` "floor", no ``under_partner``): ``tabletop.floor_map_bands`` - the floor map (h 0, kind / plane "floor") of the
          target, the floor object policed at :attr:`floor_z` in the policy's floor mode;
        * a top (``kind`` "tabletop": the partner's top of the ontop place, the init support's): ``tabletop.map_bands`` at that top
          (:meth:`band_top`); an interior surface of the init support has no map kind yet;
        * the floor under a partner (``under_partner``): no map kind yet.

        A miss logs ``MAP_REQUEST`` (the lookup's, or this method's for the kinds without a map), then rejects ``missing_map`` under
        ``ISPATIALGYM_REACH_MAPS_STRICT``, else ``legacy_band_blocked`` (``reach_bands.block_legacy_band``: the legacy band it would need) -
        never ``ReachBands.load`` / ``BandComputer``. Memoised per (kind, support, under, arm); recorded in
        ``template_facts["arm_bands_by_factor"]``. Everything downstream stays exact (stance body checks, the camera rules, F_init / F_goal
        by exact IK, paths, the support-policy re-check)."""
        arm = arm or TT.default_arm(self)
        support = "floor" if kind == "floor" else support_obj.name
        under = under_partner.name if under_partner is not None else None
        factor = f"{kind}:{support}{'-under-' + under if under else ''}"
        memo = self.__dict__.setdefault("_bands", {})
        key = ("v2", kind, support, under, arm)
        if key in memo:
            return memo[key]
        spec = self.make_band_spec()
        scene = str(getattr(self.world, "scene_model", ""))
        rb, what = None, factor
        if kind == "floor" and under_partner is None:
            rb = TT.floor_map_bands(self, spec, arm, floor=self.obj("floor"), z_floor=self.floor_z)
            what = "the floor"
        elif kind == "floor":
            TT.map_request(self, kind="floor_under", plane="floor", under=under, object=spec.object, grasp_key=spec.grasp_key, arm=arm, scene=scene)
            what = f"the floor under {under} (no map kind)"
        else:
            surface = TT.surface_band_kwargs(self) if support_obj is (getattr(self, "surface", None) or {}).get("obj") else {}
            if surface:
                TT.map_request(self, kind="interior", support=support, surface_z=surface.get("surface_z"), object=spec.object, grasp_key=spec.grasp_key,
                               arm=arm, scene=scene)
                what = f"{support}'s interior surface (no map kind)"
            else:
                rb = TT.map_bands(self, spec, self.band_top(support_obj), arm)
                what = f"{support}'s top"
        if rb is None:
            msg = f"{self.gp.variant}: no {arm}-arm v2 reach map for {spec.object} ({spec.grasp_key}) at {what}"
            if os.environ.get("ISPATIALGYM_REACH_MAPS_STRICT"):
                raise Reject("missing_map", msg)
            RB.block_legacy_band(f"{msg} - the legacy {factor} band")
            raise Reject("missing_map", msg)  # (not reached: v2 only blocks every legacy band)
        n_cells = sum(len(c) for cells in (rb.cells_so or {None: rb.cells}).values() for c in cells.values())
        if n_cells == 0:
            # the map's verdict (no feasible cell at any trunk) stands - no unfiltered fallback as for an empty legacy band
            log.warning("%s: the %s-arm reach map of %s at %s is empty", self.gp.variant, arm, spec.object, what)
            self.template_facts.setdefault("empty_maps", []).append({"factor": factor, "arm": arm, "map": rb.meta.get("map_path")})
        log.info("%s: %s-arm reach map %s: %s", self.gp.variant, arm, factor, rb.meta.get("map_path"))
        self.template_facts.setdefault("arm_bands_by_factor", {}).setdefault(factor, {})[arm] = TT.band_facts(rb)
        memo[key] = rb
        return rb

    def band_top(self, support_obj):
        """The measured top of ``support_obj`` a top's band / map is read at: the init surface's top when it is the init support, the
        partner's top of the ontop place (``place["top"]``) when it is the partner, else ``support_top`` with the target ignored."""
        surf = getattr(self, "surface", None) or {}
        if surf.get("obj") is support_obj and surf.get("top") is not None:
            return surf["top"]
        place = getattr(self, "place", None) or {}
        if support_obj is getattr(self, "partner", None) and place.get("top") is not None:
            return place["top"]
        return self.support_top(support_obj, ignore=[self.target])

    def other_arm_band(self, base: RB.ReachBands, kind: str, support: str, under: Optional[str], arm: str) -> Optional[RB.ReachBands]:
        """``arm``'s band of the factors of ``base`` (the other arm's band; owner 2026-09-27, first-demo v2.2): its own cached band
        (``ReachBands.load(..., arm=arm)``; an empty one is a miss), else ``base`` mirrored in memory (``ReachBands.mirrored``: tabletop
        and floor bands only). Never written, never computed in-process: :meth:`compute_band` / ``compute_bands`` write files whose
        names carry no arm. None when neither exists; recorded in ``template_facts["arm_bands_by_factor"]``."""
        c = self.choice.models["target"]
        try:
            rb = RB.ReachBands.load(self.world.scene_model, support, f"{c.category}/{c.model}", self.grasp_key, under=under, scale=c.scale, arm=arm)
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
                raise FileNotFoundError("empty band")
            log.info("%s: %s-arm reach band %s (%s) from %s", self.gp.variant, arm, support, under or "-", rb.path)
        except FileNotFoundError:
            rb = base.mirrored() if RB.mirrorable(base.meta) else None
            log.info("%s: %s-arm reach band %s (%s): %s", self.gp.variant, arm, support, under or "-",
                     f"the {base.arm} arm's band mirrored ({base.path})" if rb is not None else f"none (the {base.arm} arm's band is not mirrored)")
        self.template_facts.setdefault("arm_bands_by_factor", {}).setdefault(f"{kind}:{support}{'-under-' + under if under else ''}", {})[arm] = TT.band_facts(rb)
        return rb

    def compute_band(self, kind: str, support_obj=None, under_partner=None) -> RB.ReachBands:
        """Compute-on-miss of the floor-mode bands (``--support floor [--under partner]``) with the computation the CLI
        runs (the ``stance_free_floor`` / ``stance_under`` stance, the 2 cm floor lattice (under: the clear cells), the
        library grasp mask per trunk incl. tilt −1.2); written to the cache root in the CLI's layout. Computed on
        ``reach_bands.band_kit`` (``self.kit`` is not touched), recorded in ``meta.schedule``. Past the sampling run's time budget
        (``self.deadline``) the computation stops, nothing is written: ``Reject("band_budget")``. ``ISPATIALGYM_BAND_TRUNK_CHECK=idle-arm``
        (owner 2026-10-12 item 4: the whole-body check skipped every deep crouch, so no floor pose was within reach of the
        remaining shoulders): the CLI's ``--trunk-check idle-arm``, keyed, named and recorded like the CLI's."""
        c = self.choice.models["target"]
        kit = RB.band_kit(self.world, self.kit)
        bc = RB.BandComputer(self.world, kit, arms=("right",))
        bc.deadline = getattr(self, "deadline", None)
        tc_key = RB.band_trunk_check(bc, "right")
        t0 = time.time()
        parked = self.world.snapshot()
        try:
            self.world.park(self.target, list(self.world.spawned).index(self.target.name))
            kit.sync_world()
            if kind == "floor":
                z = self.floor_z
                if under_partner is not None:
                    stance = bc.stance_under(under_partner, RB.STAND_OFF)
                    lattice_world = self.place["cells"] if self.relation == "under" and len(self.place.get("cells", [])) else self._under_raycast(under_partner, 0.0, 0.02)
                    ignore = [self.obj("floor")]
                else:
                    room = (self.gp.rooms(self.scene) or self.world.rooms or [None])[0]
                    stance = bc.stance_free_floor(room)
                    origin, fwd, left = bc.base_frame(stance)
                    xs = th.arange(0.2, RB.STRIP_MAX_DEPTH + 0.5, 0.02)
                    ls = th.arange(-RB.STRIP_HALF_WIDTH, RB.STRIP_HALF_WIDTH + 1e-6, 0.02)
                    gx, gl = th.meshgrid(xs, ls, indexing="ij")
                    lattice_world = origin + gx.flatten().unsqueeze(1) * fwd + gl.flatten().unsqueeze(1) * left
                    lattice_world = th.stack([cell for cell in lattice_world if self.kit.trav_free(cell)]) if len(lattice_world) else lattice_world
                    ignore = [self.obj("floor")]
                top_json = {"z": z, "kind": "floor"}
            else:
                raise ValueError("compute_band handles the floor modes; tabletop bands go through tabletop.compute_bands")
            hull, rest = self.hull, self.rest
            max_off = max(float(th.linalg.norm(g.position)) for g in self.grasps) if self.grasps else 0.0

            def pose_fn(world_xy, yaw, _h=hull, _r=rest, _z=z):
                return bc.object_pose_at(world_xy, yaw, _z, _h, _r)

            trunks = RB.trunk_grid(z - self.floor_z)
            result = bc.compute(stance, lattice_world, z, pose_fn, self.grasps, ignore, arm="right", trunks=trunks, max_grasp_offset=max_off)
        except RB.BandDeadline as e:
            raise Reject("band_budget", f"{self.gp.variant}: in-process {kind} band ({under_partner.name if under_partner is not None else '-'}) not finished: {e} "
                                        f"({time.time() - t0:.0f} s in); nothing written - a band job publishes it") from e
        finally:
            self.world.restore(parked)
            kit.sync_world()
        meta = {"scene": self.world.scene_model, "rooms": list(self.world.rooms or []), "support": "floor" if kind == "floor" else support_obj.name, "lattice": bc.lattice,
                "stand_off": RB.STAND_OFF, "arms": "untucked", "grid": RB.grid_meta((z - self.floor_z) < RB.LOW_TOP_Z), "seeds": 12,
                "pre_grasp": RB.PRE_GRASP, "yaws": RB.YAWS, "mode": "floor" if kind == "floor" else "tabletop", "scale": c.scale, "placed_specs": [],
                "object": f"{c.category}/{c.model}", "grasp_key": self.grasp_key, "n_grasps": len(self.grasps), "grasp_names": [g.name for g in self.grasps],
                "rest_quat_xyzw": self.rest.tolist(), "generic_pinch": False, "stance": stance, "top": top_json, "z_support": z,
                "under_partner": under_partner.name if under_partner is not None else None, "trunks": result["trunks"], "computed_by": "iors_large compute-on-miss",
                "time_s": round(time.time() - t0, 1), "schedule": RB.band_schedule(kit, self.kit)}
        if tc_key:
            meta["trunk_check"] = RB.trunk_check_meta("right")
        lat_all = np.concatenate([cc[:, 1] for cc in result["cells"].values() if len(cc)]) if any(len(cc) for cc in result["cells"].values()) else np.zeros(0)
        meta["union_extent"] = [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None
        meta["n_reachable_total"] = int(sum(len(cc) for cc in result["cells"].values()))
        cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in result["cells"].items()}
        pairs = {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in result.get("pairs", {}).items()}
        rb = RB.ReachBands(meta, cells, pairs, None)
        try:
            out_dir = RB.cache_root()
            os.makedirs(out_dir, exist_ok=True)
            key = RB._key({"scene": meta["scene"], "support": meta["support"], "z": round(z, 2), "object": meta["object"], "scale": c.scale, "under": meta["under_partner"],
                           "grasp_key": self.grasp_key, "computed_by": "iors_large", **tc_key})
            name = (f"{meta['mode']}__{meta['scene']}__{meta['support']}{RB.TRUNK_CHECK_IDLE_TAG if tc_key else ''}"
                    f"{'-under-' + meta['under_partner'] if meta['under_partner'] else ''}__{c.category}-{c.model}__{key}")
            arrays = {RB.trunk_key(tr, ti): cc for (tr, ti), cc in cells.items()}
            for (tr, ti), p in pairs.items():
                arrays[RB.trunk_key(tr, ti) + "__pairs"] = p
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
            with open(os.path.join(out_dir, name + ".json"), "w") as f:
                json.dump(meta, f, indent=1, default=str)
            rb.path = os.path.join(out_dir, name + ".npz")
        except Exception as e:  # noqa: BLE001 - the band is still usable in memory
            log.warning("%s: could not write the computed reach band: %r", self.gp.variant, e)
        log.info("%s: reach band computed on miss (%s, %d cells, %.0fs)", self.gp.variant, meta["support"], meta["n_reachable_total"], meta["time_s"])
        return rb

    def band_cells_world(self, rb: Optional[RB.ReachBands], stance: Stance) -> th.Tensor:
        """The band of the stance's grid trunk as world xy at this concrete stance (the band is base-relative)."""
        if rb is None:
            return th.zeros(0, 2)
        cells = rb.band(stance.translate, stance.tilt, **TT.band_kw(rb, stance))
        if len(cells) == 0:
            return th.zeros(0, 2)
        return th.as_tensor(rb.to_world(cells, stance.base_pose2d), dtype=th.float32)

    # ------------------------------------------------------------------ stances A / B per surface candidate

    def try_surface(self, kind: str, cand, rng: np.random.Generator):
        """Build the template facts of stances A and B on one init-surface candidate, and the C survivors' existence."""
        self.surface = {"kind": kind, "obj": None if kind == "floor" else cand}
        if kind == "floor":
            self.try_floor_surface(rng)
        else:
            self.try_support_surface(cand, rng)
        # stance C must exist for the init pose class (checked on the surface's representative init xy) for one of the offered arms
        c_survivors = self.placement_candidates(self.surface["init_xy"])
        for arm in TT.offered_arms(self)[1:]:
            if c_survivors:
                break
            c_survivors = self.placement_candidates(self.surface["init_xy"], arm=arm)
        if not c_survivors:
            raise Reject("no_placement_stance", f"no stance C at {self.partner.name} for an init pose at {self.surface['init_xy'].tolist()}")
        self.surface["n_c"] = len(c_survivors)
        self.template_facts["stances"] = {"init": self.surface["init"].to_meta(), "n_manipulation_candidates": len(self.surface["manip"]), "n_placement_candidates": len(c_survivors)}

    def try_support_surface(self, support, rng: np.random.Generator):
        """Stances A and B on a support (tabletop Phases 1a / 1b / 1c / 2 of ``tabletop.py``): per edge (random order)
        the init trunk, the lateral draw redrawn (≤ 6) until the partner is visible (≥ 4 / 8 corners + centre), the
        Phase-2 survivors with the band of (support, target) - with every offered arm's band (owner 2026-09-27, first-demo v2.2:
        ``tabletop.arm_bands_for``, the other arm's a mirror when it has none; survivors tagged ``focus_arms``)."""
        self.kit.sync_world()
        offered = TT.offered_arms(self)
        half, height = self.target_dims()
        top = TT.support_surface(self, support, ignore=[self.target], height=height)
        if int(top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        self.surface["top"] = top
        self.bands = None  # ``TT.ensure_bands`` caches one band per support on the sampler
        margins = {"w_t": half, "w_p": 0.0, "W_p": 0.0, "D_p_max": 0.0, "safety": 0.02, "relation": self.relation, "g_NT": None, "H_max": height}
        # A5 "partner's own top": the partner is the support under the target, so the partner-visibility redraw does not
        # apply (the target cell in frame is on the partner); stances A / B come from the tabletop phases unchanged.
        # Owner decision 2026‑09‑16 (a)/(d): stance A sees the target (tabletop Phase 1), the init surface (the target's cell
        # on it) and the partner's RELEVANT REGION (ontop: the top-face patch around the goal cell + the nearest bottom
        # edge; under: the nearest bottom edge) — no "≥ 4 of 8 AABB corners + centre" any more; (b): stance B (the grasp
        # stance, Phase 2) has no partner rule — only its target-in-frame rule.
        own_top = support is self.partner
        errors: List[str] = []
        for edge, init in TT.edge_candidates(self, top, rng):
            try:
                init = TT.phase1b_init_trunk(self, init, height, top=top)
                chosen = None
                for _ in range(1 if own_top else 6):  # §5.6: redraw the lateral position (≤ 6) until the goal region is visible
                    st = TT.phase1c_lateral(self, init, margins, rng, top=top)
                    if own_top or self.goal_region_visible(st):
                        chosen = st
                        break
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e.reason}")
                    continue
                raise
            free_start = bool(self.gp.gen("free_start")) and not own_top
            if chosen is None and not free_start:
                errors.append(f"{edge.name}: goal_region_not_visible (the partner's top patch around the goal cell / nearest bottom edge out of frame)")
                continue
            # oracle-fix 2026-11-04 (owner item 2, opt-in generation.free_start): the start pose need not stand at the support's edge -
            # Phase 2 (stance B) is built from the edge stance as usual; when that stance does not see the goal region, stance A is
            # searched on free floor (:meth:`free_start_stance`) with the target and the same goal-region test in frame
            ref = chosen if chosen is not None else st
            ref.name = "init"
            arm_bands = TT.arm_bands_for(self, offered, top=top, edge=ref.edge) if len(offered) > 1 else None
            manip = TT.phase2_manipulation_stance(self, ref, half, target_height=height, top=top, arm_bands=arm_bands)
            if not manip:
                errors.append(f"{edge.name}: no_manipulation_stance")
                continue
            if chosen is None:
                try:
                    chosen = self.free_start_stance(manip[0].focus[:2].clone(), top.z, height, rng)
                except Reject as e:
                    errors.append(f"{edge.name}: goal_region_not_visible at the edge; free start: {e}")
                    continue
                self.template_facts["free_start"] = {"edge": edge.name, "base_pose2d": [round(float(v), 3) for v in chosen.base_pose2d],
                                                     "trunk": [chosen.translate, chosen.tilt], **{k: v for k, v in chosen.extras.items() if k in ("distance", "bearing_deg", "tries")}}
            self.surface.update({"edge": edge, "init": chosen, "manip": manip, "init_xy": manip[0].focus[:2].clone(), "z": top.z, "band": getattr(self, "bands", None),
                                 "arm_bands": dict(arm_bands or {offered[0]: getattr(self, "bands", None)})})
            return
        raise Reject("no_free_edge", f"{support.name}: " + ("; ".join(errors) or "no edge with a free, collision-free stance"))

    def goal_region_points(self, from_xy) -> th.Tensor:
        """Owner decision 2026‑09‑16 (a)/(d): the **relevant region** of the partner stance A must see — ontop: the top-face
        patch (± 0.10 m) around the goal cell the placement will use (the goal cell nearest to ``from_xy``, the rule C
        applies) at the top's height, plus the partner's bottom edge nearest to ``from_xy`` (its two bottom corners and
        the midpoint); under: the nearest bottom edge only (the goal lies beneath the partner)."""
        partner = self.partner
        lo, hi = partner.aabb
        f = th.as_tensor([float(from_xy[0]), float(from_xy[1])])
        pts = []
        if self.relation == "ontop" and len(self.place.get("cells", [])):
            cells = self.place["cells"]
            g = cells[int(th.argmin(th.linalg.norm(cells.float() - f, dim=1)))].float()
            z = float(self.place["z"])
            r = float(self.num("goal_patch_half_m", 0.10))
            for dx, dy in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r)):
                pts.append(th.tensor([float(g[0]) + dx, float(g[1]) + dy, z]))
        # nearest bottom edge of the AABB (as seen from ``from_xy``)
        edges = aabb_edges(lo, hi)
        edge = nearest_edge(edges, f)
        half_len = edge.length / 2
        for t in (-half_len, 0.0, half_len):
            p = edge.mid + edge.tangent * t
            pts.append(th.tensor([float(p[0]), float(p[1]), float(lo[2])]))
        return th.stack(pts)

    def goal_region_visible(self, stance: Stance, from_xy=None) -> bool:
        """The relevant partner region (:meth:`goal_region_points`) in frame with the 5 % margin from ``stance``
        (:meth:`partner_points_in_view`)."""
        cam = self.camera_pose_at(stance)
        if from_xy is None:
            from_xy = stance.focus[:2] if stance.focus is not None else th.as_tensor(stance.base_pose2d[:2])
        return self.partner_points_in_view(cam, self.goal_region_points(from_xy))

    def partner_points_in_view(self, cam, points) -> bool:
        """Partner points in frame with the 5 % margin (stance A's goal region, the place-under stance C's near-edge line): every point,
        or at least ``generation.partner_view_min_points`` of them (owner 2026-10-12 item 2a; :func:`enough_in_view`)."""
        k = self.gp.gen("partner_view_min_points")
        inside = self.world.camera_model.inside(cam, G.t(points).reshape(-1, 3), P0.IN_FRAME_MARGIN)
        return enough_in_view(inside, None if k is None else int(k))

    def partner_visible(self, stance: Stance, n_min: int) -> bool:
        cam = self.camera_pose_at(stance)
        lo, hi = self.partner.aabb
        corners = G.aabb_corners(lo.float(), hi.float())
        inside = self.world.camera_model.inside(cam, corners, 0.0)
        centre_in = bool(self.world.camera_model.inside(cam, ((lo + hi) / 2).float().reshape(1, 3), 0.0)[0])
        return int(inside.sum()) >= n_min and centre_in

    def try_floor_surface(self, rng: np.random.Generator):
        """§5.6 floor init: the anchor cell c_t, stance B facing it from {0.35, 0.45, 0.55} with the trunk from the floor
        band, stance A on the eroded map 0.6–1.5 m from c_t with the target and the partner in frame."""
        fi = dict(self.gp.gen("floor_init") or {})
        cell = float(fi.get("cell_m", 0.02))
        clearance = float(fi.get("clearance_m", 0.02))
        max_geo = float(fi.get("max_geodesic_m", self.num("max_partner_geodesic_m", 3.0)))
        floor = self.obj("floor")
        partner = self.partner
        lo, hi = floor.aabb
        plo, phi = partner.aabb
        rb = self.band("floor")
        self.surface["band"] = rb
        offered = TT.offered_arms(self)
        arm_bands = {offered[0]: rb, **{a: self.band("floor", arm=a) for a in offered[1:]}}
        self.surface["arm_bands"] = arm_bands
        for a in offered[1:]:
            self.template_facts.setdefault("arm_bands", {})[a] = TT.band_facts(arm_bands[a])
        half, height = self.target_dims()
        objs = self.floor_blocking_objects()
        aabbs = [(o.aabb[0][:2].float() - clearance, o.aabb[1][:2].float() + clearance) for o in objs]
        # candidate anchors: free floor cells within max_geo of the partner's near side, outside every AABB and the partner footprint
        xs = th.arange(float(lo[0]) + cell / 2, float(hi[0]), cell * 4)  # 8 cm search grid (the 2 cm raster is the draw's resolution)
        ys = th.arange(float(lo[1]) + cell / 2, float(hi[1]), cell * 4)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        near_c = th.tensor([float(plo[0] + phi[0]) / 2, float(plo[1] + phi[1]) / 2])
        pts = pts[th.linalg.norm(pts - near_c, dim=1) <= max_geo + 1.5]
        anchors = []
        stats = {"grid": int(len(pts)), "trav_free": 0, "in_rooms": 0, "outside_partner": 0, "outside_objects": 0}
        blockers: Dict[str, int] = {}
        for p in pts:
            if not self.kit.trav_free(p):
                continue
            stats["trav_free"] += 1
            if not P0.in_task_rooms(self, p):
                continue
            stats["in_rooms"] += 1
            if bool((p >= plo[:2].float() - half).all() and (p <= phi[:2].float() + half).all()):
                continue  # inside the partner footprint (the goal must be false at init)
            stats["outside_partner"] += 1
            hit = [o.name for o, (a, b) in zip(objs, aabbs) if bool((p >= a).all() and (p <= b).all())]
            if hit:
                blockers[hit[0]] = blockers.get(hit[0], 0) + 1
                continue
            stats["outside_objects"] += 1
            anchors.append(p)
        self.template_facts["floor_cells"] = {**stats, "blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:6]), "floor": floor.name,
                                              "floor_aabb": [lo[:2].tolist(), hi[:2].tolist()]}
        if not anchors:
            raise Reject("no_floor_cell", f"no free floor cell beside the partner ({stats}; blocking AABBs {dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:4])}; "
                                          f"floor {floor.name} aabb {[round(float(v), 2) for v in lo[:2].tolist()]}..{[round(float(v), 2) for v in hi[:2].tolist()]})")
        anchors = th.stack(anchors)
        order = rng.permutation(len(anchors))
        errors = []
        for i in order[:40]:
            c_t = anchors[int(i)]
            try:
                _, dist = self.kit.geodesic(c_t, self.near_side_point(partner, c_t))
            except Exception:  # noqa: BLE001
                dist = None
            if dist is None or dist > max_geo:
                continue
            try:
                manip = self.floor_stance_b(c_t, rb, half, rng, arm_bands=arm_bands)
                init = self.floor_stance_a(c_t, rng, height, like=manip[0])
                with self.ikc.ignoring([self.obj("floor")]):
                    self.record_leg_a_b(init, manip[0])  # §5.6: a path A -> B exists (navigate-then-torso, else torso-then-navigate)
                self.sidecar.transitions.clear()
            except Reject as e:
                errors.append(f"{e.reason}: {e}")
                continue
            self.sidecar.paths.clear()
            self.surface.update({"c_t": c_t, "init": init, "manip": manip, "init_xy": c_t.clone(), "z": self.floor_z, "edge": None})
            self.template_facts["floor_init"] = {"c_t": c_t.tolist(), "geodesic_m": dist, "n_anchor_candidates": int(len(anchors))}
            return
        raise Reject("no_floor_cell", f"no anchor cell admits stances A / B ({len(errors)} tried): " + "; ".join(sorted(set(errors))[-4:]))

    def floor_stance_b(self, c_t: th.Tensor, rb: Optional[RB.ReachBands], half: float, rng: np.random.Generator,
                       arm_bands: Optional[Dict[str, Optional[RB.ReachBands]]] = None) -> List[Stance]:
        """Stance B candidates of the floor init: base facing c_t at {0.35, 0.45, 0.55} (bearing uniform), trunk from the
        floor band (c_t must be a band cell), collision-free, target in frame, axis within 8° (else 20°) of c_t.
        ``arm_bands`` (owner 2026-09-27, first-demo v2.2: ``{arm: floor band}`` of the offered arms, the default arm's being ``rb``):
        the other arm's candidates turn the base the mirror way (:func:`mirrored_offsets`: c_t to the robot's left for the left
        arm) and hold c_t in that arm's band; every candidate records the arms whose band holds c_t at its trunk in
        ``extras["focus_arms"]`` and works with the first of them."""
        arm0 = TT.default_arm(self)
        others = {a: b for a, b in (arm_bands or {}).items() if a != arm0 and b is not None} if rb is not None else {}
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in list(RB.TILTS) + [RB.LOW_TOP_TILT]]
        for b in others.values():
            trunks += [k for k in b.trunks() if k not in trunks]
        skip = {tuple(round(float(v), 3) for v in t) for t in (self.gp.gen("floor_b_trunks_exclude") or [])}
        if skip:  # oracle-fix 2026-11-04 round 3 (mouse): the floor-level trunks wobble the torso > 1 deg at the grasp (QA robot_wobble)
            trunks = [t for t in trunks if tuple(round(float(v), 3) for v in t) not in skip] or trunks
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + 0.03])
        cands = []
        # the floor band lies to the robot's RIGHT of the centreline (B3 under-edge probe: lateral −0.40…−0.16; the floor
        # bands' cells at translate 0.95 likewise): the base is turned so that c_t sits at lateral ``y_off`` (0 = straight
        # ahead, negative = right) — a stance facing c_t squarely puts it between the arms where no floor cell is reachable
        # (countertop-floor 18:50: 41 band cells, "c_t is in no band cell of any stance" for 29 anchors); the left arm's
        # (mirrored) band lies to the robot's left
        offsets = [float(v) for v in (self.gp.gen("floor_init_lateral_offsets_m") or (-0.25, -0.15, -0.35, 0.0))]
        lanes = list(offsets)
        for a in others:
            lanes += [v for v in mirrored_offsets(offsets, a) if all(abs(v - u) > 1e-9 for u in lanes)]
        bands = {arm0: rb, **others}

        def holds(b: RB.ReachBands, st: Stance) -> bool:
            cells = self.band_cells_world(b, st)
            return len(cells) > 0 and float(th.linalg.norm(cells - c_t, dim=1).min()) <= 0.51 * float(b.meta.get("lattice", RB.LATTICE))

        for _ in range(12):
            bearing = float(rng.uniform(-math.pi, math.pi))
            for dist in STAND_OFFS:
                xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
                if not P0.base_ok(self, xy):
                    continue
                yaw0 = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
                for y_off in lanes:
                    if abs(y_off) >= dist:
                        continue
                    yaw = offset_yaw(yaw0, y_off, dist)  # c_t at the base-frame lateral y_off (negative: the robot's right)
                    for tr, ti in trunks:
                        st = Stance("manipulation", (float(xy[0]), float(xy[1]), yaw), tr, ti, dist, None, None, 0.0, focus.clone(), [arm0])
                        st.extras["lateral_offset_m"] = y_off
                        if rb is not None:
                            focus_arms = [a for a, b in bands.items() if holds(b, st)]
                            if not focus_arms:
                                continue
                            if others:
                                st.arms = [focus_arms[0]]
                                st.extras["focus_arms"] = focus_arms
                        cands.append(st)
        if not cands:
            raise Reject("no_manipulation_stance", "floor init: c_t is in no band cell of any stance")
        # the floor band has cells at the deep trunks only (translate 0.95): like the push-under stance C, the floor mesh
        # is removed from the stance check (the cuRobo ground slab stays — B3's "torso / head within 1 cm of the floor
        # mesh" finding) and the optical-axis rule is 45° (the crouched head hangs over the cell)
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        n_cands, trunks_seen = len(cands), sorted({(float(st.translate), round(float(st.tilt), 2)) for st in cands})
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        if not cands:
            raise Reject("no_manipulation_stance", f"floor init: every stance B collides ({n_cands} candidates at trunks {trunks_seen})")
        pts = G.local_points_world(self.hull, (focus, self.rest))
        classed = []
        n_frame, n_axis = 0, 0
        axis_max = math.radians(float(self.num("floor_init_axis_max_deg", 45.0)))
        for st, cam in zip(cands, P0.camera_poses(self, cands)):
            if not P0.in_frame(self, cam, pts):
                n_frame += 1
                continue
            ang = P0.axis_angle(self, cam, focus)
            if ang > axis_max:
                n_axis += 1
                continue
            st.extras["angle_class"] = 1 if ang > math.radians(P0.AXIS_TIGHT_DEG) else 0
            st.extras["axis_angle_deg"] = math.degrees(ang)
            classed.append((st, st.extras["angle_class"]))
        out = P0.best_class(classed)
        if not out:
            raise Reject("no_manipulation_stance", f"floor init: no stance B sees c_t ({len(cands)} collision-free; {n_frame} target out of frame, {n_axis} axis > {math.degrees(axis_max):.0f}°)")
        return out

    def floor_stance_a(self, c_t: th.Tensor, rng: np.random.Generator, height: float, like: Optional[Stance] = None) -> Stance:
        """Stance A of the floor init: base on the eroded map 0.6–1.5 m from c_t facing it, with the target and the
        partner's relevant region (goal patch / nearest bottom edge, owner decision (a)/(d)) in frame, collision-free.
        Trunk: **torso straight — tilt 0 at any translate** (owner 2026‑09‑19 21:56 (i): "it can have the lowest height
        but no tilt"), the **most upright** translate among those that see the target and the partner region, the
        optical-axis angle to c_t breaking ties (owner 2026-09-24). Until then the smallest optical-axis angle won: with
        the tucked spawn of ``1dd3820b`` the hanging hands of a crouched A touched the floor and ``body_free`` dropped it,
        so A came out upright by accident (0921-045713: 0.25); with the untucked spawn the 0.95 crouch passes and looks
        down at a floor target best, and it then swept the target on the direct A -> B drive at every draw and became the
        carry's second travel trunk (0924-225021: 8/8 draws at 0.95, no instance). (``729a1ec`` had A take stance B's
        crouch so that A -> B needed no sweep; that is now the *fallback* of the A -> B leg — a departure transition at
        A — not the init posture. ``like`` is kept for the signature; ``trunk_like_b`` is recorded for the log.)"""
        fi = dict(self.gp.gen("floor_init") or {})
        d_lo, d_hi = fi.get("distance_range_m", [0.6, 1.5])
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + height / 2])
        pts = G.local_points_world(self.hull, (th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + self.world.bottom_offset(self.hull, self.rest)]), self.rest))
        for _ in range(30):
            bearing = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(float(d_lo), float(d_hi)))
            xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
            if not self.kit.trav_free(xy, eroded=True) or not P0.in_task_rooms(self, xy):
                continue
            yaw = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
            trunks = [(float(t), 0.0) for t in RB.TRANSLATES]  # (i): straight torso, any height
            cands = [Stance("init", (float(xy[0]), float(xy[1]), yaw), t, ti, dist) for t, ti in trunks]
            cams = P0.camera_poses(self, cands)
            feas = []
            for st, cam in zip(cands, cams):
                if not P0.in_frame(self, cam, pts) or not self.goal_region_visible(st, from_xy=c_t):
                    continue
                feas.append((st, P0.axis_angle(self, cam, focus)))
            if not feas:
                continue
            # floor class: A spawns UNTUCKED like every other family's init (owner 2026-09-24; ``Stance.arms`` default). 1dd3820b
            # (2026-09-16) spawned it tucked because A then stood at B's crouch (0.95, 0) and the untucked hands hanging at floor
            # level swept the floor target on the A -> B drive; A is now the most upright straight trunk that sees the target
            # (below) and the crouch happens at the transition, where the arms were untucked again anyway (arm_to(left,right),
            # 161 steps). The A -> B path sweep still judges the hands: an A whose untucked hands reach the target is rejected
            # ``path_collision`` and the init is re-drawn.
            free = P0.body_free(self, [st for st, _ in feas])
            feas = [(st, a) for (st, a), f in zip(feas, free.tolist()) if f]
            if not feas or not P0.trunk_sweep_free(self, (float(xy[0]), float(xy[1]), yaw)):
                continue
            # the most upright translate wins (the hands highest on the drive, the carry's travel trunk straightest); the
            # optical-axis angle breaks ties (owner 2026-09-24)
            like_t = None if like is None else float(like.translate)
            st = min(feas, key=lambda sa: (sa[0].translate, round(sa[1], 2)))[0]
            st.extras.update({"bearing_deg": math.degrees(bearing), "distance": dist, "trunk_like_b": bool(like_t is not None and abs(st.translate - like_t) < 1e-6)})
            return st
        raise Reject("no_stance", "floor init: no stance A sees the target at c_t and the partner")

    def carry_lift_extra(self) -> float:
        """Opt-in ``generation.carry_lift_by_extent_m`` = margin (oracle-fix 2026-11-04, owner item 4): the carry lifts are raised by the
        carried object's height + this margin, so the object's bottom clears a surface at the pick height by its own extent (a pillow
        carried over the other bed); 0 when unset."""
        m = self.gp.gen("carry_lift_by_extent_m")
        if m is None:
            return 0.0
        _, height = self.target_dims()
        return float(height) + float(m)

    def free_start_stance(self, c_xy: th.Tensor, z_top: float, height: float, rng: np.random.Generator) -> Stance:
        """Opt-in ``generation.free_start`` (oracle-fix 2026-11-04, owner item 2; ``{distance_range_m, tries}``, default 0.6-2.5 m, 120): a
        start pose anywhere on free floor (eroded map, the task rooms) ``distance_range_m`` from the target's cell ``c_xy`` on its support
        (top at ``z_top``), facing it, from which the target (its hull at rest on the top) and the partner's goal region
        (:meth:`goal_region_visible`, the edge stance's test) are in the head-camera frame - straight torso, the most upright translate
        that sees both (:meth:`floor_stance_a`'s rule), collision-free with a free trunk sweep."""
        cfg = self.gp.gen("free_start")
        cfg = dict(cfg) if isinstance(cfg, dict) else {}
        d_lo, d_hi = (float(v) for v in cfg.get("distance_range_m", [0.6, 2.5]))
        tries = int(cfg.get("tries", 120))
        c_xy = th.as_tensor([float(c_xy[0]), float(c_xy[1])])
        focus = th.tensor([float(c_xy[0]), float(c_xy[1]), float(z_top) + height / 2])
        pts = G.local_points_world(self.hull, (th.tensor([float(c_xy[0]), float(c_xy[1]), float(z_top) + self.world.bottom_offset(self.hull, self.rest)]), self.rest))
        stats = {"floor": 0, "view": 0, "body": 0, "sweep": 0}
        for k in range(tries):
            bearing = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(d_lo, d_hi))
            xy = c_xy + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
            if not self.kit.trav_free(xy, eroded=True) or not P0.in_task_rooms(self, xy):
                stats["floor"] += 1
                continue
            yaw = math.atan2(float(c_xy[1] - xy[1]), float(c_xy[0] - xy[0]))
            cands = [Stance("init", (float(xy[0]), float(xy[1]), yaw), float(t), 0.0, dist) for t in RB.TRANSLATES]
            cams = P0.camera_poses(self, cands)
            feas = [(st, P0.axis_angle(self, cam, focus)) for st, cam in zip(cands, cams)
                    if P0.in_frame(self, cam, pts) and self.goal_region_visible(st, from_xy=c_xy)]
            if not feas:
                stats["view"] += 1
                continue
            free = P0.body_free(self, [st for st, _ in feas])
            feas = [(st, a) for (st, a), f in zip(feas, free.tolist()) if f]
            if not feas:
                stats["body"] += 1
                continue
            if not P0.trunk_sweep_free(self, (float(xy[0]), float(xy[1]), yaw)):
                stats["sweep"] += 1
                continue
            st = min(feas, key=lambda sa: (sa[0].translate, round(sa[1], 2)))[0]
            st.extras.update({"bearing_deg": round(math.degrees(bearing), 1), "distance": round(dist, 3), "tries": k + 1, "free_start": True})
            log.info("%s: free start: stance A at %s (%.2f m from the target cell, trunk (%.2f, 0)) after %d draws %s", self.gp.variant,
                     [round(float(v), 2) for v in st.base_pose2d], dist, st.translate, k + 1, stats)
            return st
        raise Reject("no_stance", f"free start: no stance A {d_lo:.1f}-{d_hi:.1f} m from the target sees it and the goal region ({tries} draws: {stats})")

    # ------------------------------------------------------------------ stance C (placement)

    def placement_candidates(self, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """§5.6 stance C survivors for an init pose: base square to the partner edge nearest ``init_xy`` — when that edge
        admits no stance (no free floor in front of it: countertop-floor 20:24, edge +v ``base_ok 0``), the other
        edges follow in order of distance from the init pose (§5.6 wording proposed: "the nearest edge that admits a
        stance") — stand-off {0.35, 0.45, 0.55}, lateral along the edge minus margins, trunk from the band; goal cell
        per relation; visibility per relation; the best angle class (each survivor carries its ``goal_cell`` in
        ``extras``). For ``arm`` (default: the segment arm; owner 2026-09-27, first-demo v2.2): its band, its push lanes, its
        working arm (the other arm's idle posture derived for it)."""
        arm = arm or self.seg_arm
        p = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        edges = sorted(self.place["edges"], key=lambda e: float(th.linalg.norm(e.mid - p)))
        allow = self.gp.gen("placement_edges")  # optional whitelist of partner edge names (+x / -x / +y / -y; top-raster edge names for ontop)
        if allow:
            edges = [e for e in edges if e.name in set(allow)] or edges
        # owner 2026-09-20 09:50: the survivors of EVERY edge are pooled (nearest edge first, ``extras.edge_rank``) - the coupling
        # decides; the nearest edge alone boxed the table probe in (the room behind the short end blocks every drive-back pose
        # of the carry arrival, 0920-092254). ``generation.placement_pool_edges = false`` restores "the nearest edge that admits
        # a stance" (spec 5.6 wording)
        pool = self.gp.gen("placement_pool_edges", True) is not False
        tried, out_all = [], []
        for rank, edge in enumerate(edges):
            out = self._placement_candidates_at(edge, init_xy, arm)
            if out:
                for st in out:
                    st.extras["edge_rank"] = rank
                if not pool:
                    if tried:
                        log.info("%s: stance C: edge(s) %s nearest the init pose admit no stance; %s taken", self.gp.variant, tried, edge.name)
                    return out
                out_all += out
            else:
                tried.append(edge.name)
        if pool and out_all:
            log.info("%s: stance C: %d survivors pooled over the edges %s%s", self.gp.variant, len(out_all), [e.name for e in edges if e.name not in tried],
                     f" (no stance at {tried})" if tried else "")
        return out_all

    def _placement_candidates_at(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        arm = arm or self.seg_arm
        partner = self.partner
        half = self.place["half"]
        if self.push_under:
            return self.push_under_candidates(edge, init_xy, arm)
        rb = self.band("tabletop", partner, arm=arm) if self.relation == "ontop" else self.band("floor", None, partner, arm=arm)
        self.place["band"] = rb
        self.place.setdefault("bands", {})[arm] = rb
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in RB.TILTS]
        margin = half + 0.10
        L = edge.length
        half_run = L / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        init_xy = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        goal_cells = self.place["cells"]
        stats: Dict[str, int] = {"goal_cells": int(len(goal_cells)), "band_trunks": len(trunks), "lattice": 0, "base_ok": 0, "band_cells": 0, "goal_in_band": 0}
        cands: List[Stance] = []
        for so in [float(v) for v in (self.gp.gen("stand_offs_m") or STAND_OFFS)]:
            for lat in laterals:
                xy = edge.base_xy(so, lat)
                stats["lattice"] += len(trunks)
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += len(trunks)
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), tr, ti, so, edge, None, lat, None, [arm])
                    cells = self.band_cells_world(rb, st)
                    if rb is not None and len(cells) == 0:
                        continue
                    stats["band_cells"] += 1
                    pool = goal_cells if rb is None else self._snap(cells, goal_cells, float(rb.meta.get("lattice", RB.LATTICE)))
                    if len(pool) == 0:
                        continue
                    stats["goal_in_band"] += 1
                    gc = pool[int(th.argmin(th.linalg.norm(pool - init_xy, dim=1)))]
                    st.extras["goal_cell"] = [float(gc[0]), float(gc[1])]
                    cands.append(st)
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: stance C at %s edge %s: no candidate (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        free = P0.body_free(self, cands)
        if os.environ.get("ISPATIALGYM_STANCE_BLAME"):  # oracle-fix 2026-11-04 (owner item 3)
            P0.blame_stances(self, [st for st, f in zip(cands, free.tolist()) if not f], f"stance C {partner.name} edge {edge.name}")
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if not cands:
            log.info("%s: stance C at %s edge %s: every candidate collides (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        tangent = th.tensor([-float(edge.inward[1]), float(edge.inward[0])])
        classed = []
        for st, cam in zip(cands, cams):
            gc = st.extras["goal_cell"]
            if self.relation == "ontop":
                pose = self.goal_pose_at(gc, self.rest)
                if not P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose)):
                    continue
                focus = th.tensor([gc[0], gc[1], self.place["z"] + self.place["height"] / 2])
                # owner decision 2026‑09‑16 (c): the ontop goal-visibility rule is "goal cell (the target at the goal) in frame
                # (5 %)" — the 20° optical-axis cone stays for under only (the target hides beneath the partner). Diagnosed
                # on box_of_shampoo-sofa-bar: 13/13 collision-free C candidates had the goal in frame, unoccluded, at 30–31° off
                # the axis (camera rigid on the torso, trunk (0, −1.2) from the band, stand-off 0.35 m); an optional cone
                # (``ontop_axis_cone_deg``) is kept for experiments.
                cone = self.gp.gen("ontop_axis_cone_deg")
                ang = P0.axis_angle(self, cam, focus)
                st.extras["axis_angle_deg"] = math.degrees(ang)
                k = (1 if ang <= math.radians(float(cone)) else None) if cone is not None else 1
            else:
                # the near edge's bottom line where the target passes under it — the two bottom points of the edge at the
                # goal cell's lateral position ± (half extent + 0.10) — in frame; the optical axis within 20° of the
                # (hidden) goal cell. (The spec's "two bottom AABB corners" lie ±1 m to the sides of a bed / sofa edge, 65°
                # off the optical axis at s = 0.45 m — outside the 99° hfov for every stance; wording change proposed.)
                corners = G.aabb_corners(lo.float(), hi.float())
                bottom = corners[corners[:, 2] <= float(lo[2]) + 1e-4]
                d = (bottom[:, :2] - edge.mid) @ edge.inward
                near_line = bottom[d.abs() <= float(d.abs().min()) + 0.02]
                on_line = near_line[:, :2].mean(dim=0) + tangent * float(th.dot(th.tensor([gc[0], gc[1]]) - near_line[:, :2].mean(dim=0), tangent))
                span = half + 0.10
                near = th.stack([th.cat([on_line - tangent * span, lo[2:3].float()]), th.cat([on_line + tangent * span, lo[2:3].float()])])
                if not self.partner_points_in_view(cam, near):
                    continue
                focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                ang = P0.axis_angle(self, cam, focus)
                # ``generation.under_axis_max_deg`` (owner 2026-10-12 item 2c; default the 20° above): the under band's crouched head
                # hangs over the edge like the push-under's (45° there)
                cone_u = self.gp.gen("under_axis_max_deg")
                st.extras["axis_angle_deg"] = math.degrees(ang)
                k = 1 if ang <= math.radians(float(P0.AXIS_LOOSE_DEG if cone_u is None else cone_u)) else None
            if k is None:
                continue
            st.focus = focus
            st.extras["angle_class"] = k
            classed.append((st, k))
        stats["visible"] = len(classed)
        out = P0.best_class(classed)
        log.info("%s: stance C at %s edge %s: %d survivors (%s)", self.gp.variant, partner.name, edge.name, len(out), stats)
        if not out and cands:
            self.debug_placement_visibility(cands, cams, edge, init_xy)
        return out

    def _object_of_prim(self, prim_path: str) -> Optional[str]:
        for o in self.world.scene.objects:
            for l in o.links.values():
                if prim_path == l.prim_path or prim_path.startswith(l.prim_path + "/"):
                    return o.name
        return None

    def debug_placement_visibility(self, cands: Sequence[Stance], cams: Sequence[Any], edge: EdgeFrame, init_xy) -> None:
        """Owner decision 2026‑09‑16 (c) diagnosis: for every collision-free stance-C candidate the goal cell's frame test
        (5 %), its angle to the optical axis, the first object a ray from the camera to the goal point hits (occluder), the
        trunk / stand-off / lateral; plus the partner edges that were considered and why one was chosen (the edge nearest
        the init pose is the only one the spec uses)."""
        try:
            partner = self.partner
            z_top = float(self.place["z"])
            rows = []
            for st, cam in zip(cands, cams):
                gc = st.extras.get("goal_cell")
                if gc is None:
                    continue
                if self.relation == "ontop":
                    focus = th.tensor([gc[0], gc[1], z_top + self.place["height"] / 2])
                    pose = self.goal_pose_at(gc, self.rest)
                    corners_in = P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose))
                else:
                    focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                    corners_in = None
                cell_in = P0.in_frame(self, cam, focus.reshape(1, 3))
                ang = math.degrees(P0.axis_angle(self, cam, focus))
                cam_p = G.t(cam[0]).float()
                hit = self.world.raycast(cam_p.reshape(1, 3), focus.reshape(1, 3))[0]
                occluder = None
                if hit.get("hit"):
                    name = self._object_of_prim(hit.get("rigidBody", ""))
                    d_hit = float(th.linalg.norm(th.as_tensor(hit["position"]).float() - cam_p))
                    d_goal = float(th.linalg.norm(focus - cam_p))
                    if name is not None and name != partner.name and d_hit < d_goal - 0.03:
                        occluder = f"{name} at {d_hit:.2f} m (goal at {d_goal:.2f} m)"
                    elif name == partner.name and d_hit < d_goal - 0.05:
                        occluder = f"{partner.name} itself at {d_hit:.2f} m (goal at {d_goal:.2f} m: armrest / back / cushion in front of the cell)"
                rows.append({"stand_off": round(float(st.stand_off or 0.0), 2), "lateral": round(float(st.lateral), 2), "trunk": [float(st.translate), round(float(st.tilt), 2)],
                             "goal_cell": [round(float(v), 2) for v in gc], "goal_in_frame": bool(cell_in), "goal_hull_in_frame": corners_in, "axis_angle_deg": round(ang, 1),
                             "occluder": occluder, "camera_z": round(float(cam_p[2]), 2)})
            f = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
            edges = [{"name": e.name, "length_m": round(float(e.length), 2), "dist_to_init_m": round(float(th.linalg.norm(e.mid - f)), 2), "chosen": e.name == edge.name,
                      "why": "nearest to the init pose (§5.6: edge = the partner edge nearest the target's init pose)" if e.name == edge.name else "not the nearest edge"}
                     for e in self.place["edges"]]
            self.template_facts["placement_visibility_debug"] = {"candidates": rows, "edges": edges}
            n_in = sum(1 for r in rows if r["goal_in_frame"])
            n_cone = sum(1 for r in rows if r["axis_angle_deg"] <= P0.AXIS_LOOSE_DEG)
            n_occ = sum(1 for r in rows if r["occluder"])
            log.info("%s: stance C visibility debug at %s edge %s: %d candidates — goal cell in frame %d, within the 20° cone %d, occluded %d; edges %s; per candidate %s",
                     self.gp.variant, partner.name, edge.name, len(rows), n_in, n_cone, n_occ, edges, rows)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            log.info("%s: stance C visibility debug failed: %r", self.gp.variant, e)

    def push_under_candidates(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C of the push-under (owner decision 2026‑09‑16 item 7): base square to the partner's near AABB edge,
        facing inward (= the push direction); per lateral position λ the **goal cell** g = edge point + inward·0.07
        (target centre 5–10 cm inside the footprint, clear under: ray ≥ height + 3 cm) and the **drop cell** c_d = edge
        point − inward·(half + 0.10) on free floor; stand-off s ∈ {0.5…0.7} so that c_d lies 0.25–0.6 m ahead of the
        base (B3: the floor is reachable 0.24–0.64 m ahead at the deep trunks); trunks = the floor trunks (translate ≥
        0.75, tilt ≤ −0.9); whole-body collision-free (A6 idle-arm retry), the drop footprint and the edge's bottom line
        at λ ± (half + 0.10) in frame, optical axis within 20° of the drop cell. Each survivor carries ``goal_cell``,
        ``drop_cell``, ``push_dir``, ``drop_ahead_m``. The lanes are laid out for the right arm and mirrored for the left
        (owner 2026-09-27, first-demo v2.2: :func:`frontal_lane_lateral`, :func:`side_push_pose`). ``generation.push_under_stand_offs_m`` /
        ``push_under_drop_ahead_m`` / ``floor_trunks`` / ``frontal_lane_offset_m`` override the stand-offs, the drop cell's window, the trunks
        and the lanes; the pitched hand's defaults stand farther out (:data:`PUSH_FAR_STAND_OFFS`, :data:`PUSH_FAR_DROP_AHEAD_M`,
        :data:`PUSH_FAR_TRUNKS`, :data:`PUSH_FAR_LANE_OFFSETS`)."""
        arm = arm or self.seg_arm
        partner = self.partner
        half, height = float(self.place["half"]), float(self.place["height"])
        need = float(self.place.get("need", height + 0.03))
        d2 = th.as_tensor([float(edge.inward[0]), float(edge.inward[1])])
        tangent = th.tensor([-float(d2[1]), float(d2[0])])
        inside = float(self.num("push_under_inside_m", PUSH_UNDER_INSIDE_M))
        out = half + float(self.num("push_under_drop_extra_m", PUSH_UNDER_DROP_EXTRA_M))
        margin = half + 0.10
        half_run = edge.length / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        # side stances may use the lane up to the edge's end minus the target's half extent (the deep trunks were clear only
        # with the base near the +tangent end of the bed edge: the lane 0.45 m further along is at the edge's end)
        half_run_side = edge.length / 2 - half - 0.02
        laterals_side = P0.lattice_values(-half_run_side, half_run_side, 0.10) if half_run_side > 0 else [0.0]
        # the pitched hand (owner 2026-10-12 ~20:55 item C) holds its tips low only 0.50-0.90 m ahead, at the deepest trunks: farther stances
        far = self.push_hand == "pitched_far"
        stand_offs = [float(v) for v in (self.gp.gen("push_under_stand_offs_m") or (PUSH_FAR_STAND_OFFS if far else PUSH_UNDER_STAND_OFFS))]
        trunks = [tuple(t) for t in (self.gp.gen("floor_trunks") or (PUSH_FAR_TRUNKS if far else FLOOR_TRUNKS))]
        lo_ahead, hi_ahead = (float(v) for v in (self.gp.gen("push_under_drop_ahead_m") or (PUSH_FAR_DROP_AHEAD_M if far else PUSH_UNDER_DROP_AHEAD_M)))
        cells = self.place["cells"]
        mid = th.as_tensor([float(edge.mid[0]), float(edge.mid[1])])
        stats: Dict[str, Any] = {"laterals": len(laterals), "goal_clear": 0, "drop_free": 0, "base_ok": 0, "candidates": 0, "side_candidates": 0, "drop_blocked": {},
                                 "drop_out_m": round(out, 3), "edge_mid": [round(float(v), 2) for v in mid.tolist()], "inward": [round(float(v), 2) for v in d2.tolist()]}
        # side stances: the base stands BESIDE the push lane, facing along the edge with the partner on its right (the
        # right arm pushes sideways, toward the partner; the left arm faces the other way); the lane lies ``x_ahead`` ahead of the base and the base centre
        # ``b`` outside the edge — the whole lane (drop 0.22 m outside → goal 0.07 m inside) stays 0.35–0.45 m ahead, inside
        # the floor reach, and the crouched torso bends along the edge, not into the partner (the frontal stances need
        # s ≥ 0.75 m to keep the deep trunk clear of a bed, which puts the lane 0.46–0.82 m ahead: out of reach)
        # the lane must clear the base rim: with x_ahead = 0.35 the blade start (0.35 m ahead, 6 cm high) sits on the
        # base's front edge and every push pre-pose fails the IK (run 16:04) — 0.45 m and more only
        side_ahead = [float(v) for v in (self.gp.gen("push_under_side_ahead_m") or (0.45, 0.55, 0.6))]
        side_b = [float(v) for v in (self.gp.gen("push_under_side_standoff_m") or (0.35, 0.40))]
        cands: List[Stance] = []
        lat_all = sorted({round(float(v), 4) for v in list(laterals) + list(laterals_side)})
        lat_frontal = {round(float(v), 4) for v in laterals}
        for lat in lat_all:
            g = mid + tangent * float(lat) + d2 * inside
            cl = self.clearance_at(g)
            clear = (cl is not None and cl >= need) or (cl is None and len(cells) > 0 and float(th.linalg.norm(cells - g, dim=1).min()) <= 0.03)
            if not clear:
                continue
            stats["goal_clear"] += 1
            c_d = mid + tangent * float(lat) - d2 * out
            if not self._free_floor_cell(c_d, stats["drop_blocked"]):
                continue
            stats["drop_free"] += 1
            common = {"goal_cell": [float(g[0]), float(g[1])], "drop_cell": [float(c_d[0]), float(c_d[1])], "push_dir": [float(d2[0]), float(d2[1])], "inside_m": inside,
                      "edge": edge.name}
            # frontal stances: the lane to the robot's RIGHT of the base centre by ``frontal_lane_offset_m`` — B3's under-edge
            # probe (bed_rrcvaq_0, briefcase, s = 0.6) reaches the edge strip at lateral −0.40…−0.16 only (the right arm's
            # side), a centred lane at s = 0.45 / 0.6 has 0–1 cells; the left arm's lane to its LEFT
            for off in ([float(v) for v in (self.gp.gen("frontal_lane_offset_m") or (PUSH_FAR_LANE_OFFSETS if far else (0.25, 0.0)))] if lat in lat_frontal else []):
                for so in stand_offs:
                    ahead = so - out
                    if ahead < lo_ahead or ahead > hi_ahead:
                        continue
                    lat_b = frontal_lane_lateral(float(lat), off, arm)
                    xy = edge.base_xy(so, lat_b)  # base beside the lane by ``off``, away from the arm's side (+tangent = the robot's left)
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), float(tr), float(ti), so, edge, None, lat_b, None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(ahead), stance_kind="frontal", lane_offset_m=arm_sign(arm) * off)
                        cands.append(st)
                        stats["candidates"] += 1
            for x_ahead in side_ahead:
                for b in side_b:
                    x_b, y_b, yaw_b = side_push_pose(mid, tangent, d2, float(lat), x_ahead, b, arm)
                    xy = th.tensor([x_b, y_b])
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (x_b, y_b, yaw_b), float(tr), float(ti), b, None, None, float(lat), None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(x_ahead), stance_kind="side", side_standoff_m=b)
                        cands.append(st)
                        stats["side_candidates"] += 1
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: no candidate (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        # the deep trunks (translate 0.95) are what reaches the floor (B3), and at a floor stance they trip the 1 cm margin
        # against the floor *mesh* (torso / head "within 1 cm of the floor" — B3 README; scene-specific, the floor object
        # sits a little above floor_z): the stance check runs with the floor object removed, the cuRobo ground slab
        # (which stays) keeps the robot from penetrating the floor. Documented relaxation for the push-under stance.
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        cands_all, free_all = list(cands), free
        deep_fail = [st for st, f in zip(cands, free.tolist()) if not f and st.translate >= 0.95]
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if deep_fail and not any(st.translate >= 0.95 for st in cands) and not getattr(self, "_deep_debugged", False):
            # which spheres / postures fail at the floor-reaching trunks (B3: "torso / head within 1 cm of the floor")
            self._deep_debugged = True
            with self.ikc.ignoring([self.obj("floor")]):
                P0.debug_collisions(self, deep_fail[len(deep_fail) // 2:len(deep_fail) // 2 + 2], arms_excluded=False)
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: every candidate collides (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        z_bottom = float(lo[2])
        classed = []
        for st, cam in zip(cands, cams):
            c_d = th.as_tensor(st.extras["drop_cell"])
            # the drop footprint (a disc of the half extent around the drop cell, on the floor) and the edge's bottom line at
            # the push lane in frame (5 %); the optical axis within 20° of the drop cell
            disc = th.stack([th.cat([c_d + tangent * half, th.tensor([self.floor_z])]), th.cat([c_d - tangent * half, th.tensor([self.floor_z])]),
                             th.cat([c_d + d2 * half, th.tensor([self.floor_z])]), th.cat([c_d - d2 * half, th.tensor([self.floor_z])])])
            e0 = mid + tangent * float(st.lateral)
            lane = th.cat([e0, th.tensor([z_bottom])]).reshape(1, 3)  # the edge's bottom point at the push lane
            # a side stance looks along the edge: the lane point sits at the frame's lateral border (45–49° off-axis at
            # b = 0.35–0.40 m); the drop footprint in frame is what it needs (the partner's side fills that half of the image)
            pts = disc if st.extras.get("stance_kind") == "side" else th.cat([disc, lane])
            kind = st.extras.get("stance_kind", "frontal")
            if not P0.in_frame(self, cam, pts):
                stats[f"{kind}_out_of_frame"] = stats.get(f"{kind}_out_of_frame", 0) + 1
                if kind == "side" and stats[f"{kind}_out_of_frame"] == 1:
                    log.info("%s: side stance C example out of frame: base %s trunk %s camera %s focus %s disc %s", self.gp.variant, [round(float(v), 2) for v in st.base_pose2d],
                             (st.translate, st.tilt), [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()],
                             [[round(float(v), 2) for v in p] for p in disc.tolist()])
                continue
            ang = P0.axis_angle(self, cam, st.focus)
            # at the crouched trunks the head hangs ≈ 0.6 m ahead of the base, i.e. over the lane: the drop cell lies almost
            # straight below the camera, 30–40° off the (74–90° pitched) optical axis while squarely in frame — the 20° rule
            # is relaxed to ``push_under_axis_max_deg`` (45°) for the push-under stance (deviation from §5.6 to record)
            if ang > math.radians(float(self.num("push_under_axis_max_deg", 45.0))):
                stats[f"{kind}_axis_off"] = stats.get(f"{kind}_axis_off", 0) + 1
                if kind == "side" and stats[f"{kind}_axis_off"] == 1:
                    log.info("%s: side stance C example axis off by %.0f deg: trunk %s camera %s focus %s", self.gp.variant, math.degrees(ang), (st.translate, st.tilt),
                             [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()])
                continue
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        out_ = P0.best_class(classed)
        stats["survivor_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in out_})
        stats["survivor_stand_offs"] = sorted({round(float(st.stand_off), 2) for st in out_})
        stats["survivor_kinds"] = {k: sum(1 for st in out_ if st.extras.get("stance_kind") == k) for k in ("side", "frontal")}
        stats["side_survivors_by_ahead"] = {str(a): sum(1 for st in out_ if st.extras.get("stance_kind") == "side" and abs(float(st.extras.get("drop_ahead_m", -1)) - a) < 1e-6)
                                            for a in side_ahead}
        side_fail = [st for st, f in zip(cands_all, free_all.tolist()) if not f and st.extras.get("stance_kind") == "side" and float(st.extras.get("drop_ahead_m", 0)) >= 0.45]
        if side_fail and not any(st.extras.get("stance_kind") == "side" and float(st.extras.get("drop_ahead_m", 0)) >= 0.45 for st in out_) \
                and not getattr(self, "_side_debugged", False):
            self._side_debugged = True
            with self.ikc.ignoring([self.obj("floor")]):
                P0.debug_collisions(self, side_fail[len(side_fail) // 2:len(side_fail) // 2 + 2], arms_excluded=False)
        stats["collision_free_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in cands})
        log.info("%s: push-under stance C at %s edge %s: %d survivors (%s)", self.gp.variant, partner.name, edge.name, len(out_), stats)
        return out_

    def debug_drop_ik(self, place: Stance, release_xy, yaws: Sequence[float]) -> str:
        """Diagnostic when F_goal is empty at the drop cell: for the first yaw and every library grasp, the drop grasp pose
        from stance C with the floor ignored vs. with every collision object ignored (reach vs. collision), and the eef
        height above the floor / distance ahead of the base."""
        try:
            floor = self.obj("floor")
            others = [o for o in self.world.scene.objects if o.name in self.world.scene_object_names]
            yaw = float(yaws[0]) if len(yaws) else 0.0
            quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
            pose = self.drop_pose_at(release_xy, quat)
            parts = []
            for g in self.grasps[:3]:
                eef = IK.grasp_pose(pose, g.pose)
                r1 = self.ikc.exact([eef], place, self.seg_arm, ignore_for_last=[floor], cache=False)
                with self.ikc.ignoring(others):
                    r2 = self.ikc.exact([eef], place, self.seg_arm, cache=False)
                p_base = P0.to_base_frame(eef[0], place.base_pose2d, P0.base_z(self))
                parts.append(f"{g.name}: floor-ignored {r1.ok}, world-free {r2.ok}, eef in base {[round(float(v), 2) for v in p_base.tolist()]}")
            return f"stance d {place.stand_off:.2f} trunk ({place.translate}, {place.tilt:.1f}) drop ahead {place.extras.get('drop_ahead_m')}: " + "; ".join(parts)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            return f"debug failed: {e}"

    @staticmethod
    def _snap(band_world: th.Tensor, goal_cells: th.Tensor, lattice: float) -> th.Tensor:
        """Goal cells within half a lattice step of a band cell (the band admits them)."""
        if len(band_world) == 0 or len(goal_cells) == 0:
            return th.zeros(0, 2)
        d = th.cdist(goal_cells.float(), band_world.float())
        return goal_cells[d.min(dim=1).values <= 0.51 * lattice]

    def goal_pose_at(self, goal_cell, quat, eef_in_object=None) -> G.Pose:
        """§5.6 goal pose: ontop → z = partner face + bottom offset + 1 cm; under → z = floor + bottom offset + 3 mm (``eef_in_object``,
        the grasp of the inside mode's per-grasp release height, is unused here)."""
        z = self.place["z"] + self.world.bottom_offset(self.hull, quat) + self.num("goal_drop_m", 0.01 if self.relation == "ontop" else 0.003)
        return th.tensor([float(goal_cell[0]), float(goal_cell[1]), z]), quat

    def screen_release_tilts(self, F_goal, release_xy, yaws, d_yaw: float, place: Stance, arm: str, ignore_place, yaw_step: float):
        """Release orientations screened besides the level one at the goal cell (the inside mode's ``release_tilt_deg``): ``(F_goal, {pair
        key: [tilt deg, ...]})`` - here the level F_goal alone."""
        return F_goal, {}

    def release_quat(self, level_quat, grasp, tilt_deg: float):
        """The release orientation of ``grasp`` at ``tilt_deg`` (the inside mode's tilt); here the level orientation."""
        return level_quat

    def release_lift_per_grasp(self) -> bool:
        """Whether the release pose depends on the drawn grasp (the inside mode's ``release_lift_per_grasp``, owner 2026-10-13 ~00:50 item 3);
        here never."""
        return False

    def pre_place_of(self, place_eef: G.Pose, place: Stance) -> G.Pose:
        """The pre-place of a release at ``place_eef`` from stance C: ``PRE_DIST`` back along the gripper's approach (``IK.pre_pose``); the
        inside mode may override it (``generation.inside_pre_outward``: straight out of the opening, oracle-fix 2026-11-04 round 3). A push-under
        drop with ``generation.drop_pre_m``: that distance instead (the briefcase: the drop's pre-place fingertip left the crouched stance's frame)."""
        d = self.gp.gen("drop_pre_m") if getattr(self, "push_under", False) else None
        # oracle-fix 2026-11-04 round 3 (pillow 158): opt-in generation.place_pre_up_m - an ontop release's pre-place straight above the place (a level
        # pinch's pre-place 10 cm back along the approach kept the carried pillow at mattress height, against the bed: every C attached IK failure)
        up = self.gp.gen("place_pre_up_m") if (self.relation == "ontop" and not getattr(self, "push_under", False)) else None
        if up is not None:
            return G.t(place_eef[0]).float() + th.tensor([0.0, 0.0, float(up)]), G.t(place_eef[1]).float()
        return IK.pre_pose(place_eef, float(d)) if d is not None else IK.pre_pose(place_eef)

    def pre_place_tag(self, place: Stance) -> Optional[str]:
        """A cache tag when :meth:`pre_place_of` differs from ``IK.pre_pose`` at ``place`` (None: the default)."""
        d = self.gp.gen("drop_pre_m") if getattr(self, "push_under", False) else None
        up = self.gp.gen("place_pre_up_m") if (self.relation == "ontop" and not getattr(self, "push_under", False)) else None
        if up is not None:
            return f"up {float(up):.3f}"
        return None if d is None else f"drop_pre {float(d):.3f}"

    def place_ignores(self) -> List[Any]:
        """The objects out of the world at the release's last pose (the place / drop): the partner under an ontop place, else the floor;
        the touching lean (``iors_large_touching.LeanTouchingMode``) adds the partner it leans on (oracle-fix 2026-11-04 round 3)."""
        return [self.partner] if self.relation == "ontop" else [self.obj("floor")]

    def release_feasible(self, release_xy, yaws, release_pose, place: Stance, arm: str, ignore_place) -> List[Tuple[float, Any]]:
        """F_goal at the release cell for the C yaws ``yaws``: the (yaw, library row) pairs whose [pre-place, place] pass the screening IK
        with the target at ``release_pose(release_xy, yaw * rest)``; with :meth:`release_lift_per_grasp` each row at its own release
        (``release_pose(xy, quat, row.pose)``) in the same batch."""
        q_yaw = lambda yaw: G.qmul(G.quat_from_yaw(yaw), self.rest)  # noqa: E731
        tag = getattr(self, "pre_place_tag", lambda _p: None)(place)
        pre_kw = {"pre_of": (lambda eef: self.pre_place_of(eef, place)), "pre_tag": tag} if tag else {}
        if self.release_lift_per_grasp():
            out = self.ikc.feasible_sets([release_xy], yaws, self.grasps, lambda xy, yaw, g: release_pose(xy, q_yaw(yaw), g.pose), place, arm, ignore_place,
                                         pose_per_grasp=True, **pre_kw).get(0, [])
        else:
            out = self.ikc.feasible_sets([release_xy], yaws, self.grasps, lambda xy, yaw: release_pose(xy, q_yaw(yaw)), place, arm, ignore_place, **pre_kw).get(0, [])
        if not out and os.environ.get("ISPATIALGYM_STANCE_BLAME"):
            self.blame_release(release_xy, yaws, release_pose, place, arm, ignore_place, q_yaw)
        elif out and os.environ.get("ISPATIALGYM_STANCE_BLAME_VALIDATE") and getattr(self, "_blame_validated", 0) < 2:
            # round 3: the blame on a pair the screening IK found feasible must say "reached" (validation of the blame itself)
            self._blame_validated = getattr(self, "_blame_validated", 0) + 1
            self.blame_release(release_xy, [out[0][0]], release_pose, place, arm, ignore_place, q_yaw, grasps=[out[0][1]], label="VALIDATE (a feasible pair)")
        return out

    def blame_release(self, release_xy, yaws, release_pose, place: Stance, arm: str, ignore_place, q_yaw, grasps=None, label: str = "F_goal") -> None:
        """Opt-in (``ISPATIALGYM_STANCE_BLAME``, oracle-fix 2026-11-04 owner items 2 / 3): why F_goal is empty - for a few (yaw, grasp) pairs
        the [pre-place, place] eef poses reached with collisions off from the stance (:meth:`_place_ik_contacts`: mm short / joint limit,
        or the links x meshes in contact). ``grasps`` / ``label``: the rows tried (default the first two) and the log tag (the
        ``ISPATIALGYM_STANCE_BLAME_VALIDATE`` run on a feasible pair)."""
        try:
            per = self.release_lift_per_grasp()
            for yaw in list(yaws)[:: max(1, len(yaws) // 4)][:4]:
                for g in (list(grasps) if grasps is not None else list(self.grasps)[:2]):
                    obj = release_pose(release_xy, q_yaw(yaw), g.pose) if per else release_pose(release_xy, q_yaw(yaw))
                    eef = IK.grasp_pose(obj, g.pose)
                    pre = IK.pre_pose(eef)
                    q0 = self.ikc.q_locked(place)
                    log.info("%s: STANCE_BLAME %s at %s (C base %s trunk (%.2f, %+.2f)) yaw %.2f grasp %s: pre-place: %s | place: %s", self.gp.variant, label,
                             [round(float(v), 2) for v in release_xy[:2]], [round(float(v), 2) for v in place.base_pose2d], place.translate, place.tilt, float(yaw),
                             getattr(g, "name", g), self._place_ik_contacts(pre, q0, arm, [])[:260], self._place_ik_contacts(eef, q0, arm, list(ignore_place))[:260])
        except Exception as e:  # noqa: BLE001 - a diagnostic never fails the draw
            log.info("%s: STANCE_BLAME F_goal: blame failed (%r)", self.gp.variant, e)

    def release_eef_in_object(self) -> Optional[G.Pose]:
        """The recorded grasp's ``T_obj→eef`` the release pose is built for when :meth:`release_lift_per_grasp`, else None (the goal and
        the settle re-derive the release the C exact IK used)."""
        if not self.release_lift_per_grasp():
            return None
        rec = self.sidecar.grasp_of(self.tname)
        return None if rec is None else rec.eef_in_object

    # ------------------------------------------------------------------ template

    def sample_template(self):
        self.template_facts.update({"relation": self.relation, "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "band_union_extent": None if self.surface.get("band") is None else list(self.surface["band"].union_extent())})

    # ------------------------------------------------------------------ instance (§5.6)

    def pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
        return th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull, quat) + 0.003]), quat

    def yaws(self) -> List[float]:
        """The object yaws screened for F_init / F_goal: ``generation.n_yaws`` equal steps (default 16 - the 8 of the reach
        bands leave 2-4 feasible pairs for a floor pick at the crouch)."""
        n = int(self.num("n_yaws", 16))
        return [k * 2.0 * math.pi / n for k in range(n)]

    def init_yaw_jitter(self) -> float:
        """The init yaw's draw around the screened yaw (± this, rad; the B exact IK's retries too): :data:`TARGET_JITTER`."""
        return TARGET_JITTER

    def end_aligned_init_yaw(self, yaw0: float, d_yaw: float):
        """The init yaw near the screened ``yaw0`` whose carried goal yaw (``+ d_yaw``) a mode aligns (the touching mode's
        ``goal_align_partner``); None here (the drawn yaw as screened), False for a pair the alignment cannot use."""
        return None

    def alignable_draws(self, F: Sequence[Tuple[float, Any]], draw_order: Sequence[int], d_yaw: float) -> List[int]:
        """``draw_order`` without the pairs of ``F`` that :meth:`end_aligned_init_yaw` rejects at this C's ``d_yaw`` (as is when no mode
        aligns), so that the ``f_draws_max`` draws of a C are all usable."""
        ends = [self.end_aligned_init_yaw(float(F[int(i)][0]), d_yaw) for i in draw_order]
        if all(e is None for e in ends):
            return list(draw_order)
        return [i for i, e in zip(draw_order, ends) if e is not False]

    # ------------------------------------------------------------------ legs A -> B and B -> C (owner 2026‑09‑19 21:56)

    def _drop_leg_records(self, to: str) -> None:
        self.sidecar.paths = [p for p in self.sidecar.paths if p.to != to]
        self.sidecar.transitions = [t for t in self.sidecar.transitions if t.to != to]

    def record_leg_a_b(self, init: Stance, manip: Stance) -> Dict[str, Any]:
        """(ii) Init -> grasp stance. Primary: **navigate at the init trunk, then the torso** — the §3.7 path at A's
        trunk / arms and the §3.8 arrival transition at B (trunk sweep at s_T on B's axis, drive-in with B's arms).
        Fallback when no s_T admits that: **torso first, then navigate** — a departure transition at A (drive back
        along A's own axis if needed, sweep to B's trunk there) and the path driven at B's trunk with B's arms from that
        pose (what ``729a1ec`` did by spawning A in B's crouch; A itself stays upright, (i)). ``Reject`` when neither
        leg exists. Returns the leg facts for the summary."""
        errors: List[str] = []
        try:
            path = self.record_path(init, manip)
            tr = self.record_transition(init, manip)
            return {"order": "navigate_then_torso", "s_T": tr.stand_off, "path_m": round(path.length_m, 3)}
        except Reject as e:
            if e.reason not in ("no_transition", "path_collision", "no_path"):
                raise
            errors.append(f"navigate-then-torso: {e.reason}: {e}")
            self._drop_leg_records(manip.name)
        b_trunk = (float(manip.translate), float(manip.tilt))
        try:
            from ispatialgym.sampling.v2.ik import stance_arm_vectors

            dep = self.record_departure_transition(init, manip, b_trunk)
            r, l, _ = stance_arm_vectors(self.kit, manip)  # the oracle moves both arms to B's posture before the sweep
            path = self.record_path(init, manip, trunk=b_trunk, arm_q={"right": r, "left": l}, start_pose2d=NAV.departure_pose(init, dep.stand_off))
            # oracle-fix 2026-11-06 (survey1 under_large#02: 3 / 3 demos 'collision | navigate | right_realsense_link in contact with
            # walls_mrghjk_0'): the path is swept with B's arms (the departure put them there), so the record carries them as
            # core.record_edge_travel's does (2297006d) - without paths[].arms_q the oracle's navigate re-posed the empty right arm from
            # B's 'ready' to untucked and drove the untucked wrist camera into the wall at (18.17, -11.47), where the primary
            # navigate-then-torso sweep (untucked, trunk 0.5 / 0) had already found the leg colliding
            if not path.carried and path.arms_q is None:
                path.arms_q = {"right": [round(float(v), 5) for v in r.tolist()], "left": [round(float(v), 5) for v in l.tolist()]}
            log.info("%s: A -> B: navigate-then-torso impossible (%s); torso-then-navigate at A (s_T %.2f, drive at trunk %s)", self.gp.variant, errors[-1], dep.stand_off,
                     tuple(round(v, 2) for v in b_trunk))
            return {"order": "torso_then_navigate", "s_T": dep.stand_off, "path_m": round(path.length_m, 3), "primary_error": errors[-1]}
        except Reject as e:
            self._drop_leg_records(manip.name)
            errors.append(f"torso-then-navigate: {e.reason}: {e}")
        raise Reject("no_transition", "init -> manipulation: " + "; ".join(errors))

    def _leg_ab_clear_of_target(self, init: Stance, manip: Stance, t_pose) -> Optional[str]:
        """Owner 2026-09-26 (shampoo 0926-015905, wave 8: navigate[0] 'wheel_motor_link1 in contact with the dispenser' twice - the
        recorded A -> B line passed 5.6 cm from the dispenser standing on the floor): the A -> B leg is recorded before the target
        has a pose (``record_leg_a_b`` runs with the target parked), so its sweep never saw where the target ends up - harmless for
        a target on a table, not for one on the floor B drives up to. Re-sweep the recorded A -> B path (its strides + the
        yaw-lag envelope, the trunk and arms it was checked with, the floor relaxation of the original check) with the target
        placed at ``t_pose`` (a face_travel record: its schedule, ``NAV.record_sweep_poses``); returns the reason when it collides, None
        when it clears (or when there is no path)."""
        path_ab = next((p for p in self.sidecar.paths if p.to == manip.name), None)
        if path_ab is None or len(path_ab.waypoints) < 2:
            return None
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        trunk = tuple(float(v) for v in path_ab.trunk) if path_ab.trunk is not None else (float(init.translate), float(init.tilt))
        at_init = abs(trunk[0] - float(init.translate)) < 1e-6 and abs(trunk[1] - float(init.tilt)) < 1e-6
        r, l, _ = stance_arm_vectors(self.kit, init if at_init else manip)
        q_arms = {"right": r, "left": l}
        if path_ab.arms_q:
            q_arms.update({a: th.as_tensor(q, dtype=th.float32) for a, q in path_ab.arms_q.items()})
        tgt = self.target
        p0, q0 = (x.clone() for x in tgt.get_position_orientation())
        try:
            tgt.set_position_orientation(*t_pose)
            tgt.keep_still()
            self.kit.sync_world()
            # oracle-fix 2026-11-06: a face_travel A -> B record is swept as its schedule (+ the yaw-lag envelope), what the oracle drives
            poses = NAV.record_sweep_poses(path_ab)
            with self._floor_relaxed():
                ok, bad = NAV.path_collision_free(self.kit, poses, trunk, path_ab.posture, margin=0.01, q_arms=q_arms)
        finally:
            tgt.set_position_orientation(p0, q0)
            tgt.keep_still()
            self.kit.sync_world()
        if ok:
            return None
        return (f"B: the A -> B drive passes the target drawn at ({float(t_pose[0][0]):.3f}, {float(t_pose[0][1]):.3f}): swept pose {bad}/{len(poses)} "
                f"({tuple(round(float(v), 3) for v in poses[bad])}) collides with it placed")

    def travel_trunks(self, init: Stance, manip: Stance) -> List[Tuple[float, float]]:
        """(iii') carry trunks in order: the grasp stance's translate with the torso straightened (tilt 0 — the object
        stays low), then the init translate straightened; ``generation.travel_trunks`` overrides."""
        cfg = self.gp.gen("travel_trunks")
        if cfg:
            return [(float(t), float(k)) for t, k in cfg]
        out: List[Tuple[float, float]] = []
        for t in (float(manip.translate), float(init.translate)):
            if all(abs(t - u) > 1e-6 for u, _ in out):
                out.append((t, 0.0))
        return out

    def record_carry_leg(self, init: Stance, manip: Stance, place: Stance, tname: str, q_lift_arm: Optional[th.Tensor], max_len: float) -> Any:
        """(iii') Grasp stance -> place stance with the object in hand. Primary: **straighten the torso, then navigate**
        — a departure transition at B (drive back along B's own axis to the first s_T where the sweep to the travel trunk
        is free — "a spot along the way it has moved before" — with the holding arm at its LIFT configuration and the
        object attached), the path from that pose at the travel trunk, and the arrival transition at C (the pre-place
        stand-off where C's trunk is set, then the drive-in). Every travel trunk of :meth:`travel_trunks` is tried.
        Fallback: the drive at B's trunk (the earlier behaviour) with the arrival transition. The holding arm is the segment arm.
        Returns the path record."""
        arm = self.seg_arm
        if q_lift_arm is not None:
            self.record_carry_arm_q(tname, arm, q_lift_arm)
        # the placement stance is reached WITH the object in hand: the objects-placed check attaches it there (and ignores it at
        # its init pose - 0920-004500: stance C 'collided' with the target still standing on its floor cell next to C)
        place.carried = tname
        errors: List[str] = []
        # oracle-fix 2026-11-04 round 4 (opt-in generation.carry_departure_min_back_m, pillow 158): the departure drives back at least this far from B
        # before the path turns (probe_pillow: the path turned 0.41 m from bed_kddswt_1's side, a wheel sphere 8.7 mm from it, in the 0.9 m corridor)
        back = self.gp.gen("carry_departure_min_back_m")
        dep_kw = {} if back is None else {"min_standoff": float(manip.stand_off or 0.0) + float(back)}
        # oracle-fix 2026-11-06 (opt-in generation.carry_arrive_straight_m, hardback inside_large#01): the carried path first ends that far behind C
        # along C's axis and drives straight in (NAV.plan_path arrive_straight_m), so the turn to C's heading happens away from the partner;
        # the plain path after it (local_hb 2026-11-06: every carry turned into C 0.6 m from bookcase_jaysra_0 with the book ~0.8 m ahead, and
        # the straightened base's wheels touched bed_rrcvaq_1 0.37 m away on the same turn)
        arrive = float(self.gp.gen("carry_arrive_straight_m") or 0.0)
        variants = ([{"arrive_straight_m": arrive}] if arrive > 0.0 else []) + [{}]
        k_plain = len(variants) - 1
        if carry_sideways_on():  # i7-goalmaps (lead decision): the sideways drive after every variant of today (CARRY_SIDEWAYS_ENV)
            variants += [{"heading_offset": o} for o in sideways_offsets(manip.base_xy.tolist(), place.base_xy.tolist(), float(place.yaw))]
        if carry_wide_path_on(self.gp):  # i7-goalmaps (lead decision): a global path kept farther from obstacles (CARRY_WIDE_PATH_ENV)
            variants += [{"global_extra_erosion_m": w} for w in CARRY_WIDE_PATH_EROSIONS_M]
        chosen: Dict[str, Any] = {}

        def carried_path(redo=None, **kw):
            """The carried path B -> C: per variant (straight arrival first) the local path, then the eroded global one on a path_collision (the
            local shortest path hugs the partner's corner with the object out front, 0920-100543); a straight arrival that finds no path at all
            gives way to the plain one. ``redo`` re-records what a failed try dropped (the departure transition). i7-goalmaps: the opt-in EXTRA
            variants after the plain one (CARRY_SIDEWAYS_ENV, CARRY_WIDE_PATH_ENV) are tried after the plain one's path_collision; whatever they
            fail with, the plain one's error is what the leg raises."""
            tries = [(k, v, force) for k, v in enumerate(variants) for force in (False, True)
                     if not (force is False and "global_extra_erosion_m" in v)]  # a wide-path variant is global only
            i = 0
            plain_err: Optional[Reject] = None
            while True:
                k, v, force = tries[i]
                plain = k >= k_plain
                extra = k > k_plain
                if "heading_offset" in v:
                    TT._count(self, "carry_path.sideways_tried")
                if "global_extra_erosion_m" in v:
                    TT._count(self, "carry_path.wide_tried")
                try:
                    p = self.record_path(manip, place, carried=tname, arm=arm, max_length_m=max_len, force_global=force, **kw, **v)
                    chosen.clear()
                    chosen.update(v)
                    if "heading_offset" in v:
                        TT._count(self, "carry_path.sideways_used")
                        log.info("%s: B -> C carry: SIDEWAYS path (heading offset %+.2f rad, %s) %.2f m", self.gp.variant, float(v["heading_offset"]),
                                 "global" if force else "local", p.length_m)
                    if "global_extra_erosion_m" in v:
                        TT._count(self, "carry_path.wide_used")
                        log.info("%s: B -> C carry: WIDE path (the global map eroded %.2f m more) %.2f m", self.gp.variant, float(v["global_extra_erosion_m"]),
                                 p.length_m)
                    return p
                except Reject as e:
                    if extra:
                        log.info("%s: B -> C carry: extra variant %s %s: %s: %s", self.gp.variant, v, "global" if force else "local", e.reason, str(e)[:300])
                        i += 1
                        if i >= len(tries):
                            raise plain_err if plain_err is not None else e
                    else:
                        if plain and e.reason != "path_collision":
                            raise
                        if plain:
                            plain_err = e
                        if not plain:
                            log.info("%s: B -> C carry: the straight arrival (%.2f m) %s: %s: %s", self.gp.variant, arrive, "global" if force else "local", e.reason,
                                     str(e)[:600])
                        # the next try: the global path after a local collision, else the next variant (a straight arrival without any path skips its global try)
                        i = i + 1 if e.reason == "path_collision" else next(j for j, t in enumerate(tries) if t[0] == k + 1)
                        if i >= len(tries):
                            raise
                    self._drop_leg_records(place.name)
                    if redo is not None:
                        redo()

        for tt in self.travel_trunks(init, manip):
            if abs(tt[0] - manip.translate) < 1e-6 and abs(tt[1] - manip.tilt) < 1e-6:
                continue  # B already straight: the plain leg below is the same thing
            try:
                dep = self.record_departure_transition(manip, place, tt, carried=tname, arm=arm, **dep_kw)
                start = NAV.departure_pose(manip, dep.stand_off)
                drive_q = self.carry_drive_arms(place, tname, arm)
                path = carried_path(redo=lambda tt=tt: self.record_departure_transition(manip, place, tt, carried=tname, arm=arm, **dep_kw),
                                    trunk=tt, start_pose2d=start, arm_q=drive_q)
                # a carry into a deep crouch backs off further before the sweep (the hand frozen at the lift pose swings ~0.5 m
                # forward / 0.9 m down): s_T up to stand-off + generation.carry_transition_extra_m (1.0)
                self.record_transition(manip, place, carried=tname, arm=arm, trunk_from=tt, extra_max=self.num("carry_transition_extra_m", 1.0))
                where = "at the grasp spot" if dep.stand_off <= (manip.stand_off or 0.0) + 1e-6 else "after a drive-back"
                log.info("%s: B -> C carry: torso straightened to %s at s_T %.2f (%s), path %.2f m", self.gp.variant, tuple(round(v, 2) for v in tt), dep.stand_off, where,
                         path.length_m)
                self._leg_bc = {"order": "straighten_then_navigate", "travel_trunk": [tt[0], tt[1]], "s_T_departure": dep.stand_off, "straightened": where,
                                "path_m": round(path.length_m, 3), "lift_arm_q": q_lift_arm is not None, "tried": errors,
                                "arrive_straight_m": chosen.get("arrive_straight_m", 0.0), "heading_offset": chosen.get("heading_offset", 0.0)}
                return path
            except Reject as e:
                self._drop_leg_records(place.name)
                errors.append(f"travel trunk {tuple(round(v, 2) for v in tt)}: {e.reason}: {e}")
                # oracle-fix 2026-11-06 (hardback): the carry leg's error text is cut at 700 characters by its caller; each travel trunk's whole reason
                log.info("%s: B -> C carry: travel trunk %s failed: %s: %s", self.gp.variant, tuple(round(v, 2) for v in tt), e.reason, str(e)[:1500])
        # fallback: drive at B's trunk (the object low in the crouch), the trunk set at C's pre-place stand-off
        try:
            path = carried_path()
            self.record_transition(manip, place, carried=tname, arm=arm, extra_max=self.num("carry_transition_extra_m", 1.0))
            log.info("%s: B -> C carry: no travel trunk admits straighten-then-navigate (%s); driving at B's trunk %s", self.gp.variant, "; ".join(errors),
                     (manip.translate, round(manip.tilt, 2)))
            self.template_facts.setdefault("carry_fallbacks", []).append(errors)
            self._leg_bc = {"order": "drive_at_grasp_trunk", "travel_trunk": [manip.translate, manip.tilt], "path_m": round(path.length_m, 3),
                            "lift_arm_q": q_lift_arm is not None, "tried": errors, "arrive_straight_m": chosen.get("arrive_straight_m", 0.0),
                            "heading_offset": chosen.get("heading_offset", 0.0)}
            return path
        except Reject as e:
            self._drop_leg_records(place.name)
            raise Reject(e.reason, f"{e}; straighten-then-navigate: " + "; ".join(errors))

    def carry_drive_arms(self, place: Stance, tname: str, arm: str) -> Optional[Dict[str, th.Tensor]]:
        """The arms (arm -> 7 joints) the carried drive after B's departure transition is checked with; None = :meth:`record_path`'s
        default (the departing stance's arms, the holding arm at its carry configuration). The inside mode overrides it."""
        return None

    #: carry pose candidates in B's base frame: (ahead, lateral, raise above the lift) — beside the body first (owner 2026-09-20
    #: 06:40: a pose in front of the chest swings INTO the base when the torso tilts 54 deg into the push-under crouch; a hand
    #: held to the right of the body swings down outside the base footprint and ends above the floor), then in front
    #: (0920-071230: the poses 0.20-0.30 m ahead fold the elbow onto its limit - right_arm_joint4 within 2-19 mrad, the 0.05 rad
    #: witness margin rejects them - so the beside-the-body poses sit 0.40-0.55 m ahead; the raise stays >= 0 so the straight
    #: line from the lift never dips toward the partner's top edge). The right arm's; the left arm holds them mirrored (:func:`carry_poses_for`)
    CARRY_POSES = ((0.45, -0.45, 0.0), (0.50, -0.40, 0.05), (0.40, -0.45, 0.10), (0.55, -0.35, 0.05), (0.35, -0.45, 0.15), (0.50, None, 0.10))

    def carry_poses_at_b(self, manip: Stance, eef: G.Pose, carry_height: float, target, eef_in_object) -> List[Tuple[G.Pose, List[G.Pose], th.Tensor]]:
        """(iii') the CARRY poses (owner 2026‑09‑20 05:20): after the lift the held object is brought to a compact pose —
        ``CARRY_POSES`` / ``generation.carry_poses`` as (ahead, lateral (None = the lift's), raise) in B's base frame,
        orientation kept — by a straight in-hand line (``Reorient`` mode "line", 3 intermediate waypoints + end), so that
        the trunk sweeps of the carry legs move a hand near the body, not one stretched 0.85 m over the partner
        (0920-043725: the frozen lift configuration swept the held sandal into the table top / the floor at every s_T up to
        1.75 m). Solved from B with the object attached: exact IK at the carry pose, §3.9 waypoints from the lift; the segment
        arm holds it (the poses beside the body on its side). Returns the feasible ``(carry pose, waypoints, 7 arm joints)``
        candidates in order (the legs decide)."""
        arm = self.seg_arm
        lift = (eef[0] + th.tensor([0.0, 0.0, float(carry_height)]), eef[1])
        base_xy = manip.base_xy
        fwd, left = manip.forward, manip.left
        rel = th.tensor([float(lift[0][0]), float(lift[0][1])]) - base_xy
        lat_lift = float(rel @ left)
        out: List[Tuple[G.Pose, List[G.Pose], th.Tensor]] = []
        why: List[str] = []
        cands = [(ahead, lat, raise_m, False) for ahead, lat, raise_m in carry_poses_for(self.gp.gen("carry_poses") or self.CARRY_POSES, arm)]
        if carry_centred_on(self.gp):  # i7-goalmaps (lead decision): the object centred on the body's centre line first (CARRY_CENTRED_ENV)
            lat_c = centred_hand_lateral(lift, eef_in_object, self.hull, left)
            cands = [(a, lat_c, r, True) for a, r in CARRY_CENTRED_POSES] + cands
        self._carry_centred_n = 0
        for ahead, lat, raise_m, centred in cands:
            lat_v = lat_lift if lat is None else float(lat)
            xy = base_xy + fwd * float(ahead) + left * lat_v
            carry = (th.tensor([float(xy[0]), float(xy[1]), float(lift[0][2]) + float(raise_m)]), lift[1])
            tag = f"({float(ahead):.2f}, {lat_v:+.2f}, z {float(carry[0][2]):.2f})"
            # the in-hand move is a manipulation action: the FOV monitor is armed (owner 2026-09-16 10:15) - every waypoint incl. the
            # carry pose must keep the fingertips in the head camera frame from B (0920-135626: reorient failed gripper_out_of_fov at
            # u = 1060 for the carry pose (0.55, -0.35))
            cam_b = self.camera_pose_at(manip)
            wps = list(C.line_waypoints(lift, carry, 4))
            if not all(self.ikc.eef_in_frame(w, cam_b) for w in wps):
                why.append(f"{tag}: out of the head camera frame")
                continue
            with attached(self.kit, arm, target, eef_in_object=eef_in_object):
                r = self.ikc.exact([carry], manip, arm, cache=False)
                if not r.ok or r.q is None:
                    why.append(f"{tag}: IK {r.reason}")
                    continue
                ok = self.ikc.waypoints_ok([lift] + wps, manip, arm, cache=False)
            if not ok.ok:
                why.append(f"{tag}: line from the lift {ok.reason}")
                continue
            out.append((carry, wps, r.q[self.kit.robot.arm_control_idx[arm]].clone()))
            if centred:
                self._carry_centred_n += 1
                TT._count(self, "carry_pose.centred_offered")
        log.info("%s: carry poses from B (lift %.2f m at %s in base): %d feasible%s", self.gp.variant, carry_height,
                 [round(float(v), 2) for v in P0.to_base_frame(lift[0], manip.base_pose2d, P0.base_z(self)).tolist()], len(out), ("; " + "; ".join(why)) if why else "")
        return out

    #: oracle-fix 2026-11-07 (owner prompt A 3c, the hardback): compact carry poses probed at the TRAVEL trunk (B's base, the torso straightened),
    #: in front of the chest and higher than the lift: (ahead, lateral toward the arm's side (right arm -), world z); fix3b_hb: every carry
    #: pose of CARRY_POSES failed at B (5 x out of the head camera frame, 1 x joint 4 / 6 within 2 mrad) and all 12 instances carried at the
    #: lift configuration (hand 0.51 m ahead, z 0.90), whose leg at C swept the book into bookcase_jaysra_0 (81 carry-leg-at-C failures)
    CARRY_COMPACT_PROBE_POSES = tuple((a, l, z) for a in (0.30, 0.38, 0.46) for l in (0.0, -0.12, -0.24) for z in (1.00, 1.10, 1.20))
    CARRY_COMPACT_PROBE_MAX = 12

    def carry_compact_probe(self, manip: Stance, place: Stance, arm: str, tname: str, target, g, eef: G.Pose, pre_place: G.Pose, place_eef: G.Pose,
                            ignore_place, lift_h: float) -> Optional[Dict[str, Any]]:
        """Opt-in ``ISPATIALGYM_CARRY_COMPACT_PROBE=1`` (oracle-fix 2026-11-07, a diagnostic: nothing recorded changes): after a carry leg at C
        failed, each :data:`CARRY_COMPACT_PROBE_POSES` pose (the lift's orientation) at B's base with the travel trunk - the head camera's
        eef box there (``eef_in_frame``), the exact IK with the object attached (joint-limit margins included), the in-hand line from the
        current carry configuration, and the carry leg at C from it (:meth:`_carry_track_at_c`, the carry configuration swapped in and
        restored). One ``CARRY_COMPACT_PROBE {json}`` line; at most :data:`CARRY_COMPACT_PROBE_MAX` probes per process."""
        if os.environ.get("ISPATIALGYM_CARRY_COMPACT_PROBE", "") != "1":
            return None
        n = self.__dict__.setdefault("_compact_probe_n", [0])
        if n[0] >= self.CARRY_COMPACT_PROBE_MAX:
            return None
        n[0] += 1
        from ispatialgym.oracle import r1pro_model as M

        tt = tuple((getattr(self, "_leg_bc", None) or {}).get("travel_trunk") or (manip.translate, 0.0))
        st = manip.with_trunk(float(tt[0]), float(tt[1]), name=f"{manip.name}~compact_probe")
        cam = self.camera_pose_at(st)
        idx = self.kit.robot.arm_control_idx[arm]
        q_keep = self.carry_arm_q.get(tname)
        q_now = self.ikc.q_locked(st).clone()
        if q_keep is not None:
            q_now[idx] = q_keep
        h_now = self.kit.fk([M.EEF_LINK[arm]], q_now)[M.EEF_LINK[arm]]
        rows: List[Dict[str, Any]] = []
        s = arm_sign(arm)
        try:
            for ahead, lat, z in self.CARRY_COMPACT_PROBE_POSES:
                xy = st.base_xy + st.forward * float(ahead) + st.left * (s * float(lat))
                pose = (th.tensor([float(xy[0]), float(xy[1]), float(z)]), eef[1])
                row: Dict[str, Any] = {"pose": [ahead, s * lat, z], "in_frame": bool(self.ikc.eef_in_frame(pose, cam))}
                with attached(self.kit, arm, target, eef_in_object=g.pose):
                    r = self.ikc.exact([pose], st, arm, cache=False)
                    row["ik"] = "ok" if r.ok else str(r.reason)[:120]
                    if r.ok and r.q is not None:
                        wps = list(C.line_waypoints((G.t(h_now[0]), G.t(h_now[1])), pose, 4))
                        ok = self.ikc.waypoints_ok([(G.t(h_now[0]), G.t(h_now[1]))] + wps, st, arm, cache=False)
                        row["line"] = "ok" if ok.ok else str(ok.reason)[:120]
                        row["line_in_frame"] = bool(all(self.ikc.eef_in_frame(w, cam) for w in wps))
                if r.ok and r.q is not None:
                    self.carry_arm_q[tname] = r.q[idx].clone()
                    res = self._carry_track_at_c(place, arm, tname, target, g, pre_place, place_eef, ignore_place)
                    row["leg_at_c"] = res[3] if res[3] == "ok" else f"{res[3]}: {str(res[2])[:160]}"
                rows.append(row)
        finally:
            if q_keep is not None:
                self.carry_arm_q[tname] = q_keep
            else:
                self.carry_arm_q.pop(tname, None)
        summ = {"n": len(rows), "in_frame": sum(r["in_frame"] for r in rows), "ik_ok": sum(r.get("ik") == "ok" for r in rows),
                "line_ok": sum(r.get("line") == "ok" for r in rows), "leg_ok": sum(r.get("leg_at_c") == "ok" for r in rows),
                "leg_ok_in_frame": sum(r.get("leg_at_c") == "ok" and r["in_frame"] and r.get("line_in_frame", False) for r in rows)}
        rec = {"travel_trunk": [round(float(v), 3) for v in tt], "lift_m": lift_h, "stance_c": place.name, "summary": summ, "rows": rows}
        log.info("CARRY_COMPACT_PROBE %s", json.dumps(rec))
        return rec

    def lift_arm_q(self, manip: Stance, eef: G.Pose, carry_height: float, target, eef_in_object) -> Optional[th.Tensor]:
        """The 7 arm-joint values of the LIFT pose (the grasp raised ``carry_height`` straight up, orientation kept) solved
        from stance B with the object attached — the configuration the oracle carries with (``program_check`` solves the
        same ``lift_end``); None when the lift has no solution (the leg then uses the stance posture, as before). The segment arm's."""
        arm = self.seg_arm
        lift = (eef[0] + th.tensor([0.0, 0.0, float(carry_height)]), eef[1])
        # owner 2026-09-25 (shampoo 0925-171050 / 194056: 8/8 draws lost at the program check's lift - 'lift: 1/11 tracked chain
        # configurations collide' - after this end-pose check had passed): the lift is TRACKED from the grasp configuration exactly
        # as the program check tracks it (IKChecker.continue_chain: the oracle's differential IK, the grasp's supports and the
        # ground off for a floor-class lift), so an untrackable lift drops the grasp / stance here instead of the whole draw later
        q_grasp = self.ikc.solutions.get((arm, self.ikc._pose_key(eef)))
        floor_class = (getattr(self, "surface", None) or {}).get("kind") == "floor"
        ignore = list(self.grasp_ignore_objects())
        with attached(self.kit, arm, target, eef_in_object=eef_in_object):
            if q_grasp is not None and self.ikc.reaches(q_grasp, eef, arm):
                r = self.ikc.continue_chain(q_grasp, [eef, lift], arm, ignore=ignore, ignore_end=[], soft_stall_m=self.ikc.FREE_LEG_STALL_M,
                                            ignore_start=ignore + list(self.support_objects()), ground_along=not floor_class, ground_meshes_along=not floor_class)
                if not r.ok:
                    log.info("%s: lift %.2f m untrackable from the grasp configuration (%s): grasp skipped", self.gp.variant, carry_height, r.reason[:200])
                    return None
            else:
                r = self.ikc.exact([lift], manip, arm, ignore_for_last=ignore, cache=False)
        if not r.ok or r.q is None:
            return None
        return r.q[self.kit.robot.arm_control_idx[arm]].clone()

    # ------------------------------------------------------------------ per-attempt budget

    def _check_deadline(self, stage: str) -> None:
        """Per-attempt time budget (``ATTEMPT_BUDGET_S``): the attempt stops with ``Reject("time_budget")`` naming the stage
        it reached once the budget is spent (a > 2 h attempt is a bug, not a search)."""
        deadline = getattr(self, "_deadline", math.inf)
        if time.time() > deadline:
            spent = time.time() - getattr(self, "_attempt_t0", deadline)
            raise Reject("time_budget", f"attempt budget of {self.num('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent ({spent:.0f} s) at {stage}",
                         stage=stage)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init: Stance = self.surface["init"]
        self.sidecar.add_stance(init)
        kind = self.surface["kind"]
        b_pool = list(self.surface["manip"])
        errors: List[str] = []
        self._attempt_t0 = time.time()
        self._deadline = self._attempt_t0 + float(self.num("attempt_budget_s", ATTEMPT_BUDGET_S))
        focus_xy = self.surface["c_t"] if kind == "floor" else None
        z_init = self.surface["z"]
        yaws = self.yaws()
        # the arms of the segment (owner 2026-09-27, first-demo v2.2): F_init of the default arm at every B, of the other arm where its
        # band holds the target, each at the B with that arm working; a B's usable pairs are the union
        offered = TT.offered_arms(self)
        self.seg_arm = offered[0]
        arm_bands = self.surface.get("arm_bands") or {}
        # F_init pre-screen (countertop-floor 0919-222621: F_init 2-4 pairs per B at the floor crouch, 6 B tried blind and the
        # attempt spent its 900 s): the screening IK of F_init needs no leg, so it runs first over up to ``b_prescreen`` random B
        # candidates and the coupling starts from the B with the most (yaw, grasp) pairs
        n_pre = int(self.num("b_prescreen", 8))
        pre: List[Tuple[int, Stance, List[Any], Dict[str, Tuple[Stance, List[Any]]]]] = []
        ov_opts = self.pick_overhang_opts()

        def _screen(n: int) -> None:
            for i_b in range(min(n, len(b_pool))):
                self._check_deadline(f"F_init pre-screen of B candidate {len(pre) + 1}")
                kb = int(rng.integers(len(b_pool)))
                b_grid = b_pool.pop(kb)
                manip = b_grid.with_trunk(b_grid.translate, b_grid.tilt, "manipulation")
                manip.focus = b_grid.focus
                fxy = focus_xy if focus_xy is not None else manip.focus[:2]
                by_arm: Dict[str, Tuple[Stance, List[Any]]] = {}
                for a, st in self._b_stances(manip, offered).items():
                    if self.v2_overhang is not None and ov_opts is not None and ov_opts["mode"] == "only":
                        F_a = []  # i7-overhang: the direct pairs are dropped in mode only - no batched screen of them
                    else:
                        F_a = self.ikc.feasible_sets([fxy], yaws, self.grasps, lambda xy, yaw: self.pose_on(xy, yaw, z_init), st, a, self.grasp_ignore_objects()).get(0, [])
                    if ov_opts is not None and kind != "floor":
                        F_a = self._overhang_pairs(manip, F_a, fxy, yaws, z_init, ov_opts, arm=a, st=st)
                    by_arm[a] = (st, F_a)
                    if arm_bands.get(a) is not None and arm_bands[a].meta.get("mirrored_from"):
                        TT._count(self, "arm_band_mirrored.foci")
                        if not F_a:
                            TT._count(self, "arm_band_mirrored.empty_F")
                F_b = union_feasible({a: F for a, (_, F) in by_arm.items()}, 2.0 * math.pi / max(1, len(yaws)))
                lead = next((by_arm[a][0] for a in self._segment_order(manip, by_arm)), next(iter(by_arm.values()), (manip, []))[0])
                pre.append((len(F_b), lead, F_b, by_arm))
        _screen(n_pre)
        # order: a COMPACT grasp first (the target cell's distance ahead of the base, 10 cm buckets) among the B candidates whose F_init
        # has at least ``b_prescreen_min_f`` pairs, then the larger F_init - a grasp 0.85 m out over the table (0920-100542/3) leaves an
        # extended arm that neither lifts higher nor rounds the table corner while carrying; ``b_prescreen_order = "f"`` restores F-only
        min_f = int(self.num("b_prescreen_min_f", 8))
        # push-under (2026-09-20 19:50): the object keeps its yaw relative to the robot from B to C, so which face the floor blade meets at
        # C is decided by the init yaw graspable at B - a B whose F_init has no yaw with a large blade contact offset at C (a sandal
        # toe-first: 0.11 m; its strap: 0.03 m, the blade goal then 0.85 m ahead and out of reach, 0920-192539/40) is tried last
        pushable: Dict[int, float] = {}
        if self.push_under and pre:
            faces_of: Dict[str, Any] = {}

            def _faces(a: str):
                """The (base yaw, push direction) of ``a``'s stances C (the left arm's side stances face the other way)."""
                if a not in faces_of:
                    try:
                        pool0 = self.placement_candidates(pre[0][1].focus[:2] if pre[0][1].focus is not None else self.surface["c_t"], arm=a)
                    except Reject:
                        pool0 = []
                    faces_of[a] = {(round(float(c.yaw), 2), tuple(round(float(v), 3) for v in c.extras["push_dir"])) for c in pool0 if c.extras.get("push_dir") is not None}
                return faces_of[a]
            _faces(offered[0])
            min_off_ = float(self.num("push_min_contact_m", 0.10))

            def _offsets() -> None:
                for idx, (_, m, _, by_arm) in enumerate(pre):
                    if idx in pushable:
                        continue
                    best = -1.0
                    for a, (_, F_a) in by_arm.items():
                        if not F_a:
                            continue
                        for c_yaw, pd in _faces(a):
                            d_yaw = 0.0 if str(self.gp.gen("carry_yaw", "relative")) == "world" else float(c_yaw - m.yaw)
                            for y in sorted({float(yy) for yy, _ in F_a}):
                                q = G.qmul(G.quat_from_yaw(y + d_yaw), self.rest)
                                o = self.push_contact_offset(q, th.as_tensor(pd))
                                if o is not None and self.push_yaw_kept(q, pd, o):  # generation.push_under_yaw: the yaws it keeps
                                    best = max(best, o)
                    pushable[idx] = best
            _offsets()
            # no pushable B among the first batch: screen more (up to ``b_prescreen_batches`` batches) before coupling blind
            for _ in range(int(self.num("b_prescreen_batches", 3)) - 1):
                if max(pushable.values(), default=-1.0) >= min_off_ or not b_pool:
                    break
                _screen(n_pre)
                _offsets()
            lock = self.push_yaw_lock()
            log.info("%s: pre-screen blade contact offsets at C per B%s: %s", self.gp.variant, f" (push-under yaw lock {lock})" if lock else "",
                     [round(pushable[i], 3) for i in range(len(pre))])
        min_off = float(self.num("push_min_contact_m", 0.10))
        idx_of = {id(t): i for i, t in enumerate(pre)}
        push_rank = (lambda t: 0 if pushable.get(idx_of[id(t)], 1.0) >= min_off else 1) if pushable else (lambda t: 0)  # noqa: E731

        def _ahead(m: Stance) -> float:
            f = m.focus[:2] if m.focus is not None else m.base_xy
            return float((th.as_tensor(f, dtype=th.float32) - m.base_xy) @ m.forward)

        def _margin(t) -> float:
            m = t[1]
            f = m.focus if m.focus is not None else th.tensor([float(m.base_xy[0]), float(m.base_xy[1]), z_init])
            # owner 2026-09-27 (first-demo v2.2): by whichever arm whose band holds the target at this B reaches better
            arms = [a for a in (m.extras.get("focus_arms") or [offered[0]]) if a in t[3]] or [offered[0]]
            return self.best_reach_margin(m, [float(f[0]), float(f[1]), z_init + 0.5 * float(self.place["height"])], arms)[0]
        order_mode = str(self.gp.gen("b_trunk_order", self.gp.gen("b_prescreen_order", "reach")))
        if order_mode == "reach":
            # W2 (owner 2026-09-20 22:15): the B whose trunk leaves the most reach margin to the target first (5 cm buckets, capped at
            # 0.30 m so a nearly-folded arm does not outrank a comfortable one), then a usable F_init, the compact grasp, the larger F_init;
            # 1 cm buckets with generation.grasp_reach_margin_m (owner 2026-10-13 ~03:15 item 1: the slightly closer B first)
            bucket = 0.01 if self.grasp_reach_margin() is not None else 0.05
            margins = {id(t): min(round(_margin(t) / bucket) * bucket, 0.30) for t in pre}
            pre.sort(key=lambda t: (push_rank(t), -margins[id(t)], 0 if t[0] >= min_f else 1, round(_ahead(t[1]), 1), -t[0]))
            log.info("%s: pre-screen reach margins per B (trunk -> margin): %s", self.gp.variant,
                     [(round(float(t[1].translate), 2), round(float(t[1].tilt), 2), round(_margin(t), 3)) for t in pre])
        elif order_mode == "compact":
            pre.sort(key=lambda t: (push_rank(t), 0 if t[0] >= min_f else 1, round(_ahead(t[1]), 1), -t[0]))
        else:
            pre.sort(key=lambda t: (push_rank(t), -t[0]))
        if ov_opts is not None and ov_opts["rank_b"] and kind != "floor":
            # owner 2026-10-13 ~03:15 item 4: the B candidates whose object a short push brings into an overhang, the push's far side within the
            # arm's reach, first (stable: the order above within each group) - the direct grasp's B often has the object deep on the support
            ranks = {id(t): self._overhang_b_rank(t, z_init, ov_opts) for t in pre}
            pre.sort(key=lambda t: ranks[id(t)][:2])
            log.info("%s: overhang pick ranks per B (trunk, stand-off -> rank, reach margin at the push start, push): %s", self.gp.variant,
                     [(round(float(t[1].translate), 2), round(float(t[1].tilt), 2), t[1].stand_off, ranks[id(t)][0], round(ranks[id(t)][1], 3), ranks[id(t)][2]) for t in pre])
        self.template_facts.setdefault("F_init_prescreen", []).append([t[0] for t in pre])
        if len(offered) > 1:
            self.template_facts.setdefault("F_init_prescreen_arms", []).append([{a: len(F) for a, (_, F) in t[3].items()} for t in pre])
        log.info("%s: F_init pre-screen over %d B candidates (%d yaws): %s%s", self.gp.variant, len(pre), len(yaws), [t[0] for t in pre],
                 "" if len(offered) < 2 else f" (per arm {[{a: len(F) for a, (_, F) in t[3].items()} for t in pre]})")
        n_couple = int(self.num("b_candidates_max", 6))
        for i_b, (n_f, manip, F_init, by_arm) in enumerate(pre[:n_couple]):
            self._check_deadline(f"B candidate {i_b + 1} of {min(n_couple, len(pre))}")
            self.sidecar.transitions.clear()
            self.sidecar.paths.clear()
            try:
                if not F_init:
                    raise Reject("grasp_unreachable", "empty feasible set F_init at B's focus cell")
                # A -> B (§3.7 / §3.8; owner 2026-09-19 21:56 (ii)); floor class: the floor mesh out of the sweep / drive-in checks
                # like stance B itself
                with self._floor_relaxed():
                    leg_ab = self.record_leg_a_b(init, manip)
                focus_xy = self.surface["c_t"] if kind == "floor" else manip.focus[:2]
                d, leg_ab = self._couple_segment(rng, init, manip, by_arm, focus_xy, z_init, leg_ab)
                fallback = getattr(self, "init_fallback", None)
                d.summary.update({"relation": self.relation, "init_surface": {"class": kind, "name": None if kind == "floor" else self.surface["obj"].name,
                                                                              **({"fallback": fallback} if fallback is not None else {})},
                                  "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(manip),
                                  "legs": {"A_B": leg_ab, "B_C": getattr(self, "_leg_bc", None)}})
                return d
            except Reject as e:
                if e.reason == "time_budget":
                    raise  # the attempt is over, not this B candidate
                errors.append(f"B candidate: {e.reason}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.stances.pop("placement", None)
                self.sidecar.stances.pop("push", None)
                self.sidecar.stances.pop(OH.PICK_STANCE, None)
                self.sidecar.contacts.pop(OH.CONTACT_SLOT, None)
                self.sidecar.grasps.clear()
                self.carry_arm_q.clear()
                self.carry_end_q.clear()
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def _b_stances(self, manip: Stance, offered: Sequence[str]) -> Dict[str, Stance]:
        """Stance B per offered arm (owner 2026-09-27, first-demo v2.2): the default arm's - ``manip`` itself when the B search solved it
        for that arm - always, the other arm's where its band holds the target (``extras["focus_arms"]``); each with the other arm idle
        in the posture re-derived for it (``tabletop.stance_for_arm``: A6 untucked / ready / tucked; an arm whose idle partner collides
        in every posture is not offered)."""
        arm0 = offered[0]
        out: Dict[str, Stance] = {}
        if manip.working_arms in ([], [arm0]):
            manip.arms = [arm0]
            out[arm0] = manip
        else:
            st = TT.stance_for_arm(self, manip, arm0)
            if st is not None:
                out[arm0] = st
        focus_arms = manip.extras.get("focus_arms") or [arm0]
        for a in offered[1:]:
            if a in focus_arms:
                st = TT.stance_for_arm(self, manip, a)
                if st is not None:
                    out[a] = st
        return out

    @staticmethod
    def _segment_order(manip: Stance, by_arm: Dict[str, Tuple[Stance, List[Any]]]) -> List[str]:
        """The arms with pairs at B in the order their couplings run: those whose band holds the target (``focus_arms``) first, the
        default arm first among them (a B shortlisted by the other arm's band alone starts with that arm)."""
        arms = [a for a, (_, F) in by_arm.items() if F]
        focus = manip.extras.get("focus_arms") or arms[:1]
        return [a for a in arms if a in focus] + [a for a in arms if a not in focus]

    def _same_arm_q(self, a: Stance, b: Stance) -> bool:
        """Both arms at the same joint values in the two stances (a leg checked at one holds at the other)."""
        ra, la, _ = self.ikc.arm_vectors(a)
        rb, lb, _ = self.ikc.arm_vectors(b)
        return bool(th.equal(ra, rb) and th.equal(la, lb))

    def _clear_segment_records(self, manip_name: str) -> None:
        """What a failed coupling recorded past B: the C / push stances, the grasp, the push contact, the carry configurations and
        every path / transition but the A -> B leg."""
        self.sidecar.stances.pop("placement", None)
        self.sidecar.stances.pop("push", None)
        self.sidecar.stances.pop(OH.PICK_STANCE, None)
        self.sidecar.contacts.pop("push", None)
        self.sidecar.contacts.pop(OH.CONTACT_SLOT, None)
        self.sidecar.grasps.clear()
        self.carry_arm_q.clear()
        self.carry_end_q.clear()
        self.sidecar.paths = [p for p in self.sidecar.paths if p.to == manip_name]
        self.sidecar.transitions = [t for t in self.sidecar.transitions if t.to == manip_name]

    def _couple_segment(self, rng, init: Stance, manip: Stance, by_arm: Dict[str, Tuple[Stance, List[Any]]], focus_xy, z_init: float,
                        leg_ab: Dict[str, Any]) -> Tuple[Draft, Dict[str, Any]]:
        """The grasp -> carry -> place -> release segment from stance B (owner 2026-09-27, first-demo v2.2): one coupling (:meth:`_couple`)
        per arm with pairs in its F_init (:meth:`_segment_order`). ``manip`` (the record; its A -> B leg recorded) takes the coupling
        arm's working arm and idle posture, the leg recorded again when that changes the arms' joint values. A pair whose grasp at B
        another arm won waits for that arm's coupling; the pairs a failed coupling won are regrasped by the next arm (the segment
        fallback). Returns the draft and the A -> B leg facts."""
        arms = self._segment_order(manip, by_arm)
        # ``manip`` takes each coupling's arm in place: the segment keeps a copy of the stance it was for its own arm
        seg = SegmentArms({a: st.with_trunk(st.translate, st.tilt, st.name) if st is manip else st for a, (st, _) in by_arm.items()},
                          {a: list(F) for a, (_, F) in by_arm.items()}, {a: (self.surface.get("arm_bands") or {}).get(a) for a in by_arm},
                          2.0 * math.pi / max(1, len(self.yaws())))
        errors: List[str] = []
        last: Optional[Reject] = None
        for k, arm in enumerate(arms):
            st_a = by_arm[arm][0]
            try:
                if k > 0:
                    self._clear_segment_records(manip.name)
                if manip.working_arms != [arm] or manip.idle_posture != st_a.idle_posture:
                    same = self._same_arm_q(manip, st_a)
                    TT.adopt_arm(manip, st_a)
                    if not same:
                        self._drop_leg_records(manip.name)
                        seg.leg += 1
                        with self._floor_relaxed():
                            leg_ab = self.record_leg_a_b(init, manip)
                self.seg_arm = arm
                c_pool = self.placement_candidates(focus_xy)
                if not c_pool:
                    raise Reject("no_placement_stance", "no stance C survivor for this init pose")
                return self._couple(rng, init, manip, c_pool, by_arm[arm][1], focus_xy, z_init, seg), leg_ab
            except Reject as e:
                if e.reason == "time_budget" or len(arms) == 1:
                    raise
                seg.done.append(arm)
                seg.why.setdefault(arm, f"{e.reason}: {str(e)[:300]}")
                errors.append(f"{arm} arm: {e.reason}: {e}")
                last = e
                log.info("%s: B (%.2f, %.2f): the %s arm's segment failed (%s)%s", self.gp.variant, manip.translate, manip.tilt, arm, e.reason,
                         f"; the {arms[k + 1]} arm's coupling next" if k + 1 < len(arms) else "")
        raise Reject(last.reason if last is not None else "grasp_unreachable", " | ".join(errors) or "no arm has pairs in F_init at B")

    @staticmethod
    def _b_options(pick, g, eef: G.Pose, pre: G.Pose) -> Tuple[List[Tuple[Any, G.Pose, G.Pose, bool]], Optional[Dict[str, Any]]]:
        """The finger assignments a grasp at B may be recorded with, the pick's first (``(grasp row or twin, eef, pre-grasp, is twin)``),
        and its ``twin_choice``."""
        if pick is None:
            return [(g, eef, pre, False)], None
        opts = [(pick.grasp, pick.eef, pick.pre, pick.twin)]
        if pick.other is not None:
            opts.append((pick.other[0], pick.other[2], pick.other[3], not pick.twin))
        return opts, pick.facts

    def grasp_reach_margin(self) -> Optional[float]:
        """``generation.grasp_reach_margin_m`` (m; default None = off, owner 2026-10-13 ~03:15 item 1): the reach margin a grasp at B must
        keep along its approach (:meth:`grasp_margin_fails`)."""
        v = self.gp.gen("grasp_reach_margin_m")
        return float(v) if v is not None and float(v) > 0.0 else None

    def grasp_margin_fails(self, stance: Stance, arm: str, eef: G.Pose, pre: G.Pose) -> Optional[str]:
        """Why the grasp ``[pre, eef]`` from ``stance`` lacks the reach margin (None: it keeps it, or no margin is asked): the same chain with
        the eef :meth:`grasp_reach_margin` deeper along its approach axis must pass the exact IK too, with the B solve's ignore set. The
        sampler's exact IK admits a grasp at the edge of the reach that the program check's tracked approach to the settled target then
        misses (under#02 pitched_c2_k0, 2 of 2 attempts past the goal check: 'differential IK did not converge at waypoint 2/4: pos_err=7.0mm
        ...; the end pose has no solution on its own either'). The IK memory is left as it was."""
        m = self.grasp_reach_margin()
        if m is None:
            return None
        a = C.approach_axis(G.t(eef[1])).float()
        deep = [(G.t(pre[0]).float() + m * a, G.t(pre[1])), (G.t(eef[0]).float() + m * a, G.t(eef[1]))]
        saved = dict(self.ikc.solutions)
        try:
            res = self.ikc.exact(deep, stance, arm, ignore_for_last=self.grasp_ignore_objects(), scale_key=self.choice.models["target"].scale, cache=False)
        finally:
            self.ikc.solutions.clear()
            self.ikc.solutions.update(saved)
        TT._count(self, "grasp_reach_margin.checked")
        if res.ok:
            return None
        TT._count(self, "grasp_reach_margin.rejected")
        return str(res.reason)

    def _b_frame_options(self, opts_facts, stance: Stance, where: str) -> Tuple[Optional[Tuple[List[Tuple[Any, G.Pose, G.Pose, bool]], Optional[Dict[str, Any]]]], str]:
        """Opt-in ``generation.b_fingertip_frame_check`` (oracle-fix 2026-11-05, speed): :meth:`_b_options`' finger assignments whose pre-grasp
        and grasp keep the fingertip point inside the image of ``stance`` by the program check's own gate (``program_check.fingertips_in_frame``,
        5 % + its slack), in their order; ``(None, why)`` when none does. brief15 (under_large#02): 60 of 123 rejected attempts failed that gate at
        ``grasp.pre_grasp`` in the program check, 144 s each after the sample / settle / goal stages. Unchanged without the key."""
        if not self.gp.gen("b_fingertip_frame_check"):
            return opts_facts, ""
        from ispatialgym.sampling.v2 import program_check as PCK

        opts, facts = opts_facts
        kept, why = [], []
        for o in opts:
            out = PCK.fingertips_in_frame(self, stance, "grasp", [("pre_grasp", [o[2]]), ("approach", [o[1]])])
            if out is None:
                kept.append(o)
            else:
                why.append(f"{getattr(o[0], 'name', o[0])}: {out['key']} fingertip outside the image (5 %): u {out['u']}, v {out['v']}, depth {out['depth_m']} m")
                TT._count(self, "b_frame.rejected")
        if not kept:
            return None, f"[{where}] " + "; ".join(why)
        if facts is not None and kept[0] is not opts[0]:
            facts = dict(facts, chosen="twin" if kept[0][3] else "row", frame_flip=f"the {'twin' if opts[0][3] else 'row'} fails the fingertip frame gate: {why[0]}")
        return (kept, facts), ""

    def _b_margin_options(self, opts_facts, stance: Stance, arm: str, where: str) -> Tuple[Optional[Tuple[List[Tuple[Any, G.Pose, G.Pose, bool]], Optional[Dict[str, Any]]]], str]:
        """:meth:`_b_options`' finger assignments that keep the grasp reach margin (:meth:`grasp_margin_fails`), in their order, and the
        ``twin_choice`` (``chosen`` follows the first one kept; ``reach_margin_flip`` names the dropped first one); ``(None, why)`` when none
        does. Unchanged without ``generation.grasp_reach_margin_m``. One INFO line 'grasp reach margin [..]: ...' per grasp checked.
        First :meth:`_b_frame_options` (opt-in ``generation.b_fingertip_frame_check``)."""
        opts_facts, why_f = self._b_frame_options(opts_facts, stance, where)
        if opts_facts is None:
            return None, why_f
        m = self.grasp_reach_margin()
        if m is None:
            return opts_facts, ""
        opts, facts = opts_facts
        kept, why = [], []
        for o in opts:
            w = self.grasp_margin_fails(stance, arm, o[1], o[2])
            if w is None:
                kept.append(o)
            else:
                why.append(f"{getattr(o[0], 'name', o[0])}: {w[:160]}")
        log.info("%s: grasp reach margin [%s] %.3f m deeper along the approach (%s arm, stance (%.2f, %.2f)): %s kept%s", self.gp.variant, where, m, arm,
                 stance.translate, stance.tilt, [getattr(o[0], "name", o[0]) for o in kept] or "none", f"; {'; '.join(why)}" if why else "")
        if not kept:
            return None, "; ".join(why)
        if facts is not None and kept[0] is not opts[0]:
            facts = dict(facts, chosen="twin" if kept[0][3] else "row", reach_margin_flip=f"the {'twin' if opts[0][3] else 'row'} fails the reach margin: {why[0]}")
        return (kept, facts), "; ".join(why)

    def _regrasp_at_b(self, g, t_pose: G.Pose, stance: Stance, arm: str, z_init: float, twins: bool, bands: Dict[str, Any]):
        """The segment fallback's grasp at B (owner 2026-09-27, first-demo v2.2; ``tabletop.segment_fallback_grasp``'s rule, keeping the
        other finger assignment for the place at C): the library row ``g`` at the target pose the first grasp was drawn at, with ``arm``
        only (both turns by the orientation rule when the twins are on); :meth:`_b_options` or None."""
        TT._count(self, "segment_fallbacks")
        eef = IK.grasp_pose(t_pose, g.pose)
        pre = IK.pre_pose(eef)
        kw = {"ignore_for_last": self.grasp_ignore_objects(), "scale_key": self.choice.models["target"].scale, "cache": False}
        res = self.ikc.exact([pre, eef], stance, arm, **kw)
        pick = None
        if twins and getattr(g, "name", None):
            pick = TT.pick_twin(self, stance, arm, g, g.pose, eef, pre, res, kw, support_z=float(z_init), where="B, segment fallback", arms=[arm], stances={arm: stance})
            res = pick.res
            self._count_mirrored_ik(pick, bands)
        opts = None
        if res.ok:
            opts, why_m = self._b_margin_options(self._b_options(pick, g, eef, pre), stance, arm, "B, segment fallback")
            if opts is None:
                res = IK.IKResult(False, None, f"grasp reach margin: {why_m}", 2)
        if not res.ok:
            log.info("%s: segment fallback: the %s arm's regrasp of %s at B fails (%s)", self.gp.variant, arm, getattr(g, "name", g), str(res.reason)[:200])
            return None
        TT._count(self, f"segment_fallback_taken.{arm}")
        return opts

    def _count_mirrored_ik(self, pick, bands: Dict[str, Any]) -> None:
        """``arm_band_mirrored.ik_tried`` / ``.ik_failed`` for a grasp's solved options of an arm whose band is a mirror (as
        ``tabletop.phase3_target`` counts them)."""
        for o in pick.facts["orient"]["options"]:
            b = bands.get(o.get("arm"))
            if o.get("source") == "ik" and b is not None and b.meta.get("mirrored_from"):
                TT._count(self, "arm_band_mirrored.ik_tried")
                if not o.get("ok"):
                    TT._count(self, "arm_band_mirrored.ik_failed")

    #: oracle-fix 2026-11-05 (opt-in ``generation.carry_via_search``): the carry waypoints tried at C when the straight carry leg does not
    #: track - ``(where, raise m, lateral m, orientation fraction toward the pre-place's)``: ``start`` = at the carry pose (rotate first),
    #: ``mid`` = the leg's midpoint, ``end`` = at the pre-place (translate first); lateral is horizontal, across the leg
    CARRY_VIAS_AT_C = (("start", 0.0, 0.0, 1.0), ("start", 0.05, 0.0, 1.0), ("start", 0.10, 0.0, 1.0),
                       ("mid", 0.05, 0.0, 1.0), ("mid", 0.10, 0.0, 1.0), ("mid", 0.05, 0.05, 1.0), ("mid", 0.05, -0.05, 1.0),
                       ("mid", 0.0, 0.0, 0.5), ("mid", 0.05, 0.0, 0.5), ("mid", 0.10, 0.0, 0.5),
                       ("end", 0.05, 0.0, 0.0), ("end", 0.10, 0.0, 0.0))

    def _carry_vias_at_c(self, h0: G.Pose, pre_place: G.Pose) -> List[Tuple[str, G.Pose]]:
        """The candidate carry waypoints (:data:`CARRY_VIAS_AT_C`) between the hand at C ``h0`` and the pre-place, labelled."""
        p0, q0 = G.t(h0[0]).float(), G.t(h0[1]).float()
        p1, q1 = G.t(pre_place[0]).float(), G.t(pre_place[1]).float()
        d = p1 - p0
        across = th.tensor([-float(d[1]), float(d[0]), 0.0])
        across = across / max(float(th.linalg.norm(across)), 1e-6)
        out: List[Tuple[str, G.Pose]] = []
        for where, raise_m, lat, frac in self.CARRY_VIAS_AT_C:
            base = {"start": p0, "mid": 0.5 * (p0 + p1), "end": p1}[where]
            pos = base + th.tensor([0.0, 0.0, float(raise_m)]) + across * float(lat)
            out.append((f"{where} raise {raise_m:.2f} lateral {lat:+.2f} turn {frac:.1f}", (pos, C.slerp(q0, q1, float(frac)).float())))
        return out

    @staticmethod
    def _carry_leg_poses(chain: Sequence[G.Pose]) -> List[G.Pose]:
        """The carry leg's poses through ``chain`` as ``Place._carry_leg`` builds them (a pose every ``CARRY_WAYPOINT_SPACING``)."""
        from ispatialgym.oracle import actions as A

        out: List[G.Pose] = []
        for a, b in zip(chain[:-1], chain[1:]):
            n = max(A.N_WAYPOINTS, int(math.ceil(float(th.linalg.norm(G.t(b[0]) - G.t(a[0]))) / A.CARRY_WAYPOINT_SPACING)))
            out += [tuple(w) for w in C.line_waypoints(a, b, n)]
        return out

    def _carry_track_at_c(self, place: Stance, arm: str, tname: str, target, g, pre_place: G.Pose, place_eef: G.Pose,
                          ignore_place) -> Tuple[Optional[th.Tensor], Optional[float], Optional[str], str, Optional[Dict[str, Any]]]:
        """The oracle's carry leg at C - a straight Cartesian line from where the hand is (``carry_arm_q`` at C's trunk) to the pre-place,
        tracked by differential IK (:meth:`_carry_track_kw`) - then the pre-place / place IK seeded at its end, which must stay within
        ``carry_finish_max_rad`` of it. ``(q_end, swing, reason, kind, via)``: kind ``"ok"`` (reason None), ``"untrackable"`` (no q_end /
        no drop IK from it) or ``"branch"`` (the witness swings more than the bound). With ``generation.carry_via_search`` a straight leg
        that fails is retried through each carry waypoint of :meth:`_carry_vias_at_c` (every carry pose's fingertip in C's image by the
        program check's gate, the whole chain tracked); ``via`` = ``{"label", "pose"}`` of the one that passed (recorded as
        ``grasps[k].carry``, the oracle carries through it), else None and the straight leg's failure."""
        from ispatialgym.oracle import r1pro_model as M
        from ispatialgym.sampling.v2 import program_check as PCK

        idx_r = self.kit.robot.arm_control_idx[arm]
        q_c = self.ikc.q_locked(place).clone()
        q_c[idx_r] = self.carry_arm_q[tname]
        h0 = self.kit.fk([M.EEF_LINK[arm]], q_c)[M.EEF_LINK[arm]]
        vias: List[Tuple[Optional[str], Optional[G.Pose]]] = [(None, None)]
        if bool(self.gp.gen("carry_via_search") or False):
            vias += self._carry_vias_at_c(h0, pre_place)
        straight = None
        n_frame, n_fail = 0, 0
        with attached(self.kit, arm, target, eef_in_object=g.pose), self._floor_relaxed():
            for label, via in vias:
                chain = [h0] + ([via] if via is not None else []) + [pre_place]
                if via is not None and PCK.fingertips_in_frame(self, place, "place", [("carry", self._carry_leg_poses(chain))]) is not None:
                    n_frame += 1
                    continue
                q_end, why_t = self.ikc.track_chain(q_c, chain, arm, ignore=[self.obj("floor")], soft_stall_m=float(self.num("carry_soft_stall_m", 0.02)),
                                                    **self._carry_track_kw())
                r2b = None
                if q_end is not None:
                    r2b = self.ikc.exact([pre_place, place_eef], place, arm, ignore_for_last=ignore_place, cache=False, q_seed=q_end)
                if q_end is None or r2b is None or not r2b.ok or r2b.q is None:
                    res = (None, None, (why_t if q_end is None else (r2b.reason if r2b is not None else "no IK")), "untrackable")
                else:
                    swing = float(th.max(th.abs(r2b.q[idx_r] - q_end[idx_r])))
                    if swing > float(self.num("carry_finish_max_rad", 0.5)):
                        res = (q_end, swing, f"pre-place witness {swing:.2f} rad from the tracked carry end", "branch")
                    else:
                        res = (q_end, swing, None, "ok")
                if res[3] == "ok":
                    if via is not None:
                        log.info("%s: carry leg at C through the carry waypoint %s (the straight leg: %s; %d waypoints out of frame, %d untrackable before it)",
                                 self.gp.variant, label, str(straight[2])[:300] if straight else "-", n_frame, n_fail)
                    return res + (None if via is None else {"label": label, "pose": via},)
                if via is None:
                    straight = res
                else:
                    n_fail += 1
        if len(vias) > 1:
            return straight[:2] + (f"{straight[2]}; none of {len(vias) - 1} carry waypoints passes ({n_frame} out of frame, {n_fail} untrackable)", straight[3], None)
        return straight + (None,)

    def _carry_track_kw(self) -> Dict[str, Any]:
        """oracle-fix 2026-11-05 (opt-in ``generation.carry_track_oracle``; hardback hb16_desk_s1 attempt 3): the carry leg at C tracked with
        the program check's continuation acceptance (``IKChecker.continue_chain``) - the oracle's soft waypoint tolerances and iteration
        budget, a stall finished only within ``FINISH_MAX_EXCURSION`` of where it stalled. Without the bound the carry that stalled at dense
        pose 9/45 (11.1 mm) was finished 1.12 rad away and the program check rejected the place after 1468 s (block design's re-posed
        carries take the same bound). ``{}`` when off: the tracking as before."""
        kw: Dict[str, Any] = {}
        if bool(self.gp.gen("carry_track_oracle") or False):
            kw = {"finish_max_excursion": self.ikc.FINISH_MAX_EXCURSION, "soft_pos_tol": self.ikc.ORACLE_SOFT_POS_TOL,
                  "soft_rot_tol": self.ikc.ORACLE_SOFT_ROT_TOL, "max_iters": self.ikc.ORACLE_DIFF_IK_MAX_ITERS}
        rem = self.gp.gen("carry_finish_max_remaining_m")
        if rem is not None:
            # oracle-fix 2026-11-05 (hbD7: a carry stall 0.38 m from the pre-place passed here, the oracle's planned finish failed)
            kw["finish_max_remaining_m"] = float(rem)
        return kw

    # ------------------------------------------------------------------ the overhang pick (owner 2026-10-13 ~03:15 item 4)

    def pick_overhang_opts(self) -> Optional[Dict[str, Any]]:
        """``generation.pick_overhang_push`` (:func:`overhang.options`; None = off)."""
        try:
            return OH.options(self.gp.gen("pick_overhang_push"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")

    def _overhang_edges(self, centre_xy, d, laterals: Sequence[float], z_top: float) -> List[Optional[float]]:
        """The support edge's distance along ``d`` from ``centre_xy`` on each lateral line (:func:`overhang.edge_from_hits`): vertical rays
        every ``EDGE_PROBE_STEP_M`` that meet one of the support objects within 1 cm of its top (the robot and the target out of the
        rays); the support-top raster's cells when the raycast is unavailable."""
        u, lat = OH.unit_xy(d), OH.lateral_unit(d)
        n = int(round(OH.EDGE_PROBE_MAX_M / OH.EDGE_PROBE_STEP_M)) + 1
        c = G.t(centre_xy).float()[:2]
        rows = [[c + lat * float(l) + u * (k * OH.EDGE_PROBE_STEP_M) for k in range(n)] for l in laterals]
        try:
            from ispatialgym.sampling.sim_utils import raytest_batch

            prims = {l.prim_path for o in self.support_objects() for l in getattr(o, "links", {}).values()}
            ignore = [l.prim_path for o in (self.robot, self.target) for l in getattr(o, "links", {}).values()]
            raw = raytest_batch([[float(p[0]), float(p[1]), z_top + 0.03] for row in rows for p in row],
                                [[float(p[0]), float(p[1]), z_top - 0.03] for row in rows for p in row], only_closest=True, ignore_bodies=ignore or None)
            flat = []
            for h in raw:
                body = str(h.get("rigidBody")) if h.get("hit") else ""
                flat.append(bool(body) and (body in prims or any(body.startswith(pp + "/") for pp in prims)) and abs(float(h["position"][2]) - z_top) <= 0.01)
        except Exception as e:  # noqa: BLE001 - no raycast here: the 2 cm raster
            log.info("%s: overhang edge probe by raycast unavailable (%r): the support-top raster instead", self.gp.variant, e)
            top = self.surface.get("top")
            flat = [bool(top is not None and top.contains(p)) for row in rows for p in row]
        return [OH.edge_from_hits(flat[i * n:(i + 1) * n]) for i in range(len(rows))]

    def _overhang_geometry(self, t_pose: G.Pose, g, d, z_top: float, opts: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """The push the grasp ``g`` of the object at ``t_pose`` needs (module docstring of :mod:`overhang`): the fingers' sweep below the
        support top (:func:`overhang.sweep_min_along`, from the open gripper to the pinch width), the edge on the fingers' lines and on
        the centre of mass's (:meth:`_overhang_edges`), the push-distance ladder. ``(facts, "")`` or ``(None, why)``."""
        u = OH.unit_xy(d)
        lat = OH.lateral_unit(d)
        eef0 = IK.grasp_pose(t_pose, g.pose)
        pinch = TT.grasp_pinch_width(g)
        sweep = OH.sweep_min_along(eef0, u, z_top, C.GRIPPER_OPEN_M, float(pinch or 0.0))
        if sweep is None:
            return None, "no finger of the grasp reaches below the support top (nothing to push for)"
        c = G.t(t_pose[0]).float()[:2]
        com = G.local_points_world(self.target_com_local().reshape(1, 3), (G.t(t_pose[0]).float(), G.t(t_pose[1]).float()))[0][:2]
        side_pts = OH.finger_sweep_points(eef0, C.GRIPPER_OPEN_M, float(pinch or 0.0))[sweep[1]]
        below = side_pts[side_pts[:, 2] <= z_top]
        lats = ((below if len(below) else side_pts)[:, :2] - c) @ lat
        l_lo, l_hi = float(lats.min()), float(lats.max())
        com_lat = float((com - c) @ lat)
        edges = self._overhang_edges(c, u, [com_lat, l_lo, 0.5 * (l_lo + l_hi), l_hi], z_top)
        com_ahead = float((com - c) @ u)
        e_com = None if edges[0] is None else edges[0] - com_ahead  # the centre of mass's distance to the edge on its own line
        fing = [e for e in edges[1:] if e is not None]
        if e_com is None or not fing:
            return None, f"no support edge found along the push within {OH.EDGE_PROBE_MAX_M:.1f} m (edges {edges})"
        e_finger = max(fing)
        x_sweep = float(sweep[0]) - float(c @ u)
        d_min, d_max = OH.push_distance_range(e_finger, e_com, x_sweep, opts["finger_clearance_m"], opts["com_margin_m"], opts["max_push_m"])
        ladder = OH.distance_ladder(d_min, d_max, opts["distance_step_m"], opts["max_distances"])
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(t_pose[1]).float()))
        back, front = OH.extent_along(pts[:, :2], u)
        v2 = getattr(self, "v2_overhang", None)
        v2_push = None
        v2_sweep_beyond = None
        if v2 is not None:
            # i7-overhang 2026-11-07: the push that leaves the object at the library's overhang rest (the face measured_distance_m beyond the edge
            # on the centre's line) - one rung. The library validated the grasp at exactly that rest (the exact finger hulls against the slab
            # box within the policy's 2 mm, the physics trial with the edge there), so the finger-box sweep rule (finger_clearance_m on the
            # conservative open-width box) is replaced by it unless v2_finger_clearance_m asks for it; the centre-of-mass margin and max_push_m
            # stay (d_max), every exact check after it too
            v2_push = OH.v2_rest_push(float(v2.get("measured_distance_m") or v2["distance_m"]), float(edges[0]), front)
            v2_sweep_beyond = float(v2_push) + x_sweep - e_finger  # the box sweep's distance beyond the edge after the push (< 0: over the support)
            fc = opts.get("v2_finger_clearance_m")
            lo = max(0.01, (e_finger + float(fc) - x_sweep) if fc is not None else 0.01)
            ladder = [round(v2_push, 4)] if lo - 1e-4 <= v2_push <= d_max + 1e-4 else []
        facts = {"direction": [round(float(u[0]), 4), round(float(u[1]), 4)], "edge_com_m": round(e_com, 4), "com_ahead_m": round(com_ahead, 4), "edge_finger_m": round(e_finger, 4),
                 "edge_lines_m": [None if e is None else round(e, 4) for e in edges], "finger_lateral_m": [round(l_lo, 4), round(l_hi, 4)],
                 "sweep_along_m": round(x_sweep, 4), "sweep_side": int(sweep[1]), "pinch_width_m": None if pinch is None else round(float(pinch), 4),
                 "push_min_m": round(d_min, 4), "push_max_m": round(d_max, 4), "ladder_m": ladder, "extent_front_m": round(front, 4), "extent_back_m": round(back, 4),
                 "com_xy": [round(float(v), 4) for v in com.tolist()]}
        if v2 is not None:
            facts["v2_rest"] = {"tag": v2.get("tag"), "distance_m": v2.get("measured_distance_m"), "push_m": round(float(v2_push), 4),
                                "box_sweep_beyond_edge_m": round(float(v2_sweep_beyond), 4), "finger_rule": "library rest" if opts.get("v2_finger_clearance_m") is None
                                else f"box sweep {float(opts['v2_finger_clearance_m']):.3f} m beyond the edge"}
            if not ladder:
                return None, (f"the v2 overhang rest's push {v2_push:.3f} m (face {v2.get('measured_distance_m')} m beyond the edge) is outside the range "
                              f"[{lo:.3f}, {d_max:.3f}] m (centre of mass {opts['com_margin_m']:.3f} inside, push <= {opts['max_push_m']:.2f}"
                              + (f", the finger box {float(opts['v2_finger_clearance_m']):.3f} beyond the edge" if opts.get("v2_finger_clearance_m") is not None else "") + ")")
        if not ladder:
            return None, (f"no push distance: the fingers need >= {d_min:.3f} m (edge {e_finger:.3f} m on their lines + clearance {opts['finger_clearance_m']:.3f} - sweep "
                          f"{x_sweep:.3f}), the centre of mass allows <= {d_max:.3f} m (edge {e_com:.3f} - margin {opts['com_margin_m']:.3f})")
        return facts, ""

    def _overhang_pairs(self, manip: Stance, F_direct: Sequence[Tuple[float, Any]], xy, yaws: Sequence[float], z_init: float,
                        opts: Dict[str, Any], arm: Optional[str] = None, st: Optional[Stance] = None) -> List[Tuple[float, Any]]:
        """F_init at B with the overhang pick on: the direct grasp's pairs (mode ``"only"``: none of them) and every (yaw, library row)
        whose overhang geometry has a push ladder (:meth:`_overhang_geometry`, no IK) - the pinch on the side the push brings past the
        edge, which the direct screen rarely admits (grasp_5 at B's direct yaws pinches the plate's side: no push distance). The geometry
        is memoised per (focus, yaw, row) for the attempt (the arms share it). i7-overhang 2026-11-07: at the v2 overhang rest the one yaw
        with the overhanging face flush with B's edge (``overhang.flush_yaw``) and only the rows whose grasp the overhang map holds feasible
        at the post-push cell (:meth:`_v2_overhang_map_ok`, ``arm`` at its stance ``st``)."""
        if manip.edge is None or manip.focus is None:
            return list(F_direct)
        u = OH.unit_xy(-manip.edge.inward)
        v2 = getattr(self, "v2_overhang", None)
        if v2 is not None:
            yaws = [OH.flush_yaw(v2["normal_xy"], u)]
        step = 2.0 * math.pi / max(1, len(yaws))
        only = opts["mode"] == "only"
        have = {pair_key(y, g, step) for y, g in F_direct}
        memo = self.__dict__.setdefault("_overhang_pair_memo", {})
        extra = []
        for y in yaws:
            for g in self.grasps:
                if not only and pair_key(y, g, step) in have:
                    continue
                key = (round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(y), 4), getattr(g, "name", str(g)), round(float(u[0]), 4), round(float(u[1]), 4))
                if key not in memo:
                    memo[key] = self._overhang_geometry(self.pose_on(xy, y, z_init), g, u, float(z_init), opts)[0] is not None
                if memo[key] and v2 is not None:
                    ok_m, why_m = self._v2_overhang_map_ok(st or manip, arm or TT.default_arm(self), g, self.pose_on(xy, y, z_init), u, float(z_init), opts)
                    TT._count(self, f"pick_overhang.v2_map.{'hit' if ok_m else ('infeasible' if ok_m is False else 'miss')}")
                    if not ok_m:
                        log.info("%s: overhang pick pair %s yaw %.3f at B (%.2f, %.2f), %s arm: not in the overhang map (%s)", self.gp.variant, getattr(g, "name", g), y,
                                 manip.translate, manip.tilt, arm, why_m)
                        continue
                if memo[key]:
                    extra.append((float(y), g))
        if extra:
            TT._count(self, "pick_overhang.pairs_added")
        log.info("%s: overhang pick pairs at B (%.2f, %.2f) d %s: %d added to F_init's %d (yaws %s)%s", self.gp.variant, manip.translate, manip.tilt, manip.stand_off, len(extra),
                 len(F_direct), sorted({round(y, 2) for y, _ in extra}), " - mode only: the direct pairs dropped" if only else "")
        return ([] if only else list(F_direct)) + extra

    def _v2_overhang_map_ok(self, st: Stance, arm: str, g, t_pose: G.Pose, u, z_init: float, opts: Dict[str, Any]) -> Tuple[Optional[bool], str]:
        """i7-overhang 2026-11-07 (the map-hit step of the v2 overhang pick): whether ``arm``'s OVERHANG map (the band view of
        ``surface["arm_bands"]``) holds the library row ``g`` (or its twin / a family member of it) feasible at the post-push cell - the
        object at ``t_pose`` pushed to the library rest (:meth:`_overhang_geometry`'s one rung) - from ``st`` at its trunk, or from ``st``
        backed off ``grasp_back_offs_m`` (the map's edge distance implied by the post-push root: its base-frame x minus the rest's
        ``edge_along_m``, nearest map distance within 3 cm; the cell nearest in y within half the lattice). ``(True, "")``, ``(False, why)``
        (the map's verdict), ``(None, why)`` (no overhang map view for the arm / the cell outside the map)."""
        from ispatialgym.sampling import reach_maps as RMAP

        rb = (self.surface.get("arm_bands") or {}).get(arm)
        path = (rb.meta.get("map_path") if rb is not None else None) if getattr(rb, "meta", None) else None
        if not path or rb.meta.get("map_kind") != "overhang":
            return None, f"no overhang map view for the {arm} arm"
        cache = self.__dict__.setdefault("_v2_overhang_maps", {})
        if path not in cache:
            cache[path] = RMAP.ReachMap(path)
        rm = cache[path]
        geo, why = self._overhang_geometry(t_pose, g, u, z_init, opts)
        if geo is None:
            return False, why
        d3 = th.tensor([float(OH.unit_xy(u)[0]), float(OH.unit_xy(u)[1]), 0.0])
        post = G.t(t_pose[0]).float() + d3 * float(geo["ladder_m"][0])
        pb = st.world_to_base(post[:2])
        ea = float(rm.meta["overhang"]["edge_along_m"])
        lat = float(rm.meta["config"]["lattice_m"])
        d_es = [float(d) for d in rm.meta["edge_distances"]]
        trunks = [(float(t["translate"]), float(t["tilt"])) for t in rm.meta["trunks"] if t.get("eligible")]
        same = [k for k in trunks if abs(k[0] - float(st.translate)) < 1e-6]
        if not same:
            return None, f"trunk translate {st.translate:.2f} not in the map ({sorted({k[0] for k in trunks})})"
        tk = min(same, key=lambda k: abs(k[1] - float(st.tilt)))
        base_name = C.split_twin(getattr(g, "name", str(g)))[0].split(GL_MEMBER_SEP)[0]
        tried = []
        for off in [0.0] + [float(o) for o in opts["grasp_back_offs_m"]]:
            d_impl = float(pb[0]) + off - ea
            d = min(d_es, key=lambda v: abs(v - d_impl))
            if abs(d - d_impl) > 0.03:
                tried.append(f"back-off {off:.2f}: edge distance {d_impl:.3f} m not in the map {d_es}")
                continue
            row = np.nonzero(np.abs(rm.cells[:, 0] - (d + ea)) <= 1e-4)[0]
            if not len(row):
                continue
            k = row[int(np.argmin(np.abs(rm.cells[row, 1] - float(pb[1]))))]
            if abs(float(rm.cells[k, 1]) - float(pb[1])) > 0.51 * lat + 1e-6:
                tried.append(f"back-off {off:.2f}: lateral {float(pb[1]):.3f} m outside the map's cells")
                continue
            st_k = rm.status(d, *tk)[k, 0]
            feas = [rm.meta["grasps"][i]["base"] for i in np.nonzero(st_k == RMAP.STATUS["feasible"])[0]]
            if base_name in feas:
                return True, ""
            tried.append(f"back-off {off:.2f}: d_e {d:.2f} trunk {tk} cell y {float(rm.cells[k, 1]):+.2f}: {base_name} not feasible (feasible {sorted(set(feas))[:6]})")
        if all("not in the map" in t or "outside" in t for t in tried) and tried:
            return None, "; ".join(tried)[:400]
        return False, "; ".join(tried)[:400]

    def _overhang_b_rank(self, entry, z_init: float, opts: Dict[str, Any]) -> Tuple[int, float, Optional[float]]:
        """How a pre-screened B candidate (``(n_F, stance, F_init, by_arm)``) suits the overhang pick, without an IK: ``(0, -reach margin,
        push)`` when a pair of an arm's F_init has a push-distance ladder (:meth:`_overhang_geometry`) and the push's start (behind the
        object's far side, at the fingertips' height) lies within that arm's reach (:meth:`reach_margin` >= -2 cm); else ``(1, 0, None)``."""
        _, lead, _, by_arm = entry
        if lead.edge is None or lead.focus is None:
            return 1, 0.0, None
        u = OH.unit_xy(-lead.edge.inward)
        best: Optional[Tuple[int, float, Optional[float]]] = None
        for a, (st, F) in by_arm.items():
            for yaw, g in {(round(float(y), 3), getattr(g_, "name", str(g_))): (y, g_) for y, g_ in F}.values():
                t_pose = self.pose_on(lead.focus[:2], yaw, z_init)
                geo, _ = self._overhang_geometry(t_pose, g, u, float(z_init), opts)
                if geo is None:
                    continue
                c = G.t(t_pose[0]).float()
                start = th.tensor([float(c[0]) - float(u[0]) * (geo["extent_back_m"] + 0.03), float(c[1]) - float(u[1]) * (geo["extent_back_m"] + 0.03),
                                   float(z_init) + float(opts["tip_m"]) + C.FINGERTIP_LENGTH])
                m = self.reach_margin(st, start, a)
                if m < -0.02:
                    continue
                post = (c + th.tensor([float(u[0]), float(u[1]), 0.0]) * float(geo["ladder_m"][0]), G.t(t_pose[1]).float())
                pre = IK.pre_pose(IK.grasp_pose(post, g.pose))
                if float(st.world_to_base(pre[0][:2])[0]) < float(opts["pre_min_ahead_m"]):
                    continue  # the grasp's pre-grasp at the overhang nearer the robot's body than the key allows
                cand = (0, -m, geo["ladder_m"][0])
                if best is None or cand < best:
                    best = cand
        return best if best is not None else (1, 0.0, None)

    def _overhang_touch_ok(self, chain: Sequence[G.Pose], arm: str, opts: Dict[str, Any]) -> Tuple[bool, str, float]:
        """The light sweep of a push chain solved with the support out of the world - the rule of ``approach_finger_support_touch``
        (``IKChecker._finger_touch_ok``) at the push's own opening: every configuration of the chain (a) collision-free at margin 0 with the
        support and the target out (every other object, self-collision), (b) every link but ``arm``'s finger links clear of the world at
        margin 0 (the support present), (c) the finger links clear of it at ``-support_touch_m`` (light contact). ``(ok, why, the deepest finger
        penetration in m, to the mm)``."""
        from ispatialgym.oracle import r1pro_model as M

        qs = O.chain_qs(self.ikc, arm, chain)
        others = self.ikc._non_finger_links(arm)
        if not qs or not others:
            return False, "the chain's configurations or the sphere-to-link map are missing", 0.0
        q = th.stack([G.t(x).float() for x in qs])
        fingers = list(M.FINGER_LINKS[arm])
        allow = float(opts["support_touch_m"])
        with self.ikc.ignoring([self.target] + list(self.support_objects())):
            if not bool(self.kit.collision_free(q, margin=0.0).all()):
                return False, "a configuration collides with something else than the support", 0.0
        with self.ikc.ignoring([self.target]):
            if not bool(self.kit.links_clear(q, others, 0.0).all()):
                return False, "a link other than the fingers meets the support", 0.0
            if not bool(self.kit.links_clear(q, fingers, -allow).all()):
                return False, f"a finger more than {allow * 1000:.1f} mm into the support", 0.0
            deep = 0.0
            for k in range(0, int(math.ceil(allow * 1000)) + 1):
                if bool(self.kit.links_clear(q, fingers, -0.001 * k).all()):
                    deep = 0.001 * k
                    break
        return True, "", deep

    def _overhang_push_chain(self, st: Stance, arm: str, t_pose: G.Pose, d, push: float, opts: Dict[str, Any], cam) -> Tuple[Optional[Dict[str, Any]], str]:
        """The push from B that moves the object at ``t_pose`` ``push`` m along ``d``: per push hand (``fingers``) and pitch the contact
        behind the object's far side (:func:`overhang.contact_along`), the chain ``[pre, start, mid, goal]`` (:func:`overhang.push_chain_poses`),
        both wrist turns by the orientation rule; exact IK from ``st`` at the hand's opening with the target out of the world and the
        support present, else (``support_touch_m`` > 0) with it out and the light-sweep rule (:meth:`_overhang_touch_ok`); the fingertips
        of the start and the goal in B's image. ``(chain facts, "")`` or ``(None, why)``."""
        u = OH.unit_xy(d)
        z_top = float(self.surface["z"])
        tip_z = z_top + float(opts["tip_m"])
        root = G.t(t_pose[0]).float()
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(t_pose[1]).float()))
        rel = th.cat([pts[:, :2], (pts[:, 2] + float(root[2])).reshape(-1, 1)], dim=1)
        sup = list(self.support_objects())
        why: List[str] = []
        for fingers in opts["fingers"]:
            opening = OH.OPEN_FINGERS_M if fingers == "open" else C.FINGER_CLOSED
            for pitch in opts["pitches_deg"]:
                z_hi = tip_z + OH.FINGER_SECTION_M * math.cos(math.radians(pitch))
                contact = OH.contact_along(rel, u, OH.lateral_bands(fingers), tip_z, z_hi, pitch, tip_z)
                if contact is None:
                    why.append(f"{fingers} pitch {pitch:.0f}: nothing of the object in the fingers' bands")
                    continue
                props, touched = [], {}
                for rank, sign in enumerate((1.0, -1.0)):
                    quat = OH.push_quat(pitch, u, sign)
                    ch = OH.push_chain_poses(root[:2], u, contact, pitch, quat, tip_z, OH.face_half(fingers), opts["contact_clearance_m"], push)
                    props.append(O.Proposal(f"{fingers} pitch {pitch:.0f}{'' if sign > 0 else '-'}", rank, ch, [ch["pre"], ch["start"], ch["mid"], ch["goal"]]))

                def evaluate(p, opening=opening):
                    with self.ikc.ignoring([self.target]):
                        res = self.ikc.exact(p.chain, st, arm, cache=False, fingers=opening, q_first=p.q_first)
                    if res.ok:
                        return True, res, None, ""
                    if float(opts["support_touch_m"]) <= 0.0:
                        return False, res, None, str(res.reason)
                    with self.ikc.ignoring([self.target] + sup):
                        res2 = self.ikc.exact(p.chain, st, arm, cache=False, fingers=opening, q_first=p.q_first)
                    if not res2.ok:
                        return False, res, None, f"{res.reason}; with the support out: {res2.reason}"
                    ok_t, why_t, deep = self._overhang_touch_ok(p.chain, arm, opts)
                    if not ok_t:
                        return False, res2, None, f"{res.reason}; with the support out: {why_t}"
                    touched[p.label] = deep
                    return True, res2, None, ""

                chosen, decision = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(st, fingers=opening), camera_idx=[1],
                                            stage=f"overhang push {fingers} pitch {pitch:.0f}")
                if chosen is None:
                    why.extend(f"{o['label']}: {str(o.get('why', ''))[:140]}" for o in decision["options"] if not o.get("ok"))
                    continue
                pre, start, mid, goal = chosen.proposal.chain
                ch = chosen.proposal.payload
                if not (self.ikc.eef_in_frame(start, cam) and self.ikc.eef_in_frame(goal, cam)):
                    why.append(f"{chosen.proposal.label}: the push's fingertips outside B's image (5 %)")
                    continue
                why_b = self._overhang_push_branch(st, arm, chosen, opening)
                if why_b:
                    why.append(f"{chosen.proposal.label}: {why_b}")
                    continue
                label = chosen.proposal.label
                return {"pre": pre, "start": start, "mid": mid, "goal": goal, "direction": ch["direction"], "fingers": fingers, "opening": float(opening),
                        "pitch_deg": float(pitch), "wrist": label, "contact_along_m": round(float(contact), 4), "hand_travel_m": round(float(ch["hand_travel_m"]), 4),
                        "tip_m": float(opts["tip_m"]), "support_touch_m": None if label not in touched else round(touched[label], 4), "orient": decision,
                        "seed": push_chain_seed(self.kit, chosen, arm)}, ""
        return None, "; ".join(why)[:900]

    def _overhang_push_branch(self, st: Stance, arm: str, chosen, opening: float) -> str:
        """oracle-fix 2026-11-05 (opt-in ``generation.overhang_push_branch_check``; hardback hb21_desk_s55, the first instance through the
        program check: its push pre-contact witness lay 3.13 rad from the stance posture's IK branch, the program check moved it there
        (``program_check._rewitness_flipped_push``) and the oracle's straight push from that branch stalled at waypoint 21/28, plan_failed
        after a 1365 s attempt): the chosen push chain must track as the oracle's straight push (``program_check.track_line_push``) from the
        configuration the oracle leaves from - the stance posture's branch when the witness is flipped
        (``program_check._flipped_push_witness``), else the witness itself. ``""`` or why."""
        if not bool(self.gp.gen("overhang_push_branch_check") or False):
            return ""
        from ispatialgym.sampling.v2 import program_check as PCK

        qs = list(getattr(chosen, "qs", None) or [])
        if not qs or qs[0] is None:
            return ""
        chain = list(chosen.proposal.chain)
        flipped = PCK._flipped_push_witness(self, chain, st, arm, qs[0], opening)
        q_from = qs[0] if flipped is None else flipped[1]
        if flipped is not None:
            TT._count(self, "pick_overhang.push_flipped")
        ok, why_l, _ = PCK.track_line_push(self, arm, q_from, chain[1:])
        if ok:
            return ""
        if flipped is not None and bool(self.gp.gen("push_flipped_keep_recorded") or False):
            # the program check keeps a flipped witness whose straight push tracks (program_check._rewitness_flipped_push)
            ok_k, why_k, _ = PCK.track_line_push(self, arm, qs[0], chain[1:])
            if ok_k:
                TT._count(self, "pick_overhang.push_flipped_kept")
                return ""
            why_l = f"{why_l}; from the witness too: {why_k}"
        TT._count(self, "pick_overhang.push_line_fails")
        where = "the witness" if flipped is None else f"the stance posture's branch (the witness is {flipped[0]:.2f} rad away)"
        return f"the straight push from {where} does not track: {why_l}"

    def _overhang_drifted(self, post: G.Pose, st: Stance, arm: str, d=None) -> G.Pose:
        """oracle-fix 2026-11-05 (opt-in ``generation.overhang_push_drift`` = ``{"lateral_m", "yaw_rad"}``; diagnostics hbD3 / hbD7: the
        open-hand push of the hardback 0.16 m along -x left it -2.4 cm (left arm) / +2.65 cm (right arm) aside - away from the pushing arm
        - and turned +0.048 / -0.096 rad; the grasp followed the book and the carry at C, checked from the predicted pose, stalled): the
        predicted post-push pose shifted ``lateral_m`` away from the pushing arm (along B's base -left for the left arm, +left for the
        right) and turned ``yaw_rad`` toward the same side's sign (+ for the left arm), so the grasp, the lift, the carry and the place
        are planned for where the book ends up. ``post`` unchanged when off.
        ``snap_max_rad`` (hb29_desk_s44 demo, 10:58: the book at yaw -1.963 pushed along -x turned to -1.574 = flush with the pad, 0.389 rad,
        not the 0.07 assumed; the pre-grasp then lay 54 mm / 0.32 rad off the witness and closing it touched the torso): a flat pad turns a
        box until a face lies flush with it - with the push direction ``d`` given and the yaw off the nearest flush orientation (the book's
        axes vs ``d``, modulo 90 deg) within ``snap_max_rad``, the predicted turn is that whole offset; nearer 45 deg (the diagnostics' books,
        no face wins) the small ``yaw_rad`` turn stands."""
        drift = self.gp.gen("overhang_push_drift")
        if not isinstance(drift, dict):
            return post
        side = 1.0 if arm == "left" else -1.0
        yaw_b = float(st.base_pose2d[2])
        left = th.tensor([-math.sin(yaw_b), math.cos(yaw_b), 0.0])
        pos = G.t(post[0]).float() - left * side * float(drift.get("lateral_m", 0.0))
        turn = side * float(drift.get("yaw_rad", 0.0))
        snap = drift.get("snap_max_rad")
        if snap is not None and d is not None:
            q0 = G.t(post[1]).float()
            psi = math.atan2(2.0 * (float(q0[3]) * float(q0[2]) + float(q0[0]) * float(q0[1])), 1.0 - 2.0 * (float(q0[1]) ** 2 + float(q0[2]) ** 2))
            r = math.remainder(psi - math.atan2(float(d[1]), float(d[0])), math.pi / 2.0)  # the yaw off the nearest flush orientation, in [-45, 45] deg
            if abs(r) <= float(snap):
                turn = -r
        quat = G.qmul(C.quat_from_yaw(turn).float(), G.t(post[1]).float())
        return (pos, quat / th.linalg.norm(quat))

    def _overhang_settle(self, post: G.Pose, d, opts: Dict[str, Any]) -> Tuple[Optional[str], G.Pose, Dict[str, Any]]:
        """The physics settle of the pushed object (owner: "a physics settle of the pushed plate: not falling, little rotation"): the target
        put at its predicted post-push pose on the support, settled (the robot held), measured (:func:`overhang.settle_facts`) and judged
        (:func:`overhang.settle_verdict`); the world restored (and the target parked again when it was). ``(why or None, settled pose,
        facts)``."""
        from ispatialgym.sampling.sim_utils import PARK_POSITION

        W, target = self.world, self.target
        p0, _ = target.get_position_orientation()
        was_parked = float(p0[2]) > float(PARK_POSITION[2]) - 1.0
        state = W.snapshot()
        try:
            W.place(target, G.t(post[0])[:2], G.t(post[1]), float(self.surface["z"]), local_hull=self.hull, drop=0.003)
            placed = tuple(x.clone().float() for x in target.get_position_orientation())
            residual = W.settle([target], steps=60)
            settled = tuple(x.clone().float() for x in target.get_position_orientation())
        finally:
            W.restore(state)
            if was_parked:
                W.park(target, list(W.spawned).index(target.name))
            self.kit.sync_world()
        facts = OH.settle_facts(placed, settled, d, rest_offset=0.003)
        facts["residual_velocity"] = residual
        return OH.settle_verdict(facts, opts), (settled[0], settled[1]), facts

    def _overhang_pick(self, manip: Stance, st_of: Dict[str, Stance], arms: Sequence[str], g, yaw: float, xy, z_init: float, twins: bool,
                       opts: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """The overhang pick of the drawn pair (``g`` at ``yaw``, the object at ``xy`` on the support) from B (module docstring): the push
        direction toward B's support edge, the geometry (:meth:`_overhang_geometry`), then per arm of ``arms`` (each at its stance B) and
        push distance of the ladder: the grasp at the pushed pose (exact IK with the support present, the twin by the orientation rule,
        in B's image), the push chain (:meth:`_overhang_push_chain`), the settle (:meth:`_overhang_settle`; the grasp re-checked at the
        settled pose when it moved > 2 mm). ``(pick, "")`` - ``arm``, ``t_pose`` (the init pose), ``post_pose`` (settled), ``eef`` /
        ``pre`` / ``res`` / ``pick`` (the grasp at the overhang), ``chain``, ``push_m``, ``facts`` - or ``(None, why)``."""
        if self.surface.get("kind") == "floor" or manip.edge is None:
            return None, "no support edge at B (the floor init never pushes)"
        TT._count(self, "pick_overhang.tried")
        t_pose = self.pose_on(xy, yaw, z_init)
        z_top = float(z_init)
        u = OH.unit_xy(-manip.edge.inward)
        d3 = th.tensor([float(u[0]), float(u[1]), 0.0])
        geo, why_g = self._overhang_geometry(t_pose, g, u, z_top, opts)
        if geo is None:
            TT._count(self, "pick_overhang.no_geometry")
            return None, why_g
        kw = {"ignore_for_last": [], "scale_key": self.choice.models["target"].scale, "cache": False}
        why: List[str] = []
        for arm in arms:
            st = st_of.get(arm, manip)
            cam = self.camera_pose_at(st)
            chain_fail: Optional[str] = None
            for push in geo["ladder_m"]:
                self._check_deadline(f"overhang pick ({arm} arm, push {push:.3f} m) at B ({manip.translate}, {manip.tilt:.2f})")
                post = self._overhang_drifted((G.t(t_pose[0]).float() + d3 * float(push), G.t(t_pose[1]).float()), st, arm, d3)
                grasp_st, g_cam = st, cam
                grasp = self._overhang_grasp(st, arm, g, post, kw, twins, cam)
                if isinstance(grasp, str):
                    # the pre-grasp at the overhang lies ~0.1 m nearer the body than the grasp: from B backed off along its axis (stance "pick")
                    for st_b in self._overhang_grasp_stances(st, arm, opts):
                        g_b = self._overhang_grasp(st_b, arm, g, post, kw, twins, self.camera_pose_at(st_b))
                        if not isinstance(g_b, str):
                            grasp, grasp_st, g_cam = g_b, st_b, self.camera_pose_at(st_b)
                            break
                if isinstance(grasp, str):
                    why.append(f"{arm} push {push:.3f}: grasp {grasp}")
                    continue
                if chain_fail is None:
                    chain, why_c = self._overhang_push_chain(st, arm, t_pose, u, float(push), opts, cam)
                    if chain is None:
                        # the start and the hand do not depend on the push distance: a chain failing from its start fails at every rung
                        chain_fail = why_c
                        why.append(f"{arm} push {push:.3f}: push chain: {why_c}")
                        break
                else:
                    break
                verdict, settled, sfacts = self._overhang_settle(post, u, opts)
                if verdict is not None:
                    TT._count(self, "pick_overhang.settle_rejected")
                    why.append(f"{arm} push {push:.3f}: settle {verdict} ({sfacts})")
                    continue
                if float(th.linalg.norm(G.t(settled[0]).float() - post[0])) > 0.002 or G.quat_angle(G.t(settled[1]).float(), post[1]) > 0.01:
                    grasp = self._overhang_grasp(grasp_st, arm, g, settled, kw, twins, g_cam)
                    if isinstance(grasp, str):
                        why.append(f"{arm} push {push:.3f}: grasp at the settled pose {grasp}")
                        continue
                else:
                    settled = post
                eef, pre, res, pick = grasp
                overhang = OH.overhang_m(geo["edge_finger_m"], push, geo["extent_front_m"])
                facts = {**geo, "arm": arm, "push_m": round(float(push), 4), "overhang_m": round(overhang, 4),
                         "com_inside_m": round(float(geo["edge_com_m"]) - float(push), 4), "settle": sfacts,
                         "grasp_stance": None if grasp_st is st else {"name": grasp_st.name, "stand_off": grasp_st.stand_off, "trunk": [grasp_st.translate, round(grasp_st.tilt, 3)]},
                         "chain": {k: chain[k] for k in ("fingers", "opening", "pitch_deg", "wrist", "contact_along_m", "hand_travel_m", "tip_m", "support_touch_m")}}
                TT._count(self, "pick_overhang.found")
                if geo.get("v2_rest"):
                    TT._count(self, "pick_overhang.v2_found")
                log.info("%s: overhang pick at B (%.2f, %.2f), %s arm, grasp %s yaw %.2f: push %.3f m toward the edge (edge %.3f m, overhang %.3f m, centre of mass "
                         "%.3f m inside), %s hand pitch %.0f (%s), settle %s, grasp from %s", self.gp.variant, manip.translate, manip.tilt, arm, getattr(g, "name", g), yaw, push,
                         geo["edge_finger_m"], overhang, facts["com_inside_m"], chain["fingers"], chain["pitch_deg"], chain["wrist"], sfacts,
                         "B" if grasp_st is st else f"B backed off to stand-off {grasp_st.stand_off:.2f} at trunk ({grasp_st.translate:.2f}, {grasp_st.tilt:.2f})")
                return {"arm": arm, "stance": st, "grasp_stance": grasp_st, "t_pose": t_pose, "post_pose": settled, "predicted_pose": post, "eef": eef, "pre": pre, "res": res,
                        "pick": pick, "chain": chain, "push_m": float(push), "facts": facts}, ""
        TT._count(self, "pick_overhang.failed")
        return None, f"ladder {geo['ladder_m']} m: " + " | ".join(why)[:1200]

    def _overhang_memo_key(self, manip: Stance, st_of: Dict[str, Stance], arms: Sequence[str], g, yaw0: float, focus_xy, z_init: float,
                           twins: bool) -> Optional[Tuple[Any, ...]]:
        """oracle-fix 2026-11-05 (opt-in ``generation.overhang_fail_memo``; hardback): the key under which a FAILED overhang pick is remembered
        for the rest of the run - B's base pose and trunk, each offered arm's stance, the library row, the yaw bucket, the focus cell, the
        support height. The pick depends on nothing else that changes between attempts (the scene, the options and the grasp-stance memo are
        the run's), so the same inputs fail the same way: hb16_desk_s3 logged 199 overhang failures of 105 distinct texts (up to 11 repeats),
        hb21_desk_s42 the same 8 pairs at its one B in 16-19 of 18 attempts. None when off."""
        if not bool(self.gp.gen("overhang_fail_memo") or False):
            return None

        def st_key(st: Stance) -> Tuple[Any, ...]:
            return tuple(round(float(v), 3) for v in st.base_pose2d) + (round(float(st.translate), 3), round(float(st.tilt), 3))

        return (st_key(manip), tuple((a, st_key(st_of.get(a, manip))) for a in arms), getattr(g, "name", str(g)), round(float(yaw0), 4),
                tuple(round(float(v), 3) for v in G.t(focus_xy).reshape(-1).tolist()), round(float(z_init), 4), bool(twins))

    def _overhang_memo_persisted(self, key: Tuple[Any, ...], why: Optional[str] = None) -> Optional[str]:
        """Opt-in ``generation.overhang_fail_memo_persist`` (oracle-fix 2026-11-05, speed): the failed-pick memo of
        :meth:`_overhang_memo_key` in the sampling cache (``ISPATIALGYM_SAMPLING_CACHE``; job_val.sh SHARED_SAMPLING_CACHE shares it between
        the jobs of one alias), keyed with the IK cache scope (scene, rooms, crop) and the alias's whole generation block - a customs change
        is another key. The hardback's attempt 0 spent ~1100 s re-proving picks earlier jobs had failed (only witnesses are cached). With
        ``why``: stores the failure; without: the stored failure or None."""
        if not bool(self.gp.gen("overhang_fail_memo_persist") or False):
            return None
        from ispatialgym.sampling.cache import cache_get, cache_key, cache_put

        gen = {k: v for k, v in (getattr(self.gp, "generation", None) or {}).items() if not str(k).startswith("_")}
        ck = cache_key("overhang_fail_v1", {"scope": getattr(self.ikc, "cache_scope", {}), "variant": self.gp.variant, "generation": gen, "key": list(key)})
        if why is not None:
            cache_put(ck, {"why": str(why)[:900]})
            return why
        hit = cache_get(ck)
        return None if not isinstance(hit, dict) else hit.get("why")

    def _overhang_grasp_stances(self, st: Stance, arm: str, opts: Dict[str, Any]) -> List[Stance]:
        """The grasp stances of the two-stance overhang pick (stance ``"pick"``): B backed off along its axis by ``grasp_back_offs_m``,
        at B's trunk then ``grasp_trunks``, B's arms and idle posture, on free floor and whole-body collision-free (memoised per B and arm
        for the attempt). The push stays B's; the program drives B -> pick (a recorded transition) between the push's retreat and the grasp."""
        if st.edge is None or not opts["grasp_back_offs_m"]:
            return []
        key = (tuple(round(float(v), 4) for v in st.base_pose2d), round(float(st.translate), 3), round(float(st.tilt), 3), arm)
        memo = self.__dict__.setdefault("_overhang_grasp_stance_memo", {})
        if key in memo:
            return memo[key]
        out: List[Stance] = []
        for off in opts["grasp_back_offs_m"]:
            base = st.shifted(stand_off=float(st.stand_off or 0.0) + float(off), name=OH.PICK_STANCE)
            if not P0.base_ok(self, base.base_xy):
                continue
            for tr, ti in [(float(st.translate), float(st.tilt))] + [t for t in opts["grasp_trunks"] if abs(t[0] - st.translate) + abs(t[1] - st.tilt) > 1e-6]:
                cand = base.with_trunk(tr, ti, OH.PICK_STANCE)
                cand.focus, cand.extras = st.focus, {"overhang_grasp_stance": True, "backed_off_m": float(off)}
                if bool(P0.body_free(self, [cand])[0]):
                    out.append(cand)
        memo[key] = out
        return out

    def _overhang_grasp(self, st: Stance, arm: str, g, obj_pose: G.Pose, kw: Dict[str, Any], twins: bool, cam):
        """The library grasp ``g`` at ``obj_pose`` (the pushed object) from ``st``: exact IK ``[pre-grasp, grasp]`` with the support present
        and the target out of the world, the twin by the orientation rule (no support-plane camera rule: the support is in the world), the
        grasp in B's image. ``(eef, pre, res, pick)`` or the reason it fails."""
        eef = IK.grasp_pose(obj_pose, g.pose)
        pre = IK.pre_pose(eef)
        with self.ikc.ignoring([self.target]):
            res = self.ikc.exact([pre, eef], st, arm, **kw)
            pick = None
            if twins and getattr(g, "name", None):
                pick = TT.pick_twin(self, st, arm, g, g.pose, eef, pre, res, kw, support_z=None, where="B, overhang", arms=[arm], stances={arm: st})
                res = pick.res
                eef, pre = pick.eef, pick.pre
        if not res.ok:
            return str(res.reason)[:200]
        if not self.ikc.eef_in_frame(eef, cam):
            return "outside B's image (5 %)"
        # oracle-fix 2026-11-05: the pre-grasp and the approach's end judged by the program check's own fingertip gate (5 % + 2 cm slack) from
        # this stance - hb16_desk_s3 / hb17_desk_s3_diag attempt 0: the grasp from B backed off 0.10 m passed the eef box above, the program
        # check rejected its pre-grasp at v 1058.7 (depth 0.425 m) after 3410 s; the next back-off keeps the hand deeper in the image
        from ispatialgym.sampling.v2 import program_check as PCK
        out = PCK.fingertips_in_frame(self, st, "grasp", [("pre_grasp", [pre]), ("approach", [eef])])
        if out is not None:
            return f"{out['key']} fingertip outside the image (5 %): u {out['u']}, v {out['v']}, depth {out['depth_m']} m"
        return eef, pre, res, pick

    def overhang_records(self, ov: Dict[str, Any], grasp: GraspRecord) -> None:
        """The overhang pick's records: ``contacts.push_pick`` (``kind: "pick_overhang"``; ``distance_m`` = the object's displacement) and
        ``grasps[0].pick_overhang`` + ``ignore_last`` / ``ignore_every`` (the program check keeps the support in the grasp)."""
        ch, f = ov["chain"], ov["facts"]
        post = ov["post_pose"]
        post_d = {"position": [round(float(v), 5) for v in G.t(post[0]).tolist()], "orientation": [round(float(v), 6) for v in G.t(post[1]).tolist()]}
        pred = ov["predicted_pose"]
        sup = self.bindings.one("support").name if self.bindings.has("support") else None  # the bddl instance (the push's QA whitelist)
        ignore_every = [self.tname] + (["supports"] if ch.get("support_touch_m") is not None else [])
        rec = ContactRecord(OH.CONTACT_SLOT, ov["arm"], ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], float(ov["push_m"]), None, self.tname, "manipulation")
        rec.extras = _jsonable({"kind": OH.KIND, "support": sup, "fingers": ch["fingers"],
                                **({"fingers_opening_m": ch["opening"]} if ch["fingers"] == "open" else {}), "pitch_deg": ch["pitch_deg"], "tip_m": ch["tip_m"],
                                "hand_travel_m": ch["hand_travel_m"], "contact_along_m": ch["contact_along_m"], "edge_m": f["edge_finger_m"], "overhang_m": f["overhang_m"],
                                "com_inside_m": f["com_inside_m"], "post_push_pose": post_d,
                                "predicted_pose": {"position": [round(float(v), 5) for v in G.t(pred[0]).tolist()], "orientation": [round(float(v), 6) for v in G.t(pred[1]).tolist()]},
                                "settle": f["settle"], "support_touch_m": ch.get("support_touch_m"), "ignore_every": ignore_every, "ignore_last": [],
                                **({"orient": ch["orient"]} if ch.get("orient") else {}), **ch.get("seed", {})})
        up = self.gp.gen("overhang_retreat_up_m")
        if isinstance(up, (int, float)) and up > 0:
            # oracle-fix 2026-11-05 (opt-in generation.overhang_retreat_up_m; diagnostic hbD1: after the 5 cm escape the base's back-off to the
            # grasp stance left the hand at the book's far end and no transit to the pre-grasp planned with the desk present): the escape off
            # the support is this long (programs._pick_actions -> Retreat.escape_up_m; the program check replays the same leg)
            rec.extras["retreat_up_m"] = float(up)
        if bool(self.gp.gen("carry_via_online") or False):
            # oracle-fix 2026-11-05 (opt-in generation.carry_via_online; diagnostic hbD4: the push left the book 2.7 cm aside / 5.5 deg turned,
            # the carry checked from this lift configuration stalled from the real one): the oracle re-searches the carry at C from where the
            # arm is when the recorded leg does not track (execute._carry_online)
            grasp.extras["carry_online"] = True
        self.sidecar.add_contact(rec)
        if sup is not None and self.surface.get("obj") is not None:
            self.sidecar.describe(sup, str(self.surface["obj"].category).replace("_", " "))  # the push's annotation names the support
        opts = self.pick_overhang_opts() or {}
        c = self.choice.models["target"]
        role = (getattr(self.gp, "roles", None) or {}).get("target")
        tags = role.equivalence_tags(c.category, c.model) if role is not None and hasattr(role, "equivalence_tags") else []
        yaw_free = bool(opts.get("yaw_free", True)) and OH.revolution_about_z(tags)
        grasp.extras.update(_jsonable({"pick_overhang": {"contact": OH.CONTACT_SLOT, "post_push_pose": post_d, "overhang_m": f["overhang_m"], "push_m": f["push_m"],
                                                         "finger_clearance_m": float(opts.get("finger_clearance_m", 0.0)), "sweep_along_m": f["sweep_along_m"],
                                                         "yaw_free": yaw_free},
                                       "ignore_last": [], "ignore_every": []}))

    #: failed attached place / drop IKs per C candidate logged with the failing pose, the IK's reason and the contacts (owner 2026-10-13
    #: ~00:50 item 2; diagnostics only)
    PLACE_IK_DIAG_PER_C = 3

    def _ik_solution_of(self, arm: str, pose) -> Optional[th.Tensor]:
        """``IKChecker.solutions``' entry of ``pose`` (``exact`` writes a fresh one for every pose it solves), None without one."""
        try:
            return self.ikc.solutions.get((arm, self.ikc._pose_key(pose)))
        except Exception:  # noqa: BLE001
            return None

    def _place_ik_fail_line(self, r2, place: Stance, g, yaw: float, tilt: float, pre_place, place_eef, arm: str, q_seed, ignore_place, pre_before,
                            push: bool = False) -> str:
        """The log line of a failed attached [pre-place, place] exact IK at C (diagnostic, called inside the attachment, no effect on the draw):
        the pose that failed - the place / drop when the call left a fresh pre-place solution (``pre_before``: the entry before the call) -,
        the IK's reason and the contacts of a configuration reaching that pose (:meth:`_place_ik_contacts`, from the pre-place solution /
        the carry seed)."""
        q_pre = self._ik_solution_of(arm, pre_place)
        pre_ok = q_pre is not None and q_pre is not pre_before
        which, pose, ign = ("drop" if push else "place", place_eef, list(ignore_place)) if pre_ok else ("pre-place", pre_place, [])
        try:
            contacts = self._place_ik_contacts(pose, q_pre if pre_ok else (q_seed if q_seed is not None else self.ikc.q_locked(place)), arm, ign)
        except Exception as e:  # noqa: BLE001 - a diagnostic never fails the draw
            contacts = f"not computed ({e!r})"[:200]
        return (f"C place IK (attached) fail at d {place.stand_off or 0.0:.2f} λ {place.lateral:+.2f} trunk ({place.translate:.2f}, {place.tilt:.2f}): "
                f"grasp {getattr(g, 'name', g)} yaw {float(yaw):.2f} tilt {float(tilt):g}: pose {which}: {r2.reason}; contacts: {contacts}")

    def _place_ik_contacts(self, pose, q_start, arm: str, ignored) -> str:
        """Contacts of a configuration reaching ``pose`` (diagnostic): damped least squares from ``q_start`` with collisions off (as
        ``IKChecker.polish_witness``), then with ``ignored`` out of the world ``links x object/mesh`` per collision mesh of the partner or a
        spawned object whose removal frees links (the carried object is ``attached_object_<arm>_eef_link``), the self-collision pairs and
        ``_collision_blame`` for links no such mesh frees - or why there are none (not reached: reach / joint limits; no contact: the
        IK's seeds or its joint-limit margin)."""
        from ispatialgym.oracle import r1pro_model as M

        be, ikc = self.kit.backend, self.ikc
        # oracle-fix 2026-11-04 round 3: tracked along the line from the start (IKChecker.reach_collisions_off); the former single far
        # waypoint returned no configuration and the residual of the START was reported (the bookcase's 674-846 mm 'short')
        if hasattr(ikc, "reach_collisions_off"):
            r = ikc.reach_collisions_off(pose, [G.t(q_start)], arm)
            q = r["q"]
            lim = f"{r['limit_joint']} {r['limit_mrad']:.0f} mrad from its limit"
            if not r["reached"]:
                return (f"none - not reached with collisions off ({r['short_mm']:.0f} mm short, {r['rot_rad']:.2f} rad off, {lim}"
                        f"{', at limits ' + '/'.join(r['at_limits']) if r['at_limits'] else ''})")
        else:  # a checker without the tracked solve (tests' stubs): one DLS solve from the start
            self.kit._ensure_synced()
            res = be.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(pose[0]), G.t(pose[1]))}], q_start=G.t(q_start), pos_tol=1e-3, rot_tol=5e-3,
                                   max_iters=100, check_collisions=False, sync_world=False, max_joint_step=0.3)
            q = res.q_traj[-1].clone() if getattr(res, "q_traj", None) is not None else G.t(q_start).clone()
            m_lim, jn = ikc.joint_limit_margin(q, arm)
            lim = f"{jn} {m_lim * 1000:.0f} mrad from its limit"
            if not ikc.reaches(q, pose, arm):
                return f"none - not reached with collisions off ({ikc.fk_residual(q, pose, arm) * 1000:.0f} mm short, {lim})"
        pairs: List[str] = []
        with ikc.ignoring(list(ignored)):
            if bool(self.kit.collision_free(q, margin=0.0)[0]):
                return f"none at a configuration reaching the pose with collisions off ({lim})"
            touches = lambda link: bool(be.link_proximity(q, [link], 0.0, emb_name="whole_body")[0])  # noqa: E731
            hit = {l for l in set(be._collision_costs(be.kin_entry("whole_body"))["link_of_sphere"]) if touches(l)}
            off = set(be._disabled_obstacles) | set(getattr(be, "_held_off", set()))
            freed_any: set = set()
            for o in [self.partner] + list(getattr(self.world, "spawned", {}).values()):
                for link in getattr(o, "links", {}).values():
                    for cm in getattr(link, "collision_meshes", {}).values():
                        name = cm.prim_path
                        if not hit or name not in be._world_mesh_names or name in off:
                            continue
                        be._mesh_enable(name, False)
                        try:
                            freed = {l for l in hit if not touches(l)}
                        finally:
                            be._mesh_enable(name, True)
                        if freed:
                            freed_any |= freed
                            pairs.append(f"{'/'.join(sorted(freed))} x {getattr(o, 'name', o)}/{name.rsplit('/', 1)[-1]}")
        pairs += [f"{a} x {b} (self, {pen * 1000:.0f} mm)" for (a, b), pen in ikc.self_collision_pairs(q)[:2]]
        rest = sorted(hit - freed_any)
        if rest or not pairs:
            pairs.append(f"{'/'.join(rest) or 'no link in the world'}: {ikc._collision_blame(q, list(ignored))}")
        return f"{', '.join(pairs)} ({lim})"

    def _couple(self, rng, init: Stance, manip: Stance, c_pool: List[Stance], F_init, focus_xy, z_init: float, seg: Optional[SegmentArms] = None) -> Draft:
        """B ↔ C coupling: for C survivors in random order, F_goal at the goal cell for the init yaws, the draw from
        F_init ∩ F_goal, the target pose at B's focus, the exact IK at B and at C (attached), the B → C path and
        transition; the draft and the sidecar records. For the segment arm (``self.seg_arm``; owner 2026-09-27, first-demo v2.2):
        ``F_init`` / ``c_pool`` are its own, the grasp at B is ``tabletop.pick_twin`` over the arms of ``seg`` whose F_init holds the
        pair (shared with the other arm's coupling: a pair the other arm won is left to its coupling, one it won and lost later is
        regrasped here - :meth:`SegmentArms.role`)."""
        tname, pname = self.tname, self.pname
        partner, target = self.partner, self.target
        arm = self.seg_arm
        arm0 = TT.default_arm(self)
        offered = TT.offered_arms(self)
        # §5.6: B → C ≤ max_partner_geodesic + extra. The geodesic rule measures near side to near side; the recorded stances
        # sit at particular edges (the bar's short end → the sofa's front), so the stance-to-stance drive is longer — extra
        # 3 m (sofa-bar 17:04: every C survivor failed with path_too_long at 4 m; the B-C IK never got its turn)
        geo = self.num("max_partner_geodesic_m", 3.0)
        # the customs pin path_max_extra_m = 1.0 (D1 generation_defaults): the stance-to-stance drive is bounded here by
        # max(geodesic + extra, 2 × geodesic) — the base rounds the support / partner between B and C (D1: raise the default)
        max_len = max(geo + self.num("path_max_extra_m", 3.0), 2.0 * geo)
        order = list(rng.permutation(len(c_pool)))
        n_try = int(self.num("c_candidates_max", 12))  # C survivors tried per B candidate (the per-attempt budget bounds the total)
        if self.push_under:
            # side stances first (the frontal ones put the lane out of the floor reach — see push_under_candidates), the
            # deepest trunk first within a kind (B3: the floor is reached at translate 0.95 / tilt −1.2 only); every side
            # survivor is tried before the frontal ones
            side = [k for k in order if c_pool[k].extras.get("stance_kind") == "side"]
            front = [k for k in order if c_pool[k].extras.get("stance_kind") != "side"]
            # frontal stances: the push goal must stay inside the floor reach (B3: 0.24-0.64 m ahead at the deep trunks) - the goal
            # lies drop_ahead + push distance (~0.3 m) ahead, so the drop cell should sit ~0.3 m ahead; ordering by the LARGEST
            # drop_ahead first (the bed heuristic: keep the deep trunk clear of the frame) put every chain at 0.7-0.8 m ahead and
            # 260 chain searches failed the exact IK in 0919-233031
            # (2026-09-20 01:40 correction: with the 0.30 m preference the 0.65-0.70 m stand-offs came first and every chain failed
            # its PRE-pose - the blade start sits on the base rim there; the two instances that were written (0919-211642 /
            # 221909) used d 0.80 with the goal ~0.8 m ahead, which the deep crouch does reach. Default = the old order, largest
            # drop_ahead first; ``generation.push_under_drop_ahead_pref_m`` switches to the nearest-to-preference order.)
            ahead_pref = self.push_drop_ahead_pref()
            # the lane to the robot's RIGHT first (frontal_lane_offset_m 0.25): B3's under-edge probe reaches the floor strip at
            # lateral -0.40..-0.16 only, and the one chain that ever solved (0919-211642) ran at lateral -0.25; the centred lane
            # (offset 0) tied with it in the order and 482 centred chains failed their pre-pose in 0920-040824
            lane = lambda k: 0 if abs(float(c_pool[k].extras.get("lane_offset_m", 0.0))) > 1e-6 else 1  # noqa: E731
            erank = lambda k: int(c_pool[k].extras.get("edge_rank", 0))  # noqa: E731  (the nearest allowed partner edge first)
            depth = lambda k: (erank(k), -float(c_pool[k].translate), float(c_pool[k].tilt), -float(c_pool[k].extras.get("drop_ahead_m", 0.0)), lane(k))  # noqa: E731
            reach = lambda k: (erank(k), lane(k), abs(float(c_pool[k].extras.get("drop_ahead_m", 0.0)) - float(ahead_pref)), -float(c_pool[k].translate), float(c_pool[k].tilt))  # noqa: E731
            if str(self.gp.gen("c_order", "travel")) == "travel":
                # W3 (owner 2026-09-20 22:15): the survivors of the feasibility filters (collision-free crouch, free drop lane, goal cell
                # under the top) are ordered by the B -> C base travel (shortest path on the eroded map, 0.25 m buckets; the straight
                # distance as fallback), deterministic given B: the same B gave C at the far end (4.2 m), the middle (3.2 m) and the near
                # end (2.6 m) of the table with the seeded permutation deciding (0920-143722 / 154503 / 154504). Within a bucket the
                # right lane, the stand-off nearest ``push_under_drop_ahead_pref_m`` and the deepest trunk keep their order.
                trav_cache: Dict[Any, float] = {}
                trav = lambda k: round(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) / 0.25)  # noqa: E731
                pref = float(ahead_pref) if ahead_pref is not None else 0.6
                key_t = lambda k: (trav(k), lane(k), abs(float(c_pool[k].extras.get("drop_ahead_m", 0.0)) - pref), -float(c_pool[k].translate), float(c_pool[k].tilt))  # noqa: E731
                order = sorted(side + front, key=key_t)
                if order:
                    k0 = order[0]
                    log.info("%s: C order by travel from B %s: nearest %s (%.2f m, edge %s, d %.2f, λ %+.2f), farthest %.2f m over %d survivors", self.gp.variant,
                             [round(float(v), 2) for v in manip.base_xy.tolist()], [round(float(v), 2) for v in c_pool[k0].base_xy.tolist()],
                             NAV.travel_length(self.kit, manip.base_xy, c_pool[k0].base_xy, trav_cache), c_pool[k0].extras.get("edge"), c_pool[k0].stand_off or 0.0,
                             c_pool[k0].lateral, max(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) for k in order), len(order))
            else:
                order = sorted(side, key=depth) + sorted(front, key=reach if ahead_pref is not None else depth)
            n_try = max(int(self.num("c_tries", 36)), len(side))  # the chain-first screen makes a C try cheap (a few IKs) until one has a chain
        elif str(self.gp.gen("c_order", "travel")) == "travel":
            # W3: ontop / place variants - the nearest survivor by base travel first (edge rank as the tie-break), deterministic given B
            trav_cache = {}
            order.sort(key=lambda k: (round(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) / 0.25), int(c_pool[k].extras.get("edge_rank", 0))))
        else:
            order.sort(key=lambda k: int(c_pool[k].extras.get("edge_rank", 0)))  # ontop / place variants: nearest edge first, random within
        trunk_first = bool(self.gp.gen("c_order_trunk_first", False))
        if trunk_first:
            # i7-placeorder (owner decision 2026-11-07, option 1 for inside_large_01): the least extended trunk first, then the nearest
            # stand-off - the order only (:func:`trunk_first_order`); the outcome memory's reweighting (i7-memory) layers on this order
            order = trunk_first_order(c_pool, order)
            TT._count(self, "stance_c.trunk_first")
        least_translate = min((float(c.translate) for c in c_pool), default=0.0)
        # i7-memory (owner 2026-11-08 01:45 part C): the per-alias outcome memory reweights the C order from outcomes. The order above is
        # the PRIOR and is kept; the candidates of an (arm, trunk[, stand-off]) bin the memory down-weighted move to the end unless the
        # bin keeps its place this coupling (Thompson draw, never below the exploration floor). No memory / nothing down-weighted = the
        # order unchanged and no random number drawn (alias_memory.reorder; kit alias_memory.place.*; provenance in the stance extras)
        order, mem_prov = AM.reorder(self, "place", order, [AM.place_cfg(arm, c) for c in c_pool], rng, max_level=AM.PLACE_HOOK_LEVELS)
        yaw_step = 2.0 * math.pi / max(1, len(self.yaws()))  # the screening grid (yaw buckets of the caches / the intersection)
        chain_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}  # (C index, yaw bucket) -> push chain (push-under: the chain depends on the yaw only)
        seg = seg or SegmentArms({arm: manip}, {arm: list(F_init)}, yaw_step=yaw_step)
        st_of = {**seg.stances, arm: manip}
        # (yaw bucket, grasp) -> the grasp at B: yaw, target pose, the arm that won it, the arms viable there, per arm [(grasp row or twin,
        # grasp eef, pre-grasp, is twin), ...] preferred first + twin_choice; None = infeasible for every arm
        b_ik_cache = seg.b_ik
        deferred: set = set()  # pairs whose grasp at B an arm with its coupling still to come won
        twins_b = TT.grasp_twins_on(self)
        errors = []
        carry_margin = float(self.gp.gen("carry_collision_margin_m", 0.04))
        carry_failed_bases: set = set()  # base poses whose carry leg failed: the trunk variants at the same pose fail the same way
        skip_dead_b = bool(self.gp.gen("c_skip_dead_b", False))  # oracle-fix 2026-11-06 (speed): see _c_dead_at_b
        n_dead_b = 0
        for i_c, kc in enumerate(order[:n_try]):
            self._check_deadline(f"C candidate {i_c + 1} of {min(n_try, len(order))} (F_goal feasible set) at B ({manip.translate}, {manip.tilt:.2f})")
            F_live = [yg for yg in F_init if pair_key(yg[0], yg[1], yaw_step) not in deferred] if deferred else F_init
            if not F_live:
                errors.append(f"every pair of the {arm} arm's F_init ({len(F_init)}) at B went to another arm's grasp")
                break
            place = c_pool[int(kc)].with_trunk(c_pool[int(kc)].translate, c_pool[int(kc)].tilt, "placement")
            place.focus, place.arms = c_pool[int(kc)].focus, [arm]
            place.extras = dict(c_pool[int(kc)].extras)
            if int(kc) in mem_prov:
                place.extras["alias_memory"] = mem_prov[int(kc)]  # provenance: the memory version, bin and weight this C was drawn with
            gc = place.extras["goal_cell"]
            push = self.push_under
            if skip_dead_b and self._c_dead_at_b(F_live, b_ik_cache, place, manip, yaw_step):
                # oracle-fix 2026-11-06 (speed, generation.c_skip_dead_b): every draw this C can take would end at 'B: [pre-grasp, grasp] exact IK
                # (cached)' - skipped before its push chain / F_goal / release-tilt screens (only the random draws a skipped C consumed differ)
                n_dead_b += 1
                TT._count(self, "c_skip_dead_b")
                if n_dead_b == 1:
                    log.info("%s: C candidate %d of %d (d %.2f λ %+.2f) and the next like it skipped (generation.c_skip_dead_b): every pair of the %s arm's "
                             "F_init (%d) its draws can take failed its exact IK at B (%.2f, %.2f)", self.gp.variant, i_c + 1, min(n_try, len(order)),
                             place.stand_off or 0.0, place.lateral or 0.0, arm, len(F_live), manip.translate, manip.tilt)
                errors.append(f"C (d {place.stand_off or 0.0:.2f} λ {place.lateral or 0.0:+.2f}): skipped - every F_init pair its draws can take failed its "
                              f"exact IK at B (generation.c_skip_dead_b; {n_dead_b} C skipped)")
                continue
            base_key = tuple(round(float(v), 3) for v in place.base_pose2d)
            if base_key in carry_failed_bases:
                errors.append(f"C (d {place.stand_off or 0.0:.2f} λ {place.lateral:+.2f}): skipped - the carry leg failed at this base pose")
                continue
            # 2026-09-21 02:20: a C whose base pose has no room for the carried sweep's margin is skipped before the F_goal screening
            # (the short end next to B stands 0.38 m from the walls: every arrival sweep failed there at 2.5 / 4 cm while the 1 cm
            # stance checks pass, 0921-012344 / 014859 - 6 candidates x 3-4 min per B). Checked at the first travel trunk with the
            # untucked arms: the wall proximity comes from the body, not the carried object.
            if carry_margin > 0.011:
                tt0 = self.travel_trunks(init, manip)[0] if self.travel_trunks(init, manip) else (0.25, 0.0)
                probe = place.with_trunk(float(tt0[0]), float(tt0[1]), "placement")
                # the BODY's room, not the arms': with the untucked arms the hanging hand hovered over the countertop within 4 cm and
                # every ontop C was skipped (0921-031852, shampoo_dispenser-countertop-floor: 6 attempts, no instance) - the compact
                # ``ready`` posture keeps the hands at the chest (collision-free up to translate 1.25 on the JoyLo curve)
                with self._floor_relaxed():
                    if not self.ikc.stance_collision_free(probe, margin=carry_margin, arms="ready"):
                        why_m = ""
                        try:
                            why_m = " - " + self.ikc._collision_blame(self.ikc.q_locked(probe, arms="ready"), [self.obj("floor")], carry_margin)
                        except Exception:  # noqa: BLE001
                            pass
                        errors.append(f"C (d {place.stand_off or 0.0:.2f} λ {place.lateral:+.2f}): no room for the carried sweep at margin {carry_margin:.3f}{why_m}")
                        log.info("%s: C skipped (d %.2f λ %+.2f edge %s): the body has no room for the carried sweep's margin %.3f at the travel trunk%s", self.gp.variant,
                                 place.stand_off or 0.0, place.lateral, place.extras.get("edge"), carry_margin, why_m)
                        carry_failed_bases.add(base_key)
                        continue
            cd = place.extras.get("drop_cell") if push else None
            # push-under = DROP FIRST, THEN CROUCH (owner 2026-09-20 08:30): the object cannot be carried into the deep push crouch
            # (the hand frozen at any carry pose sweeps into the partner / the floor / the base, and no compact carry pose keeps the
            # grasp orientation within the wrist limits - 0920-043725 .. 075127), so the release happens from a drop trunk at the
            # same base pose ("placement" = the drop stance, drop height ``drop_height_m``), and the deep crouch ("push" stance) is
            # entered with empty hands by a trunk-only transition before the floor push. ``generation.drop_trunks`` (default
            # ``PUSH_UNDER_DROP_TRUNKS``: (0.95, -1.2) first, then moderate ones) lists the trunks tried for the drop stance (the
            # deepest survivors of the C search stay the push stance).
            push_st = None
            if push and str(self.gp.gen("push_under_drop_first", "on")) != "off":
                push_st = place.with_trunk(place.translate, place.tilt, "push")
                push_st.focus, push_st.arms, push_st.extras = place.focus, [arm], dict(place.extras)
                drop_st = None
                for tr, ti in [tuple(float(v) for v in t) for t in (self.gp.gen("drop_trunks") or PUSH_UNDER_DROP_TRUNKS)]:
                    cand = place.with_trunk(tr, ti, "placement")
                    cand.focus, cand.arms, cand.extras = place.focus, [arm], dict(place.extras)
                    with self._floor_relaxed():
                        if bool(P0.body_free(self, [cand])[0]):
                            drop_st = cand
                            break
                if drop_st is None:
                    errors.append(f"C (d {place.stand_off:.2f}): no drop trunk is collision-free at the push stance's base pose")
                    continue
                place = drop_st
            if self.relation == "under" and not push:
                cl = self.clearance_at(gc)
                if cl is not None and not under_admissible(cl, self.place["height"], self.num("under_clearance_extra_m", 0.17)):
                    errors.append(f"C: under clearance {cl:.2f} m at the goal cell < height + 0.17")
                    continue
            yaws = sorted({float(y) for y, _ in F_live})
            d_yaw = 0.0 if str(self.gp.gen("carry_yaw", "relative")) == "world" else float(place.yaw - manip.yaw)
            if push:
                # chain FIRST (2026-09-20 19:20): the floor push's IK depends on where this C stands along the edge (0920-184335: 505 chains
                # at 12 C candidates, every one unreachable, while 0920-143722 / 154503 pushed from other laterals of the same edge) and
                # on the object's yaw through the blade's contact offset - so the chain is solved for the two yaws with the LARGEST
                # contact offset (the toe-first yaws of a sandal: the blade goal nearest the base) before the 60 s F_goal screening,
                # and a C without a chain there is skipped
                d_push = th.as_tensor(place.extras["push_dir"], dtype=th.float32)
                offs = {}
                for y in yaws:
                    o = self.push_contact_offset(G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), d_push)
                    offs[y] = -1.0 if o is None else o
                # generation.push_under_yaw (owner 2026-10-13 ~02:55): only the yaws it keeps are chained and drawn at this C
                yaws_c = self.push_yaws_locked(yaws, d_yaw, d_push, offs, place)
                if yaws and not yaws_c:
                    errors.append(f"C (d {place.stand_off:.2f} λ {place.lateral:+.2f}): the push-under yaw lock {self.push_yaw_lock()} keeps none of the {len(yaws)} yaws")
                    continue
                pst0 = push_st if push_st is not None else place
                best = sorted(yaws_c, key=lambda y: self.push_yaw_rank(pst0, G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), offs[y]))[:2]
                found = False
                for y in best:
                    ck = (int(kc), int(round(y / yaw_step)))
                    if ck not in chain_cache:
                        chain_cache[ck] = self._push_under_chain(push_st if push_st is not None else place, G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), cd, gc, place.extras["push_dir"])
                    if chain_cache[ck] is not None:
                        found = True
                        ch = chain_cache[ck]
                        pst = push_st if push_st is not None else place
                        log.info("%s: push chain SOLVED at C (d %.2f λ %+.2f push trunk (%.2f, %.2f), yaw %.2f offset %.3f m): blade start %s -> goal %s in base, pitch %.0f, tips %.3f m",
                                 self.gp.variant, place.stand_off or 0.0, place.lateral, pst.translate, pst.tilt, y, offs[y],
                                 [round(float(v), 2) for v in P0.to_base_frame(ch["start"][0], pst.base_pose2d, P0.base_z(self)).tolist()],
                                 [round(float(v), 2) for v in P0.to_base_frame(ch["goal"][0], pst.base_pose2d, P0.base_z(self)).tolist()], ch["pitch_deg"], ch["fingertip_z_m"])
                        break
                if not found:
                    lock = self.push_yaw_lock()
                    errors.append(f"C (d {place.stand_off:.2f} λ {place.lateral:+.2f}): no push chain at the {self.PUSH_YAW_ORDER}{' ' + lock if lock else ''} yaws "
                                  f"({', '.join(f'{offs[y]:.2f}' for y in best)} m)")
                    continue
            ignore_place = self.place_ignores()
            # the release pose: the goal cell (ontop / under by placement) or the drop cell outside the edge (push-under)
            release_xy = cd if push else gc
            release_pose = (lambda xy, quat, eio=None: self.drop_pose_at(xy, quat)) if push else (lambda xy, quat, eio=None: self.goal_pose_at(xy, quat, eio))
            # A carried object keeps its yaw RELATIVE TO THE ROBOT (no re-grasp), so at C its world yaw is the init yaw plus
            # the base-yaw change B -> C: F_goal is screened at yaw + d_yaw and intersected with F_init by the init yaw
            # (``generation.carry_yaw = "world"`` restores the spec §5.6 wording "goal yaw = init world yaw", which holds
            # only when B and C share a base yaw - countertop-floor 0916-231451: F_init 3-21 / F_goal 0-5 / F empty on
            # every candidate). The goal orientation written to the TRO is the rotated one.
            F_goal = self.release_feasible(release_xy, [y + d_yaw for y in yaws], release_pose, place, arm, ignore_place)
            F_goal = [(y - d_yaw, g) for y, g in F_goal]
            # owner 2026-10-12 ~22:35 (inside): tilted releases as alternatives to the level one; per pair the tilts (deg, 0 = level) it passed
            tilts_of: Dict[Tuple[int, Any], List[float]] = {}
            if not push:
                F_goal, tilts_of = self.screen_release_tilts(F_goal, release_xy, yaws, d_yaw, place, arm, ignore_place, yaw_step)
            F = intersect_feasible(F_live, F_goal, yaw_step=yaw_step)
            if push:
                log.info("%s: push-under C try %s d %.2f λ %+.2f trunk (%.2f, %.1f): F_init %d, F_goal %d, F %d at drop %s", self.gp.variant, place.extras.get("stance_kind"),
                         place.stand_off or 0.0, place.lateral, place.translate, place.tilt, len(F_live), len(F_goal), len(F), [round(float(v), 2) for v in release_xy])
            if not F:
                if push and not F_goal and not getattr(self, "_drop_debugged", False):
                    self._drop_debugged = True
                    log.info("%s: drop IK debug — %s", self.gp.variant, self.debug_drop_ik(place, release_xy, yaws))
                errors.append(f"C: F_init ({len(F_live)}) ∩ F_goal ({len(F_goal)}) empty at {'drop' if push else 'goal'} cell {[round(v, 2) for v in release_xy]}")
                continue
            fails: Dict[str, int] = {}
            n_place_diag = 0  # failed attached place IKs logged at this C (PLACE_IK_DIAG_PER_C)
            # pairs of F tried per C: the push-under chain search (4 pitches x 2 wrist turns x exact IK) is the expensive leg, so the
            # push mode keeps the earlier 4 draws (0919-233031: 8 draws x 12 C candidates spent the 900 s budget on attempts 0 and 2)
            draw_order = list(rng.permutation(len(F)))
            if push:
                # the blade sits half an extent behind the object centre: the LONG axis along the push puts the blade goal ~7 cm
                # closer to the base (sandal: 0.72 vs 0.78-0.80 m ahead - the chain solved only at the long-axis yaws, 0919-211642;
                # every short-axis chain of 0920-052659 had no IK at its goal) - longest extent first, random within a bucket
                # (2026-09-20 19:20: ordered by the blade's CONTACT offset at its height band - a sandal toe-first before its strap)
                rank = {i: self.push_yaw_rank(pst0, G.qmul(G.quat_from_yaw(float(F[i][0]) + d_yaw), self.rest), offs.get(float(F[i][0]), -1.0)) for i in range(len(F))}
                draw_order.sort(key=lambda i: round(rank[i], 2))
                if len(yaws_c) < len(yaws):  # the push-under yaw lock: the pairs at the yaws it keeps
                    kept_b = {int(round(y / yaw_step)) for y in yaws_c}
                    draw_order = [i for i in draw_order if int(round(float(F[i][0]) / yaw_step)) in kept_b]
                    if not draw_order:
                        fails[f"C: the push-under yaw lock {self.push_yaw_lock()} keeps none of F's yaws"] = len(F)
            n_draws = len(draw_order)
            draw_order = self.alignable_draws(F, draw_order, d_yaw)
            if n_draws and not draw_order:
                fails["C: no aligned goal yaw near F's yaws"] = n_draws
            for i in draw_order[:int(self.num("f_draws_max", 8))]:  # failed B pairs are cached and skipped, so 8 draws cost little
                self._check_deadline(f"F draw at C candidate {i_c + 1} (exact IK at B / C{', push chain' if push else ''})")
                yaw0, g = F[int(i)]
                y_end = self.end_aligned_init_yaw(yaw0, d_yaw)
                if y_end is False:
                    fails["C: no aligned goal yaw near the pair's"] = fails.get("C: no aligned goal yaw near the pair's", 0) + 1
                    continue
                y_init = yaw0 if y_end is None else float(y_end)
                jit = self.init_yaw_jitter()
                yaw = y_init + float(rng.uniform(-jit, jit))
                if push:
                    # the push chain depends on the object's yaw (its half extent along the push), not on the grasp: solve it once per
                    # (C, yaw bucket) BEFORE the grasp IKs and skip the draw when this stance cannot push at that yaw
                    ck = (int(kc), int(round(yaw0 / yaw_step))) if y_end is None else (int(kc), "aligned", round(y_init, 4))
                    if ck not in chain_cache:
                        q_chain = G.qmul(G.quat_from_yaw(y_init + d_yaw), self.rest)
                        chain_cache[ck] = self._push_under_chain(push_st if push_st is not None else place, q_chain, cd, gc, place.extras["push_dir"])
                    if chain_cache[ck] is None:
                        fails["C: push chain unreachable at this yaw"] = fails.get("C: push chain unreachable at this yaw", 0) + 1
                        continue
                # the grasp side (target pose at B's focus, pre-grasp + grasp exact IK from B) does not depend on C: solved once per
                # (yaw bucket, grasp) for this B and reused across the C candidates (a failed pair is skipped everywhere)
                bk = (int(round(yaw0 / yaw_step)) if y_end is None else round(y_init, 4), getattr(g, "name", str(g)))
                if bk not in b_ik_cache:
                    # the exact IK at B is tried at the screened pose first (the focus cell, the grid yaw: what F_init passed) and then
                    # at up to two jittered poses; a pair is cached as infeasible only when all fail, with the reasons logged
                    # (0920-235717/8/36: 7-8 of 8 toe-first draws failed here with one jittered pose each, no reason recorded).
                    # Owner 2026-09-27 (first-demo v2.2): the arms whose F_init holds the pair compete for it (the default arm first,
                    # each at its own stance B) but an arm whose segment already failed at this B; the pick is shared by the couplings
                    held = seg.arms_of(yaw0, g) or [arm]
                    arms_p = [a for a in held if a not in seg.done] or [arm]
                    a0 = arms_p[0]
                    res, t_pose, eef, pre, pick, b_opts_m = None, None, None, None, None, None
                    why_b: List[str] = []
                    ov_opts = self.pick_overhang_opts()
                    overhang = None
                    for k_try in range(int(self.num("b_ik_tries", 3)) if ov_opts is None or ov_opts["mode"] != "only" else 0):
                        yaw_k = y_init if k_try == 0 else y_init + float(rng.uniform(-jit, jit))
                        t_pose = None
                        for _ in range(5):
                            if self.surface["kind"] == "floor" or k_try == 0:
                                xy = focus_xy
                            else:
                                r, a = float(rng.uniform(0, FOCUS_RADIUS)), float(rng.uniform(0, 2 * math.pi))
                                xy = focus_xy + th.tensor([r * math.cos(a), r * math.sin(a)])
                            if self.surface["kind"] != "floor" and not self.surface["top"].polygon_supported(self.world.footprint(self.hull, self.pose_on(xy, yaw_k, z_init)), margin=SUPPORT_MARGIN):
                                continue
                            t_pose = self.pose_on(xy, yaw_k, z_init)
                            break
                        if t_pose is None:
                            why_b.append("target footprint off the init surface")
                            continue
                        eef = IK.grasp_pose(t_pose, g.pose)
                        pre = IK.pre_pose(eef)
                        res = self.ikc.exact([pre, eef], st_of[a0], a0, ignore_for_last=self.grasp_ignore_objects(), scale_key=self.choice.models["target"].scale, cache=False)
                        if (twins_b or len(arms_p) > 1) and getattr(g, "name", None):
                            pick = TT.pick_twin(self, st_of[a0], a0, g, g.pose, eef, pre, res,
                                                {"ignore_for_last": self.grasp_ignore_objects(), "scale_key": self.choice.models["target"].scale, "cache": False},
                                                support_z=float(z_init), where="B", arms=arms_p, stances=st_of, twins=twins_b)
                            res = pick.res
                            self._count_mirrored_ik(pick, seg.bands)
                        if res.ok:
                            arm_k = pick.arm if pick is not None else a0
                            st_k = pick.stance if pick is not None and pick.stance is not None else st_of.get(arm_k, st_of[a0])
                            b_opts_m, why_m = self._b_margin_options(self._b_options(pick, g, eef, pre), st_k, arm_k, "B")
                            if b_opts_m is not None:
                                yaw = yaw_k
                                break
                            res = IK.IKResult(False, None, f"grasp reach margin: {why_m}", 2)
                        why_b.append(f"try {k_try} yaw {yaw_k:.2f}: {res.reason}")
                    if (res is None or not res.ok) and ov_opts is not None:
                        # owner 2026-10-13 ~03:15 item 4: the direct pick failed - push the object to B's support edge, grasp the overhang
                        key_ov = self._overhang_memo_key(manip, st_of, arms_p, g, yaw0, focus_xy, z_init, twins_b)
                        memo_ov = self.__dict__.setdefault("_overhang_fail_memo", {})
                        if key_ov is not None and key_ov not in memo_ov:
                            persisted = self._overhang_memo_persisted(key_ov)
                            if persisted is not None:
                                TT._count(self, "pick_overhang.memo_persisted_hit")
                                memo_ov[key_ov] = persisted
                        if key_ov is not None and key_ov in memo_ov:
                            TT._count(self, "pick_overhang.memo_hit")
                            overhang, why_ov = None, f"{memo_ov[key_ov]} (remembered: the same pick failed in an earlier attempt)"
                        else:
                            overhang, why_ov = self._overhang_pick(manip, st_of, arms_p, g, yaw0, focus_xy, z_init, twins_b, ov_opts)
                            if overhang is None and key_ov is not None:
                                memo_ov[key_ov] = why_ov
                                self._overhang_memo_persisted(key_ov, why_ov)
                        if overhang is not None:
                            res, t_pose, eef, pre, pick, yaw = overhang["res"], overhang["t_pose"], overhang["eef"], overhang["pre"], overhang["pick"], yaw0
                            b_opts_m, why_m = self._b_margin_options(self._b_options(pick, g, eef, pre), overhang["grasp_stance"], overhang["arm"], "B, overhang")
                            if b_opts_m is None:
                                res, overhang = IK.IKResult(False, None, f"overhang grasp reach margin: {why_m}", 2), None
                                why_b.append(str(res.reason))
                        else:
                            why_b.append(f"overhang pick: {why_ov}")
                            log.info("%s: overhang pick fails for yaw bucket %.2f / grasp %s at B (%.2f, %.2f): %s", self.gp.variant, yaw0, getattr(g, "name", g),
                                     manip.translate, manip.tilt, why_ov[:900])
                    if res is None or not res.ok:
                        b_ik_cache[bk] = None
                        fails["B: [pre-grasp, grasp] exact IK"] = fails.get("B: [pre-grasp, grasp] exact IK", 0) + 1
                        log.info("%s: B exact IK fails for yaw bucket %.2f / grasp %s at B (%.2f, %.2f): %s", self.gp.variant, yaw0, getattr(g, "name", g), manip.translate, manip.tilt,
                                 " | ".join(why_b)[:600])
                        continue
                    arm_b = overhang["arm"] if overhang is not None else (pick.arm if pick is not None else a0)
                    b_ik_cache[bk] = {"yaw": yaw, "t_pose": t_pose, "arm": arm_b, "viable": O.viable_arms(pick.facts["orient"]) if pick is not None else [arm_b],
                                      "opts": {arm_b: b_opts_m}, "after": [a for a in held if a in seg.done], "overhang": overhang}
                hit = b_ik_cache[bk]
                if hit is None:
                    fails["B: [pre-grasp, grasp] exact IK (cached)"] = fails.get("B: [pre-grasp, grasp] exact IK (cached)", 0) + 1
                    continue
                role = seg.role(hit["arm"], hit["viable"], arm)
                if role == "fallback" and hit.get("overhang") is not None:
                    role = "none"  # an overhang pick is the pushing arm's own: no regrasp of the object where it lies by the other arm
                fallback = None
                if role == "defer":
                    deferred.add(bk)
                    fails[f"B: the {hit['arm']} arm's grasp (its coupling next)"] = fails.get(f"B: the {hit['arm']} arm's grasp (its coupling next)", 0) + 1
                    continue
                if role == "fallback":
                    # the segment fallback (owner 2026-09-27): the arm that won the grasp failed a later contact; this arm, viable at the
                    # grasp, regrasps the library row at the same target pose (once per pair)
                    if arm not in hit["opts"]:
                        hit["opts"][arm] = self._regrasp_at_b(g, hit["t_pose"], manip, arm, z_init, twins_b, seg.bands)
                    if hit["opts"][arm] is None:
                        role = "none"
                    else:
                        won_opts, won_facts = hit["opts"].get(hit["arm"]) or ([(None,)], None)
                        fallback = {"from_arm": hit["arm"], "from_grasp": getattr(won_opts[0][0], "name", None), "why": (seg.failed.get(bk) or seg.why.get(hit["arm"], ""))[:300],
                                    "arm_choice": (won_facts or {}).get("arm_choice")}
                elif hit.get("after"):
                    # drawn after the segment of an arm that held the pair failed at this B: that arm is not offered again
                    fallback = {"from_arm": hit["after"][0], "from_grasp": None, "why": seg.why.get(hit["after"][0], "")[:300]}
                if role == "none":
                    fails[f"B: no {arm}-arm grasp (the {hit['arm']} arm's)"] = fails.get(f"B: no {arm}-arm grasp (the {hit['arm']} arm's)", 0) + 1
                    continue
                yaw, t_pose = hit["yaw"], hit["t_pose"]
                b_opts, b_facts = hit["opts"][arm]
                # the stance the grasp is taken from: B, or for a two-stance overhang pick B backed off along its axis (the lift, the carry
                # pose and the B -> C leg leave from there)
                pick_st = (hit.get("overhang") or {}).get("grasp_stance") or manip
                lk = (bk, seg.leg)
                if lk not in seg.leg_ok:
                    seg.leg_ok[lk] = self._leg_ab_clear_of_target(init, manip, t_pose)
                    if seg.leg_ok[lk]:
                        # the A -> B drive was swept before the target had a pose (parked): a draw on the drive's corridor is dropped
                        fails["B: the A -> B drive passes the placed target"] = fails.get("B: the A -> B drive passes the placed target", 0) + 1
                        log.info("%s: %s", self.gp.variant, seg.leg_ok[lk])
                        continue
                elif seg.leg_ok[lk]:
                    fails["B: [pre-grasp, grasp] exact IK (cached)"] = fails.get("B: [pre-grasp, grasp] exact IK (cached)", 0) + 1
                    continue
                level_quat = G.qmul(G.quat_from_yaw(d_yaw), t_pose[1]) if abs(d_yaw) > 1e-9 else t_pose[1]
                if push and not self.push_yaw_kept(level_quat, place.extras["push_dir"]):  # the yaw drawn at B (jittered on a retry) off the lock's
                    why_l = f"C: push-under yaw lock {self.push_yaw_lock()}: the drawn yaw is not kept"
                    fails[why_l] = fails.get(why_l, 0) + 1
                    seg.failed.setdefault(bk, why_l)
                    continue
                # the release tilts the pair passed at F_goal in order (the level release alone without them); per tilt the recorded finger
                # assignment first, the other one (its B chain passed as well) only when the first fails here
                g_row = g
                c_opts = [(tilt, j, o) for tilt in (tilts_of.get(pair_key(yaw0, g_row, yaw_step)) or [0.0]) for j, o in enumerate(b_opts)]
                c_fails: List[str] = []
                lift_seeds: Dict[int, Optional[th.Tensor]] = {}
                for tilt, j_opt, (g, eef, pre, flip) in c_opts:
                    goal_quat = self.release_quat(level_quat, g_row, tilt)
                    goal = release_pose(release_xy, goal_quat, g.pose)
                    place_eef = IK.grasp_pose(goal, g.pose)
                    pre_place = self.pre_place_of(place_eef, place)
                    # the place / drop IK is seeded with the LIFT configuration (the arm the object is carried with, and the branch the
                    # carry pose's straight line stays in) and must land in the same branch: the oracle reaches the pre-place from the
                    # carry with a bounded joint swing (0920-135625: the recorded witness lay 2.09 rad away, 'plan_failed' at the drop)
                    if j_opt not in lift_seeds:
                        lift_seeds[j_opt] = self.lift_arm_q(pick_st, eef, float(self.num("carry_height_m", 0.12)), target, g.pose)
                    q_lift_seed = lift_seeds[j_opt]
                    q_seed = None
                    if q_lift_seed is not None:
                        q_seed = self.ikc.q_locked(place).clone()
                        q_seed[self.kit.robot.arm_control_idx[arm]] = q_lift_seed
                    pre_before = self._ik_solution_of(arm, pre_place) if n_place_diag < self.PLACE_IK_DIAG_PER_C else None
                    with attached(self.kit, arm, target, eef_in_object=g.pose):
                        r2 = self.ikc.exact([pre_place, place_eef], place, arm, ignore_for_last=ignore_place, cache=False, q_seed=q_seed)
                        if not r2.ok and n_place_diag < self.PLACE_IK_DIAG_PER_C:
                            n_place_diag += 1
                            log.info("%s: %s", self.gp.variant, self._place_ik_fail_line(r2, place, g, yaw, tilt, pre_place, place_eef, arm, q_seed, ignore_place,
                                                                                       pre_before, push))
                    if not r2.ok:
                        c_fails.append(f"C: [pre-place, {'drop' if push else 'place'}] exact IK (attached): {r2.reason[:200]}")
                        continue
                    if q_lift_seed is not None and r2.q is not None:
                        # a loose sanity bound only (2026-09-21 01:30): the branch the oracle reaches the pre-place in is decided by the straight
                        # carry leg it tracks from the carry configuration - checked below by ``track_chain`` once the carry leg is recorded
                        # (0921-005514/6/8: the 1.2 rad proxy against the LIFT configuration rejected 1-4 draws per C)
                        swing = float(th.max(th.abs(r2.q[self.kit.robot.arm_control_idx[arm]] - q_lift_seed)))
                        if swing > float(self.num("place_branch_max_rad", 2.6)):
                            c_fails.append("C: place branch differs from the carry (joint swing)")
                            continue
                    break
                if len(c_fails) == len(c_opts):
                    fails[c_fails[0]] = fails.get(c_fails[0], 0) + 1
                    seg.failed.setdefault(bk, c_fails[0])
                    continue
                twin_facts = b_facts
                if b_facts is not None and j_opt > 0:
                    twin_facts = dict(b_facts, chosen="twin" if flip else "row", late_flip=f"the {'row' if flip else 'twin'} failed at C: {c_fails[0]}")
                    log.info("%s: grasp twin [C] %s: the %s's place failed at C (%s), the %s's passes", self.gp.variant, g.name, "row" if flip else "twin", c_fails[0],
                             "twin" if flip else "row")
                if self.relation == "ontop" and not push:
                    # owner 2026-09-26 (shampoo, waves 6-8: 8 of 22 attempts rejected in goal_check 'place eef pose outside the placement image
                    # (5 %)' after 7-13 min of sampling each): the goal check's own frame test, applied to this C candidate's place now, so a C
                    # whose place the placement camera cannot see is skipped within the attempt instead of failing it at the end. The goal
                    # check tests the SETTLED goal, generation.goal_drop_m (1 cm) below the release pose (wave 9 0926-023738 still failed it
                    # with only the release pose tested here), so both poses are tested
                    cam_c = self.camera_pose_at(place)
                    rest = (goal[0] - th.tensor([0.0, 0.0, float(self.num("goal_drop_m", 0.01))]), goal[1])
                    if not (self.ikc.eef_in_frame(place_eef, cam_c) and self.ikc.eef_in_frame(IK.grasp_pose(rest, g.pose), cam_c)):
                        fails["C: place eef outside the placement image"] = fails.get("C: place eef outside the placement image", 0) + 1
                        seg.failed.setdefault(bk, "C: place eef outside the placement image")
                        continue
                chain = None
                if push:  # the straight floor push drop → goal from the same stance (closed gripper, floor + target ignored)
                    chain = self._push_under_chain(push_st if push_st is not None else place, goal_quat, cd, gc, place.extras["push_dir"])
                    if chain is None:
                        errors.append("C: push chain unreachable")
                        seg.failed.setdefault(bk, "C: push chain unreachable")
                        continue
                # B -> C with the object attached (§3.7 ≤ max_partner_geodesic + 1 m; §3.8 transition; owner 2026-09-19 21:56 (iii'):
                # straighten the torso first, navigate, set C's trunk at its pre-place stand-off). The grasp record goes in
                # BEFORE the leg: ``carried_attachment`` reads its ``eef_in_object`` to pose the object in the hand for every
                # attached check (without it the object was attached at its parked pose - the WARNING of 0919-202952)
                self._check_deadline(f"B -> C path / transition at C candidate {i_c + 1}")
                self.sidecar.add_stance(manip)
                if pick_st is not manip:
                    # the two-stance overhang pick: after the push's retreat the robot backs off to the grasp stance (an empty-handed transition)
                    self.sidecar.add_stance(pick_st)
                    # oracle-fix 2026-11-04 round 4: an earlier draw's B -> pick transition goes first (a draw failing after it - the carry leg, the
                    # B -> C path - drops only the placement / push records: hb13_desk_s3 reached the program check with 4 'manipulation' -> 'pick')
                    self._drop_leg_records(pick_st.name)
                    try:
                        self.record_transition(manip, pick_st)
                    except Reject as e:
                        errors.append(f"B -> pick (overhang grasp stance) transition: {e}")
                        seg.failed.setdefault(bk, f"B -> pick transition: {e.reason}")
                        self.sidecar.stances.pop(pick_st.name, None)
                        self._drop_leg_records(pick_st.name)
                        continue
                else:
                    self.sidecar.stances.pop(OH.PICK_STANCE, None)  # a grasp stance an earlier draw recorded
                    self._drop_leg_records(OH.PICK_STANCE)
                self.sidecar.add_stance(place)
                if push_st is not None:
                    # the deep push stance at the same base pose, entered with EMPTY hands after the drop: a trunk-only transition
                    self.sidecar.add_stance(push_st)
                    try:
                        with self._floor_relaxed():
                            self.record_transition(place, push_st)
                    except Reject as e:
                        errors.append(f"C: drop -> push trunk transition: {e}")
                        seg.failed.setdefault(bk, f"C: drop -> push trunk transition: {e.reason}")
                        self.sidecar.stances.pop("placement", None)
                        self.sidecar.stances.pop("push", None)
                        self._drop_leg_records("push")
                        break
                floor_name = self.bindings.one("floor").name if self.bindings.has("floor") else None
                grasp = GraspRecord(tname, arm, g.key, g.name, yaw, pre, eef, pre_place, place_eef, finger_q=list(g.finger_q), eef_in_object=g.pose,
                                    place_on=(floor_name or pname) if push else pname, stance=pick_st.name)
                if twin_facts is not None:
                    grasp.extras.update({"wrist_flip": bool(flip), "twin_choice": twin_facts})
                if tilts_of:
                    grasp.extras["release_tilt_deg"] = float(tilt)
                if len(offered) > 1 or arm != arm0:
                    grasp.extras["arm_bands"] = {a: TT.band_facts((self.surface.get("arm_bands") or {}).get(a)) for a in offered}
                if fallback is not None:
                    grasp.extras["segment_fallback"] = fallback
                if push:  # the release is a drop on the floor outside the edge (like the inside rule's ``place_kind: drop``)
                    grasp.extras.update({"place_kind": "drop", "drop_height_m": float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M)),
                                         "drop_cell": [float(v) for v in cd], "goal_cell": [float(v) for v in gc], "then": "push_under"})
                self.sidecar.add_grasp(grasp)
                # carry-height ladder (0920-014945: with the hand frozen at a 12 cm lift over the table, the arrival sweep into the
                # push-under crouch at C dragged the held sandal through the table top and into the floor at every s_T) - the lift
                # is raised until the departure sweep, the carry and the arrival sweep are free; the oracle lifts by the recorded height
                path_bc, carry_errors = None, []
                # oracle-fix 2026-11-05 (with generation.carry_track_oracle): the carry leg at C is tracked per carry option, so a carry
                # configuration whose branch cannot reach the pre-place (hb16_desk_s1 / hb18_desk_s1: stalled at 9-10 of 45, finish 1.12-1.21
                # rad away) gives way to the next option instead of the whole C candidate
                track_in_loop, carry_c, track_fails = bool(self.gp.gen("carry_track_oracle") or False), None, 0
                heights = [float(v) for v in (self.gp.gen("carry_heights_m") or (grasp.carry_height, 0.25))]
                lift_extra = self.carry_lift_extra()
                if lift_extra > 0:  # oracle-fix 2026-11-04 (owner item 4, opt-in generation.carry_lift_by_extent_m): each lift + the object's height + margin
                    heights = [h + lift_extra for h in heights]
                for h in heights:
                    q_lift = self.lift_arm_q(pick_st, eef, h, target, g.pose)
                    if q_lift is None:
                        carry_errors.append(f"lift {h:.2f} m: no IK solution from B")
                        continue
                    grasp.carry_height = h
                    # carry pose candidates (beside the body first), the plain lift configuration last
                    options: List[Tuple[str, Optional[Tuple[G.Pose, List[G.Pose], th.Tensor]]]] = []
                    self._carry_centred_n = 0
                    if str(self.gp.gen("carry_pose", "on")) != "off":
                        options += [(f"carry pose {[round(float(v), 2) for v in P0.to_base_frame(cp[0][0], manip.base_pose2d, P0.base_z(self)).tolist()]}", cp)
                                    for cp in self.carry_poses_at_b(pick_st, eef, h, target, g.pose)]
                    n_centred = int(getattr(self, "_carry_centred_n", 0) or 0)  # the first n options are centred carry poses (CARRY_CENTRED_ENV)
                    options = [(("centred " + lb) if i < n_centred else lb, cp) for i, (lb, cp) in enumerate(options)]
                    options.append(("lift configuration", None))
                    for label, cp in options:
                        if cp is not None:
                            carry_pose, wps, q_carry = cp
                            grasp.waypoints = wps  # the in-hand line lift -> carry pose the program's Reorient tracks
                            grasp.extras["carry_pose"] = {"position": [float(v) for v in carry_pose[0]], "orientation": [float(v) for v in carry_pose[1]]}
                            q_arm = q_carry  # the legs are checked with the arm at the carry pose (where the oracle holds it)
                        else:
                            grasp.waypoints = None
                            grasp.extras.pop("carry_pose", None)
                            q_arm = q_lift
                        try:
                            with self._floor_relaxed():  # the floor mesh out (deep-crouch 1 cm margin, as the stance / program checks); the ground slab stays
                                path_bc = self.record_carry_leg(init, pick_st, place, tname, q_arm, max_len)
                        except Reject as e:
                            carry_errors.append(f"lift {h:.2f} m, {label}: {e.reason}: {str(e)[:700]}")
                            self._drop_leg_records("placement")
                            self.record_carry_arm_q(tname, arm, None)
                            continue
                        if track_in_loop:
                            carry_c = self._carry_track_at_c(place, arm, tname, target, g, pre_place, place_eef, ignore_place)
                            if carry_c[3] != "ok":
                                self.carry_compact_probe(pick_st, place, arm, tname, target, g, eef, pre_place, place_eef, ignore_place, h)  # opt-in diagnostic
                                carry_errors.append(f"lift {h:.2f} m, {label}: carry leg at C {carry_c[3]}: {str(carry_c[2])[:400]}")
                                track_fails += 1
                                path_bc, carry_c = None, None
                                self._drop_leg_records("placement")
                                self.record_carry_arm_q(tname, arm, None)
                                continue
                        log.info("%s: carry with lift %.2f m at the %s%s", self.gp.variant, h, label, f" (after: {'; '.join(carry_errors)})" if carry_errors else "")
                        if label.startswith("centred "):
                            TT._count(self, "carry_pose.centred_used")
                        break
                    if path_bc is not None:
                        break
                if path_bc is None:
                    errors.append("C: carry leg: " + "; ".join(carry_errors[-3:]))
                    seg.failed.setdefault(bk, "C: carry leg: " + (carry_errors[0][:300] if carry_errors else "no carry option"))
                    log.info("%s: C (d %.2f λ %+.2f) carry leg failed: %s", self.gp.variant, place.stand_off or 0.0, place.lateral, " || ".join(carry_errors)[:1200])
                    fails["B -> C path / transition"] = fails.get("B -> C path / transition", 0) + 1
                    if track_fails:
                        fails["C: carry leg untrackable (per option)"] = fails.get("C: carry leg untrackable (per option)", 0) + track_fails
                    else:  # a failed tracking depends on C's trunk: the trunk variants at this base pose stay candidates
                        carry_failed_bases.add(base_key)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    break
                # the oracle's carry leg at C is a straight Cartesian line from where the hand is (the carry configuration after the
                # arrival) to the pre-place, tracked by differential IK and finished by ``arm_to`` to the recorded witness within 0.6
                # rad (0920-135625: a witness 2.09 rad away = 'plan_failed'). Track that line here from the carry configuration at C;
                # its end configuration is the branch the witnesses must be in - the drop IK is re-solved from it and must stay within
                # ``carry_finish_max_rad`` (0.5) of it; ``carry_end_q`` seeds the program check's place witness the same way
                idx_r = self.kit.robot.arm_control_idx[arm]
                q_end, fin_swing, reason, kind_c, via_c = carry_c if carry_c is not None else self._carry_track_at_c(place, arm, tname, target, g, pre_place, place_eef, ignore_place)
                if kind_c == "untrackable":
                    errors.append(f"C: carry leg untrackable to the pre-place: {reason}")
                    seg.failed.setdefault(bk, f"C: carry leg untrackable to the pre-place: {str(reason)[:200]}")
                    log.info("%s: C (d %.2f λ %+.2f) carry leg hand -> pre-place untrackable / drop IK from its end fails: %s", self.gp.variant, place.stand_off or 0.0, place.lateral, reason)
                    fails["C: carry leg untrackable"] = fails.get("C: carry leg untrackable", 0) + 1
                    self._drop_leg_records("placement")
                    self.record_carry_arm_q(tname, arm, None)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    continue
                if kind_c == "branch":
                    errors.append(f"C: pre-place witness {fin_swing:.2f} rad from the tracked carry end")
                    seg.failed.setdefault(bk, f"C: pre-place witness {fin_swing:.2f} rad from the tracked carry end")
                    log.info("%s: C (d %.2f λ %+.2f) pre-place IK from the tracked carry end swings %.2f rad (> 0.5)", self.gp.variant, place.stand_off or 0.0, place.lateral, fin_swing)
                    fails["C: pre-place branch off the tracked carry"] = fails.get("C: pre-place branch off the tracked carry", 0) + 1
                    self._drop_leg_records("placement")
                    self.record_carry_arm_q(tname, arm, None)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    continue
                self.carry_end_q[tname] = q_end[idx_r].clone()
                if via_c is not None:  # the oracle's carry leg passes through it (grasps[k].carry, Place.carry_waypoints)
                    grasp.carry = [via_c["pose"]]
                    grasp.extras["carry_via_at_c"] = via_c["label"]
                log.info("%s: carry leg hand -> pre-place tracked at C (%.2f rad swing at the finish)", self.gp.variant, fin_swing)
                if push:
                    chain = {**chain, "record": {**chain.get("record", {}),
                                                 **self.push_strokes_record(chain["distance_m"], chain.get("record", {}).get("push_contact_form"))}}
                    self.sidecar.add_contact(PushUnderRecord(arm, chain["pre"], chain["start"], chain["mid"], chain["goal"], chain["direction"], chain["distance_m"],
                                                             tname, stance="push" if push_st is not None else "placement", pitch_deg=chain["pitch_deg"], blade_standoff_m=chain["blade_standoff_m"],
                                                             push_height_m=chain["push_height_m"], half_extent_m=chain["half_extent_m"], fingertip_z_m=chain["fingertip_z_m"],
                                                             ground_ignored=bool(chain.get("ground_ignored", False)),
                                                             drop_cell=[float(v) for v in cd], goal_cell=[float(v) for v in gc], partner=pname,
                                                             **({"orient": chain["orient"]} if chain.get("orient") else {}), **push_under_palm_flags(chain),
                                                             **chain.get("record", {}), **chain.get("seed", {})))
                if hit.get("overhang") is not None:
                    self.overhang_records(hit["overhang"], grasp)
                cam = self.camera_pose_at(init)
                if not self.frustum_ok(cam, self.hull_corners_world(self.hull, t_pose), 0.05):
                    raise Reject("not_in_frame", "target not in the init image (5 %)")
                path_ab = next((p for p in self.sidecar.paths if p.to == "manipulation"), None)
                place.extras.update({"init_surface": {"class": self.surface["kind"], "name": None if self.surface["kind"] == "floor" else self.surface["obj"].name},
                                     "path_length_m": {"A_B": None if path_ab is None else path_ab.length_m, "B_C": path_bc.length_m}})
                d = Draft()
                d.object_poses[tname] = t_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"goal_cell": gc, "grasp": {"key": g.key, "name": g.name, "yaw": yaw}, "placement_stance": P0.summarize_stance(place),
                                  "path_length_m": place.extras["path_length_m"], "F_init": len(F_init), "F_goal": len(F_goal), "F": len(F),
                                  # the carried object's yaw follows the base: goal orientation = init orientation rotated by C.yaw - B.yaw
                                  "goal_quat": [float(v) for v in goal_quat], "carry_yaw_offset_rad": round(d_yaw, 4)})
                if tilts_of:
                    d.summary.update({"release_tilt_deg": float(tilt), "release_level_quat": [float(v) for v in level_quat]})
                    log.info("%s: inside release tilt %g deg coupled (yaw %.2f, grasp %s, C d %.2f λ %+.2f)", self.gp.variant, tilt, yaw, g.name, place.stand_off or 0.0, place.lateral)
                d.summary["arm"] = TT.segment_arm_summary(grasp, offered)
                if hit.get("overhang") is not None:
                    d.summary["pick_overhang"] = _jsonable(hit["overhang"]["facts"])
                band_c = None if push else (self.place.get("bands") or {}).get(arm)  # the partner-top band of the place is the segment arm's too
                if band_c is not None and band_c.meta.get("mirrored_from"):
                    d.summary["arm"]["mirrored_band"] = True
                arms_here = (seg.arms_of(yaw0, g) if seg is not None else []) or [arm]
                grasp.extras["arm_offer"] = TT.arm_offer_record(offered, [a for a in offered if a in arms_here] or [arm], grasp, TT.arm_offer(self)[1])
                if arm != arm0 or fallback is not None:
                    log.info("%s: segment grasp -> carry -> %s with the %s arm (%s)%s", self.gp.variant, "drop -> push" if push else "place", arm, d.summary["arm"]["reason"],
                             f"; the {fallback['from_arm']} arm failed: {fallback['why'][:200]}" if fallback is not None else "")
                if push:
                    d.summary["push_under"] = {"drop_cell": [float(v) for v in cd], "distance_m": chain["distance_m"], "pitch_deg": chain["pitch_deg"],
                                               "blade_standoff_m": chain["blade_standoff_m"], "push_height_m": chain["push_height_m"], "half_extent_m": chain["half_extent_m"],
                                               "drop_ahead_m": place.extras.get("drop_ahead_m"), **chain.get("record", {})}
                # §5.6: pose targets are measured for the TARGET only — the partner is fixed furniture and the large-scale
                # iddl's (:measure) has (pose_reached ?target) alone, so a TRO pose target for the partner fails
                # ``validate_tro_against_problem`` at write time (sandal-breakfast_table 0919-202952: the first large-scale
                # IORS draft to reach the writer). The sampler's own "partner unchanged <= 5 mm at the goal" check
                # (:meth:`refine_targets_from_goal`) reads the position recorded here instead.
                pp, _ = partner.get_position_orientation()
                d.summary["partner_position"] = [float(v) for v in pp]
                d.pose_targets[tname] = full_pose_target(t_pose[0], t_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, partner.category.replace("_", " "), path_length_m=path_bc.length_m)
                if push and floor_name:
                    self.sidecar.describe(floor_name, "floor", push_distance_cm=100.0 * float(chain["distance_m"]))
                if trunk_first:
                    extended = float(place.translate) > least_translate + 1e-6
                    d.summary["c_order"] = {"mode": "trunk_first", "rank": i_c, "translate": float(place.translate), "stand_off": place.stand_off,
                                            "least_translate": least_translate, "extended": extended}
                    if extended:
                        TT._count(self, "stance_c.extended_used")
                        log.info("%s: stance C trunk-first order: trunk (%.2f, %.2f) d %.2f coupled at rank %d, after the less extended candidates (least "
                                 "translate %.2f) failed", self.gp.variant, place.translate, place.tilt, place.stand_off or 0.0, i_c, least_translate)
                return d
            errors.append(f"C (d {place.stand_off:.2f} trunk ({place.translate}, {place.tilt:.1f})): no (yaw, grasp) of F ({len(F)}) passes the exact IK at B and C — {fails}")
            log.info("%s: C try exhausted (d %.2f λ %+.2f edge %s, push trunk %s): %s", self.gp.variant, place.stand_off or 0.0, place.lateral, place.extras.get("edge"),
                     None if push_st is None else (push_st.translate, round(push_st.tilt, 2)), fails)
        raise Reject("grasp_unreachable", "; ".join(errors[-3:]) if errors else "no placement candidate")

    # ------------------------------------------------------------------ push-under: the floor mesh and the deep trunks

    def _floor_relaxed(self):
        """Context: the floor object out of the collision world (the cuRobo ground slab stays) — the push-under stance C
        stands at the deep trunks, where the 1 cm margin against the floor mesh is what fails (see
        :meth:`push_under_candidates`); applied to the core's stance / transition re-check and to the program check. Every under
        variant: the place-under (``under_mode`` != "push") sets the object down on the floor from the floor band's crouch as well
        (owner 2026-10-12 item 2c; before, only a floor-class init took the floor out for it)."""
        floor_class = (getattr(self, "surface", None) or {}).get("kind") == "floor"
        return self.ikc.ignoring([self.obj("floor")]) if (self.relation == "under" or floor_class) else _NullContext()

    def check_stances_with_objects(self, draft: Draft):
        # oracle-fix 2026-11-04 round 3 (chair 155): with the opt-in generation.push_end_ignore_target (a base push: the robot ends where the
        # target stood) the stances are checked with the target out - the objects stand at their INIT poses here (11 x
        # stance_collision_with_objects ['placement'] in chair7_lobby_s0); the push stance itself was checked with the target placed
        with self._floor_relaxed():
            super().check_stances_with_objects(draft)

    def stance_check_ignores(self, stance) -> List[Any]:
        """With ``generation.push_end_ignore_target``: the target, for every stance but init (the robot reaches them after the target left its init
        pose: the push end where the chair stood, the push-under crouch next to the briefcase's start - brief10 blame: base_link x lwbctq)."""
        if self.gp.gen("push_end_ignore_target") and getattr(stance, "name", "") != "init":
            return [self.target]
        return []

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        with self._floor_relaxed():
            return super().program_check(draft)

    # ------------------------------------------------------------------ goal (§5.6 / §6.12)

    def goal_quat_of(self, draft: Draft) -> th.Tensor:
        """The target's goal orientation: the init orientation rotated by the base-yaw change B -> C (``summary.goal_quat``,
        relative carry yaw), the init orientation itself under ``carry_yaw = "world"``."""
        gq = draft.summary.get("goal_quat")
        return th.tensor([float(v) for v in gq]) if gq is not None else draft.object_poses[self.tname][1]

    def goal(self, draft: Draft):
        tname = self.tname
        goal = self.goal_pose_at(draft.summary["goal_cell"], self.goal_quat_of(draft), self.release_eef_in_object())
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        end = self.sidecar.stances.get("push") or self.sidecar.stance("placement")  # where the robot ends: the push stance when the drop and the push split
        self.sidecar.set_goal(end.base_pose2d, end.translate, end.tilt, arms="same")

    def after_settle(self, draft: Draft):
        tname = self.tname
        goal = self.goal_pose_at(draft.summary["goal_cell"], self.goal_quat_of(draft), self.release_eef_in_object())
        draft.goal_poses[tname] = goal
        tgt = draft.pose_targets[tname]
        tgt["position"], tgt["orientation"] = [float(v) for v in goal[0]], [float(v) for v in goal[1]]
        self.record_partner_joints(draft)

    def refine_targets_from_goal(self, draft: Draft):
        super().refine_targets_from_goal(draft)
        p, _ = self.partner.get_position_orientation()
        ref = draft.summary.get("partner_position")
        if ref is None:
            return
        moved = float(th.linalg.norm(p.float() - th.tensor(ref, dtype=th.float32)))
        if moved > self.num("partner_moved_max_m", 0.005):
            raise Reject("goal_partner_moved", f"partner moved {moved:.3f} m at the goal")

    def after_goal_check(self, draft: Draft):
        """Place / pre-place at the settled goal pose re-checked from C with the object attached (the record kept), with the recorded
        grasp's / push's arm (the segment arm)."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place = self.sidecar.stance("placement")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"])
        # the release's last-pose ignores as screened at C (oracle-fix 2026-11-04 round 4: the lean also leaves the partner it leans on out - the re-check
        # with the floor alone met hall_tree_upkvgr_0 at the release on umb5_upk_up_s0 / umb5_upk_flat_s0 / _s1)
        ignore = self.place_ignores()
        if self.push_under:
            # the release is the drop outside the edge (unchanged by the settle): re-check it as recorded; the push goal blade
            # follows the settled goal centre (a few mm), the chain start → mid → goal is re-solved (closed gripper)
            with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([grasp.pre_place, grasp.place], place, grasp.arm, ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / drop fail the exact IK from stance C at the goal check: {res.reason}")
            push = self.sidecar.contacts.get("push")
            if push is not None:
                d3 = th.as_tensor(push.direction, dtype=th.float32)
                off = float(push.extras["half_extent_m"]) + float(push.extras["blade_standoff_m"]) - float(self.num("push_goal_extra_m", 0.03))
                z_push = float(push.contact[0][2])
                goal_b = th.tensor([float(goal_pos[0]), float(goal_pos[1]), z_push]) - d3 * off
                start_b = push.contact[0]
                mid_b = 0.5 * (start_b + goal_b)
                q = push.contact[1]
                push_from = self.sidecar.stances.get("push") or place  # the chain is pushed from the deep push stance when the drop and the push split
                with self.ikc.ignoring([self.obj("floor"), self.target], ground=not bool(push.extras.get("ground_ignored", False))):
                    r3, seed_why = self._push_under_goal_chain(push, [push.pre, (start_b, q), (mid_b, q), (goal_b, q)], push_from)
                if not r3.ok:
                    raise Reject("push_unreachable", f"push-under chain to the settled goal fails the exact IK from stance C: {r3.reason}"
                                                     + (f" (seeded from the sampled chain's pre-pose configuration: {seed_why})" if seed_why else ""))
                push.mid, push.goal = (mid_b, q), (goal_b, q)
                push.distance_m = float(th.linalg.norm(goal_pos[:2].float() - th.as_tensor(push.extras["drop_cell"], dtype=th.float32)))
                push.extras["goal_cell"] = [float(goal_pos[0]), float(goal_pos[1])]
        else:
            place_eef = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
            pre_place = self.pre_place_of(place_eef, place)
            with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([pre_place, place_eef], place, grasp.arm, ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / place at the settled goal fail the exact IK from stance C: {res.reason}")
            grasp.pre_place, grasp.place = pre_place, place_eef
        st_b = self.sidecar.stance(getattr(grasp, "stance", None) or "manipulation")  # B, or the overhang pick's grasp stance
        cam_b = self.camera_pose_at(st_b)
        if str(self.gp.gen("grasp_frame_rule") or "box") == "fingertip":
            # oracle-fix 2026-11-04 round 3 (under_large#02, opt-in): the grasp judged by QA's own criterion - the fingertip point 5 % inside the
            # image (program_check.fingertips_in_frame) - instead of the +-6 cm eef box (brief6_pitched_s0: 5 / 5 attempts past the push re-solve
            # rejected here, the box bottom at v 1000-1046 of the 1026 limit, the fingertip at v 930-985)
            from ispatialgym.sampling.v2 import program_check as PCK
            out = PCK.fingertips_in_frame(self, st_b, "grasp_frame", [("grasp_frame", [grasp.grasp])], slack_m=0.0)
            if out is not None:
                raise Reject("gripper_not_in_frame", f"grasp fingertip outside the manipulation image (5 %): {out}")
        elif not self.ikc.eef_in_frame(grasp.grasp, cam_b):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if self.relation == "ontop" and not self.ikc.eef_in_frame(place_eef, self.camera_pose_at(place)):
            raise Reject("gripper_not_in_frame", "place eef pose outside the placement image (5 %)")

    def _push_under_goal_chain(self, push: ContactRecord, chain: List[G.Pose], stance: Stance) -> Tuple[Any, Optional[str]]:
        """The goal check's exact IK of the push-under chain ``[pre, start, mid, goal]`` (the goal at the settled goal) from ``stance``, in the
        caller's ignore context (owner 2026-10-13 ~01:05 item 1). First the chain leaves from the sampled chain's pre-pose configuration (the
        record's ``chain_arm_q``, :func:`push_chain_seed`; ``exact``'s ``q_first``) when that is still a witness of the pre-pose (reaches it,
        collision-free here): the whole chain is tracked from it, so it stays in the sampled wrist branch where a fresh solve lands in the
        default one (under#02 pitched_c2_k0, 3/3 attempts: the left wrist into the base just after the pre-pose, where the sampler had taken
        the wrist-flipped branch). When that fails, the unseeded solve with the IK memory as it was before. Returns ``(result, why the seeded
        solve failed or None)``; one INFO line 'push-under goal check re-solve: ...'."""
        arm = push.arm
        fingers = push_record_fingers(push.extras)
        arm_q = (push.extras.get("chain_arm_q") or [None])[0]
        if arm_q is None:
            why = "no seed (the record has no chain_arm_q)"
        else:
            q_first = self.ikc.q_locked(stance, fingers=fingers).clone()
            q_first[self.kit.robot.arm_control_idx[arm]] = th.as_tensor(arm_q, dtype=q_first.dtype)
            if not self.ikc.collision_free_at(q_first, chain[0], arm):
                why = "the sampled pre-pose configuration is off the pre-pose or collides here"
            else:
                saved = dict(self.ikc.solutions)
                res = self.ikc.exact(chain, stance, arm, cache=False, fingers=fingers, q_first=q_first)
                if res.ok:
                    branch = {"map": "wrist-flipped", "ik": "solved"}.get(push.extras.get("chain_branch"), str(push.extras.get("chain_branch")))
                    log.info("%s: push-under goal check re-solve: seeded ok (the sampled chain's %s branch, %s arm)", self.gp.variant, branch, arm)
                    return res, None
                self.ikc.solutions.clear()
                self.ikc.solutions.update(saved)
                why = str(res.reason)
        res = self.ikc.exact(chain, stance, arm, cache=False, fingers=fingers)
        log.info("%s: push-under goal check re-solve: %s -> unseeded %s", self.gp.variant, why if arm_q is None else f"seeded failed ({why[:300]})",
                 "ok" if res.ok else "failed")
        return res, None if arm_q is None else why

    def turn_recheck(self, draft: Draft, grasp, q_first: Optional[th.Tensor] = None) -> Optional[str]:
        """``tabletop.held_turns``: not offered here. The held chain runs through state recorded for the recorded turn alone - the carry
        pose (``grasps[].waypoints``, ``carry_arm_q``), the B -> C departure transition / path / arrival swept with the arm there, and
        the carry leg at C tracked from it to the pre-place (``carry_end_q``, which seeds the program check's place) - so the other turn
        means re-planning the carry from the carry-pose search on."""
        raise TT.TurnNotOffered("the held chain runs through the carry pose, the B -> C carried travel swept at the recorded carry configuration "
                                "(carry_arm_q) and the carry leg at C tracked from it (carry_end_q): the other turn needs that carry re-planned")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        g = self.sidecar.grasp_of(self.tname)
        place = self.sidecar.stances.get("placement")
        if g is None or place is None:
            return []
        c = self.choice.models["target"]
        return [{"scope": {"scene": self.scene, "partner": self.partner.name, "target": f"{c.category}/{c.model}"}, "type": "placement",
                 "edge": place.edge.name if place.edge else None, "stand_off": place.stand_off, "lateral": place.lateral, "trunk": [place.translate, round(place.tilt, 3)],
                 "goal_cell": draft.summary.get("goal_cell"), "grasp": [round(g.yaw, 2), g.grasp_name], "hard": False}]

    # ------------------------------------------------------------------ inside / touching / nextto (registered 2026-10-12, modes next)

    def _adopt_mode(self, mode: type) -> None:
        """The relation's mode takes over: ``mode`` (the class its hook returns, from the mode's own module, written like an IORSLarge
        subclass or mixin; not ``@register``-ed) goes in front of this sampler's class, so every method it defines (``bind_scene_roles``,
        ``prepare_bindings``, ``sample_template``, ``sample_instance``, ``goal``, ``after_goal_check``, ... or a helper of the ontop / under
        flow) replaces IORSLarge's, which ``super()`` still reaches."""
        self.relation_mode = mode
        self.__class__ = type(f"{type(self).__name__}_{mode.__name__}", (mode, type(self)), {})

    def _sample_inside(self) -> type:
        """``inside`` (``...-inside_large_scale``): the hook of the inside mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method. Until the mode lands every template of the subtask is rejected with
        ``relation_not_implemented``."""
        from ispatialgym.sampling.v2.tasks import iors_large_inside as M
        return M.InsideMode

    def _sample_touching(self) -> type:
        """``touching`` (``...-touching_large_scale``): the hook of the touching mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method. Until the mode lands every template of the subtask is rejected with
        ``relation_not_implemented``."""
        gp = getattr(self, "gp", None)
        if gp is not None and str(gp.gen("touching_push") or "").lower() == "base":
            # oracle-fix 2026-11-03 (touching_large#01, the ottoman): a floor-standing target the base pushes against the partner's face
            # (no grasp, no drop) - the nextto mode's base push with a touching goal
            from ispatialgym.sampling.v2.tasks import iors_large_nextto as NM

            return NM.TouchingBasePushMode
        from ispatialgym.sampling.v2.tasks import iors_large_touching as M
        if gp is not None and str(gp.gen("touching_place") or "").lower() == M.LEAN_PLACE_KIND:
            # oracle-fix 2026-11-04 round 3 (touching_large#02, the umbrella): placed leaning on the partner, its lower end on the floor
            return M.LeanTouchingMode
        if gp is not None and str(gp.gen("touching_place") or "").lower() == M.SURFACE_PLACE_KIND:
            # oracle-fix 2026-11-06 (owner feedback on the touching_large#02 demo: "the largest compartment (the top one) has a surface that the
            # umbrella can lay flat on"): set down on an interior board of the partner - the inside mode's raster, stance C and release
            return M.SurfaceTouchingMode
        return M.TouchingMode

    def _sample_nextto(self) -> type:
        """``nextto`` (``...-nextto_large_scale``): the hook of the nextto mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method: ``iors_large_nextto.NextToMode``, the floor-standing target pushed next
        to the partner by the base."""
        from ispatialgym.sampling.v2.tasks import iors_large_nextto as M

        return M.NextToMode
