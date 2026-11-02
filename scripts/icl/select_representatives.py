#!/usr/bin/env python3
"""Select the representative object models (+ variation + held-out) per subtask from the icl.

Owner request (2026-09-15): ~30 representative picks per task family, >= 5 per subtask (block design excepted), with

* diverse categories (one category per pick within a subtask; categories re-used across subtasks only when the pool
  is short), few overlapping *models* across the whole selection;
* every pick pinned with a **representative** model (the one ground_isg / sample_isg / the oracle run first), a
  **variation** model (same category, different appearance: a texture variant of the representative where one exists
  - V-OBJ "model change (texture variant)" - and a different geometry - VB-MOBJ "model change"), and a **held-out**
  model (generalization; VB-MOBJ / VSB-NOBJ test only). Held-out is **global**: one held-out model per category, the
  same in every subtask, and it is never used in any other role (partner, distractor, obstacle) either;
* scene roles (supports, big partners, scene articulated targets, wall regions) get the same three slots where the
  scene inventory allows (VB-ISC "support height / model change", VB-MOBJ "partner / target model change").

Owner decisions 2026-09-16 (TODO.md, A3 / A5):

* **A3 support triplets**: every §4 tabletop pick (the 11 subtasks whose objects rest on a scene support) carries N
  support triplets (``--support-triplets``, default 3) in its ``support`` role: T1 = the 2026-09-15 triplet (test pin
  ``breakfast_table_rhjoby_0`` / same-scene height variation / other-scene held-out), T2 / T3 = the owner's pins
  (:data:`SUPPORT_TRIPLETS`), each member verified against the B4 inventory (exists, top height in the tabletop range,
  usable edge >= the longest §4.4 margin row the pilot measured). Held-out supports are global (held out in every
  tabletop subtask). Rows keep the slot names ``representative`` / ``variation_geometry`` / ``heldout`` and carry the
  triplet index in the ``triplet`` column; within a pick the rows are written T2, T3, ..., T1 so that readers indexing
  (pick, role, slot) without the column keep the T1 binding (backward compatibility); T1 rows are unchanged.
* **Edge-length-vs-margin check** (:meth:`Selector.support_admission`): per support and per edge the usable length =
  min(edge length, free-floor run at stand-off 0.45 with the chairs removed), 0 when an object stands closer than
  :data:`BLOCKED_IN_FRONT_MAX_M` in front of the edge (the breakfast table's window side) or the run is below the 0.60 m
  admission; the longest §4.4 margin row (:data:`MARGIN_ROWS`, m_R + m_L) the best usable edge admits is recorded per
  support and supports that admit only the short rows are flagged (``short_rows_only``). The report has the table.
* **A5 partner's own top** (spec §5.6 b, IORS under large-scale): when no admitted support lies within 3 m of the
  representative partner, the init surface is the partner instance itself if its top admits the A / B stances per the
  B4 inventory (an admitted support whose best usable edge admits the large-scale init row for the pick's targets);
  the floor stays the last class. ``--patch-only under_large_scale`` applies the branch to the existing
  ``representatives.csv`` without re-running the selection (a re-run on a rebuilt icl shifts other picks);
  ``--patch-only support_triplets`` appends the T2..TN rows the same way. Both patch the report in place.

Inputs: ``ispatialgym/assets/icl/{roles,require}/*.csv`` (built by ``build_icl.py``), the duplicate / texture-variant
reviews under ``runlogs/*/duplicates*.json``, the B4 scene inventory ``runlogs/scene_inventory_20260915/summary.json``
(+ ``<scene>/supports/<name>.json`` for the per-edge runs, blocked objects and rooms).

Outputs (``ispatialgym/assets/icl/roles/``):

* ``representatives.csv`` - one row per (pick, role, slot): ``family, subtask, pick, role, slot, category, model,
  scene, name, joint, kind, grasp_key, scale, geometry_group, exact_group, status, note, triplet``; slots are
  ``representative``, ``variation_texture``, ``variation_geometry``, ``heldout`` (heldout rows are the global per-category
  choice); ``triplet`` is set on the support rows of the tabletop picks only (1 = T1, 2 = T2, ...);
* ``candidates_extra.csv`` - candidate rows the icl did not have (same columns as ``candidates.csv`` + ``plan``),
  ``status = incomplete:<reason>``; ``build_icl.py`` / ``ground_isg`` merge them until a survey validates them (support
  rows are added only for selected supports that ``candidates.csv`` lacks - none today);
* ``representatives_report.md`` - per subtask: the picks, which variation / generalization axes are unavailable and why;
  the support triplets and the per-support admission table.

Run: ``python scripts/icl/select_representatives.py [--out ispatialgym/assets/icl/roles] [--per-family 30]
[--support-triplets 3]`` (full selection) or ``... --patch-only under_large_scale,support_triplets`` (targeted patch of
the files under ``--out``). Deterministic (no randomness): greedy by a documented score, ties by icl rank / row order.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ICL = os.path.join(REPO, "ispatialgym", "assets", "icl")
SELECT = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
SCENE_INVENTORY = os.path.join(REPO, "runlogs", "scene_inventory_20260915", "summary.json")
DUP_FILES = [
    os.path.join(SELECT, "duplicates", "verdicts.json"),
    os.path.join(REPO, "runlogs", "world_alignment_selection_20260911", "duplicates.json"),
    os.path.join(REPO, "runlogs", "world_horizontal_selection_20260912", "duplicates.json"),
    os.path.join(REPO, "runlogs", "obstacle_selection_20260912", "duplicates.json"),
]

IORS = "inter_object_relation_satisfaction"
CM = "configuration_matching"
TEST_SCENE = "house_single_floor"
TEST_ROOM = "living_room_0"
TEST_SUPPORT = "breakfast_table_rhjoby_0"

#: subtask -> (family, {role: spec}); spec: kind (sampleable/group/scene), pipeline (grasps), extra filters
SUBTASKS: Dict[str, dict] = {
    "camera_relocalization-standard": {"family": "camera_relocalization", "roles": {}, "n": 0,
                                       "note": "no object role: variation comes from the room (VSB-NOBJ n/a by design)"},
    "camera_relocalization-obstructed": {"family": "camera_relocalization", "roles": {"obstacles": {"kind": "group", "natural": TEST_SCENE}}},
    "object_scaling": {"family": "object_scaling", "roles": {"target": {"kind": "sampleable", "pipeline": "floating"}}},
    "mental_rotation": {"family": "mental_rotation", "roles": {"target": {"kind": "sampleable", "pipeline": "floating", "asymmetric": True}}},
    "disembedding": {"family": "disembedding", "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop", "pools": True},
                                                          "support": {"kind": "scene"}}},
    f"{CM}-block_design": {"family": CM, "roles": {"targets": {"kind": "group"}, "support": {"kind": "scene"}}, "n": 1},
    f"{CM}-articulation_open_small_scale": {"family": CM, "roles": {"target": {"kind": "sampleable", "handle": True}, "support": {"kind": "scene"}}},
    f"{CM}-articulation_close_small_scale": {"family": CM, "roles": {"target": {"kind": "sampleable", "handle": True}, "support": {"kind": "scene"}}},
    f"{CM}-articulation_open_large_scale": {"family": CM, "roles": {"target": {"kind": "scene_artic"}}},
    f"{CM}-articulation_close_large_scale": {"family": CM, "roles": {"target": {"kind": "scene_artic"}}},
    f"{CM}-attachment": {"family": CM, "roles": {"target": {"kind": "sampleable", "pipeline": "attachment", "partner_model": True},
                                                  "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    "alignment-world_vertical": {"family": "alignment", "roles": {"target": {"kind": "sampleable", "frame": "wall"}, "support": {"kind": "wall"}}},
    "alignment-axial": {"family": "alignment", "roles": {"target": {"kind": "sampleable", "frame": "axial"}, "support": {"kind": "scene"}}},
    f"{IORS}-ontop_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-under_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-inside_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "insert"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-nextto_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-touching_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-ontop_large_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "scene_partner"}, "support": {"kind": "scene"}}},
    f"{IORS}-under_large_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "scene_partner"}, "support": {"kind": "scene"}}},
}
FAMILY_ORDER = ["camera_relocalization", "object_scaling", "mental_rotation", "disembedding", CM, "alignment", IORS]
ONTOP_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table", "countertop")
UNDER_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table")
#: tabletop-grasp categories the large-scale IORS target pools may be filled from (owner: fill missing candidates;
#: the physical requirement is only "small graspable object that rests on a table / mattress"); validated later
LARGE_TARGET_FILL_FROM = {f"{IORS}-ontop_small_scale", f"{IORS}-nextto_small_scale", f"{IORS}-touching_small_scale", "disembedding"}
UNDER_LARGE_MAX_HEIGHT = 0.20  # m: must fit under a 0.30 m clearance with the gripper below the wrist (spec 6.12)

REP_COLUMNS = ["family", "subtask", "pick", "role", "slot", "category", "model", "scene", "name", "joint", "kind", "grasp_key",
               "scale", "geometry_group", "exact_group", "status", "note", "triplet"]

#: the §4 tabletop subtasks (objects rest on a scene support chosen by ``support_slots``): the support triplets apply here;
#: the two large-scale IORS subtasks choose their init surface per partner (spec 5.6) and wall subtasks use wall regions
TABLETOP_SUPPORT_SUBTASKS = [st for st, sp in SUBTASKS.items()
                             if sp["roles"].get("support", {}).get("kind") == "scene"
                             and sp["roles"].get("partner", {}).get("kind") != "scene_partner"]
SUPPORT_TOP_Z = (0.55, 1.20)                # spec 3.2 tabletop range (build_icl admission)
SUPPORT_MIN_AREA_M2 = 0.12
EDGE_FREE_RUN_MIN_M = 0.60                  # spec 3.2 / build_icl: >= 1 edge with a free-floor run >= 0.60 m at stand-off 0.45
ADMIT_STAND_OFF = "0.45"                    # the B4 run used for admission (Phase 1a init stand-off)
STAND_OFFS = ("0.35", "0.45", "0.55")       # the B4 runs (Phase 2 scans the three)
#: an object closer than this in front of the edge midpoint (B4 ray at top + 0.3 m, 1.2 m out) leaves no room for the base
#: at any stand-off (0.45 + ~0.3 m base half-length; the pilot: the breakfast table's +u side with the window at 0.635 m)
BLOCKED_IN_FRONT_MAX_M = 0.80
OVERHANG_FLAG_M = 0.05                      # spec 4.2: the stand-off is measured from the AABB when the overhang is >= 5 cm
#: abs(lambda_min) + max(0, lambda_max): lateral span of the right-arm band union at the init trunk [pilot]. Back-solved from the
#: two measured margin rows: toolbox (w_a 0.30) 1.30 = span + 0.61 + 0.04 -> 0.65; IORS 35 cm platter with a 7 cm target
#: 2.35 = span + 0.14 + 0.76 + 0.70 + 0.04 -> 0.71 (the band lies to the robot's right, lambda_max <= 0.08).
BAND_SPAN_M = 0.70
#: §4.4 margin rows (m_R + m_L, the edge length a row needs after the lateral margins), longest first. ``pilot`` rows are the
#: numbers the E1 pilot measured on breakfast_table_rhjoby_0 (TODO.md, 2026-09-15 19:50 / 2026-09-16 01:40); ``estimate`` rows
#: apply the §4.4 formula with BAND_SPAN_M to the picked objects (per-pick values in the note). Two long rows (>= 2.30 m) and
#: five short rows: a support whose best usable edge is below the long rows is flagged ``short_rows_only``.
MARGIN_ROWS: List[dict] = [
    {"id": "iors_small_scale", "need": 2.35, "how": "pilot",
     "subtasks": [f"{IORS}-ontop_small_scale", f"{IORS}-under_small_scale", f"{IORS}-inside_small_scale", f"{IORS}-nextto_small_scale", f"{IORS}-touching_small_scale"],
     "note": "abs(lambda_min) + max(0, lambda_max) + 2 w_t + 2 D_p + 4 w_p + 0.04 with a 35 cm platter partner (D_p 0.38, w_p 0.175)"},
    {"id": "disembedding", "need": 2.30, "how": "pilot", "subtasks": ["disembedding"],
     "note": "2 (b + 0.437 + 0.02), b = 0.66-0.70 at tilt 0 with the 99 deg hfov (two fixed trays along the edge)"},
    {"id": "articulation_small_scale", "need": 1.30, "how": "pilot",
     "subtasks": [f"{CM}-articulation_open_small_scale", f"{CM}-articulation_close_small_scale"],
     "note": "toolbox (0.61 m at q_init): span + 2 w_a + 0.04; the 1.0 m carton pick would need ~1.75"},
    {"id": "axial", "need": 1.30, "how": "estimate", "subtasks": ["alignment-axial"],
     "note": "span + 2 w_diag + 0.04, w_diag = 1/2 sqrt(long^2 + short^2): frying pan 1.29, keyboard 1.25, books ~1.1"},
    {"id": "attachment", "need": 1.20, "how": "estimate", "subtasks": [f"{CM}-attachment"],
     "note": "span + 2 max(w_target, w_partner) + 0.04: table-lamp pair 1.20, bottle / candle pairs ~0.85; the skateboard-deck pair (1.04 m) ~1.8"},
    {"id": "iors_large_scale_init", "need": 1.10, "how": "estimate", "subtasks": [f"{IORS}-ontop_large_scale", f"{IORS}-under_large_scale"],
     "note": "target only on the init surface (spec 5.6 stances A / B): span + 2 w_t + 0.04: briefcase 1.07, sandal 0.98, marker 0.89, apple 0.83"},
    {"id": "block_design", "need": 0.80, "how": "estimate", "subtasks": [f"{CM}-block_design"],
     "note": "span + 0.0125 + 0.026 + 0.0125 + 0.04 (half edge + one pitch to the right of the anchor)"},
]
LONG_ROWS = ("iors_small_scale", "disembedding")
LONGEST_NEED_M = max(r["need"] for r in MARGIN_ROWS)

#: owner decision 2026-09-16 (A3): the support triplets after T1, (scene, name) per slot; T1 is derived in ``support_slots``
#: (test pin + same-scene height variation + other-scene held-out). ``heldout_alt`` replaces ``heldout`` when it fails the checks.
SUPPORT_TRIPLETS: List[dict] = [
    {"representative": ("house_single_floor", "countertop_kelker_0"), "variation_geometry": ("house_single_floor", "bar_udatjt_0"),
     "heldout": ("restaurant_diner", "commercial_kitchen_table_vxvtec_0"), "heldout_alt": None},
    {"representative": ("office_cubicles_right", "conference_table_fpkmyw_0"), "variation_geometry": ("office_cubicles_right", "desk_mdhelw_1"),
     "heldout": ("restaurant_diner", "bar_xxftww_0"), "heldout_alt": ("Rs_int", "bottom_cabinet_jhymlr_0")},
]
SUPPORT_SLOTS = ("representative", "variation_geometry", "heldout")
FLOOR = "floor"
PATCH_KINDS = ("under_large_scale", "support_triplets", "articulation_large_scale")

#: large-scale articulation (owner decision 2026-09-16 06:50, item 6): the crowded kitchen picks are replaced by roomy B4
#: units, one pick per door class, shared by the open / close subtasks. Per slot a preference list of (scene, instance,
#: joint); the first candidate that passes :meth:`Selector.artic_check` takes the slot, every candidate is reported.
ARTIC_FREE_MIN_M = 3.0          # free run in front (B4 ``front_free_m``: base centre on free floor, Prompt-1 Phase 1 rule 1)
ARTIC_FREE_ERODED_MIN_M = 2.0   # robot-eroded run (proxy for the whole-body check; view stances scan d up to 2.5 m)
ARTIC_HANDLE_Z = (0.30, 1.70)   # world height of the pinch (build_icl HANDLE_Z)
ARTIC_PINCH_MM = (15.0, 80.0)   # pinch width after the instance scale (build_icl PINCH_MM, ground_isg PINCH_RANGE_M)
ARTIC_LARGE_PICKS: List[dict] = [
    {"class": "cabinet doors", "category": "bottom_cabinet",
     "representative": [("office_cubicles_right", "bottom_cabinet_rvunhj_0", "j_link_2")],
     "variation_geometry": [("office_cubicles_right", "bottom_cabinet_rvunhj_5", "j_link_2")],
     "heldout": [("office_cubicles_right", "bottom_cabinet_rvunhj_3", "j_link_2")]},
    {"class": "fridge door", "category": "fridge",
     "representative": [("restaurant_diner", "fridge_jtqazu_1", "j_link_0")],
     "variation_geometry": [("house_double_floor_lower", "fridge_petcxr_0", "j_left_door")],
     "heldout": [("restaurant_diner", "fridge_jtqazu_0", "j_link_0")]},
    {"class": "top cabinet doors", "category": "top_cabinet",
     "representative": [("house_double_floor_lower", "top_cabinet_tynnnw_1", "j_rightdoor")],
     "variation_geometry": [("Rs_int", "top_cabinet_lsyzkh_0", "j_link_1")],
     "heldout": []},
    {"class": "oven door", "category": "oven",
     "representative": [("restaurant_diner", "oven_amblrk_0", "j_door_leaf1")],
     "variation_geometry": [("house_double_floor_lower", "oven_ffitak_0", "j_door")],
     "heldout": [("Rs_int", "oven_wuinhm_0", "j_link_1")]},
    {"class": "drawers", "category": "bottom_cabinet",
     "representative": [("Rs_int", "bottom_cabinet_bamfsz_0", "j_link_4")],
     "variation_geometry": [("Rs_int", "bottom_cabinet_bamfsz_1", "j_link_4")],
     "heldout": [("Rs_int", "bottom_cabinet_jhymlr_0", "j_link_2")]},
]
ARTIC_LARGE_SUBTASKS = (f"{CM}-articulation_open_large_scale", f"{CM}-articulation_close_large_scale")
HANDLE_ADMISSIBLE_VERDICTS = ("handle", "edge", "manual")


def follow_class(category: str, joint_kind: str) -> str:
    """Owner decision A4: cabinets / fridge / dishwasher / oven doors and flaps -> manipulation stance on the hinge side,
    arm-only follow; drawers stay base mode; room doors -> hybrid follow."""
    if joint_kind == "drawer":
        return "base"
    if category == "door":
        return "hybrid"
    return "arm_only"


# ----------------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------------
def read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_duplicates() -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    """(category, model) -> exact-duplicate group id; (category, model) -> same-geometry (texture variant) group id."""
    exact: Dict[Tuple[str, str], str] = {}
    geom: Dict[Tuple[str, str], str] = {}
    for path in DUP_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        clusters = data if isinstance(data, list) else data.get("clusters", [])
        for cl in clusters:
            cat = cl["category"]
            cid = cl["id"]
            verdict = cl.get("verdict")
            for grp in cl.get("exact_groups", []) or []:
                gid = f"{cid}/exact:{'+'.join(sorted(grp))}"
                for m in grp:
                    exact[(cat, m)] = gid
            if verdict == "exact_same":
                for m in cl["models"]:
                    exact.setdefault((cat, m), f"{cid}/exact")
            if verdict in ("texture_variants", "exact_same") or verdict is None:
                # selection duplicates.json: pairs with tier same_shape / identical and same_textures flag
                pairs = cl.get("pairs", [])
                if verdict is None:
                    same_geo = [p for p in pairs if p.get("tier") in ("same_shape", "identical")]
                    if not same_geo:
                        continue
                    for p in same_geo:
                        for m in (p["a"], p["b"]):
                            geom[(cat, m)] = cid
                        if p.get("same_textures"):
                            exact.setdefault((cat, p["a"]), f"{cid}/exact:{p['a']}+{p['b']}")
                            exact.setdefault((cat, p["b"]), f"{cid}/exact:{p['a']}+{p['b']}")
                else:
                    for m in cl["models"]:
                        geom[(cat, m)] = cid
    return exact, geom


def admissible_grasps(grasps: List[dict]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(category, model) -> {pipeline: grasp key} for the admissible rows (verdict ok, or flag with only one_finger)."""
    out: Dict[Tuple[str, str], Dict[str, str]] = collections.defaultdict(dict)
    for r in grasps:
        if not r["require"].startswith("grasps:"):
            continue
        flags = {x.strip() for x in r["flags"].split(",") if x.strip()}
        if r["verdict"] == "ok" or (r["verdict"] == "flag" and flags <= {"one_finger"}):
            pipe = r["require"].split(":", 1)[1]
            out[(r["category"], r["model"])].setdefault(pipe, r["key"])
    return out


def admissible_handles(handles: List[dict]) -> Dict[Tuple[str, str], List[str]]:
    """canonical (category, model) -> joints with an admissible handle row (handle / edge / manual)."""
    out: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    for r in handles:
        if r["scene"]:
            continue
        if r["verdict"] in ("handle", "edge", "manual") and r.get("review_graspable", "") not in ("False", "false", "0"):
            if r["joint"] not in out[(r["category"], r["model"])]:
                out[(r["category"], r["model"])].append(r["joint"])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------------------------------------------------
class Selector:
    def __init__(self, icl_dir: str, per_family: int, min_per_subtask: int, n_triplets: int = 3):
        self.per_family = per_family
        self.min_per_subtask = min_per_subtask
        self.n_triplets = max(1, n_triplets)
        self.icl_dir = icl_dir
        self.cands = read_csv(os.path.join(icl_dir, "roles", "candidates.csv"))
        self.compat = read_csv(os.path.join(icl_dir, "roles", "constraints_compatible.csv"))
        self.conflict = read_csv(os.path.join(icl_dir, "roles", "constraints_conflicting.csv"))
        self.grasps = admissible_grasps(read_csv(os.path.join(icl_dir, "require", "grasps.csv")))
        handle_rows = read_csv(os.path.join(icl_dir, "require", "handles.csv"))
        self.handles = admissible_handles(handle_rows)
        # B4 scene-instance handle rows: (scene, instance, joint) -> rows (every verdict; artic_check filters them)
        self.handle_instance_rows: Dict[Tuple[str, str, str], List[dict]] = collections.defaultdict(list)
        for r in handle_rows:
            if r.get("scene"):
                self.handle_instance_rows[(r["scene"], r["instance"], r["joint"])].append(r)
        self.exact, self.geom = load_duplicates()
        with open(SCENE_INVENTORY) as f:
            self.inventory = json.load(f)["scenes"]
        self.lookalike: Dict[Tuple[str, str], str] = {(r["category"], r["model"]): r["group_id"] for r in self.conflict}
        # global state
        self.heldout: Dict[str, str] = {}          # category -> held-out model (global)
        self.used_models: Set[Tuple[str, str]] = set()   # models used as representative / variation anywhere
        self.used_categories: collections.Counter = collections.Counter()  # category -> number of picks using it
        self.rows: List[dict] = []
        self.extra: List[dict] = []
        self.report: Dict[str, List[str]] = collections.defaultdict(list)
        self._admission: Dict[Tuple[str, str], dict] = {}   # (scene, name) -> support_admission record (cached)
        self._artic_checks: Dict[Tuple[str, str, str], dict] = {}  # (scene, name, joint) -> artic_check record (cached)
        self._triplets: Optional[List[dict]] = None          # support_triplets() cache
        self.support_notes: List[str] = []                    # triplet / own-top notes for the report
        self.patch_log: Dict[str, List[str]] = collections.defaultdict(list)  # --patch-only: what changed
        self.by_subtask_role: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
        self.bbox: Dict[Tuple[str, str], Tuple[float, float, float]] = {}  # (category, model) -> native bbox (m)
        for r in self.cands:
            self.by_subtask_role[(r["subtask"], r["role"])].append(r)
            if r.get("kind") == "sampleable" and r.get("bbox_x") and (r["category"], r["model"]) not in self.bbox:
                try:
                    self.bbox[(r["category"], r["model"])] = (float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_z"]))
                except ValueError:
                    pass
        # every pipeline a category is required with anywhere (for the global held-out choice)
        self.category_pipelines: Dict[str, Set[str]] = collections.defaultdict(set)
        for st, spec in SUBTASKS.items():
            for role, rs in spec["roles"].items():
                if rs.get("pipeline"):
                    for r in self.by_subtask_role.get((st, role), []):
                        self.category_pipelines[r["category"]].add(rs["pipeline"])

    # -- helpers ------------------------------------------------------------------------------------------------------
    def geom_id(self, cat: str, model: str) -> str:
        return self.geom.get((cat, model), f"{cat}/{model}")

    def exact_id(self, cat: str, model: str) -> str:
        return self.exact.get((cat, model), f"{cat}/{model}")

    def distinct_geometry(self, cat: str, a: str, b: str) -> bool:
        return self.geom_id(cat, a) != self.geom_id(cat, b)

    def model_ok(self, r: dict, rs: dict) -> Tuple[bool, str]:
        cat, model = r["category"], r["model"]
        if r.get("ready", "1") not in ("", "1"):
            return False, "not_ready"
        if r.get("status", "").startswith("geometric_reject"):
            return False, r["status"]
        if rs.get("pipeline"):
            pipes = self.grasps.get((cat, model), {})
            if rs["pipeline"] not in pipes:
                return False, f"missing_grasps:{rs['pipeline']}"
        if rs.get("asymmetric") and r.get("symmetry", "none") not in ("", "none"):
            return False, f"symmetric:{r['symmetry']}"
        if rs.get("handle") and not self.handles.get((cat, model)):
            return False, "missing_handle"
        if rs.get("frame") == "wall":
            if not (r.get("frame_front") and r.get("frame_up")) and not r.get("frame_level"):
                return False, "missing_frame"
            if r.get("symmetric_180", "") == "":
                return True, "symmetric_180 unfilled (spec 2.2: wall targets need 0/1; treated as admissible for now)"
        if rs.get("frame") == "axial" and not r.get("frame_long"):
            return False, "missing_frame"
        if rs.get("natural") and rs["natural"] not in (r.get("natural_scenes") or ""):
            return False, f"not_natural_in:{rs['natural']}"
        return True, ""

    def pool(self, subtask: str, role: str, rs: dict) -> Dict[str, List[dict]]:
        """category -> admissible rows (deduplicated exact groups: one row per exact group), in icl order."""
        out: Dict[str, List[dict]] = collections.defaultdict(list)
        seen_exact: Set[str] = set()
        for r in self.by_subtask_role.get((subtask, role), []):
            ok, why = self.model_ok(r, rs)
            if not ok:
                continue
            ex = self.exact_id(r["category"], r["model"])
            if ex in seen_exact:
                continue
            seen_exact.add(ex)
            out[r["category"]].append(r)
        return out

    def rank_of(self, r: dict) -> int:
        try:
            return int(r.get("rank") or 9)
        except ValueError:
            return 9

    def has_texture_variant(self, cat: str, rows: List[dict], model: str) -> Optional[dict]:
        gid = self.geom.get((cat, model))
        if gid is None:
            return None
        for r in rows:
            if r["model"] != model and self.geom.get((cat, r["model"])) == gid and self.exact_id(cat, r["model"]) != self.exact_id(cat, model):
                return r
        return None

    # -- slot assignment ----------------------------------------------------------------------------------------------
    def choose_heldout(self, cat: str, rows: List[dict], exclude: Set[str]) -> Optional[str]:
        """Global held-out model of a category: admissible for every pipeline the category is required with (if such a
        model exists), different geometry from the excluded (representative / variation) models, lowest icl rank among
        the remaining candidates (the reviewers' pick 1 / pick 2 stay as representative / variation)."""
        if cat in self.heldout:
            return self.heldout[cat]
        need = self.category_pipelines.get(cat, set())
        exclude = set(exclude) | {m for c, m in self.used_models if c == cat}  # never a model another pick already trains on
        cands = [r for r in rows if r["model"] not in exclude and all(self.distinct_geometry(cat, r["model"], e) for e in exclude)]
        if not cands:
            cands = [r for r in rows if r["model"] not in exclude]
        if not cands:
            return None

        def key(r):
            pipes = set(self.grasps.get((cat, r["model"]), {}))
            full = need <= pipes
            return (0 if full else 1, -self.rank_of(r) if self.rank_of(r) >= 3 else 5, self.rank_of(r))
        cands.sort(key=key)
        # prefer models ranked "alternate"/"candidate" (3-4) over pick 1 / pick 2 so the best-reviewed pair stays for training
        best = cands[0]
        self.heldout[cat] = best["model"]
        return best["model"]

    def assign_slots(self, subtask: str, role: str, cat: str, rows: List[dict], rs: dict, pick: str,
                     prefer_unused: bool = True) -> Tuple[Optional[dict], List[str]]:
        """Return (representative row, notes) and append the slot rows for one (pick, role) of a sampleable/group role."""
        notes: List[str] = []
        by_rank = sorted(rows, key=lambda r: (self.rank_of(r), rows.index(r)))
        held = self.heldout.get(cat)
        avail = [r for r in by_rank if r["model"] != held]
        if not avail:
            return None, ["no model left after the global held-out"]
        # representative: the icl representative mark if admissible, else the best rank; prefer models unused elsewhere
        rep = None
        marked = [r for r in avail if r.get("representative") == "1"]
        pool_order = marked + [r for r in avail if r not in marked]
        if not prefer_unused and rows and rows[0] is not avail[0] and rows[0] in avail:
            # caller-ordered pool (attachment: models that mate with the chosen female first) wins over the icl mark
            pool_order = [r for r in rows if r in avail]
        if prefer_unused:
            unused = [r for r in pool_order if (cat, r["model"]) not in self.used_models]
            if unused:
                pool_order = unused + [r for r in pool_order if r not in unused]
        rep = pool_order[0]
        rest = [r for r in avail if r is not rep]
        # variation slots prefer models no other pick uses (fewer overlapping models across the selection)
        rest = [r for r in rest if (cat, r["model"]) not in self.used_models] + [r for r in rest if (cat, r["model"]) in self.used_models]
        # variation slots
        tex = self.has_texture_variant(cat, rest, rep["model"])
        geo_cands = [r for r in rest if self.distinct_geometry(cat, r["model"], rep["model"])]
        # held-out: chosen globally now (exclude rep, tex, and the best geometry candidate so it stays for variation)
        if held is None:
            exclude = {rep["model"]}
            if tex is not None:
                exclude.add(tex["model"])
            if geo_cands:
                exclude.add(geo_cands[0]["model"])
            held = self.choose_heldout(cat, rows, exclude)
            rest = [r for r in rest if r["model"] != held]
            geo_cands = [r for r in geo_cands if r["model"] != held]
            if tex is not None and tex["model"] == held:
                tex = None
        geo = geo_cands[0] if geo_cands else None
        if tex is None:
            notes.append("V-OBJ texture-variant model: none in the pool (no same-geometry variant) - colour change only")
        if geo is None:
            notes.append("VB-MOBJ variation (different geometry, same category): unavailable - pool too small")
        if held is None:
            notes.append("held-out model: unavailable - pool too small")
        elif not self.distinct_geometry(cat, held, rep["model"]):
            notes.append("held-out model is a texture variant of the representative (no third geometry in the pool)")
        self.used_models.add((cat, rep["model"]))
        for slot, r in (("representative", rep), ("variation_texture", tex), ("variation_geometry", geo)):
            if r is None:
                continue
            if slot != "representative":
                self.used_models.add((cat, r["model"]))
            self.rows.append(self.row(subtask, pick, role, slot, r, rs))
        if held is not None:
            hr = next((r for r in rows if r["model"] == held), None)
            if hr is not None:
                self.rows.append(self.row(subtask, pick, role, "heldout", hr, rs, note="global held-out of the category"))
        return rep, notes

    def slots_str(self, pick: str, role: str) -> str:
        parts = []
        for r in self.rows:
            if r["pick"] == pick and r["role"] == role and r["slot"] != "representative":
                parts.append(f"{r['slot'].replace('variation_', 'var:')}={r['model'] or r['name']}")
        return "[" + ", ".join(parts) + "]" if parts else "[no other slot]"

    def row(self, subtask: str, pick: str, role: str, slot: str, r: dict, rs: dict, note: str = "", **over) -> dict:
        cat, model = r["category"], r["model"]
        key = r.get("grasp_key", "")
        if rs.get("pipeline"):
            key = self.grasps.get((cat, model), {}).get(rs["pipeline"], key)
        d = {"family": SUBTASKS[subtask]["family"], "subtask": subtask, "pick": pick, "role": role, "slot": slot, "category": cat,
             "model": model, "scene": r.get("scene", ""), "name": "", "joint": r.get("joint", ""), "kind": r.get("kind", ""),
             "grasp_key": key, "scale": r.get("scale", ""), "geometry_group": self.geom.get((cat, model), ""),
             "exact_group": self.exact.get((cat, model), ""), "status": r.get("status", "untested"), "note": note}
        d.update(over)
        return d

    # -- category scoring ----------------------------------------------------------------------------------------------
    def category_score(self, cat: str, rows: List[dict], rs: dict) -> float:
        n = len(rows)
        n_geo = len({self.geom_id(cat, r["model"]) for r in rows})
        s = 0.0
        s += 4.0 if n >= 3 else (1.5 if n == 2 else 0.0)
        s += 1.0 if n >= 4 else 0.0
        s += 2.0 if n_geo >= 3 else (0.5 if n_geo == 2 else -2.0)  # rep + variation + held-out want three geometries
        s += min(n, 8) * 0.15  # many candidates preferred
        rep = next((r for r in rows if r.get("representative") == "1"), rows[0])
        if self.has_texture_variant(cat, rows, rep["model"]) is not None:
            s += 2.0
        if cat in self.heldout and self.heldout[cat] not in {r["model"] for r in rows}:
            s -= 3.0  # global held-out not admissible here -> generalization axis lost
        if any(r.get("status", "").startswith("incomplete") for r in rows):
            s -= 2.0  # filled-in candidates rank after the surveyed ones
        s -= 1.5 * self.used_categories[cat]  # diversity across the whole selection
        s -= 1.0 if any((cat, r["model"]) in self.used_models for r in rows[:2]) else 0.0
        return s

    # -- scene roles --------------------------------------------------------------------------------------------------
    def supports_admitted(self) -> List[dict]:
        out = []
        for scene, d in self.inventory.items():
            for name, s in d["supports"].items():
                z = s["top_z"]
                if not (0.55 <= z <= 1.20) or s["area_m2"] < 0.12 or s["n_free_cells"] == 0 or s["best_run_m"] < 0.60:
                    continue
                out.append({"scene": scene, "name": name, "category": s["category"], "top_z": z, "area": s["area_m2"],
                            "run": s["best_run_m"], "edge": s["best_edge"], "in_union": s.get("in_union", False),
                            "rooms": s.get("rooms", [])})
        return out

    # -- edge length vs §4.4 margins ---------------------------------------------------------------------------------
    def support_record(self, scene: str, name: str) -> Optional[dict]:
        """The B4 per-support JSON (edges with runs at the three stand-offs, blocked objects, rooms in front)."""
        path = os.path.join(os.path.dirname(SCENE_INVENTORY), scene, "supports", f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def support_admission(self, scene: str, name: str) -> dict:
        """Edge-length-vs-margin check of one support (owner decision 2026-09-16, A3 follow-up).

        Per edge: ``usable`` = min(edge length, longest free-floor run at stand-off 0.45 with the chairs within 1 m removed),
        0 when the run is below the 0.60 m admission or an object stands closer than BLOCKED_IN_FRONT_MAX_M in front of the
        edge midpoint (B4 ray; the free-floor run does not see walls between the edge and the stand-off line - the run of an
        edge that faces a wall is the floor of the next room). ``longest_row`` = the first :data:`MARGIN_ROWS` entry whose
        m_R + m_L fits the best usable edge; flags: ``short_rows_only`` (below both long rows), ``no_measured_row`` (below
        every pilot-measured row), ``overhang`` (Phase 1a measures the stand-off from the AABB), ``run035_short`` (the
        Phase 2 stand-off 0.35 run of the best edge does not hold the admitted row), ``top_z`` / ``area`` out of range."""
        key = (scene, name)
        if key in self._admission:
            return self._admission[key]
        rec = self.support_record(scene, name)
        summ = (self.inventory.get(scene) or {}).get("supports", {}).get(name)
        out: dict = {"scene": scene, "name": name, "exists": rec is not None or summ is not None, "edges": [], "best": None,
                     "usable_m": 0.0, "longest_row": None, "admits": [], "flags": [], "blocked": [],
                     "category": (rec or summ or {}).get("category", ""), "model": (rec or {}).get("model", ""),
                     "rooms": list((rec or summ or {}).get("rooms") or []),
                     "top_z": float(((rec or {}).get("top") or {}).get("z", (summ or {}).get("top_z", 0.0)) or 0.0),
                     "area": float(((rec or {}).get("top") or {}).get("area_m2", (summ or {}).get("area_m2", 0.0)) or 0.0),
                     "summary_edge": (summ or {}).get("best_edge"), "summary_run": float((summ or {}).get("best_run_m", 0.0) or 0.0)}
        if rec is None:
            out["flags"].append("no_B4_support_record" if summ is not None else "not_in_B4_inventory")
            self._admission[key] = out
            return out
        for e in rec.get("edges") or []:
            runs = {s: float(((e.get("runs") or {}).get(s) or {}).get("longest_free_run_m", 0.0)) for s in STAND_OFFS}
            blocked = e.get("blocked_in_front") or None
            dist = float(blocked["distance_m"]) if blocked else None
            is_blocked = dist is not None and dist < BLOCKED_IN_FRONT_MAX_M
            length = float(e.get("length_m", 0.0))
            usable = 0.0 if is_blocked or runs[ADMIT_STAND_OFF] < EDGE_FREE_RUN_MIN_M else round(min(length, runs[ADMIT_STAND_OFF]), 2)
            edge = {"name": e["name"], "length": length, "runs": runs, "usable": usable, "overhang": float(e.get("overhang_m", 0.0)),
                    "room": e.get("room_in_front"), "blocked": f"{blocked['object']}@{dist:.2f}" if is_blocked else "",
                    "in_front": f"{blocked['object']}@{dist:.2f}" if blocked else ""}
            out["edges"].append(edge)
            if is_blocked:
                out["blocked"].append(f"{e['name']}:{blocked['object']}@{dist:.2f}")
        usable_edges = [e for e in out["edges"] if e["usable"] > 0]
        if usable_edges:  # ties: the longer 0.35 run, then the summary's best edge, then the name
            best = max(usable_edges, key=lambda e: (e["usable"], e["runs"]["0.35"], e["name"] == out["summary_edge"], e["name"]))
            out["best"], out["usable_m"] = best, best["usable"]
        out["admits"] = [r["id"] for r in MARGIN_ROWS if out["usable_m"] >= r["need"]]
        out["longest_row"] = out["admits"][0] if out["admits"] else None
        if not (SUPPORT_TOP_Z[0] <= out["top_z"] <= SUPPORT_TOP_Z[1]):
            out["flags"].append(f"top_z={out['top_z']:.2f} outside {SUPPORT_TOP_Z[0]:.2f}-{SUPPORT_TOP_Z[1]:.2f}")
        if out["area"] < SUPPORT_MIN_AREA_M2:
            out["flags"].append(f"area={out['area']:.2f} < {SUPPORT_MIN_AREA_M2}")
        if not usable_edges:
            out["flags"].append("no_usable_edge")
        elif out["usable_m"] < min(r["need"] for r in MARGIN_ROWS if r["id"] in LONG_ROWS):
            out["flags"].append("short_rows_only")
            if out["usable_m"] < min(r["need"] for r in MARGIN_ROWS if r["how"] == "pilot"):
                out["flags"].append("no_measured_row")
        if out["best"] is not None:
            if out["best"]["overhang"] >= OVERHANG_FLAG_M:
                out["flags"].append(f"overhang={out['best']['overhang']:.2f} (stand-off from the AABB {0.45 + out['best']['overhang']:.2f} m)")
            need = next((r["need"] for r in MARGIN_ROWS if r["id"] == out["longest_row"]), None)
            if need is not None and min(out["best"]["length"], out["best"]["runs"]["0.35"]) < need:
                out["flags"].append(f"run035_short={out['best']['runs']['0.35']:.2f}")
        self._admission[key] = out
        return out

    def support_dict(self, adm: dict) -> dict:
        """The support dict used for rows (schema of ``supports_admitted``) from an admission record: the usable best edge."""
        best = adm["best"] or {}
        return {"scene": adm["scene"], "name": adm["name"], "category": adm["category"], "top_z": adm["top_z"], "area": adm["area"],
                "run": adm["usable_m"], "edge": best.get("name", "-"), "rooms": adm["rooms"], "longest_row": adm["longest_row"],
                "flags": list(adm["flags"])}

    def pin_check(self, scene: str, name: str, need: float = LONGEST_NEED_M) -> Tuple[bool, List[str]]:
        """A pinned triplet support must exist in the B4 inventory, have its top in the tabletop range and a usable edge that
        admits the longest §4.4 row (``need``)."""
        adm = self.support_admission(scene, name)
        why = []
        if not adm["exists"] or "no_B4_support_record" in adm["flags"]:
            return False, [f"{scene}/{name}: not in the B4 inventory"]
        if not (SUPPORT_TOP_Z[0] <= adm["top_z"] <= SUPPORT_TOP_Z[1]):
            why.append(f"top z {adm['top_z']:.2f} outside the tabletop range")
        if adm["area"] < SUPPORT_MIN_AREA_M2:
            why.append(f"area {adm['area']:.2f} m2 < {SUPPORT_MIN_AREA_M2}")
        if adm["usable_m"] < need:
            why.append(f"best usable edge {adm['usable_m']:.2f} m < {need:.2f} m (longest admissible row: {adm['longest_row'] or 'none'})")
        return not why, [f"{scene}/{name}: " + "; ".join(why)] if why else []

    def support_triplets(self) -> List[dict]:
        """The N support triplets shared by every tabletop pick: ``[{"triplet": i, "slots": {slot: support dict}, "notes": [...]}]``.
        T1 = the 2026-09-15 rule (test pin, same-scene height variation, other-scene held-out; unchanged), T2.. = the owner's
        pins verified by :meth:`pin_check` (held-out falls back to ``heldout_alt``; a failing representative drops the triplet)."""
        if self._triplets is not None:
            return self._triplets
        notes: List[str] = []
        sup = self.supports_admitted()
        by_name = {(s["scene"], s["name"]): s for s in sup}
        rep = by_name.get((TEST_SCENE, TEST_SUPPORT))
        if rep is None:
            notes.append(f"support pin {TEST_SUPPORT} not admitted by the geometric filter")
            rep = sorted(sup, key=lambda s: (-s["in_union"], -s["run"]))[0]
        same_scene = [s for s in sup if s["scene"] == rep["scene"] and s["name"] != rep["name"]]
        var = sorted(same_scene, key=lambda s: (-(abs(s["top_z"] - rep["top_z"]) >= 0.10), -s["in_union"], -s["run"]))
        var = var[0] if var else None
        held = [s for s in sup if s["scene"] != rep["scene"] and s["category"] != rep["category"]]
        held = sorted(held, key=lambda s: (-s["in_union"], -s["run"]))[0] if held else None
        if var is None:
            notes.append("VB-ISC support change: no second admitted support in the test scene")
        elif abs(var["top_z"] - rep["top_z"]) < 0.10:
            notes.append("VB-ISC support height change < 10 cm in the test scene (same-height alternates only)")
        t1 = {"triplet": 1, "slots": {}, "notes": notes, "pick_notes": list(notes)}
        for slot, s in zip(SUPPORT_SLOTS, (rep, var, held)):
            if s is not None:
                t1["slots"][slot] = s
        out = [t1]
        for i, pin in enumerate(SUPPORT_TRIPLETS[: self.n_triplets - 1], start=2):
            t = {"triplet": i, "slots": {}, "notes": [], "pick_notes": []}
            for slot in SUPPORT_SLOTS:
                cands = [pin[slot]] + ([pin["heldout_alt"]] if slot == "heldout" and pin.get("heldout_alt") else [])
                chosen = None
                for j, (scene, name) in enumerate(cands):
                    ok, why = self.pin_check(scene, name)
                    if ok:
                        chosen = self.support_dict(self.support_admission(scene, name))
                        if j > 0:
                            t["notes"].append(f"T{i} {slot}: {cands[0][0]}/{cands[0][1]} failed the checks, alternative {scene}/{name} used")
                        break
                    t["notes"].append(f"T{i} {slot} rejected - " + "; ".join(why))
                if chosen is None:
                    if slot == "representative":
                        t["notes"].append(f"T{i} dropped: the representative support failed the checks")
                        t["slots"] = {}
                        break
                    continue
                t["slots"][slot] = chosen
            if "variation_geometry" in t["slots"] and abs(t["slots"]["representative"]["top_z"] - t["slots"]["variation_geometry"]["top_z"]) < 0.10:
                t["notes"].append(f"T{i}: VB-ISC support height change < 10 cm")
            elif t["slots"] and "variation_geometry" not in t["slots"]:
                t["notes"].append(f"T{i}: VB-ISC support change unavailable (variation support failed the checks)")
            if t["slots"]:
                out.append(t)
            else:
                notes.extend(t["notes"])
        if self.n_triplets - 1 > len(SUPPORT_TRIPLETS):
            notes.append(f"only {len(SUPPORT_TRIPLETS) + 1} support triplets are pinned (--support-triplets {self.n_triplets})")
        self._triplets = out
        return out

    def support_row(self, subtask: str, pick: str, slot: str, s: dict, triplet: int) -> dict:
        note = f"top_z={s['top_z']:.2f} area={s['area']:.2f} best_edge={s['edge']} run045={s['run']:.2f}"
        status = "geometric_ok" if s["run"] >= EDGE_FREE_RUN_MIN_M else "geometric_weak"
        if triplet > 1:  # T1 notes / statuses stay exactly as written on 2026-09-15 (summary best edge); T2.. use the usable edge
            note += f" margin_row={s.get('longest_row') or 'none'} rooms={','.join(s.get('rooms') or []) or '-'}"
            if slot == "heldout":
                note += " (global held-out support)"
            if s.get("flags"):
                note += " flags=" + ";".join(s["flags"])
            status = "geometric_ok" if s.get("longest_row") in LONG_ROWS else "geometric_weak:short_rows_only"
        return {"family": SUBTASKS[subtask]["family"], "subtask": subtask, "pick": pick, "role": "support", "slot": slot,
                "category": s["category"], "model": "", "scene": s["scene"], "name": s["name"], "joint": "", "kind": "scene",
                "grasp_key": "", "scale": "", "geometry_group": "", "exact_group": "", "status": status, "note": note, "triplet": str(triplet)}

    def support_slots(self, subtask: str, pick: str) -> List[str]:
        """Append the support rows of a tabletop pick: N triplets (``support_triplets``), written T2, T3, ..., T1 so that
        readers indexing (pick, role, slot) without the ``triplet`` column keep the T1 binding. Returns the T1 notes (the
        per-pick report line); the T2.. notes go to the support section of the report."""
        triplets = self.support_triplets()
        for t in triplets[1:] + triplets[:1]:
            for slot in SUPPORT_SLOTS:
                s = t["slots"].get(slot)
                if s is not None:
                    self.rows.append(self.support_row(subtask, pick, slot, s, t["triplet"]))
        return list(triplets[0]["pick_notes"])

    # -- IORS large-scale init surface: the partner's own top (spec 5.6 b, owner decision A5) ---------------------------
    def init_row_need(self, target_rows: List[dict]) -> Tuple[float, str]:
        """m_R + m_L of the large-scale init row for a pick: span + 2 w_t + 0.04 with w_t = half the larger horizontal bbox
        side over the pick's target models (representative / variation / held-out; the largest decides; bbox from the icl)."""
        w, who = 0.0, ""
        for r in target_rows:
            bb = self.bbox.get((r["category"], r["model"]))
            if bb is None:
                continue
            wt = max(bb[0], bb[1]) / 2
            if wt > w:
                w, who = wt, f"{r['category']}/{r['model']}"
        return round(BAND_SPAN_M + 2 * w + 0.04, 2), who

    def floor_row(self, subtask: str, pick: str, slot: str, p: dict, why: str = "") -> dict:
        return {"family": IORS, "subtask": subtask, "pick": pick, "role": "support", "slot": slot, "category": FLOOR, "model": "",
                "scene": p["scene"], "name": FLOOR, "joint": "", "kind": "scene", "grasp_key": "", "scale": "", "geometry_group": "",
                "exact_group": "", "status": "untested", "note": f"floor beside {p['name']}" + (f" ({why})" if why else ""), "triplet": ""}

    def fix_heldout_support_collisions(self, subtask: str) -> List[str]:
        """Global held-out rule for the own-top class: a partner instance that another pick of the subtask trains on as its
        init surface cannot be a held-out support as well; that heldout slot falls back to the floor beside the held-out
        partner (the last init-surface class). Returns the log lines."""
        trained: Dict[Tuple[str, str], str] = {}
        for r in self.rows:
            if r["subtask"] == subtask and r["role"] == "support" and r["slot"] != "heldout" and r["category"] != FLOOR:
                trained.setdefault((r["scene"], r["name"]), r["pick"])
        log = []
        for i, r in enumerate(self.rows):
            if r["subtask"] != subtask or r["role"] != "support" or r["slot"] != "heldout" or r["category"] == FLOOR:
                continue
            if not r["note"].startswith("partner's own top") or (r["scene"], r["name"]) not in trained:
                continue
            other = trained[(r["scene"], r["name"])]
            why = f"own top not held out: {r['name']} is the init surface of {other}"
            self.rows[i] = self.floor_row(subtask, r["pick"], "heldout", {"scene": r["scene"], "name": r["name"]}, why)
            log.append(f"{r['pick']}: heldout support {r['name']}@{r['scene']} -> floor ({why})")
        return log

    def own_top_admitted(self, subtask: str, p: dict, need: float) -> Tuple[bool, dict, str]:
        """The partner's own top admits the A / B stances per the B4 inventory when the instance is an admitted support (top
        height, area, free cells, >= 0.60 m run) whose best usable edge admits the pick's large-scale init row (``need``)."""
        adm = self.support_admission(p["scene"], p["name"])
        admitted = {(s["scene"], s["name"]) for s in self.supports_admitted()}
        if (p["scene"], p["name"]) not in admitted:
            return False, adm, f"{p['name']}@{p['scene']}: top not an admitted support (B4: top z {adm['top_z']:.2f}, area {adm['area']:.2f}, flags {';'.join(adm['flags']) or '-'})"
        if adm["best"] is None or adm["usable_m"] < need:
            return False, adm, (f"{p['name']}@{p['scene']}: best usable edge {adm['usable_m']:.2f} m < init row {need:.2f} m "
                                f"(blocked: {', '.join(adm['blocked']) or '-'}; flags {';'.join(adm['flags']) or '-'})")
        return True, adm, ""

    def own_top_rows(self, subtask: str, pick: str, partner_slots: Dict[str, Optional[dict]], target_rows: List[dict]) -> Tuple[List[dict], List[str]]:
        """Support rows for the own-top class of an under large-scale pick: one row per partner slot whose own top is admitted
        (the support of a slot is that slot's partner instance); a slot whose partner top is not admitted gets the floor
        beside that partner. Empty when the representative partner's top is not admitted (the pick stays on the floor)."""
        need, who = self.init_row_need(target_rows)
        rows, notes = [], []
        rep = partner_slots.get("representative")
        ok, adm, why = self.own_top_admitted(subtask, rep, need)
        if not ok:
            return [], [why]
        for slot in SUPPORT_SLOTS:
            p = partner_slots.get(slot)
            if p is None:
                continue
            ok, adm, why = self.own_top_admitted(subtask, p, need)
            if not ok:
                notes.append(f"{slot}: {why} -> floor beside {p['name']}")
                rows.append(self.floor_row(subtask, pick, slot, p))
                continue
            best = adm["best"]
            why_class = (f"no admitted support within 3 m of {p['name']}" if slot == "representative"
                         else f"the pick's class: no admitted support within 3 m of the representative partner {rep['name']}")
            note = (f"partner's own top (spec 5.6 b): {why_class}; usable edge {best['name']} "
                    f"{adm['usable_m']:.2f} m >= init row {need:.2f} m ({who}); top_z={adm['top_z']:.2f} area={adm['area']:.2f} "
                    f"run035/045/055={best['runs']['0.35']:.2f}/{best['runs']['0.45']:.2f}/{best['runs']['0.55']:.2f} "
                    f"rooms={','.join(adm['rooms']) or '-'}")
            if adm["blocked"]:
                note += " blocked_edges=" + ",".join(adm["blocked"])
            if adm["flags"]:
                note += " flags=" + ";".join(adm["flags"])
            near = p.get("near_supports") or []
            if near:
                note += f" (supports within 3 m of this partner: {','.join(near)})"
            rows.append({"family": IORS, "subtask": subtask, "pick": pick, "role": "support", "slot": slot, "category": p["category"], "model": "",
                         "scene": p["scene"], "name": p["name"], "joint": "", "kind": "scene", "grasp_key": "", "scale": "", "geometry_group": "",
                         "exact_group": "", "status": "geometric_ok", "note": note, "triplet": ""})
            notes.append(f"{slot}: {p['name']}@{p['scene']} own top, usable edge {best['name']} {adm['usable_m']:.2f} m >= {need:.2f} m"
                         + (f" [{';'.join(adm['flags'])}]" if adm["flags"] else ""))
        return rows, notes

    def wall_slots(self, subtask: str, pick: str) -> List[str]:
        regions = self.by_subtask_role.get((subtask, "support"), [])
        reps = [r for r in regions if r.get("representative") == "1"]
        rep = reps[0] if reps else (regions[0] if regions else None)
        if rep is None:
            return ["no wall region rows"]
        others = [r for r in regions if r is not rep]
        var = next((r for r in others if r.get("representative") == "1" and r.get("room") == rep.get("room")),
                   next((r for r in others if r["scene"] == rep["scene"]), None))
        held = next((r for r in others if r["scene"] != rep["scene"] and float(r.get("length_m") or 0) >= 0.5), None)
        for slot, r in (("representative", rep), ("variation_geometry", var), ("heldout", held)):
            if r is None:
                continue
            self.rows.append({"family": SUBTASKS[subtask]["family"], "subtask": subtask, "pick": pick, "role": "support", "slot": slot,
                              "category": r["category"], "model": r["model"], "scene": r["scene"], "name": r.get("region_id", ""), "joint": "",
                              "kind": "scene", "grasp_key": "", "scale": "", "geometry_group": "", "exact_group": "", "status": r.get("status", ""),
                              "note": f"wall region {r.get('region_id', '')} hang_z={r.get('hang_z', '')} length={r.get('length_m', '')}"})
        return [] if var is not None else ["VB-ISC: one wall region only in the test room"]

    def partners_admitted(self, subtask: str) -> List[dict]:
        cats = UNDER_LARGE_PARTNERS if subtask.endswith("under_large_scale") else ONTOP_LARGE_PARTNERS
        out = []
        for scene, d in self.inventory.items():
            for name, p in d["partners"].items():
                if p["category"] not in cats:
                    continue
                if subtask.endswith("under_large_scale") and p.get("max_clearance_m", 0) < 0.30:
                    continue
                near = p.get("supports_within_3m") or []
                out.append({"scene": scene, "name": name, "category": p["category"], "face_z": p["face_z"], "clearance": p.get("max_clearance_m", 0),
                            "near_supports": [n.split(" (")[0] for n in near], "model": name.rsplit("_", 1)[0].split("_")[-1], "run": p.get("best_run_m", 0)})
        return out

    def artic_admitted(self) -> List[dict]:
        out = []
        for scene, d in self.inventory.items():
            for name, a in d["artic"].items():
                for j in a["joints"]:
                    if j["status"] not in ("handle", "edge") or j["kind"] not in ("door", "drawer", "flap"):
                        continue
                    handles = j.get("handles") or []
                    kind, height, width = (handles[0] if handles else (j.get("fallback", "edge"), None, None))
                    if height is None or not (0.30 <= height <= 1.70):
                        continue
                    if width is not None and not (15 <= width <= 80):
                        continue
                    if (j.get("front_free_eroded_m") or 0) < 0.60:
                        continue
                    out.append({"scene": scene, "name": name, "category": a["category"], "model": a["model"], "joint": j["joint"], "kind": j["kind"],
                                "handle_kind": kind, "height": height, "width": width, "front_free": j.get("front_free_eroded_m"),
                                "annotated": a.get("canonical_annotation", False), "prior": j.get("prior_review")})
        return out

    # -- per-subtask selection ----------------------------------------------------------------------------------------
    def n_for(self, subtask: str, n_subtasks_with_objects: int) -> int:
        spec = SUBTASKS[subtask]
        if "n" in spec:
            return spec["n"]
        return max(self.min_per_subtask, math.ceil(self.per_family / max(1, n_subtasks_with_objects)))

    def select_single(self, subtask: str, role: str, rs: dict, n: int, with_support: Optional[str]) -> None:
        pool = self.pool(subtask, role, rs)
        if rs.get("pools"):
            pool = self.filter_disembedding(pool)
        scored = sorted(pool.items(), key=lambda kv: (-self.category_score(kv[0], kv[1], rs), kv[0]))
        chosen = 0
        for cat, rows in scored:
            if chosen >= n:
                break
            if chosen >= self.min_per_subtask and len(rows) < 2:
                continue  # beyond the minimum only categories that allow at least one variation model
            pick = f"{subtask}#{chosen + 1:02d}"
            rep, notes = self.assign_slots(subtask, role, cat, rows, rs, pick)
            if rep is None:
                continue
            self.used_categories[cat] += 1
            chosen += 1
            if with_support == "scene":
                notes += self.support_slots(subtask, pick)
            elif with_support == "wall":
                notes += self.wall_slots(subtask, pick)
            self.report[subtask].append(f"{pick}: {role} **{cat}/{rep['model']}** {self.slots_str(pick, role)} (pool {len(rows)}; {'; '.join(notes) or 'all axes available'})")
        if chosen < n:
            self.report[subtask].append(f"only {chosen} of {n} picks: the pool has {len(pool)} admissible categories")

    def filter_disembedding(self, pool: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        """Keep targets whose same-category distractor pool is non-empty after the look-alike exclusion."""
        same = collections.defaultdict(set)
        other = collections.defaultdict(set)
        for r in self.compat:
            if r["subtask"] != "disembedding":
                continue
            if r["relation"] == "distractor_same_category":
                same[r["category_a"]].add(r["model_b"])
            elif r["relation"] == "distractor_other_category":
                other[r["category_a"]].add((r["category_b"], r["model_b"]))
        out = {}
        for cat, rows in pool.items():
            keep = []
            for r in rows:
                grp = self.lookalike.get((cat, r["model"]))
                s = {m for m in same[cat] if m != r["model"] and (grp is None or self.lookalike.get((cat, m)) != grp)}
                if s and other[cat]:
                    keep.append(r)
            if keep:
                out[cat] = keep
        return out

    def select_pairs(self, subtask: str, n: int) -> None:
        spec = SUBTASKS[subtask]
        trs, prs = spec["roles"]["target"], spec["roles"]["partner"]
        tpool = self.pool(subtask, "target", trs)
        if prs["kind"] == "scene_partner":
            self.select_large_pairs(subtask, n, tpool, trs)
            return
        ppool = self.pool(subtask, "partner", prs)
        pairs = collections.defaultdict(set)   # target category -> partner categories (or (cat, model) for attachment)
        model_pairs = collections.defaultdict(set)  # attachment: target model -> partner models
        for r in self.compat:
            if r["subtask"] != subtask or r["relation"] != "partner_of":
                continue
            if r.get("ready", "") not in ("", "1", "True", "true"):
                continue
            pairs[r["category_a"]].add(r["category_b"])
            if r["model_a"]:
                model_pairs[(r["category_a"], r["model_a"])].add((r["category_b"], r["model_b"]))
        cands = []
        for tcat, trows in tpool.items():
            for pcat in pairs.get(tcat, ()):
                if pcat not in ppool:
                    continue
                if trs.get("partner_model"):
                    trows_ok = [r for r in trows if any(pc == pcat for pc, _ in model_pairs.get((tcat, r["model"]), ()))]
                    if not trows_ok:
                        continue
                prow = [r for r in ppool[pcat]]
                if tcat == pcat:
                    continue
                s = self.category_score(tcat, trows, trs) + 0.7 * self.category_score(pcat, prow, prs)
                if trs.get("partner_model"):
                    s -= 0.5 * (len(trows) - len(trows_ok))  # attachment: models without a mate for this female do not help
                cands.append((s, tcat, pcat))
        cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        chosen_t, chosen_p, chosen = set(), collections.Counter(), 0
        for _, tcat, pcat in cands:
            if chosen >= n:
                break
            if tcat in chosen_t:
                continue
            if chosen_p[pcat] >= 1 and any(c not in chosen_p for _, t2, c in cands if t2 not in chosen_t):
                # prefer an unused partner category while one is still available for some unused target
                continue
            trows = tpool[tcat]
            if trs.get("partner_model"):
                # attachment: the representative must mate with this female; variation / held-out may come from the whole
                # category (their female follows the pair table, recorded in the note)
                mates = [r for r in trows if any(pc == pcat for pc, _ in model_pairs.get((tcat, r["model"]), ()))]
                trows = mates + [r for r in trows if r not in mates]
            pick = f"{subtask}#{chosen + 1:02d}"
            rep, notes = self.assign_slots(subtask, "target", tcat, trows, trs, pick, prefer_unused=not trs.get("partner_model"))
            if rep is None:
                continue
            if trs.get("partner_model"):
                for r in self.rows:
                    if r["pick"] == pick and r["role"] == "target" and r["slot"] != "representative":
                        mates_of = sorted({f"{pc}/{pm}" for pc, pm in model_pairs.get((tcat, r["model"]), ())})
                        r["note"] = (r["note"] + "; " if r["note"] else "") + "female for this male: " + (", ".join(mates_of) or "none in the pair table")
            prows = ppool[pcat]
            if trs.get("partner_model"):
                allowed = {pm for pc, pm in model_pairs.get((tcat, rep["model"]), ()) if pc == pcat}
                prows = [r for r in prows if r["model"] in allowed] or prows
            prep, pnotes = self.assign_slots(subtask, "partner", pcat, prows, prs, pick)
            if prep is None:
                continue
            if self.lookalike.get((tcat, rep["model"])) and self.lookalike.get((tcat, rep["model"])) == self.lookalike.get((pcat, prep["model"])):
                self.report[subtask].append(f"skipped {tcat}/{pcat}: look-alike pair")
                continue
            chosen_t.add(tcat)
            chosen_p[pcat] += 1
            self.used_categories[tcat] += 1
            self.used_categories[pcat] += 1
            chosen += 1
            snotes = self.support_slots(subtask, pick)
            self.report[subtask].append(
                f"{pick}: target **{tcat}/{rep['model']}** {self.slots_str(pick, 'target')} (pool {len(trows)}) + partner **{pcat}/{prep['model']}** "
                f"{self.slots_str(pick, 'partner')} (pool {len(prows)}); target: {'; '.join(notes) or 'ok'}; partner: {'; '.join(pnotes) or 'ok'}"
                f"{'; ' + '; '.join(snotes) if snotes else ''}")
        if chosen < n:
            self.report[subtask].append(f"only {chosen} of {n} picks: {len(tpool)} admissible target categories, {len(ppool)} partner categories, "
                                        f"{len(cands)} compatible pairs")

    def select_large_pairs(self, subtask: str, n: int, tpool: Dict[str, List[dict]], trs: dict) -> None:
        partners = self.partners_admitted(subtask)
        under = subtask.endswith("under_large_scale")
        # fill the target pool from the other tabletop pools (owner: fill missing candidates; validated later)
        have = set(tpool)
        for st in LARGE_TARGET_FILL_FROM:
            for r in self.by_subtask_role.get((st, "target"), []):
                cat = r["category"]
                if cat in have:
                    continue
                ok, _ = self.model_ok(r, trs)
                if not ok:
                    continue
                try:
                    h = float(r.get("bbox_z") or 0)
                except ValueError:
                    h = 0
                if under and h > UNDER_LARGE_MAX_HEIGHT:
                    continue
                if not under and h > 0.35:
                    continue
                rr = dict(r)
                rr["subtask"] = subtask
                rr["status"] = "incomplete:filled_from_" + st
                rr["representative"] = ""
                rr["test"] = ""
                tpool.setdefault(cat, [])
                if all(x["model"] != rr["model"] for x in tpool[cat]):
                    tpool[cat].append(rr)
                    ex = dict(rr)
                    ex["plan"] = ("large-scale IORS target filled from the %s pool (tabletop grasps, height %.2f m); validate with the "
                                  "representative test of ground_isg (place on the partner face / under the partner, whole-body IK)" % (st, h))
                    self.extra.append(ex)
        pcats = sorted({p["category"] for p in partners})
        scored = sorted(tpool.items(), key=lambda kv: (-self.category_score(kv[0], kv[1], trs), kv[0]))
        chosen = 0
        pc_used: collections.Counter = collections.Counter()
        for cat, rows in scored:
            if chosen >= n or not pcats:
                break
            pick = f"{subtask}#{chosen + 1:02d}"
            rep, notes = self.assign_slots(subtask, "target", cat, rows, trs, pick)
            if rep is None:
                continue
            # partner category: least used so far; representative partner = the instance in the test scene with the most
            # nearby supports, variation = another instance of the same category (other model preferred), held-out = another scene
            pcat = min(pcats, key=lambda c: (pc_used[c], pcats.index(c)))
            pc_used[pcat] += 1
            inst = [p for p in partners if p["category"] == pcat]
            inst.sort(key=lambda p: (p["scene"] != TEST_SCENE, -len(p["near_supports"]), -p["clearance"], -p["run"]))
            prep = inst[0]
            var = next((p for p in inst[1:] if p["model"] != prep["model"]), next((p for p in inst[1:] if p["scene"] == prep["scene"]), None))
            held = next((p for p in inst[1:] if p["scene"] != prep["scene"] and p is not var), None)
            pnotes = []
            if var is None:
                pnotes.append("VB-MOBJ partner model change: single instance of the category")
            elif var["model"] == prep["model"]:
                pnotes.append("VB-MOBJ partner model change: only same-model instances available")
            if held is None:
                pnotes.append("held-out partner: no instance in another scene")
            for slot, p in (("representative", prep), ("variation_geometry", var), ("heldout", held)):
                if p is None:
                    continue
                self.rows.append({"family": IORS, "subtask": subtask, "pick": pick, "role": "partner", "slot": slot, "category": p["category"],
                                  "model": p["model"], "scene": p["scene"], "name": p["name"], "joint": "", "kind": "scene", "grasp_key": "", "scale": "",
                                  "geometry_group": "", "exact_group": "", "status": "geometric_ok",
                                  "note": f"face_z={p['face_z']:.2f} clearance={p['clearance']:.2f} near_supports={','.join(p['near_supports']) or '-'}"})
            # support: a support within 3 m of the representative partner (init surface); under: the partner's own top
            # is the second choice (spec 5.6 b, owner decision A5 2026-09-16); floor is always the last class
            sup = self.supports_admitted()
            near = [s for s in sup if s["scene"] == prep["scene"] and s["name"] in prep["near_supports"]]
            snote = []
            if near:
                tables = ("breakfast_table", "desk", "coffee_table", "countertop", "bar", "nightstand", "console_table", "table")
                srep = sorted(near, key=lambda s: (s["category"] not in tables, -s["area"], -s["run"]))[0]
                self.rows.append({"family": IORS, "subtask": subtask, "pick": pick, "role": "support", "slot": "representative", "category": srep["category"],
                                  "model": "", "scene": srep["scene"], "name": srep["name"], "joint": "", "kind": "scene", "grasp_key": "", "scale": "",
                                  "geometry_group": "", "exact_group": "", "status": "geometric_ok", "note": f"init surface within 3 m of {prep['name']}"})
            else:
                partner_slots = {"representative": prep, "variation_geometry": var, "heldout": held}
                target_rows = [r for r in self.rows if r["pick"] == pick and r["role"] == "target"]
                own_rows, own_notes = self.own_top_rows(subtask, pick, partner_slots, target_rows) if under else ([], [])
                if own_rows:
                    snote.append("no admitted support within 3 m of the partner: init surface = the partner's own top (spec 5.6 b); " + "; ".join(own_notes))
                    self.rows.extend(own_rows)
                else:
                    snote.append("no admitted support within 3 m of the partner: init surface = floor"
                                 + ("; own top not admitted: " + "; ".join(own_notes) if own_notes else ""))
                    self.rows.append(self.floor_row(subtask, pick, "representative", prep))
            self.used_categories[cat] += 1
            chosen += 1
            self.report[subtask].append(
                f"{pick}: target **{cat}/{rep['model']}** {self.slots_str(pick, 'target')} (pool {len(rows)}"
                f"{', filled' if any(r['status'].startswith('incomplete') for r in rows) else ''}) "
                f"+ partner **{prep['category']} {prep['name']}@{prep['scene']}** {self.slots_str(pick, 'partner')}; target: {'; '.join(notes) or 'ok'}; "
                f"partner: {'; '.join(pnotes) or 'ok'}{'; ' + '; '.join(snote) if snote else ''}")
        if chosen < n:
            self.report[subtask].append(f"only {chosen} of {n} picks ({len(tpool)} target categories after filling, {len(pcats)} partner categories)")
        for line in self.fix_heldout_support_collisions(subtask):
            self.report[subtask].append(line)

    # -- large-scale articulation: pinned roomy units (owner decision 2026-09-16 06:50, item 6) ------------------------
    def artic_check(self, scene: str, name: str, joint: str) -> dict:
        """Verify one (scene, instance, joint) against the B4 inventory and ``require/handles.csv``: the joint exists (door /
        drawer / flap, B4 status handle / edge), the free run in front is >= ARTIC_FREE_MIN_M (``front_free_m``) with the
        robot-eroded run >= ARTIC_FREE_ERODED_MIN_M (flag ``eroded<3.0`` below ARTIC_FREE_MIN_M), and an admissible handle
        row exists for the joint (verdict handle / edge / manual, not reviewed ungraspable, pinch width 15-80 mm after the
        instance scale, world height 0.30-1.70 m). The pinch = the reviewer's ``chosen`` row, else handle before edge rows.
        Nothing is guessed: a joint without an admissible row is reported as such."""
        key = (scene, name, joint)
        if key in self._artic_checks:
            return self._artic_checks[key]
        a = (self.inventory.get(scene) or {}).get("artic", {}).get(name)
        out: dict = {"scene": scene, "name": name, "joint": joint, "ok": False, "reasons": [], "flags": [], "pinch": None,
                     "category": (a or {}).get("category", name.rsplit("_", 2)[0]), "model": (a or {}).get("model", ""),
                     "scale": (a or {}).get("scale"), "annotated": bool((a or {}).get("canonical_annotation")), "rooms": list((a or {}).get("rooms") or []),
                     "kind": "", "b4_status": "", "free": None, "free_eroded": None, "prior": None, "rows": [], "class": ""}
        if a is None:
            out["reasons"].append("instance not in the B4 artic inventory")
            self._artic_checks[key] = out
            return out
        j = next((x for x in a.get("joints") or [] if x.get("joint") == joint), None)
        if j is None:
            out["reasons"].append(f"joint {joint} not in the B4 record (joints: {', '.join(x.get('joint', '?') for x in a.get('joints') or [])})")
            self._artic_checks[key] = out
            return out
        out.update({"kind": j.get("kind", ""), "b4_status": j.get("status", ""), "free": j.get("front_free_m"), "free_eroded": j.get("front_free_eroded_m"),
                    "prior": j.get("prior_review"), "class": follow_class(out["category"], j.get("kind", ""))})
        if j.get("kind") not in ("door", "drawer", "flap"):
            out["reasons"].append(f"joint kind {j.get('kind')} (door / drawer / flap only)")
        if j.get("status") not in ("handle", "edge"):
            out["reasons"].append(f"B4 status {j.get('status')} (not graspable)")
        free, eroded = out["free"], out["free_eroded"]
        if free is None or float(free) < ARTIC_FREE_MIN_M:
            out["reasons"].append(f"free run in front {free if free is None else f'{float(free):.2f}'} m < {ARTIC_FREE_MIN_M:.1f} m")
        if eroded is None or float(eroded) < ARTIC_FREE_ERODED_MIN_M:
            out["reasons"].append(f"robot-eroded free run {eroded if eroded is None else f'{float(eroded):.2f}'} m < {ARTIC_FREE_ERODED_MIN_M:.1f} m (no view stance)")
        elif float(eroded) < ARTIC_FREE_MIN_M:
            out["flags"].append(f"eroded={float(eroded):.2f}")
        rows = self.handle_instance_rows.get(key, [])
        out["rows"] = rows
        if not rows:
            out["reasons"].append("no handles.csv row for the instance / joint (missing handle annotation)")
        admissible = []
        for h in rows:
            why = []
            if h.get("verdict") not in HANDLE_ADMISSIBLE_VERDICTS:
                why.append(f"verdict {h.get('verdict')}")
            if (h.get("review_graspable") or "") in ("False", "false", "0"):
                why.append("reviewed ungraspable")
            try:
                w = float(h.get("width_m") or "nan") * 1000
            except ValueError:
                w = float("nan")
            if not (ARTIC_PINCH_MM[0] <= w <= ARTIC_PINCH_MM[1]):
                why.append(f"pinch {w:.0f} mm" if w == w else "pinch width unknown")
            try:
                z = float(h.get("height_world_m") or "nan")
            except ValueError:
                z = float("nan")
            if not (ARTIC_HANDLE_Z[0] <= z <= ARTIC_HANDLE_Z[1]):
                why.append(f"height {z:.2f} m" if z == z else "height unknown")
            if not why:
                admissible.append((h, w, z))
        if rows and not admissible:
            seen = sorted({f"{h.get('kind')} {float(h['width_m']) * 1000:.0f}mm@{float(h['height_world_m']):.2f}m/{h.get('verdict')}"
                           for h in rows if h.get("width_m") and h.get("height_world_m")})
            out["reasons"].append("no admissible handle row: " + ", ".join(seen))
        if admissible:
            order = {"handle": 0, "manual": 1, "edge": 2}
            h, w, z = sorted(admissible, key=lambda t: ((t[0].get("chosen") or "") not in ("1", "True", "true"), order.get(t[0].get("verdict"), 9), t[0].get("name") or ""))[0]
            out["pinch"] = {"kind": h.get("kind", ""), "width_mm": w, "height": z, "verdict": h.get("verdict", ""), "row": h.get("name", ""),
                            "chosen": (h.get("chosen") or "") in ("1", "True", "true"), "n_rows": len(rows), "n_admissible": len(admissible)}
            if h.get("verdict") == "edge":
                out["flags"].append("edge pinch (the protruding handles fail the 15 mm pinch filter)")
        out["ok"] = not out["reasons"]
        self._artic_checks[key] = out
        return out

    def artic_row(self, subtask: str, pick: str, slot: str, c: dict) -> dict:
        p = c["pinch"] or {}
        note = (f"{c['kind']} {p.get('kind', '-')} h={p.get('height', 0.0):.2f} w={p.get('width_mm', 0.0):.0f}mm "
                f"front_free={float(c['free']):.2f}({float(c['free_eroded']):.2f}) prior={c['prior'] or '-'} pinch_row={p.get('row', '-')}({p.get('verdict', '-')}) "
                f"class={c['class']} rooms={','.join(c['rooms']) or '-'}")
        if c["flags"]:
            note += " flags=" + ";".join(c["flags"])
        return {"family": CM, "subtask": subtask, "pick": pick, "role": "target", "slot": slot, "category": c["category"], "model": c["model"],
                "scene": c["scene"], "name": c["name"], "joint": c["joint"], "kind": "scene", "grasp_key": "", "scale": "", "geometry_group": "",
                "exact_group": "", "status": "geometric_ok" if c["annotated"] else "incomplete:handle_from_B4_heuristic", "note": note, "triplet": ""}

    def select_scene_artic_pinned(self, subtask: str, n: int) -> None:
        """The owner's pinned large-scale articulation picks (:data:`ARTIC_LARGE_PICKS`): one pick per door class; per slot
        the first candidate that passes :meth:`artic_check`; failing candidates are reported, never replaced by a guess."""
        chosen = 0
        for pin in ARTIC_LARGE_PICKS:
            pick = f"{subtask}#{chosen + 1:02d}"
            slots: Dict[str, dict] = {}
            notes: List[str] = []
            for slot in SUPPORT_SLOTS:
                for scene, name, joint in pin.get(slot) or []:
                    c = self.artic_check(scene, name, joint)
                    if c["ok"]:
                        slots[slot] = c
                        break
                    notes.append(f"{slot} candidate {scene}/{name}:{joint} rejected - " + "; ".join(c["reasons"]))
            rep = slots.get("representative")
            if rep is None:
                self.report[subtask].append(f"{pin['class']} ({pin['category']}): no representative passed the checks - " + "; ".join(notes))
                continue
            var, held = slots.get("variation_geometry"), slots.get("heldout")
            if var is None:
                notes.append("VB-MOBJ target model change: no roomy second unit of the class" if not pin.get("variation_geometry")
                             else "VB-MOBJ target model change unavailable (candidates rejected)")
            elif var["model"] == rep["model"]:
                notes.append("VB-MOBJ: only same-model instances available for variation")
            if held is None:
                notes.append("held-out target: no roomy third unit" if not pin.get("heldout") else "held-out target unavailable (candidates rejected)")
            for slot, c in slots.items():
                self.rows.append(self.artic_row(subtask, pick, slot, c))
            p = rep["pinch"]
            self.report[subtask].append(
                f"{pick}: target **{rep['category']} {rep['name']}:{rep['joint']}@{rep['scene']}** ({rep['kind']}, {p['kind']} {p['width_mm']:.0f} mm at "
                f"{p['height']:.2f} m, free {float(rep['free']):.2f}({float(rep['free_eroded']):.2f}) m, {pin['class']}, follow {rep['class']}) "
                f"{self.slots_str(pick, 'target')}; {'; '.join(notes) or 'all axes available'}")
            chosen += 1
        if chosen < n:
            self.report[subtask].append(f"{chosen} of {n} picks: the owner's class list (2026-09-16 item 6: cabinets, a fridge, top cabinets, the oven, drawers) "
                                        "has five classes; the heuristic fill is off for the pinned subtasks")

    def select_scene_artic(self, subtask: str, n: int) -> None:
        if ARTIC_LARGE_PICKS:
            self.select_scene_artic_pinned(subtask, n)
            return
        arts = self.artic_admitted()
        # one pick per (category, joint kind) preferring the test scene, handle over edge, canonical annotation, reviewer-neutral
        arts.sort(key=lambda a: (a["scene"] != TEST_SCENE, a["handle_kind"] not in ("bar", "knob"), not a["annotated"], -(a["front_free"] or 0)))
        by_cat: Dict[str, List[dict]] = collections.defaultdict(list)
        for a in arts:
            by_cat[a["category"]].append(a)
        # category order: most instances (variation + held-out possible) first, doors / fridges / cabinets are the natural picks
        order = sorted(by_cat, key=lambda c: (-min(len({(a['scene'], a['name']) for a in by_cat[c]}), 3), -len(by_cat[c]), c))
        chosen = 0
        used_names: Set[Tuple[str, str]] = set()
        for cat in order:
            if chosen >= n:
                break
            inst = [a for a in by_cat[cat] if (a["scene"], a["name"]) not in used_names]
            if not inst:
                continue
            rep = inst[0]
            others = [a for a in inst if (a["scene"], a["name"]) != (rep["scene"], rep["name"])]
            var = next((a for a in others if a["model"] != rep["model"]), next((a for a in others), None))
            held = next((a for a in others if a is not var and (var is None or (a["scene"], a["name"]) != (var["scene"], var["name"]))), None)
            pick = f"{subtask}#{chosen + 1:02d}"
            notes = []
            if var is None:
                notes.append("VB-MOBJ target model change: single admitted instance of the category")
            elif var["model"] == rep["model"]:
                notes.append("VB-MOBJ: only same-model instances (scaled copies) available for variation")
            if held is None:
                notes.append("held-out target: no third instance")
            for slot, a in (("representative", rep), ("variation_geometry", var), ("heldout", held)):
                if a is None:
                    continue
                used_names.add((a["scene"], a["name"]))
                self.rows.append({"family": CM, "subtask": subtask, "pick": pick, "role": "target", "slot": slot, "category": a["category"], "model": a["model"],
                                  "scene": a["scene"], "name": a["name"], "joint": a["joint"], "kind": "scene", "grasp_key": "", "scale": "",
                                  "geometry_group": "", "exact_group": "",
                                  "status": "geometric_ok" if a["annotated"] else "incomplete:handle_from_B4_heuristic",
                                  "note": f"{a['kind']} {a['handle_kind']} h={a['height']:.2f} w={a['width']}mm front_free={a['front_free']} prior={a['prior'] or '-'}"})
            chosen += 1
            self.report[subtask].append(f"{pick}: target **{cat} {rep['name']}:{rep['joint']}@{rep['scene']}** ({rep['kind']}, {rep['handle_kind']}) "
                                        f"{self.slots_str(pick, 'target')}; {'; '.join(notes) or 'all axes available'}")
        if chosen < n:
            self.report[subtask].append(f"only {chosen} of {n} picks: {len(by_cat)} categories with an admitted (handle/edge) joint")

    def select_blocks(self, subtask: str) -> None:
        rows = self.by_subtask_role.get((subtask, "targets"), [])
        pick = f"{subtask}#01"
        for r in rows:
            self.rows.append(self.row(subtask, pick, "targets", "representative", r, {}, note="Kohs block (ispatialgym-assets); 4 identical members"))
        self.report[subtask].append(f"{pick}: targets **toy_dice/isgblk** x4 (only 1 block model exists; the 2 further block designs are pending) — "
                                    "VB-MOBJ / VSB-NOBJ unavailable (X in the table); " + "; ".join(self.support_slots(subtask, pick)))

    # -- driver ---------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        for family in FAMILY_ORDER:
            subs = [s for s, sp in SUBTASKS.items() if sp["family"] == family]
            with_objects = [s for s in subs if SUBTASKS[s]["roles"] and SUBTASKS[s].get("n", 1) != 0]
            for st in subs:
                spec = SUBTASKS[st]
                roles = spec["roles"]
                if not roles:
                    self.report[st].append(spec.get("note", ""))
                    continue
                n = self.n_for(st, len([s for s in with_objects if SUBTASKS[s].get("n") != 1]))
                if st == f"{CM}-block_design":
                    self.select_blocks(st)
                elif "partner" in roles:
                    self.select_pairs(st, n)
                elif roles.get("target", {}).get("kind") == "scene_artic":
                    self.select_scene_artic(st, n)
                else:
                    role = "obstacles" if "obstacles" in roles else "target"
                    sup = "wall" if roles.get("support", {}).get("kind") == "wall" else ("scene" if "support" in roles else None)
                    self.select_single(st, role, roles[role], n, sup)

    # -- targeted patches (--patch-only) ---------------------------------------------------------------------------------
    def patch(self, out_dir: str, what: Set[str]) -> None:
        """Apply the owner's targeted patches to the existing ``representatives.csv`` under ``out_dir`` (no re-selection):
        ``under_large_scale`` - the own-top branch for the under large-scale picks whose init surface is the floor (A5);
        ``support_triplets`` - the T2..TN support rows of every tabletop pick (A3). Other rows are copied unchanged."""
        unknown = what - set(PATCH_KINDS)
        if unknown:
            raise SystemExit(f"--patch-only: unknown patch {sorted(unknown)} (known: {', '.join(PATCH_KINDS)})")
        existing = read_csv(os.path.join(out_dir, "representatives.csv"))
        picks: List[Tuple[str, List[dict]]] = []
        for r in existing:  # consecutive runs of one pick (the selection writes the rows of a pick together)
            if picks and picks[-1][0] == r["pick"]:
                picks[-1][1].append(r)
            else:
                picks.append((r["pick"], [r]))
        artic_new: Dict[str, List[dict]] = {}
        if "articulation_large_scale" in what:
            # re-select the two subtasks from the pins (self-contained: no global selection state involved)
            saved = self.rows
            for st in ARTIC_LARGE_SUBTASKS:
                self.rows = []
                self.report[st] = []
                self.select_scene_artic(st, self.n_for(st, 5))
                artic_new[st] = self.rows
            self.rows = saved
        out: List[dict] = []
        artic_done: Set[str] = set()
        for pick, prs in picks:
            subtask = prs[0]["subtask"]
            if subtask in artic_new:
                if subtask not in artic_done:  # the new picks take the place of the old block (in order)
                    artic_done.add(subtask)
                    old = [f"{r['pick']}/{r['slot']} {r['name']}:{r['joint']}" for p2, rs in picks if p2.split('#')[0] == subtask for r in rs]
                    new = [f"{r['pick']}/{r['slot']} {r['name']}:{r['joint']}" for r in artic_new[subtask]]
                    self.patch_log["articulation_large_scale"].append(
                        f"{subtask}: {len({r['pick'] for p2, rs in picks if p2.split('#')[0] == subtask for r in rs})} picks / {len(old)} rows -> "
                        f"{len({r['pick'] for r in artic_new[subtask]})} picks / {len(new)} rows; removed: " + ", ".join(old) + "; added: " + ", ".join(new))
                    out.extend(artic_new[subtask])
                continue
            if "under_large_scale" in what and subtask == f"{IORS}-under_large_scale":
                prs = self.patch_own_top(pick, prs)
            if "support_triplets" in what and subtask in TABLETOP_SUPPORT_SUBTASKS:
                prs = self.patch_triplets(pick, prs)
            out.extend(prs)
        for st, rows in artic_new.items():
            if st not in artic_done:
                out.extend(rows)
                self.patch_log["articulation_large_scale"].append(f"{st}: {len(rows)} rows appended (the sheet had none)")
        self.rows = out
        if "under_large_scale" in what:
            for line in self.fix_heldout_support_collisions(f"{IORS}-under_large_scale"):
                self.patch_log["under_large_scale"].append(line)
                pick = line.split(":", 1)[0]
                if f"line:{pick}" in self.patch_log:
                    self.patch_log[f"line:{pick}"][0] += "; " + line.split(": ", 1)[1]

    def patch_own_top(self, pick: str, prs: List[dict]) -> List[dict]:
        subtask = prs[0]["subtask"]
        sup_rows = [r for r in prs if r["role"] == "support"]
        if not (len(sup_rows) == 1 and sup_rows[0]["category"] == FLOOR):
            self.patch_log["under_large_scale"].append(f"{pick}: unchanged (init surface {', '.join(r['name'] for r in sup_rows) or '-'})")
            return prs
        partner_slots: Dict[str, Optional[dict]] = {}
        for r in prs:
            if r["role"] == "partner" and r["kind"] == "scene":
                near = ((self.inventory.get(r["scene"]) or {}).get("partners", {}).get(r["name"]) or {}).get("supports_within_3m") or []
                partner_slots[r["slot"]] = {"scene": r["scene"], "name": r["name"], "category": r["category"],
                                            "near_supports": [n.split(" (")[0] for n in near]}
        if "representative" not in partner_slots:
            self.patch_log["under_large_scale"].append(f"{pick}: unchanged (no scene partner rows)")
            return prs
        target_rows = [r for r in prs if r["role"] == "target"]
        own_rows, notes = self.own_top_rows(subtask, pick, partner_slots, target_rows)
        if not own_rows:
            self.patch_log["under_large_scale"].append(f"{pick}: unchanged, own top not admitted - " + "; ".join(notes))
            return prs
        idx = prs.index(sup_rows[0])
        old = sup_rows[0]
        self.patch_log["under_large_scale"].append(
            f"{pick}: support representative `{old['name']}@{old['scene']}` ({old['category']}) -> "
            + ", ".join(f"{r['slot']}=`{r['name']}@{r['scene']}` ({r['category']})" for r in own_rows) + "; " + "; ".join(notes))
        self.patch_log[f"line:{pick}"] = ["no admitted support within 3 m of the partner: init surface = the partner's own top (spec 5.6 b; A5 patch 2026-09-16); "
                                          + "; ".join(notes)]
        return prs[:idx] + own_rows + prs[idx + 1:]

    def patch_triplets(self, pick: str, prs: List[dict]) -> List[dict]:
        subtask = prs[0]["subtask"]
        t1 = [r for r in prs if r["role"] == "support" and r["kind"] == "scene" and r["category"] not in ("walls", FLOOR)]
        if not t1 or any((r.get("triplet") or "") not in ("", "1") for r in prs):
            self.patch_log["support_triplets"].append(f"{pick}: unchanged (no T1 support rows or triplets already present)")
            return prs
        triplets = self.support_triplets()
        new: List[dict] = []
        for t in triplets[1:]:
            for slot in SUPPORT_SLOTS:
                s = t["slots"].get(slot)
                if s is not None:
                    new.append(self.support_row(subtask, pick, slot, s, t["triplet"]))
        for r in t1:
            r["triplet"] = "1"
        idx = prs.index(t1[0])
        self.patch_log["support_triplets"].append(f"{pick}: +{len(new)} rows (T{', T'.join(str(t['triplet']) for t in triplets[1:])}) before the T1 rows")
        return prs[:idx] + new + prs[idx:]

    # -- support candidates the icl lacks -------------------------------------------------------------------------------
    def ensure_support_candidates(self) -> List[dict]:
        """``candidates_extra.csv`` support rows (kind ``scene``; scene / instance / room columns as the existing support rows)
        for every scene support of the selection that ``roles/candidates.csv`` does not list for the subtask (any status)."""
        added: List[dict] = []
        have = {(r["subtask"], r["role"], r["scene"], r["model"]) for r in self.cands if r.get("kind") == "scene"}
        have |= {(r["subtask"], r["role"], r["scene"], r["model"]) for r in self.extra if r.get("kind") == "scene"}
        for r in self.rows:
            if r["role"] != "support" or r["kind"] != "scene" or r["category"] in ("walls", FLOOR):
                continue
            key = (r["subtask"], "support", r["scene"], r["name"])
            if key in have:
                continue
            have.add(key)
            adm = self.support_admission(r["scene"], r["name"])
            rec = self.support_record(r["scene"], r["name"]) or {}
            aabb = rec.get("aabb_world") or {}
            lo, hi = aabb.get("min"), aabb.get("max")
            row = {k: "" for k in self.cands[0].keys()}
            row.update({"subtask": r["subtask"], "role": "support", "kind": "scene", "category": r["category"], "model": r["name"],
                        "display_name": r["category"].replace("_", " "), "scene": r["scene"], "room": ",".join(adm["rooms"]), "rank": "4",
                        "status": "incomplete:support_from_B4_inventory", "top_z": f"{adm['top_z']:.3f}" if adm["top_z"] else "",
                        "world_ext_x": f"{hi[0] - lo[0]:.3f}" if lo and hi else "", "world_ext_y": f"{hi[1] - lo[1]:.3f}" if lo and hi else "",
                        "pos_x": f"{(lo[0] + hi[0]) / 2:.3f}" if lo and hi else "", "pos_y": f"{(lo[1] + hi[1]) / 2:.3f}" if lo and hi else "",
                        "note": f"B4: top z {adm['top_z']:.2f}, area {adm['area']:.2f} m2; usable edges="
                                + ",".join(f"{e['name']}({e['usable']:.2f})" for e in adm["edges"] if e["usable"] > 0)
                                + f"; margin_row={adm['longest_row'] or 'none'}" + ("; flags=" + ";".join(adm["flags"]) if adm["flags"] else ""),
                        "source": f"scene_inventory_20260915/{r['scene']}/supports/{r['name']}.json",
                        "plan": "support selected by select_representatives (A3 triplet / A5 own top) that roles/candidates.csv does not list for "
                                "this subtask; validate with the sampler's stance search (rejections -> customs blacklist per (scene, support))"})
            self.extra.append(row)
            added.append(row)
        return added

    # -- output ---------------------------------------------------------------------------------------------------------
    def write(self, out_dir: str, patch: bool = False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "representatives.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REP_COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in REP_COLUMNS})
        cols = list(self.cands[0].keys()) + ["plan"]
        extra_path = os.path.join(out_dir, "candidates_extra.csv")
        write_extra = True
        if patch:
            # the filled candidates of the last full run stay as they are; only support rows the icl lacks are appended
            self.extra = read_csv(extra_path) if os.path.exists(extra_path) else []
            added = self.ensure_support_candidates()
            self.patch_log["candidates_extra"].append(f"{len(added)} support rows appended" if added
                                                      else "no support row needed (candidates.csv lists every selected support)")
            write_extra = bool(added)
        else:
            self.ensure_support_candidates()
        if write_extra:
            with open(extra_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                seen = set()
                for r in self.extra:
                    key = (r["subtask"], r["role"], r["category"], r["model"])
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow({k: r.get(k, "") for k in cols})
        self.write_report(os.path.join(out_dir, "representatives_report.md"), patch=patch)

    def support_report_lines(self) -> List[str]:
        """The two support sections: the triplets (A3) and the per-support admission table (edge length vs §4.4 margins)."""
        triplets = self.support_triplets()
        tabletop_picks = sorted({r["pick"] for r in self.rows if r["subtask"] in TABLETOP_SUPPORT_SUBTASKS and r["role"] == "support"
                                 and r["kind"] == "scene" and r["slot"] == "representative"})
        with_t = collections.Counter(r.get("triplet") or "" for r in self.rows if r["subtask"] in TABLETOP_SUPPORT_SUBTASKS and r["role"] == "support"
                                     and r["slot"] == "representative")
        lines = ["## Support triplets (owner decision 2026-09-16, A3)", "",
                 f"Every §4 tabletop pick ({len(TABLETOP_SUPPORT_SUBTASKS)} subtasks, {len(tabletop_picks)} picks) carries the same "
                 f"{len(triplets)} support triplets in its `support` role (`triplet` column; rows per pick: "
                 + ", ".join(f"T{t}: {n}" for t, n in sorted(with_t.items()) if t) + "). "
                 "T1 = the 2026-09-15 rule (test pin `breakfast_table_rhjoby_0`, same-scene height variation, other-scene held-out), "
                 "T2 / T3 = the owner's pins (TODO.md A3), each verified against the B4 inventory: exists, top height in the tabletop range "
                 f"({SUPPORT_TOP_Z[0]:.2f}-{SUPPORT_TOP_Z[1]:.2f} m), best usable edge >= the longest §4.4 margin row the pilot measured "
                 f"({LONGEST_NEED_M:.2f} m, see the admission table). Held-out supports are **global**: held out in every tabletop subtask "
                 "(`ground_isg` excludes a held-out scene object from every other combination of the subtask). Within a pick the rows are "
                 "written T2, T3, T1 so that readers indexing (pick, role, slot) without the `triplet` column keep the T1 binding; the T1 rows "
                 "are unchanged (summary best edge in the note), T2 / T3 notes carry the usable edge, the longest admissible row and the flags.", "",
                 "| triplet | slot | scene / support | category | rooms | top z | best usable edge (m, 0.45 run 0.35/0.45/0.55) | longest admissible row | status |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for t in triplets:
            for slot in SUPPORT_SLOTS:
                s = t["slots"].get(slot)
                if s is None:
                    lines.append(f"| T{t['triplet']} | {slot} | - | | | | | | unavailable |")
                    continue
                adm = self.support_admission(s["scene"], s["name"])
                best = adm["best"]
                edge = (f"{best['name']} {adm['usable_m']:.2f} ({best['runs']['0.35']:.2f}/{best['runs']['0.45']:.2f}/{best['runs']['0.55']:.2f})"
                        if best else "none")
                status = "ok" if adm["longest_row"] in LONG_ROWS else ("short rows only" if adm["longest_row"] else "no usable edge")
                other_flags = [f for f in adm["flags"] if f != "short_rows_only"]
                if other_flags:
                    status += " [" + "; ".join(other_flags) + "]"
                lines.append(f"| T{t['triplet']} | {slot} | `{s['scene']}/{s['name']}` | {s['category']} | {', '.join(adm['rooms']) or '-'} | "
                             f"{adm['top_z']:.2f} | {edge} | {adm['longest_row'] or 'none'} | {status} |")
        lines.append("")
        notes = [n for t in triplets for n in t["notes"]] + self.support_notes
        for pin in SUPPORT_TRIPLETS:
            alt = pin.get("heldout_alt")
            if alt and not any(s["scene"] == alt[0] and s["name"] == alt[1] for t in triplets for s in t["slots"].values()):
                adm = self.support_admission(*alt)
                notes.append(f"alternative held-out `{alt[0]}/{alt[1]}` checked, not needed: best usable edge {adm['usable_m']:.2f} m -> "
                             f"longest admissible row {adm['longest_row'] or 'none'}" + (f", flags {'; '.join(adm['flags'])}" if adm["flags"] else ""))
        for i, t in enumerate(triplets):
            for j, u in enumerate(triplets):
                if j <= i:
                    continue
                for sa, a in t["slots"].items():
                    for sb, b in u["slots"].items():
                        if a["category"] == b["category"]:
                            notes.append(f"category `{a['category']}` appears in T{t['triplet']} {sa} (`{a['name']}`) and T{u['triplet']} {sb} "
                                         f"(`{b['name']}`): category-based variant names need the triplet / instance to stay distinct")
        if any(len(t["slots"]) >= 2 for t in triplets):
            for t in triplets:
                r, v = t["slots"].get("representative"), t["slots"].get("variation_geometry")
                if r and v:
                    notes.append(f"T{t['triplet']} VB-ISC height change: {r['top_z']:.2f} -> {v['top_z']:.2f} m ({abs(v['top_z'] - r['top_z']) * 100:.0f} cm)")
        for n in notes:
            lines.append(f"- {n}")
        # per-support admission table: every scene support of the selection + the alternative pins
        lines += ["", "## Support admission: edge length vs §4.4 margins", "",
                  "Per edge the usable length is min(edge length, longest free-floor run at stand-off 0.45 with the chairs within 1 m removed); "
                  f"an edge is unusable when an object stands closer than {BLOCKED_IN_FRONT_MAX_M:.2f} m in front of its midpoint (B4 ray at top + 0.3 m; the "
                  "free-floor run does not see walls between the edge and the stand-off line, so the run of a wall-facing edge is the floor of the "
                  f"next room) or its run is below the {EDGE_FREE_RUN_MIN_M:.2f} m admission. The longest §4.4 row (m_R + m_L) the best usable edge "
                  "admits is recorded; supports below both long rows are flagged `short_rows_only`. Rows:", "",
                  "| row | m_R + m_L (m) | source | subtasks | derivation |", "|---|---|---|---|---|"]
        for r in MARGIN_ROWS:
            lines.append(f"| {r['id']} | {r['need']:.2f} | {r['how']} | {', '.join(s.replace(IORS, 'IORS').replace(CM, 'CM') for s in r['subtasks'])} | {r['note']} |")
        lines += ["", f"Band span abs(lambda_min) + max(0, lambda_max) = {BAND_SPAN_M:.2f} m [pilot] (back-solved from the toolbox and the platter rows). "
                  "The estimate rows use the picked objects' bboxes; the sampler's stance search stays the validation.", "",
                  "| scene / support | category | rooms | top z | area m² | edges: usable m @0.45 (runs 0.35/0.45/0.55; room in front) | blocked edges | best usable | longest admissible row | flags | used as |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        used: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        for r in self.rows:
            if r["role"] == "support" and r["kind"] == "scene" and r["category"] not in ("walls", FLOOR):
                tag = (f"T{r['triplet']} " if r.get("triplet") else f"{r['subtask'].replace(IORS, 'IORS')} ") + r["slot"]
                if tag not in used[(r["scene"], r["name"])]:
                    used[(r["scene"], r["name"])].append(tag)
        for pin in SUPPORT_TRIPLETS:
            alt = pin.get("heldout_alt")
            if alt and alt not in used:
                used[alt].append("alternative held-out (not used)")
        for (scene, name), tags in sorted(used.items()):
            adm = self.support_admission(scene, name)
            edges = "; ".join(f"{e['name']} {e['usable']:.2f} ({e['runs']['0.35']:.2f}/{e['runs']['0.45']:.2f}/{e['runs']['0.55']:.2f}; {e['room'] or '-'})"
                              + (f" overhang {e['overhang']:.2f}" if e["overhang"] >= OVERHANG_FLAG_M else "")
                              for e in adm["edges"] if e["usable"] > 0) or "none"
            lines.append(f"| `{scene}/{name}` | {adm['category']} | {', '.join(adm['rooms']) or '-'} | {adm['top_z']:.2f} | {adm['area']:.2f} | {edges} | "
                         f"{', '.join(adm['blocked']) or '-'} | {adm['usable_m']:.2f} | {adm['longest_row'] or 'none'} | {'; '.join(adm['flags']) or '-'} | "
                         f"{', '.join(tags)} |")
        return lines

    def artic_report_lines(self) -> List[str]:
        """The large-scale articulation section: the pinned units with the free-run and handle checks (item 6)."""
        if not ARTIC_LARGE_PICKS:
            return []
        lines = ["## Large-scale articulation picks (owner decision 2026-09-16 06:50, item 6)", "",
                 "The crowded kitchen picks (`bottom_cabinet_fancyy_0`: 1.50 m handle next to `oven_ffitak_0`, 0.95 m robot-eroded free run; "
                 "`fridge_dszchb_0` 1.0 m; `dishwasher_ynrhuk_0` 1.0 m) are replaced by roomy B4 units, one pick per door class, shared by the open / "
                 f"close subtasks. Checks per (scene, instance, joint): B4 free run in front >= {ARTIC_FREE_MIN_M:.1f} m (`front_free_m`, the un-eroded map "
                 f"Prompt-1 Phase 1 rule 1 uses) with the robot-eroded run >= {ARTIC_FREE_ERODED_MIN_M:.1f} m (proxy for the whole-body check at the view "
                 f"stance; flagged below {ARTIC_FREE_MIN_M:.1f} m); a `require/handles.csv` row for the instance / joint with verdict handle / edge / manual, "
                 f"not reviewed ungraspable, pinch width {ARTIC_PINCH_MM[0]:.0f}-{ARTIC_PINCH_MM[1]:.0f} mm after the instance scale and world height "
                 f"{ARTIC_HANDLE_Z[0]:.2f}-{ARTIC_HANDLE_Z[1]:.2f} m (the reviewer's `chosen` row, else handle before edge rows). Units that fail are "
                 "reported, not replaced by guesses. Follow class per owner decision A4 (doors / flaps of cabinets and appliances: arm-only from the "
                 "hinge side; drawers: base mode; room doors: hybrid). Room doors (A4: hybrid follow after the pilot), `bottom_cabinet_no_top_gjrero_*` "
                 "(same-model copies, 1.7 m) and the dishwashers (kitchen, 1.0 / 2.35 m) are no longer picked in these two subtasks.", "",
                 "| class | slot | scene / instance | joint | kind | follow | free m (eroded) | handle row (kind, width, height, verdict) | canonical annotation | verdict |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for pin in ARTIC_LARGE_PICKS:
            for slot in SUPPORT_SLOTS:
                cands = pin.get(slot) or []
                if not cands:
                    lines.append(f"| {pin['class']} | {slot} | - | | | | | | | no roomy candidate in the inventory |")
                taken = False
                for scene, name, joint in cands:
                    c = self.artic_check(scene, name, joint)
                    p = c["pinch"]
                    free = (f"{float(c['free']):.2f} ({float(c['free_eroded']):.2f})" if c["free"] is not None and c["free_eroded"] is not None else "-")
                    hrow = (f"{p['row']}: {p['kind']} {p['width_mm']:.0f} mm @ {p['height']:.2f} m, {p['verdict']}" + (" (chosen)" if p["chosen"] else "")
                            if p else ("none admissible" if c["rows"] else "no row"))
                    if c["ok"] and not taken:
                        verdict, taken = "ok" + (" [" + "; ".join(c["flags"]) + "]" if c["flags"] else ""), True
                    elif c["ok"]:
                        verdict = "ok, not needed"
                    else:
                        verdict = "rejected: " + "; ".join(c["reasons"])
                    lines.append(f"| {pin['class']} | {slot} | `{scene}/{name}` | {joint} | {c['kind'] or '-'} | {c['class'] or '-'} | {free} | {hrow} | "
                                 f"{'yes' if c['annotated'] else 'B4 heuristic'} | {verdict} |")
        lines += ["", "- The two `bottom_cabinet` picks (cabinet doors / drawers) share the category-based variant token; the follow class differs per joint "
                  "kind (A4), so the grounder / sampler must keep the class per bound object (or split the variant by joint kind).",
                  "- `bottom_cabinet_rvunhj_*`: the protruding handles are 7-8 mm wide after the 1.6x instance scale (below the 15 mm pinch filter); the "
                  "admissible pinch is the exposed door-leaf edge (`j_link_2#F`, 39 mm, verdict `edge`); the other three doors of each unit have no "
                  "exposed edge (`edge_blocked`) and `rvunhj_0` j_link_0 / j_link_1 have 0.0 m eroded free run.",
                  "- Drawers: no drawer unit in the inventory reaches 3.0 m robot-eroded free run; `bottom_cabinet_bamfsz_0` (3.00 / 2.30 m, canonical "
                  "annotation, 19-20 mm rail pulls) is the best; `j_link_4` is its top drawer (0.82 m; j_link_1-3 at 0.38 / 0.60 / 0.71 m are alternatives)."]
        return lines

    def patch_report(self, text: str) -> str:
        """Splice the support / articulation sections and the patched per-subtask lines into an existing report (``--patch-only``)."""
        lines = text.rstrip("\n").split("\n")
        out: List[str] = []
        for line in lines:
            for key, val in self.patch_log.items():
                if key.startswith("line:") and line.startswith(f"- {key[5:]}: "):
                    line = line.rsplit("; no admitted support within 3 m of the partner", 1)[0] + "; " + val[0]
            if line.startswith("- **Supports** (`geometric_ok` only)"):
                line = self.supports_validation_line()
            if line.startswith("- **Scene articulated targets without a canonical handle annotation**") and self.patch_log.get("articulation_large_scale"):
                line = self.artic_validation_line()
            out.append(line)
        # re-selected subtasks (articulation_large_scale): their per-subtask blocks are regenerated
        if self.patch_log.get("articulation_large_scale"):
            for st in ARTIC_LARGE_SUBTASKS:
                head = f"### {st}  ("
                if not any(l.startswith(head) for l in out):
                    continue
                i = next(i for i, l in enumerate(out) if l.startswith(head))
                j = next((k for k in range(i + 1, len(out)) if out[k].startswith("### ") or out[k].startswith("## ")), len(out))
                n = len({r["pick"] for r in self.rows if r["subtask"] == st and r["role"] == "target" and r["slot"] == "representative"})
                block = [f"### {st}  ({n} picks)", ""] + [f"- {l}" for l in self.report.get(st, [])] + [""]
                out = out[:i] + block + out[j:]
        # drop a previous version of the support / articulation sections, then insert the fresh ones before the held-out list
        for head in ("## Support triplets", "## Support admission", "## Large-scale articulation picks"):
            if any(l.startswith(head) for l in out):
                i = next(i for i, l in enumerate(out) if l.startswith(head))
                j = next((k for k in range(i + 1, len(out)) if out[k].startswith("## ")), len(out))
                out = out[:i] + out[j:]
        anchor = next((i for i, l in enumerate(out) if l.startswith("## Held-out models (global)")), len(out))
        patched = ["", "## Targeted patches (`--patch-only`)", "",
                   "Applied to the 2026-09-15 selection without re-running it (a re-run on the rebuilt icl shifts the teacup picks and the global "
                   "held-out `cpozxi` -> `wopjex`, TODO.md A5):", ""]
        for kind in PATCH_KINDS + ("candidates_extra",):
            entries = self.patch_log.get(kind, [])
            if kind == "support_triplets":  # one line per outcome instead of one per pick
                by_outcome: Dict[str, List[str]] = collections.defaultdict(list)
                for entry in entries:
                    pick, outcome = entry.split(": ", 1)
                    by_outcome[outcome].append(pick)
                for outcome, pk in by_outcome.items():
                    subs = sorted({p.rsplit("#", 1)[0] for p in pk})
                    patched.append(f"- {kind}: {len(pk)} picks {outcome} ({len(subs)} subtasks: {', '.join(subs)})")
                continue
            for entry in entries:
                patched.append(f"- {kind}: {entry}")
        if any(l.startswith("## Targeted patches") for l in out):
            i = next(i for i, l in enumerate(out) if l.startswith("## Targeted patches"))
            j = next((k for k in range(i + 1, len(out)) if out[k].startswith("## ")), len(out))
            out = out[:i] + out[j:]
            anchor = next((i for i, l in enumerate(out) if l.startswith("## Held-out models (global)")), len(out))
        artic = self.artic_report_lines()
        out = out[:anchor] + self.support_report_lines() + ([""] + artic if artic else []) + patched + [""] + out[anchor:]
        return "\n".join(out) + "\n"

    @staticmethod
    def artic_validation_line() -> str:
        return ("- **Scene articulated targets without a canonical handle annotation** (`incomplete:handle_from_B4_heuristic`, e.g. bottom_cabinet_rvunhj_0 / _5 "
                "(exposed door-edge pinch), top_cabinet_tynnnw_1, oven_amblrk_0): the B4 geometric handle proposal is used through the `handle_file` role option; "
                "validate with the C3 review round on `runlogs/scene_inventory_20260915/<scene>/artic/strips/` (verdicts -> `require/handles.csv`), then the "
                "representative test.")

    @staticmethod
    def supports_validation_line() -> str:
        return ("- **Supports** (`geometric_ok` only): admitted by top height, area, short side and a >= 0.60 m free-floor edge run at stand-off 0.45 "
                "(0.35 / 0.55 in the note); the edge-length-vs-margin check (table above) records the longest §4.4 row each support admits and flags "
                "`short_rows_only`; the sampler's stance search is the validation (rejections `no_support` / `no_stance` -> customs blacklist per (scene, support)).")

    def write_report(self, path: str, patch: bool = False) -> None:
        if patch:
            with open(path) as f:
                text = f.read()
            with open(path, "w") as f:
                f.write(self.patch_report(text))
            return
        picks = collections.Counter()
        for r in self.rows:
            if r["slot"] == "representative" and r["role"] in ("target", "targets", "obstacles"):
                picks[(r["family"], r["subtask"])] += 1
        lines = ["# Representative selection (owner request 2026-09-15)", "",
                 "Generated by `scripts/icl/select_representatives.py` from the icl; deterministic. Slots per pick and role: "
                 "`representative` (training / pipeline test), `variation_texture` (V-OBJ texture variant, same geometry), "
                 "`variation_geometry` (VB-MOBJ / VB-ISC different geometry, same category), `heldout` (generalization; global per category, "
                 "never used in any other role). Category-change axes (VSB-NOBJ) are covered by the other picks of the same subtask.", "",
                 "## Global rules", "",
                 f"- {len(self.heldout)} held-out models (one per category, global): the same model is held out in every subtask that uses the category "
                 "and is excluded from partner / distractor / obstacle draws (ground_isg blacklists them with reason `heldout`).",
                 f"- {len(self.used_models)} distinct representative / variation models; {sum(1 for c, n in self.used_categories.items() if n > 1)} categories "
                 "re-used across subtasks (listed below).", "",
                 "## Picks per family", ""]
        for fam in FAMILY_ORDER:
            n = sum(v for (f, _), v in picks.items() if f == fam)
            lines.append(f"- **{fam}**: {n} picks")
        lines += ["", "## Per subtask", ""]
        for st in SUBTASKS:
            lines.append(f"### {st}  ({picks[(SUBTASKS[st]['family'], st)]} picks)")
            lines.append("")
            for l in self.report.get(st, []):
                lines.append(f"- {l}")
            lines.append("")
        lines += self.support_report_lines()
        artic = self.artic_report_lines()
        if artic:
            lines += [""] + artic
        lines += ["", "## Held-out models (global)", ""]
        for cat in sorted(self.heldout):
            lines.append(f"- {cat}: `{self.heldout[cat]}`")
        reused = sorted((c for c, n in self.used_categories.items() if n > 1), key=lambda c: -self.used_categories[c])
        lines += ["", "## Categories used by more than one subtask", ""]
        for c in reused:
            lines.append(f"- {c}: {self.used_categories[c]} picks")
        # slot availability per subtask (picks with a variation_geometry / variation_texture / heldout slot on the main role)
        lines += ["", "## Slot availability per subtask (main role)", "", "| subtask | picks | var. texture | var. geometry | held-out |", "|---|---|---|---|---|"]
        for st in SUBTASKS:
            main = "obstacles" if "obstacles" in SUBTASKS[st]["roles"] else ("targets" if "targets" in SUBTASKS[st]["roles"] else "target")
            pk = {r["pick"] for r in self.rows if r["subtask"] == st and r["role"] == main and r["slot"] == "representative"}
            if not pk:
                continue
            cnt = {s: len({r["pick"] for r in self.rows if r["subtask"] == st and r["role"] == main and r["slot"] == s}) for s in ("variation_texture", "variation_geometry", "heldout")}
            lines.append(f"| {st} | {len(pk)} | {cnt['variation_texture']} | {cnt['variation_geometry']} | {cnt['heldout']} |")
        lines += ["", "## Filled (incomplete) candidates and the plan to validate them", "",
                  f"{len({(r['subtask'], r['category'], r['model']) for r in self.extra})} rows in `candidates_extra.csv` (status `incomplete:*`, see the `plan` column). "
                  "What is still unvalidated and how it gets validated:", "",
                  "- **Large-scale IORS targets** (`incomplete:filled_from_*`): tabletop-grasp models of the other IORS / disembedding pools, filtered by height "
                  "(ontop <= 0.35 m, under <= 0.20 m). Validate: the representative test of `ground_isg` (place on the partner face / under the partner, "
                  "whole-body IK at the placement stance; spec 5.6) and the B3 reach bands with `--support floor --under <partner>`; failures go to the customs blacklist.",
                  self.artic_validation_line(),
                  self.supports_validation_line(),
                  "- **Alignment targets**: `(contacts target)` cannot be checked until B6 writes `require/contacts.csv`; the v1 geometric push / brace contacts are used meanwhile; "
                  "wall regions stay the 2026-09-11 survey until B5.",
                  "- **Short pools** (variation / held-out unavailable, see the per-subtask notes): small-scale articulation (5 categories, 8 annotated models -> annotate handles of "
                  "more small articulated models with the B4 heuristic + review: cooler, violin_case, second laptop / toolbox / blender models), attachment (one female per male "
                  "-> more (male, female) meta-link pairs from the KB), obstacles with 1-2 models (obstacle selection round-2 categories), block design (2 more block models), "
                  "IORS under_small / touching (4 admissible target categories each -> grasp tests for the round-1 touching targets flagged NEEDS GRASP TEST).",
                  "- **Held-out rule**: `ground_isg` must blacklist the held-out models in every role (reason `heldout`); the v1 bridge already removes them from the disembedding pools."]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icl", default=ICL)
    ap.add_argument("--out", default=os.path.join(ICL, "roles"))
    ap.add_argument("--per-family", type=int, default=30)
    ap.add_argument("--min-per-subtask", type=int, default=5)
    ap.add_argument("--support-triplets", type=int, default=3, help="support triplets per tabletop pick: T1 (derived) + the pinned T2, T3 (A3)")
    ap.add_argument("--patch-only", default="", metavar="KINDS",
                    help=f"comma list of {' / '.join(PATCH_KINDS)}: patch the files under --out in place instead of re-running the selection")
    args = ap.parse_args(argv)
    sel = Selector(args.icl, args.per_family, args.min_per_subtask, n_triplets=args.support_triplets)
    if args.patch_only:
        what = {w.strip() for w in args.patch_only.split(",") if w.strip()}
        sel.patch(args.out, what)
        sel.write(args.out, patch=True)
        for kind in PATCH_KINDS + ("candidates_extra",):
            for entry in sel.patch_log.get(kind, []):
                print(f"{kind}: {entry}")
        print(f"patched {sorted(what)}: {len(sel.rows)} slot rows -> {args.out}")
        return
    sel.run()
    sel.write(args.out)
    n_picks = sum(1 for r in sel.rows if r["slot"] == "representative" and r["role"] in ("target", "targets", "obstacles"))
    print(f"{n_picks} picks, {len(sel.rows)} slot rows, {len(sel.heldout)} held-out categories, {len(sel.extra)} extra candidate rows, "
          f"{len(sel.support_triplets())} support triplets -> {args.out}")


if __name__ == "__main__":
    main()
