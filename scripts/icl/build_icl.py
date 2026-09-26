#!/usr/bin/env python3
"""Build the icl (iSpatialGym candidate list) from the selection runlogs.

Inputs (all committed under ``runlogs/``):

* ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` - the reviewed selection sheet (old task ids /
  role names), incl. distractor pools, attachment partners, alignment frames, obstacle naturalness, table / wall rows,
* ``runlogs/grasp_library_20260905/select/test_instances.csv`` - pick keys / scores per (task, category),
* ``runlogs/grasp_library_20260905/results/<key>.json`` + ``cands/<key>.yaml`` + ``review/`` - the grasp library,
* ``runlogs/grasp_library_20260905/artic/{artic_results.csv,scene_assign.json,handles/,manual_handles.json}`` -
  articulated objects and the canonical handle annotations + reviewer verdicts,
* ``runlogs/scene_inventory_<date>/`` (``--scene-inventory``, B4) - ``summary.json`` + ``<scene>/{supports,partners,artic}/
  <name>.json``: regenerates the scene rows of ``roles/candidates.csv`` (A5-lite) - the supports of every tabletop subtask
  (spec 3.2: top z 0.55-1.20 m, top polygon >= 0.12 m2 and >= 0.30 m short side, >= 1 edge with a free-floor run >= 0.60 m
  at stand-off 0.45, edges holding at all three stand-offs preferred), the large partners of the IORS large-scale
  subtasks (spec 5.6 / 6.12: face raster; under: maximum under-clearance >= 0.30 m) with the supports within 3.0 m
  geodesic of an admitted partner, and the scene articulated targets (one row per joint: door / drawer / flap, B4 status
  handle / edge, pinch at 0.30-1.70 m with >= 0.60 m eroded free floor in front and a 15-80 mm width after the instance
  scale). Admitted rows get ``status = geometric_ok``, rejected ones stay in the sheet as ``geometric_reject:<reason>``
  without representative / test marks. The ``artic/<name>.json`` files are also the per-instance (scaled) handle
  annotations of ``require/handles.csv``,
* ``runlogs/contacts_<date>/<category>__<model>.json`` - push / brace contact annotations of the alignment targets
  (B6; the sheet is header-only until they exist),
* ``runlogs/table_selection_20260911/inventory.csv`` - geometry of every work surface / big partner in the 7 scenes,
* ``runlogs/world_alignment_selection_20260911/wall_candidates.json`` - the 78 validated wall regions,
* ``ispatialgym/assets/task_specs/task_custom_lists.json`` - the v1 test pins and the look-alike groups,
* ``runlogs/symmetry_<date>/symmetry.json`` (``--symmetry``) - the rotational-symmetry survey (geometry + texture,
  ``scripts/symmetry/survey_symmetry.py``) -> ``symmetry`` / ``symmetry_near`` columns of ``roles/candidates.csv``;
  ``review_verdicts.json`` next to it (``scripts/symmetry/aggregate_review.py``, the reviewer round) overrides the
  survey for the reviewed models and settles ``symmetric_180`` where the review answered the disagreement.

Outputs (``ispatialgym/assets/icl/``, schema in its ``README.md``):

* ``roles/`` - matching concrete categories and models to the iddl roles: ``candidates.csv`` (one row per candidate
  of a (subtask, role)), ``constraints_compatible.csv`` (which candidates go together: partner_of, distractor pools),
  ``constraints_conflicting.csv`` (which candidates must not appear together / as each other: look-alike groups);
* ``require/`` - the artefacts the iddl ``:require`` block points at, one compact summary row per computed pose:
  ``grasps.csv``, ``handles.csv``, ``contacts.csv`` (shared core columns, see ``REQUIRE_CORE``), plus ``lib/`` symlinks
  to the libraries the ``path`` column refers to.

Every sampleable / group candidate gets ``status = untested``: the old "viable" verdicts came from the sampler this
rebuild replaces, so they are provenance (``source_selection``), not status. Scene rows regenerated from the B4
inventory carry the geometric verdict (``geometric_ok`` / ``geometric_reject:<reason>``); rooms and wall regions stay
``untested``. The v1 test pins are kept as ``test`` / ``representative`` marks on the candidate rows (there is no
separate combinations sheet; ``ground_isg`` derives the representative combinations from the marks and
``constraints_compatible.csv``); a pin whose scene row is rejected is reported and left unmarked.

Run inside the ``ispatialgym`` conda env (BDDL knowledge base for synsets / bounding boxes / readiness; graceful
degradation without it)::

    python scripts/icl/build_icl.py [--out ispatialgym/assets/icl] [--symmetry runlogs/symmetry_<date>/symmetry.json]
                                    [--scene-inventory runlogs/scene_inventory_<date> | none]
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
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SELECT = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
LIB = os.path.join(REPO, "runlogs", "grasp_library_20260905")
ARTIC = os.path.join(LIB, "artic")
INVENTORY = os.path.join(REPO, "runlogs", "table_selection_20260911", "inventory.csv")
WALLS = os.path.join(REPO, "runlogs", "world_alignment_selection_20260911", "wall_candidates.json")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
SCENE_INVENTORY = os.path.join(REPO, "runlogs", "scene_inventory_20260915")
CONTACTS_GLOB = os.path.join(REPO, "runlogs", "contacts_*")
SYMMETRY_DEFAULT = os.path.join(REPO, "runlogs", "symmetry_20260915", "symmetry.json")
sys.path.insert(0, REPO)

try:  # knowledge base (synsets, bboxes, readiness)
    from ispatialgym.sampling import knowledge as K  # type: ignore
except Exception as _e:  # pragma: no cover
    K = None
    _K_ERROR = repr(_e)
try:
    from ispatialgym.sampling import grasp_library as GL  # type: ignore
except Exception as _e:  # pragma: no cover
    GL = None
    _GL_ERROR = repr(_e)

if K is None or GL is None:
    # Without the knowledge base the builder silently writes empty ``synset`` / ``bbox_*`` / ``ready`` cells (seen when
    # run from the ``base`` env, 2026-09-15). Refuse instead of producing a plausible-looking but wrong icl.
    if os.environ.get("ISG_ICL_ALLOW_NO_KB", "0") != "1":
        sys.exit("build_icl.py: the BEHAVIOR knowledge base / grasp library could not be imported "
                 f"(knowledge: {globals().get('_K_ERROR', 'ok')}; grasp_library: {globals().get('_GL_ERROR', 'ok')}). "
                 "Run in the `ispatialgym` conda env, or set ISG_ICL_ALLOW_NO_KB=1 for a deliberately partial build.")

SCENES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "hotel_suite_large",
          "restaurant_diner", "office_cubicles_right"]
IORS = "inter_object_relation_satisfaction"
CM = "configuration_matching"
TABLETOP_SUBTASKS = ["disembedding", f"{CM}-block_design", f"{CM}-articulation_open_small_scale", f"{CM}-articulation_close_small_scale",
                     f"{CM}-attachment", "alignment-axial", f"{IORS}-ontop_small_scale", f"{IORS}-under_small_scale", f"{IORS}-inside",
                     f"{IORS}-nextto", f"{IORS}-touching", f"{IORS}-ontop_large_scale", f"{IORS}-under_large_scale"]
WALL_SUBTASKS = ["alignment-world_vertical"]
ONTOP_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table", "countertop")
UNDER_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table")
SCENE_ARTIC_CATEGORIES = ("bottom_cabinet", "bottom_cabinet_no_top", "top_cabinet", "fridge", "oven", "dishwasher", "door")
RANK = {"spec": 0, "pick 1": 1, "pick 2": 2, "alternate": 3, "candidate": 4, "viable": 4, "partner": 4, "pool": 5, "reserve": 6}
TEST_SUPPORT = ("house_single_floor", "breakfast_table_rhjoby_0")
LARGE_SCALE_IORS = (f"{IORS}-ontop_large_scale", f"{IORS}-under_large_scale")

# -- geometric admissibility of the scene rows regenerated from the B4 scene inventory (A5-lite; spec 3.2 "support
# admissibility", 5.6 / 6.12 large partners, the B4 README). Every threshold is a constant so the report can quote it.
STATUS_OK = "geometric_ok"
STATUS_REJECT = "geometric_reject"          # `geometric_reject:<reason>`, the first failing check; every failing check in `note`
SUPPORT_TOP_Z = (0.55, 1.20)                # m -> `support_low` / `support_high`
SUPPORT_MIN_AREA_M2 = 0.12                  # top polygon area -> `support_small`
SUPPORT_MIN_SHORT_SIDE_M = 0.30             # minimum width of the top's convex hull -> `support_small`
EDGE_FREE_RUN_MIN_M = 0.60                  # longest free-floor run (chairs within 1 m removed) in front of an edge -> `no_free_edge`
STAND_OFFS = ("0.35", "0.45", "0.55")       # the B4 stand-offs; an edge is admitted at ADMIT_STAND_OFF, preferred when it holds at all three
ADMIT_STAND_OFF = "0.45"
PARTNER_GEODESIC_MAX_M = 3.0                # support -> admitted partner on the eroded map (spec 5.6 max_partner_geodesic) -> `no_near_partner`
UNDER_CLEARANCE_MIN_M = 0.30                # maximum under-clearance of an under(large-scale) partner -> `under_clearance`
HANDLE_Z = (0.30, 1.70)                     # world height of the pinch -> `handle_low` / `handle_high`
FRONT_FREE_MIN_M = 0.60                     # eroded free floor in front of the face -> `front_blocked`
PINCH_MM = (15.0, 80.0)                     # pinch width after the instance scale -> `pinch_narrow` / `pinch_wide`
ARTIC_JOINT_KINDS = ("door", "drawer", "flap")
ARTIC_GRASPABLE = ("handle", "edge")        # B4 joint status admitted (`ungraspable` rejected)
STRUCTURAL_CATEGORIES = ("walls", "ceilings", "floors", "fixed_window")  # top "occupants" that are neighbours, not objects on the top

ROLE_COLUMNS = ["subtask", "role", "kind", "category", "model", "display_name", "synset", "scene", "room", "representative", "test",
                "rank", "source_selection", "status", "grasp_key", "scale", "bbox_x", "bbox_y", "bbox_z", "natural_scenes",
                "frame_front", "frame_up", "frame_level", "frame_long", "symmetric_180", "symmetry", "symmetry_near", "joint", "handle_key", "handle_kind",
                "handle_annotated", "instance_scale", "region_id", "p0", "p1", "normal", "hang_z", "length_m", "max_object_m",
                "top_z", "world_ext_x", "world_ext_y", "pos_x", "pos_y", "yaw_deg", "dataset", "ready", "note", "source"]
# roles.csv `kind`: how ground_isg grounds the role (the iddl files carry no role declarations)
KIND_SAMPLEABLE = "sampleable"  # one spawned instance, model drawn per template
KIND_GROUP = "group"            # member candidate of a group role (size / draw mode in the customs list generation.groups)
KIND_SCENE = "scene"            # scene object / wall region / room bound per scene
PAIR_COLUMNS = ["subtask", "relation", "role_a", "category_a", "model_a", "role_b", "category_b", "model_b", "tier", "attach_type",
                "ready", "note", "source"]
CONFLICT_COLUMNS = ["relation", "group_id", "category", "model", "reason", "source"]
# require/ sheets: one row per computed pose (grasp / handle contact / push-brace contact), shared core columns first.
# `require` is the iddl :require form the row satisfies (grasps:<pipeline> | handle | contacts); `position`,
# `axis_approach` (a, the eef +z direction at contact) and `axis_close` (c, the eef +y / pinch axis) are in the object's
# base_link frame; `path` is the library file (repo-relative) the summary was taken from.
REQUIRE_CORE = ["key", "category", "model", "scene", "instance", "require", "kind", "name", "position", "axis_approach", "axis_close",
                "width_m", "verdict", "flags", "note", "path", "source"]
GRASP_COLUMNS = REQUIRE_CORE + ["quat_xyzw", "finger_q1", "finger_q2", "pad_clearance_m", "com_perp_m", "antipodal", "coverage", "tilt_deg",
                                "rest_quat_wxyz", "scale", "task", "pipeline"]
HANDLE_COLUMNS = REQUIRE_CORE + ["joint", "joint_type", "joint_kind", "status", "chosen", "exposed", "length_m", "depth_m", "pull_dir",
                                 "axis_dir", "hinge_point", "lower", "upper", "closed_q", "instance_scale", "position_world", "height_world_m",
                                 "front_free_m", "front_free_eroded_m", "review_graspable", "review_pinch"]
CONTACT_COLUMNS = REQUIRE_CORE + ["contact_role", "direction", "shape_class", "scale"]
#: grasp-library `constraint` -> iddl grasp pipeline
GRASP_PIPELINE = {"floating": "floating", "handle_half": "floating", "tabletop": "tabletop", "tabletop_insert": "insert",
                  "flower_stem_top": "insert", "attachment_free": "attachment"}
GRIPPER_OPEN_M = 0.10  # two fingers at 0.05 each

#: v1 test set (task_custom_lists.json pins): kept as `test` marks (and `representative` where the category had none) on
#: the candidate rows. Scene objects are `scene:name`, `scene:room/*` = every wall region of the room, rooms `scene:room`.
_SUP = f"{TEST_SUPPORT[0]}:{TEST_SUPPORT[1]}"
_ROOM = "house_single_floor:living_room_0"
_WALLS = "house_single_floor:living_room_0/*"
PINS: Dict[str, Dict[str, List[str]]] = {
    "camera_relocalization-standard": {"room": [_ROOM]},
    "camera_relocalization-obstructed": {"room": [_ROOM]},
    "object_scaling": {"target": ["bottle_of_beer/gxxbhh"], "room": [_ROOM]},
    "mental_rotation": {"target": ["toy_figure/zuntus"], "room": [_ROOM]},
    "disembedding": {"target": ["box_of_cereal/iucshm"], "source": ["tray/uekqey"], "destination": ["tray/hvlfig"], "support": [_SUP]},
    f"{CM}-block_design": {"targets": ["toy_dice/isgblk"], "support": [_SUP]},
    f"{CM}-articulation_open_small_scale": {"target": ["toolbox/redmtl"], "support": [_SUP]},
    f"{CM}-articulation_close_small_scale": {"target": ["toolbox/redmtl"], "support": [_SUP]},
    f"{CM}-attachment": {"target": ["cap/xxqyrt"], "partner": ["lemonade_bottle/yprkek"], "support": [_SUP]},
    "alignment-axial": {"target": ["keyboard/hgtbot", "keyboard/injrgc"], "support": [_SUP]},
    "alignment-world_vertical": {"target": ["painting/dacihj", "painting/ljabhq"], "support": [_WALLS]},
    f"{IORS}-ontop_small_scale": {"target": ["pack_of_kielbasa/dgodnn"], "partner": ["platter/csanbr"], "support": [_SUP]},
    f"{IORS}-under_large_scale": {"target": ["toy_figure/cobfpg"], "partner": ["house_single_floor:bed_rrcvaq_0"],
                                  "support": ["house_single_floor:desk_xhjsub_0"]},
    f"{IORS}-inside": {"target": ["lemon/ovbcff"], "partner": ["bowl/belcml"], "support": [_SUP]},
    f"{IORS}-nextto": {"target": ["spatula/hkvsep"], "partner": ["soup_ladle/cfzneb"], "support": [_SUP]},
    f"{IORS}-touching": {"target": ["can_of_baking_mix/blrqqz", "can_of_baking_mix/orwvfx"], "partner": ["board_game/eoywpy", "board_game/gcfkqa"],
                         "support": [_SUP]},
}
for _st in TABLETOP_SUBTASKS:  # the test support is the representative support of every tabletop subtask
    _sups = PINS.setdefault(_st, {}).setdefault("support", [])
    if _SUP not in _sups:
        _sups.append(_SUP)

#: (role, category, v1 model) -> (model, note): the disembedding tray models replaced after the v1 selection sheet.
#: Decision 2026-09-24: the v1 source tray ``tray/gsxbym`` has a collision floor of four convex-hull slabs with a 1.6 mm
#: step between them (2 mm at the 1.25 spawn scale); objects lying across the seams (the milk bottles, canonically on
#: their sides) never come to rest - PhysX contact chatter, 10-50 mm/s residual velocity, visible trembling in the
#: demos (pilot_20260915 disembedding 0916-141051 / 0916-181259) - and the chrome material mirrors the scene.
#: ``tray/uekqey`` has a single-slab floor (asleep at zero velocity in the same drop test), a matte white material and an
#: inner floor of 0.1673 m2 at scale 1 = the 0.107 m2 of gsxbym x 1.25^2, so it spawns unscaled with the same budget.
TRAY_OVERRIDES: Dict[Tuple[str, str, str], Tuple[str, str]] = {
    ("source", "tray", "gsxbym"): ("uekqey", "single-slab collision floor + matte material (decision 2026-09-24; gsxbym's 4-slab floor made lying objects tremble)"),
}

DISPLAY_OVERRIDES = {
    "toy_dice": "block", "pot_plant": "potted plant", "bottom_cabinet_no_top": "cabinet", "bottom_cabinet": "cabinet",
    "top_cabinet": "wall cabinet", "straight_chair": "chair", "swivel_chair": "office chair", "public_trash_can": "trash can",
    "taboret": "stool", "music_stool": "stool", "alphabet_abacus": "abacus", "breakfast_table": "table", "pedestal_table": "table",
    "conference_table": "conference table", "coffee_table": "coffee table", "walls": "wall", "floors": "floor",
    "hingeless_jar": "jar", "half_lime": "lime half", "half_bagel": "bagel half", "pack_of_kielbasa": "pack of sausages",
    "can_of_soda": "soda can", "box_of_cereal": "cereal box", "carton_of_milk": "milk carton", "bottle_of_beer": "beer bottle",
    "bottle_of_wine": "wine bottle", "bottle_of_pop": "soda bottle", "bottle_of_water": "water bottle", "bottle_of_medicine": "medicine bottle",
    "box_of_cookies": "cookie box", "box_of_tissues": "tissue box", "jar_of_spaghetti_sauce": "sauce jar", "toy_figure": "toy figure",
}


# ----------------------------------------------------------------------------------------------- helpers

def display_name(category: str) -> str:
    if category in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[category]
    return category.replace("_", " ")


def synset(category: str) -> str:
    if K is None:
        return ""
    try:
        return K.synset_of_category(category)
    except Exception:
        return ""


def bbox(model: str) -> Tuple[str, str, str]:
    if K is None or not model:
        return "", "", ""
    try:
        bb = K.canonical_bbox(model)
    except Exception:
        bb = None
    return tuple(f"{v:.4f}" for v in bb) if bb else ("", "", "")


def ready(model: str) -> str:
    if K is None or not model:
        return ""
    try:
        return "1" if K.is_ready(model) else "0"
    except Exception:
        return ""


def grasp_key_from_note(note: str) -> str:
    m = re.search(r"grasp key (\S+?)(?:;|,|\s|$)", note or "")
    return m.group(1) if m else ""


def empty_row(cols: List[str]) -> Dict[str, str]:
    return {c: "" for c in cols}


# ----------------------------------------------------------------------------------------------- inputs

def read_selection_blocks() -> Dict[Tuple[str, str], List[Tuple[str, dict]]]:
    """(old task, old role) -> [(category, row)] with the category header carried down to its model rows."""
    blocks: Dict[Tuple[str, str], List[Tuple[str, dict]]] = collections.OrderedDict()
    cur, cat = None, None
    with open(os.path.join(SELECT, "task_object_selection.csv")) as f:
        for r in csv.DictReader(f):
            if r["task"]:
                cur, cat = (r["task"], r["role"]), None
                blocks[cur] = []
                continue
            if r["category"] and not re.match(r"^\d+ \(\d+\)$", r["category"]):
                cat = r["category"]
            blocks[cur].append((cat, r))
    return blocks


def read_test_instances() -> Dict[Tuple[str, str], dict]:
    out = {}
    with open(os.path.join(SELECT, "test_instances.csv")) as f:
        for r in csv.DictReader(f):
            out[(r["task"], r.get("relation", ""), r["category"])] = r
    return out


def read_inventory() -> List[dict]:
    with open(INVENTORY) as f:
        return list(csv.DictReader(f))


def read_wall_regions() -> List[dict]:
    """The 78 validated regions (one full old-sampler instance each) of every room of the 7 scenes."""
    d = json.load(open(WALLS))
    out = []
    for scene, sd in d["scenes"].items():
        for room, rd in (sd.get("rooms") or {}).items():
            for r in rd.get("regions") or []:
                if not r.get("validated"):
                    continue
                out.append({"scene": scene, "room": room, "wall": r["wall_name"], "id": f"{room}/{r['wall_name']}/{r['id']}", "p0": r["p0"], "p1": r["p1"],
                            "normal": r["normal"], "hang_z": r.get("hang_z", d.get("hang_height_m", 1.45)), "length_m": r.get("length_m", ""),
                            "max_object_m": round(float(r["max_patch_m"]) - 0.10, 2) if r.get("max_patch_m") is not None else ""})
    return out


def read_scene_inventory(root: str) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """``{"supports" | "partners" | "artic": {scene: {name: record}}}`` from the per-object JSON files of a B4 run
    (``<root>/summary.json`` is the index; an object listed there without its JSON - e.g. an inventory error - is reported)."""
    summary = json.load(open(os.path.join(root, "summary.json")))["scenes"]
    out: Dict[str, Dict[str, Dict[str, dict]]] = {part: {} for part in ("supports", "partners", "artic")}
    for scene, sd in summary.items():
        for part in out:
            recs = {}
            for name in sd.get(part) or {}:
                path = os.path.join(root, scene, part, f"{name}.json")
                if os.path.exists(path):
                    recs[name] = json.load(open(path))
                else:
                    print(f"WARNING scene inventory: {scene}/{part}/{name}.json missing (listed in summary.json)")
            out[part][scene] = recs
    return out


def hull_min_width(hull) -> float:
    """Minimum width of a convex polygon (rotating calipers over its edges): the short side of a rectangular top whatever
    its yaw, the bounding-rectangle short side of an L-shaped counter."""
    pts = [(float(p[0]), float(p[1])) for p in hull or []]
    if len(pts) < 3:
        return 0.0
    best = float("inf")
    for i, (x0, y0) in enumerate(pts):
        x1, y1 = pts[(i + 1) % len(pts)]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-9:
            continue
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length
        proj = [px * nx + py * ny for px, py in pts]
        best = min(best, max(proj) - min(proj))
    return best if best < float("inf") else 0.0


def grasp_results_index() -> Dict[str, dict]:
    idx = {}
    for f in glob.glob(os.path.join(LIB, "results", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        idx[d.get("key", os.path.basename(f)[:-5])] = d
    return idx


# ----------------------------------------------------------------------------------------------- roles

class Builder:
    def __init__(self, scene_inventory: Optional[str] = None):
        self.blocks = read_selection_blocks()
        self.tests = read_test_instances()
        self.inventory = read_inventory()
        self.walls = read_wall_regions()
        self.results = grasp_results_index()
        self.spec = json.load(open(SPEC))
        self.roles: List[dict] = []
        self.pairs: List[dict] = []
        self.keys_used: set = set()
        # B4 scene inventory (A5-lite): when given, the scene rows (supports, large partners, scene articulated targets)
        # are regenerated from it with the geometric filters; otherwise the hand-copied seeds below are used
        self.scene_inventory = scene_inventory
        self.inv = read_scene_inventory(scene_inventory) if scene_inventory else None
        self.admitted_partners: Dict[str, set] = {st: set() for st in LARGE_SCALE_IORS}
        self.geo: Dict[str, list] = {"supports": [], "partners": [], "artic": [], "skipped_supports": [], "pins_rejected": []}

    # -- generic object row
    def add_object(self, subtask: str, role: str, category: str, row: dict, kind: str = KIND_SAMPLEABLE, **extra) -> dict:
        r = empty_row(ROLE_COLUMNS)
        model = row.get("model", "")
        sel = row.get("selection", "")
        key = grasp_key_from_note(row.get("note", "")) or extra.pop("grasp_key", "")
        res = self.results.get(key) if key else None
        r.update({
            "subtask": subtask, "role": role, "kind": kind, "category": category, "model": model, "display_name": display_name(category),
            "synset": synset(category), "representative": "1" if sel in ("pick 1", "spec") else "", "test": "1" if sel in ("pick 1", "pick 2", "spec") else "",
            "rank": str(RANK.get(sel, 4)), "source_selection": sel, "status": "untested", "grasp_key": key,
            "scale": json.dumps(res.get("scale")) if res and res.get("scale") else "", "ready": ready(model),
            "note": (row.get("note", "") or "")[:300], "source": "task_object_selection.csv",
        })
        r["bbox_x"], r["bbox_y"], r["bbox_z"] = bbox(model)
        if row.get("duplicates"):
            r["note"] = (r["note"] + " | " + row["duplicates"][:200]).strip(" |")
        r.update({k: v for k, v in extra.items() if k in r})
        if key:
            self.keys_used.add(key)
        self.roles.append(r)
        return r

    def add_fixed(self, subtask: str, role: str, scene: str, name: str, category: str, room: str = "", **extra) -> dict:
        r = empty_row(ROLE_COLUMNS)
        r.update({"subtask": subtask, "role": role, "kind": KIND_SCENE, "category": category, "model": name, "display_name": display_name(category),
                  "synset": synset(category), "scene": scene, "room": room, "rank": "4", "status": "untested", "source": extra.pop("source", "")})
        r.update({k: v for k, v in extra.items() if k in r})
        self.roles.append(r)
        return r

    def add_pair(self, subtask: str, relation: str, role_a: str, cat_a: str, model_a: str, role_b: str, cat_b: str, model_b: str = "", **extra):
        p = empty_row(PAIR_COLUMNS)
        p.update({"subtask": subtask, "relation": relation, "role_a": role_a, "category_a": cat_a, "model_a": model_a, "role_b": role_b,
                  "category_b": cat_b, "model_b": model_b, "source": extra.pop("source", "task_object_selection.csv")})
        p.update({k: v for k, v in extra.items() if k in p})
        self.pairs.append(p)

    # -- per family
    def build_camera_relocalization(self):
        for st in ("camera_relocalization-standard", "camera_relocalization-obstructed"):
            self.add_fixed(st, "room", "house_single_floor", "living_room_0", "floors", room="living_room_0", source="task_custom_lists.json",
                           note="camera relocalization constrained to this room for now (decision 2026-09-14)")
        for cat, row in self.blocks[("camera_relocalization-0", "obstacle")]:
            nat = re.sub(r"^natural in:\s*", "", row.get("pair", "")).replace("*", "")
            self.add_object("camera_relocalization-obstructed", "obstacles", cat, row, kind=KIND_GROUP, natural_scenes=nat)

    def build_held(self):
        for old, new in (("object_scaling-0", "object_scaling"), ("mental_rotation-0", "mental_rotation")):
            self.add_fixed(new, "room", "house_single_floor", "living_room_0", "floors", room="living_room_0", source="task_custom_lists.json")
            for cat, row in self.blocks[(old, "target")]:
                self.add_object(new, "target", cat, row)

    def build_disembedding(self):
        st = "disembedding"
        for cat, row in self.blocks[("disembedding-0", "target")]:
            self.add_object(st, "target", cat, row)
        seen = set()
        for cat, row in self.blocks[("disembedding-0", "same_category_pool")]:
            model = row["model"]
            self.add_pair(st, "distractor_same_category", "target", cat, "", "distractors", cat, model, tier="0",
                          note=(row.get("pair", "") + " | " + row.get("duplicates", ""))[:200])
            if (cat, model) not in seen:
                seen.add((cat, model))
                self.add_object(st, "distractors", cat, {"model": model, "selection": row["selection"], "note": row.get("note", ""),
                                                          "duplicates": row.get("duplicates", "")}, kind=KIND_GROUP)
        for tcat, row in self.blocks[("disembedding-0", "distractor_pool")]:
            if "/" not in row["model"]:
                continue
            dcat, dmodel = row["model"].split("/", 1)
            m = re.search(r"tier (\d)", row.get("pair", ""))
            self.add_pair(st, "distractor_other_category", "target", tcat, "", "distractors", dcat, dmodel, tier=m.group(1) if m else "",
                          note=row.get("pair", "")[:160])
            if (dcat, dmodel) not in seen:
                seen.add((dcat, dmodel))
                self.add_object(st, "distractors", dcat, {"model": dmodel, "selection": row["selection"], "note": row.get("note", ""),
                                                            "duplicates": row.get("duplicates", "")}, kind=KIND_GROUP)
        for old_role, new_role in (("tray_source", "source"), ("tray_dest", "destination")):
            for cat, row in self.blocks[("disembedding-0", old_role)]:
                override = TRAY_OVERRIDES.get((new_role, cat, row.get("model", "")))
                if override is not None:
                    row = dict(row, model=override[0], note=override[1] + " | was " + row.get("model", "") + ": " + row.get("note", ""))
                self.add_object(st, new_role, cat, row, note="fixed model for now (decision 2026-09-14); " + row.get("note", ""))

    def build_block_design(self):
        st = f"{CM}-block_design"
        for cat, row in self.blocks[("configuration_matching-0 (block_design)", "blocks")]:
            self.add_object(st, "targets", cat, row, kind=KIND_GROUP, dataset="ispatialgym-assets",
                            note="custom Blender Kohs block; group size / identical members are customs list parameters "
                                 "(generation.groups.targets: size 4, draw same_model); " + row.get("note", ""))

    def build_articulation(self):
        # small-scale: every table/floor articulated model with a graspable handle or edge (artic results), one row per joint
        rows = list(csv.DictReader(open(os.path.join(ARTIC, "artic_results.csv"))))
        for r in rows:
            if r["setting"] != "table/floor" or r["review_graspable"] != "True":
                continue
            for st in (f"{CM}-articulation_open_small_scale", f"{CM}-articulation_close_small_scale"):
                sel = "viable" if r["result"] == "opened" else "candidate"
                hk = f"{r['category']}__{r['model']}"
                self.add_object(st, "target", r["category"], {"model": r["model"], "selection": sel,
                                                               "note": f"{r['kind']}; handle {r['review_chosen']} {r['review_pinch']}; old result {r['result']}; {r['handle_desc']}"},
                                joint=r["joint"], handle_key=hk, handle_kind=r["review_pinch"],
                                handle_annotated="1" if os.path.exists(os.path.join(ARTIC, "handles", hk + ".json")) else "0",
                                source="artic/artic_results.csv")
        # large-scale: the articulated scene objects, per scene, with their instance scale - one row per (object, joint) from
        # the B4 scene inventory (A5-lite), or the hosts of the swap tests as seeds when no inventory is given
        if self.inv is not None:
            self.build_scene_artic_from_inventory()
            return
        sa = json.load(open(os.path.join(ARTIC, "scene_assign.json")))
        seen = set()
        tested = {(r["category"], r["model"]): r for r in rows if r["setting"].startswith("scene") and r.get("swap") in ("False", "")}
        for e in sa:
            host = e["host"]
            hcat = host.rsplit("_", 2)[0]
            hmodel = e["host_model"]
            if (e["scene"], host) in seen:
                continue
            seen.add((e["scene"], host))
            hk = f"{hcat}__{hmodel}"
            annotated = os.path.exists(os.path.join(ARTIC, "handles", hk + ".json"))
            joints = ""
            if annotated:
                try:
                    hd = json.load(open(os.path.join(ARTIC, "handles", hk + ".json")))
                    joints = ";".join(f"{j['joint']}:{j.get('status','')}" for j in hd.get("joints", []))
                except Exception:
                    pass
            old = tested.get((hcat, hmodel))
            # instance scale of the scene object relative to its canonical model: host AABB (world axes) over the canonical
            # bbox; the two horizontal axes are matched by sorted extent (scene yaws are multiples of 90 deg), z directly.
            scale = e.get("scale_factors") if not e.get("swap") else None
            if scale is None and e.get("host_aabb"):
                bb = bbox(hmodel)
                if all(bb):
                    ha, cb = e["host_aabb"], [float(v) for v in bb]
                    hx, hy = sorted(ha[:2]); cx, cy = sorted(cb[:2])
                    scale = [round(hx / cx, 3), round(hy / cy, 3), round(ha[2] / cb[2], 3)]
            for st in (f"{CM}-articulation_open_large_scale", f"{CM}-articulation_close_large_scale"):
                self.add_fixed(st, "target", e["scene"], host, hcat, handle_key=hk, handle_annotated="1" if annotated else "0",
                               joint=joints, instance_scale=json.dumps(scale) if scale else "",
                               note=("old test: " + old["result"] if old else "not tested as the scene instance") +
                                    ("; needs handle annotation (scaled instance)" if not annotated else "; annotation exists for the canonical model, rescale to the instance"),
                               source="artic/scene_assign.json")

    #: female parts that cannot stand on a table (wall / vehicle / furniture parts): kept as candidates, flagged for review
    NON_TABLETOP_PARTNERS = {"recreational_vehicle", "wall_nail", "bush", "bookcase_side", "cabinet_base", "skateboard_deck", "fishing_rod", "monitor"}

    def build_attachment(self):
        st = f"{CM}-attachment"
        for cat, row in self.blocks[("configuration_matching-3 (attachment)", "male")]:
            r = self.add_object(st, "target", cat, row)
            pair = row.get("pair", "")
            m = re.search(r"\[attachment type ([^\]]+)\]", pair)
            atype = m.group(1) if m else ""
            for part in re.findall(r"([a-z_]+)/([a-z]+)(\s*\(not ready\))?", pair.split("[")[0]):
                fcat, fmodel, nr = part
                flag = "PARTNER NOT TABLETOP-PLACEABLE (review); " if fcat in self.NON_TABLETOP_PARTNERS else ""
                self.add_pair(st, "partner_of", "target", cat, row["model"], "partner", fcat, fmodel, attach_type=atype,
                              ready="0" if nr else "1", note=(flag + pair)[:200])
            if "(no attachment meta-link)" in pair:
                r["note"] = "NO ATTACHMENT META-LINK: cannot attach; " + r["note"]
        for cat, row in self.blocks[("configuration_matching-3 (attachment)", "female")]:
            flag = "NOT TABLETOP-PLACEABLE (review: the partner must stand on the support); " if cat in self.NON_TABLETOP_PARTNERS else ""
            self.add_object(st, "partner", cat, row, note=flag + row.get("pair", "") + "; " + row.get("note", ""))

    def _frame_axes(self, pair: str) -> Dict[str, str]:
        out = {}
        for key, col in (("front", "frame_front"), ("up", "frame_up"), ("level", "frame_level")):
            m = re.search(rf"{key} ([+-]?[xyz])", pair)
            if m:
                out[col] = m.group(1) if m.group(1)[0] in "+-" else "+" + m.group(1)
        m = re.search(r"long axis ([xyz])", pair)
        if m:
            out["frame_long"] = m.group(1)
        out["symmetric_180"] = "1" if "symmetric 180" in pair else ""
        return out

    def build_alignment(self):
        for old, new in (("world_alignment-0 (table_surface)", "alignment-axial"), ("world_alignment-1 (world_vertical)", "alignment-world_vertical")):
            for cat, row in self.blocks[(old, "target")]:
                self.add_object(new, "target", cat, row, **self._frame_axes(row.get("pair", "")))
        for st in WALL_SUBTASKS:
            for w in self.walls:
                self.add_fixed(st, "support", w["scene"], w["wall"], "walls", room=w.get("room", ""), region_id=w["id"], p0=json.dumps(w["p0"]),
                               p1=json.dumps(w["p1"]), normal=json.dumps(w["normal"]), hang_z=str(w.get("hang_z", "")),
                               length_m=str(w.get("length_m", "")), max_object_m=str(w.get("max_object_m", "")),
                               source="world_alignment_selection_20260911/wall_candidates.json",
                               note="validated with the old arc / right-arm method; candidate for the Prompt-0 re-survey")

    def build_iors(self):
        # ontop: object partners -> small scale; "(scene)" partners -> large scale
        for cat, row in self.blocks[("inter_object_relation_satisfaction-0 (ontop)", "target")]:
            pair = row.get("pair", "")
            pcat = re.sub(r"^->\s*", "", pair).replace("(scene)", "").strip()
            if "(scene)" in pair:
                st = f"{IORS}-ontop_large_scale"
                self.add_object(st, "target", cat, row)
                self.add_pair(st, "partner_of", "target", cat, "", "partner", pcat, "", note="scene partner category")
            else:
                st = f"{IORS}-ontop_small_scale"
                self.add_object(st, "target", cat, row)
                self.add_pair(st, "partner_of", "target", cat, "", "partner", pcat, "")
        for cat, row in self.blocks[("inter_object_relation_satisfaction-0 (ontop)", "partner")]:
            if row.get("model"):
                self.add_object(f"{IORS}-ontop_small_scale", "partner", cat, row)
        # under: parse "-> <partner(s)> under <target> (target grasped)" (small) vs "-> under bed / desk" (large)
        for cat, row in self.blocks[("inter_object_relation_satisfaction-1 (under)", "target")]:
            pair = re.sub(r"^->\s*", "", row.get("pair", ""))
            if pair.startswith("under"):
                st = f"{IORS}-under_large_scale"
                self.add_object(st, "target", cat, row)
                for pcat in re.findall(r"[a-z_]+", pair.split("(")[0].replace("under", "")):
                    pcat = {"couch": "sofa"}.get(pcat, pcat)
                    if pcat in UNDER_LARGE_PARTNERS:
                        self.add_pair(st, "partner_of", "target", cat, "", "partner", pcat, "", note="scene partner category")
            else:
                st = f"{IORS}-under_small_scale"
                self.add_object(st, "target", cat, row)
                for pcat in re.findall(r"[a-z_]+", pair.split(" under ")[0]):
                    self.add_pair(st, "partner_of", "target", cat, "", "partner", pcat, "", note="(under partner target)")
        for cat, row in self.blocks[("inter_object_relation_satisfaction-1 (under)", "partner")]:
            if row.get("model") and cat not in ("bed", "couch", "sofa", "desk"):
                self.add_object(f"{IORS}-under_small_scale", "partner", cat, row)
        for old, new in (("inter_object_relation_satisfaction-2 (inside)", f"{IORS}-inside"), ("inter_object_relation_satisfaction-3 (nextto)", f"{IORS}-nextto"),
                         ("inter_object_relation_satisfaction-4 (touching)", f"{IORS}-touching")):
            for cat, row in self.blocks[(old, "target")]:
                r = self.add_object(new, "target", cat, row)
                if not r["grasp_key"]:
                    r["note"] = "NEEDS GRASP TEST (new oracle grasps then pushes); " + r["note"]
                pcat = re.sub(r"^->\s*", "", row.get("pair", "")).split("(")[0].strip()
                if pcat:
                    self.add_pair(new, "partner_of", "target", cat, "", "partner", pcat, "")
            for cat, row in self.blocks[(old, "partner")]:
                if row.get("model"):
                    self.add_object(new, "partner", cat, row)
        # large partners (fixed scene objects): the B4 scene inventory (A5-lite) or the table-selection inventory as seeds
        if self.inv is not None:
            self.build_partners_from_inventory()
            return
        for inv in self.inventory:
            if inv["category"] in ONTOP_LARGE_PARTNERS:
                self._fixed_from_inventory(f"{IORS}-ontop_large_scale", "partner", inv)
            if inv["category"] in UNDER_LARGE_PARTNERS:
                self._fixed_from_inventory(f"{IORS}-under_large_scale", "partner", inv)

    def _fixed_from_inventory(self, subtask: str, role: str, inv: dict, **extra):
        self.add_fixed(subtask, role, inv["scene"], inv["name"], inv["category"], room=inv["rooms"], top_z=inv["top_z"],
                       world_ext_x=inv["world_ext_x"], world_ext_y=inv["world_ext_y"], pos_x=inv["pos_x"], pos_y=inv["pos_y"], yaw_deg=inv["yaw_deg"],
                       bbox_x=inv["bbox_x"], bbox_y=inv["bbox_y"], bbox_z=inv["bbox_z"], source="table_selection_20260911/inventory.csv",
                       note=f"inventory verdict {inv['verdict']}" + (("; " + extra["note"]) if extra.get("note") else ""))

    def build_supports(self):
        """Support rows of every tabletop subtask: from the B4 scene inventory with the geometric filters (A5-lite) when one is
        given, else the union of every support that appears as spec / viable for any task (31 scene objects, seeds)."""
        if self.inv is not None:
            return self.build_supports_from_inventory()
        union: Dict[Tuple[str, str], dict] = {}
        for (task, role), items in self.blocks.items():
            if role != "table":
                continue
            for cat, row in items:
                if row.get("model") and row.get("selection") in ("viable", "spec"):
                    scene = row.get("pair", "").split(":")[0].strip()
                    union.setdefault((scene, row["model"]), {"category": cat, "rooms": row.get("pair", "").split(":", 1)[-1].strip(), "tasks": set()})["tasks"].add(task)
        inv_by_name = {(i["scene"], i["name"]): i for i in self.inventory}
        for (scene, name), info in sorted(union.items()):
            inv = inv_by_name.get((scene, name))
            for st in TABLETOP_SUBTASKS:
                if inv is not None:
                    self._fixed_from_inventory(st, "support", inv, note=f"old sampler viable/spec for {len(info['tasks'])} task(s)")
                else:
                    self.add_fixed(st, "support", scene, name, info["category"], room=info["rooms"], source="task_object_selection.csv",
                                   note=f"old sampler viable/spec for {len(info['tasks'])} task(s); not in inventory.csv")
        self._add_floor_supports()
        return len(union)

    def _add_floor_supports(self):
        """The floor of every scene as an admissible init support for the large-scale IORS subtasks (spec 5.6 (c))."""
        for st in LARGE_SCALE_IORS:
            for scene in SCENES:
                self.add_fixed(st, "support", scene, "floor", "floors", note="floor init: the target starts on the floor beside the partner (explicit support choice, never a fallback)",
                               source="decision 2026-09-14")

    # -- scene rows from the B4 scene inventory (A5-lite) ---------------------------------------------------------------
    @staticmethod
    def _status(reasons: List[str]) -> str:
        return STATUS_OK if not reasons else f"{STATUS_REJECT}:{reasons[0]}"

    def _inv_source(self, scene: str, part: str, name: str) -> str:
        return f"{os.path.basename(os.path.normpath(self.scene_inventory))}/{scene}/{part}/{name}.json"

    def _surface_columns(self, rec: dict) -> dict:
        """`top_z`, world extents / centre / yaw and the canonical bbox of a B4 support / partner record."""
        top, aabb = rec.get("top") or {}, rec.get("aabb_world") or {}
        lo, hi, ext = aabb.get("min"), aabb.get("max"), aabb.get("extent")
        cols = {"top_z": f"{top['z']:.3f}" if "z" in top else "", "yaw_deg": f"{math.degrees(float(top['yaw'])):.1f}" if "yaw" in top else "",
                "world_ext_x": f"{ext[0]:.3f}" if ext else "", "world_ext_y": f"{ext[1]:.3f}" if ext else "",
                "pos_x": f"{(lo[0] + hi[0]) / 2:.3f}" if lo and hi else "", "pos_y": f"{(lo[1] + hi[1]) / 2:.3f}" if lo and hi else ""}
        cols["bbox_x"], cols["bbox_y"], cols["bbox_z"] = bbox(rec.get("model", ""))
        return cols

    @staticmethod
    def _edge_runs(edge: dict) -> List[float]:
        return [float((edge.get("runs") or {}).get(s, {}).get("longest_free_run_m", 0.0)) for s in STAND_OFFS]

    def _support_geometry(self, rec: dict) -> dict:
        """Spec 3.2 support admissibility on one B4 support record: top height, polygon size (area + hull minimum width),
        >= 1 edge with a free-floor run >= EDGE_FREE_RUN_MIN_M at stand-off 0.45 (chairs within 1 m removed); edges whose run
        holds at all three stand-offs are listed first (`edges=`), the ones holding at 0.45 only under `edges_0.45_only=`."""
        top = rec.get("top") or {}
        z, area = float(top.get("z", 0.0)), float(top.get("area_m2", 0.0))
        short = hull_min_width(top.get("hull_xy"))
        if not short and top.get("local_bounds_uv"):
            b = top["local_bounds_uv"]
            short = min(float(b[1]) - float(b[0]), float(b[3]) - float(b[2]))
        reasons: List[str] = []
        cat = rec.get("category", "")
        if "chair" in cat or cat in ("stool", "taboret", "music_stool"):
            reasons.append("chair")
        if z < SUPPORT_TOP_Z[0]:
            reasons.append("support_low")
        elif z > SUPPORT_TOP_Z[1]:
            reasons.append("support_high")
        if area < SUPPORT_MIN_AREA_M2 or short < SUPPORT_MIN_SHORT_SIDE_M:
            reasons.append("support_small")
        full, partial = [], []
        for e in rec.get("edges") or []:
            runs = self._edge_runs(e)
            if runs[STAND_OFFS.index(ADMIT_STAND_OFF)] < EDGE_FREE_RUN_MIN_M:
                continue
            (full if all(r >= EDGE_FREE_RUN_MIN_M for r in runs) else partial).append((e["name"], runs))
        if not full and not partial:
            reasons.append("no_free_edge")
        best = max((r[STAND_OFFS.index(ADMIT_STAND_OFF)] for _, r in full + partial), default=0.0)
        fmt = lambda edges: ",".join(f"{n}({r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f})" for n, r in edges)  # noqa: E731
        note = f"B4: top z {z:.2f}, area {area:.2f} m2, short side {short:.2f} m"
        if full:
            note += f"; edges={fmt(full)}"
        if partial:
            note += f"; edges_0.45_only={fmt(partial)}"
        if full or partial:
            note += f"; best_run={best:.2f}"
        if partial and not full:
            note += "; no edge holds at all three stand-offs"
        chairs = rec.get("chairs_within_1m") or []
        if chairs:
            note += f"; chairs<=1m={len(chairs)}"
        occupants = [o for o in (top.get("occupants") or []) if o.rsplit("_", 2)[0] not in STRUCTURAL_CATEGORIES]
        if occupants:
            note += f"; occupants={','.join(occupants[:6])}" + ("..." if len(occupants) > 6 else "")
        if rec.get("in_union"):
            note += "; in the old support union"
        return {"reasons": reasons, "note": note, "z": z, "area": area, "short": short, "edges": full, "edges_partial": partial, "best_run": best}

    def build_partners_from_inventory(self):
        """`partner` rows of the two large-scale IORS subtasks from the B4 partner inventory (spec 5.6 / 6.12): ontop = every bed
        / sofa / desk / breakfast table / countertop with a face raster; under = every bed / sofa / desk / breakfast table whose
        maximum under-clearance is >= UNDER_CLEARANCE_MIN_M (`under_clearance`). Rejected partners stay in the sheet."""
        for st, cats in ((LARGE_SCALE_IORS[0], ONTOP_LARGE_PARTNERS), (LARGE_SCALE_IORS[1], UNDER_LARGE_PARTNERS)):
            for scene in SCENES:
                for name, rec in sorted(self.inv["partners"].get(scene, {}).items()):
                    if rec.get("category") not in cats:
                        continue
                    top, uc = rec.get("top") or {}, rec.get("under_clearance") or {}
                    reasons: List[str] = []
                    if not top.get("n_free_cells"):
                        reasons.append("no_face_raster")
                    best = max((self._edge_runs(e)[STAND_OFFS.index(ADMIT_STAND_OFF)] for e in rec.get("edges") or []), default=0.0)
                    near = [(s["support"], s["geodesic_m"]) for s in rec.get("nearby_supports") or []
                            if s.get("geodesic_m") is not None and s["geodesic_m"] <= PARTNER_GEODESIC_MAX_M]
                    note = (f"B4: face z {float(top.get('z', 0.0)):.2f}, {top.get('n_free_cells', 0)} face cells ({float(top.get('area_m2', 0.0)):.2f} m2)"
                            f"; best_run_0.45={best:.2f}")
                    ucmax = uc.get("max_clearance_m")
                    if st == LARGE_SCALE_IORS[1]:
                        if ucmax is None or float(ucmax) < UNDER_CLEARANCE_MIN_M:
                            reasons.append("under_clearance")
                        note += f"; under_clearance_max={float(ucmax):.2f}" if ucmax is not None else "; under_clearance_max=unknown"
                        frac = (uc.get("fraction_ge") or {}).get(f"{UNDER_CLEARANCE_MIN_M:.2f}")
                        if frac is not None:
                            note += f" (fraction>={UNDER_CLEARANCE_MIN_M:.2f}: {frac})"
                    note += "; supports<=3m=" + (",".join(f"{n}({d:.2f})" for n, d in near) if near else "-")
                    if reasons:
                        note += "; rejects=" + ",".join(reasons)
                    if not reasons:
                        self.admitted_partners[st].add((scene, name))
                    self.geo["partners"].append({"subtask": st, "scene": scene, "name": name, "category": rec["category"], "status": self._status(reasons),
                                                 "face_z": float(top.get("z", 0.0)), "under_max": ucmax, "near": near})
                    self.add_fixed(st, "partner", scene, name, rec["category"], room=",".join(rec.get("rooms") or []), status=self._status(reasons),
                                   note=note, source=self._inv_source(scene, "partners", name), **self._surface_columns(rec))

    def build_supports_from_inventory(self) -> int:
        """`support` rows of every tabletop subtask from the B4 support inventory with the spec 3.2 filters (`_support_geometry`);
        supports without a top raster (0 free cells) are skipped and reported. For the two large-scale IORS subtasks a support
        must in addition lie within PARTNER_GEODESIC_MAX_M (eroded-map geodesic, from the partner records) of an admitted partner
        of that subtask (`no_near_partner`; the partners are named in `note` as `near_partners=`). Then the floors."""
        near: Dict[str, Dict[Tuple[str, str], list]] = {st: collections.defaultdict(list) for st in LARGE_SCALE_IORS}
        unknown: Dict[str, Dict[Tuple[str, str], list]] = {st: collections.defaultdict(list) for st in LARGE_SCALE_IORS}
        for st in LARGE_SCALE_IORS:
            for scene, pname in sorted(self.admitted_partners[st]):
                for s in self.inv["partners"][scene][pname].get("nearby_supports") or []:
                    if s.get("geodesic_m") is not None and s["geodesic_m"] <= PARTNER_GEODESIC_MAX_M:
                        near[st][(scene, s["support"])].append((pname, float(s["geodesic_m"])))
                    elif s.get("geodesic_m") is None and float(s.get("aabb_distance_m", 99)) <= PARTNER_GEODESIC_MAX_M:
                        unknown[st][(scene, s["support"])].append((pname, float(s["aabb_distance_m"])))
        geo: Dict[Tuple[str, str], dict] = {}
        for st in TABLETOP_SUBTASKS:
            for scene in SCENES:
                for name, rec in sorted(self.inv["supports"].get(scene, {}).items()):
                    if not (rec.get("top") or {}).get("n_free_cells"):
                        if st == TABLETOP_SUBTASKS[0]:
                            self.geo["skipped_supports"].append((scene, name, rec.get("category", "")))
                        continue
                    if (scene, name) not in geo:
                        geo[(scene, name)] = self._support_geometry(rec)
                        self.geo["supports"].append({"scene": scene, "name": name, "category": rec["category"], **geo[(scene, name)]})
                    g = geo[(scene, name)]
                    reasons, note = list(g["reasons"]), g["note"]
                    if st in LARGE_SCALE_IORS:
                        ps = near[st].get((scene, name), [])
                        if ps:
                            note += "; near_partners=" + ",".join(f"{p}({d:.2f})" for p, d in sorted(ps, key=lambda x: x[1]))
                        else:
                            reasons.append("no_near_partner")
                            unk = unknown[st].get((scene, name), [])
                            if unk:
                                note += "; partners_aabb<=3m_geodesic_unknown=" + ",".join(f"{p}({d:.2f})" for p, d in sorted(unk, key=lambda x: x[1]))
                    if reasons:
                        note += "; rejects=" + ",".join(reasons)
                    self.add_fixed(st, "support", scene, name, rec["category"], room=",".join(rec.get("rooms") or []), status=self._status(reasons),
                                   note=note, source=self._inv_source(scene, "supports", name), **self._surface_columns(rec))
        self._add_floor_supports()
        return sum(1 for g in self.geo["supports"] if not g["reasons"])

    def _artic_joint_geometry(self, j: dict) -> dict:
        """Admissibility of one B4 joint: kind door / drawer / flap, status handle / edge, and a pinch (a protruding handle, else
        the exposed edge fallback) at HANDLE_Z world height with FRONT_FREE_MIN_M eroded free floor in front of the face and a
        width inside PINCH_MM after the instance scale. The candidate with the fewest failing checks is the row's pinch."""
        kind, status = j.get("kind", ""), j.get("status", "")
        reasons: List[str] = []
        if status == "error" or not kind:
            return {"reasons": ["error"], "kind": "", "height": None, "width_mm": None, "via_fallback": False, "front_free": j.get("front_free_eroded_m")}
        if kind not in ARTIC_JOINT_KINDS:
            reasons.append(f"joint_kind_{kind}")
        if status not in ARTIC_GRASPABLE:
            reasons.append("ungraspable")
        ff = j.get("front_free_eroded_m")
        cands = [(h, False) for h in j.get("handles") or []]
        if j.get("fallback") and j["fallback"].get("exposed"):
            cands.append((j["fallback"], True))
        best = None
        for h, is_fallback in cands:
            rs: List[str] = []
            hz, w = h.get("height_world"), h.get("width")
            if hz is None:
                rs.append("handle_height_unknown")
            elif float(hz) < HANDLE_Z[0]:
                rs.append("handle_low")
            elif float(hz) > HANDLE_Z[1]:
                rs.append("handle_high")
            if ff is None:
                rs.append("front_free_unknown")
            elif float(ff) < FRONT_FREE_MIN_M:
                rs.append("front_blocked")
            if w is None:
                rs.append("pinch_unknown")
            elif float(w) * 1000 < PINCH_MM[0]:
                rs.append("pinch_narrow")
            elif float(w) * 1000 > PINCH_MM[1]:
                rs.append("pinch_wide")
            key = (len(rs), is_fallback)
            if best is None or key < best[0]:
                best = (key, h, rs, is_fallback)
            if not rs and not is_fallback:
                break
        if best is None:
            if "ungraspable" not in reasons:
                reasons.append("ungraspable")
            return {"reasons": reasons, "kind": "", "height": None, "width_mm": None, "via_fallback": False, "front_free": ff}
        _, h, rs, is_fallback = best
        return {"reasons": reasons + rs, "kind": h.get("kind", ""), "height": h.get("height_world"),
                "width_mm": float(h["width"]) * 1000 if h.get("width") is not None else None, "via_fallback": is_fallback and bool(j.get("handles")),
                "front_free": ff}

    def build_scene_artic_from_inventory(self):
        """`target` rows of the two large-scale articulation subtasks: one row per (scene object, joint) of the B4 artic inventory
        with the `_artic_joint_geometry` verdict; `joint` is the joint, `handle_kind` the pinch used (bar / knob / edge_*),
        `handle_annotated` whether the canonical model has a reviewed annotation, `instance_scale` the measured scene scale."""
        for st in (f"{CM}-articulation_open_large_scale", f"{CM}-articulation_close_large_scale"):
            for scene in SCENES:
                for name, d in sorted(self.inv["artic"].get(scene, {}).items()):
                    hk = f"{d['category']}__{d['model']}"
                    annotated = bool(d.get("canonical_annotation")) or os.path.exists(os.path.join(ARTIC, "handles", hk + ".json"))
                    for j in d.get("joints") or []:
                        g = self._artic_joint_geometry(j)
                        pinch = (f"{g['kind']} h={g['height']:.2f} w={g['width_mm']:.0f}mm" if g["kind"] and g["height"] is not None and g["width_mm"] is not None
                                 else "no pinch candidate")
                        ff = j.get("front_free_m")
                        note = (f"B4 {j.get('kind', '?')} {j.get('status', '?')}: {pinch}; front_free="
                                + (f"{float(ff):.2f}({float(g['front_free']):.2f})" if ff is not None and g["front_free"] is not None else "unknown"))
                        if g["via_fallback"]:
                            note += "; via the exposed edge fallback (the protruding handles fail the pinch filters)"
                        note += "; canonical annotation " + ("exists (rescale to the instance)" if annotated else "missing (C3 review)")
                        prior = j.get("prior_review")
                        if isinstance(prior, dict):
                            note += f"; prior canonical review {prior.get('chosen', '-')}/{prior.get('result', '-')}"
                        if g["reasons"]:
                            note += "; rejects=" + ",".join(g["reasons"])
                        if st.endswith("open_large_scale"):
                            self.geo["artic"].append({"scene": scene, "name": name, "category": d["category"], "joint": j.get("joint", ""), "joint_kind": j.get("kind", ""),
                                                      "b4_status": j.get("status", ""), "status": self._status(g["reasons"]), "pinch_kind": g["kind"],
                                                      "height": g["height"], "width_mm": g["width_mm"]})
                        self.add_fixed(st, "target", scene, name, d["category"], room=",".join(d.get("rooms") or []), joint=j.get("joint", ""), handle_key=hk,
                                       handle_kind=g["kind"], handle_annotated="1" if annotated else "0",
                                       instance_scale=json.dumps(d.get("scale")) if d.get("scale") else "", status=self._status(g["reasons"]), note=note,
                                       source=self._inv_source(scene, "artic", name), **dict(zip(("bbox_x", "bbox_y", "bbox_z"), bbox(d.get("model", "")))))

    def report_scene_rows(self):
        """Stdout report of the regenerated scene rows: counts per (subtask, role, status), the skipped supports, the admitted
        supports per scene with their edges, the admitted partners and the admitted articulated targets (for picking
        representatives), the geodesic gaps of the B4 partner records."""
        print(f"scene rows from {os.path.relpath(self.scene_inventory, REPO)} (filters: top z {SUPPORT_TOP_Z}, area >= {SUPPORT_MIN_AREA_M2} m2, "
              f"short side >= {SUPPORT_MIN_SHORT_SIDE_M} m, edge run >= {EDGE_FREE_RUN_MIN_M} m at {ADMIT_STAND_OFF}, partner geodesic <= {PARTNER_GEODESIC_MAX_M} m, "
              f"under clearance >= {UNDER_CLEARANCE_MIN_M} m, handle z {HANDLE_Z}, front free >= {FRONT_FREE_MIN_M} m, pinch {PINCH_MM} mm)")
        by = collections.Counter((r["subtask"], r["role"], r["status"]) for r in self.roles if r["kind"] == KIND_SCENE)
        for (st, role, status), n in sorted(by.items()):
            print(f"  {st:58s} {role:8s} {status:36s} {n}")
        if self.geo["skipped_supports"]:
            print(f"  skipped (no top raster / 0 free cells): {len(self.geo['skipped_supports'])} supports: "
                  + ", ".join(f"{s}:{n}" for s, n, _ in self.geo["skipped_supports"]))
        print("  admitted supports per scene (name, category, top z, edges at 0.35/0.45/0.55):")
        for scene in SCENES:
            rows = [g for g in self.geo["supports"] if g["scene"] == scene and not g["reasons"]]
            print(f"    {scene}: {len(rows)}")
            for g in rows:
                edges = ",".join(f"{n}({r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f})" for n, r in g["edges"])
                partial = ",".join(f"{n}({r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f})" for n, r in g["edges_partial"])
                print(f"      {g['name']:34s} {g['category']:26s} z {g['z']:.2f}  {g['area']:.2f} m2 x {g['short']:.2f} m  edges={edges}"
                      + (f"  0.45-only={partial}" if partial else ""))
        rej = collections.Counter(g["reasons"][0] for g in self.geo["supports"] if g["reasons"])
        print(f"  rejected supports by first reason: {dict(rej)}")
        for st in LARGE_SCALE_IORS:
            rows = [g for g in self.geo["partners"] if g["subtask"] == st]
            ok = [g for g in rows if g["status"] == STATUS_OK]
            print(f"  {st}: {len(ok)} / {len(rows)} partners admitted")
            for g in rows:
                near = ",".join(f"{n}({d:.2f})" for n, d in g["near"]) or "-"
                under = f" under_max={float(g['under_max']):.2f}" if g["under_max"] is not None and st.endswith("under_large_scale") else ""
                print(f"      {g['status']:36s} {g['scene']}:{g['name']} ({g['category']}, face z {g['face_z']:.2f}{under}) supports<=3m={near}")
        ok = [g for g in self.geo["artic"] if g["status"] == STATUS_OK]
        print(f"  articulated scene targets: {len(ok)} / {len(self.geo['artic'])} (object, joint) admitted; "
              f"categories: {dict(collections.Counter(g['category'] for g in ok))}")
        for g in ok:
            print(f"      {g['scene']}:{g['name']} {g['joint']} {g['joint_kind']} {g['pinch_kind']} h={g['height']:.2f} w={g['width_mm']:.0f}mm")
        rej = collections.Counter(g["status"] for g in self.geo["artic"] if g["status"] != STATUS_OK)
        print(f"  rejected (object, joint) by status: {dict(rej)}")
        for st, role, pin, status in self.geo["pins_rejected"]:
            print(f"  WARNING pin rejected by the geometric filter (no representative / test mark): {st} {role} {pin} -> {status}")

    # -- pins (the former combos sheet): test / representative marks on the candidate rows
    def mark_pins(self):
        """Apply the v1 test pins as ``test = 1`` marks (and ``representative = 1`` where the (subtask, role, category) had
        none) on the candidate rows; scene pins (test support, wall regions of the test room, test partner, room bounds)
        become representative scene rows. Missing rows are reported, never invented."""
        by_key: Dict[Tuple[str, str, str, str], List[dict]] = collections.defaultdict(list)
        for r in self.roles:
            by_key[(r["subtask"], r["role"], r["category"], r["model"])].append(r)
        has_rep: set = {(r["subtask"], r["role"], r["category"]) for r in self.roles if r["representative"] == "1"}
        for st, roles in PINS.items():
            for role, pins in roles.items():
                for pin in pins:
                    if ":" in pin:  # scene object / wall regions of a room / room
                        scene, name = pin.split(":", 1)
                        if name.endswith("/*"):
                            hits = [r for r in self.roles if r["subtask"] == st and r["role"] == role and r["scene"] == scene and r["room"] == name[:-2]]
                        else:
                            hits = [r for r in self.roles if r["subtask"] == st and r["role"] == role and r["scene"] == scene and r["model"] == name]
                        # a scene row rejected by the geometric filter keeps its row but never a representative / test mark
                        admitted = [r for r in hits if not r["status"].startswith(STATUS_REJECT)]
                        for r in admitted:
                            r["test"] = "1"
                            r["representative"] = "1"
                        if hits and not admitted:
                            self.geo["pins_rejected"].append((st, role, pin, hits[0]["status"]))
                    else:
                        cat, model = pin.split("/", 1)
                        hits = by_key.get((st, role, cat, model), [])
                        for r in hits:
                            r["test"] = "1"
                            if (st, role, cat) not in has_rep:
                                r["representative"] = "1"
                        if hits and (st, role, cat) not in has_rep:
                            has_rep.add((st, role, cat))
                    if not hits:
                        print(f"WARNING pin without candidate row: {st} {role} {pin} (add the row to its source, e.g. A5 scene inventory)")
        # attachment: the representative male model is the best-ranked one that has a READY partner (cap/hbafeb and
        # blackberry/tvpfvb have none); a category without any ready partner has no representative
        partner_ready: Dict[Tuple[str, str], bool] = collections.defaultdict(bool)
        for pr in self.pairs:
            if pr["subtask"] == f"{CM}-attachment" and pr["relation"] == "partner_of" and pr["ready"] != "0":
                partner_ready[(pr["category_a"], pr["model_a"])] = True
        by_cat: Dict[str, List[dict]] = collections.defaultdict(list)
        for r in self.roles:
            if r["subtask"] == f"{CM}-attachment" and r["role"] == "target":
                by_cat[r["category"]].append(r)
        for cat, rows in by_cat.items():
            rows.sort(key=lambda r: int(r["rank"] or 9))
            ready_rows = [r for r in rows if partner_ready[(cat, r["model"])]]
            for r in rows:
                if r["representative"] == "1" and not partner_ready[(cat, r["model"])]:
                    r["representative"] = ""
                    r["note"] = "not the representative: no READY attachment partner; " + r["note"]
            if ready_rows and not any(r["representative"] == "1" for r in ready_rows):
                ready_rows[0]["representative"] = "1"

    # -- require/ sheets
    @staticmethod
    def _axes_from_quat_xyzw(q) -> Tuple[List[float], List[float]]:
        """Approach (eef +z) and closing (eef +y) axes of an eef orientation given as xyzw."""
        x, y, z, w = [float(v) for v in q]
        # rotation matrix columns
        col_y = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
        col_z = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        return [round(v, 4) for v in col_z], [round(v, 4) for v in col_y]

    @staticmethod
    def _rel(path: str) -> str:
        return os.path.relpath(path, REPO)

    def build_grasps(self) -> List[dict]:
        out = []
        if GL is None:
            return out
        for key in sorted(self.keys_used):
            res = self.results.get(key, {})
            pipeline = GRASP_PIPELINE.get(res.get("constraint", ""), "")
            common = {"key": key, "category": res.get("category", ""), "model": res.get("model", ""),
                      "require": f"grasps:{pipeline}" if pipeline else "", "kind": f"{res.get('pipeline', '')}/{res.get('constraint', '')}".strip("/"),
                      "path": self._rel(os.path.join(LIB, "results", f"{key}.json")), "source": os.path.basename(LIB),
                      "task": res.get("task", ""), "pipeline": pipeline, "scale": json.dumps(res.get("scale")) if res.get("scale") else ""}
            try:
                grasps = GL.load_grasps(key)
            except Exception as e:
                out.append({**empty_row(GRASP_COLUMNS), **common, "verdict": "none", "flags": f"load failed: {e}"})
                continue
            cm = res.get("candidates_meta", {})
            for g in grasps:
                fq = list(g.finger_q) if g.finger_q else ["", ""]
                width = g.meta.get("width")
                q = [round(float(v), 5) for v in g.quat.tolist()]
                a, c = self._axes_from_quat_xyzw(q)
                row = empty_row(GRASP_COLUMNS)
                row.update(common)
                row.update({
                    "name": g.name, "position": json.dumps([round(float(v), 4) for v in g.position.tolist()]),
                    "axis_approach": json.dumps(a), "axis_close": json.dumps(c), "quat_xyzw": json.dumps(q),
                    "rest_quat_wxyz": json.dumps(cm.get("rest_quat_wxyz")) if cm.get("rest_quat_wxyz") else "",
                    "finger_q1": fq[0] if len(fq) > 0 else "", "finger_q2": fq[1] if len(fq) > 1 else "",
                    "width_m": width if width is not None else "",
                    "pad_clearance_m": f"{(GRIPPER_OPEN_M - float(width)) / 2:.4f}" if width is not None else "",
                    "com_perp_m": g.meta.get("com_perp", ""), "antipodal": g.meta.get("antipodal", ""), "coverage": g.meta.get("coverage", ""),
                    "tilt_deg": g.meta.get("tilt_deg", ""), "verdict": g.verdict, "flags": ";".join(g.flags or []), "note": (g.note or "")[:200],
                })
                out.append(row)
        return out

    def _handle_rows(self, d: dict, path: str, scene: str = "", instance: str = "", reviews: Optional[dict] = None,
                     manual: Optional[dict] = None, **extra) -> List[dict]:
        """One row per handle candidate of every joint of one annotation file (canonical model or scaled scene instance).
        Candidate names follow the review strips: ``<joint>#A..D`` protruding handles, ``#F`` the edge fallback, ``#M`` a
        reviewer-placed manual pinch."""
        rows = []
        key, cat, model = d.get("key", ""), d.get("category", ""), d.get("model", "")
        for j in d.get("joints", []):
            joint = j["joint"]
            rev = (reviews or {}).get((cat, model, joint), {})
            man = ((manual or {}).get(key) or {}).get(joint) or {}
            chosen_marker = rev.get("review_chosen", "") if rev.get("review_chosen", "") not in ("", "none") else ""
            if man.get("use"):
                chosen_marker = man["use"]
            base = {**empty_row(HANDLE_COLUMNS), "key": key, "category": cat, "model": model, "scene": scene, "instance": instance,
                    "require": "handle", "path": self._rel(path), "source": os.path.basename(os.path.dirname(os.path.dirname(path))) if scene else "grasp_library_20260905/artic/handles",
                    "joint": joint, "joint_type": j.get("type", ""), "joint_kind": j.get("kind", ""), "status": j.get("status", ""),
                    "axis_dir": json.dumps(j.get("axis_dir_obj")), "hinge_point": json.dumps(j.get("hinge_point_obj")),
                    "lower": j.get("lower", ""), "upper": j.get("upper", ""), "closed_q": j.get("closed_q", ""),
                    "front_free_m": j.get("front_free_m", ""), "front_free_eroded_m": j.get("front_free_eroded_m", ""),
                    "review_graspable": rev.get("review_graspable", ""), "review_pinch": rev.get("review_pinch", ""),
                    "note": (rev.get("handle_desc", "") or (extra.get("prior_review") or ""))[:200]}
            base.update({k: v for k, v in extra.items() if k in base and k != "note"})
            cands = [(f"{joint}#{chr(65 + i)}", h, h.get("kind", "")) for i, h in enumerate(j.get("handles", []))]
            if j.get("fallback"):
                cands.append((f"{joint}#F", j["fallback"], j["fallback"].get("kind", "")))
            if man.get("contact"):
                cands.append((f"{joint}#M", man, man.get("kind", "edge_manual")))
            scale = extra.get("instance_scale")
            unit_scale = not scale or all(abs(float(v) - 1.0) <= 0.02 for v in json.loads(scale))
            for name, h, kind in cands:
                r = dict(base)
                exposed = h.get("exposed")
                if name.endswith("#M"):  # reviewer-placed pinch, given in the canonical object frame
                    verdict = "manual" if unit_scale else "manual_unscaled"
                elif kind in ("bar", "knob"):
                    verdict = "handle"
                else:
                    verdict = "edge" if exposed in (True, None) else "edge_blocked"
                if man.get("skip"):
                    verdict = "skip"
                r.update({"name": name, "kind": kind, "verdict": verdict, "chosen": "1" if name.endswith("#" + chosen_marker) and chosen_marker else "",
                          "position": json.dumps(h.get("contact")), "axis_approach": json.dumps(h.get("approach_dir")),
                          "axis_close": json.dumps(h.get("pinch_axis")), "pull_dir": json.dumps(h.get("pull_dir")) if h.get("pull_dir") else "",
                          "width_m": h.get("width", ""), "length_m": h.get("length", ""), "depth_m": h.get("depth", ""),
                          "exposed": "" if exposed is None else ("1" if exposed else "0"),
                          "position_world": json.dumps(h.get("contact_world")) if h.get("contact_world") else "",
                          "height_world_m": h.get("height_world", ""),
                          "flags": ";".join(f for f in (("manual_skip: " + man["skip"]) if man.get("skip") else "",
                                                        ("manual: " + man["note"]) if man.get("note") and name.endswith("#M") else "",
                                                        "manual contact is in the canonical frame: rescale to the instance" if verdict == "manual_unscaled" else "",
                                                        h.get("note", "") if kind.startswith("edge") and not name.endswith("#M") else "") if f)[:200]})
                rows.append(r)
            if not cands:
                r = dict(base)
                r.update({"name": f"{joint}#-", "kind": "none", "verdict": "skip" if man.get("skip") else "ungraspable",
                          "flags": ("manual_skip: " + man["skip"]) if man.get("skip") else "no protruding handle and no exposed edge"})
                rows.append(r)
        return rows

    def build_handles(self) -> List[dict]:
        """require/handles.csv: canonical handle annotations (grasp library artic/handles) with the reviewer verdicts, plus the
        per-instance scaled annotations of the scene articulated objects (scene inventory, B4)."""
        reviews = {}
        try:
            for r in csv.DictReader(open(os.path.join(ARTIC, "artic_results.csv"))):
                reviews.setdefault((r["category"], r["model"], r["joint"]), r)
        except FileNotFoundError:
            pass
        manual = {}
        mp = os.path.join(ARTIC, "manual_handles.json")
        if os.path.exists(mp):
            manual = json.load(open(mp))
        out: List[dict] = []
        for path in sorted(glob.glob(os.path.join(ARTIC, "handles", "*.json"))):
            d = json.load(open(path))
            out += self._handle_rows(d, path, reviews=reviews, manual=manual)
        for path in sorted(glob.glob(os.path.join(SCENE_INVENTORY, "*", "artic", "*.json"))):
            d = json.load(open(path))
            out += self._handle_rows(d, path, scene=d.get("scene", ""), instance=d.get("name", ""), reviews=reviews, manual=manual,
                                     instance_scale=json.dumps(d.get("scale")) if d.get("scale") else "", prior_review=d.get("prior_review") or "")
        return out

    def build_contacts(self) -> List[dict]:
        """require/contacts.csv from ``runlogs/contacts_<date>/<category>__<model>.json`` (B6). Expected file content:
        ``{"key", "category", "model", "scale", "contacts": [{"name", "role": "push|brace|stop_edge", "position", "approach",
        "closing", "direction", "shape_class", "verdict", "flags", "note"}]}`` (object base_link frame, metres). Header-only
        while no such directory exists."""
        out: List[dict] = []
        dirs = sorted(glob.glob(CONTACTS_GLOB))
        if not dirs:
            return out
        root = dirs[-1]
        for path in sorted(glob.glob(os.path.join(root, "*.json"))):
            d = json.load(open(path))
            for c in d.get("contacts", []):
                out.append({**empty_row(CONTACT_COLUMNS), "key": d.get("key", os.path.splitext(os.path.basename(path))[0]),
                            "category": d.get("category", ""), "model": d.get("model", ""), "require": "contacts", "kind": c.get("role", ""),
                            "name": c.get("name", ""), "position": json.dumps(c.get("position")), "axis_approach": json.dumps(c.get("approach")),
                            "axis_close": json.dumps(c.get("closing")), "width_m": "", "verdict": c.get("verdict", "ok"),
                            "flags": ";".join(c.get("flags", [])), "note": (c.get("note", "") or "")[:200], "path": self._rel(path),
                            "source": os.path.basename(root), "contact_role": c.get("role", ""), "direction": json.dumps(c.get("direction")),
                            "shape_class": c.get("shape_class", ""), "scale": json.dumps(d.get("scale")) if d.get("scale") else ""})
        return out

    # -- rotational symmetry (survey runlog)
    def apply_symmetry(self, path: Optional[str]) -> Dict[str, int]:
        """``symmetry`` / ``symmetry_near`` of every sampleable / group candidate from ``symmetry.json`` of
        ``scripts/symmetry/survey_symmetry.py`` (comma-separated generator tags ``x2 y2 z2 x4 y4 z4 z3 z5 z6 xinf yinf
        zinf full``; ``none`` = surveyed and asymmetric / nothing to review; empty = not surveyed). A row whose grasp-tested
        ``scale`` matches the surveyed scale gets the scaled variant's verdict (a non-uniform scale can break a symmetry).

        When ``review_verdicts.json`` (``scripts/symmetry/aggregate_review.py``, the reviewer round) sits next to
        ``symmetry.json``, its per-model verdicts replace the survey's for the reviewed models (``preserved`` tags joined
        ``symmetry``, ``broken`` ones left ``symmetry_near``). ``symmetric_180`` (the selection reviewer's column of the
        alignment targets) is only changed where the review answered the disagreement question of that (subtask,
        model) explicitly: ``preserved`` -> ``1``, ``broken`` -> ``0``; otherwise the reviewer's value stays."""
        stats = {"surveyed": 0, "scaled": 0, "not_surveyed": 0, "reviewed": 0, "symmetric_180_set": 0}
        self.symmetry_source: Dict[str, str] = {}  # key -> "survey" | "review" (provenance for the dropped-rows file)
        if not path or not os.path.exists(path):
            return stats
        sym = json.load(open(path))
        review_path = os.path.join(os.path.dirname(path), "review_verdicts.json")
        review = json.load(open(review_path)).get("models", {}) if os.path.exists(review_path) else {}
        for r in self.roles:
            if r["kind"] not in (KIND_SAMPLEABLE, KIND_GROUP) or not r["model"]:
                continue
            key = f"{r['category']}__{r['model']}"
            rec = sym.get(key)
            if rec is None:
                stats["not_surveyed"] += 1
                continue
            rv = review.get(key)
            src = rv if rv else rec
            tags, near = src.get("symmetry") or ["none"], src.get("symmetry_near") or []
            if r["scale"] and src.get("symmetry_scaled") and rec.get("scale") == json.loads(r["scale"]):
                tags, near = src["symmetry_scaled"] or ["none"], src.get("symmetry_near_scaled") or []
                stats["scaled"] += 1
            r["symmetry"], r["symmetry_near"] = ",".join(tags), ",".join(near) if near else "none"
            self.symmetry_source[key] = "review verdict" if rv else "symmetry survey"
            stats["surveyed"] += 1
            if rv:
                stats["reviewed"] += 1
                if r["subtask"].startswith("alignment") and r["role"] == "target":
                    for q in rv.get("symmetric_180", []):
                        if q.get("subtask") == r["subtask"] and q.get("value") in ("0", "1"):
                            r["symmetric_180"] = q["value"]
                            stats["symmetric_180_set"] += 1
        return stats

    # -- spec 3.13 rule 1: mental-rotation targets must be asymmetric
    def drop_symmetric_mental_rotation(self, dropped_csv: Optional[str]) -> dict:
        """Remove every ``mental_rotation`` / ``target`` row whose ``symmetry`` is a confirmed symmetry (``symmetry != none``;
        empty = not surveyed and ``symmetry_near``-only rows stay until a review verdict confirms the tag - the drop then
        applies on the next build). A dropped ``representative`` / ``test`` mark is handed to the next-ranked remaining
        model of the category (by ``rank``, then the original row order): the first remaining model becomes the
        representative, the test set is refilled to its previous size (a category with one remaining model keeps one);
        the promoted row's ``note`` gets ``promoted: replaces <model> (<tags>)``, every other mark stays. A category left
        without a model is reported, never refilled. The dropped rows go to ``dropped_csv`` (candidates.csv columns +
        ``reason``) so the removal stays traceable."""
        st, role = "mental_rotation", "target"
        rows = [r for r in self.roles if r["subtask"] == st and r["role"] == role]
        dropped = [r for r in rows if r["symmetry"] and r["symmetry"] != "none"]
        stats = {"dropped": [], "promotions": [], "empty_categories": []}
        by_cat: Dict[str, List[dict]] = collections.OrderedDict()
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        dropped_ids = {id(r) for r in dropped}
        for cat, crows in by_cat.items():
            gone = [r for r in crows if id(r) in dropped_ids]
            if not gone:
                continue
            remaining = [r for r in crows if id(r) not in dropped_ids]
            for r in gone:
                stats["dropped"].append((cat, r["model"], r["symmetry"], r["representative"] == "1", r["test"] == "1"))
            if not remaining:
                stats["empty_categories"].append(cat)
                continue
            order = sorted(remaining, key=lambda r: (int(r["rank"] or 9), crows.index(r)))
            gone_reps = [r for r in gone if r["representative"] == "1"]
            gone_tests = [r for r in gone if r["test"] == "1"]
            if gone_reps and not any(r["representative"] == "1" for r in remaining):
                new_rep = order[0]
                new_rep["representative"] = "1"
                new_rep["note"] = f"promoted: replaces {gone_reps[0]['model']} ({gone_reps[0]['symmetry']}); " + new_rep["note"]
                stats["promotions"].append((cat, new_rep["model"], "representative", gone_reps[0]["model"], gone_reps[0]["symmetry"]))
            n_test_before = sum(1 for r in crows if r["test"] == "1")
            n_test = sum(1 for r in remaining if r["test"] == "1")
            k = 0
            for r in order:
                if n_test >= n_test_before:
                    break
                if r["test"] != "1":
                    r["test"] = "1"
                    n_test += 1
                    src = gone_tests[min(k, len(gone_tests) - 1)] if gone_tests else gone[0]
                    k += 1
                    tag = f"promoted: replaces {src['model']} ({src['symmetry']}); "
                    if not r["note"].startswith(tag):
                        r["note"] = tag + r["note"]
                    stats["promotions"].append((cat, r["model"], "test", src["model"], src["symmetry"]))
        self.roles = [r for r in self.roles if id(r) not in dropped_ids]
        if dropped_csv:
            os.makedirs(os.path.dirname(dropped_csv), exist_ok=True)
            with open(dropped_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ROLE_COLUMNS + ["reason"])
                w.writeheader()
                for r in dropped:
                    key = f"{r['category']}__{r['model']}"
                    w.writerow({**{c: r.get(c, "") for c in ROLE_COLUMNS},
                                "reason": f"mental_rotation target with symmetry {r['symmetry']} ({self.symmetry_source.get(key, 'symmetry survey')}; "
                                          f"spec 3.13 rule 1, decision 2026-09-15)"})
        return stats

    def build_lookalikes(self) -> List[dict]:
        out = []
        groups = (self.spec.get("disembedding-0", {}).get("distractors", {}) or {}).get("lookalike_groups", [])
        for i, g in enumerate(groups):
            members = g["members"] if isinstance(g, dict) else g
            reason = g.get("reason", "") if isinstance(g, dict) else ""
            for m in members:
                cat, model = m.split("/", 1)
                out.append({"relation": "lookalike", "group_id": f"g{i:02d}", "category": cat, "model": model, "reason": reason,
                            "source": "task_custom_lists.json (disembedding-0.distractors.lookalike_groups)"})
        return out

    @staticmethod
    def link_libraries(req_dir: str):
        """``require/lib/<name>`` -> the library directories the ``path`` columns refer to (relative symlinks, replaced on
        every run; a library that does not exist yet gets no link)."""
        lib_dir = os.path.join(req_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        targets = {"grasps": LIB, "handles": os.path.join(ARTIC, "handles"), "scene_inventory": SCENE_INVENTORY}
        contacts = sorted(glob.glob(CONTACTS_GLOB))
        if contacts:
            targets["contacts"] = contacts[-1]
        for name, target in targets.items():
            link = os.path.join(lib_dir, name)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            if os.path.isdir(target):
                os.symlink(os.path.relpath(target, lib_dir), link)

    def run(self, out_dir: str, symmetry: Optional[str] = None):
        self.build_camera_relocalization()
        self.build_held()
        self.build_disembedding()
        self.build_block_design()
        self.build_articulation()
        self.build_attachment()
        self.build_alignment()
        self.build_iors()
        n_sup = self.build_supports()
        self.mark_pins()
        sym_stats = self.apply_symmetry(symmetry)
        drop_stats = self.drop_symmetric_mental_rotation(
            os.path.join(os.path.dirname(symmetry), "dropped_mental_rotation.csv") if symmetry and os.path.exists(symmetry) else None)
        grasps = self.build_grasps()
        handles = self.build_handles()
        contacts = self.build_contacts()
        lookalikes = self.build_lookalikes()
        roles_dir, req_dir = os.path.join(out_dir, "roles"), os.path.join(out_dir, "require")
        os.makedirs(roles_dir, exist_ok=True)
        os.makedirs(req_dir, exist_ok=True)
        for path, rows, cols in ((os.path.join(roles_dir, "candidates.csv"), self.roles, ROLE_COLUMNS),
                                 (os.path.join(roles_dir, "constraints_compatible.csv"), self.pairs, PAIR_COLUMNS),
                                 (os.path.join(roles_dir, "constraints_conflicting.csv"), lookalikes, CONFLICT_COLUMNS),
                                 (os.path.join(req_dir, "grasps.csv"), grasps, GRASP_COLUMNS),
                                 (os.path.join(req_dir, "handles.csv"), handles, HANDLE_COLUMNS),
                                 (os.path.join(req_dir, "contacts.csv"), contacts, CONTACT_COLUMNS)):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({c: r.get(c, "") for c in cols})
        self.link_libraries(req_dir)
        # summary
        by = collections.Counter((r["subtask"], r["role"]) for r in self.roles)
        print(f"roles/candidates.csv: {len(self.roles)} rows; supports: {n_sup} {'admitted' if self.inv is not None else 'union'} scene objects; "
              f"wall regions: {len(self.walls)}; representative rows: {sum(r['representative'] == '1' for r in self.roles)}, "
              f"test rows: {sum(r['test'] == '1' for r in self.roles)}")
        for (st, role), n in sorted(by.items()):
            print(f"  {st:58s} {role:12s} {n}")
        if self.inv is not None:
            self.report_scene_rows()
        else:
            print("scene rows: no --scene-inventory -> hand-copied seeds (status untested)")
        if symmetry and os.path.exists(symmetry):
            unsurveyed = collections.Counter((r["subtask"], r["role"]) for r in self.roles
                                             if r["kind"] in (KIND_SAMPLEABLE, KIND_GROUP) and r["model"] and not r["symmetry"])
            review_path = os.path.join(os.path.dirname(symmetry), "review_verdicts.json")
            print(f"symmetry ({os.path.relpath(symmetry, REPO)}): {sym_stats['surveyed']} candidate rows with symmetry / symmetry_near "
                  f"({sym_stats['scaled']} at the grasp-tested scale), {sym_stats['not_surveyed']} sampleable/group rows not surveyed: "
                  + ", ".join(f"{st}/{role} {n}" for (st, role), n in sorted(unsurveyed.items())))
            if os.path.exists(review_path):
                print(f"  review verdicts ({os.path.relpath(review_path, REPO)}): {sym_stats['reviewed']} rows use the reviewed values, "
                      f"symmetric_180 set from the review on {sym_stats['symmetric_180_set']} alignment rows")
            else:
                print("  no review_verdicts.json next to symmetry.json -> survey values only (run scripts/symmetry/aggregate_review.py after the review round)")
            print(f"  symmetry values: {collections.Counter(r['symmetry'] for r in self.roles if r['symmetry']).most_common(12)}")
            print(f"  mental_rotation targets dropped (symmetry != none, spec 3.13 rule 1): {len(drop_stats['dropped'])} -> "
                  f"{os.path.relpath(os.path.join(os.path.dirname(symmetry), 'dropped_mental_rotation.csv'), REPO)}")
            for cat, model, tags, rep, test in drop_stats["dropped"]:
                print(f"    dropped {cat}/{model} ({tags}){' representative' if rep else ''}{' test' if test else ''}")
            for cat, model, mark, old, tags in drop_stats["promotions"]:
                print(f"    promoted {cat}/{model} -> {mark} (replaces {old} ({tags}))")
            for cat in drop_stats["empty_categories"]:
                print(f"    WARNING category {cat} has no mental_rotation target left")
        else:
            print("symmetry: no survey file given / found -> symmetry / symmetry_near columns left empty, no mental_rotation rows dropped")
        print(f"roles/constraints_compatible.csv: {len(self.pairs)} rows ({collections.Counter(p['relation'] for p in self.pairs)})")
        print(f"roles/constraints_conflicting.csv: {len(lookalikes)} rows")
        print(f"require/grasps.csv: {len(grasps)} rows from {len(self.keys_used)} keys ({collections.Counter(g['require'] for g in grasps)})")
        print(f"require/handles.csv: {len(handles)} rows ({collections.Counter(h['verdict'] for h in handles)}; "
              f"{sum(1 for h in handles if h['scene'])} scene-instance rows)")
        print(f"require/contacts.csv: {len(contacts)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "ispatialgym", "assets", "icl"))
    ap.add_argument("--symmetry", default=SYMMETRY_DEFAULT if os.path.exists(SYMMETRY_DEFAULT) else None,
                    help="symmetry.json of scripts/symmetry/survey_symmetry.py (default: the latest committed survey runlog if it exists)")
    ap.add_argument("--scene-inventory", default=SCENE_INVENTORY if os.path.isdir(SCENE_INVENTORY) else None, metavar="DIR",
                    help="B4 scene inventory (summary.json + <scene>/{supports,partners,artic}/<name>.json): regenerates the scene rows - "
                         "supports of the tabletop subtasks, large partners and their nearby supports, scene articulated targets - with "
                         "the geometric filters of spec 3.2 / 5.6 / 6.12 (status geometric_ok / geometric_reject:<reason>). Default: "
                         "runlogs/scene_inventory_20260915 if it exists; 'none' keeps the hand-copied seeds (status untested)")
    args = ap.parse_args()
    inv = None if not args.scene_inventory or args.scene_inventory.lower() == "none" else args.scene_inventory
    if inv and not os.path.exists(os.path.join(inv, "summary.json")):
        ap.error(f"--scene-inventory {inv}: no summary.json there")
    Builder(scene_inventory=inv).run(args.out, symmetry=args.symmetry)


if __name__ == "__main__":
    main()
