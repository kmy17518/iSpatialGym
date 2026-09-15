#!/usr/bin/env python3
"""Build the icl (iSpatialGym candidate list) from the selection runlogs.

Inputs (all committed under ``runlogs/``):

* ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` - the reviewed selection sheet (old task ids /
  role names), incl. distractor pools, attachment partners, alignment frames, obstacle naturalness, table / wall rows,
* ``runlogs/grasp_library_20260905/select/test_instances.csv`` - pick keys / scores per (task, category),
* ``runlogs/grasp_library_20260905/results/<key>.json`` + ``cands/<key>.yaml`` + ``review/`` - the grasp library,
* ``runlogs/grasp_library_20260905/artic/{artic_results.csv,scene_assign.json,handles/}`` - articulated objects,
* ``runlogs/table_selection_20260911/inventory.csv`` - geometry of every work surface / big partner in the 7 scenes,
* ``runlogs/world_alignment_selection_20260911/wall_candidates.json`` - the 78 validated wall regions,
* ``ispatialgym/assets/task_specs/task_custom_lists.json`` - the current test pins and the look-alike groups.

Outputs (``ispatialgym/assets/icl/``): ``roles.csv``, ``pairs.csv``, ``lookalikes.csv``, ``combos.csv``, ``grasps.csv``
and ``README.md`` (schema). Every candidate gets ``status = untested``: the old "viable" verdicts came from the sampler
this rebuild replaces, so they are provenance (``source_selection``), not status.

Run inside the ``ispatialgym`` conda env (BDDL knowledge base for synsets / bounding boxes / readiness; graceful
degradation without it)::

    python scripts/icl/build_icl.py [--out ispatialgym/assets/icl]
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
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
sys.path.insert(0, REPO)

try:  # knowledge base (synsets, bboxes, readiness); optional
    from ispatialgym.sampling import knowledge as K  # type: ignore
except Exception:  # pragma: no cover
    K = None
try:
    from ispatialgym.sampling import grasp_library as GL  # type: ignore
except Exception:  # pragma: no cover
    GL = None

SCENES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "hotel_suite_large",
          "restaurant_diner", "office_cubicles_right"]
IORS = "inter_object_relation_satisfaction"
CM = "configuration_matching"
TABLETOP_SUBTASKS = ["disembedding", f"{CM}-block_design", f"{CM}-articulation_open_small_scale", f"{CM}-articulation_close_small_scale",
                     f"{CM}-attachment", "alignment-axial", f"{IORS}-ontop_small_scale", f"{IORS}-under_small_scale", f"{IORS}-inside",
                     f"{IORS}-nextto", f"{IORS}-touching", f"{IORS}-ontop_large_scale", f"{IORS}-under_large_scale"]
WALL_SUBTASKS = ["alignment-world_vertical", "alignment-world_horizontal"]
ONTOP_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table", "countertop")
UNDER_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table")
SCENE_ARTIC_CATEGORIES = ("bottom_cabinet", "bottom_cabinet_no_top", "top_cabinet", "fridge", "oven", "dishwasher", "door")
RANK = {"spec": 0, "pick 1": 1, "pick 2": 2, "alternate": 3, "candidate": 4, "viable": 4, "partner": 4, "pool": 5, "reserve": 6}
TEST_SUPPORT = ("house_single_floor", "breakfast_table_rhjoby_0")

ROLE_COLUMNS = ["subtask", "role", "kind", "category", "model", "display_name", "synset", "scene", "room", "representative", "test",
                "rank", "source_selection", "status", "grasp_key", "scale", "bbox_x", "bbox_y", "bbox_z", "natural_scenes",
                "frame_front", "frame_up", "frame_level", "frame_long", "symmetric_180", "joint", "handle_key", "handle_kind",
                "handle_annotated", "instance_scale", "region_id", "p0", "p1", "normal", "hang_z", "length_m", "max_object_m",
                "top_z", "world_ext_x", "world_ext_y", "pos_x", "pos_y", "yaw_deg", "dataset", "count", "ready", "note", "source"]
PAIR_COLUMNS = ["subtask", "relation", "role_a", "category_a", "model_a", "role_b", "category_b", "model_b", "tier", "attach_type",
                "ready", "note", "source"]
COMBO_COLUMNS = ["combo_id", "subtask", "kind", "test", "scene", "assignment", "note"]
GRASP_COLUMNS = ["key", "category", "model", "task", "pipeline", "constraint", "rest_quat_wxyz", "scale", "grasp", "finger_q1", "finger_q2",
                 "width_m", "pad_clearance_m", "com_perp_m", "antipodal", "coverage", "tilt_deg", "verdict", "flags"]
GRIPPER_OPEN_M = 0.10  # two fingers at 0.05 each

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
    def __init__(self):
        self.blocks = read_selection_blocks()
        self.tests = read_test_instances()
        self.inventory = read_inventory()
        self.walls = read_wall_regions()
        self.results = grasp_results_index()
        self.spec = json.load(open(SPEC))
        self.roles: List[dict] = []
        self.pairs: List[dict] = []
        self.combos: List[dict] = []
        self.keys_used: set = set()

    # -- generic object row
    def add_object(self, subtask: str, role: str, category: str, row: dict, kind: str = "object", **extra) -> dict:
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
        r.update({"subtask": subtask, "role": role, "kind": "fixed", "category": category, "model": name, "display_name": display_name(category),
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
            self.add_object("camera_relocalization-obstructed", "obstacles", cat, row, kind="group", natural_scenes=nat, count="1")

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
                                                          "duplicates": row.get("duplicates", "")}, kind="group")
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
                                                            "duplicates": row.get("duplicates", "")}, kind="group")
        for old_role, new_role in (("tray_source", "source"), ("tray_dest", "destination")):
            for cat, row in self.blocks[("disembedding-0", old_role)]:
                self.add_object(st, new_role, cat, row, note="fixed model for now (decision 2026-09-14); " + row.get("note", ""))

    def build_block_design(self):
        st = f"{CM}-block_design"
        for cat, row in self.blocks[("configuration_matching-0 (block_design)", "blocks")]:
            self.add_object(st, "targets", cat, row, kind="group", dataset="ispatialgym-assets", count="4",
                            note="custom Blender Kohs block; 4 blocks (2x2) for now; " + row.get("note", ""))

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
        # large-scale: the actual articulated scene objects (hosts of the swap tests), per scene, with their instance scale
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
        for old, new in (("world_alignment-0 (table_surface)", "alignment-axial"), ("world_alignment-1 (world_vertical)", "alignment-world_vertical"),
                         ("world_alignment-2 (world_horizontal)", "alignment-world_horizontal")):
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
        # large partners (fixed scene objects) from the inventory
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
        """Union of every support that appears as spec / viable for any task (31 scene objects) - candidates for every tabletop subtask."""
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
        # floor as an admissible init support for the large-scale IORS subtasks
        for st in (f"{IORS}-ontop_large_scale", f"{IORS}-under_large_scale"):
            for scene in SCENES:
                self.add_fixed(st, "support", scene, "floor", "floors", note="floor init: the target starts on the floor beside the partner (explicit support choice, never a fallback)",
                               source="decision 2026-09-14")
        return len(union)

    # -- combos
    def build_combos(self):
        cl = self.spec
        sup = f"support={TEST_SUPPORT[0]}:{TEST_SUPPORT[1]}"
        def add(cid, st, kind, scene, assignment, note=""):
            self.combos.append({"combo_id": cid, "subtask": st, "kind": kind, "test": "1" if kind == "spec" else "", "scene": scene,
                                "assignment": assignment, "note": note})
        add("spec_camera_standard", "camera_relocalization-standard", "spec", "house_single_floor", "room=living_room_0", "current test spec")
        add("spec_object_scaling", "object_scaling", "spec", "house_single_floor", "target=bottle_of_beer/gxxbhh;room=living_room_0", "current test spec")
        add("spec_mental_rotation", "mental_rotation", "spec", "house_single_floor", "target=toy_figure/zuntus;room=living_room_0", "current test spec")
        add("spec_disembedding", "disembedding", "spec", "house_single_floor",
            f"target=box_of_cereal/iucshm;source=tray/gsxbym;destination=tray/hvlfig;{sup}", "distractors drawn from the pools at ground time")
        add("spec_block_design", f"{CM}-block_design", "spec", "house_single_floor", f"targets=toy_dice/isgblk*4;{sup}", "pattern test_rbbr")
        for d in ("open", "close"):
            add(f"spec_articulation_{d}_small", f"{CM}-articulation_{d}_small_scale", "spec", "house_single_floor", f"target=toolbox/redmtl(j_lid);{sup}")
        add("spec_attachment", f"{CM}-attachment", "spec", "house_single_floor", f"target=cap/xxqyrt;partner=lemonade_bottle/yprkek;{sup}")
        for m in ("hgtbot", "injrgc"):
            add(f"spec_axial_{m}", "alignment-axial", "spec", "house_single_floor", f"target=keyboard/{m};{sup}")
        for m in ("dacihj", "ljabhq"):
            add(f"spec_world_vertical_{m}", "alignment-world_vertical", "spec", "house_single_floor", f"target=painting/{m};support=house_single_floor:living_room_0/*", "any validated region of the room")
        for m in ("kqrmrh", "crivoa"):
            add(f"spec_world_horizontal_{m}", "alignment-world_horizontal", "spec", "house_single_floor", f"target=towel_rack/{m};support=house_single_floor:living_room_0/*")
        add("spec_iors_ontop_small", f"{IORS}-ontop_small_scale", "spec", "house_single_floor", f"target=pack_of_kielbasa/dgodnn;partner=platter/csanbr;{sup}")
        add("spec_iors_under_large", f"{IORS}-under_large_scale", "spec", "house_single_floor",
            "target=toy_figure/cobfpg;partner=house_single_floor:bed_rrcvaq_0;support=house_single_floor:desk_xhjsub_0", "childs_room_0 test pair")
        add("spec_iors_inside", f"{IORS}-inside", "spec", "house_single_floor", f"target=lemon/ovbcff;partner=bowl/belcml;{sup}")
        add("spec_iors_nextto", f"{IORS}-nextto", "spec", "house_single_floor", f"target=spatula/hkvsep;partner=soup_ladle/cfzneb;{sup}")
        for t in ("blrqqz", "orwvfx"):
            for p in ("eoywpy", "gcfkqa"):
                add(f"spec_iors_touching_{t}_{p}", f"{IORS}-touching", "spec", "house_single_floor", f"target=can_of_baking_mix/{t};partner=board_game/{p};{sup}")
        # representative combos: pick 1 of every target category with the pick-1 partner of its paired category on the test support
        reps = collections.defaultdict(dict)  # (subtask, role) -> category -> model
        for r in self.roles:
            if r["representative"] == "1" and r["kind"] in ("object", "group"):
                reps[(r["subtask"], r["role"])].setdefault(r["category"], r["model"])
        partner_of = collections.defaultdict(list)
        for p in self.pairs:
            if p["relation"] == "partner_of":
                partner_of[(p["subtask"], p["category_a"], p["model_a"])].append((p["category_b"], p["model_b"], p["ready"]))
        # attachment: the representative male model is the best-ranked one that has a READY partner (cap/hbafeb and
        # blackberry/tvpfvb have none)
        by_rank = collections.defaultdict(list)
        for r in self.roles:
            if r["subtask"] == f"{CM}-attachment" and r["role"] == "target":
                by_rank[r["category"]].append((int(r["rank"] or 9), r["model"]))
        for cat, lst in by_rank.items():
            for _, model in sorted(lst):
                if any(rd != "0" for _, _, rd in partner_of.get((f"{CM}-attachment", cat, model), [])):
                    reps[(f"{CM}-attachment", "target")][cat] = model
                    break
            else:
                reps[(f"{CM}-attachment", "target")].pop(cat, None)  # no ready partner in the category: no representative test
        for (st, role), cats in sorted(reps.items()):
            if role != "target":
                continue
            for cat, model in sorted(cats.items()):
                parts = [f"target={cat}/{model}"]
                partners = partner_of.get((st, cat, model)) or partner_of.get((st, cat, ""))
                if partners:
                    pcat, pmodel, prd = partners[0]
                    if prd == "0" and len(partners) > 1:
                        pcat, pmodel, prd = next(((c, m, rd) for c, m, rd in partners if rd != "0"), partners[0])
                    if not pmodel:
                        pmodel = reps.get((st, "partner"), {}).get(pcat, "")
                    if st.endswith("large_scale"):
                        parts.append(f"partner={pcat}")
                    else:
                        parts.append(f"partner={pcat}/{pmodel}" if pmodel else f"partner={pcat}")
                if st in TABLETOP_SUBTASKS:
                    parts.append(sup)
                elif st in WALL_SUBTASKS:
                    parts.append("support=house_single_floor:living_room_0/*")
                elif st.startswith("camera_relocalization") or st in ("object_scaling", "mental_rotation"):
                    parts.append("room=living_room_0")
                add(f"rep_{st}_{cat}", st, "representative", "house_single_floor", ";".join(parts), "ground_isg representative-model test")

    # -- grasps
    def build_grasps(self) -> List[dict]:
        out = []
        if GL is None:
            return out
        for key in sorted(self.keys_used):
            res = self.results.get(key, {})
            try:
                grasps = GL.load_grasps(key)
            except Exception as e:
                out.append({**empty_row(GRASP_COLUMNS), "key": key, "flags": f"load failed: {e}"})
                continue
            cm = res.get("candidates_meta", {})
            for g in grasps:
                fq = list(g.finger_q) if g.finger_q else ["", ""]
                width = g.meta.get("width")
                row = empty_row(GRASP_COLUMNS)
                row.update({
                    "key": key, "category": res.get("category", ""), "model": res.get("model", ""), "task": res.get("task", ""),
                    "pipeline": res.get("pipeline", ""), "constraint": res.get("constraint", ""),
                    "rest_quat_wxyz": json.dumps(cm.get("rest_quat_wxyz")) if cm.get("rest_quat_wxyz") else "",
                    "scale": json.dumps(res.get("scale")) if res.get("scale") else "", "grasp": g.name,
                    "finger_q1": fq[0] if len(fq) > 0 else "", "finger_q2": fq[1] if len(fq) > 1 else "",
                    "width_m": width if width is not None else "",
                    "pad_clearance_m": f"{(GRIPPER_OPEN_M - float(width)) / 2:.4f}" if width is not None else "",
                    "com_perp_m": g.meta.get("com_perp", ""), "antipodal": g.meta.get("antipodal", ""), "coverage": g.meta.get("coverage", ""),
                    "tilt_deg": g.meta.get("tilt_deg", ""), "verdict": g.verdict, "flags": ";".join(g.flags or []),
                })
                out.append(row)
        return out

    def build_lookalikes(self) -> List[dict]:
        out = []
        groups = (self.spec.get("disembedding-0", {}).get("distractors", {}) or {}).get("lookalike_groups", [])
        for i, g in enumerate(groups):
            members = g["members"] if isinstance(g, dict) else g
            reason = g.get("reason", "") if isinstance(g, dict) else ""
            for m in members:
                cat, model = m.split("/", 1)
                out.append({"group_id": f"g{i:02d}", "category": cat, "model": model, "reason": reason, "source": "task_custom_lists.json (disembedding-0.distractors.lookalike_groups)"})
        return out

    def run(self, out_dir: str):
        self.build_camera_relocalization()
        self.build_held()
        self.build_disembedding()
        self.build_block_design()
        self.build_articulation()
        self.build_attachment()
        self.build_alignment()
        self.build_iors()
        n_sup = self.build_supports()
        self.build_combos()
        grasps = self.build_grasps()
        lookalikes = self.build_lookalikes()
        os.makedirs(out_dir, exist_ok=True)
        for name, rows, cols in (("roles.csv", self.roles, ROLE_COLUMNS), ("pairs.csv", self.pairs, PAIR_COLUMNS),
                                 ("combos.csv", self.combos, COMBO_COLUMNS), ("grasps.csv", grasps, GRASP_COLUMNS),
                                 ("lookalikes.csv", lookalikes, ["group_id", "category", "model", "reason", "source"])):
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({c: r.get(c, "") for c in cols})
        # summary
        by = collections.Counter((r["subtask"], r["role"]) for r in self.roles)
        print(f"roles.csv: {len(self.roles)} rows; supports union: {n_sup} scene objects; wall regions: {len(self.walls)}")
        for (st, role), n in sorted(by.items()):
            print(f"  {st:58s} {role:12s} {n}")
        print(f"pairs.csv: {len(self.pairs)} rows ({collections.Counter(p['relation'] for p in self.pairs)})")
        print(f"combos.csv: {len(self.combos)} rows ({collections.Counter(c['kind'] for c in self.combos)})")
        print(f"grasps.csv: {len(grasps)} rows from {len(self.keys_used)} keys; lookalikes.csv: {len(lookalikes)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "ispatialgym", "assets", "icl"))
    args = ap.parse_args()
    Builder().run(args.out)


if __name__ == "__main__":
    main()
