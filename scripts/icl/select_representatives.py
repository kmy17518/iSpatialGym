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

Inputs: ``ispatialgym/assets/icl/{roles,require}/*.csv`` (built by ``build_icl.py``), the duplicate / texture-variant
reviews under ``runlogs/*/duplicates*.json``, the B4 scene inventory ``runlogs/scene_inventory_20260915/summary.json``.

Outputs (``ispatialgym/assets/icl/roles/``):

* ``representatives.csv`` - one row per (pick, role, slot): ``family, subtask, pick, role, slot, category, model,
  scene, name, joint, kind, grasp_key, scale, geometry_group, exact_group, status, note``; slots are ``representative``,
  ``variation_texture``, ``variation_geometry``, ``heldout`` (heldout rows are the global per-category choice);
* ``candidates_extra.csv`` - candidate rows the icl did not have (same columns as ``candidates.csv`` + ``plan``),
  ``status = incomplete:<reason>``; ``build_icl.py`` / ``ground_isg`` merge them until a survey validates them;
* ``representatives_report.md`` - per subtask: the picks, which variation / generalization axes are unavailable and why.

Run: ``python scripts/icl/select_representatives.py [--out ispatialgym/assets/icl/roles] [--per-family 30]``.
Deterministic (no randomness): greedy by a documented score, ties by icl rank / row order.
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
    "alignment-world_horizontal": {"family": "alignment", "roles": {"target": {"kind": "sampleable", "frame": "wall"}, "support": {"kind": "wall"}}},
    "alignment-axial": {"family": "alignment", "roles": {"target": {"kind": "sampleable", "frame": "axial"}, "support": {"kind": "scene"}}},
    f"{IORS}-ontop_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-under_small_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-inside": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "insert"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-nextto": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-touching": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "sampleable"}, "support": {"kind": "scene"}}},
    f"{IORS}-ontop_large_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "scene_partner"}, "support": {"kind": "scene"}}},
    f"{IORS}-under_large_scale": {"family": IORS, "roles": {"target": {"kind": "sampleable", "pipeline": "tabletop"}, "partner": {"kind": "scene_partner"}, "support": {"kind": "scene"}}},
}
FAMILY_ORDER = ["camera_relocalization", "object_scaling", "mental_rotation", "disembedding", CM, "alignment", IORS]
ONTOP_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table", "countertop")
UNDER_LARGE_PARTNERS = ("bed", "sofa", "desk", "breakfast_table")
#: tabletop-grasp categories the large-scale IORS target pools may be filled from (owner: fill missing candidates;
#: the physical requirement is only "small graspable object that rests on a table / mattress"); validated later
LARGE_TARGET_FILL_FROM = {f"{IORS}-ontop_small_scale", f"{IORS}-nextto", f"{IORS}-touching", "disembedding"}
UNDER_LARGE_MAX_HEIGHT = 0.20  # m: must fit under a 0.30 m clearance with the gripper below the wrist (spec 6.12)

REP_COLUMNS = ["family", "subtask", "pick", "role", "slot", "category", "model", "scene", "name", "joint", "kind", "grasp_key",
               "scale", "geometry_group", "exact_group", "status", "note"]


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
    def __init__(self, icl_dir: str, per_family: int, min_per_subtask: int):
        self.per_family = per_family
        self.min_per_subtask = min_per_subtask
        self.cands = read_csv(os.path.join(icl_dir, "roles", "candidates.csv"))
        self.compat = read_csv(os.path.join(icl_dir, "roles", "constraints_compatible.csv"))
        self.conflict = read_csv(os.path.join(icl_dir, "roles", "constraints_conflicting.csv"))
        self.grasps = admissible_grasps(read_csv(os.path.join(icl_dir, "require", "grasps.csv")))
        self.handles = admissible_handles(read_csv(os.path.join(icl_dir, "require", "handles.csv")))
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
        self.by_subtask_role: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
        for r in self.cands:
            self.by_subtask_role[(r["subtask"], r["role"])].append(r)
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

    def support_slots(self, subtask: str, pick: str) -> List[str]:
        """Representative support = the test pin (breakfast_table_rhjoby_0); variation = a support of a different height
        (>= 10 cm) in the same scene; held-out = a support of another category / scene."""
        notes = []
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
        for slot, s in (("representative", rep), ("variation_geometry", var), ("heldout", held)):
            if s is None:
                continue
            self.rows.append({"family": SUBTASKS[subtask]["family"], "subtask": subtask, "pick": pick, "role": "support", "slot": slot,
                              "category": s["category"], "model": "", "scene": s["scene"], "name": s["name"], "joint": "", "kind": "scene",
                              "grasp_key": "", "scale": "", "geometry_group": "", "exact_group": "",
                              "status": "geometric_ok" if s["run"] >= 0.6 else "geometric_weak",
                              "note": f"top_z={s['top_z']:.2f} area={s['area']:.2f} best_edge={s['edge']} run045={s['run']:.2f}"})
        return notes

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
            # is the second choice (spec 5.6 b); floor is always a third
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
                snote.append("no admitted support within 3 m of the partner: init surface = floor" + (" or the partner's own top" if under else ""))
                self.rows.append({"family": IORS, "subtask": subtask, "pick": pick, "role": "support", "slot": "representative", "category": "floor",
                                  "model": "", "scene": prep["scene"], "name": "floor", "joint": "", "kind": "scene", "grasp_key": "", "scale": "",
                                  "geometry_group": "", "exact_group": "", "status": "untested", "note": f"floor beside {prep['name']}"})
            self.used_categories[cat] += 1
            chosen += 1
            self.report[subtask].append(
                f"{pick}: target **{cat}/{rep['model']}** {self.slots_str(pick, 'target')} (pool {len(rows)}"
                f"{', filled' if any(r['status'].startswith('incomplete') for r in rows) else ''}) "
                f"+ partner **{prep['category']} {prep['name']}@{prep['scene']}** {self.slots_str(pick, 'partner')}; target: {'; '.join(notes) or 'ok'}; "
                f"partner: {'; '.join(pnotes) or 'ok'}{'; ' + '; '.join(snote) if snote else ''}")
        if chosen < n:
            self.report[subtask].append(f"only {chosen} of {n} picks ({len(tpool)} target categories after filling, {len(pcats)} partner categories)")

    def select_scene_artic(self, subtask: str, n: int) -> None:
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

    # -- output ---------------------------------------------------------------------------------------------------------
    def write(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "representatives.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REP_COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in REP_COLUMNS})
        cols = list(self.cands[0].keys()) + ["plan"]
        with open(os.path.join(out_dir, "candidates_extra.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            seen = set()
            for r in self.extra:
                key = (r["subtask"], r["role"], r["category"], r["model"])
                if key in seen:
                    continue
                seen.add(key)
                w.writerow({k: r.get(k, "") for k in cols})
        self.write_report(os.path.join(out_dir, "representatives_report.md"))

    def write_report(self, path: str) -> None:
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
        lines += ["## Held-out models (global)", ""]
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
                  "- **Scene articulated targets without a canonical handle annotation** (`incomplete:handle_from_B4_heuristic`, e.g. bottom_cabinet_fancyy_0, "
                  "bottom_cabinet_no_top_gjrero_*): the B4 geometric handle proposal is used through the `handle_file` role option; validate with the C3 review round "
                  "on `runlogs/scene_inventory_20260915/<scene>/artic/strips/` (verdicts -> `require/handles.csv`), then the representative test.",
                  "- **Supports** (`geometric_ok` only): admitted by top height, area, short side and a >= 0.60 m free-floor edge run at stand-off 0.45 (0.35 / 0.55 in the note); "
                  "the sampler's stance search is the validation (rejections `no_support` / `no_stance` -> customs blacklist per (scene, support)).",
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
    args = ap.parse_args(argv)
    sel = Selector(args.icl, args.per_family, args.min_per_subtask)
    sel.run()
    sel.write(args.out)
    n_picks = sum(1 for r in sel.rows if r["slot"] == "representative" and r["role"] in ("target", "targets", "obstacles"))
    print(f"{n_picks} picks, {len(sel.rows)} slot rows, {len(sel.heldout)} held-out categories, {len(sel.extra)} extra candidate rows -> {args.out}")


if __name__ == "__main__":
    main()
