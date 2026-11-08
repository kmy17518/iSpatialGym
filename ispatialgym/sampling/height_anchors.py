"""Exact support-height anchors of the scene-independent reach maps (owner prompt 2026-11-02, reach bands section 2).

A reach map is computed at an ACTUAL support-surface height above the floor, never at a bin centre or a rounded grid height:

1. surfaces: every support instance the role tables reference (``candidates.csv`` / ``representatives.csv`` support rows, repeated task
   rows deduplicated), its top from the scene inventory (``runlogs/scene_inventory_20260915/<scene>/supports/<name>.json``,
   ``top.z``: the median of the most populated 2 cm height bin of the support's hits, recorded to 1e-4 m) and its interior placement
   planes (shelves / boards: ``bindings.support_surfaces`` of the customs whose ``top_z`` is more than ``SURFACE_MATCH_TOL`` below the
   top - authored to 1e-2 m, status ``unmeasured_interior``: flagged for measurement, never replaced by the furniture's top);
2. floor datum: the nearest known floor level of the scene (0 and the partners' ``under_clearance.floor_z`` - only a floor the partner
   stands on, :data:`FLOOR_STAND_TOL_M`: i7-goalmaps fix) at or below the support's AABB bottom (+2 cm); heights are ``top - floor``;
3. canonical keys: the height at its recorded precision (``"%.4f"`` for the inventory) - distinct measured heights stay distinct, the
   same recorded height of two supports shares one key. Two measurements of one surface that disagree beyond their precision are
   listed in ``conflicts``;
4. ranges: anchors sorted; each owns the interval between the midpoints to its neighbours (a tie at a midpoint goes to the LOWER
   anchor), clipped to ``max_approx_m`` (default 0.01 m) around it. A query outside every range is uncovered (exact-IK fallback or an
   explicitly requested map) - never a distant anchor. The nominal 2 cm bins only organise the inventory (``bins``).

Goal surfaces (i7-goalmaps 2026-11-08, owner decision: prompt_goalmaps_axialrest_overpath.md section 1): the reach maps must cover every
goal surface an IORS large predicate needs. :data:`GOAL_PLANES` names, per subtask, where on the partner its goal side reads a map; for a
``"top"`` goal (ontop: a tabletop map at the partner's top) the ``partner`` rows of both role tables (``representatives.csv`` names the
instance in ``name``) are surfaces too, measured by the same rules (the inventory top - ``supports/<name>.json``, else
``partners/<name>.json`` -, the floor datum, the canonical key, the window), recorded in ``Surface.goal_subtasks``; :func:`model_surfaces`
returns them with the supports. A goal row on a surface a support row already measured adds no anchor (its key stays); the support
surfaces, their keys and their ``subtasks`` are unchanged (``goal_surfaces=False`` / ``--no-goal-surfaces``: the former set). Interior
goals (inside, touching on a board) read the registry spec's map (``sampling.interiors``: no anchor axis); push-under, the floor push and
the base push read none (exact IK).

Workload (reported before generation): ``N_heights(model)`` = the distinct anchors of the surfaces the model's task rows can use, + 1
when the model also needs a floor-mode map.

    python -m ispatialgym.sampling.height_anchors --candidates ispatialgym/assets/icl/roles/candidates.csv --out anchors.json [--model C/M]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ispatialgym import REPO_ROOT

SCHEMA = "isg-height-anchors/1"
INVENTORY = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
CUSTOMS = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")
SURFACE_MATCH_TOL = 0.03  # reach_bands.SURFACE_MATCH_TOL: a binding within 3 cm of the top IS the top
BIN_M = 0.02
MAX_APPROX_M = 0.01
NON_SUPPORT_CATEGORIES = ("walls", "floors")
#: i7-goalmaps 2026-11-08 (owner decision): the plane of the partner each IORS large predicate's goal side reads a reach map at - "top" (a
#: tabletop map at the partner's measured top: its partner rows are surfaces of the target model), "interior" (an interior map of the
#: registry spec, ``sampling.interiors``: no anchor axis), None (no goal map: push-under, floor push, base push - exact IK). The variant
#: ``touching_place: "surface"`` (touching_large_02) reads an interior map; it is a customs choice, not the subtask's.
GOAL_PLANES: Dict[str, Optional[str]] = {
    "inter_object_relation_satisfaction-ontop_large_scale": "top",
    "inter_object_relation_satisfaction-inside_large_scale": "interior",
    "inter_object_relation_satisfaction-under_large_scale": None,
    "inter_object_relation_satisfaction-nextto_large_scale": None,
    "inter_object_relation_satisfaction-touching_large_scale": None,
}


def key(h: float, precision: float = 1e-4) -> str:
    """Canonical serialization at the recorded precision (no float artifacts, no merging of distinct measured heights)."""
    digits = max(0, int(round(-math.log10(precision))))
    return f"{round(float(h), digits):.{digits}f}"


@dataclass
class Surface:
    scene: str
    support: str
    surface: str  # "top" or "interior@<authored z>"
    category: str
    height: float  # above the floor datum (m)
    precision: float
    source: str
    floor_z: float
    floor_source: str
    status: str = "measured"
    rows: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    goal_subtasks: List[str] = field(default_factory=list)  # i7-goalmaps: the subtasks whose GOAL surface this is (partner rows)

    @property
    def id(self) -> str:
        return f"{self.scene}/{self.support}/{self.surface}"

    @property
    def key(self) -> str:
        return key(self.height, self.precision)


#: i7-goalmaps 2026-11-08 (owner decision 5c.1, the floor datum bug): a partner's ``under_clearance.floor_z`` is a floor level only when the
#: partner stands on it - within this of the partner's AABB bottom. The raycast under a counter can stop on the cabinet inside it
#: (countertop_kelzer_0: 0.4977 m, its AABB bottom 0.8514 m), which measured 15 house_single_floor tops from a cabinet top.
FLOOR_STAND_TOL_M = 0.05


def _floor_levels(scene: str, inventory: str, real_floors: bool = True) -> List[float]:
    """The scene's floor levels: 0 and the partners' ``under_clearance.floor_z`` - with ``real_floors`` (default; the bug fix) only those
    the partner stands on (:data:`FLOOR_STAND_TOL_M` of its AABB bottom); ``real_floors=False``: every one (the former rule)."""
    lv = {0.0}
    for f in glob.glob(os.path.join(inventory, scene, "partners", "*.json")):
        try:
            d = json.load(open(f))
            z = (d.get("under_clearance") or {}).get("floor_z")
            bottom = float(d["aabb_world"]["min"][2]) if real_floors else None
        except Exception:  # noqa: BLE001
            z, bottom = None, None
        if z is None:
            continue
        if real_floors and (bottom is None or abs(float(z) - bottom) > FLOOR_STAND_TOL_M):
            continue
        lv.add(round(float(z), 4))
    return sorted(lv)


def _floor_datum(levels: Sequence[float], aabb_min_z: float) -> Tuple[float, str]:
    below = [z for z in levels if z <= aabb_min_z + 0.02]
    if not below:
        return 0.0, "default_0"
    z = max(below)
    return z, ("scene_floor_0" if z == 0.0 else "partner_floor_z")


def _support_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    out = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("role") == "support" and r.get("category") not in NON_SUPPORT_CATEGORIES:
                    r = dict(r)
                    r["_table"] = os.path.basename(p)
                    out.append(r)
    return out


def _instance(r: Dict[str, str]) -> str:
    """The scene instance a role row names: ``representatives.csv``'s ``name`` (its ``model`` is the model id), else ``model``."""
    return (r.get("name") or "").strip() or (r.get("model") or "").strip()


def _goal_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    """i7-goalmaps: the ``partner`` rows of the subtasks whose goal is the partner's top (:data:`GOAL_PLANES` "top"), both tables' schemas."""
    out = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("role") != "partner" or GOAL_PLANES.get(r.get("subtask") or "") != "top" or r.get("category") in NON_SUPPORT_CATEGORIES:
                    continue
                if not r.get("scene") or not _instance(r):
                    continue
                r = dict(r)
                r["_table"] = os.path.basename(p)
                r["_inst"] = _instance(r)
                out.append(r)
    return out


def _customs_surfaces(path: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """(scene, support) -> the ``bindings.support_surfaces`` entries of every customs entry (with the entry key)."""
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if not os.path.exists(path):
        return out
    d = json.load(open(path))

    def walk(o, path_):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "support_surfaces" and isinstance(v, list):
                    scene = next((path_[i + 1] for i in range(len(path_) - 1) if path_[i] == "scenes"), None)
                    for e in v:
                        if isinstance(e, dict) and e.get("name") and e.get("top_z") is not None:
                            out.setdefault((scene, e["name"]), []).append({**e, "_entry": path_[0] if path_ else None})
                else:
                    walk(v, path_ + [k])
        elif isinstance(o, list):
            for v in o:
                walk(v, path_)

    walk(d, [])
    return out


def build_surfaces(tables: Sequence[str], inventory: str = INVENTORY, customs: str = CUSTOMS,
                   goal_surfaces: bool = True, real_floors: bool = True) -> Tuple[List[Surface], Dict[str, Any]]:
    """The measured surfaces of the support rows (+ with ``goal_surfaces`` the goal-surface partner rows, :func:`_goal_rows`, after them:
    a goal row never changes a support surface's height, key or ``subtasks``) and the report."""
    rows = _support_rows(tables)
    binds = _customs_surfaces(customs)
    surfaces: Dict[str, Surface] = {}
    conflicts: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    levels_cache: Dict[str, List[float]] = {}
    for r in rows:
        scene, name = r["scene"], r["model"]
        inv = os.path.join(inventory, scene, "supports", f"{name}.json")
        if not os.path.exists(inv):
            missing.append({"scene": scene, "support": name, "row": r.get("subtask"), "table": r["_table"]})
            continue
        s = json.load(open(inv))
        if scene not in levels_cache:
            levels_cache[scene] = _floor_levels(scene, inventory, real_floors)
        fz, fsrc = _floor_datum(levels_cache[scene], float(s["aabb_world"]["min"][2]))
        top = float(s["top"]["z"])
        sid = f"{scene}/{name}/top"
        if sid not in surfaces:
            surfaces[sid] = Surface(scene, name, "top", s.get("category", r.get("category", "")), round(top - fz, 4), 1e-4, "inventory_top.z", fz, fsrc)
        surf = surfaces[sid]
        surf.rows.append(f"{r['_table']}:{r.get('subtask')}#{r.get('rank') or r.get('pick') or ''}")
        if r.get("subtask") and r["subtask"] not in surf.subtasks:
            surf.subtasks.append(r["subtask"])
        # the role table's own top_z (3 decimals): a second measurement of the same surface
        try:
            tz = float(r.get("top_z") or "nan")
        except ValueError:
            tz = float("nan")
        if math.isfinite(tz) and abs((tz - fz) - surf.height) > 0.0015 and abs(tz - top) <= SURFACE_MATCH_TOL:
            conflicts.append({"surface": sid, "inventory_m": surf.height, "role_table_m": round(tz - fz, 4), "row": r.get("subtask")})
        for b in binds.get((scene, name), []):
            bz = float(b["top_z"])
            if abs(bz - surf.height) <= SURFACE_MATCH_TOL:
                continue  # the binding names the top
            iid = f"{scene}/{name}/interior@{bz:.2f}"
            if iid not in surfaces:
                surfaces[iid] = Surface(scene, name, f"interior@{bz:.2f}", surf.category, round(bz, 2), 1e-2, "customs.bindings.support_surfaces.top_z",
                                        fz, fsrc, status="unmeasured_interior")
            if b.get("_entry") and b["_entry"] not in surfaces[iid].subtasks:
                surfaces[iid].subtasks.append(b["_entry"])
    n_support_surfaces = len(surfaces)
    goal_rows = _goal_rows(tables) if goal_surfaces else []
    goal_missing: List[Dict[str, Any]] = []
    new_goal: List[str] = []
    for r in goal_rows:
        scene, name = r["scene"], r["_inst"]
        inv = next((p for p in (os.path.join(inventory, scene, d, f"{name}.json") for d in ("supports", "partners")) if os.path.exists(p)), None)
        if inv is None:
            goal_missing.append({"scene": scene, "partner": name, "row": r.get("subtask"), "table": r["_table"]})
            continue
        s = json.load(open(inv))
        if not s.get("top") or s["top"].get("z") is None:
            goal_missing.append({"scene": scene, "partner": name, "row": r.get("subtask"), "table": r["_table"], "why": "no top in the inventory"})
            continue
        if scene not in levels_cache:
            levels_cache[scene] = _floor_levels(scene, inventory, real_floors)
        fz, fsrc = _floor_datum(levels_cache[scene], float(s["aabb_world"]["min"][2]))
        top = float(s["top"]["z"])
        sid = f"{scene}/{name}/top"
        if sid not in surfaces:
            surfaces[sid] = Surface(scene, name, "top", s.get("category", r.get("category", "")), round(top - fz, 4), 1e-4,
                                    f"inventory_top.z ({os.path.basename(os.path.dirname(inv))})", fz, fsrc)
            new_goal.append(sid)
        surf = surfaces[sid]
        surf.rows.append(f"{r['_table']}:{r.get('subtask')}#{r.get('pick') or r.get('rank') or ''}:goal")
        if r["subtask"] not in surf.goal_subtasks:
            surf.goal_subtasks.append(r["subtask"])
        try:
            tz = float(r.get("top_z") or "nan")
        except ValueError:
            tz = float("nan")
        if math.isfinite(tz) and abs((tz - fz) - surf.height) > 0.0015 and abs(tz - top) <= SURFACE_MATCH_TOL:
            conflicts.append({"surface": sid, "inventory_m": surf.height, "role_table_m": round(tz - fz, 4), "row": r.get("subtask")})
    report = {"n_rows": len(rows), "n_surfaces": len(surfaces), "conflicts": conflicts, "missing_inventory": missing,
              "floor_levels": levels_cache, "n_support_surfaces": n_support_surfaces, "n_goal_rows": len(goal_rows),
              "goal_surfaces_new": new_goal, "goal_missing_inventory": goal_missing}
    return sorted(surfaces.values(), key=lambda s: (s.height, s.id)), report


@dataclass
class AnchorSet:
    anchors: List[Dict[str, Any]]  # sorted by height: {"key", "height", "precision", "surfaces", "range": [lo, hi]}
    max_approx_m: float = MAX_APPROX_M

    @staticmethod
    def from_surfaces(surfaces: Sequence[Surface], max_approx_m: float = MAX_APPROX_M) -> "AnchorSet":
        by: Dict[str, Dict[str, Any]] = {}
        for s in surfaces:
            a = by.setdefault(s.key, {"key": s.key, "height": float(s.key), "precision": s.precision, "surfaces": [], "status": set()})
            a["surfaces"].append(s.id)
            a["status"].add(s.status)
        anchors = sorted(by.values(), key=lambda a: a["height"])
        for i, a in enumerate(anchors):
            lo = (anchors[i - 1]["height"] + a["height"]) / 2 if i > 0 else -math.inf
            hi = (a["height"] + anchors[i + 1]["height"]) / 2 if i + 1 < len(anchors) else math.inf
            a["range"] = [max(lo, a["height"] - max_approx_m), min(hi, a["height"] + max_approx_m)]
            a["status"] = sorted(a["status"])
        return AnchorSet(anchors, max_approx_m)

    def heights(self) -> List[float]:
        return [a["height"] for a in self.anchors]

    def lookup(self, h: float) -> Tuple[Optional[Dict[str, Any]], str]:
        """``(anchor, kind)``: kind ``exact`` (the query is an anchor at its precision), ``approx`` (inside an anchor's clipped range),
        ``uncovered`` (anchor None). Anchor j owns (m[j-1], m[j]] between the midpoints to its neighbours - a query exactly on a shared
        midpoint belongs to the LOWER anchor - clipped to ``max_approx_m`` around it; every range holds its own anchor.

        Several exact matches (a coarse anchor - an interior board authored at 1e-2 m - and a measured top at 1e-4 m): the FINEST
        precision wins, then the nearest height (of-v2maps 2026-11-06: the anchor list is sorted by height, so the authored 0.81 matched
        the breakfast table's measured 0.8126 first and every map lookup at that table missed)."""
        exact = [a for a in self.anchors if key(h, a["precision"]) == a["key"]]
        if exact:
            return min(exact, key=lambda a: (float(a["precision"]), abs(float(h) - float(a["height"])))), "exact"
        hs = self.heights()
        if not hs:
            return None, "uncovered"
        mids = [(hs[i] + hs[i + 1]) / 2 for i in range(len(hs) - 1)]
        j = bisect.bisect_left(mids, h)  # the first midpoint >= h: h <= mids[j] -> anchor j (ties to the lower anchor)
        a = self.anchors[j]
        if abs(h - a["height"]) <= self.max_approx_m + 1e-12:
            return a, "approx"
        return None, "uncovered"

    def bins(self, bin_m: float = BIN_M) -> Dict[str, List[str]]:
        """Nominal 2 cm ranges (organisation only): bin start -> the anchor keys in it."""
        out: Dict[str, List[str]] = {}
        for a in self.anchors:
            b = math.floor(a["height"] / bin_m + 1e-9) * bin_m
            out.setdefault(f"{b:.2f}", []).append(a["key"])
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "max_approx_m": self.max_approx_m, "anchors": self.anchors}


def model_surfaces(table, category: str, model: str, surfaces: Sequence[Surface]) -> List[Surface]:
    """The surfaces the model's target rows can use: the supports of every subtask that lists it as a target, and (i7-goalmaps) the goal
    surfaces of those subtasks (``Surface.goal_subtasks``). ``table``: one role table or several."""
    subs = set()
    for t in ([table] if isinstance(table, str) else list(table)):
        with open(t) as f:
            for r in csv.DictReader(f):
                if r.get("role") == "target" and r.get("category") == category and r.get("model") == model:
                    subs.add(r.get("subtask"))
    return [s for s in surfaces if (set(s.subtasks) | set(s.goal_subtasks)) & subs]


def workload(table: str, models: Iterable[Tuple[str, str]], surfaces: Sequence[Surface], floor_models: Iterable[Tuple[str, str]] = ()) -> Dict[str, Any]:
    floor_models = set(floor_models)
    out = {}
    for c, m in models:
        ss = model_surfaces(table, c, m, surfaces)
        keys = sorted({s.key for s in ss}, key=float)
        out[f"{c}/{m}"] = {"N_heights": len(keys) + (1 if (c, m) in floor_models else 0), "table_heights": keys, "floor": (c, m) in floor_models,
                           "surfaces": [s.id for s in ss], "goal_surfaces": [s.id for s in ss if s.goal_subtasks]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=os.path.join(REPO_ROOT, "ispatialgym", "assets", "icl", "roles", "candidates.csv"))
    ap.add_argument("--representatives", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--max-approx", type=float, default=MAX_APPROX_M)
    ap.add_argument("--no-goal-surfaces", action="store_true", help="i7-goalmaps: leave the goal-surface partner rows out (the former set)")
    a = ap.parse_args(argv)
    tables = [a.candidates] + ([a.representatives] if a.representatives else [])
    surfaces, rep = build_surfaces(tables, goal_surfaces=not a.no_goal_surfaces)
    aset = AnchorSet.from_surfaces(surfaces, a.max_approx)
    tops = [s for s in surfaces if s.surface == "top"]
    print(f"{rep['n_rows']} support rows -> {len(surfaces)} surfaces ({len(tops)} tops, {len(surfaces) - len(tops)} interior), "
          f"{len(aset.anchors)} exact anchors ({len({s.key for s in tops})} from tops); conflicts {len(rep['conflicts'])}, missing inventory {len(rep['missing_inventory'])}")
    print("nominal 2 cm bins:", len(aset.bins()), "occupied; bins holding several distinct anchors:", sum(1 for v in aset.bins().values() if len(v) > 1))
    if not a.no_goal_surfaces:
        print(f"goal surfaces: {rep['n_goal_rows']} partner rows, {len(rep['goal_surfaces_new'])} new surfaces, "
              f"{len(rep['goal_missing_inventory'])} without an inventory top")
    if a.model:
        wl = workload(tables, [tuple(m.split("/")) for m in a.model], surfaces)
        for k, v in wl.items():
            print(k, "N_heights", v["N_heights"], v["table_heights"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump({**aset.to_json(), "surfaces": [s.__dict__ | {"id": s.id, "key": s.key} for s in surfaces], "report": rep}, f, indent=1)


if __name__ == "__main__":
    main()
