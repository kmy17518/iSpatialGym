"""F from the map's content (of-v2maps 2026-11-07, owner prompt I item 4; the isg1020 spec Prompt 2 §6 / §8).

The tabletop sampler's Phase 3 screens 8 fixed yaws x every library grasp at the focus by batched IK and draws (yaw, grasp) uniformly from
the survivors. A map view (``map_sampling.as_bands``) already holds, per (edge distance, trunk, cell), the feasible (yaw, grasp) pairs of
the map's grasp axis - the validated grasps, their twins and their family members - so with ``ISPATIALGYM_MAP_F``:

* ``diag``: behaviour unchanged; the map's pairs at the focus cell are compared with the screened F (counters ``map_F.diag.*``: pairs
  both hold, screen-only, map-only, on (snapped yaw, base grasp)).
* ``seed``: F is the map's pairs at the focus cell (the map cell nearest the focus in the stance's base frame, at the stance's trunk and
  the nearest map edge distance) - no batched screen; every pair is still confirmed by the exact IK of the draw at the real height, stance
  and scene. Yaws are the map's yaws in the world (the stance's base yaw + the map yaw: continuous, not the sampler's 8 world yaws);
  a family member (``param`` != 0) becomes a grasp of its own (:func:`grasp_library.family_member`: its ``T_object_eef``, the
  family id and a CONTINUOUS parameter drawn within half the member spacing; ``ISPATIALGYM_MAP_F_FAMILY=0``: representatives only); a
  twin row is merged onto its row (the draw's twin choice tries both turns). Pairs are deduplicated and drawn in a weighted order -
  each (row, twin) family's measure counts once whatever its sample count (``ReachMap.cell_weights``' normalisation) - and the draw's
  yaw jitter / focus radius are scaled per map cell by :class:`map_sampling.AdaptiveWidth` (halved after a focus whose F ran dry
  without an exact confirmation, recovered by 1.25 after a confirmation). A focus outside the map's lattice keeps the screen.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_maps as RM

_RM: Dict[str, RM.ReachMap] = {}


def mode() -> str:
    v = os.environ.get("ISPATIALGYM_MAP_F", "").strip().lower()
    return v if v in ("diag", "seed") else ""


def family_on() -> bool:
    return os.environ.get("ISPATIALGYM_MAP_F_FAMILY", "1") != "0"


def is_view(b) -> bool:
    return isinstance(getattr(b, "meta", None), dict) and b.meta.get("mode") == "map_view" and bool(b.meta.get("map_path"))


def reach_map_of(view) -> RM.ReachMap:
    p = str(view.meta["map_path"])
    if p not in _RM:
        _RM[p] = RM.ReachMap(p)
    return _RM[p]


@dataclass
class MapCell:
    rm: RM.ReachMap
    edge_distance: float
    trunk: Tuple[float, float]
    index: int
    offset_m: float
    base_yaw: float

    @property
    def key(self) -> Tuple[str, float, float, float, int]:
        return (str(self.rm.meta.get("key")), self.edge_distance, self.trunk[0], self.trunk[1], self.index)


def focus_cell(view, stance, fx) -> Optional[MapCell]:
    """The map cell of the focus ``fx`` (world xy) at ``stance``: the nearest lattice cell in the stance's base frame within half a lattice
    step (else None), the nearest map edge distance to the stance's stand-off + the view's overhang, the stance's trunk (the map trunk with
    the same translate and the nearest tilt; None when no trunk shares the translate)."""
    rm = reach_map_of(view)
    so = stance.stand_off if getattr(stance, "stand_off", None) is not None else view.meta.get("stand_off")
    if so is None:
        return None
    d_e = min((float(d) for d in rm.meta["edge_distances"]), key=lambda d: abs(d - (float(so) + float(view.meta.get("overhang") or 0.0))))
    trunks = [(float(t["translate"]), float(t["tilt"])) for t in rm.meta["trunks"] if t.get("eligible")]
    same = [t for t in trunks if abs(t[0] - float(stance.translate)) < 1e-6]
    if not same:
        return None
    tr = min(same, key=lambda t: abs(t[1] - float(stance.tilt)))
    rb = view.to_base(np.asarray(fx, dtype=float).reshape(1, 2), stance.base_pose2d)[0]
    lat = float(rm.meta["config"]["lattice_m"])
    d = np.abs(rm.cells - rb[None, :]).max(axis=1)
    ci = int(np.argmin(d))
    if float(d[ci]) > 0.5 * lat + 1e-6:
        return None
    return MapCell(rm, d_e, tr, ci, float(np.linalg.norm(rm.cells[ci] - rb)), float(stance.base_pose2d[2]))


def cell_pairs(mc: MapCell) -> np.ndarray:
    st = mc.rm.status(mc.edge_distance, *mc.trunk)[mc.index] == RM.STATUS["feasible"]
    return np.argwhere(st)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def F_from_map(mc: MapCell, grasps: Sequence[Any], rng: np.random.Generator, width: float = 1.0
               ) -> Tuple[List[Tuple[float, Any]], Dict[Tuple[float, str], float], Dict[str, Any]]:
    """``(F, weights, facts)`` at the map cell: F = [(world yaw, grasp)] (module docstring), weights by ``(round(yaw, 4), grasp name)``."""
    from ispatialgym.sampling import grasp_library as GL

    rm = mc.rm
    rows = rm.meta["grasps"]
    fam_n: Dict[Tuple[str, bool], int] = {}
    for r in rows:
        fam_n[(r["base"], bool(r["twin"]))] = fam_n.get((r["base"], bool(r["twin"])), 0) + 1
    steps: Dict[str, float] = {}
    for r in rows:
        if r.get("param") is not None and r.get("family_id"):
            ps = sorted({float(x["param"]) for x in rows if x["base"] == r["base"] and x.get("param") is not None})
            steps[r["base"]] = float(np.min(np.diff(ps))) if len(ps) > 1 else 0.0
    by_name = {}
    for g in grasps:
        by_name.setdefault(str(getattr(g, "name", "")).split("~")[0], g)
    lib = GL.v2_entry(getattr(grasps[0], "key", "")) if grasps else None
    fams = lib.families() if lib is not None else {}
    F: Dict[Tuple[float, str], Tuple[float, Any]] = {}
    W: Dict[Tuple[float, str], float] = {}
    n_rows = n_missing = n_members = 0
    for yi, gi in cell_pairs(mc):
        r = rows[int(gi)]
        n_rows += 1
        g0 = by_name.get(r["base"])
        if g0 is None:
            n_missing += 1  # a row the sampler dropped (not attachable, an inclined tier it does not offer)
            continue
        yaw = _wrap(mc.base_yaw + float(rm.yaws[int(yi)]))
        p = r.get("param")
        g = g0
        if family_on() and p is not None and abs(float(p)) > 1e-9 and r.get("family_id") in fams:
            step = steps.get(r["base"], 0.0)
            pc = float(p) + (float(rng.uniform(-0.5, 0.5)) * step * width if step > 0 else 0.0)
            g = GL.family_member(g0, fams[r["family_id"]], pc)
            n_members += 1
        elif p is not None and abs(float(p)) > 1e-9:
            continue  # members off: the representative's own row covers it
        k = (round(yaw, 4), str(g.name))
        if k not in F:
            F[k] = (yaw, g)
        W[k] = W.get(k, 0.0) + 1.0 / fam_n[(r["base"], bool(r["twin"]))]
    facts = {"map_rows": n_rows, "sampler_missing": n_missing, "members": n_members, "pairs": len(F), "offset_m": round(mc.offset_m, 4)}
    return list(F.values()), W, facts


def draw_order(rng: np.random.Generator, F: Sequence[Tuple[float, Any, Any]], weights: Optional[Dict[Tuple[float, str], float]]) -> np.ndarray:
    """The draw order of F: ``rng.permutation`` (as before) unless every pair has a map weight - then a weighted permutation without
    replacement (Efraimidis-Spirakis keys u^(1/w))."""
    if not weights:
        return rng.permutation(len(F))
    w = [weights.get((round(float(f[0]), 4), str(getattr(f[1], "name", "")))) for f in F]
    if any(v is None or v <= 0 for v in w):
        return rng.permutation(len(F))
    u = rng.random(len(F))
    return np.argsort(-(np.log(u) / np.asarray(w, dtype=float)))


def width_of(s) -> MS.AdaptiveWidth:
    w = getattr(s, "_map_F_width", None)
    if w is None:
        w = s._map_F_width = MS.AdaptiveWidth()
    return w


def diag_compare(s, count, mc: MapCell, F_screen: Sequence[Tuple[float, Any]], yaws: Sequence[float]) -> None:
    """``diag``: the screened F vs the map's pairs at the cell, on (the sampler yaw nearest the map yaw, base grasp)."""
    rows = mc.rm.meta["grasps"]

    def snap(y: float) -> int:
        return int(min(range(len(yaws)), key=lambda i: abs(_wrap(float(y) - float(yaws[i])))))

    M = {(snap(_wrap(mc.base_yaw + float(mc.rm.yaws[int(yi)]))), rows[int(gi)]["base"]) for yi, gi in cell_pairs(mc)}
    S = {(snap(y), str(getattr(g, "name", "")).split("~")[0].split("@")[0]) for y, g in F_screen}
    count(s, "map_F.diag.foci")
    count(s, "map_F.diag.screen_pairs", len(S))
    count(s, "map_F.diag.map_pairs", len(M))
    count(s, "map_F.diag.both", len(S & M))
    count(s, "map_F.diag.screen_only", len(S - M))
    count(s, "map_F.diag.map_only", len(M - S))


# ------------------------------------------------------------------------------------------------ release ranking (prompt I item 6)


def release_on() -> bool:
    """``ISPATIALGYM_MAP_RELEASE=1``: the place draw ranks its valid partner poses by :func:`release_count` (map entries matching the
    place eef pose) - an ordering only: an empty answer means "no compatible map entry", never infeasible, and every place keeps its
    exact checks (the support policy's scope unchanged)."""
    return os.environ.get("ISPATIALGYM_MAP_RELEASE", "") == "1"


def release_count(view, stance, place_eef_world, grasp_name: str, floor_z: float, place_support_z: float) -> Optional[int]:
    """How many feasible map entries (``map_sampling.release_candidates``: same grasp row, the map's own support height, 4 cm / 0.35 rad)
    lie at the place eef pose in the stance's base frame; None when the view is no map, the place support is not at the map's height
    (+-1 cm) or the stance's trunk is not in the map."""
    if not is_view(view):
        return None
    rm = reach_map_of(view)
    h = float(rm.meta["height"]["value"])
    if abs((float(place_support_z) - float(floor_z)) - h) > 0.01:
        return None
    so = stance.stand_off if getattr(stance, "stand_off", None) is not None else view.meta.get("stand_off")
    if so is None:
        return None
    d_e = min((float(d) for d in rm.meta["edge_distances"]), key=lambda d: abs(d - float(so)))
    same = [(float(t["translate"]), float(t["tilt"])) for t in rm.meta["trunks"] if t.get("eligible") and abs(float(t["translate"]) - float(stance.translate)) < 1e-6]
    if not same:
        return None
    tr = min(same, key=lambda t: abs(t[1] - float(stance.tilt)))
    x0, y0, yaw = (float(v) for v in stance.base_pose2d)
    T_wb = np.eye(4)
    T_wb[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
    T_wb[:3, 3] = [x0, y0, float(floor_z)]
    T_q = np.linalg.inv(T_wb) @ np.asarray(place_eef_world, dtype=float)
    name = str(grasp_name).split("@")[0]
    return len(MS.release_candidates(rm, d_e, tr[0], tr[1], T_q, name))
