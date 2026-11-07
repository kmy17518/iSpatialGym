"""Sampling from scene-independent reach maps (:mod:`reach_maps`): the sampler-facing view and the weighted, continuous proposals.

* :func:`as_bands` - a map at a real support (its actual top height) as a :class:`reach_bands.ReachBands`-shaped view (multi-distance:
  one stand-off per map edge distance, ``stand_off = d_e - overhang``; the sampler passes ``overhang = 0`` - its stance stands
  ``stand_off + overhang`` outside the raster edge, i.e. ``stand_off`` from the physical edge the map's ``d_e`` measures; of-v2maps
  2026-11-07, prompt I item 8) so the tabletop sampler's stance search
  (``band_cells_world``, ``workspace``, ``phase2_manipulation_stance``) runs on it unchanged. The view is tagged
  ``meta["mode"] = "map_view"`` and has no file path (band refinement and job-local band copies never touch it); a legacy band is
  never built from a map and a map is never read as a band.
* :func:`proposals` - after the sampler fixed a real stance (support edge, base pose, trunk): cells drawn with probability
  ~ ``w^temperature`` of the map's plausibility (``ReachMap.cell_weights``: feasible (yaw, grasp) measure, normalised for the
  discretisation of yaw and family axes) mixed with ``uniform_mix`` of the feasible cells for diversity; a feasible (yaw, grasp) of the
  cell, then CONTINUOUS values - the cell centre jittered within half a lattice step, the yaw within half a yaw step, a family member's
  parameter within half its sample spacing - each proposal carrying its spacing and the feasible fraction of its neighbours
  (uncertainty: isolated successes do not make a guaranteed interval). Every proposal must be verified by exact IK at the actual
  height, stance and scene geometry; :class:`AdaptiveWidth` shrinks the jitter where verifications fail.

Switch: ``ISPATIALGYM_REACH_MAPS=<maps root>`` (+ a v2 grasp library, ``ISPATIALGYM_GRASP_LIBRARY_V2``) makes
``tabletop.ensure_bands`` prefer a map; without a compatible map at an anchor covering the support height the sampler keeps its
legacy path (``ISPATIALGYM_REACH_MAPS_STRICT=1``: reject ``missing_map`` instead).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.sampling import reach_maps as RM

ENV_ROOT = "ISPATIALGYM_REACH_MAPS"
ENV_STRICT = "ISPATIALGYM_REACH_MAPS_STRICT"
ENV_ANCHORS = "ISPATIALGYM_REACH_MAP_ANCHORS"


def maps_root() -> Optional[str]:
    return os.environ.get(ENV_ROOT) or None


def as_bands(rm: RM.ReachMap, overhang: float = 0.0):
    """The map as a multi-distance ``ReachBands`` view (cells / pairs per trunk per stand-off) at a support with ``overhang``."""
    from ispatialgym.sampling import reach_bands as RB

    # i7-generic 2026-11-07: a generic map (reach_maps.compute_generic_map) reads like the legacy --generic-pinch band - no grasp names,
    # pairs [cell, -1, class] (ReachBands.feasible -> ("generic", class)), the class names in meta["classes"] and per cell the bitmask of
    # the classes feasible there (bit k = class k; ReachBands.classes_at / tabletop.cell_classes / joint_placement read it unchanged)
    generic = rm.meta.get("kind") == RM.GENERIC_KIND
    meta = {"mode": "map_view", "map_key": rm.meta["key"], "map_path": rm.path, "lattice": float(rm.meta["config"]["lattice_m"]),
            "arm": rm.meta["arm"], "grasp_names": [] if generic else [g["name"] for g in rm.meta["grasps"]], "trunks": [],
            "stand_offs": [], "stand_off": None, "overhang": float(overhang), "height": rm.meta["height"]}
    if generic:
        meta.update({"map_kind": RM.GENERIC_KIND, "generic_pinch": True, "classes": [g["name"] for g in rm.meta["grasps"]],
                     "pose_set": rm.meta.get("pose_set")})
    cells_so: Dict[float, Dict[Tuple[float, float], np.ndarray]] = {}
    pairs_so: Dict[float, Dict[Tuple[float, float], np.ndarray]] = {}
    classes_so: Dict[float, Dict[Tuple[float, float], np.ndarray]] = {}
    for d_e in rm.meta["edge_distances"]:
        so = round(float(d_e) - float(overhang), 4)
        meta["stand_offs"].append(so)
        cells_so[so], pairs_so[so] = {}, {}
        for t in rm.meta["trunks"]:
            if not t.get("eligible"):
                continue
            key = (float(t["translate"]), float(t["tilt"]))
            st = rm.status(d_e, *key)
            ok = st == RM.STATUS["feasible"]
            ci = np.nonzero(ok.any(axis=(1, 2)))[0]
            cells_so[so][key] = rm.cells[ci].astype(np.float64)
            remap = {int(c): i for i, c in enumerate(ci)}
            p = np.argwhere(ok)
            if generic:
                pairs_so[so][key] = np.array([[remap[int(c)], -1, int(g)] for c, _, g in p], dtype=np.int32).reshape(-1, 3)
                bits = (ok.any(axis=1).astype(np.int64) << np.arange(ok.shape[2], dtype=np.int64)[None, :]).sum(axis=1)
                classes_so.setdefault(so, {})[key] = bits[ci]
            else:
                pairs_so[so][key] = np.array([[remap[int(c)], int(y), int(g)] for c, y, g in p], dtype=np.int32).reshape(-1, 3)
            meta["trunks"].append({"translate": key[0], "tilt": key[1], "stand_off": so, "posture": t.get("posture")})
    ref = meta["stand_offs"][len(meta["stand_offs"]) // 2]
    meta["stand_off"] = ref
    return RB.ReachBands(meta, dict(cells_so[ref]), dict(pairs_so[ref]), None, dict(classes_so.get(ref, {})), cells_so, pairs_so, classes_so)


@dataclass
class Proposal:
    cell_base: Tuple[float, float]
    yaw_base: float
    grasp: Dict[str, Any]
    cell_index: int
    yaw_index: int
    grasp_index: int
    spacing: Dict[str, float]
    neighbour_feasible_fraction: float
    weight: float


@dataclass
class AdaptiveWidth:
    """Per-cell jitter scale: halves after a failed exact verification, recovers by 1.25 after a success (bounded [0.125, 1])."""

    scale: Dict[int, float] = field(default_factory=dict)

    def of(self, cell: int) -> float:
        return self.scale.get(cell, 1.0)

    def update(self, cell: int, ok: bool) -> None:
        s = self.of(cell)
        self.scale[cell] = min(1.0, s * 1.25) if ok else max(0.125, s * 0.5)


def proposals(rm: RM.ReachMap, edge_distance: float, translate: float, tilt: float, n: int, rng: np.random.Generator,
              temperature: float = 1.0, uniform_mix: float = 0.25, width: Optional[AdaptiveWidth] = None,
              allowed_cells: Optional[np.ndarray] = None) -> List[Proposal]:
    """``n`` continuous proposals at a fixed stance (module docstring). ``allowed_cells`` (bool per map cell): the cells the real scene
    admits (support fit, clutter) - weights outside are zero."""
    st = rm.status(edge_distance, translate, tilt) == RM.STATUS["feasible"]
    w = rm.cell_weights(edge_distance, translate, tilt).astype(float)
    if allowed_cells is not None:
        w = w * np.asarray(allowed_cells, dtype=float)
    feas = np.nonzero(w > 0)[0]
    if len(feas) == 0:
        return []
    p = w[feas] ** float(temperature)
    p = p / p.sum()
    p = (1 - uniform_mix) * p + uniform_mix / len(feas)
    lat = float(rm.meta["config"]["lattice_m"])
    nY = len(rm.yaws)
    dyaw = 2 * math.pi / nY
    grasps = rm.meta["grasps"]
    fam_step: Dict[str, float] = {}
    for g in grasps:
        if g.get("param") is not None and g.get("family_id"):
            fam_step.setdefault(g["base"], 0.0)
    for base in fam_step:
        ps = sorted({float(g["param"]) for g in grasps if g["base"] == base})
        fam_step[base] = float(np.min(np.diff(ps))) if len(ps) > 1 else 0.0
    cell_lookup = {tuple(np.round(c, 4)): i for i, c in enumerate(rm.cells)}
    out: List[Proposal] = []
    width = width or AdaptiveWidth()
    for _ in range(n):
        ci = int(rng.choice(feas, p=p))
        pairs = np.argwhere(st[ci])
        share = np.array([1.0 / sum(1 for h in grasps if h["base"] == grasps[g]["base"] and h["twin"] == grasps[g]["twin"]) for _, g in pairs])
        yi, gi = pairs[int(rng.choice(len(pairs), p=share / share.sum()))]
        sc = width.of(ci)
        cx, cy = rm.cells[ci]
        cell = (float(cx + rng.uniform(-0.5, 0.5) * lat * sc), float(cy + rng.uniform(-0.5, 0.5) * lat * sc))
        yaw = float(rm.yaws[yi] + rng.uniform(-0.5, 0.5) * dyaw * sc)
        g = dict(grasps[gi])
        step = fam_step.get(g["base"], 0.0)
        if step > 0:
            g["param_continuous"] = float(g["param"]) + float(rng.uniform(-0.5, 0.5) * step * sc)
        nb = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                j = cell_lookup.get((round(float(cx + dx * lat), 4), round(float(cy + dy * lat), 4)))
                if j is not None:
                    nb.append(bool(st[j].any()))
        out.append(Proposal(cell, yaw, g, ci, int(yi), int(gi), {"cell_m": lat * sc, "yaw_rad": dyaw * sc, "family_param": step * sc},
                            float(np.mean(nb)) if nb else 0.0, float(w[ci])))
    return out


def release_candidates(rm: RM.ReachMap, edge_distance: float, translate: float, tilt: float, T_base_eef_query: np.ndarray, grasp_name: str,
                       pos_tol: float = 0.04, rot_tol_rad: float = 0.35) -> List[Dict[str, Any]]:
    """Release / pre-release ranking: map entries whose world eef pose (same grasp row, the map's own support height) lies within the
    tolerances of a queried robot-relative eef pose - an acceleration that matches the FULL eef pose and grasp, not the height alone.
    The answer is a kinematic proposal: held-object collisions, the release opening and the destination still need exact checks, and an
    empty answer means "no compatible map entry", not infeasible (fall back to exact IK)."""
    from ispatialgym.grasping import schema as S

    names = [g["name"] for g in rm.meta["grasps"]]
    if grasp_name not in names:
        return []
    gi = names.index(grasp_name)
    st = rm.status(edge_distance, translate, tilt)[:, :, gi] == RM.STATUS["feasible"]
    T_oe = S.pose_of(rm.meta["grasps"][gi]["T_object_eef"])
    obj = rm.meta["object"]
    rest = obj["rests"][rm.meta["rest_id"]]
    h = float(rm.meta["height"]["value"])
    out = []
    Rq = S.quat_to_mat(rest["quat_xyzw"])
    for ci, yi in np.argwhere(st):
        To = np.eye(4)
        yaw = float(rm.yaws[yi])
        To[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]]) @ Rq
        To[:3, 3] = [rm.cells[ci][0], rm.cells[ci][1], h + rest["h_root"]]
        T = To @ T_oe
        dp = float(np.linalg.norm(T[:3, 3] - T_base_eef_query[:3, 3]))
        R = T[:3, :3].T @ T_base_eef_query[:3, :3]
        da = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if dp <= pos_tol and da <= rot_tol_rad:
            out.append({"cell": rm.cells[ci].tolist(), "yaw": yaw, "dpos": dp, "drot": da, "support_height": h,
                        "effective_object_height": h + rest["h_root"]})
    return sorted(out, key=lambda d: d["dpos"] + 0.1 * d["drot"])
