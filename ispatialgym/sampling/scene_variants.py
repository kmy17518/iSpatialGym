"""Scene variants: the stable scene minus objects that must not be there for a task (chairs around the work table).

A variant is a copy of ``<scene>_stable.json`` without the listed objects, written next to the sampler output
(``<out>/scene_variants/<scene>__<hash>.json``). The sampler loads it as the scene file, so templates (merged from
it) and every instance are self-consistent, and the reload checks compare against it.

The traversability map is a PNG baked with the furniture, so removing a chair leaves its footprint marked as an
obstacle. :func:`patch_trav_map` frees those cells in the loaded ``TraversableMap`` (only where the no-object map has
floor, and not under a remaining object's footprint). Templates record the freed footprints
(``template_meta.scene_variant.freed_footprints``) so that evaluation can apply the same patch
(:func:`patch_trav_map_from_meta`) - the oracle's base planner otherwise still treats the chair spots as blocked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling import knowledge as K

#: categories treated as "chairs" around a work table
CHAIR_CATEGORIES = ("straight_chair", "armchair", "swivel_chair", "folding_chair", "highchair", "stool", "bar_stool", "rocking_chair", "office_chair")


def _quat_to_yaw(q: Sequence[float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _bbox_footprint(init_args: dict, state: dict) -> Optional[th.Tensor]:
    """World xy corners (4, 2) of the object's canonical bounding box projected on the floor (yaw only), or None."""
    model = init_args.get("model")
    bb = K.canonical_bbox(model) if model else None
    if bb is None:
        return None
    meta = _metadata(init_args.get("category"), model)
    offset = th.tensor(meta.get("base_link_offset", [0.0, 0.0, 0.0]), dtype=th.float32)
    scale = th.tensor(init_args.get("scale") or [1.0, 1.0, 1.0], dtype=th.float32)
    pos = th.tensor(state["root_link"]["pos"], dtype=th.float32)
    yaw = _quat_to_yaw(state["root_link"]["ori"])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = th.tensor([[c, -s], [s, c]])
    centre_local = -scale * offset  # OmniGibson: scaled_bbox_center_in_base_frame
    centre = pos[:2] + rot @ centre_local[:2]
    hx, hy = float(bb[0] * scale[0]) / 2, float(bb[1] * scale[1]) / 2
    corners = th.tensor([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return centre + corners @ rot.T


def _bbox_zrange(init_args: dict, state: dict) -> Tuple[float, float]:
    model = init_args.get("model")
    bb = K.canonical_bbox(model) if model else None
    meta = _metadata(init_args.get("category"), model)
    scale = init_args.get("scale") or [1.0, 1.0, 1.0]
    z = float(state["root_link"]["pos"][2]) - float(scale[2]) * float(meta.get("base_link_offset", [0, 0, 0])[2])
    h = float(bb[2] * scale[2]) if bb else 0.5
    return z - h / 2, z + h / 2


_META_CACHE: Dict[str, dict] = {}


def _metadata(category: Optional[str], model: Optional[str]) -> dict:
    if not category or not model:
        return {}
    key = f"{category}/{model}"
    if key not in _META_CACHE:
        from omnigibson.utils.asset_utils import get_dataset_path

        path = os.path.join(get_dataset_path("behavior-1k-assets"), "objects", category, model, "misc", "metadata.json")
        try:
            with open(path) as f:
                _META_CACHE[key] = json.load(f)
        except OSError:
            _META_CACHE[key] = {}
    return _META_CACHE[key]


def _rect_distance(p: th.Tensor, lo: th.Tensor, hi: th.Tensor) -> float:
    d = th.maximum(th.maximum(lo - p, p - hi), th.zeros(2))
    return float(th.linalg.norm(d))


def chairs_near_tables(stable: dict, rooms: Iterable[str], tables: Iterable[str], distance: float = 1.0,
                       categories: Sequence[str] = CHAIR_CATEGORIES) -> List[str]:
    """Names of chair-like objects (in ``rooms``) whose footprint centre lies within ``distance`` of the footprint of
    any object in ``tables`` (scene object names)."""
    init = stable["objects_info"]["init_info"]
    reg = stable["state"]["registry"]["object_registry"]
    rects = []
    for t in tables:
        if t in init and t in reg:
            fp = _bbox_footprint(init[t]["args"], reg[t])
            if fp is not None:
                rects.append((fp.min(dim=0).values, fp.max(dim=0).values))
    if not rects:
        return []
    rooms = set(rooms)
    out = []
    for name, info in init.items():
        a = info["args"]
        if a.get("category") not in categories or name not in reg:
            continue
        if rooms and not (set(a.get("in_rooms") or []) & rooms):
            continue
        fp = _bbox_footprint(a, reg[name])
        if fp is None:
            continue
        c = fp.mean(dim=0)
        if any(_rect_distance(c, lo, hi) <= distance for lo, hi in rects):
            out.append(name)
    return sorted(out)


def tables_in_rooms(stable: dict, rooms: Iterable[str], names_or_categories: Iterable[str]) -> List[str]:
    """Scene object names in ``rooms`` matching the given names or categories."""
    init = stable["objects_info"]["init_info"]
    rooms = set(rooms)
    wanted = set(names_or_categories)
    out = []
    for name, info in init.items():
        a = info["args"]
        if (name in wanted or a.get("category") in wanted) and (not rooms or set(a.get("in_rooms") or []) & rooms):
            out.append(name)
    return sorted(out)


def write_variant(stable_path: str, remove: Sequence[str], out_dir: str) -> Tuple[str, dict]:
    """Write the stable scene minus ``remove`` to ``out_dir`` and return (path, json)."""
    with open(stable_path) as f:
        scene = json.load(f)
    variant = copy.deepcopy(scene)
    init = variant["objects_info"]["init_info"]
    reg = variant["state"]["registry"]["object_registry"]
    removed = []
    for name in remove:
        if name in init:
            init.pop(name)
            reg.pop(name, None)
            removed.append(name)
    variant.setdefault("metadata", {}).setdefault("ispatialgym", {})["removed_objects"] = removed
    variant["metadata"]["ispatialgym"]["source_scene_file"] = os.path.basename(stable_path)
    tag = hashlib.sha256(",".join(sorted(removed)).encode()).hexdigest()[:10]
    scene_model = os.path.basename(stable_path).replace("_stable.json", "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene_model}__{tag}.json")
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(variant, f)
        os.replace(tmp, path)
    return path, variant


def freed_footprints(stable: dict, removed: Sequence[str], keep_blocked_below_z: float = 1.4) -> Dict[str, List]:
    """Footprints to free in the traversability map (removed objects) and footprints that stay blocked (remaining
    non-structural objects that a chair may have been tucked under, e.g. the table)."""
    init = stable["objects_info"]["init_info"]
    reg = stable["state"]["registry"]["object_registry"]
    free, block = [], []
    for name in removed:
        if name in init and name in reg:
            fp = _bbox_footprint(init[name]["args"], reg[name])
            if fp is not None:
                free.append(fp.tolist())
    removed_set = set(removed)
    for name, info in init.items():
        a = info["args"]
        if name in removed_set or name not in reg or a.get("category") in K.STRUCTURE_CATEGORIES:
            continue
        fp = _bbox_footprint(a, reg[name])
        if fp is None:
            continue
        zlo, zhi = _bbox_zrange(a, reg[name])
        if zlo > keep_blocked_below_z or zhi < 0.05:
            continue  # ceiling fixtures / wall-mounted high objects do not block the base
        # only footprints that overlap a freed one matter
        fmin, fmax = fp.min(dim=0).values, fp.max(dim=0).values
        for f in free:
            t = th.tensor(f)
            if bool(((t.max(dim=0).values >= fmin) & (t.min(dim=0).values <= fmax)).all()):
                block.append(fp.tolist())
                break
    return {"free": free, "block": block}


def _rasterise(trav_map, polygons: List[List[List[float]]], value: int, mask: Optional[th.Tensor] = None):
    """Set map cells whose centre lies inside any convex polygon (world xy) to ``value`` (where ``mask`` is True)."""
    m = trav_map.floor_map[0]
    h, w = m.shape
    for poly in polygons:
        pts = th.tensor(poly, dtype=th.float32)
        ij = trav_map.world_to_map(pts)  # (4, 2) map indices (float)
        i0, i1 = int(th.floor(ij[:, 0].min())) - 1, int(th.ceil(ij[:, 0].max())) + 1
        j0, j1 = int(th.floor(ij[:, 1].min())) - 1, int(th.ceil(ij[:, 1].max())) + 1
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, h - 1), min(j1, w - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = th.meshgrid(th.arange(i0, i1 + 1), th.arange(j0, j1 + 1), indexing="ij")
        cells = th.stack([ii.flatten(), jj.flatten()], dim=1).float()
        xy = trav_map.map_to_world(cells)
        n = len(pts)
        # convex polygon given in consistent order: inside if all edge cross products share a sign
        signs = []
        for k in range(n):
            a, b = pts[k], pts[(k + 1) % n]
            signs.append((b[0] - a[0]) * (xy[:, 1] - a[1]) - (b[1] - a[1]) * (xy[:, 0] - a[0]))
        sg = th.stack(signs, dim=1)
        inside = (sg >= -1e-6).all(dim=1) | (sg <= 1e-6).all(dim=1)
        sel = cells[inside].long()
        if mask is not None:
            keep = mask[sel[:, 0], sel[:, 1]]
            sel = sel[keep]
        m[sel[:, 0], sel[:, 1]] = value


def patch_trav_map(scene, footprints: Dict[str, List], scene_dir: Optional[str] = None) -> int:
    """Free the removed objects' footprints in ``scene.trav_map`` (where the no-object map has floor), then re-block
    the remaining objects' footprints that overlap them. Returns the number of cells freed."""
    import cv2

    trav = scene.trav_map
    before = int((trav.floor_map[0] == 255).sum())
    floor_ok = None
    scene_dir = scene_dir or getattr(scene, "scene_dir", None)
    if scene_dir:
        p = os.path.join(scene_dir, "layout", "floor_trav_no_obj_0.png")
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            size = trav.floor_map[0].shape[0]
            img = cv2.resize(img, (size, size))
            floor_ok = th.tensor(img) >= 255
    _rasterise(trav, footprints.get("free", []), 255, mask=floor_ok)
    _rasterise(trav, footprints.get("block", []), 0)
    return int((trav.floor_map[0] == 255).sum()) - before


def patch_trav_map_from_meta(scene, template_meta: dict) -> int:
    """Apply the traversability patch recorded in ``template_meta['scene_variant']`` (no-op when absent)."""
    sv = template_meta.get("scene_variant") or {}
    fps = sv.get("freed_footprints")
    if not fps:
        return 0
    return patch_trav_map(scene, fps)
