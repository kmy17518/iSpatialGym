"""Traversability-map patches of an instance's scene variant.

The traversability map is baked with the scene's furniture. A template records the footprints of the objects moved out
of the scene (``scene_variant.freed_footprints`` of its metadata) and an instance the footprints of objects added to it
(``scene_variant.blocked_footprints`` of its TRO scene block); the loader applies both to the loaded map so it matches
the scene. Footprints are convex polygons in world xy.
"""

from __future__ import annotations

import os

import torch as th


def _rasterise(trav_map, polygons: list[list[list[float]]], value: int, mask: th.Tensor | None = None) -> None:
    """Set the cells whose centre lies inside any polygon to ``value`` (where ``mask`` is True)."""
    m = trav_map.floor_map[0]
    h, w = m.shape
    for poly in polygons:
        pts = th.tensor(poly, dtype=th.float32)
        ij = trav_map.world_to_map(pts)
        i0, i1 = max(int(th.floor(ij[:, 0].min())) - 1, 0), min(int(th.ceil(ij[:, 0].max())) + 1, h - 1)
        j0, j1 = max(int(th.floor(ij[:, 1].min())) - 1, 0), min(int(th.ceil(ij[:, 1].max())) + 1, w - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = th.meshgrid(th.arange(i0, i1 + 1), th.arange(j0, j1 + 1), indexing="ij")
        cells = th.stack([ii.flatten(), jj.flatten()], dim=1).float()
        xy = trav_map.map_to_world(cells)
        signs = []
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            signs.append((b[0] - a[0]) * (xy[:, 1] - a[1]) - (b[1] - a[1]) * (xy[:, 0] - a[0]))
        signs = th.stack(signs, dim=1)
        # a convex polygon contains the points on the same side of every edge
        inside = (signs >= -1e-6).all(dim=1) | (signs <= 1e-6).all(dim=1)
        sel = cells[inside].long()
        if mask is not None:
            sel = sel[mask[sel[:, 0], sel[:, 1]]]
        m[sel[:, 0], sel[:, 1]] = value


def patch_trav_map_from_meta(scene, template_meta: dict) -> int:
    """Free the template's moved-object footprints (where the scene's no-object map has floor), then re-block the
    remaining objects' footprints recorded with them. Returns the number of cells freed."""
    footprints = (template_meta.get("scene_variant") or {}).get("freed_footprints")
    if not footprints:
        return 0
    import cv2

    trav = scene.trav_map
    before = int((trav.floor_map[0] == 255).sum())
    floor_ok = None
    no_obj_map = (
        os.path.join(scene.scene_dir, "layout", "floor_trav_no_obj_0.png")
        if getattr(scene, "scene_dir", None)
        else None
    )
    if no_obj_map and os.path.exists(no_obj_map):
        size = trav.floor_map[0].shape[0]
        floor_ok = th.tensor(cv2.resize(cv2.imread(no_obj_map, cv2.IMREAD_GRAYSCALE), (size, size))) >= 255
    _rasterise(trav, footprints.get("free", []), 255, mask=floor_ok)
    _rasterise(trav, footprints.get("block", []), 0)
    return int((trav.floor_map[0] == 255).sum()) - before


def block_trav_map_from_meta(scene, scene_meta: dict) -> int:
    """Block the instance's added-object footprints. Returns the number of cells blocked."""
    footprints = (scene_meta.get("scene_variant") or {}).get("blocked_footprints")
    if not footprints:
        return 0
    trav = scene.trav_map
    before = int((trav.floor_map[0] == 255).sum())
    _rasterise(trav, footprints.get("block", []), 0)
    return before - int((trav.floor_map[0] == 255).sum())
