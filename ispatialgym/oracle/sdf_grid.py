"""Signed-distance voxel grids (cuRobo ``VoxelGrid`` layers) standing in for static mesh obstacles - the opt-in
``--collision sdf-grid`` mode of ``reach_bands`` (approximate but ~25x cheaper per collision evaluation).

cuRobo 0.8.0 facts this module relies on (``curobo/_src/geom/data/data_voxel.py``):

* ESDF convention: **negative inside, positive outside**; the unified sphere kernel computes
  ``penetration = -sdf + radius + activation`` (``wp_collision_kernel.py:151``), the same as for meshes and cuboids;
* features are stored **float16** (``VoxelData.create_cache``: ``torch.full(..., dtype=torch.float16)``); the lookup is a
  **trilinear** interpolation with voxel centres at ``(i - (n - 1) / 2) * voxel_size`` in the grid frame
  (``sample_voxel_sdf_with_grad``); corners outside the grid are dropped from the interpolation and a query with no
  valid corner returns ``max_esdf_distance`` (10000 for ``create_cache``) - **outside the grid is free space**;
* every layer of a voxel store has the same voxel count (``features: (envs, layers, n_voxels, 1)``); layers are
  addressed by name and can be enabled / disabled (``VoxelData.set_enabled``) and re-posed (``update_pose``) like meshes;
  ``load_batch`` (not ``add_obstacle``, which replaces the whole set) loads several layers at once;
* the kernel truncates ``params = dims / voxel_size`` to ``int`` per axis, so the grid dims must divide by the voxel
  size *exactly in float32* (:func:`grid_spec` nudges them);
* the mesh SDF of a point farther than half the (patched) bounding-box diagonal from a mesh is that half-diagonal
  (``data_mesh.compute_local_sdf``); the grids are sampled with the very same per-mesh function, so they reproduce the
  exact checker's values at the voxel centres (including that floor) and differ only by the interpolation between
  centres and the float16 rounding.

Grid geometry: axis-aligned in the **world** frame, centred on the stance (base xy), ``2 * HALF_XY`` wide, from
``floor - Z_BELOW`` to ``floor + Z_ABOVE``; the layer's pose in the cuRobo (robot root) frame is set when loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

#: half width of the grid around the base (m): the arm span from the base centre is < 1.3 m (ObstacleScope.RADIUS = 1.6)
HALF_XY = 1.8
#: grid bottom below the floor / top above it (m). The top covers the TrunkBound ball of the band arm at the tallest trunk
#: (shoulder 1.77 m at translate 0.95 + 1.0 m arm ball + margins = 2.83 m; ``BandComputer.sdf_extent_check`` verifies per band)
Z_BELOW = 0.1
Z_ABOVE = 2.9
#: value of voxels no gridded mesh reaches (m). Any value above the mesh query floor (0.5 * sqrt(3) * MESH_QUERY_MIN_DIMS =
#: 0.346 m) is equivalent for the kernel: no sphere (<= 0.18 m) plus activation (<= 0.03 m) can penetrate it.
FAR = 1.0
#: stored values are clamped to +-CLAMP (float16 keeps <= 4 mm resolution there; only |sdf| < radius + activation matters)
CLAMP = 8.0
#: bump when the sampling semantics change (invalidates the on-disk cache)
VERSION = 1
LAYER_STATIC = "__sdf_static__"
LAYER_SUPPORT = "__sdf_support__"
LAYER_GROUND = "__sdf_ground__"
LAYERS = (LAYER_STATIC, LAYER_SUPPORT, LAYER_GROUND)


def grid_spec(centre_xy: Sequence[float], z_lo: float, z_hi: float, res: float, half_xy: float = HALF_XY) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n (3,) int, dims (3,) float, centre (3,) float)`` of the axis-aligned grid: voxel counts, extents (m) and centre.

    ``dims[k] / res`` is an exact integer in float32 arithmetic (cuRobo's kernel truncates ``dims / voxel_size``), which
    a plain ``n * res`` product is not for e.g. 0.01 (``float32(3.6) / float32(0.01) = 359.99997``)."""
    n = np.array([int(round(2 * half_xy / res)), int(round(2 * half_xy / res)), int(round((z_hi - z_lo) / res))], dtype=int)
    dims = []
    for k in range(3):
        d = float(n[k] * res)
        for _ in range(64):
            if int(np.float32(np.float32(d) / np.float32(res))) == n[k] and round(d / res) == n[k]:
                break
            d = float(np.nextafter(np.float32(d), np.float32(np.inf)))
        else:
            raise ValueError(f"no float32-exact grid extent for n={n[k]} res={res}")
        dims.append(d)
    centre = np.array([float(centre_xy[0]), float(centre_xy[1]), 0.5 * (float(z_lo) + float(z_hi))])
    return n, np.array(dims), centre


def grid_bounds(n: np.ndarray, res: float, centre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame ``(lo, hi)`` of the grid box (outer voxel faces)."""
    half = np.asarray(n, dtype=float) * res / 2.0
    return np.asarray(centre) - half, np.asarray(centre) + half


def voxel_centre_index(pts: np.ndarray, n: np.ndarray, res: float, centre: np.ndarray) -> np.ndarray:
    """Continuous voxel coordinates of world points (``i`` where voxel ``i`` is centred at ``centre + (i - (n - 1) / 2) res``)."""
    return (np.asarray(pts, dtype=float) - np.asarray(centre)) / res + (np.asarray(n, dtype=float) - 1.0) / 2.0


def voxel_subbox(lo_w: np.ndarray, hi_w: np.ndarray, dilate: float, n: np.ndarray, res: float, centre: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Voxel index box ``(lo (3,), span (3,))`` covering the world box ``[lo_w, hi_w]`` dilated by ``dilate`` (m), clipped to
    the grid; ``None`` when the dilated box misses the grid."""
    lo_i = np.floor(voxel_centre_index(np.asarray(lo_w, dtype=float) - dilate, n, res, centre)).astype(int)
    hi_i = np.ceil(voxel_centre_index(np.asarray(hi_w, dtype=float) + dilate, n, res, centre)).astype(int)
    lo_i = np.maximum(lo_i, 0)
    hi_i = np.minimum(hi_i, np.asarray(n) - 1)
    if np.any(hi_i < lo_i):
        return None
    return lo_i, hi_i - lo_i + 1


def quat_wxyz_to_mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def mesh_world_aabb(vertices: np.ndarray, scale: Sequence[float], pose_wxyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """World AABB of a mesh given its local vertices, scale and pose ``[x, y, z, qw, qx, qy, qz]``."""
    v = np.asarray(vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    w = v @ quat_wxyz_to_mat(pose_wxyz[3:7]).T + np.asarray(pose_wxyz[:3], dtype=np.float64)
    return w.min(axis=0), w.max(axis=0)


def geometry_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def mesh_record(name: str, pose_wxyz: Sequence[float], scale: Sequence[float], vertices: np.ndarray, faces: np.ndarray) -> Dict[str, Any]:
    """Cache-key record of one mesh: prim path, world pose / scale (rounded to 0.1 mm), geometry hash and sizes."""
    return {"name": name, "pose": [round(float(v), 4) for v in pose_wxyz], "scale": [round(float(v), 4) for v in scale], "geom": geometry_hash(vertices, faces),
            "n_verts": int(len(vertices)), "n_faces": int(len(faces))}


def cache_key(scene: str, res: float, n: np.ndarray, centre: np.ndarray, records: Sequence[Dict[str, Any]], min_dims: float, far: float = FAR) -> str:
    """sha256 (first 24 hex digits) of everything the sampled layer depends on."""
    d = {"v": VERSION, "scene": scene, "res": float(res), "n": [int(v) for v in n], "centre": [round(float(v), 4) for v in centre], "min_dims": float(min_dims),
         "far": float(far), "clamp": CLAMP, "meshes": sorted(records, key=lambda r: r["name"])}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:24]


def cache_paths(cache_dir: str, key: str) -> Tuple[str, str]:
    return os.path.join(cache_dir, key + ".npz"), os.path.join(cache_dir, key + ".json")


def load_cached(cache_dir: str, key: str, n: np.ndarray, device) -> Optional[th.Tensor]:
    """The cached float16 layer ``(nx, ny, nz)`` on ``device`` or ``None``."""
    npz, _ = cache_paths(cache_dir, key)
    if not os.path.exists(npz):
        return None
    try:
        arr = np.load(npz)["features"]
    except (OSError, ValueError, KeyError):
        return None
    if arr.dtype != np.float16 or tuple(arr.shape) != tuple(int(v) for v in n):
        return None
    return th.from_numpy(arr).to(device)


def save_cached(cache_dir: str, key: str, features16: th.Tensor, meta: Dict[str, Any]) -> Tuple[str, int]:
    """Write ``<key>.npz`` (float16 ``features``) + ``<key>.json`` atomically; returns ``(npz path, bytes)``."""
    os.makedirs(cache_dir, exist_ok=True)
    npz, js = cache_paths(cache_dir, key)
    tmp = npz[:-4] + f".tmp{os.getpid()}.npz"  # numpy appends .npz to other suffixes
    np.savez_compressed(tmp, features=features16.detach().cpu().numpy().astype(np.float16))
    os.replace(tmp, npz)
    with open(js, "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return npz, os.path.getsize(npz)


# ----------------------------------------------------------------------------------------------- sampling (GPU)
# The kernel reuses cuRobo's own per-mesh SDF function (registered as a wp.func of this module, as cuRobo's tests do), so
# the sampled values are exactly what the live mesh checker computes at those points.
import warp as wp  # noqa: E402  (pip warp: curobo is imported before Isaac Sim by the callers)
from curobo._src.geom.data import data_mesh  # noqa: E402
from curobo._src.geom.data.data_mesh import MeshDataWarp  # noqa: E402

_mesh_sdf = wp.func(data_mesh.compute_local_sdf, module=__name__)
_mesh_tf = wp.func(data_mesh.load_obstacle_transform, module=__name__)


@wp.kernel(enable_backward=False)
def esdf_mesh_min_kernel(obs_set: MeshDataWarp, local_idx: wp.int32, n: wp.vec3i, vs: wp.float32, centre: wp.vec3, lo: wp.vec3i, span: wp.vec3i,
                         feat: wp.array(dtype=wp.float32)):
    """feat[voxel] = min(feat[voxel], sdf of mesh slot ``local_idx`` at the voxel centre) over the voxel sub-box [lo, lo + span)."""
    tid = wp.tid()
    sy = span[1]
    sz = span[2]
    k = tid - (tid / sz) * sz
    j = (tid / sz) - ((tid / sz) / sy) * sy
    i = tid / (sz * sy)
    if i >= span[0]:
        return
    ix = lo[0] + i
    iy = lo[1] + j
    iz = lo[2] + k
    p = wp.vec3(centre[0] + (wp.float32(ix) - (wp.float32(n[0]) - 1.0) * 0.5) * vs,
                centre[1] + (wp.float32(iy) - (wp.float32(n[1]) - 1.0) * 0.5) * vs,
                centre[2] + (wp.float32(iz) - (wp.float32(n[2]) - 1.0) * 0.5) * vs)
    inv_t = _mesh_tf(obs_set, 0, local_idx)
    lp = wp.transform_point(inv_t, p)
    d = _mesh_sdf(obs_set, 0, local_idx, lp)
    flat = ix * n[1] * n[2] + iy * n[2] + iz
    feat[flat] = wp.min(feat[flat], d)


def sample_layer(meshes: Sequence[Dict[str, Any]], n: np.ndarray, res: float, centre: np.ndarray, device_cfg, min_dims: float, far: float = FAR) -> Tuple[th.Tensor, Dict[str, Any]]:
    """Sample one layer: ``float16 (nx, ny, nz)`` = min over ``meshes`` of cuRobo's per-mesh signed distance at the voxel centres
    (a temporary mesh store holding exactly these meshes, bounding boxes patched to ``min_dims`` like the live checker), ``far``
    where no mesh's query radius reaches. ``meshes``: dicts ``{name, pose (wxyz, world), scale, vertices, faces}``."""
    from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
    from curobo.scene import Mesh, Scene

    dev = device_cfg.device
    t0 = time.time()
    n_vox = int(np.prod(n))
    if not meshes:
        return th.full((int(n[0]), int(n[1]), int(n[2])), float(far), device=dev, dtype=th.float16), {"n_meshes": 0, "n_voxel_queries": 0, "sample_s": 0.0}
    tmp = create_collision_checker(SceneCollisionCfg(device_cfg=device_cfg, scene_model=None, cache={"mesh": len(meshes), "cuboid": 1}))
    tmp.load_collision_model(Scene(mesh=[Mesh(name=m["name"], pose=[float(v) for v in m["pose"]], vertices=np.asarray(m["vertices"], dtype=np.float32),
                                              faces=np.asarray(m["faces"], dtype=np.int64), scale=np.asarray(m["scale"], dtype=np.float32)) for m in meshes]))
    with th.no_grad():
        tmp.data.meshes.dims[..., :3].clamp_(min=float(min_dims))
    dims_patched = tmp.data.meshes.dims[0, :, :3].detach().cpu().numpy()
    feat = th.full((n_vox,), float(far), device=dev, dtype=th.float32)
    obs = tmp.data.meshes.to_warp()
    n_q = 0
    for m in meshes:
        slot = tmp.data.meshes.get_idx(m["name"])
        lo_w, hi_w = mesh_world_aabb(m["vertices"], m["scale"], m["pose"])
        box = voxel_subbox(lo_w, hi_w, 0.5 * float(np.linalg.norm(dims_patched[slot])), n, res, centre)
        if box is None:
            continue
        lo_i, span = box
        cnt = int(np.prod(span))
        n_q += cnt
        wp.launch(kernel=esdf_mesh_min_kernel, dim=cnt,
                  inputs=[obs, int(slot), wp.vec3i(int(n[0]), int(n[1]), int(n[2])), float(res), wp.vec3(*[float(c) for c in centre]),
                          wp.vec3i(*[int(v) for v in lo_i]), wp.vec3i(*[int(v) for v in span]), wp.from_torch(feat)],
                  device=str(dev))
    wp.synchronize_device(str(dev))
    th.cuda.synchronize()
    out = feat.clamp_(-CLAMP, CLAMP).to(th.float16).view(int(n[0]), int(n[1]), int(n[2])).contiguous()
    del feat, obs, tmp
    th.cuda.empty_cache()
    return out, {"n_meshes": len(meshes), "n_voxel_queries": int(n_q), "sample_s": round(time.time() - t0, 3)}


def layer_stats(features16: th.Tensor, res: float) -> Dict[str, Any]:
    f = features16.float()
    return {"n_inside": int((f < 0).sum()), "n_within_0.2m": int((f.abs() < 0.2).sum()), "min_m": round(float(f.min()), 4), "res": float(res)}
