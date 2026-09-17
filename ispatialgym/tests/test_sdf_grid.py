"""Simulator-free checks of the ``--collision sdf-grid`` helpers (:mod:`ispatialgym.oracle.sdf_grid`): grid geometry that cuRobo's
voxel kernel indexes exactly, the sub-box / cache-key arithmetic and the reach_bands flag plumbing (default mode untouched)."""

import numpy as np
import pytest

from ispatialgym.oracle import sdf_grid as SG
from ispatialgym.sampling import reach_bands as RB


@pytest.mark.parametrize("res", [0.02, 0.01, 0.005, 0.0125, 0.03, 0.0075])
def test_grid_spec_dims_divide_exactly_in_float32(res):
    n, dims, centre = SG.grid_spec((1.0, -2.0), -SG.Z_BELOW, SG.Z_ABOVE, res)
    assert n.shape == (3,) and dims.shape == (3,) and centre.shape == (3,)
    for k in range(3):
        # what cuRobo's kernel computes: int(float32(dims) / float32(voxel_size)) must be the voxel count (VoxelGrid.get_grid_shape rounds)
        assert int(np.float32(np.float32(dims[k]) / np.float32(res))) == n[k]
        assert round(dims[k] / res) == n[k]
        assert abs(dims[k] - n[k] * res) < 1e-5
    assert n[0] == n[1] == round(2 * SG.HALF_XY / res)
    assert n[2] == round((SG.Z_ABOVE + SG.Z_BELOW) / res)
    assert centre.tolist() == [1.0, -2.0, 0.5 * (SG.Z_ABOVE - SG.Z_BELOW)]
    lo, hi = SG.grid_bounds(n, res, centre)
    assert np.allclose(hi - lo, n * res)


def test_voxel_index_math_and_subbox():
    res = 0.01
    n, dims, centre = SG.grid_spec((0.0, 0.0), -0.05, 2.3, res)
    # voxel i is centred at centre + (i - (n - 1) / 2) res: the first centre is half a voxel inside the lower face
    lo, hi = SG.grid_bounds(n, res, centre)
    first = lo + res / 2
    assert np.allclose(SG.voxel_centre_index(first[None], n, res, centre)[0], 0.0, atol=1e-6)
    last = hi - res / 2
    assert np.allclose(SG.voxel_centre_index(last[None], n, res, centre)[0], n - 1, atol=1e-6)
    # a 0.2 m box at the grid centre dilated by 0.35 -> ~ (0.9 / 0.01) voxels per axis, symmetric around the middle
    box = SG.voxel_subbox(centre - 0.1, centre + 0.1, 0.35, n, res, centre)
    assert box is not None
    lo_i, span = box
    assert (span >= 90).all() and (span <= 92).all()
    assert np.allclose(lo_i + (span - 1) / 2.0, (n - 1) / 2.0, atol=1.0)
    # clipped to the grid at a corner
    lo_i, span = SG.voxel_subbox(lo - 1.0, lo + 0.05, 0.0, n, res, centre)
    assert (lo_i == 0).all() and (span >= 5).all() and (span <= 7).all()
    # entirely outside: nothing to sample
    assert SG.voxel_subbox(hi + 1.0, hi + 2.0, 0.3, n, res, centre) is None
    assert SG.voxel_subbox(lo - 2.0, lo - 1.0, 0.5, n, res, centre) is None


def test_mesh_world_aabb_applies_scale_and_pose():
    v = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
    lo, hi = SG.mesh_world_aabb(v, [0.5, 1.0, 2.0], [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(lo, [9.5, -1.0, -2.0]) and np.allclose(hi, [10.5, 1.0, 2.0])
    s = np.sqrt(0.5)
    lo, hi = SG.mesh_world_aabb(v, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, s, 0.0, 0.0, s])  # 90 deg about z
    assert np.allclose(lo, [-1, -1, -1], atol=1e-6) and np.allclose(hi, [1, 1, 1], atol=1e-6)
    v2 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    lo, hi = SG.mesh_world_aabb(v2, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, s, 0.0, 0.0, s])  # x axis -> y axis
    assert np.allclose(lo, [0, 0, 0], atol=1e-6) and np.allclose(hi, [0, 1, 0], atol=1e-6)


def test_cache_key_depends_on_geometry_pose_grid_but_not_order():
    v = np.zeros((4, 3), dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    a = SG.mesh_record("/World/a", [0, 0, 0, 1, 0, 0, 0], [1, 1, 1], v, f)
    b = SG.mesh_record("/World/b", [1, 0, 0, 1, 0, 0, 0], [1, 1, 1], v, f)
    n, dims, centre = SG.grid_spec((0.0, 0.0), -0.05, 2.3, 0.01)
    k = SG.cache_key("scene", 0.01, n, centre, [a, b], 0.4)
    assert len(k) == 24
    assert SG.cache_key("scene", 0.01, n, centre, [b, a], 0.4) == k  # order-insensitive
    assert SG.cache_key("other", 0.01, n, centre, [a, b], 0.4) != k
    assert SG.cache_key("scene", 0.02, n, centre, [a, b], 0.4) != k
    assert SG.cache_key("scene", 0.01, n, centre + [0.01, 0, 0], [a, b], 0.4) != k
    assert SG.cache_key("scene", 0.01, n, centre, [a], 0.4) != k
    moved = SG.mesh_record("/World/b", [1.001, 0, 0, 1, 0, 0, 0], [1, 1, 1], v, f)
    assert SG.cache_key("scene", 0.01, n, centre, [a, moved], 0.4) != k  # 1 mm pose change
    jitter = SG.mesh_record("/World/b", [1.00001, 0, 0, 1, 0, 0, 0], [1, 1, 1], v, f)
    assert SG.cache_key("scene", 0.01, n, centre, [a, jitter], 0.4) == k  # below the 0.1 mm rounding
    other_geom = SG.mesh_record("/World/b", [1, 0, 0, 1, 0, 0, 0], [1, 1, 1], v + 0.1, f)
    assert SG.cache_key("scene", 0.01, n, centre, [a, other_geom], 0.4) != k
    assert SG.cache_key("scene", 0.01, n, centre, [a, b], 0.5) != k


def test_cache_roundtrip(tmp_path):
    import torch as th

    n = np.array([4, 3, 2])
    feat = th.arange(24, dtype=th.float32).view(4, 3, 2).to(th.float16)
    path, size = SG.save_cached(str(tmp_path), "abc", feat, {"x": 1})
    assert path.endswith("abc.npz") and size > 0
    back = SG.load_cached(str(tmp_path), "abc", n, "cpu")
    assert back is not None and back.dtype == th.float16 and th.equal(back, feat)
    assert SG.load_cached(str(tmp_path), "abc", np.array([4, 3, 3]), "cpu") is None  # shape mismatch = miss
    assert SG.load_cached(str(tmp_path), "nope", n, "cpu") is None


def test_reach_bands_flags_default_to_the_exact_mode():
    ap = RB.build_parser()
    a = ap.parse_args(["--scene", "s", "--support", "t"])
    assert a.collision is None and RB.sdf_grid_config(a) is None
    assert a.sdf_res == RB.SDF_RES and a.sdf_mesh_cache == RB.SDF_MESH_CACHE and a.sdf_cache_dir is None
    g = ap.parse_args(["--scene", "s", "--support", "t", "--collision", "sdf-grid", "--sdf-res", "0.02"])
    cfg = RB.sdf_grid_config(g)
    assert cfg["res"] == 0.02 and cfg["mesh_cache"] == RB.SDF_MESH_CACHE
    assert cfg["n"] == [180, 180, int(round((SG.Z_ABOVE + SG.Z_BELOW) / 0.02))]  # 117 (2.35 / 0.02 = 117.4999... in binary floating point)
    for d, nk in zip(cfg["dims"], cfg["n"]):
        assert int(np.float32(np.float32(d) / np.float32(0.02))) == nk
    assert RB.sdf_cache_dir(g, "/tmp/x") == "/tmp/x/sdf_grids"
    g.sdf_cache_dir = "/elsewhere"
    assert RB.sdf_cache_dir(g, "/tmp/x") == "/elsewhere"
    assert set(SG.LAYERS) == {SG.LAYER_STATIC, SG.LAYER_SUPPORT, SG.LAYER_GROUND}
