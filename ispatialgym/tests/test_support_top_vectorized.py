"""i7-stancememo r3 (lead decision #45): SupportTop.contains / polygon_supported vectorized (disembedding's start draw spent 220 of 285 s
in the scalar loops). The verdicts must be the former loops' exactly - checked against a verbatim copy of them on random rasters,
points, margins and footprints (float32 and float64 inputs, points on cell borders and off the raster)."""
import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import SupportTop


def _old_contains(top, xy, margin=0.0):
    pts = [G.t(xy)]
    if margin > 0:
        for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin), (margin, -margin), (-margin, margin)):
            pts.append(G.t(xy) + th.tensor([dx, dy]))
    for p in pts:
        i = int(th.round((p[0] - top.xs[0]) / top.resolution))
        j = int(th.round((p[1] - top.ys[0]) / top.resolution))
        if i < 0 or j < 0 or i >= len(top.xs) or j >= len(top.ys) or not bool(top.free[i, j]):
            return False
    return True


def _old_polygon_supported(top, poly_xy, margin=0.02):
    for p in poly_xy:
        if not _old_contains(top, p, margin):
            return False
    return _old_contains(top, poly_xy.mean(dim=0), 0.0)


def _top(rng, nx=40, ny=30, res=0.02):
    xs = th.arange(nx, dtype=th.float32) * res + float(rng.uniform(-3, 3))
    ys = th.arange(ny, dtype=th.float32) * res + float(rng.uniform(-3, 3))
    mask = th.as_tensor(rng.random((nx, ny)) < 0.9)
    occ = th.as_tensor(rng.random((nx, ny)) < 0.1)
    return SupportTop("t", xs, ys, mask, occ, 0.8, 0.3, resolution=res)


def test_contains_and_polygon_supported_match_the_former_loops():
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(20):
        top = _top(rng)
        x0, y0 = float(top.xs[0]), float(top.ys[0])
        for _ in range(300):
            margin = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.03, -0.01]))
            dtype = th.float32 if rng.random() < 0.5 else th.float64
            c = th.tensor([x0 + rng.uniform(-0.1, 0.9), y0 + rng.uniform(-0.1, 0.7)], dtype=dtype)
            if rng.random() < 0.3:  # exactly on a cell border (half-way: the rounding's tie)
                c = th.tensor([x0 + 0.01 * int(rng.integers(0, 80)), y0 + 0.01 * int(rng.integers(0, 60))], dtype=dtype)
            assert top.contains(c, margin) == _old_contains(top, c, margin)
            poly = c + th.as_tensor(rng.uniform(-0.08, 0.08, size=(int(rng.integers(3, 9)), 2)), dtype=dtype)
            got, want = top.polygon_supported(poly, margin), _old_polygon_supported(top, poly, margin)
            assert got == want
            n += int(want)
    assert n > 50  # both verdicts occur
