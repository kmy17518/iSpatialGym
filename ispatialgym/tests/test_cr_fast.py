"""train-fast camera relocalization switches: the geodesic field (CR_GEODESIC_FIELD) and the obstacle-cell draw
(CR_OBSTACLE_PREFILTER), pure helpers only."""

import math

import numpy as np
import pytest

from ispatialgym.sampling.v2.tasks import camera_relocalization as CR


def _old_path_cells(rng, n_points, max_cells):
    if n_points < 3:
        return []
    lo, hi = int(math.floor(0.3 * (n_points - 1))), int(math.ceil(0.7 * (n_points - 1)))
    idx = list(range(max(lo, 1), min(hi, n_points - 2) + 1)) or list(range(1, n_points - 1))
    perm = rng.permutation(len(idx))
    return [idx[int(i)] for i in perm[:max_cells]]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7, 11, 40, 101])
def test_path_cells_unchanged_without_allowed(n):
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert CR.path_cells(a, n, 8) == _old_path_cells(b, n, 8)
    assert a.integers(1 << 30) == b.integers(1 << 30)  # same rng consumption


def test_path_cells_allowed_restricts_the_draw():
    rng = np.random.default_rng(0)
    rng_idx = CR.path_cell_range(41)
    allowed = rng_idx[::3] + [0, 40]  # 0 / 40 lie outside the middle: never drawn
    got = CR.path_cells(rng, 41, 100, allowed=allowed)
    assert sorted(got) == sorted(rng_idx[::3])
    assert CR.path_cells(rng, 41, 8, allowed=[]) == []


def test_field_open_grid_is_octile_distance():
    free = np.ones((30, 40), dtype=bool)
    f = CR.geodesic_field(free, (3, 5))
    for (i, j) in [(3, 5), (3, 30), (20, 5), (25, 39), (0, 0), (29, 0)]:
        di, dj = abs(i - 3), abs(j - 5)
        assert f[i, j] == pytest.approx(math.sqrt(2) * min(di, dj) + abs(di - dj), abs=1e-9)


def test_field_wall_and_unreachable():
    free = np.ones((20, 20), dtype=bool)
    free[:, 10] = False
    free[15, 10] = True  # one door
    free[0:3, 15:18] = False
    free[1, 16] = True  # an enclosed pocket
    f = CR.geodesic_field(free, (5, 2))
    assert math.isinf(f[1, 16])
    assert math.isinf(f[0, 10])  # blocked cells are never reached
    assert np.isfinite(f[5, 18])
    assert f[5, 18] > math.hypot(0, 16) + 1  # around through the door


def test_field_source_admitted_when_not_free():
    free = np.ones((5, 5), dtype=bool)
    free[2, 2] = False
    f = CR.geodesic_field(free, (2, 2))
    assert f[2, 2] == 0.0 and f[2, 4] == pytest.approx(2.0)
    assert np.isinf(CR.geodesic_field(free, (9, 9))).all()  # a source off the map reaches nothing


def test_field_matches_heap_reference_on_random_maps():
    rng = np.random.default_rng(3)
    for _ in range(5):
        free = rng.random((35, 28)) > 0.3
        src = (int(rng.integers(35)), int(rng.integers(28)))
        a = CR.geodesic_field(free, src)
        f2 = free.copy()
        f2[src] = True
        b = CR._geodesic_field_heap(f2, src)
        assert np.array_equal(np.isinf(a), np.isinf(b))
        fin = np.isfinite(a)
        assert np.allclose(a[fin], b[fin], atol=1e-9)
