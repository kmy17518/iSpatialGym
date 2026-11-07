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


# ------------------------------------------------------------------------------------------------------------------
# i7-unblock 2026-11-07: CR_FAST_ASTAR (BEHAVIOR's A* without its per-call score table) and CR_OBSTACLE_HEADROOM
# ------------------------------------------------------------------------------------------------------------------


def _behavior_astar(search_map, start, goal, eight_connected=True):
    """Verbatim copy of ``omnigibson.utils.motion_planning_utils.astar`` (frozen Behavior-1K 6a6194600): the reference."""
    import heapq

    import torch as th

    def heuristic(node):
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    def get_neighbors(cell):
        if eight_connected:
            return [(cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1),
                    (cell[0] + 1, cell[1] + 1), (cell[0] - 1, cell[1] - 1), (cell[0] + 1, cell[1] - 1), (cell[0] - 1, cell[1] + 1)]
        return [(cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]), (cell[0], cell[1] + 1), (cell[0], cell[1] - 1)]

    def is_valid(cell):
        return 0 <= cell[0] < search_map.shape[0] and 0 <= cell[1] < search_map.shape[1] and search_map[cell] != 0

    def cost(cell1, cell2):
        if cell1[0] == cell2[0] or cell1[1] == cell2[1]:
            return 1
        return math.sqrt(2)

    open_set = [(0, start)]
    came_from = {}
    visited = set()
    rows, cols = search_map.shape
    g_score = {(i.item(), j.item()): float("inf") for i, j in th.cartesian_prod(th.arange(rows), th.arange(cols))}
    g_score[start] = 0
    while open_set:
        _, current = heapq.heappop(open_set)
        visited.add(current)
        if current == goal:
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            path.insert(0, start)
            return th.tensor(path)
        for neighbor in get_neighbors(current):
            if not is_valid(neighbor) or neighbor in visited:
                continue
            tentative_g_score = g_score[current] + cost(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))
    return None


def _random_map(rng, shape, p_block):
    import torch as th

    m = (rng.random(shape) >= p_block).astype(np.uint8) * 255
    # a few walls with gaps: ties between equally long detours
    for _ in range(3):
        r = int(rng.integers(shape[0]))
        m[r, :] = 0
        for g in rng.integers(0, shape[1], size=2):
            m[r, int(g)] = 255
    return th.tensor(m)


@pytest.mark.parametrize("eight", [True, False])
def test_astar_exact_equals_behavior_astar_on_random_maps(eight):
    from ispatialgym.sampling.v2 import navigation as NAV

    rng = np.random.default_rng(11)
    n_paths = 0
    for k in range(60):
        m = _random_map(rng, (int(rng.integers(8, 40)), int(rng.integers(8, 40))), float(rng.choice([0.0, 0.1, 0.3])))
        a = (int(rng.integers(m.shape[0])), int(rng.integers(m.shape[1])))
        b = (int(rng.integers(m.shape[0])), int(rng.integers(m.shape[1])))
        if k % 7 == 0:
            b = a  # the start is the goal
        ref, got = _behavior_astar(m, a, b, eight), NAV.astar_exact(m, a, b, eight)
        assert (ref is None) == (got is None), (k, a, b)
        if ref is not None:
            n_paths += 1
            assert ref.shape == got.shape and ref.dtype == got.dtype and bool((ref == got).all()), (k, a, b)
    assert n_paths > 20
    m = th_ones = _random_map(rng, (20, 20), 0.0)
    m[:, :] = 255
    assert NAV.astar_exact(th_ones.numpy(), (2, 2), (17, 17)).tolist() == _behavior_astar(m, (2, 2), (17, 17)).tolist()  # numpy input


class _Trav:
    """The ``TraversableMap`` surface ``get_shortest_path`` uses (BEHAVIOR's formulas: map_base.world_to_map / map_to_world,
    _erode_trav_map with the robot's chassis radius, waypoint_interval) and its ``get_shortest_path`` with the verbatim A*."""

    def __init__(self, m, res=0.1):
        import torch as th

        self.floor_map, self.map_resolution, self.map_size, self.waypoint_interval = [th.tensor(m)], res, m.shape[0], int(0.2 / res)
        self._th = th

    def world_to_map(self, xy):
        th = self._th
        xy = th.as_tensor(xy)
        p = xy / self.map_resolution + self.map_size / 2.0
        return th.flip(p, dims=tuple(range(p.dim()))).int()

    def map_to_world(self, xy):
        th = self._th
        dims = 0 if xy.dim() == 1 else 1
        return th.flip((xy - self.map_size / 2.0) * self.map_resolution, dims=(dims,))

    def _erode_trav_map(self, trav_map, robot=None):
        import cv2

        th = self._th
        radius = th.norm(robot.reset_joint_pos_aabb_extent[:2]) / 2.0 + 0.2
        k = int(math.ceil(float(radius) / self.map_resolution))
        return th.tensor(cv2.erode(trav_map.cpu().numpy(), th.ones((k, k)).cpu().numpy()))

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
        th = self._th
        s, t = tuple(self.world_to_map(source_world).tolist()), tuple(self.world_to_map(target_world).tolist())
        path_map = _behavior_astar(self._erode_trav_map(th.clone(self.floor_map[floor]), robot=robot), s, t)
        if path_map is None:
            return None, None
        pw = self.map_to_world(path_map)
        return pw[:: self.waypoint_interval], th.sum(th.norm(pw[1:] - pw[:-1], dim=1))


def test_shortest_path_exact_equals_get_shortest_path():
    import torch as th

    from ispatialgym.sampling.v2 import navigation as NAV

    rng = np.random.default_rng(3)
    m = np.full((80, 80), 255, dtype=np.uint8)
    m[30:34, 0:60] = 0
    m[50:54, 20:80] = 0
    m[rng.random((80, 80)) < 0.02] = 0
    trav = _Trav(m)
    robot = type("R", (), {"reset_joint_pos_aabb_extent": th.tensor([0.4, 0.4, 1.0])})()
    for a, b in (((-3.0, -3.0), (3.0, 3.0)), ((-3.5, 2.5), (2.0, -3.0)), ((0.0, 0.0), (0.1, 0.0)), ((-3.0, -3.0), (3.9, 3.9))):
        a, b = th.tensor(a), th.tensor(b)
        p_ref, g_ref = trav.get_shortest_path(0, a, b, entire_path=True, robot=robot)
        p, g = NAV.shortest_path_exact(trav, 0, a, b, robot=robot)
        assert (p_ref is None) == (p is None)
        if p is not None:
            assert bool(th.equal(p, p_ref)) and g == float(g_ref)


def test_detour_headroom_is_exact_for_detour_failure():
    by = CR.MIN_DETOUR_M_BY_BUCKET
    assert not CR.detour_headroom(4.86, 5.0, 0.3, by)  # the teddy bear's (3, 5] goal: 0.14 m above the free path, minimum detour 0.20
    assert CR.detour_headroom(4.80, 5.0, 0.3, by) and CR.detour_headroom(2.00, 3.0, 0.3, by) and CR.detour_headroom(6.88, 10.0, 0.3, by)
    # 3.0 - 2.70 = 0.29999999999999982 < 0.30: a blocked path at the bucket's top (3.0) is itself a short detour - the float rule of
    # detour_failure, kept exactly
    assert not CR.detour_headroom(2.75, 3.0, 0.3, by) and not CR.detour_headroom(2.70, 3.0, 0.3, by) and CR.detour_failure(3.0, 2.70, 3.0, 0.3, by)
    assert CR.detour_headroom(2.69, 3.0, 0.3, by)
    assert not CR.detour_headroom(1.9, 2.0, 0.3, None)  # the fallback minimum detour of an unlisted bucket
    rng = np.random.default_rng(0)
    for _ in range(4000):
        hi = float(rng.choice([3.0, 5.0, 10.0, 2.5]))
        free = float(rng.uniform(0.3, hi))
        ok = CR.detour_headroom(free, hi, 0.3, by)
        gs = [free + float(d) for d in rng.uniform(0.0, 1.5, size=20)] + [hi, free + CR.bucket_min_detour(hi, 0.3, by)]
        admissible = [g for g in gs if CR.detour_failure(g, free, hi, 0.3, by) is None]
        if not ok:
            assert admissible == []  # no blocked path can pass: the goal is skipped without losing an instance
        else:
            assert CR.detour_failure(hi, free, hi, 0.3, by) is None  # and with headroom a blocked path at the bucket's top would pass
