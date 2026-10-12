"""Simulator-free tests of the v2 held-object and camera-relocalization task modules (spec §6.1–§6.3, §3.9, §3.13):
bucket arithmetic, goal-set construction (depth grid with |Δd| >= 0.10, angle range), waypoint counts, the fixed-hold
eef pose, equivalence expansion from symmetry tags through the customs / conventions helpers
(``pytest ispatialgym/tests/test_v2_held_camera.py``). Importing the task modules pulls in the OmniGibson package
(no simulator is launched)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.tasks import camera_relocalization as CR
from ispatialgym.sampling.v2.tasks import held_object as HO

# ----------------------------------------------------------------------------------------------------------------
# camera relocalization: buckets (§2.3 / §6.1)
# ----------------------------------------------------------------------------------------------------------------

BUCKETS = [{"max_m": 3, "n": 10}, {"max_m": 5, "n": 10}, {"max_m": 10, "n": 5}]


def test_bucket_bounds_are_half_open():
    b = CR.normalise_buckets(BUCKETS)
    assert CR.bucket_bounds(b) == [(0.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
    assert CR.bucket_index(3.0, b) == 0  # (0, 3] includes 3
    assert CR.bucket_index(3.0001, b) == 1
    assert CR.bucket_index(5.0, b) == 1
    assert CR.bucket_index(10.0, b) == 2
    assert CR.bucket_index(10.5, b) == -1
    assert CR.bucket_index(0.0, b) == -1
    assert CR.bucket_of(4.2, b) == {"max_m": 5.0, "n": 10}
    assert CR.bucket_of(12.0, b) is None


def test_buckets_are_sorted_and_validated():
    b = CR.normalise_buckets([{"max_m": 10, "n": 5}, {"max_m": 3, "n": 10}])
    assert [x["max_m"] for x in b] == [3.0, 10.0]
    with pytest.raises(ValueError):
        CR.normalise_buckets([{"max_m": 3, "n": 1}, {"max_m": 3, "n": 1}])
    with pytest.raises(ValueError):
        CR.normalise_buckets([])
    with pytest.raises(ValueError):
        CR.normalise_buckets([{"max_m": 3, "n": -1}])


def test_bucket_deficits_and_choice():
    b = CR.normalise_buckets(BUCKETS)
    counts = {3.0: 10, 5.0: 4, 10.0: 5}
    assert CR.bucket_deficits(counts, b) == [0, 6, 0]
    rng = np.random.default_rng(0)
    # only the (3, 5] bucket has a deficit: it is always chosen
    assert all(CR.choose_bucket(rng, b, counts) == 1 for _ in range(20))
    # a requested bucket wins
    assert CR.choose_bucket(rng, b, counts, requested_max_m=10) == 2
    with pytest.raises(ValueError):
        CR.choose_bucket(rng, b, counts, requested_max_m=4)
    # all full: uniform over all buckets
    full = {3.0: 10, 5.0: 10, 10.0: 5}
    seen = {CR.choose_bucket(rng, b, full) for _ in range(200)}
    assert seen == {0, 1, 2}


def test_detour_failure_rules():
    assert CR.detour_failure(None, 2.0, 5.0, 0.3) == "disconnected"
    assert CR.detour_failure(2.2, 2.0, 5.0, 0.3) == "short_detour"
    assert CR.detour_failure(5.5, 2.0, 5.0, 0.3) == "too_long"
    assert CR.detour_failure(2.4, 2.0, 5.0, 0.3) is None


def test_path_yaws_follow_the_next_segment():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)]
    yaws = CR.path_yaws(pts, final_yaw=0.7)
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[1] == pytest.approx(math.pi / 2)
    assert yaws[2] == pytest.approx(math.pi / 2)  # zero-length step keeps the previous heading
    assert yaws[3] == pytest.approx(0.7)


def test_path_cells_middle_of_the_path():
    rng = np.random.default_rng(1)
    idx = CR.path_cells(rng, 11, 8)
    assert set(idx) <= set(range(3, 8))
    assert CR.path_cells(rng, 2, 8) == []
    assert set(CR.path_cells(rng, 3, 8)) == {1}


def _straight(n, step=0.1, y=0.0):
    return [(i * step, y) for i in range(n)]


def test_divergence_index_common_prefix():
    before = _straight(10)
    # the detour leaves the straight path after waypoint 4 (shares 0..4, then swerves to y = 0.1)
    after = before[:5] + [(0.5, 0.1), (0.6, 0.1), (0.7, 0.1), (0.8, 0.0), (0.9, 0.0)]
    assert CR.divergence_index(before, after, tol=0.05) == 4
    # identical paths share everything: the divergence is the last common waypoint
    assert CR.divergence_index(before, list(before), tol=0.05) == 9
    # paths that part immediately
    assert CR.divergence_index(before, [(0.0, 0.0), (0.1, 0.3)], tol=0.05) == 0
    # the tolerance is half a map cell: a 4 cm wobble still counts as the same waypoint
    wobble = [(x, y + (0.04 if i == 2 else 0.0)) for i, (x, y) in enumerate(before)]
    assert CR.divergence_index(before, wobble, tol=0.05) == 9
    # a shorter path cannot share more than its own length
    assert CR.divergence_index(before, before[:3], tol=0.05) == 2
    # torch paths (what the planner returns) work too
    assert CR.divergence_index(th.tensor(before), th.tensor(after), tol=0.05) == 4


def test_view_candidates_restricted_to_before_divergence_with_unobstructed_heading():
    before = _straight(10)
    after = before[:5] + [(0.5, 0.1), (0.6, 0.2), (0.7, 0.2), (0.8, 0.1), (0.9, 0.0)]
    idx, pts, yaws, k = CR.view_candidates(after, final_yaw=1.0, path_before=before, tol=0.05)
    assert k == 4
    assert idx == [0, 1, 2, 3, 4]
    assert pts == after[:5]
    # headings along the *unobstructed* path: straight ahead everywhere, also at the divergence waypoint (toward where
    # the unobstructed path would have continued), never the detour's or the goal's yaw
    assert yaws == pytest.approx([0.0] * 5)
    # without a reference path every waypoint counts, with the blocked path's own headings and the goal yaw last
    idx, pts, yaws, k = CR.view_candidates(after, final_yaw=1.0)
    assert k is None and idx == list(range(10)) and yaws[-1] == pytest.approx(1.0)
    assert yaws[4] == pytest.approx(math.atan2(0.1, 0.1))  # the detour heading at the divergence waypoint (not admissible as a view heading)


def test_obstacle_seen_only_after_the_detour_is_rejected():
    """An obstacle that enters the frame only at the last waypoint (the base turns to the goal yaw and looks back) is
    not admissible; one seen on the shared prefix is (its first admissible waypoint is recorded)."""
    divergence = 4
    assert CR.first_admissible_view([9], divergence) is None  # seen only after the detour
    assert CR.first_admissible_view([6, 7, 8], divergence) is None  # seen only along the detour
    assert CR.first_admissible_view([], divergence) is None
    assert CR.first_admissible_view([4], divergence) == 4  # the divergence waypoint itself counts
    assert CR.first_admissible_view([9, 3, 7], divergence) == 3  # the first shared waypoint at which it is in frame
    assert CR.first_admissible_view([9, 3], None) == 3  # no reference path: every waypoint counts


def test_obstructed_path_record_carries_view_facts():
    from ispatialgym.sampling.v2.stances import PathRecord

    base = PathRecord("init", "goal", "global", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.3)], 2.0, posture="tucked", carried=None, trunk=(0.5, 0.0))
    view = {"view_waypoint_index": {"o": 1}, "divergence_waypoint_index": {"o": 1}, "view_pose2d": {"o": [1.0, 0.0, 0.0]}}
    rec = CR.ObstructedPathRecord(base, view)
    d = rec.to_sidecar()
    assert PathRecord.schema_ok(d)
    assert d["view_waypoint_index"] == {"o": 1} and d["divergence_waypoint_index"] == {"o": 1} and d["view_pose2d"]["o"] == [1.0, 0.0, 0.0]
    assert d["waypoints"][-1] == [2.0, 0.0, 0.3] and d["length_m"] == 2.0
    # the F6 sidecar dataclass keeps the extra facts
    from ispatialgym.oracle import sidecar as SC

    p = SC.Path.from_dict(d, "paths[0]")
    assert p.extra["divergence_waypoint_index"] == {"o": 1} and p.goal == [2.0, 0.0, 0.3]


# ----------------------------------------------------------------------------------------------------------------
# camera relocalization: the recorded path is the driven schedule (owner 2026-09-22)
# ----------------------------------------------------------------------------------------------------------------


class _FakeTrav:
    """A 0.1 m grid whose world frame is the map frame (cell (i, j) at (0.1 i, 0.1 j)); erosion is the identity."""

    def __init__(self, n: int = 60, blocked=()):
        self.map_resolution = 0.1
        m = th.full((n, n), 255, dtype=th.uint8)
        for i0, i1, j0, j1 in blocked:
            m[i0:i1, j0:j1] = 0
        self.floor_map = [m]

    def _erode_trav_map(self, m, robot=None):
        return m

    def world_to_map(self, xy):
        return th.tensor([int(round(float(xy[0]) / self.map_resolution)), int(round(float(xy[1]) / self.map_resolution))])

    def map_to_world(self, ij):
        return th.tensor([float(ij[0]) * self.map_resolution, float(ij[1]) * self.map_resolution])


def _fake_camera_sampler(trav: _FakeTrav, fov_deg: float = 80.0, reach_m: float = 4.0):
    """A ``CameraRelocalizationV2`` without a simulator: the kit puts the camera at the base pose, the frustum test is a
    planar bearing / range test on the obstacle box's centre."""
    from types import SimpleNamespace

    world = SimpleNamespace(scene=SimpleNamespace(trav_map=trav))
    kit = SimpleNamespace(world=world, robot=None, q_at_base_pose=lambda q, x, y, yaw: th.tensor([x, y, yaw]), camera_poses_batch=lambda qs: [tuple(float(v) for v in q) for q in qs])
    cr = object.__new__(CR.CameraRelocalizationV2)
    cr.kit, cr.world = kit, world
    cr.q_reset_rel = th.zeros(3)
    cr.gp = SimpleNamespace(alias="camera_relocalization-test")
    cr.g = lambda key, default=None: {"init_pose": {"trunk": 0.5, "arms": "untucked"}}[key]

    def frustum_ok(cam, corners, margin):
        cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())
        bearing = math.atan2(cy - cam[1], cx - cam[0]) - cam[2]
        return abs(C.wrap_angle(bearing)) <= math.radians(fov_deg / 2) and math.hypot(cx - cam[0], cy - cam[1]) <= reach_m

    cr.frustum_ok = frustum_ok
    cr.obstacle_aabb = lambda name, pose: (pose[0] - 0.15, pose[0] + 0.15)
    return cr


def _dist_to_segment(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _grid_path(trav: _FakeTrav, a, b) -> th.Tensor:
    """BEHAVIOR-style dense cell path between two world points on the fake map (8-connected A*)."""
    from ispatialgym.sampling.v2 import navigation as NAV

    free = trav.floor_map[0] == 255
    cells = NAV.grid_shortest_path(free, tuple(int(v) for v in trav.world_to_map(a).tolist()), tuple(int(v) for v in trav.world_to_map(b).tolist()))
    assert cells is not None
    return th.stack([trav.map_to_world(th.tensor(c)) for c in cells])


def test_camera_relocalization_records_the_shortcut_path_and_checks_the_driven_schedule(monkeypatch):
    from ispatialgym.robot import base_path as BP
    from ispatialgym.sampling.v2 import navigation as NAV

    # a wall stump between init and goal: the grid path is a staircase around it
    trav = _FakeTrav(blocked=[(20, 23, 0, 12)])
    cr = _fake_camera_sampler(trav)
    base0, base1 = (0.5, 0.5, math.pi), (4.5, 0.5, 0.0)  # the robot starts facing away from its path
    path_b = _grid_path(trav, base0[:2], base1[:2])
    assert len(path_b) >= 30  # one waypoint per cell
    waypoints, sched = cr.path_schedule(path_b, base0, base1)
    # the record is the line-of-sight shortcut: a handful of corners, the exact base poses at the ends, headings inside
    assert 3 <= len(waypoints) <= 8
    assert waypoints[0] == pytest.approx(base0) and waypoints[-1] == pytest.approx(base1)
    for (x0, y0, yaw), (x1, y1, _) in zip(waypoints[1:-1], waypoints[2:]):
        assert yaw == pytest.approx(math.atan2(y1 - y0, x1 - x0))
    # the recorded polyline never goes through the stump (line of sight on the map it was planned on: at most a corner
    # clip of one cell, which the real eroded map's >= 0.53 m margin absorbs) and the drive stays within the fillet budget
    # of the polyline; the whole-body check below sees the driven poses themselves
    cfg = BP.BasePathConfig()
    free_xy = th.stack([trav.map_to_world(ij) for ij in th.nonzero(trav.floor_map[0] == 255)])
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for f in np.linspace(0.0, 1.0, 50):
            p = th.tensor([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f])
            assert float(th.linalg.norm(free_xy - p, dim=1).min()) <= trav.map_resolution * math.sqrt(0.5) + 0.03, p.tolist()
    for x, y, _ in sched.stride_poses(0.05):
        d = min(_dist_to_segment((x, y), a, b) for a, b in zip(waypoints[:-1], waypoints[1:]))
        assert d <= cfg.fillet_deviation_m + 1e-6, (x, y, d)
    # facing away from the path: the schedule turns on the spot first (the hybrid policy)
    assert [d.kind for d in sched.dwells if d.kind == "start"] == ["start"]
    # the record's whole-body check sees the schedule: the start sweep at the init cell, then the drive every 0.1 m
    seen = {}

    def fake_check(kit, poses, trunk, posture, **kw):
        seen["poses"] = list(poses)
        return True, -1

    monkeypatch.setattr(NAV, "path_collision_free", fake_check)
    rec = cr.build_path_record(path_b, base0, base1, (0.5, 0.0), plan=(waypoints, sched))
    assert rec is not None and rec.style == BP.FACE_TRAVEL and rec.mode == "global"
    assert rec.waypoints == waypoints and rec.length_m == pytest.approx(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(waypoints[:-1], waypoints[1:])))
    poses = seen["poses"]
    assert rec.n_checked == len(poses) == len(sched.stride_poses(NAV.STRIDE_M))
    at_init = [p for p in poses if math.hypot(p[0] - base0[0], p[1] - base0[1]) < 1e-6]
    assert len(at_init) >= 8  # the on-the-spot turn is checked in <= 0.2 rad steps
    assert rec.to_sidecar()["style"] == BP.FACE_TRAVEL
    # a record re-read from its waypoints rebuilds the same schedule (the oracle drives exactly what was checked)
    again = BP.BasePath.build(rec.waypoints, BP.BasePathConfig())
    assert again.length == pytest.approx(sched.length) and len(again.pieces) == len(sched.pieces)
    assert [d.kind for d in again.dwells] == [d.kind for d in sched.dwells]


def test_camera_relocalization_starts_untucked_and_navigates_in_the_init_posture(monkeypatch):
    """Owner 2026-10-11: the init arms - also the navigation posture of the recorded path and the goal's arms when they
    stay as at init - are untucked; an entry grounded before keeps its JoyLo reset arms (``init_pose.arms = zero``)."""
    from ispatialgym.sampling import generation_defaults as GD
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import navigation as NAV

    for subtask in (CR.SUBTASK_STANDARD, CR.SUBTASK_OBSTRUCTED):
        assert GD.defaults_for(subtask)["init_pose"] == {"trunk": 0.5, "arms": "untucked"}
    trav = _FakeTrav()
    cr = _fake_camera_sampler(trav)
    base0, base1 = (0.5, 0.5, 0.0), (4.5, 0.5, 0.0)
    checked = []
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture, **kw: checked.append(posture) or (True, -1))
    for arms, mode, posture, init_arms in (("untucked", "untucked", "untucked", "untucked"), ("zero", "tucked", "JoyLo reset (arms zero)", [])):
        cr.g = lambda key, default=None, arms=arms: {"init_pose": {"trunk": 0.5, "arms": arms}}[key]
        assert cr.init_arms == mode and cr.posture_name() == posture
        assert cr.stance_arms() == init_arms and cr.stance_arms("same") == init_arms and cr.stance_arms("untucked") == "untucked"
        rec = cr.build_path_record(_grid_path(trav, base0[:2], base1[:2]), base0, base1, (0.5, 0.0))
        assert rec.posture == mode and checked[-1] == mode
    cr.g = lambda key, default=None: {"init_pose": {"trunk": 0.5, "arms": "ready"}}[key]
    with pytest.raises(Reject):
        cr.init_arms


def test_camera_relocalization_view_rule_on_the_driven_schedule():
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import Reject

    trav_free = _FakeTrav()
    base0, base1 = (0.5, 0.5, 0.0), (4.5, 0.5, 0.0)
    path_free = _grid_path(trav_free, base0[:2], base1[:2])  # the straight shortest path without the obstacle
    # the obstacle stands on that path at x = 2.1; the blocked map forces the detour above it
    trav_b = _FakeTrav(blocked=[(19, 24, 3, 8)])
    cr = _fake_camera_sampler(trav_b)
    path_b = _grid_path(trav_b, base0[:2], base1[:2])
    k = CR.divergence_index(path_free, path_b, cr.map_tol())
    _, sched = cr.path_schedule(path_b, base0, base1)
    kk, s_div = cr.divergence_s(sched, path_b, path_free)
    assert kk == k and 0.0 <= s_div <= sched.length
    obstacle = (th.tensor([2.1, 0.5, 0.2]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    corners = G.aabb_corners(*cr.obstacle_aabb("obs", obstacle))
    hits = cr.view_indices(sched, path_b, corners, 0.05, path_before=path_free)
    ss, drive = sched.drive_poses(0.1)
    assert hits and all(ss[i] <= s_div + 1e-9 for i in hits)
    # the pose the fact is recorded at is a driven pose: on the shared prefix, facing the way the robot drives
    view = cr.view_record({"obs": obstacle}, path_b, sched, 0.05, {"obs": path_free})
    first = view["view_waypoint_index"]["obs"]
    assert first == hits[0] and view["view_pose2d"]["obs"] == pytest.approx(list(drive[first]))
    assert view["view_s_m"]["obs"] <= view["divergence_s_m"]["obs"] and view["divergence_waypoint_index"]["obs"] == k
    sc = cr.sidecar_view(view)
    assert sc["view_heading"] == "driven schedule" and set(sc) >= {"view_waypoint_index", "view_s_m", "divergence_waypoint_index", "divergence_s_m", "view_pose2d"}
    # an obstacle that only comes into the frame after the departure (behind and above the detour) is not admissible
    behind = (th.tensor([0.2, 3.5, 0.2]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    with pytest.raises(Reject):
        cr.view_record({"obs": behind}, path_b, sched, 0.05, {"obs": path_free})


def test_structural_categories_never_count():
    for c in ("walls", "floors", "ceilings", "window", "door", "downlight", "room_light", "table_lamp", "curtain"):
        assert CR.is_structural(c)
    for c in ("sofa", "coffee_table", "pot_plant", "footstool"):
        assert not CR.is_structural(c)
    assert CR.is_structural(None)


def test_heading_between_axes():
    assert CR.heading_between([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert CR.heading_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(math.pi / 2)


# ----------------------------------------------------------------------------------------------------------------
# held object: goal sets (§6.3)
# ----------------------------------------------------------------------------------------------------------------


def test_depth_grid_is_snapped_and_inclusive():
    grid = HO.depth_grid([0.15, 0.55], 0.01)
    assert len(grid) == 41 and grid[0] == 0.15 and grid[-1] == 0.55 and grid[7] == 0.22


def test_depth_window_takes_the_run_through_the_reference_and_its_margin():
    grid = HO.depth_grid([0.15, 0.55], 0.01)
    # the measured shape: IK fails below 0.21 and above 0.52
    passed = [0.21 - 1e-9 <= d <= 0.52 + 1e-9 for d in grid]
    window, run = HO.depth_window(grid, passed, 0.501, 0.02)
    assert run == (0.21, 0.52) and window == (0.23, 0.50)
    # a hole splits the runs: the run through the reference depth wins over a longer one elsewhere
    holed = [p and not (0.40 - 1e-9 <= d <= 0.41 + 1e-9) for d, p in zip(grid, passed)]
    window, run = HO.depth_window(grid, holed, 0.501, 0.02)
    assert run == (0.42, 0.52) and window == (0.44, 0.50)
    # the reference depth in no run: the longest run
    window, run = HO.depth_window(grid, holed, 0.60, 0.02)
    assert run == (0.21, 0.39) and window == (0.23, 0.37)
    # a run shorter than the two margins leaves no window; no passing depth leaves nothing
    short = [0.30 - 1e-9 <= d <= 0.33 + 1e-9 for d in grid]
    assert HO.depth_window(grid, short, 0.31, 0.02) == (None, (0.30, 0.33))
    assert HO.depth_window(grid, [False] * len(grid), 0.5, 0.02) == (None, None)


def test_draw_start_goal_moves_both_ways_inside_the_window():
    rng = np.random.default_rng(0)
    draws = [HO.draw_start_goal(rng, (0.23, 0.49), 0.05) for _ in range(2000)]
    assert all(0.23 <= d0 <= 0.49 and 0.23 <= d1 <= 0.49 for d0, d1 in draws)
    assert all(abs(d1 - d0) >= 0.05 for d0, d1 in draws)
    away = sum(d1 > d0 for d0, d1 in draws)
    assert 0.4 < away / len(draws) < 0.6  # toward and away alike
    starts = sorted(d0 for d0, _ in draws)
    assert starts[0] < 0.24 and starts[-1] > 0.48  # the whole window is used
    with pytest.raises(ValueError):
        HO.draw_start_goal(rng, (0.30, 0.34), 0.05)


def test_draw_start_goal_honours_the_requested_direction_and_start_band():
    """owner 2026-10-11 (eval): customs requested_direction / start_depth_band; without them the draw is unchanged."""
    w, m = (0.23, 0.49), 0.05
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    assert [HO.draw_start_goal(a, w, m) for _ in range(50)] == [HO.draw_start_goal(b, w, m, direction=None, start_band=None) for _ in range(50)]
    rng = np.random.default_rng(0)
    for direction, (s_lo, s_hi) in (("toward", (0.28, 0.49)), ("away", (0.23, 0.44))):  # the starts that admit the move
        for i in range(5):
            band = (i / 5, (i + 1) / 5)
            for d0, d1 in (HO.draw_start_goal(rng, w, m, direction=direction, start_band=band) for _ in range(200)):
                assert 0.23 <= d1 <= 0.49 and abs(d1 - d0) >= m - 1e-12 and ((d1 < d0) if direction == "toward" else (d1 > d0))
                assert s_lo + band[0] * (s_hi - s_lo) - 1e-12 <= d0 <= s_lo + band[1] * (s_hi - s_lo) + 1e-12
    # the five bands of a direction spread the starts over the window
    starts = sorted(HO.draw_start_goal(rng, w, m, direction="away", start_band=(i / 5, (i + 1) / 5))[0] for i in range(5) for _ in range(50))
    assert starts[0] < 0.25 and starts[-1] > 0.42
    # a direction alone: the whole start range; a band alone: both directions
    assert all(d1 < d0 for d0, d1 in (HO.draw_start_goal(rng, w, m, direction="toward") for _ in range(100)))
    both = [HO.draw_start_goal(rng, w, m, start_band=(0.4, 0.6)) for _ in range(400)]
    assert all(abs(d1 - d0) >= m for d0, d1 in both) and 0.3 < sum(d1 > d0 for d0, d1 in both) / len(both) < 0.7
    with pytest.raises(ValueError):
        HO.draw_start_goal(rng, w, m, direction="sideways")
    with pytest.raises(ValueError):
        HO.draw_start_goal(rng, w, m, direction="away", start_band=(0.6, 0.4))


def test_slide_to_depth_puts_the_object_root_at_the_depth():
    """The start / goal hand pose at any depth: the object root lands at the depth along the optical axis, its lateral
    offset from the axis and the hand orientation are kept, and the hand-object transform is unchanged."""
    cam_pos = th.tensor([0.24, 0.0, 1.39])
    axis = th.tensor([math.cos(math.radians(22.6)), 0.0, -math.sin(math.radians(22.6))])
    eef = (th.tensor([0.70, 0.03, 1.20]), C.quat_from_axis_angle([0.0, 0.0, 1.0], 0.4))
    eef_in_object = (th.tensor([0.01, -0.02, 0.03]), C.quat_from_axis_angle([1.0, 0.0, 0.0], 1.2))
    obj_ref = C.compose(eef, C.invert(eef_in_object))
    lateral_ref = (obj_ref[0] - cam_pos) - axis * float(th.dot(obj_ref[0] - cam_pos, axis))
    for depth in (0.21, 0.35, 0.52):
        (e_p, e_q), (o_p, o_q) = HO.slide_to_depth(eef, eef_in_object, cam_pos, axis, depth)
        assert float(th.dot(o_p - cam_pos, axis)) == pytest.approx(depth, abs=1e-6)
        lateral = (o_p - cam_pos) - axis * float(th.dot(o_p - cam_pos, axis))
        assert th.allclose(lateral, lateral_ref, atol=1e-6)
        assert C.quat_angle(e_q, eef[1]) == pytest.approx(0.0, abs=1e-6) and C.quat_angle(o_q, obj_ref[1]) == pytest.approx(0.0, abs=1e-6)
        rel = C.relative((e_p, e_q), (o_p, o_q))
        assert th.allclose(rel[0], eef_in_object[0], atol=1e-5) and C.quat_angle(rel[1], eef_in_object[1]) == pytest.approx(0.0, abs=1e-5)


def test_rotation_axes_one_per_antipodal_pair():
    axes = HO.rotation_axes()
    assert axes.shape == (13, 3)
    assert th.allclose(th.linalg.norm(axes, dim=1), th.ones(13), atol=1e-6)
    for i in range(13):
        for j in range(13):
            if i != j:
                assert float(th.abs(th.dot(axes[i], axes[j]))) < 1.0 - 1e-6, "two axes are (anti)parallel"


def test_rotation_goal_set_angle_range():
    """The 15° grid of ``scripts/held_comfort`` (the sampler draws its starts and goals continuously)."""
    goals = HO.rotation_goal_set([30.0, 120.0], 15.0)
    mags = sorted({abs(a) for _, a in goals})
    assert mags == [30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
    assert len(goals) == 13 * 7 * 2
    assert all(30.0 <= abs(a) <= 120.0 for _, a in goals)
    assert any(a < 0 for _, a in goals) and any(a > 0 for _, a in goals)
    with pytest.raises(ValueError):
        HO.rotation_goal_set([0.0, 120.0])


def _assert_uniform_on_the_sphere(axes: th.Tensor):
    """Unit axes spread uniformly over the sphere: each coordinate ~ U(−1, 1) (Archimedes), no two alike, and no more of
    them within 5° of a cube face / edge / corner direction than that area holds (draws among the 13 axes: all of them)."""
    a = axes.double().numpy()
    n = len(a)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6) and len({tuple(r) for r in a.round(6).tolist()}) == n
    for k in range(3):
        hist, _ = np.histogram(a[:, k], bins=8, range=(-1.0, 1.0))
        assert hist.min() > 0.8 * n / 8 and hist.max() < 1.2 * n / 8
    near = (np.abs(a @ HO.rotation_axes().double().numpy().T).max(axis=1) > math.cos(math.radians(5.0))).mean()
    assert near < 2 * 26 * (1.0 - math.cos(math.radians(5.0))) / 2  # twice the 26 caps' share of the sphere (~5 %)


def test_draw_rotation_goals_draws_uniform_axes_and_continuous_signed_angles():
    """Owner 2026-10-11 (fully continuous): every goal candidate an axis uniform on the sphere - not one of the 13 cube
    axes -, |angle| ~ U(goal_angle_range_deg) off the 15° grid and a sign + / − alike; the same seed draws the same
    candidates."""
    n = 4000
    goals = HO.draw_rotation_goals(np.random.default_rng(0), [30.0, 120.0], n)
    mags = np.array([abs(a) for _, a in goals])
    assert len(goals) == n and mags.min() >= 30.0 and mags.max() <= 120.0
    off_grid = np.minimum(np.remainder(mags, 15.0), 15.0 - np.remainder(mags, 15.0))
    assert (off_grid > 1e-6).all() and len(set(mags.tolist())) == n
    hist, _ = np.histogram(mags, bins=9, range=(30.0, 120.0))
    assert hist.min() > 0.8 * n / 9 and hist.max() < 1.2 * n / 9  # uniform in the range
    assert 0.45 < np.mean([a > 0 for _, a in goals]) < 0.55
    _assert_uniform_on_the_sphere(th.stack([ax for ax, _ in goals]))
    a, b = (HO.draw_rotation_goals(np.random.default_rng(7), [30.0, 120.0], 16) for _ in range(2))
    c = HO.draw_rotation_goals(np.random.default_rng(8), [30.0, 120.0], 16)
    assert [(x.tolist(), d) for x, d in a] == [(x.tolist(), d) for x, d in b] and [d for _, d in a] != [d for _, d in c]
    # a narrower range is honoured; a range touching 0 is not a goal range
    assert all(45.0 <= abs(d) <= 50.0 for _, d in HO.draw_rotation_goals(np.random.default_rng(1), [45.0, 50.0], 200))
    with pytest.raises(ValueError):
        HO.draw_rotation_goals(np.random.default_rng(0), [0.0, 120.0], 4)


def test_depth_tolerance_floor_and_fraction():
    tol = {"depth_min_m": 0.02, "depth_fraction": 0.20}
    assert HO.depth_tolerance(0.05, tol) == pytest.approx(0.02)  # floor
    assert HO.depth_tolerance(0.30, tol) == pytest.approx(0.06)
    assert HO.depth_tolerance(-0.30, tol) == pytest.approx(0.06)


def test_visible_fraction():
    assert HO.visible_fraction(40, 100) == pytest.approx(0.4)
    assert HO.visible_fraction(0, 0) == 0.0
    assert HO.visible_fraction(120, 100) == 1.0


# ----------------------------------------------------------------------------------------------------------------
# waypoints (§3.9): 3 intermediate + the goal
# ----------------------------------------------------------------------------------------------------------------


def _pose(x, y, z, yaw=0.0):
    return th.tensor([x, y, z]), C.quat_from_yaw(yaw)


def test_line_key_poses_count_and_ends():
    p0, p1 = _pose(0.0, 0.0, 1.0), _pose(0.4, 0.0, 1.0, yaw=0.0)
    wps = HO.line_key_poses(p0, p1, 3)
    assert len(wps) == 4  # 3 intermediate + goal
    assert th.allclose(wps[-1][0], p1[0], atol=1e-6)
    xs = [float(p[0][0]) for p in wps]
    assert xs == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert all(not th.allclose(p[0], p0[0]) for p in wps), "the start pose is not repeated"


def test_orbit_key_poses_rotate_about_the_pivot():
    eef0 = _pose(0.2, 0.0, 1.0)
    pivot = th.tensor([0.0, 0.0, 1.0])
    dq = C.quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(90.0))
    wps = HO.orbit_key_poses(eef0, pivot, dq, 3)
    assert len(wps) == 4
    for p, _ in wps:
        assert float(th.linalg.norm(p - pivot)) == pytest.approx(0.2, abs=1e-5)  # rigid about the pivot
    assert th.allclose(wps[-1][0], th.tensor([0.0, 0.2, 1.0]), atol=1e-5)
    assert C.quat_angle(wps[-1][1], C.qmul(dq, eef0[1])) == pytest.approx(0.0, abs=1e-5)
    # the intermediate rotations are 22.5°, 45°, 67.5° about the pivot (SLERP)
    angles = [math.degrees(C.quat_angle(q, eef0[1])) for _, q in wps]
    assert angles == pytest.approx([22.5, 45.0, 67.5, 90.0], abs=1e-3)


def test_generation_waypoints_default_is_three():
    from ispatialgym.sampling import generation_defaults as GD

    for st, roll in (("object_scaling", 90.0), ("mental_rotation", 15.0)):
        d = GD.defaults_for(st)
        assert d["waypoints"] == 3
        assert d["hold"] == {"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": roll}
        assert d["min_visible_fraction"] == pytest.approx(0.40)
    osc = GD.defaults_for("object_scaling")
    assert (osc["depth_search_range_m"], osc["depth_search_grid_m"], osc["depth_margin_m"], osc["min_depth_change_m"]) == ([0.15, 0.55], 0.01, 0.02, 0.05)
    assert osc["requested_direction"] is None and osc["start_depth_band"] is None  # eval aliases set them in the customs
    assert "goal_depth_range_m" not in osc and "goal_depth_grid_m" not in osc
    mr = GD.defaults_for("mental_rotation")
    assert "depth_search_range_m" not in mr and "goal_draws_per_start" not in osc
    assert (mr["goal_angle_range_deg"], mr["goal_draws_per_start"], mr["init_angle_range_deg"]) == ([30.0, 120.0], 192, [0.0, 180.0])
    assert mr["init_draws_per_template"] == 312 and "init_draws_per_template" not in osc
    assert mr["goal_draws_per_start"] >= 2 * HO.MentalRotationV2.GOAL_SCREEN_MAX


# ----------------------------------------------------------------------------------------------------------------
# the fixed hold (§6.3)
# ----------------------------------------------------------------------------------------------------------------


def test_hold_constants_from_generation():
    hc = HO.HoldConstants.from_generation({"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": 15.0})
    assert (hc.depth_m, hc.translate, hc.tilt, hc.roll_deg) == (0.5, 0.5, 0.0, 15.0)
    with pytest.raises(ValueError):
        HO.HoldConstants.from_generation(None)
    k1 = HO.hold_cache_key("r1pro", hc)
    k2 = HO.hold_cache_key("r1pro", HO.HoldConstants(0.55, 0.5, 0.0, 15.0))
    assert k1 != k2 and len(k1) == 16


def test_hold_eef_pose_geometry():
    """Pad centre on the optical axis at the depth, approach = base +y, closing = world down rolled 15° about it; the
    v1 runs' recorded hold quaternion (runlogs/isg_reps_20260915 template metas) is reproduced."""
    hc = HO.HoldConstants()
    yaw = 0.3
    # a camera 1.2 m up, 0.3 m ahead of the base, looking along base +x (pitched slightly down)
    base_q = C.quat_from_yaw(yaw)
    cam_quat = C.qmul(base_q, C.quat_from_euler(0.0, math.radians(-10.0), 0.0))
    # the USD camera looks along its -z: build a frame whose -z is the viewing direction
    look = C.quat_rotate(base_q, th.tensor([math.cos(math.radians(10.0)), 0.0, -math.sin(math.radians(10.0))]))
    cam_quat = C.quat_with_z_axis(-look, cam_quat)
    cam_pos = C.quat_rotate(base_q, th.tensor([0.3, 0.0, 1.2]))
    (origin, quat), pad = HO.hold_eef_pose((cam_pos, cam_quat), yaw, hc)
    axis = C.quat_rotate(cam_quat, th.tensor([0.0, 0.0, -1.0]))
    assert float(th.dot(pad - cam_pos, axis)) == pytest.approx(0.50, abs=1e-6)
    assert float(th.linalg.norm(th.cross(pad - cam_pos, axis, dim=0))) == pytest.approx(0.0, abs=1e-6)
    a = C.approach_axis(quat)
    assert th.allclose(a, C.quat_rotate(base_q, th.tensor([0.0, 1.0, 0.0])), atol=1e-6)
    assert th.allclose(C.pad_center((origin, quat)), pad, atol=1e-6)
    # closing axis: 15° from world down, perpendicular to the approach
    c = C.closing_axis(quat)
    assert math.degrees(math.acos(float(th.dot(c, th.tensor([0.0, 0.0, -1.0]))))) == pytest.approx(15.0, abs=1e-4)
    assert float(th.dot(c, a)) == pytest.approx(0.0, abs=1e-6)
    # in the base frame the v1 hold quaternion is recovered (yaw 0 base)
    (o0, q0), _ = HO.hold_eef_pose((th.tensor([0.3, 0.0, 1.2]), C.quat_with_z_axis(th.tensor([-1.0, 0.0, 0.0]), C.IDENTITY_QUAT)), 0.0, hc)
    v1 = th.tensor([-0.7010573744773865, 0.09229596704244614, 0.09229596704244614, 0.7010573744773865])
    assert C.quat_angle(q0, v1) == pytest.approx(0.0, abs=1e-3)
    assert float(o0[1]) == pytest.approx(-C.CONTACT_CENTER_Z, abs=1e-6)  # origin = pad − CONTACT_CENTER_Z·a


def test_object_scaling_hold_closes_the_fingers_horizontally():
    """Object scaling's roll 90°: the closing axis is horizontal and points back at the robot, so a cylinder pinched
    across its diameter (axis = approach × closing) stands upright and faces the camera with its side."""
    yaw = 0.7
    base_q = C.quat_from_yaw(yaw)
    cam = (C.quat_rotate(base_q, th.tensor([0.24, 0.0, 1.39])), C.quat_with_z_axis(C.quat_rotate(base_q, th.tensor([-1.0, 0.0, 0.0])), base_q))
    (_, quat), _ = HO.hold_eef_pose(cam, yaw, HO.HoldConstants(0.50, 0.5, 0.0, 90.0))
    a, c = C.approach_axis(quat), C.closing_axis(quat)
    assert th.allclose(c, C.quat_rotate(base_q, th.tensor([-1.0, 0.0, 0.0])), atol=1e-5)
    assert th.allclose(th.cross(a, c, dim=0), th.tensor([0.0, 0.0, 1.0]), atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# B7 prior
# ----------------------------------------------------------------------------------------------------------------


def test_visibility_prior_roundtrip(tmp_path):
    path = tmp_path / "visibility.json"
    path.write_text(json.dumps({"records": [
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_0", "visible_fraction": 0.83, "in_frame": True},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_1", "visible_fraction": 0.21, "in_frame": True},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_2", "visible_fraction": 0.9, "in_frame": False},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_3", "visible_fraction": None, "in_frame": None},
    ]}))
    prior = HO.load_visibility_prior(str(path))
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_0", 0.4) is True
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_1", 0.4) is False
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_2", 0.4) is False
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_3", 0.4) is None
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_9", 0.4) is None
    assert HO.load_visibility_prior(str(tmp_path / "missing.json")) == {}


def test_visibility_prior_applies_only_at_its_hold(tmp_path):
    path = tmp_path / "visibility.json"
    rec = {"grasp_key": "banana__znakxm", "grasp_name": "grasp_0", "visible_fraction": 0.2, "in_frame": True}
    path.write_text(json.dumps({"hold": {"depth_m": 0.5, "trunk": [0.5, 0.0], "roll_deg": 15.0}, "records": [rec]}))
    at_15 = HO.load_visibility_prior(str(path), HO.HoldConstants(0.50, 0.5, 0.0, 15.0))
    assert HO.prior_admits(at_15, "banana__znakxm", "grasp_0", 0.4) is False
    assert HO.load_visibility_prior(str(path), HO.HoldConstants(0.50, 0.5, 0.0, 90.0)) == {}
    assert HO.load_visibility_prior(str(path), HO.HoldConstants(0.55, 0.5, 0.0, 15.0)) == {}
    assert len(HO.load_visibility_prior(str(path))) == 1  # no hold given: the survey as is
    path.write_text(json.dumps({"records": [rec]}))
    assert HO.load_visibility_prior(str(path), HO.HoldConstants(0.50, 0.5, 0.0, 15.0)) == {}  # a survey without a hold record


# ----------------------------------------------------------------------------------------------------------------
# equivalences from the symmetry tags (§3.13) through the customs / problem helpers
# ----------------------------------------------------------------------------------------------------------------


def _held_entry(tags):
    return {
        "iddl": "mental_rotation/mental_rotation.iddl",
        "definition_id": 0,
        "roles": {"target": {"category": "toy_figure", "sampleable": True, "require": {"grasps": "floating"}, "representative": "zuntus",
                             "grasp_keys": {"toy_figure/zuntus": "toy_figure__zuntus"}, "orientation_equivalences": {"toy_figure/zuntus": tags}},
                  "floor": {"sampleable": False}},
        "scenes": {"house_single_floor": {"rooms": ["living_room_0"], "bindings": {"floor": {"room": "living_room_0"}},
                                          "whitelist": {"target": {"toy_figure": ["zuntus"]}}, "blacklist": {}}},
        "generation": {"templates_max": 1, "instances_per_template": 1, "tolerances": {"orientation": 0.15}, "groups": {}, "path_buckets": None},
        "constraints": [],
    }


def test_equivalence_expansion_from_tags():
    from ispatialgym.sampling import customs as CU
    from ispatialgym.sampling.v2.problem import GroundedProblem

    z2x2 = CU.equivalence_group(["z2", "x2"])
    assert len(z2x2) == 4 and z2x2[0] == [0.0, 0.0, 0.0, 1.0]
    assert len(CU.equivalence_group(["z4", "x2"])) == 8
    assert len(CU.equivalence_group(["zinf"])) == 72
    assert CU.equivalence_group(["full"]) is None
    assert CU.parse_symmetry_tags("none") == [] and CU.parse_symmetry_tags("zinf x2") == ["zinf", "x2"]

    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry(["z2"]))
    eq = gp.orientation_equivalences("target", "toy_figure", "zuntus")
    assert eq is not None and len(eq) == 2
    assert eq[1] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-6)  # the half turn about z
    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry(["none"]))
    assert gp.orientation_equivalences("target", "toy_figure", "zuntus") is None
    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry([]))
    assert gp.orientation_equivalences("target", "toy_figure", "zuntus") is None


def test_registry_has_the_three_subtasks():
    from ispatialgym.sampling.v2.core import REGISTRY

    assert REGISTRY["object_scaling"] is HO.ObjectScalingV2
    assert REGISTRY["mental_rotation"] is HO.MentalRotationV2
    assert REGISTRY[CR.SUBTASK_STANDARD] is CR.CameraRelocalizationV2
    assert REGISTRY[CR.SUBTASK_OBSTRUCTED] is CR.CameraRelocalizationV2


def test_both_held_object_subtasks_write_the_start_drive_targets(monkeypatch):
    """Mental rotation records ``robot_state.controller_targets`` like object scaling (owner 2026-09-26): the settle of
    the start and the recorded targets live in the shared held-object sampler, object scaling adds its start depths.
    Without recorded targets the TRO has no ``controller_targets`` and the loader holds the loaded joints as before."""
    from ispatialgym.sampling.v2.core import SamplerV2

    monkeypatch.setattr(SamplerV2, "tro_meta_block", lambda self, draft, instance_id, seed: {"robot_state": {"joint_positions": {}}})
    targets = {"torso_joint1": 1.025, "left_arm_joint1": 0.0, "right_arm_joint6": 0.4263}
    for cls in (HO.MentalRotationV2, HO.ObjectScalingV2):
        sampler = object.__new__(cls)
        sampler._controller_targets = {}
        assert "controller_targets" not in sampler.tro_meta_block(None, "iid", 0)["robot_state"]
        sampler._controller_targets = dict(targets)
        assert sampler.tro_meta_block(None, "iid", 0)["robot_state"]["controller_targets"] == targets
        assert cls.write_instance is HO.HeldObjectV2.write_instance and cls.settle_start is HO.HeldObjectV2.settle_start
    assert HO.MentalRotationV2.record_settled_start is HO.HeldObjectV2.record_settled_start
    assert HO.ObjectScalingV2.record_settled_start is not HO.HeldObjectV2.record_settled_start


# ----------------------------------------------------------------------------------------------------------------
# mental rotation: the drawn start (owner 2026-10-11)
# ----------------------------------------------------------------------------------------------------------------


def test_draw_rotation_starts_uniform_axes_and_angles_in_the_range():
    """Owner 2026-10-11 (fully continuous): each start an axis uniform on the sphere and an angle ~ U(init_angle_range_deg),
    unsigned - uniform in the angle, not the Haar density ∝ 1 − cos θ; the same seed draws the same starts."""
    n = 4000
    starts = HO.draw_rotation_starts(np.random.default_rng(0), [0.0, 180.0], n)
    angles = np.array([d for _, d in starts])
    assert len(starts) == n and angles.min() >= 0.0 and angles.max() <= 180.0 and len(set(angles.tolist())) == n
    hist, _ = np.histogram(angles, bins=9, range=(0.0, 180.0))
    assert hist.min() > 0.8 * n / 9 and hist.max() < 1.2 * n / 9
    assert (np.minimum(np.remainder(angles, 15.0), 15.0 - np.remainder(angles, 15.0)) > 1e-6).all()  # off the 15° grid
    _assert_uniform_on_the_sphere(th.stack([ax for ax, _ in starts]))
    a, b = (HO.draw_rotation_starts(np.random.default_rng(7), [0.0, 180.0], 16) for _ in range(2))
    c = HO.draw_rotation_starts(np.random.default_rng(8), [0.0, 180.0], 16)
    assert [(x.tolist(), d) for x, d in a] == [(x.tolist(), d) for x, d in b] and [d for _, d in a] != [d for _, d in c]
    assert all(20.0 <= d <= 40.0 for _, d in HO.draw_rotation_starts(np.random.default_rng(1), [20.0, 40.0], 200))
    with pytest.raises(ValueError):
        HO.draw_rotation_starts(np.random.default_rng(0), [-15.0, 90.0], 4)


def test_a_templates_start_set_has_no_twin_orientations():
    """The grid's ±180° pairs (13 orientations twice) are gone: no two of a template's 312 starts turn the hold alike (the
    closest two of a random set are a few tenths of a degree apart)."""
    starts = HO.draw_rotation_starts(HO.rotation_start_rng(0), [0.0, 180.0], 312)
    q = th.stack([C.quat_from_axis_angle(ax, math.radians(d)) for ax, d in starts]).double()
    dots = th.abs(q @ q.T) - 2.0 * th.eye(len(q), dtype=th.float64)  # |cos(half the angle between two starts)|
    assert float(2.0 * th.acos(th.clamp(dots.max(), max=1.0))) > math.radians(0.05)


def test_rotation_start_rng_is_the_template_seeds_own_stream():
    """The start set's RNG: one stream per template seed (the same seed, the same draws), never the stream of an attempt
    (``run_instances``: ``default_rng(base_seed * 1000003 + index * 1009 + attempt)``) or of the bare template seed."""
    first = lambda rng: rng.random(8).tolist()  # noqa: E731
    assert first(HO.rotation_start_rng(0)) == first(HO.rotation_start_rng(0)) != first(HO.rotation_start_rng(7919))
    for base_seed in (0, 1, 7919, 2**29 * 7919):
        start = first(HO.rotation_start_rng(base_seed))
        assert start != first(np.random.default_rng(base_seed))
        assert all(start != first(np.random.default_rng(base_seed * 1000003 + index * 1009 + attempt)) for index in range(3) for attempt in range(30))


def test_turned_about_root_keeps_the_position_and_turns_about_the_base_axis():
    obj = (th.tensor([1.0, 2.0, 0.8]), C.quat_from_yaw(0.3))
    p, q = HO.turned_about_root(obj, math.pi / 2, [1.0, 0.0, 0.0], 90.0)  # base yaw pi/2: base +x is world +y
    assert th.allclose(p, obj[0])
    assert C.quat_angle(q, C.qmul(C.quat_from_axis_angle(th.tensor([0.0, 1.0, 0.0]), math.pi / 2), obj[1])) < 1e-5


class _MRKit:
    """A planner without a simulator: every screening / exact IK solution is its seed; ``batch_ok`` decides the goal
    screening, ``start_ok`` the start set."""

    def __init__(self, start_ok=None, batch_ok=True):
        from ispatialgym.oracle import r1pro_model as M

        self.M = M
        self.floor_z = 0.0
        self.jidx = {n: i for i, n in enumerate(M.BASE_VIRTUAL_JOINTS)}
        self.start_ok = start_ok or (lambda pose: True)
        self.batch_ok = batch_ok

    def world_base_pose2d(self):
        return (0.0, 0.0, 0.0)

    def apply(self, q, base):
        pass

    def sync_world(self):
        pass

    def ignoring(self, objs):
        import contextlib

        return contextlib.nullcontext()

    def ik_batch_solve(self, poses, q_seed, arm="right", **kw):
        return th.tensor([bool(self.start_ok(p)) for p in poses]), q_seed.unsqueeze(0).repeat(len(poses), 1)

    def ik_batch(self, poses, q_seed, arm="right", **kw):
        return th.full((len(poses),), bool(self.batch_ok))

    def collision_free(self, q, margin=0.01):
        return th.ones(1, dtype=th.bool)

    def random_free_points(self, rng, n, room_filter=None):
        return [th.tensor([1.0, 2.0])]

    def q_at_base_pose(self, q_rel, x, y, yaw):
        q = q_rel.clone()
        names = self.M.BASE_VIRTUAL_JOINTS
        q[self.jidx[names[0]]], q[self.jidx[names[1]]], q[self.jidx[names[-1]]] = x, y, yaw
        return q


def _mr_sampler(monkeypatch, kit, base_seed=0):
    import contextlib
    from types import SimpleNamespace

    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.v2.stances import SidecarDraft

    s = object.__new__(HO.MentalRotationV2)
    # a customs entry seeded before goal_draws_per_start / init_draws_per_template existed: ``g`` takes them from
    # generation_defaults
    gen = {"init_angle_range_deg": [0.0, 180.0], "goal_angle_range_deg": [30.0, 120.0], "waypoints": 3, "tolerances": {"orientation": 0.15}}
    s.subtask, s.gen = "mental_rotation", (lambda key, default=None: gen.get(key, default))
    s.ctx = SimpleNamespace(base_seed=base_seed)  # the template seed (``ContextV2.base_seed``)
    s.kit, s.arm, s.hold, s.hold_arm, s.prior, s.grasp_stats, s.equivalences = kit, "right", HO.HoldConstants(), {}, {}, {}, None
    s.eef_in_base = (th.tensor([0.5, 0.0, 1.2]), C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), 0.4))
    s.cam_in_base = (th.tensor([0.1, 0.0, 1.5]), C.IDENTITY_QUAT.clone())
    s.q_hold_rel = th.linspace(0.0, 1.0, 28)
    target = SimpleNamespace(name="block_0", set_position_orientation=lambda p, q: None, keep_still=lambda: None)
    s.world = SimpleNamespace(park=lambda obj, i: None, spawned={"block_0": target})
    s.obj = lambda role: target
    s.bindings = SimpleNamespace(one=lambda role: target)
    s.grasps = [GL.Grasp("block__f", "grasp_0", th.tensor([0.0, 0.0, 0.05]), C.IDENTITY_QUAT.clone(), [0.01, 0.01])]
    s.grasp_key = "block__f"
    s.scene_obstacles = lambda: []
    s.whole_in_frame = lambda cam, pose: True
    s.in_room = lambda xy: True
    s.template_facts, s.template_meta = {}, {}
    s.gp = SimpleNamespace(alias="mental_rotation-test")
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=None, category="classroom_mock_up")})
    s.ikc = SimpleNamespace(waypoints_ok=lambda poses, stance, arm, q_seed=None, scale_key=None: SimpleNamespace(ok=True, q=q_seed.clone(), reason=""))
    s.robot = SimpleNamespace(arm_control_idx={"right": th.arange(10, 17)})
    s.sidecar = SidecarDraft("mental_rotation-test", "mental_rotation")
    s.arm_offer_record = lambda: {"fired": False}
    monkeypatch.setattr(HO, "attached", lambda kit, arm, obj: contextlib.nullcontext())
    return s


def test_mental_rotation_draws_a_start_and_turns_it_to_the_goal(monkeypatch):
    s = _mr_sampler(monkeypatch, _MRKit())
    s.precompute_options()
    opts = s.options["grasp_0"]
    meta = s.template_meta["init_orientation"]
    assert len(opts) == 312 and s.template_facts["start_options"]["feasible"] == len(opts)
    assert meta["drawn"] and meta["init_axis"] == meta["goal_axis"] == "uniform on the sphere" and meta["init_angle"] == meta["goal_angle"] == "continuous"
    assert (meta["init_draws_per_template"], meta["goal_draws_per_start"]) == (312, 192)
    assert meta["start_set_rng"] == {"template_seed": 0, "spawn_key": [HO.ROTATION_START_STREAM]}
    starts, angles = set(), []
    for seed in range(6):
        s.sidecar.reset_instance() if hasattr(s.sidecar, "reset_instance") else None
        d = s.sample_instance(np.random.default_rng(seed))
        info = d.summary["goal"]
        obj0, goal = d.object_poses["block_0"], d.goal_poses["block_0"]
        base2d = d.base_pose2d
        assert 30.0 <= abs(info["angle_deg"]) <= 120.0 and "start" in info and "axis_world" in info["start"]
        assert 0.0 <= info["start"]["angle_deg"] <= 180.0
        assert th.allclose(G_t(goal[0]), G_t(obj0[0]))  # the object turns in place
        want = HO.turned_about_root(obj0, base2d[2], info["axis_base"], info["angle_deg"])
        assert C.quat_angle(goal[1], want[1]) < 1e-5
        # every axis recorded as a unit vector (base and world frame); the base-frame axes are not the 13 cube axes
        for ax in (info["axis_base"], info["axis_world"], info["start"]["axis_base"], info["start"]["axis_world"]):
            assert len(ax) == 3 and float(th.linalg.norm(G_t(ax))) == pytest.approx(1.0, abs=1e-6)
        for ax in (info["axis_base"], info["start"]["axis_base"]):
            assert float(th.abs(HO.rotation_axes() @ G_t(ax)).max()) < 1.0 - 1e-6
        # the start is the hold's object turned by the recorded start rotation
        obj_hold = C.compose(C.compose(s.base_pose3d(base2d), s.eef_in_base), C.invert(s.grasps[0].pose))
        want0 = HO.turned_about_root(obj_hold, base2d[2], info["start"]["axis_base"], info["start"]["angle_deg"])
        assert th.allclose(G_t(obj0[0]), G_t(want0[0]), atol=1e-5) and C.quat_angle(obj0[1], want0[1]) < 1e-5
        opt = next(o for o in opts if o["info"] == {k: v for k, v in info["start"].items() if k != "axis_world"})
        assert th.allclose(d.robot_q, s.kit.q_at_base_pose(opt["q_rel"], *base2d))
        rec = s.sidecar.grasp_of("block_0")
        assert len(rec.waypoints) == 4 and th.allclose(G_t(rec.waypoints[-1][0]), G_t(C.compose(goal, s.grasps[0].pose)[0]), atol=1e-5)
        assert s.sidecar.language["goal_numbers"]["angle_deg"] == pytest.approx(info["angle_deg"])
        starts.add(info["start"]["angle_deg"])
        angles.append(info["angle_deg"])
    assert len(starts) > 1  # the instances start from different orientations
    # owner 2026-10-11: the goal angle is continuous - off the 15° grid - and the same seed draws the same goal
    assert all(min(abs(a) % 15.0, 15.0 - abs(a) % 15.0) > 1e-3 for a in angles) and len(set(angles)) == len(angles)
    s.sidecar.reset_instance()
    again = s.sample_instance(np.random.default_rng(5)).summary["goal"]
    assert again["angle_deg"] == angles[5] and again["axis_base"] == info["axis_base"] and again["start"] == info["start"]


def test_the_start_set_is_the_templates_whatever_the_grasps_and_their_order(monkeypatch):
    """Owner 2026-10-11: one start set per template seed - the same for every grasp, whichever grasp is screened first and
    however many there are, and again when the search re-runs (the segment fallback's other arm); another template seed,
    another set. Each grasp keeps the starts its own screening passes, in the set's order."""
    from ispatialgym.sampling import grasp_library as GL

    grasps = [GL.Grasp("block__f", f"grasp_{i}", th.tensor([0.0, 0.01 * i, 0.05]), C.quat_from_axis_angle(th.tensor([0.0, 0.0, 1.0]), 0.3 * i),
                       [0.01, 0.01]) for i in range(3)]

    def start_sets(grasps, base_seed=0, kit=None):
        s = _mr_sampler(monkeypatch, kit or _MRKit(), base_seed=base_seed)
        s.grasps = list(grasps)
        s.precompute_options()
        return {name: [o["info"] for o in opts] for name, opts in s.options.items()}

    sets = start_sets(grasps)
    full = sets["grasp_0"]
    assert len(full) == 312 and sets == {g.name: full for g in grasps}
    assert start_sets(grasps[::-1]) == sets and start_sets(grasps[2:]) == {"grasp_2": full}
    assert start_sets(grasps, base_seed=7919)["grasp_0"] != full
    s = _mr_sampler(monkeypatch, _MRKit())
    s.precompute_options()
    s.precompute_options()
    assert [o["info"] for o in s.options["grasp_0"]] == full
    # a screening that passes the hand poses above the hold's: per grasp a different part of the same set
    z_hold = float(C.compose(s.base_pose3d((0.0, 0.0, 0.0)), s.eef_in_base)[0][2])
    screened = start_sets(grasps, kit=_MRKit(start_ok=lambda pose: float(pose[0][2]) > z_hold))
    assert set(screened) == {g.name for g in grasps} and any(screened[a.name] != screened[b.name] for a, b in zip(grasps, grasps[1:]))
    for kept in screened.values():
        assert 0 < len(kept) < len(full) and [info for info in full if info in kept] == kept


def test_one_template_and_attempt_seed_draw_one_start_and_goal(monkeypatch):
    """Same seed -> same instance: two samplers of one template seed draw the same (grasp, start, goal) at one attempt seed;
    another attempt seed draws another goal, another template seed another start set."""

    def draw(base_seed, attempt_seed):
        s = _mr_sampler(monkeypatch, _MRKit(), base_seed=base_seed)
        s.precompute_options()
        d = s.sample_instance(np.random.default_rng(attempt_seed))
        return d.summary["grasp"]["name"], d.summary["goal"]

    name, goal = draw(11, 5)
    assert (name, goal) == draw(11, 5)
    assert draw(11, 6)[1]["angle_deg"] != goal["angle_deg"]
    assert draw(12, 5)[1]["start"] != goal["start"]


def test_mental_rotation_without_a_reachable_goal_rejects_the_attempt(monkeypatch):
    from ispatialgym.sampling.sim_utils import Reject

    s = _mr_sampler(monkeypatch, _MRKit(batch_ok=False))
    s.precompute_options()
    with pytest.raises(Reject) as e:
        s.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "goal_infeasible" and "screening IK" in str(e.value)
    # no start the screening IK reaches: the template has no option
    s = _mr_sampler(monkeypatch, _MRKit(start_ok=lambda pose: False))
    with pytest.raises(Reject) as e:
        s.precompute_options()
    assert e.value.reason == "goal_infeasible"


def test_mental_rotation_screens_the_first_draws_in_frame(monkeypatch):
    """Per start ``goal_draws_per_start`` (192) continuous candidates; the first ``GOAL_SCREEN_MAX`` (96) that keep the
    object in frame go to the screening IK (3 waypoints + the goal each), fewer in frame: all of them, none: no goal."""
    from ispatialgym.sampling.sim_utils import Reject

    kit = _MRKit()
    batches, ik_batch = [], kit.ik_batch
    kit.ik_batch = lambda poses, q_seed, arm="right", **kw: batches.append(len(poses)) or ik_batch(poses, q_seed, arm, **kw)
    s = _mr_sampler(monkeypatch, kit)
    s.precompute_options()
    calls = []
    for every, examined, screened in ((1, 96, 96), (3, 192, 64)):
        calls.clear()
        batches.clear()
        s.whole_in_frame = lambda cam, pose, every=every: calls.append(1) is None and (len(calls) - 1) % every == 0
        s.sidecar.reset_instance()
        s.sample_instance(np.random.default_rng(0))
        assert len(calls) == examined and batches == [4 * screened]
    s.whole_in_frame = lambda cam, pose: False
    s.sidecar.reset_instance()
    with pytest.raises(Reject) as e:
        s.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "goal_infeasible" and "no goal in frame" in str(e.value)


def test_mental_rotation_spreads_its_instances_over_the_grasps(monkeypatch):
    """Three rows, six kept instances at run_instances' attempt seeds: two each; the summary records the attempt's order and
    the counts it was sorted by (owner 2026-10-11)."""
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.v2.core import SamplerV2

    s = _mr_sampler(monkeypatch, _MRKit())
    s.grasps = [GL.Grasp("block__f", f"grasp_{i}", th.tensor([0.0, 0.01 * i, 0.05]), C.IDENTITY_QUAT.clone(), [0.01, 0.01]) for i in range(3)]
    s.precompute_options()
    monkeypatch.setattr(SamplerV2, "constraints_write_back", lambda self, draft, iid: [])
    used = []
    for index in range(6):
        s.sidecar.reset_instance()
        d = s.sample_instance(np.random.default_rng(index * 1009))
        name, spread = d.summary["grasp"]["name"], d.summary["grasp"]["spread"]
        assert spread["order"][0] == name and spread["instances"][name] == min(spread["instances"].values()) and spread["flagged"] == []
        assert d.extras["grasps"][0]["grasp_name"] == name
        s.constraints_write_back(d, f"iid_{index}")
        used.append(name)
    assert sorted(used[:3]) == sorted(used[3:]) == ["grasp_0", "grasp_1", "grasp_2"]


# ----------------------------------------------------------------------------------------------------------------
# held object: grasps spread over a template's instances (owner 2026-10-11)
# ----------------------------------------------------------------------------------------------------------------


def _spread_sampler(n=5, flagged=(), verdicts=None):
    """An ``ObjectScalingV2`` whose ``n`` library rows all have options; rows in ``flagged`` carry the tolerated
    ``one_finger`` flag, ``verdicts`` = {row: reviewer verdict}."""
    from ispatialgym.sampling import grasp_library as GL

    s = object.__new__(HO.ObjectScalingV2)
    s.grasps = [GL.Grasp("mug__ppzttc", f"grasp_{i}", th.zeros(3), C.IDENTITY_QUAT.clone(), [0.01, 0.01], flags=(["one_finger"] if i in flagged else []),
                         verdict=(verdicts or {}).get(i, "ok")) for i in range(n)]
    s.options = {g.name: {"window": (0.22, 0.51), "reference_depth": 0.5} for g in s.grasps}
    s.grasp_stats, s.prior, s.grasp_key = {}, {}, "mug__ppzttc"
    return s


def test_spread_order_least_used_unflagged_first_ties_random():
    names = ["a", "b", "c", "d", "e"]
    uses = {"a": 2, "b": 0, "c": 1, "d": 0}
    tails = set()
    for seed in range(40):
        order = HO.spread_order(np.random.default_rng(seed), names, uses, flagged=["e", "b"])
        assert order[:3] == ["d", "c", "a"] and sorted(order[3:]) == ["b", "e"]  # the flagged rows last, whatever their uses
        tails.add(tuple(order[3:]))
    assert tails == {("b", "e"), ("e", "b")}  # ties in a random order
    assert len({tuple(HO.spread_order(np.random.default_rng(seed), names, {}, ())) for seed in range(200)}) > 50


def test_grasp_order_spreads_a_templates_instances_over_the_unflagged_grasps():
    """Every instance takes the least used unflagged row (ties random): 5 rows, 12 instances -> 2 or 3 each, no row a third
    before every row has two; the flagged row stays behind."""
    s = _spread_sampler(6, flagged=(5,))
    for index in range(12):
        order = s.grasp_order(np.random.default_rng(index * 1009))
        chosen = order[0]
        assert s.grasp_uses(chosen) == min(s.grasp_uses(f"grasp_{i}") for i in range(5)) and order[-1] == "grasp_5"
        s.count_grasp(chosen, "instances")
        counts = [s.grasp_stats.get(f"grasp_{i}", {}).get("instances", 0) for i in range(5)]
        assert max(counts) - min(counts) <= 1
    assert sorted(counts) == [2, 2, 2, 3, 3] and "grasp_5" not in s.grasp_stats


def test_flagged_rows_stay_the_fallback():
    """A tolerated flag (``one_finger``) or a verdict other than ok puts a row behind every unflagged row, however often those
    were used; with only flagged rows left they are spread among themselves."""
    s = _spread_sampler(4, flagged=(0,), verdicts={1: "flag"})
    for i in (2, 3):
        for _ in range(5):
            s.count_grasp(f"grasp_{i}", "instances")
    for seed in range(50):
        order = s.grasp_order(np.random.default_rng(seed))
        assert set(order[:2]) == {"grasp_2", "grasp_3"} and set(order[2:]) == {"grasp_0", "grasp_1"}
    del s.options["grasp_2"], s.options["grasp_3"]
    s.count_grasp("grasp_0", "instances")
    assert all(s.grasp_order(np.random.default_rng(seed)) == ["grasp_1", "grasp_0"] for seed in range(20))
    assert s.spread_record(["grasp_1", "grasp_0"])["flagged"] == ["grasp_1", "grasp_0"]


def test_a_failed_attach_pushes_the_row_back_by_one_use():
    """A failed attach counts as one use: the row is not first on the next attempt while rows used as often remain, and a
    row whose attach always fails is tried about once per round of the others instead of on every attempt."""
    s = _spread_sampler(5)
    s.count_grasp("grasp_3", "attach_failed")
    assert all(s.grasp_order(np.random.default_rng(seed))[-1] == "grasp_3" for seed in range(50))
    # run_instances' attempt seeds; the attempt takes the first row of the order and the broken row's attach rejects it
    s = _spread_sampler(5)
    tries = 0
    for index in range(10):
        for attempt in range(30):
            chosen = s.grasp_order(np.random.default_rng(index * 1009 + attempt))[0]
            if chosen == "grasp_3":
                tries += 1
                s.count_grasp(chosen, "attach_failed")
                continue
            s.count_grasp(chosen, "instances")
            break
    good = [s.grasp_stats[f"grasp_{i}"]["instances"] for i in (0, 1, 2, 4)]
    assert sum(good) == 10 and max(good) - min(good) <= 1
    assert 1 <= tries <= 3 and s.grasp_stats["grasp_3"] == {"instances": 0, "attach_failed": tries}


def test_uses_count_kept_instances_and_failed_attaches_not_attach_successes(monkeypatch):
    """A failed attach counts at once (``apply_draft``, the rejection names the row); a successful attach does not count -
    the instance counts when the core keeps it (``constraints_write_back``, after the goal-occlusion gate)."""
    from types import SimpleNamespace

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.tasks.base import Draft
    from ispatialgym.sampling.v2.core import SamplerV2

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    def fail(obj, pose):
        raise Reject("grasp_failed", "assisted grasping did not attach the object (both pads in contact)")

    s = _spread_sampler(3)
    target = SimpleNamespace(name="mug_0", set_position_orientation=lambda p, q: None, disable_gravity=lambda: None, wake=lambda: None,
                             keep_still=lambda: None, get_position_orientation=stop)
    s.kit, s.obj, s.bindings, s.grasp_debug = SimpleNamespace(apply=lambda q, base: None), (lambda role: target), SimpleNamespace(one=lambda role: target), {}
    d = Draft(robot_q=th.zeros(28), object_poses={"mug_0": (th.zeros(3), C.IDENTITY_QUAT.clone())}, extras={"grasps": [{"grasp_name": "grasp_1"}]})
    s.establish_grasp = fail
    with pytest.raises(Reject) as e:
        s.apply_draft(d)
    assert e.value.details["grasp"] == "grasp_1" and s.grasp_stats == {"grasp_1": {"instances": 0, "attach_failed": 1}}
    s.establish_grasp = lambda obj, pose: None
    with pytest.raises(Stop):  # attached: apply_draft goes on to read the realised pose
        s.apply_draft(d)
    assert s.grasp_stats == {"grasp_1": {"instances": 0, "attach_failed": 1}}
    monkeypatch.setattr(SamplerV2, "constraints_write_back", lambda self, draft, iid: ["records"])
    assert s.constraints_write_back(d, "iid") == ["records"]
    assert s.grasp_stats == {"grasp_1": {"instances": 1, "attach_failed": 1}} and s.grasp_uses("grasp_1") == 2


def test_one_instance_per_template_takes_a_uniformly_random_unflagged_grasp_whatever_the_prior():
    """The eval runs (one instance per template): the order is seeded (the same seed, the same order) and over seeds every
    unflagged row comes first equally often - also with a B7 prior in which one row is far more visible than the rest."""
    s = _spread_sampler(6, flagged=(5,))
    s.prior = {("mug__ppzttc", f"grasp_{i}"): {"visible_fraction": 0.99 if i == 4 else 0.41, "in_frame": True} for i in range(6)}
    assert s.grasp_order(np.random.default_rng(17)) == s.grasp_order(np.random.default_rng(17))
    firsts = [s.grasp_order(np.random.default_rng(seed))[0] for seed in range(2000)]
    freq = {n: firsts.count(n) / len(firsts) for n in s.options}
    assert freq["grasp_5"] == 0.0 and all(0.16 < freq[f"grasp_{i}"] < 0.24 for i in range(5)), freq


def G_t(x):
    return th.as_tensor(x, dtype=th.float32)
