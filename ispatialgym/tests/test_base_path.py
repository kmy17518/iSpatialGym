"""Simulator-free tests of the shared driven base path (``ispatialgym.robot.base_path``; owner decisions 2026-09-22):
the grid staircase collapses to its corners, fillets stay inside the deviation budget and are tangent-continuous, the
hybrid yaw policy (on the spot above 90°, blended below), the sharp-corner dwells, the degenerate paths, and the
time-parameterised reference (no speed dips at the staircase's 45° stairs, cruise most of the way, dwells as
trapezoidal yaw profiles). ``pytest ispatialgym/tests/test_base_path.py``."""

from __future__ import annotations

import math

import pytest
import torch as th

from ispatialgym.robot import base_path as BP

DT = 1.0 / 30.0
V_LIN, V_ANG, A_LIN, A_YAW = 0.4, 0.64, 0.6, 1.5

#: the pilot's obstructed demo (footstool, 0916-141035): the dense 8-connected grid path, 0.2 m cells, headings
#: alternating 0 / -45 / 0 / -27 ...; init yaw -173.8°, goal yaw -155°
STAIRCASE = [
    (10.8, 2.3, math.radians(-173.8)), (11.0, 2.3, 0.0), (11.2, 2.3, 0.0), (11.4, 2.3, math.radians(-45.0)), (11.6, 2.1, 0.0),
    (11.8, 2.1, 0.0), (12.0, 2.1, 0.0), (12.2, 2.1, math.radians(-26.6)), (12.4, 2.0, 0.0), (12.6, 2.0, 0.0), (12.8, 2.0, 0.0),
    (13.0, 2.0, 0.0), (13.2, 2.0, 0.0), (13.4, 2.0, math.radians(-45.0)), (13.6, 1.8, math.radians(-26.6)), (13.8, 1.7, 0.0),
    (14.0, 1.7, 0.0), (14.2, 1.7, 0.0), (14.4, 1.7, 0.0), (14.6, 1.7, 0.0), (14.8, 1.7, 0.0), (15.0, 1.7, 0.0),
    (15.2, 1.7, math.radians(26.6)), (15.4, 1.8, math.radians(26.6)), (15.6, 1.9, math.radians(26.6)), (15.8, 2.0, math.radians(26.6)),
    (16.0, 2.1, math.radians(26.6)), (16.2, 2.2, math.radians(-155.0)),
]


def _dist_to_polyline(p, poly):
    best = float("inf")
    for a, b in zip(poly[:-1], poly[1:]):
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        best = min(best, math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)))
    return best


def _profile(ref):
    v = th.linalg.norm(ref[1:, :2] - ref[:-1, :2], dim=1) / DT
    w = (ref[1:, 2] - ref[:-1, 2]) / DT
    return v, w


def _reversals(w, thresh=0.1):
    sign, rev = None, 0
    for wi in w.tolist():
        if abs(wi) < thresh:
            continue
        s = 1 if wi > 0 else -1
        if sign is not None and s != sign:
            rev += 1
        sign = s
    return rev


# ----------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------


def test_douglas_peucker_keeps_ends_and_real_corners():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)]
    assert BP.douglas_peucker(pts, 0.05) == [0, 2, 4]
    assert BP.douglas_peucker(pts, 0.0) == [0, 1, 2, 3, 4]
    assert BP.douglas_peucker(pts[:2], 0.1) == [0, 1]


def test_staircase_collapses_to_its_corners_within_one_cell():
    path = BP.BasePath.build(STAIRCASE)
    # 28 dense cells -> a handful of real corners; the driven path stays within one cell + the fillet budget of the record
    assert 4 <= len(path.waypoints) <= 9
    budget = path.cfg.simplify_tol_m + path.cfg.fillet_deviation_m + 1e-6
    _, poses = path.drive_poses(0.02)
    assert max(_dist_to_polyline(p, STAIRCASE) for p in poses) <= budget
    # the record's ends are the exact base poses
    assert path.waypoints[0] == pytest.approx(STAIRCASE[0]) and path.waypoints[-1] == pytest.approx(STAIRCASE[-1])
    # the recorded interior yaws are the headings of the following segments
    for (x0, y0, yaw), (x1, y1, _) in zip(path.waypoints[1:-1], path.waypoints[2:]):
        assert yaw == pytest.approx(math.atan2(y1 - y0, x1 - x0))


def test_fillet_respects_deviation_budget_and_is_tangent_continuous():
    cfg = BP.BasePathConfig(simplify_tol_m=0.0)
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (2.0, 0.0, float("nan")), (2.0, 2.0, math.pi / 2)], cfg)
    arcs = [p for p in path.pieces if isinstance(p, BP.Arc)]
    assert len(arcs) == 1 and not path.dwells
    arc = arcs[0]
    # r (sec(θ/2) - 1) = budget at a 90° corner
    assert arc.radius == pytest.approx(cfg.fillet_deviation_m / (math.sqrt(2.0) - 1.0), rel=1e-6)
    mid = path.position(0.5 * (arc.s0 + arc.s1))
    assert math.hypot(mid[0] - 2.0, mid[1] - 0.0) == pytest.approx(cfg.fillet_deviation_m, abs=1e-6)
    # tangent continuity at both arc ends, and the arc ends where the lines begin
    for s in (arc.s0, arc.s1):
        assert BP.wrap_angle(path.tangent(s, -1) - path.tangent(s, 1)) == pytest.approx(0.0, abs=1e-9)
        pa, pb = path._piece(s, -1).position(s), path._piece(s, 1).position(s)
        assert pa == pytest.approx(pb, abs=1e-9)
    assert path.length == pytest.approx(4.0 - 2 * arc.radius + arc.radius * math.pi / 2, rel=1e-6)
    # the body turns over a window wide enough to keep cruising, centred on the corner, from heading 0 to π/2
    assert path.yaw(0.0) == pytest.approx(0.0) and path.yaw(path.length) == pytest.approx(math.pi / 2)
    peak = max(abs(path.yaw_rate_ds(s)) for s in path.drive_grid(0.01))
    assert peak <= cfg.corner_yaw_rate_rad_per_m * 1.001


def test_sharp_corner_becomes_a_dwell_with_checked_sweep():
    cfg = BP.BasePathConfig(simplify_tol_m=0.0)
    # a 150° turn: too sharp for a fillet -> stop, rotate on the spot, go on
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (1.0, 0.0, float("nan")), (1.0 + math.cos(math.radians(150)), math.sin(math.radians(150)), float("nan"))], cfg)
    corners = [d for d in path.dwells if d.kind == "corner"]
    assert len(corners) == 1 and corners[0].s == pytest.approx(1.0)
    assert BP.wrap_angle(path.yaw(1.0, 1) - path.yaw(1.0, -1)) == pytest.approx(math.radians(150), abs=1e-6)
    poses = path.stride_poses(0.1, dwell_step_rad=0.2)
    at_corner = [p for p in poses if math.hypot(p[0] - 1.0, p[1]) < 1e-6]
    assert len(at_corner) >= math.ceil(math.radians(150) / 0.2)  # the sweep is collision-checked in <= 0.2 rad steps
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    v, _ = _profile(ref)
    # the base stops at the corner
    i_corner = int(th.argmin(th.linalg.norm(ref[:, :2] - th.tensor([1.0, 0.0]), dim=1)))
    assert float(v[max(i_corner - 1, 0)]) < 0.05


def test_stride_poses_bound_the_heading_step():
    """Owner 2026-10-13 item 3a: ``max_yaw_step`` re-samples the schedule where it turns faster than the bound per stride
    (fillet turn windows, short end blends), sweeps a sharp corner from its arrival yaw and the dwells at the smaller step;
    the default (None) is unchanged and a straight drive gets nothing extra."""
    cfg = BP.BasePathConfig(simplify_tol_m=0.0)
    step = math.radians(3.0)
    wps = [(0.0, 0.0, math.radians(100)), (1.0, 0.0, float("nan")), (1.0 + math.cos(math.radians(150)), math.sin(math.radians(150)), float("nan")),
           (0.0, 1.0, float("nan")), (0.0, 1.3, math.radians(-30))]
    path = BP.BasePath.build(wps, cfg)
    assert {d.kind for d in path.dwells} >= {"start", "corner"}
    plain, bounded = path.stride_poses(0.1), path.stride_poses(0.1, max_yaw_step=step)
    assert plain == path.stride_poses(0.1, max_yaw_step=None) and len(bounded) > len(plain)
    assert max(abs(BP.wrap_angle(b[2] - a[2])) for a, b in zip(bounded[:-1], bounded[1:])) <= step + 1e-9
    assert max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(bounded[:-1], bounded[1:])) <= 0.1 + 1e-9
    # every bounded pose lies on the schedule (the drive re-sampled, not interpolated between strides)
    for x, y, yaw in bounded:
        s = path.project((x, y), ds=0.001)
        assert math.hypot(*(u - v for u, v in zip(path.position(s), (x, y)))) < 2e-3
    straight = BP.BasePath.build([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    assert straight.stride_poses(0.5, max_yaw_step=step) == straight.stride_poses(0.5)


def test_hybrid_end_turns():
    cfg = BP.BasePathConfig()
    # 170° off the first heading: turn on the spot first, then drive facing the path
    path = BP.BasePath.build([(0.0, 0.0, math.radians(170)), (3.0, 0.0, 0.0)], cfg)
    assert [d.kind for d in path.dwells] == ["start"]
    assert path.yaw(0.0) == pytest.approx(0.0) and path.yaw_start == pytest.approx(0.0)
    poses = path.stride_poses(0.1)
    assert poses[0] == pytest.approx((0.0, 0.0, math.radians(170)))  # the departing pose is checked as it stands
    # 60° off: blended over the first blend_m while driving; the body faces the way at the end of the window
    path = BP.BasePath.build([(0.0, 0.0, math.radians(60)), (3.0, 0.0, 0.0)], cfg)
    assert not path.dwells
    assert path.yaw(0.0) == pytest.approx(math.radians(60)) and path.yaw(cfg.blend_m) == pytest.approx(0.0, abs=1e-9)
    assert abs(path.yaw(0.5 * cfg.blend_m) - math.radians(30)) < 1e-6  # the blend's midpoint
    # goal yaw 120° off the last heading: drive to the end, then turn on the spot
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (3.0, 0.0, math.radians(120))], cfg)
    assert [d.kind for d in path.dwells] == ["end"]
    assert path.yaw(path.length) == pytest.approx(0.0) and path.yaw_goal == pytest.approx(math.radians(120))
    # both ends blend on a short path: the windows split the path
    path = BP.BasePath.build([(0.0, 0.0, math.radians(45)), (1.0, 0.0, math.radians(-45))], cfg)
    assert not path.dwells and path.yaw(0.0) == pytest.approx(math.radians(45)) and path.yaw(1.0) == pytest.approx(math.radians(-45))
    assert path.yaw(0.5) == pytest.approx(0.0, abs=1e-9)


def test_degenerate_paths():
    # pure rotation: a dwell from the departing to the goal yaw, no drive
    path = BP.BasePath.build([(1.0, 2.0, 0.0), (1.0, 2.0, 1.0)])
    assert path.length == 0.0 and [d.kind for d in path.dwells] == ["end"]
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    assert th.allclose(ref[:, :2], th.tensor([[1.0, 2.0]]).expand(len(ref), 2))
    assert float(ref[-1, 2]) == pytest.approx(1.0) and len(ref) > 5
    # nothing to do at all
    path = BP.BasePath.build([(1.0, 2.0, 0.3), (1.0, 2.0, 0.3)])
    assert path.length == 0.0 and not path.dwells
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    assert ref.shape == (1, 3)
    # a single waypoint
    path = BP.BasePath.build([(0.0, 0.0, 0.3)])
    assert path.stride_poses(0.1) == [(0.0, 0.0, 0.3), (0.0, 0.0, 0.3)]
    # a two-point straight drive with equal yaws is a plain straight line facing along it
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    assert len(path.pieces) == 1 and not path.dwells
    assert path.stride_poses(0.5) == pytest.approx([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 0.0)])


def test_project_and_drive_grid():
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    assert path.project((1.234, 0.3)) == pytest.approx(1.23, abs=0.011)
    assert path.drive_grid(0.5) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert path.drive_grid(0.3)[-1] == pytest.approx(2.0)


# ----------------------------------------------------------------------------------------------------------------
# time parameterisation
# ----------------------------------------------------------------------------------------------------------------


def test_dwell_yaws_trapezoid():
    yaws = BP.dwell_yaws(0.0, 3.0, DT, V_ANG, A_YAW)
    assert yaws[-1] == pytest.approx(3.0)
    rates = [(b - a) / DT for a, b in zip(yaws[:-1], yaws[1:])]
    assert max(rates) <= V_ANG * 1.01 and min(rates) >= -1e-9
    accs = [(b - a) / DT for a, b in zip(rates[:-1], rates[1:])]
    assert max(abs(a) for a in accs) <= A_YAW * 1.05
    # shortest way, both directions
    yaws = BP.dwell_yaws(3.0, -3.0, DT, V_ANG, A_YAW)
    assert yaws[-1] == pytest.approx(3.0 + (2 * math.pi - 6.0))  # +0.28 rad, not -6 rad
    assert BP.dwell_yaws(1.0, 1.0, DT, V_ANG, A_YAW) == []


def test_staircase_reference_cruises_without_zigzag():
    path = BP.BasePath.build(STAIRCASE)
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    v, w = _profile(ref)
    assert float(ref[-1, 0]) == pytest.approx(16.2) and float(ref[-1, 1]) == pytest.approx(2.2)
    assert BP.wrap_angle(float(ref[-1, 2]) - math.radians(-155.0)) == pytest.approx(0.0, abs=1e-5)
    # the drive (outside the two on-the-spot end turns) runs at cruise most of the way and never dips at the stairs
    driving = v > 1e-3
    v_drive = v[driving]
    assert float((v_drive >= 0.9 * V_LIN).float().mean()) > 0.7
    # inner part (after the acceleration ramp, before the deceleration): the slowest sample stays above half cruise
    i0, i1 = int(driving.nonzero()[0]), int(driving.nonzero()[-1])
    inner = v[i0 + 30 : i1 - 30]
    assert float(inner.min()) > 0.5 * V_LIN
    # the body turns left / right at most as often as the real corners (the dense record reversed 6 times)
    assert _reversals(w[i0:i1]) <= 3
    # velocity and yaw-rate limits hold
    assert float(v.max()) <= V_LIN * 1.01 and float(w.abs().max()) <= V_ANG * 1.05
    # acceleration limit of the drive (the sampled hand-overs at the very ends aside: the tracker clamps those anyway)
    a = (v[1:] - v[:-1]) / DT
    assert float(a[i0 + 10 : i1 - 10].abs().max()) <= A_LIN * 1.1


def test_reference_is_shorter_than_the_dense_record_would_be():
    """The reference of the pilot's obstructed record: well below the 36.8 s the corner-aware trapezoid on the dense
    staircase took (the two ~175° end turns are ~5 s each at the yaw limits, the 5.7 m drive ~15 s)."""
    path = BP.BasePath.build(STAIRCASE)
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    assert len(ref) * DT < 28.0


def test_l_shaped_drive_keeps_speed_through_the_corner_window():
    cfg = BP.BasePathConfig(simplify_tol_m=0.0)
    path = BP.BasePath.build([(0.0, 0.0, 0.0), (3.0, 0.0, float("nan")), (3.0, 3.0, math.pi / 2)], cfg)
    ref = BP.time_parameterise(path, DT, V_LIN, V_ANG, A_LIN, A_YAW)
    v, w = _profile(ref)
    # the 90° corner (12 cm fillet) is limited by the lateral acceleration, not by the body's turn: sqrt(a r) ~ 0.27 m/s
    arc = next(p for p in path.pieces if isinstance(p, BP.Arc))
    v_corner = math.sqrt(A_LIN * arc.radius)
    inner = v[30:-30]
    assert float(inner.min()) >= 0.9 * v_corner
    assert float(w.abs().max()) <= V_ANG * 1.05
    assert _reversals(w) == 0
