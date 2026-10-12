"""Simulator-free tests of the v2 tabletop phases (``sampling/v2/tabletop.py``) and the pure rules of the tabletop task
modules: the §4.4 margins table per task, the §4.3 Phase-1b clamp / camera-FK inverse, the §6.5 cube-grasp tilt pairing
and visible-face rule on synthetic blocks, the §4.6 / §3.12 distractor spacing and look-alike exclusion, and the §6.10
axial yaw draw (``pytest ispatialgym/tests/test_v2_tabletop.py``). Nothing here imports OmniGibson."""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import tabletop as TT


def test_module_imports_without_the_simulator():
    # in a fresh interpreter: other test modules of the suite legitimately import OmniGibson first, so ``sys.modules``
    # of this process proves nothing (same pattern as ``test_conventions.py::test_module_is_pure``)
    import subprocess

    code = ("import sys, ispatialgym.sampling.v2.tabletop; "
            "assert 'omnigibson' not in sys.modules, 'tabletop.py must not import omnigibson at module level'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


# ----------------------------------------------------------------------------------------------------------------
# §4.4 margins table
# ----------------------------------------------------------------------------------------------------------------


def test_margins_iors_row_draws_over_the_whole_edge_and_skips_edges_too_short_for_both_objects():
    """A1 (owner 2026-10-12): the IORS row draws the lateral over the whole edge and only asks the edge to hold both objects at the
    smallest gap, m_end from its ends: L >= 2 w_t + g_min + 2 w_p + 2 m_end; D_p = min(W_p + 0.03, R_img) as before."""
    row = {"w_t": 0.05, "w_p": 0.10, "W_p": 0.20, "D_p_max": 0.23, "safety": 0.02, "relation": "ontop", "g_NT": None}
    m_R, m_L, info = TT.lateral_margins(row, lam_min=-0.30, lam_max=0.40, b=0.9, R_img=0.6)
    assert (m_R, m_L) == (0.0, 0.0) and info["D_p"] == pytest.approx(0.23)  # min(W_p + 0.03, R_img) = W_p + 0.03 here
    assert info["g_min"] == pytest.approx(0.03) and info["m_end"] == pytest.approx(0.10)
    assert info["L_min"] == pytest.approx(0.10 + 0.03 + 0.20 + 0.20)
    assert TT.lateral_bounds(0.53, m_R, m_L) == (pytest.approx(-0.265), pytest.approx(0.265))  # the whole edge
    # R_img caps D_p; the edge rule depends on neither D_p nor the band's extent (the old margins carried both, twice)
    _, _, info2 = TT.lateral_margins(row, -0.56, 0.16, b=0.9, R_img=0.10)
    assert info2["D_p"] == pytest.approx(0.10) and info2["L_min"] == pytest.approx(info["L_min"])
    # generation.table_end_margin_m / footprint_gap_min_m of the alias; nextto's smallest gap is g_NT + 0.03
    assert TT.lateral_margins(dict(row, m_end=0.05, g_min=0.04), -0.3, 0.4, 0.9, 0.6)[2]["L_min"] == pytest.approx(0.10 + 0.04 + 0.20 + 0.10)
    nt = TT.lateral_margins(dict(row, relation="nextto", g_NT=0.063), -0.3, 0.4, 0.9, 0.6)[2]
    assert nt["g_min"] == pytest.approx(0.093) and nt["L_min"] == pytest.approx(0.10 + 0.093 + 0.20 + 0.20)
    # the s3 cases: under_small#03's plate (w_t 0.123) and place mat (w_p 0.283) need 1.042 m - its 1.06 m pedestal table and the 1.16 m
    # breakfast-table ends pass (the old margins asked 3.25 / 3.33 m); nextto#04's shoes (0.154 / 0.154, g_NT 0.063) need 0.91 m - the
    # bench's 1.14 m edge passes, its 0.36 m ends do not (0.909 m with these rounded extents)
    plate = TT.lateral_margins({**row, "w_t": 0.123, "w_p": 0.283, "W_p": 0.566, "D_p_max": 0.596}, -0.48, 0.16, 0.8, 0.8)[2]
    assert plate["L_min"] == pytest.approx(1.042) and plate["L_min"] < 1.06
    shoes = TT.lateral_margins({**row, "w_t": 0.154, "w_p": 0.154, "W_p": 0.308, "D_p_max": 0.338, "relation": "nextto", "g_NT": 0.063}, -0.48, 0.16, 0.8, 0.8)[2]
    assert shoes["L_min"] == pytest.approx(0.909) and 0.36 < shoes["L_min"] < 1.14


def test_margins_nextto_rejects_pair_when_dp_below_gnt_plus_3cm():
    row = {"w_t": 0.05, "w_p": 0.10, "W_p": 0.20, "D_p_max": 0.23, "safety": 0.02, "relation": "nextto", "g_NT": 0.25}
    _, _, info = TT.lateral_margins(row, -0.3, 0.4, 0.9, 0.6)
    assert info["reject"][0] == "pair_gap"
    row_ok = dict(row, g_NT=0.10)
    _, _, info_ok = TT.lateral_margins(row_ok, -0.3, 0.4, 0.9, 0.6)
    assert "reject" not in info_ok


def test_margins_other_rows():
    right, left = 0.30, 0.40
    m_R, m_L, _ = TT.lateral_margins({"kind": "attachment", "w_target": 0.03, "w_partner": 0.06}, -right, left, 1.0, 1.0)
    assert (m_R, m_L) == (pytest.approx(right + 0.06 + 0.02), pytest.approx(left + 0.06 + 0.02))
    m_R, m_L, info = TT.lateral_margins({"kind": "axial", "long": 0.40, "short": 0.10}, -right, left, 1.0, 1.0)
    w_diag = 0.5 * math.hypot(0.40, 0.10)
    assert info["w_diag"] == pytest.approx(w_diag) and m_R == pytest.approx(right + w_diag + 0.02) and m_L == pytest.approx(left + w_diag + 0.02)
    m_R, m_L, _ = TT.lateral_margins({"kind": "block_design"}, -right, left, 1.0, 1.0)
    assert m_R == pytest.approx(right + 0.0125 + 0.026 + 0.02) and m_L == pytest.approx(left + 0.0125 + 0.02)
    m_R, m_L, _ = TT.lateral_margins({"kind": "disembedding", "lateral_margin": 0.437}, -right, left, b=0.55, R_img=0.55)
    assert m_R == m_L == pytest.approx(0.55 + 0.437 + 0.02)
    m_R, m_L, _ = TT.lateral_margins({"kind": "articulation_small", "w_a": 0.31}, -right, left, 1.0, 1.0)
    assert m_R == pytest.approx(right + 0.31 + 0.02) and m_L == pytest.approx(left + 0.31 + 0.02)
    with pytest.raises(ValueError):
        TT.lateral_margins({"w_t": 0.1}, -0.1, 0.1, 1.0, 1.0)


def test_lateral_bounds_skips_short_edges():
    assert TT.lateral_bounds(1.6, 0.5, 0.6) == (pytest.approx(-0.3), pytest.approx(0.2))
    assert TT.lateral_bounds(0.9, 0.5, 0.6) is None  # m_R + m_L > L


# ----------------------------------------------------------------------------------------------------------------
# §4.3 Phase 1b: camera FK constants and the clamp
# ----------------------------------------------------------------------------------------------------------------


def test_camera_fk_matches_the_pilot_constants():
    fk = TT.camera_fk()
    assert math.degrees(fk.pitch) == pytest.approx(22.6, abs=0.3)  # §4.3 [pilot] 22.6° down at tilt 0
    assert fk.x_at(0.5) == pytest.approx(TT.CAMERA_X_PILOT, abs=0.03)  # x_c ≈ 0.22 m ahead of the base
    zs = [fk.z_at(t) for t in np.linspace(0, 1, 11)]
    assert all(a > b for a, b in zip(zs[:-1], zs[1:]))  # z_c(t) decreases monotonically with t
    assert fk.hand_below == pytest.approx(TT.HAND_BELOW_CAMERA_PILOT)
    # the untucked hands sit at 1.20 m at translate 0.5 (§3.3) -> the camera 0.19 m above them ≈ 1.39 m
    assert fk.z_at(0.5) == pytest.approx(1.20 + 0.19, abs=0.03)
    assert math.degrees(fk.phi_m(0.05)) == pytest.approx(46.6, abs=0.2)


def test_camera_fk_inverse_is_clamped_to_the_grid():
    fk = TT.camera_fk()
    for t in (0.1, 0.5, 0.9):
        assert fk.t_of_z(fk.z_at(t)) == pytest.approx(t, abs=0.02)
    assert fk.t_of_z(3.0) == 0.0  # higher than the upright camera -> translate 0
    assert fk.t_of_z(0.2) == TT.MAX_TRANSLATE  # lower than the deepest camera -> 0.95, never 1.0 (§3.3 self-collision)
    assert fk.tilt_limit(0.5) < 0 and fk.tilt_limit(0.0) < fk.tilt_limit(0.95)  # deeper trunks leave less tilt


def test_phase1b_clamp_logic():
    fk = TT.camera_fk()
    phi = fk.phi_m()
    # Δ_aim inside [Δ_min, Δ_max] -> Δ_aim
    r = TT.init_trunk_delta(edge_depth=0.45, d_ws=0.40, d_near=0.15, x_c=0.22, H_max=0.20, hand_below=0.19, pitch=fk.pitch, phi_m=phi)
    assert r["delta_aim"] == pytest.approx(0.45 + 0.40 - 0.22)
    assert r["delta_min"] == pytest.approx(0.20 + 0.10 + 0.19)
    assert r["delta_max"] == pytest.approx((0.45 + 0.15 - 0.22) * math.tan(fk.pitch + phi))
    assert r["delta"] == pytest.approx(r["delta_aim"]) and not r["conflict"]
    # a tall object lifts Δ to Δ_min (hands >= 10 cm above it)
    r2 = TT.init_trunk_delta(0.45, 0.40, 0.15, 0.22, H_max=0.45, hand_below=0.19, pitch=fk.pitch, phi_m=phi)
    assert r2["delta"] == pytest.approx(r2["delta_min"]) and r2["delta_min"] > r2["delta_aim"]
    # a deep workspace with a near edge close to the robot caps Δ at Δ_max (near edge in frame)
    r3 = TT.init_trunk_delta(0.45, 0.90, 0.0, 0.22, H_max=0.05, hand_below=0.19, pitch=fk.pitch, phi_m=phi)
    assert r3["delta"] == pytest.approx(r3["delta_max"]) and r3["delta_max"] < r3["delta_aim"]
    # conflicting bounds (Δ_min > Δ_max): the upper bound wins (clamp semantics) and the conflict is flagged
    r4 = TT.init_trunk_delta(0.45, 0.30, 0.0, 0.22, H_max=0.80, hand_below=0.19, pitch=fk.pitch, phi_m=phi)
    assert r4["conflict"] and r4["delta"] == pytest.approx(r4["delta_max"])
    assert TT.clamp(5.0, 0.0, 1.0) == 1.0 and TT.clamp(-1.0, 0.0, 1.0) == 0.0 and TT.clamp(0.4, 0.0, 1.0) == 0.4


# ----------------------------------------------------------------------------------------------------------------
# §6.5 cube grasp rule and visible-face rule on synthetic blocks
# ----------------------------------------------------------------------------------------------------------------


def _rotation_with_up(face: str) -> th.Tensor:
    """A cube rotation (xyzw) that puts ``face`` along world +z."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = th.zeros(3, 3)
            for i, (p, sg) in enumerate(zip(perm, signs)):
                m[i, p] = float(sg)
            if abs(float(th.det(m)) - 1.0) < 1e-6:
                q = C.mat_to_quat(m)
                if float(TT.face_normal_world(face, q)[2]) > 0.99:
                    return q
    raise AssertionError(face)


def test_cube_tilt_pairing_has_exactly_one_downward_pick():
    # d_pick = R_a(∓90°)·d_place: tilt_pick = tilt_place − 90·flip_sign (§6.5), θ = 45° gives the two symmetric 45° grasps
    assert TT.cube_tilt_pair(45.0, +1) == (pytest.approx(-45.0), 45.0)
    assert TT.cube_tilt_pair(45.0, -1) == (pytest.approx(135.0), 45.0)
    block = (th.tensor([0.5, 0.0, 0.8125]), C.IDENTITY_QUAT.clone())
    downward = []
    for sign in (+1, -1):
        tilt_pick, _ = TT.cube_tilt_pair(45.0, sign)
        g = C.cube_grasp(block, "y", tilt_pick, yaw=0.0)
        downward.append(g.downward)
        assert abs(float(th.dot(g.closing, th.tensor([0.0, 1.0, 0.0])))) > 0.99  # pinch across the y faces, pads flat
    assert downward.count(True) == 1  # exactly one tilt sign has a downward pick approach
    for tilt in (0.0, 30.0, 45.0, 60.0):  # the place approaches of the cube grasp set
        assert C.cube_grasp(block, "x", tilt, yaw=math.pi / 2).downward


def test_cube_flip_plan_smallest_rotation():
    table_yaw = math.radians(-90.0)  # goal grid +x = the robot's right
    for up in ("+z", "+x", "-y", "-z", "+y", "-x"):
        q0 = C.qmul(C.quat_from_yaw(math.radians(-30.0)), _rotation_with_up(up))
        for goal in TT.FACE_AXES:
            plan = TT.cube_flip_plan(q0, goal, table_yaw)
            if goal == TT.opposite_face(up):
                assert plan is None  # the init up face is never the opposite of the goal face
                continue
            assert plan is not None
            assert abs(plan["yaw_deg"]) <= 45.0 + 1e-6  # yaw-align |ψ| <= 45°
            assert TT.up_face(plan["goal_quat"]) == goal  # the goal face ends up on top
            # the goal orientation is table-aligned: every horizontal face normal lies on a table axis
            for f in TT.FACE_AXES:
                n = TT.face_normal_world(f, plan["goal_quat"])
                if abs(float(n[2])) < 0.5:
                    ang = math.atan2(float(n[1]), float(n[0])) - table_yaw
                    assert abs(math.sin(2 * ang)) < 1e-4
            if goal == up:
                assert not plan["flips"] and plan["flip_sign"] == 0 and plan["flip_axis"] is None
            else:
                assert plan["flips"] and plan["flip_sign"] in (-1, 1)
                a = plan["flip_axis"]
                assert abs(float(a[2])) < 1e-6  # horizontal table axis
                # one flip only: the pinch (face pair) axis is parallel to the flip axis after the yaw-align
                q1 = C.qmul(C.quat_from_yaw(math.radians(plan["yaw_deg"])), q0)
                pair_axis = C.quat_rotate(q1, th.tensor(TT.FACE_AXES["+" + plan["face_pair"]]))
                assert abs(abs(float(th.dot(pair_axis, a))) - 1.0) < 1e-4
                # a single 90° flip about the axis reaches the goal orientation from the yaw-aligned one
                q_flip = C.quat_from_axis_angle(a, plan["flip_sign"] * math.pi / 2)
                assert C.quat_angle(C.qmul(q_flip, q1), plan["goal_quat"]) < 1e-4


def test_visible_face_rule_turns_the_goal_face_toward_the_robot():
    toward = th.tensor([-1.0, 0.0])  # robot at -x looking at +x: -inward
    for up in ("+z", "+x", "-y"):
        q_up = _rotation_with_up(up)
        for goal in TT.FACE_AXES:
            if goal in (up, TT.opposite_face(up)):
                continue
            yaw = TT.facing_coarse_yaw(q_up, goal, toward)
            assert abs((yaw / (math.pi / 2)) - round(yaw / (math.pi / 2))) < 1e-9  # a multiple of 90°
            n = TT.face_normal_world(goal, C.qmul(C.quat_from_yaw(yaw), q_up))
            assert float(n[0]) < -0.99 and abs(float(n[2])) < 1e-6  # the goal face is a robot-facing side face
    # a pattern that is on no up face is adjacent to at least three of the up faces (§4.6): every non-up face except the
    # opposite one can be shown by some block of a 2 x 2 with four distinct up faces
    ups = ["+z", "+x", "+y", "-x"]
    for goal in TT.FACE_AXES:
        if goal in ups:
            continue
        assert sum(1 for u in ups if goal != TT.opposite_face(u)) >= 3


def test_face_points_grid_lies_on_the_face():
    pose = (th.tensor([0.0, 0.0, 0.0125]), C.IDENTITY_QUAT.clone())
    pts = TT.face_points(pose, "-x", edge=0.025, n=3)
    assert pts.shape == (9, 3)
    assert th.allclose(pts[:, 0], th.full((9,), -0.0125))  # on the -x face plane
    assert float(pts[:, 1].abs().max()) < 0.0125 and float((pts[:, 2] - 0.0125).abs().max()) < 0.0125


# ----------------------------------------------------------------------------------------------------------------
# §4.6 step 3 / §3.12 distractors
# ----------------------------------------------------------------------------------------------------------------


def test_distractor_spacing_from_the_pinch_width():
    # x = (open width 0.10 − pinch width of the drawn grasp)/2 + 0.01
    assert TT.distractor_spacing(0.0534) == pytest.approx((0.10 - 0.0534) / 2 + 0.01)
    assert TT.distractor_spacing(0.10) == pytest.approx(0.01)

    class G:
        meta = {"width": 0.064}

    class Ginf:
        meta = {"width": float("inf")}

    assert TT.grasp_pinch_width(G()) == pytest.approx(0.064)
    assert TT.grasp_pinch_width(Ginf()) is None
    assert TT.grasp_pinch_width(object()) is None


def test_lookalike_exclusion_keeps_one_model_per_group_and_none_of_the_targets_group():
    pool = [("beaker", "a"), ("beaker", "b"), ("beaker", "c"), ("mug", "x"), ("mug", "y"), ("mug", "z"), ("cup", "q")]
    groups = [[("beaker", "a"), ("beaker", "b")], [("mug", "x"), ("mug", "y")]]
    out = TT.exclude_lookalikes(pool, ("beaker", "a"), groups)
    assert ("beaker", "a") not in out and ("beaker", "b") not in out  # the target and its look-alikes
    assert ("beaker", "c") in out and ("cup", "q") in out and ("mug", "z") in out
    assert sum(1 for m in out if m in (("mug", "x"), ("mug", "y"))) == 1  # at most one model per look-alike group
    # the target itself is excluded even without a group; order is preserved without an rng
    assert TT.exclude_lookalikes(pool, ("cup", "q"), []) == [m for m in pool if m != ("cup", "q")]
    rng = np.random.default_rng(3)
    out_r = TT.exclude_lookalikes(pool, ("beaker", "a"), groups, rng)
    assert set(out_r) <= set(pool) and ("beaker", "a") not in out_r and ("beaker", "b") not in out_r


# ----------------------------------------------------------------------------------------------------------------
# §6.10 axial yaw draw
# ----------------------------------------------------------------------------------------------------------------


def test_axial_yaw_draw_range_and_sign():
    rng = np.random.default_rng(0)
    signs = set()
    for _ in range(200):
        yaw, delta = TT.axial_init_yaw(rng, aligned_yaw=0.3, deviation_range_deg=(20.0, 70.0))
        assert math.radians(20.0) - 1e-9 <= abs(delta) <= math.radians(70.0) + 1e-9
        assert yaw == pytest.approx(C.wrap_angle(0.3 + delta))
        signs.add(1 if delta > 0 else -1)
    assert signs == {1, -1}  # random sign


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers shared by the phases
# ----------------------------------------------------------------------------------------------------------------


def test_hull_extents_and_corners():
    hull = th.tensor([[-0.05, -0.02, 0.0], [0.05, 0.02, 0.10]])
    assert TT.half_extent(hull, C.IDENTITY_QUAT) == pytest.approx(0.05)
    assert TT.height_of(hull, C.IDENTITY_QUAT) == pytest.approx(0.10)
    q = C.quat_from_yaw(math.pi / 2)
    ext = TT.hull_extents(hull, q)
    assert ext[0].item() == pytest.approx(0.04, abs=1e-6) and ext[1].item() == pytest.approx(0.10, abs=1e-6)
    corners = TT.hull_corners_world(hull, (th.tensor([1.0, 2.0, 0.5]), C.IDENTITY_QUAT))
    assert corners.shape == (8, 3) and corners[:, 2].min().item() == pytest.approx(0.5) and corners[:, 2].max().item() == pytest.approx(0.6)
    box = TT.box_corners((0.0, 0.0), 0.1, 0.8, 0.9)
    assert box.shape == (8, 3)


def test_tilt_scan_runs_from_zero_to_the_joint_limit():
    fk = TT.camera_fk()
    ts = TT.tilt_scan(fk, 0.5, n=7)
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(fk.tilt_limit(0.5)) and all(a > b for a, b in zip(ts[:-1], ts[1:]))


# ----------------------------------------------------------------------------------------------------------------
# §6.4 disembedding: the tray inner-floor raster (``tasks/disembedding.TrayFloor``); the module imports the sampler core
# (OmniGibson importable without a launch, as in test_v2_large_scale.py) but nothing here needs a live world
# ----------------------------------------------------------------------------------------------------------------


def _disembedding():
    return pytest.importorskip("ispatialgym.sampling.v2.tasks.disembedding")


def _synthetic_tray_hits(nx=30, ny=44, rim=2, floor_z=0.005, rim_z=0.030, tilt_per_cell=0.0, miss=()):
    """Per-ray tray hit heights of a raster over a tray footprint: ``rim`` cells of wall top around an inner floor
    (optionally sloping ``tilt_per_cell`` along x); cells in ``miss`` hit nothing (nan)."""
    zs = th.full((nx * ny,), float("nan"))
    inner = th.zeros(nx * ny, dtype=th.bool)
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            if (i, j) in miss:
                continue
            if rim <= i < nx - rim and rim <= j < ny - rim:
                zs[k] = floor_z + tilt_per_cell * i
                inner[k] = True
            else:
                zs[k] = rim_z
    return zs, inner


def test_tray_floor_cells_keep_the_inner_floor_and_drop_the_rim():
    D = _disembedding()
    zs, inner = _synthetic_tray_hits(miss={(0, 0), (15, 20)})
    floor_z, keep = D.TrayFloor.floor_cells(zs)
    assert floor_z == pytest.approx(0.005)
    assert th.equal(keep, inner)  # every inner cell, no rim cell, no missed ray
    assert int(keep.sum()) == 26 * 40 - 1 >= D.FLOOR_MIN_CELLS
    # a tray that toppled (floor sloping 1 cm per cell): only one 4 mm band of cells survives near the lowest level
    zs_t, _ = _synthetic_tray_hits(rim=0, tilt_per_cell=0.01)
    fz_t, keep_t = D.TrayFloor.floor_cells(zs_t)
    assert int(keep_t.sum()) <= 44 < 0.1 * int(keep.sum()) and fz_t > 0.005
    # no tray hit at all: nan level, empty mask (the caller reports the raycast geometry)
    fz_n, keep_n = D.TrayFloor.floor_cells(th.full((12,), float("nan")))
    assert math.isnan(fz_n) and not bool(keep_n.any()) and keep_n.shape == (12,)


def test_tray_floor_grid_and_world_view():
    D = _disembedding()
    res = 0.01
    xs = th.arange(-0.125, 0.126, res)  # 26 cells along x (short axis), centred on the root
    ys = th.arange(-0.195, 0.196, res)  # 40 cells along y (long axis)
    gx, gy = th.meshgrid(xs, ys, indexing="ij")
    cells = th.stack([gx.flatten(), gy.flatten()], dim=1)
    floor = D.TrayFloor(cells, z_local=0.005, res=res, rim_local=0.030)
    assert floor.n_cells == 26 * 40 and floor.area() == pytest.approx(26 * 40 * res * res)
    assert floor.contains_local((0.0, 0.0)) and floor.contains_local((0.0, 0.0), margin=0.005)
    assert floor.contains_local((0.122, 0.19)) and not floor.contains_local((0.122, 0.19), margin=0.01)  # a margin pushes the outermost cell out
    assert not floor.contains_local((0.14, 0.0)) and not floor.contains_local((0.0, -0.21))
    # world view: tray root at (1, 2, 0.82) yawed 90 deg -> the long (local y) axis lies along world x, the short one along world y
    pose = (th.tensor([1.0, 2.0, 0.82]), C.quat_from_yaw(math.pi / 2))
    w = floor.at(pose)
    assert w.z == pytest.approx(0.825) and w.rim_z == pytest.approx(0.85)
    assert th.allclose(w.centre, th.tensor([1.0, 2.0]), atol=1e-6)
    assert w.contains((1.0, 2.0)) and w.contains((1.0 - 0.19, 2.0)) and w.contains((1.0, 2.0 + 0.12))
    assert not w.contains((1.0, 2.0 + 0.19)) and not w.contains((1.0 + 0.25, 2.0))  # the short axis is along world y now
    square = th.tensor([[0.98, 1.98], [1.02, 1.98], [1.02, 2.02], [0.98, 2.02]])
    assert w.polygon_supported(square, margin=0.005)
    assert not w.polygon_supported(square + th.tensor([0.0, 0.115]), margin=0.005)  # straddles the long side
    assert w.cells().shape == (26 * 40, 2) and w.area() == pytest.approx(floor.area())


def test_tray_rest_tilt_and_repark_slots():
    D = _disembedding()
    assert D.tilt_deg(C.IDENTITY_QUAT) == pytest.approx(0.0, abs=1e-6)
    assert D.tilt_deg(C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), math.radians(14.5))) == pytest.approx(14.5, abs=1e-4)
    assert D.tilt_deg(C.quat_from_yaw(1.0)) == pytest.approx(0.0, abs=1e-5)  # a yaw is not a tilt
    assert D.REST_Z_TOL == pytest.approx(0.01) and D.REST_TILT_MAX_DEG == pytest.approx(3.0)

    class World:
        def __init__(self):
            self.spawned = {"target_0": object(), "source_tray_0": object(), "destination_tray_0": object()}
            self.calls = []

        def park(self, obj, index):
            self.calls.append((obj, index))

    class S:
        world = World()

    D.repark_spawned(S)
    # every spawned object, at the slot the core uses (its position in ``world.spawned``)
    assert S.world.calls == [(o, list(S.world.spawned).index(n)) for n, o in S.world.spawned.items()]


def _ring_top():
    """A 1.2 x 0.8 m counter whose free cells ring an occupied middle (a sink / burner block at x 0.4-0.8, y 0.24-0.56)."""
    from ispatialgym.sampling.sim_utils import SupportTop

    xs, ys = th.arange(60) * 0.02, th.arange(40) * 0.02
    occupied = th.zeros(60, 40, dtype=th.bool)
    occupied[20:40, 12:28] = True
    return SupportTop("countertop_x_0", xs, ys, th.ones(60, 40, dtype=th.bool), occupied, 0.9, 0.0)


def test_tray_measurement_drop_spots_avoid_an_occupied_middle():
    """Owner 2026-10-11 ~20:25: the measurement drop went to the free cells' centroid, which on a counter with a sink, burner or
    microwave in the middle lies on that object (disembedding#02 c1 / c5, #06 c1 / c4). The spots are free cells with the most
    clearance (distance transform), the first one at least the tray's half-diagonal from any occupied cell or the edge, and the
    next ones well apart."""
    D = _disembedding()
    top = _ring_top()
    assert not top.contains(top.cells(free_only=True).mean(dim=0))  # the old drop point: on the occupied middle
    half_diag = 0.12
    spots = D.TrayFloor.drop_spots(top, half_diag)
    assert len(spots) == D.MEASURE_DROP_SPOTS == 3
    assert all(top.contains(xy) for xy, _ in spots) and spots[0][1] >= half_diag
    assert [round(c, 3) for _, c in spots] == sorted((round(c, 3) for _, c in spots), reverse=True)
    for (a, _), (b, _) in itertools.combinations(spots, 2):
        assert float(th.linalg.norm(a - b)) >= half_diag
    empty = _ring_top()
    empty.occupied[:] = True
    assert len(D.TrayFloor.drop_spots(empty, half_diag)) == 1  # no free cell: the raster centre, as before


def test_tray_measurement_drop_tries_the_next_spot_after_a_failed_rest():
    """A drop that does not rest (the tray sank into a sink / tipped off a burner) parks the tray and drops it at the next spot;
    resting at none of them is the support rejection that lists every drop."""
    from types import SimpleNamespace

    from ispatialgym.sampling.sim_utils import Reject

    D = _disembedding()
    top = _ring_top()
    hull = _box(0.3, 0.2, 0.04)

    class World:
        spawned = {"target_0": None, "source_tray_0": None}

        def __init__(self):
            self.parks = []

        def park(self, obj, slot):
            self.parks.append(slot)

        def bottom_offset(self, hull, quat):
            return 0.02

        def activate(self, obj):
            pass

        def step_physics(self, n, still=()):
            pass

    class Tray:
        name = "source_tray_0"

        def __init__(self, sinks):
            self.sinks, self.drops, self.placed = sinks, 0, []

        def set_position_orientation(self, pos, quat):
            self.drops += 1
            self.placed.append(pos.clone())

        def get_position_orientation(self):
            p = self.placed[-1].clone()
            if self.drops <= self.sinks:
                p[2] -= 0.05
            return p, C.IDENTITY_QUAT.clone()

    spots = D.TrayFloor.drop_spots(top, 0.18)
    s = SimpleNamespace(world=World(), robot=None)
    tray = Tray(sinks=1)
    rp, rq, placed, where = D.TrayFloor.rest_drop(s, tray, top, hull, spots)
    assert tray.drops == 2 and th.allclose(placed[:2], spots[1][0]) and s.world.parks == [1]
    s2 = SimpleNamespace(world=World(), robot=None)
    with pytest.raises(Reject) as err:
        D.TrayFloor.rest_drop(s2, Tray(sinks=3), top, hull, spots)
    assert err.value.reason == "support" and "did not rest on the top at any of 3 drop spots" in str(err.value) and s2.world.parks == [1, 1]


def _box(dx, dy, dz):
    return th.tensor([[sx * dx / 2, sy * dy / 2, sz * dz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_disembedding_distractors_take_the_target_pose():
    """Handoff 2026-10-10: on the milk picks the distractors lie (smallest extent vertical: long axis horizontal, on the broadest
    side) or stand (largest extent vertical, the wider end down: a bottle on its base, not on its cap); a model already in that pose
    keeps its canonical orientation, and every other pick keeps the canonical one."""
    D = _disembedding()
    carton = _box(0.084, 0.123, 0.181)  # an upright carton (#04 lays it down)
    q, how = D.pose_rest_quat(carton, "lying")
    ext = TT.hull_extents(carton, q)
    assert float(ext[2]) == pytest.approx(0.084, abs=1e-5) and sorted(float(v) for v in ext[:2]) == pytest.approx([0.123, 0.181], abs=1e-5)
    assert how.startswith("lying: local")
    q, how = D.pose_rest_quat(_box(0.154, 0.058, 0.0585), "lying")  # a bottle lying in its canonical pose stays
    assert how == "lying: canonical" and th.allclose(q, C.IDENTITY_QUAT)
    assert D.pose_rest_quat(carton, "standing")[1] == "standing: canonical" and D.pose_rest_quat(carton, None)[1] == "canonical"
    # a lying bottle (long axis x): a 58 mm body ending in a 20 mm neck stands on its body end, whichever end of x that is
    body = [[x, sy * 0.029, sz * 0.029] for x in (-0.077, 0.03) for sy in (-1, 1) for sz in (-1, 1)]
    neck = [[0.077, sy * 0.01, sz * 0.01] for sy in (-1, 1) for sz in (-1, 1)]
    for flip in (1.0, -1.0):
        pts = th.tensor(body + neck) * th.tensor([flip, 1.0, 1.0])
        q, how = D.pose_rest_quat(pts, "standing")
        up = C.transform_points((th.zeros(3), q), pts)
        assert float(up[:, 2].max() - up[:, 2].min()) == pytest.approx(0.154, abs=1e-5)  # long axis vertical
        bottom = up[up[:, 2] < float(up[:, 2].min()) + 1e-4]
        assert float(bottom[:, 0].max() - bottom[:, 0].min()) == pytest.approx(0.058, abs=1e-5), how  # on the body, not the neck


def test_disembedding_figure_frame_and_layout_origins():
    """The embedded figure on the source tray (handoff 2026-10-10): +x = the robot's right (-tangent), +y = away from the robot
    (inward); each part's x / y from ``origin_in_figure_m`` (``origin_in_tray_m`` in the file's first version)."""
    D = _disembedding()
    x_axis, y_axis, yaw = D.figure_axes(th.tensor([0.0, 1.0]), th.tensor([1.0, 0.0]))  # facing world +x: the robot's left is +y
    assert x_axis.tolist() == [0.0, -1.0] and y_axis.tolist() == [1.0, 0.0] and yaw == pytest.approx(-math.pi / 2)
    assert C.quat_rotate(C.quat_from_yaw(yaw), th.tensor([1.0, 0.0, 0.0]))[:2].tolist() == pytest.approx(x_axis.tolist(), abs=1e-6)
    origins = D.layout_origins({"parts": {"isgefa": {"origin_in_figure_m": [-0.03065, 0.057516, 0.0]},
                                          "isgefb": {"origin_in_tray_m": [0.0, 0.0, 0.006]}, "other": {"role": "context"}}})
    assert origins == {"isgefa": (-0.03065, 0.057516), "isgefb": (0.0, 0.0)}
    assert D.FIXED_LAYOUT_RELEASE_GAP_M == pytest.approx(0.0005)


def test_phase3_target_fixed_pose_screens_and_places_only_that_pose(monkeypatch):
    """A fixed-layout target (the embedded figure's): F is screened at the given (xy, yaw) only, the object lands exactly there (no
    jitter, no offset from the focus) and F stays out of the band."""
    from types import SimpleNamespace

    calls = {}

    class IKC:
        def feasible_sets(self, cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of):
            calls["cells"], calls["yaws"] = [[float(v) for v in c] for c in cells], list(yaws)
            return {0: [(yaws[0], grasps[0])]}

        def exact(self, poses, stance, arm, **kw):
            return SimpleNamespace(ok=True, reason="", q=None)

    class Top:
        z = 0.8

        def polygon_supported(self, fp, margin=0.0):
            return True

    stored = []
    monkeypatch.setattr(TT, "store_feasible_set", lambda *a, **k: stored.append(a))
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda k, d=None: False if k == "grasp_twins" else d), ikc=IKC(), bands=None, kit=SimpleNamespace(stats={}),
                        world=SimpleNamespace(bottom_offset=lambda hull, q: 0.01, footprint=lambda hull, pose: th.zeros(4, 2)),
                        bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name="jigsaw_puzzle_piece.n.01_1")))
    manip = SimpleNamespace(focus=th.tensor([1.0, 2.0, 0.8]), translate=0.3, tilt=0.0, name="manipulation", working_arms=["right"])
    g = SimpleNamespace(name="g0", key="jigsaw_puzzle_piece__isgefa", pose=(th.zeros(3), C.IDENTITY_QUAT.clone()), finger_q=None)
    pose, rec = TT.phase3_target(s, manip, np.random.default_rng(0), [g], _box(0.1, 0.07, 0.02), C.IDENTITY_QUAT.clone(), [], top=Top(), support_z=0.8,
                                 fixed=(th.tensor([1.05, 1.95]), 0.3))
    assert calls["cells"] == [pytest.approx([1.05, 1.95])] and calls["yaws"] == [0.3] and stored == []
    assert pose[0][:2].tolist() == pytest.approx([1.05, 1.95]) and rec.yaw == pytest.approx(0.3) and rec.grasp_name == "g0"
    assert th.allclose(pose[1], C.quat_from_yaw(0.3), atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------
# sidecar blocks the tabletop tasks emit, read back by the F6 sidecar / programs (schema compatibility, no simulator)
# ----------------------------------------------------------------------------------------------------------------


def _pose(x, y, z, quat=None):
    return th.tensor([x, y, z]), (C.IDENTITY_QUAT.clone() if quat is None else quat)


def _draft(alias, family):
    from ispatialgym.sampling.v2.stances import EdgeFrame, SidecarDraft, Stance, Transition

    sd = SidecarDraft(alias, family)
    edge = EdgeFrame("e0", th.tensor([1.0, 0.0]), th.tensor([1.0, 0.0]), 1.6)
    init = Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
    manip = Stance("manipulation", (0.65, 0.0, 0.0), 0.75, -0.3, 0.35, edge, focus=th.tensor([1.0, 0.0, 0.85]), arms=["right"])
    sd.add_stance(init)
    sd.add_stance(manip)
    sd.add_transition(Transition("init", "manipulation", 0.55))
    return sd, init, manip


def _f6(sd):
    from ispatialgym.oracle.sidecar import Sidecar

    d = sd.to_dict()
    d["tro_sha256"] = "0" * 64
    return Sidecar.from_dict(d)


def test_disembedding_sidecar_blocks_expand():
    from ispatialgym.sampling.v2.stances import GraspRecord, PathRecord

    sd, init, manip = _draft("disembedding-box_of_cereal-box_of_cereal_2-breakfast_table", "disembedding")
    place_st = manip.shifted(lateral=0.4, name="placement")
    place_st.focus = th.tensor([1.0, 0.4, 0.85])
    sd.add_stance(place_st)
    down = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    g = GraspRecord("box_of_cereal.n.01_1", "right", "box_of_cereal__iucshm", "grasp_0", 0.3, pre_grasp=_pose(1.0, 0.0, 1.05, down), grasp=_pose(1.0, 0.0, 0.95, down),
                    pre_place=_pose(1.0, 0.4, 1.05, down), place=_pose(1.0, 0.4, 0.95, down), carry_height=0.15, eef_in_object=_pose(0, 0, 0.1, down), place_on="tray.n.01_2")
    sd.add_grasp(g)
    sd.add_path(PathRecord("manipulation", "placement", "local", [(0.65, 0.0, 0.0), (0.65, 0.4, 0.0)], 0.4, carried="box_of_cereal.n.01_1"))
    for n, name in (("box_of_cereal.n.01_1", "box of cereal"), ("tray.n.01_1", "source tray"), ("tray.n.01_2", "destination tray")):
        sd.describe(n, name)
    sc = _f6(sd)
    assert sc.problems(check_program=True) == []
    from ispatialgym.oracle.programs import expand

    names = [a.name for a in expand(sc.alias, sc)]
    assert names == ["transition", "grasp", "lift", "navigate", "place", "retreat"]


def test_axial_and_attachment_contact_blocks_expand():
    from ispatialgym.sampling.v2.stances import ContactRecord, GraspRecord

    # axial: push (right, arc with mid) + stop-edge brace (left) on the same target
    sd, init, manip = _draft("alignment-axial-keyboard-breakfast_table", "alignment")
    manip.arms = ["right", "left"]
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))
    d = th.tensor([0.0, 1.0, 0.0])
    sd.add_contact(ContactRecord("push", "right", _pose(1.2, -0.2, 1.0, q), _pose(1.2, -0.2, 0.9, q), _pose(1.21, -0.1, 0.9, q), _pose(1.2, 0.0, 0.9, q), d, 0.2, object="keyboard.n.01_1"))
    sd.add_contact(ContactRecord("brace", "left", _pose(0.8, 0.4, 1.0, q), _pose(0.8, 0.4, 0.9, q), direction=-d, brace_kind="stop_edge", object="keyboard.n.01_1"))
    sd.describe("keyboard.n.01_1", "keyboard")
    sc = _f6(sd)
    assert sc.problems(check_program=True) == []
    # attachment: grasp with SLERP waypoints + pre-attach / attach + partner brace
    sd, init, manip = _draft("configuration_matching-attachment-cap-lemonade_bottle-breakfast_table", "configuration_matching")
    manip.arms = ["right", "left"]
    g = GraspRecord("cap.n.01_1", "right", "cap__xxqyrt", "grasp_1", 0.0, pre_grasp=_pose(1.0, 0.0, 1.05, q), grasp=_pose(1.0, 0.0, 0.95, q), pre_place=_pose(1.0, 0.3, 1.15, q),
                    place=_pose(1.0, 0.3, 1.05, q), carry_height=0.12, eef_in_object=_pose(0, 0, 0.05, q), place_on="lemonade_bottle.n.01_1",
                    waypoints=[_pose(1.0, 0.1, 1.1, q), _pose(1.0, 0.2, 1.12, q), _pose(1.0, 0.3, 1.15, q)])
    sd.add_grasp(g)
    sd.add_contact(ContactRecord("brace", "left", _pose(1.0, 0.5, 0.95, q), _pose(1.0, 0.4, 0.95, q), direction=th.tensor([0.0, -1.0, 0.0]), brace_kind="partner_face",
                                 object="lemonade_bottle.n.01_1"))
    sd.describe("cap.n.01_1", "cap")
    sd.describe("lemonade_bottle.n.01_1", "lemonade bottle")
    assert _f6(sd).problems(check_program=True) == []


def test_block_and_articulation_blocks_expand():
    from ispatialgym.sampling.v2.stances import ContactRecord, pose_to_sidecar

    sd, init, manip = _draft("configuration_matching-block_design-block_4-breakfast_table", "configuration_matching")
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))
    blocks = []
    for i in range(4):
        rec = {"instance": f"toy_dice.n.01_{i + 1}", "order": i, "face_pair": "y", "tilt_pick_deg": -45.0 if i else 0.0, "tilt_place_deg": 45.0 if i else 0.0, "yaw_deg": 10.0,
               "yaw_align_deg": 12.0, "flip_axis": [0.0, 1.0, 0.0] if i else None, "flip_sign": 1 if i else 0, "pick": pose_to_sidecar(_pose(1.0, 0.1 * i, 0.9, q)),
               "place": pose_to_sidecar(_pose(1.0, 0.0, 0.9, q)), "carry_height": 0.08, "pre_pick": pose_to_sidecar(_pose(1.0, 0.1 * i, 1.0, q))}
        if i % 2 == 1:
            rec["push_row"] = ContactRecord("push", "right", _pose(1.0, 0.1, 0.95, q), _pose(1.0, 0.05, 0.9, q), None, _pose(1.0, 0.045, 0.9, q), th.tensor([0.0, -1.0, 0.0]), 0.005,
                                            object=rec["instance"]).to_sidecar()
            rec["brace_row"] = ContactRecord("brace", "left", _pose(1.0, -0.1, 0.95, q), _pose(1.0, -0.05, 0.9, q), direction=th.tensor([0.0, 1.0, 0.0]), brace_kind="row_left",
                                             object="toy_dice.n.01_1").to_sidecar()
        blocks.append(rec)
        sd.describe(rec["instance"], "block")
    sd.blocks = blocks
    sc = _f6(sd)
    assert sc.problems(check_program=True) == []
    # small-scale articulation: brace + handle grasp + arm-mode follow
    sd, init, manip = _draft("configuration_matching-articulation_open_small_scale-toolbox-breakfast_table", "configuration_matching")
    manip.arms = ["right"]
    wp = {"contact": [1.0, 0.0, 1.0], "approach": [0.0, 0.0, -1.0], "pinch": [1.0, 0.0, 0.0]}
    sd.articulation = {"object": "toolbox.n.01_1", "joint": "j_lid", "joint_type": "revolute", "q_init": 0.0, "q_goal": 1.5,
                       "waypoints": {"0.0": wp, "0.25": wp, "0.5": wp, "0.75": wp, "1": wp}, "handle_reach": {"0.25": True, "0.5": True, "0.75": False, "1": False},
                       "passive_from": 0.5, "follow_mode": "arm"}
    sd.add_contact(ContactRecord("brace", "left", _pose(1.0, 0.3, 1.0, q), _pose(1.0, 0.2, 0.9, q), direction=th.tensor([0.0, -1.0, 0.0]), brace_kind="opposite_face",
                                 object="toolbox.n.01_1"))
    sd.describe("toolbox.n.01_1", "toolbox")
    assert _f6(sd).problems(check_program=True) == []


# ----------------------------------------------------------------------------------------------------------------
# A1 joint two-arm placement (owner decision 2026-09-16): synthetic right / left bands, no simulator
# ----------------------------------------------------------------------------------------------------------------


def _synthetic_band(cells_by_trunk, classes_by_trunk=None, class_names=None, lattice=0.08, arm="right"):
    """A ``ReachBands`` with the given base-frame cells per trunk (x ahead, lateral + = left) at a stance of yaw 0."""
    from ispatialgym.sampling.reach_bands import ReachBands

    meta = {"scene": "s", "support": "t", "lattice": lattice, "arm": arm, "stance": {"base_pose2d": [0.0, 0.0, 0.0]},
            "trunks": [{"translate": k[0], "tilt": k[1], "collides": False} for k in cells_by_trunk]}
    if class_names:
        meta["classes"] = list(class_names)
    cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in cells_by_trunk.items()}
    pairs = {k: np.zeros((0, 3), dtype=np.int32) for k in cells}
    if classes_by_trunk is not None:
        try:
            return ReachBands(meta, cells, pairs, None, {k: np.asarray(v, dtype=np.int64) for k, v in classes_by_trunk.items()})
        except TypeError:  # loader before B3's classes field: attach the masks the way ``cell_classes`` reads them
            rb = ReachBands(meta, cells, pairs, None)
            rb.classes = {k: np.asarray(v, dtype=np.int64) for k, v in classes_by_trunk.items()}
            return rb
    return ReachBands(meta, cells, pairs, None)


def _stance(translate, tilt=0.0, yaw=0.0, xy=(0.0, 0.0)):
    from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

    edge = EdgeFrame("e0", th.tensor([float(xy[0]) + 0.45 * math.cos(yaw), float(xy[1]) + 0.45 * math.sin(yaw)]), th.tensor([math.cos(yaw), math.sin(yaw)]), 1.6)
    return Stance("manipulation", (float(xy[0]), float(xy[1]), float(yaw)), translate, tilt, 0.45, edge, arms=["right", "left"])


def test_height_class_uses_two_centimetre_bins():
    assert TT.height_class(0.049) == pytest.approx(0.04)
    assert TT.height_class(0.051) == pytest.approx(0.06)
    assert TT.height_class(0.002) == pytest.approx(0.02)  # never the empty class
    assert TT.contact_height(0.81, 0.10) == pytest.approx(0.81 + 0.75 * 0.10)


def test_joint_placement_requires_both_cells_at_the_same_trunk():
    right = _synthetic_band({(0.5, 0.0): [[0.60, -0.20], [0.68, -0.20]], (0.25, 0.0): [[0.60, 0.00]]})
    left = _synthetic_band({(0.5, 0.0): [[0.60, 0.20]]}, arm="left")
    st = _stance(0.5)
    ok = TT.joint_placement(right, left, st, work_xy=(0.61, -0.21), brace_xy=(0.59, 0.21))
    assert ok["ok"] and ok["work_ok"] and ok["brace_ok"] and ok["left_band"]
    assert np.allclose(ok["work_cell"], [0.60, -0.20]) and np.allclose(ok["brace_cell"], [0.60, 0.20])
    # the brace point outside the half-cell tolerance of every left cell
    assert not TT.joint_placement(right, left, st, (0.61, -0.21), (0.60, 0.30))["ok"]
    # the same points at another trunk: the right band has a cell there, the left band none -> rejected as a pair
    st2 = _stance(0.25)
    r2 = TT.joint_placement(right, left, st2, (0.60, 0.0), (0.60, 0.20))
    assert r2["work_ok"] and not r2["brace_ok"] and not r2["ok"]
    # a rotated stance: the band cells follow the base frame
    st3 = _stance(0.5, yaw=math.pi / 2, xy=(1.0, 1.0))
    r3 = TT.joint_placement(right, left, st3, work_xy=(1.20, 1.60), brace_xy=(0.80, 1.60))
    assert r3["ok"]
    # no left band yet: the brace side is left to the exact two-arm IK behind a reach proxy, flagged for the record
    r4 = TT.joint_placement(right, None, st, (0.61, -0.21), (0.70, 0.30))
    assert r4["ok"] and not r4["left_band"] and r4["brace_cell"] is None
    assert not TT.joint_placement(right, None, st, (0.61, -0.21), (1.05, 0.10))["brace_ok"]  # a far-side stop point, beyond every band
    assert not TT.joint_placement(right, None, st, (0.61, -0.21), (0.60, -0.80))["brace_ok"]


def test_joint_placement_checks_the_brace_class_when_the_band_records_classes():
    names = ["near_v", "near_h", "far_v", "far_h", "left_v", "left_h", "right_v", "right_h", "top_y0", "top_y90"]
    right = _synthetic_band({(0.5, 0.0): [[0.60, -0.20]]})
    # the single left cell passed near_v (bit 0) and left_h (bit 5) only
    left = _synthetic_band({(0.5, 0.0): [[0.60, 0.20]]}, {(0.5, 0.0): [(1 << 0) | (1 << 5)]}, names, arm="left")
    st = _stance(0.5)
    assert TT.cell_classes(left, 0.5, 0.0, [0.60, 0.20]) == ["near_v", "left_h"]
    assert TT.cell_classes(left, 0.5, 0.0, [0.60, 0.50]) is None  # not a band cell
    assert TT.cell_classes(right, 0.5, 0.0, [0.60, -0.20]) is None  # no mask recorded
    assert TT.joint_placement(right, left, st, (0.60, -0.20), (0.60, 0.20), brace_class="near_v")["ok"]
    r = TT.joint_placement(right, left, st, (0.60, -0.20), (0.60, 0.20), brace_class="far_v")
    assert r["brace_ok"] and not r["brace_class_ok"] and not r["ok"] and r["brace_classes"] == ["near_v", "left_h"]
    # a band without masks accepts any class
    plain = _synthetic_band({(0.5, 0.0): [[0.60, 0.20]]}, arm="left")
    assert TT.joint_placement(right, plain, st, (0.60, -0.20), (0.60, 0.20), brace_class="far_v")["ok"]


def test_contact_class_names_follow_the_base_frame():
    st = _stance(0.5)  # facing +x, robot's left = +y
    up, fwd, left = th.tensor([0.0, 0.0, 1.0]), th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 1.0, 0.0])
    assert TT.contact_class(st, fwd, up) == "near_v"  # pads on the face toward the robot, pushing away, fingers stacked
    assert TT.contact_class(st, -fwd, left) == "far_h"
    assert TT.contact_class(st, -left, up) == "left_v"  # the hand on the object's robot-left face pushing to the right
    assert TT.contact_class(st, left, fwd) == "right_h"
    assert TT.contact_class(st, -up, left) == "top_y0" and TT.contact_class(st, -up, fwd) == "top_y90"
    assert TT.contact_class(st, C.unit(fwd - up), up) is None  # a 45° approach is no contact class
    st2 = _stance(0.5, yaw=math.pi / 2)  # facing +y: the robot's left is -x
    assert TT.contact_class(st2, th.tensor([0.0, 1.0, 0.0]), up) == "near_v" and TT.contact_class(st2, th.tensor([1.0, 0.0, 0.0]), up) == "left_v"


def test_approach_pitch_order_prefers_the_passing_pitched_classes():
    assert TT.approach_pitch_order(None) == [0.0, 30.0, 45.0, 60.0]
    assert TT.approach_pitch_order(["p45_y0", "p60_y90"]) == [45.0, 60.0, 0.0, 30.0]
    assert TT.approach_pitch_order(["p00_y0", "p00_y90", "p30_y0"]) == [0.0, 30.0, 45.0, 60.0]
    assert TT.approach_pitch_order(["near_v"]) == [0.0, 30.0, 45.0, 60.0]  # contact names carry no pitch
    assert TT.class_pitch_deg("p30_y90") == 30.0 and TT.class_pitch_deg("top_y0") is None
    a = TT.pitched_approach(60.0, th.tensor([1.0, 0.0]))
    assert a[2] == pytest.approx(-math.cos(math.radians(60))) and a[0] == pytest.approx(math.sin(math.radians(60)))


def test_union_extent_feeds_the_margins_table():
    right = _synthetic_band({(0.5, 0.0): [[0.60, -0.56], [0.60, 0.00]]})
    left = _synthetic_band({(0.5, 0.0): [[0.60, 0.16], [0.60, 0.56]]}, arm="left")
    assert TT.union_extent(right) == (pytest.approx(-0.56), pytest.approx(0.0))
    assert TT.union_extent(right, left) == (pytest.approx(-0.56), pytest.approx(0.56))
    assert TT.union_extent(right, None) == (pytest.approx(-0.56), pytest.approx(0.0))
    assert TT.union_extent(None) is None
    # §4.4 axial row over the union: both margins grow to |λ| + w_diag + 0.02
    lo, hi = TT.union_extent(right, left)
    m_R, m_L, info = TT.lateral_margins({"kind": "axial", "long": 0.30, "short": 0.10}, lo, hi, 0.5, 0.5)
    assert m_R == pytest.approx(0.56 + info["w_diag"] + 0.02) and m_L == pytest.approx(0.56 + info["w_diag"] + 0.02)
    m_R1, m_L1, _ = TT.lateral_margins({"kind": "axial", "long": 0.30, "short": 0.10}, *TT.union_extent(right), 0.5, 0.5)
    assert m_R1 == pytest.approx(m_R) and m_L1 < m_L


def test_block_record_extras_survive_the_f6_sidecar():
    """Block follow-ups: ``yaw_align_deg`` on every record and the root->centre ``centre_offset`` (isgblk root off-centre)
    reach the oracle through ``BlockRecord.extra``; the brace's ``pose_class`` rides along in the contact record."""
    from ispatialgym.sampling.v2.stances import ContactRecord, pose_to_sidecar

    sd, init, manip = _draft("configuration_matching-block_design-block_4-breakfast_table", "configuration_matching")
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    rec = {"instance": "toy_dice.n.01_1", "order": 0, "face_pair": "x", "tilt_pick_deg": 0.0, "tilt_place_deg": 0.0, "yaw_deg": 90.0, "yaw_align_deg": 0.0,
           "flip_axis": None, "flip_sign": 0, "pick": pose_to_sidecar(_pose(1.0, 0.0, 0.9, q)), "pre_pick": pose_to_sidecar(_pose(1.0, 0.0, 1.0, q)),
           "place": pose_to_sidecar(_pose(1.0, 0.1, 0.9, q)), "pre_place": pose_to_sidecar(_pose(1.0, 0.1, 1.0, q)), "carry_height": 0.08, "goal_face": "+z", "slot": 0,
           "centre_offset": [0.0, 0.0, 0.0125]}
    rec["brace_row"] = ContactRecord("brace", "left", _pose(1.0, -0.1, 0.95, q), _pose(1.0, -0.05, 0.9, q), direction=th.tensor([0.0, 1.0, 0.0]), brace_kind="row_left",
                                     object="toy_dice.n.01_2").to_sidecar()
    rec["brace_row"]["pose_class"] = "side_v"
    rec["push_row"] = ContactRecord("push", "right", _pose(1.0, 0.1, 0.95, q), _pose(1.0, 0.05, 0.9, q), None, _pose(1.0, 0.045, 0.9, q), th.tensor([0.0, -1.0, 0.0]), 0.005,
                                    object="toy_dice.n.01_1").to_sidecar()
    sd.blocks = [rec]
    sd.describe("toy_dice.n.01_1", "block")
    sd.describe("toy_dice.n.01_2", "block")
    sc = _f6(sd)
    assert sc.problems(check_program=True) == []
    b = sc.blocks[0]
    assert b.extra["centre_offset"] == [0.0, 0.0, 0.0125] and b.extra["yaw_align_deg"] == 0.0
    from ispatialgym.oracle.actions import yaw_align_deg, block_reorients

    assert yaw_align_deg(b) == 0.0 and not block_reorients(b)
    assert b.brace_row.extra["pose_class"] == "side_v"


# ----------------------------------------------------------------------------------------------------------------
# closed-finger profile, blade stand-offs and the swept-region test (owner review of the axial demo, 2026-09-16)
# ----------------------------------------------------------------------------------------------------------------


def test_finger_profile_and_blade_standoff():
    assert TT.finger_half_thickness(0.0) == pytest.approx(0.0084) and TT.finger_half_thickness(0.03) == pytest.approx(0.0084)
    assert TT.finger_half_thickness(0.031) == pytest.approx(0.018) and TT.finger_half_thickness(0.07) == pytest.approx(0.018)
    assert TT.finger_half_thickness(0.08) is None  # the palm / mounting bracket
    assert TT.finger_half_thickness(0.0, side=True) == pytest.approx(0.007)
    top = 0.8125
    # the pilot board game (2.5 cm tall) with the fingertips 11.3 mm above the top: the face stays in the tip section
    assert TT.blade_standoff((top, top + 0.0252), top + 0.0113, 0.0, 0.005) == (pytest.approx(0.0134), pytest.approx(0.0084))
    # a 6 cm object reaches the 18 mm section: the blade moves outward to 23 mm
    assert TT.blade_standoff((top, top + 0.06), top + 0.003, 0.0, 0.005) == (pytest.approx(0.023), pytest.approx(0.018))
    # pitched 45°: the finger-axis distance grows by 1/cos -> the same 6 cm face reaches the palm -> no blade
    assert TT.blade_standoff((top, top + 0.06), top + 0.003, 45.0, 0.005) is None
    assert TT.blade_standoff((top, top + 0.10), top + 0.003, 0.0, 0.005) is None
    assert TT.blade_standoff((top, top + 0.002), top + 0.003, 0.0, 0.005) is None  # face entirely below the tips
    # the 2.5 cm cube push (contact, no clearance): 8.4 mm pad face, 7 mm finger side
    assert TT.blade_standoff((top, top + 0.025), top + 0.003, 0.0, 0.0)[0] == pytest.approx(0.0084)
    assert TT.blade_standoff((top, top + 0.025), top + 0.003, 0.0, 0.0, side=True)[0] == pytest.approx(0.007)


def test_blade_sweep_clear_rejects_a_corner_stop_and_accepts_a_face_stop():
    # a 0.255 x 0.133 box (the pilot board game) rotating CCW by 57° about its centre toward the goal (long axis along x)
    hull = th.tensor([[x, y, z] for x in (-0.1275, 0.1275) for y in (-0.0667, 0.0667) for z in (0.0, 0.025)])
    delta = math.radians(57.0)
    pose_init = (th.tensor([1.0, 0.0, 0.8]), C.quat_from_yaw(-delta))  # rotated back from the goal by δ
    com = th.tensor([1.0, 0.0])
    # CCW toward the goal: the left end (x < 0) descends onto the near long face's line (y = -0.0667, d_stop = -y) — a blade
    # 13.4 mm outside that face near the descending end is met only at the goal ...
    ok, gap = TT.blade_sweep_clear(hull, pose_init, com, delta, th.tensor([1.0 - 0.10, -0.0667 - 0.0134]), 0.0084, 0.005)
    assert ok and gap >= 0.005
    # ... while the face's middle and its other half rise through the blade's position before the goal (the pilot's
    # blocked demo: the blade stood on the swept part of the face)
    for x in (0.0, 0.06, 0.1275):
        ok_x, gap_x = TT.blade_sweep_clear(hull, pose_init, com, delta, th.tensor([1.0 + x, -0.0667 - 0.0134]), 0.0084, 0.005)
        assert not ok_x and gap_x < 0.005
    # the mirror case (CW rotation) frees the other end
    pose_cw = (th.tensor([1.0, 0.0, 0.8]), C.quat_from_yaw(delta))
    assert TT.blade_sweep_clear(hull, pose_cw, com, -delta, th.tensor([1.0 + 0.10, -0.0667 - 0.0134]), 0.0084, 0.005)[0]
    assert not TT.blade_sweep_clear(hull, pose_cw, com, -delta, th.tensor([1.0 - 0.10, -0.0667 - 0.0134]), 0.0084, 0.005)[0]
    # a blade far from the object never meets it
    assert TT.blade_sweep_clear(hull, pose_init, com, delta, th.tensor([1.0, -0.5]), 0.0084, 0.005)[0]


def test_brace_record_extras_reach_the_f6_sidecar():
    sd, init, manip = _draft("alignment-axial-board_game-breakfast_table", "alignment")
    manip.arms = ["right", "left"]
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))
    from ispatialgym.sampling.v2.stances import ContactRecord

    sd.add_contact(ContactRecord("push", "right", _pose(1.0, -0.1, 0.95, q), _pose(1.0, -0.1, 0.84, q), _pose(1.02, -0.05, 0.84, q), _pose(1.05, 0.0, 0.84, q),
                                 th.tensor([0.0, 1.0, 0.0]), 0.1, object="board_game.n.01_1"))
    sd.add_contact(TT.BraceRecord("brace", "left", _pose(1.0, 0.2, 0.95, q), _pose(1.0, 0.2, 0.84, q), direction=th.tensor([0.0, -1.0, 0.0]), brace_kind="stop_edge",
                                  object="board_game.n.01_1", extras={"blade_half_thickness_m": 0.0084, "stop_clearance_m": 0.005, "pose_class": "blade_0", "skip": None}))
    sd.describe("board_game.n.01_1", "board game")
    d = sd.to_dict()
    assert d["contacts"]["brace"]["blade_half_thickness_m"] == 0.0084 and d["contacts"]["brace"]["stop_clearance_m"] == 0.005 and "skip" not in d["contacts"]["brace"]
    sc = _f6(sd)
    assert sc.problems(check_program=True) == []
    assert sc.contacts.brace.extra["blade_half_thickness_m"] == 0.0084 and sc.contacts.brace.extra["pose_class"] == "blade_0"


def test_table_flip_yaw_first_decomposition_exists_for_every_goal_face():
    """§6.5 flip-on-the-table prep (owner 2026-09-16 17:10): two quarter turns about different horizontal axes compose to a
    120° body-diagonal rotation, so the two-leg chain is yaw-first (ψ + k·90° in hand, placed on a free cell) followed by
    ONE quarter turn about a horizontal table axis that brings the goal face up — such a pair exists for every goal side
    face and init yaw jitter, and the result is table-aligned with the goal face up."""
    import math
    import torch as th
    from ispatialgym.robot import conventions as C

    for gf in ("+x", "-x", "+y", "-y"):
        for jitter_deg in (0.0, 20.0, -15.0):
            q_init = C.quat_from_yaw(math.radians(jitter_deg))
            plan = TT.cube_flip_plan(q_init, gf, 0.0, 45.0)
            psi = math.radians(plan["yaw_deg"])
            found = []
            for k in (1, 2, 3):
                q_mid = C.qmul(C.quat_from_yaw(psi + k * math.pi / 2), q_init)
                assert TT.up_face(q_mid) == TT.up_face(q_init)  # the yaw leg keeps the block on its face
                for ax in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
                    for sgn in (1.0, -1.0):
                        q_end = C.qmul(C.quat_from_axis_angle(ax, sgn * math.pi / 2), q_mid)
                        if TT.up_face(q_end) == gf:
                            found.append((k, ax, sgn))
                            for face in ("+x", "+y", "+z"):
                                v = C.quat_rotate(q_end, th.tensor(TT.FACE_AXES[face]))
                                assert max(abs(float(c)) for c in v) > 0.999  # table-aligned
            assert len(found) == 3, (gf, jitter_deg, found)


# ----------------------------------------------------------------------------------------------------------------
# owner 2026-09-24 regressions 3-5: pinch re-centring only when the pinch geometry survives it; the partner top under
# the fingertips; the male's rest orientation by stability
# ----------------------------------------------------------------------------------------------------------------


def _ring(r, z, n=36):
    return [[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z] for k in range(n)]


def _eef(pos, a, c):
    """An eef pose (object frame) from its origin, approach axis ``a`` (+z of the eef) and closing axis ``c`` (+y)."""
    a, c = th.tensor(a, dtype=th.float32), th.tensor(c, dtype=th.float32)
    a, c = a / a.norm(), c / c.norm()
    x = th.cross(c, a, dim=0)
    R = th.stack([x, c, a], dim=1)
    return th.tensor(pos, dtype=th.float32), C.mat_to_quat(R)


def _iors_stub(hull, lib_width=None, attachable=True):
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import iors_small as IS

    class Stub:
        pass

    s = Stub()
    s.hull = {"target": th.tensor(hull, dtype=th.float32)}
    s.gen = lambda key, default=None: default
    s.grasp_key = "test"
    s.ikc = SimpleNamespace(attachable=lambda tgt, pose, arm: attachable)
    s.template_facts = {}
    s.pinch_offset = lambda pose: IS.IORSSmall.pinch_offset(s, pose)
    s.pinch_chord = lambda pose: IS.IORSSmall.pinch_chord(s, pose)
    return s, IS


class _G:
    """A grasp-library row stand-in: ``pose`` = (position, quat) in the object frame, ``meta`` the library facts."""

    def __init__(self, name, pos, quat, width=None):
        self.name, self.position, self.quat = name, pos, quat
        self.meta = {} if width is None else {"width": width}

    @property
    def pose(self):
        return self.position, self.quat


def test_recentre_keeps_a_rim_pinch_of_a_thin_walled_cup_and_recentres_a_solid_marker():
    # a cup: inner and outer walls (collision decomposition boundary points), rim at z = 0.08
    cup = _ring(0.035, 0.0) + _ring(0.035, 0.08) + _ring(0.030, 0.01) + _ring(0.030, 0.08)
    s, IS = _iors_stub(cup, lib_width=0.007)
    # a rim pinch: the eef 3 cm above the rim at the +x wall, approach down, closing along x (across the wall)
    pos, quat = _eef([0.0325, 0.0, 0.08 + 0.01], a=[0, 0, -1], c=[1, 0, 0])  # the eef 1 cm above the rim: the tips reach 9 mm down the wall
    g = _G("grasp_4", pos, quat, width=0.007)
    off = IS.IORSSmall.pinch_offset(s, g.pose)
    assert abs(off - 0.0325) < 0.003, off  # the pinch sits at the wall, 32.5 mm off the cup's centre along c
    out, offsets = IS.IORSSmall.recentre_pinches(s, [g], object())
    assert out[0] is g, "the rim pinch keeps its offset"
    fact = s.template_facts["grasp_pinch_recentre"]["grasp_4"]
    assert fact["applied"] is False and fact["chord_recentred_m"] > 0.06 and "different width" in fact["reason"]
    assert abs(offsets["grasp_4"] - off) < 1e-6
    # a solid marker lying along x (a sparse prism hull: vertices at the end rings only), r = 8 mm, pinched from above across y
    marker = [[x, 0.008 * math.cos(t), 0.008 * math.sin(t) + 0.008] for x in (-0.06, 0.06) for t in [2 * math.pi * k / 16 for k in range(16)]]
    s2, _ = _iors_stub(marker, lib_width=0.016)
    pos, quat = _eef([0.0, 0.015, 0.016 + 0.01], a=[0, 0, -1], c=[0, 1, 0])  # the eef 1 cm above the marker, the pinch centre 15 mm off the axis along c
    g2 = _G("grasp_1", pos, quat, width=0.016)
    assert abs(IS.IORSSmall.pinch_offset(s2, g2.pose) - 0.015) < 1e-3
    out, _ = IS.IORSSmall.recentre_pinches(s2, [g2], object())
    assert out[0] is not g2 and abs(float(out[0].position[1])) < 1e-4, "the solid body is re-centred onto its axis"
    assert s2.template_facts["grasp_pinch_recentre"]["grasp_1"]["applied"] is True
    # a pinch already within the limit is untouched and unlisted
    pos, quat = _eef([0.0, 0.002, 0.026], a=[0, 0, -1], c=[0, 1, 0])
    g3 = _G("grasp_2", pos, quat, width=0.016)
    out, _ = IS.IORSSmall.recentre_pinches(s2, [g3], object())
    assert out[0] is g3 and "grasp_2" not in s2.template_facts["grasp_pinch_recentre"]


def test_rejected_manipulation_candidates_move_down_the_order_in_later_attempts():
    """Owner 2026-09-25 (cap 0925-220515): 45 survivors, the same 6 best-scored stances in every attempt. A rejection collected in an
    earlier attempt adds CANDIDATE_REJECT_PENALTY to the candidate's score, so the next attempt's window of six reaches new stances;
    the untried ones keep the depth / trunk order."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import attachment as A

    def cand(depth, translate=0.5, tilt=-0.45):
        return SimpleNamespace(focus=th.tensor([depth, 0.0, 0.8]), translate=translate, tilt=tilt, world_to_base=lambda xy: th.tensor([float(xy[0]), 0.0]))

    s = SimpleNamespace(manip_candidates=[cand(0.60), cand(0.61), cand(0.62), cand(0.70), cand(0.75), cand(0.80), cand(0.85), cand(0.90), cand(0.95)])
    rng = np.random.default_rng(0)
    first = A.Attachment._candidate_order(s, rng, {})
    assert set(first[:3]) == {0, 1, 2} and first[3:6] == [3, 4, 5]  # the arm-friendly depths first (ties by the draw), then by depth
    rejects = {k: 1 for k in first[:6]}  # every stance of the first window was rejected
    second = A.Attachment._candidate_order(s, np.random.default_rng(1), rejects)
    assert second[:3] == [6, 7, 8], second  # the untried stances come first now
    assert set(second[3:6]) == {0, 1, 2}  # the rejected ones follow in their own preference order
    twice = dict(rejects)
    twice[6] = 2
    third = A.Attachment._candidate_order(s, np.random.default_rng(2), twice)
    assert third[:2] == [7, 8] and third[-1] == 6, third  # two rejections rank below one


def test_a_grasp_lift_a_floor_demands_is_not_optional():
    """Owner 2026-09-25 (cork 0925-195316 / 205211): the partner-top floor asked for a 31 mm lift, the lifted grasp had no IK
    (right_arm_joint5 near its limit) and the sampler kept the library height - the fingertips ended 26 mm below the rim and the
    demo's insertion jammed on the neck. Now: an IK failure with a lift a FLOOR demands rejects the candidate (``lift_unreachable``);
    without a floor (the lift is only ``generation.grasp_height_extra_m``) the library height is still kept."""
    from types import SimpleNamespace

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.tasks import attachment as A

    cork = th.tensor(_ring(0.009, 0.0) + _ring(0.009, 0.04), dtype=th.float32)  # 40 mm tall, resting on the support at z = 0
    m_pose = (th.zeros(3), C.IDENTITY_QUAT.clone())

    def stub(extra_m, clearance):
        s = SimpleNamespace()
        s.gp = SimpleNamespace(alias="cork-test")
        s.gen = lambda key, default=None: extra_m if key == "grasp_height_extra_m" else default
        s.hull, s.t_role = {"male": cork}, "male"
        s.world = SimpleNamespace(hull_at=lambda hull, pose: hull.clone())
        s.counters = {}
        s.choice = SimpleNamespace(models={"male": SimpleNamespace(scale=1.0)})
        s.obj = lambda role: "support-object"
        s._finger_lowest_z = lambda eef: None  # no robot meshes needed here: the partner floor is the one under test
        s._attach_clearance = lambda grasp: clearance
        s._contact_band_centre = lambda eef, pose: None
        s.ikc = SimpleNamespace(exact=lambda *a, **k: SimpleNamespace(ok=False, reason="ik_joint_limit: right_arm_joint5 within 34 mrad of its limit"))
        return s

    def grasp_at(tip_above_bottom):
        # a top-down pinch: the fingertip axis point ``tip_above_bottom`` above the cork's bottom
        pos, quat = _eef([0.0, 0.0, tip_above_bottom + C.FINGERTIP_LENGTH], a=[0, 0, -1], c=[1, 0, 0])
        return SimpleNamespace(eef_in_object=(pos.clone(), quat.clone()), grasp=(pos.clone(), quat.clone()), pre_grasp=None, extras={}, arm="right")

    # the tips 2 mm above the bottom; at the attach they would end 25.9 mm below the rim: the floor asks for 30.9 mm, the IK fails
    floor_case = {"lift_sign": 1.0, "h_tip_m": -0.0259, "h_pad_m": -0.0198, "depth_m": 0.0184, "r_male_m": 0.009, "r_tips_m": [0.009, 0.009],
                  "pinch_width_m": 0.017, "partner_points_under_tips": 7, "depth_within_male_radius_m": 0.0184, "depth_under_tips_m": None,
                  "pair_class": "into_opening", "floor_rule": "rim floor", "r_rim_m": 0.013, "r_opening_m": 0.0095, "male_length_m": 0.04}
    s = stub(0.0, floor_case)
    with pytest.raises(Reject) as ei:
        A.Attachment._apply_grasp_height_extra(s, grasp_at(0.002), m_pose, None)
    assert ei.value.reason == "lift_unreachable" and "partner top (fingertips)" in str(ei.value) and s.counters["lift_unreachable"] == 1
    # no floor (the tips clear the rim by 2 cm at the attach), the lift is only the generation knob: the library height is kept
    clear_case = dict(floor_case, h_tip_m=0.02, h_pad_m=0.013)
    s = stub(0.01, clear_case)
    g = grasp_at(0.02)
    A.Attachment._apply_grasp_height_extra(s, g, m_pose, None)
    assert g.extras["grasp_height_extra_effective_m"] == 0.0 and "library height kept (no floor asked for it)" in g.extras["grasp_height_extra_reason"]
    assert "lift_unreachable" not in s.counters


def test_partner_top_is_taken_under_the_fingertips():
    """Item 4: the cap's fingertips (r ~ 21 mm) hang over the bottle's shoulder, not over the 13 mm neck rim: no partner floor;
    the cork's tips (r ~ 9 mm) stand over the rim and keep it."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import attachment as A

    def stub(male_hull, partner_hull):
        s = SimpleNamespace()
        s.template_facts = {}
        s.hull = {"male": th.tensor(male_hull, dtype=th.float32), "partner": th.tensor(partner_hull, dtype=th.float32)}
        s.t_role, s.p_role = "male", "partner"
        s.parent_link_in_female = (th.tensor([0.0, 0.0, 0.20]), C.IDENTITY_QUAT.clone())  # the female link at the neck top, 20 cm up
        s.root_in_child_link = (th.zeros(3), C.IDENTITY_QUAT.clone())
        s.insertion_axis_local = th.tensor([0.0, 0.0, 1.0])
        s.rest = {"male": C.IDENTITY_QUAT.clone()}
        s._pinch_width_at = lambda eef_in_object: A.Attachment._pinch_width_at(s, eef_in_object)
        return s

    # a bottle: body r = 30 mm to z = 0.15, a shoulder ring at r = 22 mm z = 0.17, the neck r = 13 mm from z = 0.17 to 0.20 (rim at 0.20)
    bottle = _ring(0.030, 0.0) + _ring(0.030, 0.15) + _ring(0.022, 0.17) + _ring(0.013, 0.17) + _ring(0.013, 0.20)
    # the cap: r = 25.5 mm, 11 mm tall, pinched across its rim from above: eef 1.2 cm above the cap's top at its centre
    cap = _ring(0.0255, 0.0) + _ring(0.0255, 0.011)
    s = stub(cap, bottle)
    pos, quat = _eef([0.0, 0.0, 0.011 + 0.012], a=[0, 0, -1], c=[1, 0, 0])
    g = SimpleNamespace(eef_in_object=(pos, quat))
    clr = A.Attachment._attach_clearance(s, g)
    assert clr is not None and clr["pinch_width_m"] is not None and abs(clr["pinch_width_m"] - 0.051) < 0.002
    assert all(r > 0.02 for r in clr["r_tips_m"]), clr
    # under the cap's tips (r ~25.5 mm) the band meets the shoulder ring (r 22 mm, z 0.17 = 3 cm below the link) - a surface INSIDE
    # the tips' radius, not a top the finger (whose body extends outward from the pad face) stands on: no partner floor; the old
    # rule took the rim (z 0.20) within the cap's radius
    assert clr["pair_class"] == "over_partner" and clr["depth_m"] is None and "edge / inside" in clr["floor_rule"], clr
    assert abs(clr["depth_within_male_radius_m"] - 0.0) < 1e-3  # the rim at the link height (clamped legacy depth)
    # a bottle without a shoulder under the tips: no partner floor at all
    s = stub(cap, _ring(0.013, 0.17) + _ring(0.013, 0.20) + _ring(0.030, 0.0))
    clr = A.Attachment._attach_clearance(s, SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["depth_m"] is None and clr["h_tip_m"] is None
    assert s.template_facts["attach_pair_class"]["pair_class"] == "over_partner"
    # the real cap / lemonade bottle (0925-172312): the collar reaches r 27 mm at the mouth (4.6 mm above the link), the cap's tips
    # at r 25.5 mm stand over 1.5 mm of collar edge - no partner floor (the 09-21 success's grasp), class over_partner
    collar = _ring(0.030, 0.0) + _ring(0.030, 0.10) + _ring(0.0197, 0.20) + _ring(0.027, 0.20) + _ring(0.027, 0.2046) + _ring(0.0197, 0.2046)
    s = stub(cap, collar)
    clr = A.Attachment._attach_clearance(s, SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["pair_class"] == "over_partner" and "edge / inside" in clr["floor_rule"] and clr["depth_m"] is None and clr["h_tip_m"] is None, clr
    # a collar wider than the cap (r 32 mm): the cap sits ON that rim (on_rim) and the rim floor applies at the link height
    wide = _ring(0.030, 0.0) + _ring(0.030, 0.10) + _ring(0.0197, 0.20) + _ring(0.032, 0.20) + _ring(0.032, 0.2046)
    clr = A.Attachment._attach_clearance(stub(cap, wide), SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["pair_class"] == "on_rim" and "rim floor" in clr["floor_rule"] and clr["depth_m"] is not None and abs(clr["depth_m"]) < 1e-3, clr
    # the cork: r = 9 mm, 40 mm tall, pinched across from above 1 cm above its top: the tips stand over the rim
    cork = _ring(0.009, 0.0) + _ring(0.009, 0.04)
    s = stub(cork, bottle)
    pos, quat = _eef([0.0, 0.0, 0.04 + 0.01], a=[0, 0, -1], c=[1, 0, 0])
    clr = A.Attachment._attach_clearance(s, SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["pair_class"] == "into_opening" and "rim floor" in clr["floor_rule"], clr
    assert clr["depth_m"] is not None and abs(clr["depth_m"]) < 1e-3 and all(abs(r - 0.009) < 0.002 for r in clr["r_tips_m"]), clr  # the rim at the link height
    # h_tip = the fingertip axis point above the rim: the eef 5 cm up the cork (its bottom at the link), tips 18.8 mm below the eef
    assert clr["h_tip_m"] is not None and abs(clr["h_tip_m"] - ((0.05 - C.FINGERTIP_LENGTH) - 0.0)) < 1e-3, clr
    # owner 2026-09-25 (cork 0925-145307): a TILTED pad pinch whose tips stand 17 mm off the axis - the 4.25 mm band meets no hull
    # vertex between the rim and the shoulder / body; the pair is into_opening (9 mm cork, 13 mm rim), so the rim floor holds and
    # the fingertips must clear the rim by 5 mm whatever lies under them
    pos, quat = _eef([0.015, 0.0, 0.04 + 0.01], a=[0, 0, -1], c=[0, 1, 0])
    clr = A.Attachment._attach_clearance(s, SimpleNamespace(eef_in_object=(pos, quat)))
    assert all(r > 0.014 for r in clr["r_tips_m"]) and (clr["depth_under_tips_m"] is None or clr["depth_under_tips_m"] < -0.02), clr
    assert clr["pair_class"] == "into_opening" and abs(clr["depth_m"] - 0.0) < 1e-3 and abs(clr["h_tip_m"] - (0.05 - C.FINGERTIP_LENGTH)) < 1e-3, clr
    assert s.template_facts["attach_pair_class"]["depth_rim_m"] == 0.0 and s.template_facts["attach_pair_class"]["r_rim_m"] is not None
    # over_partner whose under-tips point lies deeper than the male is long (a lid whose knob pinch hangs over the pot's interior):
    # the rim rule again
    lid = _ring(0.03, 0.0) + _ring(0.03, 0.01) + _ring(0.006, 0.03)
    pot = _ring(0.022, 0.0) + _ring(0.022, 0.20) + _ring(0.020, 0.20) + _ring(0.020, 0.0) + _ring(0.006, 0.0)  # a thin-walled 20 cm pot, its interior bottom under the knob
    s = stub(lid, pot)
    pos, quat = _eef([0.0, 0.0, 0.03 + 0.01], a=[0, 0, -1], c=[1, 0, 0])
    clr = A.Attachment._attach_clearance(s, SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["pair_class"] == "over_partner" and "deeper than the male" in clr["floor_rule"] and abs(clr["depth_m"] - 0.0) < 1e-3, clr
    # a lid between the opening and the outer radius (it sits on a thick rim): the rim rule
    pot2 = _ring(0.05, 0.0) + _ring(0.05, 0.20) + _ring(0.02, 0.20) + _ring(0.02, 0.0)
    clr = A.Attachment._attach_clearance(stub(lid, pot2), SimpleNamespace(eef_in_object=(pos, quat)))
    assert clr["pair_class"] == "on_rim" and abs(clr["depth_m"] - 0.0) < 1e-3, clr


def test_male_rest_orientation_keeps_the_library_rest_unless_physics_toppled_it():
    """Owner 2026-09-25 (cap 0925-144739: flipped on geometry alone, open side up, no grasp reachable): the library rest is kept by
    default whatever the tipping margins; only a previous draw's toppling (``force``) may turn the male, and only about the
    insertion direction (never for a vertical insertion: the program does no in-hand 180 deg turn) with a grasp that still
    approaches from above."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import attachment as A

    cork = _ring(0.006, 0.0, 24) + _ring(0.010, 0.04, 24)  # a frustum on its narrow end: the flip would be the more stable end
    q, f = A.stable_rest_quat(th.tensor(cork), C.IDENTITY_QUAT)
    assert not f["flipped"] and th.allclose(q, C.IDENTITY_QUAT) and f["margin_flipped"] > 1.3 * f["margin_canonical"]
    # physics toppled it, but the insertion axis is vertical: a horizontal turn would reverse it -> kept
    q, f = A.stable_rest_quat(th.tensor(cork), C.IDENTITY_QUAT, force=True, insertion_axis_local=[0.0, 0.0, 1.0])
    assert not f["flipped"] and not f["flip_keeps_insertion_direction"] and "reverse the insertion" in f["reason"]
    # a male inserted sideways (insertion axis along x): the turn about x keeps the direction; a top-down grasp survives the turn
    # only if some library grasp still points down afterwards
    down_grasp = SimpleNamespace(pose=(th.zeros(3), C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), math.pi)))  # eef +z down
    q, f = A.stable_rest_quat(th.tensor(cork), C.IDENTITY_QUAT, force=True, insertion_axis_local=[1.0, 0.0, 0.0], grasps=[down_grasp])
    assert f["flip_keeps_insertion_direction"] and not f["flipped"] and f["grasps_from_above_after_flip"] == 0
    up_grasp = SimpleNamespace(pose=(th.zeros(3), C.IDENTITY_QUAT))  # eef +z up at the library rest: points down once turned
    q, f = A.stable_rest_quat(th.tensor(cork), C.IDENTITY_QUAT, force=True, insertion_axis_local=[1.0, 0.0, 0.0], grasps=[down_grasp, up_grasp])
    assert f["flipped"] and f["grasps_from_above_after_flip"] == 1
    assert float(C.quat_rotate(q, th.tensor([1.0, 0.0, 0.0]))[0]) > 0.99  # the insertion direction is kept
    assert float(C.quat_rotate(q, th.tensor([0.0, 0.0, 1.0]))[2]) < -0.99  # the wide end is the support now
    cap = _ring(0.0255, 0.0, 24) + _ring(0.0255, 0.011, 24)
    q, f = A.stable_rest_quat(th.tensor(cap), C.IDENTITY_QUAT)
    assert not f["flipped"] and th.allclose(q, C.IDENTITY_QUAT)


# ---------------------------------------------------------------------------------------------------------------- support surfaces


def _surface_sampler(surfaces, top_z, interior=None, top_raises=False):
    """A sampler for ``TT.support_surface``: the support top at ``top_z``, ``support_interior`` returning ``interior`` (calls in
    ``s.interior_calls``), ``bindings.support_surfaces`` = ``surfaces``."""
    import types

    from ispatialgym.sampling.sim_utils import SupportTop

    top = SupportTop("bookcase_x_0", th.arange(10) * 0.02, th.arange(10) * 0.02, th.ones(10, 10, dtype=th.bool), th.zeros(10, 10, dtype=th.bool), top_z, 0.0)
    s = types.SimpleNamespace(choice=types.SimpleNamespace(bindings={"support_surfaces": surfaces}), gp=types.SimpleNamespace(alias="a"), template_facts={},
                              interior_calls=[], gen=lambda k, d=None: d)

    def support_top(support, ignore=()):
        if top_raises:
            raise RuntimeError("no top cells")
        return top

    def support_interior(support, z, resolution=0.02, ignore=()):
        s.interior_calls.append(round(z, 4))
        return interior

    s.support_top = support_top
    s.world = types.SimpleNamespace(floor_z=0.05, spawned={}, support_interior=support_interior)
    return s, top


class _Interior:
    def __init__(self, z, headroom):
        self.name, self.z = "bookcase_x_0", z
        self.mask = th.ones(4, 4, dtype=th.bool)
        self.occupied = th.zeros(4, 4, dtype=th.bool)
        self.headroom = headroom
        self.open_edges = ["-v"]

    @property
    def free(self):
        return self.mask & ~self.occupied


def test_support_surface_keeps_the_top_without_a_sheet_surface_or_a_matching_one():
    import types

    sup = types.SimpleNamespace(name="bookcase_x_0")
    s, top = _surface_sampler([], 1.12)
    assert TT.support_surface(s, sup) is top and s.surface_z is None and TT.surface_band_kwargs(s) == {}
    s, top = _surface_sampler([{"name": "bookcase_x_0", "init": "ontop", "top_z": 1.09}], 1.12)  # 1.07 above the floor: within 3 cm
    assert TT.support_surface(s, sup) is top and s.surface_z is None and s.template_facts["support_surface"]["surface"] == "top"
    # owner 2026-10-11 06:27: a row marked surface: top stays on the top whatever its height
    s, top = _surface_sampler([{"name": "bookcase_x_0", "init": "ontop", "top_z": 0.84, "surface": "top", "why": "no surface"}], 1.12)
    assert TT.support_surface(s, sup) is top and s.interior_calls == [] and s.template_facts["support_surface"]["why"] == "no surface"


def test_support_surface_goes_inside_for_init_inside_or_a_height_mismatch():
    import types

    sup = types.SimpleNamespace(name="bookcase_x_0")
    hr = th.full((4, 4), 0.30)
    hr[0, :] = 0.10  # one row of cells under a low board
    s, _ = _surface_sampler([{"name": "other_0", "init": "ontop", "top_z": 0.5}, {"name": "bookcase_x_0", "init": "inside", "top_z": 0.73}], 1.12,
                            interior=_Interior(0.78, hr))
    it = TT.support_surface(s, sup, height=0.12)
    assert s.interior_calls == [0.78] and s.surface_z == 0.73 and TT.surface_band_kwargs(s) == {"surface_z": 0.73}
    assert int(it.free.sum()) == 12 and not bool(it.free[0].any())  # headroom 0.10 < 0.12 + 0.02
    f = s.template_facts["support_surface"]
    assert f["surface"] == "interior" and f["why"] == "init=inside" and f["n_low_headroom_cells"] == 4 and f["open_edges"] == ["-v"]
    s, _ = _surface_sampler([{"name": "bookcase_x_0", "init": "ontop", "top_z": 0.82}], 1.12, interior=_Interior(0.87, th.full((4, 4), 0.4)))
    TT.support_surface(s, sup)
    assert s.interior_calls == [0.87] and s.surface_z == 0.82 and "> 0.03 m" in s.template_facts["support_surface"]["why"]
    s, _ = _surface_sampler([{"name": "bookcase_x_0", "init": "ontop", "top_z": 0.56}], 0.0, interior=_Interior(0.61, th.full((4, 4), 0.4)), top_raises=True)
    TT.support_surface(s, sup)
    assert s.surface_z == 0.56 and s.template_facts["support_surface"]["why"] == "no support top raster"


def test_edge_frames_of_an_interior_surface_are_its_open_sides():
    import types

    from ispatialgym.sampling.sim_utils import SupportTop

    top = SupportTop("bookcase_x_0", th.arange(10) * 0.02, th.arange(10) * 0.02, th.ones(10, 10, dtype=th.bool), th.zeros(10, 10, dtype=th.bool), 0.8, 0.0)
    s = types.SimpleNamespace(world=types.SimpleNamespace(obj=lambda name: types.SimpleNamespace(aabb=(th.tensor([-0.02, -0.02, 0.0]), th.tensor([0.2, 0.2, 1.8])))))
    assert sorted(e.name for e, _ in TT.edge_frames(s, top)) == ["+u", "+v", "-u", "-v"]
    top.open_edges = ["-v"]
    assert [e.name for e, _ in TT.edge_frames(s, top)] == ["-v"]


BU_REST_XYZW = [0.45451947767204387, 0.4545194776720438, -0.5416752204197018, 0.5416752204197021]  # hardback__acbrnv__bu (wxyz in the library)


def test_upright_shelf_key_leans_toward_the_robots_left(monkeypatch):
    from ispatialgym.sampling import grasp_library as GL

    monkeypatch.setattr(GL, "_candidates", lambda key: {"ispatialgym": {"rest_pose": "upright_shelf", "context": "bookcase_floor_wall"}} if key.endswith("__bu") else {"ispatialgym": {"rest_pose": "canonical"}})
    f = TT.upright_shelf_facts("hardback__acbrnv__bu", th.tensor(BU_REST_XYZW))
    assert f["lean_side"] == 1.0 and abs(f["lean_deg"] - 10.0) < 0.1
    assert TT.upright_shelf_facts("hardback__acbrnv__bo", C.IDENTITY_QUAT.clone()) is None


def test_shelf_lean_pose_puts_the_spine_behind_the_edge_and_the_top_edge_at_the_side_board():
    import types

    from ispatialgym.sampling.sim_utils import SupportTop
    from ispatialgym.sampling.v2.stances import EdgeFrame

    h = th.tensor([0.22274, 0.17153, 0.03120]) / 2
    hull = th.tensor([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)
    rest = th.tensor(BU_REST_XYZW)
    xs, ys = th.arange(-0.39, 0.40, 0.02), th.arange(0.01, 0.30, 0.02)  # board cells: x in [-0.4, 0.4], y in [0, 0.3] (front edge y = 0)
    top = SupportTop("bookcase_x_0", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), 0.70, 0.0)
    edge = EdgeFrame("-v", th.tensor([0.0, 0.01]), th.tensor([0.0, 1.0]), 0.78)  # raster edge line at the first cell centres (y = 0.01)
    link = types.SimpleNamespace(prim_path="/World/bookcase_x_0/base_link")
    support = types.SimpleNamespace(name="bookcase_x_0", links={"base_link": link})
    rays = []

    def raycast(starts, ends):
        rays.append((starts.clone(), ends.clone()))  # the left board's face at x = -0.30 (the tangent of inward +y is -x)
        return [{"hit": True, "rigidBody": "/World/bookcase_x_0/base_link/collisions", "distance": 0.30, "position": [-0.30, float(starts[0, 1]), float(starts[0, 2])]}]

    def bottom_offset(hl, quat):
        return -float(C.transform_points((th.zeros(3), quat), hl)[:, 2].min())

    s = types.SimpleNamespace(world=types.SimpleNamespace(raycast=raycast, bottom_offset=bottom_offset))
    pose, facts = TT.shelf_lean_pose(s, hull, rest, top, edge, support, 1.0)
    pts = C.transform_points(pose, hull)
    # spine 2 cm behind the board's front edge (the raster edge line - half a cell = y 0.0)
    assert abs(float(pts[:, 1].min()) - 0.02) < 1e-4
    # the leaning top edge 2 mm short of the left board (x = -0.30), the bottom edge farther from it
    assert abs(float(pts[:, 0].min()) - (-0.30 + TT.SHELF_WALL_GAP_M)) < 1e-4
    top_pts, bot_pts = pts[pts[:, 2] > float(pts[:, 2].max()) - 0.01], pts[pts[:, 2] < float(pts[:, 2].min()) + 0.01]
    assert float(top_pts[:, 0].min()) < float(bot_pts[:, 0].min()) - 0.03
    # standing on its bottom edge on the board (3 mm drop), the ray at 0.8 of the book's height
    assert abs(float(pts[:, 2].min()) - (0.70 + TT.Z_OFFSET)) < 1e-4 and abs(facts["ray_z"] - (0.70 + 0.8 * 0.2247)) < 2e-3
    assert facts["lean_side"] == "left" and facts["n_bottom_off_board"] == 0
    # no side board on that side -> Reject
    s.world.raycast = lambda a, b: [{"hit": True, "rigidBody": "/World/other_book/base_link", "distance": 0.1}]
    with pytest.raises(Exception, match="shelf_wall|side board"):
        TT.shelf_lean_pose(s, hull, rest, top, edge, support, 1.0)


def test_multi_distance_band_is_read_at_the_stances_stand_off():
    """owner 2026-10-11 item 1: a band whose meta lists ``stand_offs`` is read at the stance's nominal stand-off (the nearest stored
    one) by every stance-based lookup; a band without them reads its reference maps as before."""
    k = (0.5, 0.0)
    ref = {k: [[0.60, 0.00]]}  # the plain maps hold the reference (0.45) cells
    rb = _synthetic_band(ref)
    rb.meta["stand_offs"], rb.meta["stand_off"] = [0.35, 0.45, 0.55], 0.45
    rb.cells_so = {0.35: {k: np.array([[0.50, 0.10]], dtype=np.float32)}, 0.45: {k: np.array([[0.60, 0.00]], dtype=np.float32)},
                   0.55: {k: np.array([[0.70, -0.10]], dtype=np.float32)}}
    rb.pairs_so = {s: {k: np.zeros((0, 3), dtype=np.int32)} for s in rb.cells_so}
    near, far = _stance(*k), _stance(*k)
    near.stand_off, far.stand_off = 0.36, 0.60
    assert TT.band_stand_off(rb, near) == 0.36 and TT.band_kw(rb, far) == {"stand_off": 0.60}
    assert th.allclose(TT.band_cells_world(rb, near), th.tensor([[0.50, 0.10]]))
    assert th.allclose(TT.band_cells_world(rb, far), th.tensor([[0.70, -0.10]]))
    assert TT.in_band(rb, far, th.tensor([0.70, -0.10]))[0] and not TT.in_band(rb, far, th.tensor([0.60, 0.00]))[0]
    assert th.allclose(TT.band_union_cells(rb, near), th.tensor([[0.50, 0.10]]))
    plain = _synthetic_band(ref)
    assert TT.band_stand_off(plain, near) is None and TT.band_kw(plain, near) == {}
    assert th.allclose(TT.band_cells_world(plain, far), th.tensor([[0.60, 0.00]]))


def test_upright_lean_and_mirrored_lean_orientations():
    """touching#03 (owner 2026-10-11): the partner stands on its long side leaning toward the requested side; the target's goal
    is its start lean mirrored about the shelf's depth axis (spine kept toward the robot)."""
    from ispatialgym.sampling.v2.stances import EdgeFrame

    h = th.tensor([0.22274, 0.17153, 0.03120]) / 2
    hull = th.tensor([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)
    for side in (1.0, -1.0):
        q = TT.upright_lean_quat(hull, side, 10.0)
        pts = C.transform_points((th.zeros(3), q), hull)
        ext = pts.max(dim=0).values - pts.min(dim=0).values
        assert abs(float(ext[2]) - (0.22274 * math.cos(math.radians(10)) + 0.0312 * math.sin(math.radians(10)))) < 1e-3  # long side up
        assert abs(float(ext[0]) - 0.17153) < 1e-3  # the width along the depth (rest +x)
        top, bot = pts[pts[:, 2] > 0.1], pts[pts[:, 2] < -0.1]
        assert side * float(top[:, 1].mean() - bot[:, 1].mean()) > 0.03  # the top leans toward the requested side
    # mirrored: the bu rest (leaning left, spine toward -x) turned to lean right about the inward axis of an edge facing +x
    edge = EdgeFrame("-v", th.tensor([0.0, 0.0]), th.tensor([1.0, 0.0]), 0.8)
    q0 = th.tensor(BU_REST_XYZW)
    qm = TT.mirrored_lean_quat(q0, edge, 1.0, 10.0)
    p0, pm = C.transform_points((th.zeros(3), q0), hull), C.transform_points((th.zeros(3), qm), hull)
    lean = lambda p: float(p[p[:, 2] > 0.1][:, 1].mean() - p[p[:, 2] < -0.1][:, 1].mean())
    assert lean(p0) > 0.03 and lean(pm) < -0.03 and abs(lean(p0) + lean(pm)) < 1e-3
    assert abs(float(pm[:, 0].min()) - float(p0[:, 0].min())) < 1e-4  # the spine stays at the same depth


def test_phase2_angle_classes_customs_then_subtask_default_then_global(monkeypatch):
    from types import SimpleNamespace as NS

    from ispatialgym.sampling import generation_defaults as GD

    gen = {}
    s = NS(subtask="alignment-world_vertical", gen=lambda k, d=None: gen.get(k, d))
    assert TT.phase2_angle_classes(s) == tuple(TT.ANGLE_CLASSES_DEG)
    real = GD.defaults_for
    monkeypatch.setattr(GD, "defaults_for", lambda st: {**real(st), "phase2_tight_class_deg": 8.0, "phase2_loose_class_deg": 20.0} if st == s.subtask else real(st))
    assert TT.phase2_angle_classes(s) == (8.0, 20.0)
    gen.update(phase2_tight_class_deg=25, phase2_loose_class_deg=60)
    assert TT.phase2_angle_classes(s) == (25.0, 60.0)
    assert TT.phase2_angle_classes(NS(subtask="no-such-subtask", gen=lambda k, d=None: d)) == tuple(TT.ANGLE_CLASSES_DEG)
