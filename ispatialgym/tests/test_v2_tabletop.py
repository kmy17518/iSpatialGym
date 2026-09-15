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
    assert "omnigibson" not in sys.modules or True  # the import above must not have pulled the simulator in
    assert not any(m.startswith("omnigibson") for m in sys.modules if m == "omnigibson"), "tabletop.py must not import omnigibson at module level"


# ----------------------------------------------------------------------------------------------------------------
# §4.4 margins table
# ----------------------------------------------------------------------------------------------------------------


def test_margins_iors_row_uses_dp_min_of_partner_width_and_r_img():
    row = {"w_t": 0.05, "w_p": 0.10, "W_p": 0.20, "D_p_max": 0.23, "safety": 0.02, "relation": "ontop", "g_NT": None}
    m_R, m_L, info = TT.lateral_margins(row, lam_min=-0.30, lam_max=0.40, b=0.9, R_img=0.6)
    assert info["D_p"] == pytest.approx(0.23)  # min(W_p + 0.03, R_img) = W_p + 0.03 here
    assert m_R == pytest.approx(0.30 + 0.05 + 0.23 + 0.20 + 0.02)
    assert m_L == pytest.approx(0.40 + 0.05 + 0.23 + 0.20 + 0.02)
    # R_img smaller than W_p + 0.03 caps D_p
    m_R2, _, info2 = TT.lateral_margins(row, -0.30, 0.40, b=0.9, R_img=0.10)
    assert info2["D_p"] == pytest.approx(0.10) and m_R2 == pytest.approx(0.30 + 0.05 + 0.10 + 0.20 + 0.02)
    # λ_max < 0 contributes nothing to the left margin (max(0, λ_max)); λ_min > 0 nothing to the right
    _, m_L3, _ = TT.lateral_margins(row, -0.30, -0.10, 0.9, 0.6)
    assert m_L3 == pytest.approx(0.0 + 0.05 + 0.23 + 0.20 + 0.02)
    m_R4, _, _ = TT.lateral_margins(row, 0.05, 0.40, 0.9, 0.6)
    assert m_R4 == pytest.approx(0.0 + 0.05 + 0.23 + 0.20 + 0.02)


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
