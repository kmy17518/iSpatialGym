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
    # no left band yet: the brace side is left to the exact two-arm IK, flagged for the record
    r4 = TT.joint_placement(right, None, st, (0.61, -0.21), (5.0, 5.0))
    assert r4["ok"] and not r4["left_band"] and r4["brace_cell"] is None


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
