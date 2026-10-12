"""Simulator-free tests of the v2 large-scale sampler (spec §5.2 / §5.3 / §5.5 / §5.6): the face-frame math, the
Prompt-0 approach ray and scans, the reach-volume lookup frame, the handle trajectory h(q) against the conventions,
the base-follow composition, the arm / follow-mode rules, the brace-kind ordering and the init-surface classes
(``pytest ispatialgym/tests/test_v2_large_scale.py``).

The task modules import the sampler core (OmniGibson is importable without launching Isaac Sim); nothing here needs a
live world."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.eef_reach_volume import TILTS, TRANSLATES
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.stances import Stance

AL = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_large")
AW = pytest.importorskip("ispatialgym.sampling.v2.tasks.alignment_wall")
IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


# ----------------------------------------------------------------------------------------------------------------
# face frame (§3.3) and the Prompt-0 ray / scans (§5.2)
# ----------------------------------------------------------------------------------------------------------------


def test_face_frame_yaw_and_dl():
    face = P0.make_face("face", [1.0, 0.0, 0.3], [2.0, 1.0, 1.1])  # n is made horizontal and unit
    assert th.allclose(face.n, th.tensor([1.0, 0.0, 0.0]))
    assert abs(C.wrap_angle(P0.face_yaw(face) - math.pi)) < 1e-6  # the robot faces along -n
    st = P0.square_stance(face, d=0.6, lam=0.2, translate=0.5, tilt=-0.3)
    # base = P + n d + tangent λ with tangent = robot's left when facing the face = (n_y, -n_x) = (0, -1)
    assert abs(st.base_pose2d[0] - 2.6) < 1e-6 and abs(st.base_pose2d[1] - 0.8) < 1e-6
    assert abs(C.wrap_angle(st.yaw - math.pi)) < 1e-6 and st.stand_off == 0.6 and st.lateral == 0.2
    assert st.face.d == 0.6 and st.face.lam == 0.2
    assert st.to_sidecar()["face"] == "face" and st.to_sidecar()["lambda"] == 0.2
    # the robot's left in the world at that yaw is the face tangent
    assert th.allclose(st.left, face.tangent, atol=1e-6)


def test_approach_ray_faces_P():
    face = P0.make_face("f", [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
    for beta in (0.0, math.radians(20), math.radians(-35)):
        x, y, yaw = P0.approach_ray_base(face, beta, 1.2)
        assert abs(math.hypot(x, y) - 1.2) < 1e-6
        # bearing β about n through P
        assert abs(math.atan2(y, x) - (math.atan2(-1.0, 0.0) + beta)) < 1e-6
        # the base faces P
        assert abs(C.wrap_angle(yaw - math.atan2(-y, -x))) < 1e-6
    st = P0.ray_stance(face, math.radians(20), 0.9, 0.25)
    assert st.extras["bearing_deg"] == pytest.approx(20.0) and st.extras["distance"] == 0.9 and st.stand_off == 0.9


def test_scans_and_lattice():
    assert P0.d_scan(0.5, 2.5) == pytest.approx([0.5 + 0.05 * k for k in range(41)])
    assert P0.d_scan(0.5, 0.52) == [0.5]
    assert P0.lattice_values(0.35, 0.90, 0.05) == pytest.approx([0.35 + 0.05 * k for k in range(12)])
    assert P0.lattice_values(-0.10, 0.60, 0.05) == pytest.approx([-0.10 + 0.05 * k for k in range(15)])
    assert P0.lattice_values(0.3, 0.2, 0.05) == []
    face = P0.make_face("f", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    lat = P0.phase2_lattice(face, (0.35, 0.90), (-0.10, 0.60))
    assert len(lat) == 12 * 15 * len(TRANSLATES) * len(TILTS)
    assert all(st.extras["band"][0] <= st.tilt <= st.extras["band"][1] for st in lat)


def test_tilt_band_and_angle_classes():
    lo, hi = P0.tilt_band(0.0, 0.5)
    assert hi == 0.0 and lo == pytest.approx(-0.15)
    lo, hi = P0.tilt_band(-0.9, 0.95)
    assert lo >= P0.tilt_min(0.95) - 1e-9 and hi == pytest.approx(-0.75)
    assert P0.angle_class(math.radians(5)) == 0 and P0.angle_class(math.radians(12)) == 1 and P0.angle_class(math.radians(25)) is None
    assert P0.best_class([("a", 1), ("b", 0), ("c", 0)]) == ["b", "c"]
    assert P0.best_class([("a", 1)]) == ["a"] and P0.best_class([]) == []
    assert P0.phase1_choice([(0.0, 0.3), (0.5, 0.1), (0.75, 0.1)]) == 0.5  # smallest angle, tie -> smaller t
    assert P0.phase1_choice([]) is None


def test_reach_volume_frame_and_classes():
    # a world point ahead of a base at yaw pi/2 is +x in the base frame; z relative to base_link
    p = P0.to_base_frame([0.0, 1.5, 1.2], (0.0, 0.5, math.pi / 2), base_z=0.0053)
    assert th.allclose(p, th.tensor([1.0, 0.0, 1.2 - 0.0053]), atol=1e-6)
    # a forward pinch with a vertical closing axis at that base -> fwd_pinch_v
    q = C.eef_quat([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    assert P0.reach_class(q, math.pi / 2) == "fwd_pinch_v"
    assert P0.reach_class(q, math.pi / 2, allowed=("fwd_pinch_h",)) is None
    qh = C.eef_quat([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])  # closing horizontal (along the base's -y)
    assert P0.reach_class(qh, math.pi / 2) == "fwd_pinch_h"
    assert P0.reach_class(qh, math.pi / 2, allowed=("fwd_pinch_v", "fwd_pinch_h")) == "fwd_pinch_h"  # a knob admits either
    # pushes: approach up with the pads along the base x -> push_up
    qu = C.eef_quat([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert P0.reach_class(qu, math.pi / 2) == "push_up"
    # facing away (the pinch approach points at the robot) -> no class
    q_back = C.eef_quat([0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
    assert P0.reach_class(q_back, math.pi / 2) is None


def test_contact_spec_key_and_poses():
    q = C.eef_quat([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    spec = P0.ContactSpec("handle", "right", (th.tensor([1.0, 0.0, 1.0]), q), classes=("fwd_pinch_v",))
    pre = spec.pre_pose()
    assert th.allclose(pre[0], th.tensor([0.9, 0.0, 1.0]), atol=1e-6)  # 0.10 m back along -a
    assert len(spec.screen_poses()) == 1
    key = P0.contact_set_key([spec])
    assert key[0][0] == "handle" and key[0][1] == "right" and len(key[0][3][0]) == 7


# ----------------------------------------------------------------------------------------------------------------
# articulation (§5.3): h(q), waypoints, arm rule, follow mode, base follow
# ----------------------------------------------------------------------------------------------------------------

_DOOR = {"contact": [-0.0759, 0.3842, -0.1247], "pinch_axis": [0.0, 1.0, 0.0], "approach_dir": [1.0, 0.0, 0.0], "kind": "bar", "width": 0.0202}
_DOOR_JOINT = {"joint": "j_leaf", "type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, 1.0], "hinge_point_obj": [-0.05511, -0.41211, 0.0], "lower": 0.0,
               "upper": 1.9084, "closed_q": 0.0, "link_bbox_min": [-0.0879, -0.4178, -1.1189], "link_bbox_max": [0.0578, 0.4183, 1.1181]}
_DRAWER = {"contact": [0.2831, -0.0151, -0.0383], "pinch_axis": [0.0, 0.0, -1.0], "approach_dir": [-1.0, 0.0, 0.0], "kind": "bar", "width": 0.0121}
_DRAWER_JOINT = {"joint": "j_link_1", "type": "PrismaticJoint", "axis_dir_obj": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 0.36, "closed_q": 0.0}


def test_waypoint_values_and_required():
    wps = AL.waypoint_values(0.0, 1.2)
    assert [k for k, _ in wps] == [0.25, 0.5, 0.75, 1.0] and [q for _, q in wps] == pytest.approx([0.3, 0.6, 0.9, 1.2])
    wps = AL.waypoint_values(1.2, 0.0)  # close: from the open angle to the lower limit
    assert [q for _, q in wps] == pytest.approx([0.9, 0.6, 0.3, 0.0])
    assert AL.required_fractions(None) == [0.25, 0.5, 0.75, 1.0]
    assert AL.required_fractions(0.5) == [0.25, 0.5]
    assert AL.required_fractions(0.5, mode=True) == [0.25, 0.5, 0.75, 1.0]
    assert AL.required_fractions(None, mode=False) == [0.25, 0.5, 0.75]
    assert AL.fraction_key(0.25) == "0.25" and AL.fraction_key(1.0) == "1.0" and AL.fraction_key(0.0) == "0.0" and AL.fraction_key(0.5) == "0.5"


def test_handle_trajectory_revolute_matches_conventions():
    obj_pose = (th.tensor([-6.91, 4.80, 1.16]), C.quat_from_yaw(math.pi - 0.016))
    qs = [0.0, 0.3, 0.6, 1.2]
    wps = C.handle_waypoints(_DOOR, _DOOR_JOINT, qs, obj_pose=obj_pose)
    assert len(wps) == 4
    hinge_w = C.transform_points(obj_pose, th.tensor(_DOOR_JOINT["hinge_point_obj"]))[0]
    r0 = float(th.linalg.norm((wps[0].contact - hinge_w)[:2]))
    for w in wps[1:]:
        # the contact orbits the vertical hinge at constant radius and height
        assert abs(float(th.linalg.norm((w.contact - hinge_w)[:2])) - r0) < 1e-4
        assert abs(float(w.contact[2] - wps[0].contact[2])) < 1e-5
        # the approach stays horizontal and perpendicular to the (vertical) pinch axis
        assert abs(float(w.approach[2])) < 1e-6 and abs(float(th.dot(w.approach, w.pinch))) < 1e-5
        # eef origin = contact - 0.012·a; pre-grasp 0.10 m further back
        assert th.allclose(w.eef[0], w.contact - 0.012 * w.approach, atol=1e-6)
        assert th.allclose(w.pre_grasp[0], w.eef[0] - 0.10 * w.approach, atol=1e-6)
        assert th.allclose(C.approach_axis(w.eef[1]), w.approach, atol=1e-5)
    # the swept angle between q = 0 and q = 1.2 is 1.2 rad about the hinge
    a0, a1 = (wps[0].contact - hinge_w)[:2], (wps[3].contact - hinge_w)[:2]
    ang = math.atan2(float(a1[1]), float(a1[0])) - math.atan2(float(a0[1]), float(a0[0]))
    assert abs(C.wrap_angle(ang)) == pytest.approx(1.2, abs=1e-5)
    # sidecar block of a waypoint
    d = wps[1].to_sidecar()
    assert set(d) == {"contact", "approach", "pinch"} and len(d["contact"]) == 3


def test_handle_trajectory_prismatic_matches_conventions():
    obj_pose = (th.tensor([1.0, 2.0, 0.4]), C.quat_from_yaw(0.5))
    wps = C.handle_waypoints(_DRAWER, _DRAWER_JOINT, [0.0, 0.09, 0.18, 0.36], obj_pose=obj_pose)
    axis_w = C.quat_rotate(obj_pose[1], th.tensor(_DRAWER_JOINT["axis_dir_obj"]))
    for w in wps[1:]:
        delta = w.contact - wps[0].contact
        assert abs(float(th.linalg.norm(delta)) - w.q) < 1e-6  # translated by q along the axis
        assert th.allclose(delta / w.q, axis_w, atol=1e-5)
        assert th.allclose(w.approach, wps[0].approach) and th.allclose(w.pinch, wps[0].pinch)  # directions unchanged


def test_arm_rule_and_follow_mode_and_classes():
    assert AL.arm_rule(0.25) == "left" and AL.arm_rule(-0.25) == "right" and AL.arm_rule(0.05) == "right" and AL.arm_rule(-0.05) == "right"
    assert AL.arm_rule(0.25, override="right") == "right" and AL.arm_rule(0.25, override="auto") == "left"
    assert AL.follow_mode("RevoluteJoint", [0, 0, 1]) == "base"  # vertical-axis door
    assert AL.follow_mode("RevoluteJoint", [0, 1, 0]) == "arm"  # horizontal-axis flap / oven door
    assert AL.follow_mode("PrismaticJoint", [1, 0, 0]) == "base"  # horizontal drawer
    assert AL.follow_mode("PrismaticJoint", [0, 0, 1]) == "arm"  # vertical slider
    assert AL.pinch_classes([0, 0, 1], "bar") == ("fwd_pinch_v",)
    assert AL.pinch_classes([0, 1, 0], "bar") == ("fwd_pinch_h",)
    assert set(AL.pinch_classes([0, 0, 1], "knob")) == {"fwd_pinch_v", "fwd_pinch_h"}
    assert AL.joint_tolerance("RevoluteJoint", 1.2, {"revolute": 0.05, "prismatic": 0.01, "joint_fraction": 0.10}) == pytest.approx(0.12)
    assert AL.joint_tolerance("RevoluteJoint", 0.2, {"revolute": 0.05, "prismatic": 0.01, "joint_fraction": 0.10}) == pytest.approx(0.05)
    assert AL.joint_tolerance("PrismaticJoint", 0.36, {"revolute": 0.05, "prismatic": 0.01, "joint_fraction": 0.10}) == pytest.approx(0.036)


def test_a4_door_class_follow_mode_and_hinge_side():
    # A4 (owner decision 2026-09-16): class by category / joint type, follow mode by class
    assert AL.door_class("bottom_cabinet", "RevoluteJoint") == "cabinet"
    assert AL.door_class("fridge", "RevoluteJoint") == "cabinet" and AL.door_class("dishwasher", "RevoluteJoint") == "cabinet"
    assert AL.door_class("door", "RevoluteJoint") == "room_door"
    assert AL.door_class("bottom_cabinet", "PrismaticJoint") == "drawer" and AL.door_class("door", "PrismaticJoint") == "drawer"  # sliding door = prismatic
    assert AL.follow_mode_by_class("cabinet", "RevoluteJoint", [0, 0, 1]) == "arm"
    assert AL.follow_mode_by_class("room_door", "RevoluteJoint", [0, 0, 1]) == "hybrid"
    assert AL.follow_mode_by_class("drawer", "PrismaticJoint", [1, 0, 0]) == "base"
    assert AL.follow_mode_by_class("drawer", "PrismaticJoint", [0, 0, 1]) == "arm"
    assert AL.axis_is_vertical([0, 0, 1]) and AL.axis_is_vertical([0.1, 0, 0.99]) and not AL.axis_is_vertical([0, 1, 0])
    # the hinge side is read along the face frame's left tangent: hinge on the robot's left -> base to the left (Λ ⊂ [0, 0.6]) -> right arm
    tangent_left = th.tensor([0.0, 1.0])  # the robot faces +x, its left is +y
    assert AL.hinge_side([1.0, 0.5], [1.0, 0.0], tangent_left) == "left"
    assert AL.hinge_side([1.0, -0.5], [1.0, 0.0], tangent_left) == "right"
    assert AL.hinge_side_lateral("left") == (0.0, 0.60) and AL.hinge_side_lateral("right", 0.5) == (-0.5, 0.0)
    assert AL.arm_for_hinge_side("left") == "right" and AL.arm_for_hinge_side("right") == "left"


def test_a_follow_that_needs_the_base_is_sequential_where_the_hand_drives_it():
    # owner 2026-09-27 (dishwasher close 0926-065206): no base motion while the hand pulls / pushes the door
    assert AL.base_follow_policy(None, "close", "cabinet") == "sequential"
    assert AL.base_follow_policy(None, "open", "room_door") == "sequential"
    assert AL.base_follow_policy("sequential", "open", "cabinet") == "sequential"
    assert AL.base_follow_policy("hybrid", "close", "cabinet") == "hybrid"  # an explicit customs policy wins
    # a room door's close is pushed shut (push_close): its follow is a sampling gate, the knob's default stays
    assert AL.base_follow_policy(None, "close", "room_door") == AL.AH.DEFAULTS["policy"]


def test_edge_pinch_leaf_plane_approach():
    # office cabinet rvunhj_0 / j_link_2 (B4 edge_door fallback): approach parallel to the pinch axis -> repaired along the leaf toward the hinge
    handle = {"contact": [0.262, -0.0352, -0.1119], "pinch_axis": [1.0, 0.0, 0.0], "approach_dir": [-0.9999, -0.0122, 0.0]}
    jinfo = {"type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, -1.0], "hinge_point_obj": [0.27, -0.399, -0.3]}
    a = AL.leaf_plane_approach(handle, jinfo)
    assert a is not None and a[1] == pytest.approx(-1.0) and abs(a[0]) < 1e-6 and abs(a[2]) < 1e-6
    # the repaired frame is valid: at 90 deg open the approach points into the face (-x) and the pinch is horizontal (-y)
    handle2 = dict(handle, approach_dir=a)
    wps = C.handle_waypoints(handle2, dict(jinfo, lower=0.0, closed_q=0.0), [0.0, math.pi / 2])
    assert th.allclose(wps[1].approach, th.tensor([-1.0, 0.0, 0.0]), atol=1e-4) and th.allclose(wps[1].pinch.abs(), th.tensor([0.0, 1.0, 0.0]), atol=1e-4)
    # a proper bar handle (approach perpendicular to the pinch) is left alone
    assert AL.leaf_plane_approach({"contact": [0, 0, 0], "pinch_axis": [0, 0, 1], "approach_dir": [-1, 0, 0]}, jinfo) is None
    # a drawer front's edge: along the motion axis
    d = AL.leaf_plane_approach({"contact": [0.3, 0, 0], "pinch_axis": [1, 0, 0], "approach_dir": [-1, 0, 0]}, {"type": "PrismaticJoint", "axis_dir_obj": [1, 0, 0]})
    assert d is not None and abs(d[0]) == pytest.approx(1.0)


def test_a4_reach_grid_and_goal_set_restriction():
    step = math.radians(10.0)
    grid = AL.reach_grid(0.0, math.radians(95.0), step)
    assert grid[0] == pytest.approx(step) and grid[-1] == pytest.approx(math.radians(95.0)) and len(grid) == 10
    assert all(b > a for a, b in zip(grid, grid[1:]))
    assert AL.reach_grid(0.5, 0.5, step) == [] and AL.reach_grid(0.0, 0.05, 0.05) == [pytest.approx(0.05)]
    stable = [0.96, 1.20, 1.44, 1.68, 1.92]
    min_open = math.radians(50.0)
    # the arm reaches up to 1.3 rad: the stable angles below it form the goal set
    assert AL.restrict_goal_set(stable, 1.30, 0.0, min_open) == [0.96, 1.20]
    # nothing stable within reach but a 52 deg opening (below the first stable angle 0.96 rad = 55 deg): the reach limit itself
    assert AL.restrict_goal_set(stable, math.radians(52.0), 0.0, min_open) == [pytest.approx(math.radians(52.0))]
    # below the minimum opening: no goal
    assert AL.restrict_goal_set(stable, math.radians(40.0), 0.0, min_open) == []
    # a reach limit equal to a stable angle keeps it (tolerance)
    assert AL.restrict_goal_set(stable, 0.96 - 1e-9, 0.0, min_open) == [0.96]


def test_a4_reach_margin_inside_a_true_boundary_only():
    """The goal is drawn ``reach_margin`` inside the arm's reach boundary (the dishwasher goal AT the boundary left the
    drives saturated 7 mm short); a sweep that reaches its end (the joint limit) keeps the boundary itself."""
    m = math.radians(3.0)
    assert AL.reach_with_margin(math.radians(24.0), 0.0, m, ended_at_miss=True) == pytest.approx(math.radians(21.0))
    assert AL.reach_with_margin(math.radians(90.0), 0.0, m, ended_at_miss=False) == pytest.approx(math.radians(90.0))
    # never below the sweep's start
    assert AL.reach_with_margin(math.radians(2.0), 0.0, m, ended_at_miss=True) == 0.0
    assert AL.REACH_MARGIN["revolute_deg"] > 0 and AL.REACH_MARGIN["prismatic_m"] > 0


def test_passive_probe_needs_a_real_swing_on():
    """§5.3 passive continuation: released at q_k the link must END within the band of q_goal AND cover at least half the
    remaining gap on its own - the 0917 dishwasher record's ``passive_from`` 0.75 passed a 0.10 rad band wider than the
    0.087 rad waypoint spacing without the door moving (the friction then held it 0.098 rad short)."""
    q_goal, tol = math.radians(20.0), 0.5 * 0.05
    q75 = 0.75 * q_goal
    # the door does not move: not a pass, whatever the band says
    assert not AL.passive_reaches(q75, q75, q_goal, tol)
    assert not AL.passive_reaches(q75, q75, q_goal, tol=0.10)
    # it swings on to the goal: a pass
    assert AL.passive_reaches(q75, q_goal - 0.01, q_goal, tol)
    # ends in the band but covered less than half the gap (a nudge, not a swing): not a pass
    q_k = q_goal - 0.04
    assert not AL.passive_reaches(q_k, q_goal - 0.024, q_goal, tol)
    assert AL.passive_reaches(q_k, q_goal - 0.015, q_goal, tol)
    # overshooting past the goal within the band is fine; beyond the band is not
    assert AL.passive_reaches(q75, q_goal + 0.02, q_goal, tol)
    assert not AL.passive_reaches(q75, q_goal + 0.03, q_goal, tol)
    # a waypoint already at the goal passes trivially (nothing to cover)
    assert AL.passive_reaches(q_goal, q_goal, q_goal, tol)


def test_base_follow_composition():
    """b_k = h(q_k) ∘ T_eef→base(q*): with the arm locked, the base follows the handle exactly for a vertical door."""
    obj_pose = (th.tensor([0.0, 0.0, 1.0]), C.IDENTITY_QUAT.clone())
    wps = C.handle_waypoints(_DOOR, _DOOR_JOINT, [0.0, 0.5, 1.0], obj_pose=obj_pose)
    base0 = (0.9, 0.3, math.pi)  # the manipulation stance faces -x (the door's outward normal is +x in this frame)
    bz = 0.0053
    targets = AL.base_follow_targets([w.eef for w in wps], wps[0].eef, base0, bz)
    assert len(targets) == 3
    # at q_init the composition returns the manipulation base pose itself
    assert targets[0][0] == pytest.approx(base0[0], abs=1e-5) and targets[0][1] == pytest.approx(base0[1], abs=1e-5)
    assert abs(C.wrap_angle(targets[0][2] - base0[2])) < 1e-5
    # the eef-to-base offset is rigid: the base moves with the handle by the same swept angle about the hinge
    hinge_w = th.tensor(_DOOR_JOINT["hinge_point_obj"]) + obj_pose[0]
    for w, b in zip(wps, targets):
        eef_pose = w.eef
        base_pose = C.pose2d_to_pose(b[0], b[1], b[2], z=bz)
        rel = C.relative(base_pose, eef_pose)
        rel0 = C.relative(C.pose2d_to_pose(*base0, z=bz), wps[0].eef)
        assert th.allclose(rel[0], rel0[0], atol=1e-4) and C.quat_angle(rel[1], rel0[1]) < 1e-4
        r_b = math.hypot(b[0] - float(hinge_w[0]), b[1] - float(hinge_w[1]))
        r_b0 = math.hypot(base0[0] - float(hinge_w[0]), base0[1] - float(hinge_w[1]))
        assert r_b == pytest.approx(r_b0, abs=1e-4)
    assert abs(C.wrap_angle(targets[2][2] - base0[2])) == pytest.approx(1.0, abs=1e-4)  # yaw follows the joint by q_goal - q_init
    # the planar stand-in of "the grasped link ignored for the gripper only"
    assert AL.leaf_clear_of_base(th.tensor([[1.0, 1.0], [1.2, 1.0]]), (0.0, 0.0))
    assert not AL.leaf_clear_of_base(th.tensor([[0.2, 0.1]]), (0.0, 0.0))


def test_handle_rows_ranking_and_exclusion():
    rows = [AL.HandleRow("knob", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.02, "record:knob"),
            AL.HandleRow("bar", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.02, "record:bar"),
            AL.HandleRow("edge", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.05, "k:j#F", verdict="edge"),
            AL.HandleRow("edge", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.05, "k:j#F2", verdict="edge_blocked"),
            AL.HandleRow("manual", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.03, "k:j#M", verdict="manual")]
    ranked = AL.rank_handle_rows(rows)
    assert [r.kind for r in ranked] == ["bar", "knob", "manual", "edge"]  # edge_blocked dropped
    rows[0].chosen = True
    assert AL.rank_handle_rows(rows)[0].kind == "knob"  # the reviewer's choice first
    assert not AL.joint_excluded(rows)
    assert AL.joint_excluded([AL.HandleRow("bar", [0, 0, 0], [0, 1, 0], [1, 0, 0], 0.02, "x", verdict="ungraspable")])
    # the B4 door record: bar + knob + the blocked edge fallback
    rec = {"handles": [{"kind": "bar", "contact": [0, 0.38, -0.12], "pinch_axis": [0, 1, 0], "approach_dir": [1, 0, 0], "width": 0.0202},
                       {"kind": "knob", "contact": [0, 0.38, 0], "pinch_axis": [0, 1, 0], "approach_dir": [1, 0, 0], "width": 0.0202}],
           "fallback": {"kind": "edge_door", "contact": [0.05, 0.38, 0.17], "pinch_axis": [-0.1, -1, 0], "approach_dir": [-1, 0, 0], "width": 0.1457, "exposed": False}}
    rr = AL.record_handle_rows(rec)
    assert [r.kind for r in rr] == ["bar", "knob", "edge"] and rr[2].verdict == "edge_blocked"
    assert [r.kind for r in AL.rank_handle_rows(rr)] == ["bar", "knob"]


def test_sane_hinge_point_projects_the_runaway():
    j = dict(_DOOR_JOINT, hinge_point_obj=[-0.05511, -0.41211, 128.6673])  # B4 run-away along the vertical axis
    h = AL.sane_hinge_point(j)
    assert h[0] == pytest.approx(-0.05511, abs=1e-5) and h[1] == pytest.approx(-0.41211, abs=1e-5) and abs(h[2] - (-1.1189 + 1.1181) / 2) < 1e-4


def test_icl_handle_rows_scene_instance(tmp_path):
    csv = tmp_path / "handles.csv"
    csv.write_text(
        "key,category,model,scene,instance,require,kind,name,position,axis_approach,axis_close,width_m,verdict,flags,note,path,source,joint,chosen\n"
        'door__x,door,x,,,handle,bar,j#A,"[0.0, 0.5, 0.0]","[1.0, 0.0, 0.0]","[0.0, 1.0, 0.0]",0.02,handle,,,p,s,j,1\n'
        'door__x,door,x,,,handle,edge_door,j#F,"[0.1, 0.5, 0.2]","[-0.1, -1.0, 0.0]","[-1.0, 0.0, 0.0]",0.14,edge_blocked,,,p,s,j,\n'
        'door__x,door,x,house,door_x_0,handle,bar,j#A,"[0.0, 0.6, 0.0]","[1.0, 0.0, 0.0]","[0.0, 1.0, 0.0]",0.021,handle,,,p,house,j,\n'
        'door__x,door,x,other,door_x_1,handle,bar,j#A,"[0.0, 0.7, 0.0]","[1.0, 0.0, 0.0]","[0.0, 1.0, 0.0]",0.021,handle,,,p,other,j,\n'
        'door__x,door,x,,,handle,bar,k#A,"[0.0, 0.5, 0.0]","[1.0, 0.0, 0.0]","[0.0, 1.0, 0.0]",0.02,handle,,,p,s,k,\n')
    rows = AL.icl_handle_rows("door__x", "j", "house", "door_x_0", path=str(csv), instance_scale=[1.0, 2.0, 1.0])
    kinds = sorted((r.kind, r.verdict, round(r.contact[1], 3)) for r in rows)
    # the canonical bar is scaled by the instance scale (y doubled), the in-place row is taken as is, the other scene's row is skipped
    assert ("bar", "handle", 1.0) in kinds and ("bar", "handle", 0.6) in kinds and ("edge", "edge_blocked", 1.0) in kinds
    assert all(r.contact[1] != 0.7 for r in rows)
    assert [r for r in rows if r.chosen and r.contact[1] == 1.0]
    assert [r.kind for r in AL.rank_handle_rows(rows)][0] == "bar"


def test_alias_handle_row_and_pinch_width_floor():
    """Owner 2026-10-12 02:20 (open / close_large#03, office cabinet door rvunhj_0 / j_link_2): ``generation.handle_row`` takes
    the named row before the ranking and ``generation.pinch_width_min_m`` lowers the 15 mm floor; with neither the 8 mm bar
    and the 7 mm knobs fall through to the 39 mm edge pinch, as before."""
    from ispatialgym.sampling.sim_utils import Reject

    a, p = [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    rec_bar = AL.HandleRow("bar", [0.2856, -0.0348, -0.0658], p, a, 0.008, "record:bar")
    rec_knob = AL.HandleRow("knob", [0.2826, -0.0355, 0.0535], p, a, 0.007, "record:knob")
    bar = AL.HandleRow("bar", [0.2856, -0.0348, -0.0658], p, a, 0.008, "bottom_cabinet__rvunhj:j_link_2#C")
    knob_b = AL.HandleRow("knob", [0.2826, -0.0355, 0.0535], p, a, 0.007, "bottom_cabinet__rvunhj:j_link_2#B")
    edge = AL.HandleRow("edge", [0.262, -0.0352, -0.1119], [1.0, 0.0, 0.0], [-0.9999, -0.0122, 0.0], 0.0386, "bottom_cabinet__rvunhj:j_link_2#F", verdict="edge")
    blocked = AL.HandleRow("edge", [0.26, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0386, "bottom_cabinet__rvunhj:j_link_2#F2", verdict="edge_blocked")
    rows = [rec_bar, rec_knob, bar, knob_b, edge, blocked]
    row, ranked = AL.pick_handle_row(rows)  # the defaults: 15-80 mm, the ranking
    assert row is edge and ranked == AL.rank_handle_rows(rows)
    row, _ = AL.pick_handle_row(rows, width_range=(0.006, 0.080))  # the floor alone: the ranking's first row (a bar) now fits
    assert row.kind == "bar"
    for name, want in (("j_link_2#C", bar), ("bottom_cabinet__rvunhj:j_link_2#C", bar), ("j_link_2#B", knob_b), ("record:bar", rec_bar), ("j_link_2#F", edge)):
        row, ranked = AL.pick_handle_row(rows, name, (0.006, 0.080))
        assert row is want and ranked[0] is want and sorted(map(id, ranked)) == sorted(map(id, AL.rank_handle_rows(rows)))
    with pytest.raises(Reject) as e:  # a named row below the floor is rejected, never swapped for the edge pinch
        AL.pick_handle_row(rows, "j_link_2#C", AL.PINCH_WIDTH_RANGE, where="bottom_cabinet_rvunhj_0/j_link_2")
    assert e.value.reason == "missing_handle" and "8 mm outside [15.0, 80.0] mm" in str(e.value)
    for name in ("j_link_2#Z", "j_link_2#F2", "j_link_0#C"):  # unknown / not admissible (edge_blocked) / another joint's row
        with pytest.raises(Reject) as e:
            AL.pick_handle_row(rows, name, (0.006, 0.080))
        assert e.value.reason == "spec"
    with pytest.raises(Reject) as e:
        AL.pick_handle_row([rec_bar, rec_knob, bar, knob_b])  # nothing inside 15-80 mm and no edge
    assert e.value.reason == "missing_handle" and "8 mm outside [15.0, 80.0] mm" in str(e.value)
    with pytest.raises(Reject):
        AL.pick_handle_row(rows, None, (0.09, 0.08))
    assert AL.handle_row_matches(bar, " j_link_2#C ") and not AL.handle_row_matches(bar, "") and not AL.handle_row_matches(bar, "j_link_2#c")


def test_load_handle_takes_the_alias_bar_of_the_office_cabinet_door():
    """The recorded office cabinet door (B4 inventory + icl rows, instance frame): by default the edge pinch j_link_2#F; with
    ``handle_row: j_link_2#C`` and ``pinch_width_min_m: 0.006`` the 8 mm vertical bar (h 0.73 m, approached from the front)
    keeps its grasp slide along its 0.28 m length (the length from the record's bar) and the template facts name the row."""
    gen = {"joint": "j_link_2", "handle_key": "bottom_cabinet__rvunhj"}

    def sampler():
        al = AL.ArticulationLarge.__new__(AL.ArticulationLarge)
        al.gp = SimpleNamespace(alias="cabinet", entry={"roles": {}}, gen=lambda k, d=None: gen.get(k, d))
        al.target_role, al.world, al.template_facts = "target", SimpleNamespace(scene_model="office_cubicles_right"), {}
        return al

    obj = SimpleNamespace(name="bottom_cabinet_rvunhj_0", category="bottom_cabinet", model="rvunhj", scale=[1.6025, 1.5341, 1.7222], links={},
                          joints={"j_link_2": SimpleNamespace(is_revolute=True, lower_limit=0.0, upper_limit=1.5708)})
    al = sampler()
    al.load_handle(obj)
    assert al.handle["kind"] == "edge" and al.template_facts["handle"]["row"] == "bottom_cabinet__rvunhj:j_link_2#F"
    assert "handle_row" not in al.template_facts["handle"] and "pinch_width_range_mm" not in al.template_facts["handle"]
    gen.update({"handle_row": "j_link_2#C", "pinch_width_min_m": 0.006})
    al = sampler()
    al.load_handle(obj)
    h, facts = al.handle, al.template_facts["handle"]
    assert facts["row"] == "bottom_cabinet__rvunhj:j_link_2#C" and facts["kind"] == "bar" and facts["width_mm"] == 8.0
    assert facts["handle_row"] == "j_link_2#C" and facts["pinch_width_range_mm"] == [6.0, 80.0]
    assert h["contact"] == pytest.approx([0.2856, -0.0348, -0.0658]) and h["pinch_axis"] == pytest.approx([0.0, -1.0, 0.0])
    assert h["approach_dir"] == pytest.approx([-1.0, 0.0, 0.0]) and "approach_dir_recorded" not in h  # straight from the front, no edge repair
    assert h["length"] == pytest.approx(0.2765)
    assert AL.bar_axis(h) == pytest.approx([0.0, 0.0, 1.0]) and AL.bar_offsets(h) == [0.0, 0.078, -0.078]  # the slide along the vertical bar
    assert AL.pinch_classes(h["pinch_axis"], h["kind"]) == ("fwd_pinch_h",)


# ----------------------------------------------------------------------------------------------------------------
# alignment (§5.5): frame, hang pose, contacts, brace ordering
# ----------------------------------------------------------------------------------------------------------------


def test_frame_axes_and_hang_quat():
    f, s, u = AW.frame_axes("+x", "+z")
    assert th.allclose(s, th.tensor([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError):
        AW.frame_axes("+x", "+x")
    q = AW.hang_quat([0.0, -1.0, 0.0], f, s, u)
    # the front axis maps onto the wall normal, the up axis onto world z
    assert th.allclose(C.quat_rotate(q, f), th.tensor([0.0, -1.0, 0.0]), atol=1e-5)
    assert th.allclose(C.quat_rotate(q, u), th.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def _wall_target_rows():
    """The reviewed wall-target rows of the shipped icl (``roles/candidates.csv``): (subtask, category, model, frame, bbox)."""
    import csv
    import os

    from ispatialgym.sampling import ground_isg as GI

    path = os.path.join(GI.DEFAULT_ICL, "roles", "candidates.csv")
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["subtask"] not in GI.WALL_SUBTASKS or row["role"] != "target" or row.get("kind") == "scene":
                continue
            frame = GI.wall_frame_of(row, row["subtask"])
            bbox = [float(row[k]) for k in ("bbox_x", "bbox_y", "bbox_z")] if all(row.get(k) for k in ("bbox_x", "bbox_y", "bbox_z")) else None
            out.append((row["subtask"], row["category"], row["model"], (row.get("representative") or "") == "1", frame, bbox))
    return out


def test_hang_quat_puts_every_reviewed_wall_frame_upright():
    """Owner 2026‑09‑16 22:55 (spec §5.5 / §6.9): for every reviewed wall target the hang pose built from the icl frame
    (as ``ground_isg`` writes it into the alias) puts the object's reviewed up axis along world +z and its front axis
    along the wall normal; for flat objects (plate, picture, painting: thinnest bbox axis ≤ 1/5 of the middle one) the
    thickness axis ends horizontal — along the normal — instead of world-up (the table-flat license plate of run
    0916‑212037 came from the painting-shaped default +x / +z applied to a +z / +y / +x frame)."""
    rows = _wall_target_rows()
    reps = [r for r in rows if r[3]]
    assert len(reps) >= 15, f"expected the wall representatives in the icl, found {len(reps)}"
    non_default = [r for r in reps if r[4].get("front_axis") != "+x" or r[4].get("up_axis") != "+z"]
    assert {(r[1]) for r in non_default} >= {"license_plate", "picture", "umbrella", "hanger", "fire_extinguisher", "guitar"}
    normals = [th.tensor([0.0, -1.0, 0.0]), th.tensor([1.0, 0.0, 0.0]), th.tensor([-0.6, 0.8, 0.0])]
    z = th.tensor([0.0, 0.0, 1.0])
    for subtask, cat, model, _rep, frame, bbox in rows:
        assert frame.get("front_axis") and frame.get("up_axis"), f"{subtask} {cat}/{model}: the icl row has no complete frame {frame}"
        assert set(frame) == {"front_axis", "up_axis"}, f"{cat}/{model}: unexpected frame keys {frame}"
        front, side, up = AW.frame_axes(frame["front_axis"], frame["up_axis"])
        for n in normals:
            q = AW.hang_quat(n, front, side, up)
            assert th.allclose(C.quat_rotate(q, up), z, atol=1e-5), f"{cat}/{model}: up {frame['up_axis']} not along world +z"
            assert th.allclose(C.quat_rotate(q, front), n, atol=1e-5), f"{cat}/{model}: front {frame['front_axis']} not along the wall normal"
            assert abs(float(C.quat_rotate(q, side)[2])) < 1e-5, f"{cat}/{model}: level axis not horizontal"
            if bbox is not None and sorted(bbox)[0] <= 0.2 * sorted(bbox)[1]:
                # flat object (one axis distinctly thinner than the other two — plates, pictures, paintings, boards; not rods):
                # the thinnest body axis is the reviewed front (thickness into the wall), so it hangs horizontal
                thin = th.zeros(3)
                thin[int(np.argmin(bbox))] = 1.0
                world_thin = C.quat_rotate(q, thin)
                assert abs(float(world_thin[2])) < 1e-5, f"{cat}/{model}: the thickness axis {bbox} points along world up in the hang pose"
                assert abs(abs(float(th.dot(world_thin, n))) - 1.0) < 1e-5, f"{cat}/{model}: the thickness axis is not along the wall normal"


def test_wall_frame_written_per_model_and_never_defaulted():
    """``ground_isg.write_wall_frame`` writes the dict form the sampler's ``target_param`` resolves (``front_axis`` /
    ``up_axis`` only; a bare axis letter gets its sign); a hand-set scalar is replaced by the dict; empty columns write
    nothing."""
    from ispatialgym.sampling import ground_isg as GI

    gen = {"front_axis": "+x"}
    row = {"frame_front": "+z", "frame_up": "+y"}
    assert GI.write_wall_frame(gen, "alignment-world_vertical", "license_plate", "hjdaqn", row) == {"front_axis": "+z", "up_axis": "+y"}
    assert gen == {"front_axis": {"license_plate/hjdaqn": "+z"}, "up_axis": {"license_plate/hjdaqn": "+y"}}
    GI.write_wall_frame(gen, "alignment-world_vertical", "hanger", "agrpio", {"frame_front": "z", "frame_up": "-x"})
    assert gen["front_axis"] == {"license_plate/hjdaqn": "+z", "hanger/agrpio": "+z"} and gen["up_axis"]["hanger/agrpio"] == "-x"
    gen_v: dict = {}
    assert GI.write_wall_frame(gen_v, "alignment-world_vertical", "painting", "dacihj", {"frame_front": "+x", "frame_up": "+z", "frame_level": "+y"}) == {
        "front_axis": "+x", "up_axis": "+z"}
    assert "level_axis" not in gen_v  # the icl's frame_level column is not a sampler parameter
    assert GI.write_wall_frame({}, "alignment-world_vertical", "x", "y", {"frame_front": "", "frame_up": ""}) == {}


def _wall_stub(gap=0.003):
    """An AlignmentWall without a simulator: the painting_s070 box (2.91 × 35.1 × 29.3 cm, front +x / up +z), its COM at
    the box centre, the wall spot of the first-demo instance (normal −y)."""
    S = object.__new__(AW.AlignmentWall)
    t, w, h = 0.0291, 0.351, 0.293
    S.hull = th.tensor([[sx * t / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    S.com_local = th.zeros(3)
    S.front, S.side, S.up = AW.frame_axes("+x", "+z")
    gen = {"wall_gap_m": gap, "contact_extreme_band_m": 0.03, "brace_offset_m": 0.005, "brace_clearance_m": 0.02,
           "brace_kinds": ["below_high", "above_low", "side_high"]}
    S.param = lambda k, d=None, v1=None: gen.get(k, d)
    spot = {"point": th.tensor([17.507, 3.94606, 1.449]), "normal": th.tensor([0.0, -1.0, 0.0]), "hang_z": 1.449, "wall_protrusion_m": 0.004}
    return S, spot, (t, w, h)


def test_wall_target_hangs_against_the_wall():
    """Owner 2026-09-27 (first-demo v1.2 review, 0926-065156): the painting hung 1 cm off the wall by construction
    (thickness/2 + 1 cm). Its back now stands ``wall_gap_m`` in front of the most protruding wall point of the patch."""
    S, spot, (t, w, h) = _wall_stub()
    for deg in (0.0, 20.0, -23.97):
        p, q = S.hang_pose(spot, math.radians(deg))
        back = (S.hull @ C.quat_to_mat(q).T + p) @ -spot["normal"]
        wall = float(spot["point"] @ -spot["normal"])
        assert wall - float(back.max()) == pytest.approx(0.004 + 0.003, abs=1e-5)  # protrusion + gap, at every tilt
    assert S.depth_extents() == (pytest.approx(t / 2), pytest.approx(t))


def test_wall_contacts_act_in_the_wall_plane():
    """Owner 2026-09-27 (0926-065156): push and brace act in the wall plane. The fingers point into the wall at every push
    and brace pose, the pads close in the wall plane (a finger's outer side meets the edge, its normal in the plane), every
    pre-contact lies in the contact's plane (the approach slides along the wall), the fingertips never pass the target's
    mid-plane (the fist stays off the wall), and the push carries the low end's CORNER: the finger side stands square to
    the corner's circle tangent at init and along the arc."""
    S, spot, (t, w, h) = _wall_stub()
    n = spot["normal"]
    for deg in (-23.97, 23.97, -15.0, 25.0):
        delta = math.radians(deg)
        geo = S.contacts_for(spot, delta)
        corner, tangent = geo["E_push"], geo["push"]["direction"]
        assert float(corner[2]) == pytest.approx(float(geo["hull_tilted"][:, 2].min()), abs=1e-4)  # the lowest point = the low end's corner
        assert float(tangent[2]) > 0 and abs(float(th.dot(tangent, n))) < 1e-6  # the corner rises, in the wall plane
        assert geo["push"]["arm"] == ("right" if deg < 0 else "left")  # the arm on the low end's side
        mid_depth = float(geo["C"] @ -n)
        for v in geo["push"]["variants"]:
            for key in ("pre", "init", "mid", "goal"):
                p, q = v[key]
                a, c = C.approach_axis(q), C.closing_axis(q)
                assert float(th.dot(a, -n)) >= math.cos(math.radians(26.0)) - 1e-6, f"{v['wrist']} {key}: the fingers do not point into the wall"
                assert abs(float(th.dot(c, n))) < 1e-5, f"{v['wrist']} {key}: the pads do not close in the wall plane"
                assert float(C.fingertip(v[key]) @ -n) <= mid_depth + 1e-5  # the fingertips on the mid-plane, off the wall
            assert abs(float(th.dot(v["pre"][0] - v["init"][0], n))) < 1e-6  # in-plane approach
            assert float(th.dot(v["init"][0] - v["pre"][0], tangent)) == pytest.approx(C.PRE_DIST, abs=1e-5)
            assert abs(abs(float(th.dot(C.closing_axis(v["init"][1]), tangent))) - 1.0) < 1e-5  # the finger side square to the tangent
        for b in geo["braces"]:
            a, c = C.approach_axis(b["pose"][1]), C.closing_axis(b["pose"][1])
            assert float(th.dot(a, -n)) >= math.cos(math.radians(26.0)) - 1e-6 and abs(float(th.dot(c, n))) < 1e-5
            assert abs(float(th.dot(b["pre"][0] - b["pose"][0], n))) < 1e-6 and abs(float(th.dot(b["a"], n))) < 1e-6
            assert b["kind"] in AW.BRACE_KINDS and S.brace_clear(geo, b)


def test_wall_push_finger_side_carries_the_corner():
    """With the gripper assets: the corner meets the middle of the finger side - the closed fist's hull at the recorded push
    poses overlaps the (rotating) painting only where its lower edge rests on the finger's leading edge beside the corner
    (< half the finger's width across the tangent) at init, δ/2 and the goal, identically along the arc, and is clear of it
    at the pre-contact."""
    from ispatialgym.oracle import r1pro_model as M

    pts = M.finger_points_eef(C.FINGER_CLOSED, "right")
    if pts is None:
        pytest.skip("robot assets unavailable")
    fist = th.tensor(pts, dtype=th.float32)
    S, spot, (t, w, h) = _wall_stub()
    half = th.tensor([t / 2, w / 2, h / 2])

    def overlap(pose_f, pose_o):
        world = fist @ C.quat_to_mat(pose_f[1]).T + pose_f[0]
        loc = (world - pose_o[0]) @ C.quat_to_mat(pose_o[1])
        return float((half - loc.abs()).min(dim=1).values.max())

    n = spot["normal"]
    for deg in (-23.97, 18.0):
        delta = math.radians(deg)
        geo = S.contacts_for(spot, delta)
        p0, q0 = geo["pose_tilted"]
        across = AW.fist_half_extents(t / 2)[1]
        for v in geo["push"]["variants"]:
            assert overlap(v["pre"], geo["pose_tilted"]) < -0.02
            ov = []
            for frac in (0.0, 0.5, 1.0):
                rot = C.quat_from_axis_angle(n, -delta * frac)
                pose_o = (geo["C"] + C.quat_rotate(rot, p0 - geo["C"]), C.qmul(rot, q0))
                pose_f = (geo["C"] + C.quat_rotate(rot, v["init"][0] - geo["C"]), C.qmul(rot, v["init"][1]))
                ov.append(overlap(pose_f, pose_o))
            assert max(ov) < 0.5 * across and max(ov) - min(ov) < 1e-4, f"{v['wrist']}: the fist overlaps the target {ov}"


def _disc_stub(radius=0.282, thickness=0.048, n_rim=64, phase=0.0):
    """``_wall_stub`` with a round wall clock of jerval's size: ``n_rim`` rim points starting at ``phase``, the COM hinge at
    the centre."""
    S, spot, _ = _wall_stub()
    angles = [phase + 2 * math.pi * k / n_rim for k in range(n_rim)]
    S.hull = th.tensor([[x, radius * math.cos(a), radius * math.sin(a)] for a in angles for x in (-thickness / 2, thickness / 2)], dtype=th.float32)
    return S, spot


def test_wall_round_target_pushes_the_level_axis_end_without_brace():
    """Owner 2026-10-11 (world_vertical#02, round wall clocks): a disc turning about its COM hinge has no low corner - its
    lowest point stays under the hinge at every tilt, where the rim's tessellation decided the push arm and direction (the
    -25 deg template filter found no stance on any wall). The push now acts at the low end of the tilted level axis, its arm
    and rising direction follow the tilt sign, and the disc gets no brace (a stop at the level outline touches the unchanged
    tilted outline). A box keeps its lowest corner and its braces up to the deviation cap."""
    for phase in (0.0, 0.049):
        S, spot = _disc_stub(phase=phase)
        n = spot["normal"]
        for deg in (25.0, -25.0, 15.0, -18.0):
            geo = S.contacts_for(spot, math.radians(deg))
            assert geo["no_corner"] and geo["braces"] == []
            assert th.allclose(geo["E_push"], geo["E_low"], atol=1e-6)  # the low end of the tilted level axis
            t = geo["push"]["direction"]
            assert float(t[2]) > 0.5 and abs(float(th.dot(t, n))) < 1e-6  # rises, in the wall plane
            assert geo["push"]["arm"] == ("right" if deg < 0 else "left")  # the arm on the low end's side, as for the box
    S, spot, _ = _wall_stub()
    for deg in (AW.MAX_DEVIATION_DEG, -AW.MAX_DEVIATION_DEG, 15.0):
        geo = S.contacts_for(spot, math.radians(deg))
        assert not geo["no_corner"] and geo["braces"]


def test_wall_screen_keeps_push_only_stances_of_a_round_target(monkeypatch):
    """Phase 2 steps 2-3 for the disc: a stance whose push reaches survives without any brace (the 85d54c41 sim check raised
    IndexError on the missing brace point); a box without a reachable brace still has none."""
    reach = SimpleNamespace(volume_ok=lambda rv, st, bz: True)
    monkeypatch.setattr(P0, "reach_volume", lambda: None)
    monkeypatch.setattr(P0, "base_z", lambda s: 0.0)
    monkeypatch.setattr(P0, "camera_poses", lambda s, sts: [None] * len(sts))
    monkeypatch.setattr(P0, "in_frame", lambda s, cam, pts: len(pts) >= 2)
    monkeypatch.setattr(P0, "axis_angle", lambda s, cam, c: 0.05)
    monkeypatch.setattr(P0, "angle_class", lambda ang: 0)
    monkeypatch.setattr(P0, "best_class", lambda classed: [st for st, _ in classed])
    S, spot = _disc_stub()
    S.contact_specs = lambda geo: (reach, [])
    cands = [SimpleNamespace(extras={}, focus=None) for _ in range(3)]
    stats = {}
    out = S.screen(cands, S.contacts_for(spot, math.radians(-20.0)), stats)
    assert len(out) == 3 and all(st.extras["brace_kinds"] == [] and st.extras["class_pool"] == "best" for st in out) and stats["step3_visible"] == 3
    B, spot_b, _ = _wall_stub()
    B.contact_specs = lambda geo: (reach, [])
    assert B.screen([SimpleNamespace(extras={}, focus=None)], B.contacts_for(spot_b, math.radians(-20.0))) == []


def test_brace_order():
    assert AW.brace_candidate_order(height=0.6, width=0.8) == ["below_high", "above_low", "side_high"]
    assert AW.brace_candidate_order(height=0.1, width=0.8) == ["below_high", "above_low"]  # side_high needs height >= width/2
    assert AW.brace_candidate_order(0.6, 0.8, kinds=["above_low", "side_high"]) == ["above_low", "side_high"]


def _bar_hull(width=0.8, height=0.1, thickness=0.02, n=9):
    xs = th.linspace(-thickness / 2, thickness / 2, 3)
    ys = th.linspace(-width / 2, width / 2, n)
    zs = th.linspace(-height / 2, height / 2, 3)
    return th.stack(th.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)


def test_alignment_contacts_geometry():
    """A towel bar tilted by δ about the wall normal: E_low is the lower end, the push tangent raises it, the arc ends at
    the level pose, the brace candidates sit on the level hull as §5.5 describes."""
    n = th.tensor([0.0, -1.0, 0.0])  # wall normal toward the room
    hull = _bar_hull()
    front, side, up = AW.frame_axes("+x", "+z")
    q_up = AW.hang_quat(n, front, side, up)
    delta = math.radians(30)
    q_t = C.qmul(C.quat_from_axis_angle(n, delta), q_up)
    C_ = th.tensor([0.0, -0.02, 1.45])
    pts_t = C.transform_points((C_, q_t), hull)
    pts_l = C.transform_points((C_, q_up), hull)
    level_axis_t = C.quat_rotate(q_t, side)
    E_low, E_high = AW.extreme_points(pts_t, level_axis_t, 0.03)
    assert E_low[2] < E_high[2]
    assert float(th.linalg.norm((E_low - C_)[:2] + (E_high - C_)[:2])) < 1e-4  # symmetric ends about the hinge
    a = AW.leveling_tangent(E_low, C_, n, delta)
    assert a[2] > 0 and abs(float(th.dot(a, n))) < 1e-6 and abs(float(th.dot(a, E_low - C_))) < 1e-5  # tangent, in the wall plane, rising
    push = C.contact_eef(E_low, a, closing=n)
    assert th.allclose(C.pad_center(push), E_low, atol=1e-6)
    mid, goal = C.arc_waypoints(push, C_, n, -delta, n=2)
    # the goal contact is the same material point at the level pose
    E_low_level = C_ + C.quat_rotate(C.quat_from_axis_angle(n, -delta), E_low - C_)
    assert th.allclose(C.pad_center(goal), E_low_level, atol=1e-5)
    assert abs(float(C.pad_center(goal)[2]) - float(C_[2])) < 0.06  # level: the end is at the hinge height (bar half-height)
    assert abs(float(th.linalg.norm(C.pad_center(mid) - C_)) - float(th.linalg.norm(E_low - C_))) < 1e-5  # on the circle
    # brace candidates on the level hull, E_high's side
    high_sign = 1.0 if float(th.dot(E_high - C_, level_axis_t)) > 0 else -1.0
    side_l = C.quat_rotate(q_up, side)
    B, a_b = AW.brace_point(pts_l, "below_high", side_l, high_sign, C_)
    assert a_b[2] == 1.0 and float((B - C_) @ side_l * high_sign) > 0 and abs(float(B[2]) - float(pts_l[:, 2].min())) < 1e-4
    B2, a_b2 = AW.brace_point(pts_l, "above_low", side_l, high_sign, C_)
    assert a_b2[2] == -1.0 and float((B2 - C_) @ side_l * high_sign) < 0 and abs(float(B2[2]) - float(pts_l[:, 2].max())) < 1e-4
    B3, a_b3 = AW.brace_point(pts_l, "side_high", side_l, high_sign, C_)
    assert abs(float(a_b3[2])) < 1e-6 and float(th.dot(a_b3, C_ - B3)) > 0  # horizontal, toward C
    eef = C.fingertip_eef(B, a_b, closing=n, clearance=0.005)
    assert th.allclose(C.fingertip(eef), B - 0.005 * a_b, atol=1e-6)  # fingertip face 5 mm below B
    # push arm = the arm on E_low's side (robot facing the wall: left = (n_y, -n_x))
    left = th.tensor([float(n[1]), -float(n[0]), 0.0])
    arm = AW.push_arm_of(E_low, C_, left)
    assert arm == ("left" if float(th.dot((E_low - C_)[:2], left[:2])) > 0 else "right")
    assert AW.other_arm(arm) != arm
    # symmetric_180: the half turn about the front axis is a unit quaternion of angle pi
    q180 = AW.half_turn_equivalence(front)
    assert abs(math.hypot(*q180) - 1.0) < 1e-6 and abs(q180[3]) < 1e-6


# ----------------------------------------------------------------------------------------------------------------
# IORS large-scale (§5.6): init-surface classes, edges, erosion, coupling
# ----------------------------------------------------------------------------------------------------------------


def test_init_surface_class_enumeration():
    assert IL.init_surface_class("under", ["floor"], ["bed_rrcvaq_0"]) == "floor"
    assert IL.init_surface_class("ontop", None, ["bed_rrcvaq_0"], alias_support_token="floor") == "floor"
    assert IL.init_surface_class("under", ["bed_rrcvaq_0"], ["bed_rrcvaq_0"]) == "partner"
    assert IL.init_surface_class("under", None, ["bed_rrcvaq_0"], alias_support_token="bed") == "partner"
    assert IL.init_surface_class("under", ["desk_xhjsub_0"], ["bed_rrcvaq_0"]) == "support"
    assert IL.init_surface_class("ontop", {"category": ["desk", "nightstand"]}, ["bed_rrcvaq_0"]) == "support"
    assert IL.init_surface_class("ontop", None, ["bed_rrcvaq_0"], alias_support_token="desk") == "support"
    with pytest.raises(ValueError):
        IL.init_surface_class("ontop", ["bed_rrcvaq_0"], ["bed_rrcvaq_0"])
    st = "inter_object_relation_satisfaction-under_large_scale"
    assert IL.alias_support_token(f"{st}-toy_figure-bed-floor", st) == "floor"
    assert IL.alias_support_token(f"{st}-toy_figure-bed-desk", st) == "desk"
    assert IL.alias_support_token("under", st) is None
    assert set(IL.INIT_SURFACE_CLASSES) == {"support", "partner", "floor"}
    # A5 (owner decision 2026-09-16): the class is fixed by the alias — D1's floor aliases bind the support to a room dict,
    # the own-top fallback repeats the partner token (``…-<partner>-<partner>``); the selection order is alias-level
    assert IL.init_surface_class("under", {"room": "childs_room_0"}, ["desk_xhjsub_0"], alias_support_token="floor") == "floor"
    assert IL.init_surface_class("ontop", {"room": "childs_room_0"}, ["desk_xhjsub_0"]) == "floor"
    assert IL.alias_partner_token(f"{st}-sandal-bed-bed", st) == "bed" and IL.alias_support_token(f"{st}-sandal-bed-bed", st) == "bed"
    assert IL.init_surface_class("under", None, ["bed_rrcvaq_0"], alias_support_token="bed", alias_partner_token="bed") == "partner"
    assert IL.init_surface_class("under", ["bed_rrcvaq_0"], ["bed_rrcvaq_0"], alias_support_token="bed", alias_partner_token="bed") == "partner"
    with pytest.raises(ValueError):
        IL.init_surface_class("ontop", None, ["desk_xhjsub_0"], alias_support_token="desk", alias_partner_token="desk")
    assert IL.SELECTION_ORDER == {"under": ("support", "partner", "floor"), "ontop": ("support", "floor")}
    assert IL.alias_tokens(f"{st}-sandal-bed-desk", st) == ["sandal", "bed", "desk"] and IL.alias_tokens(st, st) == []


def test_push_under_record_and_geometry_constants():
    # owner decision 2026-09-16 item 7: the drop + straight floor push of the large-scale under relation
    q = th.tensor([0.0, 0.0, 0.0, 1.0])
    start, mid, goal = th.tensor([1.0, 0.0, 0.05]), th.tensor([1.1, 0.0, 0.05]), th.tensor([1.2, 0.0, 0.05])
    pre = (th.tensor([0.95, 0.0, 0.12]), q)
    rec = IL.PushUnderRecord("right", pre, (start, q), (mid, q), (goal, q), th.tensor([1.0, 0.0, 0.0]), 0.2, "sandal.n.01_1", stance="placement",
                             pitch_deg=45.0, blade_standoff_m=0.0134, push_height_m=0.045, half_extent_m=0.05, drop_cell=[0.9, 0.0], goal_cell=[1.1, 0.0])
    d = rec.to_sidecar()
    assert rec.kind == "push" and d["kind"] == "push_under" and d["arm"] == "right" and d["object"] == "sandal.n.01_1" and d["stance"] == "placement"
    assert set(("pre", "init", "mid", "goal", "direction", "distance_m")) <= set(d)
    assert d["distance_m"] == pytest.approx(0.2) and d["direction"] == [1.0, 0.0, 0.0]
    assert d["pitch_deg"] == 45.0 and d["blade_standoff_m"] == pytest.approx(0.0134) and d["drop_cell"] == [0.9, 0.0] and d["goal_cell"] == [1.1, 0.0]
    # the F6 PushContact accepts the block (unknown keys land in ``extra``)
    from ispatialgym.oracle import sidecar as SC
    pc = SC.PushContact.from_dict(d, "contacts.push")
    assert pc.extra["kind"] == "push_under" and pc.mid is not None and len(pc.waypoints) == 3
    # geometry: the goal centre 5-10 cm inside the edge, the drop half + 10 cm outside, the drop cell 0.25-0.6 m ahead of the base
    assert 0.05 <= IL.PUSH_UNDER_INSIDE_M <= 0.10 and IL.PUSH_UNDER_DROP_EXTRA_M == pytest.approx(0.10) and 0.02 <= IL.PUSH_UNDER_DROP_HEIGHT_M <= 0.15
    for so in IL.PUSH_UNDER_STAND_OFFS:
        assert IL.PUSH_UNDER_DROP_AHEAD_M[0] <= so - (0.05 + 0.10) or so - (0.12 + 0.10) <= IL.PUSH_UNDER_DROP_AHEAD_M[1]
    assert all(t >= 0.75 and k <= -0.9 for t, k in IL.FLOOR_TRUNKS)


def test_a8_deviation_cap_constant():
    # A8: |δ| ≤ 25° until the two-stroke push exists (the module caps a wider customs range at MAX_DEVIATION_DEG)
    assert AW.MAX_DEVIATION_DEG == 25.0


def test_aabb_edges_and_nearest_and_erosion():
    edges = IL.aabb_edges([0.0, 0.0, 0.0], [2.0, 1.0, 0.5])
    assert [e.name for e in edges] == ["+x", "-x", "+y", "-y"]
    e = IL.nearest_edge(edges, (2.5, 0.5))
    assert e.name == "+x" and th.allclose(e.inward, th.tensor([-1.0, 0.0]))
    xy = e.base_xy(0.45, 0.2)
    assert xy[0] == pytest.approx(2.45) and xy[1] == pytest.approx(0.5 + 0.2 * float(e.tangent[1]))
    assert IL.edge_overhang((2.0, 0.5), (1.0, 0.0), [0.0, 0.0, 0.0], [2.11, 1.0, 0.5]) == pytest.approx(0.11)
    assert IL.edge_overhang((2.0, 0.5), (1.0, 0.0), [0.0, 0.0, 0.0], [2.03, 1.0, 0.5]) == 0.0  # < 5 cm does not count
    cells = th.tensor([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
    rim = th.tensor([[0.0, 0.5], [2.0, 0.5]])
    kept = IL.erode_cells(cells, rim, 0.15)
    assert kept.tolist() == [[1.0, 0.5]]
    grid = {"resolution": 0.5, "origin": [0.0, 0.0], "shape": [2, 2], "grid": [[0.1, 0.3], [None, 0.4]]}
    cells = IL.under_clearance_cells(grid, 0.25)
    assert sorted(cells.tolist()) == [[0.25, 0.75], [0.75, 0.75]]
    assert IL.under_admissible(0.40, 0.20) and not IL.under_admissible(0.35, 0.20)


def test_feasible_set_intersection():
    class Gr:
        def __init__(self, name):
            self.name = name

    g1, g2 = Gr("grasp_1"), Gr("grasp_2")
    F_init = [(0.0, g1), (0.0, g2), (math.pi / 4, g1)]
    F_goal = [(0.0, Gr("grasp_2")), (math.pi / 4, Gr("grasp_1")), (math.pi / 2, Gr("grasp_1"))]
    F = IL.intersect_feasible(F_init, F_goal)
    assert [(round(y, 3), g.name) for y, g in F] == [(0.0, "grasp_2"), (round(math.pi / 4, 3), "grasp_1")]


def test_stance_records_large_scale_blocks():
    """The sidecar blocks the large-scale programs read exist in the draft records (F6 shapes)."""
    from ispatialgym.sampling.v2.stances import ContactRecord, SidecarDraft

    sd = SidecarDraft("alignment-world_vertical-painting-wall", "alignment")
    face = P0.make_face("wall", [0.0, -1.0, 0.0], [0.0, 0.0, 1.45])
    init = P0.ray_stance(face, 0.1, 1.0, 0.5)
    manip = P0.square_stance(face, 0.5, 0.1, 0.5, -0.3)
    manip.arms = ["left", "right"]
    sd.add_stance(init)
    sd.add_stance(manip)
    q = C.eef_quat([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    p = (th.zeros(3), q)
    sd.add_contact(ContactRecord("push", "left", p, p, p, p, th.tensor([0.0, 0.0, 1.0]), 0.1, object="painting.n.01_1"))
    sd.add_contact(ContactRecord("brace", "right", p, p, direction=th.tensor([0.0, 0.0, 1.0]), brace_kind="below_high", object="painting.n.01_1"))
    d = sd.to_dict()
    assert d["stances"]["manipulation"]["face"] == "wall" and d["stances"]["manipulation"]["arms"] == ["left", "right"]
    assert d["contacts"]["push"]["mid"] is not None and d["contacts"]["brace"]["kind"] == "below_high"
    assert d["stances"]["init"]["stand_off"] == 1.0


def test_bar_axis_and_offsets():
    from ispatialgym.sampling.v2.tasks.articulation_large import bar_axis, bar_offsets

    h = {"kind": "bar", "approach_dir": [-1.0, 0.0, 0.0], "pinch_axis": [0.0, 0.0, -1.0], "length": 0.6424}
    ax = bar_axis(h)
    assert pytest.approx(abs(ax[1]), abs=1e-6) == 1.0 and abs(ax[0]) < 1e-6 and abs(ax[2]) < 1e-6  # the bar runs along y
    assert bar_offsets(h) == [0.0, 0.131, -0.131, 0.261, -0.261]  # usable half 0.3212 - 0.06 = 0.2612: its half and itself
    assert bar_offsets({"kind": "bar", "length": 0.3959, "approach_dir": [1, 0, 0], "pinch_axis": [0, 0, 1]}) == [0.0, 0.069, -0.069, 0.138, -0.138]
    assert bar_offsets({"kind": "knob"}) == [0.0]
    assert bar_offsets({"kind": "bar", "length": 0.2, "approach_dir": [1, 0, 0], "pinch_axis": [0, 0, 1]}) == [0.0, 0.04, -0.04]  # usable half 0.04
    assert bar_offsets({"kind": "bar", "length": 0.15, "approach_dir": [1, 0, 0], "pinch_axis": [0, 0, 1]}) == [0.0]  # below the 4 cm step


def test_wall_hinge_resistance_record():
    """Owner 2026‑09‑16 22:05 (spec §7): the wall hinge record carries a Coulomb friction torque and a viscous damping from
    the target's inertia about the hinge axis (I_n = m (w² + h²) / 12): friction = I_n · 10 rad/s²; the damping is the brake
    of 2026-09-22, friction / 0.0025 rad/s (the plain damper I_n / 0.02 s let a residual gravity torque creep the levelled
    painting 0.25 deg/s out of tolerance, 0922-180242) - a legacy record's damping is raised to that floor by
    ``hinge_brake_damping``. The frictionless (axial) record has neither key; the loader-side drive reads them back as the
    drive's max force and per-degree damping."""
    from ispatialgym.sampling import constraints as K

    res = K.hinge_resistance(2.25, (0.354, 0.2946))  # the painting_s070 of demo 0916-150205 (probe 2026-09-16 23:24)
    assert res["inertia_kgm2"] == pytest.approx(2.25 * (0.354**2 + 0.2946**2) / 12)
    assert res["friction"] == pytest.approx(res["inertia_kgm2"] * 10.0) and res["damping"] == pytest.approx(res["friction"] / K.HINGE_BRAKE_OMEGA_RAD_S)
    assert 0.3 < res["friction"] < 0.5 and 150 < res["damping"] < 200  # the painting: ~0.4 N·m (≈ 1.6 N at the pad), Coulomb down to 0.0025 rad/s
    assert res["brake_omega_rad_s"] == K.HINGE_BRAKE_OMEGA_RAD_S == 0.0025
    legacy = K.hinge_resistance(2.25, (0.354, 0.2946), omega_brake=0.0)  # the record as written before the brake
    assert legacy["damping"] == pytest.approx(legacy["inertia_kgm2"] / 0.02) and 1.5 < legacy["damping"] < 2.5
    assert K.hinge_brake_damping(legacy["friction"], legacy["damping"]) == pytest.approx(res["damping"])
    assert K.hinge_brake_damping(res["friction"], res["damping"]) == pytest.approx(res["damping"])  # a braking record is left alone
    assert K.hinge_brake_damping(None, None) is None and K.hinge_brake_damping(None, 1.5) == 1.5  # the frictionless axial hinge
    h = K.Hinge("/World/x/isg_hinge", "painting", "walls", "base_link", th.tensor([1.0, 2.0, 1.45]), th.tensor([0.0, -1.0, 0.0]),
                friction=res["friction"], damping=res["damping"], facts={k: v for k, v in res.items() if k not in ("friction", "damping")})
    rec = h.to_tro("painting.n.01_1", "wall.n.01_1")
    assert rec["type"] == "revolute" and rec["limits"] is None and rec["friction"] == pytest.approx(res["friction"]) and rec["damping"] == pytest.approx(res["damping"])
    assert "clamp(-damping * omega, +-friction)" in rec["drive"] and rec["resistance_facts"]["mass_kg"] == 2.25
    plain = K.Hinge("/World/y/isg_hinge", "board_game", "table", "base_link", th.zeros(3), th.tensor([0.0, 0.0, 1.0])).to_tro("board_game.n.01_1", "table.n.01_1")
    assert "friction" not in plain and "damping" not in plain and "drive" not in plain  # the axial table hinge stays frictionless
    # the loader's drive: absent keys keep a frictionless hinge (no drive at all)
    assert K.apply_hinge_drive.__doc__ and "π / 180" in K.apply_hinge_drive.__doc__


def test_counterbalance_switches_gravity_off_on_the_leaf_assembly_and_damps_the_joint():
    """Owner decision 2026-09-19 (articulation_open_large_scale): a horizontal-axis appliance door is spring-balanced -
    gravity off on every link of the leaf assembly (the moving link and the links fixed to it), viscous damping on the
    joint; links that do not exist or cannot switch are skipped and reported, ``damping`` None leaves the drive alone."""
    from ispatialgym.sampling import constraints as K

    class Link:
        def __init__(self):
            self.gravity = True

        def disable_gravity(self):
            self.gravity = False

    class Joint:
        damping = 0.05

    class Target:
        links = {"leaf": Link(), "glass": Link(), "base_link": Link()}

    t, j = Target(), Joint()
    applied = K.apply_counterbalance(t, j, ["leaf", "glass", "handle_missing"], 5.0)
    assert applied == {"links": ["leaf", "glass"], "damping": 5.0}
    assert not t.links["leaf"].gravity and not t.links["glass"].gravity and t.links["base_link"].gravity
    assert j.damping == 5.0
    j2 = Joint()
    assert K.apply_counterbalance(t, j2, [], None) == {"links": []} and j2.damping == 0.05


def test_push_under_attempt_budget_rejects_time_budget_with_the_stage():
    """Owner 2026‑09‑16 22:55 item 4 (run 0916‑190702: > 2 h in one push-under attempt, rc 124, nothing recorded): the
    per-attempt budget (900 s, ``generation.attempt_budget_s``) stops the attempt with ``Reject("time_budget")`` naming the
    stage reached; before the deadline the check is silent."""
    import time

    from ispatialgym.sampling.sim_utils import Reject

    class S:
        def num(self, key, default):
            return default

    s = S()
    assert IL.ATTEMPT_BUDGET_S == 900.0
    s._attempt_t0 = time.time()
    s._deadline = s._attempt_t0 + 100.0
    IL.IORSLarge._check_deadline(s, "C candidate 3 of 56 (F_goal feasible set)")  # silent before the deadline
    s._deadline = s._attempt_t0 - 1.0
    with pytest.raises(Reject) as e:
        IL.IORSLarge._check_deadline(s, "C candidate 3 of 56 (F_goal feasible set)")
    assert e.value.reason == "time_budget" and "C candidate 3 of 56" in str(e.value) and e.value.details.get("stage", "").startswith("C candidate")


# ----------------------------------------------------------------------------------------------------------------
# hybrid base-follow (§5.3 A4 / D-011 C; ``articulation_hybrid``)
# ----------------------------------------------------------------------------------------------------------------

AH = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_hybrid")


def test_hybrid_dense_grid_contains_the_fractions_and_bounds_the_step():
    """The dense joint grid runs from q_init (excluded) to the last required waypoint, contains every §5.3 fraction value
    in between and splits every interval into equal sub-steps no larger than the step (10° default), either direction."""
    q0, q1 = 0.0, math.radians(90.0)
    marks = [q0 + k * (q1 - q0) for k in (0.25, 0.5, 0.75, 1.0)]
    vals = AH.dense_values(q0, q1, marks, math.radians(10.0))
    assert vals[0] > q0 and vals[-1] == pytest.approx(q1)
    for m in marks:
        assert any(abs(v - m) < 1e-9 for v in vals)
    steps = np.diff([q0] + vals)
    assert steps.max() <= math.radians(10.0) + 1e-9 and steps.min() > 0
    # 22.5° intervals -> 3 sub-steps of 7.5° each: 12 values
    assert len(vals) == 12
    # closing: descending grid, the marks still inside
    down = AH.dense_values(q1, q0, [q1 - 0.25 * (q1 - q0)], math.radians(10.0))
    assert down[0] < q1 and down[-1] == pytest.approx(q0) and all(np.diff(down) < 0)
    assert AH.dense_values(0.3, 0.3, [], 0.1) == []
    # prismatic: 5 cm steps, a 0.3 m pull with the 0.25 mark (0.075) inside
    pv = AH.dense_values(0.0, 0.3, [0.075, 0.15, 0.225, 0.3], 0.05)
    assert max(np.diff([0.0] + pv)) <= 0.05 + 1e-9 and any(abs(v - 0.075) < 1e-9 for v in pv)


def test_hybrid_candidates_ordered_by_displacement_and_deduped():
    b_prev = (1.0, 2.0, 0.0)
    b_rigid = (1.4, 2.3, 0.6)  # the rigid-sweep pose 0.5 m / 0.6 rad away
    cands = AH.lattice_candidates(b_prev, b_rigid, 0.10, 0.30, math.radians(15.0), math.radians(15.0), 0.3, alphas=[0.25, 0.5, 1.0])
    costs = [c[0] for c in cands]
    assert costs == sorted(costs) and all(c[0] > 0 for c in cands)  # b_prev itself is not a candidate
    tags = [c[2] for c in cands]
    assert "alpha=0.25" in tags and "alpha=1" in tags and "lattice" in tags
    # the alpha poses interpolate position and yaw together
    a = next(c for c in cands if c[2] == "alpha=0.5")
    assert a[1] == pytest.approx((1.2, 2.15, 0.3))
    # no duplicates (5 mm / 0.5° key), every lattice position within the radius, yaws between the current and the rigid yaw ± 15°
    keys = {(round(c[1][0], 3), round(c[1][1], 3), round(c[1][2], 3)) for c in cands}
    assert len(keys) == len(cands)
    for _, p, tag in cands:
        if tag == "lattice":
            assert math.hypot(p[0] - 1.0, p[1] - 2.0) <= 0.30 + 1e-9
            assert -math.radians(15.0) - 1e-9 <= p[2] <= 0.6 + math.radians(15.0) + 1e-9
    yaws = AH.yaw_candidates(0.0, 0.6, math.radians(15.0), math.radians(15.0))
    assert yaws[0] == 0.0 and pytest.approx(0.6) in [pytest.approx(y) for y in yaws] and min(yaws) == pytest.approx(-math.radians(15.0))
    # the displacement cost weighs the yaw at w metres per radian
    assert AH.displacement_cost((0, 0, 0), (0.3, 0.4, 1.0), 0.3) == pytest.approx(0.5 + 0.3)
    assert AH.pose2d_lerp((0, 0, math.pi - 0.1), (0, 0, -math.pi + 0.1), 0.5)[2] == pytest.approx(math.pi, abs=1e-6) or \
        AH.pose2d_lerp((0, 0, math.pi - 0.1), (0, 0, -math.pi + 0.1), 0.5)[2] == pytest.approx(-math.pi, abs=1e-6)  # shortest way through ±π


def test_hybrid_planar_stand_ins():
    boxes = [(0.0, 0.0, 1.0, 0.5)]
    assert AH.disc_clear_of_boxes((1.5, 0.25), 0.3, boxes)  # 0.5 m from the box edge
    assert not AH.disc_clear_of_boxes((1.2, 0.25), 0.3, boxes)  # 0.2 m from the edge < 0.3
    assert not AH.disc_clear_of_boxes((0.5, 0.25), 0.3, boxes)  # inside
    assert AH.disc_clear_of_boxes((1.25, 0.75), 0.3, boxes)  # diagonal: 0.25, 0.25 -> 0.354 > 0.3
    assert AH.leaf_clear_of_base(th.tensor([[1.0, 1.0], [1.2, 1.0]]), (0.0, 0.0), 0.33)
    assert not AH.leaf_clear_of_base(th.tensor([[0.2, 0.1]]), (0.0, 0.0), 0.33)
    assert AH.leaf_clear_of_base(th.zeros(0, 2), (0.0, 0.0), 0.33)


def _hybrid_poses():
    q_init, q_goal = 0.0, 1.2
    poses = [AH.FollowPose(0.0, q_init, (4.05, -1.45, math.pi), [0.1] * 7, chosen="stance")]
    for k, chosen, step in ((0.25, "arm_only", 0.0), (0.5, "alpha=0.5", 0.2), (0.75, "lattice", 0.1), (1.0, "arm_only", 0.0)):
        prev = poses[-1].base_pose2d
        b = (prev[0] + step, prev[1] - step / 2, prev[2] - step)
        poses.append(AH.FollowPose(k, q_init + k * (q_goal - q_init), b, [0.1 + k] * 7, fk_residual_m=0.0005, joint_limit_margin_rad=0.2,
                                   base_step_m=math.hypot(step, step / 2), base_step_rad=step, chosen=chosen, n_ik=3))
    return q_init, q_goal, poses


def test_hybrid_path_records_mirror_the_drawer_base_follow():
    q_init, q_goal, poses = _hybrid_poses()
    marks = {"0.25": 0.3, "0.5": 0.6, "0.75": 0.9, "1.0": 1.2}
    bf = AH.base_follow_dict(poses, marks)
    assert set(bf) == set(marks) and bf["0.5"] == list(poses[2].base_pose2d)
    # only the followed fractions are recorded (the passive continuation has no base pose)
    assert set(AH.base_follow_dict(poses[:3], marks)) == {"0.25", "0.5"}
    st = AH.path_stats(poses, q_init, q_goal)
    assert st["n_poses"] == 5 and st["n_base_moves"] == 2 and st["followed_fraction"] == 1.0 and st["arm_only_fraction"] == 0.5
    hp = AH.HybridPath("right", (0.5, -0.2), poses, bf, "left", [0.0] * 7, 0.05, {"revolute_rad": math.radians(10.0)}, st, AH.knobs(None))
    block = hp.to_sidecar()
    assert block["mode"] == "hybrid" and block["poses"][0]["k"] == 0.0 and len(block["poses"][2]["arm_q"]) == 7 and block["poses"][2]["chosen"] == "alpha=0.5"
    assert block["knobs"]["step_revolute_deg"] == 10.0 and "record_eef" not in block["knobs"]
    # generation.hybrid_follow overrides the defaults
    assert AH.knobs({"lattice_radius_m": 0.8, "budget_s": None})["lattice_radius_m"] == 0.8 and AH.knobs({"budget_s": None})["budget_s"] == 600.0


def test_hybrid_sidecar_schema_and_base_follow_targets(tmp_path):
    """The oracle's sidecar accepts ``follow_mode: hybrid`` (``base_follow`` per followed fraction as in base mode plus the
    dense ``base_follow_path``); ``BaseFollow.targets()`` returns the recorded base poses and the dense path (a pure read);
    the program-check tracker continues from the path's end for the retreat."""
    import json
    import os

    from ispatialgym.oracle import programs as PR
    from ispatialgym.oracle.sidecar import Sidecar
    from ispatialgym.sampling.v2 import program_check as PC
    from ispatialgym.sampling.v2.stances import SidecarDraft

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(here, "tests", "data", "sidecars", "configuration_matching-articulation_open_large_scale.json")))
    art = d["articulation"]
    q_init, q_goal, poses = _hybrid_poses()
    art["q_init"], art["q_goal"] = q_init, q_goal
    marks = {"0.25": 0.3, "0.5": 0.6, "0.75": 0.9, "1.0": 1.2}
    art["follow_mode"] = "hybrid"
    art["base_follow"] = AH.base_follow_dict(poses, marks)
    art["base_follow_path"] = AH.HybridPath("right", (0.5, -0.2), poses, art["base_follow"], step={"revolute_rad": 0.17}).to_sidecar()
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [], sc.problems()
    actions = PR.expand(sc.alias, sc)
    follow = next(a for a in actions if a.name == "base_follow")
    t = follow.targets(sc, None)
    assert t["mode"] == "hybrid" and set(t) == set(follow.TARGET_KEYS)
    # passive_from 0.75: the oracle drives through 0.25 / 0.5 / 0.75 with the recorded (not the rigid-sweep) base poses
    assert t["fractions"] == [0.25, 0.5, 0.75] and t["base_poses"] == [art["base_follow"][k] for k in ("0.25", "0.5", "0.75")]
    assert len(t["follow_path"]) == 5 and t["follow_path"][0]["k"] == 0.0 and len(t["follow_path"][3]["arm_q"]) == 7
    fine, numbers = follow.annotate(sc)
    assert "driving the base and moving the arm together" in fine
    # a hybrid record without the dense path, or with a path that does not start at the grasp, is a problem
    d2 = json.loads(json.dumps(d))
    del d2["articulation"]["base_follow_path"]
    assert any("base_follow_path" in p for p in Sidecar.from_dict(d2).problems())
    d3 = json.loads(json.dumps(d))
    d3["articulation"]["base_follow_path"]["poses"][0]["k"] = 0.1
    assert any("k = 0" in p for p in Sidecar.from_dict(d3).problems())
    # the tracker: after a hybrid base_follow the retreat is checked from the follow's end configuration
    draft = SidecarDraft.from_dict(d) if hasattr(SidecarDraft, "from_dict") else None
    tracker = PC._Tracker(draft) if draft is not None else None
    if tracker is not None:
        w = {"follow_end": {"base_pose2d": list(poses[-1].base_pose2d), "q": [0.0] * 28, "translate": 0.5, "tilt": -0.2, "arm": "right"}}
        tracker.after(follow, w, None)
        assert tracker.stance_obj is not None and tuple(tracker.stance_obj.base_pose2d) == poses[-1].base_pose2d and tracker.stance_obj.working_arms == ["right"]
        tracker.after(actions[0], {}, None)  # the transition / navigate that follows clears it
    st = AH.end_stance_from_witness({"base_pose2d": [1.0, 2.0, 0.3], "q": [0.0] * 28, "translate": 0.4, "tilt": 0.1, "arm": "left"})
    assert st.base_pose2d == (1.0, 2.0, 0.3) and st.translate == 0.4 and st.working_arms == ["left"] and st.q is not None
    assert AH.end_stance_from_witness({}) is None


def test_hybrid_leaf_footprint_moves_with_the_joint():
    """The oriented leaf footprint: the closed leaf's AABB corners rotated about the hinge by q − q_ref (the same
    convention as ``handle_waypoints``), or translated along a prismatic axis; a slab at 90° stands perpendicular to the
    closed slab instead of filling the world AABB of the rotated leaf."""

    class _Link:
        def __init__(self, lo, hi):
            self.aabb = (th.tensor(lo), th.tensor(hi))

    # a 0.8 m wide, 4 cm thick leaf hinged at its left edge (x = 0), closed along +x, world frame = object frame
    leaf = {"leaf": _Link([0.0, -0.02, 0.0], [0.8, 0.02, 2.0])}
    jinfo = {"type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, 1.0], "hinge_point_obj": [0.0, 0.0, 0.0]}
    fp = AH.LeafFootprint(None, leaf, jinfo, (th.zeros(3), C.IDENTITY_QUAT.clone()), q_ref=0.0)
    assert fp.analytic
    closed = fp.at(0.0)
    assert closed.shape == (8, 2) and closed[:, 0].max() == pytest.approx(0.8) and abs(closed[:, 1]).max() == pytest.approx(0.02)
    open90 = fp.at(math.pi / 2)
    assert open90[:, 1].max() == pytest.approx(0.8, abs=1e-5) and abs(open90[:, 0]).max() == pytest.approx(0.02, abs=1e-5)
    # a base 0.5 m in front of the closed leaf's middle is clear when closed, hit by the leaf at 90° when it stands in the swing
    assert AH.leaf_clear_of_base(closed, (0.4, -0.5), 0.33)
    assert not AH.leaf_clear_of_base(open90, (0.1, 0.5), 0.33)
    # the same object frame rotated in the world: the footprint follows the object pose
    fp2 = AH.LeafFootprint(None, leaf, jinfo, (th.tensor([2.0, 1.0, 0.0]), C.quat_from_yaw(math.pi / 2)), q_ref=0.0)
    # (the link AABBs are world boxes already; only the hinge / axis are mapped) -> the hinge is at (2, 1), the axis still vertical
    assert th.allclose(fp2.hinge_w, th.tensor([2.0, 1.0, 0.0]), atol=1e-6) and abs(float(fp2.axis_w[2])) == pytest.approx(1.0)
    # prismatic: translated along the axis
    fpp = AH.LeafFootprint(None, leaf, {"type": "PrismaticJoint", "axis_dir_obj": [0.0, -1.0, 0.0]}, (th.zeros(3), C.IDENTITY_QUAT.clone()), q_ref=0.0)
    pulled = fpp.at(0.3)
    assert pulled[:, 1].min() == pytest.approx(-0.32) and pulled[:, 1].max() == pytest.approx(-0.28)
    # no hinge on a revolute record -> not analytic, the fallback (the sampler's AABB at q) is returned
    fpn = AH.LeafFootprint(None, leaf, {"type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, 1.0]}, (th.zeros(3), C.IDENTITY_QUAT.clone()), q_ref=0.0)
    assert not fpn.analytic and fpn.at(0.5, fallback=th.tensor([[1.0, 2.0, 0.0]])).tolist() == [[1.0, 2.0]]
