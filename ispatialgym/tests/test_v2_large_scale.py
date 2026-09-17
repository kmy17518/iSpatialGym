"""Simulator-free tests of the v2 large-scale sampler (spec §5.2 / §5.3 / §5.5 / §5.6): the face-frame math, the
Prompt-0 approach ray and scans, the reach-volume lookup frame, the handle trajectory h(q) against the conventions,
the base-follow composition, the arm / follow-mode rules, the brace-kind ordering and the init-surface classes
(``pytest ispatialgym/tests/test_v2_large_scale.py``).

The task modules import the sampler core (OmniGibson is importable without launching Isaac Sim); nothing here needs a
live world."""

from __future__ import annotations

import math

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


# ----------------------------------------------------------------------------------------------------------------
# alignment (§5.5): frame, hang pose, contacts, brace ordering
# ----------------------------------------------------------------------------------------------------------------


def test_frame_axes_and_hang_quat():
    f, s, u = AW.frame_axes("+x", "+z")
    assert th.allclose(s, th.tensor([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError):
        AW.frame_axes("+x", "+x")
    with pytest.raises(ValueError):
        AW.frame_axes("+x", "+z", level="z")
    AW.frame_axes("+x", "+z", level="y")
    q = AW.hang_quat([0.0, -1.0, 0.0], f, s, u)
    # the front axis maps onto the wall normal, the up axis onto world z
    assert th.allclose(C.quat_rotate(q, f), th.tensor([0.0, -1.0, 0.0]), atol=1e-5)
    assert th.allclose(C.quat_rotate(q, u), th.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_swept_extent_and_brace_order():
    assert AW.swept_extent(1.0, 0.1, 0.0) == 1.0
    assert AW.swept_extent(1.0, 0.1, math.radians(30)) == pytest.approx(math.hypot(1.0, 0.1))  # atan(0.1) = 5.7° <= 30°: the diagonal
    assert AW.swept_extent(1.0, 1.0, math.radians(30)) == pytest.approx(math.cos(math.radians(30)) + math.sin(math.radians(30)))  # atan(1) = 45° > 30°
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
    assert bar_offsets({"kind": "bar", "length": 0.2, "approach_dir": [1, 0, 0], "pinch_axis": [0, 0, 1]}) == [0.0]
