"""Simulator-free checks of the shared support-contact policy (``ispatialgym.grasping.contact_policy``) on the exported USD collision
hulls of the R1Pro hand (``assets/robot/r1pro_hand_collision.json``)."""

import math

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import hand_geometry as H


def _eef(z_axis, y_axis, origin):
    z = np.asarray(z_axis, float)
    y = np.asarray(y_axis, float)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return th.as_tensor(T, dtype=th.float32)


TIP = 0.0188  # eef origin to the fingertips along +z (conventions.FINGERTIP_LENGTH, the hulls' max z)


@pytest.fixture(scope="module")
def checker():
    return CP.HandChecker("right", CP.DEFAULT, spacing=0.003)


def test_geometry_is_the_usd_export():
    g = H.load("right")
    assert g.source == "usd_collision"
    tips = max(float(l.points[:, 2].max()) for l in g.by_role("finger"))
    assert abs(tips - TIP) < 0.0005


def test_finger_membership_is_explicit():
    r = H.role_links("right")
    assert r["finger"] == ("right_gripper_finger_link1", "right_gripper_finger_link2")
    assert "right_gripper_link" not in r["finger"] and "right_realsense_link" not in r["finger"]
    assert H.role_of("right_gripper_link", "right") == "palm"
    assert H.role_of("right_realsense_link", "right") == "wrist_camera"
    assert CP.DEFAULT.finger_links("left") == ("left_gripper_finger_link1", "left_gripper_finger_link2")


def test_top_down_finger_touch_within_2mm_passes_deeper_fails(checker):
    plane = CP.Support.plane(0.0)
    down = ([0, 0, -1.0], [0, 1.0, 0])
    ok, reason, cl = checker.check(_eef(*down, [0, 0, TIP - 0.0015]), plane, (0.03, 0.03))
    assert bool(ok[0]) and reason[0] == "ok", (reason, cl["finger"])
    assert -0.002 <= float(cl["finger"][0]) < 0.0
    ok, reason, cl = checker.check(_eef(*down, [0, 0, TIP - 0.003]), plane, (0.03, 0.03))
    assert not bool(ok[0]) and reason[0] == "finger_penetration"


def test_side_grasp_camera_below_table_fails_with_fingers_clear(checker):
    plane = CP.Support.plane(0.0)
    # approach +x, closing +y: eef x = y cross z = -z world -> the camera (eef x ~ -0.056) is ABOVE; its twin puts it below. The palm
    # (eef x +-0.03) and the finger bases (+-0.024) need the eef 3 cm above the table
    T_up = _eef([1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.032])
    T_twin = _eef([1.0, 0, 0], [0, -1.0, 0], [0, 0, 0.032])
    ok, reason, cl = checker.check(th.stack([T_up, T_twin]), plane, (0.03, 0.03))
    assert float(cl["finger"][0]) > -0.002 and float(cl["finger"][1]) > -0.002  # fingers clear in both
    assert bool(ok[0]), reason[0]
    assert not bool(ok[1]) and reason[1] in ("wrist_camera_contact", "palm_contact")
    assert float(cl["wrist_camera"][1]) < 0.0


def test_palm_contact_fails_even_if_fingers_pass(checker):
    plane = CP.Support.plane(0.0)
    # tilted approach: the palm swings below the fingertips' height
    a = np.array([math.cos(math.radians(80)), 0, -math.sin(math.radians(80))])
    T = _eef(a, [0, 1.0, 0], [0, 0, 0.004])
    ok, reason, cl = checker.check(T, plane, (0.03, 0.03))
    if float(cl["palm"][0]) < 0 or float(cl["wrist_camera"][0]) < 0:
        assert not bool(ok[0]) and reason[0] in ("palm_contact", "wrist_camera_contact")


def test_approach_through_the_slab_fails_though_the_grasp_pose_passes(checker):
    # a top-down grasp at the slab's far edge whose pre-grasp lies above, then the same pose approached horizontally through the edge
    slab = CP.Support.box([0.5, 0.0, -0.025], [0.3, 0.5, 0.025])
    side = _eef([1.0, 0, 0], [0, 0, 1.0], [0.2, 0.0, -0.01])  # below the top, outside the slab (x < 0.2)
    ok, reason, _ = checker.check(side, slab, (0.03, 0.03))
    res = CP.check_grasp_geometry(checker, side.unsqueeze(0) if side.dim() == 2 else side, slab, [0.02], open_q=0.03)
    # the grasp pose itself sits outside the box ...
    side_in = _eef([1.0, 0, 0], [0, 0, 1.0], [0.25, 0.0, -0.01])
    res2 = CP.check_grasp_geometry(checker, side_in.unsqueeze(0), slab, [0.02], open_q=0.03)
    assert not bool(res2["ok"][0])  # the fingertips are inside the slab's edge at the grasp pose
    T = _eef([0, 0, -1.0], [0, 1.0, 0], [0.5, 0.0, TIP - 0.001])
    res3 = CP.check_grasp_geometry(checker, T.unsqueeze(0), slab, [0.02], open_q=0.03)
    assert bool(res3["ok"][0]), res3


def test_approach_waypoint_catches_a_crossing(checker):
    # the endpoint is above the plane but the straight approach from the pre-grasp (10 cm back) dips through a raised block
    block = CP.Support.box([0.0, 0.0, 0.10], [0.05, 0.2, 0.02])  # a bar the hand passes under
    T = _eef([0, 0, -1.0], [0, 1.0, 0], [0.0, 0.0, 0.05])
    res = CP.check_grasp_geometry(checker, T.unsqueeze(0), block, [0.02], open_q=0.03)
    assert not bool(res["ok"][0]) and (res["stage"][0].startswith("approach") or res["stage"][0] == "pre_grasp")


def test_sphere_rule_excludes_only_the_fingers():
    plane = CP.Support.plane(0.0)
    links = ["right_gripper_finger_link1", "right_gripper_link", "right_realsense_link", "wheel_motor_link1"]
    S = th.tensor([[[0, 0, -0.001, 0.01], [0, 0, 0.5, 0.04], [0, 0, 0.5, 0.018], [0, 0, 0.05, 0.06]]], dtype=th.float32)
    ok, worst = CP.sphere_support_ok(S, links, plane, "right")
    assert not bool(ok[0]) and worst[0] == "wheel_motor_link1"  # the finger sphere below the plane is not judged by spheres
    ok, worst = CP.sphere_support_ok(S, links, plane, "right", ground_contact_links=["wheel_motor_link1"], floor_support=True)
    assert bool(ok[0])  # floor mode: the wheels stand on it
    S2 = S.clone()
    S2[0, 2, 2] = 0.01  # camera sphere into the support
    ok, worst = CP.sphere_support_ok(S2, links, plane, "right", ground_contact_links=["wheel_motor_link1"], floor_support=True)
    assert not bool(ok[0]) and worst[0] == "right_realsense_link"


def test_policy_key_tracks_the_tolerance():
    a = CP.ContactPolicy()
    b = CP.ContactPolicy(finger_support_max_penetration_m=0.008)
    assert a.key() != b.key() and a.key().startswith("isg-support-contact-v1-")
    assert a.meta()["finger_support_max_penetration_m"] == 0.002


def test_finger_spheres_sit_inside_the_finger_hulls():
    """The padding note of the module docstring, measured: the cuRobo finger spheres (+2 mm buffer) do not reach the hulls' outer
    faces, so a sphere-based 2 mm rule is not a 2 mm rule on the finger."""
    g = H.load("right")
    Cs, Rs, L = H.hand_spheres_eef("right", (0.05, 0.05), buffer=0.002)
    fing = [i for i, l in enumerate(L) if l == "right_gripper_finger_link1"]
    P = [l for l in g.by_role("finger") if l.link.endswith("1")][0].at(0.05)
    tip_hull = float(P[:, 2].max())
    tip_sph = max(float(Cs[i, 2] + Rs[i]) for i in fing)
    outer_hull = float(P[:, 1].max())
    outer_sph = max(float(Cs[i, 1] + Rs[i]) for i in fing)
    print(f"fingertip: hull {tip_hull:.4f}, spheres {tip_sph:.4f}; outer face: hull {outer_hull:.4f}, spheres {outer_sph:.4f}")
    assert abs(tip_hull - tip_sph) < 0.004
