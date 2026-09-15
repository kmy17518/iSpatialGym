"""Tests of ``ispatialgym.robot.conventions`` (TODO F5) - pure geometry, no simulator, no OmniGibson import.

The FK numbers come from oracle doc §5.2 (shoulder table), spec §4.3 (camera pitch at tilt 0), the R1Pro USD
(``zed_link`` pose at the zero configuration, camera prim) and a sampled camera-relocalization instance whose goal
camera pose was measured in simulation from a recorded (base pose, translate, tilt).
"""

import math
import os
import sys

import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C

# ------------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------------


def _rand_quat(gen):
    q = th.randn(4, generator=gen)
    return q / q.norm()


def _same_quat(a, b, tol=1e-5):
    a, b = C.as_tensor(a), C.as_tensor(b)
    return min(float((a - b).abs().max()), float((a + b).abs().max())) < tol


def _close(a, b, tol=1e-5):
    return float((C.as_tensor(a) - C.as_tensor(b)).abs().max()) < tol


GEN = th.Generator().manual_seed(1234)


def test_module_is_pure():
    # in a fresh interpreter (other test modules of the suite legitimately import OmniGibson / cuRobo)
    import subprocess

    code = ("import sys, ispatialgym.robot.conventions; "
            "assert 'omnigibson' not in sys.modules, 'conventions imports OmniGibson'; "
            "assert not any(m.startswith('curobo') for m in sys.modules), 'conventions imports cuRobo'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_public_names_of_the_table_exist():
    for name in ("grasp_pose", "pre_pose", "handle_eef", "contact_eef", "line_waypoints", "slerp_waypoints", "orbit_waypoints",
                 "arc_waypoints", "handle_waypoints", "transition_poses", "base_follow_poses", "torso_q", "translate_to_torso_q",
                 "torso_q_to_translate_tilt", "camera_offset_in_zed", "camera_pose", "q_with_base_pose", "cube_grasp",
                 "flip_waypoints", "FINGER_OPEN", "CONTACT_CENTER_Z", "FINGERTIP_LENGTH", "GRIPPER_OPEN_M", "FramedPose"):
        assert hasattr(C, name), name
        assert name in C.__all__, name
    assert C.GRIPPER_OPEN_M == 0.10 and C.FINGER_OPEN == 0.05 and C.CONTACT_CENTER_Z == pytest.approx(-0.01186)
    assert C.PAD_Z_RANGE[0] < C.CONTACT_CENTER_Z < C.PAD_Z_RANGE[1] < C.FINGERTIP_LENGTH


# ------------------------------------------------------------------------------------------------------------------
# quaternion / pose primitives
# ------------------------------------------------------------------------------------------------------------------


def test_quaternion_primitives_round_trip():
    for _ in range(50):
        q = _rand_quat(GEN)
        R = C.quat_to_mat(q)
        assert _close(R @ R.T, th.eye(3)) and abs(float(th.det(R)) - 1.0) < 1e-5
        assert _same_quat(C.mat_to_quat(R), q)
        roll, pitch, yaw = C.euler_of(q)
        assert _same_quat(C.quat_from_euler(roll, pitch, yaw), q, 1e-4)
        assert _same_quat(C.qmul(q, C.qinv(q)), C.IDENTITY_QUAT)
        v = th.randn(3, generator=GEN)
        assert _close(C.quat_rotate(q, v), R @ v)
        assert _close(C.quat_rotate(q, v.unsqueeze(0)), (R @ v).unsqueeze(0))
    # composition order: R(qmul(a, b)) = R(a) R(b)
    a, b = _rand_quat(GEN), _rand_quat(GEN)
    assert _close(C.quat_to_mat(C.qmul(a, b)), C.quat_to_mat(a) @ C.quat_to_mat(b))
    # yaw / axis-angle conventions
    q = C.quat_from_yaw(0.7)
    assert C.yaw_of(q) == pytest.approx(0.7, abs=1e-6)
    assert _close(C.quat_rotate(q, th.tensor([1.0, 0.0, 0.0])), th.tensor([math.cos(0.7), math.sin(0.7), 0.0]))
    assert _same_quat(C.quat_from_axis_angle([0.0, 0.0, 2.0], 0.7), q)
    # right-hand rule about z: x -> y
    assert _close(C.quat_rotate(C.quat_from_axis_angle([0, 0, 1], math.pi / 2), th.tensor([1.0, 0.0, 0.0])), th.tensor([0.0, 1.0, 0.0]))
    assert C.wrap_angle(3 * math.pi) == pytest.approx(math.pi) and C.wrap_angle(-math.pi) == pytest.approx(math.pi)
    assert C.wrap_angle(0.3) == pytest.approx(0.3) and C.wrap_angle(2 * math.pi + 0.3) == pytest.approx(0.3)


def test_compose_invert_relative_and_frames():
    for _ in range(20):
        a = (th.randn(3, generator=GEN), _rand_quat(GEN))
        b = (th.randn(3, generator=GEN), _rand_quat(GEN))
        ab = C.compose(a, b)
        # matrix form agrees
        mat = C.pose_to_mat(a) @ C.pose_to_mat(b)
        p, q = C.mat_to_pose(mat)
        assert _close(ab[0], p, 1e-4) and _same_quat(ab[1], q, 1e-4)
        # relative undoes compose: relative(compose(a, b), a) == b
        rel = C.relative(ab, a)
        assert _close(rel[0], b[0], 1e-4) and _same_quat(rel[1], b[1], 1e-4)
        # invert
        ident = C.compose(a, C.invert(a))
        assert _close(ident[0], th.zeros(3), 1e-4) and _same_quat(ident[1], C.IDENTITY_QUAT, 1e-4)
        # transform_points == compose on a point
        pt = th.randn(3, generator=GEN)
        assert _close(C.transform_points(a, pt)[0], C.compose(a, (pt, C.IDENTITY_QUAT))[0], 1e-4)
    fp = C.FramedPose.create([1.0, 2.0, 3.0], frame="obj:mug")
    assert fp.frame == "obj:mug" and _same_quat(fp.quat, C.IDENTITY_QUAT)
    moved = C.pre_pose(fp, 0.1)
    assert isinstance(moved, C.FramedPose) and moved.frame == "obj:mug"
    x, y, yaw = C.pose_to_pose2d(C.pose2d_to_pose(1.0, -2.0, 2.5, z=0.3))
    assert (x, y) == (1.0, -2.0) and yaw == pytest.approx(2.5, abs=1e-6)


def test_slerp_endpoints_and_geodesic():
    for _ in range(20):
        q0, q1 = _rand_quat(GEN), _rand_quat(GEN)
        assert _same_quat(C.slerp(q0, q1, 0.0), q0, 1e-5)
        assert _same_quat(C.slerp(q0, q1, 1.0), q1, 1e-5)
        total = C.quat_angle(q0, q1)
        half = C.slerp(q0, q1, 0.5)
        assert C.quat_angle(q0, half) == pytest.approx(total / 2, abs=1e-4)
        assert C.quat_angle(half, q1) == pytest.approx(total / 2, abs=1e-4)
    wps = C.slerp_waypoints(q0, q1, 4)
    assert wps.shape == (4, 4) and _same_quat(wps[-1], q1)
    full = C.slerp_waypoints(q0, q1, 4, endpoints=True)
    assert full.shape == (5, 4) and _same_quat(full[0], q0)
    # nearly identical quaternions take the linear branch and stay normalised
    q2 = C.qmul(C.quat_from_axis_angle([1, 0, 0], 1e-4), q0)
    assert abs(float(C.slerp(q0, q2, 0.5).norm()) - 1.0) < 1e-6


def test_quat_with_z_axis_and_top_down():
    ref = C.quat_from_yaw(0.4)
    q = C.top_down_quat(ref)
    R = C.quat_to_mat(q)
    assert _close(R[:, 2], th.tensor([0.0, 0.0, -1.0]))
    assert abs(math.atan2(float(R[1, 0]), float(R[0, 0])) - 0.4) < 1e-5  # x axis keeps the reference yaw
    z = C.unit([1.0, 1.0, -1.0])
    q2 = C.quat_with_z_axis(z, ref)
    assert _close(C.approach_axis(q2), z)


# ------------------------------------------------------------------------------------------------------------------
# eef frame, pre-poses, contacts
# ------------------------------------------------------------------------------------------------------------------


def test_eef_quat_axes():
    a, c = [0.0, 0.0, -1.0], [0.3, 1.0, 0.2]  # closing not exactly perpendicular: projected
    q = C.eef_quat(a, c)
    assert _close(C.approach_axis(q), th.tensor(a))
    cw = C.closing_axis(q)
    assert abs(float(th.dot(cw, C.approach_axis(q)))) < 1e-6
    assert _close(cw, C.unit([0.3, 1.0, 0.0]))
    R = C.quat_to_mat(q)
    assert _close(R[:, 0], th.cross(R[:, 1], R[:, 2], dim=0))  # x = c × a
    with pytest.raises(ValueError):
        C.eef_quat(a, [0.0, 0.0, 1.0])


def test_pre_pose_moves_back_along_minus_a():
    for _ in range(10):
        pose = (th.randn(3, generator=GEN), _rand_quat(GEN))
        pre = C.pre_pose(pose)
        d = pose[0] - pre[0]
        assert float(d.norm()) == pytest.approx(0.10, abs=1e-6)
        assert _close(d / d.norm(), C.approach_axis(pose[1]))
        assert _same_quat(pre[1], pose[1])
        pre2 = C.pre_pose(pose, dist=0.25)
        assert float((pose[0] - pre2[0]).norm()) == pytest.approx(0.25, abs=1e-6)


def test_contact_and_handle_eef_offsets():
    point = th.tensor([1.0, 0.5, 0.9])
    a = C.unit([1.0, 0.2, -0.1])
    c = th.tensor([0.0, 0.0, 1.0])
    eef = C.contact_eef(point, a, c)
    # pad centre exactly on the point; origin 1.19 cm beyond it along a
    assert _close(C.pad_center(eef), point, 1e-6)
    assert _close(eef[0], point - C.CONTACT_CENTER_Z * a, 1e-6)
    assert _close(C.approach_axis(eef[1]), a) and _close(C.closing_axis(eef[1]), C.unit(c - th.dot(c, a) * a))
    assert _close(C.fingertip(eef), point + (C.FINGERTIP_LENGTH - C.CONTACT_CENTER_Z) * a, 1e-6)
    # default closing axis: horizontal and perpendicular to a
    eef_d = C.contact_eef(point, a)
    cd = C.closing_axis(eef_d[1])
    assert abs(float(cd[2])) < 1e-6 and abs(float(th.dot(cd, a))) < 1e-6
    # handle grasp: eef origin 12 mm before the contact along a; the fingertip-side pad end lands on the contact
    h = C.handle_eef(point, a, c)
    assert _close(h[0], point - C.HANDLE_EEF_OFFSET * a, 1e-6)
    pad_far_end = h[0] + C.PAD_Z_RANGE[1] * C.approach_axis(h[1])
    assert float((pad_far_end - point).norm()) < 1e-3
    assert _close(C.pre_pose(h)[0], h[0] - 0.10 * a, 1e-6)
    # a recorded sidecar waypoint (dict) or a (contact, approach, pinch) triple is accepted as the single argument
    h2 = C.handle_eef({"contact": point.tolist(), "approach": a.tolist(), "pinch": c.tolist()})
    h3 = C.handle_eef((point, a, c))
    assert _close(h2[0], h[0], 1e-6) and _same_quat(h2[1], h[1]) and _close(h3[0], h[0], 1e-6)
    with pytest.raises(TypeError):
        C.handle_eef(point, a)  # the closing axis is never defaulted for a handle
    assert _close(C.default_closing_axis(a), C.closing_axis(C.contact_eef(point, a)[1]), 1e-6)
    # fingertip stop: fingertip plane `clearance` before the point
    f = C.fingertip_eef(point, a, c, clearance=0.005)
    assert _close(C.fingertip(f), point - 0.005 * a, 1e-6)
    assert C.pad_clearance(0.03) == pytest.approx(0.035)


# ------------------------------------------------------------------------------------------------------------------
# waypoints
# ------------------------------------------------------------------------------------------------------------------


def test_line_waypoints_rule():
    p0, p1 = th.tensor([0.0, 0.0, 0.0]), th.tensor([0.4, 0.0, 0.0])
    w = C.line_waypoints(p0, p1, 4)
    assert w.shape == (4, 3) and _close(w[:, 0], th.tensor([0.1, 0.2, 0.3, 0.4]))
    full = C.line_waypoints(p0, p1, 4, endpoints=True)
    assert full.shape == (5, 3) and _close(full[0], p0) and _close(full[-1], p1)
    # poses in -> poses out, orientation SLERPed
    q0, q1 = C.IDENTITY_QUAT, C.quat_from_yaw(1.0)
    wp = C.line_waypoints((p0, q0), (p1, q1), 2)
    assert len(wp) == 2 and _close(wp[0][0], th.tensor([0.2, 0.0, 0.0])) and C.yaw_of(wp[0][1]) == pytest.approx(0.5, abs=1e-5)
    assert _close(wp[-1][0], p1) and _same_quat(wp[-1][1], q1)
    fp = C.FramedPose.create(p0, q0, frame="base")
    wf = C.line_waypoints(fp, C.FramedPose.create(p1, q1, frame="base"), 1)
    assert isinstance(wf[0], C.FramedPose) and wf[0].frame == "base"
    with pytest.raises(ValueError):
        C.line_waypoints(p0, p1, 0)


def test_orbit_keeps_pivot_distance_and_reaches_delta():
    pose = (th.tensor([0.5, -0.2, 0.9]), _rand_quat(GEN))
    pivot = th.tensor([0.6, -0.1, 0.85])
    delta = C.quat_from_axis_angle([0.2, 0.3, 1.0], math.radians(75))
    r0 = float((pose[0] - pivot).norm())
    wps = C.orbit_waypoints(pose, pivot, delta, n=5)
    assert len(wps) == 5
    for k, (p, q) in enumerate(wps, start=1):
        assert float((p - pivot).norm()) == pytest.approx(r0, abs=1e-5)
        assert C.quat_angle(pose[1], q) == pytest.approx(math.radians(75) * k / 5, abs=1e-4)
        # the eef stays rigidly attached to the pivot: the pivot expressed in the eef frame is constant
        piv_in_eef = C.relative((pivot, C.IDENTITY_QUAT), (p, q))[0]
        assert _close(piv_in_eef, C.relative((pivot, C.IDENTITY_QUAT), pose)[0], 1e-4)
    assert _same_quat(wps[-1][1], C.qmul(delta, pose[1]), 1e-4)
    # eef-frame delta: R_0 R_delta
    wps_e = C.orbit_waypoints(pose, pivot, delta, n=1, frame="eef")
    assert _same_quat(wps_e[-1][1], C.qmul(pose[1], delta), 1e-4)
    with pytest.raises(ValueError):
        C.orbit_waypoints(pose, pivot, delta, n=1, frame="object")


def test_arc_waypoints_chord_and_tangent():
    centre = th.tensor([0.0, 0.0, 1.45])  # COM of a hung target (spec 5.5)
    n_wall = th.tensor([1.0, 0.0, 0.0])  # wall normal toward the robot = rotation axis
    delta = math.radians(30.0)
    e_low = centre + th.tensor([0.0, 0.25 * math.cos(-delta), 0.25 * math.sin(-delta)])  # low end, tilted by -30 deg
    r = e_low - centre
    tangent = th.cross(n_wall, r, dim=0)  # tangent of the circle about C through E_low
    tangent = tangent / tangent.norm() * (1.0 if float(tangent[2]) > 0 else -1.0)  # signed so the low end rises
    contact = C.contact_eef(e_low, tangent, n_wall)
    wps = C.arc_waypoints(contact, centre, n_wall, delta, n=2)
    assert len(wps) == 2
    radius = float(r.norm())
    for k, (p, q) in enumerate(wps, start=1):
        pad = C.pad_center((p, q))
        assert float((pad - centre).norm()) == pytest.approx(radius, abs=1e-5)  # the material point stays on the circle
        chord = float((pad - e_low).norm())
        assert chord == pytest.approx(2 * radius * math.sin(delta * k / 2 / 2), abs=1e-5)
        a = C.approach_axis(q)
        assert abs(float(th.dot(a, pad - centre))) < 1e-5  # a stays tangent
        assert _close(C.closing_axis(q), n_wall, 1e-5)  # c ∥ the axis stays put
    # the level pose: the low end is at the same height as the centre after +30 deg
    assert float(C.pad_center(wps[-1])[2]) == pytest.approx(float(centre[2]), abs=1e-5)
    # bare point form
    pts = C.arc_waypoints(e_low, centre, n_wall, delta, n=2)
    assert pts.shape == (2, 3) and _close(pts[-1], C.pad_center(wps[-1]), 1e-5)


def test_handle_waypoints_revolute_and_prismatic():
    handle = {"contact": [0.20, 0.05, 0.30], "pinch_axis": [0.0, 0.0, 1.0], "approach_dir": [-1.0, 0.0, 0.0]}
    joint = {"type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, 1.0], "hinge_point_obj": [0.20, 0.30, 0.0], "lower": 0.0, "closed_q": 0.0}
    qs = [0.0, 0.4, 0.8]
    wps = C.handle_waypoints(handle, joint, qs)
    assert [w.q for w in wps] == qs
    assert _close(wps[0].contact, th.tensor(handle["contact"]))
    hinge, axis = th.tensor(joint["hinge_point_obj"]), th.tensor(joint["axis_dir_obj"])
    for w in wps:
        rel = w.contact - hinge
        radial = rel - th.dot(rel, axis) * axis
        assert float(radial.norm()) == pytest.approx(0.25, abs=1e-5)  # distance to the hinge axis kept
        assert abs(float(th.dot(w.approach, w.pinch))) < 1e-6
        assert _close(w.eef[0], w.contact - C.HANDLE_EEF_OFFSET * w.approach, 1e-6)
        assert _close(C.approach_axis(w.eef[1]), w.approach) and _close(C.closing_axis(w.eef[1]), w.pinch)
        assert _close(w.pre_grasp[0], w.eef[0] - 0.10 * w.approach, 1e-6)
    # rotation by 0.8 rad about +z: the approach direction turns with the door
    assert _close(wps[-1].approach, C.quat_rotate(C.quat_from_axis_angle(axis, 0.8), th.tensor([-1.0, 0.0, 0.0])), 1e-5)
    sd = wps[1].to_sidecar()
    assert set(sd) == {"contact", "approach", "pinch"}
    # prismatic drawer: translation along the axis, directions unchanged
    joint_p = {"type": "PrismaticJoint", "axis_dir_obj": [-1.0, 0.0, 0.0], "lower": 0.0}
    wp = C.handle_waypoints(handle, joint_p, [0.0, 0.15])
    assert _close(wp[1].contact - wp[0].contact, th.tensor([-0.15, 0.0, 0.0]), 1e-6)
    assert _close(wp[1].approach, wp[0].approach)
    # world frame through obj_pose
    obj = (th.tensor([2.0, 1.0, 0.0]), C.quat_from_yaw(math.pi / 2))
    ww = C.handle_waypoints(handle, joint_p, [0.15], obj_pose=obj)[0]
    assert _close(ww.contact, C.compose(obj, (wp[1].contact, C.IDENTITY_QUAT))[0], 1e-5)
    assert _close(ww.approach, C.quat_rotate(obj[1], wp[1].approach), 1e-5)
    with pytest.raises(ValueError):
        C.handle_waypoints(handle, {"type": "FixedJoint", "axis_dir_obj": [1, 0, 0]}, [0.0])


# ------------------------------------------------------------------------------------------------------------------
# trunk keyframes
# ------------------------------------------------------------------------------------------------------------------


def test_torso_q_round_trip_and_keyframes():
    assert _close(C.torso_q(0.5, 0.0), th.tensor(M.JOYLO_RESET_JOINT_POS[6:10]))  # JoyLo reset trunk = translate 0.5
    assert _close(C.translate_to_torso_q(0.0), th.tensor(M.TORSO_UPRIGHT))
    assert _close(C.translate_to_torso_q(1.0), th.tensor(M.TORSO_DOWNWARD))
    assert _close(C.translate_to_torso_q(2.0), th.tensor(M.TORSO_GROUND))
    for tr in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0, 1.5):
        for tilt in (-0.9, -0.3, 0.0, 0.25):
            t2, k2 = C.torso_q_to_translate_tilt(C.torso_q(tr, tilt))
            assert t2 == pytest.approx(tr, abs=1e-5) and k2 == pytest.approx(tilt, abs=1e-5)
    # tilt clamping with the URDF limits
    q = C.torso_q(0.5, -5.0, C.TORSO_JOINT_LIMITS)
    assert float(q[M.TRUNK_TILT_JOINT_INDEX]) == pytest.approx(float(C.TORSO_JOINT_LIMITS[0][M.TRUNK_TILT_JOINT_INDEX]))
    assert _close(C.translate_to_torso_q(-1.0), th.tensor(M.TORSO_UPRIGHT))  # out-of-range translate is clamped
    # the oracle's TrunkState still works through the re-export
    from ispatialgym.oracle.trunk import TrunkState

    ts = TrunkState.from_q(C.torso_q(0.75, -0.4))
    assert ts.translate == pytest.approx(0.75, abs=1e-5) and ts.tilt == pytest.approx(-0.4, abs=1e-5)


def test_trunk_sweep_and_transition_standoffs():
    sw = C.trunk_sweep((0.5, 0.0), (0.75, -0.5), n=5)
    assert len(sw) == 5 and sw[-1] == pytest.approx((0.75, -0.5)) and sw[0] == pytest.approx((0.55, -0.1))
    assert C.trunk_sweep((0.5, 0.0), (0.75, -0.5), n=5, endpoints=True)[0] == (0.5, 0.0)
    assert C.transition_standoffs(0.35) == pytest.approx([0.35, 0.45, 0.55, 0.65, 0.75, 0.85])


# ------------------------------------------------------------------------------------------------------------------
# configurations, transitions, base follow
# ------------------------------------------------------------------------------------------------------------------


def test_q_with_base_pose_layout():
    q = th.arange(28, dtype=th.float32) * 0.01
    out = C.q_with_base_pose(q, 1.5, -2.0, 0.3)
    assert C.JOINT_INDEX["base_footprint_x_joint"] == 0 and C.JOINT_INDEX["base_footprint_rz_joint"] == 5
    assert float(out[0]) == pytest.approx(1.5) and float(out[1]) == pytest.approx(-2.0) and float(out[5]) == pytest.approx(0.3)
    assert _close(out[6:], q[6:]) and _close(out[2:5], q[2:5])  # nothing else touched
    assert float(q[0]) == 0.0  # input not modified
    # root frame offset: the joints hold the offset in the root frame (Skills.q_with_base_pose formula)
    root = (10.0, 5.0, math.pi / 2)
    out = C.q_with_base_pose(q, 10.0, 6.0, math.pi / 2 + 0.2, root_pose2d=root)
    assert float(out[0]) == pytest.approx(1.0, abs=1e-6) and float(out[1]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[5]) == pytest.approx(0.2, abs=1e-6)
    x, y, yaw = C.base_pose2d_of_q(out, root_pose2d=root)
    assert (x, y) == pytest.approx((10.0, 6.0), abs=1e-5) and yaw == pytest.approx(math.pi / 2 + 0.2, abs=1e-6)
    # RobotKit.q_at_base_pose behaviour: upright at the nominal height, z relative to the root
    out = C.q_with_base_pose(q, 1.0, 1.0, 0.0, base_z=0.0053, root_z=1.2)
    assert float(out[2]) == pytest.approx(0.0053 - 1.2) and float(out[3]) == 0.0 and float(out[4]) == 0.0
    # a custom joint index map is honoured
    jidx = {n: 27 - i for i, n in enumerate(M.ALL_JOINTS)}
    out = C.q_with_base_pose(q, 1.0, 2.0, 0.5, joint_index=jidx)
    assert float(out[27]) == 1.0 and float(out[26]) == 2.0 and float(out[22]) == pytest.approx(0.5)


def test_transition_poses():
    st_from = {"base_pose2d": [1.0, 2.0, 0.5], "translate": 0.5, "tilt": 0.0, "stand_off": 0.45}
    st_to = {"base_pose2d": [1.1, 2.2, 0.5], "translate": 0.75, "tilt": -0.42, "stand_off": 0.35}
    tp = C.transition_poses(st_from, st_to, 0.55)
    assert tp.start == (1.0, 2.0, 0.5) and tp.drive_in == (1.1, 2.2, 0.5)
    back_pose, trunk, drive_in = tp  # unpacks as the three legs
    assert back_pose == tp.drive_back and trunk == tp.trunk and drive_in == tp.drive_in
    assert tp.trunk == (0.75, -0.42) and tp.trunk_from == (0.5, 0.0) and tp.stand_off == 0.55
    assert tp.order == ("drive-back", "trunk", "drive-in")
    back = th.tensor(tp.drive_in[:2]) - th.tensor(tp.drive_back[:2])
    assert float(back.norm()) == pytest.approx(0.20, abs=1e-6)  # s_T - s along the arriving stance's forward axis
    assert _close(back / back.norm(), th.tensor([math.cos(0.5), math.sin(0.5)]), 1e-6)
    assert tp.drive_back[2] == 0.5
    assert C.transition_poses(st_from, st_to, 0.35).drive_back == pytest.approx(tp.drive_in)  # s_T = s: no drive-back leg
    with pytest.raises(ValueError):
        C.transition_poses(st_from, st_to, 0.30)

    class St:  # duck-typed sampler stance
        base_pose2d = (0.0, 0.0, math.pi)
        translate, tilt, stand_off = 0.25, -0.3, 0.45

    tp2 = C.transition_poses(St(), St(), 0.65)
    assert tp2.drive_back[0] == pytest.approx(0.20, abs=1e-6) and tp2.drive_back[1] == pytest.approx(0.0, abs=1e-6)


def test_base_follow_poses_planar_for_vertical_axis():
    # a door handle swinging about a vertical hinge: h(q_k) are the eef poses, the base follows in the plane
    hinge = th.tensor([2.0, 0.0, 0.0])
    axis = th.tensor([0.0, 0.0, 1.0])
    h0 = C.handle_eef(th.tensor([2.0, 0.8, 1.0]), th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 1.0]))
    base0 = C.pose2d_to_pose(1.2, 0.6, 0.1, z=C.NOMINAL_BASE_Z)
    T_eef_base = C.eef_base_transform(h0, base0)
    hs = [h0] + C.orbit_waypoints(h0, hinge, C.quat_from_axis_angle(axis, 1.0), n=4)
    b = C.base_follow_poses(hs, T_eef_base)
    assert len(b) == 5 and b[0] == pytest.approx((1.2, 0.6, 0.1), abs=1e-5)
    assert C.base_follow_residual(hs, T_eef_base) < 1e-5
    for hk, (x, y, yaw) in zip(hs, b):
        # driving the base to b_k with the arm locked puts the eef back on h(q_k)
        eef = C.compose(C.pose2d_to_pose(x, y, yaw, z=C.NOMINAL_BASE_Z), C.invert(T_eef_base))
        assert _close(eef[0], hk[0], 1e-4) and _same_quat(eef[1], hk[1], 1e-4)
    assert b[-1][2] == pytest.approx(0.1 + 1.0, abs=1e-5)  # the base yaw turns with the door
    # a horizontal-axis flap cannot be followed by the planar base: residual is large
    hs_h = [h0] + C.orbit_waypoints(h0, hinge, C.quat_from_axis_angle([0.0, 1.0, 0.0], 0.8), n=2)
    assert C.base_follow_residual(hs_h, T_eef_base) > 0.3


# ------------------------------------------------------------------------------------------------------------------
# FK numbers: torso chain, head camera
# ------------------------------------------------------------------------------------------------------------------


def test_zed_link_matches_usd_at_zero_configuration():
    fk = C.torso_fk([0.0, 0.0, 0.0, 0.0])
    zp, zq = fk["zed_link"]
    usd_p = th.tensor([-0.013520993292331696, 0.06000335142016411, 1.6182500198483467])
    usd_q = C.quat_wxyz_to_xyzw(th.tensor([0.40556812286376953, -0.5792345404624939, 0.5792366862297058, -0.40556958317756653]))
    assert _close(zp, usd_p, 1e-6) and C.quat_angle(zq, usd_q) < 1e-5
    assert _close(fk["torso_link4"][0], th.tensor([-0.079032, 0.0001005 - 0.00010015, 0.34265 + 0.4 + 0.3 + 0.09962]), 1e-6)
    # camera prim in zed_link: (0.06, 0, 0.01), 180 deg about x
    cp, cq = C.camera_offset_in_zed()
    assert _close(cp, th.tensor([0.06, 0.0, 0.01])) and C.quat_angle(cq, th.tensor([1.0, 0.0, 0.0, 0.0])) < 1e-6
    cam = fk["camera"]
    assert _close(cam[0], C.compose((zp, zq), (cp, cq))[0], 1e-6)


def test_shoulder_table_of_oracle_doc_5_2():
    table = {0.0: (0.13, 1.41), 0.5: (0.16, 1.23), 0.75: (0.14, 1.09), 1.0: (0.10, 0.92)}
    for translate, (x, z) in table.items():
        for arm in ("right", "left"):
            s = C.shoulder_position(translate, 0.0, arm)
            assert float(s[0]) == pytest.approx(x, abs=0.0075), (translate, arm)
            assert float(s[2]) + C.NOMINAL_BASE_Z == pytest.approx(z, abs=0.0075), (translate, arm)
    assert float(C.shoulder_position(0.5, 0.0, "left")[1]) == pytest.approx(0.097234 + 0.0735, abs=1e-3)
    assert float(C.shoulder_position(0.5, 0.0, "right")[1]) == pytest.approx(-(0.097236 + 0.0735), abs=1e-3)


def test_camera_pose_reproduces_a_sampled_camera_relocalization_goal():
    # camera_relocalization-0__house_single_floor__0000 (bench2-l40s): goal_base_pose2d, goal_trunk -> measured sensor pose
    base = (12.100000381469727, 3.4000000953674316, -0.009270520212727895)
    translate, tilt = 0.7563274801164571, 0.044868714644566765
    target_p = th.tensor([12.295814514160156, 3.3981618881225586, 1.2537109851837158])
    target_q = th.tensor([0.4044118821620941, -0.40817201137542725, -0.5814059972763062, 0.5760327577590942])
    cam = C.camera_pose(C.torso_q(translate, tilt), base)
    assert float((cam[0] - target_p).norm()) < 0.002  # the instance's own goal_fk_error was 1.4 mm (FK vs measured)
    assert C.quat_angle(cam[1], target_q) < 1e-3
    # the 28-D path (virtual base joints, root at the origin) gives the same pose
    q28 = th.zeros(28)
    q28[[C.JOINT_INDEX[n] for n in M.TORSO_JOINTS]] = C.torso_q(translate, tilt)
    q28 = C.q_with_base_pose(q28, *base, base_z=C.NOMINAL_BASE_Z)
    cam2 = C.camera_pose(q28)
    assert _close(cam2[0], cam[0], 1e-5) and _same_quat(cam2[1], cam[1], 1e-5)
    # explicit base pose overrides the virtual joints; a Pose base is taken as is
    cam3 = C.camera_pose(q28, base_pose=C.pose2d_to_pose(*base, z=C.NOMINAL_BASE_Z))
    assert _close(cam3[0], cam[0], 1e-5)
    with pytest.raises(ValueError):
        C.camera_pose(th.zeros(7))


def test_camera_axis_conventions_at_tilt_zero():
    # spec 4.3: at tilt 0 the optical axis pitches ~22.6 deg down (design value), the camera is ~0.2 m ahead of the base
    # and its height decreases monotonically with translate
    prev_z = None
    for translate in (0.0, 0.25, 0.5, 0.75, 0.95):
        cam = C.camera_pose(C.torso_q(translate, 0.0))
        axis = C.quat_rotate(cam[1], th.tensor([0.0, 0.0, -1.0]))  # USD camera looks along -z
        up = C.quat_rotate(cam[1], th.tensor([0.0, 1.0, 0.0]))
        assert float(axis[0]) > 0.9 and abs(float(axis[1])) < 1e-3  # looks ahead, no yaw / roll
        assert math.degrees(math.asin(-float(axis[2]))) == pytest.approx(22.6, abs=0.5)
        assert float(up[2]) > 0.9
        assert 0.18 < float(cam[0][0]) < 0.25
        if prev_z is not None:
            assert float(cam[0][2]) < prev_z
        prev_z = float(cam[0][2])
    # a negative tilt (JoyLo "look down") pitches the camera further down
    a0 = C.quat_rotate(C.camera_pose(C.torso_q(0.5, 0.0))[1], th.tensor([0.0, 0.0, -1.0]))
    a1 = C.quat_rotate(C.camera_pose(C.torso_q(0.5, -0.3))[1], th.tensor([0.0, 0.0, -1.0]))
    assert float(a1[2]) < float(a0[2]) - 0.2


def test_camera_offset_in_zed_measures_a_duck_typed_robot():
    class Prim:
        def __init__(self, pose):
            self._pose = pose

        def get_position_orientation(self):
            return self._pose

    zed = (th.tensor([1.0, 2.0, 1.6]), C.quat_from_axis_angle([0.3, -0.2, 0.9], 1.1))
    cam = C.compose(zed, C.CAMERA_IN_ZED)

    class Robot:
        sensors = {"robot_r1:left_realsense_link:Camera:0": Prim(zed), "robot_r1:zed_link:Camera:0": Prim(cam)}
        links = {M.HEAD_CAMERA_LINK: Prim(zed)}

    off = C.camera_offset_in_zed(Robot())
    assert _close(off[0], C.CAMERA_IN_ZED[0], 1e-5) and C.quat_angle(off[1], C.CAMERA_IN_ZED[1]) < 1e-5
    off2 = C.camera_offset_in_zed(Robot(), sensor_key="robot_r1:zed_link:Camera:0")
    assert _close(off2[0], off[0], 1e-6)


def test_torso_chain_matches_the_urdf_when_available():
    from ispatialgym import REPO_ROOT  # (not M.robot_assets_dir(): that would import OmniGibson to find the dataset)

    urdf = os.path.join(REPO_ROOT, "Behavior-1K", "datasets", "omnigibson-robot-assets", "models", "r1pro", "urdf", "r1pro.urdf")
    if not os.path.exists(urdf):
        pytest.skip("R1Pro URDF not available")
    import xml.etree.ElementTree as ET

    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    for i, (xyz, axis) in enumerate(C._TORSO_CHAIN):
        j = joints[f"torso_joint{i + 1}"]
        assert [float(v) for v in j.find("origin").get("xyz").split()] == pytest.approx(list(xyz), abs=1e-9)
        assert [float(v) for v in j.find("axis").get("xyz").split()] == pytest.approx(list(axis))
        lim = j.find("limit")
        assert float(lim.get("lower")) == pytest.approx(float(C.TORSO_JOINT_LIMITS[0][i])) and float(lim.get("upper")) == pytest.approx(float(C.TORSO_JOINT_LIMITS[1][i]))
    zed = joints["zed_joint"].find("origin")
    assert [float(v) for v in zed.get("xyz").split()] == pytest.approx(list(C._ZED_IN_TORSO4[0]), abs=1e-9)
    assert [float(v) for v in zed.get("rpy").split()] == pytest.approx(list(C._ZED_IN_TORSO4[1]), abs=1e-9)
    for arm in ("left", "right"):
        base = [float(v) for v in joints[f"{arm}_arm_base_joint"].find("origin").get("xyz").split()]
        j1 = [float(v) for v in joints[f"{arm}_arm_joint1"].find("origin").get("xyz").split()]
        assert [a + b for a, b in zip(base, j1)] == pytest.approx(list(C._SHOULDER_IN_TORSO4[arm]), abs=1e-9)
    # purity of this test's imports is checked in a fresh interpreter by test_module_is_pure


# ------------------------------------------------------------------------------------------------------------------
# blocks (spec 6.5)
# ------------------------------------------------------------------------------------------------------------------


def test_cube_grasp_geometry_and_tilt_pairing():
    block = (th.tensor([0.6, -0.1, 0.72 + C.BLOCK_EDGE / 2]), C.quat_from_yaw(math.radians(12.0)))
    g = C.cube_grasp(block, "y", 0.0, yaw=math.radians(12.0))
    # top-down pinch: a straight down, pads close along the block's y axis, wrist yaw snapped to the block
    assert _close(g.approach, th.tensor([0.0, 0.0, -1.0]), 1e-6)
    assert abs(abs(float(th.dot(g.closing, C.quat_rotate(block[1], th.tensor([0.0, 1.0, 0.0]))))) - 1.0) < 1e-6
    assert abs(C.wrap_angle(g.yaw - math.radians(12.0))) < 1e-6 or abs(C.wrap_angle(g.yaw - math.radians(12.0) - math.pi)) < 1e-6
    # the pinch point is on the approach line through the centre, the fingertips 3 mm above the bottom plane
    bottom = float(block[0][2]) - C.BLOCK_EDGE / 2
    assert float(C.fingertip(g.eef)[2]) == pytest.approx(bottom + 0.003, abs=1e-6)
    assert abs(float(g.pinch_point[0] - block[0][0])) < 1e-6 and abs(float(g.pinch_point[1] - block[0][1])) < 1e-6
    assert _close(C.pad_center(g.eef), g.pinch_point, 1e-6)
    # the pads still reach past the block centre (contact on both faces)
    pad_far = g.eef[0] + C.PAD_Z_RANGE[1] * g.approach
    assert float(pad_far[2]) < float(block[0][2])
    # a big block: pad centre at its centre
    big = C.cube_grasp((th.tensor([0.0, 0.0, 0.1]), C.IDENTITY_QUAT), "x", 0.0, yaw=math.pi / 2, edge=0.2)
    assert _close(big.pinch_point, th.tensor([0.0, 0.0, 0.1]), 1e-6)
    # tilt: the approach turns about the closing axis by the tilt, staying perpendicular to it (pads flat)
    for tilt in (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0):
        gt = C.cube_grasp(block, "y", tilt, yaw=math.radians(12.0))
        assert abs(float(th.dot(gt.approach, gt.closing))) < 1e-6
        assert _close(gt.closing, g.closing, 1e-6)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(gt.approach, g.approach))))))
        assert ang == pytest.approx(abs(tilt), abs=1e-4)
        assert gt.downward
        assert float(C.fingertip(gt.eef)[2]) >= bottom + 0.003 - 1e-6
    # pairing d_pick = R_a(-/+90) d_place: place tilt +45 <-> pick tilt -45 with a +90 deg flip about the pinch axis
    theta = 45.0
    pick = C.cube_grasp(block, "y", C.cube_pick_tilt(theta, +1), yaw=math.radians(12.0))
    assert pick.tilt_deg == -45.0 and pick.downward
    flipped = (block[0], C.qmul(C.quat_from_axis_angle(pick.closing, math.radians(90.0)), block[1]))
    place_eef = C.compose(flipped, pick.eef_in_block)  # the same rigid grasp after the flip
    a_place = C.approach_axis(place_eef[1])
    expected = C.cube_grasp(flipped, "y", theta, yaw=math.radians(12.0))
    assert _close(a_place, expected.approach, 1e-5)
    assert a_place[2] < 0  # the place approach comes from above
    assert _close(C.quat_rotate(C.quat_from_axis_angle(pick.closing, math.radians(90.0)), pick.approach), a_place, 1e-5)
    # exactly one flip sign gives a downward pick approach for the place tilt theta
    downs = [C.cube_grasp(block, "y", C.cube_pick_tilt(theta, s), yaw=math.radians(12.0)).downward for s in (+1, -1)]
    assert downs.count(True) == 1
    # a vertical face pair cannot be pinched for a flip
    with pytest.raises(ValueError):
        C.cube_grasp(block, "z", 0.0, yaw=0.0)
    with pytest.raises(ValueError):
        C.cube_grasp(block, "w", 0.0, yaw=0.0)


def test_flip_waypoints_turn_the_block_in_place():
    block = (th.tensor([0.6, -0.1, 0.80]), C.quat_from_yaw(0.2))
    g = C.cube_grasp(block, "y", -45.0, yaw=0.2)
    pivot = block[0]
    psi = -0.2
    flip_axis = C.quat_rotate(C.quat_from_yaw(psi), g.closing)  # the pinch axis after the yaw stage = a table axis
    wps = C.flip_waypoints(g.eef, pivot, psi, flip_axis, +1, n_yaw=2, n_flip=3)
    assert len(wps) == 5
    r0 = float((g.eef[0] - pivot).norm())
    for p, q in wps:
        assert float((p - pivot).norm()) == pytest.approx(r0, abs=1e-5)
    # the block, rigidly held, ends yaw-aligned and flipped by 90 deg about the table axis
    block_end = C.compose(wps[-1], C.invert(g.eef_in_block))
    expected_q = C.qmul(C.quat_from_axis_angle(flip_axis, math.pi / 2), C.qmul(C.quat_from_yaw(psi), block[1]))
    assert _close(block_end[0], pivot, 1e-4) and C.quat_angle(block_end[1], expected_q) < 1e-4
    # the place approach of the flipped block is the +45 deg pick partner
    assert math.degrees(math.acos(-float(C.approach_axis(wps[-1][1])[2]))) == pytest.approx(45.0, abs=1e-3)
    with pytest.raises(ValueError):
        C.flip_waypoints(g.eef, pivot, 0.0, [0.0, 0.0, 1.0])
    # without a yaw stage only the flip poses are returned; the counts are keyword-only
    assert len(C.flip_waypoints(g.eef, pivot, 0.0, flip_axis, n_flip=4)) == 4
    with pytest.raises(TypeError):
        C.flip_waypoints(g.eef, pivot, 0.0, flip_axis, 1, 4)


# ------------------------------------------------------------------------------------------------------------------
# grasp library
# ------------------------------------------------------------------------------------------------------------------


def test_grasp_pose_from_transform_and_yaw_semantics():
    eef_in_obj = (th.tensor([0.0, 0.02, 0.05]), C.quat_from_axis_angle([1.0, 0.0, 0.0], math.pi))  # approach -z of the object
    obj = (th.tensor([1.0, 2.0, 0.75]), C.quat_from_yaw(0.3))
    eef = C.grasp_pose_from_transform(obj, eef_in_obj)
    assert _close(eef[0], obj[0] + C.quat_rotate(obj[1], eef_in_obj[0]), 1e-6)
    assert _close(C.approach_axis(eef[1]), th.tensor([0.0, 0.0, -1.0]), 1e-6)
    # the object pose is recovered from the eef through the inverse transform (the TRO's object_pose_in_eef_frame)
    back = C.compose(eef, C.invert(eef_in_obj))
    assert _close(back[0], obj[0], 1e-5) and _same_quat(back[1], obj[1], 1e-5)


def _library_key():
    from ispatialgym.sampling import grasp_library as GL

    key = "apple__agveuv__t"
    if not os.path.exists(os.path.join(GL.library_root(), "results", f"{key}.json")):
        pytest.skip("grasp library not available")
    return key


def test_grasp_pose_from_the_library():
    key = _library_key()
    g = C.grasp_entry(key, "grasp_0")
    assert g.key == key and g.name == "grasp_0"
    obj = (th.tensor([0.5, 0.2, 0.75]), C.rest_quat(key))
    eef = C.grasp_pose(obj, key, "grasp_0")
    assert _close(eef[0], C.compose(obj, g.pose)[0], 1e-6) and _same_quat(eef[1], C.compose(obj, g.pose)[1], 1e-6)
    # a tabletop grasp approaches from above
    assert float(C.approach_axis(eef[1])[2]) < -0.5
    # yaw given: the object stands at R_z(yaw) * rest orientation - equal to passing that object pose with yaw=None,
    # and independent of the orientation obj_pose carries (no double application with the recorded yaw)
    yaw = 0.8
    e1 = C.grasp_pose(obj, key, "grasp_0", yaw)
    e2 = C.grasp_pose((obj[0], C.qmul(C.quat_from_yaw(yaw), C.rest_quat(key))), key, "grasp_0")
    e3 = C.grasp_pose((obj[0], C.quat_from_yaw(-2.0)), key, "grasp_0", yaw)
    assert _close(e1[0], e2[0], 1e-6) and _same_quat(e1[1], e2[1], 1e-6)
    assert _close(e1[0], e3[0], 1e-6) and _same_quat(e1[1], e3[1], 1e-6)
    assert float((e1[0] - obj[0]).norm()) == pytest.approx(float((eef[0] - obj[0]).norm()), abs=1e-6)  # rigid about the object
    assert _close(C.quat_rotate(C.quat_from_yaw(yaw), C.approach_axis(eef[1])), C.approach_axis(e1[1]), 1e-5)  # eef turns with it
    # object in eef = inverse of the entry
    oie = C.object_in_eef(key, "grasp_0")
    ident = C.compose(g.pose, oie)
    assert _close(ident[0], th.zeros(3), 1e-6) and _same_quat(ident[1], C.IDENTITY_QUAT, 1e-6)
    with pytest.raises(KeyError):
        C.grasp_entry(key, "no_such_grasp")
