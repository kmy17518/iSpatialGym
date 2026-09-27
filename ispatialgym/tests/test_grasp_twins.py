"""Tests of the gripper-symmetry ("twin") grasps: ``conventions.twin_pose`` / ``split_twin`` / ``twin_arm_q`` (checked against
a numpy FK of ``r1pro.urdf``) and the ``~r180`` alias of ``grasp_entry``, ``grasp_library.twin`` / ``with_twins``, the
wrist-camera clearance and ``pick_twin`` of ``sampling/v2/tabletop.py`` (the orientation rule of ``sampling/v2/orient.py``,
tested on its own in ``test_orient.py``) on a stub IK, the band write-back of an alias and the call sites ``phase3_target`` /
``IORSSmall.reselect_grasp`` / ``IORSSmall.grasp_facts``. No simulator is launched.
"""

import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.v2 import tabletop as TT

GEN = th.Generator().manual_seed(7)


def _rand_quat():
    q = th.randn(4, generator=GEN)
    return q / q.norm()


def _same_quat(a, b, tol=1e-5):
    a, b = C.as_tensor(a), C.as_tensor(b)
    return min(float((a - b).abs().max()), float((a + b).abs().max())) < tol


# ------------------------------------------------------------------------------------------------------------------
# conventions
# ------------------------------------------------------------------------------------------------------------------


def test_twin_pose_is_the_same_pinch_turned_half_about_the_approach():
    half = (th.zeros(3), C.quat_from_axis_angle([0.0, 0.0, 1.0], math.pi))
    for _ in range(20):
        pose = (th.randn(3, generator=GEN), _rand_quat())
        pos_t, quat_t = C.twin_pose(pose)
        R, Rt = C.quat_to_mat(pose[1]), C.quat_to_mat(quat_t)
        assert th.allclose(pos_t, pose[0])
        assert th.allclose(Rt[:, 2], R[:, 2], atol=1e-5), "the approach axis is kept"
        assert th.allclose(Rt[:, 1], -R[:, 1], atol=1e-5), "the closing axis is negated"
        assert th.allclose(Rt[:, 0], -R[:, 0], atol=1e-5)
        assert _same_quat(C.compose(pose, half)[1], quat_t)
        back = C.twin_pose((pos_t, quat_t))
        assert th.allclose(back[0], pose[0]) and _same_quat(back[1], pose[1])
        # the pad centre and the fingertip point lie on the approach axis: the twin pinches at the same place
        assert th.allclose(C.pad_center((pos_t, quat_t)), C.pad_center(pose), atol=1e-6)
        assert th.allclose(C.fingertip((pos_t, quat_t)), C.fingertip(pose), atol=1e-6)
    framed = C.twin_pose(C.FramedPose.create([0.1, 0.2, 0.3], C.IDENTITY_QUAT, frame="obj:apple"))
    assert isinstance(framed, C.FramedPose) and framed.frame == "obj:apple"
    for name in ("HALF_TURN_Z", "TWIN_SUFFIX", "WRIST_ROLL_LIMIT", "twin_pose", "split_twin", "twin_name", "twin_arm_q"):
        assert name in C.__all__ and hasattr(C, name), name


def test_split_twin_and_twin_name():
    assert C.split_twin("grasp_3~r180") == ("grasp_3", True)
    assert C.split_twin("grasp_3") == ("grasp_3", False)
    assert C.split_twin("grasp~r180_x") == ("grasp~r180_x", False), "only the suffix names a twin"
    assert C.twin_name("grasp_3") == "grasp_3~r180" and C.twin_name("grasp_3~r180") == "grasp_3"
    assert C.twin_name(C.twin_name("grasp_12")) == "grasp_12"


def test_twin_arm_q_maps_the_wrist_within_joint_5s_range():
    q = th.tensor([0.1, -0.5, 0.3, -1.0, math.radians(100.0), 0.4, -0.6])
    t = C.twin_arm_q(q)
    assert th.allclose(t[:4], q[:4]), "the shoulder and the elbow are kept"
    assert float(t[4]) == pytest.approx(math.radians(100.0) - math.pi, abs=1e-6)
    assert float(t[5]) == pytest.approx(-0.4) and float(t[6]) == pytest.approx(0.6)
    # joint 5's margin moves from limit - |q5| (0.61 rad) to |q5| - 45° (0.96 rad)
    assert C.WRIST_ROLL_LIMIT - abs(float(t[4])) == pytest.approx(math.radians(100.0) - math.pi / 4, abs=1e-5)
    assert th.allclose(C.twin_arm_q(t), q, atol=1e-6)
    assert float(C.twin_arm_q([0.0] * 4 + [-math.radians(120.0), 0.0, 0.0])[4]) == pytest.approx(math.radians(60.0), abs=1e-5)
    # with the same shoulder and elbow the twin exists only for |q5| >= 45°
    assert C.twin_arm_q([0.0] * 4 + [math.radians(44.0), 0.0, 0.0]) is None
    assert C.twin_arm_q([0.0] * 4 + [-math.radians(44.0), 0.0, 0.0]) is None
    assert C.twin_arm_q([0.0] * 4 + [math.radians(46.0), 0.0, 0.0]) is not None
    assert C.twin_arm_q([0.0] * 7) is None
    assert C.twin_arm_q(q.double()).dtype == th.float64
    with pytest.raises(ValueError):
        C.twin_arm_q(th.zeros(28))


def _r1pro_urdf():
    from ispatialgym import REPO_ROOT

    roots = [os.path.join(REPO_ROOT, "Behavior-1K", "datasets")]
    if os.environ.get("OMNIGIBSON_DATA_PATH"):
        roots.append(os.environ["OMNIGIBSON_DATA_PATH"])
    spec = importlib.util.find_spec("omnigibson")  # the installed package's location; OmniGibson itself is not imported
    if spec is not None and spec.origin:
        roots.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(spec.origin))), "datasets"))
    for root in roots:
        path = os.path.join(root, "omnigibson-robot-assets", "models", "r1pro", "urdf", "r1pro.urdf")
        if os.path.exists(path):
            return path
    return None


def _hom(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _axis_rot(axis, angle):
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K
    return T


def _arm_eef_fk(joints, arm, q):
    """eef pose (4 x 4) in ``<arm>_arm_base_link`` for the 7 arm joint values: the URDF chain, then ``EEF_IN_GRIPPER``."""

    def origin(j):
        o = j.find("origin")
        return _hom([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy", "0 0 0").split()])

    T = np.eye(4)
    for i in range(7):
        j = joints[f"{arm}_arm_joint{i + 1}"]
        T = T @ origin(j) @ _axis_rot([float(v) for v in j.find("axis").get("xyz").split()], float(q[i]))
    T = T @ origin(joints[f"{arm}_gripper_joint"])
    return T @ _hom(M.EEF_IN_GRIPPER_XYZ, M.EEF_IN_GRIPPER_RPY)


def test_twin_arm_q_holds_the_twin_pose_in_the_urdf_fk():
    urdf = _r1pro_urdf()
    if urdf is None:
        pytest.skip("R1Pro URDF not available")
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    half = np.diag([-1.0, -1.0, 1.0, 1.0])  # R_z(π)
    rng = np.random.default_rng(3)
    for arm in ("left", "right"):
        lim = [(float(joints[f"{arm}_arm_joint{i}"].find("limit").get("lower")), float(joints[f"{arm}_arm_joint{i}"].find("limit").get("upper")))
               for i in range(1, 8)]
        # the premises: joint 5's range is WRIST_ROLL_LIMIT, joints 6 and 7 are symmetric
        assert lim[4] == pytest.approx((-C.WRIST_ROLL_LIMIT, C.WRIST_ROLL_LIMIT))
        assert lim[5][0] == pytest.approx(-lim[5][1]) and lim[6][0] == pytest.approx(-lim[6][1])
        n_twins = 0
        for _ in range(300):
            q = np.array([rng.uniform(lo, hi) for lo, hi in lim])
            t = C.twin_arm_q(th.tensor(q, dtype=th.float64))
            if abs(q[4]) < math.pi / 4:
                assert t is None
                continue
            n_twins += 1
            tq = t.numpy()
            assert all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(tq, lim)), "the twin stays within the joint limits"
            want = _arm_eef_fk(joints, arm, q) @ half
            got = _arm_eef_fk(joints, arm, tq)
            assert np.abs(got[:3, 3] - want[:3, 3]).max() < 1e-9 and np.abs(got[:3, :3] - want[:3, :3]).max() < 1e-9
        assert n_twins > 100


def _library_key():
    key = "apple__agveuv__t"
    if not os.path.exists(os.path.join(GL.library_root(), "results", f"{key}.json")):
        pytest.skip("grasp library not available")
    return key


def test_grasp_entry_resolves_the_twin_alias():
    key = _library_key()
    g = C.grasp_entry(key, "grasp_0")
    t = C.grasp_entry(key, "grasp_0~r180")
    assert t.name == "grasp_0~r180" and t.key == key and t.meta.get("twin_of") == "grasp_0"
    assert th.allclose(t.position, g.position) and _same_quat(t.quat, C.twin_pose(g.pose)[1])
    assert t.finger_q == list(reversed(g.finger_q))
    # the pinch-width look-ups by name (oracle pre-shape, attachment) read the row's width through the alias
    assert (t.meta or {}).get("width") == (g.meta or {}).get("width")
    obj = (th.tensor([0.5, 0.2, 0.75]), C.rest_quat(key))
    e_twin, e_row = C.grasp_pose(obj, key, "grasp_0~r180"), C.twin_pose(C.grasp_pose(obj, key, "grasp_0"))
    assert th.allclose(e_twin[0], e_row[0], atol=1e-6) and _same_quat(e_twin[1], e_row[1])
    ident = C.compose(t.pose, C.object_in_eef(key, "grasp_0~r180"))
    assert th.allclose(ident[0], th.zeros(3), atol=1e-6) and _same_quat(ident[1], C.IDENTITY_QUAT)
    assert C.grasp_entry(key, "grasp_0") is g and g.name == "grasp_0" and "twin_of" not in g.meta
    with pytest.raises(KeyError):
        C.grasp_entry(key, "no_such_grasp~r180")


# ------------------------------------------------------------------------------------------------------------------
# grasp library
# ------------------------------------------------------------------------------------------------------------------


def _grasp(name, pos, quat, finger_q=(0.05, 0.02)):
    return GL.Grasp(key="k", name=name, position=th.tensor(pos, dtype=th.float32), quat=C.as_tensor(quat), finger_q=list(finger_q), meta={"width": 0.03})


def test_library_twin_stored_twin_and_with_twins():
    a = _grasp("grasp_0", [0.0, 0.0, 0.1], C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    b = _grasp("grasp_1", [0.0, 0.03, 0.1], C.eef_quat([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    # a row that already is grasp_0's twin up to 8 mm and 10° about the approach axis
    c = _grasp("grasp_2", [0.008, 0.0, 0.1], C.qmul(C.twin_pose(a.pose)[1], C.quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(10.0))))
    t = GL.twin(a)
    assert t.name == "grasp_0~r180" and t.key == "k" and t.finger_q == [0.02, 0.05] and t.meta == {"width": 0.03, "twin_of": "grasp_0"}
    assert th.allclose(t.position, a.position) and _same_quat(t.quat, C.twin_pose(a.pose)[1])
    tt = GL.twin(t)
    assert tt.name == "grasp_0" and tt.meta == {"width": 0.03} and tt.finger_q == a.finger_q and _same_quat(tt.quat, a.quat)
    rows = [a, b, c]
    assert GL.stored_twin(a, rows) is c and GL.stored_twin(c, rows) is a and GL.stored_twin(b, rows) is None
    assert GL.stored_twin(a, rows, ang_tol_deg=5.0) is None and GL.stored_twin(a, rows, pos_tol=0.005) is None
    assert [g.name for g in GL.with_twins(rows)] == ["grasp_0", "grasp_1", "grasp_2", "grasp_1~r180"]


# ------------------------------------------------------------------------------------------------------------------
# the switch, the wrist camera, the band write-back
# ------------------------------------------------------------------------------------------------------------------


def test_grasp_twins_switch_defaults_on():
    def on(v):
        return TT.grasp_twins_on(SimpleNamespace(gp=SimpleNamespace(gen=lambda k, d=None: v if k == "grasp_twins" else d)))

    assert on(True) and on("on") and on(None) and on(1)
    assert not on(False) and not on("off") and not on("OFF") and not on(0) and not on("false")
    assert TT.grasp_twins_on(SimpleNamespace(gen=lambda k, d=None: d)), "a generation block without the key: on"
    assert TT.grasp_twins_on(object())


def test_wrist_camera_changes_side_with_the_twin():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    assert [tuple(c) for c in TT.WRIST_CAMERA_SPHERES_EEF] == [tuple(c) for c in IS.HAND_SPHERES_EEF["wrist camera"][0]]
    assert TT.WRIST_CAMERA_SPHERE_R == IS.HAND_SPHERES_EEF["wrist camera"][1]
    # a horizontal side pinch 5 cm above a table top at z = 0.70 with eef +x up hangs the camera (eef -x side) below the pads
    eef = (th.tensor([0.6, 0.0, 0.75]), C.eef_quat([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]))
    assert float(C.quat_to_mat(eef[1])[2, 0]) == pytest.approx(1.0)
    assert TT.wrist_camera_clearance(eef, 0.70) == pytest.approx(0.05 - 0.056 - 0.018, abs=1e-5)
    assert TT.wrist_camera_clearance(C.twin_pose(eef), 0.70) == pytest.approx(0.05 + 0.056 - 0.018, abs=1e-5)
    # a top-down pinch keeps the camera 7.6 cm above the eef origin whichever way the fingers close
    down = (th.tensor([0.6, 0.0, 0.72]), C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    assert TT.wrist_camera_clearance(down, 0.70) == pytest.approx(0.02 + 0.076 - 0.018, abs=1e-5)
    assert TT.wrist_camera_clearance(C.twin_pose(down), 0.70) == pytest.approx(TT.wrist_camera_clearance(down, 0.70), abs=1e-6)


def test_store_feasible_set_indexes_a_twin_by_its_library_row():
    class Bands:
        def __init__(self):
            self.meta = {"grasp_names": ["grasp_0", "grasp_1", "grasp_2"]}
            self.cells = {(0.5, 0.0): np.array([[0.6, 0.0], [0.7, 0.1]], dtype=np.float32)}
            self.pairs = {}
            self.path = None

        def nearest_trunk(self, translate, tilt):
            return (0.5, 0.0)

        def to_base(self, xy, pose2d):
            return np.asarray(xy, dtype=float).reshape(1, 2)

        def cell_index(self, translate, tilt, cell):
            return 1

    bands = Bands()
    stance = SimpleNamespace(focus=th.tensor([0.7, 0.1, 0.8]), translate=0.5, tilt=0.0, base_pose2d=(0.0, 0.0, 0.0))
    TT.store_feasible_set(bands, stance, [(0.0, SimpleNamespace(name="grasp_2~r180")), (math.pi / 4, SimpleNamespace(name="grasp_0"))])
    assert bands.pairs[(0.5, 0.0)].tolist() == [[1, 0, 2], [1, 1, 0]], "no -1 (read back as the last row's name)"
    assert [n for _, n in bands.meta["feasible_sets"][0]["feasible"]] == ["grasp_2", "grasp_0"]


# ------------------------------------------------------------------------------------------------------------------
# IKChecker.twin_q and tabletop.pick_twin on a stub IK
# ------------------------------------------------------------------------------------------------------------------

#: right-arm slots of the 28-D OmniGibson vector and the r1pro.urdf limits of right_arm_joint1..7
ARM_IDX = th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS["right"]])
ARM_LO = [-4.4506, -3.1416, -2.356196, -2.0944, -2.356196, -1.047198, -1.5708]
ARM_HI = [1.3090, 0.1745, 2.356196, 0.3491, 2.356196, 1.047198, 1.5708]


def _q(arm_values):
    q = th.zeros(28)
    q[ARM_IDX] = th.tensor(arm_values, dtype=th.float32)
    return q


class _Sampler:
    """A sampler stand-in for ``pick_twin``: a real ``IKChecker`` on a kit with the URDF arm limits, the stance configuration
    ``q_start`` and an ``exact`` answered by ``self.answer(poses, kw) -> (q_pre, q_end, reason)`` (``q_end`` None = failed); the
    witnesses it returns are remembered like ``IKChecker.exact`` does."""

    def __init__(self, q_start, answer=None, witness_ok=True, twins=True):
        from ispatialgym.sampling.v2.ik import IKChecker, IKResult

        lo, hi = th.full((28,), -10.0), th.full((28,), 10.0)
        lo[ARM_IDX], hi[ARM_IDX] = th.tensor(ARM_LO), th.tensor(ARM_HI)
        self.kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": ARM_IDX}, joint_lower_limits=lo, joint_upper_limits=hi), stats={})
        self.ikc = ikc = IKChecker(self.kit)
        ikc.q_locked = lambda stance, **kw: q_start.clone()
        ikc._witness_ok = lambda q, pose, arm: witness_ok
        self.calls = []
        self.answer = answer
        self.gen = lambda key, default=None: (twins if key == "grasp_twins" else default)

        def exact(poses, stance, arm="right", **kw):
            self.calls.append(kw)
            q_pre, q_end, reason = self.answer(poses, kw)
            if q_pre is not None:
                ikc.solutions[(arm, ikc._pose_key(poses[0]))] = q_pre.clone()
            if q_end is None:
                return IKResult(False, None, reason or "exact IK failed at pose 1 of 2", len(poses))
            ikc.solutions[(arm, ikc._pose_key(poses[-1]))] = q_end.clone()
            return IKResult(True, q_end.clone(), "", len(poses))

        ikc.exact = exact

    def row(self, pre, eef, q_pre, q_end, reason=""):
        """The row's exact IK as the caller ran it: witnesses remembered, failed with ``reason`` when ``q_end`` is None."""
        from ispatialgym.sampling.v2.ik import IKResult

        if q_pre is not None:
            self.ikc.solutions[("right", self.ikc._pose_key(pre))] = q_pre.clone()
        if q_end is None:
            return IKResult(False, None, reason, 2)
        self.ikc.solutions[("right", self.ikc._pose_key(eef))] = q_end.clone()
        return IKResult(True, q_end.clone(), "", 2)


def _top_down_row(name="grasp_4", obj_z=0.75):
    from ispatialgym.sampling.v2 import ik as IK

    g = _grasp(name, [0.0, 0.0, 0.12], C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    eef = IK.grasp_pose((th.tensor([0.6, -0.2, obj_z]), C.IDENTITY_QUAT), g.pose)
    return g, g.pose, eef, IK.pre_pose(eef)


def _twin_of(s, q):
    return s.ikc.twin_q(q, "right")


def _answer(s, eef, row, twin):
    """``exact`` answers by chain: ``row`` / ``twin`` = the fresh chain's ``(q_pre, q_end, reason)`` (told apart by the closing
    axis of the chain's last pose); a chain started from ``q_first`` (a wrist-flipped configuration) ends at the joint map of
    the other chain's end."""
    c_row = C.closing_axis(eef[1])

    def answer(poses, kw):
        is_row = float(th.dot(C.closing_axis(poses[-1][1]), c_row)) > 0.0
        if kw.get("q_first") is not None:
            return kw["q_first"], _twin_of(s, (twin if is_row else row)[1]), ""
        return row if is_row else twin

    return answer


def test_ik_twin_q_maps_the_arm_joints_only():
    s = _Sampler(_q([0.0] * 7))
    q = th.arange(28, dtype=th.float32) * 0.01
    q[ARM_IDX] = th.tensor([0.1, -0.5, 0.3, -1.0, -2.0, 0.4, -0.6])
    t = s.ikc.twin_q(q, "right")
    others = [i for i in range(28) if i not in ARM_IDX.tolist()]
    assert th.equal(t[others], q[others])
    assert th.allclose(t[ARM_IDX], C.twin_arm_q(q[ARM_IDX]))
    q[ARM_IDX[4]] = 0.3
    assert s.ikc.twin_q(q, "right") is None and s.ikc.twin_q(None, "right") is None


def test_pick_twin_takes_the_twin_when_the_row_fails_at_joint_5():
    g, eio, eef, pre = _top_down_row()
    q_pre = _q([0.3, -1.0, 0.2, -1.2, 2.20, 0.4, -0.5])
    q_end_row = _q([0.3, -1.1, 0.2, -1.3, 2.35, 0.45, -0.55])  # the row's tracked approach ends on joint 5's limit
    s = _Sampler(_q([0.0, -1.0, 0.0, -1.5, -1.57, 0.0, 0.0]))
    s.answer = _answer(s, eef, row=(q_pre, q_end_row, ""), twin=(None, None, "exact IK failed at pose 1 of 2"))
    res = s.row(pre, eef, q_pre, None, "ik_joint_limit: right_arm_joint5 within 5 mrad of its limit at the end of the tracked approach")
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, res, {"ignore_for_last": [], "scale_key": None}, support_z=0.75, where="test")
    assert pick.twin and pick.passed and pick.res.ok and pick.grasp.name == "grasp_4~r180" and pick.grasp.finger_q == [0.02, 0.05]
    assert th.allclose(pick.eef[0], eef[0]) and _same_quat(pick.eef[1], C.twin_pose(eef)[1]) and _same_quat(pick.eef_in_object[1], C.twin_pose(eio)[1])
    assert th.allclose(pick.pre[0], pre[0], atol=1e-6), "the twin's pre-grasp stands where the row's does"
    # the twin's own solve failed; the row's pre-grasp witness, wrist-flipped, is the twin chain's start (B2's rescue path)
    assert len(s.calls) == 2 and "q_first" not in s.calls[0] and s.calls[0]["ignore_for_last"] == []
    assert s.calls[1]["cache"] is False and th.allclose(s.calls[1]["q_first"], _twin_of(s, q_pre)) and s.calls[1]["scale_key"] is None
    f = pick.facts
    assert f["stage"] == "test" and f["chosen"] == "twin" and f["reason"] == "only feasible" and f["source"] == "map"
    assert f["row"]["name"] == "grasp_4" and f["row"]["ok"] is False and f["row"]["why"].startswith("ik_joint_limit: right_arm_joint5")
    assert f["twin"]["name"] == "grasp_4~r180" and f["twin"]["ok"] and f["twin"]["source"] == "map" and f["twin"]["margin_rad"] == pytest.approx(0.597, abs=2e-3)
    assert f["orient"]["against_default"] is True
    assert pick.other is None, "the row did not solve: nothing to fall back to"
    assert s.kit.stats["twin_picks"] == 1 and s.kit.stats["twin_taken"] == 1 and s.kit.stats["twin_taken.only_feasible"] == 1 and s.kit.stats["twin_solves"] == 2
    assert s.kit.stats["orient_choices"] == 1 and s.kit.stats["orient_source.map"] == 1 and s.kit.stats["orient_against_default"] == 1


def test_pick_twin_takes_the_safe_twin_over_a_row_near_a_limit_in_the_rows_branch():
    g, eio, eef, pre = _top_down_row()
    q_pre = _q([0.3, -1.0, 0.2, -1.2, 2.25, 0.3, -0.4])
    q_end = _q([0.3, -1.1, 0.2, -1.3, 2.33, 0.35, -0.45])  # 26 mrad from joint 5's limit: within the 50 mrad joint-limit margin
    s = _Sampler(_q([0.3, -1.0, 0.2, -1.2, 2.0, 0.3, -0.4]))
    s.answer = _answer(s, eef, row=(q_pre, q_end, ""), twin=(None, None, "exact IK failed at pose 1 of 2"))
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q_pre, q_end), {"ignore_for_last": []}, support_z=0.75, where="B")
    assert pick.twin and pick.facts["reason"] == "safety" and pick.facts["source"] == "map"
    assert pick.facts["row"]["nearest_joint"] == "right_arm_joint5" and pick.facts["row"]["margin_rad"] == pytest.approx(0.026, abs=2e-3)
    assert pick.facts["twin"]["margin_rad"] >= 0.3 and th.allclose(s.calls[-1]["q_first"], _twin_of(s, q_pre))
    assert pick.other is not None and pick.other[0] is g and pick.other[2] is eef, "the row solved: a late flip may still record it"


def test_pick_twin_keeps_a_safe_row_that_is_easier_to_reach():
    g, eio, eef, pre = _top_down_row()
    q_pre = _q([0.3, -1.0, 0.2, -1.2, 2.10, 0.3, -0.4])
    q_end = _q([0.3, -1.1, 0.2, -1.3, 2.15, 0.35, -0.45])  # 206 mrad from joint 5's limit: safe; the twin is safe too
    s = _Sampler(_q([0.3, -1.0, 0.2, -1.2, 2.0, 0.3, -0.4]))
    s.answer = _answer(s, eef, row=(q_pre, q_end, ""), twin=(_twin_of(s, q_pre), _twin_of(s, q_end), ""))
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q_pre, q_end), {}, support_z=0.75)
    assert not pick.twin and pick.passed and pick.grasp is g and pick.eef is eef and pick.facts["reason"] == "reach"
    assert pick.facts["row"]["reach_rad"] == pytest.approx(0.10, abs=1e-3) and pick.facts["twin"]["reach_rad"] == pytest.approx(math.pi - 0.10, abs=1e-3)
    assert len(s.calls) == 1, "the twin solved once; the row is the caller's solve"
    assert pick.other is not None and pick.other[0].name == "grasp_4~r180"
    assert "twin_taken" not in s.kit.stats and s.kit.stats["orient_reason.reach"] == 1 and "orient_against_default" not in s.kit.stats


def test_pick_twin_keeps_the_row_within_the_reach_tolerance():
    g, eio, eef, pre = _top_down_row()
    q = _q([0.0, -1.0, 0.0, -1.5, -1.57, 0.0, 0.0])
    s = _Sampler(_q([0.0, -1.0, 0.0, -1.5, 0.0, 0.0, 0.0]))  # joint 5 halfway between the row's and the twin's wrist
    s.answer = _answer(s, eef, row=(q, q, ""), twin=(_twin_of(s, q), _twin_of(s, q), ""))
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q, q), {}, support_z=0.75)
    assert not pick.twin and pick.facts["reason"] == "default" and pick.facts["chosen"] == "row"
    assert pick.facts["row"]["reach_rad"] == pytest.approx(pick.facts["twin"]["reach_rad"], abs=0.01)


def test_pick_twin_takes_the_twin_that_is_easier_to_reach():
    g, eio, eef, pre = _top_down_row()
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.2, 0.0, 0.0])  # 2.8 rad of joint 5 from the stance posture; the twin (1.94) 0.34 rad
    s = _Sampler(_q([0.0, -1.0, 0.0, -1.5, 1.6, 0.0, 0.0]))
    s.answer = _answer(s, eef, row=(q_row, q_row, ""), twin=(_twin_of(s, q_row), _twin_of(s, q_row), ""))
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q_row, q_row), {}, support_z=0.75)
    assert pick.twin and pick.facts["reason"] == "reach" and pick.facts["source"] == "ik", "the fresh solve before its mapped copy"
    assert pick.facts["row"]["reach_rad"] == pytest.approx(2.8, abs=1e-3) and pick.facts["twin"]["reach_rad"] == pytest.approx(math.pi - 2.8, abs=1e-3)


def test_pick_twin_with_neither_safe_takes_the_branch_with_more_clearance():
    g, eio, eef, pre = _top_down_row()
    q_row = _q([0.3, 0.15, 0.2, -1.2, -1.5, 0.2, 0.1])  # joint 2 25 mrad from its limit
    q_tw = _q([0.3, 0.13, 0.2, -1.2, 1.6, -0.2, -0.1])  # the twin's own solve: joint 2 45 mrad from it
    s = _Sampler(_q([0.0, -1.0, 0.0, -1.5, 0.0, 0.0, 0.0]))
    s.answer = _answer(s, eef, row=(q_row, q_row, ""), twin=(q_tw, q_tw, ""))
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q_row, q_row), {}, support_z=0.75)
    # the twin's branch has the clearance (both assignments in it); within the reach tolerance the row comes before the twin
    assert not pick.twin and pick.facts["reason"] == "default" and pick.facts["source"] == "map"
    assert [o["label"] for o in pick.facts["orient"]["options"] if o["ok"] and not o.get("predicted")] == ["grasp_4", "grasp_4~r180", "grasp_4"]
    assert pick.facts["row"]["margin_rad"] == pytest.approx(0.0445, abs=1e-3) and th.allclose(s.calls[-1]["q_first"], _twin_of(s, q_tw))
    assert th.allclose(pick.res.q[ARM_IDX], _twin_of(s, q_tw)[ARM_IDX])


def test_pick_twin_skips_a_row_at_the_reach_boundary_and_solves_an_out_of_range_twin_from_scratch():
    g, eio, eef, pre = _top_down_row()
    s = _Sampler(_q([0.0] * 7), answer=lambda poses, kw: pytest.fail("no twin solve at the reach boundary"))
    res = s.row(pre, eef, None, None, "ik_reach: witness FK residual 12.0 mm > 5 mm at pose 2 of 2")
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, res, {})
    assert not pick.twin and not pick.passed and pick.res is res and pick.facts["reason"] == "neither feasible"
    assert "reach boundary" in pick.facts["twin"]["why"]
    # |q5| < 45°: the joint map has no twin; the twin chain is solved from scratch (and fails here)
    q_row = _q([0.0, -1.0, 0.0, -1.5, 0.3, 0.0, 0.0])
    s2 = _Sampler(_q([0.0] * 7), answer=lambda poses, kw: (None, None, "exact IK failed at pose 1 of 2"))
    pick2 = TT.pick_twin(s2, None, "right", g, eio, eef, pre, s2.row(pre, eef, q_row, q_row), {"cache": True})
    assert not pick2.twin and pick2.passed and pick2.facts["twin"]["source"] == "ik" and pick2.facts["reason"] == "only feasible"
    assert len(s2.calls) == 1 and "q_first" not in s2.calls[0] and s2.calls[0]["cache"] is True


def test_pick_twin_turns_the_wrist_camera_out_of_the_support():
    from ispatialgym.sampling.v2 import ik as IK

    # a horizontal side pinch 5 cm above the support top with the camera side down: the row's camera is 24 mm into the top
    g = _grasp("grasp_1", [-0.04, 0.0, 0.0], C.eef_quat([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]))
    eef = IK.grasp_pose((th.tensor([0.6, 0.0, 0.75]), C.IDENTITY_QUAT), g.pose)
    pre = IK.pre_pose(eef)
    q = _q([0.0, -1.0, 0.0, -1.5, -1.9, 0.2, 0.3])
    s = _Sampler(q.clone())
    s.answer = _answer(s, eef, row=(q, q, ""), twin=(_twin_of(s, q), _twin_of(s, q), ""))
    pick = TT.pick_twin(s, None, "right", g, g.pose, eef, pre, s.row(pre, eef, q, q), {}, support_z=0.70)
    assert pick.twin and pick.facts["reason"] == "only feasible"
    assert "wrist camera -24 mm from the support top" in pick.facts["row"]["why"]
    assert pick.other is None, "the row fails the camera check in either branch"
    # without the support height the camera is not judged: both safe, the twin 3.14 rad of wrist spin away -> the row
    s2 = _Sampler(q.clone())
    s2.answer = _answer(s2, eef, row=(q, q, ""), twin=(_twin_of(s2, q), _twin_of(s2, q), ""))
    pick2 = TT.pick_twin(s2, None, "right", g, g.pose, eef, pre, s2.row(pre, eef, q, q), {})
    assert not pick2.twin and pick2.facts["reason"] == "reach"


def test_pick_twin_applies_the_callers_check_to_both_assignments():
    g, eio, eef, pre = _top_down_row()
    q_row = _q([0.0, -1.0, 0.0, -1.5, -1.9, 0.0, 0.0])
    s = _Sampler(q_row.clone())
    s.answer = _answer(s, eef, row=(q_row, q_row, ""), twin=(_twin_of(s, q_row), _twin_of(s, q_row), ""))
    row_end = float(q_row[ARM_IDX[4]])
    only_twin = lambda q: abs(float(q[ARM_IDX[4]]) - row_end) > 1.0  # noqa: E731
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q_row, q_row), {}, accept=only_twin)
    assert pick.twin and pick.passed and pick.facts["row"]["why"] == "fails the caller's check"
    assert pick.other is None
    s2 = _Sampler(_q([0.0] * 7))
    s2.answer = _answer(s2, eef, row=(q_row, q_row, ""), twin=(_twin_of(s2, q_row), _twin_of(s2, q_row), ""))
    pick2 = TT.pick_twin(s2, None, "right", g, eio, eef, pre, s2.row(pre, eef, q_row, q_row), {}, accept=lambda q: False)
    assert not pick2.twin and not pick2.passed and not pick2.res.ok and "caller's check" in pick2.res.reason


def test_pick_twin_never_offers_a_barred_row():
    g, eio, eef, pre = _top_down_row()
    q = _q([0.0, -1.0, 0.0, -1.5, -1.9, 0.0, 0.0])
    s = _Sampler(q.clone())
    s.answer = _answer(s, eef, row=(q, q, ""), twin=(_twin_of(s, q), _twin_of(s, q), ""))
    # the caller's own verdict passed (it solved the row before barring it): still the twin, the row in neither branch
    pick = TT.pick_twin(s, None, "right", g, eio, eef, pre, s.row(pre, eef, q, q), {}, row_allowed=False)
    assert pick.twin and pick.facts["row"]["why"] == "the row is barred by the caller" and pick.other is None


# ------------------------------------------------------------------------------------------------------------------
# call sites
# ------------------------------------------------------------------------------------------------------------------


def _phase3_stub(twins, row_answer, row_end, twin_answer=(None, None, "exact IK failed at pose 1 of 2")):
    """``phase3_target``'s sampler with one library row in F: ``row_answer`` / ``twin_answer`` = the row / twin chain's own
    ``(q_pre, q_end, reason)``; a chain started from ``q_first`` ends at the joint map of the other chain's end (the row's:
    ``row_end``)."""
    g = _grasp("grasp_4", [0.0, 0.0, 0.12], C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    c_row = th.tensor([0.0, 1.0, 0.0])  # the row's closing axis at yaw 0; the jittered yaw stays within ±22.5°

    def answer(poses, kw):
        is_row = float(th.dot(C.closing_axis(poses[-1][1]), c_row)) > 0.0
        if kw.get("q_first") is not None:
            return kw["q_first"], _twin_of(s, twin_answer[1] if is_row else row_end), ""
        return row_answer if is_row else twin_answer

    s = _Sampler(_q([0.0, -1.0, 0.0, -1.5, -1.57, 0.0, 0.0]), answer=answer, twins=twins)
    s.world = SimpleNamespace(bottom_offset=lambda hull, quat: 0.02, footprint=lambda hull, pose: th.zeros(4, 2))
    s.ikc.feasible_sets = lambda cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of: {0: [(0.0, g)]}
    manip = SimpleNamespace(focus=th.tensor([0.6, -0.2, 0.8]), name="manipulation", translate=0.5, tilt=0.0)
    top = SimpleNamespace(z=0.75, polygon_supported=lambda fp, margin=0.0: True)
    return s, g, manip, top


def _phase3(s, g, manip, top):
    return TT.phase3_target(s, manip, np.random.default_rng(0), [g], th.zeros(8, 3), C.IDENTITY_QUAT, [], top=top, object_name="apple.n.01_1", n_positions=1)


def test_phase3_target_records_the_twin_the_oracle_executes():
    q_pre = _q([0.3, -1.0, 0.2, -1.2, 2.20, 0.4, -0.5])
    q_end = _q([0.3, -1.1, 0.2, -1.3, 2.35, 0.45, -0.55])  # where the row's tracked approach ends: joint 5's limit
    s, g, manip, top = _phase3_stub(True, (q_pre, None, "ik_joint_limit: right_arm_joint5 within 5 mrad of its limit"), q_end)
    pose, rec = _phase3(s, g, manip, top)
    assert rec.grasp_key == "k" and rec.grasp_name == "grasp_4~r180" and rec.finger_q == [0.02, 0.05]
    assert rec.extras["wrist_flip"] is True and rec.extras["twin_choice"]["reason"] == "only feasible" and rec.extras["twin_choice"]["stage"] == "phase3_target"
    assert rec.extras["twin_choice"]["source"] == "map", "the twin left from the row's pre-grasp witness"
    assert _same_quat(rec.eef_in_object[1], C.twin_pose(g.pose)[1])
    # what the oracle executes (``Grasp.grasp_pose``: object pose ∘ the recorded eef_in_object) is the recorded grasp pose
    e = C.compose(pose, rec.eef_in_object)
    assert th.allclose(e[0], rec.grasp[0], atol=1e-5) and _same_quat(e[1], rec.grasp[1])
    sd = rec.to_sidecar()
    assert sd["grasp_name"] == "grasp_4~r180" and sd["wrist_flip"] is True and sd["twin_choice"]["chosen"] == "twin"
    assert sd["eef_in_object"]["position"] == pytest.approx(g.position.tolist())


def test_phase3_target_without_twins_records_the_row_as_before():
    q = _q([0.0, -1.0, 0.0, -1.5, -2.2, 0.0, 0.0])  # 0.16 rad from joint 5's limit: safe, 0.63 rad of wrist from the stance
    s, g, manip, top = _phase3_stub(False, (q, q, ""), q)
    _, rec = _phase3(s, g, manip, top)
    assert rec.grasp_name == "grasp_4" and "wrist_flip" not in rec.extras and "twin_choice" not in rec.extras and len(s.calls) == 1
    assert rec.eef_in_object[0] is g.position and "twin_picks" not in s.kit.stats
    # with twins the safe row stays: its twin (the wrist flipped, 2.5 rad from the stance) is the harder one to reach
    s_on, g_on, _, _ = _phase3_stub(True, (q, q, ""), q)
    _, rec_on = _phase3(s_on, g_on, manip, top)
    assert rec_on.grasp_name == "grasp_4" and rec_on.extras["wrist_flip"] is False and rec_on.extras["twin_choice"]["reason"] == "reach"
    assert len(s_on.calls) == 2, "the row, then the twin's own solve; its mapped configuration was not needed"
    # a row 26 mrad from joint 5's limit and a safe twin (its own solve, |q5| < 45 deg: no flipped counterpart): the twin
    q_bad = _q([0.0, -1.0, 0.0, -1.5, -2.33, 0.0, 0.0])
    s_tw, g_tw, _, _ = _phase3_stub(True, (q_bad, q_bad, ""), q_bad, twin_answer=(_q([0.0, -1.0, 0.0, -1.5, 0.5, 0.0, 0.0]),) * 2 + ("",))
    _, rec_tw = _phase3(s_tw, g_tw, manip, top)
    assert rec_tw.grasp_name == "grasp_4~r180" and rec_tw.extras["twin_choice"]["reason"] == "safety" and rec_tw.extras["twin_choice"]["source"] == "ik"


def _reselect_stub(twins, pads_ok):
    """``IORSSmall.reselect_grasp``'s sampler: library rows grasp_1 (flush, the drawn grasp, its pre-grasp remembered) and
    grasp_2; every chain solves - a row at ``q``, a twin at the joint map of ``q`` - and ``pads_ok(q)`` is the pad check."""
    from ispatialgym.sampling.v2 import ik as IK
    from ispatialgym.sampling.v2.stances import GraspRecord

    obj = (th.tensor([0.6, -0.2, 0.77]), C.IDENTITY_QUAT)
    g1 = _grasp("grasp_1", [0.0, 0.0, 0.12], C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    g2 = _grasp("grasp_2", [0.0, 0.0, 0.12], C.eef_quat([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]))
    q = _q([0.0, -1.0, 0.0, -1.5, -1.9, 0.1, 0.1])

    def answer(poses, kw):
        if kw.get("q_first") is None:
            return q, q, ""
        return kw["q_first"], s.ikc.twin_q(q, "right"), ""

    s = _Sampler(q.clone(), answer=answer, twins=twins)
    s.grasps_flush, s.grasps = [g1], [g1, g2]
    s.support_objects = lambda: []
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=None)})
    s.obj = lambda role: "target"
    s.top = SimpleNamespace(z=0.75)
    s.open_pads_clear_of_support = lambda q_: pads_ok(q_)
    s.apply_height_extra_to_grasp = lambda grasp, t_pose, manip: None
    e1 = IK.grasp_pose(obj, g1.pose)
    rec = GraspRecord("apple.n.01_1", "right", "k", "grasp_1", 0.0, IK.pre_pose(e1), e1, finger_q=list(g1.finger_q), eef_in_object=g1.pose)
    s.ikc.solutions[("right", s.ikc._pose_key(IK.pre_pose(e1)))] = q.clone()  # the drawn grasp's pre-grasp solved at the settled pose
    return s, SimpleNamespace(object_poses={"apple.n.01_1": obj}), rec, g1


def test_reselect_grasp_offers_the_twin_of_the_barred_grasp():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    # the drawn grasp's open pads stand on the support and its twin's clear (only the twin's wrist is 3.14 rad away) -> the twin
    row_q5 = -1.9
    s, draft, rec, g1 = _reselect_stub(True, lambda q_: abs(float(q_[ARM_IDX[4]]) - row_q5) > 1.0)
    chosen = IS.IORSSmall.reselect_grasp(s, draft, rec, None, "apple.n.01_1", require_pad_clearance=True)
    assert chosen == "grasp_1~r180" and rec.grasp_name == "grasp_1~r180" and rec.finger_q == [0.02, 0.05]
    assert _same_quat(rec.eef_in_object[1], C.twin_pose(g1.pose)[1]) and _same_quat(rec.grasp[1], C.twin_pose(C.compose(draft.object_poses["apple.n.01_1"], g1.pose))[1])
    assert rec.extras["wrist_flip"] is True and rec.extras["twin_choice"]["stage"] == "reselect_grasp"
    assert rec.extras["twin_choice"]["row"]["why"] == "excluded by name" and rec.extras["twin_choice"]["source"] == "map"
    assert len(s.calls) == 2 and "q_first" not in s.calls[0] and s.calls[1]["q_first"] is not None and s.calls[1]["approached"] == ["target"], \
        "the twin's own solve (pads on the support), then the twin from the remembered pre-grasp, with the caller's IK options"
    # without twins the barred grasp is skipped and the next library row is taken (today's behaviour)
    s_off, draft_off, rec_off, _ = _reselect_stub(False, lambda q_: True)
    assert IS.IORSSmall.reselect_grasp(s_off, draft_off, rec_off, None, "apple.n.01_1", require_pad_clearance=True) == "grasp_2"
    assert "wrist_flip" not in rec_off.extras and len(s_off.calls) == 1
    # exclude applies to each finger assignment by name; grasp_2's twin is as easy to reach as the row, which is kept
    s_ex, draft_ex, rec_ex, _ = _reselect_stub(True, lambda q_: True)
    assert IS.IORSSmall.reselect_grasp(s_ex, draft_ex, rec_ex, None, "apple.n.01_1", exclude=["grasp_1", "grasp_1~r180"]) == "grasp_2"
    assert rec_ex.extras["wrist_flip"] is False and rec_ex.extras["twin_choice"]["reason"] == "default"
    # a drawn twin excluded (the descent retry's ``tried``): its row is re-selected alone and the draw's facts are dropped
    s_tw, draft_tw, rec_tw, _ = _reselect_stub(True, lambda q_: True)
    rec_tw.grasp_name, rec_tw.extras = "grasp_1~r180", {"wrist_flip": True, "twin_choice": {"stage": "phase3_target", "chosen": "twin"}}
    assert IS.IORSSmall.reselect_grasp(s_tw, draft_tw, rec_tw, None, "apple.n.01_1", exclude=["grasp_1~r180"]) == "grasp_1"
    assert rec_tw.extras["wrist_flip"] is False and "twin_choice" not in rec_tw.extras and len(s_tw.calls) == 1


def test_grasp_facts_reads_the_original_pinch_offset_through_the_alias():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    hull = th.tensor([[x, y, z] for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (0.0, 0.1)], dtype=th.float32)
    s = SimpleNamespace(hull={"target": hull}, rest={"target": C.IDENTITY_QUAT}, top=None, template_facts={"grasp_pinch_offset_m": {"grasp_4": 0.012}})
    for name in ("pinch_offset", "pinch_vs_long_axis_deg", "closing_tilt_deg", "pivot_arm"):
        setattr(s, name, (lambda f: (lambda pose: f(s, pose)))(getattr(IS.IORSSmall, name)))
    row_pose = (th.tensor([0.0, 0.012, 0.12]), C.eef_quat([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    row = SimpleNamespace(eef_in_object=row_pose, grasp_name="grasp_4", grasp=None)
    twin = SimpleNamespace(eef_in_object=C.twin_pose(row_pose), grasp_name="grasp_4~r180", grasp=None)
    f_row, f_twin = IS.IORSSmall.grasp_facts(s, row), IS.IORSSmall.grasp_facts(s, twin)
    assert f_row["pinch_offset_m"] == pytest.approx(0.012, abs=1e-4) and f_row["pinch_offset_original_m"] == pytest.approx(0.012)
    assert f_twin["pinch_offset_m"] == pytest.approx(-0.012, abs=1e-4) and f_twin["pinch_offset_original_m"] == pytest.approx(-0.012)
    for k in ("pinch_vs_long_axis_deg", "closing_tilt_deg", "pivot_arm_m"):
        assert f_twin[k] == pytest.approx(f_row[k], abs=1e-3), k
