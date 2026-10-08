"""The start-state fingerprint of a loaded instance and its comparison (engineering track, ``oracle/start_state.py``); stubs, no
simulator."""

from types import SimpleNamespace

import torch as th

from ispatialgym.oracle import start_state as S


class _Ent:
    def __init__(self, pos, quat, joints=None):
        self.pos, self.quat, self.joints = th.tensor(pos), th.tensor(quat), None if joints is None else th.tensor(joints)
        self.n_joints = 0 if joints is None else len(joints)

    def get_position_orientation(self):
        return self.pos, self.quat

    def get_joint_positions(self):
        return self.joints


def _loader(dx=0.0, q=0.3, ag="apple.n.01_1", extra_obj=False):
    robot = SimpleNamespace(joints={"a": 0, "b": 1}, get_position_orientation=lambda: (th.tensor([1.0 + dx, 2.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])),
                            get_joint_positions=lambda: th.tensor([0.1, q]), get_joint_velocities=lambda: th.tensor([0.0, 0.0]),
                            controllers={"arm_right": SimpleNamespace(_goal={"target": th.tensor([0.1, q])})},
                            _ag_obj_in_hand={"right": SimpleNamespace(name=ag) if ag else None, "left": None},
                            _ag_obj_constraint_params={"right": {"obj_link_offset": th.tensor([0.0, 0.0, 0.01])}}, obs_modalities=["proprio", "rgb"])
    physical = {"apple.n.01_1": _Ent([0.5, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]), "laptop.n.01_1": _Ent([0.7, 0.2, 0.8], [0.0, 0.0, 0.0, 1.0], [0.5])}
    if extra_obj:
        physical["plate.n.01_1"] = _Ent([0.6, 0.1, 0.8], [0.0, 0.0, 0.0, 1.0])
    return SimpleNamespace(robot=robot, physical=physical, hinges=[], report={"constraints": [{"type": "mass", "object": "tray.n.01_1", "ok": True}]},
                           grasps=[{"arm": "right", "object": "apple.n.01_1"}])


def test_identical_loads_fingerprint_equal_and_compare_ok():
    a, b = S.fingerprint(_loader(), render=False), S.fingerprint(_loader(), render=False)
    assert a["sha256"] == b["sha256"] and a["values"]["objects"]["laptop.n.01_1"]["joints"] == [0.5]
    rep = S.compare(a, b)
    assert rep["ok"] and rep["identical"] and rep["fails"] == []


def test_differences_within_tolerance_pass_beyond_fail():
    a = S.fingerprint(_loader(), render=False)
    near = S.fingerprint(_loader(dx=5e-5), render=False)
    rep = S.compare(a, near)
    assert rep["ok"] and not rep["identical"] and abs(rep["max"]["pos_m"] - 5e-5) < 1e-6
    far = S.fingerprint(_loader(dx=1e-3, q=0.31), render=False)
    rep2 = S.compare(a, far)
    assert not rep2["ok"] and any("robot base" in f for f in rep2["fails"]) and any("robot joints" in f for f in rep2["fails"])


def test_quantized_quaternions_compare_by_rotation_angle():
    import math

    q = [-6e-05, 5.2e-05, 0.706203, 0.708009]
    assert S._quat_angle(q, q) < 1e-9 and S._quat_angle(q, [-x for x in q]) < 1e-9
    h = 0.5e-3
    turned = [0.0, 0.0, math.sin(h), math.cos(h)]
    assert abs(S._quat_angle([0.0, 0.0, 0.0, 1.0], turned) - 1e-3) < 1e-9
    assert abs(S._quat_angle([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]) - math.pi) < 1e-9


def test_sim_objects_serialize_by_name_not_address():
    class Obj:
        def __init__(self, name):
            self.name = name

    a, b = _loader(), _loader()
    a.robot._ag_obj_constraint_params = {"right": {"target_obj": Obj("apple.n.01_1")}}
    b.robot._ag_obj_constraint_params = {"right": {"target_obj": Obj("apple.n.01_1")}}
    fa, fb = S.fingerprint(a, render=False), S.fingerprint(b, render=False)
    assert fa["values"]["robot"]["assisted_grasp_params"]["right"]["target_obj"] == "<Obj apple.n.01_1>"
    assert fa["sha256"] == fb["sha256"]


def test_discrete_differences_fail():
    a = S.fingerprint(_loader(), render=False)
    rep = S.compare(a, S.fingerprint(_loader(ag=None), render=False))
    assert not rep["ok"] and any("assisted_grasp" in f for f in rep["fails"])
    rep2 = S.compare(a, S.fingerprint(_loader(extra_obj=True), render=False))
    assert not rep2["ok"] and any("objects differ" in f for f in rep2["fails"])
