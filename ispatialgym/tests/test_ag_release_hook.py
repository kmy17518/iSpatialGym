"""Our Behavior-1K fork's pad-clear release (owner 2026-10-10): once the gripper's control stops applying grasp with an object in
hand, the assisted-grasp weld stays while both pads still hold the object (``_calculate_in_hand_object``), at most
``ag_release_max_delay_s``; then ``_release_grasp`` (release callbacks, wake). Off by default: the stock release in the open
command's own physics step. ``release_grasp_immediately`` bypasses it; a pending release travels with the robot's state.
The real ``Robot`` methods run on a stand-in robot (``ControllerView``, the prim deletion and ``og.sim`` replaced)."""
import math
from types import MethodType, SimpleNamespace

import pytest
import torch as th

import omnigibson as og
from omnigibson.robots import robot as R

DT = 0.01


class _Ctrl:
    """ControllerView stand-in: one gripper group (two finger dofs) whose control the test sets per physics step."""

    control = None

    @staticmethod
    def get_dof_idx(group_key):
        return th.tensor([0, 1])

    @classmethod
    def get_control(cls, group_key, controller_idx):
        return cls.control

    @staticmethod
    def get_control_type(group_key):
        return R.ControlType.POSITION


def _robot(monkeypatch, on=True, cap=0.2):
    monkeypatch.setattr(og, "sim", SimpleNamespace(get_physics_dt=lambda: DT, update_handles=lambda: None), raising=False)
    monkeypatch.setattr(R, "ControllerView", _Ctrl)
    deleted = []
    monkeypatch.setattr(R, "delete_or_deactivate_prim", lambda path: deleted.append(path))
    obj = SimpleNamespace(name="cap_0", woken=0)
    obj.wake = lambda: setattr(obj, "woken", obj.woken + 1)
    calls = []
    pads = {"touch": True}
    r = SimpleNamespace(
        is_manipulation=True, arm_names=["right"], default_arm="right", _controllers={"gripper_right": ("gripper", 0)},
        joint_upper_limits=th.tensor([0.05, 0.05]), _grasping_direction="lower", _ag_obj_in_hand={"right": obj},
        _ag_obj_constraints={"right": SimpleNamespace(GetPath=lambda: SimpleNamespace(pathString="/World/robot/ag_joint"))},
        _ag_obj_constraint_params={"right": {"target_obj": obj}}, _ag_release_counter={"right": None}, _ag_grasp_counter={"right": None},
        _ag_release_pending={"right": None}, ag_release_log=[],
        release_callbacks=[lambda robot, arm, released: calls.append((arm, released.name, robot._ag_obj_constraints[arm]))],
    )
    if on:
        r.ag_release_on_pad_clear, r.ag_release_max_delay_s = True, cap
    r._calculate_in_hand_object = lambda arm="default": (obj, "base_link") if pads["touch"] else None
    for name in ("_handle_assisted_grasping", "_ag_keep_weld", "_release_grasp", "_handle_release_window", "release_grasp_immediately",
                 "_dump_ag_release_pending", "_load_ag_release_pending"):
        setattr(r, name, MethodType(getattr(R.Robot, name), r))
    return r, obj, calls, pads, deleted


def _step(r, grasp=False):
    """One physics step of the assisted-grasp logic; the control at the open limit (the +1 command) unless ``grasp``."""
    _Ctrl.control = th.tensor([0.02, 0.02]) if grasp else th.tensor([0.05, 0.05])
    r._handle_assisted_grasping()


def test_the_weld_outlives_the_open_command_while_both_pads_touch_and_goes_when_a_pad_leaves(monkeypatch):
    r, obj, calls, pads, deleted = _robot(monkeypatch)
    for k in range(5):
        _step(r)
        assert r._ag_obj_constraints["right"] is not None and not calls and not deleted and r._ag_release_pending["right"] == k
    pads["touch"] = False
    _step(r)
    # released in this step: the joint deleted, the callbacks see it gone (the real removal), the object woken
    assert deleted == ["/World/robot/ag_joint"] and calls == [("right", "cap_0", None)] and obj.woken == 1
    assert r._ag_release_pending["right"] is None
    assert r.ag_release_log == [{"arm": "right", "object": "cap_0", "by": "pads_clear", "delay_steps": 5, "delay_s": pytest.approx(5 * DT)}]
    for _ in range(int(math.ceil(R.m.RELEASE_WINDOW / DT))):  # the registry clears after OmniGibson's release window, as before
        _step(r)
    assert r._ag_obj_in_hand["right"] is None and len(calls) == 1


def test_blocked_pads_are_let_go_at_the_cap(monkeypatch):
    cap = 0.05
    r, obj, calls, pads, deleted = _robot(monkeypatch, cap=cap)
    n = 0
    while r._ag_obj_constraints["right"] is not None:
        _step(r)
        n += 1
        assert n < 100
    expected = next(s for s in range(100) if s * DT >= cap)
    assert n == expected + 1 and calls == [("right", "cap_0", None)]
    assert r.ag_release_log[-1]["by"] == "cap" and r.ag_release_log[-1]["delay_steps"] == expected


def test_the_hook_off_is_the_stock_release(monkeypatch):
    r, obj, calls, pads, deleted = _robot(monkeypatch, on=False)
    _step(r)
    assert r._ag_obj_constraints["right"] is None and calls == [("right", "cap_0", None)] and r.ag_release_log == []
    assert r._ag_release_pending["right"] is None


def test_grasp_reapplied_keeps_the_weld_and_an_immediate_release_bypasses_the_delay(monkeypatch):
    r, obj, calls, pads, deleted = _robot(monkeypatch)
    _step(r)
    _step(r, grasp=True)
    assert r._ag_release_pending["right"] is None and r._ag_obj_constraints["right"] is not None and not calls
    _step(r)
    assert r._ag_release_pending["right"] == 0
    r.release_grasp_immediately("right")
    assert r._ag_obj_constraints["right"] is None and r._ag_obj_in_hand["right"] is None and r._ag_release_pending["right"] is None
    assert calls == [("right", "cap_0", None)] and r.ag_release_log == []


def test_a_pending_release_travels_with_the_state(monkeypatch):
    r, obj, calls, pads, deleted = _robot(monkeypatch)
    for _ in range(3):
        _step(r)
    state = {}
    r._dump_ag_release_pending(state)
    assert state == {"ag_release_pending": {"right": 2}}
    r._load_ag_release_pending({})  # a state without a pending release cancels the one pending here
    assert r._ag_release_pending["right"] is None
    r._load_ag_release_pending(state)
    assert r._ag_release_pending["right"] == 2
    _step(r)
    assert r._ag_release_pending["right"] == 3  # the delay goes on from the restored count
    nothing = {}
    r._ag_release_pending["right"] = None
    r._dump_ag_release_pending(nothing)
    assert nothing == {}  # a stock state is unchanged
    # serialized: one block per pending arm, after which the next object's values follow untouched
    blocks = R._serialize_ag_release_pending(["left", "right"], {"right": 7})
    flat = th.cat([th.tensor([1.0, 2.0])] + blocks + [th.tensor([42.0])])
    assert R._deserialize_ag_release_pending(["left", "right"], flat, 2) == ({"right": 7}, 5) and flat[5].item() == 42.0
    assert R._deserialize_ag_release_pending(["left", "right"], th.tensor([1.0, 42.0]), 1) == ({}, 1)
    assert R._serialize_ag_release_pending(["left", "right"], {}) == []
    assert th.tensor([R._AG_PENDING_MAGIC]).item() == R._AG_PENDING_MAGIC and R._AG_PENDING_MAGIC != R._AG_MAGIC  # exact in float32
