"""Held-object sampling across instances (2026-10-11 multi-instance stuck attaches): the assisted-grasp weld of one instance must
not outlive it. ``HeldObjectV2.check_goal_feasible`` teleports the robot to the goal configuration (fingers at the open limit,
``FINGER_OPEN``) with the object welded and steps physics: the stock release deleted the weld in that step, the restore
(``Robot._load_state``) re-created it with the release window still running, the window then cleared ``_ag_obj_in_hand``
while the joint stayed, and ``release`` (which asked for the object in hand) left the object welded to the hand for the next
instances. The real OmniGibson ``Robot`` assisted-grasp methods run on a stand-in robot (``ControllerView``, the joint
creation / deletion and ``og.sim`` replaced); a physics step is ``Robot.post_step``."""
from types import SimpleNamespace

import pytest
import torch as th

import omnigibson as og
from omnigibson.robots import robot as R

from ispatialgym.sampling.v2.tasks import held_object as HO

DT = 1 / 120.0
ARM = "right"
OPEN, CLOSED = th.tensor([0.05, 0.05]), th.tensor([0.02, 0.02])


class _Ctrl:
    """ControllerView stand-in: one gripper group whose control the test sets per physics step."""

    control = CLOSED

    @staticmethod
    def get_dof_idx(group_key):
        return th.tensor([0, 1])

    @classmethod
    def get_control(cls, group_key, controller_idx):
        return cls.control

    @staticmethod
    def get_control_type(group_key):
        return R.ControlType.POSITION

    @staticmethod
    def dump_state(group_key, controller_idx):
        return {}

    @staticmethod
    def load_state(group_key, controller_idx, state):
        pass


@pytest.fixture
def robot(monkeypatch):
    calls = []
    monkeypatch.setattr(og, "sim", SimpleNamespace(get_physics_dt=lambda: DT, update_handles=lambda: calls.append("update_handles")), raising=False)
    monkeypatch.setattr(R, "ControllerView", _Ctrl)
    monkeypatch.setattr(R, "delete_or_deactivate_prim", lambda path: calls.append(("delete", path)))
    monkeypatch.setattr(R, "create_joint", lambda prim_path, **kw: calls.append(("create", prim_path)) or SimpleNamespace(GetPath=lambda: SimpleNamespace(pathString=prim_path)))
    # the base object's state (poses, joints) is not part of these tests
    monkeypatch.setattr(R.USDObject, "_dump_state", lambda self: {})
    monkeypatch.setattr(R.USDObject, "_load_state", lambda self, state: None)
    target = SimpleNamespace(name="target_mug_0", links={"base_link": SimpleNamespace(prim_path="/World/scene_0/target_mug_0/base_link")}, wake=lambda: None)
    attrs = {"is_manipulation": True, "grasping_mode": "assisted", "arm_names": [ARM], "default_arm": ARM, "joint_upper_limits": OPEN.clone(),
             "eef_links": {ARM: SimpleNamespace(prim_path="/World/scene_0/robot_r1/right_eef_link")},
             "scene": SimpleNamespace(object_registry=lambda key, value, default=None: target if value == target.name else default)}
    cls = type("HeldAgWeldTestRobot", (R.Robot,), attrs)
    r = object.__new__(cls)
    r._controllers = {f"gripper_{ARM}": ("gripper", 0)}
    r._grasping_direction = "lower"
    r._control_enabled, r._articulation_view, r._disable_grasp_handling = True, object(), False
    for name in ("_ag_obj_in_hand", "_ag_obj_constraints", "_ag_obj_constraint_params", "_ag_release_counter", "_ag_grasp_counter", "_ag_release_pending"):
        setattr(r, name, {ARM: None})
    r._calculate_in_hand_object = lambda arm="default": None  # no new grasp window: the attaches are explicit
    yield r, target, calls
    R.Robot._cls_registry.pop("HeldAgWeldTestRobot", None)


def _params(target, k):
    return {"target_obj": target, "target_link_name": "base_link", "parent_frame_pos": th.tensor([0.0, 0.0, 0.01 * k]),
            "parent_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]), "child_frame_pos": th.zeros(3), "child_frame_orn": th.tensor([0.0, 0.0, 0.0, 1.0]),
            "joint_type": "FixedJoint"}


def _step(r, control=CLOSED, n=1):
    _Ctrl.control = control
    for _ in range(n):
        r.post_step()


def _welded(r):
    return r._ag_obj_constraints[ARM] is not None


def _instance(r, target, k, fixed):
    """One written instance as the sampler's AG sees it: attach -> goal check (snapshot, one physics step at the goal
    configuration's open fingers, restore) -> settle -> write -> release. Returns the release counter after the restore."""
    r._create_assisted_grasp_joint(ARM, _params(target, k))  # establish_grasp: the grasp window's joint
    _step(r, n=3)
    state = r._dump_state()
    if fixed:
        with HO.assisted_grasp_paused(r):
            _step(r, OPEN)
    else:
        _step(r, OPEN)
    r._load_state(state)  # world.restore
    counter = r._ag_release_counter[ARM]
    _step(r, n=40)  # settle_start: the gripper held closed on the object
    if fixed:
        HO.drop_weld(r, ARM)
    elif r._ag_obj_in_hand.get(ARM) is not None:  # the release before the fix
        r.release_grasp_immediately(ARM)
    _step(r, OPEN, n=10)  # release: the gripper opens
    return counter


def test_before_the_fix_the_goal_check_releases_the_weld_and_the_restore_brings_it_back(robot):
    r, target, calls = robot
    counter = _instance(r, target, 0, fixed=False)
    path = "/World/scene_0/robot_r1/right_eef_link/ag_constraint"
    joint_ops = [c for c in calls if isinstance(c, tuple)]
    assert joint_ops[:3] == [("create", path), ("delete", path), ("create", path)]  # the attach, the release in the step, the restore
    if counter is not None:
        # stock Robot._load_state keeps the release window running: the robot forgets the object while the joint stays, and
        # the release that asks for the object in hand leaves it welded to the hand for the next instance's attach
        assert _welded(r) and r._ag_obj_in_hand[ARM] is None and r._ag_release_counter[ARM] is None
    else:
        assert not _welded(r)  # a _load_state that ends the release window: the object stays in hand and is released


def _forgotten_weld(r, target):
    """What the goal check left before the fix (stock OmniGibson): the joint in place, nothing in hand, no release window."""
    r._create_assisted_grasp_joint(ARM, _params(target, 0))
    r._ag_obj_in_hand[ARM] = None


def test_every_instance_starts_and_ends_without_a_weld(robot):
    r, target, calls = robot
    for k in range(6):
        assert not _welded(r) and r._ag_obj_in_hand[ARM] is None, f"instance {k} starts welded"
        _instance(r, target, k, fixed=True)
        assert not _welded(r) and r._ag_obj_in_hand[ARM] is None and r._ag_release_counter[ARM] is None and r._ag_grasp_counter[ARM] is None
    # one joint per instance (the attach): the paused goal check neither released nor re-created it
    creates = [c for c in calls if isinstance(c, tuple) and c[0] == "create"]
    deletes = [c for c in calls if isinstance(c, tuple) and c[0] == "delete"]
    assert len(creates) == 6 and len(deletes) == 6


def test_the_paused_goal_check_keeps_the_weld_and_the_restore_is_a_no_op(robot):
    r, target, calls = robot
    r._create_assisted_grasp_joint(ARM, _params(target, 1))
    joint = r._ag_obj_constraints[ARM]
    state = r._dump_state()
    with HO.assisted_grasp_paused(r):
        _step(r, OPEN)
        assert r._ag_obj_constraints[ARM] is joint and r._ag_obj_in_hand[ARM] is target
    assert r._disable_grasp_handling is False
    r._load_state(state)
    assert r._ag_obj_constraints[ARM] is joint and r._ag_release_counter[ARM] is None
    _step(r, n=10)
    assert r._ag_obj_in_hand[ARM] is target
    with pytest.raises(RuntimeError):
        with HO.assisted_grasp_paused(r):
            raise RuntimeError("a rejection inside the goal check")
    assert r._disable_grasp_handling is False


def test_drop_weld_removes_a_forgotten_weld_and_is_a_no_op_without_one(robot):
    r, target, calls = robot
    _forgotten_weld(r, target)
    HO.drop_weld(r, ARM)
    assert not _welded(r) and r._ag_obj_constraint_params[ARM] is None and r._ag_obj_in_hand[ARM] is None
    assert calls[-2:] == [("delete", "/World/scene_0/robot_r1/right_eef_link/ag_constraint"), "update_handles"]
    n = len(calls)
    HO.drop_weld(r, ARM)
    assert len(calls) == n
