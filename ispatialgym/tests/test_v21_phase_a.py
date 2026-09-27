"""first-demo v2.1 Phase A (owner findings on v2.0): each rule fires only in its situation, detected at run time.

A1 (cap 0927-040231): an attach place opens the fingers just enough when they would meet the partner before the part seats.
A2 (block design 0927-040243): a travel leg keeps an arm that holds the recorded hand pose in another IK branch when it stays clear
and the next contact tracks from there; the next contact then runs on the kept branch's configurations.
A3 (bar soap 0927-034001): a library pinch whose fingers would reach below the object's bottom is lifted; a support place of an
object that hangs lower in the hand than recorded is raised by the difference.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.sidecar import Pose
from ispatialgym.robot import conventions as C
from ispatialgym.tests.test_v2_oracle import FakeBackend, iors_world, make_executor


def _pose(xyz, quat=(0.0, 0.0, 0.0, 1.0)):
    return Pose(position=[float(v) for v in xyz], orientation=[float(v) for v in quat])


# ------------------------------------------------------------------------------------------------ A3


def _grasp_setup(bottom=0.80, top=0.86):
    obj = SimpleNamespace(aabb=(th.tensor([0.5, -0.05, bottom]), th.tensor([0.6, 0.05, top])))
    state = SimpleNamespace(object=lambda name: obj)
    action = SimpleNamespace(kind="library", object="soap", arm="right")
    grasp = _pose([0.55, 0.0, 0.83], C.quat_from_yaw(0.0).tolist())
    pre = _pose([0.55, 0.0, 0.93])
    targets = {"pre_grasp": pre, "grasp": grasp, "approach": [pre, grasp]}
    return action, state, targets


def test_a3_grasp_lifted_by_the_finger_deficit_below_the_object(monkeypatch):
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.794)  # 6 mm under the object's bottom
    ctx = {}
    out = EX.lift_grasp_off_support(action, state, targets, ctx, EX.ExecuteConfig())
    need = 0.80 + EX.ExecuteConfig().grasp_support_clearance_m - 0.794
    assert ctx["grasp_support_lift"]["applied"] and ctx["grasp_support_lift"]["lift_m"] == pytest.approx(need, abs=1e-4)
    for k in ("grasp", "pre_grasp"):
        assert out[k].position[2] == pytest.approx(targets[k].position[2] + need, abs=1e-6)
        assert out[k].position[:2] == pytest.approx(targets[k].position[:2])
    assert [p.position[2] for p in out["approach"]] == pytest.approx([p.position[2] + need for p in targets["approach"]])


def test_a3_grasp_kept_when_clear_or_when_the_pads_would_leave_the_object(monkeypatch):
    action, state, targets = _grasp_setup()
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.81)  # already 10 mm above the bottom
    ctx = {}
    assert EX.lift_grasp_off_support(action, state, targets, ctx, EX.ExecuteConfig()) is targets and not ctx["grasp_support_lift"]["applied"]
    action, state, targets = _grasp_setup(top=0.835)  # a 5 mm-tall part: a lift would take the pads off it
    monkeypatch.setattr(EX, "_finger_low_z", lambda pos, quat, arm: 0.79)
    ctx = {}
    assert EX.lift_grasp_off_support(action, state, targets, ctx, EX.ExecuteConfig()) is targets
    assert not ctx["grasp_support_lift"]["applied"] and "pads" in ctx["grasp_support_lift"]["reason"]
    ctx = {}
    assert EX.lift_grasp_off_support(SimpleNamespace(kind="cube", object="b", arm="right"), state, targets, ctx, EX.ExecuteConfig()) is targets
    assert ctx == {}


def _place_setup(hang_extra):
    """Recorded: the hand 5 cm above the object's origin; live: 5 cm + ``hang_extra`` (the object hangs lower in the hand)."""
    obj = SimpleNamespace(name="soap")
    eef_now = (th.tensor([0.4, 0.1, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj_now = (th.tensor([0.4, 0.1, 1.0 - 0.05 - hang_extra]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    state = SimpleNamespace(object=lambda n: obj, held_object=lambda arm: obj, eef_pose=lambda arm: eef_now, object_pose=lambda n: obj_now)
    g = SimpleNamespace(object="soap", arm="right", extra={"eef_in_object": {"position": [0.0, 0.0, 0.05], "orientation": [0.0, 0.0, 0.0, 1.0]}})
    sidecar = SimpleNamespace(grasps=[g])
    place, pre = _pose([0.7, 0.0, 0.88]), _pose([0.7, 0.0, 0.98])
    targets = {"pre_place": pre, "place": place, "descent": [pre, place], "retreat": [place, pre], "carry": [_pose([0.5, 0.0, 1.0]), pre]}
    return SimpleNamespace(kind="support", object="soap", arm="right"), state, sidecar, targets


def test_a3_place_raised_when_the_object_hangs_lower_in_the_hand():
    action, state, sidecar, targets = _place_setup(0.015)
    ctx = {}
    out = EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig())
    assert ctx["place_inhand_raise"]["applied"] and ctx["place_inhand_raise"]["raised_m"] == pytest.approx(0.015, abs=1e-4)
    for k in ("place", "pre_place"):
        assert out[k].position[2] == pytest.approx(targets[k].position[2] + 0.015, abs=1e-5)
    assert out["carry"][-1].position[2] == pytest.approx(targets["carry"][-1].position[2] + 0.015, abs=1e-5)
    assert out["carry"][0].position[2] < targets["carry"][0].position[2] + 0.015  # the carry ramps up to the raised pre-place


def test_a3_place_never_lowered_and_only_for_support_places():
    action, state, sidecar, targets = _place_setup(-0.010)  # higher in the hand: the recorded place stands
    ctx = {}
    assert EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig()) is targets and not ctx["place_inhand_raise"]["applied"]
    action = SimpleNamespace(kind="attach", object="soap", arm="right")
    ctx = {}
    assert EX.raise_place_for_inhand_pose(action, state, sidecar, targets, ctx, EX.ExecuteConfig()) is targets and ctx == {}


# ------------------------------------------------------------------------------------------------ A1


def _box(cx, cy, z0, z1, hw=0.002, hd=0.002):
    xs, ys, zs = np.meshgrid([cx - hw, cx + hw], [cy - hd, cy + hd], [z0, z1], indexing="ij")
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)


def test_a1_attach_place_opens_the_fingers_just_enough(monkeypatch):
    """Two finger boxes 26 mm either side of the eef axis reaching 20 mm below the eef, a partner disc of radius 27 mm whose top the
    fingertips would reach 5 mm before the seat: the fingers open by the smallest 0.5 mm step whose rays miss the disc."""
    fingers = [(_box(0.0, 0.026, -0.02, 0.0), np.array([0.0, 1.0, 0.0])), (_box(0.0, -0.026, -0.02, 0.0), np.array([0.0, -1.0, 0.0]))]
    monkeypatch.setattr(EX, "finger_surface_points", lambda arm, spacing: fingers)
    top, r_rim = 0.815, 0.027  # the seat puts the eef at z 0.83: fingertips at 0.81, 5 mm below the partner's top

    def raycast(starts, ends):
        out = []
        for s, e in zip(starts, ends):
            s, e = np.asarray(s), np.asarray(e)
            r = float(np.hypot(s[0], s[1]))
            if r <= r_rim and s[2] > top >= e[2]:
                out.append([{"hit": True, "distance": float(s[2] - top), "rigidBody": "/World/partner/base_link"}])
            else:
                out.append([{"hit": False}])
        return out

    monkeypatch.setattr(EX, "_raycast_all", raycast)
    partner = SimpleNamespace(prim_path="/World/partner", name="partner")
    part = SimpleNamespace(name="cap")
    robot = SimpleNamespace(gripper_control_idx={"right": th.tensor([0, 1])}, get_joint_positions=lambda: th.tensor([0.0, 0.0]))
    emitted = []
    motions = SimpleNamespace(robot=robot, state=SimpleNamespace(gripper_cmd={}, stop_base=lambda: None), cfg=SimpleNamespace(gripper_preshape_steps=3),
                              _gripper_limits=lambda arm: (0.0, 0.05), gripper_command=lambda arm, q: 2 * q / 0.05 - 1, _emit=lambda: emitted.append(1) or th.zeros(1))
    state = SimpleNamespace(object=lambda n: partner if n == "bottle" else part, held_object=lambda arm: part)
    action = SimpleNamespace(arm="right", on="bottle", object="cap")
    targets = {"pre_place": _pose([0.0, 0.0, 0.93]), "place": _pose([0.0, 0.0, 0.83])}
    ctx = {}
    list(EX.clear_fingers_over_partner(action, motions, state, targets, ctx, EX.ExecuteConfig()))
    rec = ctx["finger_clearance"]
    assert rec["before_seat_m"] == pytest.approx(0.005, abs=1e-4)
    # the inner corners stand at r = hypot(2, 24) mm: 3 mm (six steps) takes them past the 27 mm rim
    assert rec["applied"] and rec["opened_m"] == pytest.approx(0.003, abs=1e-6) and rec["before_seat_after_m"] is None
    assert motions.state.gripper_cmd["right"] == pytest.approx(2 * 0.003 / 0.05 - 1) and len(emitted) == 3 and rec["held"]


def test_a1_attach_place_leaves_clear_fingers_alone(monkeypatch):
    fingers = [(_box(0.0, 0.030, -0.02, 0.0), np.array([0.0, 1.0, 0.0])), (_box(0.0, -0.030, -0.02, 0.0), np.array([0.0, -1.0, 0.0]))]
    monkeypatch.setattr(EX, "finger_surface_points", lambda arm, spacing: fingers)
    monkeypatch.setattr(EX, "_raycast_all", lambda starts, ends: [[{"hit": False}] for _ in starts])
    partner = SimpleNamespace(prim_path="/World/partner", name="partner")
    robot = SimpleNamespace(gripper_control_idx={"right": th.tensor([0, 1])}, get_joint_positions=lambda: th.tensor([0.0, 0.0]))
    motions = SimpleNamespace(robot=robot, state=SimpleNamespace(gripper_cmd={}, stop_base=lambda: None), cfg=SimpleNamespace(gripper_preshape_steps=3),
                              _gripper_limits=lambda arm: (0.0, 0.05), gripper_command=lambda arm, q: 0.0, _emit=lambda: th.zeros(1))
    state = SimpleNamespace(object=lambda n: partner, held_object=lambda arm: None)
    ctx = {}
    list(EX.clear_fingers_over_partner(SimpleNamespace(arm="right", on="bottle", object="cap"), motions, state,
                                        {"pre_place": _pose([0.0, 0.0, 0.93]), "place": _pose([0.0, 0.0, 0.83])}, ctx, EX.ExecuteConfig()))
    assert not ctx["finger_clearance"]["applied"] and ctx["finger_clearance"]["before_seat_m"] is None and motions.state.gripper_cmd == {}
