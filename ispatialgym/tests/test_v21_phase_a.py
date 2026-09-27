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
