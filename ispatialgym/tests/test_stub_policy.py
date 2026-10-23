"""The stub policies of ``run_isg --mode policy`` smoke tests (``ispatialgym.eval.stub_policy``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.eval.stub_policy import DemoPolicy, HoldPolicy
from ispatialgym.tests.test_v2_oracle import FakeRobot


def test_demo_policy_plays_the_demo_back_then_ends(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    from ispatialgym.oracle import run_isg as R

    path = tmp_path / "demo.hdf5"
    acts = np.arange(3 * 23, dtype=np.float32).reshape(3, 23)
    with h5py.File(path, "w") as f:
        f.create_dataset("data/demo_0/action", data=acts)
    monkeypatch.setenv("ISG_STUB_DEMO", str(path))
    policy = R.load_policy("ispatialgym.eval.stub_policy:DemoPolicy")
    assert isinstance(policy, DemoPolicy)
    meta = {}
    policy.bind(executor=None, loader=None, meta=meta, out_dir=None, tro_path=None)
    out = [policy(None, None) for _ in range(4)]
    assert [o.tolist() for o in out[:3]] == acts.tolist() and out[3] is None
    assert meta["stub_policy"] == {"policy": "DemoPolicy", "demo": str(path), "n_actions": 3}
    monkeypatch.delenv("ISG_STUB_DEMO")
    with pytest.raises(ValueError, match="needs a demo"):
        DemoPolicy().bind(loader=None)


def test_hold_policy_holds_the_start_configuration(monkeypatch):
    from ispatialgym.eval.policy import HoldPolicy as EvalHoldPolicy, import_policy, make_policy

    monkeypatch.setenv("ISG_STUB_HOLD_STEPS", "2")
    robot = FakeRobot()
    policy = HoldPolicy()
    policy.bind(executor=SimpleNamespace(robot=robot))
    a = policy(None, None)
    idx = robot.controller_action_idx
    assert th.equal(a[idx["arm_right"]], robot.q[robot.arm_control_idx["right"]]) and th.equal(a[idx["trunk"]], robot.q[robot.trunk_control_idx])
    assert th.equal(a[idx["base"]], th.zeros(3)) and float(a[idx["gripper_left"]]) == float(a[idx["gripper_right"]]) == 1.0
    assert policy(None, None) is not None and policy(None, None) is None

    # with loader.hold (preserved controller_targets and smooth gripper command)
    hold_vec = th.arange(23, dtype=th.float32)
    loader = SimpleNamespace(hold=SimpleNamespace(action=lambda: hold_vec.clone()))
    ep = make_policy(import_policy("ispatialgym.eval.policy:HoldPolicy"), {"steps": 1})
    assert isinstance(ep, EvalHoldPolicy)
    ep.bind(loader=loader)
    assert th.equal(ep(None, None), hold_vec) and ep(None, None) is None
