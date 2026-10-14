"""Stub policies for ``run_isg --mode policy`` without a learned model (smoke tests of the policy path: streaming, ``--qa``
logging, ``--record-view``):

* ``--policy ispatialgym.eval.stub_policy:DemoPolicy`` plays the instance's own demo back, one action per call (an eval
  dataset instance's ``demo.hdf5``, or ``ISG_STUB_DEMO=<demo.hdf5>``), then ends the stream; the demo's settle rows are
  marked as such for the QA metrics, as in a replay;
* ``--policy ispatialgym.eval.stub_policy:HoldPolicy`` holds the robot where it starts (arms and trunk at their joints, base
  still, grippers open) for ``ISG_STUB_HOLD_STEPS`` steps (default 150).
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch as th


class DemoPolicy:
    def __init__(self):
        self.actions: Optional[List[th.Tensor]] = None
        self.i = 0

    def bind(self, executor=None, loader=None, meta=None, out_dir=None, tro_path=None):
        from ispatialgym.oracle import run_isg as R

        path = os.environ.get("ISG_STUB_DEMO")
        if not path and loader is not None and loader.tmeta.get("_paths"):
            rec = R._dataset_record(loader)
            path = rec.demo_path() if rec is not None else None
        if not path:
            raise ValueError("DemoPolicy needs a demo: an eval dataset instance, or ISG_STUB_DEMO=<demo.hdf5>")
        self.actions = list(R.replay_actions(path))
        if executor is not None:
            executor.stream_phases = R.replay_phases(path)
        if meta is not None:
            meta["stub_policy"] = {"policy": "DemoPolicy", "demo": path, "n_actions": len(self.actions)}

    def __call__(self, obs, info):
        if self.actions is None or self.i >= len(self.actions):
            return None
        self.i += 1
        return self.actions[self.i - 1]


class HoldPolicy:
    def __init__(self):
        self.n = int(os.environ.get("ISG_STUB_HOLD_STEPS", "150"))
        self.action: Optional[th.Tensor] = None
        self.i = 0

    def bind(self, executor=None, loader=None, meta=None, out_dir=None, tro_path=None):
        from ispatialgym.oracle import r1pro_model as M

        robot = executor.robot
        q = th.as_tensor(robot.get_joint_positions(), dtype=th.float32).reshape(-1)
        idx = robot.controller_action_idx
        a = th.zeros(M.ACTION_DIM)
        a[idx["trunk"]] = q[robot.trunk_control_idx]
        for arm in M.ARMS:
            a[idx[f"arm_{arm}"]] = q[robot.arm_control_idx[arm]]
            a[idx[f"gripper_{arm}"]] = 1.0
        self.action = a
        if meta is not None:
            meta["stub_policy"] = {"policy": "HoldPolicy", "n_steps": self.n}

    def __call__(self, obs, info):
        if self.action is None or self.i >= self.n:
            return None
        self.i += 1
        return self.action.clone()
