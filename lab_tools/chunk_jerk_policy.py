"""Instrumented policies for the chunk-boundary jerk diagnostic (lab/diagnostics/chunk-jerk, 2026-10-08).

Diagnostic-only code: the production bridge (``act_policy.py`` / ``replay_policy.py``) is untouched. Both callables record,
at every call (= before the action of that step is applied, i.e. the observation the policy sees):

* ``proprio``: the raw 61-D ``robot_r1::proprio`` the policy receives;
* ``truth``: the simulator's own state at the same moment, read from the robot object (not from the observation): base x, y,
  yaw, the 4 torso joint positions, the 14 arm joint positions, the measured joint velocities of the torso;
* ``action``: the 23-D action returned (sent to the executor);
* ``chunk_pos``: the step's position in the action chunk (0 = a re-plan / new chunk; default execution re-plans every
  ``action_horizon`` steps, temporal aggregation every step);
* ``t_wall``: wall time of the call.

A/B modes (item 4) through ``CONTEXT``: ``proprio_mode = "commanded"`` replaces the torso and arm joint positions of the proprio
sent to the server by the last commanded targets (the dataset's semantics: the joint reading equals the previous command);
``render_before_obs = True`` renders once more and re-reads the cameras before packing the images (fresh frame of the current
physics state).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np

from lab_tools import act_policy as AP
from lab_tools import replay_policy as RP

TORSO_IDX = slice(53, 57)  # raw proprio: trunk_qpos
ARM_L_IDX = slice(3, 10)  # arm_left_qpos
ARM_R_IDX = slice(28, 35)  # arm_right_qpos
ACT_TORSO = slice(3, 7)  # 23-D action: trunk targets
ACT_ARM_L = slice(7, 14)
ACT_ARM_R = slice(15, 22)


def _robot():
    import omnigibson as og

    return og.sim.scenes[0].robots[0]


def _yaw(q):
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def truth_snapshot(robot) -> Dict[str, Any]:
    pos, quat = robot.get_position_orientation()
    pos, quat = pos.detach().cpu().numpy(), quat.detach().cpu().numpy()
    q = robot.get_joint_positions().detach().cpu().numpy()
    qd = robot.get_joint_velocities().detach().cpu().numpy()
    tr = np.asarray(robot.trunk_control_idx)
    arms = {a: np.asarray(robot.arm_control_idx[a]) for a in robot.arm_names}
    return {"base_xyyaw": [float(pos[0]), float(pos[1]), _yaw(quat)], "torso_q": [float(v) for v in q[tr]], "torso_qd": [float(v) for v in qd[tr]],
            "arm_q": {a: [float(v) for v in q[idx]] for a, idx in arms.items()}}


class _Recorder:
    def _init_log(self, ctx):
        ctx["jerk_log"] = {"proprio": [], "action": [], "truth": [], "chunk_pos": [], "t_wall": [], "proprio_sent": []}
        self._last_cmd = None

    def _record(self, ctx, proprio, truth, chunk_pos, proprio_sent=None):
        lg = ctx["jerk_log"]
        lg["proprio"].append(np.asarray(proprio, dtype=np.float32).copy())
        lg["truth"].append(truth)
        lg["chunk_pos"].append(int(chunk_pos))
        lg["t_wall"].append(time.time())
        lg["proprio_sent"].append(None if proprio_sent is None else np.asarray(proprio_sent, dtype=np.float32).copy())

    def _record_action(self, ctx, action):
        ctx["jerk_log"]["action"].append(None if action is None else np.asarray(action, dtype=np.float32).copy())


class InstrumentedActPolicy(AP.ActPolicy, _Recorder):
    """ActPolicy with per-step logging and the two A/B switches."""

    def begin_episode(self, ctx):
        super().begin_episode(ctx)
        self._init_log(ctx)
        self._robot_obj = _robot()
        self._env = ctx.get("env")

    def __call__(self, obs, info):
        ctx = AP.CONTEXT
        if ctx is not self.episode_ctx:
            self.begin_episode(ctx)
        truth = truth_snapshot(self._robot_obj)
        name = self.robot_name or ctx.get("robot") or next(iter(k for k in obs if isinstance(obs[k], dict) and "proprio" in obs[k]))
        raw = AP._np(obs[name]["proprio"]).astype(np.float32).reshape(-1)
        ah = int((ctx.get("server_metadata") or {}).get("action_horizon") or 16)
        chunk_pos = self.steps % ah
        sent = raw
        if ctx.get("proprio_mode") == "commanded" and self._last_cmd is not None:
            sent = raw.copy()
            sent[TORSO_IDX] = self._last_cmd[ACT_TORSO]
            sent[ARM_L_IDX] = self._last_cmd[ACT_ARM_L]
            sent[ARM_R_IDX] = self._last_cmd[ACT_ARM_R]
            obs = dict(obs)
            robot_obs = dict(obs[name])
            robot_obs["proprio"] = sent
            obs[name] = robot_obs
        env = self._env or (ctx.get("shared") or {}).get("env")
        if ctx.get("render_before_obs") and env is not None:
            self._env = env
            import omnigibson as og

            from ispatialgym import single_env as SE

            og.sim.render()
            fresh, _ = SE.get_obs(env)
            robot_obs = dict(obs[name])
            for k, v in fresh[name].items():
                if k != "proprio":
                    robot_obs[k] = v
            obs = dict(obs)
            obs[name] = robot_obs
        self._record(ctx, raw, truth, chunk_pos, proprio_sent=sent if ctx.get("proprio_mode") == "commanded" else None)
        action = super().__call__(obs, info)
        self._record_action(ctx, action)
        if action is not None:
            self._last_cmd = np.asarray(action, dtype=np.float32).copy()
        return action


class InstrumentedReplayPolicy(RP.ReplayPolicy, _Recorder):
    """ReplayPolicy (recorded actions) with the same per-step logging (chunk_pos = -1: no chunks)."""

    def begin_episode(self, ctx):
        super().begin_episode(ctx)
        self._init_log(ctx)
        self._robot_obj = _robot()

    def __call__(self, obs, info):
        ctx = RP.CONTEXT
        if ctx is not self.episode_ctx:
            self.begin_episode(ctx)
        truth = truth_snapshot(self._robot_obj)
        pr = obs[self.robot.name]["proprio"]
        raw = (pr.detach().cpu().numpy() if hasattr(pr, "detach") else np.asarray(pr)).astype(np.float32).reshape(-1)
        self._record(ctx, raw, truth, -1)
        action = super().__call__(obs, info)
        self._record_action(ctx, action)
        return action
