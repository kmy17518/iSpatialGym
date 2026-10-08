"""Gate 3a of the vectorized evaluator: ``VecStreamExecutor`` on N fake environments (no simulator).

Every environment runs its own scripted stream against the single-env fake world of ``test_v2_oracle`` (toy kinematics,
30 Hz); the per-env goal tracker mimics ``ISpatialGymTask``'s two-level info (``first_success_step`` / ``binary_success_ever``
/ gating pose errors / conditions). Checked:

* per-env success tracking and ``first_success_step``;
* the epilogue windows and verdicts per env (success after max(50, 20 %), the never-held 10-step window -> ``tolerance`` with
  ``never_reached``, goal held then lost -> second window -> ``tolerance`` at the last step, a timeout verdict without a window);
* environments finishing at different steps, one never reaching the goal; a finished env's robot receives its no-op (last
  action, base stopped) and does not move while the others continue;
* parity: every slot's result equals the single-env ``Executor.run_stream`` run of the same scenario alone (same n_steps,
  n_action_steps, reason, settle windows, first_success_step, final robot pose), so the batch is N independent episodes.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import AGENT_INSTANCE, Executor, ExecutorConfig
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
from ispatialgym.tests.test_v2_oracle import CAM_KEY, FakeEnv, FakeRobot

AGENT = "agent.n.01_1"


# ---------------------------------------------------------------------------------------------------------------- fakes


class GoalTracker:
    """``ISpatialGymTask``'s per-env evaluation on the fake: the goal is the robot base within 5 cm of ``goal_xy``."""

    def __init__(self, goal_xy, tol: float = 0.05):
        self.goal = th.tensor(goal_xy, dtype=th.float32)
        self.tol = tol
        self.first = -1
        self.errors_at_first = None

    def __call__(self, env: FakeEnv) -> Dict[str, Any]:
        pos, _ = env.robot.get_position_orientation()
        err = float(th.linalg.norm(pos[:2] - self.goal))
        within = err <= self.tol
        errors = {"pose_errors": {AGENT: {"position_error": err, "orientation_error": 0.0, "within_tolerance": within, "gating": True}},
                  "layout_errors": {}, "max_position_error": err, "max_orientation_error": 0.0, "max_layout_position_error": None, "max_layout_orientation_error": None}
        if within and self.first < 0:
            self.first = env._current_step
            self.errors_at_first = copy.deepcopy(errors)
        return {"binary_success": within, "binary_success_ever": self.first >= 0, "first_success_step": self.first, "official_success": self.first >= 0,
                "settled": False, "settled_success": False, "first_settled_success_step": -1, "settle_latency_steps": -1,
                "errors": errors, "errors_at_first_success": self.errors_at_first, "final_errors": self.errors_at_first or errors,
                "conditions": [{"index": 0, "bddl": f"(pose_reached ?{AGENT})", "kind": "binary", "satisfied": within}], "measured": [], "group_assignments": {}}


class FakeVecEnv:
    """N single-env fakes stepped together (``og.Environment`` with ``num_envs`` scenes)."""

    def __init__(self, envs: List[FakeEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self.scenes = [e.scene for e in envs]
        self.action_log: List[th.Tensor] = []

    def step(self, actions, n_render_iterations=1):
        actions = th.as_tensor(actions, dtype=th.float32)
        assert actions.shape == (self.num_envs, M.ACTION_DIM)
        self.action_log.append(actions.clone())
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, t, tr, info = e.step(actions[i], n_render_iterations=n_render_iterations)
            obs.append(o)
            rew.append(float(r))
            term.append(bool(t))
            trunc.append(bool(tr))
            infos.append(info)
        return obs, th.tensor(rew), th.tensor(term), th.tensor(trunc), infos


def hold_action(robot: FakeRobot, base=(0.0, 0.0, 0.0)) -> th.Tensor:
    a = th.zeros(M.ACTION_DIM)
    idx = robot.controller_action_idx
    a[idx["base"]] = th.tensor(base, dtype=th.float32)
    a[idx["trunk"]] = robot.q[robot.trunk_control_idx]
    for arm in M.ARMS:
        a[idx[f"arm_{arm}"]] = robot.q[robot.arm_control_idx[arm]]
        opening = float(robot.q[robot.gripper_control_idx[arm]].max())
        a[idx[f"gripper_{arm}"]] = 2.0 * opening / 0.05 - 1.0
    return a


class Drive:
    """A scripted stream: drive straight at ``vx`` (normalized) for ``horizon`` steps; ``stop_on_success`` ends it (``None``)
    once the task reports ``official_success`` - the ACT bridge's rule."""

    def __init__(self, robot: FakeRobot, vx: float, horizon: int, stop_on_success: bool = True):
        self.robot, self.vx, self.horizon, self.stop_on_success = robot, vx, horizon, stop_on_success
        self.steps = 0
        self.stop_reason: Optional[str] = None

    def __call__(self, obs, info):
        isg = (info or {}).get("done", {}).get("ispatialgym") or {}
        if self.stop_on_success and isg.get("official_success"):
            self.stop_reason = "success"
            return None
        if self.steps >= self.horizon:
            self.stop_reason = "horizon"
            return None
        self.steps += 1
        return hold_action(self.robot, base=(self.vx, 0.0, 0.0))


SCENARIOS = {
    # name: (goal distance along +x, normalized vx, horizon, stop_on_success, max_steps)
    "reach_then_stop": (0.5, 0.5, 120, True, 20000),  # success: goal held through the 50-step window
    "never_reached": (5.0, 0.5, 120, True, 20000),  # never-held rule: 10-step window, tolerance verdict
    "reach_and_pass": (0.5, 0.5, 80, False, 20000),  # goal held around step 40 then lost: second window, tolerance at the end
    "timeout": (5.0, 0.5, 120, True, 30),  # QA timeout at step 30: no window
}


def make_world(name: str):
    dist, vx, horizon, stop, max_steps = SCENARIOS[name]
    robot = FakeRobot()
    env = FakeEnv(robot, {}, evaluate=GoalTracker((dist, 0.0)))
    source = Drive(robot, vx, horizon, stop_on_success=stop)
    cfg = ExecutorConfig(max_steps=max_steps)
    return env, source, cfg


class _State:
    held_objects = staticmethod(lambda: [])


def run_single(name: str) -> Dict[str, Any]:
    """The single-env executor on the same scenario, with the lab's never-held rule applied as the production patch does
    (10-step window, no second window, when the goal never held during the stream)."""
    env, source, cfg = make_world(name)
    ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=None, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
    obs, info = env._obs(), env._info()
    ex.last_obs, ex.last_info = obs, info

    def stream():
        o, i = obs, info
        while True:
            a = source(o, i)
            if a is None:
                return
            yield a
            o, i = ex.last_obs, ex.last_info

    # the production epilogue patch (lab_tools/evaluator_patches.py): decided on binary_success_ever at the end of the stream
    orig = ex.epilogue

    def patched(verdict, n_action_steps):
        if not ex.isg.get("binary_success_ever", ex.isg.get("binary_success", False)):
            saved = (cfg.settle_second_window_factor, cfg.settle_fraction, cfg.settle_min_steps)
            cfg.settle_second_window_factor, cfg.settle_fraction, cfg.settle_min_steps = 0.0, 0.0, 10
            try:
                return orig(verdict, n_action_steps)
            finally:
                cfg.settle_second_window_factor, cfg.settle_fraction, cfg.settle_min_steps = saved
        return orig(verdict, n_action_steps)

    ex.epilogue = patched
    result = ex.run_stream(stream())
    result["_pose"] = env.robot.get_position_orientation()
    result["_steps"] = env.steps
    return result


def run_vec(names: List[str]):
    worlds = [make_world(n) for n in names]
    venv = FakeVecEnv([w[0] for w in worlds])
    slots = [Slot(idx=i, robot=w[0].robot, source=w[1], cfg=w[2], task=w[0].task, tracked={}, held=lambda: [], never_held_window=10) for i, w in enumerate(worlds)]
    vex = VecStreamExecutor(venv, slots, sim=None)
    results = vex.run([w[0]._obs() for w in worlds], [w[0]._info() for w in worlds])
    return venv, slots, results


# ---------------------------------------------------------------------------------------------------------------- tests


def test_four_environments_finish_independently_with_their_own_verdicts():
    names = ["reach_then_stop", "never_reached", "reach_and_pass", "timeout"]
    venv, slots, results = run_vec(names)
    by = dict(zip(names, results))

    r = by["reach_then_stop"]
    assert r["success"] is True and r["reason"] is None
    assert r["first_success_step"] >= 0 and r["binary_success_ever"] and r["binary_success_final"]
    assert r["settle_windows"] == [50] and r["n_steps"] == r["n_action_steps"] + 50
    assert r["n_action_steps"] == r["first_success_step"]  # the stream ends at the first official success (the fake counts steps after stepping)

    r = by["never_reached"]
    assert r["success"] is False and r["reason"] == "tolerance" and r["verdict"]["details"]["never_reached"] is True
    assert r["first_success_step"] == -1 and r["settle_windows"] == [10] and r["n_action_steps"] == 120 and r["n_steps"] == 130

    r = by["reach_and_pass"]
    # the goal held, then the robot drove on: the second window runs, and the gating target out of tolerance at the end
    # is the executor's ``tolerance`` verdict (not the never-reached one)
    assert r["success"] is False and r["reason"] == "tolerance" and not r["verdict"]["details"].get("never_reached")
    assert r["binary_success_ever"] and not r["binary_success_final"]
    assert r["settle_windows"] == [50, 100] and r["n_action_steps"] == 80 and r["n_steps"] == 230
    assert "first_window" in r["motion_delta"]

    r = by["timeout"]
    assert r["success"] is False and r["reason"] == "timeout" and r["n_steps"] == 30 and r["n_action_steps"] == 30
    assert r["n_settle_steps"] == 0 and r["settle_windows"] == [10]  # the never-held window is recorded, not run (timeout verdict)

    # the batch ran until the longest episode; every env was stepped every batch step
    assert len(venv.action_log) == max(r["n_steps"] for r in results) == 230
    for e in venv.envs:
        assert e.steps == 230


def test_finished_environment_receives_its_noop_and_stays_put():
    names = ["timeout", "reach_then_stop", "reach_and_pass"]
    venv, slots, results = run_vec(names)
    for i, (name, r) in enumerate(zip(names, results)):
        done_at = r["n_steps"]
        last = slots[i].last_action.clone()
        last[slots[i].robot.controller_action_idx["base"]] = 0.0
        later = [a[i] for a in venv.action_log[done_at:]]
        if done_at == len(venv.action_log):
            continue  # the last episode of the batch: no later step
        assert later, name
        for a in later:
            assert th.equal(a, last), name
    # the timed-out robot (done at step 30) did not move afterwards
    env_t = venv.envs[0]
    pose_mid = None

    env2, src2, cfg2 = make_world("timeout")
    # replay the batch actions for env 0 on a fresh fake up to its verdict, then compare with the final pose
    for a in venv.action_log[:30]:
        env2.step(a[0])
    pose_mid = env2.robot.get_position_orientation()
    pose_end = env_t.robot.get_position_orientation()
    assert th.allclose(pose_mid[0], pose_end[0]) and th.allclose(pose_mid[1], pose_end[1])


@pytest.mark.parametrize("names", [
    ["reach_then_stop", "never_reached", "reach_and_pass", "timeout"],
    ["never_reached", "never_reached", "never_reached"],  # the same episode in every slot
    ["reach_and_pass"],
])
def test_each_slot_matches_the_single_env_executor(names):
    venv, slots, results = run_vec(names)
    keys = ("success", "reason", "n_steps", "n_action_steps", "n_settle_steps", "settle_window", "settle_windows", "first_success_step",
            "binary_success_ever", "binary_success_final", "settled_success")
    for name, r, s in zip(names, results, slots):
        single = run_single(name)
        for k in keys:
            assert r[k] == single[k], (name, k, r[k], single[k])
        if r["verdict"] is not None:
            assert r["verdict"]["category"] == single["verdict"]["category"]
            assert r["verdict"]["message"] == single["verdict"]["message"]
        assert r["errors"] == single["errors"] and r["errors_final_step"] == single["errors_final_step"]
        pos, quat = s.robot.get_position_orientation()
        assert th.equal(pos, single["_pose"][0]) and th.equal(quat, single["_pose"][1]), name
        for k in ("joint_rad", "base_m", "base_rad"):
            assert math.isclose(r["motion_delta"][k], single["motion_delta"][k], abs_tol=1e-9), (name, k)


def test_result_dict_has_the_single_env_keys():
    _, _, (r,) = run_vec(["reach_then_stop"])
    single = run_single("reach_then_stop")
    missing = set(single) - set(r) - {"_pose", "_steps"}
    assert not missing, missing
