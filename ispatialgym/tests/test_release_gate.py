"""The release check as a demo gate (``release_commanded``): the gates per ``--mode`` under ``--demo-gating auto``, the
verdict only with the gate on, ``release_check`` recorded as a fact in every mode - by the single-env executor and, with
the same rule, per slot by ``VecStreamExecutor``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import Executor, ExecutorConfig
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
from ispatialgym.tests.test_v2_oracle import CAM_KEY, FakeEnv, FakeRobot
from ispatialgym.tests.test_vec_stream import Drive, FakeVecEnv, GoalTracker, _State, make_world

ORACLE_GATES = ("placed_orientation", "root_moved", "robot_wobble")


@pytest.mark.parametrize("occlusion", [True, False])
def test_demo_gating_per_mode(monkeypatch, occlusion):
    from ispatialgym.oracle import run_isg as R

    monkeypatch.setattr(R.goal_occlusion, "enabled", lambda: occlusion)
    assert R.demo_gating_for("oracle", "auto") == ORACLE_GATES + (("goal_occlusion",) if occlusion else ()) + ("release_commanded", "arm_stutter")
    assert R.demo_gating_for("replay", "auto") == ("release_commanded", "arm_stutter")
    assert R.demo_gating_for("policy", "auto") == ()
    for mode in ("oracle", "replay", "policy"):
        assert R.demo_gating_for(mode, "none") == ()
        assert R.demo_gating_for(mode, "release_commanded") == ("release_commanded",)
        assert R.demo_gating_for(mode, "root_moved") == ("root_moved",)  # an explicit gate replaces auto: no release check
    assert R.parse_args(["--instance", "x", "--demo-gating", "release_commanded"]).demo_gating == "release_commanded"


def _single(env, source, cfg, violation=None):
    ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=None, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
    ex.last_obs, ex.last_info = env._obs(), env._info()
    if violation is not None:
        ex.release_check["violations"].append(dict(violation))

    def stream():
        while True:
            a = source(ex.last_obs, ex.last_info)
            if a is None:
                return
            yield a

    return ex.run_stream(stream())


@pytest.mark.parametrize("mode", ["oracle", "replay", "policy"])
def test_release_check_is_a_fact_and_the_gate_decides(monkeypatch, mode):
    from ispatialgym.oracle import run_isg as R

    monkeypatch.setattr(R.goal_occlusion, "enabled", lambda: False)
    env, source, cfg = make_world("reach_then_stop")
    # the mode's gates; the oracle's placement / root / torso gates judge a program, which a stream does not have
    cfg.demo_gating = tuple(g for g in R.demo_gating_for(mode, "auto") if g not in ORACLE_GATES)
    bad = {"step": 7, "arm": "right", "object": "cup_0", "kind": "released_between_steps"}
    r = _single(env, source, cfg, violation=bad)
    assert r["release_check"]["violations"] == [bad]
    if mode == "policy":
        assert r["success"] is True and r["reason"] is None
    else:
        assert r["success"] is False and r["reason"] == "release_not_commanded"


class GraspEnv(FakeEnv):
    """The fake with ``cup_0`` in the right hand's assisted grasp: the fake's rule lets it go at an open command (+1);
    ``drop_at`` lets it go at that step whatever the command (a release between gripper commands)."""

    def __init__(self, robot, evaluate, drop_at=None):
        super().__init__(robot, {}, evaluate=evaluate)
        self.drop_at = drop_at
        robot._ag_obj_in_hand["right"] = SimpleNamespace(name="cup_0")
        self._sync()

    def _sync(self):
        self.robot._ag_obj_constraints = {a: (True if self.robot._ag_obj_in_hand[a] is not None else None) for a in M.ARMS}

    def step(self, action, n_render_iterations=1):
        out = super().step(action, n_render_iterations=n_render_iterations)
        if self.drop_at is not None and self.steps == self.drop_at:
            self.robot._ag_obj_in_hand["right"] = None
        self._sync()
        return out


class GripperDrive(Drive):
    """``Drive`` holding the right gripper closed (-1), opened (+1) from action ``open_at`` on."""

    def __init__(self, robot, open_at=None):
        super().__init__(robot, 0.5, 120, stop_on_success=True)
        self.open_at = open_at

    def __call__(self, obs, info):
        a = super().__call__(obs, info)
        if a is not None:
            a[self.robot.controller_action_idx["gripper_right"]] = 1.0 if self.open_at is not None and self.steps >= self.open_at else -1.0
        return a


def _grasp_world(kind):
    robot = FakeRobot()
    env = GraspEnv(robot, GoalTracker((0.5, 0.0)), drop_at=12 if kind == "drop" else None)
    return env, GripperDrive(robot, open_at=12 if kind == "open" else None), ExecutorConfig(max_steps=20000)


@pytest.mark.parametrize("kind", ["open", "drop"])
def test_vec_records_releases_with_the_single_env_rule(kind):
    env, source, cfg = _grasp_world(kind)
    single = _single(env, source, cfg)
    venv_env, vsource, vcfg = _grasp_world(kind)
    slot = Slot(idx=0, robot=venv_env.robot, source=vsource, cfg=vcfg, task=venv_env.task, tracked={}, held=lambda: [], never_held_window=10)
    (r,) = VecStreamExecutor(FakeVecEnv([venv_env]), [slot], sim=None).run([venv_env._obs()], [venv_env._info()])
    assert r["release_check"] == single["release_check"]
    (rel,) = r["release_check"]["releases"]
    assert rel["step"] == 12 and rel["arm"] == "right" and rel["object"] == "cup_0"
    if kind == "open":
        assert rel["gripper_command"] == 1.0 and r["release_check"]["violations"] == []
    else:
        assert rel["gripper_command"] == -1.0 and r["release_check"]["violations"][0]["kind"] == "gripper_command_below_open"
    assert r["success"] is True and r["reason"] is None  # facts only: no release verdict in the vectorized evaluator
