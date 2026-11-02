"""The release check: ``release_check`` recorded as a fact in every mode - by the single-env executor and, with the same
rule, per slot by ``VecStreamExecutor`` - and the ``--qa`` hard metric ``bad_release`` (an assisted-grasp release without a
gripper command of +1): it fails an oracle demo at once and is only logged in a stream."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ispatialgym.eval import metrics as MET
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import Executor, ExecutorConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
from ispatialgym.tests.test_v2_oracle import CAM_KEY, FakeEnv, FakeRobot, _place_goal, iors_world, make_executor, quiet_qa, success_eval
from ispatialgym.tests.test_vec_stream import Drive, FakeVecEnv, GoalTracker, _State


def _single(env, source, cfg):
    ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=None, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
    ex.last_obs, ex.last_info = env._obs(), env._info()

    def stream():
        while True:
            a = source(ex.last_obs, ex.last_info)
            if a is None:
                return
            yield a

    return ex.run_stream(stream())


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
    env.task.tracker = MET.SuccessTracker(settle_eval=False)  # scored as a policy stream (no settle window)
    return env, GripperDrive(robot, open_at=12 if kind == "open" else None), ExecutorConfig(max_steps=20000)


@pytest.mark.parametrize("kind", ["open", "drop"])
def test_vec_records_releases_with_the_single_env_rule_and_logs_a_bad_one(kind):
    env, source, cfg = _grasp_world(kind)
    single = _single(env, source, cfg)
    venv_env, vsource, vcfg = _grasp_world(kind)
    slot = Slot(idx=0, robot=venv_env.robot, source=vsource, cfg=vcfg, task=venv_env.task, tracked={}, held=lambda: [])
    (r,) = VecStreamExecutor(FakeVecEnv([venv_env]), [slot], sim=None, qa=True).run([venv_env._obs()], [venv_env._info()])
    assert r["release_check"] == single["release_check"]
    (rel,) = r["release_check"]["releases"]
    assert rel["step"] == 12 and rel["arm"] == "right" and rel["object"] == "cup_0"
    bad = r["qa"]["metrics"]["bad_release"]
    assert bad == single["qa"]["metrics"]["bad_release"]
    if kind == "open":
        assert rel["gripper_command"] == 1.0 and r["release_check"]["violations"] == [] and not bad["triggered"]
    else:
        assert rel["gripper_command"] == -1.0 and r["release_check"]["violations"][0]["kind"] == "gripper_command_below_open"
        assert bad["triggered"] and [v["step"] for v in bad["violations"]] == [12] and "bad_release" in r["qa"]["hard_triggered"]
    assert r["success"] is True and r["reason"] is None  # a stream only logs


def test_bad_release_fails_an_oracle_demo_without_a_retry(tmp_path, monkeypatch):
    """The apple slips out of the assisted grasp while held (no open command): ``bad_release`` ends the demo at that step."""
    quiet_qa(monkeypatch)
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    robot = env.robot
    dropped = {"step": None, "held": 0}

    def sync():
        robot._ag_obj_constraints = {a: (True if robot._ag_obj_in_hand[a] is not None else None) for a in M.ARMS}

    def slip(action):
        if dropped["step"] is None and robot._ag_obj_in_hand["right"] is not None:
            dropped["held"] += 1
            if dropped["held"] == 10:
                robot._ag_obj_in_hand["right"] = None
                dropped["step"] = env.steps
        sync()

    sync()
    env.after_step = slip
    ex, *_ = make_executor(env, sc, tmp_path, qa=True)
    result = ex.run_program(expand(sc.variant, sc))
    assert dropped["step"] is not None
    assert not result["success"] and result["reason"] == "bad_release" and result["qa"]["stopped_by"] == "bad_release"
    v = result["verdict"]
    assert v["details"]["object"] == "apple_omzprq_0" and len(result["actions"][v["action_index"]]["attempts"]) == 1
    assert result["release_check"]["violations"] and result["release_check"]["violations"][0]["object"] == "apple_omzprq_0"
