"""QA of a replay / policy stream in the single-env executor and in ``VecStreamExecutor``: the oracle's retry rules never end a
stream (``Monitors.check_timeout_only``: the collision statistics are kept, only the step budget applies), the ``--qa`` metrics
are logged the same way by both, and the ``run_isg`` switches (``--qa`` / ``--no-qa``, ``--record-view``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ispatialgym.oracle.executor import Executor
from ispatialgym.oracle.qa import Monitors, QAStats, Verdict
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
from ispatialgym.tests.test_v2_oracle import CAM_KEY
from ispatialgym.tests.test_vec_stream import FakeVecEnv, _State, make_world

HITS = (10, 11, 12)


class FakeMonitors:
    """Retry rules whose collision rule fires at the steps ``HITS``; ``check_timeout_only`` is the real one."""

    check_timeout_only = Monitors.check_timeout_only

    def __init__(self):
        self.stats = QAStats()
        self.episode_steps = self.action_steps = 0
        self.collision_checks = 0

    def begin_action(self, *a, **k):
        pass

    def end_action(self, *a, **k):
        pass

    def begin_epilogue(self, *a, **k):
        pass

    def report(self):
        return {"stats": self.stats.to_dict()}

    def _contact_pairs(self, step=None):
        return set()

    def _check_collision(self):
        self.collision_checks += 1
        if self.episode_steps in HITS:
            return Verdict("collision", f"gripper touched cup_0 at step {self.episode_steps}", step=self.episode_steps,
                           details={"link": "gripper", "object": "cup_0"})
        return None

    def _check_timeout(self):
        return None

    def check(self, step=None):
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        return self._check_collision() or self._check_timeout()


def run_single(qa=True):
    env, source, cfg = make_world("reach_then_stop")
    cfg.qa = qa
    mon = FakeMonitors()
    ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=mon, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
    ex.last_obs, ex.last_info = env._obs(), env._info()

    def stream():
        while True:
            a = source(ex.last_obs, ex.last_info)
            if a is None:
                return
            yield a

    return ex.run_stream(stream()), mon


def run_vec(qa=True):
    env, source, cfg = make_world("reach_then_stop")
    mon = FakeMonitors()
    slot = Slot(idx=0, robot=env.robot, source=source, cfg=cfg, task=env.task, monitors=mon, tracked={}, held=lambda: [])
    (r,) = VecStreamExecutor(FakeVecEnv([env]), [slot], sim=None, qa=qa).run([env._obs()], [env._info()])
    return r, mon


@pytest.mark.parametrize("qa", [True, False])
def test_a_stream_is_judged_by_its_goal_single_env_and_vectorized(qa):
    single, smon = run_single(qa)
    vec, vmon = run_vec(qa)
    for k in ("success", "reason", "n_steps", "n_action_steps", "epilogue_steps", "geometric_success_step", "symbolic_success_step"):
        assert vec[k] == single[k], (k, vec[k], single[k])
    assert single["success"] is True and single["reason"] is None and single["n_action_steps"] > HITS[-1]
    assert smon.collision_checks >= single["n_action_steps"] and vmon.collision_checks >= vec["n_action_steps"]  # statistics kept
    if not qa:
        assert single["qa"] == vec["qa"] == {"enabled": False}
        return
    s, v = single["qa"], vec["qa"]
    assert s["mode"] == v["mode"] == "stream" and s["stopped_by"] is None and v["stopped_by"] is None
    assert s["steps_observed"] == v["steps_observed"] == single["n_steps"]
    for name in s["metrics"]:
        if name != "goal_occlusion":  # the vectorized evaluator does not render the goal frame
            assert v["metrics"][name] == s["metrics"][name], name


def test_check_timeout_only_discards_the_collision_verdict_and_keeps_the_timeout():
    calls = []
    stub = SimpleNamespace(episode_steps=0, action_steps=0,
                           _check_collision=lambda: calls.append("collision") or Verdict("collision", "hit", step=1),
                           _check_timeout=lambda: None)
    assert Monitors.check_timeout_only(stub) is None and calls == ["collision"]
    assert (stub.episode_steps, stub.action_steps) == (1, 1)
    assert Monitors.check_timeout_only(stub, 7) is None and (stub.episode_steps, stub.action_steps) == (7, 2)

    def broken():
        raise RuntimeError("no contact view")

    stub._check_collision, stub._check_timeout = broken, lambda: Verdict("timeout", "episode exceeded 7 steps", step=8)
    assert Monitors.check_timeout_only(stub, 8).category == "timeout"


def test_run_isg_qa_and_record_view_switches(monkeypatch):
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.oracle import run_isg as R

    a = R.parse_args(["--instance", "x"])
    assert a.qa is True and a.record_view == ()
    a = R.parse_args(["--instance", "x", "--mode", "replay", "--no-qa", "--record-view", "tiled, head,head,right_wrist"])
    assert a.qa is False and a.record_view == ("tiled", "head", "right_wrist")
    for argv in (["--record-view", "head,top"], ["--stream-qa", "log"], ["--demo-gating", "none"], ["--dev"]):
        with pytest.raises(SystemExit):
            R.parse_args(["--instance", "x"] + argv)
    monkeypatch.setattr(M, "head_camera_only", lambda: True)
    assert R.parse_args(["--instance", "x", "--record-view", "head,external"]).record_view == ("head", "external")
    for view in ("tiled", "left_wrist"):
        with pytest.raises(SystemExit):
            R.parse_args(["--instance", "x", "--record-view", view])
