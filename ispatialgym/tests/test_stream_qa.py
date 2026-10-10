"""Stream QA as a switch (``stream_qa`` off | stop | log) in the single-env executor and in ``VecStreamExecutor``, and
``Monitors.check_timeout_only`` (``off``: the lab evaluator's former ``Monitors.check`` patch)."""

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
    """QA monitors whose collision monitor fires at the steps ``HITS``; ``check_timeout_only`` is the real one."""

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


def run_single(mode):
    env, source, cfg = make_world("reach_then_stop")
    cfg.stream_qa = mode
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


def run_vec(mode):
    env, source, cfg = make_world("reach_then_stop")
    mon = FakeMonitors()
    slot = Slot(idx=0, robot=env.robot, source=source, cfg=cfg, task=env.task, monitors=mon, tracked={}, held=lambda: [], never_held_window=None)
    (r,) = VecStreamExecutor(FakeVecEnv([env]), [slot], sim=None, stream_qa=mode).run([env._obs()], [env._info()])
    return r, mon


@pytest.mark.parametrize("mode", ["stop", "log", "off"])
def test_stream_qa_modes_single_env_and_vectorized(mode):
    single, smon = run_single(mode)
    vec, vmon = run_vec(mode)
    for k in ("success", "reason", "n_steps", "n_action_steps", "settle_windows", "first_success_step", "stream_qa"):
        assert vec[k] == single[k], (mode, k, vec[k], single[k])
    if mode == "stop":
        assert single["success"] is False and single["reason"] == "collision" and single["n_action_steps"] == HITS[0]
        assert single["stream_qa"] is None
    elif mode == "log":
        assert single["success"] is True and single["n_action_steps"] > HITS[-1]
        (entry,) = single["stream_qa"]
        assert entry["key"] == ["collision", "gripper", "cup_0"] and (entry["first_step"], entry["last_step"], entry["n"]) == (10, 12, 3)
    else:
        assert single["success"] is True and single["stream_qa"] is None and single["n_action_steps"] > HITS[-1]
        assert smon.collision_checks >= single["n_action_steps"] and vmon.collision_checks >= vec["n_action_steps"]  # statistics kept


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


def test_run_isg_accepts_the_three_modes():
    from ispatialgym.oracle import run_isg as R

    for mode in ("stop", "log", "off"):
        assert R.parse_args(["--instance", "x", "--mode", "policy", "--stream-qa", mode]).stream_qa == mode
    assert R.parse_args(["--instance", "x"]).stream_qa == "stop"  # v3.0's default
    with pytest.raises(ValueError):
        VecStreamExecutor(FakeVecEnv([]), [], stream_qa="bogus")
