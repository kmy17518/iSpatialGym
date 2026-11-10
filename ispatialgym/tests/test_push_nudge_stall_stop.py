"""i7-measure 2026-11-10 (lead #90b) ``ISPATIALGYM_PUSH_NUDGE_STALL_STOP`` (default off): a plain push base nudge stops once the base stalls
short of its goal (``execute._NudgeStall``, the brace hold's stall rule) instead of commanding ~6 mm/s into a base that does not move until
the QA pause fails the demo. On the fake robot of ``test_v2_oracle``."""

from __future__ import annotations

import pytest

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.executor import NOTE_KEYS
from ispatialgym.tests.test_push_nudge_brace_hold import _run, _world


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv(EX.PUSH_NUDGE_STALL_STOP_ENV, raising=False)
    assert not EX.push_nudge_stall_stop_on()
    monkeypatch.setenv(EX.PUSH_NUDGE_STALL_STOP_ENV, "1")
    assert EX.push_nudge_stall_stop_on()
    assert "push_nudge_stall" in NOTE_KEYS


def test_off_and_on_nudge_the_same_when_the_base_moves(monkeypatch):
    cfg = EX.ExecuteConfig()
    moved = {}
    for v in ("0", "1"):
        monkeypatch.setenv(EX.PUSH_NUDGE_STALL_STOP_ENV, v)
        robot, env, motions, state = _world()
        rec, n = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02))
        moved[v] = float(robot.pos[0])
        if v == "0":
            assert rec is None
        else:
            assert rec["kind"] == "stall_stop" and rec["stop"] == "nudge_goal"
    assert moved["0"] == pytest.approx(moved["1"], abs=1e-6)


def test_a_stalled_base_stops_the_nudge(monkeypatch):
    monkeypatch.setenv(EX.PUSH_NUDGE_STALL_STOP_ENV, "1")
    cfg = EX.ExecuteConfig()
    robot, env, motions, state = _world()
    pose = motions.base_pose2d()
    monkeypatch.setattr(motions, "base_pose2d", lambda: pose)  # the base does not answer the command (the deadband the demos show)
    rec, n = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02))
    assert rec["stop"] == "stalled" and rec["steps"] == cfg.push_nudge_stall_steps and n < 30
