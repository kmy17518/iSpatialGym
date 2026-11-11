"""i7-measure 2026-11-10 (lead #90b) ``ISPATIALGYM_PUSH_NUDGE_STALL_STOP`` (default off): a plain push base nudge stops once the base stalls
short of its goal (``execute._NudgeStall``, the brace hold's stall rule) instead of commanding ~6 mm/s into a base that does not move until
the QA pause fails the demo. On the fake robot of ``test_v2_oracle``."""

from __future__ import annotations

import pytest

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.executor import NOTE_KEYS
from ispatialgym.tests.test_push_nudge_brace_hold import _run, _world


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv(EX.PUSH_NUDGE_STALL_STOP_ENV, raising=False)
    assert EX.push_nudge_stall_stop_on()  # default ON since the 360102 recollect (owner #92)
    monkeypatch.setenv(EX.PUSH_NUDGE_STALL_STOP_ENV, "0")
    assert not EX.push_nudge_stall_stop_on()
    assert "push_nudge_stall" in NOTE_KEYS


def test_off_and_on_nudge_the_same_when_the_base_moves(monkeypatch):
    monkeypatch.setenv(EX.PUSH_NUDGE_MIN_SPEED_ENV, "0")
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
    monkeypatch.setenv(EX.PUSH_NUDGE_MIN_SPEED_ENV, "0")
    cfg = EX.ExecuteConfig()
    robot, env, motions, state = _world()
    pose = motions.base_pose2d()
    monkeypatch.setattr(motions, "base_pose2d", lambda: pose)  # the base does not answer the command (the deadband the demos show)
    monkeypatch.setattr(state, "base_pose2d", lambda: pose)
    rec, n = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02))
    assert rec["stop"] == "stalled" and rec["steps"] == cfg.push_nudge_stall_steps and n < 30


def test_min_speed_parsing(monkeypatch):
    monkeypatch.delenv(EX.PUSH_NUDGE_MIN_SPEED_ENV, raising=False)
    assert EX.push_nudge_min_speed() == EX.PUSH_NUDGE_MIN_SPEED_DEFAULT_M_S  # default ON since owner #92
    for v, want in (("1", EX.PUSH_NUDGE_MIN_SPEED_DEFAULT_M_S), ("on", EX.PUSH_NUDGE_MIN_SPEED_DEFAULT_M_S), ("0.03", 0.03), ("0", None), ("off", None), ("x", None)):
        monkeypatch.setenv(EX.PUSH_NUDGE_MIN_SPEED_ENV, v)
        assert EX.push_nudge_min_speed() == want, v


def test_min_speed_raises_the_slow_command_and_reaches_the_goal(monkeypatch):
    # lead #90c: the tracker's command decays below the base's breakaway near the goal; the floor keeps it above until within tolerance
    monkeypatch.setenv(EX.PUSH_NUDGE_STALL_STOP_ENV, "0")
    cfg = EX.ExecuteConfig()
    res = {}
    for v in ("0", "0.025"):
        monkeypatch.setenv(EX.PUSH_NUDGE_MIN_SPEED_ENV, v)
        robot, env, motions, state = _world()
        rec, n = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02))
        res[v] = (rec, n, float(robot.pos[0]))
    assert res["0"][0] is None
    rec, n, x = res["0.025"]
    assert rec["kind"] == "min_speed" and rec["raised_steps"] > 0 and n <= res["0"][1]
    assert x == pytest.approx(0.02 + cfg.push_base_nudge_overshoot_m, abs=0.008)
