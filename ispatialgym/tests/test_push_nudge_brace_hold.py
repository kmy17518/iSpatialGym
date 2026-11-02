"""The brace hand during a push's base nudge (owner 2026-10-14 ~04:25, after touching#02 pair1013b_c8_k9 / k10 / k12: every demo needed a
brace-retry rung - the rock-back round's base nudge carried the brace hand off the partner's far face and pushed the pair 2.1-2.2 cm past
the goal, the partner 26.5 mm in k10): the executor reads the standing braces off the program (``Executor.braces_held``), and
``execute._base_nudge`` holds each brace hand where it stands in the world (``_NudgeBraceHold``: one IK solve per control step), stopping
the base once the pushed object is at its goal or the base stalls against the braced pair; ``ExecuteConfig.push_nudge_hold_brace`` False
keeps the former nudge. On the fake robot of ``test_v2_oracle`` (its IK / FK follow the live base, its env integrates the base command)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.executor import NOTE_KEYS, Executor, SimState
from ispatialgym.oracle.motions import Motions, MotionsConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.tests.test_v2_oracle import SIDECARS, FakeEnv, FakeRobot, _FkBackend


def _world():
    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    env = FakeEnv(robot, {})
    motions = Motions(env, _FkBackend(robot), MotionsConfig(settle_steps=5, gripper_steps=5))
    return robot, env, motions, SimState(env, env.task, robot)


def _run(env, gen):
    n = 0
    while True:
        try:
            a = next(gen)
        except StopIteration as e:
            return e.value, n
        env.step(a)
        n += 1
        assert n < 5000


def test_braces_held_span_each_brace_to_its_arms_next_retreat():
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-touching.json"))
    actions = expand(sc.variant, sc)
    names = [a.name for a in actions]
    b, p = names.index("brace"), names.index("push")
    brace = actions[b]
    ret = next(i for i, a in enumerate(actions) if a.name == "retreat" and a.arm == brace.contact.arm and i > b)
    ex = SimpleNamespace(program_actions=actions)
    assert b < p < ret
    for i in range(len(actions)):
        held = Executor.braces_held(ex, i)
        want = [{"arm": brace.contact.arm, "object": brace.contact.object, "brace_index": b}] if b < i < ret else []
        assert held == want, (i, names[i], held)
    # a wide stop that lifts off with its fingers open (no release, block design's pack stage): held up to that retreat only
    prog = [SimpleNamespace(name="brace", contact=SimpleNamespace(arm="left", object="row")), SimpleNamespace(name="push", arm="right"),
            SimpleNamespace(name="retreat", arm="right", release=True), SimpleNamespace(name="retreat", arm="left", release=False),
            SimpleNamespace(name="push", arm="right")]
    assert [bool(Executor.braces_held(SimpleNamespace(program_actions=prog), i)) for i in range(len(prog))] == [False, True, True, False, False]


def test_the_held_arms_exclude_the_push_arm_and_follow_the_switch():
    ctx = {"braces_held": [{"arm": "left", "object": "p"}, {"arm": "right", "object": "q"}, {"arm": "left", "object": "p2"}]}
    assert EX.held_brace_arms(ctx, "right", EX.ExecuteConfig()) == ["left"]
    assert EX.held_brace_arms(ctx, "right", EX.ExecuteConfig(push_nudge_hold_brace=False)) == []
    assert EX.held_brace_arms({}, "right", EX.ExecuteConfig()) == []
    assert {"push_speed_m_s", "push_nudge_brace_hold"} <= set(NOTE_KEYS)  # both reach meta.json actions[].notes


def test_a_base_nudge_holds_the_brace_hand_where_it_stands():
    cfg = EX.ExecuteConfig()
    robot, env, motions, state = _world()
    left0, right0 = robot.eef_pose("left")[0].clone(), robot.eef_pose("right")[0].clone()
    rec, n = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02, hold_arms=["left"]))
    moved = float(robot.pos[0])
    assert moved == pytest.approx(0.02 + cfg.push_base_nudge_overshoot_m, abs=0.008)  # the nudge itself is unchanged
    assert float(th.linalg.norm(robot.eef_pose("left")[0] - left0)) < 0.001  # the brace hand stayed where it stood
    hold = rec["arms"]["left"]
    assert hold["max_drift_m"] < 0.001 and hold["ik_failures"] == 0 and rec["steps"] > 30 and rec["stop"] == "nudge_goal"
    assert rec["base_moved_m"] == pytest.approx(moved, abs=0.002) and rec["label"] == "push"
    assert float(robot.eef_pose("right")[0][0] - right0[0]) == pytest.approx(moved, abs=1e-3)  # the push hand rides with the base
    # the former nudge: the brace hand moves with the base
    robot2, env2, motions2, state2 = _world()
    l0 = robot2.eef_pose("left")[0].clone()
    rec2, _ = _run(env2, EX._base_nudge(motions2, state2, "right", "box", [1.0, 0.0, 0.0], 0.02, cfg, "push", speed=0.02))
    assert rec2 is None and float(robot2.eef_pose("left")[0][0] - l0[0]) == pytest.approx(float(robot2.pos[0]), abs=1e-3)


def test_a_held_nudge_stops_at_the_goal_or_when_the_base_stalls():
    cfg = EX.ExecuteConfig()
    robot, env, motions, state = _world()
    calls = {"n": 0}

    def at_goal():
        calls["n"] += 1
        return calls["n"] >= 12

    rec, _ = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.05, cfg, "push", speed=0.02, hold_arms=["left"], stop_when=at_goal))
    assert rec["stop"] == "object_at_goal" and rec["steps"] == 12 and 0.0 < float(robot.pos[0]) < 0.01
    # the base pressing the pair against the brace does not advance: stopped after push_nudge_stall_steps steps, nothing forced on
    robot, env, motions, state = _world()
    real_step = env.step

    def stuck(action, **kw):
        p = robot.pos.clone()
        out = real_step(action, **kw)
        robot.pos = p
        return out

    env.step = stuck
    rec, _ = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.05, cfg, "push", speed=0.02, hold_arms=["left"]))
    assert rec["stop"] == "stalled_against_the_brace" and rec["steps"] == cfg.push_nudge_stall_steps and abs(float(robot.pos[0])) < 1e-9
    # a nudge settling into the tracker's tolerance is no stall
    robot, env, motions, state = _world()
    rec, _ = _run(env, EX._base_nudge(motions, state, "right", "box", [1.0, 0.0, 0.0], 0.004, cfg, "push", speed=0.02, hold_arms=["left"]))
    assert rec["stop"] == "nudge_goal"
