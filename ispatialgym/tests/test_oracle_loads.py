"""Loads the hands bear (owner 2026-10-09 audit of the v3.0-reconciled demos): the drives stop pressing before a loaded contact is let go
of, a free-hanging object is released as the fingers open, a next-to / touching release keeps the opening fingers off the partner, and a
demo whose torso a load bent visibly is discarded (``robot_wobble``)."""

from __future__ import annotations

import math

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import ExecutorConfig, TorsoLoad
from ispatialgym.oracle.motions import as_world_pose
from ispatialgym.oracle.programs import expand
from ispatialgym.tests.test_v2_oracle import FakeRobot, _place_goal, iors_world, make_executor, success_eval


def _motions():
    sc, env = iors_world()
    _ex, motions, _backend, _rec = make_executor(env, sc)
    return env, motions


def test_unload_bleeds_the_arm_and_the_torso_onto_the_measured_joints_and_is_silent_unloaded():
    env, motions = _motions()
    r = env.robot
    motions.state.hold_current()
    assert list(motions.unload(["right"])) == []  # nothing presses: no step, no target changed
    r.q[r.trunk_control_idx[0]] += 0.02  # a load bent the torso 1.1 deg past its target
    r.q[r.arm_control_idx["right"][3]] -= 0.03
    gen = motions.unload(["right"])
    steps = []
    try:
        while True:
            steps.append(next(gen))
    except StopIteration as stop:
        out = stop.value
    assert len(steps) == motions.cfg.unload_steps
    assert th.allclose(motions.state.trunk_target, r.q[r.trunk_control_idx])
    assert th.allclose(motions.state.arm_target["right"], r.q[r.arm_control_idx["right"]])
    assert out["trunk"] == pytest.approx(0.02, abs=1e-6) and out["right"] == pytest.approx(0.03, abs=1e-6)
    assert float(steps[0][r.controller_action_idx["base"]].abs().max()) == 0.0


def test_unwind_trunk_ignores_deflections_within_the_threshold():
    env, motions = _motions()
    r = env.robot
    motions.state.hold_current()
    t0 = motions.state.trunk_target.clone()
    r.q[r.trunk_control_idx[1]] += 0.0005
    assert motions.unwind_trunk(0.001) == 0.0 and th.equal(motions.state.trunk_target, t0)
    r.q[r.trunk_control_idx[1]] += 0.002
    assert motions.unwind_trunk(0.001) == pytest.approx(0.0025, abs=1e-6)
    assert th.allclose(motions.state.trunk_target, r.q[r.trunk_control_idx])


def test_until_stops_after_the_step_that_meets_the_condition_and_closes_the_motion():
    closed = []

    def motion():
        try:
            for i in range(10):
                yield i
        finally:
            closed.append(True)

    seen = []

    def stop():
        return len(seen) >= 3

    gen = EX._until(motion(), stop)
    try:
        while True:
            seen.append(next(gen))
    except StopIteration as e:
        stopped = e.value
    assert seen == [0, 1, 2] and stopped is True and closed == [True]
    gen2 = EX._until(iter(range(4)), lambda: False)
    out = []
    try:
        while True:
            out.append(next(gen2))
    except StopIteration as e:
        assert e.value is False
    assert out == [0, 1, 2, 3]


def test_gripper_instant_opening_reaches_the_open_command_on_its_first_step():
    env, motions = _motions()
    r = env.robot
    idx = r.controller_action_idx["gripper_right"]
    r.q[r.gripper_control_idx["right"]] = 0.02
    first = next(motions.gripper("right", close=False, instant=True))
    assert float(first[idx]) == pytest.approx(1.0)
    r.q[r.gripper_control_idx["right"]] = 0.02
    ramped = next(motions.gripper("right", close=False))
    assert float(ramped[idx]) < 0.0  # the ramp starts at the fingers' opening


def test_free_standing_follow_runs_the_handle_arc_reaimed_for_the_live_body_and_slows_while_the_torso_bends():
    from types import SimpleNamespace as NS

    from ispatialgym.robot import conventions as C

    hinge, axis, r = th.tensor([1.0, 0.0, 1.0]), th.tensor([0.0, 1.0, 0.0]), 0.1
    q0 = th.tensor([0.0, 0.0, 0.0, 1.0])

    def on_circle(ang):
        return (hinge + th.tensor([r * math.cos(ang), 0.0, r * math.sin(ang)]), q0)

    body = {"pos": th.tensor([1.0, 0.1, 0.8]), "calls": 0}

    class State:
        def object(self, name):
            return NS(fixed_base=False)

        def object_pose(self, name):
            body["calls"] += 1
            p = body["pos"] + (th.tensor([0.005, 0.0, 0.0]) if body["calls"] > 3 else th.zeros(3))  # the body slides 5 mm mid-follow
            return p, q0

    seen, trunk = [], [0.0, math.radians(0.3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class Mot:
        cfg = NS(cartesian_speed=0.1)
        state = NS(trunk_error=lambda: trunk[min(len(seen) - 1, len(trunk) - 1)])

        def eef_pose(self, arm):
            return on_circle(0.0)

        def arm_along(self, arm, wps, converge=True, speed=None, **kw):
            seen.append(([w[0].clone() for w in wps], speed))
            return iter(())

    art = NS(object="blender", extra={"hinge_world": hinge.tolist(), "axis_world": axis.tolist(),
                                      "object_pose": {"position": body["pos"].tolist(), "orientation": q0.tolist()}})
    note = {}
    cfg = EX.ExecuteConfig()
    list(EX._follow_handle_poses(NS(arm="right", articulation=art), Mot(), State(), [on_circle(0.31), on_circle(0.62)], cfg, note))
    info = note["live_reaim"]
    assert info["arc"] and info["waypoints"] == 22 and info["chunks"] == 6  # 0.31 rad / 0.03 -> 11 arc waypoints per segment, 4 per chunk
    first = seen[0][0]
    assert all(abs(float(th.linalg.norm((p - hinge)[[0, 2]])) - r) < 1e-4 for p in first)  # on the handle's circle, not a chord
    assert info["reaimed"] == 5 and info["max_move_m"] == pytest.approx(0.005, abs=1e-4)  # every chunk after the body slid
    assert th.allclose(seen[0][0][-1], on_circle(4 * 0.31 / 11)[0], atol=1e-4)  # before the slide: the recorded arc
    assert th.allclose(seen[-1][0][-1], on_circle(0.62)[0] + th.tensor([0.005, 0.0, 0.0]), atol=1e-4)  # moved with the body
    assert seen[1][1] is None and seen[2][1] == pytest.approx(0.1 * cfg.follow_slow_factor) and seen[3][1] is None and info["slowed"] == 1
    seen.clear()
    fixed_state = State()
    fixed_state.object = lambda name: NS(fixed_base=True)
    list(EX._follow_handle_poses(NS(arm="right", articulation=art), Mot(), fixed_state, [on_circle(0.31), on_circle(0.62)], cfg, {}))
    assert len(seen) == 1 and len(seen[0][0]) == 2  # a fixed-base articulation: the recorded poses as they are
    seen.clear()
    unknown = State()
    unknown.object = lambda name: NS()
    list(EX._follow_handle_poses(NS(arm="right", articulation=art), Mot(), unknown, [on_circle(0.31), on_circle(0.62)], cfg, {}))
    assert len(seen) == 1  # no fixed_base attribute: as recorded


def test_partner_clear_shift_moves_the_place_away_from_the_partner_by_what_the_opening_misses():
    place, partner = th.tensor([1.0, 0.0, 0.8]), th.tensor([0.9, 0.0, 0.85])
    shift, need = EX.partner_clear_shift(0.012, th.tensor([-1.0, 0.0, 0.0]), place, partner, 0.03)
    assert need == pytest.approx(0.012) and th.allclose(shift, th.tensor([0.012, 0.0, 0.0]))
    oblique = th.tensor([-math.cos(math.radians(60)), -math.sin(math.radians(60)), 0.0])
    shift, need = EX.partner_clear_shift(0.012, oblique, place, partner, 0.03)
    assert need == pytest.approx(0.024, abs=1e-6) and th.allclose(shift, th.tensor([0.024, 0.0, 0.0]), atol=1e-6)
    shift, need = EX.partner_clear_shift(0.05, None, place, partner, 0.03)
    assert need == pytest.approx(0.05) and th.allclose(shift, th.tensor([0.03, 0.0, 0.0]))
    shift, _ = EX.partner_clear_shift(0.01, None, th.tensor([0.9, 0.0, 0.5]), partner, 0.03)
    assert shift is None


def test_shift_place_targets_moves_every_place_target_and_keeps_the_orientations():
    from ispatialgym.oracle.actions import Pose

    q = [0.0, 0.0, 0.0, 1.0]
    tg = {"pre_place": Pose(position=[1.0, 0.0, 1.0], orientation=q), "place": Pose(position=[1.0, 0.0, 0.9], orientation=q),
          "descent": [Pose(position=[1.0, 0.0, 0.95], orientation=q)], "retreat": [Pose(position=[1.0, 0.0, 1.0], orientation=q)],
          "carry": [Pose(position=[0.5, 0.0, 1.0], orientation=q)], "carry_via": [], "other": 7}
    out = EX.shift_place_targets(tg, th.tensor([0.01, -0.02, 0.0]))
    for k in ("pre_place", "place"):
        p, o = as_world_pose(out[k])
        p0, _ = as_world_pose(tg[k])
        assert th.allclose(p, p0 + th.tensor([0.01, -0.02, 0.0])) and th.allclose(o, th.tensor(q))
    for k in ("descent", "retreat", "carry"):
        assert th.allclose(as_world_pose(out[k][0])[0], as_world_pose(tg[k][0])[0] + th.tensor([0.01, -0.02, 0.0]))
    assert out["carry_via"] == [] and out["other"] == 7


def test_torso_load_records_the_peak_per_action_and_forgets_a_rolled_back_attempt():
    r = FakeRobot()
    tl = TorsoLoad(r)
    a = th.zeros(M.ACTION_DIM)
    a[r.controller_action_idx["trunk"]] = r.q[r.trunk_control_idx]
    tl.observe(1, a, "grasp")
    assert tl.report()["max_deg"] == 0.0 and tl.report()["at_step"] is None
    r.q[r.trunk_control_idx[0]] += math.radians(0.8)
    tl.observe(2, a, "grasp")
    r.q[r.trunk_control_idx[0]] += math.radians(3.0)
    tl.observe(3, a, "push_close")
    rep = tl.report()
    assert rep["max_deg"] == pytest.approx(3.8, abs=1e-3) and rep["at_step"] == 3 and rep["action"] == "push_close"
    assert rep["per_action_deg"]["grasp"] == pytest.approx(0.8, abs=1e-3)
    tl.trim(2)
    assert tl.report()["max_deg"] == pytest.approx(0.8, abs=1e-3) and tl.report()["action"] == "grasp"


@pytest.mark.parametrize("deg, gated", [(2.0, True), (0.5, False)])
def test_robot_wobble_gate_discards_a_demo_whose_torso_a_load_bent_visibly(tmp_path, deg, gated, monkeypatch):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("robot_wobble",)))
    seen = {"n": 0}
    real = TorsoLoad.observe

    def observe(self, step, action, name):
        real(self, step, action, name)
        if name == "place" and seen["n"] < 3:
            seen["n"] += 1
            self.steps[int(step)] = (deg, name)  # the place's contact bent the torso this far at three steps

    monkeypatch.setattr(TorsoLoad, "observe", observe)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["torso_load"]["max_deg"] == pytest.approx(deg, abs=1e-6) and result["torso_load"]["action"] == "place"
    if gated:
        assert not result["success"] and result["reason"] == "robot_wobble"
    else:
        assert result["success"], result["verdict"]


def test_goal_occlusion_gate_discards_with_its_own_verdict(tmp_path, monkeypatch):
    from ispatialgym.oracle.executor import Executor

    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("goal_occlusion",)))
    rep = {"method": "live head-camera depth", "threshold": 0.5, "objects": {}, "max_occluded": 0.62, "worst": "target_apple_0", "occluded": True}
    monkeypatch.setattr(Executor, "goal_occlusion_report", lambda self: dict(rep))
    result = ex.run_program(expand(sc.alias, sc))
    assert not result["success"] and result["reason"] == "goal_occluded" and result["goal_occlusion"]["worst"] == "target_apple_0"
