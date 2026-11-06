"""oracle-fix 2026-11-06 (alignment-axial_01 / _05, owner request "aliases that have low yield & take long time ... fix them"): the free-object
arc push's reach end. The dominant failed demo was push ``plan_failed`` "arm_arc_closed_loop(right): 10 consecutive differential-IK
misses": the hand orbits 0.07-0.16 rad ahead of the object, the elbow straightens and every step misses.

- ``generation.push_arc_stall_ends_arc`` (oracle): those misses end the arc (``stop = "ik_stall"``) instead of failing the push.
- ``generation.push_arc_replay_lag_rad`` (sampler, ``program_check.replay_arc_push``): the replayed arc must track that far past the band.
Both default off (today's behaviour)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.motions import Motions, MotionsConfig, SkillError
from ispatialgym.oracle.executor import SimState
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.tests.test_v2_oracle import _FkBackend, _Res, axial_world


class _StallBackend(_FkBackend):
    """The closed-loop arc's one-waypoint steps (``rot_tol`` 5e-4, 20 iterations) succeed ``ok_steps`` times, then miss for good - the
    reach end of the axial demos (pos_err 0.3-0.9 mm, no joint at a limit); every other solve is the fake's."""

    def __init__(self, robot, ok_steps: int):
        super().__init__(robot)
        self.ok_steps, self.arc_calls = ok_steps, 0

    def solve_ik_path(self, emb, waypoints, q_start=None, pos_tol=2e-3, rot_tol=1e-2, check_collisions=True, sync_world=True, **kw):
        if len(waypoints) == 1 and rot_tol == 5e-4 and kw.get("max_iters") == 20:
            self.arc_calls += 1
            if self.arc_calls > self.ok_steps:
                return _Res(False, status="differential IK did not converge at waypoint 0/1: pos_err=0.4mm rot_err=0.000rad at_lower=[] at_upper=[] (fake)")
        return super().solve_ik_path(emb, waypoints, q_start=q_start, pos_tol=pos_tol, rot_tol=rot_tol, check_collisions=check_collisions,
                                     sync_world=sync_world, **kw)


def _run_push(monkeypatch, generation: Dict[str, Any], ok_steps: int = 15):
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, tol=0.10)
    backend = _StallBackend(env.robot, ok_steps)
    motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
    state = SimState(env, env.task, env.robot)
    monkeypatch.setattr(EX, "variant_generation", lambda ctx: dict(generation))
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(push_arc_tracking="closed_loop", push_backoff_m=0.02), "witnesses": {}, "hooks": {}}
    targets = push.targets(sc, state)
    for a in EX.run_action(push, motions, state, targets, ctx):
        env.step(a)
    return ctx, backend


def test_reach_end_misses_fail_the_push_by_default(monkeypatch):
    with pytest.raises(SkillError) as e:
        _run_push(monkeypatch, {})
    assert e.value.reason == "plan_failed" and "consecutive differential-IK misses" in (e.value.message or "")


def test_reach_end_misses_end_the_arc_when_opted_in(monkeypatch):
    ctx, backend = _run_push(monkeypatch, {"push_arc_stall_ends_arc": True})
    cl = ctx["push_arc_closed_loop"]
    assert cl["stop"] == "ik_stall" and "10 consecutive differential-IK misses" in cl["stall"]
    assert cl["ik_failures"] == 10 and backend.arc_calls >= 15 + 10  # the push arc, then the re-contact rounds (each a 10-miss arc here)
    assert "push_residual" in ctx  # the push went on to its residual (re-push / back-off) instead of failing


# ------------------------------------------------------------------------------------------------ the sampler's replay


class _ToyKit:
    """A toy arm whose eef position is ``q[idx[:3]]`` and yaw ``q[idx[3]]`` (no kinematics to speak of), and an IK that solves any
    target up to ``reach_rad`` round the pivot and misses beyond - the arm's reach end."""

    def __init__(self, pivot, reach_rad: float):
        self.pivot, self.reach_rad = pivot, reach_rad
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(10, 17)})
        self.backend = SimpleNamespace(control_dt=1.0 / 30.0, solve_ik_path=self.solve_ik_path)

    def _ensure_synced(self):
        pass

    def fk(self, frames, q):
        i = self.robot.arm_control_idx["right"]
        return {f: (q[i[:3]].clone(), C.quat_from_yaw(float(q[i[3]]))) for f in frames}

    def angle_of(self, p) -> float:
        return math.atan2(float(p[1] - self.pivot[1]), float(p[0] - self.pivot[0]))

    def solve_ik_path(self, emb, waypoints, q_start=None, **kw):
        q = th.as_tensor(q_start).reshape(-1).float().clone()
        i = self.robot.arm_control_idx["right"]
        path = [q.clone()]
        for wp in waypoints:
            p, o = next(iter(wp.values()))
            if self.angle_of(p) > self.reach_rad + 1e-6:
                return SimpleNamespace(success=False, q_traj=None, status="differential IK did not converge at waypoint 0/1: pos_err=0.4mm (toy)")
            q = q.clone()
            q[i[:3]] = th.as_tensor(p).float()
            q[i[3]] = float(C.yaw_of(th.as_tensor(o).float()))
            path.append(q.clone())
        return SimpleNamespace(success=True, q_traj=th.stack(path), status="ok")


def _replay(monkeypatch, reach_rad: float, lag: float):
    pivot = th.tensor([0.0, 0.0, 0.8])
    r, angle = 0.10, 0.8
    pose = lambda a: (pivot + th.tensor([r * math.cos(a), r * math.sin(a), 0.0]), C.quat_from_yaw(a))  # noqa: E731
    kit = _ToyKit(pivot, reach_rad)
    sampler = SimpleNamespace(kit=kit, ikc=SimpleNamespace(joint_limit_margin=lambda q, arm: (0.5, "right_arm_joint1")))
    monkeypatch.setattr(PC, "replay_contact_escape", lambda *a, **k: ("replayed", "escape tracked (toy)", {"escape": True}, None))
    q0 = th.zeros(28)
    i = kit.robot.arm_control_idx["right"]
    p0, _ = pose(-0.02)
    q0[i[:3]] = p0
    q0[i[3]] = -0.02
    waypoints = [pose(a) for a in (0.0, 0.2, 0.4, 0.6, angle)]
    return PC.replay_arc_push(sampler, "right", q0, pose(-0.02), waypoints, [0.0, 1.0, 0.0], None, [], [], 0.015, 0.1, lag_allowance_rad=lag)


def test_replay_lag_allowance_rejects_an_arc_that_ends_at_the_reach_end(monkeypatch):
    # the arm reaches 0.85 rad round the pivot: the 0.8 rad arc (band 0.015) replays as today, not with 0.08 rad of lag on top
    ok, why, rep = _replay(monkeypatch, 0.85, 0.0)
    assert ok, why
    assert rep["arc"]["stop"] == "joint_at_target" and "lag_allowance_rad" not in rep["arc"]
    ok, why, rep = _replay(monkeypatch, 0.85, 0.08)
    assert not ok and why.startswith("approach_untrackable") and "lag allowance" in why
    # an arm that reaches 0.95 rad tracks the allowance (swept >= 0.8 + 0.08) and the retreat leaves from there
    ok, why, rep = _replay(monkeypatch, 0.95, 0.08)
    assert ok, why
    assert rep["arc"]["swept_rad"] >= 0.88 - 1e-3 and rep["arc"]["lag_allowance_rad"] == 0.08
