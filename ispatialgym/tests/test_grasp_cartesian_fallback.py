"""oracle-fix 2026-11-06 (owner, inside_large#02: "when grasping the jar the hand seems to brush on the table's surface"): when no plan reaches
the pre-grasp with the supports present, a Cartesian leg tracked with them present (``Motions.cartesian_transit``: straight, else
up-over-down, in the witness's IK branch) comes before the supports-ignored plan (``ExecuteConfig.pre_grasp_cartesian_fallback``).
Toy kinematics (test_v2_oracle's fake robot / backend), no simulator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.motions import SkillError


def _setup(tmp_path):
    from ispatialgym.tests.test_v2_oracle import iors_world, make_executor

    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    arm = "right"
    p0, q0 = motions.eef_pose(arm)
    target = (p0 + th.tensor([0.30, 0.10, -0.06]), q0)  # the pre-grasp: across and a little lower than the hand
    q_end = env.robot.q.clone()
    q_end[env.robot.arm_control_idx[arm]] = env.robot.ik(arm, target)  # the sampler's witness of it
    return env, ex, motions, backend, arm, target, q_end


def _run(env, gen):
    out = []
    try:
        while True:
            env.step(next(gen))
    except StopIteration as stop:
        out.append(stop.value)
    return out[0] if out else None


def test_cartesian_transit_runs_the_straight_leg_with_the_world_synced(tmp_path):
    env, ex, motions, backend, arm, target, q_end = _setup(tmp_path)
    n0 = backend.world_updates
    rec = _run(env, motions.cartesian_transit(arm, target, q_end, approach=[(target[0] + th.tensor([0.0, 0.0, -0.05]), target[1])]))
    assert rec["leg"] == "straight" and backend.world_updates == n0 + 1
    assert float(th.linalg.norm(motions.eef_pose(arm)[0] - target[0])) < 5e-3
    assert "plan_pose" not in backend.calls and motions._dip is None  # no trajopt; the gate context restored


def test_cartesian_transit_raises_before_moving_when_no_leg_qualifies(tmp_path):
    env, ex, motions, backend, arm, target, q_end = _setup(tmp_path)
    p0 = motions.eef_pose(arm)[0].clone()
    backend.fail_ik_paths = 4  # straight and up-over-down, forward and backward
    with pytest.raises(SkillError) as e:
        _run(env, motions.cartesian_transit(arm, target, q_end))
    assert e.value.reason == "plan_failed" and "no Cartesian leg qualifies" in e.value.message and "up-over-down" in e.value.message
    assert float(th.linalg.norm(motions.eef_pose(arm)[0] - p0)) < 1e-6
    with pytest.raises(SkillError, match="no witness"):  # without the witness's IK branch: no leg at all
        _run(env, motions.cartesian_transit(arm, target, None))


def _grasp_transit(env, ex, motions, backend, arm, target, q_end, monkeypatch, cfg=None):
    def no_plan_with_supports(*a, **kw):
        raise SkillError("plan_failed", "every plan with the supports present failed (fake)", stage="trajopt")
        yield  # noqa: unreachable - a generator

    monkeypatch.setattr(EX, "reach_pre_pose", no_plan_with_supports)
    table = SimpleNamespace(name="breakfast_table_0")
    action = SimpleNamespace(grasp=SimpleNamespace(extra={}))
    ctx = {"witnesses": {"pre_grasp": q_end}, **({"cfg": cfg} if cfg is not None else {})}
    _run(env, EX._grasp_transit(action, motions, ex.state, {"pre_grasp": target, "approach": []}, ctx, arm, [table], []))
    return ctx


def test_grasp_transit_takes_the_cartesian_leg_before_ignoring_the_supports(tmp_path, monkeypatch):
    env, ex, motions, backend, arm, target, q_end = _setup(tmp_path)
    ctx = _grasp_transit(env, ex, motions, backend, arm, target, q_end, monkeypatch)
    assert ctx["pre_grasp_transit"].startswith("cartesian straight leg, supports present (fallback: every plan")
    assert ctx["pre_grasp_cartesian"]["ran"] and ctx["pre_grasp_cartesian"]["leg"] == "straight"
    assert not {"plan_pose", "plan_joint"} & set(backend.calls)  # the supports-ignored plan never ran
    assert float(th.linalg.norm(motions.eef_pose(arm)[0] - target[0])) < 5e-3


def test_grasp_transit_ignores_the_supports_only_when_no_leg_qualifies_or_the_knob_is_off(tmp_path, monkeypatch):
    env, ex, motions, backend, arm, target, q_end = _setup(tmp_path)
    backend.fail_ik_paths = 4
    ctx = _grasp_transit(env, ex, motions, backend, arm, target, q_end, monkeypatch)
    assert ctx["pre_grasp_transit"].startswith("supports ignored (fallback:") and not ctx["pre_grasp_cartesian"]["ran"]
    assert backend.calls[-1] == "plan_joint"  # the supports-ignored plan to the witness, after the 4 failed legs
    env, ex, motions, backend, arm, target, q_end = _setup(tmp_path)
    ctx = _grasp_transit(env, ex, motions, backend, arm, target, q_end, monkeypatch, cfg=EX.ExecuteConfig(pre_grasp_cartesian_fallback=False))
    assert ctx["pre_grasp_transit"].startswith("supports ignored (fallback:") and "pre_grasp_cartesian" not in ctx
    assert "solve_ik_path" not in backend.calls  # as before: straight to the supports-ignored plan
