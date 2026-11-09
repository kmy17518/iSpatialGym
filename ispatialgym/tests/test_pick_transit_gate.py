"""i7-startdraw2 (lead decision #49): the pick transit gate (``ISPATIALGYM_PICK_TRANSIT_GATE``). inside_large_01's demos failed at the
overhang pick's transit over the book (escape -> pre-grasp at stance B): the sampler never judged that move (OVERPATH off), the oracle's
straight ramps hit the desk / the book, its trajopt / PROTECT re-plans failed or its supports-ignored fallback ran into the desk. The gate
runs the oracle's own transit ladder kinematically WITHOUT trajectory optimisation: a pick passes only when a checked straight ramp (direct
or via a posture) carries the transit - the move the oracle then takes without planning. Toy backend of test_overhang_overpath (the arm's
first three joints are the hand position, one sphere; the book's box at its settled pose)."""

import torch as th

from ispatialgym.oracle import overpath as OP
from ispatialgym.tests.test_overhang_overpath import BOOK_P1, IL, _hb_hull, _p1_q0, _p1_sampler, _ToyCurobo  # noqa: F401

ENV = "ISPATIALGYM_PICK_TRANSIT_GATE"

def _gate(s, q_start, pre_pos):
    if not hasattr(s.ikc, "solutions"):
        s.ikc.solutions = {}
    s.ikc._pose_key = lambda p: tuple(round(float(v), 4) for v in th.as_tensor(p[0]).reshape(-1).tolist())
    s._overpath_start = lambda st, gst, arm, chain, settled: (q_start, "escape", "")
    s.support_objects = lambda: []
    s._pick_transit_gate = IL.IORSLarge._pick_transit_gate.__get__(s)
    pre = (th.tensor(pre_pos), th.tensor([0.0, 0.0, 0.0, 1.0]))
    st = type("St", (), {"name": "manipulation"})()
    return s._pick_transit_gate(st, st, "right", {}, BOOK_P1, pre, pre)

def test_switch_default_off_and_memo_key(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(OP.ENV, raising=False)
    assert not IL.pick_transit_gate_on()
    gen = {"overhang_fail_memo": True}
    s = type("S", (), {})()
    s.gp = type("GP", (), {"gen": staticmethod(lambda k, d=None: gen.get(k, d))})()
    st = type("St", (), {"base_pose2d": (0.0, 0.0, 0.0), "translate": 0.5, "tilt": 0.0})()
    key_off = IL.IORSLarge._overhang_memo_key(s, st, {}, ["right"], "side_0", 0.1, th.zeros(2), 0.8, False)
    monkeypatch.setenv(ENV, "1")
    key_on = IL.IORSLarge._overhang_memo_key(s, st, {}, ["right"], "side_0", 0.1, th.zeros(2), 0.8, False)
    assert key_on == key_off + ("transit_gate_v1",)  # the switch enters the failure memo's key only when on

def test_straight_only_motions_never_run_trajopt():
    from ispatialgym.sampling.v2 import overpath_planner as OPP

    b = _ToyCurobo(OP.box_of(BOOK_P1, _hb_hull()))
    robot = type("R", (), {"arm_control_idx": {"right": th.arange(7), "left": th.arange(7, 14)}})()
    m = OPP.plan_only_motions(robot, b, _p1_q0(), straight_only=True)
    r = m.backend.plan_joint("right_arm", _p1_q0())
    assert not r.success and "straight_only" in r.status and b.calls == []
    m2 = OPP.plan_only_motions(robot, b, _p1_q0())
    assert m2.backend.plan_joint("right_arm", _p1_q0(), max_attempts=1).success and b.calls == ["plan_joint"]

def test_gate_passes_a_clear_straight_ramp_and_rejects_what_only_trajopt_carries(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("ISPATIALGYM_FAST_STRAIGHT_ARM", "1")
    monkeypatch.setenv("ISPATIALGYM_FAST_STRAIGHT_ARM_VIA", "1")
    # far from the book on the same side: the straight ramp clears the world and the book -> the oracle takes it, the gate passes
    s = _p1_sampler()
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, -0.40, 0.98])
    facts, key, why = _gate(s, q0, [0.0, -0.30, 0.90])
    assert facts is not None and facts["branch"] == "straight", (key, why)
    assert "plan_joint" not in s.kit.backend.calls and "plan_pose" not in s.kit.backend.calls
    assert s.kit.stats.get("pick_transit_gate.ms", 0) >= 0
    s = _p1_sampler()
    facts, key, why = _gate(s, _p1_q0(), [0.0, -0.16, 0.79])  # across the book: the ramp via a posture clears it -> the oracle's straight_via
    assert facts is not None and facts["branch"] == "straight_via" and "plan_joint" not in s.kit.backend.calls
    # across the book (the live 6-9 mm pass), the via-posture ramps off (in the toy the postures lift the hand clear of the book, which
    # passes - test above): the direct ramp misses the book's clearance; the toy trajopt WOULD plan it - the gate rejects
    monkeypatch.setenv("ISPATIALGYM_FAST_STRAIGHT_ARM_VIA", "0")
    s = _p1_sampler()
    facts, key, why = _gate(s, _p1_q0(), [0.0, -0.16, 0.79])
    assert facts is None and key in ("no_straight", "book", "collision"), (key, why)
    assert "plan_joint" not in s.kit.backend.calls and "plan_pose" not in s.kit.backend.calls
    # the book is back where it was after the check
    assert th.allclose(s.target.pose[0], th.tensor([5.0, 5.0, -10.0]))

def test_gate_rejects_without_a_witness(monkeypatch):
    from ispatialgym.sampling.v2.ik import IKResult

    monkeypatch.setenv(ENV, "1")
    s = _p1_sampler()
    s.ikc.exact = lambda poses, st, arm, **kw: IKResult(False, None, "ik_reach: no solution", 1)
    facts, key, why = _gate(s, _p1_q0(), [0.0, -0.16, 0.79])
    assert facts is None and key == "witness"


def test_gate_judges_the_recorded_witness(monkeypatch):
    """The oracle's q_hint is the pick's recorded pre-grasp witness (``ikc.solutions`` of the [pre-grasp, grasp] chain): the gate plans to
    that configuration, a fresh IK only when none is recorded (kit pick_transit_gate.witness.recorded / .fresh_ik)."""
    monkeypatch.setenv(ENV, "1")
    seen = []
    from ispatialgym.sampling.v2 import overpath_planner as OPP

    monkeypatch.setattr(OPP, "plan_transit", lambda kit, arm, q0, pre, q_hint, *a, **k: seen.append(q_hint.clone()) or (None, {"why": "stub"}))
    s = _p1_sampler()
    q_rec = th.full((28,), 0.123)
    s.ikc.solutions = {("right", (0.0, -0.3, 0.9)): q_rec}
    _gate(s, _p1_q0(), [0.0, -0.30, 0.90])
    assert th.allclose(seen[-1], q_rec) and s.kit.stats["pick_transit_gate.witness.recorded"] == 1
    s.ikc.solutions = {}
    _gate(s, _p1_q0(), [0.0, -0.30, 0.90])
    assert not th.allclose(seen[-1], q_rec) and s.kit.stats["pick_transit_gate.witness.fresh_ik"] == 1
