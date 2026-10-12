"""Attachment's carry from the lifted grasp to the pre-attach (``Attachment._carry_to_pre_attach``): by default the lift ladder and the
straight SLERP line, as before; with ``generation.carry_over_partner_m`` (owner 2026-10-12 attachment#04 3b, the lid-stockpot aliases)
the 12 cm lift, a vertical rise until the male's lowest point clears the partner's top by the clearance, then the line - a ladder of
clearances tried in order, a carry whose fingertips leave the image not tried."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.tasks import attachment as AT


def _box(lo, hi):
    return th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=th.float32)


GRASP_Z = 0.85  # eef 4.4 cm above the lid's bottom (a knob pinch) on a table at 0.81
PRE = (th.tensor([0.95, -0.30, 1.10]), C.IDENTITY_QUAT.clone())


def _sampler(gen, ok=lambda chain: True):
    """A lid (22 cm, 5 cm tall, root at its bottom) grasped from above at x 0.80 and a 14.3 cm pot whose root stands on the table at
    (0.95, -0.30); ``ok(chain)`` decides each tracked chain. Returns the sampler, the grasp, the partner pose and the chains tried."""
    s = object.__new__(AT.Attachment)
    s.t_role, s.p_role = "target", "partner"
    s.hull = {"target": _box((-0.11, -0.11, 0.0), (0.11, 0.11, 0.05)), "partner": _box((-0.146, -0.114, 0.0), (0.146, 0.114, 0.143))}
    s.gen = lambda key, default=None: gen.get(key, default)
    s.g = lambda key: {"reorientation_waypoints": 3}[key]
    s.obj = lambda role: NS(name=role)
    chains = []

    def waypoints_ok(chain, stance, arm, ignore_all=(), track_dense=False):
        chains.append([(p[0].clone(), p[1].clone()) for p in chain])
        good = ok(chain)
        return IK.IKResult(good, None, "screen_mid_exact_ends" if good else "approach_untrackable: collides", len(chain))

    s.ikc = NS(waypoints_ok=waypoints_ok)
    grasp = NS(grasp=(th.tensor([0.80, 0.0, GRASP_Z]), C.IDENTITY_QUAT.clone()), eef_in_object=(th.tensor([0.0, 0.0, 0.044]), C.IDENTITY_QUAT.clone()))
    f_pose = (th.tensor([0.95, -0.30, 0.81]), C.IDENTITY_QUAT.clone())
    return s, grasp, f_pose, chains


def _z(p):
    return round(float(p[0][2]), 4)


def test_carry_over_ladder_reads_the_generation_key():
    assert AT.carry_over_ladder(None) == () and AT.carry_over_ladder(False) == () and AT.carry_over_ladder(True) == ()
    assert AT.carry_over_ladder([]) == () and AT.carry_over_ladder(0) == () and AT.carry_over_ladder(0.0) == ()
    assert AT.carry_over_ladder(0.2) == (0.2,) and AT.carry_over_ladder([0.2, 0.12, 0.06]) == (0.2, 0.12, 0.06)
    assert AT.carry_over_ladder([0.2, None, 0]) == (0.2,)


def test_default_carry_is_the_lift_ladder_and_the_straight_line():
    s, grasp, f_pose, chains = _sampler({}, ok=lambda chain: abs(float(chain[0][0][2]) - (GRASP_Z + 0.05)) < 1e-6)
    plan = s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))
    assert plan.result.ok and plan.carry == 0.05 and plan.over is None and plan.ladder == f"carry ladder {AT.CARRY_HEIGHTS}"
    assert [_z(c[0]) for c in chains] == [round(GRASP_Z + h, 4) for h in AT.CARRY_HEIGHTS]
    for chain, h in zip(chains, AT.CARRY_HEIGHTS):
        lift = (grasp.grasp[0] + th.tensor([0.0, 0.0, h]), grasp.grasp[1])
        line = C.line_waypoints(lift, PRE, 4)
        assert len(chain) == 5 and all(float(th.linalg.norm(a[0] - b[0])) < 1e-6 for a, b in zip(chain[1:], line))
    assert len(plan.waypoints) == 4 and float(th.linalg.norm(plan.waypoints[-1][0] - PRE[0])) < 1e-6
    # every carry fails: the last one tried is returned
    s, grasp, f_pose, chains = _sampler({}, ok=lambda chain: False)
    plan = s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))
    assert not plan.result.ok and plan.carry == 0.05 and len(chains) == 3


def test_carry_over_the_partner_rises_until_the_male_clears_its_top():
    s, grasp, f_pose, chains = _sampler({"carry_over_partner_m": [0.2, 0.06]}, ok=lambda chain: len(chains) == 2)
    s._carry_out_of_frame = lambda st, wps: None
    plan = s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))
    lift_z = GRASP_Z + max(AT.CARRY_HEIGHTS)
    pot_top = 0.81 + 0.143
    # first the 0.20 m clearance (failed), then 0.06 m: lift 12 cm, rise straight up, then the line to the pre-attach
    assert len(chains) == 2 and plan.result.ok and plan.carry == max(AT.CARRY_HEIGHTS)
    for chain, clr in zip(chains, (0.2, 0.06)):
        rise_z = pot_top + clr + 0.044  # the lid's bottom 4.4 cm below the eef
        lift, rise = chain[0], chain[1]
        assert _z(lift) == round(lift_z, 4) and _z(rise) == pytest.approx(rise_z, abs=1e-4)
        assert float(th.linalg.norm(rise[0][:2] - lift[0][:2])) < 1e-6 and float(th.linalg.norm(rise[1] - lift[1])) < 1e-6
        line = C.line_waypoints(rise, PRE, 4)
        assert len(chain) == 6 and all(float(th.linalg.norm(a[0] - b[0])) < 1e-5 for a, b in zip(chain[2:], line))
    assert len(plan.waypoints) == 5 and all(float(th.linalg.norm(a[0] - b[0])) < 1e-6 for a, b in zip(plan.waypoints, chains[1][1:]))
    assert plan.over["clearance_m"] == 0.06 and plan.over["partner_top_z"] == pytest.approx(pot_top, abs=1e-4)
    assert plan.over["male_low_below_eef_m"] == pytest.approx(0.044, abs=1e-4) and plan.over["rise_m"] == pytest.approx(pot_top + 0.06 + 0.044 - lift_z, abs=1e-4)
    assert plan.ladder == "carry over the partner [0.2, 0.06] m"


def test_carry_over_the_partner_skips_a_carry_out_of_the_image_and_needs_no_rise_above_it():
    s, grasp, f_pose, chains = _sampler({"carry_over_partner_m": [0.2, 0.06]})
    seen = []
    s._carry_out_of_frame = lambda st, wps: seen.append(_z(wps[0])) or ({"key": "waypoints", "index": 0} if len(seen) == 1 else None)
    plan = s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))
    assert len(seen) == 2 and len(chains) == 1 and plan.over["clearance_m"] == 0.06 and plan.result.ok
    # a partner lower than the lifted male: the carry starts at the lift (no rise pose), like the default's first try
    s, grasp, f_pose, chains = _sampler({"carry_over_partner_m": 0.02})
    s.hull["partner"] = _box((-0.1, -0.1, 0.0), (0.1, 0.1, 0.03))
    s._carry_out_of_frame = lambda st, wps: None
    plan = s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))
    assert len(chains[0]) == 5 and plan.over["rise_m"] == 0.0 and len(plan.waypoints) == 4
