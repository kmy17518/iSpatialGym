"""i7-attachmargin (owner decision A8d (b)): ``ISPATIALGYM_ATTACH_CARRY_END_MARGIN`` - a carry whose tracked end has an arm joint within the
margin of its limit fails like an untrackable carry (the next carry of the ladder is tried); default off, ``shadow:<rad>`` counts only."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.tasks import attachment as AT

GRASP_Z = 0.85
PRE = (th.tensor([0.95, -0.30, 1.10]), C.IDENTITY_QUAT.clone())


def _sampler(end_margins):
    """A carry ladder whose i-th passing chain ends ``end_margins[i]`` rad from the nearest limit (joint ``right_arm_joint5``)."""
    s = object.__new__(AT.Attachment)
    s.t_role, s.p_role = "target", "partner"
    s.gen = lambda key, default=None: default
    s.g = lambda key: {"reorientation_waypoints": 3}[key]
    s.obj = lambda role: NS(name=role)
    s.attach_route = None
    s.gp = NS(variant="v")
    s.kit = NS(stats={}, robot=NS(arm_control_idx={"right": list(range(7))}))
    chains = []

    def waypoints_ok(chain, stance, arm, ignore_all=(), track_dense=False):
        chains.append(chain)
        q = th.zeros(7)
        q[0] = float(end_margins[len(chains) - 1])  # the margin, read back by joint_limit_margin below
        return IK.IKResult(True, q, "screen_mid_exact_ends+tracked", len(chain))

    s.ikc = NS(waypoints_ok=waypoints_ok, joint_limit_margin=lambda q, arm: (float(q[0]), "right_arm_joint5"))
    grasp = NS(grasp=(th.tensor([0.80, 0.0, GRASP_Z]), C.IDENTITY_QUAT.clone()), eef_in_object=(th.tensor([0.0, 0.0, 0.044]), C.IDENTITY_QUAT.clone()))
    f_pose = (th.tensor([0.95, -0.30, 0.81]), C.IDENTITY_QUAT.clone())
    return s, grasp, f_pose, chains


def _carry(s, grasp, f_pose):
    return s._carry_to_pre_attach(grasp, PRE, f_pose, NS(name="manipulation"), th.zeros(3))


def test_switch_parsing(monkeypatch):
    monkeypatch.delenv(AT.ATTACH_CARRY_END_MARGIN_ENV, raising=False)
    assert AT.ATTACH_CARRY_END_MARGIN_DEFAULT == "0" and AT.attach_carry_end_margin() == (0.0, False)
    for v, want in (("0", (0.0, False)), ("off", (0.0, False)), ("0.02", (0.02, False)), ("shadow:0.02", (0.02, True)), (" SHADOW:0.05 ", (0.05, True))):
        monkeypatch.setenv(AT.ATTACH_CARRY_END_MARGIN_ENV, v)
        assert AT.attach_carry_end_margin() == want
    for bad in ("shadow", "x", "-0.1", "1.0"):
        monkeypatch.setenv(AT.ATTACH_CARRY_END_MARGIN_ENV, bad)
        with pytest.raises(ValueError):
            AT.attach_carry_end_margin()
    assert AT.carry_end_margin_bucket(0.001) == "lt5mrad" and AT.carry_end_margin_bucket(0.015) == "lt20mrad" and AT.carry_end_margin_bucket(0.2) == "ge50mrad"


def test_off_takes_the_first_passing_carry_and_records_its_end(monkeypatch):
    monkeypatch.delenv(AT.ATTACH_CARRY_END_MARGIN_ENV, raising=False)
    s, grasp, f_pose, chains = _sampler([0.001, 0.3, 0.3])
    plan = _carry(s, grasp, f_pose)
    assert plan.result.ok and len(chains) == 1 and plan.end_limit["margin_rad"] == 0.001 and plan.end_limit["joint"] == "right_arm_joint5"
    assert s.kit.stats == {"carry_end_margin.current.lt5mrad": 1}


def test_on_rejects_an_end_at_the_limit_and_tries_the_next_carry(monkeypatch):
    monkeypatch.setenv(AT.ATTACH_CARRY_END_MARGIN_ENV, "0.02")
    s, grasp, f_pose, chains = _sampler([0.001, 0.015, 0.3])
    plan = _carry(s, grasp, f_pose)
    assert plan.result.ok and len(chains) == 3 and plan.carry == AT.CARRY_HEIGHTS[2] and plan.end_limit["margin_rad"] == 0.3
    k = s.kit.stats
    assert k["attach_carry_end_margin.current.checked"] == 3 and k["attach_carry_end_margin.current.rejected"] == 2
    assert k["attach_carry_end_margin.current.rejected.right_arm_joint5"] == 2 and k["attach_carry_end_margin.current.rescued"] == 1
    # every carry at the limit: the last one tried fails with the margin's reason
    s, grasp, f_pose, chains = _sampler([0.001, 0.001, 0.001])
    plan = _carry(s, grasp, f_pose)
    assert not plan.result.ok and plan.result.reason.startswith("carry_end_limit") and len(chains) == 3
    assert "attach_carry_end_margin.current.rescued" not in s.kit.stats


def test_shadow_counts_without_rejecting(monkeypatch):
    monkeypatch.setenv(AT.ATTACH_CARRY_END_MARGIN_ENV, "shadow:0.02")
    s, grasp, f_pose, chains = _sampler([0.001, 0.3, 0.3])
    plan = _carry(s, grasp, f_pose)
    assert plan.result.ok and len(chains) == 1
    assert s.kit.stats["attach_carry_end_margin.current.would_reject"] == 1 and "attach_carry_end_margin.current.rejected" not in s.kit.stats
