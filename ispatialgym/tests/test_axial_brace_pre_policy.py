"""i7-axialrest (owner 2026-11-08 5v.2): the stop brace's pre pose checked at sampling with the program check's own call behind
``ISPATIALGYM_AXIAL_BRACE_PRE_POLICY`` (default ON since owner 5y.1; ``=0`` off) - simulator-free: the call matches the program check's brace ``pre`` target, a
support-policy refusal fails the candidate, and (owner 5y.1) so does any other failure of the call (a joint limit, no IK), the remembered
solutions are restored."""

from __future__ import annotations

from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.tasks import alignment_axial as AX


class _Ikc:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.solutions = {("left", (1.0,)): th.zeros(3)}

    def exact(self, poses, stance, arm="right", ignore_for_last=(), **kw):
        self.calls.append({"poses": poses, "stance": stance, "arm": arm, "ignore_for_last": list(ignore_for_last), **kw})
        self.solutions[(arm, (9.0,))] = th.ones(3)  # a fresh solve writes its witness
        return self.result


def _stub(result):
    target, table = SimpleNamespace(name="tablespoon"), SimpleNamespace(name="table")
    s = SimpleNamespace(ikc=_Ikc(result), kit=SimpleNamespace(stats={}), gp=SimpleNamespace(variant="v"), support_objects=lambda: [table])
    s._brace_pre_policy = AX.AlignmentAxial._brace_pre_policy.__get__(s)
    pre, contact = (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), (th.ones(3), th.tensor([0.0, 0.0, 0.0, 1.0]))
    p = SimpleNamespace(label="left_h-", payload=(pre, contact, "left_h-", {}), chain=[pre, contact])
    return s, p, target, table


def test_the_pre_policy_is_on_by_default_and_the_probe_off(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_PRE_POLICY", raising=False)
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_PROBE", raising=False)
    assert AX.axial_brace_pre_policy_on() and not AX.axial_brace_probe_on()  # owner 5y.1: the pre policy is the default
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_PRE_POLICY", "0")
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_PROBE", "1")
    assert not AX.axial_brace_pre_policy_on() and AX.axial_brace_probe_on()


def test_the_check_is_the_program_checks_pre_call():
    s, p, target, table = _stub(IK.IKResult(True, th.zeros(3), ""))
    assert s._brace_pre_policy("manip", p, target) == ""
    (c,) = s.ikc.calls
    assert c["arm"] == "left" and c["stance"] == "manip" and len(c["poses"]) == 1 and c["poses"][0] is p.chain[0]
    assert target in c["ignore_for_last"] and table in c["ignore_for_last"]  # program check: ignore_last + ignore_all + approached
    assert c["approached"] == [target] and c["fingers"] == 0.0 and c["cache"] is False and "q_seed" not in c  # closed stop, the stance's seed
    assert s.kit.stats == {"axial_brace.pre_policy.checked": 1, "axial_brace.pre_policy.passed": 1}
    assert list(s.ikc.solutions) == [("left", (1.0,))]  # the brace chain's witnesses stay the recorded ones


def test_a_support_policy_failure_refuses_the_candidate():
    s, p, target, _ = _stub(IK.IKResult(False, None, "support_policy: non-finger link touches the support"))
    why = s._brace_pre_policy("manip", p, target)
    assert why.startswith("brace pre_policy: pre: support_policy: non-finger link touches the support")
    assert s.kit.stats["axial_brace.pre_policy.refused"] == 1 and s.kit.stats["axial_brace.pre_policy.refused.left_h-"] == 1
    assert s.kit.stats["axial_brace.pre_policy.refused_reason.support_policy"] == 1


def test_a_joint_limit_failure_is_refused():
    """owner 2026-11-08 5y.1: the pre call refuses on ANY failure, e.g. axial_03 310002 att 0's opp_blade_45- (i7ar_pol)."""
    reason = ("ik_joint_limit: left_arm_joint6 within 47 mrad of its limit at the pose (every solution found; the approach from it would "
              "stall)")
    s, p, target, _ = _stub(IK.IKResult(False, None, reason))
    why = s._brace_pre_policy("manip", p, target)
    assert why == f"brace pre_policy: pre: {reason}"
    st = s.kit.stats
    assert st["axial_brace.pre_policy.refused"] == 1 and st["axial_brace.pre_policy.refused.left_h-"] == 1
    assert st["axial_brace.pre_policy.refused_reason.ik_joint_limit"] == 1 and "axial_brace.pre_policy.passed" not in st
    assert list(s.ikc.solutions) == [("left", (1.0,))]


def test_any_other_ik_failure_is_refused():
    s, p, target, _ = _stub(IK.IKResult(False, None, "exact IK failed at pose 1 of 1"))
    assert s._brace_pre_policy("manip", p, target) == "brace pre_policy: pre: exact IK failed at pose 1 of 1"
    assert s.kit.stats["axial_brace.pre_policy.refused_reason.exact_ik_failed_at_pose_1_of_1"] == 1
