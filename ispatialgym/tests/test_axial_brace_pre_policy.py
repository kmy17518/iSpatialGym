"""i7-axialrest (owner 2026-11-08 5v.2): the stop brace's pre pose checked at sampling with the program check's own call behind
``ISPATIALGYM_AXIAL_BRACE_PRE_POLICY`` (default off) - simulator-free: the call matches the program check's brace ``pre`` target, a
support-policy refusal fails the candidate, any other IK failure is counted and not refused, the remembered solutions are restored."""

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


def test_the_switches_are_off_by_default(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_PRE_POLICY", raising=False)
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_PROBE", raising=False)
    assert not AX.axial_brace_pre_policy_on() and not AX.axial_brace_probe_on()
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_PRE_POLICY", "1")
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_PROBE", "1")
    assert AX.axial_brace_pre_policy_on() and AX.axial_brace_probe_on()


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


def test_another_ik_failure_is_counted_not_refused():
    s, p, target, _ = _stub(IK.IKResult(False, None, "exact IK failed at pose 1 of 1"))
    assert s._brace_pre_policy("manip", p, target) == ""
    assert s.kit.stats["axial_brace.pre_policy.other_fail"] == 1 and "axial_brace.pre_policy.refused" not in s.kit.stats
