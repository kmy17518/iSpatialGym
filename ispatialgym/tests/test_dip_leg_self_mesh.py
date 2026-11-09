"""i7-artlarge r3 (lead #54): ``ISPATIALGYM_DIP_LEG_SELF_MESH`` - the transit dip gate's Cartesian leg refused on the arm's link3 / link5
collision-mesh clearance (pair_mesh). Simulator-free: pair_clearance stubbed."""

from __future__ import annotations

import torch as th

from ispatialgym.oracle import motions as MO
from ispatialgym.sampling.v2 import pair_mesh as PM


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_DIP_LEG_SELF_MESH", raising=False)
    assert MO.dip_leg_self_mesh_on() is True
    monkeypatch.setenv("ISPATIALGYM_DIP_LEG_SELF_MESH", "0")
    assert MO.dip_leg_self_mesh_on() is False


def test_why_refuses_at_the_margin_and_checks_the_last(monkeypatch):
    calls = []

    def clearance(q, arm, links):
        calls.append(int(q[0]))
        assert links == (f"{arm}_arm_link3", f"{arm}_arm_link5")
        return (0.0001 if int(q[0]) == 6 else 0.02), "x"

    monkeypatch.setattr(PM, "pair_clearance", clearance)
    q = th.zeros(7, 28)
    q[:, 0] = th.arange(7, dtype=th.float32)
    why = MO.dip_leg_self_mesh_why("left", q, stride=4)
    assert calls == [0, 4, 6] and why is not None and "left_arm_link3 / left_arm_link5" in why and "waypoint 6/7" in why
    calls.clear()
    monkeypatch.setattr(PM, "pair_clearance", lambda q, arm, links: (0.011, "x"))
    assert MO.dip_leg_self_mesh_why("right", q, stride=2) is None
