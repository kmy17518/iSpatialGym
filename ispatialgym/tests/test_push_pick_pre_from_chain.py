"""i7-startdraw2 r2 (lead #71, the transit-start gap): ``ISPATIALGYM_PUSH_PICK_PRE_FROM_CHAIN``. In 8 of 32 inside_large_01 demos the program
check's fresh solve of the overhang push's pre-contact witness landed ~3 rad (another IK branch) from the sampled chain's pre-contact
configuration while the push waypoints' witnesses stayed the chain's: the oracle reached the pre-contact in the other branch, pushed,
retreated and began the grasp transit there - a branch the push chain, the escape replay and every transit prediction never saw. With the
switch the chain's configuration is the pre witness whenever it still reaches the pose collision-free."""

from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import program_check as PC

ENV = "ISPATIALGYM_PUSH_PICK_PRE_FROM_CHAIN"


def _sampler(ok=True):
    seen = {}

    def cfa(q, pose, arm, ignore=()):
        seen["ignore"] = list(ignore)
        seen["q"] = q.clone()
        return ok

    ikc = SimpleNamespace(q_locked=lambda st, fingers=None: th.zeros(28), collision_free_at=cfa)
    kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(11, 24, 2), "left": th.arange(10, 23, 2)}), stats={})
    return SimpleNamespace(ikc=ikc, kit=kit), seen


def _action(chain=True):
    extra = {"kind": "pick_overhang", **({"chain_arm_q": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [9.0] * 7]} if chain else {})}
    return SimpleNamespace(name="push", contact=SimpleNamespace(extra=extra))


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert PC.push_pick_pre_from_chain_on()  # default on since the r3 A/B (lead #73)
    monkeypatch.setenv(ENV, "0")
    assert not PC.push_pick_pre_from_chain_on()
    monkeypatch.setenv(ENV, "1")
    assert PC.push_pick_pre_from_chain_on() and PC._pick_overhang_push(_action())


def test_chain_configuration_is_the_witness_when_it_reaches():
    s, seen = _sampler(ok=True)
    res = PC._push_pick_pre_from_chain(s, _action(), "manipulation", "right", "pose", 0.05, ["desk"])
    assert res is not None and res.ok
    assert th.allclose(res.q[th.arange(11, 24, 2)], th.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])) and float(res.q[10]) == 0.0
    assert seen["ignore"] == ["desk"] and s.kit.stats == {"push_pick_pre.chain": 1}


def test_falls_back_to_the_fresh_solve():
    s, _ = _sampler(ok=False)
    assert PC._push_pick_pre_from_chain(s, _action(), "manipulation", "right", "pose", 0.05, []) is None and s.kit.stats == {"push_pick_pre.fresh": 1}
    s, _ = _sampler(ok=True)
    assert PC._push_pick_pre_from_chain(s, _action(chain=False), "manipulation", "right", "pose", 0.05, []) is None
    assert s.kit.stats == {"push_pick_pre.no_chain": 1}
