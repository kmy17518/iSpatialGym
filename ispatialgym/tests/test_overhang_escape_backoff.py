"""i7-startdraw2 r2 (lead #71, the transit-start gap): ``ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF``. ``execute_push`` backs the pad off
``push_backoff_m`` (2 cm) along the reversed push before the retreat; the sampler's escape replay (``IORSLarge._overpath_start``, read by
the pick transit gate and OVERPATH) started at the push's goal - 2.0 cm deeper in every inside_large_01 demo, 0.12-0.20 rad from the
live retreat's end. With the switch the replay leaves from the backed-off pad, tracked from the goal configuration."""

from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import program_check as PCK
from ispatialgym.sampling.v2.ik import IKResult
from ispatialgym.sampling.v2.tasks import iors_large as IL

ENV = "ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF"


def _setup(monkeypatch):
    seen = {}

    def escape(s, arm, q, obj, sup, ign, dist, start="", escape_up_m=None):
        seen["q_escape_from"] = q.clone()
        return "replayed", "", None, q

    monkeypatch.setattr(PCK, "replay_contact_escape", escape)

    def cont(q_start, wps, arm, **kw):
        seen["back"] = wps[-1]
        seen["first"] = wps[0]
        q = q_start.clone()
        q[11] = 7.0
        return IKResult(True, q, "ok", 1)

    ikc = SimpleNamespace(q_locked=lambda st, fingers=None: th.zeros(28), continue_chain=cont)
    s = SimpleNamespace(kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(11, 24, 2)}), stats={}), ikc=ikc,
                        support_objects=lambda: ["desk"], gp=SimpleNamespace(gen=lambda k, d=None: d), target="book")
    st = SimpleNamespace(name="manipulation")
    chain = {"seed": {"chain_arm_q": [[0.0] * 7, [0.5] * 7]}, "direction": [-1.0, 0.0, 0.0],
             "goal": (th.tensor([23.73, 13.80, 0.79]), th.tensor([1.0, 0.0, 0.0, 0.0])), "opening": 0.05}
    return s, st, chain, seen


def test_off_replays_from_the_goal(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    s, st, chain, seen = _setup(monkeypatch)
    q0, start, why = IL.IORSLarge._overpath_start(s, st, st, "right", chain, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    assert q0 is not None and "back" not in seen and float(seen["q_escape_from"][11]) == 0.5


def test_on_replays_from_the_backed_off_pad(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    s, st, chain, seen = _setup(monkeypatch)
    q0, start, why = IL.IORSLarge._overpath_start(s, st, st, "right", chain, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    from ispatialgym.oracle.execute import CONFIG

    back_p = seen["back"][0]
    assert th.allclose(back_p, th.tensor([23.73 + CONFIG.push_backoff_m, 13.80, 0.79]), atol=1e-6)  # 2 cm back against the push
    assert seen["first"] is chain["goal"]  # tracked on from where the arm is (continue_chain's first pose)
    assert float(seen["q_escape_from"][11]) == 7.0 and s.kit.stats["overhang_escape_backoff.tracked"] == 1


def test_one_switch_for_the_sampler_and_the_program_check(monkeypatch):
    """r3 (lead #73): the program check's retreat after the overhang push replays the back-off under the same switch."""
    monkeypatch.delenv(ENV, raising=False)
    assert PCK.overhang_escape_backoff_on() and IL.overhang_escape_backoff_on()  # default on since the r3 A/B (lead #73)
    monkeypatch.setenv(ENV, "0")
    assert not PCK.overhang_escape_backoff_on() and not IL.overhang_escape_backoff_on()
    monkeypatch.setenv(ENV, "1")
    assert PCK.overhang_escape_backoff_on() and IL.overhang_escape_backoff_on()
