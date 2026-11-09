"""i7-ul02 2026-11-09 ``ISPATIALGYM_PUSH_CHAIN_LIVE_B`` (default on since lead #75e): the push-under C loop skips a draw whose B key already failed before
solving its push chain, and ranks the pre-screen's yaws among those that still hold a live B pair (:func:`iors_large.live_yaws_at_b`)."""
import math
from types import SimpleNamespace

from ispatialgym.sampling.v2 import stance_memo as SM
from ispatialgym.sampling.v2.tasks import iors_large as IL


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv(IL.PUSH_CHAIN_LIVE_B_ENV, raising=False)
    assert IL.push_chain_live_b_on()
    for v in ("1", "on", "true"):
        monkeypatch.setenv(IL.PUSH_CHAIN_LIVE_B_ENV, v)
        assert IL.push_chain_live_b_on()
    monkeypatch.setenv(IL.PUSH_CHAIN_LIVE_B_ENV, "0")
    assert not IL.push_chain_live_b_on()


def test_b_key_is_the_draw_loops_key():
    step = math.pi / 8
    g = SimpleNamespace(name="top_2")
    assert IL.b_key(2 * step + 0.01, g, step) == (2, "top_2")
    assert IL.b_key(2 * step, g, step, y_end=0.123456) == (0.1235, "top_2")
    assert IL.b_key(0.0, "raw", step) == (0, "raw")


def test_live_yaws_at_b():
    step = math.pi / 8
    g0, g1 = SimpleNamespace(name="g0"), SimpleNamespace(name="g1")
    F = [(0.0, g0), (step, g0), (step, g1), (2 * step, g1)]
    # yaw 0: its only pair failed; yaw step: one of two pairs failed, the other untried; yaw 2 step: solved
    cache = {(0, "g0"): None, (1, "g0"): None, (2, "g1"): {"yaw": 2 * step}}
    assert IL.live_yaws_at_b(F, cache, step) == {step, 2 * step}
    assert IL.live_yaws_at_b(F, {**cache, (1, "g1"): None}, step) == {2 * step}
    assert IL.live_yaws_at_b(F, {}, step) == {0.0, step, 2 * step}
    assert IL.live_yaws_at_b([], cache, step) == set()
    # aligned init yaws: the key is the aligned yaw; False = no draw
    assert IL.live_yaws_at_b(F, {(0.5, "g0"): None, (0.5, "g1"): None}, step, end_aligned=lambda y: 0.5) == set()
    assert IL.live_yaws_at_b(F, {}, step, end_aligned=lambda y: False) == set()
    # consistent with dead_at_b: a C is dead exactly when no yaw is live
    for c in (cache, {**cache, (1, "g1"): None, (2, "g1"): None}, {}):
        assert IL.dead_at_b(F, c, step) == (not IL.live_yaws_at_b(F, c, step))


def test_stance_memo_class():
    assert SM.c_class("C (d 0.90 λ +1.21): skipped - every F_init pair at its 4 push yaws failed its exact IK at B (PUSH_CHAIN_LIVE_B)") == "dead_at_b"
