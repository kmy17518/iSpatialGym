"""i7-contact3 (owner 2026-11-08 answers b, item 2): ``Executor.push_outcome`` accepts an in-tolerance push braced by a top press
(``ISPATIALGYM_TOP_PRESS_ACCEPT``) - no improvement rung, so ``offset_brace`` never raises the press; the ladder is unchanged for every
other brace kind and for every push out of tolerance. Simulator-free (a stub executor)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ispatialgym.oracle import executor as E

TOL = 0.1  # close_02's lid: the carton body's orientation tolerance


def _me():
    target = SimpleNamespace(orientation_tolerance=TOL, position_tolerance=0.01)
    return SimpleNamespace(state=SimpleNamespace(pose_target=lambda name: target), isg={"goal_first_step": None}, cfg=E.ExecutorConfig(), counters={})


def _outcome(me, residual, kind="top_press", hits=(120,), brace=True):
    push = SimpleNamespace(contact=SimpleNamespace(object="carton"))
    b = SimpleNamespace(contact=SimpleNamespace(kind=kind, object="carton")) if brace else None
    ctx = {"push_residual": {"final": residual, "kind": "arc", "angle_sign": 1.0}, "watch_hits": list(hits)}
    return E.Executor.push_outcome(me, push, ctx, brace=b)


@pytest.fixture
def switch(monkeypatch):
    def set_(on):
        monkeypatch.setenv(E.TOP_PRESS_ACCEPT_ENV, "1" if on else "0")
    return set_


def test_default_and_env(monkeypatch):
    monkeypatch.delenv(E.TOP_PRESS_ACCEPT_ENV, raising=False)
    assert E.top_press_accept_on() == (E.TOP_PRESS_ACCEPT_DEFAULT == "1")


def test_an_in_tolerance_top_press_is_accepted(switch):
    switch(True)
    me = _me()
    out = _outcome(me, 0.014)  # close_02's rung 0: +0.014 rad, inside 0.1 but outside the aim -0.05 +- 0.04
    assert out["in_tolerance"] and not out["aimed"]
    assert out["reason"] is None and "accepted" in out
    assert me.counters == {"push_outcome.top_press_in_tolerance_accepted": 1}
    # switch off: today's ladder (a brace hit before the goal -> "short", the press would be moved outward = up)
    switch(False)
    me = _me()
    out = _outcome(me, 0.014)
    assert out["reason"] == "short" and "accepted" not in out and me.counters == {}


def test_an_out_of_tolerance_top_press_still_retries(switch):
    switch(True)
    me = _me()
    assert _outcome(me, 0.15)["reason"] == "short"  # short beyond the tolerance, the press touched: the ladder
    assert _outcome(me, -0.15)["reason"] == "overshoot"
    assert me.counters == {}


@pytest.mark.parametrize("kind", ["opposite_face", "wide_stop", "below_high", "stop_edge", "row_left"])
def test_other_brace_kinds_are_unchanged(switch, kind):
    for on in (False, True):
        switch(on)
        me = _me()
        out = _outcome(me, 0.014, kind=kind)
        assert out["reason"] == "short" and "accepted" not in out and me.counters == {}
    # and without a brace argument (callers of before)
    switch(True)
    me = _me()
    assert _outcome(me, 0.014, brace=False)["reason"] == "short"


def test_an_aimed_push_is_unchanged(switch):
    switch(True)
    me = _me()
    out = _outcome(me, -0.05)  # at the aim centre: accepted by the aim, as before (not counted as a top-press acceptance)
    assert out["reason"] is None and out["aimed"] and "accepted" not in out and me.counters == {}
