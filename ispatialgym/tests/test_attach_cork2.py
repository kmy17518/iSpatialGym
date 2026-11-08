"""i7-cork round 2 (owner decisions 2026-11-08 5n.1-5n.3): the early rest-nudge test (ISG_ATTACH_EARLY_REST), the per-attempt fallback
routes (ISG_ATTACH_FALLBACK) and the lying route's min-turn partner yaw (ISG_LYING90_MIN_TURN)."""

import types

import pytest

pytest.importorskip("omnigibson")

from ispatialgym.sampling.tasks.base import Draft  # noqa: E402
from ispatialgym.sampling.v2.tasks import attachment as AT  # noqa: E402


def test_early_rest_mode(monkeypatch):
    monkeypatch.delenv(AT.EARLY_REST_SWITCH, raising=False)
    assert AT.early_rest_mode() == "off"
    for v, want in (("0", "off"), ("shadow", "shadow"), ("1", "on"), ("on", "on"), ("SHADOW", "shadow")):
        monkeypatch.setenv(AT.EARLY_REST_SWITCH, v)
        assert AT.early_rest_mode() == want
    assert issubclass(AT.EarlyRestUnstable, AT.Reject)


def _stub(early):
    s = AT.Attachment.__new__(AT.Attachment)
    s.t_role = "target"
    s.counters = {}
    s.kit = types.SimpleNamespace(stats={})
    s.gp = types.SimpleNamespace(variant="v")
    s._early_rest_last = early
    male = types.SimpleNamespace(name="cork_1")
    s.obj = lambda role: male if role == "target" else None
    return s


@pytest.mark.parametrize("early,late_bad,want", [(False, {}, "agree_stable"), (True, {"cork_1": 9.0}, "agree_unstable"),
                                                  (True, {}, "early_unstable_late_stable"), (False, {"cork_1": 5.0}, "early_stable_late_unstable")])
def test_early_vs_late_bookkeeping(early, late_bad, want):
    s = _stub({"route": "current", "tilt_deg": 7.0 if early else 0.1, "unstable": early})
    d = Draft()
    s._early_vs_late(d, {"cork_1": 9.0 if late_bad else 0.2}, late_bad)
    assert s.counters == {f"early_rest.vs_late.{want}": 1} and s.kit.stats == s.counters
    assert d.summary["early_rest"]["vs_late"] == want


def test_early_vs_late_partner_only_and_not_run():
    s = _stub({"route": "current", "tilt_deg": 0.1, "unstable": False})
    s._early_vs_late(Draft(), {"cork_1": 0.2, "bottle_1": 8.0}, {"bottle_1": 8.0})
    assert s.counters == {"early_rest.vs_late.agree_stable": 1, "early_rest.vs_late.late_partner_only": 1}
    s = _stub(None)
    d = Draft()
    s._early_vs_late(d, {"cork_1": 9.0}, {"cork_1": 9.0})
    assert s.counters == {} and "early_rest" not in d.summary
