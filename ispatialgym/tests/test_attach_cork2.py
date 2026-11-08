"""i7-cork round 2 (owner decisions 2026-11-08 5n.1-5n.3): the early rest-nudge test (ISG_ATTACH_EARLY_REST), the per-attempt fallback
routes (ISG_ATTACH_FALLBACK) and the lying route's min-turn partner yaw (ISG_LYING90_MIN_TURN)."""

import math
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


def _rand_quat(g):
    import torch as th

    q = th.tensor(g.normal(size=4), dtype=th.float64)
    return q / th.linalg.norm(q)


def test_min_turn_yaw_matches_brute_force():
    import numpy as np

    from ispatialgym.robot import conventions as C

    g = np.random.default_rng(3)
    for _ in range(20):
        ql, qp = _rand_quat(g), _rand_quat(g)
        yaw, turn = AT.min_turn_yaw(ql, qp)
        brute = min(C.quat_angle(ql, C.qmul(C.quat_from_yaw(y), qp)) for y in np.linspace(-math.pi, math.pi, 7201))
        assert turn == pytest.approx(brute, abs=2e-3)
        assert C.quat_angle(ql, C.qmul(C.quat_from_yaw(yaw), qp)) == pytest.approx(turn, abs=1e-4)
        assert -math.pi <= yaw < math.pi


def test_min_turn_yaw_of_a_lying_male_is_its_tilt():
    """A male lying on its side (a 90 deg tilt of the upright goal orientation, any yaw): the smallest carry turn is the 90 deg tilt,
    a uniform partner yaw gives up to 180 deg (the recorded lying turns 125-179 deg)."""
    import numpy as np
    import torch as th

    from ispatialgym.robot import conventions as C

    g = np.random.default_rng(5)
    for _ in range(10):
        qp = _rand_quat(g)
        y0 = float(g.uniform(-math.pi, math.pi))
        tilt = C.quat_from_axis_angle(th.tensor([math.cos(y0 * 0.3), math.sin(y0 * 0.3), 0.0]), math.pi / 2)
        ql = C.qmul(tilt, C.qmul(C.quat_from_yaw(y0), qp))
        yaw, turn = AT.min_turn_yaw(ql, qp)
        assert math.degrees(turn) == pytest.approx(90.0, abs=1e-3)
        worst = max(C.quat_angle(ql, C.qmul(C.quat_from_yaw(y), qp)) for y in np.linspace(-math.pi, math.pi, 721))
        assert math.degrees(worst) > 170.0
    assert AT.carry_turn_bucket(95.0) == "lt100" and AT.carry_turn_bucket(120.0) == "lt140" and AT.carry_turn_bucket(150.0) == "ge140"


def test_fallback_switch(monkeypatch):
    for env in list(AT.ATTACH_ROUTE_SWITCHES.values()) + [AT.ATTACH_FALLBACK_SWITCH]:
        monkeypatch.delenv(env, raising=False)
    assert not AT.attach_fallback_on()
    monkeypatch.setenv(AT.ATTACH_FALLBACK_SWITCH, "1")
    assert AT.attach_fallback_on()
    monkeypatch.setenv("ISG_ATTACH_LYING90", "1")
    with pytest.raises(ValueError):
        AT.attach_fallback_on()


def _fb_stub(outcomes, routes=(None, "lying90", "handover")):
    """``outcomes``: route -> an exception to raise or a Draft to return from the route's draw."""
    s = _stub(None)
    s._route_states = {r: {} for r in routes}
    s.attach_route = None
    switched = []

    def switch(route, rng):
        switched.append(route)
        s.attach_route = route

    def draw(rng):
        out = outcomes[s.attach_route]
        if isinstance(out, Exception):
            s._early_rest_last = {"tilt_deg": 9.0} if isinstance(out, AT.EarlyRestUnstable) else None
            raise out
        return out

    s._switch_route = switch
    s._sample_instance_route = draw
    return s, switched


def test_fallback_after_the_current_rest_fails():
    d = Draft()
    s, switched = _fb_stub({None: AT.EarlyRestUnstable("rest_unstable", "early"), "lying90": d})
    assert s.sample_instance(None) is d and switched == ["lying90"]
    assert d.summary["attach_fallback"]["route"] == "lying90"
    assert d.summary["attach_fallback"]["tried"] == [{"route": "current", "reject": "rest_unstable", "early_rest": True, "tilt_deg": 9.0}]
    assert s.counters == {"attach_fallback.current.rest_unstable": 1, "attach_fallback.used.lying90": 1}


def test_fallback_chain_to_handover_and_exhausted():
    d = Draft()
    s, switched = _fb_stub({None: AT.EarlyRestUnstable("rest_unstable", "early"), "lying90": AT.Reject("target_placement", "carry"), "handover": d})
    assert s.sample_instance(None) is d and switched == ["lying90", "handover"]
    assert s.counters["attach_fallback.lying90.rejected.target_placement"] == 1 and s.counters["attach_fallback.used.handover"] == 1
    s, _ = _fb_stub({None: AT.EarlyRestUnstable("rest_unstable", "a"), "lying90": AT.EarlyRestUnstable("rest_unstable", "b"),
                     "handover": AT.EarlyRestUnstable("rest_unstable", "c")})
    with pytest.raises(AT.EarlyRestUnstable):
        s.sample_instance(None)
    assert s.counters["attach_fallback.exhausted"] == 1 and s.counters["attach_fallback.handover.rest_unstable"] == 1


def test_fallback_keeps_the_current_routes_other_rejections():
    s, switched = _fb_stub({None: AT.Reject("partner_placement", "x"), "lying90": Draft()})
    with pytest.raises(AT.Reject) as e:
        s.sample_instance(None)
    assert e.value.reason == "partner_placement" and switched == [] and s.counters == {}
    d = Draft()
    s, switched = _fb_stub({None: AT.EarlyRestUnstable("rest_unstable", "early"), "handover": d}, routes=(None, "handover"))
    assert s.sample_instance(None) is d and switched == ["handover"]


def test_no_fallback_states_is_the_route_draw():
    s = _stub(None)
    d = Draft()
    s._sample_instance_route = lambda rng: d
    assert s.sample_instance(None) is d and "attach_fallback" not in d.summary
