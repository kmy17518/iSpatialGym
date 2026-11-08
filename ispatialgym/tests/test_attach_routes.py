"""i7-cork (owner decision 2026-11-08 4.2): the attachment routes behind their default-off switches - the switch reading, the route keys
and the oracle program's dispatch on the recorded ``grasps[0].attach_route``."""

import copy
import json
import os

import pytest

from ispatialgym.oracle.programs import attach_route_of, expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError

DATA = os.path.join(os.path.dirname(__file__), "data", "sidecars")
ST = "configuration_matching-attachment"


def _raw():
    with open(os.path.join(DATA, f"{ST}.json")) as f:
        return json.load(f)


def test_switches_default_off(monkeypatch):
    pytest.importorskip("omnigibson")
    from ispatialgym.sampling.v2.tasks import attachment as AT

    for env in AT.ATTACH_ROUTE_SWITCHES.values():
        monkeypatch.delenv(env, raising=False)
    assert AT.attach_route_requested() is None
    monkeypatch.setenv("ISG_ATTACH_LYING90", "1")
    assert AT.attach_route_requested() == "lying90"
    monkeypatch.setenv("ISG_ATTACH_HANDOVER", "1")
    with pytest.raises(ValueError):
        AT.attach_route_requested()
    monkeypatch.setenv("ISG_ATTACH_LYING90", "0")
    assert AT.attach_route_requested() == "handover"
    assert AT.attach_route_keys("cork__ncxgpe", "lying90") == ("cork__ncxgpe__lying90",)
    assert AT.attach_route_keys("cork__ncxgpe", "handover") == ("cork__ncxgpe__handover_pick", "cork__ncxgpe__handover_hold")


def test_program_reads_the_recorded_route():
    base = Sidecar.from_dict(_raw())
    assert attach_route_of(base) is None
    d = copy.deepcopy(_raw())
    d["grasps"][0]["attach_route"] = "lying90"
    sc = Sidecar.from_dict(d)
    assert attach_route_of(sc) == "lying90"
    # lying90 = the attachment program as it is (the recorded SLERP carry turns the male 90 deg in hand)
    assert [a.name for a in expand(sc.variant, sc)] == [a.name for a in expand(base.variant, base)]
    d["grasps"][0]["attach_route"] = "sideways"
    with pytest.raises(SidecarError):
        attach_route_of(Sidecar.from_dict(d))
