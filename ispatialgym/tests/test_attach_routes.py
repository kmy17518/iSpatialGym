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


def _handover_raw():
    """The fixture made a hand-over: grasps[0] the right hand's pick (its waypoints end at the hand-over), grasps[1] the left hand's hold
    (grasp at the hand-over, eef_in_object, waypoints / pre_place / place = the attach), no brace."""
    d = copy.deepcopy(_raw())
    d.pop("contacts", None)
    d["summary"] = {"brace": "none"}
    g0 = d["grasps"][0]
    g0["attach_route"] = "handover"
    g0["handover"] = {"role": "pick", "object_pose": [[7.0, 3.1, 1.05], [0.0, 0.0, 0.0, 1.0]]}
    hold = copy.deepcopy(g0)
    hold.update({"arm": "left", "grasp_name": "top_3", "grasp": {"position": [7.0, 3.13, 1.05], "orientation": [0.7071068, 0.0, 0.0, 0.7071068]},
                 "pre_grasp": {"position": [7.0, 3.23, 1.05], "orientation": [0.7071068, 0.0, 0.0, 0.7071068]},
                 "eef_in_object": {"position": [0.0, 0.0, 0.03], "orientation": [1.0, 0.0, 0.0, 0.0]}, "carry_height": 0.0,
                 "handover": {"role": "hold", "object_pose": g0["handover"]["object_pose"]}})
    for k in ("pre_place", "place"):
        g0.pop(k, None)
    d["grasps"] = [g0, hold]
    return d


def test_handover_program_sequence():
    sc = Sidecar.from_dict(_handover_raw())
    assert attach_route_of(sc) == "handover"
    actions = expand(sc.variant, sc)
    names = [a.name for a in actions]
    assert names[-8:] == ["grasp", "lift", "reorient", "grasp", "retreat", "reorient", "place", "retreat"]
    pick, lift, to_h, take, release, carry, place, back = actions[-8:]
    assert pick.arm == lift.arm == to_h.arm == release.arm == "right"
    assert take.arm == carry.arm == place.arm == back.arm == "left"
    assert release.release and release.after is to_h and carry.after is take
    assert place.kind == "attach" and place.on == "lemonade_bottle.n.01_1"


def test_handover_program_rejects_a_brace_or_one_hand():
    d = _handover_raw()
    d["grasps"][1]["arm"] = "right"
    with pytest.raises(SidecarError):
        expand(Sidecar.from_dict(d).variant, Sidecar.from_dict(d))
    d = _handover_raw()
    d["contacts"] = copy.deepcopy(_raw()["contacts"])
    with pytest.raises(SidecarError):
        expand(Sidecar.from_dict(d).variant, Sidecar.from_dict(d))


def test_grasp_of_prefers_the_arm():
    from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft

    sd = SidecarDraft("configuration_matching-attachment-x", "configuration_matching")
    a = sd.add_grasp(GraspRecord("cork_1", "right", "k", "g", 0.0))
    b = sd.add_grasp(GraspRecord("cork_1", "left", "k2", "h", 0.0))
    assert sd.grasp_of("cork_1") is a and sd.grasp_of("cork_1", arm="left") is b and sd.grasp_of("cork_1", arm="right") is a
    assert sd.grasp_of("cork_1", arm="nope") is a and sd.grasp_of("other") is None
