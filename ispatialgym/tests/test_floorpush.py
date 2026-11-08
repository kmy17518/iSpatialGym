"""i7-floorpush (owner decision 2026-11-08 item 5, under_large_02 / 03): the pitched floor push's variants behind their switches - (a) the open
gripper, (b) the both-hands probe, (c) the strokes, (d) the lower fingertips - on the pitched-chain fixtures of
``test_iors_large_push_pitched`` (no simulator): every switch off leaves today's chain and record, each acts only on the pitched hand of the
under relation, falls back to today's chain, and counts where it acted."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.actions import Push
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import sidecar as SC
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.tests import test_iors_large_push_pitched as PP

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

ENVS = (IL.FLOORPUSH_OPEN_ENV, IL.FLOORPUSH_BOTH_ENV, IL.FLOORPUSH_STROKE_ENV, IL.FLOORPUSH_TIP_ENV, "ISPATIALGYM_FLOORPUSH_YAW_GAIN",
        "ISPATIALGYM_FLOORPUSH_STROKES_MAX")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENVS:
        monkeypatch.delenv(k, raising=False)


def _s(**gen):
    s = PP._sampler(push_under_hand="pitched_far", **gen)
    s.kit = SimpleNamespace(floor_z=0.0, stats={})
    return s


def _cylinder(r: float = 0.112, h: float = 0.27, n: int = 48) -> th.Tensor:
    return th.tensor([[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z] for k in range(n) for z in (0.0, h)], dtype=th.float32)


def test_switches_parse():
    assert IL.floorpush_switches({}) == {}
    fp = IL.floorpush_switches({IL.FLOORPUSH_OPEN_ENV: "1", IL.FLOORPUSH_BOTH_ENV: "0", IL.FLOORPUSH_STROKE_ENV: "0.05", IL.FLOORPUSH_TIP_ENV: "0.05"})
    assert fp == {"open": True, "stroke_m": 0.05, "tip_m": 0.05, "yaw_gain": IL.FLOORPUSH_YAW_GAIN_M_PER_RAD, "strokes_max": IL.FLOORPUSH_STROKES_MAX}
    assert IL.floorpush_switches({IL.FLOORPUSH_STROKE_ENV: "0.1", "ISPATIALGYM_FLOORPUSH_YAW_GAIN": "0", "ISPATIALGYM_FLOORPUSH_STROKES_MAX": "3"}) == {
        "stroke_m": 0.1, "yaw_gain": 0.0, "strokes_max": 3}
    for bad in ("0", "-0.05", "nan"):
        with pytest.raises(ValueError):
            IL.floorpush_switches({IL.FLOORPUSH_TIP_ENV: bad})


def test_switches_act_only_on_the_pitched_under_push(monkeypatch):
    monkeypatch.setenv(IL.FLOORPUSH_OPEN_ENV, "1")
    assert _s().floorpush() == {"open": True}
    blade = PP._sampler()
    assert blade.floorpush() == {}  # the blade (under_large_01's sandal) and every other hand: untouched
    s = _s()
    s.relation = "ontop"
    assert s.floorpush() == {}
    monkeypatch.setenv(IL.FLOORPUSH_TIP_ENV, "-1")
    with pytest.raises(IL.Reject) as e:
        _s().floorpush()
    assert e.value.reason == "spec"


def test_every_switch_off_leaves_todays_chain():
    s = _s()
    ch = PP._push(s)
    assert "floorpush" not in ch["record"] and "face_half_m" not in ch["record"] and "push_contact_form" not in ch["record"]
    assert s.kit.stats == {} and all(c["fingers"] == C.FINGER_CLOSED for c in s.ikc.calls) and len(s.ikc.calls) == 4
    assert s.push_strokes_record(0.31) == {}


def test_open_contact_geometry():
    # a 0.27 m wide flat back: both fingertips on it, the gap 60 % of the width capped at 8 cm, met at the face itself
    fc = IL.pitched_open_contact(PP._box(0.33, 0.27, 0.27), (1.0, 0.0), 0.074, 0.086)
    assert fc["opening_m"] == pytest.approx(0.08) and fc["off_m"] == pytest.approx(0.165, abs=1e-6)
    assert fc["fingers_m"] == pytest.approx(IL.finger_joint_for_opening(0.08)) and fc["strips_m"] == [[round(c, 5), round(hw, 5)] for c, hw in IL.push_contact_strips("two_point", 0.08)[0]]
    # a 0.12 m briefcase end: 7.2 cm apart
    assert IL.pitched_open_contact(PP._box(0.33, 0.12, 0.27), (1.0, 0.0), 0.074, 0.086)["opening_m"] == pytest.approx(0.072, abs=1e-5)
    # a round can: the fingertips meet the back off its rearmost point, nearer the root than the closed hand's centred contact
    fc = IL.pitched_open_contact(_cylinder(), (1.0, 0.0), 0.074, 0.086)
    c = 0.5 * fc["opening_m"] + 0.5 * float(TT.FINGER_PROFILE[0][1])
    assert fc["off_m"] < 0.112 and fc["off_m"] == pytest.approx(math.sqrt(0.112 ** 2 - (c - 0.5 * float(TT.FINGER_PROFILE[0][1])) ** 2), abs=3e-3)
    assert IL.pitched_open_contact(PP._box(0.3, 0.3, 0.05), (1.0, 0.0), 0.074, 0.086) is None  # nothing in the band


def test_open_variant_chain_record_and_fall_back(monkeypatch):
    monkeypatch.setenv(IL.FLOORPUSH_OPEN_ENV, "1")
    s = _s()
    ch = PP._push(s)
    rec = ch["record"]
    fc = IL.pitched_open_contact(s.hull, (1.0, 0.0), *rec["fingertip_band_m"])
    assert rec["floorpush"] == {"variant": "open", "open": True} and ch["wrist_sign"] == "lateral"
    assert rec["push_contact_form"] == "two_point" and rec["fingers_opening_m"] == pytest.approx(fc["fingers_m"], abs=1e-5)
    assert rec["contact_width_m"] == pytest.approx(0.08) and rec["face_half_m"] == pytest.approx(ch["blade_standoff_m"] - IL.PUSH_UNDER_CLEARANCE_M, abs=1e-5)
    # only the lateral turns, every IK call with the opening (the closing axis across the push)
    assert len(s.ikc.calls) == 2 and all(c["fingers"] == pytest.approx(fc["fingers_m"], abs=1e-6) for c in s.ikc.calls)
    assert abs(float(C.closing_axis(ch["start"][1])[1])) == pytest.approx(1.0, abs=1e-6)
    assert s.kit.stats == {"floorpush.open.chain": 1}
    # the oracle and the goal check read the opening through the sidecar
    pr = IL.PushUnderRecord("right", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], "briefcase.n.01_1", stance="push",
                            pitch_deg=ch["pitch_deg"], blade_standoff_m=ch["blade_standoff_m"], half_extent_m=ch["half_extent_m"], **rec)
    pc = SC.PushContact.from_dict(pr.to_sidecar(), "contacts.push")
    assert IL.push_record_fingers(pc.extra) == pytest.approx(fc["fingers_m"], abs=1e-5) and pc.extra["floorpush"]["variant"] == "open"
    # the open hand pushes in one stroke even with (c) on
    monkeypatch.setenv(IL.FLOORPUSH_STROKE_ENV, "0.05")
    assert _s().push_strokes_record(0.31, "two_point") == {}
    # open IK fails: today's closed chain, counted
    monkeypatch.delenv(IL.FLOORPUSH_STROKE_ENV)
    s = _s()
    s.ikc = PP._IKC(ok=lambda chain: True)
    s.ikc.exact = (lambda real: (lambda poses, stance, arm="right", fingers=None, **kw: (
        SimpleNamespace(ok=False, q=None, reason="stub: open fails") if fingers != C.FINGER_CLOSED else real(poses, stance, arm, fingers=fingers, **kw))))(s.ikc.exact)
    ch = PP._push(s)
    assert ch["record"]["floorpush"]["variant"] == "today" and "push_contact_form" not in ch["record"]
    assert s.kit.stats == {"floorpush.open.unsolved": 1, "floorpush.today.chain": 1}


def test_low_tip_variant_and_fall_back(monkeypatch):
    monkeypatch.setenv(IL.FLOORPUSH_TIP_ENV, "0.05")
    s = _s()
    ch = PP._push(s)
    p = math.radians(60.0)
    h = TT.flat_tip_half_height(dict(TT.pitched_hand_quats(60.0, (1.0, 0.0)))["lateral"])
    assert ch["fingertip_z_m"] == pytest.approx(0.05) and ch["record"]["contact_height_m"] == pytest.approx(0.05 + h * math.sin(p))
    assert ch["record"]["floorpush"]["variant"] == "low_tip" and s.kit.stats == {"floorpush.low_tip.chain": 1}
    # the low hand unreachable: today's 8 cm
    s = _s()
    s.ikc = PP._IKC(ok=lambda chain: float(chain[1][0][2]) > 0.07)
    ch = PP._push(s)
    assert ch["fingertip_z_m"] == pytest.approx(IL.PUSH_FAR_TIP_M) and ch["record"]["floorpush"]["variant"] == "today"
    assert s.kit.stats == {"floorpush.low_tip.unsolved": 1, "floorpush.today.chain": 1}


@pytest.mark.parametrize("stroke, n", [(0.05, 7), (0.10, 4)])
def test_stroke_variant_record(monkeypatch, stroke, n):
    monkeypatch.setenv(IL.FLOORPUSH_STROKE_ENV, str(stroke))
    s = _s()
    plan = s.push_strokes_record(0.311)["strokes"]
    assert plan["n"] == n and plan["ends_m"][-1] == pytest.approx(0.311) and plan["yaw_gain_m_per_rad"] == pytest.approx(IL.FLOORPUSH_YAW_GAIN_M_PER_RAD)
    assert plan["backoff_m"] == pytest.approx(0.03) and plan["reaim_max_m"] == pytest.approx(0.03)
    assert s.kit.stats == {"floorpush.strokes.recorded": 1}
    ch = PP._push(s)
    rec = ch["record"]
    assert rec["floorpush"]["variant"] == "today" and rec["floorpush"]["stroke_m"] == stroke
    assert rec["face_half_m"] == pytest.approx(ch["blade_standoff_m"] - IL.PUSH_UNDER_CLEARANCE_M, abs=1e-5)
    assert rec["contact_strips_m"] == [[0.0, round(float(TT.FINGER_PROFILE[0][1]), 5)]]  # the lateral turn: two fingers side by side
    # the stroke plan reaches the oracle with the knobs it reads
    pr = IL.PushUnderRecord("right", ch["pre"], ch["start"], ch["mid"], ch["goal"], ch["direction"], ch["distance_m"], "briefcase.n.01_1",
                            blade_standoff_m=ch["blade_standoff_m"], **rec, **s.push_strokes_record(ch["distance_m"]))
    extra = SC.PushContact.from_dict(pr.to_sidecar(), "contacts.push").extra
    assert EX.push_stroke_plan(extra)["n"] == math.ceil(ch["distance_m"] / stroke - 1e-9) and extra["face_half_m"] == pytest.approx(rec["face_half_m"])
    # another hand ignores the switch (the blade keeps its generation key)
    assert PP._sampler().push_strokes_record(0.311) == {}


def test_both_probe_counts_and_falls_back(monkeypatch):
    monkeypatch.setenv(IL.FLOORPUSH_BOTH_ENV, "1")
    s = _s()
    arms = []
    real = s.ikc.exact

    def exact(poses, stance, arm="right", **kw):
        arms.append(arm)
        return real(poses, stance, arm, **kw) if arm == "right" else SimpleNamespace(ok=False, q=None, reason="stub: out of reach")

    s.ikc.exact = exact
    ch = PP._push(s)
    both = ch["record"]["floorpush"]["both"]
    assert both["feasible"] is False and both["arm"] == "left" and arms.count("left") == len(IL.FLOORPUSH_BOTH_OFFSETS_M) * 2
    assert s.kit.stats == {"floorpush.today.chain": 1, "floorpush.both.probe": 1, "floorpush.both.infeasible": 1}
    # the left hand's poses: the push chain moved across the push toward the base's left (the other arm), same height
    s = _s()
    seen = []
    real = s.ikc.exact
    s.ikc.exact = lambda poses, stance, arm="right", **kw: (seen.append((arm, [G.t(p[0]).clone() for p in poses])), real(poses, stance, arm, **kw))[1]
    ch = PP._push(s)
    assert ch["record"]["floorpush"]["both"] == {"arm": "left", "feasible": True, "offset_m": IL.FLOORPUSH_BOTH_OFFSETS_M[0], "why": ""}
    left = [p for a, p in seen if a == "left"][0]
    assert (left[1] - G.t(ch["start"][0])).tolist() == pytest.approx([0.0, IL.FLOORPUSH_BOTH_OFFSETS_M[0], 0.0], abs=1e-6)
    assert s.kit.stats["floorpush.both.feasible"] == 1


def test_push_under_annotation_names_the_open_hand():
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    pose = SC.Pose(position=[0.5, 0.0, 0.09], orientation=[float(v) for v in q.tolist()])
    lang = SimpleNamespace(phrase=lambda name: name.split(".")[0], objects={})
    for extra, word in (({}, "closed"), ({"fingers_opening_m": 0.0215, "push_contact_form": "two_point"}, "open")):
        c = SC.PushContact(arm="right", pre=pose, init=pose, mid=pose, goal=pose, direction=[1.0, 0.0, 0.0], distance_m=0.3, object="briefcase.n.01_1",
                           extra={"kind": "push_under", "partner": "desk.n.01_1", **extra})
        fine, numbers = Push(contact=c, base_yaw=0.0).annotate(SimpleNamespace(language=lang))
        assert f"with the {word} right hand" in fine and f"with the {word} right hand" in numbers
