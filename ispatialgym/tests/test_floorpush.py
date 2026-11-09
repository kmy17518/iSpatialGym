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
    monkeypatch.setenv(IL.FLOORPUSH_STROKE_ENV, "off")  # the variants one at a time (the stroke's default is tested on its own)


def _s(**gen):
    s = PP._sampler(push_under_hand="pitched_far", **gen)
    s.kit = SimpleNamespace(floor_z=0.0, stats={})
    return s


def _cylinder(r: float = 0.112, h: float = 0.27, n: int = 48) -> th.Tensor:
    return th.tensor([[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z] for k in range(n) for z in (0.0, h)], dtype=th.float32)


def test_switches_parse():
    off = {IL.FLOORPUSH_STROKE_ENV: "off"}
    assert IL.floorpush_switches(off) == {} and IL.floorpush_switches({IL.FLOORPUSH_STROKE_ENV: "0"}) == {}
    # the stroke's default (2026-11-08): 0.10 m strokes with no switch set
    assert IL.floorpush_switches({}) == {"stroke_m": IL.FLOORPUSH_STROKE_DEFAULT_M, "yaw_gain": IL.FLOORPUSH_YAW_GAIN_M_PER_RAD,
                                         "strokes_max": IL.FLOORPUSH_STROKES_MAX} and IL.FLOORPUSH_STROKE_DEFAULT_M == 0.10
    fp = IL.floorpush_switches({IL.FLOORPUSH_OPEN_ENV: "1", IL.FLOORPUSH_BOTH_ENV: "0", IL.FLOORPUSH_STROKE_ENV: "0.05", IL.FLOORPUSH_TIP_ENV: "0.05"})
    assert fp == {"open": True, "stroke_m": 0.05, "tip_m": 0.05, "yaw_gain": IL.FLOORPUSH_YAW_GAIN_M_PER_RAD, "strokes_max": IL.FLOORPUSH_STROKES_MAX}
    assert IL.floorpush_switches({IL.FLOORPUSH_STROKE_ENV: "0.1", "ISPATIALGYM_FLOORPUSH_YAW_GAIN": "0", "ISPATIALGYM_FLOORPUSH_STROKES_MAX": "3"}) == {
        "stroke_m": 0.1, "yaw_gain": 0.0, "strokes_max": 3}
    for bad in ("0", "-0.05", "nan"):
        with pytest.raises(ValueError):
            IL.floorpush_switches({**off, IL.FLOORPUSH_TIP_ENV: bad})


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



def _turned_box(yaw_deg: float = 20.0) -> th.Tensor:
    """The briefcase-like box (0.33 x 0.12 x 0.27) turned ``yaw_deg`` about its root: pushed along +x, its back is a corner and a slanted face."""
    b = PP._box(0.33, 0.12, 0.27)
    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return th.stack([c * b[:, 0] - s_ * b[:, 1], s_ * b[:, 0] + c * b[:, 1], b[:, 2]], dim=1)


def test_open_palm_test_sees_the_spread_fingers_and_the_empty_middle(monkeypatch):
    hull = _turned_box(35.0)
    rear = float(-(hull[:, 0]).min())
    fc = IL.pitched_open_contact(hull, (1.0, 0.0), 0.074, 0.086)
    assert fc["off_m"] < rear - 0.005  # the first fingertip meets the slanted face ahead of the rearmost corner
    q = dict(TT.pitched_hand_quats(60.0, (1.0, 0.0)))["lateral"]
    so, rise, _ = TT.pitched_tip_offsets(60.0, q, IL.PUSH_UNDER_CLEARANCE_M)
    eef = th.tensor([-(fc["off_m"] + so), 0.0, 0.08 + rise])
    closed_gap, _ = IL.pitched_palm_gap(hull, (1.0, 0.0), eef, q)
    open_gap, part = IL.pitched_palm_gap(hull, (1.0, 0.0), eef, q, open_finger_m=fc["fingers_m"])
    # the closed-sphere test reads the corner beside the hand (before the fix every open turn read 'the face reaches the finger', fp_open
    # under_02 2026-11-08); the open hand's spheres meet only the back within their own lateral extent
    assert closed_gap < 0.0 <= open_gap
    monkeypatch.setenv(IL.FLOORPUSH_OPEN_ENV, "1")
    s = _s()
    s.hull = hull
    ch = PP._push(s)
    assert ch["record"]["floorpush"]["variant"] == "open" and ch["half_extent_m"] == pytest.approx(fc["off_m"], abs=1e-6)


def test_strokes_are_the_default_for_the_pitched_push_only(monkeypatch):
    monkeypatch.delenv(IL.FLOORPUSH_STROKE_ENV)
    plan = _s().push_strokes_record(0.311)["strokes"]
    assert plan["n"] == 4 and plan["yaw_gain_m_per_rad"] == pytest.approx(IL.FLOORPUSH_YAW_GAIN_M_PER_RAD)
    ch = PP._push(_s())
    assert ch["record"]["floorpush"]["stroke_m"] == 0.10 and "face_half_m" in ch["record"]
    assert PP._sampler().push_strokes_record(0.311) == {}  # the blade (under_large_01) keeps its generation key


# ---------------------------------------------------------------------------------------------------------------- r2 (lead #60 / #61)


def test_broadside_lock_keeps_only_the_long_face_yaws(monkeypatch):
    monkeypatch.setenv(IL.FLOORPUSH_BROADSIDE_ENV, "1")
    s = _s(hull=PP._box(0.33, 0.12, 0.27))  # the briefcase: long axis along x at the rest
    assert s.push_yaw_lock() == "broadside" and "broadside" in IL.PUSH_UNDER_YAW_LOCKS
    q = lambda deg: G.qmul(G.quat_from_yaw(math.radians(deg)), s.rest)  # noqa: E731
    assert s.push_yaw_kept(q(90.0), (1.0, 0.0)) and s.push_yaw_kept(q(-90.0), (1.0, 0.0)) and s.push_yaw_kept(q(100.0), (1.0, 0.0))
    assert not s.push_yaw_kept(q(0.0), (1.0, 0.0)) and not s.push_yaw_kept(q(22.5), (1.0, 0.0)) and not s.push_yaw_kept(q(67.5), (1.0, 0.0))
    # broadside the contact offset is the half depth (the thin side), on the end it is the half length
    assert s.push_contact_offset(q(90.0), (1.0, 0.0)) == pytest.approx(0.06, abs=1e-5) and s.push_contact_offset(q(0.0), (1.0, 0.0)) == pytest.approx(0.165, abs=1e-5)
    # a round footprint keeps every yaw; the generation key wins over the switch; off without the switch
    s2 = _s(hull=_cylinder())
    assert s2.push_yaw_kept(q(13.0), (1.0, 0.0))
    monkeypatch.delenv(IL.FLOORPUSH_BROADSIDE_ENV)
    assert _s().push_yaw_lock() is None


def _transit_world(monkeypatch, standoff_ok=True):
    calls = []

    def reach(motions, state, arm, pre, supports, through, q_hint, ctx, cfg, **kw):
        calls.append(("plan", [round(float(v), 3) for v in EX.as_world_pose(pre)[0].tolist()], [str(o) for o in through]))
        if not standoff_ok and not through:
            raise EX.SkillError("plan_failed", "stub: no plan with the can present")
        yield th.zeros(1)

    monkeypatch.setattr(EX, "reach_pre_pose", reach)
    monkeypatch.setattr(EX, "push_transit_witness_clear", lambda *a, **k: (False, "witness within 10 mm of the can's box"))
    motions = SimpleNamespace(cfg=SimpleNamespace(diff_ik_soft_pos_tol_free=0.01))

    def along(arm, wps, **kw):
        calls.append(("along", [round(float(v), 3) for v in EX.as_world_pose(wps[0])[0].tolist()]))
        yield th.zeros(1)

    motions.arm_along = along
    return calls, motions


@pytest.mark.parametrize("standoff_ok", [True, False])
def test_push_transit_via_a_standoff_with_the_target_present(monkeypatch, standoff_ok):
    calls, motions = _transit_world(monkeypatch, standoff_ok)
    pre = SC.Pose(position=[1.0, 0.0, 0.15], orientation=[0.0, 0.0, 0.0, 1.0])
    cfg = EX.ExecuteConfig()
    assert cfg.push_transit_standoff_m == 0.0  # default off
    cfg = EX.replace(cfg, push_transit_standoff_m=0.10)
    ctx = {"push_direction": [1.0, 0.0, 0.0], "plan_ignored": ["can"]}
    list(EX.reach_push_pre(motions, None, "right", pre, [], ["can"], None, ctx, cfg))
    if standoff_ok:
        assert calls == [("plan", [0.95, 0.0, 0.25], []), ("along", [1.0, 0.0, 0.15])]  # 10 cm up, 5 cm back, the can present; then down
        assert ctx["push_transit"]["target"] == "present to the standoff" and ctx["plan_ignored"] == []
    else:
        assert calls == [("plan", [0.95, 0.0, 0.25], []), ("plan", [1.0, 0.0, 0.15], ["can"])]  # the old left-out plan after it
        assert ctx["push_transit"]["target"] == "left out" and "push_transit_standoff_failed" in ctx


def test_transit_standoff_env(monkeypatch):
    for v, want in (("", 0.0), ("0", 0.0), ("off", 0.0), ("1", 0.10), ("0.12", 0.12), ("x", 0.0)):
        monkeypatch.setenv("ISPATIALGYM_PUSH_TRANSIT_STANDOFF", v)
        assert EX._transit_standoff_env() == pytest.approx(want)
