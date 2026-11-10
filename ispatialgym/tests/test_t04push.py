"""i7-t04push (owner A7, 2026-11-09): the push tip screen at the draw (ISPATIALGYM_PUSH_TIP_SCREEN) and the short strokes of the tabletop
touching push (ISPATIALGYM_TOUCH_PUSH_STROKE_M) - switches default off, the stroke record the oracle's stroke executor reads, the screen's
verdict from the hand-overlap test of the goal check's candidate stream."""
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.tasks import iors_small as IS


def test_switches_default_off(monkeypatch):
    for k in (IS.PUSH_TIP_SCREEN_ENV, IS.TOUCH_PUSH_STROKE_ENV):
        monkeypatch.delenv(k, raising=False)
    assert IS.PUSH_TIP_SCREEN_DEFAULT == "0" and IS.push_tip_screen_mode() == "off" and IS.touch_push_stroke_m() is None
    for v, mode in (("1", "on"), ("shadow", "shadow"), ("0", "off")):
        monkeypatch.setenv(IS.PUSH_TIP_SCREEN_ENV, v)
        assert IS.push_tip_screen_mode() == mode
    for v, m in (("off", None), ("0", None), ("0.03", 0.03)):
        monkeypatch.setenv(IS.TOUCH_PUSH_STROKE_ENV, v)
        assert IS.touch_push_stroke_m() == m
    monkeypatch.setenv(IS.TOUCH_PUSH_STROKE_ENV, "-0.02")
    with pytest.raises(ValueError):
        IS.touch_push_stroke_m()


def _me(stats):
    return SimpleNamespace(kit=SimpleNamespace(stats=stats), shelf_start=None, gp=SimpleNamespace(variant="t04"))


def test_the_stroke_record_is_a_plan_the_oracle_reads_with_the_fingertips_geometry(monkeypatch):
    stats = {}
    me = _me(stats)
    monkeypatch.delenv(IS.TOUCH_PUSH_STROKE_ENV, raising=False)
    assert IS.IORSSmall.touch_push_strokes(me, "push_tip_h-", {}, 0.081) == {}
    monkeypatch.setenv(IS.TOUCH_PUSH_STROKE_ENV, "0.03")
    rec = IS.IORSSmall.touch_push_strokes(me, "push_tip_h-", {"stop_clearance_m": 0.005}, 0.081)
    plan = EX.push_stroke_plan(rec)
    assert plan is not None and plan["n"] == 3 and plan["ends_m"][-1] == pytest.approx(0.081)
    assert plan["backoff_m"] == IS.TOUCH_PUSH_STROKE_BACKOFF_M and plan["reaim_max_m"] == IS.TOUCH_PUSH_STROKE_REAIM_MAX_M
    assert rec["face_half_m"] == pytest.approx(C.FINGERTIP_LENGTH) and rec["blade_standoff_m"] == pytest.approx(C.FINGERTIP_LENGTH + 0.005)
    assert rec["contact_strips_m"] == [[0.0, pytest.approx(0.0084, abs=1e-4)]] and rec["contact_band_m"] == [-0.01, 0.01]
    v = IS.IORSSmall.touch_push_strokes(me, "push_tip_v-", {}, 0.05)
    assert v["contact_strips_m"][0][1] < rec["contact_strips_m"][0][1] and EX.push_stroke_plan(v)["n"] == 2
    # blades / the open hand / the shelf layout keep one stroke
    assert IS.IORSSmall.touch_push_strokes(me, "push_blade_lean20", {}, 0.08) == {}
    assert IS.IORSSmall.touch_push_strokes(me, "push_tip_h-", {"fingers_opening_m": 0.03, "push_fingers": 0.02}, 0.08) in ({}, IS.IORSSmall.touch_push_strokes(me, "push_tip_h-", {}, 0.08))
    me.shelf_start = object()
    assert IS.IORSSmall.touch_push_strokes(me, "push_tip_h-", {}, 0.08) == {}
    assert stats["touch_push_strokes.recorded"] >= 2 and stats["touch_push_strokes.skipped_class"] >= 1


def test_the_tip_screen_passes_on_one_clean_class_and_names_the_overlaps(monkeypatch):
    """push_overlap_screen_why over a stub stream: None once a class clears the hull; else the count of each overlapping part; the tip cap
    (when on) and the pair's height limit skip classes before the test."""
    stats = {}
    me = _me(stats)
    direction = th.tensor([0.0, 1.0, 0.0])
    me.touching_points = lambda t, p, st: (th.zeros(3), th.zeros(3))
    me.world = SimpleNamespace(hull_at=lambda h, pose: th.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.08, 0.0], [0.08, 0.08, 0.27]]))
    me.hull = {"target": None, "partner": None}
    me.push_distance = lambda gap: 0.08
    me.push_line_xy = lambda pose: None
    me.push_gripper_open_on = lambda: False
    me.pair_goal_align = lambda: None
    me.gen = lambda k, d=None: d
    st = SimpleNamespace(tangent=direction[:2])
    pose = (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))
    calls = []

    def stream(h_ok):
        return [("push_tip_h-", (th.tensor([0.0, 0.0, h]), pose[1]), {"contact_height_m": h}) for h in (0.05, 0.08)] if h_ok else []

    me.touching_push_candidates = lambda st_, p, z, d, t_hull_w=None, line_xy=None: iter(stream(True))
    me.push_hand_overlap = lambda init, goal, t, d, dist, against="box": calls.append(round(float(init[0][2]), 4)) or (None if float(init[0][2]) > 0.07 else "knuckle -15 mm from the hull (margin 5 mm)")
    monkeypatch.setenv(IS.PUSH_TIP_CAP_ENV, "0")
    assert IS.IORSSmall.push_overlap_screen_why(me, st, pose, pose) is None and calls == [0.05, 0.08]
    monkeypatch.setenv(IS.PUSH_TIP_CAP_ENV, "1")  # the 8 cm class is above the cap (1.5 x 4 cm = 6 cm): only the 5 cm one is tested
    calls.clear()
    why = IS.IORSSmall.push_overlap_screen_why(me, st, pose, pose)
    assert calls == [0.05] and "1x knuckle" in why and "tip cap" in why
