"""i7-t03 (owner request 2026-11-08 item 1, ISPATIALGYM_SHELF_STANCE_SHIFT, default off): when the shelf layout's fixed start fails the
grasp or the transit check at a manipulation candidate, the base is moved along the edge - the computed stance (the book at
SHELF_STANCE_BOOK_LATERAL_M from the robot's centre line, same stand-off / trunk) first, then the scan - with the same checks at every
stance. Simulator-free."""

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

# the touching_03 frame: the robot faces -y (yaw -pi/2), the edge tangent (+ = the robot's left) is +x
EDGE = EdgeFrame("+u", th.tensor([-3.2418, -0.115]), th.tensor([0.0, -1.0]), 1.2)
BOOK = (th.tensor([-3.5176, -0.1222, 0.9484]), th.tensor([0.5761, 0.0, -0.8174, 0.0]))


def _cand(lateral: float, stand_off: float = 0.55) -> Stance:
    xy = EDGE.base_xy(stand_off, lateral)
    st = Stance("manipulation", (float(xy[0]), float(xy[1]), EDGE.yaw), 0.25, 0.0, stand_off, EDGE, None, lateral)
    st.focus = th.tensor([-3.62, -0.155, 0.947])
    return st


def _sampler(free_floor=lambda xy: True, collides=lambda st: False, gen=None):
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.shelf_pose = BOOK
    s.gen = lambda k, d=None: (gen or {}).get(k, d)
    s.kit = SimpleNamespace(trav_free=lambda xy: bool(free_floor(xy)), stats={})
    s.ikc = SimpleNamespace(stances_collision_free=lambda sts: th.tensor([not collides(st) for st in sts]))
    s.gp = SimpleNamespace(variant="t03")
    s.travel_dropped = []
    s.drop_travel_records = lambda a, b: s.travel_dropped.append((a, b))
    s._shelf_camera_rule = lambda cand: (math.radians(40.0), lambda st, cam, focus: True)
    s.world = SimpleNamespace(camera_model=SimpleNamespace(angle_off_axis=lambda cam, focus: 0.1))
    return s


@pytest.fixture(autouse=True)
def _cams(monkeypatch):
    monkeypatch.setattr(TT, "stance_cameras", lambda s, sts: [None for _ in sts])


def test_the_measured_book_lateral():
    """The 7 written demos: the book 0.185-0.200 m left of the base (stand-off 0.55, trunk 0.25 / 0) - the fix2 success at lateral -0.465."""
    st = _cand(-0.465)
    depth, lat = st.local(BOOK[0][:2])
    assert lat == pytest.approx(0.189, abs=2e-3) and IS.SHELF_STANCE_BOOK_LATERAL_M == 0.19


@pytest.mark.parametrize("arm,want", [("left", 0.19), ("right", -0.19)])
def test_computed_stance_puts_the_book_at_the_measured_lateral(arm, want):
    s = _sampler()
    cand = _cand(-0.065)  # 0.2 m right of the successes' start (the book 0.40 m to the robot's left)
    out = s.shelf_shift_stances(cand, arm)
    label, st, shift = out[0]
    assert label == "computed" and st.stand_off == cand.stand_off and (st.translate, st.tilt) == (cand.translate, cand.tilt)
    assert st.local(BOOK[0][:2])[1] == pytest.approx(want, abs=1e-4) and st.yaw == pytest.approx(cand.yaw)
    # the scan: shift list x STAND_OFFS (the computed lateral at the other stand-offs first), the computed stance not repeated
    assert all(lab == "scan" for lab, _, _ in out[1:])
    keys = {(round(float(x.lateral), 4), x.stand_off) for _, x, _ in out}
    assert len(keys) == len(out) == len(IS.SHELF_STANCE_SCAN_SHIFTS_M) * len(TT.STAND_OFFS)
    assert out[1][2] == out[0][2] and out[1][1].stand_off != pytest.approx(0.55)


def test_a_candidate_that_passes_is_never_shifted(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    s = _sampler()
    cand = _cand(-0.465)
    calls = []
    st, g = s.shelf_fixed_start(None, cand, cand, "left", lambda m: calls.append(m) or "grasp")
    assert st is cand and g == "grasp" and calls == [cand] and s.kit.stats == {}


@pytest.mark.parametrize("switch", ["", "0"])
def test_switch_off_is_the_former_behaviour(monkeypatch, switch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, switch)
    s = _sampler()
    cand = _cand(-0.065)
    calls = []

    def fail(m):
        calls.append(m)
        raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose from manipulation")

    with pytest.raises(Reject) as e:
        s.shelf_fixed_start(None, cand, cand, "left", fail)
    assert "no library grasp" in str(e.value) and calls == [cand] and s.kit.stats == {}


def test_shifts_never_land_off_the_floor_or_in_collision(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    off_floor = lambda xy: float(xy[0]) > -3.70  # noqa: E731 - the computed stance (x -3.7076) is free, every stance further right is not
    collides = lambda st: st.stand_off is not None and st.stand_off < 0.5  # noqa: E731
    s = _sampler(free_floor=off_floor, collides=collides, gen={"shelf_stance_scan_max": 50})
    cand = _cand(-0.065)
    seen = []

    def check(m):
        seen.append(m)
        if m is cand:
            raise Reject("grasp_unreachable", "transit check: the pre-grasp witness lies 3.52 rad from the transit's start posture (> 1.60)")
        raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose")

    init = SimpleNamespace(name="init")
    with pytest.raises(Reject):
        s.shelf_fixed_start(init, cand, cand, "left", check)
    tried = seen[1:]
    assert tried and all(float(m.base_xy[0]) > -3.70 and m.stand_off >= 0.5 for m in tried)
    st = s.kit.stats
    assert st["stance_shift.need.transit"] == 1 and st["stance_shift.computed_tried"] == 1 and st.get("stance_shift.rejected.floor", 0) > 0
    assert st.get("stance_shift.rejected.collision", 0) > 0 and st["stance_shift.rejected.check"] == len(tried)
    assert ("init", "manipulation") in s.travel_dropped


def test_the_first_passing_shift_is_the_stance(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    s = _sampler()
    cand = _cand(-0.065)
    grasp = SimpleNamespace(grasp_name="grasp_0~r180", extras={"shelf_transit": {"reach_rad": 1.48}})

    def check(m):
        if m is cand:
            raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose from manipulation")
        return grasp

    st, g = s.shelf_fixed_start(SimpleNamespace(name="init"), cand, cand, "left", check)
    assert g is grasp and st is not cand and st.local(BOOK[0][:2])[1] == pytest.approx(0.19, abs=1e-4)
    rec = st.extras["stance_shift"]
    assert rec["label"] == "computed" and rec["need"] == "grasp" and rec["branch_after"] == "near" and st.arms == ["left"]
    assert s.kit.stats == {"stance_shift.need.grasp": 1, "stance_shift.computed_tried": 1, "stance_shift.computed_passed": 1,
                           f"stance_shift.offset.{rec['shift_m']:+.2f}": 1}


def test_failures_the_base_cannot_change_are_not_shifted(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    assert IS.shelf_need_of(Reject("no_transition", "init -> manipulation: no s_T")) is None
    assert IS.shelf_need_of(Reject("no_transition", "shelf back-off: no s_T >= 0.75")) == "transit"
    assert IS.reach_branch(3.52) == "flipped" and IS.reach_branch(1.48) == "near" and IS.reach_branch(None) == "-"
