"""i7-contact3 (owner 3.6, 2026-11-08): simulator-free tests of the touching place depth cap by the touch_push map's reach
(``iors_small.IORSSmall.place_depth_caps`` / ``place_depth_cap_filter``, switch ``ISPATIALGYM_PLACE_DEPTH_CAP``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

F = CMAP.STATUS["feasible"]
OFF = CMAP.STATUS["reach_bound"]


class _Map:
    """A touch_push map stand-in: cells on a 0.08 m lattice x 0.35..1.15, one trunk (0.25, -0.3), 2 yaw slots x 2 classes; a cell is
    feasible (one class, one yaw) up to ``x_max`` ahead of the base at every edge distance."""

    def __init__(self, x_max=0.75):
        xs = np.round(np.arange(0.35, 1.151, 0.08), 3)
        self.cells = np.array([[x, y] for x in xs for y in (-0.08, 0.0, 0.08)], dtype=float)
        self.yaws = np.array([0.0, np.pi / 2])
        self.classes = ["push_tip_h", "push_v"]
        self.path = f"fake_{x_max}.json"
        self.meta = {"edge_distances": [0.35, 0.45, 0.55], "config": {"lattice_m": 0.08},
                     "classes": [{"class": "push_tip_h"}, None, {"class": "push_tip_h"}, {"class": "push_v"}]}
        st = np.full((len(self.cells), 2, 2), OFF, dtype=np.uint8)
        st[self.cells[:, 0] <= x_max + 1e-9, 0, 0] = F
        st[self.cells[:, 0] <= x_max + 1e-9, 1, 1] = F
        st[np.isclose(self.cells[:, 0], x_max + 0.08), 0, 1] = F  # a feasible status on a class the yaw slot lacks: never counts
        self.st = st

    def nearest_trunk(self, translate, tilt):
        return (0.25, -0.3) if abs(translate - 0.25) <= 0.02 else None

    def status(self, d_e, translate, tilt):
        return self.st

    def class_rows(self, y):
        return self.meta["classes"][2 * y:2 * y + 2]


def _cand(depth, stand_off=0.45, lateral=0.0, translate=0.25):
    # the edge at x = 1.0, inward +x; the focus `depth` behind it
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 2.0)
    st = Stance("manipulation", (0.0, 0.0, 0.0), translate, -0.3, None, edge).shifted(stand_off=stand_off, lateral=lateral, name="manipulation")
    st.focus = th.tensor([1.0 + depth, float(lateral), 1.0])
    return st


def _me(monkeypatch, pm, free=lambda st: True, mode="1"):
    monkeypatch.setenv(IS.PLACE_DEPTH_CAP_ENV, mode)
    monkeypatch.setattr(TT, "contact_map", lambda s, kind, role, top, **kw: pm)
    monkeypatch.setattr(TT, "contact_maps_on", lambda: True)
    stats = {}
    me = SimpleNamespace(relation="touching", shelf_start=None, shelf_pose=None, top=None, gen=lambda k, d=None: d, gp=SimpleNamespace(variant="t"),
                         kit=SimpleNamespace(stats=stats), ikc=SimpleNamespace(stances_collision_free=lambda sts: th.tensor([bool(free(s)) for s in sts])))
    for name in ("push_stance_stand_offs", "place_depth_cap_applies", "push_reach_x", "place_depth_caps", "place_depth_cap_filter"):
        setattr(me, name, getattr(IS.IORSSmall, name).__get__(me))
    return me, stats


def test_mode_parsing(monkeypatch):
    monkeypatch.delenv(IS.PLACE_DEPTH_CAP_ENV, raising=False)
    assert IS.place_depth_cap_mode() == "on" and IS.PLACE_DEPTH_CAP_DEFAULT == "1"  # the default since the 2026-11-08 A/B
    for v, m in (("0", "off"), ("", "on"), ("shadow", "shadow"), ("1", "on"), ("on", "on")):
        monkeypatch.setenv(IS.PLACE_DEPTH_CAP_ENV, v)
        assert IS.place_depth_cap_mode() == m


def test_reach_is_the_deepest_feasible_cell_of_a_present_class_plus_half_the_lattice(monkeypatch):
    me, _ = _me(monkeypatch, _Map(0.75))
    xr = me.push_reach_x(_Map(0.75), 0.25, -0.3)
    assert set(xr) == {0.35, 0.45, 0.55}
    assert all(v == pytest.approx(0.75 + 0.04) for v in xr.values())  # the status on the absent class at 0.83 does not count
    assert me.push_reach_x(_Map(0.75), 0.75, 0.0) == {}  # no map trunk at this translate


def test_the_cap_follows_the_nearest_collision_free_stand_off(monkeypatch):
    """The base can get no closer than its collision limit: the cap is x_reach - (that stand-off) - overhang (0 here)."""
    pm = _Map(0.75)
    me, _ = _me(monkeypatch, pm)
    c = _cand(0.30)
    f = me.place_depth_caps([c])[id(c)]
    assert f["s_min"] == 0.15 and f["cap_m"] == pytest.approx(0.79 - 0.15) and not f["remove"]
    # the bar: the base collides at stand-offs below 0.35 m
    me, _ = _me(monkeypatch, pm, free=lambda st: float(st.stand_off) >= 0.35 - 1e-9)
    deep, ok = _cand(0.50), _cand(0.40)
    facts = me.place_depth_caps([deep, ok])
    assert facts[id(deep)]["s_min"] == 0.35 and facts[id(deep)]["cap_m"] == pytest.approx(0.79 - 0.35)
    assert facts[id(deep)]["remove"] and not facts[id(ok)]["remove"]
    # no free stance at all: nothing reachable
    me, _ = _me(monkeypatch, pm, free=lambda st: False)
    c = _cand(0.10)
    f = me.place_depth_caps([c])[id(c)]
    assert f["cap_m"] is None and f["remove"]


def test_filter_on_removes_shadow_counts_off_untouched(monkeypatch):
    pm = _Map(0.75)
    bar = lambda st: float(st.stand_off) >= 0.35 - 1e-9  # noqa: E731
    cands = [_cand(0.50), _cand(0.40), _cand(0.30)]
    me, stats = _me(monkeypatch, pm, free=bar, mode="1")
    kept, facts = me.place_depth_cap_filter(list(cands))
    assert kept == cands[1:] and stats["place_depth_cap.removed"] == 1 and stats["place_depth_cap.candidates"] == 3
    me, stats = _me(monkeypatch, pm, free=bar, mode="shadow")
    kept, facts = me.place_depth_cap_filter(list(cands))
    assert kept == cands and stats["place_depth_cap.shadow_would_remove"] == 1 and "place_depth_cap.removed" not in stats
    assert facts[id(cands[0])]["remove"]
    me, stats = _me(monkeypatch, pm, free=bar, mode="0")
    kept, facts = me.place_depth_cap_filter(list(cands))
    assert kept == cands and facts == {} and stats == {}
    # not a mapped push: the shelf layout, another relation
    me, stats = _me(monkeypatch, pm, free=bar, mode="1")
    me.shelf_start = object()
    assert me.place_depth_cap_filter(list(cands)) == (cands, {})
    me.shelf_start, me.relation = None, "nextto"
    assert me.place_depth_cap_filter(list(cands)) == (cands, {})


def test_filter_all_removed_rejects_and_no_map_keeps(monkeypatch):
    me, stats = _me(monkeypatch, _Map(0.75), free=lambda st: float(st.stand_off) >= 0.55 - 1e-9)
    with pytest.raises(Reject) as e:
        me.place_depth_cap_filter([_cand(0.40), _cand(0.50)])
    assert e.value.reason == "contact_unreachable" and stats["place_depth_cap.all_removed"] == 1
    me, stats = _me(monkeypatch, None)
    cands = [_cand(0.9)]
    assert me.place_depth_cap_filter(cands) == (cands, {}) and stats["place_depth_cap.no_map"] == 1


# ------------------------------------------------------------------------------------------------ touching_04 follow-ups (lead 2026-11-08)


def test_follow_up_switch_defaults(monkeypatch):
    for env in (IS.PLACE_DEPTH_CAP_REDRAW_ENV, IS.PLACE_DEPTH_CAP_BASIS_ENV, IS.TOUCH_LIFT_SCREEN_ENV):
        monkeypatch.delenv(env, raising=False)
    assert not IS.place_depth_cap_redraw_on() and IS.place_depth_cap_basis() == "map"
    assert IS.touch_lift_screen_mode() == "on" and IS.TOUCH_LIFT_SCREEN_DEFAULT == "1"  # the default since the owner's 5r.1 (2026-11-08)
    monkeypatch.setenv(IS.TOUCH_LIFT_SCREEN_ENV, "0")
    assert IS.touch_lift_screen_mode() == "off"
    monkeypatch.setenv(IS.PLACE_DEPTH_CAP_BASIS_ENV, "a1")
    monkeypatch.setenv(IS.TOUCH_LIFT_SCREEN_ENV, "shadow")
    assert IS.place_depth_cap_basis() == "a1" and IS.touch_lift_screen_mode() == "shadow"


def test_a1_basis_replaces_the_map_verdict(monkeypatch):
    """(b): with the a1 basis a candidate is kept when A1's own push stance test passes for the target at its focus at some map yaw slot."""
    pm = _Map(0.75)
    me, stats = _me(monkeypatch, pm, free=lambda st: float(st.stand_off) >= 0.35 - 1e-9)
    monkeypatch.setenv(IS.PLACE_DEPTH_CAP_BASIS_ENV, "a1")
    me.segment_arms = ["right"]
    me.hull, me.rest = {"target": None}, {"target": None}
    me.top = SimpleNamespace(z=1.0)
    monkeypatch.setattr(TT, "object_pose_on", lambda s, hull, rest, xy, yaw, z: (th.tensor([float(xy[0]), float(xy[1]), z]), yaw))
    calls = []

    def psc(manip, pose, laterals, p_pose=None, collision=True):
        calls.append(pose[1])
        return ([(laterals[0], None, {})], {}) if abs(pose[1] - np.pi / 2) < 1e-6 else ([], {})  # only the second yaw slot passes

    me.push_stance_candidates = psc
    me._a1_reach_at_focus = IS.IORSSmall._a1_reach_at_focus.__get__(me)
    deep = _cand(0.50)  # beyond the map cap (0.44): kept by the a1 basis
    f = me.place_depth_caps([deep])[id(deep)]
    assert f["remove_map"] and not f["remove"] and f["basis"] == "a1" and f["a1_yaw_slot"] == 90.0 and len(calls) == 2
    assert stats["place_depth_cap.a1_ok_map_off"] == 1
    me.push_stance_candidates = lambda *a, **k: ([], {})
    f = me.place_depth_caps([_cand(0.30)])
    assert list(f.values())[0]["remove"] and not list(f.values())[0]["remove_map"]


def _lift_me(monkeypatch, mode, rows, gap, ik_ok, rule="auto"):
    from contextlib import contextmanager

    monkeypatch.setenv(IS.TOUCH_LIFT_SCREEN_ENV, mode)
    monkeypatch.setattr(TT, "contact_maps_on", lambda: True)

    @contextmanager
    def att(*a, **k):
        yield

    monkeypatch.setattr(IS, "attached", att)
    monkeypatch.setattr(IS.PC, "lift_end_pose", lambda g: ("lift", g.carry_height))
    seen = {}

    def exact(poses, st, arm, **kw):
        seen["h"] = poses[0][1]
        return SimpleNamespace(ok=ik_ok, q=th.zeros(3) if ik_ok else None, reason="exact IK failed at pose 1 of 1")

    stats = {}
    me = SimpleNamespace(shelf_start=None, gen=lambda k, d=None: d, kit=SimpleNamespace(stats=stats), ikc=SimpleNamespace(exact=exact),
                         carry_rule=lambda: rule, place_pose_for=lambda t, p, m: "place", push_stance_candidates=lambda *a, **k: (rows, {}),
                         drive_partner_gap=lambda t, p, frm, to: gap, carry_height_over=lambda partner: 0.31 if partner else 0.03,
                         obj=lambda r: None, support_objects=lambda: [], _lift_screen_shadow=False)
    grasp = SimpleNamespace(carry_height=0.03, arm="right", eef_in_object=None)
    return me, grasp, stats, seen


def test_lift_screen_rejects_only_where_the_goal_check_is_certain_to(monkeypatch):
    """(a): the goal check's lift test at the partner height, at the draw; certain = every A1 push stance is a real second base pose whose
    drive passes within carry_clearance_m of the partner (the auto rule then raises the box over the partner)."""
    st = SimpleNamespace(name="push")
    screen = IS.IORSSmall._touch_lift_screen
    me, g, stats, seen = _lift_me(monkeypatch, "1", [(-0.12, st, {}), (-0.16, st, {})], gap=0.0, ik_ok=False)
    with pytest.raises(Reject) as e:
        screen(me, SimpleNamespace(name="manipulation"), g, "t", "p", 3)
    assert e.value.reason == "lift_unreachable" and seen["h"] == 0.31 and g.carry_height == 0.03 and stats["touch_lift_screen.rejected"] == 1
    # shadow: counted, nothing rejected
    me, g, stats, _ = _lift_me(monkeypatch, "shadow", [(-0.12, st, {})], gap=0.0, ik_ok=False)
    screen(me, SimpleNamespace(name="manipulation"), g, "t", "p", 3)
    assert stats["touch_lift_screen.shadow_would_reject"] == 1 and me._lift_screen_shadow
    # uncertain: a drive that clears the partner (the box would stay at the table height), a manipulation-stance entry, no A1 stance
    for rows, gap in (([(-0.12, st, {})], 0.05), ([(-0.12, None, {})], 0.0), ([], 0.0)):
        me, g, stats, seen = _lift_me(monkeypatch, "1", rows, gap=gap, ik_ok=False)
        screen(me, SimpleNamespace(name="manipulation"), g, "t", "p", 3)
        assert stats == {"touch_lift_screen.uncertain": 1} and not seen
    # the partner rule is certain without the A1 test; a passing lift passes
    me, g, stats, _ = _lift_me(monkeypatch, "1", [], gap=0.0, ik_ok=True, rule="partner")
    screen(me, SimpleNamespace(name="manipulation"), g, "t", "p", 3)
    assert stats == {"touch_lift_screen.checked": 1, "touch_lift_screen.passed": 1}
    # off / the table-only rule: nothing
    me, g, stats, _ = _lift_me(monkeypatch, "1", [], gap=0.0, ik_ok=False, rule="table")
    screen(me, SimpleNamespace(name="manipulation"), g, "t", "p", 3)
    assert stats == {}


def test_phase2_keep_pairs_draws_within_the_cap(monkeypatch):
    """(c): with the draw switch on (and the cap on) Phase 2's keep_pairs keeps the (stance, position) pairs within the cap."""
    me, stats = _me(monkeypatch, _Map(0.75), free=lambda st: float(st.stand_off) >= 0.35 - 1e-9)
    me._pdc_keep_pairs = IS.IORSSmall._pdc_keep_pairs.__get__(me)
    monkeypatch.setenv(IS.PLACE_DEPTH_CAP_DRAW_ENV, "0")
    assert me._pdc_keep_pairs() is None
    monkeypatch.delenv(IS.PLACE_DEPTH_CAP_DRAW_ENV, raising=False)
    assert IS.place_depth_cap_draw_on() and IS.PLACE_DEPTH_CAP_DRAW_DEFAULT == "1"  # the default since the owner's 5r.1 (2026-11-08)
    keep = me._pdc_keep_pairs()
    cands = [_cand(0.50), _cand(0.40), _cand(0.30)]
    assert keep(cands) == cands[1:] and stats["place_depth_cap.phase2_removed"] == 1
    monkeypatch.setenv(IS.PLACE_DEPTH_CAP_ENV, "0")
    assert me._pdc_keep_pairs() is None  # the cap off: no cut
