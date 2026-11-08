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
    me = SimpleNamespace(relation="touching", shelf_start=None, shelf_pose=None, top=None, gen=lambda k, d=None: d,
                         kit=SimpleNamespace(stats=stats), ikc=SimpleNamespace(stances_collision_free=lambda sts: th.tensor([bool(free(s)) for s in sts])))
    for name in ("push_stance_stand_offs", "place_depth_cap_applies", "push_reach_x", "place_depth_caps", "place_depth_cap_filter"):
        setattr(me, name, getattr(IS.IORSSmall, name).__get__(me))
    return me, stats


def test_mode_parsing(monkeypatch):
    monkeypatch.delenv(IS.PLACE_DEPTH_CAP_ENV, raising=False)
    assert IS.place_depth_cap_mode() == "off" and IS.PLACE_DEPTH_CAP_DEFAULT == "0"
    for v, m in (("0", "off"), ("", "off"), ("shadow", "shadow"), ("1", "on"), ("on", "on")):
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
