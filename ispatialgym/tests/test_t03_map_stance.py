"""i7-t03both r2 (lead #84b, both DEFAULT OFF): ISPATIALGYM_SHELF_MAP_STANCE - at the fixed shelf start the working arm's interior map is read
at the book (``map_book_feasible``) and the stance shift tries first the MAP stances (``shelf_map_stances``: the base moved so the book lands
on a feasible map cell, at that cell's stand-off / trunk); ISPATIALGYM_SHELF_FREE_START_CONNECTED - the free-floor start keeps the candidates'
eroded-map region (``free_component``). Simulator-free (the touching_03 frame of test_t03_stance_shift, a synthetic map)."""

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.tests.test_t03_stance_shift import BOOK, IS, _cand, _sampler

LAT = 0.08
XS = [0.43, 0.51, 0.59, 0.67]
YS = [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24]


def _map(feasible):
    """A raw interior map: cells on the 8 cm lattice, trunk keys s<d_e>_t<t>_k<k>, status 1 (feasible) at ``feasible`` {key: [(x, y)]}, else 5."""
    cells = np.array([[x, y] for x in XS for y in YS], dtype=float)
    z = {"cells": cells}
    for key in ("s0.55_t0.25_k+0.0", "s0.45_t0.25_k+0.0", "s0.55_t0.50_k+0.0"):
        a = np.full((len(cells), 1, 2), 5, dtype=np.uint8)
        for (x, y) in feasible.get(key, []):
            i = int(np.argmin(np.abs(cells - np.array([x, y])).max(axis=1)))
            a[i, 0, 1] = 1
        z[key] = a
    return {"edge_distances": [0.35, 0.45, 0.55], "config": {"lattice_m": LAT}}, z


def _s(feasible, monkeypatch):
    s = _sampler()
    s.kit.stats = {}
    rm = _map(feasible)
    monkeypatch.setattr(IS.IORSSmall, "_raw_map", lambda self, arm: rm)
    return s


def _book_xy(s, st):
    return s._diag_base_frame(st, BOOK[0])


def test_the_switches_are_off_by_default(monkeypatch):
    for env, fn in ((IS.SHELF_MAP_STANCE_ENV, IS.shelf_map_stance_on), (IS.SHELF_FREE_START_CONNECTED_ENV, IS.shelf_free_start_connected_on)):
        monkeypatch.delenv(env, raising=False)
        tip = type("T", (), {"shelf_tip": lambda self: True})()
        assert fn(tip) is False
        monkeypatch.setenv(env, "1")
        assert fn(tip) is True


def test_the_map_is_read_at_the_book(monkeypatch):
    cand = _cand(-0.465)  # the book 0.19 m left of the base, 0.557 m ahead (this frame's book)
    s = _s({"s0.55_t0.25_k+0.0": [(0.59, 0.16)]}, monkeypatch)
    x, y = _book_xy(s, cand)
    assert x == pytest.approx(0.557, abs=0.01) and y == pytest.approx(0.19, abs=0.01)
    assert s.map_book_feasible(cand, "left") is True  # cell (0.59, 0.16): 3.3 / 3 cm off, within half the lattice
    assert s.map_book_feasible(_cand(-0.665), "left") is False  # the book at y 0.39: off the lattice's feasible cell
    t = _cand(-0.465).with_trunk(0.95, -0.157, "manipulation")
    assert s.map_book_feasible(t, "left") is False  # a trunk the map does not hold has no verdict -> infeasible
    monkeypatch.setattr(IS.IORSSmall, "_raw_map", lambda self, arm: None)
    assert s.map_book_feasible(cand, "left") is None


def test_map_stances_put_the_book_on_feasible_cells(monkeypatch):
    feas = {"s0.55_t0.25_k+0.0": [(0.59, -0.08), (0.59, 0.0)], "s0.45_t0.25_k+0.0": [(0.43, 0.08)], "s0.55_t0.50_k+0.0": [(0.59, 0.24)]}
    s = _s(feas, monkeypatch)
    cand = _cand(-0.065)  # the book at y +0.59: no feasible cell there
    out = s.shelf_map_stances(cand, "right")
    assert out and all(lbl == "map" for lbl, _, _ in out) and len(out) <= IS.SHELF_MAP_STANCE_MAX
    seen = set()
    for _, st, shift in out:
        x, y = _book_xy(s, st)
        key = s._map_trunk_key(st.stand_off, st.translate, st.tilt)
        cell = (min(XS, key=lambda v: abs(v - x)), min(YS, key=lambda v: abs(v - y)))
        assert abs(cell[0] - x) <= LAT / 2 + 1e-6 and abs(cell[1] - y) < 1e-6  # the book exactly on the cell's lateral, its depth within the row
        assert (round(cell[0], 2), round(cell[1], 2)) in [(round(a, 2), round(b, 2)) for a, b in feas[key]]
        assert s.map_book_feasible(st, "right") is True
        seen.add(key)
    assert seen == set(feas)
    # the candidate's own trunk first (0.25 / 0), the other trunk last
    assert (out[0][1].translate, out[-1][1].translate) == (0.25, 0.50) and out[-1][1].extras.get("stance_shift_trunk") == [0.5, 0.0]
    assert s.kit.stats["map_stance.generated"] == len(out)


def test_a_map_infeasible_candidate_goes_straight_to_the_map_stances(monkeypatch):
    monkeypatch.setenv(IS.SHELF_MAP_STANCE_ENV, "1")
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    monkeypatch.delenv(IS.SHELF_JOINT_STANCE_ENV, raising=False)
    s = _s({"s0.55_t0.25_k+0.0": [(0.59, 0.0)]}, monkeypatch)
    calls = []

    def try_fn(st, again=False):
        calls.append(round(float(_book_xy(s, st)[1]), 3))
        if abs(_book_xy(s, st)[1]) > 1e-6:
            raise Reject("grasp_unreachable", "exact IK failed at pose 1 of 2")
        return type("Gr", (), {"extras": {}, "grasp_name": "side_1"})()

    monkeypatch.setattr(IS.IORSSmall, "shelf_stance_ok", lambda self, st, cand, arm: (st, ""))
    cand = _cand(-0.065)
    st, g = s.shelf_fixed_start(None, cand, cand, "right", try_fn)
    assert calls == [0.0]  # the candidate itself never ran the exact checks; the first map stance passed
    assert st.extras["stance_shift"]["label"] == "map" and s.kit.stats["map_stance.candidate.infeasible"] == 1
    assert s.kit.stats["stance_shift.map_passed"] == 1
    # off: the candidate runs first (as before)
    monkeypatch.setenv(IS.SHELF_MAP_STANCE_ENV, "0")
    calls.clear()
    s._stance_shift_budget = {"scan": 0, "seen": set()}
    with pytest.raises(Reject):
        s.shelf_fixed_start(None, cand, cand, "right", lambda st, again=False: (calls.append(1), (_ for _ in ()).throw(Reject("no_path", "x")))[1])
    assert calls == [1]


def test_free_component_labels_the_eroded_map(monkeypatch):
    from ispatialgym.sampling.v2 import navigation as NAV

    s = _sampler()
    free = th.zeros(10, 10, dtype=th.bool)
    free[1:4, 1:4] = True
    free[6:9, 6:9] = True
    monkeypatch.setattr(NAV, "eroded_map", lambda kit, floor=0: free)
    monkeypatch.setattr(NAV, "_to_map", lambda kit, xy: (int(xy[0]), int(xy[1])))
    a, b, c = s.free_component((2, 2)), s.free_component((7, 7)), s.free_component((3, 1))
    assert a is not None and b is not None and a != b and a == c
    assert s.free_component((4, 4)) == a  # snapped to the nearest free cell
