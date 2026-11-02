"""train-fast BAND_REFINE (owner design 2026-11-01): reach bands refined per actual trunk key, only verified reachability failures remove
anything (``v2/band_refine.py``, ``tabletop.record_refined_screen`` / ``verified_exact_failure``, ``ReachBands.band``); stubs, no simulator."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import band_evidence as EV
from ispatialgym.sampling.v2 import band_refine as BR
from ispatialgym.sampling.v2 import tabletop as TT

NEAR, MID, FAR = (0.56, 0.0), (0.72, 0.0), (0.96, 0.08)


def _band(tmp_path, name="b"):
    """Grid tilts -0.9 (NEAR, MID) and -1.2 (NEAR, MID, FAR) at translate 0.25, -0.3 (NEAR) at 0.5; a file path for the sidecars."""
    cells = {(0.25, -0.9): np.array([NEAR, MID]), (0.25, -1.2): np.array([NEAR, MID, FAR]), (0.5, -0.3): np.array([NEAR])}
    path = str(tmp_path / f"{name}.npz")
    open(path, "wb").close()
    return RB.ReachBands({"lattice": 0.08}, cells, {}, path, {}, {}, {}, {})


def _cells(a):
    return sorted((round(float(x), 3), round(float(y), 3)) for x, y in np.asarray(a).reshape(-1, 2))


def test_a_new_key_starts_from_the_nearest_trunk_or_the_bracket(tmp_path, monkeypatch):
    b = _band(tmp_path)
    assert _cells(b.band(0.25, -1.07)) == _cells([NEAR, MID, FAR])  # nearest (the default): a copy of the nearest grid trunk (-1.2)
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_REFINE_INIT", "bracket")
    b2 = _band(tmp_path, "b2")
    assert _cells(b2.band(0.25, -1.07)) == _cells([NEAR, MID])  # bracket: the cells -0.9 and -1.2 both hold (FAR is -1.2's only)
    assert _cells(b2.band(0.25, -1.2)) == _cells([NEAR, MID, FAR])  # on a grid tilt: that trunk's cells
    assert _cells(b2.band(0.25, -1.35)) == _cells([NEAR, MID, FAR])  # beyond the grid: the nearest trunk's
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_REFINE", "0")
    assert _cells(_band(tmp_path, "b3").band(0.25, -1.07)) == _cells([NEAR, MID, FAR])  # off: the grid trunk as before


class _IK:
    """feasible_sets: the pairs reachable at a cell, minus those a task object blocks while it is in the world; exact: likewise."""

    def __init__(self, reach, blocked_by_objects=()):
        self.reach, self.blocked, self.out = reach, set(blocked_by_objects), False

    @contextmanager
    def ignoring(self, objs):
        self.out = True
        try:
            yield
        finally:
            self.out = False

    def feasible_sets(self, cells, yaws, grasps, pose_at, stance, arm, ignore, eef_of, cache=True):
        key = (round(float(cells[0][0]), 3), round(float(cells[0][1]), 3))
        pairs = [] if (key in self.blocked and not self.out) else list(self.reach.get(key, []))
        return {0: pairs} if pairs else {}

    def exact(self, poses, stance, arm, **kw):
        key = poses[0]
        ok = key in self.reach and (self.out or key not in self.blocked)
        return SimpleNamespace(ok=ok, reason="" if ok else "exact IK failed")


def _rec_screen(ik, store, cell, objs=("partner",)):
    s = SimpleNamespace(ikc=ik)
    ck = BR.cell_key((0.25, -1.07), cell)
    args = ([np.asarray(cell)], [0.0], ["g"], None, None, "right", [], None)
    found = ik.feasible_sets(*args).get(0, [])
    return TT.record_refined_screen(s, (store, BR.WORLD, ck), found, args, objs, lambda g: g), ck


def test_a_non_reach_failure_never_blacklists(tmp_path):
    b = _band(tmp_path)
    store = BR.store_of(b)
    ik = _IK({MID: [(0.0, "g")]}, blocked_by_objects=[MID])  # reachable, but a task object is in the way
    for _ in range(5):
        verdict, ck = _rec_screen(ik, store, MID)
        assert verdict == "other"
    assert not BR.dead(store.record(BR.WORLD, ck)) and MID in [tuple(c) for c in _cells(b.band(0.25, -1.07))]
    s = SimpleNamespace(ikc=ik)
    assert not TT.verified_exact_failure(s, MID, None, None, "right", {}, ["partner"])  # the exact IK passes with the objects out


def test_a_verified_reach_failure_blacklists_after_k(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_REFINE_K", "2")
    b = _band(tmp_path)
    store = BR.store_of(b)
    ik = _IK({})  # MID out of reach whatever is in the world
    verdict, ck = _rec_screen(ik, store, MID)
    assert verdict == "reach" and MID in [tuple(c) for c in _cells(b.band(0.25, -1.07))]  # one failure is not enough
    _rec_screen(ik, store, MID)
    assert BR.dead(store.record(BR.WORLD, ck)) and MID not in [tuple(c) for c in _cells(b.band(0.25, -1.07))]
    assert MID in [tuple(c) for c in _cells(b.band(0.25, -0.9))]  # another trunk key keeps it
    assert TT.verified_exact_failure(SimpleNamespace(ikc=ik), MID, None, None, "right", {}, ["partner"])


def test_a_cell_once_found_reachable_is_never_blacklisted(tmp_path):
    b = _band(tmp_path)
    store = BR.store_of(b)
    _rec_screen(_IK({MID: [(0.0, "g")]}), store, MID)
    for _ in range(4):
        _rec_screen(_IK({}), store, MID)
    assert MID in [tuple(c) for c in _cells(b.band(0.25, -1.07))]


def test_additions_join_the_key(tmp_path):
    b = _band(tmp_path)
    store = BR.store_of(b)
    assert BR.neighbours(b.band(0.25, -1.07), MID, 0.08) == [(0.8, 0.0), (0.64, 0.0), (0.72, 0.08), (0.72, -0.08)]
    store.add_cell(BR.cell_key((0.25, -1.07), (0.80, 0.0)), [(0.0, "g")])
    assert (0.8, 0.0) in [tuple(c) for c in _cells(b.band(0.25, -1.07))]
    assert (0.8, 0.0) not in [tuple(c) for c in _cells(b.band(0.25, -0.9))]  # only for that key
    assert BR.store_of(_band(tmp_path)).record(BR.WORLD, BR.cell_key((0.25, -1.07), (0.8, 0.0)))["added"]  # persisted next to the band


def test_per_job_deltas_merge_as_a_union_without_double_counting(tmp_path):
    """The real flow: a job fetches at its start (its own records still empty) and publishes only its own records at its end."""
    share = tmp_path / "share"
    dirs = {j: tmp_path / j for j in "abc"}
    for d in dirs.values():
        d.mkdir()
    ck = BR.cell_key((0.25, -1.07), MID)
    added = BR.cell_key((0.25, -1.07), (0.8, 0.0))
    # job a: one verified empty screen and one addition
    ba = _band(dirs["a"], "band")
    EV.fetch(str(dirs["a"]), str(share))
    BR.store_of(ba).add_screen(BR.WORLD, ck, [])
    BR.store_of(ba).add_cell(added, [(0.0, "g")])
    EV.publish(str(dirs["a"]), str(share), "1")
    # job b starts after a: a's records come in as its base; it adds one more empty screen of its own
    bb = _band(dirs["b"], "band")
    EV.fetch(str(dirs["b"]), str(share))
    sb = BR.store_of(bb)
    assert sb.record(BR.WORLD, ck)["empty"] == 1 and not sb.own
    sb.add_screen(BR.WORLD, ck, [])
    EV.publish(str(dirs["b"]), str(share), "2")
    # job c: the union of a's and b's own records - b's base (a's records) was not re-published
    bc = _band(dirs["c"], "band")
    EV.fetch(str(dirs["c"]), str(share))
    sc = BR.store_of(bc)
    assert sc.record(BR.WORLD, ck)["empty"] == 2 and sc.record(BR.WORLD, added)["added"]
    assert BR.dead(sc.record(BR.WORLD, ck))  # K = 2 verified failures across jobs blacklist the cell for job c
    assert MID not in [tuple(c) for c in _cells(bc.band(0.25, -1.07))] and (0.8, 0.0) in [tuple(c) for c in _cells(bc.band(0.25, -1.07))]
    assert os.path.exists(str(dirs["c"] / "band.refine.base.json")) and os.path.exists(str(share / "band.refine.2.json"))
