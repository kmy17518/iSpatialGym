"""train-fast 2026-11-01: band evidence (``v2/band_evidence``), job-local bands (``reach_bands.BAND_JOB_LOCAL``) and the growing
exact-IK cache entry (``IKChecker._grown_entry``) - pure parts, no simulator."""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import band_evidence as EV
from ispatialgym.sampling.v2.ik import IKChecker

W = EV.world_key("house", ["dining_room_0"], {"mode": "region", "center_object": "table", "radius_m": 2.5})
C = EV.cell_key((0.5, 0.0), (0.4, -0.1))


def test_world_key_separates_crops():
    assert EV.world_key("house", ["b", "a"]) == EV.world_key("house", ["a", "b"], {"mode": "full"})
    assert W != EV.world_key("house", ["dining_room_0"])


def test_feasible_set_is_the_union_of_screens_minus_demoted(tmp_path):
    ev = EV.Evidence(str(tmp_path / "band.npz"))
    ev.add_screen(W, C, [(0.0, "g1"), (1.571, "g2")])
    ev.add_screen(W, C, [(0.0, "g1"), (3.142, "g3")])  # a re-screen finds g3: the set grows
    assert sorted(g for _, g in EV.feasible_pairs(ev.record(W, C))) == ["g1", "g2", "g3"]
    for _ in range(2):
        ev.add_exact(W, C, 1.571, "g2", False)
    assert "g2" in [g for _, g in EV.feasible_pairs(ev.record(W, C))]  # two failures are not evidence enough
    ev.add_exact(W, C, 1.571, "g2", False)
    assert "g2" not in [g for _, g in EV.feasible_pairs(ev.record(W, C))]  # demoted after PAIR_FAIL_K
    for _ in range(4):
        ev.add_exact(W, C, 0.0, "g1", False)
    ev.add_exact(W, C, 0.0, "g1", True)
    ev.add_exact(W, C, 0.0, "g1", True)
    assert "g1" in [g for _, g in EV.feasible_pairs(ev.record(W, C))]  # 4 failures vs 2 confirmations: kept (not > 2x)


def test_a_cell_dies_only_after_repeated_empty_screens(tmp_path):
    ev = EV.Evidence(str(tmp_path / "band.npz"))
    for k in range(EV.CELL_EMPTY_M - 1):
        ev.add_screen(W, C, [])
        assert not EV.cell_dead(ev.record(W, C))
    ev.add_screen(W, C, [])
    assert EV.cell_dead(ev.record(W, C))
    ev.add_screen(W, C, [(0.0, "g1")])
    assert not EV.cell_dead(ev.record(W, C))  # once a pair was found the cell is never dead


def test_worlds_do_not_read_each_others_counts(tmp_path):
    ev = EV.Evidence(str(tmp_path / "band.npz"))
    ev.add_screen(W, C, [(0.0, "g1")])
    assert ev.record(EV.world_key("house", ["dining_room_0"]), C) is None


def test_own_counts_persist_and_share_never_double_counts(tmp_path):
    job_a, job_b, share = tmp_path / "a", tmp_path / "b", tmp_path / "share"
    for d in (job_a, job_b):
        d.mkdir()
        (d / "band.npz").write_bytes(b"")
    ev = EV.Evidence(str(job_a / "band.npz"))
    ev.add_screen(W, C, [(0.0, "g1")])
    assert EV.Evidence(str(job_a / "band.npz")).record(W, C)["screens"] == 1  # written back atomically
    assert EV.publish(str(job_a), str(share), "1") == 1
    assert EV.fetch(str(job_b), str(share)) == 1
    eb = EV.Evidence(str(job_b / "band.npz"))
    assert eb.record(W, C)["screens"] == 1 and not eb.own  # fetched into the base, not the job's own counts
    eb.add_screen(W, C, [])
    EV.publish(str(job_b), str(share), "2")
    EV.fetch(str(job_a), str(share))
    assert EV.Evidence(str(job_a / "band.npz")).record(W, C)["screens"] == 3  # base (1 + 1) + own 1: job b's base was not re-published


def _band(d, name, mtime):
    meta = {"scene": "house", "support": "table", "object": "cup", "lattice": 0.05, "trunks": [[0.5, 0.0]]}
    (d / f"{name}.json").write_text(json.dumps(meta))
    (d / f"{name}.npz").write_bytes(b"x")
    for f in (d / f"{name}.json", d / f"{name}.npz"):
        os.utime(f, (mtime, mtime))


def test_the_jobs_copy_wins_over_a_newer_committed_band(tmp_path, monkeypatch):
    job, committed = tmp_path / "job", tmp_path / "committed"
    job.mkdir(), committed.mkdir()
    now = time.time()
    _band(job, "b1", now - 100)
    _band(committed, "b1", now)  # rewritten by another job: newer
    monkeypatch.setenv("ISPATIALGYM_REACH_BANDS", str(job))
    got = RB.ReachBands.find("house", "table", "cup", dirs=[str(job), str(committed)])
    assert got == [str(job / "b1.npz")]
    monkeypatch.setenv("ISPATIALGYM_FAST_BAND_JOB_LOCAL", "0")
    assert RB.ReachBands.find("house", "table", "cup", dirs=[str(job), str(committed)])[0] == str(committed / "b1.npz")  # the old order


def test_a_band_outside_the_job_is_copied_in(tmp_path, monkeypatch):
    job, committed = tmp_path / "job", tmp_path / "committed"
    job.mkdir(), committed.mkdir()
    _band(committed, "b2", time.time())
    monkeypatch.setenv("ISPATIALGYM_REACH_BANDS", str(job))
    dst = RB.job_local_copy(str(committed / "b2.npz"))
    assert dst == str(job / "b2.npz") and os.path.exists(dst) and os.path.exists(str(job / "b2.json"))
    assert os.path.getmtime(dst) == pytest.approx(os.path.getmtime(committed / "b2.npz"))
    monkeypatch.delenv("ISPATIALGYM_REACH_BANDS")
    assert RB.job_local_copy(str(committed / "b2.npz")) == str(committed / "b2.npz")


def _chain(v):
    q = [0.0] * 28
    q[3] = v
    return [q, q]


def test_ik_cache_entry_grows_by_branch():
    stub = SimpleNamespace(kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.tensor([3, 4])})), counters={})
    stub._count = lambda k, n=1: stub.counters.__setitem__(k, stub.counters.get(k, 0) + n)
    grow = IKChecker._grown_entry.__get__(stub)
    stub.BRANCH_MIN_RAD, stub.MAX_CHAINS = IKChecker.BRANCH_MIN_RAD, IKChecker.MAX_CHAINS
    old = {"ok": True, "q": _chain(0.0)[-1], "qs": _chain(0.0)}
    same = grow(old, {"ok": True, "q": None, "qs": _chain(0.1)}, "right")
    assert same is old  # within BRANCH_MIN_RAD: the same branch
    e = grow(old, {"ok": True, "q": None, "qs": _chain(1.0)}, "right")
    assert e["qs"] == old["qs"] and e["alts"] == [_chain(1.0)]
    for k in range(10):
        e = grow(e, {"ok": True, "q": None, "qs": _chain(2.0 + k)}, "right")
    assert len(e["alts"]) == IKChecker.MAX_CHAINS - 1
    assert grow(None, {"ok": True, "q": None, "qs": _chain(5.0)}, "right")["qs"] == _chain(5.0)


def test_exact_ik_scope_is_shared_between_crop_and_full(monkeypatch):
    full = SimpleNamespace(cache_scope={"scene": "house", "rooms": ["dining", "kitchen", "living"], "floor_z": 0.0})
    crop = SimpleNamespace(cache_scope={"scene": "house", "rooms": ["dining", "living"], "floor_z": 0.0, "scene_crop": "region:table:2.5",
                                        "full_rooms": ["dining", "kitchen", "living"]})
    scope = lambda o: IKChecker.exact_cache_scope.__get__(o)()
    assert scope(crop) == scope(full) == full.cache_scope  # the crop reads and writes the full world's exact-IK entries
    assert crop.cache_scope != full.cache_scope  # the feasible-set / start-draw keys (cache_scope itself) keep the crop
    monkeypatch.setenv("ISPATIALGYM_FAST_IK_CACHE_SHARE_CROP", "0")
    assert scope(crop)["scene_crop"] == "region:table:2.5" and "full_rooms" not in scope(crop)


def test_an_unreachable_cell_is_dead_at_once_and_stays_dead_when_shared(tmp_path):
    ev = EV.Evidence(str(tmp_path / "band.npz"))
    ev.add_screen(W, C, [(0.0, "g1")])
    ev.mark_unreachable(W, C)
    assert EV.cell_dead(ev.record(W, C))
    assert EV.merge_records(EV.new_record(), ev.record(W, C)).get("unreachable") is True
