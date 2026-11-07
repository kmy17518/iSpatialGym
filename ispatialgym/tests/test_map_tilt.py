"""i7-plateqtf 2026-11-07 (owner 23:40 item 3): the expanding, self-correcting reach map (v2/map_tilt.py). Simulator-free.

- a missing tilt inherits the union of its two built neighbours (cells and pairs), and an in-between yaw the union of its two slots;
- a repeated reachability failure invalidates an inherited cell, a collision / support-policy failure never does, a success confirms;
- a built tilt / yaw slot reads the map unchanged (and so does everything with the switch off);
- the overlay never touches the frozen map: per-process shards, an offline merge into new version files.
"""

import glob
import hashlib
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import map_tilt as MT
from ispatialgym.sampling.v2 import tabletop as TT

# cells (base frame) of the fake map; tilts -0.3 and -0.6 at translate 0.25 (and 0.0 / -0.9 to have a range)
CELLS = np.array([[0.51, 0.0], [0.59, 0.0], [0.67, 0.0], [0.51, 0.08]])
FEAS = {-0.0: [0], -0.3: [0, 1], -0.6: [1, 2], -0.9: [2]}  # tilt -> feasible cells (at yaw slot 0 / grasp 0; cell 3 never)


def _map(d):
    yaws = np.array([k * math.pi / 4 for k in range(8)])
    grasps = [{"name": "top_0", "base": "top_0", "twin": False, "param": None, "family_id": None},
              {"name": "side_1", "base": "side_1", "twin": False, "param": None, "family_id": None}]
    arrays = {"cells": CELLS, "yaws": yaws}
    trunks = []
    for ti, cs in FEAS.items():
        st = np.full((len(CELLS), 8, 2), RM.STATUS["ik_grasp"], dtype=np.uint8)
        for c in cs:
            st[c, 0, 0] = RM.STATUS["feasible"]
        if ti == -0.6:
            st[1, 2, 1] = RM.STATUS["feasible"]  # a second pair on cell 1 at -0.6 only: yaw slot 2, side_1
        arrays[f"s0.45_t0.25_k{ti:+.1f}"] = st
        trunks.append({"translate": 0.25, "tilt": ti, "eligible": True})
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "key": "mkey0123456789ab", "edge_distances": [0.45], "config": {"lattice_m": 0.08},
            "trunks": trunks, "grasps": grasps, "arm": "right", "height": {"value": 0.8}}
    base = os.path.join(str(d), "map__a__b__h0.8000__mkey0123456789ab")
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    np.savez_compressed(base + ".npz", **arrays)
    return base + ".json"


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    MT._OVERLAYS.clear()
    monkeypatch.setenv("ISPATIALGYM_FAST_MAP_TILT_INHERIT", "1")
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY", raising=False)
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY_VERSION", raising=False)
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", raising=False)
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_SHARDS", str(tmp_path / "shards"))
    (tmp_path / "maps").mkdir()
    mp = _map(tmp_path / "maps")
    view = MS.as_bands(RM.ReachMap(mp), overhang=0.0)
    s = SimpleNamespace(kit=SimpleNamespace(stats={}))
    view._map_tilt_sampler = s
    yield SimpleNamespace(view=view, s=s, map_path=mp, tmp=tmp_path)
    MT._OVERLAYS.clear()


def _xy(a):
    return sorted(tuple(np.round(c, 3)) for c in np.asarray(a).reshape(-1, 2))


def test_missing_tilt_inherits_the_union_of_its_neighbours(env):
    v = env.view
    got = v.band(0.25, -0.53, stand_off=0.45)
    assert _xy(got) == _xy(CELLS[[0, 1, 2]])  # -0.3 holds {0, 1}, -0.6 holds {1, 2}: the union
    assert env.s.kit.stats["map_tilt.inherited_columns"] == 1 and env.s.kit.stats["map_tilt.inherited_feasible"] == 3
    v.band(0.25, -0.531, stand_off=0.45)  # the same column (0.01 rad key): not counted again
    assert env.s.kit.stats["map_tilt.inherited_columns"] == 1
    # pairs: the union, re-indexed onto the column's cells (cell 1 has slot 0 / top_0 from both and slot 2 / side_1 from -0.6)
    f = v.feasible(0.25, -0.53, CELLS[1], stand_off=0.45)
    assert sorted((round(y, 4), g) for y, g in f) == sorted([(0.0, "top_0"), (round(math.pi / 2, 4), "side_1")])
    assert v.cell_index(0.25, -0.53, CELLS[3], stand_off=0.45) is None  # in neither neighbour
    # the pure union and the bracket
    assert MT.bracket(list(v.cells.keys()), 0.25, -0.53) == ((0.25, -0.6), (0.25, -0.3))
    assert MT.bracket(list(v.cells.keys()), 0.25, -1.0) is None  # outside the built range: one neighbour, no column
    assert MT.bracket(list(v.cells.keys()), 0.5, -0.53) is None  # no trunk of that translate


def test_in_between_yaw_inherits_the_union_of_its_slots():
    yaws = [k * math.pi / 4 for k in range(8)]
    st = np.zeros((8, 3), dtype=np.uint8)
    st[0, 0] = 1
    st[1, 1] = 1
    st[7, 2] = 1
    row, br = MT.yaw_union(st, yaws, math.radians(20))
    assert br == (0, 1) and row.tolist() == [True, True, False]
    row, br = MT.yaw_union(st, yaws, math.radians(-20))  # between slot 7 (315 deg) and slot 0
    assert br == (7, 0) and row.tolist() == [True, False, True]
    row, br = MT.yaw_union(st, yaws, math.radians(45))  # on a slot: that slot unchanged
    assert br is None and row.tolist() == [False, True, False]


def test_existing_tilt_and_switch_off_are_unchanged(env, monkeypatch):
    v = env.view
    for ti, cs in FEAS.items():
        assert _xy(v.band(0.25, ti, stand_off=0.45)) == _xy(CELLS[cs])
    assert _xy(v.band(0.25, -0.602, stand_off=0.45)) == _xy(CELLS[FEAS[-0.6]])  # within EPS_TILT of a built tilt
    assert _xy(v.band(0.25, -1.05, stand_off=0.45)) == _xy(CELLS[FEAS[-0.9]])  # outside the range: the nearest, as before
    assert "map_tilt.inherited_columns" not in env.s.kit.stats
    monkeypatch.setenv("ISPATIALGYM_FAST_MAP_TILT_INHERIT", "0")
    assert _xy(v.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[FEAS[-0.6]])  # off: the nearest built tilt
    assert MT.view_tag("mkey0123456789ab") is None


def test_reach_failure_repeated_invalidates_collision_never(env):
    v, s = env.view, env.s
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    fe = MT.focus_entry(v, st, CELLS[0], 0.45)
    assert fe is not None and fe.cell == "0.510,0.000"
    assert MT.focus_entry(v, SimpleNamespace(**{**st.__dict__, "tilt": -0.6}), CELLS[0], 0.45) is None  # a built tilt: no inherited entry
    # collisions / the support policy / the camera frame never count, however often
    for why in ("support_policy: non-finger link touches the support", "collides with breakfast_table", "gripper not in frame"):
        assert not MT.reach_reason(why)
        MT.note(s, fe, "other_fail", why=why)
    assert not fe.overlay.dead(fe.entry) and len(v.band(0.25, -0.53, stand_off=0.45)) == 3
    assert MT.reach_reason("exact IK failed (pose 1)") and MT.reach_reason("ik_joint_limit: right_arm_joint2") and MT.reach_reason("ik_reach 0.012")
    # one reachability failure: kept (IK seeds are random); the second: invalidated, Phase 2 no longer reads the cell
    MT.note(s, fe, "reach_fail", why="empty F; scene out: 0 pairs")
    assert not fe.overlay.dead(fe.entry) and len(v.band(0.25, -0.53, stand_off=0.45)) == 3
    MT.note(s, fe, "reach_fail", why="empty F; scene out: 0 pairs")
    assert fe.overlay.dead(fe.entry)
    assert _xy(v.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[1, 2]])
    assert s.kit.stats["map_tilt.cells_invalidated"] == 1 and s.kit.stats["map_tilt.reach_fail"] == 2 and s.kit.stats["map_tilt.other_fail"] == 3
    assert s.kit.stats["map_tilt.by_map.mkey0123.cells_invalidated"] == 1
    assert _xy(v.band(0.25, -0.3, stand_off=0.45)) == _xy(CELLS[FEAS[-0.3]])  # the built column keeps the cell
    # a confirmed cell is never invalidated
    fe1 = MT.focus_entry(v, st, CELLS[1], 0.45)
    MT.note(s, fe1, "ok", yaw=0.0, grasp="top_0")
    for _ in range(4):
        MT.note(s, fe1, "reach_fail")
    assert not fe1.overlay.dead(fe1.entry) and s.kit.stats["map_tilt.cells_confirmed"] == 1 and s.kit.stats["map_yaw.pairs_confirmed"] == 1


def test_sampler_hooks_count_empty_f_and_verify_with_the_scene_out(env, monkeypatch):
    """tabletop.map_tilt_empty_f: an empty F verified with every scene object out - still empty counts, pairs found (the scene) do not."""
    v, s = env.view, env.s
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    reach = {"ok": False}

    class _Ign:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    s.ikc = SimpleNamespace(ignoring=lambda objs, **kw: _Ign(),
                            feasible_sets=lambda *a, **k: ({0: [(0.0, "g")]} if reach["ok"] else {}))
    s.world = SimpleNamespace(scene=SimpleNamespace(objects=[]))
    monkeypatch.setattr(TT, "is_map_view", lambda b: True)
    fes = TT.map_tilt_entries(s, {"right": v}, {"right": st}, ["right"], np.asarray(CELLS[0]))
    assert list(fes) == ["right"]
    reach["ok"] = True
    TT.map_tilt_empty_f(s, fes, CELLS[0], [0.0], ["g"], None, {"right": st}, [], None)
    TT.map_tilt_empty_f(s, fes, CELLS[0], [0.0], ["g"], None, {"right": st}, [], None)
    assert not fes["right"].overlay.dead(fes["right"].entry)  # the scene caused both
    reach["ok"] = False
    TT.map_tilt_empty_f(s, fes, CELLS[0], [0.0], ["g"], None, {"right": st}, [], None)
    TT.map_tilt_empty_f(s, fes, CELLS[0], [0.0], ["g"], None, {"right": st}, [], None)
    assert fes["right"].overlay.dead(fes["right"].entry)
    assert s.kit.stats["map_tilt.empty_f_on_inherited"] == 4 and s.kit.stats["map_tilt.cells_invalidated"] == 1
    # the cached candidate standing on it leaves the draws
    s.bands = v
    cand = SimpleNamespace(**st.__dict__, focus=np.array([CELLS[0][0], CELLS[0][1], 0.9]))
    assert TT.map_tilt_focus_dead(s, cand)


def test_overlay_never_touches_the_frozen_map(env, monkeypatch):
    v, s = env.view, env.s
    npz = env.map_path[:-5] + ".npz"
    before = (_sha(env.map_path), _sha(npz), sorted(os.listdir(os.path.dirname(env.map_path))))
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    fe = MT.focus_entry(v, st, CELLS[0], 0.45)
    MT.note(s, fe, "reach_fail")
    # a second process (another job) writes its own shard: no shared file, no race
    other = MT.Overlay(fe.map_key, shard_path=str(env.tmp / "shards" / fe.map_key / "otherhost_1.jsonl"))
    other.record(fe.entry, "reach_fail")
    assert (_sha(env.map_path), _sha(npz), sorted(os.listdir(os.path.dirname(env.map_path)))) == before
    shards = sorted(glob.glob(str(env.tmp / "shards" / "*" / "*.jsonl")))
    assert len(shards) == 2 and all(os.path.dirname(env.map_path) not in p for p in shards)
    root = str(env.tmp / "overlay")
    res = MT.merge(shards, root)
    assert res[fe.map_key]["version"] == 1 and res[fe.map_key]["new_events"] == 2
    assert MT.is_dead(res[fe.map_key]["entries"][fe.entry])  # two jobs' single failures add up offline
    assert MT.merge(shards, root)[fe.map_key]["new_events"] == 0  # already folded: no new version
    assert sorted(os.listdir(os.path.join(root, fe.map_key))) == ["v1.json"]
    assert (_sha(env.map_path), _sha(npz)) == before[:2]
    # a job that reads the merged overlay starts from it, and its identity names the version
    MT._OVERLAYS.clear()
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY", root)
    assert MT.view_tag(fe.map_key) == "inherit_k2+ov1"
    v2 = MS.as_bands(RM.ReachMap(env.map_path), overhang=0.0)
    assert _xy(v2.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[1, 2]])


def _shard(env, name, entry, outcomes, mk="mkey0123456789ab"):
    ov = MT.Overlay(mk, shard_path=str(env.tmp / "shards" / mk / f"{name}.jsonl"))
    for o in outcomes:
        ov.record(entry, o)
    return ov.shard_path


def test_jobs_read_a_pinned_release_not_the_latest(env, monkeypatch):
    """i7-plateqtf2 (owner 2026-11-08 02:35 item 5): every merge writes a release naming each map key's latest version; a job pinned to a
    release (ISPATIALGYM_MAP_OVERLAY_RELEASE, job.env) reads exactly those versions - a later merge changes nothing for it - and its view
    identity names the version it applies."""
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    entry = MT.focus_entry(env.view, st, CELLS[0], 0.45).entry
    MT._OVERLAYS.clear()
    root = str(env.tmp / "overlay")
    r1 = MT.merge([_shard(env, "a_1", entry, ["reach_fail", "reach_fail"])], root)
    assert r1["mkey0123456789ab"]["version"] == 1 and r1["mkey0123456789ab"]["release"] == 1
    assert MT.load_release(root, 1) == {"mkey0123456789ab": 1}
    assert MT.pinned_env(root) == {"ISPATIALGYM_MAP_OVERLAY": root, "ISPATIALGYM_MAP_OVERLAY_RELEASE": "1"}
    assert MT.release(root) == 1  # nothing new: the same release
    # a second batch confirms the cell -> v2, release 2
    r2 = MT.merge([_shard(env, "b_1", entry, ["ok"])], root)
    assert r2["mkey0123456789ab"]["version"] == 2 and MT.latest_release(root) == 2
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY", root)
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "1")
    assert MT.view_tag("mkey0123456789ab") == "inherit_k2+ov1" and MT.overlay_of("mkey0123456789ab").pin == "r1"
    v1 = MS.as_bands(RM.ReachMap(env.map_path), overhang=0.0)
    assert _xy(v1.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[1, 2]])  # v1: invalidated
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "2")
    assert MT.view_tag("mkey0123456789ab") == "inherit_k2+ov2"
    v2 = MS.as_bands(RM.ReachMap(env.map_path), overhang=0.0)
    assert _xy(v2.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[0, 1, 2]])  # v2: confirmed (a confirmed cell is never dead)
    # a map key the release does not name reads no overlay; release 0 (none yet) reads none
    assert MT.overlay_of("otherkey00000000").base_version == 0 and MT.overlay_of("otherkey00000000").pin == "r2"
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "0")
    assert MT.view_tag("mkey0123456789ab") == "inherit_k2+ov0"
    # a pinned release that cannot be read fails, it never falls back to another version
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "7")
    with pytest.raises(FileNotFoundError):
        MT.overlay_of("mkey0123456789ab")
    # unpinned (no release in the environment): the latest version, flagged
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY_RELEASE")
    assert MT.overlay_of("mkey0123456789ab").pin == "latest" and MT.overlay_of("mkey0123456789ab").base_version == 2
    assert sorted(os.listdir(os.path.join(root, "releases"))) == ["r1.json", "r2.json"]


def test_one_view_never_mixes_overlaid_and_plain_columns(env, monkeypatch):
    """The in-memory column cache keys on the overlay object (its pin) and its version: the same view read under another overlay pin
    recomputes the column instead of returning the other pin's pruned cells."""
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    v = env.view
    entry = MT.focus_entry(v, st, CELLS[0], 0.45).entry
    root = str(env.tmp / "overlay")
    MT.merge([_shard(env, "a_1", entry, ["reach_fail", "reach_fail"])], root)
    MT._OVERLAYS.clear()
    assert _xy(v.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[0, 1, 2]])  # no overlay: the union
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY", root)
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "1")
    assert _xy(v.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[1, 2]])  # the pinned v1 applied, not the cached plain column
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY")
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY_RELEASE")
    assert _xy(v.band(0.25, -0.53, stand_off=0.45)) == _xy(CELLS[[0, 1, 2]])


def test_map_view_records_the_overlay_pin(env, monkeypatch):
    """tabletop.map_view_of: the view's meta names the applied overlay version and its pin; kit counters map_overlay.read_v<N> / .unpinned."""
    st = SimpleNamespace(translate=0.25, tilt=-0.53, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45)
    entry = MT.focus_entry(env.view, st, CELLS[0], 0.45).entry
    root = str(env.tmp / "overlay")
    MT.merge([_shard(env, "a_1", entry, ["reach_fail"])], root)
    MT._OVERLAYS.clear()
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY", root)
    monkeypatch.setenv("ISPATIALGYM_MAP_OVERLAY_RELEASE", "1")
    s = SimpleNamespace(kit=SimpleNamespace(stats={}))
    view = TT.map_view_of(s, RM.ReachMap(env.map_path))
    assert view.meta["map_overlay"] == {"version": 1, "pin": "r1"} and view.meta["map_tilt"] == "inherit_k2+ov1"
    assert s.kit.stats == {"map_overlay.read_v1": 1}
    monkeypatch.delenv("ISPATIALGYM_MAP_OVERLAY_RELEASE")
    TT.map_view_of(s, RM.ReachMap(env.map_path))
    assert s.kit.stats["map_overlay.unpinned"] == 1
