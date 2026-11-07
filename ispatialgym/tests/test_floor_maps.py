"""i7-floor 2026-11-07: FLOOR-mode v2 reach maps (``reach_maps`` kind ``floor``: the object on the floor plane the robot stands on, h 0,
no edge) - the floor lattice and support mask, the map identity (key, ``ReachMap.find``), the band view and the CLI's floor kind.
Simulator-free."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.grasping import schema as S
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

FLOOR_TRUNK = (0.95, -1.2)


# ------------------------------------------------------------------------------------------------------------------
# the floor kind: lattice, support, config, trunks
# ------------------------------------------------------------------------------------------------------------------


def test_floor_lattice_starts_at_the_floor_raster_and_has_no_edge():
    cfg = RM.floor_config(arm="left")
    cells = RM.lattice(cfg)
    assert cfg.edge_distances == (0.0,) and cfg.arm == "left" and cfg.preshape == "grasp"
    assert abs(cells[:, 0].min() - 0.20) < 1e-9 and cells[:, 0].max() <= cfg.x_max_m + 1e-9  # the 0.20-0.32 m floor cells are in
    assert abs(cells[:, 1].min() + 0.80) < 1e-9 and abs(cells[:, 1].max() - 0.80) < 1e-9
    slab_cells = RM.lattice(RM.MapConfig())
    assert abs(slab_cells[:, 0].min() - 0.35) < 1e-9 and abs(slab_cells[:, 1].max() - 0.88) < 1e-9  # a slab map's lattice as before
    # the support: everywhere on the floor (a footprint reaching 0.3 m back toward the robot still stands on it); a slab needs it behind d_e
    reach_back = np.array([-0.30, 0.0])
    assert RM.support_mask(cells, reach_back, 0.0, "floor").all()
    m = RM.support_mask(np.array([[0.40, 0.0], [0.70, 0.0]]), reach_back, 0.45, "tabletop")
    assert m.tolist() == [[False, False], [False, True]] and m.shape == (2, 2)
    fl = RM.map_support("floor", 0.0, 0.0, cfg)
    assert fl.kind == "plane" and fl.top_z == 0.0 and fl.name == "floor"
    assert RM.map_support("tabletop", 0.8, 0.45, RM.MapConfig()).kind == "box"


def test_floor_config_enters_the_key_and_slab_keys_stay():
    assert "x_min_m" not in RM.MapConfig().key()  # a slab map's config key as before the field existed
    k = RM.floor_config().key()
    assert k["x_min_m"] == 0.20 and k["y_half_m"] == 0.80 and tuple(k["edge_distances"]) == (0.0,) and k["preshape"] == "grasp"
    assert RM.config_value({"config": {}}, "x_min_m") is None and RM.config_value({"config": k}, "x_min_m") == 0.20
    # the trunks a floor map is computed at: trunk_grid(0) = TRANSLATES x (TILTS + the low-top tilt) + the ground trunks
    grid = RB.trunk_grid(RM.FLOOR_HEIGHT)
    assert len(grid) == 27 and FLOOR_TRUNK in grid and all(tuple(t) in grid for t in RB.GROUND_TRUNKS)
    assert "floor" in RM.KINDS and RM.PLANE_OF_KIND["floor"] == "floor" and RM.FLOOR_ANCHOR["height"] == 0.0


def _old_map_key(meta):
    """``map_key`` as it was before the floor kind (the slab maps' keys must not move)."""
    import hashlib

    ident = {k: meta[k] for k in ("schema", "kind", "arm", "object_key", "rest_id", "edge_distances", "config", "policy_key", "robot")}
    ident["brace"] = meta.get("brace")
    ident["height"] = meta["height"]["value"]
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def _meta(*, kind="floor", plane="floor", arm="right", object_key="obj-key", rest_id="r0", height=0.0, policy_key="pk", robot="rbt", config=None):
    cfg = config if config is not None else (RM.floor_config(arm=arm) if kind == "floor" else RM.MapConfig(arm=arm)).key()
    meta = {"schema": RM.SCHEMA, "kind": kind, "arm": arm, "object_key": object_key, "rest_id": rest_id, "height": {"value": height, "anchor_key": "floor"},
            "edge_distances": list(cfg["edge_distances"]), "config": cfg, "policy_key": policy_key, "robot": robot, "brace": None,
            "object": {"category": "apple", "model": "x"},
            "grasps": [{"name": "g0", "base": "g0", "twin": False}, {"name": "g0~r180", "base": "g0", "twin": True}],
            "trunks": [{"translate": FLOOR_TRUNK[0], "tilt": FLOOR_TRUNK[1], "eligible": True, "posture": "untucked", "key": RB.trunk_key(*FLOOR_TRUNK)},
                       {"translate": 0.5, "tilt": 0.0, "eligible": False, "key": RB.trunk_key(0.5, 0.0)}]}
    if plane is not None:
        meta["plane"] = plane
    meta["key"] = RM.map_key(meta)
    return meta


def _write_map(d, meta, feasible_cells=(0, 3)):
    os.makedirs(d, exist_ok=True)
    cfg = RM.floor_config() if meta["kind"] == "floor" else RM.MapConfig()
    cells = RM.lattice(cfg).astype(np.float32)
    st = np.full((len(cells), 8, len(meta["grasps"])), RM.STATUS["ik_pre"], dtype=np.uint8)
    for c in feasible_cells:
        st[c, 0, 0] = RM.STATUS["feasible"]
    arrays = {"cells": cells, "yaws": np.array([2 * np.pi * k / 8 for k in range(8)], dtype=np.float32)}
    for d_e in meta["edge_distances"]:
        arrays[f"s{d_e:.2f}_{RB.trunk_key(*FLOOR_TRUNK)}"] = st
    base = os.path.join(d, f"map__apple__x__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return base + ".json", cells


def test_floor_map_key_carries_kind_and_plane():
    fl = _meta()
    tt = _meta(kind="tabletop", plane=None, height=0.8126)
    assert RM.map_key(tt) == _old_map_key(tt)  # a slab map's key did not move
    no_plane = {k: v for k, v in fl.items() if k != "plane"}
    assert RM.map_key(fl) != RM.map_key(no_plane) and RM.map_key(fl) != RM.map_key(dict(fl, kind="tabletop"))
    assert RM.map_plane(fl) == "floor" and RM.map_plane(tt) == "slab" and RM.map_plane({"kind": "floor"}) == "floor"


def test_find_a_floor_map_enforces_the_identity(tmp_path):
    root = str(tmp_path)
    path, _ = _write_map(root, _meta())
    _write_map(root, _meta(kind="tabletop", plane=None, config=RM.MapConfig().key()))  # a slab map of the same entry, also at h 0
    lib = SimpleNamespace(identity_key=lambda: "obj-key")
    ok = dict(rest_id="r0", arm="right", anchors=None, policy_key="pk", robot="rbt", config_require={"preshape": "grasp"})
    rm = RM.ReachMap.find(root, lib, 0.0, kind="floor", **ok)
    assert rm is not None and rm.path == path and rm.meta["plane"] == "floor"
    assert RM.ReachMap.find(root, lib, 0.0, **{**ok, "config_require": None}).meta["kind"] == "tabletop"  # a tabletop lookup: never the floor map
    for bad in ({"arm": "left"}, {"rest_id": "r1"}, {"policy_key": "other"}, {"robot": "other"}, {"config_require": {"preshape": "max"}},
                {"config_require": {"x_min_m": 0.35}}):
        assert RM.ReachMap.find(root, lib, 0.0, kind="floor", **{**ok, **bad}) is None, bad
    assert RM.ReachMap.find(root, SimpleNamespace(identity_key=lambda: "another"), 0.0, kind="floor", **ok) is None
    assert RM.ReachMap.find(root, lib, 0.01, kind="floor", **ok) is None  # the floor map is at h 0 exactly
    # a "floor" map whose plane is a slab (or a tabletop map relabelled) does not serve
    d2 = str(tmp_path / "relabelled")
    _write_map(d2, _meta(plane="slab"))
    assert RM.ReachMap.find(d2, lib, 0.0, kind="floor", **ok) is None


def test_band_view_of_a_floor_map_has_one_stand_off(tmp_path):
    path, cells = _write_map(str(tmp_path), _meta(), feasible_cells=(0, 3))
    view = MS.as_bands(RM.ReachMap(path), overhang=0.0)
    assert view.meta["mode"] == "map_view" and view.meta["stand_offs"] == [0.0] and view.meta["stand_off"] == 0.0
    assert view.trunks() == [FLOOR_TRUNK]  # the eligible trunks only
    got = view.band(*FLOOR_TRUNK, stand_off=0.45)  # a floor stance B at 0.45 m from c_t reads the one stored distance
    assert np.allclose(got, cells[[0, 3]]) and float(got[:, 0].min()) < 0.25
    assert view.feasible(*FLOOR_TRUNK, cells[3], stand_off=0.35) == [(0.0, "g0")]


# ------------------------------------------------------------------------------------------------------------------
# the CLI's floor kind (job_glib.sh GB_KIND=floor)
# ------------------------------------------------------------------------------------------------------------------


def _cli_args(tmp_path, **kw):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"id": "apple__x", "category": "apple", "model": "x", "scale": None, "constraint": "tabletop"}]))
    a = dict(jobs=str(jobs), glib=str(tmp_path / "glib"), anchors=None, heights=None, heights_by_id=None, only=None, shard=None,
             out=str(tmp_path / "maps"), mode="full", lattice=0.08, seeds=8, batch=2048, kind="floor", skip_existing=True, include_derived=False,
             arm="left")
    a.update(kw)
    return SimpleNamespace(**a)


def test_compute_cli_floor_kind_builds_one_h0_map_per_stable_rest(tmp_path, monkeypatch, capsys):
    from ispatialgym.grasping import object_geometry as OG

    got = []
    lib = SimpleNamespace(path=str(tmp_path / "glib" / "models" / "apple__x" / "library.json"), grasps=lambda rid=None: ["g"],
                          rests={"r0": SimpleNamespace(stable=True), "r1": SimpleNamespace(stable=False)})
    monkeypatch.setattr(RM, "SynthWorld", lambda cfg: SimpleNamespace(cfg=cfg))
    monkeypatch.setattr(S, "load_model", lambda *a, **k: lib)
    monkeypatch.setattr(OG, "load", lambda d: None)

    def compute_map(w, lib_, rid, height, anchor, geom=None, cfg=None, kind="tabletop"):
        got.append((rid, height, anchor["key"], kind, cfg))
        return {"trunks": [], "timing": {}, "grasps": []}, {}

    monkeypatch.setattr(RM, "compute_map", compute_map)
    monkeypatch.setattr(RM, "save", lambda out, meta, arrays: os.path.join(out, "map.json"))
    RM._compute_cli(_cli_args(tmp_path, heights=["0.8126"]))
    assert [g[:4] for g in got] == [("r0", 0.0, "floor", "floor")]
    cfg = got[0][4]
    assert cfg.edge_distances == (0.0,) and cfg.x_min_m == 0.20 and cfg.y_half_m == 0.80 and cfg.arm == "left" and cfg.twins
    out = capsys.readouterr().out
    assert "GB_MAP_NOTE" in out and "GB_MAP_DONE" in out
    # --skip-existing: this arm's floor map exists -> nothing computed; a tabletop map at h 0 of the same entry does not count
    d = tmp_path / "maps" / "apple__x"
    os.makedirs(d)
    (d / "map__apple__x__h0.0000__aaa.json").write_text(json.dumps({"kind": "tabletop", "arm": "left"}))
    got.clear()
    RM._compute_cli(_cli_args(tmp_path))
    assert len(got) == 1
    (d / "map__apple__x__h0.0000__bbb.json").write_text(json.dumps({"kind": "floor", "arm": "left"}))
    got.clear()
    RM._compute_cli(_cli_args(tmp_path))
    assert got == []
    with pytest.raises(SystemExit):
        RM._compute_cli(_cli_args(tmp_path, kind="tabletop"))  # a slab kind still needs its anchors
