"""i7-floor 2026-11-07: FLOOR-mode v2 reach maps (``reach_maps`` kind ``floor``: the object on the floor plane the robot stands on, h 0,
no edge) and their IORSLarge consumer under v2 only - the floor lattice and support mask, the map identity (key, ``ReachMap.find``), the
band view, ``tabletop.floor_map_bands`` (a hit polices the floor in the policy's floor mode, a miss logs MAP_REQUEST), ``map_bands``'
MAP_REQUEST on a miss, ``iors_large.band``'s v2-only dispatch (each arm's own map, never a legacy band), the exact support-policy
re-check's floor mode and the CLI's floor kind. Simulator-free."""

import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.sim_utils import Reject

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
# tabletop.floor_map_bands / map_bands
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def v2(tmp_path, monkeypatch):
    """A v2 library entry (legacy key apple__x__t, stable rest r0) and a maps root."""
    glib = tmp_path / "glib"
    d = S.model_dir("apple", "x", [1, 1, 1], str(glib))  # the directory v2_entry re-loads the entry from
    os.makedirs(d)
    S.save_json(os.path.join(d, "library.json"), {
        "schema": S.LIBRARY, "category": "apple", "model": "x", "scale": [1, 1, 1], "arm": "right", "legacy_key": "apple__x__t",
        "policy_key": CP.DEFAULT.key("right"), "rests": {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.04, "stable": True}},
        "grasps": [{"name": "g0", "rest_id": "r0", "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]}, "width_m": 0.03}]})
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(glib))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(tmp_path / "maps"))
    for k in ("ISPATIALGYM_REACH_MAP_ANCHORS", "ISPATIALGYM_REACH_MAP_REQUIRE", "ISPATIALGYM_SUPPORT_POLICY", "ISPATIALGYM_V2_ONLY", "ISPATIALGYM_REACH_MAPS_STRICT"):
        monkeypatch.delenv(k, raising=False)
    GL._v2_index.cache_clear()
    lib = GL.v2_entry("apple__x__t")
    assert lib is not None
    yield SimpleNamespace(lib=lib, entry=os.path.basename(d), maps=str(tmp_path / "maps" / os.path.basename(d)))
    GL._v2_index.cache_clear()


def _floor_sampler(floor_z=0.003, world_floor=0.0):
    floor = SimpleNamespace(name="floors_kitchen_0")
    return SimpleNamespace(world=SimpleNamespace(floor_z=world_floor, scene_model="house"), floor_z=floor_z, ikc=SimpleNamespace(),
                           obj=lambda role: floor), floor


def _requests(caplog):
    return [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]


def test_floor_map_bands_hit_polices_the_floor_in_floor_mode(v2, caplog):
    from ispatialgym.sampling.v2 import tabletop as TT

    meta = _meta(object_key=v2.lib.identity_key(), policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest())
    path, _ = _write_map(v2.maps, meta)
    s, floor = _floor_sampler()
    spec = TT.BandSpec(object="apple/x", grasp_key="apple__x__t", scale=[1, 1, 1])
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        view = TT.floor_map_bands(s, spec, "right")
        assert view is not None and view.meta["map_path"] == path and view.meta["stand_offs"] == [0.0]
        assert s.ikc.support_plane_z == {floor.name: 0.003} and s.ikc.support_floor_names == {floor.name}  # the floor the object rests on
        assert os.environ["ISPATIALGYM_SUPPORT_POLICY"] == "grasp" and s.ikc.support_policy_rest["rest_id"] == "r0"
        assert TT.band_facts(view) == {"arm": "right", "path": None, "map_path": path}
        # the left arm has no floor map: a miss with one MAP_REQUEST (kind floor, h 0), never the right arm's map
        assert TT.floor_map_bands(s, spec, "left") is None
        # a floor 3 cm off the robot's floor (beyond FLOOR_DATUM_TOL_M): not served by the h-0 map
        s2, _ = _floor_sampler(floor_z=0.03)
        assert TT.floor_map_bands(s2, spec, "right") is None
    reqs = _requests(caplog)
    assert [r["arm"] for r in reqs] == ["left", "right"] and all(r["kind"] == "floor" and r["plane"] == "floor" and r["h"] == 0.0 for r in reqs)
    assert reqs[0]["entry"] == v2.entry and reqs[0]["rest_id"] == "r0" and reqs[0]["support"] == floor.name and reqs[1]["datum"] == 0.03


def test_floor_map_bands_rejects_another_scale_under_v2_only(v2, monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    s, _ = _floor_sampler()
    spec = TT.BandSpec(object="apple/x", grasp_key="apple__x__t", scale=[0.6, 0.6, 0.6])
    assert TT.floor_map_bands(s, spec, "right") is None
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    with pytest.raises(Reject) as e:
        TT.floor_map_bands(s, spec, "right")
    assert e.value.reason == "v2_scale_mismatch"


def test_map_bands_never_reads_a_floor_map_and_needs_a_maps_root(v2, caplog, monkeypatch):
    # (the MAP_REQUEST of a regular map_bands miss: test_v2_lookup.test_map_bands_miss_logs_the_map_to_build)
    from ispatialgym.sampling.v2 import tabletop as TT

    s = SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model="house"), ikc=SimpleNamespace())
    spec = TT.BandSpec(object="apple/x", grasp_key="apple__x__t")
    _write_map(v2.maps, _meta(object_key=v2.lib.identity_key(), policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest()))
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.map_bands(s, spec, SimpleNamespace(name="low_top", z=0.0), "right") is None  # a top at h 0: the floor map does not serve
        assert not hasattr(s.ikc, "support_plane_z")
        monkeypatch.delenv("ISPATIALGYM_REACH_MAPS")
        assert TT.map_bands(s, spec, SimpleNamespace(name="table_0", z=0.8126), "right") is None  # no maps root: a miss, not a TypeError
        assert TT.floor_map_bands(_floor_sampler()[0], spec, "right") is None
    assert [(r["kind"], r["h"]) for r in _requests(caplog)] == [("tabletop", 0.0), ("tabletop", 0.8126), ("floor", 0.0)]


# ------------------------------------------------------------------------------------------------------------------
# iors_large.band under v2 only
# ------------------------------------------------------------------------------------------------------------------


def _view(arm, path):
    return SimpleNamespace(meta={"mode": "map_view", "arm": arm, "map_path": path, "lattice": 0.08}, path=None, arm=arm,
                           cells_so={0.0: {FLOOR_TRUNK: np.array([[0.4, -0.2]])}}, cells={FLOOR_TRUNK: np.array([[0.4, -0.2]])})


@pytest.fixture
def large(monkeypatch):
    IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
    from ispatialgym.sampling.v2 import tabletop as TT

    floor, partner = SimpleNamespace(name="floors_0"), SimpleNamespace(name="table_7")
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(category="apple", model="x", scale=None)})
    s.world = SimpleNamespace(scene_model="s_large", floor_z=0.0)
    s.grasp_key = "apple__x__t"
    s.gp = SimpleNamespace(variant="ispatialgym-test-large")
    s.template_facts = {}
    s.ikc = SimpleNamespace(default_working_arms=["right"])
    s.floor_z = 0.004
    s.obj = lambda role: {"floor": floor, "partner": partner}[role]
    s.place = {"top": SimpleNamespace(name="table_7", z=0.75)}
    s.surface = {}
    spec = TT.BandSpec(object="apple/x", grasp_key="apple__x__t")
    s.make_band_spec = lambda: spec
    calls = []
    maps = {}  # (kind, arm) -> view (missing: a miss)

    def floor_map_bands(s_, spec_, arm=None, *, floor=None, z_floor=None):
        calls.append(("floor", arm, floor.name, z_floor, spec_ is spec))
        return maps.get(("floor", arm))

    def map_bands(s_, spec_, top, arm=None, **kw):
        calls.append(("tabletop", arm, top.name, top.z, spec_ is spec))
        return maps.get(("tabletop", arm))

    def legacy(*a, **k):
        raise AssertionError("a legacy band was loaded / computed under v2 only")

    monkeypatch.setattr(TT, "floor_map_bands", floor_map_bands)
    monkeypatch.setattr(TT, "map_bands", map_bands)
    monkeypatch.setattr(TT, "surface_band_kwargs", lambda s_: {})
    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(legacy))
    monkeypatch.setattr(RB, "BandComputer", legacy)
    monkeypatch.setattr(TT, "compute_bands", legacy)
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS_STRICT", "1")
    return SimpleNamespace(s=s, calls=calls, maps=maps, floor=floor, partner=partner)


def test_iors_large_band_reads_each_arms_own_map_under_v2_only(large):
    s, calls, maps = large.s, large.calls, large.maps
    maps[("floor", "right")] = _view("right", "floor_right.json")
    maps[("floor", "left")] = _view("left", "floor_left.json")
    maps[("tabletop", "left")] = _view("left", "table_left.json")
    right = s.band("floor")
    assert right.meta["map_path"] == "floor_right.json" and calls == [("floor", "right", "floors_0", 0.004, True)]
    assert s.band("floor", arm="right") is right and len(calls) == 1  # the default arm's view, memoised
    left = s.band("floor", arm="left")
    assert left.meta["map_path"] == "floor_left.json" and "mirrored_from" not in left.meta and calls[-1][:2] == ("floor", "left")
    assert s.band("floor", arm="left") is left and len(calls) == 2
    top = s.band("tabletop", large.partner, arm="left")  # stance C of the ontop place: the partner's measured top
    assert top.meta["map_path"] == "table_left.json" and calls[-1] == ("tabletop", "left", "table_7", 0.75, True)
    facts = s.template_facts["arm_bands_by_factor"]
    assert facts["floor:floor"] == {"right": {"arm": "right", "path": None, "map_path": "floor_right.json"},
                                    "left": {"arm": "left", "path": None, "map_path": "floor_left.json"}}
    assert facts["tabletop:table_7"]["left"]["map_path"] == "table_left.json"


def test_iors_large_band_miss_under_v2_only(large, monkeypatch, caplog):
    s = large.s
    with pytest.raises(Reject) as e:
        s.band("floor", arm="left")  # no left-arm floor map: STRICT rejects, the right arm's map is not mirrored
    assert e.value.reason == "missing_map" and "left-arm" in str(e.value)
    monkeypatch.delenv("ISPATIALGYM_REACH_MAPS_STRICT")
    with pytest.raises(Reject) as e:
        s.band("tabletop", large.partner)
    assert e.value.reason == "legacy_band_blocked"  # without STRICT: the legacy band it would need is blocked
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS_STRICT", "1")
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        with pytest.raises(Reject) as e:
            s.band("floor", None, large.partner, arm="right")  # the floor under a partner: no map kind yet
    assert e.value.reason == "missing_map"
    reqs = _requests(caplog)
    assert len(reqs) == 1 and reqs[0]["kind"] == "floor_under" and reqs[0]["under"] == "table_7" and reqs[0]["arm"] == "right"
    assert ("floor", "right") not in [c[:2] for c in large.calls]  # the floor map is not read for the floor under a partner


def test_iors_large_band_off_v2_only_is_unchanged(large, monkeypatch):
    s = large.s
    monkeypatch.delenv("ISPATIALGYM_V2_ONLY")
    loads = []

    def load(scene, support, obj, gkey, **kw):
        loads.append((support, kw.get("arm", "right")))
        return SimpleNamespace(meta={"arm": "right", "n_reachable_total": 3}, path="legacy.npz", arm="right")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(load))
    rb = s.band("floor")
    assert rb.path == "legacy.npz" and loads == [("floor", "right")] and large.calls == []  # the legacy loader, no map lookup


# ------------------------------------------------------------------------------------------------------------------
# the exact support-policy re-check: the floor mode
# ------------------------------------------------------------------------------------------------------------------


def test_policy_links_drop_the_ground_contact_links_on_a_registered_floor():
    from ispatialgym.sampling.v2 import ik as IK

    floor, table = SimpleNamespace(name="floors_0"), SimpleNamespace(name="table_0")
    others = ["base_link", "wheel_motor_link1", "torso_link1", "right_arm_link5"]
    gc = ["base_link", "wheel_motor_link1"]
    assert IK.policy_links(others, [floor], {"floors_0"}, gc) == ["torso_link1", "right_arm_link5"]
    assert IK.policy_links(others, [table], {"floors_0"}, gc) == others  # a table: every non-finger link, as before
    assert IK.policy_links(others, [floor], set(), gc) == others  # an unregistered floor: as before
    assert IK.floor_support_names(SimpleNamespace()) == set() and IK.floor_support_names(SimpleNamespace(support_floor_names={"f"})) == {"f"}


def test_support_policy_eval_polices_a_registered_floor_in_floor_mode():
    import contextlib

    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling.v2 import ik as IK

    floor = SimpleNamespace(name="floors_0", aabb=([0, 0, -0.1], [5, 5, 0.004]))
    table = SimpleNamespace(name="table_0", aabb=([0, 0, 0], [1, 1, 0.75]))
    seen = {"links": [], "floor": []}
    links = ["base_link", "wheel_motor_link1", "torso_link1", "right_arm_link5", "right_gripper_finger_link1"]

    def links_clear(q, others, margin):
        seen["links"].append(list(others))
        return th.tensor([True])

    def check(T, plane, fq, floor=False):
        seen["floor"].append((plane.name, plane.top_z, floor))
        return th.tensor([True]), ["ok"], {r: th.tensor([0.01]) for r in CP.H.ROLES}

    backend = SimpleNamespace(ground_contact_links={"base_link", "wheel_motor_link1"}, kin_entry=lambda n: None,
                              _collision_costs=lambda e: {"link_of_sphere": links},
                              fk=lambda ls, q: {M.EEF_LINK["right"]: (th.tensor([0.5, 0.0, 0.05]), th.tensor([0.0, 0.0, 0.0, 1.0]))})
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=backend, links_clear=links_clear, robot=SimpleNamespace(gripper_control_idx={"right": [0, 1]}))
    chk.ignoring = lambda objs: contextlib.nullcontext()
    chk._hand_checker = SimpleNamespace(arm="right", check=check)
    chk.support_plane_z = {"floors_0": 0.004}
    chk.support_floor_names = {"floors_0"}
    q = th.zeros(28)
    ev = chk.support_policy_eval(q, "right", [floor], [])
    assert ev["ok"] and ev["floor"] and seen["floor"] == [("floors_0", 0.004, True)]
    assert "base_link" not in seen["links"][-1] and "wheel_motor_link1" not in seen["links"][-1] and "torso_link1" in seen["links"][-1]
    ev = chk.support_policy_eval(q, "right", [table], [])
    assert ev["ok"] and not ev["floor"] and seen["floor"][-1] == ("table_0", 0.75, False) and "base_link" in seen["links"][-1]


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
