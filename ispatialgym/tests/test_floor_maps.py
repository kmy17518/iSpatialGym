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
    chk.support_plane_z = {"floors_0": 0.004, "table_0": 0.75}  # 2026-11-07: only registered supports are policed
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
    assert len(got) == 1  # i7-floorfix: a floor map of the former finger rule (no floor_fingers: "spheres") does not stand in for the new one
    (d / "map__apple__x__h0.0000__ccc.json").write_text(json.dumps({"kind": "floor", "arm": "left", "config": {"floor_fingers": "policy"}}))
    got.clear()
    RM._compute_cli(_cli_args(tmp_path))
    assert got == []
    with pytest.raises(SystemExit):
        RM._compute_cli(_cli_args(tmp_path, kind="tabletop"))  # a slab kind still needs its anchors


# ------------------------------------------------------------------------------------------------------------------
# i7-floorfix 2026-11-07: the floor kind's finger rule (floor_fingers "policy": the IK without the ground slab, the fingers by the
# policy's floor allowance) - the map's config / identity / lookup, compute_map's IK, the sampler's exact IK at a floor grasp
# ------------------------------------------------------------------------------------------------------------------


def test_floor_fingers_enters_the_floor_key_only_and_old_floor_maps_read_spheres():
    k = RM.floor_config().key()
    assert RM.floor_config().floor_fingers == RM.FLOOR_FINGERS == "policy" and k["floor_fingers"] == "policy"
    assert "floor_fingers" not in RM.MapConfig().key() and "floor_fingers" not in RM.MapConfig(preshape="max").key()  # slab keys stay
    assert RM.floor_config(floor_fingers="spheres").key()["floor_fingers"] == "spheres"
    old = {kk: v for kk, v in k.items() if kk != "floor_fingers"}
    assert RM.config_value({"config": old}, "floor_fingers") == "spheres" and RM.config_value({"config": k}, "floor_fingers") == "policy"
    assert RM.map_key(_meta(config=old)) != RM.map_key(_meta())  # a new rule = a new map identity


def test_find_and_floor_map_bands_never_serve_a_floor_map_of_the_former_finger_rule(v2, caplog):
    from ispatialgym.sampling.v2 import tabletop as TT

    old_cfg = {kk: v for kk, v in RM.floor_config().key().items() if kk != "floor_fingers"}
    ident = dict(object_key=v2.lib.identity_key(), policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest())
    old_path, _ = _write_map(v2.maps, _meta(config=old_cfg, **ident))
    lib = SimpleNamespace(identity_key=v2.lib.identity_key)
    ok = dict(rest_id="r0", arm="right", anchors=None, policy_key=ident["policy_key"], robot=ident["robot"])
    assert RM.ReachMap.find(v2.maps, lib, 0.0, kind="floor", **ok).path == old_path  # without the requirement it would serve
    assert RM.ReachMap.find(v2.maps, lib, 0.0, kind="floor", config_require={"floor_fingers": "policy"}, **ok) is None
    s, _ = _floor_sampler()
    spec = TT.BandSpec(object="apple/x", grasp_key="apple__x__t", scale=[1, 1, 1])
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.floor_map_bands(s, spec, "right") is None  # the old floor map does not serve the floor grasp's rule
    assert [r["kind"] for r in _requests(caplog)] == ["floor"]
    new_path, _ = _write_map(v2.maps, _meta(**ident))
    view = TT.floor_map_bands(s, spec, "right")
    assert view is not None and view.meta["map_path"] == new_path


class _FloorWorld:
    """The SynthWorld surface compute_map uses (floor kind), on CPU: a configuration is a pose (flattened 4x4) + one value, the IK seed's
    'elbow' (>= 0.5: the elbow high, else 0.5 m into the floor); the ground slab is a switchable cuboid; the fingers' spheres always 1 cm
    into the floor and a wheel on it (the policy exempts both)."""

    LINKS = ["right_arm_link4", "right_gripper_finger_link1", "wheel_motor_link1"]

    def __init__(self, cfg):
        self.cfg, self.dev = cfg, th.device("cpu")
        self.slab = {"__ground__": True}
        self.eligibility = []
        cub = SimpleNamespace(names=[["__ground__"]], set_enabled=lambda n, on: self.slab.__setitem__(n, bool(on)))

        def check_collision(q, emb_name, margin):
            self.eligibility.append(self.slab["__ground__"])
            return th.tensor([False])

        self.backend = SimpleNamespace(check_collision=check_collision, checker=SimpleNamespace(data=SimpleNamespace(cuboids=cub)), GROUND_SLAB_NAME="__ground__",
                                       robot_spheres_world=lambda q: (self._spheres(th.ones(1)), list(self.LINKS)),
                                       ground_contact_links={"wheel_motor_link1"}, position_tolerance=0.005, orientation_tolerance=0.05)

    def _spheres(self, elbow):
        n = elbow.shape[0]
        out = th.zeros(n, 3, 4)
        out[:, 0] = th.stack([th.zeros(n), th.zeros(n), th.where(elbow >= 0.5, th.full((n,), 1.0), th.full((n,), -0.5)), th.full((n,), 0.05)], 1)
        out[:, 1] = th.tensor([0.5, 0.0, -0.01, 0.005])
        out[:, 2] = th.tensor([0.0, 0.2, 0.0, 0.05])
        return out

    def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
        return th.tensor([translate, tilt, fingers])

    def shoulder(self, translate, tilt):
        return np.array([0.0, 0.0, 0.6]), 1.5

    def entry(self):
        return None

    def link_of_sphere(self):
        return list(self.LINKS)

    def full_q(self, entry, q_active, q_lock):
        return q_active

    def spheres(self, q_full):
        return self._spheres(q_full[:, 16])

    def eef(self, q_full):
        return q_full[:, :16].reshape(-1, 4, 4)


def _floor_solve(calls):
    """Every pre-grasp seed solves (seed 0 with the elbow through the floor, the others high); the grasp has no IK while the slab is in
    (the lower finger's padded spheres in it) and solves without it."""

    def solve(world, entry, T, q_lock, seed_active=None):
        n, S_ = T.shape[0], world.cfg.seeds
        on = world.slab["__ground__"]
        calls.append(("grasp" if seed_active is not None else "pre", on))
        q = th.cat([T.reshape(n, 1, 16).repeat(1, S_, 1), th.arange(S_, dtype=th.float32).view(1, S_, 1).repeat(n, 1, 1)], dim=2)
        if seed_active is None:
            return th.ones(n, S_, dtype=th.bool), q
        q[..., 16] = 1.0
        return th.full((n, S_), not on, dtype=th.bool), q

    return solve


def _floor_lib(tmp_path):
    d = S.model_dir("apple", "x", [1, 1, 1], str(tmp_path / "glib"))
    os.makedirs(d)
    S.save_json(os.path.join(d, "library.json"), {
        "schema": S.LIBRARY, "category": "apple", "model": "x", "scale": [1, 1, 1], "arm": "right", "legacy_key": "apple__x__t",
        "policy_key": CP.DEFAULT.key("right"), "rests": {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.04, "stable": True}},
        "grasps": [{"name": "g0", "rest_id": "r0", "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]}, "width_m": 0.03}]})
    return S.load_model("apple", "x", [1, 1, 1], str(tmp_path / "glib"))


@pytest.mark.parametrize("rule", ["policy", "spheres"])
def test_compute_map_floor_policy_runs_the_ik_without_the_ground_slab(tmp_path, monkeypatch, rule):
    calls = []
    monkeypatch.setattr(RM, "_solve", _floor_solve(calls))
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [FLOOR_TRUNK])
    cfg = RM.floor_config(seeds=2, twins=False, lattice_m=0.4, floor_fingers=rule)
    w = _FloorWorld(cfg)
    meta, arrays = RM.compute_map(w, _floor_lib(tmp_path), "r0", 0.0, RM.FLOOR_ANCHOR, cfg=cfg, kind="floor")
    st = arrays[f"s0.00_{RB.trunk_key(*FLOOR_TRUNK)}"]
    assert w.slab["__ground__"] and w.eligibility and all(w.eligibility)  # restored; the trunk eligibility kept the slab
    assert meta["config"]["floor_fingers"] == rule and meta["floor"]["fingers"] == rule and meta["key"] == RM.map_key(meta)
    if rule == "policy":
        # both IKs without the slab; the pre-grasp is seed 1 (seed 0's elbow is through the floor), the fingers / wheel exempt: feasible
        assert {c for c, _ in calls} == {"pre", "grasp"} and not any(on for _, on in calls)
        assert int((st == RM.STATUS["feasible"]).sum()) > 0 and not (st == RM.STATUS["ik_grasp"]).any()
        assert not (st == RM.STATUS["slab_spheres"]).any() and "self-collision only" in meta["ik"]["collision_world"]
    else:
        # the former rule: the slab in both IKs - the grasp has no IK anywhere
        assert all(on for _, on in calls) and not (st == RM.STATUS["feasible"]).any() and (st == RM.STATUS["ik_grasp"]).any()


def test_compute_map_policy_pre_grasp_seed_falls_back_and_floor_fingers_is_floor_only(tmp_path, monkeypatch):
    calls = []
    solve = _floor_solve(calls)

    def low_elbow_pre(world, entry, T, q_lock, seed_active=None):
        ok, q = solve(world, entry, T, q_lock, seed_active)
        if seed_active is None:
            q[..., 16] = 0.0  # every pre-grasp seed with the elbow through the floor
        return ok, q

    monkeypatch.setattr(RM, "_solve", low_elbow_pre)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [FLOOR_TRUNK])
    lib = _floor_lib(tmp_path)
    cfg = RM.floor_config(seeds=2, twins=False, lattice_m=0.4)
    meta, arrays = RM.compute_map(_FloorWorld(cfg), lib, "r0", 0.0, RM.FLOOR_ANCHOR, cfg=cfg, kind="floor")
    st = arrays[f"s0.00_{RB.trunk_key(*FLOOR_TRUNK)}"]
    assert not (st == RM.STATUS["feasible"]).any() and (st == RM.STATUS["slab_spheres"]).any()  # the non-finger links still meet the floor
    with pytest.raises(ValueError):
        RM.compute_map(_FloorWorld(cfg), lib, "r0", 0.0, RM.FLOOR_ANCHOR, cfg=RM.floor_config(floor_fingers="none"), kind="floor")
    with pytest.raises(ValueError):
        RM.compute_map(_FloorWorld(cfg), lib, "r0", 0.8, {"key": "0.8"}, cfg=RM.MapConfig(floor_fingers="policy"), kind="tabletop")


class _FloorKit:
    """eef position = right arm joints 0..2 (identity orientation), the fingers' opening = the gripper joints. Obstacles are horizontal
    half-spaces (their top); at a horizontal approach 4 cm above the floor the fully open lower finger's spheres are 1.6 cm into it, the
    palm 1.5 cm and the arm 13 cm above it."""

    RIGHT, GRIP = th.arange(7, 14), th.tensor([21, 22])
    BOTTOMS = {"right_gripper_finger_link1": -0.0564, "right_gripper_finger_link2": 0.03, "right_gripper_link": -0.0249, "right_arm_link6": 0.09}

    def __init__(self, obstacles, attached=None):
        self.robot = SimpleNamespace(arm_control_idx={"right": self.RIGHT, "left": th.arange(14, 21)}, gripper_control_idx={"right": self.GRIP},
                                     joint_lower_limits=th.full((28,), -3.0), joint_upper_limits=th.full((28,), 3.0))
        self.obstacles, self.disabled, self.stats = list(obstacles), set(), {}
        kit = self

        class _Backend:
            _attached = attached or {}

            def solve_ik_path(self, name, targets, q_start, **kw):
                traj = [q_start.clone()]
                for t in targets:
                    q = traj[-1].clone()
                    q[kit.RIGHT[:3]] = list(t.values())[0][0]
                    traj.append(q)
                return SimpleNamespace(success=True, q_traj=th.stack(traj), status="ok", diagnostics={})

            def kin_entry(self, name):
                return name

            def _collision_costs(self, entry):
                return {"link_of_sphere": list(kit.BOTTOMS) + [""]}

        self.backend = _Backend()

    def _ensure_synced(self):
        pass

    def links_clear(self, qs, links, margin):
        qs = qs.unsqueeze(0) if qs.dim() == 1 else qs
        return th.tensor([all(float(q[self.RIGHT[2]]) + self.BOTTOMS[l] - o.top >= float(margin) - 1e-12 for l in links for o in self.obstacles
                              if o not in self.disabled) for q in qs], dtype=th.bool)

    def collision_free(self, qs, margin=0.0):
        return self.links_clear(qs, list(self.BOTTOMS), margin)

    def fk(self, links, q):
        return {links[0]: (q[self.RIGHT[:3]].clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    def ik(self, pose, q_seed, arm="right"):
        q = q_seed.clone()
        q[self.robot.arm_control_idx[arm][:3]] = pose[0]
        return q


def _floor_checker(kit, registered=True):
    from ispatialgym.sampling.v2 import ik as IK

    ikc = IK.IKChecker(kit)
    ikc.ignoring = lambda objs, ground=True, ground_meshes=True: _disabling(kit, objs)
    if registered:
        ikc.support_plane_z = {"floors_0": 0.0}
        ikc.support_floor_names = {"floors_0"}
    evals = []

    def eval_(q, arm, sup, appr, finger_links=False):
        evals.append((round(float(q[kit.RIGHT[0]]), 3), [o.name for o in sup], [o.name for o in appr]))
        return {"ok": True, "reason": "", "stage": "hulls"}

    ikc.support_policy_eval = eval_
    return ikc, evals


@__import__("contextlib").contextmanager
def _disabling(kit, objs):
    added = [o for o in objs if o not in kit.disabled]
    kit.disabled.update(added)
    try:
        yield
    finally:
        kit.disabled.difference_update(added)


class _Half:
    """A horizontal half-space obstacle (hashable by identity, as the kit's ignore set needs)."""

    def __init__(self, name, top):
        self.name, self.top = name, float(top)


def _hpose(s):
    """A horizontal approach along +x, the eef 4 cm above the floor, ``s`` m before the grasp."""
    return th.tensor([0.5 - s, 0.0, 0.04]), th.tensor([0.0, 0.0, 0.0, 1.0])


def test_exact_floor_grasp_chain_meets_the_floor_only_through_the_policy(monkeypatch):
    floor, target = _Half("floors_0", 0.0), _Half("mouse_0", -10.0)
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "grasp")
    q0 = th.zeros(28)
    q0[_FloorKit.GRIP] = 0.05
    chain = [_hpose(0.10), _hpose(0.0)]
    # the floor not registered (a table chain's rule): the pre-grasp and the approach against the whole world - the fully open lower
    # finger's padded spheres are in the floor along the horizontal approach
    kit = _FloorKit([floor, target])
    ikc, evals = _floor_checker(kit, registered=False)
    res = ikc.exact(chain, None, "right", ignore_for_last=[floor], q_seed=q0, cache=False, approached=[target])
    assert not res.ok and "approach_untrackable" in res.reason and not evals
    # a registered floor (tabletop.floor_map_bands): out of the world for the whole chain, every configuration policed against it -
    # the pre-grasp once, the approach's configurations after its start (10), the grasp once; nothing policed twice on the tail
    ikc, evals = _floor_checker(kit)
    res = ikc.exact(chain, None, "right", ignore_for_last=[floor], q_seed=q0, cache=False, approached=[target])
    assert res.ok, res.reason
    assert all(sup == ["floors_0"] for _, sup, _ in evals) and len(evals) == 12
    assert evals[0][0] == 0.4 and evals[-1][0] == 0.5 and sorted({x for x, _, _ in evals[1:-1]}) == [round(0.41 + 0.01 * k, 3) for k in range(10)]
    assert all(appr == ["mouse_0"] for _, _, appr in evals[1:11])  # the approach: the approached target left out, as its collision check did
    assert not kit.disabled  # every ignore context exited

    # a pre-grasp the policy rejects fails the chain at pose 1
    def reject_pre(q, arm, sup, appr, finger_links=False):
        return {"ok": float(q[kit.RIGHT[0]]) > 0.45, "reason": "finger_penetration on floors_0", "stage": "hulls"}

    ikc.support_policy_eval = reject_pre
    res = ikc.exact(chain, None, "right", ignore_for_last=[floor], q_seed=q0, cache=False, approached=[target])
    assert not res.ok and res.reason.startswith("support_policy at pose 1 of 2") and kit.stats["support_policy_floor_chain_rejected"] == 1
    # an arm that holds something (a place onto the floor, scope "grasp"): the chain as before
    kit2 = _FloorKit([floor, target], attached={"right": {"obj": "cup"}})
    ikc2, evals2 = _floor_checker(kit2)
    assert not ikc2.exact(chain, None, "right", ignore_for_last=[floor], q_seed=q0, cache=False, approached=[target]).ok and not evals2
    # the policy off: as before
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY")
    ikc3, evals3 = _floor_checker(_FloorKit([floor, target]))
    assert not ikc3.exact(chain, None, "right", ignore_for_last=[floor], q_seed=q0, cache=False, approached=[target]).ok and not evals3


def test_policy_tail_polices_a_floor_chain_segment_whole_and_independent_of_the_tail_switch(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IK

    floor, target = SimpleNamespace(name="floors_0"), SimpleNamespace(name="mouse_0")
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=SimpleNamespace(), stats={})
    chk.support_plane_z, chk.support_floor_names = {"floors_0": 0.0}, {"floors_0"}
    chk.approached_objects = lambda pose: []
    evals = []
    chk.support_policy_eval = lambda q, arm, sup, appr: evals.append((int(q), [o.name for o in sup], [o.name for o in appr])) or {"ok": True}
    traj = list(range(11))
    args = (traj, 10, 0.1, "right", None, None, [8, 9], [1, 2], [])
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "grasp")
    assert chk._policy_tail(*args, ignore_end=[floor], ignore_start=(), ignore_mid=[target, floor]) == ""
    assert [e[0] for e in evals] == list(range(1, 11)) and all(e[1] == ["floors_0"] and e[2] == ["mouse_0"] for e in evals)
    evals.clear()
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY_TAIL", "0")
    assert chk._policy_tail(*args, ignore_end=[floor], ignore_start=(), ignore_mid=[target, floor]) == "" and len(evals) == 10
    evals.clear()
    chk.support_policy_eval = lambda q, arm, sup, appr: {"ok": int(q) != 5, "reason": "palm_contact on floors_0"}
    why = chk._policy_tail(*args, ignore_end=[floor], ignore_start=(), ignore_mid=[target, floor])
    assert "along the floor grasp's approach (palm_contact on floors_0, 0.05 m" in why and chk.kit.stats["support_policy_floor_chain_rejected"] == 1
    # an unregistered support in ignore_mid (the approached target): no floor-chain job
    assert chk._policy_tail(*args, ignore_end=(), ignore_start=(), ignore_mid=[target]) == ""
