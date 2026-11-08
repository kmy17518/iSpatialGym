"""i7-lean 2026-11-07 (owner scope addition: unblock inter_object_relation_satisfaction-inside_large_scale_02, the jar exmsmm into
fridge_petcxr_0's compartment, under v2 only): the interior spec's fridge pieces - the open door as an oriented obstacle box at its joint
value, the joints and stand-offs in the identity, ``spec_for_support`` / ``live_mismatch`` matching the joints -, the interior map at a
STABLE rest (the compartment fixed in the map, the footprint between its walls, the door blocking what it blocks) in the fake synthetic
world, ``ReachMap.find`` checking the fridge spec, and ``InsideMode.inside_band`` reading the interior map under v2 only (a miss rejects
``missing_map``; off v2 only the legacy band path as before). The fridge numbers here are ILLUSTRATIVE (the measured spec comes from
``python -m ispatialgym.sampling.interiors measure``). Simulator-free."""

import json
import logging
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import interior_place as IP
from ispatialgym.sampling import interiors as IN
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.tests.test_interior_maps import TRUNK, FakeWorld, _fake_solve

SCENE, FRIDGE = "house_double_floor_lower", "fridge_petcxr_0"
JAR_KEY = "jar_of_spaghetti_sauce__exmsmm"
DOOR = {"j_left_door": 2.3}


def _fridge(**kw):
    d = {"height_m": 0.7064, "depth_m": 0.55, "width_m": 0.42, "headroom_m": 0.2434, "board_thickness_m": 0.02, "top_thickness_m": 0.02,
         "wall_thickness_m": 0.03, "long_m": 0.48, "board_setback_m": 0.03, "back_thickness_m": 0.02, "centre_y_m": 0.0,
         "edge_distances": [0.35, 0.45, 0.55, 0.65], "joints": dict(DOOR),
         # the left door swung out 2.3 rad about its hinge at the compartment's front-left corner: in front of the front plane, to the left
         "extra_boxes": [{"name": "door_left_door", "centre": [-0.10, 0.20, 0.30], "half": [0.22, 0.03, 0.95], "yaw": round(2.3 - math.pi / 2, 4)}],
         "supports": [[SCENE, FRIDGE]], "source": {"for": "test (illustrative values)"}}
    d.update(kw)
    return IN.InteriorSpec.from_json(d, name=kw.pop("name", "fridge_petcxr_0@0.7064/j_left_door2.3"))


# ------------------------------------------------------------------------------------------------------------------ the spec


def test_the_open_door_is_an_oriented_obstacle_of_the_spec():
    sp = _fridge()
    names = [n for n, _ in sp.boxes(sp.height_m, 0.45)]
    assert names == ["board", "above", "wall_left", "wall_right", "back", "door_left_door"] and sp.open_sides == ["front"]
    door = dict(sp.boxes(sp.height_m, 0.45))["door_left_door"]
    assert door.kind == "box" and abs(door.yaw - (2.3 - math.pi / 2)) < 1e-4
    assert abs(door.centre[0] - (0.45 - 0.10)) < 1e-9 and abs(door.centre[2] - (0.7064 + 0.30)) < 1e-9  # x from the front plane, z from the board
    # the yaw turns the panel: a point along its long (local x) axis from the centre is inside, the same offset along the world x is not
    a = np.array([math.cos(door.yaw), math.sin(door.yaw), 0.0])
    p_in = th.tensor([list(np.array(door.centre) + 0.2 * a)], dtype=th.float32)
    p_out = th.tensor([list(np.array(door.centre) + 0.2 * np.array([0.0, 1.0, 0.0]) * 0 + 0.2 * np.array([-a[1], a[0], 0.0]))], dtype=th.float32)
    assert float(door.sdf(p_in)[0]) < 0 and float(door.sdf(p_out)[0]) > 0
    assert sp.obstacles(sp.height_m, 0.45)[-1].name == "interior_door_left_door"
    base = sp.digest()
    assert _fridge(joints={"j_left_door": 2.0}).digest() != base
    assert _fridge(edge_distances=[0.35, 0.45, 0.55]).digest() != base
    assert _fridge(extra_boxes=[]).digest() != base and _fridge(centre_y_m=0.05).digest() != base
    js = sp.to_json()
    assert js["joints"] == DOOR and js["edge_distances"] == [0.35, 0.45, 0.55, 0.65] and js["extra_boxes"][0]["name"] == "door_left_door"
    assert IN.InteriorSpec.from_json(js).digest() == base  # the registry round trip


def test_the_spec_is_found_only_at_its_joint_values(tmp_path):
    sp = _fridge()
    reg = {sp.name: sp}
    assert IN.spec_for_support(SCENE, FRIDGE, 0.7064, registry=reg, joints={"j_left_door": 2.3}) is sp
    assert IN.spec_for_support(SCENE, FRIDGE, 0.7064, registry=reg, joints={"j_left_door": 2.31}) is sp  # within JOINT_TOL
    assert IN.spec_for_support(SCENE, FRIDGE, 0.7064, registry=reg, joints={"j_left_door": 2.0}) is None
    assert IN.spec_for_support(SCENE, FRIDGE, 0.7064, registry=reg) is None  # the door in its scene state: not this spec
    assert IN.spec_for_support(SCENE, FRIDGE, 0.7064, registry=reg, joints={"j_left_door": 2.3, "j_right_door": 1.0}) is None
    top = SimpleNamespace(z=0.7064, headroom=th.tensor([[0.2384, 0.2384]]), mask=th.tensor([[True, True]]), open_edges=["+u"])
    assert IN.live_mismatch(sp, top, joints=DOOR) is None
    assert "joints" in IN.live_mismatch(sp, top, joints={"j_left_door": 1.0})
    assert "open sides" in IN.live_mismatch(sp, SimpleNamespace(**{**vars(top), "open_edges": ["+u", "-u"]}), joints=DOOR)
    # the registry file the measure CLI writes reads back
    path = str(tmp_path / "interiors.json")
    IN.write_registry(path, [sp])
    back = IN.load_registry(path)[sp.name]
    assert back.digest() == sp.digest() and back.supports == ((SCENE, FRIDGE),)
    assert IN.spec_named(path).digest() == sp.digest() and IN.spec_named(f"{path}::{sp.name}").digest() == sp.digest()
    os.environ["ISPATIALGYM_INTERIORS"] = path
    try:  # a campaign registry adds to the repo's
        reg2 = IN.load_registry()
        assert sp.name in reg2 and "bookcase_kmuacd_0@0.8354" in reg2 and IN.spec_named(sp.name).digest() == sp.digest()
    finally:
        del os.environ["ISPATIALGYM_INTERIORS"]


def test_the_measure_helpers_box_a_turned_door():
    yaw = 0.73
    c, s_ = math.cos(yaw), math.sin(yaw)
    local = np.array([[x, y, z] for x in (-0.22, 0.22) for y in (-0.03, 0.03) for z in (-0.9, 0.9)])
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    pts = local @ R.T + np.array([0.1, -0.2, 0.3])
    ctr, half = IN.oriented_box(pts, yaw)
    assert np.allclose(ctr, [0.1, -0.2, 0.3], atol=1e-9) and np.allclose(half, [0.22, 0.03, 0.9], atol=1e-9)
    # a link whose x axis points along the tangent (the robot's left) is turned +90 deg from the inward axis
    _, inward, tangent = IN.compartment_frame([0.0, 0.0], [0.0, -1.0])  # inward -y (the fridge faces +y): tangent = +x
    q = [0.0, 0.0, 0.0, 1.0]  # link x = world +x = the tangent
    assert abs(IN.link_yaw_in(q, tangent.numpy()) - math.pi / 2) < 1e-6
    assert [round(float(v), 6) for v in tangent] == [1.0, 0.0]


def test_interior_config_of_a_stable_rest_and_keys_that_stay():
    sp = _fridge()
    cfg = RM.interior_config(sp, rest_kind="stable")
    assert cfg.n_yaws == 8 and cfg.edge_distances == (0.35, 0.45, 0.55, 0.65) and cfg.interior == sp.digest()
    assert cfg.x_max_m == round(0.65 + 0.55, 4) and RM.interior_config(sp).n_yaws == 1  # a lean rest: one yaw
    assert RM.MapConfig().key() == {k: v for k, v in RM.MapConfig().key().items()} and "interior" not in RM.MapConfig().key()
    assert RM.rest_kind_ok("interior", "stable") and RM.rest_kind_ok("interior", "lean") and not RM.rest_kind_ok("tabletop", "lean")
    m = RM.support_mask(np.array([[0.70, 0.0], [0.70, 0.16], [0.70, -0.25]]), np.array([-0.04]), 0.45, "interior", reach_front=np.array([0.04]),
                        interior=sp, lateral=(np.array([-0.04]), np.array([0.04]), 0.0))
    assert m[:, 0].tolist() == [True, True, False]  # between the walls (+-0.21) - the last footprint crosses the right wall


# ------------------------------------------------------------------------------------------------------------------ the map (fake world)


def _jar_lib(path="jar/library.json"):
    data = {"schema": S.LIBRARY, "category": "jar_of_spaghetti_sauce", "model": "exmsmm", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets",
            "arm": "right", "policy_key": CP.DEFAULT.key("right"), "legacy_key": JAR_KEY, "families": {},
            "rests": {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.075, "stable": True}},
            # a side pinch from the robot's side: approach +x at mid height, closing y
            "grasps": [{"name": "side_0", "rest_id": "r0", "family_id": None, "T_object_eef": {"position": [-0.052, 0.0, 0.0], "orientation_xyzw": [0.0, 0.7071068, 0.0, 0.7071068]},
                        "width_m": 0.08, "preshape_q": 0.05, "finger_q": [0.04, 0.04]}]}
    return S.ModelLibrary(path=path, data=data, rests={"r0": S.Rest.from_record(data["rests"]["r0"])})


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [TRUNK])
    import trimesh

    jar = trimesh.creation.box(extents=[0.08, 0.08, 0.15])
    return SimpleNamespace(geom=SimpleNamespace(union=jar, pieces=[jar]))


def _compute(fake, sp, lib=None):
    cfg = RM.interior_config(sp, rest_kind="stable", seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked",), lattice_m=0.12, y_half_m=0.36,
                             n_yaws=2)
    anchor = {"key": f"interior:{sp.name}", "height": sp.height_m}
    meta, arrays = RM.compute_map(FakeWorld(cfg), lib or _jar_lib(), "r0", sp.height_m, anchor, geom=fake.geom, cfg=cfg, kind="interior", interior=sp)
    return meta, arrays, arrays[f"s0.45_{RB.trunk_key(*TRUNK)}"]


def test_a_stable_rest_inside_the_fridge_in_the_fake_world(fake):
    no_door = _fridge(extra_boxes=[], name="no_door")
    meta0, arr0, st0 = _compute(fake, no_door)
    meta, arr, st = _compute(fake, _fridge())
    assert meta["kind"] == "interior" and meta["interior_placement"] == "fixed" and "lean" not in meta and meta["rest_id"] == "r0"
    assert meta["interior"]["joints"] == DOOR and meta["interior"]["extra_boxes"][0]["name"] == "door_left_door" and meta["key"] != meta0["key"]
    cells = arr["cells"]
    x0, x1 = 0.45 + 0.03, 0.45 + 0.55
    on = (cells[:, 0] - 0.04 >= x0) & (cells[:, 0] + 0.04 <= x1) & (np.abs(cells[:, 1]) + 0.04 <= 0.21)
    assert (st0[~on] == RM.STATUS["off_support"]).all() and (st[~on] == RM.STATUS["off_support"]).all()  # off the board or through a wall
    f0, f1 = (st0 == RM.STATUS["feasible"]).any(axis=(1, 2)), (st == RM.STATUS["feasible"]).any(axis=(1, 2))
    assert f0[on].any() and not (f1 & ~f0).any()  # the door only removes
    blocked = f0 & ~f1
    assert blocked.any() and (cells[blocked, 1] > 0).all()  # the forearm meets the swung-out door on the hinge (left) side only
    assert ((st[blocked] == RM.STATUS["slab_spheres"]) | (st[blocked] == RM.STATUS["hand_policy"])).all()


# ------------------------------------------------------------------------------------------------------------------ identity + find


#: i7-interior 2026-11-07: the place rule of the goal side's lookup (inside_large_02: the outward pre-place, drop 0.03)
PLACE = IP.place_rule(pre="outward", drop_m=0.03)


def _meta(lib, sp, *, arm="right", rest_id="r0", place=None):
    cfg = RM.interior_config(sp, rest_kind="stable", arm=arm, place=place)
    m = {"schema": RM.SCHEMA, "kind": "interior", "plane": "interior", "arm": arm, "object_key": lib.identity_key(), "rest_id": rest_id,
         "height": {"value": sp.height_m, "anchor_key": "interior"}, "edge_distances": list(cfg.edge_distances), "config": cfg.key(),
         "policy_key": CP.DEFAULT.key(arm), "robot": RM.robot_digest(), "brace": None, "object": {"category": "jar_of_spaghetti_sauce", "model": "exmsmm"},
         "interior": sp.to_json(), "interior_placement": "fixed",
         "grasps": [{"name": "side_0", "base": "side_0", "twin": False}, {"name": "side_0~r180", "base": "side_0", "twin": True}],
         "trunks": [{"translate": TRUNK[0], "tilt": TRUNK[1], "eligible": True, "posture": "untucked", "key": RB.trunk_key(*TRUNK)}]}
    m["key"] = RM.map_key(m)
    return m


def _write(d, meta, feasible=(40, 41)):
    os.makedirs(d, exist_ok=True)
    cfg = RM.MapConfig(**{k: v for k, v in meta["config"].items() if k in RM.MapConfig.__dataclass_fields__})
    cells = RM.lattice(cfg).astype(np.float32)
    st = np.full((len(cells), cfg.n_yaws, len(meta["grasps"])), RM.STATUS["ik_pre"], dtype=np.uint8)
    for c in feasible:
        st[c, 0, 0] = RM.STATUS["feasible"]
    arrays = {"cells": cells, "yaws": np.array([2 * np.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)], dtype=np.float32)}
    for d_e in meta["edge_distances"]:
        arrays[f"s{d_e:.2f}_{RB.trunk_key(*TRUNK)}"] = st
    base = os.path.join(d, f"map__jar_of_spaghetti_sauce__exmsmm__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return base + ".json"


def test_find_checks_the_fridge_spec(tmp_path):
    sp, lib = _fridge(), _jar_lib()
    path = _write(str(tmp_path), _meta(lib, sp))
    ok = dict(rest_id="r0", arm="right", anchors=None, policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest(), kind="interior")
    rm = RM.ReachMap.find(str(tmp_path), lib, sp.height_m, interior=sp, **ok)
    assert rm is not None and rm.path == path and rm.meta["edge_distances"] == [0.35, 0.45, 0.55, 0.65]
    for other in (_fridge(joints={"j_left_door": 2.0}), _fridge(edge_distances=[0.35, 0.45, 0.55]), _fridge(extra_boxes=[]), IN.spec_named("bookcase_kmuacd_0@0.8354")):
        assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, interior=other, **ok) is None
    assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, **{**ok, "kind": "tabletop"}) is None  # never a tabletop lookup
    assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, interior=sp, **{**ok, "arm": "left"}) is None


# ------------------------------------------------------------------------------------------------------------------ inside_band


@pytest.fixture
def jar(tmp_path, monkeypatch):
    glib, maps = tmp_path / "glib", tmp_path / "maps"
    d = S.model_dir("jar_of_spaghetti_sauce", "exmsmm", [1, 1, 1], str(glib))
    os.makedirs(d)
    S.save_json(os.path.join(d, "library.json"), _jar_lib().data)
    reg = str(tmp_path / "interiors.json")
    IN.write_registry(reg, [_fridge()])
    for k, v in (("ISPATIALGYM_GRASP_LIBRARY_V2", str(glib)), ("ISPATIALGYM_REACH_MAPS", str(maps)), ("ISPATIALGYM_INTERIORS", reg),
                 ("ISPATIALGYM_V2_ONLY", "1")):
        monkeypatch.setenv(k, v)
    for k in ("ISPATIALGYM_REACH_MAP_ANCHORS", "ISPATIALGYM_REACH_MAP_REQUIRE", "ISPATIALGYM_SUPPORT_POLICY", "ISPATIALGYM_REACH_MAPS_STRICT"):
        monkeypatch.delenv(k, raising=False)
    GL._v2_index.cache_clear()
    lib = GL.v2_entry(JAR_KEY)
    assert lib is not None
    # i7-interior: the goal side reads the PLACE map of its rule; a former interior map (the whole robot at the rest pose) beside it never serves
    former = _write(str(maps / os.path.basename(d)), _meta(lib, _fridge()))
    path = _write(str(maps / os.path.basename(d)), _meta(lib, _fridge(), place=PLACE))
    yield SimpleNamespace(lib=lib, path=path, former=former)
    GL._v2_index.cache_clear()


def _inside(joints=None):
    IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large_inside")

    class Stand:
        inside_band = IL.InsideMode.inside_band

        def inside_place_rule(self):
            return dict(PLACE)

    s = Stand()
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(category="jar_of_spaghetti_sauce", model="exmsmm", scale=None)})
    s.grasp_key = JAR_KEY
    s.world = SimpleNamespace(floor_z=0.0, scene_model=SCENE)
    s.gp = SimpleNamespace(variant="inside_large_scale_02")
    s.ikc = SimpleNamespace(default_working_arms=["right"])
    s.partner = SimpleNamespace(name=FRIDGE)
    s.template_facts = {"partner_joint_init": {"applied": dict(DOOR if joints is None else joints)}}
    top = SimpleNamespace(name=FRIDGE, z=0.7064, headroom=th.tensor([[0.2384, 0.2384]]), mask=th.tensor([[True, True]]), open_edges=["+u"])
    s.place = {"z_requested": 0.70, "z": 0.7064, "top": top}
    return s


def test_inside_band_reads_the_fridge_map_under_v2_only(jar, monkeypatch, caplog):
    def legacy(*a, **k):
        raise AssertionError("a legacy band was loaded under v2 only")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(legacy))
    s = _inside()
    rb = s.inside_band("right")
    assert rb.meta["map_path"] == jar.path and rb.meta["interior"] == _fridge().name and rb.meta["stand_offs"] == [0.35, 0.45, 0.55, 0.65]
    assert s.inside_band("right") is rb and s.template_facts["inside_bands"]["right"]["map_path"] == jar.path
    assert not hasattr(s.ikc, "support_plane_z")  # a place band: no support-policy registration
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        with pytest.raises(Reject) as e:
            s.inside_band("left")  # no left-arm map: rejected - never the right arm's mirrored, never band-less
        assert e.value.reason == "missing_map"
        with pytest.raises(Reject) as e:
            _inside(joints={"j_left_door": 1.2}).inside_band("right")  # the door at another angle: another spec, none registered
        assert e.value.reason == "missing_map"
    reqs = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]
    assert [(r["kind"], r["arm"]) for r in reqs] == [("interior", "left"), ("interior", "right")]
    assert reqs[0]["joints"] == DOOR and reqs[0]["interior"] == _fridge().name and reqs[0]["entry"] == os.path.basename(os.path.dirname(jar.lib.path))
    assert reqs[0]["interior_place"] == PLACE and rb.meta["interior_place"] == PLACE


def test_inside_band_never_reads_a_former_interior_map_or_another_place_rule(jar, monkeypatch, caplog):
    """i7-interior: the former interior map (whole robot, rest pose, one posture / lateral) is stricter than stance C's exact place - it
    never serves the goal side; nor does a place map of another rule (the sink's approach pre-place / tilts for the fridge's outward one)."""
    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy"))))
    os.rename(jar.path, jar.path + ".off")
    os.rename(jar.path[:-5] + ".npz", jar.path[:-5] + ".npz.off")
    with pytest.raises(Reject) as e:
        _inside().inside_band("right")
    assert e.value.reason == "missing_map" and "under the place rule" in str(e.value)
    other = _write(os.path.dirname(jar.path), _meta(jar.lib, _fridge(), place=IP.place_rule(pre="approach", drop_m=0.03, tilts_deg=[30, 45])))
    with pytest.raises(Reject):
        _inside().inside_band("right")
    os.remove(other)
    os.remove(other[:-5] + ".npz")
    tilted = _write(os.path.dirname(jar.path), _meta(jar.lib, _fridge(), place=IP.place_rule(pre="outward", drop_m=0.03, tilts_deg=[30])))
    assert _inside().inside_band("right").meta["map_path"] == tilted  # a place map with more tilts covers the level-only rule


def test_inside_band_off_v2_only_keeps_the_legacy_path(jar, monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_V2_ONLY")
    calls = []

    def load(scene, support, obj, gkey, **kw):
        calls.append((support, kw.get("arm"), kw.get("surface_z"), kw.get("support_joints")))
        raise FileNotFoundError("no band")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(load))
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(TT, "default_arm", lambda s: "right")
    assert _inside().inside_band("right") is None and calls == [(FRIDGE, "right", 0.70, DOOR)]


def test_a_fixed_compartment_picks_the_idle_posture_clear_of_the_door(fake):
    """i7_fridge (job 18367880): every map trunk kept the untucked idle arm, which stood in the open door - stance C's postures
    (untucked / ready / tucked) are judged against the compartment's obstacles too; the map records the posture and the sphere hits."""
    from ispatialgym.tests.test_interior_maps import _Backend

    door_xy = (0.45 - 0.10, 0.20)  # the test door's centre in the map frame at d_e 0.45

    class PostureBackend(_Backend):
        def robot_spheres_world(self, q):
            sph, links = super().robot_spheres_world(q)
            if float(q[0, 0]) == 0.0:  # untucked: the idle left forearm in the door
                sph = th.cat([sph, th.tensor([[[door_xy[0], door_xy[1], 1.0, 0.05]]])], dim=1)
                links = links + ["left_arm_link5"]
            return sph, links

    class PostureWorld(FakeWorld):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.backend = PostureBackend()

        def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
            q = th.zeros(28)
            q[0] = {"untucked": 0.0, "ready": 1.0, "tucked": 2.0}[posture]
            return q

    sp = _fridge()
    cfg = RM.interior_config(sp, rest_kind="stable", seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked", "ready", "tucked"), lattice_m=0.12,
                             y_half_m=0.36, n_yaws=2)
    meta, _ = RM.compute_map(PostureWorld(cfg), _jar_lib(), "r0", sp.height_m, {"key": "x", "height": sp.height_m}, geom=fake.geom, cfg=cfg, kind="interior",
                             interior=sp)
    t = meta["trunks"][0]
    assert t["eligible"] and t["posture"] == "ready" and t["obstacle_free_by_posture"] == {"untucked": [False], "ready": [True]}
    assert isinstance(meta["interior_sphere_hits"], dict)
    # a slab map keeps its rule (the board only): the untucked posture
    cfg_t = RM.MapConfig(seeds=2, batch=64, edge_distances=(0.45,), lattice_m=0.24, y_half_m=0.24, n_yaws=1)
    lib = _jar_lib()
    mt, _ = RM.compute_map(PostureWorld(cfg_t), lib, "r0", 0.6789, {"key": "0.6789", "height": 0.6789}, geom=fake.geom, cfg=cfg_t, kind="tabletop")
    assert mt["trunks"][0]["posture"] == "untucked" and "obstacle_free_by_posture" not in mt["trunks"][0]


def test_registered_digests_stay_and_the_open_top_basin():
    """The specs' digests are the maps' identity: the kmuacd cube (maps rebuilt on trunk_893d7e61) and the measured fridge (c623427d265d)
    keep theirs; an open-top basin (inside_large_scale_03's drop-in sink) has no board above, walls to its rim and a near wall."""
    reg = IN.load_registry(IN.REGISTRY)
    assert reg["bookcase_kmuacd_0@0.8354"].digest() == "b58d4e5c790e"
    fr = reg["fridge_petcxr_0@0.7064/j_left_door2.3"]
    assert fr.digest() == "c623427d265d" and fr.joint_values == {"j_left_door": 2.3} and fr.extra_boxes[0][0] == "door_left_door"
    sink = IN.InteriorSpec(name="sink", height_m=0.77, depth_m=0.40, width_m=0.45, headroom_m=1.505, rim_m=0.19, front_wall_x_m=0.06, open_top=True,
                           back_thickness_m=0.02, board_setback_m=0.06, supports=(("house_double_floor_lower", "drop_in_sink_lkklqs_0"),))
    assert sink.open_sides == ["top"] and sink.wall_height_m == 0.19
    boxes = dict(sink.boxes(0.77, 0.45))
    assert list(boxes) == ["board", "wall_left", "wall_right", "back", "front"]  # no board above
    wl, fw = boxes["wall_left"], boxes["front"]
    assert abs(wl.centre[2] + wl.half[2] - (0.77 + 0.19)) < 1e-9  # the walls end at the rim
    assert abs(fw.centre[0] + fw.half[0] - (0.45 + 0.06)) < 1e-9 and abs(fw.centre[0] - fw.half[0] - 0.45) < 1e-9  # front plane .. inner face
    assert abs(fw.centre[2] + fw.half[2] - 0.96) < 1e-9
    g = sink.geometry()
    assert g["open_top"] and g["rim_m"] == 0.19 and g["front_wall_x_m"] == 0.06 and "open_top" not in fr.geometry()
    assert IN.InteriorSpec.from_json(sink.to_json()).digest() == sink.digest() != IN.InteriorSpec.from_json({**sink.to_json(), "rim_m": 0.25}).digest()
