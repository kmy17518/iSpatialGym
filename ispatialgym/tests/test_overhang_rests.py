"""i7-overhang 2026-11-07 (owner decision: the "overhang" rest option of the v2 grasp library, hardback bqvvvx / inside_large_01): the
overhang spec and its identity / keys, the settle world's slab and the measured record, the rest kind in the schema (entry ids,
compatibility, identity keys), the grasp library's overhang / stable separation (an overhang rest never serves a stable lookup, nor the
reverse), the overhang reach-map kind (config, cells, support mask, compute in a fake synthetic world, key, find). Simulator-free."""

import json
import math
import os

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import overhang as OV
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

BO_KEY = "hardback__bqvvvx__bo"
#: the hardback's collision box (results/hardback__bqvvvx__bo.json bbox 0.2318 x 0.1693 x 0.0305; the spine face at y -0.0861)
BOX = (np.array([-0.1159, -0.0861, -0.01525]), np.array([0.1159, 0.0832, 0.01525]))
#: the measured push of the inside_large_01 demos: the spine 0.0294 m beyond the desk edge at the pinch line (hb30_desk_s41 + 51 fast-flow copies)
D_DEMO = 0.0294
SPEC = OV.OverhangSpec("-y", D_DEMO)
#: the legacy bo grasp_1 (cands/hardback__bqvvvx__bo.yaml): the eef 1.8 mm outside the spine, approach +y 10 deg down, closing ~vertical
BO_GRASP_T = {"position": [0.0348, -0.0879, 0.0019], "orientation_xyzw": [0.0, -0.7660444, -0.6427876, 0.0]}


def _box_vertices(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)


# ------------------------------------------------------------------------------------------------------------------ the spec + keys


def test_overhang_spec_identity_separates_side_distance_and_slab():
    others = [OV.OverhangSpec("+y", D_DEMO), OV.OverhangSpec("-y", 0.03), OV.OverhangSpec("-x", D_DEMO), OV.OverhangSpec("-y", D_DEMO, slab_thickness_m=0.04)]
    assert len({SPEC.digest(), *[o.digest() for o in others]}) == 5 and len({SPEC.tag(), *[o.tag() for o in others]}) == 5
    assert SPEC.tag().startswith("overhang_yn_29mm_") and len(SPEC.tag().split("_")[-1]) == 6 and SPEC.rest_id() == SPEC.tag()
    assert OV.OverhangSpec("+x", 0.05).tag().startswith("overhang_xp_50mm_")
    j = SPEC.to_json()
    assert j["kind"] == "overhang" and j["side"] == "-y" and j["distance_m"] == D_DEMO and j["edge"] == "flush" and j["digest"] == SPEC.digest()
    assert OV.OverhangSpec.from_json(json.loads(json.dumps(j))) == SPEC and SPEC.axis == 1 and SPEC.sign == -1.0
    for bad in ({"side": "-z", "distance_m": 0.03}, {"side": "-y", "distance_m": 0.0}, {"side": "-y", "distance_m": 0.3},
                {"side": "-y", "distance_m": 0.03, "edge": "diagonal"}):
        with pytest.raises(ValueError):
            OV.OverhangSpec(**bad)


def test_overhang_keys_split_and_never_parse_a_lean_or_plain_key():
    k = OV.overhang_key(BO_KEY, SPEC)
    assert k == f"{BO_KEY}@{SPEC.tag()}" and OV.split_overhang_key(k) == (BO_KEY, SPEC.tag()) and OV.is_overhang_key(k)
    assert OV.split_overhang_key(BO_KEY) == (BO_KEY, None) and not OV.is_overhang_key(None)
    from ispatialgym.grasping import lean as LN

    lk = LN.lean_key("hardback__acbrnv__bu", LN.LeanSpec(11, "left"))
    assert OV.split_overhang_key(lk) == (lk, None) and LN.split_lean_key(k) == (k, None)  # the two key kinds never parse each other
    assert GL.overhang_key_parts(k) == (BO_KEY, SPEC.tag()) and GL.lean_key_parts(k) == (k, None)


# ------------------------------------------------------------------------------------------------------------------ the settle geometry


def test_slab_edge_lies_the_distance_inside_the_face():
    V = _box_vertices(*BOX)
    assert abs(OV.face_extreme(V, SPEC) - (-0.0861)) < 1e-9
    assert abs(OV.edge_coordinate(V, SPEC) - (-0.0861 + D_DEMO)) < 1e-9
    sl = OV.slab_for(V, SPEC, obj_xy=(0.0, 0.0), half=0.35, slab_top_z=0.75)
    assert abs(sl["edge_world"] - (-0.0567)) < 1e-6
    c, dims = np.array(sl["centre"]), np.array(sl["dims"])
    assert abs((c[1] - dims[1] / 2) - sl["edge_world"]) < 1e-9  # the slab's -y face IS the edge, the slab on the object's +y side
    assert abs(c[2] + dims[2] / 2 - 0.75) < 1e-9 and c[0] == 0.0
    # +x side: the edge 0.05 inside the +x face, the slab on the -x side
    sp = OV.OverhangSpec("+x", 0.05)
    sl2 = OV.slab_for(V, sp, obj_xy=(1.0, 2.0), half=0.3)
    assert abs(sl2["edge_world"] - (1.0 + 0.1159 - 0.05)) < 1e-9 and abs(sl2["centre"][0] + 0.3 - sl2["edge_world"]) < 1e-9 and sl2["centre"][1] == 2.0


def test_overhang_record_and_its_acceptance():
    V = _box_vertices(*BOX)
    q = np.array([0, 0, 0, 1.0])
    p = np.array([0.0, 0.0, 0.75 + 0.01525])
    sl = OV.slab_for(V, SPEC, half=0.35)
    rec = OV.overhang_record(SPEC, V, q, p, sl["edge_world"], [0.0, -0.0015, 0.0], sl["centre"], sl["dims"], q)
    assert abs(rec["measured_distance_m"] - D_DEMO) < 1e-6 and abs(rec["edge_along_m"] - (0.0861 - D_DEMO)) < 1e-6
    assert abs(rec["com_inside_m"] - (0.0861 - D_DEMO - 0.0015)) < 1e-6 and rec["normal_xy"] == [0.0, -1.0] and abs(rec["flush_turn_deg"]) < 1e-9
    assert OV.overhang_ok(rec) == (True, "ok")
    # the settled object slid 6 mm off the edge: not this rest
    p2 = p + np.array([0.0, 0.006, 0.0])
    r2 = OV.overhang_record(SPEC, V, q, p2, sl["edge_world"], [0, 0, 0], sl["centre"], sl["dims"], q)
    assert not OV.overhang_ok(r2)[0] and "overhangs" in OV.overhang_ok(r2)[1]
    # turned 5 deg about z: not flush
    t = math.radians(5.0)
    q3 = np.array([0, 0, math.sin(t / 2), math.cos(t / 2)])
    r3 = OV.overhang_record(SPEC, V, q3, p, sl["edge_world"], [0, 0, 0], sl["centre"], sl["dims"], q)
    assert abs(abs(r3["flush_turn_deg"]) - 5.0) < 1e-6 and not OV.overhang_ok(r3, tol_m=0.02)[0]
    # the support box in the rest frame: its edge where it was measured, its top at z_top
    b = OV.support_in_rest(rec, z_top=-0.01525)
    assert b.kind == "box" and abs(b.centre[1] - b.half[1] - (-0.0861 + D_DEMO)) < 1e-6 and abs(b.top_z + 0.01525) < 1e-9
    pts = th.tensor([[0.0, -0.075, -0.03], [0.0, -0.04, -0.03]])  # below the slab top: beyond the edge free, inside it solid
    sd = b.sdf(pts)
    assert float(sd[0]) > 0.01 and float(sd[1]) < 0


# ------------------------------------------------------------------------------------------------------------------ schema


def _base_job():
    return {"category": "hardback", "model": "bqvvvx", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "constraint": "book_overhang",
            "rest_hint_wxyz": None, "legacy_key": BO_KEY}


def test_entry_ids_and_job_dirs_keep_stable_ones_and_separate_overhang_ones(tmp_path):
    j = OV.make_job(_base_job(), SPEC, base_key=BO_KEY)
    assert j["id"] == f"hardback__bqvvvx__s1__{SPEC.tag()}" and j["rest_kind"] == "overhang" and j["constraint"] == "tabletop"
    assert j["legacy_key"] is None and j["overhang_keys"] == [BO_KEY] and OV.is_overhang_job(j) and S.job_overhang(j) == j["overhang"]
    assert S.entry_id("hardback", "bqvvvx", None) == "hardback__bqvvvx__s1"  # the stable id as before
    assert S.job_dir(j, str(tmp_path)).endswith(j["id"]) and S.job_dir({**_base_job(), "constraint": "tabletop"}, str(tmp_path)).endswith("hardback__bqvvvx__s1")
    assert S.job_overhang({**j, "rest_kind": "stable"}) is None


def _ov_rest(spec=SPEC, edge_along=0.0861 - D_DEMO, q=(0, 0, 0, 1.0), h=0.01525, rid=None):
    return {"schema": S.REST, "rest_id": rid or spec.tag(), "quat_xyzw": list(q), "h_root": h, "stable": True, "rest_kind": "overhang",
            "overhang": {**spec.to_json(), "measured_distance_m": spec.distance_m, "com_inside_m": edge_along, "edge_along_m": edge_along,
                         "normal_xy": [0.0, -1.0], "flush_turn_deg": 0.0,
                         "slab_in_root": {"centre": [0.0, -edge_along + 0.35, -0.04], "half": [0.35, 0.35, 0.025]}}}


def _ov_lib(spec=SPEC, edge_along=0.0861 - D_DEMO, grasp_T=BO_GRASP_T, path="ov/library.json"):
    data = {"schema": S.LIBRARY, "category": "hardback", "model": "bqvvvx", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "arm": "right",
            "policy_key": CP.DEFAULT.key("right"), "rest_kind": "overhang", "overhang": spec.to_json(), "overhang_keys": [BO_KEY], "legacy_key": None,
            "rests": {spec.tag(): _ov_rest(spec, edge_along)}, "families": {},
            "grasps": [{"name": "side_0", "rest_id": spec.tag(), "family_id": None, "T_object_eef": grasp_T, "width_m": 0.0305, "preshape_q": 0.02525,
                        "finger_q": [0.0159, 0.0159]}]}
    return S.ModelLibrary(path=path, data=data, rests={k: S.Rest.from_record(v) for k, v in data["rests"].items()})


def test_rest_kind_compatibility_never_crosses():
    lib = _ov_lib()
    r = lib.rests[SPEC.tag()]
    assert r.kind == "overhang" and r.overhang["digest"] == SPEC.digest() and r.lean is None and lib.rest_kind == "overhang"
    S.check_compatible(lib, rest_id=SPEC.tag(), rest_kind="overhang", overhang=SPEC.to_json())
    with pytest.raises(S.IncompatibleRecord, match="overhang rest"):
        S.check_compatible(lib, rest_id=SPEC.tag())  # a stable consumer
    with pytest.raises(S.IncompatibleRecord, match="has overhang"):
        S.check_compatible(lib, rest_id=SPEC.tag(), rest_kind="overhang", overhang=OV.OverhangSpec("-y", 0.03).to_json())
    # the identity carries the overhang; a stable rest's identity is as before
    ident = lib.identity()["rests"][SPEC.tag()]
    assert ident["rest_kind"] == "overhang" and ident["overhang"] == SPEC.digest() and ident["edge_along_m"] == round(0.0861 - D_DEMO, 6)
    st = S.Rest.from_record({"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.015, "stable": True})
    lib2 = S.ModelLibrary(path="x", data=dict(lib.data, rests={}), rests={"r0": st})
    assert lib2.identity()["rests"]["r0"] == {"quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.015}


# ------------------------------------------------------------------------------------------------------------------ the grasp library


def _write_entry(root, eid, data):
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "library.json"), "w") as f:
        json.dump(data, f)
    with open(os.path.join(d, "candidates.json"), "w") as f:
        json.dump({"rest_hint_wxyz": None}, f)


@pytest.fixture
def v2root(tmp_path, monkeypatch):
    root = str(tmp_path / "glib")
    lib = _ov_lib()
    _write_entry(root, f"hardback__bqvvvx__s1__{SPEC.tag()}", lib.data)
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", root)
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()
    yield root
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()


def test_an_overhang_entry_never_serves_the_stable_key_and_serves_its_own(v2root, monkeypatch):
    assert GL.v2_entry(BO_KEY) is None  # the plain key: no stable entry (the legacy library serves it, or v2only blocks it)
    k = OV.overhang_key(BO_KEY, SPEC)
    lib = GL.v2_entry(k)
    assert lib is not None and lib.rest_kind == "overhang" and GL.v2_rest_id_for_key(lib, k) == SPEC.tag() and GL.v2_rest_id(lib) is None
    gs = GL.load_grasps(k)
    assert [g.name for g in gs] == ["side_0"] and gs[0].meta["rest_id"] == SPEC.tag()
    assert GL.v2_overhang_tags(BO_KEY) == [(SPEC.tag(), SPEC.to_json())] and GL.v2_overhang_tags("other__key") == []
    assert th.allclose(GL.rest_quat_xyzw(k), th.tensor([0.0, 0.0, 0.0, 1.0]))
    # another overhang tag of the same key: no entry
    with pytest.raises(ValueError, match="no v2 overhang entry"):
        GL.load_grasps(OV.overhang_key(BO_KEY, OV.OverhangSpec("-y", 0.03)))
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    with pytest.raises(ValueError, match="legacy grasps are not served"):
        GL.load_grasps(BO_KEY)  # v2only: the plain key stays blocked (an overhang entry is no stable one)
    from ispatialgym.robot import conventions as C

    g = C.grasp_entry(k, "side_0~r180")  # the oracle resolves the recorded name through the same key
    assert g.name == "side_0~r180"


# ------------------------------------------------------------------------------------------------------------------ the overhang map kind


def test_overhang_geometry_config_cells_and_mask():
    lib = _ov_lib()
    r = lib.rests[SPEC.tag()]
    geo = RM.overhang_geometry(r)
    assert abs(geo["yaw_rad"] + math.pi / 2) < 1e-6 and abs(geo["edge_along_m"] - (0.0861 - D_DEMO)) < 1e-9 and geo["digest"] == SPEC.digest()
    # the yaw turns the face normal to -x of the map frame
    c, s = math.cos(geo["yaw_rad"]), math.sin(geo["yaw_rad"])
    n = np.array([c * 0.0 - s * -1.0, s * 0.0 + c * -1.0])
    assert np.allclose(n, [-1.0, 0.0], atol=1e-6)
    cfg = RM.overhang_config(r)
    assert cfg.n_yaws == 1 and cfg.edge_distances == RM.OVERHANG_EDGE_DISTANCES and cfg.lattice_m == RM.OVERHANG_LATTICE_M and cfg.overhang == SPEC.digest()
    assert cfg.key()["overhang"] == SPEC.digest() and "overhang" not in RM.MapConfig().key()  # other kinds' keys stay
    cells = RM.overhang_cells(cfg, geo["edge_along_m"])
    nl = len(np.arange(-cfg.y_half_m, cfg.y_half_m + 1e-9, cfg.lattice_m))
    assert cells.shape == (nl * len(cfg.edge_distances), 2)
    assert set(np.round(cells[:, 0], 6)) == {round(d + geo["edge_along_m"], 6) for d in cfg.edge_distances}
    m = RM.support_mask(cells, np.zeros(1), 0.45, "overhang", lateral=geo["edge_along_m"])
    assert m[:, 0].sum() == nl and np.allclose(cells[m[:, 0], 0], 0.45 + geo["edge_along_m"])
    assert "overhang" in RM.KINDS and RM.PLANE_OF_KIND["overhang"] == "slab_edge" and RM.REST_KINDS_OF_KIND["overhang"] == ("overhang",)
    assert not RM.rest_kind_ok("tabletop", "overhang") and not RM.rest_kind_ok("overhang", "stable")


@pytest.fixture
def fake_world(monkeypatch):
    from types import SimpleNamespace

    from ispatialgym.tests import test_interior_maps as TIM

    monkeypatch.setattr(RM, "_solve", TIM._fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [TIM.TRUNK])
    import trimesh

    lo, hi = BOX
    box = trimesh.creation.box(extents=list(hi - lo))
    box.apply_translation((lo + hi) / 2)
    return TIM, SimpleNamespace(union=box, pieces=[box])


def test_overhang_map_in_a_fake_synthetic_world(fake_world):
    TIM, geom = fake_world
    lib = _ov_lib()
    r = lib.rests[SPEC.tag()]
    cfg = RM.overhang_config(r, seeds=2, batch=64, edge_distances=(0.45, 0.55), postures=("untucked",), y_half_m=0.16)
    world = TIM.FakeWorld(cfg)
    anchor = {"key": "0.7671", "height": 0.7671, "surfaces": ["house_single_floor/desk_bhkhxo_0/top"]}
    meta, arrays = RM.compute_map(world, lib, SPEC.tag(), 0.7671, anchor, geom=geom, cfg=cfg, kind="overhang")
    assert meta["kind"] == "overhang" and meta["plane"] == "slab_edge" and meta["overhang"]["digest"] == SPEC.digest() and meta["slab"] is not None
    assert len(arrays["yaws"]) == 1 and abs(float(arrays["yaws"][0]) + math.pi / 2) < 1e-6
    cells = arrays["cells"]
    for d_e in (0.45, 0.55):
        st = arrays[f"s{d_e:.2f}_{RB.trunk_key(*TIM.TRUNK)}"]
        row = np.abs(cells[:, 0] - (d_e + r.overhang["edge_along_m"])) < 1e-6
        assert (st[~row] == RM.STATUS["off_support"]).all()
        assert (st[row] == RM.STATUS["feasible"]).any()  # the lower finger out over the free space beyond the edge
    # the same pinch 3 cm deeper into the book: its lower finger under the slab top inside the edge - the hand policy fails it
    deep = dict(BO_GRASP_T, position=[0.0348, -0.0579, 0.0019])
    _, arr2 = RM.compute_map(world, _ov_lib(grasp_T=deep), SPEC.tag(), 0.7671, anchor, geom=geom, cfg=cfg, kind="overhang")
    st2 = arr2[f"s0.45_{RB.trunk_key(*TIM.TRUNK)}"]
    assert not (st2 == RM.STATUS["feasible"]).any() and (st2 == RM.STATUS["hand_policy"]).any()
    # the kinds never cross: an overhang rest in a tabletop map, a tabletop config in an overhang map
    with pytest.raises(ValueError, match="stable rest"):
        RM.compute_map(world, lib, SPEC.tag(), 0.7671, anchor, geom=geom, cfg=RM.MapConfig(), kind="tabletop")
    with pytest.raises(ValueError, match="overhang_config"):
        RM.compute_map(world, lib, SPEC.tag(), 0.7671, anchor, geom=geom, cfg=RM.MapConfig(), kind="overhang")
    # the key carries the overhang rest; find checks it
    assert meta["key"] == RM.map_key(meta) and meta["key"] != RM.map_key(dict(meta, overhang=dict(meta["overhang"], digest="other")))


def test_find_an_overhang_map_checks_the_rest(fake_world, tmp_path):
    TIM, geom = fake_world
    lib = _ov_lib()
    r = lib.rests[SPEC.tag()]
    cfg = RM.overhang_config(r, seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked",), y_half_m=0.08)
    world = TIM.FakeWorld(cfg)
    anchor = {"key": "0.7671", "height": 0.7671}
    meta, arrays = RM.compute_map(world, lib, SPEC.tag(), 0.7671, anchor, geom=geom, cfg=cfg, kind="overhang")
    p = RM.save(str(tmp_path), meta, arrays)
    rm = RM.ReachMap.find(str(tmp_path), lib, 0.7671, rest_id=SPEC.tag(), kind="overhang", policy_key=CP.DEFAULT.key("right"))
    assert rm is not None and rm.path == p
    assert RM.ReachMap.find(str(tmp_path), lib, 0.7671, rest_id=SPEC.tag(), kind="tabletop") is None  # an overhang rest: overhang maps only
    assert RM.ReachMap.find(str(tmp_path), lib, 0.7671, rest_id=SPEC.tag(), kind="overhang", arm="left") is None
    other = _ov_lib(spec=OV.OverhangSpec("-y", 0.03))
    assert RM.ReachMap.find(str(tmp_path), other, 0.7671, rest_id=OV.OverhangSpec("-y", 0.03).tag(), kind="overhang") is None
