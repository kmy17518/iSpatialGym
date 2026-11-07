"""i7-lean 2026-11-07 (owner decision; prompt I item 7): the INTERIOR reach-map kind at a LEAN rest - the compartment spec (boards, walls,
open sides; the registry's kmuacd cube), its identity in the map key and in ``ReachMap.find``, the support mask on the board, a map computed
in a FAKE interior synthetic world (the compartment following the cell, the board policed, the walls normal obstacles). touching_03's
consumer: ``test_interior_consumer.py``. Simulator-free."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import interiors as IN
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

KMUACD = "bookcase_kmuacd_0@0.8354"
SCENE, SUPPORT = "house_double_floor_upper", "bookcase_kmuacd_0"
HB_KEY = "hardback__acbrnv__bu"
TRUNK = (0.5, 0.0)


def _spec(**kw):
    base = IN.spec_named(KMUACD)
    if not kw:
        return base
    d = {**base.geometry(), **kw, "supports": [list(x) for x in base.supports]}
    return IN.InteriorSpec.from_json(d, name=kw.get("name", base.name + "_x"))


# ------------------------------------------------------------------------------------------------------------------ the compartment spec


def test_the_registry_states_the_kmuacd_compartment():
    sp = IN.spec_named(KMUACD)
    assert sp.height_m == 0.8354 and sp.width_m == 0.346 and sp.headroom_m == 0.3511 and sp.back_thickness_m is None
    assert sp.open_sides == ["front", "back"] and (SCENE, SUPPORT) in sp.supports and sp.source.get("measured")
    boxes = dict(sp.boxes(0.8354, 0.45))
    assert list(dict(sp.boxes(0.8354, 0.45))) == ["board", "above", "wall_left", "wall_right"]  # open back: no back panel
    b, a, wl, wr = boxes["board"], boxes["above"], boxes["wall_left"], boxes["wall_right"]
    assert abs(b.top_z - 0.8354) < 1e-9 and abs((b.centre[0] - b.half[0]) - (0.45 + 0.0067)) < 1e-9 and abs(b.centre[0] + b.half[0] - (0.45 + 0.3056)) < 1e-9
    assert abs((a.centre[2] - a.half[2]) - (0.8354 + 0.3511)) < 1e-9  # the underside above, the headroom up
    assert abs((wl.centre[1] - wl.half[1]) - 0.173) < 1e-9 and abs((wr.centre[1] + wr.half[1]) + 0.173) < 1e-9  # inner faces width apart
    assert abs(wl.centre[0] - wl.half[0] - 0.45) < 1e-9  # the walls' front faces are the front plane (d_e)
    assert sp.policed(0.8354, 0.45).name == "interior_board" and [o.name for o in sp.obstacles(0.8354, 0.45)] == ["interior_above", "interior_wall_left", "interior_wall_right"]
    assert sp.board_x_range(0.45) == (0.45 + 0.0067, 0.45 + 0.3056)
    closed = _spec(back_thickness_m=0.01)
    assert closed.open_sides == ["front"] and [n for n, _ in closed.boxes(0.8354, 0.45)][-1] == "back" and closed.digest() != sp.digest()
    assert _spec(width_m=0.35).digest() != sp.digest() and IN.InteriorSpec.from_json(sp.to_json()).digest() == sp.digest()
    assert IN.InteriorSpec.from_json(dict(sp.to_json(), source={"x": 1}, name="other")).digest() == sp.digest()  # provenance is no geometry


def test_spec_lookup_by_support_and_the_live_check():
    sp = IN.spec_for_support(SCENE, SUPPORT, 0.8401)
    assert sp is not None and sp.name == KMUACD
    assert IN.spec_for_support(SCENE, SUPPORT, 0.86) is None and IN.spec_for_support(SCENE, "bookcase_otwukr_1", 0.8354) is None
    assert IN.spec_for_support("Rs_int", SUPPORT, 0.8354) is None
    mask = th.tensor([[True, True, False]])
    top = SimpleNamespace(z=0.8354, headroom=th.tensor([[0.3465, 0.3460, 1.5]]), mask=mask, open_edges=["+u", "-u"])
    assert IN.live_mismatch(sp, top) is None  # the raster's headroom is measured 5 mm above the board
    assert "board at" in IN.live_mismatch(sp, SimpleNamespace(**{**vars(top), "z": 0.86}))
    assert "headroom" in IN.live_mismatch(sp, SimpleNamespace(**{**vars(top), "headroom": th.tensor([[0.30, 0.30, 1.5]])}))
    assert "open sides" in IN.live_mismatch(sp, SimpleNamespace(**{**vars(top), "open_edges": ["+u"]}))
    assert IN.compartment_offsets(np.array([0.0, 0.1]), 0.06, 1.0, 0.346).tolist() == pytest.approx([0.06 - 0.173, 0.16 - 0.173])
    assert IN.compartment_offsets(np.array([0.0]), -0.06, -1.0, 0.346).tolist() == pytest.approx([-0.06 + 0.173])
    assert IN.fits(0.346, 0.06, 1.0, -0.02) and not IN.fits(0.346, 0.06, 1.0, -0.30)


def test_interior_config_enters_the_key_and_other_keys_stay():
    sp = IN.spec_named(KMUACD)
    cfg = RM.interior_config(sp, arm="left")
    k = cfg.key()
    assert k["interior"] == sp.digest() and k["n_yaws"] == 1 and k["x_max_m"] == round(0.55 + 0.3056, 4) and k["arm"] == "left"
    assert "interior" not in RM.MapConfig().key() and "interior" not in RM.floor_config().key()
    cells = RM.lattice(cfg)
    assert abs(cells[:, 0].min() - 0.35) < 1e-9 and cells[:, 0].max() <= 0.55 + 0.3056 + 1e-9
    m = RM.support_mask(np.array([[0.50, 0.0], [0.60, 0.0], [0.70, 0.0]]), np.array([-0.085]), 0.45, "interior", reach_front=np.array([0.085]), interior=sp)
    assert m[:, 0].tolist() == [False, True, False]  # the footprint on the board (0.4567 .. 0.7556)
    assert "interior" in RM.KINDS and RM.PLANE_OF_KIND["interior"] == "interior" and RM.REST_KIND_OF_KIND == {"tabletop": "stable", "brace": "stable", "floor": "stable", "interior": "lean"}


# ------------------------------------------------------------------------------------------------------------------ a lean library entry


LEAN = LN.LeanSpec(11.0, "left")


def _lean_rest(face=0.06, spec=LEAN, q=(0, 0, 0, 1.0), h=0.1115):
    return {"schema": S.REST, "rest_id": spec.tag(), "quat_xyzw": list(q), "h_root": h, "stable": True, "rest_kind": "lean",
            "lean": {**spec.to_json(), "measured_deg": spec.deg + 0.4, "measured_side": spec.side, "board_face_y_m": face,
                     "board_in_root": {"centre": [0.0, face + 0.01, 0.09], "half": [0.2, 0.01, 0.2]}, "board_pose": {"centre": [0, 0, 0], "dims": [0.4, 0.02, 0.4]}}}


#: a side grasp of an upright book (0.17 deep, 0.031 thick, 0.223 tall; identity rest for the fake world): approach +x at the spine, closing y
GRASP_T = {"position": [-0.073, 0.0, 0.0], "orientation_xyzw": [0.0, 0.7071068, 0.0, 0.7071068]}


def _lean_lib(face=0.06, spec=LEAN, path="lean/library.json"):
    data = {"schema": S.LIBRARY, "category": "hardback", "model": "acbrnv", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "arm": "right",
            "policy_key": CP.DEFAULT.key("right"), "rest_kind": "lean", "lean": spec.to_json(), "lean_keys": [HB_KEY], "legacy_key": None,
            "rests": {spec.tag(): _lean_rest(face, spec)}, "families": {},
            "grasps": [{"name": "side_0", "rest_id": spec.tag(), "family_id": None, "T_object_eef": GRASP_T, "width_m": 0.031, "preshape_q": 0.0255,
                        "finger_q": [0.016, 0.016]}]}
    return S.ModelLibrary(path=path, data=data, rests={k: S.Rest.from_record(v) for k, v in data["rests"].items()})


def _stable_lib():
    lib = _lean_lib()
    d = dict(lib.data, rests={"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.1115, "stable": True}})
    d.pop("rest_kind")
    d["grasps"] = [dict(g, rest_id="r0") for g in d["grasps"]]
    return S.ModelLibrary(path="stable/library.json", data=d, rests={"r0": S.Rest.from_record(d["rests"]["r0"])})


# ------------------------------------------------------------------------------------------------------------------ the fake synthetic world


class _Backend:
    position_tolerance = 0.005
    orientation_tolerance = 0.05
    ground_contact_links = ()

    def check_collision(self, q, emb_name="whole_body", margin=0.0):
        return th.zeros(q.shape[0], dtype=th.bool)

    def robot_spheres_world(self, q):
        return th.tensor([[[0.0, 0.0, 1.0, 0.1], [-0.3, 0.0, 0.3, 0.05]]]), ["torso_link1", "base_link"]


class FakeWorld:
    """The synthetic world's interface for ``compute_map`` without a simulator or cuRobo: an IK 'solution' carries its target pose (12
    numbers: the 3x4 [R | p]); FK reads it back; the spheres are the torso / base (fixed) and one forearm sphere 0.15 m behind the eef."""

    def __init__(self, cfg):
        self.cfg, self.dev, self.backend = cfg, "cpu", _Backend()

    def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
        return th.zeros(28)

    def shoulder(self, translate, tilt):
        return np.array([0.0, 0.0, 1.1]), 2.0

    def entry(self):
        return SimpleNamespace(rsc=SimpleNamespace(joint_names=[f"j{i}" for i in range(12)]))

    def full_q(self, entry, q_active, q_lock):
        return q_active.to(th.float32)

    def eef(self, q_full):
        T = th.eye(4).repeat(q_full.shape[0], 1, 1)
        T[:, :3, :4] = q_full.reshape(-1, 3, 4)
        return T

    def spheres(self, q_full):
        T = self.eef(q_full)
        n = T.shape[0]
        arm = T[:, :3, 3] - 0.15 * T[:, :3, 2]
        fixed = th.tensor([[0.0, 0.0, 1.0, 0.1], [-0.3, 0.0, 0.3, 0.05]]).repeat(n, 1, 1)
        return th.cat([fixed, th.cat([arm, th.full((n, 1), 0.04)], 1)[:, None]], 1)

    def link_of_sphere(self):
        return ["torso_link1", "base_link", "right_arm_link6"]


def _fake_solve(world, entry, T, q_lock, seed_active=None):
    n = T.shape[0]
    q = T[:, :3, :4].reshape(n, 1, 12).repeat(1, world.cfg.seeds, 1)
    return th.ones(n, world.cfg.seeds, dtype=th.bool), q


@pytest.fixture
def fake_world(monkeypatch):
    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [TRUNK])
    import trimesh

    box = trimesh.creation.box(extents=[0.17, 0.031, 0.223])
    geom = SimpleNamespace(union=box, pieces=[box])
    cfg = RM.interior_config(IN.spec_named(KMUACD), seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked",), lattice_m=0.16, y_half_m=0.32)
    return FakeWorld(cfg), geom, cfg


def test_interior_map_in_a_fake_synthetic_world(fake_world):
    world, geom, cfg = fake_world
    sp = IN.spec_named(KMUACD)
    anchor = {"key": f"interior:{sp.name}", "height": sp.height_m, "surfaces": [f"{SCENE}/{SUPPORT}"]}
    lib = _lean_lib(face=0.06)
    meta, arrays = RM.compute_map(world, lib, LEAN.tag(), sp.height_m, anchor, geom=geom, cfg=cfg, kind="interior", interior=sp)
    assert meta["kind"] == "interior" and meta["plane"] == "interior" and meta["interior"]["digest"] == sp.digest() and meta["slab"] is None
    assert meta["lean"]["digest"] == LEAN.digest() and meta["lean"]["board_face_y_m"] == 0.06 and meta["rest_id"] == LEAN.tag()
    st = arrays[f"s0.45_{RB.trunk_key(*TRUNK)}"]
    cells = arrays["cells"]
    assert st.shape == (len(cells), 1, len(meta["grasps"])) and len(arrays["yaws"]) == 1  # one object yaw
    on = (cells[:, 0] + -0.085 >= 0.45 + 0.0067) & (cells[:, 0] + 0.085 <= 0.45 + 0.3056)
    assert on.any() and (~on).any()
    assert (st[~on] == RM.STATUS["off_support"]).all()  # the footprint off the board
    feas = st == RM.STATUS["feasible"]
    assert feas[on].any() and not feas[~on].any()
    # the compartment follows the cell: every on-board cell of the lateral strip sees the same walls (the lattice's y does not matter)
    rows = {round(float(c[0]), 3) for c in cells[on]}
    for x in rows:
        col = [i for i in np.nonzero(on)[0] if round(float(cells[i, 0]), 3) == x]
        assert len({tuple(st[i].ravel().tolist()) for i in col}) == 1
    # the wall 2 cm off the root (the board-side finger in it): the hand fails the normal obstacle - hand_policy, nothing feasible
    tight = _lean_lib(face=0.02)
    _, arr2 = RM.compute_map(world, tight, LEAN.tag(), sp.height_m, anchor, geom=geom, cfg=cfg, kind="interior", interior=sp)
    st2 = arr2[f"s0.45_{RB.trunk_key(*TRUNK)}"]
    assert not (st2 == RM.STATUS["feasible"]).any() and (st2[on] == RM.STATUS["hand_policy"]).all()
    # the board above lowered onto the forearm's path: the forearm sphere 0.15 m behind the eef meets it - slab_spheres (sphere obstacle)
    low = _spec(headroom_m=0.115, name="low")
    cfg_low = RM.interior_config(low, seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked",), lattice_m=0.16, y_half_m=0.32)
    world_low = FakeWorld(cfg_low)
    _, arr3 = RM.compute_map(world_low, lib, LEAN.tag(), low.height_m, anchor, geom=geom, cfg=cfg_low, kind="interior", interior=low)
    st3 = arr3[f"s0.45_{RB.trunk_key(*TRUNK)}"]
    assert not (st3 == RM.STATUS["feasible"]).any() and ((st3[on] == RM.STATUS["hand_policy"]) | (st3[on] == RM.STATUS["slab_spheres"])).all()
    assert meta["key"] != RM.compute_map(world, lib, LEAN.tag(), sp.height_m, anchor, geom=geom, cfg=RM.interior_config(_spec(width_m=0.36), seeds=2, batch=64,
                                                     edge_distances=(0.45,), postures=("untucked",), lattice_m=0.16, y_half_m=0.32), kind="interior",
                                                     interior=_spec(width_m=0.36))[0]["key"]


def test_compute_never_crosses_rest_kinds(fake_world):
    world, geom, cfg = fake_world
    sp = IN.spec_named(KMUACD)
    anchor = {"key": "x", "height": sp.height_m}
    with pytest.raises(ValueError, match="lean rest"):
        RM.compute_map(world, _stable_lib(), "r0", sp.height_m, anchor, geom=geom, cfg=cfg, kind="interior", interior=sp)
    with pytest.raises(ValueError, match="stable rest"):
        RM.compute_map(world, _lean_lib(), LEAN.tag(), 0.8126, anchor, geom=geom, cfg=RM.MapConfig(), kind="tabletop")
    with pytest.raises(ValueError, match="interior_config"):
        RM.compute_map(world, _lean_lib(), LEAN.tag(), sp.height_m, anchor, geom=geom, cfg=RM.MapConfig(), kind="interior", interior=sp)
    with pytest.raises(ValueError, match="board height"):
        RM.compute_map(world, _lean_lib(), LEAN.tag(), 0.80, anchor, geom=geom, cfg=cfg, kind="interior", interior=sp)
    with pytest.raises(ValueError, match="does not fit"):
        RM.compute_map(world, _lean_lib(face=0.40), LEAN.tag(), sp.height_m, anchor, geom=geom, cfg=cfg, kind="interior", interior=sp)


# ------------------------------------------------------------------------------------------------------------------ identity + find


def _meta(lib, sp, *, arm="right", rest_id=None, lean=LEAN, kind="interior"):
    cfg = RM.interior_config(sp, arm=arm) if kind == "interior" else RM.MapConfig(arm=arm)
    m = {"schema": RM.SCHEMA, "kind": kind, "arm": arm, "object_key": lib.identity_key(), "rest_id": rest_id or lean.tag(),
         "height": {"value": sp.height_m, "anchor_key": "interior"}, "edge_distances": list(cfg.edge_distances), "config": cfg.key(),
         "policy_key": CP.DEFAULT.key(arm), "robot": RM.robot_digest(), "brace": None, "object": {"category": "hardback", "model": "acbrnv"},
         "grasps": [{"name": "side_0", "base": "side_0", "twin": False}, {"name": "side_0~r180", "base": "side_0", "twin": True}],
         "trunks": [{"translate": TRUNK[0], "tilt": TRUNK[1], "eligible": True, "posture": "untucked", "key": RB.trunk_key(*TRUNK)}]}
    if kind == "interior":
        m.update({"plane": "interior", "interior": sp.to_json(), "lean": {"digest": lean.digest(), "tag": lean.tag(), "board_face_y_m": 0.06}})
    m["key"] = RM.map_key(m)
    return m


def _write(d, meta, feasible=(10, 11)):
    os.makedirs(d, exist_ok=True)
    cfg = RM.MapConfig(**{k: v for k, v in meta["config"].items() if k in RM.MapConfig.__dataclass_fields__})
    cells = RM.lattice(cfg).astype(np.float32)
    nY = cfg.n_yaws
    st = np.full((len(cells), nY, len(meta["grasps"])), RM.STATUS["ik_pre"], dtype=np.uint8)
    for c in feasible:
        st[c, 0, 0] = RM.STATUS["feasible"]
    arrays = {"cells": cells, "yaws": np.array([2 * np.pi * k / nY for k in range(nY)], dtype=np.float32)}
    for d_e in meta["edge_distances"]:
        arrays[f"s{d_e:.2f}_{RB.trunk_key(*TRUNK)}"] = st
    base = os.path.join(d, f"map__hardback__acbrnv__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return base + ".json"


def test_map_key_carries_the_compartment_and_the_lean():
    sp = IN.spec_named(KMUACD)
    lib = _lean_lib()
    m = _meta(lib, sp)
    assert RM.map_key(m) != RM.map_key(dict(m, interior=_spec(width_m=0.35).to_json())) != RM.map_key(dict(m, lean={"digest": "other"}))
    tt = _meta(_stable_lib(), sp, kind="tabletop", rest_id="r0")
    import hashlib

    ident = {k: tt[k] for k in ("schema", "kind", "arm", "object_key", "rest_id", "edge_distances", "config", "policy_key", "robot")}
    ident["brace"], ident["height"] = None, tt["height"]["value"]
    assert RM.map_key(tt) == hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]  # a slab map's key as before
    assert RM.map_plane(m) == "interior"


def test_find_an_interior_map_checks_the_compartment(tmp_path):
    sp = IN.spec_named(KMUACD)
    lib = _lean_lib()
    root = str(tmp_path)
    path = _write(root, _meta(lib, sp))
    ok = dict(rest_id=LEAN.tag(), arm="right", anchors=None, policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest(), kind="interior")
    rm = RM.ReachMap.find(root, lib, sp.height_m, interior=sp, **ok)
    assert rm is not None and rm.path == path and RM.ReachMap.find(root, lib, sp.height_m, interior=sp.digest(), **ok).path == path
    assert RM.ReachMap.find(root, lib, sp.height_m, **ok) is None  # the compartment is required
    assert RM.ReachMap.find(root, lib, sp.height_m, interior=_spec(width_m=0.33), **ok) is None  # another compartment
    assert RM.ReachMap.find(root, lib, sp.height_m, interior=sp, **{**ok, "arm": "left"}) is None
    assert RM.ReachMap.find(root, lib, sp.height_m + 0.01, interior=sp, **ok) is None
    # a tabletop lookup at the same height never returns it (nor any lookup of the lean rest)
    assert RM.ReachMap.find(root, lib, sp.height_m, **{**ok, "kind": "tabletop"}) is None
    # another lean of the same entry identity / an entry whose lean rest is another lean: miss
    other = _lean_lib(spec=LN.LeanSpec(20.0, "left"))
    assert RM.ReachMap.find(root, other, sp.height_m, interior=sp, **{**ok, "rest_id": LN.LeanSpec(20.0, "left").tag()}) is None
    # a stable rest is never looked up in interior maps
    assert RM.ReachMap.find(root, _stable_lib(), sp.height_m, interior=sp, **{**ok, "rest_id": "r0"}) is None
    view = MS.as_bands(rm, overhang=0.0)
    assert view.meta["mode"] == "map_view" and view.meta["stand_offs"] == [0.35, 0.45, 0.55] and view.trunks() == [TRUNK]
    assert view.feasible(*TRUNK, rm.cells[10], stand_off=0.45) == [(0.0, "side_0")]
    assert RM._map_meta_matches(path, "right", "interior", interior=sp.digest()) and not RM._map_meta_matches(path, "right", "interior", interior="x")
