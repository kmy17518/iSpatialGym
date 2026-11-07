"""i7-overhang 2026-11-07 (owner decision: the overhang pick's primary path - hardback bqvvvx, inside_large_01): the sampler side of the v2
OVERHANG rest - the push to the library rest and the flush yaw, the overhang map's init-cell band view, the template's overhang key and
its map lookup (hit, miss = MAP_REQUEST + None, never a legacy band), IORSLarge's geometry / pairs / map-hit step on stubs, the goal-side
lean key of the inside mode. Simulator-free."""

import json
import logging
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import overhang as OV
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.tests import test_overhang_rests as TR

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
H = 0.7671


# ------------------------------------------------------------------------------------------------------------------ the pure helpers


def test_v2_rest_push_and_flush_yaw():
    # the face 0.0861 ahead of the centre, the edge 0.20 ahead: the push leaves the face 0.0294 beyond it
    p = OH.v2_rest_push(0.0294, 0.20, 0.0861)
    assert p == pytest.approx(0.1433) and OH.overhang_m(0.20, p, 0.0861) == pytest.approx(0.0294)
    # the spine (rest-frame -y) onto the push direction -x: yaw -pi/2 (mod 2 pi); onto +y: yaw pi
    y = OH.flush_yaw([0.0, -1.0], (-1.0, 0.0))
    R = C.quat_to_mat(C.quat_from_yaw(y))
    assert np.allclose((R @ th.tensor([0.0, -1.0, 0.0])).numpy()[:2], [-1.0, 0.0], atol=1e-6)
    y2 = OH.flush_yaw([0.0, -1.0], (0.0, 1.0))
    assert np.allclose((C.quat_to_mat(C.quat_from_yaw(y2)) @ th.tensor([0.0, -1.0, 0.0])).numpy()[:2], [0.0, 1.0], atol=1e-6)
    o = OH.options({"mode": "only", "v2_init_push_m": 0.16})
    assert o["v2_init_push_m"] == [0.16] and o["v2_rest"] is None and OH.options(True)["v2_init_push_m"] == [0.12, 0.16, 0.20]
    for bad in ({"v2_init_push_m": [0.5]}, {"v2_init_push_m": []}, {"v2_rest": "lean_left_11_x"}):
        with pytest.raises(ValueError):
            OH.options(bad)


# ------------------------------------------------------------------------------------------------------------------ a map on disk


@pytest.fixture
def v2maps(tmp_path, monkeypatch, fake_world):
    """A v2 root (the overhang entry of test_overhang_rests) + its overhang map at 0.7671 (right arm, fake synthetic world) + anchors."""
    TIM, geom = fake_world
    root = str(tmp_path / "glib")
    lib = TR._ov_lib()
    eid = f"hardback__bqvvvx__s1__{TR.SPEC.tag()}"
    TR._write_entry(root, eid, lib.data)
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", root)
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()
    lib2 = GL.v2_entry(OV.overhang_key(TR.BO_KEY, TR.SPEC))
    r = lib2.rests[TR.SPEC.tag()]
    cfg = RM.overhang_config(r, seeds=2, batch=64, edge_distances=(0.35, 0.45, 0.55), postures=("untucked",), y_half_m=0.16)
    meta, arrays = RM.compute_map(TIM.FakeWorld(cfg), lib2, TR.SPEC.tag(), H, {"key": "0.7671", "height": H}, geom=geom, cfg=cfg, kind="overhang")
    maps = str(tmp_path / "maps")
    path = RM.save(os.path.join(maps, eid), meta, arrays)
    an = str(tmp_path / "anchors.json")
    with open(an, "w") as f:
        json.dump({"schema": "isg-height-anchors/1", "max_approx_m": 0.01, "anchors": [{"key": "0.7671", "height": H, "precision": 0.0001, "surfaces": []}]}, f)
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", maps)
    monkeypatch.setenv("ISPATIALGYM_REACH_MAP_ANCHORS", an)
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "")  # a hit sets it (setdefault): restored after the test, never leaked into the next ones
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY")
    yield SimpleNamespace(root=root, maps=maps, path=path, meta=meta, arrays=arrays, lib=lib2, rest=r, TIM=TIM)
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()


fake_world = TR.fake_world


def test_init_band_view_moves_the_post_cells_back_by_the_pushes(v2maps):
    rm = RM.ReachMap(v2maps.path)
    ea = float(rm.meta["overhang"]["edge_along_m"])
    view = MS.as_overhang_init_bands(rm, [0.12, 0.16], [0.10])
    assert view.meta["mode"] == "map_view" and view.meta["map_kind"] == "overhang" and view.meta["map_path"] == v2maps.path
    assert sorted(view.cells_so) == [0.35, 0.45, 0.55]
    tk = (float(v2maps.TIM.TRUNK[0]), float(v2maps.TIM.TRUNK[1]))
    post, _ = MS.overhang_post_cells(rm, 0.35, *tk)
    post_b, _ = MS.overhang_post_cells(rm, 0.45, *tk)  # B at 0.35 backed off 0.10: the 0.45 row moved 0.10 toward the edge
    cells = view.cells_so[0.35][tk]
    want = {(round(float(c[0]) + p, 4), round(float(c[1]), 4)) for p in (0.12, 0.16) for c in np.concatenate([post, post_b - np.array([0.10, 0.0])])}
    assert {(round(float(c[0]), 4), round(float(c[1]), 4)) for c in cells} == want
    assert all(abs(float(c[0]) - (0.35 + ea + p)) < 1e-6 for c in cells for p in [min((0.12, 0.16), key=lambda q: abs(float(c[0]) - (0.35 + ea + q)))])
    # the 0.55 stand-off has no 0.65 row: its own row only
    assert len(view.cells_so[0.55][tk]) == 2 * len(MS.overhang_post_cells(rm, 0.55, *tk)[0])
    pr = view.pairs_so[0.35][tk]
    assert len(pr) and set(pr[:, 1].tolist()) == {6}  # the map yaw -pi/2 on the band grid (RB.YAWS[6] = 3 pi / 2)
    assert view.band(*tk, stand_off=0.35).shape == cells.shape


# ------------------------------------------------------------------------------------------------------------------ the template's key + its map


class _S:
    def __init__(self, gen=None):
        self.gp = SimpleNamespace(variant="v", gen=lambda k, d=None: (gen or {}).get(k, d))
        self.template_facts = {}
        self.world = SimpleNamespace(floor_z=0.0, scene_model="house_single_floor")
        self.ikc = SimpleNamespace(support_plane_z={})
        self.kit = SimpleNamespace(stats={})


def _top(z=H):
    return SimpleNamespace(z=z, name="desk_bhkhxo_0")


def test_overhang_key_and_its_map(v2maps, monkeypatch, caplog):
    s = _S({"pick_overhang_push": {"mode": "only"}})
    k = TT.overhang_key(s, TR.BO_KEY)
    assert k == OV.overhang_key(TR.BO_KEY, TR.SPEC) and s.template_facts["overhang_rest"]["tag"] == TR.SPEC.tag()
    assert TT.overhang_key(s, "other__key") is None
    ov = TT.overhang_rest_of_key(k)
    assert ov["rest_id"] == TR.SPEC.tag() and ov["normal_xy"] == [0.0, -1.0] and TT.overhang_rest_of_key(TR.BO_KEY) is None
    spec = TT.BandSpec(object="hardback/bqvvvx", grasp_key=k, arm="right")
    assert TT.is_overhang_spec(spec) and not TT.is_lean_spec(spec)
    monkeypatch.setattr(TT.SS, "register_top", lambda *a, **kw: None)
    view = TT.overhang_map_bands(s, spec, _top(), "right")
    assert view is not None and view.meta["map_kind"] == "overhang" and view.meta["rest_id"] == TR.SPEC.tag()
    assert s.ikc.support_plane_z == {"desk_bhkhxo_0": H} and os.environ.get("ISPATIALGYM_SUPPORT_POLICY") == "grasp"
    # the v2 dispatch sends the overhang spec to its map; the left arm has no map: None + MAP_REQUEST (kind overhang)
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.overhang_map_bands(_S(), spec, _top(), "left") is None
    req = [json.loads(r.getMessage().split("MAP_REQUEST ", 1)[1]) for r in caplog.records if "MAP_REQUEST" in r.getMessage()]
    assert req and req[-1]["kind"] == "overhang" and req[-1]["arm"] == "left" and req[-1]["rest_id"] == TR.SPEC.tag()
    monkeypatch.setattr(TT, "surface_band_kwargs", lambda s_: {})
    assert TT.v2_map_view(_S(), spec, _top(), "right").meta["map_kind"] == "overhang"
    # another height: a miss; an interior surface: missing_map
    assert TT.overhang_map_bands(_S(), spec, _top(0.80), "right") is None
    monkeypatch.setattr(TT, "surface_band_kwargs", lambda s_: {"surface_z": 0.94})
    with pytest.raises(Exception, match="overhang rest on an interior surface"):
        TT.v2_map_view(_S(), spec, _top(), "right")
    # no maps root: no overhang key (the legacy path stays as before)
    monkeypatch.delenv("ISPATIALGYM_REACH_MAPS")
    assert TT.overhang_key(_S(), TR.BO_KEY) is None


# ------------------------------------------------------------------------------------------------------------------ IORSLarge on stubs


class _Book(IL.IORSLarge):
    """The hardback on a desk whose edge toward the robot (+x) lies at ``edge_x``: the geometry / pairs / map-hit methods of IORSLarge on
    stubs (no simulator)."""

    def __init__(self, edge_x=1.20, v2=None, arm_bands=None):  # noqa: D107 - no IORSLarge.__init__ (it needs the context)
        self.gp = SimpleNamespace(variant="hb", gen=lambda k, d=None: {"pick_overhang_push": {"mode": "only"}}.get(k, d))
        self.edge_x = edge_x
        self.hull = th.tensor(TR._box_vertices(*TR.BOX), dtype=th.float32)
        self.rest = C.IDENTITY_QUAT.clone()
        self.surface = {"kind": "support", "z": H, "top": SimpleNamespace(contains=lambda xy, m=0.0: float(xy[0]) < self.edge_x, z=H),
                        "obj": SimpleNamespace(name="desk_bhkhxo_0", links={}), "arm_bands": arm_bands or {}}
        self.kit = SimpleNamespace(stats={})
        self.v2_overhang = v2
        self.__dict__["grasps"] = []

    def support_objects(self):
        return [self.surface["obj"]]

    def target_com_local(self):
        return th.zeros(3)

    def pose_on(self, xy, yaw, z):
        return th.tensor([float(xy[0]), float(xy[1]), z + 0.01525 + 0.003]), C.quat_from_yaw(yaw)

    def _overhang_edges(self, centre_xy, d, laterals, z_top):
        u = OH.unit_xy(d)
        return [float((self.edge_x - float(centre_xy[0])) / float(u[0])) if abs(float(u[0])) > 1e-6 else None for _ in laterals]


def _bo_grasp(name="side_0"):
    T = TR.BO_GRASP_T
    return SimpleNamespace(name=name, pose=(th.tensor(T["position"]), th.tensor(T["orientation_xyzw"])), meta={"width": 0.0305})


def _manip_b(base_x=1.60, so=0.40):
    from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

    edge = EdgeFrame("+u", th.tensor([1.20, 0.0]), th.tensor([-1.0, 0.0]), 1.2)
    return Stance("manipulation", (base_x, 0.0, math.pi), 0.5, 0.0, so, edge, None, 0.0, th.tensor([0.95, 0.0, H]), ["right"])


def _v2(dist=TR.D_DEMO):
    return {"tag": TR.SPEC.tag(), "distance_m": TR.D_DEMO, "measured_distance_m": dist, "edge_along_m": 0.0861 - dist, "normal_xy": [0.0, -1.0]}


def test_the_geometry_pushes_to_the_library_rest():
    opts = OH.options({"mode": "only"})
    # the book flush (spine toward +x, the robot), its centre 0.25 m behind the edge: the push leaves the spine 0.0294 beyond it
    yaw = OH.flush_yaw([0.0, -1.0], (1.0, 0.0))
    s = _Book(v2=_v2())
    t_pose = s.pose_on((0.95, 0.0), yaw, H)
    geo, why = s._overhang_geometry(t_pose, _bo_grasp(), (1.0, 0.0), H, opts)
    assert geo is not None, why
    p = geo["ladder_m"][0]
    assert len(geo["ladder_m"]) == 1 and p == pytest.approx(0.25 + TR.D_DEMO - 0.0861, abs=1e-3)
    assert geo["v2_rest"]["tag"] == TR.SPEC.tag() and OH.overhang_m(geo["edge_com_m"], p, geo["extent_front_m"]) == pytest.approx(TR.D_DEMO, abs=1e-3)
    # the legacy ladder for the same pose: its first rung = the fingers' clearance (the v2 rest replaces it)
    legacy = _Book(v2=None)._overhang_geometry(t_pose, _bo_grasp(), (1.0, 0.0), H, opts)[0]
    assert legacy["ladder_m"][0] != pytest.approx(p, abs=1e-4) or len(legacy["ladder_m"]) > 1
    assert geo["v2_rest"]["finger_rule"] == "library rest" and "box_sweep_beyond_edge_m" in geo["v2_rest"]
    # a library rest whose overhang leaves the centre of mass within the margin of the edge: no rung (the COM rule stays)
    geo2, why2 = _Book(v2=_v2(dist=0.06))._overhang_geometry(t_pose, _bo_grasp(), (1.0, 0.0), H, opts)
    assert geo2 is None and "outside the range" in why2 and "centre of mass" in why2
    # the finger-box rule on request: the bo pinch's box needs more than the library rest gives
    sweep = geo["v2_rest"]["box_sweep_beyond_edge_m"]
    o_box = OH.options({"mode": "only", "v2_finger_clearance_m": max(0.0, sweep) + 0.01})
    geo3, why3 = _Book(v2=_v2())._overhang_geometry(t_pose, _bo_grasp(), (1.0, 0.0), H, o_box)
    assert geo3 is None and "finger box" in why3
    assert _Book(v2=_v2())._overhang_geometry(t_pose, _bo_grasp(), (1.0, 0.0), H, OH.options({"mode": "only", "v2_finger_clearance_m": sweep - 0.002}))[0] is not None


def test_pairs_take_the_flush_yaw_and_the_map_hit():
    opts = OH.options({"mode": "only"})
    calls = []
    s = _Book(v2=_v2())
    s.grasps = [_bo_grasp("side_0"), _bo_grasp("side_1")]
    s._v2_overhang_map_ok = lambda st, arm, g, t_pose, u, z, o: (calls.append((arm, g.name)) or (g.name == "side_0", "" if g.name == "side_0" else "not feasible"))
    yaws = [k * 2.0 * math.pi / 16 for k in range(16)]
    F = s._overhang_pairs(_manip_b(), [], th.tensor([0.95, 0.0]), yaws, H, opts, arm="left", st=_manip_b())
    flush = OH.flush_yaw([0.0, -1.0], (1.0, 0.0))
    assert [(round(y, 6), g.name) for y, g in F] == [(round(flush, 6), "side_0")]
    assert calls == [("left", "side_0"), ("left", "side_1")]
    assert s.kit.stats["pick_overhang.v2_map.hit"] == 1 and s.kit.stats["pick_overhang.v2_map.infeasible"] == 1


def test_the_map_hit_step_reads_the_post_push_cell(v2maps):
    rm = RM.ReachMap(v2maps.path)
    view = MS.as_overhang_init_bands(rm, [0.16], [0.10])
    s = _Book(v2=_v2(), arm_bands={"right": view})
    opts = OH.options({"mode": "only"})
    tk = v2maps.TIM.TRUNK
    ea = float(rm.meta["overhang"]["edge_along_m"])
    yaw = OH.flush_yaw([0.0, -1.0], (1.0, 0.0))
    g = _bo_grasp()
    # B 0.45 m from the edge (base x 1.65 facing -x): the post-push root 0.45 + ea ahead - the map's row at 0.45, the cell at y 0
    st = _manip_b(base_x=1.20 + 0.45, so=0.45)
    st.translate, st.tilt = tk
    t_pose = s.pose_on((1.20 - ea - 0.16, 0.0), yaw, H)
    ok, why = s._v2_overhang_map_ok(st, "right", g, t_pose, (1.0, 0.0), H, opts)
    assert ok is True, why
    # the object 0.5 m to the side: outside the map's lateral cells (no verdict)
    ok2, why2 = s._v2_overhang_map_ok(st, "right", g, s.pose_on((1.20 - ea - 0.16, 0.5), yaw, H), (1.0, 0.0), H, opts)
    assert ok2 is None and "outside" in why2
    # B 0.95 m from the edge: no map row from it nor from it backed off
    far = _manip_b(base_x=1.20 + 0.95, so=0.95)
    far.translate, far.tilt = tk
    ok3, why3 = s._v2_overhang_map_ok(far, "right", g, t_pose, (1.0, 0.0), H, opts)
    assert ok3 is None and "not in the map" in why3
    # the left arm has no overhang map view: no verdict
    assert s._v2_overhang_map_ok(st, "left", g, t_pose, (1.0, 0.0), H, opts)[0] is None


# ------------------------------------------------------------------------------------------------------------------ the goal side


def test_inside_goal_key_takes_the_lean_entry_on_the_arms_side(tmp_path, monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_large_inside as ILI
    from ispatialgym.tests import test_interior_maps as TIM

    root = str(tmp_path / "glib")
    for side in ("left", "right"):
        spec = LN.LeanSpec(16.7, side)
        lib = TIM._lean_lib(spec=spec)
        d = dict(lib.data, lean_keys=["hardback__bqvvvx__bu"], model="bqvvvx")
        TR._write_entry(root, f"hardback__bqvvvx__s1__{spec.tag()}", d)
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", root)
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()
    gen = {"inside_goal_lean": {"key": "hardback__bqvvvx__bu", "deg": 16.7}, "inside_shelf_place": {"side": "near"}}
    s = SimpleNamespace(gp=SimpleNamespace(variant="hb", gen=lambda k, d=None: gen.get(k, d)), template_facts={}, partner=SimpleNamespace(name="bookcase_jaysra_0"),
                        kit=SimpleNamespace(stats={}))
    key_l = ILI.InsideMode.inside_goal_key(s, "left")
    key_r = ILI.InsideMode.inside_goal_key(s, "right")
    assert key_l == LN.lean_key("hardback__bqvvvx__bu", LN.LeanSpec(16.7, "left")) and key_r == LN.lean_key("hardback__bqvvvx__bu", LN.LeanSpec(16.7, "right"))
    assert s.template_facts["inside_goal_lean"]["left"]["side"] == "left"
    gen["inside_shelf_place"] = {"side": "right"}
    assert ILI.InsideMode.inside_goal_key(s, "left") == key_r  # an explicit shelf side wins over the arm
    gen["inside_goal_lean"] = {"key": "hardback__bqvvvx__bu", "deg": 20.0}
    with pytest.raises(Exception, match="no v2 lean entry"):
        ILI.InsideMode.inside_goal_key(s, "left")
    gen.pop("inside_goal_lean")
    assert ILI.InsideMode.inside_goal_key(s, "left") is None
    for f in (GL._v2_index, GL._v2_lean_index, GL._v2_overhang_index):
        f.cache_clear()


# ------------------------------------------------------------------------------------------------------------------ owner decisions 2026-11-07


def test_the_v2_overhang_finger_rule_defaults_to_the_library_rest_and_the_switch_restores_the_box_rule():
    """Owner 2026-11-07: on the v2 overhang path the pick goes to the library's physics-validated rest (no finger-box clearance); the
    off-by-default ``v2_finger_clearance_m`` restores the box rule there; the legacy path keeps the box rule (finger_clearance_m)."""
    assert OH.DEFAULTS["v2_finger_clearance_m"] is None and OH.options(True)["v2_finger_clearance_m"] is None
    yaw = OH.flush_yaw([0.0, -1.0], (1.0, 0.0))
    # a pinch 3 cm deeper into the book than the legacy row: its finger box ends over the desk at the library rest
    deep = SimpleNamespace(name="deep", pose=(th.tensor([0.0348, -0.0579, 0.0019]), th.tensor(TR.BO_GRASP_T["orientation_xyzw"])), meta={"width": 0.0305})
    s = _Book(v2=_v2())
    t_pose = s.pose_on((0.95, 0.0), yaw, H)
    geo, why = s._overhang_geometry(t_pose, deep, (1.0, 0.0), H, OH.options({"mode": "only"}))
    assert geo is not None, why
    assert geo["v2_rest"]["finger_rule"] == "library rest" and geo["v2_rest"]["box_sweep_beyond_edge_m"] < 0  # the box over the desk: still the rest
    assert geo["ladder_m"] == [geo["v2_rest"]["push_m"]]
    geo2, why2 = s._overhang_geometry(t_pose, deep, (1.0, 0.0), H, OH.options({"mode": "only", "v2_finger_clearance_m": 0.01}))
    assert geo2 is None and "finger box 0.010" in why2  # the switch: the box rule again
    # the legacy path (no v2 rest): the box rule's ladder, its first rung the fingers' 1 cm clearance
    legacy = _Book(v2=None)._overhang_geometry(t_pose, deep, (1.0, 0.0), H, OH.options({"mode": "only"}))
    assert legacy[0] is None or "v2_rest" not in legacy[0]
