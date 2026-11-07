"""The support split (i7-split 2026-11-07, ``ispatialgym.sampling.v2.support_split``): the tolerance from a top raster, the piece
classification by height around the registered plane, ``IKChecker.ignoring(objs, split=True)`` switching off only the support pieces of a
registered support, the approached exclusion, the cache-key identity, the straddler clip and the draw's protrusion count - all on fake
meshes / fake checkers (no simulator)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import support_split as SS
from ispatialgym.sampling.v2.ik import IKChecker

ROOT = (1.0, 2.0, 0.80)  # the tray's root on a table top at 0.80
FLOOR_TOP = 0.006  # the tray's inner floor above its root origin
RIM_TOP = 0.021  # the rims: +15 mm above the floor


def _box(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float64)


def _wedge() -> np.ndarray:
    """A sloped piece by the +x wall: its top runs from the floor (x 0.12) up to the rim (x 0.19) - it reaches down onto the floor cells."""
    return np.array([[0.12, -0.1, 0.0], [0.12, 0.1, 0.0], [0.19, -0.1, 0.0], [0.19, 0.1, 0.0],
                     [0.12, -0.1, FLOOR_TOP], [0.12, 0.1, FLOOR_TOP], [0.19, -0.1, RIM_TOP], [0.19, 0.1, RIM_TOP]], dtype=np.float64)


def _faces(v: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    return np.asarray(ConvexHull(v).simplices, dtype=np.int64)


class _CM:
    def __init__(self, path: str, verts: np.ndarray, root):
        self.prim_path, self.verts, self.root = path, np.asarray(verts, dtype=np.float32), root

    def get_position_orientation(self):
        return th.tensor(self.root[0], dtype=th.float32), th.tensor(self.root[1], dtype=th.float32)


class _Obj:
    """A spawned object with one link whose collision pieces are given in its root frame (the piece prims sit at the root pose)."""

    def __init__(self, name: str, pieces, root=ROOT, quat=(0.0, 0.0, 0.0, 1.0), model="uekqey", aabb=None):
        self.name, self.model, self.scale = name, model, th.ones(3)
        self.prim_path = f"/World/scene_0/{name}"
        self._root = (tuple(root), tuple(quat))
        self.links = {"base_link": SimpleNamespace(collision_meshes={k: _CM(f"{self.prim_path}/base_link/collisions/{k}", v, self._root)
                                                                     for k, v in pieces.items()})}
        self.aabb = aabb if aabb is not None else (th.tensor([root[0] - 0.2, root[1] - 0.15, root[2]]), th.tensor([root[0] + 0.2, root[1] + 0.15, root[2] + RIM_TOP]))

    def get_position_orientation(self):
        return th.tensor(self._root[0], dtype=th.float32), th.tensor(self._root[1], dtype=th.float32)

    def path(self, piece: str) -> str:
        return f"{self.prim_path}/base_link/collisions/{piece}"


def _tray(name="tray_source", wedge=False, **kw) -> _Obj:
    pieces = {"floor": _box((-0.2, -0.15, 0.0), (0.2, 0.15, FLOOR_TOP)),
              "rim_px": _box((0.195, -0.15, 0.0), (0.2, 0.15, RIM_TOP)), "rim_nx": _box((-0.2, -0.15, 0.0), (-0.195, 0.15, RIM_TOP)),
              "rim_py": _box((-0.2, 0.145, 0.0), (0.2, 0.15, RIM_TOP)), "rim_ny": _box((-0.2, -0.15, 0.0), (0.2, -0.145, RIM_TOP))}
    if wedge:
        pieces["wedge"] = _wedge()
    return _Obj(name, pieces, **kw)


def _floor_cells() -> np.ndarray:
    xs, ys = np.arange(-0.18, 0.181, 0.02), np.arange(-0.13, 0.131, 0.02)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)


class _Meshes:
    def __init__(self):
        self.added, self.enabled, self.poses = {}, {}, {}

    def add(self, mesh):
        self.added[mesh.name] = mesh
        self.enabled[mesh.name] = True

    def update_pose(self, name, w_obj_pose=None):
        self.poses[name] = w_obj_pose

    def set_enabled(self, name, enabled):
        self.enabled[name] = bool(enabled)


class _Backend:
    def __init__(self, objs):
        self.enabled = {cm.prim_path: True for o in objs for l in o.links.values() for cm in l.collision_meshes.values()}
        self._world_mesh_names = set(self.enabled)
        self._disabled_obstacles = set()
        self.checker = SimpleNamespace(data=SimpleNamespace(meshes=_Meshes()))
        self.device_cfg = SimpleNamespace(device="cpu")

    def _mesh_enable(self, name, enabled):
        self.enabled[name] = bool(enabled)

    def _mesh_geometry(self, cm):
        return cm.verts, _faces(cm.verts), np.ones(3, dtype=np.float32)

    def root_pose(self):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def off(self):
        return {n for n, e in self.enabled.items() if not e}


def _kit(objs, spawned=None):
    be = _Backend(objs)
    robot = SimpleNamespace(arm_control_idx={"right": th.arange(7, 14), "left": th.arange(14, 21)}, gripper_control_idx={"right": th.tensor([21, 22])})
    world = SimpleNamespace(spawned={o.name: o for o in (spawned if spawned is not None else objs)})
    return SimpleNamespace(backend=be, _ensure_synced=lambda: None, stats={}, robot=robot, world=world)


def _register_tray(ikc, tray, heights=None, cells=True):
    tol = SS.tolerance_from_heights(np.zeros(60) if heights is None else heights, 0.0)
    return SS.register(ikc, tray, ROOT[2] + FLOOR_TOP, tol, dz=FLOOR_TOP, cells_local=_floor_cells() if cells else None, what="tray_floor")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(SS.SWITCH, raising=False)
    monkeypatch.delenv(SS.CLIP_SWITCH, raising=False)
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_POLICY", raising=False)
    SS._CLASS_CACHE.clear()
    SS._CLIP_CACHE.clear()
    SS._ACTIVE_CLIPS.clear()
    IK._ACTIVE_IGNORES.clear()


# ----------------------------------------------------------------------------------------------------------------
# tolerance
# ----------------------------------------------------------------------------------------------------------------


def test_tolerance_flat_top_is_the_minimum():
    t = SS.tolerance_from_heights(0.80 + np.random.default_rng(0).uniform(-0.0004, 0.0004, 200), 0.80)
    assert t.tol == pytest.approx(SS.MIN_TOL_M) and t.why == "" and t.n == 200


def test_tolerance_is_q95_plus_2mm_clamped():
    zs = np.concatenate([np.zeros(80), np.full(20, -0.005)])  # q95 of |d| = 5 mm
    t = SS.tolerance_from_heights(zs, 0.0)
    assert t.spread == pytest.approx(0.005) and t.tol == pytest.approx(0.007)
    t2 = SS.tolerance_from_heights(np.concatenate([np.zeros(50), np.full(50, -0.0095)]), 0.0)
    assert t2.tol == pytest.approx(SS.MAX_TOL_M)  # 9.5 + 2 mm clamped to 10 mm


def test_rim_cells_above_the_cap_are_not_spread():
    """A tray / bed raster's rim or rail cells (+15 mm, inside SupportTop's +-3 cm mask) are protrusions under any tolerance: they do not
    turn the support 'nonflat' (the spread over every cell is logged)."""
    zs = np.concatenate([np.zeros(70), np.full(30, 0.015)])
    t = SS.tolerance_from_heights(zs, 0.0)
    assert t.tol == pytest.approx(SS.MIN_TOL_M) and t.spread_all == pytest.approx(0.015)


def test_nonflat_top_keeps_the_whole_ignore():
    zs = np.concatenate([np.zeros(60), np.linspace(-0.03, -0.012, 40)])  # a drooping mattress
    t = SS.tolerance_from_heights(zs, 0.0)
    assert t.tol is None and t.why == "nonflat" and t.spread > SS.MAX_TOL_M


def test_tolerance_without_cells():
    assert SS.tolerance_from_heights(np.full(10, np.nan), 0.0).why == "no_raster"
    assert SS.tolerance_from_heights(np.zeros(3), 0.0).why == "few_cells"


def test_tolerance_from_a_support_top_raster():
    from ispatialgym.sampling.sim_utils import SupportTop

    xs, ys = th.arange(0.01, 0.2, 0.02), th.arange(0.01, 0.2, 0.02)
    zs = th.full((len(xs), len(ys)), 0.75)
    zs[0, :] = 0.766  # a lip along one side (+16 mm)
    mask = th.ones(len(xs), len(ys), dtype=th.bool)
    top = SupportTop("desk_0", xs, ys, mask, th.zeros_like(mask), 0.75, 0.0, zs=zs)
    t = SS.tolerance_from_top(top)
    assert t.tol == pytest.approx(SS.MIN_TOL_M) and t.n == 90
    cells = SS.plane_cells_world(top, 0.75, t.tol)
    assert len(cells) == 90 and float(cells[:, 0].min()) > 0.02  # the lip row is not a plane cell
    assert SS.tolerance_from_top(SupportTop("desk_0", xs, ys, mask, th.zeros_like(mask), 0.75, 0.0)).why == "no_raster"


# ----------------------------------------------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------------------------------------------


def test_classify_tray_floor_vs_rims():
    tray = _tray()
    pieces = [(k, cm.verts) for k, cm in tray.links["base_link"].collision_meshes.items()]
    c = SS.classify(pieces, FLOOR_TOP, 0.003, _floor_cells())
    assert c.support == ("floor",) and set(c.protrusion) == {"rim_px", "rim_nx", "rim_py", "rim_ny"} and c.straddling == ()
    assert set(c.tops_mm.values()) == {15.0} and len(c.hulls) == 4


def test_classify_pieces_below_the_plane_are_support():
    """Legs / an apron under a table top (tops below the plane) go off with the top, as before."""
    pieces = [("top", _box((0, 0, 0.70), (1, 1, 0.75))), ("leg", _box((0, 0, 0), (0.05, 0.05, 0.70))),
              ("back_panel", _box((0, 0.95, 0.70), (1, 1, 0.90)))]
    c = SS.classify(pieces, 0.75, 0.003)
    assert set(c.support) == {"top", "leg"} and c.protrusion == ("back_panel",)


def test_classify_sloped_piece_over_floor_cells_straddles():
    c = SS.classify([("wedge", _wedge()), ("rim", _box((0.195, -0.15, 0.0), (0.2, 0.15, RIM_TOP)))], FLOOR_TOP, 0.003, _floor_cells())
    assert c.straddling == ("wedge",) and c.protrusion == ("rim",)
    # without the raster cells there is no straddler test: a protrusion
    assert SS.classify([("wedge", _wedge())], FLOOR_TOP, 0.003).protrusion == ("wedge",)


def test_hull_helpers():
    h = SS.hull2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]))
    assert len(h) == 4
    assert SS.inside_hull(h, np.array([[0.5, 0.5], [1.5, 0.5], [0.999, 0.5]]), 0.002).tolist() == [True, False, False]


# ----------------------------------------------------------------------------------------------------------------
# IKChecker.ignoring(objs, split=True)
# ----------------------------------------------------------------------------------------------------------------


def test_ignoring_switches_only_the_support_pieces_of_a_registered_support():
    tray, target = _tray(), _Obj("target", {"body": _box((-0.02, -0.02, FLOOR_TOP), (0.02, 0.02, 0.08))}, model="mug")
    kit = _kit([tray, target])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    assert ikc.support_plane_tol[tray.name] == pytest.approx(SS.MIN_TOL_M) and ikc.support_split_z[tray.name] == pytest.approx(ROOT[2] + FLOOR_TOP)
    with ikc.ignoring([tray, target]):
        assert kit.backend.off() == {tray.path("floor"), target.path("body")}  # the rims stay obstacles; the target goes whole
    assert kit.backend.off() == set()
    assert kit.stats["support_split.applied"] == 1 and kit.stats["support_split.protrusion_pieces"] == 4
    with ikc.ignoring([tray], split=False):
        assert kit.backend.off() == set(kit.backend.enabled)  - {target.path("body")}
    assert kit.backend.off() == set()


def test_split_switch_off_is_the_whole_ignore(monkeypatch):
    tray = _tray()
    kit = _kit([tray])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    monkeypatch.setenv(SS.SWITCH, "0")
    with ikc.ignoring([tray]):
        assert kit.backend.off() == set(kit.backend.enabled)


def test_unregistered_or_nonflat_supports_go_whole():
    tray, wall = _tray(), _Obj("wall_3", {"w": _box((0, 0, 0), (0.1, 2, 2.5))}, model="wall")
    kit = _kit([tray, wall])
    ikc = IKChecker(kit)
    SS.register(ikc, tray, ROOT[2] + FLOOR_TOP, SS.tolerance_from_heights(np.linspace(-0.03, 0.0, 50), 0.0), dz=FLOOR_TOP)
    assert ikc.support_plane_tol[tray.name] is None
    with ikc.ignoring([tray, wall]):
        assert kit.backend.off() == set(kit.backend.enabled)
    assert kit.stats["support_split.whole_ignore_nonflat"] == 1


def test_nested_contexts_keep_the_refcount():
    tray = _tray()
    kit = _kit([tray])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    with ikc.ignoring([tray]):
        with ikc.ignoring([tray], split=False):
            assert kit.backend.off() == set(kit.backend.enabled)
        assert kit.backend.off() == {tray.path("floor")}
    assert kit.backend.off() == set()


def test_the_plane_is_in_the_tray_frame():
    """A tray moved (another focus, another table height): the same pieces split - the plane rides with its root."""
    for root in ((1.0, 2.0, 0.80), (3.0, -1.0, 0.62)):
        tray = _tray(root=root, quat=(0.0, 0.0, 0.3826834, 0.9238795))
        kit = _kit([tray])
        ikc = IKChecker(kit)
        _register_tray(ikc, tray)
        with ikc.ignoring([tray]):
            assert kit.backend.off() == {tray.path("floor")}


def test_a_plane_no_piece_reaches_keeps_the_whole_ignore():
    """A floor datum 6 mm below the floor mesh (or a top registered below its top piece): no piece's top lies at the plane - the split
    would keep the support itself in the world; the whole ignore instead."""
    floor = _Obj("floors_0", {"slab": _box((-5, -5, -0.05), (5, 5, 0.006)), "step": _box((2, 2, 0.0), (3, 3, 0.15))}, root=(0.0, 0.0, 0.0), model="floor")
    kit = _kit([floor])
    ikc = IKChecker(kit)
    SS.register(ikc, floor, 0.0, SS.Tolerance.fixed(SS.MIN_TOL_M, "floor"), what="floor")
    with ikc.ignoring([floor]):
        assert kit.backend.off() == set(kit.backend.enabled)
    assert kit.stats["support_split.whole_ignore_no_plane_piece"] == 1
    SS.register(ikc, floor, 0.006, SS.Tolerance.fixed(SS.MIN_TOL_M, "floor"), what="floor")
    with ikc.ignoring([floor]):
        assert kit.backend.off() == {floor.path("slab")}  # at its datum the step stays an obstacle


def test_tilted_support_goes_whole():
    tray = _tray(quat=(0.2588190, 0.0, 0.0, 0.9659258))  # 30 deg about x
    kit = _kit([tray])
    ikc = IKChecker(kit)
    SS.register(ikc, tray, ROOT[2] + FLOOR_TOP, SS.Tolerance(0.003))
    assert ikc.support_plane_tol[tray.name] is None  # a world plane registered on a tilted object
    SS.register(ikc, tray, ROOT[2] + FLOOR_TOP, SS.Tolerance(0.003), dz=FLOOR_TOP)  # an object-frame plane: checked at use
    assert ikc.support_plane_tol[tray.name] == pytest.approx(0.003)
    with ikc.ignoring([tray]):
        assert kit.backend.off() == set(kit.backend.enabled)
    assert kit.stats["support_split.whole_ignore_tilted"] == 1


def test_register_top_derives_the_plane_above_the_root():
    tray = _tray()
    ikc = IKChecker(_kit([tray]))
    xs, ys = th.arange(0.85, 1.15, 0.02), th.arange(1.9, 2.1, 0.02)
    zs = th.full((len(xs), len(ys)), ROOT[2] + FLOOR_TOP)
    top = SimpleNamespace(name=tray.name, z=ROOT[2] + FLOOR_TOP, xs=xs, ys=ys, mask=th.ones(len(xs), len(ys), dtype=th.bool), zs=zs)
    rec = SS.register_top(ikc, tray, top)
    assert rec.dz == pytest.approx(FLOOR_TOP, abs=1e-6) and rec.cells is not None and len(rec.cells) == len(xs) * len(ys)
    assert float(np.abs(rec.cells).max()) < 0.2  # in the tray's root frame
    assert SS.register_top(ikc, tray, top) is rec  # the same registration again: kept


# ----------------------------------------------------------------------------------------------------------------
# straddlers: the clipped part above the plane + tol
# ----------------------------------------------------------------------------------------------------------------


def _fake_curobo(monkeypatch):
    class _Pose:
        def __init__(self, position=None, quaternion=None):
            self.position, self.quaternion = position, quaternion

    class _Mesh:
        def __init__(self, name, pose, vertices, faces, scale):
            self.name, self.pose, self.vertices, self.faces, self.scale = name, pose, vertices, faces, scale

    cu, types, scene = ModuleType("curobo"), ModuleType("curobo.types"), ModuleType("curobo.scene")
    types.Pose, scene.Mesh = _Pose, _Mesh
    for name, mod in (("curobo", cu), ("curobo.types", types), ("curobo.scene", scene)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_straddler_goes_off_and_its_upper_part_is_added(monkeypatch):
    _fake_curobo(monkeypatch)
    tray = _tray(wedge=True)
    kit = _kit([tray])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    clip = tray.path("wedge") + SS.CLIP_SUFFIX
    meshes = kit.backend.checker.data.meshes
    with ikc.ignoring([tray]):
        assert kit.backend.off() == {tray.path("floor"), tray.path("wedge")}
        assert meshes.enabled[clip] is True
        v = np.asarray(meshes.added[clip].vertices, dtype=np.float64)
        assert float(v[:, 2].min()) >= FLOOR_TOP + 0.003 - 1e-6 and float(v[:, 2].max()) == pytest.approx(RIM_TOP, abs=1e-6)
        pose = meshes.added[clip].pose
        assert pose[:3] == pytest.approx(list(ROOT)) and pose[3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert meshes.enabled[clip] is False and kit.backend.off() == set()
    assert kit.stats["support_split.straddling_pieces"] == 1 and kit.stats["support_split.clipped"] == 1
    with ikc.ignoring([tray]):  # the second time: re-posed and switched on, not added again
        assert meshes.enabled[clip] is True and clip in meshes.poses


def test_straddler_without_the_clip_goes_off_whole(monkeypatch):
    monkeypatch.setenv(SS.CLIP_SWITCH, "0")
    tray = _tray(wedge=True)
    kit = _kit([tray])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    with ikc.ignoring([tray]):
        assert kit.backend.off() == {tray.path("floor"), tray.path("wedge")}
    assert kit.backend.checker.data.meshes.added == {}


# ----------------------------------------------------------------------------------------------------------------
# approached exclusion, cache keys
# ----------------------------------------------------------------------------------------------------------------


def test_approached_objects_exclude_registered_supports(monkeypatch):
    tray = _tray()
    target = _Obj("target", {"body": _box((-0.02, -0.02, FLOOR_TOP), (0.02, 0.02, 0.08))}, model="mug",
                  aabb=(th.tensor([0.98, 1.98, 0.806]), th.tensor([1.02, 2.02, 0.88])))
    kit = _kit([tray, target])
    ikc = IKChecker(kit)
    end = (th.tensor([1.0, 2.0, 0.85]), th.tensor([1.0, 0.0, 0.0, 0.0]))  # top-down: the fingertip inside the tray's AABB + 3 cm and the target's
    assert {o.name for o in ikc.approached_objects(end)} == {"tray_source", "target"}  # not registered yet: the former behaviour
    _register_tray(ikc, tray)
    assert [o.name for o in ikc.approached_objects(end)] == ["target"]
    assert kit.stats["support_split.approached_excluded"] == 1
    ikc.support_plane_z = {"other_tray": 0.8}
    monkeypatch.setenv(SS.SWITCH, "0")
    assert {o.name for o in ikc.approached_objects(end)} == {"tray_source", "target"}


def test_policy_registered_supports_are_excluded_too():
    tray = _tray()
    kit = _kit([tray])
    ikc = IKChecker(kit)
    ikc.support_plane_z = {tray.name: ROOT[2] + FLOOR_TOP}
    assert SS.drop_registered(ikc, [tray]) == []


def test_cache_identity_names_the_split_supports(monkeypatch):
    tray, flat = _tray(), _Obj("desk_0", {"top": _box((-0.5, -0.5, -0.03), (0.5, 0.5, 0.0))}, root=(0.0, 0.0, 0.75), model="desk")
    kit = _kit([tray, flat])
    ikc = IKChecker(kit)
    assert SS.cache_identity(ikc, [tray, flat]) == {}
    _register_tray(ikc, tray)
    SS.register(ikc, flat, 0.75, SS.Tolerance(0.003))
    ident = SS.cache_identity(ikc, [tray, flat])
    assert list(ident) == ["support_split"] and ident["support_split"][0] == SS.VERSION
    assert [r[0] for r in ident["support_split"][1]] == [tray.name]  # a flat support splits into nothing: the key as before
    monkeypatch.setenv(SS.SWITCH, "0")
    assert SS.cache_identity(ikc, [tray]) == {}


def test_exact_and_feasible_set_keys_change_with_the_split(monkeypatch):
    tray = _tray()
    kit = _kit([tray])
    ikc = IKChecker(kit)
    seen = []

    class _Stop(Exception):
        pass

    def key(ns, payload):
        seen.append((ns, payload))
        return f"{ns}:{len(seen)}"

    def get(k):
        if k.startswith("exact"):
            raise _Stop()
        return {}

    monkeypatch.setattr(IK, "cache_key", key)
    monkeypatch.setattr(IK, "cache_get", get)
    st = SimpleNamespace(base_pose2d=(0.0, 0.0, 0.0), translate=0.5, tilt=0.0)
    pose = (th.tensor([1.0, 2.0, 0.9]), th.tensor([1.0, 0.0, 0.0, 0.0]))
    grasp = SimpleNamespace(key="k", name="g0", pose=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))

    def both():
        seen.clear()
        with pytest.raises(_Stop):
            ikc.exact([pose], st, "right", ignore_for_last=[tray], q_seed=th.zeros(28))
        ikc.feasible_sets([th.zeros(2)], [0.0], [grasp], lambda c, y: pose, st, "right", ignore_for_grasp=[tray])
        return {ns: p for ns, p in seen}

    before = both()
    _register_tray(ikc, tray)
    after = both()
    for ns in ("exact_ik_v3", "feasible_sets_v2"):
        assert "support_split" not in before[ns] and after[ns]["support_split"][1][0][0] == tray.name


# ----------------------------------------------------------------------------------------------------------------
# the draw's protrusion count, the task wiring
# ----------------------------------------------------------------------------------------------------------------


def test_protrusion_hit_counts_fingers_in_a_rim():
    tray = _tray()
    kit = _kit([tray])
    ikc = IKChecker(kit)
    _register_tray(ikc, tray)
    s = SimpleNamespace(ikc=ikc)
    inside_rim = np.array([[ROOT[0] + 0.197, ROOT[1], ROOT[2] + 0.012]])
    assert SS.protrusion_hit(ikc, [tray], inside_rim) == "tray_source/rim_px"
    assert SS.protrusion_hit(ikc, [tray], inside_rim + np.array([0.0, 0.0, 0.05])) is None  # above the rim
    assert SS.protrusion_hit(ikc, [tray], np.array([[ROOT[0], ROOT[1], ROOT[2] + 0.012]])) is None  # over the floor
    # a top-down pinch closing along x 5 cm in from the +x rim: the fully open finger lands in the rim, a 2 cm opening clears it
    from ispatialgym.robot import conventions as C

    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))
    eef = (th.tensor([ROOT[0] + 0.197 - C.FINGER_OPEN, ROOT[1], ROOT[2] + 0.002]), q)
    pts = SS.finger_points(eef)
    assert float(pts[:, 2].max()) < ROOT[2] + RIM_TOP and float(pts[:, 0].max()) == pytest.approx(ROOT[0] + 0.197, abs=1e-5)
    assert SS.count_protrusion_hit(s, eef, [tray]) == "tray_source/rim_px" and kit.stats["support_split.protrusion_hits"] == 1
    assert SS.count_protrusion_hit(s, eef, [tray], opening=0.02) is None and kit.stats["support_split.protrusion_hits"] == 1
