"""Simulator-free tests of the in-process band kit (owner 2026-10-12 item 2a: the bands the sampler computes itself use the CLI's
``--mesh-cache auto`` store, scoped to band computation): ``reach_bands.band_kit`` builds a kit of its own with a snug store and
leaves the sampler's kit alone, ``ISPATIALGYM_BAND_MESH_CACHE`` switches it, and ``tabletop.compute_bands``,
``IORSLarge.compute_band`` and ``ArticulationSmall._brace_band`` compute on it and record it in ``meta.schedule``;
``ISPATIALGYM_BAND_TRUNK_CHECK=idle-arm`` gives the first two the CLI's ``--trunk-check idle-arm`` (key, file name, meta)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import tabletop as TT

ENV = RB.BAND_MESH_CACHE_ENV
REAL_BAND_KIT = RB.band_kit


class _Obj:
    def __init__(self, name, n_meshes, visual_only=False):
        self.name, self.visual_only = name, visual_only
        self.links = {"base_link": NS(collision_meshes={f"m{i}": NS(prim_path=f"/World/{name}/base_link/m{i}") for i in range(n_meshes)})}


class _World:
    """Objects with collision meshes; ``far_*`` objects fail the obstacle filter (never loaded)."""

    def __init__(self, objs):
        self.robot = _Obj("robot_r1", 40)
        self.scene = NS(objects=[self.robot] + list(objs))
        self.scene_model, self.rooms = "house_single_floor", ["kitchen_0"]
        self.spawned, self.log = {}, []

    def obstacle_filter(self, o):
        return not o.name.startswith("far_")

    def obj(self, name):
        return NS(name=name)

    def snapshot(self):
        return "snap"

    def park(self, obj, i):
        self.log.append(("park", obj.name, i))

    def restore(self, snap):
        self.log.append(("restore", snap))


class _Kit:
    """A ``RobotKit`` stand-in: the mesh store holds every mesh a sync ever loaded (as cuRobo's keeps removed objects' meshes)."""

    def __init__(self, world, prebuild=("right_arm",), mesh_cache_size=None):
        self.world, self.prebuild, self.floor_z = world, tuple(prebuild), 0.0
        self.calls = []
        self.backend = NS(mesh_cache_size=int(mesh_cache_size or 8192), _world_mesh_names=set(), _entries={},
                          _batch_ik={("right_arm", 128, 12): NS(rsc=NS(reset_seed=lambda: self.calls.append("reset_seed")))})

    def sync_world(self):
        self.calls.append("sync")
        self.backend._world_mesh_names |= RB._collision_mesh_names(self.world)

    def q_config(self, **kw):
        return th.zeros(28)

    def q_at_base_pose(self, q, x, y, yaw):
        return q

    def collision_free(self, q, margin=0.0):
        self.calls.append("collision_free")
        return th.tensor([False])

    def trav_free(self, xy, **kw):
        return True


def _setup(n_table=300, n_chair=256):
    """A world with 556 loadable collision meshes (the kitchen of house_single_floor) and the sampler's kit (cuRobo's 8192)."""
    world = _World([_Obj("table_0", n_table), _Obj("chair_0", n_chair), _Obj("lamp_0", 9, visual_only=True), _Obj("far_sofa_0", 50)])
    kit = _Kit(world, mesh_cache_size=8192)
    kit.floor_z = 0.0123
    return world, kit


def test_snug_store_is_the_cli_formula():
    assert RB.snug_mesh_cache(0) == RB.MESH_CACHE_MIN == 512
    assert RB.snug_mesh_cache(256) == 512 and RB.snug_mesh_cache(257) == 576
    assert RB.snug_mesh_cache(556) == 832  # house_single_floor's kitchen: 556 meshes -> 832 slots instead of 8192
    assert all(RB.snug_mesh_cache(n) % 64 == 0 and RB.snug_mesh_cache(n) >= n + RB.MESH_CACHE_HEADROOM for n in range(0, 3000, 37))


def test_band_kit_is_its_own_kit_with_a_snug_store_and_the_sampler_kit_is_untouched(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    world, kit = _setup()
    assert len(RB._collision_mesh_names(world)) == 556  # robot, visual-only and filtered objects excluded
    bk = RB.band_kit(world, kit, kit_cls=_Kit)
    assert bk is not kit and kit._band_kit is bk and bk.world is world
    assert bk.backend.mesh_cache_size == 832 and bk.prebuild == ()  # --mesh-cache auto, no planners (as reach_bands.load_world)
    assert bk.calls == ["sync"] and bk.floor_z == kit.floor_z
    assert kit.calls == [] and kit.backend.mesh_cache_size == 8192 and kit.backend._world_mesh_names == set()
    # the next band: the same kit, synced again, the sampler kit's re-measured floor, fresh solver seed streams
    world.scene.objects.append(_Obj("target_lid_0", 5))
    kit.floor_z = 0.02
    assert RB.band_kit(world, kit, kit_cls=_Kit) is bk
    assert bk.calls == ["sync", "sync", "reset_seed"] and bk.floor_z == 0.02 and kit.calls == []
    assert RB.band_schedule(bk, kit) == {"mesh_cache": 832, "kit": "band"}
    assert RB.band_schedule(kit, kit) == {"mesh_cache": 8192, "kit": "sampler"}


def test_band_kit_is_rebuilt_before_its_store_would_overflow(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    world, kit = _setup()
    bk = RB.band_kit(world, kit, kit_cls=_Kit)
    # removed objects' meshes stay in the store: 556 loaded + 250 new + 64 reserve > 832 slots -> a new kit sized to the world now
    world.scene.objects[:] = [o for o in world.scene.objects if o.name != "chair_0"] + [_Obj("partner_pot_0", 250), _Obj("target_lid_0", 50)]
    bk2 = RB.band_kit(world, kit, kit_cls=_Kit)
    assert bk2 is not bk and kit._band_kit is bk2 and bk2.backend.mesh_cache_size == RB.snug_mesh_cache(300 + 250 + 50)
    # another world behind the same sampler kit (never happens with WorldPool, which builds a kit per world): rebuilt as well
    other = _World([_Obj("desk_0", 30)])
    assert RB.band_kit(other, kit, kit_cls=_Kit).world is other


def test_band_kit_switch(monkeypatch):
    world, kit = _setup()
    for off in ("off", "OFF", "0", "sampler"):
        monkeypatch.setenv(ENV, off)
        assert RB.band_kit(world, kit, kit_cls=_Kit) is kit and getattr(kit, "_band_kit", None) is None
    monkeypatch.setenv(ENV, "1024")
    assert RB.band_kit(world, kit, kit_cls=_Kit).backend.mesh_cache_size == 1024
    world, kit = _setup()
    warnings = []
    monkeypatch.setattr(RB.log, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setenv(ENV, "100")  # fewer slots than meshes: the snug store, with a warning
    assert RB.band_kit(world, kit, kit_cls=_Kit).backend.mesh_cache_size == 832
    assert len(warnings) == 1 and "too small" in warnings[0]
    monkeypatch.setenv(ENV, "lots")
    with pytest.raises(ValueError, match=ENV):
        RB.band_kit(world, kit, kit_cls=_Kit)
    # a sampler kit whose store is already snug (ISPATIALGYM_SAMPLER_MESH_CACHE=auto) keeps its bands: no second kit
    monkeypatch.delenv(ENV, raising=False)
    world, kit = _setup()
    kit.backend.mesh_cache_size = 832
    assert RB.band_kit(world, kit, kit_cls=_Kit) is kit


class _BC:
    """``BandComputer`` stand-in: records the kit it is built on, returns one trunk with two cells."""

    kits = []
    lattice = RB.LATTICE

    def __init__(self, world, kit, arms=("right",), lattice=RB.LATTICE):
        _BC.kits.append(kit)
        self.kit = kit

    def stance_at_edge(self, top, support, stand_off, edge_name=None):
        self.kit.collision_free(th.zeros(28))
        return {"edge": "+u", "base_pose2d": [6.1, -1.01, 1.57], "edge_mid": [6.1, -0.56], "inward": [0.0, 1.0]}

    def stance_free_floor(self, room):
        self.kit.collision_free(th.zeros(28))
        return {"edge": None, "base_pose2d": [1.0, 2.0, 0.0], "edge_mid": [1.0, 2.0], "inward": [1.0, 0.0]}

    def base_frame(self, stance):
        return th.tensor(stance["base_pose2d"][:2]), th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0])

    def lattice_on_cells(self, stance, world_cells):
        return th.zeros(3, 2), th.zeros(3, 2)

    def compute(self, stance, lattice_world, z_support, pose_fn, grasp_sets, ignore, arm="right", trunks=None, max_grasp_offset=0.0):
        self.kit.calls.append("compute")
        return {"trunks": [{"translate": 0.5, "tilt": 0.0, "collides": False, "n_reachable": 2}],
                "cells": {(0.5, 0.0): np.array([[0.4, 0.1], [0.48, 0.1]], dtype=np.float32)},
                "pairs": {(0.5, 0.0): np.array([[0, -1, 0], [1, -1, 1]], dtype=np.int32)}, "classes": {(0.5, 0.0): np.array([1, 2])}}


@pytest.fixture
def band_kit_of_fakes(monkeypatch, tmp_path):
    """``reach_bands.band_kit`` with the stand-in kit class, ``BandComputer`` recorded, the band cache in ``tmp_path``."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("ISPATIALGYM_REACH_BANDS", str(tmp_path))
    monkeypatch.setattr(RB, "band_kit", lambda world, kit: REAL_BAND_KIT(world, kit, kit_cls=_Kit))
    monkeypatch.setattr(RB, "BandComputer", _BC)
    _BC.kits = []
    return tmp_path


def _compute_tabletop(world, kit, out_dir):
    top = NS(name="countertop_kelzer_0", z=0.89, cells=lambda free_only=True: th.zeros(4, 2), to_json=lambda: {"name": "countertop_kelzer_0"})
    spec = TT.BandSpec(object="stockpot/oshwps", grasp_key="attach:lid__pbmsji__a", generic_pinch=True, height=0.2)
    return TT.compute_bands(NS(world=world, kit=kit), spec, top, out_dir=str(out_dir))


def test_compute_bands_runs_on_the_band_kit_and_records_it(band_kit_of_fakes):
    world, kit = _setup()
    rb = _compute_tabletop(world, kit, band_kit_of_fakes)
    bk = kit._band_kit
    assert _BC.kits == [bk] and bk.calls == ["sync", "collision_free", "sync", "compute", "sync"]  # stance search, band, re-sync
    assert kit.calls == []  # the sampler's kit: no sync, no collision check, no IK
    assert rb.meta["schedule"] == {"mesh_cache": 832, "kit": "band"} and rb.meta["n_reachable_total"] == 2
    with open(rb.path[:-4] + ".json") as f:
        assert json.load(f)["schedule"] == {"mesh_cache": 832, "kit": "band"}


def test_compute_bands_with_the_switch_off_is_the_old_path(band_kit_of_fakes, monkeypatch):
    monkeypatch.setenv(ENV, "off")
    world, kit = _setup()
    rb = _compute_tabletop(world, kit, band_kit_of_fakes)
    assert _BC.kits == [kit] and kit.calls == ["collision_free", "sync", "compute", "sync"] and getattr(kit, "_band_kit", None) is None
    assert rb.meta["schedule"] == {"mesh_cache": 8192, "kit": "sampler"}
    # the file name does not depend on the kit (as the CLI's --mesh-cache): an exact and a snug band of the same factors share it
    monkeypatch.delenv(ENV)
    world2, kit2 = _setup()
    assert os.path.basename(_compute_tabletop(world2, kit2, band_kit_of_fakes).path) == os.path.basename(rb.path)


def test_iors_large_floor_band_runs_on_the_band_kit(band_kit_of_fakes):
    IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
    world, kit = _setup()
    world.spawned = {"target_apple_0": NS(name="target_apple_0")}
    s = NS(choice=NS(models={"target": NS(category="apple", model="agveuv", scale=None)}), world=world, kit=kit, target=NS(name="target_apple_0"),
           floor_z=0.0, gp=NS(rooms=lambda scene: ["living_room_0"], alias="iors-large-test"), scene="house_single_floor", obj=lambda role: NS(name="floors_0"),
           hull=th.zeros(8, 3), rest=th.tensor([0.0, 0.0, 0.0, 1.0]), grasps=[NS(name="grasp_0", position=th.tensor([0.0, 0.0, 0.1]))],
           grasp_key="apple__agveuv__t", relation="ontop", place={})
    rb = IL.IORSLarge.compute_band(s, "floor")
    bk = kit._band_kit
    assert _BC.kits == [bk] and "compute" in bk.calls and kit.calls == []
    assert world.log == [("park", "target_apple_0", 0), ("restore", "snap")]
    assert rb.meta["schedule"] == {"mesh_cache": 832, "kit": "band"} and os.path.exists(rb.path)


def _handle_sampler(world, kit, band_trunks):
    return NS(world=world, kit=kit, top=NS(name="bottom_cabinet_0", z=0.9, cells=lambda free_only=True: th.zeros(3, 2), to_json=lambda: {}),
              choice=NS(models={"target": NS(category="microwave", model="abzvij")}), role="target", joint_name="j_link_0", _support=lambda: NS(name="bottom_cabinet_0"),
              face_yaw=1.57, handle={"contact": [0.1, 0.0, 0.2], "approach_dir": [1.0, 0.0, 0.0], "pinch_axis": [0.0, 0.0, 1.0]}, obj=lambda role: NS(scale=None),
              _band_q_values=lambda: [0.0, 1.2], gp=NS(alias="articulation-test"), _stage2_enabled=lambda: False, required=0.5, passive_from=0.8,
              gen=lambda key, default=None: band_trunks if key == "band_trunks" else default, handle_arm="right", brace_arm="left", hull=th.zeros(8, 3))


def test_handle_band_on_the_band_kit_and_none_built_on_a_cache_hit(band_kit_of_fakes, monkeypatch):
    AS = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_small")
    scopes = []

    class Scope:
        def __init__(self, kit, world, base_xy):
            scopes.append(kit)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(RB, "ObstacleScope", Scope)
    hit = NS(meta={"mode": "brace"}, cells={(0.5, 0.0): np.zeros((2, 2))}, path="brace.npz")
    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(lambda *a, mode=None, **kw: hit if mode == "brace" else (_ for _ in ()).throw(FileNotFoundError())))
    world, kit = _setup()
    assert AS.ArticulationSmall._brace_band(_handle_sampler(world, kit, [(0.5, 0.0)]), NS(name="+u"), None) is hit
    assert getattr(kit, "_band_kit", None) is None and _BC.kits == [kit] and kit.calls == ["collision_free"]  # the stance keys the lookup

    def miss(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(miss))
    world, kit = _setup()
    _BC.kits = []
    rb = AS.ArticulationSmall._brace_band(_handle_sampler(world, kit, [(0.5, 0.0), (0.25, -0.3)]), NS(name="+u"), None)
    bk = kit._band_kit
    assert _BC.kits == [kit, bk] and scopes == [bk] and kit.calls == ["collision_free"]  # only the stance search on the sampler's kit
    assert bk.calls == ["sync", "sync", "collision_free", "collision_free", "sync"]  # both trunks checked on the band kit (both collide)
    assert rb.meta["schedule"] == {"mesh_cache": 832, "kit": "band"} and os.path.exists(rb.path)


# ---------------------------------------------------------------------------------------------------------------- the time budget
# s3 touching#02 c5_k1: an in-process left contact band ran 2945 s of a 1800 s sampling budget (the job ended at 3429 s)


class _Scope:
    exits = 0

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _Scope.exits += 1
        return False


def _bare_computer(deadline):
    """A ``BandComputer`` without a kit (``__new__``): just what :meth:`compute` and :meth:`_trunk` read."""
    import contextlib

    from ispatialgym.sampling import geometry as G

    bc = RB.BandComputer.__new__(RB.BandComputer)
    bc.G, bc.world, bc.deadline = G, NS(floor_z=0.0), deadline
    bc.kit = NS(ignoring=lambda objs: contextlib.nullcontext())
    bc.scope, bc.cell_obstacle, bc.mesh_groups, bc.protect_extra = None, None, {}, []
    bc.postures, bc.last_trunk_arm_excluded = ("untucked",), False
    bc.lattice_on_cells = lambda stance, cells: (th.tensor([[0.5, 0.0], [0.5, 0.1]]), th.tensor([[0.5, 0.0], [0.5, 0.1]]))
    bc.base_frame = lambda stance: (th.zeros(2), th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0]))
    return bc


def test_band_computation_stops_at_the_time_budget_before_the_next_trunk(monkeypatch):
    import time

    monkeypatch.setattr(RB, "ObstacleScope", _Scope)
    _Scope.exits = 0
    bc = _bare_computer(time.time() + 1000.0)
    done = []

    def trunk(tr, ti, *a, **kw):
        done.append((tr, ti))
        if len(done) == 2 and bc.deadline is not None:
            bc.deadline = time.time() - 1.0  # the budget runs out during the second trunk
        return {"translate": tr, "tilt": ti}, np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)

    bc._trunk = trunk
    trunks = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.95, 0.0)]
    with pytest.raises(RB.BandDeadline, match=r"before trunk 3 of 5 \(0.50, \+0.0\)"):
        bc.compute({"base_pose2d": [0.0, 0.0, 0.0]}, th.zeros(2, 2), 0.8, None, None, [], trunks=trunks)
    assert done == trunks[:2] and _Scope.exits == 1 and bc.scope is None  # the obstacle scope is left as after a full band
    bc.deadline = None  # the CLI: no limit
    done.clear()
    assert len(bc.compute({"base_pose2d": [0.0, 0.0, 0.0]}, th.zeros(2, 2), 0.8, None, None, [], trunks=trunks)["trunks"]) == 5


def test_band_computation_stops_at_the_time_budget_before_the_next_ik_round():
    """A trunk's IK rounds (one batched pre-grasp + grasp pass per 8 (grasp, yaw) pairs) check the budget too: a trunk can take minutes."""
    import time

    from ispatialgym.robot import conventions as C

    bc = _bare_computer(time.time() + 1000.0)
    bc.TRUNK_BOUND = False
    bc.trunk_config = lambda tr, ti, x0, y0, yaw0: (th.zeros(28), "untucked")
    bc.idle_arm_q = lambda q: {}
    bc.shoulder_and_reach = lambda q, arm: (th.tensor([0.5, 0.0, 1.0]), 1.0)
    bc.grasp_eef_pair = lambda obj_pose, g: [(th.tensor([0.5, 0.0, 1.2]), C.IDENTITY_QUAT), (th.tensor([0.5, 0.0, 1.1]), C.IDENTITY_QUAT)]
    screens = []

    def ik_screen(poses, q, arm):  # pre-grasps feasible, grasps not: every cell stays open for the next round
        screens.append(len(poses))
        bc.deadline = time.time() - 1.0
        return th.ones(len(poses), dtype=th.bool) if len(screens) % 2 else th.zeros(len(poses), dtype=th.bool)

    bc.ik_screen = ik_screen
    grasps = [NS(name=f"grasp_{i}") for i in range(3)]  # 3 grasps x 8 yaws = 3 rounds of 8 pairs
    yaw_order = list(RB.YAWS)
    with pytest.raises(RB.BandDeadline, match=r"in trunk \(0.50, \+0.0\), round 2"):
        bc._trunk(0.5, 0.0, th.tensor([[0.5, 0.0], [0.5, 0.1]]), th.tensor([[0.5, 0.0], [0.5, 0.1]]), {"base_pose2d": [0.0, 0.0, 0.0]}, 0.8,
                  lambda xy, yaw: (th.zeros(3), C.IDENTITY_QUAT), grasps, [], "right", 0.0, yaw_order, {y: k for k, y in enumerate(RB.YAWS)},
                  {g.name: k for k, g in enumerate(grasps)}, th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0]))
    assert screens == [16, 16]  # round 1 only: 2 cells x 8 pairs, pre-grasp then grasp


def test_compute_bands_past_the_time_budget_rejects_and_writes_nothing(band_kit_of_fakes):
    import time

    from ispatialgym.sampling.sim_utils import Reject

    seen = []

    class _Late(_BC):
        def compute(self, *a, **kw):
            seen.append(self.deadline)
            raise RB.BandDeadline("the sampling time budget ended before trunk 7 of 20 (0.25, -0.6)")

    RB.BandComputer = _Late  # the fixture restores it
    world, kit = _setup()
    top = NS(name="bookcase_jpdtvg_0", z=0.84, cells=lambda free_only=True: th.zeros(4, 2), to_json=lambda: {"name": "bookcase_jpdtvg_0"})
    spec = TT.BandSpec(object=TT.left_contact_object(0.24), generic_pinch=True, height=0.24, arm="left", contact=True)
    end = time.time() + 5.0
    with pytest.raises(Reject) as e:
        TT.compute_bands(NS(world=world, kit=kit, deadline=end), spec, top, out_dir=str(band_kit_of_fakes))
    assert e.value.reason == "band_budget" and "before trunk 7 of 20" in str(e.value) and "nothing written" in str(e.value)
    assert seen == [end] and os.listdir(band_kit_of_fakes) == []
    # a sampler without a budget (unit tests, the CLI's callers) computes as before
    RB.BandComputer = _BC
    assert TT.compute_bands(NS(world=world, kit=kit), spec, top, out_dir=str(band_kit_of_fakes)).meta["n_reachable_total"] == 2


def test_handle_band_past_the_time_budget_rejects_and_writes_nothing(band_kit_of_fakes, monkeypatch):
    import time

    from ispatialgym.sampling.sim_utils import Reject

    AS = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_small")
    monkeypatch.setattr(RB, "ObstacleScope", _Scope)
    _Scope.exits = 0

    def miss(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(miss))
    world, kit = _setup()
    s = _handle_sampler(world, kit, [(0.5, 0.0), (0.25, -0.3)])
    s.deadline = time.time() - 1.0
    with pytest.raises(Reject) as e:
        AS.ArticulationSmall._brace_band(s, NS(name="+u"), None)
    assert e.value.reason == "band_budget" and "before handle trunk 1 of 2" in str(e.value)
    assert _Scope.exits == 1 and not any(f.startswith("handle__") for f in os.listdir(band_kit_of_fakes))


# ---------------------------------------------------------------------------------------------------------------- the trunk check
# owner 2026-10-12 item 4: the iors_large jobs' in-process floor band of mouse/pljjtu (living_room_0) had 0 cells with the whole-body
# check; the CLI's --trunk-check idle-arm band has 101

TC = RB.BAND_TRUNK_CHECK_ENV


class _RecBC(_BC):
    """``_BC`` that keeps its instances (the trunk check is set on the computer) and has the under stance."""

    made = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trunk_skip_arm = None
        _RecBC.made.append(self)

    def stance_under(self, partner, stand_off):
        self.kit.collision_free(th.zeros(28))
        return {"edge": None, "base_pose2d": [1.0, 2.0, 0.0], "edge_mid": [1.0, 2.0], "inward": [1.0, 0.0]}


def test_band_trunk_check_switch(monkeypatch):
    monkeypatch.delenv(TC, raising=False)
    bc = NS(trunk_skip_arm=None)
    assert RB.band_trunk_check(bc, "right") == {} and bc.trunk_skip_arm is None
    monkeypatch.setenv(TC, "whole-body")
    assert RB.band_trunk_check(bc, "right") == {} and bc.trunk_skip_arm is None
    for idle in ("idle-arm", "IDLE_ARM"):
        monkeypatch.setenv(TC, idle)
        bc = NS(trunk_skip_arm=None)
        assert RB.band_trunk_check(bc, "left") == {"trunk_check": "idle_arm"} and bc.trunk_skip_arm == "left"
    monkeypatch.setenv(TC, "idle")
    with pytest.raises(ValueError, match=TC):
        RB.band_trunk_check(NS(trunk_skip_arm=None), "right")
    # the CLI's --trunk-check idle-arm record, shared with the in-process bands
    assert RB.trunk_check_meta("right") == {"mode": "idle_arm", "note": "a trunk whose idle-arm postures all collide with the whole body is retried with the "
                                            "right arm out of the check (its configuration comes from the IK at every cell); see trunks[].active_arm_excluded"}


def test_compute_bands_idle_arm_band_is_keyed_named_and_recorded_like_the_cli(band_kit_of_fakes, monkeypatch):
    monkeypatch.setattr(RB, "BandComputer", _RecBC)
    _RecBC.made = []
    world, kit = _setup()
    whole = _compute_tabletop(world, kit, band_kit_of_fakes)
    monkeypatch.setenv(TC, "whole-body")
    assert _compute_tabletop(world, kit, band_kit_of_fakes).path == whole.path
    monkeypatch.setenv(TC, "idle-arm")
    idle = _compute_tabletop(world, kit, band_kit_of_fakes)
    assert [b.trunk_skip_arm for b in _RecBC.made] == [None, None, "right"]
    assert os.path.basename(whole.path).startswith("tabletop__house_single_floor__countertop_kelzer_0__stockpot-oshwps__")
    assert os.path.basename(idle.path).startswith("tabletop__house_single_floor__countertop_kelzer_0-tcidle__stockpot-oshwps__")
    assert idle.meta["key"] != whole.meta["key"] and idle.meta["trunk_check"] == RB.trunk_check_meta("right") and "trunk_check" not in whole.meta
    for rb in (whole, idle):  # two files: neither band overwrites the other
        with open(rb.path[:-4] + ".json") as f:
            assert json.load(f).get("trunk_check") == rb.meta.get("trunk_check")


def test_iors_large_idle_arm_floor_bands_are_keyed_named_and_recorded_like_the_cli(band_kit_of_fakes, monkeypatch):
    IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
    monkeypatch.setattr(RB, "BandComputer", _RecBC)
    _RecBC.made = []
    world, kit = _setup()
    world.spawned = {"target_mouse_0": NS(name="target_mouse_0")}
    s = NS(choice=NS(models={"target": NS(category="mouse", model="pljjtu", scale=None)}), world=world, kit=kit, target=NS(name="target_mouse_0"),
           floor_z=0.0, gp=NS(rooms=lambda scene: ["living_room_0"], alias="iors-large-test"), scene="house_single_floor", obj=lambda role: NS(name="floors_0"),
           hull=th.zeros(8, 3), rest=th.tensor([0.0, 0.0, 0.0, 1.0]), grasps=[NS(name="grasp_0", position=th.tensor([0.0, 0.0, 0.1]))],
           grasp_key="mouse__pljjtu__t", relation="ontop", place={})
    whole = IL.IORSLarge.compute_band(s, "floor")
    monkeypatch.setenv(TC, "idle-arm")
    idle = IL.IORSLarge.compute_band(s, "floor")
    s.relation, s.place = "under", {"cells": th.zeros(3, 2)}
    under = IL.IORSLarge.compute_band(s, "floor", under_partner=NS(name="sofa_0"))
    assert [b.trunk_skip_arm for b in _RecBC.made] == [None, "right", "right"]
    assert os.path.basename(whole.path).startswith("floor__house_single_floor__floor__mouse-pljjtu__")
    assert os.path.basename(idle.path).startswith("floor__house_single_floor__floor-tcidle__mouse-pljjtu__")
    assert os.path.basename(under.path).startswith("floor__house_single_floor__floor-tcidle-under-sofa_0__mouse-pljjtu__")  # the CLI's tag order
    assert idle.meta["trunk_check"] == under.meta["trunk_check"] == RB.trunk_check_meta("right") and "trunk_check" not in whole.meta
    assert os.path.exists(whole.path) and os.path.exists(idle.path)


def test_handle_band_keeps_the_whole_body_trunk_check(band_kit_of_fakes, monkeypatch):
    AS = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_small")
    monkeypatch.setenv(TC, "idle-arm")
    monkeypatch.setattr(RB, "ObstacleScope", _Scope)
    monkeypatch.setattr(RB, "BandComputer", _RecBC)

    def miss(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(miss))
    _RecBC.made = []
    world, kit = _setup()
    rb = AS.ArticulationSmall._brace_band(_handle_sampler(world, kit, [(0.5, 0.0)]), NS(name="+u"), None)
    assert _RecBC.made and all(b.trunk_skip_arm is None for b in _RecBC.made)
    assert RB.TRUNK_CHECK_IDLE_TAG not in os.path.basename(rb.path) and "trunk_check" not in rb.meta
