"""Simulator-free tests of ``generation.partner_joint_init`` (owner 2026-10-12 ~16:05, inside_large_scale#02: the fridge's left door
open at init): the joint request, the template setup (set, park, settle, re-read), what the open door changes (park keep-away, path
footprints, the TRO ``joint_positions`` constraint) and the loader's restore / end-of-episode report
(``pytest ispatialgym/tests/test_partner_joint_init.py``)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import object_joints as OJ
from ispatialgym.sampling.sim_utils import Reject, SupportInterior
from ispatialgym.sampling.tasks.base import Draft

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
from ispatialgym.sampling.v2 import core as CORE  # noqa: E402
from ispatialgym.sampling.v2 import navigation as NAV  # noqa: E402

VARIANT = "inter_object_relation_satisfaction-inside_large_scale-jar_of_spaghetti_sauce-fridge-bar"
IDENT = th.tensor([0.0, 0.0, 0.0, 1.0])


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = VARIANT, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)

    def rooms(self, scene):
        return ["kitchen_0"]


class _Link:
    """A link whose pose follows one joint (``x`` = the joint value) and whose AABB is given per joint value."""

    def __init__(self, obj, joint, box):
        self.obj, self.joint, self.box, self.collision_meshes = obj, joint, box, {}

    def get_position_orientation(self):
        x = 0.0 if self.joint is None else float(self.obj.q[self.joint])
        return th.tensor([x, 0.0, 0.0]), IDENT.clone()

    @property
    def aabb(self):
        return self.box(0.0 if self.joint is None else float(self.obj.q[self.joint]))


class _Fridge:
    """fridge_petcxr_0 in miniature: a body, the left door swinging out to +x / +y with j_left_door, the right door closed."""

    name = "fridge_petcxr_0"
    root_link_name = "base_link"

    def __init__(self, drift: float = 0.0):
        self.joints = {"j_left_door": SimpleNamespace(lower_limit=0.0, upper_limit=2.356), "j_right_door": SimpleNamespace(lower_limit=0.0, upper_limit=2.356)}
        self.q = th.zeros(2)
        self.drift = drift
        self.calls = []
        door = lambda q: (th.tensor([0.0, 0.5, 0.0]), th.tensor([0.5 + 0.3 * math.sin(q), 0.5 + 0.45 * math.sin(q), 2.0]))  # noqa: E731
        self.links = {"base_link": _Link(self, None, lambda q: (th.tensor([-0.5, -0.5, 0.0]), th.tensor([0.5, 0.45, 2.0]))),
                      "left_door": _Link(self, 0, door),
                      "right_door": _Link(self, 1, lambda q: (th.tensor([-0.5, 0.45, 0.0]), th.tensor([0.0, 0.5, 2.0])))}

    @property
    def aabb(self):
        return OJ.union_box([link.aabb for link in self.links.values()])

    def get_joint_positions(self):
        return self.q.clone()

    def set_joint_positions(self, q):
        self.q = th.as_tensor(q, dtype=th.float32).clone()
        self.calls.append(("set", [round(float(v), 4) for v in self.q]))

    def set_joint_velocities(self, v):
        self.calls.append(("vel",))

    def keep_still(self):
        pass


class _S(IL.IORSLarge):
    """IORSLarge with the bound partner and the parking stubbed (the order of the setup steps logged)."""

    partner = property(lambda self: self._partner)
    pname = property(lambda self: "fridge.n.01_1")

    def park_robot(self):
        self.log.append(("park", [o.name for o in self.park_keep_away()]))


def _sampler(drift: float = 0.0, **gen) -> _S:
    s = _S.__new__(_S)
    s.gp, s._partner, s.template_facts, s.log = _GP(**gen), _Fridge(drift), {}, []

    def settle(ents, steps=60, **kw):
        s.log.append(("settle", [e.name for e in ents], steps))
        s._partner.q[0] -= s._partner.drift  # the door eases back a little while it settles
        return None

    s.world = SimpleNamespace(settle=settle)
    return s


# ----------------------------------------------------------------------------------------------------------------
# object_joints helpers
# ----------------------------------------------------------------------------------------------------------------


def test_parse_joint_spec_clamps_and_rejects():
    limits = {"j_left_door": (0.0, 2.356), "j_right_door": (0.0, 2.356)}
    spec, clamped = OJ.parse_joint_spec({"j_left_door": 2.2}, ["j_left_door", "j_right_door"], limits)
    assert spec == {"j_left_door": pytest.approx(2.2)} and clamped == {}
    spec, clamped = OJ.parse_joint_spec({"j_left_door": "3.0"}, ["j_left_door"], limits)
    assert spec == {"j_left_door": pytest.approx(2.356)} and clamped == {"j_left_door": {"requested": 3.0, "applied": pytest.approx(2.356)}}
    for bad in ({}, None, [("j_left_door", 1.0)], {"j_nope": 1.0}, {"j_left_door": "wide"}, {"j_left_door": float("nan")}):
        with pytest.raises(ValueError):
            OJ.parse_joint_spec(bad, ["j_left_door"], limits)


def test_moved_links_boxes_footprints_and_raster():
    before = {"base_link": (th.zeros(3), IDENT), "left_door": (th.zeros(3), IDENT), "right_door": (th.zeros(3), IDENT)}
    after = {"base_link": (th.tensor([0.004, 0.0, 0.0]), IDENT), "left_door": (th.tensor([0.3, 0.2, 0.0]), IDENT),
             "right_door": (th.zeros(3), th.tensor([0.0, 0.0, math.sin(0.04), math.cos(0.04)]))}
    assert OJ.moved_links(before, after) == ["left_door", "right_door"]  # 4 mm: still; 0.08 rad: moved
    after["right_door"] = (th.zeros(3), th.tensor([0.0, 0.0, math.sin(0.02), math.cos(0.02)]))  # 0.04 rad: still
    assert OJ.moved_links(before, after) == ["left_door"]
    fridge = _Fridge()
    fridge.q[0] = 2.2
    lo, hi = OJ.static_aabb(fridge, ["left_door"])  # the body and the closed right door only
    assert hi.tolist() == pytest.approx([0.5, 0.5, 2.0]) and float(fridge.aabb[1][1]) > 0.8
    assert [b.tolist() for b in OJ.static_aabb(fridge, [])] == [b.tolist() for b in fridge.aabb]
    (fp,) = OJ.footprints([(th.tensor([0.0, 0.5, 0.0]), th.tensor([0.4, 0.9, 2.0]))], 0.02)
    assert th.allclose(th.tensor(fp), th.tensor([[-0.02, 0.48], [0.42, 0.48], [0.42, 0.92], [-0.02, 0.92]]), atol=1e-6)
    xs, ys = th.arange(5) * 0.1, th.arange(5) * 0.1
    top = SupportInterior("fridge_petcxr_0", xs, ys, th.ones(5, 5, dtype=th.bool), th.zeros(5, 5, dtype=th.bool), 0.97, 0.0)
    assert OJ.mask_out_boxes(top, [(th.tensor([0.25, 0.25, 0.0]), th.tensor([0.45, 0.45, 2.0]))]) == 4
    assert int(top.mask.sum()) == 21 and not bool(top.mask[3, 3]) and bool(top.mask[2, 2])
    assert OJ.mask_out_boxes(top, []) == 0
    assert OJ.specs_match({"j_left_door": 2.2}, {"j_left_door": 2.21}) and not OJ.specs_match({"j_left_door": 2.2}, {"j_left_door": 2.3})
    assert OJ.specs_match(None, {}) and not OJ.specs_match(None, {"j_left_door": 2.2})


# ----------------------------------------------------------------------------------------------------------------
# the template setup
# ----------------------------------------------------------------------------------------------------------------


def test_partner_joint_init_set_park_settle_reread():
    s = _sampler(drift=0.01, partner_joint_init={"j_left_door": 2.2})
    s.apply_partner_joint_init()
    f = s._partner
    assert f.calls[0] == ("set", [2.2, 0.0])  # the request first, the other door as it was
    assert s.log == [("park", ["fridge_petcxr_0"]), ("settle", ["fridge_petcxr_0"], IL.PARTNER_JOINT_SETTLE_STEPS)]  # re-parked away, then settled
    assert s.partner_joints == {"j_left_door": pytest.approx(2.19), "j_right_door": 0.0} and s.partner_moved == ["left_door"]
    facts = s.template_facts["partner_joint_init"]
    assert facts["requested"] == {"j_left_door": 2.2} and facts["applied"] == {"j_left_door": pytest.approx(2.2)} and facts["clamped"] == {}
    assert facts["settled"] == {"j_left_door": pytest.approx(2.19)} and facts["max_settle_dev_rad"] == pytest.approx(0.01) and facts["moved_links"] == ["left_door"]
    assert facts["scene_file"] == {"j_left_door": 0.0}
    assert s.park_keep_away() == [f]
    (fp,) = s.path_blocked()  # the open door's footprint (+ 2 cm) on the planner's local map
    lo, hi = f.links["left_door"].aabb
    want = [[float(lo[0]) - 0.02, float(lo[1]) - 0.02], [float(hi[0]) + 0.02, float(lo[1]) - 0.02], [float(hi[0]) + 0.02, float(hi[1]) + 0.02],
            [float(lo[0]) - 0.02, float(hi[1]) + 0.02]]
    assert th.allclose(th.tensor(fp), th.tensor(want), atol=1e-6)
    assert s.partner_static_aabb()[1].tolist() == pytest.approx([0.5, 0.5, 2.0])
    parts = s.partner_part_boxes()  # the raster leaves out both doors (the open one and the closed one), not the body
    assert [[b.tolist() for b in box] for box in parts] == [[b.tolist() for b in f.links[n].aabb] for n in ("left_door", "right_door")]


def test_partner_joint_init_default_and_bad_requests():
    s = _sampler()
    s.apply_partner_joint_init()  # no key: nothing set, nothing parked, nothing kept away or blocked
    assert s._partner.calls == [] and s.log == [] and s.partner_joints is None and s.partner_moved == []
    assert s.park_keep_away() == [] and s.path_blocked() == [] and s.partner_part_boxes() == [] and "partner_joint_init" not in s.template_facts
    for bad in ({"j_nope": 1.0}, {"j_left_door": "open"}):
        s = _sampler(partner_joint_init=bad)
        with pytest.raises(Reject) as e:
            s.apply_partner_joint_init()
        assert e.value.reason == "spec" and "partner_joint_init" in str(e.value)
    s = _sampler(partner_joint_init={"j_left_door": 4.0})  # past the limit: clamped, recorded
    s.apply_partner_joint_init()
    assert s.template_facts["partner_joint_init"]["clamped"] == {"j_left_door": {"requested": 4.0, "applied": pytest.approx(2.356)}}


def test_every_attempt_applies_and_records_the_joints(monkeypatch):
    s = _sampler(partner_joint_init={"j_left_door": 2.2})
    s.apply_partner_joint_init()
    monkeypatch.setattr(CORE.SamplerV2, "sample", lambda self, rng: Draft())
    d = s.sample(np.random.default_rng(0))
    assert d.object_joints["fridge.n.01_1"].tolist() == pytest.approx([2.2, 0.0])  # base.apply_draft sets them before the settle
    d.object_joints["fridge.n.01_1"] = th.tensor([2.195, 0.001])  # as re-read after the instance's settle
    d.extras["constraints"] = [{"type": "joint_positions", "object": "fridge.n.01_1", "joints": {"j_left_door": 1.0}}, {"type": "mass", "object": "x"}]
    s.record_partner_joints(d)
    recs = [c for c in d.extras["constraints"] if c["type"] == "joint_positions"]
    assert len(recs) == 1 and recs[0]["object"] == "fridge.n.01_1" and recs[0]["joints"] == {"j_left_door": pytest.approx(2.195)}
    assert recs[0]["source"] == "generation.partner_joint_init" and {"type": "mass", "object": "x"} in d.extras["constraints"]
    assert d.summary["partner_joints"] == {"object": "fridge_petcxr_0", "init": {"j_left_door": pytest.approx(2.195), "j_right_door": pytest.approx(0.001)},
                                           "requested": {"j_left_door": pytest.approx(2.195)}}
    plain = _sampler()
    plain.apply_partner_joint_init()
    d2 = plain.sample(np.random.default_rng(0))
    plain.record_partner_joints(d2)
    assert d2.object_joints == {} and "constraints" not in d2.extras and "partner_joints" not in d2.summary


def test_core_park_and_path_hooks(monkeypatch):
    """``SamplerV2.park_robot`` keeps away from :meth:`park_keep_away` too, ``record_path`` hands :meth:`path_blocked` to the planner;
    both empty by default."""
    seen = {}

    def park_pose(kit, rooms, keep_away=(), **kw):
        seen["keep"] = [getattr(o, "name", o) for o in keep_away]
        return th.tensor([1.0, 2.0])

    monkeypatch.setattr(NAV, "park_pose", park_pose)
    monkeypatch.setattr(NAV, "plan_path", lambda kit, start, to, **kw: kw["blocked"])

    class _P(CORE.SamplerV2):
        support_objects = lambda self: []  # noqa: E731

    for keep, blocked in (([], []), ([SimpleNamespace(name="fridge_petcxr_0")], [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])):
        s = _P.__new__(_P)
        s.gp, s.scene, s.sidecar = _GP(), "house_double_floor_lower", SimpleNamespace(add_path=lambda p: p)
        s.world = SimpleNamespace(spawned={"jar": SimpleNamespace(name="jar")}, rooms=["kitchen_0"], measure_floor_z=lambda xy: 0.0, park_robot=lambda xy, floor_z=0.0: None)
        s.kit = SimpleNamespace(sync_world=lambda: None, floor_z=0.0)
        s.park_keep_away = lambda keep=keep: keep
        s.path_blocked = lambda blocked=blocked: blocked
        s.park_robot()
        assert seen["keep"] == ["jar"] + [o.name for o in keep]
        a = SimpleNamespace(name="init", posture="untucked", carried=None)
        b = SimpleNamespace(name="manipulation", posture="untucked", carried=None)
        assert s.record_path(a, b, blocked=[[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]]) == [[[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]] + blocked


# ----------------------------------------------------------------------------------------------------------------
# the goal-side band of the partner with its door open
# ----------------------------------------------------------------------------------------------------------------


def test_band_cli_sets_the_support_joints(monkeypatch):
    """``reach_bands --support-joints``: the support's joints set and clamped, the robot re-parked away from it, a settle, the world
    re-synced; the meta keys (the lookup key = the request as applied), the stance AABB without the moved door, the doors' pieces."""
    RB = pytest.importorskip("ispatialgym.sampling.reach_bands")
    fridge, log = _Fridge(), []
    world = SimpleNamespace(obj=lambda name: fridge, robot="robot", floor_z=0.0, loaded_rooms=["kitchen_0"],
                            park_robot=lambda xy, floor_z=0.0: log.append(("park", [round(float(v), 2) for v in xy])),
                            step_physics=lambda n, still=(): log.append(("steps", n, list(still))), full_step=lambda render=True: log.append(("full",)))
    kit = SimpleNamespace(sync_world=lambda: log.append(("sync",)))
    monkeypatch.setattr(NAV, "park_pose", lambda kit, rooms, keep_away=(), **kw: log.append(("away_from", [o.name for o in keep_away], rooms)) or th.tensor([3.0, 1.0]))
    bc = SimpleNamespace(support_aabb=None, support_part_boxes=[])
    meta = {}
    spec = RB.apply_support_joints(world, kit, bc, "fridge_petcxr_0", '{"j_left_door": 2.3}', ["kitchen_0"], meta)
    assert spec == {"j_left_door": pytest.approx(2.3)} and fridge.q.tolist() == pytest.approx([2.3, 0.0])
    assert log == [("away_from", ["fridge_petcxr_0"], ["kitchen_0"]), ("park", [3.0, 1.0]), ("steps", RB.SUPPORT_JOINT_SETTLE_STEPS, ["robot"]), ("full",), ("sync",)]
    assert meta["support_joints"] == {"j_left_door": 2.3} and meta["support_joints_settled"] == {"j_left_door": 2.3} and meta["support_moved_links"] == ["left_door"]
    assert bc.support_aabb[1].tolist() == pytest.approx([0.5, 0.5, 2.0]) and len(bc.support_part_boxes) == 2  # both doors leave an interior raster
    meta2 = {}
    RB.apply_support_joints(world, kit, SimpleNamespace(), "fridge_petcxr_0", {"j_left_door": 9.0}, ["kitchen_0"], meta2)
    assert meta2["support_joints"] == {"j_left_door": 2.356} and "support_joints_clamped" in meta2
    with pytest.raises(SystemExit):
        RB.apply_support_joints(world, kit, bc, "fridge_petcxr_0", '{"j_nope": 1.0}', ["kitchen_0"], {})
    with pytest.raises(SystemExit):
        RB.apply_support_joints(world, kit, bc, "fridge_petcxr_0", "not json", ["kitchen_0"], {})


def test_inside_band_asks_for_the_partner_joints(monkeypatch):
    """The inside mode looks its goal-side band up with the joints ``partner_joint_init`` applied (no request: the plain band)."""
    RB = pytest.importorskip("ispatialgym.sampling.reach_bands")
    from ispatialgym.sampling.v2.tasks import iors_large_inside as IN

    asked = []

    def load(*a, **kw):
        asked.append(kw.get("support_joints"))
        raise FileNotFoundError("no band")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(load))
    for facts, want in (({"partner_joint_init": {"applied": {"j_left_door": 2.3}}}, {"j_left_door": 2.3}), ({}, None)):
        s = SimpleNamespace(choice=SimpleNamespace(models={"target": SimpleNamespace(category="jar_of_spaghetti_sauce", model="exmsmm", scale=None)}),
                            place={"z_requested": 1.20}, world=SimpleNamespace(scene_model="house_double_floor_lower"), partner=SimpleNamespace(name="fridge_petcxr_0"),
                            grasp_key="jar_of_spaghetti_sauce__exmsmm__t", template_facts=dict(facts), gp=_GP())
        assert IN.InsideMode.inside_band(s, "right") is None
        assert asked[-1] == want and s.template_facts["inside_bands"] == {"right": None}


# ----------------------------------------------------------------------------------------------------------------
# the loader
# ----------------------------------------------------------------------------------------------------------------


def test_loader_sets_holds_and_reports_the_joint_positions():
    R = pytest.importorskip("ispatialgym.oracle.run_isg")
    fridge = _Fridge()
    loader = object.__new__(R.InstanceLoader)
    loader.physical = {"fridge.n.01_1": fridge, "jar_of_spaghetti_sauce.n.01_1": SimpleNamespace(name="jar")}
    loader.tro = SimpleNamespace(meta=SimpleNamespace(articulation_targets={}))
    rec = {"type": "joint_positions", "object": "fridge.n.01_1", "joints": {"j_left_door": 2.195}, "source": "generation.partner_joint_init"}
    out = loader._restore_constraints([rec])
    assert out == [{"type": "joint_positions", "object": "fridge.n.01_1", "joints": {"j_left_door": pytest.approx(2.195)}}]  # one report per record
    assert fridge.q.tolist() == pytest.approx([2.195, 0.0])
    held = loader._held_articulations(loader.physical)  # held through the settle, restored after a reset
    assert list(held) == ["fridge.n.01_1"] and held["fridge.n.01_1"].tolist() == pytest.approx([2.195, 0.0])
    fridge.q[0] = 1.9  # the episode pushed the door
    assert loader.joint_positions_report() == {"fridge.n.01_1": {"scene_object": "fridge_petcxr_0", "init": {"j_left_door": pytest.approx(2.195)},
                                                                 "end": {"j_left_door": pytest.approx(1.9)}}}
    with pytest.raises(R.LoadError):
        loader._restore_constraints([dict(rec, joints={"j_nope": 1.0})])
    with pytest.raises(R.LoadError):
        loader._restore_constraints([dict(rec, object="cabinet.n.01_1")])
    plain = object.__new__(R.InstanceLoader)
    plain.physical, plain.tro = dict(loader.physical), loader.tro
    assert plain._restore_constraints([]) == [] and plain._held_articulations(plain.physical) == {} and plain.joint_positions_report() == {}
