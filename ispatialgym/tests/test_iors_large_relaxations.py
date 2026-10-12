"""Simulator-free tests of the large-scale IORS relaxation keys (owner 2026-10-12 item 2, the large-scale eval picks; spec §5.6
"Relaxation keys"): every key unset keeps the rule as before (``pytest ispatialgym/tests/test_iors_large_relaxations.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import geometry as G

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


class _GP:
    def __init__(self, alias: str = "inter_object_relation_satisfaction-ontop_large_scale-mouse-desk-floor", **gen):
        self.alias, self.generation = alias, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


# ----------------------------------------------------------------------------------------------------------------
# 2a: partner partly in view (generation.partner_view_min_points)
# ----------------------------------------------------------------------------------------------------------------


def test_enough_in_view_counts_the_points_in_frame():
    inside = th.tensor([True, False, False])
    assert not IL.enough_in_view(inside)  # default: every point, as before
    assert IL.enough_in_view(th.tensor([True, True, True]))
    assert IL.enough_in_view(inside, 1) and not IL.enough_in_view(inside, 2)
    assert IL.enough_in_view(th.tensor([True, True, True]), 5) and not IL.enough_in_view(th.tensor([True, True, False]), 5)  # fewer points: all
    assert not IL.enough_in_view(th.tensor([False, False]), 1)
    assert "partner_view_min_points" in IL.RELAXATION_KEYS


class _ViewStub:
    """The attributes :meth:`IORSLarge.goal_region_visible` / :meth:`partner_points_in_view` read: a 90° pinhole camera at the origin
    looking down (-z), the region points given."""

    goal_region_visible = IL.IORSLarge.goal_region_visible
    partner_points_in_view = IL.IORSLarge.partner_points_in_view

    def __init__(self, points, **gen):
        self.gp = _GP(**gen)
        self.world = SimpleNamespace(camera_model=G.CameraModel(640, 480, 320.0, 320.0, 320.0, 240.0))
        self._points = th.tensor(points, dtype=th.float32)

    def camera_pose_at(self, stance):
        return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def goal_region_points(self, from_xy):
        return self._points


def test_goal_region_visible_with_part_of_the_partner_in_frame():
    st = SimpleNamespace(focus=th.tensor([0.0, 0.0, 0.0]), base_pose2d=(0.0, 0.0, 0.0))
    # 5 top-patch points in frame + 3 bottom-edge points outside (u > 640 at x = 2 m, 1 m below the camera)
    pts = [[0.0, 0.0, -1.0], [0.1, 0.1, -1.0], [0.1, -0.1, -1.0], [-0.1, 0.1, -1.0], [-0.1, -0.1, -1.0], [2.0, 0.0, -1.0], [2.0, 0.5, -1.0], [2.0, -0.5, -1.0]]
    assert not _ViewStub(pts).goal_region_visible(st)  # today: every point
    assert _ViewStub(pts, partner_view_min_points=1).goal_region_visible(st)
    assert _ViewStub(pts, partner_view_min_points=5).goal_region_visible(st) and not _ViewStub(pts, partner_view_min_points=6).goal_region_visible(st)
    # the 5 % margin is the one of P0.in_frame: a point 3 % inside the left border does not count
    edge = [[-1.0 + 0.03 * 2.0, 0.0, -1.0]]
    assert not _ViewStub(edge, partner_view_min_points=1).goal_region_visible(st)
    # nothing in frame (behind the camera) never passes
    assert not _ViewStub([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], partner_view_min_points=1).goal_region_visible(st)
    assert _ViewStub([[0.0, 0.0, -1.0]] * 3).goal_region_visible(st)


# ----------------------------------------------------------------------------------------------------------------
# 2b: init-support geodesic cap and the run-time fallback (generation.init_support_max_geodesic_m / init_support_fallback)
# ----------------------------------------------------------------------------------------------------------------


class _SurfaceStub:
    """What :meth:`IORSLarge.surface_candidates` / :meth:`_prepare_bindings` read; ``distances`` = support name -> geodesic to the
    partner (None = no path); ``try_surface`` records the class it was called with."""

    surface_candidates = IL.IORSLarge.surface_candidates
    _support_near_partner = IL.IORSLarge._support_near_partner
    init_support_fallback = IL.IORSLarge.init_support_fallback
    num = IL.IORSLarge.num
    _prepare_bindings = IL.IORSLarge._prepare_bindings

    def __init__(self, relation, support, distances, categories=None, **gen):
        self.subtask = f"inter_object_relation_satisfaction-{relation}_large_scale"
        self.relation = relation
        self.gp = _GP(alias=f"{self.subtask}-mouse-desk-countertop", **gen)
        self.partner = SimpleNamespace(name="desk_iotfzl_0", category="desk", in_rooms=["living_room_0"])
        self.floor = SimpleNamespace(name="floors_abc_0", aabb=(th.zeros(3), th.zeros(3)))
        self.target = SimpleNamespace(name="mouse.n.01_1")
        cats = categories or {}
        objs = {n: SimpleNamespace(name=n, category=cats.get(n, "countertop"), in_rooms=["living_room_0"]) for n in distances}

        def obj(name):
            if name not in objs:
                raise KeyError(name)
            return objs[name]

        self.world = SimpleNamespace(obj=obj, scene=SimpleNamespace(objects=list(objs.values()) + [self.partner]), scene_object_names=set(objs) | {self.partner.name},
                                     rooms=["living_room_0"], local_hull=lambda t: None)
        self.choice = SimpleNamespace(bindings={"support": support})
        self.ctx = SimpleNamespace(base_seed=0)
        self.template_facts, self._dist, self.tried, self.roles = {}, dict(distances), [], {}
        self.bindings = SimpleNamespace(one=lambda role: role)

    def obj(self, role):
        return self.floor

    def geodesic_between(self, a, b):
        return self._dist.get(a.name)

    def make_band_spec(self):
        return None

    def prepare_partner(self):
        pass

    def try_surface(self, kind, cand, rng):
        self.tried.append((kind, cand.name))

    def alias_role(self, role, bound):
        self.roles[role] = bound

    def bind_scene_object(self, role, obj):
        self.roles[role] = obj.name


def test_init_support_geodesic_cap_named_and_by_category():
    d = {"countertop_ikwqer_0": 2.1, "countertop_sjxber_0": 12.4, "countertop_far_0": None}
    # default: named supports as bound (no geodesic rule), a missing name raises as before
    kind, out = _SurfaceStub("ontop", ["countertop_ikwqer_0", "countertop_sjxber_0"], d).surface_candidates()
    assert kind == "support" and [o.name for o in out] == ["countertop_ikwqer_0", "countertop_sjxber_0"]
    with pytest.raises(KeyError):
        _SurfaceStub("ontop", ["countertop_missing_0"], d).surface_candidates()
    # the cap: too far / no geodesic / missing from the scene are dropped and noted
    s = _SurfaceStub("ontop", ["countertop_ikwqer_0", "countertop_sjxber_0", "countertop_far_0", "countertop_missing_0"], d, init_support_max_geodesic_m=10.0)
    kind, out = s.surface_candidates()
    assert [o.name for o in out] == ["countertop_ikwqer_0"]
    drops = s.template_facts["init_support_dropped"]
    assert drops["countertop_sjxber_0"].startswith("geodesic 12.40 m > 10.00 m") and drops["countertop_far_0"] == "no geodesic to the partner"
    assert drops["countertop_missing_0"] == "missing from the scene"
    # by category: max_partner_geodesic_m (3.0) unless the cap is set
    cat = {"category": ["countertop"]}
    assert [o.name for o in _SurfaceStub("ontop", cat, d).surface_candidates()[1]] == ["countertop_ikwqer_0"]
    assert [o.name for o in _SurfaceStub("ontop", cat, {**d, "countertop_sjxber_0": 5.0}, max_partner_geodesic_m=10.0).surface_candidates()[1]] == \
        ["countertop_ikwqer_0", "countertop_sjxber_0"]
    s = _SurfaceStub("ontop", cat, {**d, "countertop_sjxber_0": 5.0}, max_partner_geodesic_m=10.0, init_support_max_geodesic_m=4.0)
    assert [o.name for o in s.surface_candidates()[1]] == ["countertop_ikwqer_0"] and "countertop_sjxber_0" in s.template_facts["init_support_dropped"]
    assert "init_support_dropped" not in _SurfaceStub("ontop", cat, d).template_facts


def test_init_support_fallback_floor_for_ontop_and_own_top_for_under(monkeypatch):
    from ispatialgym.sampling.sim_utils import Reject

    monkeypatch.setattr(IL.TT, "library_grasps", lambda s, role: ([], None, th.tensor([0.0, 0.0, 0.0, 1.0])))
    d = {"countertop_ikwqer_0": 12.0}
    # off (default): A5, no runtime class switching - the template is rejected
    s = _SurfaceStub("ontop", ["countertop_ikwqer_0"], d, init_support_max_geodesic_m=10.0)
    assert s.init_support_fallback("support") is None
    with pytest.raises(Reject) as e:
        s._prepare_bindings()
    assert e.value.reason == "no_init_surface" and s.tried == []
    # on: ontop -> the floor init, recorded with its reason
    s = _SurfaceStub("ontop", ["countertop_ikwqer_0"], d, init_support_max_geodesic_m=10.0, init_support_fallback=True)
    s._prepare_bindings()
    assert s.tried == [("floor", "floors_abc_0")] and s.roles["support"] == "floor"
    fb = s.template_facts["init_surface"]["fallback"]
    assert fb["from"] == "support" and fb["to"] == "floor" and "countertop_ikwqer_0: geodesic 12.00 m > 10.00 m" in fb["reason"]
    assert s.template_facts["init_surface"]["class"] == "floor" and s.init_fallback == fb
    assert s.template_facts["relaxations"] == {"init_support_max_geodesic_m": 10.0, "init_support_fallback": True}
    # under -> the partner's own top; a support missing from the scene counts as none left
    s = _SurfaceStub("under", ["countertop_missing_0"], d, init_support_max_geodesic_m=10.0, init_support_fallback=True)
    s._prepare_bindings()
    assert s.tried == [("partner", "desk_iotfzl_0")] and s.roles["support"] == "partner"
    assert s.template_facts["init_surface"]["fallback"] == {"from": "support", "to": "partner", "reason": "countertop_missing_0: missing from the scene"}
    # a support within the cap is used as before, no fallback recorded
    s = _SurfaceStub("ontop", ["countertop_ikwqer_0"], {"countertop_ikwqer_0": 2.1}, init_support_max_geodesic_m=10.0, init_support_fallback=True)
    s._prepare_bindings()
    assert s.tried == [("support", "countertop_ikwqer_0")] and "fallback" not in s.template_facts["init_surface"] and s.init_fallback is None
    # the floor / own-top classes never fall back (they always have their one candidate)
    assert s.init_support_fallback("floor") is None and s.init_support_fallback("partner") is None


# ----------------------------------------------------------------------------------------------------------------
# 2c: the place-under (generation.under_mode != "push") end to end
# ----------------------------------------------------------------------------------------------------------------


def test_a_place_under_plans_around_the_partner_and_leaves_out_the_floor(tmp_path):
    """The oracle's place ``under`` the partner (``program_iors_large``'s place-under: kind support, ``on`` = the partner) leaves the
    partner's support - the floor the object is set on - out of the planner's world and keeps the partner; ``on`` / ``into`` places
    still leave out the partner they are set on, only that."""
    from dataclasses import replace

    from ispatialgym.oracle.actions import Place
    from ispatialgym.oracle.programs import expand
    from ispatialgym.tests.test_v2_oracle import _place_goal, iors_world, make_executor, success_eval

    ignored = {}
    for relation in ("on", "under"):
        sc, env = iors_world()
        env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
        ex, motions, backend, _ = make_executor(env, sc, tmp_path / relation)
        floor = SimpleNamespace(name="floors_x_0", category="floors")
        ex.state.contact_supports = lambda name, margin=0.10, depth=2, _f=floor: [_f]
        actions = [replace(a, relation=relation) if isinstance(a, Place) else a for a in expand(sc.alias, sc)]
        result = ex.run_program(actions)
        assert result["success"], result["verdict"]
        ignored[relation] = ex.action_records[[a.name for a in actions].index("place")].notes["plan_ignored"]
    assert ignored == {"on": ["plate_pkkgzc_0"], "under": ["floors_x_0"]}


def test_every_under_variant_takes_the_floor_mesh_out():
    class S:
        _floor_relaxed = IL.IORSLarge._floor_relaxed

        def __init__(self, relation, init_class):
            self.relation, self.surface = relation, {"kind": init_class}
            self.ikc = SimpleNamespace(ignoring=lambda objs: ("ignoring", [o.name for o in objs]))

        def obj(self, role):
            return SimpleNamespace(name="floors_x_0")

    assert S("under", "support")._floor_relaxed() == ("ignoring", ["floors_x_0"])  # place-under from a support (before: the floor stayed)
    assert S("under", "partner")._floor_relaxed() == ("ignoring", ["floors_x_0"])
    assert S("ontop", "floor")._floor_relaxed() == ("ignoring", ["floors_x_0"])  # the floor init, as before
    assert isinstance(S("ontop", "support")._floor_relaxed(), IL._NullContext)  # ontop from a support, as before
