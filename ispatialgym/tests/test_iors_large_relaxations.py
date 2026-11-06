"""Simulator-free tests of the large-scale IORS relaxation keys (owner 2026-10-12 item 2, the large-scale eval picks; spec §5.6
"Relaxation keys"): every key unset keeps the rule as before (``pytest ispatialgym/tests/test_iors_large_relaxations.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import geometry as G

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


class _GP:
    def __init__(self, variant: str = "inter_object_relation_satisfaction-ontop_large_scale-mouse-desk-floor", **gen):
        self.variant, self.generation = variant, dict(gen)

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
    bind_init_surface = IL.IORSLarge.bind_init_surface

    def __init__(self, relation, support, distances, categories=None, **gen):
        self.subtask = f"inter_object_relation_satisfaction-{relation}_large_scale"
        self.relation = relation
        self.gp = _GP(variant=f"{self.subtask}-mouse-desk-countertop", **gen)
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

    def variant_role(self, role, bound):
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


def _kitchen_kit():
    """OmniGibson's own traversability map (0.1 m cells, world x = (column - 40) * 0.1, y = (row - 40) * 0.1; the R1Pro erosion: 7 cells)
    of a 6 m kitchen: an L-shaped counter run along the south and west walls (south arm y < -2.2, west arm x < -2.2), an island, and a
    closet east of the wall at x = 3.1..3.4. The kit's geodesic is ``RobotKit.geodesic``'s call."""
    from omnigibson.maps.traversable_map import TraversableMap

    def cells(x0, x1, y0, y1):
        return slice(int(round(y0 / 0.1)) + 40, int(round(y1 / 0.1)) + 40), slice(int(round(x0 / 0.1)) + 40, int(round(x1 / 0.1)) + 40)

    floor = th.zeros(80, 80, dtype=th.uint8)
    floor[cells(-3.0, 3.1, -3.0, 3.1)] = 255
    floor[cells(3.4, 5.4, -3.0, 3.1)] = 255
    for box in ((-3.0, 3.1, -3.0, -2.2), (-3.0, -2.2, -3.0, 1.0), (-1.0, 2.0, -0.5, 0.5), (4.6, 5.2, 2.0, 2.5)):
        floor[cells(*box)] = 0
    trav = TraversableMap(map_resolution=0.1)
    trav.map_size, trav.floor_map, trav.floor_heights = 80, [floor], [0.0]
    robot = SimpleNamespace(reset_joint_pos_aabb_extent=th.tensor([0.728, 0.684, 1.478]))
    world = SimpleNamespace(scene=SimpleNamespace(trav_map=trav))

    def geodesic(a, b):
        path, dist = trav.get_shortest_path(0, G.t(a)[:2], G.t(b)[:2], entire_path=True, robot=robot)
        return path, (None if dist is None else float(dist))

    return SimpleNamespace(world=world, robot=robot, eroded_trav_map=lambda floor=0: trav._erode_trav_map(th.clone(trav.floor_map[0]), robot=robot),
                           geodesic=geodesic)


class _GeoStub:
    geodesic_between = IL.IORSLarge.geodesic_between
    _geodesic = IL.IORSLarge._geodesic
    free_cell_near = IL.IORSLarge.free_cell_near
    near_side_point = IL.IORSLarge.near_side_point

    def __init__(self, kit):
        self.kit = kit


def _obj(name, lo, hi):
    lo, hi = th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)
    return SimpleNamespace(name=name, aabb=(lo, hi), aabb_center=(lo + hi) / 2)


def test_geodesic_between_moves_off_map_anchors_onto_the_eroded_map():
    """inside_large#03: the drop-in sink sits in the counter run, so the counter's anchor (the sink's centre + 0.5 m) is on the counter,
    the L-joined counter's anchor on the south arm, the sink's anchor toward it in the erosion band - OmniGibson's A* has no path and the
    2b cap dropped both counters as "no geodesic". Anchors on free floor keep the plain geodesic; disconnected floor stays None."""
    kit = _kitchen_kit()
    s = _GeoStub(kit)
    sink = _obj("drop_in_sink_awvzkn_0", (-0.4, -2.9, 0.7), (0.4, -2.35, 1.1))
    south, west = _obj("countertop_kelzer_0", (-2.2, -3.0, 0.85), (3.0, -2.2, 0.89)), _obj("countertop_kelker_0", (-3.0, -3.0, 0.85), (-2.2, 1.0, 0.89))
    island, shelf = _obj("bar_udatjt_0", (-1.0, -0.5, 0.0), (2.0, 0.5, 1.05)), _obj("shelf_0", (4.6, 2.0, 0.0), (5.2, 2.5, 1.5))

    def plain(a, b):
        return kit.geodesic(s.near_side_point(a, b.aabb_center[:2]), s.near_side_point(b, a.aabb_center[:2]))[1]

    assert plain(south, sink) is None and plain(west, sink) is None  # the bug: both counters had no geodesic
    d_south, d_west = s.geodesic_between(south, sink), s.geodesic_between(west, sink)
    assert d_south is not None and d_south < 1.0 and d_west is not None and d_west < 1.5
    assert s.geodesic_between(island, sink) == pytest.approx(float(plain(island, sink)))  # anchors on the map: unchanged
    assert s.geodesic_between(shelf, sink) is None  # the closet's floor does not connect to the kitchen
    # the snapped end is the centre of a free cell of the eroded map, which world_to_map takes back to that cell
    c = s.free_cell_near(th.tensor([0.5, -2.625]))
    free = kit.eroded_trav_map() == 255
    assert bool(free[tuple(int(v) for v in kit.world.scene.trav_map.world_to_map(c).tolist())]) and float(c[1]) > -2.2 + 0.3
    # the 2b cap keeps the counters now
    ns = SimpleNamespace(support_drops={}, partner=sink, geodesic_between=s.geodesic_between)
    assert IL.IORSLarge._support_near_partner(ns, south, 10.0) and IL.IORSLarge._support_near_partner(ns, west, 10.0) and ns.support_drops == {}


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
        actions = [replace(a, relation=relation) if isinstance(a, Place) else a for a in expand(sc.variant, sc)]
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


class _PlaceUnderStub:
    """The place-under stance C search (:meth:`IORSLarge._placement_candidates_at`, ``under_mode`` "place") on a 1.2 x 0.6 m desk with
    no band (every grid trunk), every base free, the near-edge line in view: only the optical-axis cone decides."""

    _placement_candidates_at = IL.IORSLarge._placement_candidates_at
    band_cells_world = IL.IORSLarge.band_cells_world

    def __init__(self, **gen):
        self.gp = _GP(variant="inter_object_relation_satisfaction-under_large_scale-trash_can-desk-floor", under_mode="place", **gen)
        self.relation, self.seg_arm, self.push_under, self.floor_z = "under", "right", False, 0.0
        self.partner = SimpleNamespace(name="desk_mdhelw_1", aabb=(th.tensor([0.0, 0.0, 0.0]), th.tensor([1.2, 0.6, 0.75])))
        self.place = {"half": 0.15, "height": 0.28, "cells": th.tensor([[0.6, 0.3]])}
        self.kit = SimpleNamespace(trav_free=lambda xy, eroded=False: True)
        self.world = SimpleNamespace(rooms=None)
        self.template_facts = {}

    def band(self, kind, support_obj=None, under_partner=None, arm=None):
        return None

    def partner_points_in_view(self, cam, points):
        return True

    def debug_placement_visibility(self, cands, cams, edge, init_xy):
        pass


def test_place_under_stance_c_axis_cone_key(monkeypatch):
    import math

    monkeypatch.setattr(IL.P0, "body_free", lambda s, cands, **kw: th.ones(len(cands), dtype=th.bool))
    monkeypatch.setattr(IL.P0, "camera_poses", lambda s, cands: [(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))] * len(cands))
    monkeypatch.setattr(IL.P0, "axis_angle", lambda s, cam, point: math.radians(30.0))  # the hidden goal cell 30° off the optical axis
    edge = next(e for e in IL.aabb_edges([0.0, 0.0, 0.0], [1.2, 0.6, 0.75]) if e.name == "-y")
    assert _PlaceUnderStub()._placement_candidates_at(edge, (0.6, -1.0)) == []  # default: within 20°, as before
    out = _PlaceUnderStub(under_axis_max_deg=45.0)._placement_candidates_at(edge, (0.6, -1.0))
    assert out and all(st.extras["angle_class"] == 1 and st.extras["axis_angle_deg"] == pytest.approx(30.0) for st in out)
    assert all(st.extras["goal_cell"] == pytest.approx([0.6, 0.3]) for st in out)
    assert _PlaceUnderStub(under_axis_max_deg=25.0)._placement_candidates_at(edge, (0.6, -1.0)) == []
    assert "under_axis_max_deg" in IL.RELAXATION_KEYS
