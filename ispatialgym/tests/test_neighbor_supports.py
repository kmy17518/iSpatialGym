"""Simulator-free tests of neighbouring supports (owner decision A4, 2026-10-12): the sheet encoding (``neighbors=`` on a support row,
``ground_isg``), the union of the tops (``sampling/v2/neighbors.py``: a footprint on one top, across a flush seam, never across a step),
the per-support bands, the hooks of the tabletop phases and the sampler, and that a support without neighbours is unchanged."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

RES = 0.02


def _top(name, x0, nx, z, ny=30, y0=0.0, free=None):
    """A ``SupportTop`` of ``nx`` x ``ny`` 2 cm cells from (x0, y0), every cell exposed (``free``: an explicit free mask)."""
    from ispatialgym.sampling.sim_utils import SupportTop

    xs = x0 + RES / 2 + RES * th.arange(nx, dtype=th.float32)
    ys = y0 + RES / 2 + RES * th.arange(ny, dtype=th.float32)
    mask = th.ones(nx, ny, dtype=th.bool) if free is None else free.clone()
    return SupportTop(name, xs, ys, mask, th.zeros(nx, ny, dtype=th.bool), z, 0.0, [], RES)


def _square(cx, cy, h=0.05):
    return th.tensor([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


A, B = "table_a_0", "table_b_0"


def _pair(z_b):
    """Table A at x in [0, 0.4), table B beside it at x in [0.4, 0.8) (the seam at x = 0.4), both y in [0, 0.6); A at 0.90 m."""
    return _top(A, 0.0, 20, 0.90), _top(B, 0.4, 20, z_b)


# ----------------------------------------------------------------------------------------------------------------
# the sheet encoding -> customs
# ----------------------------------------------------------------------------------------------------------------


def test_neighbors_note_key_is_parsed_in_order_without_duplicates_or_the_row_itself():
    assert GI.parse_neighbors("init=ontop; top_z=0.89 rooms=kitchen_0 neighbors=countertop_kelzer_0 src=B4") == ["countertop_kelzer_0"]
    assert GI.parse_neighbors("top_z=0.91 neighbors=oven_ffitak_0,bar_byvbuc_0; hob on the oven") == ["oven_ffitak_0", "bar_byvbuc_0"]
    assert GI.parse_neighbors("neighbours=b_0,a_0,b_0,burner_x_0", own="burner_x_0") == ["b_0", "a_0"]
    assert GI.parse_neighbors("init=ontop; top_z=0.89 rooms=kitchen_0") == [] and GI.parse_neighbors("") == [] and GI.parse_neighbors(None) == []
    assert GI.parse_neighbors("no_neighbors=x_0") == []  # a key that merely ends in the word is not the key


def test_support_surface_entry_carries_the_neighbors_and_is_unchanged_without():
    row = {"scene": "house_single_floor", "name": "countertop_kelker_0", "note": "init=ontop; top_z=0.89 area=1.44 rooms=kitchen_0 src=B4; counter"}
    plain = GI.support_surface(row, "p#04~T1")
    assert plain == {"name": "countertop_kelker_0", "init": "ontop", "top_z": 0.89, "row": "p#04~T1"}
    paired = GI.support_surface(dict(row, note=row["note"].replace("rooms=kitchen_0", "rooms=kitchen_0 neighbors=countertop_kelzer_0")), "p#04~T1")
    assert paired == {**plain, "neighbors": ["countertop_kelzer_0"]}
    # a support of SUPPORT_TOP_SURFACE keeps its surface key next to the neighbours
    scene, name = sorted(GI.SUPPORT_TOP_SURFACE)[0]
    s = GI.support_surface({"scene": scene, "name": name, "note": "init=ontop; top_z=0.90 neighbors=x_0"}, "p")
    assert s["surface"] == "top" and s["neighbors"] == ["x_0"]


def test_the_candidate_record_names_the_neighbors_only_when_the_entry_has_them():
    from ispatialgym.sampling.v2.problem import GroundedProblem, subtask_of_alias

    alias = "inter_object_relation_satisfaction-inside-garlic_clove-saucepot-countertop"
    surfaces = [{"name": "countertop_kelzer_0", "init": "ontop", "top_z": 0.89, "row": "p#04~T2"},
                {"name": "countertop_kelker_0", "init": "ontop", "top_z": 0.89, "row": "p#04~T1", "neighbors": ["countertop_kelzer_0"]}]
    entry = {"iddl": "inter_object_relation_satisfaction/inside.iddl", "definition_id": 62,
             "roles": {"target": {"category": "garlic_clove", "sampleable": True}, "partner": {"category": "saucepot", "sampleable": True},
                       "support": {"category": "countertop", "sampleable": False}, "floor": {"sampleable": False}},
             "scenes": {"house_single_floor": {"rooms": ["kitchen_0"], "bindings": {"support": ["countertop_kelzer_0", "countertop_kelker_0"],
                                                                                   "support_surfaces": surfaces, "floor": {"room": "kitchen_0"}},
                                               "whitelist": {}, "blacklist": {}}},
             "generation": {}, "constraints": []}
    gp = GroundedProblem(alias, copy.deepcopy(entry), subtask=subtask_of_alias(alias))
    rec = gp.override_support("house_single_floor", "countertop_kelker_0", rooms=["kitchen_0"])
    assert rec["neighbors"] == ["countertop_kelzer_0"] and gp.bindings("house_single_floor")["support_surfaces"] == [surfaces[1]]
    assert "neighbors" not in gp.override_support("house_single_floor", "countertop_kelzer_0", rooms=["kitchen_0"])


# ----------------------------------------------------------------------------------------------------------------
# the union of the tops
# ----------------------------------------------------------------------------------------------------------------


def test_a_footprint_lies_on_one_top_or_across_a_flush_seam():
    a, b = _pair(0.905)  # 5 mm higher: flush within the 1 cm tolerance
    u = NB.UnionTop.build([a, b])
    assert u.name == A and u.z == 0.90 and int(u.free.sum()) == int(a.free.sum()) + int(b.free.sum())
    assert u.polygon_supported(_square(0.20, 0.30)) and u.polygon_supported(_square(0.60, 0.30))  # on one top each
    straddle = _square(0.40, 0.30)
    assert not a.polygon_supported(straddle) and u.polygon_supported(straddle)  # across the flush seam
    assert not u.polygon_supported(_square(0.90, 0.30)) and not u.polygon_supported(_square(0.78, 0.30))  # off the tops / over B's far edge
    assert u.contains(th.tensor([0.40, 0.30]), 0.05) and u.contains([0.6, 0.3]) and not u.contains([0.85, 0.3])
    assert u.member_at([0.2, 0.3]) == 0 and u.member_at([0.6, 0.3]) == 1 and u.member_at([0.9, 0.3]) is None
    assert u.z_at([0.6, 0.3]) == 0.905 and u.z_under(straddle) == 0.905 and u.tops_under(straddle) == [A, B]


def test_a_footprint_never_straddles_a_step():
    a, b = _pair(0.82)  # 8 cm lower (the burner's neighbours)
    u = NB.UnionTop.build([a, b])
    assert u.polygon_supported(_square(0.20, 0.30)) and u.polygon_supported(_square(0.60, 0.30))
    assert not u.polygon_supported(_square(0.40, 0.30)) and not u.contains([0.40, 0.30], 0.02)
    assert u.contains([0.60, 0.30], 0.05) and u.z_at([0.6, 0.3]) == 0.82 and u.z_at([0.2, 0.3]) == 0.90 and u.z_at([5.0, 5.0]) == 0.90
    # the tolerance is a parameter (generation.neighbor_flush_tol_m)
    assert NB.UnionTop.build([a, b], flush_tol=0.10).polygon_supported(_square(0.40, 0.30))


def test_the_support_cells_keep_their_own_raster_and_occupied_cells_stay_occupied():
    free_a = th.ones(20, 30, dtype=th.bool)
    a, b = _pair(0.90)
    a.occupied = ~free_a
    a.occupied[5:8, 5:8] = True  # a pot on table A
    b.mask[:, :5] = False  # B's near strip is not exposed
    u = NB.UnionTop.build([a, b])
    assert th.equal(u.free[:20], a.free) and th.equal(u.free[20:], b.free) and th.equal(u.member[:20], th.zeros(20, 30, dtype=th.long))
    assert not u.contains([0.13, 0.13]) and not u.contains([0.60, 0.05]) and int((u.member == -1).sum()) == 20 * 5


def test_the_edges_are_the_supports_own_extended_along_their_line_over_the_neighbour():
    a, b = _pair(0.82)
    u = NB.UnionTop.build([a, b])
    own = {e[0]: e for e in a.edges()}
    ext = {e[0]: e for e in u.edges()}
    # +v / -v (lines y = 0.59 / 0.01) run along x: B beside A extends them over the seam
    for name in ("+v", "-v"):
        assert ext[name][3] == pytest.approx(0.78) and th.allclose(ext[name][1], th.tensor([0.40, float(own[name][1][1])]))
        assert th.equal(ext[name][2], own[name][2])
    # +u (line x = 0.39) and -u (x = 0.01) face across: unchanged (B's cells lie farther than the strip from -u, along +u's own extent)
    for name in ("+u", "-u"):
        assert ext[name][3] == pytest.approx(own[name][3]) and th.allclose(ext[name][1], own[name][1])
    # a neighbour far behind the support's front line does not extend it (the strip is 10 cm in front / 30 cm behind)
    far = _top("table_far_0", 0.4, 20, 0.90, ny=10, y0=0.5)
    ext2 = {e[0]: e for e in NB.UnionTop.build([a, far]).edges()}
    assert ext2["-v"][3] == pytest.approx(own["-v"][3]) and ext2["+v"][3] == pytest.approx(0.78)


def test_rest_on_and_grasp_from_follow_the_top_under_the_footprint():
    a, b = _pair(0.82)
    u = NB.UnionTop.build([a, b])
    pose = (th.tensor([0.6, 0.3, 0.953]), C.IDENTITY_QUAT.clone())
    moved, z = NB.rest_on(u, pose, _square(0.60, 0.30), 0.90)
    assert z == 0.82 and float(moved[0][2]) == pytest.approx(0.873) and moved[1] is pose[1]
    assert NB.grasp_from(u, _square(0.60, 0.30)) == B and NB.grasp_from(u, _square(0.20, 0.30)) is None
    rec = NS(extras={})
    NB.note_grasp_from(u, _square(0.60, 0.30), rec)
    assert rec.extras == {"grasp_from": B}


def test_neighbors_only_keeps_the_neighbour_cells_heights_and_tops():
    """``generation.start_region_on_neighbors`` (ontop_small#02 6d): the union with the support's own cells taken out - a footprint
    lies on the neighbour, never on the support or across the seam, even a flush one; heights and grasp_from as on the union."""
    a, b = _pair(0.905)
    u = NB.UnionTop.build([a, b])
    mask0 = u.mask.clone()
    n = u.neighbors_only()
    assert isinstance(n, NB.UnionTop) and n.name == A and n.z == 0.90 and n.members == u.members and th.equal(n.member, u.member)
    assert th.equal(n.free, u.free & (u.member == 1)) and int(n.free.sum()) == int(b.free.sum()) and th.equal(u.mask, mask0)  # the union untouched
    assert n.polygon_supported(_square(0.60, 0.30)) and not n.polygon_supported(_square(0.20, 0.30)) and not n.polygon_supported(_square(0.40, 0.30))
    assert u.polygon_supported(_square(0.40, 0.30))  # the union itself still takes the flush seam
    assert n.contains([0.60, 0.30], 0.05) and not n.contains([0.20, 0.30]) and not n.contains([0.42, 0.30], 0.05)
    assert n.z_at([0.6, 0.3]) == 0.905 and n.z_at([0.2, 0.3]) == 0.90 and NB.grasp_from(n, _square(0.60, 0.30)) == B


# ----------------------------------------------------------------------------------------------------------------
# tabletop.support_surface -> union_top
# ----------------------------------------------------------------------------------------------------------------


def _surface_sampler(surfaces, tops, floor_z=0.0):
    """A sampler for ``TT.support_surface``: ``support_top`` by name from ``tops``, ``world.obj`` raising ``KeyError`` for an unknown
    name, ``bindings.support_surfaces`` = ``surfaces``."""

    def obj(name):
        if name not in tops:
            raise KeyError(f"no object {name!r}")
        return NS(name=name)

    s = NS(choice=NS(bindings={"support_surfaces": surfaces}), gp=NS(alias="a"), template_facts={}, gen=lambda k, d=None: d)
    s.support_top = lambda support, ignore=(): tops[support.name]
    s.world = NS(floor_z=floor_z, spawned={}, obj=obj)
    return s


def test_a_neighbour_must_hold_something():
    a = _top(A, 0.0, 20, 0.90)
    strip = _top("oven_x_0", 0.0, 40, 0.84, ny=2, y0=-0.04)  # 80 free cells in a 4 cm strip (the oven front in front of its counter)
    assert NB.usable_cells(a, 0.04) == 16 * 26 and NB.usable_cells(strip, 0.04) == 0 and NB.usable_cells(strip, 0.0) == 80


def test_support_surface_unites_the_named_neighbours_and_reports_each():
    a, b = _pair(0.82)
    strip = _top("oven_x_0", 0.0, 40, 0.84, ny=2, y0=-0.04)
    tops = {A: a, B: b, "oven_x_0": strip}
    spec = {"name": A, "init": "ontop", "top_z": 0.90, "row": "p#04~T4", "neighbors": ["oven_x_0", B, "bar_missing_0"]}
    s = _surface_sampler([spec], tops)
    u = TT.support_surface(s, NS(name=A), height=0.1)
    assert isinstance(u, NB.UnionTop) and [m.name for m in u.members] == [A, B] and s.surface_z is None
    facts = {f["name"]: f for f in s.template_facts["support_surface"]["neighbors"]}
    assert facts[B]["used"] and facts[B]["step_m"] == pytest.approx(-0.08) and not facts[B]["flush"] and facts[B]["n_usable_cells"] == 416
    assert not facts["oven_x_0"]["used"] and facts["oven_x_0"]["n_free_cells"] == 80 and facts["oven_x_0"]["n_usable_cells"] == 0
    assert not facts["bar_missing_0"]["used"] and "no top raster" in facts["bar_missing_0"]["why"]
    assert [o.name for o in s.neighbors_of[A]] == [B] and [o.name for o in NB.neighbor_objects(s)] == [B]
    # a later call for a support without neighbours forgets them
    s.choice.bindings["support_surfaces"] = [{"name": A, "init": "ontop", "top_z": 0.90, "row": "p#04~T4"}]
    assert TT.support_surface(s, NS(name=A)) is a and s.neighbors_of == {} and NB.neighbor_objects(s) == []


def test_support_surface_without_neighbours_is_unchanged():
    a, b = _pair(0.90)
    for surfaces in ([], [{"name": A, "init": "ontop", "top_z": 0.90, "row": "p#01~T1"}], [{"name": A, "init": "ontop", "top_z": 0.90, "neighbors": []}],
                     [{"name": B, "init": "ontop", "top_z": 0.90, "neighbors": [A]}]):  # another support's neighbours
        s = _surface_sampler(surfaces, {A: a, B: b})
        assert TT.support_surface(s, NS(name=A)) is a
        assert not hasattr(s, "neighbors_of") and "neighbors" not in s.template_facts["support_surface"]
    # an only neighbour that cannot be used: the support's own top
    s = _surface_sampler([{"name": A, "init": "ontop", "top_z": 0.90, "neighbors": ["nope_0"]}], {A: a})
    assert TT.support_surface(s, NS(name=A)) is a and not getattr(s, "neighbors_of", None)


def test_an_interior_surface_is_never_united():
    class _Interior:
        def __init__(self):
            self.name, self.z = A, 0.78
            self.mask = th.ones(4, 4, dtype=th.bool)
            self.occupied = th.zeros(4, 4, dtype=th.bool)
            self.headroom = th.full((4, 4), 0.4)
            self.open_edges = ["-v"]

        @property
        def free(self):
            return self.mask & ~self.occupied

    a, b = _pair(0.90)
    it = _Interior()
    s = _surface_sampler([{"name": A, "init": "inside", "top_z": 0.73, "neighbors": [B]}], {A: a, B: b}, floor_z=0.05)
    s.world.support_interior = lambda support, z, resolution=0.02, ignore=(): it
    assert TT.support_surface(s, NS(name=A), height=0.1) is it and not getattr(s, "neighbors_of", None)
    assert s.template_facts["support_surface"]["neighbors"] == [{"name": B, "used": False, "why": "interior surface"}]


# ----------------------------------------------------------------------------------------------------------------
# bands: per support, each read on its own top
# ----------------------------------------------------------------------------------------------------------------


def _band(cells, arm="right", name="x"):
    cells = np.asarray(cells, dtype=np.float32)
    return RB.ReachBands({"lattice": 0.08, "grasp_names": [], "arm": arm, "support": name}, {(0.5, 0.0): cells.copy()}, {}, None)


STANCE = NS(translate=0.5, tilt=0.0, base_pose2d=(0.0, 0.0, 0.0), stand_off=0.45, focus=None)
CELLS = [[0.2, 0.3], [0.6, 0.3], [0.9, 0.3]]  # on A, on B, off both (base frame = world at the identity stance)


def test_each_band_is_read_on_its_own_top():
    a, b = _pair(0.82)
    u = NB.UnionTop.build([a, b])
    nb = NB.NeighborBands(u, [_band(CELLS, name=A), _band(CELLS, name=B)])
    assert nb.meta["support"] == A and nb.arm == "right" and nb.path is None  # every other read is the support's own band
    on_tops = th.tensor([[0.2, 0.3], [0.6, 0.3]])
    assert th.allclose(TT.band_cells_world(nb, STANCE), on_tops)  # A's cell on A, B's cell on B, nothing off the tops
    assert th.allclose(TT.band_union_cells(nb, STANCE), on_tops)
    assert nb.member_for([0.6, 0.3]) is nb.members[1] and nb.member_for([0.9, 0.3]) is None
    assert TT.in_band(nb, STANCE, [0.6, 0.3])[0] and not TT.in_band(nb, STANCE, [0.9, 0.3])[0]
    # the feasible set of a focus on B goes to B's band: an empty F removes B's cell, A's band keeps all of its cells
    TT.store_feasible_set(nb, NS(**{**STANCE.__dict__, "focus": th.tensor([0.6, 0.3, 0.9])}), [])
    assert len(nb.members[1].cells[(0.5, 0.0)]) == 2 and len(nb.members[0].cells[(0.5, 0.0)]) == 3
    # the other arm: each top's own band of that arm
    left = nb.for_arm("left", lambda band: _band(CELLS, arm="left", name=band.meta["support"]))
    assert isinstance(left, NB.NeighborBands) and [bb.meta["support"] for bb in left.members] == [A, B] and left.arm == "left"
    assert nb.for_arm("right", lambda band: None) is nb


def test_ensure_bands_loads_each_neighbours_band_and_nothing_else_without_neighbours(monkeypatch):
    a, b = _pair(0.82)
    calls = []

    def load(scene, support, obj=None, gkey=None, **kw):
        calls.append((scene, support, obj, gkey, tuple(kw.get("scale") or ())))
        return _band(CELLS, name=support)

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(load))
    spec = TT.BandSpec(object="garlic_clove/xotobt", grasp_key="garlic_clove__xotobt__s", scale=[3.724] * 3)
    s = NS(world=NS(scene_model="house_single_floor"), bands=None, template_facts={})
    plain = TT.ensure_bands(s, spec, top=a)
    assert isinstance(plain, RB.ReachBands) and s.bands is plain and calls == [("house_single_floor", A, "garlic_clove/xotobt", "garlic_clove__xotobt__s", (3.724,) * 3)]
    assert "neighbor_bands" not in s.template_facts
    calls.clear()
    s = NS(world=NS(scene_model="house_single_floor"), bands=None, template_facts={})
    nb = TT.ensure_bands(s, spec, top=NB.UnionTop.build([a, b]))
    assert isinstance(nb, NB.NeighborBands) and [m.meta["support"] for m in nb.members] == [A, B]
    assert [c[1] for c in calls] == [A, B] and len({c[2:] for c in calls}) == 1  # same object / grasp key / scale for both supports
    assert [f["support"] for f in s.template_facts["neighbor_bands"]] == [A, B]


def test_compute_bands_of_a_union_is_the_supports_own():
    a, b = _pair(0.82)
    assert NB.own_top(NB.UnionTop.build([a, b])) is a and NB.own_top(a) is a


# ----------------------------------------------------------------------------------------------------------------
# placement heights, the :init check, support_objects
# ----------------------------------------------------------------------------------------------------------------


class _World:
    def footprint(self, hull, pose):
        pts = C.transform_points((th.as_tensor(pose[0]), th.as_tensor(pose[1])), hull)[:, :2]
        lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
        return th.tensor([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])

    def bottom_offset(self, hull, quat):
        return -float(C.transform_points((th.zeros(3), th.as_tensor(quat)), hull)[:, 2].min())


def _box(h):
    return th.tensor([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


def test_the_partner_rests_on_the_top_under_it():
    a, b = _pair(0.82)
    edge = EdgeFrame("-v", th.tensor([0.4, 0.01]), th.tensor([0.0, 1.0]), 0.78)
    stance = Stance("manipulation", (0.4, -0.44, math.pi / 2), 0.5, 0.0, 0.45, edge)
    t_pose = (th.tensor([0.30, 0.30, 0.953]), C.IDENTITY_QUAT.clone())
    s = NS(world=_World())
    args = (np.random.default_rng(0), t_pose, _box(0.03), _box(0.05), C.IDENTITY_QUAT.clone(), stance, -1.0, (0.10, 0.12))
    p = TT.place_partner(s, *args, top=NB.UnionTop.build([a, b]), yaw=0.0)  # robot's right = +x: onto B, 8 cm lower
    assert float(p[0][0]) > 0.42 and float(p[0][2]) == pytest.approx(0.82 + 0.05 + TT.Z_OFFSET, abs=1e-5)
    p = TT.place_partner(s, *args[:6], 1.0, args[7], top=NB.UnionTop.build([a, b]), yaw=0.0)  # robot's left = -x: on A
    assert float(p[0][0]) < 0.4 and float(p[0][2]) == pytest.approx(0.90 + 0.05 + TT.Z_OFFSET, abs=1e-5)
    with pytest.raises(Exception, match="partner_placement|no partner position"):  # A alone has no room on the right of the target
        TT.place_partner(s, *args, top=a, yaw=0.0)


class OnTop:
    pass


class Inside:
    pass


def test_ontop_the_support_holds_on_a_neighbour_top_only_for_a_support_with_neighbours():
    clove, a, b = NS(name="clove_0"), NS(name=A), NS(name=B)
    truth = {("OnTop", ("clove_0", B))}
    calls = []

    def evaluate(cls, *ents):
        calls.append((cls.__name__, tuple(e.name for e in ents)))
        return (cls.__name__, tuple(e.name for e in ents)) in truth

    s = NS(neighbors_of={A: [b]})
    assert NB.evaluate_on_tops(s, OnTop, [clove, a], evaluate) and calls == [("OnTop", ("clove_0", A)), ("OnTop", ("clove_0", B))]
    calls.clear()
    assert not NB.evaluate_on_tops(s, Inside, [clove, a], evaluate) and calls == [("Inside", ("clove_0", A))]
    calls.clear()
    assert not NB.evaluate_on_tops(NS(), OnTop, [clove, a], evaluate) and calls == [("OnTop", ("clove_0", A))]


def test_the_sampler_hooks_are_the_identity_without_neighbours(monkeypatch):
    from ispatialgym.sampling import predicates as PRED
    from ispatialgym.sampling.v2 import core

    seen = []
    monkeypatch.setattr(PRED, "evaluate_predicate", lambda cls, *ents: seen.append((cls, ents)) or ents[1].name == B)
    clove, a, b = NS(name="clove_0"), NS(name=A), NS(name=B)
    plain = NS(scope={"clove.n.03_1": clove, "countertop.n.01_1": a}, bindings=NS(has=lambda role: False))
    assert not core.SamplerV2._evaluate(plain, OnTop, "clove.n.03_1", "countertop.n.01_1") and seen == [(OnTop, (clove, a))]
    assert core.SamplerV2.support_objects(plain) == []
    paired = NS(scope=plain.scope, bindings=plain.bindings, neighbors_of={A: [b]})
    assert core.SamplerV2._evaluate(paired, OnTop, "clove.n.03_1", "countertop.n.01_1") and core.SamplerV2.support_objects(paired) == [b]
    a_top = _pair(0.90)[0]
    pose = (th.tensor([0.2, 0.3, 0.95]), C.IDENTITY_QUAT.clone())
    assert NB.z_at(a_top, [0.2, 0.3]) == 0.90 and NB.rest_on(a_top, pose, _square(0.2, 0.3), 0.90) == (pose, 0.90)
    assert NB.grasp_from(a_top, _square(0.2, 0.3)) is None and NB.with_neighbor_bands(NS(), "band", None, a_top, None) == "band"
