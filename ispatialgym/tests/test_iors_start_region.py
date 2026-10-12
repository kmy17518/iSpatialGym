"""IORS small start region (A1, owner 2026-10-12; spec §4.4 / §4.6 step 2): the Phase-1c edge rule, the per-cell gap intervals on a
synthetic table (a long edge keeps every band cell the reach rule admits, a short edge its middle part, touching the robot's left only),
the partner side fixed before the target is placed (eval: ``generation.partner_side``), the split-cone pool per side and the attempt's
partner draw. No simulator."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.geometry import CameraModel
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

ONTOP = "inter_object_relation_satisfaction-ontop_small_scale"
#: the foci of the synthetic Phase 2: one row 0.31 m deep, at the band's laterals
LATTICE = [round(float(y), 2) for y in np.arange(-0.56, 0.161, 0.08)]


def _box(hx, hy, h):
    return th.tensor([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0.0, h)])


class _Band:
    """One trunk's reach band: cells 0.60-1.00 m ahead of the base, λ in [-0.565, 0.165] (breakfast_table_rhjoby_0's, half a centimetre
    wider so that no rule ends exactly on a grid gap)."""

    meta: dict = {}

    def __init__(self, lo=-0.565, hi=0.165):
        self.cells = np.array([(x, y) for x in np.arange(0.60, 1.001, 0.08) for y in np.linspace(lo, hi, 10)])

    def band(self, translate, tilt, **kw):
        return self.cells


def _table(L, depth=0.6):
    """A top behind an edge at x = 1.0 seen from a robot at x = 0.55 facing +x (lateral + = +y), ``L`` long, centred on y = 0."""
    xs, ys = th.arange(1.0, 1.0 + depth + 1e-6, 0.02), th.arange(-L / 2, L / 2 + 1e-6, 0.02)
    top = SupportTop("table_x_0", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), 0.75, 0.0)
    return top, EdgeFrame.from_edge("-x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), L)


def _world():
    return types.SimpleNamespace(footprint=lambda h, pose: G.convex_hull_2d(G.local_points_world(h, pose)[:, :2]), hull_at=G.local_points_world,
                                 bottom_offset=lambda h, q: -float(G.local_points_world(h, (th.zeros(3), G.t(q)))[:, 2].min()), camera_model=None)


def _sampler(relation="ontop", L=3.0, gen=None, pair=None):
    """An ``IORSSmall`` without a simulator: a 0.10 m target, a 0.20 m partner (w_t 0.05, w_p 0.10, D_p 0.23) on the synthetic table."""
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.top, s.edge = _table(L)
    s.relation, s.pair, s.spec, s.gp, s.world = relation, dict(pair or {}), types.SimpleNamespace(generation=dict(gen or {})), types.SimpleNamespace(alias="a"), _world()
    s.hull = {"target": _box(0.05, 0.05, 0.05), "partner": _box(0.10, 0.10, 0.02)}
    s.rest = {"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}
    s.bands, s.arm_bands, s.segment_arms, s.template_facts = _Band(), {}, ["right"], {}
    s.shelf_start = s.shelf_partner_pose = s.region_side = s.partner_draw = None
    s.region_sides = []
    s.init_stance = Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, s.edge)
    s.init_stance.extras["margins"] = {"D_p": 0.23}
    s.camera_pose_at = lambda st: None
    return s


def _candidates(s, depth=0.31):
    out = []
    for y in LATTICE:
        c = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, s.edge)
        c.focus = th.tensor([1.0 + depth, y, 0.78])
        out.append(c)
    return out


def _regions(kept):
    return {round(float(c.focus[1]), 2): c.extras[TT.START_REGION] for c in kept}


# ---------------------------------------------------------------------------------------------------------------- Phase 1c edge rule


def test_phase1c_skips_only_edges_too_short_for_both_objects_and_draws_over_the_whole_edge(monkeypatch):
    monkeypatch.setattr(TT, "camera_fk", lambda s=None, hand_below=None: types.SimpleNamespace(z_at=lambda t: 1.3))
    monkeypatch.setattr(TT, "image_border_lateral", lambda *a, **k: (0.7, 0.7))
    monkeypatch.setattr(TT, "trunk_sweep_free", lambda s, st: True)
    s = types.SimpleNamespace(camera_pose_at=lambda st: None, kit=types.SimpleNamespace(trav_free=lambda xy: True, floor_z=0.0),
                              ikc=types.SimpleNamespace(stance_collision_free=lambda st: True))
    row = {"kind": "iors", "w_t": 0.05, "w_p": 0.10, "W_p": 0.20, "D_p_max": 0.23, "relation": "ontop"}  # L_min = 0.10 + 0.03 + 0.20 + 0.20

    def init(L):
        st = Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, _table(L)[1])
        st.extras.update({"lam_min": -0.56, "lam_max": 0.16, "d_ws": 0.3, "t_range": [0.5, 0.5]})
        return st

    top = types.SimpleNamespace(z=0.75)
    with pytest.raises(Reject) as e:
        TT.phase1c_lateral(s, init(0.52), row, np.random.default_rng(0), top=top)
    assert e.value.reason == "edge_too_short" and "2 w_t + g_min + 2 w_p + 2 m_end = 0.53" in str(e.value)
    lats = [TT.phase1c_lateral(s, init(0.60), row, np.random.default_rng(k), top=top).lateral for k in range(200)]
    assert min(lats) < -0.27 and max(lats) > 0.27 and max(abs(v) for v in lats) <= 0.30  # uniform over the whole 0.60 m edge


# ---------------------------------------------------------------------------------------------------------------- pure rules


def test_lateral_gap_masks_ends_and_reach_rules():
    gs = TT.gap_grid(0.03, 0.23)
    assert len(gs) == 21 and gs[0] == pytest.approx(0.03) and gs[-1] == pytest.approx(0.23)
    # partner (0.20 wide) right of a target spanning [-0.05, 0.05] on an edge whose ends are at +-0.40, 0.10 kept from each end
    m = TT.lateral_gap_masks(gs, -1.0, (-0.05, 0.05), (-0.10, 0.10), ends=(-0.40, 0.40), m_end=0.10)
    assert TT.longest_run(gs, m["ends"]) == (pytest.approx(0.03), pytest.approx(0.05))
    assert not TT.lateral_gap_masks(gs, -1.0, (0.22, 0.32), (-0.10, 0.10), ends=(-0.40, 0.40), m_end=0.10)["ends"].any()  # target too near the end
    # ontop / under / inside: the partner's centre a band lateral; nextto: the target's place position (origin + side * (g - place gap))
    centre = TT.lateral_gap_masks(gs, 1.0, (-0.08, 0.02), (-0.10, 0.10), band=(-0.56, 0.16), reach="partner_centre")["reach"]
    assert TT.longest_run(gs, centre) == (pytest.approx(0.03), pytest.approx(0.04))  # centre = 0.02 + g + 0.10 <= 0.16
    goal = TT.lateral_gap_masks(gs, -1.0, (-0.05, 0.05), (-0.10, 0.10), band=(-0.20, 0.16), reach="target_goal", t_origin=0.0, place_gap=0.02)["reach"]
    assert TT.longest_run(gs, goal) == (pytest.approx(0.03), pytest.approx(0.22))  # goal = -(g - 0.02) >= -0.20
    assert not TT.lateral_gap_masks(gs, 1.0, (0, 0.1), (-0.1, 0.1), band=None, reach="partner_centre")["reach"].any()  # no band cell there
    assert "reach" not in TT.lateral_gap_masks(gs, 1.0, (0, 0.1), (-0.1, 0.1), band=(-1, 1), reach=None)  # touching: the push-stance screen
    assert TT.longest_run([0.1, 0.2, 0.3, 0.4, 0.5], [True, False, True, True, False]) == (0.3, 0.4) and TT.longest_run([0.1], [False]) is None


def test_polygons_supported_matches_the_raster_test():
    xs, ys = th.arange(0.0, 1.0, 0.02), th.arange(0.0, 0.6, 0.02)
    occupied = th.zeros(len(xs), len(ys), dtype=th.bool)
    occupied[20:24, 10:14] = True
    top = SupportTop("t", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), occupied, 0.8, 0.0)
    rng = np.random.default_rng(3)
    polys = []
    for _ in range(200):
        c, r, a = rng.uniform([-0.1, -0.1], [1.1, 0.7]), rng.uniform(0.02, 0.15), rng.uniform(0, math.pi)
        polys.append(th.tensor([[c[0] + r * math.cos(a + k * math.pi / 2.5), c[1] + r * math.sin(a + k * math.pi / 2.5)] for k in range(5)], dtype=th.float32))
    got = TT.polygons_supported(top, th.stack(polys), 0.02)
    assert got.tolist() == [top.polygon_supported(p, margin=0.02) for p in polys] and 0 < int(got.sum()) < len(polys)


# ---------------------------------------------------------------------------------------------------------------- start region


def test_start_region_long_edge_keeps_the_band_the_reach_rule_admits():
    s = _sampler(L=3.0)
    reg = _regions(s.start_region(_candidates(s)))
    assert sorted(reg) == LATTICE  # every band cell starts a partner on some side
    # ontop: the partner's centre a band lateral (λ <= 0.165 / >= -0.565): left of the target y + 0.05 + g + 0.10, right y - 0.15 - g
    assert sorted(y for y, r in reg.items() if "left" in r) == [y for y in LATTICE if y <= -0.015]
    assert sorted(y for y, r in reg.items() if "right" in r) == [y for y in LATTICE if y >= -0.385]
    assert reg[0.0]["right"] == pytest.approx([0.03, 0.23]) and reg[-0.08]["left"] == pytest.approx([0.03, 0.09])
    facts = s.template_facts["start_region"]
    assert facts["cells"] == {"left": 7, "right": 7} and facts["sides"] == ["left", "right"] and s.region_sides == [1.0, -1.0]
    assert facts["edge_length_m"] == 3.0 and facts["assigned"] is None


def test_start_region_short_edge_keeps_its_middle_part():
    s = _sampler(L=0.81)  # ends at +-0.405: both footprints within [-0.305, 0.305]
    reg = _regions(s.start_region(_candidates(s)))
    assert sorted(reg) == [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16]  # the band's right part (-0.56 .. -0.32) is off the middle of the edge
    assert sorted(y for y, r in reg.items() if "left" in r) == [-0.24, -0.16, -0.08]
    assert sorted(y for y, r in reg.items() if "right" in r) == [0.0, 0.08, 0.16]
    assert reg[0.0]["right"] == pytest.approx([0.03, 0.05])  # the partner's far end at -0.05 - g - 0.20 >= -0.305
    assert reg[-0.24]["left"] == pytest.approx([0.03, 0.23]) and reg[-0.08]["left"] == pytest.approx([0.03, 0.09])
    # an edge exactly L_min long has room for both objects at one target position only - none of these cells
    exact = _sampler(L=0.53)
    assert exact.start_region(_candidates(exact)) == [] and exact.region_sides == []


def test_start_region_records_the_arms_whose_band_meets_the_reach_rule():
    """Both arms offered for the segment (the left one with the mirrored band, λ in [-0.165, 0.565]): a side counts for an arm only when
    that arm's band holds the partner's centre - the attempt then offers just those arms."""
    s = _sampler(L=3.0)
    s.segment_arms, s.arm_bands = ["right", "left"], {"right": _Band(), "left": _Band(-0.165, 0.565)}
    cands = _candidates(s)
    for c in cands:
        c.extras["focus_arms"] = ["right", "left"]
    kept = {round(float(c.focus[1]), 2): c for c in s.start_region(cands)}
    assert kept[-0.24].extras[IS.REGION_ARMS] == {"left": ["right", "left"], "right": ["right"]}  # the left arm cannot place on the far right
    assert kept[0.16].extras[IS.REGION_ARMS] == {"left": ["left"], "right": ["right", "left"]}
    assert kept[0.16].extras[TT.START_REGION]["left"] == pytest.approx([0.03, 0.23])  # the left arm reaches the whole gap range there


def test_start_region_touching_is_the_robots_left_only():
    t = _sampler("touching", L=0.81, pair={"g_NT": 0.05})
    kept = t.start_region(_candidates(t))
    assert all(set(c.extras[TT.START_REGION]) == {"left"} for c in kept) and t.region_sides == [1.0]
    # no band cut (its push-stance screen judges the draw): the end rule alone, y + 0.05 + g + 0.20 <= 0.305
    assert sorted(_regions(kept)) == [-0.24, -0.16, -0.08, 0.0]
    long = _sampler("touching", L=3.0, pair={"g_NT": 0.05})
    assert sorted(_regions(long.start_region(_candidates(long)))) == LATTICE


# ---------------------------------------------------------------------------------------------------------------- side choice


def test_side_is_fixed_before_the_target_eval_assigned_train_uniform(monkeypatch):
    assert IS.partner_sides("touching") == [1.0] and IS.partner_sides("ontop") == [1.0, -1.0]
    assert IS.region_sides([1.0, -1.0], None, {1.0: 3, -1.0: 0}) == [1.0] and IS.region_sides([1.0, -1.0], None, {1.0: 2, -1.0: 5}) == [1.0, -1.0]
    assert IS.region_sides([1.0, -1.0], -1.0, {1.0: 9, -1.0: 0}) == [] and IS.region_sides([1.0, -1.0], -1.0, {-1.0: 1}) == [-1.0]
    # eval: generation.partner_side alone, whose region decides the template
    e = _sampler(L=0.81, gen={"partner_side": "right"})
    kept = e.start_region(_candidates(e))
    assert e.region_sides == [-1.0] and kept and all(set(c.extras[TT.START_REGION]) == {"right"} for c in kept)
    assert e.template_facts["start_region"]["assigned"] == "right" and e.template_facts["start_region"]["cells"] == {"right": 3}
    with pytest.raises(Reject) as r:
        _sampler("touching", gen={"partner_side": "right"}, pair={"g_NT": 0.05}).template_sides()
    assert r.value.reason == "partner_side"
    with pytest.raises(Reject) as r:
        _sampler("nextto", gen={"partner_side": "right"}, pair={"g_NT": 0.05, "handedness": {"target_side": "right"}}).template_sides()
    assert r.value.reason == "partner_side"  # the right shoe ends right of the left one: its partner stands on the left
    with pytest.raises(Reject) as r:
        _sampler(gen={"partner_side": "up"}).template_sides()
    assert r.value.reason == "spec"
    # train: uniform among the sides with a cell; one side consumes no draw
    s = _sampler()
    s.region_sides = [1.0, -1.0]
    draws = [s.draw_partner_side(np.random.default_rng(k)) for k in range(400)]
    assert set(draws) == {1.0, -1.0} and 150 < draws.count(1.0) < 250
    s.region_sides = [-1.0]
    rng = np.random.default_rng(1)
    assert s.draw_partner_side(rng) == -1.0 and rng.random() == np.random.default_rng(1).random()
    s.region_sides = []
    with pytest.raises(Reject):
        s.draw_partner_side(rng)
    # the attempt draws the side before anything else is sampled (the split cone's position and the target come after)
    seen = []
    monkeypatch.setattr(IS.SamplerV2, "sample", lambda self, rng: seen.append(self.region_side) or "draft")
    s.region_sides = [-1.0]
    assert IS.IORSSmall.sample(s, np.random.default_rng(0)) == "draft" and seen == [-1.0]


def test_shelf_layout_partner_side_is_the_board_opposite_the_lean():
    """touching#03 (owner 2026-10-12 item 7b): in the shelf layout the partner leans against the board opposite the target's lean, so that
    is its side in template_sides (template_facts.partner_sides) and the only eval side it admits - the right for hardback__acbrnv__bu,
    which leans left; the tabletop touching keeps the robot's left."""
    assert IS.partner_sides("touching", shelf_lean=1.0) == [-1.0] and IS.partner_sides("touching", shelf_lean=-1.0) == [1.0]

    def shelf(lean, gen=None):
        s = _sampler("touching", gen=gen, pair={"g_NT": 0.05})
        s.shelf_start = {"lean_side": lean, "lean_deg": 15.0}
        return s

    for lean in (1.0, -1.0):
        side = TT.side_name(-lean)
        assert shelf(lean).template_sides() == [-lean]  # train: the layout's side
        assert shelf(lean, {"partner_side": side}).template_sides() == [-lean]  # eval: the true side
        with pytest.raises(Reject) as r:
            shelf(lean, {"partner_side": TT.side_name(lean)}).template_sides()
        assert r.value.reason == "partner_side" and "shelf layout" in str(r.value)
    assert TT.side_name(-1.0) == "right"
    assert _sampler("touching", gen={"partner_side": "left"}, pair={"g_NT": 0.05}).template_sides() == [1.0]


def test_split_cone_pool_follows_the_attempts_partner_side():
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)

    def cand(pos, sides):
        st = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
        st.focus = th.tensor([0.9, 0.1 * pos, 0.78])
        st.extras.update({"angle_class": 0, "init_view_class": 0, "_position": (pos, 0), TT.START_REGION: {sd: [0.03, 0.1] for sd in sides}})
        return st

    cands = [cand(1, ["left"]), cand(1, ["left", "right"]), cand(2, ["right"]), cand(3, ["left"])]
    left, right = TT.split_cone_positions(cands, side=1.0), TT.split_cone_positions(cands, side=-1.0)
    assert left == {(1, 0): cands[:2], (3, 0): [cands[3]]} and right == {(1, 0): [cands[1]], (2, 0): [cands[2]]}
    assert set(TT.split_cone_positions(cands)) == {(1, 0), (2, 0), (3, 0)}
    assert TT.in_start_region(cands[2], None) and not TT.in_start_region(cands[2], 1.0) and TT.in_start_region(Stance("m", (0, 0, 0), 0.5), 1.0)
    s = types.SimpleNamespace(subtask=ONTOP, gen=lambda k, d=None: d, manip_candidates=list(cands))
    drawn = {1.0: set(), -1.0: set()}
    for seed in range(40):
        for side in (-1.0, 1.0):  # the left pool is built from the template's candidates, not from the right draw's
            s.region_side = side
            TT.split_cone_draw(s, np.random.default_rng(seed))
            assert all(TT.in_start_region(c, side) for c in s.manip_candidates)
            drawn[side].add(tuple(s.manip_candidates[0].extras["_position"]))
    assert drawn == {1.0: {(1, 0), (3, 0)}, -1.0: {(1, 0), (2, 0)}}


# ---------------------------------------------------------------------------------------------------------------- the attempt


def _attempt(s, y, side):
    c = next(c for c in s.start_region(_candidates(s)) if abs(float(c.focus[1]) - y) < 1e-6)
    manip = c.with_trunk(c.translate, c.tilt, "manipulation")
    manip.arms = ["right"]
    s.region_side = side
    return manip, (th.tensor([1.31, y, 0.753]), C.IDENTITY_QUAT.clone())


def test_attempt_draws_the_partner_from_the_cells_interval():
    s = _sampler(L=0.81)
    manip, t_pose = _attempt(s, -0.24, 1.0)
    t_fp = s.world.footprint(s.hull["target"], t_pose)
    for seed in range(12):
        p_pose = s.place_partner(np.random.default_rng(seed), t_pose, manip, None)
        p_fp = s.world.footprint(s.hull["partner"], p_pose)
        gap = float(p_fp[:, 1].min() - t_fp[:, 1].max())  # the partner on the robot's left (+y)
        assert 0.03 - 1e-4 <= gap <= 0.23 + 1e-4 and float(p_fp[:, 1].max()) <= 0.305 + 1e-4  # 0.10 from the edge's end
        assert abs(float(p_pose[0][0]) - 1.31) < 1e-5 and float(p_pose[0][2]) == pytest.approx(0.75 + 0.003, abs=1e-5)
        assert s.partner_draw["side"] == "left" and s.partner_draw["cell_interval_m"] == pytest.approx([0.03, 0.23])
        assert s.partner_draw["gap_interval_m"][0] - 1e-6 <= s.partner_draw["gap_m"] <= s.partner_draw["gap_interval_m"][1] + 1e-6
    # no orientation of the partner fits left of a target next to the edge's left end
    with pytest.raises(Reject) as e:
        s.place_partner(np.random.default_rng(0), (th.tensor([1.31, 0.16, 0.753]), C.IDENTITY_QUAT.clone()), manip, None)
    assert e.value.reason == "partner_placement" and "ends 0/" in str(e.value)


def test_partner_gap_masks_frame_rule():
    s = _sampler(L=4.0)
    s.world.camera_model = CameraModel(1080, 1080, 459.0, 459.0, 540.0, 540.0)
    cam = C.camera_pose(C.torso_q(0.5, 0.0), base_pose=(0.55, 0.0, 0.0))
    t_fp = s.world.footprint(s.hull["target"], (th.tensor([1.31, 0.0, 0.753]), C.IDENTITY_QUAT.clone()))
    gs = TT.gap_grid(0.03, 1.5, 0.05)
    m = TT.partner_gap_masks(s, 1.0, t_fp, 0.31, s.hull["partner"], C.IDENTITY_QUAT.clone(), s.init_stance, gs, top=s.top, cam=cam, m_end=0.10)
    assert set(m) == {"ends", "top", "frame"} and m["top"].all() and m["ends"].all()
    assert m["frame"][0] and not m["frame"][-1] and TT.longest_run(gs, m["frame"])[0] == pytest.approx(0.03)  # it leaves the image farther out


# ---------------------------------------------------------------------------------------------------------------- A5 partner on a meta link


def test_meta_link_world_points_object_frame_to_world():
    """A5 (owner 2026-10-12): ``meta_links.base_link.<type>`` entries (object frame, as burner/mdanhg's heatsource block) at the object's
    pose and scale."""
    entries = {"0": [{"position": [-0.114, -0.254, 0.0075], "orientation": [0.0, 0.0, -0.7071, 0.7071]}],
               "1": [{"position": [0.073, -0.254, 0.0054]}, {"position": [0.073, -0.150, 0.0054]}]}
    root = (th.tensor([2.0, 1.0, 0.9]), G.quat_from_yaw(math.pi / 2))
    pts = dict(TT.meta_link_world_points(entries, root, scale=[0.5, 0.5, 2.0]))
    assert sorted(pts) == ["0", "1/0", "1/1"]
    # x, y scaled by 0.5 then turned +90 deg about z: (x, y) -> (-y, x)
    assert pts["0"].tolist() == pytest.approx([2.0 + 0.127, 1.0 - 0.057, 0.9 + 0.015], abs=1e-5)
    assert pts["1/1"].tolist() == pytest.approx([2.0 + 0.075, 1.0 + 0.0365, 0.9 + 0.0108], abs=1e-5)


def _meta_sampler(monkeypatch, points, **kw):
    s = _sampler(gen={"partner_on_meta_link": "heatsource", **(kw.pop("gen", {}) or {})}, **kw)
    monkeypatch.setattr(IS.IORSSmall, "meta_link_points", lambda self, link: [(n, th.tensor(p, dtype=th.float32)) for n, p in points] if link == "heatsource" else [])
    return s


#: burner spots on the synthetic table: A / B beside the foci row (x 1.31), C off the top, D on the top but 0.14 m deeper than the row, E on
#: the top 5 cm from its end (the partner centred there overhangs it, as a pot's handle a stove)
SPOTS = [("A", [1.31, 0.105, 0.76]), ("B", [1.31, -0.405, 0.76]), ("C", [1.31, 2.0, 0.76]), ("D", [1.45, 0.105, 0.76]), ("E", [1.31, 1.45, 0.76])]


def test_meta_link_partner_start_region_follows_the_spots(monkeypatch):
    s = _meta_sampler(monkeypatch, SPOTS)
    kept = s.start_region(_candidates(s))
    spots = s.template_facts["meta_link_spots"]
    assert spots["link"] == "heatsource" and spots["usable"] == ["A", "B", "D", "E"]
    assert [f["on_top"] for f in spots["spots"]] == [True, True, False, True, True]
    reg = _regions(kept)
    names = {round(float(c.focus[1]), 2): {sd: [s.meta_spots[k][0] for k in ks] for sd, ks in c.extras[IS.META_SPOTS].items()} for c in kept}
    # A (lateral 0.105) is the partner left of the targets at -0.24 / -0.16 / -0.08 (gap 0.105 - y - 0.15), B (-0.405) right of -0.16 / -0.08
    # (gap y + 0.255); D never beside the row (0.14 m deeper than w_p), C off the top, E too far from every focus
    assert names == {-0.24: {"left": ["A"]}, -0.16: {"left": ["A"], "right": ["B"]}, -0.08: {"left": ["A"], "right": ["B"]}}
    assert reg[-0.24]["left"] == pytest.approx([0.195, 0.195], abs=1e-4) and reg[-0.08]["right"] == pytest.approx([0.175, 0.175], abs=1e-4)
    assert s.template_facts["start_region"]["meta_link"] == ["A", "B", "D", "E"] and s.region_sides == [1.0, -1.0]
    # eval right: B's cells only; a spot off the band is not reachable for the place (λ >= -0.565: B moved to -0.60)
    e = _meta_sampler(monkeypatch, SPOTS, gen={"partner_side": "right"})
    assert sorted(_regions(e.start_region(_candidates(e)))) == [-0.16, -0.08]
    far = _meta_sampler(monkeypatch, [("B", [1.31, -0.60, 0.76])], gen={"partner_side": "right"})
    assert far.start_region(_candidates(far)) == [] and far.template_facts["meta_link_spots"]["usable"] == ["B"]
    assert _sampler().meta_link() is None
    with pytest.raises(Reject) as r:  # touching braces its partner beside the push: no meta-link spot
        _meta_sampler(monkeypatch, SPOTS, relation="touching", pair={"g_NT": 0.05}).meta_link()
    assert r.value.reason == "spec"


def test_meta_link_partner_attempt_centres_the_partner_on_a_spot(monkeypatch):
    s = _meta_sampler(monkeypatch, SPOTS)
    manip, t_pose = _attempt(s, -0.16, -1.0)
    t_fp = s.world.footprint(s.hull["target"], t_pose)
    for seed in range(6):
        p_pose = s.place_partner(np.random.default_rng(seed), t_pose, manip, None)
        fp = s.world.footprint(s.hull["partner"], p_pose)
        centre = 0.5 * (fp.max(dim=0).values + fp.min(dim=0).values)
        assert centre.tolist() == pytest.approx([1.31, -0.405], abs=1e-4) and s.partner_draw["meta_link"] == "B"
        assert 0.03 <= float(t_fp[:, 1].min() - fp[:, 1].max()) <= 0.23 and s.partner_draw["side"] == "right"
    with pytest.raises(Reject) as e:  # the target drawn far from the spot: the gap leaves the range
        s.place_partner(np.random.default_rng(0), (th.tensor([1.31, 0.16, 0.753]), C.IDENTITY_QUAT.clone()), manip, None)
    assert e.value.reason == "partner_placement" and "lateral gap" in str(e.value)


def test_meta_link_points_from_the_loaded_prims_else_the_model_metadata(monkeypatch):
    import os

    from omnigibson.utils.asset_utils import get_dataset_path

    s = _sampler()
    link = types.SimpleNamespace(is_meta_link=True, meta_link_type="heatsource", meta_link_id="left", get_position_orientation=lambda: (th.tensor([1.0, 2.0, 0.9]), None))
    other = types.SimpleNamespace(is_meta_link=False)
    obj = types.SimpleNamespace(links={"meta": link, "base_link": other}, category="burner", model="mdanhg", scale=th.ones(3),
                                get_position_orientation=lambda: (th.zeros(3), C.IDENTITY_QUAT.clone()))
    monkeypatch.setattr(IS.IORSSmall, "support", property(lambda self: obj))
    (name, p), = s.meta_link_points("heatsource")
    assert name == "left" and p.tolist() == pytest.approx([1.0, 2.0, 0.9])
    obj.links = {"base_link": other}
    for category, model, n in (("burner", "mdanhg", 5), ("stove", "ykretu", 6)):  # the owner's counts (misc/metadata.json)
        if not os.path.exists(os.path.join(get_dataset_path("behavior-1k-assets"), "objects", category, model, "misc", "metadata.json")):
            pytest.skip("no behavior-1k-assets dataset")
        obj.category, obj.model = category, model
        assert len(s.meta_link_points("heatsource")) == n


# ---------------------------------------------------------------------------------------------------------------- 6d target on a neighbour


def _union_sampler(monkeypatch, points, gen=None):
    """The synthetic table cut between y = 0.02 and 0.04: the support (y >= 0.04, 0.75 m, the burner the partner stands on) and a
    neighbour (y <= 0.02, 2 cm lower: not flush), one union top; the band widened to λ in [-0.565, 0.565] so that both spots pass the
    reach rule."""
    s = _meta_sampler(monkeypatch, points, gen=gen)
    xs = th.arange(1.0, 1.6 + 1e-6, 0.02)

    def top(name, ys, z):
        return SupportTop(name, xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), z, 0.0)

    s.top = NB.UnionTop.build([top("burner_x_0", th.arange(0.04, 1.5 + 1e-6, 0.02), 0.75), top("bar_x_0", th.arange(-1.5, 0.02 + 1e-6, 0.02), 0.73)])
    s.bands = _Band(-0.565, 0.565)
    return s


#: the partner's spots on the support: A beside the foci -0.24 / -0.16 / -0.08 (on the neighbour), F beside 0.0 / 0.08 (on the support
#: or across the seam)
UNION_SPOTS = [("A", [1.31, 0.105, 0.76]), ("F", [1.31, 0.305, 0.76])]


def test_start_region_on_neighbors_keeps_the_foci_on_the_neighbour_top(monkeypatch):
    """Owner 2026-10-12 (ontop_small#02 6d): with ``start_region_on_neighbors`` the A5 start region keeps only the foci whose Phase-2
    footprint (w_t + 0.02 around them) lies on a neighbour's cells; without the key the same union gives A5's region unchanged."""
    plain = _union_sampler(monkeypatch, UNION_SPOTS)
    names = lambda kept, s: {round(float(c.focus[1]), 2): sorted(s.meta_spots[k][0] for ks in c.extras[IS.META_SPOTS].values() for k in ks) for c in kept}
    assert names(plain.start_region(_candidates(plain)), plain) == {-0.24: ["A"], -0.16: ["A"], -0.08: ["A"], 0.0: ["F"], 0.08: ["F"]}
    assert "on_neighbors" not in plain.template_facts["start_region"] and plain.neighbor_start_top() is None
    s = _union_sampler(monkeypatch, UNION_SPOTS, gen={"start_region_on_neighbors": True})
    kept = s.start_region(_candidates(s))
    assert names(kept, s) == {-0.24: ["A"], -0.16: ["A"], -0.08: ["A"]} and s.region_sides == [1.0]
    assert _regions(kept)[-0.08]["left"] == pytest.approx([0.035, 0.035], abs=1e-4)  # A's region itself is A5's
    facts = s.template_facts["start_region"]
    assert facts["on_neighbors"] == ["bar_x_0"] and facts["off_neighbors"] == 3 and facts["cells"] == {"left": 3, "right": 0}  # 0.0, 0.08, 0.16 off
    assert s.template_facts["meta_link_spots"]["usable"] == ["A", "F"]  # the spots stay on the support


def test_start_region_on_neighbors_draws_the_target_on_the_neighbour_and_needs_one(monkeypatch):
    s = _union_sampler(monkeypatch, UNION_SPOTS, gen={"start_region_on_neighbors": True})
    nb = s.neighbor_start_top()
    assert isinstance(nb, NB.UnionTop) and th.equal(nb.free, s.top.free & (s.top.member == 1))
    # Phase 3 (the target's draw) gets that top; without the key the union
    seen = []
    monkeypatch.setattr(TT, "phase3_target", lambda self, manip, rng, gs, *a, top=None, **k: seen.append(top) or ("pose", "grasp"))
    s.grasps = s.grasps_flush = ["g"]
    s.support_objects = lambda: []
    s.choice = types.SimpleNamespace(models={"target": types.SimpleNamespace(scale=None)})
    assert s.phase3_two_tier(None, np.random.default_rng(0)) == ("pose", "grasp")
    assert isinstance(seen[0], NB.UnionTop) and th.equal(seen[0].free, nb.free) and seen[0].z_at([1.3, -0.3]) == 0.73
    plain = _union_sampler(monkeypatch, UNION_SPOTS)
    plain.grasps = plain.grasps_flush = ["g"]
    plain.support_objects, plain.choice = s.support_objects, s.choice
    plain.phase3_two_tier(None, np.random.default_rng(0))
    assert seen[1] is plain.top
    # the key needs the partner on a meta link, and a support whose top is united with a neighbour's
    with pytest.raises(Reject) as r:
        _union_sampler(monkeypatch, UNION_SPOTS, gen={"start_region_on_neighbors": True, "partner_on_meta_link": None}).neighbor_start_top()
    assert r.value.reason == "spec"
    lone = _meta_sampler(monkeypatch, UNION_SPOTS, gen={"start_region_on_neighbors": True})
    with pytest.raises(Reject) as r:
        lone.start_region(_candidates(lone))
    assert r.value.reason == "no_start_region" and "no neighbour top" in str(r.value)
