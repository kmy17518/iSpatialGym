"""Simulator-free tests of the large-scale IORS inside mode (owner 2026-10-12 ~16:05 item 1; spec §5.6 "Inside (large-scale)"): the interior
goal geometry, stance C's side, the release height, the drop test's protrusion, the ``inside`` rays rule of a container without a fillable
meta link and the oracle program (``pytest ispatialgym/tests/test_iors_large_inside.py``)."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import predicates as PRED
from ispatialgym.sampling.sim_utils import Reject, SupportInterior
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
from ispatialgym.sampling.v2.tasks import iors_large_inside as IN  # noqa: E402
from ispatialgym.sampling.v2.tasks.iors_small import gripper_corners_world  # noqa: E402

SUBTASK = "inter_object_relation_satisfaction-inside_large_scale"
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecars", f"{SUBTASK}.json")


class _GP:
    def __init__(self, alias: str = f"{SUBTASK}-plate-drop_in_sink-countertop", **gen):
        self.alias, self.generation = alias, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


# ----------------------------------------------------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------------------------------------------------


def test_raster_rim_and_erosion():
    xs, ys = th.arange(6) * 0.02, th.arange(6) * 0.02
    mask = th.ones(6, 6, dtype=th.bool)
    rim = IN.raster_rim(mask, xs, ys)
    assert len(rim) == 20  # the 6 x 6 border
    cells = th.stack(th.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    kept = IN.erode_from(cells, rim, 0.025)  # > 1 cell from the border
    assert len(kept) == 4 and bool((kept >= 0.04 - 1e-6).all() and (kept <= 0.06 + 1e-6).all())
    assert len(IN.erode_from(cells, rim, 0.05)) == 0
    assert len(IN.erode_from(cells, th.zeros(0, 2), 0.05)) == 36


def test_stance_side_picks_the_raster_side_facing_it():
    edges = [EdgeFrame("+u", th.tensor([0.0, 0.3]), th.tensor([0.0, -1.0]), 0.5), EdgeFrame("-u", th.tensor([0.0, -0.3]), th.tensor([0.0, 1.0]), 0.5),
             EdgeFrame("+v", th.tensor([0.3, 0.0]), th.tensor([-1.0, 0.0]), 0.6), EdgeFrame("-v", th.tensor([-0.3, 0.0]), th.tensor([1.0, 0.0]), 0.6)]
    assert [e.name for e in IN.edges_facing(edges, IN.side_vector("+y"))] == ["+u"]
    assert [e.name for e in IN.edges_facing(edges, IN.side_vector("-x"))] == ["-v"]
    assert [e.name for e in IN.edges_facing(edges, IN.side_vector([1.0, 0.8]))] == ["+v"]  # 39 deg from +x, 51 from +y
    assert [e.name for e in IN.edges_facing(edges, IN.side_vector([1.0, 1.0]))] == ["+u", "+v"]  # exactly 45 deg from both: kept, ties in order
    assert len(IN.edges_facing(edges, None)) == 4 and IN.side_vector(None) is None
    with pytest.raises(ValueError):
        IN.side_vector("up")


def test_overhang_release_lift_rim_and_protrusion():
    lo, hi = th.tensor([-0.5, -0.5, 0.0]), th.tensor([0.5, 0.62, 2.0])
    assert IN.overhang(th.tensor([0.0, 0.5]), th.tensor([0.0, 1.0]), lo, hi) == pytest.approx(0.12, abs=1e-5)
    assert IN.overhang(th.tensor([0.0, 0.58]), th.tensor([0.0, 1.0]), lo, hi) == 0.0  # 4 cm: no overhang
    assert IN.release_lift([0.71, 0.75], 0.70, 0.02) == pytest.approx(0.01)
    assert IN.release_lift([0.75], 0.70, 0.02) == 0.0 and IN.release_lift([], 0.70, 0.02) == 0.0
    assert IN.rim_height(0.89, 0.70, 0.026, 1.20) == (0.89, "support_top")  # a sink's deck above the basin floor
    assert IN.rim_height(0.705, 0.70, 0.026, 1.20) == (1.20, "aabb_top")  # the top raster found the basin floor itself: permissive
    assert IN.rim_height(None, 0.70, 0.026, 1.20) == (1.20, "aabb_top")
    assert IN.protrusion_ok(0.90, 0.89, 0.026, 0.5) == (True, pytest.approx(0.01))
    assert IN.protrusion_ok(0.91, 0.89, 0.026, 0.5)[0] is False


def test_best_per_base_keeps_the_most_reach_margin_per_base():
    def st(x, tr, ti, m):
        return SimpleNamespace(base_pose2d=(x, -1.1, -1.57), translate=tr, tilt=ti, extras={"reach_margin_m": m})

    stances = [st(5.6, 0.0, 0.0, 0.1), st(5.6, 0.5, -0.3, 0.3), st(5.7, 0.0, 0.0, 0.5), st(5.6, 0.25, 0.0, 0.3), st(5.6, 0.95, -0.9, 0.2)]
    out = IN.best_per_base(stances, 2)
    assert [(s.base_pose2d[0], s.translate) for s in out] == [(5.6, 0.5), (5.6, 0.25), (5.7, 0.0)]  # bases in order, the tie in input order
    assert IN.best_per_base(stances, 9) == [stances[1], stances[3], stances[4], stances[0], stances[2]] and IN.best_per_base([], 3) == []


# ----------------------------------------------------------------------------------------------------------------
# the mode on stubs
# ----------------------------------------------------------------------------------------------------------------


class FloorBound(Exception):
    pass


def _floor_bound(self):
    raise FloorBound(self.relation)


def test_hook_adopts_the_inside_mode(monkeypatch):
    monkeypatch.setattr(IL.IORSLarge, "bind_floor_v2", _floor_bound)
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = "inside", {}, {}
    with pytest.raises(FloorBound):
        s.bind_scene_roles()  # the mode adopted, then IORSLarge's binding (partner + floor)
    assert s.relation_mode is IN.InsideMode and isinstance(s, IN.InsideMode) and isinstance(s, IL.IORSLarge)
    assert type(s).prepare_partner is IN.InsideMode.prepare_partner and type(s)._couple is IL.IORSLarge._couple


def _interior(nx: int = 30, ny: int = 25, res: float = 0.02, z: float = 0.70, headroom: float = 1.5) -> SupportInterior:
    """A basin floor 0.60 x 0.50 m centred at the origin at height ``z`` (the sink frame = the world frame)."""
    xs = (th.arange(nx) - (nx - 1) / 2) * res
    ys = (th.arange(ny) - (ny - 1) / 2) * res
    mask = th.ones(nx, ny, dtype=th.bool)
    return SupportInterior("drop_in_sink_awvzkn_0", xs, ys, mask, th.zeros_like(mask), z, 0.0, [], res, headroom=th.full((nx, ny), headroom), z_requested=0.80,
                           board_pieces=["/World/sink/base_link/collisions/floor"])


class _Stub:
    """What :meth:`InsideMode.prepare_partner` / :meth:`release_above` / :meth:`goal_pose_at` read."""

    prepare_partner = IN.InsideMode.prepare_partner
    inside_surface_spec = IN.InsideMode.inside_surface_spec
    release_above = IN.InsideMode.release_above
    goal_pose_at = IN.InsideMode.goal_pose_at
    num = IL.IORSLarge.num

    def __init__(self, interior=None, grasps=(), half=0.12, height=0.026, top_z=0.89, **gen):
        self.gp = _GP(**gen)
        self.floor_z = 0.0
        self.template_facts = {}
        self.partner = SimpleNamespace(name="drop_in_sink_awvzkn_0", aabb=(th.tensor([-0.36, -0.30, 0.55]), th.tensor([0.36, 0.30, 1.20])))
        self._interior = interior or _interior()
        self.calls = []
        self.world = SimpleNamespace(support_interior=self._support_interior, spawned={}, bottom_offset=lambda hull, quat: 0.0108)
        self._half, self._height, self._top_z = half, height, top_z
        self.grasps = list(grasps)
        self.hull = th.zeros(8, 3)

    def _support_interior(self, partner, z, **kw):
        self.calls.append((z, kw))
        return self._interior

    def target_dims(self):
        return self._half, self._height

    def support_top(self, partner):
        return SimpleNamespace(z=self._top_z)


def test_prepare_partner_goal_cells_sides_and_rim():
    s = _Stub(inside_surface_z_m=0.80, inside_surface_tol_m=0.25, inside_stance_side="+y")
    s.prepare_partner()
    (z_req, kw), = s.calls
    assert z_req == pytest.approx(0.80) and kw["z_margin"] == pytest.approx(0.25)  # the band job's --surface-z / --surface-tol
    cells = s.place["cells"]
    # eroded by half (0.12) + opening_margin_m (0.02) from the board's outline: |x| <= 0.29 - 0.14, |y| <= 0.24 - 0.14
    assert len(cells) > 0 and float(cells[:, 0].abs().max()) < 0.29 - 0.14 + 1e-6 and float(cells[:, 1].abs().max()) < 0.24 - 0.14 + 1e-6
    assert [e.name for e in s.place["edges"]] == ["+v"] and float(s.place["edges"][0].inward[1]) == pytest.approx(-1.0)  # the side facing +y
    assert s.place["edges"][0].overhang == pytest.approx(0.30 - 0.24 if 0.30 - 0.24 >= 0.05 else 0.0, abs=1e-5)
    assert s.place["rim_z"] == pytest.approx(0.89) and s.place["z"] == pytest.approx(0.70) and s.place["z_requested"] == pytest.approx(0.80)
    assert s.template_facts["inside_surface"]["n_goal_cells"] == len(cells)


def test_prepare_partner_leaves_the_open_door_out():
    """``generation.partner_joint_init`` (inside_large_scale#02, the fridge door open): stance C's overhang is measured on the partner's
    AABB without the links the request moved, and the interior cells under the partner's doors leave the raster (a door bin at the
    shelf's height in front of the opening would move the side stance C squares to)."""
    nx, ny, res = 30, 35, 0.02
    xs = (th.arange(nx) - (nx - 1) / 2) * res  # -0.29 .. 0.29
    ys = (th.arange(ny) - 12) * res  # -0.24 .. 0.44
    gx, gy = th.meshgrid(xs, ys, indexing="ij")
    bin_cells = (gx >= 0.20) & (gy > 0.25) & (gy <= 0.40 + 1e-6)
    mask = (gy <= 0.24 + 1e-6) | bin_cells  # the shelf + a door bin in front of it at the same height
    itop = SupportInterior("fridge_petcxr_0", xs, ys, mask, th.zeros_like(mask), 0.70, 0.0, [], res, headroom=th.full((nx, ny), 1.5), z_requested=0.80)
    s = _Stub(interior=itop, inside_surface_z_m=0.80, inside_surface_tol_m=0.25, inside_stance_side="+y")
    s.partner.aabb = (th.tensor([-0.36, -0.30, 0.55]), th.tensor([0.60, 0.90, 1.20]))  # the door swung out to +x / +y
    s.partner_static_aabb = lambda: (th.tensor([-0.36, -0.30, 0.55]), th.tensor([0.36, 0.30, 1.20]))
    s.partner_part_boxes = lambda: [(th.tensor([0.19, 0.265, 0.0]), th.tensor([0.60, 0.90, 2.0]))]  # - the half-cell pad: from y 0.255
    s.prepare_partner()
    (edge,) = s.place["edges"]
    assert edge.name == "+v" and float(edge.mid[1]) == pytest.approx(0.24, abs=1e-5)  # the shelf's front, not the bin's 0.40
    assert edge.overhang == pytest.approx(0.06, abs=1e-5)  # 0.66 with the door in the AABB
    assert s.template_facts["inside_surface"]["cells_under_doors"] == int(bin_cells.sum()) == 40
    assert float(s.place["cells"][:, 1].max()) < 0.24 - 0.14 + 1e-6


def test_prepare_partner_rejects_without_height_room_or_side():
    with pytest.raises(Reject) as e:
        _Stub().prepare_partner()
    assert e.value.reason == "spec"
    with pytest.raises(Reject) as e:
        _Stub(half=0.40, inside_surface_z_m=0.80).prepare_partner()
    assert e.value.reason == "inside_no_goal_cell"
    with pytest.raises(Reject) as e:
        _Stub(interior=_interior(headroom=0.03), inside_surface_z_m=0.80).prepare_partner()  # headroom < height + 2 cm
    assert e.value.reason == "inside_no_goal_cell"
    with pytest.raises(Reject) as e:
        _Stub(inside_surface_z_m=0.80, inside_stance_side="up").prepare_partner()
    assert e.value.reason == "spec"


def _top_down_grasp(eef_above_origin: float):
    """A top-down pinch: the eef ``eef_above_origin`` above the object origin, approach straight down."""
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    return SimpleNamespace(name="g", pose=(th.tensor([0.0, 0.0, eef_above_origin]), q))


def test_release_above_clears_the_surface_with_every_grasp():
    s = _Stub(inside_surface_z_m=0.80)
    s.prepare_partner()
    quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    s.grasps = [_top_down_grasp(0.10)]  # fingers far above the object's bottom: the plain drop height
    assert s.release_above(quat) == pytest.approx(IN.DROP_HEIGHT_M)
    s2 = _Stub(inside_surface_z_m=0.80, grasps=[_top_down_grasp(0.10), _top_down_grasp(0.0)])  # the second pinch's fingers below the bottom
    s2.prepare_partner()
    above = s2.release_above(quat)
    assert above > IN.DROP_HEIGHT_M
    pose = s2.goal_pose_at([0.05, -0.02], quat)
    assert float(pose[0][2]) == pytest.approx(0.70 + 0.0108 + above) and float(pose[0][0]) == pytest.approx(0.05)
    lows = [float(gripper_corners_world((pose[0] + g.pose[0], g.pose[1]))[:, 2].min()) for g in s2.grasps]
    assert min(lows) >= 0.70 + IN.RELEASE_CLEARANCE_M - 1e-5


#: the trunks the sink's goal-side band job (tband 17831152, the same base as stance C at stand-off 0.55) found colliding with every
#: idle-arm posture: the deep leans over the counter run
SINK_COLLIDING_TRUNKS = {(0.5, -0.9), (0.75, -0.6), (0.75, -0.9), (0.95, -0.6), (0.95, -0.9)}


class _StanceCStub:
    """What :meth:`InsideMode._placement_candidates_at` reads without a goal-side band: the sink's +y raster side (18 cm long, 9 cm
    AABB overhang); the reach margin grows with the lean as the shoulder's (the deep leans first)."""

    _placement_candidates_at = IN.InsideMode._placement_candidates_at
    num = IL.IORSLarge.num

    def __init__(self, **gen):
        self.gp = _GP(**gen)
        self.seg_arm = "right"
        self.partner = SimpleNamespace(name="drop_in_sink_awvzkn_0")
        self.place = {"half": 0.12, "height": 0.026, "z": 0.7316, "cells": th.tensor([[5.683, -1.79], [5.70, -1.85]])}
        self.template_facts = {}
        self.hull, self.rest = th.zeros(8, 3), th.tensor([0.0, 0.0, 0.0, 1.0])

    def inside_band(self, arm):
        return None

    def reach_margin(self, st, point, arm):
        return 0.2 * st.translate - 0.31 * st.tilt

    def goal_pose_at(self, gc, quat):
        return th.tensor([gc[0], gc[1], 0.75]), quat

    def hull_corners_world(self, hull, pose):
        return th.zeros(8, 3)


def test_no_band_stance_c_keeps_the_free_trunks_with_the_most_reach_margin(monkeypatch):
    """inside_large#03 (bar alias): without a band the reach-margin cut ran before the collision check and kept the three deepest leans
    per base - the very trunks the counter blocks -, so every stance C candidate collided while 15 of 20 trunks were free there."""
    edge = EdgeFrame.from_edge("+u", th.tensor([5.683, -1.752]), th.tensor([0.0025, 1.0]), 0.18, 0.09)
    monkeypatch.setattr(IN.P0, "base_ok", lambda s, xy: True)
    monkeypatch.setattr(IN.P0, "body_free", lambda s, cands, **kw: th.tensor([(c.translate, c.tilt) not in SINK_COLLIDING_TRUNKS for c in cands]))
    monkeypatch.setattr(IN.P0, "camera_poses", lambda s, cands: list(cands))  # the stance stands in for its camera
    monkeypatch.setattr(IN.P0, "in_frame", lambda s, cam, pts: True)
    monkeypatch.setattr(IN.P0, "axis_angle", lambda s, cam, point: 0.3)
    s = _StanceCStub()
    margin = {(t, k): s.reach_margin(SimpleNamespace(translate=t, tilt=k), None, None) for t in IN.RB.TRANSLATES for k in IN.RB.TILTS}
    assert set(sorted(margin, key=lambda tk: -margin[tk])[:3]) <= SINK_COLLIDING_TRUNKS  # the cut before the check kept colliding trunks only
    out = s._placement_candidates_at(edge, (8.25, 0.64), "right")
    best_free = {(0.25, -0.9), (0.5, -0.6), (0.95, -0.3)}
    assert len(out) == 12 and all({(c.translate, c.tilt) for c in out if c.stand_off == so} == best_free for so in IN.STAND_OFFS)
    stats = s.template_facts["placement_stats"]
    assert stats["collision_free"] == 4 * 15 and stats["visible"] == 4 * 15 and stats["no_band_kept"] == 12
    assert all(c.extras["goal_cell"] == pytest.approx([5.683, -1.79]) and c.extras["angle_class"] == 1 for c in out)
    # the cut comes after the frame test too: the tilt -0.9 trunks out of frame give way to the next free trunk in frame
    monkeypatch.setattr(IN.P0, "in_frame", lambda s, cam, pts: cam.tilt > -0.85)
    s = _StanceCStub()
    out = s._placement_candidates_at(edge, (8.25, 0.64), "right")
    assert {(c.translate, c.tilt) for c in out if c.stand_off == 0.55} == {(0.5, -0.6), (0.95, -0.3), (0.75, -0.3)}
    # inside_no_band_trunks still sets the number kept
    assert len(_StanceCStub(inside_no_band_trunks=1)._placement_candidates_at(edge, (8.25, 0.64), "right")) == 4


def test_stance_postures_key():
    assert IN.stance_postures(None) == ["untucked"] and IN.stance_postures("ready") == ["ready"]
    assert IN.stance_postures(["untucked", "ready", "tucked", "ready"]) == ["untucked", "ready", "tucked"]
    for bad in ([], ["crouched"], ["untucked", "reset"]):
        with pytest.raises(ValueError):
            IN.stance_postures(bad)


#: the trunks of the fridge's goal-side band (tband 17832286 / 17832291: a trunk is free only with the arms ready or tucked); the
#: deepest needs tucked arms here
FRIDGE_BAND_TRUNKS = [(0.25, 0.0), (0.5, -0.3), (0.75, -0.3), (0.95, -0.6)]
POSTURE_Q = {"untucked": 0.0, "ready": 1.0, "tucked": 2.0}


class _BandStanceCStub(_StanceCStub):
    """:class:`_StanceCStub` with a goal-side band whose cells admit every goal cell (inside_large#02: fridge_petcxr_0's +y side)."""

    def inside_band(self, arm):
        return SimpleNamespace(trunks=lambda: list(FRIDGE_BAND_TRUNKS), meta={"lattice": 0.08})

    def band_cells_world(self, rb, st):
        return self.place["cells"]

    def _snap(self, cells, goal_cells, lattice):
        return goal_cells


def _fridge_body_free(calls):
    """``prompt0.body_free`` at the open fridge: the untucked check (working arm untucked, the idle arm's A6 retry) collides everywhere,
    both arms ready are free up to translate 0.75, both tucked everywhere."""

    def body_free(s, cands, posture=None, **kw):
        calls.append((posture, [(c.stand_off, c.translate) for c in cands]))
        if posture is None:
            return th.zeros(len(cands), dtype=th.bool)
        return th.tensor([posture == "tucked" or c.translate <= 0.75 for c in cands])

    return body_free


def _visible_everywhere(monkeypatch):
    monkeypatch.setattr(IN.P0, "base_ok", lambda s, xy: True)
    monkeypatch.setattr(IN.P0, "camera_poses", lambda s, cands: list(cands))
    monkeypatch.setattr(IN.P0, "in_frame", lambda s, cam, pts: True)
    monkeypatch.setattr(IN.P0, "axis_angle", lambda s, cam, point: 0.3)


def test_stance_c_keeps_the_first_free_posture(monkeypatch):
    """inside_large#02 (owner 2026-10-12 ~20:00): the band admitted stance C with the arms ready / tucked only, while the stance-C check
    held the working arm untucked - every candidate collided. With ``inside_stance_postures`` [untucked, ready, tucked] a candidate keeps
    the first free posture and records it as its idle posture (the sidecar record, the drive-in); untucked survivors are unchanged."""
    edge = EdgeFrame.from_edge("+u", th.tensor([5.683, -1.752]), th.tensor([0.0025, 1.0]), 0.18, 0.09)
    _visible_everywhere(monkeypatch)
    calls = []
    monkeypatch.setattr(IN.P0, "body_free", _fridge_body_free(calls))
    assert _BandStanceCStub()._placement_candidates_at(edge, (8.25, 0.64), "right") == []  # the 19:32 run: every candidate collides
    calls.clear()
    s = _BandStanceCStub(inside_stance_postures=["untucked", "ready", "tucked"])
    out = s._placement_candidates_at(edge, (8.25, 0.64), "right")
    lattice = [(so, tr) for so in IN.STAND_OFFS for tr, _ in FRIDGE_BAND_TRUNKS]
    assert [(c.stand_off, c.translate) for c in out] == lattice  # the lattice order
    assert [c.idle_posture for c in out] == [c.extras["stance_posture"] for c in out] == ["ready" if tr <= 0.75 else "tucked" for _, tr in lattice]
    # each posture is tried on what the earlier ones left
    assert [(p, len(c)) for p, c in calls] == [(None, 16), ("ready", 16), ("tucked", 4)] and all(tr > 0.75 for _, tr in calls[2][1])
    stats = s.template_facts["placement_stats"]
    assert stats["collision_free"] == 16 and stats["postures"] == {"untucked": 0, "ready": 12, "tucked": 4} and stats["visible"] == 16
    # the placement stance _couple builds from a survivor keeps the posture: its sidecar record and the arms of every check from C
    c = out[0]
    place = c.with_trunk(c.translate, c.tilt, "placement")
    place.arms, place.extras = ["right"], dict(c.extras)
    rec = place.to_sidecar()
    assert rec["arms"] == ["right"] and rec["idle_posture"] == "ready" and rec["stance_posture"] == "ready"
    kit = SimpleNamespace(robot=None, arm_q=lambda name, arm: th.full((7,), POSTURE_Q[name]))
    r, l, _ = IN.IK.stance_arm_vectors(kit, place)
    assert bool((l == POSTURE_Q["ready"]).all()) and bool((r == POSTURE_Q["untucked"]).all())  # the working arm: the carry / the exact IK
    # an untucked survivor stays as the untucked check left it (the A6 retry's idle posture, no stance_posture)

    def a6(s_, cands, posture=None, **kw):
        if posture is None:
            for c_ in cands:
                c_.idle_posture = "tucked" if c_.translate == 0.25 else None
            return th.tensor([c_.translate == 0.25 for c_ in cands])
        return th.ones(len(cands), dtype=th.bool)

    monkeypatch.setattr(IN.P0, "body_free", a6)
    s = _BandStanceCStub(inside_stance_postures=["untucked", "ready", "tucked"])
    out = s._placement_candidates_at(edge, (8.25, 0.64), "right")
    assert [(c.idle_posture, c.extras.get("stance_posture")) for c in out if c.translate == 0.25] == [("tucked", None)] * 4
    assert {c.extras.get("stance_posture") for c in out if c.translate != 0.25} == {"ready"}
    assert s.template_facts["placement_stats"]["postures"] == {"untucked": 4, "ready": 12, "tucked": 0}


def test_stance_c_default_postures_are_the_untucked_check(monkeypatch):
    """Without ``inside_stance_postures``: one ``prompt0.body_free`` call with today's arguments, the survivors as it left them, the
    placement stats without ``postures``; an unknown posture name is a ``spec`` rejection."""
    edge = EdgeFrame.from_edge("+u", th.tensor([5.683, -1.752]), th.tensor([0.0025, 1.0]), 0.18, 0.09)
    _visible_everywhere(monkeypatch)
    calls = []
    monkeypatch.setattr(IN.P0, "body_free", lambda s, cands, **kw: (calls.append(kw), th.tensor([c.translate <= 0.5 for c in cands]))[1])
    for gen in ({}, {"inside_stance_postures": ["untucked"]}):
        calls.clear()
        s = _BandStanceCStub(**gen)
        out = s._placement_candidates_at(edge, (8.25, 0.64), "right")
        assert calls == [{}] and [(c.stand_off, c.translate) for c in out] == [(so, tr) for so in IN.STAND_OFFS for tr, _ in FRIDGE_BAND_TRUNKS if tr <= 0.5]
        assert all(c.idle_posture is None and "stance_posture" not in c.extras for c in out)
        stats = s.template_facts["placement_stats"]
        assert stats["collision_free"] == 8 and "postures" not in stats
    with pytest.raises(Reject) as e:
        _BandStanceCStub(inside_stance_postures=["untucked", "crouched"])._placement_candidates_at(edge, (8.25, 0.64), "right")
    assert e.value.reason == "spec"


def test_carry_drive_arms_follow_the_posture_stance_c_kept():
    """The oracle's departure transition at B puts the idle arm in C's recorded posture and the carried drive keeps it: a C kept in
    ready / tucked has the B -> C drive checked with C's arms and the holding arm at its carry configuration; else IORSLarge's default."""
    cls = type("IORSLarge_InsideMode", (IN.InsideMode, IL.IORSLarge), {})
    s = cls.__new__(cls)
    s.kit = SimpleNamespace(robot=None, arm_q=lambda name, arm: th.full((7,), POSTURE_Q[name]))
    place = Stance("placement", (5.68, -1.2, -1.57), 0.5, -0.3, 0.45, None, None, 0.0, None, ["right"])
    place.idle_posture, place.extras["stance_posture"] = "tucked", "tucked"
    assert s.carry_drive_arms(place, "jar", "right") is None  # no carry configuration recorded
    q_carry = th.full((7,), 9.0)
    s.record_carry_arm_q("jar", "right", q_carry)
    arms = s.carry_drive_arms(place, "jar", "right")
    assert sorted(arms) == ["left", "right"] and bool((arms["right"] == q_carry).all()) and bool((arms["left"] == POSTURE_Q["tucked"]).all())
    place.idle_posture = None
    del place.extras["stance_posture"]
    assert s.carry_drive_arms(place, "jar", "right") is None and IL.IORSLarge.carry_drive_arms(s, place, "jar", "right") is None


class _CarryLegStub:
    """What :meth:`IORSLarge.record_carry_leg` reads; ``record_path`` records its keyword arguments (``collide`` first calls fail)."""

    record_carry_leg = IL.IORSLarge.record_carry_leg
    num = IL.IORSLarge.num

    def __init__(self, drive_q, travel, collide=0):
        self.gp, self.seg_arm, self.template_facts = _GP(), "right", {}
        self.drive_q, self.travel, self.collide, self.paths = drive_q, travel, collide, []

    def record_carry_arm_q(self, carried, arm, q):
        pass

    def travel_trunks(self, init, manip):
        return list(self.travel)

    def record_departure_transition(self, frm, to, tt, carried=None, arm=None):
        return SimpleNamespace(stand_off=0.65)

    def carry_drive_arms(self, place, tname, arm):
        return self.drive_q

    def record_path(self, frm, to, **kw):
        self.paths.append(kw)
        if len(self.paths) <= self.collide:
            raise Reject("path_collision", "stride 4/35")
        return SimpleNamespace(length_m=3.2)

    def record_transition(self, frm, to, **kw):
        pass

    def _drop_leg_records(self, to):
        pass


def test_record_carry_leg_drives_after_the_departure_with_the_carry_drive_arms():
    manip = Stance("manipulation", (2.0, 1.0, 0.0), 0.75, -0.3, 0.45, None, None, 0.0, None, ["right"])
    place = Stance("placement", (5.68, -1.2, -1.57), 0.5, -0.3, 0.45, None, None, 0.0, None, ["right"])
    drive_q = {"right": th.zeros(7), "left": th.ones(7)}
    s = _CarryLegStub(drive_q, [(0.75, 0.0)], collide=1)  # the local path collides: the eroded global map, same arms
    s.record_carry_leg(None, manip, place, "jar", th.zeros(7), 6.0)
    assert [(kw["arm_q"] is drive_q, kw["trunk"], kw.get("force_global", False)) for kw in s.paths] == [(True, (0.75, 0.0), False), (True, (0.75, 0.0), True)]
    assert s._leg_bc["order"] == "straighten_then_navigate" and place.carried == "jar"
    s = _CarryLegStub(None, [(0.75, 0.0)])
    s.record_carry_leg(None, manip, place, "jar", th.zeros(7), 6.0)
    assert [kw["arm_q"] for kw in s.paths] == [None]  # IORSLarge's default: record_path's own arms
    s = _CarryLegStub(drive_q, [(0.75, -0.3)])  # B already straight: the drive at B's trunk keeps B's arms (no departure re-poses them)
    s.record_carry_leg(None, manip, place, "jar", th.zeros(7), 6.0)
    assert [("arm_q" in kw, "trunk" in kw) for kw in s.paths] == [(False, False)] and s._leg_bc["order"] == "drive_at_grasp_trunk"


# ----------------------------------------------------------------------------------------------------------------
# inside of a container without a fillable meta link
# ----------------------------------------------------------------------------------------------------------------


def test_inside_rule_two_axes_with_the_container_on_both_sides():
    assert PRED.inside_both_sides({"z": (True, False), "x": (True, True), "y": (True, True)}) == (True, ["x", "y"])  # a plate in a basin
    assert PRED.inside_both_sides({"z": (True, True), "x": (True, True), "y": (True, False)}) == (True, ["z", "x"])  # a jar on a fridge shelf
    assert PRED.inside_both_sides({"z": (True, True), "x": (True, False), "y": (False, True)})[0] is False  # on a shelf beside the wall
    assert PRED.inside_both_sides({"z": (True, False), "x": (True, True), "y": (False, False)})[0] is False


def test_has_container_link():
    def link(meta, kind=None):
        return SimpleNamespace(is_meta_link=meta, meta_link_type=kind)

    sink = SimpleNamespace(links={"base_link": link(False), "togglebutton_0": link(True, "togglebutton"), "fluidsink_0": link(True, "fluidsink")})
    bowl = SimpleNamespace(links={"base_link": link(False), "fillable_0": link(True, "fillable")})
    assert not PRED.has_container_link(sink) and PRED.has_container_link(bowl)


def test_inside_atom_uses_the_rays_for_a_container_without_a_fillable_link(monkeypatch):
    bddl_utils = pytest.importorskip("omnigibson.utils.bddl_utils")
    monkeypatch.setattr(bddl_utils, "evaluate_bddl_predicate", lambda *a: False)
    monkeypatch.setattr(PRED, "inside_rays", lambda inner, container: {"ok": True, "axes_both_sides": ["x", "y"]})

    def obj(name, links):
        return SimpleNamespace(name=name, links=links, get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))

    plate = obj("plate", {"base_link": SimpleNamespace(is_meta_link=False, prim_path="/World/plate/base_link")})
    sink = obj("sink", {"base_link": SimpleNamespace(is_meta_link=False, prim_path="/World/sink/base_link")})
    bowl = obj("bowl", {"fillable_0": SimpleNamespace(is_meta_link=True, meta_link_type="fillable", prim_path="/World/bowl/fillable_0")})
    PRED.reset_branches()
    assert PRED.evaluate_predicate("inside", plate, sink) is True
    assert PRED.evaluate_predicate("inside", plate, bowl) is False  # OmniGibson judges a fillable container itself
    assert [(e["branch"], e["ok"]) for e in PRED.branches()[-2:]] == [("rays", True), ("omnigibson", False)]


# ----------------------------------------------------------------------------------------------------------------
# the oracle program
# ----------------------------------------------------------------------------------------------------------------


def test_inside_program_drops_into_the_partner():
    sc = Sidecar.load(FIXTURE)
    assert sc.problems() == []
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "navigate", "transition", "place", "retreat"]
    place, retreat = actions[-2:]
    assert (place.kind, place.relation, place.on) == ("drop", "into", "sink.n.01_1") and retreat.after is place
    assert "let it drop" in place.annotate(sc)[0] and "into the kitchen sink" in place.annotate(sc)[0]
    d = Sidecar.load(FIXTURE).to_dict()
    d["paths"] = []
    with pytest.raises(Exception, match="paths"):
        expand(sc.alias, Sidecar.from_dict(d))
