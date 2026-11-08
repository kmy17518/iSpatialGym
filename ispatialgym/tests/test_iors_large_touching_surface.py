"""Simulator-free tests of the large-scale IORS touching SURFACE place (oracle-fix 2026-11-06, owner feedback on the touching_large#02 demo:
the umbrella set down lying on hall_tree_upkvgr_0's alcove board above the bench instead of on the floor): the ``generation.touching_place``
mode selection, the ``place_kind`` "surface" routing in the oracle program / the sidecar / ``program_check.check_blocks``, the erosion and the
footprint rule on a 0.43 x 1.30 m board, the aligned release yaws of the F_goal screen, the goal check's board rule, the records and the
inside mode's ``inside_trunks`` (``pytest ispatialgym/tests/test_iors_large_touching_surface.py``)."""

from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject, SupportInterior
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
TM = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large_touching")
from ispatialgym.sampling.v2.tasks import iors_large_inside as IN  # noqa: E402
from ispatialgym.sampling.v2.tasks import iors_large_nextto as NM  # noqa: E402

SUBTASK = "inter_object_relation_satisfaction-touching_large_scale"
VARIANT = f"{SUBTASK}-umbrella-hall_tree-floor"
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecars", f"{SUBTASK}.json")
Q0 = (0.0, 0.0, 0.0, 1.0)

#: the alcove board of hall_tree_upkvgr_0 (house_double_floor_lower; decrypted collision hull c13): x [-1.222, -0.788], y [2.954, 4.258], top 0.581
BOARD_X0, BOARD_Y0, NX, NY, RES, BOARD_Z = -1.212, 2.964, 22, 65, 0.02, 0.581
#: umbrella/ziqzqo lying at its rest orientation about its root (the TRO AABB at yaw 0): the shaft along x, the crook toward +y at -x
UMBRELLA_LO, UMBRELLA_HI = (-0.392, -0.039, -0.037), (0.426, 0.0785, 0.036)
#: the generation keys of the alias (isg_customs.json)
SURFACE_GEN = {"touching_place": "surface", "inside_surface_z_m": 0.58, "inside_surface_tol_m": 0.03, "inside_stance_side": "+x", "inside_erosion_m": 0.08}


class _GP:
    def __init__(self, variant: str = VARIANT, **gen):
        self.variant, self.generation = variant, dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


def _bare():
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = "touching", {}, {}
    return s


def _box_hull(lo, hi) -> th.Tensor:
    return th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=th.float32)


def _board(headroom: float = 1.44) -> SupportInterior:
    xs = BOARD_X0 + RES * th.arange(NX, dtype=th.float32)
    ys = BOARD_Y0 + RES * th.arange(NY, dtype=th.float32)
    mask = th.ones(NX, NY, dtype=th.bool)
    return SupportInterior("hall_tree_upkvgr_0", xs, ys, mask, th.zeros_like(mask), BOARD_Z, 0.0, [], RES, headroom=th.full((NX, NY), headroom), z_requested=0.58,
                           board_pieces=["/World/hall_tree_upkvgr_0/base_link/collisions/collision_13"])


def _surface_mode(**gen):
    """SurfaceTouchingMode adopted on a bare IORSLarge with what its prepare_partner / F_goal screen / records read: the alcove board, the
    hall tree at identity yaw (its AABB's +x line 0.28 m in front of the board), the lying umbrella's hull."""
    s = _bare()
    s._adopt_mode(TM.SurfaceTouchingMode)
    s.gp = _GP(**{**SURFACE_GEN, **gen})
    lo, hi = th.tensor([-1.392, 2.176, -0.03]), th.tensor([-0.507, 7.606, 2.389])
    partner = SimpleNamespace(name="hall_tree_upkvgr_0", aabb=(lo, hi), get_position_orientation=lambda: (th.zeros(3), th.tensor(Q0)))
    s.obj = lambda role: partner if role == "partner" else f"{role}_obj"
    s.partner_static_aabb = lambda: (lo, hi)
    s.partner_joints, s.partner_moved = None, []
    s.support_top = lambda p: SimpleNamespace(z=2.389)
    s.world = SimpleNamespace(support_interior=lambda p, z, **kw: _board(), spawned={})
    s.hull, s.rest, s.floor_z, s.seg_arm = _box_hull(UMBRELLA_LO, UMBRELLA_HI), G.IDENTITY_QUAT.clone(), 0.0, "right"
    s.bindings = SimpleNamespace(one=lambda role: SimpleNamespace(name={"target": "umbrella.n.01_1", "partner": "hallstand.n.01_1"}.get(role, f"{role}.n.01_1")))
    return s


def _fits_at(c, yaw_deg: float, margin: float = TM.SURFACE_FOOTPRINT_MARGIN_M) -> bool:
    """The rule by hand: the umbrella's x / y extents at a quarter-turn yaw inside the board's cell span (+- half a cell) by ``margin``."""
    k = int(round(yaw_deg / 90.0)) % 4
    lo, hi = UMBRELLA_LO, UMBRELLA_HI
    ex = [(lo[0], hi[0]), (-hi[1], -lo[1]), (-hi[0], -lo[0]), (lo[1], hi[1])][k]  # world x extent per yaw
    ey = [(lo[1], hi[1]), (lo[0], hi[0]), (-hi[1], -lo[1]), (-hi[0], -lo[0])][k]  # world y extent per yaw
    x_lo, x_hi = BOARD_X0 - RES / 2, BOARD_X0 + RES * (NX - 1) + RES / 2
    y_lo, y_hi = BOARD_Y0 - RES / 2, BOARD_Y0 + RES * (NY - 1) + RES / 2
    return (float(c[0]) + ex[0] - margin >= x_lo and float(c[0]) + ex[1] + margin <= x_hi and float(c[1]) + ey[0] - margin >= y_lo
            and float(c[1]) + ey[1] + margin <= y_hi)


# ----------------------------------------------------------------------------------------------------------------
# the mode
# ----------------------------------------------------------------------------------------------------------------


def test_touching_place_selects_the_mode():
    """``generation.touching_place``: unset = the floor push (TouchingMode, today's), "lean" = LeanTouchingMode (the floor configuration stays
    selectable), "surface" = SurfaceTouchingMode; ``touching_push: "base"`` keeps its precedence."""
    for gen, mode in (({}, TM.TouchingMode), ({"touching_place": "lean"}, TM.LeanTouchingMode), ({"touching_place": "surface"}, TM.SurfaceTouchingMode),
                      ({"touching_place": "Surface"}, TM.SurfaceTouchingMode), ({"touching_push": "base", "touching_place": "surface"}, NM.TouchingBasePushMode)):
        s = _bare()
        s.gp = _GP(**gen)
        assert IL.IORSLarge._sample_touching(s) is mode, gen
    s = _surface_mode()
    assert isinstance(s, TM.SurfaceTouchingMode) and isinstance(s, IN.InsideMode) and isinstance(s, IL.IORSLarge) and not isinstance(s, TM.TouchingMode)
    assert type(s).prepare_partner is TM.SurfaceTouchingMode.prepare_partner and type(s)._couple is IL.IORSLarge._couple
    assert type(s)._placement_candidates_at is IN.InsideMode._placement_candidates_at and type(s).goal_pose_at is IN.InsideMode.goal_pose_at
    assert s.push_under is False
    assert s.place_ignores() == ["floor_obj"]  # the partner stays in the release's collision world: its board is under the gripper
    # TouchingMode's goal alignment: no init jitter, the release yaw on the partner's axes within half a screening step
    assert s.goal_align_partner() and s.init_yaw_jitter() == 0.0 and s.align_snap() == pytest.approx(math.pi / 16)
    assert s.end_aligned_init_yaw(math.radians(80.0), math.radians(5.0)) == pytest.approx(math.radians(85.0))
    assert s.end_aligned_init_yaw(math.radians(70.0), math.radians(5.0)) is False
    s2 = _surface_mode(goal_align_partner=False)
    assert s2.init_yaw_jitter() == IL.TARGET_JITTER and s2.end_aligned_init_yaw(1.0, 0.3) is None and s2.release_yaw(1.3) == pytest.approx(1.3)


def _surface_sidecar(place_kind: str = "surface") -> dict:
    with open(FIXTURE) as f:
        d = json.load(f)
    d2 = copy.deepcopy(d)
    d2["contacts"].pop("push")
    if not d2["contacts"]:
        d2.pop("contacts")
    d2["grasps"][0].update({"place_kind": place_kind, "place_on": "bookcase.n.01_1"})
    d2["stances"].pop("push", None)
    d2["transitions"] = [t for t in d2.get("transitions", []) if "push" not in (t.get("from"), t.get("to"))]
    return d2


def test_surface_place_kind_routes_like_the_lean():
    """A sidecar whose grasp records ``place_kind`` "surface" and no push validates (sidecar + ``program_check.check_blocks``) and expands to the
    lean's program: the pick, the carry and a drop-release at the recorded place on the partner's board, the retreat - no push."""
    d2 = _surface_sidecar()
    sc = Sidecar.from_dict(d2)
    assert sc.problems() == []
    assert PC.check_blocks(SUBTASK, d2) == []
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "navigate", "place", "retreat"]
    place = actions[4]
    assert place.kind == "drop" and place.on == "bookcase.n.01_1" and actions[5].after is place
    assert [a.name for a in expand(sc.variant, Sidecar.from_dict(_surface_sidecar("lean")))] == [a.name for a in actions]
    # without the relaxation the push is required again: check_blocks names it, the sidecar / the program refuse
    d3 = _surface_sidecar("support")
    assert "missing block contacts.push" in PC.check_blocks(SUBTASK, d3)
    assert any("contacts.push" in p for p in Sidecar.from_dict(d3).problems())
    with pytest.raises(SidecarError, match="surface"):
        expand(sc.variant, Sidecar.from_dict(d3))


# ----------------------------------------------------------------------------------------------------------------
# the board: erosion and the footprint rule
# ----------------------------------------------------------------------------------------------------------------


def test_board_erosion_and_footprint_fit():
    """hall_tree_upkvgr_0's alcove board (0.44 x 1.30 m raster, 1430 cells): the inside mode's default erosion (the half extent 0.409 + 2 cm)
    leaves no goal cell on a board as deep as the umbrella is long; ``inside_erosion_m`` 0.08 leaves ~700, of which the footprint rule
    keeps those where the umbrella lies ALONG the board (yaw 90 / 270 deg) - never across it."""
    with pytest.raises(Reject) as e:
        _surface_mode(inside_erosion_m=None).prepare_partner()
    assert e.value.reason == "inside_no_goal_cell"
    s = _surface_mode()
    s.prepare_partner()
    inside = s.template_facts["inside_surface"]
    assert inside["n_board_cells"] == NX * NY == 1430 and inside["erosion_m"] == pytest.approx(0.08)
    board = _board()
    eroded = IN.erode_from(board.cells_usable(0.073, IN.HEADROOM_MARGIN_M), IN.raster_rim(board.mask, board.xs, board.ys), 0.08)
    assert inside["n_goal_cells"] == len(eroded) > 600 and inside["stance_edges"] == ["+u"]
    assert float(eroded[:, 0].min()) >= BOARD_X0 + 0.08 - 1e-3 and float(eroded[:, 0].max()) <= BOARD_X0 + RES * (NX - 1) - 0.08 + 1e-3  # 0.26 m of depth left
    (edge,) = s.place["edges"]
    assert float(edge.inward[0]) == pytest.approx(-1.0) and edge.overhang == pytest.approx(-0.507 - (BOARD_X0 + RES * (NX - 1)), abs=1e-4)  # the canopy's 0.28 m
    facts = s.template_facts["surface_touching"]
    assert facts["n_goal_cells_eroded"] == len(eroded) and facts["fit_per_yaw"]["0.0"] == 0 and facts["fit_per_yaw"]["180.0"] == 0
    assert facts["fit_per_yaw"]["90.0"] > 0 and facts["fit_per_yaw"]["270.0"] > 0 and facts["misfits"]["footprint_off_board"] > 0
    cells = s.place["cells"]
    assert len(cells) == facts["n_goal_cells"] > 0 and s.template_facts["partner"]["n_goal_cells"] == len(cells)
    assert all(_fits_at(c, 90.0) or _fits_at(c, 270.0) for c in cells)
    assert not any(_fits_at(c, 0.0) or _fits_at(c, 180.0) for c in cells)
    assert len(cells) == sum(1 for c in eroded if _fits_at(c, 90.0) or _fits_at(c, 270.0))
    # the shaft's 0.82 m along y: the root at least ~0.39 m from either end of the board
    assert float(cells[:, 1].min()) >= BOARD_Y0 - RES / 2 + 0.392 - 1e-4 and float(cells[:, 1].max()) <= BOARD_Y0 + RES * (NY - 1) + RES / 2 - 0.392 + 1e-4
    # a board too short for the shaft: no cell fits
    s3 = _surface_mode()
    s3.world.support_interior = lambda p, z, **kw: SupportInterior("hall_tree_upkvgr_0", _board().xs, _board().ys[:35], th.ones(NX, 35, dtype=th.bool),
                                                                  th.zeros(NX, 35, dtype=th.bool), BOARD_Z, 0.0, [], RES, headroom=th.full((NX, 35), 1.44))
    with pytest.raises(Reject) as e:
        s3.prepare_partner()
    assert e.value.reason == "surface_no_fit"


def test_touching_wall_keeps_the_cells_at_the_wall():
    """``generation.touching_wall`` "+y" (the wardrobe's side at the board's +y end): only the cells whose footprint ends within
    ``touching_wall_hug_m`` of that end of the board; a malformed side is a spec rejection."""
    s0 = _surface_mode()
    s0.prepare_partner()
    s = _surface_mode(touching_wall="+y")
    s.prepare_partner()
    cells, all_cells = s.place["cells"], s0.place["cells"]
    assert 0 < len(cells) < len(all_cells)
    y_end = BOARD_Y0 + RES * (NY - 1) + RES / 2
    hug = TM.SURFACE_WALL_HUG_M
    for c in cells:  # the +y end of the footprint (0.426 at 90 deg, 0.392 at 270 deg) within hug + a cell of the board's end
        ends = [float(c[1]) + e for yaw, e in ((90.0, 0.426), (270.0, 0.392)) if _fits_at(c, yaw)]
        assert any(y_end - end <= hug + RES + 1e-4 for end in ends)
    assert float(cells[:, 1].min()) > float(all_cells[:, 1].max()) - hug - 2 * RES - 1e-4
    assert s.template_facts["surface_touching"]["wall"] == {"side": [0.0, 1.0], "hug_m": hug}
    with pytest.raises(Reject) as e:
        _surface_mode(touching_wall="up").prepare_partner()
    assert e.value.reason == "spec"


# ----------------------------------------------------------------------------------------------------------------
# F_goal at the aligned, fitting release yaws
# ----------------------------------------------------------------------------------------------------------------


def _stance_c() -> Stance:
    edge = EdgeFrame.from_edge("+u", th.tensor([-0.792, 3.604]), th.tensor([1.0, 0.0]), 1.28, 0.285)
    xy = edge.base_xy(0.20, 0.0)
    return Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, -0.3, 0.20, edge, None, 0.0, None, ["right"])


def test_release_feasible_screens_the_aligned_fitting_yaws(monkeypatch):
    """F_goal is screened at the partner-aligned yaw of each screened yaw (the yaw the draw releases at) whose footprint fits at the goal cell -
    at the board's middle both along-board yaws, near its +y end the one whose shorter half points there; the pairs come back at the screened
    yaws (F_init's keys) and the draw's aligned init yaw lands on the screened release."""
    s = _surface_mode()
    s.prepare_partner()
    seen = []

    def screen(self, release_xy, yaws, release_pose, place, arm, ignore_place):
        seen.append([float(y) for y in yaws])
        return [(round(float(y), 4), "grasp_0") for y in yaws]  # the feasible-set cache hands yaws back to 4 decimals

    monkeypatch.setattr(IL.IORSLarge, "release_feasible", screen)
    d_yaw = math.radians(7.0)
    screened = [k * 2.0 * math.pi / 16 + d_yaw for k in range(16)]
    out = s.release_feasible((-0.892, 3.60), screened, None, _stance_c(), "right", ["floor_obj"])
    (aligned,) = seen
    assert sorted(round(math.degrees(y) % 360.0, 3) for y in aligned) == [90.0, 270.0]
    assert sorted(round(math.degrees(y) % 360.0, 3) for y, _ in out) == [97.0, 277.0]  # the screened yaws, not the aligned ones
    for y, _ in out:  # the draw: the init yaw (screened - d_yaw) turned so the carried yaw is the screened release
        y_init = s.end_aligned_init_yaw(y - d_yaw, d_yaw)
        assert any(abs(math.remainder(y_init + d_yaw - a, 2 * math.pi)) < 1e-6 for a in aligned)
    seen.clear()
    near_end = (-0.892, float(s.place["cells"][:, 1].max()))
    s.release_feasible(near_end, screened, None, _stance_c(), "right", ["floor_obj"])
    assert [round(math.degrees(y) % 360.0, 3) for y in seen[0]] == [270.0]  # at 90 deg the 0.426 m half would leave the board
    seen.clear()
    assert s.release_feasible((-0.892, 3.0), screened, None, _stance_c(), "right", ["floor_obj"]) == [] and seen == []  # nothing fits: not screened


# ----------------------------------------------------------------------------------------------------------------
# goal check and records
# ----------------------------------------------------------------------------------------------------------------


def test_goal_check_keeps_the_target_on_the_board(monkeypatch):
    """No drop-test protrusion (InsideMode's rule is not run: touching is the goal); the settled umbrella must still lie on the board."""
    monkeypatch.setattr(IL.IORSLarge, "refine_targets_from_goal", lambda self, draft: None)
    monkeypatch.setattr(IN.InsideMode, "refine_targets_from_goal", lambda self, draft: (_ for _ in ()).throw(AssertionError("protrusion rule run")))
    s = _surface_mode()
    s.prepare_partner()
    pose = {"p": th.tensor([-0.95, 3.6, BOARD_Z + 0.037])}
    target = SimpleNamespace(get_position_orientation=lambda: (pose["p"], th.tensor(Q0)))
    s.obj = (lambda obj: (lambda role: target if role == "target" else obj(role)))(s.obj)
    s.world.hull_at = lambda hull, p: G.local_points_world(hull, p)
    d = Draft()
    s.refine_targets_from_goal(d)
    assert d.summary["surface_rest"]["bottom_above_surface_m"] == pytest.approx(0.0, abs=1e-4) and d.summary["surface_rest"]["surface_z"] == pytest.approx(BOARD_Z)
    pose["p"] = th.tensor([-0.40, 3.6, 0.037])  # rolled off onto the floor in front of the bench
    with pytest.raises(Reject) as e:
        s.refine_targets_from_goal(Draft())
    assert e.value.reason == "goal_off_surface"


def test_surface_records_and_the_place_kind_after_the_goal_check(monkeypatch):
    """``grasps[0].place_kind`` "surface" at the draw and again after InsideMode's goal-check facts (which write the drop's); ``summary.
    surface_touching`` names the board, the goal cell and the release yaw."""
    s = _surface_mode()
    s.prepare_partner()
    grasp = SimpleNamespace(extras={})
    s.sidecar = SimpleNamespace(grasp_of=lambda name: grasp if name == "umbrella.n.01_1" else None)
    d = Draft()
    gc = [-0.892, 3.60]
    d.summary.update({"goal_cell": gc, "goal_quat": [float(v) for v in G.quat_from_yaw(math.pi / 2)]})
    s.surface_records(d)
    assert grasp.extras == {"place_kind": "surface", "then": "surface", "goal_cell": gc, "partner": "hallstand.n.01_1"}
    st = d.summary["surface_touching"]
    assert st["surface_z"] == pytest.approx(BOARD_Z) and st["release_yaw_deg"] == pytest.approx(90.0) and st["footprint"] == "fits" and st["goal_cell"] == gc

    def inside_after(self, draft):
        facts = {"place_kind": "drop", "surface_z": BOARD_Z}
        grasp.extras.update(facts)
        draft.summary["release"] = facts

    monkeypatch.setattr(IN.InsideMode, "after_goal_check", inside_after)
    s.after_goal_check(d)
    assert grasp.extras["place_kind"] == "surface" and d.summary["release"]["place_kind"] == "surface"


# ----------------------------------------------------------------------------------------------------------------
# the inside mode's stance C trunks
# ----------------------------------------------------------------------------------------------------------------


class _StanceC:
    _placement_candidates_at = IN.InsideMode._placement_candidates_at
    num = IL.IORSLarge.num

    def __init__(self, band=None, **gen):
        self.gp = _GP(**gen)
        self.seg_arm = "right"
        self.partner = SimpleNamespace(name="hall_tree_upkvgr_0")
        self.place = {"half": 0.409, "height": 0.073, "z": BOARD_Z, "cells": th.tensor([[-0.892, 3.60], [-0.912, 3.70]])}
        self.template_facts = {}
        self.hull, self.rest = th.zeros(8, 3), th.tensor(Q0)
        self._band = band

    def inside_band(self, arm):
        return self._band

    def band_cells_world(self, rb, st):
        return self.place["cells"]

    def _snap(self, cells, goal_cells, lattice):
        return goal_cells

    def reach_margin(self, st, point, arm):
        return 0.2 * st.translate - 0.31 * st.tilt

    def goal_pose_at(self, gc, quat):
        return th.tensor([gc[0], gc[1], 0.62]), quat

    def hull_corners_world(self, hull, pose):
        return th.zeros(8, 3)


def test_inside_trunks_replace_the_no_band_grid(monkeypatch):
    """``generation.inside_trunks`` (opt-in): without a goal-side band stance C is tried at these trunks only - the whole grid's reach-margin cut
    keeps the deepest leans, out of reach / frame for the umbrella's release on the board; with a band its trunks stay."""
    monkeypatch.setattr(IN.P0, "base_ok", lambda s, xy: True)
    monkeypatch.setattr(IN.P0, "body_free", lambda s, cands, **kw: th.ones(len(cands), dtype=th.bool))
    monkeypatch.setattr(IN.P0, "camera_poses", lambda s, cands: list(cands))
    monkeypatch.setattr(IN.P0, "in_frame", lambda s, cam, pts: True)
    monkeypatch.setattr(IN.P0, "axis_angle", lambda s, cam, point: 0.3)
    edge = EdgeFrame.from_edge("+u", th.tensor([-0.792, 3.604]), th.tensor([1.0, 0.0]), 1.28, 0.285)
    default = _StanceC(inside_stand_offs_m=[0.10, 0.20, 0.30])._placement_candidates_at(edge, (0.2, 4.7), "right")
    margin = {(t, k): _StanceC().reach_margin(SimpleNamespace(translate=t, tilt=k), None, None) for t in IN.RB.TRANSLATES for k in IN.RB.TILTS}
    deepest = set(sorted(margin, key=lambda tk: -margin[tk])[:IN.NO_BAND_TRUNKS])
    assert {(c.translate, c.tilt) for c in default} == deepest and all(k <= -0.6 for _, k in deepest)  # the deepest leans by reach margin
    trunks = [[0.5, -0.3], [0.5, 0.0], [0.75, -0.3]]
    s = _StanceC(inside_stand_offs_m=[0.10, 0.20, 0.30], inside_trunks=trunks)
    out = s._placement_candidates_at(edge, (0.2, 4.7), "right")
    assert {(c.translate, c.tilt) for c in out} == {tuple(t) for t in trunks} and {c.stand_off for c in out} == {0.10, 0.20, 0.30}
    assert s.template_facts["placement_stats"]["lattice"] == 3 * len(P0_laterals(edge, 0.409)) * len(trunks)
    band = SimpleNamespace(trunks=lambda: [(0.25, 0.0)], meta={"lattice": 0.08})
    out = _StanceC(band=band, inside_trunks=trunks)._placement_candidates_at(edge, (0.2, 4.7), "right")
    assert {(c.translate, c.tilt) for c in out} == {(0.25, 0.0)}


def P0_laterals(edge: EdgeFrame, half: float):
    half_run = edge.length / 2 - (half + 0.10)
    return IN.P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]


def test_stance_c_frame_test_at_the_release_yaw(monkeypatch):
    """InsideMode's stance-C frame test poses the target at the mode's ``frame_quat_at`` (the surface place: the first release yaw that fits,
    along the board) instead of the rest yaw (the umbrella across the 0.43 m board, its near end 0.43 m closer to the base)."""
    s = _surface_mode()
    s.prepare_partner()
    q = s.frame_quat_at((-0.892, 3.60))
    assert math.degrees(G.yaw_of(q)) % 360.0 == pytest.approx(90.0, abs=1e-3)
    assert s.frame_quat_at((-0.892, 3.0)) is s.rest  # nothing fits there: the rest orientation

    class _Framed(_StanceC):
        def __init__(self, quat=None, **gen):
            super().__init__(**gen)
            self.hull = _box_hull(UMBRELLA_LO, UMBRELLA_HI)
            if quat is not None:
                self.frame_quat_at = lambda gc: quat

        def hull_corners_world(self, hull, pose):
            from ispatialgym.sampling.v2 import tabletop as TT

            return TT.hull_corners_world(hull, pose)

    monkeypatch.setattr(IN.P0, "base_ok", lambda s_, xy: True)
    monkeypatch.setattr(IN.P0, "body_free", lambda s_, cands, **kw: th.ones(len(cands), dtype=th.bool))
    monkeypatch.setattr(IN.P0, "camera_poses", lambda s_, cands: list(cands))
    monkeypatch.setattr(IN.P0, "in_frame", lambda s_, cam, pts: float(pts[:, 0].max() - pts[:, 0].min()) < 0.3)  # the near end stays ahead of the frame's edge
    monkeypatch.setattr(IN.P0, "axis_angle", lambda s_, cam, point: 0.3)
    edge = EdgeFrame.from_edge("+u", th.tensor([-0.792, 3.604]), th.tensor([1.0, 0.0]), 1.28, 0.285)
    gen = {"inside_stand_offs_m": [0.20], "inside_trunks": [[0.5, -0.3]]}
    assert _Framed(**gen)._placement_candidates_at(edge, (0.2, 4.7), "right") == []  # at the rest yaw: across the board, out of frame
    assert len(_Framed(quat=q, **gen)._placement_candidates_at(edge, (0.2, 4.7), "right")) == len(P0_laterals(edge, 0.409))


def test_inside_only_switches_leave_the_surface_stance_c_unchanged(monkeypatch):
    """Owner 5t (2026-11-08): ISPATIALGYM_INSIDE_ARM_SIDE and the stance-C map cap (ISPATIALGYM_INSIDE_MAP_CAP) act for the relation "inside"
    only. SurfaceTouchingMode (touching_large_02, relation "touching") inherits InsideMode's stance C: with both switches on its candidates,
    goal cells and placement_stats are exactly those with both off; the same stance C with relation "inside" is changed by them."""
    monkeypatch.setattr(IN.P0, "base_ok", lambda s_, xy: True)
    monkeypatch.setattr(IN.P0, "body_free", lambda s_, cands, **kw: th.ones(len(cands), dtype=th.bool))
    monkeypatch.setattr(IN.P0, "camera_poses", lambda s_, cands: list(cands))
    monkeypatch.setattr(IN.P0, "in_frame", lambda s_, cam, pts: True)
    monkeypatch.setattr(IN.P0, "axis_angle", lambda s_, cam, point: 0.3)
    edge = EdgeFrame.from_edge("+u", th.tensor([-0.792, 3.604]), th.tensor([1.0, 0.0]), 1.28, 0.285)
    cells = th.tensor([[-0.892, 3.30 + 0.04 * i] for i in range(16)])
    band = SimpleNamespace(trunks=lambda: [(t, k) for t in (0.25, 0.5, 0.75) for k in (0.0, -0.3)], meta={"lattice": 0.08, "mode": "map_view"})

    def run(relation, on):
        monkeypatch.setenv(IN.INSIDE_ARM_SIDE_ENV, "1" if on else "0")
        monkeypatch.setenv(IN.INSIDE_MAP_CAP_ENV, "1" if on else "0")
        s = _StanceC(band=band, inside_stand_offs_m=[0.20, 0.30])
        s.relation, s.kit = relation, SimpleNamespace(stats={})
        s.place["cells"] = cells
        out = s._placement_candidates_at(edge, (0.2, 4.7), "right")
        return ([(tuple(round(float(v), 6) for v in c.base_pose2d), c.translate, c.tilt, c.stand_off, c.lateral, c.extras["goal_cell"]) for c in out],
                s.template_facts["placement_stats"], s.kit.stats)

    off, on = run("touching", False), run("touching", True)
    assert on[0] == off[0] and on[1] == off[1] and off[0]
    assert not any(k.startswith(("inside_arm_side.", "inside_map_cap.")) for k in on[2])
    assert "map_cap" not in on[1]
    i_off, i_on = run("inside", False), run("inside", True)
    assert i_off[0] == off[0]  # the switches off: the inside and the surface stance C alike
    assert i_on[0] != i_off[0] and i_on[2].get("inside_arm_side.aimed", 0) > 0 and i_on[2].get("inside_map_cap.dropped", 0) > 0 and "map_cap" in i_on[1]
