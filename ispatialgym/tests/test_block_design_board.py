"""Block design with the puzzle board (spec §6.5): the init's flip-room move keeps a board position on the top at the
manipulation stance (owner 2026-10-12, block_design#01). A rhjoby-like top: a 1.16 m edge, 2.90 m deep, the 2 cm raster of
``SimWorld.support_top``; the robot at the short edge, init stand-off 0.45, Phase 1c's lateral range [0.044, 0.547]."""
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.geometry import CameraModel
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import block_design as BD

BOUNDS = [0.044, 0.5474]
BOARD_XY = [[0.46, -0.02], [0.44, -0.10], [0.48, 0.04], [0.52, -0.06], [0.60, 0.05]]
GEN = {"patterns": {"test_rbbr": ["red", "blue", "blue", "red"]}, "block_no_flip": True, "flip_directions": ["right"],
       "manip_lateral_shifts_m": [0.35, 0.45, 0.25], "board_base_xy_candidates": BOARD_XY}


def _top_and_edge():
    xs, ys = th.arange(0.01, 2.92, 0.02), th.arange(0.01, 1.18, 0.02)  # cell centres: 2.90 m deep, a 1.16 m edge along y
    mask = th.ones(len(xs), len(ys), dtype=th.bool)
    top = SupportTop("breakfast_table_rhjoby_0", xs, ys, mask, th.zeros_like(mask), 0.81, 0.0, [], 0.02)
    name, mid, normal, length = next(e for e in top.edges() if e[0] == "-u")
    return top, EdgeFrame.from_edge(name, mid, normal, length)


def _sampler(top, board=True, gen=None):
    s = object.__new__(BD.BlockDesign)
    s.spec = SimpleNamespace(generation={**GEN, **(gen or {})})
    s.subtask, s.defaults, s.role = BD.SUBTASK, BD.GD.defaults_for(BD.SUBTASK), "targets"
    s.gp = SimpleNamespace(variant="configuration_matching-block_design-block_4-breakfast_table_board", roles={"targets": None, **({"board": None} if board else {})})
    s.kit = SimpleNamespace(trav_free=lambda xy: True)
    s.top = top
    return s


def _init(edge, lateral):
    st = Stance("init", (0.0, 0.0, 0.0), 0.54, 0.0, 0.45, edge).shifted(stand_off=0.45, lateral=lateral, name="init")
    st.extras["lateral_bounds"] = list(BOUNDS)
    return st


def _half(s):
    return 0.5 * float(s._board_geometry()["plate_m"])


def test_board_off_the_top_test_is_the_placement_test():
    """``_board_on_top`` (the test ``_place_board`` runs) on the rhjoby-like top: by depth stand-off 0.45 keeps (0.52, -0.06) and
    (0.60, +0.05) on the top and stand-off 0.55 only (0.60, +0.05); 3 cm from the edge's end (manipulation lateral 0.547)
    (0.60, +0.05) is off the end, so stand-off 0.55 has none - the 195 / 195 attempts of s3 c1_k1."""
    top, edge = _top_and_edge()
    s = _sampler(top)
    end = _init(edge, BOUNDS[1] - 0.35)
    assert s._board_fit(end, 0.35, _half(s)) == {"0.35": ["(0.46, -0.02)", "(0.44, -0.10)", "(0.52, -0.06)"], "0.45": ["(0.52, -0.06)"], "0.55": []}
    k4 = _init(edge, 0.07)  # eval seed k4's draw: its manipulation stance at 0.42
    assert s._board_fit(k4, 0.35, _half(s)) == {"0.35": [f"({d:.2f}, {l:+.2f})" for d, l in BOARD_XY], "0.45": ["(0.52, -0.06)", "(0.60, +0.05)"],
                                                "0.55": ["(0.60, +0.05)"]}
    manip = _init(edge, 0.42).shifted(stand_off=0.55, name="manipulation")
    right, toward, _ = s._table_axes(manip)
    assert s._board_on_top(manip, 0.60, 0.05, _half(s), right, -toward) and not s._board_on_top(manip, 0.52, -0.06, _half(s), right, -toward)


@pytest.mark.parametrize("drawn", [0.24, 0.38, 0.48])
def test_flip_room_move_puts_the_board_on_the_top_at_every_stand_off(drawn):
    """Eval seeds k1 / k2 / k3 drew 0.24 / 0.38 / 0.48 on the 1.16 m edge. The minimum move (to 0.197, the manipulation stance at
    the range's end 0.547) leaves stand-off 0.55 without a board position on the top - the failing case; the move now goes on to
    the first lateral whose manipulation stance has one at every Phase 2 stand-off, recorded for template_facts."""
    top, edge = _top_and_edge()
    s = _sampler(top)
    init = _init(edge, drawn)
    today = init.shifted(lateral=BOUNDS[1] - 0.35)
    assert not all(s._board_fit(today, 0.35, _half(s)).values())
    moved = s._init_with_room_for_flips(init)
    fit = s._board_fit(moved, 0.35, _half(s))
    assert all(fit.values()) and set(fit) == {f"{so:.2f}" for so in TT.STAND_OFFS}
    assert moved.lateral == pytest.approx(0.1274, abs=1e-6) and moved.translate == init.translate and moved.stand_off == init.stand_off
    assert moved.extras["flip_room_shift_m"] == pytest.approx(moved.lateral - drawn) and moved.extras["lateral_bounds"] == BOUNDS
    # one step less still leaves stand-off 0.55 bare: the smallest such move
    assert not all(s._board_fit(init.shifted(lateral=moved.lateral + BD.FLIP_ROOM_STEP), 0.35, _half(s)).values())
    rec = s._flip_room
    assert rec["lateral_drawn"] == drawn and rec["lateral"] == pytest.approx(0.1274) and rec["move_m"] == pytest.approx(0.1274 - drawn, abs=1e-4)
    assert rec["minimum_move_m"] == pytest.approx(BOUNDS[1] - 0.35 - drawn, abs=1e-4) and rec["board_fit"] == fit
    assert rec["board_fit_at_minimum_move"]["0.55"] == [] and "stand-off 0.55" in rec["reason"] and "every Phase 2 stand-off has one" in rec["reason"]


def test_flip_room_move_leaves_the_other_inits_alone():
    """The inits that need no move (legacy seed 0's 0.06, eval seed k4's 0.07) are returned as drawn - only the record is
    added; without a board the move is the minimum one, as before."""
    top, edge = _top_and_edge()
    s = _sampler(top)
    for drawn in (0.06, 0.07):
        init = _init(edge, drawn)
        assert s._init_with_room_for_flips(init) is init
        assert s._flip_room["move_m"] == 0.0 and all(s._flip_room["board_fit"].values())
    bare = _sampler(top, board=False)
    moved = bare._init_with_room_for_flips(_init(edge, 0.38))
    assert moved.lateral == pytest.approx(BOUNDS[1] - 0.35) and bare._flip_room["move_m"] == pytest.approx(BOUNDS[1] - 0.35 - 0.38, abs=1e-4)
    assert "board_fit" not in bare._flip_room
    # no flips expected (a pattern of distinct colours): nothing moves, nothing is recorded
    plain = _sampler(top, gen={"patterns": {"p": ["red", "blue", "white", "yellow"]}})
    init = _init(edge, 0.38)
    assert plain._init_with_room_for_flips(init) is init and plain._flip_room is None


def test_flip_room_move_is_recorded_in_template_facts(monkeypatch):
    top, edge = _top_and_edge()
    drawn = _init(edge, 0.38)
    corners = th.tensor([[x, y, z] for x in (-0.0127, 0.0127) for y in (-0.0127, 0.0127) for z in (0.0, 0.0254)])
    monkeypatch.setattr(TT, "edge_candidates", lambda s, top, rng: iter([(edge, None)]))
    monkeypatch.setattr(TT, "ensure_bands", lambda *a, **k: SimpleNamespace(path=None))
    monkeypatch.setattr(TT, "left_band_compute", lambda: False)
    monkeypatch.setattr(TT, "load_left_contact_band", lambda *a, **k: None)
    monkeypatch.setattr(TT, "phase1b_init_trunk", lambda s, st, h, **k: drawn.with_trunk(drawn.translate, drawn.tilt))
    monkeypatch.setattr(TT, "phase1c_lateral", lambda s, init, margins, rng, **k: init)
    monkeypatch.setattr(TT, "phase2_manipulation_stance", lambda s, init_k, half, **kw: [init_k.with_trunk(0.5, 0.0, "manipulation")])
    s = _sampler(top, gen={"phase2_split_cone": False})
    s.ctx = SimpleNamespace(base_seed=0)
    s.world = SimpleNamespace(local_hull=lambda obj: corners)
    s.choice = SimpleNamespace(groups={"targets": [SimpleNamespace(dataset="none", category="toy_dice", model="isgblk", scale=None)]})
    s.objs = lambda role: [object()] * 4
    s.obj = lambda role: SimpleNamespace(name="breakfast_table_rhjoby_0")
    s.support_top = lambda support: top
    s.template_facts, s.faces, s.bands, s.left_bands, s.init_stance, s.manip_candidates = {}, dict(BD.DEFAULT_FACES), None, None, None, []
    BD.BlockDesign.sample_template(s)
    rec = s.template_facts["init_flip_room_move"]
    assert s.init_stance.lateral == pytest.approx(0.1274) and rec["lateral_drawn"] == 0.38 and rec["lateral"] == pytest.approx(0.1274)
    assert rec["move_m"] == pytest.approx(-0.2526, abs=1e-4) and rec["minimum_move_m"] == pytest.approx(-0.1826, abs=1e-4)
    assert all(rec["board_fit"].values()) and rec["board_fit_at_minimum_move"]["0.55"] == [] and rec["reason"]
    # the manipulation candidates stand where the scan's shifts put them from the moved init (0.45 leaves the range)
    assert sorted({round(c.lateral - s.init_stance.lateral, 6) for c in s.manip_candidates}) == [0.0, 0.25, 0.35]


def test_board_fit_also_moves_an_init_whose_shift_fits():
    """The -v draws of eval seeds k0 / k5 (0.15): the shift fits the range, but the manipulation stance at 0.50 has no board
    position on the top at stand-off 0.55 ((0.60, +0.05) is off the edge's end) - the same search from the drawn lateral on."""
    top, edge = _top_and_edge()
    s = _sampler(top)
    init = _init(edge, 0.15)
    assert s._board_fit(init, 0.35, _half(s))["0.55"] == []
    moved = s._init_with_room_for_flips(init)
    assert moved.lateral == pytest.approx(0.13, abs=1e-6) and all(s._board_fit(moved, 0.35, _half(s)).values())
    assert not all(s._board_fit(init.shifted(lateral=0.14), 0.35, _half(s)).values())
    rec = s._flip_room
    assert rec["minimum_move_m"] == 0.0 and rec["move_m"] == pytest.approx(-0.02, abs=1e-4) and rec["board_fit_at_minimum_move"]["0.55"] == []
    assert "fits the lateral range" in rec["reason"] and "stand-off 0.55" in rec["reason"]
    assert _sampler(top, board=False)._init_with_room_for_flips(init) is init
    # no move inside the range gives more stand-offs (the range ends 1 mm right of the drawn lateral): the init stays
    narrow = _init(edge, 0.15)
    narrow.extras["lateral_bounds"] = [0.149, BOUNDS[1]]
    assert s._init_with_room_for_flips(narrow) is narrow and s._flip_room["move_m"] == 0.0 and s._flip_room["board_fit"]["0.55"] == []
    assert "no move along the range gives more" in s._flip_room["reason"]


# s3 c1_k0's region cells (base depth, lateral of the manipulation stance; the init 0.35 m to its right): candidate 10 (also c1_k1's
# candidate 11) and candidate 11
REGION_10 = [(0.56, -0.24), (0.64, -0.16), (0.64, -0.32), (0.56, -0.08), (0.56, -0.40), (0.72, -0.24)]
REGION_11 = [(0.56, -0.16), (0.64, -0.08), (0.64, -0.24), (0.56, -0.32), (0.56, 0.00), (0.72, -0.16)]


def _layout_scene(stand_off=0.45):
    top, edge = _top_and_edge()
    s = _sampler(top)
    s.edge = 0.0254
    init = _init(edge, 0.1274).with_trunk(0.55, 0.0, "init")
    manip = init.shifted(stand_off=stand_off, lateral=init.lateral + 0.35, name="manipulation")
    cam = C.camera_pose(C.torso_q(init.translate, init.tilt), base_pose=tuple(float(v) for v in init.base_pose2d))
    cm = CameraModel(1080, 1080, 459.0, 459.0, 540.0, 540.0)

    def img_score(xy):
        return float(cm.project(cam, th.tensor([[float(xy[0]), float(xy[1]), 0.81 + 0.0127]]))[0][0].sum())

    faces = [{c: f for f, c in BD.DEFAULT_FACES.items()}[p] for p in GEN["patterns"]["test_rbbr"]]
    return s, init, manip, img_score, faces


def _clear_of(board, region, half_b=0.5 * 0.0908 + 0.5 * 0.0254 + 0.06):
    return [(d, l) for d, l in region if abs(l - board[1]) > half_b or abs(d - board[0]) > half_b]


def test_flip_set_needs_two_flip_cells_with_a_clear_approach_side():
    """Why the -v draws ended at the layout: at shift 0.35 the flip cells lie in lateral [-0.28, -0.10] of the manipulation base
    (two columns of the region's diagonal lattice) and a flipping block needs no other block within 0.20 m to its right and
    0.12 m in depth. Candidate 10's cells have no set at all; candidate 11's have one only while the board keeps the near row
    clear - (0.52, -0.06), the first position on the top at stand-off 0.45, takes it, (0.44, -0.10) keeps it."""
    s, init, manip, img_score, faces = _layout_scene()

    def has_set(cells_b):
        return s._has_flip_set([manip.base_to_world(th.tensor(c)) for c in cells_b], 4, img_score, faces, manip, init, "right", True)

    assert [c for c in REGION_10 if s._flip_cell_ok(manip.base_to_world(th.tensor(c)), manip, init, "right")] == [(0.56, -0.24), (0.64, -0.16), (0.72, -0.24)]
    assert not has_set(REGION_10) and not has_set(_clear_of((0.52, -0.06), REGION_10))
    assert has_set(REGION_11) and has_set(_clear_of((0.44, -0.10), REGION_11))
    assert not has_set(_clear_of((0.52, -0.06), REGION_11)) and not has_set(_clear_of((0.60, 0.05), REGION_11))
    # the plan of candidate 11's set: the two flip cells of one column flip, the blocks left of them show their faces
    cs = sorted([manip.base_to_world(th.tensor(c)) for c in [(0.56, -0.16), (0.64, -0.08), (0.56, 0.00), (0.72, -0.16)]], key=img_score)
    flips = s._flip_plan(cs, img_score, faces, manip, init, "right", True)
    assert flips is not None and sorted(round(float(manip.world_to_base(cs[i])[0]), 2) for i in flips) == [0.56, 0.72]
    # a neighbour 8 cm right of a flipping block, 8 cm nearer, is on its approach side: no plan
    blocked = cs[:3] + [manip.base_to_world(th.tensor((0.64, -0.24)))]
    assert s._flip_plan(sorted(blocked, key=img_score), img_score, faces, manip, init, "right", True) is None


def test_board_placement_keeps_a_flip_set():
    """``_place_board`` takes the first position that also leaves a cell set the layout accepts: at stand-off 0.35 every position is
    on the top; (0.46, -0.02) passes the count of flip cells but leaves candidate 11's cells no set, (0.44, -0.10) keeps one. Without
    the test the first position is kept, as before; when no position keeps a set the board is not placed."""
    s, init, manip, img_score, faces = _layout_scene(stand_off=0.35)
    cells = [manip.base_to_world(th.tensor(c)) for c in REGION_11]
    probes = []
    s.bindings = SimpleNamespace(one=lambda role: SimpleNamespace(name="tray.n.01_1"))
    s.objects = {"tray.n.01_1": object()}
    s._teleport = lambda objects: None
    s.world = SimpleNamespace(park=lambda obj, i: None)
    s.ikc = SimpleNamespace(exact=lambda poses, st, arm, **kw: probes.append(st) or SimpleNamespace(ok=True))
    right, toward, yaw = s._table_axes(manip)
    eligible = lambda c: s._flip_cell_ok(c, manip, init, "right")  # noqa: E731
    layout_ok = lambda kept: s._has_flip_set(kept, 4, img_score, faces, manip, init, "right", True)  # noqa: E731

    def placed_at(**kw):
        _, pose, kept = s._place_board(manip, right, toward, yaw, list(cells), eligible, 4, None, **kw)
        b = manip.world_to_base(pose[0][:2])
        return (round(float(b[0]), 3), round(float(b[1]), 3)), len(kept)

    assert placed_at() == ((0.46, -0.02), 5)
    n_probe = len(probes)
    assert placed_at(layout_ok=layout_ok) == ((0.44, -0.10), 6)
    assert len(probes) == 3 * n_probe  # the first position's place probe still runs (before its layout test), then the next one's
    with pytest.raises(Reject) as e:
        s._place_board(manip, right, toward, yaw, list(cells), eligible, 4, None, layout_ok=lambda kept: False)
    assert e.value.reason == "board_placement" and "no cell set of them lets every repeated-colour block flip" in str(e.value)
