"""Simulator-free tests of the large-scale IORS inside mode (owner 2026-10-12 ~16:05 item 1; spec §5.6 "Inside (large-scale)"): the interior
goal geometry, stance C's side, the release height, the drop test's protrusion, the ``inside`` rays rule of a container without a fillable
meta link and the oracle program (``pytest ispatialgym/tests/test_iors_large_inside.py``)."""

from __future__ import annotations

import contextlib
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
    _log_release_lift = IN.InsideMode._log_release_lift
    release_lift_per_grasp = IN.InsideMode.release_lift_per_grasp
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


def test_goal_region_key():
    assert IN.goal_region(None) is None
    assert IN.goal_region({"x": [5.27, 5.8], "y": (-0.8, -0.2)}) == {"frame": "world", "x": [5.27, 5.8], "y": [-0.8, -0.2]}
    assert IN.goal_region({"frame": " Partner ", "y": [0, 1]}) == {"frame": "partner", "x": None, "y": [0.0, 1.0]}  # x unbounded
    assert IN.goal_region({"x": [0.1, 0.1], "y": None})["x"] == [0.1, 0.1]
    for bad in ([5.27, 5.8], "left", {}, {"frame": "world"}, {"x": None, "y": None}, {"x": [1.0, 0.0]}, {"x": [0.0]}, {"x": [0, 1, 2]},
                {"x": "01"}, {"x": ["a", 1]}, {"x": 5.3}, {"frame": "door", "x": [0, 1]}, {"x": [0, 1], "z": [0, 1]}, {"x": [float("nan"), 1.0]}):
        with pytest.raises(ValueError):
            IN.goal_region(bad)


def test_in_goal_region_world_and_partner_frame():
    cells = th.tensor([[5.30, -0.50], [5.60, -0.30], [5.10, -0.50], [5.70, -0.50], [5.40, -0.10]])
    world = IN.goal_region({"x": [5.27, 5.66], "y": [-0.75, -0.25]})
    assert IN.in_goal_region(cells, world).tolist() == [True, True, False, False, False]
    assert IN.in_goal_region(cells, IN.goal_region({"x": [5.27, 5.66]})).tolist() == [True, True, False, False, True]  # y unbounded
    # the partner frame: origin (5.0, -0.5), yaw +90 deg (its x axis = world +y, its y axis = world -x)
    partner = IN.goal_region({"frame": "partner", "x": [-0.1, 0.1], "y": [-0.45, -0.25]})
    assert IN.in_goal_region(cells, partner, origin=(5.0, -0.5), yaw=math.pi / 2).tolist() == [True, False, False, False, False]
    assert IN.in_goal_region(th.zeros(0, 2), world).tolist() == []


def _two_compartments(res: float = 0.02) -> SupportInterior:
    """inside_large#02's 0.97 m shelf (the world frame): the right compartment x -0.56..-0.06 and the left one x 0.00..0.34 (cell centres;
    the divider's 2 cells between them are no board cells), y -0.40..0.00, the opening facing +y."""
    xs, ys = (th.arange(46) - 28) * res, (th.arange(21) - 20) * res
    gx, _ = th.meshgrid(xs, ys, indexing="ij")
    mask = (gx <= -0.06 + 1e-6) | (gx >= -1e-6)
    return SupportInterior("fridge_petcxr_0", xs, ys, mask, th.zeros_like(mask), 0.97, 0.0, [], res, headroom=th.full(mask.shape, 0.30), z_requested=0.97,
                           board_pieces=["/World/fridge/base_link/collisions/left", "/World/fridge/base_link/collisions/right"])


def _fridge_stub(**gen):
    s = _Stub(interior=_two_compartments(), half=0.045, height=0.13, inside_surface_z_m=0.97, inside_stance_side="+y", **gen)
    s.partner = SimpleNamespace(name="fridge_petcxr_0", aabb=(th.tensor([-0.60, -0.45, 0.0]), th.tensor([0.40, 0.25, 1.80])),
                                get_position_orientation=lambda: (th.tensor([-0.10, 0.25, 0.0]), th.tensor([0.0, 0.0, 0.7071068, 0.7071068])))
    return s


def test_prepare_partner_goal_region_keeps_one_compartment(caplog):
    """``generation.inside_goal_region`` (owner 2026-10-13 ~03:15 item 2): the shelf spans both compartments; the box keeps the goal cells
    of the left one (after the erosion, which keeps them off the divider already); the stance side stays the whole raster's."""
    s0 = _fridge_stub()
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        s0.prepare_partner()
    assert "inside goal region" not in caplog.text and "goal_region" not in s0.template_facts["inside_surface"]  # default: every goal cell
    cells0 = s0.place["cells"]
    left0 = cells0[cells0[:, 0] > -0.03]
    # eroded by 0.045 + 0.02 from each compartment's outline: left x 0.08..0.26 (10) x y -0.32..-0.08 (13), right x -0.48..-0.14 (18) x 13
    assert len(cells0) == 130 + 234 and len(left0) == 130
    caplog.clear()
    s = _fridge_stub(inside_goal_region={"frame": "world", "x": [-0.03, 0.40], "y": [-0.45, 0.05]})
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        s.prepare_partner()
    cells = s.place["cells"]
    assert len(cells) == 130 and th.equal(cells, left0)
    assert float(cells[:, 0].min()) == pytest.approx(0.08) and float(cells[:, 0].max()) == pytest.approx(0.26)
    (line,) = [r.getMessage() for r in caplog.records if "inside goal region" in r.getMessage()]
    assert "fridge_petcxr_0 (world frame, x [-0.03, 0.4], y [-0.45, 0.05]): 364 -> 130 goal cells" in line
    assert "-> x 0.080..0.260 y -0.320..-0.080)" in line
    facts = s.template_facts["inside_surface"]
    assert facts["goal_region"] == {"frame": "world", "x": [-0.03, 0.40], "y": [-0.45, 0.05], "n_before": 364, "n_after": 130}
    assert facts["n_goal_cells"] == 130 and s.template_facts["partner"]["n_goal_cells"] == 130
    assert [e.name for e in s.place["edges"]] == ["+v"] and s.place["edges"][0].length == pytest.approx(0.90)  # the whole raster's side
    assert float(s.place["edges"][0].mid[0]) == pytest.approx(float(s0.place["edges"][0].mid[0]))
    # the partner frame (origin (-0.10, 0.25), yaw +90 deg: its x = world y - 0.25, its y = -0.10 - world x): the same compartment
    sp = _fridge_stub(inside_goal_region={"frame": "partner", "y": [-0.45, 0.0]})
    caplog.clear()
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        sp.prepare_partner()
    assert th.equal(sp.place["cells"], left0) and "(partner frame, x any, y [-0.45, 0.0]): 364 -> 130 goal cells" in caplog.text
    assert sp.template_facts["inside_surface"]["goal_region"]["partner_yaw"] == pytest.approx(math.pi / 2, abs=1e-4)


def test_prepare_partner_goal_region_rejects():
    with pytest.raises(Reject) as e:
        _fridge_stub(inside_goal_region={"x": [0.27, 0.40]}).prepare_partner()  # the left compartment's wall side: no goal cell after the erosion
    assert e.value.reason == "spec" and "keeps none of the 364 goal cells of fridge_petcxr_0 (world x -0.480..0.260 y -0.320..-0.080)" in str(e.value)
    with pytest.raises(Reject) as e:
        _fridge_stub(inside_goal_region={"x": [0.4, 0.3]}).prepare_partner()
    assert e.value.reason == "spec" and "lo <= hi" in str(e.value)
    with pytest.raises(Reject) as e:  # no goal cell before the box either: the erosion's reject, not the box's
        _Stub(half=0.40, inside_surface_z_m=0.80, inside_goal_region={"x": [-1.0, 1.0]}).prepare_partner()
    assert e.value.reason == "inside_no_goal_cell"


def _top_down_grasp(eef_above_origin: float, name: str = "g"):
    """A top-down pinch: the eef ``eef_above_origin`` above the object origin, approach straight down."""
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    return SimpleNamespace(name=name, pose=(th.tensor([0.0, 0.0, eef_above_origin]), q))


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


def test_release_lift_per_grasp_logged_once_per_template(caplog):
    """Owner 2026-10-13 ~00:50 item 2: one line per template with the lift each library grasp needs and the one used (their max); the
    release heights are those of the unlogged computation."""
    quat, turned = th.tensor([0.0, 0.0, 0.0, 1.0]), IN.G.quat_from_yaw(0.7)
    high, low = _top_down_grasp(0.10), SimpleNamespace(name="g_low", pose=_top_down_grasp(0.0).pose)
    s = _Stub(inside_surface_z_m=0.80, grasps=[high, low])
    s.prepare_partner()
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        above, above_turned = s.release_above(quat), s.release_above(turned)  # a second orientation: computed, not logged again
    lines = [r.getMessage() for r in caplog.records if "inside release lift per grasp" in r.getMessage()]
    z_s, clearance = 0.70, IN.RELEASE_CLEARANCE_M
    pos = th.tensor([0.0, 0.0, z_s + 0.0108 + IN.DROP_HEIGHT_M])
    lows = [float(gripper_corners_world(IN.IK.grasp_pose((pos, quat), g.pose))[:, 2].min()) for g in s.grasps]
    assert above == pytest.approx(IN.DROP_HEIGHT_M + IN.release_lift(lows, z_s, clearance)) and above_turned == pytest.approx(above, abs=1e-6)
    lift = above - IN.DROP_HEIGHT_M
    assert lift > 0.0 and len(lines) == 1
    assert f"inside release lift per grasp: {{g: 0.000 m, g_low: {lift:.3f} m}}, used (max) {lift:.3f} m" in lines[0]


#: plate__luhkiz__ts grasp_5, the plate's one library row: a side pinch at the rim from outside (approach inward, 28 deg below horizontal)
GRASP_5 = SimpleNamespace(name="grasp_5", pose=(th.tensor([-0.091, 0.067, 0.014]), th.tensor([-0.33107, 0.78989, -0.36502, 0.36502])))
PLATE_RADIUS = 0.1205


def _plate_hull(n: int = 24) -> th.Tensor:
    """A 24 cm disc 2.6 cm thick, its origin at the bottom centre."""
    a = th.arange(n) * (2.0 * math.pi / n)
    ring = th.stack([PLATE_RADIUS * th.cos(a), PLATE_RADIUS * th.sin(a)], dim=1)
    return th.cat([th.cat([ring, th.zeros(n, 1)], dim=1), th.cat([ring, th.full((n, 1), 0.026)], dim=1)])


def _far_rim() -> th.Tensor:
    """The rim point opposite grasp_5 (object frame)."""
    p = GRASP_5.pose[0]
    return th.tensor([float(-p[0]), float(-p[1]), 0.0]) * (PLATE_RADIUS / math.hypot(float(p[0]), float(p[1])))


def test_release_tilts_key():
    assert IN.release_tilts(None) == [] and IN.release_tilts([]) == [] and IN.release_tilts(30) == [30.0]
    assert IN.release_tilts([30, 45, 30, 0]) == [30.0, 45.0]  # the level release is always screened: 0 and repeats dropped
    for bad in (-5, 90, [30, "steep"], [120]):
        with pytest.raises(ValueError):
            IN.release_tilts(bad)


@pytest.mark.parametrize("yaw", [0.0, 0.7, 2.5, -2.0])
def test_tilt_raises_the_grasp_side(yaw):
    """yaw x tilt x rest: about the horizontal axis perpendicular to the grasp's radial direction, the grasp side up and the far rim down
    at every yaw (owner 2026-10-12 ~22:35: at 45 deg grasp_5's rim point ~16 cm above the far rim, its approach ~70 deg down)."""
    rest = th.tensor([0.0, 0.0, 0.0, 1.0])
    level = IN.G.qmul(IN.G.quat_from_yaw(yaw), rest)
    p, far = GRASP_5.pose[0], _far_rim()
    assert IN.tilted_quat(level, p, 0.0) is level
    axis = IN.tilt_axis(p, level)
    radial = IN.G.rotate(level, p)
    assert float(axis[2]) == 0.0 and float(th.dot(axis, radial)) == pytest.approx(0.0, abs=1e-6)
    lift = {}
    for deg in (30.0, 45.0):
        q = IN.tilted_quat(level, p, deg)
        z_g, z_f = float(IN.G.rotate(q, p)[2]), float(IN.G.rotate(q, far)[2])
        lift[deg] = z_g - z_f
        assert z_g > float(p[2]) and z_f < 0.0
        # the same orientation composed in the object's terms: yaw * tilt * rest, the tilt about the rested frame's axis
        q_obj = IN.G.qmul(IN.G.quat_from_yaw(yaw), IN.G.qmul(IN.G.quat_about_axis(IN.tilt_axis(p, rest), math.radians(deg)), rest))
        assert C.quat_angle(q, q_obj) < 1e-4
        assert C.quat_angle(q, level) == pytest.approx(math.radians(deg), abs=1e-4)
    assert lift[30.0] == pytest.approx(0.129, abs=0.003) and lift[45.0] == pytest.approx(0.175, abs=0.003)
    a = IN.G.rotate(IN.G.qmul(IN.tilted_quat(level, p, 45.0), GRASP_5.pose[1]), th.tensor([0.0, 0.0, 1.0]))
    assert math.degrees(math.asin(-float(a[2]))) == pytest.approx(69.5, abs=1.0)
    assert IN.tilt_axis(th.tensor([0.0, 0.0, 0.1]), level) is None and IN.tilted_quat(level, th.tensor([0.0, 0.0, 0.1]), 45.0) is level


class _TiltStub(_Stub):
    """:class:`_Stub` with the plate (a disc hull, the true bottom offset, grasp_5) and the release-tilt hooks; ``ikc.feasible_sets``
    records its calls and admits the pairs ``admit(tilt deg, yaw index)`` allows."""

    release_tilts_deg = IN.InsideMode.release_tilts_deg
    release_quat = IN.InsideMode.release_quat
    screen_release_tilts = IN.InsideMode.screen_release_tilts

    def __init__(self, admit=lambda deg, i: False, **gen):
        super().__init__(grasps=[GRASP_5], **gen)
        self.hull, self.rest = _plate_hull(), th.tensor([0.0, 0.0, 0.0, 1.0])
        self.world.bottom_offset = lambda hull, quat: -float(IN.G.local_points_world(hull, (th.zeros(3), IN.G.t(quat)))[:, 2].min())
        self.admit, self.screened = admit, []
        self.ikc = SimpleNamespace(feasible_sets=self._feasible_sets)
        self.prepare_partner()

    def _feasible_sets(self, cells, yaws, grasps, pose_at, place, arm, ignore):
        deg = round(math.degrees(C.quat_angle(pose_at(cells[0], 0.0)[1], th.tensor([0.0, 0.0, 0.0, 1.0]))))
        self.screened.append((deg, list(yaws), [g.name for g in grasps], pose_at))
        return {0: [(y, grasps[0]) for i, y in enumerate(yaws) if self.admit(deg, i)]}


PLACE_C = SimpleNamespace(stand_off=0.45, lateral=0.1, translate=0.5, tilt=-0.3)


def test_release_tilt_default_is_the_level_release():
    """Without ``release_tilt_deg``: the level F_goal as it came, no extra screening, the level orientation (IORSLarge's hooks: the same)."""
    s = _TiltStub(inside_surface_z_m=0.80, drop_height_m=0.02)
    F_goal = [(0.0, GRASP_5)]
    out, tilts_of = s.screen_release_tilts(F_goal, [0.05, -0.02], [0.0], 0.2, PLACE_C, "right", [], math.pi / 8)
    assert out is F_goal and tilts_of == {} and s.screened == []
    q = IN.G.quat_from_yaw(0.3)
    assert IL.IORSLarge.release_quat(s, q, GRASP_5, 45.0) is q and IL.IORSLarge.screen_release_tilts(s, F_goal, None, [], 0.0, PLACE_C, "right", [], 0.4) == (F_goal, {})
    with pytest.raises(Reject) as e:
        _TiltStub(inside_surface_z_m=0.80, release_tilt_deg=[30, 95]).screen_release_tilts(F_goal, [0.05, -0.02], [0.0], 0.2, PLACE_C, "right", [], math.pi / 8)
    assert e.value.reason == "spec"


def test_release_tilt_screen_counts_and_tilted_release_pose(caplog):
    """``release_tilt_deg`` [30, 45]: each tilt screened per library row at the level screen's yaws (+ the carry's yaw offset), the pairs
    only a tilt admits after the level ones, per pair the tilts it passed in order, one log line with F_goal per tilt; the release pose
    puts the tilted hull's lowest point ``drop_height_m`` above the basin floor, the grasp side ~17 cm above the far rim."""
    yaws, d_yaw, step = [0.0, math.pi / 8, math.pi / 4], 0.2, math.pi / 8
    s = _TiltStub(admit=lambda deg, i: (deg == 30 and i == 0) or deg == 45, inside_surface_z_m=0.80, drop_height_m=0.02, release_tilt_deg=[30, 45])
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        out, tilts_of = s.screen_release_tilts([], [0.05, -0.02], yaws, d_yaw, PLACE_C, "right", [], step)
    assert [(deg, names) for deg, _, names, _ in s.screened] == [(30, ["grasp_5"]), (45, ["grasp_5"])]
    assert all(ys == pytest.approx([y + d_yaw for y in yaws]) for _, ys, _, _ in s.screened)
    assert [(y, g.name) for y, g in out] == [(pytest.approx(y), "grasp_5") for y in yaws]  # the init yaws, back from the C yaws
    assert tilts_of == {(0, "grasp_5"): [30.0, 45.0], (1, "grasp_5"): [45.0], (2, "grasp_5"): [45.0]}
    assert "inside release tilt F_goal at C" in caplog.text and "0 deg: F_goal 0, 30 deg: F_goal 1, 45 deg: F_goal 3" in caplog.text
    # a level pair keeps the level release first
    s2 = _TiltStub(admit=lambda deg, i: True, inside_surface_z_m=0.80, drop_height_m=0.02, release_tilt_deg=[30, 45])
    out2, tilts2 = s2.screen_release_tilts([(0.0, GRASP_5)], [0.05, -0.02], [0.0], 0.0, PLACE_C, "right", [], step)
    assert len(out2) == 1 and tilts2 == {(0, "grasp_5"): [0.0, 30.0, 45.0]}
    # the pose the 45 deg screen posed: the tilted hull's lowest point drop_height_m above the surface, the grasp side up
    pose = s.screened[1][3]([0.05, -0.02], 0.4)
    z_s = float(s.place["z"])
    assert float(IN.G.local_points_world(s.hull, pose)[:, 2].min()) == pytest.approx(z_s + 0.02, abs=1e-5)
    assert s.release_above(pose[1]) == pytest.approx(0.02)  # the open gripper far above the floor: no lift
    level = IN.G.quat_from_yaw(0.4)
    assert s.release_above(level) > 0.02  # level, grasp_5's fingers reach below the plate: the release rises
    g_w = pose[0] + IN.G.rotate(pose[1], GRASP_5.pose[0])
    f_w = pose[0] + IN.G.rotate(pose[1], _far_rim())
    assert float(g_w[2] - f_w[2]) == pytest.approx(0.175, abs=0.003) and float(f_w[2]) == pytest.approx(z_s + 0.02, abs=0.005)


def test_goal_upright_measures_a_tilted_release_against_level():
    """A plate released at 45 deg that lands flat: the stock up-axis check against the goal pose (the tilted release) reads 45 deg and
    rejects (goal_tipped); with ``summary.release_tilt_deg`` it is measured against the level orientation and the release is kept."""
    from ispatialgym.sampling.tasks.base import Draft

    cls = type("IORSLarge_InsideMode", (IN.InsideMode, IL.IORSLarge), {})
    s = cls.__new__(cls)
    level = IN.G.quat_from_yaw(0.4)
    plate = SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), level))
    s.bindings, s.objects = SimpleNamespace(one=lambda role: SimpleNamespace(name="plate")), {"plate": plate}
    s.spec, s.counters = SimpleNamespace(generation={}), {}
    release = (th.tensor([5.69, -1.81, 0.90]), IN.tilted_quat(level, GRASP_5.pose[0], 45.0))
    d = Draft()
    d.goal_poses["plate"] = release
    with pytest.raises(Reject) as e:
        s.check_goal_upright(d)
    assert e.value.reason == "goal_tipped" and d.summary["goal_tilt_deg"]["plate"] == pytest.approx(45.0, abs=0.1)
    d.summary.update({"release_tilt_deg": 45.0, "release_level_quat": [float(v) for v in level]})
    s.check_goal_upright(d)
    assert d.summary["goal_tilt_deg"]["plate"] == pytest.approx(0.0, abs=0.1) and d.goal_poses["plate"] is release
    d.summary["release_tilt_deg"] = 0.0  # a level release of a tilt-screened draft: the stock check
    with pytest.raises(Reject):
        s.check_goal_upright(d)


def test_oracle_drop_reaches_a_tilted_release_from_the_level_carry():
    """The oracle's drop needs no change for a tilted release (eef = tilted object pose x grasp): the carry leg runs from the hand at the
    level carry to the recorded pre-place with the orientation SLERPed, the descent and the retreat follow the tilted approach axis."""
    from ispatialgym.oracle.actions import Place
    from ispatialgym.oracle.sidecar import Pose
    from ispatialgym.sampling.v2 import ik as IK

    level = IN.G.quat_from_yaw(0.4)
    release = (th.tensor([5.69, -1.81, 0.90]), IN.tilted_quat(level, GRASP_5.pose[0], 45.0))
    place_eef = IK.grasp_pose(release, GRASP_5.pose)
    pre = IK.pre_pose(place_eef)
    carry = IK.grasp_pose((th.tensor([5.40, -1.60, 1.05]), level), GRASP_5.pose)
    as_pose = lambda p: Pose([float(v) for v in p[0]], [float(v) for v in p[1]])  # noqa: E731
    action = Place(kind="drop", arm="right", object="plate", place=as_pose(place_eef), pre_place=as_pose(pre), on="sink", relation="into")
    t = action.targets(SimpleNamespace(grasps=[]), SimpleNamespace(eef_pose=lambda arm: carry))
    turn = [C.quat_angle(carry[1], th.tensor(w.orientation)) for w in t["carry"]]
    assert all(b >= a - 1e-5 for a, b in zip(turn, turn[1:])) and turn[-1] == pytest.approx(C.quat_angle(carry[1], pre[1]), abs=1e-4)
    assert t["carry"][-1].position == pytest.approx([float(v) for v in pre[0]], abs=1e-5) and turn[-1] > math.radians(40)
    a = C.approach_axis(place_eef[1])
    assert math.degrees(math.asin(-float(a[2]))) > 65.0  # the release approaches steeply, its back-off goes up
    for w in t["descent"] + t["retreat"]:
        off = th.tensor(w.position) - place_eef[0]
        assert float(th.linalg.norm(th.linalg.cross(off, a))) < 1e-4 and C.quat_angle(th.tensor(w.orientation), place_eef[1]) < 1e-4
    assert t["retreat"][-1].position[2] > float(place_eef[0][2]) + 0.09


# ----------------------------------------------------------------------------------------------------------------
# the per-grasp release height (owner 2026-10-13 ~00:50 item 3, inside_large#02: the jar into the fridge's lower shelf)
# ----------------------------------------------------------------------------------------------------------------

LEVEL = th.tensor([0.0, 0.0, 0.0, 1.0])
#: top-down pinches whose open fingertips, at the plain drop height, stay clear (no lift), end 0.7 cm above the surface and 0.3 cm below it
#: (lifts 1.3 / 2.3 cm for the 2 cm clearance)
LIFT_GRASPS = [_top_down_grasp(0.10, "grasp_0"), _top_down_grasp(0.01, "grasp_1"), _top_down_grasp(0.0, "grasp_2")]
#: grasp_5 turned half way round the plate's axis: the same rim pinch from the opposite side (its rim goes down when grasp_5's goes up)
GRASP_5_OPPOSITE = SimpleNamespace(name="grasp_5o", pose=(IN.G.rotate(IN.G.quat_from_yaw(math.pi), GRASP_5.pose[0]),
                                                          IN.G.qmul(IN.G.quat_from_yaw(math.pi), GRASP_5.pose[1])))


class _GoalCheckMixin:
    """IORSLarge's F_goal screen / goal / settle and :meth:`InsideMode.after_goal_check` with what they read but the sidecar draft and the
    IK checker (:func:`_goal_check_release`)."""

    release_feasible = IL.IORSLarge.release_feasible
    goal = IL.IORSLarge.goal
    after_settle = IL.IORSLarge.after_settle
    goal_quat_of = IL.IORSLarge.goal_quat_of
    release_eef_in_object = IL.IORSLarge.release_eef_in_object
    after_goal_check = IN.InsideMode.after_goal_check
    tname, kit, target = "jar", None, SimpleNamespace(name="jar")

    def tol(self, key, default):
        return default

    def record_partner_joints(self, draft):
        pass

    def obj(self, role):
        return SimpleNamespace(name=role)

    def camera_pose_at(self, stance):
        return None


class _LiftStub(_GoalCheckMixin, _Stub):
    pass


class _TiltLiftStub(_GoalCheckMixin, _TiltStub):
    pass


def _goal_check_release(s, drawn, quat, monkeypatch, gc=(0.05, -0.02)):
    """``IORSLarge._couple``'s record of the drawn grasp - the C exact IK's release ``goal_pose_at(gc, quat, drawn.pose)``, the place eef
    from it, the grasp record with ``eef_in_object`` - then IORSLarge's goal / settle and :meth:`InsideMode.after_goal_check` on the draft
    (the exact IK and the frame tests pass). Returns (the C release, the draft, the grasp record, the release goal() posed)."""
    from ispatialgym.sampling.tasks.base import Draft
    from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft

    monkeypatch.setattr(IN, "attached", lambda *a, **kw: contextlib.nullcontext())
    s.ikc = SimpleNamespace(exact=lambda *a, **kw: SimpleNamespace(ok=True), eef_in_frame=lambda eef, cam: True)
    release = s.goal_pose_at(gc, quat, drawn.pose)
    place_eef = IN.IK.grasp_pose(release, drawn.pose)
    s.sidecar = SidecarDraft(s.gp.alias, "iors")
    for name in ("manipulation", "placement"):
        s.sidecar.add_stance(Stance(name, (5.4, -0.9, 1.57), 0.5, -0.3, 0.55, None, None, 0.0, None, ["right"]))
    rec = s.sidecar.add_grasp(GraspRecord("jar", "right", "jar__k", drawn.name, 0.0, None, None, IN.IK.pre_pose(place_eef), place_eef, eef_in_object=drawn.pose))
    d = Draft()
    d.summary.update({"goal_cell": list(gc), "goal_quat": [float(v) for v in quat]})
    s.goal(d)
    posed = d.goal_poses["jar"]
    s.after_settle(d)
    s.after_goal_check(d)
    return release, d, rec, posed


def test_release_lift_per_grasp_key():
    assert [IN.lift_per_grasp(v) for v in (None, False, "off", "false", 0)] == [False] * 5
    assert [IN.lift_per_grasp(v) for v in (True, "on", "true", 1)] == [True] * 4
    for bad in ("max", 2, [True]):
        with pytest.raises(ValueError):
            IN.lift_per_grasp(bad)
    assert _Stub(release_lift_per_grasp=True).release_lift_per_grasp() is True and _Stub().release_lift_per_grasp() is False
    with pytest.raises(Reject) as e:
        _Stub(release_lift_per_grasp="max").release_lift_per_grasp()
    assert e.value.reason == "spec"
    assert IL.IORSLarge.release_lift_per_grasp(SimpleNamespace()) is False  # ontop / under / push-under: one release pose per cell


def test_release_lift_per_grasp_log_line_lists_the_lifts_used(caplog):
    """With the key the once-per-template line (owner item 2) lists every library row's lift - each grasp's release is ``drop_height_m`` +
    its listed lift - and names the per-grasp use with the default's max beside it, also when the first computation is a per-grasp one."""
    s = _Stub(inside_surface_z_m=0.80, grasps=LIFT_GRASPS, release_lift_per_grasp=True)
    s.prepare_partner()
    with caplog.at_level("INFO", logger="ispatialgym.sampling.v2"):
        own = [s.release_above(LEVEL, g.pose) for g in LIFT_GRASPS]
    (line,) = [r.getMessage() for r in caplog.records if "inside release lift per grasp" in r.getMessage()]
    assert "{grasp_0: 0.000 m, grasp_1: 0.013 m, grasp_2: 0.023 m}, used per grasp (release_lift_per_grasp: the drawn grasp's own; max 0.023 m)" in line
    assert [round(h - IN.DROP_HEIGHT_M, 3) for h in own] == [0.0, 0.013, 0.023] and "used (max)" not in line


def test_release_lift_per_grasp_default_is_one_height_for_every_grasp():
    """Without ``release_lift_per_grasp`` (or false): every grasp's release at the one height the highest lift over the library sets - a
    grasp passed to :meth:`release_above` / :meth:`goal_pose_at` changes nothing (today's release)."""
    for gen in ({}, {"release_lift_per_grasp": False}):
        s = _Stub(inside_surface_z_m=0.80, grasps=LIFT_GRASPS, **gen)
        s.prepare_partner()
        top = s.release_above(LEVEL)
        assert top == pytest.approx(IN.DROP_HEIGHT_M + 0.023, abs=1e-5)  # grasp_2's fingertips 1.88 cm below its eef at the origin
        assert [s.release_above(LEVEL, g.pose) for g in LIFT_GRASPS] == [top] * 3
        pose = s.goal_pose_at([0.05, -0.02], LEVEL)
        assert all(bool(th.equal(s.goal_pose_at([0.05, -0.02], LEVEL, g.pose)[0], pose[0])) for g in LIFT_GRASPS)


def test_release_lift_per_grasp_lifts_the_release_for_the_drawn_grasp_alone():
    """``release_lift_per_grasp`` true: the release rises only as far as the drawn grasp's own open gripper needs - never above the
    default's height (the highest lift), the plain drop height for a grasp whose fingers end above the target's bottom, every grasp's open
    gripper still ``release_clearance_m`` above the surface (exactly, when it was lifted); without a grasp (stance C's frame test, before a
    grasp is drawn) the default's height."""
    s = _Stub(inside_surface_z_m=0.80, grasps=LIFT_GRASPS, release_lift_per_grasp=True)
    s.prepare_partner()
    top = s.release_above(LEVEL)
    own = [s.release_above(LEVEL, g.pose) for g in LIFT_GRASPS]
    assert top == pytest.approx(IN.DROP_HEIGHT_M + 0.023, abs=1e-5)
    assert own == [pytest.approx(IN.DROP_HEIGHT_M), pytest.approx(IN.DROP_HEIGHT_M + 0.013, abs=1e-5), pytest.approx(top)]
    assert all(h <= top + 1e-9 for h in own)
    for g, h in zip(LIFT_GRASPS, own):
        pose = s.goal_pose_at([0.05, -0.02], LEVEL, g.pose)
        assert float(pose[0][2]) == pytest.approx(0.70 + 0.0108 + h)
        low = float(gripper_corners_world(IN.IK.grasp_pose(pose, g.pose))[:, 2].min())
        assert low >= 0.70 + IN.RELEASE_CLEARANCE_M - 1e-5
        if h > IN.DROP_HEIGHT_M + 1e-6:
            assert low == pytest.approx(0.70 + IN.RELEASE_CLEARANCE_M, abs=1e-5)


def test_release_lift_per_grasp_at_the_tilted_release():
    """The drawn grasp's own lift at its release orientation: at grasp_5's 45 deg tilt the opposite rim pinch's open fingers hang 3.7 cm
    below the basin floor, so by default (one height for every row at that orientation) grasp_5's tilted release rises 5.65 cm; with the key
    the tilt screen poses each row at its own tilt and lift (none: the low rim ``drop_height_m`` above the floor)."""
    for per, lift in ((False, 0.0565), (True, 0.0)):
        s = _TiltStub(admit=lambda deg, i: True, inside_surface_z_m=0.80, drop_height_m=0.02, release_tilt_deg=[45], release_lift_per_grasp=per)
        s.grasps = [GRASP_5, GRASP_5_OPPOSITE]
        s.screen_release_tilts([], [0.05, -0.02], [0.0], 0.0, PLACE_C, "right", [], math.pi / 8)
        assert [(deg, names) for deg, _, names, _ in s.screened] == [(45, ["grasp_5"]), (45, ["grasp_5o"])]
        z_s = float(s.place["z"])
        for (_, _, _, pose_at), g in zip(s.screened, s.grasps):
            pose = pose_at([0.05, -0.02], 0.4)
            assert float(IN.G.local_points_world(s.hull, pose)[:, 2].min()) == pytest.approx(z_s + 0.02 + lift, abs=1e-3)
            assert float(gripper_corners_world(IN.IK.grasp_pose(pose, g.pose))[:, 2].min()) >= z_s + IN.RELEASE_CLEARANCE_M - 1e-5


def test_release_lift_per_grasp_f_goal_screen():
    """IORSLarge.release_feasible, the level F_goal screen of a C candidate: by default ``pose_at(cell, yaw)`` with the call's arguments as
    before; with the key one batch with ``pose_per_grasp`` and ``pose_at(cell, yaw, row)`` at the row's own release height."""
    calls = []

    def feasible_sets(cells, yaws, grasps, pose_at, stance, arm, ignore, **kw):
        calls.append((pose_at, kw))
        return {0: [(yaws[0], grasps[0])]}

    for per in (False, True):
        calls.clear()
        s = _LiftStub(inside_surface_z_m=0.80, grasps=LIFT_GRASPS, release_lift_per_grasp=per)
        s.prepare_partner()
        s.rest, s.ikc = LEVEL, SimpleNamespace(feasible_sets=feasible_sets)
        out = s.release_feasible([0.05, -0.02], [0.3], lambda xy, quat, eio=None: s.goal_pose_at(xy, quat, eio), PLACE_C, "right", [])
        (pose_at, kw), = calls
        assert out == [(0.3, LIFT_GRASPS[0])]
        if per:
            heights = [float(pose_at([0.05, -0.02], 0.3, g)[0][2]) - 0.70 - 0.0108 for g in LIFT_GRASPS]
            assert kw == {"pose_per_grasp": True} and heights == [pytest.approx(s.release_above(LEVEL, g.pose), abs=1e-6) for g in LIFT_GRASPS]
            assert max(heights) == pytest.approx(s.release_above(LEVEL), abs=1e-6) and heights[0] < heights[1] < heights[2]
        else:
            assert kw == {} and float(pose_at([0.05, -0.02], 0.3)[0][2]) == pytest.approx(0.70 + 0.0108 + s.release_above(LEVEL), abs=1e-6)


def test_feasible_sets_pose_per_grasp(monkeypatch):
    """``IKChecker.feasible_sets(..., pose_per_grasp=True)``: every pair's eef from ``pose_at(cell, yaw, grasp)`` in one screening batch, the
    cache key over the yaw-0 poses per grasp plus the flag; without it ``pose_at(cell, yaw)`` once per (cell, yaw) and the key as before."""
    IKM = IN.IK
    payloads, batches = [], []
    monkeypatch.setattr(IKM, "cache_key", lambda kind, payload: payloads.append(payload) or "feasible_sets_v2-test")
    monkeypatch.setattr(IKM, "cache_get", lambda key: None)
    monkeypatch.setattr(IKM, "cache_put", lambda key, value: None)
    ikc = IKM.IKChecker.__new__(IKM.IKChecker)
    ikc.cache_scope = {}
    ikc.screen_pairs = lambda pairs, stance, arm, ignore_for_last=(): (batches.append(pairs), th.tensor([i % 2 == 0 for i in range(len(pairs))]))[1]
    place = Stance("placement", (5.4, -0.9, 1.57), 0.5, -0.3, 0.55, None, None, 0.0, None, ["right"])
    grasps, cell, yaws = LIFT_GRASPS[:2], [5.39, -0.37], [0.0, 0.4]
    raised = {"grasp_0": 0.0, "grasp_1": 0.05}
    per_calls = []

    def pose_at(c, yaw, g):
        per_calls.append((yaw, g.name))
        return th.tensor([c[0], c[1], 0.9 + raised[g.name]]), IN.G.quat_from_yaw(yaw)

    out = ikc.feasible_sets([cell], yaws, grasps, pose_at, place, "right", [], pose_per_grasp=True)
    assert {i: [(y, g.name) for y, g in lst] for i, lst in out.items()} == {0: [(0.0, "grasp_0"), (0.4, "grasp_0")]}
    (pairs,) = batches
    assert [(y, n) for y, n in per_calls[-4:]] == [(0.0, "grasp_0"), (0.0, "grasp_1"), (0.4, "grasp_0"), (0.4, "grasp_1")]
    by_name = {g.name: g for g in grasps}
    for (pre, eef), (y, n) in zip(pairs, per_calls[-4:]):
        want = IKM.grasp_pose((th.tensor([cell[0], cell[1], 0.9 + raised[n]]), IN.G.quat_from_yaw(y)), by_name[n].pose)
        assert th.allclose(eef[0], want[0]) and th.allclose(eef[1], want[1]) and th.allclose(pre[0], IKM.pre_pose(want)[0])
    assert payloads[-1]["pose_per_grasp"] is True and len(payloads[-1]["poses0"]) == len(grasps)
    batches.clear()
    out = ikc.feasible_sets([cell], yaws, grasps, lambda c, yaw: pose_at(c, yaw, grasps[0]), place, "right", [])
    assert {i: [(y, g.name) for y, g in lst] for i, lst in out.items()} == {0: [(0.0, "grasp_0"), (0.4, "grasp_0")]}
    assert len(batches) == 1 and len(batches[0]) == 4 and "pose_per_grasp" not in payloads[-1]
    assert sorted(payloads[-1]) == ["arm", "base", "cells", "grasps", "ignore", "poses0", "scale", "seeds", "trunk", "yaws"]
    assert payloads[-1]["poses0"] == [pytest.approx([cell[0], cell[1], 0.9] + [float(v) for v in IN.G.quat_from_yaw(0.0)], abs=1e-6)]


@pytest.mark.parametrize("per", [False, True])
def test_release_lift_per_grasp_record_and_drop_test_release(monkeypatch, per):
    """The drawn grasp's release through the goal check: IORSLarge's goal / settle re-derive the release the C exact IK used (from the
    recorded grasp's ``eef_in_object``) - the pose the drop test teleports the target to -, and ``summary.release`` / ``grasps[0]`` carry
    its height: grasp_1's own lift (1.3 cm, below grasp_2's 2.3 cm) with ``release_above_max_m`` beside it; without the key today's record
    (the highest lift, no new field)."""
    s = _LiftStub(inside_surface_z_m=0.80, grasps=LIFT_GRASPS, release_lift_per_grasp=per)
    s.prepare_partner()
    release, d, rec, posed = _goal_check_release(s, LIFT_GRASPS[1], LEVEL, monkeypatch)
    top = IN.DROP_HEIGHT_M + 0.023
    want = IN.DROP_HEIGHT_M + 0.013 if per else top
    assert float(release[0][2]) == pytest.approx(0.70 + 0.0108 + want, abs=1e-5)
    assert bool(th.equal(posed[0], release[0])) and bool(th.equal(d.goal_poses["jar"][0], release[0]))  # goal(), then the settle
    assert d.pose_targets["jar"]["position"] == pytest.approx([float(v) for v in release[0]])
    facts = d.summary["release"]
    assert facts["release_above_m"] == pytest.approx(want, abs=1e-4) and rec.extras["release_above_m"] == facts["release_above_m"]
    if per:
        assert facts["release_lift_per_grasp"] is True and facts["release_above_max_m"] == pytest.approx(top, abs=1e-4)
        assert facts["finger_clearance_m"] == pytest.approx(IN.RELEASE_CLEARANCE_M, abs=1e-4)  # lifted exactly to the clearance
    else:
        assert "release_lift_per_grasp" not in facts and "release_above_max_m" not in facts
        assert facts["finger_clearance_m"] == pytest.approx(IN.RELEASE_CLEARANCE_M + 0.010, abs=1e-4)  # 1 cm higher than grasp_1 needs


def test_release_lift_per_grasp_record_at_a_tilted_release(monkeypatch):
    """A tilted release (summary.goal_quat = grasp_5's 45 deg orientation): the goal / settle release and the record use grasp_5's own lift at
    that orientation (none: the plate's low rim ``drop_height_m`` above the floor) with the key, the opposite row's 5.65 cm without it."""
    q45 = IN.tilted_quat(IN.G.quat_from_yaw(0.4), GRASP_5.pose[0], 45.0)
    for per, above in ((False, 0.02 + 0.0565), (True, 0.02)):
        s = _TiltLiftStub(inside_surface_z_m=0.80, drop_height_m=0.02, release_tilt_deg=[45], release_lift_per_grasp=per)
        s.grasps = [GRASP_5, GRASP_5_OPPOSITE]
        release, d, rec, posed = _goal_check_release(s, GRASP_5, q45, monkeypatch)
        z_s = float(s.place["z"])
        assert float(IN.G.local_points_world(s.hull, d.goal_poses["jar"])[:, 2].min()) == pytest.approx(z_s + above, abs=1e-3)
        assert d.summary["release"]["release_above_m"] == pytest.approx(above, abs=1e-3)
        assert bool(th.equal(posed[0], release[0])) and bool(th.equal(d.goal_poses["jar"][0], release[0]))


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


# ----------------------------------------------------------------------------------------------------------------
# the attached place IK at C: failure diagnostics (owner 2026-10-13 ~00:50 item 2)
# ----------------------------------------------------------------------------------------------------------------

FRIDGE = "/World/scene_0/fridge_petcxr_0"
SHELF_ABOVE, DOOR, BOARD = f"{FRIDGE}/shelf_2/collisions/petcxr__shelf_2__collision__0", f"{FRIDGE}/door/collisions/petcxr__door__collision__0", \
    f"{FRIDGE}/shelf_1/collisions/petcxr__shelf_1__collision__0"
JAR_LINK = "attached_object_right_eef_link"


class _PlaceDiagStub:
    """What :meth:`IORSLarge._place_ik_fail_line` reads: a backend whose links touch the meshes ``touch`` names while those are enabled,
    a collision-blind differential IK landing on ``q_reach`` (reaching the pose when ``reach``), an IK checker with its ``solutions``."""

    _ik_solution_of = IL.IORSLarge._ik_solution_of
    _place_ik_fail_line = IL.IORSLarge._place_ik_fail_line
    _place_ik_contacts = IL.IORSLarge._place_ik_contacts

    def __init__(self, touch, reach=True):
        self.touch, self.enabled, self.ik_paths, self.ignored = dict(touch), {m: True for m in (SHELF_ABOVE, DOOR, BOARD)}, [], []
        self.q_reach = th.full((28,), 0.3)
        mesh = lambda path: SimpleNamespace(prim_path=path)  # noqa: E731
        self.partner = SimpleNamespace(name="fridge_petcxr_0", links={n: SimpleNamespace(collision_meshes={"c": mesh(p)})
                                                                       for n, p in (("shelf_2", SHELF_ABOVE), ("door", DOOR), ("shelf_1", BOARD))})
        self.world = SimpleNamespace(spawned={"jar": SimpleNamespace(name="jar", links={})})
        be = SimpleNamespace(_world_mesh_names=set(self.enabled), _disabled_obstacles=set(), _held_off=set(), kin_entry=lambda name: name,
                             _collision_costs=lambda entry: {"link_of_sphere": [JAR_LINK, JAR_LINK, "right_gripper_link", "right_arm_link4"]},
                             _mesh_enable=self.enabled.__setitem__, link_proximity=self._link_proximity, solve_ik_path=self._solve_ik_path)
        self.kit = SimpleNamespace(backend=be, _ensure_synced=lambda: None,
                                   collision_free=lambda q, margin=0.0: th.tensor([not any(self._touches(l) for l in self.touch)]))
        self.ikc = SimpleNamespace(solutions={}, _pose_key=IL.IK.IKChecker._pose_key, joint_limit_margin=lambda q, arm: (0.31, "right_arm_joint6"),
                                   reaches=lambda q, pose, arm: reach, fk_residual=lambda q, pose, arm: 0.034, q_locked=lambda st: th.zeros(28),
                                   ignoring=self._ignoring, self_collision_pairs=lambda q: [],
                                   _collision_blame=lambda q, ign: "collides with the scene object wall_3 (1.10 m from the base)",
                                   exact=lambda *a, **k: pytest.fail("the diagnostic re-solved the IK"))

    def _touches(self, link):
        return any(self.enabled[m] for m in self.touch.get(link, ()))

    def _link_proximity(self, q, links, margin, emb_name="whole_body"):
        assert th.equal(q, self.q_reach) and margin == 0.0
        return th.tensor([any(self._touches(l) for l in links)])

    def _solve_ik_path(self, emb, waypoints, q_start=None, check_collisions=True, sync_world=True, **kw):
        assert check_collisions is False and sync_world is False
        self.ik_paths.append((emb, waypoints, q_start))
        return SimpleNamespace(success=True, q_traj=th.stack([th.as_tensor(q_start, dtype=th.float32), self.q_reach]))

    def _ignoring(self, objs):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            self.ignored.append([getattr(o, "name", o) for o in objs])
            yield

        return ctx()


def _place_poses():
    q = th.tensor([0.0, 0.70710677, 0.0, 0.70710677])
    return (th.tensor([0.40, 0.10, 0.92]), q), (th.tensor([0.50, 0.10, 0.92]), q)


@pytest.mark.parametrize("pre_solved", [False, True])
def test_place_ik_failure_line_names_the_pose_reason_and_contacts(pre_solved):
    """The pose that failed (the place when the IK call wrote a fresh pre-place solution), the IK's reason and the robot / carried-object
    links x the partner's meshes they touch at a collision-blind configuration reaching that pose; the IK result, the IK checker's
    remembered solutions and the collision world are as before (no re-solve; the meshes switched off one at a time are back on)."""
    s = _PlaceDiagStub({JAR_LINK: [SHELF_ABOVE], "right_gripper_link": [DOOR, SHELF_ABOVE]})
    pre_place, place_eef = _place_poses()
    key = ("right", IL.IK.IKChecker._pose_key(pre_place))
    stale = th.full((28,), 0.1)
    s.ikc.solutions[key] = stale
    pre_before = s._ik_solution_of("right", pre_place)
    q_pre = th.full((28,), 0.2)
    if pre_solved:
        s.ikc.solutions[key] = q_pre  # what exact() leaves when the pre-place solved
    reason = "approach_untrackable: a configuration just before the end pose collides" if pre_solved else "exact IK failed at pose 1 of 2"
    r2 = IL.IK.IKResult(False, None, reason, 2)
    q_seed, floor = th.full((28,), 0.05), SimpleNamespace(name="floor_0")
    solutions = dict(s.ikc.solutions)
    line = s._place_ik_fail_line(r2, PLACE_C, SimpleNamespace(name="grasp_3"), 1.25, 0.0, pre_place, place_eef, "right", q_seed, [floor], pre_before)
    assert line.startswith("C place IK (attached) fail at d 0.45 λ +0.10 trunk (0.50, -0.30): grasp grasp_3 yaw 1.25 tilt 0: ")
    # the jar touches the shelf above alone; the gripper touches the door and that shelf at once: no single mesh frees it
    assert line.endswith(f"pose {'place' if pre_solved else 'pre-place'}: {reason}; contacts: {JAR_LINK} x fridge_petcxr_0/petcxr__shelf_2__collision__0, "
                         "right_gripper_link: collides with the scene object wall_3 (1.10 m from the base) (right_arm_joint6 310 mrad from its limit)")
    (emb, wps, q_start), = s.ik_paths
    assert emb == "right_arm" and th.equal(q_start, q_pre if pre_solved else q_seed)
    assert th.allclose(list(wps[0].values())[0][0], (place_eef if pre_solved else pre_place)[0])
    assert s.ignored == [["floor_0"] if pre_solved else []]
    assert (r2.ok, r2.q, r2.reason) == (False, None, reason) and s.ikc.solutions == solutions and all(s.enabled.values())


def test_place_ik_failure_line_single_mesh_pairs_rest_and_unreached():
    """Each mesh whose removal frees links is named with them; links touching two meshes at once (or a scene mesh) fall to
    ``_collision_blame``; a pose the collision-blind IK does not reach is a reach / joint-limit failure, without a collision query."""
    s = _PlaceDiagStub({JAR_LINK: [BOARD], "right_arm_link4": [DOOR, SHELF_ABOVE]})
    pre_place, place_eef = _place_poses()
    r2 = IL.IK.IKResult(False, None, "exact IK failed at pose 1 of 2", 2)
    line = s._place_ik_fail_line(r2, PLACE_C, SimpleNamespace(name="grasp_0"), 0.0, 30.0, pre_place, place_eef, "right", None, [], None)
    contacts = line.split("contacts: ", 1)[1]
    assert "tilt 30: pose pre-place" in line and contacts.startswith(f"{JAR_LINK} x fridge_petcxr_0/petcxr__shelf_1__collision__0, ")
    assert "right_arm_link4: collides with the scene object wall_3 (1.10 m from the base)" in contacts and all(s.enabled.values())
    s2 = _PlaceDiagStub({JAR_LINK: [BOARD]}, reach=False)
    s2.kit.collision_free = lambda q, margin=0.0: pytest.fail("a collision query for an unreached pose")
    line2 = s2._place_ik_fail_line(r2, PLACE_C, SimpleNamespace(name="grasp_0"), 0.0, 0.0, pre_place, place_eef, "right", None, [], None, push=True)
    assert line2.endswith("pose pre-place: exact IK failed at pose 1 of 2; contacts: none - not reached with collisions off (34 mm short, "
                          "right_arm_joint6 310 mrad from its limit)")
    s3 = _PlaceDiagStub({})
    s3.kit.backend.solve_ik_path = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuRobo down"))
    line3 = s3._place_ik_fail_line(r2, PLACE_C, SimpleNamespace(name="grasp_0"), 0.0, 0.0, pre_place, place_eef, "right", None, [], None)
    assert "contacts: not computed (RuntimeError('cuRobo down'))" in line3
