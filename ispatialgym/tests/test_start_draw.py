"""i7-startdraw (owner 2026-11-08 3.5, "templates should never fix the robot's start base"): ``ISPATIALGYM_START_DRAW=1`` makes the
start-draw families draw the robot's start base per attempt - the template's own start search rerun with the attempt's rng.

No simulator: each family's sampler is built with ``__new__`` and the tabletop phases are replaced by a fake layout (4 edges of a 2 m
square top; edge ``e3`` fails the §4.2 floor / collision test, Phase 1c rejects laterals within 0.1 m of the edge midpoint - its
collision check -, ``phase2_manipulation_stance`` returns 3 stances at the init). Every test checks: off = the template's start, the rng
untouched; on = a start per attempt that differs across attempts and passes the fake rules AND the family's own rule in the reused code
(axial: the reference edge in frame; attachment: the partner room; disembedding: the free-floor run and the destination fit;
articulation: the facing yaw follows the edge; IORS large: the goal region in frame from stance A)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import core as CORE
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import alignment_axial as AX
from ispatialgym.sampling.v2.tasks import articulation_small as AS
from ispatialgym.sampling.v2.tasks import attachment as AT
from ispatialgym.sampling.v2.tasks import block_design as BD
from ispatialgym.sampling.v2.tasks import disembedding as DE
from ispatialgym.sampling.v2.tasks import iors_large as IL
from ispatialgym.sampling.v2.tasks import iors_small as IS

LAT_DEAD = 0.1  # Phase 1c's fake collision: |lateral| < LAT_DEAD
BLOCKED_EDGE = "e3"  # fails the fake §4.2 floor / collision test


def _edges():
    out = []
    for i, (mx, my, nx, ny) in enumerate(((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0), (-1.0, 0.0, -1.0, 0.0), (0.0, -1.0, 0.0, -1.0))):
        out.append(EdgeFrame.from_edge(f"e{i}", th.tensor([mx, my]), th.tensor([nx, ny]), 2.0))
    return out


def _init_at(edge, lateral=0.0, name="init"):
    xy = edge.base_xy(0.45, lateral)
    return Stance(name, (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, 0.45, edge, lateral=lateral)


@pytest.fixture
def fake_tt(monkeypatch):
    """The tabletop phases replaced by the fake layout; ``calls`` counts the Phase-1c draws."""
    calls = {"phase1c": 0}

    def edge_frames(s, top, rng=None, stand_off=0.45):
        es = _edges()
        order = [int(i) for i in rng.permutation(len(es))] if rng is not None else list(range(len(es)))
        return [(es[i], _init_at(es[i])) for i in order]

    def edge_candidates(s, top, rng=None, stand_off=0.45):
        for e, st in edge_frames(s, top, rng, stand_off):
            if e.name != BLOCKED_EDGE:
                yield e, st

    def phase1b(s, init, H, **kw):
        st = init.with_trunk(0.5, 0.0, "init")
        st.extras.update({"t_range": [0.5, 0.5], "lam_min": -0.3, "lam_max": 0.3, "d_ws": 0.6})
        return st

    def phase1c(s, init, margins, rng, *, top=None, bands=None, n_draws=20, in_frame_margin=0.05, lateral=None):
        calls["phase1c"] += 1
        for _ in range(n_draws):
            lat = float(rng.uniform(-0.6, 0.6))
            if abs(lat) < LAT_DEAD:
                continue
            st = init.shifted(lateral=lat, name="init")
            st.extras.update({"lateral_bounds": [-0.6, 0.6]})
            return st
        raise Reject("stance_collision", "no lateral")

    def phase2(s, init, half, **kw):
        out = []
        for k in range(3):
            c = init.with_trunk(0.5, 0.0, "manipulation")
            c.focus = th.tensor([*init.world(0.5, init.lateral + 0.05 * k).tolist(), 0.8])
            out.append(c)
        keep = kw.get("keep_pairs")
        return keep(out) if keep is not None else out

    monkeypatch.setattr(TT, "edge_frames", edge_frames)
    monkeypatch.setattr(TT, "edge_candidates", edge_candidates)
    monkeypatch.setattr(TT, "phase1b_init_trunk", phase1b)
    monkeypatch.setattr(TT, "phase1c_lateral", phase1c)
    monkeypatch.setattr(TT, "phase2_manipulation_stance", phase2)
    monkeypatch.setattr(TT, "ensure_bands", lambda s, *a, **k: "bands")
    monkeypatch.setattr(TT, "load_left_contact_band", lambda *a, **k: None)
    monkeypatch.setattr(TT, "left_band_compute", lambda: False)
    monkeypatch.setattr(TT, "contact_maps_on", lambda: False)
    monkeypatch.setattr(TT, "phase2_angle_classes", lambda s: None)
    monkeypatch.setattr(TT, "split_cone_opts", lambda s: None)
    monkeypatch.setattr(TT, "trunk_sweep_free", lambda s, st, *a, **k: False)
    monkeypatch.setattr(TT, "arm_offer", lambda s: (["right"], ""))
    monkeypatch.setattr(TT, "arm_bands_for", lambda *a, **k: {})
    monkeypatch.setattr(TT, "offered_arms", lambda s: ["right"])
    return calls


def _base(cls, **attrs):
    s = cls.__new__(cls)
    stats = {}
    s.kit = SimpleNamespace(stats=stats, trav_free=lambda xy, **k: True)
    s.gp = SimpleNamespace(variant="stub", gen=lambda k, d=None: d)
    s.gen = lambda k, d=None: d
    s.g = lambda k, d=None: d
    s.template_facts = {}
    if not isinstance(getattr(cls, "top", None), property):
        s.top = SimpleNamespace(z=0.8, name="top")
    s.ikc = SimpleNamespace(stance_collision_free=lambda st: False)
    s.camera_pose_at = lambda st: None
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _draws(s, n=12):
    out = []
    for i in range(n):
        s.spa_draw(np.random.default_rng([7, i]))
        out.append((s._spa_facts["edge"], float(s._spa_facts["lateral_m"]), tuple(s._spa_facts["base_pose2d"])))
    return out


def _check_varied(draws, blocked=BLOCKED_EDGE):
    assert len({d[2] for d in draws}) >= len(draws) - 1  # a start per attempt (continuous laterals: no two alike)
    assert len({d[0] for d in draws}) >= 2  # more than one edge
    for edge, lat, _ in draws:
        assert edge != blocked and abs(lat) >= LAT_DEAD  # the §4.2 test (edge_candidates) and Phase 1c's checks


def _check_off(s, monkeypatch, attr="init_stance"):
    monkeypatch.delenv("ISPATIALGYM_START_DRAW", raising=False)
    before = getattr(s, attr)
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    s.spa_redraw_start(rng)
    assert getattr(s, attr) is before and rng.bit_generator.state == state  # the template's start, the rng untouched
    assert not CORE.start_draw_active(s) and not CORE.start_per_attempt(s)


# ------------------------------------------------------------------------------------------------------------------- the switch


def test_switch_is_family_scoped(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_START_PER_ATTEMPT", raising=False)
    monkeypatch.delenv("ISPATIALGYM_START_DRAW", raising=False)
    fam, other = SimpleNamespace(START_DRAW_FAMILY=True), SimpleNamespace(START_DRAW_FAMILY=False)
    assert not CORE.start_per_attempt(fam) and not CORE.start_per_attempt()
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    assert CORE.start_per_attempt(fam) and not CORE.start_per_attempt(other) and not CORE.start_per_attempt()
    for cls in (AX.AlignmentAxial, AS.ArticulationSmall, AT.Attachment, BD.BlockDesign, DE.Disembedding, IL.IORSLarge, IS.IORSSmall):
        assert cls.START_DRAW_FAMILY
    from ispatialgym.sampling.v2.tasks import iors_large_nextto as NM

    assert not NM.NextToMode.START_DRAW_FAMILY and not NM.TouchingBasePushMode.START_DRAW_FAMILY  # stance A per attempt already
    assert not CORE.SamplerV2.START_DRAW_FAMILY


class _Stub:
    sample = CORE.SamplerV2.sample
    spa_draw = CORE.SamplerV2.spa_draw
    START_DRAW_FAMILY = True

    def __init__(self):
        self.manip_candidates = []
        self.sidecar = SimpleNamespace(reset_instance=lambda: None, stances={})
        self.gp = SimpleNamespace(variant="stub")
        self.kit = SimpleNamespace(stats={})
        self.template_facts = {}
        self.drawn = 0

    def gen(self, key, default=None):
        return default

    def spa_redraw_start(self, rng):
        if CORE.start_draw_active(self):
            self.drawn += 1
            self._spa_facts = {"edge": f"e{int(rng.integers(3))}"}

    def sample_instance(self, rng):
        return SimpleNamespace(summary={}, robot_q=0, base_pose2d=(0, 0, 0))

    def goal(self, draft):
        pass


def test_core_off_is_identical_on_counts(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_START_PER_ATTEMPT", raising=False)
    monkeypatch.delenv("ISPATIALGYM_START_DRAW", raising=False)
    s = _Stub()
    d = s.sample(np.random.default_rng(0))
    assert s.drawn == 0 and "start_per_attempt" not in d.summary and s.kit.stats == {}
    monkeypatch.setenv("ISPATIALGYM_START_PER_ATTEMPT", "1")  # train85's switch alone: the hook runs, a start-draw module returns at once
    d = s.sample(np.random.default_rng(0))
    assert s.drawn == 0 and s.kit.stats == {}
    monkeypatch.delenv("ISPATIALGYM_START_PER_ATTEMPT")
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    for i in range(6):
        d = s.sample(np.random.default_rng(i))
    assert s.drawn == 6 and s.kit.stats["start_draw.drawn"] == 6 and "start_per_attempt" in d.summary
    assert sum(v for k, v in s.kit.stats.items() if k.startswith("start_draw.edge.")) == 6

    def bad(rng):
        raise Reject("no_stance", "x")

    s.spa_redraw_start = bad
    with pytest.raises(Reject):
        s.sample(np.random.default_rng(9))
    assert s.kit.stats["start_draw.rejected.no_stance"] == 1


def test_repeat_outcome_counters(monkeypatch):
    s = SimpleNamespace(START_DRAW_FAMILY=True, kit=SimpleNamespace(stats={}), template_meta={}, instance_seeds=[1])
    s.instance_filter = lambda p: None
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    CORE.SamplerV2.check_instance_filter(s, "/nonexistent-tro_state.json")
    s.instance_filter = lambda p: "repeats eval 0"
    with pytest.raises(Reject):
        CORE.SamplerV2.check_instance_filter(s, "/nonexistent-tro_state.json")
    assert s.kit.stats == {"start_draw.repeat_passed": 1, "start_draw.repeat_rejected": 1}
    monkeypatch.delenv("ISPATIALGYM_START_DRAW")
    s.kit.stats.clear()
    s.instance_filter = lambda p: None
    CORE.SamplerV2.check_instance_filter(s, "/nonexistent-tro_state.json")
    assert s.kit.stats == {}


# ------------------------------------------------------------------------------------------------------------------- the families


def _axial(monkeypatch):
    s = _base(AX.AlignmentAxial, band_spec=None, left_bands=None, edge_frame=None, init_stance=None, manip_candidates=[],
              _start_margins={"kind": "axial", "w_diag": 0.2, "safety": 0.02, "H_max": 0.05})

    def visible(st, cam, lat, pose, with_in_frame=False):
        n = 9 if st.lateral > -0.3 else 0  # the reference edge out of frame left of -0.3
        return (n, n) if with_in_frame else n

    s._edge_visible_points = visible
    return s


def test_axial(fake_tt, monkeypatch):
    s = _axial(monkeypatch)
    s.ctx = SimpleNamespace(base_seed=5)
    assert s._start_search(np.random.default_rng(5), s._start_margins, [])
    t0 = (s.edge_frame.name, s.init_stance.lateral)
    s2 = _axial(monkeypatch)
    s2._start_search(np.random.default_rng(5), s2._start_margins, [])
    assert (s2.edge_frame.name, s2.init_stance.lateral) == t0  # the template seed's search is deterministic
    _check_off(s, monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    draws = _draws(s)
    _check_varied(draws)
    assert all(lat > -0.3 for _, lat, _ in draws)  # the reference edge in frame from the init camera
    assert s.kit.stats["start_draw.drawn"] == 12


def test_attachment(fake_tt, monkeypatch):
    s = _base(AT.Attachment, band_spec=None, left_bands=None, edge_frame=None, init_stance=None, manip_candidates=[], t_role="target", p_role="partner",
              hull={"target": th.zeros(4, 3), "partner": th.zeros(4, 3)}, rest={"target": None, "partner": None}, height={"target": 0.1, "partner": 0.1},
              _candidate_rejects={3: "x"})
    s._attach_band = lambda spec, edge, view: "attach"
    monkeypatch.setattr(TT, "half_extent", lambda *a, **k: 0.05)

    def room(cands, edge):
        if edge.name == "e1":
            raise Reject("no_partner_cell", "none")
        return cands

    s._with_partner_room = room
    s._start_args = ("spec", None, 0.1, {"kind": "attachment"})
    _check_off(s, monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    draws = _draws(s)
    _check_varied(draws)
    assert all(e != "e1" for e, _, _ in draws) and s._candidate_rejects == {}  # the partner room; the old candidates' memory dropped


def test_block_design(fake_tt, monkeypatch):
    s = _base(BD.BlockDesign, band_spec=None, left_bands=None, edge_frame=None, init_stance=None, manip_candidates=[], edge=0.03,
              _start_margins={"kind": "block_design", "edge": 0.03, "H_max": 0.03})
    s._init_with_room_for_flips = lambda init: init
    s._manip_lateral_shifts = lambda init: [0.0]
    s._flips_expected = lambda: False
    _check_off(s, monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    _check_varied(_draws(s))


def test_articulation_small(fake_tt, monkeypatch):
    s = _base(AS.ArticulationSmall, bands=None, left_bands=None, edge_frame=None, init_stance=None, manip_candidates=[], face_yaw=None, brace_arm="right",
              facing_local=th.tensor([1.0, 0.0, 0.0]), hull=th.zeros(4, 3), handle={"contact": [0.0, 0.0, 0.1]},
              _start_margins={"kind": "articulation_small", "w_a": 0.1, "H_max": 0.2})
    s._brace_band = lambda edge, st: f"band@{edge.name}"
    s.g = lambda k, d=None: (0.35, 0.45) if k == "stand_offs_m" else d
    _check_off(s, monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    for i in range(8):
        s.spa_draw(np.random.default_rng([2, i]))
        e = s.edge_frame
        want = math.atan2(float(-e.inward[1]), float(-e.inward[0]))  # facing_local +x: the target faces the robot across the drawn edge
        assert abs(math.remainder(s.face_yaw - want, 2 * math.pi)) < 1e-6 and s.bands == f"band@{e.name}"
    _check_varied(_draws(s))


def test_disembedding(fake_tt, monkeypatch):
    s = _base(DE.Disembedding, band_spec=None, bands=None, edge=None, init_stance=None, manip_candidates=[], segment_arms=["right"], arm_offer_why="",
              arm_bands={}, hull={"source": th.zeros(4, 3)}, src_role="source", height={"source": 0.05},
              src_floor=SimpleNamespace(z_local=0.01), _start_margins={"kind": "disembedding", "H_max": 0.1, "destination_side": "left"})
    monkeypatch.setattr(TT, "half_extent", lambda *a, **k: 0.1)
    # free floor nowhere in front of e1 (the free-floor run rule), the destination never fits right of lateral +0.3
    s.kit.trav_free = lambda xy, **k: not (float(xy[1]) > 1.2)
    s._destination_fit = lambda init, cs: [c for c in cs if init.lateral <= 0.3]
    s._source_tray_in_frame = None
    _check_off(s, monkeypatch)
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    draws = _draws(s, 16)
    _check_varied(draws, blocked=None)  # every edge with a free-floor run (edge_frames, not the midpoint test)
    assert all(e != "e1" and lat <= 0.3 for e, lat, _ in draws)


def _iors_large(kind="support"):
    s = _base(IL.IORSLarge, relation="ontop")
    s.__class__ = type("IL", (IL.IORSLarge,), {"partner": None, "target": None})
    s.target_dims = lambda: (0.1, 0.2)
    s.goal_region_visible = lambda st, from_xy=None: st.lateral > 0.0  # the partner's goal region in frame from stance A
    s.kit.sync_world = lambda: None
    support = SimpleNamespace(name="table")
    top = SimpleNamespace(free=th.ones(100), z=0.8, name="table")
    tmpl = _init_at(_edges()[0], 0.2)
    s.surface = {"kind": kind, "obj": None if kind == "floor" else support, "top": top, "init": tmpl, "manip": [tmpl], "n_c": 7}
    return s


def test_iors_large_support(fake_tt, monkeypatch):
    s = _iors_large()
    _check_off(s, monkeypatch, attr="surface")
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    draws = _draws(s)
    _check_varied(draws)
    assert all(lat > 0.0 for _, lat, _ in draws)  # the goal region rule of stance A
    assert s.surface["n_c"] == 7 and s.surface["top"].name == "table" and s.surface["init"].name == "init"


def test_iors_large_floor_and_reject(fake_tt, monkeypatch):
    s = _iors_large("floor")
    keep = s.surface

    def floor(rng):
        c_t = th.tensor([float(rng.uniform(-2, 2)), 2.0])
        init = Stance("init", (float(c_t[0]), 1.0, 0.0), 0.5, 0.0)
        s.surface.update({"c_t": c_t, "init": init, "manip": [init], "init_xy": c_t, "z": 0.0, "edge": None})

    s.try_floor_surface = floor
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    xs = {s.spa_draw(np.random.default_rng(i)) or s.surface["init"].base_pose2d[0] for i in range(6)}
    assert len(xs) == 6 and s._spa_facts["kind"] == "floor" and "c_t" in s._spa_facts

    def fail(rng):
        raise Reject("no_floor_cell", "none")

    s.try_floor_surface = fail
    before = s.surface
    with pytest.raises(Reject):
        s.spa_draw(np.random.default_rng(0))
    assert s.surface["init"] is before["init"] and s.surface["c_t"] is before["c_t"]  # the last good surface kept
    assert s.kit.stats["start_draw.rejected.no_floor_cell"] == 1
    assert keep["kind"] == "floor"


def test_iors_small_sample_draws_under_the_switch(monkeypatch):
    """IORS small drew per attempt under train85's switch already; START_DRAW alone now routes it the same way (``spa_draw``)."""
    monkeypatch.delenv("ISPATIALGYM_START_PER_ATTEMPT", raising=False)
    calls = []
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.shelf_start = None
    s.kit = SimpleNamespace(stats={})
    s.gp = SimpleNamespace(variant="stub")
    s.spa_redraw_start = lambda rng: calls.append(1) or setattr(s, "_spa_facts", {"edge": "e0"})
    s.draw_partner_side = lambda rng: 1.0
    monkeypatch.setattr(CORE.SamplerV2, "sample", lambda self, rng: "drawn")
    monkeypatch.delenv("ISPATIALGYM_START_DRAW", raising=False)
    assert IS.IORSSmall.sample(s, np.random.default_rng(0)) == "drawn" and calls == []
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    IS.IORSSmall.sample(s, np.random.default_rng(0))
    assert calls == [1] and s._spa_drawn and s.kit.stats["start_draw.drawn"] == 1
