"""i7-artsmall (owner 2026-11-09, "articulation small: draw both joint angles per attempt, as articulation_large does"):
``ISPATIALGYM_ARTSMALL_DRAW_ANGLES=1`` - the (q_init, q_goal) pairs from the stable set S (open: closed -> s; close: s -> closed, a push-close
start clamped as the customs start is), the per-pair template work (passive probe, H_max), the per-attempt draw with the start search for the
drawn pair, the prefix map lookup's nearest start and the per-start push-close validation grid. Simulator-free (the samplers are built with
``__new__``; the tabletop phases are test_start_draw's fake layout)."""

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import core as CORE
from ispatialgym.sampling.v2.tasks import articulation_small as AS
from ispatialgym.tests.test_artic_prefix_maps import _pmeta, _spec, _write_prefix
from ispatialgym.tests import test_start_draw as SD

_base = SD._base
fake_tt = SD.fake_tt  # the fake tabletop layout (fixture)
S_CARTON = [1.1781, 1.3548, 1.5315, 1.7082, 1.885, 2.0617, 2.2384, 2.3562]  # close_02's S (int2_5 template)


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_ARTSMALL_DRAW_ANGLES", raising=False)
    assert not AS.draw_angles_on()
    monkeypatch.setenv("ISPATIALGYM_ARTSMALL_DRAW_ANGLES", "1")
    assert AS.draw_angles_on()


def test_angle_pairs():
    assert AS.angle_pairs("open", [1.5, 1.2, 2.1], 0.0, 2.1435, False) == [(0.0, 1.2), (0.0, 1.5), (0.0, 2.1)]
    # two-flap v2 / v4: the goal must lie past q_ajar
    assert AS.angle_pairs("open", [0.3, 0.35, 0.7], 0.0, 2.36, False, q_ajar=0.35) == [(0.0, 0.7)]
    # push-close: every s of S a start, the upper limit clamped 1 mrad inside (the customs start's clamp); goal closed
    p = AS.angle_pairs("close", S_CARTON, 0.0, 2.3562, True)
    assert len(p) == 8 and all(g == 0.0 for _, g in p) and p[-1][0] == pytest.approx(2.3552) and p[0][0] == pytest.approx(1.1781)
    assert AS.close_start(0.01, 0.0, 1.0) == pytest.approx(0.05)
    # free-hinge close: S as is; duplicates once
    assert AS.angle_pairs("close", [2.6, 2.6, 2.4], 0.0, 2.618, False) == [(2.4, 0.0), (2.6, 0.0)]
    assert AS.pair_key(1.17811, 0.0) == "1.1781->0.0000"


def _sampler(direction="close", close_push=True, stable=S_CARTON, upper=2.3562):
    s = _base(AS.ArticulationSmall, bands=None, left_bands=None, edge_frame=None, init_stance=None, manip_candidates=[], face_yaw=None, brace_arm="right",
              facing_local=th.tensor([1.0, 0.0, 0.0]), hull=th.zeros(4, 3), handle={"contact": [0.0, 0.0, 0.1]}, direction=direction, close_push=close_push,
              stable=list(stable), lower=0.0, upper=upper, q_ajar=None, v2=False, joint_name="j", handle_arm="right", q_init=None, q_goal=None)
    s.ctx = type("C", (), {"base_seed": 11})()
    s.g = lambda k, d=None: (0.35, 0.45) if k == "stand_offs_m" else d
    s.top = type("T", (), {"z": 0.8, "name": "top", "to_json": lambda self: {}})()
    probes = []

    def required(obj, q0=None, q1=None):
        probes.append((q0, q1))
        return ([0.25, 0.5] if q0 > 1.5 else [0.25, 0.5, 0.75, 1.0]), (0.5 if q0 > 1.5 else None)

    s._required_waypoints = required
    s._hull_at_q = lambda obj, q: th.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 0.1 + 0.1 * q]])
    s._brace_band = lambda edge, st: f"band@{edge.name}@{s.q_init:.4f}"
    return s, probes


@pytest.mark.usefixtures("fake_tt")
def test_template_probes_each_pair_once_and_finds_a_start():
    s, probes = _sampler()
    s._template_with_drawn_angles(target=type("O", (), {"name": "carton"})())
    assert len(s._pairs) == 8 and len(probes) == 8 and len(set(probes)) == 8  # once per (q_init, q_goal)
    assert s.init_stance is not None and s._template_pair["key"] == s._pair["key"]
    p = s._pair
    assert (s.q_init, s.q_goal, s.required, s.passive_from) == (p["q_init"], p["q_goal"], p["required"], p["passive_from"])
    assert s._start_margins["H_max"] == pytest.approx(0.1 + 0.1 * s.q_init)  # H_max at the pair's q_init
    assert s.bands.endswith(f"@{s.q_init:.4f}")  # the band of the pair's start
    facts = s.template_facts["artsmall_draw"]
    assert len(facts["pairs"]) == 8 and facts["template_pair"] == p["key"]


@pytest.mark.usefixtures("fake_tt")
def test_template_passes_over_a_pair_without_map(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")  # a prefix map miss rejects missing_map (artic_prefix under v2 only)
    s, _ = _sampler()

    def band(edge, st):
        if s.q_init < 2.0:
            raise Reject("missing_map", "no map")
        return "band"

    s._brace_band = band
    s._template_with_drawn_angles(target=type("O", (), {"name": "carton"})())
    assert s.q_init > 2.0  # a pair with a map
    s2, _ = _sampler()
    s2._brace_band = lambda edge, st: (_ for _ in ()).throw(Reject("missing_map", "none"))
    with pytest.raises(Reject) as e:
        s2._template_with_drawn_angles(target=type("O", (), {"name": "carton"})())
    assert e.value.reason == "missing_map"


@pytest.mark.usefixtures("fake_tt")
def test_attempts_draw_uniformly_and_redo_the_start_search(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    s, _ = _sampler()
    s._template_with_drawn_angles(target=type("O", (), {"name": "carton"})())
    seen = []
    for i in range(400):
        s.spa_draw(np.random.default_rng([5, i]))
        seen.append(s.q_init)
        assert s.bands.endswith(f"@{s.q_init:.4f}") and s._spa_facts["artsmall_pair"] == s._pair["key"]
        assert s.required == s._pair["required"] and s._start_margins is s._pair["margins"]
    vals, counts = np.unique(np.round(seen, 4), return_counts=True)
    assert len(vals) == 8 and counts.min() >= 30  # uniform over the 8 starts (400 draws: 50 each expected)
    assert s.kit.stats["artsmall_draw.drawn"] == 400
    assert sum(v for k, v in s.kit.stats.items() if k.startswith("artsmall_draw.q_init.")) == 400


@pytest.mark.usefixtures("fake_tt")
def test_a_failed_search_restores_the_previous_pair(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    s, _ = _sampler()
    s._template_with_drawn_angles(target=type("O", (), {"name": "carton"})())
    before = (s._pair["key"], s.init_stance, s.bands)
    s._brace_band = lambda edge, st: (_ for _ in ()).throw(Reject("missing_map", "none"))
    with pytest.raises(Reject) as e:
        s.spa_redraw_start(np.random.default_rng(1))
    assert e.value.reason == "missing_map" and (s._pair["key"], s.init_stance, s.bands) == before
    assert s.kit.stats["artsmall_draw.start_rejected.missing_map"] == 1


@pytest.mark.usefixtures("fake_tt")
def test_without_start_draw_the_pair_is_drawn_in_sample_instance(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "0")
    s, _ = _sampler(direction="open", close_push=False, stable=[1.2861, 1.5004, 2.1435], upper=2.1435)
    s._template_with_drawn_angles(target=type("O", (), {"name": "blender"})())
    s.spa_redraw_start(np.random.default_rng(0))  # no start draw: a no-op
    assert not getattr(s, "_pair_drawn", False)
    got = []
    s.manip_candidates = [object()]
    s._deadline = None
    s.sidecar = type("SC", (), {"reset_instance": lambda self: None, "add_stance": lambda self, st: None})()
    s.world = type("W", (), {"spawned": {}})()
    s.kit.sync_world = lambda: None
    s.manip_candidates = [s.init_stance.with_trunk(0.5, 0.0, "manipulation")]

    def inner(rng, init, manip, k):
        d = type("D", (), {})()
        d.summary = {}
        got.append((s.q_init, s.q_goal))
        return d

    s._sample_with = inner
    for i in range(30):
        d = s.sample_instance(np.random.default_rng([3, i]))
        assert d.summary["artsmall_draw"]["q_goal"] == s.q_goal and d.summary["artsmall_draw"]["q_init"] == 0.0
    assert {g for _, g in got} == {1.2861, 1.5004, 2.1435} and s.kit.stats["artsmall_draw.drawn"] == 30


@pytest.mark.usefixtures("fake_tt")
def test_off_keeps_the_template_pair(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_START_DRAW", "1")
    s, _ = _sampler()
    s.q_init, s.q_goal, s._start_margins = 1.6493, 0.0, {"kind": "articulation_small", "w_a": 0.1, "H_max": 0.2}
    for i in range(5):
        s.spa_draw(np.random.default_rng([2, i]))
        assert s.q_init == 1.6493 and "artsmall_pair" not in s._spa_facts
    assert not any(k.startswith("artsmall_draw.") for k in s.kit.stats)
    assert CORE.start_draw_active(s)


def test_find_prefix_serves_the_nearest_start(tmp_path):
    root = str(tmp_path / RM.PREFIX_DIR)
    ok = dict(policy_key=None, robot=None)
    a = _write_prefix(root, _pmeta(q_start=0.7854))  # the customs start (close_init_fraction 0.9 of the blender's 0.8727)
    b = _write_prefix(root, _pmeta(q_start=0.7636))  # a grid start 22 mrad away (within the 0.03 rad tolerance)
    assert RM.find_prefix(root, _spec(q_start=0.7854), **ok).path == a
    assert RM.find_prefix(root, _spec(q_start=0.7636), **ok).path == b
    assert RM.find_prefix(root, _spec(q_start=0.7700), **ok).path == b
    assert RM.find_prefix(root, _spec(q_start=0.70), **ok) is None


def test_validation_grid_matches_the_sampler_pairs():
    from ispatialgym.grasping import validate_artic as VA

    motion = {"lower": 0.0, "upper": 2.3562, "profile_spec": {"open_fraction_range": [0.5, 0.95], "stable_set_fractions": 7}}
    grid = VA.grid_q_starts(motion)
    pairs = AS.angle_pairs("close", S_CARTON, 0.0, 2.3562, True)
    assert len(grid) == 8 and all(abs(g - p[0]) < RM.PREFIX_Q_TOL_RAD for g, p in zip(grid, pairs))
    assert VA.library_path_q("/m", "a-b", 1.1781, "left") == "/m/artic_library__a-b__q1.1781__left.json"
    assert VA.library_path_q("/m", "a-b", 1.1781) == "/m/artic_library__a-b__q1.1781.json"
