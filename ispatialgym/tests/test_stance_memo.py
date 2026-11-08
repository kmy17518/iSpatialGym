"""i7-stancememo (owner 2026-11-08, 5x.1): the per-instance floor stance memo (ISPATIALGYM_STANCE_MEMO) and the redraw after 2 B / C
search failures (ISPATIALGYM_FLOOR_REDRAW). The memo is per instance (a second instance starts empty), a redraw clears it, both
switches off = today (no memo, no redraw, the pre-screen's random draw unchanged)."""
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import core as CORE
from ispatialgym.sampling.v2 import stance_memo as SM
from ispatialgym.sampling.v2.stances import Stance
from ispatialgym.sampling.v2.tasks import iors_large as IL

B1 = (1.0, 2.0, 0.5, 0.95, -1.2)
B2 = (1.5, 2.0, 0.5, 0.95, -0.9)
A1 = (0.0, 0.0, 0.0, 0.0, 0.0)
A2 = (0.3, -0.4, 0.1, 0.0, 0.0)
C1 = (3.0, 1.0, 1.57, 0.95, -1.2)


@pytest.fixture
def env(monkeypatch):
    for k in (SM.ENV_MEMO, SM.ENV_REDRAW, SM.ENV_RECHECK):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _memo():
    counts = {}
    return SM.StanceMemo(lambda k: counts.__setitem__(k, counts.get(k, 0) + 1)), counts


# ------------------------------------------------------------------------------------------------------------------- the memo


def test_switches_default_off(env):
    assert not SM.memo_on() and SM.redraw_after() == 0 and not SM.active()
    env.setenv(SM.ENV_REDRAW, "1")
    assert SM.redraw_after() == 2 and SM.active()
    env.setenv(SM.ENV_REDRAW, "3")
    assert SM.redraw_after() == 3


def test_memo_is_per_instance(env):
    env.setenv(SM.ENV_RECHECK, "0")
    m, counts = _memo()
    assert m.enter_instance(330003)
    m.b_failed(B1, A1, "grasp_unreachable", "C (d 0.88 λ +0.23): no push chain at the largest-offset yaws")
    ck = m.c_key(B1, C1, "right", 0.0)
    m.c_failed(ck, "C (d 0.88 λ +0.23): no push chain at the largest-offset yaws (0.11, 0.11 m)", A1)
    assert m.lookup_b(B1, A2)[0]["reason"] == "grasp_unreachable"  # not A-dependent: any stance A
    assert m.lookup_c(ck, A2)[0]["reason"] == "push_chain"
    assert not m.enter_instance(330003)  # the same instance: kept
    assert m.lookup_b(B1, A1)[0] is not None
    assert m.enter_instance(330004)  # the next instance starts empty
    assert m.lookup_b(B1, A1) == (None, False) and m.lookup_c(ck, A1) == (None, False)
    assert counts["memo.cleared.instance"] == 1 and counts["memo.recorded.b"] == 1 and counts["memo.recorded.c"] == 1
    assert counts["memo.skipped.b.grasp_unreachable"] == 2 and counts["memo.skipped.c.push_chain"] == 1


def test_redraw_clears_the_memo(env):
    env.setenv(SM.ENV_RECHECK, "0")
    m, counts = _memo()
    m.enter_instance(7)
    m.b_failed(B1, A1, "grasp_unreachable", "empty feasible set F_init at B's focus cell")
    m.search_fails = 2
    m.new_start()
    assert m.start_id == 1 and m.search_fails == 0 and m.lookup_b(B1, A1) == (None, False) and counts["memo.cleared.redraw"] == 1


def test_a_dependent_failures_match_only_their_stance_a(env):
    env.setenv(SM.ENV_RECHECK, "0")
    m, _ = _memo()
    m.enter_instance(1)
    m.b_failed(B1, A1, "no_transition", "init -> manipulation: navigate-then-torso: path_collision: ...")
    m.b_failed(B2, A1, "grasp_unreachable", "C ...: no (yaw, grasp) of F (41) passes", a_dep=True)  # a draw met the A -> B drive
    assert m.lookup_b(B1, A1)[0] is not None and m.lookup_b(B1, A2) == (None, False)
    assert m.lookup_b(B2, A1)[0] is not None and m.lookup_b(B2, A2) == (None, False)
    ck = m.c_key(B1, C1, "right", 0.0)
    m.c_failed(ck, "C (d 0.90 ...): no (yaw, grasp) of F (36) passes the exact IK at B and C — {'B: the A -> B drive passes the placed target': 8}", A1)
    assert m.lookup_c(ck, A1)[0] is not None and m.lookup_c(ck, A2) == (None, False)
    assert m.c_key(B1, C1, "right", 0.0) != m.c_key(B1, C1, "right", 0.5)  # A's trunk translate: the carry's travel trunk


def test_attempt_stops_are_never_recorded(env):
    m, counts = _memo()
    m.enter_instance(1)
    m.b_failed(B1, A1, "time_budget", "attempt budget of 1500 s spent")
    m.b_failed(B1, A1, "time_cap", "attempt cap of 650 s spent")
    assert not m.entries and "memo.recorded.b" not in counts


def test_every_kth_hit_is_rechecked(env):
    env.setenv(SM.ENV_RECHECK, "3")
    m, counts = _memo()
    m.enter_instance(1)
    m.b_failed(B1, A1, "grasp_unreachable", "x")
    seen = [m.lookup_b(B1, A1)[1] for _ in range(6)]
    assert seen == [False, False, True, False, False, True] and counts["memo.recheck.b"] == 2
    m.b_passed(B1, A1)
    assert m.lookup_b(B1, A1) == (None, False) and counts["memo.would_have_passed.b"] == 1


def test_c_classes():
    assert SM.c_class("C (d 0.45 λ -0.10): skipped - the carry leg failed at this base pose") == "carry_leg"
    assert SM.c_class("C: F_init (10) ∩ F_goal (8) empty at goal cell [-1.44, -2.39]") == "empty_F"
    assert SM.c_class("C (d 0.55 trunk (0.5, -0.3)): no (yaw, grasp) of F (1) passes the exact IK at B and C — {}") == "no_pair"
    assert SM.c_class("something new") == "other"


# ------------------------------------------------------------------------------------------------------------------- IORS large


def _iors(env, kind="floor"):
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.kit = SimpleNamespace(stats={})
    s.gp = SimpleNamespace(variant="stub", gen=lambda k, d=None: d)
    s.ctx = SimpleNamespace(base_seed=11)
    init = Stance("init", (0.0, 0.0, 0.0), 0.0, 0.0)
    b = Stance("manipulation", (1.0, 2.0, 0.5), 0.95, -1.2)
    s.surface = {"kind": kind, "obj": None, "c_t": th.tensor([1.0, 2.5]), "init": init, "manip": [b], "n_c": 5}
    return s


def test_iors_off_is_today(env):
    s = _iors(env)
    assert s._sm() is None
    s.attempt_begin(1, 0)
    s.attempt_end(1, 0, "rejected", "sample", "target_placement")
    assert "_stance_memo" not in s.__dict__ and not s.kit.stats
    # the pre-screen's random draw without the memo: one rng.integers + pop, as before
    pool = [Stance("manipulation", (float(i), 0.0, 0.0), 0.95, -1.2) for i in range(5)]
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    got = s._sm_draw_b(None, list(pool), r1, s.surface["init"])
    ref = list(pool)
    assert got is ref.pop(int(r2.integers(len(ref)))) and r1.bit_generator.state == r2.bit_generator.state


def test_iors_memo_only_on_floor_inits(env):
    env.setenv(SM.ENV_MEMO, "1")
    assert _iors(env, kind="support")._sm() is None
    assert _iors(env)._sm() is not None


def test_iors_skips_a_memo_b_in_the_prescreen_draw(env):
    env.setenv(SM.ENV_MEMO, "1")
    env.setenv(SM.ENV_RECHECK, "0")
    s = _iors(env)
    s.attempt_begin(5, 0)
    m = s._sm()
    pool = [Stance("manipulation", (float(i), 0.0, 0.0), 0.95, -1.2) for i in range(4)]
    dead = {SM.stance_key(p) for p in pool[:3]}
    for k in dead:
        m.b_failed(k, SM.stance_key(s.surface["init"]), "grasp_unreachable", "x")
    s._sm_recheck_b, s._sm_skipped_b = set(), 0
    # the same random draw as without the memo; the memo's B are passed over until a live one comes up
    ref, r_ref, n_dead = list(pool), np.random.default_rng(8), 0
    while True:
        b = ref.pop(int(r_ref.integers(len(ref))))
        if SM.stance_key(b) not in dead:
            break
        n_dead += 1
    got = s._sm_draw_b(m, list(pool), np.random.default_rng(8), s.surface["init"])
    assert n_dead == 3 and got is pool[3] and s._sm_skipped_b == n_dead and s.kit.stats.get("memo.skipped.b.grasp_unreachable", 0) == n_dead
    assert s._sm_draw_b(m, list(pool[:3]), np.random.default_rng(0), s.surface["init"]) is None  # every B of the pool in the memo


def test_iors_instance_change_clears_and_restores_the_template_start(env):
    env.setenv(SM.ENV_MEMO, "1")
    env.setenv(SM.ENV_REDRAW, "1")
    s = _iors(env)
    tmpl_ct = s.surface["c_t"]
    s.attempt_begin(10, 0)
    m = s._sm()
    m.b_failed((9.0, 9.0, 0.0, 0.95, -1.2), (0.0,) * 5, "grasp_unreachable", "x")
    for att in (0, 1):  # two B / C search failures on the template's start
        s._sm_search_failed = "target_placement"
        s.attempt_end(10, att, "rejected", "sample", "target_placement")
    assert m.search_fails == 2 and s.kit.stats["redraw.search_fail"] == 2

    new_b = Stance("manipulation", (4.0, 4.0, 0.0), 0.95, -1.2)

    def try_surface(kind, cand, rng):
        assert kind == "floor" and cand is None and isinstance(rng, np.random.Generator)
        s.surface = {"kind": "floor", "obj": None, "c_t": th.tensor([4.0, 4.5]), "init": Stance("init", (5.0, 5.0, 0.0), 0.0, 0.0),
                     "manip": [new_b], "n_c": 3}

    s.try_surface = try_surface
    s.attempt_begin(10, 2)
    s._sm_maybe_redraw(m)
    assert s.kit.stats["redraw.fired"] == 1 and m.start_id == 1 and not m.entries and float(s.surface["c_t"][0]) == 4.0
    s.attempt_begin(11, 0)  # the next instance: an empty memo, on the template's floor start
    assert s.surface["c_t"] is tmpl_ct and m.instance == 11 and m.start_id == 0 and s.kit.stats["redraw.template_restored"] == 1


def test_iors_redraw_failure_keeps_the_start(env):
    env.setenv(SM.ENV_REDRAW, "1")
    s = _iors(env)
    s.attempt_begin(3, 0)
    m = s._sm()
    m.search_fails = 2
    before = s.surface

    def fail(kind, cand, rng):
        s.surface = {"kind": "floor", "obj": None}  # what try_surface leaves on a rejection
        raise Reject("no_floor_cell", "no anchor cell admits stances A / B")

    s.try_surface = fail
    s._sm_maybe_redraw(m)
    assert s.surface is before and m.search_fails == 0 and s.kit.stats["redraw.failed.no_floor_cell"] == 1 and "redraw.fired" not in s.kit.stats


def test_iors_redraw_needs_two_search_failures(env):
    env.setenv(SM.ENV_REDRAW, "1")
    s = _iors(env)
    s.attempt_begin(3, 0)
    m = s._sm()
    calls = []
    s.try_surface = lambda *a: calls.append(a)
    s._sm_search_failed = None  # rejected after the search (program_check): not a B / C search failure
    s.attempt_end(3, 0, "rejected", "program_check", "approach_untrackable")
    s._sm_search_failed = "target_placement"
    s.attempt_end(3, 1, "rejected", "sample", "target_placement")
    s._sm_maybe_redraw(m)
    assert not calls and m.search_fails == 1


def test_iors_third_attempt_is_logged_with_the_redraw_off(env):
    env.setenv(SM.ENV_MEMO, "1")
    s = _iors(env)
    s.attempt_begin(4, 0)
    for att in (0, 1):
        s._sm_search_failed = "target_placement"
        s.attempt_end(4, att, "rejected", "sample", "target_placement")
    assert s.kit.stats["redraw.would_fire"] == 1 and s._sm().pending_third
    s._sm_search_failed = None
    s.attempt_end(4, 2, "written", "goal_occlusion", None)
    assert s.kit.stats["redraw.third_attempt_would_have_succeeded.written"] == 1 and not s._sm().pending_third


def test_core_calls_the_attempt_hooks():
    src = inspect.getsource(CORE.SamplerV2.run_instances)
    assert "self.attempt_begin(index, attempt)" in src
    assert 'self.attempt_end(index, attempt, "written", phase, None)' in src and 'self.attempt_end(index, attempt, "rejected", phase, e.reason)' in src
    src_c = inspect.getsource(IL.IORSLarge._couple)
    assert "order if sm is not None else order[:n_try]" in src_c  # the memo's skipped C do not use up a try; off = the former slice
    assert math.isfinite(1.0)
