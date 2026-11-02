"""train-fast SPLIT_CONE_RESTORE (2026-11-01): under start-per-attempt (train85), ``SamplerV2.sample`` reset the split-cone caches
every attempt while ``tabletop.split_cone_draw`` had narrowed ``manip_candidates`` to the previous attempt's position, so every
attempt after the first drew from that one position (disembedding_03: 0 written in 85 attempts when the first position failed).
A stub sampler (no simulator): candidates at 6 positions, the base no-op ``spa_redraw_start``."""

from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling.v2 import core as CORE


def _cand(pos, k):
    return SimpleNamespace(extras={"_position": (pos, 0), "init_view_class": 0, "angle_class": 0}, focus=None, name=f"c{pos}_{k}")


class _Stub:
    sample = CORE.SamplerV2.sample
    spa_redraw_start = CORE.SamplerV2.spa_redraw_start

    def __init__(self):
        self.manip_candidates = [_cand(p, k) for p in range(6) for k in range(2)]
        self.sidecar = SimpleNamespace(reset_instance=lambda: None, stances={})
        self.gp = SimpleNamespace(alias="stub")
        self.seen = []

    def gen(self, key, default=None):
        return True if key == "phase2_split_cone" else default

    def sample_instance(self, rng):
        self.seen.append({c.extras["_position"][0] for c in self.manip_candidates})
        return SimpleNamespace(summary={}, robot_q=0, base_pose2d=(0, 0, 0))

    def goal(self, draft):
        pass


def _positions_over_attempts(monkeypatch, restore: bool, n: int = 30):
    monkeypatch.setenv("ISPATIALGYM_START_PER_ATTEMPT", "1")
    monkeypatch.setenv("ISPATIALGYM_FAST_SPLIT_CONE_RESTORE", "1" if restore else "0")
    s = _Stub()
    for i in range(n):
        s.sample(np.random.default_rng(i))
    assert all(len(p) == 1 for p in s.seen)  # each attempt's candidates are one position's stances
    return [next(iter(p)) for p in s.seen], s


def test_every_attempt_draws_among_all_positions(monkeypatch):
    drawn, s = _positions_over_attempts(monkeypatch, restore=True)
    assert len(set(drawn)) >= 4
    assert len(s._split_cone_cands) == 12


def test_switch_off_is_the_collapse(monkeypatch):
    drawn, _ = _positions_over_attempts(monkeypatch, restore=False)
    assert len(set(drawn[1:])) == 1 and drawn[1] == drawn[0]  # the patch's behaviour: stuck at the first attempt's position


def test_without_start_per_attempt_nothing_changes(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_START_PER_ATTEMPT", raising=False)
    s = _Stub()
    for i in range(20):
        s.sample(np.random.default_rng(i))
    assert len({next(iter(p)) for p in s.seen}) >= 4  # the pools are built once and kept
