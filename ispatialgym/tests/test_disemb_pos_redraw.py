"""i7-yield2 r3 (lead #66): disembedding_10's target_placement - with the split cone each attempt tried only the drawn position's work
stances; ``ISPATIALGYM_DISEMB_POS_REDRAWS=N`` redraws the position (without replacement, from the drawn start's pool) within the attempt."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import disembedding as DE


def _task(good_keys, n_pos=5):
    cls = next(c for c in vars(DE).values() if isinstance(c, type) and "_sample_position" in vars(c))
    t = cls.__new__(cls)
    pool = {(i, 0): [SimpleNamespace(pos=i, k=j) for j in range(2)] for i in range(n_pos)}
    t._split_cone_pools = {None: pool}
    t.region_side = None
    t.manip_candidates = list(pool[(0, 0)])
    t.kit = SimpleNamespace(stats={})
    t.seen = []

    def sample_position(rng):
        pos = t.manip_candidates[0].pos
        t.seen.append(pos)
        if (pos, 0) in good_keys:
            return SimpleNamespace(summary={})
        raise Reject("target_placement", f"candidate {pos}: [grasp_unreachable] empty feasible set F")

    t._sample_position = sample_position
    return t


def test_switch_parsing(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_DISEMB_POS_REDRAWS", raising=False)
    assert DE.disemb_pos_redraws() == 0
    monkeypatch.setenv("ISPATIALGYM_DISEMB_POS_REDRAWS", "8")
    assert DE.disemb_pos_redraws() == 8
    monkeypatch.setenv("ISPATIALGYM_DISEMB_POS_REDRAWS", "x")
    assert DE.disemb_pos_redraws() == 0


def test_off_tries_the_drawn_position_only(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_DISEMB_POS_REDRAWS", raising=False)
    t = _task({(3, 0)})
    with pytest.raises(Reject) as e:
        t.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "target_placement" and t.seen == [0] and t.kit.stats == {}


def test_on_redraws_without_replacement_until_a_position_passes(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_DISEMB_POS_REDRAWS", "8")
    t = _task({(3, 0)})
    d = t.sample_instance(np.random.default_rng(0))
    assert t.seen[0] == 0 and t.seen[-1] == 3 and len(set(t.seen)) == len(t.seen)  # never the same position twice
    assert d.summary["pos_redraw"]["redraws"] == len(t.seen) - 1
    assert t.kit.stats["pos_redraw.passed"] == 1 and t.kit.stats["pos_redraw.redrawn"] == len(t.seen) - 1


def test_on_exhausted_and_other_rejects_pass_through(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_DISEMB_POS_REDRAWS", "2")
    t = _task(set())
    with pytest.raises(Reject) as e:
        t.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "target_placement" and len(t.seen) == 3 and t.kit.stats["pos_redraw.exhausted"] == 1
    t = _task(set())

    def other(rng):
        raise Reject("goal_occluded", "x")

    t._sample_position = other
    with pytest.raises(Reject) as e:
        t.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "goal_occluded"
    t = _task({(0, 0)})
    t._split_cone_pools = None  # no split cone: the single position, as before
    t.sample_instance(np.random.default_rng(0))
    assert t.seen == [0]
