"""i7-artlarge (2026-11-08): ``ISPATIALGYM_ARTLARGE_INIT_DRAW`` - the large articulation sampler's init (start) stance drawn from the
whole Phase-1 feasible set, the starts near an eval start last, and step 6's alternatives. Simulator-free: the pure helpers and
``_connect_init`` with ``prompt0.connect`` replaced."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.stances import Stance

AL = pytest.importorskip("ispatialgym.sampling.v2.tasks.articulation_large")


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", raising=False)
    assert AL.init_draw_on() is False
    monkeypatch.setenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", "1")
    assert AL.init_draw_on() is True
    monkeypatch.setenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", "0")
    assert AL.init_draw_on() is False


def test_start_near_needs_both_position_and_yaw():
    ev = [(9.53, -6.58, 3.08)]
    assert AL.start_near((9.60, -6.50, 3.10), ev)  # 0.11 m / 1 deg
    assert not AL.start_near((9.90, -6.58, 3.08), ev)  # 0.37 m
    assert not AL.start_near((9.53, -6.58, 3.08 - math.radians(20)), ev)  # 20 deg
    assert AL.start_near((9.53, -6.58, -3.14), [(9.53, -6.58, 3.14)])  # the yaw wraps
    assert not AL.start_near((0.0, 0.0, 0.0), [])


def test_order_puts_far_starts_first_and_is_a_permutation():
    ev = [(0.0, 0.0, 0.0)]
    poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.1, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    order, n_near = AL.order_init_candidates(poses, ev, np.random.default_rng(3))
    assert sorted(order) == list(range(5)) and n_near == 2
    assert set(order[:3]) == {1, 3, 4} and set(order[3:]) == {0, 2}
    # no eval starts: every candidate is "far", a random order
    order2, n2 = AL.order_init_candidates(poses, [], np.random.default_rng(3))
    assert sorted(order2) == list(range(5)) and n2 == 0


def _sampler(alts):
    s = AL.ArticulationLarge.__new__(AL.ArticulationLarge)
    s._init_alts = alts
    s._init_used = None
    s.gp = SimpleNamespace(variant="v")
    s.kit = SimpleNamespace(stats={})
    added = []
    s.sidecar = SimpleNamespace(add_stance=lambda st: added.append(st))
    return s, added


def _st(x):
    return Stance("init", (x, 0.0, 0.0), 0.5, 0.0)


def test_connect_init_off_raises_as_today(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", raising=False)
    s, _ = _sampler([_st(2.0)])

    def connect(_s, init, manip, path_max):
        raise Reject("path_collision", "blocked")

    monkeypatch.setattr(AL.P0, "connect", connect)
    with pytest.raises(Reject):
        s._connect_init(_st(1.0), _st(0.0), 3.0)
    assert s.kit.stats == {}


def test_connect_init_tries_alternatives(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", "1")
    s, added = _sampler([_st(2.0), _st(3.0), _st(4.0)])
    tried = []

    def connect(_s, init, manip, path_max):
        tried.append(init.base_pose2d[0])
        if init.base_pose2d[0] < 3.0:
            raise Reject("path_collision", "blocked")

    monkeypatch.setattr(AL.P0, "connect", connect)
    used = s._connect_init(_st(1.0), _st(0.0), 3.0)
    assert used.base_pose2d[0] == 3.0 and s._init_used is used and added == [used]
    assert tried == [1.0, 2.0, 3.0]
    assert s.kit.stats == {"artlarge_init.alt_tried": 2, "artlarge_init.alt_connected": 1}


def test_connect_init_other_reasons_and_exhaustion(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", "1")
    s, _ = _sampler([_st(2.0)])
    monkeypatch.setattr(AL.P0, "connect", lambda *_a: (_ for _ in ()).throw(Reject("handle_unreachable", "x")))
    with pytest.raises(Reject) as e:
        s._connect_init(_st(1.0), _st(0.0), 3.0)
    assert e.value.reason == "handle_unreachable" and s.kit.stats == {}
    s, _ = _sampler([_st(2.0), _st(3.0)])
    monkeypatch.setattr(AL.P0, "connect", lambda *_a: (_ for _ in ()).throw(Reject("no_transition", "y")))
    with pytest.raises(Reject) as e:
        s._connect_init(_st(1.0), _st(0.0), 3.0)
    assert e.value.reason == "no_transition"
    assert s.kit.stats == {"artlarge_init.alt_tried": 2, "artlarge_init.alt_exhausted": 1}


def test_connect_init_first_ok_counts(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_ARTLARGE_INIT_DRAW", "1")
    s, added = _sampler([_st(2.0)])
    monkeypatch.setattr(AL.P0, "connect", lambda *_a: None)
    init = _st(1.0)
    assert s._connect_init(init, _st(0.0), 3.0) is init and added == []
    assert s.kit.stats == {"artlarge_init.connect_first": 1}
