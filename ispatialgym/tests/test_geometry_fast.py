"""Batched ``relation_error`` / ``orientation_error`` (``ispatialgym.eval.geometry_fast``) against the reference
implementation in ``ispatialgym.eval.geometry``, bit for bit (no simulator)."""

import math
import struct

import pytest
import torch as th

from ispatialgym.eval import geometry as G
from ispatialgym.eval import geometry_fast as F

YAW72 = [G.axis_angle_quat([0.0, 0.0, 1.0], 2 * math.pi * k / 72) for k in range(1, 72)]
YAW4 = [G.axis_angle_quat([0.0, 0.0, 1.0], k * math.pi / 2) for k in range(1, 4)]
FLIP = [G.axis_angle_quat([1.0, 0.0, 0.0], math.pi)]
TOLERANCES = [(None, None), (0.02, 0.2618), (0.01, 0.0873), (0.0, 0.1), (0.05, None)]


def _bits(values):
    return [struct.pack("<d", float(v)) for v in values]


def _cases(n, seed):
    gen = th.Generator().manual_seed(seed)

    def quat():
        return G.quat_normalize(th.randn(4, generator=gen).float())

    for c in range(n):
        eqs = [[], FLIP, YAW4, YAW4 + FLIP, [quat() for _ in range(5)]][c % 5]
        a = (th.randn(3, generator=gen).float() * 0.3, quat())
        b = (th.randn(3, generator=gen).float() * 0.3, quat())
        goal = G.relative_pose(a, b)
        noise = 10.0 ** -(1 + c % 4)
        goal = (
            goal[0] + th.randn(3, generator=gen).float() * noise,
            G.quat_normalize(goal[1] + th.randn(4, generator=gen).float() * noise),
        )
        yield a, b, goal, eqs, TOLERANCES[c % len(TOLERANCES)]


def test_relation_and_orientation_error_match_the_reference_bitwise():
    for a, b, goal, eqs, (pt, ot) in _cases(60, seed=0):
        assert _bits(F.relation_error(a, b, goal, eqs, eqs, pt, ot)) == _bits(G.relation_error(a, b, goal, eqs, eqs, pt, ot))
        assert _bits([F.orientation_error(a[1], goal[1], eqs)]) == _bits([G.orientation_error(a[1], goal[1], eqs)])


def test_batched_path_on_a_part_with_72_yaw_equivalences():
    if not F.self_check():
        pytest.skip("batched results differ from the reference on this CPU (geometry_fast falls back to it)")
    a = (th.tensor([0.1, 0.2, 0.3]), G.axis_angle_quat([0.0, 0.0, 1.0], 0.3))
    b = (th.tensor([0.1, 0.25, 0.36]), G.axis_angle_quat([0.3, 0.1, 1.0], 1.1))
    # the goal reached exactly at an equivalent yaw, and missed by a few millimetres / degrees
    for goal in (
        G.relative_pose(a, (b[0], G.quat_multiply(b[1], YAW72[10]))),
        G.relative_pose(a, (b[0] + th.tensor([0.004, 0.0, -0.002]), G.quat_multiply(b[1], G.axis_angle_quat([1.0, 0.0, 0.0], 0.05)))),
    ):
        got = F._relation_error_batched(a, b, goal, YAW72, YAW72, 0.02, 0.2618)
        assert got is not None
        assert _bits(got) == _bits(G.relation_error(a, b, goal, YAW72, YAW72, 0.02, 0.2618))
        got_o = F._orientation_error_batched(a[1], goal[1], YAW72)
        assert _bits([got_o]) == _bits([G.orientation_error(a[1], goal[1], YAW72)])


def test_inputs_other_than_float32_take_the_reference():
    p64, q64 = th.zeros(3, dtype=th.float64), G.IDENTITY_QUAT.double()
    assert F._relation_error_batched((p64, q64), (p64, q64), (p64, q64), [], [], None, None) is None
    assert F._orientation_error_batched(q64, G.IDENTITY_QUAT, []) is None


def test_check_mode_returns_the_reference_and_counts_calls(monkeypatch):
    monkeypatch.setattr(F, "CHECK", True)
    monkeypatch.setitem(F._COUNTS, "calls", 0)
    monkeypatch.setitem(F._COUNTS, "mismatches", 0)
    for a, b, goal, eqs, (pt, ot) in _cases(5, seed=1):
        assert _bits(F.relation_error(a, b, goal, eqs, eqs, pt, ot)) == _bits(G.relation_error(a, b, goal, eqs, eqs, pt, ot))
        assert _bits([F.orientation_error(a[1], goal[1], eqs)]) == _bits([G.orientation_error(a[1], goal[1], eqs)])
    assert F._COUNTS["calls"] == 10
    assert F._COUNTS["mismatches"] == 0


def test_non_finite_errors_take_the_reference():
    a = (th.tensor([float("nan"), 0.0, 0.0]), G.IDENTITY_QUAT.clone())
    b = (th.tensor([0.1, 0.0, 0.0]), G.axis_angle_quat([0.0, 0.0, 1.0], 0.2))
    goal = (th.tensor([0.1, 0.0, 0.0]), G.IDENTITY_QUAT.clone())
    assert _bits(F.relation_error(a, b, goal, YAW4, YAW4, 0.02, 0.26)) == _bits(G.relation_error(a, b, goal, YAW4, YAW4, 0.02, 0.26))
