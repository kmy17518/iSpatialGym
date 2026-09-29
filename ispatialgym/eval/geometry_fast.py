"""Batched ``relation_error`` / ``orientation_error``: the float32 operations of :mod:`ispatialgym.eval.geometry` over
all symmetry equivalences at once, returning the same floats bit for bit (opt-in: ``ISPATIALGYM_FAST_GEOMETRY=1``,
read by :mod:`ispatialgym.eval.group_entity`; ``=check`` computes both on every call, returns the reference's and
prints ``GEOMETRY_FAST_CHECK`` lines with the call / mismatch counts).

``geometry.relation_error`` visits every (from, to) pair of symmetry equivalences in Python, one tiny torch op per
scalar: 73 x 73 = 5,329 pairs for a part with 72 yaw equivalences, ~0.4 s per call, and an attachment task's goal
check calls it on every simulator step. Here every quantity is one op over the batch. The ops are elementwise (the
same IEEE float32 result whatever the batch shape), per-row reductions over a last axis of 3 or 4 values, a 3 x 3
matrix-vector product per row, and ``acos``; torch computes the latter three batched exactly as for one vector on
the CPUs checked, and :func:`self_check` compares both implementations on random inputs at first use and keeps the
reference for the whole process if any bit differs. Inputs that are not all float32 take the reference path (a
0-dim float64 operand promotes in the reference but not against a batch), and so does a selection over non-finite
errors (the reference's tuple comparisons treat NaN in their own way).
"""
from __future__ import annotations

import logging
import math
import os
import struct
from typing import Iterable, Optional, Sequence, Tuple

import torch as th

from ispatialgym.eval import geometry as G

log = logging.getLogger(__name__)

_STATE = {"ok": None}
CHECK = os.environ.get("ISPATIALGYM_FAST_GEOMETRY") == "check"
_COUNTS = {"calls": 0, "mismatches": 0}


def _check(name: str, ref, got) -> None:
    if got is None:
        return
    _COUNTS["calls"] += 1
    ref_t, got_t = (ref if isinstance(ref, tuple) else (ref,)), (got if isinstance(got, tuple) else (got,))
    if [struct.pack("<d", float(v)) for v in ref_t] != [struct.pack("<d", float(v)) for v in got_t]:
        _COUNTS["mismatches"] += 1
        if _COUNTS["mismatches"] <= 20:
            print(f"GEOMETRY_FAST_CHECK mismatch {name}: reference {ref!r} batched {got!r}", flush=True)
    if _COUNTS["calls"] % 1000 == 0:
        print(f"GEOMETRY_FAST_CHECK calls {_COUNTS['calls']} mismatches {_COUNTS['mismatches']}", flush=True)


def _stack(qs: Sequence[th.Tensor]) -> th.Tensor:
    return th.stack([q.reshape(4) for q in qs])


def _quat_normalize(q: th.Tensor) -> th.Tensor:
    return q / th.linalg.vector_norm(q, dim=-1, keepdim=True)


def _quat_inverse(q: th.Tensor) -> th.Tensor:
    conj = th.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)
    return conj / th.sum(q * q, dim=-1, keepdim=True)


def _quat_multiply(q1: th.Tensor, q0: th.Tensor) -> th.Tensor:
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=-1,
    )


def _quat2mat(q: th.Tensor) -> th.Tensor:
    n = _quat_normalize(q)
    x, y, z, w = n[..., 0], n[..., 1], n[..., 2], n[..., 3]
    return th.stack(
        [
            th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            th.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            th.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _quat_angle(q1: th.Tensor, q2_normalized: th.Tensor) -> th.Tensor:
    dot = th.abs(th.sum(_quat_normalize(q1) * q2_normalized, dim=-1))
    return 2.0 * th.acos(th.clamp(dot, -1.0, 1.0))


def _all_float32(*tensors) -> bool:
    return all(isinstance(t, th.Tensor) and t.dtype == th.float32 for t in tensors)


def _select(pos: th.Tensor, ori: th.Tensor, pos_tol, ori_tol) -> Optional[Tuple[float, float]]:
    """The reference's lexicographic minimum of ``_error_key`` over (e1, e2) in loop order; None for non-finite input."""
    P = pos.double()[:, None].expand_as(ori)
    O = ori.double()
    if not (bool(th.isfinite(P).all()) and bool(th.isfinite(O).all())):
        return None
    within = th.ones_like(O, dtype=th.bool)
    if pos_tol is not None:
        within &= P <= pos_tol
    if ori_tol is not None:
        within &= O <= ori_tol
    normalized = th.maximum(P / pos_tol, O / ori_tol) if (pos_tol and ori_tol) else P + O
    mask = within if bool(within.any()) else th.ones_like(within)
    for key in (normalized, P, O):
        mask = mask & (key == key[mask].min())
    i, j = divmod(int(th.nonzero(mask.flatten())[0]), ori.shape[1])
    return float(pos[i]), float(ori[i, j])


def _relation_error_batched(pose_from, pose_to, goal_relative, equivalences_from, equivalences_to, pos_tol, ori_tol):
    eqs_from = [G.IDENTITY_QUAT] + list(equivalences_from or ())
    eqs_to = [G.IDENTITY_QUAT] + list(equivalences_to or ())
    p_goal, q_goal = goal_relative
    if not _all_float32(pose_from[0], pose_from[1], pose_to[0], pose_to[1], p_goal, q_goal, *eqs_from, *eqs_to):
        return None
    p_rel, q_rel = G.relative_pose(pose_from, pose_to)
    e1_inv = _quat_inverse(_stack(eqs_from))
    pos = th.linalg.vector_norm(th.matmul(_quat2mat(e1_inv), p_rel) - p_goal, dim=-1)
    q_left = _quat_multiply(e1_inv, q_rel)
    ori = _quat_angle(_quat_multiply(q_left[:, None, :], _stack(eqs_to)[None, :, :]), G.quat_normalize(q_goal))
    return _select(pos, ori, pos_tol, ori_tol)


def _orientation_error_batched(current, goal, equivalences):
    eqs = list(equivalences or ())
    if not _all_float32(current, goal, *eqs):
        return None
    best = G.quat_angle(current, goal)
    if eqs:
        for v in _quat_angle(_quat_multiply(current, _stack(eqs)), G.quat_normalize(goal)).tolist():
            best = min(best, v)
    return best


def self_check(n_cases: int = 24, seed: int = 0) -> bool:
    """Both implementations on random float32 inputs (symmetry sets of 0-72 quaternions, with and without tolerances);
    True when every result is bitwise equal."""
    gen = th.Generator().manual_seed(seed)

    def quat():
        return G.quat_normalize(th.randn(4, generator=gen).float())

    def pose():
        return (th.randn(3, generator=gen).float() * 0.3, quat())

    def yaw_steps(k):
        return [G.axis_angle_quat([0.0, 0.0, 1.0], 2 * math.pi * i / (k + 1)) for i in range(1, k + 1)]

    tolerances = [(None, None), (0.02, 0.26), (0.05, None), (0.0, 0.1), (1.0, 3.2)]
    for c in range(n_cases):
        k = (0, 1, 3, 23, 72)[c % 5]
        eqs = yaw_steps(k) if c % 2 else [quat() for _ in range(k)]
        a, b = pose(), pose()
        goal = G.relative_pose(a, b) if c % 3 == 0 else pose()
        if c % 3 == 0:
            goal = (goal[0] + th.randn(3, generator=gen).float() * 0.01, G.quat_normalize(goal[1] + th.randn(4, generator=gen).float() * 0.05))
        pt, ot = tolerances[c % len(tolerances)]
        ref = G.relation_error(a, b, goal, eqs, eqs, pt, ot)
        got = _relation_error_batched(a, b, goal, eqs, eqs, pt, ot)
        if got is None or got != ref:
            log.warning("geometry_fast self-check: relation_error %r != reference %r (case %d)", got, ref, c)
            return False
        ref_o = G.orientation_error(a[1], goal[1], eqs)
        got_o = _orientation_error_batched(a[1], goal[1], eqs)
        if got_o is None or got_o != ref_o:
            log.warning("geometry_fast self-check: orientation_error %r != reference %r (case %d)", got_o, ref_o, c)
            return False
    return True


def _enabled() -> bool:
    if _STATE["ok"] is None:
        _STATE["ok"] = self_check()
        if not _STATE["ok"]:
            log.warning("geometry_fast: batched results differ from the reference on this machine; using the reference")
    return _STATE["ok"]


def relation_error(
    pose_from: G.Pose,
    pose_to: G.Pose,
    goal_relative: G.Pose,
    equivalences_from: Optional[Sequence[th.Tensor]] = None,
    equivalences_to: Optional[Sequence[th.Tensor]] = None,
    position_tolerance: Optional[float] = None,
    orientation_tolerance: Optional[float] = None,
) -> Tuple[float, float]:
    """:func:`ispatialgym.eval.geometry.relation_error`, batched."""
    args = (pose_from, pose_to, goal_relative, equivalences_from, equivalences_to, position_tolerance, orientation_tolerance)
    if CHECK:
        ref = G.relation_error(*args)
        _check("relation_error", ref, _relation_error_batched(*args))
        return ref
    if _enabled():
        got = _relation_error_batched(*args)
        if got is not None:
            return got
    return G.relation_error(*args)


def orientation_error(current: th.Tensor, goal: th.Tensor, equivalences: Optional[Iterable[th.Tensor]] = None) -> float:
    """:func:`ispatialgym.eval.geometry.orientation_error`, batched."""
    if CHECK:
        eqs = list(equivalences or ())
        ref = G.orientation_error(current, goal, eqs)
        _check("orientation_error", ref, _orientation_error_batched(current, goal, eqs))
        return ref
    if _enabled():
        eqs = list(equivalences or ())
        got = _orientation_error_batched(current, goal, eqs)
        if got is not None:
            return got
        return G.orientation_error(current, goal, eqs)
    return G.orientation_error(current, goal, equivalences)
