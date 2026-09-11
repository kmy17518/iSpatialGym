"""Small, dependency-light pose utilities shared by the iSpatialGym evaluator.

All quaternions are ``[x, y, z, w]`` (OmniGibson convention); positions are meters; angles are radians. Rotation
composition follows :mod:`omnigibson.utils.transform_utils`: ``quat_multiply(q1, q0)`` is ``q1 * q0`` (apply
``q0`` first, then ``q1``), and a pose ``(p, q)`` maps a point ``x`` in its own frame to ``R(q) x + p``.

Only ``torch`` is required so the unit tests can exercise this module without importing OmniGibson.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

Pose = Tuple[th.Tensor, th.Tensor]

IDENTITY_QUAT = th.tensor([0.0, 0.0, 0.0, 1.0])


def as_tensor(value, dtype=th.float32) -> th.Tensor:
    if isinstance(value, th.Tensor):
        return value.detach().to(dtype=dtype)
    return th.tensor(value, dtype=dtype)


def quat_normalize(q: th.Tensor) -> th.Tensor:
    return q / th.norm(q)


def quat_conjugate(q: th.Tensor) -> th.Tensor:
    return th.stack([-q[0], -q[1], -q[2], q[3]])


def quat_inverse(q: th.Tensor) -> th.Tensor:
    return quat_conjugate(q) / th.sum(q * q)


def quat_multiply(q1: th.Tensor, q0: th.Tensor) -> th.Tensor:
    """``q1 * q0`` (Hamilton product, xyzw)."""
    x0, y0, z0, w0 = q0[0], q0[1], q0[2], q0[3]
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ]
    )


def quat2mat(q: th.Tensor) -> th.Tensor:
    x, y, z, w = quat_normalize(q)
    return th.stack(
        [
            th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            th.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            th.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def quat_apply(q: th.Tensor, v: th.Tensor) -> th.Tensor:
    return quat2mat(q) @ v


def axis_angle_quat(axis: Sequence[float], angle: float) -> th.Tensor:
    axis = as_tensor(axis)
    axis = axis / th.norm(axis)
    half = angle / 2.0
    return th.cat([axis * math.sin(half), th.tensor([math.cos(half)])])


def quat_angle(q1: th.Tensor, q2: th.Tensor) -> float:
    """Angular distance in ``[0, pi]`` between two orientations."""
    dot = th.abs(th.sum(quat_normalize(q1) * quat_normalize(q2)))
    return float(2.0 * th.acos(th.clamp(dot, -1.0, 1.0)))


def compose_pose(pose_a: Pose, pose_b: Pose) -> Pose:
    """``T_a * T_b`` (b expressed in a's frame -> world)."""
    pa, qa = pose_a
    pb, qb = pose_b
    return quat_apply(qa, pb) + pa, quat_normalize(quat_multiply(qa, qb))


def invert_pose(pose: Pose) -> Pose:
    p, q = pose
    q_inv = quat_inverse(q)
    return -quat_apply(q_inv, p), q_inv


def relative_pose(pose_from: Pose, pose_to: Pose) -> Pose:
    """Pose of ``to`` expressed in the frame of ``from``: ``T_from^-1 * T_to``."""
    return compose_pose(invert_pose(pose_from), pose_to)


def orientation_error(current: th.Tensor, goal: th.Tensor, equivalences: Optional[Iterable[th.Tensor]] = None) -> float:
    """Angle between ``current`` and ``goal`` after snapping ``current`` to its closest visual equivalent.

    ``equivalences`` are object-frame symmetry quaternions ``e`` (``current * e`` looks identical). The identity is
    always considered.
    """
    best = quat_angle(current, goal)
    for eq in equivalences or ():
        best = min(best, quat_angle(quat_multiply(current, eq), goal))
    return best


def relation_error(
    pose_from: Pose,
    pose_to: Pose,
    goal_relative: Pose,
    equivalences_from: Optional[Sequence[th.Tensor]] = None,
    equivalences_to: Optional[Sequence[th.Tensor]] = None,
    position_tolerance: Optional[float] = None,
    orientation_tolerance: Optional[float] = None,
) -> Tuple[float, float]:
    """Error of the measured relative pose ``T_from^-1 * T_to`` against ``goal_relative``.

    Symmetries are accounted for on both ends: if ``from`` can be replaced by ``from * E1`` and ``to`` by
    ``to * E2`` without changing appearance, the relative pose becomes ``E1^-1 * T_rel * E2`` (which also rotates
    the relative *position*). The returned pair is the best over all ``(E1, E2)`` combinations, preferring pairs that
    satisfy both tolerances, then the smallest normalized error.
    """
    eqs_from = [IDENTITY_QUAT] + list(equivalences_from or ())
    eqs_to = [IDENTITY_QUAT] + list(equivalences_to or ())
    p_rel, q_rel = relative_pose(pose_from, pose_to)
    p_goal, q_goal = goal_relative

    best = None
    for e1 in eqs_from:
        e1_inv = quat_inverse(e1)
        p_cand = quat_apply(e1_inv, p_rel)
        pos_err = float(th.norm(p_cand - p_goal))
        q_left = quat_multiply(e1_inv, q_rel)
        for e2 in eqs_to:
            ori_err = quat_angle(quat_multiply(q_left, e2), q_goal)
            key = _error_key(pos_err, ori_err, position_tolerance, orientation_tolerance)
            if best is None or key < best[0]:
                best = (key, pos_err, ori_err)
    return best[1], best[2]


def _error_key(pos_err, ori_err, pos_tol, ori_tol):
    within = (pos_tol is None or pos_err <= pos_tol) and (ori_tol is None or ori_err <= ori_tol)
    if pos_tol and ori_tol:
        normalized = max(pos_err / pos_tol, ori_err / ori_tol)
    else:
        normalized = pos_err + ori_err
    return (0 if within else 1, normalized, pos_err, ori_err)


# ----------------------------------------------------------------------------------------------------------------
# Member-to-slot assignment
# ----------------------------------------------------------------------------------------------------------------


def assign_by_slot_position(
    member_positions: Dict[str, th.Tensor], slot_positions: Dict[str, th.Tensor]
) -> Dict[str, str]:
    """Minimum total position-error one-to-one assignment of members to slots.

    Deterministic: members are sorted by name and slots by id; among equal-cost assignments the lexicographically
    first permutation wins. Exhaustive for <= 7 members, Hungarian (scipy) beyond that.
    """
    members = sorted(member_positions)
    slots = sorted(slot_positions)
    if len(members) != len(slots):
        raise ValueError(f"Need exactly one slot per member; got {len(members)} members and {len(slots)} slots")
    cost = th.tensor(
        [[float(th.norm(member_positions[m] - slot_positions[s])) for s in slots] for m in members], dtype=th.float64
    )
    return _assign(members, slots, lambda perm: float(sum(cost[i, j] for i, j in enumerate(perm))), cost)


def assign_by_cost_fn(members: Sequence[str], slots: Sequence[str], perm_cost) -> Dict[str, str]:
    """Exhaustive minimum-cost assignment for arbitrary (non-additive) permutation costs.

    ``perm_cost(perm)`` receives a tuple ``perm`` where member ``i`` (in sorted order) is assigned slot
    ``slots[perm[i]]`` and returns a comparable cost (float or tuple). Lexicographically-first permutation wins ties.
    """
    members = sorted(members)
    slots = sorted(slots)
    if len(members) != len(slots):
        raise ValueError(f"Need exactly one slot per member; got {len(members)} members and {len(slots)} slots")
    if len(members) > 8:
        raise ValueError("Exhaustive assignment supports at most 8 members")
    best_perm, best_cost = None, None
    for perm in itertools.permutations(range(len(slots))):
        c = perm_cost(perm)
        if best_cost is None or c < best_cost:
            best_perm, best_cost = perm, c
    return {m: slots[j] for m, j in zip(members, best_perm)}


def _assign(members: List[str], slots: List[str], perm_cost, cost_matrix: th.Tensor) -> Dict[str, str]:
    n = len(members)
    if n <= 7:
        return assign_by_cost_fn(members, slots, perm_cost)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("scipy is required to assign more than 7 members") from e
    rows, cols = linear_sum_assignment(cost_matrix.numpy())
    return {members[i]: slots[j] for i, j in zip(rows, cols)}
