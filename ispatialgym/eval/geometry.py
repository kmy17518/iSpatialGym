"""Pose utilities of the evaluator (torch only).

Quaternions are ``[x, y, z, w]`` (OmniGibson convention), positions in meters, angles in radians. ``quat_multiply(q1, q0)``
is ``q1 * q0`` (apply ``q0`` first), and a pose ``(p, q)`` maps a point ``x`` of its frame to ``R(q) x + p``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import torch as th

Pose = tuple[th.Tensor, th.Tensor]

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


def quat_angle(q1: th.Tensor, q2: th.Tensor) -> float:
    """Angular distance in ``[0, pi]`` between two orientations."""
    dot = th.abs(th.sum(quat_normalize(q1) * quat_normalize(q2)))
    return float(2.0 * th.acos(th.clamp(dot, -1.0, 1.0)))


def compose_pose(pose_a: Pose, pose_b: Pose) -> Pose:
    """``T_a * T_b`` (``b`` expressed in ``a``'s frame -> ``a``'s parent frame)."""
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


def orientation_error(current: th.Tensor, goal: th.Tensor, equivalences: Iterable[th.Tensor] | None = None) -> float:
    """Angle between ``current`` and ``goal`` over ``current``'s visual equivalents ``current * e`` (identity included)."""
    best = quat_angle(current, goal)
    for eq in equivalences or ():
        best = min(best, quat_angle(quat_multiply(current, eq), goal))
    return best


def relation_error(
    pose_from: Pose,
    pose_to: Pose,
    goal_relative: Pose,
    equivalences_from: Sequence[th.Tensor] | None = None,
    equivalences_to: Sequence[th.Tensor] | None = None,
    position_tolerance: float | None = None,
    orientation_tolerance: float | None = None,
) -> tuple[float, float]:
    """``(position, orientation)`` error of the relative pose ``T_from^-1 * T_to`` against ``goal_relative``.

    Both ends may be replaced by a symmetric equivalent (``from * E1``, ``to * E2``), which turns the relative pose into
    ``E1^-1 * T_rel * E2``. The best pair is returned: within both tolerances first, then the smallest normalized error.
    """
    eqs_from = [IDENTITY_QUAT] + list(equivalences_from or ())
    eqs_to = [IDENTITY_QUAT] + list(equivalences_to or ())
    p_rel, q_rel = relative_pose(pose_from, pose_to)
    p_goal, q_goal = goal_relative

    best = None
    for e1 in eqs_from:
        e1_inv = quat_inverse(e1)
        pos_err = float(th.norm(quat_apply(e1_inv, p_rel) - p_goal))
        q_left = quat_multiply(e1_inv, q_rel)
        for e2 in eqs_to:
            ori_err = quat_angle(quat_multiply(q_left, e2), q_goal)
            key = _error_key(pos_err, ori_err, position_tolerance, orientation_tolerance)
            if best is None or key < best[0]:
                best = (key, pos_err, ori_err)
    return best[1], best[2]


def _error_key(pos_err, ori_err, pos_tol, ori_tol):
    within = (pos_tol is None or pos_err <= pos_tol) and (ori_tol is None or ori_err <= ori_tol)
    normalized = max(pos_err / pos_tol, ori_err / ori_tol) if pos_tol and ori_tol else pos_err + ori_err
    return (0 if within else 1, normalized, pos_err, ori_err)


def assign_by_slot_position(
    member_positions: dict[str, th.Tensor], slot_positions: dict[str, th.Tensor]
) -> dict[str, str]:
    """One-to-one assignment of members to slots minimizing the total position error.

    Deterministic: members and slots are sorted and ties go to the lexicographically first permutation. Exhaustive up
    to 7 members, Hungarian (scipy) beyond.
    """
    members, slots = sorted(member_positions), sorted(slot_positions)
    if len(members) != len(slots):
        raise ValueError(f"Need exactly one slot per member; got {len(members)} members and {len(slots)} slots")
    cost = th.tensor(
        [[float(th.norm(member_positions[m] - slot_positions[s])) for s in slots] for m in members], dtype=th.float64
    )
    if len(members) <= 7:
        return assign_by_cost_fn(members, slots, lambda perm: float(sum(cost[i, j] for i, j in enumerate(perm))))
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost.numpy())
    return {members[i]: slots[j] for i, j in zip(rows, cols, strict=True)}


def assign_by_cost_fn(members: Sequence[str], slots: Sequence[str], perm_cost) -> dict[str, str]:
    """Exhaustive minimum-cost assignment for a non-additive permutation cost.

    ``perm_cost(perm)`` gets a tuple assigning member ``i`` (sorted order) to ``slots[perm[i]]`` (sorted order) and
    returns a comparable cost; the lexicographically first permutation wins ties.
    """
    members: list[str] = sorted(members)
    slots: list[str] = sorted(slots)
    if len(members) != len(slots):
        raise ValueError(f"Need exactly one slot per member; got {len(members)} members and {len(slots)} slots")
    if len(members) > 8:
        raise ValueError("Exhaustive assignment supports at most 8 members")
    best_perm, best_cost = None, None
    for perm in itertools.permutations(range(len(slots))):
        c = perm_cost(perm)
        if best_cost is None or c < best_cost:
            best_perm, best_cost = perm, c
    return {m: slots[j] for m, j in zip(members, best_perm, strict=True)}
