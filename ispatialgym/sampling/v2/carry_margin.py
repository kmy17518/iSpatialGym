"""i7-framing (owner decision 2026-11-08, item 5 - the middle option for the lifted-carry framing check): the NOMINAL protect margin of
the straight carry, as the oracle's plan check judges it (``oracle.protection.check_trajectory``): the moving geometry's clearance to
each protected object's base-aligned box along the leg, the threshold ``min(INFLATION_M, start - ENDPOINT_SLACK_M, end -
ENDPOINT_SLACK_M)`` over the leg's interior; ``margin = interior minimum - threshold`` (< 0: the oracle's check fires and the carry goes
via above).

The oracle uses cuRobo's sphere model of the moving arm and the carried object along the TRACKED configurations. The sampler and the
offline replay (``$OFH/framing_20261108/margin_replay.py``) use the same cheap proxy, so the band edges set on the replay apply to the
sampler unchanged: the carried object's base-aligned collision box posed by the grasp's ``eef_in_object`` (the hand box optional, off:
every carry firing in the campaign logs was the attached object's, 42 / 42), sampled on its surface (corners, edge midpoints, face
centres), the eef moving straight between the chain's poses (orientation SLERPed), every :data:`STEP_M`; every clearance less
:data:`SPHERE_OVERHANG_M` (cuRobo's sphere fit of the attached object overhangs its mesh - calibrated on the replay: the offset that
separates the fired carries best, AUC 0.893 at 9 mm vs 0.825 without).

Pure numpy; quaternions x y z w (OmniGibson's order)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: oracle.protection.INFLATION_M / ENDPOINT_SLACK_M (a test keeps them equal)
INFLATION_M = 0.02
ENDPOINT_SLACK_M = 0.002
#: how far cuRobo's sphere fit of the attached object reaches past its collision box (m; the replay's best separation, 1,423 demos)
SPHERE_OVERHANG_M = 0.009
#: the sampling step along the leg (m)
STEP_M = 0.01
#: the hand box in the eef frame (centre, half extents): IKChecker.eef_in_frame's +-6 cm across, -2..+6 cm along the approach (+z)
HAND_BOX = (np.array([0.0, 0.0, 0.02]), np.array([0.06, 0.06, 0.04]))

Box = Tuple[np.ndarray, np.ndarray, np.ndarray]  # world centre (3,), rotation (3, 3) whose columns are the box axes, half extents (3,)


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0, q1, t: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    s0, s1 = math.sin((1 - t) * th0) / math.sin(th0), math.sin(t * th0) / math.sin(th0)
    return s0 * q0 + s1 * q1


def _as_pose(p) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(p, dict):
        return np.asarray(p["position"], float), np.asarray(p["orientation"], float)
    return np.asarray([float(v) for v in p[0]], float), np.asarray([float(v) for v in p[1]], float)


def world_box(pose, local_box) -> Box:
    """A box given in an object's frame (centre, half extents) at the object's world ``pose``."""
    p, q = _as_pose(pose)
    R = quat_to_mat(q)
    c, half = np.asarray(local_box[0], float), np.asarray(local_box[1], float)
    return p + R @ c, R, half


def box_points(center, half) -> np.ndarray:
    """Points on a box surface (local frame): 8 corners, 12 edge midpoints, 6 face centres."""
    pts = [np.array([i, j, k], float) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    return np.asarray(center, float) + np.stack(pts) * np.asarray(half, float)


def point_box_distance(points: np.ndarray, box: Box) -> np.ndarray:
    """Signed distance of ``points`` (N, 3) world to ``box`` (negative inside), as oracle.protection.sphere_box_clearance with r = 0."""
    c, R, half = box
    p = (np.asarray(points, float) - c) @ R
    q = np.abs(p) - half
    outside = np.linalg.norm(np.clip(q, 0.0, None), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def leg_samples(chain: Sequence[Any], step: float = STEP_M) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The eef poses along the straight legs through ``chain`` (start and end included), every ``step`` (>= 2 per segment)."""
    poses = [_as_pose(p) for p in chain]
    out = [poses[0]]
    for (a, qa), (b, qb) in zip(poses[:-1], poses[1:]):
        n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append((a + t * (b - a), _slerp(qa, qb, t)))
    return out


def carry_protect_margin(chain: Sequence[Any], eef_in_object, object_box_local, protected: Sequence[Box], step: float = STEP_M,
                         inflation: float = INFLATION_M, slack: float = ENDPOINT_SLACK_M, hand_box=None, overhang: float = SPHERE_OVERHANG_M) -> Dict[str, Any]:
    """The nominal protect margin of the straight carry through ``chain`` (eef poses: lift end, [recorded vias], pre-place), the carried
    object attached by ``eef_in_object`` (T_obj->eef: eef = obj o eio) with its base-aligned box ``object_box_local`` (centre, half
    extents in the object frame), against the ``protected`` world boxes. ``hand_box`` (eef frame; e.g. :data:`HAND_BOX`): the hand too -
    off by default: every carry firing of the campaign logs was the attached object's (42 / 42), and the coarse hand box reaches into
    containers' boxes at the pre-place (frying pan) where the real fingers do not. ``{"start_m", "end_m", "interior_min_m", "threshold_m",
    "margin_m", "n"}`` (margin = +inf without a protected box: nothing to fire on)."""
    if not protected:
        return {"start_m": None, "end_m": None, "interior_min_m": None, "threshold_m": None, "margin_m": float("inf"), "n": 0}
    ep, eq = _as_pose(eef_in_object)
    Re = quat_to_mat(eq)
    # obj = eef o inv(eio): R_obj = R_eef Re^T, p_obj = p_eef - R_obj ep
    obj_pts = box_points(*object_box_local)
    hand_pts = box_points(*hand_box) if hand_box is not None else np.zeros((0, 3))
    clear = []
    for p, q in leg_samples(chain, step):
        R = quat_to_mat(q)
        Ro = R @ Re.T
        po = p - Ro @ ep
        pts = np.concatenate([obj_pts @ Ro.T + po, hand_pts @ R.T + p])
        clear.append(min(float(point_box_distance(pts, b).min()) for b in protected) - float(overhang))
    c = np.asarray(clear)
    thr = min(float(inflation), float(c[0]) - slack, float(c[-1]) - slack)
    interior = float(c[1:-1].min()) if len(c) > 2 else float("inf")
    return {"start_m": round(float(c[0]), 4), "end_m": round(float(c[-1]), 4), "interior_min_m": round(interior, 4), "threshold_m": round(thr, 4),
            "margin_m": round(interior - thr, 4), "n": int(len(c))}
