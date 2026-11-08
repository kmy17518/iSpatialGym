"""i7-overpath 2026-11-08 (owner decision 3(a), inside_large_01): the overhang pick's RECORDED transit over the book.

The overhang pick (``programs._pick_actions``) pushes the book TOWARD the robot to B's support edge (the hand pushes from the far side
and ends beyond the book's centre), escapes up off it (``Retreat.escape_up_m``) and grasps the overhanging edge from a pre-grasp in
front of it (~0.1 m nearer the body than the grasp). The hand has to cross back over the protected book. Until now the oracle planned
that transit live (``execute._grasp_transit``): the first plan went into the book, the via-above re-plan found no plan or still passed the
book at 9 mm (i7_po 2 / 4 demos ``plan_failed`` at the grasp). With the switch :data:`ENV` the SAMPLER plans the transit and records it
(``grasps[k].overpath``, :func:`record`); the program's Grasp carries it (``Grasp.overpath``); the oracle follows the recorded joint
waypoints (``Motions.recorded_joint_path`` -> ``execute_joint_trajectory``, the existing joint tracking) instead of planning, and the
program check replays it (``program_check.check_overpath``). The path, in the pushing arm at the grasp stance (after the drive to the
"pick" back-off when there is one):

1. from where the escape leaves the hand, rise to the book's top + a clearance (:data:`CLEARANCES_M`, the first that passes);
2. move across at that height to above the pre-grasp (the hand turns to the pre-grasp's orientation on the way);
3. descend to the pre-grasp in front of the overhanging edge.

The sampler tracks those Cartesian legs by differential IK from the start (no branch flip: ``track_chain``'s 0.2 rad step bound),
subsamples the tracked configurations every :data:`WAYPOINT_SPACING_RAD` and checks the path the oracle runs - the piecewise-linear joint
path through them, :func:`densify` - whole-body against the desk and the scene and with the moving arm's spheres at least
:data:`BOOK_MARGIN_M` from the book's box. Pure helpers here (no simulator), shared by the sampler, the program check and the oracle.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch as th

#: the sampler switch (default off): plan and record the transit; the oracle follows a recorded one whenever the sidecar has it
ENV = "ISPATIALGYM_OVERHANG_OVERPATH"
#: ``grasps[k]`` key of the record
RECORD_KEY = "overpath"
SCHEMA = "isg-overpath/1"
#: the eef's height above the book's top on the across leg, tried in order (owner: start at 0.08-0.10 m; the hand's lowest sphere hangs
#: ~4-6 cm below the eef point at the push / pre-grasp orientations, so 0.08 leaves the fingers 2-4 cm over the book; 0.12 for headroom)
CLEARANCES_M = (0.08, 0.10, 0.12)
#: the sampler's clearance of the moving arm's spheres to the book's box along the executed path (the oracle's protected-object check
#: is ``protection.INFLATION_M`` 0.02 m; 1 cm more for the book's post-push drift)
BOOK_MARGIN_M = 0.03
#: largest arm-joint change between two recorded waypoints (the executed path interpolates linearly between them)
WAYPOINT_SPACING_RAD = 0.05
#: largest per-control-step arm-joint change of the executed path (the straight ramp's ``STRAIGHT_ARM_STEP_RAD``)
STEP_RAD = 0.02
#: the oracle / program check follow the record only when the arm starts within this of its first configuration (largest arm joint)
START_GAP_MAX_RAD = 0.35
#: a rise shorter than this is skipped (the start is already above the across height)
MIN_LEG_M = 0.01


def on() -> bool:
    """The sampler switch :data:`ENV` ("1" on; default off)."""
    return os.environ.get(ENV, "0").strip().lower() in ("1", "on", "true", "yes")


def leg_poses(start_pos, start_quat, pre_pos, pre_quat, z_across: float) -> List[Tuple[th.Tensor, th.Tensor]]:
    """The eef poses of the transit: the start, the rise to ``z_across`` above the start (skipped when the start is within
    :data:`MIN_LEG_M` of it or higher), the point above the pre-grasp at ``z_across`` (the pre-grasp's orientation), the pre-grasp.
    Positions (3,) / quaternions xyzw (4,) as float tensors."""
    p0 = th.as_tensor(start_pos, dtype=th.float32).reshape(3)
    q0 = th.as_tensor(start_quat, dtype=th.float32).reshape(4)
    p1 = th.as_tensor(pre_pos, dtype=th.float32).reshape(3)
    q1 = th.as_tensor(pre_quat, dtype=th.float32).reshape(4)
    z = max(float(z_across), float(p1[2]))
    out = [(p0, q0)]
    if z - float(p0[2]) > MIN_LEG_M:
        out.append((th.tensor([float(p0[0]), float(p0[1]), z]), q0))
    over = th.tensor([float(p1[0]), float(p1[1]), max(z, float(p0[2]))])
    if float(th.linalg.norm(over - out[-1][0])) > MIN_LEG_M:
        out.append((over, q1))
    if float(th.linalg.norm(p1 - out[-1][0])) > 1e-4:
        out.append((p1, q1))
    return out


def subsample(traj: th.Tensor, spacing: float = WAYPOINT_SPACING_RAD) -> List[int]:
    """Indices of ``traj`` ((T, d) joint rows) kept as recorded waypoints: the first, every row whose largest joint change from the last
    kept one reaches ``spacing``, the last."""
    t = th.as_tensor(traj).float()
    if t.ndim != 2 or len(t) == 0:
        return []
    keep = [0]
    for k in range(1, len(t) - 1):
        if float((t[k] - t[keep[-1]]).abs().max()) >= float(spacing):
            keep.append(k)
    if len(t) > 1:
        keep.append(len(t) - 1)
    return keep


def densify(q_start, waypoints: Sequence[Any], step: float = STEP_RAD) -> th.Tensor:
    """The path the oracle runs: the piecewise-linear joint path ``q_start`` -> ``waypoints[0]`` -> ... -> ``waypoints[-1]`` (rows of the
    same length: 7 arm joints, or full configurations), time-parameterised by ONE cosine profile over its cumulative largest-joint length
    (zero speed at both ends, no stop at a waypoint) with the steepest step <= ``step`` (pi/2 x the mean). Returns ``(n + 1, d)``; the
    first row is ``q_start``, the last the last waypoint."""
    pts = [th.as_tensor(q_start).reshape(-1).float().cpu()] + [th.as_tensor(w).reshape(-1).float().cpu() for w in waypoints]
    P = th.stack(pts)
    seg = (P[1:] - P[:-1]).abs().max(dim=1).values if len(P) > 1 else th.zeros(0)
    total = float(seg.sum()) if len(seg) else 0.0
    if total <= 1e-9:
        return P[-1:].clone()
    cum = th.cat([th.zeros(1), th.cumsum(seg, dim=0)])
    n = max(2, int(math.ceil((math.pi / 2.0) * total / float(step))))
    s = total * 0.5 * (1.0 - th.cos(th.linspace(0.0, math.pi, n + 1)))
    out = []
    for v in s.tolist():
        j = int(th.searchsorted(cum, th.tensor(v), right=True)) - 1
        j = min(max(j, 0), len(seg) - 1)
        L = float(seg[j])
        f = 0.0 if L <= 1e-12 else min(max((v - float(cum[j])) / L, 0.0), 1.0)
        out.append(P[j] + f * (P[j + 1] - P[j]))
    traj = th.stack(out)
    traj[0], traj[-1] = P[0], P[-1]
    return traj


def box_of(pose, hull_local) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """The object's box at ``pose`` (``protection.object_box``'s convention: centre (3,), rotation whose columns are the box axes (3, 3),
    half extents (3,)) - its local hull's bounding box in the object frame, posed."""
    from ispatialgym.robot import conventions as C

    h = th.as_tensor(hull_local).float().reshape(-1, 3)
    lo, hi = h.min(dim=0).values, h.max(dim=0).values
    R = C.quat_to_mat(th.as_tensor(pose[1]).float().reshape(4)).float()
    c = th.as_tensor(pose[0]).float().reshape(3) + R @ (0.5 * (lo + hi))
    return c, R, 0.5 * (hi - lo)


def record(arm: str, stance: str, start: str, qs: Sequence[Any], poses: Sequence[Any], **facts: Any) -> Dict[str, Any]:
    """``grasps[k].overpath``: ``q`` the recorded arm joints (the first = where the sampler planned the path from), ``eef`` the eef
    positions there (diagnostic), the facts (clearance, book top, minimum book clearance, largest tracked step, ...)."""
    return {"schema": SCHEMA, "arm": str(arm), "stance": str(stance), "start": str(start),
            "q": [[round(float(v), 5) for v in th.as_tensor(q).reshape(-1).tolist()] for q in qs],
            "eef": [[round(float(v), 4) for v in th.as_tensor(p).reshape(-1)[:3].tolist()] for p in poses],
            **{k: v for k, v in facts.items()}}


def parse(rec: Any, where: str = "overpath") -> List[th.Tensor]:
    """The recorded arm joints of ``rec`` (each (7,)); ``ValueError`` when malformed."""
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        raise ValueError(f"{where}: not an {SCHEMA} record")
    qs = rec.get("q")
    if not isinstance(qs, list) or len(qs) < 2 or any(not isinstance(q, (list, tuple)) or len(q) != 7 for q in qs):
        raise ValueError(f"{where}.q: expected >= 2 lists of 7 arm joints")
    return [th.tensor([float(v) for v in q], dtype=th.float32) for q in qs]
