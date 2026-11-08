"""i7-contact 2026-11-07: the touching push / brace POSE RULES as pure functions - the record builders of ``IORSSmall`` (``touching_push_candidates``,
``swept_push_candidates``, ``open_push_candidates``, ``touching_brace_candidates`` and their helpers) moved here so that the sampler and the
contact reach-map builder (:mod:`ispatialgym.sampling.contact_maps`, kinds ``touch_push`` / ``touch_brace``) produce the SAME poses from the
same generation knobs. Nothing here touches a simulator: every rule reads hull points (world xyz), the stance's ``forward`` / ``left`` axes
(any object with them - a :class:`~ispatialgym.sampling.v2.stances.Stance` or :data:`MAP_FRAME`) and a generation source ``gs`` (anything with
``gs.gen(key, default)`` - the sampler itself, or :class:`PoseGen` for the map builder; ``TT.phase2_gen`` also reads ``gs.subtask``).

``IORSSmall``'s methods of the same names delegate to these functions unchanged (their records are pinned by ``test_contact_poses``' golden
lists of 63925048); the module-level helpers (``HullSections``, ``face_run_at``, ``min_area_rect``, ...) and constants keep their
``iors_small`` names by re-export.

What the oracle EXECUTES of a recorded contact (``ispatialgym/oracle/execute.py``, read-only here): :func:`executed_push_pre` (a straight push's
pre-contact lowered to ``push_pre_dist_m`` 0.04 m before the init along the approach unless it is an arc or a pitched >= 30 deg blade class:
``push_lowered_pre``), :func:`contact_fingers` (the record's ``fingers_opening_m``, else closed: ``execute_push`` / ``execute_brace``).
"""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import contact_height_extra, requested_height_extra
from ispatialgym.sampling.v2.stances import Stance


#: Tip-to-tip spread of the CLOSED fingers (the finger joints sit at y = ±0.0135 in the gripper link; URDF ``gripper_finger_joint``):
#: a wide stop with tip spread ``s`` pre-shapes each finger to ``(s - CLOSED_TIP_SPREAD_M) / 2``.
CLOSED_TIP_SPREAD_M = 0.027
#: ``generation.brace_open_when_wide``: flat face kept between each open fingertip and the braced face's end, and the depth band (m)
#: within which the face's cut points count as its flat run (``wide_stop_spread``'s ``end_margin`` / ``face_depth_tol``)
BRACE_OPEN_END_MARGIN_M = 0.01
BRACE_OPEN_FACE_DEPTH_TOL_M = 0.01
#: Wrist turns of a blade push in the order to try (owner 2026-09-18 19:39): the wrist camera (``right_realsense_link``, three
#: r 1.8 cm spheres 4-7 cm to the side of the gripper at the palm-line height) lies on the eef's -y side; ``eef_quat(a, c·outward)``
#: puts eef +y = c·outward, so c = +1 hangs the camera over the pushed object's top and c = -1 turns it away from it. The camera
#: contacting the box top pushed the robot down against the table (demos 0918-190236 / 191811: robot and table shaking).
#: ``left_realsense_link`` sits at the same eef-frame position and every class derives its axes from the push direction, so the
#: left hand's push along -tangent (the touching shelf layout) keeps this order: camera away from the pushed object / up first.
PUSH_WRIST_TURNS = (-1.0, 1.0)
#: Hand geometry beyond the finger profile, sphere centres in the EEF frame (gripper-link spheres of
#: ``r1pro_description_curobo_default.yaml`` mapped by ``EEF_IN_GRIPPER`` (0, 0, -0.06; rpy 0, pi, 0): p_eef = (-x, y, -(z + 0.06)))
#: — what the IK collision check does not see against the pushed object (ignored at the contact pose, §3.6).
HAND_SPHERES_EEF: Dict[str, Tuple[Tuple[Tuple[float, float, float], ...], float]] = {
    "wrist camera": (((-0.056, -0.002, -0.076), (-0.056, -0.030, -0.076), (-0.056, -0.057, -0.076)), 0.018),
    "knuckle": (((-0.015, -0.042, -0.0675), (0.015, -0.042, -0.0675), (-0.015, 0.042, -0.0675), (0.015, 0.042, -0.0675)), 0.015),
    "gripper body": (((0.0, 0.0, -0.097),), 0.04),
}



#: §6.11 touching: the brace / push stop - the fingertip plane this far before the face (``generation.stop_clearance_m``)
STOP_CLEARANCE = 0.005
#: oracle-fix 2026-11-06 (owner on the touching#04 spray-bottle demo, ``generation.push_height_sweep``): the lowest contact height (m above
#: the target's bottom = the support top) of each push class at which the hand clears the support - the vertical blade's fingertips 5 mm
#: (the old blade's height), a leaning blade's fingertips ``blade_lean_tip_above_m`` (its leading edge dips below the tips), the horizontal
#: hand of the fingertip / pad pushes closing horizontally 5 cm (the gripper body, r 4 cm on the eef axis, 1 cm clear), the vertically
#: closing ones 6.2 cm (the lower knuckle 4.2 cm below the pads, r 1.5 cm, 5 mm clear); the pad pushes ``push_h`` / ``push_v`` (one turn
#: each: the wrist camera hangs 7.4 / 7.5 cm below the contact with its r 1.8 cm) 8 cm - below it the pre-pose's camera stands in the table
PUSH_SWEEP_MIN_HEIGHT_M = {"blade": 0.005, "tip_h": 0.05, "tip_v": 0.062, "pad": 0.08}
#: the push contact's window around the centre line (m, across it and in height): the outermost material within it is the face the
#: fingertips stop ``stop_clearance_m`` before (no part of them starts inside the target)
PUSH_SWEEP_WINDOW_M = 0.01
#: the footprint the push line goes through: the target's hull cut this far above its bottom (the part it stands on)
PUSH_SWEEP_BASE_CUT_M = 0.005
#: the raised blade (palm above the top) is the sweep's last resort for a target this tall over its smallest footprint side or less
#: (``generation.push_raised_max_aspect``; default: every target - the sweep tries every lower contact first): the spray bottle aliases set
#: 1.5 (it rocked 3.6 -> 20.3 deg pushed near its top); with 1.5 for every alias touching_02's cereal box lost its only feasible class
#: (fix1 2026-11-06: 1 instance in 146 attempts vs trainfast's 33 per instance, 9 goal-check contact_unreachable)
PUSH_RAISED_MAX_ASPECT = math.inf
#: oracle-fix 2026-11-06 (owner on touching#02's cereal box, ``generation.push_gripper_open``): the open push's finger joint opening (m per
#: finger, ``generation.push_open_opening_m``): the fingertip centres 7.7 cm apart - closing vertically at the knuckle's lowest eef height
#: (6.2 cm) the lower tip pushes 2.35 cm above the support, the upper one 10.05 cm (the x0.8 cereal box's middle: 10.9 cm), the lower
#: finger's base sphere 8.5 mm clear of the support
PUSH_OPEN_OPENING_M = 0.035
#: the fingertip centre's offset from the closing plane at the finger joint opening 0 (the tip sphere of ``tabletop.FINGER_SPHERES_EEF``:
#: y 3.5 mm, r 5 mm); open by ``q`` a tip's centre lies ``q`` + this from the eef axis
OPEN_TIP_CENTRE_M = 0.0035
#: the open fingers' lowest point above the support (m): the eef axis of an open push is raised until the lower finger clears it by this
PUSH_OPEN_SUPPORT_CLEARANCE_M = 0.005


# ------------------------------------------------------------------------------------------------ the generation source and the map frame


class PoseGen:
    """A generation source for the rules outside a sampler (the map builder): ``gen(key, default)`` over a dict of knobs, ``subtask`` for
    ``TT.phase2_gen``'s generation_defaults fallback (None: the dict and the call's default only)."""

    def __init__(self, knobs: Optional[Dict[str, Any]] = None, subtask: Optional[str] = None):
        self.knobs = dict(knobs or {})
        self.subtask = subtask

    def gen(self, key: str, default=None):
        v = self.knobs.get(key)
        return default if v is None else v


#: the robot frame of a reach map (base_link on the floor: +x ahead, +y the robot's left): the stance axes the rules read
MAP_FRAME = SimpleNamespace(forward=th.tensor([1.0, 0.0]), left=th.tensor([0.0, 1.0]), tangent=th.tensor([0.0, 1.0]))


# ------------------------------------------------------------------------------------------------ hull helpers (moved from iors_small)


def open_fingers_drop_below(eef_quat, opening: float) -> float:
    """How far the OPEN fingers (``tabletop.FINGER_SPHERES_EEF`` moved ``opening`` outward along the closing axis) reach below the eef origin
    at the eef orientation ``eef_quat`` (m; the hand above them is ``tabletop.hand_drop_below``)."""
    R = C.quat_to_mat(G.t(eef_quat).float())
    worst = -math.inf
    for (x, y, z), r in TT.FINGER_SPHERES_EEF:
        c = R @ th.tensor([x, y + math.copysign(float(opening), y), z], dtype=th.float32)
        worst = max(worst, -float(c[2]) + float(r))
    return worst


def push_face_point(hull_w: th.Tensor, direction) -> th.Tensor:
    """The contact of a push along ``direction`` (world, horizontal) on a hull given in world points: the face the pad meets (the
    hull's extreme against ``direction``) at mid height, on the line through the hull centre along ``direction``. Along +tangent
    this is the right-face point of ``touching_points``, along -tangent the left-face point."""
    d = th.tensor([float(direction[0]), float(direction[1])])
    d = d / th.linalg.norm(d)
    z = float(hull_w[:, 2].min()) + 0.5 * float(hull_w[:, 2].max() - hull_w[:, 2].min())
    face = float((hull_w[:, :2] @ d).min())
    centre = hull_w.mean(dim=0)
    return th.tensor([*(centre[:2] + d * (face - float(centre[:2] @ d))).tolist(), z])


def hull_cut_points(hull_w: th.Tensor, z: float) -> th.Tensor:
    """The cut of a convex hull, given by its points (world xyz), by the horizontal plane at ``z``: the points where the segment between
    every point at or below ``z`` and every point at or above it crosses the plane (N, 2; world xy). The hull's edges are among those
    segments and every crossing lies inside the hull, so their convex hull is the exact cut. Empty when the hull does not reach ``z``."""
    p = G.t(hull_w).float()
    lo, hi = p[p[:, 2] <= z], p[p[:, 2] >= z]
    if len(lo) == 0 or len(hi) == 0:
        return th.zeros(0, 2)
    a, b = lo[:, None, :], hi[None, :, :]
    da, db = z - a[..., 2], b[..., 2] - z  # both >= 0
    span = da + db
    t = th.where(span > 1e-12, da / span.clamp(min=1e-12), th.zeros_like(span))
    return (a[..., :2] + t[..., None] * (b[..., :2] - a[..., :2])).reshape(-1, 2)


def face_run_at(hull_w: th.Tensor, z: float, outward: th.Tensor, side_h: th.Tensor, face_depth_tol: float = 0.01) -> Optional[Tuple[float, float]]:
    """The flat run of the hull's face toward ``outward`` (horizontal unit) at height ``z``: ``(lo, hi)`` along ``side_h`` of the cut's
    points (:func:`hull_cut_points`) within ``face_depth_tol`` of its outermost offset - the face's width at that height, whatever the
    hull's vertices (a box hull has them at its top and bottom corners only). None when the hull does not reach ``z``."""
    cut = hull_cut_points(hull_w, z)
    if len(cut) == 0:
        return None
    off = cut @ G.t(outward)[:2].float()
    face = cut[off >= float(off.max()) - face_depth_tol]
    s = face @ G.t(side_h)[:2].float()
    return float(s.min()), float(s.max())


def polygon_centroid(poly) -> Optional[th.Tensor]:
    """The area centroid (xy) of a convex polygon (V, 2; CCW corners); the corners' mean for a degenerate one, None for an empty one."""
    p = G.t(poly).float().reshape(-1, 2)
    if len(p) == 0:
        return None
    q = th.roll(p, -1, dims=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-12:
        return p.mean(dim=0)
    return th.stack([((p[:, 0] + q[:, 0]) * cross).sum(), ((p[:, 1] + q[:, 1]) * cross).sum()]) / (6.0 * a)


def point_triangle_distances(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Euclidean distance (N, F) from every point of ``pts`` (N, 3) to every triangle of ``tris`` (F, 3, 3): the closest point on each
    triangle by its Voronoi regions (vertex, edge, face; Ericson, Real-Time Collision Detection 5.1.5), vectorised."""
    p = np.asarray(pts, dtype=float).reshape(-1, 1, 3)
    t = np.asarray(tris, dtype=float).reshape(1, -1, 3, 3)
    a, b, c = t[:, :, 0], t[:, :, 1], t[:, :, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def div(n, m):
        return n / np.where(np.abs(m) > 1e-18, m, 1e-18)

    s = va + vb + vc
    q = a + ab * div(vb, s)[..., None] + ac * div(vc, s)[..., None]  # inside the face region
    # the regions in reverse priority, each overriding the ones after it
    q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None], q)
    q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * div(d2, d2 - d6)[..., None], q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
    q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * div(d1, d1 - d3)[..., None], q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
    return np.linalg.norm(p - q, axis=-1)


class HullSections:
    """oracle-fix 2026-11-06 (``generation.push_height_sweep``): a convex hull given by its points (world xyz) - its horizontal cuts
    (:meth:`cut`), the outermost material of a cut on a line (:meth:`outermost`), a cut's area centroid and the distance of points to the
    hull (:meth:`distance`). qhull's triangles give the hull's edges (a cut's corners are where they cross the plane: a few dozen points,
    where :func:`hull_cut_points` pairs every point below with every point above) and its faces (the exact point-to-hull distance). Without
    qhull (a flat or degenerate point set) the cuts are :func:`hull_cut_points`' and :meth:`distance` is None."""

    def __init__(self, hull_w):
        self.pts = G.t(hull_w).float().reshape(-1, 3)
        self.z_lo, self.z_hi = float(self.pts[:, 2].min()), float(self.pts[:, 2].max())
        self.edges = self.tris = self.planes = None
        self._cuts: Dict[float, th.Tensor] = {}
        try:
            from scipy.spatial import ConvexHull

            P = self.pts.double().numpy()
            ch = ConvexHull(P)
            s = ch.simplices
            e = np.unique(np.sort(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]), axis=1), axis=0)
            self.edges, self.tris, self.planes = (P[e[:, 0]], P[e[:, 1]]), P[s], ch.equations
        except Exception:  # noqa: BLE001 - a flat / degenerate point set (or no scipy): the all-pairs cut, no distance
            pass

    def cut(self, z: float) -> th.Tensor:
        """The cut at height ``z`` (CCW corners, (V, 2) world xy; empty when the hull does not reach ``z``)."""
        key = round(float(z), 5)
        poly = self._cuts.get(key)
        if poly is None:
            if self.edges is None:
                pts = hull_cut_points(self.pts, float(z))
            else:
                a, b = self.edges
                za, zb = a[:, 2] - float(z), b[:, 2] - float(z)
                hit = (np.minimum(za, zb) <= 0.0) & (np.maximum(za, zb) >= 0.0)
                a, b, za, zb = a[hit], b[hit], za[hit], zb[hit]
                span = zb - za
                t = np.where(np.abs(span) > 1e-12, -za / np.where(np.abs(span) > 1e-12, span, 1.0), 0.0)
                pts = th.tensor(a[:, :2] + t[:, None] * (b[:, :2] - a[:, :2]), dtype=th.float32).reshape(-1, 2)
            poly = G.convex_hull_2d(pts) if len(pts) >= 3 else th.zeros(0, 2)
            self._cuts[key] = poly
        return poly

    def outermost(self, z: float, normal, tangent, t0: float) -> Optional[float]:
        """The largest coordinate along ``normal`` of the cut at ``z`` on the line {p · ``tangent`` = ``t0``} (``iors_large_nextto.outermost_on_line``);
        None when the line misses it."""
        from ispatialgym.sampling.v2.tasks.iors_large_nextto import outermost_on_line

        poly = self.cut(z)
        if len(poly) < 3:
            return None
        return outermost_on_line([poly], normal, tangent, t0)

    def centroid(self, z: float) -> Optional[th.Tensor]:
        """The area centroid (xy) of the cut at ``z``; None when the hull does not reach it."""
        poly = self.cut(z)
        return polygon_centroid(poly) if len(poly) >= 3 else None

    def distance(self, pts) -> Optional[th.Tensor]:
        """Distance (m) of each point (N, 3; world) to the hull, 0 inside; None without qhull's faces."""
        if self.tris is None:
            return None
        P = G.t(pts).double().reshape(-1, 3).numpy()
        inside = (P @ self.planes[:, :3].T + self.planes[:, 3]).max(axis=1) <= 0.0
        d = point_triangle_distances(P, self.tris).min(axis=1)
        d[inside] = 0.0
        return th.tensor(d, dtype=th.float32)


def min_area_rect(pts) -> Tuple[float, float, float, th.Tensor]:
    """The minimum-area bounding rectangle of the points ``pts`` (N, 2+; world xy, e.g. a footprint polygon): ``(angle, long, short, centre)``
    - the world angle (rad, in [0, pi)) of its longer side, the lengths of its two sides (one side lies along an edge of the convex hull)
    and its centre (world xy)."""
    xy = G.t(pts)[:, :2].float()
    hull = G.convex_hull_2d(xy) if len(xy) > 2 else xy
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if float(th.linalg.norm(e)) < 1e-9:
            continue
        ang = math.atan2(float(e[1]), float(e[0]))
        c, s = math.cos(ang), math.sin(ang)
        u, v = hull[:, 0] * c + hull[:, 1] * s, -hull[:, 0] * s + hull[:, 1] * c
        a, b = float(u.max() - u.min()), float(v.max() - v.min())
        if best is None or a * b < best[0] - 1e-12:
            uc, vc = 0.5 * float(u.max() + u.min()), 0.5 * float(v.max() + v.min())
            best = (a * b, ang, a, b, th.tensor([uc * c - vc * s, uc * s + vc * c]))
    if best is None:
        return 0.0, 0.0, 0.0, (xy.mean(dim=0) if len(xy) else th.zeros(2))
    _, ang, a, b, centre = best
    if b > a:
        ang, a, b = ang + math.pi / 2, b, a
    return ang % math.pi, a, b, centre


def touching_points(t_hull_w: th.Tensor, p_hull_w: th.Tensor, tangent) -> Tuple[th.Tensor, th.Tensor]:
    """§6.11 touching (the tabletop layout; ``IORSSmall.touching_points``): the push contact on the target's right face (mid height) and the
    brace point on the partner's far (left) face (mid height), world xyz, from the two hulls in world points and the stance tangent (the
    robot's left, the push direction)."""
    tangent = G.t(tangent).float()[:2]
    t_hull, p_hull = G.t(t_hull_w).float(), G.t(p_hull_w).float()
    z_t = float(t_hull[:, 2].min()) + 0.5 * float(t_hull[:, 2].max() - t_hull[:, 2].min())
    right_face = float((t_hull[:, :2] @ tangent).min())
    centre_t = t_hull.mean(dim=0)
    work = th.tensor([*(centre_t[:2] + tangent * (right_face - float(centre_t[:2] @ tangent))).tolist(), z_t])
    left_face = float((p_hull[:, :2] @ tangent).max())
    centre_p = p_hull.mean(dim=0)
    z_p = float(p_hull[:, 2].min()) + 0.5 * float(p_hull[:, 2].max() - p_hull[:, 2].min())
    brace = th.tensor([*(centre_p[:2] + tangent * (left_face - float(centre_p[:2] @ tangent))).tolist(), z_p])
    return work, brace


def hand_overlap(eef_pose, hull_local, obj_pose, margin: float = 0.005, against: str = "box", hull_w: Optional[th.Tensor] = None) -> Optional[str]:
    """Which part of the hand beyond the finger profile (``HAND_SPHERES_EEF``: wrist camera, knuckles, gripper body) comes within ``margin`` of
    the object - its box in its own frame (``against="box"``) or its hull (``"hull"``: :class:`HullSections` of ``hull_w``, the hull in world
    points; default ``hull_local`` posed at ``obj_pose``; the box when qhull cannot build it) - at the eef pose; None when all clear
    (``IORSSmall.hand_overlap``: the IK ignores the pushed / braced object at its contact pose, this is what sees the hand there)."""
    hull = G.t(hull_local)
    eef = (G.t(eef_pose[0]), G.t(eef_pose[1]))
    if against == "hull":
        sec = HullSections(hull_w if hull_w is not None else C.transform_points((G.t(obj_pose[0]), G.t(obj_pose[1])), hull))
        worst = None
        for part, (centres, r) in HAND_SPHERES_EEF.items():
            dist = sec.distance(C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
            if dist is None:
                break
            dmin = float(dist.min())
            if dmin < r + margin and (worst is None or dmin - r < worst[1]):
                worst = (part, dmin - r)
        else:
            return None if worst is None else f"{worst[0]} {worst[1] * 1000:+.0f} mm from the hull (margin {margin * 1000:.0f} mm)"
    lo, hi = hull.min(dim=0).values, hull.max(dim=0).values
    inv = C.invert((G.t(obj_pose[0]), G.t(obj_pose[1])))
    worst = None
    for part, (centres, r) in HAND_SPHERES_EEF.items():
        pts_o = C.transform_points(inv, C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
        d = th.clamp(lo - pts_o, min=0.0) + th.clamp(pts_o - hi, min=0.0)
        dist = float(th.linalg.norm(d, dim=1).min())
        if dist < r + margin and (worst is None or dist - r < worst[1]):
            worst = (part, dist - r)
    if worst is None:
        return None
    return f"{worst[0]} {worst[1] * 1000:+.0f} mm from the box (margin {margin * 1000:.0f} mm)"


# ------------------------------------------------------------------------------------------------ the push rules


def touching_push_candidates(gs, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                             t_hull_w: Optional[th.Tensor] = None, line_xy: Optional[th.Tensor] = None):
    """Push poses of the closed right gripper at the target's right face (``face_point``, mid height), pushing along
    ``direction`` (robot's left, into the partner), in the order to try: (1) the pads on the face — pad centre on the
    point, approach along ``direction``, closing horizontal then vertical; (2) the top-down blade beside the face
    (``tabletop.blade_standoff`` over the target's face height above the tips, pitched per ``approach_pitch_order``,
    pads' faces parallel to the pushed face, either wrist turn) — a sideways pad push 10 cm right of the target was
    out of reach for the right arm on every attempt of the sanitary-napkins loop. Yields ``(kind, eef pose, extras)``.
    ``t_hull_w`` (the target's hull at the place, given under :meth:`push_height_sweep_on`): :meth:`swept_push_candidates` instead
    (``line_xy``: its push line's point, :func:`push_line_t0`)."""
    if t_hull_w is not None:
        yield from swept_push_candidates(gs, manip, t_hull_w, face_point, face_z, direction, line_xy=line_xy)
        return
    d = C.unit(G.t(direction))
    outward = -d  # the pushed (right) face's outward normal
    z0, z1 = face_z
    clearance = float(gs.gen("stop_clearance_m", STOP_CLEARANCE))
    # A/B knob: pad contacts lifted by the constant (pad centre + half the finger height below the face top), blades lifted
    # with the fingertip kept 1 cm below the face top
    pad_lift, pad_why = contact_height_extra(gs, "push", pad_center_z=float(face_point[2]), band_top_z=z1)
    fp = G.t(face_point) + th.tensor([0.0, 0.0, pad_lift])
    pad_facts = {"height_extra_m": requested_height_extra(gs), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}
    # owner 2026-09-18 19:39 / 20:10: FINGERTIP-FRONT pushes at the face's mid height first - the hand horizontal along the
    # push direction, the closed tips ``clearance`` before the face (``fingertip_eef``; ``contact_eef`` puts the pad centre
    # on the face, tips 3 cm inside the box). Pushing at the centre-of-mass height the tall narrow box slides; every blade
    # pushed it 9 cm up and it fell over (0918-194552: z 0.867 -> 0.848). Horizontal closing: the knuckles beside the
    # pads, the wrist camera 7.6 cm behind the tips (never over the box), the gripper body ~1.5 cm above the table at the
    # mid height; ``pre_dist`` 5 cm (the 10 cm pre 10 cm to the right of the face was out of IK reach on every attempt).
    # A vertical closing hangs the camera 6-7 cm below / above the pads: only the camera-UP turn is admissible.
    h_axis = C.default_closing_axis(d)
    tip_pre = float(gs.gen("push_tip_pre_dist_m", 0.05))
    for c_sign in PUSH_WRIST_TURNS:
        pose = C.fingertip_eef(fp, d, c_sign * h_axis, clearance)
        yield f"push_tip_h{'' if c_sign > 0 else '-'}", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True,
                                                                                        "stop_clearance_m": clearance, **pad_facts}
    pose = C.fingertip_eef(fp, d, th.tensor([0.0, 0.0, -1.0]), clearance)  # eef +y = down: the camera (eef -y side) points up
    yield "push_tip_v-", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **pad_facts}
    yield "push_h", IK.contact_eef(fp, d), {"blade_half_thickness_m": None, **pad_facts}
    yield "push_v", IK.contact_eef(fp, d, closing=th.tensor([0.0, 0.0, 1.0])), {"blade_half_thickness_m": None, **pad_facts}
    raised_fallback = None
    # owner 2026-09-18 15:50: a sideways push uses the VERTICAL blade only - ``pitched_approach`` leans the blade about the
    # push axis, so a pitched finger side meets the face obliquely and yaws the box (0918-153503: 45 deg blade, 29 deg yaw,
    # contact lost); ``generation.push_blade_pitches_deg`` lists the admissible pitches
    pitches = [float(p) for p in (gs.gen("push_blade_pitches_deg", None) or [0.0])]
    for pitch in [p for p in TT.approach_pitch_order(None) if p in pitches]:
        z_tip = z0 + 0.005
        if bool(gs.gen("blade_toward_com", False)):  # off since 2026-09-18 16:00: the 2 cm raise put the recorded retreat's clear pose at the reach limit (0918-154421: 4/8 'retreat: exact IK failed'); the oracle raises the pad at run time for the palm
            # owner 2026-09-17 23:39: push nearer the centre of mass - the finger section (centre ~3.5 cm above the
            # tips) is centred on the face's mid height when the face allows it, else it stays at the face bottom
            z_tip = max(z_tip, min(float(face_point[2]) - 0.035, z1 - 0.03))
        blade_lift, blade_why = contact_height_extra(gs, "blade", fingertip_z=z_tip, band_top_z=z1)
        z_tip += blade_lift
        # a push needs the blade on the lower part of the face only (unlike a stop brace, which the whole face may
        # meet): faces taller than the finger section are pushed over their lowest ``PALM_FROM_TIP`` (7 cm)
        # (the finger section is PALM_FROM_TIP long along the finger axis: pitched, it covers cos(pitch) of that in height)
        z_face_hi = min(z1, z_tip + (TT.PALM_FROM_TIP - 0.005) * math.cos(math.radians(pitch)))
        # owner 2026-09-17 23:39 (touching 0917-22*: the palm pressed the sanitary-napkins box and shook the table): a
        # face that reaches the palm section of a vertical blade (z1 above the fingers' 7 cm) is only pushed by a
        # PITCHED blade - the palm leans away from the face by ~ PALM_FROM_TIP·sin(pitch) - never by the 0° one,
        # whose stand-off (finger profile only) puts the wider palm inside the face
        # (default 0 = off since 2026-09-18 15:45: ``pitched_approach`` leans the blade FORWARD, about the push axis, so
        # a pitched blade does not move the palm away from a face lying to the pad's side - the 45 deg blade of
        # 0918-144517 knocked the box over; the oracle raises a vertical blade until its palm clears the face top)
        if z1 > z_tip + TT.PALM_FROM_TIP and pitch < float(gs.gen("blade_palm_min_pitch_deg", 0.0)):
            # fallback (after the pitched blades, which come first in the order): the vertical blade RAISED so that its
            # palm clears the face top - the finger section pushes the top PALM_FROM_TIP of the face (re-sample
            # 0918-010325: every pitched blade fails the exact IK from the stance)
            z_hi_tip = z1 - TT.PALM_FROM_TIP + 0.005
            if bool(gs.gen("blade_palm_clear_fallback", True)) and z_hi_tip <= z1 - 0.03:
                raised_fallback = (z_hi_tip,)  # yielded after the pitched blades (below)
            continue
        leans = gs.gen("blade_lean_deg", [20.0, 12.0, 8.0])
        leans = [float(v) for v in (leans if isinstance(leans, (list, tuple)) else [leans]) if float(v) > 0]
        if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and leans:
            # owner 2026-09-18 19:05: a tall face is pushed LOW by a blade leaning along the push axis (fingertips lead, the
            # palm recedes from the face by h·tan(lean)): the raised blade pushed the top 4 cm of the 11 cm box and it
            # tipped instead of sliding - each base nudge closed 10.9 mm and the same 10.9 mm reopened at the back-off
            # (0918-184752: touching held during the push, lost at the last step). Leans in order: 20 deg has no pre-pose
            # IK at these stances (0918-191811, the wrist roll at its limit); at 12 deg the thick finger section touches at
            # ~ the centre-of-mass height and the palm still recedes 1.5 cm.
            # The tips ``blade_lean_tip_above_m`` above the face bottom: leaning, the fingertip's leading edge dips below the tip
            # height and 5 mm above the table collided with it at the contact pose (0918-190236)
            z_lean_tip = z0 + float(gs.gen("blade_lean_tip_above_m", 0.025))
            for lean in leans:
                yield from leaning_blade(gs, manip, face_point, (z0, z1), d, outward, clearance, z_lean_tip, lean)
        if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and bool(gs.gen("blade_palm_clear_first", True)):
            # owner 2026-09-18 18:45: a face taller than the finger section is pushed by the RAISED vertical blade first (palm
            # ``blade_palm_clear_m`` above the face top, validated by the chain check) - the low blade's palm overlaps the top
            # and the oracle raised it 3.9 cm at run time to an unvalidated pose: 'joints did not converge 0.087 rad'
            # (0918-182409); with 5 mm of clearance the wider palm / knuckle came down ON the box's top edge as the blade
            # descended beside the face (0918-183535, same error) - 2.5 cm by default
            palm_clear = float(gs.gen("blade_palm_clear_m", 0.025))
            yield from raised_blade(gs, manip, face_point, (z0, z1), d, outward, clearance, z1 - TT.PALM_FROM_TIP + palm_clear)
        so = TT.blade_standoff((z0, z_face_hi), z_tip, pitch, clearance, side=False)
        if so is None:
            continue  # nothing of the face above the tips: no blade push
        standoff, half = so
        a_vec = TT.pitched_approach(pitch, manip.forward)
        origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
        for c_sign in PUSH_WRIST_TURNS:
            yield f"push_blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                "height_extra_m": requested_height_extra(gs), "height_extra_effective_m": round(blade_lift, 4), "height_extra_clip_reason": blade_why}
    if raised_fallback is not None:
        yield from raised_blade(gs, manip, face_point, (z0, z1), d, outward, clearance, raised_fallback[0])


def leaning_blade(gs, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float,
                   z_tip: float, lean_deg: float):
    """The closed blade with its fingertips ``z_tip`` (5 mm above the face bottom) and the hand leaning ``lean_deg`` along
    the push direction (approach = down rotated toward ``d``): the fingertips lead, every higher finger section recedes
    from the face by h·tan(lean), so only the tips touch (below the centre of mass: the box slides, no tipping) and the
    palm never reaches the face. Stand-off = the largest protrusion of the finger profile toward the face + clearance."""
    z0, z1 = face_z
    theta = math.radians(float(lean_deg))
    d3 = th.tensor([float(d[0]), float(d[1]), 0.0])
    d3 = d3 / max(float(th.linalg.norm(d3)), 1e-9)
    a_vec = C.unit(th.tensor([0.0, 0.0, -math.cos(theta)]) + d3 * math.sin(theta))
    protrude, h_prev = 0.0, 0.0
    for h, half in TT.FINGER_PROFILE:  # a section spans (h_prev, h]; its lowest point protrudes the most
        protrude = max(protrude, float(half) - h_prev * math.tan(theta))
        h_prev = float(h)
    standoff = protrude + float(clearance)
    # eef origin = fingertip - a·FINGERTIP_LENGTH, the tips ``standoff`` outside the face at ``z_tip``
    tip = th.tensor([float(face_point[0]), float(face_point[1]), float(z_tip)]) + outward * standoff
    origin = tip - a_vec * C.FINGERTIP_LENGTH
    for c_sign in PUSH_WRIST_TURNS:
        yield f"push_blade_lean{lean_deg:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
            "blade_half_thickness_m": float(TT.FINGER_PROFILE[0][1]), "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": float(z_tip),
            "palm_clear": True, "lean_deg": float(lean_deg), "height_extra_m": requested_height_extra(gs), "height_extra_effective_m": 0.0,
            "height_extra_clip_reason": "leaning_blade"}


def raised_blade(gs, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float, z_tip: float):
    """The vertical (0 deg) blade with its fingertips at ``z_tip`` = face top − PALM_FROM_TIP: the palm stands above
    the face, the finger section covers the face's top part. Yielded after every pitched blade failed."""
    z0, z1 = face_z
    so = TT.blade_standoff((z0, z1), z_tip, 0.0, clearance, side=False)
    if so is None:
        return
    standoff, half = so
    a_vec = TT.pitched_approach(0.0, manip.forward)
    origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH]) + outward * standoff
    for c_sign in PUSH_WRIST_TURNS:
        yield f"push_blade_0_high{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
            "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
            "palm_clear": True, "height_extra_m": requested_height_extra(gs), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "palm_clear"}


def push_line_t0(sec: "HullSections", t_hull_w: th.Tensor, z0: float, side2: th.Tensor, line_xy: Optional[th.Tensor] = None) -> Tuple[float, Dict[str, Any]]:
    """The push line's offset along ``side2`` (the across-the-push axis): through the target's FOOTPRINT centre (the area centroid of its
    hull's cut :data:`PUSH_SWEEP_BASE_CUT_M` above the bottom; the whole hull's when the cut misses) - or, with ``line_xy`` (world xy, the
    i7-contact3 switch ``ISPATIALGYM_PUSH_LINE_COM``: the target's mass centre / root), through that point. ``(t0, facts)``; the facts name the
    line's point and its offset from the footprint centre (m, + = along ``side2``) when ``line_xy`` is given, else ``{}``."""
    centre = sec.centroid(z0 + PUSH_SWEEP_BASE_CUT_M)
    if centre is None:
        centre = polygon_centroid(G.convex_hull_2d(G.t(t_hull_w)[:, :2].float()))
    t_fp = float(centre @ side2)
    if line_xy is None:
        return t_fp, {}
    t0 = float(G.t(line_xy)[:2].float() @ side2)
    return t0, {"push_line": {"through": "com", "offset_from_footprint_m": round(t0 - t_fp, 4)}}


def swept_push_candidates(gs, manip: Stance, t_hull_w: th.Tensor, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor,
                          line_xy: Optional[th.Tensor] = None):
    """:meth:`push_height_sweep_on` (touching#04: every class at mid height failed, so the palm-clear raised blade pushed the 27 cm
    spray bottle at 83 % of its height and it tilted 3.6 -> 20.3 deg): today's push classes - the fingertip pushes ``push_tip_h`` /
    ``push_tip_v-``, the pad pushes ``push_h`` / ``push_v``, the leaning blades ``blade_lean_deg``, the vertical blade - at every
    contact height (m above the target's bottom, the support top) from the class's lowest that clears the support
    (:data:`PUSH_SWEEP_MIN_HEIGHT_M`; a leaning blade's ``blade_lean_tip_above_m``) in ``push_height_step_m`` (0.01) steps up to
    ``push_max_height_fraction`` (0.5) x the target's height, yielded by contact height (a blade's fingertips, else the pads / tips;
    today's class order breaks ties, a class's two wrist turns stay together). The contact lies on the line through the target's
    footprint centre (the area centroid of its hull's cut :data:`PUSH_SWEEP_BASE_CUT_M` above the bottom: the part it stands on, not
    the trigger head the whole hull's outermost point was) along the push, on the hull's cut there: the outermost material within
    :data:`PUSH_SWEEP_WINDOW_M` of the line and of the height, so the tips start ``stop_clearance_m`` off everything they face; a
    blade stands off from the cut over its whole finger section (``TT.FINGER_PROFILE``). The hand above the fingers is left to
    :meth:`hand_overlap` against the hull (a vertical blade beside a tall face fails there: its palm and knuckles reach into it), so
    a blade that passes keeps its recorded height (``palm_clear``: the oracle does not raise it). The raised blade comes last and only
    for a target that is not tall (height / smallest footprint side <= ``push_raised_max_aspect``, :data:`PUSH_RAISED_MAX_ASPECT`);
    a tall target without a low contact fails ``contact_unreachable`` at the stance, and another stance is tried. Yields
    ``(kind, eef pose, extras)``; the extras carry ``contact_height_m``, ``height_fraction`` and ``contact_point`` (world)."""
    d = C.unit(G.t(direction).float())
    outward = -d
    d2 = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
    side2, out2 = th.tensor([-float(d2[1]), float(d2[0])]), -d2
    z0, z1 = float(face_z[0]), float(face_z[1])
    height = max(z1 - z0, 1e-6)
    sec = HullSections(t_hull_w)
    t0, line_facts = push_line_t0(sec, t_hull_w, z0, side2, line_xy)
    clearance = float(gs.gen("stop_clearance_m", STOP_CLEARANCE))
    step = max(float(TT.phase2_gen(gs, "push_height_step_m", 0.01)), 0.002)
    h_cap = float(TT.phase2_gen(gs, "push_max_height_fraction", 0.5)) * height
    tip_pre = float(gs.gen("push_tip_pre_dist_m", 0.05))
    knob = {"height_extra_m": requested_height_extra(gs), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "push_height_sweep", **line_facts}

    def face(z: float, dz: float = PUSH_SWEEP_WINDOW_M) -> Optional[float]:
        """The outermost material (along ``outward``) within the window around the centre line at height ``z``; None where the line
        itself misses the cut."""
        if sec.outermost(z, out2, side2, t0) is None:
            return None
        zs = [zz for zz in ((z - dz, z, z + dz) if dz > 0 else (z,)) if z0 <= zz <= z1] or [z]
        vals = [sec.outermost(zz, out2, side2, t0 + k * 0.5 * PUSH_SWEEP_WINDOW_M) for zz in zs for k in (-2, -1, 0, 1, 2)]
        return max(v for v in vals if v is not None)

    def point(s: float, z: float) -> th.Tensor:
        xy = side2 * t0 + out2 * float(s)
        return th.tensor([float(xy[0]), float(xy[1]), float(z)])

    def heights(h_min: float) -> List[float]:
        return [round(h_min + k * step, 4) for k in range(int(math.floor((h_cap - h_min) / step + 1e-6)) + 1)] if h_min <= h_cap + 1e-9 else []

    def facts(h: float, contact: th.Tensor) -> Dict[str, Any]:
        return {"contact_height_m": round(h, 4), "height_fraction": round(h / height, 3), "contact_point": [round(float(v), 4) for v in contact],
                "push_height_sweep": True, **knob}

    def tips(h: float) -> List[Tuple[str, G.Pose, Dict[str, Any]]]:  # push_tip_h / push_tip_h- (closing horizontal)
        s = face(z0 + h)
        if s is None:
            return []
        fp, h_axis = point(s, z0 + h), C.default_closing_axis(d)
        return [(f"push_tip_h{'' if c > 0 else '-'}", C.fingertip_eef(fp, d, c * h_axis, clearance),
                 {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **facts(h, fp)}) for c in PUSH_WRIST_TURNS]

    def tip_v(h: float):  # push_tip_v- (closing vertical, the camera up)
        s = face(z0 + h)
        if s is None:
            return []
        fp = point(s, z0 + h)
        return [("push_tip_v-", C.fingertip_eef(fp, d, th.tensor([0.0, 0.0, -1.0]), clearance),
                 {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, **facts(h, fp)})]

    def pad(h: float, vertical: bool):  # push_h / push_v (the pad centre on the face)
        s = face(z0 + h)
        if s is None:
            return []
        fp = point(s, z0 + h)
        pose = IK.contact_eef(fp, d, closing=th.tensor([0.0, 0.0, 1.0])) if vertical else IK.contact_eef(fp, d)
        return [("push_v" if vertical else "push_h", (G.t(pose[0]), G.t(pose[1])), {"blade_half_thickness_m": None, **facts(h, fp)})]

    def section_standoff(z_tip: float, lean: float) -> Optional[Tuple[float, float, float]]:
        """``(tip offset along outward, the face at the tips, the thickest half)``: the fingertip axis this far out that every finger
        section (``TT.FINGER_PROFILE``, receding h·sin(lean) when leaning) keeps ``clearance`` off the cut; None without material."""
        th_l = math.radians(lean)
        need, half_max, h_prev = None, 0.0, 0.0
        for h_sec, half in TT.FINGER_PROFILE:
            for hf in np.linspace(h_prev, float(h_sec), max(2, int(math.ceil((float(h_sec) - h_prev) / 0.01)) + 1)):
                zz = z_tip + float(hf) * math.cos(th_l)
                if not (z0 <= zz <= z1):
                    continue
                s = face(zz, dz=0.0)
                if s is None:
                    continue
                v = s + float(half) - float(hf) * math.sin(th_l)
                need, half_max = (v if need is None else max(need, v)), max(half_max, float(half))
            h_prev = float(h_sec)
        s_tip = face(z_tip)
        if need is None or s_tip is None:
            return None
        return need + clearance, s_tip, half_max

    def blade(h: float):  # push_blade_0 / push_blade_0- (vertical, beside the face)
        z_tip = z0 + h
        so = section_standoff(z_tip, 0.0)
        if so is None:
            return []
        s_o, s_tip, half = so
        a_vec = TT.pitched_approach(0.0, manip.forward)
        origin = point(s_o, z_tip + C.FINGERTIP_LENGTH)
        return [(f"push_blade_0{'' if c > 0 else '-'}", (origin, C.eef_quat(a_vec, c * outward)),
                 {"blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": round(s_o - s_tip, 4), "fingertip_z": z_tip, "palm_clear": True,
                  **facts(h, point(s_tip, z_tip))}) for c in PUSH_WRIST_TURNS]

    def leaning(h: float, lean: float):  # push_blade_lean<deg> (the tips lead, the palm recedes)
        z_tip = z0 + h
        so = section_standoff(z_tip, lean)
        if so is None:
            return []
        s_o, s_tip, _half = so
        theta = math.radians(lean)
        a_vec = C.unit(th.tensor([0.0, 0.0, -math.cos(theta)]) + th.tensor([float(d[0]), float(d[1]), 0.0]) * math.sin(theta))
        origin = point(s_o, z_tip) - a_vec * C.FINGERTIP_LENGTH
        return [(f"push_blade_lean{lean:.0f}{'' if c > 0 else '-'}", (origin, C.eef_quat(a_vec, c * outward)),
                 {"blade_half_thickness_m": float(TT.FINGER_PROFILE[0][1]), "stop_clearance_m": clearance, "blade_standoff_m": round(s_o - s_tip, 4),
                  "fingertip_z": float(z_tip), "palm_clear": True, "lean_deg": float(lean), **facts(h, point(s_tip, z_tip))}) for c in PUSH_WRIST_TURNS]

    leans = gs.gen("blade_lean_deg", [20.0, 12.0, 8.0])
    leans = [float(v) for v in (leans if isinstance(leans, (list, tuple)) else [leans]) if float(v) > 0]
    pitches = [float(p) for p in (gs.gen("push_blade_pitches_deg", None) or [0.0])]
    lean_tip = float(gs.gen("blade_lean_tip_above_m", 0.025))
    # (height, class rank = today's order, builder): the fingertip and pad pushes, the leaning blades, the vertical blade
    groups: List[Tuple[float, int, Any]] = []
    groups += [(h, 0, (lambda h=h: tips(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["tip_h"])]
    groups += [(h, 1, (lambda h=h: tip_v(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["tip_v"])]
    groups += [(h, 2, (lambda h=h: pad(h, False))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["pad"])]
    groups += [(h, 3, (lambda h=h: pad(h, True))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["pad"])]
    for i, lean in enumerate(leans):
        groups += [(h, 4 + i, (lambda h=h, lean=lean: leaning(h, lean))) for h in heights(lean_tip)]
    if 0.0 in pitches:
        groups += [(h, 4 + len(leans), (lambda h=h: blade(h))) for h in heights(PUSH_SWEEP_MIN_HEIGHT_M["blade"])]
    for _h, _rank, build in sorted(groups, key=lambda g: (g[0], g[1])):
        yield from build()
    # the raised blade (palm above the top: the fingers push its top PALM_FROM_TIP) - the last resort of a target that is not tall
    _ang, _long, short, _c = min_area_rect(G.convex_hull_2d(G.t(t_hull_w)[:, :2].float()))
    aspect = height / max(short, 1e-6)
    if aspect <= float(gs.gen("push_raised_max_aspect", PUSH_RAISED_MAX_ASPECT)) + 1e-9 and z1 - z0 > TT.PALM_FROM_TIP:
        z_tip = z1 - TT.PALM_FROM_TIP + float(gs.gen("blade_palm_clear_m", 0.025))
        for kind, pose, extras in raised_blade(gs, manip, face_point, (z0, z1), d, outward, clearance, z_tip):
            yield kind, pose, dict(extras, contact_height_m=round(z_tip - z0, 4), height_fraction=round((z_tip - z0) / height, 3), push_height_sweep="raised_last",
                                   aspect=round(aspect, 2))


def open_push_candidates(gs, manip: Stance, t_hull_w: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor, line_xy: Optional[th.Tensor] = None):
    """:meth:`push_gripper_open_on`: the right gripper open ``push_open_opening_m`` per finger (:data:`PUSH_OPEN_OPENING_M`), approaching
    along the push with both fingertips ``stop_clearance_m`` before the face on the line through the target's footprint centre (as
    :meth:`swept_push_candidates`). ``push_open_v-``: closing vertical, the wrist camera up (``push_tip_v-``'s turn) - the tips one above
    the other on the line, the lower one the contact, the hand above it by the half opening; ``push_open_h-`` / ``push_open_h``: closing
    horizontal - the tips side by side at the contact height (the wider contact). A class's eef axis runs from its lowest
    (:data:`PUSH_SWEEP_MIN_HEIGHT_M`: the lower knuckle 4.2 cm below the axis / the gripper body; raised until the open fingers clear the
    support by :data:`PUSH_OPEN_SUPPORT_CLEARANCE_M`) in ``push_height_step_m`` steps up to ``push_max_height_fraction`` x the target's
    height; both tips must land on the face's flat run :data:`BRACE_OPEN_END_MARGIN_M` from its ends, their outermost material within
    :data:`BRACE_OPEN_FACE_DEPTH_TOL_M` of each other (one tip leading would yaw the target). Yielded by the lower fingertip's height
    (``contact_height_m``), ``push_open_v-`` first on a tie; the extras carry ``fingers_opening_m`` (the oracle's opening),
    ``tip_heights_m``, ``eef_height_m``, ``tip_spread_m``, ``hand_clearance_m`` (the hand's lowest point - knuckle, gripper body, camera or
    the open fingers - above the support) and ``contact_point`` (world: the lower tip / the tips' middle on the face)."""
    d = C.unit(G.t(direction).float())
    d2 = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
    side2, out2 = th.tensor([-float(d2[1]), float(d2[0])]), -d2
    side3, out3 = th.tensor([float(side2[0]), float(side2[1]), 0.0]), th.tensor([float(out2[0]), float(out2[1]), 0.0])
    z0, z1 = float(face_z[0]), float(face_z[1])
    height = max(z1 - z0, 1e-6)
    sec = HullSections(t_hull_w)
    t0, line_facts = push_line_t0(sec, t_hull_w, z0, side2, line_xy)
    clearance = float(gs.gen("stop_clearance_m", STOP_CLEARANCE))
    step = max(float(TT.phase2_gen(gs, "push_height_step_m", 0.01)), 0.002)
    h_cap = float(TT.phase2_gen(gs, "push_max_height_fraction", 0.5)) * height
    tip_pre = float(gs.gen("push_tip_pre_dist_m", 0.05))
    q_open = min(float(C.FINGER_OPEN), max(0.0, float(TT.phase2_gen(gs, "push_open_opening_m", PUSH_OPEN_OPENING_M))))
    tip_off = q_open + OPEN_TIP_CENTRE_M  # a fingertip centre from the eef axis
    tip_half = float(TT.FINGER_SIDE_PROFILE[0][1])  # a fingertip's half width across the closing axis
    knob = {"height_extra_m": requested_height_extra(gs), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "push_gripper_open", **line_facts}
    down = th.tensor([0.0, 0.0, -1.0])
    h_axis = C.default_closing_axis(d)

    def face_at(z: float, t: float) -> Optional[float]:
        """The outermost material (along ``outward``) within a fingertip's window around (``t``, ``z``); None where the line misses the cut."""
        if not (z0 <= z <= z1) or sec.outermost(z, out2, side2, t) is None:
            return None
        vals = [sec.outermost(zz, out2, side2, t + k * tip_half) for zz in (z - PUSH_SWEEP_WINDOW_M, z, z + PUSH_SWEEP_WINDOW_M) if z0 <= zz <= z1
                for k in (-1, 0, 1)]
        return max(v for v in vals if v is not None)

    def on_run(z: float, lo: float, hi: float) -> bool:
        run = face_run_at(t_hull_w, z, out3, side3, BRACE_OPEN_FACE_DEPTH_TOL_M)
        return run is not None and run[0] + BRACE_OPEN_END_MARGIN_M <= lo + 1e-9 and hi <= run[1] - BRACE_OPEN_END_MARGIN_M + 1e-9

    def point(s: float, t: float, z: float) -> th.Tensor:
        xy = side2 * float(t) + out2 * float(s)
        return th.tensor([float(xy[0]), float(xy[1]), float(z)])

    def facts(h_lo: float, h_hi: float, h_eef: float, contact: th.Tensor, pose: G.Pose) -> Dict[str, Any]:
        drop = max(TT.hand_drop_below(pose[1]), open_fingers_drop_below(pose[1], q_open))
        return {"blade_half_thickness_m": None, "pre_dist_m": tip_pre, "tip_front": True, "stop_clearance_m": clearance, "push_gripper_open": True,
                "fingers_opening_m": round(q_open, 4), "tip_spread_m": round(2.0 * tip_off, 4), "contact_height_m": round(h_lo, 4),
                "height_fraction": round(h_lo / height, 3), "tip_heights_m": [round(h_lo, 4), round(h_hi, 4)], "eef_height_m": round(h_eef, 4),
                "hand_clearance_m": round(float(pose[0][2]) - drop - z0, 4), "contact_point": [round(float(v), 4) for v in contact], **knob}

    def open_v(h_eef: float):  # push_open_v- (closing vertical, the camera up): the tips at the axis -/+ tip_off
        z_ax = z0 + h_eef
        z_lo, z_hi = z_ax - tip_off, z_ax + tip_off
        s_lo, s_hi = face_at(z_lo, t0), face_at(z_hi, t0)
        if s_lo is None or s_hi is None or abs(s_lo - s_hi) > BRACE_OPEN_FACE_DEPTH_TOL_M:
            return []
        if not (on_run(z_lo, t0 - tip_half, t0 + tip_half) and on_run(z_hi, t0 - tip_half, t0 + tip_half)):
            return []
        pose = C.fingertip_eef(point(max(s_lo, s_hi), t0, z_ax), d, down, clearance)  # eef +y = down: the camera (eef -y side) up
        pose = (G.t(pose[0]), G.t(pose[1]))
        return [("push_open_v-", pose, facts(z_lo - z0, z_hi - z0, h_eef, point(s_lo, t0, z_lo), pose))]

    def open_h(h_eef: float):  # push_open_h- / push_open_h (closing horizontal): the tips at the line -/+ tip_off
        z = z0 + h_eef
        if not on_run(z, t0 - tip_off - tip_half, t0 + tip_off + tip_half):
            return []
        vals = [face_at(z, t) for t in (t0 - tip_off, t0 + tip_off)]
        if any(v is None for v in vals) or abs(vals[0] - vals[1]) > BRACE_OPEN_FACE_DEPTH_TOL_M:
            return []
        out = []
        for c in PUSH_WRIST_TURNS:
            pose = C.fingertip_eef(point(max(vals), t0, z), d, c * h_axis, clearance)
            pose = (G.t(pose[0]), G.t(pose[1]))
            ex = facts(h_eef, h_eef, h_eef, point(max(vals), t0, z), pose)
            if ex["hand_clearance_m"] >= PUSH_OPEN_SUPPORT_CLEARANCE_M - 1e-6:  # the camera-down turn hangs its camera 7.4 cm below the axis
                out.append((f"push_open_h{'' if c > 0 else '-'}", pose, ex))
        return out

    def heights(h_min: float) -> List[float]:
        return [round(h_min + k * step, 4) for k in range(int(math.floor((h_cap - h_min) / step + 1e-6)) + 1)] if h_min <= h_cap + 1e-9 else []

    # the eef axis's lowest per class: the hand's (the lower knuckle / the gripper body), raised until the open lower finger clears the support
    q_v = C.eef_quat(d, down)
    v_min = max(PUSH_SWEEP_MIN_HEIGHT_M["tip_v"], open_fingers_drop_below(q_v, q_open) + PUSH_OPEN_SUPPORT_CLEARANCE_M)
    h_min = max(PUSH_SWEEP_MIN_HEIGHT_M["tip_h"], open_fingers_drop_below(C.eef_quat(d, h_axis), q_open) + PUSH_OPEN_SUPPORT_CLEARANCE_M)
    groups: List[Tuple[float, int, Any]] = [(h - tip_off, 0, (lambda h=h: open_v(h))) for h in heights(round(v_min, 4))]
    groups += [(h, 1, (lambda h=h: open_h(h))) for h in heights(round(h_min, 4))]
    for _h, _rank, build in sorted(groups, key=lambda g: (round(g[0], 4), g[1])):
        yield from build()


def wide_stop_spread(partner_hull_w: th.Tensor, b_point: th.Tensor, outward: th.Tensor, side_h: th.Tensor, s_push: float,
                     spreads: Sequence[float] = (0.09, 0.07, 0.05), end_margin: float = 0.01, face_depth_tol: float = 0.01, band: float = 0.03) -> Optional[float]:
    """Owner 2026-09-19 00:49 rule for a WIDE stop (both fingertips on the braced face, spread along it): the largest
    ``spreads`` value whose two landing points ``s_push ± s/2`` lie on the far face's flat run with ``end_margin`` to its
    ends. The flat run = the side coordinates of the hull points within ``face_depth_tol`` of the face's outermost
    offset in the ``band`` around the brace height — a protrusion at one end leaves only its own points there (a short
    run: no wide stop), a face shorter than the smallest spread + margins gets none either. None = narrow (closed) stop."""
    z_b = float(b_point[2])
    pts = partner_hull_w[(partner_hull_w[:, 2] > z_b - band) & (partner_hull_w[:, 2] < z_b + band)]
    if len(pts) < 2:
        return None
    off = pts[:, :2] @ outward[:2]
    face = pts[off >= float(off.max()) - face_depth_tol]
    s = face[:, :2] @ side_h[:2]
    s_min, s_max = float(s.min()) + end_margin, float(s.max()) - end_margin
    for spread in spreads:
        if s_min <= s_push - spread / 2 and s_push + spread / 2 <= s_max:
            return float(spread)
    return None


#: the touching wide stop's tip spreads (m, ``generation.brace_wide_spreads_m``), the widest that fits the braced face is used
BRACE_WIDE_SPREADS_M: Tuple[float, ...] = (0.09, 0.07, 0.05)
#: owner decision 2026-11-07 ("an open brace opens FULLY (0.05 per finger), narrower only when the braced face cannot take the full spread"):
#: the ONE switch ``ISPATIALGYM_BRACE_FULL_OPEN`` and its rule live in ``tabletop`` (:data:`tabletop.BRACE_FULL_OPEN_ENV`,
#: :func:`tabletop.brace_full_open`, :data:`tabletop.BRACE_FULL_OPENING_M`; i7-contactbrace 2026-11-08 - this module defined its own copy).
#: On, the touching wide stop's spreads start with the full-open spread 2 x 0.05 + CLOSED_TIP_SPREAD_M = 0.127 m, the narrower ones kept as
#: the fallback where the face is too short for it
BRACE_FULL_OPEN_ENV = TT.BRACE_FULL_OPEN_ENV
BRACE_FULL_OPEN_SPREAD_M = round(2.0 * TT.BRACE_FULL_OPENING_M + CLOSED_TIP_SPREAD_M, 4)
#: whether the switch is on (:func:`tabletop.brace_full_open`; the name the contact library's identity reads)
brace_full_open_on = TT.brace_full_open


def brace_wide_spreads(gs) -> List[float]:
    """The wide stop's spreads in the order given (``generation.brace_wide_spreads_m``, default :data:`BRACE_WIDE_SPREADS_M`), the full-open
    spread :data:`BRACE_FULL_OPEN_SPREAD_M` first under ``ISPATIALGYM_BRACE_FULL_OPEN=1`` (each finger at ``(spread - CLOSED_TIP_SPREAD_M) / 2``:
    0.05 m for it)."""
    spreads = [float(v) for v in gs.gen("brace_wide_spreads_m", list(BRACE_WIDE_SPREADS_M))]
    if brace_full_open_on() and not any(abs(v - BRACE_FULL_OPEN_SPREAD_M) < 1e-9 for v in spreads):
        spreads = [BRACE_FULL_OPEN_SPREAD_M] + spreads
    return spreads


def brace_tip_heights(gs, b_point: th.Tensor, face_z: Tuple[float, float]) -> List[float]:
    """The fingertip-front brace heights on the partner's far face, the higher first: the braced point's height raised to
    ``brace_tip_min_above_m`` above the face's bottom (the gripper body clears the table) and ``z0 + 0.075`` (at most 1 cm under the
    top); heights within 1 cm of the top dropped."""
    z0, z1 = face_z
    heights = [max(float(b_point[2]), z0 + float(gs.gen("brace_tip_min_above_m", 0.055))), min(z1 - 0.01, z0 + 0.075)]
    return sorted({round(h, 4) for h in heights if h + 0.01 <= z1}, reverse=True)


def brace_open_rule(gs, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: Optional[th.Tensor],
                    push_point: Optional[th.Tensor] = None, *, open_when_wide: bool = False) -> Optional[Dict[str, Any]]:
    """``generation.brace_open_when_wide``: whether the partner's braced (far) face takes the open fingertips. The face's width = its
    flat run at the higher brace height (:meth:`brace_tip_heights`) in the hull's horizontal cut (:func:`face_run_at`); wide = at least
    the smallest open spread of ``brace_wide_spreads_m`` plus ``BRACE_OPEN_END_MARGIN_M`` at each end. The spread is the largest one
    whose two tips straddle the push line (centred on it) with that margin to the face's ends, as the wide stop's rule
    (:meth:`wide_stop_spread`) - which reads the run off the hull's VERTICES within 3 cm of the brace height: a box hull's lie at its
    top and bottom corners, so a tall box's mid-height band holds few or none (touching#02 pair1013_c8_k8: two of dkhgxn's 56 points on
    its 15.5 cm face, a 9.5 cm run off the push line - no spread, closed tips). None without the key (``open_when_wide``: the sampler's ``brace_open_when_wide()``), a partner hull or a brace height;
    else the record (``required`` = wide: the open-finger stop or the draw is rejected)."""
    if not open_when_wide or partner_hull_w is None:
        return None
    heights = brace_tip_heights(gs, b_point, face_z)
    if not heights:
        return None
    a = C.unit(G.t(a_face).float())
    outward = -a
    side_h = C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), outward, dim=0))
    spreads = sorted(brace_wide_spreads(gs), reverse=True)
    end = BRACE_OPEN_END_MARGIN_M
    min_width = min(spreads) + 2.0 * end
    run = face_run_at(partner_hull_w, heights[0], outward, side_h, BRACE_OPEN_FACE_DEPTH_TOL_M)
    s_b = float(G.t(b_point)[:2].float() @ side_h[:2])
    s_push = float(G.t(push_point)[:2].float() @ side_h[:2]) if push_point is not None else s_b
    width = 0.0 if run is None else run[1] - run[0]
    rec: Dict[str, Any] = {"rule": "face width at the brace height >= the smallest open spread + 2 end margins", "height_m": round(heights[0], 4),
                           "face_width_m": round(width, 4), "min_width_m": round(min_width, 4), "wide": bool(run is not None and width >= min_width - 1e-9),
                           "push_line_from_face_centre_m": None if run is None else round(s_push - 0.5 * (run[0] + run[1]), 4),
                           "push_line_side_m": round(s_push - s_b, 4), "spread_m": None, "required": False}
    if rec["wide"]:
        rec["required"] = True
        rec["spread_m"] = next((s for s in spreads if run[0] + end <= s_push - s / 2 + 1e-9 and s_push + s / 2 <= run[1] - end + 1e-9), None)
    return rec


def open_brace_candidates(gs, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor, partner_hull_w: th.Tensor,
                           rule: Dict[str, Any], clearance: float):
    """The open-finger stops of :meth:`brace_open_rule`: the fingertips ``clearance`` before the far face, pre-shaped to the rule's
    spread straddling the push line, the hand horizontal along the brace direction - at each brace height whose cut takes the spread
    (the higher first), both wrist turns (camera up first, ``brace_wide_h-``; ``brace_wide_h``; ``_lo`` at the lower height)."""
    spread = rule.get("spread_m")
    if spread is None:
        return
    a = C.unit(G.t(a_face).float())
    side_h = C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), -a, dim=0))
    heights = brace_tip_heights(gs, b_point, face_z)
    tip_pre = float(gs.gen("push_tip_pre_dist_m", 0.05))
    shift = float(rule["push_line_side_m"])
    s_b = float(G.t(b_point)[:2].float() @ side_h[:2])
    q_open = max(0.0, min(C.FINGER_OPEN, (spread - CLOSED_TIP_SPREAD_M) / 2))
    h_brace = C.default_closing_axis(a)
    facts = {k: rule[k] for k in ("face_width_m", "min_width_m", "spread_m", "required")}
    for z in heights:
        run = face_run_at(partner_hull_w, z, -a, side_h, BRACE_OPEN_FACE_DEPTH_TOL_M)
        if run is None or not (run[0] + BRACE_OPEN_END_MARGIN_M <= s_b + shift - spread / 2 + 1e-9 and s_b + shift + spread / 2 <= run[1] - BRACE_OPEN_END_MARGIN_M + 1e-9):
            continue
        tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), z]) + side_h * shift
        for c_sign in PUSH_WRIST_TURNS:
            pose = C.fingertip_eef(tip_pt, a, c_sign * h_brace, clearance)  # c = -1: eef +y = -h, the camera (eef -x side) up
            contact = (G.t(pose[0]), G.t(pose[1]))
            kind = f"brace_wide_h{'' if c_sign > 0 else '-'}{'' if z == heights[0] else '_lo'}"
            yield IK.pre_pose(contact, tip_pre), contact, kind, {
                "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": kind, "tip_front": True, "pitch_deg": 0.0,
                "pre_dist_m": tip_pre, "contact_z": round(z, 4), "fingers_opening_m": round(q_open, 4), "stop_width_m": spread,
                "push_line_side_m": round(shift, 4), "open_when_wide": facts, "height_extra_m": requested_height_extra(gs),
                "height_extra_effective_m": round(z - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}


def touching_brace_candidates(gs, manip: Stance, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor,
                              partner_hull_w: Optional[th.Tensor] = None, push_point: Optional[th.Tensor] = None, *, open_when_wide: bool = False):
    """Brace poses of the closed left gripper at the partner's far (robot-left) face, in the order to try — the axial
    stop-brace rule instead of the former 12 mm pad-slab at the face point: (1) the pads on the face, pad centre
    ``STOP_CLEARANCE`` outside it (closing axis vertical then horizontal, either wrist turn); (2) the top-down blade
    beside the face (``tabletop.blade_standoff``: eef origin ``finger half thickness over the face height above the
    tips + STOP_CLEARANCE`` outside the face, approach pitched per ``tabletop.approach_pitch_order``, the pads' faces
    parallel to the object face). Thin partners (< ``THIN_PARTNER_HEIGHT``) try the blade first. Yields ``(pre, contact,
    kind, extras)`` with the brace record fields ``blade_half_thickness_m`` / ``stop_clearance_m`` / ``blade_standoff_m`` /
    ``pose_class``."""
    a = C.unit(G.t(a_face))  # into the face
    outward = -a
    z_axis = th.tensor([0.0, 0.0, 1.0])
    side_h = C.unit(th.cross(z_axis, outward, dim=0))
    z0, z1 = face_z
    clearance = float(gs.gen("stop_clearance_m", STOP_CLEARANCE))
    rule = brace_open_rule(gs, b_point, face_z, a_face, partner_hull_w, push_point, open_when_wide=open_when_wide)
    if rule is not None and rule["required"]:
        # generation.brace_open_when_wide on a wide face: the open-finger stop or nothing (``touching_contacts`` rejects the draw)
        yield from open_brace_candidates(gs, b_point, face_z, a_face, partner_hull_w, rule, clearance)
        return

    # owner 2026-09-18 20:48: the FINGERTIP-FRONT brace first, mirrored from the push class that made 0918-200452 - the left
    # hand horizontal along -push direction, the closed tips ``clearance`` before the partner's far face, wrist camera UP
    # (closing = -default: the camera lies on the eef's -x side, the default turn hung it at the table), the contact raised
    # to ``z0 + brace_tip_min_above_m`` so the gripper body (r 4 cm, 9.7 cm behind the tips) clears the table - at the
    # partner's mid height (0.85 on a 0.81 table) it sat on it and every brace pre-pose failed the IK. 5 cm pre.
    if bool(gs.gen("brace_tip_first", True)):
        tip_pre = float(gs.gen("push_tip_pre_dist_m", 0.05))
        heights = brace_tip_heights(gs, b_point, face_z)  # the higher first: the body further off the table
        # WIDE stop first when the rule admits it (``wide_stop_spread``): the fingers pre-shaped so both tips land on the far
        # face straddling the push line - the lever arm against the pivot a point stop cannot resist (0918-224622: the
        # partner yawed 0.8 deg about the closed tips and the pair lost contact). Camera-up wrist turn, same pose family.
        if heights and partner_hull_w is not None and bool(gs.gen("brace_wide_stop", True)):
            s_b = float(G.t(b_point)[:2] @ side_h[:2])
            s_push = float(G.t(push_point)[:2] @ side_h[:2]) if push_point is not None else s_b
            spread = wide_stop_spread(partner_hull_w, b_point, outward, side_h, s_push, spreads=brace_wide_spreads(gs))
            if spread is not None:
                q_open = max(0.0, min(C.FINGER_OPEN, (spread - CLOSED_TIP_SPREAD_M) / 2))  # per-finger joint value for that tip spread
                tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), heights[0]]) + side_h * (s_push - s_b)
                h_brace = C.default_closing_axis(a)
                pose = C.fingertip_eef(tip_pt, a, -h_brace, clearance)  # eef +y = -h: the camera (eef -x side) up
                contact = (G.t(pose[0]), G.t(pose[1]))
                yield IK.pre_pose(contact, tip_pre), contact, "brace_wide_h-", {
                    "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": "brace_wide_h-", "tip_front": True, "pitch_deg": 0.0,
                    "pre_dist_m": tip_pre, "contact_z": round(heights[0], 4), "fingers_opening_m": round(q_open, 4), "stop_width_m": spread,
                    "push_line_side_m": round(s_push - s_b, 4), "height_extra_m": requested_height_extra(gs),
                    "height_extra_effective_m": round(heights[0] - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}
        for pitch in [float(p) for p in (gs.gen("brace_tip_pitches_deg", None) or [0.0, 15.0])]:
            # pitched: the hand comes from above-left pointing down-right; the body 9.7 cm behind the tips rises 9.7·sin(pitch) cm
            th_p = math.radians(pitch)
            a_p = C.unit(a * math.cos(th_p) + th.tensor([0.0, 0.0, -math.sin(th_p)]))
            h_brace = C.default_closing_axis(a_p)
            for z_tip_brace in heights:
                tip_pt = th.tensor([float(b_point[0]), float(b_point[1]), z_tip_brace])
                for c_sign in PUSH_WRIST_TURNS:
                    pose = C.fingertip_eef(tip_pt, a_p, c_sign * h_brace, clearance)
                    contact = (G.t(pose[0]), G.t(pose[1]))
                    kind = f"brace_tip_h{'' if c_sign > 0 else '-'}{'' if pitch == 0 else f'_p{pitch:.0f}'}{'' if z_tip_brace == heights[0] else '_lo'}"
                    yield IK.pre_pose(contact, tip_pre), contact, kind, {
                        "blade_half_thickness_m": None, "stop_clearance_m": clearance, "pose_class": kind, "tip_front": True, "pitch_deg": pitch,
                        "pre_dist_m": tip_pre, "contact_z": round(z_tip_brace, 4), "height_extra_m": requested_height_extra(gs),
                        "height_extra_effective_m": round(z_tip_brace - float(b_point[2]), 4), "height_extra_clip_reason": "gripper body above the table"}
    pad_lift, pad_why = contact_height_extra(gs, "brace", pad_center_z=float(b_point[2]), band_top_z=z1)  # A/B knob
    pad_facts = {"height_extra_m": requested_height_extra(gs), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}

    def side_contacts():
        point = G.t(b_point) + outward * clearance + th.tensor([0.0, 0.0, pad_lift])
        for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
            name = TT.contact_class(manip, a, c_axis) or f"side_{suffix}"
            for c_sign in (1.0, -1.0):
                contact = IK.contact_eef(point, a, c_sign * c_axis)
                yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": clearance, "pose_class": name, "blade_half_thickness_m": None, **pad_facts}

    def blades():
        for pitch in TT.approach_pitch_order(None):
            z_tip = z0 + 0.005
            blade_lift, blade_why = contact_height_extra(gs, "blade", fingertip_z=z_tip, band_top_z=z1)
            z_tip += blade_lift  # the fingertips just above the support: the fingers span the face height
            variants = []  # (z_tip, raised)
            palm_clear = float(gs.gen("blade_palm_clear_m", 0.025))
            if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP - palm_clear:
                # the face reaches (within the palm clearance of) the palm line: the 8 cm tissue box vs the 7 cm finger section -
                # the low blade's knuckles sit 2 mm from the face top (hand gate, 0918-214155). Owner 2026-09-18 21:20: the RAISED
                # vertical blade first - tips at face top - PALM_FROM_TIP + palm clearance, the finger section on the face's top
                # part, palm / knuckles / camera above it; the left arm comes from ABOVE the partner, not across the right
                # forearm at its push goal (the horizontal brace_tip classes pass alone but collide in the two-arm solve)
                variants.append((z1 - TT.PALM_FROM_TIP + palm_clear, True))
            variants.append((z_tip, False))
            for z_tip_v, raised in variants:
                so = TT.blade_standoff((z0, z1), z_tip_v, pitch, clearance, side=False)
                if so is None:
                    continue  # the face lies entirely below the fingertips
                standoff, half = so
                a_vec = TT.pitched_approach(pitch, manip.forward)
                origin = th.tensor([float(b_point[0]), float(b_point[1]), z_tip_v + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
                name = f"blade_{pitch:.0f}{'_high' if raised else ''}"
                lift_eff, why_eff = (z_tip_v - (z0 + 0.005), "palm_clear") if raised else (blade_lift, blade_why)
                for c_sign in PUSH_WRIST_TURNS:  # the wrist camera away from the partner first
                    contact = (origin, C.eef_quat(a_vec, c_sign * outward))
                    yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {
                        "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip_v, "pose_class": name,
                        "palm_clear": raised, "height_extra_m": requested_height_extra(gs), "height_extra_effective_m": round(lift_eff, 4), "height_extra_clip_reason": why_eff}

    # blades before the side contacts: the side classes hang the camera down / need the hand across the right forearm
    order = (blades, side_contacts)
    for gen in order:
        yield from gen()


# ------------------------------------------------------------------------------------------------ the oracle's run-time transforms (called, not re-implemented)


def _oracle():
    """``(execute, actions, sidecar)`` of the oracle, imported on first use (read-only: the transforms below CALL them)."""
    from ispatialgym.oracle import actions as AC
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle import sidecar as SC

    return EX, AC, SC


def oracle_config():
    """The oracle's default ``ExecuteConfig`` (the run-time knobs the transforms read: ``push_pre_dist_m``, ``brace_touch_stop_m`` ...)."""
    return _oracle()[0].ExecuteConfig()


def _sc_pose(pose):
    SC = _oracle()[2]
    p, q = G.t(pose[0]).float(), G.t(pose[1]).float()
    return SC.Pose([float(v) for v in p.tolist()], [float(v) for v in q.tolist()])


def executed_push_pre(kind: Optional[str], init, pre, *, arc: bool = False, extras: Optional[Dict[str, Any]] = None, cfg=None):
    """The pre-contact the oracle reaches a recorded push at: ``execute.push_lowered_pre`` on the record (``pose_class`` = ``kind``; an arc has a
    ``mid``) - the recorded ``pre`` lowered to ``push_pre_dist_m`` (0.04 m) before ``init`` along the approach when it lies farther, else
    ``pre`` itself. Returns ``(pos, quat)`` tensors."""
    EX = _oracle()[0]
    cfg = cfg or oracle_config()
    contact = SimpleNamespace(mid=object() if arc else None, extra=dict(extras or {}, pose_class=kind))
    low = EX.push_lowered_pre(contact, _sc_pose(pre), _sc_pose(init), cfg)
    if low is None:
        return G.t(pre[0]).float(), G.t(pre[1]).float()
    return th.tensor(low.position, dtype=th.float32), th.tensor(low.orientation, dtype=th.float32)


def recorded_pre_dist(extras: Optional[Dict[str, Any]]) -> float:
    """The pre-contact distance of a touching push record (``IORSSmall._touching_push_choice``'s chain: ``pre_dist_m``, else ``IK.PRE_DIST``)."""
    return float((extras or {}).get("pre_dist_m") or IK.PRE_DIST)


def executed_push_pre_dist(kind: Optional[str], extras: Optional[Dict[str, Any]], *, arc: bool = False, cfg=None) -> float:
    """How far before the init along the approach the oracle reaches a touching push's pre-contact (:func:`executed_push_pre` of the record's
    pre at :func:`recorded_pre_dist`)."""
    init = (th.zeros(3), C.IDENTITY_QUAT.clone())
    pre = IK.pre_pose(init, recorded_pre_dist(extras))
    low = executed_push_pre(kind, init, pre, arc=arc, extras=extras, cfg=cfg)
    return round(float(th.linalg.norm(low[0] - init[0])), 6)


#: the executors of a push record (``execute.EXECUTORS``): ``push`` = ``execute_push`` (a ``Push``: touching, axial, block design, the
#: push-under, the wall push), ``base_push`` = ``execute_base_push`` (a ``BasePush``: ``kind: "base_push"``, nextto_large / touching_large_01)
PUSH_EXECUTORS = ("push", "base_push")


def pre_pose_rule(kind: Optional[str], extras: Optional[Dict[str, Any]] = None, *, arc: bool = False, executor: str = "push",
                  recorded_m: Optional[float] = None, cfg=None) -> Dict[str, Any]:
    """Owner decision 6 (2026-11-07): the pre-pose the oracle EXECUTES for a push record, as a schema field. ``execute_push`` puts a straight
    push's ``init`` into its targets (no ``mid``) and calls ``push_lowered_pre``: the recorded pre-contact is replaced by one
    ``push_pre_dist_m`` (0.04 m) before the init along the recorded approach unless the record has a ``mid`` (an arc: the axial push, the
    push-under, the wall push), its ``pose_class`` is a pitched >= 30 deg blade, or the recorded pre lies within ``push_pre_dist_m`` + 5 mm of
    the init; ``execute_base_push`` (a ``BasePush``) never lowers. ``rule`` = ``"lowered_to_push_pre_dist_m"`` | ``"recorded"``, with the
    distance (m before the init along the approach) and the condition that decided it."""
    cfg = cfg or oracle_config()
    rec = float(recorded_pre_dist(extras) if recorded_m is None else recorded_m)
    pdist = cfg.push_pre_dist_m
    if executor == "base_push":
        why, lowered = "BasePush: execute_base_push reaches the recorded pre-contact (no push_lowered_pre)", False
    elif executor != "push":
        raise ValueError(f"pre_pose_rule: executor {executor!r} (one of {PUSH_EXECUTORS})")
    elif arc:
        why, lowered = "the record has a mid (an arc / waypoint push): push_lowered_pre keeps the recorded pre", False
    elif pdist is None:
        why, lowered = "ExecuteConfig.push_pre_dist_m is None", False
    elif pitched_push_class(kind):
        why, lowered = f"pose_class {kind!r} is a pitched >= 30 deg blade: push_lowered_pre keeps the recorded pre", False
    elif rec <= float(pdist) + PUSH_PRE_SLACK_M:
        why, lowered = f"the recorded pre ({rec:.3f} m) is within push_pre_dist_m + {PUSH_PRE_SLACK_M} m of the init", False
    else:
        why, lowered = f"straight push (no mid), recorded pre {rec:.3f} m > push_pre_dist_m + {PUSH_PRE_SLACK_M} m: lowered", True
    return {"rule": "lowered_to_push_pre_dist_m" if lowered else "recorded", "distance_m": round(float(pdist) if lowered else rec, 5),
            "recorded_m": round(rec, 5), "condition": why, "executor": executor,
            "source": "execute.push_lowered_pre (called by execute_push for a straight Push)" if executor == "push" else "execute.execute_base_push"}


#: ``push_lowered_pre``'s slack (its ``<= push_pre_dist_m + 0.005`` test)
PUSH_PRE_SLACK_M = 0.005


def pitched_push_class(kind: Optional[str]) -> bool:
    """``execute._pitched_push_class`` of a record's ``pose_class`` (a pitched blade of 30 deg or steeper keeps its recorded pre)."""
    return bool(_oracle()[0]._pitched_push_class(SimpleNamespace(extra={"pose_class": kind})))


def contact_fingers(extras: Optional[Dict[str, Any]], kind: str = "push") -> float:
    """The finger joints (m per finger) the oracle holds a push / brace at: ``actions.push_is_open`` / ``brace_is_open`` of the record - its
    ``fingers_opening_m`` - else closed (``execute_push`` / ``execute_brace``). A brace: :func:`tabletop.brace_fingers` (the one rule of the
    brace checks, i7-contactbrace 2026-11-08)."""
    if kind != "push":
        return TT.brace_fingers(extras)
    AC = _oracle()[1]
    rec = SimpleNamespace(extra=dict(extras or {}))
    return float(rec.extra["fingers_opening_m"]) if AC.push_is_open(rec) else float(IK.FINGER_CLOSED)


def brace_stop_rule(cfg=None) -> Dict[str, Any]:
    """The oracle's brace leg (``execute_brace``): straight pre -> contact, stopped once the braced object moved ``brace_touch_stop_m``, then
    backed off ``brace_touch_backoff_m`` along the approach."""
    cfg = cfg or oracle_config()
    return {"leg": "straight pre -> contact", "touch_stop_m": float(cfg.brace_touch_stop_m), "backoff_m": float(cfg.brace_touch_backoff_m)}


def digest(obj: Any, n: int = 16) -> str:
    """sha1 of the canonical JSON of ``obj`` (sorted keys), first ``n`` hex digits - the schema / pose-set identities."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()[:n]


# ------------------------------------------------------------------------------------------------ the shared primitives (re-exported)

#: the pose primitives every builder composes (``conventions`` / ``v2.ik`` / ``v2.tabletop``) - one vocabulary for the samplers and libraries
fingertip_eef = C.fingertip_eef
contact_eef = C.contact_eef
default_closing_axis = C.default_closing_axis
eef_quat = C.eef_quat
pre_pose = IK.pre_pose
blade_standoff = TT.blade_standoff
pitched_approach = TT.pitched_approach


# ------------------------------------------------------------------------------------------------ the floor pushes (iors_large: push-under / touching_large)


def floor_blade_height(height: float, floor_z: float, pitch_deg: float, frac: float = 0.5, clear: float = 0.015, cap: float = 0.05) -> Tuple[float, float]:
    """``(z_push, z_tip)`` of the floor blade (``IORSLarge.push_height``): the fingertips at ``frac`` (``push_tip_frac``) of the target's height
    above the floor, at least ``clear`` (``push_tip_clearance_m``) and at most ``cap`` up; the eef origin ``FINGERTIP_LENGTH cos(pitch)`` above."""
    z_tip = floor_z + max(clear, min(frac * height, cap))
    drop = C.FINGERTIP_LENGTH * math.cos(math.radians(float(pitch_deg)))
    return z_tip + drop, z_tip


def floor_push_chain(drop_xy, goal_xy, d2, z: float, offset: float, goal_extra: float, quat, *, pre: Tuple[Any, ...] = ("approach",),
                     shift: Optional[th.Tensor] = None) -> List[Tuple[th.Tensor, th.Tensor]]:
    """The straight floor push ``[pre, start, mid, goal]`` (``IORSLarge._push_under_chain`` / ``_push_under_flat_chain`` /
    ``_push_under_pitched_chain``): the eef at height ``z``, ``offset`` behind the drop cell along the push ``d2`` (the contact offset + the
    hand's stand-off), the goal ``offset - goal_extra`` behind the goal cell (``push_goal_extra_m``: the oracle's leg stops short of its last
    waypoint), the mid-point; ``shift`` (the palm's stand-off) moves the three back; ``pre``: ``("approach",)`` = ``pre_pose`` (PRE_DIST back
    along the approach), ``("back", d)`` = ``d`` behind the start along the push, ``("back_up", d, up)`` = ``d`` behind and ``up`` above it."""
    d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
    start = th.tensor([float(drop_xy[0]), float(drop_xy[1]), z]) - d3 * offset
    goal = th.tensor([float(goal_xy[0]), float(goal_xy[1]), z]) - d3 * (offset - goal_extra)
    mid = 0.5 * (start + goal)
    if shift is not None:
        start, mid, goal = start - shift, mid - shift, goal - shift
    if pre[0] == "approach":
        p = C.pre_pose((start, quat))
    elif pre[0] == "back":
        p = (start - d3 * float(pre[1]), quat)
    elif pre[0] == "back_up":
        p = (start - d3 * float(pre[1]) + th.tensor([0.0, 0.0, float(pre[2])]), quat)
    else:
        raise ValueError(f"floor_push_chain: pre rule {pre!r}")
    return [p, (start, quat), (mid, quat), (goal, quat)]


# ------------------------------------------------------------------------------------------------ the base push (iors_large_nextto / touching_large)


def base_push_approach(direction_xy, pitch_deg: float) -> th.Tensor:
    """Approach axis along the horizontal push ``direction`` tilted down by ``pitch_deg`` (fingertips forward-down onto the face;
    ``iors_large_nextto.pitched_approach``)."""
    d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])], dtype=th.float32)
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    p = math.radians(float(pitch_deg))
    return th.tensor([math.cos(p) * float(d[0]), math.cos(p) * float(d[1]), -math.sin(p)])


def base_push_hand_poses(point, direction_xy, pitch_deg: float, wrist: float, pre_m: float) -> Tuple[G.Pose, G.Pose]:
    """``(pre, contact)`` eef poses of the closed hand of the base push (``iors_large_nextto.push_hand_poses``): the fingertips on ``point``
    with the approach of :func:`base_push_approach`, the pads closing horizontally across the push (``wrist`` +-1 = the two wrist turns), the
    pre-contact ``pre_m`` back along the horizontal push."""
    a = base_push_approach(direction_xy, pitch_deg)
    d = th.tensor([float(direction_xy[0]), float(direction_xy[1]), 0.0])
    d = d / max(float(th.linalg.norm(d)), 1e-9)
    perp = th.tensor([-float(d[1]), float(d[0]), 0.0]) * float(wrist)
    q = C.eef_quat(a, perp)
    origin = th.as_tensor(point, dtype=th.float32) - C.FINGERTIP_LENGTH * a
    return (origin - d * float(pre_m), q), (origin, q)


# ------------------------------------------------------------------------------------------------ the wall contacts (alignment_wall: world_vertical)


def wall_fist_axes(normal, closing, tilt: float = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """(approach a, closing c) of a wall contact that acts in the wall plane (``alignment_wall.wall_fist_axes``): the fingers point into the
    wall (a = -n), turned by ``tilt`` (rad) about c, and the pads close along ``closing`` (in the wall plane) - the outer side of a finger
    (normal +-c) meets the target's edge, so the contact normal lies in the wall plane."""
    n = C.unit(th.as_tensor(normal, dtype=th.float32))
    c = th.as_tensor(closing, dtype=th.float32)
    c = C.unit(c - n * float(th.dot(c, n)))
    k = C.unit(th.cross(c, n, dim=0))
    a = C.unit(-n * math.cos(tilt) + k * math.sin(tilt))
    return a, c


def wall_contact_pose(point, normal, direction, face: float, gap: float = 0.0, tilt: float = 0.0, turn: float = 1.0) -> Tuple[th.Tensor, th.Tensor]:
    """eef pose of a wall contact (push or brace) acting along ``direction`` (in the wall plane, toward the target;
    ``alignment_wall.wall_contact_pose``): the outer side of a finger (``face`` from the eef axis) stands ``gap`` before ``point``, the
    fingertips at ``point``'s depth (on the target's mid-plane), the pads closing along ``turn`` x direction."""
    d = th.as_tensor(direction, dtype=th.float32)
    a, c = wall_fist_axes(normal, d * (1.0 if turn >= 0 else -1.0), tilt)
    tip = th.as_tensor(point, dtype=th.float32) - C.unit(d) * (float(face) + float(gap))
    return tip - C.FINGERTIP_LENGTH * a, C.eef_quat(a, c)


# ------------------------------------------------------------------------------------------------ the axial push (alignment_axial)

#: the closed fingertips this far above the axial target's top at the floor of the push height (``alignment_axial.PUSH_TIP_CLEARANCE``)
AXIAL_PUSH_TIP_CLEARANCE_M = 0.003


def axial_push_height(top_z: float, object_height: float, tip_clearance: float = AXIAL_PUSH_TIP_CLEARANCE_M) -> float:
    """The axial push's eef origin height (``AlignmentAxial._push_height`` before its A/B knob): the generic pinch's ``0.6 h`` above the top
    with the floor at the fingertip length + ``tip_clearance`` (0.0218 m: the closed tips 3 mm above the top of the thinnest targets)."""
    return float(top_z) + max(0.6 * float(object_height), C.FINGERTIP_LENGTH + float(tip_clearance))


def axial_blade_geometry(z_push: float, face_z: Tuple[float, float], pitch: float, clearance: float, side: bool = False) -> Optional[Tuple[float, float, float]]:
    """``(stand-off, finger half thickness, tip z)`` of the axial push's top-down blade beside the target's face at the push height
    ``z_push`` (``AlignmentAxial._blade_geometry``): the fingertips ``FINGERTIP_LENGTH cos(pitch)`` below the eef origin, the half thickness
    the finger profile's maximum over the face height above the tips (``tabletop.blade_standoff``); None when the face reaches the palm."""
    z_tip = z_push - C.FINGERTIP_LENGTH * math.cos(math.radians(float(pitch)))
    so = TT.blade_standoff(face_z, z_tip, pitch, clearance, side=side)
    if so is None:
        return None
    return so[0], so[1], z_tip


def extreme_point(pts_w: th.Tensor, direction, near=None, long_w=None, window: float = 0.04) -> th.Tensor:
    """The hull point (world) with the largest projection on ``direction`` (xy) - with ``near`` / ``long_w`` (the long axis, world xy) only
    the points within ``window`` of that end along the long axis when at least 3 are (``AlignmentAxial._extreme_point``)."""
    pts = pts_w
    dxy = th.tensor([float(direction[0]), float(direction[1])])
    dxy = dxy / th.linalg.norm(dxy)
    if near is not None and long_w is not None:
        along = (pts[:, :2] - TT._t(near)[:2]) @ long_w
        sel = along.abs() <= window
        if int(sel.sum()) >= 3:
            pts = pts[sel]
    proj = pts[:, :2] @ dxy
    return pts[int(th.argmax(proj))]


#: the axial stop brace (``alignment_axial.STOP_CLEARANCE`` / ``THIN_TARGET_HEIGHT``): the pads' / blade's outer face this far outside the
#: goal hull; below this target height the blade is tried before the side contacts
AXIAL_STOP_CLEARANCE_M = 0.005
AXIAL_THIN_TARGET_HEIGHT_M = 0.04
#: the stop face points' fractions of the face span from the pushed end's arrival (``AlignmentAxial._stop_face_points``)
AXIAL_STOP_FACE_FRACTIONS = (0.08, 0.2, 0.35, 0.5, 0.65, 0.8)


def axial_stop_face_points(hull: th.Tensor, goal_pose, d_stop: th.Tensor, near: th.Tensor, z: float, fractions=AXIAL_STOP_FACE_FRACTIONS) -> List[th.Tensor]:
    """Points of the goal hull's face extreme along ``d_stop`` (projection within 3 mm of the maximum) at the given fractions of the face's
    span from the end nearest ``near`` (the pushed end's arrival) to the far end, at height ``z``; a face shorter than 6 cm yields its midpoint
    only (``AlignmentAxial._stop_face_points``)."""
    pts = C.transform_points(goal_pose, hull)
    dxy = th.tensor([float(d_stop[0]), float(d_stop[1])])
    dxy = dxy / th.linalg.norm(dxy)
    proj = pts[:, :2] @ dxy
    face = pts[proj >= float(proj.max()) - 0.003]
    f = TT._t(near)[:2]
    dist = th.linalg.norm(face[:, :2] - f, dim=1)
    p_near, p_far = face[int(th.argmin(dist))], face[int(th.argmax(dist))]
    span = float(th.linalg.norm(p_far[:2] - p_near[:2]))
    if span < 0.06:
        mid = 0.5 * (p_near + p_far)
        return [th.tensor([float(mid[0]), float(mid[1]), float(z)])]
    out = []
    for t in fractions:
        p = p_near + (p_far - p_near) * t
        out.append(th.tensor([float(p[0]), float(p[1]), float(z)]))
    return out


def axial_stop_geometry(hull: th.Tensor, com_local: th.Tensor, pose, end: th.Tensor, com: th.Tensor, spin: float, yaw_goal: float, z_brace: float):
    """§6.10 stop edge of the pushed ``end`` (``AlignmentAxial._stop_geometry``): the goal pose (the centre of mass fixed, yaw = aligned), the
    direction the end moves in at the goal (``d_stop``) and the stop points on the goal hull's face along it at the brace height
    (:func:`axial_stop_face_points`, nearest the pushed end first). Returns ``(goal_pose, stop_pts, d_stop)``."""
    z_axis = th.tensor([0.0, 0.0, 1.0])
    q_goal = C.quat_from_yaw(yaw_goal)
    goal_pose = (com - C.quat_rotate(q_goal, com_local), q_goal)
    r_goal = C.quat_rotate(q_goal, C.quat_rotate(C.qinv(pose[1]), end - TT._t(pose[0]))) + goal_pose[0] - com
    d_stop = spin * th.cross(z_axis, th.tensor([float(r_goal[0]), float(r_goal[1]), 0.0]), dim=0)
    d_stop = d_stop / th.linalg.norm(d_stop)
    stop_pts = axial_stop_face_points(hull, goal_pose, d_stop, near=com + r_goal, z=z_brace)
    return goal_pose, stop_pts, d_stop


def axial_brace_candidates(manip, hull: th.Tensor, stop_pts: List[th.Tensor], d_stop: th.Tensor, passing: Optional[List[str]], pose_init, com: th.Tensor,
                           d_goal: float, *, z_push: float, face_z: Tuple[float, float], height: float, stop_clearance: float = AXIAL_STOP_CLEARANCE_M,
                           thin_height: float = AXIAL_THIN_TARGET_HEIGHT_M, skips: Optional[Dict[str, int]] = None):
    """The axial stop brace's poses of the closed left gripper at the stop face (``AlignmentAxial._brace_candidates``; ``stop_pts`` on the goal
    hull, the pushed end's corner side first), in the order to try: the contact classes with the pads on the face (approach -d_stop into the
    face, pad centre ``stop_clearance`` outside it; closing vertical ``_v`` then horizontal ``_h``, either wrist turn; the classes in
    ``passing`` first) and the top-down blade beside the stop face (straight down / pitched toward ``manip.forward``, either wrist turn;
    :func:`axial_blade_geometry` at the push height ``z_push``), each clear of the target's sweep from ``pose_init`` to the goal
    (``tabletop.blade_sweep_clear``); a target under ``thin_height`` tries the blade first (a side contact at 3/4 of its height puts the wrist
    on the table). ``skips`` counts the sweep / palm skips. Yields ``(pre, contact, kind, extras)``."""
    skips = {} if skips is None else skips
    z_axis = th.tensor([0.0, 0.0, 1.0])
    a = -d_stop  # into the stop face
    side_h = C.unit(th.cross(z_axis, d_stop, dim=0))
    contact_classes = []
    for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
        name = TT.contact_class(manip, a, c_axis)
        contact_classes.append((name or f"side_{suffix}", c_axis))
    if passing:
        contact_classes.sort(key=lambda nc: 0 if nc[0] in passing else 1)
    r_side = TT.finger_half_thickness(0.0, side=True) or 0.007

    def side_contacts():
        for i, stop_pt in enumerate(stop_pts):
            point = stop_pt + d_stop * stop_clearance  # pad centre 5 mm outside the goal face
            # the pads' footprint (a finger's width across, a finger's thickness deep) must be clear of the sweep too
            ok, gap = TT.blade_sweep_clear(hull, pose_init, com[:2], d_goal, (point + d_stop * r_side)[:2], r_side, stop_clearance)
            if not ok:
                skips["sweep"] = skips.get("sweep", 0) + 1
                continue
            for name, c_axis in contact_classes:
                for c_sign in (1.0, -1.0):  # either wrist turn of the same pad contact
                    brace = C.contact_eef(point, a, c_sign * c_axis)
                    yield C.pre_pose(brace), brace, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": stop_clearance, "pose_class": name,
                                                                                           "stop_face_fraction": i, "sweep_min_gap_m": gap}

    def blades():
        for pitch in TT.approach_pitch_order(None):
            geo = axial_blade_geometry(z_push, face_z, pitch, stop_clearance)
            if geo is None:
                skips["palm"] = skips.get("palm", 0) + 1
                continue
            standoff, half, z_tip = geo
            a_vec = TT.pitched_approach(pitch, manip.forward)
            for i, stop_pt in enumerate(stop_pts):
                origin = th.tensor([float(stop_pt[0]), float(stop_pt[1]), z_push]) + d_stop * standoff
                # the blade must not meet the rotating target before the goal (the face rising through it, a corner)
                ok, gap = TT.blade_sweep_clear(hull, pose_init, com[:2], d_goal, origin[:2], max(half, r_side), stop_clearance)
                if not ok:
                    skips["sweep"] = skips.get("sweep", 0) + 1
                    continue
                for c_sign in (1.0, -1.0):
                    brace = (origin, C.eef_quat(a_vec, c_sign * d_stop))
                    yield C.pre_pose(brace), brace, f"blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", {
                        "blade_half_thickness_m": half, "stop_clearance_m": stop_clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                        "sweep_min_gap_m": gap, "pose_class": f"blade_{pitch:.0f}", "stop_face_fraction": i}

    order = (blades, side_contacts) if height < thin_height else (side_contacts, blades)
    for gen in order:
        yield from gen()


def axial_push_chain(side_pt: th.Tensor, d: th.Tensor, half: float, approach: th.Tensor, closing: th.Tensor, c_sign: float, centre: th.Tensor,
                     d_goal: float) -> List[Tuple[th.Tensor, th.Tensor]]:
    """The axial push ``[pre, init, mid, goal]`` of one class / wrist turn (``AlignmentAxial._sample_with``): the closed blade ``half`` (its
    half thickness at the pitch) before the face point ``side_pt`` along the push ``d``, approach ``approach``, closing ``c_sign x closing``,
    the arc about ``centre`` (the centre of mass at the push height) through ``d_goal`` (``conventions.arc_waypoints``, 2 waypoints), the
    pre-contact ``pre_pose`` of the init (an arc keeps it: ``execute.push_lowered_pre``)."""
    quat = C.eef_quat(approach, c_sign * closing)
    p_init = (side_pt - d * half, quat)
    arc = C.arc_waypoints(p_init, centre, th.tensor([0.0, 0.0, 1.0]), d_goal, n=2)
    return [C.pre_pose(p_init), p_init, arc[0], arc[1]]


# ------------------------------------------------------------------------------------------------ the contact SCHEMA (push / brace library entries)

CONTACT_SCHEMA = "isg-contact/1"
CONTACT_KINDS = ("push", "brace")
#: the arm roles of an entry: the working (pushing) arm, the bracing arm, or the base push's two hands
ARM_ROLES = ("work", "brace", "both")


def _mat(pose) -> np.ndarray:
    if isinstance(pose, np.ndarray):
        return pose.astype(float)
    p, q = G.t(pose[0]).double().numpy(), G.t(pose[1]).double().numpy()
    x, y, z, w = (float(v) for v in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _pose_dict(T: np.ndarray) -> Dict[str, List[float]]:
    from ispatialgym.grasping import schema as S

    d = S.mat_pose(T)
    q = d["orientation_xyzw"]
    if q[3] < 0:
        q = [-v for v in q]
    return {"position": [round(float(v), 6) for v in d["position"]], "orientation_xyzw": [round(float(v), 7) for v in q]}


def _pose_of(d: Dict[str, Any]) -> Tuple[th.Tensor, th.Tensor]:
    return th.tensor(d["position"], dtype=th.float32), th.tensor(d["orientation_xyzw"], dtype=th.float32)


def _jsonable(v):
    if isinstance(v, th.Tensor):
        v = v.tolist()
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    return v


def contact_entry(kind: str, cls: str, family: str, arm_role: str, T_world_obj, contact, pre, extras: Optional[Dict[str, Any]] = None, *,
                  face_rule: str = "", direction=None, stroke_m: Optional[float] = None, arc: bool = False, support_z: Optional[float] = None,
                  object_height: Optional[float] = None, pitch_deg: Optional[float] = None, wrist_turn: Optional[float] = None, executor: str = "push",
                  cfg=None) -> Dict[str, Any]:
    """One schema entry (``CONTACT_SCHEMA``) in the OBJECT frame from a record builder's world output (the object at ``T_world_obj``):
    ``kind`` (push | brace) and ``cls`` (the builder's class label), ``family`` (the builder), ``arm_role``; the contact eef pose, its
    approach (eef +z), closing axis (eef +y), pitch and wrist turn; the fingertip point; the finger state AS EXECUTED (:func:`contact_fingers`);
    the height rule (the fingertip / eef height above the support, absolute and as a fraction of the object's height); the pre-contact as
    recorded and AS EXECUTED (a push: :func:`executed_push_pre`, ``pre_pose_rule`` = :func:`pre_pose_rule` (owner decision 6); a base push /
    a brace: the recorded one, and :func:`brace_stop_rule`); a push's direction
    and stroke (``goal = contact + stroke x direction``); the record's extras."""
    if kind not in CONTACT_KINDS or arm_role not in ARM_ROLES:
        raise ValueError(f"contact_entry: kind {kind!r} / arm role {arm_role!r}")
    extras = dict(extras or {})
    To = _mat(T_world_obj)
    Ti = np.linalg.inv(To)
    Tc_w = _mat(contact)
    Tc = Ti @ Tc_w
    Tp = Ti @ _mat(pre)
    if kind == "push" and executor == "push":
        low = executed_push_pre(cls, contact, pre, arc=arc, extras=extras, cfg=cfg)
        Tpe = Ti @ _mat(low)
    else:
        Tpe = Tp  # a brace, a base push: the recorded pre-contact
    tip_w = Tc_w[:3, 3] + C.FINGERTIP_LENGTH * Tc_w[:3, 2]
    height_rule: Dict[str, Any] = {"of": "fingertip plane centre above the support"}
    if support_z is not None:
        h_abs = float(tip_w[2]) - float(support_z)
        height_rule.update(absolute_m=round(h_abs, 5), fraction=None if not object_height else round(h_abs / float(object_height), 4))
    if extras.get("contact_height_m") is not None:
        height_rule["rule_height_m"] = round(float(extras["contact_height_m"]), 5)
    entry: Dict[str, Any] = {
        "schema": CONTACT_SCHEMA, "kind": kind, "class": cls, "family": family, "arm_role": arm_role, "face_rule": face_rule,
        "T_object_eef": _pose_dict(Tc), "contact_point": [round(float(v), 6) for v in (Ti[:3, :3] @ tip_w + Ti[:3, 3])],
        "approach": [round(float(v), 6) for v in Tc[:3, 2]], "closing": [round(float(v), 6) for v in Tc[:3, 1]],
        "pitch_deg": None if pitch_deg is None else float(pitch_deg), "wrist_turn": None if wrist_turn is None else float(wrist_turn),
        "fingers": {"state": "open" if contact_fingers(extras, kind) > 0 else "closed", "opening_m": round(contact_fingers(extras, kind), 5)},
        "height_rule": height_rule,
        "pre": {"recorded": _pose_dict(Tp), "executed": _pose_dict(Tpe), "recorded_m": round(float(np.linalg.norm(Tp[:3, 3] - Tc[:3, 3])), 5),
                "executed_m": round(float(np.linalg.norm(Tpe[:3, 3] - Tc[:3, 3])), 5),
                "rule": "execute.push_lowered_pre" if kind == "push" else "recorded (execute_brace: straight pre -> contact)"},
        "pre_pose_rule": (pre_pose_rule(cls, extras, arc=arc, executor=executor, recorded_m=float(np.linalg.norm(Tp[:3, 3] - Tc[:3, 3])), cfg=cfg) if kind == "push"
                          else {"rule": "recorded", "distance_m": round(float(np.linalg.norm(Tp[:3, 3] - Tc[:3, 3])), 5), "executor": "brace",
                                "condition": "execute_brace reaches the recorded pre-contact, then the straight leg to the contact", "source": "execute.execute_brace"}),
        "push": None, "stop": brace_stop_rule(cfg) if kind == "brace" else None, "extras": _jsonable(extras),
        # the object's pose in the frame the record builder ran in: the extras' world values (fingertip_z, contact_z, contact_point) are in it
        "T_rule_object": _pose_dict(To)}
    if kind == "push":
        dv = np.asarray(_jsonable(direction if direction is not None else [0.0, 0.0, 0.0]), dtype=float)
        dv = Ti[:3, :3] @ (dv / max(float(np.linalg.norm(dv)), 1e-12))
        entry["push"] = {"direction": [round(float(v), 6) for v in dv], "stroke_m": None if stroke_m is None else round(float(stroke_m), 5), "arc": bool(arc)}
    entry["digest"] = digest({k: v for k, v in entry.items() if k != "digest"})
    return entry


def entry_world(entry: Dict[str, Any], T_world_obj) -> Dict[str, Any]:
    """A schema entry at an object pose: world ``{"contact", "pre", "pre_executed", "goal", "direction"}`` (``goal``/``direction`` for a push
    with a stroke), poses as ``(pos, quat)`` tensors."""
    To = _mat(T_world_obj)

    def w(d):
        from ispatialgym.grasping import schema as S

        T = To @ S.pose_of(d)
        return th.tensor(T[:3, 3], dtype=th.float32), th.tensor(S.mat_to_quat(T[:3, :3]), dtype=th.float32)

    out = {"contact": w(entry["T_object_eef"]), "pre": w(entry["pre"]["recorded"]), "pre_executed": w(entry["pre"]["executed"]), "goal": None, "direction": None}
    push = entry.get("push")
    if push is not None:
        dv = th.tensor(To[:3, :3] @ np.asarray(push["direction"], dtype=float), dtype=th.float32)
        out["direction"] = dv
        if push.get("stroke_m") is not None:
            out["goal"] = (out["contact"][0] + dv * float(push["stroke_m"]), out["contact"][1])
    return out
