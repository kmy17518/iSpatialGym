"""The overhang pick (owner 2026-10-13 ~03:15 item 4; ``generation.pick_overhang_push``): a flat object whose library grasp puts a
finger below the object's bottom (inside_large#03's plate: grasp_5 pinches the rim from outside, the lower finger under the lip) is
not picked where it lies. The hand first pushes it along the support toward the support edge the pick stance stands at, until it
overhangs that edge far enough for the grasp's fingers - their closing sweep included - to stay in the free space beyond the edge
below the support top, while its centre of mass stays a margin inside; then the library grasp is taken at the overhang. Pure
helpers (no simulator); ``IORSLarge._overhang_pick`` drives them.

Frames: ``d`` is the horizontal push direction (unit, world xy; toward the edge), "along" coordinates are measured along it from
the object's centre, "lateral" ones along ``lateral_unit(d)``; heights are world z.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.robot import conventions as C

#: ``contacts.<slot>`` of the push before the pick (``contacts.push`` stays free for the relation's own push: push-under / touching)
CONTACT_SLOT = "push_pick"
#: ``contacts.push_pick.kind`` (the oracle's residual / stop rule and the program check's world advance key on it)
KIND = "pick_overhang"
#: the open gripper's opening per finger (m) when the push is made with the open hand
OPEN_FINGERS_M = C.FINGER_OPEN
#: the finger envelope beyond the pad faces (``iors_small.FINGER_THICKNESS`` / ``FINGER_HALF_WIDTH``, conservative)
FINGER_THICKNESS = 0.012
FINGER_HALF_WIDTH = 0.012
#: the closed fingers' half extent across the closing axis at the tip (``tabletop.FINGER_SIDE_PROFILE``'s first section): the closed
#: hand's contact band on the push line
CLOSED_HALF_WIDTH = 0.007
#: the push fingers' section below the palm (``tabletop.PALM_FROM_TIP``): object points above it are not touched by the fingers
FINGER_SECTION_M = 0.07
#: the closed fingers' half thickness along the closing axis at the tip (``tabletop.FINGER_PROFILE``'s first section)
CLOSED_HALF_THICKNESS = 0.0084
#: the edge probe: rays every this (m) along the push line, up to this far from the object's centre
EDGE_PROBE_STEP_M = 0.005
EDGE_PROBE_MAX_M = 0.6

DEFAULTS: Dict[str, Any] = {
    "mode": "fallback",               # "fallback": the direct pick first, the overhang pick when it fails; "only": the overhang pick alone
    "fingers": ["open", "closed"],    # the push hand(s) in order: the open fingers straddle the far rim, the closed fingers are one blade
    "pitches_deg": [0.0, 20.0, 40.0],  # the push hand leaned from straight down toward the robot's forward (the fingertips away from the object)
    "tip_m": 0.006,                   # the fingertips above the support top during the push
    "contact_clearance_m": 0.01,      # the push starts this far behind the first contact
    "finger_clearance_m": 0.01,       # the grasp's fingers (closing sweep) this far beyond the edge wherever they are below the support top
    "com_margin_m": 0.05,             # the object's centre stays this far inside the edge after the push
    "distance_step_m": 0.01,          # the push-distance ladder
    "max_distances": 4,               # at most this many rungs are IK-checked per pair
    "max_push_m": 0.30,               # never push farther
    "support_touch_m": 0.005,         # the push fingers may sweep the support this deep (0 = never touch it)
    "settle_max_slide_m": 0.01,       # the pushed object at rest after the settle: planar slide, drop (not falling), tilt, turn
    "settle_max_drop_m": 0.005,
    "settle_max_tilt_deg": 3.0,
    "settle_max_yaw_deg": 5.0,
    "yaw_free": True,                 # a body of revolution about z (orientation class zinf) is grasped at the recorded post-push yaw
    "rank_b": True,                   # the B candidates that suit the overhang pick (a push ladder, the push's start in reach) coupled first
    "pre_min_ahead_m": 0.0,           # ... and the grasp's pre-grasp at the overhang at least this far ahead of the base (0 = not checked: at
                                      # trunk (0, -0.92) the arm reaches a pre-grasp 0.24 m ahead, sim check 17838534)
    "grasp_back_offs_m": [0.10, 0.20],  # the grasp failing at B: tried from B backed off this far along its axis (stance "pick"; [] = B only)
    "grasp_trunks": [],               # ... at B's trunk, then these [translate, tilt]
    # i7-overhang 2026-11-07 (the v2 OVERHANG rest: the grasp library's entry settled overhanging a slab edge, its overhang map): the object's
    # INIT cells the overhang map's band view lays out behind each feasible post-push cell (the demos pushed 0.160 m; +-0.04) and the v2
    # overhang rest's tag when the library holds several (None: the first)
    "v2_init_push_m": [0.12, 0.16, 0.20],
    "v2_rest": None,
    # ... and the finger rule at the v2 rest: None = the library's (the grasp validated at that rest, its exact finger hulls against the slab
    # box within the policy's 2 mm); a number = also the finger-box sweep rule (finger_clearance_m's) at this clearance
    "v2_finger_clearance_m": None,
}
#: the grasp stance of the two-stance overhang pick (B backed off along its axis)
PICK_STANCE = "pick"
MODES = ("fallback", "only")
FINGER_MODES = ("open", "closed")


def options(value: Any) -> Optional[Dict[str, Any]]:
    """``generation.pick_overhang_push``: None / False / "off" -> None (off); True / "on" -> :data:`DEFAULTS`; a dict -> the defaults
    with its keys. ``ValueError`` for an unknown key or an invalid value."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("off", "false", "0", "no", ""):
            return None
        if s not in ("on", "true", "1", "yes"):
            raise ValueError(f"pick_overhang_push {value!r}: expected true / false or a dict")
        value = True
    if value is True:
        out = dict(DEFAULTS)
    elif isinstance(value, dict):
        unknown = sorted(set(value) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"pick_overhang_push: unknown keys {unknown} (known: {sorted(DEFAULTS)})")
        out = {**DEFAULTS, **value}
    else:
        raise ValueError(f"pick_overhang_push {value!r}: expected true / false or a dict")
    if out["mode"] not in MODES:
        raise ValueError(f"pick_overhang_push.mode {out['mode']!r}: expected one of {list(MODES)}")
    fingers = [out["fingers"]] if isinstance(out["fingers"], str) else list(out["fingers"])
    if not fingers or any(f not in FINGER_MODES for f in fingers):
        raise ValueError(f"pick_overhang_push.fingers {out['fingers']!r}: expected a non-empty list of {list(FINGER_MODES)}")
    out["fingers"] = list(dict.fromkeys(fingers))
    pitches = [out["pitches_deg"]] if isinstance(out["pitches_deg"], (int, float)) else list(out["pitches_deg"])
    if not pitches or any(not 0.0 <= float(p) < 80.0 for p in pitches):
        raise ValueError(f"pick_overhang_push.pitches_deg {out['pitches_deg']!r}: expected angles in [0, 80) deg")
    out["pitches_deg"] = [float(p) for p in pitches]
    for k in ("tip_m", "contact_clearance_m", "finger_clearance_m", "com_margin_m", "max_push_m", "support_touch_m", "settle_max_slide_m",
              "settle_max_drop_m", "settle_max_tilt_deg", "settle_max_yaw_deg", "pre_min_ahead_m"):
        out[k] = float(out[k])
        if not math.isfinite(out[k]) or out[k] < 0.0:
            raise ValueError(f"pick_overhang_push.{k} {out[k]!r}: expected a number >= 0")
    out["distance_step_m"] = float(out["distance_step_m"])
    if not out["distance_step_m"] > 0.0:
        raise ValueError(f"pick_overhang_push.distance_step_m {out['distance_step_m']!r}: expected > 0")
    out["max_distances"] = int(out["max_distances"])
    if out["max_distances"] < 1:
        raise ValueError(f"pick_overhang_push.max_distances {out['max_distances']!r}: expected >= 1")
    out["yaw_free"] = bool(out["yaw_free"])
    out["rank_b"] = bool(out["rank_b"])
    offs = [out["grasp_back_offs_m"]] if isinstance(out["grasp_back_offs_m"], (int, float)) else list(out["grasp_back_offs_m"] or [])
    if any(not 0.0 < float(v) <= 0.6 for v in offs):
        raise ValueError(f"pick_overhang_push.grasp_back_offs_m {out['grasp_back_offs_m']!r}: expected distances in (0, 0.6] m")
    out["grasp_back_offs_m"] = [float(v) for v in offs]
    try:
        out["grasp_trunks"] = [(float(t[0]), float(t[1])) for t in (out["grasp_trunks"] or [])]
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"pick_overhang_push.grasp_trunks {out['grasp_trunks']!r}: expected [[translate, tilt], ...]") from None
    pushes = [out["v2_init_push_m"]] if isinstance(out["v2_init_push_m"], (int, float)) else list(out["v2_init_push_m"] or [])
    if not pushes or any(not 0.0 < float(v) <= out["max_push_m"] + 1e-9 for v in pushes):
        raise ValueError(f"pick_overhang_push.v2_init_push_m {out['v2_init_push_m']!r}: expected push distances in (0, max_push_m]")
    out["v2_init_push_m"] = [float(v) for v in pushes]
    if out["v2_finger_clearance_m"] is not None:
        out["v2_finger_clearance_m"] = float(out["v2_finger_clearance_m"])
        if not math.isfinite(out["v2_finger_clearance_m"]) or out["v2_finger_clearance_m"] < 0.0:
            raise ValueError(f"pick_overhang_push.v2_finger_clearance_m {out['v2_finger_clearance_m']!r}: expected None or a number >= 0")
    if out["v2_rest"] is not None and not str(out["v2_rest"]).startswith("overhang_"):
        raise ValueError(f"pick_overhang_push.v2_rest {out['v2_rest']!r}: expected an overhang rest tag (overhang_...)")
    return out


def v2_rest_push(rest_distance: float, edge_line: float, front: float) -> float:
    """i7-overhang 2026-11-07: the push that leaves the object at its v2 OVERHANG rest - the face ``front`` ahead of the centre along the
    push, the edge ``edge_line`` ahead of it on the face's line: the face ends ``rest_distance`` beyond the edge
    (:func:`overhang_m` = ``rest_distance``)."""
    return float(rest_distance) + float(edge_line) - float(front)


def flush_yaw(normal_xy, d) -> float:
    """i7-overhang 2026-11-07: the world yaw (about z, applied left of the rest orientation) that turns the rest-frame face normal
    ``normal_xy`` (support frame at yaw 0) onto the push direction ``d``: the overhanging face flush with the edge, facing it."""
    u = unit_xy(d)
    return math.atan2(float(u[1]), float(u[0])) - math.atan2(float(normal_xy[1]), float(normal_xy[0]))


def unit_xy(d) -> th.Tensor:
    v = th.as_tensor([float(d[0]), float(d[1])], dtype=th.float32)
    n = float(th.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("push direction is zero")
    return v / n


def lateral_unit(d) -> th.Tensor:
    """The horizontal unit vector across the push (``d`` turned +90 deg about z)."""
    u = unit_xy(d)
    return th.tensor([-float(u[1]), float(u[0])])


# ----------------------------------------------------------------------------------------------------------------
# linear extremes over convex point sets
# ----------------------------------------------------------------------------------------------------------------


def _crossings(pts: th.Tensor, normal: th.Tensor, offset: float) -> th.Tensor:
    """Points where the segments between every pair of ``pts`` cross the plane ``normal . x = offset`` (in the convex hull)."""
    s = pts @ normal - float(offset)
    i, j = th.triu_indices(len(pts), len(pts), offset=1)
    si, sj = s[i], s[j]
    m = (si * sj) < 0.0
    if not bool(m.any()):
        return th.zeros(0, 3, dtype=pts.dtype)
    i, j, si, sj = i[m], j[m], si[m], sj[m]
    t = (si / (si - sj)).unsqueeze(1)
    return pts[i] + t * (pts[j] - pts[i])


def min_over_hull(points, direction, constraints: Sequence[Tuple[Sequence[float], float]] = ()) -> Optional[float]:
    """``min direction . x`` over the convex hull of ``points`` (N, 3) intersected with the half-spaces ``n . x <= o`` of
    ``constraints``; None when that set is empty. Exact for the hull's vertices and the crossings of the vertex pairs with each
    constraint plane taken one at a time and with every pair of planes' crossings re-crossed (enough for the slabs the callers
    pass: two planes per axis); a linear programme over the hull's facets when scipy's qhull can build it."""
    pts = th.as_tensor(points, dtype=th.float64).reshape(-1, 3)
    c = th.as_tensor([float(v) for v in direction], dtype=th.float64)
    cons = [(th.as_tensor([float(v) for v in n], dtype=th.float64), float(o)) for n, o in constraints]
    if len(pts) == 0:
        return None
    try:
        import numpy as np
        from scipy.optimize import linprog
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts.numpy())
        eq = hull.equations  # n . x + b <= 0 inside
        a_ub = np.vstack([eq[:, :3]] + [np.asarray([n.numpy()]) for n, _ in cons]) if cons else eq[:, :3]
        b_ub = np.concatenate([-eq[:, 3]] + [np.asarray([o]) for _, o in cons]) if cons else -eq[:, 3]
        res = linprog(c.numpy(), A_ub=a_ub, b_ub=b_ub + 1e-9, bounds=[(None, None)] * 3, method="highs")
        if res.status == 2:
            return None
        if res.status == 0:
            return float(res.fun)
    except Exception:  # noqa: BLE001 - degenerate point sets (a flat hull) or no scipy: the enumeration below
        pass
    cand = [pts]
    for n, o in cons:
        cand.append(_crossings(pts, n, o))
    allc = th.cat(cand, dim=0)
    for k, (n, o) in enumerate(cons):  # the edges of the cut hull: crossings of the crossings
        for n2, o2 in cons[k + 1:]:
            cut = _crossings(pts, n, o)
            if len(cut) >= 2:
                allc = th.cat([allc, _crossings(cut, n2, o2)], dim=0)
    ok = th.ones(len(allc), dtype=th.bool)
    for n, o in cons:
        ok &= (allc @ n) <= o + 1e-9
    if not bool(ok.any()):
        return None
    return float((allc[ok] @ c).min())


# ----------------------------------------------------------------------------------------------------------------
# the grasp at the overhang
# ----------------------------------------------------------------------------------------------------------------


def finger_sweep_points(eef, open_width: float, pinch_width: float, thickness: float = FINGER_THICKNESS,
                        half_width: float = FINGER_HALF_WIDTH) -> Dict[int, th.Tensor]:
    """World points of each finger's swept box over the closure from ``open_width`` to ``pinch_width`` (gap between the pads, m):
    per side ``s`` (+1 / -1 along the closing axis c) the 8 corners of the box ``PAD_Z_RANGE[0] .. FINGERTIP_LENGTH`` along the
    approach, from the pad face ``w / 2`` to ``w / 2 + thickness`` along ``s c``, ``+-half_width`` along eef x, at both widths
    (the finger translates along c, so the two boxes' hull is its sweep)."""
    pos, quat = th.as_tensor(eef[0], dtype=th.float32), th.as_tensor(eef[1], dtype=th.float32)
    R = C.quat_to_mat(quat)
    a, c, x = R[:, 2], R[:, 1], R[:, 0]
    out: Dict[int, th.Tensor] = {}
    for s in (1, -1):
        pts = []
        for w in (float(open_width), float(pinch_width)):
            for za in (float(C.PAD_Z_RANGE[0]), float(C.FINGERTIP_LENGTH)):
                for off in (0.5 * w, 0.5 * w + float(thickness)):
                    for sx in (-1.0, 1.0):
                        pts.append(pos + za * a + s * off * c + sx * float(half_width) * x)
        out[s] = th.stack(pts)
    return out


def sweep_min_along(eef, d, z_plane: float, open_width: float, pinch_width: float, thickness: float = FINGER_THICKNESS,
                    half_width: float = FINGER_HALF_WIDTH) -> Optional[Tuple[float, int]]:
    """The smallest along-``d`` coordinate (world, ``d`` horizontal) any finger reaches below the plane ``z = z_plane`` over the
    closure (:func:`finger_sweep_points`): ``(value, side)``; None when no finger goes below the plane (the grasp needs no overhang)."""
    u = unit_xy(d)
    dirv = (float(u[0]), float(u[1]), 0.0)
    best: Optional[Tuple[float, int]] = None
    for side, pts in finger_sweep_points(eef, open_width, pinch_width, thickness, half_width).items():
        v = min_over_hull(pts, dirv, [((0.0, 0.0, 1.0), float(z_plane))])
        if v is not None and (best is None or v < best[0]):
            best = (v, side)
    return best


def push_distance_range(edge_finger: float, edge_com: float, sweep_along: float, finger_clearance: float, com_margin: float,
                        max_push: float, min_push: float = 0.01) -> Tuple[float, float]:
    """``(D_min, D_max)`` of the object's displacement along the push: the fingers' sweep (``sweep_along``, relative to the object's
    centre along ``d``) ``finger_clearance`` beyond the edge (``edge_finger`` from the centre on the fingers' lines), the centre at most
    ``edge_com - com_margin`` far (the edge on the centre line), never more than ``max_push``, never less than ``min_push``. Empty
    when ``D_min > D_max``."""
    d_min = max(float(edge_finger) + float(finger_clearance) - float(sweep_along), float(min_push))
    d_max = min(float(edge_com) - float(com_margin), float(max_push))
    return d_min, d_max


def distance_ladder(d_min: float, d_max: float, step: float, n_max: int) -> List[float]:
    """``d_min, d_min + step, ...`` up to ``d_max`` (inclusive within 0.1 mm), at most ``n_max`` values; [] for an empty range."""
    if d_min > d_max + 1e-4:
        return []
    out, k = [], 0
    while len(out) < int(n_max):
        v = float(d_min) + k * float(step)
        if v > float(d_max) + 1e-4:
            break
        out.append(round(min(v, float(d_max)), 4))
        k += 1
    return list(dict.fromkeys(out))


def edge_from_hits(hits: Sequence[bool], step: float = EDGE_PROBE_STEP_M) -> Optional[float]:
    """The support edge's distance from the probe line's start: rays at ``0, step, 2 step, ...``; the edge is half a step past the
    last ray of the leading run of hits. None when the first ray misses (the start is not on the support) or every ray hits (no edge
    within the probe)."""
    hits = [bool(h) for h in hits]
    if not hits or not hits[0]:
        return None
    for k, h in enumerate(hits):
        if not h:
            return (k - 0.5) * float(step)
    return None


def overhang_m(edge_along: float, push: float, extent_front: float) -> float:
    """How far the object's front (``extent_front`` ahead of its centre along the push) reaches past the edge after the push."""
    return float(push) + float(extent_front) - float(edge_along)


def extent_along(points, d) -> Tuple[float, float]:
    """``(back, front)``: how far the points (xy relative to the object's centre) reach behind / ahead of it along ``d``."""
    u = unit_xy(d)
    proj = th.as_tensor(points, dtype=th.float32)[:, :2] @ u
    return float(-proj.min()), float(proj.max())


# ----------------------------------------------------------------------------------------------------------------
# the push
# ----------------------------------------------------------------------------------------------------------------


def push_quat(pitch_deg: float, d, sign: float = 1.0) -> th.Tensor:
    """The push hand's eef orientation: the approach straight down leaned ``pitch_deg`` toward ``-d`` (the robot's forward: the
    fingertips away from the object, the finger faces leaning over its far edge - they press it down, never wedge under it), the
    closing axis horizontal across the push (``sign`` picks the wrist turn: the twin)."""
    u = unit_xy(d)
    f = th.tensor([-float(u[0]), -float(u[1]), 0.0])
    p = math.radians(float(pitch_deg))
    a = C.unit(th.tensor([0.0, 0.0, -1.0]) * math.cos(p) + f * math.sin(p))
    lat = lateral_unit(d)
    return C.eef_quat(a, float(sign) * th.tensor([float(lat[0]), float(lat[1]), 0.0]))


def lateral_bands(fingers: str, open_m: float = OPEN_FINGERS_M) -> List[Tuple[float, float]]:
    """The lateral intervals (relative to the push line) the push fingers occupy: the open hand's two fingers (pad face at ``open_m``
    per finger, ``FINGER_THICKNESS`` outward), the closed hand's one blade."""
    if fingers == "open":
        lo, hi = float(open_m), float(open_m) + FINGER_THICKNESS
        return [(-hi, -lo), (lo, hi)]
    return [(-CLOSED_HALF_WIDTH, CLOSED_HALF_WIDTH)]


def face_half(fingers: str) -> float:
    """The push fingers' half extent along the push direction at the tip (the face that meets the object)."""
    return FINGER_HALF_WIDTH if fingers == "open" else CLOSED_HALF_THICKNESS


def contact_along(points_rel, d, bands: Sequence[Tuple[float, float]], z_lo: float, z_hi: float, pitch_deg: float, tip_z: float) -> Optional[float]:
    """Where the push fingers first meet the object: ``min (q_along - (q_z - tip_z) tan pitch)`` over the object's hull points ``q``
    (relative to its centre: xy along ``d`` / across it, world z) inside a lateral band and the fingers' height band ``[z_lo, z_hi]``
    - a finger leaned ``pitch`` over the object has its face that much farther along the push at height ``z`` above the fingertips.
    None when nothing of the object lies in the bands."""
    u = unit_xy(d)
    lat = lateral_unit(d)
    pts = th.as_tensor(points_rel, dtype=th.float32).reshape(-1, 3)
    tan_p = math.tan(math.radians(float(pitch_deg)))
    # coordinates (along, lateral, z); minimise along - tan_p * z (+ tan_p * tip_z)
    q = th.stack([pts[:, :2] @ u, pts[:, :2] @ lat, pts[:, 2]], dim=1)
    best: Optional[float] = None
    for lo, hi in bands:
        v = min_over_hull(q, (1.0, 0.0, -tan_p), [((0.0, 0.0, 1.0), float(z_hi)), ((0.0, 0.0, -1.0), -float(z_lo)), ((0.0, 1.0, 0.0), float(hi)),
                                                ((0.0, -1.0, 0.0), -float(lo))])
        if v is not None:
            v += tan_p * float(tip_z)
            best = v if best is None else min(best, v)
    return best


def push_chain_poses(centre_xy, d, contact: float, pitch_deg: float, quat, tip_z: float, face: float, clearance: float, push: float,
                     pre_dist: float = C.PRE_DIST) -> Dict[str, Any]:
    """The push's eef poses: the fingertips ``tip_z`` high (world), their face (``face`` ahead of the tips along the push, over the
    pitch's cosine) ``clearance`` behind the first contact (``contact``: :func:`contact_along`, relative to the object's centre at
    ``centre_xy``), then moved ``clearance + push`` along ``d`` (the object travels ``push``); ``mid`` halfway, ``pre`` ``pre_dist`` back
    along the approach. Returns ``{"pre", "start", "mid", "goal", "direction", "hand_travel_m", "tip_start", "tip_goal"}``."""
    u = unit_xy(d)
    d3 = th.tensor([float(u[0]), float(u[1]), 0.0])
    q = th.as_tensor(quat, dtype=th.float32)
    a = C.approach_axis(q)
    cos_p = max(math.cos(math.radians(float(pitch_deg))), 1e-3)
    tip_along = float(contact) - float(face) / cos_p - float(clearance)
    c = th.as_tensor([float(centre_xy[0]), float(centre_xy[1])], dtype=th.float32)
    tip0 = th.tensor([float(c[0] + u[0] * tip_along), float(c[1] + u[1] * tip_along), float(tip_z)])
    travel = float(clearance) + float(push)
    tip1 = tip0 + d3 * travel
    start = (tip0 - C.FINGERTIP_LENGTH * a, q)
    goal = (tip1 - C.FINGERTIP_LENGTH * a, q)
    mid = (0.5 * (start[0] + goal[0]), q)
    pre = (start[0] - float(pre_dist) * a, q)
    return {"pre": pre, "start": start, "mid": mid, "goal": goal, "direction": d3, "hand_travel_m": travel, "tip_start": tip0, "tip_goal": tip1}


# ----------------------------------------------------------------------------------------------------------------
# the settle of the pushed object
# ----------------------------------------------------------------------------------------------------------------


def tilt_deg(q_ref, q) -> float:
    """Angle (deg) between the object's up axis at ``q_ref`` and at ``q``."""
    R0, R1 = C.quat_to_mat(th.as_tensor(q_ref, dtype=th.float32)), C.quat_to_mat(th.as_tensor(q, dtype=th.float32))
    up0, up1 = R0[:, 2], R1[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up0 @ up1)))))


def settle_facts(predicted, settled, d, rest_offset: float = 0.0) -> Dict[str, float]:
    """How the pushed object moved in the settle from its predicted post-push pose (placed ``rest_offset`` above its rest): planar
    ``slide_m`` (and its part ``along_m`` / ``across_m`` the push), ``drop_m`` (below the rest height, > 0 when it sank / fell),
    ``tilt_deg``, ``yaw_deg``."""
    p0, q0 = (th.as_tensor(x, dtype=th.float32) for x in predicted)
    p1, q1 = (th.as_tensor(x, dtype=th.float32) for x in settled)
    u, lat = unit_xy(d), lateral_unit(d)
    dp = p1 - p0
    yaw = math.degrees(C.wrap_angle(C.yaw_of(q1) - C.yaw_of(q0)))
    return {"slide_m": round(float(th.linalg.norm(dp[:2])), 4), "along_m": round(float(dp[:2] @ u), 4), "across_m": round(float(dp[:2] @ lat), 4),
            "drop_m": round(float(-dp[2]) - float(rest_offset), 4), "tilt_deg": round(tilt_deg(q0, q1), 2), "yaw_deg": round(yaw, 2)}


def settle_verdict(facts: Dict[str, float], opts: Dict[str, Any]) -> Optional[str]:
    """Why a settled push is not accepted (None: accepted): it fell / sank, tipped, slid or turned beyond the options' limits."""
    if facts["drop_m"] > float(opts["settle_max_drop_m"]):
        return f"fell / sank {facts['drop_m'] * 1000:.0f} mm (> {float(opts['settle_max_drop_m']) * 1000:.0f})"
    if facts["tilt_deg"] > float(opts["settle_max_tilt_deg"]):
        return f"tipped {facts['tilt_deg']:.1f} deg (> {float(opts['settle_max_tilt_deg']):.1f})"
    if facts["slide_m"] > float(opts["settle_max_slide_m"]):
        return f"slid {facts['slide_m'] * 1000:.0f} mm (> {float(opts['settle_max_slide_m']) * 1000:.0f})"
    if abs(facts["yaw_deg"]) > float(opts["settle_max_yaw_deg"]):
        return f"turned {facts['yaw_deg']:.1f} deg (> {float(opts['settle_max_yaw_deg']):.1f})"
    return None


def yaw_aligned_quat(current_quat, recorded_quat) -> th.Tensor:
    """The current orientation turned about world z by the yaw from it to ``recorded_quat`` (a body of revolution about its z axis:
    the same body, its yaw put back to the recorded one)."""
    qc = th.as_tensor(current_quat, dtype=th.float32)
    dy = C.wrap_angle(C.yaw_of(th.as_tensor(recorded_quat, dtype=th.float32)) - C.yaw_of(qc))
    return C.qmul(C.quat_from_yaw(dy), qc)


def revolution_about_z(equivalences: Any) -> bool:
    """Whether a target's orientation equivalences declare a body of revolution about its z axis (``zinf``)."""
    if equivalences is None:
        return False
    if isinstance(equivalences, str):
        return equivalences.strip().lower() == "zinf"
    try:
        return any(isinstance(e, str) and e.strip().lower() == "zinf" for e in equivalences)
    except TypeError:
        return False


# ---------------------------------------------------------------------------------------------------- i7-overpath 2026-11-08 (owner 3(b))
#: the spine guard (opt-in ``generation.overhang_required_side`` + ``overhang_pinch_edge_max_m``): the v2 overhang entry's side and every
#: drawn grasp's pinch must be the named face of the object (inside_large_01: the hardback's spine, -y; the shelf place rolls about the
#: spine's outward normal and releases it outward, ``inside_spine_out``) - the legacy spine-only filter ``generation.grasp_rows`` is skipped
#: under v2 (``IORSLarge._prepare_bindings``), so a +y entry in the root would be drawn and pinch the page block
SIDES = ("+x", "-x", "+y", "-y")
#: the pinch (``T_object_eef``'s position) may lie this far inside the face (the six bqvvvx overhang grasps: 1.08-1.69 cm) ...
PINCH_EDGE_MAX_M = 0.03
#: ... and this far outside it (the eef point between the pads never lies outside a pinched face)
PINCH_EDGE_OUT_M = 0.005


def side_axis(side: Any) -> Tuple[int, float]:
    """``"-y"`` -> (1, -1.0): the object-frame axis index and the sign of the named face's outward normal; ``ValueError`` otherwise."""
    s = str(side).strip().lower()
    if s not in SIDES:
        raise ValueError(f"overhang side {side!r}: expected one of {SIDES}")
    return "xyz".index(s[1]), (1.0 if s[0] == "+" else -1.0)


def pinch_depth(eef_pos_obj, hull_obj, side: Any) -> float:
    """How far inside the named face of the object (its local hull's extreme along the face's outward normal) the grasp's eef point
    (``T_object_eef``'s position, object frame) lies: > 0 inside, < 0 outside."""
    i, sign = side_axis(side)
    h = th.as_tensor(hull_obj, dtype=th.float32).reshape(-1, 3)
    face = float(h[:, i].max()) if sign > 0 else float(h[:, i].min())
    p = float(th.as_tensor(eef_pos_obj, dtype=th.float32).reshape(-1)[i])
    return (face - p) if sign > 0 else (p - face)


def spine_entry_why(required: Any, entry_side: Any) -> Optional[str]:
    """None when the v2 overhang entry's side is ``required``; else why (the attempt is rejected)."""
    i, s = side_axis(required)
    if entry_side is None:
        return f"the v2 overhang entry records no side (required {required})"
    try:
        j, t = side_axis(entry_side)
    except ValueError:
        return f"the v2 overhang entry's side {entry_side!r} is not a face name (required {required})"
    if (i, s) != (j, t):
        return f"the v2 overhang entry's side {entry_side} is not the required {required} (generation.overhang_required_side)"
    return None


def spine_grasp_why(required: Any, grasp_name: Any, eef_pos_obj, hull_obj, max_m: float = PINCH_EDGE_MAX_M, out_m: float = PINCH_EDGE_OUT_M) -> Optional[str]:
    """None when the grasp pinches the ``required`` face: its eef point within ``[-out_m, max_m]`` of that face (:func:`pinch_depth`);
    else why (the attempt is rejected)."""
    d = pinch_depth(eef_pos_obj, hull_obj, required)
    if -float(out_m) <= d <= float(max_m):
        return None
    return (f"grasp {grasp_name}: its pinch lies {d * 100:.1f} cm inside the {required} face (allowed {-float(out_m) * 100:.1f}..{float(max_m) * 100:.1f} cm; "
            f"generation.overhang_required_side / overhang_pinch_edge_max_m)")
