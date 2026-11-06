"""Navigation feasibility (spec §3.7), stance transitions (§3.8) and parking (§3.3) of the v2 sampler.

Pure at import time (no simulator): every function takes the ``RobotKit`` / ``SimWorld`` it needs, so the
enumeration helpers (``transition_standoffs``) are unit-tested without Isaac Sim.

§3.7 — stances lie 0.35–1.1 m from a face, inside BEHAVIOR's eroded band, so:

1. if the goal stance lies in the same connected region of the **locally un-eroded** map (the un-eroded floor map of
   the loaded floor, optionally restricted to the task rooms) as the start stance, the shortest grid path on that map is
   taken and validated with whole-body collision checks at the **navigation posture** (untucked arms, carried object
   attached, trunk as recorded) at 0.1 m strides (``mode = "local"``);
2. otherwise the eroded map is planned between the nearest eroded-free cells of the two stances and each stance is
   connected to the global path by a straight local segment, collision-checked the same way (``mode = "global"``).

§3.8 — the trunk change between two consecutive stances happens at the smallest transition stand-off
``s_T ∈ {s, s + 0.1, …, s + 0.5}`` (along -inward from the arriving stance) at which the swept trunk motion (5 samples
between the two trunk values, arms untucked, carried object attached) is whole-body collision-free.

§3.3 — parking: tucked at a free cell inside the task room's floor polygon, >= 1 m from every spawned object and support.
"""

from __future__ import annotations

import heapq
import logging
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.fastflags import fast_on  # train-fast 2026-10-31: ISPATIALGYM_FAST_* switches (off = the former behaviour)
from ispatialgym.sampling.v2.stances import PathRecord, Stance, Transition, path_length

if TYPE_CHECKING:  # pragma: no cover
    from ispatialgym.sampling.robot_utils import RobotKit

log = logging.getLogger("ispatialgym.sampling.v2")

STRIDE_M = 0.1
ATTACH_FIT_SEED = 20260915
TRANSITION_STEP = 0.1
TRANSITION_EXTRA_MAX = 0.5
TRANSITION_SAMPLES = 5
PARK_MIN_DIST = 1.0


class NavigationError(Exception):
    """A path / transition could not be established (the caller turns it into ``Reject(<reason>)``)."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def transition_standoffs(s: float, step: float = TRANSITION_STEP, extra_max: float = TRANSITION_EXTRA_MAX) -> List[float]:
    """§3.8 candidate set ``{s, s + 0.1, …, s + 0.5}`` (smallest first)."""
    n = int(round(extra_max / step))
    return [round(float(s) + k * step, 6) for k in range(n + 1)]


#: densest trunk motion between two checked sweep samples (owner 2026-09-19 21:56 carry legs: the 5-sample sweep of a
#: (0, 0) -> (0.95, -1.2) transition moved the held sandal ~25 cm between samples and stepped over the 4 cm table top it
#: then brushed in the demo, 0919-221909 action 6)
SWEEP_MAX_TRANSLATE_STEP = 0.05
SWEEP_MAX_TILT_STEP = 0.10


def sweep_samples(trunk_from: Tuple[float, float], trunk_to: Tuple[float, float], n: int = TRANSITION_SAMPLES) -> int:
    """The number of sweep samples for a trunk motion: at least ``n``, and enough that no sample step exceeds
    ``SWEEP_MAX_TRANSLATE_STEP`` / ``SWEEP_MAX_TILT_STEP``."""
    dt, dk = abs(float(trunk_to[0]) - float(trunk_from[0])), abs(float(trunk_to[1]) - float(trunk_from[1]))
    return max(int(n), int(math.ceil(dt / SWEEP_MAX_TRANSLATE_STEP)) + 1, int(math.ceil(dk / SWEEP_MAX_TILT_STEP)) + 1)


def trunk_sweep(trunk_from: Tuple[float, float], trunk_to: Tuple[float, float], n: int = TRANSITION_SAMPLES) -> List[Tuple[float, float]]:
    """Trunk samples between (and including) the two trunk values, linear in (translate, tilt): ``n`` of them, or more
    when the motion is large (:func:`sweep_samples`)."""
    (t0, k0), (t1, k1) = trunk_from, trunk_to
    if n <= 1:
        return [(float(t1), float(k1))]
    n = sweep_samples(trunk_from, trunk_to, n)
    return [(t0 + (t1 - t0) * i / (n - 1), k0 + (k1 - k0) * i / (n - 1)) for i in range(n)]


#: yaw step of the lag envelope a swept stride pose is repeated at (``stride_poses(yaw_envelope=True)``), at most the heading
#: step bound of the check (``max_yaw_step``)
YAW_ENVELOPE_STEP_RAD = math.radians(10.0)
#: owner 2026-10-13 ~04:30 item 3a (under#03 144 pitched c1_k1, 17836644 / 17838303 'held:base_link in contact with desk_mdhelw_1'
#: at navigate[5]: the B -> C carry's last 12.7 cm segment turns 134 deg and its 10 cm strides were checked at 46, 113 and 180 deg
#: only - at 137 deg the can, 0.73 m from the base centre, swept into the desk): the largest heading step between two consecutive
#: checked poses of a swept-pose check (:func:`stride_poses`, the yaw-lag envelope, :func:`face_travel_poses`), and the farthest
#: robot / carried point's travel per heading step (:func:`yaw_step_bound`) - as ``SWEEP_MAX_TILT_STEP`` bounds a trunk sweep
YAW_STEP_MAX_RAD = math.radians(3.0)
YAW_SWEEP_MAX_M = 0.03
#: how far back along the path the yaw reference still shapes the tracker's yaw: the tracker (kp_yaw 2 -> ~0.5 s time constant)
#: at the 0.3-0.5 m/s of a stance-to-stance drive lags what the reference did over the last ~0.2 m; a long straight run has
#: caught up, a shuffle of short turning segments has not
YAW_LAG_LOOKBACK_M = 0.25
#: the largest yaw lag the envelope assumes (owner 2026-09-26, wave 6 shampoo 0926-001742: the look-back hull alone swept up to ~40 deg
#: behind on the utility room's 36 cm / 57 deg shuffle and rejected most B -> C paths - 3 attempts, 42 min, no instance, where wave 3
#: sampled at attempt 0 in 407 s; the measured lag of the brush it guards against was 6 deg: the tracker's yaw feed-forward keeps the
#: lag a fraction of the reference's change) - twice the measured lag
YAW_LAG_MAX_RAD = math.radians(12.0)


def yaw_step_bound(radius_m: Optional[float] = None) -> float:
    """The heading step (rad) allowed between two consecutive checked poses: :data:`YAW_STEP_MAX_RAD`, less when a point
    ``radius_m`` from the base centre (:func:`sweep_radius`) would move more than :data:`YAW_SWEEP_MAX_M` per step."""
    if radius_m is None or not radius_m > 0.0:
        return YAW_STEP_MAX_RAD
    return min(YAW_STEP_MAX_RAD, YAW_SWEEP_MAX_M / float(radius_m))


def _yaw_steps(dyaw: float, max_yaw_step: Optional[float]) -> int:
    """How many steps of at most ``max_yaw_step`` a heading change ``dyaw`` takes (0 without a bound)."""
    if not max_yaw_step or max_yaw_step <= 0.0:
        return 0
    return int(math.ceil(abs(float(dyaw)) / float(max_yaw_step) - 1e-9))


def bound_yaw_steps(poses: Sequence[Sequence[float]], max_yaw_step: Optional[float] = YAW_STEP_MAX_RAD) -> List[Tuple[float, float, float]]:
    """``poses`` (drive order) with poses interpolated linearly (x, y, the yaw the shortest way) between two consecutive ones
    whose yaws differ by more than ``max_yaw_step``; the others as they are."""
    pts = [tuple(float(v) for v in p) for p in poses]
    out = pts[:1]
    for a, b in zip(pts[:-1], pts[1:]):
        d = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
        m = _yaw_steps(d, max_yaw_step)
        out.extend((a[0] + (b[0] - a[0]) * k / m, a[1] + (b[1] - a[1]) * k / m, a[2] + d * k / m) for k in range(1, m))
        out.append(b)
    return out


def stride_poses(waypoints: Sequence[Sequence[float]], stride: float = STRIDE_M, yaw_envelope: bool = False,
                 max_yaw_step: Optional[float] = YAW_STEP_MAX_RAD) -> List[Tuple[float, float, float]]:
    """Base poses every ``stride`` metres along a polyline of (x, y[, yaw]) waypoints; the yaw is interpolated linearly
    (shortest way) between consecutive waypoint yaws, or taken along the segment when a waypoint has none. A segment whose
    yaw turns more than ``max_yaw_step`` per stride is sampled densely enough that no two consecutive poses differ by more
    (owner 2026-10-13 item 3a, :data:`YAW_STEP_MAX_RAD` / :func:`yaw_step_bound`; None = the strides alone).

    ``yaw_envelope`` (owner 2026-09-25; shampoo 0925-211757 / 222156: the left finger brushed ``wardrobe_pyuwgc_0`` 38 steps into
    a 36 cm shuffle that turns 57 deg - the base 4 mm off the line, its yaw 6 deg BEHIND the interpolated one, while the 3 cm
    sweep of the interpolated yaws had passed): the oracle's tracker (``BaseTrajectoryTracker``: 0.8 rad/s, 1.5 rad/s^2, kp_yaw
    2) lags the yaw reference on a turn, so every stride pose is repeated at the yaws the reference held over the last
    :data:`YAW_LAG_LOOKBACK_M` of path (the angular hull of those yaws and the interpolated one, each capped at
    :data:`YAW_LAG_MAX_RAD` from the interpolated yaw, every :data:`YAW_ENVELOPE_STEP_RAD` or ``max_yaw_step`` if smaller) -
    the swept volume covers the lag; a straight run with a constant yaw adds nothing."""
    out: List[Tuple[float, float, float]] = []
    wps = [tuple(float(v) for v in w) for w in waypoints]
    if not wps:
        return out
    env_step = min(YAW_ENVELOPE_STEP_RAD, float(max_yaw_step)) if max_yaw_step and max_yaw_step > 0.0 else YAW_ENVELOPE_STEP_RAD
    strides: List[Tuple[float, float, float, float]] = []  # (x, y, yaw, arc length) of the plain stride poses
    arc = 0.0
    for a, b in zip(wps[:-1], wps[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw_a = a[2] if len(a) > 2 else math.atan2(b[1] - a[1], b[0] - a[0])
        yaw_b = b[2] if len(b) > 2 else yaw_a
        dyaw = (yaw_b - yaw_a + math.pi) % (2 * math.pi) - math.pi
        n = max(1, int(math.ceil(seg / stride)), _yaw_steps(dyaw, max_yaw_step))
        for i in range(n):
            s = i / n
            strides.append((a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, yaw_a + dyaw * s, arc + seg * s))
        arc += seg
    last = wps[-1]
    strides.append((last[0], last[1], last[2] if len(last) > 2 else strides[-1][2] if strides else 0.0, arc))
    for j, (x, y, yaw, s_j) in enumerate(strides):
        out.append((x, y, yaw))
        if not yaw_envelope:
            continue
        # the yaws the lagging robot can hold here: the hull of the reference yaws over the last YAW_LAG_LOOKBACK_M of path
        offs = [0.0]
        for k in range(j - 1, -1, -1):
            if s_j - strides[k][3] > YAW_LAG_LOOKBACK_M:
                break
            o = (strides[k][2] - yaw + math.pi) % (2 * math.pi) - math.pi
            offs.append(max(-YAW_LAG_MAX_RAD, min(YAW_LAG_MAX_RAD, o)))
        for o in (min(offs), max(offs)):
            m = int(math.ceil(abs(o) / env_step - 1e-9))
            for k in range(1, m + 1):
                out.append((x, y, yaw + o * min(1.0, k / m)))
    return out


def yaw_lag_envelope(poses: Sequence[Tuple[float, float, float]], max_yaw_step: Optional[float] = YAW_STEP_MAX_RAD) -> List[Tuple[float, float, float]]:
    """The yaw-lag envelope of :func:`stride_poses` over an arbitrary drive-ordered pose sequence (the ``face_travel``
    schedule's ``BasePath.stride_poses``: the arc length is the travelled xy distance, a dwell's on-the-spot sweep adds
    none): every pose repeated at the yaws the sequence held over the last :data:`YAW_LAG_LOOKBACK_M`, each capped at
    :data:`YAW_LAG_MAX_RAD`, every :data:`YAW_ENVELOPE_STEP_RAD` or ``max_yaw_step`` if smaller - so the schedule and the
    polyline are checked alike. Consecutive poses of the sequence more than ``max_yaw_step`` apart in yaw get poses in
    between first (:func:`bound_yaw_steps`)."""
    pts = bound_yaw_steps(poses, max_yaw_step)
    env_step = min(YAW_ENVELOPE_STEP_RAD, float(max_yaw_step)) if max_yaw_step and max_yaw_step > 0.0 else YAW_ENVELOPE_STEP_RAD
    arc: List[float] = []
    s_ = 0.0
    for i, p in enumerate(pts):
        if i:
            s_ += math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1])
        arc.append(s_)
    out: List[Tuple[float, float, float]] = []
    for j, (x, y, yaw) in enumerate(pts):
        out.append((x, y, yaw))
        offs = [0.0]
        for k in range(j - 1, -1, -1):
            if arc[j] - arc[k] > YAW_LAG_LOOKBACK_M:
                break
            o = (pts[k][2] - yaw + math.pi) % (2 * math.pi) - math.pi
            offs.append(max(-YAW_LAG_MAX_RAD, min(YAW_LAG_MAX_RAD, o)))
        for o in (min(offs), max(offs)):
            m = int(math.ceil(abs(o) / env_step - 1e-9))
            for k in range(1, m + 1):
                out.append((x, y, yaw + o * min(1.0, k / m)))
    return out


def face_travel_poses(waypoints: Sequence[Sequence[float]], stride: float = STRIDE_M,
                      max_yaw_step: Optional[float] = YAW_STEP_MAX_RAD, cfg: Optional[Dict[str, float]] = None) -> List[Tuple[float, float, float]]:
    """The whole-body-checked poses of the ``face_travel`` schedule of ``waypoints`` (owner 2026-09-26: the shared
    ``robot.base_path`` schedule the oracle drives for ``style = "face_travel"``, built with the module's default config;
    ``cfg`` = the record's own overrides of it, oracle-fix 2026-11-06 ``paths[].face_travel_cfg``), no two consecutive ones more
    than ``max_yaw_step`` apart in yaw (``BasePath.stride_poses``)."""
    from ispatialgym.robot import base_path as BP

    sched = BP.BasePath.build([tuple(float(v) for v in w) for w in waypoints], BP.config_with(cfg))
    return [tuple(float(v) for v in p) for p in sched.stride_poses(stride, max_yaw_step=max_yaw_step)]


def record_sweep_poses(rec: PathRecord, yaw_envelope: bool = True, max_yaw_step: Optional[float] = YAW_STEP_MAX_RAD) -> List[Tuple[float, float, float]]:
    """The poses a re-sweep of a recorded path checks (oracle-fix 2026-11-06, owner feedback on the inside_large#02 demo: the
    re-sweeps swept the polyline of a record the oracle drives as the face_travel schedule, which leaves the polyline by up to
    simplify + fillet = 0.15 m): ``rec.style == "face_travel"`` -> the schedule (with the record's ``face_travel_cfg``) and its
    yaw-lag envelope, as :func:`plan_path` checked it; otherwise the polyline's strides (+ the yaw-lag envelope with
    ``yaw_envelope``), as before."""
    stride = float(getattr(rec, "stride_m", None) or STRIDE_M)
    if getattr(rec, "style", None) == "face_travel":
        ft = face_travel_poses(rec.waypoints, stride, max_yaw_step=max_yaw_step, cfg=getattr(rec, "face_travel_cfg", None))
        return yaw_lag_envelope(ft, max_yaw_step=max_yaw_step)
    return stride_poses(rec.waypoints, stride, yaw_envelope=yaw_envelope, max_yaw_step=max_yaw_step)


def total_rotation(poses: Sequence[Sequence[float]]) -> float:
    """How far the body turns along a drive-ordered pose sequence (rad, the sum of the shortest-way yaw changes): of a polyline's
    waypoints = what the oracle's recorded style turns (each segment's yaw blended the shortest way), of :func:`face_travel_poses` =
    what the schedule turns (oracle-fix 2026-11-06: within 3 deg of ``Motions._base_reference`` on 100 of 104 IORS-large records)."""
    return float(sum(abs((float(b[2]) - float(a[2]) + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(poses[:-1], poses[1:])))


#: owner 2026-09-26: per-process tally of the log-only face_travel probe of CARRIED legs (they stay ``recorded``): how often the
#: schedule fails the 1 cm check where the recorded polyline passed - reported in the morning
FACE_TRAVEL_CARRIED_PROBE = {"checked": 0, "collides": 0}


#: edge slide (owner 2026-09-24, disembedding carry): two stances at the same support edge are within this of each
#: other's edge frame (midpoint, inward normal) and body yaw to be joined by a slide along the edge
EDGE_SLIDE_MID_TOL_M = 0.02
EDGE_SLIDE_YAW_TOL_RAD = math.radians(2.0)
EDGE_SLIDE_MODE = "edge_slide"


def same_edge(a: Stance, b: Stance, tol_m: float = EDGE_SLIDE_MID_TOL_M, tol_rad: float = EDGE_SLIDE_YAW_TOL_RAD) -> bool:
    """Whether two tabletop stances stand at the **same support edge facing it**: the same edge frame (name, midpoint within
    ``tol_m``, inward normal within ``tol_rad``) and the same body yaw (within ``tol_rad``; a stance faces its edge, so
    the yaws agree whenever the frames do)."""
    if a.edge is None or b.edge is None or a.edge.name != b.edge.name:
        return False
    if float(th.linalg.norm(a.edge.mid - b.edge.mid)) > tol_m:
        return False
    if float(th.dot(a.edge.inward, b.edge.inward)) < math.cos(tol_rad):
        return False
    dyaw = (float(a.yaw) - float(b.yaw) + math.pi) % (2 * math.pi) - math.pi
    return abs(dyaw) <= tol_rad


def edge_slide_waypoints(frm: Stance, to: Stance, back_m: float = 0.0) -> List[Tuple[float, float, float]]:
    """§3.7 **edge slide** (owner 2026-09-24): the base path between two stances at the same support edge with the body
    facing the edge the whole way (the R1Pro base is holonomic). Drive back along ``-inward`` to the slide stand-off (the
    larger of the two stand-offs + ``back_m``), slide along the edge to the arriving stance's lateral position, drive in
    to its stand-off. The first waypoint is the departing pose; every other waypoint carries the **arriving stance's
    yaw**, so ``stride_poses`` / the oracle's tracker (style ``recorded``) blend no heading change and the slide is a pure
    lateral velocity. Waypoints closer than 1 mm collapse (equal stand-offs: two waypoints, a straight slide). The
    positions come from the stances' actual base poses and the edge normal alone, not from their ``lateral`` bookkeeping."""
    inward = to.inward
    s_from = float(frm.stand_off if frm.stand_off is not None else 0.0)
    s_to = float(to.stand_off if to.stand_off is not None else 0.0)
    s_slide = max(s_from, s_to) + max(0.0, float(back_m))
    p0, p3 = frm.base_xy, to.base_xy
    p1 = p0 - inward * (s_slide - s_from)
    p2 = p3 - inward * (s_slide - s_to)
    pts: List[th.Tensor] = []
    for p in (p0, p1, p2, p3):
        if pts and float(th.linalg.norm(p - pts[-1])) < 1e-3:
            continue
        pts.append(p)
    yaw = float(to.yaw)
    out = [tuple(float(v) for v in frm.base_pose2d)]
    out += [(float(p[0]), float(p[1]), yaw) for p in pts[1:]]
    if len(out) == 1:  # the stances coincide
        out.append((float(p3[0]), float(p3[1]), yaw))
    return out


def grid_shortest_path(free: th.Tensor, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* on a boolean grid (8-connected, no corner cutting through blocked cells); None when disconnected."""
    h, w = free.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if not bool(free[start]) or not bool(free[goal]):
        return None
    fr = free.numpy() if hasattr(free, "numpy") else free

    def hcost(a):
        return math.hypot(a[0] - goal[0], a[1] - goal[1])

    openq = [(hcost(start), 0.0, start)]
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    dist = {start: 0.0}
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    while openq:
        _, d, cur = heapq.heappop(openq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if d > dist.get(cur, float("inf")):
            continue
        for di, dj, c in nbrs:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w) or not fr[nb[0], nb[1]]:
                continue
            if di and dj and not (fr[cur[0] + di, cur[1]] and fr[cur[0], cur[1] + dj]):
                continue
            nd = d + c
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                came[nb] = cur
                heapq.heappush(openq, (nd + hcost(nb), nd, nb))
    return None


def simplify_path(cells: Sequence[Tuple[int, int]], free: th.Tensor) -> List[Tuple[int, int]]:
    """Greedy line-of-sight shortcutting of a grid path (Bresenham visibility on ``free``)."""
    if len(cells) <= 2:
        return list(cells)
    fr = free.numpy() if hasattr(free, "numpy") else free

    def visible(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
        err = dx - dy
        x, y = x0, y0
        while True:
            if not fr[x, y]:
                return False
            if (x, y) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not visible(cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


# ----------------------------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------------------------


def local_map(kit: "RobotKit", rooms: Optional[Sequence[str]] = None, floor: int = 0, blocked: Sequence[Sequence[Sequence[float]]] = ()) -> th.Tensor:
    """The **locally un-eroded** map of §3.7: the un-eroded floor map of the loaded floor (255 = free), restricted to the
    task ``rooms`` (+ a one-cell dilation so doorways connect) and minus ``blocked`` footprints (polygons, world xy)."""
    trav = kit.world.scene.trav_map
    free = trav.floor_map[floor] == 255
    if rooms:
        seg = kit.world.scene.seg_map
        ids = [seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id]
        if ids and seg.room_ins_map.shape == free.shape:
            mask = th.zeros_like(free)
            for i in ids:
                mask |= seg.room_ins_map == i
            k = 3
            mask = th.nn.functional.max_pool2d(mask.float()[None, None], k, stride=1, padding=k // 2)[0, 0] > 0.5
            free = free & mask
    if blocked:
        from ispatialgym.sampling.scene_variants import polygon_cells

        for poly in blocked:
            cells = polygon_cells(trav, poly, floor)
            if len(cells):
                free[cells[:, 0], cells[:, 1]] = False
    return free


def eroded_map(kit: "RobotKit", floor: int = 0) -> th.Tensor:
    return kit.eroded_trav_map(floor) == 255


def _to_map(kit: "RobotKit", xy) -> Tuple[int, int]:
    ij = kit.world.scene.trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
    return int(ij[0]), int(ij[1])


def _to_world(kit: "RobotKit", ij: Tuple[int, int]) -> th.Tensor:
    return kit.world.scene.trav_map.map_to_world(th.tensor([int(ij[0]), int(ij[1])]))


def nearest_free_cell(free: th.Tensor, ij: Tuple[int, int], max_radius: int = 15) -> Optional[Tuple[int, int]]:
    h, w = free.shape
    if 0 <= ij[0] < h and 0 <= ij[1] < w and bool(free[ij]):
        return ij
    for r in range(1, max_radius + 1):
        best = None
        for di in range(-r, r + 1):
            for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                a, b = ij[0] + di, ij[1] + dj
                if 0 <= a < h and 0 <= b < w and bool(free[a, b]):
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, (a, b))
        if best is not None:
            return best[1]
    return None


def same_region(free: th.Tensor, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Whether two cells are connected on ``free`` (flood fill, 4-connected)."""
    h, w = free.shape
    if not (bool(free[a]) and bool(free[b])):
        return False
    fr = free.numpy() if hasattr(free, "numpy") else free
    seen = {a}
    stack = [a]
    while stack:
        cur = stack.pop()
        if cur == b:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + di, cur[1] + dj)
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in seen and fr[nb[0], nb[1]]:
                seen.add(nb)
                stack.append(nb)
    return False


# ----------------------------------------------------------------------------------------------------------------
# whole-body checks along a path
# ----------------------------------------------------------------------------------------------------------------


def path_collision_free(kit: "RobotKit", poses: Sequence[Tuple[float, float, float]], trunk: Tuple[float, float], posture: str = "untucked",
                        margin: float = 0.01, q_arms: Optional[Dict[str, th.Tensor]] = None, fingers: Optional[float] = None) -> Tuple[bool, int]:
    """Whole-body collision check (cuRobo, ``margin``) of the base poses at the navigation posture (``q_arms`` /
    ``fingers`` override the arm vectors / gripper opening); the carried object must already be attached to the planner
    (``kit.backend.attach_object``). Returns (all free, first failing index)."""
    if not poses:
        return True, -1
    q0 = _nav_q(kit, trunk, posture, q_arms, fingers)
    qs = th.stack([kit.q_at_base_pose(q0, x, y, yaw) for x, y, yaw in poses])
    free = kit.collision_free(qs, margin=margin)
    if bool(free.all()):
        return True, -1
    return False, int(th.nonzero(~free)[0])


def path_blame(kit: "RobotKit", pose: Tuple[float, float, float], trunk: Tuple[float, float], posture: str, q_arms: Optional[Dict[str, th.Tensor]],
               margin: float) -> str:
    """Opt-in (``ISPATIALGYM_STANCE_BLAME``, or ``ISPATIALGYM_PATH_BLAME`` for this blame alone; oracle-fix 2026-11-04 owner item 4): at a colliding swept pose (the carried object still
    attached), the robot / attached links in the world (by class) and the world meshes in contact (``world_collision_offenders``)."""
    try:
        q = kit.q_at_base_pose(_nav_q(kit, trunk, posture, q_arms), *pose)
        costs = kit.backend._collision_costs(kit.backend.kin_entry("whole_body"))
        links = sorted(set(costs.get("link_of_sphere", [])))
        hits = [l for l in links if not bool(kit.links_clear(q, [l], float(margin))[0])]
        offs = []
        try:
            # oracle-fix 2026-11-04 round 4: at the path's margin (a margin-only contact named whatever mesh sorts first), object + mesh names
            offs = [(lambda p: f"{p[-4]}:{p[-1]}" if len(p) >= 4 else str(o))(str(o).split("/"))
                    for o in kit.backend.world_collision_offenders(q, emb_name="whole_body", limit=4, margin=float(margin))]
        except Exception as e:  # noqa: BLE001
            offs = [f"offenders failed ({e!r})"]
        # the decisive splits: with self-collision off (a carried object or an arm against the robot's own links), with the ground slab
        # off would need a world re-sync (not done: it would undo the caller's ignore set); the mesh offender search names whatever mesh it disables first when neither
        # helps (2026-11-04: it named an armchair, then ceiling downlights, for the same pose)
        q1 = q.unsqueeze(0)
        world_hit = bool(kit.backend.check_collision(q1, emb_name="whole_body", sync_world=False, self_collision=False, margin=float(margin))[0])
        self_only = not world_hit
        bodies = _world_bodies_freeing(kit, q1, pose, float(margin)) if world_hit else ""
        return (f"; blame: {'SELF-COLLISION only (world clear with self-collision off)' if self_only else 'world contact'}"
                f"; links near the world {hits}; world meshes {offs}{bodies}")
    except Exception as e:  # noqa: BLE001
        return f"; blame failed ({e!r})"


def _world_bodies_freeing(kit: "RobotKit", q1: th.Tensor, pose: Tuple[float, float, float], margin: float) -> str:
    """Oracle-fix 2026-11-04 round 3 (item 3b): the world bodies whose removal ALONE frees a colliding swept pose - each scene object whose
    AABB lies within 1.5 m of the base (nearest first, 14 at most) taken out by ``ik._Ignoring`` (its collision meshes off, as
    ``IKChecker._collision_blame`` does for spawned objects), the world re-checked without self-collision; the cuRobo ground slab and the
    floor objects too. The base pose and the nearest AABB distances are logged to check the pose's frame against the scene
    (``world_collision_offenders`` names whatever mesh it disables first)."""
    try:
        from ispatialgym.sampling.v2.ik import _Ignoring

        x, y = float(pose[0]), float(pose[1])
        rows = []
        for o in list(getattr(getattr(getattr(kit, "world", None), "scene", None), "objects", None) or []):
            try:
                lo, hi = o.aabb
            except Exception:  # noqa: BLE001
                continue
            d = math.hypot(max(float(lo[0]) - x, 0.0, x - float(hi[0])), max(float(lo[1]) - y, 0.0, y - float(hi[1])))
            if d < 1.5 and "ceiling" not in o.name:
                rows.append((d, o))
        rows.sort(key=lambda r: r[0])

        def free_without(objs, ground: bool = True) -> bool:
            with _Ignoring(kit, list(objs), ground=ground):
                return not bool(kit.backend.check_collision(q1, emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)[0])

        freeing = [f"{o.name} ({d:.2f} m)" for d, o in rows[:14] if free_without([o])]
        slab = free_without([], ground=False)
        floors = [o for _, o in rows if "floor" in o.name]
        fl = free_without(floors) if floors else None
        near = ", ".join(f"{o.name} {d:.2f}" for d, o in rows[:8])
        return (f"; base pose ({x:.2f}, {y:.2f}) nearest AABBs: {near}; freed by removing alone: {freeing or 'none'}; ground slab off: "
                f"{'free' if slab else 'still collides'}; floor objects off: {fl}")
    except Exception as e:  # noqa: BLE001
        return f"; body search failed ({e!r})"


def _nav_q(kit: "RobotKit", trunk: Tuple[float, float], posture: str = "untucked", q_arms: Optional[Dict[str, th.Tensor]] = None,
           fingers: Optional[float] = None) -> th.Tensor:
    kw = {} if fingers is None else {"fingers": fingers}
    return kit.q_config(arms=posture, translate=trunk[0], tilt=trunk[1], right_arm=(q_arms or {}).get("right"), left_arm=(q_arms or {}).get("left"), **kw)


def sweep_radius(kit: "RobotKit", trunk: Tuple[float, float], posture: str = "untucked", q_arms: Optional[Dict[str, th.Tensor]] = None,
                 fingers: Optional[float] = None) -> Optional[float]:
    """The farthest point of the navigation posture's collision spheres (the robot's and an attached carried object's) from
    the base centre in xy (m): what a heading step sweeps the most (:func:`yaw_step_bound`). None when the backend has no
    sphere forward kinematics (``robot_spheres_world``)."""
    try:
        q = kit.q_at_base_pose(_nav_q(kit, trunk, posture, q_arms, fingers), 0.0, 0.0, 0.0)
        sp, _ = kit.backend.robot_spheres_world(q.unsqueeze(0))
        sp = th.as_tensor(sp, dtype=th.float32).reshape(-1, 4)
        sp = sp[sp[:, 3] > 0.0]
        return float((th.linalg.norm(sp[:, :2], dim=1) + sp[:, 3]).max()) if len(sp) else None
    except Exception:  # noqa: BLE001 - no sphere model: the angular bound alone
        return None


def attached_checks_enabled() -> bool:
    """``ISG_ATTACHED_CHECKS=off`` (development switch, recorded in every summary as ``checks.attached_object``): the
    carried object is *not* attached for the checks. Exists because the robot model exempts the attached object from
    self-collision only against the finger links (README "Backend facts"), so with the current backend a grasped object
    deep in the gripper fails every attached check; the switch lets the rest of the pipeline be exercised until the
    model is fixed. It is not a sampler fallback: the setting is global, logged and written into the artefacts."""
    import os

    return os.environ.get("ISG_ATTACHED_CHECKS", "on").lower() not in ("off", "0", "false")


class attached:
    """Context manager: the carried object attached to the planner for the checks inside (§3.7 / §3.8 / §3.5).

    ``R1ProCuRobo.attach_object`` fits the object's collision meshes in the eef frame from the *current* relative pose
    of object and eef. While sampling the object rests on its support and the hands are untucked, so the recorded grasp
    transform ``eef_in_object`` (T_obj→eef of the drawn grasp) is applied first: the object is moved (gravity off) to
    ``eef_pose ∘ inv(T_obj→eef)``, attached, and put back exactly where it was. Without ``eef_in_object`` the current
    relative pose is used (the object is already in the hand)."""

    def __init__(self, kit: "RobotKit", arm: Optional[str], obj: Any, eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None):
        self.kit, self.arm, self.obj, self.eef_in_object = kit, arm, obj, eef_in_object
        if not attached_checks_enabled():
            self.obj = None
        self._saved = None
        self._no_step = False

    @staticmethod
    def _step_skippable(obj) -> bool:
        """ATTACH_NO_STEP (train-fast 2026-10-31) applies to a single rigid body only. What the sphere fit reads: the eef pose from the
        robot's PhysX view (current the moment it is asked) and each collision mesh's pose through OmniGibson's Fabric hierarchy
        (``get_world_pose``), which ``set_position_orientation`` refreshes itself (``sync_physx_to_fabric`` right after the PhysX write);
        ``og.sim.step_physics`` does NOT refresh Fabric (``sync_render_transforms``: "stale after physics-only steps"), so the two steps
        never fed the fit - they only let PhysX react for one step to the object teleported into the hand (gravity on the robot,
        contact impulses against the fingers) before the velocities were zeroed again. An articulated object's child links, however,
        follow a root teleport only through PhysX itself, so it keeps the steps."""
        try:
            return not bool(getattr(obj, "articulated", False)) and len(getattr(obj, "links", {}) or {}) <= 1
        except Exception:  # noqa: BLE001
            return False

    def __enter__(self):
        if self.obj is None or self.arm is None:
            return self
        if self.eef_in_object is not None:
            import omnigibson as og

            from ispatialgym.sampling import geometry as G

            p, q = self.obj.get_position_orientation()
            world = getattr(self.kit, "world", None)
            # the object's gravity state before the teleport: parked objects have it off persistently (USD), placed ones on
            grav = (not world.gravity_disabled_in_usd(self.obj)) if world is not None and hasattr(world, "gravity_disabled_in_usd") else True
            self._saved = (p.clone(), q.clone(), grav)
            eef = self.kit.robot.eef_links[self.arm].get_position_orientation()
            obj_pose = G.compose((eef[0].float(), eef[1].float()), G.invert((G.t(self.eef_in_object[0]), G.t(self.eef_in_object[1]))))
            self.obj.disable_gravity()
            self.obj.set_position_orientation(obj_pose[0], obj_pose[1])
            self.obj.keep_still()
            # train-fast 2026-10-31 (ATTACH_NO_STEP): no physics step for a single rigid body - the pose set above is already what the sphere
            # fit reads (see ``_step_skippable``); off, or an articulated object: the step and the keep-stills as before
            self._no_step = fast_on("ATTACH_NO_STEP") and self._step_skippable(self.obj)
            if not self._no_step:
                og.sim.step_physics()
                self.kit.robot.keep_still()
                self.obj.keep_still()
        # the sphere fit of ``attach_object`` samples the mesh at random (5–7 spheres for a small object, a different set
        # per call): seed torch around it so every check of one attempt sees the same attached geometry
        import random

        import numpy as np

        cpu_state = th.get_rng_state()
        cuda_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        np_state, py_state = np.random.get_state(), random.getstate()
        th.manual_seed(ATTACH_FIT_SEED)
        np.random.seed(ATTACH_FIT_SEED)
        random.seed(ATTACH_FIT_SEED)
        try:
            # the kit's timed wrapper (``attach_calls`` / ``attach_s`` / ``fit_cache_hits``; train-fast 2026-10-31), the backend on a kit without it
            attach = getattr(self.kit, "attach_object", None)
            if callable(attach):
                attach(self.arm, self.obj)
            else:
                self.kit.backend.attach_object(self.arm, self.obj)
        finally:
            th.set_rng_state(cpu_state)
            if cuda_state is not None:
                th.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
        return self

    def __exit__(self, *exc):
        if self.obj is None or self.arm is None:
            return False
        self.kit.backend.detach_object(self.arm)
        if self._saved is not None:
            import omnigibson as og

            p, q, grav = self._saved
            self.obj.set_position_orientation(p, q)
            world = getattr(self.kit, "world", None)
            if world is not None and hasattr(world, "set_gravity"):
                world.set_gravity(self.obj, grav)  # back to the state it had (parked objects stay gravity-free)
            elif grav:
                self.obj.enable_gravity()
            self.obj.keep_still()
            if not self._no_step:  # ATTACH_NO_STEP: the restored pose is in PhysX and Fabric already (train-fast 2026-10-31)
                og.sim.step_physics()
                self.kit.robot.keep_still()
                self.obj.keep_still()
            self._saved = None
        return False


# ----------------------------------------------------------------------------------------------------------------
# §3.7 paths
# ----------------------------------------------------------------------------------------------------------------


def plan_edge_slide(kit: "RobotKit", stance_from: Stance, stance_to: Stance, posture: str = "untucked", carried: Optional[str] = None,
                    carried_obj: Any = None, carried_arm: str = "right", max_length_m: Optional[float] = None, stride: float = STRIDE_M,
                    trunk: Optional[Tuple[float, float]] = None, eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None,
                    arm_q: Optional[Dict[str, th.Tensor]] = None, margin: float = 0.01, back_m: float = 0.0) -> PathRecord:
    """§3.7 edge slide (owner 2026-09-24): the :func:`edge_slide_waypoints` path between two stances at the same support
    edge, validated exactly like :func:`plan_path` (whole-body checks every ``stride`` m at ``trunk`` / ``posture`` with
    ``carried_obj`` attached), recorded with ``mode = "edge_slide"``. The grid planner is not consulted: the strip along
    the edge at the stances' stand-off is checked against the real collision world (the trav map still carries the baked
    footprints of removed chairs there). Raises ``NavigationError`` ``not_same_edge`` / ``path_too_long`` /
    ``edge_slide_collision``."""
    if not same_edge(stance_from, stance_to):
        raise NavigationError("not_same_edge", f"{stance_from.name} -> {stance_to.name}: not at the same support edge facing it")
    trunk = (stance_from.translate, stance_from.tilt) if trunk is None else trunk
    if arm_q is None:
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        r, l, _ = stance_arm_vectors(kit, stance_from, posture if posture != "untucked" else None)
        arm_q = {"right": r, "left": l}
    waypoints = edge_slide_waypoints(stance_from, stance_to, back_m=back_m)
    length = path_length(waypoints)
    if max_length_m is not None and length > max_length_m:
        raise NavigationError("path_too_long", f"edge slide {length:.2f} m > {max_length_m:.2f} m")
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
        poses = stride_poses(waypoints, stride, max_yaw_step=yaw_step_bound(sweep_radius(kit, trunk, posture, arm_q)))
        ok, bad = path_collision_free(kit, poses, trunk, posture, margin=float(margin), q_arms=arm_q)
    if not ok:
        raise NavigationError("edge_slide_collision", f"edge slide {stance_from.name} -> {stance_to.name} collides at stride {bad}/{len(poses)} ({poses[bad]}) at trunk {trunk}"
                                                      + (f" (margin {float(margin):.3f})" if float(margin) > 0.011 else ""))
    return PathRecord(stance_from.name, stance_to.name, EDGE_SLIDE_MODE, waypoints, length, posture, carried, trunk, stride, len(poses))


def plan_axis_leg(kit: "RobotKit", stance_from: Stance, stance_to: Stance, back_offs: Sequence[float] = (0.0,), **kw) -> PathRecord:
    """Owner 2026-10-13 (disembedding, COMPLAINTS.md global rule: "moving to the place pose, the robot moves only forward /
    backward or left / right, keeping its heading; it can combine the two"): the leg into the place pose as the edge slide of
    :func:`plan_edge_slide` (``kw`` passed on) - drive back, slide along the edge, drive in, the stances' heading throughout -
    at the first extra stand-off of ``back_offs`` whose strides are collision-free; never the grid path, whose intermediate
    waypoints carry the travel heading. The record's ``axis_leg`` names the drive-back used and the moves. Raises
    ``NavigationError`` ``place_leg_heading`` (the stances do not face the same edge at the same yaw within
    ``BP.AXIS_LEG_YAW_TOL_RAD``, or the waypoints break the rule) / ``place_leg_collision`` (every drive-back collides) /
    ``path_too_long``."""
    from ispatialgym.robot import base_path as BP

    if not same_edge(stance_from, stance_to, tol_rad=BP.AXIS_LEG_YAW_TOL_RAD):
        dyaw = math.degrees((float(stance_to.yaw) - float(stance_from.yaw) + math.pi) % (2 * math.pi) - math.pi)
        raise NavigationError("place_leg_heading", f"{stance_from.name} -> {stance_to.name}: not at the same support edge with the same heading "
                                                   f"(yaw {dyaw:+.2f} deg; the leg into the place pose keeps the robot's heading)")
    tried, errors = [], []
    for b in back_offs:
        tried.append(round(float(b), 4))
        try:
            rec = plan_edge_slide(kit, stance_from, stance_to, back_m=float(b), **kw)
        except NavigationError as e:
            if e.reason == "path_too_long":
                raise
            errors.append(f"back {float(b):.2f} m: {e}")
            continue
        why = BP.axis_leg_violation(rec.waypoints)
        if why is not None:
            raise NavigationError("place_leg_heading", f"{stance_from.name} -> {stance_to.name}: {why}")
        rec.axis_leg = {"back_m": round(float(b), 4), "tried_back_m": tried, "moves": [[a, round(m, 4)] for a, m in BP.axis_leg_moves(rec.waypoints)]}
        if errors:
            log.info("%s -> %s: axis-aligned leg at drive-back %.2f m (%s)", stance_from.name, stance_to.name, float(b), "; ".join(errors))
        return rec
    raise NavigationError("place_leg_collision", f"{stance_from.name} -> {stance_to.name}: no forward / backward / lateral leg keeping the heading "
                                                 f"is collision-free ({'; '.join(errors)})"[:1500])


def plan_path(kit: "RobotKit", stance_from: Stance, stance_to: Stance, posture: str = "untucked", carried: Optional[str] = None,
              carried_obj: Any = None, carried_arm: str = "right", rooms: Optional[Sequence[str]] = None, blocked: Sequence[Any] = (),
              max_length_m: Optional[float] = None, stride: float = STRIDE_M, trunk: Optional[Tuple[float, float]] = None,
              eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None, force_global: bool = False, arm_q: Optional[Dict[str, th.Tensor]] = None,
              margin: float = 0.01, edge_slide: bool = False, edge_slide_back_m: float = 0.0, style: str = "recorded",
              probe_face_travel: bool = False, global_blocked: Sequence[Any] = (), face_travel_cfgs: Sequence[Dict[str, float]] = (), arrive_straight_m: float = 0.0,
              face_travel_max_extra_rotation_deg: Optional[float] = None) -> PathRecord:
    """§3.7: the base path from ``stance_from`` to ``stance_to`` (world (x, y, yaw) waypoints), ``mode`` ``local`` or
    ``global``, validated with whole-body checks every ``stride`` m at the navigation posture with ``carried_obj``
    attached to ``carried_arm`` (posed in the hand per ``eef_in_object``, the grasp record's T_obj→eef). ``trunk``
    defaults to the **departing** stance's trunk: the trunk changes at the transition stand-off after the drive (§3.8),
    so the path is driven at the trunk the robot leaves with. ``arm_q`` (per arm) fixes the arm posture of the drive
    (default: the departing stance's recorded arms when it has a configuration, else ``posture``). ``force_global``
    plans on the eroded map end to end (camera relocalization). ``edge_slide`` (owner 2026-09-24): when the two stances
    stand at the same support edge, try the :func:`plan_edge_slide` path first (body facing the edge throughout,
    ``mode = "edge_slide"``) and fall back to the grid planner when it is not available or collides. Raises
    ``NavigationError`` with reason ``no_path`` / ``path_collision`` / ``path_too_long``.

    ``style`` (owner 2026-09-26, ``generation.base_path_style``): ``"face_travel"`` checks the shared ``robot.base_path``
    schedule of the path (:func:`face_travel_poses`) with the same posture, carried object, margin and yaw-lag envelope
    as the polyline check and records ``style = "face_travel"`` when it clears; when a stride of the schedule collides the
    polyline check decides and the record says ``"recorded"`` (never a rejection because of the schedule). A carried leg
    (oracle-fix 2026-11-06, ``generation.carried_base_path_style``) is checked alike: the object attached, the carry arms
    ``arm_q``, the carry ``margin``. ``face_travel_cfgs`` (oracle-fix 2026-11-06, ``generation.face_travel_fallback_cfgs``):
    ``robot.base_path.BasePathConfig`` overrides tried in order when the default schedule collides, before the polyline - the
    first that clears is recorded as ``paths[].face_travel_cfg`` (the oracle builds its schedule with it).
    ``face_travel_max_extra_rotation_deg`` (oracle-fix 2026-11-06, ``generation.face_travel_max_extra_rotation_deg``): a schedule
    that turns the body more than this beyond the polyline as recorded (:func:`total_rotation`) is not checked - the polyline
    decides (a straight 10 cm back-up as face_travel spins 180 deg twice; on 104 IORS-large records face_travel everywhere drove
    544 s / 11899 deg against recorded 431 s / 7496 deg, the rule at 10 deg 403 s / 6894 deg). ``probe_face_travel``
    (carried legs that stay recorded): the schedule is checked at 1 cm, logged and tallied in
    :data:`FACE_TRAVEL_CARRIED_PROBE`, never used."""
    trunk = (stance_from.translate, stance_from.tilt) if trunk is None else trunk
    if arm_q is None:
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        r, l, _ = stance_arm_vectors(kit, stance_from, posture if posture != "untucked" else None)
        arm_q = {"right": r, "left": l}
    a_xy, b_xy = stance_from.base_xy, stance_to.base_xy
    if float(th.linalg.norm(a_xy - b_xy)) < 1e-3 and abs(stance_from.yaw - stance_to.yaw) < 1e-3:
        return PathRecord(stance_from.name, stance_to.name, "local", [tuple(stance_to.base_pose2d)], 0.0, posture, carried, trunk, stride, 0)
    if edge_slide and not force_global and same_edge(stance_from, stance_to):
        try:
            return plan_edge_slide(kit, stance_from, stance_to, posture=posture, carried=carried, carried_obj=carried_obj, carried_arm=carried_arm,
                                   max_length_m=max_length_m, stride=stride, trunk=trunk, eef_in_object=eef_in_object, arm_q=arm_q, margin=margin,
                                   back_m=edge_slide_back_m)
        except NavigationError as e:
            log.info("%s -> %s: edge slide not available (%s); falling back to the grid path", stance_from.name, stance_to.name, e)
    # oracle-fix 2026-11-06 (``arrive_straight_m``; hardback carry, generation.carry_arrive_straight_m): the grid path ends that far behind
    # ``stance_to`` along its own axis, at its heading, and the last leg drives straight in - the turn to the stance's heading happens there,
    # not at the stance (the hardback's carried book, ~0.8 m ahead of the base, swept into the bookcase on the turn at stance C)
    tail: List[Tuple[float, float, float]] = []
    if arrive_straight_m and float(arrive_straight_m) > 0.0:
        fwd = th.tensor([math.cos(float(stance_to.yaw)), math.sin(float(stance_to.yaw))], dtype=b_xy.dtype)
        b_xy = b_xy - fwd * float(arrive_straight_m)
        tail = [(float(b_xy[0]), float(b_xy[1]), float(stance_to.yaw))]
    free_local = local_map(kit, rooms, blocked=blocked)
    a, b = _to_map(kit, a_xy), _to_map(kit, b_xy)
    a_free, b_free = nearest_free_cell(free_local, a, 3), nearest_free_cell(free_local, b, 3)
    waypoints: Optional[List[Tuple[float, float, float]]] = None
    mode = "local"
    if not force_global and a_free is not None and b_free is not None and same_region(free_local, a_free, b_free):
        cells = grid_shortest_path(free_local, a_free, b_free)
        if cells is not None:
            cells = simplify_path(cells, free_local)
            pts = [_to_world(kit, c) for c in cells[1:-1]]
            waypoints = [tuple(stance_from.base_pose2d)] + [(float(p[0]), float(p[1]), float("nan")) for p in pts] + tail + [tuple(stance_to.base_pose2d)]
    if waypoints is None:
        mode = "global"
        free_g = eroded_map(kit)
        if global_blocked:
            # oracle-fix 2026-11-04 round 4 (core.path_keep_off_footprints): footprints the global path keeps off too (the eroded map alone let the
            # pillow's carried path turn 0.41 m from a bed side - a wheel sphere 8.7 mm from it)
            from ispatialgym.sampling.scene_variants import polygon_cells

            free_g = free_g.clone()
            for poly in global_blocked:
                cells = polygon_cells(kit.world.scene.trav_map, poly, 0)
                if len(cells):
                    free_g[cells[:, 0], cells[:, 1]] = False
        ga, gb = nearest_free_cell(free_g, a, 20), nearest_free_cell(free_g, b, 20)
        if ga is None or gb is None:
            raise NavigationError("no_path", f"no eroded-free cell near {stance_from.name} / {stance_to.name}")
        cells = grid_shortest_path(free_g, ga, gb)
        if cells is None:
            raise NavigationError("no_path", f"{stance_from.name} -> {stance_to.name}: disconnected on the eroded map")
        cells = simplify_path(cells, free_g)
        pts = [_to_world(kit, c) for c in cells]
        waypoints = [tuple(stance_from.base_pose2d)] + [(float(p[0]), float(p[1]), float("nan")) for p in pts] + tail + [tuple(stance_to.base_pose2d)]
    # yaws of the intermediate waypoints: heading of the following segment
    filled: List[Tuple[float, float, float]] = []
    for i, w in enumerate(waypoints):
        yaw = w[2]
        if yaw != yaw:  # nan
            nxt = waypoints[min(i + 1, len(waypoints) - 1)]
            yaw = math.atan2(nxt[1] - w[1], nxt[0] - w[0]) if (nxt[0] != w[0] or nxt[1] != w[1]) else filled[-1][2]
        filled.append((float(w[0]), float(w[1]), float(yaw)))
    length = path_length(filled)
    if max_length_m is not None and length > max_length_m:
        raise NavigationError("path_too_long", f"{length:.2f} m > {max_length_m:.2f} m")
    record_style: Optional[str] = None
    if style == "face_travel":
        # oracle-fix 2026-11-06: the default schedule, then each face_travel_cfgs override (one attachment for all of them)
        cfgs: List[Optional[Dict[str, float]]] = [None] + [dict(c) for c in (face_travel_cfgs or ()) if c]
        what = f" carrying {carried}" if carried_obj is not None else ""
        if face_travel_max_extra_rotation_deg is not None:
            # oracle-fix 2026-11-06: a schedule that turns the body more than the polyline as recorded does (a straight leg whose ends do
            # not face the way: the schedule turns to the travel heading and back) is not smoother - not checked, the polyline decides
            rot_rec, keep = total_rotation(filled), []
            for cfg in cfgs:
                rot_ft = total_rotation(face_travel_poses(filled, stride, max_yaw_step=None, cfg=cfg))
                if rot_ft > rot_rec + math.radians(float(face_travel_max_extra_rotation_deg)):
                    log.info("face_travel: %s path %s -> %s%s %.2f m: the schedule%s turns the body %.0f deg, the polyline %.0f deg (> +%.0f deg) - not checked",
                             mode, stance_from.name, stance_to.name, what, length, "" if cfg is None else f" with {cfg}", math.degrees(rot_ft),
                             math.degrees(rot_rec), float(face_travel_max_extra_rotation_deg))
                else:
                    keep.append(cfg)
            cfgs = keep
        if cfgs:
            with attached(kit, carried_arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
                step = yaw_step_bound(sweep_radius(kit, trunk, posture, arm_q))
                for k, cfg in enumerate(cfgs):
                    ft = face_travel_poses(filled, stride, max_yaw_step=step, cfg=cfg)
                    ft_swept = yaw_lag_envelope(ft, max_yaw_step=step)
                    ok_ft, bad_ft = path_collision_free(kit, ft_swept, trunk, posture, margin=float(margin), q_arms=arm_q)
                    if ok_ft:
                        log.info("face_travel: %s path %s -> %s%s %.2f m recorded as the face_travel schedule%s (%d schedule poses, %d swept with the yaw-lag envelope)",
                                 mode, stance_from.name, stance_to.name, what, length, "" if cfg is None else f" with {cfg}", len(ft), len(ft_swept))
                        return PathRecord(stance_from.name, stance_to.name, mode, filled, length, posture, carried, trunk, stride, len(ft), style="face_travel",
                                          face_travel_cfg=cfg)
                    log.info("face_travel: %s path %s -> %s%s: the schedule%s collides at swept pose %d/%d (%s)%s", mode, stance_from.name, stance_to.name, what,
                             "" if cfg is None else f" with {cfg}", bad_ft, len(ft_swept), tuple(round(v, 3) for v in ft_swept[bad_ft]),
                             " - the polyline is checked and recorded instead" if k == len(cfgs) - 1 else "")
        record_style = "recorded"
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
        radius = sweep_radius(kit, trunk, posture, arm_q)
        step = yaw_step_bound(radius)
        poses = stride_poses(filled, stride, max_yaw_step=step)
        swept = stride_poses(filled, stride, yaw_envelope=True, max_yaw_step=step)  # the strides + the tracker's yaw lag on the turns (owner 2026-09-25)
        ok, bad = path_collision_free(kit, swept, trunk, posture, margin=float(margin), q_arms=arm_q)
        # oracle-fix 2026-11-06: ISPATIALGYM_PATH_BLAME turns on this blame alone (STANCE_BLAME also turns on the stance / roll blames: 16x slower)
        blame = path_blame(kit, swept[bad], trunk, posture, arm_q, float(margin)) if not ok and (os.environ.get("ISPATIALGYM_STANCE_BLAME")
                                                                                              or os.environ.get("ISPATIALGYM_PATH_BLAME")) else ""
    log.debug("%s path %s -> %s: heading step <= %.2f deg between checked poses (farthest robot / carried point %s from the base centre): %d strides, %d swept poses",
              mode, stance_from.name, stance_to.name, math.degrees(step), "unknown" if radius is None else f"{radius:.2f} m", len(poses), len(swept))
    if not ok:
        raise NavigationError("path_collision", f"{mode} path {stance_from.name} -> {stance_to.name} collides at swept pose {bad}/{len(swept)} ({tuple(round(v, 3) for v in swept[bad])}, "
                                                f"{len(poses)} strides + the yaw-lag envelope) at trunk {trunk}"
                                                + (f" (margin {float(margin):.3f})" if float(margin) > 0.011 else "") + blame)
    if probe_face_travel and carried_obj is not None:
        try:
            ft_swept = yaw_lag_envelope(face_travel_poses(filled, stride, max_yaw_step=step), max_yaw_step=step)
            with attached(kit, carried_arm, carried_obj, eef_in_object=eef_in_object):
                ok_p, bad_p = path_collision_free(kit, ft_swept, trunk, posture, margin=0.01, q_arms=arm_q)
            FACE_TRAVEL_CARRIED_PROBE["checked"] += 1
            FACE_TRAVEL_CARRIED_PROBE["collides"] += 0 if ok_p else 1
            log.info("FACE_TRAVEL carried-leg probe (log only, the leg stays recorded): %s path %s -> %s carrying %s: schedule %s at 1 cm (tally %d / %d collide)",
                     mode, stance_from.name, stance_to.name, carried, "clear" if ok_p else f"collides at swept pose {bad_p}/{len(ft_swept)}",
                     FACE_TRAVEL_CARRIED_PROBE["collides"], FACE_TRAVEL_CARRIED_PROBE["checked"])
        except Exception as e:  # noqa: BLE001 - a probe never fails the path
            log.info("FACE_TRAVEL carried-leg probe unavailable (%r)", e)
    return PathRecord(stance_from.name, stance_to.name, mode, filled, length, posture, carried, trunk, stride, len(poses), style=record_style)


def travel_length(kit: "RobotKit", a_xy, b_xy, cache: Optional[Dict[Any, float]] = None) -> float:
    """W3 (owner 2026-09-20 22:15): the base travel from ``a_xy`` to ``b_xy`` = the shortest path on the eroded map (no
    whole-body checks: an ordering key, not a record), the straight distance when either end has no eroded-free cell
    within 20 cells or the two are disconnected. ``cache`` (keyed by the rounded end points) makes the per-candidate
    ordering of a stance pool cheap."""
    a_xy, b_xy = th.as_tensor(a_xy, dtype=th.float32)[:2], th.as_tensor(b_xy, dtype=th.float32)[:2]
    key = (tuple(round(float(v), 2) for v in a_xy.tolist()), tuple(round(float(v), 2) for v in b_xy.tolist()))
    if cache is not None and key in cache:
        return cache[key]
    straight = float(th.linalg.norm(a_xy - b_xy))
    length = straight
    try:
        free_g = eroded_map(kit)
        ga, gb = nearest_free_cell(free_g, _to_map(kit, a_xy), 20), nearest_free_cell(free_g, _to_map(kit, b_xy), 20)
        if ga is not None and gb is not None:
            cells = grid_shortest_path(free_g, ga, gb)
            if cells:
                pts = [_to_world(kit, c) for c in simplify_path(cells, free_g)]
                length = max(straight, path_length([(float(p[0]), float(p[1]), 0.0) for p in pts]))
    except Exception:  # noqa: BLE001 - an ordering key never fails the attempt
        length = straight
    if cache is not None:
        cache[key] = length
    return length


# ----------------------------------------------------------------------------------------------------------------
# §3.8 transitions
# ----------------------------------------------------------------------------------------------------------------


def transition(kit: "RobotKit", stance_from: Stance, stance_to: Stance, carried_obj: Any = None, carried_arm: str = "right",
               posture: Optional[str] = None, margin: float = 0.01, n_samples: int = TRANSITION_SAMPLES,
               eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None, min_standoff: Optional[float] = None,
               check_drive_in: bool = True, path_end_standoff: Optional[float] = None,
               arm_q_override: Optional[Dict[str, th.Tensor]] = None, trunk_from: Optional[Tuple[float, float]] = None,
               extra_max: float = TRANSITION_EXTRA_MAX, blame: Optional[Any] = None, arms_to_q: Optional[Dict[str, th.Tensor]] = None,
               fingers: Optional[float] = None) -> Optional[Transition]:
    """§3.8: the smallest ``s_T ∈ transition_standoffs(s)`` (``s`` = the arriving stance's stand-off, driven back along
    -inward) at which (a) the swept trunk motion (``n_samples`` trunks between the two stances' values, both endpoints
    included) and (b) the **drive-in** ``s_T → s`` (``check_drive_in``, strides of ``STRIDE_M`` at the arriving trunk)
    are whole-body collision-free with the *arriving* stance's arms (its recorded configuration / working arms untucked /
    idle arm in its A6 posture — the oracle moves the arms before the trunk motion; the robot then drives in with them,
    as ``prompt0.connect`` checks) and the carried object attached / posed in the hand per ``eef_in_object``.

    ``min_standoff``: candidates below it are not tried (a re-sweep with the objects placed must never lower the
    recorded s_T: the recorded path ends at its drive-back pose). ``path_end_standoff``: the stand-off where the recorded
    path ends (``s`` when it ends at the stance, the recorded s_T when ``prompt0.connect`` planned it to the drive-back
    pose); a candidate beyond it adds a drive-back from the path's end to the new pose, checked at the *departing*
    trunk and arms. Returns the ``Transition`` or None (caller: ``Reject("no_transition")``). Stances with equal trunks
    need no sweep (``n_samples = 0``; the drive-in is still checked when ``s_T > s``).

    ``arm_q_override`` (arm -> 7 joint values): arms that keep a configuration through the transition — the arm that
    CARRIES the object holds its lift configuration (the oracle never moves a holding arm: ``move_idle_arms`` skips it),
    so the sweep / drive-in / drive-back are checked with that arm there instead of the stances' recorded postures
    (IORS large-scale 0919-211642: the sweep into the placement crouch was checked with the untucked arm while the
    oracle held the sandal at the lift pose, and the sandal brushed the table top). ``trunk_from``: the trunk the robot
    arrives with when it differs from ``stance_from``'s (a departure transition straightened it to a travel trunk).

    ``arms_to_q`` (arm -> 7 joint values; oracle-fix 2026-11-06, the shelf back-off ``generation.shelf_backoff_transit_m``): arms the
    oracle moves to these joints AFTER the drive-back, before the trunk motion (the record's ``arms_q``, ``execute.move_idle_arms``) -
    the sweep and the drive-in are checked with them, the drive-back from the path's end with the departing arms as they are; a
    holding arm's ``arm_q_override`` wins. ``fingers``: the arriving arms' finger opening for the sweep and the drive-in (the opening
    the oracle pre-shapes to before the arms move, ``transitions[].preshape_m``); None = the arriving stance's, as before."""
    tf, tt = (stance_from.translate, stance_from.tilt) if trunk_from is None else (float(trunk_from[0]), float(trunk_from[1])), (stance_to.translate, stance_to.tilt)
    s = float(stance_to.stand_off if stance_to.stand_off is not None else 0.0)
    same_trunk = abs(tf[0] - tt[0]) < 1e-6 and abs(tf[1] - tt[1]) < 1e-6
    floor = s if min_standoff is None else max(s, float(min_standoff))
    if same_trunk and floor <= s + 1e-9:
        return Transition(stance_from.name, stance_to.name, s, trunk_from=tf, trunk_to=tt, n_samples=0)
    inward = stance_to.inward
    x0, y0, yaw = (float(v) for v in stance_to.base_pose2d)
    sweep = trunk_sweep(tf, tt, n_samples)
    from ispatialgym.sampling.v2.ik import stance_arm_vectors

    # the arms of the arriving stance (its recorded configuration / working arms untucked / idle arm in its A6 posture):
    # the oracle moves the idle arm there before the trunk motion (``idle_posture`` / ``arms_q`` in the sidecar)
    right_arm, left_arm, f_to = stance_arm_vectors(kit, stance_to, posture)
    fingers = f_to if fingers is None else float(fingers)
    arms_to = {"right": right_arm, "left": left_arm}
    r_from, l_from, f_from = stance_arm_vectors(kit, stance_from)
    arms_from = {"right": r_from, "left": l_from}
    for a, q in (arms_to_q or {}).items():  # oracle-fix 2026-11-06: moved there after the drive-back; the departing arms keep theirs
        arms_to[a] = q
    for a, q in (arm_q_override or {}).items():
        arms_to[a], arms_from[a] = q, q
    right_arm, left_arm = arms_to["right"], arms_to["left"]
    end_back = None if path_end_standoff is None else max(0.0, float(path_end_standoff) - s)
    why: List[str] = []
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
        for s_T in transition_standoffs(s, extra_max=float(extra_max)):
            if s_T < floor - 1e-9:
                continue
            back = s_T - s
            bx, by = x0 - float(inward[0]) * back, y0 - float(inward[1]) * back
            if back > 0 and not kit.trav_free((bx, by)):
                why.append(f"s_T {s_T:.2f}: drive-back pose off the free floor")
                continue
            if not same_trunk:  # (a) the trunk sweep at the drive-back pose with the arriving arms
                qs = th.stack([kit.q_at_base_pose(kit.q_config(arms="untucked", translate=tr, tilt=ti, right_arm=right_arm, left_arm=left_arm, fingers=fingers),
                                                  bx, by, yaw) for tr, ti in sweep])
                free = kit.collision_free(qs, margin=margin)
                if not bool(free.all()):
                    bad = int(th.nonzero(~free)[0])
                    detail = ""
                    last = transition_standoffs(s, extra_max=float(extra_max))[-1]
                    if blame is not None and (not why or s_T >= last - 1e-9):  # the blame for the first and the last candidate only (it is slow)
                        try:
                            detail = f" - {blame(qs[bad], [], margin)}"
                        except Exception:  # noqa: BLE001
                            detail = ""
                    why.append(f"s_T {s_T:.2f}: sweep sample {bad + 1}/{len(sweep)} at trunk {tuple(round(float(v), 2) for v in sweep[bad])} collides{detail}")
                    continue
            if back > 0 and check_drive_in:  # (b) the drive-in s_T -> s at the arriving trunk and arms
                drive = stride_poses([(bx, by, yaw), (x0, y0, yaw)], STRIDE_M)
                ok, bad = path_collision_free(kit, drive, tt, "untucked", margin=margin, q_arms=arms_to, fingers=fingers)
                if not ok:
                    why.append(f"s_T {s_T:.2f}: drive-in collides at stride {bad}/{len(drive)}")
                    continue
            if end_back is not None and back > end_back + 1e-9:  # (c) the extra drive-back from the path's end, departing trunk / arms
                ex, ey = x0 - float(inward[0]) * end_back, y0 - float(inward[1]) * end_back
                extra = stride_poses([(ex, ey, yaw), (bx, by, yaw)], STRIDE_M)
                ok, bad = path_collision_free(kit, extra, tf, "untucked", margin=margin, q_arms=arms_from, fingers=f_from)
                if not ok:
                    why.append(f"s_T {s_T:.2f}: drive-back from the path's end collides at stride {bad}/{len(extra)}")
                    continue
            return Transition(stance_from.name, stance_to.name, s_T, trunk_from=tf, trunk_to=tt, n_samples=0 if same_trunk else n_samples, holding=stance_to.carried)
    log.info("transition %s -> %s: no s_T (trunk %s -> %s%s): %s", stance_from.name, stance_to.name, tuple(round(v, 2) for v in tf), tuple(round(v, 2) for v in tt),
             f", carrying {getattr(carried_obj, 'name', carried_obj)}" if carried_obj is not None else "", "; ".join(why[:3] + (["..."] if len(why) > 4 else []) + why[-1:] if len(why) > 3 else why))
    return None


def recorded_arms_to(extras: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, th.Tensor]], Optional[float]]:
    """oracle-fix 2026-11-06 (the shelf back-off): ``(arms_to_q, fingers)`` for :func:`transition` from a transition record's extras - the
    arm joints the oracle moves the arms to after the drive-back (``arms_q``: arm -> 7) and the pre-shaped finger opening (``preshape_m``:
    arm -> m, the widest); ``(None, None)`` without them (every record before 2026-11-06)."""
    ex = extras or {}
    out: Dict[str, th.Tensor] = {}
    for a, v in (ex.get("arms_q") or {}).items():
        try:
            q = th.as_tensor([float(x) for x in v], dtype=th.float32)
        except (TypeError, ValueError):
            continue
        if q.numel() == 7:
            out[str(a)] = q
    shape = [float(v) for v in (ex.get("preshape_m") or {}).values() if v is not None]
    return (out or None), (max(shape) if shape else None)


def departure_pose(stance_from: Stance, s_T: float) -> Tuple[float, float, float]:
    """Drive-back pose of a departure transition: ``stance_from`` moved ``s_T - stand_off`` backwards along its own
    forward direction (``conventions.departure_transition_poses``)."""
    s = float(stance_from.stand_off if stance_from.stand_off is not None else 0.0)
    back = float(s_T) - s
    x0, y0, yaw = (float(v) for v in stance_from.base_pose2d)
    return (x0 - back * math.cos(yaw), y0 - back * math.sin(yaw), yaw)


def departure_transition(kit: "RobotKit", stance_from: Stance, stance_to: Stance, travel_trunk: Tuple[float, float], carried_obj: Any = None,
                         carried_arm: str = "right", margin: float = 0.01, n_samples: int = TRANSITION_SAMPLES,
                         eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None, min_standoff: Optional[float] = None,
                         arm_q_override: Optional[Dict[str, th.Tensor]] = None, arms_stance: Optional[Stance] = None) -> Optional[Transition]:
    """Owner 2026‑09‑19 21:56 ("make torso straight, then navigate"): the **departure** transition at ``stance_from`` —
    the smallest ``s_T ∈ transition_standoffs(stance_from.stand_off)`` at which (a) the drive-back ``stance_from ->
    departure_pose(s_T)`` (strides at the departing trunk, when ``s_T > s``) and (b) the trunk sweep ``stance_from's
    trunk -> travel_trunk`` at that pose are whole-body collision-free with the carried object attached (posed in the
    hand) and the arms the oracle drives with: ``arms_stance``'s recorded posture (default: the *arriving* stance's —
    the oracle moves the idle arm there before the trunk motion, as at an arrival) with ``arm_q_override`` on top (the
    holding arm at its lift configuration). "Preferably along the way it has moved before": the drive-back runs along
    the departing stance's own -forward, i.e. back out the way the arrival drive-in came. Returns the departure
    ``Transition`` (``at = "departure"``, ``trunk_to = travel_trunk``) or None.

    ``extra_max`` of :func:`transition`: how far beyond the arriving stand-off the s_T candidates go (§3.8: 0.5 m); a
    CARRY arrival into a deep crouch may need more — the hand frozen at the lift pose swings ~0.5 m forward and ~0.9 m
    down during the sweep, so it must start far enough from the partner's top (0920-033402: no s_T in s..s+0.5)."""
    tf = (float(stance_from.translate), float(stance_from.tilt))
    tt = (float(travel_trunk[0]), float(travel_trunk[1]))
    s = float(stance_from.stand_off if stance_from.stand_off is not None else 0.0)
    same_trunk = abs(tf[0] - tt[0]) < 1e-6 and abs(tf[1] - tt[1]) < 1e-6
    floor = s if min_standoff is None else max(s, float(min_standoff))
    if same_trunk and floor <= s + 1e-9:
        return Transition(stance_from.name, stance_to.name, s, trunk_from=tf, trunk_to=tt, n_samples=0, holding=stance_from.carried, at="departure")
    from ispatialgym.sampling.v2.ik import stance_arm_vectors

    right_arm, left_arm, fingers = stance_arm_vectors(kit, arms_stance if arms_stance is not None else stance_to)
    arms = {"right": right_arm, "left": left_arm}
    for a, q in (arm_q_override or {}).items():
        arms[a] = q
    x0, y0, yaw = (float(v) for v in stance_from.base_pose2d)
    sweep = trunk_sweep(tf, tt, n_samples)
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
        for s_T in transition_standoffs(s):
            if s_T < floor - 1e-9:
                continue
            bx, by, _ = departure_pose(stance_from, s_T)
            back = s_T - s
            if back > 0 and not kit.trav_free((bx, by)):
                continue
            if back > 0:  # (a) the drive-back at the departing trunk with the arms as they are
                drive = stride_poses([(x0, y0, yaw), (bx, by, yaw)], STRIDE_M)
                ok, _ = path_collision_free(kit, drive, tf, "untucked", margin=margin, q_arms=arms, fingers=fingers)
                if not ok:
                    continue
            if not same_trunk:  # (b) the trunk sweep to the travel trunk at the drive-back pose
                qs = th.stack([kit.q_at_base_pose(kit.q_config(arms="untucked", translate=tr, tilt=ti, right_arm=arms["right"], left_arm=arms["left"], fingers=fingers),
                                                  bx, by, yaw) for tr, ti in sweep])
                if not bool(kit.collision_free(qs, margin=margin).all()):
                    continue
            return Transition(stance_from.name, stance_to.name, s_T, trunk_from=tf, trunk_to=tt, n_samples=0 if same_trunk else n_samples,
                              holding=stance_from.carried, at="departure")
    return None


# ----------------------------------------------------------------------------------------------------------------
# §3.3 parking
# ----------------------------------------------------------------------------------------------------------------


def park_pose(kit: "RobotKit", rooms: Sequence[str], keep_away: Sequence[Any] = (), min_dist: float = PARK_MIN_DIST, floor: int = 0) -> th.Tensor:
    """§3.3: a free (eroded) cell inside the task rooms' floor, >= ``min_dist`` from every AABB in ``keep_away``
    (spawned objects and supports); of these admissible cells the one nearest to their centre that keeps the tucked
    robot clear of every scene object (``SimWorld.choose_park_cell``, shared with the world-load park cell)."""
    trav = kit.world.scene.trav_map
    free = kit.eroded_trav_map(floor) == 255
    seg = kit.world.scene.seg_map
    cells = th.nonzero(free)
    if len(cells) == 0:
        raise NavigationError("no_park", "no free traversable cell")
    if rooms and seg.room_ins_map.shape == free.shape:
        ids = [seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id]
        if ids:
            in_room = th.zeros(len(cells), dtype=th.bool)
            for i in ids:
                in_room |= seg.room_ins_map[cells[:, 0], cells[:, 1]] == i
            if bool(in_room.any()):
                cells = cells[in_room]
    pts = trav.map_to_world(cells.float()) if len(cells) else th.zeros(0, 2)
    ok = th.ones(len(pts), dtype=th.bool)
    for obj in keep_away:
        try:
            lo, hi = obj.aabb
        except Exception:
            continue
        lo2, hi2 = lo[:2], hi[:2]
        d = th.maximum(th.maximum(lo2 - pts, pts - hi2), th.zeros_like(pts))
        ok &= th.linalg.norm(d, dim=1) >= min_dist
    if not bool(ok.any()):
        raise NavigationError("no_park", f"no cell >= {min_dist} m from the task objects inside {list(rooms)}")
    pts = pts[ok]
    centre = pts.mean(dim=0)
    return pts[kit.world.choose_park_cell(pts, th.linalg.norm(pts - centre, dim=1), "park_pose")]
