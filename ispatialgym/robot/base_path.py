"""The driven base path shared by the sampler and the oracle (oracle doc §3.7; owner decisions 2026-09-22).

A recorded base path is a polyline of ``[x, y, yaw]`` waypoints. The sampler validates what the robot will do
(whole-body checks along the drive, the obstructed camera-relocalization view rule) and the oracle drives it, so the
geometry the two sides derive from the polyline lives here once, pure (``math`` + torch, no simulator):

* **simplification** — Douglas-Peucker on the interior waypoints within ``simplify_tol_m`` (one map cell): the dense
  grid path of the pilot's camera relocalization records (one waypoint per cell, headings alternating 0° / 45° — the
  robot slowed at every stair and zig-zagged, 0916-141035) collapses to its real corners; a path the sampler already
  shortcut by line of sight keeps every corner (jogs smaller than the tolerance aside);
* **fillets** — every interior corner of turn θ ≤ ``fillet_max_turn_deg`` is rounded by a circular arc tangent to both
  segments, its radius bounded by the deviation budget ``fillet_deviation_m`` from the corner vertex
  (r (sec(θ/2) − 1) ≤ budget) and by the adjacent segments (arcs never overlap); a sharper corner, or one whose arc
  would be smaller than ``fillet_min_radius_m``, stays sharp: the robot stops and turns on the spot there;
* **yaw** (``face_travel``, hybrid end turns) — the body faces the heading of the straights; each corner's heading
  change is blended (quintic smoother-step, C²) over a **turn window** centred on the corner that is at least the
  fillet arc and at least ``1.875 |θ| / corner_yaw_rate_rad_per_m`` long, so the body turns no faster per metre than
  the cruise speed and the yaw-rate limit allow (Sprunk et al.: the rotation is decoupled from the tight position arc
  — through a 90° corner of 12 cm radius the body would otherwise have to spin 90° within 19 cm of travel and crawl);
  the change from the departing yaw to the first heading and from the last heading to the goal yaw is turned **on the
  spot** when it exceeds ``spot_turn_deg`` (a robot facing away from its path turns first, then drives) and otherwise
  **blended** over the first / last ``blend_m`` of arc length while driving.

:class:`BasePath` is arc-length parameterised: ``pose(s)`` for ``s ∈ [0, length]`` plus the :class:`Dwell` rotations
(on-the-spot turns of zero arc length at the start, at sharp corners, at the end). The sampler samples it with
``stride_poses`` (every stride plus the dwell sweeps) for the whole-body check and with ``drive_poses`` for the view
rule; the oracle turns it into the tracker's ``(x, y, yaw)`` reference with :func:`time_parameterise`: a trapezoidal
speed profile with v ≤ v_lin, v ≤ sqrt(a_lin / |κ|) through the arcs (lateral acceleration), v ≤ ω_max / |ψ'(s)| and
v ≤ sqrt(a_yaw / |ψ''(s)|) where the yaw blends turn the body, zero at the ends and at the sharp corners, ramps at
``a_lin`` (forward / backward passes), the dwells as trapezoidal yaw profiles at ``ω_max`` / ``a_yaw``.

Both sides must build the path with the same :class:`BasePathConfig` (the module default unless a run overrides the
oracle's ``MotionsConfig.face_travel``): what the sampler checked is what the oracle drives.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch as th

#: ``paths[].style`` / ``Navigate.style``: the recorded polyline tracked as recorded (interior yaws blended linearly per
#: segment, the corner-aware trapezoid of ``Motions._base_reference``) vs. the driven schedule of this module.
RECORDED = "recorded"
FACE_TRAVEL = "face_travel"
STYLES = (RECORDED, FACE_TRAVEL)

#: yaw step between the whole-body-checked samples of an on-the-spot turn (rad, ~11°)
DWELL_CHECK_STEP_RAD = 0.2

Pose2D = Tuple[float, float, float]


def wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


#: peak slope of the quintic blend (at u = 1/2): a blend of Δ over a window W turns at most ``BLEND_PEAK_SLOPE Δ / W``
BLEND_PEAK_SLOPE = 1.875


def _smooth(u: float) -> float:
    """Quintic smoother-step on [0, 1] (clamped): C² — the yaw acceleration is continuous where two turn windows meet
    (an S-curve's rate reversal), which the cubic smooth-step is not (its second derivative jumps at the window ends
    and the yaw-acceleration limit then cuts the speed there)."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _smooth_d(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 30.0 * u * u * (1.0 - u) * (1.0 - u)


def _smooth_dd(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)


@dataclass(frozen=True)
class BasePathConfig:
    #: Douglas-Peucker tolerance of the interior waypoints (m; one map cell); 0 keeps every recorded waypoint
    simplify_tol_m: float = 0.10
    #: largest distance of a fillet arc from the corner vertex it rounds (m)
    fillet_deviation_m: float = 0.05
    #: corners turning more than this stay sharp (stop, turn on the spot, go on); deg
    fillet_max_turn_deg: float = 120.0
    #: a fillet whose radius would fall below this is dropped for a sharp corner (m)
    fillet_min_radius_m: float = 0.02
    #: heading changes below this are not corners at all (deg)
    fillet_min_turn_deg: float = 1.0
    #: the body yaw rate per metre driven the corner turn windows are sized for (rad / m): a window of
    #: ``BLEND_PEAK_SLOPE |θ| / rate`` keeps the peak yaw rate of the blend at ``rate`` — the oracle's yaw limit over its
    #: cruise speed (0.64 rad/s / 0.4 m/s at the defaults), so a corner costs no speed beyond what its arc's lateral
    #: acceleration asks
    corner_yaw_rate_rad_per_m: float = 1.6
    #: arc length over which an end yaw change ≤ ``spot_turn_deg`` is blended while driving (m; at most half the path
    #: when both ends blend)
    blend_m: float = 1.5
    #: end yaw changes above this are turned on the spot (deg) — the hybrid policy (owner 2026-09-22)
    spot_turn_deg: float = 90.0


def config_with(overrides: Optional[Dict[str, float]] = None, base: Optional[BasePathConfig] = None) -> BasePathConfig:
    """``base`` (default :class:`BasePathConfig`) with the fields of ``overrides`` replaced — a path record's own schedule
    configuration (oracle-fix 2026-11-06, ``paths[].face_travel_cfg``: the schedule the sampler checked when the default one
    collided, ``generation.face_travel_fallback_cfgs``), built alike on both sides. ``ValueError`` for a key that is not a field."""
    import dataclasses

    base = base or BasePathConfig()
    if not overrides:
        return base
    names = {f.name for f in dataclasses.fields(BasePathConfig)}
    bad = sorted(set(overrides) - names)
    if bad:
        raise ValueError(f"base path config: unknown field(s) {bad} (one of {sorted(names)})")
    return dataclasses.replace(base, **{k: float(v) for k, v in overrides.items()})


@dataclass(frozen=True)
class Line:
    s0: float
    s1: float
    p0: Tuple[float, float]
    heading: float

    def position(self, s: float) -> Tuple[float, float]:
        d = min(max(s - self.s0, 0.0), self.s1 - self.s0)
        return self.p0[0] + d * math.cos(self.heading), self.p0[1] + d * math.sin(self.heading)

    def tangent(self, s: float) -> float:
        return self.heading

    curvature = 0.0


@dataclass(frozen=True)
class Arc:
    s0: float
    s1: float
    centre: Tuple[float, float]
    radius: float
    phi0: float  # polar angle of the arc's start about the centre
    sign: float  # +1 left turn (counter-clockwise), -1 right turn
    heading0: float  # tangent heading at the start

    def position(self, s: float) -> Tuple[float, float]:
        d = min(max(s - self.s0, 0.0), self.s1 - self.s0)
        phi = self.phi0 + self.sign * d / self.radius
        return self.centre[0] + self.radius * math.cos(phi), self.centre[1] + self.radius * math.sin(phi)

    def tangent(self, s: float) -> float:
        d = min(max(s - self.s0, 0.0), self.s1 - self.s0)
        return self.heading0 + self.sign * d / self.radius

    @property
    def curvature(self) -> float:
        return self.sign / self.radius


@dataclass(frozen=True)
class Dwell:
    """An on-the-spot rotation of zero arc length at ``s``: ``start`` (departing yaw → first heading), ``corner`` (the
    tangent jump of a sharp corner; its yaws are the schedule's yaw on either side of ``s``), ``end`` (last heading →
    goal yaw)."""

    s: float
    kind: str
    yaw_from: float
    yaw_to: float


def douglas_peucker(points: Sequence[Sequence[float]], tol: float) -> List[int]:
    """Indices of the points kept by Douglas-Peucker on the xy polyline (first and last always); ``tol <= 0`` keeps all."""
    n = len(points)
    if n <= 2 or tol <= 0.0:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        (x0, y0), (x1, y1) = (float(points[i][0]), float(points[i][1])), (float(points[j][0]), float(points[j][1]))
        L = math.hypot(x1 - x0, y1 - y0)
        best, bi = -1.0, -1
        for k in range(i + 1, j):
            x, y = float(points[k][0]), float(points[k][1])
            d = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / L if L > 1e-12 else math.hypot(x - x0, y - y0)
            if d > best:
                best, bi = d, k
        if best > tol:
            keep[bi] = True
            stack.append((i, bi))
            stack.append((bi, j))
    return [k for k in range(n) if keep[k]]


class BasePath:
    """The driven schedule of a recorded polyline (see the module docstring). Build with :meth:`build`."""

    def __init__(self, pieces: List[object], dwells: List[Dwell], waypoints: List[Pose2D], cfg: BasePathConfig,
                 yaw_start: float, yaw_goal: float, delta0: float, delta1: float, blend0: float, blend1: float,
                 heading0: float = 0.0, turns: Sequence[Tuple[float, float, float]] = (), jumps: Sequence[Tuple[float, float]] = ()):
        self.pieces = pieces
        self.dwells = dwells
        self.waypoints = waypoints  # the simplified polyline with yaws (departing, heading of the following segment, goal)
        self.cfg = cfg
        self.length = float(pieces[-1].s1) if pieces else 0.0
        self.yaw_start = float(yaw_start)  # body yaw when the drive begins (after the start dwell, if any)
        self.yaw_goal = float(yaw_goal)
        self._delta0, self._delta1 = float(delta0), float(delta1)
        self._b0, self._b1 = float(blend0), float(blend1)
        self._h0 = float(heading0)  # heading of the first straight
        self._turns = [(float(s), float(w), float(t)) for s, w, t in turns]  # (window centre, window length, signed turn) per fillet
        self._jumps = [(float(s), float(t)) for s, t in jumps]  # (arc length, signed turn) per sharp corner (dwell)
        self._s0 = [p.s0 for p in pieces]

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, waypoints: Sequence[Sequence[float]], cfg: Optional[BasePathConfig] = None) -> "BasePath":
        """``waypoints``: ``[x, y, yaw]`` (yaw may be NaN / absent for interior waypoints — it is recomputed anyway; the
        first / last yaw default to the first / last heading when missing)."""
        cfg = cfg or BasePathConfig()
        raw: List[Tuple[float, float, float]] = []
        for w in waypoints:
            x, y = float(w[0]), float(w[1])
            yaw = float(w[2]) if len(w) > 2 and w[2] is not None else float("nan")
            raw.append((x, y, yaw))
        if not raw:
            raise ValueError("BasePath.build: no waypoints")
        # consecutive duplicates collapse (the departing yaw of the first, the goal yaw of the last are kept)
        pts: List[Tuple[float, float, float]] = [raw[0]]
        for p in raw[1:]:
            if math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 1e-6:
                pts[-1] = (pts[-1][0], pts[-1][1], p[2] if not math.isnan(p[2]) else pts[-1][2])
            else:
                pts.append(p)
        yaw0 = raw[0][2]
        yaw_goal = raw[-1][2]
        if len(pts) == 1:
            x, y, _ = pts[0]
            yaw0 = 0.0 if math.isnan(yaw0) else yaw0
            yaw_goal = yaw0 if math.isnan(yaw_goal) else yaw_goal
            dwells = [Dwell(0.0, "end", yaw0, yaw_goal)] if abs(wrap_angle(yaw_goal - yaw0)) > 1e-6 else []
            return cls([], dwells, [(x, y, yaw0)] if not dwells else [(x, y, yaw0), (x, y, yaw_goal)], cfg, yaw0, yaw_goal, 0.0, 0.0, 0.0, 0.0)
        keep = douglas_peucker(pts, cfg.simplify_tol_m)
        pts = [pts[k] for k in keep]
        n = len(pts)
        heads = [math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]) for i in range(n - 1)]
        lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(n - 1)]
        # -- corners: straight / fillet / sharp
        min_turn, max_turn = math.radians(cfg.fillet_min_turn_deg), math.radians(cfg.fillet_max_turn_deg)
        turns = [wrap_angle(heads[i] - heads[i - 1]) for i in range(1, n - 1)]  # signed, per interior vertex
        kinds = ["straight" if abs(t) < min_turn else ("sharp" if abs(t) > max_turn else "fillet") for t in turns]
        tangents = [0.0] * len(turns)
        for c, t in enumerate(turns):
            if kinds[c] != "fillet":
                continue
            th_abs = abs(t)
            i = c + 1  # vertex index
            avail_in = lens[i - 1] * (0.5 if (c - 1 >= 0 and kinds[c - 1] == "fillet") else 1.0)
            avail_out = lens[i] * (0.5 if (c + 1 < len(kinds) and kinds[c + 1] == "fillet") else 1.0)
            tan_half = math.tan(th_abs / 2.0)
            r = min(cfg.fillet_deviation_m / (1.0 / math.cos(th_abs / 2.0) - 1.0), avail_in / tan_half, avail_out / tan_half)
            if r < cfg.fillet_min_radius_m:
                kinds[c] = "sharp"
            else:
                tangents[c] = r * tan_half
        # -- pieces
        pieces: List[object] = []
        dwells: List[Dwell] = []
        turn_windows: List[Tuple[float, float, float]] = []  # (arc midpoint, window, signed turn) per fillet
        jumps: List[Tuple[float, float]] = []  # (arc length, signed turn) per sharp corner
        s = 0.0
        cur = (pts[0][0], pts[0][1])

        def add_line(p_to: Tuple[float, float], heading: float):
            nonlocal s, cur
            L = math.hypot(p_to[0] - cur[0], p_to[1] - cur[1])
            if L > 1e-9:
                pieces.append(Line(s, s + L, cur, heading))
                s += L
            cur = p_to

        for c, t in enumerate(turns):
            i = c + 1
            v = (pts[i][0], pts[i][1])
            if kinds[c] == "fillet":
                tl = tangents[c]
                u_in = (math.cos(heads[i - 1]), math.sin(heads[i - 1]))
                u_out = (math.cos(heads[i]), math.sin(heads[i]))
                a = (v[0] - u_in[0] * tl, v[1] - u_in[1] * tl)
                b = (v[0] + u_out[0] * tl, v[1] + u_out[1] * tl)
                add_line(a, heads[i - 1])
                sign = 1.0 if t > 0 else -1.0
                r = tl / math.tan(abs(t) / 2.0)
                centre = (a[0] - sign * u_in[1] * r, a[1] + sign * u_in[0] * r)
                phi0 = math.atan2(a[1] - centre[1], a[0] - centre[0])
                arc_len = r * abs(t)
                pieces.append(Arc(s, s + arc_len, centre, r, phi0, sign, heads[i - 1]))
                turn_windows.append((s + 0.5 * arc_len, max(arc_len, BLEND_PEAK_SLOPE * abs(t) / max(cfg.corner_yaw_rate_rad_per_m, 1e-6)), t))
                s += arc_len
                cur = b
            else:
                add_line(v, heads[i - 1])
                if kinds[c] == "sharp":
                    dwells.append(Dwell(s, "corner", heads[i - 1], heads[i]))
                    jumps.append((s, t))
                else:
                    turn_windows.append((s, 0.0, t))  # a heading change below the corner threshold: a step of the yaw
        add_line((pts[-1][0], pts[-1][1]), heads[-1])
        length = s
        # a turn window never reaches past the ends of the drive
        turn_windows = [(sm, min(w, 2.0 * sm, 2.0 * (length - sm)), t) for sm, w, t in turn_windows]
        # -- yaw policy: hybrid end turns
        h_first, h_last = heads[0], heads[-1]
        yaw0 = h_first if math.isnan(yaw0) else yaw0
        yaw_goal = h_last if math.isnan(yaw_goal) else yaw_goal
        spot = math.radians(cfg.spot_turn_deg)
        d0 = wrap_angle(yaw0 - h_first)
        yaw_start = yaw0
        if abs(d0) > spot:
            dwells.insert(0, Dwell(0.0, "start", yaw0, h_first))
            yaw_start, d0 = h_first, 0.0
        d1 = wrap_angle(yaw_goal - h_last)
        if abs(d1) > spot:
            dwells.append(Dwell(length, "end", h_last, yaw_goal))
            d1 = 0.0
        both = abs(d0) > 1e-9 and abs(d1) > 1e-9
        b0 = min(cfg.blend_m, length / 2.0 if both else length) if abs(d0) > 1e-9 else 0.0
        b1 = min(cfg.blend_m, length / 2.0 if both else length) if abs(d1) > 1e-9 else 0.0
        filled = [(pts[0][0], pts[0][1], yaw0)] + [(pts[i][0], pts[i][1], heads[i]) for i in range(1, n - 1)] + [(pts[-1][0], pts[-1][1], yaw_goal)]
        return cls(pieces, dwells, filled, cfg, yaw_start, yaw_goal, d0, d1, b0, b1, heading0=h_first, turns=turn_windows, jumps=jumps)

    # ------------------------------------------------------------------ geometry

    @property
    def polyline_length(self) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.waypoints[:-1], self.waypoints[1:]))

    def _piece(self, s: float, side: int = 1):
        if not self.pieces:
            return None
        i = (bisect.bisect_right(self._s0, s) if side >= 0 else bisect.bisect_left(self._s0, s)) - 1
        return self.pieces[min(max(i, 0), len(self.pieces) - 1)]

    def position(self, s: float) -> Tuple[float, float]:
        p = self._piece(s)
        if p is None:
            return self.waypoints[0][0], self.waypoints[0][1]
        return p.position(s)

    def tangent(self, s: float, side: int = 1) -> float:
        """Heading of the path at ``s``; ``side`` picks the piece before (-1) or after (+1) a sharp corner at ``s``."""
        p = self._piece(s, side)
        return self.yaw_start if p is None else p.tangent(s)

    def curvature(self, s: float, side: int = 1) -> float:
        p = self._piece(s, side)
        return 0.0 if p is None else float(p.curvature)

    def _schedule(self, s: float, side: int = 1) -> Tuple[float, float, float]:
        """The body yaw at ``s`` and its first / second derivatives in ``s``: the first heading, plus every corner's
        turn blended over its window (a step for the sharp corners: ``side`` picks before / after at the corner itself),
        plus the end blends."""
        L = self.length
        yaw = self._h0
        d1 = d2 = 0.0
        for sm, w, t in self._turns:
            if w <= 1e-9:
                if s > sm or (s >= sm and side > 0):
                    yaw += t
                continue
            u = (s - (sm - 0.5 * w)) / w
            yaw += t * _smooth(u)
            d1 += t * _smooth_d(u) / w
            d2 += t * _smooth_dd(u) / (w * w)
        for sj, t in self._jumps:
            if s > sj + 1e-9 or (abs(s - sj) <= 1e-9 and side > 0):
                yaw += t
        if self._b0 > 0.0:
            u = s / self._b0
            yaw += self._delta0 * (1.0 - _smooth(u))
            d1 += -self._delta0 * _smooth_d(u) / self._b0
            d2 += -self._delta0 * _smooth_dd(u) / (self._b0 * self._b0)
        if self._b1 > 0.0:
            u = (s - (L - self._b1)) / self._b1
            yaw += self._delta1 * _smooth(u)
            d1 += self._delta1 * _smooth_d(u) / self._b1
            d2 += self._delta1 * _smooth_dd(u) / (self._b1 * self._b1)
        return yaw, d1, d2

    def yaw(self, s: float, side: int = 1) -> float:
        if not self.pieces:
            return self.yaw_goal
        return self._schedule(s, side)[0]

    def yaw_rate_ds(self, s: float, side: int = 1) -> float:
        """dψ/ds at ``s`` (rad / m)."""
        return self._schedule(s, side)[1]

    def yaw_acc_ds2(self, s: float) -> float:
        """d²ψ/ds² at ``s`` (rad / m²)."""
        return self._schedule(s)[2]

    def pose(self, s: float, side: int = 1) -> Pose2D:
        x, y = self.position(s)
        return x, y, self.yaw(s, side)

    def project(self, xy: Sequence[float], ds: float = 0.01) -> float:
        """Arc length of the point of the path nearest to ``xy`` (sampled every ``ds``)."""
        if not self.pieces:
            return 0.0
        x, y = float(xy[0]), float(xy[1])
        n = max(1, int(math.ceil(self.length / ds)))
        best, best_s = float("inf"), 0.0
        for k in range(n + 1):
            s = min(self.length, k * ds)
            px, py = self.position(s)
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best:
                best, best_s = d, s
        return best_s

    # ------------------------------------------------------------------ sampling

    def drive_grid(self, stride: float) -> List[float]:
        """Arc lengths every ``stride`` from 0, the end included."""
        if self.length <= 1e-9:
            return [0.0]
        n = max(1, int(math.ceil(self.length / stride - 1e-9)))
        out = [min(self.length, k * stride) for k in range(n)]
        out.append(self.length)
        return out

    def drive_poses(self, stride: float) -> Tuple[List[float], List[Pose2D]]:
        """The poses the robot drives through every ``stride`` (the dwell sweeps excluded; at ``s = 0`` the body already
        faces the way it departs)."""
        ss = self.drive_grid(stride)
        return ss, [self.pose(s) for s in ss]

    @staticmethod
    def sweep(yaw_from: float, yaw_to: float, step: float) -> List[float]:
        """Yaws from ``yaw_from`` (excluded) to ``yaw_to`` (included) the shortest way, at most ``step`` apart."""
        d = wrap_angle(yaw_to - yaw_from)
        n = max(1, int(math.ceil(abs(d) / max(step, 1e-6))))
        return [yaw_from + d * k / n for k in range(1, n + 1)]

    def _yaw_fill(self, s_a: float, side_a: int, s_b: float, side_b: int, step: float, depth: int = 0) -> List[float]:
        """Arc lengths strictly between ``s_a`` and ``s_b`` (bisection) at which the drive is sampled so that no two
        consecutive samples differ by more than ``step`` in yaw (no sharp corner lies strictly between them)."""
        sm = 0.5 * (s_a + s_b)
        ya, ym, yb = self.yaw(s_a, side_a), self.yaw(sm), self.yaw(s_b, side_b)
        if depth >= 16 or s_b - s_a <= 1e-6 or max(abs(ym - ya), abs(yb - ym), abs(yb - ya)) <= step:
            return []
        return self._yaw_fill(s_a, side_a, sm, 1, step, depth + 1) + [sm] + self._yaw_fill(sm, 1, s_b, side_b, step, depth + 1)

    def stride_poses(self, stride: float, dwell_step_rad: float = DWELL_CHECK_STEP_RAD, max_yaw_step: Optional[float] = None) -> List[Pose2D]:
        """Every pose the whole-body check must see: the departing pose, the start dwell's sweep, the drive every
        ``stride`` (the sharp corners' sweeps where they occur), the end dwell's sweep, the goal pose. ``max_yaw_step``
        (rad; the sampler's ``navigation.YAW_STEP_MAX_RAD``): no two consecutive poses differ by more in yaw - the drive
        is sampled between two strides where the schedule turns, a sharp corner from its arrival yaw, the dwells at the
        smaller of the two steps."""
        bound = max_yaw_step is not None and max_yaw_step > 0.0
        if bound:
            dwell_step_rad = min(dwell_step_rad, float(max_yaw_step))
        x0, y0 = self.position(0.0)
        out: List[Pose2D] = [(x0, y0, self.waypoints[0][2])]
        corners = sorted((d for d in self.dwells if d.kind == "corner"), key=lambda d: d.s)
        for d in self.dwells:
            if d.kind == "start":
                out.extend((x0, y0, yaw) for yaw in self.sweep(d.yaw_from, d.yaw_to, dwell_step_rad))
        if self.pieces:
            k = 0
            last: Optional[Tuple[float, int]] = None  # (arc length, side) of the last drive sample
            for s in self.drive_grid(stride):
                while k < len(corners) and corners[k].s <= s + 1e-9:
                    sc = corners[k].s
                    cx, cy = self.position(sc)
                    if bound and last is not None:
                        out.extend(self.pose(u) for u in self._yaw_fill(last[0], last[1], sc, -1, float(max_yaw_step)))
                        out.append((cx, cy, self.yaw(sc, -1)))
                    out.extend((cx, cy, yaw) for yaw in self.sweep(self.yaw(sc, -1), self.yaw(sc, 1), dwell_step_rad))
                    last = (sc, 1)
                    k += 1
                if bound and last is not None:
                    out.extend(self.pose(u) for u in self._yaw_fill(last[0], last[1], s, 1, float(max_yaw_step)))
                out.append(self.pose(s))
                last = (s, 1)
        xe, ye = self.position(self.length)
        for d in self.dwells:
            if d.kind == "end":
                out.extend((xe, ye, yaw) for yaw in self.sweep(d.yaw_from, d.yaw_to, dwell_step_rad))
        out.append((xe, ye, self.yaw_goal))
        return out


# ----------------------------------------------------------------------------------------------------------------
# time parameterisation (the oracle's tracker reference)
# ----------------------------------------------------------------------------------------------------------------


def dwell_yaws(yaw_from: float, yaw_to: float, dt: float, w_max: float, a_yaw: float) -> List[float]:
    """Yaw samples every ``dt`` of an on-the-spot turn from ``yaw_from`` to ``yaw_to`` (shortest way): a trapezoidal
    rate profile at ``w_max`` with ramps at ``a_yaw`` (triangular when the turn is too short to reach ``w_max``)."""
    d = wrap_angle(yaw_to - yaw_from)
    D = abs(d)
    if D < 1e-9:
        return []
    w_max, a_yaw = max(float(w_max), 1e-6), max(float(a_yaw), 1e-6)
    t_acc = w_max / a_yaw
    if a_yaw * t_acc * t_acc >= D:  # triangular
        t_acc = math.sqrt(D / a_yaw)
        T, w_peak = 2.0 * t_acc, a_yaw * t_acc
    else:
        T, w_peak = 2.0 * t_acc + (D - a_yaw * t_acc * t_acc) / w_max, w_max

    def angle(t: float) -> float:
        if t <= t_acc:
            return 0.5 * a_yaw * t * t
        if t >= T - t_acc:
            tr = T - t
            return D - 0.5 * a_yaw * tr * tr
        return 0.5 * a_yaw * t_acc * t_acc + w_peak * (t - t_acc)

    n = max(1, int(math.ceil(T / dt)))
    sign = 1.0 if d > 0 else -1.0
    return [yaw_from + sign * angle(T * m / n) for m in range(1, n + 1)]


def time_parameterise(path: BasePath, dt: float, v_lin: float, v_ang: float, a_lin: float, a_yaw: float, ds: float = 0.005,
                      v_start: float = 0.05) -> th.Tensor:
    """``(T, 3)`` ``(x, y, yaw)`` reference sampled every ``dt`` along ``path`` (see the module docstring for the speed
    limits). The yaw is continuous (unwrapped) so the tracker's finite differences are meaningful."""
    dt = float(dt)
    v_lin, v_ang, a_lin, a_yaw = float(v_lin), float(v_ang), float(a_lin), float(a_yaw)
    samples: List[List[float]] = []
    x0, y0 = path.position(0.0)
    for d in path.dwells:
        if d.kind == "start":
            samples.extend([x0, y0, yaw] for yaw in dwell_yaws(d.yaw_from, d.yaw_to, dt, v_ang, a_yaw))
    L = path.length
    corners = sorted((d for d in path.dwells if d.kind == "corner"), key=lambda d: d.s)
    if L > 1e-9:
        n_grid = max(1, int(math.ceil(L / ds)))
        grid = [min(L, k * L / n_grid) for k in range(n_grid + 1)]
        v = []
        for s in grid:
            cap = v_lin
            kappa = abs(path.curvature(s))
            if kappa > 1e-9:
                cap = min(cap, math.sqrt(a_lin / kappa))
            dpsi = abs(path.yaw_rate_ds(s))
            if dpsi > 1e-9:
                cap = min(cap, v_ang / dpsi)
            ddpsi = abs(path.yaw_acc_ds2(s))
            if ddpsi > 1e-9:
                cap = min(cap, math.sqrt(a_yaw / ddpsi))
            v.append(max(cap, 1e-3))
        v[0] = min(v[0], v_start)
        v[-1] = 0.0
        for c in corners:
            j = min(range(len(grid)), key=lambda k: abs(grid[k] - c.s))
            v[j] = 0.0
        for j in range(1, len(grid)):
            v[j] = min(v[j], math.sqrt(v[j - 1] ** 2 + 2.0 * a_lin * (grid[j] - grid[j - 1])))
        for j in range(len(grid) - 2, -1, -1):
            v[j] = min(v[j], math.sqrt(v[j + 1] ** 2 + 2.0 * a_lin * (grid[j + 1] - grid[j])))
        t_grid = [0.0]
        for j in range(1, len(grid)):
            vm = max(0.5 * (v[j] + v[j - 1]), 0.02)
            t_grid.append(t_grid[-1] + (grid[j] - grid[j - 1]) / vm)
        total_t = t_grid[-1]
        n = max(1, int(math.ceil(total_t / dt)))
        pending = list(corners)
        for i in range(1, n + 1):
            ti = min(total_t, i * dt)
            jj = min(max(bisect.bisect_right(t_grid, ti) - 1, 0), len(grid) - 2)
            f = 0.0 if t_grid[jj + 1] - t_grid[jj] < 1e-12 else (ti - t_grid[jj]) / (t_grid[jj + 1] - t_grid[jj])
            s = grid[jj] + (grid[jj + 1] - grid[jj]) * f
            while pending and pending[0].s <= s + 1e-9:
                cx, cy = path.position(pending[0].s)
                samples.extend([cx, cy, yaw] for yaw in dwell_yaws(path.yaw(pending[0].s, -1), path.yaw(pending[0].s, 1), dt, v_ang, a_yaw))
                pending.pop(0)
            x, y, yaw = path.pose(s)
            samples.append([x, y, yaw])
        for c in pending:
            cx, cy = path.position(c.s)
            samples.extend([cx, cy, yaw] for yaw in dwell_yaws(path.yaw(c.s, -1), path.yaw(c.s, 1), dt, v_ang, a_yaw))
    xe, ye = path.position(L)
    for d in path.dwells:
        if d.kind == "end":
            samples.extend([xe, ye, yaw] for yaw in dwell_yaws(d.yaw_from, d.yaw_to, dt, v_ang, a_yaw))
    samples.append([xe, ye, path.yaw_goal])
    ref = th.tensor(samples, dtype=th.float64).reshape(-1, 3)
    # unwrap the yaw column: consecutive samples differ by the wrapped difference
    for i in range(1, len(ref)):
        ref[i, 2] = ref[i - 1, 2] + wrap_angle(float(ref[i, 2] - ref[i - 1, 2]))
    return ref.to(th.float32)


# ----------------------------------------------------------------------------------------------------------------
# axis-aligned legs (owner 2026-10-13, disembedding: "moving to the place pose, the robot moves only forward / backward or
# left / right, keeping its heading; it can combine the two to reach the place pose")
# ----------------------------------------------------------------------------------------------------------------

#: a recorded axis-aligned leg: every waypoint's yaw within this of the first waypoint's (rad)
AXIS_LEG_YAW_TOL_RAD = math.radians(0.5)
#: ... and every segment within this of the heading's forward / backward / left / right axis (rad)
AXIS_LEG_DIR_TOL_RAD = math.radians(1.0)
#: segments up to this long carry no direction (m)
AXIS_LEG_MIN_SEG_M = 0.001
_AXIS_NAMES = {0: "forward", 1: "left", 2: "backward", -2: "backward", -1: "right"}


def _axis_of(dx: float, dy: float, yaw: float) -> Tuple[str, float, float]:
    """(robot axis name, angle of the motion (dx, dy) in the frame of heading ``yaw``, its offset from that axis); rad."""
    rel = wrap_angle(math.atan2(dy, dx) - yaw)
    k = int(round(rel / (math.pi / 2.0)))
    return _AXIS_NAMES[k], rel, abs(wrap_angle(rel - k * math.pi / 2.0))


def axis_leg_violation(waypoints: Sequence[Sequence[float]], yaw_tol: float = AXIS_LEG_YAW_TOL_RAD, dir_tol: float = AXIS_LEG_DIR_TOL_RAD,
                       min_seg_m: float = AXIS_LEG_MIN_SEG_M) -> Optional[str]:
    """Why a recorded ``[x, y, yaw]`` polyline is not an axis-aligned leg, None when it is: every waypoint's yaw within
    ``yaw_tol`` of the first one's (the heading is kept) and every segment longer than ``min_seg_m`` within ``dir_tol`` of
    that heading's forward / backward / left / right axis (moves one after the other, no diagonal)."""
    pts = [tuple(float(v) for v in w) for w in waypoints]
    if not pts:
        return None
    yaw0 = pts[0][2]
    for i, p in enumerate(pts[1:], 1):
        d = abs(wrap_angle(p[2] - yaw0))
        if d > yaw_tol:
            return f"waypoint {i} turns the heading by {math.degrees(d):.2f} deg (> {math.degrees(yaw_tol):.2f})"
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L <= min_seg_m:
            continue
        _, rel, off = _axis_of(b[0] - a[0], b[1] - a[1], yaw0)
        if off > dir_tol:
            return (f"segment {i} ({L:.3f} m) runs {math.degrees(rel):+.1f} deg from the heading, {math.degrees(off):.1f} deg off the "
                    f"nearest robot axis (> {math.degrees(dir_tol):.1f})")
    return None


def axis_leg_moves(waypoints: Sequence[Sequence[float]], min_seg_m: float = AXIS_LEG_MIN_SEG_M) -> List[Tuple[str, float]]:
    """The moves of a polyline in the robot frame of its first waypoint: ``(forward | backward | left | right, metres)`` per
    segment longer than ``min_seg_m``, consecutive moves along one axis merged."""
    pts = [tuple(float(v) for v in w) for w in waypoints]
    out: List[Tuple[str, float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L <= min_seg_m:
            continue
        name = _axis_of(b[0] - a[0], b[1] - a[1], pts[0][2])[0]
        if out and out[-1][0] == name:
            out[-1] = (name, out[-1][1] + L)
        else:
            out.append((name, L))
    return out


def _point_polyline_dist(x: float, y: float, pts: Sequence[Tuple[float, ...]]) -> float:
    if len(pts) == 1:
        return math.hypot(x - pts[0][0], y - pts[0][1])
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, ((x - a[0]) * dx + (y - a[1]) * dy) / L2))
        best = min(best, math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy)))
    return best


def axis_leg_tracking(poses: Sequence[Sequence[float]], waypoints: Sequence[Sequence[float]]) -> Dict[str, float]:
    """How a driven base trajectory (``poses``, ``[x, y, yaw]`` per step) kept to the axis-aligned leg ``waypoints``:
    ``max_yaw_dev_rad`` = the largest heading difference from the leg's (first waypoint's) yaw, ``max_corridor_m`` = the
    largest distance from the polyline."""
    pts = [tuple(float(v) for v in w) for w in waypoints]
    if not pts or not poses:
        return {"max_yaw_dev_rad": 0.0, "max_corridor_m": 0.0, "n_poses": len(poses)}
    yaw0 = pts[0][2]
    dyaw = max(abs(wrap_angle(float(p[2]) - yaw0)) for p in poses)
    corridor = max(_point_polyline_dist(float(p[0]), float(p[1]), pts) for p in poses)
    return {"max_yaw_dev_rad": dyaw, "max_corridor_m": corridor, "n_poses": len(poses)}
