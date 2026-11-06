"""IORS inside, large-scale (spec §5.6 "Inside (large-scale)"; owner 2026-10-12 ~16:05 item 1: inside_large_scale#03 plate -> drop-in sink,
#02 jar -> fridge shelf): :class:`InsideMode`, the mode ``IORSLarge._sample_inside`` returns (``iors_large._adopt_mode`` puts it in front of
IORSLarge; stances A / B, the B -> C coupling, the carry and the records stay IORSLarge's).

Neither partner has a fillable meta link, so the goal surface is the partner's **interior surface**: ``SimWorld.support_interior`` at
``generation.inside_surface_z_m`` (m above the floor) with ``inside_surface_tol_m`` as its height tolerance - a sink's basin floor (the
downward rays start just under the rim; a generous tolerance finds the floor whatever the basin's depth) or a fridge shelf (the code of the
bookcase compartments). The same two numbers are the goal-side band's ``--surface-z`` / ``--surface-tol``: ``ReachBands.load(...,
surface_z=...)`` matches the requested height. Goal cells: the free board cells with headroom for the target (+ 2 cm), at least the target's
half extent + ``opening_margin_m`` from the board's outline (the opening shrunk; ``inside_erosion_m`` replaces that distance - a plate as wide
as a basin's flat floor rests with its rim over the flaring walls).

Stance C faces the opening: base square to a side of the interior raster (``inside_stance_side``: the world direction the opening faces, e.g.
``"+y"``; default every side, pooled nearest first), stand-offs ``inside_stand_offs_m`` (default :data:`STAND_OFFS`) + the partner's AABB overhang,
laterals along the side, trunks from the goal-side band (without one: the :data:`NO_BAND_TRUNKS` trunks per base pose, among ``inside_trunks``
when set, with the most reach margin to the goal cell); the goal cell = the band-admitted goal cell nearest the base; the target at the goal in frame
(5 %). The collision check tries the arm postures ``inside_stance_postures`` in order (:func:`free_in_postures`; default ``untucked``: the
working arm untucked, the idle arm's A6 retry; ``ready`` / ``tucked``: both arms, as the goal-side band's ``trunk_config`` checks) and a
candidate keeps the first that is free.
Release (the small-scale drop): the target's bottom ``drop_height_m`` above the surface, raised until the open gripper's lowest point
clears the surface by ``release_clearance_m`` for every library grasp; the partner stays in the collision world (the oracle's drop keeps its
container). Goal check: teleport to the release, settle (the drop test), ``inside`` (the BDDL goal; ``predicates.inside_rays`` decides for a
container without a fillable meta link) and the target's top at most ``max_protrusion_fraction`` of its height above the rim. The oracle
program places with kind ``drop`` (``programs.program_iors_large_inside``).

Release tilt (owner 2026-10-12 ~22:35, inside_large#03: the plate's one grasp pinches its rim from outside, so a level release low in the
basin leaves the hand no room beside the plate): ``generation.release_tilt_deg`` (e.g. [30, 45]; default none = the level release only)
screens each tilt as an alternative to the level release - the target at yaw x tilt x rest, the tilt about the horizontal axis perpendicular
to the grasp's radial direction (the grasp side up, the far rim down), the release height from the tilted hull's lowest point. The draw tries
a pair's tilts in order (level first), the records carry ``release_tilt_deg`` and the goal check's up-axis rule measures a tilted release
against the level orientation. The oracle's drop needs nothing more: the carry leg to the recorded pre-place SLERPs the hand's orientation.

Per-grasp release height (owner 2026-10-13 ~00:50 item 3, inside_large#02: the jar into the fridge's lower shelf, ~0.23 m headroom):
``generation.release_lift_per_grasp`` (default false = one height for every grasp, the highest lift over the library) raises the release
only as far as the drawn grasp's own open gripper needs (its lowest point ``release_clearance_m`` above the surface at the release
orientation, level or tilted): the F_goal screen poses each library row at its own release (``ik.feasible_sets(..., pose_per_grasp=True)``),
the C exact IK, the goal / settle (the drop test's release) use the drawn grasp's, and ``summary.release`` carries its ``release_above_m``
with ``release_above_max_m`` (the default's height) beside it.

Goal region (owner 2026-10-13 ~03:15 item 2, inside_large#02: the fridge's 0.97 m shelf spans both compartments, the right one behind its
closed door): ``generation.inside_goal_region`` ``{"frame": "world" | "partner", "x": [lo, hi], "y": [lo, hi]}`` (m; default none = every
goal cell) keeps the goal cells (after the erosion) inside an xy box - in the world frame, or the partner's root frame (its position, turned
by its yaw); an axis left out is unbounded. One INFO line ``inside goal region ...: N -> M goal cells``; a box that keeps none of the cells
is ``Reject("spec")``.

The shelf place of a standing book (oracle-fix 2026-11-06, owner feedback on inside_large#01, hardback into bookcase_jaysra_0: "I want the
book to be placed inside the compartment like the init poses of hardback touching hardback"), three opt-ins on the rolled release
(``generation.release_roll_deg``): ``release_roll_axis: "edge_normal"`` rolls about the pinched edge's outward normal (the horizontal object
axis nearest the grasp's radial direction) instead of the radial direction itself, so every library row's rolled release is exactly upright;
``inside_spine_out`` (needs ``inside_free_yaw``; ``inside_spine_out_tol_deg``, default 0) screens a rolled release only at the yaw(s) whose
spine - that edge - faces out of the opening (-``edge.inward``, toward the robot); ``inside_shelf_place`` (needs ``inside_spine_out``)
replaces the goal cell by the touching#03 shelf pose (``tabletop.shelf_lean_pose``): the spine ``spine_behind_edge_m`` behind the board's
front edge, the upright book's near cover ``wall_gap_m`` from the side board of the goal cell's bay on ``side`` (``"near"``: the working
arm's side, else ``"left"`` / ``"right"`` facing the opening), the release leaning ``release_lean_deg`` toward that board about the edge's
inward axis (past the book's tipping angle: the goal settle tips it onto the board) and moved away from the board, per (orientation, library
row), until the fully open jaw (``gripper_corners_world``, ``C.GRIPPER_OPEN_M``) and its retreat (``jaw_sweep_m`` back along -a, the
oracle place's ``RETREAT_DIST``) keep ``jaw_clear_m`` from the board while behind the front edge (:func:`shelf_release_xy`); ``roll_fits``
checks the footprint there. The facts go to ``summary.release.shelf_place``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import object_joints as OJ
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks.iors_small import gripper_corners_world

log = logging.getLogger("ispatialgym.sampling.v2")

#: stand-offs of stance C from the interior raster's side (+ the partner's AABB overhang); one more than ontop's: the raster side of a basin
#: lies 10-20 cm behind the counter's front edge
STAND_OFFS = (0.35, 0.45, 0.55, 0.65)
SURFACE_TOL_M = 0.03
HEADROOM_MARGIN_M = TT.INTERIOR_HEADROOM_MARGIN_M
OPENING_MARGIN_M = 0.02
DROP_HEIGHT_M = 0.005
RELEASE_CLEARANCE_M = 0.02
MAX_PROTRUSION_FRACTION = 0.5
#: without a goal-side band: the trunks per base pose kept for the coupling, by reach margin to the goal cell (shoulder -> cell), chosen among
#: the collision-free ones with the goal in frame (the most margin = the deepest lean, which the counter run in front of a basin blocks)
NO_BAND_TRUNKS = 3
SIDE_MAX_DEG = 45.0
#: ``generation.inside_stance_postures``: the arm postures stance C's collision check tries in order (the goal-side band's ``trunk_config``
#: tries untucked -> ready -> tucked, both arms in one posture)
STANCE_POSTURES = ("untucked",)
POSTURE_NAMES = ("untucked", "ready", "tucked")


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def raster_rim(mask: th.Tensor, xs: th.Tensor, ys: th.Tensor) -> th.Tensor:
    """World xy (N, 2) of the raster cells of ``mask`` with a missing 4-neighbour (the board's outline)."""
    m = mask.bool()
    pad = th.nn.functional.pad(m.float()[None, None], (1, 1, 1, 1))[0, 0] > 0.5
    interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    idx = th.nonzero(m & ~interior)
    return th.stack([xs[idx[:, 0]], ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)


def erode_from(cells: th.Tensor, rim: th.Tensor, margin: float) -> th.Tensor:
    """The cells (N, 2) farther than ``margin`` from every rim point."""
    if len(cells) == 0 or len(rim) == 0:
        return cells
    return cells[th.cdist(cells.float(), rim.float()).min(dim=1).values > float(margin)]


def goal_region(value: Any) -> Optional[Dict[str, Any]]:
    """``generation.inside_goal_region``: ``{"frame": "world" | "partner", "x": [lo, hi], "y": [lo, hi]}`` (m; ``frame`` default
    ``"world"``; an axis left out or None is unbounded, at least one bounded) -> ``{"frame", "x", "y"}``; None -> None (every goal cell)."""
    if value is None:
        return None
    form = '{"frame": "world" | "partner", "x": [lo, hi], "y": [lo, hi]}'
    if not isinstance(value, dict) or set(value) - {"frame", "x", "y"}:
        raise ValueError(f"inside_goal_region {value!r}: expected {form}")
    frame = str(value.get("frame", "world")).strip().lower()
    if frame not in ("world", "partner"):
        raise ValueError(f"inside_goal_region {value!r}: frame must be world or partner")
    out: Dict[str, Any] = {"frame": frame, "x": None, "y": None}
    for ax in ("x", "y"):
        b = value.get(ax)
        if b is None:
            continue
        try:
            lo, hi = float(b[0]), float(b[1])
            ok = isinstance(b, (list, tuple)) and len(b) == 2 and lo <= hi
        except (TypeError, ValueError, IndexError, KeyError):
            ok = False
        if not ok:
            raise ValueError(f"inside_goal_region {value!r}: {ax} must be [lo, hi] (m) with lo <= hi")
        out[ax] = [lo, hi]
    if out["x"] is None and out["y"] is None:
        raise ValueError(f"inside_goal_region {value!r}: bound x or y ({form})")
    return out


def in_goal_region(cells: th.Tensor, region: Dict[str, Any], origin=(0.0, 0.0), yaw: float = 0.0) -> th.Tensor:
    """Which world cells (N, 2) lie in ``region`` (:func:`goal_region`), its x / y taken in the frame at world ``origin`` turned by ``yaw``
    (the defaults: the world frame) -> bool (N,)."""
    d = G.t(cells).float().reshape(-1, 2) - th.tensor([float(origin[0]), float(origin[1])])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    local = th.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=1)
    keep = th.ones(len(local), dtype=th.bool)
    for i, ax in enumerate(("x", "y")):
        if region.get(ax) is not None:
            keep &= (local[:, i] >= region[ax][0] - 1e-6) & (local[:, i] <= region[ax][1] + 1e-6)
    return keep


def cell_span(cells: th.Tensor) -> str:
    """``"x lo..hi y lo..hi"`` of the cells (N, 2), ``"none"`` without any."""
    if len(cells) == 0:
        return "none"
    lo, hi = cells.min(dim=0).values, cells.max(dim=0).values
    return f"x {float(lo[0]):.3f}..{float(hi[0]):.3f} y {float(lo[1]):.3f}..{float(hi[1]):.3f}"


def side_vector(side: Any) -> Optional[th.Tensor]:
    """``generation.inside_stance_side``: ``"+x" / "-x" / "+y" / "-y"`` or ``[dx, dy]`` -> unit world xy; None -> None."""
    if side is None:
        return None
    if isinstance(side, str):
        s = side.strip().lower()
        table = {"+x": (1.0, 0.0), "x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "y": (0.0, 1.0), "-y": (0.0, -1.0)}
        if s not in table:
            raise ValueError(f"inside_stance_side {side!r}: expected +x / -x / +y / -y or [dx, dy]")
        v = th.tensor(table[s])
    else:
        v = th.tensor([float(side[0]), float(side[1])])
    n = float(th.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"inside_stance_side {side!r}: zero vector")
    return v / n


def stance_postures(value: Any) -> List[str]:
    """``generation.inside_stance_postures``: a posture name or a list of them (:data:`POSTURE_NAMES`) -> the names in order, repeats
    dropped; None -> :data:`STANCE_POSTURES`."""
    if value is None:
        return list(STANCE_POSTURES)
    names = [value] if isinstance(value, str) else list(value)
    bad = [p for p in names if p not in POSTURE_NAMES]
    if bad or not names:
        raise ValueError(f"inside_stance_postures {value!r}: expected a non-empty list of {list(POSTURE_NAMES)}")
    return list(dict.fromkeys(names))


def edges_facing(edges: Sequence[EdgeFrame], side: Optional[th.Tensor], max_deg: float = SIDE_MAX_DEG) -> List[EdgeFrame]:
    """The edges whose outward normal (-inward) lies within ``max_deg`` of the world direction ``side``, best aligned first; every
    edge when ``side`` is None."""
    if side is None:
        return list(edges)
    cos_min = math.cos(math.radians(max_deg)) - 1e-6
    scored = [(float(th.dot(-e.inward.float(), side.float())), e) for e in edges]
    return [e for c, e in sorted(scored, key=lambda t: -t[0]) if c >= cos_min]


def overhang(mid, outward, lo, hi) -> float:
    """AABB overhang beyond a raster side along its outward normal (>= 5 cm counts, else 0; ``iors_large.edge_overhang``'s rule)."""
    lo, hi = G.t(lo).float(), G.t(hi).float()
    corners = G.aabb_corners(lo, hi)[:, :2]
    n = G.t(outward).float()[:2]
    over = float((corners @ n).max()) - float(th.dot(G.t(mid).float()[:2], n))
    return over if over >= 0.05 else 0.0


def release_lift(finger_lows: Sequence[float], surface_z: float, clearance: float) -> float:
    """How far the release must rise so that the lowest open-gripper point of every grasp clears the surface by ``clearance``."""
    if not finger_lows:
        return 0.0
    return max(0.0, float(surface_z) + float(clearance) - min(float(v) for v in finger_lows))


def lift_per_grasp(value: Any) -> bool:
    """``generation.release_lift_per_grasp``: true / "on" -> the release height from the drawn grasp's own open gripper; None / false /
    "off" -> one height for every library grasp (the highest lift)."""
    if value is None or isinstance(value, bool):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("on", "true", "1", "yes"):
        return True
    if s in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"release_lift_per_grasp {value!r}: expected true / false")


def rim_height(top_z: Optional[float], surface_z: float, height: float, aabb_top: float) -> Tuple[float, str]:
    """The rim the protrusion is measured against: the partner's support top (its most populated top height) when it lies at least the
    target's height above the interior surface (a sink's deck, a fridge's top), else the partner's AABB top (permissive)."""
    if top_z is not None and float(top_z) >= float(surface_z) + float(height):
        return float(top_z), "support_top"
    return float(aabb_top), "aabb_top"


def protrusion_ok(target_top: float, rim_z: float, height: float, fraction: float) -> Tuple[bool, float]:
    """§6.11 inside drop test: the target's top at most ``fraction`` of its height above the rim; (ok, protrusion m)."""
    prot = float(target_top) - float(rim_z)
    return prot <= float(fraction) * float(height), prot


def nearest_cell(pool: th.Tensor, xy) -> th.Tensor:
    p = G.t(xy).float()[:2]
    return pool[int(th.argmin(th.linalg.norm(pool.float() - p, dim=1)))]


def best_per_base(stances: Sequence[Stance], k: int) -> List[Stance]:
    """Per base pose, the ``k`` stances with the most ``extras["reach_margin_m"]`` (bases in their first order, ties in input order)."""
    groups: Dict[Tuple[float, ...], List[Stance]] = {}
    for st in stances:
        groups.setdefault(tuple(round(float(v), 4) for v in st.base_pose2d), []).append(st)
    return [st for g in groups.values() for st in sorted(g, key=lambda s_: -float(s_.extras["reach_margin_m"]))[:k]]


def release_tilts(value: Any) -> List[float]:
    """``generation.release_tilt_deg``: a tilt (deg) or a list of them -> the tilts screened besides the level release, in order, 0 and
    repeats dropped (the level release is always screened); None / [] -> [] (the level release only)."""
    if value is None:
        return []
    out: List[float] = []
    for v in [value] if isinstance(value, (int, float)) else list(value):
        try:
            d = float(v)
        except (TypeError, ValueError):
            d = float("nan")
        if not 0.0 <= d < 90.0:
            raise ValueError(f"release_tilt_deg {value!r}: expected tilts in degrees, 0 <= tilt < 90")
        if d > 0.0 and d not in out:
            out.append(d)
    return out


def tilt_axis(eef_in_object_pos, level_quat) -> Optional[th.Tensor]:
    """The world axis of the release tilt for a grasp whose eef sits at ``eef_in_object_pos`` in the object frame, the object at the level
    orientation ``level_quat``: horizontal, perpendicular to the grasp's radial direction (the grasp point's horizontal direction from the
    object origin), signed so that a positive turn raises the grasp side (radial x z). None when the grasp point lies on the vertical
    through the origin."""
    p = G.rotate(G.t(level_quat).float(), G.t(eef_in_object_pos).float())
    n = math.hypot(float(p[0]), float(p[1]))
    if n < 1e-6:
        return None
    return th.tensor([float(p[1]) / n, -float(p[0]) / n, 0.0])


def tilted_quat(level_quat, eef_in_object_pos, deg: float):
    """The release orientation tilted by ``deg``: ``level_quat`` (yaw * rest) turned about :func:`tilt_axis` - yaw * tilt * rest, the tilt
    about the rested frame's horizontal axis perpendicular to the grasp's radial direction -, the grasp side up and the far rim down;
    ``level_quat`` itself at 0 deg or without a radial direction."""
    axis = tilt_axis(eef_in_object_pos, level_quat) if deg else None
    if axis is None:
        return level_quat
    return G.qmul(G.quat_about_axis(axis, math.radians(float(deg))), G.t(level_quat).float())


#: a release roll (oracle-fix 2026-11-04 round 3, inside_large#01) travels through the tilt bookkeeping as ``ROLL_KEY + roll deg``
ROLL_KEY = 1000.0


def release_rolls(value: Any) -> List[float]:
    """``generation.release_roll_deg``: a roll (deg, -180 < roll <= 180, not 0) or a list -> the rolls screened, in order; None / [] -> []."""
    if value is None:
        return []
    out: List[float] = []
    for v in [value] if isinstance(value, (int, float)) else list(value):
        try:
            d = float(v)
        except (TypeError, ValueError):
            d = float("nan")
        if not -180.0 < d <= 180.0 or d == 0.0:
            raise ValueError(f"release_roll_deg {value!r}: expected rolls in degrees, -180 < roll <= 180, not 0")
        if d not in out:
            out.append(d)
    return out


def _rolls_of(s) -> List[float]:
    """The variant's ``generation.release_roll_deg`` (:func:`release_rolls`); ``generation.release_level`` false (default true) drops the
    level release when rolls are screened (a hardback must stand, not lie, in the compartment - oracle-fix 2026-11-04 round 3 item 2c)."""
    try:
        return release_rolls(s.gp.gen("release_roll_deg"))
    except ValueError as e:
        raise Reject("spec", f"{s.gp.variant}: {e}")


def is_roll(key: float) -> bool:
    return float(key) >= 0.5 * ROLL_KEY


#: ``generation.release_roll_axis`` (oracle-fix 2026-11-06, owner feedback on inside_large#01): the axis a release roll turns about -
#: ``"radial"`` (default) the grasp point's horizontal radial direction, ``"edge_normal"`` the horizontal object axis nearest it (the pinched
#: edge's outward normal; hardback__bqvvvx__bo grasp_1 / 2 pinch at x +-0.0348 m: the radial roll left them 21.6 deg off upright)
ROLL_AXES = ("radial", "edge_normal")
#: an object axis counts as horizontal at the level orientation when its world z is at most this (sin 5.7 deg)
HORIZONTAL_AXIS_TOL = 0.1


def roll_axis_mode(value: Any) -> str:
    """``generation.release_roll_axis``: :data:`ROLL_AXES`; None -> ``"radial"``."""
    if value is None:
        return "radial"
    s = str(value).strip().lower()
    if s not in ROLL_AXES:
        raise ValueError(f"release_roll_axis {value!r}: expected one of {list(ROLL_AXES)}")
    return s


def pinched_edge_normal(eef_in_object_pos, level_quat) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """The outward normal of the edge a grasp pinches: of the signed object axes horizontal at ``level_quat`` (world z within
    :data:`HORIZONTAL_AXIS_TOL`), the one nearest the grasp point's horizontal radial direction -> (that axis in the object frame, its world
    direction made horizontal); None without a radial direction or a horizontal object axis."""
    q = G.t(level_quat).float()
    p = G.rotate(q, G.t(eef_in_object_pos).float())
    n = math.hypot(float(p[0]), float(p[1]))
    if n < 1e-6:
        return None
    best: Optional[Tuple[float, th.Tensor, th.Tensor]] = None
    for i in range(3):
        for sgn in (1.0, -1.0):
            e = th.zeros(3)
            e[i] = sgn
            w = G.rotate(q, e)
            h = math.hypot(float(w[0]), float(w[1]))
            if abs(float(w[2])) > HORIZONTAL_AXIS_TOL or h < 1e-6:
                continue
            c = (float(w[0]) * float(p[0]) + float(w[1]) * float(p[1])) / (h * n)
            if best is None or c > best[0] + 1e-9:
                best = (c, e, th.tensor([float(w[0]) / h, float(w[1]) / h, 0.0]))
    return None if best is None else (best[1], best[2])


def roll_axis(eef_in_object_pos, level_quat, mode: str = "radial") -> Optional[th.Tensor]:
    """The world axis of a release roll: the grasp's horizontal radial direction (the grasp point's direction from the object origin at the
    level orientation) - a roll turns the object about it, the grasped edge staying where it is (a flat book pinched at its spine stands up,
    the spine toward the hand). None when the grasp point lies on the vertical through the origin. ``mode`` ``"edge_normal"`` (oracle-fix
    2026-11-06): the pinched edge's outward normal instead (:func:`pinched_edge_normal`) - an off-centre pinch's release stands upright too."""
    if mode == "edge_normal":
        found = pinched_edge_normal(eef_in_object_pos, level_quat)
        return None if found is None else found[1]
    p = G.rotate(G.t(level_quat).float(), G.t(eef_in_object_pos).float())
    n = math.hypot(float(p[0]), float(p[1]))
    if n < 1e-6:
        return None
    return th.tensor([float(p[0]) / n, float(p[1]) / n, 0.0])


def rolled_quat(level_quat, eef_in_object_pos, deg: float, mode: str = "radial"):
    """The release orientation rolled by ``deg`` about :func:`roll_axis` (``mode``); ``level_quat`` itself without a radial direction."""
    axis = roll_axis(eef_in_object_pos, level_quat, mode) if deg else None
    if axis is None:
        return level_quat
    return G.qmul(G.quat_about_axis(axis, math.radians(float(deg))), G.t(level_quat).float())


# ---------------------------------------------------------------- the spine-out yaw and the shelf place (oracle-fix 2026-11-06, inside_large#01)


def spine_out_tol(value: Any) -> float:
    """``generation.inside_spine_out_tol_deg`` (deg, 0 <= tol < 180; None -> 0) -> rad."""
    try:
        d = 0.0 if value is None else float(value)
    except (TypeError, ValueError):
        d = float("nan")
    if not 0.0 <= d < 180.0:
        raise ValueError(f"inside_spine_out_tol_deg {value!r}: expected degrees, 0 <= tol < 180")
    return math.radians(d)


def spine_out_yaw(q_at_yaw0, axis_obj, outward) -> Optional[float]:
    """The yaw (rad, in [0, 2 pi)) that turns a release orientation - ``q_at_yaw0`` at yaw 0, ``R_z(yaw) q_at_yaw0`` at ``yaw`` (a rolled
    release's axis turns with the yaw) - so that the object axis ``axis_obj`` (the spine: the pinched edge's outward normal) points along the
    horizontal ``outward``; None when that axis is vertical there."""
    s = G.rotate(G.t(q_at_yaw0).float(), G.t(axis_obj).float())
    o = G.t(outward).float()
    if math.hypot(float(s[0]), float(s[1])) < 1e-6 or math.hypot(float(o[0]), float(o[1])) < 1e-9:
        return None
    return (math.atan2(float(o[1]), float(o[0])) - math.atan2(float(s[1]), float(s[0]))) % (2.0 * math.pi)


def spine_facing(quat, axis_obj, outward) -> float:
    """cos of the angle between the object axis ``axis_obj`` at ``quat`` (world, 3-D) and the horizontal direction ``outward``."""
    s = G.rotate(G.t(quat).float(), G.t(axis_obj).float())
    o = G.t(outward).float()
    o3 = th.tensor([float(o[0]), float(o[1]), 0.0])
    return float(th.dot(s, o3)) / max(float(th.linalg.norm(s)) * float(th.linalg.norm(o3)), 1e-9)


def yaws_within(yaws: Sequence[float], centre: float, tol: float) -> List[float]:
    """The yaws (rad) of ``yaws`` within ``tol`` of ``centre`` (wrapped), nearest first, ``centre`` itself excluded."""
    d = [(abs(G.wrap_angle(float(y) - float(centre))), float(y)) for y in yaws]
    return [y for dd, y in sorted(d) if 1e-6 < dd <= float(tol) + 1e-9]


#: ``generation.inside_shelf_place`` (module docstring): its keys and defaults - ``jaw_clear_m`` None turns the open-jaw rule off,
#: ``jaw_sweep_m`` = the oracle place's retreat along -a (``oracle.actions.RETREAT_DIST``)
SHELF_PLACE_DEFAULTS: Dict[str, Any] = {"spine_behind_edge_m": 0.03, "wall_gap_m": 0.04, "side": "near", "release_lean_deg": 9.0, "jaw_clear_m": 0.005,
                                        "jaw_sweep_m": 0.10}
SHELF_SIDES = ("near", "left", "right")
#: the open jaw's retreat sweep is sampled this finely (m)
JAW_SWEEP_STEP_M = 0.005


def shelf_place_spec(value: Any) -> Optional[Dict[str, Any]]:
    """``generation.inside_shelf_place``: true / a dict of :data:`SHELF_PLACE_DEFAULTS` keys -> the full spec; None / false -> None."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict) or set(value) - set(SHELF_PLACE_DEFAULTS):
        raise ValueError(f"inside_shelf_place {value!r}: expected true or a dict of {sorted(SHELF_PLACE_DEFAULTS)}")
    out = {**SHELF_PLACE_DEFAULTS, **value}
    if str(out["side"]).strip().lower() not in SHELF_SIDES:
        raise ValueError(f"inside_shelf_place side {out['side']!r}: expected one of {list(SHELF_SIDES)}")
    out["side"] = str(out["side"]).strip().lower()
    try:
        for k in ("spine_behind_edge_m", "wall_gap_m", "release_lean_deg", "jaw_sweep_m"):
            out[k] = float(out[k])
        out["jaw_clear_m"] = None if out["jaw_clear_m"] is None else float(out["jaw_clear_m"])
    except (TypeError, ValueError):
        raise ValueError(f"inside_shelf_place {value!r}: the lengths (m) and release_lean_deg must be numbers")
    if out["wall_gap_m"] < 0.0 or out["spine_behind_edge_m"] < 0.0 or out["jaw_sweep_m"] < 0.0 or not 0.0 <= out["release_lean_deg"] < 45.0:
        raise ValueError(f"inside_shelf_place {value!r}: lengths >= 0, 0 <= release_lean_deg < 45")
    return out


def shelf_lean_side(side: str, arm: str) -> float:
    """The side board of the shelf place along the edge tangent (+1 = the robot's left facing the opening, -1 its right): ``"left"`` /
    ``"right"``, or ``"near"`` the working arm's side."""
    if side == "left" or (side == "near" and arm == "left"):
        return 1.0
    return -1.0


def leaned_quat(quat, inward, lean_side: float, deg: float) -> th.Tensor:
    """``quat`` turned ``deg`` about the edge's horizontal inward axis so that its top leans toward ``lean_side`` (+1 the edge tangent, the
    robot's left; ``tabletop.upright_lean_quat``'s sense); ``quat`` itself at 0 deg."""
    q = G.t(quat).float()
    if not deg:
        return q
    axis = th.tensor([float(inward[0]), float(inward[1]), 0.0])
    return G.qmul(G.quat_about_axis(axis, -float(lean_side) * math.radians(float(deg))), q)


def open_jaw_reach(eef, mid, inward, lean_side: float, res: float, sweep_m: float, clear: float) -> Optional[float]:
    """How far toward the side board the fully open jaw at the eef pose ``eef`` reaches - m along the edge tangent x ``lean_side`` from the
    edge midpoint ``mid`` -, over its box envelope (``gripper_corners_world``, ``C.GRIPPER_OPEN_M``) and its retreat ``sweep_m`` back along -a,
    counting the points behind the board's front edge (the raster edge line + half a cell ``res``) or within ``clear`` of it; None when none
    is (the jaw outside the compartment)."""
    corners = gripper_corners_world(eef)
    a = G.rotate(G.t(eef[1]).float(), th.tensor([0.0, 0.0, 1.0]))
    inn = th.tensor([float(inward[0]), float(inward[1])])
    tan = th.tensor([-float(inward[1]), float(inward[0])]) * float(lean_side)
    m = G.t(mid).float()[:2]
    n = max(1, int(math.ceil(float(sweep_m) / JAW_SWEEP_STEP_M)))
    best = None
    for k in range(n + 1):
        pts = (corners - a * (float(sweep_m) * k / n))[:, :2] - m
        behind = pts @ inn + 0.5 * float(res) > -float(clear)
        if bool(behind.any()):
            u = float((pts[behind] @ tan).max())
            best = u if best is None else max(best, u)
    return best


def _spec_of(s, key: str, parse):
    """``parse(generation.<key>)`` of the variant (None without a ``gp``: a test stub); a malformed key is ``Reject("spec")``."""
    gp = getattr(s, "gp", None)
    try:
        return parse(None if gp is None else gp.gen(key))
    except ValueError as e:
        raise Reject("spec", f"{getattr(gp, 'variant', '?')}: {e}")


def _roll_mode(s) -> str:
    """``generation.release_roll_axis`` (:func:`roll_axis_mode`; default ``"radial"``)."""
    return _spec_of(s, "release_roll_axis", roll_axis_mode)


def _spine_out(s) -> Optional[float]:
    """``generation.inside_spine_out`` (default false): its tolerance (rad, ``inside_spine_out_tol_deg``) when on, else None."""
    gp = getattr(s, "gp", None)
    if gp is None or not gp.gen("inside_spine_out", False):
        return None
    return _spec_of(s, "inside_spine_out_tol_deg", spine_out_tol)


def _shelf_spec(s, checked: bool = False) -> Optional[Dict[str, Any]]:
    """``generation.inside_shelf_place`` (:func:`shelf_place_spec`; default none); ``checked``: ``Reject("spec")`` when it is on without
    ``inside_spine_out`` (the shelf pose is laid out for the spine-out release)."""
    spec = _spec_of(s, "inside_shelf_place", shelf_place_spec)
    if checked and spec is not None and _spine_out(s) is None:
        raise Reject("spec", f"{s.gp.variant}: generation.inside_shelf_place needs generation.inside_spine_out (the shelf pose is the spine-out release's)")
    return spec


def _qkey(quat) -> Tuple[float, ...]:
    return tuple(round(float(v), 5) for v in G.t(quat).float().tolist())


def shelf_context(s, place) -> Optional[Dict[str, Any]]:
    """A C's shelf context (:func:`shelf_release_xy`, the lean of :meth:`InsideMode.release_quat`) from its ``extras["shelf_place"]``
    (:meth:`InsideMode.shelf_place_at`); None without one."""
    sp = (getattr(place, "extras", None) or {}).get("shelf_place")
    edge = getattr(place, "edge", None)
    if sp is None or edge is None:
        return None
    return {"inward": [float(v) for v in G.t(edge.inward).tolist()[:2]], "mid": [float(v) for v in G.t(edge.mid).tolist()[:2]], "side": float(sp["lean_side"]),
            "board_u": float(sp["board_u_m"]), "res": float(s.place["top"].resolution), "lean_deg": float(sp["release_lean_deg"]),
            "jaw_clear": sp.get("jaw_clear_m"), "sweep": float(sp["jaw_sweep_m"]), "quats": set(), "memo": {}, "facts": sp}


def spine_out_yaws(s, grasp, t: float, place, tol: float) -> List[float]:
    """``generation.inside_spine_out``: the release yaws of the library row ``grasp`` at the roll key ``t`` whose spine (the pinched edge's
    outward normal, :func:`pinched_edge_normal`) faces out of the opening (-``place.edge.inward``): the exact one (:func:`spine_out_yaw`)
    first, then the yaw grid's within ``tol`` (rad), nearest first; [] without a pinched edge."""
    found = pinched_edge_normal(grasp.pose[0], s.rest)
    if found is None:
        return []
    q0 = rolled_quat(s.rest, grasp.pose[0], float(t) - ROLL_KEY, _roll_mode(s))
    yaw = spine_out_yaw(q0, found[0], -G.t(place.edge.inward).float())
    return [] if yaw is None else [yaw] + yaws_within(s.yaws(), yaw, tol)


def shelf_release_xy(s, goal_cell, quat, eef_in_object=None) -> Tuple[float, float, float]:
    """The release xy of the shelf place (module docstring; ``s._shelf_ctx``, the C's, set by :meth:`InsideMode.screen_release_tilts`) for
    the release orientation ``quat`` and the library row ``eef_in_object`` (None: the worst of the library rows), and how far it moved:
    ``goal_cell`` (the upright book's shelf pose) moved away from the side board until the fully open jaw (:func:`open_jaw_reach`) keeps
    ``jaw_clear_m`` from the board, the leaned book :data:`tabletop.SHELF_WALL_GAP_M`. ``goal_cell`` itself without the context or for an
    orientation the context did not lean (anything but the screened / drawn rolled releases: the flat level screen, the stance-C frame test)."""
    x0, y0 = float(goal_cell[0]), float(goal_cell[1])
    ctx = getattr(s, "_shelf_ctx", None)
    q = G.t(quat).float()
    if ctx is None or _qkey(q) not in ctx["quats"]:
        return x0, y0, 0.0
    rows = [eef_in_object] if eef_in_object is not None else [g.pose for g in s.grasps]
    key = (_qkey(q), round(x0, 4), round(y0, 4)) + tuple(tuple(round(float(v), 4) for v in list(G.t(r[0]).tolist()) + list(G.t(r[1]).tolist())) for r in rows)
    memo = ctx["memo"]
    if key not in memo:
        base = th.tensor([x0, y0, 0.0])
        inward, mid, side = ctx["inward"], th.tensor(ctx["mid"]), float(ctx["side"])
        tan = th.tensor([-float(inward[1]), float(inward[0])]) * side
        hull_u = float(((G.local_points_world(s.hull, (base, q))[:, :2] - mid) @ tan).max())
        need = hull_u + TT.SHELF_WALL_GAP_M - float(ctx["board_u"])
        if ctx["jaw_clear"] is not None:
            for r in rows:
                reach = open_jaw_reach(IK.grasp_pose((base, q), r), mid, inward, side, ctx["res"], ctx["sweep"], ctx["jaw_clear"])
                if reach is not None:
                    need = max(need, reach + float(ctx["jaw_clear"]) - float(ctx["board_u"]))
        d = max(0.0, need)
        memo[key] = (x0 - float(tan[0]) * d, y0 - float(tan[1]) * d, d)
    return memo[key]


def footprint_fits(pts_xy: th.Tensor, xs: th.Tensor, ys: th.Tensor, free: th.Tensor, margin: float) -> bool:
    """Whether every footprint point (N, 2), and each moved ``margin`` along the raster axes, falls on a free cell of the raster."""
    if len(xs) < 2 or len(ys) < 2:
        return False
    res_x, res_y = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    offs = th.tensor([[0.0, 0.0], [margin, 0.0], [-margin, 0.0], [0.0, margin], [0.0, -margin]])
    p = (G.t(pts_xy).float()[:, None, :2] + offs[None]).reshape(-1, 2)
    i = th.round((p[:, 0] - float(xs[0])) / res_x).long()
    j = th.round((p[:, 1] - float(ys[0])) / res_y).long()
    if bool((i < 0).any() or (i >= len(xs)).any() or (j < 0).any() or (j >= len(ys)).any()):
        return False
    return bool(free.bool()[i, j].all())


def thin_axis_xy(pts_xy: th.Tensor) -> th.Tensor:
    """The horizontal direction of a footprint's smallest extent (the principal axis of least spread): an upright book's thickness."""
    q = G.t(pts_xy).float()[:, :2]
    q = q - q.mean(dim=0)
    w, v = th.linalg.eigh(q.T @ q)
    return v[:, 0] / v[:, 0].norm().clamp_min(1e-9)


# ----------------------------------------------------------------------------------------------------------------
# the mode
# ----------------------------------------------------------------------------------------------------------------


def free_in_postures(s, cands: Sequence[Stance], postures: Sequence[str]) -> Tuple[List[Stance], Dict[str, int]]:
    """Stance C's collision check: each candidate in the first posture of ``postures`` that is free, the candidates' order kept -
    ``untucked``: :func:`prompt0.body_free`'s default (the working arm untucked, the idle arm's A6 retry, which records the idle posture
    that passed); ``ready`` / ``tucked``: both arms in that posture. A survivor in ``ready`` / ``tucked`` records it as its
    ``idle_posture`` - the posture of its sidecar record (``idle_posture`` / ``arms_q``), of the drive-in and of the carried drive after
    B's departure transition (:meth:`InsideMode.carry_drive_arms`) - and in ``extras["stance_posture"]``; its working arm at C is the
    carry configuration, then the exact IK with the target attached. Returns the survivors and the number per posture."""
    passed: Dict[int, str] = {}
    todo = list(range(len(cands)))
    for name in postures:
        if not todo:
            break
        part = [cands[i] for i in todo]
        free = P0.body_free(s, part) if name == "untucked" else P0.body_free(s, part, posture=name)
        passed.update({i: name for i, f in zip(todo, free.tolist()) if f})
        todo = [i for i in todo if i not in passed]
    out = []
    for i in sorted(passed):
        st = cands[i]
        if passed[i] != "untucked":
            st.idle_posture = passed[i]
            st.extras["stance_posture"] = passed[i]
        out.append(st)
    return out, {name: sum(1 for p in passed.values() if p == name) for name in postures}


def restrict_to_goal_region(s, cells: th.Tensor) -> Tuple[th.Tensor, Optional[Dict[str, Any]]]:
    """``generation.inside_goal_region`` (module docstring; default none = every cell): the goal cells (world xy) inside the box, one INFO
    line with the counts. ``Reject("spec")`` for a malformed key or a box that keeps none of the cells. Returns (cells, facts or None)."""
    try:
        region = goal_region(s.gp.gen("inside_goal_region"))
    except ValueError as e:
        raise Reject("spec", f"{s.gp.variant}: {e}")
    if region is None:
        return cells, None
    origin, yaw = (0.0, 0.0), 0.0
    if region["frame"] == "partner":
        p, q = s.partner.get_position_orientation()
        origin, yaw = (float(p[0]), float(p[1])), float(G.yaw_of(q))
        region.update({"partner_origin": [round(v, 4) for v in origin], "partner_yaw": round(yaw, 4)})
    kept = cells[in_goal_region(cells, region, origin, yaw)] if len(cells) else cells
    log.info("%s: inside goal region of %s (%s frame, x %s, y %s): %d -> %d goal cells (world %s -> %s)", s.gp.variant, s.partner.name, region["frame"],
             region["x"] or "any", region["y"] or "any", len(cells), len(kept), cell_span(cells), cell_span(kept))
    if len(cells) and not len(kept):
        raise Reject("spec", f"{s.gp.variant}: generation.inside_goal_region {s.gp.gen('inside_goal_region')} keeps none of the {len(cells)} goal cells of "
                             f"{s.partner.name} (world {cell_span(cells)})")
    return kept, {**region, "n_before": int(len(cells)), "n_after": int(len(kept))}


class InsideMode:
    """The inside mode of :class:`~ispatialgym.sampling.v2.tasks.iors_large.IORSLarge` (module docstring): the partner's interior surface
    as the goal geometry, stance C at the opening, the release above the surface and the drop test. Every other stage is IORSLarge's."""

    # ------------------------------------------------------------------ partner geometry for C

    def inside_surface_spec(self) -> Dict[str, float]:
        z = self.gp.gen("inside_surface_z_m")
        if z is None:
            raise Reject("spec", f"{self.gp.variant}: generation.inside_surface_z_m (the partner's interior surface, m above the floor) is required "
                                 "for IORS large-scale inside")
        return {"z_m": float(z), "tol_m": float(self.gp.gen("inside_surface_tol_m", SURFACE_TOL_M))}

    def prepare_partner(self):
        """The interior raster of the partner at the variant's height, the goal cells (headroom, the opening shrunk by the half extent +
        ``opening_margin_m``, ``inside_goal_region``), the raster's sides as the stance-C edges (``inside_stance_side``) and the rim of the
        drop test."""
        partner = self.partner
        half, height = self.target_dims()
        spec = self.inside_surface_spec()
        # generation.partner_joint_init (the open door): its links are no part of the stand-off nor of the surface
        static = getattr(self, "partner_static_aabb", None)
        lo, hi = static() if static is not None else (G.t(v).float().clone() for v in partner.aabb)
        try:
            itop = self.world.support_interior(partner, self.floor_z + spec["z_m"], resolution=float(self.num("support_resolution", 0.02)), z_margin=spec["tol_m"],
                                               ignore=list(self.world.spawned.values()))
        except RuntimeError as e:
            raise Reject("inside_surface", f"{partner.name}: no interior surface at {spec['z_m']:.2f} m (+-{spec['tol_m']:.2f}): {e}")
        parts = getattr(self, "partner_part_boxes", None)
        n_part_cells = OJ.mask_out_boxes(itop, parts() if parts is not None else [], pad=0.5 * float(itop.resolution))
        if n_part_cells and not bool(itop.mask.any()):
            raise Reject("inside_surface", f"{partner.name}: every interior cell at {float(itop.z):.3f} m lies under its doors (generation.partner_joint_init)")
        rolled = bool(_rolls_of(self))
        if rolled:
            # oracle-fix 2026-11-04 round 3 (inside_large#01): a rolled release stands the target up - headroom for its longest extent, the
            # goal cells eroded by its half thickness only (each rolled pair's footprint at its yaw is checked by roll_fits)
            ext = (lambda p: p.max(dim=0).values - p.min(dim=0).values)(G.local_points_world(self.hull, (th.zeros(3), self.rest)))
            height, half = float(ext.max()), 0.5 * float(ext.min())
        usable = itop.usable(height, HEADROOM_MARGIN_M)
        idx = th.nonzero(usable)
        cells = th.stack([itop.xs[idx[:, 0]], itop.ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)
        margin = half + float(self.num("opening_margin_m", OPENING_MARGIN_M))
        if self.gp.gen("inside_erosion_m") is not None:  # a wide flat target in a basin whose walls flare (the plate's rim above them)
            margin = float(self.gp.gen("inside_erosion_m"))
        goal_cells, region = restrict_to_goal_region(self, erode_from(cells, raster_rim(itop.mask, itop.xs, itop.ys), margin))
        try:
            side = side_vector(self.gp.gen("inside_stance_side"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")
        all_edges = [EdgeFrame.from_edge(name, mid, normal, length, overhang(mid, normal, lo, hi)) for name, mid, normal, length in itop.edges()]
        edges = edges_facing(all_edges, side)
        try:
            top_z = float(self.support_top(partner).z)
        except (RuntimeError, ValueError):
            top_z = None
        rim_z, rim_src = rim_height(top_z, itop.z, height, float(hi[2]))
        if self.gp.gen("inside_rim_z_m") is not None:
            rim_z, rim_src = self.floor_z + float(self.gp.gen("inside_rim_z_m")), "generation.inside_rim_z_m"
        facts = {"z_requested_above_floor": spec["z_m"], "tol_m": spec["tol_m"], "z_world": round(float(itop.z), 4), "z_above_floor": round(float(itop.z) - self.floor_z, 4),
                 "n_board_cells": int(itop.mask.sum()), "n_free_cells": int(itop.free.sum()), "n_usable_cells": int(len(cells)), "n_goal_cells": int(len(goal_cells)),
                 "erosion_m": round(margin, 4), "open_edges": list(itop.open_edges), "board_pieces": len(itop.board_pieces), "cells_under_doors": n_part_cells,
                 "headroom_m": (itop.to_json() or {}).get("headroom_m"), "rim_z": round(rim_z, 4), "rim_source": rim_src,
                 "stance_side": None if side is None else [round(float(v), 3) for v in side], "stance_edges": [e.name for e in edges]}
        if region is not None:
            facts["goal_region"] = region
        self.place = {"top": itop, "cells": goal_cells, "edges": edges, "z": float(itop.z), "height": height, "half": half, "surface": facts, "rim_z": rim_z,
                      "z_requested": spec["z_m"]}
        self.template_facts["inside_surface"] = facts
        self.template_facts["partner"] = {"name": partner.name, "n_goal_cells": int(len(goal_cells)), "edges": [e.to_json() for e in edges]}
        log.info("%s: inside surface of %s at %.3f m (requested %.2f +- %.2f above the floor): %d board / %d usable / %d goal cells (eroded %.2f m), rim %.3f (%s), "
                 "stance sides %s", self.gp.variant, partner.name, float(itop.z), spec["z_m"], spec["tol_m"], int(itop.mask.sum()), len(cells), len(goal_cells), margin,
                 rim_z, rim_src, [e.name for e in edges])
        if not edges:
            raise Reject("inside_no_side", f"{partner.name}: no side of the interior raster faces {self.gp.gen('inside_stance_side')!r} "
                                           f"(sides {[(e.name, [round(float(v), 2) for v in (-e.inward).tolist()]) for e in all_edges]})")
        if len(goal_cells) == 0:
            raise Reject("inside_no_goal_cell", f"{partner.name}: no interior cell at {float(itop.z):.3f} m fits the target ({len(cells)} with headroom >= "
                                                f"{height + HEADROOM_MARGIN_M:.2f} m, none {margin:.2f} m from the board's outline)")

    def goal_region_points(self, from_xy) -> th.Tensor:
        """Stance A's partner region: the goal-cell patch on the interior surface (as ontop's on the top) + the partner's nearest bottom
        edge (IORSLarge's)."""
        edge_pts = super().goal_region_points(from_xy)
        cells = self.place.get("cells")
        if cells is None or len(cells) == 0:
            return edge_pts
        g = nearest_cell(cells, from_xy)
        z = float(self.place["z"])
        r = float(self.num("goal_patch_half_m", 0.10))
        patch = [th.tensor([float(g[0]) + dx, float(g[1]) + dy, z]) for dx, dy in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r))]
        return th.cat([th.stack(patch), edge_pts], dim=0)

    # ------------------------------------------------------------------ the goal-side band

    def inside_band(self, arm: str) -> Optional[RB.ReachBands]:
        """The goal-side band of ``arm`` (the target on the partner's interior surface, ``surface_z`` = the variant's requested height): the
        arm's own band, else (the other arm) the default arm's mirrored, else None - no band filter, the reach margin orders the trunks.
        An empty band counts as none. Never computed in-process (a band job publishes it)."""
        memo = self.__dict__.setdefault("_inside_bands", {})
        if arm in memo:
            return memo[arm]
        c = self.choice.models["target"]
        z = float(self.place["z_requested"])
        joints = (self.template_facts.get("partner_joint_init") or {}).get("applied") or None  # the band of the partner with its door open
        rb, why = None, ""
        try:
            rb = RB.ReachBands.load(self.world.scene_model, self.partner.name, f"{c.category}/{c.model}", self.grasp_key, scale=c.scale, arm=arm, surface_z=z,
                                    support_joints=joints)
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
                rb, why = None, "empty band"
        except FileNotFoundError:
            why = "no band"
        arm0 = TT.default_arm(self)
        if rb is None and arm != arm0:
            base = self.inside_band(arm0)
            if base is not None and RB.mirrorable(base.meta):
                rb, why = base.mirrored(), f"the {arm0} arm's band mirrored"
        log.info("%s: inside goal-side band (%s arm, surface_z %.2f%s): %s", self.gp.variant, arm, z, f", support joints {joints}" if joints else "",
                 rb.path if rb is not None and not why else why or "none")
        self.template_facts.setdefault("inside_bands", {})[arm] = None if rb is None else {**(TT.band_facts(rb) or {}), "n_cells": int(rb.meta.get("n_reachable_total", 0) or 0)}
        memo[arm] = rb
        return rb

    # ------------------------------------------------------------------ stance C

    def _placement_candidates_at(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C at one side of the interior raster (module docstring): stand-off x lateral x trunk, collision-free in the first free
        posture of ``inside_stance_postures`` (:func:`free_in_postures`), the target at the goal cell in frame (without a band: then the
        :data:`NO_BAND_TRUNKS` per base pose with the most reach margin); each survivor carries ``goal_cell`` (and ``reach_margin_m``
        without a band). With ``inside_shelf_place`` the goal cell picks the bay and :meth:`shelf_place_at` the pose in it (``goal_cell`` =
        the upright book's shelf pose, ``extras["shelf_place"]`` its facts; a cell without a side board there drops the candidate)."""
        arm = arm or self.seg_arm
        partner = self.partner
        try:
            postures = stance_postures(self.gp.gen("inside_stance_postures"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")
        half, height = float(self.place["half"]), float(self.place["height"])
        rb = self.inside_band(arm)
        self.place["band"] = rb
        self.place.setdefault("bands", {})[arm] = rb
        trunks = rb.trunks() if rb is not None else [(float(t), float(k)) for t in RB.TRANSLATES for k in RB.TILTS]
        if rb is None and self.gp.gen("inside_trunks"):
            # oracle-fix 2026-11-06 (owner feedback on the touching_large#02 demo, the umbrella on the hall tree's bench board): opt-in
            # generation.inside_trunks - the C trunks without a goal-side band instead of the whole grid (whose reach-margin cut keeps the deepest
            # leans: offline, a top-down release 0.55-0.65 m ahead at 0.65 m is out of their IK reach or of the head camera's frame)
            trunks = [(float(t), float(k)) for t, k in self.gp.gen("inside_trunks")]
        half_run = edge.length / 2 - (half + 0.10)
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        goal_cells = self.place["cells"]
        z_mid = float(self.place["z"]) + height / 2
        lattice = float(rb.meta.get("lattice", RB.LATTICE)) if rb is not None else RB.LATTICE
        stats: Dict[str, Any] = {"goal_cells": int(len(goal_cells)), "band": rb is not None, "band_trunks": len(trunks), "lattice": 0, "base_ok": 0, "band_cells": 0,
                                 "goal_in_band": 0}
        shelf = _shelf_spec(self, checked=True)
        cands: List[Stance] = []
        for so in [float(v) for v in (self.gp.gen("inside_stand_offs_m") or STAND_OFFS)]:
            for lat in laterals:
                xy = edge.base_xy(so, lat)
                stats["lattice"] += len(trunks)
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += len(trunks)
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), tr, ti, so, edge, None, lat, None, [arm])
                    if rb is not None:
                        cells = self.band_cells_world(rb, st)
                        if len(cells) == 0:
                            continue
                        stats["band_cells"] += 1
                        pool = self._snap(cells, goal_cells, lattice)
                    else:
                        pool = goal_cells
                    if len(pool) == 0:
                        continue
                    stats["goal_in_band"] += 1
                    # oracle-fix 2026-11-04 round 3 (inside_large#01): opt-in generation.inside_goal_lateral_m - the goal cell nearest a point that far to the
                    # working arm's side of the base (in front of its shoulder; straight ahead of the base the standing book's pre-place put the forearm
                    # across the chest: right_arm_link4/5 x torso_link4 13-27 mm, joint2 at its limit)
                    lat_g = float(self.gp.gen("inside_goal_lateral_m", 0.0) or 0.0)
                    aim = xy
                    if lat_g:
                        yaw_b = float(st.base_pose2d[2])
                        side = -1.0 if arm == "right" else 1.0  # the right arm on the base's -y side
                        aim = th.tensor([float(xy[0]) - math.sin(yaw_b) * side * lat_g, float(xy[1]) + math.cos(yaw_b) * side * lat_g])
                    gc = nearest_cell(pool, aim)
                    if shelf is not None:
                        # oracle-fix 2026-11-06 (owner feedback on inside_large#01): the goal cell picks the bay, the shelf place the pose in it
                        sp = self.shelf_place_at(edge, gc, arm)
                        if sp is None:
                            stats["shelf_place_failed"] = stats.get("shelf_place_failed", 0) + 1
                            continue
                        gc = th.tensor(sp["xy"])
                        st.extras["shelf_place"] = sp
                    st.extras["goal_cell"] = [float(gc[0]), float(gc[1])]
                    if rb is None:
                        st.extras["reach_margin_m"] = round(self.reach_margin(st, [float(gc[0]), float(gc[1]), z_mid], arm), 4)
                    cands.append(st)
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: inside stance C at %s side %s: no candidate (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        before = list(cands)
        cands, by_posture = free_in_postures(self, cands, postures)
        stats["collision_free"] = len(cands)
        if os.environ.get("ISPATIALGYM_STANCE_BLAME"):  # oracle-fix 2026-11-04 (owner item 3): which body every rejected candidate hits
            kept = {id(st) for st in cands}
            P0.blame_stances(self, [st for st in before if id(st) not in kept], f"inside stance C {partner.name} side {edge.name}", postures=postures)
        if postures != list(STANCE_POSTURES):
            stats["postures"] = by_posture
        if not cands:
            log.info("%s: inside stance C at %s side %s: every candidate collides (%s)", self.gp.variant, partner.name, edge.name, stats)
            return []
        cone = self.gp.gen("inside_axis_cone_deg")
        classed = []
        for st, cam in zip(cands, P0.camera_poses(self, cands)):
            gc = st.extras["goal_cell"]
            focus = th.tensor([gc[0], gc[1], z_mid])
            # oracle-fix 2026-11-06: a mode may name the orientation the target is released at there (the touching surface place's yaw along the
            # board - the 0.82 m umbrella at the rest yaw reached 0.43 m toward the base, under the head camera's frame); the rest orientation else
            quat_f = getattr(self, "frame_quat_at", lambda _gc: self.rest)(gc)
            corners = self.hull_corners_world(self.hull, self.goal_pose_at(gc, quat_f))
            if not P0.in_frame(self, cam, corners):
                if os.environ.get("ISPATIALGYM_STANCE_BLAME"):
                    uv, depth = self.world.camera_model.project(cam, corners)
                    log.info("%s: STANCE_BLAME inside stance C %s side %s: collision-free %s trunk (%.2f, %+.2f) so %.2f: the target at the goal cell %s "
                             "out of frame (uv x %.0f..%.0f y %.0f..%.0f of %d px, depth min %.2f m, axis %.1f deg)%s", self.gp.variant, partner.name, edge.name,
                             [round(float(v), 2) for v in st.base_pose2d], st.translate, st.tilt, st.stand_off, [round(v, 2) for v in gc],
                             float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()), float(uv[:, 1].max()), self.world.camera_model.width,
                             float(depth.min()), math.degrees(P0.axis_angle(self, cam, focus)),
                             f"; reach margin {st.extras['reach_margin_m']}" if "reach_margin_m" in st.extras else "")
                continue
            ang = P0.axis_angle(self, cam, focus)
            st.extras["axis_angle_deg"] = math.degrees(ang)
            if cone is not None and ang > math.radians(float(cone)):
                continue
            st.focus = focus
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        if rb is None:
            kept = {id(st) for st in best_per_base([st for st, _ in classed], int(self.num("inside_no_band_trunks", NO_BAND_TRUNKS)))}
            classed = [(st, k) for st, k in classed if id(st) in kept]
            stats["no_band_kept"] = len(classed)
        out = P0.best_class(classed)
        log.info("%s: inside stance C at %s side %s: %d survivors (%s)", self.gp.variant, partner.name, edge.name, len(out), stats)
        return out

    def carry_drive_arms(self, place: Stance, tname: str, arm: str) -> Optional[Dict[str, th.Tensor]]:
        """The arms the carried drive after B's departure transition is checked with: the oracle's departure transition puts the idle arm
        in C's recorded posture (``move_idle_arms``) and the carried drive keeps it there, so a C kept in ``ready`` / ``tucked`` by
        :func:`free_in_postures` is driven to with C's arms and the holding arm at its carry configuration; otherwise IORSLarge's."""
        q_carry = self.carry_arm_q.get(tname)
        if not place.extras.get("stance_posture") or q_carry is None:
            return super().carry_drive_arms(place, tname, arm)
        r, l, _ = IK.stance_arm_vectors(self.kit, place)
        return {"right": r, "left": l, arm: q_carry}

    # ------------------------------------------------------------------ the shelf place (oracle-fix 2026-11-06, inside_large#01)

    def shelf_up_quat(self, edge: EdgeFrame) -> Optional[th.Tensor]:
        """The upright spine-out release the shelf place is laid out for: the first library row at the first ``release_roll_deg`` roll
        (:func:`rolled_quat`, ``release_roll_axis``), turned to the yaw whose spine faces out of ``edge`` (:func:`spine_out_yaw`); None
        without a roll, a row or a horizontal pinched edge."""
        rolls, g0 = _rolls_of(self), next(iter(self.grasps), None)
        found = None if g0 is None else pinched_edge_normal(g0.pose[0], self.rest)
        if not rolls or found is None:
            return None
        q0 = rolled_quat(self.rest, g0.pose[0], rolls[0], _roll_mode(self))
        yaw = spine_out_yaw(q0, found[0], -G.t(edge.inward).float())
        return None if yaw is None else G.qmul(G.quat_from_yaw(yaw), q0)

    def shelf_place_at(self, edge: EdgeFrame, cell, arm: str) -> Optional[Dict[str, Any]]:
        """``generation.inside_shelf_place`` (module docstring) for the goal cell ``cell`` of stance C's side ``edge``: the side board of the
        cell's bay on the spec's ``side`` (a ray from the cell's lateral, ``tabletop.shelf_wall_lateral``) and the upright book's shelf pose
        there (``tabletop.shelf_lean_pose`` with :meth:`shelf_up_quat` as the rest: the spine ``spine_behind_edge_m`` behind the front edge,
        the near cover ``wall_gap_m`` from the board). Returns the JSON facts (``xy`` = the new goal cell, ``lean_side``, ``board_u_m``: the
        board's face along the tangent x lean_side from ``edge.mid``) or None (no board there / the pose off the board: one INFO line);
        memoised per template."""
        spec = _shelf_spec(self)
        side = shelf_lean_side(spec["side"], arm)
        memo = self.place.setdefault("shelf_places", {})
        key = (edge.name, round(float(cell[0]), 3), round(float(cell[1]), 3), side)
        if key in memo:
            return memo[key]
        memo[key] = None
        q_up = self.shelf_up_quat(edge)
        if q_up is None:
            log.info("%s: inside shelf place: no upright spine-out release (release_roll_deg %s, %d rows)", self.gp.variant, self.gp.gen("release_roll_deg"), len(self.grasps))
            return None
        rest_up = G.qmul(G.quat_from_yaw(-edge.yaw), q_up)
        mid, tan = G.t(edge.mid).float()[:2], G.t(edge.tangent).float()
        lat0 = float(th.dot(G.t(cell).float()[:2] - mid, tan))
        try:
            pose, facts = TT.shelf_lean_pose(self, self.hull, rest_up, self.place["top"], edge, self.partner, side, wall_gap=spec["wall_gap_m"],
                                             spine_behind=spec["spine_behind_edge_m"], lateral0=lat0)
        except Reject as e:
            log.info("%s: inside shelf place at cell %s (side %s, %s board): %s", self.gp.variant, [round(float(v), 3) for v in cell[:2]], edge.name,
                     "left" if side > 0 else "right", e)
            return None
        u_a = float(th.dot(pose[0][:2] - mid, tan * side))
        cover = float((G.local_points_world(self.hull, (th.zeros(3), q_up))[:, :2] @ (tan * side)).max())
        out = {**{k: spec[k] for k in SHELF_PLACE_DEFAULTS if k != "side"}, "side_rule": spec["side"], "side": "left" if side > 0 else "right", "lean_side": side,
               "xy": [round(float(pose[0][0]), 5), round(float(pose[0][1]), 5)], "cell": [round(float(cell[0]), 4), round(float(cell[1]), 4)], "edge": edge.name,
               "board_u_m": round(u_a + cover + spec["wall_gap_m"], 5), "lateral0_m": round(lat0, 4), "up_quat": [round(float(v), 5) for v in q_up.tolist()],
               "shelf_lean_pose": {k: facts[k] for k in ("depth_m", "ray_z", "wall_lateral_m", "lateral_m", "n_bottom_points") if k in facts}}
        log.info("%s: inside shelf place at cell %s (side %s): the %s board's face %.3f m along the tangent, the upright book at %s (spine %.3f m behind the "
                 "front edge, near cover %.3f m from the board)", self.gp.variant, out["cell"], edge.name, out["side"], float(facts["wall_lateral_m"]), out["xy"],
                 spec["spine_behind_edge_m"], spec["wall_gap_m"])
        memo[key] = out
        return out

    def release_eef_in_object(self):
        """IORSLarge's (None unless ``release_lift_per_grasp``); with a shelf context the recorded grasp's ``T_obj→eef`` - the shelf place's
        release xy is the drawn row's own (:func:`shelf_release_xy`), the goal and the settle re-derive it."""
        if getattr(self, "_shelf_ctx", None) is not None:
            rec = self.sidecar.grasp_of(self.tname)
            if rec is not None:
                return rec.eef_in_object
        return super().release_eef_in_object()

    # ------------------------------------------------------------------ the release (the small-scale drop)

    def release_lift_per_grasp(self) -> bool:
        """IORSLarge's hook: ``generation.release_lift_per_grasp`` (module docstring; default false)."""
        try:
            return lift_per_grasp(self.gp.gen("release_lift_per_grasp"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")

    def release_above(self, quat, eef_in_object=None) -> float:
        """Height of the target's bottom above the interior surface at the release: ``drop_height_m``, raised until the open gripper's
        lowest point clears the surface by ``release_clearance_m`` for every library grasp (one height for every grasp: the F_goal
        screening poses the target before a grasp is drawn) - with ``generation.release_lift_per_grasp`` and a grasp's
        ``eef_in_object`` (``T_obj→eef``), for that grasp's open gripper alone. Memoised per orientation (and grasp)."""
        quat = G.t(quat).float()
        own = eef_in_object is not None and self.release_lift_per_grasp()
        key = tuple(round(float(v), 4) for v in quat.tolist())
        if own:
            key += tuple(round(float(v), 4) for v in G.t(eef_in_object[0]).tolist() + G.t(eef_in_object[1]).tolist())
        memo = self.__dict__.setdefault("_inside_release", {})
        if key not in memo:
            drop = float(self.num("drop_height_m", DROP_HEIGHT_M))
            z_s = float(self.place["z"])
            pos = th.tensor([0.0, 0.0, z_s + self.world.bottom_offset(self.hull, quat) + drop])
            low_of = lambda p: float(gripper_corners_world(IK.grasp_pose((pos, quat), p))[:, 2].min())  # noqa: E731
            lows = [low_of(g.pose) for g in self.grasps]
            memo[key] = drop + release_lift([low_of(eef_in_object)] if own else lows, z_s, float(self.num("release_clearance_m", RELEASE_CLEARANCE_M)))
            if not self.__dict__.get("_inside_release_logged"):
                self.__dict__["_inside_release_logged"] = True
                self._log_release_lift(lows, z_s, drop)
        return memo[key]

    def _log_release_lift(self, lows: Sequence[float], z_s: float, drop: float) -> None:
        """Once per template (the first release orientation :meth:`release_above` computes): the lift each library grasp needs for its open
        gripper to clear the surface by ``release_clearance_m`` and the one used, their max - each grasp its own with
        ``release_lift_per_grasp`` (owner 2026-10-13 ~00:50 items 2 / 3)."""
        clearance = float(self.num("release_clearance_m", RELEASE_CLEARANCE_M))
        per = ", ".join(f"{getattr(g, 'name', i)}: {release_lift([lo], z_s, clearance):.3f} m" for i, (g, lo) in enumerate(zip(self.grasps, lows)))
        if self.release_lift_per_grasp():
            log.info("%s: inside release lift per grasp: {%s}, used per grasp (release_lift_per_grasp: the drawn grasp's own; max %.3f m) (drop_height_m %.3f, "
                     "release_clearance_m %.3f: the target's bottom drop_height_m + the drawn grasp's lift above the surface)", self.gp.variant, per,
                     release_lift(lows, z_s, clearance), drop, clearance)
            return
        log.info("%s: inside release lift per grasp: {%s}, used (max) %.3f m (drop_height_m %.3f, release_clearance_m %.3f: the target's bottom %.3f m "
                 "above the surface)", self.gp.variant, per, release_lift(lows, z_s, clearance), drop, clearance, drop + release_lift(lows, z_s, clearance))

    def goal_pose_at(self, goal_cell, quat, eef_in_object=None) -> G.Pose:
        """The release pose: the goal cell, the target's bottom :meth:`release_above` above the interior surface (the goal check settles it);
        ``eef_in_object``: the grasp the release is for (:meth:`release_above`; the shelf place's xy, :func:`shelf_release_xy`)."""
        quat = G.t(quat).float()
        z = float(self.place["z"]) + self.world.bottom_offset(self.hull, quat) + self.release_above(quat, eef_in_object)
        x, y, _ = shelf_release_xy(self, goal_cell, quat, eef_in_object)
        return th.tensor([x, y, z]), quat

    # ------------------------------------------------------------------ the release tilt (owner 2026-10-12 ~22:35, inside_large#03)

    def release_tilts_deg(self) -> List[float]:
        try:
            return release_tilts(self.gp.gen("release_tilt_deg"))
        except ValueError as e:
            raise Reject("spec", f"{self.gp.variant}: {e}")

    def release_quat(self, level_quat, grasp, tilt_deg: float):
        """IORSLarge's hook: the release orientation of ``grasp`` (a library row) at ``tilt_deg`` - :func:`tilted_quat` about the axis
        perpendicular to the row's radial direction; a roll key (``ROLL_KEY + roll``, ``generation.release_roll_deg``) is
        :func:`rolled_quat` about the radial direction itself (``release_roll_axis`` ``"edge_normal"``: about the pinched edge's normal),
        leaned toward the side board with a shelf context (``inside_shelf_place``)."""
        if is_roll(tilt_deg):
            # oracle-fix 2026-11-04 round 4 (opt-in generation.inside_free_yaw): after a C's screen, the drawn row's release yaw is the one its rolled
            # release passed at (the screen itself passes explicit yaws: the map is empty while it runs)
            yg = (getattr(self, "_free_roll_yaw", None) or {}).get((getattr(grasp, "name", None), float(tilt_deg)))
            if yg is not None:
                level_quat = G.qmul(G.quat_from_yaw(float(yg)), self.rest)
            q = rolled_quat(level_quat, grasp.pose[0], float(tilt_deg) - ROLL_KEY, _roll_mode(self))
            ctx = getattr(self, "_shelf_ctx", None)
            if ctx is not None:
                # oracle-fix 2026-11-06 (owner feedback on inside_large#01): released leaning toward the side board, past the tipping angle - the
                # settle tips it onto the board (touching#03's look); the orientation is the context's, so goal_pose_at moves it off the board
                q = leaned_quat(q, ctx["inward"], ctx["side"], ctx["lean_deg"])
                ctx["quats"].add(_qkey(q))
            return q
        return tilted_quat(level_quat, grasp.pose[0], tilt_deg)


    def pre_place_of(self, place_eef, place: Stance):
        """IORSLarge's hook: with ``generation.inside_pre_outward`` the pre-place is the release moved ``inside_pre_m`` (default PRE_DIST)
        straight out of the opening - along the stance side's outward normal, the orientation kept - instead of back along the gripper's
        approach, which for a side pinch runs across a narrow compartment (inside_large#02: the jar's attached pre-place into the fridge's
        side walls, 37-52 exact-IK fails per run; oracle-fix 2026-11-04 round 3). The oracle tracks the recorded pre-place -> place line."""
        if not self.gp.gen("inside_pre_outward") or getattr(place, "edge", None) is None:
            return IK.pre_pose(place_eef)
        d = float(self.num("inside_pre_m", IK.PRE_DIST))
        n = G.t(place.edge.inward).float()
        out = th.tensor([-float(n[0]), -float(n[1]), 0.0]) / max(float(th.linalg.norm(n[:2])), 1e-9)
        return G.t(place_eef[0]).float() + out * d, G.t(place_eef[1]).float()

    def pre_place_tag(self, place: Stance):
        if not self.gp.gen("inside_pre_outward") or getattr(place, "edge", None) is None:
            return None
        n = place.edge.inward
        # oracle-fix 2026-11-06: the shelf place's open-jaw rule moves the screened releases by what the yaw-0 pose of the screen cache key
        # does not show - its numbers join the key (the tag is part of it)
        sp = (getattr(place, "extras", None) or {}).get("shelf_place")
        shelf = f" shelf jaw {sp.get('jaw_clear_m')} {sp.get('jaw_sweep_m')}" if sp else ""
        return f"outward {float(n[0]):.3f},{float(n[1]):.3f} {float(self.num('inside_pre_m', IK.PRE_DIST)):.3f}{shelf}"

    def blame_roll(self, release_xy, yaws, d_yaw: float, place: Stance, arm: str, ignore_place, t: float) -> None:
        """Opt-in (``ISPATIALGYM_STANCE_BLAME``, oracle-fix 2026-11-04 round 3): why a rolled release has an empty F_goal at this C - for the
        first two grasps, at the screened yaw whose pinched edge faces the base most, the [pre-place, place] reached with collisions off
        (``_place_ik_contacts``) and the footprint rule."""
        try:
            base = th.tensor([float(place.base_pose2d[0]), float(place.base_pose2d[1])])
            to_base = base - G.t(release_xy).float()[:2]
            for g in list(self.grasps)[:2]:
                def facing(yw, g=g):
                    q = self.release_quat(G.qmul(G.quat_from_yaw(yw), self.rest), g, t)
                    p = G.rotate(q, G.t(g.pose[0]).float())
                    return float(p[:2] @ to_base) / max(float(th.linalg.norm(p[:2])) * float(th.linalg.norm(to_base)), 1e-9)
                yw = max([y_ + d_yaw for y_ in yaws], key=facing)
                quat = self.release_quat(G.qmul(G.quat_from_yaw(yw), self.rest), g, t)
                pose = self.goal_pose_at(release_xy, quat, g.pose)
                eef = IK.grasp_pose(pose, g.pose)
                pre = self.pre_place_of(eef, place)
                q0 = self.ikc.q_locked(place)
                log.info("%s: STANCE_BLAME roll %g F_goal at %s (C base %s trunk (%.2f, %+.2f)) yaw %.2f (edge facing the base %.2f) grasp %s, footprint %s: "
                         "pre-place: %s | place: %s", self.gp.variant, t - ROLL_KEY, [round(float(v), 2) for v in release_xy[:2]],
                         [round(float(v), 2) for v in place.base_pose2d], place.translate, place.tilt, yw, facing(yw), getattr(g, "name", g),
                         self.roll_fits(pose[0][:2], quat) or "fits", self._place_ik_contacts(pre, q0, arm, [])[:240],
                         self._place_ik_contacts(eef, q0, arm, list(ignore_place))[:240])
        except Exception as e:  # noqa: BLE001 - a diagnostic never fails the draw
            log.info("%s: STANCE_BLAME roll: blame failed (%r)", self.gp.variant, e)

    def roll_fits(self, xy, quat) -> Optional[str]:
        """A rolled release's footprint rule at the goal cell ``xy``: every hull point over a free board cell with ``roll_footprint_margin_m``
        (default 0.005 m) to spare, and with ``inside_wall_hug_m`` set, a side wall within that distance along the footprint's thin axis (the
        book stands beside a side board and leans on it if it tips). None when it fits, else why."""
        itop = self.place["top"]
        pts = G.local_points_world(self.hull, (th.tensor([float(xy[0]), float(xy[1]), 0.0]), G.t(quat).float()))[:, :2]
        margin = float(self.num("roll_footprint_margin_m", 0.005))
        if not footprint_fits(pts, itop.xs, itop.ys, itop.free, margin):
            return "footprint_off_board"
        hug = self.gp.gen("inside_wall_hug_m")
        if hug is not None:
            t = thin_axis_xy(pts)
            if all(footprint_fits(pts + t * sgn * float(hug), itop.xs, itop.ys, itop.free, 0.0) for sgn in (1.0, -1.0)):
                return "no_side_wall_within_hug"
        return None

    def screen_release_tilts(self, F_goal, release_xy, yaws, d_yaw: float, place: Stance, arm: str, ignore_place, yaw_step: float):
        """IORSLarge's hook after the level F_goal screen of a C candidate: each ``generation.release_tilt_deg`` tilt is an alternative to
        the level release - screened per library row (the tilt axis follows the row's radial direction) at the yaws of the level screen,
        the release :meth:`goal_pose_at` the tilted orientation (the tilted hull's lowest point :meth:`release_above` the surface, the row's
        own lift with ``release_lift_per_grasp``). Returns
        the level pairs followed by the pairs only a tilt admits, and per pair key (``iors_large.pair_key``) the tilts (deg, 0 = level) it
        passes in the order the draw tries them; one log line per C with F_goal per tilt. Without the key: ``(F_goal, {})``."""
        from ispatialgym.sampling.v2.tasks.iors_large import pair_key

        # oracle-fix 2026-11-06 (inside_shelf_place): this C's shelf context - the lean of its rolled releases, their xy off the side board
        self._shelf_ctx = shelf_context(self, place)
        tilts = self.release_tilts_deg() + [ROLL_KEY + r for r in _rolls_of(self)]
        if not tilts:
            return F_goal, {}
        # opt-in generation.inside_free_yaw (oracle-fix 2026-11-04 round 4, hardback inside_large#01: F_init 9 x F_goal 1-3 of the rolled releases met at no
        # carried yaw): a rolled release is screened over the whole yaw grid, a row passing at any yaw pairs with every init yaw, and release_quat gives
        # the drawn row that yaw (the book stands in the compartment whatever yaw it was picked at; the wrist turns it)
        free_yaw = bool(self.gp.gen("inside_free_yaw", False))
        # oracle-fix 2026-11-06 (opt-in generation.inside_spine_out, owner feedback on inside_large#01: the first free yaw that passed stood the book
        # diagonal, its spine 45 deg off the opening's normal): a rolled release is screened only at the yaw(s) whose spine faces out of the opening
        spine_tol = _spine_out(self)
        if spine_tol is not None and not free_yaw:
            raise Reject("spec", f"{self.gp.variant}: generation.inside_spine_out needs generation.inside_free_yaw (the release yaw is not the carried one)")
        spine_yaws: Dict[str, List[float]] = {}
        self._free_roll_yaw = {}
        free_found: Dict[Tuple[Any, float], float] = {}
        tilts_of: Dict[Tuple[int, Any], List[float]] = {}
        level = bool(self.gp.gen("release_level", True))
        if level:
            for y, g in F_goal:
                tilts_of.setdefault(pair_key(y, g, yaw_step), []).append(0.0)
        out, counts = (list(F_goal) if level else []), [(0.0, len(F_goal) if level else 0)]
        misfit: Dict[str, int] = {}
        for t in tilts:
            n = 0
            for g in self.grasps:
                pose_at = lambda xy, yaw, g=g, t=t: self.goal_pose_at(xy, self.release_quat(G.qmul(G.quat_from_yaw(yaw), self.rest), g, t), g.pose)  # noqa: E731
                tag = getattr(self, "pre_place_tag", lambda _p: None)(place)
                pre_kw = {"pre_of": (lambda eef: self.pre_place_of(eef, place)), "pre_tag": tag} if tag else {}
                screen_yaws = list(self.yaws()) if (free_yaw and is_roll(t)) else [y_ + d_yaw for y_ in yaws]
                if spine_tol is not None and is_roll(t):
                    screen_yaws = spine_out_yaws(self, g, t, place, spine_tol)
                    spine_yaws[f"{getattr(g, 'name', g)} roll {t - ROLL_KEY:g}"] = [round(y_, 3) for y_ in screen_yaws]
                    if not screen_yaws:
                        continue
                for y, gg in self.ikc.feasible_sets([release_xy], screen_yaws, [g], pose_at, place, arm, ignore_place, **pre_kw).get(0, []):
                    if is_roll(t):
                        q_r = self.release_quat(G.qmul(G.quat_from_yaw(y), self.rest), gg, t)
                        why = self.roll_fits(self.goal_pose_at(release_xy, q_r, gg.pose)[0][:2], q_r)  # at the release's own xy (the shelf place's)
                        if why is not None:
                            misfit[why] = misfit.get(why, 0) + 1
                            continue
                    if free_yaw and is_roll(t):
                        free_found.setdefault((getattr(gg, "name", None), float(t)), float(y))
                        continue
                    k = pair_key(y - d_yaw, gg, yaw_step)
                    if k not in tilts_of:
                        out.append((y - d_yaw, gg))
                    tilts_of.setdefault(k, []).append(t)
                    n += 1
            if free_yaw and is_roll(t):
                for (name, tt), yg in free_found.items():
                    if tt != float(t):
                        continue
                    gg = next((x for x in self.grasps if getattr(x, "name", None) == name), None)
                    for y0 in yaws:  # every init yaw: the release yaw is yg
                        k = pair_key(y0, gg, yaw_step)
                        if k not in tilts_of:
                            out.append((y0, gg))
                        tilts_of.setdefault(k, []).append(t)
                        n += 1
            counts.append((t, n))
            if n == 0 and is_roll(t) and os.environ.get("ISPATIALGYM_STANCE_BLAME"):
                self.blame_roll(release_xy, yaws, d_yaw, place, arm, ignore_place, t)
        self._free_roll_yaw = dict(free_found)  # read by release_quat for this C's draws
        log.info("%s: inside release tilt F_goal at C (d %.2f λ %+.2f trunk (%.2f, %.2f)) goal cell %s: %s%s", self.gp.variant, place.stand_off or 0.0, place.lateral,
                 place.translate, place.tilt, [round(float(v), 2) for v in release_xy],
                 ", ".join((f"roll {t - ROLL_KEY:g} deg" if is_roll(t) else f"{t:g} deg") + f": F_goal {n}" for t, n in counts), f" (roll footprint misfits {misfit})" if misfit else "")
        if spine_yaws or self._shelf_ctx is not None:
            ctx = self._shelf_ctx
            log.info("%s: inside spine-out release yaws at C (d %.2f λ %+.2f): %s%s", self.gp.variant, place.stand_off or 0.0, place.lateral, spine_yaws or "off",
                     "" if ctx is None else f"; shelf place on the {ctx['facts']['side']} board (its face {ctx['board_u']:.3f} m to that side of the edge "
                                            f"midpoint), lean {ctx['lean_deg']:g} deg, the releases moved off it {sorted({round(v[2], 3) for v in ctx['memo'].values()})} m")
        return out, tilts_of

    # ------------------------------------------------------------------ goal check

    def check_goal_upright(self, draft: Draft):
        """SamplerV2's up-axis check after the goal settle; a tilted release (``summary.release_tilt_deg`` > 0) is measured against its level
        orientation (``summary.release_level_quat``): the stock reference, the goal pose, is the tilted release, which a target that lands
        flat is ``release_tilt_deg`` off."""
        tname = self.tname
        level = draft.summary.get("release_level_quat")
        if not draft.summary.get("release_tilt_deg") or level is None or tname not in draft.goal_poses or is_roll(float(draft.summary["release_tilt_deg"])):
            return super().check_goal_upright(draft)  # a rolled release (the standing book) is measured against itself
        release = draft.goal_poses[tname]
        draft.goal_poses[tname] = (release[0], th.tensor([float(v) for v in level]))
        try:
            return super().check_goal_upright(draft)
        finally:
            draft.goal_poses[tname] = release

    def refine_targets_from_goal(self, draft: Draft):
        """IORSLarge's (settled pose -> targets, partner unchanged), then the drop test's protrusion rule (``inside`` itself is the BDDL
        goal the goal check evaluates next)."""
        super().refine_targets_from_goal(draft)
        p, q = self.target.get_position_orientation()
        pts = self.world.hull_at(self.hull, (p.float(), q.float()))
        t_lo, t_hi = pts.min(dim=0).values, pts.max(dim=0).values
        height = float(t_hi[2] - t_lo[2])
        frac = float(self.num("max_protrusion_fraction", MAX_PROTRUSION_FRACTION))
        ok, prot = protrusion_ok(float(t_hi[2]), float(self.place["rim_z"]), height, frac)
        draft.summary["inside"] = {"protrusion_m": round(prot, 4), "max_protrusion_m": round(frac * height, 4), "rim_z": round(float(self.place["rim_z"]), 4),
                                   "rim_source": self.place["surface"].get("rim_source"), "settled_bottom_above_surface_m": round(float(t_lo[2]) - float(self.place["z"]), 4),
                                   "surface_z": round(float(self.place["z"]), 4)}
        if not ok:
            raise Reject("does_not_fit_inside", f"target protrudes {prot:.3f} m above the rim {float(self.place['rim_z']):.3f} (> {frac:.2f} x height {height:.3f})")

    def after_goal_check(self, draft: Draft):
        """The release as recorded (the place is the release above the surface, the goal the drop settled to): pre-place / place re-checked
        from C with the object attached and the partner present; the grasp in the manipulation image, the release in the placement image;
        the release facts in ``grasps[0]`` and ``summary.release``."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place = self.sidecar.stance("placement")
        with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
            res = self.ikc.exact([grasp.pre_place, grasp.place], place, grasp.arm, ignore_for_last=[self.obj("floor")], cache=False)
        if not res.ok:
            raise Reject("place_unreachable", f"pre-place / release fail the exact IK from stance C at the goal check: {res.reason}")
        z_s = float(self.place["z"])
        clearance = float(self.num("release_clearance_m", RELEASE_CLEARANCE_M))
        low = float(gripper_corners_world(grasp.place)[:, 2].min())
        if low < z_s + 0.5 * clearance:
            raise Reject("release_too_low", f"open gripper lowest point {low:.3f} vs the surface {z_s:.3f} at the recorded release")
        rel = draft.goal_poses.get(tname)
        facts = {"place_kind": "drop", "surface_z": round(z_s, 4), "surface_z_above_floor": round(z_s - self.floor_z, 4),
                 "release_above_m": None if rel is None else round(self.release_above(rel[1], grasp.eef_in_object), 4), "finger_clearance_m": round(low - z_s, 4),
                 "drop_height_m": float(self.num("drop_height_m", DROP_HEIGHT_M)), "release_clearance_m": clearance,
                 "settled_goal_z": round(float(draft.pose_targets[tname]["position"][2]), 4)}
        if "release_tilt_deg" in draft.summary:
            facts["release_tilt_deg"] = draft.summary["release_tilt_deg"]
        ctx = getattr(self, "_shelf_ctx", None)
        if ctx is not None and rel is not None:
            # oracle-fix 2026-11-06 (inside_shelf_place): where the shelf place put the drawn release, for the validation of the settled pose
            _, _, moved = shelf_release_xy(self, draft.summary["goal_cell"], rel[1], grasp.eef_in_object)
            sp = ctx["facts"]
            spine = pinched_edge_normal(grasp.eef_in_object[0], self.rest)
            facts["shelf_place"] = {**{k: sp[k] for k in ("side", "side_rule", "spine_behind_edge_m", "wall_gap_m", "release_lean_deg", "jaw_clear_m", "jaw_sweep_m",
                                                          "board_u_m", "cell", "xy")},
                                    "release_xy": [round(float(rel[0][0]), 4), round(float(rel[0][1]), 4)], "moved_off_board_m": round(moved, 4),
                                    "spine_facing_out": None if spine is None else round(spine_facing(rel[1], spine[0], -G.t(ctx["inward"]).float()), 4)}
            log.info("%s: inside shelf place: the release on the %s board at %s (the upright pose %s moved %.3f m off the board for the open jaw), spine facing "
                     "out %s, lean %g deg", self.gp.variant, sp["side"], facts["shelf_place"]["release_xy"], sp["xy"], moved, facts["shelf_place"]["spine_facing_out"],
                     sp["release_lean_deg"])
        if self.release_lift_per_grasp() and rel is not None:
            facts.update({"release_lift_per_grasp": True, "release_above_max_m": round(self.release_above(rel[1]), 4)})
            log.info("%s: inside release at the drawn grasp %s's own lift: %.3f m above the surface (every library grasp: %.3f m)", self.gp.variant,
                     grasp.grasp_name, facts["release_above_m"], facts["release_above_max_m"])
        grasp.extras.update(facts)
        draft.summary["release"] = facts
        if not self.ikc.eef_in_frame(grasp.grasp, self.camera_pose_at(self.sidecar.stance(getattr(grasp, "stance", None) or "manipulation"))):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if not self.ikc.eef_in_frame(grasp.place, self.camera_pose_at(place)):
            raise Reject("gripper_not_in_frame", "release eef pose outside the placement image (5 %)")
