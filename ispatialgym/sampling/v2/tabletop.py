"""§4 tabletop initialization as functions on a ``SamplerV2`` (spec ``docs/instance_sampling_spec.md`` §4.2–4.7; README
"tabletop.py — signatures the reference task calls").

Phases (every rule number refers to the spec)::

    phase1_edge_and_standoff(s, top, rng, stand_off=0.45)      -> (EdgeFrame, Stance)   §4.2  any edge, s = 0.45 + overhang (>= 5 cm)
    phase1b_init_trunk(s, init, H_max)                          -> Stance               §4.3  Δ_aim / Δ_min / Δ_max -> t*, tilt 0; b, R_img
    phase1c_lateral(s, init, margins, rng)                      -> Stance               §4.4  margins table -> lateral draw, checks (1)–(3)
    phase2_manipulation_stance(s, init, target_half_extent)     -> list[Stance]         §4.5  stand-off x translate x continuous tilt, band cells,
                                                                                              focus per survivor, best angle class
    phase3_target(s, manip, rng, grasps, hull, rest, ignore)    -> (obj_pose, GraspRecord)  §4.6 step 1 (F at the focus, exact IK)

plus the shared helpers of §4.6 steps 2–6 (``place_partner``, the IORS start region ``partner_gap_masks``, ``place_distractors``,
``all_in_frame``, ``whole_body_check_at``, ``brace_second_arm``), the §4.7 ``placement_stance`` and the reach-band access of §4.1 / §3.10
(``BandSpec``, ``ensure_bands``: ``ReachBands.load`` on a cache hit, ``BandComputer`` in-process on a miss, written back to the
cache directory).

Camera constants come from the FK once per robot (``camera_fk``: pitch at tilt 0, x_c(t), z_c(t), hand height below the
camera) and are cached on the module; the ``[pilot]`` design values of §4.3 (22.6°, 0.22 m, 0.19 m) are the defaults of the
pure helpers and what the tests compare the FK against.

Pure helpers (no simulator, unit-tested in ``tests/test_v2_tabletop.py``): ``lateral_margins`` (the §4.4 table),
``init_trunk_delta`` (§4.3 clamp), ``CameraFK`` (conventions FK), ``cube_flip_plan`` / ``cube_tilt_pair`` /
``facing_coarse_yaw`` (§6.5 cube grasp and visible-face rules), ``distractor_spacing`` / ``exclude_lookalikes`` (§4.6 step 3,
§3.12), ``axial_init_yaw`` (§6.10), ``lateral_gap_masks`` / ``longest_run`` (§4.6 step 2 IORS start region). Everything that touches
the simulator imports ``ik`` / ``sim_utils`` lazily so the
module (and the tests) import without OmniGibson.

Frames: base frame x ahead, lateral + = robot's left (§3.3); band cells are base-frame (x, lateral) at *any* stance of
the same support height (the band is robot-relative, §4.1: "the stand-off is not a dimension").
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.fastflags import fast_on  # train-fast 2026-10-31: BAND_FEASIBLE_REUSE
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Pose, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

# ----------------------------------------------------------------------------------------------------------------
# constants (§4; the [pilot] numbers of §4.3 are the defaults of the pure helpers)
# ----------------------------------------------------------------------------------------------------------------

STAND_OFF = 0.45  # §4.2
STAND_OFFS = (0.35, 0.45, 0.55)  # §4.5
TRANSLATES = tuple(float(t) for t in RB.TRANSLATES)  # §3.3 grid (0.95 = deepest trunk)
TILT_GRID = tuple(float(k) for k in RB.TILTS)  # §3.3 grid tilts of the bands
N_TILT_SAMPLES = 7  # §4.5 "tilt continuous from 0 to the torso_joint3 limit": samples of the scan
ANGLE_CLASSES_DEG = (20.0, 30.0)  # §4.5 step 3 (owner 2026-10-11: 8 / 20 left the target almost no spread)
SPLIT_INIT_VIEW_CLASSES_DEG = (20.0, 30.0)  # split cone (generation.phase2_split_cone): the start position's classes at the init view
SPLIT_LATERAL_SHIFTS = (0.0, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30)  # split cone: work-stance shifts along the edge (attachment's second base pose)


def phase2_gen(s, key: str, default=None):
    """``generation.<key>`` of the variant's customs entry for sampler ``s``, else of its subtask's generation_defaults, else
    ``default`` - so a subtask's Phase-2 cone is set in one place and a variant can still override it."""
    v = s.gen(key, None)
    if v is not None:
        return v
    from ispatialgym.sampling import generation_defaults as GD

    try:
        sub = GD.defaults_for(s.subtask)
    except (KeyError, AttributeError):
        sub = {}
    v = sub.get(key)
    return default if v is None else v


def phase2_angle_classes(s, default: Sequence[float] = ANGLE_CLASSES_DEG) -> Tuple[float, float]:
    """Phase 2's (tight, loose) angle classes for sampler ``s``: ``generation.phase2_tight_class_deg`` / ``phase2_loose_class_deg``
    through :func:`phase2_gen`, else ``default``."""
    return (float(phase2_gen(s, "phase2_tight_class_deg", default[0])), float(phase2_gen(s, "phase2_loose_class_deg", default[1])))


IN_FRAME_MARGIN = 0.05  # §3.4 "in frame (5 %)"
FOCUS_RADIUS = 0.05  # §4.6 step 1: target within 5 cm of the focus
YAW_JITTER = math.radians(22.5)  # §4.6 step 1
YAWS = tuple(RB.YAWS)  # §4.6 step 1: 8 fixed yaws (45° steps)
SUPPORT_MARGIN = 0.02  # §3.2 footprint inside the top shrunk by 2 cm
TRAY_MARGIN = 0.005  # §3.2 0.5 cm on a tray floor
MIN_GAP = 0.03  # §3.2 footprint clearance target–partner
TABLE_END_MARGIN = 0.10  # §4.4 IORS (A1, owner 2026-10-12; generation.table_end_margin_m): target / partner footprints this far from the edge's ends
START_REGION_GAP_STEP = 0.01  # §4.6 step 2 IORS start region: the gap grid of a cell's interval
OVERHANG_MIN = 0.05  # §4.2 AABB overhang beyond the raster edge counts when >= 5 cm
HANDS_ABOVE = 0.10  # §4.3 hands >= 10 cm above the tallest object
HAND_BELOW_CAMERA_PILOT = 0.19  # §4.3 [pilot]
CAMERA_X_PILOT = 0.22  # §4.3 [pilot]
CAMERA_PITCH_PILOT = math.radians(22.6)  # §4.3 [pilot]
MAX_TRANSLATE = 0.95  # §3.3: translate 1.0 self-collides, 0.95 is the deepest trunk everywhere
TRUNK_SWEEP_TRANSLATES = TRANSLATES  # §4.4 check (3)
GENERIC_YAWS = 2  # §4.1 generic pinch: 2 wrist yaws x 2 approach pitches
GENERIC_TILTS_DEG = (0.0, 30.0)
Z_OFFSET = 0.003  # §3.2 z = support top + bottom offset + 3 mm
PLACE_DROP = 0.003


def _reject(reason: str, message: str = "", **details):
    from ispatialgym.sampling.sim_utils import Reject

    raise Reject(reason, message, **details)


def _ik():
    from ispatialgym.sampling.v2 import ik as IK

    return IK


def _top(s, top=None):
    """The support top a sampler works on (``s.top`` as the reference task keeps it, or the one Phase 1a recorded)."""
    if top is not None:
        return top
    t = getattr(s, "top", None)
    if t is None:
        t = getattr(s, "_tabletop_top", None)
    if t is None:
        raise ValueError("no support top: call phase1_edge_and_standoff(s, top, ...) first or set s.top")
    return t


def _t(x) -> th.Tensor:
    return x if isinstance(x, th.Tensor) else th.tensor(x, dtype=th.float32)


#: the selection's surface height and the support top's differ by more than this -> the sampler works on the interior surface at
#: the selection's height (owner 2026-10-11 item 2)
SURFACE_MISMATCH_M = 0.03
#: headroom an interior cell needs above the tallest object (``reach_bands.INTERIOR_HEADROOM_MARGIN``)
INTERIOR_HEADROOM_MARGIN_M = 0.02


def support_surface_spec(s, support) -> Optional[Dict[str, Any]]:
    """The surface the selection names for ``support`` in this scene: the first ``scenes.<scene>.bindings.support_surfaces``
    entry of its name (``ground_isg.support_surface``: ``{"name", "init", "top_z", "row"[, "surface": "top"]}``), else None."""
    name = getattr(support, "name", support)
    req = s.choice.bindings.get("support_surfaces")
    for e in req if isinstance(req, list) else []:
        if isinstance(e, dict) and e.get("name") == name:
            return e
    return None


def support_surface(s, support, ignore: Sequence[Any] = (), height: Optional[float] = None):
    """The raster the sampler works on for ``support`` (owner 2026-10-11 item 2): ``SimWorld.support_interior`` at the selection's
    ``top_z`` (m above the floor) when its row says ``init=inside`` or that height is more than :data:`SURFACE_MISMATCH_M` from the
    support top's, else the support top (rows marked ``surface: top`` always). On an interior surface the cells whose headroom
    is below ``height`` + :data:`INTERIOR_HEADROOM_MARGIN_M` count as occupied, and ``s.surface_z`` (the selection's height) makes
    the band lookups ask for the interior band (``ReachBands.load(..., surface_z=...)``); on the top ``s.surface_z`` is None.
    The decision goes to ``template_facts["support_surface"]``. A row naming ``neighbors`` (owner 2026-10-12 A4) gets the support
    top united with theirs (``neighbors.union_top``; never an interior surface)."""
    spec = support_surface_spec(s, support)
    s.surface_z = None
    top, top_err = None, None
    try:
        top = s.support_top(support, ignore=ignore)
    except (RuntimeError, ValueError) as e:
        top_err = str(e)
    floor_z = float(s.world.floor_z)
    fact: Dict[str, Any] = {"support": support.name, "spec": spec, "top_z_above_floor": round(float(top.z) - floor_z, 4) if top is not None else None,
                            "top_error": top_err}
    interior = False
    if spec is not None and spec.get("top_z") is not None and spec.get("surface") != "top":
        if spec.get("init") == "inside":
            interior, fact["why"] = True, "init=inside"
        elif top is None or int(top.free.sum()) == 0:
            interior, fact["why"] = True, "no support top raster"
        elif abs(float(spec["top_z"]) - (float(top.z) - floor_z)) > SURFACE_MISMATCH_M:
            interior, fact["why"] = True, f"top_z {float(spec['top_z']):.2f} vs support top {float(top.z) - floor_z:.2f} (> {SURFACE_MISMATCH_M} m)"
    if not interior:
        fact["surface"] = "top"
        if spec is not None and spec.get("surface") == "top":
            fact["why"] = spec.get("why") or "surface: top"
        tf = getattr(s, "template_facts", None)
        if isinstance(tf, dict):
            tf["support_surface"] = fact
        if top is None:
            _reject("support_small", f"{support.name}: no support top ({top_err})")
        return NB.union_top(s, support, top, spec, ignore=ignore)
    z_req = float(spec["top_z"])
    try:
        itop = s.world.support_interior(support, floor_z + z_req, resolution=float(s.gen("support_resolution", 0.02)),
                                        ignore=list(ignore) + list(s.world.spawned.values()))
    except RuntimeError as e:
        _reject("support_surface", f"{support.name}: interior surface at {z_req:.2f} m: {e}")
    if height is not None and getattr(itop, "headroom", None) is not None:
        low = itop.mask & (itop.headroom < float(height) + INTERIOR_HEADROOM_MARGIN_M)
        itop.occupied = itop.occupied | low
        fact.update({"height_m": round(float(height), 4), "n_low_headroom_cells": int(low.sum())})
    s.surface_z = z_req
    fact.update({"surface": "interior", "z_world": round(float(itop.z), 4), "z_above_floor": round(float(itop.z) - floor_z, 4),
                 "open_edges": list(getattr(itop, "open_edges", []) or []), "n_board_cells": int(itop.mask.sum()), "n_free_cells": int(itop.free.sum())})
    log.info("%s: interior surface of %s at %.2f m above the floor (%s): %d free cells, open sides %s", s.gp.variant, support.name, z_req, fact.get("why"),
             fact["n_free_cells"], fact["open_edges"] or "none")
    tf = getattr(s, "template_facts", None)
    if isinstance(tf, dict):
        tf["support_surface"] = fact
    return NB.union_top(s, support, itop, spec, ignore=ignore)


#: upright shelf start (owner 2026-10-11 item 3, touching#03): the spine this far behind the shelf's front edge, the leaning edge
#: this far from the side board before the settle (gravity closes the gap, the book is never held up artificially)
SHELF_SPINE_BEHIND_EDGE_M = 0.02
SHELF_WALL_GAP_M = 0.002
#: a side-board ray hit nearer than this to the ray's start is the body the ray starts in (the edge midpoint inside a divider of a
#: multi-bay compartment), not a board face - ``SimWorld.support_interior`` drops such hits of its rays the same way
SHELF_RAY_MIN_HIT_M = 1e-3


def upright_shelf_facts(grasp_key: Optional[str], rest_quat) -> Optional[Dict[str, Any]]:
    """Facts of an upright-shelf grasp key (library ``rest_pose: upright_shelf``, e.g. ``hardback__acbrnv__bu``): the book stands
    on its bottom edge leaning against a wall in its rest orientation, its spine toward the approach (rest-frame -x). ``lean_side``
    = +1 when the top edge leans toward rest-frame +y (with the spine toward the robot: the robot's left), -1 toward -y;
    ``lean_deg`` from the rest orientation. None for any other key."""
    from ispatialgym.sampling import grasp_library as GL

    if not grasp_key:
        return None
    try:
        meta = (GL._candidates(grasp_key) or {}).get("ispatialgym") or {}
    except (FileNotFoundError, ValueError):
        return None
    if meta.get("rest_pose") != "upright_shelf":
        return None
    R = C.transform_points((th.zeros(3), _t(rest_quat)), th.eye(3))  # rows: the local axes in the rest frame
    a = int(th.argmax(R[:, 2].abs()))
    up = R[a] * (1.0 if float(R[a, 2]) > 0 else -1.0)  # the book's bottom -> top direction
    return {"key": grasp_key, "lean_side": 1.0 if float(up[1]) > 0 else -1.0, "lean_deg": round(math.degrees(math.asin(min(1.0, float(th.linalg.norm(up[:2]))))), 2),
            "context": meta.get("context")}


def shelf_wall_lateral(s, edge: EdgeFrame, depth: float, z: float, side: float, support, lateral0: float = 0.0) -> Tuple[Optional[float], Dict[str, Any]]:
    """Lateral position (m along ``edge.tangent`` from ``edge.mid``, + = the robot's left) of the inner face of the support's side
    board on the robot's ``side``: a horizontal ray at height ``z``, ``depth`` behind the edge line, from the edge midpoint (shifted
    ``lateral0`` along the tangent: the compartment of ``generation.shelf_bay_lateral_m``, oracle-fix 2026-11-04 round 3) outward.
    None when it meets nothing, another object first (a room wall behind a plank), or a body it starts in (a hit within
    :data:`SHELF_RAY_MIN_HIT_M` of the start: the midpoint inside a divider)."""
    start = edge.mid + edge.inward * float(depth) + edge.tangent * float(lateral0)
    end = start + edge.tangent * float(side) * (0.5 * float(edge.length) + 0.30)
    h = s.world.raycast(th.tensor([[float(start[0]), float(start[1]), float(z)]]), th.tensor([[float(end[0]), float(end[1]), float(z)]]))[0]
    paths = tuple(str(l.prim_path) for l in support.links.values())
    fact = {"hit": bool(h.get("hit")), "body": h.get("rigidBody"), "distance": h.get("distance")}
    if h.get("hit") and float(h.get("distance", 1.0)) < SHELF_RAY_MIN_HIT_M:
        fact["why"] = "the ray starts inside the body it hits"
        return None, fact
    if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(paths):
        return None, fact
    return float(lateral0) + float(side) * float(h["distance"]), fact


def shelf_lean_pose(s, hull: th.Tensor, rest_quat, top, edge: EdgeFrame, support, lean_side: float, *, wall_gap: Optional[float] = None,
                    spine_behind: Optional[float] = None, lateral0: Optional[float] = None) -> Tuple[Pose, Dict[str, Any]]:
    """The upright shelf start (owner 2026-10-11 item 3): the object in its rest orientation turned to the edge's yaw (spine toward
    the robot), its spine :data:`SHELF_SPINE_BEHIND_EDGE_M` behind the board's front edge (the raster edge line + half a cell), its
    leaning edge :data:`SHELF_WALL_GAP_M` short of the side board on the ``lean_side`` (:func:`shelf_wall_lateral`), resting on
    the board; gravity does the rest. ``Reject("shelf_wall")`` without a side board there, ``Reject("shelf_board")`` when the
    bottom edge is off the board. oracle-fix 2026-11-06 (inside_large#01, ``iors_large_inside``'s shelf place): ``wall_gap`` /
    ``spine_behind`` (m) / ``lateral0`` (m along the edge tangent from ``edge.mid``: where the side-board ray starts) replace
    :data:`SHELF_WALL_GAP_M` and the ``generation.shelf_spine_behind_edge_m`` / ``shelf_bay_lateral_m`` lookups when given."""
    pts = C.transform_points((th.zeros(3), _t(rest_quat)), hull)
    spine = -float(pts[:, 0].min())
    lat_ext = float(pts[:, 1].max()) if lean_side > 0 else -float(pts[:, 1].min())
    height = float(pts[:, 2].max() - pts[:, 2].min())
    # oracle-fix 2026-11-04 round 3 (touching#03): ``generation.shelf_spine_behind_edge_m`` (the spine 1 cm behind the edge left the book's
    # bottom-front corners under the start camera's frame border at every kmuacd_0 cell) and ``generation.shelf_bay_lateral_m`` {support: lateral} (the compartment)
    gen = getattr(s, "gen", None)
    if spine_behind is None:
        spine_behind = float(gen("shelf_spine_behind_edge_m", SHELF_SPINE_BEHIND_EDGE_M)) if callable(gen) else SHELF_SPINE_BEHIND_EDGE_M
    if lateral0 is None:
        bays = gen("shelf_bay_lateral_m") if callable(gen) else None  # {support name: lateral of the compartment's centre, m along the edge tangent}
        bay = float((bays or {}).get(getattr(support, "name", ""), 0.0)) if isinstance(bays, dict) else 0.0
    else:
        bay = float(lateral0)
    gap = SHELF_WALL_GAP_M if wall_gap is None else float(wall_gap)
    depth = -0.5 * float(top.resolution) + spine_behind + spine
    z_ray = float(top.z) + 0.8 * height
    wall, ray = shelf_wall_lateral(s, edge, depth, z_ray, lean_side, support, bay)
    facts: Dict[str, Any] = {"edge": edge.name, "lean_side": "left" if lean_side > 0 else "right", "depth_m": round(depth, 4), "ray_z": round(z_ray, 4), "ray": ray}
    if wall is None:
        _reject("shelf_wall", f"{support.name}: no side board on the {facts['lean_side']} at {z_ray:.2f} m ({ray})")
    lat = wall - lean_side * (lat_ext + gap)
    xy = edge.mid + edge.inward * depth + edge.tangent * lat
    pose = object_pose_on(s, hull, rest_quat, xy, edge.yaw, float(top.z))
    world = C.transform_points(pose, hull)
    bottom = world[world[:, 2] <= float(world[:, 2].min()) + 0.005]
    off = [p for p in bottom if not top.contains(p[:2])]
    facts.update({"wall_lateral_m": round(wall, 4), "lateral_m": round(lat, 4), "position": [round(float(v), 4) for v in pose[0]],
                  "n_bottom_points": len(bottom), "n_bottom_off_board": len(off)})
    if off:
        _reject("shelf_board", f"{support.name}: {len(off)}/{len(bottom)} bottom-edge points of the leaning start are off the board ({facts})")
    return pose, facts


def upright_lean_quat(hull: th.Tensor, lean_side: float, lean_deg: float) -> th.Tensor:
    """A book's upright leaning rest orientation from its local hull (owner 2026-10-11, the touching#03 partner, which has no
    library key): the longest local axis vertical, the middle one along rest-frame +x (into the shelf) and the shortest across,
    tilted ``lean_deg`` about +x so that the top leans toward rest-frame +y (``lean_side`` +1, the robot's left with yaw = the
    edge's) or -y (-1)."""
    ext = hull.max(dim=0).values - hull.min(dim=0).values
    order = [int(i) for i in th.argsort(ext)]  # thickness, width, height
    e = th.eye(3, dtype=th.float64)
    e_w, e_h = e[order[1]], e[order[2]]
    R0 = th.stack([e_w, th.linalg.cross(e_h, e_w), e_h], dim=0)  # rows: local width -> +x, local (height x width) -> +y, height -> +z
    phi = -float(lean_side) * math.radians(float(lean_deg))
    Rx = th.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(phi), -math.sin(phi)], [0.0, math.sin(phi), math.cos(phi)]], dtype=th.float64)
    return C.mat_to_quat(Rx @ R0).float()


def releaned_quat(rest_quat, lean_side0: float, lean_deg0: float, lean_side: float, lean_deg: float) -> th.Tensor:
    """oracle-fix 2026-11-06 (touching#03, ``generation.shelf_start_draw``): an upright book's rest orientation ``rest_quat`` that leans
    ``lean_deg0`` toward ``lean_side0`` (:func:`upright_shelf_facts`) turned about rest-frame +x (spine -> into the shelf) until it leans
    ``lean_deg`` toward ``lean_side`` (+1 = rest-frame +y, the robot's left; :func:`upright_lean_quat`'s sign) - the spine keeps facing
    the robot and the grasp library rows stay where they are on the book (object frame); the exact IK judges them at the new pose."""
    half = 0.5 * math.radians(-float(lean_side) * float(lean_deg) + float(lean_side0) * float(lean_deg0))
    return C.qmul(th.tensor([math.sin(half), 0.0, 0.0, math.cos(half)]), _t(rest_quat).float()).float()


def mirrored_lean_quat(quat, edge: EdgeFrame, lean_side: float, lean_deg: float) -> th.Tensor:
    """``quat`` (an upright book leaning ``lean_deg`` toward ``lean_side`` of ``edge``) turned about the edge's inward axis so it
    leans the same angle to the other side (the touching#03 goal: the target rests against the partner, parallel to it). A book
    leaning ``a`` deg ends leaning ``b`` deg the other way with ``lean_deg`` = (a + b) / 2 (the turn is 2 ``lean_deg``)."""
    axis = th.tensor([float(edge.inward[0]), float(edge.inward[1]), 0.0])
    axis = axis / th.linalg.norm(axis)
    half = 0.5 * 2.0 * float(lean_side) * math.radians(float(lean_deg))
    turn = th.tensor([*(axis * math.sin(half)).tolist(), math.cos(half)])  # xyzw
    return C.qmul(turn, _t(quat))


def grasp_at_pose(s, manip: Stance, pose: Pose, grasps: Sequence[Any], ignore_for_grasp: Sequence[Any], yaw: float, *, arm: str = "right",
                  scale_key: Any = None, object_name: Optional[str] = None, fingers: Optional[float] = None,
                  accept: Optional[Callable[[GraspRecord], Optional[str]]] = None, prescreen: bool = False) -> GraspRecord:
    """The first library grasp of ``grasps`` whose pre-grasp and grasp at the fixed object ``pose`` pass the exact IK from ``manip``
    with ``arm`` (the upright shelf start: the pose is set by the shelf, not drawn from F). The library pinch serves either hand
    (the two grippers and their wrist cameras are alike in the eef frame); with ``generation.grasp_twins`` it is offered in both
    turns by the orientation rule (:func:`pick_twin`, as the segment fallback takes a pinch with the other arm), the wrist camera
    checked against the world only - the board under the object is no infinite plane, a camera in front of its edge is clear.
    ``accept(record) -> why or None`` (oracle-fix 2026-11-05, speed): a further check of each passing grasp; one it refuses is skipped
    for the next. ``prescreen`` (oracle-fix 2026-11-05, speed): every row's [pre-grasp, grasp] and its twin's in one batched screen
    (``IKChecker.screen_pairs``, the feasible-set screen) first - a row whose row and twin both fail it gets no exact IK (touching#03:
    83 % of the attempt time in this function's exact IK, most rows failing at the pre-grasp). ``Reject("grasp_unreachable")`` otherwise."""
    IK = _ik()
    name = object_name or s.bindings.one("target").name
    exact_kw = {"ignore_for_last": list(ignore_for_grasp), "scale_key": scale_key}
    if fingers is not None:  # oracle-fix 2026-11-04 round 3: the pre-shaped opening the oracle approaches with (touching#03's side-board finger)
        exact_kw["fingers"] = float(fingers)
    twins = grasp_twins_on(s)
    reasons = []
    screened = None
    if prescreen and grasps:
        pairs = []
        for g in grasps:
            eef = IK.grasp_pose(pose, g.pose)
            pairs.append([IK.pre_pose(eef), eef])
            if twins and getattr(g, "name", None):
                eef_t = C.twin_pose((_t(eef[0]), _t(eef[1])))
                pairs.append([IK.pre_pose(eef_t), eef_t])
        ok = s.ikc.screen_pairs(pairs, manip, arm, ignore_for_last=list(ignore_for_grasp), fingers=fingers).tolist()
        screened, i = [], 0
        for g in grasps:
            n = 2 if (twins and getattr(g, "name", None)) else 1
            screened.append(any(ok[i:i + n]))
            i += n
    for k, g in enumerate(grasps):
        if screened is not None and not screened[k]:
            _count(s, "fixed_start_prescreen.dropped")
            reasons.append(f"{getattr(g, 'name', '?')}: neither the row nor its twin passes the batched screen")
            continue
        eio = g.pose
        eef = IK.grasp_pose(pose, eio)
        pre = IK.pre_pose(eef)
        res = s.ikc.exact([pre, eef], manip, arm, **exact_kw)
        pick = None
        if twins and getattr(g, "name", None):
            pick = pick_twin(s, manip, arm, g, eio, eef, pre, res, exact_kw, where="fixed start")
            g, eio, eef, pre, res = pick.grasp, pick.eef_in_object, pick.eef, pick.pre, pick.res
        if not (pick.passed if pick is not None else res.ok):
            reasons.append(f"{getattr(g, 'name', '?')}: {res.reason}"[:120])
            continue
        rec = GraspRecord(name, arm, getattr(g, "key", None), getattr(g, "name", None), float(yaw), pre, eef,
                          finger_q=list(getattr(g, "finger_q", []) or []) or None, eef_in_object=eio)
        if pick is not None:
            rec.extras.update({"wrist_flip": pick.twin, "twin_choice": pick.facts})
        rec.stance = manip.name
        why = accept(rec) if accept is not None else None
        if why is not None:
            reasons.append(f"{getattr(g, 'name', '?')}: {why}"[:160])
            continue
        return rec
    _reject("grasp_unreachable", f"no library grasp reaches the fixed start pose from {manip.name}: {reasons[:4]}")


def surface_band_kwargs(s) -> Dict[str, Any]:
    """``{"surface_z": ...}`` for ``ReachBands.load`` / ``find`` while the sampler works on an interior surface, else ``{}``."""
    z = getattr(s, "surface_z", None)
    return {"surface_z": float(z)} if z is not None else {}


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §4.3 trunk solve, §4.4 margins table
# ----------------------------------------------------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    """``clamp(x, lo, hi) = min(max(x, lo), hi)``: when ``lo > hi`` the upper bound wins (torch / numpy semantics; §4.3
    ``Δ = clamp(Δ_aim, Δ_min, Δ_max)``: the near edge stays in frame, the hand clearance is then checked by the
    whole-body test of §4.4)."""
    return min(max(float(x), float(lo)), float(hi))


def init_trunk_delta(edge_depth: float, d_ws: float, d_near: float, x_c: float, H_max: float, hand_below: float = HAND_BELOW_CAMERA_PILOT,
                     pitch: float = CAMERA_PITCH_PILOT, phi_m: float = math.radians(46.6), hands_above: float = HANDS_ABOVE) -> Dict[str, float]:
    """§4.3: the camera height above the top ``Δ`` for the init trunk (tilt 0).

    ``edge_depth`` = distance base centre -> front edge (s + overhang), ``d_ws`` / ``d_near`` = depth from the edge of
    the workspace centre / near edge, ``x_c`` = camera ahead of the base, ``H_max`` = tallest object to be placed,
    ``hand_below`` = untucked hands below the camera, ``pitch`` = camera pitch down at tilt 0, ``phi_m`` = image-border
    angle including the visibility margin (half the vertical FOV minus the margin)::

        Δ_aim = edge_depth + d_ws − x_c                    45° elevation onto the workspace centre
        Δ_min = H_max + hands_above + hand_below           hands >= 10 cm above the tallest object
        Δ_max = (edge_depth + d_near − x_c) · tan(pitch + φ_m)   workspace near edge inside the frame
        Δ     = clamp(Δ_aim, Δ_min, Δ_max)
    """
    d_aim = float(edge_depth) + float(d_ws) - float(x_c)
    d_min = float(H_max) + float(hands_above) + float(hand_below)
    d_max = (float(edge_depth) + float(d_near) - float(x_c)) * math.tan(float(pitch) + float(phi_m))
    return {"delta": clamp(d_aim, d_min, d_max), "delta_aim": d_aim, "delta_min": d_min, "delta_max": d_max, "conflict": d_min > d_max}


def lateral_margins(margins: Dict[str, Any], lam_min: float, lam_max: float, b: float, R_img: float) -> Tuple[float, float, Dict[str, Any]]:
    """§4.4 margin table: ``(m_R, m_L, info)`` for the task row named by ``margins["kind"]`` (a row without ``kind`` but
    with ``relation`` is the IORS row the reference task passes: ``w_t, w_p, W_p, D_p_max, safety, relation, g_NT``).

    ``lam_min <= 0`` (right end of the band union), ``lam_max`` (left end), ``b`` = lateral offset of the image border at
    the source-tray depth, ``R_img`` = surface distance from the target (centre line) to the image border. Rows::

        iors (ontop, under, inside, nextto, touching; A1, owner 2026-10-12): 0 / 0 - the lateral is drawn over the whole edge, which
                                                       must be at least info["L_min"] = 2·w_t + g_min + 2·w_p + 2·m_end long (g_min = 0.03,
                                                       nextto g_NT + 0.03; m_end = generation.table_end_margin_m); D_p = min(W_p + 0.03,
                                                       R_img); nextto: D_p < g_NT + 0.03 -> info["reject"]
        attachment:        |λ_min| + max(w_target, w_partner) + 0.02 / max(0, λ_max) + max(w_target, w_partner) + 0.02
        axial:             |λ_min| + w_diag + 0.02 / max(0, λ_max) + w_diag + 0.02,  w_diag = ½√(long² + short²)
        block_design:      |λ_min| + edge/2 + (edge + 0.001) + 0.02 / max(0, λ_max) + edge/2 + 0.02   (0.0125 + 0.026 + 0.02 for the 2.5 cm cube)
        disembedding:      the ``destination_side`` b + lateral_margin + 0.02 (lateral_margin = 0.437 [pilot]), the other side only the
                           source tray: |λ_min| (right) / max(0, λ_max) (left) + ``source_half_length`` + 0.02 (owner 2026-10-12 A2a);
                           without ``destination_side`` both sides b + lateral_margin + 0.02
        articulation_small (§5.4): |λ_min| + w_a + 0.02 / max(0, λ_max) + w_a + 0.02
    """
    kind = margins.get("kind") or ("iors" if "relation" in margins else None)
    if kind is None:
        raise ValueError(f"margins row needs a 'kind' (or 'relation' for the IORS row): {sorted(margins)}")
    safety = float(margins.get("safety", 0.02))
    right = abs(min(float(lam_min), 0.0))
    left = max(0.0, float(lam_max))
    info: Dict[str, Any] = {"kind": kind, "lam_min": float(lam_min), "lam_max": float(lam_max), "b": float(b), "R_img": float(R_img), "safety": safety}
    if kind == "iors":
        w_t, w_p = float(margins["w_t"]), float(margins["w_p"])
        D_p_max = float(margins.get("D_p_max", float(margins.get("W_p", 2 * w_p)) + 0.03))
        D_p = min(D_p_max, float(R_img))
        info["D_p"] = D_p
        g_min = float(margins.get("g_min", MIN_GAP))
        if margins.get("relation") == "nextto" and margins.get("g_NT") is not None:
            g_lo = float(margins["g_NT"]) + float(margins.get("nextto_gap_extra", 0.03))
            info["g_NT_lo"] = g_lo
            g_min = g_lo
            if D_p < g_lo:
                info["reject"] = ("pair_gap", f"D_p {D_p:.3f} < g_NT + {float(margins.get('nextto_gap_extra', 0.03)):.2f} = {g_lo:.3f}")
        m_end = float(margins.get("m_end", TABLE_END_MARGIN))
        info.update({"g_min": g_min, "m_end": m_end, "L_min": 2 * w_t + g_min + 2 * w_p + 2 * m_end})
        m_R = m_L = 0.0
    elif kind == "attachment":
        w = max(float(margins["w_target"]), float(margins["w_partner"]))
        m_R, m_L = right + w + safety, left + w + safety
    elif kind == "axial":
        if "w_diag" in margins:
            w_diag = float(margins["w_diag"])
        else:
            w_diag = 0.5 * math.hypot(float(margins["long"]), float(margins["short"]))
        info["w_diag"] = w_diag
        m_R, m_L = right + w_diag + safety, left + w_diag + safety
    elif kind == "block_design":
        edge = float(margins.get("edge", C.BLOCK_EDGE))
        pitch = edge + float(margins.get("block_gap", 0.001))
        info.update({"half_edge": edge / 2, "pitch": pitch})
        m_R = right + edge / 2 + pitch + safety  # half edge + one pitch to the robot's right of the anchor
        m_L = left + edge / 2 + safety
    elif kind == "disembedding":
        lm = float(margins.get("lateral_margin", 0.437))
        m_R = m_L = float(b) + lm + safety
        side = margins.get("destination_side")
        if side is not None:
            if side not in ("left", "right"):
                raise ValueError(f"disembedding margins: destination_side must be 'left' or 'right', got {side!r}")
            half = float(margins["source_half_length"])
            info.update({"destination_side": side, "source_half_length": half})
            if side == "right":
                m_L = left + half + safety
            else:
                m_R = right + half + safety
    elif kind == "articulation_small":
        w_a = float(margins["w_a"])
        m_R, m_L = right + w_a + safety, left + w_a + safety
    else:
        raise ValueError(f"unknown margins row {kind!r}")
    info.update({"m_R": m_R, "m_L": m_L})
    return m_R, m_L, info


def lateral_bounds(edge_length: float, m_R: float, m_L: float) -> Optional[Tuple[float, float]]:
    """§4.4 step 2: ``[−L/2 + m_R, L/2 − m_L]`` or None when ``m_R + m_L > L`` (skip the edge)."""
    lo, hi = -0.5 * float(edge_length) + float(m_R), 0.5 * float(edge_length) - float(m_L)
    return (lo, hi) if lo <= hi else None


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: camera FK tables (§3.3 "from FK once per robot")
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CameraFK:
    """Camera offsets at tilt 0 as functions of the JoyLo translate: ``x_c(t)`` ahead of the base, ``z_c(t)`` above the
    floor (base_link at ``NOMINAL_BASE_Z``), the constant pitch down, and the untucked hand height below the camera."""

    ts: np.ndarray
    x_c: np.ndarray
    z_c: np.ndarray
    pitch: float
    hand_below: float = HAND_BELOW_CAMERA_PILOT
    fy: float = 459.0
    height_px: int = 1080

    def x_at(self, t: float) -> float:
        return float(np.interp(clamp(t, 0.0, 1.0), self.ts, self.x_c))

    def z_at(self, t: float) -> float:
        return float(np.interp(clamp(t, 0.0, 1.0), self.ts, self.z_c))

    def t_of_z(self, z: float, t_max: float = MAX_TRANSLATE) -> float:
        """``z_c⁻¹``: the translate whose camera height is ``z`` (monotone decreasing table), clamped to ``[0, t_max]``."""
        zs = self.z_c[::-1]
        ts = self.ts[::-1]
        if z >= float(zs[-1]):
            return 0.0
        if z <= float(zs[0]):
            return float(t_max)
        return clamp(float(np.interp(z, zs, ts)), 0.0, t_max)

    def phi_m(self, margin: float = IN_FRAME_MARGIN) -> float:
        """Image-border angle including the visibility margin: ``atan((H/2 − m·H) / fy)`` (§4.3 φ_m)."""
        return math.atan((0.5 * self.height_px - margin * self.height_px) / self.fy)

    def tilt_limit(self, translate: float) -> float:
        """Most negative tilt (camera down) the ``torso_joint3`` limit allows at ``translate`` (§4.5 scan range)."""
        j = C.M.TRUNK_TILT_JOINT_INDEX
        return float(C.TORSO_JOINT_LIMITS[0][j] - C.translate_to_torso_q(translate)[j])


def camera_table(n: int = 101) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(ts, x_c, z_c, pitch)`` at tilt 0 from the conventions FK (base_link at ``NOMINAL_BASE_Z`` above the floor)."""
    ts = np.linspace(0.0, 1.0, n)
    xs, zs, pitches = [], [], []
    for t in ts:
        pos, quat = C.camera_pose(C.torso_q(float(t), 0.0), base_pose=(0.0, 0.0, 0.0))
        axis = C.quat_rotate(quat, th.tensor([0.0, 0.0, -1.0]))
        xs.append(float(pos[0]))
        zs.append(float(pos[2]))
        pitches.append(math.atan2(-float(axis[2]), math.hypot(float(axis[0]), float(axis[1]))))
    return ts, np.asarray(xs), np.asarray(zs), float(np.mean(pitches))


_FK_CACHE: Dict[Any, CameraFK] = {}


def camera_fk(s=None, hand_below: Optional[float] = None) -> CameraFK:
    """The camera FK tables, once per robot (cached on the kit); without a sampler the pure conventions tables with the
    ``[pilot]`` hand height."""
    key = id(s.kit) if s is not None else None
    if key in _FK_CACHE:
        return _FK_CACHE[key]
    ts, xs, zs, pitch = camera_table()
    fk = CameraFK(ts, xs, zs, pitch)
    if s is not None:
        cm = s.world.camera_model
        fk.fy, fk.height_px = float(cm.fy), int(cm.height)
        if hand_below is None:
            try:
                q = s.kit.q_config(arms="untucked", translate=0.5, tilt=0.0)
                z_cam = float(s.kit.camera_pose(q)[0][2])
                z_hand = max(float(s.kit.eef_pose(q, arm)[0][2]) for arm in C.M.ARMS)
                hand_below = z_cam - z_hand
            except Exception as e:  # noqa: BLE001 - the pilot value stands in when the kit cannot answer
                log.warning("camera_fk: hand height from the kit failed (%r); using the [pilot] value %.2f", e, HAND_BELOW_CAMERA_PILOT)
                hand_below = HAND_BELOW_CAMERA_PILOT
        fk.hand_below = float(hand_below)
        log.info("camera FK: pitch %.1f deg, x_c(0.5) %.3f m, z_c(0..1) %.3f..%.3f m, hands %.3f m below the camera", math.degrees(fk.pitch),
                 fk.x_at(0.5), fk.z_at(0.0), fk.z_at(1.0), fk.hand_below)
    elif hand_below is not None:
        fk.hand_below = float(hand_below)
    _FK_CACHE[key] = fk
    return fk


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §6.5 cube grasp rule and visible-face rule
# ----------------------------------------------------------------------------------------------------------------

FACE_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}


def opposite_face(face: str) -> str:
    return ("-" if face[0] == "+" else "+") + face[1]


def face_normal_world(face: str, quat) -> th.Tensor:
    return C.quat_rotate(quat, th.tensor(FACE_AXES[face]))


def up_face(quat) -> str:
    return max(FACE_AXES, key=lambda f: float(face_normal_world(f, quat)[2]))


def cube_tilt_pair(theta_place_deg: float, flip_sign: int) -> Tuple[float, float]:
    """§6.5: ``(tilt_pick, tilt_place)`` of a flipping block — the in-hand transform is rigid, so
    ``d_pick = R_a(∓90°)·d_place``: ``tilt_pick = tilt_place − flip_sign·90`` (conventions ``cube_pick_tilt``)."""
    return C.cube_pick_tilt(theta_place_deg, flip_sign), float(theta_place_deg)


def cube_flip_plan(block_quat, goal_face: str, table_yaw: float, yaw_max_deg: float = 45.0) -> Optional[Dict[str, Any]]:
    """§6.5 smallest rotation from a block's init orientation to the table-aligned goal orientation with ``goal_face`` up:
    yaw-align (|ψ| <= ``yaw_max_deg``) to the table axes (``table_yaw`` = world yaw of the goal grid's +x, i.e. the
    robot's right), then — when the goal face is not already up — one flip of ±90° about the horizontal table axis that
    brings it up. Returns ``None`` when the init up face is the opposite of the goal face (§4.6: never allowed).

    Output: ``yaw_deg`` ψ (the yaw-align), ``flip_axis`` (unit, world; None without a flip), ``flip_sign`` (±1 / 0),
    ``face_pair`` = the block axis parallel to the flip axis after the yaw-align (the pinch axis, ``"x"`` / ``"y"`` /
    ``"z"``), ``goal_quat`` = the table-aligned orientation with ``goal_face`` up reached by ψ then the flip."""
    q0 = _t(block_quat)
    up0 = up_face(q0)
    if up0 == opposite_face(goal_face):
        return None
    # yaw-align: the block's horizontal face normals are 90° apart, as are the table axes, so exactly one ψ in
    # (−45°, 45°] aligns them (|ψ| <= yaw_max_deg by construction for a cube)
    horiz = [f for f in FACE_AXES if f not in (up0, opposite_face(up0))]
    n0 = face_normal_world(horiz[0], q0)
    theta = math.atan2(float(n0[1]), float(n0[0]))
    psi = min((C.wrap_angle(float(table_yaw) + k * math.pi / 2 - theta) for k in range(4)), key=abs)
    if abs(psi) > math.radians(yaw_max_deg) + 1e-9:
        return None
    q1 = C.qmul(C.quat_from_yaw(psi), q0)
    if goal_face == up0:
        return {"yaw_deg": math.degrees(psi), "flip_axis": None, "flip_sign": 0, "face_pair": None, "goal_quat": q1, "flips": False}
    # one flip about the horizontal table axis a = n_goal × ẑ: R_a(+90°) n_goal = +ẑ brings the (side) goal face up
    n_goal = face_normal_world(goal_face, q1)
    axis = th.cross(n_goal, th.tensor([0.0, 0.0, 1.0]), dim=0)
    axis = axis / th.linalg.norm(axis)
    q_flip = C.quat_from_axis_angle(axis, math.pi / 2)
    q_goal = C.qmul(q_flip, q1)
    if float(face_normal_world(goal_face, q_goal)[2]) < 0.99:  # numerical guard: the other sense
        q_flip = C.quat_from_axis_angle(axis, -math.pi / 2)
        q_goal = C.qmul(q_flip, q1)
        sign = -1
    else:
        sign = 1
    # pinch axis = the block axis parallel to the flip axis after the yaw-align (pads flat, spec §6.5)
    pair = max("xyz", key=lambda ax: abs(float(th.dot(C.quat_rotate(q1, th.tensor(FACE_AXES["+" + ax])), axis))))
    return {"yaw_deg": math.degrees(psi), "flip_axis": axis, "flip_sign": sign, "face_pair": pair, "goal_quat": q_goal, "flips": True}


def facing_coarse_yaw(block_quat_up: th.Tensor, goal_face: str, toward_xy) -> float:
    """§4.6 visible-face rule: the coarse yaw (multiple of 90°) about the vertical that turns ``goal_face`` (a side face
    of the block at orientation ``block_quat_up``) toward the robot (``toward_xy`` = −inward, unit). Returns the yaw
    (rad) to apply (``R_z(yaw) ∘ block_quat_up``)."""
    tw = _t([float(toward_xy[0]), float(toward_xy[1]), 0.0])
    best, best_dot = 0.0, -2.0
    for k in range(4):
        yaw = k * math.pi / 2
        n = face_normal_world(goal_face, C.qmul(C.quat_from_yaw(yaw), _t(block_quat_up)))
        d = float(th.dot(n, tw))
        if d > best_dot:
            best, best_dot = yaw, d
    return best


def face_points(block_pose: Pose, face: str, edge: float = C.BLOCK_EDGE, n: int = 3, inset: float = 0.2) -> th.Tensor:
    """3 x 3 grid of world points on ``face`` of a cube (§4.6 visible-face ray test), inset from the face border."""
    p, q = _t(block_pose[0]), _t(block_pose[1])
    normal = th.tensor(FACE_AXES[face])
    ax = "xyz"[[abs(v) for v in FACE_AXES[face]].index(1.0)]
    others = [a for a in "xyz" if a != ax]
    u = th.tensor(FACE_AXES["+" + others[0]])
    v = th.tensor(FACE_AXES["+" + others[1]])
    half = 0.5 * edge
    coords = th.linspace(-half * (1 - inset), half * (1 - inset), n)
    pts = []
    for a in coords.tolist():
        for b in coords.tolist():
            pts.append(normal * half + u * a + v * b)
    return C.transform_points((p, q), th.stack(pts))


# ----------------------------------------------------------------------------------------------------------------
# pure helpers: §4.6 step 3 distractor spacing, §3.12 look-alikes, §6.10 yaw draw
# ----------------------------------------------------------------------------------------------------------------


def distractor_spacing(pinch_width: float, extra: float = 0.01, open_width: float = C.GRIPPER_OPEN_M) -> float:
    """§4.6 step 3: ``x = (open width 0.10 − pinch width)/2 + 0.01`` (``Grasp.pad_clearance`` + margin)."""
    return 0.5 * (float(open_width) - float(pinch_width)) + float(extra)


def grasp_pinch_width(grasp) -> Optional[float]:
    """Pinch width of a library grasp (``meta.width``; the assisted-grasp results record it), None when unknown / inf."""
    w = (getattr(grasp, "meta", None) or {}).get("width")
    if w is None:
        return None
    w = float(w)
    return None if not math.isfinite(w) else w


def exclude_lookalikes(pool: Sequence[Tuple[str, str]], target: Tuple[str, str], groups: Sequence[Sequence[Tuple[str, str]]],
                       rng=None) -> List[Tuple[str, str]]:
    """§3.12 rule 1: the pool minus the target and every member of its look-alike group, then at most one model per
    look-alike group (the kept member is the first in ``pool`` order, or a random one with ``rng``); exact duplicates of
    the target never survive."""
    grp_of: Dict[Tuple[str, str], int] = {}
    for gi, g in enumerate(groups):
        for m in g:
            grp_of[tuple(m)] = gi
    target = tuple(target)
    banned = set(groups[grp_of[target]]) if target in grp_of else set()
    banned.add(target)
    cands = [tuple(m) for m in pool if tuple(m) not in banned]
    if rng is not None and cands:
        cands = [cands[int(i)] for i in rng.permutation(len(cands))]
    seen_groups: Set[int] = set()
    out: List[Tuple[str, str]] = []
    for m in cands:
        gi = grp_of.get(m)
        if gi is not None:
            if gi in seen_groups:
                continue
            seen_groups.add(gi)
        out.append(m)
    return out


def axial_init_yaw(rng, aligned_yaw: float, deviation_range_deg: Sequence[float] = (20.0, 70.0)) -> Tuple[float, float]:
    """§6.10: ``yaw = aligned yaw ± U(20°, 70°)`` with a random sign; returns ``(yaw, delta)`` (delta signed, rad)."""
    lo, hi = float(deviation_range_deg[0]), float(deviation_range_deg[1])
    delta = math.radians(float(rng.uniform(lo, hi))) * (1.0 if rng.random() < 0.5 else -1.0)
    return C.wrap_angle(float(aligned_yaw) + delta), delta


# ----------------------------------------------------------------------------------------------------------------
# object poses on a support
# ----------------------------------------------------------------------------------------------------------------


def object_pose_on(s, hull: th.Tensor, rest_quat, xy, yaw: float, z_support: float, drop: float = Z_OFFSET) -> Pose:
    """§3.2: orientation ``R_z(yaw) ∘ rest``, z = support top + bottom offset + 3 mm."""
    quat = C.qmul(C.quat_from_yaw(yaw), _t(rest_quat))
    pos = th.tensor([float(xy[0]), float(xy[1]), float(z_support) + s.world.bottom_offset(hull, quat) + float(drop)])
    return pos, quat


def hull_extents(hull: th.Tensor, rest_quat) -> th.Tensor:
    """AABB extents (3,) of a local hull at the rest orientation."""
    pts = C.transform_points((th.zeros(3), _t(rest_quat)), hull)
    return pts.max(dim=0).values - pts.min(dim=0).values


def half_extent(hull: th.Tensor, rest_quat) -> float:
    """w_x of §4.4: half the larger horizontal bbox side at the rest orientation."""
    ext = hull_extents(hull, rest_quat)
    return 0.5 * float(max(float(ext[0]), float(ext[1])))


def height_of(hull: th.Tensor, rest_quat) -> float:
    return float(hull_extents(hull, rest_quat)[2])


#: facet snap (owner 2026-09-24): a lying body of revolution whose convex hull is a coarse prism is placed rolled onto a
#: facet. Search window about the long axis, step, coplanarity of the two lowest vertex lines, and the minimum width of
#: the support line the mass centre must lie inside (with ``FACET_COM_MARGIN`` to spare on both sides)
FACET_SNAP_MAX_DEG = 25.0
FACET_SNAP_STEP_DEG = 0.25
FACET_COPLANAR_TOL = 2e-4
FACET_MIN_WIDTH = 0.004
FACET_COM_MARGIN = 0.001


def prism_rest_quat(hull: th.Tensor, rest_quat, com=None) -> Tuple[th.Tensor, float]:
    """Rest orientation of a **lying prism** (owner 2026-09-24): the roll about its horizontal long axis, nearest to
    ``rest_quat``, at which the hull stands on a *facet* - its lowest vertex lines coplanar (``FACET_COPLANAR_TOL``) and at
    least ``FACET_MIN_WIDTH`` apart with the mass centre ``com`` (local, default the root origin = OmniGibson's authored
    centre of mass of these assets) strictly between them - instead of on an *edge*. The bottles' collision hulls are
    18-56-vertex prisms; ``bottle_of_milk/mrejrs`` at its canonical roll rests on two vertex lines 2 mm apart at y = +5 mm,
    so it tipped 7-8 deg onto the next facet at every release and kept rolling (0924-235434 attempt 15: 7.8 deg net). Not
    lying (the up extent is not the smallest, or the body is not elongated) or already facet-down: ``rest_quat``
    unchanged. Returns ``(quat, roll_rad)``."""
    q0 = _t(rest_quat)
    pts0 = C.transform_points((th.zeros(3), q0), hull)
    ext = pts0.max(dim=0).values - pts0.min(dim=0).values
    horiz = max(float(ext[0]), float(ext[1]))
    if not (float(ext[2]) <= min(float(ext[0]), float(ext[1])) * 1.05 and horiz > 1.5 * float(ext[2])):
        return q0, 0.0  # standing / flat objects: no roll axis
    ax = 0 if float(ext[0]) >= float(ext[1]) else 1  # the long axis (world-aligned after rest_quat)
    lat = 1 - ax
    axis = th.zeros(3)
    axis[ax] = 1.0
    c = th.zeros(3) if com is None else _t(com)
    c0 = C.quat_rotate(q0, c)

    def facet_down(theta: float) -> bool:
        q = C.quat_from_axis_angle(axis, theta)
        pts = C.quat_rotate(q, pts0)
        cc = C.quat_rotate(q, c0)
        z = pts[:, 2]
        low = pts[z < float(z.min()) + FACET_COPLANAR_TOL]
        lo, hi = float(low[:, lat].min()), float(low[:, lat].max())
        return (hi - lo) >= FACET_MIN_WIDTH and lo + FACET_COM_MARGIN <= float(cc[lat]) <= hi - FACET_COM_MARGIN

    if facet_down(0.0):
        return q0, 0.0
    step = math.radians(FACET_SNAP_STEP_DEG)
    n = int(round(math.radians(FACET_SNAP_MAX_DEG) / step))
    for k in range(1, n + 1):
        for theta in (k * step, -k * step):
            if facet_down(theta):
                return C.qmul(C.quat_from_axis_angle(axis, theta), q0), float(theta)
    return q0, 0.0


def hull_corners_world(hull: th.Tensor, pose: Pose) -> th.Tensor:
    pts = C.transform_points(pose, hull)
    lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
    return th.tensor([[float(lo[0] if sx == 0 else hi[0]), float(lo[1] if sy == 0 else hi[1]), float(lo[2] if sz == 0 else hi[2])]
                      for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])


def box_corners(centre_xy, half: float, z0: float, z1: float) -> th.Tensor:
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    return th.tensor([[cx + sx * half, cy + sy * half, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)])


# ----------------------------------------------------------------------------------------------------------------
# grasp library access (pipeline suffixes tolerated: the facade writes ``t`` / ``i`` / ``a`` / ``f``)
# ----------------------------------------------------------------------------------------------------------------

PIPELINE_VARIANTS = {"tabletop": "tabletop", "t": "tabletop", "insert": "insert", "i": "insert", "fl": "insert", "floating": "floating", "f": "floating",
                    "": "floating", "attachment": "attachment", "a": "attachment", "handle_half": "floating", "h": "floating"}


def library_grasps(s, role: str = "target", default_pipeline: str = "tabletop", kinds: Optional[Sequence[str]] = None):
    """``(grasps, key, rest_quat)`` of the role's model for the pipeline the customs entry requires (``roles.<role>.
    require.grasps``); ``Reject("missing_grasps:<pipeline>")`` when the library has no admissible row."""
    from ispatialgym.sampling import grasp_library as GL

    choice = s.choice.models[role]
    pipeline = PIPELINE_VARIANTS.get(s.gp.roles[role].grasp_pipeline() or default_pipeline, s.gp.roles[role].grasp_pipeline() or default_pipeline)
    kinds = list(kinds) if kinds else [pipeline]
    try:
        key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, kinds)
        grasps = GL.load_grasps(key)
    except (FileNotFoundError, ValueError) as e:
        _reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: {e}")
    if not grasps:
        _reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: no admissible grasp row")
    try:
        rest = C.rest_quat(key)
    except FileNotFoundError:
        rest = C.IDENTITY_QUAT.clone()
    return grasps, key, rest


# ----------------------------------------------------------------------------------------------------------------
# reach bands (§4.1, §3.10 caches)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class BandSpec:
    """What a task's reach band is computed for: the grasped object (``object`` = ``category/model``, ``grasps`` = its
    library rows, ``hull`` / ``rest_quat`` for the poses; ``generic_pinch`` for blocks / push poses), the spawn ``scale``
    (§3.10 cache key), ``placed`` objects (cache key tokens), the support height override ``z_support`` (a tray floor),
    the lattice override ``lattice_world`` (default: the free cells of the top), extra ``ignore`` objects for the grasp
    pose (a tray) and the ``arm``."""

    object: Optional[str] = None
    grasp_key: Optional[str] = None
    grasps: Optional[List[Any]] = None
    hull: Optional[th.Tensor] = None
    rest_quat: th.Tensor = field(default_factory=lambda: C.IDENTITY_QUAT.clone())
    scale: Optional[List[float]] = None
    placed: List[str] = field(default_factory=list)
    z_support: Optional[float] = None
    lattice_world: Optional[th.Tensor] = None
    ignore: List[Any] = field(default_factory=list)
    arm: str = "right"
    generic_pinch: bool = False
    height: float = 0.05
    eef_of_grasp: Optional[Callable[[Any], Pose]] = None  # T_obj->eef of a grasp entry (default ``g.pose``)
    contact: bool = False  # A1: B3's generic CONTACT set of the brace arm (closed gripper) instead of the pinch set

    @staticmethod
    def from_sampler(s, role: str = "target") -> "BandSpec":
        """The target's band from the attributes the reference task keeps (``grasps``, ``grasp_key``, ``hull[role]``,
        ``rest[role]``); a task module without library grasps sets ``s.band_spec`` explicitly."""
        mc = s.choice.models[role]
        grasps = getattr(s, "grasps", None)
        hull = (getattr(s, "hull", None) or {}).get(role) if isinstance(getattr(s, "hull", None), dict) else None
        rest = (getattr(s, "rest", None) or {}).get(role) if isinstance(getattr(s, "rest", None), dict) else None
        if hull is None:
            hull = s.world.local_hull(s.obj(role))
        if rest is None:
            rest = C.IDENTITY_QUAT.clone()
        return BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=getattr(s, "grasp_key", None) or mc.grasp_key, grasps=grasps, hull=hull,
                        rest_quat=_t(rest), scale=list(mc.scale) if mc.scale else None, generic_pinch=not grasps, height=height_of(hull, rest))


def ensure_bands(s, spec: Optional[BandSpec] = None, top=None, edge: Optional[EdgeFrame] = None, force: bool = False) -> RB.ReachBands:
    """The reach band of the sampler's support and object (cached on ``s.bands``): ``ReachBands.load`` from the cache
    directories (§3.10), else computed in-process with ``BandComputer`` (the function the CLI runs) and written to
    ``reach_bands.cache_root()``. On a union top (neighbouring supports) each neighbour's band too: ``neighbors.NeighborBands``."""
    cached = getattr(s, "bands", None)
    if cached is not None and not force:
        return cached
    top = _top(s, top)
    spec = spec or getattr(s, "band_spec", None) or BandSpec.from_sampler(s)
    scene, support = s.world.scene_model, top.name
    gkey = "generic" if spec.generic_pinch else spec.grasp_key
    surface = surface_band_kwargs(s)
    if map_sampling_root() and not spec.generic_pinch and not spec.placed and not surface:
        mb = map_bands(s, spec, top)
        if mb is not None:
            s.bands = mb
            return s.bands
        if os.environ.get("ISPATIALGYM_REACH_MAPS_STRICT"):
            _reject("missing_map", f"no reach map for {spec.object} ({gkey}) at {support}'s height")
    try:
        bands = RB.ReachBands.load(scene, support, spec.object, gkey, placed=spec.placed, scale=spec.scale, **surface)
        log.info("reach band cache hit for %s / %s (%s%s): %s", support, spec.object, gkey, f", surface_z {surface['surface_z']:.2f}" if surface else "", bands.path)
    except FileNotFoundError:
        if surface:
            # the in-process computation ignores the whole support at the grasp pose; inside a bookcase the board above and the
            # side walls must stay obstacles (reach_bands --surface-z)
            _reject("missing_band", f"no interior reach band for {support} at surface_z {surface['surface_z']:.2f} ({spec.object}, {gkey})")
        log.info("reach band cache miss for %s / %s (%s): computing in-process", support, spec.object, gkey)
        bands = compute_bands(s, spec, top, edge)
    s.bands = NB.with_neighbor_bands(s, bands, spec, top, compute_bands)
    return s.bands


def map_sampling_root() -> Optional[str]:
    """``ISPATIALGYM_REACH_MAPS`` (``ispatialgym.sampling.map_sampling``): the root of the scene-independent reach maps, or None (legacy bands)."""
    return os.environ.get("ISPATIALGYM_REACH_MAPS") or None


def map_bands(s, spec: BandSpec, top, arm: Optional[str] = None) -> Optional[RB.ReachBands]:
    """The scene-independent reach map of the sampler's object at the support's actual height above the floor, as a multi-distance band
    view (``map_sampling.as_bands``); None when the object has no v2 library entry or no compatible map covers the height (the anchors of
    ``ISPATIALGYM_REACH_MAP_ANCHORS``: exact or within 1 cm, else uncovered). Turns the grasp pose's support policy on
    (``ISPATIALGYM_SUPPORT_POLICY``): exact sampling keeps the support for the palm / wrist camera / arm and the 2 mm finger rule.
    ``arm`` (default ``spec.arm``): the arm whose map is looked up (:func:`arm_band`'s other segment arm)."""
    arm = arm or spec.arm
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling import map_sampling as MS
    from ispatialgym.sampling import reach_maps as RMAP
    from ispatialgym.sampling.height_anchors import AnchorSet

    lib = GL.v2_entry(spec.grasp_key) if spec.grasp_key else None
    if lib is None or not lib.rests:
        return None
    rid = sorted(lib.rests, key=lambda r: (not lib.rests[r].stable, r))[0]
    h = float(top.z) - float(getattr(s.world, "floor_z", 0.0))
    anchors = None
    ap = os.environ.get("ISPATIALGYM_REACH_MAP_ANCHORS")
    if ap and os.path.exists(ap):
        with open(ap) as f:
            an = json.load(f)
        anchors = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
    root = os.path.join(map_sampling_root(), os.path.basename(os.path.dirname(lib.path)))
    rm = RMAP.ReachMap.find(root, lib, h, rest_id=rid, arm=arm, anchors=anchors)
    if rm is None:
        log.info("reach map miss for %s (%s) at h %.4f, %s arm (%s)", spec.object, spec.grasp_key, h, arm, root)
        return None
    # of-v2maps 2026-11-06: the grasp approach only (ik.support_policy_scope) - "1" also policed places / carries / releases
    os.environ.setdefault("ISPATIALGYM_SUPPORT_POLICY", "grasp")
    # ... against the support's MEASURED top (the height the map was computed at), not its AABB top (ik.support_plane_z)
    ikc = getattr(s, "ikc", None)
    if ikc is not None:
        if not isinstance(getattr(ikc, "support_plane_z", None), dict):
            ikc.support_plane_z = {}
        ikc.support_plane_z[str(top.name)] = float(top.z)
        # the support-policy diagnostic's expected object z (ik._support_policy_diag): the plane + the v2 rest's root height
        ikc.support_policy_rest = {"entry": os.path.basename(os.path.dirname(lib.path)), "rest_id": rid, "h_root": float(lib.rests[rid].h_root),
                                 "quat_xyzw": [float(v) for v in lib.rests[rid].quat_xyzw]}
    log.info("reach map hit for %s at h %.4f, %s arm: %s", spec.object, h, arm, rm.path)
    return MS.as_bands(rm, overhang=0.0)


def compute_bands(s, spec: BandSpec, top, edge: Optional[EdgeFrame] = None, lattice: float = RB.LATTICE, out_dir: Optional[str] = None,
                  trunks: Optional[Sequence[Tuple[float, float]]] = None) -> RB.ReachBands:
    """§4.1 steps 1–4 in-process (``reach_bands.BandComputer``) at the band stance (edge midpoint, s = 0.45 + overhang) of
    ``edge`` (default: the longest free edge), written as ``<cache>/<name>.npz/.json`` like the CLI. A union top's band is its
    support's own (bands are per support). Computed on ``reach_bands.band_kit`` (the band's own kit with a snug mesh store;
    ``s.kit`` is not touched), recorded in ``meta.schedule``. Capped by the sampling run's time budget (``s.deadline``,
    ``sample_isg --time-budget-s``): past it the computation stops before its next trunk / IK round, nothing is written, and
    ``Reject("band_budget")``. ``ISPATIALGYM_BAND_TRUNK_CHECK=idle-arm``: the CLI's ``--trunk-check idle-arm``, keyed, named and
    recorded like the CLI's (``reach_bands.band_trunk_check``)."""
    world = s.world
    kit = RB.band_kit(world, s.kit)
    top = NB.own_top(top)
    support = world.obj(top.name)
    bc = RB.BandComputer(world, kit, arms=(spec.arm,), lattice=lattice)
    bc.deadline = getattr(s, "deadline", None)
    tc_key = RB.band_trunk_check(bc, spec.arm)
    try:
        stance = bc.stance_at_edge(top, support, RB.STAND_OFF, edge.name if edge is not None else None)
    except RuntimeError as e:
        # oracle-fix 2026-11-03 (disembedding#06 c3_k0, breakfast_table_rhjoby_0: "no stance on free floor around ..." killed the template
        # setup with a traceback, no instance in 508 s): no band stance on free floor at any edge of this top is a normal rejection of
        # the (support, edge) - the sampler goes on to its next candidate - not a crash of the run
        _reject("no_manipulation_stance", f"band stance of {top.name}: {e}", support=top.name, edge=edge.name if edge is not None else None)
    z = float(spec.z_support) if spec.z_support is not None else float(top.z)
    lattice_world = spec.lattice_world if spec.lattice_world is not None else top.cells(free_only=True)
    ignore = [support] + list(spec.ignore)
    h = float(spec.height)
    contact_mode = bool(spec.contact and hasattr(bc, "contact_height"))
    if spec.contact and not contact_mode:
        log.info("reach_bands.BandComputer has no contact mode yet: the left band uses the generic pinch set with the left arm")
    if contact_mode:
        # B3's contact set: closed fingers, side contacts at CONTACT_HEIGHT_FRACTION·h / top presses, the classes bitmask
        bc.contact_height = h
        bc.fingers = float(getattr(C, "FINGER_CLOSED", 0.0))
        pose_fn, grasp_sets, max_off = None, None, 0.0
        z_eval = z
    elif spec.generic_pinch or not spec.grasps:
        pose_fn, grasp_sets, max_off = None, None, 0.0
        z_eval = z + max(0.6 * h, 0.03)
    else:
        hull, rest = spec.hull, spec.rest_quat

        def pose_fn(world_xy, yaw, _hull=hull, _rq=rest, _z=z):
            return object_pose_on(s, _hull, _rq, world_xy, yaw, _z)

        eef_of = spec.eef_of_grasp or (lambda g: g.pose)
        grasp_sets = [_BandGrasp(g.name, eef_of(g)) for g in spec.grasps]
        max_off = max(float(th.linalg.norm(_t(g.position))) for g in grasp_sets)
        z_eval = z
    kit.sync_world()
    gen_trunks = getattr(s, "gen", lambda k, d=None: d)("band_trunks", None)
    trunks = list(trunks) if trunks is not None else ([tuple(float(v) for v in t) for t in gen_trunks] if gen_trunks else RB.trunk_grid(z - kit.floor_z))
    t0 = time.time()
    try:
        result = bc.compute(stance, lattice_world, z_eval, pose_fn, grasp_sets, ignore, arm=spec.arm, trunks=trunks, max_grasp_offset=max_off)
    except RB.BandDeadline as e:
        kit.sync_world()
        _reject("band_budget", f"in-process {spec.arm}-arm band of {top.name} / {spec.object or 'generic'} not finished: {e} ({time.time() - t0:.0f} s in); "
                               f"nothing written - a band job publishes it")
    kit.sync_world()
    lat_all = np.concatenate([c[:, 1] for c in result["cells"].values() if len(c)]) if any(len(c) for c in result["cells"].values()) else np.zeros(0)
    meta: Dict[str, Any] = {"scene": world.scene_model, "rooms": list(world.rooms or []), "support": top.name, "lattice": lattice, "stand_off": RB.STAND_OFF,
                            "arms": "untucked", "grid": {"translates": list(RB.TRANSLATES), "tilts": list(RB.TILTS), "low_top_tilt": RB.LOW_TOP_TILT},
                            "seeds": 12, "pre_grasp": RB.PRE_GRASP, "yaws": list(RB.YAWS), "mode": "tabletop", "smoke": False, "scale": spec.scale,
                            "placed_specs": list(spec.placed), "object": spec.object, "grasp_key": None if spec.generic_pinch else spec.grasp_key,
                            "generic_pinch": bool(spec.generic_pinch or not spec.grasps),
                            "grasp_names": [g.name for g in (grasp_sets or [])], "rest_quat_xyzw": _t(spec.rest_quat).tolist(), "stance": stance,
                            "top": top.to_json(), "z_support": z_eval, "trunks": result["trunks"],
                            "union_extent": [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None,
                            "n_reachable_total": int(sum(len(c) for c in result["cells"].values())), "time_s": round(time.time() - t0, 1),
                            "computed_by": "sampling.v2.tabletop.compute_bands", "arm": spec.arm, "schedule": RB.band_schedule(kit, s.kit)}
    if contact_mode:  # B3's contact-band fields (``ReachBands.find(arm="left", contact_set=..., height=...)`` matches them)
        meta.update({"contact_set": CONTACT_SET, "height_class": height_class(h), "contact_height_fraction": CONTACT_HEIGHT_FRACTION,
                     "classes": [c["name"] for c in getattr(RB, "CONTACT_CLASSES", [])], "fingers": bc.fingers})
    elif grasp_sets is None:
        meta["classes"] = list(getattr(RB, "GENERIC_CLASSES", []))
    if tc_key:
        meta["trunk_check"] = RB.trunk_check_meta(spec.arm)
    key = RB._key({"scene": world.scene_model, "support": top.name, "z": round(z_eval, 2), "stance": stance["edge"], "s": RB.STAND_OFF, "object": spec.object,
                   "scale": spec.scale, "grasps": [(g.name, _t(g.position).tolist(), _t(g.quat).tolist()) for g in (grasp_sets or [])] or "generic",
                   "rest": _t(spec.rest_quat).tolist(), "placed": list(spec.placed), "grid": meta["grid"], "lattice": lattice, "arms": "untucked", "seeds": 12,
                   **tc_key})
    meta["key"] = key
    out_dir = out_dir or RB.cache_root()
    os.makedirs(out_dir, exist_ok=True)
    oname = (spec.object or "generic").replace("/", "-")
    name = f"tabletop__{world.scene_model}__{top.name}{RB.TRUNK_CHECK_IDLE_TAG if tc_key else ''}__{oname}__{key}"
    arrays = {RB.trunk_key(tr, ti): np.asarray(c, dtype=np.float32).reshape(-1, 2) for (tr, ti), c in result["cells"].items()}
    for (tr, ti), p in result.get("pairs", {}).items():
        arrays[RB.trunk_key(tr, ti) + "__pairs"] = np.asarray(p, dtype=np.int32).reshape(-1, 3)
    for (tr, ti), m in (result.get("classes") or {}).items():  # B3's per-cell class bitmask (generic / contact sets)
        if m is not None and len(m):
            arrays[RB.trunk_key(tr, ti) + "__classes"] = np.asarray(m, dtype=np.int64).reshape(-1)
    path = os.path.join(out_dir, name + ".npz")
    np.savez_compressed(path, **arrays)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in result["cells"].items() if not _trunk_collides(result["trunks"], k)}
    pairs = {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in result.get("pairs", {}).items() if k in cells}
    classes = {k: np.asarray(v, dtype=np.int64).reshape(-1) for k, v in (result.get("classes") or {}).items() if k in cells and v is not None and len(v)}
    free = RB.gpu_free_gb()
    log.info("reach band computed for %s / %s: %d reachable cells over %d trunks in %.0fs -> %s (%s kit, mesh store %d%s)", top.name, spec.object,
             meta["n_reachable_total"], len(cells), meta["time_s"], path, meta["schedule"]["kit"], meta["schedule"]["mesh_cache"],
             "" if free is None else f", GPU free {free:.2f} GB")
    try:
        return RB.ReachBands(meta, cells, pairs, path, classes)
    except TypeError:  # loader without the classes field
        return RB.ReachBands(meta, cells, pairs, path)


def _trunk_collides(entries: Sequence[Dict[str, Any]], key: Tuple[float, float]) -> bool:
    for e in entries:
        if abs(float(e["translate"]) - key[0]) < 1e-6 and abs(float(e["tilt"]) - key[1]) < 1e-6:
            return bool(e.get("collides"))
    return False


class _BandGrasp:
    """Minimal grasp-like entry for the band computation (``.name``, ``.pose`` = T_obj->eef, ``.position`` / ``.quat``)."""

    def __init__(self, name: str, pose: Pose):
        self.name = name
        self.position, self.quat = _t(pose[0]), _t(pose[1])

    @property
    def pose(self) -> Pose:
        return self.position, self.quat


# ----------------------------------------------------------------------------------------------------------------
# the arm of a single-arm segment (owner 2026-09-27, first-demo v2.2)
# ----------------------------------------------------------------------------------------------------------------


def default_arm(s) -> str:
    """The task's working arm (``generation.arm``, core's ``IKChecker.default_working_arms``; ``right`` by default)."""
    arms = list(getattr(getattr(s, "ikc", None), "default_working_arms", None) or ["right"])
    return arms[0] if len(arms) == 1 else "right"


def object_free(s, role: str = "target") -> bool:
    """Whether the object of ``role`` is free: not articulated (the problem gives the role no joint) and not attached at the
    goal (no ``attached`` atom in the grounded BDDL's goal)."""
    gp = getattr(s, "gp", None)
    r = (getattr(gp, "roles", None) or {}).get(role)
    if r is not None and getattr(r, "joint", None):
        return False
    text = str(getattr(gp, "bddl_text", "") or "")
    return "attached" not in (text.split(":goal", 1)[1] if ":goal" in text else "")


def arm_offer(s, *, other_arm_roles: Sequence[str] = (), role: str = "target", free: Optional[bool] = None) -> Tuple[List[str], str]:
    """The situation predicate of the other arm (owner 2026-09-27, first-demo v2.2; side-effect free: it reads the plan, the problem
    and ``generation``): the arms offered for a single-arm segment (grasp -> carry -> place -> release, or a held object's hold ->
    reorient), the default arm first - the other arm too when it has no role besides the segment (``other_arm_roles`` empty: no
    brace, push or hold planned) and the object is free (:func:`object_free`); ``generation.other_arm`` false switches it off.
    Returns ``(arms, why the other arm is not offered or "")``. Which offered arm makes the segment is the orientation rule's choice
    at the segment's first contact."""
    arms = [default_arm(s)]
    gen = getattr(s, "gen", None)
    on = gen("other_arm", True) if callable(gen) else True
    if on is False or str(on).strip().lower() in ("off", "false", "0", "no"):
        return arms, "generation.other_arm is off"
    if other_arm_roles:
        return arms, f"the other arm has a role besides the segment: {', '.join(other_arm_roles)}"
    if not (object_free(s, role) if free is None else free):
        return arms, "the object is not free (articulated or attached at the goal)"
    return arms + [RB.other_arm(arms[0])], ""


def offered_arms(s, *, other_arm_roles: Sequence[str] = (), role: str = "target", free: Optional[bool] = None) -> List[str]:
    """The arms of :func:`arm_offer`."""
    return arm_offer(s, other_arm_roles=other_arm_roles, role=role, free=free)[0]


def arm_offer_record(template_arms: Sequence[str], arms_here: Sequence[str], rec: GraspRecord, not_offered: str = "") -> Dict[str, Any]:
    """``grasps[].arm_offer`` (fired / not fired, the oracle copies it into the notes of the segment's actions): ``fired`` when the
    segment's first contact was offered to more than one arm; the arms offered for the template and at this segment's focus, the arm
    that makes it and why (:func:`segment_arm_summary`), else why the other arm was not offered (the template's predicate, or its band
    does not cover the focus)."""
    s_arm = segment_arm_summary(rec, arms_here)
    fired = len(arms_here) > 1
    out = {"fired": fired, "template_arms": list(template_arms), "arms": list(arms_here), "arm": rec.arm, "reason": s_arm["reason"],
           "against_default": s_arm["against_default"], "mirrored_band": s_arm["mirrored_band"]}
    if not fired:
        out["not_fired"] = not_offered or ("the other arm's band does not cover the focus" if len(template_arms) > 1 else "one arm offered")
    return out


def band_facts(bands: Optional[RB.ReachBands]) -> Optional[Dict[str, Any]]:
    """What a record says about the band a choice used: its arm, file, and for a mirror its source."""
    if bands is None:
        return None
    d: Dict[str, Any] = {"arm": bands.meta.get("arm", "right"), "path": bands.path}
    if bands.meta.get("mirrored_from"):
        d.update({"mirrored_from": bands.meta["mirrored_from"], "mirror_source": bands.meta.get("mirror_source")})
    return d


def arm_band(s, arm: str, *, spec: Optional[BandSpec] = None, top=None, edge: Optional[EdgeFrame] = None, bands: Optional[RB.ReachBands] = None) -> RB.ReachBands:
    """``arm``'s reach band for the sampler's support and object (cached on ``s.bands_by_arm``): ``bands`` (default
    :func:`ensure_bands`) when it is that arm's; else the arm's own band from the cache directories, else ``bands`` mirrored in
    memory (``ReachBands.mirrored``: never written, never computed in-process - a missing default-arm band is computed as
    before and then mirrored). The choice goes to ``template_facts["arm_bands"]``. Neighbouring supports: each top's band for ``arm``."""
    base = bands if bands is not None else ensure_bands(s, spec, top=top, edge=edge)
    if isinstance(base, NB.NeighborBands):
        return base.for_arm(arm, lambda b: arm_band(s, arm, bands=b))
    if base.arm == arm:
        return base
    cache = s.__dict__.setdefault("bands_by_arm", {})
    key = (arm, id(base))
    if key in cache:
        return cache[key]
    m = base.meta
    if m.get("mode") == "map_view":
        own = map_view_arm_band(s, arm, spec=spec, top=top, edge=edge)
        cache[key] = own
        tf = getattr(s, "template_facts", None)
        if isinstance(tf, dict):
            tf.setdefault("arm_bands", {})[arm] = band_facts(own)
        return own
    srf = m.get("surface") or {}
    surface = {"surface_z": float(srf["z_requested_above_floor"])} if srf.get("mode") == "interior" and srf.get("z_requested_above_floor") is not None else {}
    try:
        own = RB.ReachBands.load(m.get("scene"), m.get("support"), m.get("object"), "generic" if m.get("generic_pinch") else m.get("grasp_key"),
                                 placed=RB.canonical_placed(m), under=m.get("under_partner"), scale=m.get("scale"), arm=arm,
                                 under_edge=None if m.get("mode") != "floor" else bool(m.get("under_edge")), **surface)
        log.info("%s-arm reach band: %s", arm, own.path)
    except FileNotFoundError:
        own = base.mirrored()
        log.info("%s-arm reach band: the %s band mirrored (%s)", arm, base.arm, base.path)
    cache[key] = own
    tf = getattr(s, "template_facts", None)
    if isinstance(tf, dict):
        tf.setdefault("arm_bands", {})[arm] = band_facts(own)
    return own


def map_view_arm_band(s, arm: str, *, spec: Optional[BandSpec] = None, top=None, edge: Optional[EdgeFrame] = None) -> RB.ReachBands:
    """of-v2maps 2026-11-06: ``arm``'s band when the default arm's is a reach-map view (``mode: map_view``, :func:`map_bands`). A map view
    is never mirrored (``ReachBands.mirrored`` raises: the map is computed per arm in the synthetic world), so this is ``arm``'s own map;
    without one, the legacy band path for that arm (a miss keeps the legacy path, per arm): ``arm``'s cached band, else the default
    arm's legacy band (cached, or computed in-process as :func:`ensure_bands` would) mirrored. One INFO line names the choice."""
    sp = spec or getattr(s, "band_spec", None) or BandSpec.from_sampler(s)
    t = _top(s, top)
    own = map_bands(s, sp, t, arm=arm)
    if own is not None:
        log.info("%s-arm reach band: the %s-arm reach map (%s)", arm, arm, own.meta.get("map_path"))
        return own
    scene, support = s.world.scene_model, t.name
    gkey = "generic" if sp.generic_pinch else sp.grasp_key
    try:
        own = RB.ReachBands.load(scene, support, sp.object, gkey, placed=sp.placed, scale=sp.scale, arm=arm)
        log.info("%s-arm reach band: no %s-arm reach map, the legacy band %s", arm, arm, own.path)
        return own
    except FileNotFoundError:
        pass
    try:
        legacy = RB.ReachBands.load(scene, support, sp.object, gkey, placed=sp.placed, scale=sp.scale, arm=sp.arm)
    except FileNotFoundError:
        log.info("reach band cache miss for %s / %s (%s): computing in-process (the %s arm has no reach map)", support, sp.object, gkey, arm)
        legacy = compute_bands(s, sp, t, edge)
    own = legacy if legacy.arm == arm else legacy.mirrored()
    log.info("%s-arm reach band: no %s-arm reach map, the legacy %s band%s (%s)", arm, arm, legacy.arm, "" if own is legacy else " mirrored", legacy.path)
    return own


def arm_bands_for(s, arms: Sequence[str], **kw) -> Dict[str, RB.ReachBands]:
    """``{arm: band}`` for ``arms`` (:func:`arm_band`), in their order."""
    return {a: arm_band(s, a, **kw) for a in arms}


def stance_for_arm(s, stance: Stance, arm: str) -> Optional[Stance]:
    """``stance`` with ``arm`` as its one working arm: the stance itself when it already is, else a copy whose other arm idles in
    the first recorded collision-free posture (A6: untucked, ready, tucked - the two arms' postures follow the joint sign map);
    None when every idle posture collides."""
    if stance.working_arms == [arm]:
        return stance
    st = stance.with_trunk(stance.translate, stance.tilt, stance.name)
    st.arms, st.idle_posture = [arm], None
    st.extras = dict(stance.extras)
    if not bool(s.ikc.stances_collision_free([st])[0]):
        return None
    return st


def adopt_arm(stance: Stance, arm_stance: Stance) -> None:
    """Make ``stance`` (the caller's record) the stance of ``arm_stance``'s working arm and idle posture, in place."""
    stance.arms, stance.idle_posture = list(arm_stance.arms), arm_stance.idle_posture


def band_stand_off(bands: Optional[RB.ReachBands], stance) -> Optional[float]:
    """The stand-off a band is read at (owner 2026-10-11 item 1): the stance's nominal stand-off when the band is a multi-distance
    band (meta ``stand_offs``; ``ReachBands`` takes the nearest stored one), else None (the reference maps, as before)."""
    if bands is None or stance is None or not bands.meta.get("stand_offs"):
        return None
    so = getattr(stance, "stand_off", None)
    return None if so is None else float(so)


def band_kw(bands: Optional[RB.ReachBands], stance) -> Dict[str, Any]:
    """``{"stand_off": s}`` for the ``ReachBands`` getters at ``stance`` (:func:`band_stand_off`), ``{}`` when it is None."""
    so = band_stand_off(bands, stance)
    return {} if so is None else {"stand_off": so}


def band_cells_world(bands: RB.ReachBands, stance: Stance, translate: Optional[float] = None, tilt: Optional[float] = None) -> th.Tensor:
    """The band cells of the grid trunk (same translate, nearest tilt; a multi-distance band's at the stance's stand-off) in the
    world at ``stance`` (robot-relative cells, §4.5 step 2 "band shifted with the stand-off"); neighbouring supports: each band's
    cells on its own top."""
    if isinstance(bands, NB.NeighborBands):
        return bands.cells_world(lambda b: band_cells_world(b, stance, translate, tilt))
    tr = stance.translate if translate is None else translate
    ti = stance.tilt if tilt is None else tilt
    cells = bands.band(float(tr), float(ti), **band_kw(bands, stance))
    if len(cells) == 0:
        return th.zeros(0, 2)
    return th.tensor(bands.to_world(cells, stance.base_pose2d), dtype=th.float32)


def band_union_cells(bands: RB.ReachBands, stance=None) -> th.Tensor:
    """Every band cell over every trunk (base frame), deduplicated; a multi-distance band's at ``stance``'s stand-off; neighbouring
    supports: each band's cells that fall on its own top at ``stance``."""
    if isinstance(bands, NB.NeighborBands):
        return bands.union_cells(stance, lambda b: band_union_cells(b, stance))
    so = band_stand_off(bands, stance)
    cells_by_trunk = bands._maps(so)[0] if so is not None else bands.cells
    if not cells_by_trunk:
        return th.zeros(0, 2)
    allc = np.concatenate([c for c in cells_by_trunk.values() if len(c)]) if any(len(c) for c in cells_by_trunk.values()) else np.zeros((0, 2))
    if len(allc) == 0:
        return th.zeros(0, 2)
    uniq = np.unique(np.round(allc, 4), axis=0)
    return th.tensor(uniq, dtype=th.float32)


def workspace(s, bands: RB.ReachBands, stance: Stance, top, erosion: float = 0.0) -> Dict[str, Any]:
    """§4.3 workspace = band union ∩ top of this edge at ``stance``: world cells, depths from the edge and laterals from
    the centre line (``d_near`` / ``d_ws`` / ``lam_min`` / ``lam_max``)."""
    base = band_union_cells(bands, stance)
    out: Dict[str, Any] = {"cells_world": th.zeros(0, 2), "cells_base": th.zeros(0, 2), "depths": [], "laterals": []}
    if len(base) == 0:
        return out
    world_xy = th.tensor(bands.to_world(base.numpy(), stance.base_pose2d), dtype=th.float32)
    keep = [i for i in range(len(world_xy)) if top.contains(world_xy[i], erosion)]
    if not keep:
        return out
    cw = world_xy[keep]
    cb = base[keep]
    depths = [stance.local(c)[0] for c in cw]
    lats = [float(v) for v in cb[:, 1].tolist()]
    out.update({"cells_world": cw, "cells_base": cb, "depths": depths, "laterals": lats, "d_near": min(depths), "d_ws": float(np.mean(depths)),
                "d_far": max(depths), "lam_min": min(lats), "lam_max": max(lats)})
    return out


def stored_feasible_set(bands: RB.ReachBands, stance: Stance, grasps: Sequence[Any], yaws: Sequence[float] = YAWS) -> Optional[List[Tuple[float, Any]]]:
    """train-fast 2026-10-31 (``ISPATIALGYM_FAST_BAND_FEASIBLE_REUSE``): the feasible set F a sampler stored in the band for this stance's
    (trunk, focus cell) (:func:`store_feasible_set` appends ``meta["feasible_sets"]`` entries ``{trunk, cell, feasible: [[yaw, grasp name]]}``
    with the sampler's world yaws and library grasp names; ``removed`` marks an empty F) mapped back onto ``grasps`` / ``yaws``, or None
    when nothing was stored for that cell (or the stored names match no grasp): the caller then screens with the batched IK as before
    (two padded 64 x 12 cuRobo calls per arm and candidate, 17-40 s on a shared TITAN RTX). The band's own generation-time ``pairs``
    (yaw offsets relative to the base yaw) are not read here - their yaw convention differs."""
    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(stance.focus[:2]) if stance.focus is not None else None
    if bands is None or stance.focus is None or not isinstance(getattr(bands, "meta", None), dict) or bands.meta.get("stand_offs"):
        return None
    entries = bands.meta.get("feasible_sets") or []
    if not entries:
        return None
    k = bands.nearest_trunk(float(stance.translate), float(stance.tilt))
    if k is None:
        return None
    cell_base = bands.to_base(np.asarray(stance.focus[:2].tolist()), stance.base_pose2d)[0]
    i = bands.cell_index(k[0], k[1], cell_base)
    if i is None:
        return None
    for fs in reversed(entries):  # the most recent store of this (trunk, cell) wins
        tr = fs.get("trunk") or []
        if len(tr) != 2 or abs(float(tr[0]) - float(k[0])) > 1e-6 or abs(float(tr[1]) - float(k[1])) > 1e-6:
            continue
        if bands.cell_index(k[0], k[1], np.asarray(fs.get("cell") or [1e9, 1e9], dtype=float)) != i:
            continue
        if fs.get("removed"):
            return []
        by_name: Dict[str, Any] = {}
        for g in grasps:
            by_name.setdefault(C.split_twin(getattr(g, "name", str(g)))[0], g)
        out: List[Tuple[float, Any]] = []
        for yaw, gname in fs.get("feasible") or []:
            g = by_name.get(str(gname))
            if g is None:
                continue
            yaw0 = min(yaws, key=lambda y: abs(C.wrap_angle(float(yaw) - float(y))))
            out.append((float(yaw0), g))
        return out if out else None
    return None


def evidence_cell(s, bands: RB.ReachBands, stance: Stance) -> Optional[Tuple[Any, str, str]]:
    """train-fast 2026-11-01 (BAND_EVIDENCE): ``(evidence, world key, cell key)`` of ``stance``'s (trunk, focus cell) in ``bands`` (the band
    of the top under the focus for neighbouring supports), None when the band has no such cell or keeps per-stand-off maps."""
    from ispatialgym.sampling.v2 import band_evidence as EV

    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(stance.focus[:2]) if stance.focus is not None else None
    if bands is None or stance.focus is None or not isinstance(getattr(bands, "meta", None), dict) or bands.meta.get("stand_offs"):
        return None
    from ispatialgym.sampling.v2 import band_refine as BR

    if BR.refine_on():
        # BAND_REFINE: the stance's own trunk (0.01) and the focus cell (a refined cell: possibly an addition, not on the grid trunk);
        # verified reachability does not depend on the world -> one world key
        cb = bands.to_base(np.asarray(stance.focus[:2].tolist()), stance.base_pose2d)[0]
        return BR.store_of(bands), BR.WORLD, BR.cell_key((float(stance.translate), float(stance.tilt)), cb)
    k = bands.nearest_trunk(float(stance.translate), float(stance.tilt))
    if k is None:
        return None
    cell_base = bands.to_base(np.asarray(stance.focus[:2].tolist()), stance.base_pose2d)[0]
    i = bands.cell_index(k[0], k[1], cell_base)
    if i is None:
        return None
    crop = s._world_scene_crop() if hasattr(s, "_world_scene_crop") else None
    wk = EV.world_key(getattr(s.world, "scene_model", "?"), list(getattr(s.world, "loaded_rooms", None) or []), crop)
    # the stance's own trunk, not the band's nearest grid trunk: Phase 2 tilts lie anywhere in a +-0.15 rad band around a grid tilt
    # (prompt0.tilt_band), and a cell reachable at the grid tilt may not be at the stance's (inside_02: x 0.96 m at 0.25 / -1.2 yes,
    # at 0.25 / -1.07 no) - one record per grid trunk would let empty screens at one tilt kill the cell for the other
    return EV.evidence_of(bands), wk, EV.cell_key((round(float(stance.translate), 3), round(float(stance.tilt), 3)), bands.cells[k][i])


def store_feasible_set(bands: RB.ReachBands, stance: Stance, F: Sequence[Tuple[float, Any]], yaws: Sequence[float] = YAWS) -> None:
    """§4.6 step 1 / §3.10: store the feasible set F of the focus cell under (trunk, cell) in the band (``pairs`` +
    ``meta.feasible_sets``); an empty F removes the cell from the band. In-memory (and written back to the band's
    ``.npz`` / ``.json`` when the band was loaded from or saved to a cache file). Neighbouring supports: the band of the top under
    the focus."""
    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(stance.focus[:2]) if stance.focus is not None else None
    if bands is None or stance.focus is None:
        return
    if bands.meta.get("stand_offs"):
        return  # the write-back below holds the reference maps only: a multi-distance band's per-stand-off arrays would be dropped
    k = bands.nearest_trunk(float(stance.translate), float(stance.tilt))
    if k is None:
        return
    cell_base = bands.to_base(np.asarray(stance.focus[:2].tolist()), stance.base_pose2d)[0]
    i = bands.cell_index(k[0], k[1], cell_base)
    fs = bands.meta.setdefault("feasible_sets", [])
    if i is None:
        return
    if not F:
        cells = bands.cells[k]
        keep = np.array([j for j in range(len(cells)) if j != i], dtype=int)
        bands.cells[k] = cells[keep] if len(keep) else np.zeros((0, 2))
        p = bands.pairs.get(k)
        if p is not None and len(p):
            p = p[p[:, 0] != i]
            remap = {old: new for new, old in enumerate(keep.tolist())}
            p = np.array([(remap[c], y, g) for c, y, g in p.tolist() if c in remap], dtype=np.int32).reshape(-1, 3)
            bands.pairs[k] = p
        fs.append({"trunk": list(k), "cell": [float(v) for v in cell_base], "feasible": [], "removed": True})
    else:
        names = bands.meta.get("grasp_names") or []
        rows = []
        for yaw, g in F:
            yi = min(range(len(yaws)), key=lambda j: abs(C.wrap_angle(float(yaw) - yaws[j])))
            gname = C.split_twin(getattr(g, "name", str(g)))[0]  # the band indexes library rows; a twin is its row's pinch
            gi = names.index(gname) if gname in names else -1
            rows.append((i, yi, gi))
        p = bands.pairs.get(k)
        p = np.zeros((0, 3), dtype=np.int32) if p is None else p
        bands.pairs[k] = np.concatenate([p, np.array(rows, dtype=np.int32).reshape(-1, 3)]) if rows else p
        fs.append({"trunk": list(k), "cell": [float(v) for v in cell_base], "feasible": [[round(float(y), 3), C.split_twin(getattr(g, "name", str(g)))[0]] for y, g in F]})
    if bands.path:
        try:
            arrays = {RB.trunk_key(tr, ti): np.asarray(c, dtype=np.float32).reshape(-1, 2) for (tr, ti), c in bands.cells.items()}
            for (tr, ti), pp in bands.pairs.items():
                arrays[RB.trunk_key(tr, ti) + "__pairs"] = np.asarray(pp, dtype=np.int32).reshape(-1, 3)
            # train-fast 2026-10-31: atomic (per-process temp file + rename) - several train jobs of one template write the same band file,
            # and a torn .npz / .json would break the next job's band load (the write-back itself is the base code's)
            tmp_npz = f"{bands.path[:-4]}.tmp{os.getpid()}.npz"
            np.savez_compressed(tmp_npz, **arrays)
            tmp_json = f"{bands.path[:-4]}.json.tmp{os.getpid()}"
            with open(tmp_json, "w") as f:
                json.dump(bands.meta, f, indent=1, default=str)
            os.replace(tmp_npz, bands.path)
            os.replace(tmp_json, bands.path[:-4] + ".json")
        except OSError as e:  # noqa: BLE001 - a read-only cache is not a sampling error
            log.debug("band write-back skipped (%r)", e)


# ----------------------------------------------------------------------------------------------------------------
# A1 — left-arm contact bands, per-cell approach classes and the joint two-arm placement (owner decision 2026-09-16;
# spec §4.1 band scope, §4.4 margins = union of both bands, §4.6 step 6 two-arm IK)
# ----------------------------------------------------------------------------------------------------------------

#: approach pitches of the generic push / stop set (deg from straight down toward the robot's forward), B3's
#: ``TILTS_GENERIC``: the owner's A1 decision added 45° and 60° so far cells (0.9–1.0 m) keep an IK-feasible arc. B3's
#: per-cell class bitmask (``<trunk>__classes`` in the ``.npz``, names in ``meta["classes"]``) names the generic classes
#: ``p<pitch>_y<wrist yaw>`` (pitch-major) and the left contact classes ``near|far|left|right_v|h`` / ``top_y0|y90``.
APPROACH_PITCHES_DEG: Tuple[float, ...] = tuple(float(t) for t in getattr(RB, "TILTS_GENERIC", (0.0, 30.0, 45.0, 60.0)))
#: B3's contact set of the brace arm (``reach_bands.CONTACT_SET``): closed gripper, pads on a vertical face at 0.75 of
#: the object's height (``CONTACT_HEIGHT_FRACTION``) or a top press; one band per 2 cm object height class
CONTACT_SET = getattr(RB, "CONTACT_SET", "generic_contact")
CONTACT_HEIGHT_FRACTION = float(getattr(RB, "CONTACT_HEIGHT_FRACTION", 0.75))
#: a point is "in a band cell" when it lies within this fraction of the lattice from a cell centre (half a cell = the
#: cell's own square)
CELL_TOLERANCE = 0.5
#: brace-point reach proxy used while no left contact band exists (base frame, m): the measured band envelopes of the
#: B3 run (x 0.48–0.80 m ahead, |λ| <= 0.56) with a small allowance
BRACE_PROXY_X = (0.30, 0.88)
BRACE_PROXY_LATERAL = 0.65


def height_class(height: float) -> float:
    """B3's object height class (2 cm bins, the class value = the rounded height) of the left contact bands."""
    if hasattr(RB, "height_class"):
        return float(RB.height_class(float(height)))
    return round(max(1, int(round(float(height) / 0.02))) * 0.02, 4)


def contact_height(top_z: float, height: float) -> float:
    """Height of a side contact of the brace arm on an object ``height`` tall standing on ``top_z`` (B3's contact set:
    pads at ``CONTACT_HEIGHT_FRACTION`` of the height)."""
    return float(top_z) + CONTACT_HEIGHT_FRACTION * float(height)


def margins_union() -> bool:
    """§4.4 / A1 "margins use the union of both bands' extents": measured on the pilot support the union
    ([−0.56, +0.56] on `breakfast_table_rhjoby_0`) makes every two-arm task ``edge_too_short`` (block design m_R + m_L
    = 1.21 m, axial 1.46 m > L = 1.16 m), while the object is placed at right-band cells only — the left band decides
    which of those placements are valid (joint test), it does not widen where objects go. Default: the right band's
    extent; ``ISG_MARGIN_UNION=1`` applies the union as decided (owner to confirm)."""
    return os.environ.get("ISG_MARGIN_UNION", "0").strip().lower() in ("1", "true", "yes")


def program_check_stub() -> bool:
    """Smoke-only: ``ISG_PROGRAM_CHECK=stub`` set inside the smoke command lets a task module record a program-check
    rejection that only asks for an optional block (axial stop brace, block braces) as a stub mode instead of rejecting
    the instance, until the oracle program makes those blocks optional. Nothing else switches it (no repo markers)."""
    return os.environ.get("ISG_PROGRAM_CHECK", "").strip().lower() == "stub"


def left_band_compute() -> bool:
    """Whether a missing left contact band is computed in-process (15–25 min per support / height class on an L40S):
    ``ISG_LEFT_BAND_COMPUTE=1``; default off — until B3's left bands exist the brace side is decided by the exact two-arm
    IK alone and the instance records ``left_band: false``."""
    return os.environ.get("ISG_LEFT_BAND_COMPUTE", "0").strip().lower() in ("1", "true", "yes")


def left_contact_object(cls: float) -> str:
    """The ``meta.object`` token under which an in-process left contact band is cached (distinct from every real object
    so that a loader without the arm filter never mistakes it for a right-arm generic band)."""
    return f"contact-left/h{cls:.2f}"


def load_left_contact_band(s, height: float, *, top=None, edge: Optional[EdgeFrame] = None, compute: bool = True) -> Optional[RB.ReachBands]:
    """A1: the left arm's generic CONTACT band (closed gripper, brace approach) of the support for the object's height
    class, cached on ``s.bands_left[cls]``. Loader order: B3's left-arm mode
    ``ReachBands.load(scene, support, arm="left", contact_set="generic_contact", height=h)``; a loader that predates the
    keywords (``TypeError``) or has no such band (``FileNotFoundError``) falls back to the in-process band of
    ``compute_bands`` with ``BandSpec(arm="left", generic_pinch=True, contact=True)`` (B3's ``BandComputer`` contact
    mode when it has one — closed fingers, the contact classes —, else the generic top-down / pitched set with the left
    arm) under the object token ``left_contact_object(cls)`` with the meta fields B3's ``find`` matches (``arm``,
    ``contact_set``, ``height_class``). On an interior surface (``s.surface_z``, :func:`support_surface`) every lookup asks for
    that surface's band (``reach_bands --surface-z --generic-contact``; ``find`` matches the surface on the meta) and a miss is
    not computed in-process (as :func:`ensure_bands`: the board above and the side walls must stay obstacles) - None.
    ``compute=False`` returns None on a miss (unit tests, dry runs)."""
    cls = height_class(height)
    cache = getattr(s, "bands_left", None)
    if cache is None:
        cache = {}
        try:
            s.bands_left = cache
        except AttributeError:  # a sampler without the attribute slot: no caching
            pass
    if cls in cache:
        return cache[cls]
    top = _top(s, top)
    scene, support = s.world.scene_model, top.name
    surface = surface_band_kwargs(s)
    bands = None
    try:
        bands = RB.ReachBands.load(scene, support, arm="left", contact_set=CONTACT_SET, height=float(height), **surface)
        log.info("left contact band cache hit for %s (height class %.2f): %s", support, cls, bands.path)
    except TypeError:
        log.info("reach_bands loader has no left-arm mode yet: using the in-process left contact band")
        try:
            bands = RB.ReachBands.load(scene, support, left_contact_object(cls), "generic")
        except FileNotFoundError:
            bands = None
    except FileNotFoundError:
        # the B3 manifest computes the classes of its own object-height table (band metas' object_extent): a neighbouring
        # class stands in when the exact 2 cm class is missing (contact height differs by <= a few cm)
        try:
            bands = RB.ReachBands.load(scene, support, arm="left", contact_set=CONTACT_SET, height=None, **surface)
            near = bands.meta.get("height_class")
            if near is not None and abs(float(near) - cls) > 0.04 + 1e-6:
                log.info("left contact band of %s: nearest class %.2f is > 4 cm from %.2f; not used", support, float(near), cls)
                bands = None
            else:
                log.info("left contact band for %s: class %.2f stands in for %.2f (%s)", support, float(near) if near is not None else float("nan"), cls, bands.path)
        except (FileNotFoundError, TypeError):
            log.info("no B3 left contact band for %s (height class %.2f%s)", support, cls, f", surface_z {surface['surface_z']:.2f}" if surface else "")
    if bands is None and compute:
        if surface:
            log.info("left contact band of the interior surface of %s at surface_z %.2f: not computed in-process", support, surface["surface_z"])
        else:
            spec = BandSpec(object=left_contact_object(cls), generic_pinch=True, height=cls, arm="left", contact=True)
            bands = compute_bands(s, spec, top, edge)
    cache[cls] = bands
    return bands


def cell_classes(bands: Optional[RB.ReachBands], translate: float, tilt: float, cell_base, stand_off: Optional[float] = None) -> Optional[List[str]]:
    """B3's per-cell pose classes that pass at ``cell_base`` (base frame) of the trunk (``ReachBands.classes_at``:
    names in bit order of ``meta["classes"]`` from the ``<trunk>__classes`` array); None when the band records no
    classes (library-grasp bands, bands computed before A1) or the cell is not in the band. ``stand_off``: a multi-distance
    band's nearest stored stand-off (:func:`band_stand_off`; None = the reference maps)."""
    if bands is None or cell_base is None:
        return None
    names = bands.meta.get("classes") if isinstance(bands.meta, dict) else None
    so = {} if stand_off is None else {"stand_off": float(stand_off)}
    masks = bands._maps(**so)[2] if so and hasattr(bands, "_maps") else getattr(bands, "classes", None)
    if not names or not masks:
        return None
    k = bands.nearest_trunk(float(translate), float(tilt), **so)
    if k is None or k not in masks:
        return None
    i = bands.cell_index(k[0], k[1], np.asarray(cell_base, dtype=np.float32), **so)
    if i is None or i >= len(masks[k]):
        return None
    m = int(masks[k][i])
    return [n for b, n in enumerate(names) if (m >> b) & 1]


def class_pitch_deg(name: str) -> Optional[float]:
    """Approach pitch of a generic class name ``p<pitch>_y<yaw>`` (None for other classes)."""
    if not name or not name.startswith("p") or "_" not in name:
        return None
    try:
        return float(int(name[1:].split("_", 1)[0]))
    except ValueError:
        return None


def approach_pitch_order(classes: Optional[Sequence[str]], pitches_deg: Sequence[float] = APPROACH_PITCHES_DEG) -> List[float]:
    """The approach pitches (deg from straight down, forward) to try for a generic push / stop pose at a cell: the
    pitches of the cell's passing classes first (shallowest first), then the remaining ones so that the 45° / 60°
    classes are tried before the placement is rejected; without a recorded mask the plain order."""
    pitches = [float(p) for p in pitches_deg]
    if not classes:
        return pitches
    passing = sorted({p for p in (class_pitch_deg(c) for c in classes) if p is not None and p in pitches})
    return passing + [p for p in pitches if p not in passing]


def contact_class(stance: Stance, approach_world, closing_world) -> Optional[str]:
    """B3's contact class name of a brace pose: ``approach_world`` = the eef approach axis (into the face / down onto
    the top), ``closing_world`` = the closing axis; base frame of ``stance`` (x ahead, lateral + = robot's left).
    Side classes ``near`` (pads on the face toward the robot, approach +x), ``far`` (−x), ``left`` (approach −lateral: the
    hand on the object's robot-left face), ``right`` (+lateral) with ``_v`` (closing vertical) / ``_h`` (horizontal);
    ``top_y0`` / ``top_y90`` for a downward approach with the closing axis lateral / ahead."""
    a = _t(approach_world)
    c = _t(closing_world)
    fx, fy = float(stance.forward[0]), float(stance.forward[1])
    lx, ly = float(stance.left[0]), float(stance.left[1])
    ax, al, az = float(a[0]) * fx + float(a[1]) * fy, float(a[0]) * lx + float(a[1]) * ly, float(a[2])
    cx, cl, cz = float(c[0]) * fx + float(c[1]) * fy, float(c[0]) * lx + float(c[1]) * ly, float(c[2])
    if az < -0.9:
        return "top_y0" if abs(cl) >= abs(cx) else "top_y90"
    if abs(az) > 0.3:  # a pitched approach belongs to no contact class
        return None
    side = ("near" if ax > 0 else "far") if abs(ax) >= abs(al) else ("left" if al < 0 else "right")
    return f"{side}_{'v' if abs(cz) > 0.7 else 'h'}"


def pitched_approach(pitch_deg: float, forward: th.Tensor) -> th.Tensor:
    """Approach direction of the generic set: straight down rotated ``pitch_deg`` toward ``forward`` (the robot's
    forward, horizontal)."""
    z = th.tensor([0.0, 0.0, 1.0])
    f = th.tensor([float(forward[0]), float(forward[1]), 0.0])
    f = f / max(float(th.linalg.norm(f)), 1e-9)
    a = math.radians(float(pitch_deg))
    return C.unit(-z * math.cos(a) + f * math.sin(a))


def in_band(bands: Optional[RB.ReachBands], stance: Stance, xy, tol: float = CELL_TOLERANCE) -> Tuple[bool, Optional[np.ndarray]]:
    """Whether the world point ``xy`` lies within ``tol``·lattice of a cell of ``bands`` at the stance's trunk (same
    translate, nearest tilt); returns ``(ok, nearest cell in base coordinates)``. Neighbouring supports: the band of the top under
    ``xy``."""
    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(xy)
    if bands is None:
        return False, None
    cells = bands.band(float(stance.translate), float(stance.tilt), **band_kw(bands, stance))
    if len(cells) == 0:
        return False, None
    base = bands.to_base(np.asarray([[float(xy[0]), float(xy[1])]], dtype=np.float32), stance.base_pose2d)[0]
    d = np.linalg.norm(np.asarray(cells, dtype=np.float32) - base, axis=1)
    j = int(d.argmin())
    lat = float(bands.meta.get("lattice", RB.LATTICE)) if isinstance(bands.meta, dict) else RB.LATTICE
    return bool(d[j] <= tol * lat + 1e-6), np.asarray(cells[j], dtype=np.float32)


def joint_placement(bands_right: RB.ReachBands, bands_left: Optional[RB.ReachBands], stance: Stance, work_xy, brace_xy, tol: float = CELL_TOLERANCE,
                    brace_class: Optional[str] = None, tol_brace: Optional[float] = None) -> Dict[str, Any]:
    """A1 placement test for a candidate (target pose → work point of the right arm, brace point of the left arm):
    ``work_xy`` ∈ a right band cell AND ``brace_xy`` ∈ a left contact band cell **at the same trunk** (the stance's);
    with ``brace_class`` (B3 contact class of the intended brace pose, ``contact_class``) and a band that records the
    per-cell classes, the class must have passed at that cell. Without a left band (None) the brace side is left to
    the exact two-arm IK behind the reach proxy ``BRACE_PROXY_X`` / ``BRACE_PROXY_LATERAL`` and ``left_band`` is False
    so the caller can record it. The dict carries ``ok`` /
    ``work_ok`` / ``brace_ok`` / ``brace_class_ok``, the nearest cells (base frame), the work cell's ``classes`` and the
    brace cell's ``brace_classes`` (B3 bitmasks, None when absent)."""
    work_ok, work_cell = in_band(bands_right, stance, work_xy, tol)
    brace_classes = None
    class_ok = True
    if bands_left is None:
        # no left band yet: a reach proxy keeps the exact IK from being spent on hopeless brace points (the arms' bands
        # span x 0.48–0.80 m ahead and |λ| <= 0.56 on the pilot supports, README of the B3 run)
        bx, bl = (float(v) for v in stance.world_to_base(_t(brace_xy)[:2]).tolist())
        brace_ok, brace_cell = (BRACE_PROXY_X[0] <= bx <= BRACE_PROXY_X[1] and abs(bl) <= BRACE_PROXY_LATERAL), None
    else:
        brace_ok, brace_cell = in_band(bands_left, stance, brace_xy, tol if tol_brace is None else tol_brace)
        if brace_ok:
            brace_classes = cell_classes(bands_left, stance.translate, stance.tilt, brace_cell, band_stand_off(bands_left, stance))
            if brace_class is not None and brace_classes is not None:
                class_ok = brace_class in brace_classes
    return {"ok": bool(work_ok and brace_ok and class_ok), "work_ok": bool(work_ok), "brace_ok": bool(brace_ok), "brace_class_ok": bool(class_ok),
            "work_cell": work_cell, "brace_cell": brace_cell, "left_band": bands_left is not None,
            "classes": cell_classes(bands_right, stance.translate, stance.tilt, work_cell, band_stand_off(bands_right, stance)), "brace_classes": brace_classes, "brace_class": brace_class}


def joint_two_arm_placement(s, stance: Stance, bands_right: RB.ReachBands, bands_left: Optional[RB.ReachBands], work_xy, brace_xy, work_arm: str,
                            work_poses: Sequence[Pose], brace_arm: str, brace_poses: Sequence[Pose], *, ignore_work_last: Sequence[Any] = (),
                            ignore_brace_last: Sequence[Any] = (), tol: float = CELL_TOLERANCE, brace_class: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """A1 in one call: the band test (``joint_placement``) first, then the exact two-arm IK of §4.6 step 6
    (``brace_second_arm``: work arm poses, brace arm [pre-contact, contact] with the work arm locked, combined whole-body
    check). Returns ``(ok, info)`` with ``info["reason"]`` = ``"work_off_band"`` / ``"brace_off_band"`` /
    ``"brace_class_failed"`` / the IK reason."""
    info = joint_placement(bands_right, bands_left, stance, work_xy, brace_xy, tol, brace_class=brace_class)
    if not info["work_ok"]:
        info["reason"] = "work_off_band"
        return False, info
    if not info["brace_ok"]:
        info["reason"] = "brace_off_band"
        return False, info
    if not info["brace_class_ok"]:
        info["reason"] = "brace_class_failed"
        return False, info
    res = brace_second_arm(s, stance, work_arm, work_poses, brace_arm, brace_poses, ignore_work_last=ignore_work_last, ignore_brace_last=ignore_brace_last)
    info["ik"] = res
    info["reason"] = None if res.ok else f"ik: {res.reason}"
    return bool(res.ok), info


# ----------------------------------------------------------------------------------------------------------------
# closed-finger profile and blade stand-offs (owner review 2026-09-16 of the axial demo: the closed finger is a wedge,
# not a 24 mm slab — measured on r1pro/urdf/meshes/left_gripper_finger_link1.obj in the closed configuration)
# ----------------------------------------------------------------------------------------------------------------

#: outer face of the closed fingers from the closing plane (the eef origin along the closing axis) as a function of the
#: distance along the finger above the fingertip: ``(dz_max, half thickness)`` rows, the last row ends where the palm /
#: mounting bracket begins (far wider: no blade contact possible above it)
FINGER_PROFILE: Tuple[Tuple[float, float], ...] = ((0.03, 0.0084), (0.07, 0.018))
#: the fingers' half width across the closing axis (the finger's side pushes when the closing axis is set across the
#: push direction), same sections
FINGER_SIDE_PROFILE: Tuple[Tuple[float, float], ...] = ((0.03, 0.007), (0.07, 0.014))
PALM_FROM_TIP = FINGER_PROFILE[-1][0]
#: margin of the brace configuration against the target at its goal pose (§4.6 step 6 with the target not ignored)
BRACE_TARGET_MARGIN = 0.003


def finger_half_thickness(dz: float, side: bool = False) -> Optional[float]:
    """Half thickness of the closed fingers ``dz`` m above the fingertip along the finger (outer face from the closing
    plane; ``side``: half width across the closing axis); None at / above the palm section."""
    prof = FINGER_SIDE_PROFILE if side else FINGER_PROFILE
    if dz < 0.0:
        dz = 0.0
    for dz_max, half in prof:
        if dz <= dz_max + 1e-9:
            return float(half)
    return None


def blade_standoff(face_z_range: Tuple[float, float], tip_z: float, pitch_deg: float = 0.0, clearance: float = 0.0, side: bool = False) -> Optional[Tuple[float, float]]:
    """Stand-off of a blade contact (closed gripper beside an object face, approach from above pitched ``pitch_deg``):
    the eef origin sits ``half + clearance`` outside the face along the closing axis, ``half`` = the largest finger half
    thickness over the height range the face spans above the fingertip (finger-axis distance = height / cos pitch).
    Returns ``(stand-off, half)``; None when the face reaches the palm section (no blade contact — use a side contact)."""
    z0, z1 = float(min(face_z_range)), float(max(face_z_range))
    if z1 <= tip_z:
        return None  # the face lies entirely below the fingertip: nothing to touch
    c = max(math.cos(math.radians(float(pitch_deg))), 1e-3)
    dz_lo, dz_hi = max(0.0, z0 - float(tip_z)) / c, (z1 - float(tip_z)) / c
    prof = FINGER_SIDE_PROFILE if side else FINGER_PROFILE
    half = None
    for dz_max, h in prof:
        if dz_lo <= dz_max + 1e-9:
            half = float(h) if half is None else max(half, float(h))
        if dz_hi <= dz_max + 1e-9:
            break
    else:
        if dz_hi > prof[-1][0] + 1e-9:
            return None
    if half is None:
        return None
    return half + float(clearance), half


#: the hand above the finger section, eef frame: the gripper link's cuRobo spheres (palm body r 4 cm, four knuckles r 1.5 cm;
#: ``r1pro_description_curobo_default.yaml`` mapped through ``EEF_IN_GRIPPER`` as ``iors_small.HAND_SPHERES_EEF``, the same for both
#: arms) - with the wrist camera's (``WRIST_CAMERA_SPHERES_EEF``) what meets a face that rises past the fingers
PALM_SPHERES_EEF: Tuple[Tuple[Tuple[float, float, float], float], ...] = (
    ((0.0, 0.0, -0.097), 0.04),
    ((-0.015, -0.042, -0.0675), 0.015), ((0.015, -0.042, -0.0675), 0.015), ((-0.015, 0.042, -0.0675), 0.015), ((0.015, 0.042, -0.0675), 0.015),
)


def palm_half_extent(eef_quat, direction) -> float:
    """How far the hand above the finger section (``PALM_SPHERES_EEF`` and the wrist camera's spheres) reaches from the eef origin
    along ``direction`` (world) at the eef orientation ``eef_quat``: the largest ``centre · u + radius``, ``u`` = ``direction`` in the
    eef frame. A vertical blade closing along the push: 5.7 cm (the knuckles) with the camera turned away from the face, 7.5 cm (the
    camera) toward it; closing across the push: 4.0 cm (the palm body) / 7.4 cm (the camera)."""
    u = C.quat_to_mat(_t(eef_quat)).T @ C.unit(_t(direction).float())
    spheres = list(PALM_SPHERES_EEF) + [(c, WRIST_CAMERA_SPHERE_R) for c in WRIST_CAMERA_SPHERES_EEF]
    return max(float(th.dot(_t(c), u)) + float(r) for c, r in spheres)


# ----------------------------------------------------------------------------------------------------------------
# flat hand (owner 2026-10-12 ~19:40, under_large #02 / #03 push-under): the hand turned flat, the closed fingers pointing
# forward along the push - the pitched blade at 90 deg; only the fingertips touch the object's back face
# ----------------------------------------------------------------------------------------------------------------


def flat_hand_quats(direction) -> List[Tuple[str, th.Tensor]]:
    """eef orientations of a flat hand pushing along the horizontal ``direction``: approach = the push direction
    (``pitched_approach(90, direction)``), the closing axis lateral (horizontal, across the push) or vertical - the two wrist turns -,
    each with its twin (the closing axis negated: the wrist camera on the other side). ``[(label, quat), ...]`` labelled
    ``lateral``, ``lateral-``, ``vertical``, ``vertical-``."""
    a = pitched_approach(90.0, direction)
    up = th.tensor([0.0, 0.0, 1.0])
    out = []
    for name, c in (("lateral", C.unit(th.cross(up, a, dim=0))), ("vertical", up)):
        for sign in (1.0, -1.0):
            out.append((name if sign > 0 else f"{name}-", C.eef_quat(a, sign * c)))
    return out


def flat_hand_standoff(clearance: float = 0.0) -> float:
    """Stand-off of a flat hand: the fingertips touch, so the eef origin stands ``FINGERTIP_LENGTH + clearance`` behind the contact
    point along the push."""
    return C.FINGERTIP_LENGTH + float(clearance)


def hand_drop_below(eef_quat) -> float:
    """How far the hand's lowest collision sphere lies below the eef origin at the eef orientation ``eef_quat``: ``palm_half_extent``
    downward (the gripper body, the knuckles, the wrist camera; the closed fingers and ``arm_link7`` stay above the gripper body's sphere
    for a horizontal approach)."""
    return palm_half_extent(eef_quat, th.tensor([0.0, 0.0, -1.0]))


def flat_tip_half_height(quat) -> float:
    """Half height of the closed fingertips at orientation ``quat`` (the band they press on a vertical face): the fingers' half thickness
    along the closing axis when it is vertical, their half width across it when it is horizontal (``FINGER_PROFILE`` /
    ``FINGER_SIDE_PROFILE`` at the tip)."""
    c = C.closing_axis(_t(quat))
    return float(FINGER_PROFILE[0][1]) if abs(float(c[2])) >= 0.5 else float(FINGER_SIDE_PROFILE[0][1])


# ----------------------------------------------------------------------------------------------------------------
# pitched hand (owner 2026-10-12 ~20:55 item C, under_large #02 / #03 push-under): the closed hand pitched 60-70 deg from straight
# down toward the push, the palm leaning away from the face, the fingertips low on the object's back face
# ----------------------------------------------------------------------------------------------------------------

#: the closed fingers' cuRobo spheres in the eef frame, both fingers (``r1pro_description_curobo_default.yaml`` through the URDF with the
#: finger joints at 0, the same for both arms; ``collision_sphere_buffer`` not included): a pitched hand's lowest part can be a fingertip
FINGER_SPHERES_EEF: Tuple[Tuple[Tuple[float, float, float], float], ...] = tuple(
    ((0.0, s * y, z), r) for s in (1.0, -1.0) for y, z, r in ((0.0085, -0.0381, 0.010), (0.0075, -0.0181, 0.008), (0.0055, -0.0031, 0.007), (0.0035, 0.0119, 0.005)))


def hand_spheres_eef() -> List[Tuple[str, Tuple[float, float, float], float]]:
    """The hand's collision spheres in the eef frame, ``(part, centre, radius)``: the gripper body and the knuckles (``PALM_SPHERES_EEF``),
    the wrist camera and the closed fingers (``FINGER_SPHERES_EEF``)."""
    parts = ["gripper body"] + ["knuckle"] * (len(PALM_SPHERES_EEF) - 1)
    return ([(p, c, r) for p, (c, r) in zip(parts, PALM_SPHERES_EEF)] + [("wrist camera", c, WRIST_CAMERA_SPHERE_R) for c in WRIST_CAMERA_SPHERES_EEF]
            + [("finger", c, r) for c, r in FINGER_SPHERES_EEF])


def pitched_hand_quats(pitch_deg: float, direction) -> List[Tuple[str, th.Tensor]]:
    """eef orientations of the hand pushing along the horizontal ``direction`` pitched ``pitch_deg`` from straight down toward it
    (``pitched_approach``: the fingertips ahead of and below the palm, the palm leaning away from the face): the closing axis lateral with the
    wrist camera on the hand's upper side (``lateral``; eef +x = c x a points back and down) or below it (``lateral-``), or in the push's
    vertical plane (``plane``: the camera on the push's right, ``plane-``: on its left). ``[(label, quat), ...]`` in that order."""
    a = pitched_approach(pitch_deg, direction)
    lat = C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), a, dim=0))
    n = C.unit(th.cross(a, lat, dim=0))  # across the finger axis in the push's vertical plane, toward the face and up
    return [("lateral", C.eef_quat(a, lat)), ("lateral-", C.eef_quat(a, -lat)), ("plane", C.eef_quat(a, n)), ("plane-", C.eef_quat(a, -n))]


def pitched_hand_drop(eef_quat) -> float:
    """How far the hand's lowest collision sphere (:func:`hand_spheres_eef`) lies below the fingertips - the point ``FINGERTIP_LENGTH``
    along the approach from the eef origin - at the eef orientation ``eef_quat``: a fingertip sphere for a pitched hand with the camera up,
    a knuckle with the closing axis in the push plane, the camera with it below."""
    R = C.quat_to_mat(_t(eef_quat)).float()
    tip_z = C.FINGERTIP_LENGTH * float(R[2, 2])
    return tip_z - min(float(R[2] @ _t(c)) - float(r) for _, c, r in hand_spheres_eef())


def pitched_tip_offsets(pitch_deg: float, eef_quat, clearance: float = 0.0) -> Tuple[float, float, float]:
    """The pitched hand against a vertical face: the fingertips' leading edge - ``h`` (:func:`flat_tip_half_height`, the tips' half extent
    across the finger axis in the push plane) toward the face and up from the fingertip point - touches ``clearance`` short of it. Returns
    ``(stand-off, rise, h)``: the eef origin ``FINGERTIP_LENGTH·sin p + h·cos p + clearance`` behind the contact point along the push,
    ``FINGERTIP_LENGTH·cos p`` above the fingertip point (which the leading edge stands ``h·sin p`` above)."""
    p = math.radians(float(pitch_deg))
    h = flat_tip_half_height(eef_quat)
    return C.FINGERTIP_LENGTH * math.sin(p) + h * math.cos(p) + float(clearance), C.FINGERTIP_LENGTH * math.cos(p), h


def blade_sweep_clear(hull: th.Tensor, pose_init: Pose, com_xy, delta: float, blade_xy, radius: float, clearance: float, n: int = 12) -> Tuple[bool, float]:
    """Whether a blade (disc of ``radius`` at ``blade_xy``) stays ``clearance`` clear of the object's footprint while it
    rotates about ``com_xy`` from ``pose_init`` by ``delta`` (rad, about +z) — sampled at ``n`` angles short of the goal
    (the goal contact itself is the designed stop). Returns ``(ok, min gap before the goal)``."""
    from ispatialgym.sampling import geometry as G

    pts = C.transform_points(pose_init, hull)[:, :2]
    c = th.tensor([float(com_xy[0]), float(com_xy[1])])
    b = th.tensor([float(blade_xy[0]), float(blade_xy[1])])
    square = th.stack([b + th.tensor([sx * radius, sy * radius]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    worst = math.inf
    for k in range(n):
        th_k = float(delta) * (k / n)  # 0 ... (n-1)/n of the rotation: strictly before the goal
        cs, sn = math.cos(th_k), math.sin(th_k)
        rel = pts - c
        rot = th.stack([cs * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + cs * rel[:, 1]], dim=1) + c
        gap = polygon_gap(G.convex_hull_2d(rot), square)
        worst = min(worst, gap)
        if gap < clearance:
            return False, worst
    return True, worst


@dataclass
class BraceRecord(ContactRecord):
    """A brace ``ContactRecord`` with the blade / stop geometry the oracle and the audit need (spec §7 extras):
    ``blade_half_thickness_m`` (finger profile value used), ``stop_clearance_m``, ``pose_class``, ... — merged into the
    sidecar dict (the F6 ``BraceContact.extra`` keeps unknown keys)."""

    extras: Dict[str, Any] = field(default_factory=dict)

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d.update({k: v for k, v in self.extras.items() if v is not None})
        return d


def union_extent(*bands: Optional[RB.ReachBands]) -> Optional[Tuple[float, float]]:
    """§4.4 (A1): the lateral extent [λ_min, λ_max] of the union of the given bands (the right grasp band and the left
    contact band), None when none has cells."""
    lo, hi = math.inf, -math.inf
    for b in bands:
        if b is None:
            continue
        ext = b.union_extent() if hasattr(b, "union_extent") else None
        if ext is None:
            continue
        lo, hi = min(lo, float(ext[0])), max(hi, float(ext[1]))
    return None if lo > hi else (lo, hi)


# ----------------------------------------------------------------------------------------------------------------
# camera helpers (§3.4)
# ----------------------------------------------------------------------------------------------------------------


def image_border_lateral(s, cam_pose: Pose, stance: Stance, depth: float, z: float, margin: float = IN_FRAME_MARGIN) -> Tuple[float, float]:
    """``(b_right, b_left)``: lateral distance (m, along the edge tangent) from the robot's centre line to the point where
    the image's right / left border (with the ``margin`` fraction) meets the plane ``z`` at ``depth`` from the front edge
    (§4.3 b: "lateral offset of the image border at the source-tray depth"; §4.6 "border at tray depth")."""
    cm = s.world.camera_model
    p = stance.world(float(depth), 0.0)
    pt = th.tensor([[float(p[0]), float(p[1]), float(z)]])
    uv, dep = cm.project(cam_pose, pt)
    v = float(uv[0, 1])
    out = []
    for u in (cm.width - margin * cm.width, margin * cm.width):  # image right = robot's right (facing forward), image left = robot's left
        hit = cm.ray_plane_intersection(cam_pose, u, v, float(z))
        if hit is None:
            out.append(float("inf"))
            continue
        _, lat = stance.local(hit[:2])
        out.append(abs(lat))
    return out[0], out[1]


def r_img(s, cam_pose: Pose, stance: Stance, target_xy, z: float, side: float, margin: float = IN_FRAME_MARGIN) -> float:
    """§4.3 R_img: surface distance from the target to the image border on ``side`` (+1 = robot's left, −1 = right)."""
    depth, lat = stance.local(target_xy)
    b_r, b_l = image_border_lateral(s, cam_pose, stance, depth, z, margin)
    return max(0.0, (b_l - lat) if side > 0 else (b_r + lat))


def in_frame(s, cam_pose: Pose, points: th.Tensor, margin: float = IN_FRAME_MARGIN) -> bool:
    return bool(s.world.camera_model.all_inside(cam_pose, points, margin))


#: images ``frame_diag`` renders per process (the JSON line is written for every rejection)
FRAME_DIAG_IMAGES = 6


def frame_diag(s, stance: Stance, arm: str, eef_pose: Pose, what: str, obj: Any = None, obj_pose: Optional[Pose] = None,
               cam_pose: Optional[Pose] = None, tag: str = "") -> Optional[Dict[str, Any]]:
    """Opt-in ``ISPATIALGYM_FRAME_DIAG=<dir>`` (of-v2maps 2026-11-06, the plate's gripper_not_in_frame): at a frame rejection of the eef
    pose ``eef_pose`` from ``stance``, one ``FRAME_DIAG {json}`` line - the image coordinates (u, v), depth and in-image / in-margin
    flags of the 8 corners ``IKChecker.eef_in_frame`` tests (a 12 x 12 x 8 cm box around the eef: +-6 cm across, -2..+6 cm along the
    approach) and of the two finger links (by FK of the IK solution at the pose, else the stance posture), whether the IK found one,
    and per point the distance (px) past the 5 % border. For the first :data:`FRAME_DIAG_IMAGES` rejections of a process whose world
    has the head camera (``ISPATIALGYM_FAST_SAMPLER_CAMERAS=0``), the robot is posed at the stance with that configuration and ``obj``
    at ``obj_pose``, the head camera rendered, the points and the border drawn, ``<dir>/<variant>_<n>_<what>.png`` saved, and the
    robot / object put back (no physics step between). Never fails the caller."""
    out_dir = os.environ.get("ISPATIALGYM_FRAME_DIAG")
    if not out_dir:
        return None
    try:
        from ispatialgym.sampling import geometry as G

        cam = cam_pose if cam_pose is not None else s.camera_pose_at(stance)
        cm = s.world.camera_model
        p, q = _t(eef_pose[0]).float(), _t(eef_pose[1]).float()
        names, pts = [], []
        for dx in (-0.06, 0.06):
            for dy in (-0.06, 0.06):
                for dz in (-0.02, 0.06):
                    names.append(f"box{'+' if dx > 0 else '-'}{'+' if dy > 0 else '-'}{'a' if dz > 0 else 'b'}")
                    pts.append(p + G.rotate(q, th.tensor([dx, dy, dz])))
        r = s.ikc.exact([eef_pose], stance, arm, cache=False)
        q_full = r.q if (r.ok and r.q is not None) else s.ikc.q_locked(stance)
        for ln in (f"{arm}_gripper_finger_link1", f"{arm}_gripper_finger_link2"):
            try:
                fk = s.kit.backend.fk([ln], q_full)[ln]
                names.append(ln.rsplit("_", 1)[-1])
                pts.append(_t(fk[0]).float())
            except Exception:  # noqa: BLE001
                pass
        P = th.stack(pts)
        uv, depth = cm.project(cam, P)
        W, H = float(cm.width), float(cm.height)
        mu, mv = IN_FRAME_MARGIN * W, IN_FRAME_MARGIN * H
        rows = []
        for i, nm in enumerate(names):
            u, v, d = float(uv[i, 0]), float(uv[i, 1]), float(depth[i])
            past = max(mu - u, u - (W - mu), mv - v, v - (H - mv), 0.0)
            rows.append({"pt": nm, "u": round(u, 1), "v": round(v, 1), "depth": round(d, 3), "in_image": bool(d > 0.05 and 0 <= u <= W and 0 <= v <= H),
                         "in_margin": bool(d > 0.05 and mu <= u <= W - mu and mv <= v <= H - mv), "past_border_px": round(past, 1)})
        n = s.__dict__.get("_frame_diag_n", 0)
        s._frame_diag_n = n + 1
        rec = {"what": what, "tag": tag, "n": n, "arm": arm, "stance": [round(float(v), 3) for v in stance.base_pose2d], "trunk": [stance.translate, round(stance.tilt, 3)],
               "ik_ok": bool(r.ok), "image": [int(W), int(H)], "box_in_margin": sum(x["in_margin"] for x in rows[:8]), "box_in_image": sum(x["in_image"] for x in rows[:8]),
               "fingers_in_margin": sum(x["in_margin"] for x in rows[8:]), "max_past_border_px": max(x["past_border_px"] for x in rows), "points": rows}
        if n < FRAME_DIAG_IMAGES and getattr(s.world, "cam", None) is not None:
            rec["png"] = _frame_diag_image(s, stance, q_full, obj, obj_pose, uv, rows, mu, mv, W, H, out_dir, n, what)
        log.info("FRAME_DIAG %s", json.dumps(rec))
        return rec
    except Exception as e:  # noqa: BLE001 - a diagnostic never fails an attempt
        log.info("FRAME_DIAG failed: %r", e)
        return None


def _frame_diag_image(s, stance: Stance, q_full: th.Tensor, obj: Any, obj_pose: Optional[Pose], uv: th.Tensor, rows: List[Dict[str, Any]],
                      mu: float, mv: float, W: float, H: float, out_dir: str, n: int, what: str) -> Optional[str]:
    """:func:`frame_diag`'s render: the robot at ``stance`` with ``q_full``, ``obj`` at ``obj_pose``, the head camera's rgb with the projected
    points (green inside the 5 % border, red outside) and the border; the robot and the object restored afterwards."""
    from PIL import Image, ImageDraw

    robot = s.world.robot
    q0 = robot.get_joint_positions().clone()
    o0 = obj.get_position_orientation() if obj is not None else None
    try:
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        robot.set_joint_positions(_t(s.kit.q_at_base_pose(_t(q_full).float(), x, y, yaw)))
        if obj is not None and obj_pose is not None:
            obj.set_position_orientation(_t(obj_pose[0]).float(), _t(obj_pose[1]).float())
        img = s.world.rgb(n_renders=3).numpy().astype("uint8")
    finally:
        robot.set_joint_positions(q0)
        if o0 is not None:
            obj.set_position_orientation(*o0)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([mu, mv, W - mu, H - mv], outline=(255, 255, 0), width=2)
    for i, row in enumerate(rows):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        col = (0, 220, 0) if row["in_margin"] else (255, 0, 0)
        if row["pt"] in ("link1", "link2"):
            col = (0, 120, 255) if row["in_margin"] else (255, 0, 255)
        dr.ellipse([u - 5, v - 5, u + 5, v + 5], outline=col, width=3)
        dr.text((u + 6, v - 6), row["pt"], fill=col)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{getattr(getattr(s, 'gp', None), 'variant', 'x')}_{os.getpid()}_{n}_{what}.png")
    im.save(path)
    return path


def all_in_frame(s, cam_pose: Pose, items: Sequence[Tuple[str, th.Tensor, Pose]], margin: float = IN_FRAME_MARGIN, skip: Sequence[str] = ()) -> List[str]:
    """§4.6 step 4: names of the ``(name, hull, pose)`` items whose AABB corners are not fully inside the image (``margin``);
    ``skip`` names are exempt (the disembedding destination tray)."""
    out = []
    for name, hull, pose in items:
        if name in skip:
            continue
        if not in_frame(s, cam_pose, hull_corners_world(hull, pose), margin):
            out.append(name)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Phase 1a — edge and stand-off (§4.2)
# ----------------------------------------------------------------------------------------------------------------


def row_edges(s, top) -> List[str]:
    """The edges the selection's support row keeps for the stance (the ``support_surfaces`` entry's ``edges``, sheet note ``edges=``,
    ``ground_isg.parse_edges``); [] = every edge."""
    try:
        spec = support_surface_spec(s, getattr(top, "name", top))
    except AttributeError:  # a sampler without a selection
        return []
    return [str(e) for e in (spec or {}).get("edges") or []]


def gen_edges(s) -> List[str]:
    """Opt-in ``generation.init_edges`` (oracle-fix 2026-11-06, block_design#01): the edges Phase 1a may stand at, applied after the
    support row's (:func:`row_edges`). The support row lives in the template's bindings (``multiply_isg`` rebuilds the choice from
    ``template_meta.json``, so a customs edit of the row never reaches the fast flow), the generation block in the customs the flow
    loads; [] = every edge."""
    edges = s.gen("init_edges", None) if hasattr(s, "gen") else None
    return [str(e) for e in edges or []]


def edge_frames(s, top, rng=None, stand_off: float = STAND_OFF) -> List[Tuple[EdgeFrame, Stance]]:
    """Every edge of the top (random order with ``rng``) with its stance at ``stand_off`` + overhang (>= 5 cm), trunk
    (0.5, 0) — before the floor / collision checks of Phase 1a. A support row naming ``edges`` (:func:`row_edges`) keeps only those,
    and ``generation.init_edges`` (:func:`gen_edges`) those of them it names."""
    support = s.world.obj(top.name)
    lo, hi = support.aabb
    aabb_xy = th.tensor([[float(lo[0] if sx == 0 else hi[0]), float(lo[1] if sy == 0 else hi[1])] for sx in (0, 1) for sy in (0, 1)])
    edges = list(top.edges())
    open_edges = list(getattr(top, "open_edges", None) or [])
    if open_edges:  # an interior surface is reached through the open sides of its compartment only
        edges = [e for e in edges if e[0] in open_edges]
    keep = row_edges(s, top)
    if keep:
        log.info("phase 1a: the support row keeps edge(s) %s of %s", keep, [e[0] for e in edges])
        edges = [e for e in edges if e[0] in keep]
    keep_gen = gen_edges(s)
    if keep_gen:
        log.info("phase 1a: generation.init_edges keeps edge(s) %s of %s", keep_gen, [e[0] for e in edges])
        edges = [e for e in edges if e[0] in keep_gen]
    order =[int(i) for i in rng.permutation(len(edges))] if rng is not None else list(range(len(edges)))
    out = []
    for i in order:
        name, mid, normal, length = edges[i]
        n = _t(normal)[:2]
        overhang = max(0.0, float((aabb_xy @ n).max()) - float(th.dot(_t(mid)[:2], n)))
        edge = EdgeFrame.from_edge(name, mid, normal, length, overhang if overhang >= OVERHANG_MIN else 0.0)
        xy = edge.base_xy(stand_off, 0.0)
        st = Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, float(stand_off), edge, lateral=0.0)
        out.append((edge, st))
    return out


def shifted_init_stance(s, edge: EdgeFrame, stand_off: float) -> Optional[Stance]:
    """Opt-in ``generation.init_edge_laterals_m`` (oracle-fix 2026-11-03, alias 151 hardback-desk: the desk's only open edge has its
    midpoint stance against the goal bookcase): when the Phase-1a midpoint stance collides, the first of these shifts along the edge
    (within the edge, 0.30 m from its ends) whose stance is on floor and collision-free stands in for it. Phase 1c still draws the lateral."""
    lats = s.gen("init_edge_laterals_m", None) if hasattr(s, "gen") else None
    if not lats:
        return None
    half = 0.5 * float(edge.length) - 0.30
    for lat in (float(v) for v in lats):
        if abs(lat) > half:
            continue
        xy = edge.base_xy(stand_off, lat)
        if not s.kit.trav_free(xy):
            continue
        st = Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, float(stand_off), edge, lateral=lat)
        if s.ikc.stance_collision_free(st):
            log.info("phase 1a: edge %s midpoint stance collides; shifted %+.2f m along the edge (generation.init_edge_laterals_m) to %s",
                     edge.name, lat, [round(float(v), 2) for v in xy.tolist()])
            return st
    return None


def edge_candidates(s, top, rng=None, stand_off: float = STAND_OFF) -> Iterator[Tuple[EdgeFrame, Stance]]:
    """§4.2 survivors in random order: the edge stance is on free floor and whole-body collision-free at trunk (0.5, 0)
    with untucked arms. Task modules that retry edges (axial ``edge_not_visible``) iterate this."""
    skipped: Dict[str, str] = {}
    s._tabletop_edge_log = skipped  # 1a outcomes per edge (reported with the rejections)
    for edge, st in edge_frames(s, top, rng, stand_off):
        if not s.kit.trav_free(st.base_xy):
            skipped[edge.name] = f"no floor at {[round(float(v), 2) for v in st.base_xy.tolist()]}"
            log.info("phase 1a: edge %s (L %.2f, overhang %.2f) skipped: %s", edge.name, edge.length, edge.overhang, skipped[edge.name])
            continue
        if not s.ikc.stance_collision_free(st):
            shifted = shifted_init_stance(s, edge, stand_off)
            if shifted is None:
                skipped[edge.name] = f"stance at {[round(float(v), 2) for v in st.base_xy.tolist()]} collides at trunk (0.5, 0)"
                if os.environ.get("ISPATIALGYM_STANCE_BLAME"):  # oracle-fix 2026-11-03: name the body (IKChecker._collision_blame; opt-in, costs world syncs)
                    try:
                        skipped[edge.name] += " - " + s.ikc._collision_blame(s.ikc.q_locked(st), (), margin=0.01)
                    except Exception as e:  # noqa: BLE001
                        skipped[edge.name] += f" - blame failed ({e!r})"
                log.info("phase 1a: edge %s (L %.2f, overhang %.2f) skipped: %s", edge.name, edge.length, edge.overhang, skipped[edge.name])
                continue
            st = shifted
        log.info("phase 1a: edge %s (L %.2f, overhang %.2f) stance at %s", edge.name, edge.length, edge.overhang, [round(float(v), 2) for v in st.base_xy.tolist()])
        s._tabletop_top = top
        yield edge, st


def phase1_edge_and_standoff(s, top, rng=None, stand_off: float = STAND_OFF) -> Tuple[EdgeFrame, Stance]:
    """§4.2: any edge of the top polygon (no preference), yaw = inward normal, s = 0.45 m + the AABB overhang when >= 5 cm;
    ``Stance("init", translate 0.5, tilt 0)`` at the edge midpoint. ``Reject("no_free_edge")`` when no edge stance is on
    free floor and collision-free."""
    for edge, st in edge_candidates(s, top, rng, stand_off):
        return edge, st
    _reject("no_free_edge", f"{top.name}: no edge has a free, collision-free stance at s = {stand_off} (+ overhang): {getattr(s, '_tabletop_edge_log', {})}")


# ----------------------------------------------------------------------------------------------------------------
# Phase 1b — init trunk height (§4.3)
# ----------------------------------------------------------------------------------------------------------------


def phase1b_init_trunk(s, init: Stance, H_max: float, *, bands: Optional[RB.ReachBands] = None, top=None, margin: float = IN_FRAME_MARGIN,
                       left_bands: Optional[RB.ReachBands] = None) -> Stance:
    """§4.3: the init trunk (translate t*, tilt 0) from the camera FK: Δ = clamp(Δ_aim, Δ_min, Δ_max), t* = z_c⁻¹(h + Δ)
    clamped to [0, 0.95]; records ``b`` (image border lateral at the workspace centre depth), ``R_img`` (centre line),
    the workspace depths, the band union extent [λ_min, λ_max] and the translate range of [Δ_min, Δ_max] in
    ``stance.extras`` for Phase 1c. With ``left_bands`` (A1: the left contact band of a two-arm task) the lateral extent
    is the union of both bands' workspaces on this top (§4.4 margins use the union; the depths stay the work arm's)."""
    top = _top(s, top)
    bands = bands or ensure_bands(s, top=top, edge=init.edge)
    fk = camera_fk(s)
    ws = workspace(s, bands, init, top)
    if len(ws["cells_world"]) == 0:
        _reject("no_workspace", f"{top.name} edge {init.edge.name}: no band cell of {bands.path or 'the band'} falls on the top at s = {init.stand_off:.2f}")
    if left_bands is not None:
        ws_l = workspace(s, left_bands, init, top)
        if len(ws_l["cells_world"]):
            ws["lam_min_right"], ws["lam_max_right"] = ws["lam_min"], ws["lam_max"]
            ws["lam_min_left"], ws["lam_max_left"] = ws_l["lam_min"], ws_l["lam_max"]
            if margins_union():
                ws["lam_min"], ws["lam_max"] = min(ws["lam_min"], ws_l["lam_min"]), max(ws["lam_max"], ws_l["lam_max"])
            else:
                log.info("phase 1b: λ of the right band %s kept for the margins (left band %s; ISG_MARGIN_UNION=1 for the union)",
                         [round(ws["lam_min"], 2), round(ws["lam_max"], 2)], [round(ws_l["lam_min"], 2), round(ws_l["lam_max"], 2)])
        else:
            log.info("phase 1b: the left contact band has no cell on %s at edge %s; λ from the right band alone", top.name, init.edge.name)
    h = float(top.z) - float(s.kit.floor_z)
    phi_m = fk.phi_m(margin)
    t = float(init.translate)
    res = None
    for _ in range(3):  # x_c depends weakly on t: two fixed-point passes settle it
        res = init_trunk_delta(init.edge_depth, ws["d_ws"], ws["d_near"], fk.x_at(t), H_max, fk.hand_below, fk.pitch, phi_m)
        t = fk.t_of_z(h + res["delta"])
    st = init.with_trunk(t, 0.0, "init")
    cam = s.camera_pose_at(st)
    b_r, b_l = image_border_lateral(s, cam, st, ws["d_ws"], top.z, margin)
    b = min(b_r, b_l)
    t_lo = fk.t_of_z(h + res["delta_max"])  # deepest camera height allowed -> smallest t? (z decreases with t)
    t_hi = fk.t_of_z(h + res["delta_min"])
    st.extras.update({"H_max": float(H_max), **res, "d_ws": ws["d_ws"], "d_near": ws["d_near"], "d_far": ws["d_far"], "lam_min": ws["lam_min"],
                      "lam_max": ws["lam_max"], "x_c": fk.x_at(t), "z_c": fk.z_at(t), "b": b, "b_right": b_r, "b_left": b_l, "R_img": b,
                      "t_range": [min(t_lo, t_hi), max(t_lo, t_hi)], "camera_pitch_deg": math.degrees(fk.pitch), "hand_below": fk.hand_below,
                      "n_workspace_cells": int(len(ws["cells_world"])), "left_band": left_bands is not None,
                      **{k: ws[k] for k in ("lam_min_right", "lam_max_right", "lam_min_left", "lam_max_left") if k in ws}})
    log.info("phase 1b: edge %s h=%.2f d_ws=%.2f d_near=%.2f Δ_aim=%.2f Δ_min=%.2f Δ_max=%.2f -> Δ=%.2f t*=%.2f b=%.2f λ=[%.2f, %.2f]", init.edge.name, h,
             ws["d_ws"], ws["d_near"], res["delta_aim"], res["delta_min"], res["delta_max"], res["delta"], t, b, ws["lam_min"], ws["lam_max"])
    return st


# ----------------------------------------------------------------------------------------------------------------
# Phase 1c — lateral position (§4.4)
# ----------------------------------------------------------------------------------------------------------------


def trunk_sweep_free(s, stance: Stance, translates: Sequence[float] = TRUNK_SWEEP_TRANSLATES, margin: float = 0.01) -> bool:
    """§4.4 check (3): base + torso links + head collision-free at every grid translate, tilt 0, arms excluded. cuRobo
    checks whole-robot spheres, so the arms are hung (tucked, inside the base column) and the self-collision term is
    off — the closest available reading of "arms excluded"."""
    s.kit._ensure_synced()
    qs = th.stack([s.kit.q_at_base_pose(s.kit.q_config(arms="tucked", translate=float(tr), tilt=0.0), *stance.base_pose2d) for tr in translates])
    coll = s.kit.backend.check_collision(qs, emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)
    return not bool(coll.any())


def phase1c_bounds(s, init: Stance, margins: Dict[str, Any], *, top=None, in_frame_margin: float = IN_FRAME_MARGIN, at: Optional[Stance] = None,
                   strict: bool = False) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any], float]:
    """§4.4 step 2 for a Phase-1b init trunk ``init`` (its ``lam_min`` / ``lam_max`` / ``d_ws`` extras): ``(bounds, info, b)`` - the lateral
    bounds [−L/2 + m_R, L/2 − m_L] (None when m_R + m_L > L), the margin row's ``info`` and the image-border lateral ``b`` at the camera of
    ``at`` (default ``init``; :func:`phase1c_lateral` re-evaluates them at a moved trunk). A row that rejects (nextto ``pair_gap``) raises
    its rejection; ``strict`` raises the two ``edge_too_short`` rejections of :func:`phase1c_lateral` as well. Factored out 2026-10-31
    (train-fast) so the IORS start-per-attempt memo discretises the same bounds Phase 1c draws over."""
    top = _top(s, top)
    st = init if at is None else at
    lam_min, lam_max = float(init.extras["lam_min"]), float(init.extras["lam_max"])
    L = float(init.edge.length)
    cam = s.camera_pose_at(st)
    b_r, b_l = image_border_lateral(s, cam, st, float(init.extras["d_ws"]), top.z, in_frame_margin)
    b = min(b_r, b_l)
    m_R, m_L, info = lateral_margins(margins, lam_min, lam_max, b, b)
    if info.get("reject"):
        _reject(info["reject"][0], info["reject"][1])
    bounds = lateral_bounds(L, m_R, m_L)
    if strict:
        if bounds is None:
            _reject("edge_too_short", f"edge {init.edge.name}: m_R + m_L = {info['m_R'] + info['m_L']:.2f} > L = {L:.2f} ({info})")
        if L < float(info.get("L_min", 0.0)):
            _reject("edge_too_short", f"edge {init.edge.name}: L = {L:.2f} < 2 w_t + g_min + 2 w_p + 2 m_end = {info['L_min']:.2f} ({info})")
    return bounds, info, b


def phase1c_lateral(s, init: Stance, margins: Dict[str, Any], rng, *, top=None, bands: Optional[RB.ReachBands] = None, n_draws: int = 20,
                    in_frame_margin: float = IN_FRAME_MARGIN, lateral: Optional[float] = None) -> Stance:
    """§4.4: margins of the task's row from the band union extent, ``b`` and ``R_img`` of Phase 1b; lateral position
    uniform over [−L/2 + m_R, L/2 − m_L] (``Reject("edge_too_short")`` when m_R + m_L > L, or for the IORS row when L is below its
    ``L_min``: the edge cannot hold both objects at the smallest gap ``m_end`` from its ends); valid when (1) the base is on
    free floor, (2) the init configuration (t*, 0, untucked, open) is whole-body collision-free — else t moves within
    [Δ_min, Δ_max] (b, R_img, margins recomputed) or the lateral is redrawn —, (3) the trunk sweep of translates
    0 … 0.95 at tilt 0 is collision-free with the arms excluded. The result is the instance's **init pose**.

    ``lateral`` (train-fast 2026-10-31, the IORS start-per-attempt memo): that one lateral instead of the draws - the checks (1)-(3) and
    the translate retry run unchanged on it, ``rng`` is not consulted (may be None), one try (``n_draws`` is ignored)."""
    top = _top(s, top)
    if "lam_min" not in init.extras:
        init = phase1b_init_trunk(s, init, float(margins.get("H_max", init.extras.get("H_max", 0.0))), bands=bands, top=top)
    fk = camera_fk(s)
    h = float(top.z) - float(s.kit.floor_z)
    L = float(init.edge.length)
    errors: Dict[str, int] = {"no_floor": 0, "collision": 0, "trunk_sweep": 0, "out_of_bounds_after_t": 0}
    t_lo, t_hi = init.extras.get("t_range", [init.translate, init.translate])

    def bounds_for(st: Stance):
        return phase1c_bounds(s, init, margins, top=top, in_frame_margin=in_frame_margin, at=st)

    bounds, info, b = bounds_for(init)
    if bounds is None:
        _reject("edge_too_short", f"edge {init.edge.name}: m_R + m_L = {info['m_R'] + info['m_L']:.2f} > L = {L:.2f} ({info})")
    if L < float(info.get("L_min", 0.0)):
        _reject("edge_too_short", f"edge {init.edge.name}: L = {L:.2f} < 2 w_t + g_min + 2 w_p + 2 m_end = {info['L_min']:.2f} ({info})")
    n_draws = 1 if lateral is not None else int(n_draws)
    for _ in range(n_draws):
        lat = float(lateral) if lateral is not None else float(rng.uniform(bounds[0], bounds[1]))
        st = init.shifted(lateral=lat, name="init")
        if not s.kit.trav_free(st.base_xy):
            errors["no_floor"] += 1
            continue
        if not s.ikc.stance_collision_free(st):
            # move t within [Δ_min, Δ_max]: the other translates of the admissible range, nearest first
            fixed = None
            for t in sorted(np.linspace(float(t_lo), float(t_hi), 5).tolist(), key=lambda v: abs(v - st.translate)):
                if abs(t - st.translate) < 1e-6:
                    continue
                st2 = st.with_trunk(clamp(t, 0.0, MAX_TRANSLATE), 0.0, "init")
                if s.ikc.stance_collision_free(st2):
                    b2, info2, bb = bounds_for(st2)
                    if b2 is not None and b2[0] <= lat <= b2[1]:
                        st2.extras.update({"b": bb, "R_img": bb, "z_c": fk.z_at(st2.translate), "delta": fk.z_at(st2.translate) - h})
                        fixed, info, bounds = st2, info2, b2
                        break
                    errors["out_of_bounds_after_t"] += 1
            if fixed is None:
                errors["collision"] += 1
                continue
            st = fixed
        if not trunk_sweep_free(s, st):
            errors["trunk_sweep"] += 1
            continue
        st.extras.update({"margins": info, "lateral_bounds": list(bounds)})
        log.info("phase 1c: edge %s lateral %.2f in [%.2f, %.2f] (m_R %.2f, m_L %.2f), t* %.2f", init.edge.name, lat, bounds[0], bounds[1], info["m_R"],
                 info["m_L"], st.translate)
        return st
    _reject("stance_collision", f"edge {init.edge.name}: no valid lateral position in [{bounds[0]:.2f}, {bounds[1]:.2f}] after {n_draws} draws ({errors})"
                                + ("" if lateral is None else f" at the given lateral {float(lateral):+.3f}"))


# ----------------------------------------------------------------------------------------------------------------
# Phase 2 — manipulation stance (§4.5)
# ----------------------------------------------------------------------------------------------------------------


def tilt_scan(fk: CameraFK, translate: float, n: int = N_TILT_SAMPLES) -> List[float]:
    """§4.5 "tilt continuous from 0 to the torso_joint3 limit": ``n`` samples from 0 down to the limit at ``translate``."""
    lim = fk.tilt_limit(translate)
    return [float(v) for v in np.linspace(0.0, lim, int(n)).tolist()]


def default_in_frame(s, stance: Stance, cam: Pose, focus: th.Tensor, info: Any, half_extent: float, height: float, margin: float) -> bool:
    """§4.5 step 3 default: the target AABB (a box of ``half_extent`` around the focus from the top to ``height``) in frame."""
    z0 = float(focus[2]) - 0.5 * height
    return in_frame(s, cam, box_corners(focus[:2], half_extent, z0, z0 + height), margin)


def phase2_manipulation_stance(s, init: Stance, target_half_extent: float, *, target_height: Optional[float] = None,
                               in_frame_fn: Optional[Callable[..., bool]] = None, focus_candidates: Optional[Callable[..., List[Tuple[th.Tensor, Any]]]] = None,
                               stand_offs: Sequence[float] = STAND_OFFS, translates: Sequence[float] = TRANSLATES, n_tilts: int = N_TILT_SAMPLES,
                               angle_classes_deg: Sequence[float] = ANGLE_CLASSES_DEG, margin: float = IN_FRAME_MARGIN, bands: Optional[RB.ReachBands] = None,
                               class1_when_deep_m: Optional[float] = None,
                               top=None, erosion_extra: float = 0.02, focus_z: Optional[float] = None, order: Optional[str] = None,
                               arm_bands: Optional[Dict[str, RB.ReachBands]] = None, split_cone: Optional[Dict[str, Any]] = None,
                               keep_pairs: Optional[Callable[[List[Stance]], List[Stance]]] = None) -> List[Stance]:
    """§4.5: scan stand-off {0.35, 0.45, 0.55} x translate {0, …, 0.95} x tilt (0 → torso_joint3 limit, ``n_tilts`` samples)
    at the init edge and lateral; (1) whole-body collision check without objects; (2) band = the grid trunk with the
    same translate and nearest tilt, focus candidates = its cells inside the top eroded by ``target_half_extent`` +
    0.02 (shifted with the stand-off); (3) keep (stance, focus) pairs whose optical axis is within 20° of the focus (else
    30°) and whose target box is in frame (5 %; ``in_frame_fn(s, stance, cam, focus, info)`` overrides: the disembedding
    source tray, the block region); ``focus_candidates(s, stance, cam, cells_world)`` overrides the focus cells (block
    regions: ``[(focus_xy, info), ...]``). Returns the survivors of the **best angle class**, each a ``Stance
    ("manipulation")`` copy with ``focus`` (x, y, z) set and ``extras`` ``axis_angle`` / ``angle_class`` / ``focus_info``;
    the caller draws uniformly (§4.5 step 4) and checks the init → manipulation transition (step 5). ``order="reach"``
    (W2, owner 2026-09-20 22:15; opt-in through the task's ``generation`` so the valid small-scale demos keep their uniform
    draw) returns the survivors sorted by the right arm's reach margin to their focus (stretched arm length minus the
    shoulder -> focus distance, 5 cm buckets, deepest margin first): the tallest torso bowing forward (translate 0,
    tilt -0.6: the grasp 0.88 m out) comes last.

    ``arm_bands`` (owner 2026-09-27, first-demo v2.2: ``{arm: band}`` of every arm offered for the segment, the default arm first):
    the focus candidates are the cells of every offered arm's band (the other arm's band a mirror when it has no band of its
    own), each survivor records in ``extras["focus_arms"]`` the arms whose band covers its focus, and ``order="reach"`` ranks by
    whichever of those arms reaches better - so stances that suit the other arm reach the orientation chooser.

    ``split_cone`` (:func:`split_cone_opts`, opt-in ``generation.phase2_split_cone``): :func:`phase2_split_cone` instead - the
    start position in a wide cone of the init view, the work stance in its own cone with a search along the edge. When it finds
    no (stance, position) pair and the options carry ``fallback_classes_deg``, the plain Phase 2 below with those classes.

    ``keep_pairs(survivors) -> survivors`` (a task's own test of the (stance, position) pairs, disembedding's destination fit):
    applied to the split cone's survivors before the fallback is decided - none kept counts as none found - and to the plain Phase
    2's per angle class."""
    if split_cone is not None:
        opts = {k: v for k, v in split_cone.items() if k != "fallback_classes_deg"}
        out = phase2_split_cone(s, init, target_half_extent, target_height=target_height, in_frame_fn=in_frame_fn, focus_candidates=focus_candidates,
                                stand_offs=stand_offs, translates=translates, n_tilts=n_tilts, margin=margin, bands=bands,
                                class1_when_deep_m=class1_when_deep_m, top=top, erosion_extra=erosion_extra, focus_z=focus_z, arm_bands=arm_bands, **opts)
        if out and keep_pairs is not None:
            out = keep_pairs(out)
        fallback = split_cone.get("fallback_classes_deg")
        if out or not fallback:
            return out
        angle_classes_deg = (float(fallback[0]), float(fallback[1]))
        log.info("phase 2 (split cone): no (stance, position) pair - the plain Phase 2 with the %.0f / %.0f deg classes", *angle_classes_deg)
    top = _top(s, top)
    bands = bands or ensure_bands(s, top=top, edge=init.edge)
    arm_bands = {a: b for a, b in (arm_bands or {}).items() if b is not None}
    if len(arm_bands) < 2:
        arm_bands = {}
    fk = camera_fk(s)
    height = float(target_height) if target_height is not None else 2.0 * float(target_half_extent)
    cands: List[Stance] = []
    for so in stand_offs:
        base = init.shifted(stand_off=float(so), name="manipulation")
        if not s.kit.trav_free(base.base_xy):
            continue
        for tr in translates:
            for ti in tilt_scan(fk, float(tr), n_tilts):
                cands.append(base.with_trunk(float(tr), float(ti), "manipulation"))
    if not cands:
        return []
    free = s.ikc.stances_collision_free(cands)
    cands = [c for c, f in zip(cands, free.tolist()) if f]
    if not cands:
        log.info("phase 2: every trunk candidate collides")
        return []
    cams = s.kit.camera_poses_batch(th.stack([s.stance_q(c) for c in cands]))
    cm = s.world.camera_model
    erosion = float(target_half_extent) + float(erosion_extra)
    classes: Dict[int, List[Stance]] = {0: [], 1: []}
    n_cells = 0
    diag: List[str] = []  # per trunk candidate: cells on the top, min axis angle, nearest focus (base frame) — for the cone question
    for st, cam in zip(cands, cams):
        cell_arms: Dict[Tuple[float, float], List[str]] = {}
        if arm_bands:
            cells_l = []
            for a, b in arm_bands.items():
                for c in band_cells_world(b, st):
                    k = (round(float(c[0]), 3), round(float(c[1]), 3))
                    if k not in cell_arms:
                        cell_arms[k] = []
                        cells_l.append(c)
                    cell_arms[k].append(a)
            cells = th.stack(cells_l) if cells_l else th.zeros(0, 2)
        else:
            cells = band_cells_world(bands, st)
        if len(cells) == 0:
            continue
        keep = [c for c in cells if top.contains(c, erosion)]
        if not keep:
            continue
        cw = th.stack(keep)
        foci = focus_candidates(s, st, cam, cw) if focus_candidates is not None else [(c, None) for c in cw]
        angs = []
        for focus_xy, info in foci:
            fz = float(focus_z) if focus_z is not None else NB.z_at(top, focus_xy) + 0.5 * height
            focus = th.tensor([float(focus_xy[0]), float(focus_xy[1]), fz])
            ang = cm.angle_off_axis(cam, focus)
            angs.append((float(ang), focus_xy))
            if ang > math.radians(angle_classes_deg[1]):
                continue
            n_cells += 1
            ok = (in_frame_fn(s, st, cam, focus, info) if in_frame_fn is not None
                  else default_in_frame(s, st, cam, focus, info, float(target_half_extent), height, margin))
            if not ok:
                continue
            # TILT_SNAP (2026-11-01): Phase 3 re-applies these work-camera rules to a stance moved to its grid tilt (the plain path's
            # classes, also after a split-cone fallback)
            s._phase2_camera_rule = {"max_angle": math.radians(angle_classes_deg[1]), "boxed": lambda st_, cam_, f_, i_: bool(
                in_frame_fn(s, st_, cam_, f_, i_) if in_frame_fn is not None
                else default_in_frame(s, st_, cam_, f_, i_, float(target_half_extent), height, margin))}
            c = st.with_trunk(st.translate, st.tilt, "manipulation")
            c.focus = focus
            c.extras.update({"axis_angle": ang, "angle_class": 0 if ang <= math.radians(angle_classes_deg[0]) else 1, "focus_info": info})
            if arm_bands:
                c.extras["focus_arms"] = list(cell_arms.get((round(float(focus_xy[0]), 3), round(float(focus_xy[1]), 3)), [next(iter(arm_bands))]))
            classes[c.extras["angle_class"]].append(c)
        if angs:
            a_min, f_min = min(angs, key=lambda t: t[0])
            diag.append(f"s={st.stand_off:.2f} t=({st.translate:.2f},{st.tilt:+.2f}): {len(foci)} foci, min axis angle {math.degrees(a_min):.0f}° at base "
                        f"{[round(float(v), 2) for v in st.world_to_base(f_min[:2]).tolist()]}")
    if keep_pairs is not None:
        classes = {k: (keep_pairs(v) if v else v) for k, v in classes.items()}
    out = classes[0] or classes[1]
    which = "0" if classes[0] else "1"
    if class1_when_deep_m is not None and classes[0] and classes[1]:
        # D-009 B (prepared, owner 2026-09-16 21:54): the tight class puts the focus where the head camera's axis meets the
        # top — 0.77–0.81 m base-forward at the attachment trunks, beyond the arm's lift / carry reach — so when EVERY
        # class-0 focus is deeper than ``class1_when_deep_m`` the class-1 survivors are admitted as well (class 0 first)
        depth0 = [float(c.world_to_base(c.focus[:2])[0]) for c in classes[0]]
        if min(depth0) > float(class1_when_deep_m):
            out = classes[0] + classes[1]
            which = f"0+1 (class-0 foci {min(depth0):.2f}–{max(depth0):.2f} m base-forward, all deeper than {float(class1_when_deep_m):.2f})"
    log.info("phase 2: %d trunk candidates collision-free, %d (stance, focus) pairs within %.0f°, %d survivors in class %s%s", len(cands), n_cells,
             float(angle_classes_deg[1]), len(out), which,
             "" if not arm_bands else " (focus in the band of " + ", ".join(f"{a}: {sum(1 for c in out if a in c.extras.get('focus_arms', []))}" for a in arm_bands) + ")")
    if diag and (not out or os.environ.get("ISG_PHASE2_DEBUG")):
        # ISG_PHASE2_DEBUG=1: the per-trunk view even with survivors (which trunks the cone / in-frame rule excludes)
        log.info("phase 2: per trunk candidate: %s", " | ".join(diag[:40]))
        if os.environ.get("ISG_PHASE2_DEBUG"):
            trunks = sorted({(round(float(c.stand_off or 0), 2), round(float(c.translate), 2), round(float(c.tilt), 2)) for c in out})
            log.info("phase 2: surviving (stand-off, translate, tilt): %s", trunks)
    if order == "reach" and out:
        out = sorted(out, key=lambda c: -min(round(best_reach_margin(s, c, c.focus, c.extras.get("focus_arms") or ("right",))[0] / 0.05) * 0.05, 0.30))
    return out


def split_cone_opts(s) -> Optional[Dict[str, Any]]:
    """Owner 2026-10-11 (cone A/B variant C), opt-in per subtask with ``generation.phase2_split_cone``: the keyword arguments of
    :func:`phase2_split_cone` - the start position's classes at the init view (``init_view_class_deg``), the work stance's shifts
    along the edge (``phase2_lateral_shifts_m``) and its classes (``phase2_tight_class_deg`` / ``phase2_loose_class_deg``) - or None
    (off: Phase 2 as before). Every key through :func:`phase2_gen` (the variant's customs, else its subtask's generation_defaults).
    ``phase2_split_fallback_classes_deg`` adds ``fallback_classes_deg``: the plain Phase 2's classes for a template whose split
    cone finds no (stance, position) pair."""
    if not phase2_gen(s, "phase2_split_cone", False):
        return None
    iv = phase2_gen(s, "init_view_class_deg") or SPLIT_INIT_VIEW_CLASSES_DEG
    out = {"init_view_classes_deg": (float(iv[0]), float(iv[1])),
           "lateral_shifts": tuple(float(v) for v in (phase2_gen(s, "phase2_lateral_shifts_m") or SPLIT_LATERAL_SHIFTS)),
           "work_classes_deg": phase2_angle_classes(s)}
    fb = phase2_gen(s, "phase2_split_fallback_classes_deg")
    if fb:
        out["fallback_classes_deg"] = (float(fb[0]), float(fb[1]))
    return out


def stance_cameras(s, stances: Sequence[Stance]) -> List[Pose]:
    """Head-camera poses at ``stances`` through the kit's batched FK (as Phase 2)."""
    return list(s.kit.camera_poses_batch(th.stack([s.stance_q(c) for c in stances])))


def phase2_split_cone(s, init: Stance, target_half_extent: float, *, init_view_classes_deg: Sequence[float], lateral_shifts: Sequence[float],
                      work_classes_deg: Sequence[float], target_height: Optional[float] = None, in_frame_fn: Optional[Callable[..., bool]] = None,
                      focus_candidates: Optional[Callable[..., List[Tuple[th.Tensor, Any]]]] = None, stand_offs: Sequence[float] = STAND_OFFS,
                      translates: Sequence[float] = TRANSLATES, n_tilts: int = N_TILT_SAMPLES, margin: float = IN_FRAME_MARGIN,
                      bands: Optional[RB.ReachBands] = None, class1_when_deep_m: Optional[float] = None, top=None, erosion_extra: float = 0.02,
                      focus_z: Optional[float] = None, arm_bands: Optional[Dict[str, RB.ReachBands]] = None) -> List[Stance]:
    """Owner 2026-10-11 (cone A/B variant C): Phase 2 with one cone for the start image and one for the work. Work stances = the init
    stance shifted along the edge by ``lateral_shifts`` x stand-off x translate x tilt (the §4.5 scan), on free floor and whole-body
    collision-free; their band cells inside the eroded top (``focus_candidates`` / ``in_frame_fn`` / ``arm_bands`` as in Phase 2) are
    the possible start positions. A (work stance, position) pair survives when the position lies within ``init_view_classes_deg[1]`` of
    the INIT camera's optical axis with the target box in frame from the init stance, and within ``work_classes_deg[1]`` of the work
    camera's axis with the box in frame from there. Each survivor carries ``axis_angle`` / ``angle_class`` (work camera),
    ``init_view_angle`` / ``init_view_class``, ``lateral_shift_m`` and ``_position`` (the position snapped to the band lattice in the
    init stance's base frame). Every survivor is returned; :func:`split_cone_draw` draws a position per attempt and hands the task that
    position's work stances of the best class."""
    top = _top(s, top)
    bands = bands or ensure_bands(s, top=top, edge=init.edge)
    arm_bands = {a: b for a, b in (arm_bands or {}).items() if b is not None}
    if len(arm_bands) < 2:
        arm_bands = {}
    fk = camera_fk(s)
    height = float(target_height) if target_height is not None else 2.0 * float(target_half_extent)
    erosion = float(target_half_extent) + float(erosion_extra)
    lattice = float(bands.meta.get("lattice", RB.LATTICE))
    iv0, iv1 = (math.radians(float(v)) for v in init_view_classes_deg)
    w0, w1 = (math.radians(float(v)) for v in work_classes_deg)
    cands: List[Stance] = []
    for shift in lateral_shifts:
        for so in stand_offs:
            base = init.shifted(stand_off=float(so), lateral=float(init.lateral) + float(shift), name="manipulation")
            if not s.kit.trav_free(base.base_xy):
                continue
            base.extras["lateral_shift_m"] = float(shift)
            for tr in translates:
                for ti in tilt_scan(fk, float(tr), n_tilts):
                    cands.append(base.with_trunk(float(tr), float(ti), "manipulation"))
    if cands:
        free = s.ikc.stances_collision_free(cands)
        cands = [c for c, f in zip(cands, free.tolist()) if f]
    if not cands:
        log.info("phase 2 (split cone): no work stance on free floor and collision-free over the lateral shifts %s", list(lateral_shifts))
        return []
    cams = stance_cameras(s, cands)
    cam0 = stance_cameras(s, [init])[0]
    cm = s.world.camera_model

    def boxed(stance, cam, focus, info) -> bool:
        return bool(in_frame_fn(s, stance, cam, focus, info) if in_frame_fn is not None
                    else default_in_frame(s, stance, cam, focus, info, float(target_half_extent), height, margin))

    # TILT_SNAP (2026-11-01): Phase 3 re-applies these work-camera rules to a stance moved to its grid tilt
    s._phase2_camera_rule = {"max_angle": float(w1), "boxed": boxed}

    seen: Dict[Tuple[float, float], Tuple[float, bool]] = {}  # position (mm) -> (init-view angle, admitted at the init view)
    out: List[Stance] = []
    n_pairs = 0
    for st, cam in zip(cands, cams):
        cell_arms: Dict[Tuple[float, float], List[str]] = {}
        if arm_bands:
            cells_l = []
            for a, b in arm_bands.items():
                for c in band_cells_world(b, st):
                    k = (round(float(c[0]), 3), round(float(c[1]), 3))
                    if k not in cell_arms:
                        cell_arms[k] = []
                        cells_l.append(c)
                    cell_arms[k].append(a)
            cells = th.stack(cells_l) if cells_l else th.zeros(0, 2)
        else:
            cells = band_cells_world(bands, st)
        keep = [c for c in cells if top.contains(c, erosion)]
        if not keep:
            continue
        foci = focus_candidates(s, st, cam, th.stack(keep)) if focus_candidates is not None else [(c, None) for c in keep]
        for focus_xy, info in foci:
            fz = float(focus_z) if focus_z is not None else NB.z_at(top, focus_xy) + 0.5 * height
            focus = th.tensor([float(focus_xy[0]), float(focus_xy[1]), fz])
            ang = cm.angle_off_axis(cam, focus)
            if ang > w1:
                continue
            # the init view depends on the position alone, except through a hook's per-stance region (block design)
            key = (round(float(focus[0]), 3), round(float(focus[1]), 3)) if info is None else None
            if key is None or key not in seen:
                a0 = cm.angle_off_axis(cam0, focus)
                view = (a0, a0 <= iv1 and boxed(init, cam0, focus, info))
                if key is not None:
                    seen[key] = view
            else:
                view = seen[key]
            a0, ok0 = view
            if not ok0:
                continue
            n_pairs += 1
            if not boxed(st, cam, focus, info):
                continue
            c = st.with_trunk(st.translate, st.tilt, "manipulation")
            c.focus = focus
            fb = init.world_to_base(focus[:2])
            c.extras.update({"axis_angle": ang, "angle_class": 0 if ang <= w0 else 1, "focus_info": info, "init_view_angle": a0,
                             "init_view_class": 0 if a0 <= iv0 else 1,
                             "_position": (int(math.floor(float(fb[0]) / lattice + 0.5)), int(math.floor(float(fb[1]) / lattice + 0.5)))})
            if arm_bands:
                c.extras["focus_arms"] = list(cell_arms.get((round(float(focus_xy[0]), 3), round(float(focus_xy[1]), 3)), [next(iter(arm_bands))]))
            out.append(c)
    s._split_cone_class1_deep = class1_when_deep_m
    log.info("phase 2 (split cone): %d work stances collision-free over the lateral shifts %s, %d (stance, position) pairs within %.0f° of the work camera "
             "and %.0f° of the init camera, %d survivors at %d positions (%d within %.0f° of the init view)", len(cands), list(lateral_shifts), n_pairs,
             math.degrees(w1), math.degrees(iv1), len(out), len({c.extras["_position"] for c in out}),
             len({c.extras["_position"] for c in out if c.extras["init_view_class"] == 0}), math.degrees(iv0))
    return out


def split_cone_positions(cands: Sequence[Stance], class1_when_deep_m: Optional[float] = None, side: Optional[float] = None) -> Dict[Tuple[int, int], List[Stance]]:
    """Variant C: position -> its work stances of the best class (the tight class, else the loose; D-009 B: the loose-class stances
    admitted too when every tight-class focus lies deeper than ``class1_when_deep_m`` base-forward), over the positions of the best
    init-view class. With ``side`` (IORS, A1) only the work stances whose start region admits a partner on that side
    (:func:`in_start_region`) count."""
    cands = [c for c in cands if in_start_region(c, side)]
    tier = 0 if any(c.extras["init_view_class"] == 0 for c in cands) else 1
    by_pos: Dict[Tuple[int, int], List[Stance]] = {}
    for c in cands:
        if c.extras["init_view_class"] <= tier:
            by_pos.setdefault(tuple(c.extras["_position"]), []).append(c)
    out: Dict[Tuple[int, int], List[Stance]] = {}
    for key, cs in by_pos.items():
        c0 = [c for c in cs if c.extras["angle_class"] == 0]
        best = c0 or cs
        if class1_when_deep_m is not None and c0 and len(c0) < len(cs):
            if min(float(c.world_to_base(c.focus[:2])[0]) for c in c0) > float(class1_when_deep_m):
                best = c0 + [c for c in cs if c.extras["angle_class"] != 0]
        out[key] = best
    return out


def split_cone_draw(s, rng: np.random.Generator) -> None:
    """Variant C, once per attempt (``SamplerV2.sample``): the object's start position drawn uniformly among the positions of the best
    init-view class that still have a work stance after the task's own filters (``s.manip_candidates`` as ``sample_template`` left
    them), and that position's work stances of the best class made the task's candidates for this attempt (the task's own order
    applies among them). A no-op that leaves ``rng`` untouched unless the split cone is on. A task's memory of failed candidates
    (attachment's ``_candidate_rejects``, keyed by the candidate) is kept across positions (owner 2026-10-12, attachment 7h: a
    position drawn again tries the stances it has not failed with yet). With ``s.region_side`` (IORS, A1: the partner side
    the attempt drew before the target is placed) the positions come from that side's pool (:func:`split_cone_positions` with the
    side), each pool built once from the candidates ``sample_template`` left."""
    if not phase2_gen(s, "phase2_split_cone", False):
        return
    cands = getattr(s, "_split_cone_cands", None)
    if cands is None:
        cands = list(getattr(s, "manip_candidates", None) or [])
        if not cands or "_position" not in cands[0].extras:
            return
        s._split_cone_cands = cands
    side = getattr(s, "region_side", None)
    pools = getattr(s, "_split_cone_pools", None)
    if pools is None:
        pools = s._split_cone_pools = {}
    if side not in pools:
        pools[side] = split_cone_positions(cands, getattr(s, "_split_cone_class1_deep", None), side=side)
    pool = pools[side]
    if not pool:
        return
    keys = sorted(pool)
    s.manip_candidates = list(pool[keys[int(rng.integers(len(keys)))]])


def reach_margin(s, stance: Stance, point_world, arm: str = "right") -> float:
    """``arm``'s stretched length (kit FK, arm joints at zero; cached on the sampler) minus the shoulder -> ``point_world``
    distance at the stance's URDF-clamped trunk (``conventions.torso_fk`` / ``torso_q`` with ``TORSO_JOINT_LIMITS``)."""
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling import geometry as G
    lens = s.__dict__.setdefault("_arm_len", {}) if hasattr(s, "__dict__") else {}
    if arm not in lens:
        q = s.kit.q_config(arms="untucked", translate=0.5, tilt=0.0)
        q[s.kit.robot.arm_control_idx[arm]] = 0.0
        fk = s.kit.fk(["torso_link4", M.EEF_LINK[arm]], q)
        sh = G.compose(fk["torso_link4"], (th.as_tensor(C._SHOULDER_IN_TORSO4[arm], dtype=th.float32), G.IDENTITY_QUAT))[0]
        lens[arm] = float(th.linalg.norm(fk[M.EEF_LINK[arm]][0] - sh))
    sh_b = C.torso_fk(C.torso_q(float(stance.translate), float(stance.tilt), C.TORSO_JOINT_LIMITS))[f"{arm}_shoulder"][0]
    x, y, yaw = (float(v) for v in stance.base_pose2d)
    cs, sn = math.cos(yaw), math.sin(yaw)
    sh_w = th.tensor([x + cs * float(sh_b[0]) - sn * float(sh_b[1]), y + sn * float(sh_b[0]) + cs * float(sh_b[1]), float(s.kit.floor_z) + C.NOMINAL_BASE_Z + float(sh_b[2])])
    p = th.as_tensor([float(v) for v in point_world], dtype=th.float32)
    return lens[arm] - float(th.linalg.norm(p - sh_w))


def best_reach_margin(s, stance: Stance, point_world, arms: Sequence[str] = ("right",)) -> Tuple[float, str]:
    """The largest :func:`reach_margin` to ``point_world`` over ``arms`` and its arm (the first arm on a tie): a stance ranked by
    whichever arm reaches better (owner 2026-09-27)."""
    best: Optional[Tuple[float, str]] = None
    for a in arms:
        m = reach_margin(s, stance, point_world, a)
        if best is None or m > best[0] + 1e-9:
            best = (m, a)
    return best if best is not None else (reach_margin(s, stance, point_world), "right")


# ----------------------------------------------------------------------------------------------------------------
# Gripper symmetry: a library pinch or its twin (the eef turned π about the approach axis, the fingers swapped)
# ----------------------------------------------------------------------------------------------------------------

#: a support-ignored grasp pose (the row or its twin) must keep the wrist camera's spheres this far above the support top (m)
TWIN_CAMERA_CLEARANCE_M = 0.005
#: the wrist camera (``<arm>_realsense_link``), the one part of the hand that is not C2-symmetric about the approach axis: its
#: cuRobo spheres in the eef frame (as ``iors_small.HAND_SPHERES_EEF["wrist camera"]``) and their radius (m)
WRIST_CAMERA_SPHERES_EEF = ((-0.056, -0.002, -0.076), (-0.056, -0.030, -0.076), (-0.056, -0.057, -0.076))
WRIST_CAMERA_SPHERE_R = 0.018


def grasp_twins_on(s) -> bool:
    """``generation.grasp_twins`` (default on; ``false`` / ``"off"`` switch it off): whether a library pinch drawn for a free
    object may be recorded as its twin."""
    gen = getattr(getattr(s, "gp", None), "gen", None) or getattr(s, "gen", None)
    v = gen("grasp_twins", True) if gen is not None else True
    return v is None or (v is not False and str(v).strip().lower() not in ("off", "false", "0", "no"))


def wrist_camera_clearance(eef: Pose, plane_z: float) -> float:
    """Smallest height (m) of the wrist camera's spheres above the plane ``z = plane_z`` at the eef pose ``eef`` (negative:
    the camera dips below it)."""
    pts = C.transform_points((_t(eef[0]), _t(eef[1])), th.tensor(WRIST_CAMERA_SPHERES_EEF, dtype=th.float32))
    return float(pts[:, 2].min()) - WRIST_CAMERA_SPHERE_R - float(plane_z)


@dataclass
class TwinPick:
    """What :func:`pick_twin` chose. ``twin``: the recorded eef is the row's twin; ``grasp``: the library row or
    ``grasp_library.twin(row)`` (name ``<row>~r180``, key, reversed ``finger_q``); ``eef_in_object`` / ``eef`` / ``pre`` /
    ``res``: the recorded chain; ``passed``: that chain solved and passed the caller's check; ``other``: ``(grasp,
    eef_in_object, eef, pre)`` of the other finger assignment when it could be recorded as well (the row: its chain passed
    the caller's check; the twin: every check); ``facts``: the grasp record's ``twin_choice``."""

    twin: bool
    grasp: Any
    eef_in_object: Pose
    eef: Pose
    pre: Pose
    res: Any
    passed: bool
    other: Optional[Tuple[Any, Pose, Pose, Pose]]
    facts: Dict[str, Any]
    #: the arm that makes the grasp and the stance it was solved at (its working arm; the other arm idles there)
    arm: str = "right"
    stance: Optional[Stance] = None


def _count(s, key: str) -> None:
    st = getattr(getattr(s, "kit", None), "stats", None)
    if isinstance(st, dict):
        st[key] = st.get(key, 0) + 1


def _twin_side(d: Dict[str, Any]) -> str:
    src = f"{d['source']} " if d.get("source") else ""
    if d.get("margin_rad") is None:
        return f"{src}fail: {str(d.get('why', d.get('reason', '')))[:120]}"
    verdict = "ok" if d.get("ok") else f"rejected ({d.get('why', d.get('reason', ''))})"
    return f"{src}{verdict} margin {d['margin_rad']:.3f} ({d.get('nearest_joint')}) reach {d.get('reach_rad')}"


def pick_twin(s, stance: Stance, arm: str, g, eef_in_object: Pose, eef: Pose, pre: Pose, res, exact_kw: Dict[str, Any], *,
              support_z: Optional[float] = None, accept: Optional[Callable[[th.Tensor], bool]] = None, where: str = "",
              row_allowed: bool = True, arms: Optional[Sequence[str]] = None, stances: Optional[Dict[str, Stance]] = None,
              twins: bool = True) -> TwinPick:
    """The library pinch ``g`` drawn for a free object, or its twin (``conventions.twin_pose``: the same pinch with the fingers
    swapped), by the orientation rule of :mod:`ispatialgym.sampling.v2.orient` (owner 2026-09-27: success, then joint-limit
    safe, then easier to reach; the library row wins within ``orient.REACH_TOL``). ``eef_in_object`` is the row's
    ``T_obj→eef``, ``eef`` / ``pre`` its grasp / pre-grasp poses and ``res`` their exact IK (``s.ikc.exact([pre, eef], stance,
    arm, **exact_kw)``; it may have failed).

    The twin's chain is solved with the same ``exact_kw``; each passing chain also offers the other finger assignment in its own
    shoulder / elbow branch (``IKChecker.twin_q``, checked when it is the one to try). A row rejected at the reach boundary
    (``ik_reach``) is final - the twin has the same wrist centre. A chain passes when it solved, ``accept`` holds at its grasp
    configuration, given ``support_z`` (the support top ignored at the grasp pose) the wrist camera clears that top by
    ``TWIN_CAMERA_CLEARANCE_M``, and the camera clears the world (``orient.camera_clear``). Margins are the smallest arm
    joint-limit margin over the two witnesses, reach the largest joint change of the pre-grasp witness from the stance
    configuration (``IKChecker.q_locked``), where the pre-grasp transit starts. ``row_allowed`` False (the caller bars the row by
    name): the row is offered in no branch.

    ``arms`` (owner 2026-09-27, first-demo v2.2; default ``[arm]``): the arms offered for the grasp's segment, the default arm
    first; ``stances[a]`` is the stance with ``a`` working (default :func:`stance_for_arm`; an arm without a collision-free stance
    is not offered). Both turns of every offered arm are proposals (``orient.with_arms``: within ``REACH_TOL`` the default arm,
    then the row), each solved with its arm's IK at its arm's stance, each arm's reach from its own starting posture; ``res`` is
    ``arm``'s row. The pick carries the chosen ``arm`` and its ``stance``; ``other`` is the chosen arm's other finger assignment."""
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.v2 import orient as O

    IK = _ik()
    ikc = s.ikc
    arms = list(arms) if arms else [arm]
    st_of: Dict[str, Any] = {}
    for a in arms:
        if a == arm:
            st_of[a] = (stances or {}).get(a, stance)
            continue
        st_a = (stances or {}).get(a) or stance_for_arm(s, stance, a)
        if st_a is not None:
            st_of[a] = st_a
    arms = [a for a in arms if a in st_of]
    multi = len(arms) > 1
    q_start = {a: _t(ikc.q_locked(st_of[a])) for a in arms} if multi else _t(ikc.q_locked(st_of[arms[0]] if arms else stance))
    gt = GL.twin(g)
    eio_t = C.twin_pose((_t(eef_in_object[0]), _t(eef_in_object[1])))
    eef_t = C.twin_pose((_t(eef[0]), _t(eef[1])))
    pre_t = IK.pre_pose(eef_t)
    row_name, twin_name = getattr(g, "name", None), gt.name
    reach_limited = res is not None and (not res.ok) and str(res.reason).startswith("ik_reach")

    def evaluate(p) -> Tuple[bool, Any, Optional[List[th.Tensor]], str]:
        side, a = p.payload[0], p.arm
        if side == "row" and p.q_first is None and a == arm and res is not None:
            if not row_allowed and res.ok:
                return False, None, None, "the row is barred by the caller"
            r = res
        elif side == "row" and not row_allowed:
            return False, None, None, "the row is barred by the caller"
        elif reach_limited and a == arm and side == "twin":
            return False, None, None, "the row failed at the arm's reach boundary; the twin has the same wrist centre"
        else:
            kw = dict(exact_kw)
            if p.q_first is not None:
                kw.update({"q_first": p.q_first, "cache": False})
            _count(s, "twin_solves")
            r = ikc.exact(p.chain, st_of[a], a, **kw)
        if not r.ok or r.q is None:
            return False, r, None, str(r.reason)
        cam = None if support_z is None else wrist_camera_clearance(p.chain[-1], float(support_z))
        if cam is not None and cam < TWIN_CAMERA_CLEARANCE_M:
            return False, r, None, f"wrist camera {cam * 1000:+.0f} mm from the support top (< {TWIN_CAMERA_CLEARANCE_M * 1000:.0f} mm)"
        if accept is not None and not bool(accept(_t(r.q))):
            return False, r, None, "fails the caller's check"
        return True, r, None, ""

    _count(s, "twin_picks")
    turns = lambda: [O.Proposal(str(row_name), 0, ("row", g, eef_in_object, eef, pre), [pre, eef])] + (  # noqa: E731
        [O.Proposal(str(twin_name), 1, ("twin", gt, eio_t, eef_t, pre_t), [pre_t, eef_t])] if twins else [])
    props = O.with_arms([(a, turns()) for a in arms]) if arms else []
    chosen, decision = O.choose(ikc, arm, props, evaluate, q_start, stage=f"grasp {where}".strip())
    opts = decision["options"]
    c_arm = chosen.proposal.arm if chosen is not None else (arms[0] if arms else arm)
    lab = (lambda name: f"{c_arm}:{name}") if multi else (lambda name: str(name))  # noqa: E731

    def side(label: str) -> Dict[str, Any]:
        """The chosen configuration for the chosen assignment, the caller's / fresh solve for the other."""
        if chosen is not None and chosen.proposal.label == label:
            return chosen.facts()
        return next((dict(o) for o in opts if o["label"] == label and o["source"] == "ik"), {"ok": False})

    take = chosen is not None and chosen.proposal.payload[0] == "twin"
    why = decision["reason"] if chosen is not None else "neither feasible"
    facts = {"stage": where, "arm": c_arm, "chosen": "twin" if take else "row", "reason": why, "source": decision["source"],
             "row": dict(side(lab(row_name)), name=row_name), "twin": dict(side(lab(twin_name)), name=twin_name), "orient": decision}
    if multi:
        facts["arm_choice"] = decision.get("arm_choice")
    if take:
        _count(s, "twin_taken")
        _count(s, f"twin_taken.{why.replace(' ', '_')}")
    log.info("grasp twin [%s] %s (%s, %s%s) | row %s: %s | twin %s: %s%s", where, twin_name if take else row_name, facts["chosen"], why,
             ", wrist-flipped branch" if decision["source"] == "map" else "", row_name, _twin_side(facts["row"]), twin_name, _twin_side(facts["twin"]),
             f" | {c_arm} arm of {arms} ({(decision.get('arm_choice') or {}).get('reason')})" if multi else "")
    ok_labels = {o["label"] for o in opts if o.get("ok") and not o.get("predicted")}
    st_c = st_of.get(c_arm, stance)
    if chosen is None:
        # the row's own verdict: its IK failure, or the check it failed after solving (camera, the caller's check)
        failed = res if (res is not None and not res.ok) else IK.IKResult(False, None, str(facts["row"].get("why") or "the row fails the camera / caller's check"), 2)
        return TwinPick(False, g, eef_in_object, eef, pre, failed, False, None, facts, c_arm, st_c)
    if take:
        other = (g, eef_in_object, eef, pre) if lab(row_name) in ok_labels else None
        return TwinPick(True, gt, eio_t, eef_t, pre_t, chosen.result, True, other, facts, c_arm, st_c)
    other = (gt, eio_t, eef_t, pre_t) if lab(twin_name) in ok_labels else None
    return TwinPick(False, g, eef_in_object, eef, pre, chosen.result, True, other, facts, c_arm, st_c)


class TurnNotOffered(Exception):
    """Raised by a task module's ``turn_recheck`` when it cannot re-check the other finger assignment of a recorded pinch; the
    message is recorded as the reason (:func:`held_turns`: not offered)."""


#: the fields of a ``GraspRecord`` its finger assignment decides (:func:`flip_grasp`)
_TURN_FIELDS = ("grasp_name", "finger_q", "eef_in_object", "pre_grasp", "grasp", "pre_place", "place", "carry", "waypoints", "extras")
#: a row the draw itself barred (``pick_twin(row_allowed=False)`` / the caller's name rule): never offered back by :func:`held_turns`
_BARRED = ("excluded by name", "the row is barred by the caller")


def flip_grasp(g) -> Dict[str, Any]:
    """Record ``g`` as the other finger assignment of its pinch: every eef pose it holds turned pi about its approach axis
    (``conventions.twin_pose`` - the carry and in-hand waypoints too: the SLERP of two twins is the twin of the SLERP),
    ``conventions.twin_name``, ``finger_q`` reversed, ``wrist_flip`` toggled. Returns the fields as they were
    (:func:`unflip_grasp`; flipping twice negates the quaternions, which the IK memory's pose keys tell apart)."""
    was = {k: getattr(g, k) for k in _TURN_FIELDS}

    def tw(p):
        return None if p is None else C.twin_pose((_t(p[0]), _t(p[1])))

    g.grasp_name = C.twin_name(str(g.grasp_name))
    g.finger_q = None if g.finger_q is None else list(reversed(list(g.finger_q)))
    for k in ("eef_in_object", "pre_grasp", "grasp", "pre_place", "place"):
        setattr(g, k, tw(getattr(g, k)))
    for k in ("carry", "waypoints"):
        setattr(g, k, None if getattr(g, k) is None else [tw(p) for p in getattr(g, k)])
    g.extras = dict(g.extras, wrist_flip=not bool(g.extras.get("wrist_flip")))
    return was


def unflip_grasp(g, was: Dict[str, Any]) -> None:
    """Put back the fields :func:`flip_grasp` returned (the very objects)."""
    for k, v in was.items():
        setattr(g, k, v)


def _held_snapshot(s, draft) -> Dict[str, Any]:
    """Everything :func:`held_turns`' re-check of the other assignment may write besides the grasp record: the program check's
    witnesses and the travel records' ``arms_q`` (``program_check.choose_departure_arms``), the waypoint policy and the carry facts
    (``verify_carry_legs``), the summary, the carry configurations and the IK memory."""
    sc = s.sidecar
    recs = list(getattr(sc, "paths", []) or []) + list(getattr(sc, "transitions", []) or [])
    return {"ik_witnesses": list(sc.ik_witnesses or []), "waypoint_policy": sc.waypoint_policy, "paths": list(getattr(sc, "paths", []) or []),
            "transitions": list(getattr(sc, "transitions", []) or []), "records": [(r, {k: (dict(v) if isinstance(v, dict) else v) for k, v in vars(r).items()}) for r in recs],
            "summary": dict(draft.summary), "solutions": dict(s.ikc.solutions),
            "carry": {k: dict(getattr(s, k)) for k in ("carry_arm_q", "carry_end_q") if isinstance(getattr(s, k, None), dict)}}


def _held_restore(s, draft, snap: Dict[str, Any]) -> None:
    sc = s.sidecar
    sc.ik_witnesses, sc.waypoint_policy = snap["ik_witnesses"], snap["waypoint_policy"]
    if hasattr(sc, "paths"):
        sc.paths[:] = snap["paths"]
    if hasattr(sc, "transitions"):
        sc.transitions[:] = snap["transitions"]
    for r, d in snap["records"]:
        vars(r).update(d)
    draft.summary.clear()
    draft.summary.update(snap["summary"])
    s.ikc.solutions.clear()
    s.ikc.solutions.update(snap["solutions"])
    for k, d in snap["carry"].items():
        getattr(s, k).clear()
        getattr(s, k).update(d)


def _turn_scored(O, ikc, label: str, chain: Optional[Dict[str, Any]], q_start: Optional[th.Tensor], arm: str, source: str, why: str = ""):
    """``orient.Scored`` of one finger assignment from its ``program_check.turn_chain``: margin = the smallest over its pick and
    held chain, reach = the largest arm-joint change of its recorded pre-grasp witness from the stance configuration."""
    ok = chain is not None and chain.get("margin") is not None
    q_pre = chain.get("q_pre") if chain else None
    reach = O.reach(ikc, th.as_tensor(q_pre, dtype=th.float32), q_start, arm) if (q_pre is not None and q_start is not None) else None
    return O.Scored(O.Proposal(label, 1 if C.split_twin(label)[1] else 0, None, [], source=source, arm=arm), ok, margin=chain.get("margin") if ok else None,
                    nearest=chain.get("joint") if ok else None, reach=reach, why=why)


def held_turns(s, draft, report: Dict[str, Any], rerun: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """The orientation rule of :mod:`ispatialgym.sampling.v2.orient` over the whole chain a library pinch decides (owner 2026-09-27:
    count the carry / pre-place / place configurations in the joint-limit ranking; ranking only). :func:`pick_twin` scored the pick
    alone, but the grasp also decides the configurations the program check tracks from its witness: the lift, an in-hand reorient,
    the carry, the pre-place, the descent and the place (``program_check.turn_chain``; the place's release retreat keeps the oracle's
    own 0.05 rad joint-limit box, never past its start, and is left out). ``report`` = the program check of the recorded assignment.
    A record whose chain there is joint-limit safe (every configuration >= ``IKChecker.JOINT_LIMIT_MARGIN``) is left alone - the
    rule ranks it first anyway. Otherwise the record is flipped to its other finger assignment (:func:`flip_grasp`), which must pass
    the task module's turn checks (``s.turn_recheck(draft, grasp, q_first)``, started from the recorded pre-grasp witness
    wrist-flipped), the carry legs (``s.verify_carry_legs``) and the program check again (``rerun``); the two are ordered by
    ``orient.prefer`` on the whole chain - safe first, neither safe: more clearance (``orient.CLEARANCE_EQ``), then the smaller reach
    of the recorded pre-grasp witness from the stance configuration, the generator's order (the library row first) within
    ``orient.REACH_TOL``. When the other assignment fails a check or is not preferred, the grasp record, the witnesses, the travel
    records, the waypoint policy, the summary, the carry configurations and the IK memory are put back exactly: a draw is never
    rejected here. Not offered (recorded with the reason): a row the draw barred, a task module without ``turn_recheck``, one that
    raises :class:`TurnNotOffered`. The decision is ``twin_choice.held`` on the grasp record plus an ``orient [held chain, <arm> arm]``
    log line in the format of ``orient.choose``; kit counters ``held_turn_checks`` / ``held_turn_switched`` / ``held_turn_kept`` /
    ``held_turn_not_offered`` (``held_turn_errors``: a fault of this step, which then keeps the recorded assignment too). Returns the
    program-check report the sidecar keeps."""
    for g in list(s.sidecar.grasps):
        flipped: Dict[str, Any] = {}
        try:
            report = _held_turn(s, draft, g, report, rerun, flipped)
        except Exception as e:  # noqa: BLE001 - the rule only ranks: a fault here keeps the recorded assignment and the instance
            if flipped:
                unflip_grasp(g, flipped["was"])
                _held_restore(s, draft, flipped["snap"])
            _count(s, "held_turn_errors")
            log.warning("held chain [%s]: %r - the recorded assignment %s stays", g.object, e, g.grasp_name)
    return report


def _held_turn(s, draft, g, report: Dict[str, Any], rerun: Callable[[], Dict[str, Any]], flipped: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`held_turns` for one grasp record; ``flipped`` holds ``was`` / ``snap`` while the record stands flipped and uncommitted."""
    import functools

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import orient as O
    from ispatialgym.sampling.v2 import program_check as PC

    tc = g.extras.get("twin_choice")
    if not isinstance(tc, dict) or not g.grasp_name:
        return report
    safe = float(s.ikc.JOINT_LIMIT_MARGIN)
    mine = PC.turn_chain(report, g.object)
    if mine["margin"] is None or mine["margin"] >= safe:
        return report
    _count(s, "held_turn_checks")
    stance = s.sidecar.stances.get(g.stance)
    q_start = _t(s.ikc.q_locked(stance)) if stance is not None else None
    label0, label1 = str(g.grasp_name), C.twin_name(str(g.grasp_name))
    a = _turn_scored(O, s.ikc, label0, mine, q_start, g.arm, "recorded")
    base = {"stage": "held chain", "arm": g.arm, "recorded": label0, "safe_margin_rad": round(safe, 4), "actions": list(mine["actions"])}
    barred = str((tc.get("row") or {}).get("why") or "") if C.split_twin(label0)[1] else ""  # only a row is ever barred
    recheck = getattr(s, "turn_recheck", None)
    why_not = f"the draw barred it ({barred})" if barred in _BARRED else (None if recheck is not None else "no turn re-check in this task module")
    b, rep2, why, theirs = None, None, "", None
    if why_not is None:
        q_first = None
        if mine["q_pre"] is not None and q_start is not None:
            q_rec = q_start.clone()
            idx = s.kit.robot.arm_control_idx[g.arm]
            q_rec[idx] = th.as_tensor(mine["q_pre"], dtype=th.float32)[idx]
            q_first = s.ikc.twin_q(q_rec, g.arm)
        flipped["snap"] = _held_snapshot(s, draft)
        flipped["was"] = flip_grasp(g)
        try:
            why = recheck(draft, g, q_first=q_first) or ""
            if not why:
                g.carry = None  # this assignment's carry leg is screened afresh
                s.verify_carry_legs(draft)
                rep2 = rerun()
        except TurnNotOffered as e:
            why_not = str(e) or "the task module cannot re-check it"
        except Reject as e:
            why = str(e)[:200]  # "[<reason>] <message>"
        except Exception as e:  # noqa: BLE001 - the recorded assignment stays; a failed re-check never costs the instance
            why = f"re-check raised {e!r}"[:200]
        theirs = PC.turn_chain(rep2, g.object) if rep2 is not None else None
        if rep2 is not None and theirs["margin"] is None:
            why = "the program check left no witnesses of its chain"
        source = "ik"
        if q_first is not None and theirs is not None and theirs["q_pre"] is not None:
            idx = s.kit.robot.arm_control_idx[g.arm]
            if float((th.as_tensor(theirs["q_pre"], dtype=th.float32)[idx] - q_first[idx]).abs().max()) < 1e-2:
                source = "map"  # the chain left from the recorded pre-grasp witness wrist-flipped
        b = _turn_scored(O, s.ikc, label1, theirs if not why else None, q_start, g.arm, source, why=why)
    if why_not is not None:
        if flipped:
            unflip_grasp(g, flipped.pop("was"))
            _held_restore(s, draft, flipped.pop("snap"))
        opts = [dict(a.facts(), where=mine["where"]), {"label": label1, "arm": g.arm, "ok": False, "why": f"not offered: {why_not}"[:200]}]
        tc["held"] = dict(base, status="not offered", chosen=label0, reason=None, why=why_not, against_default=a.proposal.rank > 0, options=opts)
        _count(s, "held_turn_not_offered")
        log.info("orient [held chain, %s arm] %s (not offered) | %s | %s not offered: %s", g.arm, label0, O._side(opts[0]), label1, why_not[:120])
        return report
    pool = sorted([x for x in (a, b) if x.ok], key=functools.cmp_to_key(O.prefer))
    chosen = pool[0]
    reason = O.reason(chosen, pool[1:])
    switched = chosen is b
    opts = [dict(a.facts(), where=mine["where"]), dict(b.facts(), **({"where": theirs["where"]} if b.ok else {}))]
    held = dict(base, status="switched" if switched else "kept", chosen=chosen.proposal.label, reason=reason, against_default=chosen.proposal.rank > 0,
                options=opts, **({"why": why} if why else {}))
    log.info("orient [held chain, %s arm] %s (%s%s)%s | %s", g.arm, chosen.proposal.label, reason, ", wrist-flipped branch" if chosen.proposal.source == "map" else "",
             " AGAINST the generator's order" if chosen.proposal.rank > 0 else "", " | ".join(O._side(o) for o in opts))
    if not switched:
        unflip_grasp(g, flipped.pop("was"))
        _held_restore(s, draft, flipped.pop("snap"))
        tc["held"] = held
        _count(s, "held_turn_kept")
        return report
    flipped.clear()  # committed: the other assignment is the record now
    tc["held"] = held
    tc["chosen"] = "twin" if C.split_twin(label1)[1] else "row"
    if isinstance(draft.summary.get("grasp"), dict):
        draft.summary["grasp"] = dict(draft.summary["grasp"], name=label1)
    _count(s, "held_turn_switched")
    return rep2


def held_arms(s, draft, report: Dict[str, Any], rerun: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """The arm of a segment ranked on its whole held chain (owner 2026-09-27, first-demo v2.2: the arm is ordered "as for the wrist
    turn", joint-limit safe over every configuration of the segment's chains; ranking only, after :func:`held_turns`).

    Predicate (side-effect free: it reads the grasp record and the program-check report): a grasp record whose arm choice left
    another viable arm (:func:`fallback_arms`) and whose held chain (``program_check.turn_chain``) is below
    ``IKChecker.JOINT_LIMIT_MARGIN``. When it does not fire nothing changes (``arm_offer.held_chain`` says why). When it fires the
    segment is switched to the other arm (the module's ``arm_switch(draft, grasp, arm)``: its stances take that arm as their working
    arm and their travel is recorded again), the grasp is re-checked with that arm (``turn_recheck``; the recorded finger assignment,
    then the other one when that fails), the carry legs and the program check run again, and ``orient.prefer`` orders the two
    segments on the whole chain (safe, more clearance, the reach of each arm's recorded pre-grasp witness from that arm's own
    stance posture; the default arm within ``orient.REACH_TOL``). The loser is put back exactly - grasp record, stances, travel
    records, witnesses, summary, carry configurations, IK memory - and a draw is never rejected here. Not offered (recorded): a
    module without ``arm_switch`` / ``turn_recheck``, or one that raises :class:`TurnNotOffered`. Records ``twin_choice.held_arm``,
    ``arm_offer.held_chain``, the log ``orient [held chain, arms] ...``, kit ``held_arm_checks`` / ``_switched`` / ``_kept`` /
    ``_not_offered`` / ``_errors``."""
    for g in list(s.sidecar.grasps):
        state: Dict[str, Any] = {}
        try:
            report = _held_arm(s, draft, g, report, rerun, state)
        except Exception as e:  # noqa: BLE001 - the rule only ranks: a fault here keeps the recorded segment and the instance
            if state:
                _held_arm_restore(s, draft, g, state)
            _count(s, "held_arm_errors")
            log.warning("held chain [%s]: %r - the recorded %s arm stays", g.object, e, g.arm)
    return report


def _held_arm_snapshot(s, draft, g) -> Dict[str, Any]:
    return {"snap": _held_snapshot(s, draft), "grasp": {k: getattr(g, k) for k in _TURN_FIELDS + ("arm",)},
            "stances": {n: (st.arms, st.idle_posture) for n, st in s.sidecar.stances.items()}}


def _held_arm_restore(s, draft, g, state: Dict[str, Any]) -> None:
    for k, v in state["grasp"].items():
        setattr(g, k, v)
    for n, (arms, idle) in state["stances"].items():
        st = s.sidecar.stances.get(n)
        if st is not None:
            st.arms, st.idle_posture = arms, idle
    _held_restore(s, draft, state["snap"])
    state.clear()


def _held_arm(s, draft, g, report: Dict[str, Any], rerun: Callable[[], Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`held_arms` for one grasp record; ``state`` holds the snapshot while the segment stands switched and uncommitted."""
    import functools

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import orient as O
    from ispatialgym.sampling.v2 import program_check as PC

    offer = g.extras.get("arm_offer") if isinstance(g.extras.get("arm_offer"), dict) else None
    others = fallback_arms(g)
    if not others:
        if offer is not None:
            offer["held_chain"] = {"fired": False, "why": "no other viable arm recorded at the grasp"}
        return report
    safe = float(s.ikc.JOINT_LIMIT_MARGIN)
    mine = PC.turn_chain(report, g.object)
    if mine["margin"] is None or mine["margin"] >= safe:
        if offer is not None:
            offer["held_chain"] = {"fired": False, "why": "the held chain is joint-limit safe" if mine["margin"] is not None else "no held chain witnesses"}
        return report
    _count(s, "held_arm_checks")
    default = default_arm(s)
    arm0, arm1 = g.arm, others[0]
    stance = s.sidecar.stances.get(g.stance)
    q_start0 = _t(s.ikc.q_locked(stance)) if stance is not None else None

    def scored(arm: str, chain: Optional[Dict[str, Any]], q_start, source: str, why: str = ""):
        ok = chain is not None and chain.get("margin") is not None
        q_pre = chain.get("q_pre") if chain else None
        reach = O.reach(s.ikc, th.as_tensor(q_pre, dtype=th.float32), q_start, arm) if (q_pre is not None and q_start is not None) else None
        return O.Scored(O.Proposal(f"{arm}:{g.grasp_name}", 0 if arm == default else 1, None, [], source=source, arm=arm), ok,
                        margin=chain.get("margin") if ok else None, nearest=chain.get("joint") if ok else None, reach=reach, why=why)

    a = scored(arm0, mine, q_start0, "recorded")
    base = {"stage": "held chain", "recorded": arm0, "other": arm1, "safe_margin_rad": round(safe, 4), "actions": list(mine["actions"])}
    switch, recheck = getattr(s, "arm_switch", None), getattr(s, "turn_recheck", None)
    why_not = None if (switch is not None and recheck is not None) else "no arm switch / turn re-check in this task module"
    b, rep2, why, theirs, q_start1 = None, None, "", None, None
    if why_not is None:
        state.update(_held_arm_snapshot(s, draft, g))
        g.extras = dict(g.extras)  # the re-checks write into the record's extras: the snapshot keeps the recorded dict
        try:
            why = switch(draft, g, arm1) or ""
            if not why:
                g.arm = arm1
                why = recheck(draft, g, q_first=None) or ""
                if why:
                    was = flip_grasp(g)
                    why2 = recheck(draft, g, q_first=None) or ""
                    if why2:
                        unflip_grasp(g, was)
                        why = f"{why}; the other finger assignment: {why2}"
                    else:
                        why = ""
            if not why:
                g.carry = None  # this segment's carry leg is screened afresh
                s.verify_carry_legs(draft)
                rep2 = rerun()
        except TurnNotOffered as e:
            why_not = str(e) or "the task module cannot switch the arm"
        except Reject as e:
            why = str(e)[:200]
        except Exception as e:  # noqa: BLE001 - the recorded segment stays; a failed re-check never costs the instance
            why = f"re-check raised {e!r}"[:200]
        st1 = s.sidecar.stances.get(g.stance)
        q_start1 = _t(s.ikc.q_locked(st1)) if (st1 is not None and why_not is None) else None
        theirs = PC.turn_chain(rep2, g.object) if rep2 is not None else None
        if rep2 is not None and theirs["margin"] is None:
            why = "the program check left no witnesses of its chain"
        b = scored(arm1, theirs if not why else None, q_start1, "ik", why=why)
    if why_not is not None:
        if state:
            _held_arm_restore(s, draft, g, state)
        opts = [dict(a.facts(), where=mine["where"]), {"label": f"{arm1}:{g.grasp_name}", "arm": arm1, "ok": False, "why": f"not offered: {why_not}"[:200]}]
        rec = dict(base, status="not offered", chosen=arm0, reason=None, why=why_not, options=opts)
        _count(s, "held_arm_not_offered")
        log.info("orient [held chain, arms] %s arm (not offered) | %s | %s arm not offered: %s", arm0, O._side(opts[0]), arm1, why_not[:120])
    else:
        pool = sorted([x for x in (a, b) if x.ok], key=functools.cmp_to_key(O.prefer))
        chosen = pool[0]
        reason = O.reason(chosen, pool[1:])
        switched = chosen is b
        opts = [dict(a.facts(), where=mine["where"]), dict(b.facts(), **({"where": theirs["where"]} if b.ok else {}))]
        rec = dict(base, status="switched" if switched else "kept", chosen=chosen.proposal.arm, reason=reason, against_default=chosen.proposal.arm != default,
                   options=opts, **({"why": why} if why else {}))
        log.info("orient [held chain, arms] %s arm (%s)%s | %s", chosen.proposal.arm, reason, " AGAINST the default arm" if chosen.proposal.arm != default else "",
                 " | ".join(O._side(o) for o in opts))
        if not switched:
            _held_arm_restore(s, draft, g, state)
            _count(s, "held_arm_kept")
        else:
            state.clear()  # committed: the other arm makes the segment now
            _count(s, "held_arm_switched")
            report = rep2
    tc = g.extras.setdefault("twin_choice", {})
    tc["held_arm"] = rec
    offer = g.extras.get("arm_offer") if isinstance(g.extras.get("arm_offer"), dict) else None
    if offer is not None:
        offer["held_chain"] = {"fired": True, "status": rec["status"], "why": rec.get("why") or rec.get("reason")}
        if rec["status"] == "switched":
            offer.update({"arm": g.arm, "reason": "held chain", "against_default": g.arm != offer.get("template_arms", [default])[0]})
    if rec["status"] == "switched" and isinstance(draft.summary.get("arm"), dict):
        draft.summary["arm"] = dict(draft.summary["arm"], arm=g.arm, reason="held chain", against_default=g.arm != default)
    return report


# ----------------------------------------------------------------------------------------------------------------
# Phase 3 step 1 — target init pose (§4.6)
# ----------------------------------------------------------------------------------------------------------------


def phase3_target(s, manip: Stance, rng, grasps: Sequence[Any], hull: th.Tensor, rest_quat, ignore_for_grasp: Sequence[Any], *, top=None,
                  support_z: Optional[float] = None, margin: float = SUPPORT_MARGIN, focus_radius: float = FOCUS_RADIUS, yaws: Sequence[float] = YAWS,
                  arm: str = "right", object_name: Optional[str] = None, scale_key: Any = None, eef_of_grasp: Optional[Callable[[Any], Pose]] = None,
                  bands: Optional[RB.ReachBands] = None, jitter: float = YAW_JITTER, n_positions: int = 5, arms: Optional[Sequence[str]] = None,
                  arm_bands: Optional[Dict[str, RB.ReachBands]] = None, fixed: Optional[Tuple[Any, float]] = None) -> Tuple[Pose, GraspRecord]:
    """§4.6 step 1: screen the 8 fixed yaws x ``grasps`` at the focus cell (12-seed IK from ``manip``, support ignored for
    the grasp pose only) -> feasible set F (stored in the band cache under (trunk, cell); empty F removes the cell);
    draw (yaw, grasp) uniformly from F, jitter ±22.5°, place the object within 5 cm of the focus with its footprint on
    the top (``margin``; a tray floor passes ``top`` = the tray raster and ``support_z`` = its floor; on neighbouring supports at the
    height of the top under the footprint, a start on a neighbour recorded as ``extras["grasp_from"]``), derive grasp =
    object pose ∘ T_obj→eef and pre-grasp 10 cm back, confirm both with the exact IK; next element of F without
    replacement on failure; ``Reject("grasp_unreachable")`` when F is exhausted. With ``generation.grasp_twins`` each drawn
    pinch may be recorded as its twin (:func:`pick_twin`; extras ``wrist_flip`` / ``twin_choice``). Returns the object pose
    and the ``GraspRecord`` (``yaw``, ``grasp_key``, ``grasp_name``, both eef poses, ``eef_in_object``).

    ``arms`` (owner 2026-09-27, first-demo v2.2; default ``[arm]``): the arms offered for the target's segment, the default arm
    first (the caller offers the other arm only where its band covers the focus). F is screened per arm at the arm's stance
    (:func:`stance_for_arm`) and stored in that arm's band (``arm_bands``; a mirrored band keeps it in memory); the draw is
    uniform over the union, and at the drawn pose the arms whose F holds the pair compete with both turns (:func:`pick_twin`).
    The record carries the chosen arm; ``manip`` is switched to that arm's working arm and idle posture in place.

    ``fixed = (xy, yaw)``: the object's pose is given (a part of a fixed layout: the embedded-figure target): F is screened at that
    pose only and drawn there without jitter or offset; it is not stored in the band (its cell and single yaw are not the band's).
    ``yaws`` other than the band's 8 (:data:`YAWS`): F over those is not stored either - the band indexes F by the 8 yaws, and an
    empty F over a subset would remove a cell other yaws reach."""
    IK = _ik()
    top = _top(s, top)
    band_yaws = len(yaws) == len(YAWS) and all(abs(C.wrap_angle(float(a) - float(b))) < 1e-6 for a, b in zip(yaws, YAWS))
    z = float(support_z) if support_z is not None else float(top.z)
    focus = manip.focus
    if focus is None:
        raise ValueError("phase3_target needs a manipulation stance with a focus (Phase 2 survivor)")
    fx = focus[:2]
    if fixed is not None:
        fx = _t(fixed[0])[:2].float()
        yaws, jitter, focus_radius, n_positions = [float(fixed[1])], 0.0, 0.0, 1
    eef_of = eef_of_grasp or (lambda g: g.pose)
    grasps = list(grasps)
    twins = grasp_twins_on(s)
    offered = list(arms) if arms else [arm]
    st_of: Dict[str, Any] = {}
    if offered == [arm]:
        st_of[arm] = manip
    for a in (offered if not st_of else []):
        st_a = manip if (a == offered[0] and (not manip.working_arms or manip.working_arms == [a])) else stance_for_arm(s, manip, a)
        if st_a is not None:
            st_of[a] = st_a
        else:
            log.info("phase3_target: no collision-free idle posture for the %s arm with the %s arm working; not offered", RB.other_arm(a), a)
    offered = [a for a in offered if a in st_of]
    if not offered:
        _reject("stance_collision", f"no offered arm has a collision-free stance at the focus ({arms or [arm]})")
    band_of = {a: (arm_bands or {}).get(a) for a in offered}
    if band_of.get(offered[0]) is None:
        band_of[offered[0]] = bands or getattr(s, "bands", None)

    def pose_at(cell, yaw):
        return object_pose_on(s, hull, rest_quat, cell, yaw, z if support_z is not None else NB.z_at(top, cell))

    F_of: Dict[str, List[Tuple[float, Any]]] = {}
    # train-fast 2026-11-01 (BAND_EVIDENCE, default on; replaces BAND_FEASIBLE_REUSE and, while on, the band write-back that deleted a
    # cell on one empty screen): F = every pair a screen of this (world, trunk, cell) ever passed minus the pairs the exact IK demoted;
    # a cell without a record, or one attempt in RESCREEN_P, is screened afresh (the set grows), and a reused F that runs dry is
    # re-screened once below
    from ispatialgym.sampling.v2 import band_evidence as EV

    from ispatialgym.sampling.v2 import band_refine as BR

    # BAND_REFINE (owner design 2026-11-01): the evidence path with world-independent keys at the stance's own trunk and only verified
    # reachability failures counted (band_refine.py); off: BAND_EVIDENCE as before
    refine = bool(fixed is None and band_yaws and BR.refine_on())
    ev_on = fixed is None and band_yaws and (refine or fast_on("BAND_EVIDENCE"))
    task_objs = list((getattr(getattr(s, "world", None), "spawned", None) or {}).values()) if refine else []
    ev_of: Dict[str, Optional[Tuple[Any, str, str]]] = {}
    by_gname: Dict[str, Any] = {}
    for g in grasps:
        by_gname.setdefault(C.split_twin(getattr(g, "name", str(g)))[0], g)

    def gname(g) -> str:
        return C.split_twin(getattr(g, "name", str(g)))[0]

    def ev_F(a: str) -> Optional[List[Tuple[float, Any]]]:
        e = ev_of.get(a)
        rec = e[0].record(e[1], e[2]) if e is not None else None
        if rec is None:
            return None
        if (BR.dead(rec) if refine else EV.cell_dead(rec)):
            return []
        out = []
        for yaw, gn in EV.feasible_pairs(rec, k=BR.k() if refine else None):
            if gn in by_gname:
                out.append((min(yaws, key=lambda y: abs(C.wrap_angle(float(yaw) - float(y)))), by_gname[gn]))
        return out

    def screen(a: str) -> List[Tuple[float, Any]]:
        e = ev_of.get(a)
        # a cell with evidence is screened afresh (the evidence is its memory); without one, the screen cache as before
        args = ([fx], list(yaws), grasps, pose_at, st_of[a], a, list(ignore_for_grasp), eef_of)
        if e is None:
            found = s.ikc.feasible_sets(*args).get(0, [])
        else:
            try:
                found = s.ikc.feasible_sets(*args, cache=False).get(0, [])
            except TypeError:  # a stand-in checker without the cache switch (tests)
                found = s.ikc.feasible_sets(*args).get(0, [])
        if e is not None and refine:
            if not e[0].seen((float(st_of[a].translate), float(st_of[a].tilt))):
                _count(s, "refine.new_key")
            verdict = record_refined_screen(s, e, found, args, task_objs, gname)
            if verdict != "found":
                _count(s, "refine.verify_other" if verdict == "other" else "refine.verify_reach_fail")
            if verdict == "reach" and BR.dead(e[0].record(e[1], e[2])):
                _count(s, "refine.blacklisted")
        elif e is not None:
            e[0].add_screen(e[1], e[2], [(float(y), gname(g)) for y, g in found])
        _count(s, "band_evidence.screen")
        return found

    screened: Dict[str, bool] = {}
    for a in offered:
        if ev_on:
            ev_of[a] = evidence_cell(s, band_of.get(a), st_of[a])
            known = ev_F(a)
            if known is not None and (not known or float(rng.random()) >= EV.rescreen_p()):
                F_of[a] = known
                screened[a] = False
                _count(s, "band_evidence.reused" if known else "band_evidence.dead_cell")
            else:
                found = screen(a)
                screened[a] = True
                F_of[a] = ev_F(a) if ev_of.get(a) is not None else found  # the union with what earlier screens found
            if (band_of.get(a) is not None) and band_of[a].meta.get("mirrored_from"):
                _count(s, "arm_band_mirrored.foci")
                if not F_of[a]:
                    _count(s, "arm_band_mirrored.empty_F")
            continue
        # train-fast 2026-10-31 (BAND_FEASIBLE_REUSE, superseded 2026-11-01 by BAND_EVIDENCE; off by default): F this sampler (or an earlier
        # job of the template, through the published band) stored for the same (trunk, focus cell) is reused instead of the two padded
        # batched-IK screens; the exact IK below confirms every drawn pair, but a reused F that runs dry is not re-screened
        F_reused = stored_feasible_set(band_of.get(a), st_of[a], grasps, yaws) if (fixed is None and band_yaws and fast_on("BAND_FEASIBLE_REUSE", default=False)) else None
        if F_reused is not None:
            F_of[a] = F_reused
            _count(s, "band_feasible_reused")
            if not getattr(s, "_band_feasible_reuse_logged", False):
                s._band_feasible_reuse_logged = True
                log.info("%s: BAND_FEASIBLE_REUSE: feasible set of the focus cell read from the band (%d pairs) instead of the batched IK screen",
                         getattr(getattr(s, "gp", None), "variant", "?"), len(F_reused))
            continue
        F_of[a] = s.ikc.feasible_sets([fx], list(yaws), grasps, pose_at, st_of[a], a, list(ignore_for_grasp), eef_of).get(0, [])
        if fixed is None and band_yaws:
            if not F_of[a] and fast_on("BAND_VERIFY_STORE"):
                # train-fast 2026-11-01 (owner): an empty screen removes the cell from the band only when it is a reachability failure
                spawned = list((getattr(getattr(s, "world", None), "spawned", None) or {}).values())
                if _screen_quiet(s, ([fx], list(yaws), grasps, pose_at, st_of[a], a, list(ignore_for_grasp), eef_of), spawned):
                    _count(s, "refine.verify_other")
                else:
                    store_feasible_set(band_of.get(a), st_of[a], F_of[a], yaws)
            else:
                store_feasible_set(band_of.get(a), st_of[a], F_of[a], yaws)
        if (band_of.get(a) is not None) and band_of[a].meta.get("mirrored_from"):
            _count(s, "arm_band_mirrored.foci")
            if not F_of[a]:
                _count(s, "arm_band_mirrored.empty_F")
    if refine and float(rng.random()) < BR.add_p():
        _refine_add(s, band_of.get(offered[0]), st_of[offered[0]], offered[0], fx, top, yaws, grasps, pose_at, ignore_for_grasp, eef_of, task_objs, rng)

    def pool_of(F_by_arm, skip=()) -> List[Tuple[float, Any, List[str]]]:
        pool: Dict[Tuple[float, str], Tuple[float, Any, List[str]]] = {}
        for a in offered:
            for yaw0, g in F_by_arm[a]:
                k = (round(float(yaw0), 4), str(getattr(g, "name", id(g))))
                if k in skip:
                    continue
                if k not in pool:
                    pool[k] = (float(yaw0), g, [])
                pool[k][2].append(a)
        return list(pool.values())

    def ev_unreachable(a: str) -> bool:
        e = ev_of.get(a)
        rec = e[0].record(e[1], e[2]) if e is not None else None
        return bool(rec and rec.get("unreachable"))

    F = pool_of(F_of)
    if not F and ev_on and not all(screened.get(a, True) or ev_unreachable(a) for a in offered):
        # a dead cell by the evidence: one fresh screen before rejecting (the counts may be another world's or an old unlucky streak);
        # a cell TILT_SNAP found out of reach at this trunk is not re-screened
        for a in offered:
            if not screened.get(a, True) and not ev_unreachable(a):
                screen(a)
                screened[a] = True
                F_of[a] = ev_F(a) or []
        F = pool_of(F_of)
    if not F and fixed is None and fast_on("TILT_SNAP", default=False) and not all(ev_unreachable(a) for a in offered):
        # TILT_SNAP (owner 2026-11-01, inside_02; default off since 2026-11-01: it lowered inside_02's yield). Phase 2 stances take continuous
        # tilts between the band's grid tilts, and a far cell the band holds at the grid tilt may be out of reach at the stance's (x 0.96 m at
        # 0.25 / -1.2, not at 0.25 / -1.07). An empty F at the stance's tilt: the stance moves to the grid tilt its band cells came from when it
        # stays collision-free and Phase 2's work-camera rules (axis angle, target in frame) still pass there, and F is screened there;
        # otherwise the cell is kept only when both grid tilts around the stance's tilt hold it - else it is out of reach at this trunk
        # (evidence: unreachable, never re-screened).
        F_snap, snap_facts = _tilt_snap(s, manip, st_of, offered, band_of, fx, yaws, grasps, pose_at, ignore_for_grasp, eef_of)
        if F_snap:
            for a in offered:
                ev_of[a] = evidence_cell(s, band_of.get(a), st_of[a]) if ev_on else None
                F_of[a] = F_snap.get(a, [])
                e = ev_of.get(a)
                if e is not None:
                    e[0].add_screen(e[1], e[2], [(float(y), gname(g)) for y, g in F_of[a]])
                    F_of[a] = ev_F(a) or F_of[a]
                screened[a] = True
            F = pool_of(F_of)
            _count(s, "tilt_snap.adopted")
            log.info("%s: TILT_SNAP: stance tilt %.3f -> grid tilt %.3f (camera rules pass, %d pairs)", getattr(getattr(s, "gp", None), "variant", "?"),
                     snap_facts["from"][1], snap_facts["grid"][1], len(F))
        else:
            _count(s, "tilt_snap." + snap_facts.get("outcome", "none"))
            if snap_facts.get("outcome") == "out_of_bracket" and ev_on and not refine:
                for a in offered:
                    e = ev_of.get(a)
                    if e is not None:
                        e[0].mark_unreachable(e[1], e[2])
    if not F and fast_on("DIAG_EMPTY_F", default=False):
        _diag_empty_f(s, fx, yaws, grasps, pose_at, st_of, offered, ignore_for_grasp, eef_of)
    if not F:
        _reject("grasp_unreachable", f"empty feasible set F at the focus {[round(float(v), 3) for v in fx.tolist()]} (trunk {manip.translate:.2f}/{manip.tilt:+.2f}, "
                                     f"{len(yaws)} yaws x {len(grasps)} grasps{', arms ' + str(offered) if len(offered) > 1 else ''})")
    name = object_name or s.bindings.one("target").name
    fails = {"support": 0, "ik": 0}
    last_reason = ""
    exact_kw = {"ignore_for_last": list(ignore_for_grasp), "scale_key": scale_key}
    tried: set = set()
    n_pairs = len(F)
    while F:
        out = _phase3_draw(s, F, rng, jitter, n_positions, focus_radius, fx, pose_at, hull, top, margin, support_z, z, eef_of, IK, st_of, exact_kw, twins,
                           band_of, name, offered, arm, manip, fails, ev_of if ev_on else None, gname, tried, verify=task_objs if refine else None)
        if isinstance(out, tuple):
            return out
        last_reason = out or last_reason
        if not (ev_on and not all(screened.get(a, True) for a in offered)):
            break
        # BAND_EVIDENCE: the reused F ran dry - one fresh screen of the arms whose F came from the evidence, then the pairs not tried yet
        for a in offered:
            if not screened.get(a, True):
                screen(a)
                screened[a] = True
                F_of[a] = ev_F(a) or []
        _count(s, "band_evidence.rescreen_dry")
        F = pool_of(F_of, skip=tried)
        n_pairs += len(F)
    _reject("grasp_unreachable", f"F ({n_pairs} pairs) exhausted without an exact-IK confirmation ({fails}; last: {last_reason[:160]})")


def _tilt_snap(s, manip: Stance, st_of: Dict[str, Stance], offered: Sequence[str], band_of, fx, yaws, grasps, pose_at, ignore_for_grasp,
               eef_of) -> Tuple[Optional[Dict[str, List[Tuple[float, Any]]]], Dict[str, Any]]:
    """TILT_SNAP (owner 2026-11-01) for :func:`phase3_target`'s empty F: ``(F per arm at the grid tilt, facts)`` when the stance moved to
    its band's grid tilt (``st_of`` and ``manip`` take that trunk), else ``(None, facts)`` with ``facts["outcome"]`` one of ``no_band``,
    ``on_grid`` (already at a grid tilt), ``collides``, ``camera`` (Phase 2's work-camera rules fail there), ``no_rule`` (no Phase 2 rule
    recorded), ``empty_at_grid``, then the bracket test: ``in_bracket`` (both grid tilts around the stance's tilt hold the cell: kept,
    an ordinary miss) or ``out_of_bracket`` (out of reach at this trunk)."""
    a0 = offered[0]
    st = st_of[a0]
    bands = band_of.get(a0)
    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(st.focus[:2]) if st.focus is not None else None
    facts: Dict[str, Any] = {"from": [float(st.translate), float(st.tilt)]}
    if bands is None or st.focus is None or not isinstance(getattr(bands, "meta", None), dict) or bands.meta.get("stand_offs"):
        facts["outcome"] = "no_band"
        return None, facts
    k = bands.nearest_trunk(float(st.translate), float(st.tilt))
    if k is None:
        facts["outcome"] = "no_band"
        return None, facts
    facts["grid"] = [float(k[0]), float(k[1])]
    outcome = None
    if abs(float(k[1]) - float(st.tilt)) <= 1e-3 and abs(float(k[0]) - float(st.translate)) <= 1e-3:
        outcome = "on_grid"
    rule = getattr(s, "_phase2_camera_rule", None)
    if outcome is None and rule is None:
        outcome = "no_rule"
    if outcome is None:
        snapped = {a: st_of[a].with_trunk(float(k[0]), float(k[1]), st_of[a].name) for a in offered}
        free = s.ikc.stances_collision_free(list(snapped.values())).tolist()
        if not all(free):
            outcome = "collides"
        else:
            cam = stance_cameras(s, [snapped[a0]])[0]
            focus = _t(manip.focus).float()
            info = manip.extras.get("focus_info")
            if float(s.world.camera_model.angle_off_axis(cam, focus)) > float(rule["max_angle"]) or not rule["boxed"](snapped[a0], cam, focus, info):
                outcome = "camera"
            else:
                found = {}
                for a in offered:
                    args = ([fx], list(yaws), grasps, pose_at, snapped[a], a, list(ignore_for_grasp), eef_of)
                    try:
                        found[a] = s.ikc.feasible_sets(*args, cache=False).get(0, [])
                    except TypeError:  # a stand-in checker without the cache switch (tests)
                        found[a] = s.ikc.feasible_sets(*args).get(0, [])
                if any(found.values()):
                    for a in offered:
                        st_of[a] = snapped[a]
                    manip.translate, manip.tilt = float(k[0]), float(k[1])
                    manip.q = None
                    manip.extras["tilt_snap"] = {"from": facts["from"], "to": facts["grid"]}
                    facts["outcome"] = "adopted"
                    return found, facts
                outcome = "empty_at_grid"
    facts["first"] = outcome
    # the bracket: the band's grid tilts at this translate just below and above the stance's tilt must both hold the cell
    tilts = sorted({float(t) for (tr, t) in bands.cells if abs(float(tr) - float(st.translate)) < 1e-6 and len(bands.cells[(tr, t)])})
    lo = max((t for t in tilts if t <= float(st.tilt) + 1e-6), default=None)
    hi = min((t for t in tilts if t >= float(st.tilt) - 1e-6), default=None)
    cell_base = bands.to_base(np.asarray(st.focus[:2].tolist()), st.base_pose2d)[0]
    both = lo is not None and hi is not None and all(bands.cell_index(float(st.translate), t, cell_base) is not None for t in {lo, hi})
    facts.update({"bracket": [lo, hi], "outcome": "in_bracket" if both else "out_of_bracket"})
    return None, facts


def _diag_empty_f(s, fx, yaws, grasps, pose_at, st_of, offered, ignore_for_grasp, eef_of, limit: int = 4) -> None:
    """train-fast 2026-11-01 diagnostic (``ISPATIALGYM_FAST_DIAG_EMPTY_F=1``, off by default; the inside_02 crop divergence): for the first
    ``limit`` empty feasible sets of the process, the screen again with every scene object out of the collision world (pairs found = the
    collision world caused it), then with each scene object within 1.5 m of the focus out alone (the culprits); one INFO line each."""
    n = s.__dict__.get("_diag_empty_f_n", 0)
    if n >= limit:
        return
    s._diag_empty_f_n = n + 1
    try:
        a = offered[0]
        st = st_of[a]
        base = list(st.base_pose2d)

        def screen(ignore_extra) -> int:
            with s.ikc.ignoring(list(ignore_extra)):
                return len(s.ikc.feasible_sets([fx], list(yaws), grasps, pose_at, st, a, list(ignore_for_grasp), eef_of, cache=False).get(0, []))

        scene = [o for o in s.world.scene.objects if o is not s.robot and not getattr(o, "visual_only", False)]
        n_all = screen(scene)
        near = []
        for o in scene:
            lo, hi = o.aabb
            c = (lo[:2] + hi[:2]) / 2
            if float(th.linalg.norm(c - _t(fx)[:2].float())) < 1.5:
                near.append(o)
        culprits = []
        if n_all:
            for o in near:
                k = screen([o])
                if k:
                    culprits.append(f"{o.name}({k})")
        log.info("DIAG_EMPTY_F %s: focus %s base %s trunk %.2f/%+.2f arm %s: %d pairs with no scene object, %d scene objects (%d within 1.5 m); "
                 "pairs with one object out: %s", getattr(getattr(s, "gp", None), "variant", "?"), [round(float(v), 3) for v in _t(fx).tolist()],
                 [round(float(v), 3) for v in base], float(st.translate), float(st.tilt), a, n_all, len(scene), len(near), culprits or "none")
    except Exception as e:  # noqa: BLE001 - a diagnostic never fails an attempt
        log.info("DIAG_EMPTY_F failed: %r", e)


def _phase3_draw(s, F, rng, jitter, n_positions, focus_radius, fx, pose_at, hull, top, margin, support_z, z, eef_of, IK, st_of, exact_kw, twins, band_of,
                 name, offered, arm, manip, fails, ev_of, gname, tried, verify=None):
    """:func:`phase3_target`'s draw over F (without replacement): ``(pose, GraspRecord)`` on the first exact-IK confirmation, else the last
    IK failure reason. With ``ev_of`` (BAND_EVIDENCE) each tried pair's outcome is counted in its arm's evidence (a pair whose positions
    all left the support counts nothing); ``tried`` collects the pairs drawn."""
    last_reason = ""
    for i in rng.permutation(len(F)):
        yaw0, g, arms_i = F[int(i)]
        a0 = arms_i[0]
        tried.add((round(float(yaw0), 4), str(getattr(g, "name", id(g)))))
        yaw = float(yaw0) + float(rng.uniform(-jitter, jitter))
        n_ik = 0
        last_fail = None
        for _ in range(int(n_positions)):
            r, a = float(rng.uniform(0.0, focus_radius)), float(rng.uniform(0.0, 2 * math.pi))
            xy = th.tensor([float(fx[0]) + r * math.cos(a), float(fx[1]) + r * math.sin(a)])
            pose = pose_at(xy, yaw)
            fp = s.world.footprint(hull, pose)
            if not top.polygon_supported(fp, margin=margin):
                fails["support"] += 1
                continue
            pose, z_here = (pose, z) if support_z is not None else NB.rest_on(top, pose, fp, NB.z_at(top, xy))
            eef = IK.grasp_pose(pose, eef_of(g))
            pre = IK.pre_pose(eef)
            if os.environ.get("ISPATIALGYM_SUPPORT_POLICY_DIAG"):
                s.ikc._diag_object = (pose, hull, getattr(g, "name", None))  # of-v2maps: the candidate object pose the policy diagnostic reports
            res = s.ikc.exact([pre, eef], st_of[a0], a0, **exact_kw)
            g_rec, eef_in_object, pick, arm_c = g, eef_of(g), None, a0
            if (twins or len(arms_i) > 1) and getattr(g, "name", None):
                pick = pick_twin(s, st_of[a0], a0, g, eef_in_object, eef, pre, res, exact_kw, support_z=z_here, where="phase3_target", arms=arms_i, stances=st_of,
                                 twins=twins)
                g_rec, eef_in_object, eef, pre, res, arm_c = pick.grasp, pick.eef_in_object, pick.eef, pick.pre, pick.res, pick.arm
                for o in pick.facts["orient"]["options"]:
                    if o.get("source") == "ik" and band_of.get(o.get("arm")) is not None and band_of[o["arm"]].meta.get("mirrored_from"):
                        _count(s, "arm_band_mirrored.ik_tried")
                        if not o.get("ok"):
                            _count(s, "arm_band_mirrored.ik_failed")
            n_ik += 1
            if not res.ok:
                fails["ik"] += 1
                last_reason = res.reason
                last_fail = (pre, eef)
                continue
            if ev_of is not None and ev_of.get(a0) is not None:
                e = ev_of[a0]
                e[0].add_exact(e[1], e[2], float(yaw0), gname(g), True)
            finger_q = list(getattr(g_rec, "finger_q", []) or []) or None
            rec = GraspRecord(name, arm_c, getattr(g_rec, "key", None), getattr(g_rec, "name", None), yaw, pre, eef, finger_q=finger_q, eef_in_object=eef_in_object)
            if pick is not None:
                rec.extras.update({"wrist_flip": pick.twin, "twin_choice": pick.facts})
            if len(offered) > 1 or arm_c != arm:
                rec.extras["arm_bands"] = {a: band_facts(band_of.get(a)) for a in offered}
            NB.note_grasp_from(top, fp, rec)
            if st_of[arm_c] is not manip:
                adopt_arm(manip, st_of[arm_c])
            rec.stance = manip.name
            return pose, rec
        if n_ik and ev_of is not None and ev_of.get(a0) is not None:
            e = ev_of[a0]
            if verify is None:
                e[0].add_exact(e[1], e[2], float(yaw0), gname(g), False)
            elif last_fail is not None:
                if not verified_exact_failure(s, last_fail[0], last_fail[1], st_of[a0], a0, exact_kw, verify):
                    _count(s, "refine.verify_other")
                else:
                    _count(s, "refine.verify_reach_fail")
                    e[0].add_exact(e[1], e[2], float(yaw0), gname(g), False)
                    from ispatialgym.sampling.v2 import band_refine as BR

                    rec = e[0].record(e[1], e[2]) or {}
                    if _pair_demoted(rec, float(yaw0), gname(g), BR.k()):
                        _count(s, "refine.blacklisted")
    return last_reason


def verified_exact_failure(s, pre: Pose, eef: Pose, stance: Stance, arm: str, exact_kw: Dict[str, Any], objs) -> bool:
    """BAND_REFINE: whether a pair's exact-IK failure is a reachability failure - the exact IK at the same [pre, eef] still fails with the
    task-spawned objects ``objs`` out of the collision world (the static scene stays). The exact IK's own rules count when they persist
    that way - ``exact IK failed``, ``ik_joint_limit`` (joint-limit margin), ``ik_reach`` (witness residual), ``approach_untrackable``:
    properties of the arm at that stance and pose; a pass means the failure was a collision with a task object."""
    with s.ikc.ignoring(list(objs)):
        again = s.ikc.exact([pre, eef], stance, arm, **exact_kw)
    return not again.ok


def _pair_demoted(rec: Dict[str, Any], yaw: float, grasp: str, k: int) -> bool:
    from ispatialgym.sampling.v2 import band_evidence as EV

    v = (rec.get("pairs") or {}).get(EV.pair_key(yaw, grasp))
    return bool(v) and EV.demoted(v, k)


def record_refined_screen(s, e: Tuple[Any, str, str], found, args, objs, gname) -> str:
    """BAND_REFINE: record a feasible-set screen of a (trunk, cell) key in its refinement store ``e`` = (store, world key, cell key) and
    return the verdict - ``found`` (pairs: recorded as positives), ``other`` (empty in this world but not with the task-spawned objects
    ``objs`` out: a collision with task objects - only the reachable pairs are recorded, nothing counts against the cell) or ``reach``
    (empty with them out too: a verified reachability failure, recorded as an empty screen)."""
    if found:
        e[0].add_screen(e[1], e[2], [(float(y), gname(g)) for y, g in found])
        return "found"
    reach = _screen_quiet(s, args, objs)
    if reach:
        e[0].add_screen(e[1], e[2], [(float(y), gname(g)) for y, g in reach])
        return "other"
    e[0].add_screen(e[1], e[2], [])
    return "reach"


def _screen_quiet(s, args, objs) -> List[Tuple[float, Any]]:
    """The feasible-set screen of ``args`` with ``objs`` (the task-spawned objects) out of the collision world: the static scene stays
    (BAND_REFINE / BAND_VERIFY_STORE reachability verification)."""
    with s.ikc.ignoring(list(objs)):
        try:
            return s.ikc.feasible_sets(*args, cache=False).get(0, [])
        except TypeError:  # a stand-in checker without the cache switch (tests)
            return s.ikc.feasible_sets(*args).get(0, [])


def _refine_add(s, bands, st: Stance, a: str, fx, top, yaws, grasps, pose_at, ignore_for_grasp, eef_of, objs, rng) -> None:
    """BAND_REFINE additions: one lattice neighbour of the focus cell missing from the stance trunk's refined cells, on the top, screened
    with the task-spawned objects out; a pair found adds the cell to the key."""
    from ispatialgym.sampling.v2 import band_refine as BR

    if isinstance(bands, NB.NeighborBands):
        bands = bands.member_for(_t(fx)[:2]) if fx is not None else None
    if bands is None or not getattr(bands, "path", None) or bands.meta.get("stand_offs") or st.focus is None:
        return
    cells = bands.band(float(st.translate), float(st.tilt))
    cb = bands.to_base(np.asarray(st.focus[:2].tolist()), st.base_pose2d)[0]
    lattice = float(bands.meta.get("lattice", 0.08) or 0.08)
    cand = []
    for nb in BR.neighbours(cells, cb, lattice):
        w = bands.to_world(np.asarray([nb], dtype=float), st.base_pose2d)[0]
        if top is None or top.contains(th.tensor([float(w[0]), float(w[1])]), 0.0):
            cand.append((nb, w))
    if not cand:
        return
    nb, w = cand[int(rng.integers(len(cand)))]
    found = _screen_quiet(s, ([th.tensor([float(w[0]), float(w[1])])], list(yaws), grasps, pose_at, st, a, list(ignore_for_grasp), eef_of), objs)
    _count(s, "refine.add_screened")
    if found:
        BR.store_of(bands).add_cell(BR.cell_key((float(st.translate), float(st.tilt)), nb),
                                    [(float(y), C.split_twin(getattr(g, "name", str(g)))[0]) for y, g in found])
        _count(s, "refine.added")


def fallback_arms(rec: GraspRecord) -> List[str]:
    """The other arms whose grasp passed when ``rec``'s arm was chosen for the segment (``orient.viable_arms`` of its choice),
    in the chooser's order: where the segment is retried when one of its later contacts fails."""
    from ispatialgym.sampling.v2 import orient as O

    dec = ((rec.extras or {}).get("twin_choice") or {}).get("orient") or {}
    return [a for a in O.viable_arms(dec) if a != rec.arm]


def segment_arm_summary(rec: GraspRecord, offered: Sequence[str]) -> Dict[str, Any]:
    """The instance summary's ``arm``: the arm of the segment, the arms offered, why (the chooser's ``arm_choice`` reason, the
    segment fallback, or a single arm), whether it is not the default arm and whether its band was a mirror."""
    ex = rec.extras or {}
    tc = ex.get("twin_choice") or {}
    ac = tc.get("arm_choice") or (tc.get("orient") or {}).get("arm_choice") or {}
    fb = ex.get("segment_fallback")
    bf = (ex.get("arm_bands") or {}).get(rec.arm) or {}
    why = "segment fallback" if fb else (ac.get("reason") or ("only arm offered" if len(offered) < 2 else "only arm feasible at the draw"))
    return {"arm": rec.arm, "offered": list(offered), "reason": why, "against_default": bool(offered) and rec.arm != offered[0],
            "fallback": fb, "mirrored_band": bool(bf.get("mirrored_from"))}


def segment_fallback_grasp(s, manip: Stance, obj_pose: Pose, rec: GraspRecord, grasps: Sequence[Any], arm: str, *, ignore_for_grasp: Sequence[Any],
                           scale_key: Any = None, support_z: Optional[float] = None, eef_of_grasp: Optional[Callable[[Any], Pose]] = None,
                           why: str = "") -> Optional[GraspRecord]:
    """The segment's fallback (owner 2026-09-27, first-demo v2.2): the preferred arm's grasp passed but a later contact of its
    segment failed (``why``), so the whole segment is retried with ``arm``: the same object pose and library pinch, both turns of
    ``arm`` by the orientation rule at ``arm``'s stance. Returns the new record (``manip`` switched to ``arm`` in place), None
    when ``arm`` cannot take the pinch there."""
    IK = _ik()
    row = C.split_twin(str(rec.grasp_name))[0]
    g = next((x for x in grasps if getattr(x, "name", None) == row), None)
    st = stance_for_arm(s, manip, arm) if g is not None else None
    if st is None:
        return None
    eef_of = eef_of_grasp or (lambda x: x.pose)
    eio = eef_of(g)
    eef = IK.grasp_pose(obj_pose, eio)
    pre = IK.pre_pose(eef)
    kw = {"ignore_for_last": list(ignore_for_grasp), "scale_key": scale_key}
    res = s.ikc.exact([pre, eef], st, arm, **kw)
    pick = pick_twin(s, st, arm, g, eio, eef, pre, res, kw, support_z=support_z, where="segment fallback", twins=grasp_twins_on(s))
    _count(s, "segment_fallbacks")
    if not pick.passed:
        log.info("segment fallback: the %s arm cannot take %s at the same pose (%s)", arm, row, str(pick.res.reason)[:120])
        return None
    _count(s, f"segment_fallback_taken.{arm}")
    finger_q = list(getattr(pick.grasp, "finger_q", []) or []) or None
    new = GraspRecord(rec.object, arm, getattr(pick.grasp, "key", None), getattr(pick.grasp, "name", None), rec.yaw, pick.pre, pick.eef, finger_q=finger_q,
                      eef_in_object=pick.eef_in_object)
    new.extras.update({k: v for k, v in (rec.extras or {}).items() if k not in ("wrist_flip", "twin_choice", "height_extra_m", "height_extra_effective_m",
                                                                                 "height_extra_clip_reason")})
    new.extras.update({"wrist_flip": pick.twin, "twin_choice": pick.facts, "segment_fallback": {"from_arm": rec.arm, "from_grasp": rec.grasp_name, "why": str(why)[:300]}})
    new.stance = rec.stance
    adopt_arm(manip, st)
    log.info("segment fallback: the %s arm takes %s after the %s arm's segment failed (%s)", arm, new.grasp_name, rec.arm, str(why)[:160])
    return new


# ----------------------------------------------------------------------------------------------------------------
# §4.6 steps 2–6 — partner, distractors, frame, whole-body checks, brace
# ----------------------------------------------------------------------------------------------------------------


#: extras key of a Phase-2 candidate's IORS start region (A1, owner 2026-10-12): ``{"left" | "right": [g_lo, g_hi]}`` - the partner sides (of the
#: target, robot frame) whose partner fits at some edge-to-edge gap of that interval when the target starts at the candidate's focus
START_REGION = "start_region"


def side_name(side: float) -> str:
    """``"left"`` for +1 (the robot's left of the target), ``"right"`` for -1."""
    return "left" if float(side) > 0 else "right"


def in_start_region(c: Stance, side: Optional[float]) -> bool:
    """Whether Phase-2 candidate ``c`` lets the attempt's partner stand on ``side``: its start region has an interval for that side. True
    without a side or without a region (the shelf layout, the tasks that compute none)."""
    reg = c.extras.get(START_REGION)
    return side is None or reg is None or side_name(side) in reg


def gap_grid(lo: float, hi: float, step: float = START_REGION_GAP_STEP) -> np.ndarray:
    """The gaps ``lo``, ``lo + step``, ... up to ``hi`` (``hi`` itself last); empty when ``hi < lo``."""
    if hi < lo:
        return np.zeros(0)
    gs = float(lo) + float(step) * np.arange(int(math.floor((float(hi) - float(lo)) / float(step) + 1e-9)) + 1)
    return gs if float(hi) - gs[-1] <= 1e-6 else np.append(gs, float(hi))


def longest_run(gs: Sequence[float], ok: Sequence[bool]) -> Optional[Tuple[float, float]]:
    """``(first, last)`` gap of the longest run of consecutive passing grid points (the first of equal runs), None when none passes."""
    best, start = None, None
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return None if best is None else (float(gs[best[0]]), float(gs[best[1]]))


def lateral_gap_masks(gs: Sequence[float], side: float, t_lat: Tuple[float, float], p_lat: Tuple[float, float], *, ends: Optional[Tuple[float, float]] = None,
                      m_end: float = 0.0, band: Optional[Tuple[float, float]] = None, reach: Optional[str] = None, t_origin: Optional[float] = None,
                      place_gap: float = 0.0) -> Dict[str, np.ndarray]:
    """§4.6 step 2, IORS start region (A1, owner 2026-10-12), the rules along the edge: per gap of ``gs`` between a target footprint spanning
    ``t_lat`` = (min, max) and a partner on ``side`` of it (+1 = the robot's left) whose footprint spans ``p_lat`` about its own origin -
    laterals along the stance tangent, + = the robot's left. Masks (absent when the rule does not apply):

    * ``ends``: both footprints at least ``m_end`` inside ``ends`` = (min, max), the edge's ends;
    * ``reach`` with ``band`` = (λ_min, λ_max), the band's laterals at the target's depth (None: no band cell there - nothing passes):
      ``"partner_centre"`` (ontop / under / inside) the centre of the partner's footprint is a band lateral; ``"target_goal"`` (nextto,
      touching without its push stance) the target's place position - its origin ``t_origin`` moved toward the partner until ``place_gap``
      is left - is one; None (touching: its push-stance screen at the draw) no mask."""
    gs = np.asarray(gs, dtype=float)
    t_lo, t_hi = float(t_lat[0]), float(t_lat[1])
    width = float(p_lat[1]) - float(p_lat[0])
    p_lo = t_hi + gs if side > 0 else t_lo - gs - width
    p_hi = p_lo + width
    out: Dict[str, np.ndarray] = {}
    if ends is not None:
        lo, hi = float(ends[0]) + float(m_end) - 1e-9, float(ends[1]) - float(m_end) + 1e-9
        out["ends"] = (p_lo >= lo) & (p_hi <= hi) & (t_lo >= lo) & (t_hi <= hi)
    if reach is not None:
        if reach == "partner_centre":
            x = 0.5 * (p_lo + p_hi)
        elif reach == "target_goal":
            x = float(t_origin) + float(side) * (gs - float(place_gap))
        else:
            raise ValueError(f"unknown reach rule {reach!r}")
        out["reach"] = np.zeros(len(gs), dtype=bool) if band is None else (x >= float(band[0])) & (x <= float(band[1]))
    return out


def polygons_supported(top, polys: th.Tensor, margin: float = SUPPORT_MARGIN) -> th.Tensor:
    """``SupportTop.polygon_supported`` for a batch of footprints ``polys`` (G, V, 2): (G,) bool - every vertex with its ``margin``
    neighbours and the centroid on free cells."""
    offs = [[0.0, 0.0]]
    if margin > 0:
        m = float(margin)
        offs += [[m, 0.0], [-m, 0.0], [0.0, m], [0.0, -m], [m, m], [-m, -m], [m, -m], [-m, m]]
    pts = (polys[:, :, None, :] + th.tensor(offs, dtype=polys.dtype)).reshape(len(polys), -1, 2)
    pts = th.cat([pts, polys.mean(dim=1, keepdim=True)], dim=1)
    xs, ys = _t(top.xs).to(polys.dtype), _t(top.ys).to(polys.dtype)
    i = th.round((pts[..., 0] - xs[0]) / float(top.resolution)).long()
    j = th.round((pts[..., 1] - ys[0]) / float(top.resolution)).long()
    inside = (i >= 0) & (j >= 0) & (i < len(xs)) & (j < len(ys))
    free = top.free[i.clamp(0, len(xs) - 1), j.clamp(0, len(ys) - 1)]
    return (inside & free).all(dim=1)


def partner_origin_xy(side: float, t_lat: Tuple[float, float], p_lat: Tuple[float, float], depth: float, stance: Stance, gs: Sequence[float]) -> th.Tensor:
    """World xy (G, 2) of the partner's origin at each gap of ``gs`` beside the target (as :func:`lateral_gap_masks` lays it out), ``depth``
    from the front edge of ``stance``."""
    g = th.tensor(np.asarray(gs, dtype=float), dtype=th.float32)
    lat = (float(t_lat[1]) - float(p_lat[0]) + g) if side > 0 else (float(t_lat[0]) - float(p_lat[1]) - g)
    base = stance.base_xy + stance.inward * (stance.edge_depth + float(depth))
    return base[None, :] + lat[:, None] * stance.tangent[None, :]


def partner_gap_masks(s, side: float, t_fp: th.Tensor, depth: float, p_hull: th.Tensor, p_quat, stance: Stance, gs: Sequence[float], *, top=None,
                      cam: Optional[Pose] = None, margin: float = SUPPORT_MARGIN, in_frame_margin: float = IN_FRAME_MARGIN, m_end: float = 0.0,
                      band: Optional[Tuple[float, float]] = None, reach: Optional[str] = None, t_origin: Optional[float] = None,
                      place_gap: float = 0.0) -> Dict[str, np.ndarray]:
    """§4.6 step 2, IORS start region (A1, owner 2026-10-12): per gap of ``gs`` for the partner at orientation ``p_quat`` on ``side`` of the
    target footprint ``t_fp`` (world polygon), at ``depth`` from the front edge of ``stance`` (the target's): :func:`lateral_gap_masks` with
    the ends of ``stance.edge`` plus ``top`` (the partner footprint on free cells of the top with ``margin``, the ``place_partner`` test) and,
    with ``cam``, ``frame`` (its AABB fully in that image)."""
    top = _top(s, top)
    tangent = stance.tangent
    t_lat = (_t(t_fp)[:, :2].float() - stance.base_xy) @ tangent
    t_lat = (float(t_lat.min()), float(t_lat.max()))
    fp = s.world.footprint(p_hull, (th.zeros(3), _t(p_quat))).float()  # about the partner's origin
    p_lat = fp @ tangent
    p_lat = (float(p_lat.min()), float(p_lat.max()))
    ends = None
    if stance.edge is not None:
        mid = float((stance.edge.mid - stance.base_xy) @ tangent)
        ends = (mid - 0.5 * float(stance.edge.length), mid + 0.5 * float(stance.edge.length))
    out = lateral_gap_masks(gs, side, t_lat, p_lat, ends=ends, m_end=m_end, band=band, reach=reach, t_origin=t_origin, place_gap=place_gap)
    if len(gs) == 0:
        return out
    xy = partner_origin_xy(side, t_lat, p_lat, depth, stance, gs)
    out["top"] = polygons_supported(top, fp[None] + xy[:, None, :], margin).numpy()
    if cam is not None:
        z = float(top.z) + s.world.bottom_offset(p_hull, p_quat) + Z_OFFSET
        corners = hull_corners_world(p_hull, (th.tensor([0.0, 0.0, z]), _t(p_quat)))
        pts = corners[None] + th.cat([xy, th.zeros(len(xy), 1)], dim=1)[:, None, :]
        out["frame"] = s.world.camera_model.inside(cam, pts.reshape(-1, 3), in_frame_margin).reshape(len(xy), -1).all(dim=1).numpy()
    return out


def gap_masks_why(gs: Sequence[float], masks: Dict[str, np.ndarray]) -> str:
    """How many of the grid's gaps pass each rule of a mask dict (``"ends 0/12, top 12/12"``) - the reason a start-region interval is empty."""
    if not len(gs):
        return "empty gap range"
    return ", ".join(f"{k} {int(np.asarray(v).sum())}/{len(gs)}" for k, v in masks.items())


def partner_pose_at(s, side: float, t_fp: th.Tensor, depth: float, p_hull: th.Tensor, p_quat, stance: Stance, g: float, *, top=None) -> Pose:
    """The partner pose of :func:`partner_gap_masks` at gap ``g``: its footprint ``g`` beside the target's along the stance tangent, its origin
    ``depth`` from the front edge, standing on the top (z = top + bottom offset + 3 mm, §3.2)."""
    top = _top(s, top)
    tangent = stance.tangent
    t_lat = (_t(t_fp)[:, :2].float() - stance.base_xy) @ tangent
    p_lat = s.world.footprint(p_hull, (th.zeros(3), _t(p_quat))).float() @ tangent
    xy = partner_origin_xy(side, (float(t_lat.min()), float(t_lat.max())), (float(p_lat.min()), float(p_lat.max())), depth, stance, [g])[0]
    return th.tensor([float(xy[0]), float(xy[1]), float(top.z) + s.world.bottom_offset(p_hull, p_quat) + Z_OFFSET]), _t(p_quat)


def partner_pose_why(s, t_fp: th.Tensor, p_hull: th.Tensor, p_pose: Pose, *, top=None, cam: Optional[Pose] = None, margin: float = SUPPORT_MARGIN,
                     min_gap: float = MIN_GAP, in_frame_margin: float = IN_FRAME_MARGIN) -> Optional[str]:
    """§4.6 step 2 checks of a drawn partner pose (those ``place_partner`` makes): why it fails - footprint off the top (``margin``), closer
    than ``min_gap`` to the target footprint ``t_fp``, not fully in ``cam``'s image - or None."""
    top = _top(s, top)
    p_fp = s.world.footprint(p_hull, p_pose)
    if not top.polygon_supported(p_fp, margin=margin):
        return "partner footprint off the top"
    if _polygons_overlap(t_fp, p_fp, min_gap):
        return f"partner closer than {min_gap:.2f} m to the target"
    if cam is not None and not in_frame(s, cam, hull_corners_world(p_hull, p_pose), in_frame_margin):
        return "partner not in the init image"
    return None


def meta_link_world_points(entries: Dict[str, Any], root_pose: Pose, scale=None) -> List[Tuple[str, th.Tensor]]:
    """A5 (owner 2026-10-12): the ``meta_links.<link>.<type>`` block of an object's ``misc/metadata.json`` (``{id: [{"position": [x, y, z],
    ...}, ...]}`` in the object frame) as ``[(id, world xyz)]`` at the object's root pose and ``scale`` (several entries of one id:
    ``id/k``)."""
    s = _t([1.0, 1.0, 1.0] if scale is None else scale).float().reshape(3)
    root = (_t(root_pose[0]).float(), _t(root_pose[1]).float())
    out = []
    for mid, subs in entries.items():
        subs = subs if isinstance(subs, list) else [subs]
        for k, e in enumerate(subs):
            p = C.transform_points(root, (_t(e["position"]).float() * s)[None])[0]
            out.append((str(mid) if len(subs) == 1 else f"{mid}/{k}", p))
    return out


def meta_spot_gaps(side: float, t_centre: Tuple[float, float], w_t: float, spots: Sequence[Tuple[float, float]], w_p: float, gap_range: Sequence[float], *,
                   ends: Optional[Tuple[float, float]] = None, m_end: float = 0.0) -> List[Tuple[int, float]]:
    """A5 (owner 2026-10-12): the spots (depth, lateral; stance frame) where a partner of half extent ``w_p``, centred there, stands on
    ``side`` (+1 = the robot's left) of a target of half extent ``w_t`` centred at ``t_centre`` (depth, lateral): beside it (the spot's depth
    within ``w_p`` of the target's), the lateral edge-to-edge gap in ``gap_range``, the target's footprint ``m_end`` inside ``ends``.
    ``[(spot index, gap)]``."""
    depth_t, lat_t = float(t_centre[0]), float(t_centre[1])
    if ends is not None and not (float(ends[0]) + m_end - 1e-9 <= lat_t - w_t and lat_t + w_t <= float(ends[1]) - m_end + 1e-9):
        return []
    out = []
    for k, (depth, lat) in enumerate(spots):
        g = float(side) * (float(lat) - lat_t) - w_t - w_p
        if abs(float(depth) - depth_t) <= w_p and float(gap_range[0]) - 1e-9 <= g <= float(gap_range[1]) + 1e-9:
            out.append((k, g))
    return out


def pose_centred_on(s, hull: th.Tensor, quat, xy, *, top=None) -> Pose:
    """The object at ``quat`` with the centre of its footprint's bounding box at world ``xy``, standing on the top (§3.2: z = top + bottom
    offset + 3 mm)."""
    top = _top(s, top)
    fp = s.world.footprint(hull, (th.zeros(3), _t(quat))).float()
    c = 0.5 * (fp.max(dim=0).values + fp.min(dim=0).values)
    return th.tensor([float(xy[0]) - float(c[0]), float(xy[1]) - float(c[1]), float(top.z) + s.world.bottom_offset(hull, quat) + Z_OFFSET]), _t(quat)


def place_partner(s, rng, t_pose: Pose, t_hull: th.Tensor, p_hull: th.Tensor, p_rest, stance: Stance, side: Optional[float], gap_range: Sequence[float], *,
                  top=None, cam: Optional[Pose] = None, z: Optional[float] = None, margin: float = SUPPORT_MARGIN, min_gap: float = MIN_GAP,
                  in_frame_margin: float = IN_FRAME_MARGIN, yaw: Optional[float] = None, n_tries: int = 40) -> Pose:
    """§4.6 step 2: the partner on ``side`` of the target (+1 = robot's left, −1 = right, None = random per try) along the
    edge at an edge-to-edge gap g ~ U(``gap_range``), same depth as the target, random yaw (or ``yaw``), footprint on the
    top (``margin``; on neighbouring supports at the height of the top under it), >= ``min_gap`` from the target, and — with ``cam``
    — fully in frame. ``Reject("partner_placement")``."""
    top = _top(s, top)
    on_top = z is None
    z = float(top.z) if z is None else float(z)
    t_fp = s.world.footprint(t_hull, t_pose)
    t_lat = (t_fp - stance.base_xy) @ stance.tangent
    depth_t, _ = stance.local(t_pose[0])
    g_lo, g_hi = float(gap_range[0]), float(max(gap_range[0], gap_range[1]))
    for _ in range(int(n_tries)):
        sd = float(side) if side is not None else (1.0 if rng.random() < 0.5 else -1.0)
        g = float(rng.uniform(g_lo, g_hi))
        yw = float(rng.uniform(-math.pi, math.pi)) if yaw is None else float(yaw)
        p_quat = C.qmul(C.quat_from_yaw(yw), _t(p_rest))
        p_pts = C.transform_points((th.zeros(3), p_quat), p_hull)[:, :2]
        p_lat = p_pts @ stance.tangent
        lat = (float(t_lat.max()) + g - float(p_lat.min())) if sd > 0 else (float(t_lat.min()) - g - float(p_lat.max()))
        xy = stance.world(depth_t, lat)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + s.world.bottom_offset(p_hull, p_quat) + Z_OFFSET])
        p_pose = (pos, p_quat)
        p_fp = s.world.footprint(p_hull, p_pose)
        if not top.polygon_supported(p_fp, margin=margin):
            continue
        if on_top:
            p_pose = NB.rest_on(top, p_pose, p_fp, z)[0]
        if _polygons_overlap(t_fp, p_fp, min_gap):
            continue
        if cam is not None and not in_frame(s, cam, hull_corners_world(p_hull, p_pose), in_frame_margin):
            continue
        return p_pose
    _reject("partner_placement", f"no partner position with gap in [{g_lo:.2f}, {g_hi:.2f}] on the top and in frame after {n_tries} tries")


def polygon_gap(a: th.Tensor, b: th.Tensor) -> float:
    """Exact clearance between two convex polygons (N, 2) / (M, 2): 0 when they intersect, else the smallest
    vertex-to-edge distance (pure torch; the edge-normal SAT alone under-estimates the gap of small rotated bodies)."""
    from ispatialgym.sampling import geometry as G

    if G.polygons_overlap(a, b, margin=0.0):
        return 0.0

    def seg_dist(p: th.Tensor, poly: th.Tensor) -> float:
        n = poly.shape[0]
        best = float("inf")
        for i in range(n):
            s0, s1 = poly[i], poly[(i + 1) % n]
            e = s1 - s0
            l2 = float(th.dot(e, e))
            t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, float(th.dot(p - s0, e)) / l2))
            best = min(best, float(th.linalg.norm(p - (s0 + t * e))))
        return best

    return min(min(seg_dist(p, b) for p in a), min(seg_dist(p, a) for p in b))


def _polygons_overlap(a: th.Tensor, b: th.Tensor, margin: float = 0.0) -> bool:
    """Whether two convex footprints are closer than ``margin`` (exact clearance, see :func:`polygon_gap`)."""
    return polygon_gap(a, b) < float(margin)


def place_distractors(s, rng, raster, z: float, target_pose: Pose, target_hull: th.Tensor, items: Sequence[Tuple[str, th.Tensor, Any]], *,
                      margin: float = TRAY_MARGIN, gap: float = 0.01, cam: Optional[Pose] = None, in_frame_margin: float = IN_FRAME_MARGIN,
                      n_tries: int = 80, budget: Optional[float] = None) -> Dict[str, Pose]:
    """§4.6 step 3 / §6.4: each ``(name, hull, rest_quat)`` item on the ``raster`` (``cells()`` world xy, ``polygon_supported
    (fp, margin)``, ``area()``) at a random cell and yaw, footprint supported (``margin``) and >= ``gap`` from every placed
    footprint, the target's included. With ``budget`` the footprints (each model's bbox at its rest orientation) may cover at most
    that fraction of the raster area (``tray_budget``). Owner 2026-10-12 (A2b): no spacing from the target along the grasp's pinch
    axis and no pad-clearance growth of the budgeted footprints - the grasp may touch the distractors (``grasps[].may_touch``).
    ``Reject("distractor_placement")``."""
    placed: Dict[str, Pose] = {}
    fps: Dict[str, th.Tensor] = {"__target__": s.world.footprint(target_hull, target_pose)}
    cells = raster.cells()
    if len(cells) == 0:
        _reject("distractor_placement", "raster has no cells")
    used_area = 0.0
    for name, hull, rest in items:
        ok = None
        for i in rng.permutation(len(cells))[: int(n_tries)]:
            xy = cells[int(i)]
            yw = float(rng.uniform(-math.pi, math.pi))
            pose = object_pose_on(s, hull, rest, xy, yw, z)
            fp = s.world.footprint(hull, pose)
            if not raster.polygon_supported(fp, margin=margin):
                continue
            if any(_polygons_overlap(fp, other, gap) for other in fps.values()):
                continue
            if cam is not None and not in_frame(s, cam, hull_corners_world(hull, pose), in_frame_margin):
                continue
            ok = (pose, fp)
            break
        if ok is None:
            _reject("distractor_placement", f"no free position for {name} (gap {gap})")
        placed[name], fps[name] = ok
        if budget is not None and hasattr(raster, "area"):
            # the model's own footprint at its rest orientation (yaw-invariant); the yawed world AABB used before inflated a long
            # thin object up to (long + short)²/2 at 45 deg and rejected sets the grounding had admitted
            ext = hull_extents(hull, rest)
            used_area += float(ext[0]) * float(ext[1])
            if used_area > float(budget) * float(raster.area()):
                _reject("distractor_placement", f"distractor footprints cover {used_area / max(raster.area(), 1e-9):.2f} > {budget} of the floor")
    return placed


def sync_moved(kit, objects) -> None:
    """train-fast 2026-10-31 (SYNC_ONLY_OBJECTS): re-pose only ``objects`` in the cuRobo world (``RobotKit.sync_world(only_objects=...)``,
    which itself falls back to the full pass when the robot root moved); a kit stand-in without that keyword (tests) and the switch off
    take the plain full sync."""
    if objects and fast_on("SYNC_ONLY_OBJECTS"):
        try:
            kit.sync_world(only_objects=list(objects))
            return
        except TypeError:
            pass
    kit.sync_world()


def whole_body_check_at(s, stances: Sequence[Stance], objects: Dict[str, Pose], *, margin: float = 0.01, require: bool = False, reason: str = "stance_collision") -> List[bool]:
    """§4.6 step 5: teleport the drafted ``objects`` (bddl instance -> pose) into the world, sync the collision world and
    whole-body check every stance (untucked arms, its trunk). Returns the per-stance mask; with ``require`` a failing
    stance raises ``Reject(reason)``. The objects stay placed (``apply_draft`` sets the same poses again)."""
    for name, pose in objects.items():
        obj = s.objects[name]
        obj.set_position_orientation(_t(pose[0]), _t(pose[1]))
        obj.keep_still()
    sync_moved(s.kit, [s.objects[n] for n in objects])  # train-fast 2026-10-31: only these objects moved
    free = s.ikc.stances_collision_free(list(stances), margin=margin).tolist()
    if require:
        bad = [st.name for st, f in zip(stances, free) if not f]
        if bad:
            _reject(reason, f"stances {bad} collide with the objects placed")
    return [bool(f) for f in free]


def brace_second_arm(s, stance: Stance, work_arm: str, work_poses: Sequence[Pose], brace_arm: str, brace_poses: Sequence[Pose], *,
                     ignore_work_last: Sequence[Any] = (), ignore_brace_last: Sequence[Any] = (), fingers: Optional[float] = None,
                     brace_only_check: Optional[Tuple[Sequence[Any], float]] = None, ignore_while_locked: Optional[Sequence[Any]] = None,
                     q_first_brace: Optional[th.Tensor] = None):
    """§4.6 step 6: the working arm's poses first (exact IK), then the other arm's [pre-contact, contact] with the working
    arm locked at its solution, combined whole-body check of the final configuration. Same steps as
    ``IKChecker.exact_both_arms``, except that the combined check ignores the objects the two final poses touch
    (``ignore_work_last`` ∪ ``ignore_brace_last``: the support and the pushed / braced object — §3.6 "ignored for the
    contact pose only"); with them in the world every pad-on-face contact reads "combined configuration collides".
    ``brace_only_check = (ignored objects, margin)``: an extra check of the brace arm's final configuration with the
    working arm back at the stance posture against everything but the ignored objects at ``margin`` — the way to test
    a stop blade against the target at its goal pose (teleported there by the caller) while the push hand, which
    touches the target, is not in the picture. ``q_first_brace``: the brace chain starts from this arm configuration (``exact``'s
    ``q_first``; a wrist-flipped solution, ``orient``). Returns the ``IKResult``."""
    IK = _ik()
    ikc = s.ikc
    r1 = ikc.exact(list(work_poses), stance, work_arm, list(ignore_work_last), cache=False, fingers=fingers)
    if not r1.ok:
        return IK.IKResult(False, None, f"{work_arm}: {r1.reason}")
    # owner 2026-09-20 21:20: the working arm is locked at its CONTACT (touching the target) while the brace chain is solved - with
    # the target in the world every configuration of the brace chain read "in collision" through the locked arm's own contact
    # before the brace arm was even evaluated (carton close 0920-191145 / 204807: 'the arms are in each other's way'). The objects
    # the locked arm touches (``ignore_while_locked``, default = the working pose's ignore set; a caller may pass just the touched
    # link) leave the world for the whole brace chain; the final pose ignores ``ignore_brace_last`` on top, as before.
    locked_ign = list(ignore_work_last) if ignore_while_locked is None else list(ignore_while_locked)
    q_f = None
    if q_first_brace is not None:
        q_f = r1.q.clone()
        idx_b = s.kit.robot.arm_control_idx[brace_arm]
        q_f[idx_b] = _t(q_first_brace)[idx_b]
    with ikc.ignoring(locked_ign):
        r2 = ikc.exact(list(brace_poses), stance, brace_arm, [o for o in ignore_brace_last if all(o is not w for w in locked_ign)], q_seed=r1.q.clone(), cache=False, fingers=fingers,
                       q_first=q_f)
    if not r2.ok:
        return IK.IKResult(False, None, f"{brace_arm} (brace): {r2.reason}")
    touched = list(ignore_work_last) + [o for o in ignore_brace_last if all(o is not w for w in ignore_work_last)]
    with ikc.ignoring(touched):
        free = bool(s.kit.collision_free(r2.q, margin=0.01)[0])
    if not free:
        return IK.IKResult(False, None, "combined configuration collides")
    if brace_only_check is not None:
        ignored, margin = brace_only_check
        q_b = ikc.q_locked(stance, fingers=fingers).clone()
        idx = s.kit.robot.arm_control_idx[brace_arm]
        q_b[idx] = r2.q[idx]
        with ikc.ignoring(list(ignored)):
            ok_b = bool(s.kit.collision_free(q_b, margin=float(margin))[0])
        if not ok_b:
            return IK.IKResult(False, None, f"{brace_arm} (brace) intersects the target at its goal pose (margin {float(margin) * 1000:.0f} mm)")
    return IK.IKResult(True, r2.q, "", len(work_poses) + len(brace_poses))


@contextmanager
def posed_in_hand(s, arm: str, obj, eef_in_object: Pose):
    """Put ``obj`` where it sits in the robot's ``arm`` per the grasp transform ``T_obj→eef`` for the duration of the
    block, so that ``kit.backend.attach_object`` (which reads the *current* object-in-eef offset) attaches it as carried;
    the object's pose is restored and the collision world re-synced afterwards. Wrap ``s.record_path`` /
    ``s.record_transition`` / place IK checks with a carried object in it."""
    p0, q0 = obj.get_position_orientation()
    p0, q0 = p0.clone(), q0.clone()
    eef = s.kit.eef_pose(s.kit.q_current(), arm)
    pos, quat = C.compose((_t(eef[0]), _t(eef[1])), C.invert((_t(eef_in_object[0]), _t(eef_in_object[1]))))
    obj.set_position_orientation(pos, quat)
    obj.keep_still()
    try:
        yield
    finally:
        obj.set_position_orientation(p0, q0)
        obj.keep_still()
        s.kit.sync_world()


def facade_v1_spec(s):
    """v1 facade only (``GroundedProblem.from_v1_spec``): the v1 ``ProblemSpec`` the entry was built from (its
    ``distractors`` pools, role options), read from the default v1 customs list; None outside the facade."""
    name = (s.gp.entry.get("_v1") or {}).get("name")
    if not name:
        return None
    from ispatialgym.sampling.spec import load_task_spec
    from ispatialgym.sampling.v2 import problem as P

    cands = [p for p in ([s.gp.customs_path] if s.gp.customs_path else []) + list(P._CUSTOMS_CANDIDATES) if p and p.endswith("task_custom_lists.json") and os.path.exists(p)]
    for path in cands:
        try:
            return load_task_spec(path, check_ready=False)[name]
        except Exception:  # noqa: BLE001 - another list, or the entry is not a v1 one
            continue
    return None


# ----------------------------------------------------------------------------------------------------------------
# §4.7 — placement stance
# ----------------------------------------------------------------------------------------------------------------


def placement_stance(s, manip: Stance, dest_xy, *, name: str = "placement", check_collision: bool = True) -> Stance:
    """§4.7: the manipulation stance shifted along the edge so that the destination centre is at the focus (same stand-off,
    translate, tilt); the base must be on free floor and — ``check_collision`` — whole-body collision-free (the caller
    repeats the check with the trays and distractors placed, ``whole_body_check_at``). ``Reject("no_placement_stance")``."""
    if manip.focus is None or manip.edge is None:
        raise ValueError("placement_stance needs a manipulation stance with a focus and an edge frame")
    dlat = float(th.dot(_t(dest_xy)[:2] - manip.focus[:2], manip.tangent))
    st = manip.shifted(lateral=manip.lateral + dlat, name=name)
    st.focus = th.tensor([float(dest_xy[0]), float(dest_xy[1]), float(manip.focus[2])])
    st.arms = list(manip.arms) if isinstance(manip.arms, list) else manip.arms
    if not s.kit.trav_free(st.base_xy):
        _reject("no_placement_stance", f"no floor at the placement stance {st.base_xy.tolist()} (lateral shift {dlat:+.2f})")
    if check_collision and not s.ikc.stance_collision_free(st):
        _reject("no_placement_stance", f"placement stance collides (lateral shift {dlat:+.2f})")
    st.extras["lateral_shift"] = dlat
    return st


__all__ = [
    "STAND_OFF", "STAND_OFFS", "TRANSLATES", "TILT_GRID", "YAWS", "FOCUS_RADIUS", "YAW_JITTER", "SUPPORT_MARGIN", "TRAY_MARGIN", "MIN_GAP",
    "TABLE_END_MARGIN", "START_REGION_GAP_STEP", "START_REGION",
    "clamp", "init_trunk_delta", "lateral_margins", "lateral_bounds", "CameraFK", "camera_table", "camera_fk",
    "opposite_face", "face_normal_world", "up_face", "cube_tilt_pair", "cube_flip_plan", "facing_coarse_yaw", "face_points",
    "distractor_spacing", "grasp_pinch_width", "exclude_lookalikes", "axial_init_yaw",
    "object_pose_on", "hull_extents", "half_extent", "height_of", "hull_corners_world", "box_corners", "library_grasps",
    "BandSpec", "ensure_bands", "compute_bands", "band_cells_world", "band_union_cells", "workspace", "store_feasible_set",
    "image_border_lateral", "r_img", "in_frame", "all_in_frame",
    "edge_frames", "edge_candidates", "phase1_edge_and_standoff", "phase1b_init_trunk", "trunk_sweep_free", "phase1c_bounds", "phase1c_lateral", "tilt_scan",
    "phase2_manipulation_stance", "phase3_target", "place_partner", "place_distractors", "whole_body_check_at", "brace_second_arm", "posed_in_hand",
    "placement_stance", "facade_v1_spec",
    "side_name", "in_start_region", "gap_grid", "longest_run", "lateral_gap_masks", "polygons_supported", "partner_origin_xy", "partner_gap_masks",
    "gap_masks_why", "partner_pose_at", "partner_pose_why", "meta_link_world_points", "meta_spot_gaps", "pose_centred_on",
]
