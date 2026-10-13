"""IORS — ontop / under, large-scale (spec §5.6 / §6.12; oracle doc §6.2 row "IORS ontop / under (large-scale)").

Roles: ``target`` (spawned, library grasps), fixed ``partner`` (bed, sofa, desk, breakfast table, countertop), fixed
``support`` (the **init surface**: a support object within ``max_partner_geodesic`` of the partner, the partner's own
top — under only — or the floor beside the partner; the class is an explicit choice of the alias / bindings, never a
runtime fallback unless the alias sets ``generation.init_support_fallback``: owner 2026-10-12 item 2b), ``floor``. Three recorded stances: **A** init (tabletop Phase 1 at the support + the partner
visible: ≥ 4 of its 8 AABB corners and its centre in frame), **B** manipulation (tabletop Phase 2 at the support, band
from the reach bands), **C** placement (base square to the partner edge nearest the init pose, stand-off {0.35, 0.45,
0.55}, lateral along the edge, trunk from the band; ontop: goal cell = the band cell inside the partner's top face
eroded by half-extent + 0.05 nearest the init pose; under: the floor strip under the partner where an upward ray hits
it ≥ target height + 0.03 above the floor, within 0.45 m of the near edge, clearance ≥ height + 0.17 (§6.12)).

B ↔ C coupling: one grasp for pick and place — (yaw, grasp) drawn uniformly from F_init ∩ F_goal (F_init = the feasible
set at B's focus cell, F_goal = the library grasps screened at the goal cell for the init yaw from C), exact IK at B
(pre-grasp + grasp, init surface ignored for the grasp pose) and at C (pre-place + place with the object attached;
ontop: the partner's top ignored for the place pose only; under: only the floor ignored). Paths A → B (§3.7 / §3.8)
and B → C (≤ ``max_partner_geodesic`` + 1 m, object attached) and the transitions are recorded; the sidecar carries
``stances.placement`` (+ ``goal_cell`` / ``init_surface`` / ``path_length_m`` in its extras and the language block),
``grasps[0]`` with ``pre_place`` / ``place`` / ``place_on``.

Reach bands (B3, ``sampling.reach_bands``): loaded through ``ReachBands.load`` for (support, target), (partner-as-
support, target) and ``--support floor [--under partner]``; on a cache miss they are computed in-process with the
module's ``BandComputer`` (the function the CLI runs) and written to the cache directory (``ISG_REACH_BANDS_COMPUTE=0``
turns the compute-on-miss into ``Reject("missing_band")``).

Tabletop phases 1a / 1b / 1c / 2 are ``v2/tabletop.py``'s (edge candidates, init trunk, lateral draw, Phase-2 survivors
with the (support, target) band). Floor init: Stance A's trunk is the translate whose optical axis is closest to the
floor cell with the target and the partner in frame (the §4.3 Δ solve has no edge on the floor; documented stand-in).

**Push-under (owner decision 2026‑09‑16 06:50, item 7; ``generation.under_mode`` = ``"push"`` default).** The floor under
a bed / sofa is out of the arm's reach (B3: 0.24–0.64 m ahead at the deep trunks only), so the under goal is reached in
two strokes from stance C (base square to the near footprint edge, facing inward = the push direction, stand-off
0.5–0.7 m, floor trunks): (1) a **drop** of the carried target on a free floor cell ``half + 0.10`` m outside the edge
(object bottom 3 cm above the floor; ``grasps[0].place`` / ``pre_place`` = the release, ``place_kind: "drop"``,
``place_on`` = the floor), (2) a **straight floor-level push** of the closed gripper from behind the target to the
**goal cell** 7 cm inside the footprint edge (the ``under`` predicate needs the centre column only; the cell must clear
target height + 3 cm) — ``contacts.push`` with ``kind: "push_under"``, ``pre`` / ``init`` (start) / ``mid`` / ``goal`` blade
poses (blade stand-off from ``tabletop.blade_standoff`` over the target's face height, approach pitched 0–60° toward
the push), ``direction``, ``distance_m``, ``push_height_m``, ``drop_cell`` / ``goal_cell``. Exact IK on the fly from C for
the drop (floor ignored, object attached) and the push chain (floor + target ignored, partner present). The program
appends ``push → retreat`` after ``place(drop) → retreat``; ``under`` is read in the epilogue; the goal-feasibility check
teleports the target to the goal cell as before.

**The arm of the segment (owner 2026-09-27, first-demo v2.2).** Grasp at B → carry → place / drop at C (→ the push-under blade:
``programs.py`` wants the grasp arm) is one single-arm segment; the other arm idles in its recorded posture (A6, re-derived per
working arm). ``tabletop.offered_arms`` offers both arms (the other arm has no role, the target is free); the other arm's bands
are its own cached bands, else the working arm's tabletop / floor bands mirrored in memory (never written, never computed
in-process). B candidates are generated for every offered arm (floor: the lateral offsets mirrored; support: Phase 2 with both
bands), each tagged ``focus_arms`` (the bands that hold the target) and ranked by the arm that reaches better; F_init is
screened per arm (the default arm always, the other one where its band holds the target). At B one coupling runs per arm
with pairs, the arms whose band holds the target first (the default arm first among them): the grasp is
``tabletop.pick_twin`` over the arms whose F_init holds the drawn pair (the
orientation rule decides; a pair the other arm wins waits for that arm's coupling), and C, the carry poses, the place, the
push lane and every record follow the segment arm (right-side laterals mirrored for the left arm). A coupling that fails
after its grasps passed hands the pairs it won to the other arm: a regrasp at B with that arm at the same target pose and
library row (``grasps[0].extras.segment_fallback``). With one arm offered (``generation.other_arm`` false) nothing changes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import neighbors as NB
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_large_scale": "ontop",
    "inter_object_relation_satisfaction-under_large_scale": "under",
    "inter_object_relation_satisfaction-inside_large_scale": "inside",
    "inter_object_relation_satisfaction-touching_large_scale": "touching",
    "inter_object_relation_satisfaction-nextto_large_scale": "nextto",
}
V1_DEFINITION_ID = {"ontop": 0, "under": 1}
#: relations registered ahead of their modes (owner 2026-10-12): relation -> its hook, which ``bind_scene_roles`` (the first stage of the
#: template setup) calls before any of the ontop / under flow; the hook returns the relation's mode class (:meth:`IORSLarge._adopt_mode`)
#: and rejects until the mode lands
MODE_HOOKS = {"inside": "_sample_inside", "touching": "_sample_touching", "nextto": "_sample_nextto"}
#: scene categories a fixed partner may bind to (B4 ``PARTNER_CATEGORIES``)
PARTNER_CATEGORIES = ("bed", "sofa", "couch", "desk", "breakfast_table", "countertop")
SUPPORT_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar", "nightstand", "console_table",
                      "side_table", "shelf", "bench")
INIT_SURFACE_CLASSES = ("support", "partner", "floor")
SCENE_INVENTORY_ROOT = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
STAND_OFFS = (0.35, 0.45, 0.55)
SUPPORT_MARGIN = 0.02
# ---- push-under (owner decision 2026‑09‑16 06:50, item 7): the under goal is reached by a drop just outside the partner's
# footprint edge and a straight floor-level push of the target under the edge ------------------------------------------------
PUSH_UNDER_INSIDE_M = 0.07          # goal = the target centre this far inside the footprint edge (5–10 cm; ``under`` needs the centre column)
PUSH_UNDER_DROP_EXTRA_M = 0.10      # drop centre = half extent along the push + this outside the edge
PUSH_UNDER_DROP_HEIGHT_M = 0.15     # object bottom above the floor at the recorded drop pose (owner 2026-10-10: the oracle releases every free drop
                                    # at once after bringing it straight down to 2 cm above the floor; from 0.25 m the sandal's wrist flipped 5-9 cm
                                    # down and the fall tumbled it, from 0.15 m at the (0.95, -1.2) drop trunk the descent comes the full way: landed
                                    # 1.2 mm aside; z < 0.08 m stays out of the arm's reach even at translate 0.95)
#: drop-stance trunks tried in turn (``generation.drop_trunks`` overrides): the deeper (0.95, -1.2) first, from which the 0.15 m drop comes down
#: to the floor, then the moderate ones (2026-09-20: (0.5, -0.6) cannot reach the sandal's drop beside the breakfast table)
PUSH_UNDER_DROP_TRUNKS = ((0.95, -1.2), (0.5, -0.6), (0.5, -0.9), (0.75, -0.6), (0.25, -0.6), (0.75, -0.9))
PUSH_UNDER_STAND_OFFS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)  # base stand-off from the edge: the drop cell 0.25–0.6 m ahead (B3: the floor is reachable 0.24–0.64 m ahead)
PUSH_UNDER_DROP_AHEAD_M = (0.25, 0.60)
PUSH_UNDER_CLEARANCE_M = 0.005      # blade clearance to the target's back face at the push start
#: trunks that reach the floor (B3 floor bands, A6 recompute: cells only at translate ≥ 0.75, tilt ≤ −0.9) + the ground half of the
#: JoyLo curve (W1, owner 2026-09-20 22:15: (2.0, 0) puts the shoulder 8 cm lower than (0.95, -1.2) and reaches the floor blade
#: 0.40-0.85 m ahead instead of 0.40-0.70; ``eef_reach_volume.GROUND_TRUNKS``)
FLOOR_TRUNKS = ((0.75, -0.9), (0.75, -1.2), (0.95, -0.9), (0.95, -1.2)) + tuple(tuple(t) for t in RB.GROUND_TRUNKS)
#: per-attempt time budget (owner 2026‑09‑16 22:55 item 4; as attachment's 094cfe3): run 0916‑190702 spent > 2 h in one push-under
#: attempt (6 B candidates × up to 56 C candidates × the F_goal feasible sets) and hit the 7200 s run timeout with nothing
#: written and no rejection recorded — the attempt now stops with ``Reject("time_budget")`` naming the stage it reached
#: (``generation.attempt_budget_s`` overrides). 900 -> 1500 s (2026-09-20): the F_init pre-screen (8 B x 16 yaws) and the
#: carry-leg checks (departure sweep, path, arrival sweep with the object attached) add ~3 min per attempt
ATTEMPT_BUDGET_S = 1500.0


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class PushUnderRecord(ContactRecord):
    """``contacts.push`` of the push-under (F6 ``PushContact`` + extras): ``kind: "push_under"``, straight floor-level
    push ``init → mid → goal`` of the closed gripper (blade behind the target along the push direction), from the
    placement stance; ``push_height_m``, ``pitch_deg``, ``blade_standoff_m``, ``half_extent_m`` (target half extent along
    the push), ``drop_cell`` / ``goal_cell`` (target centres, world xy)."""

    def __init__(self, arm: str, pre, start, mid, goal, direction, distance_m: float, object: str, stance: str = "placement", **extras):
        super().__init__("push", arm, pre, start, mid, goal, direction, distance_m, None, object, stance)
        self.extras: Dict[str, Any] = dict(extras)

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d["kind"] = "push_under"
        for k, v in self.extras.items():
            if k not in d:
                d[k] = _jsonable(v)
        return d
TARGET_JITTER = math.radians(22.5)
FOCUS_RADIUS = 0.05
YAWS = [k * math.pi / 4 for k in range(8)]


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


#: §5.6 / owner decision A5 (2026‑09‑16): the alias-level selection order of the init surface, applied by ``ground_isg``
#: through fallback aliases (primary → fallback when the primary yields nothing); a module never switches class at run time
SELECTION_ORDER = {"under": ("support", "partner", "floor"), "ontop": ("support", "floor")}


def init_surface_class(relation: str, support_binding: Any, partner_names: Sequence[str], alias_support_token: Optional[str] = None,
                       alias_partner_token: Optional[str] = None) -> str:
    """§5.6 init-surface class fixed by the alias (A5: no runtime switching): ``"floor"`` (the binding / alias token
    says floor, or the binding is a room dict), ``"partner"`` (the support binding names the partner, or the alias
    repeats the partner token ``…-<partner>-<partner>``; under only) or ``"support"`` (a support object). ``ValueError``
    for the ontop + partner combination (the goal would hold at init)."""
    names = [n for n in support_binding if isinstance(n, str)] if isinstance(support_binding, list) else []
    token = (alias_support_token or "").lower()
    ptoken = (alias_partner_token or "").lower()
    if names == ["floor"] or token in ("floor", "floors") or (isinstance(support_binding, dict) and (support_binding.get("floor") or support_binding.get("room"))):
        return "floor"
    own_top = (names and all(n in set(partner_names) for n in names)) or (not names and token and ptoken and token == ptoken)
    if own_top:
        if relation == "ontop":
            raise ValueError("ontop with the partner's own top as the init surface: the goal would hold at init")
        return "partner"
    if not names and not isinstance(support_binding, dict) and token and token in {c for c in PARTNER_CATEGORIES} and relation == "under":
        return "partner"
    return "support"


def alias_tokens(alias: str, subtask: str) -> List[str]:
    """The ``{target}-{partner}-{support}`` tokens of ``<subtask>-{target}-{partner}-{support}`` ([] without tokens)."""
    rest = alias[len(subtask):].strip("-") if alias.startswith(subtask) else ""
    return [p for p in rest.split("-") if p]


def alias_support_token(alias: str, subtask: str) -> Optional[str]:
    """The ``{support}`` token of ``<subtask>-{target}-{partner}-{support}`` (None when the alias has no tokens)."""
    parts = alias_tokens(alias, subtask)
    return parts[-1] if len(parts) >= 3 else None


def alias_partner_token(alias: str, subtask: str) -> Optional[str]:
    """The ``{partner}`` token (None when the alias has fewer than three tokens)."""
    parts = alias_tokens(alias, subtask)
    return parts[-2] if len(parts) >= 3 else None


def nearest_edge(edges: Sequence[EdgeFrame], xy) -> EdgeFrame:
    """The partner edge whose midpoint is nearest to a world xy (§5.6: the edge nearest the target's init pose)."""
    p = th.as_tensor([float(xy[0]), float(xy[1])])
    return min(edges, key=lambda e: float(th.linalg.norm(e.mid - p)))


def aabb_edges(lo, hi, overhang_from_top: Optional[Dict[str, float]] = None) -> List[EdgeFrame]:
    """The four AABB edges of a partner as EdgeFrames (inward normals), named ``+x / -x / +y / -y``."""
    lo, hi = th.as_tensor(lo, dtype=th.float32)[:2], th.as_tensor(hi, dtype=th.float32)[:2]
    cx, cy = float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2
    out = []
    for name, mid, outward, length in (("+x", (float(hi[0]), cy), (1.0, 0.0), float(hi[1] - lo[1])), ("-x", (float(lo[0]), cy), (-1.0, 0.0), float(hi[1] - lo[1])),
                                       ("+y", (cx, float(hi[1])), (0.0, 1.0), float(hi[0] - lo[0])), ("-y", (cx, float(lo[1])), (0.0, -1.0), float(hi[0] - lo[0]))):
        out.append(EdgeFrame.from_edge(name, th.tensor(mid), th.tensor(outward), length, float((overhang_from_top or {}).get(name, 0.0))))
    return out


def edge_overhang(top_edge_mid, outward, aabb_lo, aabb_hi) -> float:
    """AABB overhang beyond a top-raster edge along its outward normal (§4.2 / §5.6: ≥ 5 cm counts, else 0)."""
    corners = G.aabb_corners(th.as_tensor(aabb_lo, dtype=th.float32), th.as_tensor(aabb_hi, dtype=th.float32))[:, :2]
    n = th.as_tensor(outward, dtype=th.float32)[:2]
    over = float((corners @ n).max()) - float(th.dot(th.as_tensor(top_edge_mid, dtype=th.float32)[:2], n))
    return over if over >= 0.05 else 0.0


def erode_cells(cells: th.Tensor, boundary_xy: th.Tensor, margin: float) -> th.Tensor:
    """Cells (N, 2) farther than ``margin`` from every point of a boundary polyline / raster (the top eroded by the
    target's half-extent + 0.05)."""
    if len(cells) == 0 or len(boundary_xy) == 0:
        return cells
    d = th.cdist(cells.float(), boundary_xy.float())
    return cells[d.min(dim=1).values > margin]


def under_clearance_cells(grid: Dict[str, Any], min_clearance: float) -> th.Tensor:
    """World xy of the B4 under-clearance raster cells (``partners/<name>.json`` ``under_clearance``) whose clearance
    ≥ ``min_clearance``."""
    res = float(grid["resolution"])
    ox, oy = (float(v) for v in grid["origin"])
    g = grid["grid"]
    out = []
    for i, row in enumerate(g):
        for j, c in enumerate(row):
            if c is not None and float(c) >= min_clearance:
                out.append((ox + (i + 0.5) * res, oy + (j + 0.5) * res))
    return th.tensor(out, dtype=th.float32) if out else th.zeros(0, 2)


def under_admissible(clearance_at_goal: float, target_height: float, extra: float = 0.17) -> bool:
    """§6.12 under admissibility: clearance ≥ target height + 0.12 (gripper below the wrist) + 0.05."""
    return float(clearance_at_goal) >= float(target_height) + float(extra)


#: owner 2026-10-12 item 2 (large-scale eval picks): generation keys that relax a rule of this module; unset (None / False) keeps the
#: rule as before. The keys an alias sets are recorded in ``template_facts["relaxations"]``.
RELAXATION_KEYS = ("partner_view_min_points", "init_support_max_geodesic_m", "init_support_fallback", "under_axis_max_deg")


def enough_in_view(inside, min_points: Optional[int] = None) -> bool:
    """Owner 2026-10-12 item 2a (``generation.partner_view_min_points``): the partner's region points in frame (``inside``: one flag per
    point, the 5 % margin) - every point when ``min_points`` is None (the default, as before), else at least ``min_points`` of them
    (every point when the region has fewer)."""
    inside = th.as_tensor(inside, dtype=th.bool).reshape(-1)
    if min_points is None:
        return bool(inside.all())
    return int(inside.sum()) >= min(int(min_points), int(inside.numel()))


def intersect_feasible(F_init: Sequence[Tuple[float, Any]], F_goal: Sequence[Tuple[float, Any]], yaw_step: float = math.pi / 8) -> List[Tuple[float, Any]]:
    """F_init ∩ F_goal over (yaw, grasp) pairs: grasps compared by name, yaws by their bucket on the screening grid
    (``yaw_step``; 16 yaws = π/8). The §3.10 feasible-set cache stores floats rounded to 4 decimals (``cache._plain``),
    so a cached F_goal came back with yaws like 0.3927 against F_init's 0.39269908 and the earlier 6-decimal comparison
    emptied the intersection on every cache hit (0920-021701: 'F_init 21, F_goal 21, F 0' next to 'F 17' for the same sets)."""
    key = lambda yg: (int(round(float(yg[0]) / yaw_step)), getattr(yg[1], "name", yg[1]))  # noqa: E731
    goal_keys = {key(yg) for yg in F_goal}
    return [yg for yg in F_init if key(yg) in goal_keys]


def pair_key(yaw: float, grasp: Any, yaw_step: float = math.pi / 8) -> Tuple[int, Any]:
    """A (yaw, grasp) pair's key on the screening grid (:func:`intersect_feasible`'s): yaw bucket, grasp name."""
    return int(round(float(yaw) / yaw_step)), getattr(grasp, "name", grasp)


def union_feasible(F_by_arm: Dict[str, Sequence[Tuple[float, Any]]], yaw_step: float = math.pi / 8) -> List[Tuple[float, Any]]:
    """The usable pairs of a B over its arms (owner 2026-09-27, first-demo v2.2): the first arm's F in order, then the pairs only
    the next arms hold."""
    out: List[Tuple[float, Any]] = []
    seen = set()
    for F in F_by_arm.values():
        for yg in F:
            k = pair_key(yg[0], yg[1], yaw_step)
            if k not in seen:
                seen.add(k)
                out.append(yg)
    return out


# ---- the arm of the segment (owner 2026-09-27, first-demo v2.2): the module's right-side laterals, laid out for the right arm,
# mirrored for the left arm ----------------------------------------------------------------------------------------------------


def arm_sign(arm: str) -> float:
    """+1 for the right arm, -1 for the left: the sign of the module's right-side laterals (floor B offsets, carry poses, frontal
    push lane, side push stance) for ``arm``."""
    return 1.0 if arm == "right" else -1.0


def mirrored_offsets(offsets: Sequence[float], arm: str) -> List[float]:
    """Stance B's lateral offsets of the floor anchor cell for ``arm``: ``floor_init_lateral_offsets_m`` put c_t to the robot's right
    (the right arm's floor band), negated for the left arm (the base turned the other way, c_t to the robot's left)."""
    return [arm_sign(arm) * float(v) for v in offsets]


def offset_yaw(yaw0: float, y_off: float, dist: float) -> float:
    """Base yaw that puts the point ``dist`` away at the bearing ``yaw0`` at the base-frame lateral ``y_off`` (+ = the robot's left)."""
    return yaw0 + math.asin(-y_off / dist)


def carry_poses_for(poses: Sequence[Sequence[Any]], arm: str) -> List[Tuple[float, Optional[float], float]]:
    """``CARRY_POSES`` / ``generation.carry_poses`` (ahead, lateral, raise) for ``arm``: the lateral beside the body on the arm's side
    (the right arm's -0.35..-0.45 m positive for the left arm); None (= the lift's lateral) kept."""
    s = arm_sign(arm)
    return [(ahead, None if lat is None else s * float(lat), raise_m) for ahead, lat, raise_m in poses]


def frontal_lane_lateral(lane_lat: float, offset: float, arm: str) -> float:
    """Base lateral (along the edge tangent, + = the robot's left facing the edge) of a frontal push-under stance whose push lane at
    ``lane_lat`` lies ``offset`` to ``arm``'s side of the base (right arm: the base left of the lane)."""
    return float(lane_lat) + arm_sign(arm) * float(offset)


def side_push_pose(mid: th.Tensor, tangent: th.Tensor, inward: th.Tensor, lane_lat: float, x_ahead: float, b: float, arm: str) -> Tuple[float, float, float]:
    """Base pose of a side push-under stance: ``b`` outside the edge, ``x_ahead`` short of the lane point along the edge, facing along
    the edge with the partner on ``arm``'s side (right arm: facing +tangent, the robot's right = inward; left arm: facing -tangent)."""
    s = arm_sign(arm)
    xy = mid + tangent * (float(lane_lat) - s * float(x_ahead)) - inward * float(b)
    return float(xy[0]), float(xy[1]), math.atan2(s * float(tangent[1]), s * float(tangent[0]))


@dataclass
class SegmentArms:
    """The arms of the segment at one stance B (owner 2026-09-27, first-demo v2.2): ``stances[arm]`` (the arm working, the other one
    in its A6 idle posture) and ``F_init[arm]``, the default arm first; ``bands[arm]`` the band that shortlisted B; ``b_ik`` the grasps
    at B shared by the arms' couplings ((yaw bucket, grasp) -> None or the pick: ``yaw``, ``t_pose``, the ``arm`` that won it, the arms
    ``viable`` there, ``opts`` per arm, ``after`` the arms holding the pair whose segment had failed); ``done`` the arms whose coupling
    failed, ``failed`` a pair's first failure after its grasp, ``why`` a coupling's error; ``leg`` the version of the recorded A -> B
    leg and ``leg_ok`` the drive's clearance of the placed target per (pair, leg)."""

    stances: Dict[str, Stance]
    F_init: Dict[str, List[Tuple[float, Any]]]
    bands: Dict[str, Any] = field(default_factory=dict)
    yaw_step: float = math.pi / 8
    b_ik: Dict[Tuple[int, Any], Optional[Dict[str, Any]]] = field(default_factory=dict)
    done: List[str] = field(default_factory=list)
    failed: Dict[Tuple[int, Any], str] = field(default_factory=dict)
    why: Dict[str, str] = field(default_factory=dict)
    leg: int = 0
    leg_ok: Dict[Tuple[Any, int], Optional[str]] = field(default_factory=dict)

    @property
    def arms(self) -> List[str]:
        return list(self.stances)

    def arms_of(self, yaw: float, grasp: Any) -> List[str]:
        """The arms whose F_init holds the pair, in order (the arms a grasp at B offers)."""
        k = pair_key(yaw, grasp, self.yaw_step)
        return [a for a in self.stances if any(pair_key(y, g, self.yaw_step) == k for y, g in self.F_init.get(a, []))]

    def role(self, chosen: str, viable: Sequence[str], arm: str) -> str:
        """What a pair whose grasp at B went to ``chosen`` is in ``arm``'s coupling: ``own`` (chosen is the arm), ``defer`` (the chosen
        arm's coupling has not run yet), ``fallback`` (the chosen arm's coupling failed and ``arm`` was viable at the grasp) or
        ``none``."""
        if chosen == arm:
            return "own"
        if chosen not in self.done:
            return "defer"
        return "fallback" if arm in viable else "none"


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*RELATION_OF_SUBTASK)
class IORSLarge(SamplerV2):
    relation_mode: Optional[type] = None  # the relation's mode class once adopted (inside / touching / nextto)

    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = RELATION_OF_SUBTASK[self.subtask]
        if self.gp.bddl is None and self.relation in V1_DEFINITION_ID:
            self.definition_id = V1_DEFINITION_ID[self.relation]
        self.hull: Optional[th.Tensor] = None
        self.rest: th.Tensor = G.IDENTITY_QUAT.clone()
        self.bands: Optional[RB.ReachBands] = None  # ``tabletop.ensure_bands`` caches the init-support band here
        self.band_spec: Optional[TT.BandSpec] = None  # what ``ensure_bands`` computes on a miss (set once the grasps are loaded)
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.surface: Dict[str, Any] = {}  # kind, obj, top, band, arm_bands, edge, init, manip candidates
        self.place: Dict[str, Any] = {}  # partner geometry for C: top / under cells, edges, band
        self.floor_z = 0.0
        self.seg_arm: str = TT.default_arm(self)  # the arm of the grasp -> carry -> place -> release segment being coupled

    # ------------------------------------------------------------------ roles

    def bind_scene_roles(self):
        if self.relation in MODE_HOOKS and self.relation_mode is None:
            self._adopt_mode(getattr(self, MODE_HOOKS[self.relation])())
            return self.bind_scene_roles()  # the mode's own, else the ontop / under flow from here on with the mode's overrides
        self.bind_floor_v2()
        req = self.choice.bindings.get("partner")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if names:
            partner = self.world.obj(names[0])
        else:
            cats = list((req or {}).get("category", PARTNER_CATEGORIES)) if isinstance(req, dict) or req is None else list(PARTNER_CATEGORIES)
            rooms = set(self.world.rooms or [])
            cands = sorted([o for o in self.world.scene.objects if o.category in cats and (not rooms or set(o.in_rooms or []) & rooms)], key=lambda o: o.name)
            if not cands:
                raise Reject("spec", f"{self.gp.alias}: no partner candidate of {cats} in {sorted(rooms)}")
            partner = cands[0]
        self.bind_scene_object("partner", partner)

    @property
    def partner(self):
        return self.obj("partner")

    @property
    def target(self):
        return self.obj("target")

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    @property
    def pname(self) -> str:
        return self.bindings.one("partner").name

    def support_objects(self) -> List[Any]:
        """The init surface when it is a piece of furniture (parking keep-away, grasp ignore) and the neighbouring supports its top
        is united with; the floor is handled by :meth:`grasp_ignore_objects` (its AABB would forbid every parking cell)."""
        out = []
        if self.surface.get("obj") is not None:
            out.append(self.surface["obj"])
        return out + NB.neighbor_objects(self)

    def grasp_ignore_objects(self) -> List[Any]:
        """§5.6: the init surface ignored for the grasp pose only (the floor for the floor init)."""
        return [self.obj("floor")] if self.surface.get("kind") == "floor" else self.support_objects()

    # duck-typed attributes the shared tabletop phases read (``tabletop.py`` / the TEMP stand-in of ``iors_small``)
    @property
    def support(self):
        return self.surface.get("obj")

    @property
    def top(self) -> Optional[SupportTop]:
        return self.surface.get("top")

    def num(self, key: str, default: float) -> float:
        v = self.gp.gen(key)
        if v is None and key.endswith("_m"):
            v = self.gp.gen(key[:-2])  # v1 spelling (max_partner_geodesic)
        return float(default if v is None else v)

    # ------------------------------------------------------------------ init surface (§5.6)

    def surface_candidates(self) -> Tuple[str, List[Any]]:
        b = self.choice.bindings
        req = b.get("support")
        try:
            kind = init_surface_class(self.relation, req, [self.partner.name], alias_support_token(self.gp.alias, self.subtask),
                                      alias_partner_token(self.gp.alias, self.subtask))
        except ValueError as e:
            raise Reject("spec", str(e))
        self.template_facts["init_surface_selection_order"] = list(SELECTION_ORDER.get(self.relation, ()))
        if kind == "floor":
            return kind, [self.obj("floor")]
        if kind == "partner":
            return kind, [self.partner]
        # owner 2026-10-12 item 2b: ``generation.init_support_max_geodesic_m`` drops a support, named or by category, whose geodesic to
        # the partner is None or larger (unset: named supports are taken as bound, category ones within ``max_partner_geodesic_m``);
        # ``self.support_drops`` = {name: why} for the fallback's reason
        cap = self.gp.gen("init_support_max_geodesic_m")
        self.support_drops: Dict[str, str] = {}
        names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
        if names:
            if cap is None:
                return kind, [self.world.obj(n) for n in names]
            out = []
            for n in names:
                try:
                    o = self.world.obj(n)
                except KeyError:
                    self.support_drops[n] = "missing from the scene"
                    continue
                if self._support_near_partner(o, float(cap)):
                    out.append(o)
            self.template_facts["init_support_dropped"] = dict(self.support_drops)
            return kind, out
        cats = list((req or {}).get("category", SUPPORT_CATEGORIES)) if isinstance(req, dict) or req is None else list(SUPPORT_CATEGORIES)
        rooms = set(self.world.rooms or [])
        cands = [o for o in self.world.scene.objects if o.category in cats and o.name in self.world.scene_object_names and (not rooms or set(o.in_rooms or []) & rooms)]
        cands = [o for o in cands if o is not self.partner]
        max_geo = self.num("max_partner_geodesic_m", 3.0) if cap is None else float(cap)
        out = [o for o in sorted(cands, key=lambda o: o.name) if self._support_near_partner(o, max_geo)]
        if not cands:
            self.support_drops["category"] = f"no {cats} object in the rooms {sorted(rooms)}"
        if cap is not None:
            self.template_facts["init_support_dropped"] = dict(self.support_drops)
        return kind, out

    def _support_near_partner(self, o, max_geo: float) -> bool:
        """The support's geodesic to the partner exists and is ≤ ``max_geo`` (else noted in ``self.support_drops``)."""
        dist = self.geodesic_between(o, self.partner)
        if dist is not None and dist <= max_geo:
            return True
        self.support_drops[o.name] = "no geodesic to the partner" if dist is None else f"geodesic {dist:.2f} m > {max_geo:.2f} m"
        return False

    def init_support_fallback(self, kind: str) -> Optional[Dict[str, Any]]:
        """Owner 2026-10-12 item 2b (``generation.init_support_fallback``, default off: A5's "no runtime class switching"): when no support
        is left (too far, missing from the scene, none of the category), ontop falls back to the floor init and under to the partner's own
        top. Returns ``{from, to, reason}`` (recorded as ``template_facts.init_surface.fallback``) or None."""
        if kind != "support" or not bool(self.gp.gen("init_support_fallback", False)):
            return None
        drops = getattr(self, "support_drops", {}) or {}
        reason = "; ".join(f"{n}: {why}" for n, why in drops.items()) or "no support candidate"
        return {"from": "support", "to": "floor" if self.relation == "ontop" else "partner", "reason": reason}

    def near_side_point(self, obj, toward_xy, off: float = 0.5) -> th.Tensor:
        """A point 0.5 m outside the AABB side of ``obj`` that faces ``toward_xy`` (geodesic anchor)."""
        lo, hi = obj.aabb
        p = th.as_tensor([float(toward_xy[0]), float(toward_xy[1])])
        near = th.minimum(th.maximum(p, lo[:2].float()), hi[:2].float())
        d = p - near
        n = float(th.linalg.norm(d))
        return near + (d / n if n > 1e-6 else th.tensor([1.0, 0.0])) * off

    def geodesic_between(self, a, b) -> Optional[float]:
        """``kit.geodesic`` (OmniGibson's A* on the robot-eroded map) between the :meth:`near_side_point` anchors of ``a`` and ``b``. That
        A* finds no path when an anchor lies off the eroded map - in a support whose AABB holds the partner (a drop-in sink set in the
        counter run), on a neighbour (an L-joined counter), in the erosion band in front of a counter -, so then both anchors move to
        their nearest eroded-free cell (:meth:`free_cell_near`) and the geodesic is taken between those."""
        ca, cb = a.aabb_center[:2].float(), b.aabb_center[:2].float()
        pa, pb = self.near_side_point(a, cb), self.near_side_point(b, ca)
        dist = self._geodesic(pa, pb)
        if dist is None:
            ends = [self.free_cell_near(p) for p in (pa, pb)]
            if all(e is not None for e in ends):
                dist = self._geodesic(*ends)
        return dist

    def _geodesic(self, pa, pb) -> Optional[float]:
        try:
            _, dist = self.kit.geodesic(pa, pb)
        except Exception:  # noqa: BLE001
            return None
        return dist

    def free_cell_near(self, xy, max_cells: int = 20) -> Optional[th.Tensor]:
        """Centre (world xy) of the eroded-free traversability cell nearest ``xy`` (``navigation.nearest_free_cell``: ``xy``'s own cell when
        free, None beyond ``max_cells``). The centre, not the corner ``map_to_world`` gives: ``world_to_map`` truncates, and a corner can
        come back as the blocked neighbour."""
        ij = NAV.nearest_free_cell(NAV.eroded_map(self.kit), NAV._to_map(self.kit, xy), max_cells)
        if ij is None:
            return None
        return NAV._to_world(self.kit, ij) + 0.5 * float(self.kit.world.scene.trav_map.map_resolution)

    def prepare_bindings(self):
        try:
            self._prepare_bindings()
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def _prepare_bindings(self):
        """Target geometry and grasps, the partner's placement geometry, then the init surface: one candidate of the
        grounded class drawn uniformly, the rest of the class in random order; the support role bound (aliased to the
        partner / floor bound when it is the same object)."""
        relaxed = {k: self.gp.gen(k) for k in RELAXATION_KEYS if self.gp.gen(k) not in (None, False)}
        if relaxed:
            self.template_facts["relaxations"] = _jsonable(relaxed)
            log.info("%s: relaxed rules (owner 2026-10-12 item 2): %s", self.gp.alias, relaxed)
        tgt = self.target
        self.hull = self.world.local_hull(tgt)
        self.grasps, self.grasp_key, self.rest = TT.library_grasps(self, "target")
        self.band_spec = self.make_band_spec()
        self.floor_z = float(self.obj("floor").aabb[1][2])
        self.prepare_partner()
        kind, cands = self.surface_candidates()
        self.init_fallback = self.init_support_fallback(kind) if not cands else None
        if self.init_fallback is not None:
            log.info("%s: no init support left (%s): the %s init instead (generation.init_support_fallback)", self.gp.alias, self.init_fallback["reason"],
                     self.init_fallback["to"])
            kind = self.init_fallback["to"]
            cands = [self.obj("floor")] if kind == "floor" else [self.partner]
        if not cands:
            raise Reject("no_init_surface", f"{self.gp.alias}: no candidate of the init-surface class {kind!r}")
        rng = np.random.default_rng(self.ctx.base_seed)
        order = [cands[int(i)] for i in rng.permutation(len(cands))]
        errors = []
        for cand in order:
            try:
                self.try_surface(kind, cand, rng)
            except Reject as e:
                errors.append(f"{cand.name}: {e.reason}: {e}")
                continue
            if kind == "floor":
                self.alias_role("support", self.bindings.one("floor"))
            elif kind == "partner":
                self.alias_role("support", self.bindings.one("partner"))
            else:
                self.bind_scene_object("support", cand)
            self.template_facts["init_surface"] = {"class": kind, "name": cand.name, "tried": len(errors) + 1,
                                                   **({"fallback": self.init_fallback} if self.init_fallback is not None else {})}
            return
        raise Reject("no_init_surface", f"{self.gp.alias}: no {kind} candidate admits stances A–C: " + "; ".join(errors[-3:]))

    # ------------------------------------------------------------------ partner geometry for C

    def partner_record(self) -> Optional[Dict[str, Any]]:
        p = os.path.join(SCENE_INVENTORY_ROOT, self.world.scene_model, "partners", f"{self.partner.name}.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def target_dims(self) -> Tuple[float, float]:
        pts = G.local_points_world(self.hull, (th.zeros(3), self.rest))
        ext = pts.max(dim=0).values - pts.min(dim=0).values
        return float(ext[:2].max()) / 2, float(ext[2])

    def prepare_partner(self):
        partner = self.partner
        half, height = self.target_dims()
        margin = self.num("goal_cell_margin_m", 0.05) + half
        lo, hi = partner.aabb
        if self.relation == "ontop":
            top = self.support_top(partner)
            if int(top.free.sum()) < 50:
                raise Reject("partner_top_small", f"{partner.name}: too little free top")
            cells = top.cells(free_only=True)
            # boundary = raster cells with a missing neighbour (the face outline) — eroded by half-extent + 0.05
            rim = self._raster_rim(top)
            goal_cells = erode_cells(cells, rim, margin)
            edges = []
            for name, mid, normal, length in top.edges():
                edges.append(EdgeFrame.from_edge(name, mid, normal, length, edge_overhang(mid, normal, lo, hi)))
            self.place = {"top": top, "cells": goal_cells, "edges": edges, "z": top.z, "height": height, "half": half}
            if len(goal_cells) == 0:
                raise Reject("partner_too_small", f"{partner.name}: no top cell survives the erosion by {margin:.2f} m")
        else:
            rec = self.partner_record()
            need = height + self.num("under_ray_extra_m", 0.03)
            if rec is not None and rec.get("under_clearance"):
                cells = under_clearance_cells(rec["under_clearance"], need)
                clear_map = rec["under_clearance"]
            else:
                cells, clear_map = self._under_raycast(partner, need), None
            edges = aabb_edges(lo, hi)
            if self.push_under:
                # push-under: goal cells sit at the footprint edge (centre 5–10 cm inside; the target's outer part may stick
                # out) — keep the clear cells within 0.12 m inside any AABB edge
                if len(cells):
                    depth = th.stack([th.minimum(cells[:, 0] - lo[0].float(), hi[0].float() - cells[:, 0]),
                                      th.minimum(cells[:, 1] - lo[1].float(), hi[1].float() - cells[:, 1])], dim=1).min(dim=1).values
                    cells = cells[(depth >= 0.0) & (depth <= PUSH_UNDER_INSIDE_M + 0.05)]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell within {PUSH_UNDER_INSIDE_M + 0.05:.2f} m of its footprint edge clears {need:.2f} m")
            else:
                # inside the footprint shrunk by half-extent + 0.05 and within 0.45 m of a near AABB edge
                fp_lo, fp_hi = lo[:2].float() + margin, hi[:2].float() - margin
                if len(cells):
                    inside = ((cells >= fp_lo) & (cells <= fp_hi)).all(dim=1)
                    cells = cells[inside]
                self.place = {"cells": cells, "edges": edges, "z": self.floor_z, "height": height, "half": half, "clearance": clear_map, "need": need}
                if len(cells) == 0:
                    raise Reject("under_clearance", f"{partner.name}: no floor cell under it clears {need:.2f} m inside the footprint eroded by {margin:.2f} m")
        self.template_facts["partner"] = {"name": partner.name, "n_goal_cells": int(len(self.place["cells"])), "edges": [e.to_json() for e in self.place["edges"]]}

    def _raster_rim(self, top: SupportTop) -> th.Tensor:
        m = top.mask
        pad = th.nn.functional.pad(m.float()[None, None], (1, 1, 1, 1))[0, 0] > 0.5
        interior = pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        rim = m & ~interior
        idx = th.nonzero(rim)
        return th.stack([top.xs[idx[:, 0]], top.ys[idx[:, 1]]], dim=1) if len(idx) else th.zeros(0, 2)

    def _under_raycast(self, partner, need: float, res: float = 0.05) -> th.Tensor:
        lo, hi = partner.aabb
        xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
        ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        starts = th.cat([pts, th.full((len(pts), 1), self.floor_z + 0.005)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(hi[2]) + 0.5
        hits = self.world.raycast(starts, ends)
        prims = {l.prim_path for l in partner.links.values()}
        keep = [k for k, h in enumerate(hits) if h["hit"] and h["rigidBody"] in prims and float(h["position"][2]) - self.floor_z >= need]
        return pts[keep] if keep else th.zeros(0, 2)

    # ------------------------------------------------------------------ push-under helpers (owner decision 2026‑09‑16 item 7)

    @property
    def push_under(self) -> bool:
        """The under goal is reached by drop + straight floor push (``generation.under_mode`` = ``"push"`` default)."""
        return self.relation == "under" and str(self.gp.gen("under_mode", "push")) == "push"

    #: ground-like scene categories whose AABB spans rooms (the garden lawn's AABB covers the child's room floor strip)
    GROUND_CATEGORIES = ("floors", "ceilings", "walls", "lawn", "driveway", "roof", "pool", "carpet", "rug", "mat", "doormat", "fence", "rail_fence")
    GROUND_FOOTPRINT_M2 = 15.0

    def floor_blocking_objects(self) -> List[Any]:
        """Scene objects whose AABB blocks a floor cell for the target / the drop: not ground-like (categories above, or a
        footprint > 15 m²), not the robot, not spawned, ≥ 5 cm tall (flat floor coverings are part of the floor)."""
        out = []
        for o in self.world.scene.objects:
            if o.category in self.GROUND_CATEGORIES or o is self.robot or o.name in self.world.spawned:
                continue
            lo, hi = o.aabb
            if float(hi[2] - lo[2]) < 0.05:
                continue
            if float((hi[0] - lo[0]) * (hi[1] - lo[1])) > self.GROUND_FOOTPRINT_M2:
                continue
            out.append(o)
        return out

    def _free_floor_cell(self, xy, why: Optional[Dict[str, int]] = None) -> bool:
        """Free floor for the drop: un-eroded traversable, in the task rooms, ≥ 2 cm outside every object AABB (flat
        floor coverings < 5 cm ignored; the AABB list is built once per template). ``why`` counts the failing filter."""
        p = th.as_tensor([float(xy[0]), float(xy[1])])
        if not self.kit.trav_free(p):
            if why is not None:
                why["not_traversable"] = why.get("not_traversable", 0) + 1
            return False
        if not P0.in_task_rooms(self, p):
            if why is not None:
                why["outside_rooms"] = why.get("outside_rooms", 0) + 1
            return False
        aabbs = self.__dict__.get("_floor_aabbs")
        if aabbs is None:
            aabbs = [(o.name, o.aabb[0][:2].float() - 0.02, o.aabb[1][:2].float() + 0.02) for o in self.floor_blocking_objects()]
            self._floor_aabbs = aabbs
        hit = [n for n, a, b in aabbs if bool((p >= a).all() and (p <= b).all())]
        if hit:
            if why is not None:
                why[f"aabb:{hit[0]}"] = why.get(f"aabb:{hit[0]}", 0) + 1
            return False
        return True

    def drop_pose_at(self, xy, quat) -> G.Pose:
        """Release pose of the push-under drop: the target's bottom ``PUSH_UNDER_DROP_HEIGHT_M`` above the floor at ``xy``."""
        z = self.floor_z + self.world.bottom_offset(self.hull, quat) + float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M))
        return th.tensor([float(xy[0]), float(xy[1]), z]), quat

    #: height of the blade above the fingertips that can meet the pushed object (the finger pads, PAD_Z_RANGE span)
    BLADE_SPAN_M = 0.05

    def contact_offset_along(self, quat, direction_xy, z_lo: float, z_hi: float) -> Optional[float]:
        """Distance from the target's root back (against the push direction) to the rearmost collision point the blade can
        meet: the collision boundary points at orientation ``quat`` whose height above the target's bottom lies in
        ``[z_lo, z_hi]`` (the blade's band: fingertips to fingertips + ``BLADE_SPAN_M``). A sandal pushed along its length
        with the fingertips 3.6 cm up is met at its STRAP, ~3 cm behind the root, not at the heel 12 cm behind: modelled
        with the footprint's half extent the blade stopped 9-10 cm short of the goal cell (0920-143722 / 154503: the
        sandal's root ended 3 cm ahead of the blade). ``None`` when nothing of the target reaches the band."""
        pts = G.local_points_world(self.hull, (th.zeros(3), G.t(quat)))
        z = pts[:, 2] - pts[:, 2].min()
        sel = (z >= float(z_lo)) & (z <= float(z_hi))
        if not bool(sel.any()):
            return None
        d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])])
        d = d / th.linalg.norm(d)
        proj = pts[sel][:, :2] @ d
        return float(-proj.min())

    def reach_margin(self, stance: Stance, point_world, arm: Optional[str] = None) -> float:
        """W2 (owner 2026-09-20 22:15): ``arm``'s (default: the segment arm) stretched length minus the shoulder -> ``point_world``
        distance at the stance's trunk (shoulder through the shared torso FK, ``conventions.shoulder_position``; the stretched length
        once per arm through the kit's FK with the arm joints at zero). Positive = inside the envelope; the trunk that bows the
        tallest torso forward (154503 / 154504: B at (0.0, -0.61), the grasp 0.88 m out, 250-365 steps) scores lowest. The shoulder
        height is measured from the floor object's top (``self.floor_z``; ``tabletop.reach_margin`` takes the kit's measured floor)."""
        arm = arm or self.seg_arm
        attr = f"_arm_len_{arm}"
        if not hasattr(self, attr):
            from ispatialgym.oracle import r1pro_model as M
            q = self.kit.q_config(arms="untucked", translate=0.5, tilt=0.0)
            q[self.kit.robot.arm_control_idx[arm]] = 0.0
            fk = self.kit.fk(["torso_link4", M.EEF_LINK[arm]], q)
            sh = G.compose(fk["torso_link4"], (th.as_tensor(C._SHOULDER_IN_TORSO4[arm], dtype=th.float32), G.IDENTITY_QUAT))[0]
            setattr(self, attr, float(th.linalg.norm(fk[M.EEF_LINK[arm]][0] - sh)))
        # the URDF-clamped torso (a tilt past the torso_joint3 limit, e.g. (0.95, -1.2), is what the robot actually reaches)
        sh_b = C.torso_fk(C.torso_q(float(stance.translate), float(stance.tilt), C.TORSO_JOINT_LIMITS))[f"{arm}_shoulder"][0]
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        c, s_ = math.cos(yaw), math.sin(yaw)
        sh_w = th.tensor([x + c * float(sh_b[0]) - s_ * float(sh_b[1]), y + s_ * float(sh_b[0]) + c * float(sh_b[1]), self.floor_z + C.NOMINAL_BASE_Z + float(sh_b[2])])
        p = th.as_tensor([float(v) for v in point_world], dtype=th.float32)
        return getattr(self, attr) - float(th.linalg.norm(p - sh_w))

    def best_reach_margin(self, stance: Stance, point_world, arms: Sequence[str]) -> Tuple[float, str]:
        """The largest :meth:`reach_margin` over ``arms`` and its arm, the first arm on a tie (``tabletop.best_reach_margin``'s rule):
        a stance B ranked by whichever arm reaches better (owner 2026-09-27, first-demo v2.2)."""
        best: Optional[Tuple[float, str]] = None
        for a in arms:
            m = self.reach_margin(stance, point_world, a)
            if best is None or m > best[0] + 1e-9:
                best = (m, a)
        return best if best is not None else (self.reach_margin(stance, point_world), self.seg_arm)

    def half_extent_along(self, quat, direction_xy) -> float:
        """Half extent of the target's footprint along a horizontal direction at orientation ``quat``."""
        pts = G.local_points_world(self.hull, (th.zeros(3), quat))[:, :2]
        d = th.as_tensor([float(direction_xy[0]), float(direction_xy[1])])
        d = d / th.linalg.norm(d)
        proj = pts @ d
        return float(proj.max() - proj.min()) / 2

    #: below this fingertip height above the floor the cuRobo ground slab flags the padded fingertip spheres (first-demo loop
    #: 2026-09-16: every push pre-pose failed the IK at z_push = 4.5 cm) - the slab is switched off for such a chain's IK
    SLAB_TIP_M = 0.045

    def push_height(self, pitch_deg: float) -> Tuple[float, float]:
        """``(z_push, z_tip)`` of the floor blade: the fingertips at ``push_tip_frac`` (½) of the target's height above the
        floor, at least ``push_tip_clearance_m`` (1.5 cm) and at most 5 cm up; the eef origin ``FINGERTIP_LENGTH·cos(pitch)``
        above the tips. The old rule (eef origin at ¾ of the height, fingertips ≥ 4 cm up for the cuRobo ground slab) ran
        the blade over a 4.5 cm sandal: it caught the strap near the middle and moved the sandal 17 cm of the 29 cm
        (0920-143722, 12 cm short of the goal cell). A blade below ``SLAB_TIP_M`` has its IK solved with the slab off."""
        # the fingertips at half the height (3.6 cm for the 7.2 cm sandal chgcrm: the height the exact IK admits and 0920-154503
        # executed; 1.1 cm had no IK solution at all, 0920-170805) - what the blade meets at that height is modelled per yaw by
        # ``contact_offset_along`` (pod probe 2026-09-20: the sandal's heel face reaches 12.8 cm back only up to z 3.5 cm, its toe
        # 11.4 cm forward up to z 4.5 cm, the strap 2.7 cm behind the root above - so the blade at 3.6 cm pushes it toe-first)
        height = float(self.place["height"])
        frac = float(self.num("push_tip_frac", 0.5))
        clear = float(self.num("push_tip_clearance_m", 0.015))
        z_tip = self.floor_z + max(clear, min(frac * height, 0.05))
        drop = C.FINGERTIP_LENGTH * math.cos(math.radians(float(pitch_deg)))
        return z_tip + drop, z_tip

    #: the order of the carried yaws the push chain is tried at (named by the chain-first screen's error)
    PUSH_YAW_ORDER = "largest-offset"

    def push_yaw_rank(self, stance: Stance, quat, off: float) -> float:
        """Rank of a carried yaw for the push chain from ``stance`` (smaller first; ``quat`` = the target's orientation at C,
        ``off`` = the blade's contact offset there): the push-under's LARGEST offset first (the blade goal nearest the base,
        2026-09-20 19:20). Orders the chain-first screen's yaws and the F draws."""
        return -float(off)

    def push_drop_ahead_pref(self) -> Optional[float]:
        """The drop cell's preferred distance ahead of the base in the C order (``generation.push_under_drop_ahead_pref_m``)."""
        v = self.gp.gen("push_under_drop_ahead_pref_m")
        return None if v is None else float(v)

    def _push_under_chain(self, place: Stance, quat, drop_xy, goal_xy, d2) -> Optional[Dict[str, Any]]:
        """The straight floor-level push from the drop position to the goal cell, exact IK from stance C (closed gripper,
        floor and target ignored, partner present): blade behind the target along the push direction (stand-off from
        ``tabletop.blade_standoff`` over the target's face height), approach pitched toward the push direction (0 / 30 /
        45 / 60°, either wrist turn), poses ``[pre, start, mid, goal]``, with the segment arm (``programs.py``: the push-under blade
        is the grasp arm). Returns the record fields or None."""
        floor, target = self.obj("floor"), self.target
        arm = self.seg_arm
        d2 = th.as_tensor([float(d2[0]), float(d2[1])])
        d2 = d2 / th.linalg.norm(d2)
        d3 = th.tensor([float(d2[0]), float(d2[1]), 0.0])
        height = float(self.place["height"])
        cd = th.as_tensor([float(drop_xy[0]), float(drop_xy[1])])
        gc = th.as_tensor([float(goal_xy[0]), float(goal_xy[1])])
        distance = float(th.linalg.norm(gc - cd))
        why: List[str] = []
        half_d: Optional[float] = self.half_extent_along(quat, d2)  # the footprint's, for the log when no pitch gets a blade
        for pitch in TT.approach_pitch_order(None):
            z_push, z_tip = self.push_height(pitch)
            so = TT.blade_standoff((self.floor_z, self.floor_z + height), z_tip, pitch, PUSH_UNDER_CLEARANCE_M)
            if so is None:
                why.append(f"pitch {pitch:.0f}: face reaches the palm")
                continue
            standoff = so[0]
            # the blade meets the rearmost collision point within its height band (a sandal's strap, a box's face), not the
            # footprint's rear edge; the (generation) ``push_contact_offset_m`` overrides the model
            off_cfg = self.gp.gen("push_contact_offset_m")
            off_m = float(off_cfg) if off_cfg is not None else self.contact_offset_along(quat, d2, z_tip - self.floor_z, z_tip - self.floor_z + self.BLADE_SPAN_M)
            if off_m is None:
                why.append(f"pitch {pitch:.0f}: nothing of the target within the blade's height band")
                continue
            half_d = off_m
            a_vec = TT.pitched_approach(pitch, d2)  # pitched toward the push direction (= the robot's forward for a frontal stance, its arm's side for a side stance)
            start = th.tensor([float(cd[0]), float(cd[1]), z_push]) - d3 * (half_d + standoff)
            # the blade's goal runs ``push_goal_extra_m`` (3 cm) past the model's: the oracle's differential-IK leg stops 2-3 cm
            # short of its last waypoint (0920-143722 / 154503: eef 0.704-0.716 m ahead for a 0.736 m goal)
            goal = th.tensor([float(gc[0]), float(gc[1]), z_push]) - d3 * (half_d + standoff - float(self.num("push_goal_extra_m", 0.03)))
            mid = 0.5 * (start + goal)
            low = (z_tip - self.floor_z) < self.SLAB_TIP_M
            props = []
            for rank, c_sign in enumerate((1.0, -1.0)):
                q = C.eef_quat(a_vec, c_sign * d3)
                poses = [(start, q), (mid, q), (goal, q)]
                props.append(O.Proposal(f"pitch {pitch:.0f}{'' if c_sign > 0 else '-'}", rank, c_sign, [C.pre_pose(poses[0])] + poses))

            def evaluate(p, low=low):
                with self.ikc.ignoring([floor, target], ground=not low):
                    res = self.ikc.exact(p.chain, place, arm, cache=False, fingers=C.FINGER_CLOSED, q_first=p.q_first)
                return bool(res.ok), res, None, str(res.reason)

            # the two wrist turns by the orientation rule (owner 2026-09-27); the camera is checked at the blade's start, where the
            # target and the floor left the IK's world
            chosen, decision = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(place), camera_idx=[1], stage=f"push-under pitch {pitch:.0f}")
            if chosen is not None:
                pre, start_p, mid_p, goal_p = chosen.proposal.chain
                return {"pre": pre, "start": start_p, "mid": mid_p, "goal": goal_p, "direction": d3, "distance_m": distance, "pitch_deg": float(pitch),
                        "blade_standoff_m": float(standoff), "push_height_m": float(z_push - self.floor_z), "half_extent_m": half_d, "wrist_sign": chosen.proposal.payload,
                        "fingertip_z_m": float(z_tip - self.floor_z), "ground_ignored": bool(low), "orient": decision}
            why.extend(f"{o['label']}: {o.get('why', '')}" for o in decision["options"] if not o.get("ok"))
        log.info("%s: push-under chain unreachable from stance C (d %.2f λ %+.2f trunk %s; start %s, goal %s in base): %s", self.gp.alias, place.stand_off or 0.0,
                 place.lateral, (place.translate, round(place.tilt, 2)),
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(cd[0]), float(cd[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 [round(float(v), 2) for v in P0.to_base_frame(th.tensor([float(gc[0]), float(gc[1]), 0.0]) - d3 * half_d, place.base_pose2d, P0.base_z(self)).tolist()],
                 "; ".join(why))
        return None

    def clearance_at(self, xy) -> Optional[float]:
        cm = self.place.get("clearance")
        if not cm:
            return None
        res, (ox, oy) = float(cm["resolution"]), (float(cm["origin"][0]), float(cm["origin"][1]))
        i, j = int((float(xy[0]) - ox) / res), int((float(xy[1]) - oy) / res)
        g = cm["grid"]
        if 0 <= i < len(g) and 0 <= j < len(g[i]) and g[i][j] is not None:
            return float(g[i][j])
        return None

    # ------------------------------------------------------------------ reach bands (B3)

    def make_band_spec(self) -> TT.BandSpec:
        c = self.choice.models["target"]
        return TT.BandSpec(object=f"{c.category}/{c.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull, rest_quat=self.rest,
                           scale=list(c.scale) if c.scale else None, height=self.target_dims()[1])

    def band(self, kind: str, support_obj=None, under_partner=None, arm: Optional[str] = None) -> Optional[RB.ReachBands]:
        """The reach band of a factor set (§5.6): tabletop mode (init support / the partner's top) through
        ``tabletop.ensure_bands`` / ``compute_bands`` (``ReachBands.load`` on a hit, ``BandComputer`` on a miss), the
        floor modes (``--support floor [--under partner]``) through :meth:`compute_band` on a miss; memoised per
        (kind, support, under) for the template. ``ISG_REACH_BANDS_COMPUTE=0`` turns a miss into ``Reject("missing_band")``.
        ``arm`` (owner 2026-09-27, first-demo v2.2): that arm's band - the band above when it is that arm's, else
        :meth:`other_arm_band`."""
        c = self.choice.models["target"]
        obj = f"{c.category}/{c.model}"
        support = "floor" if kind == "floor" else support_obj.name
        under = under_partner.name if under_partner is not None else None
        memo = self.__dict__.setdefault("_bands", {})
        if arm is not None:
            base = self.band(kind, support_obj, under_partner)
            if base is None or base.arm == arm:
                return base
            key = (kind, support, under, arm)
            if key not in memo:
                memo[key] = self.other_arm_band(base, kind, support, under, arm)
            return memo[key]
        key = (kind, support, under)
        if key in memo:
            return memo[key]
        # the init support's interior surface (tabletop.support_surface) has its own band; the partner's top never does
        surface = TT.surface_band_kwargs(self) if (kind == "tabletop" and support_obj is self.surface.get("obj")) else {}
        try:
            rb = RB.ReachBands.load(self.world.scene_model, support, obj, self.grasp_key, under=under, scale=c.scale, **surface)
            log.info("%s: reach band %s (%s) from %s", self.gp.alias, support, under or "-", rb.path)
            # a band computed before the A6 posture retry can be empty (every trunk stance collided with the untucked
            # arms: the fleet's floor-under bands of the beds) — an empty band is a miss, recomputed with the current script
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0 and os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") != "0":
                log.info("%s: cached reach band %s (%s) is empty (computed with %s arms, no A6 retry): recomputing in-process", self.gp.alias, support, under or "-",
                         rb.meta.get("arms"))
                raise FileNotFoundError("empty band")
        except FileNotFoundError:
            if os.environ.get("ISG_REACH_BANDS_COMPUTE", "1") == "0" or surface:
                raise Reject("missing_band", f"no reach band for scene={self.world.scene_model} support={support} object={obj} under={under} scale={c.scale}"
                                             + (f" surface_z={surface['surface_z']:.2f}" if surface else ""))
            if kind == "tabletop":
                top = self.surface.get("top") if (self.surface.get("obj") is support_obj and self.surface.get("top") is not None) else self.support_top(support_obj, ignore=[self.target])
                rb = TT.compute_bands(self, self.make_band_spec(), top)
            else:
                rb = self.compute_band(kind, support_obj, under_partner)
        if rb is not None and int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
            # still empty after the recompute (the band stance itself collides at every trunk — e.g. the band script's
            # under stance at a blocked AABB edge of the partner): the stance search runs without the band filter and the
            # exact IK of the coupling decides (recorded in the template facts; the band owner gets the case)
            log.warning("%s: reach band %s (%s) is empty after the recompute: stance C candidates are taken from the goal cells without the band filter",
                        self.gp.alias, support, under or "-")
            self.template_facts.setdefault("empty_bands", []).append({"support": support, "under": under, "stance": rb.meta.get("stance")})
            rb = None
        memo[key] = rb
        return rb

    def other_arm_band(self, base: RB.ReachBands, kind: str, support: str, under: Optional[str], arm: str) -> Optional[RB.ReachBands]:
        """``arm``'s band of the factors of ``base`` (the other arm's band; owner 2026-09-27, first-demo v2.2): its own cached band
        (``ReachBands.load(..., arm=arm)``; an empty one is a miss), else ``base`` mirrored in memory (``ReachBands.mirrored``: tabletop
        and floor bands only). Never written, never computed in-process: :meth:`compute_band` / ``compute_bands`` write files whose
        names carry no arm. None when neither exists; recorded in ``template_facts["arm_bands_by_factor"]``."""
        c = self.choice.models["target"]
        try:
            rb = RB.ReachBands.load(self.world.scene_model, support, f"{c.category}/{c.model}", self.grasp_key, under=under, scale=c.scale, arm=arm)
            if int(rb.meta.get("n_reachable_total", 1) or 0) == 0:
                raise FileNotFoundError("empty band")
            log.info("%s: %s-arm reach band %s (%s) from %s", self.gp.alias, arm, support, under or "-", rb.path)
        except FileNotFoundError:
            rb = base.mirrored() if RB.mirrorable(base.meta) else None
            log.info("%s: %s-arm reach band %s (%s): %s", self.gp.alias, arm, support, under or "-",
                     f"the {base.arm} arm's band mirrored ({base.path})" if rb is not None else f"none (the {base.arm} arm's band is not mirrored)")
        self.template_facts.setdefault("arm_bands_by_factor", {}).setdefault(f"{kind}:{support}{'-under-' + under if under else ''}", {})[arm] = TT.band_facts(rb)
        return rb

    def compute_band(self, kind: str, support_obj=None, under_partner=None) -> RB.ReachBands:
        """Compute-on-miss of the floor-mode bands (``--support floor [--under partner]``) with the computation the CLI
        runs (the ``stance_free_floor`` / ``stance_under`` stance, the 2 cm floor lattice (under: the clear cells), the
        library grasp mask per trunk incl. tilt −1.2); written to the cache root in the CLI's layout. Computed on
        ``reach_bands.band_kit`` (``self.kit`` is not touched), recorded in ``meta.schedule``. Past the sampling run's time budget
        (``self.deadline``) the computation stops, nothing is written: ``Reject("band_budget")``. ``ISPATIALGYM_BAND_TRUNK_CHECK=idle-arm``
        (owner 2026-10-12 item 4: the whole-body check skipped every deep crouch, so no floor pose was within reach of the
        remaining shoulders): the CLI's ``--trunk-check idle-arm``, keyed, named and recorded like the CLI's."""
        c = self.choice.models["target"]
        kit = RB.band_kit(self.world, self.kit)
        bc = RB.BandComputer(self.world, kit, arms=("right",))
        bc.deadline = getattr(self, "deadline", None)
        tc_key = RB.band_trunk_check(bc, "right")
        t0 = time.time()
        parked = self.world.snapshot()
        try:
            self.world.park(self.target, list(self.world.spawned).index(self.target.name))
            kit.sync_world()
            if kind == "floor":
                z = self.floor_z
                if under_partner is not None:
                    stance = bc.stance_under(under_partner, RB.STAND_OFF)
                    lattice_world = self.place["cells"] if self.relation == "under" and len(self.place.get("cells", [])) else self._under_raycast(under_partner, 0.0, 0.02)
                    ignore = [self.obj("floor")]
                else:
                    room = (self.gp.rooms(self.scene) or self.world.rooms or [None])[0]
                    stance = bc.stance_free_floor(room)
                    origin, fwd, left = bc.base_frame(stance)
                    xs = th.arange(0.2, RB.STRIP_MAX_DEPTH + 0.5, 0.02)
                    ls = th.arange(-RB.STRIP_HALF_WIDTH, RB.STRIP_HALF_WIDTH + 1e-6, 0.02)
                    gx, gl = th.meshgrid(xs, ls, indexing="ij")
                    lattice_world = origin + gx.flatten().unsqueeze(1) * fwd + gl.flatten().unsqueeze(1) * left
                    lattice_world = th.stack([cell for cell in lattice_world if self.kit.trav_free(cell)]) if len(lattice_world) else lattice_world
                    ignore = [self.obj("floor")]
                top_json = {"z": z, "kind": "floor"}
            else:
                raise ValueError("compute_band handles the floor modes; tabletop bands go through tabletop.compute_bands")
            hull, rest = self.hull, self.rest
            max_off = max(float(th.linalg.norm(g.position)) for g in self.grasps) if self.grasps else 0.0

            def pose_fn(world_xy, yaw, _h=hull, _r=rest, _z=z):
                return bc.object_pose_at(world_xy, yaw, _z, _h, _r)

            trunks = RB.trunk_grid(z - self.floor_z)
            result = bc.compute(stance, lattice_world, z, pose_fn, self.grasps, ignore, arm="right", trunks=trunks, max_grasp_offset=max_off)
        except RB.BandDeadline as e:
            raise Reject("band_budget", f"{self.gp.alias}: in-process {kind} band ({under_partner.name if under_partner is not None else '-'}) not finished: {e} "
                                        f"({time.time() - t0:.0f} s in); nothing written - a band job publishes it") from e
        finally:
            self.world.restore(parked)
            kit.sync_world()
        meta = {"scene": self.world.scene_model, "rooms": list(self.world.rooms or []), "support": "floor" if kind == "floor" else support_obj.name, "lattice": bc.lattice,
                "stand_off": RB.STAND_OFF, "arms": "untucked", "grid": RB.grid_meta((z - self.floor_z) < RB.LOW_TOP_Z), "seeds": 12,
                "pre_grasp": RB.PRE_GRASP, "yaws": RB.YAWS, "mode": "floor" if kind == "floor" else "tabletop", "scale": c.scale, "placed_specs": [],
                "object": f"{c.category}/{c.model}", "grasp_key": self.grasp_key, "n_grasps": len(self.grasps), "grasp_names": [g.name for g in self.grasps],
                "rest_quat_xyzw": self.rest.tolist(), "generic_pinch": False, "stance": stance, "top": top_json, "z_support": z,
                "under_partner": under_partner.name if under_partner is not None else None, "trunks": result["trunks"], "computed_by": "iors_large compute-on-miss",
                "time_s": round(time.time() - t0, 1), "schedule": RB.band_schedule(kit, self.kit)}
        if tc_key:
            meta["trunk_check"] = RB.trunk_check_meta("right")
        lat_all = np.concatenate([cc[:, 1] for cc in result["cells"].values() if len(cc)]) if any(len(cc) for cc in result["cells"].values()) else np.zeros(0)
        meta["union_extent"] = [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None
        meta["n_reachable_total"] = int(sum(len(cc) for cc in result["cells"].values()))
        cells = {k: np.asarray(v, dtype=np.float32).reshape(-1, 2) for k, v in result["cells"].items()}
        pairs = {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in result.get("pairs", {}).items()}
        rb = RB.ReachBands(meta, cells, pairs, None)
        try:
            out_dir = RB.cache_root()
            os.makedirs(out_dir, exist_ok=True)
            key = RB._key({"scene": meta["scene"], "support": meta["support"], "z": round(z, 2), "object": meta["object"], "scale": c.scale, "under": meta["under_partner"],
                           "grasp_key": self.grasp_key, "computed_by": "iors_large", **tc_key})
            name = (f"{meta['mode']}__{meta['scene']}__{meta['support']}{RB.TRUNK_CHECK_IDLE_TAG if tc_key else ''}"
                    f"{'-under-' + meta['under_partner'] if meta['under_partner'] else ''}__{c.category}-{c.model}__{key}")
            arrays = {RB.trunk_key(tr, ti): cc for (tr, ti), cc in cells.items()}
            for (tr, ti), p in pairs.items():
                arrays[RB.trunk_key(tr, ti) + "__pairs"] = p
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
            with open(os.path.join(out_dir, name + ".json"), "w") as f:
                json.dump(meta, f, indent=1, default=str)
            rb.path = os.path.join(out_dir, name + ".npz")
        except Exception as e:  # noqa: BLE001 - the band is still usable in memory
            log.warning("%s: could not write the computed reach band: %r", self.gp.alias, e)
        log.info("%s: reach band computed on miss (%s, %d cells, %.0fs)", self.gp.alias, meta["support"], meta["n_reachable_total"], meta["time_s"])
        return rb

    def band_cells_world(self, rb: Optional[RB.ReachBands], stance: Stance) -> th.Tensor:
        """The band of the stance's grid trunk as world xy at this concrete stance (the band is base-relative)."""
        if rb is None:
            return th.zeros(0, 2)
        cells = rb.band(stance.translate, stance.tilt, **TT.band_kw(rb, stance))
        if len(cells) == 0:
            return th.zeros(0, 2)
        return th.as_tensor(rb.to_world(cells, stance.base_pose2d), dtype=th.float32)

    # ------------------------------------------------------------------ stances A / B per surface candidate

    def try_surface(self, kind: str, cand, rng: np.random.Generator):
        """Build the template facts of stances A and B on one init-surface candidate, and the C survivors' existence."""
        self.surface = {"kind": kind, "obj": None if kind == "floor" else cand}
        if kind == "floor":
            self.try_floor_surface(rng)
        else:
            self.try_support_surface(cand, rng)
        # stance C must exist for the init pose class (checked on the surface's representative init xy) for one of the offered arms
        c_survivors = self.placement_candidates(self.surface["init_xy"])
        for arm in TT.offered_arms(self)[1:]:
            if c_survivors:
                break
            c_survivors = self.placement_candidates(self.surface["init_xy"], arm=arm)
        if not c_survivors:
            raise Reject("no_placement_stance", f"no stance C at {self.partner.name} for an init pose at {self.surface['init_xy'].tolist()}")
        self.surface["n_c"] = len(c_survivors)
        self.template_facts["stances"] = {"init": self.surface["init"].to_meta(), "n_manipulation_candidates": len(self.surface["manip"]), "n_placement_candidates": len(c_survivors)}

    def try_support_surface(self, support, rng: np.random.Generator):
        """Stances A and B on a support (tabletop Phases 1a / 1b / 1c / 2 of ``tabletop.py``): per edge (random order)
        the init trunk, the lateral draw redrawn (≤ 6) until the partner is visible (≥ 4 / 8 corners + centre), the
        Phase-2 survivors with the band of (support, target) - with every offered arm's band (owner 2026-09-27, first-demo v2.2:
        ``tabletop.arm_bands_for``, the other arm's a mirror when it has none; survivors tagged ``focus_arms``)."""
        self.kit.sync_world()
        offered = TT.offered_arms(self)
        half, height = self.target_dims()
        top = TT.support_surface(self, support, ignore=[self.target], height=height)
        if int(top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        self.surface["top"] = top
        self.bands = None  # ``TT.ensure_bands`` caches one band per support on the sampler
        margins = {"w_t": half, "w_p": 0.0, "W_p": 0.0, "D_p_max": 0.0, "safety": 0.02, "relation": self.relation, "g_NT": None, "H_max": height}
        # A5 "partner's own top": the partner is the support under the target, so the partner-visibility redraw does not
        # apply (the target cell in frame is on the partner); stances A / B come from the tabletop phases unchanged.
        # Owner decision 2026‑09‑16 (a)/(d): stance A sees the target (tabletop Phase 1), the init surface (the target's cell
        # on it) and the partner's RELEVANT REGION (ontop: the top-face patch around the goal cell + the nearest bottom
        # edge; under: the nearest bottom edge) — no "≥ 4 of 8 AABB corners + centre" any more; (b): stance B (the grasp
        # stance, Phase 2) has no partner rule — only its target-in-frame rule.
        own_top = support is self.partner
        errors: List[str] = []
        for edge, init in TT.edge_candidates(self, top, rng):
            try:
                init = TT.phase1b_init_trunk(self, init, height, top=top)
                chosen = None
                for _ in range(1 if own_top else 6):  # §5.6: redraw the lateral position (≤ 6) until the goal region is visible
                    st = TT.phase1c_lateral(self, init, margins, rng, top=top)
                    if own_top or self.goal_region_visible(st):
                        chosen = st
                        break
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e.reason}")
                    continue
                raise
            if chosen is None:
                errors.append(f"{edge.name}: goal_region_not_visible (the partner's top patch around the goal cell / nearest bottom edge out of frame)")
                continue
            chosen.name = "init"
            arm_bands = TT.arm_bands_for(self, offered, top=top, edge=chosen.edge) if len(offered) > 1 else None
            manip = TT.phase2_manipulation_stance(self, chosen, half, target_height=height, top=top, arm_bands=arm_bands)
            if not manip:
                errors.append(f"{edge.name}: no_manipulation_stance")
                continue
            self.surface.update({"edge": edge, "init": chosen, "manip": manip, "init_xy": manip[0].focus[:2].clone(), "z": top.z, "band": getattr(self, "bands", None),
                                 "arm_bands": dict(arm_bands or {offered[0]: getattr(self, "bands", None)})})
            return
        raise Reject("no_free_edge", f"{support.name}: " + ("; ".join(errors) or "no edge with a free, collision-free stance"))

    def goal_region_points(self, from_xy) -> th.Tensor:
        """Owner decision 2026‑09‑16 (a)/(d): the **relevant region** of the partner stance A must see — ontop: the top-face
        patch (± 0.10 m) around the goal cell the placement will use (the goal cell nearest to ``from_xy``, the rule C
        applies) at the top's height, plus the partner's bottom edge nearest to ``from_xy`` (its two bottom corners and
        the midpoint); under: the nearest bottom edge only (the goal lies beneath the partner)."""
        partner = self.partner
        lo, hi = partner.aabb
        f = th.as_tensor([float(from_xy[0]), float(from_xy[1])])
        pts = []
        if self.relation == "ontop" and len(self.place.get("cells", [])):
            cells = self.place["cells"]
            g = cells[int(th.argmin(th.linalg.norm(cells.float() - f, dim=1)))].float()
            z = float(self.place["z"])
            r = float(self.num("goal_patch_half_m", 0.10))
            for dx, dy in ((0, 0), (r, r), (r, -r), (-r, r), (-r, -r)):
                pts.append(th.tensor([float(g[0]) + dx, float(g[1]) + dy, z]))
        # nearest bottom edge of the AABB (as seen from ``from_xy``)
        edges = aabb_edges(lo, hi)
        edge = nearest_edge(edges, f)
        half_len = edge.length / 2
        for t in (-half_len, 0.0, half_len):
            p = edge.mid + edge.tangent * t
            pts.append(th.tensor([float(p[0]), float(p[1]), float(lo[2])]))
        return th.stack(pts)

    def goal_region_visible(self, stance: Stance, from_xy=None) -> bool:
        """The relevant partner region (:meth:`goal_region_points`) in frame with the 5 % margin from ``stance``
        (:meth:`partner_points_in_view`)."""
        cam = self.camera_pose_at(stance)
        if from_xy is None:
            from_xy = stance.focus[:2] if stance.focus is not None else th.as_tensor(stance.base_pose2d[:2])
        return self.partner_points_in_view(cam, self.goal_region_points(from_xy))

    def partner_points_in_view(self, cam, points) -> bool:
        """Partner points in frame with the 5 % margin (stance A's goal region, the place-under stance C's near-edge line): every point,
        or at least ``generation.partner_view_min_points`` of them (owner 2026-10-12 item 2a; :func:`enough_in_view`)."""
        k = self.gp.gen("partner_view_min_points")
        inside = self.world.camera_model.inside(cam, G.t(points).reshape(-1, 3), P0.IN_FRAME_MARGIN)
        return enough_in_view(inside, None if k is None else int(k))

    def partner_visible(self, stance: Stance, n_min: int) -> bool:
        cam = self.camera_pose_at(stance)
        lo, hi = self.partner.aabb
        corners = G.aabb_corners(lo.float(), hi.float())
        inside = self.world.camera_model.inside(cam, corners, 0.0)
        centre_in = bool(self.world.camera_model.inside(cam, ((lo + hi) / 2).float().reshape(1, 3), 0.0)[0])
        return int(inside.sum()) >= n_min and centre_in

    def try_floor_surface(self, rng: np.random.Generator):
        """§5.6 floor init: the anchor cell c_t, stance B facing it from {0.35, 0.45, 0.55} with the trunk from the floor
        band, stance A on the eroded map 0.6–1.5 m from c_t with the target and the partner in frame."""
        fi = dict(self.gp.gen("floor_init") or {})
        cell = float(fi.get("cell_m", 0.02))
        clearance = float(fi.get("clearance_m", 0.02))
        max_geo = float(fi.get("max_geodesic_m", self.num("max_partner_geodesic_m", 3.0)))
        floor = self.obj("floor")
        partner = self.partner
        lo, hi = floor.aabb
        plo, phi = partner.aabb
        rb = self.band("floor")
        self.surface["band"] = rb
        offered = TT.offered_arms(self)
        arm_bands = {offered[0]: rb, **{a: self.band("floor", arm=a) for a in offered[1:]}}
        self.surface["arm_bands"] = arm_bands
        for a in offered[1:]:
            self.template_facts.setdefault("arm_bands", {})[a] = TT.band_facts(arm_bands[a])
        half, height = self.target_dims()
        objs = self.floor_blocking_objects()
        aabbs = [(o.aabb[0][:2].float() - clearance, o.aabb[1][:2].float() + clearance) for o in objs]
        # candidate anchors: free floor cells within max_geo of the partner's near side, outside every AABB and the partner footprint
        xs = th.arange(float(lo[0]) + cell / 2, float(hi[0]), cell * 4)  # 8 cm search grid (the 2 cm raster is the draw's resolution)
        ys = th.arange(float(lo[1]) + cell / 2, float(hi[1]), cell * 4)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        near_c = th.tensor([float(plo[0] + phi[0]) / 2, float(plo[1] + phi[1]) / 2])
        pts = pts[th.linalg.norm(pts - near_c, dim=1) <= max_geo + 1.5]
        anchors = []
        stats = {"grid": int(len(pts)), "trav_free": 0, "in_rooms": 0, "outside_partner": 0, "outside_objects": 0}
        blockers: Dict[str, int] = {}
        for p in pts:
            if not self.kit.trav_free(p):
                continue
            stats["trav_free"] += 1
            if not P0.in_task_rooms(self, p):
                continue
            stats["in_rooms"] += 1
            if bool((p >= plo[:2].float() - half).all() and (p <= phi[:2].float() + half).all()):
                continue  # inside the partner footprint (the goal must be false at init)
            stats["outside_partner"] += 1
            hit = [o.name for o, (a, b) in zip(objs, aabbs) if bool((p >= a).all() and (p <= b).all())]
            if hit:
                blockers[hit[0]] = blockers.get(hit[0], 0) + 1
                continue
            stats["outside_objects"] += 1
            anchors.append(p)
        self.template_facts["floor_cells"] = {**stats, "blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:6]), "floor": floor.name,
                                              "floor_aabb": [lo[:2].tolist(), hi[:2].tolist()]}
        if not anchors:
            raise Reject("no_floor_cell", f"no free floor cell beside the partner ({stats}; blocking AABBs {dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:4])}; "
                                          f"floor {floor.name} aabb {[round(float(v), 2) for v in lo[:2].tolist()]}..{[round(float(v), 2) for v in hi[:2].tolist()]})")
        anchors = th.stack(anchors)
        order = rng.permutation(len(anchors))
        errors = []
        for i in order[:40]:
            c_t = anchors[int(i)]
            try:
                _, dist = self.kit.geodesic(c_t, self.near_side_point(partner, c_t))
            except Exception:  # noqa: BLE001
                dist = None
            if dist is None or dist > max_geo:
                continue
            try:
                manip = self.floor_stance_b(c_t, rb, half, rng, arm_bands=arm_bands)
                init = self.floor_stance_a(c_t, rng, height, like=manip[0])
                with self.ikc.ignoring([self.obj("floor")]):
                    self.record_leg_a_b(init, manip[0])  # §5.6: a path A -> B exists (navigate-then-torso, else torso-then-navigate)
                self.sidecar.transitions.clear()
            except Reject as e:
                errors.append(f"{e.reason}: {e}")
                continue
            self.sidecar.paths.clear()
            self.surface.update({"c_t": c_t, "init": init, "manip": manip, "init_xy": c_t.clone(), "z": self.floor_z, "edge": None})
            self.template_facts["floor_init"] = {"c_t": c_t.tolist(), "geodesic_m": dist, "n_anchor_candidates": int(len(anchors))}
            return
        raise Reject("no_floor_cell", f"no anchor cell admits stances A / B ({len(errors)} tried): " + "; ".join(sorted(set(errors))[-4:]))

    def floor_stance_b(self, c_t: th.Tensor, rb: Optional[RB.ReachBands], half: float, rng: np.random.Generator,
                       arm_bands: Optional[Dict[str, Optional[RB.ReachBands]]] = None) -> List[Stance]:
        """Stance B candidates of the floor init: base facing c_t at {0.35, 0.45, 0.55} (bearing uniform), trunk from the
        floor band (c_t must be a band cell), collision-free, target in frame, axis within 8° (else 20°) of c_t.
        ``arm_bands`` (owner 2026-09-27, first-demo v2.2: ``{arm: floor band}`` of the offered arms, the default arm's being ``rb``):
        the other arm's candidates turn the base the mirror way (:func:`mirrored_offsets`: c_t to the robot's left for the left
        arm) and hold c_t in that arm's band; every candidate records the arms whose band holds c_t at its trunk in
        ``extras["focus_arms"]`` and works with the first of them."""
        arm0 = TT.default_arm(self)
        others = {a: b for a, b in (arm_bands or {}).items() if a != arm0 and b is not None} if rb is not None else {}
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in list(RB.TILTS) + [RB.LOW_TOP_TILT]]
        for b in others.values():
            trunks += [k for k in b.trunks() if k not in trunks]
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + 0.03])
        cands = []
        # the floor band lies to the robot's RIGHT of the centreline (B3 under-edge probe: lateral −0.40…−0.16; the floor
        # bands' cells at translate 0.95 likewise): the base is turned so that c_t sits at lateral ``y_off`` (0 = straight
        # ahead, negative = right) — a stance facing c_t squarely puts it between the arms where no floor cell is reachable
        # (countertop-floor 18:50: 41 band cells, "c_t is in no band cell of any stance" for 29 anchors); the left arm's
        # (mirrored) band lies to the robot's left
        offsets = [float(v) for v in (self.gp.gen("floor_init_lateral_offsets_m") or (-0.25, -0.15, -0.35, 0.0))]
        lanes = list(offsets)
        for a in others:
            lanes += [v for v in mirrored_offsets(offsets, a) if all(abs(v - u) > 1e-9 for u in lanes)]
        bands = {arm0: rb, **others}

        def holds(b: RB.ReachBands, st: Stance) -> bool:
            cells = self.band_cells_world(b, st)
            return len(cells) > 0 and float(th.linalg.norm(cells - c_t, dim=1).min()) <= 0.51 * float(b.meta.get("lattice", RB.LATTICE))

        for _ in range(12):
            bearing = float(rng.uniform(-math.pi, math.pi))
            for dist in STAND_OFFS:
                xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
                if not P0.base_ok(self, xy):
                    continue
                yaw0 = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
                for y_off in lanes:
                    if abs(y_off) >= dist:
                        continue
                    yaw = offset_yaw(yaw0, y_off, dist)  # c_t at the base-frame lateral y_off (negative: the robot's right)
                    for tr, ti in trunks:
                        st = Stance("manipulation", (float(xy[0]), float(xy[1]), yaw), tr, ti, dist, None, None, 0.0, focus.clone(), [arm0])
                        st.extras["lateral_offset_m"] = y_off
                        if rb is not None:
                            focus_arms = [a for a, b in bands.items() if holds(b, st)]
                            if not focus_arms:
                                continue
                            if others:
                                st.arms = [focus_arms[0]]
                                st.extras["focus_arms"] = focus_arms
                        cands.append(st)
        if not cands:
            raise Reject("no_manipulation_stance", "floor init: c_t is in no band cell of any stance")
        # the floor band has cells at the deep trunks only (translate 0.95): like the push-under stance C, the floor mesh
        # is removed from the stance check (the cuRobo ground slab stays — B3's "torso / head within 1 cm of the floor
        # mesh" finding) and the optical-axis rule is 45° (the crouched head hangs over the cell)
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        n_cands, trunks_seen = len(cands), sorted({(float(st.translate), round(float(st.tilt), 2)) for st in cands})
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        if not cands:
            raise Reject("no_manipulation_stance", f"floor init: every stance B collides ({n_cands} candidates at trunks {trunks_seen})")
        pts = G.local_points_world(self.hull, (focus, self.rest))
        classed = []
        n_frame, n_axis = 0, 0
        axis_max = math.radians(float(self.num("floor_init_axis_max_deg", 45.0)))
        for st, cam in zip(cands, P0.camera_poses(self, cands)):
            if not P0.in_frame(self, cam, pts):
                n_frame += 1
                continue
            ang = P0.axis_angle(self, cam, focus)
            if ang > axis_max:
                n_axis += 1
                continue
            st.extras["angle_class"] = 1 if ang > math.radians(P0.AXIS_TIGHT_DEG) else 0
            st.extras["axis_angle_deg"] = math.degrees(ang)
            classed.append((st, st.extras["angle_class"]))
        out = P0.best_class(classed)
        if not out:
            raise Reject("no_manipulation_stance", f"floor init: no stance B sees c_t ({len(cands)} collision-free; {n_frame} target out of frame, {n_axis} axis > {math.degrees(axis_max):.0f}°)")
        return out

    def floor_stance_a(self, c_t: th.Tensor, rng: np.random.Generator, height: float, like: Optional[Stance] = None) -> Stance:
        """Stance A of the floor init: base on the eroded map 0.6–1.5 m from c_t facing it, with the target and the
        partner's relevant region (goal patch / nearest bottom edge, owner decision (a)/(d)) in frame, collision-free.
        Trunk: **torso straight — tilt 0 at any translate** (owner 2026‑09‑19 21:56 (i): "it can have the lowest height
        but no tilt"), the **most upright** translate among those that see the target and the partner region, the
        optical-axis angle to c_t breaking ties (owner 2026-09-24). Until then the smallest optical-axis angle won: with
        the tucked spawn of ``1dd3820b`` the hanging hands of a crouched A touched the floor and ``body_free`` dropped it,
        so A came out upright by accident (0921-045713: 0.25); with the untucked spawn the 0.95 crouch passes and looks
        down at a floor target best, and it then swept the target on the direct A -> B drive at every draw and became the
        carry's second travel trunk (0924-225021: 8/8 draws at 0.95, no instance). (``729a1ec`` had A take stance B's
        crouch so that A -> B needed no sweep; that is now the *fallback* of the A -> B leg — a departure transition at
        A — not the init posture. ``like`` is kept for the signature; ``trunk_like_b`` is recorded for the log.)"""
        fi = dict(self.gp.gen("floor_init") or {})
        d_lo, d_hi = fi.get("distance_range_m", [0.6, 1.5])
        focus = th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + height / 2])
        pts = G.local_points_world(self.hull, (th.tensor([float(c_t[0]), float(c_t[1]), self.floor_z + self.world.bottom_offset(self.hull, self.rest)]), self.rest))
        for _ in range(30):
            bearing = float(rng.uniform(-math.pi, math.pi))
            dist = float(rng.uniform(float(d_lo), float(d_hi)))
            xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
            if not self.kit.trav_free(xy, eroded=True) or not P0.in_task_rooms(self, xy):
                continue
            yaw = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
            trunks = [(float(t), 0.0) for t in RB.TRANSLATES]  # (i): straight torso, any height
            cands = [Stance("init", (float(xy[0]), float(xy[1]), yaw), t, ti, dist) for t, ti in trunks]
            cams = P0.camera_poses(self, cands)
            feas = []
            for st, cam in zip(cands, cams):
                if not P0.in_frame(self, cam, pts) or not self.goal_region_visible(st, from_xy=c_t):
                    continue
                feas.append((st, P0.axis_angle(self, cam, focus)))
            if not feas:
                continue
            # floor class: A spawns UNTUCKED like every other family's init (owner 2026-09-24; ``Stance.arms`` default). 1dd3820b
            # (2026-09-16) spawned it tucked because A then stood at B's crouch (0.95, 0) and the untucked hands hanging at floor
            # level swept the floor target on the A -> B drive; A is now the most upright straight trunk that sees the target
            # (below) and the crouch happens at the transition, where the arms were untucked again anyway (arm_to(left,right),
            # 161 steps). The A -> B path sweep still judges the hands: an A whose untucked hands reach the target is rejected
            # ``path_collision`` and the init is re-drawn.
            free = P0.body_free(self, [st for st, _ in feas])
            feas = [(st, a) for (st, a), f in zip(feas, free.tolist()) if f]
            if not feas or not P0.trunk_sweep_free(self, (float(xy[0]), float(xy[1]), yaw)):
                continue
            # the most upright translate wins (the hands highest on the drive, the carry's travel trunk straightest); the
            # optical-axis angle breaks ties (owner 2026-09-24)
            like_t = None if like is None else float(like.translate)
            st = min(feas, key=lambda sa: (sa[0].translate, round(sa[1], 2)))[0]
            st.extras.update({"bearing_deg": math.degrees(bearing), "distance": dist, "trunk_like_b": bool(like_t is not None and abs(st.translate - like_t) < 1e-6)})
            return st
        raise Reject("no_stance", "floor init: no stance A sees the target at c_t and the partner")

    # ------------------------------------------------------------------ stance C (placement)

    def placement_candidates(self, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """§5.6 stance C survivors for an init pose: base square to the partner edge nearest ``init_xy`` — when that edge
        admits no stance (no free floor in front of it: countertop-floor 20:24, edge +v ``base_ok 0``), the other
        edges follow in order of distance from the init pose (§5.6 wording proposed: "the nearest edge that admits a
        stance") — stand-off {0.35, 0.45, 0.55}, lateral along the edge minus margins, trunk from the band; goal cell
        per relation; visibility per relation; the best angle class (each survivor carries its ``goal_cell`` in
        ``extras``). For ``arm`` (default: the segment arm; owner 2026-09-27, first-demo v2.2): its band, its push lanes, its
        working arm (the other arm's idle posture derived for it)."""
        arm = arm or self.seg_arm
        p = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        edges = sorted(self.place["edges"], key=lambda e: float(th.linalg.norm(e.mid - p)))
        allow = self.gp.gen("placement_edges")  # optional whitelist of partner edge names (+x / -x / +y / -y; top-raster edge names for ontop)
        if allow:
            edges = [e for e in edges if e.name in set(allow)] or edges
        # owner 2026-09-20 09:50: the survivors of EVERY edge are pooled (nearest edge first, ``extras.edge_rank``) - the coupling
        # decides; the nearest edge alone boxed the table probe in (the room behind the short end blocks every drive-back pose
        # of the carry arrival, 0920-092254). ``generation.placement_pool_edges = false`` restores "the nearest edge that admits
        # a stance" (spec 5.6 wording)
        pool = self.gp.gen("placement_pool_edges", True) is not False
        tried, out_all = [], []
        for rank, edge in enumerate(edges):
            out = self._placement_candidates_at(edge, init_xy, arm)
            if out:
                for st in out:
                    st.extras["edge_rank"] = rank
                if not pool:
                    if tried:
                        log.info("%s: stance C: edge(s) %s nearest the init pose admit no stance; %s taken", self.gp.alias, tried, edge.name)
                    return out
                out_all += out
            else:
                tried.append(edge.name)
        if pool and out_all:
            log.info("%s: stance C: %d survivors pooled over the edges %s%s", self.gp.alias, len(out_all), [e.name for e in edges if e.name not in tried],
                     f" (no stance at {tried})" if tried else "")
        return out_all

    def _placement_candidates_at(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        arm = arm or self.seg_arm
        partner = self.partner
        half = self.place["half"]
        if self.push_under:
            return self.push_under_candidates(edge, init_xy, arm)
        rb = self.band("tabletop", partner, arm=arm) if self.relation == "ontop" else self.band("floor", None, partner, arm=arm)
        self.place["band"] = rb
        self.place.setdefault("bands", {})[arm] = rb
        trunks = rb.trunks() if rb is not None else [(t, k) for t in RB.TRANSLATES for k in RB.TILTS]
        margin = half + 0.10
        L = edge.length
        half_run = L / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        init_xy = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
        goal_cells = self.place["cells"]
        stats: Dict[str, int] = {"goal_cells": int(len(goal_cells)), "band_trunks": len(trunks), "lattice": 0, "base_ok": 0, "band_cells": 0, "goal_in_band": 0}
        cands: List[Stance] = []
        for so in [float(v) for v in (self.gp.gen("stand_offs_m") or STAND_OFFS)]:
            for lat in laterals:
                xy = edge.base_xy(so, lat)
                stats["lattice"] += len(trunks)
                if not P0.base_ok(self, xy):
                    continue
                stats["base_ok"] += len(trunks)
                for tr, ti in trunks:
                    st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), tr, ti, so, edge, None, lat, None, [arm])
                    cells = self.band_cells_world(rb, st)
                    if rb is not None and len(cells) == 0:
                        continue
                    stats["band_cells"] += 1
                    pool = goal_cells if rb is None else self._snap(cells, goal_cells, float(rb.meta.get("lattice", RB.LATTICE)))
                    if len(pool) == 0:
                        continue
                    stats["goal_in_band"] += 1
                    gc = pool[int(th.argmin(th.linalg.norm(pool - init_xy, dim=1)))]
                    st.extras["goal_cell"] = [float(gc[0]), float(gc[1])]
                    cands.append(st)
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: stance C at %s edge %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        free = P0.body_free(self, cands)
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if not cands:
            log.info("%s: stance C at %s edge %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        tangent = th.tensor([-float(edge.inward[1]), float(edge.inward[0])])
        classed = []
        for st, cam in zip(cands, cams):
            gc = st.extras["goal_cell"]
            if self.relation == "ontop":
                pose = self.goal_pose_at(gc, self.rest)
                if not P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose)):
                    continue
                focus = th.tensor([gc[0], gc[1], self.place["z"] + self.place["height"] / 2])
                # owner decision 2026‑09‑16 (c): the ontop goal-visibility rule is "goal cell (the target at the goal) in frame
                # (5 %)" — the 20° optical-axis cone stays for under only (the target hides beneath the partner). Diagnosed
                # on box_of_shampoo-sofa-bar: 13/13 collision-free C candidates had the goal in frame, unoccluded, at 30–31° off
                # the axis (camera rigid on the torso, trunk (0, −1.2) from the band, stand-off 0.35 m); an optional cone
                # (``ontop_axis_cone_deg``) is kept for experiments.
                cone = self.gp.gen("ontop_axis_cone_deg")
                ang = P0.axis_angle(self, cam, focus)
                st.extras["axis_angle_deg"] = math.degrees(ang)
                k = (1 if ang <= math.radians(float(cone)) else None) if cone is not None else 1
            else:
                # the near edge's bottom line where the target passes under it — the two bottom points of the edge at the
                # goal cell's lateral position ± (half extent + 0.10) — in frame; the optical axis within 20° of the
                # (hidden) goal cell. (The spec's "two bottom AABB corners" lie ±1 m to the sides of a bed / sofa edge, 65°
                # off the optical axis at s = 0.45 m — outside the 99° hfov for every stance; wording change proposed.)
                corners = G.aabb_corners(lo.float(), hi.float())
                bottom = corners[corners[:, 2] <= float(lo[2]) + 1e-4]
                d = (bottom[:, :2] - edge.mid) @ edge.inward
                near_line = bottom[d.abs() <= float(d.abs().min()) + 0.02]
                on_line = near_line[:, :2].mean(dim=0) + tangent * float(th.dot(th.tensor([gc[0], gc[1]]) - near_line[:, :2].mean(dim=0), tangent))
                span = half + 0.10
                near = th.stack([th.cat([on_line - tangent * span, lo[2:3].float()]), th.cat([on_line + tangent * span, lo[2:3].float()])])
                if not self.partner_points_in_view(cam, near):
                    continue
                focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                ang = P0.axis_angle(self, cam, focus)
                # ``generation.under_axis_max_deg`` (owner 2026-10-12 item 2c; default the 20° above): the under band's crouched head
                # hangs over the edge like the push-under's (45° there)
                cone_u = self.gp.gen("under_axis_max_deg")
                st.extras["axis_angle_deg"] = math.degrees(ang)
                k = 1 if ang <= math.radians(float(P0.AXIS_LOOSE_DEG if cone_u is None else cone_u)) else None
            if k is None:
                continue
            st.focus = focus
            st.extras["angle_class"] = k
            classed.append((st, k))
        stats["visible"] = len(classed)
        out = P0.best_class(classed)
        log.info("%s: stance C at %s edge %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out), stats)
        if not out and cands:
            self.debug_placement_visibility(cands, cams, edge, init_xy)
        return out

    def _object_of_prim(self, prim_path: str) -> Optional[str]:
        for o in self.world.scene.objects:
            for l in o.links.values():
                if prim_path == l.prim_path or prim_path.startswith(l.prim_path + "/"):
                    return o.name
        return None

    def debug_placement_visibility(self, cands: Sequence[Stance], cams: Sequence[Any], edge: EdgeFrame, init_xy) -> None:
        """Owner decision 2026‑09‑16 (c) diagnosis: for every collision-free stance-C candidate the goal cell's frame test
        (5 %), its angle to the optical axis, the first object a ray from the camera to the goal point hits (occluder), the
        trunk / stand-off / lateral; plus the partner edges that were considered and why one was chosen (the edge nearest
        the init pose is the only one the spec uses)."""
        try:
            partner = self.partner
            z_top = float(self.place["z"])
            rows = []
            for st, cam in zip(cands, cams):
                gc = st.extras.get("goal_cell")
                if gc is None:
                    continue
                if self.relation == "ontop":
                    focus = th.tensor([gc[0], gc[1], z_top + self.place["height"] / 2])
                    pose = self.goal_pose_at(gc, self.rest)
                    corners_in = P0.in_frame(self, cam, self.hull_corners_world(self.hull, pose))
                else:
                    focus = th.tensor([gc[0], gc[1], self.floor_z + self.place["height"] / 2])
                    corners_in = None
                cell_in = P0.in_frame(self, cam, focus.reshape(1, 3))
                ang = math.degrees(P0.axis_angle(self, cam, focus))
                cam_p = G.t(cam[0]).float()
                hit = self.world.raycast(cam_p.reshape(1, 3), focus.reshape(1, 3))[0]
                occluder = None
                if hit.get("hit"):
                    name = self._object_of_prim(hit.get("rigidBody", ""))
                    d_hit = float(th.linalg.norm(th.as_tensor(hit["position"]).float() - cam_p))
                    d_goal = float(th.linalg.norm(focus - cam_p))
                    if name is not None and name != partner.name and d_hit < d_goal - 0.03:
                        occluder = f"{name} at {d_hit:.2f} m (goal at {d_goal:.2f} m)"
                    elif name == partner.name and d_hit < d_goal - 0.05:
                        occluder = f"{partner.name} itself at {d_hit:.2f} m (goal at {d_goal:.2f} m: armrest / back / cushion in front of the cell)"
                rows.append({"stand_off": round(float(st.stand_off or 0.0), 2), "lateral": round(float(st.lateral), 2), "trunk": [float(st.translate), round(float(st.tilt), 2)],
                             "goal_cell": [round(float(v), 2) for v in gc], "goal_in_frame": bool(cell_in), "goal_hull_in_frame": corners_in, "axis_angle_deg": round(ang, 1),
                             "occluder": occluder, "camera_z": round(float(cam_p[2]), 2)})
            f = th.as_tensor([float(init_xy[0]), float(init_xy[1])])
            edges = [{"name": e.name, "length_m": round(float(e.length), 2), "dist_to_init_m": round(float(th.linalg.norm(e.mid - f)), 2), "chosen": e.name == edge.name,
                      "why": "nearest to the init pose (§5.6: edge = the partner edge nearest the target's init pose)" if e.name == edge.name else "not the nearest edge"}
                     for e in self.place["edges"]]
            self.template_facts["placement_visibility_debug"] = {"candidates": rows, "edges": edges}
            n_in = sum(1 for r in rows if r["goal_in_frame"])
            n_cone = sum(1 for r in rows if r["axis_angle_deg"] <= P0.AXIS_LOOSE_DEG)
            n_occ = sum(1 for r in rows if r["occluder"])
            log.info("%s: stance C visibility debug at %s edge %s: %d candidates — goal cell in frame %d, within the 20° cone %d, occluded %d; edges %s; per candidate %s",
                     self.gp.alias, partner.name, edge.name, len(rows), n_in, n_cone, n_occ, edges, rows)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            log.info("%s: stance C visibility debug failed: %r", self.gp.alias, e)

    def push_under_candidates(self, edge: EdgeFrame, init_xy, arm: Optional[str] = None) -> List[Stance]:
        """Stance C of the push-under (owner decision 2026‑09‑16 item 7): base square to the partner's near AABB edge,
        facing inward (= the push direction); per lateral position λ the **goal cell** g = edge point + inward·0.07
        (target centre 5–10 cm inside the footprint, clear under: ray ≥ height + 3 cm) and the **drop cell** c_d = edge
        point − inward·(half + 0.10) on free floor; stand-off s ∈ {0.5…0.7} so that c_d lies 0.25–0.6 m ahead of the
        base (B3: the floor is reachable 0.24–0.64 m ahead at the deep trunks); trunks = the floor trunks (translate ≥
        0.75, tilt ≤ −0.9); whole-body collision-free (A6 idle-arm retry), the drop footprint and the edge's bottom line
        at λ ± (half + 0.10) in frame, optical axis within 20° of the drop cell. Each survivor carries ``goal_cell``,
        ``drop_cell``, ``push_dir``, ``drop_ahead_m``. The lanes are laid out for the right arm and mirrored for the left
        (owner 2026-09-27, first-demo v2.2: :func:`frontal_lane_lateral`, :func:`side_push_pose`)."""
        arm = arm or self.seg_arm
        partner = self.partner
        half, height = float(self.place["half"]), float(self.place["height"])
        need = float(self.place.get("need", height + 0.03))
        d2 = th.as_tensor([float(edge.inward[0]), float(edge.inward[1])])
        tangent = th.tensor([-float(d2[1]), float(d2[0])])
        inside = float(self.num("push_under_inside_m", PUSH_UNDER_INSIDE_M))
        out = half + float(self.num("push_under_drop_extra_m", PUSH_UNDER_DROP_EXTRA_M))
        margin = half + 0.10
        half_run = edge.length / 2 - margin
        laterals = P0.lattice_values(-half_run, half_run, 0.10) if half_run > 0 else [0.0]
        # side stances may use the lane up to the edge's end minus the target's half extent (the deep trunks were clear only
        # with the base near the +tangent end of the bed edge: the lane 0.45 m further along is at the edge's end)
        half_run_side = edge.length / 2 - half - 0.02
        laterals_side = P0.lattice_values(-half_run_side, half_run_side, 0.10) if half_run_side > 0 else [0.0]
        stand_offs = [float(v) for v in (self.gp.gen("push_under_stand_offs_m") or PUSH_UNDER_STAND_OFFS)]
        trunks = [tuple(t) for t in (self.gp.gen("floor_trunks") or FLOOR_TRUNKS)]
        lo_ahead, hi_ahead = PUSH_UNDER_DROP_AHEAD_M
        cells = self.place["cells"]
        mid = th.as_tensor([float(edge.mid[0]), float(edge.mid[1])])
        stats: Dict[str, Any] = {"laterals": len(laterals), "goal_clear": 0, "drop_free": 0, "base_ok": 0, "candidates": 0, "side_candidates": 0, "drop_blocked": {},
                                 "drop_out_m": round(out, 3), "edge_mid": [round(float(v), 2) for v in mid.tolist()], "inward": [round(float(v), 2) for v in d2.tolist()]}
        # side stances: the base stands BESIDE the push lane, facing along the edge with the partner on its right (the
        # right arm pushes sideways, toward the partner; the left arm faces the other way); the lane lies ``x_ahead`` ahead of the base and the base centre
        # ``b`` outside the edge — the whole lane (drop 0.22 m outside → goal 0.07 m inside) stays 0.35–0.45 m ahead, inside
        # the floor reach, and the crouched torso bends along the edge, not into the partner (the frontal stances need
        # s ≥ 0.75 m to keep the deep trunk clear of a bed, which puts the lane 0.46–0.82 m ahead: out of reach)
        # the lane must clear the base rim: with x_ahead = 0.35 the blade start (0.35 m ahead, 6 cm high) sits on the
        # base's front edge and every push pre-pose fails the IK (run 16:04) — 0.45 m and more only
        side_ahead = [float(v) for v in (self.gp.gen("push_under_side_ahead_m") or (0.45, 0.55, 0.6))]
        side_b = [float(v) for v in (self.gp.gen("push_under_side_standoff_m") or (0.35, 0.40))]
        cands: List[Stance] = []
        lat_all = sorted({round(float(v), 4) for v in list(laterals) + list(laterals_side)})
        lat_frontal = {round(float(v), 4) for v in laterals}
        for lat in lat_all:
            g = mid + tangent * float(lat) + d2 * inside
            cl = self.clearance_at(g)
            clear = (cl is not None and cl >= need) or (cl is None and len(cells) > 0 and float(th.linalg.norm(cells - g, dim=1).min()) <= 0.03)
            if not clear:
                continue
            stats["goal_clear"] += 1
            c_d = mid + tangent * float(lat) - d2 * out
            if not self._free_floor_cell(c_d, stats["drop_blocked"]):
                continue
            stats["drop_free"] += 1
            common = {"goal_cell": [float(g[0]), float(g[1])], "drop_cell": [float(c_d[0]), float(c_d[1])], "push_dir": [float(d2[0]), float(d2[1])], "inside_m": inside,
                      "edge": edge.name}
            # frontal stances: the lane to the robot's RIGHT of the base centre by ``frontal_lane_offset_m`` — B3's under-edge
            # probe (bed_rrcvaq_0, briefcase, s = 0.6) reaches the edge strip at lateral −0.40…−0.16 only (the right arm's
            # side), a centred lane at s = 0.45 / 0.6 has 0–1 cells; the left arm's lane to its LEFT
            for off in ([float(v) for v in (self.gp.gen("frontal_lane_offset_m") or (0.25, 0.0))] if lat in lat_frontal else []):
                for so in stand_offs:
                    ahead = so - out
                    if ahead < lo_ahead or ahead > hi_ahead:
                        continue
                    lat_b = frontal_lane_lateral(float(lat), off, arm)
                    xy = edge.base_xy(so, lat_b)  # base beside the lane by ``off``, away from the arm's side (+tangent = the robot's left)
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), float(tr), float(ti), so, edge, None, lat_b, None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(ahead), stance_kind="frontal", lane_offset_m=arm_sign(arm) * off)
                        cands.append(st)
                        stats["candidates"] += 1
            for x_ahead in side_ahead:
                for b in side_b:
                    x_b, y_b, yaw_b = side_push_pose(mid, tangent, d2, float(lat), x_ahead, b, arm)
                    xy = th.tensor([x_b, y_b])
                    if not P0.base_ok(self, xy):
                        continue
                    stats["base_ok"] += 1
                    for tr, ti in trunks:
                        st = Stance("placement", (x_b, y_b, yaw_b), float(tr), float(ti), b, None, None, float(lat), None, [arm])
                        st.focus = th.tensor([float(c_d[0]), float(c_d[1]), self.floor_z + height / 2])
                        st.extras.update(common, drop_ahead_m=float(x_ahead), stance_kind="side", side_standoff_m=b)
                        cands.append(st)
                        stats["side_candidates"] += 1
        self.template_facts["placement_stats"] = stats
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: no candidate (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        # the deep trunks (translate 0.95) are what reaches the floor (B3), and at a floor stance they trip the 1 cm margin
        # against the floor *mesh* (torso / head "within 1 cm of the floor" — B3 README; scene-specific, the floor object
        # sits a little above floor_z): the stance check runs with the floor object removed, the cuRobo ground slab
        # (which stays) keeps the robot from penetrating the floor. Documented relaxation for the push-under stance.
        with self.ikc.ignoring([self.obj("floor")]):
            free = P0.body_free(self, cands)
        cands_all, free_all = list(cands), free
        deep_fail = [st for st, f in zip(cands, free.tolist()) if not f and st.translate >= 0.95]
        cands = [st for st, f in zip(cands, free.tolist()) if f]
        stats["collision_free"] = len(cands)
        if deep_fail and not any(st.translate >= 0.95 for st in cands) and not getattr(self, "_deep_debugged", False):
            # which spheres / postures fail at the floor-reaching trunks (B3: "torso / head within 1 cm of the floor")
            self._deep_debugged = True
            with self.ikc.ignoring([self.obj("floor")]):
                P0.debug_collisions(self, deep_fail[len(deep_fail) // 2:len(deep_fail) // 2 + 2], arms_excluded=False)
        if not cands:
            log.info("%s: push-under stance C at %s edge %s: every candidate collides (%s)", self.gp.alias, partner.name, edge.name, stats)
            return []
        cams = P0.camera_poses(self, cands)
        lo, hi = partner.aabb
        z_bottom = float(lo[2])
        classed = []
        for st, cam in zip(cands, cams):
            c_d = th.as_tensor(st.extras["drop_cell"])
            # the drop footprint (a disc of the half extent around the drop cell, on the floor) and the edge's bottom line at
            # the push lane in frame (5 %); the optical axis within 20° of the drop cell
            disc = th.stack([th.cat([c_d + tangent * half, th.tensor([self.floor_z])]), th.cat([c_d - tangent * half, th.tensor([self.floor_z])]),
                             th.cat([c_d + d2 * half, th.tensor([self.floor_z])]), th.cat([c_d - d2 * half, th.tensor([self.floor_z])])])
            e0 = mid + tangent * float(st.lateral)
            lane = th.cat([e0, th.tensor([z_bottom])]).reshape(1, 3)  # the edge's bottom point at the push lane
            # a side stance looks along the edge: the lane point sits at the frame's lateral border (45–49° off-axis at
            # b = 0.35–0.40 m); the drop footprint in frame is what it needs (the partner's side fills that half of the image)
            pts = disc if st.extras.get("stance_kind") == "side" else th.cat([disc, lane])
            kind = st.extras.get("stance_kind", "frontal")
            if not P0.in_frame(self, cam, pts):
                stats[f"{kind}_out_of_frame"] = stats.get(f"{kind}_out_of_frame", 0) + 1
                if kind == "side" and stats[f"{kind}_out_of_frame"] == 1:
                    log.info("%s: side stance C example out of frame: base %s trunk %s camera %s focus %s disc %s", self.gp.alias, [round(float(v), 2) for v in st.base_pose2d],
                             (st.translate, st.tilt), [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()],
                             [[round(float(v), 2) for v in p] for p in disc.tolist()])
                continue
            ang = P0.axis_angle(self, cam, st.focus)
            # at the crouched trunks the head hangs ≈ 0.6 m ahead of the base, i.e. over the lane: the drop cell lies almost
            # straight below the camera, 30–40° off the (74–90° pitched) optical axis while squarely in frame — the 20° rule
            # is relaxed to ``push_under_axis_max_deg`` (45°) for the push-under stance (deviation from §5.6 to record)
            if ang > math.radians(float(self.num("push_under_axis_max_deg", 45.0))):
                stats[f"{kind}_axis_off"] = stats.get(f"{kind}_axis_off", 0) + 1
                if kind == "side" and stats[f"{kind}_axis_off"] == 1:
                    log.info("%s: side stance C example axis off by %.0f deg: trunk %s camera %s focus %s", self.gp.alias, math.degrees(ang), (st.translate, st.tilt),
                             [round(float(v), 2) for v in cam[0].tolist()], [round(float(v), 2) for v in st.focus.tolist()])
                continue
            st.extras["angle_class"] = 1
            classed.append((st, 1))
        stats["visible"] = len(classed)
        out_ = P0.best_class(classed)
        stats["survivor_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in out_})
        stats["survivor_stand_offs"] = sorted({round(float(st.stand_off), 2) for st in out_})
        stats["survivor_kinds"] = {k: sum(1 for st in out_ if st.extras.get("stance_kind") == k) for k in ("side", "frontal")}
        stats["side_survivors_by_ahead"] = {str(a): sum(1 for st in out_ if st.extras.get("stance_kind") == "side" and abs(float(st.extras.get("drop_ahead_m", -1)) - a) < 1e-6)
                                            for a in side_ahead}
        side_fail = [st for st, f in zip(cands_all, free_all.tolist()) if not f and st.extras.get("stance_kind") == "side" and float(st.extras.get("drop_ahead_m", 0)) >= 0.45]
        if side_fail and not any(st.extras.get("stance_kind") == "side" and float(st.extras.get("drop_ahead_m", 0)) >= 0.45 for st in out_) \
                and not getattr(self, "_side_debugged", False):
            self._side_debugged = True
            with self.ikc.ignoring([self.obj("floor")]):
                P0.debug_collisions(self, side_fail[len(side_fail) // 2:len(side_fail) // 2 + 2], arms_excluded=False)
        stats["collision_free_trunks"] = sorted({(float(st.translate), round(float(st.tilt), 2)) for st in cands})
        log.info("%s: push-under stance C at %s edge %s: %d survivors (%s)", self.gp.alias, partner.name, edge.name, len(out_), stats)
        return out_

    def debug_drop_ik(self, place: Stance, release_xy, yaws: Sequence[float]) -> str:
        """Diagnostic when F_goal is empty at the drop cell: for the first yaw and every library grasp, the drop grasp pose
        from stance C with the floor ignored vs. with every collision object ignored (reach vs. collision), and the eef
        height above the floor / distance ahead of the base."""
        try:
            floor = self.obj("floor")
            others = [o for o in self.world.scene.objects if o.name in self.world.scene_object_names]
            yaw = float(yaws[0]) if len(yaws) else 0.0
            quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
            pose = self.drop_pose_at(release_xy, quat)
            parts = []
            for g in self.grasps[:3]:
                eef = IK.grasp_pose(pose, g.pose)
                r1 = self.ikc.exact([eef], place, self.seg_arm, ignore_for_last=[floor], cache=False)
                with self.ikc.ignoring(others):
                    r2 = self.ikc.exact([eef], place, self.seg_arm, cache=False)
                p_base = P0.to_base_frame(eef[0], place.base_pose2d, P0.base_z(self))
                parts.append(f"{g.name}: floor-ignored {r1.ok}, world-free {r2.ok}, eef in base {[round(float(v), 2) for v in p_base.tolist()]}")
            return f"stance d {place.stand_off:.2f} trunk ({place.translate}, {place.tilt:.1f}) drop ahead {place.extras.get('drop_ahead_m')}: " + "; ".join(parts)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            return f"debug failed: {e}"

    @staticmethod
    def _snap(band_world: th.Tensor, goal_cells: th.Tensor, lattice: float) -> th.Tensor:
        """Goal cells within half a lattice step of a band cell (the band admits them)."""
        if len(band_world) == 0 or len(goal_cells) == 0:
            return th.zeros(0, 2)
        d = th.cdist(goal_cells.float(), band_world.float())
        return goal_cells[d.min(dim=1).values <= 0.51 * lattice]

    def goal_pose_at(self, goal_cell, quat) -> G.Pose:
        """§5.6 goal pose: ontop → z = partner face + bottom offset + 1 cm; under → z = floor + bottom offset + 3 mm."""
        z = self.place["z"] + self.world.bottom_offset(self.hull, quat) + self.num("goal_drop_m", 0.01 if self.relation == "ontop" else 0.003)
        return th.tensor([float(goal_cell[0]), float(goal_cell[1]), z]), quat

    # ------------------------------------------------------------------ template

    def sample_template(self):
        self.template_facts.update({"relation": self.relation, "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "band_union_extent": None if self.surface.get("band") is None else list(self.surface["band"].union_extent())})

    # ------------------------------------------------------------------ instance (§5.6)

    def pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        quat = G.qmul(G.quat_from_yaw(yaw), self.rest)
        return th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull, quat) + 0.003]), quat

    def yaws(self) -> List[float]:
        """The object yaws screened for F_init / F_goal: ``generation.n_yaws`` equal steps (default 16 - the 8 of the reach
        bands leave 2-4 feasible pairs for a floor pick at the crouch)."""
        n = int(self.num("n_yaws", 16))
        return [k * 2.0 * math.pi / n for k in range(n)]

    # ------------------------------------------------------------------ legs A -> B and B -> C (owner 2026‑09‑19 21:56)

    def _drop_leg_records(self, to: str) -> None:
        self.sidecar.paths = [p for p in self.sidecar.paths if p.to != to]
        self.sidecar.transitions = [t for t in self.sidecar.transitions if t.to != to]

    def record_leg_a_b(self, init: Stance, manip: Stance) -> Dict[str, Any]:
        """(ii) Init -> grasp stance. Primary: **navigate at the init trunk, then the torso** — the §3.7 path at A's
        trunk / arms and the §3.8 arrival transition at B (trunk sweep at s_T on B's axis, drive-in with B's arms).
        Fallback when no s_T admits that: **torso first, then navigate** — a departure transition at A (drive back
        along A's own axis if needed, sweep to B's trunk there) and the path driven at B's trunk with B's arms from that
        pose (what ``729a1ec`` did by spawning A in B's crouch; A itself stays upright, (i)). ``Reject`` when neither
        leg exists. Returns the leg facts for the summary."""
        errors: List[str] = []
        try:
            path = self.record_path(init, manip)
            tr = self.record_transition(init, manip)
            return {"order": "navigate_then_torso", "s_T": tr.stand_off, "path_m": round(path.length_m, 3)}
        except Reject as e:
            if e.reason not in ("no_transition", "path_collision", "no_path"):
                raise
            errors.append(f"navigate-then-torso: {e.reason}: {e}")
            self._drop_leg_records(manip.name)
        b_trunk = (float(manip.translate), float(manip.tilt))
        try:
            from ispatialgym.sampling.v2.ik import stance_arm_vectors

            dep = self.record_departure_transition(init, manip, b_trunk)
            r, l, _ = stance_arm_vectors(self.kit, manip)  # the oracle moves both arms to B's posture before the sweep
            path = self.record_path(init, manip, trunk=b_trunk, arm_q={"right": r, "left": l}, start_pose2d=NAV.departure_pose(init, dep.stand_off))
            log.info("%s: A -> B: navigate-then-torso impossible (%s); torso-then-navigate at A (s_T %.2f, drive at trunk %s)", self.gp.alias, errors[-1], dep.stand_off,
                     tuple(round(v, 2) for v in b_trunk))
            return {"order": "torso_then_navigate", "s_T": dep.stand_off, "path_m": round(path.length_m, 3), "primary_error": errors[-1]}
        except Reject as e:
            self._drop_leg_records(manip.name)
            errors.append(f"torso-then-navigate: {e.reason}: {e}")
        raise Reject("no_transition", "init -> manipulation: " + "; ".join(errors))

    def _leg_ab_clear_of_target(self, init: Stance, manip: Stance, t_pose) -> Optional[str]:
        """Owner 2026-09-26 (shampoo 0926-015905, wave 8: navigate[0] 'wheel_motor_link1 in contact with the dispenser' twice - the
        recorded A -> B line passed 5.6 cm from the dispenser standing on the floor): the A -> B leg is recorded before the target
        has a pose (``record_leg_a_b`` runs with the target parked), so its sweep never saw where the target ends up - harmless for
        a target on a table, not for one on the floor B drives up to. Re-sweep the recorded A -> B path (its strides + the
        yaw-lag envelope, the trunk and arms it was checked with, the floor relaxation of the original check) with the target
        placed at ``t_pose``; returns the reason when it collides, None when it clears (or when there is no path)."""
        path_ab = next((p for p in self.sidecar.paths if p.to == manip.name), None)
        if path_ab is None or len(path_ab.waypoints) < 2:
            return None
        from ispatialgym.sampling.v2.ik import stance_arm_vectors

        trunk = tuple(float(v) for v in path_ab.trunk) if path_ab.trunk is not None else (float(init.translate), float(init.tilt))
        at_init = abs(trunk[0] - float(init.translate)) < 1e-6 and abs(trunk[1] - float(init.tilt)) < 1e-6
        r, l, _ = stance_arm_vectors(self.kit, init if at_init else manip)
        q_arms = {"right": r, "left": l}
        if path_ab.arms_q:
            q_arms.update({a: th.as_tensor(q, dtype=th.float32) for a, q in path_ab.arms_q.items()})
        tgt = self.target
        p0, q0 = (x.clone() for x in tgt.get_position_orientation())
        try:
            tgt.set_position_orientation(*t_pose)
            tgt.keep_still()
            self.kit.sync_world()
            poses = NAV.stride_poses(path_ab.waypoints, float(path_ab.stride_m or NAV.STRIDE_M), yaw_envelope=True)
            with self._floor_relaxed():
                ok, bad = NAV.path_collision_free(self.kit, poses, trunk, path_ab.posture, margin=0.01, q_arms=q_arms)
        finally:
            tgt.set_position_orientation(p0, q0)
            tgt.keep_still()
            self.kit.sync_world()
        if ok:
            return None
        return (f"B: the A -> B drive passes the target drawn at ({float(t_pose[0][0]):.3f}, {float(t_pose[0][1]):.3f}): swept pose {bad}/{len(poses)} "
                f"({tuple(round(float(v), 3) for v in poses[bad])}) collides with it placed")

    def travel_trunks(self, init: Stance, manip: Stance) -> List[Tuple[float, float]]:
        """(iii') carry trunks in order: the grasp stance's translate with the torso straightened (tilt 0 — the object
        stays low), then the init translate straightened; ``generation.travel_trunks`` overrides."""
        cfg = self.gp.gen("travel_trunks")
        if cfg:
            return [(float(t), float(k)) for t, k in cfg]
        out: List[Tuple[float, float]] = []
        for t in (float(manip.translate), float(init.translate)):
            if all(abs(t - u) > 1e-6 for u, _ in out):
                out.append((t, 0.0))
        return out

    def record_carry_leg(self, init: Stance, manip: Stance, place: Stance, tname: str, q_lift_arm: Optional[th.Tensor], max_len: float) -> Any:
        """(iii') Grasp stance -> place stance with the object in hand. Primary: **straighten the torso, then navigate**
        — a departure transition at B (drive back along B's own axis to the first s_T where the sweep to the travel trunk
        is free — "a spot along the way it has moved before" — with the holding arm at its LIFT configuration and the
        object attached), the path from that pose at the travel trunk, and the arrival transition at C (the pre-place
        stand-off where C's trunk is set, then the drive-in). Every travel trunk of :meth:`travel_trunks` is tried.
        Fallback: the drive at B's trunk (the earlier behaviour) with the arrival transition. The holding arm is the segment arm.
        Returns the path record."""
        arm = self.seg_arm
        if q_lift_arm is not None:
            self.record_carry_arm_q(tname, arm, q_lift_arm)
        # the placement stance is reached WITH the object in hand: the objects-placed check attaches it there (and ignores it at
        # its init pose - 0920-004500: stance C 'collided' with the target still standing on its floor cell next to C)
        place.carried = tname
        errors: List[str] = []
        for tt in self.travel_trunks(init, manip):
            if abs(tt[0] - manip.translate) < 1e-6 and abs(tt[1] - manip.tilt) < 1e-6:
                continue  # B already straight: the plain leg below is the same thing
            try:
                dep = self.record_departure_transition(manip, place, tt, carried=tname, arm=arm)
                start = NAV.departure_pose(manip, dep.stand_off)
                try:
                    path = self.record_path(manip, place, carried=tname, arm=arm, max_length_m=max_len, trunk=tt, start_pose2d=start)
                except Reject as e:
                    if e.reason != "path_collision":
                        raise
                    # the local (un-eroded) shortest path hugs the partner's corner with the object out front (0920-100543: stride 4/35
                    # at the table end); the eroded global map keeps the body's distance
                    self._drop_leg_records(place.name)
                    dep = self.record_departure_transition(manip, place, tt, carried=tname, arm=arm)
                    path = self.record_path(manip, place, carried=tname, arm=arm, max_length_m=max_len, trunk=tt, start_pose2d=start, force_global=True)
                # a carry into a deep crouch backs off further before the sweep (the hand frozen at the lift pose swings ~0.5 m
                # forward / 0.9 m down): s_T up to stand-off + generation.carry_transition_extra_m (1.0)
                self.record_transition(manip, place, carried=tname, arm=arm, trunk_from=tt, extra_max=self.num("carry_transition_extra_m", 1.0))
                where = "at the grasp spot" if dep.stand_off <= (manip.stand_off or 0.0) + 1e-6 else "after a drive-back"
                log.info("%s: B -> C carry: torso straightened to %s at s_T %.2f (%s), path %.2f m", self.gp.alias, tuple(round(v, 2) for v in tt), dep.stand_off, where,
                         path.length_m)
                self._leg_bc = {"order": "straighten_then_navigate", "travel_trunk": [tt[0], tt[1]], "s_T_departure": dep.stand_off, "straightened": where,
                                "path_m": round(path.length_m, 3), "lift_arm_q": q_lift_arm is not None, "tried": errors}
                return path
            except Reject as e:
                self._drop_leg_records(place.name)
                errors.append(f"travel trunk {tuple(round(v, 2) for v in tt)}: {e.reason}: {e}")
        # fallback: drive at B's trunk (the object low in the crouch), the trunk set at C's pre-place stand-off
        try:
            try:
                path = self.record_path(manip, place, carried=tname, arm=arm, max_length_m=max_len)
            except Reject as e:
                if e.reason != "path_collision":
                    raise
                self._drop_leg_records(place.name)
                path = self.record_path(manip, place, carried=tname, arm=arm, max_length_m=max_len, force_global=True)
            self.record_transition(manip, place, carried=tname, arm=arm, extra_max=self.num("carry_transition_extra_m", 1.0))
            log.info("%s: B -> C carry: no travel trunk admits straighten-then-navigate (%s); driving at B's trunk %s", self.gp.alias, "; ".join(errors),
                     (manip.translate, round(manip.tilt, 2)))
            self.template_facts.setdefault("carry_fallbacks", []).append(errors)
            self._leg_bc = {"order": "drive_at_grasp_trunk", "travel_trunk": [manip.translate, manip.tilt], "path_m": round(path.length_m, 3),
                            "lift_arm_q": q_lift_arm is not None, "tried": errors}
            return path
        except Reject as e:
            self._drop_leg_records(place.name)
            raise Reject(e.reason, f"{e}; straighten-then-navigate: " + "; ".join(errors))

    #: carry pose candidates in B's base frame: (ahead, lateral, raise above the lift) — beside the body first (owner 2026-09-20
    #: 06:40: a pose in front of the chest swings INTO the base when the torso tilts 54 deg into the push-under crouch; a hand
    #: held to the right of the body swings down outside the base footprint and ends above the floor), then in front
    #: (0920-071230: the poses 0.20-0.30 m ahead fold the elbow onto its limit - right_arm_joint4 within 2-19 mrad, the 0.05 rad
    #: witness margin rejects them - so the beside-the-body poses sit 0.40-0.55 m ahead; the raise stays >= 0 so the straight
    #: line from the lift never dips toward the partner's top edge). The right arm's; the left arm holds them mirrored (:func:`carry_poses_for`)
    CARRY_POSES = ((0.45, -0.45, 0.0), (0.50, -0.40, 0.05), (0.40, -0.45, 0.10), (0.55, -0.35, 0.05), (0.35, -0.45, 0.15), (0.50, None, 0.10))

    def carry_poses_at_b(self, manip: Stance, eef: G.Pose, carry_height: float, target, eef_in_object) -> List[Tuple[G.Pose, List[G.Pose], th.Tensor]]:
        """(iii') the CARRY poses (owner 2026‑09‑20 05:20): after the lift the held object is brought to a compact pose —
        ``CARRY_POSES`` / ``generation.carry_poses`` as (ahead, lateral (None = the lift's), raise) in B's base frame,
        orientation kept — by a straight in-hand line (``Reorient`` mode "line", 3 intermediate waypoints + end), so that
        the trunk sweeps of the carry legs move a hand near the body, not one stretched 0.85 m over the partner
        (0920-043725: the frozen lift configuration swept the held sandal into the table top / the floor at every s_T up to
        1.75 m). Solved from B with the object attached: exact IK at the carry pose, §3.9 waypoints from the lift; the segment
        arm holds it (the poses beside the body on its side). Returns the feasible ``(carry pose, waypoints, 7 arm joints)``
        candidates in order (the legs decide)."""
        arm = self.seg_arm
        lift = (eef[0] + th.tensor([0.0, 0.0, float(carry_height)]), eef[1])
        base_xy = manip.base_xy
        fwd, left = manip.forward, manip.left
        rel = th.tensor([float(lift[0][0]), float(lift[0][1])]) - base_xy
        lat_lift = float(rel @ left)
        out: List[Tuple[G.Pose, List[G.Pose], th.Tensor]] = []
        why: List[str] = []
        for ahead, lat, raise_m in carry_poses_for(self.gp.gen("carry_poses") or self.CARRY_POSES, arm):
            lat_v = lat_lift if lat is None else float(lat)
            xy = base_xy + fwd * float(ahead) + left * lat_v
            carry = (th.tensor([float(xy[0]), float(xy[1]), float(lift[0][2]) + float(raise_m)]), lift[1])
            tag = f"({float(ahead):.2f}, {lat_v:+.2f}, z {float(carry[0][2]):.2f})"
            # the in-hand move is a manipulation action: the FOV monitor is armed (owner 2026-09-16 10:15) - every waypoint incl. the
            # carry pose must keep the fingertips in the head camera frame from B (0920-135626: reorient failed gripper_out_of_fov at
            # u = 1060 for the carry pose (0.55, -0.35))
            cam_b = self.camera_pose_at(manip)
            wps = list(C.line_waypoints(lift, carry, 4))
            if not all(self.ikc.eef_in_frame(w, cam_b) for w in wps):
                why.append(f"{tag}: out of the head camera frame")
                continue
            with attached(self.kit, arm, target, eef_in_object=eef_in_object):
                r = self.ikc.exact([carry], manip, arm, cache=False)
                if not r.ok or r.q is None:
                    why.append(f"{tag}: IK {r.reason}")
                    continue
                ok = self.ikc.waypoints_ok([lift] + wps, manip, arm, cache=False)
            if not ok.ok:
                why.append(f"{tag}: line from the lift {ok.reason}")
                continue
            out.append((carry, wps, r.q[self.kit.robot.arm_control_idx[arm]].clone()))
        log.info("%s: carry poses from B (lift %.2f m at %s in base): %d feasible%s", self.gp.alias, carry_height,
                 [round(float(v), 2) for v in P0.to_base_frame(lift[0], manip.base_pose2d, P0.base_z(self)).tolist()], len(out), ("; " + "; ".join(why)) if why else "")
        return out

    def lift_arm_q(self, manip: Stance, eef: G.Pose, carry_height: float, target, eef_in_object) -> Optional[th.Tensor]:
        """The 7 arm-joint values of the LIFT pose (the grasp raised ``carry_height`` straight up, orientation kept) solved
        from stance B with the object attached — the configuration the oracle carries with (``program_check`` solves the
        same ``lift_end``); None when the lift has no solution (the leg then uses the stance posture, as before). The segment arm's."""
        arm = self.seg_arm
        lift = (eef[0] + th.tensor([0.0, 0.0, float(carry_height)]), eef[1])
        # owner 2026-09-25 (shampoo 0925-171050 / 194056: 8/8 draws lost at the program check's lift - 'lift: 1/11 tracked chain
        # configurations collide' - after this end-pose check had passed): the lift is TRACKED from the grasp configuration exactly
        # as the program check tracks it (IKChecker.continue_chain: the oracle's differential IK, the grasp's supports and the
        # ground off for a floor-class lift), so an untrackable lift drops the grasp / stance here instead of the whole draw later
        q_grasp = self.ikc.solutions.get((arm, self.ikc._pose_key(eef)))
        floor_class = (getattr(self, "surface", None) or {}).get("kind") == "floor"
        ignore = list(self.grasp_ignore_objects())
        with attached(self.kit, arm, target, eef_in_object=eef_in_object):
            if q_grasp is not None and self.ikc.reaches(q_grasp, eef, arm):
                r = self.ikc.continue_chain(q_grasp, [eef, lift], arm, ignore=ignore, ignore_end=[], soft_stall_m=self.ikc.FREE_LEG_STALL_M,
                                            ignore_start=ignore + list(self.support_objects()), ground_along=not floor_class, ground_meshes_along=not floor_class)
                if not r.ok:
                    log.info("%s: lift %.2f m untrackable from the grasp configuration (%s): grasp skipped", self.gp.alias, carry_height, r.reason[:200])
                    return None
            else:
                r = self.ikc.exact([lift], manip, arm, ignore_for_last=ignore, cache=False)
        if not r.ok or r.q is None:
            return None
        return r.q[self.kit.robot.arm_control_idx[arm]].clone()

    # ------------------------------------------------------------------ per-attempt budget

    def _check_deadline(self, stage: str) -> None:
        """Per-attempt time budget (``ATTEMPT_BUDGET_S``): the attempt stops with ``Reject("time_budget")`` naming the stage
        it reached once the budget is spent (a > 2 h attempt is a bug, not a search)."""
        deadline = getattr(self, "_deadline", math.inf)
        if time.time() > deadline:
            spent = time.time() - getattr(self, "_attempt_t0", deadline)
            raise Reject("time_budget", f"attempt budget of {self.num('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent ({spent:.0f} s) at {stage}",
                         stage=stage)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init: Stance = self.surface["init"]
        self.sidecar.add_stance(init)
        kind = self.surface["kind"]
        b_pool = list(self.surface["manip"])
        errors: List[str] = []
        self._attempt_t0 = time.time()
        self._deadline = self._attempt_t0 + float(self.num("attempt_budget_s", ATTEMPT_BUDGET_S))
        focus_xy = self.surface["c_t"] if kind == "floor" else None
        z_init = self.surface["z"]
        yaws = self.yaws()
        # the arms of the segment (owner 2026-09-27, first-demo v2.2): F_init of the default arm at every B, of the other arm where its
        # band holds the target, each at the B with that arm working; a B's usable pairs are the union
        offered = TT.offered_arms(self)
        self.seg_arm = offered[0]
        arm_bands = self.surface.get("arm_bands") or {}
        # F_init pre-screen (countertop-floor 0919-222621: F_init 2-4 pairs per B at the floor crouch, 6 B tried blind and the
        # attempt spent its 900 s): the screening IK of F_init needs no leg, so it runs first over up to ``b_prescreen`` random B
        # candidates and the coupling starts from the B with the most (yaw, grasp) pairs
        n_pre = int(self.num("b_prescreen", 8))
        pre: List[Tuple[int, Stance, List[Any], Dict[str, Tuple[Stance, List[Any]]]]] = []

        def _screen(n: int) -> None:
            for i_b in range(min(n, len(b_pool))):
                self._check_deadline(f"F_init pre-screen of B candidate {len(pre) + 1}")
                kb = int(rng.integers(len(b_pool)))
                b_grid = b_pool.pop(kb)
                manip = b_grid.with_trunk(b_grid.translate, b_grid.tilt, "manipulation")
                manip.focus = b_grid.focus
                fxy = focus_xy if focus_xy is not None else manip.focus[:2]
                by_arm: Dict[str, Tuple[Stance, List[Any]]] = {}
                for a, st in self._b_stances(manip, offered).items():
                    F_a = self.ikc.feasible_sets([fxy], yaws, self.grasps, lambda xy, yaw: self.pose_on(xy, yaw, z_init), st, a, self.grasp_ignore_objects()).get(0, [])
                    by_arm[a] = (st, F_a)
                    if arm_bands.get(a) is not None and arm_bands[a].meta.get("mirrored_from"):
                        TT._count(self, "arm_band_mirrored.foci")
                        if not F_a:
                            TT._count(self, "arm_band_mirrored.empty_F")
                F_b = union_feasible({a: F for a, (_, F) in by_arm.items()}, 2.0 * math.pi / max(1, len(yaws)))
                lead = next((by_arm[a][0] for a in self._segment_order(manip, by_arm)), next(iter(by_arm.values()), (manip, []))[0])
                pre.append((len(F_b), lead, F_b, by_arm))
        _screen(n_pre)
        # order: a COMPACT grasp first (the target cell's distance ahead of the base, 10 cm buckets) among the B candidates whose F_init
        # has at least ``b_prescreen_min_f`` pairs, then the larger F_init - a grasp 0.85 m out over the table (0920-100542/3) leaves an
        # extended arm that neither lifts higher nor rounds the table corner while carrying; ``b_prescreen_order = "f"`` restores F-only
        min_f = int(self.num("b_prescreen_min_f", 8))
        # push-under (2026-09-20 19:50): the object keeps its yaw relative to the robot from B to C, so which face the floor blade meets at
        # C is decided by the init yaw graspable at B - a B whose F_init has no yaw with a large blade contact offset at C (a sandal
        # toe-first: 0.11 m; its strap: 0.03 m, the blade goal then 0.85 m ahead and out of reach, 0920-192539/40) is tried last
        pushable: Dict[int, float] = {}
        if self.push_under and pre:
            faces_of: Dict[str, Any] = {}

            def _faces(a: str):
                """The (base yaw, push direction) of ``a``'s stances C (the left arm's side stances face the other way)."""
                if a not in faces_of:
                    try:
                        pool0 = self.placement_candidates(pre[0][1].focus[:2] if pre[0][1].focus is not None else self.surface["c_t"], arm=a)
                    except Reject:
                        pool0 = []
                    faces_of[a] = {(round(float(c.yaw), 2), tuple(round(float(v), 3) for v in c.extras["push_dir"])) for c in pool0 if c.extras.get("push_dir") is not None}
                return faces_of[a]
            _faces(offered[0])
            z_lo = self.push_height(30.0)[1] - self.floor_z
            min_off_ = float(self.num("push_min_contact_m", 0.10))

            def _offsets() -> None:
                for idx, (_, m, _, by_arm) in enumerate(pre):
                    if idx in pushable:
                        continue
                    best = -1.0
                    for a, (_, F_a) in by_arm.items():
                        if not F_a:
                            continue
                        for c_yaw, pd in _faces(a):
                            d_yaw = 0.0 if str(self.gp.gen("carry_yaw", "relative")) == "world" else float(c_yaw - m.yaw)
                            for y in sorted({float(yy) for yy, _ in F_a}):
                                o = self.contact_offset_along(G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), th.as_tensor(pd), z_lo, z_lo + self.BLADE_SPAN_M)
                                if o is not None:
                                    best = max(best, o)
                    pushable[idx] = best
            _offsets()
            # no pushable B among the first batch: screen more (up to ``b_prescreen_batches`` batches) before coupling blind
            for _ in range(int(self.num("b_prescreen_batches", 3)) - 1):
                if max(pushable.values(), default=-1.0) >= min_off_ or not b_pool:
                    break
                _screen(n_pre)
                _offsets()
            log.info("%s: pre-screen blade contact offsets at C per B: %s", self.gp.alias, [round(pushable[i], 3) for i in range(len(pre))])
        min_off = float(self.num("push_min_contact_m", 0.10))
        idx_of = {id(t): i for i, t in enumerate(pre)}
        push_rank = (lambda t: 0 if pushable.get(idx_of[id(t)], 1.0) >= min_off else 1) if pushable else (lambda t: 0)  # noqa: E731

        def _ahead(m: Stance) -> float:
            f = m.focus[:2] if m.focus is not None else m.base_xy
            return float((th.as_tensor(f, dtype=th.float32) - m.base_xy) @ m.forward)

        def _margin(t) -> float:
            m = t[1]
            f = m.focus if m.focus is not None else th.tensor([float(m.base_xy[0]), float(m.base_xy[1]), z_init])
            # owner 2026-09-27 (first-demo v2.2): by whichever arm whose band holds the target at this B reaches better
            arms = [a for a in (m.extras.get("focus_arms") or [offered[0]]) if a in t[3]] or [offered[0]]
            return self.best_reach_margin(m, [float(f[0]), float(f[1]), z_init + 0.5 * float(self.place["height"])], arms)[0]
        order_mode = str(self.gp.gen("b_trunk_order", self.gp.gen("b_prescreen_order", "reach")))
        if order_mode == "reach":
            # W2 (owner 2026-09-20 22:15): the B whose trunk leaves the most reach margin to the target first (5 cm buckets, capped at
            # 0.30 m so a nearly-folded arm does not outrank a comfortable one), then a usable F_init, the compact grasp, the larger F_init
            margins = {id(t): min(round(_margin(t) / 0.05) * 0.05, 0.30) for t in pre}
            pre.sort(key=lambda t: (push_rank(t), -margins[id(t)], 0 if t[0] >= min_f else 1, round(_ahead(t[1]), 1), -t[0]))
            log.info("%s: pre-screen reach margins per B (trunk -> margin): %s", self.gp.alias,
                     [(round(float(t[1].translate), 2), round(float(t[1].tilt), 2), round(_margin(t), 3)) for t in pre])
        elif order_mode == "compact":
            pre.sort(key=lambda t: (push_rank(t), 0 if t[0] >= min_f else 1, round(_ahead(t[1]), 1), -t[0]))
        else:
            pre.sort(key=lambda t: (push_rank(t), -t[0]))
        self.template_facts.setdefault("F_init_prescreen", []).append([t[0] for t in pre])
        if len(offered) > 1:
            self.template_facts.setdefault("F_init_prescreen_arms", []).append([{a: len(F) for a, (_, F) in t[3].items()} for t in pre])
        log.info("%s: F_init pre-screen over %d B candidates (%d yaws): %s%s", self.gp.alias, len(pre), len(yaws), [t[0] for t in pre],
                 "" if len(offered) < 2 else f" (per arm {[{a: len(F) for a, (_, F) in t[3].items()} for t in pre]})")
        n_couple = int(self.num("b_candidates_max", 6))
        for i_b, (n_f, manip, F_init, by_arm) in enumerate(pre[:n_couple]):
            self._check_deadline(f"B candidate {i_b + 1} of {min(n_couple, len(pre))}")
            self.sidecar.transitions.clear()
            self.sidecar.paths.clear()
            try:
                if not F_init:
                    raise Reject("grasp_unreachable", "empty feasible set F_init at B's focus cell")
                # A -> B (§3.7 / §3.8; owner 2026-09-19 21:56 (ii)); floor class: the floor mesh out of the sweep / drive-in checks
                # like stance B itself
                with self._floor_relaxed():
                    leg_ab = self.record_leg_a_b(init, manip)
                focus_xy = self.surface["c_t"] if kind == "floor" else manip.focus[:2]
                d, leg_ab = self._couple_segment(rng, init, manip, by_arm, focus_xy, z_init, leg_ab)
                fallback = getattr(self, "init_fallback", None)
                d.summary.update({"relation": self.relation, "init_surface": {"class": kind, "name": None if kind == "floor" else self.surface["obj"].name,
                                                                              **({"fallback": fallback} if fallback is not None else {})},
                                  "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(manip),
                                  "legs": {"A_B": leg_ab, "B_C": getattr(self, "_leg_bc", None)}})
                return d
            except Reject as e:
                if e.reason == "time_budget":
                    raise  # the attempt is over, not this B candidate
                errors.append(f"B candidate: {e.reason}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.stances.pop("placement", None)
                self.sidecar.stances.pop("push", None)
                self.sidecar.grasps.clear()
                self.carry_arm_q.clear()
                self.carry_end_q.clear()
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def _b_stances(self, manip: Stance, offered: Sequence[str]) -> Dict[str, Stance]:
        """Stance B per offered arm (owner 2026-09-27, first-demo v2.2): the default arm's - ``manip`` itself when the B search solved it
        for that arm - always, the other arm's where its band holds the target (``extras["focus_arms"]``); each with the other arm idle
        in the posture re-derived for it (``tabletop.stance_for_arm``: A6 untucked / ready / tucked; an arm whose idle partner collides
        in every posture is not offered)."""
        arm0 = offered[0]
        out: Dict[str, Stance] = {}
        if manip.working_arms in ([], [arm0]):
            manip.arms = [arm0]
            out[arm0] = manip
        else:
            st = TT.stance_for_arm(self, manip, arm0)
            if st is not None:
                out[arm0] = st
        focus_arms = manip.extras.get("focus_arms") or [arm0]
        for a in offered[1:]:
            if a in focus_arms:
                st = TT.stance_for_arm(self, manip, a)
                if st is not None:
                    out[a] = st
        return out

    @staticmethod
    def _segment_order(manip: Stance, by_arm: Dict[str, Tuple[Stance, List[Any]]]) -> List[str]:
        """The arms with pairs at B in the order their couplings run: those whose band holds the target (``focus_arms``) first, the
        default arm first among them (a B shortlisted by the other arm's band alone starts with that arm)."""
        arms = [a for a, (_, F) in by_arm.items() if F]
        focus = manip.extras.get("focus_arms") or arms[:1]
        return [a for a in arms if a in focus] + [a for a in arms if a not in focus]

    def _same_arm_q(self, a: Stance, b: Stance) -> bool:
        """Both arms at the same joint values in the two stances (a leg checked at one holds at the other)."""
        ra, la, _ = self.ikc.arm_vectors(a)
        rb, lb, _ = self.ikc.arm_vectors(b)
        return bool(th.equal(ra, rb) and th.equal(la, lb))

    def _clear_segment_records(self, manip_name: str) -> None:
        """What a failed coupling recorded past B: the C / push stances, the grasp, the push contact, the carry configurations and
        every path / transition but the A -> B leg."""
        self.sidecar.stances.pop("placement", None)
        self.sidecar.stances.pop("push", None)
        self.sidecar.contacts.pop("push", None)
        self.sidecar.grasps.clear()
        self.carry_arm_q.clear()
        self.carry_end_q.clear()
        self.sidecar.paths = [p for p in self.sidecar.paths if p.to == manip_name]
        self.sidecar.transitions = [t for t in self.sidecar.transitions if t.to == manip_name]

    def _couple_segment(self, rng, init: Stance, manip: Stance, by_arm: Dict[str, Tuple[Stance, List[Any]]], focus_xy, z_init: float,
                        leg_ab: Dict[str, Any]) -> Tuple[Draft, Dict[str, Any]]:
        """The grasp -> carry -> place -> release segment from stance B (owner 2026-09-27, first-demo v2.2): one coupling (:meth:`_couple`)
        per arm with pairs in its F_init (:meth:`_segment_order`). ``manip`` (the record; its A -> B leg recorded) takes the coupling
        arm's working arm and idle posture, the leg recorded again when that changes the arms' joint values. A pair whose grasp at B
        another arm won waits for that arm's coupling; the pairs a failed coupling won are regrasped by the next arm (the segment
        fallback). Returns the draft and the A -> B leg facts."""
        arms = self._segment_order(manip, by_arm)
        # ``manip`` takes each coupling's arm in place: the segment keeps a copy of the stance it was for its own arm
        seg = SegmentArms({a: st.with_trunk(st.translate, st.tilt, st.name) if st is manip else st for a, (st, _) in by_arm.items()},
                          {a: list(F) for a, (_, F) in by_arm.items()}, {a: (self.surface.get("arm_bands") or {}).get(a) for a in by_arm},
                          2.0 * math.pi / max(1, len(self.yaws())))
        errors: List[str] = []
        last: Optional[Reject] = None
        for k, arm in enumerate(arms):
            st_a = by_arm[arm][0]
            try:
                if k > 0:
                    self._clear_segment_records(manip.name)
                if manip.working_arms != [arm] or manip.idle_posture != st_a.idle_posture:
                    same = self._same_arm_q(manip, st_a)
                    TT.adopt_arm(manip, st_a)
                    if not same:
                        self._drop_leg_records(manip.name)
                        seg.leg += 1
                        with self._floor_relaxed():
                            leg_ab = self.record_leg_a_b(init, manip)
                self.seg_arm = arm
                c_pool = self.placement_candidates(focus_xy)
                if not c_pool:
                    raise Reject("no_placement_stance", "no stance C survivor for this init pose")
                return self._couple(rng, init, manip, c_pool, by_arm[arm][1], focus_xy, z_init, seg), leg_ab
            except Reject as e:
                if e.reason == "time_budget" or len(arms) == 1:
                    raise
                seg.done.append(arm)
                seg.why.setdefault(arm, f"{e.reason}: {str(e)[:300]}")
                errors.append(f"{arm} arm: {e.reason}: {e}")
                last = e
                log.info("%s: B (%.2f, %.2f): the %s arm's segment failed (%s)%s", self.gp.alias, manip.translate, manip.tilt, arm, e.reason,
                         f"; the {arms[k + 1]} arm's coupling next" if k + 1 < len(arms) else "")
        raise Reject(last.reason if last is not None else "grasp_unreachable", " | ".join(errors) or "no arm has pairs in F_init at B")

    @staticmethod
    def _b_options(pick, g, eef: G.Pose, pre: G.Pose) -> Tuple[List[Tuple[Any, G.Pose, G.Pose, bool]], Optional[Dict[str, Any]]]:
        """The finger assignments a grasp at B may be recorded with, the pick's first (``(grasp row or twin, eef, pre-grasp, is twin)``),
        and its ``twin_choice``."""
        if pick is None:
            return [(g, eef, pre, False)], None
        opts = [(pick.grasp, pick.eef, pick.pre, pick.twin)]
        if pick.other is not None:
            opts.append((pick.other[0], pick.other[2], pick.other[3], not pick.twin))
        return opts, pick.facts

    def _regrasp_at_b(self, g, t_pose: G.Pose, stance: Stance, arm: str, z_init: float, twins: bool, bands: Dict[str, Any]):
        """The segment fallback's grasp at B (owner 2026-09-27, first-demo v2.2; ``tabletop.segment_fallback_grasp``'s rule, keeping the
        other finger assignment for the place at C): the library row ``g`` at the target pose the first grasp was drawn at, with ``arm``
        only (both turns by the orientation rule when the twins are on); :meth:`_b_options` or None."""
        TT._count(self, "segment_fallbacks")
        eef = IK.grasp_pose(t_pose, g.pose)
        pre = IK.pre_pose(eef)
        kw = {"ignore_for_last": self.grasp_ignore_objects(), "scale_key": self.choice.models["target"].scale, "cache": False}
        res = self.ikc.exact([pre, eef], stance, arm, **kw)
        pick = None
        if twins and getattr(g, "name", None):
            pick = TT.pick_twin(self, stance, arm, g, g.pose, eef, pre, res, kw, support_z=float(z_init), where="B, segment fallback", arms=[arm], stances={arm: stance})
            res = pick.res
            self._count_mirrored_ik(pick, bands)
        if not res.ok:
            log.info("%s: segment fallback: the %s arm's regrasp of %s at B fails (%s)", self.gp.alias, arm, getattr(g, "name", g), str(res.reason)[:200])
            return None
        TT._count(self, f"segment_fallback_taken.{arm}")
        return self._b_options(pick, g, eef, pre)

    def _count_mirrored_ik(self, pick, bands: Dict[str, Any]) -> None:
        """``arm_band_mirrored.ik_tried`` / ``.ik_failed`` for a grasp's solved options of an arm whose band is a mirror (as
        ``tabletop.phase3_target`` counts them)."""
        for o in pick.facts["orient"]["options"]:
            b = bands.get(o.get("arm"))
            if o.get("source") == "ik" and b is not None and b.meta.get("mirrored_from"):
                TT._count(self, "arm_band_mirrored.ik_tried")
                if not o.get("ok"):
                    TT._count(self, "arm_band_mirrored.ik_failed")

    def _couple(self, rng, init: Stance, manip: Stance, c_pool: List[Stance], F_init, focus_xy, z_init: float, seg: Optional[SegmentArms] = None) -> Draft:
        """B ↔ C coupling: for C survivors in random order, F_goal at the goal cell for the init yaws, the draw from
        F_init ∩ F_goal, the target pose at B's focus, the exact IK at B and at C (attached), the B → C path and
        transition; the draft and the sidecar records. For the segment arm (``self.seg_arm``; owner 2026-09-27, first-demo v2.2):
        ``F_init`` / ``c_pool`` are its own, the grasp at B is ``tabletop.pick_twin`` over the arms of ``seg`` whose F_init holds the
        pair (shared with the other arm's coupling: a pair the other arm won is left to its coupling, one it won and lost later is
        regrasped here - :meth:`SegmentArms.role`)."""
        tname, pname = self.tname, self.pname
        partner, target = self.partner, self.target
        arm = self.seg_arm
        arm0 = TT.default_arm(self)
        offered = TT.offered_arms(self)
        # §5.6: B → C ≤ max_partner_geodesic + extra. The geodesic rule measures near side to near side; the recorded stances
        # sit at particular edges (the bar's short end → the sofa's front), so the stance-to-stance drive is longer — extra
        # 3 m (sofa-bar 17:04: every C survivor failed with path_too_long at 4 m; the B-C IK never got its turn)
        geo = self.num("max_partner_geodesic_m", 3.0)
        # the customs pin path_max_extra_m = 1.0 (D1 generation_defaults): the stance-to-stance drive is bounded here by
        # max(geodesic + extra, 2 × geodesic) — the base rounds the support / partner between B and C (D1: raise the default)
        max_len = max(geo + self.num("path_max_extra_m", 3.0), 2.0 * geo)
        order = list(rng.permutation(len(c_pool)))
        n_try = int(self.num("c_candidates_max", 12))  # C survivors tried per B candidate (the per-attempt budget bounds the total)
        if self.push_under:
            # side stances first (the frontal ones put the lane out of the floor reach — see push_under_candidates), the
            # deepest trunk first within a kind (B3: the floor is reached at translate 0.95 / tilt −1.2 only); every side
            # survivor is tried before the frontal ones
            side = [k for k in order if c_pool[k].extras.get("stance_kind") == "side"]
            front = [k for k in order if c_pool[k].extras.get("stance_kind") != "side"]
            # frontal stances: the push goal must stay inside the floor reach (B3: 0.24-0.64 m ahead at the deep trunks) - the goal
            # lies drop_ahead + push distance (~0.3 m) ahead, so the drop cell should sit ~0.3 m ahead; ordering by the LARGEST
            # drop_ahead first (the bed heuristic: keep the deep trunk clear of the frame) put every chain at 0.7-0.8 m ahead and
            # 260 chain searches failed the exact IK in 0919-233031
            # (2026-09-20 01:40 correction: with the 0.30 m preference the 0.65-0.70 m stand-offs came first and every chain failed
            # its PRE-pose - the blade start sits on the base rim there; the two instances that were written (0919-211642 /
            # 221909) used d 0.80 with the goal ~0.8 m ahead, which the deep crouch does reach. Default = the old order, largest
            # drop_ahead first; ``generation.push_under_drop_ahead_pref_m`` switches to the nearest-to-preference order.)
            ahead_pref = self.push_drop_ahead_pref()
            # the lane to the robot's RIGHT first (frontal_lane_offset_m 0.25): B3's under-edge probe reaches the floor strip at
            # lateral -0.40..-0.16 only, and the one chain that ever solved (0919-211642) ran at lateral -0.25; the centred lane
            # (offset 0) tied with it in the order and 482 centred chains failed their pre-pose in 0920-040824
            lane = lambda k: 0 if abs(float(c_pool[k].extras.get("lane_offset_m", 0.0))) > 1e-6 else 1  # noqa: E731
            erank = lambda k: int(c_pool[k].extras.get("edge_rank", 0))  # noqa: E731  (the nearest allowed partner edge first)
            depth = lambda k: (erank(k), -float(c_pool[k].translate), float(c_pool[k].tilt), -float(c_pool[k].extras.get("drop_ahead_m", 0.0)), lane(k))  # noqa: E731
            reach = lambda k: (erank(k), lane(k), abs(float(c_pool[k].extras.get("drop_ahead_m", 0.0)) - float(ahead_pref)), -float(c_pool[k].translate), float(c_pool[k].tilt))  # noqa: E731
            if str(self.gp.gen("c_order", "travel")) == "travel":
                # W3 (owner 2026-09-20 22:15): the survivors of the feasibility filters (collision-free crouch, free drop lane, goal cell
                # under the top) are ordered by the B -> C base travel (shortest path on the eroded map, 0.25 m buckets; the straight
                # distance as fallback), deterministic given B: the same B gave C at the far end (4.2 m), the middle (3.2 m) and the near
                # end (2.6 m) of the table with the seeded permutation deciding (0920-143722 / 154503 / 154504). Within a bucket the
                # right lane, the stand-off nearest ``push_under_drop_ahead_pref_m`` and the deepest trunk keep their order.
                trav_cache: Dict[Any, float] = {}
                trav = lambda k: round(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) / 0.25)  # noqa: E731
                pref = float(ahead_pref) if ahead_pref is not None else 0.6
                key_t = lambda k: (trav(k), lane(k), abs(float(c_pool[k].extras.get("drop_ahead_m", 0.0)) - pref), -float(c_pool[k].translate), float(c_pool[k].tilt))  # noqa: E731
                order = sorted(side + front, key=key_t)
                if order:
                    k0 = order[0]
                    log.info("%s: C order by travel from B %s: nearest %s (%.2f m, edge %s, d %.2f, λ %+.2f), farthest %.2f m over %d survivors", self.gp.alias,
                             [round(float(v), 2) for v in manip.base_xy.tolist()], [round(float(v), 2) for v in c_pool[k0].base_xy.tolist()],
                             NAV.travel_length(self.kit, manip.base_xy, c_pool[k0].base_xy, trav_cache), c_pool[k0].extras.get("edge"), c_pool[k0].stand_off or 0.0,
                             c_pool[k0].lateral, max(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) for k in order), len(order))
            else:
                order = sorted(side, key=depth) + sorted(front, key=reach if ahead_pref is not None else depth)
            n_try = max(int(self.num("c_tries", 36)), len(side))  # the chain-first screen makes a C try cheap (a few IKs) until one has a chain
        elif str(self.gp.gen("c_order", "travel")) == "travel":
            # W3: ontop / place variants - the nearest survivor by base travel first (edge rank as the tie-break), deterministic given B
            trav_cache = {}
            order.sort(key=lambda k: (round(NAV.travel_length(self.kit, manip.base_xy, c_pool[k].base_xy, trav_cache) / 0.25), int(c_pool[k].extras.get("edge_rank", 0))))
        else:
            order.sort(key=lambda k: int(c_pool[k].extras.get("edge_rank", 0)))  # ontop / place variants: nearest edge first, random within
        yaw_step = 2.0 * math.pi / max(1, len(self.yaws()))  # the screening grid (yaw buckets of the caches / the intersection)
        chain_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}  # (C index, yaw bucket) -> push chain (push-under: the chain depends on the yaw only)
        seg = seg or SegmentArms({arm: manip}, {arm: list(F_init)}, yaw_step=yaw_step)
        st_of = {**seg.stances, arm: manip}
        # (yaw bucket, grasp) -> the grasp at B: yaw, target pose, the arm that won it, the arms viable there, per arm [(grasp row or twin,
        # grasp eef, pre-grasp, is twin), ...] preferred first + twin_choice; None = infeasible for every arm
        b_ik_cache = seg.b_ik
        deferred: set = set()  # pairs whose grasp at B an arm with its coupling still to come won
        twins_b = TT.grasp_twins_on(self)
        errors = []
        carry_margin = float(self.gp.gen("carry_collision_margin_m", 0.04))
        carry_failed_bases: set = set()  # base poses whose carry leg failed: the trunk variants at the same pose fail the same way
        for i_c, kc in enumerate(order[:n_try]):
            self._check_deadline(f"C candidate {i_c + 1} of {min(n_try, len(order))} (F_goal feasible set) at B ({manip.translate}, {manip.tilt:.2f})")
            F_live = [yg for yg in F_init if pair_key(yg[0], yg[1], yaw_step) not in deferred] if deferred else F_init
            if not F_live:
                errors.append(f"every pair of the {arm} arm's F_init ({len(F_init)}) at B went to another arm's grasp")
                break
            place = c_pool[int(kc)].with_trunk(c_pool[int(kc)].translate, c_pool[int(kc)].tilt, "placement")
            place.focus, place.arms = c_pool[int(kc)].focus, [arm]
            place.extras = dict(c_pool[int(kc)].extras)
            gc = place.extras["goal_cell"]
            push = self.push_under
            base_key = tuple(round(float(v), 3) for v in place.base_pose2d)
            if base_key in carry_failed_bases:
                errors.append(f"C (d {place.stand_off or 0.0:.2f} λ {place.lateral:+.2f}): skipped - the carry leg failed at this base pose")
                continue
            # 2026-09-21 02:20: a C whose base pose has no room for the carried sweep's margin is skipped before the F_goal screening
            # (the short end next to B stands 0.38 m from the walls: every arrival sweep failed there at 2.5 / 4 cm while the 1 cm
            # stance checks pass, 0921-012344 / 014859 - 6 candidates x 3-4 min per B). Checked at the first travel trunk with the
            # untucked arms: the wall proximity comes from the body, not the carried object.
            if carry_margin > 0.011:
                tt0 = self.travel_trunks(init, manip)[0] if self.travel_trunks(init, manip) else (0.25, 0.0)
                probe = place.with_trunk(float(tt0[0]), float(tt0[1]), "placement")
                # the BODY's room, not the arms': with the untucked arms the hanging hand hovered over the countertop within 4 cm and
                # every ontop C was skipped (0921-031852, shampoo_dispenser-countertop-floor: 6 attempts, no instance) - the compact
                # ``ready`` posture keeps the hands at the chest (collision-free up to translate 1.25 on the JoyLo curve)
                with self._floor_relaxed():
                    if not self.ikc.stance_collision_free(probe, margin=carry_margin, arms="ready"):
                        why_m = ""
                        try:
                            why_m = " - " + self.ikc._collision_blame(self.ikc.q_locked(probe, arms="ready"), [self.obj("floor")], carry_margin)
                        except Exception:  # noqa: BLE001
                            pass
                        errors.append(f"C (d {place.stand_off or 0.0:.2f} λ {place.lateral:+.2f}): no room for the carried sweep at margin {carry_margin:.3f}{why_m}")
                        log.info("%s: C skipped (d %.2f λ %+.2f edge %s): the body has no room for the carried sweep's margin %.3f at the travel trunk%s", self.gp.alias,
                                 place.stand_off or 0.0, place.lateral, place.extras.get("edge"), carry_margin, why_m)
                        carry_failed_bases.add(base_key)
                        continue
            cd = place.extras.get("drop_cell") if push else None
            # push-under = DROP FIRST, THEN CROUCH (owner 2026-09-20 08:30): the object cannot be carried into the deep push crouch
            # (the hand frozen at any carry pose sweeps into the partner / the floor / the base, and no compact carry pose keeps the
            # grasp orientation within the wrist limits - 0920-043725 .. 075127), so the release happens from a drop trunk at the
            # same base pose ("placement" = the drop stance, drop height ``drop_height_m``), and the deep crouch ("push" stance) is
            # entered with empty hands by a trunk-only transition before the floor push. ``generation.drop_trunks`` (default
            # ``PUSH_UNDER_DROP_TRUNKS``: (0.95, -1.2) first, then moderate ones) lists the trunks tried for the drop stance (the
            # deepest survivors of the C search stay the push stance).
            push_st = None
            if push and str(self.gp.gen("push_under_drop_first", "on")) != "off":
                push_st = place.with_trunk(place.translate, place.tilt, "push")
                push_st.focus, push_st.arms, push_st.extras = place.focus, [arm], dict(place.extras)
                drop_st = None
                for tr, ti in [tuple(float(v) for v in t) for t in (self.gp.gen("drop_trunks") or PUSH_UNDER_DROP_TRUNKS)]:
                    cand = place.with_trunk(tr, ti, "placement")
                    cand.focus, cand.arms, cand.extras = place.focus, [arm], dict(place.extras)
                    with self._floor_relaxed():
                        if bool(P0.body_free(self, [cand])[0]):
                            drop_st = cand
                            break
                if drop_st is None:
                    errors.append(f"C (d {place.stand_off:.2f}): no drop trunk is collision-free at the push stance's base pose")
                    continue
                place = drop_st
            if self.relation == "under" and not push:
                cl = self.clearance_at(gc)
                if cl is not None and not under_admissible(cl, self.place["height"], self.num("under_clearance_extra_m", 0.17)):
                    errors.append(f"C: under clearance {cl:.2f} m at the goal cell < height + 0.17")
                    continue
            yaws = sorted({float(y) for y, _ in F_live})
            d_yaw = 0.0 if str(self.gp.gen("carry_yaw", "relative")) == "world" else float(place.yaw - manip.yaw)
            if push:
                # chain FIRST (2026-09-20 19:20): the floor push's IK depends on where this C stands along the edge (0920-184335: 505 chains
                # at 12 C candidates, every one unreachable, while 0920-143722 / 154503 pushed from other laterals of the same edge) and
                # on the object's yaw through the blade's contact offset - so the chain is solved for the two yaws with the LARGEST
                # contact offset (the toe-first yaws of a sandal: the blade goal nearest the base) before the 60 s F_goal screening,
                # and a C without a chain there is skipped
                d_push = th.as_tensor(place.extras["push_dir"], dtype=th.float32)
                z_lo = self.push_height(30.0)[1] - self.floor_z
                offs = {}
                for y in yaws:
                    o = self.contact_offset_along(G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), d_push, z_lo, z_lo + self.BLADE_SPAN_M)
                    offs[y] = -1.0 if o is None else o
                pst0 = push_st if push_st is not None else place
                best = sorted(yaws, key=lambda y: self.push_yaw_rank(pst0, G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), offs[y]))[:2]
                found = False
                for y in best:
                    ck = (int(kc), int(round(y / yaw_step)))
                    if ck not in chain_cache:
                        chain_cache[ck] = self._push_under_chain(push_st if push_st is not None else place, G.qmul(G.quat_from_yaw(y + d_yaw), self.rest), cd, gc, place.extras["push_dir"])
                    if chain_cache[ck] is not None:
                        found = True
                        ch = chain_cache[ck]
                        pst = push_st if push_st is not None else place
                        log.info("%s: push chain SOLVED at C (d %.2f λ %+.2f push trunk (%.2f, %.2f), yaw %.2f offset %.3f m): blade start %s -> goal %s in base, pitch %.0f, tips %.3f m",
                                 self.gp.alias, place.stand_off or 0.0, place.lateral, pst.translate, pst.tilt, y, offs[y],
                                 [round(float(v), 2) for v in P0.to_base_frame(ch["start"][0], pst.base_pose2d, P0.base_z(self)).tolist()],
                                 [round(float(v), 2) for v in P0.to_base_frame(ch["goal"][0], pst.base_pose2d, P0.base_z(self)).tolist()], ch["pitch_deg"], ch["fingertip_z_m"])
                        break
                if not found:
                    errors.append(f"C (d {place.stand_off:.2f} λ {place.lateral:+.2f}): no push chain at the {self.PUSH_YAW_ORDER} yaws ({', '.join(f'{offs[y]:.2f}' for y in best)} m)")
                    continue
            ignore_place = [partner] if self.relation == "ontop" else [self.obj("floor")]
            # the release pose: the goal cell (ontop / under by placement) or the drop cell outside the edge (push-under)
            release_xy = cd if push else gc
            release_pose = (lambda xy, quat: self.drop_pose_at(xy, quat)) if push else (lambda xy, quat: self.goal_pose_at(xy, quat))
            # A carried object keeps its yaw RELATIVE TO THE ROBOT (no re-grasp), so at C its world yaw is the init yaw plus
            # the base-yaw change B -> C: F_goal is screened at yaw + d_yaw and intersected with F_init by the init yaw
            # (``generation.carry_yaw = "world"`` restores the spec §5.6 wording "goal yaw = init world yaw", which holds
            # only when B and C share a base yaw - countertop-floor 0916-231451: F_init 3-21 / F_goal 0-5 / F empty on
            # every candidate). The goal orientation written to the TRO is the rotated one.
            F_goal = self.ikc.feasible_sets([release_xy], [y + d_yaw for y in yaws], self.grasps,
                                            lambda xy, yaw: release_pose(xy, G.qmul(G.quat_from_yaw(yaw), self.rest)), place, arm, ignore_place).get(0, [])
            F_goal = [(y - d_yaw, g) for y, g in F_goal]
            F = intersect_feasible(F_live, F_goal, yaw_step=yaw_step)
            if push:
                log.info("%s: push-under C try %s d %.2f λ %+.2f trunk (%.2f, %.1f): F_init %d, F_goal %d, F %d at drop %s", self.gp.alias, place.extras.get("stance_kind"),
                         place.stand_off or 0.0, place.lateral, place.translate, place.tilt, len(F_live), len(F_goal), len(F), [round(float(v), 2) for v in release_xy])
            if not F:
                if push and not F_goal and not getattr(self, "_drop_debugged", False):
                    self._drop_debugged = True
                    log.info("%s: drop IK debug — %s", self.gp.alias, self.debug_drop_ik(place, release_xy, yaws))
                errors.append(f"C: F_init ({len(F_live)}) ∩ F_goal ({len(F_goal)}) empty at {'drop' if push else 'goal'} cell {[round(v, 2) for v in release_xy]}")
                continue
            fails: Dict[str, int] = {}
            # pairs of F tried per C: the push-under chain search (4 pitches x 2 wrist turns x exact IK) is the expensive leg, so the
            # push mode keeps the earlier 4 draws (0919-233031: 8 draws x 12 C candidates spent the 900 s budget on attempts 0 and 2)
            draw_order = list(rng.permutation(len(F)))
            if push:
                # the blade sits half an extent behind the object centre: the LONG axis along the push puts the blade goal ~7 cm
                # closer to the base (sandal: 0.72 vs 0.78-0.80 m ahead - the chain solved only at the long-axis yaws, 0919-211642;
                # every short-axis chain of 0920-052659 had no IK at its goal) - longest extent first, random within a bucket
                # (2026-09-20 19:20: ordered by the blade's CONTACT offset at its height band - a sandal toe-first before its strap)
                rank = {i: self.push_yaw_rank(pst0, G.qmul(G.quat_from_yaw(float(F[i][0]) + d_yaw), self.rest), offs.get(float(F[i][0]), -1.0)) for i in range(len(F))}
                draw_order.sort(key=lambda i: round(rank[i], 2))
            for i in draw_order[:int(self.num("f_draws_max", 8))]:  # failed B pairs are cached and skipped, so 8 draws cost little
                self._check_deadline(f"F draw at C candidate {i_c + 1} (exact IK at B / C{', push chain' if push else ''})")
                yaw0, g = F[int(i)]
                yaw = yaw0 + float(rng.uniform(-TARGET_JITTER, TARGET_JITTER))
                if push:
                    # the push chain depends on the object's yaw (its half extent along the push), not on the grasp: solve it once per
                    # (C, yaw bucket) BEFORE the grasp IKs and skip the draw when this stance cannot push at that yaw
                    ck = (int(kc), int(round(yaw0 / yaw_step)))
                    if ck not in chain_cache:
                        q_chain = G.qmul(G.quat_from_yaw(yaw0 + d_yaw), self.rest)
                        chain_cache[ck] = self._push_under_chain(push_st if push_st is not None else place, q_chain, cd, gc, place.extras["push_dir"])
                    if chain_cache[ck] is None:
                        fails["C: push chain unreachable at this yaw"] = fails.get("C: push chain unreachable at this yaw", 0) + 1
                        continue
                # the grasp side (target pose at B's focus, pre-grasp + grasp exact IK from B) does not depend on C: solved once per
                # (yaw bucket, grasp) for this B and reused across the C candidates (a failed pair is skipped everywhere)
                bk = (int(round(yaw0 / yaw_step)), getattr(g, "name", str(g)))
                if bk not in b_ik_cache:
                    # the exact IK at B is tried at the screened pose first (the focus cell, the grid yaw: what F_init passed) and then
                    # at up to two jittered poses; a pair is cached as infeasible only when all fail, with the reasons logged
                    # (0920-235717/8/36: 7-8 of 8 toe-first draws failed here with one jittered pose each, no reason recorded).
                    # Owner 2026-09-27 (first-demo v2.2): the arms whose F_init holds the pair compete for it (the default arm first,
                    # each at its own stance B) but an arm whose segment already failed at this B; the pick is shared by the couplings
                    held = seg.arms_of(yaw0, g) or [arm]
                    arms_p = [a for a in held if a not in seg.done] or [arm]
                    a0 = arms_p[0]
                    res, t_pose, eef, pre, pick = None, None, None, None, None
                    why_b: List[str] = []
                    for k_try in range(int(self.num("b_ik_tries", 3))):
                        yaw_k = yaw0 if k_try == 0 else yaw0 + float(rng.uniform(-TARGET_JITTER, TARGET_JITTER))
                        t_pose = None
                        for _ in range(5):
                            if self.surface["kind"] == "floor" or k_try == 0:
                                xy = focus_xy
                            else:
                                r, a = float(rng.uniform(0, FOCUS_RADIUS)), float(rng.uniform(0, 2 * math.pi))
                                xy = focus_xy + th.tensor([r * math.cos(a), r * math.sin(a)])
                            if self.surface["kind"] != "floor" and not self.surface["top"].polygon_supported(self.world.footprint(self.hull, self.pose_on(xy, yaw_k, z_init)), margin=SUPPORT_MARGIN):
                                continue
                            t_pose = self.pose_on(xy, yaw_k, z_init)
                            break
                        if t_pose is None:
                            why_b.append("target footprint off the init surface")
                            continue
                        eef = IK.grasp_pose(t_pose, g.pose)
                        pre = IK.pre_pose(eef)
                        res = self.ikc.exact([pre, eef], st_of[a0], a0, ignore_for_last=self.grasp_ignore_objects(), scale_key=self.choice.models["target"].scale, cache=False)
                        if (twins_b or len(arms_p) > 1) and getattr(g, "name", None):
                            pick = TT.pick_twin(self, st_of[a0], a0, g, g.pose, eef, pre, res,
                                                {"ignore_for_last": self.grasp_ignore_objects(), "scale_key": self.choice.models["target"].scale, "cache": False},
                                                support_z=float(z_init), where="B", arms=arms_p, stances=st_of, twins=twins_b)
                            res = pick.res
                            self._count_mirrored_ik(pick, seg.bands)
                        if res.ok:
                            yaw = yaw_k
                            break
                        why_b.append(f"try {k_try} yaw {yaw_k:.2f}: {res.reason}")
                    if res is None or not res.ok:
                        b_ik_cache[bk] = None
                        fails["B: [pre-grasp, grasp] exact IK"] = fails.get("B: [pre-grasp, grasp] exact IK", 0) + 1
                        log.info("%s: B exact IK fails for yaw bucket %.2f / grasp %s at B (%.2f, %.2f): %s", self.gp.alias, yaw0, getattr(g, "name", g), manip.translate, manip.tilt,
                                 " | ".join(why_b)[:600])
                        continue
                    arm_b = pick.arm if pick is not None else a0
                    b_ik_cache[bk] = {"yaw": yaw, "t_pose": t_pose, "arm": arm_b, "viable": O.viable_arms(pick.facts["orient"]) if pick is not None else [arm_b],
                                      "opts": {arm_b: self._b_options(pick, g, eef, pre)}, "after": [a for a in held if a in seg.done]}
                hit = b_ik_cache[bk]
                if hit is None:
                    fails["B: [pre-grasp, grasp] exact IK (cached)"] = fails.get("B: [pre-grasp, grasp] exact IK (cached)", 0) + 1
                    continue
                role = seg.role(hit["arm"], hit["viable"], arm)
                fallback = None
                if role == "defer":
                    deferred.add(bk)
                    fails[f"B: the {hit['arm']} arm's grasp (its coupling next)"] = fails.get(f"B: the {hit['arm']} arm's grasp (its coupling next)", 0) + 1
                    continue
                if role == "fallback":
                    # the segment fallback (owner 2026-09-27): the arm that won the grasp failed a later contact; this arm, viable at the
                    # grasp, regrasps the library row at the same target pose (once per pair)
                    if arm not in hit["opts"]:
                        hit["opts"][arm] = self._regrasp_at_b(g, hit["t_pose"], manip, arm, z_init, twins_b, seg.bands)
                    if hit["opts"][arm] is None:
                        role = "none"
                    else:
                        won_opts, won_facts = hit["opts"].get(hit["arm"]) or ([(None,)], None)
                        fallback = {"from_arm": hit["arm"], "from_grasp": getattr(won_opts[0][0], "name", None), "why": (seg.failed.get(bk) or seg.why.get(hit["arm"], ""))[:300],
                                    "arm_choice": (won_facts or {}).get("arm_choice")}
                elif hit.get("after"):
                    # drawn after the segment of an arm that held the pair failed at this B: that arm is not offered again
                    fallback = {"from_arm": hit["after"][0], "from_grasp": None, "why": seg.why.get(hit["after"][0], "")[:300]}
                if role == "none":
                    fails[f"B: no {arm}-arm grasp (the {hit['arm']} arm's)"] = fails.get(f"B: no {arm}-arm grasp (the {hit['arm']} arm's)", 0) + 1
                    continue
                yaw, t_pose = hit["yaw"], hit["t_pose"]
                b_opts, b_facts = hit["opts"][arm]
                lk = (bk, seg.leg)
                if lk not in seg.leg_ok:
                    seg.leg_ok[lk] = self._leg_ab_clear_of_target(init, manip, t_pose)
                    if seg.leg_ok[lk]:
                        # the A -> B drive was swept before the target had a pose (parked): a draw on the drive's corridor is dropped
                        fails["B: the A -> B drive passes the placed target"] = fails.get("B: the A -> B drive passes the placed target", 0) + 1
                        log.info("%s: %s", self.gp.alias, seg.leg_ok[lk])
                        continue
                elif seg.leg_ok[lk]:
                    fails["B: [pre-grasp, grasp] exact IK (cached)"] = fails.get("B: [pre-grasp, grasp] exact IK (cached)", 0) + 1
                    continue
                goal_quat = G.qmul(G.quat_from_yaw(d_yaw), t_pose[1]) if abs(d_yaw) > 1e-9 else t_pose[1]
                goal = release_pose(release_xy, goal_quat)
                # the recorded finger assignment first; the other one (its B chain passed as well) only when the first fails here
                c_fails: List[str] = []
                for j_opt, (g, eef, pre, flip) in enumerate(b_opts):
                    place_eef = IK.grasp_pose(goal, g.pose)
                    pre_place = IK.pre_pose(place_eef)
                    # the place / drop IK is seeded with the LIFT configuration (the arm the object is carried with, and the branch the
                    # carry pose's straight line stays in) and must land in the same branch: the oracle reaches the pre-place from the
                    # carry with a bounded joint swing (0920-135625: the recorded witness lay 2.09 rad away, 'plan_failed' at the drop)
                    q_lift_seed = self.lift_arm_q(manip, eef, float(self.num("carry_height_m", 0.12)), target, g.pose)
                    q_seed = None
                    if q_lift_seed is not None:
                        q_seed = self.ikc.q_locked(place).clone()
                        q_seed[self.kit.robot.arm_control_idx[arm]] = q_lift_seed
                    with attached(self.kit, arm, target, eef_in_object=g.pose):
                        r2 = self.ikc.exact([pre_place, place_eef], place, arm, ignore_for_last=ignore_place, cache=False, q_seed=q_seed)
                    if not r2.ok:
                        c_fails.append(f"C: [pre-place, {'drop' if push else 'place'}] exact IK (attached)")
                        continue
                    if q_lift_seed is not None and r2.q is not None:
                        # a loose sanity bound only (2026-09-21 01:30): the branch the oracle reaches the pre-place in is decided by the straight
                        # carry leg it tracks from the carry configuration - checked below by ``track_chain`` once the carry leg is recorded
                        # (0921-005514/6/8: the 1.2 rad proxy against the LIFT configuration rejected 1-4 draws per C)
                        swing = float(th.max(th.abs(r2.q[self.kit.robot.arm_control_idx[arm]] - q_lift_seed)))
                        if swing > float(self.num("place_branch_max_rad", 2.6)):
                            c_fails.append("C: place branch differs from the carry (joint swing)")
                            continue
                    break
                if len(c_fails) == len(b_opts):
                    fails[c_fails[0]] = fails.get(c_fails[0], 0) + 1
                    seg.failed.setdefault(bk, c_fails[0])
                    continue
                twin_facts = b_facts
                if b_facts is not None and j_opt > 0:
                    twin_facts = dict(b_facts, chosen="twin" if flip else "row", late_flip=f"the {'row' if flip else 'twin'} failed at C: {c_fails[0]}")
                    log.info("%s: grasp twin [C] %s: the %s's place failed at C (%s), the %s's passes", self.gp.alias, g.name, "row" if flip else "twin", c_fails[0],
                             "twin" if flip else "row")
                if self.relation == "ontop" and not push:
                    # owner 2026-09-26 (shampoo, waves 6-8: 8 of 22 attempts rejected in goal_check 'place eef pose outside the placement image
                    # (5 %)' after 7-13 min of sampling each): the goal check's own frame test, applied to this C candidate's place now, so a C
                    # whose place the placement camera cannot see is skipped within the attempt instead of failing it at the end. The goal
                    # check tests the SETTLED goal, generation.goal_drop_m (1 cm) below the release pose (wave 9 0926-023738 still failed it
                    # with only the release pose tested here), so both poses are tested
                    cam_c = self.camera_pose_at(place)
                    rest = (goal[0] - th.tensor([0.0, 0.0, float(self.num("goal_drop_m", 0.01))]), goal[1])
                    if not (self.ikc.eef_in_frame(place_eef, cam_c) and self.ikc.eef_in_frame(IK.grasp_pose(rest, g.pose), cam_c)):
                        fails["C: place eef outside the placement image"] = fails.get("C: place eef outside the placement image", 0) + 1
                        seg.failed.setdefault(bk, "C: place eef outside the placement image")
                        continue
                chain = None
                if push:  # the straight floor push drop → goal from the same stance (closed gripper, floor + target ignored)
                    chain = self._push_under_chain(push_st if push_st is not None else place, goal_quat, cd, gc, place.extras["push_dir"])
                    if chain is None:
                        errors.append("C: push chain unreachable")
                        seg.failed.setdefault(bk, "C: push chain unreachable")
                        continue
                # B -> C with the object attached (§3.7 ≤ max_partner_geodesic + 1 m; §3.8 transition; owner 2026-09-19 21:56 (iii'):
                # straighten the torso first, navigate, set C's trunk at its pre-place stand-off). The grasp record goes in
                # BEFORE the leg: ``carried_attachment`` reads its ``eef_in_object`` to pose the object in the hand for every
                # attached check (without it the object was attached at its parked pose - the WARNING of 0919-202952)
                self._check_deadline(f"B -> C path / transition at C candidate {i_c + 1}")
                self.sidecar.add_stance(manip)
                self.sidecar.add_stance(place)
                if push_st is not None:
                    # the deep push stance at the same base pose, entered with EMPTY hands after the drop: a trunk-only transition
                    self.sidecar.add_stance(push_st)
                    try:
                        with self._floor_relaxed():
                            self.record_transition(place, push_st)
                    except Reject as e:
                        errors.append(f"C: drop -> push trunk transition: {e}")
                        seg.failed.setdefault(bk, f"C: drop -> push trunk transition: {e.reason}")
                        self.sidecar.stances.pop("placement", None)
                        self.sidecar.stances.pop("push", None)
                        self._drop_leg_records("push")
                        break
                floor_name = self.bindings.one("floor").name if self.bindings.has("floor") else None
                grasp = GraspRecord(tname, arm, g.key, g.name, yaw, pre, eef, pre_place, place_eef, finger_q=list(g.finger_q), eef_in_object=g.pose,
                                    place_on=(floor_name or pname) if push else pname)
                if twin_facts is not None:
                    grasp.extras.update({"wrist_flip": bool(flip), "twin_choice": twin_facts})
                if len(offered) > 1 or arm != arm0:
                    grasp.extras["arm_bands"] = {a: TT.band_facts((self.surface.get("arm_bands") or {}).get(a)) for a in offered}
                if fallback is not None:
                    grasp.extras["segment_fallback"] = fallback
                if push:  # the release is a drop on the floor outside the edge (like the inside rule's ``place_kind: drop``)
                    grasp.extras.update({"place_kind": "drop", "drop_height_m": float(self.num("drop_height_m", PUSH_UNDER_DROP_HEIGHT_M)),
                                         "drop_cell": [float(v) for v in cd], "goal_cell": [float(v) for v in gc], "then": "push_under"})
                self.sidecar.add_grasp(grasp)
                # carry-height ladder (0920-014945: with the hand frozen at a 12 cm lift over the table, the arrival sweep into the
                # push-under crouch at C dragged the held sandal through the table top and into the floor at every s_T) - the lift
                # is raised until the departure sweep, the carry and the arrival sweep are free; the oracle lifts by the recorded height
                path_bc, carry_errors = None, []
                for h in [float(v) for v in (self.gp.gen("carry_heights_m") or (grasp.carry_height, 0.25))]:
                    q_lift = self.lift_arm_q(manip, eef, h, target, g.pose)
                    if q_lift is None:
                        carry_errors.append(f"lift {h:.2f} m: no IK solution from B")
                        continue
                    grasp.carry_height = h
                    # carry pose candidates (beside the body first), the plain lift configuration last
                    options: List[Tuple[str, Optional[Tuple[G.Pose, List[G.Pose], th.Tensor]]]] = []
                    if str(self.gp.gen("carry_pose", "on")) != "off":
                        options += [(f"carry pose {[round(float(v), 2) for v in P0.to_base_frame(cp[0][0], manip.base_pose2d, P0.base_z(self)).tolist()]}", cp)
                                    for cp in self.carry_poses_at_b(manip, eef, h, target, g.pose)]
                    options.append(("lift configuration", None))
                    for label, cp in options:
                        if cp is not None:
                            carry_pose, wps, q_carry = cp
                            grasp.waypoints = wps  # the in-hand line lift -> carry pose the program's Reorient tracks
                            grasp.extras["carry_pose"] = {"position": [float(v) for v in carry_pose[0]], "orientation": [float(v) for v in carry_pose[1]]}
                            q_arm = q_carry  # the legs are checked with the arm at the carry pose (where the oracle holds it)
                        else:
                            grasp.waypoints = None
                            grasp.extras.pop("carry_pose", None)
                            q_arm = q_lift
                        try:
                            with self._floor_relaxed():  # the floor mesh out (deep-crouch 1 cm margin, as the stance / program checks); the ground slab stays
                                path_bc = self.record_carry_leg(init, manip, place, tname, q_arm, max_len)
                            log.info("%s: carry with lift %.2f m at the %s%s", self.gp.alias, h, label, f" (after: {'; '.join(carry_errors)})" if carry_errors else "")
                            break
                        except Reject as e:
                            carry_errors.append(f"lift {h:.2f} m, {label}: {e.reason}: {str(e)[:700]}")
                            self._drop_leg_records("placement")
                            self.record_carry_arm_q(tname, arm, None)
                    if path_bc is not None:
                        break
                if path_bc is None:
                    errors.append("C: carry leg: " + "; ".join(carry_errors[-3:]))
                    seg.failed.setdefault(bk, "C: carry leg: " + (carry_errors[0][:300] if carry_errors else "no carry option"))
                    log.info("%s: C (d %.2f λ %+.2f) carry leg failed: %s", self.gp.alias, place.stand_off or 0.0, place.lateral, " || ".join(carry_errors)[:1200])
                    fails["B -> C path / transition"] = fails.get("B -> C path / transition", 0) + 1
                    carry_failed_bases.add(base_key)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    break
                # the oracle's carry leg at C is a straight Cartesian line from where the hand is (the carry configuration after the
                # arrival) to the pre-place, tracked by differential IK and finished by ``arm_to`` to the recorded witness within 0.6
                # rad (0920-135625: a witness 2.09 rad away = 'plan_failed'). Track that line here from the carry configuration at C;
                # its end configuration is the branch the witnesses must be in - the drop IK is re-solved from it and must stay within
                # ``carry_finish_max_rad`` (0.5) of it; ``carry_end_q`` seeds the program check's place witness the same way
                idx_r = self.kit.robot.arm_control_idx[arm]
                q_c = self.ikc.q_locked(place).clone()
                q_c[idx_r] = self.carry_arm_q[tname]
                from ispatialgym.oracle import r1pro_model as M
                h0 = self.kit.fk([M.EEF_LINK[arm]], q_c)[M.EEF_LINK[arm]]
                with attached(self.kit, arm, target, eef_in_object=g.pose), self._floor_relaxed():
                    q_end, why_t = self.ikc.track_chain(q_c, [h0, pre_place], arm, ignore=[self.obj("floor")], soft_stall_m=float(self.num("carry_soft_stall_m", 0.02)))
                    r2b = None
                    if q_end is not None:
                        r2b = self.ikc.exact([pre_place, place_eef], place, arm, ignore_for_last=ignore_place, cache=False, q_seed=q_end)
                if q_end is None or r2b is None or not r2b.ok or r2b.q is None:
                    reason = why_t if q_end is None else (r2b.reason if r2b is not None else "no IK")
                    errors.append(f"C: carry leg untrackable to the pre-place: {reason}")
                    seg.failed.setdefault(bk, f"C: carry leg untrackable to the pre-place: {str(reason)[:200]}")
                    log.info("%s: C (d %.2f λ %+.2f) carry leg hand -> pre-place untrackable / drop IK from its end fails: %s", self.gp.alias, place.stand_off or 0.0, place.lateral, reason)
                    fails["C: carry leg untrackable"] = fails.get("C: carry leg untrackable", 0) + 1
                    self._drop_leg_records("placement")
                    self.record_carry_arm_q(tname, arm, None)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    continue
                fin_swing = float(th.max(th.abs(r2b.q[idx_r] - q_end[idx_r])))
                if fin_swing > float(self.num("carry_finish_max_rad", 0.5)):
                    errors.append(f"C: pre-place witness {fin_swing:.2f} rad from the tracked carry end")
                    seg.failed.setdefault(bk, f"C: pre-place witness {fin_swing:.2f} rad from the tracked carry end")
                    log.info("%s: C (d %.2f λ %+.2f) pre-place IK from the tracked carry end swings %.2f rad (> 0.5)", self.gp.alias, place.stand_off or 0.0, place.lateral, fin_swing)
                    fails["C: pre-place branch off the tracked carry"] = fails.get("C: pre-place branch off the tracked carry", 0) + 1
                    self._drop_leg_records("placement")
                    self.record_carry_arm_q(tname, arm, None)
                    self.sidecar.stances.pop("placement", None)
                    self.sidecar.stances.pop("push", None)
                    self._drop_leg_records("push")
                    self.sidecar.grasps.clear()
                    continue
                self.carry_end_q[tname] = q_end[idx_r].clone()
                log.info("%s: carry leg hand -> pre-place tracked at C (%.2f rad swing at the finish)", self.gp.alias, fin_swing)
                if push:
                    self.sidecar.add_contact(PushUnderRecord(arm, chain["pre"], chain["start"], chain["mid"], chain["goal"], chain["direction"], chain["distance_m"],
                                                             tname, stance="push" if push_st is not None else "placement", pitch_deg=chain["pitch_deg"], blade_standoff_m=chain["blade_standoff_m"],
                                                             push_height_m=chain["push_height_m"], half_extent_m=chain["half_extent_m"], fingertip_z_m=chain["fingertip_z_m"],
                                                             ground_ignored=bool(chain.get("ground_ignored", False)),
                                                             drop_cell=[float(v) for v in cd], goal_cell=[float(v) for v in gc], partner=pname,
                                                             **({"orient": chain["orient"]} if chain.get("orient") else {})))
                cam = self.camera_pose_at(init)
                if not self.frustum_ok(cam, self.hull_corners_world(self.hull, t_pose), 0.05):
                    raise Reject("not_in_frame", "target not in the init image (5 %)")
                path_ab = next((p for p in self.sidecar.paths if p.to == "manipulation"), None)
                place.extras.update({"init_surface": {"class": self.surface["kind"], "name": None if self.surface["kind"] == "floor" else self.surface["obj"].name},
                                     "path_length_m": {"A_B": None if path_ab is None else path_ab.length_m, "B_C": path_bc.length_m}})
                d = Draft()
                d.object_poses[tname] = t_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"goal_cell": gc, "grasp": {"key": g.key, "name": g.name, "yaw": yaw}, "placement_stance": P0.summarize_stance(place),
                                  "path_length_m": place.extras["path_length_m"], "F_init": len(F_init), "F_goal": len(F_goal), "F": len(F),
                                  # the carried object's yaw follows the base: goal orientation = init orientation rotated by C.yaw - B.yaw
                                  "goal_quat": [float(v) for v in goal_quat], "carry_yaw_offset_rad": round(d_yaw, 4)})
                d.summary["arm"] = TT.segment_arm_summary(grasp, offered)
                band_c = None if push else (self.place.get("bands") or {}).get(arm)  # the partner-top band of the place is the segment arm's too
                if band_c is not None and band_c.meta.get("mirrored_from"):
                    d.summary["arm"]["mirrored_band"] = True
                arms_here = (seg.arms_of(yaw0, g) if seg is not None else []) or [arm]
                grasp.extras["arm_offer"] = TT.arm_offer_record(offered, [a for a in offered if a in arms_here] or [arm], grasp, TT.arm_offer(self)[1])
                if arm != arm0 or fallback is not None:
                    log.info("%s: segment grasp -> carry -> %s with the %s arm (%s)%s", self.gp.alias, "drop -> push" if push else "place", arm, d.summary["arm"]["reason"],
                             f"; the {fallback['from_arm']} arm failed: {fallback['why'][:200]}" if fallback is not None else "")
                if push:
                    d.summary["push_under"] = {"drop_cell": [float(v) for v in cd], "distance_m": chain["distance_m"], "pitch_deg": chain["pitch_deg"],
                                               "blade_standoff_m": chain["blade_standoff_m"], "push_height_m": chain["push_height_m"], "half_extent_m": chain["half_extent_m"],
                                               "drop_ahead_m": place.extras.get("drop_ahead_m")}
                # §5.6: pose targets are measured for the TARGET only — the partner is fixed furniture and the large-scale
                # iddl's (:measure) has (pose_reached ?target) alone, so a TRO pose target for the partner fails
                # ``validate_tro_against_problem`` at write time (sandal-breakfast_table 0919-202952: the first large-scale
                # IORS draft to reach the writer). The sampler's own "partner unchanged <= 5 mm at the goal" check
                # (:meth:`refine_targets_from_goal`) reads the position recorded here instead.
                pp, _ = partner.get_position_orientation()
                d.summary["partner_position"] = [float(v) for v in pp]
                d.pose_targets[tname] = full_pose_target(t_pose[0], t_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, partner.category.replace("_", " "), path_length_m=path_bc.length_m)
                if push and floor_name:
                    self.sidecar.describe(floor_name, "floor", push_distance_cm=100.0 * float(chain["distance_m"]))
                return d
            errors.append(f"C (d {place.stand_off:.2f} trunk ({place.translate}, {place.tilt:.1f})): no (yaw, grasp) of F ({len(F)}) passes the exact IK at B and C — {fails}")
            log.info("%s: C try exhausted (d %.2f λ %+.2f edge %s, push trunk %s): %s", self.gp.alias, place.stand_off or 0.0, place.lateral, place.extras.get("edge"),
                     None if push_st is None else (push_st.translate, round(push_st.tilt, 2)), fails)
        raise Reject("grasp_unreachable", "; ".join(errors[-3:]) if errors else "no placement candidate")

    # ------------------------------------------------------------------ push-under: the floor mesh and the deep trunks

    def _floor_relaxed(self):
        """Context: the floor object out of the collision world (the cuRobo ground slab stays) — the push-under stance C
        stands at the deep trunks, where the 1 cm margin against the floor mesh is what fails (see
        :meth:`push_under_candidates`); applied to the core's stance / transition re-check and to the program check. Every under
        variant: the place-under (``under_mode`` != "push") sets the object down on the floor from the floor band's crouch as well
        (owner 2026-10-12 item 2c; before, only a floor-class init took the floor out for it)."""
        floor_class = (getattr(self, "surface", None) or {}).get("kind") == "floor"
        return self.ikc.ignoring([self.obj("floor")]) if (self.relation == "under" or floor_class) else _NullContext()

    def check_stances_with_objects(self, draft: Draft):
        with self._floor_relaxed():
            super().check_stances_with_objects(draft)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        with self._floor_relaxed():
            return super().program_check(draft)

    # ------------------------------------------------------------------ goal (§5.6 / §6.12)

    def goal_quat_of(self, draft: Draft) -> th.Tensor:
        """The target's goal orientation: the init orientation rotated by the base-yaw change B -> C (``summary.goal_quat``,
        relative carry yaw), the init orientation itself under ``carry_yaw = "world"``."""
        gq = draft.summary.get("goal_quat")
        return th.tensor([float(v) for v in gq]) if gq is not None else draft.object_poses[self.tname][1]

    def goal(self, draft: Draft):
        tname = self.tname
        goal = self.goal_pose_at(draft.summary["goal_cell"], self.goal_quat_of(draft))
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        end = self.sidecar.stances.get("push") or self.sidecar.stance("placement")  # where the robot ends: the push stance when the drop and the push split
        self.sidecar.set_goal(end.base_pose2d, end.translate, end.tilt, arms="same")

    def after_settle(self, draft: Draft):
        tname = self.tname
        goal = self.goal_pose_at(draft.summary["goal_cell"], self.goal_quat_of(draft))
        draft.goal_poses[tname] = goal
        tgt = draft.pose_targets[tname]
        tgt["position"], tgt["orientation"] = [float(v) for v in goal[0]], [float(v) for v in goal[1]]

    def refine_targets_from_goal(self, draft: Draft):
        super().refine_targets_from_goal(draft)
        p, _ = self.partner.get_position_orientation()
        ref = draft.summary.get("partner_position")
        if ref is None:
            return
        moved = float(th.linalg.norm(p.float() - th.tensor(ref, dtype=th.float32)))
        if moved > self.num("partner_moved_max_m", 0.005):
            raise Reject("goal_partner_moved", f"partner moved {moved:.3f} m at the goal")

    def after_goal_check(self, draft: Draft):
        """Place / pre-place at the settled goal pose re-checked from C with the object attached (the record kept), with the recorded
        grasp's / push's arm (the segment arm)."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place = self.sidecar.stance("placement")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"])
        ignore = [self.partner] if self.relation == "ontop" else [self.obj("floor")]
        if self.push_under:
            # the release is the drop outside the edge (unchanged by the settle): re-check it as recorded; the push goal blade
            # follows the settled goal centre (a few mm), the chain start → mid → goal is re-solved (closed gripper)
            with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([grasp.pre_place, grasp.place], place, grasp.arm, ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / drop fail the exact IK from stance C at the goal check: {res.reason}")
            push = self.sidecar.contacts.get("push")
            if push is not None:
                d3 = th.as_tensor(push.direction, dtype=th.float32)
                off = float(push.extras["half_extent_m"]) + float(push.extras["blade_standoff_m"]) - float(self.num("push_goal_extra_m", 0.03))
                z_push = float(push.contact[0][2])
                goal_b = th.tensor([float(goal_pos[0]), float(goal_pos[1]), z_push]) - d3 * off
                start_b = push.contact[0]
                mid_b = 0.5 * (start_b + goal_b)
                q = push.contact[1]
                push_from = self.sidecar.stances.get("push") or place  # the chain is pushed from the deep push stance when the drop and the push split
                with self.ikc.ignoring([self.obj("floor"), self.target], ground=not bool(push.extras.get("ground_ignored", False))):
                    r3 = self.ikc.exact([push.pre, (start_b, q), (mid_b, q), (goal_b, q)], push_from, push.arm, cache=False, fingers=C.FINGER_CLOSED)
                if not r3.ok:
                    raise Reject("push_unreachable", f"push-under chain to the settled goal fails the exact IK from stance C: {r3.reason}")
                push.mid, push.goal = (mid_b, q), (goal_b, q)
                push.distance_m = float(th.linalg.norm(goal_pos[:2].float() - th.as_tensor(push.extras["drop_cell"], dtype=th.float32)))
                push.extras["goal_cell"] = [float(goal_pos[0]), float(goal_pos[1])]
        else:
            place_eef = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
            pre_place = IK.pre_pose(place_eef)
            with attached(self.kit, grasp.arm, self.target, eef_in_object=grasp.eef_in_object):
                res = self.ikc.exact([pre_place, place_eef], place, grasp.arm, ignore_for_last=ignore, cache=False)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / place at the settled goal fail the exact IK from stance C: {res.reason}")
            grasp.pre_place, grasp.place = pre_place, place_eef
        cam_b = self.camera_pose_at(self.sidecar.stance("manipulation"))
        if not self.ikc.eef_in_frame(grasp.grasp, cam_b):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if self.relation == "ontop" and not self.ikc.eef_in_frame(place_eef, self.camera_pose_at(place)):
            raise Reject("gripper_not_in_frame", "place eef pose outside the placement image (5 %)")

    def turn_recheck(self, draft: Draft, grasp, q_first: Optional[th.Tensor] = None) -> Optional[str]:
        """``tabletop.held_turns``: not offered here. The held chain runs through state recorded for the recorded turn alone - the carry
        pose (``grasps[].waypoints``, ``carry_arm_q``), the B -> C departure transition / path / arrival swept with the arm there, and
        the carry leg at C tracked from it to the pre-place (``carry_end_q``, which seeds the program check's place) - so the other turn
        means re-planning the carry from the carry-pose search on."""
        raise TT.TurnNotOffered("the held chain runs through the carry pose, the B -> C carried travel swept at the recorded carry configuration "
                                "(carry_arm_q) and the carry leg at C tracked from it (carry_end_q): the other turn needs that carry re-planned")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        g = self.sidecar.grasp_of(self.tname)
        place = self.sidecar.stances.get("placement")
        if g is None or place is None:
            return []
        c = self.choice.models["target"]
        return [{"scope": {"scene": self.scene, "partner": self.partner.name, "target": f"{c.category}/{c.model}"}, "type": "placement",
                 "edge": place.edge.name if place.edge else None, "stand_off": place.stand_off, "lateral": place.lateral, "trunk": [place.translate, round(place.tilt, 3)],
                 "goal_cell": draft.summary.get("goal_cell"), "grasp": [round(g.yaw, 2), g.grasp_name], "hard": False}]

    # ------------------------------------------------------------------ inside / touching / nextto (registered 2026-10-12, modes next)

    def _adopt_mode(self, mode: type) -> None:
        """The relation's mode takes over: ``mode`` (the class its hook returns, from the mode's own module, written like an IORSLarge
        subclass or mixin; not ``@register``-ed) goes in front of this sampler's class, so every method it defines (``bind_scene_roles``,
        ``prepare_bindings``, ``sample_template``, ``sample_instance``, ``goal``, ``after_goal_check``, ... or a helper of the ontop / under
        flow) replaces IORSLarge's, which ``super()`` still reaches."""
        self.relation_mode = mode
        self.__class__ = type(f"{type(self).__name__}_{mode.__name__}", (mode, type(self)), {})

    def _sample_inside(self) -> type:
        """``inside`` (``...-inside_large_scale``): the hook of the inside mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method. Until the mode lands every template of the subtask is rejected with
        ``relation_not_implemented``."""
        from ispatialgym.sampling.v2.tasks import iors_large_inside as M
        return M.InsideMode

    def _sample_touching(self) -> type:
        """``touching`` (``...-touching_large_scale``): the hook of the touching mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method. Until the mode lands every template of the subtask is rejected with
        ``relation_not_implemented``."""
        from ispatialgym.sampling.v2.tasks import iors_large_touching as M
        return M.TouchingMode

    def _sample_nextto(self) -> type:
        """``nextto`` (``...-nextto_large_scale``): the hook of the nextto mode - it returns the mode's class (:meth:`_adopt_mode`),
        imported from the mode's own module inside this method: ``iors_large_nextto.NextToMode``, the floor-standing target pushed next
        to the partner by the base."""
        from ispatialgym.sampling.v2.tasks import iors_large_nextto as M

        return M.NextToMode
