"""Disembedding (spec §6.4, init per §4 + §4.7): find the target among distractors on the source tray and move it to the
destination tray.

Roles: ``target`` (grasped, tabletop pipeline), ``source`` / ``destination`` (the two fixed trays; the v1 facade names
them ``tray_source`` / ``tray_dest``), ``distractors`` (group, §3.12 rule 1 pools), scene ``support``, ``floor``.

Template (``sample_template``): hulls / rest orientations / heights, the target's library grasps, the trays' inner
floors (rasters in the tray frame), the reach band of the target at the tray-floor height (``BandSpec``), the destination
side (``generation.destination_side``, else drawn from the template seed), Phases 1a (any edge) / 1b (H_max = tallest of
target, distractors, trays) / 1c (row ``disembedding``: m = b + 0.437 + 0.02 on the destination side, |λ| + half the source
tray's length along the edge + 0.02 on the other) and Phase 2 with the source tray (centred on the focus, long axis along the
edge) as the in-frame body, keeping only the (stance, position) candidates beside whose source tray the destination fits
(``_destination_fit``, owner 2026-10-12).

Instance (``sample_instance``): a Phase-2 survivor + transition; source tray at the focus; the target inside it from
the feasible set F (footprint on the tray floor, 0.5 cm); the destination tray on the template's side at the lateral
position where exactly 4 cm of it is inside the init image (>= 4 cm from the source tray, footprint on the top);
distractors on the source floor, footprints >= 1 cm from each other and from the target and <= ``tray_budget`` of the floor
(owner 2026-10-12 A2b: no spacing along the grasp's pinch axis, no pad-clearance growth), each at its rest orientation - the
target's pose when the variant says so (``generation.distractor_pose`` lying / standing, the milk picks of the handoff
2026-10-10) - or, for the puzzle (``generation.fixed_layout``), every part of the embedded figure at its layout position with
the target held there too, the whole figure turned by a yaw drawn among those whose grasp screening keeps a reachable grasp
(owner 2026-10-12, ``_layout_pick``; recorded in the TRO's ``scene.fixed_layout``); every object but the destination in frame
(5 %);
whole-body checks at init and manipulation with the objects placed; occlusion <= 0.30 (two segmentation renders,
visible area >= 400 px) with the destination tray confirmed visible; the §4.7 placement stance with the place /
pre-place IK (object in hand, tray + support ignored for the place pose) and the lateral path with the object at the
carry height (>= tallest distractor + tray rim + 3 cm above the source floor), the heading kept and only forward / backward /
lateral moves (owner 2026-10-13: the edge slide at the first collision-free drive-back, never the grid path). The grasp ignores the table, the source tray and
every distractor / fixed-layout part, the whole pick (pre-grasp included) the fixed-layout parts (owner 2026-10-12 A2b: the
grasp may touch or displace them as long as they stay on the source tray). Sidecar: ``init`` / ``manipulation`` / ``placement``
stances, the transition, the path, one grasp record (pre-grasp, grasp, pre-place, place, carry_height, ``place_on`` = the
destination, ``may_touch`` = those parts / distractors, the source tray they must stay on and the parts the pre-grasp ignores),
language descriptors; TRO ``initial_occlusion`` (``ratio``, ``visible_area``, ``reference_area``, ``occluder_ids``,
``camera_key``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import alias_memory as AM
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import campaign_tray_goals as CTG
from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.sampling.constraints import apply_mass
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import support_split as SS  # i7-split 2026-11-07
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register, start_draw_active, start_facts
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

log = logging.getLogger("ispatialgym.sampling.v2")


def disemb_pos_redraws() -> int:
    """i7-yield2 r3 (lead #66, 2026-11-09): with the split cone (generation.phase2_split_cone) each attempt draws the target's start position
    and tries that position's work stances only; disembedding_10 after the start-draw speed fix lost 30-38 attempts per job to
    target_placement (an empty feasible set F of 8 yaws x the 2 turn-filtered fork grasps at the drawn position), each attempt paying the
    ~70 s start draw again. ``ISPATIALGYM_DISEMB_POS_REDRAWS=N`` redraws the position (without replacement, from the drawn start's pool)
    up to N times within the attempt; 0 = the single position. DEFAULT 8 since 2026-11-09 (i7-yield2 r3 A/B: disembedding_10 40/4/3 0.307 ->
    7/4/4 0.116 GPU-h per success, control disembedding_02 13/4/4 0.113 -> 5/4/4 0.098). Kit ``pos_redraw.redrawn / passed /
    passed_after_k / exhausted``."""
    try:
        return max(0, int(os.environ.get("ISPATIALGYM_DISEMB_POS_REDRAWS", "8")))
    except ValueError:
        return 0

SUBTASK = "disembedding"


# ----------------------------------------------------------------------------------------------------------------
# tray inner floor (raster in the tray frame; §6.4 "placement raster 1 cm over the tray's inner floor")
# ----------------------------------------------------------------------------------------------------------------


#: a tray counts as resting on the top when its lowest hull point is within this of the top and its floor normal is
#: within ``REST_TILT_MAX_DEG`` of vertical after the measurement settle
REST_Z_TOL = 0.01
REST_TILT_MAX_DEG = 3.0
#: Owner 2026-10-11 ~20:25: spots of the tray-floor measurement drop - the free cell farthest from any non-free cell or the top's
#: edge, then the next best ones well apart; the template is rejected only when the tray rests at none of them (a counter's
#: free-cell centroid can lie on its sink, burner or microwave).
MEASURE_DROP_SPOTS = 3
#: floor cells = tray hits within this band of the floor level (10 % quantile of the hit heights); at least ``FLOOR_MIN_CELLS``.
#: 2026-09-24: 4 mm -> 1.5 mm and the raster eroded by one cell (``FLOOR_ERODE_CELLS``): the 4 mm band (kept for the 1.6 mm
#: step of the retired gsxbym floor) admitted the first rows of the rim fillet of uekqey / hvlfig (0.5-4 mm up, ~5 % of the
#: cells at the short ends); objects placed over them slid / rolled toward the centre and the tall apple-juice bottle
#: toppled (roll test 0924: 111 mm of travel from the end zone, <= 1.5 mm from the centre)
FLOOR_BAND = 0.0015
FLOOR_ERODE_CELLS = 0  # a one-cell erosion starved the placement of 4 objects (17/24 attempts without a free position, 0925-000707)
FLOOR_QUANTILE = 0.10
FLOOR_MIN_CELLS = 40
#: 2026-09-24: release height of the objects above the tray floor = the §3.2 3 mm (``TT.Z_OFFSET``). Measured in the headless
#: crowding test (4 facet-snapped objects, 1 cm gaps, 10 trials each): 3 mm passes the trembling gate 8/10, 1 mm 5/10, 0.2 mm
#: 2/4, 5 mm 6/10 - a release inside PhysX's contact offset turns the +-0.5-1.5 mm height variation of the raycast floor into
#: penetration kicks, while the drop itself is benign once the prism hulls land on a facet
RELEASE_GAP_M = TT.Z_OFFSET
#: 2026-09-25: the TRAYS are released at contact with the table top instead of the §3.2 3 mm above it. Reproduced in the
#: sampler's own world (house_single_floor, robot present; /tmp/isg_droptest/inworld_test.py): trays dropped 3 mm -> the
#: source tray chatters on the table top for seconds (0.4-2 mm per frame) and every bottle on it moves 2-150 mm; trays placed
#: at contact -> every object <= 0.15 mm per frame. Holding the table or the robot's stance made no difference; bottles set
#: directly on the table are still. 0.2 mm is inside PhysX's 1 mm contact offset: the contact exists from the first step.
TRAY_RELEASE_GAP_M = 0.0002
#: owner item 3 (2026-09-16 10:15): release this far above the actual surface under the footprint centre (``surface_z_under``)
RELEASE_ABOVE_SURFACE = 0.01
#: release height of the fixed-layout parts (target included) above the source floor: the embedded figure's parts sit 0.3 mm
#: apart and its physics check settled them from 0.5 mm (embedded_figure/README.txt, 2026-10-11); ``generation.fixed_layout.
#: release_gap_m`` overrides
FIXED_LAYOUT_RELEASE_GAP_M = 0.0005
#: i7-tray 2026-11-08 (owner 1.1, disembedding_01 4/4 -> 0/4): ``ISPATIALGYM_LAYOUT_PAD_CHECK`` (default ``all`` since the owner decision
#: 2026-11-08 item 4; ``0`` = the pick as before; it acts on fixed layouts only). The
#: pick ignores the fixed-layout parts for every pose (A2b ``pick_world``) and the library validated the target alone on a slab, so a
#: grasp whose pre-shaped fingers descend onto a neighbouring part (the embedded figure's frame, 0.3 mm from the target) passes the draw
#: and the oracle's approach stops on the part (``approach_support_stop``, 13-22 mm short). On: the library's finger rule - the hand
#: hulls (``contact_policy.HandChecker``) at the oracle's pre-shape opening (``tabletop.policy_opening_of``) along the pre-grasp ->
#: grasp line, a finger at most ``finger_support_max_penetration_m`` into an obstacle - is applied to the layout's parts (their convex
#: collision pieces at the layout's relative poses) and the rows whose fingers stand on a part leave the pick. The layout is rigid, so
#: the verdict is per row (its finger-swapped twin has the same footprint). Counters ``layout_pad.*`` (measured with the switch off
#: too; nothing leaves the pick then).
LAYOUT_PAD_SWITCH = "ISPATIALGYM_LAYOUT_PAD_CHECK"


def layout_pad_mode() -> str:
    """``all`` (DEFAULT since 2026-11-08, owner decision item 4; 1 / on: every layout part), ``heavy`` (only the parts at least as heavy as
    the target - the frame: A2b's "the grasp may displace the parts" kept for the light inner pieces the pads push aside) or ``off``
    (0 / off / false / no: the pick before i7-tray)."""
    v = os.environ.get(LAYOUT_PAD_SWITCH, "all").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    return "heavy" if v == "heavy" else "all"


def layout_pad_on() -> bool:
    return layout_pad_mode() != "off"


class HullObstacle:
    """A convex piece as an obstacle for ``contact_policy.HandChecker.clearance`` (the ``Support.sdf`` interface): the signed distance
    is the largest of the hull's face-plane distances (exact inside - the penetration depth - and a lower bound outside)."""

    kind = "hull"

    def __init__(self, equations: np.ndarray, name: str = "piece"):
        self.eq = th.as_tensor(np.asarray(equations, dtype=np.float64), dtype=th.float32)
        self.name = name

    @staticmethod
    def of_points(pts: np.ndarray, name: str = "piece") -> "HullObstacle":
        from scipy.spatial import ConvexHull

        return HullObstacle(ConvexHull(np.asarray(pts, dtype=np.float64)).equations, name)

    def sdf(self, p: th.Tensor) -> th.Tensor:
        eq = self.eq.to(device=p.device, dtype=p.dtype)
        return (p @ eq[:, :3].T + eq[:, 3]).max(dim=-1).values


def pose_mat(pose) -> np.ndarray:
    """4x4 of a ``(position, quaternion xyzw)`` pose."""
    m = np.eye(4)
    m[:3, :3] = SS.quat_mat(pose[1])
    m[:3, 3] = SS._np(pose[0]).ravel()[:3]
    return m


def layout_pad_clearances(grasps: Sequence[Any], parts: Dict[str, List[HullObstacle]], arm: str = "right") -> Dict[str, Dict[str, float]]:
    """``{row name: {part: smallest finger clearance (m) along the approach}}`` of each library row of ``grasps`` (eef in the target frame,
    ``g.pose``) against ``parts`` (part name -> its convex pieces in the TARGET's frame), the fingers at the oracle's pre-shape opening
    (``tabletop.policy_opening_of``; fully open without a width) on the pre-grasp (``conventions.PRE_DIST`` back along the eef +z) ->
    grasp line at the policy's ``approach_step_m``."""
    from ispatialgym.grasping import contact_policy as CP

    hc = CP.HandChecker(arm)
    out: Dict[str, Dict[str, float]] = {}
    for g in grasps:
        o = TT.policy_opening_of(g)
        o = 0.5 * float(C.GRIPPER_OPEN_M) if o is None else float(o)
        Tg = th.as_tensor(pose_mat(g.pose), dtype=th.float32)
        path = CP.approach_path(CP.pre_pose_mat(Tg), Tg, float(hc.policy.approach_step_m)).reshape(-1, 4, 4)
        out[str(getattr(g, "name", ""))] = {pname: float(hc.obstacle_clearance(path, pieces, finger_q=(o, o))["finger"].min())
                                            for pname, pieces in parts.items() if pieces}
    return out


def layout_pad_clearance(grasps: Sequence[Any], parts: Dict[str, List[HullObstacle]], arm: str = "right") -> Dict[str, Tuple[float, Optional[str]]]:
    """``{row name: (smallest finger clearance (m) to any part, that part)}`` (:func:`layout_pad_clearances`)."""
    out = {}
    for n, per in layout_pad_clearances(grasps, parts, arm).items():
        who = min(per, key=per.get) if per else None
        out[n] = (per[who] if who is not None else float("inf"), who)
    return out


def clutter_approach_proof(grasp, width, parts):
    """Measure the actual selected fingers against unchanged clutter geometry.

    This proposal screen uses the existing hand policy, with its permitted
    penetration. It adds no spacing envelope and does not alter tray occupancy.
    """
    from ispatialgym.grasping import contact_policy as CP

    if not parts or any(not pieces for pieces in parts.values()):
        raise Reject('distractor_approach_geometry', 'missing native distractor collision pieces')
    row = SimpleNamespace(name=str(grasp.grasp_name), pose=grasp.eef_in_object, meta={'width': width})
    per = layout_pad_clearances([row], parts, grasp.arm)[row.name]
    if set(per) != set(parts) or not all(math.isfinite(v) for v in per.values()):
        raise Reject('distractor_approach_geometry', 'incomplete or nonfinite finger-clearance measurement')
    allowed = float(CP.DEFAULT.finger_support_max_penetration_m)
    blocked = sorted(name for name, distance in per.items() if distance < -allowed - 1e-9)
    return {'grasp': row.name, 'arm': grasp.arm, 'pinch_width_m': width,
            'finger_clearance_m': per, 'allowed_penetration_m': allowed, 'blocked_by': blocked}


#: a model already lies / stands in its canonical orientation when its vertical extent is within this fraction of the wanted one
POSE_EXTENT_TOL = 0.05
#: standing: the two end slices (this fraction of the length each) compared to put the wider end down (a bottle on its base)
END_SLICE_FRACTION = 0.15


def _end_area(pts: th.Tensor, axis: int, upper: bool) -> float:
    """Bbox area, across ``axis``, of the hull points within ``END_SLICE_FRACTION`` of the lower / upper end along ``axis``."""
    a = pts[:, axis]
    lo, hi = float(a.min()), float(a.max())
    cut = END_SLICE_FRACTION * (hi - lo)
    sel = pts[a >= hi - cut] if upper else pts[a <= lo + cut]
    others = [i for i in range(3) if i != axis]
    ext = sel[:, others].max(dim=0).values - sel[:, others].min(dim=0).values
    return float(ext[0]) * float(ext[1])


def pose_rest_quat(hull: th.Tensor, pose: Optional[str]) -> Tuple[th.Tensor, str]:
    """Rest orientation of a distractor that takes the target's pose (handoff 2026-10-10, sheet notes ``pose: lying`` / ``pose:
    standing``): ``lying`` = its smallest extent vertical (long axis horizontal, on its broadest side), ``standing`` = its largest
    extent vertical, wider end down (a bottle on its base, not on its cap). The canonical orientation when it already lies /
    stands (vertical extent within ``POSE_EXTENT_TOL`` of the wanted one) and for any other ``pose``. Returns ``(quat, how)``."""
    if pose not in ("lying", "standing"):
        return C.IDENTITY_QUAT.clone(), "canonical"
    ext = hull.max(dim=0).values - hull.min(dim=0).values
    want = int(th.argmin(ext)) if pose == "lying" else int(th.argmax(ext))
    if abs(float(ext[2]) - float(ext[want])) <= POSE_EXTENT_TOL * float(ext[want]):
        return C.IDENTITY_QUAT.clone(), f"{pose}: canonical"
    up = -1.0 if pose == "standing" and _end_area(hull, want, upper=True) > _end_area(hull, want, upper=False) else 1.0
    # the rotation taking the local axis ``up * e_want`` onto +z
    if want == 0:
        q = C.quat_from_axis_angle(th.tensor([0.0, 1.0, 0.0]), -up * math.pi / 2)
    else:
        q = C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), up * math.pi / 2)
    return q, f"{pose}: local {'+' if up > 0 else '-'}{'xyz'[want]} up"


def figure_axes(tangent, inward) -> Tuple[th.Tensor, th.Tensor, float]:
    """World xy axes and yaw of the embedded figure's frame on the source tray (handoff 2026-10-10): +x = the robot's right
    (−tangent), +y = away from the robot (inward), aligned with the edge like the source tray."""
    x_axis = -TT._t(tangent)[:2].float()
    y_axis = TT._t(inward)[:2].float()
    return x_axis, y_axis, math.atan2(float(x_axis[1]), float(x_axis[0]))


def layout_yaws(step_deg: float) -> List[float]:
    """The yaws (rad, [0, 2π)) a fixed layout may turn by about the vertical (owner 2026-10-12, the puzzle): 0 - the figure drawn
    upright, table-aligned - and every ``step_deg`` around."""
    n = max(1, int(round(360.0 / float(step_deg))))
    return [2.0 * math.pi * k / n for k in range(n)]


def turned_axes(x_axis: th.Tensor, y_axis: th.Tensor, yaw: float, turn: float) -> Tuple[th.Tensor, th.Tensor, float]:
    """The figure frame (``figure_axes``) turned by ``turn`` about the vertical: both axes rotated, the yaw increased."""
    c, s = math.cos(turn), math.sin(turn)
    rot = th.tensor([[c, -s], [s, c]], dtype=x_axis.dtype)
    return rot @ x_axis, rot @ y_axis, yaw + turn


def layout_origins(assembly: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """``model -> (x, y)`` of every part of a layout file: ``parts.<model>.origin_in_figure_m`` (the part's bbox centre in the
    figure frame; ``origin_in_tray_m`` in the 2026-10-10 version of the file). Only x / y: every part lies flat on the floor."""
    out = {}
    for model, part in (assembly.get("parts") or {}).items():
        o = part.get("origin_in_figure_m", part.get("origin_in_tray_m"))
        if o is not None:
            out[model] = (float(o[0]), float(o[1]))
    return out


def repark_spawned(s) -> None:
    """Teleport every spawned object back onto its parking slot before any of its geometry is read.

    Why (E1 pilot, every disembedding variant): ``SimWorld.park`` disables gravity through the PhysX view only, and the
    ``og.sim.stop()`` / ``play()`` of every later ``spawn`` rebuilds PhysX from USD, so the objects parked earlier fall
    again (no floor under the parking slots) during the following simulator steps. ``SimWorld.local_hull`` then mixes
    the *render-side* (Fabric) link transforms, which physics-only steps leave at the last rendered pose, with the
    current PhysX root pose: the hull of a falling body comes out shifted by the distance fallen since the last
    render (+6.5 cm in z for the source tray), ``bottom_offset`` is off by the same amount and the tray is placed
    inside the table top (6 cm slab), pushed through it and found tilted 24 cm lower. A teleport writes PhysX *and*
    Fabric, zeroes the velocities and re-disables gravity; no stop / play happens during a template, so one re-park
    at the start of ``sample_template`` keeps every hull consistent for the template's lifetime."""
    for i, o in enumerate(s.world.spawned.values()):
        s.world.park(o, i)


def tilt_deg(quat) -> float:
    """Angle between the body's local z axis and the world vertical, in degrees."""
    up = C.quat_rotate(TT._t(quat), th.tensor([0.0, 0.0, 1.0]))
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


class TrayFloor:
    """Inner floor of a tray measured once in the tray's root frame (cells = local xy of the floor raster, ``z_local`` =
    floor height above the root origin); ``at(pose)`` gives the world view used for placements."""

    def __init__(self, cells_local: th.Tensor, z_local: float, res: float, rim_local: float, erode: int = FLOOR_ERODE_CELLS,
                 heights: Optional[th.Tensor] = None):
        self.z_local = float(z_local)
        self.res = float(res)
        self.rim_local = float(rim_local)  # top of the tray walls above the root origin
        # i7-split 2026-11-07: the floor raster before the erosion (root-frame xy) and each cell's hit height relative to the floor
        # plane (m) - the support split's straddler test and tolerance (``register_split``)
        self.raw_cells_local = cells_local.clone()
        self.heights = heights
        lo = cells_local.min(dim=0).values
        hi = cells_local.max(dim=0).values
        self.origin = lo.clone()
        nx = int(round(float(hi[0] - lo[0]) / res)) + 1
        ny = int(round(float(hi[1] - lo[1]) / res)) + 1
        grid = th.zeros(nx, ny, dtype=th.bool)
        idx = th.round((cells_local - lo) / res).long()
        grid[idx[:, 0], idx[:, 1]] = True
        # erode the boundary ring(s): the outermost floor cells border the rim fillet (2026-09-24, see ``FLOOR_BAND``)
        for _ in range(int(erode)):
            g = grid.float()[None, None]
            inner = -th.nn.functional.max_pool2d(-g, 3, stride=1, padding=1)[0, 0] > 0.5  # min-pool: all 8 neighbours are floor
            if int(inner.sum()) < FLOOR_MIN_CELLS:
                break
            grid = inner
        self.grid = grid
        ij = th.nonzero(grid)
        self.cells_local = lo + ij.float() * res

    @property
    def n_cells(self) -> int:
        return int(self.grid.sum())

    def area(self) -> float:
        return self.n_cells * self.res * self.res

    def contains_local(self, xy_local, margin: float = 0.0) -> bool:
        p = xy_local if isinstance(xy_local, th.Tensor) else th.tensor(xy_local, dtype=th.float32)
        pts = [p]
        if margin > 0:
            for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin), (margin, -margin), (-margin, margin)):
                pts.append(p + th.tensor([dx, dy]))
        for q in pts:
            i = int(round(float(q[0] - self.origin[0]) / self.res))
            j = int(round(float(q[1] - self.origin[1]) / self.res))
            if i < 0 or j < 0 or i >= self.grid.shape[0] or j >= self.grid.shape[1] or not bool(self.grid[i, j]):
                return False
        return True

    def at(self, pose: TT.Pose) -> "TrayFloorWorld":
        return TrayFloorWorld(self, pose)

    def register_split(self, ikc, tray, plane_world: float):
        """i7-split 2026-11-07: register the tray's inner floor as its support-split plane (``support_split.register``) in the tray's root
        frame (``z_local``: the tray moves with the focus) - the tolerance from the floor raster's heights, the raw floor cells for the
        straddler test; ``plane_world`` = the floor's world height with the tray on the table (reported). Its rims and walls then stay
        obstacles wherever the tray is ignored (the grasp, the lift, the place into it, the program check)."""
        tol = SS.tolerance_from_heights(self.heights, 0.0) if self.heights is not None else SS.Tolerance.none("no_raster")
        return SS.register(ikc, tray, float(plane_world), tol, dz=self.z_local, cells_local=self.raw_cells_local, what="tray_floor")

    @staticmethod
    def floor_cells(zs: th.Tensor, band: float = FLOOR_BAND, quantile: float = FLOOR_QUANTILE):
        """Floor level and cell mask from the per-ray tray hit heights (``nan`` = the ray missed the tray): the floor is the
        ``quantile`` of the hit heights (the lowest inner level; the rim and the wall tops are higher), a cell is a floor
        cell when its hit lies within ``band`` of it. Returns ``(floor_z, keep)``; ``floor_z`` is ``nan`` without hits."""
        valid = zs[~th.isnan(zs)]
        if len(valid) == 0:
            return float("nan"), th.zeros(len(zs), dtype=th.bool)
        floor_z = float(th.quantile(valid, quantile))
        keep = (~th.isnan(zs)) & ((zs - floor_z).abs() < band)
        return floor_z, keep

    @staticmethod
    def drop_spots(top, half_diag: float, n: int = MEASURE_DROP_SPOTS) -> List[Tuple[th.Tensor, float]]:
        """Up to ``n`` measurement-drop points on ``top`` as ``[(xy, clearance_m)]``: free cells ranked by their clearance - the
        distance to the nearest non-free cell or past the top's edge (a distance transform of the free grid) - each next one at
        least 2 x ``half_diag`` from the chosen ones (else ``half_diag``). A top without free cells gives its raster centre."""
        from scipy.ndimage import distance_transform_edt

        free = top.free.cpu().numpy().astype(bool)
        if not free.any():
            return [(th.tensor([float(top.xs.mean()), float(top.ys.mean())]), 0.0)]
        res = float(top.resolution)
        clear = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * res - res / 2
        idx = np.argwhere(free)
        order = np.lexsort((np.arange(len(idx)), -clear[idx[:, 0], idx[:, 1]]))
        pts = th.stack([top.xs[th.as_tensor(idx[:, 0])], top.ys[th.as_tensor(idx[:, 1])]], dim=1).float()
        chosen: List[int] = []
        for sep in (2.0 * half_diag, half_diag):
            for k in order:
                if len(chosen) == n:
                    break
                if k in chosen or any(float(th.linalg.norm(pts[k] - pts[c])) < sep for c in chosen):
                    continue
                chosen.append(int(k))
        return [(pts[k].clone(), float(clear[idx[k, 0], idx[k, 1]])) for k in chosen]

    @staticmethod
    def rest_drop(s, tray, top, hull, spots) -> Tuple[th.Tensor, th.Tensor, th.Tensor, str]:
        """Drop the tray at each of ``spots`` in turn (identity orientation, bottom on the top, 20 physics steps) until it rests on
        the top (bottom within ``REST_Z_TOL``, tilt <= ``REST_TILT_MAX_DEG``): ``(rp, rq, placed, where)``. The tray is parked
        between the drops; resting at none of them is a ``Reject("support")`` that lists every drop."""
        quat = C.IDENTITY_QUAT.clone()
        slot = list(s.world.spawned).index(tray.name)
        tried = []
        for i, (xy, clear) in enumerate(spots):
            if i:
                s.world.park(tray, slot)
            pos = th.tensor([float(xy[0]), float(xy[1]), top.z + s.world.bottom_offset(hull, quat) + TT.Z_OFFSET])
            tray.set_position_orientation(pos, quat)
            s.world.activate(tray)
            s.world.step_physics(20, still=[s.robot])
            rp, rq = tray.get_position_orientation()
            rp, rq = rp.float(), rq.float()
            bottom = float(C.transform_points((rp, rq), hull)[:, 2].min())
            tilt = tilt_deg(rq)
            where = (f"tray at {[round(float(v), 3) for v in rp.tolist()]} (placed at {[round(float(v), 3) for v in pos.tolist()]}, clearance {clear:.2f} m), "
                     f"bottom z {bottom:.3f}, tilt {tilt:.1f} deg, top z {float(top.z):.3f}")
            if abs(bottom - float(top.z)) <= REST_Z_TOL and tilt <= REST_TILT_MAX_DEG:
                if i:
                    log.info("%s: the measurement drop rests at spot %d of %d (%s)", tray.name, i + 1, len(spots), where)
                return rp, rq, pos, where
            tried.append(where)
        raise Reject("support", f"{tray.name}: did not rest on the top at any of {len(tried)} drop spots ({'; '.join(tried)})")

    @staticmethod
    def measure(s, tray, top, res: float = 0.01) -> "TrayFloor":
        """Drop the tray on the free cell of the top with the most clearance (:meth:`drop_spots`, the next spots when it does
        not rest there: :meth:`rest_drop`), raycast a ``res`` raster from above, keep the cells whose first hit is the tray at
        its lowest inner level (10 % quantile of the hit heights + 4 mm), park the tray again.

        The tray is re-parked first so that ``local_hull`` reads a consistent body (see :func:`repark_spawned`). A tray that
        rests at none of the drop spots is a ``Reject("support")`` with the geometry, not a raster of nothing."""
        s.world.park(tray, list(s.world.spawned).index(tray.name))
        hull = s.world.local_hull(tray)
        if len(hull) == 0:
            raise Reject("support", f"{tray.name}: no collision hull")
        ext = hull.max(dim=0).values - hull.min(dim=0).values
        spots = TrayFloor.drop_spots(top, 0.5 * math.hypot(float(ext[0]), float(ext[1])))
        try:
            rp, rq, pos, where = TrayFloor.rest_drop(s, tray, top, hull, spots)
            # raster over the hull's bounds at the settled pose (``tray.aabb`` lags physics-only steps; the PhysX pose does not)
            world_hull = C.transform_points((rp, rq), hull)
            lo, hi = world_hull.min(dim=0).values, world_hull.max(dim=0).values
            xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
            ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
            gx, gy = th.meshgrid(xs, ys, indexing="ij")
            starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), float(hi[2]) + 0.10)], dim=1)
            ends = starts.clone()
            ends[:, 2] = float(lo[2]) - 0.02
            hits = s.world.raycast(starts, ends)
            prims = tuple({l.prim_path for l in tray.links.values()} | {tray.prim_path})
            zs = th.tensor([float(h["position"][2]) if (h["hit"] and str(h["rigidBody"]).startswith(prims)) else float("nan") for h in hits])
            floor_z, keep = TrayFloor.floor_cells(zs)
            n_tray = int((~th.isnan(zs)).sum())
            if n_tray == 0:
                bodies = sorted({str(h["rigidBody"]) for h in hits if h["hit"]})[:6]
                raise Reject("support", f"{tray.name}: no raycast hits on the tray ({sum(1 for h in hits if h['hit'])}/{len(hits)} rays hit {bodies}; rays from z "
                                        f"{float(starts[0, 2]):.3f} to {float(ends[0, 2]):.3f}; {where})")
            world_xy = starts[keep][:, :2]
            if len(world_xy) < FLOOR_MIN_CELLS:
                valid = zs[~th.isnan(zs)]
                raise Reject("support", f"{tray.name}: inner floor too small ({int(keep.sum())} cells within {FLOOR_BAND * 1000:.0f} mm of the floor z {floor_z:.3f}; "
                                        f"{n_tray}/{len(hits)} rays hit the tray at z {float(valid.min()):.3f}..{float(valid.max()):.3f}, raster {len(xs)}x{len(ys)} "
                                        f"at {res:.3f} m; {where})")
            local = C.transform_points(C.invert((rp, rq)), th.cat([world_xy, th.full((len(world_xy), 1), floor_z)], dim=1))
            rim = float(hi[2]) - float(rp[2])
            log.info("%s: inner floor %d cells at %.3f m (%.4f m2), floor %.3f m above the root, rim %.3f m", tray.name, len(world_xy), res, len(world_xy) * res * res,
                     float(local[:, 2].mean()), rim)
            return TrayFloor(local[:, :2].clone(), float(local[:, 2].mean()), res, rim, heights=(zs[keep] - float(floor_z)).clone())
        finally:
            s.world.park(tray, list(s.world.spawned).index(tray.name))


class TrayFloorWorld:
    """World view of a :class:`TrayFloor` at a tray pose: ``cells()`` (world xy), ``z`` (world floor height),
    ``polygon_supported(fp, margin)`` / ``contains(xy, margin)`` (§3.2 tray margin 0.5 cm), ``area()``, ``centre``."""

    def __init__(self, floor: TrayFloor, pose: TT.Pose):
        self.floor = floor
        self.pose = (TT._t(pose[0]), TT._t(pose[1]))
        self.inv = C.invert(self.pose)
        pts = C.transform_points(self.pose, th.cat([floor.cells_local, th.full((len(floor.cells_local), 1), floor.z_local)], dim=1))
        self._cells = pts[:, :2].clone()
        self.z = float(pts[:, 2].mean())
        self.centre = self._cells.mean(dim=0)
        self.rim_z = float(self.pose[0][2]) + floor.rim_local
        self.name = getattr(floor, "name", "tray_floor")

    def cells(self, free_only: bool = True) -> th.Tensor:
        return self._cells

    def area(self) -> float:
        return self.floor.area()

    def _local(self, xy) -> th.Tensor:
        p = th.tensor([float(xy[0]), float(xy[1]), self.z])
        return C.transform_points(self.inv, p.unsqueeze(0))[0, :2]

    def contains(self, xy, margin: float = 0.0) -> bool:
        return self.floor.contains_local(self._local(xy), margin)

    def polygon_supported(self, poly_xy: th.Tensor, margin: float = 0.005) -> bool:
        for p in poly_xy:
            if not self.contains(p, margin):
                return False
        return self.contains(poly_xy.mean(dim=0), 0.0)


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


@register(SUBTASK)
class Disembedding(SamplerV2):
    START_DRAW_FAMILY = True  # i7-startdraw: the start base per attempt under ISPATIALGYM_START_DRAW (spa_redraw_start)

    def __init__(self, ctx):
        super().__init__(ctx)
        if self.gp.bddl is None:
            self.definition_id = 0  # v1 writer: disembedding-0
        self.defaults = GD.defaults_for(SUBTASK)
        self.src_role = "source" if "source" in self.gp.roles else "tray_source"
        self.dst_role = "destination" if "destination" in self.gp.roles else "tray_dest"
        self.top = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        self.height: Dict[str, float] = {}
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self._pad_kept: Optional[List[Any]] = None  # i7-tray: the pick's rows after the layout pad check (``pick_grasps``)
        self.edge: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.src_floor: Optional[TrayFloor] = None
        self.dst_floor: Optional[TrayFloor] = None
        self.band_spec: Optional[TT.BandSpec] = None
        self.bands = None
        self.segment_arms: List[str] = ["right"]  # the arms offered for the target's segment (sample_template)
        self.arm_offer_why = ""  # why the other arm is not offered for the template (``tabletop.arm_offer``)
        self.arm_bands: Dict[str, Any] = {}
        self.placed_key: Optional[str] = None  # band key of the stance-relative source tray (item 9)
        self.release_surface_z: Optional[float] = None  # raycast surface under the destination goal (owner item 3)
        self.layout: Optional[Dict[str, Tuple[float, float]]] = None  # fixed layout: bddl instance -> figure-frame (x, y)
        self.layout_yaws: List[float] = [0.0]  # the yaws the fixed layout may turn by (sample_template)
        self.tray_turns: List[float] = []  # those of them the edge-aligned source floor cannot hold: the tray turns with the layout
        self.dest_side: Optional[float] = None  # the destination tray's side (sample_template): +1 the robot's left, -1 its right

    # ------------------------------------------------------------------ parameters (§2.3: numbers from generation)

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    def target_equivalences(self) -> Optional[List[List[float]]]:
        """The target's pose-target ``orientation_equivalences`` (§3.13: the rotation group of the customs ``roles.target.
        orientation_equivalences`` tags); None (today's identity-only goal) unless ``generation.target_orientation_equivalences``.

        oracle-fix 2026-11-06 (survey1 disembedding_07, cellulose tape ["x2", "zinf"]: 2 of 4 demos symbolic_only at 0.26 / 0.32 rad, the
        roll turned 15.5 deg about its own axis by the opening fingers at the release - a turn the tags say leaves the roll unchanged)."""
        if not self.gen("target_orientation_equivalences", False):
            return None
        mc = self.choice.models["target"]
        return self.gp.orientation_equivalences("target", mc.category, mc.model) or None

    # ------------------------------------------------------------------ roles

    def v1_role_variants(self) -> Dict[str, str]:
        out = {}
        if self.src_role != "tray_source":
            out["tray_source"] = self.src_role
        if self.dst_role != "tray_dest":
            out["tray_dest"] = self.dst_role
        return out

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.variant}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    def support_objects(self) -> List[Any]:
        """The table and both trays (ignored for the final grasp / place pose only, §3.6); bound roles only (parking runs
        before the roles are bound)."""
        return [self.obj(role) for role in ("support", self.src_role, self.dst_role) if self.bindings.has(role)]

    def grasp_ignore_list(self) -> List[Any]:
        """What the target's grasp ignores at the grasp pose (§3.6): the table, the source tray and - owner 2026-10-12 (A2b) - every
        distractor / fixed-layout part, which the grasp may touch or displace as long as it stays on the source tray (recorded as
        ``grasps[].may_touch``: the program check ignores them for the grasp and the lift, the demo checks the tray after both)."""
        return [self.obj("support"), self.obj(self.src_role)] + [self.objects[b.name] for b in self.bindings.many("distractors")]

    def layout_parts(self) -> List[Any]:
        """The fixed layout's parts around the target (the puzzle's figure), none for a pool draw."""
        return [self.objects[b.name] for b in self.bindings.many("distractors")] if self.layout is not None else []

    def pick_world(self):
        """Owner 2026-10-12 (A2b): the fixed-layout parts out of the collision world for every pose of the target's pick - pre-grasp,
        approach and grasp, its feasible set F included (``grasps[].may_touch.pre_grasp``); nothing changes for a pool draw. Every pick
        check of a fixed layout runs inside it: F's cache key does not record the context."""
        return self.ikc.ignoring(self.layout_parts())

    @property
    def pick_grasps(self) -> List[Any]:
        """The rows the target's pick draws from: ``self.grasps``, without the rows whose fingers stand on a layout part when
        ``ISPATIALGYM_LAYOUT_PAD_CHECK`` is on (:meth:`_layout_pad_screen`, i7-tray 2026-11-08)."""
        kept = getattr(self, "_pad_kept", None)  # a sampler built without __init__ (tests) has none
        return kept if kept is not None else self.grasps

    def _part_pieces_in(self, name: str, pose, target_pose) -> List[HullObstacle]:
        """The convex collision pieces of the layout part ``name`` at the layout pose ``pose``, in the frame of the target at
        ``target_pose`` (the part's pieces read off its body wherever it is, relative to its root)."""
        o = self.objects[name]
        backend = getattr(getattr(getattr(self, "ikc", None), "kit", None), "backend", None) or getattr(getattr(self, "kit", None), "backend", None)
        root = SS._root(o)
        if backend is None or root is None:
            return []
        T_rel = np.linalg.inv(pose_mat(target_pose)) @ pose_mat(pose)
        out = []
        for path, cm in SS._pieces(o):
            v, _f, (pp, Rp) = SS.piece_geometry(backend, cm)
            loc = SS.to_local(root, v @ Rp.T + pp)
            out.append(HullObstacle.of_points(loc @ T_rel[:3, :3].T + T_rel[:3, 3], f"{name}/{path.rsplit('/', 1)[-1]}"))
        return out

    def _layout_pad_screen(self, poses: Dict[str, Any]) -> None:
        """i7-tray 2026-11-08: the layout pad check (``LAYOUT_PAD_SWITCH``) once per template - per library row the fingers' clearance to
        the layout's parts (:func:`layout_pad_clearance`) at the layout's relative poses ``poses``; a row whose fingers go deeper than the
        policy's ``finger_support_max_penetration_m`` into a part would stand on it; ``all`` drops such rows, ``heavy`` only those standing
        on a part at least as heavy as the target. Counters ``layout_pad.rows`` / ``.stand_on`` / ``.stand_on.<model>`` / ``.stand_on_heavy``
        and ``.dropped``; one INFO line with every row's clearance per part. Switch off: measured only."""
        if getattr(self, "_pad_screened", False) or self.layout is None:
            return
        self._pad_screened = True
        from ispatialgym.grasping import contact_policy as CP

        try:
            tpose = poses[self.tname]
            parts = {n: self._part_pieces_in(n, p, tpose) for n, p in poses.items() if n != self.tname}
            per = layout_pad_clearances(self.grasps, parts, self.segment_arms[0] if self.segment_arms else "right")
            t_mass = float(getattr(self.obj("target"), "mass", 0.0) or 0.0)
            heavy = {n for n in parts if float(getattr(self.objects[n], "mass", 0.0) or 0.0) >= t_mass > 0.0}
        except Exception as e:  # noqa: BLE001 - a measurement failure keeps the pick as before
            log.info("%s: LAYOUT_PAD check unavailable (%r): the pick keeps every row", self.gp.variant, e)
            TT._count(self, "layout_pad.unavailable")
            return
        allow = float(CP.DEFAULT.finger_support_max_penetration_m)
        model_of = {b.name: b.model for b in self.bindings.many("distractors")}
        on_parts = {n: sorted(p for p, d in per[n].items() if d < -allow - 1e-9) for n in per}
        bad_all = {n for n, ps in on_parts.items() if ps}
        bad_heavy = {n for n, ps in on_parts.items() if any(p in heavy for p in ps)}
        TT._countn(self, "layout_pad.rows", len(per))
        for n in sorted(bad_all):
            TT._count(self, "layout_pad.stand_on")
            for p in on_parts[n]:
                TT._count(self, f"layout_pad.stand_on.{model_of.get(p, p)}")
        TT._countn(self, "layout_pad.stand_on_heavy", len(bad_heavy))
        mode = layout_pad_mode()
        bad = bad_all if mode == "all" else (bad_heavy if mode == "heavy" else set())
        kept = [g for g in self.grasps if str(getattr(g, "name", "")) not in bad]
        fmt = lambda n: " ".join(f"{model_of.get(p, p)} {d * 1000:+.1f}" for p, d in per[n].items())  # noqa: E731
        log.info("%s: LAYOUT_PAD (%s) finger clearance (mm) to the layout parts at the pre-shape along the approach: %s; heavy parts (>= the "
                 "target's %.3f kg) %s -> %d of %d rows %s", self.gp.variant, mode if mode != "off" else "off: measured only",
                 "; ".join(f"{n} [{fmt(n)}]" for n in per), t_mass, sorted(model_of.get(p, p) for p in heavy), len(kept), len(self.grasps),
                 "kept" if mode != "off" else "drawn (nothing dropped)")
        if bad:
            if not kept:
                raise Reject("grasp_unreachable", f"layout pad check ({mode}): every row's fingers stand on a layout part ({sorted(bad)})")
            TT._countn(self, "layout_pad.dropped", len(bad))
            self._pad_kept = kept

    def spawn_roles(self):
        # owner decision 2026-09-16 4b: the source tray spawns at ``roles.source.scale`` (D1 writes 1.25; until it lands the
        # ``source_tray_scale`` default), the destination tray at 1.0. The scaled choice replaces the template's so the band
        # key, the hulls and the spawn agree
        mc = self.choice.models.get(self.src_role)
        if mc is not None and not mc.scale:
            s = self._source_scale(mc)
            if s is not None and abs(s - 1.0) > 1e-6:
                self.choice.models[self.src_role] = dataclasses.replace(mc, scale=[float(s)] * 3)
                self.template_meta["source_tray_scale"] = float(s)
        super().spawn_roles()
        if not self.bindings.has("distractors"):
            self._spawn_facade_distractors()

    def _source_scale(self, mc) -> Optional[float]:
        """Uniform spawn scale of the source tray: ``roles.<source>.scale`` of the customs entry (per model, or a scalar /
        list for the role), else the ``source_tray_scale`` generation default (1.25)."""
        role = self.gp.roles.get(self.src_role)
        raw = (self.gp.entry.get("roles") or {}).get(self.src_role, {}).get("scale") if isinstance(getattr(self.gp, "entry", None), dict) else None
        val = None
        if role is not None:
            try:
                val = role.scale_of(mc.category, mc.model)
            except AttributeError:
                val = None
        if val is None and raw is not None and not isinstance(raw, dict):
            val = raw
        if val is None:
            val = self.g("source_tray_scale")
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        return None if val is None else float(val)

    def _tray_budget(self) -> float:
        """§6.4 distractor budget: D1's ``generation.distractor_budget`` (the grounding's maths: its ``budget_fraction`` /
        ``budget`` / ``fraction`` value, or a bare number) when present, else ``generation.tray_budget`` (default
        ``generation_defaults``)."""
        db = self.gen("distractor_budget", None)
        if isinstance(db, dict):
            for k in ("budget_fraction", "budget", "fraction", "max_fraction", "tray_budget"):
                if db.get(k) is not None:
                    return float(db[k])
        elif isinstance(db, (int, float)):
            return float(db)
        return float(self.g("tray_budget"))

    def _spawn_facade_distractors(self):
        """v1 facade only (no D1 ``distractors`` group in the template choice): draw the set from the v1 spec's
        ``distractors`` pools per §3.12 rule 1 — the target's category pool (or the other-category pool) minus the target
        and its look-alike group, at most one model per look-alike group, ``size`` ∈ generation.groups.distractors."""
        v1 = TT.facade_v1_spec(self)
        target = self.choice.models["target"]
        dspec = v1.distractor_spec() if v1 is not None else None
        if dspec is None:
            raise Reject("spec", f"{self.gp.variant}: no distractors group in the template choice and no v1 distractor pools")
        rng = np.random.default_rng(self.ctx.base_seed * 31 + 17)
        same = rng.random() < float(self.gen("same_category_prob", 0.5))
        pools = {"same_category": dspec.same_category_choices(target), "other_category": dspec.other_category_choices(target)}
        pool = pools["same_category" if same else "other_category"] or pools["other_category" if same else "same_category"]
        if not pool:
            raise Reject("spec", f"both distractor pools are empty for {target.category}/{target.model} after the look-alike exclusions")
        groups = [[tuple(m) for m in g] for g in dspec.lookalike_groups]
        cands = TT.exclude_lookalikes([(c.category, c.model) for c in pool], (target.category, target.model), groups, rng)
        size = self.gp.group_spec("distractors").get("size", self.defaults["groups"]["distractors"]["size"])
        n = int(rng.integers(int(size[0]), int(size[1]) + 1)) if isinstance(size, (list, tuple)) else int(size)
        picks = cands[: min(n, len(cands))]
        if not picks:
            raise Reject("spec", "no distractor candidate survives the look-alike exclusion")
        by_tag = {(c.category, c.model): c for c in pool}
        for cm in picks:
            self.spawn_role("distractors", by_tag[cm])
        self.template_meta["distractors"] = {"pool": "same_category" if same else "other_category", "models": [f"{c}/{m}" for c, m in picks],
                                             "pool_size": len(pool), "n_after_lookalikes": len(cands)}

    # ------------------------------------------------------------------ template

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    def sample_template(self):
        tgt, src, dst = self.obj("target"), self.obj(self.src_role), self.obj(self.dst_role)
        # every hull below is read off a parked body: re-park first (consistent PhysX / render-side transforms, gravity
        # off again after the spawns' stop / play), see ``repark_spawned``
        repark_spawned(self)
        # owner 2026-09-25: the trays get ``tray_mass_kg`` (2 kg; 0 = the dataset's category mass) - a 0.39 kg tray under 4 kg
        # of bottles is a 10:1 inverted mass ratio that PhysX resolves with contact chatter (``constraints.apply_mass``);
        # recorded per instance as a TRO ``constraints`` record so the demo loads the same physics
        self.tray_mass = {}
        for key, o in ((self.src_role, src), (self.dst_role, dst)):
            self.tray_mass[o.name] = apply_mass(o, float(self.g("tray_mass_kg") or 0.0))
            if abs(self.tray_mass[o.name]["after_kg"] - self.tray_mass[o.name]["before_kg"]) > 1e-6:
                log.info("%s: %s (%s) mass %.3f -> %.3f kg (generation.tray_mass_kg)", self.gp.variant, o.name, key, self.tray_mass[o.name]["before_kg"], self.tray_mass[o.name]["after_kg"])
        self.template_facts["tray_mass_kg"] = {n: round(v["after_kg"], 4) for n, v in self.tray_mass.items()}
        self.grasps, self.grasp_key, self.rest["target"] = TT.library_grasps(self, "target", "tabletop")
        self.layout = self._load_layout()
        for key, o in (("target", tgt), (self.src_role, src), (self.dst_role, dst)):
            self.hull[key] = self.world.local_hull(o)
            self.rest.setdefault(key, C.IDENTITY_QUAT.clone())
        if self.layout is not None:
            # every part of the layout keeps the figure frame's orientation (assembly.json): the target lies as modelled
            self.rest["target"] = C.IDENTITY_QUAT.clone()
        else:
            # owner 2026-09-24: a lying prism-hulled object (the bottles) is placed rolled onto a facet of its hull, not on an edge
            # (``TT.prism_rest_quat``): bottle_of_milk/mrejrs at its canonical roll stood on two vertex lines 2 mm apart beside its
            # mass centre and tipped ~7 deg at every release, then kept rolling
            # of-v2maps 2026-11-07 (prompt I item 2): a v2 entry's rest is MEASURED (settled in physics, facet-down by construction) - the
            # facet snap is for the legacy canonical guess and would move the object off the rest its grasps and h_root belong to
            roll = 0.0
            if TT.v2_rest_info(self, self.hull["target"], self.rest["target"]) is None:
                self.rest["target"], roll = TT.prism_rest_quat(self.hull["target"], self.rest["target"])
            if roll:
                log.info("%s: target rests on a hull facet at roll %+.2f deg from its library rest orientation", self.gp.variant, math.degrees(roll))
        for key in ("target", self.src_role, self.dst_role):
            self.height[key] = TT.height_of(self.hull[key], self.rest[key])
        # handoff 2026-10-10: on the milk picks the distractors take the target's pose (generation.distractor_pose, one pose or
        # per "<category>/<model>"); the other picks keep each model's canonical rest pose
        pose_rule = self.gen("distractor_pose", None)
        rests: Dict[str, str] = {}
        for b in self.bindings.many("distractors"):
            o = self.objects[b.name]
            self.hull[b.name] = self.world.local_hull(o)
            if self.layout is not None:
                self.rest[b.name], how, roll = C.IDENTITY_QUAT.clone(), "fixed_layout", 0.0
            else:
                pose = pose_rule.get(f"{b.category}/{b.model}") if isinstance(pose_rule, dict) else pose_rule
                base, how = pose_rest_quat(self.hull[b.name], pose)
                self.rest[b.name], roll = TT.prism_rest_quat(self.hull[b.name], base)
            if roll:
                log.info("%s: %s rests on a hull facet at roll %+.2f deg", self.gp.variant, b.name, math.degrees(roll))
            if how != "canonical":
                rests[b.name] = how
                log.info("%s: %s (%s/%s) rests %s", self.gp.variant, b.name, b.category, b.model, how)
            self.height[b.name] = TT.height_of(self.hull[b.name], self.rest[b.name])
        if rests:
            self.template_facts["distractor_rest"] = rests
        support = self.obj("support")
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        res = float(self.g("tray_raster_m"))
        self.src_floor = TrayFloor.measure(self, src, self.top, res)
        self.dst_floor = TrayFloor.measure(self, dst, self.top, res)
        if os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG"):
            from pathlib import Path
            from scripts.campaign_20261110.contracts import write_json_atomic

            def area_record(binding, hull_key):
                obj = self.objects[binding.name]
                quat = self.rest[hull_key]
                area = CTG.polygon_area(self.world.footprint(self.hull[hull_key], (th.zeros(3), quat)))
                return {"instance": binding.name, "category": binding.category, "model": binding.model,
                        "area_m2": area, "scale": [float(v) for v in obj.scale],
                        "rest_quat_xyzw": [float(v) for v in quat]}

            profile = {"schema": "isg-native-tray-coverage-profile-v1", "template_id": self.ctx.template_id,
                "source_template": self.ctx.out_dir, "bddl_sha256": self.template_meta["bddl_sha256"],
                "target": area_record(self.bindings.one("target"), "target"),
                "distractors": [area_record(b, b.name) for b in self.bindings.many("distractors")],
                "source_floor_area_m2": self.src_floor.area(), "source_floor_cells": self.src_floor.n_cells,
                "source_floor_resolution_m": self.src_floor.res, "fixed_layout": self.layout is not None,
                "source": "native projected collision hull at measured rest and actual initialized scale"}
            self.template_facts["campaign_coverage_profile"] = profile
            write_json_atomic(Path(self.ctx.out_dir) / "campaign_coverage_profile.json", profile)
            log.info("CAMPAIGN_COVERAGE_PROFILE %s", profile)
        # i7-split 2026-11-07: the support split's planes - the table top (a map hit registers it again for the policy) and both trays' inner
        # floors (split only: the policy does not police the trays) - so the trays' rims / walls stay obstacles wherever they are ignored
        SS.register_top(self, support, self.top, what="top")
        for tray, role, fl in ((src, self.src_role, self.src_floor), (dst, self.dst_role, self.dst_floor)):
            fl.register_split(self.ikc, tray, float(self.top.z) + self.world.bottom_offset(self.hull[role], C.IDENTITY_QUAT) + fl.z_local)
        self.kit.sync_world()
        if self.layout is not None:
            # owner 2026-10-12 (the puzzle): the layout's yaws, and those at which the source tray turns with it
            self.layout_yaws = layout_yaws(float(self.g("fixed_layout_yaw_step_deg")))
            self.tray_turns = self._layout_turns()
            self.template_facts["fixed_layout"].update({"yaws_deg": [round(math.degrees(v), 2) for v in self.layout_yaws],
                                                        "tray_turn_yaws_deg": [round(math.degrees(v), 2) for v in self.tray_turns]})
        # band: the target grasped at the source-floor height; the tray moves with the focus, so the lattice is the top
        # itself (the tray walls are ignored for the grasp pose per §3.6; the pre-grasp is 10 cm above the rim in practice)
        mc = self.choice.models["target"]
        src_mc = self.choice.models[self.src_role]
        # band key (item 9, B3's ``reach_bands.PlacedSpec`` grammar ``CATEGORY/MODEL[*SCALE][@MODE]``): the source tray at
        # its spawn scale, mode ``focus_centred_edge_aligned`` = inner-floor centroid on the manipulation focus cell, long
        # axis along the edge — stance-relative, so every template of the variant hits the same band
        src_scale = float(src_mc.scale[0]) if src_mc.scale else 1.0
        mode = getattr(TT.RB, "PLACED_MODE_FOCUS", "focus_centred_edge_aligned")
        spec_obj = getattr(TT.RB, "PlacedSpec", None)
        if spec_obj is not None:
            self.placed_key = spec_obj.parse((f"{src_mc.category}/{src_mc.model}", src_scale if abs(src_scale - 1.0) > 1e-6 else None, mode)).canonical()
        else:
            self.placed_key = f"{src_mc.category}/{src_mc.model}" + (f"*{src_scale:g}" if abs(src_scale - 1.0) > 1e-6 else "") + f"@{mode}"
        self.band_spec = TT.BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull["target"],
                                     rest_quat=self.rest["target"], scale=list(mc.scale) if mc.scale else None, placed=[self.placed_key],
                                     z_support=self.top.z + self.src_floor.z_local + self.world.bottom_offset(self.hull[self.src_role], C.IDENTITY_QUAT),
                                     height=self.height["target"])
        H_max = max(self.height.values())
        rng = np.random.default_rng(self.ctx.base_seed)
        side = self._destination_side(rng)
        self.dest_side = 1.0 if side == "left" else -1.0
        # the source tray's half length along the edge: its longer footprint side, the axis ``_tray_yaw`` lays along the edge (the
        # longest of its turns with a fixed layout)
        margins = {"kind": "disembedding", "lateral_margin": float(self.g("lateral_margin_m")), "safety": 0.02, "H_max": H_max,
                   "destination_side": side, "source_half_length": self._source_half_length()}
        self._start_margins = margins
        errors = []
        self._start_search(rng, margins, errors)
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "band": getattr(self.bands, "path", None), "placed_key": self.placed_key, "source_tray_scale": src_scale,
                                    "src_floor_cells": self.src_floor.n_cells, "dst_floor_cells": self.dst_floor.n_cells,
                                    "H_max": H_max, "distractors": [b.name for b in self.bindings.many("distractors")], "segment_arms": list(self.segment_arms)})

    def _start_search(self, rng: np.random.Generator, margins: Dict[str, Any], errors: List[str]) -> bool:
        """The start search of the template (factored out 2026-11-08, i7-startdraw; the code as it was): every edge with a free-floor run
        (``rng`` order), the init trunk (Phase 1b), the lateral (Phase 1c: floor, collision, trunk sweep; the margins hold the template's
        destination side), the segment arms, the Phase-2 survivors with the source tray in frame and the destination tray fitting beside
        it. Sets ``edge`` / ``init_stance`` / ``manip_candidates`` and returns True on the first edge that passes; failures go to
        ``errors``. ``sample_template`` runs it with the template seed, :meth:`spa_redraw_start` with the attempt's rng."""
        H_max = float(margins["H_max"])
        if os.environ.get(CTG.ENV):
            self._campaign_tray_active = self.campaign_tray_plan()
        from ispatialgym.sampling import campaign_tabletop_starts as CTS
        anchor = getattr(getattr(self, "_campaign_tray_active", None), "anchor_base_pose2d", None)
        reference = CTG.configured_reference_start()
        if anchor is None and reference is not None:
            anchor = reference['initial_base_pose2d']
        # §4.2 has no preference among the edges and §4.4 checks floor + collision per lateral position (Phase 1c), so
        # every edge with a free-floor run in front of it is tried (a third of its 10 cm lateral raster at s = 0.45 on
        # free floor, as the IORS reference task does) instead of ``edge_candidates``, which drops an edge when its
        # *midpoint* stance alone is blocked: on breakfast_table_rhjoby_0 that is the 2.9 m edge — the only one long
        # enough for the two-sided disembedding margins of the time (m_R + m_L ≈ 2.26 m) — whose midpoint sits on a 0.4 m patch the scene
        # variant leaves blocked in the traversability map (baked chair footprint wider than the freed bbox)
        for edge, st in CTS.ordered_edges(TT.edge_frames(self, self.top, rng), anchor):
            lats = th.linspace(-edge.length / 2 + 0.1, edge.length / 2 - 0.1, max(2, int(edge.length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if self.kit.trav_free(edge.base_xy(st.stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                errors.append(f"edge {edge.name}: free floor at {n_free}/{len(lats)} lateral positions")
                log.info("%s: %s", self.gp.variant, errors[-1])
                continue
            # A fixed tray layout need not be visible from the first lateral
            # that clears the floor. Retry native start draws, bounded by the
            # group deadline, while retaining every phase-1/phase-2 check.
            draws = 4 if getattr(self, "_campaign_tray_active", None) is not None else 1
            for _ in range(draws):
                if draws > 1:
                    self.campaign_group_remaining_s()
                try:
                    bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                    init = TT.phase1b_init_trunk(self, st, H_max, bands=bands, top=self.top)
                    lateral = None
                    if anchor is not None:
                        bounds, _, _ = TT.phase1c_bounds(self, init, margins, top=self.top, strict=True)
                        lateral = CTS.near_lateral(rng, float(bounds[0]), float(bounds[1]), CTS.edge_lateral(edge, anchor))
                    init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands, lateral=lateral)
                    init.name = "init"
                    # the arms offered for the grasp -> carry -> place -> release segment (owner 2026-09-27, first-demo v2.2): the other arm's
                    # band is its own or this band mirrored
                    self.segment_arms, self.arm_offer_why = TT.arm_offer(self)
                    self.arm_bands = TT.arm_bands_for(self, self.segment_arms, bands=bands) if len(self.segment_arms) > 1 else {}
                    cands = TT.phase2_manipulation_stance(self, init, TT.half_extent(self.hull[self.src_role], C.IDENTITY_QUAT), target_height=self.height[self.src_role],
                                                          in_frame_fn=self._source_tray_in_frame, bands=bands, top=self.top, focus_z=self.top.z + self.src_floor.z_local,
                                                          arm_bands=self.arm_bands, angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self),
                                                          keep_pairs=lambda cs: self._destination_fit(init, cs))
                    if not cands:
                        fits = [f for f in self.template_facts.get("destination_fit", []) if f["edge"] == edge.name]
                        raise Reject("no_manipulation_stance", "Phase 2 left no survivor with the source tray in frame" if not fits else
                                     f"Phase 2: the destination tray fits beside the source tray at none of the positions ({fits[-1]['positions']} positions, "
                                     f"{fits[-1]['candidates_before']} candidates; e.g. {(fits[-1]['examples'] or [''])[0][:300]})")
                    self.edge, self.init_stance, self.manip_candidates = edge, init, cands
                    return True
                except Reject as e:
                    errors.append(f"edge {edge.name}: {e}")
                    log.info("%s: %s", self.gp.variant, errors[-1][:300])
                    if e.reason == "edge_too_short":
                        break
        return False

    def spa_redraw_start(self, rng: np.random.Generator) -> None:
        """i7-startdraw (owner 2026-11-08 3.5, ``ISPATIALGYM_START_DRAW=1``): this attempt's own edge, init stance and Phase-2 pool from the
        attempt's ``rng`` - the template's start search (:meth:`_start_search`) with every check it makes; the destination side stays the
        template's (an eval variant's assignment). A search that finds none restores the template's edge / segment arms. Off: the
        template's start."""
        if not start_draw_active(self):
            return
        errors: List[str] = []
        keep = (self.edge, self.init_stance, self.manip_candidates, self.segment_arms, self.arm_offer_why, self.arm_bands)
        if not self._start_search(rng, self._start_margins, errors):
            self.edge, self.init_stance, self.manip_candidates, self.segment_arms, self.arm_offer_why, self.arm_bands = keep
            raise Reject("no_stance", "start_draw: " + "; ".join(errors)[:1500])
        self._spa_facts = start_facts(self.edge, self.init_stance, self.manip_candidates)

    def _destination_side(self, rng) -> str:
        """Owner 2026-10-12 (A2a): the destination tray's side, chosen before the Phase-1c margins - ``generation.destination_side``
        ("left" / "right", the eval variants' assignment), else drawn from the template's ``rng`` (train)."""
        want = self.g("destination_side")
        plan = self.campaign_tray_plan() if os.environ.get(CTG.ENV) else None
        fixed = getattr(plan, "destination_side", None)
        if fixed is not None:
            if want is not None and str(want).lower() != fixed:
                raise Reject("campaign_tray_side", "fixed anchor side differs from the explicit template side")
            side, source = fixed, "campaign_fixed_tray_anchor"
        elif want is None:
            side, source = ("left" if rng.random() < 0.5 else "right"), "drawn"
        elif str(want).lower() in ("left", "right"):
            side, source = str(want).lower(), "generation.destination_side"
        else:
            raise Reject("spec", f"{self.gp.variant}: generation.destination_side must be 'left' or 'right', got {want!r}")
        self.template_facts["destination_side"] = {"side": side, "source": source}
        return side

    def _destination_fit(self, init: Stance, cands: List[Stance]) -> List[Stance]:
        """Owner 2026-10-12 (A2 fix a): the destination's fit decides the source tray's position - of the Phase-2 (stance, position)
        candidates only those for which :meth:`_place_destination` (the init camera, the template's destination side) places the
        destination beside the source tray centred on the position survive (Phase 2's ``keep_pairs``: none kept counts as none found, so
        the split cone's fallback runs). Phase 1c's reserve is measured around the init stance; the source tray sits on the work
        stance's focus (variant C shifts it up to 0.3 m along the edge, deeper foci widen the image border) and nothing else ties
        its position to the destination's room. The verdict depends on the position alone (the stances of one edge share its
        tangent), so each position is checked once; ``template_facts.destination_fit`` gets one record per call."""
        cam = self.camera_pose_at(init)
        verdict: Dict[Tuple[float, float], Optional[str]] = {}
        keep: List[Stance] = []
        for c in cands:
            key = (round(float(c.focus[0]), 3), round(float(c.focus[1]), 3))
            if key not in verdict:
                try:
                    campaign_plan = getattr(self, "_campaign_tray_active", None)
                    for s_pose in self._source_tray_poses(c, c.focus[:2]):
                        if campaign_plan is None:
                            self._place_destination(None, init, c, cam, s_pose)
                        else:
                            self.campaign_tray_geometry(c, cam, s_pose, tuple(TT._t(v) for v in campaign_plan.destination))
                    verdict[key] = None
                except Reject as e:
                    verdict[key] = str(e)
            if verdict[key] is None:
                keep.append(c)
        failed = [why for why in verdict.values() if why is not None]
        fact = {"edge": init.edge.name, "positions": len(verdict), "fit": len(verdict) - len(failed), "candidates_before": len(cands),
                "candidates_after": len(keep), "examples": failed[:3]}
        self.template_facts.setdefault("destination_fit", []).append(fact)
        log.info("%s: destination fit (%s side, edge %s): %d of %d positions, %d of %d (stance, position) candidates kept%s", self.gp.variant,
                 "left" if self.dest_side > 0 else "right", init.edge.name, fact["fit"], fact["positions"], len(keep), len(cands),
                 f" (e.g. {failed[0][:240]})" if failed else "")
        return keep

    def _tray_yaw(self, key: str, stance: Stance) -> float:
        """Yaw putting the tray's longer footprint axis along the front edge (stance tangent)."""
        ext = TT.hull_extents(self.hull[key], C.IDENTITY_QUAT)
        tangent_yaw = math.atan2(float(stance.tangent[1]), float(stance.tangent[0]))
        return tangent_yaw if float(ext[0]) >= float(ext[1]) else tangent_yaw - math.pi / 2

    def _tray_pose_centred(self, key: str, floor: TrayFloor, stance: Stance, centre_xy, turn: float = 0.0) -> TT.Pose:
        """Tray pose with its inner-floor centre at ``centre_xy`` and its long axis along the edge (turned by ``turn`` about the
        vertical when a fixed layout turns it)."""
        yaw = self._tray_yaw(key, stance) + float(turn)
        quat = C.quat_from_yaw(yaw)
        c_local = floor.cells_local.mean(dim=0)
        off = C.quat_rotate(quat, th.tensor([float(c_local[0]), float(c_local[1]), 0.0]))
        pos = th.tensor([float(centre_xy[0]) - float(off[0]), float(centre_xy[1]) - float(off[1]), self.top.z + self.world.bottom_offset(self.hull[key], quat) + TT.Z_OFFSET])
        return pos, quat

    def _source_tray_poses(self, stance: Stance, centre_xy) -> List[TT.Pose]:
        """The source tray centred on ``centre_xy``: edge-aligned, and turned by each of the fixed layout's ``tray_turns`` - every pose
        an instance may give it, which the Phase-2 frame check and the destination fit hold for all."""
        campaign_plan = getattr(self, "_campaign_tray_active", None)
        if campaign_plan is not None:
            return [tuple(TT._t(v) for v in campaign_plan.source)]
        return [self._tray_pose_centred(self.src_role, self.src_floor, stance, centre_xy, t) for t in [0.0] + list(self.tray_turns)]

    def _source_half_length(self) -> float:
        """The source tray's half length along the edge for the Phase-1c margins: half its longer footprint side (the axis ``_tray_yaw``
        lays along the edge), the longest over the fixed layout's ``tray_turns``."""
        hull = self.hull[self.src_role]
        ext = TT.hull_extents(hull, C.IDENTITY_QUAT)
        axis = 0 if float(ext[0]) >= float(ext[1]) else 1
        return max(0.5 * float(TT.hull_extents(hull, C.quat_from_yaw(t))[axis]) for t in [0.0] + list(self.tray_turns))

    def _source_tray_in_frame(self, s, stance: Stance, cam, focus, info) -> bool:
        """§4.5 step 3 (disembedding): the whole source tray centred on the focus, long axis along the edge (and at each turn of a
        fixed layout's ``tray_turns``), on the top and in frame (5 %)."""
        for pose in self._source_tray_poses(stance, focus[:2]):
            if not self.top.polygon_supported(self.world.footprint(self.hull[self.src_role], pose), margin=TT.SUPPORT_MARGIN):
                if getattr(self, "_campaign_tray_active", None) is not None:
                    TT._count(self, "campaign_tray_frame.off_support")
                return False
            if not TT.in_frame(self, cam, TT.hull_corners_world(self.hull[self.src_role], pose), TT.IN_FRAME_MARGIN):
                if getattr(self, "_campaign_tray_active", None) is not None:
                    TT._count(self, "campaign_tray_frame.out_of_view")
                return False
        return True

    # ------------------------------------------------------------------ fixed layout (handoff 2026-10-10: the puzzle pick)

    def _load_layout(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """The figure-frame (x, y) of the target and of every distractor by bddl instance, from the layout file of
        ``generation.fixed_layout`` (ground_isg writes it from the sheet's ``fixed_layout`` rows); None for a variant without one."""
        spec = self.gen("fixed_layout", None)
        if not spec:
            return None
        rel = spec.get("file") or ""
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            raise Reject("spec", f"{self.gp.variant}: fixed layout file {rel!r} is missing")
        with open(path, "rb") as f:
            raw = f.read()
        if spec.get("sha256") and hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            log.warning("%s: %s changed since the grounding (sha256 %s...); its current layout is used", self.gp.variant, rel, str(spec["sha256"])[:12])
        origins = layout_origins(json.loads(raw.decode("utf-8")))
        want = {self.tname: self.choice.models["target"].model, **{b.name: b.model for b in self.bindings.many("distractors")}}
        missing = sorted({m for m in want.values() if m not in origins})
        if missing:
            raise Reject("spec", f"{self.gp.variant}: {rel} places no part {missing}")
        layout = {name: origins[m] for name, m in want.items()}
        self.template_facts["fixed_layout"] = {"file": rel, "origin_in_figure_m": {name: list(xy) for name, xy in layout.items()}}
        return layout

    def _layout_poses(self, stance: Stance, src_w: TrayFloorWorld, layout_yaw: float = 0.0) -> Tuple[Dict[str, TT.Pose], float]:
        """The layout on the source floor (handoff 2026-10-10): the figure's centre at the floor centre, +x the robot's right, +y
        away from the robot - turned by ``layout_yaw`` about the floor centre as a whole (owner 2026-10-12) -, every part flat at its
        (x, y), its footprint on the floor (``tray_footprint_margin_m``), else ``Reject("fixed_layout_placement")``. Returns the poses
        by bddl instance and the figure's yaw."""
        x_axis, y_axis, yaw = turned_axes(*figure_axes(stance.tangent, stance.inward), float(layout_yaw))
        margin = float(self.g("tray_footprint_margin_m"))
        poses: Dict[str, TT.Pose] = {}
        for name, (ox, oy) in self.layout.items():
            key = "target" if name == self.tname else name
            xy = src_w.centre + ox * x_axis + oy * y_axis
            pose = TT.object_pose_on(self, self.hull[key], self.rest[key], xy, yaw, src_w.z)
            if not src_w.polygon_supported(self.world.footprint(self.hull[key], pose), margin=margin):
                raise Reject("fixed_layout_placement", f"{name} at figure ({ox:+.3f}, {oy:+.3f}) m is not on the source floor with the {margin:.3f} m margin")
            poses[name] = pose
        return poses, yaw

    def _layout_turns(self) -> List[float]:
        """The ``layout_yaws`` at which the turned figure does not fit the edge-aligned source floor (owner 2026-10-12: the tray turns
        with the layout there, so the figure keeps its pose in the tray). Geometry in a frame of its own: an edge facing +x, the floor
        centre at the origin."""
        st = Stance("layout", (-0.5, 0.0, 0.0), 0.5, 0.0, 0.5, EdgeFrame("layout", th.zeros(2), th.tensor([1.0, 0.0]), 2.0))
        src_w = self.src_floor.at(self._tray_pose_centred(self.src_role, self.src_floor, st, th.zeros(2)))
        turns = []
        for yaw in self.layout_yaws:
            try:
                self._layout_poses(st, src_w, yaw)
            except Reject:
                turns.append(yaw)
        return turns

    def _layout_yaw_screen(self, manip: Stance, arms: List[str], target_poses: Dict[float, TT.Pose]) -> Dict[float, int]:
        """§4.6 step 1 over the layout's yaws: the screening IK of [pre-grasp, grasp] of every library pinch at the target's pose of each
        yaw (``target_poses``), every offered arm at its own stance (as :func:`tabletop.phase3_target`), the support, the source tray and
        the parts ignored at the grasp (the caller holds ``pick_world``). Returns the feasible (grasp, arm) pairs per yaw."""
        yaws = list(target_poses)
        n = {yaw: 0 for yaw in yaws}
        for a in arms:
            st = manip if (a == arms[0] and (not manip.working_arms or manip.working_arms == [a])) else TT.stance_for_arm(self, manip, a)
            if st is None:
                continue
            F = self.ikc.feasible_sets([th.zeros(2)], yaws, self.pick_grasps, lambda cell, yaw: target_poses[yaw], st, a, self.grasp_ignore_list(), cache=False)
            for yaw, _ in F.get(0, []):
                n[yaw] += 1
        return n

    def _layout_pick(self, rng, manip: Stance, arms_here: List[str]):
        """Owner 2026-10-12 (disembedding#01, the puzzle): the fixed layout turned as a whole about the vertical - the target with its
        neighbouring parts, relative poses unchanged - by a yaw drawn per instance (``rng``) among the ``layout_yaws`` whose screening
        (:meth:`_layout_yaw_screen`) keeps a reachable grasp, so the bars the pinches use can face a direction the arm reaches. The
        figure turns inside the edge-aligned source tray, or with the tray at the ``tray_turns``. The drawn yaw's pick is
        :func:`tabletop.phase3_target` at the target's pose (exact IK), the next kept yaw (without replacement) when it fails. Returns
        ``(target pose, grasp record, layout poses, source tray pose, its floor, the yaw record)``; ``Reject("grasp_unreachable")``
        when no yaw keeps a grasp - with the count of yaws that keep one with every object ignored (reach alone)."""
        tname, src = self.tname, self.obj(self.src_role)
        options: Dict[float, Tuple[TT.Pose, TrayFloorWorld, Dict[str, TT.Pose], float]] = {}
        for yaw in self.layout_yaws:
            campaign_plan = getattr(self, "_campaign_tray_active", None)
            s_pose = (self._tray_pose_centred(self.src_role, self.src_floor, manip, manip.focus[:2], yaw if yaw in self.tray_turns else 0.0)
                      if campaign_plan is None else tuple(TT._t(v) for v in campaign_plan.source))
            if not self.top.polygon_supported(self.world.footprint(self.hull[self.src_role], s_pose), margin=TT.SUPPORT_MARGIN):
                continue
            src_w = self.src_floor.at(s_pose)
            try:
                poses, fig_yaw = self._layout_poses(manip, src_w, yaw)
            except Reject:
                continue
            campaign_plan = getattr(self, "_campaign_tray_active", None)
            if campaign_plan is not None and CTG.pose_error(poses[tname], campaign_plan.goal)[1] > 0.001:
                continue
            options[yaw] = (s_pose, src_w, poses, fig_yaw)
        if options:
            self._layout_pad_screen(next(iter(options.values()))[2])  # i7-tray: once per template (the layout is rigid)
        targets = {yaw: o[2][tname] for yaw, o in options.items()}
        with self.pick_world():
            pairs = self._layout_yaw_screen(manip, arms_here, targets)
        kept = [yaw for yaw in options if pairs[yaw] > 0]
        deg = lambda v: round(math.degrees(v), 2)  # noqa: E731
        rec = {"file": (self.template_facts.get("fixed_layout") or {}).get("file"), "yaws_screened": len(options),
               "yaws_reachable_deg": [deg(v) for v in kept], "pairs_per_yaw": {str(deg(v)): pairs[v] for v in options}}
        log.info("%s: layout yaws with a reachable grasp: %d of %d %s", self.gp.variant, len(kept), len(options), rec["yaws_reachable_deg"])
        if not kept:
            with self.pick_world(), SS.whole_ignoring(self.ikc, self.support_objects()):  # i7-split: the supports out WHOLE (reach alone)
                reach = self._layout_yaw_screen(manip, arms_here, targets)
            raise Reject("grasp_unreachable", f"no layout yaw keeps a reachable grasp at the focus {[round(float(v), 3) for v in manip.focus[:2].tolist()]} "
                                              f"(0 of {len(options)} yaws x {len(self.pick_grasps)} grasps, arms {arms_here}; with every object ignored "
                                              f"{sum(1 for v in reach.values() if v)} yaws keep one: {'not ' if any(reach.values()) else ''}reach alone)")
        errors = []
        for i in rng.permutation(len(kept)):
            yaw = kept[int(i)]
            s_pose, src_w, poses, fig_yaw = options[yaw]
            src.set_position_orientation(*s_pose)
            src.keep_still()
            self._place_layout(poses)
            try:
                with self.pick_world():
                    t_pose, grasp = TT.phase3_target(self, manip, rng, self.pick_grasps, self.hull["target"], self.rest["target"], self.grasp_ignore_list(), top=src_w,
                                                     support_z=src_w.z, margin=float(self.g("tray_footprint_margin_m")),
                                                     scale_key=self.choice.models["target"].scale, bands=self.bands, arms=arms_here,
                                                     arm_bands=self.arm_bands or None, fixed=(poses[tname][0][:2], fig_yaw))
            except Reject as e:
                errors.append(f"yaw {deg(yaw)}: {str(e)[:200]}")
                continue
            rec.update({"yaw_deg": deg(yaw), "figure_yaw_rad": round(float(fig_yaw), 6), "centre_xy": [round(float(v), 4) for v in src_w.centre.tolist()],
                        "tray_turned": yaw in self.tray_turns, "exact_failed": errors})
            return t_pose, grasp, poses, s_pose, src_w, rec
        raise Reject("grasp_unreachable", f"{len(kept)} layout yaws keep a screened grasp, none passes the exact IK ({'; '.join(errors)[:600]})")

    def _place_layout(self, poses: Dict[str, TT.Pose]) -> None:
        """The layout's distractors at their poses (the target is placed by its grasp draw), the collision world synced: the target's
        grasp is screened with the figure around it."""
        for name, pose in poses.items():
            if name == self.tname:
                continue
            o = self.objects[name]
            o.set_position_orientation(TT._t(pose[0]), TT._t(pose[1]))
            o.keep_still()
        self.kit.sync_world()

    # ------------------------------------------------------------------ instance

    def campaign_tray_plan(self):
        if not os.environ.get(CTG.ENV):
            return None
        if not hasattr(self, "_campaign_group_started_at"):
            self._campaign_group_started_at = time.time()
        try:
            plan = CTG.load_tray_plan(template_id=self.ctx.template_id, scene=self.scene, rooms=self.choice.rooms,
                                     activated_at=self._campaign_group_started_at,
                                     expected_sha256=getattr(self, "_campaign_tray_plan_sha", None))
            if getattr(self, "_campaign_group_written", 0) >= plan.max_members:
                raise Reject("campaign_group_complete", "tray goal-group member limit reached")
            if (plan.target_name, plan.source_name, plan.destination_name) != (
                    self.tname, self.bindings.one(self.src_role).name, self.bindings.one(self.dst_role).name):
                raise Reject("campaign_goal_plan", "tray plan role bindings differ from the template")
            self._campaign_tray_plan_sha = plan.sha256
            return plan
        except CTG.GoalGroupTimeout as exc:
            raise Reject("campaign_group_timeout", str(exc)) from exc
        except (CTG.GoalPlanError, OSError, ValueError) as exc:
            raise Reject("campaign_goal_plan", str(exc)) from exc

    def campaign_group_remaining_s(self):
        plan = self.campaign_tray_plan()
        return None if plan is None else max(0.0, plan.deadline_at - time.time())

    def campaign_tray_geometry(self, manip, cam, source, destination):
        """Judge pinned trays directly; no random proposal must rediscover them."""
        footprints = {}
        for role, p in ((self.src_role, source), (self.dst_role, destination)):
            fp = self.world.footprint(self.hull[role], p)
            if not self.top.polygon_supported(fp, margin=TT.SUPPORT_MARGIN):
                raise Reject("campaign_tray_support", f"fixed {role} footprint leaves the support")
            # Both tray axes follow the rectangular table edge; either 180-degree
            # direction is equivalent for the tray, without imposing target yaw.
            desired = self._tray_yaw(role, manip)
            error = min(abs(C.wrap_angle(float(C.yaw_of(TT._t(p[1]))) - desired + turn)) for turn in (0, math.pi))
            if error > .001:
                raise Reject("campaign_tray_alignment", f"fixed {role} is not aligned to this table edge")
            footprints[role] = fp
        # _place_destination aligns the object origins at the same edge depth.
        # Floor-mask centres can be offset differently within the two assets;
        # comparing those centres would reject the native placement itself.
        depth_s, lat_s = manip.local(source[0])
        depth_d, lat_d = manip.local(destination[0])
        if abs(depth_s - depth_d) > .005:
            raise Reject("campaign_tray_alignment", "tray origins have different distances from the table edge")
        side = 1.0 if lat_d > lat_s else -1.0
        if side != self.dest_side:
            raise Reject("campaign_tray_side", "fixed destination is on the opposite template side")
        gap = TT.polygon_gap(footprints[self.src_role], footprints[self.dst_role])
        if gap < float(self.g("destination_gap_min_m")):
            raise Reject("campaign_tray_gap", "fixed trays are too close")
        # Goal-first placement keeps at least the existing visible-width rule.
        # The legacy bisection selected the limiting width after choosing start;
        # here each proposed start camera is checked against the pinned tray.
        fp = footprints[self.dst_role]
        laterals = (fp - manip.base_xy) @ manip.tangent
        width = float(self.g("destination_visible_width_m"))
        inner = float(laterals.min()) if side > 0 else float(laterals.max())
        xy = manip.world(depth_d, inner + side * width)
        z = float(TT.hull_corners_world(self.hull[self.dst_role], destination)[:, 2].max())
        cm = self.world.camera_model
        uv, depth = cm.project(cam, th.tensor([[float(xy[0]), float(xy[1]), z]]))
        if float(depth[0]) <= .05 or not 0 <= float(uv[0, 0]) <= cm.width or not 0 <= float(uv[0, 1]) <= cm.height:
            raise Reject("campaign_destination_not_visible", "fixed destination does not expose the required width from this start")
        return destination, side, {"source": "goal_plan", "gap_m": float(gap), "visible_width_min_m": width,
                                   "alignment_basis": "object_origins", "edge_depth_residual_m": abs(depth_s - depth_d)}

    def campaign_verify_goal(self, draft, *, settled=False):
        plan = getattr(self, "_campaign_tray_active", None)
        if plan is None:
            return
        self.campaign_group_remaining_s()
        destination = draft.object_poses[plan.destination_name]
        dp, dq = CTG.pose_error(destination, plan.destination)
        position_tol, orientation_tol = self.tol("position", 0.03), self.tol("orientation", 0.20)
        group = draft.summary["campaign_goal_group"]
        group["destination_residual"] = {"position_m": dp, "orientation_rad": dq}
        if dp > position_tol or dq > orientation_tol:
            raise Reject("campaign_destination_drift", f"destination drifted {dp:.6f} m / {dq:.6f} rad")
        start = draft.object_poses[plan.target_name]
        _, orientation_error = CTG.pose_error(start, plan.goal)
        group["start_orientation_residual_rad"] = orientation_error
        if orientation_error > orientation_tol:
            raise Reject("campaign_orientation_drift", f"start/goal target orientation differs by {orientation_error:.6f} rad")
        if settled:
            actual = self.objects[plan.target_name].get_position_orientation()
            dp, dq = CTG.pose_error(actual, plan.goal)
            group["settled_goal_residual"] = {"position_m": dp, "orientation_rad": dq}
            if dp > position_tol or dq > orientation_tol:
                raise Reject("campaign_goal_drift", f"settled tray goal drifted {dp:.6f} m / {dq:.6f} rad")
        draft.goal_poses[plan.target_name] = tuple(TT._t(v) for v in plan.goal)
        draft.pose_targets[plan.target_name] = full_pose_target(*plan.goal, self.tol("position", 0.03),
                                                               self.tol("orientation", 0.20), equivalences=self.target_equivalences())
        draft.summary["campaign_goal_group"]["phase"] = "settled_goal_verified" if settled else "settled_geometry_verified"

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        if os.environ.get(CTG.ENV):
            self._campaign_tray_active = self.campaign_tray_plan()
            self._campaign_latest_proposal = None
        redraws = disemb_pos_redraws()
        if not redraws or getattr(self, "_split_cone_pools", None) is None:
            return Disembedding._sample_position(self, rng)  # (the class's own: stub tests call it with a namespace)
        # i7-yield2 r3 (lead #66): the split cone's drawn start POSITION is redrawn within the attempt (without replacement, among the
        # positions of the drawn start's pool) when every work stance there fails - the drawn start (~70 s) is kept
        side = getattr(self, "region_side", None)
        pool = self._split_cone_pools.get(side) or {}
        cur = list(self.manip_candidates or [])
        tried = {k for k, cs in pool.items() if len(cs) == len(cur) and all(a is b for a, b in zip(cs, cur))}
        errors: List[str] = []
        for r in range(redraws + 1):
            try:
                d = Disembedding._sample_position(self, rng)
                if r:
                    TT._count(self, "pos_redraw.passed")
                    TT._count(self, f"pos_redraw.passed_after_{min(r, 5)}{'+' if r >= 5 else ''}")
                    d.summary["pos_redraw"] = {"redraws": r}
                return d
            except Reject as e:
                if e.reason != "target_placement":
                    raise
                errors.append(str(e)[:400])
            left = sorted(k for k in pool if k not in tried)
            if not left or r == redraws:
                break
            key = left[int(rng.integers(len(left)))]
            tried.add(key)
            self.manip_candidates = list(pool[key])
            TT._count(self, "pos_redraw.redrawn")
        TT._count(self, "pos_redraw.exhausted")
        raise Reject("target_placement", f"{len(errors)} positions of the drawn start: " + "; ".join(errors[-2:])[:1500])

    def _sample_position(self, rng: np.random.Generator) -> Draft:
        tname, sname, dname = self.tname, self.bindings.one(self.src_role).name, self.bindings.one(self.dst_role).name
        dnames = [b.name for b in self.bindings.many("distractors")]
        init = self.init_stance
        errors = []
        order = [int(k) for k in rng.permutation(len(self.manip_candidates))]
        # i7-memory2 (owner 2026-11-08 2.3): the outcome memory's PICK hook on the manipulation stance candidates (alias_memory.reorder_pick: a
        # flagged (arm x trunk[, stand-off]) bin's candidates move to the end unless a Thompson draw keeps them; off / nothing flagged = unchanged)
        order = AM.reorder_pick(self, order, self.manip_candidates, lambda st: self.segment_arms[0], rng)
        reference = CTG.configured_reference_start()
        if reference is not None:
            from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances
            order = prioritize_written_stances(order, self.manip_candidates, [reference['manipulation']], explore=1, exploit=3)
        for k in order[:6]:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, [self.segment_arms[0]]
            self.sidecar.reset_instance()
            self.sidecar.add_stance(init)
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            try:
                d = self._sample_with(rng, init, manip, tname, sname, dname, dnames, int(k))
                return d
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.variant, errors[-1][:300])
        raise Reject("target_placement", "; ".join(errors[-3:])[:1500])

    def _sample_with(self, rng, init: Stance, manip: Stance, tname: str, sname: str, dname: str, dnames: List[str], k: int) -> Draft:
        d = Draft()
        src, tgt = self.obj(self.src_role), self.obj("target")
        campaign_plan = getattr(self, "_campaign_tray_active", None)
        if campaign_plan is not None:
            source = tuple(TT._t(v) for v in campaign_plan.source)
            floor = self.src_floor.at(source)
            manip.focus = th.tensor([float(floor.centre[0]), float(floor.centre[1]), floor.z])
        # §4.5 step 5: transition init -> manipulation
        self.record_stance_travel(init, manip)
        # source tray at the focus, aligned with the edge (§4.6 step 1, disembedding)
        s_pose = (self._tray_pose_centred(self.src_role, self.src_floor, manip, manip.focus[:2]) if campaign_plan is None
                  else tuple(TT._t(v) for v in campaign_plan.source))
        if not self.top.polygon_supported(self.world.footprint(self.hull[self.src_role], s_pose), margin=TT.SUPPORT_MARGIN):
            raise Reject("support", "source tray footprint not on the top at the focus")
        src.set_position_orientation(*s_pose)
        src.keep_still()
        self.kit.sync_world()
        src_w = self.src_floor.at(s_pose)
        if campaign_plan is not None:
            cam_init = self.camera_pose_at(init)
            self._campaign_destination = self.campaign_tray_geometry(
                manip, cam_init, s_pose, tuple(TT._t(v) for v in campaign_plan.destination))
            self._campaign_latest_proposal = {"start_poses": {sname: s_pose, dname: campaign_plan.destination},
                                              "goal_poses": {tname: campaign_plan.goal}, "base_pose2d": init.base_pose2d,
                                              "group_id": campaign_plan.group_id}
        # target inside the tray from F (footprint on the tray floor, 0.5 cm), exact IK with support + tray + distractors ignored for the
        # grasp (``grasp_ignore_list``); the other arm is tried only where its band covers the focus (owner 2026-09-27)
        default = self.segment_arms[0]
        arms_here = [a for a in self.segment_arms if a == default or a in (manip.extras.get("focus_arms") or [])]
        layout, layout_rec = None, None
        if self.layout is not None:
            # the puzzle (handoff 2026-10-10): the figure's parts at their layout positions, the target held at its own, the whole figure
            # turned by a yaw that keeps a reachable grasp (owner 2026-10-12); the pick ignores the parts (A2b: ``pick_world``)
            t_pose, grasp, layout, s_pose, src_w, layout_rec = self._layout_pick(rng, manip, arms_here)
        else:
            target_rest = self.rest["target"] if campaign_plan is None else TT._t(campaign_plan.goal[1])
            goal_draw_kw = {} if campaign_plan is None else {"yaws": [0.0], "jitter": 0.0}
            t_pose, grasp = TT.phase3_target(self, manip, rng, self.grasps, self.hull["target"], target_rest, self.grasp_ignore_list(), top=src_w,
                                             support_z=src_w.z, margin=float(self.g("tray_footprint_margin_m")), scale_key=self.choice.models["target"].scale,
                                             bands=self.bands, arms=arms_here, arm_bands=self.arm_bands or None, **goal_draw_kw)
        if grasp.arm != default:
            self.sidecar.transitions.clear()
            self.record_stance_travel(init, manip)  # the other arm idles now
        # the segment's later contacts; when they fail with the chosen arm, the whole segment is retried with the other arm if its grasp
        # passed too (owner 2026-09-27: the fallback spans the segment)
        tried: List[str] = []
        while True:
            try:
                return self._segment_rest(rng, d, init, manip, t_pose, grasp, s_pose, src_w, tname, sname, dname, dnames, k, arms_here, layout, layout_rec)
            except Reject as e:
                tried.append(grasp.arm)
                others = [a for a in TT.fallback_arms(grasp) if a not in tried]
                if not others:
                    raise
                with self.pick_world():
                    g2 = TT.segment_fallback_grasp(self, manip, t_pose, grasp, self.pick_grasps, others[0], ignore_for_grasp=self.grasp_ignore_list(),
                                                   scale_key=self.choice.models["target"].scale, support_z=src_w.z, why=f"[{e.reason}] {e}")
                if g2 is None:
                    raise
                grasp = g2
                d = Draft()
                self.sidecar.transitions.clear()
                self.sidecar.paths.clear()
                self.record_stance_travel(init, manip)
                for o in self.world.spawned.values():
                    if o.name not in (src.name, tgt.name):
                        self.world.park(o, list(self.world.spawned).index(o.name))
                if layout is not None:
                    self._place_layout(layout)

    def campaign_grasp_joint_margin(self, grasp):
        """Reject fragile approach witnesses using the existing safe joint margin."""
        if not CR.contact_recovery('tray_grasp_joint_margin'):
            return
        from ispatialgym.sampling.v2 import orient as O
        qs = O.chain_qs(self.ikc, grasp.arm, [grasp.pre_grasp, grasp.grasp])
        if not qs:
            raise Reject('campaign_grasp_margin', 'grasp approach lacks exact native witnesses')
        margin = min(self.ikc.joint_limit_margin(q, grasp.arm)[0] for q in qs)
        limit = self.ikc.JOINT_LIMIT_MARGIN
        grasp.extras['campaign_grasp_joint_margin'] = {'margin_rad': float(margin), 'required_rad': float(limit)}
        if not math.isfinite(margin) or margin < limit:
            raise Reject('campaign_grasp_margin', f'approach joint clearance {margin:.6f} rad < native safe margin {limit:.6f}')

    def _segment_rest(self, rng, d: Draft, init: Stance, manip: Stance, t_pose, grasp, s_pose, src_w, tname: str, sname: str, dname: str, dnames: List[str], k: int,
                      arms_here: List[str], layout: Optional[Dict[str, TT.Pose]] = None, layout_rec: Optional[Dict[str, Any]] = None) -> Draft:
        """The segment after its grasp, with the grasp's arm: destination, distractors (drawn on the source floor, or the fixed
        layout's ``layout`` poses at the yaw of ``layout_rec``), frame and whole-body checks, the placement stance and its place, the
        carried path, the records."""
        self.campaign_grasp_joint_margin(grasp)
        support = self.obj("support")
        dst, tgt = self.obj(self.dst_role), self.obj("target")
        grasp.place_on = dname
        # destination tray: the template's side, exactly 4 cm inside the init image, >= 4 cm from the source tray, footprint on the top
        cam_init = self.camera_pose_at(init)
        campaign_plan = getattr(self, "_campaign_tray_active", None)
        if campaign_plan is None:
            d_pose, side, dest_diag = self._place_destination(rng, init, manip, cam_init, s_pose)
        else:
            d_pose, side, dest_diag = self.campaign_tray_geometry(
                manip, cam_init, s_pose, tuple(TT._t(v) for v in campaign_plan.destination))
        dst_w = self.dst_floor.at(d_pose)
        row_name = C.split_twin(grasp.grasp_name or "")[0]  # a twin pinches the same width as its library row
        width = TT.grasp_pinch_width(next((g for g in self.grasps if g.name == row_name), None))
        if layout is not None:
            # the puzzle's parts stay where the layout puts them: no pool draw, outside the budget (handoff 2026-10-10)
            dist_poses = {n: layout[n] for n in dnames}
        else:
            # distractors on the source floor: footprints >= 1 cm from each other and from the target, within the budget. Owner
            # 2026-10-12 (A2b): no spacing along the drawn grasp's pinch axis and no pad-clearance growth of the budgeted footprints - the
            # grasp may touch them, the demo checks they are still on the source tray
            items = [(n, self.hull[n], self.rest[n]) for n in dnames]
            screen_approach = CR.contact_recovery('milk_clutter_approach')
            approach_draws = []
            for clutter_draw in range(3 if screen_approach else 1):
                dist_poses = TT.place_distractors(self, rng, src_w, src_w.z, t_pose, self.hull["target"], items, margin=float(self.g("tray_footprint_margin_m")), gap=0.01,
                                              cam=cam_init, budget=self._tray_budget(), n_tries=int(self.g("distractor_place_tries") or 80),
                                              max_backtracks=24 if os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG") else 0,
                                              search_seconds=15 if os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG") else None)
                if not screen_approach:
                    break
                parts = {name: self._part_pieces_in(name, pose, t_pose) for name, pose in dist_poses.items()}
                proof = clutter_approach_proof(grasp, width, parts)
                approach_draws.append(proof)
                TT._count(self, 'campaign_clutter_approach.draws')
                if not proof['blocked_by']:
                    TT._count(self, 'campaign_clutter_approach.passed')
                    d.summary['campaign_clutter_approach'] = {'draws': approach_draws, 'selected_draw': clutter_draw}
                    break
                TT._count(self, 'campaign_clutter_approach.obstructed')
                log.info('%s: campaign clutter approach obstructed: %s', self.gp.variant, json.dumps(proof))
            else:
                raise Reject('distractor_approach_blocked', 'three bounded clutter proposals obstruct the selected fingers: ' + json.dumps(approach_draws))
        objects = {tname: t_pose, sname: s_pose, dname: d_pose, **dist_poses}
        if campaign_plan is not None:
            self._campaign_latest_proposal = {"start_poses": objects, "goal_poses": {tname: campaign_plan.goal},
                                              "base_pose2d": init.base_pose2d, "group_id": campaign_plan.group_id}
            occupied = sum(CTG.polygon_area(self.world.footprint(self.hull["target" if n == tname else n], objects[n]))
                           for n in [tname] + dnames)
            coverage = occupied / src_w.area()
            if not campaign_plan.accepts_coverage(coverage):
                raise Reject("campaign_crowdedness_bucket", f"tray coverage {coverage:.6f} outside {campaign_plan.crowdedness_range}")
            d.summary["campaign_goal_group"] = campaign_plan.summary()
            d.summary["campaign_goal_group"].update({"phase": "geometry_pinned", "actual_crowdedness": coverage,
                "requested_start": {"target": [v.tolist() for v in t_pose], "grasp": grasp.grasp_name,
                                    "distractors": {n: [v.tolist() for v in dist_poses[n]] for n in dnames}}})
        # §4.6 step 4: every object in frame (5 %) in the init image except the destination tray
        items_all = [(tname, self.hull["target"], t_pose), (sname, self.hull[self.src_role], s_pose)] + [(n, self.hull[n], dist_poses[n]) for n in dnames]
        missing = TT.all_in_frame(self, cam_init, items_all, float(self.g("image_margin")))
        if missing:
            raise Reject("not_in_frame", f"{missing} not fully in the init image")
        # §4.6 step 5: whole-body checks at init and manipulation with the objects placed
        TT.whole_body_check_at(self, [init, manip], objects, require=True, reason="stance_collision")
        # §4.7 placement stance and the place IK (object in hand; destination tray + support ignored for the place pose only)
        goal = self._goal_pose(t_pose, dst_w)
        place_st = TT.placement_stance(self, manip, dst_w.centre)
        TT.whole_body_check_at(self, [place_st], {}, require=True, reason="no_placement_stance")
        place = IK.grasp_pose(goal, grasp.eef_in_object)
        pre_place = IK.pre_pose(place)
        with TT.posed_in_hand(self, grasp.arm, tgt, grasp.eef_in_object):
            with attached(self.kit, grasp.arm, tgt):
                res = self.ikc.exact([pre_place, place], place_st, grasp.arm, ignore_for_last=[support, dst], scale_key=self.choice.models["target"].scale)
            if not res.ok:
                raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the placement stance: {res.reason}")
            grasp.pre_place, grasp.place = pre_place, place
            grasp.extras["release_surface_z"] = None if self.release_surface_z is None else round(float(self.release_surface_z), 4)
            # carry height: object bottom >= tallest distractor + tray rim + 3 cm above the source floor
            rim_h = src_w.rim_z - src_w.z
            tallest = max([self.height[n] for n in dnames] + [rim_h])
            grasp.carry_height = float(tallest + float(self.g("carry_height_extra_m")))
            self._record_carry_path(manip, place_st, tname, grasp.arm)
        cam_manip = self.camera_pose_at(manip)
        if not self.ikc.eef_in_frame(grasp.grasp, cam_manip):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if not self.ikc.eef_in_frame(place, self.camera_pose_at(place_st)):
            raise Reject("gripper_not_in_frame", "place eef pose outside the placement image (5 %)")
        # occlusion by renders at the init stance (§6.4): ratio <= 0.30, visible area >= 400 px, destination visible
        occ = self._measure_occlusion(init, objects, tname, dnames, dname)
        # record
        self.sidecar.add_stance(manip)
        self.sidecar.add_stance(place_st)
        self.sidecar.add_grasp(grasp)
        d.object_poses.update(objects)
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.goal_poses[tname] = goal
        d.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20), equivalences=self.target_equivalences())
        d.extras["initial_occlusion"] = occ
        # the trays' mass (sample_template): the loader re-applies it (run_isg._restore_constraints), the settle / stillness gate ran with it
        masses = [{"type": "mass", "object": n, "mass_kg": round(float(self.tray_mass[self.objects[n].name]["after_kg"]), 4), "stock_kg": round(float(self.tray_mass[self.objects[n].name]["before_kg"]), 4),
                   "note": "generation.tray_mass_kg: a tray at its 0.4-0.6 kg category mass under 1 kg bottles is an inverted mass ratio PhysX resolves with contact chatter"}
                  for n in (sname, dname) if self.objects[n].name in getattr(self, "tray_mass", {}) and abs(self.tray_mass[self.objects[n].name]["after_kg"] - self.tray_mass[self.objects[n].name]["before_kg"]) > 1e-6]
        if masses:
            d.extras["constraints"] = masses
        d.summary.update({"manipulation_candidate": k, "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "pinch_width": width},
                          "dest_side": "left" if side > 0 else "right", "destination": dest_diag, "occlusion": occ,
                          "carry_height": grasp.carry_height, "placement_stance": place_st.to_meta(), "arm": TT.segment_arm_summary(grasp, arms_here)})
        if layout is not None:
            # owner 2026-10-12: the layout's yaw in the summary, the TRO (``scene.fixed_layout``) and the template meta (write_instance)
            d.summary["fixed_layout"] = dict(layout_rec or {})
            d.extras["fixed_layout"] = {k_: (layout_rec or {}).get(k_) for k_ in ("file", "yaw_deg", "figure_yaw_rad", "centre_xy", "tray_turned")}
        grasp.extras["arm_offer"] = TT.arm_offer_record(self.segment_arms, arms_here, grasp, self.arm_offer_why)
        # owner 2026-10-12 (A2b): the parts / distractors the grasp may touch or displace, each still on the source tray after it; the
        # fixed-layout parts at the pre-grasp too
        grasp.extras["may_touch"] = {"objects": list(dnames), "stay_on": sname, **({"pre_grasp": list(dnames)} if layout is not None else {})}
        self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "), self._descriptor(tname, objects, cam_init))
        self.sidecar.describe(sname, "source tray", "the tray in front of the robot")
        self.sidecar.describe(dname, "destination tray", f"the tray on the {'left' if side > 0 else 'right'}")
        for n in dnames:
            b = self.bindings.by_name()[n]
            self.sidecar.describe(n, (b.category or n).replace("_", " "), self._descriptor(n, objects, cam_init))
        return d

    def _descriptor(self, name: str, objects: Dict[str, TT.Pose], cam) -> str:
        """Image-space descriptor (left / centre / right, front / back on the source tray) for the language annotations."""
        b = self.bindings.by_name()[name]
        cm = self.world.camera_model
        uv, _ = cm.project(cam, TT._t(objects[name][0]).unsqueeze(0))
        u, v = float(uv[0, 0]) / cm.width, float(uv[0, 1]) / cm.height
        col = "left" if u < 0.4 else ("right" if u > 0.6 else "centre")
        row = "front" if v > 0.6 else ("back" if v < 0.4 else "middle")
        return f"the {(b.category or name).replace('_', ' ')} at the {row} {col} of the source tray"

    def _goal_pose(self, t_pose: TT.Pose, dst_w: TrayFloorWorld) -> TT.Pose:
        """§6.4 goal: the target at the destination floor centre with its init orientation, released
        ``release_above_surface_m`` (1 cm) above the **actual** surface under the footprint centre — a downward raycast
        (``SimWorld.surface_z_under``, core `2c59893`, the target ignored) from above the destination rim; the raster floor
        level + 3 mm when the ray misses or hits outside the tray's floor band (tray parked, ray through a slot)."""
        campaign_plan = getattr(self, "_campaign_tray_active", None)
        if campaign_plan is not None:
            goal = tuple(TT._t(v) for v in campaign_plan.goal)
            if CTG.pose_error(t_pose, goal)[1] > self.tol("orientation", 0.20):
                raise Reject("campaign_orientation_drift", "disembedding start and goal orientations must match")
            if not dst_w.polygon_supported(self.world.footprint(self.hull["target"], goal), margin=float(self.g("tray_footprint_margin_m"))):
                raise Reject("campaign_goal_support", "fixed target goal footprint leaves the destination floor")
            self.release_surface_z = float(dst_w.z)
            return goal
        quat = TT._t(t_pose[1])
        xy = th.tensor([float(dst_w.centre[0]), float(dst_w.centre[1])])
        surface = None
        if hasattr(self.world, "surface_z_under"):
            hit = self.world.surface_z_under(xy, dst_w.z + self.dst_floor.rim_local + 0.30, ignore=[self.obj("target")])
            if hit is not None and dst_w.z - 0.01 <= float(hit["z"]) <= dst_w.z + 0.02:
                surface = float(hit["z"])
        self.release_surface_z = surface
        base_z = dst_w.z + TT.Z_OFFSET if surface is None else surface + float(self.gen("release_above_surface_m", RELEASE_ABOVE_SURFACE))
        pos = th.tensor([float(xy[0]), float(xy[1]), base_z + self.world.bottom_offset(self.hull["target"], quat)])
        return pos, quat

    def _place_destination(self, rng, init: Stance, manip: Stance, cam, s_pose: TT.Pose):
        """§4.6 (disembedding): destination on the template's side (``dest_side``: only that side has the destination's Phase-1c
        margin, owner 2026-10-12 A2a), aligned with the edge and the source tray (same depth), at the lateral position where
        exactly ``destination_visible_width_m`` of it is inside the init image (bisected on the camera projection of the point
        that far inside its inner edge), >= 4 cm from the source tray, footprint on the top. Returns (pose, side, diagnostics);
        ``Reject("destination_placement")``."""
        top, cm = self.top, self.world.camera_model
        key = self.dst_role
        yaw_d = self._tray_yaw(key, manip)
        quat = C.quat_from_yaw(yaw_d)
        fp0 = self.world.footprint(self.hull[key], (th.zeros(3), quat))
        half_lat_d = float((fp0 @ manip.tangent).abs().max())
        s_fp = self.world.footprint(self.hull[self.src_role], s_pose)
        s_lat = (s_fp - manip.base_xy) @ manip.tangent
        depth_s, _ = manip.local(s_pose[0])
        w = float(self.g("destination_visible_width_m"))
        gap_min = float(self.g("destination_gap_min_m"))
        top_z_d = top.z + self.world.bottom_offset(self.hull[key], quat) + TT.height_of(self.hull[key], quat)
        sides = [self.dest_side]
        diag: Dict[str, Any] = {}

        def pose_at(lat):
            xy = manip.world(depth_s, lat)
            return (th.tensor([float(xy[0]), float(xy[1]), top.z + self.world.bottom_offset(self.hull[key], quat) + TT.Z_OFFSET]), quat)

        def border_gap(sign, lat):
            """<= 0 while >= w of the tray is inside the image, > 0 once less than w shows (raw border, the destination is exempt from the 5 % rule)."""
            p = manip.world(depth_s, lat - sign * (half_lat_d - w))
            uv, dep = cm.project(cam, th.tensor([[float(p[0]), float(p[1]), top_z_d]]))
            if float(dep[0]) <= 0.05:
                return 1.0
            u = float(uv[0, 0])
            return (u - cm.width) if sign < 0 else (-u)  # robot's right = image right border, robot's left = image left

        for sign in sides:
            l_min = (float(s_lat.max()) + gap_min + half_lat_d) if sign > 0 else (-float(s_lat.min()) + gap_min + half_lat_d)
            l_max = None
            for l in th.arange(l_min, 3.0, 0.02).tolist():
                if not top.polygon_supported(self.world.footprint(self.hull[key], pose_at(sign * l)), margin=TT.SUPPORT_MARGIN):
                    break
                l_max = l
            if l_max is None:
                diag[f"side_{'left' if sign > 0 else 'right'}"] = "no supported position"
                continue
            g_lo, g_hi = border_gap(sign, sign * l_min), border_gap(sign, sign * l_max)
            if not (g_lo <= 0 <= g_hi):
                diag[f"side_{'left' if sign > 0 else 'right'}"] = f"border gap {g_lo:.0f}..{g_hi:.0f} px over lateral {l_min:.2f}..{l_max:.2f}"
                continue
            lo, hi = l_min, l_max
            for _ in range(18):
                mid = 0.5 * (lo + hi)
                if border_gap(sign, sign * mid) <= 0:
                    lo = mid
                else:
                    hi = mid
            lat = sign * 0.5 * (lo + hi)
            pose = pose_at(lat)
            if not top.polygon_supported(self.world.footprint(self.hull[key], pose), margin=TT.SUPPORT_MARGIN):
                diag[f"side_{'left' if sign > 0 else 'right'}"] = "bisected position not supported"
                continue
            diag.update({"lateral_m": lat, "visible_width_m": w, "lateral_range": [l_min, l_max]})
            return pose, sign, diag
        raise Reject("destination_placement", f"no lateral position keeps exactly {w:.2f} m of the destination tray in the init image ({diag})")

    def _measure_occlusion(self, init: Stance, objects: Dict[str, TT.Pose], tname: str, dnames: List[str], dname: str) -> Dict[str, Any]:
        """§6.4 occlusion by two segmentation renders at the init stance: ``ratio = 1 − V/A`` (A = the target alone, the
        distractors hidden), ratio <= 0.30, V >= 400 px, target inside the 5 % margin (checked by the caller), the
        destination tray visible in the same render."""
        for name, pose in objects.items():
            o = self.objects[name]
            o.set_position_orientation(TT._t(pose[0]), TT._t(pose[1]))
            o.wake()
            o.keep_still()
        self.kit.apply(self.stance_q(init), init.base_pose2d)
        self.world.step_physics(1, still=[self.robot] + [self.objects[n] for n in objects])
        target = self.objects[tname]
        hidden = [self.objects[n] for n in dnames]
        vis = self.world.object_pixels(target)
        ref = self.world.object_pixels(target, hidden=hidden)
        if ref == 0:
            self.world.ensure_segmented(target)
            vis = self.world.object_pixels(target)
            ref = self.world.object_pixels(target, hidden=hidden)
        if ref == 0:
            raise Reject("seg_missing", f"{target.name} has no segmentation pixels even with the distractors hidden")
        ratio = 1.0 - vis / max(ref, 1)
        px = self.world.visible_pixels(exclude_structure=False)
        dest_px = px.get(self.objects[dname].name, 0)
        if ratio > float(self.g("max_occlusion_ratio")):
            raise Reject("occlusion", f"occlusion ratio {ratio:.2f} > {self.g('max_occlusion_ratio')}")
        if vis < int(self.g("min_visible_pixels")):
            raise Reject("occlusion", f"target visible area {vis} px < {self.g('min_visible_pixels')}")
        if dest_px <= 0:
            raise Reject("dest_not_visible", "destination tray not visible in the confirmation render")
        return {"target": tname, "occluders": list(dnames), "ratio": float(ratio), "visible_area": int(vis), "reference_area": int(ref),
                "occluder_ids": [self.objects[n].name for n in dnames], "camera_key": self.world.cam_key, "destination_pixels": int(dest_px)}

    # ------------------------------------------------------------------ goal / pipeline hooks

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    #: owner 2026-09-24: physics steps the trays get alone before the objects are released onto them (released at contact
    #: since 2026-09-25, so this only lets the contacts form)
    TRAY_LANDING_STEPS = 20
    #: 2026-09-24: 2.5 s of settle (90 -> 300 physics steps, + 10) before the stability / trembling judgement: the sway of
    #: the tall upright apple-juice bottle after its release decays from 0.3-1.4 mm per frame at 1.5-3 s to ~0.1 mm later
    DEFAULT_SETTLE_STEPS = 300

    def apply_draft(self, draft: Draft):
        """The trays land first, the objects after (owner 2026-09-24). The base ``apply_draft`` releases every object 3 mm
        above its support at once, so the bottles landed on trays that were themselves still dropping / rocking and rolled
        away (a lying cylinder has no restoring torque): 5 of 8 attempts of the 3-distractor milk variant (0924-233604) were
        ``unstable_init`` with v ≈ r·ω. Here the trays get ``TRAY_LANDING_STEPS`` physics steps by themselves, then the
        objects are released at their recorded heights **above the settled source tray** (its landing shifted it by up to
        the 3 mm drop); the trays' settled poses are recorded by ``settle_and_validate`` as before."""
        trays = [n for n in (self.bindings.one(self.src_role).name, self.bindings.one(self.dst_role).name) if n in draft.object_poses]
        objects = [n for n in draft.object_poses if n not in trays]
        if not trays or not objects:
            return super().apply_draft(draft)
        for name in trays:
            obj = self.objects[name]
            pos, quat = TT._t(draft.object_poses[name][0]).clone(), TT._t(draft.object_poses[name][1])
            pinned_destination = getattr(self, "_campaign_tray_active", None)
            if pinned_destination is None:
                pos[2] -= max(0.0, TT.Z_OFFSET - float(self.gen("tray_release_gap_m", TRAY_RELEASE_GAP_M)))  # at contact, not dropped
            obj.set_position_orientation(pos, quat)
            if obj.name in self.world.spawned:
                self.world.activate(obj)
            obj.keep_still()
        import omnigibson as og

        for _ in range(int(self.TRAY_LANDING_STEPS)):
            og.sim.step_physics()
            self.robot.keep_still()
        src_name = self.bindings.one(self.src_role).name
        dz = float(self.objects[src_name].get_position_orientation()[0][2]) - float(draft.object_poses[src_name][0][2]) if src_name in trays else 0.0
        gap_layout = float((self.gen("fixed_layout", None) or {}).get("release_gap_m", FIXED_LAYOUT_RELEASE_GAP_M))
        for name in objects:
            obj = self.objects[name]
            pos, quat = TT._t(draft.object_poses[name][0]).clone(), TT._t(draft.object_poses[name][1])
            # the source tray landed (dz): keep the recorded height above its floor, minus the §3.2 drop down to RELEASE_GAP_M
            gap = gap_layout if name in (self.layout or {}) else float(self.gen("release_gap_m", RELEASE_GAP_M))
            pos[2] += dz - max(0.0, TT.Z_OFFSET - gap)
            obj.set_position_orientation(pos, quat)
            if obj.name in self.world.spawned:
                self.world.activate(obj)
            obj.keep_still()
        for name, q in draft.object_joints.items():
            obj = self.objects[name]
            obj.set_joint_positions(TT._t(q))
            obj.set_joint_velocities(th.zeros_like(TT._t(q)))
            obj.keep_still()
        self.kit.apply(draft.robot_q, draft.base_pose2d)
        for name in draft.object_poses:
            self.objects[name].keep_still()
        # diagnostic (2026-09-25): the geometric clearances the release starts from - pairwise footprint gaps of the placed
        # objects (their collision hulls at the actual poses) and each hull's height above the source floor
        try:
            from ispatialgym.sampling.v2.tabletop import polygon_gap

            src_z = float(self.objects[src_name].get_position_orientation()[0][2]) + self.src_floor.z_local
            fps, bottoms = {}, {}
            for name in objects:
                p, q = self.objects[name].get_position_orientation()
                hull = self.hull.get("target" if name == self.tname else name)
                if hull is None:
                    continue
                fps[name] = self.world.footprint(hull, (p.float(), q.float()))
                bottoms[name] = round((float(p[2]) - self.world.bottom_offset(hull, q.float()) - src_z) * 1000.0, 2)
            names = list(fps)
            gaps = {f"{a}|{b}": round(float(polygon_gap(fps[a], fps[b])) * 1000.0, 1) for i, a in enumerate(names) for b in names[i + 1:]}
            draft.summary["release_geometry"] = {"bottom_above_floor_mm": bottoms, "footprint_gap_mm": gaps, "tray_dz_mm": round(dz * 1000.0, 2)}
            log.info("%s: release geometry: bottoms above the source floor %s mm; footprint gaps %s mm; tray dz %.2f mm", self.gp.variant, bottoms, gaps, dz * 1000.0)
        except Exception as e:  # noqa: BLE001 - a diagnostic must never fail the attempt
            log.debug("release geometry diagnostic failed: %r", e)
        self.apply_draft_extras(draft)

    def settle_entities(self, draft: Draft):
        return [self.objects[n] for n in draft.object_poses]

    def after_settle(self, draft: Draft):
        """The trays settled: the goal is the destination floor centre at the settled destination pose with the settled
        target orientation."""
        tname, dname = self.tname, self.bindings.one(self.dst_role).name
        dst_w = self.dst_floor.at(draft.object_poses[dname])
        goal = self._goal_pose(draft.object_poses[tname], dst_w)
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20), equivalences=self.target_equivalences())
        g = self.sidecar.grasp_of(tname)
        if g is not None:
            g.extras["release_surface_z"] = None if self.release_surface_z is None else round(float(self.release_surface_z), 4)
        draft.summary["release_surface_z"] = None if self.release_surface_z is None else round(float(self.release_surface_z), 4)
        if getattr(self, "_campaign_tray_active", None) is not None:
            self.campaign_verify_goal(draft)

    def refine_targets_from_goal(self, draft: Draft):
        super().refine_targets_from_goal(draft)
        self.campaign_verify_goal(draft, settled=True)

    def extra_checks(self, draft: Draft):
        """After settling: the occlusion measured again (the loader re-measures it after its deterministic settle)."""
        tname, dname = self.tname, self.bindings.one(self.dst_role).name
        dnames = [b.name for b in self.bindings.many("distractors")]
        init = self.sidecar.stance("init")
        cam = self.camera_pose_at(init)
        if not self.frustum_ok(cam, self.hull_corners_world(self.hull["target"], draft.object_poses[tname]), float(self.g("image_margin"))):
            raise Reject("not_in_frame", "target left the 5 % image margin while settling")
        occ = self._measure_occlusion(init, draft.object_poses, tname, dnames, dname)
        draft.extras["initial_occlusion"] = occ
        draft.summary["occlusion_settled"] = occ

    def after_goal_check(self, draft: Draft):
        """§4.7: the place / pre-place at the settled goal pose from the placement stance (object in hand, destination
        tray and support ignored for the place pose only)."""
        tname = self.tname
        grasp = self.sidecar.grasp_of(tname)
        place_st = self.sidecar.stance("placement")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"])
        place = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
        pre_place = IK.pre_pose(place)
        tgt = self.obj("target")
        with TT.posed_in_hand(self, grasp.arm, tgt, grasp.eef_in_object):
            with attached(self.kit, grasp.arm, tgt):
                res = self.ikc.exact([pre_place, place], place_st, grasp.arm, ignore_for_last=[self.obj("support"), self.obj(self.dst_role)],
                                     scale_key=self.choice.models["target"].scale)
        if not res.ok:
            raise Reject("place_unreachable", f"settled goal: pre-place / place fail the exact IK from the placement stance: {res.reason}")
        grasp.pre_place, grasp.place = pre_place, place
        if os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG") and getattr(self, "_campaign_tray_active", None) is None:
            # A fresh successful anchor supplies an admissible surface region.
            # The controller chooses a goal from it before drawing group starts.
            dst_w = self.dst_floor.at(draft.object_poses[self.bindings.one(self.dst_role).name])
            cells = dst_w.cells()
            stride = max(1, math.ceil(len(cells) / 256))
            candidates = []
            for xy in cells[::stride]:
                p = th.tensor([float(xy[0]), float(xy[1]), float(goal_pos[2])])
                if dst_w.polygon_supported(self.world.footprint(self.hull["target"], (p, goal_quat)), margin=float(self.g("tray_footprint_margin_m"))):
                    candidates.append({"position": p.tolist(), "orientation": goal_quat.tolist()})
            draft.summary["campaign_goal_space"] = {"target_poses": candidates, "source": "fresh_destination_floor_raster",
                                                     "target_height": "settled_anchor_goal", "max_candidates": 256}

    def turn_recheck(self, draft: Draft, grasp, q_first: Optional[th.Tensor] = None) -> Optional[str]:
        """``tabletop.held_turns``: this module's checks that depend on the wrist turn, for ``grasp`` just recorded as its other finger
        assignment - the pick from the manipulation stance as the draw solved it (``grasp_ignore_list`` ignored at the grasp; from
        ``q_first``, the recorded pre-grasp witness wrist-flipped, else afresh) with the wrist camera clear of the source tray's floor
        and of the world, the carried path to the placement stance swept again with this turn in the hand, and the pre-place -> place
        from the placement stance as ``after_goal_check`` solves it. None when every check passes, else the first failure."""
        from ispatialgym.sampling.v2 import orient as O

        tgt, support, dst = self.obj("target"), self.obj("support"), self.obj(self.dst_role)
        manip, place_st = self.sidecar.stance("manipulation"), self.sidecar.stance("placement")
        scale = self.choice.models["target"].scale
        # i7-split 2026-11-07: the approached objects as the program check's grasp names them (``_approached_for``: the target and its
        # ``may_touch`` distractors) - inferred, the source tray's AABB held the pad and the tray left the world along the whole approach
        appr = [tgt] + [self.objects[b.name] for b in self.bindings.many("distractors")]
        kw = {"ignore_for_last": self.grasp_ignore_list(), "scale_key": scale, "cache": False, "approached": appr}
        with self.pick_world(), TT.record_policy_opening(self, grasp):  # of-v2maps 2026-11-07: the policy at the grasp's pre-shape
            res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, q_first=q_first, **kw)
            if not res.ok and q_first is not None:
                res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, **kw)
            camera_clear = res.ok and O.camera_clear(self.ikc, res.q, grasp.arm)
        if not res.ok:
            return f"pick: {res.reason}"
        floor_z = float(self.src_floor.at(draft.object_poses[self.bindings.one(self.src_role).name]).z)
        if not TT.camera_top_clear(self, grasp.grasp, floor_z, grasp.arm) or not camera_clear:
            return "pick: the wrist camera on the tray floor / within 5 mm of the world"
        why = self.recheck_carried_travel(self.tname)
        if why:
            return f"carried path: {why}"
        with TT.posed_in_hand(self, grasp.arm, tgt, grasp.eef_in_object):
            with attached(self.kit, grasp.arm, tgt):
                r = self.ikc.exact([grasp.pre_place, grasp.place], place_st, grasp.arm, ignore_for_last=[support, dst], scale_key=scale, cache=False)
        return None if r.ok else f"place: {r.reason}"

    def arm_switch(self, draft: Draft, grasp, arm: str) -> Optional[str]:
        """``tabletop.held_arms``: the segment switched to ``arm`` in place - the manipulation and placement stances with ``arm`` working
        and the other arm in its first collision-free idle posture, the init -> manipulation transition and the carried path to the
        placement stance recorded again with ``arm`` holding the target (the objects at their settled poses). None when the switch is
        made, else why not."""
        manip, place_st, init = self.sidecar.stance("manipulation"), self.sidecar.stance("placement"), self.sidecar.stance("init")
        st_m, st_p = TT.stance_for_arm(self, manip, arm), TT.stance_for_arm(self, place_st, arm)
        if st_m is None or st_p is None:
            return f"no collision-free idle posture of the {TT.RB.other_arm(arm)} arm with the {arm} arm working at the {'manipulation' if st_m is None else 'placement'} stance"
        TT.adopt_arm(manip, st_m)
        TT.adopt_arm(place_st, st_p)
        self.sidecar.transitions[:] = [t for t in self.sidecar.transitions if not (t.frm == init.name and t.to == manip.name)]
        self.record_stance_travel(init, manip)
        self.sidecar.paths[:] = [p for p in self.sidecar.paths if not (p.frm == manip.name and p.to == place_st.name)]
        with TT.posed_in_hand(self, arm, self.obj("target"), grasp.eef_in_object):
            self._record_carry_path(manip, place_st, self.tname, arm)
        return None

    def _record_carry_path(self, manip: Stance, place_st: Stance, carried: str, arm: str):
        """§3.7 / §4.7 the carried leg manipulation -> placement (object attached). Owner 2026-10-13 (COMPLAINTS.md global rule,
        ``generation.place_leg_axis_aligned``, default true): moving to the place pose the robot moves only forward / backward or
        left / right, keeping its heading - the edge slide alone (``NAV.plan_axis_leg``) at the first extra stand-off of
        ``place_leg_back_offs_m`` (each + ``carry_edge_slide_back_m``) whose strides clear, else ``Reject(place_leg_collision)``;
        disembedding#03 c1_k0's slide collided 0.28 m before the placement stance and the grid fallback turned the robot 58 deg on
        the way. Off: the edge slide (owner 2026-09-24, ``carry_edge_slide``) with the grid path's turn-drive-turn when the strip
        collides."""
        base_back = float(self.g("carry_edge_slide_back_m") or 0.0)
        if self.g("place_leg_axis_aligned"):
            ladder = [base_back + float(b) for b in (self.g("place_leg_back_offs_m") or [0.0])]
            return self.record_path(manip, place_st, carried=carried, arm=arm, axis_leg=ladder)
        return self.record_path(manip, place_st, carried=carried, arm=arm, edge_slide=bool(self.g("carry_edge_slide")), edge_slide_back_m=base_back)

    def tro_meta_block(self, draft: Draft, instance_id: str, seed: int) -> Dict[str, Any]:
        """The TRO's scene block carries the fixed layout's yaw (owner 2026-10-12, ``scene.fixed_layout``): the oracle and the evaluator
        rebuild the parts from their recorded poses, the record says which turn of the figure they are."""
        meta = super().tro_meta_block(draft, instance_id, seed)
        if draft.extras.get("fixed_layout"):
            meta["scene"] = {**meta["scene"], "fixed_layout": dict(draft.extras["fixed_layout"])}
        return meta

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        if getattr(self, "_campaign_tray_active", None) is not None:
            self.campaign_verify_goal(draft)
            draft.summary["campaign_goal_group"]["phase"] = "settled_goal_verified"
        path = super().write_instance(draft, index, seed)
        if getattr(self, "_campaign_tray_active", None) is not None:
            self._campaign_group_written = getattr(self, "_campaign_group_written", 0) + 1
        if draft.extras.get("fixed_layout"):
            self.template_meta.setdefault("fixed_layout", {}).setdefault("per_instance", {})[self.instance_id(index)] = dict(draft.extras["fixed_layout"])
        return path

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        g = self.sidecar.grasp_of(self.tname)
        manip = self.sidecar.stance("manipulation")
        if g is None or manip.focus is None:
            return []
        mc = self.choice.models["target"]
        return [{"scope": {"target": f"{mc.category}/{mc.model}", "support_height": round(float(self.top.z - self.world.floor_z), 2)}, "type": "grasp_set",
                 "trunk": [manip.translate, round(manip.tilt, 2)], "cell": [round(v, 2) for v in manip.world_to_base(manip.focus[:2]).tolist()],
                 "feasible": [[round(g.yaw, 2), g.grasp_name]], "hard": False}]
