"""Configuration matching — articulation, large-scale, open / close (spec §5.3 / §6.6; oracle doc §6.2 row
"articulation — large-scale").

The target is a **scene articulated object** (cabinet, fridge, oven, dishwasher, drawer, door) bound per scene
(``bindings.target``); one joint per object (``roles.target.joint`` / ``roles.target.handle_key``). Nothing is spawned.
Single arm (the arm on the handle's side of the face centre), the object holds itself, no brace.

Per template (:meth:`ArticulationLarge.sample_template`): the handle annotation (B4 instance record → ``roles.target.
handle_file`` → canonical ``artic/handles/<key>.json`` scaled by the instance's per-axis scale; icl
``require/handles.csv`` verdicts: ``skip`` / ``ungraspable`` exclude the joint, ``edge`` / ``manual`` /
``manual_unscaled`` rows are admissible pinches, ``edge_blocked`` is not), the stable set S (7 fractions of
``open_fraction_range`` + the upper limit, each held 120 steps, kept within ½ tolerance), the passive-continuation
probe per (q_init, q_goal) pair (``passive_from`` / the required waypoints), the face frame, the arm rule and the
``follow_mode``.

Per instance (:meth:`sample_instance`): joint targets (open: q_init = lower, q_goal ~ U(S); close: q_init ~ U(S), q_goal
= lower), Prompt 0 Phase 1 (β_max 20°, d 0.5–2.5 m, the whole target at q_init in frame) and Phase 2 (d scan to 1.10 m,
Λ per arm about the handle contact, contact set = h(q_init) grasp of class ``fwd_pinch_v`` / ``fwd_pinch_h``, focus =
contact(q_init), the moving face patch in frame; cached per (scene, target, joint, q_init)), step 5 exact IK
([pre-grasp, grasp] at q_init, then the grasp pose at every required q_k with the world re-synced at q_k, target
ignored for the grasp poses only), base-follow feasibility (``follow_mode`` base for vertical-axis revolute /
horizontal prismatic joints: b_k = h(q_k) ∘ T_eef→base(q*), free floor, whole-body collision-free, base path per §3.7),
step 6 path + transition; the ``articulation`` sidecar block (§7 / F6 ``Articulation``).

Stand-ins (reported): the base-follow whole-body check at q_k removes the whole target from the collision world (the
spec ignores the grasped link for the gripper only; the backend has no link-level ignore) and adds a planar check of
the moving link's AABB against the base footprint; Phase 1 step 4 / arms-excluded checks as in :mod:`prompt0`.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import ASSETS_ROOT, REPO_ROOT
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.grasp_library import library_root
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.articulation import scale_handle, scale_joint_info
from ispatialgym.sampling.tasks.base import Draft, _jsonable
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import FaceFrame, Stance, pose_to_sidecar
from ispatialgym.sampling.v2.tasks import articulation_hybrid as AH

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASKS = (
    "configuration_matching-articulation_open_large_scale",
    "configuration_matching-articulation_close_large_scale",
    # spelling of ``v2.problem.SUBTASK_FAMILY`` (the iddl / F6 / generation_defaults use the two above)
    "configuration_matching-articulation_large_scale-open",
    "configuration_matching-articulation_large_scale-close",
)
#: B4 scene inventories: ``<scene>/artic/<name>.json`` per articulated scene object (instance frame)
SCENE_INVENTORY_ROOT = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915")
#: icl handle rows (verdicts / reviewer pinches / edge pinches)
HANDLES_CSV = os.path.join(ASSETS_ROOT, "icl", "require", "handles.csv")
#: v1 ``bddl_writer`` definition ids of the facade (``configuration_matching-1`` open, ``-2`` close)
V1_DEFINITION_ID = {"open": 1, "close": 2}
#: §5.3 pinch width after scaling
PINCH_WIDTH_RANGE = (0.003, 0.080)  # handle-selection pinch width (m): 3-80 mm (owner 2026-11-02; was 15-80 mm)
#: §5.3 waypoint fractions
WAYPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
#: base footprint radius of the planar leaf-vs-base stand-in check (R1Pro chassis half-diagonal)
BASE_RADIUS = 0.33
#: i7-artlarge (ISPATIALGYM_ARTLARGE_INIT_DRAW): bearing step of the init-set lattice (deg) and the most other init candidates step 6
#: tries after the drawn one
INIT_SET_BETA_STEP_DEG = 5.0
INIT_ALT_MAX = 12


def init_draw_on() -> bool:
    """i7-artlarge (2026-11-08, owner mandate "fix ... the slow low yield aliases"): ``ISPATIALGYM_ARTLARGE_INIT_DRAW`` draws the
    init (start) stance of the large articulation sampler from the whole Phase-1 feasible set (every bearing of the 5 deg lattice in
    +-beta_max, every d of the scan, each with Phase 1's own checks: free floor in the task rooms, the whole target in frame, body and
    trunk sweep collision-free) instead of the NEAREST feasible d on one random bearing, the starts near an eval instance's start
    (REPEAT_POS_M / REPEAT_ANG_DEG) last; and step 6 tries up to INIT_ALT_MAX other candidates of that order when the drawn start
    has no path / transition to the manipulation stance (the start does not enter steps 1-5 or the follow). DEFAULT ON since the A/B
    (lead #44 under the owner's mandate; $OFH/artlarge_20261108): unset = on; ``ISPATIALGYM_ARTLARGE_INIT_DRAW=0`` = the nearest-d draw."""
    return os.environ.get("ISPATIALGYM_ARTLARGE_INIT_DRAW", "1") != "0"


def start_near(pose2d: Sequence[float], starts: Sequence[Sequence[float]], pos_m: float = 0.25, ang_deg: float = 15.0) -> bool:
    """``pose2d`` (x, y, yaw) is within ``pos_m`` and ``ang_deg`` of one of ``starts`` (the repeat rule's start-base half)."""
    x, y, yaw = (float(v) for v in pose2d[:3])
    for s in starts:
        if math.hypot(x - float(s[0]), y - float(s[1])) > pos_m:
            continue
        gap = abs(math.degrees((yaw - float(s[2]) + math.pi) % (2.0 * math.pi) - math.pi))
        if gap <= ang_deg:
            return True
    return False


def order_init_candidates(poses: Sequence[Sequence[float]], eval_starts: Sequence[Sequence[float]], rng: np.random.Generator,
                          pos_m: float = 0.25, ang_deg: float = 15.0) -> Tuple[List[int], int]:
    """The draw order of the init candidates ``poses`` (indices): the ones away from every eval start in random order, then the ones near
    an eval start (:func:`start_near`) in random order; and how many are near."""
    far = [i for i, p in enumerate(poses) if not start_near(p, eval_starts, pos_m, ang_deg)]
    near = [i for i, p in enumerate(poses) if start_near(p, eval_starts, pos_m, ang_deg)]
    far = [far[int(k)] for k in rng.permutation(len(far))] if far else []
    near = [near[int(k)] for k in rng.permutation(len(near))] if near else []
    return far + near, len(near)


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def fraction_key(k: float) -> str:
    """Sidecar key of a joint fraction (``"0.0"``, ``"0.25"``, …; F6 ``Articulation.waypoints`` keys)."""
    return f"{float(k):g}" if float(k) not in (0.0, 1.0) else ("0.0" if float(k) == 0.0 else "1.0")


def waypoint_values(q_init: float, q_goal: float, fractions: Sequence[float] = WAYPOINT_FRACTIONS) -> List[Tuple[float, float]]:
    """``(k, q_k)`` with q_k = q_init + k·(q_goal − q_init) (§5.3)."""
    return [(float(k), float(q_init) + float(k) * (float(q_goal) - float(q_init))) for k in fractions]


def required_fractions(passive_from: Optional[float], fractions: Sequence[float] = WAYPOINT_FRACTIONS, mode: Any = "auto") -> List[float]:
    """§5.3 required handle waypoints: every k up to and including ``passive_from`` (the first k from which the link
    reaches q_goal passively); none passes → all of them. ``mode`` ``True`` forces q_goal (k = 1) in, ``False`` out."""
    fr = [float(k) for k in fractions]
    if mode is True:
        return fr
    if mode is False:
        return [k for k in fr if k < 1.0] or fr[:1]
    if passive_from is None:
        return fr
    return [k for k in fr if k <= float(passive_from) + 1e-9]


def arm_rule(offset_left_m: float, threshold: float = 0.10, override: Any = "auto") -> str:
    """§5.3 arm rule: the arm on the handle's side of the face centre — > ``threshold`` to the robot's left → left arm,
    > ``threshold`` to the right → right arm, else right; ``override`` (``generation.arm``) wins when it names an arm."""
    if override in ("left", "right"):
        return str(override)
    if offset_left_m > threshold:
        return "left"
    return "right"


def follow_mode(joint_type: str, axis_world) -> str:
    """§5.3 ``follow_mode``: **base** for vertical-axis revolute joints and horizontal prismatic joints (h(q) changes
    only in x, y, yaw), **arm** for horizontal-axis revolute joints (pitch / height change)."""
    a = th.as_tensor(axis_world, dtype=th.float32)
    a = a / th.linalg.norm(a)
    vertical = abs(float(a[2])) > math.cos(math.radians(20.0))
    kind = joint_type.lower()
    if "prismatic" in kind or "slider" in kind:
        return "base" if not vertical else "arm"
    return "base" if vertical else "arm"


ROOM_DOOR_CATEGORIES = ("door",)
MIN_REACHABLE_OPENING = {"revolute_deg": 50.0, "prismatic_m": 0.10}
REACH_GRID_STEP = {"revolute_deg": 10.0, "prismatic_m": 0.05}
#: A4 reach margin (2026-09-18, dishwasher 0917-213723): the arm's reach boundary along h(q) is refined by bisection
#: (``reach_refine_steps`` halvings of the last grid step) and the goal is drawn at most ``reach_margin`` INSIDE it. A
#: goal at the boundary itself left the position drives saturated against the door's joint friction 7 mm short of
#: the last handle pose (joint4 0.052 rad off its target, 25 N·m cap) - the margin keeps torque authority for the last
#: centimetres. The margin does not apply when the sweep reaches its end (the joint's own limit stops the link).
REACH_MARGIN = {"revolute_deg": 3.0, "prismatic_m": 0.02}
REACH_REFINE_STEPS = 3
#: §5.3 passive-continuation probe (2026-09-18): a waypoint passes when the link ends within ``passive_tolerance_fraction``
#: of the EVALUATION tolerance (§8 ``joint_tolerance``) of q_goal AND covered at least ``passive_min_progress`` of the
#: remaining gap on its own. The former probe tolerance (0.10 rad) exceeded the 25 % waypoint spacing of a 20° span
#: (0.087 rad), so ``passive_from`` 0.75 passed with the door not moving at all; the friction then held it at 0.25 rad
#: (0917-212441: layout error 0.098 > 0.05).
PASSIVE_TOLERANCE_FRACTION = 0.5
PASSIVE_MIN_PROGRESS = 0.5


def door_class(category: str, joint_type: str) -> str:
    """§5.3 (owner decision A4, 2026‑09‑16) articulation class of a scene joint: ``drawer`` = prismatic joint (base
    mode), ``room_door`` = the revolute leaf of a room ``door`` (hybrid follow), ``cabinet`` = every other revolute door /
    flap of a furniture piece (cabinets, fridge, dishwasher, oven, microwave, ...: arm-only follow from a stance on
    the hinge side)."""
    kind = joint_type.lower()
    if "prismatic" in kind or "slider" in kind:
        return "drawer"
    return "room_door" if category.lower() in ROOM_DOOR_CATEGORIES else "cabinet"


def follow_mode_by_class(cls: str, joint_type: str, axis_world) -> str:
    """``follow_mode`` per class: cabinet → ``arm``; room_door → ``hybrid``; drawer → the geometric rule (``base`` for
    a horizontal prismatic axis, ``arm`` for a vertical one)."""
    if cls == "cabinet":
        return "arm"
    if cls == "room_door":
        return "hybrid"
    return follow_mode(joint_type, axis_world)


def base_follow_policy(configured: Any, direction: str, cls: str, pushed: bool = False) -> str:
    """Policy of a hybrid follow that has to move the base (the arm alone cannot do it from the stance): the variant's
    ``generation.hybrid_follow.policy`` when it names one; else ``sequential`` - pull with the arm, move the base with the
    door stationary and the hand off the handle, pull again (owner 2026-09-27, dishwasher close 0926-065206: arm -> base ->
    arm as the open large-scale variants, never the base driving while the hand pulls / pushes the door) - wherever the
    oracle drives the recorded follow with the hand on the handle. A room door's close - and a cabinet door's close with
    ``generation.close_by_push`` (``pushed``) - is pushed shut instead (``programs.program_articulation_large`` ->
    ``push_close``): its follow is a sampling gate only and keeps the knob's default."""
    if configured:
        return str(configured)
    if direction == "close" and (cls == "room_door" or pushed):
        return str(AH.DEFAULTS["policy"])
    return "sequential"


def axis_is_vertical(axis_world, tol_deg: float = 20.0) -> bool:
    a = th.as_tensor(axis_world, dtype=th.float32)
    a = a / th.linalg.norm(a)
    return abs(float(a[2])) > math.cos(math.radians(tol_deg))


def hinge_side(hinge_xy, contact_xy, tangent_left) -> str:
    """Side of the hinge as seen by the robot facing the face: ``left`` when the hinge lies on the robot's left of the
    handle contact (along the face frame's left tangent), else ``right``."""
    h = th.as_tensor(hinge_xy, dtype=th.float32)[:2]
    c = th.as_tensor(contact_xy, dtype=th.float32)[:2]
    t = th.as_tensor(tangent_left, dtype=th.float32)[:2]
    return "left" if float(th.dot(h - c, t)) > 0.0 else "right"


def hinge_side_lateral(side: str, lam_max: float = 0.60) -> Tuple[float, float]:
    """A4: Λ toward the hinge — the base stands between the handle and the hinge (+λ = robot's left)."""
    return (0.0, float(lam_max)) if side == "left" else (-float(lam_max), 0.0)


def arm_for_hinge_side(side: str) -> str:
    """A4: the base on the hinge side puts the handle on the robot's other side → that arm works the handle."""
    return "right" if side == "left" else "left"


def reach_grid(q_from: float, q_to: float, step: float) -> List[float]:
    """Ascending grid ``q_from + step, q_from + 2·step, ..., q_to`` (``q_to`` always included; empty when q_to <= q_from)."""
    if q_to <= q_from + 1e-9 or step <= 0:
        return []
    n = int(math.floor((q_to - q_from) / step + 1e-9))
    out = [q_from + i * step for i in range(1, n + 1)]
    if not out or out[-1] < q_to - 1e-6:
        out.append(float(q_to))
    return [float(v) for v in out]


def restrict_goal_set(stable: Sequence[float], q_reach: float, lower: float, min_open: float, tol: float = 1e-6) -> List[float]:
    """A4: the goal set of a stance = the stable angles the arm reaches from it (``s <= q_reach``); when none does, the
    reach limit itself is the (single) goal candidate provided it opens the joint by at least ``min_open`` (a 50–70°
    opening is a valid configuration-matching goal) — its passive stability is probed by the caller."""
    out = [float(s) for s in stable if s <= q_reach + tol]
    if out:
        return out
    if q_reach - lower >= min_open - tol:
        return [float(q_reach)]
    return []


def reach_with_margin(q_boundary: float, q_from: float, margin: float, ended_at_miss: bool) -> float:
    """A4 usable reach: ``q_boundary − margin`` (never below ``q_from``) when the sweep ended at a miss — a true reach
    boundary of the arm —, the boundary itself when the arm reached the sweep's end (the joint limit stops the link;
    no authority is needed beyond it)."""
    if not ended_at_miss:
        return float(q_boundary)
    return max(float(q_from), float(q_boundary) - abs(float(margin)))


def passive_reaches(q_k: float, q_after: float, q_goal: float, tol: float, min_progress: float = PASSIVE_MIN_PROGRESS) -> bool:
    """§5.3 passive-continuation verdict of one waypoint: released at ``q_k`` the link ended at ``q_after``; it passes
    when it lies within ``tol`` of ``q_goal`` **and** covered at least ``min_progress`` of the gap ``q_goal − q_k`` on
    its own. A waypoint already inside the band cannot pass without moving (a vacuous pass sent the follow to 75 % of
    a 20° dishwasher opening and the friction held the door there)."""
    if abs(float(q_after) - float(q_goal)) > float(tol):
        return False
    gap = float(q_goal) - float(q_k)
    if abs(gap) < 1e-9:
        return True
    return (float(q_after) - float(q_k)) / gap >= float(min_progress)


def bar_axis(handle: Dict[str, Any]) -> List[float]:
    """Unit direction of a bar handle's long axis in the handle's frame: approach × pinch (both recorded, unit)."""
    a = th.as_tensor(handle["approach_dir"], dtype=th.float32)
    c = th.as_tensor(handle["pinch_axis"], dtype=th.float32)
    ax = th.linalg.cross(a, c)
    n = float(th.linalg.norm(ax))
    return (ax / n).tolist() if n > 1e-6 else [0.0, 0.0, 0.0]


def bar_offsets(handle: Dict[str, Any], fractions: Sequence[float] = (0.5, 1.0), end_margin: float = 0.06, min_step: float = 0.04) -> List[float]:
    """Grasp offsets along a bar handle: 0 first, then ± the ``fractions`` of the usable half-length (bar length / 2 −
    ``end_margin`` for the pads), offsets below ``min_step`` dropped; ``[0.0]`` for knobs / edges / short bars. The
    slide widens the Phase-2 lattice laterally: the Rs_int drawer's only survivors (λ −0.15…+0.05) stood 2 cm from a
    coffee table; its 0.40 m bar admits pinches ±0.14 m off centre, a 0.64 m bar ±0.26 m."""
    if handle.get("kind") != "bar" or handle.get("length") is None:
        return [0.0]
    half = float(handle["length"]) / 2 - end_margin
    out = [0.0]
    for fr in fractions:
        st = round(fr * half, 3)
        if st >= min_step:
            out += [float(st), -float(st)]
    return out


def pinch_classes(pinch_axis_world, kind: str, vertical_threshold: float = 0.7) -> Tuple[str, ...]:
    """§5.3 reach-volume class of the handle grasp: ``fwd_pinch_v`` when |pinch·ẑ| > 0.7 else ``fwd_pinch_h``; a knob
    admits either."""
    if kind == "knob":
        return ("fwd_pinch_v", "fwd_pinch_h")
    p = th.as_tensor(pinch_axis_world, dtype=th.float32)
    p = p / th.linalg.norm(p)
    return ("fwd_pinch_v",) if abs(float(p[2])) > vertical_threshold else ("fwd_pinch_h",)


def joint_tolerance(joint_type: str, dq: float, tolerances: Dict[str, Any]) -> float:
    """§8: max(0.05 rad, 0.10·|Δq|) revolute, max(0.01 m, 0.10·|Δq|) prismatic (floors / fraction from ``generation``)."""
    kind = "prismatic" if ("prismatic" in joint_type.lower() or "slider" in joint_type.lower()) else "revolute"
    floor = float(tolerances.get(kind, 0.05 if kind == "revolute" else 0.01))
    frac = float(tolerances.get("joint_fraction", 0.10))
    return max(floor, frac * abs(float(dq)))


def base_follow_targets(h_eefs: Sequence[Tuple[th.Tensor, th.Tensor]], eef0: Tuple[th.Tensor, th.Tensor], base_pose2d: Sequence[float],
                        base_z: float) -> List[Tuple[float, float, float]]:
    """§5.3 base-follow poses ``b_k = h(q_k) ∘ T_eef→base(q*)`` through the conventions: ``T_eef→base`` from the
    grasp eef pose ``eef0`` (world, at q_init) and the manipulation base pose; one planar base pose per handle pose."""
    base_pose = C.pose2d_to_pose(base_pose2d[0], base_pose2d[1], base_pose2d[2], z=base_z)
    T = C.eef_base_transform(eef0, base_pose)
    return C.base_follow_poses(list(h_eefs), T)


def leaf_clear_of_base(leaf_pts_xy: th.Tensor, base_xy, radius: float = BASE_RADIUS) -> bool:
    """Planar stand-in of "the grasped link ignored for the gripper only": no point of the moving link's footprint (its
    AABB corners / hull points at q_k, world xy) lies inside the base's footprint disc."""
    if leaf_pts_xy is None or len(leaf_pts_xy) == 0:
        return True
    d = th.linalg.norm(th.as_tensor(leaf_pts_xy, dtype=th.float32)[:, :2] - th.as_tensor([float(base_xy[0]), float(base_xy[1])]), dim=1)
    return bool((d > radius).all())


def _vec(x) -> Optional[th.Tensor]:
    return None if x is None else th.as_tensor(x, dtype=th.float32)


def _parse_list(s: str) -> Optional[List[float]]:
    s = (s or "").strip()
    if not s or s.lower() == "null":
        return None
    return [float(v) for v in json.loads(s)]


@dataclass
class HandleRow:
    """One admissible pinch of a joint (from the annotation file or an icl ``handles.csv`` row), object frame, closed
    configuration: ``contact``, ``pinch_axis``, ``approach_dir`` (unit), ``width`` (m), ``kind`` (bar / knob / edge /
    manual), ``source`` (record name / csv row name), ``chosen`` (icl reviewer choice)."""

    kind: str
    contact: List[float]
    pinch_axis: List[float]
    approach_dir: List[float]
    width: Optional[float]
    source: str
    chosen: bool = False
    verdict: str = "handle"

    def as_handle(self) -> Dict[str, Any]:
        return {"kind": self.kind, "contact": list(self.contact), "pinch_axis": list(self.pinch_axis), "approach_dir": list(self.approach_dir),
                "width": self.width, "source": self.source}


def rank_handle_rows(rows: Sequence[HandleRow]) -> List[HandleRow]:
    """Admissible rows in the §5.3 order: the reviewer's ``chosen`` row first, then bars, knobs, reviewer pinches
    (``manual``), edge pinches; ``edge_blocked`` / ``skip`` / ``ungraspable`` rows are not admissible."""
    order = {"bar": 0, "knob": 1, "manual": 2, "edge": 3}
    ok = [r for r in rows if r.verdict in ("handle", "edge", "manual", "manual_unscaled")]
    return sorted(ok, key=lambda r: (0 if r.chosen else 1, order.get(r.kind, 4), r.source))


def joint_excluded(rows: Sequence[HandleRow]) -> bool:
    """A joint with a ``skip`` / ``ungraspable`` verdict and no admissible pinch is excluded (§5.3)."""
    return any(r.verdict in ("skip", "ungraspable") for r in rows) and not rank_handle_rows(rows)


def handle_row_matches(row: HandleRow, name: str) -> bool:
    """``generation.handle_row`` names a candidate by its source (``<key>:j_link_2#C``, ``record:bar``) or by the icl row
    name alone (``j_link_2#C``)."""
    name = str(name).strip()
    return bool(name) and (row.source == name or row.source.split(":", 1)[-1] == name)


def pick_handle_row(rows: Sequence[HandleRow], name: Optional[str] = None, width_range: Sequence[float] = PINCH_WIDTH_RANGE,
                    where: str = "", sources: str = "") -> Tuple[HandleRow, List[HandleRow]]:
    """The pinch of a joint among its candidate rows -> ``(row, ranked rows)``. Without ``name``: the first row of
    :func:`rank_handle_rows`, or the next ranked row inside ``width_range`` when its width is outside (a row without a
    width passes). With ``name`` (``generation.handle_row``, owner 2026-10-12 02:20): that row before the ranking, its
    width checked against ``width_range`` with no fall-back to another row. ``Reject("spec")`` for a name no admissible
    row carries, ``Reject("missing_handle")`` when no row qualifies (``where`` / ``sources`` name the joint and the files)."""
    lo, hi = float(width_range[0]), float(width_range[1])
    if not 0.0 <= lo <= hi:
        raise Reject("spec", f"{where}: pinch width range {[1000 * w for w in (lo, hi)]} mm is empty")
    ranked = rank_handle_rows(rows)

    def fits(r: HandleRow) -> bool:
        return r.width is None or lo <= float(r.width) <= hi

    if name:
        named = [r for r in ranked if handle_row_matches(r, name)]
        if not named:
            raise Reject("spec", f"{where}: handle_row {name!r} is none of the admissible rows {[r.source for r in ranked]}")
        row = named[0]
        if not fits(row):
            raise Reject("missing_handle", f"{where}: handle_row {row.source} pinch width {1000 * float(row.width):.0f} mm outside {[1000 * w for w in (lo, hi)]} mm")
        return row, [row] + [r for r in ranked if r is not row]
    if not ranked:
        raise Reject("missing_handle", f"{where}: no bar / knob / edge / reviewer pinch in {sources}")
    if fits(ranked[0]):
        return ranked[0], ranked
    # the reviewer's / record's next pinch inside the range, if any
    alt = [r for r in ranked[1:] if fits(r)]
    if not alt:
        raise Reject("missing_handle", f"{where}: pinch width {1000 * float(ranked[0].width):.0f} mm outside {[1000 * w for w in (lo, hi)]} mm")
    return alt[0], ranked


def icl_handle_rows(key: str, joint: str, scene: Optional[str], instance: Optional[str], path: str = HANDLES_CSV,
                    instance_scale: Optional[Sequence[float]] = None) -> List[HandleRow]:
    """icl ``require/handles.csv`` rows of ``(key, joint)``: rows measured on this scene instance (``scene`` /
    ``instance`` columns) are taken as they are; canonical rows are scaled by the instance's per-axis scale
    (positions per axis, directions re-normalised, ``width_m`` along the pinch axis)."""
    if not os.path.exists(path):
        return []
    out: List[HandleRow] = []
    scale = None if instance_scale is None else th.as_tensor([float(v) for v in instance_scale], dtype=th.float32)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("key") != key or row.get("joint") != joint:
                continue
            row_scene, row_inst = row.get("scene") or "", row.get("instance") or ""
            if row_scene and (row_scene != scene or row_inst != instance):
                continue  # another scene instance of the same model
            pos, a, c = _parse_list(row.get("position")), _parse_list(row.get("axis_approach")), _parse_list(row.get("axis_close"))
            if pos is None or a is None or c is None:
                continue
            verdict = (row.get("verdict") or "").strip()
            kind = (row.get("kind") or "").strip()
            width = float(row["width_m"]) if row.get("width_m") else None
            name = row.get("name") or ""
            if "#F" in name or kind.startswith("edge"):
                kind_out = "edge"
            elif "#M" in name or verdict in ("manual", "manual_unscaled"):
                kind_out = "manual"
            else:
                kind_out = kind if kind in ("bar", "knob") else "bar"
            rec = {"contact": pos, "pinch_axis": c, "approach_dir": a, "width": width}
            if not row_scene and scale is not None and (verdict != "manual" or verdict == "manual_unscaled"):
                rec = scale_handle(rec, scale)
            out.append(HandleRow(kind_out, rec["contact"], rec["pinch_axis"], rec["approach_dir"], rec.get("width"), f"{key}:{name}",
                                 chosen=(row.get("chosen") or "").strip() in ("1", "true", "True"), verdict=verdict))
    return out


def record_handle_rows(joint_rec: Dict[str, Any]) -> List[HandleRow]:
    """Handle candidates of an annotation record's joint entry: bars / knobs, then the edge fallback when exposed."""
    out = []
    for h in joint_rec.get("handles", []) or []:
        if h.get("kind") in ("bar", "knob"):
            out.append(HandleRow(h["kind"], list(h["contact"]), list(h["pinch_axis"]), list(h["approach_dir"]), h.get("width"), f"record:{h['kind']}"))
    fb = joint_rec.get("fallback")
    if fb and fb.get("contact") is not None:
        exposed = fb.get("exposed", True)
        out.append(HandleRow("edge", list(fb["contact"]), list(fb["pinch_axis"]), list(fb["approach_dir"]), fb.get("width"), "record:edge",
                             verdict="edge" if exposed else "edge_blocked"))
    return out


def leaf_plane_approach(handle: Dict[str, Any], jinfo: Dict[str, Any], min_angle_deg: float = 20.0) -> Optional[List[float]]:
    """§5.3 edge pinch: the approach runs **into the exposed edge along the leaf plane** (toward the hinge / along the
    joint's motion), the pinch axis is the leaf normal. The B4 ``edge_door`` fallback rows carry the face normal as
    ``approach_dir`` — parallel to the pinch axis, so ``handle_eef`` builds an accidental frame from the 0.7° skew and
    the rotated frame at an open angle points the approach along the face (no forward reach class: office cabinet
    ``rvunhj_0`` close 0/8 ``no_manipulation_stance``, 2026‑09‑16). Returns the repaired approach (object frame) when
    the recorded one is within ``min_angle_deg`` of the pinch axis, else None (the record is kept)."""
    pinch = th.as_tensor(handle.get("pinch_axis"), dtype=th.float32)
    approach = th.as_tensor(handle.get("approach_dir", handle.get("approach_dir_obj")), dtype=th.float32)
    pinch = pinch / max(float(th.linalg.norm(pinch)), 1e-9)
    approach = approach / max(float(th.linalg.norm(approach)), 1e-9)
    if abs(float(th.dot(pinch, approach))) < math.cos(math.radians(min_angle_deg)):
        return None
    axis = th.as_tensor(jinfo.get("axis_dir_obj") or [0.0, 0.0, 1.0], dtype=th.float32)
    axis = axis / max(float(th.linalg.norm(axis)), 1e-9)
    kind = str(jinfo.get("type", "")).lower()
    if "prismatic" in kind or "slider" in kind:
        along = axis  # a drawer front's edge: into the edge along the motion axis
    else:
        along = th.cross(pinch, axis, dim=0)  # in the leaf plane, ⟂ the hinge axis
    if float(th.linalg.norm(along)) < 1e-6:
        return None
    along = along / float(th.linalg.norm(along))
    hinge = jinfo.get("hinge_point_obj")
    if hinge is not None and "prismatic" not in kind:
        to_hinge = th.as_tensor(hinge, dtype=th.float32) - th.as_tensor(handle["contact"], dtype=th.float32)
        if float(th.dot(along, to_hinge)) < 0:
            along = -along  # from the free edge toward the hinge
    return [float(v) for v in along.tolist()]


def sane_hinge_point(jinfo: Dict[str, Any], measured: Optional[th.Tensor] = None) -> Optional[List[float]]:
    """Hinge point of a revolute joint on the plane through the moving link's bbox centre (the inventories recover the
    hinge by least squares, unconstrained along the axis: B4 records show z ~ 1e8). ``measured`` (from the joint prim)
    replaces a point still > 1 m off the link after the projection."""
    hinge, axis = _vec(jinfo.get("hinge_point_obj")), _vec(jinfo.get("axis_dir_obj"))
    if hinge is None or axis is None or float(th.linalg.norm(axis)) < 1e-9:
        return None if hinge is None else hinge.tolist()
    a = axis / th.linalg.norm(axis)
    if jinfo.get("link_bbox_min") is None or jinfo.get("link_bbox_max") is None:
        return hinge.tolist()
    lo, hi = _vec(jinfo["link_bbox_min"]), _vec(jinfo["link_bbox_max"])
    centre = (lo + hi) / 2
    proj = hinge if not bool(th.isfinite(hinge).all()) else hinge + a * th.dot(centre - hinge, a)
    if bool(th.isfinite(proj).all()) and float(th.linalg.norm(proj - centre)) <= 1.0:
        return proj.tolist()
    if measured is not None:
        m = measured + a * th.dot(centre - measured, a)
        if bool(((m >= lo - 0.15) & (m <= hi + 0.15)).all()):
            return m.tolist()
    return proj.tolist()


# ----------------------------------------------------------------------------------------------------------------
# push finish of a cabinet door's closing (owner 2026-10-12 ~04:20, close_large#03)
# ----------------------------------------------------------------------------------------------------------------

#: ``generation.push_finish`` (``true``, or a dict overriding these): the closing of a revolute cabinet door that the arm follows
#: only part of the way from a stance is finished by the closed hand pushing the leaf's outer face the rest of the way shut
#: (``programs.program_articulation_large`` -> grasp, follow to ``articulation.follow_to``, release, ``push_close`` with the planned
#: chunk). Off when the key is absent.
PUSH_FINISH_DEFAULTS: Dict[str, Any] = {
    "insets_m": (0.15, 0.10, 0.22),  # contact distance from the free edge along the leaf (the room door's push_inset_m first)
    "height_offsets_m": (-0.05, 0.05, -0.15),  # contact height relative to the handle grasp (the room door's 5 cm drop first)
    "tilts_deg": (0.0, 20.0, 35.0),  # the fist's approach turned from the face normal toward the free edge (the hinge-side stance)
    "closings": ("leaf", "vertical"),  # the pads close along the leaf (the bar pinch's wrist roll) or vertically (the room door's)
    "min_radius_m": 0.10,  # the contact keeps this far from the hinge line
    "edge_margin_m": 0.05,  # ... this far inside the leaf's top and bottom
    "handle_clear_m": 0.06,  # ... and this far from the handle segment
    "pre_m": 0.10,  # pre-contact distance back along the approach (the room door's push_pre_m)
    "band_fraction": 0.5,  # the push ends within this fraction of the §8 tolerance of the lower limit
    "arc_step_deg": 5.0,  # spacing of the checked arc poses
    "max_checks": 8,  # candidates checked exactly after the batched screen
}
PUSH_CLOSINGS = ("leaf", "vertical")


def push_finish_config(raw: Any) -> Optional[Dict[str, Any]]:
    """``generation.push_finish`` -> its options (:data:`PUSH_FINISH_DEFAULTS` with the variant's overrides); None when off (absent,
    ``false``, ``{"enabled": false}``). ``Reject("spec")`` for another type, an unknown key or an unknown closing."""
    if raw is None or raw is False:
        return None
    if raw is True:
        return dict(PUSH_FINISH_DEFAULTS)
    if not isinstance(raw, dict):
        raise Reject("spec", f"generation.push_finish: true / false or a dict of options, got {raw!r}")
    if not raw.get("enabled", True):
        return None
    unknown = sorted(str(k) for k in raw if k != "enabled" and not str(k).startswith("_") and k not in PUSH_FINISH_DEFAULTS)
    if unknown:
        raise Reject("spec", f"generation.push_finish: unknown keys {unknown} (known: {sorted(PUSH_FINISH_DEFAULTS)})")
    out = dict(PUSH_FINISH_DEFAULTS)
    out.update({k: v for k, v in raw.items() if k in PUSH_FINISH_DEFAULTS})
    bad = [c for c in out["closings"] if c not in PUSH_CLOSINGS]
    if bad:
        raise Reject("spec", f"generation.push_finish.closings: {bad} not in {list(PUSH_CLOSINGS)}")
    return out


def push_finish_applies(direction: str, joint_type: str, cls: str, mode: str, vertical_axis: bool) -> bool:
    """Whether the push finish can serve a joint: the closing of a revolute ``cabinet`` door on a vertical hinge that the arm alone
    follows (A4 ``follow_mode`` arm). A room door is pushed shut from its open angle (``push_close``), drawers and flaps keep the
    handle all the way."""
    kind = joint_type.lower()
    return direction == "close" and "revolute" in kind and cls == "cabinet" and mode == "arm" and bool(vertical_axis)


def release_fraction(q_init: float, q_reach: float, q_goal: float) -> float:
    """The fraction of the joint motion ``q_init -> q_goal`` the hand covers when its reach ends at ``q_reach``, floored to 1e-4 so
    the release never lies past the value the reach sweep verified; 0 = no handle progress, 1 = all of it."""
    span = float(q_goal) - float(q_init)
    if abs(span) < 1e-9:
        return 1.0
    frac = (float(q_reach) - float(q_init)) / span
    return min(1.0, max(0.0, math.floor(frac * 1e4 + 1e-6) / 1e4))


def finish_fractions(fractions: Sequence[float], k_release: float) -> List[float]:
    """The recorded waypoint fractions of a push-finish instance: the standard ones plus the release fraction (where the hand lets
    go), sorted, no duplicate within 1e-6."""
    out = sorted(float(k) for k in fractions)
    if all(abs(k - float(k_release)) > 1e-6 for k in out):
        out = sorted(out + [float(k_release)])
    return out


def finish_required(required: Sequence[float], k_release: float) -> List[float]:
    """The handle waypoints a push-finish instance requires: the required ones up to the release (the hand's part ends there)."""
    return [float(k) for k in required if float(k) <= float(k_release) + 1e-9]


def push_reaches_goal(q_to: float, q_lower: float, band: float) -> bool:
    """The planned push ends within ``band`` of the closed (lower) limit."""
    return float(q_to) - float(q_lower) <= float(band) + 1e-9


def arc_values(q_from: float, q_to: float, step: float) -> List[float]:
    """Joint values from ``q_from`` (excluded) to ``q_to`` (included) at most ``step`` apart."""
    n = max(1, int(math.ceil(abs(float(q_to) - float(q_from)) / max(float(step), 1e-6) - 1e-9)))
    return [float(q_from) + (float(q_to) - float(q_from)) * k / n for k in range(1, n + 1)]


@dataclass
class ClosedLeaf:
    """The closed leaf of a door on a vertical hinge, world frame (push finish): ``hinge`` = a point of the hinge line, ``s`` = the
    swing sign (joint value q turns the leaf about +z by s·(q − ``q_closed``)), ``n0`` = the outer-face normal and ``d0`` = the
    direction from the hinge line toward the free edge (unit, horizontal), ``width`` = hinge line -> free edge, ``z_lo`` / ``z_hi``
    = the leaf's height span, ``face`` = the outer face's offset along n0 from the hinge line, ``handle_a`` / ``handle_b`` = the
    handle segment (a bar's ends; one point for a knob)."""

    hinge: th.Tensor
    s: float
    n0: th.Tensor
    d0: th.Tensor
    width: float
    z_lo: float
    z_hi: float
    face: float
    handle_a: th.Tensor
    handle_b: th.Tensor
    q_closed: float = 0.0

    def coords(self, p) -> Tuple[float, float]:
        """(distance from the hinge line along d0, height) of a world point of the closed leaf."""
        p = th.as_tensor(p, dtype=th.float32)
        return float(th.dot((p - self.hinge)[:2], self.d0[:2])), float(p[2])


def closed_leaf(obj_pose, jinfo: Dict[str, Any], handle: Dict[str, Any]) -> ClosedLeaf:
    """The :class:`ClosedLeaf` of a revolute joint from its joint info (object frame, closed: ``hinge_point_obj``, ``axis_dir_obj``,
    ``outward_normal``, ``link_bbox_min`` / ``link_bbox_max``, ``closed_q``) and handle (``contact``, ``kind``, a bar's ``length``)
    at the object pose ``obj_pose``. The outer face lies ``HANDLE_EEF_OFFSET`` behind a bar / knob / reviewer contact (the convention:
    the annotated contact is 12 mm off the face), on the box's front for an edge pinch; without a link box the leaf is the handle's
    radius + 3.5 cm wide and ±0.25 m tall about it. ``ValueError`` for a hinge that is not vertical."""
    p, q = (th.as_tensor(v, dtype=th.float32).reshape(-1) for v in obj_pose)
    axis = C.unit(C.quat_rotate(q, jinfo.get("axis_dir_obj") or [0.0, 0.0, 1.0]))
    if abs(float(axis[2])) < math.cos(math.radians(20.0)):
        raise ValueError(f"the hinge axis {[round(float(v), 3) for v in axis]} is not vertical")
    hinge = p + C.quat_rotate(q, jinfo["hinge_point_obj"])
    n0 = C.quat_rotate(q, jinfo.get("outward_normal") or [1.0, 0.0, 0.0])
    n0 = C.unit(th.tensor([float(n0[0]), float(n0[1]), 0.0]))
    hc = p + C.quat_rotate(q, handle["contact"])
    d0 = th.tensor([-float(n0[1]), float(n0[0]), 0.0])
    if float(th.dot(d0[:2], (hc - hinge)[:2])) < 0.0:
        d0 = -d0
    kind = str(handle.get("kind") or "bar")
    face_h = float(th.dot((hc - hinge)[:2], n0[:2])) - (C.HANDLE_EEF_OFFSET if kind in ("bar", "knob", "manual") else 0.0)
    lo, hi = jinfo.get("link_bbox_min"), jinfo.get("link_bbox_max")
    r_h = float(th.dot((hc - hinge)[:2], d0[:2]))
    if lo is not None and hi is not None:
        corners = G.aabb_corners(th.as_tensor(lo, dtype=th.float32), th.as_tensor(hi, dtype=th.float32))
        cw = p + C.quat_rotate(q, corners)
        rel = cw - hinge
        width = float((rel[:, :2] @ d0[:2]).max())
        along_n = rel[:, :2] @ n0[:2]
        face = float(along_n.max()) if kind not in ("bar", "knob", "manual") else min(max(face_h, float(along_n.min())), float(along_n.max()))
        z_lo, z_hi = float(cw[:, 2].min()), float(cw[:, 2].max())
    else:
        width, face, z_lo, z_hi = r_h + 0.035, face_h, float(hc[2]) - 0.25, float(hc[2]) + 0.25
    half = 0.5 * float(handle.get("length") or 0.0) if kind == "bar" else 0.0
    ax = C.quat_rotate(q, bar_axis(handle)) if half > 0.0 else th.zeros(3)
    return ClosedLeaf(hinge=hinge, s=1.0 if float(axis[2]) > 0.0 else -1.0, n0=n0, d0=d0, width=width, z_lo=z_lo, z_hi=z_hi, face=face,
                      handle_a=hc - half * ax, handle_b=hc + half * ax, q_closed=float(jinfo.get("closed_q", jinfo.get("lower", 0.0))))


def _segment_distance_2d(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2))
    return math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1])


def push_finish_candidates(leaf: ClosedLeaf, z_ref: float, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The fist contacts on the closed leaf's outer face in the order they are tried: per inset from the free edge (``insets_m``),
    per height offset from the handle grasp height ``z_ref`` (``height_offsets_m``), per approach tilt (``tilts_deg``), per closing
    axis (``closings``). A point closer than ``min_radius_m`` to the hinge line, outside the leaf (``edge_margin_m`` inside its top /
    bottom) or within ``handle_clear_m`` of the handle segment (in the face's (radius, height) plane) is left out."""
    ha, hb = leaf.coords(leaf.handle_a), leaf.coords(leaf.handle_b)
    out: List[Dict[str, Any]] = []
    for inset in cfg["insets_m"]:
        r = leaf.width - float(inset)
        if r < float(cfg["min_radius_m"]):
            continue
        for dz in cfg["height_offsets_m"]:
            z = float(z_ref) + float(dz)
            if z < leaf.z_lo + float(cfg["edge_margin_m"]) or z > leaf.z_hi - float(cfg["edge_margin_m"]):
                continue
            if _segment_distance_2d((r, z), ha, hb) < float(cfg["handle_clear_m"]):
                continue
            for tilt in cfg["tilts_deg"]:
                for closing in cfg["closings"]:
                    out.append({"radius_m": round(r, 4), "inset_m": float(inset), "z": round(z, 4), "dz": float(dz), "tilt_deg": float(tilt), "closing": str(closing)})
    return out


def push_contact_pose(leaf: ClosedLeaf, cand: Dict[str, Any], q: float) -> Tuple[C.Pose, th.Tensor, th.Tensor]:
    """``(fist eef pose, approach, outer-face normal)`` of a candidate with the leaf at joint value ``q``: the face point turned with
    the leaf about the hinge line by s·(q − q_closed), the approach into the face turned ``tilt_deg`` toward the free edge, the
    pads closing along the leaf or vertically, the fingertips on the point (``conventions.fingertip_eef``)."""
    turn = C.quat_from_axis_angle([0.0, 0.0, 1.0], leaf.s * (float(q) - leaf.q_closed))
    d, n = C.quat_rotate(turn, leaf.d0), C.quat_rotate(turn, leaf.n0)
    point = th.tensor([float(leaf.hinge[0]), float(leaf.hinge[1]), float(cand["z"])]) + float(cand["radius_m"]) * d + leaf.face * n
    t = math.radians(float(cand["tilt_deg"]))
    a = C.unit(-math.cos(t) * n + math.sin(t) * d)
    closing = d if cand["closing"] == "leaf" else th.tensor([0.0, 0.0, 1.0])
    return C.fingertip_eef(point, a, closing, clearance=0.0), a, n


# ----------------------------------------------------------------------------------------------------------------
# a cabinet door's close pushed shut room-door style (owner 2026-10-12, close_large#03)
# ----------------------------------------------------------------------------------------------------------------

#: ``generation.close_by_push`` (``true``, or a dict overriding these): the closing of a revolute cabinet door on a vertical hinge is
#: pushed shut as a room door's is - no grasp: the program drives init -> manipulation and ``push_close`` pushes the leaf's outer face
#: in arcs about the hinge from a stance it picks per chunk (``programs.program_articulation_large``); the follow the sampler checks
#: is a sampling gate only (the room door's close policy, :func:`base_follow_policy`). The values are ``push_close`` options
#: (``PushClose`` fields), recorded in ``articulation.push_close`` with the leaf they act on (:func:`close_by_push_record`). Off when
#: the key is absent.
CLOSE_BY_PUSH_DEFAULTS: Dict[str, Any] = {
    "push_inset_m": 0.15,  # contact distance from the free edge (the room door's: 0.25 m from the hinge line of a 0.40 m leaf)
    "push_height_drop_m": 0.05,  # contact below the closed handle (the room door's)
    "chunk_deg": 45.0,  # arc per chunk: 0.20 m of fist travel at a 0.25 m contact radius (a room door's 20 deg at ~0.68 m: 0.24 m)
    "drag_m": 0.08,  # base drag toward the leaf when an arc gains < 2 deg: ~18 deg at 0.25 m (the room door's 0.20 m at ~0.68 m: ~17 deg)
    # the contact point this far in front of the measured outer face: the push pose puts the pad centre on it, so 3.07 cm puts the
    # fingertips on the face (the push finish's convention) instead of 3 cm into the leaf the stance search's IK sees (room door: 0)
    "contact_clearance_m": round(C.FINGERTIP_LENGTH - C.CONTACT_CENTER_Z, 4),
}


def close_by_push_config(raw: Any) -> Optional[Dict[str, float]]:
    """``generation.close_by_push`` -> its options (:data:`CLOSE_BY_PUSH_DEFAULTS` with the variant's overrides); None when off (absent,
    ``false``, ``{"enabled": false}``). ``Reject("spec")`` for another type, an unknown key or a value out of range."""
    if raw is None or raw is False:
        return None
    if raw is True:
        return dict(CLOSE_BY_PUSH_DEFAULTS)
    if not isinstance(raw, dict):
        raise Reject("spec", f"generation.close_by_push: true / false or a dict of options, got {raw!r}")
    if not raw.get("enabled", True):
        return None
    unknown = sorted(str(k) for k in raw if k != "enabled" and not str(k).startswith("_") and k not in CLOSE_BY_PUSH_DEFAULTS)
    if unknown:
        raise Reject("spec", f"generation.close_by_push: unknown keys {unknown} (known: {sorted(CLOSE_BY_PUSH_DEFAULTS)})")
    out = dict(CLOSE_BY_PUSH_DEFAULTS)
    out.update({k: v for k, v in raw.items() if k in CLOSE_BY_PUSH_DEFAULTS})
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in out.values()):
        raise Reject("spec", f"generation.close_by_push: numbers expected, got {out}")
    out = {k: float(v) for k, v in out.items()}
    if not (out["push_inset_m"] > 0.0 and 0.0 < out["chunk_deg"] <= 90.0 and out["drag_m"] >= 0.0 and 0.0 <= out["contact_clearance_m"] < 0.10):
        raise Reject("spec", f"generation.close_by_push: push_inset_m > 0, chunk_deg in (0, 90], drag_m >= 0 and contact_clearance_m in [0, 0.10), got {out}")
    return out


def close_by_push_applies(direction: str, joint_type: str, cls: str, vertical_axis: bool) -> bool:
    """Whether ``generation.close_by_push`` can serve a joint: the closing of a revolute ``cabinet`` door on a vertical hinge (the room
    door's ``push_close`` turns the leaf about a vertical hinge line; a room door is pushed shut anyway)."""
    return direction == "close" and "revolute" in joint_type.lower() and cls == "cabinet" and bool(vertical_axis)


def close_by_push_record(leaf: ClosedLeaf, cfg: Dict[str, float]) -> Dict[str, Any]:
    """``articulation.push_close`` of a cabinet door pushed shut room-door style: the push's options (``cfg``) and the closed leaf they
    act on (:func:`closed_leaf`) - ``leaf_width_m`` (hinge line -> free edge), ``face_offset_m`` (the outer face in front of the hinge
    line), ``handle_standoff_m`` (the handle contact in front of the hinge line: the hinge -> handle direction turned back by
    atan(standoff / radius) runs along the leaf), ``leaf_z`` (its height span), ``swing_sign`` and ``hinge_world`` (the joint record's;
    the program checks both against the handle waypoints)."""
    hc = 0.5 * (leaf.handle_a + leaf.handle_b)
    standoff = float(th.dot((hc - leaf.hinge)[:2], leaf.n0[:2]))
    return {**{k: float(v) for k, v in cfg.items()}, "leaf_width_m": round(float(leaf.width), 4), "face_offset_m": round(float(leaf.face), 4),
            "handle_standoff_m": round(standoff, 4), "leaf_z": [round(float(leaf.z_lo), 4), round(float(leaf.z_hi), 4)], "swing_sign": float(leaf.s),
            "hinge_world": [round(float(v), 4) for v in leaf.hinge.tolist()]}


# ----------------------------------------------------------------------------------------------------------------
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*SUBTASKS)
class ArticulationLarge(SamplerV2):
    """One class for open and close; the direction comes from the subtask id (``generation.direction`` agrees)."""

    #: ``generation.close_by_push`` options when the joint qualifies (:func:`close_by_push_applies`), else None
    close_by_push_cfg: Optional[Dict[str, float]] = None
    #: i7-artlarge (ISPATIALGYM_ARTLARGE_INIT_DRAW): the init stance step 6 connected (the drawn one unless an alternative took over), the
    #: other init candidates in draw order, and the Phase-1 feasible init set per q_init
    _init_used: Optional[Stance] = None
    _init_alts: Sequence[Stance] = ()
    _init_sets: Optional[Dict[Any, List[Stance]]] = None

    def __init__(self, ctx):
        super().__init__(ctx)
        self.direction = "close" if "close" in self.subtask else "open"
        if self.gp.gen("direction") in ("open", "close"):
            self.direction = str(self.gp.gen("direction"))
        if self.gp.bddl is None:  # facade: the v1 writer's definition id
            self.definition_id = V1_DEFINITION_ID[self.direction]
        self.target_role = "target" if "target" in self.gp.roles else ("artic" if "artic" in self.gp.roles else "target")
        self.handle: Dict[str, Any] = {}
        self.bar_offset: float = 0.0  # grasp slide along a bar handle (m, along bar_axis), per attempt
        self.jinfo: Dict[str, Any] = {}
        self.joint_name: Optional[str] = None
        self.joint = None
        self.lower = self.upper = 0.0
        self.stable: List[float] = []
        self.passive: Dict[str, Optional[float]] = {}
        self.face_view: Optional[FaceFrame] = None
        self.face_manip: Optional[FaceFrame] = None
        self.arm = "right"
        self.mode = "base"
        self.door_class = "cabinet"
        self.hinge_side: Optional[str] = None
        self.lam_range: Optional[Tuple[float, float]] = None
        self.pinch_sign = 1.0
        self.hull: Optional[th.Tensor] = None
        self.handle_rows: List[HandleRow] = []
        #: ``generation.joint_friction``: PhysX articulation joint friction applied to the moved joint (None = the asset's
        #: stock 0.2); recorded as a TRO ``constraints`` record the loader re-applies (as ``articulation_small``)
        self.joint_friction: Optional[float] = None
        #: ``generation.push_finish`` options when the joint qualifies (:func:`push_finish_applies`), else None
        self.push_finish_cfg: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ roles / bindings

    def v1_role_variants(self) -> Dict[str, str]:
        return {"artic": "target"} if self.target_role == "target" else {}

    def bind_scene_roles(self):
        req = self.choice.bindings.get(self.target_role)
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.variant}: scenes.{self.scene}.bindings.{self.target_role} must list scene object names")
        obj = self.world.obj(names[0])
        self.bind_scene_object(self.target_role, obj)
        # the floor the robot will stand on: the ``floors`` object under the approach in front of the target's face (the
        # inventory record's face centre + outward normal, 1.5 m out), when the binding does not name one explicitly - a room
        # can hold several overlapping floor objects and the BDDL ``(ontop agent floor)`` is judged against the bound one
        # (door_crtrpa_0 0919-153516: garden_0's first floor object was not the slab under the stances -> init_false)
        floor_req = self.choice.bindings.get("floor")
        chosen = None
        if not (isinstance(floor_req, list) and floor_req):
            try:
                data, _, _ = self._handle_record(obj)
                joint_name = self.role_option("joint")
                recs = [j for j in (data or {}).get("joints", []) if joint_name is None or j.get("joint") == joint_name]
                rec = recs[0] if recs else None
                if rec is not None and rec.get("face_centre_world") is not None and rec.get("outward_normal_world") is not None:
                    c = th.as_tensor(rec["face_centre_world"], dtype=th.float32)
                    n = th.as_tensor(rec["outward_normal_world"], dtype=th.float32)
                    n[2] = 0.0
                    n = n / max(float(th.linalg.norm(n)), 1e-9)
                    xy = (c + 1.5 * n)[:2]
                    chosen = self.world.floor_object_under(xy) if hasattr(self.world, "floor_object_under") else None
                    if chosen is not None:
                        log.info("%s: floor role bound to %s (under the approach at (%.2f, %.2f), rooms %s)", self.gp.variant, chosen.name, float(xy[0]), float(xy[1]),
                                 getattr(chosen, "in_rooms", None))
            except Exception as e:  # noqa: BLE001 — fall back to the room's floor
                log.debug("floor-under-approach lookup failed: %r", e)
        if chosen is not None:
            self.bind_scene_object("floor", chosen)
        else:
            self.bind_floor_v2()

    @property
    def target(self):
        return self.obj(self.target_role)

    @property
    def tname(self) -> str:
        return self.bindings.one(self.target_role).name

    def role_option(self, key: str, default=None):
        """``roles.<target>.<key>`` (v2 customs; dict form keyed ``<scene>/<name>`` or ``<category>/<model>``), else
        ``generation.<key>`` (the v1 facade carries no role options)."""
        raw = (self.gp.entry.get("roles") or {}).get(self.target_role) or {}
        v = raw.get(key)
        if isinstance(v, dict):
            obj = self.target
            v = v.get(f"{self.scene}/{obj.name}", v.get(f"{obj.category}/{obj.model}", v.get(obj.name, v.get("default"))))
        if v is None:
            v = self.gp.gen(key, default)
        return v

    # ------------------------------------------------------------------ handle annotation

    def _handle_record(self, obj) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
        """(record, path, frame): ``handle_file`` → B4 instance record → canonical library file; frame ``instance``
        for records measured in place, ``canonical`` otherwise; (None, None, "") when none exists."""
        explicit = self.role_option("handle_file")
        cands: List[str] = []
        if explicit:
            cands += [explicit] if os.path.isabs(explicit) else [os.path.join(REPO_ROOT, explicit), os.path.abspath(explicit)]
        cands.append(os.path.join(SCENE_INVENTORY_ROOT, self.world.scene_model, "artic", f"{obj.name}.json"))
        key = self.role_option("handle_key") or f"{obj.category}__{obj.model}"
        cands.append(os.path.join(library_root(), "artic", "handles", f"{key}.json"))
        for p in cands:
            if os.path.exists(p):
                with open(p) as f:
                    data = json.load(f)
                return data, p, ("instance" if data.get("scene") else "canonical")
        return None, None, ""

    def load_handle(self, obj):
        """The joint (``roles.target.joint``), its handle (§5.3 rules incl. the icl rows; ``generation.handle_row`` names
        the row, ``generation.pinch_width_min_m`` the width floor: :func:`pick_handle_row`) and the joint info in the
        instance frame; ``Reject("missing_handle")`` when the joint has no admissible pinch."""
        key = self.role_option("handle_key") or f"{obj.category}__{obj.model}"
        joint_name = self.role_option("joint")
        data, path, frame = self._handle_record(obj)
        scale = th.as_tensor([float(v) for v in obj.scale], dtype=th.float32) if getattr(obj, "scale", None) is not None else th.ones(3)
        joints = [j for j in (data or {}).get("joints", []) if joint_name is None or j.get("joint") == joint_name]
        if data is not None and not joints:
            raise Reject("spec", f"joint {joint_name!r} not in the handle annotation {os.path.relpath(path, REPO_ROOT)}")
        if joint_name is None:
            if not joints:
                raise Reject("spec", f"{self.gp.variant}: roles.{self.target_role}.joint is required (no annotation to pick the joint from)")
            # no joint pinned: the joint with a handle whose height is nearest the arm's band (0.75 m) first — the
            # record order put the bottom drawer (handle 0.33 m) first for Rs_int bamfsz_1 and the reach volume covers
            # no fwd pinch that low from the Phase-2 trunks (19:48: 822 step-1 survivors, 0 reach the contact set)
            def _height_key(jd):
                hs = [h.get("contact_world", [None, None, None])[2] for h in (jd.get("handles") or []) if h.get("contact_world")]
                return (0 if hs else 1, min((abs(float(z) - 0.75) for z in hs), default=9.0))
            joints = sorted(joints, key=_height_key)
            joint_name = joints[0]["joint"]
        if joint_name not in obj.joints:
            raise Reject("spec", f"{obj.name} has no joint {joint_name!r} (joints {list(obj.joints)})")
        j = joints[0] if joints else {"joint": joint_name, "type": "RevoluteJoint" if obj.joints[joint_name].is_revolute else "PrismaticJoint"}
        jinfo = dict(j) if frame == "instance" else (scale_joint_info(j, scale) if frame == "canonical" else dict(j))
        rows = record_handle_rows(jinfo) if data is not None else []
        rows += icl_handle_rows(key, joint_name, self.world.scene_model, obj.name, instance_scale=None if frame == "instance" else scale.tolist())
        if joint_excluded(rows):
            raise Reject("missing_handle", f"{obj.name}/{joint_name}: icl verdict skip / ungraspable and no admissible pinch")
        # generation.handle_row / pinch_width_min_m (owner 2026-10-12 02:20): the variant's own handle row and width floor
        named = self.role_option("handle_row")
        width_range = (float(self.gp.gen("pinch_width_min_m", PINCH_WIDTH_RANGE[0])), PINCH_WIDTH_RANGE[1])
        row, ranked = pick_handle_row(rows, named, width_range, where=f"{obj.name}/{joint_name}",
                                      sources=f"{path and os.path.relpath(path, REPO_ROOT)} or {HANDLES_CSV}")
        handle = row.as_handle()
        width = handle.get("width")
        # bar length (instance frame) for the grasp slide along the bar (§5.3 bars: any point along the bar is a pinch)
        for hrec in (jinfo.get("handles") or []):
            if hrec.get("kind") == handle.get("kind") and hrec.get("length") is not None:
                handle["length"] = float(hrec["length"])
                break
        jt = obj.joints[joint_name]
        jinfo.setdefault("type", "RevoluteJoint" if jt.is_revolute else "PrismaticJoint")
        jinfo.setdefault("lower", float(jt.lower_limit))
        jinfo.setdefault("upper", float(jt.upper_limit))
        jinfo.setdefault("closed_q", float(jinfo.get("lower", jt.lower_limit)))
        if jinfo.get("axis_dir_obj") is None:
            jinfo["axis_dir_obj"] = self._axis_from_prim(obj, joint_name)
        jinfo["hinge_point_obj"] = sane_hinge_point(jinfo, self._hinge_from_prim(obj, joint_name))
        repaired = leaf_plane_approach(handle, jinfo)
        if repaired is not None:
            log.info("%s: %s/%s edge pinch: recorded approach %s is parallel to the pinch axis %s — replaced by the leaf-plane approach %s", self.gp.variant, obj.name,
                     joint_name, handle.get("approach_dir"), handle.get("pinch_axis"), [round(v, 3) for v in repaired])
            handle["approach_dir_recorded"] = list(handle.get("approach_dir", []))
            handle["approach_dir"] = repaired
        self.joint_name, self.joint, self.handle, self.jinfo, self.handle_rows = joint_name, jt, handle, jinfo, ranked
        self.lower, self.upper = float(jt.lower_limit), float(jt.upper_limit)
        self.template_facts["handle"] = {"source": path and os.path.relpath(path, REPO_ROOT), "frame": frame, "row": row.source, "kind": row.kind,
                                         "width_mm": None if width is None else round(1000 * float(width), 1),
                                         "contact_obj": handle["contact"], "pinch_axis_obj": handle["pinch_axis"], "approach_dir_obj": handle["approach_dir"],
                                         "joint": joint_name, "type": jinfo["type"], "axis_dir_obj": jinfo["axis_dir_obj"], "hinge_point_obj": jinfo["hinge_point_obj"],
                                         "instance_scale": scale.tolist(), "n_rows": len(ranked)}
        if named:
            self.template_facts["handle"]["handle_row"] = str(named)
        if width_range[0] != PINCH_WIDTH_RANGE[0]:
            self.template_facts["handle"]["pinch_width_range_mm"] = [round(1000 * w, 1) for w in width_range]
        log.info("%s: handle of %s/%s from %s (%s%s): %s", self.gp.variant, obj.name, joint_name, self.template_facts["handle"]["source"], row.source,
                 ", generation.handle_row" if named else "", handle)

    def _hinge_from_prim(self, obj, joint_name: str) -> Optional[th.Tensor]:
        try:
            joint = obj.joints[joint_name]
            child = obj.links[joint.body1.split("/")[-1]]
            lp, lq = child.get_position_orientation()
            lscale = child.get_world_scale() if hasattr(child, "get_world_scale") else th.ones(3)
            p_world = lp.float() + G.rotate(lq.float(), joint.local_position_1 * lscale)
            rp, rq = obj.get_position_orientation()
            return G.rotate(G.qinv(rq.float()), p_world - rp.float())
        except Exception as e:  # noqa: BLE001
            log.debug("hinge from prim failed for %s/%s: %r", obj.name, joint_name, e)
            return None

    def _axis_from_prim(self, obj, joint_name: str) -> List[float]:
        joint = obj.joints[joint_name]
        ax = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}.get(str(getattr(joint, "axis", "X")).upper(), [1.0, 0.0, 0.0])
        try:
            child = obj.links[joint.body1.split("/")[-1]]
            _, lq = child.get_position_orientation()
            rp, rq = obj.get_position_orientation()
            a_world = G.rotate(lq.float(), G.rotate(joint.local_orientation_1 if hasattr(joint, "local_orientation_1") else G.IDENTITY_QUAT, th.tensor(ax)))
            return G.rotate(G.qinv(rq.float()), a_world).tolist()
        except Exception:  # noqa: BLE001
            return ax

    # ------------------------------------------------------------------ geometry

    def obj_pose(self) -> G.Pose:
        p, q = self.target.get_position_orientation()
        return p.float().clone(), q.float().clone()

    def h(self, q_values: Sequence[float]) -> List[C.HandleWaypoint]:
        """h(q) in the world frame through the conventions (``handle_waypoints``). ``pinch_sign`` flips the closing axis
        (the same parallel-jaw pinch with the wrist turned by a half turn about the approach: one of the two is
        reachable for one arm where the other is not — the arms are not mirror images); the flipped pinch is recorded
        in the waypoints, so the oracle rebuilds the same eef frame."""
        handle = dict(self.handle)
        if self.pinch_sign < 0:
            handle["pinch_axis"] = [-float(v) for v in handle["pinch_axis"]]
        if abs(float(self.bar_offset)) > 1e-6:
            ax = bar_axis(handle)
            handle["contact"] = [float(c) + float(self.bar_offset) * float(a) for c, a in zip(handle["contact"], ax)]
        return C.handle_waypoints(handle, self.jinfo, list(q_values), obj_pose=self.obj_pose())

    def joint_index(self) -> int:
        return list(self.target.joints).index(self.joint_name)

    def set_joint(self, q: float):
        obj = self.target
        qs = obj.get_joint_positions().clone()
        qs[self.joint_index()] = float(q)
        obj.set_joint_positions(qs)
        obj.set_joint_velocities(th.zeros_like(qs))
        obj.keep_still()

    def joint_value(self) -> float:
        return float(self.target.get_joint_positions()[self.joint_index()])

    def pose_joint(self, q: float):
        """Put the joint at ``q`` for a geometric / collision query: set, one physics step so the link poses (AABBs, the
        cuRobo meshes) follow, set again (no residual velocity), then re-sync the collision world - the target's meshes
        alone (the robot is held still and nothing else moves in that one step; the full pass over the house's 4155
        meshes made every look-ahead candidate of the door search cost 30 s, 0921-041232)."""
        self.set_joint(q)
        self.world.step_physics(1, still=[self.robot])
        self.set_joint(q)
        self.kit.sync_world(only_objects=[self.target])

    def outward_world(self) -> th.Tensor:
        p, q = self.obj_pose()
        n = G.rotate(q, th.as_tensor(self.jinfo.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32))
        n[2] = 0.0
        return n / th.linalg.norm(n)

    def face_centre_world(self) -> th.Tensor:
        p, q = self.obj_pose()
        if self.jinfo.get("face_centre_world") is not None:
            return th.as_tensor(self.jinfo["face_centre_world"], dtype=th.float32)
        lo, hi = _vec(self.jinfo.get("link_bbox_min")), _vec(self.jinfo.get("link_bbox_max"))
        if lo is None or hi is None:
            lo_w, hi_w = self.target.aabb
            c = (lo_w + hi_w) / 2
            n = self.outward_world()
            ext = float(((hi_w - lo_w)[:2] * n[:2].abs()).sum()) / 2
            return c.float() + n * ext
        centre = (lo + hi) / 2
        n_obj = th.as_tensor(self.jinfo.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
        face_pt = centre + n_obj * float(((hi - lo) * n_obj.abs()).sum()) / 2
        return G.compose((p, q), (face_pt, G.IDENTITY_QUAT))[0]

    def target_extent(self) -> Tuple[th.Tensor, th.Tensor]:
        """AABB corners and centre of the target in its current (init) joint state (§5.2 Phase 1b: an open door counts)."""
        lo, hi = self.target.aabb
        return G.aabb_corners(lo.float(), hi.float()), ((lo + hi) / 2).float()

    def moving_link_footprint(self) -> th.Tensor:
        """World xy of the moving link's AABB corners at the current joint state (base-follow planar stand-in)."""
        links = [self.jinfo.get("child_link")] + [s.get("link") for s in (self.jinfo.get("sub_links") or [])]
        pts = []
        for n in links:
            if n and n in self.target.links:
                lo, hi = self.target.links[n].aabb
                pts.append(G.aabb_corners(lo.float(), hi.float())[:, :2])
        return th.cat(pts) if pts else th.zeros(0, 2)

    # ------------------------------------------------------------------ template (§5.3)

    def sample_template(self):
        try:
            self._sample_template()
        finally:  # the facts survive a template rejection (diagnostics in template_meta)
            self.template_meta["template_facts"] = _jsonable(self.template_facts)

    def _sample_template(self):
        obj = self.target
        self.load_handle(obj)
        self.hull = self.world.local_hull(obj)
        tol_probe = dict(self.gp.gen("stable_probe_tolerances") or {"revolute": 0.10, "prismatic": 0.02})
        self.probe_tol = float(tol_probe["revolute" if self.joint.is_revolute else "prismatic"])
        # generation.counterbalance (owner decision 2026-09-19): "auto" = horizontal-axis revolute joints (appliance doors that
        # swing down: dishwasher, oven) get the spring-balanced model - no gravity on the leaf assembly, viscous joint damping
        # ``counterbalance_damping`` (1/s) - so the leaf holds at any angle with the stock friction and stops where the hand
        # leaves it; vertical-axis doors / drawers keep the stock physics (no gravity moment). True / False force it either way.
        # Applied before the stable set and every physics probe below; recorded in the TRO for the loader.
        self.counterbalance: Optional[Dict[str, Any]] = None
        cb = self.gp.gen("counterbalance", "auto")
        axis_world0 = G.rotate(self.obj_pose()[1], th.as_tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))
        horizontal_revolute = bool(self.joint.is_revolute) and not axis_is_vertical(axis_world0)
        if cb is True or (cb == "auto" and horizontal_revolute):
            from ispatialgym.sampling import constraints as CS

            names = list(AH.leaf_links(obj, self.jinfo, self.joint))
            damping = float(self.gp.gen("counterbalance_damping", 5.0))
            applied = CS.apply_counterbalance(obj, self.joint, names, damping)
            self.counterbalance = {"links": applied["links"], "damping": damping}
            log.info("%s: counterbalance on %s/%s: gravity off on %s, joint damping %.2f (axis %s)", self.gp.variant, obj.name, self.joint_name, applied["links"], damping,
                     [round(float(v), 3) for v in axis_world0.tolist()])
        self.template_facts["counterbalance"] = self.counterbalance
        # oracle-fix 2026-11-04 round 3 (bottom_cabinet_rvunhj #03): opt-in generation.other_joints_init {joint: q} - other joints of the target held
        # there in the template world and in the instance's init / goal joint vectors (the neighbouring door's bar stands 70 mm from j_link_2's: the
        # closed gripper on the bar jams against the closed neighbour door - the door never moved in 53 demos)
        self.other_joints_init = {str(n): float(v) for n, v in (self.gp.gen("other_joints_init") or {}).items() if n in obj.joints and n != self.joint_name}
        if self.other_joints_init:
            names_ = list(obj.joints)
            qs_ = obj.get_joint_positions().clone()
            for n, v in self.other_joints_init.items():
                qs_[names_.index(n)] = v
            obj.set_joint_positions(qs_)
            obj.keep_still()
            self.template_facts["other_joints_init"] = dict(self.other_joints_init)
            from ispatialgym.sampling import constraints as CS_

            CS_.restrict_openable_joints(obj, [self.joint_name])  # (not (open target)) / (open target) read the moved joint only
            log.info("%s: other joints of %s held at %s", self.gp.variant, obj.name, self.other_joints_init)
        # generation.joint_friction (2026-09-18; as articulation_small): the PhysX articulation joint friction of the moved
        # joint, a number or a list of candidates tried in order until an open value holds. The stock 0.2 × (|F| + |T|)
        # brake holds the ynrhuk dishwasher door (7.5 kg, COM 0.30 m above the bottom hinge) below ≈ 41° only; 0.31+
        # holds it at any angle (gravity torque 22 N·m at 90°) at the price of a harder pull for the torque-limited arm
        jf = self.gp.gen("joint_friction", None)
        cands = [float(v) for v in (jf if isinstance(jf, (list, tuple)) else [jf])] if jf is not None else [None]
        self.stable = []
        for cand in cands:
            if cand is not None:
                self.joint.friction = cand
                self.joint_friction = cand
            self.stable = self.stable_set()
            log.info("%s: stable set of %s/%s at joint friction %s: %s", self.gp.variant, obj.name, self.joint_name, "stock" if cand is None else cand,
                     [round(s, 3) for s in self.stable])
            if self.stable:
                break
        self.template_facts["joint_friction"] = self.joint_friction
        if not self.stable:
            raise Reject("no_stable_open", f"{obj.name}/{self.joint_name}: no open fraction of {self.gp.gen('open_fraction_range', [0.5, 0.95])} holds passively")
        # passive continuation per (q_init, q_goal) pair (open: q_init = lower, q_goal ∈ S; close: q_init ∈ S, q_goal = lower)
        pairs = [(self.lower, s) for s in self.stable] if self.direction == "open" else [(s, self.lower) for s in self.stable]
        mode = self.gp.gen("require_goal_handle", "auto")
        for q0, q1 in pairs:
            k = self.passive_from(q0, q1) if mode == "auto" else None
            self.passive[f"{q0:.4f}->{q1:.4f}"] = k
        # face frames: view (P = face centre) and manipulation (P = the handle contact's projection onto the face)
        n = self.outward_world()
        Pc = self.face_centre_world()
        self.face_view = P0.make_face("face", n, Pc)
        h0 = self.h([self.lower if self.direction == "open" else self.stable[0]])[0]
        contact = h0.contact
        P_manip = contact - n * float(th.dot(contact - Pc, n))
        self.face_manip = P0.make_face("face", n, P_manip)
        off_left = float(th.dot((contact - Pc)[:2], self.face_view.tangent))
        self.arm = arm_rule(off_left, float(self.gp.gen("arm_side_threshold_m", 0.10)), self.gp.gen("arm", "auto"))
        axis_world = G.rotate(self.obj_pose()[1], th.as_tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))
        # A4 (owner decision 2026‑09‑16): class → follow mode / stance side / arm; ``generation.follow_mode_override``
        # (arm / base / hybrid) wins (D-011 C, 2026‑09‑17: the hybrid follow on the door units, ``articulation_hybrid``)
        self.door_class = door_class(obj.category, self.jinfo["type"])
        self.mode = follow_mode_by_class(self.door_class, self.jinfo["type"], axis_world)
        override = self.gp.gen("follow_mode_override")
        if override in ("arm", "base", "hybrid"):
            self.mode = str(override)
        lam_cfg = self.gp.gen("lateral_range_m") or {"right": [-0.10, 0.60], "left": [-0.60, 0.10]}
        self.lam_range = tuple(lam_cfg[self.arm]) if isinstance(lam_cfg, dict) else tuple(lam_cfg)
        self.hinge_side = None
        # the hinge-side stance rule serves the arm-only follow; the hybrid follow moves the base and keeps the full Λ
        if self.door_class == "cabinet" and self.mode == "arm" and axis_is_vertical(axis_world) and self.jinfo.get("hinge_point_obj") is not None:
            hinge_w = G.local_points_world(th.as_tensor([self.jinfo["hinge_point_obj"]], dtype=th.float32), self.obj_pose())[0]
            self.hinge_side = hinge_side(hinge_w, contact, self.face_view.tangent)
            lam_max = max(abs(float(v)) for v in self.lam_range) if self.lam_range else 0.60
            if self.gp.gen("arm", "auto") not in ("left", "right"):
                self.arm = arm_for_hinge_side(self.hinge_side)
            self.lam_range = hinge_side_lateral(self.hinge_side, lam_max)
        # generation.close_by_push (owner 2026-10-12): the whole closing pushed shut room-door style, the follow a sampling gate
        self.close_by_push_cfg = None
        if self.gp.gen("close_by_push") is not None:
            cfg = close_by_push_config(self.gp.gen("close_by_push"))
            if cfg is not None and close_by_push_applies(self.direction, self.jinfo["type"], self.door_class, axis_is_vertical(axis_world)) \
                    and self.jinfo.get("hinge_point_obj") is not None:
                self.close_by_push_cfg = cfg
            elif cfg is not None:
                log.info("%s: generation.close_by_push ignored for %s/%s (%s, class %s): it serves the closing of a revolute cabinet door on "
                         "a vertical hinge", self.gp.variant, obj.name, self.joint_name, self.direction, self.door_class)
        # generation.push_finish (owner 2026-10-12 ~04:20): a closing the arm follows only part of the way is pushed shut
        self.push_finish_cfg = None
        if self.gp.gen("push_finish") is not None and self.close_by_push_cfg is not None:
            log.info("%s: generation.push_finish ignored for %s/%s: generation.close_by_push pushes the whole closing", self.gp.variant, obj.name, self.joint_name)
        elif self.gp.gen("push_finish") is not None:
            if push_finish_applies(self.direction, self.jinfo["type"], self.door_class, self.mode, axis_is_vertical(axis_world)):
                self.push_finish_cfg = push_finish_config(self.gp.gen("push_finish"))
            else:
                log.info("%s: generation.push_finish ignored for %s/%s (%s, class %s, follow_mode %s): it serves the closing of a cabinet door on "
                         "a vertical hinge", self.gp.variant, obj.name, self.joint_name, self.direction, self.door_class, self.mode)
        # the Phase-2 lattice check (prompt0.phase2_step1) re-postures the IDLE arm of the checker's default working arm
        # (A6) - with the generic default "right" a left-handed door (door_crtrpa_0 0919-031559: handle 0.38 m left of the
        # face centre -> arm left) had its survivors validated with the LEFT arm tucked as idle, and every draw of the
        # manipulation stance (left arm working, untucked) collided in the re-check. The working arm is decided here.
        self.ikc.default_working_arms = [self.arm]
        self.template_facts.update({"joint": self.joint_name, "joint_type": self.jinfo["type"], "limits": [self.lower, self.upper], "direction": self.direction,
                                    "stable_set": self.stable, "stable_fractions": [(s - self.lower) / max(self.upper - self.lower, 1e-9) for s in self.stable],
                                    "passive_from": dict(self.passive), "face_normal": n.tolist(), "face_centre": Pc.tolist(), "P_manip": P_manip.tolist(),
                                    "handle_offset_left_m": off_left, "arm": self.arm, "follow_mode": self.mode, "axis_world": axis_world.tolist(),
                                    "door_class": self.door_class, "hinge_side": self.hinge_side, "lateral_range_m": list(self.lam_range)})
        if self.push_finish_cfg is not None:
            self.template_facts["push_finish"] = dict(self.push_finish_cfg)
        if self.close_by_push_cfg is not None:
            self.template_facts["close_by_push"] = dict(self.close_by_push_cfg)
        log.info("%s: %s/%s %s: class %s, S=%s, arm %s (handle %.2f m left of the face centre, hinge %s), follow_mode %s, Λ %s%s", self.gp.variant, obj.name,
                 self.joint_name, self.direction, self.door_class, [round(s, 3) for s in self.stable], self.arm, off_left, self.hinge_side, self.mode, self.lam_range,
                 ", push finish on" if self.push_finish_cfg is not None else (", close by push on" if self.close_by_push_cfg is not None else ""))

    def stable_set(self) -> List[float]:
        """§5.3 stable set S: 7 fractions of ``open_fraction_range`` + the upper limit, each held 120 steps and kept when
        it stays within ½ tolerance (the probe tolerance: revolute 0.10 rad, prismatic 0.02 m)."""
        lo, hi = self.gp.gen("open_fraction_range", [0.5, 0.95])
        n = int(self.gp.gen("stable_set_fractions", 7))
        fracs = [float(f) for f in th.linspace(float(lo), float(hi), n).tolist()]
        if bool(self.gp.gen("stable_set_include_full", True)):  # False: a fully open leaf no follow can reach (the dishwasher at 90 deg) is not drawn
            fracs.append(1.0)
        steps = int(self.gp.gen("stable_hold_steps", 120))
        drift_frac = float(self.gp.gen("stable_drift_tolerance_fraction", 0.5))
        state = self.world.snapshot()
        out: List[float] = []
        try:
            for f in fracs:
                q = self.lower + f * (self.upper - self.lower)
                self.set_joint(q)
                self.world.step_physics(steps, still=[self.robot])
                q_after = self.joint_value()
                ok = abs(q_after - q) <= drift_frac * self.probe_tol
                log.info("%s: %s at fraction %.2f (%.3f) -> %.3f after %d steps: %s", self.gp.variant, self.joint_name, f, q, q_after, steps, "stable" if ok else "drifts")
                if ok and all(abs(q - s) > 1e-6 for s in out):
                    out.append(q)
        finally:
            self.world.restore(state)
            self.target.keep_still()
            self.kit.sync_world()
        return out

    def passive_tolerance(self, dq: float) -> float:
        """The passive probe's band about q_goal: ``passive_tolerance_fraction`` (½) of the §8 evaluation tolerance of the
        instance's span — not the (looser) stable-probe tolerance, which exceeded the waypoint spacing of small spans."""
        frac = float(self.gp.gen("passive_tolerance_fraction", PASSIVE_TOLERANCE_FRACTION))
        return frac * joint_tolerance(self.jinfo["type"], dq, dict(self.gp.gen("tolerances") or {}))

    def passive_from(self, q_init: float, q_goal: float) -> Optional[float]:
        """§5.3 required-waypoint probe: the first k whose q_k, released and held ``stable_hold_steps`` steps, ends within
        the passive band of q_goal **having covered at least ``passive_min_progress`` of the remaining gap on its own**
        (:func:`passive_reaches`); the per-waypoint outcomes go to ``template_facts.passive_probe``."""
        steps = int(self.gp.gen("stable_hold_steps", 120))
        tol = self.passive_tolerance(q_goal - q_init)
        min_progress = float(self.gp.gen("passive_min_progress", PASSIVE_MIN_PROGRESS))
        state = self.world.snapshot()
        found = None
        probes: List[Dict[str, Any]] = []
        try:
            for k, qk in waypoint_values(q_init, q_goal, self.gp.gen("waypoint_fractions", WAYPOINT_FRACTIONS)):
                if k >= 1.0:
                    break
                self.set_joint(qk)
                self.world.step_physics(steps, still=[self.robot])
                q_after = self.joint_value()
                ok = passive_reaches(qk, q_after, q_goal, tol, min_progress)
                probes.append({"k": k, "q_k": round(qk, 4), "q_after": round(q_after, 4), "ok": ok})
                log.info("%s: %s released at k=%.2f (%.3f) -> %.3f after %d steps (goal %.3f, band %.3f): %s", self.gp.variant, self.joint_name, k, qk, q_after,
                         steps, q_goal, tol, "swings on to the goal" if ok else "does not reach the goal on its own")
                if ok:
                    found = k
                    break
        finally:
            self.world.restore(state)
            self.target.keep_still()
            self.kit.sync_world()
        self.template_facts.setdefault("passive_probe", {})[f"{q_init:.4f}->{q_goal:.4f}"] = {"tolerance": round(tol, 4), "min_progress": min_progress, "probes": probes}
        return found

    # ------------------------------------------------------------------ instance (§5.2 / §5.3)

    def draw_joint_targets(self, rng: np.random.Generator) -> Tuple[float, float]:
        if self.direction == "open":
            return self.lower, float(self.stable[int(rng.integers(len(self.stable)))])
        # oracle-fix 2026-11-04 round 3 (close_large#03): opt-in generation.close_goal_margin_rad - the close ends this far short of the closed stop
        # (inside the joint tolerance; the demos' torso bend peaked at the follow's last step, the door pressed into its stop)
        return float(self.stable[int(rng.integers(len(self.stable)))]), self.lower + float(self.gp.gen("close_goal_margin_rad", 0.0) or 0.0)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        obj = self.target
        arm_mode = self.mode == "arm"
        if arm_mode:  # A4: the goal is drawn per stance from the angles the arm reaches (open); close draws q_init from S
            q_init = self.lower if self.direction == "open" else float(self.stable[int(rng.integers(len(self.stable)))])
            q_goal = None
        else:
            q_init, q_goal = self.draw_joint_targets(rng)
        fractions = list(self.gp.gen("waypoint_fractions", WAYPOINT_FRACTIONS))
        # the target in its init state for every check of this attempt
        self.pose_joint(q_init)
        if self.direction == "close" and bool(self.gp.gen("manip_face_from_open_handle", True)):
            # close: the manipulation stances face the *open* handle (its approach axis at q_init, horizontal part), not
            # the closed frame plane - the reach-volume screen classifies the grasp in the base frame within 20 deg, so a
            # leaf open 33-49 deg left step2_reach = 0 for every stance (door 0920-000347)
            h_init = self.h([q_init])[0]
            a_h = h_init.approach.clone()
            a_h[2] = 0.0
            if float(a_h.norm()) > 0.3:
                self.face_manip = P0.make_face("face", -a_h / a_h.norm(), h_init.contact.clone())
                log.info("%s: close from %.3f: manipulation face re-aimed at the open handle (normal %s, P %s)", self.gp.variant, q_init,
                         [round(float(v), 3) for v in self.face_manip.n.tolist()], [round(float(v), 3) for v in h_init.contact.tolist()])
        gen_p0 = dict(self.gp.gen("prompt0") or {})
        beta_max = math.radians(float(gen_p0.get("beta_max_deg", 20.0)))
        d_min, d_max = float(gen_p0.get("d_min_m", 0.5)), float(gen_p0.get("d_max_m", 2.5))
        init_tilts = [float(v) for v in (gen_p0.get("init_tilts") or [0.0])]  # a lean forward for a low, near init view (the open dishwasher door)
        self._init_alts = []
        init = self._draw_init_from_set(q_init, beta_max, d_min, d_max, init_tilts, rng) if init_draw_on() else None
        if init is None:
            init = P0.phase1_init_pose(self, self.face_view, beta_max, d_min, d_max, self.target_extent, rng, tilts=init_tilts)
        init.name = "init"
        self._init_used = init
        self.sidecar.add_stance(init)
        # Phase 2: contact set = the h(q_init) grasp of the handle arm — one Phase 2 per grasp offset along a bar handle
        # (bar_offsets: 0, ±0.15, ±0.25 m within the bar), the survivors pooled with their offset
        lam_range = tuple(self.lam_range) if self.lam_range else (-0.10, 0.60)
        d_range = (0.35, float(gen_p0.get("d_scan_max_m", 1.10)))
        half = float(self.gp.gen("face_patch_half_width_m", 0.10))
        offsets = [float(v) for v in (self.gp.gen("bar_grasp_offsets_m") or bar_offsets(self.handle))]
        pool: List[Stance] = []
        h0_by: Dict[float, C.HandleWaypoint] = {}
        stats_by: Dict[float, Dict[str, Any]] = {}
        classes: Tuple[str, ...] = ()
        # Phase 2 step 1 (base on free floor, body collision-free over the lattice) does not depend on the grasp offset:
        # ISPATIALGYM_SAMPLER_STEP1_ONCE=1 runs it once per attempt for all offsets (default: once per offset)
        step1_memo: Optional[Dict[str, Any]] = {} if os.environ.get("ISPATIALGYM_SAMPLER_STEP1_ONCE", "").lower() in ("1", "true", "yes") else None
        for off in offsets:
            self.bar_offset = off
            h0 = self.h([q_init])[0]
            h0_by[off] = h0
            classes = pinch_classes(h0.pinch, self.handle.get("kind", "bar"), float(self.gp.gen("pinch_axis_vertical_threshold", 0.7)))
            contact = P0.ContactSpec("handle", self.arm, h0.eef, h0.pre_grasp, classes=classes, ignore_for_last=[obj])
            patch = th.stack([h0.contact - h0.pinch * half, h0.contact, h0.contact + h0.pinch * half])

            def extra_vis(cam, _patch=patch) -> bool:  # §5.3: contact(q_init) and the moving face patch (± 0.10 m along the pinch axis) in frame
                return P0.in_frame(self, cam, _patch)

            scope = {"scene": self.world.scene_model, "target": obj.name, "joint": self.joint_name, "q_init": round(q_init, 2), "arm": self.arm,
                     "sampler": "articulation_large", "class": self.door_class, "lam": [round(v, 2) for v in lam_range], "bar_offset": round(off, 3)}
            stats: Dict[str, Any] = {}
            survivors = P0.phase2_manipulation(self, self.face_manip, [contact], lam_range, [self.arm], d_range, extra_vis, focus=h0.contact,
                                               cache_scope=scope, cache_contacts=True, stats=stats, step1_memo=step1_memo)
            stats_by[off] = stats
            for st in survivors:
                st.extras["bar_offset_m"] = off
            pool += list(survivors)
        self.bar_offset = 0.0
        if len(offsets) > 1:
            log.info("%s: Phase 2 survivors per bar offset: %s", self.gp.variant, {k: sum(1 for st in pool if st.extras.get("bar_offset_m") == k) for k in offsets})
        if not pool:
            raise Reject("no_manipulation_stance", f"Phase 2 left no survivor for {obj.name}/{self.joint_name} at q_init {q_init:.3f} ({stats_by})")
        errors: List[str] = []
        draw_stats: Dict[str, int] = {}
        # the sequential follow rejects a stance in seconds when it admits no releasable arm pull (the elbow folds past its limit
        # on the retreat from a near stand-off, door 0921-030920) - every survivor gets its draw there instead of 8 of them
        sequential = str((self.gp.gen("hybrid_follow") or {}).get("policy", "hybrid")) == "sequential"
        for _ in range(len(pool) if sequential else min(len(pool), 8)):
            # i7-memory2 (owner 2.3): memory_arm = the outcome memory's pick hook on the draw (prompt0.draw_manipulation; the hybrid follow
            # plans from the stance drawn here - articulation_hybrid has no stance choice of its own)
            i, manip = P0.draw_manipulation(self, pool, rng, [self.arm], focus=None, extra_visibility=None, clearance_ignore=[self.target], stats=draw_stats,
                                            memory_arm=self.arm)
            pool.pop(i)
            if manip is None:
                errors.append(f"drawn tilt fails the re-check ({dict(draw_stats)})")
                continue
            self.bar_offset = float(manip.extras.get("bar_offset_m", 0.0))
            h0 = h0_by[self.bar_offset]
            manip.focus = h0.contact.clone()
            patch = th.stack([h0.contact - h0.pinch * half, h0.contact, h0.contact + h0.pinch * half])
            cam = P0.camera_poses(self, [manip])[0]
            if not P0.in_frame(self, cam, patch) or not P0.in_frame(self, cam, h0.contact.reshape(1, 3), P0.IN_FRAME_MARGIN):
                errors.append("drawn tilt: the handle contact / face patch leaves the frame")
                continue
            # both closing signs of the pinch (the wrist turned by a half turn about the approach): the reach-volume class is
            # the same, the exact IK is not — one of the two is usually reachable where the other is not. The orientation rule
            # (owner 2026-09-27) orders them by the [pre-grasp, grasp] solutions at q_init (step 5a, joint-limit safe, then
            # easier to reach from the stance, +1 within the tolerance) and the follow is solved in that order, the other sign
            # when the first falls; no wrist-flipped branch (the follow solves the grasp again from the stance's seed)
            props = []
            for rank, sign in enumerate((1.0, -1.0)):
                self.pinch_sign = sign
                h0s = self.h([q_init])[0]
                props.append(O.Proposal(f"pinch {sign:+.0f}", rank, (sign, h0s, P0.ContactSpec("handle", self.arm, h0s.eef, h0s.pre_grasp, classes=classes, ignore_for_last=[obj])),
                                        [h0s.pre_grasp, h0s.eef]))

            def grasp_ok(p):
                sign, _, contact_s = p.payload
                sol = P0.exact_contacts(self, manip, [contact_s], ignore_combined=[obj])
                if not sol.ok:
                    errors.append(f"pinch {sign:+.0f}: handle_unreachable: {sol.reason}")
                    if arm_mode:
                        self.pinch_sign = sign
                        self.debug_handle_ik(manip, contact_s)
                return bool(sol.ok), sol, None, "" if sol.ok else str(sol.reason)

            solved: Dict[str, Any] = {}

            def follow(sc):
                sign, h0s, contact_s = sc.proposal.payload
                self.pinch_sign = sign
                try:
                    goal_set = None
                    reach_info: Dict[str, Any] = {}
                    q_goal_s = q_goal
                    if arm_mode:
                        q_goal_s, goal_set, reach_info = self.goal_for_stance(manip, init, q_init, contact_s, rng)
                    # a push finish records its release as a waypoint and requires the handle only up to there
                    push = reach_info.get("push_finish")
                    fr_s = finish_fractions(fractions, push["k_release"]) if push else fractions
                    wps = self.h([q_init] + [qk for _, qk in waypoint_values(q_init, q_goal_s, fr_s)])
                    pf = self.passive_for(q_init, q_goal_s)
                    required = required_fractions(pf, fr_s, self.gp.gen("require_goal_handle", "auto"))
                    if push:
                        required = finish_required(required, push["k_release"])
                    d = self._solve_from(manip, init, q_init, q_goal_s, wps, required, contact_s, h0s, goal_set=goal_set, reach_info=reach_info, fractions=fr_s)
                except Reject as e:
                    errors.append(f"pinch {sign:+.0f}: {e.reason}: {e}")
                    # a stance that got as far as a full hybrid follow (5 min) and then fell at step 6 was invisible until the
                    # attempt's summary (door 0919-122043: 5 complete follows, no draft, no line in between)
                    log.info("%s: stance %s pinch %+.0f rejected: %s: %s", self.gp.variant, self._stance_key(manip), sign, e.reason, str(e)[:300])
                    self.sidecar.stances.pop("manipulation", None)
                    self.sidecar.transitions.clear()
                    self.sidecar.paths.clear()
                    self.sidecar.articulation = None
                    return False, f"{e.reason}: {str(e)[:120]}"
                d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_init": q_init, "q_goal": q_goal_s, "required_fractions": required,
                                  "passive_from": pf, "arm": self.arm, "follow_mode": self.mode, "door_class": self.door_class, "hinge_side": self.hinge_side,
                                  "pinch_sign": sign, "phase2": stats, "init_stance": P0.summarize_stance(self._init_used or init), "manipulation_stance": P0.summarize_stance(manip),
                                  **reach_info})
                solved["draft"] = d
                return True, ""

            chosen, decision = O.choose(self.ikc, self.arm, props, grasp_ok, self.ikc.q_locked(manip), stage="handle pinch", downstream=follow, map_twins=False)
            if chosen is not None:
                d = solved["draft"]
                d.summary["pinch_orient"] = decision
                return d
        self.pinch_sign = 1.0
        # every distinct reason once (the last 4 alone hid the hybrid search's verdict behind the tilt re-check draws, door 0919-023637)
        distinct: List[str] = []

        def reason_of(e: str) -> str:  # "pinch +1: <reason>: <message>" -> <reason>; else the text up to the first colon
            parts = e.split(": ")
            return parts[1] if len(parts) > 2 and parts[0].startswith("pinch") else parts[0]

        for e in errors:
            if not any(reason_of(x) == reason_of(e) for x in distinct):
                distinct.append(e)
        raise Reject("handle_unreachable", f"{len(survivors)} Phase-2 survivors, none passes step 5 / base-follow / step 6 ({len(errors)} errors): " + " | ".join(distinct[:6]))

    # ------------------------------------------------------------------ i7-artlarge: the init (start) stance from the Phase-1 feasible set

    def _eval_starts(self) -> List[Tuple[float, float, float]]:
        """The start bases (x, y, yaw) of the eval instances the train run's repeat filter compares (none outside a train run)."""
        out: List[Tuple[float, float, float]] = []
        for rec in getattr(self.instance_filter, "pool", None) or []:
            for p in getattr(rec, "start", None) or []:
                if getattr(p, "name", None) == "start_base" and p.yaw is not None and len(p.p) >= 2:
                    out.append((float(p.p[0]), float(p.p[1]), float(p.yaw)))
        return out

    def _init_set(self, q_init: float, beta_max: float, d_min: float, d_max: float, tilts: Sequence[float]) -> List[Stance]:
        """Every Phase-1 feasible init stance on the lattice bearings (INIT_SET_BETA_STEP_DEG in +-``beta_max``) x the d scan, each judged by
        :func:`prompt0.phase1_scan`'s checks in its order (free floor in the task rooms, the whole target in frame at a translate / tilt,
        body collision-free, trunk sweep free; t* = :func:`prompt0.phase1_choice`) with the target at ``q_init`` (the caller posed it).
        Memoized per (q_init, ranges, tilts): the scene is static between attempts. One log line with the per-stage counts."""
        key = (round(float(q_init), 4), round(beta_max, 4), round(d_min, 3), round(d_max, 3), tuple(round(float(t), 3) for t in tilts))
        if self._init_sets is None:
            self._init_sets = {}
        if key in self._init_sets:
            return self._init_sets[key]
        face = self.face_view
        corners, centre = self.target_extent()
        corners = P0._t(corners).reshape(-1, 3)
        n_b = int(math.floor(2.0 * math.degrees(beta_max) / INIT_SET_BETA_STEP_DEG + 1e-9))
        betas = [math.radians(-math.degrees(beta_max) + k * INIT_SET_BETA_STEP_DEG) for k in range(n_b + 1)] if beta_max > 0 else [0.0]
        stats = {"positions": 0, "base_ok": 0, "in_frame": 0, "body_free": 0, "sweep_free": 0}
        cells: List[Tuple[float, float, List[Stance]]] = []
        for beta in betas:
            for d in P0.d_scan(d_min, d_max):
                stats["positions"] += 1
                x, y, _ = P0.approach_ray_base(face, beta, d)
                if not P0.base_ok(self, (x, y)):
                    continue
                stats["base_ok"] += 1
                cells.append((beta, d, [P0.ray_stance(face, beta, d, t, float(tilt), "init") for t in P0.TRANSLATES for tilt in tilts]))
        flat = [st for _, _, sts in cells for st in sts]
        cams = P0.camera_poses(self, flat)
        seen_flat = [(st, cam) for st, cam in zip(flat, cams) if P0.in_frame(self, cam, corners, P0.IN_FRAME_MARGIN)]
        seen_ids = {id(st) for st, _ in seen_flat}
        free = P0.body_free(self, [st for st, _ in seen_flat]) if seen_flat else th.zeros(0, dtype=th.bool)
        free_ids = {id(st) for (st, _), f in zip(seen_flat, free.tolist()) if f}
        cam_of = {id(st): cam for st, cam in seen_flat}
        kept: List[Tuple[float, float, List[Stance]]] = []
        for beta, d, sts in cells:
            seen = [st for st in sts if id(st) in seen_ids]
            if not seen:
                continue
            stats["in_frame"] += 1
            ok = [st for st in seen if id(st) in free_ids]
            if not ok:
                continue
            stats["body_free"] += 1
            kept.append((beta, d, ok))
        out: List[Stance] = []
        if kept:
            sweeps = [Stance("sweep", tuple(sts[0].base_pose2d), float(t), 0.0) for _, _, sts in kept for t in P0.SWEEP_TRANSLATES]
            sweep_free = P0.body_free(self, sweeps, arms_excluded=True).reshape(len(kept), len(P0.SWEEP_TRANSLATES)).all(dim=1).tolist()
            for (beta, d, ok), sf in zip(kept, sweep_free):
                if not sf:
                    continue
                stats["sweep_free"] += 1
                t_star = P0.phase1_choice([(st.translate, P0.axis_angle(self, cam_of[id(st)], centre)) for st in ok])
                st = next(st for st in ok if abs(st.translate - t_star) < 1e-9)
                st.extras["axis_angle_deg"] = math.degrees(P0.axis_angle(self, cam_of[id(st)], centre))
                st.extras["target_centre"] = [round(float(v), 4) for v in P0._t(centre).tolist()]
                st.arms = "untucked"
                out.append(st)
        self._init_sets[key] = out
        xy = [st.base_pose2d for st in out]
        log.info("%s: ARTLARGE_INIT_DRAW: init set at q_init %.3f: %d feasible of %s (bearings %d x d %.2f-%.2f)%s", self.gp.variant, q_init, len(out), stats,
                 len(betas), d_min, d_max,
                 "" if not xy else f"; x {min(p[0] for p in xy):.2f}..{max(p[0] for p in xy):.2f}, y {min(p[1] for p in xy):.2f}..{max(p[1] for p in xy):.2f}, "
                 f"cells (bearing deg, d) {[(round(st.extras['bearing_deg']), round(st.stand_off, 2)) for st in out][:160]}")
        return out

    def _draw_init_from_set(self, q_init: float, beta_max: float, d_min: float, d_max: float, tilts: Sequence[float],
                            rng: np.random.Generator) -> Optional[Stance]:
        """ARTLARGE_INIT_DRAW: the attempt's init = the first of the feasible set in :func:`order_init_candidates` order (away from the
        eval starts first); the next INIT_ALT_MAX kept for step 6 (:meth:`_connect_init`). None when the set is empty (today's draw then)."""
        from ispatialgym.sampling.instance_separation import REPEAT_ANG_DEG, REPEAT_POS_M
        from ispatialgym.sampling.v2.core import _kit_count

        cands = self._init_set(q_init, beta_max, d_min, d_max, tilts)
        if not cands:
            _kit_count(self, "artlarge_init.empty_set")
            return None
        order, n_near = order_init_candidates([st.base_pose2d for st in cands], self._eval_starts(), rng, REPEAT_POS_M, REPEAT_ANG_DEG)
        picked = [copy.deepcopy(cands[i]) for i in order[:INIT_ALT_MAX + 1]]
        for st in picked:
            st.name = "init"
        near_first = n_near == len(cands)
        _kit_count(self, "artlarge_init.drawn")
        _kit_count(self, "artlarge_init.drawn_near_eval" if near_first else "artlarge_init.drawn_away")
        log.info("%s: ARTLARGE_INIT_DRAW: init drawn %s (bearing %.0f deg, d %.2f, trunk %s) from %d feasible (%d near an eval start); %d alternatives",
                 self.gp.variant, [round(float(v), 3) for v in picked[0].base_pose2d], picked[0].extras.get("bearing_deg", 0.0), picked[0].stand_off,
                 [picked[0].translate, picked[0].tilt], len(cands), n_near, len(picked) - 1)
        self._init_alts = picked[1:]
        return picked[0]

    def _connect_init(self, init: Stance, manip: Stance, path_max_m: float) -> Stance:
        """Step 6 (:func:`prompt0.connect`) from ``init``; under ARTLARGE_INIT_DRAW, when it rejects with a path / transition reason, from the
        alternatives of the attempt's draw in order (the sidecar's init stance replaced by the one that connects). Returns the init used."""
        from ispatialgym.sampling.v2.core import _kit_count

        alts = list(self._init_alts) if init_draw_on() else []
        try:
            P0.connect(self, init, manip, path_max_m)
            self._init_used = init
            if alts:
                _kit_count(self, "artlarge_init.connect_first")
            return init
        except Reject as e:
            if not alts or e.reason not in ("path_collision", "no_path", "path_too_long", "no_transition"):
                raise
            first = e
        for k, alt in enumerate(alts):
            _kit_count(self, "artlarge_init.alt_tried")
            try:
                P0.connect(self, alt, manip, path_max_m)
            except Reject as e:
                if e.reason not in ("path_collision", "no_path", "path_too_long", "no_transition"):
                    raise
                continue
            _kit_count(self, "artlarge_init.alt_connected")
            self.sidecar.add_stance(alt)
            self._init_used = alt
            log.info("%s: ARTLARGE_INIT_DRAW: step 6 from the drawn init failed (%s: %s); alternative %d %s connects", self.gp.variant, first.reason,
                     str(first)[:160], k + 1, [round(float(v), 3) for v in alt.base_pose2d])
            return alt
        _kit_count(self, "artlarge_init.alt_exhausted")
        raise first

    def passive_for(self, q_init: float, q_goal: float) -> Optional[float]:
        """The passive-continuation waypoint of a (q_init, q_goal) pair: the template's table, probed on demand for a
        goal outside S (the A4 reach limit) and remembered."""
        key = f"{q_init:.4f}->{q_goal:.4f}"
        if key not in self.passive:
            mode = self.gp.gen("require_goal_handle", "auto")
            self.passive[key] = self.passive_from(q_init, q_goal) if mode == "auto" else None
            self.template_facts["passive_from"] = dict(self.passive)
        return self.passive[key]

    # ------------------------------------------------------------------ A4: arm-only reach along h(q)

    def _stance_key(self, manip: Stance) -> str:
        return f"d{manip.stand_off:.2f}_l{manip.lateral:+.2f}_t{manip.translate:.2f}_{manip.tilt:+.2f}_p{self.pinch_sign:+.0f}"

    def arm_reach_limit(self, manip: Stance, q_from: float, q_seed: Optional[th.Tensor], q_to: Optional[float] = None,
                        refine: int = 0) -> Tuple[float, bool]:
        """The farthest joint value on the grid ``q_from → q_to`` (default the upper limit; 10° / 5 cm steps, ``q_to``
        included, either direction) whose handle grasp pose the arm reaches from ``manip`` with the world re-synced at
        that value (target ignored for the grasp pose); the sweep stops at the first miss, so every value up to the
        limit is reachable. ``refine`` > 0 then bisects between the last hit and the first miss that many times (the
        boundary to within step / 2^refine). Returns ``(reach, ended_at_miss)`` — ``ended_at_miss`` False when the arm
        reached ``q_to`` itself."""
        obj = self.target
        step_cfg = dict(self.gp.gen("reach_grid_step") or REACH_GRID_STEP)
        step = math.radians(float(step_cfg.get("revolute_deg", 10.0))) if self.joint.is_revolute else float(step_cfg.get("prismatic_m", 0.05))
        q_to = self.upper if q_to is None else float(q_to)
        grid = reach_grid(q_from, q_to, step) if q_to >= q_from else [2 * q_from - v for v in reach_grid(q_from, 2 * q_from - q_to, step)]
        q_now = self.joint_value()
        reach, seed = float(q_from), q_seed
        miss: Optional[float] = None
        # the program releases the handle where the follow ends and retreats 0.10 m along -a of the *open* handle frame
        # (``Retreat`` after ``base_follow``); that pose must be reachable with the joint-limit margin too, else the program
        # check rejects the stance after the whole follow was simulated (dishwasher 0919-195644: 7 of 8 attempts, every
        # retreat solution within 2-46 mrad of right_arm_joint7's limit). The hybrid tracker checks the same (check_retreat).
        check_retreat = bool(self.gp.gen("reach_check_retreat", True))
        self.reach_retreat_miss: Optional[float] = None  # the first joint value whose grasp reaches but whose retreat does not

        def reaches(q: float) -> bool:
            nonlocal seed
            self.pose_joint(q)
            w = self.h([q])[0]
            res = self.ikc.exact([w.eef], manip, self.arm, ignore_for_last=[obj], q_seed=seed, cache=False)
            if not (res.ok and res.q is not None):
                return False
            seed = res.q
            if check_retreat:
                back = self.ikc.exact([C.pre_pose(w.eef, C.PRE_DIST)], manip, self.arm, q_seed=res.q, cache=False)  # RETREAT_DIST == PRE_DIST (0.10 m)
                if not back.ok:
                    if self.reach_retreat_miss is None:
                        self.reach_retreat_miss = float(q)
                    log.info("%s: the retreat pose from the handle at %.3f is not reachable from %s (%s): reach boundary", self.gp.variant, q, self._stance_key(manip), back.reason)
                    return False
            return True

        try:
            for q in grid:
                if not reaches(q):
                    miss = q
                    break
                reach = q
            if miss is not None:
                for _ in range(max(0, int(refine))):
                    mid = 0.5 * (reach + miss)
                    if reaches(mid):
                        reach = mid
                    else:
                        miss = mid
        finally:
            self.pose_joint(q_now)
        return reach, miss is not None

    def reach_margin(self) -> float:
        """``generation.reach_margin`` (rad / m) for this joint type; see :data:`REACH_MARGIN`."""
        cfg = dict(self.gp.gen("reach_margin") or REACH_MARGIN)
        return math.radians(float(cfg.get("revolute_deg", REACH_MARGIN["revolute_deg"]))) if self.joint.is_revolute else float(cfg.get("prismatic_m", REACH_MARGIN["prismatic_m"]))

    def debug_handle_ik(self, manip: Stance, contact: P0.ContactSpec) -> None:
        """Diagnostic log when the handle's [pre-grasp, grasp] fails the exact IK from a Phase-2 survivor: the verdict of
        the grasp pose alone (target ignored), of the pre-grasp alone, of the grasp with every collision object ignored
        (pure kinematics) and of the other arm; the pad point in base coordinates and the reach-volume verdict."""
        try:
            obj = self.target
            others = [o for o in self.world.scene.objects if o.name in self.world.scene_object_names] if hasattr(self.world, "scene_object_names") else []
            pre = contact.pre_pose()
            r_g = self.ikc.exact([contact.pose], manip, self.arm, ignore_for_last=[obj], cache=False)
            r_p = self.ikc.exact([pre], manip, self.arm, cache=False)
            with self.ikc.ignoring(others):
                r_free = self.ikc.exact([contact.pose], manip, self.arm, cache=False)
            other = "left" if self.arm == "right" else "right"
            r_o = self.ikc.exact([pre, contact.pose], manip, other, ignore_for_last=[obj], cache=False)
            p_base = P0.to_base_frame(C.pad_center(contact.pose), manip.base_pose2d, P0.base_z(self))
            rv_ok = contact.volume_ok(P0.reach_volume(), manip, P0.base_z(self))
            culprits = self._penetrating_objects(r_free.q, [obj]) if (r_free.ok and r_free.q is not None and not r_g.ok) else None
            log.info("%s: handle IK debug at %s (arm %s, pinch %+.0f): grasp alone %s, pre alone %s, grasp without the world %s, other arm [pre, grasp] %s; "
                     "pad in base %s, classes %s, volume %s; the world-free solution penetrates %s", self.gp.variant, self._stance_key(manip), self.arm, self.pinch_sign,
                     r_g.ok, r_p.ok, r_free.ok, r_o.ok, [round(float(v), 3) for v in p_base.tolist()], list(contact.classes or ()), rv_ok, culprits)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            log.info("%s: handle IK debug failed: %s", self.gp.variant, e)

    def _penetrating_objects(self, q: th.Tensor, ignore: Sequence[Any]) -> Dict[str, Any]:
        """Diagnostic: the robot spheres of configuration ``q`` that penetrate the collision world (``ignore`` removed)
        and the scene objects whose AABB (inflated 5 cm) contains them — which object blocks a kinematically feasible
        grasp."""
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState

        be = self.kit.backend
        entry = be.kin_entry("whole_body")
        dev = be.device_cfg.device
        with self.ikc.ignoring(list(ignore)):
            self.kit._ensure_synced()
            js = be._active_js(entry, q.unsqueeze(0))
            state = entry.fk.compute_kinematics(js)
            spheres = state.robot_spheres.reshape(1, 1, -1, 4).contiguous()
            buf = CollisionBuffer.from_shape(spheres.shape, be.device_cfg)
            w = th.ones(1, device=dev)
            d0 = be.checker.get_sphere_collision(KinematicsState(robot_spheres=spheres), buf, w, th.full((1,), float(P0.COLLISION_MARGIN), device=dev)).reshape(-1)
        sp = spheres.reshape(-1, 4).cpu()
        rp = be.root_pose()[0].cpu()
        bad = (d0 > 0).nonzero().flatten().tolist()
        pts = [sp[i, :3] + rp[:3] for i in bad]
        hits: Dict[str, int] = {}
        for p in pts:
            for o in self.world.scene.objects:
                if o in ignore or o is self.robot:
                    continue
                lo, hi = o.aabb
                if bool((p >= lo.float() - 0.05).all() and (p <= hi.float() + 0.05).all()):
                    hits[o.name] = hits.get(o.name, 0) + 1
        return {"n_penetrating": len(bad), "points": [[round(float(v), 2) for v in p.tolist()] for p in pts[:4]], "objects": dict(sorted(hits.items(), key=lambda kv: -kv[1])[:5])}

    def holds_passively(self, q: float) -> bool:
        """A single stable-set probe (§5.3): ``q`` held ``stable_hold_steps`` steps stays within ½ probe tolerance."""
        if any(abs(q - s) <= 1e-6 for s in self.stable):
            return True
        steps = int(self.gp.gen("stable_hold_steps", 120))
        drift_frac = float(self.gp.gen("stable_drift_tolerance_fraction", 0.5))
        state = self.world.snapshot()
        try:
            self.set_joint(q)
            self.world.step_physics(steps, still=[self.robot])
            ok = abs(self.joint_value() - q) <= drift_frac * self.probe_tol
        finally:
            self.world.restore(state)
            self.target.keep_still()
            self.kit.sync_world()
        return ok

    def goal_for_stance(self, manip: Stance, init: Stance, q_init: float, contact: P0.ContactSpec, rng: np.random.Generator) -> Tuple[float, List[float], Dict[str, Any]]:
        """A4 arm mode: the goal set of a drawn stance = S ∩ [lower, q_reach] (q_reach = ``arm_reach_limit`` from the
        closed joint), else {q_reach} when it opens ≥ ``min_reachable_opening`` and holds passively; ``reachable_fraction``
        is recorded per stance in the template meta. Open: q_goal drawn from that set; close: q_init must lie in it
        (the whole closing sweep is reachable) and the goal is the lower limit."""
        obj = self.target
        sol = P0.exact_contacts(self, manip, [contact], ignore_combined=[obj])
        if not sol.ok:
            self.debug_handle_ik(manip, contact)
            raise Reject("handle_unreachable", sol.reason)
        span = max(self.upper - self.lower, 1e-9)
        if self.direction == "close":
            # close: the arm must follow the handle from q_init all the way down to the lower limit — sweep downward from
            # q_init; the reachable fraction is the closed part of the swing the arm covers
            q_low, _ = self.arm_reach_limit(manip, q_init, sol.q, q_to=self.lower)
            frac = (q_init - q_low) / max(q_init - self.lower, 1e-9)
            key = self._stance_key(manip)
            rf = self.template_meta.setdefault("reachable_fraction", {})
            rf[key] = round(frac, 4)
            self.template_meta["reachable_fraction_max"] = max(float(v) for v in rf.values())
            info = {"reachable_fraction": frac, "q_reach": q_low, "goal_set": [float(self.lower)]}
            log.info("%s: stance %s follows %s from %.3f down to %.3f (fraction %.2f of the closing)", self.gp.variant, key, self.joint_name, q_init, q_low, frac)
            if q_low > self.lower + 1e-6:
                short = f"closing from {q_init:.3f}: the arm follows the handle only down to {q_low:.3f} (fraction {frac:.2f}) from stance {key}"
                if self.push_finish_cfg is None:
                    raise Reject("handle_unreachable", f"{short}; the rest needs a push on the leaf (not implemented)")
                # owner 2026-10-12 ~04:20: hold the handle as far as the arm reaches, release, push the leaf the rest of the way shut
                k_rel = release_fraction(q_init, q_low, self.lower)
                if k_rel <= 0.0:
                    raise Reject("handle_unreachable", f"{short}: no handle phase to finish with a push")
                q_rel = q_init + k_rel * (self.lower - q_init)
                try:
                    info["push_finish"] = self.plan_push_finish(manip, q_init, q_rel, k_rel)
                except Reject as e:
                    raise Reject(e.reason, f"{short}; the push finish fails: {str(e).split('] ', 1)[-1]}") from None
            return float(self.lower), [float(self.lower)], info
        # the boundary is refined by bisection and the usable reach stops ``reach_margin`` inside it (REACH_MARGIN: the
        # goal at the boundary itself left the drives saturated 7 mm short of the last handle pose)
        q_bound, at_miss = self.arm_reach_limit(manip, self.lower, sol.q, refine=int(self.gp.gen("reach_refine_steps", REACH_REFINE_STEPS)))
        margin = self.reach_margin()
        q_reach = reach_with_margin(q_bound, self.lower, margin, at_miss)
        frac = (q_reach - self.lower) / span
        key = self._stance_key(manip)
        rf = self.template_meta.setdefault("reachable_fraction", {})
        rf[key] = round(frac, 4)
        self.template_meta["reachable_fraction_max"] = max(float(v) for v in rf.values())
        self.template_meta.setdefault("reach_boundary", {})[key] = {"q_boundary": round(q_bound, 4), "ended_at_miss": at_miss, "margin": round(margin, 4), "q_reach": round(q_reach, 4)}
        min_cfg = dict(self.gp.gen("min_reachable_opening") or MIN_REACHABLE_OPENING)
        min_open = math.radians(float(min_cfg.get("revolute_deg", 50.0))) if self.joint.is_revolute else float(min_cfg.get("prismatic_m", 0.10))
        goal_set = restrict_goal_set(self.stable, q_reach, self.lower, min_open)
        if goal_set and goal_set[0] not in self.stable and not self.holds_passively(goal_set[0]):
            goal_set = []
        info = {"reachable_fraction": frac, "q_reach": q_reach, "q_reach_boundary": q_bound, "reach_margin": margin if at_miss else 0.0, "goal_set": list(goal_set)}
        log.info("%s: stance %s reaches %s up to %.3f (boundary %.3f%s, margin %.3f; fraction %.2f); goal set %s", self.gp.variant, key, self.joint_name, q_reach, q_bound,
                 "" if at_miss else " = the sweep's end", margin if at_miss else 0.0, frac, [round(s, 3) for s in goal_set])
        if not goal_set:
            raise Reject("handle_unreachable", f"the arm reaches {self.joint_name} only up to {q_reach:.3f} (fraction {frac:.2f}) from stance {key}: no goal opens ≥ {min_open:.2f}")
        if self.direction == "open":
            return float(goal_set[int(rng.integers(len(goal_set)))]), goal_set, info
        if q_init > q_reach + 1e-6:
            raise Reject("handle_unreachable", f"closing from {q_init:.3f} needs the arm to reach up to it; the limit from stance {key} is {q_reach:.3f}")
        return float(self.lower), goal_set, info

    def plan_push_finish(self, manip: Stance, q_init: float, q_rel: float, k_rel: float) -> Dict[str, Any]:
        """The push finish of a closing the hand follows from ``q_init`` only to ``q_rel`` (fraction ``k_rel``) from ``manip`` (owner
        2026-10-12 ~04:20): a fist contact of the handle arm on the leaf's outer face, from the same stance. The candidates
        (:func:`push_finish_candidates`) are screened in one batch at the release and at the band edge (the lower limit +
        ``band_fraction`` × the §8 tolerance) - the screen only orders them - and checked in that order, at most ``max_checks``: the
        exact IK of [pre-contact, contact] with the leaf at ``q_rel`` (the pre-contact in the full world, the approach and the contact
        with the target out of it, closed pads), then the arc about the hinge to the lower limit - else to the band edge - tracked from
        the contact configuration as the oracle's ``arm_along`` tracks it, the target out of the world (the fist presses the leaf).
        Returns the ``articulation.push_finish`` record the oracle's ``push_close`` executes; ``Reject("push_unreachable")`` when no
        candidate passes."""
        cfg = self.push_finish_cfg or dict(PUSH_FINISH_DEFAULTS)
        obj = self.target
        if self.jinfo.get("hinge_point_obj") is None:
            raise Reject("push_unreachable", f"{obj.name}/{self.joint_name}: no hinge point")
        obj_pose = self.obj_pose()
        try:
            leaf = closed_leaf(obj_pose, self.jinfo, self.handle)
        except ValueError as e:
            raise Reject("push_unreachable", f"{obj.name}/{self.joint_name}: {e}") from None
        tol = joint_tolerance(self.jinfo["type"], q_init - self.lower, dict(self.gp.gen("tolerances") or {}))
        band = float(cfg["band_fraction"]) * tol
        q_need = self.lower + band
        z_ref = float(self.h([q_rel])[0].contact[2])  # the grasp height as drawn (bar offset, pinch sign)
        geo = {"leaf_width_m": round(leaf.width, 4), "face_offset_m": round(leaf.face, 4), "leaf_z": [round(leaf.z_lo, 3), round(leaf.z_hi, 3)], "z_ref": round(z_ref, 3)}
        cands = push_finish_candidates(leaf, z_ref, cfg)
        if not cands:
            raise Reject("push_unreachable", f"no contact on the leaf's outer face clears the hinge, the edges and the handle ({geo})")
        n = len(cands)
        ok = self.ikc.screen([push_contact_pose(leaf, c, q_rel)[0] for c in cands] + [push_contact_pose(leaf, c, q_need)[0] for c in cands], manip, self.arm,
                             ignore=[obj], fingers=C.FINGER_CLOSED)
        screened = [i for i in range(n) if bool(ok[i]) and bool(ok[n + i])]
        order = (screened + [i for i in range(n) if i not in screened])[: int(cfg["max_checks"])]
        step = math.radians(float(cfg["arc_step_deg"]))
        pre_m = float(cfg["pre_m"])
        idx = self.kit.robot.arm_control_idx[self.arm]
        key = self._stance_key(manip)
        q_now = self.joint_value()
        fails: List[str] = []
        checked = 0
        try:
            for i in order:
                c = cands[i]
                checked += 1
                tag = f"r {c['radius_m']:.2f} z {c['z']:.2f} tilt {c['tilt_deg']:.0f} {c['closing']}{'' if i in screened else ' (unscreened)'}"
                contact, a, normal = push_contact_pose(leaf, c, q_rel)
                pre = C.pre_pose(contact, pre_m)
                self.pose_joint(q_rel)
                r = self.ikc.exact([pre, contact], manip, self.arm, ignore_for_last=[obj], cache=False, fingers=C.FINGER_CLOSED, approached=[obj])
                if not r.ok:
                    fails.append(f"{tag}: pre / contact: {r.reason}")
                    continue
                q_to, arc_q = None, []
                for q_end in ([self.lower, q_need] if q_need > self.lower + 1e-6 else [self.lower]):
                    arc_q = arc_values(q_rel, q_end, step)
                    res = self.ikc.continue_chain(r.q, [contact] + [push_contact_pose(leaf, c, qq)[0] for qq in arc_q], self.arm, ignore=[obj])
                    if res.ok:
                        q_to = q_end
                        break
                    fails.append(f"{tag}: arc to {q_end:.3f}: {res.reason}")
                if q_to is None:
                    continue
                q_pre = self.ikc.solutions.get((self.arm, self.ikc._pose_key(pre)))
                p_o, q_o = obj_pose
                rec = {"rule": "owner 2026-10-12 ~04:20: hold the handle as far as the arm reaches, release, push the leaf the rest of the way shut",
                       "arm": self.arm, "stance": "manipulation", "base_pose2d": [float(v) for v in manip.base_pose2d],
                       "k_release": float(k_rel), "q_from": float(q_rel), "q_to": float(q_to), "q_lower": float(self.lower),
                       "handle_fraction": float(k_rel), "push_fraction": round((q_rel - q_to) / max(q_init - self.lower, 1e-9), 4),
                       "reaches_goal": push_reaches_goal(q_to, self.lower, band), "tolerance_rad": round(tol, 4), "band_rad": round(band, 4),
                       "hinge_world": [round(float(v), 5) for v in leaf.hinge.tolist()], "swing_sign": leaf.s, "q_closed": leaf.q_closed,
                       "contact": pose_to_sidecar(contact), "pre": pose_to_sidecar(pre), "pre_m": pre_m,
                       "approach": [round(float(v), 5) for v in a.tolist()], "normal": [round(float(v), 5) for v in normal.tolist()],
                       "arc_q": [round(v, 5) for v in arc_q], "radius_m": c["radius_m"], "inset_m": c["inset_m"], "height_m": c["z"],
                       "height_offset_m": c["dz"], "tilt_deg": c["tilt_deg"], "closing": c["closing"], **geo,
                       "arm_q_pre": None if q_pre is None else [round(float(v), 5) for v in q_pre[idx].tolist()],
                       "arm_q_contact": [round(float(v), 5) for v in r.q[idx].tolist()],
                       "object_pose": {"position": [float(v) for v in p_o.tolist()], "orientation": [float(v) for v in q_o.tolist()]},
                       "screen": {"candidates": n, "passed": len(screened), "checked": checked}}
                log.info("%s: stance %s: push finish %s from %.3f to %.3f (handle fraction %.2f + push %.2f; band %.3f): %s arm, contact %s, %d arc poses",
                         self.gp.variant, key, self.joint_name, q_rel, q_to, k_rel, rec["push_fraction"], band, self.arm, tag, len(arc_q))
                return rec
        finally:
            self.pose_joint(q_now)
        raise Reject("push_unreachable", f"{n} contact candidates ({len(screened)} pass the screen at the release {q_rel:.3f} and the band edge {q_need:.3f}), "
                                         f"{len(order)} checked: " + " | ".join(fails[:4]))

    def arm_alone_follows(self, manip: Stance, q_init: float, q_goal: float, by_k: Dict[float, C.HandleWaypoint], required: List[float],
                          arm_reach: Dict[str, bool], arm_q: Dict[str, Optional[th.Tensor]]) -> Tuple[bool, str]:
        """Whether the arm alone moves the joint through the whole follow from ``manip`` (the oracle's arm follow, no base
        motion): the grasp pose of every required waypoint has an exact arm IK with the joint there (``arm_reach``); a goal
        short of the joint's limit is also reached ``reach_margin`` beyond it (the drives' authority for the last
        centimetres, :data:`REACH_MARGIN`); and the program's retreat after the follow - ``PRE_DIST`` back along -a of the
        last handle pose - has an IK from the same stance (``reach_check_retreat``, as :meth:`arm_reach_limit`). Returns
        ``(feasible, the first check that fails)``."""
        missing = [fraction_key(k) for k in required if not arm_reach.get(fraction_key(k), False)]
        if missing:
            return False, f"no arm-only IK at the required waypoints {missing}"
        # oracle-fix 2026-11-03 (owner: the dishwasher follows arm only, else sequential): an arm-only pull longer than
        # ``hybrid_follow.arm_only_max_travel_rad`` does not count as working - the torso sags with the pull's length (dishwasher open
        # 0075: robot_wobble after 0.43 rad) - and the follow goes to the sequential search; None = no cap, as before
        cap = (self.gp.gen("hybrid_follow") or {}).get("arm_only_max_travel_rad")
        self.arm_only_capped = False
        if cap is not None and abs(float(q_goal) - float(q_init)) > float(cap) + 1e-9:
            self.arm_only_capped = True
            return False, f"arm-only travel {abs(float(q_goal) - float(q_init)):.3f} rad > hybrid_follow.arm_only_max_travel_rad {float(cap):.3f}"
        k_last = max(required)
        w = by_k[k_last]
        sign = 1.0 if q_goal >= q_init else -1.0
        room = sign * ((self.upper if sign > 0 else self.lower) - q_goal)
        try:
            if k_last >= 1.0 and room > 1e-6:
                q_auth = q_goal + sign * min(self.reach_margin(), room)
                self.pose_joint(q_auth)
                if not self.ikc.exact([self.h([q_auth])[0].eef], manip, self.arm, ignore_for_last=[self.target], q_seed=arm_q.get(fraction_key(k_last)), cache=False).ok:
                    return False, f"no arm-only IK {abs(q_auth - q_goal):.3f} past the goal {q_goal:.3f}"
            if bool(self.gp.gen("reach_check_retreat", True)):
                self.pose_joint(w.q)
                back = self.ikc.exact([C.pre_pose(w.eef, C.PRE_DIST)], manip, self.arm, q_seed=arm_q.get(fraction_key(k_last)), cache=False)
                if not back.ok:
                    return False, f"the retreat from h(q = {w.q:.3f}): {back.reason}"
        finally:
            self.pose_joint(q_init)
        return True, ""

    def _solve_from(self, manip: Stance, init: Stance, q_init: float, q_goal: float, wps: List[C.HandleWaypoint], required: List[float],
                    contact: P0.ContactSpec, h0: C.HandleWaypoint, goal_set: Optional[List[float]] = None, reach_info: Optional[Dict[str, Any]] = None,
                    fractions: Optional[Sequence[float]] = None) -> Draft:
        """Steps 5–6 of §5.2 for one drawn manipulation stance plus the §5.3 base-follow feasibility; fills the sidecar
        (``manipulation`` stance, transition, path, ``articulation``) and returns the draft. ``fractions`` = the waypoint fractions
        of ``wps[1:]`` (default ``generation.waypoint_fractions``; a push finish adds its release)."""
        obj, tname = self.target, self.tname
        manip.arms = [self.arm]
        reach_info = dict(reach_info or {})
        push = reach_info.get("push_finish")
        # step 5a: [pre-grasp, grasp] at q_init (target ignored for the grasp pose only)
        sol = P0.exact_contacts(self, manip, [contact], ignore_combined=[obj])
        if not sol.ok:
            raise Reject("handle_unreachable", sol.reason)
        q_star = sol.q
        # step 5b: arm-only IK of the grasp pose at every q_k with the world re-synced at q_k (target ignored for the
        # grasp poses) — the arm mode's waypoint check, and the oracle's secondary mode where the base follows
        fractions = list(fractions) if fractions is not None else list(self.gp.gen("waypoint_fractions", WAYPOINT_FRACTIONS))
        by_k = {k: w for (k, _), w in zip(waypoint_values(q_init, q_goal, fractions), wps[1:])}
        arm_reach: Dict[str, bool] = {}
        arm_q: Dict[str, Optional[th.Tensor]] = {}
        try:
            for k in fractions:
                if push is not None and k > float(push["k_release"]) + 1e-9:
                    # past the release the push takes the joint on: the reach sweep ended before these (not solved)
                    arm_reach[fraction_key(k)] = False
                    arm_q[fraction_key(k)] = None
                    continue
                w = by_k[k]
                self.pose_joint(w.q)
                res = self.ikc.exact([w.eef], manip, self.arm, ignore_for_last=[obj], q_seed=q_star, cache=False)
                arm_reach[fraction_key(k)] = bool(res.ok)
                arm_q[fraction_key(k)] = res.q if res.ok else None
        finally:
            self.pose_joint(q_init)
        # the primary mode decides ``handle_reach`` (F6: every followed waypoint must be reachable in the recorded mode):
        # base mode = the base-follow pose b_k passes its checks (arm-only IK recorded separately as the per-waypoint
        # fallback); arm mode = the arm-only IK
        base_follow: Optional[Dict[str, List[float]]] = None
        follow_path: Optional[Dict[str, Any]] = None  # hybrid: the dense base + arm path (``articulation.base_follow_path``)
        mode, base_follow_reason = self.mode, None
        arm_alone: Optional[Dict[str, Any]] = None
        if mode == "hybrid":
            # owner 2026-09-27: arm-only whenever feasible - the hybrid search skips the arm-only step whenever its torso-to-
            # handle pre-filter says "too far" and moves the base, although the exact IK reaches (0926-065206: 0.63-0.77 m
            # against the 0.59 m pre-filter, arm_reach all true, the base driven 0.32 m toward the dishwasher with the hand on it)
            feasible, why_not = self.arm_alone_follows(manip, q_init, q_goal, by_k, required, arm_reach, arm_q)
            arm_alone = {"feasible": feasible, "why_not": why_not or None}
            if feasible:
                mode = "arm"
                log.info("%s: stance %s: the arm alone follows every required waypoint %s and retreats -> follow_mode arm (no base motion)", self.gp.variant,
                         self._stance_key(manip), [fraction_key(k) for k in required])
        if mode in ("base", "hybrid"):
            try:
                if mode == "hybrid":
                    policy = base_follow_policy((self.gp.gen("hybrid_follow") or {}).get("policy"), self.direction, self.door_class,
                                                pushed=self.close_by_push_cfg is not None)
                    log.info("%s: stance %s: the follow needs the base (%s) -> %s follow search", self.gp.variant, self._stance_key(manip), (arm_alone or {}).get("why_not"), policy)
                    hp = AH.search_follow_path(self, manip, q_star, h0, q_init, q_goal, fractions, required, fraction_key, policy=policy)
                    base_follow, follow_path = hp.base_follow, hp.to_sidecar(bool(self.gp.gen("hybrid_follow_record_eef", True)))
                else:
                    base_follow = self._base_follow(manip, q_star, h0, by_k, required, q_init)
            except Reject as e:
                # the rigid eef->base sweep about the hinge is blocked (a door next to furniture) / no hybrid path: the
                # oracle's secondary mode carries the instance when the arm alone reaches every required waypoint (§5.3
                # "arm-only IK at each waypoint ... as the oracle's secondary mode"); otherwise the candidate is rejected
                if all(arm_reach.get(fraction_key(k), False) for k in required) and not getattr(self, "arm_only_capped", False):
                    # (not when the arm-only follow was ruled out by arm_only_max_travel_rad: then the sequential follow or nothing)
                    mode, base_follow, base_follow_reason = "arm", None, f"{e.reason}: {e}"
                    log.info("%s: %s-follow blocked (%s); the arm reaches every required waypoint -> follow_mode arm", self.gp.variant, self.mode, e)
                else:
                    raise
        if mode in ("base", "hybrid"):
            handle_reach = {fraction_key(k): (fraction_key(k) in base_follow) or arm_reach[fraction_key(k)] for k in fractions}
        else:
            handle_reach = dict(arm_reach)
        missing = [k for k in required if not handle_reach.get(fraction_key(k), False)]
        if missing:
            raise Reject("handle_unreachable", f"grasp pose unreachable in {mode} mode at required waypoints {missing} (arm-only reach {arm_reach})")
        # step 6: path + transition init -> manipulation
        manip.contacts = ["handle"]
        self.sidecar.add_stance(manip)
        init = self._connect_init(init, manip, float(self.gp.gen("path_max_m", P0.PATH_MAX_M)))
        # sidecar articulation block (§7; F6 Articulation)
        waypoints = {"0.0": h0.to_sidecar()}
        for k in fractions:
            waypoints[fraction_key(k)] = by_k[k].to_sidecar()
        pf = self.passive.get(f"{q_init:.4f}->{q_goal:.4f}")
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": "prismatic" if "rismatic" in self.jinfo["type"] else "revolute",
                                     "q_init": float(q_init), "q_goal": float(q_goal), "waypoints": waypoints, "handle_reach": handle_reach,
                                     "passive_from": pf, "follow_mode": mode, "base_follow": base_follow,
                                     "handle": {"kind": self.handle.get("kind"), "classes": list(contact.classes or ()), "width_m": self.handle.get("width"),
                                                "pinch_sign": float(self.pinch_sign), "bar_offset_m": float(self.bar_offset)},
                                     "required_fractions": [fraction_key(k) for k in required], "arm_reach": arm_reach,
                                     "primary_follow_mode": self.mode, "base_follow_blocked": base_follow_reason, "arm_alone_follow": arm_alone,
                                     # A4: the class and the stance-side rule behind the follow mode; the goal set of this stance
                                     "door_class": self.door_class, "hinge_side": self.hinge_side, "goal_set": [float(s) for s in (goal_set or self.stable)],
                                     "reachable_fraction": reach_info.get("reachable_fraction"), "q_reach": reach_info.get("q_reach"),
                                     "q_reach_boundary": reach_info.get("q_reach_boundary"), "reach_margin": reach_info.get("reach_margin"),
                                     "joint_friction": self.joint_friction, "counterbalance": self.counterbalance}
        if follow_path is not None:
            self.sidecar.articulation["base_follow_path"] = follow_path
        if push is not None:
            # the hand's part ends at the release (``follow_to``); ``push_close`` takes the leaf on with the planned chunk
            self.sidecar.articulation["follow_to"] = float(push["k_release"])
            self.sidecar.articulation["push_finish"] = dict(push)
        push_close = None
        if self.close_by_push_cfg is not None:
            # no grasp: ``push_close`` pushes the leaf shut from q_init as a room door's (the follow above was the sampling gate)
            push_close = close_by_push_record(closed_leaf(self.obj_pose(), self.jinfo, self.handle), self.close_by_push_cfg)
            self.sidecar.articulation["close_by_push"] = True
            self.sidecar.articulation["push_close"] = push_close
        self.sidecar.describe(tname, obj.category.replace("_", " "))
        # draft: joints only (nothing spawned), robot at the init stance, the moving link held while settling
        d = Draft()
        names = list(obj.joints)
        q_now = obj.get_joint_positions().clone()
        q_init_vec = q_now.clone()
        for n, j in obj.joints.items():
            q_init_vec[names.index(n)] = float(getattr(self, "other_joints_init", {}).get(n, j.lower_limit))
        q_init_vec[names.index(self.joint_name)] = float(q_init)
        q_goal_vec = q_init_vec.clone()
        q_goal_vec[names.index(self.joint_name)] = float(q_goal)
        d.object_joints[tname] = q_init_vec
        d.goal_joints[tname] = q_goal_vec
        constraints: List[Dict[str, Any]] = []
        if self.counterbalance is not None:
            # the loader re-applies it (run_isg._restore_constraints); every physics probe above ran with it
            constraints.append({"type": "counterbalance", "object": tname, "joint": self.joint_name, "links": list(self.counterbalance["links"]),
                                "damping": float(self.counterbalance["damping"]),
                                "note": "spring-balanced appliance door (generation.counterbalance): no gravity on the leaf assembly, viscous joint damping - "
                                        "the leaf holds at any angle with the stock friction and stops where the hand leaves it"})
        if self.joint_friction is not None:
            # the loader re-applies it (run_isg._restore_constraints); the stable set / passive probes above ran with it
            constraints.append({"type": "joint_friction", "object": tname, "joint": self.joint_name, "friction": float(self.joint_friction),
                                "stock_friction": 0.2, "note": "PhysX articulation joint friction of the moved joint (generation.joint_friction): the brake "
                                                                  "|F_resist| <= mu (|F| + |T|) that holds the link where the hand leaves it",
                                # oracle-fix 2026-11-04 round 3: with other_joints_init (a neighbouring door held open) Open must read the target joint only
                                **({"openable_joints": [self.joint_name]} if getattr(self, "other_joints_init", None) else {})})
        if constraints:
            d.extras["constraints"] = constraints
        tols = dict(self.gp.gen("tolerances") or {})
        d.articulation_targets[tname] = {"joints": {n: {"position": float(q_goal_vec[names.index(n)]),
                                                        "tolerance": joint_tolerance("RevoluteJoint" if obj.joints[n].is_revolute else "PrismaticJoint",
                                                                                     (q_goal - q_init) if n == self.joint_name else 0.0, tols)} for n in names}}
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.settle_free = []
        d.summary["handle_reach"] = handle_reach
        d.summary["arm_reach"] = arm_reach
        d.summary["base_follow"] = base_follow
        d.summary["follow_mode"] = mode
        d.summary["base_follow_blocked"] = base_follow_reason
        d.summary["arm_alone_follow"] = arm_alone
        if follow_path is not None:
            d.summary["hybrid_follow"] = dict(follow_path.get("stats") or {})
        if push_close is not None:
            d.summary["close_by_push"] = True
            d.summary["push_close"] = dict(push_close)
        return d

    def _base_follow(self, manip: Stance, q_star: th.Tensor, h0: C.HandleWaypoint, by_k: Dict[float, C.HandleWaypoint], required: List[float],
                     q_init: float) -> Dict[str, List[float]]:
        """§5.3 base mode: b_k = h(q_k) ∘ T_eef→base(q*) per required waypoint — free floor (un-eroded) in the task
        rooms, whole-body collision-free with the target at q_k (target removed for the check, moving link vs base
        footprint as a planar stand-in), base path b_{k−1} → b_k collision-checked per §3.7 with the arm locked."""
        obj = self.target
        bz = P0.base_z(self)
        # the whole step-5 configuration travels with the base: working arm at q*, the idle arm in the posture the exact
        # IK was seeded with (A6 ``idle_posture``, e.g. tucked) and the fingers as recorded — composing "untucked" for the
        # idle arm here collided at every b_k for the Rs_int drawer (20:14) although the stance itself was collision-free
        arm_q = {a: q_star[self.kit.robot.arm_control_idx[a]].clone() for a in ("right", "left")}
        ks = [k for k in sorted(by_k) if k in required]
        h_eefs = [by_k[k].eef for k in ks]
        targets = base_follow_targets(h_eefs, h0.eef, manip.base_pose2d, bz)
        residual = C.base_follow_residual(h_eefs, C.eef_base_transform(h0.eef, C.pose2d_to_pose(*manip.base_pose2d, z=bz)))
        if residual > 0.05:
            raise Reject("base_follow_infeasible", f"non-planar base-follow residual {residual:.3f} (m + rad) for a {self.mode}-mode joint")
        out: Dict[str, List[float]] = {}
        prev = tuple(manip.base_pose2d)
        trunk = (manip.translate, manip.tilt)
        try:
            for k, b in zip(ks, targets):
                b = (float(b[0]), float(b[1]), float(b[2]))
                if not self.kit.trav_free(b[:2]):  # §5.3: b_k on free floor (un-eroded); the room rule binds the init stance only
                    raise Reject("base_follow_infeasible", f"b_{k:g} = {[round(v, 3) for v in b]} is off the free floor")
                self.pose_joint(by_k[k].q)
                q_b = self.kit.q_at_base_pose(q_star, *b)
                with self.kit.ignoring([obj]):
                    free = bool(self.kit.collision_free(q_b, margin=P0.COLLISION_MARGIN)[0])
                if not free:
                    dbg = P0.debug_collision_q(self, q_b, f"base-follow b_{k:g} = {[round(v, 3) for v in b]}", ignore=[obj])
                    raise Reject("base_follow_infeasible", f"whole body collides at b_{k:g} (target at q_{k:g} removed from the check; "
                                                           f"ground={dbg.get('no_ground')} margin0={dbg.get('margin0')} objects={dbg.get('objects_at_spheres')})")
                if not leaf_clear_of_base(self.moving_link_footprint(), b[:2]):
                    raise Reject("base_follow_infeasible", f"the moving link sweeps into the base footprint at q_{k:g}")
                poses = NAV.stride_poses([prev, b], NAV.STRIDE_M)
                with self.kit.ignoring([obj]):
                    ok, bad = NAV.path_collision_free(self.kit, poses, trunk, "untucked", q_arms=arm_q)
                if not ok:
                    raise Reject("base_follow_infeasible", f"base path b -> b_{k:g} collides at stride {bad}/{len(poses)}")
                out[fraction_key(k)] = list(b)
                prev = b
        finally:
            self.pose_joint(q_init)
        return out

    # ------------------------------------------------------------------ pipeline hooks (§5.3 Phase 3)

    def apply_draft(self, draft: Draft):
        super().apply_draft(draft)
        # hold the moving link at q_init during settling (position control is not available on scene objects)
        self.set_joint(float(draft.object_joints[self.tname][self.joint_index()]))

    def after_settle(self, draft: Draft):
        q_now = self.joint_value()
        q_want = float(draft.object_joints[self.tname][self.joint_index()])
        if abs(q_now - q_want) > 0.5 * self.probe_tol:
            raise Reject("articulation_drift", f"init joint drifted from {q_want:.3f} to {q_now:.3f} while settling")

    def refine_targets_from_goal(self, draft: Draft):
        obj = self.target
        q = obj.get_joint_positions()
        names = list(obj.joints)
        tgt = draft.articulation_targets[self.tname]["joints"]
        for n in names:
            if abs(float(q[names.index(n)]) - tgt[n]["position"]) > 0.5 * tgt[n]["tolerance"]:
                raise Reject("goal_unstable", f"joint {n} drifted to {float(q[names.index(n)]):.3f} (target {tgt[n]['position']:.3f})")

    def support_objects(self) -> List[Any]:
        return []

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        manip = self.sidecar.stances.get("manipulation")
        if manip is None or manip.face is None:
            return []
        return [{"scope": {"scene": self.scene, "target": self.target.name, "joint": self.joint_name}, "type": "stance",
                 "face": manip.face.name, "stand_off": [manip.stand_off, manip.stand_off], "lateral": [manip.lateral, manip.lateral],
                 "trunk": {"translate": [manip.translate, manip.translate], "tilt": [round(manip.tilt, 3), round(manip.tilt, 3)]}, "arm": self.arm,
                 "hard": False},
                {"scope": {"scene": self.scene, "target": self.target.name, "joint": self.joint_name}, "type": "joint_set", "stable": list(self.stable),
                 "passive_from": dict(self.passive), "hard": False}]
