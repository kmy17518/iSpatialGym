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
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import FaceFrame, Stance

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
PINCH_WIDTH_RANGE = (0.015, 0.080)
#: §5.3 waypoint fractions
WAYPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
#: base footprint radius of the planar leaf-vs-base stand-in check (R1Pro chassis half-diagonal)
BASE_RADIUS = 0.33


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
# the task module
# ----------------------------------------------------------------------------------------------------------------


@register(*SUBTASKS)
class ArticulationLarge(SamplerV2):
    """One class for open and close; the direction comes from the subtask id (``generation.direction`` agrees)."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.direction = "close" if "close" in self.subtask else "open"
        if self.gp.gen("direction") in ("open", "close"):
            self.direction = str(self.gp.gen("direction"))
        if self.gp.bddl is None:  # facade: the v1 writer's definition id
            self.definition_id = V1_DEFINITION_ID[self.direction]
        self.target_role = "target" if "target" in self.gp.roles else ("artic" if "artic" in self.gp.roles else "target")
        self.handle: Dict[str, Any] = {}
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

    # ------------------------------------------------------------------ roles / bindings

    def v1_role_aliases(self) -> Dict[str, str]:
        return {"artic": "target"} if self.target_role == "target" else {}

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get(self.target_role)
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.{self.target_role} must list scene object names")
        self.bind_scene_object(self.target_role, self.world.obj(names[0]))

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
        """The joint (``roles.target.joint``), its handle (§5.3 rules incl. the icl rows) and the joint info in the
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
                raise Reject("spec", f"{self.gp.alias}: roles.{self.target_role}.joint is required (no annotation to pick the joint from)")
            joint_name = joints[0]["joint"]
        if joint_name not in obj.joints:
            raise Reject("spec", f"{obj.name} has no joint {joint_name!r} (joints {list(obj.joints)})")
        j = joints[0] if joints else {"joint": joint_name, "type": "RevoluteJoint" if obj.joints[joint_name].is_revolute else "PrismaticJoint"}
        jinfo = dict(j) if frame == "instance" else (scale_joint_info(j, scale) if frame == "canonical" else dict(j))
        rows = record_handle_rows(jinfo) if data is not None else []
        rows += icl_handle_rows(key, joint_name, self.world.scene_model, obj.name, instance_scale=None if frame == "instance" else scale.tolist())
        if joint_excluded(rows):
            raise Reject("missing_handle", f"{obj.name}/{joint_name}: icl verdict skip / ungraspable and no admissible pinch")
        ranked = rank_handle_rows(rows)
        if not ranked:
            raise Reject("missing_handle", f"{obj.name}/{joint_name}: no bar / knob / edge / reviewer pinch in {path and os.path.relpath(path, REPO_ROOT)} or {HANDLES_CSV}")
        row = ranked[0]
        handle = row.as_handle()
        width = handle.get("width")
        if width is not None and not (PINCH_WIDTH_RANGE[0] <= float(width) <= PINCH_WIDTH_RANGE[1]):
            # the reviewer's / record's next pinch inside the range, if any
            alt = [r for r in ranked[1:] if r.width is None or PINCH_WIDTH_RANGE[0] <= float(r.width) <= PINCH_WIDTH_RANGE[1]]
            if alt:
                row, handle = alt[0], alt[0].as_handle()
            else:
                raise Reject("missing_handle", f"{obj.name}/{joint_name}: pinch width {1000 * float(width):.0f} mm outside {[1000 * w for w in PINCH_WIDTH_RANGE]} mm")
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
            log.info("%s: %s/%s edge pinch: recorded approach %s is parallel to the pinch axis %s — replaced by the leaf-plane approach %s", self.gp.alias, obj.name,
                     joint_name, handle.get("approach_dir"), handle.get("pinch_axis"), [round(v, 3) for v in repaired])
            handle["approach_dir_recorded"] = list(handle.get("approach_dir", []))
            handle["approach_dir"] = repaired
        self.joint_name, self.joint, self.handle, self.jinfo, self.handle_rows = joint_name, jt, handle, jinfo, ranked
        self.lower, self.upper = float(jt.lower_limit), float(jt.upper_limit)
        self.template_facts["handle"] = {"source": path and os.path.relpath(path, REPO_ROOT), "frame": frame, "row": row.source, "kind": row.kind,
                                         "width_mm": None if width is None else round(1000 * float(handle.get("width") or width), 1),
                                         "contact_obj": handle["contact"], "pinch_axis_obj": handle["pinch_axis"], "approach_dir_obj": handle["approach_dir"],
                                         "joint": joint_name, "type": jinfo["type"], "axis_dir_obj": jinfo["axis_dir_obj"], "hinge_point_obj": jinfo["hinge_point_obj"],
                                         "instance_scale": scale.tolist(), "n_rows": len(ranked)}
        log.info("%s: handle of %s/%s from %s (%s): %s", self.gp.alias, obj.name, joint_name, self.template_facts["handle"]["source"], row.source, handle)

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
        cuRobo meshes) follow, set again (no residual velocity), then re-sync the collision world."""
        self.set_joint(q)
        self.world.step_physics(1, still=[self.robot])
        self.set_joint(q)
        self.kit.sync_world()

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
        self.stable = self.stable_set()
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
        # A4 (owner decision 2026‑09‑16): class → follow mode / stance side / arm
        self.door_class = door_class(obj.category, self.jinfo["type"])
        self.mode = follow_mode_by_class(self.door_class, self.jinfo["type"], axis_world)
        lam_cfg = self.gp.gen("lateral_range_m") or {"right": [-0.10, 0.60], "left": [-0.60, 0.10]}
        self.lam_range = tuple(lam_cfg[self.arm]) if isinstance(lam_cfg, dict) else tuple(lam_cfg)
        self.hinge_side = None
        if self.door_class == "cabinet" and axis_is_vertical(axis_world) and self.jinfo.get("hinge_point_obj") is not None:
            hinge_w = G.local_points_world(th.as_tensor([self.jinfo["hinge_point_obj"]], dtype=th.float32), self.obj_pose())[0]
            self.hinge_side = hinge_side(hinge_w, contact, self.face_view.tangent)
            lam_max = max(abs(float(v)) for v in self.lam_range) if self.lam_range else 0.60
            if self.gp.gen("arm", "auto") not in ("left", "right"):
                self.arm = arm_for_hinge_side(self.hinge_side)
            self.lam_range = hinge_side_lateral(self.hinge_side, lam_max)
        self.template_facts.update({"joint": self.joint_name, "joint_type": self.jinfo["type"], "limits": [self.lower, self.upper], "direction": self.direction,
                                    "stable_set": self.stable, "stable_fractions": [(s - self.lower) / max(self.upper - self.lower, 1e-9) for s in self.stable],
                                    "passive_from": dict(self.passive), "face_normal": n.tolist(), "face_centre": Pc.tolist(), "P_manip": P_manip.tolist(),
                                    "handle_offset_left_m": off_left, "arm": self.arm, "follow_mode": self.mode, "axis_world": axis_world.tolist(),
                                    "door_class": self.door_class, "hinge_side": self.hinge_side, "lateral_range_m": list(self.lam_range)})
        log.info("%s: %s/%s %s: class %s, S=%s, arm %s (handle %.2f m left of the face centre, hinge %s), follow_mode %s, Λ %s", self.gp.alias, obj.name,
                 self.joint_name, self.direction, self.door_class, [round(s, 3) for s in self.stable], self.arm, off_left, self.hinge_side, self.mode, self.lam_range)
        if self.mode == "hybrid":
            # A4: room doors follow with base poses re-chosen per handle waypoint (free floor near the current base, arm
            # IK feasible, minimum displacement) — not implemented yet (owner: after the pilot if time is short)
            raise Reject("room_door_hybrid_todo", f"{obj.name}/{self.joint_name}: class room_door needs the hybrid base-follow (§5.3 A4), not implemented")

    def stable_set(self) -> List[float]:
        """§5.3 stable set S: 7 fractions of ``open_fraction_range`` + the upper limit, each held 120 steps and kept when
        it stays within ½ tolerance (the probe tolerance: revolute 0.10 rad, prismatic 0.02 m)."""
        lo, hi = self.gp.gen("open_fraction_range", [0.5, 0.95])
        n = int(self.gp.gen("stable_set_fractions", 7))
        fracs = [float(f) for f in th.linspace(float(lo), float(hi), n).tolist()] + [1.0]
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
                log.info("%s: %s at fraction %.2f (%.3f) -> %.3f after %d steps: %s", self.gp.alias, self.joint_name, f, q, q_after, steps, "stable" if ok else "drifts")
                if ok and all(abs(q - s) > 1e-6 for s in out):
                    out.append(q)
        finally:
            self.world.restore(state)
            self.target.keep_still()
            self.kit.sync_world()
        return out

    def passive_from(self, q_init: float, q_goal: float) -> Optional[float]:
        """§5.3 required-waypoint probe: the first k whose q_k, held 120 steps, ends within tolerance of q_goal."""
        steps = int(self.gp.gen("stable_hold_steps", 120))
        tol = self.probe_tol
        state = self.world.snapshot()
        found = None
        try:
            for k, qk in waypoint_values(q_init, q_goal, self.gp.gen("waypoint_fractions", WAYPOINT_FRACTIONS)):
                if k >= 1.0:
                    break
                self.set_joint(qk)
                self.world.step_physics(steps, still=[self.robot])
                if abs(self.joint_value() - q_goal) <= tol:
                    found = k
                    break
        finally:
            self.world.restore(state)
            self.target.keep_still()
            self.kit.sync_world()
        return found

    # ------------------------------------------------------------------ instance (§5.2 / §5.3)

    def draw_joint_targets(self, rng: np.random.Generator) -> Tuple[float, float]:
        if self.direction == "open":
            return self.lower, float(self.stable[int(rng.integers(len(self.stable)))])
        return float(self.stable[int(rng.integers(len(self.stable)))]), self.lower

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
        gen_p0 = dict(self.gp.gen("prompt0") or {})
        beta_max = math.radians(float(gen_p0.get("beta_max_deg", 20.0)))
        d_min, d_max = float(gen_p0.get("d_min_m", 0.5)), float(gen_p0.get("d_max_m", 2.5))
        init = P0.phase1_init_pose(self, self.face_view, beta_max, d_min, d_max, self.target_extent, rng)
        init.name = "init"
        self.sidecar.add_stance(init)
        # Phase 2: contact set = the h(q_init) grasp of the handle arm
        h0 = self.h([q_init])[0]
        classes = pinch_classes(h0.pinch, self.handle.get("kind", "bar"), float(self.gp.gen("pinch_axis_vertical_threshold", 0.7)))
        contact = P0.ContactSpec("handle", self.arm, h0.eef, h0.pre_grasp, classes=classes, ignore_for_last=[obj])
        lam_range = tuple(self.lam_range) if self.lam_range else (-0.10, 0.60)
        d_range = (0.35, float(gen_p0.get("d_scan_max_m", 1.10)))
        half = float(self.gp.gen("face_patch_half_width_m", 0.10))
        patch = th.stack([h0.contact - h0.pinch * half, h0.contact, h0.contact + h0.pinch * half])

        def extra_vis(cam) -> bool:  # §5.3: contact(q_init) and the moving face patch (contact ± 0.10 m along the pinch axis) in frame
            return P0.in_frame(self, cam, patch)

        scope = {"scene": self.world.scene_model, "target": obj.name, "joint": self.joint_name, "q_init": round(q_init, 2), "arm": self.arm,
                 "sampler": "articulation_large", "class": self.door_class, "lam": [round(v, 2) for v in lam_range]}
        stats: Dict[str, Any] = {}
        survivors = P0.phase2_manipulation(self, self.face_manip, [contact], lam_range, [self.arm], d_range, extra_vis, focus=h0.contact, cache_scope=scope,
                                           cache_contacts=True, stats=stats)
        if not survivors:
            raise Reject("no_manipulation_stance", f"Phase 2 left no survivor for {obj.name}/{self.joint_name} at q_init {q_init:.3f} ({stats})")
        pool = list(survivors)
        errors: List[str] = []
        for _ in range(min(len(pool), 8)):
            i, manip = P0.draw_manipulation(self, pool, rng, [self.arm], focus=h0.contact, extra_visibility=extra_vis)
            pool.pop(i)
            if manip is None:
                errors.append("drawn tilt fails the re-check")
                continue
            # both closing signs of the pinch (the wrist turned by a half turn about the approach): the reach-volume class is
            # the same, the exact IK is not — one of the two is usually reachable where the other is not
            for sign in (1.0, -1.0):
                self.pinch_sign = sign
                h0s = self.h([q_init])[0]
                contact_s = P0.ContactSpec("handle", self.arm, h0s.eef, h0s.pre_grasp, classes=classes, ignore_for_last=[obj])
                try:
                    goal_set = None
                    reach_info: Dict[str, Any] = {}
                    if arm_mode:
                        q_goal, goal_set, reach_info = self.goal_for_stance(manip, init, q_init, contact_s, rng)
                    wps = self.h([q_init] + [qk for _, qk in waypoint_values(q_init, q_goal, fractions)])
                    pf = self.passive_for(q_init, q_goal)
                    required = required_fractions(pf, fractions, self.gp.gen("require_goal_handle", "auto"))
                    d = self._solve_from(manip, init, q_init, q_goal, wps, required, contact_s, h0s, goal_set=goal_set, reach_info=reach_info)
                except Reject as e:
                    errors.append(f"pinch {sign:+.0f}: {e.reason}: {e}")
                    self.sidecar.stances.pop("manipulation", None)
                    self.sidecar.transitions.clear()
                    self.sidecar.paths.clear()
                    self.sidecar.articulation = None
                    continue
                d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_init": q_init, "q_goal": q_goal, "required_fractions": required,
                                  "passive_from": pf, "arm": self.arm, "follow_mode": self.mode, "door_class": self.door_class, "hinge_side": self.hinge_side,
                                  "pinch_sign": sign, "phase2": stats, "init_stance": P0.summarize_stance(init), "manipulation_stance": P0.summarize_stance(manip),
                                  **reach_info})
                return d
        self.pinch_sign = 1.0
        raise Reject("handle_unreachable", f"{len(survivors)} Phase-2 survivors, none passes step 5 / base-follow / step 6: " + "; ".join(errors[-4:]))

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

    def arm_reach_limit(self, manip: Stance, q_from: float, q_seed: Optional[th.Tensor]) -> float:
        """The largest joint value on the grid ``q_from → upper`` (10° / 5 cm steps, upper included) whose handle grasp
        pose the arm reaches from ``manip`` with the world re-synced at that value (target ignored for the grasp pose);
        the sweep stops at the first miss, so every value up to the limit is reachable."""
        obj = self.target
        step_cfg = dict(self.gp.gen("reach_grid_step") or REACH_GRID_STEP)
        step = math.radians(float(step_cfg.get("revolute_deg", 10.0))) if self.joint.is_revolute else float(step_cfg.get("prismatic_m", 0.05))
        q_now = self.joint_value()
        reach, seed = float(q_from), q_seed
        try:
            for q in reach_grid(q_from, self.upper, step):
                self.pose_joint(q)
                w = self.h([q])[0]
                res = self.ikc.exact([w.eef], manip, self.arm, ignore_for_last=[obj], q_seed=seed, cache=False)
                if not res.ok:
                    break
                reach = q
                if res.q is not None:
                    seed = res.q
        finally:
            self.pose_joint(q_now)
        return reach

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
                     "pad in base %s, classes %s, volume %s; the world-free solution penetrates %s", self.gp.alias, self._stance_key(manip), self.arm, self.pinch_sign,
                     r_g.ok, r_p.ok, r_free.ok, r_o.ok, [round(float(v), 3) for v in p_base.tolist()], list(contact.classes or ()), rv_ok, culprits)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the attempt
            log.info("%s: handle IK debug failed: %s", self.gp.alias, e)

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
        q_reach = self.arm_reach_limit(manip, self.lower, sol.q)
        span = max(self.upper - self.lower, 1e-9)
        frac = (q_reach - self.lower) / span
        key = self._stance_key(manip)
        rf = self.template_meta.setdefault("reachable_fraction", {})
        rf[key] = round(frac, 4)
        self.template_meta["reachable_fraction_max"] = max(float(v) for v in rf.values())
        min_cfg = dict(self.gp.gen("min_reachable_opening") or MIN_REACHABLE_OPENING)
        min_open = math.radians(float(min_cfg.get("revolute_deg", 50.0))) if self.joint.is_revolute else float(min_cfg.get("prismatic_m", 0.10))
        goal_set = restrict_goal_set(self.stable, q_reach, self.lower, min_open)
        if goal_set and goal_set[0] not in self.stable and not self.holds_passively(goal_set[0]):
            goal_set = []
        info = {"reachable_fraction": frac, "q_reach": q_reach, "goal_set": list(goal_set)}
        log.info("%s: stance %s reaches %s up to %.3f (fraction %.2f); goal set %s", self.gp.alias, key, self.joint_name, q_reach, frac, [round(s, 3) for s in goal_set])
        if not goal_set:
            raise Reject("handle_unreachable", f"the arm reaches {self.joint_name} only up to {q_reach:.3f} (fraction {frac:.2f}) from stance {key}: no goal opens ≥ {min_open:.2f}")
        if self.direction == "open":
            return float(goal_set[int(rng.integers(len(goal_set)))]), goal_set, info
        if q_init > q_reach + 1e-6:
            raise Reject("handle_unreachable", f"closing from {q_init:.3f} needs the arm to reach up to it; the limit from stance {key} is {q_reach:.3f}")
        return float(self.lower), goal_set, info

    def _solve_from(self, manip: Stance, init: Stance, q_init: float, q_goal: float, wps: List[C.HandleWaypoint], required: List[float],
                    contact: P0.ContactSpec, h0: C.HandleWaypoint, goal_set: Optional[List[float]] = None, reach_info: Optional[Dict[str, Any]] = None) -> Draft:
        """Steps 5–6 of §5.2 for one drawn manipulation stance plus the §5.3 base-follow feasibility; fills the sidecar
        (``manipulation`` stance, transition, path, ``articulation``) and returns the draft."""
        obj, tname = self.target, self.tname
        manip.arms = [self.arm]
        reach_info = dict(reach_info or {})
        # step 5a: [pre-grasp, grasp] at q_init (target ignored for the grasp pose only)
        sol = P0.exact_contacts(self, manip, [contact], ignore_combined=[obj])
        if not sol.ok:
            raise Reject("handle_unreachable", sol.reason)
        q_star = sol.q
        # step 5b: arm-only IK of the grasp pose at every q_k with the world re-synced at q_k (target ignored for the
        # grasp poses) — the arm mode's waypoint check, and the oracle's secondary mode where the base follows
        fractions = list(self.gp.gen("waypoint_fractions", WAYPOINT_FRACTIONS))
        by_k = {k: w for (k, _), w in zip(waypoint_values(q_init, q_goal, fractions), wps[1:])}
        arm_reach: Dict[str, bool] = {}
        try:
            for k in fractions:
                w = by_k[k]
                self.pose_joint(w.q)
                res = self.ikc.exact([w.eef], manip, self.arm, ignore_for_last=[obj], q_seed=q_star, cache=False)
                arm_reach[fraction_key(k)] = bool(res.ok)
        finally:
            self.pose_joint(q_init)
        # the primary mode decides ``handle_reach`` (F6: every followed waypoint must be reachable in the recorded mode):
        # base mode = the base-follow pose b_k passes its checks (arm-only IK recorded separately as the per-waypoint
        # fallback); arm mode = the arm-only IK
        base_follow: Optional[Dict[str, List[float]]] = None
        mode, base_follow_reason = self.mode, None
        if mode == "base":
            try:
                base_follow = self._base_follow(manip, q_star, h0, by_k, required, q_init)
            except Reject as e:
                # the rigid eef->base sweep about the hinge is blocked (a door next to furniture): the oracle's secondary
                # mode carries the instance when the arm alone reaches every required waypoint (§5.3 "arm-only IK at each
                # waypoint ... as the oracle's secondary mode"); otherwise the candidate is rejected
                if all(arm_reach.get(fraction_key(k), False) for k in required):
                    mode, base_follow, base_follow_reason = "arm", None, f"{e.reason}: {e}"
                    log.info("%s: base-follow blocked (%s); the arm reaches every required waypoint -> follow_mode arm", self.gp.alias, e)
                else:
                    raise
        if mode == "base":
            handle_reach = {fraction_key(k): (fraction_key(k) in base_follow) or arm_reach[fraction_key(k)] for k in fractions}
        else:
            handle_reach = dict(arm_reach)
        missing = [k for k in required if not handle_reach.get(fraction_key(k), False)]
        if missing:
            raise Reject("handle_unreachable", f"grasp pose unreachable in {mode} mode at required waypoints {missing} (arm-only reach {arm_reach})")
        # step 6: path + transition init -> manipulation
        manip.contacts = ["handle"]
        self.sidecar.add_stance(manip)
        P0.connect(self, init, manip, float(self.gp.gen("path_max_m", P0.PATH_MAX_M)))
        # sidecar articulation block (§7; F6 Articulation)
        waypoints = {"0.0": h0.to_sidecar()}
        for k in fractions:
            waypoints[fraction_key(k)] = by_k[k].to_sidecar()
        pf = self.passive.get(f"{q_init:.4f}->{q_goal:.4f}")
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": "prismatic" if "rismatic" in self.jinfo["type"] else "revolute",
                                     "q_init": float(q_init), "q_goal": float(q_goal), "waypoints": waypoints, "handle_reach": handle_reach,
                                     "passive_from": pf, "follow_mode": mode, "base_follow": base_follow,
                                     "handle": {"kind": self.handle.get("kind"), "classes": list(contact.classes or ()), "width_m": self.handle.get("width"),
                                                "pinch_sign": float(self.pinch_sign)},
                                     "required_fractions": [fraction_key(k) for k in required], "arm_reach": arm_reach,
                                     "primary_follow_mode": self.mode, "base_follow_blocked": base_follow_reason,
                                     # A4: the class and the stance-side rule behind the follow mode; the goal set of this stance
                                     "door_class": self.door_class, "hinge_side": self.hinge_side, "goal_set": [float(s) for s in (goal_set or self.stable)],
                                     "reachable_fraction": reach_info.get("reachable_fraction"), "q_reach": reach_info.get("q_reach")}
        self.sidecar.describe(tname, obj.category.replace("_", " "))
        # draft: joints only (nothing spawned), robot at the init stance, the moving link held while settling
        d = Draft()
        names = list(obj.joints)
        q_now = obj.get_joint_positions().clone()
        q_init_vec = q_now.clone()
        for n, j in obj.joints.items():
            q_init_vec[names.index(n)] = float(j.lower_limit)
        q_init_vec[names.index(self.joint_name)] = float(q_init)
        q_goal_vec = q_init_vec.clone()
        q_goal_vec[names.index(self.joint_name)] = float(q_goal)
        d.object_joints[tname] = q_init_vec
        d.goal_joints[tname] = q_goal_vec
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
        return d

    def _base_follow(self, manip: Stance, q_star: th.Tensor, h0: C.HandleWaypoint, by_k: Dict[float, C.HandleWaypoint], required: List[float],
                     q_init: float) -> Dict[str, List[float]]:
        """§5.3 base mode: b_k = h(q_k) ∘ T_eef→base(q*) per required waypoint — free floor (un-eroded) in the task
        rooms, whole-body collision-free with the target at q_k (target removed for the check, moving link vs base
        footprint as a planar stand-in), base path b_{k−1} → b_k collision-checked per §3.7 with the arm locked."""
        obj = self.target
        bz = P0.base_z(self)
        arm_idx = self.kit.robot.arm_control_idx[self.arm]
        arm_q = q_star[arm_idx].clone()
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
                q_b = self.kit.q_at_base_pose(self.kit.q_config(arms="untucked", translate=trunk[0], tilt=trunk[1], **{f"{self.arm}_arm": arm_q}), *b)
                with self.kit.ignoring([obj]):
                    free = bool(self.kit.collision_free(q_b, margin=P0.COLLISION_MARGIN)[0])
                if not free:
                    raise Reject("base_follow_infeasible", f"whole body collides at b_{k:g} with the target at q_{k:g}")
                if not leaf_clear_of_base(self.moving_link_footprint(), b[:2]):
                    raise Reject("base_follow_infeasible", f"the moving link sweeps into the base footprint at q_{k:g}")
                poses = NAV.stride_poses([prev, b], NAV.STRIDE_M)
                with self.kit.ignoring([obj]):
                    ok, bad = NAV.path_collision_free(self.kit, poses, trunk, "untucked", q_arms={self.arm: arm_q})
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
