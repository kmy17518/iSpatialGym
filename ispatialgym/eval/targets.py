"""Evaluation targets stored in the reserved ``ispatialgym`` block of a TRO file (schema ``isg-tro-v1``).

This module only parses and validates the numeric targets; it does not touch the simulator. See
``docs/task_bddl_formats.md`` ("Shared TRO extension") for the schema.

Pose target (``pose_targets[<instance>]``)::

    {
      "link": null | "<link or sensor key on the instance>",
      "frame": "world" | "<instance>" | "<instance>::<link or sensor key>",
      "position": [x, y, z] | null,
      "orientation": [x, y, z, w] | null,
      "position_metric": "full" | "depth" | null,      # default: "full" if position given else null
      "orientation_metric": "full" | null,             # default: "full" if orientation given else null
      "position_tolerance": meters | null,             # required when position_metric is set
      "orientation_tolerance": radians | null,         # required when orientation_metric is set
      "orientation_equivalences": [[x, y, z, w], ...]  # optional, object-frame symmetries
    }

``position_metric = "depth"`` compares only the depth along the frame's optical axis (the frame must be a camera
sensor, USD convention: optical axis = -Z). A target whose instance is not referenced by a ``pose_reached`` goal atom
is *measured* (its errors are reported) but does not gate success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch as th

from ispatialgym.activity.problem import canonical_variant
from ispatialgym.eval.geometry import as_tensor, quat_angle, quat_normalize, relative_pose

SCHEMA_VERSION = "isg-tro-v1"
TRO_KEY = "ispatialgym"

POSITION_METRICS = ("full", "depth")
ORIENTATION_METRICS = ("full",)
ASSIGNMENT_MODES = ("exchangeable", "fixed")
ASSIGNMENT_COSTS = ("layout", "slot_position")

RELATION_POSITION_ATOL = 1e-4
RELATION_ORIENTATION_ATOL = 1e-3


class TROValidationError(ValueError):
    """Raised when the ``ispatialgym`` TRO block is malformed or inconsistent."""


# ----------------------------------------------------------------------------------------------------------------
# Field helpers
# ----------------------------------------------------------------------------------------------------------------


def _vec(value, n: int, what: str) -> th.Tensor:
    t = as_tensor(value).flatten()
    if t.numel() != n:
        raise TROValidationError(f"{what} must have {n} entries, got {value!r}")
    if not th.all(th.isfinite(t)):
        raise TROValidationError(f"{what} contains non-finite values: {value!r}")
    return t


def _quat(value, what: str) -> th.Tensor:
    q = _vec(value, 4, what)
    norm = float(th.norm(q))
    if abs(norm - 1.0) > 1e-3:
        raise TROValidationError(f"{what} must be a unit quaternion [x, y, z, w], got norm {norm:.4f}")
    return quat_normalize(q)


def _tolerance(value, what: str) -> float:
    if value is None:
        raise TROValidationError(f"{what} is required")
    tol = float(value)
    if not tol > 0.0:
        raise TROValidationError(f"{what} must be > 0, got {value!r}")
    return tol


def _equivalences(value, what: str) -> List[th.Tensor]:
    if value is None:
        return []
    return [_quat(q, f"{what}[{i}]") for i, q in enumerate(value)]


def parse_frame(frame: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``"world"`` -> ``(None, None)``; ``"inst"`` -> ``("inst", None)``; ``"inst::link"`` -> ``("inst", "link")``."""
    if frame is None or frame == "world":
        return None, None
    if "::" in frame:
        inst, link = frame.split("::", 1)
        return inst, link
    return frame, None


# ----------------------------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class PoseTarget:
    instance: str
    link: Optional[str]
    frame: str
    position: Optional[th.Tensor]
    orientation: Optional[th.Tensor]
    position_metric: Optional[str]
    orientation_metric: Optional[str]
    position_tolerance: Optional[float]
    orientation_tolerance: Optional[float]
    orientation_equivalences: List[th.Tensor] = field(default_factory=list)

    @classmethod
    def from_dict(cls, instance: str, d: Dict[str, Any]) -> "PoseTarget":
        what = f"pose_targets[{instance!r}]"
        if not isinstance(d, dict):
            raise TROValidationError(f"{what} must be an object")
        unknown = set(d) - {
            "link",
            "frame",
            "position",
            "orientation",
            "position_metric",
            "orientation_metric",
            "position_tolerance",
            "orientation_tolerance",
            "orientation_equivalences",
        }
        if unknown:
            raise TROValidationError(f"{what} has unknown fields {sorted(unknown)}")

        position = _vec(d["position"], 3, f"{what}.position") if d.get("position") is not None else None
        orientation = _quat(d["orientation"], f"{what}.orientation") if d.get("orientation") is not None else None

        pos_metric = d.get("position_metric", "full" if position is not None else None)
        ori_metric = d.get("orientation_metric", "full" if orientation is not None else None)
        if pos_metric is not None and pos_metric not in POSITION_METRICS:
            raise TROValidationError(f"{what}.position_metric must be one of {POSITION_METRICS} or null")
        if ori_metric is not None and ori_metric not in ORIENTATION_METRICS:
            raise TROValidationError(f"{what}.orientation_metric must be one of {ORIENTATION_METRICS} or null")
        if pos_metric is None and ori_metric is None:
            raise TROValidationError(f"{what} must enable at least one of position_metric / orientation_metric")
        if pos_metric is not None and position is None:
            raise TROValidationError(f"{what}.position is required for position_metric={pos_metric!r}")
        if ori_metric is not None and orientation is None:
            raise TROValidationError(f"{what}.orientation is required for orientation_metric={ori_metric!r}")

        frame = d.get("frame") or "world"
        if pos_metric == "depth" and frame == "world":
            raise TROValidationError(f"{what}: position_metric='depth' requires a camera frame, not 'world'")
        pos_tol = _tolerance(d.get("position_tolerance"), f"{what}.position_tolerance") if pos_metric else None
        ori_tol = _tolerance(d.get("orientation_tolerance"), f"{what}.orientation_tolerance") if ori_metric else None
        link = d.get("link")
        if link is not None and (not isinstance(link, str) or not link):
            raise TROValidationError(f"{what}.link must be null or a non-empty string")
        return cls(
            instance=instance,
            link=link,
            frame=frame,
            position=position if pos_metric else None,
            orientation=orientation if ori_metric else None,
            position_metric=pos_metric,
            orientation_metric=ori_metric,
            position_tolerance=pos_tol,
            orientation_tolerance=ori_tol,
            orientation_equivalences=_equivalences(d.get("orientation_equivalences"), f"{what}.orientation_equivalences"),
        )

    @property
    def frame_instance(self) -> Optional[str]:
        return parse_frame(self.frame)[0]

    @property
    def frame_link(self) -> Optional[str]:
        return parse_frame(self.frame)[1]

    def pose_reached_kwargs(self, resolved_frame: str) -> Dict[str, Any]:
        """Keyword arguments for ``PoseReached.set_goal`` once ``frame`` has been resolved to a scene name."""
        return dict(
            goal_pos=self.position,
            goal_ori=self.orientation,
            pos_axes="depth" if self.position_metric == "depth" else None,
            ori_axes=None,
            link=self.link,
            frame=resolved_frame,
            pos_tolerance=self.position_tolerance,
            ori_tolerance=self.orientation_tolerance,
            orientation_equivalences=self.orientation_equivalences or None,
        )


@dataclass
class Slot:
    slot_id: str
    position: th.Tensor
    orientation: th.Tensor

    @property
    def pose(self):
        return self.position, self.orientation


@dataclass
class LayoutRelation:
    from_slot: str
    to_slot: str
    position: th.Tensor
    orientation: th.Tensor

    @property
    def pose(self):
        return self.position, self.orientation


@dataclass
class GroupTarget:
    name: str
    members: List[str]
    assignment_mode: str
    assignment_cost: str
    member_to_slot: Optional[Dict[str, str]]
    orientation_equivalences: List[th.Tensor]
    target_slots: Dict[str, Slot]
    pose_tolerances: Optional[Tuple[float, float]]
    layout_relations: List[LayoutRelation]
    layout_tolerances: Optional[Tuple[float, float]]
    pinned: Dict[str, str] = field(default_factory=dict)  # exchangeable mode: members held to one slot, the others matched

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "GroupTarget":
        what = f"groups[{name!r}]"
        if not isinstance(d, dict):
            raise TROValidationError(f"{what} must be an object")
        unknown = set(d) - {
            "members",
            "assignment",
            "orientation_equivalences",
            "target_slots",
            "pose_tolerances",
            "layout_relations",
            "layout_tolerances",
        }
        if unknown:
            raise TROValidationError(f"{what} has unknown fields {sorted(unknown)}")
        members = list(d.get("members") or [])
        if not members or len(set(members)) != len(members):
            raise TROValidationError(f"{what}.members must be a non-empty list of distinct instances")

        slots_raw = d.get("target_slots") or {}
        slots = {
            sid: Slot(sid, _vec(s.get("position"), 3, f"{what}.target_slots[{sid}].position"),
                      _quat(s.get("orientation"), f"{what}.target_slots[{sid}].orientation"))
            for sid, s in slots_raw.items()
        }
        if slots and len(slots) != len(members):
            raise TROValidationError(f"{what}: target_slots must have exactly one slot per member")

        assignment = d.get("assignment") or {"mode": "exchangeable"}
        mode = assignment.get("mode", "exchangeable")
        if mode not in ASSIGNMENT_MODES:
            raise TROValidationError(f"{what}.assignment.mode must be one of {ASSIGNMENT_MODES}")
        member_to_slot = None
        if mode == "fixed":
            member_to_slot = dict(assignment.get("member_to_slot") or {})
            if set(member_to_slot) != set(members):
                raise TROValidationError(f"{what}.assignment.member_to_slot must map every member exactly once")
            if slots and (set(member_to_slot.values()) != set(slots) or len(set(member_to_slot.values())) != len(slots)):
                raise TROValidationError(f"{what}.assignment.member_to_slot must be a bijection onto target_slots")
        pinned = dict(assignment.get("pinned") or {})
        if pinned:
            if mode != "exchangeable":
                raise TROValidationError(f"{what}.assignment.pinned is for exchangeable groups (a fixed group maps every member)")
            if not set(pinned) <= set(members):
                raise TROValidationError(f"{what}.assignment.pinned names non-members {sorted(set(pinned) - set(members))}")
            if not set(pinned.values()) <= set(slots) or len(set(pinned.values())) != len(pinned):
                raise TROValidationError(f"{what}.assignment.pinned must map its members to distinct target_slots")

        relations_raw = d.get("layout_relations") or []
        relations = []
        for i, r in enumerate(relations_raw):
            rw = f"{what}.layout_relations[{i}]"
            if r.get("from") not in slots or r.get("to") not in slots:
                raise TROValidationError(f"{rw} references unknown slot(s) {r.get('from')!r} -> {r.get('to')!r}")
            if r["from"] == r["to"]:
                raise TROValidationError(f"{rw} relates a slot to itself")
            rel = LayoutRelation(r["from"], r["to"], _vec(r.get("position"), 3, f"{rw}.position"),
                                 _quat(r.get("orientation"), f"{rw}.orientation"))
            # Each relation must equal T_from^-1 * T_to of the slot poses.
            p_expect, q_expect = relative_pose(slots[rel.from_slot].pose, slots[rel.to_slot].pose)
            if float(th.norm(p_expect - rel.position)) > RELATION_POSITION_ATOL or quat_angle(q_expect, rel.orientation) > RELATION_ORIENTATION_ATOL:
                raise TROValidationError(f"{rw} is inconsistent with target_slots (expected T_from^-1 * T_to)")
            relations.append(rel)
        if len({(r.from_slot, r.to_slot) for r in relations}) != len(relations):
            raise TROValidationError(f"{what}.layout_relations contains duplicate (from, to) pairs")

        cost = assignment.get("cost", "layout" if relations else "slot_position")
        if cost not in ASSIGNMENT_COSTS:
            raise TROValidationError(f"{what}.assignment.cost must be one of {ASSIGNMENT_COSTS}")
        if cost == "layout" and not relations:
            raise TROValidationError(f"{what}.assignment.cost='layout' needs layout_relations")
        if cost == "slot_position" and not slots:
            raise TROValidationError(f"{what}.assignment.cost='slot_position' needs target_slots")

        def tol_pair(key):
            v = d.get(key)
            if v is None:
                return None
            return (_tolerance(v.get("position"), f"{what}.{key}.position"),
                    _tolerance(v.get("orientation"), f"{what}.{key}.orientation"))

        return cls(
            name=name,
            members=members,
            assignment_mode=mode,
            assignment_cost=cost,
            member_to_slot=member_to_slot,
            orientation_equivalences=_equivalences(d.get("orientation_equivalences"), f"{what}.orientation_equivalences"),
            target_slots=slots,
            pose_tolerances=tol_pair("pose_tolerances"),
            layout_relations=relations,
            layout_tolerances=tol_pair("layout_tolerances"),
            pinned=pinned,
        )

    @property
    def supports_pose_reached(self) -> bool:
        return bool(self.target_slots) and self.pose_tolerances is not None

    @property
    def supports_layout_reached(self) -> bool:
        return bool(self.layout_relations) and self.layout_tolerances is not None


@dataclass
class ArticulationTarget:
    instance: str
    joints: Dict[str, Tuple[float, float]]  # joint name -> (position, tolerance)

    @classmethod
    def from_dict(cls, instance: str, d: Dict[str, Any]) -> "ArticulationTarget":
        what = f"articulation_targets[{instance!r}]"
        joints_raw = (d or {}).get("joints") or {}
        if not joints_raw:
            raise TROValidationError(f"{what}.joints must list at least one joint")
        joints = {}
        for name, spec in joints_raw.items():
            if spec.get("position") is None:
                raise TROValidationError(f"{what}.joints[{name!r}].position is required")
            joints[name] = (float(spec["position"]), _tolerance(spec.get("tolerance"), f"{what}.joints[{name!r}].tolerance"))
        return cls(instance=instance, joints=joints)


# ----------------------------------------------------------------------------------------------------------------
# Whole metadata block
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ISGTroMeta:
    schema_version: str
    activity: str
    activity_definition_id: int
    task_variant: Optional[str]
    instance_id: Optional[str]
    scene: Dict[str, Any]
    object_bindings: Dict[str, Dict[str, Any]]
    pose_targets: Dict[str, PoseTarget]
    groups: Dict[str, GroupTarget]
    articulation_targets: Dict[str, ArticulationTarget]
    robot_state: Optional[Dict[str, Any]] = None
    goal_observation: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    KNOWN_FIELDS = {
        "schema_version",
        "activity",
        "activity_definition_id",
        "task_variant",
        "instance_id",
        "generation_seed",
        "evaluation_profile",
        "bddl_sha256",
        "task_spec_sha256",
        "layout_template_id",
        "scene",
        "object_bindings",
        "robot_state",
        "goal_observation",
        # 2026-11-15 batched replay (scripts/batched_replay/export_eval.py): the collection's goal record, kept beside the replay's
        # goal_observation for provenance; informational, never read
        "goal_observation_collection",
        "pose_targets",
        "groups",
        "articulation_targets",
        "grasps",
        "attachment_targets",
        "constraints",
        "initial_occlusion",
        "campaign",
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ISGTroMeta":
        if not isinstance(d, dict):
            raise TROValidationError(f"'{TRO_KEY}' block must be an object")
        if d.get("schema_version") != SCHEMA_VERSION:
            raise TROValidationError(f"schema_version must be {SCHEMA_VERSION!r}, got {d.get('schema_version')!r}")
        unknown = set(d) - cls.KNOWN_FIELDS
        if unknown:
            raise TROValidationError(f"Unknown '{TRO_KEY}' fields {sorted(unknown)}")
        if "campaign" in d:
            from scripts.campaign_20261110.contracts import validate_campaign_block

            try:
                validate_campaign_block(d["campaign"], object_bindings=d.get("object_bindings"))
            except (ValueError, TypeError, KeyError) as error:
                raise TROValidationError(f"Invalid campaign metadata: {error}") from error
        if not isinstance(d.get("activity"), str) or not d["activity"]:
            raise TROValidationError("activity is required")
        try:
            definition_id = int(d["activity_definition_id"])
        except (KeyError, TypeError, ValueError):
            raise TROValidationError("activity_definition_id must be an integer")
        pose_targets = {inst: PoseTarget.from_dict(inst, spec) for inst, spec in (d.get("pose_targets") or {}).items()}
        groups = {name: GroupTarget.from_dict(name, spec) for name, spec in (d.get("groups") or {}).items()}
        articulation = {
            inst: ArticulationTarget.from_dict(inst, spec) for inst, spec in (d.get("articulation_targets") or {}).items()
        }
        both = set(pose_targets) & set(groups)
        if both:
            raise TROValidationError(f"{sorted(both)} appear in both pose_targets and groups")
        extras = {k: d[k] for k in ("grasps", "attachment_targets", "constraints", "initial_occlusion",
                                    "generation_seed", "evaluation_profile", "bddl_sha256", "task_spec_sha256",
                                    "layout_template_id") if k in d}
        return cls(
            schema_version=d["schema_version"],
            activity=d["activity"],
            activity_definition_id=definition_id,
            task_variant=canonical_variant(d.get("task_variant")),
            instance_id=canonical_variant(d.get("instance_id")),
            scene=dict(d.get("scene") or {}),
            object_bindings=dict(d.get("object_bindings") or {}),
            pose_targets=pose_targets,
            groups=groups,
            articulation_targets=articulation,
            robot_state=d.get("robot_state"),
            goal_observation=d.get("goal_observation"),
            extras=extras,
            raw=d,
        )

    @property
    def name(self) -> str:
        return f"{self.activity}-{self.activity_definition_id}"

    def measured_instances(self) -> List[str]:
        """Instances / groups whose pose or layout errors the evaluator reports."""
        return list(self.pose_targets) + list(self.groups) + list(self.articulation_targets)
