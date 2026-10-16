"""Evaluation targets of the reserved ``ispatialgym`` block of a TRO file (schema ``isg-tro-v1``).

Parsing and validation only; nothing here touches the simulator.

Pose target (``pose_targets[<instance>]``)::

    {
      "link": null | "<link or sensor key of the instance>",
      "frame": "world" | "<instance>" | "<instance>::<link or sensor key>",
      "position": [x, y, z] | null,
      "orientation": [x, y, z, w] | null,
      "position_metric": "full" | "depth" | null,      # default "full" when position is given
      "orientation_metric": "full" | null,             # default "full" when orientation is given
      "position_tolerance": meters | null,             # required with a position metric
      "orientation_tolerance": radians | null,         # required with an orientation metric
      "orientation_equivalences": [[x, y, z, w], ...]  # object-frame symmetries
    }

``position_metric = "depth"`` compares the depth along the frame's optical axis only (a camera frame, -Z forward).
Group targets (``groups``) hold one slot per member and/or the relative poses between slots; articulation targets
(``articulation_targets``) hold joint positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch as th

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
    """A malformed or inconsistent ``ispatialgym`` TRO block."""


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


def _equivalences(value, what: str) -> list[th.Tensor]:
    return [_quat(q, f"{what}[{i}]") for i, q in enumerate(value or [])]


def parse_frame(frame: str | None) -> tuple[str | None, str | None]:
    """``"world"`` -> ``(None, None)``; ``"inst"`` -> ``("inst", None)``; ``"inst::link"`` -> ``("inst", "link")``."""
    if frame is None or frame == "world":
        return None, None
    if "::" in frame:
        inst, link = frame.split("::", 1)
        return inst, link
    return frame, None


@dataclass
class PoseTarget:
    instance: str
    link: str | None
    frame: str
    position: th.Tensor | None
    orientation: th.Tensor | None
    position_metric: str | None
    orientation_metric: str | None
    position_tolerance: float | None
    orientation_tolerance: float | None
    orientation_equivalences: list[th.Tensor] = field(default_factory=list)

    FIELDS = frozenset(
        {
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
    )

    @classmethod
    def from_dict(cls, instance: str, d: dict[str, Any]) -> PoseTarget:
        what = f"pose_targets[{instance!r}]"
        if not isinstance(d, dict):
            raise TROValidationError(f"{what} must be an object")
        unknown = set(d) - cls.FIELDS
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
            position_tolerance=_tolerance(d.get("position_tolerance"), f"{what}.position_tolerance")
            if pos_metric
            else None,
            orientation_tolerance=(
                _tolerance(d.get("orientation_tolerance"), f"{what}.orientation_tolerance") if ori_metric else None
            ),
            orientation_equivalences=_equivalences(
                d.get("orientation_equivalences"), f"{what}.orientation_equivalences"
            ),
        )

    def pose_reached_kwargs(self, resolved_frame: str) -> dict[str, Any]:
        """Keyword arguments of ``PoseReached.set_goal`` once ``frame`` is resolved to a scene name."""
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
    members: list[str]
    assignment_mode: str
    assignment_cost: str
    member_to_slot: dict[str, str] | None
    orientation_equivalences: list[th.Tensor]
    target_slots: dict[str, Slot]
    pose_tolerances: tuple[float, float] | None
    layout_relations: list[LayoutRelation]
    layout_tolerances: tuple[float, float] | None
    pinned: dict[str, str] = field(default_factory=dict)  # exchangeable groups: members held to one slot

    FIELDS = frozenset(
        {
            "members",
            "assignment",
            "orientation_equivalences",
            "target_slots",
            "pose_tolerances",
            "layout_relations",
            "layout_tolerances",
        }
    )

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> GroupTarget:
        what = f"groups[{name!r}]"
        if not isinstance(d, dict):
            raise TROValidationError(f"{what} must be an object")
        unknown = set(d) - cls.FIELDS
        if unknown:
            raise TROValidationError(f"{what} has unknown fields {sorted(unknown)}")
        members = list(d.get("members") or [])
        if not members or len(set(members)) != len(members):
            raise TROValidationError(f"{what}.members must be a non-empty list of distinct instances")

        slots = {
            sid: Slot(
                sid,
                _vec(s.get("position"), 3, f"{what}.target_slots[{sid}].position"),
                _quat(s.get("orientation"), f"{what}.target_slots[{sid}].orientation"),
            )
            for sid, s in (d.get("target_slots") or {}).items()
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
            targets = set(member_to_slot.values())
            if slots and (targets != set(slots) or len(targets) != len(slots)):
                raise TROValidationError(f"{what}.assignment.member_to_slot must be a bijection onto target_slots")
        pinned = dict(assignment.get("pinned") or {})
        if pinned:
            if mode != "exchangeable":
                raise TROValidationError(f"{what}.assignment.pinned is only valid for exchangeable groups")
            if not set(pinned) <= set(members):
                raise TROValidationError(
                    f"{what}.assignment.pinned names non-members {sorted(set(pinned) - set(members))}"
                )
            if not set(pinned.values()) <= set(slots) or len(set(pinned.values())) != len(pinned):
                raise TROValidationError(f"{what}.assignment.pinned must map its members to distinct target_slots")

        relations = []
        for i, r in enumerate(d.get("layout_relations") or []):
            rw = f"{what}.layout_relations[{i}]"
            if r.get("from") not in slots or r.get("to") not in slots:
                raise TROValidationError(f"{rw} references unknown slot(s) {r.get('from')!r} -> {r.get('to')!r}")
            if r["from"] == r["to"]:
                raise TROValidationError(f"{rw} relates a slot to itself")
            rel = LayoutRelation(
                r["from"],
                r["to"],
                _vec(r.get("position"), 3, f"{rw}.position"),
                _quat(r.get("orientation"), f"{rw}.orientation"),
            )
            p_expect, q_expect = relative_pose(slots[rel.from_slot].pose, slots[rel.to_slot].pose)
            if (
                float(th.norm(p_expect - rel.position)) > RELATION_POSITION_ATOL
                or quat_angle(q_expect, rel.orientation) > RELATION_ORIENTATION_ATOL
            ):
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
            return (
                _tolerance(v.get("position"), f"{what}.{key}.position"),
                _tolerance(v.get("orientation"), f"{what}.{key}.orientation"),
            )

        return cls(
            name=name,
            members=members,
            assignment_mode=mode,
            assignment_cost=cost,
            member_to_slot=member_to_slot,
            orientation_equivalences=_equivalences(
                d.get("orientation_equivalences"), f"{what}.orientation_equivalences"
            ),
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
    joints: dict[str, tuple[float, float]]  # joint name -> (position, tolerance)

    @classmethod
    def from_dict(cls, instance: str, d: dict[str, Any]) -> ArticulationTarget:
        what = f"articulation_targets[{instance!r}]"
        joints_raw = (d or {}).get("joints") or {}
        if not joints_raw:
            raise TROValidationError(f"{what}.joints must list at least one joint")
        joints = {}
        for name, spec in joints_raw.items():
            if spec.get("position") is None:
                raise TROValidationError(f"{what}.joints[{name!r}].position is required")
            joints[name] = (
                float(spec["position"]),
                _tolerance(spec.get("tolerance"), f"{what}.joints[{name!r}].tolerance"),
            )
        return cls(instance=instance, joints=joints)


@dataclass
class ISGTroMeta:
    schema_version: str
    activity: str
    activity_definition_id: int
    task_variant: str | None
    instance_id: str | None
    scene: dict[str, Any]
    object_bindings: dict[str, dict[str, Any]]
    pose_targets: dict[str, PoseTarget]
    groups: dict[str, GroupTarget]
    articulation_targets: dict[str, ArticulationTarget]
    robot_state: dict[str, Any] | None = None
    goal_observation: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    KNOWN_FIELDS = frozenset(
        {
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
            "pose_targets",
            "groups",
            "articulation_targets",
            "grasps",
            "attachment_targets",
            "constraints",
            "initial_occlusion",
        }
    )
    EXTRA_FIELDS = ("grasps", "attachment_targets", "constraints", "evaluation_profile")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ISGTroMeta:
        if not isinstance(d, dict):
            raise TROValidationError(f"'{TRO_KEY}' block must be an object")
        if d.get("schema_version") != SCHEMA_VERSION:
            raise TROValidationError(f"schema_version must be {SCHEMA_VERSION!r}, got {d.get('schema_version')!r}")
        unknown = set(d) - cls.KNOWN_FIELDS
        if unknown:
            raise TROValidationError(f"Unknown '{TRO_KEY}' fields {sorted(unknown)}")
        if not isinstance(d.get("activity"), str) or not d["activity"]:
            raise TROValidationError("activity is required")
        try:
            definition_id = int(d["activity_definition_id"])
        except (KeyError, TypeError, ValueError):
            raise TROValidationError("activity_definition_id must be an integer") from None
        pose_targets = {inst: PoseTarget.from_dict(inst, spec) for inst, spec in (d.get("pose_targets") or {}).items()}
        groups = {name: GroupTarget.from_dict(name, spec) for name, spec in (d.get("groups") or {}).items()}
        articulation = {
            inst: ArticulationTarget.from_dict(inst, spec)
            for inst, spec in (d.get("articulation_targets") or {}).items()
        }
        both = set(pose_targets) & set(groups)
        if both:
            raise TROValidationError(f"{sorted(both)} appear in both pose_targets and groups")
        return cls(
            schema_version=d["schema_version"],
            activity=d["activity"],
            activity_definition_id=definition_id,
            task_variant=d.get("task_variant"),
            instance_id=d.get("instance_id"),
            scene=dict(d.get("scene") or {}),
            object_bindings=dict(d.get("object_bindings") or {}),
            pose_targets=pose_targets,
            groups=groups,
            articulation_targets=articulation,
            robot_state=d.get("robot_state"),
            goal_observation=d.get("goal_observation"),
            extras={k: d[k] for k in cls.EXTRA_FIELDS if k in d},
        )

    @property
    def name(self) -> str:
        return f"{self.activity}-{self.activity_definition_id}"
