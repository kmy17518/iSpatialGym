"""Oracle sidecar (``<instance_id>.oracle.json``, schema ``isg-oracle-v1``).

The sidecar carries the **facts** the sampler recorded for the oracle (``instance_sampling_spec.md`` §3.11, §7):
stances, transitions, paths, grasps, contacts, articulation waypoints, block data, the camera-relocalization goal and
the language descriptors. It never carries the action sequence — that is the per-subtask program of
:mod:`ispatialgym.oracle.programs` (``oracle_primitives_design.md`` §3.7, §6.2).

This module is pure Python (no simulator, no torch at import time): dataclasses mirroring spec §7, JSON I/O,
structural + semantic validation, the TRO sha256 check and the alias → subtask resolution from the iddl alias
templates (``ispatialgym/assets/iddl/*/*.iddl``).

Conventions: poses are ``{"position": [x, y, z], "orientation": [x, y, z, w]}`` (quaternion **xyzw**) in the world
frame unless a field says otherwise; base poses are ``[x, y, yaw]``; trunk values are the JoyLo ``translate`` /
``tilt`` scalars; angles are radians except the block fields whose names end in ``_deg`` (spec §7 records those in
degrees).

Fields that spec §7 leaves unspecified but a program needs are documented on the dataclass that carries them
(``GraspRecord.waypoints`` / ``place_on``, ``PushContact.object`` / ``BraceContact.object``, ``Articulation.object`` /
``joint_type``, ``BlockRecord.carry_height`` / ``push_row`` / ``push_col``); see the F6 report. Unknown keys of any
block round-trip through ``extra`` so samplers may record diagnostics without a schema change.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ispatialgym import ASSETS_ROOT

SCHEMA = "isg-oracle-v1"
ARMS = ("left", "right")
#: ``hybrid`` (sampler ``articulation_hybrid``, D-011 C 2026-09-17): ``base_follow`` as in base mode plus the dense
#: ``base_follow_path.poses[]`` (``q``, ``base_pose2d``, ``arm_q``) the executor drives + tracks together
FOLLOW_MODES = ("base", "arm", "hybrid")
PATH_MODES = ("local", "global")
JOINT_TYPES = ("revolute", "prismatic")
FACE_PAIRS = ("x", "y", "z")
GOAL_ARMS = ("same", "untucked")
TRANSITION_ORDER = ["drive-back", "trunk", "drive-in"]
STANCE_NAMES = ("init", "manipulation", "placement")

# Blocks each subtask's program needs (spec §6 per-task rules, §7 "only the blocks a task uses are present").
# ``language`` is required everywhere. Connectivity between consecutive stances (a path or a transition) is checked by
# the program itself when the sidecar is expanded (``Sidecar.validate`` runs the expansion).
_TABLE = ("stances.init", "stances.manipulation")
REQUIRED_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "camera_relocalization-standard": ("stances.init", "goal", "paths", "path_bucket_m"),
    "camera_relocalization-obstructed": ("stances.init", "goal", "paths", "path_bucket_m"),
    "object_scaling": ("grasps",),
    "mental_rotation": ("grasps",),
    "disembedding": _TABLE + ("stances.placement", "grasps"),
    "configuration_matching-block_design": _TABLE + ("blocks",),
    "configuration_matching-articulation_open_large_scale": _TABLE + ("articulation",),
    "configuration_matching-articulation_close_large_scale": _TABLE + ("articulation",),
    # small articulation = brace optional (owner D-008 = A, 2026-09-16 22:02): ``contacts.brace`` is recorded only when a
    # joint two-arm cell exists, otherwise the handle-only program (grasp -> articulate -> release -> retreat)
    "configuration_matching-articulation_open_small_scale": _TABLE + ("articulation",),
    "configuration_matching-articulation_close_small_scale": _TABLE + ("articulation",),
    # attachment keeps the brace (owner 2026-09-16 17:10 item 2): the sampler records it from the grasp stance or from a
    # second base pose (``stances.attach`` + a carried path / holding transition); only when the joint placement has no
    # cell after that is the instance brace-less (``summary.brace: none``), so ``contacts.brace`` is optional here
    "configuration_matching-attachment": _TABLE + ("grasps",),
    "alignment-world_vertical": _TABLE + ("contacts.push", "contacts.brace"),
    "alignment-world_horizontal": _TABLE + ("contacts.push", "contacts.brace"),
    # axial is push-only (owner decision 2026-09-16 08:10): ``contacts.brace`` is recorded only when the joint two-arm
    # placement exists and is optional here; the oracle's push closure is the stop otherwise
    "alignment-axial": _TABLE + ("contacts.push",),
    "inter_object_relation_satisfaction-ontop_large_scale": _TABLE + ("stances.placement", "grasps", "paths"),
    "inter_object_relation_satisfaction-under_large_scale": _TABLE + ("stances.placement", "grasps", "paths"),
    "inter_object_relation_satisfaction-ontop_small_scale": _TABLE + ("grasps",),
    "inter_object_relation_satisfaction-under_small_scale": _TABLE + ("grasps",),
    "inter_object_relation_satisfaction-inside": _TABLE + ("grasps",),
    "inter_object_relation_satisfaction-nextto": _TABLE + ("grasps",),
    # touching is brace-optional like axial (owner decision 2026-09-16 17:10; core ``6de8803``): ``contacts.brace`` is
    # recorded only when the joint two-arm placement exists, the push chain is required, the push closure is the stop
    "inter_object_relation_satisfaction-touching": _TABLE + ("grasps", "contacts.push"),
}
SUBTASKS: Tuple[str, ...] = tuple(REQUIRED_BLOCKS)


class SidecarError(ValueError):
    """A sidecar that cannot be parsed, fails validation, or lacks a fact a program needs."""


# ----------------------------------------------------------------------------------------------------------------
# Typed readers (every error names the JSON path)
# ----------------------------------------------------------------------------------------------------------------


def _num(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SidecarError(f"{where}: expected a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise SidecarError(f"{where}: not finite ({v!r})")
    return f


def _vec(v: Any, n: int, where: str) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise SidecarError(f"{where}: expected {n} numbers, got {v!r}")
    return [_num(x, f"{where}[{i}]") for i, x in enumerate(v)]


def _str(v: Any, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise SidecarError(f"{where}: expected a non-empty string, got {v!r}")
    return v


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise SidecarError(f"{where}: expected a boolean, got {v!r}")
    return v


def _dict(v: Any, where: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise SidecarError(f"{where}: expected an object, got {type(v).__name__}")
    return v


def _list(v: Any, where: str) -> List[Any]:
    if not isinstance(v, list):
        raise SidecarError(f"{where}: expected a list, got {type(v).__name__}")
    return v


_MISSING = object()


def _get(d: Dict[str, Any], key: str, where: str, default: Any = _MISSING) -> Any:
    if key in d:
        return d[key]
    if default is _MISSING:
        raise SidecarError(f"{where}: missing required field '{key}'")
    return default


def _extra(d: Dict[str, Any], known: Tuple[str, ...]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in known}


def _opt(v: Any, fn, where: str):
    return None if v is None else fn(v, where)


# ----------------------------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Pose:
    """A world-frame pose: ``position`` (x, y, z) and ``orientation`` (quaternion x, y, z, w)."""

    position: List[float]
    orientation: List[float]

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Pose":
        d = _dict(d, where)
        pos = _vec(_get(d, "position", where), 3, f"{where}.position")
        quat = _vec(_get(d, "orientation", where), 4, f"{where}.orientation")
        norm = math.sqrt(sum(q * q for q in quat))
        if abs(norm - 1.0) > 1e-3:
            raise SidecarError(f"{where}.orientation: not a unit quaternion (norm {norm:.4f})")
        return cls(pos, quat)

    def to_dict(self) -> Dict[str, Any]:
        return {"position": list(self.position), "orientation": list(self.orientation)}

    def tensors(self):
        """``(pos (3,), quat_xyzw (4,))`` float32 CPU tensors (torch imported lazily)."""
        import torch as th

        return th.tensor(self.position, dtype=th.float32), th.tensor(self.orientation, dtype=th.float32)

    def approach_axis(self) -> List[float]:
        """The eef approach axis a = R·ẑ (eef +z) in the world frame (spec §3.6)."""
        x, y, z, w = self.orientation
        return [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]

    def translated(self, delta: List[float]) -> "Pose":
        return Pose([p + d for p, d in zip(self.position, delta)], list(self.orientation))


def _pose(v: Any, where: str) -> Pose:
    return Pose.from_dict(v, where)


def _poses(v: Any, where: str) -> List[Pose]:
    return [Pose.from_dict(p, f"{where}[{i}]") for i, p in enumerate(_list(v, where))]


def _pose2d(v: Any, where: str) -> List[float]:
    return _vec(v, 3, where)


@dataclass
class Stance:
    """One recorded robot stance (spec §7 ``stances.<name>``; §4.5, §4.7, §5.2, §5.6).

    ``arms``: ``"untucked"`` (both arms idle) or the list of working arms (``["right"]``); ``stand_off`` / ``edge`` /
    ``lateral`` / ``focus`` are the sampler's stance parameters; anything else the sampler records (``bearing_deg``,
    ``distance``, ``goal_cell``, ``deviation_deg``, ...) is kept in ``extra``.
    """

    base_pose2d: List[float]
    translate: float
    tilt: float
    arms: Union[str, List[str], None] = None
    stand_off: Optional[float] = None
    edge: Optional[str] = None
    lateral: Optional[float] = None
    focus: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("base_pose2d", "translate", "tilt", "arms", "stand_off", "edge", "lateral", "focus")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Stance":
        d = _dict(d, where)
        arms = d.get("arms")
        if arms is not None:
            if isinstance(arms, str):
                _str(arms, f"{where}.arms")
            else:
                arms = [_str(a, f"{where}.arms[{i}]") for i, a in enumerate(_list(arms, f"{where}.arms"))]
        return cls(
            base_pose2d=_pose2d(_get(d, "base_pose2d", where), f"{where}.base_pose2d"),
            translate=_num(_get(d, "translate", where), f"{where}.translate"),
            tilt=_num(_get(d, "tilt", where), f"{where}.tilt"),
            arms=arms,
            stand_off=_opt(d.get("stand_off"), _num, f"{where}.stand_off"),
            edge=_opt(d.get("edge"), _str, f"{where}.edge"),
            lateral=_opt(d.get("lateral"), _num, f"{where}.lateral"),
            focus=_opt(d.get("focus"), lambda v, w: _vec(v, 3, w), f"{where}.focus"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"base_pose2d": list(self.base_pose2d), "translate": self.translate, "tilt": self.tilt}
        if self.arms is not None:
            out["arms"] = self.arms if isinstance(self.arms, str) else list(self.arms)
        for k in ("stand_off", "edge", "lateral", "focus"):
            v = getattr(self, k)
            if v is not None:
                out[k] = list(v) if isinstance(v, list) else v
        out.update(self.extra)
        return out

    @property
    def trunk(self) -> Tuple[float, float]:
        return (self.translate, self.tilt)

    @property
    def working_arms(self) -> List[str]:
        """The arms this stance was solved for (empty when ``arms`` is ``"untucked"`` / unset)."""
        return list(self.arms) if isinstance(self.arms, list) else []


@dataclass
class Transition:
    """A stance transition (spec §3.8): drive back to ``stand_off`` (s_T, along −n from the arriving stance), move
    the trunk, drive in.

    Beyond spec §7: ``holding`` (in ``extra``; core ``bd10c72`` / tabletop ``58afae2``) — the bddl instance carried
    through the transition (attachment's second base pose between the grasp and the attach, spec §6.8: the sampler swept
    the drive-back / trunk / drive-in with the male part attached to the frozen arm). A program that travels this
    transition must carry exactly that object (``programs._travel``)."""

    frm: str
    to: str
    stand_off: float
    order: List[str] = field(default_factory=lambda: list(TRANSITION_ORDER))
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("from", "to", "stand_off", "order")

    @property
    def holding(self) -> Optional[str]:
        """The bddl instance carried through the transition (``None`` = nothing recorded)."""
        v = self.extra.get("holding")
        return v if isinstance(v, str) and v else None

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Transition":
        d = _dict(d, where)
        order = d.get("order", list(TRANSITION_ORDER))
        return cls(
            frm=_str(_get(d, "from", where), f"{where}.from"),
            to=_str(_get(d, "to", where), f"{where}.to"),
            stand_off=_num(_get(d, "stand_off", where), f"{where}.stand_off"),
            order=[_str(s, f"{where}.order[{i}]") for i, s in enumerate(_list(order, f"{where}.order"))],
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {"from": self.frm, "to": self.to, "stand_off": self.stand_off, "order": list(self.order)}
        out.update(self.extra)
        return out


@dataclass
class Path:
    """A recorded base path (spec §3.7): ``waypoints`` = ``[x, y, yaw]`` list the tracker follows exactly, validated
    at ``posture`` with ``carried`` attached (``None`` = nothing carried)."""

    frm: str
    to: str
    mode: str
    waypoints: List[List[float]]
    length_m: float
    posture: str
    carried: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("from", "to", "mode", "waypoints", "length_m", "posture", "carried")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Path":
        d = _dict(d, where)
        wps = [_pose2d(w, f"{where}.waypoints[{i}]") for i, w in enumerate(_list(_get(d, "waypoints", where), f"{where}.waypoints"))]
        return cls(
            frm=_str(_get(d, "from", where), f"{where}.from"),
            to=_str(_get(d, "to", where), f"{where}.to"),
            mode=_str(_get(d, "mode", where), f"{where}.mode"),
            waypoints=wps,
            length_m=_num(_get(d, "length_m", where), f"{where}.length_m"),
            posture=_str(_get(d, "posture", where), f"{where}.posture"),
            carried=_opt(d.get("carried"), _str, f"{where}.carried"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "from": self.frm,
            "to": self.to,
            "mode": self.mode,
            "waypoints": [list(w) for w in self.waypoints],
            "length_m": self.length_m,
            "posture": self.posture,
            "carried": self.carried,
        }
        out.update(self.extra)
        return out

    @property
    def goal(self) -> List[float]:
        return list(self.waypoints[-1])


@dataclass
class GraspRecord:
    """A library grasp the oracle executes (spec §7 ``grasps[]``).

    ``grasp`` / ``pre_grasp`` are the eef poses at the object's init pose (the oracle re-derives them from the
    current object pose with the same ``(grasp_key, grasp_name, yaw)`` through the conventions); ``pre_place`` /
    ``place`` the eef poses at the goal; ``carry_height`` the lift after the grasp.

    Beyond spec §7 (needed by the programs, see the F6 report): ``waypoints`` — the recorded eef key poses of an
    in-hand motion in order, last = the end pose (held-object tasks: 3 intermediate + goal, spec §6.3 / §3.9;
    attachment: the SLERP waypoints from the lifted grasp to the pre-attach pose, spec §6.8); ``place_on`` — the
    bddl instance the object is placed on / into / attached to (annotation and the intended-contact whitelist).
    """

    object: str
    arm: str
    grasp_key: str
    grasp_name: str
    yaw: float
    grasp: Pose
    pre_grasp: Optional[Pose] = None
    pre_place: Optional[Pose] = None
    place: Optional[Pose] = None
    carry_height: Optional[float] = None
    waypoints: Optional[List[Pose]] = None
    place_on: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("object", "arm", "grasp_key", "grasp_name", "yaw", "grasp", "pre_grasp", "pre_place", "place", "carry_height", "waypoints", "place_on")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "GraspRecord":
        d = _dict(d, where)
        return cls(
            object=_str(_get(d, "object", where), f"{where}.object"),
            arm=_str(_get(d, "arm", where), f"{where}.arm"),
            grasp_key=_str(_get(d, "grasp_key", where), f"{where}.grasp_key"),
            grasp_name=_str(_get(d, "grasp_name", where), f"{where}.grasp_name"),
            yaw=_num(_get(d, "yaw", where), f"{where}.yaw"),
            grasp=_pose(_get(d, "grasp", where), f"{where}.grasp"),
            pre_grasp=_opt(d.get("pre_grasp"), _pose, f"{where}.pre_grasp"),
            pre_place=_opt(d.get("pre_place"), _pose, f"{where}.pre_place"),
            place=_opt(d.get("place"), _pose, f"{where}.place"),
            carry_height=_opt(d.get("carry_height"), _num, f"{where}.carry_height"),
            waypoints=_opt(d.get("waypoints"), _poses, f"{where}.waypoints"),
            place_on=_opt(d.get("place_on"), _str, f"{where}.place_on"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "object": self.object,
            "arm": self.arm,
            "grasp_key": self.grasp_key,
            "grasp_name": self.grasp_name,
            "yaw": self.yaw,
            "grasp": self.grasp.to_dict(),
        }
        for k in ("pre_grasp", "pre_place", "place"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v.to_dict()
        if self.carry_height is not None:
            out["carry_height"] = self.carry_height
        if self.waypoints is not None:
            out["waypoints"] = [p.to_dict() for p in self.waypoints]
        if self.place_on is not None:
            out["place_on"] = self.place_on
        out.update(self.extra)
        return out


@dataclass
class PushContact:
    """``contacts.push`` (spec §7): closed gripper at ``init`` (pad centre on the contact, eef origin =
    point − CONTACT_CENTER_Z·a), moved through ``mid`` (arc waypoint at δ/2, absent for a straight push) to ``goal``;
    ``direction`` is the unit push direction at ``init``, ``distance_m`` the push length. ``object`` (beyond §7) names
    the pushed bddl instance (annotation, intended-contact whitelist)."""

    arm: str
    object: str
    pre: Pose
    init: Pose
    goal: Pose
    direction: List[float]
    distance_m: float
    mid: Optional[Pose] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("arm", "object", "pre", "init", "mid", "goal", "direction", "distance_m")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "PushContact":
        d = _dict(d, where)
        return cls(
            arm=_str(_get(d, "arm", where), f"{where}.arm"),
            object=_str(_get(d, "object", where), f"{where}.object"),
            pre=_pose(_get(d, "pre", where), f"{where}.pre"),
            init=_pose(_get(d, "init", where), f"{where}.init"),
            goal=_pose(_get(d, "goal", where), f"{where}.goal"),
            direction=_vec(_get(d, "direction", where), 3, f"{where}.direction"),
            distance_m=_num(_get(d, "distance_m", where), f"{where}.distance_m"),
            mid=_opt(d.get("mid"), _pose, f"{where}.mid"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"arm": self.arm, "object": self.object, "pre": self.pre.to_dict(), "init": self.init.to_dict()}
        if self.mid is not None:
            out["mid"] = self.mid.to_dict()
        out.update({"goal": self.goal.to_dict(), "direction": list(self.direction), "distance_m": self.distance_m})
        out.update(self.extra)
        return out

    @property
    def waypoints(self) -> List[Pose]:
        """The recorded contact key poses in order: init, (mid,) goal."""
        return [self.init] + ([self.mid] if self.mid is not None else []) + [self.goal]

    @property
    def kind(self) -> str:
        """``"push"`` (default) or ``"push_under"`` — the straight floor-level push of IORS under from the placement
        stance (extras ``partner``, ``drop_cell``, ``goal_cell``, ``push_height_m``, ...)."""
        return str(self.extra.get("kind") or "push")


@dataclass
class BraceContact:
    """``contacts.brace`` (spec §7): closed gripper placed at ``contact`` (from ``pre``) before the working arm
    moves and held until it retreats. ``kind`` = the recorded candidate (``below_high`` / ``above_low`` /
    ``side_high`` for the wall tasks, ``opposite_face`` / ``top_press`` for small-scale articulation, ``stop_edge``,
    ``far_side``, ``row_left`` / ``column_top`` for blocks, ``partner_face`` for attachment); ``direction`` = the
    approach a (into the face). ``object`` (beyond §7) names the braced bddl instance."""

    arm: str
    object: str
    kind: str
    pre: Pose
    contact: Pose
    direction: List[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("arm", "object", "kind", "pre", "contact", "direction")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "BraceContact":
        d = _dict(d, where)
        return cls(
            arm=_str(_get(d, "arm", where), f"{where}.arm"),
            object=_str(_get(d, "object", where), f"{where}.object"),
            kind=_str(_get(d, "kind", where), f"{where}.kind"),
            pre=_pose(_get(d, "pre", where), f"{where}.pre"),
            contact=_pose(_get(d, "contact", where), f"{where}.contact"),
            direction=_vec(_get(d, "direction", where), 3, f"{where}.direction"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "arm": self.arm,
            "object": self.object,
            "kind": self.kind,
            "pre": self.pre.to_dict(),
            "contact": self.contact.to_dict(),
            "direction": list(self.direction),
        }
        out.update(self.extra)
        return out


@dataclass
class Contacts:
    """``contacts`` (spec §7): the push and / or brace contact of the alignment, touching and small-scale
    articulation tasks, and the attachment partner brace."""

    push: Optional[PushContact] = None
    brace: Optional[BraceContact] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("push", "brace")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Contacts":
        d = _dict(d, where)
        return cls(
            push=_opt(d.get("push"), PushContact.from_dict, f"{where}.push"),
            brace=_opt(d.get("brace"), BraceContact.from_dict, f"{where}.brace"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.push is not None:
            out["push"] = self.push.to_dict()
        if self.brace is not None:
            out["brace"] = self.brace.to_dict()
        out.update(self.extra)
        return out


@dataclass
class HandleWaypoint:
    """h(q_k) (spec §5.3): the handle ``contact`` point, ``approach`` direction and ``pinch`` axis in the world frame
    at joint fraction k. The eef pose is contact − 0.012·approach with a = approach, c = pinch (conventions
    ``handle_eef`` / ``handle_waypoints``)."""

    contact: List[float]
    approach: List[float]
    pinch: List[float]
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("contact", "approach", "pinch")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "HandleWaypoint":
        d = _dict(d, where)
        return cls(
            contact=_vec(_get(d, "contact", where), 3, f"{where}.contact"),
            approach=_vec(_get(d, "approach", where), 3, f"{where}.approach"),
            pinch=_vec(_get(d, "pinch", where), 3, f"{where}.pinch"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"contact": list(self.contact), "approach": list(self.approach), "pinch": list(self.pinch)}
        out.update(self.extra)
        return out


def _fraction_key(k: str, where: str) -> float:
    try:
        f = float(k)
    except ValueError:
        raise SidecarError(f"{where}: waypoint key {k!r} is not a joint fraction") from None
    if not 0.0 <= f <= 1.0:
        raise SidecarError(f"{where}: waypoint key {k!r} outside [0, 1]")
    return f


@dataclass
class Articulation:
    """``articulation`` (spec §7, §5.3 / §5.4). Keys of ``waypoints`` / ``handle_reach`` / ``base_follow`` are the
    joint fractions k as strings (``"0.0"`` = q_init — needed for the handle grasp —, ``"0.25"``, ..., ``"1.0"``);
    q_k = q_init + k·(q_goal − q_init). ``passive_from`` = the first k from which the link reaches q_goal passively
    (``None`` = drive to q_goal); ``follow_mode`` ``"base"`` tracks ``base_follow[k]`` with the arm locked, ``"arm"``
    moves the arm along h(q_k).

    Beyond spec §7: ``object`` — the articulated bddl instance (small-scale: the spawned target; large-scale: the
    scene object) and ``joint_type`` (``revolute`` / ``prismatic``, for the annotation numbers).
    """

    object: str
    joint: str
    joint_type: str
    q_init: float
    q_goal: float
    waypoints: Dict[str, HandleWaypoint]
    handle_reach: Dict[str, bool]
    follow_mode: str
    passive_from: Optional[float] = None
    base_follow: Optional[Dict[str, List[float]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("object", "joint", "joint_type", "q_init", "q_goal", "waypoints", "handle_reach", "passive_from", "follow_mode", "base_follow")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Articulation":
        d = _dict(d, where)
        wps = _dict(_get(d, "waypoints", where), f"{where}.waypoints")
        reach = _dict(_get(d, "handle_reach", where), f"{where}.handle_reach")
        bf = d.get("base_follow")
        if bf is not None:
            bf = {str(k): _pose2d(v, f"{where}.base_follow[{k!r}]") for k, v in _dict(bf, f"{where}.base_follow").items()}
        return cls(
            object=_str(_get(d, "object", where), f"{where}.object"),
            joint=_str(_get(d, "joint", where), f"{where}.joint"),
            joint_type=_str(_get(d, "joint_type", where), f"{where}.joint_type"),
            q_init=_num(_get(d, "q_init", where), f"{where}.q_init"),
            q_goal=_num(_get(d, "q_goal", where), f"{where}.q_goal"),
            waypoints={str(k): HandleWaypoint.from_dict(v, f"{where}.waypoints[{k!r}]") for k, v in wps.items()},
            handle_reach={str(k): _bool(v, f"{where}.handle_reach[{k!r}]") for k, v in reach.items()},
            follow_mode=_str(_get(d, "follow_mode", where), f"{where}.follow_mode"),
            passive_from=_opt(d.get("passive_from"), _num, f"{where}.passive_from"),
            base_follow=bf,
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "object": self.object,
            "joint": self.joint,
            "joint_type": self.joint_type,
            "q_init": self.q_init,
            "q_goal": self.q_goal,
            "waypoints": {k: v.to_dict() for k, v in self.waypoints.items()},
            "handle_reach": dict(self.handle_reach),
            "passive_from": self.passive_from,
            "follow_mode": self.follow_mode,
        }
        if self.base_follow is not None:
            out["base_follow"] = {k: list(v) for k, v in self.base_follow.items()}
        out.update(self.extra)
        return out

    def fractions(self) -> List[Tuple[float, str]]:
        """``(k, key)`` pairs of the recorded waypoints, sorted by k."""
        return sorted((float(k), k) for k in self.waypoints)

    def key_at(self, k: float) -> str:
        for f, key in self.fractions():
            if abs(f - k) < 1e-9:
                return key
        raise SidecarError(f"articulation: no waypoint recorded at fraction {k}")

    def q_at(self, k: float) -> float:
        return self.q_init + k * (self.q_goal - self.q_init)

    def followed_fractions(self) -> List[Tuple[float, str]]:
        """The waypoints the oracle drives through: every k > 0 up to and including ``passive_from`` (or all when
        ``passive_from`` is ``None``), in order (spec §5.3 "up to passive_from or q_goal")."""
        out = []
        for f, key in self.fractions():
            if f <= 0.0:
                continue
            if self.passive_from is not None and f > self.passive_from + 1e-9:
                break
            out.append((f, key))
        return out


@dataclass
class BlockRecord:
    """One block of the block-design program (spec §7 ``blocks[]``, §6.5): placement ``order`` (0 = anchor), the cube
    grasp (pinch across ``face_pair``, ``tilt_pick_deg`` / ``tilt_place_deg`` about the pinch axis, wrist ``yaw_deg``),
    the flip (``flip_axis`` a in the world frame and ``flip_sign``; ``flip_sign`` 0 with ``flip_axis`` ``None`` = no
    flip, yaw-align only), ``pick`` / ``place`` eef poses, and the left-arm braces ``brace_row`` (row-left blocks) /
    ``brace_col`` (column-top blocks) where recorded.

    Beyond spec §7: ``carry_height`` (the lift after the pick), ``push_row`` / ``push_col`` (the right-hand push
    contacts paired with the braces — §6.5 records the braces but the program needs the push it braces for) and the
    optional ``waypoints`` of the in-hand reorientation when the sampler recorded them.
    """

    instance: str
    order: int
    face_pair: str
    tilt_pick_deg: float
    tilt_place_deg: float
    yaw_deg: float
    flip_sign: int
    pick: Pose
    place: Pose
    carry_height: float
    flip_axis: Optional[List[float]] = None
    brace_row: Optional[BraceContact] = None
    brace_col: Optional[BraceContact] = None
    push_row: Optional[PushContact] = None
    push_col: Optional[PushContact] = None
    waypoints: Optional[List[Pose]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = (
        "instance", "order", "face_pair", "tilt_pick_deg", "tilt_place_deg", "yaw_deg", "flip_axis", "flip_sign",
        "pick", "place", "carry_height", "brace_row", "brace_col", "push_row", "push_col", "waypoints",
    )

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "BlockRecord":
        d = _dict(d, where)
        order = _get(d, "order", where)
        if isinstance(order, bool) or not isinstance(order, int):
            raise SidecarError(f"{where}.order: expected an integer, got {order!r}")
        sign = _get(d, "flip_sign", where)
        if isinstance(sign, bool) or not isinstance(sign, int):
            raise SidecarError(f"{where}.flip_sign: expected an integer, got {sign!r}")
        return cls(
            instance=_str(_get(d, "instance", where), f"{where}.instance"),
            order=order,
            face_pair=_str(_get(d, "face_pair", where), f"{where}.face_pair"),
            tilt_pick_deg=_num(_get(d, "tilt_pick_deg", where), f"{where}.tilt_pick_deg"),
            tilt_place_deg=_num(_get(d, "tilt_place_deg", where), f"{where}.tilt_place_deg"),
            yaw_deg=_num(_get(d, "yaw_deg", where), f"{where}.yaw_deg"),
            flip_sign=sign,
            pick=_pose(_get(d, "pick", where), f"{where}.pick"),
            place=_pose(_get(d, "place", where), f"{where}.place"),
            carry_height=_num(_get(d, "carry_height", where), f"{where}.carry_height"),
            flip_axis=_opt(d.get("flip_axis"), lambda v, w: _vec(v, 3, w), f"{where}.flip_axis"),
            brace_row=_opt(d.get("brace_row"), BraceContact.from_dict, f"{where}.brace_row"),
            brace_col=_opt(d.get("brace_col"), BraceContact.from_dict, f"{where}.brace_col"),
            push_row=_opt(d.get("push_row"), PushContact.from_dict, f"{where}.push_row"),
            push_col=_opt(d.get("push_col"), PushContact.from_dict, f"{where}.push_col"),
            waypoints=_opt(d.get("waypoints"), _poses, f"{where}.waypoints"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "instance": self.instance,
            "order": self.order,
            "face_pair": self.face_pair,
            "tilt_pick_deg": self.tilt_pick_deg,
            "tilt_place_deg": self.tilt_place_deg,
            "yaw_deg": self.yaw_deg,
            "flip_axis": None if self.flip_axis is None else list(self.flip_axis),
            "flip_sign": self.flip_sign,
            "pick": self.pick.to_dict(),
            "place": self.place.to_dict(),
            "carry_height": self.carry_height,
        }
        for k in ("brace_row", "brace_col", "push_row", "push_col"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v.to_dict()
        if self.waypoints is not None:
            out["waypoints"] = [p.to_dict() for p in self.waypoints]
        out.update(self.extra)
        return out

    @property
    def flips(self) -> bool:
        return self.flip_sign != 0

    @property
    def reorients(self) -> bool:
        """Whether the block needs an in-hand reorientation (a flip, or a non-zero yaw-align)."""
        return self.flips or abs(self.yaw_deg) > 1e-6


@dataclass
class Goal:
    """The camera-relocalization goal configuration (spec §7 ``goal``, §6.1): base pose, trunk pair and whether the
    arms stay as at init (``"same"``) or are lifted to untucked first (``"untucked"``, self-collision at the goal
    trunk)."""

    base_pose2d: List[float]
    translate: float
    tilt: float
    arms: str = "same"
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("base_pose2d", "translate", "tilt", "arms")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Goal":
        d = _dict(d, where)
        return cls(
            base_pose2d=_pose2d(_get(d, "base_pose2d", where), f"{where}.base_pose2d"),
            translate=_num(_get(d, "translate", where), f"{where}.translate"),
            tilt=_num(_get(d, "tilt", where), f"{where}.tilt"),
            arms=_str(_get(d, "arms", where, "same"), f"{where}.arms"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"base_pose2d": list(self.base_pose2d), "translate": self.translate, "tilt": self.tilt, "arms": self.arms}
        out.update(self.extra)
        return out

    @property
    def trunk(self) -> Tuple[float, float]:
        return (self.translate, self.tilt)


@dataclass
class ObjectLanguage:
    """Per-object language facts (spec §6.3 / oracle doc §6.3): ``display_name`` and the disambiguating
    ``descriptor`` ("the cereal box on the left"), recorded at sampling time."""

    display_name: str
    descriptor: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("display_name", "descriptor")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "ObjectLanguage":
        d = _dict(d, where)
        return cls(
            display_name=_str(_get(d, "display_name", where), f"{where}.display_name"),
            descriptor=_opt(d.get("descriptor"), _str, f"{where}.descriptor"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"display_name": self.display_name}
        if self.descriptor is not None:
            out["descriptor"] = self.descriptor
        out.update(self.extra)
        return out

    @property
    def phrase(self) -> str:
        """The noun phrase the annotations use: the descriptor when recorded, else "the <display_name>"."""
        return self.descriptor if self.descriptor else f"the {self.display_name}"


@dataclass
class Language:
    """``language`` (spec §7): ``objects`` keyed by bddl instance and the ``goal_numbers`` of the instance-level
    description."""

    objects: Dict[str, ObjectLanguage] = field(default_factory=dict)
    goal_numbers: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = ("objects", "goal_numbers")

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Language":
        d = _dict(d, where)
        objs = _dict(_get(d, "objects", where, {}), f"{where}.objects")
        return cls(
            objects={_str(k, f"{where}.objects key"): ObjectLanguage.from_dict(v, f"{where}.objects[{k!r}]") for k, v in objs.items()},
            goal_numbers=dict(_dict(_get(d, "goal_numbers", where, {}), f"{where}.goal_numbers")),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"objects": {k: v.to_dict() for k, v in self.objects.items()}, "goal_numbers": dict(self.goal_numbers)}
        out.update(self.extra)
        return out

    def phrase(self, instance: str) -> str:
        """Noun phrase for a bddl instance; raises :class:`SidecarError` when the instance was not recorded."""
        entry = self.objects.get(instance)
        if entry is None:
            raise SidecarError(f"language.objects: no entry for {instance!r}")
        return entry.phrase

    def display_name(self, instance: str) -> str:
        entry = self.objects.get(instance)
        if entry is None:
            raise SidecarError(f"language.objects: no entry for {instance!r}")
        return entry.display_name


# ----------------------------------------------------------------------------------------------------------------
# Alias → subtask (iddl alias templates)
# ----------------------------------------------------------------------------------------------------------------

IDDL_ROOT = os.path.join(ASSETS_ROOT, "iddl")
_SLOT_RE = re.compile(r"^\{[a-z_]+\}$")
_TOKEN_RE = re.compile(r"^[a-z0-9_]+$")


@functools.lru_cache(maxsize=None)
def alias_templates(root: str = IDDL_ROOT) -> Dict[str, Tuple[str, str]]:
    """``{subtask_id: (family, alias_template)}`` read from every ``*.iddl`` under ``root``."""
    out: Dict[str, Tuple[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*.iddl"))):
        with open(path) as f:
            text = "\n".join(line.split(";", 1)[0] for line in f.read().splitlines())
        m_id = re.search(r"\(define\s+\(iddl\s+([^\s()]+)\s*\)", text)
        m_family = re.search(r"\(:family\s+([^\s()]+)\s*\)", text)
        m_alias = re.search(r"\(:alias\s+([^\s()]+)\s*\)", text)
        if not (m_id and m_family and m_alias):
            raise SidecarError(f"{path}: cannot read the iddl id / family / alias template")
        out[m_id.group(1)] = (m_family.group(1), m_alias.group(1))
    if not out:
        raise SidecarError(f"no iddl files under {root}")
    return out


def subtask_of_alias(alias: str, root: str = IDDL_ROOT) -> str:
    """Resolve a grounded alias (``inter_object_relation_satisfaction-inside-apple-bowl-breakfast_table``) or an alias
    template itself to the iddl subtask id (``inter_object_relation_satisfaction-inside``). Slots ``{role}`` match one
    ``-``-separated token of ``[a-z0-9_]+`` (category names, group tokens such as ``block_4``)."""
    if not isinstance(alias, str) or not alias:
        raise SidecarError(f"alias must be a non-empty string, got {alias!r}")
    templates = alias_templates(root)
    tokens = alias.split("-")
    matches = []
    for subtask, (_family, template) in templates.items():
        if alias == template:
            return subtask
        t_tokens = template.split("-")
        if len(t_tokens) != len(tokens):
            continue
        ok = True
        for tok, t_tok in zip(tokens, t_tokens):
            if _SLOT_RE.match(t_tok):
                if not _TOKEN_RE.match(tok):
                    ok = False
                    break
            elif tok != t_tok:
                ok = False
                break
        if ok:
            matches.append(subtask)
    if not matches:
        raise SidecarError(f"alias {alias!r} matches no iddl alias template")
    if len(matches) > 1:
        raise SidecarError(f"alias {alias!r} is ambiguous: {matches}")
    return matches[0]


def family_of_subtask(subtask: str, root: str = IDDL_ROOT) -> str:
    try:
        return alias_templates(root)[subtask][0]
    except KeyError:
        raise SidecarError(f"unknown subtask {subtask!r}") from None


# ----------------------------------------------------------------------------------------------------------------
# Sidecar
# ----------------------------------------------------------------------------------------------------------------


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class Sidecar:
    """The oracle sidecar of one instance (spec §7). Blocks a task does not use are ``None`` / empty."""

    instance_id: str
    tro_sha256: str
    task_family: str
    alias: str
    language: Language
    schema: str = SCHEMA
    path_bucket_m: Optional[float] = None
    stances: Dict[str, Stance] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)
    grasps: List[GraspRecord] = field(default_factory=list)
    contacts: Optional[Contacts] = None
    articulation: Optional[Articulation] = None
    blocks: List[BlockRecord] = field(default_factory=list)
    goal: Optional[Goal] = None
    waypoint_policy: str = "checked"
    extra: Dict[str, Any] = field(default_factory=dict)

    KNOWN = (
        "schema", "instance_id", "tro_sha256", "task_family", "alias", "path_bucket_m", "stances", "transitions",
        "paths", "grasps", "contacts", "articulation", "blocks", "goal", "language", "waypoint_policy",
    )

    # -- I/O -------------------------------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Any) -> "Sidecar":
        where = "sidecar"
        d = _dict(d, where)
        schema = _get(d, "schema", where)
        if schema != SCHEMA:
            raise SidecarError(f"{where}.schema: expected {SCHEMA!r}, got {schema!r}")
        stances = _dict(_get(d, "stances", where, {}), f"{where}.stances")
        return cls(
            schema=schema,
            instance_id=_str(_get(d, "instance_id", where), f"{where}.instance_id"),
            tro_sha256=_str(_get(d, "tro_sha256", where), f"{where}.tro_sha256"),
            task_family=_str(_get(d, "task_family", where), f"{where}.task_family"),
            alias=_str(_get(d, "alias", where), f"{where}.alias"),
            path_bucket_m=_opt(d.get("path_bucket_m"), _num, f"{where}.path_bucket_m"),
            stances={_str(k, f"{where}.stances key"): Stance.from_dict(v, f"{where}.stances[{k!r}]") for k, v in stances.items()},
            transitions=[Transition.from_dict(t, f"{where}.transitions[{i}]") for i, t in enumerate(_list(_get(d, "transitions", where, []), f"{where}.transitions"))],
            paths=[Path.from_dict(p, f"{where}.paths[{i}]") for i, p in enumerate(_list(_get(d, "paths", where, []), f"{where}.paths"))],
            grasps=[GraspRecord.from_dict(g, f"{where}.grasps[{i}]") for i, g in enumerate(_list(_get(d, "grasps", where, []), f"{where}.grasps"))],
            contacts=_opt(d.get("contacts"), Contacts.from_dict, f"{where}.contacts"),
            articulation=_opt(d.get("articulation"), Articulation.from_dict, f"{where}.articulation"),
            blocks=[BlockRecord.from_dict(b, f"{where}.blocks[{i}]") for i, b in enumerate(_list(_get(d, "blocks", where, []), f"{where}.blocks"))],
            goal=_opt(d.get("goal"), Goal.from_dict, f"{where}.goal"),
            language=Language.from_dict(_get(d, "language", where), f"{where}.language"),
            waypoint_policy=_str(_get(d, "waypoint_policy", where), f"{where}.waypoint_policy"),
            extra=_extra(d, cls.KNOWN),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema": self.schema,
            "instance_id": self.instance_id,
            "tro_sha256": self.tro_sha256,
            "task_family": self.task_family,
            "alias": self.alias,
            "path_bucket_m": self.path_bucket_m,
        }
        if self.stances:
            out["stances"] = {k: v.to_dict() for k, v in self.stances.items()}
        if self.transitions:
            out["transitions"] = [t.to_dict() for t in self.transitions]
        if self.paths:
            out["paths"] = [p.to_dict() for p in self.paths]
        if self.grasps:
            out["grasps"] = [g.to_dict() for g in self.grasps]
        if self.contacts is not None:
            out["contacts"] = self.contacts.to_dict()
        if self.articulation is not None:
            out["articulation"] = self.articulation.to_dict()
        if self.blocks:
            out["blocks"] = [b.to_dict() for b in self.blocks]
        if self.goal is not None:
            out["goal"] = self.goal.to_dict()
        out["language"] = self.language.to_dict()
        out["waypoint_policy"] = self.waypoint_policy
        out.update(self.extra)
        return out

    @classmethod
    def load(cls, path: str) -> "Sidecar":
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise SidecarError(f"{path}: invalid JSON ({e})") from None
        try:
            return cls.from_dict(data)
        except SidecarError as e:
            raise SidecarError(f"{path}: {e}") from None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    # -- accessors -------------------------------------------------------------------------------------------------

    @property
    def subtask(self) -> str:
        return subtask_of_alias(self.alias)

    def stance(self, name: str) -> Stance:
        try:
            return self.stances[name]
        except KeyError:
            raise SidecarError(f"stances: no {name!r} stance recorded") from None

    def path(self, frm: str, to: str) -> Optional[Path]:
        hits = [p for p in self.paths if p.frm == frm and p.to == to]
        if len(hits) > 1:
            raise SidecarError(f"paths: {len(hits)} paths recorded from {frm!r} to {to!r}")
        return hits[0] if hits else None

    def transition(self, frm: str, to: str) -> Optional[Transition]:
        hits = [t for t in self.transitions if t.frm == frm and t.to == to]
        if len(hits) > 1:
            raise SidecarError(f"transitions: {len(hits)} transitions recorded from {frm!r} to {to!r}")
        return hits[0] if hits else None

    def has_block(self, name: str) -> bool:
        """``name`` is a dotted block name of :data:`REQUIRED_BLOCKS` (``stances.init``, ``contacts.push``, ...)."""
        if name.startswith("stances."):
            return name.split(".", 1)[1] in self.stances
        if name == "contacts.push":
            return self.contacts is not None and self.contacts.push is not None
        if name == "contacts.brace":
            return self.contacts is not None and self.contacts.brace is not None
        value = getattr(self, name)
        return bool(value) if isinstance(value, (list, dict)) else value is not None

    # -- TRO binding -----------------------------------------------------------------------------------------------

    def verify_tro(self, tro_path: str) -> str:
        """Check that ``tro_path`` is the TRO this sidecar was written for; returns the digest, raises
        :class:`SidecarError` on a mismatch (the executor refuses to run a sidecar against another TRO)."""
        digest = sha256_of_file(tro_path)
        if digest != self.tro_sha256:
            raise SidecarError(f"tro_sha256 mismatch: sidecar {self.tro_sha256} vs {tro_path} {digest}")
        return digest

    # -- validation ------------------------------------------------------------------------------------------------

    def problems(self, check_program: bool = True) -> List[str]:
        """Every validation problem found (empty = valid). Semantic checks on top of the structural ones
        ``from_dict`` already enforced; with ``check_program`` the alias's program is expanded and every action's
        objects must have language entries and annotate without error."""
        out: List[str] = []
        if self.schema != SCHEMA:
            out.append(f"schema: expected {SCHEMA!r}, got {self.schema!r}")
        if not _SHA_RE.match(self.tro_sha256):
            out.append("tro_sha256: expected 64 lowercase hex digits")
        try:
            subtask = self.subtask
        except SidecarError as e:
            out.append(str(e))
            return out
        family = family_of_subtask(subtask)
        if self.task_family != family:
            out.append(f"task_family: {self.task_family!r} but the alias belongs to {family!r}")
        for name in REQUIRED_BLOCKS[subtask]:
            if not self.has_block(name):
                out.append(f"{subtask}: required block '{name}' is missing")
        if self.path_bucket_m is not None and self.path_bucket_m <= 0:
            out.append("path_bucket_m: must be positive")

        stance_names = set(self.stances)
        for name, st in self.stances.items():
            w = f"stances[{name!r}]"
            if not 0.0 <= st.translate <= 1.0:
                out.append(f"{w}.translate: {st.translate} outside [0, 1]")
            if isinstance(st.arms, str) and st.arms != "untucked":
                out.append(f"{w}.arms: string must be 'untucked', got {st.arms!r}")
            for a in st.working_arms:
                if a not in ARMS:
                    out.append(f"{w}.arms: unknown arm {a!r}")
            if st.stand_off is not None and st.stand_off < 0:
                out.append(f"{w}.stand_off: negative")

        seen = set()
        for i, t in enumerate(self.transitions):
            w = f"transitions[{i}]"
            for end in (t.frm, t.to):
                if end not in stance_names:
                    out.append(f"{w}: stance {end!r} is not recorded")
            if t.frm == t.to:
                out.append(f"{w}: from == to")
            if t.stand_off < 0:
                out.append(f"{w}.stand_off: negative")
            if t.order != TRANSITION_ORDER:
                out.append(f"{w}.order: expected {TRANSITION_ORDER}, got {t.order}")
            if (t.frm, t.to) in seen:
                out.append(f"{w}: duplicate transition {t.frm!r} -> {t.to!r}")
            seen.add((t.frm, t.to))

        path_ends = stance_names | ({"goal"} if self.goal is not None else set())
        seen = set()
        for i, p in enumerate(self.paths):
            w = f"paths[{i}]"
            for end in (p.frm, p.to):
                if end not in path_ends:
                    out.append(f"{w}: endpoint {end!r} is neither a recorded stance nor the goal")
            if p.mode not in PATH_MODES:
                out.append(f"{w}.mode: expected one of {PATH_MODES}, got {p.mode!r}")
            if not p.waypoints:
                out.append(f"{w}.waypoints: empty")
            if p.length_m < 0:
                out.append(f"{w}.length_m: negative")
            if (p.frm, p.to) in seen:
                out.append(f"{w}: duplicate path {p.frm!r} -> {p.to!r}")
            seen.add((p.frm, p.to))

        for i, g in enumerate(self.grasps):
            w = f"grasps[{i}]"
            if g.arm not in ARMS:
                out.append(f"{w}.arm: unknown arm {g.arm!r}")
            if g.carry_height is not None and g.carry_height < 0:
                out.append(f"{w}.carry_height: negative")
            if g.waypoints is not None and not g.waypoints:
                out.append(f"{w}.waypoints: empty list")

        if self.contacts is not None:
            out.extend(_contact_problems(self.contacts.push, self.contacts.brace, "contacts"))

        if self.articulation is not None:
            out.extend(_articulation_problems(self.articulation))

        if self.blocks:
            out.extend(_block_problems(self.blocks))

        if self.goal is not None:
            if not 0.0 <= self.goal.translate <= 1.0:
                out.append(f"goal.translate: {self.goal.translate} outside [0, 1]")
            if self.goal.arms not in GOAL_ARMS:
                out.append(f"goal.arms: expected one of {GOAL_ARMS}, got {self.goal.arms!r}")

        if check_program and not out:
            out.extend(self._program_problems())
        return out

    def _program_problems(self) -> List[str]:
        from ispatialgym.oracle import programs  # lazy: programs imports this module

        try:
            actions = programs.expand(self.alias, self)
        except SidecarError as e:
            return [f"program: {e}"]
        out: List[str] = []
        for i, action in enumerate(actions):
            w = f"program[{i}] {action.name}"
            for arm in action.arms:
                if arm not in ARMS:
                    out.append(f"{w}: unknown arm {arm!r}")
            for obj in action.objects:
                if obj not in self.language.objects:
                    out.append(f"{w}: language.objects has no entry for {obj!r}")
            if out:
                continue
            try:
                fine, numbers = action.annotate(self)
            except SidecarError as e:
                out.append(f"{w}: annotate failed: {e}")
                continue
            if not fine or not numbers:
                out.append(f"{w}: empty annotation")
        return out

    def validate(self, check_program: bool = True) -> "Sidecar":
        problems = self.problems(check_program=check_program)
        if problems:
            raise SidecarError(f"{self.instance_id}: invalid sidecar:\n  " + "\n  ".join(problems))
        return self


def _unit_problem(vec: List[float], where: str) -> Optional[str]:
    norm = math.sqrt(sum(v * v for v in vec))
    if abs(norm - 1.0) > 1e-3:
        return f"{where}: not a unit vector (norm {norm:.4f})"
    return None


def _contact_problems(push: Optional[PushContact], brace: Optional[BraceContact], where: str) -> List[str]:
    out: List[str] = []
    if push is not None:
        w = f"{where}.push"
        if push.arm not in ARMS:
            out.append(f"{w}.arm: unknown arm {push.arm!r}")
        if push.distance_m <= 0:
            out.append(f"{w}.distance_m: must be positive")
        p = _unit_problem(push.direction, f"{w}.direction")
        if p:
            out.append(p)
        elif push.mid is None:
            # straight push: the recorded goal must be init + distance * direction
            expect = [a + push.distance_m * d for a, d in zip(push.init.position, push.direction)]
            err = math.sqrt(sum((a - b) ** 2 for a, b in zip(expect, push.goal.position)))
            if err > 2e-3:
                out.append(f"{w}: straight push (no mid) but goal is {err * 1000:.1f} mm off init + distance_m * direction")
    if brace is not None:
        w = f"{where}.brace"
        if brace.arm not in ARMS:
            out.append(f"{w}.arm: unknown arm {brace.arm!r}")
        p = _unit_problem(brace.direction, f"{w}.direction")
        if p:
            out.append(p)
    if push is not None and brace is not None and push.arm == brace.arm:
        out.append(f"{where}: push and brace use the same arm {push.arm!r}")
    return out


def _articulation_problems(art: Articulation) -> List[str]:
    out: List[str] = []
    w = "articulation"
    if art.joint_type not in JOINT_TYPES:
        out.append(f"{w}.joint_type: expected one of {JOINT_TYPES}, got {art.joint_type!r}")
    if abs(art.q_goal - art.q_init) < 1e-9:
        out.append(f"{w}: q_goal equals q_init")
    if art.follow_mode not in FOLLOW_MODES:
        out.append(f"{w}.follow_mode: expected one of {FOLLOW_MODES}, got {art.follow_mode!r}")
    fractions = {}
    for key in art.waypoints:
        try:
            fractions[key] = _fraction_key(key, f"{w}.waypoints")
        except SidecarError as e:
            out.append(str(e))
    if not any(abs(f) < 1e-9 for f in fractions.values()):
        out.append(f"{w}.waypoints: no waypoint at fraction 0.0 (the handle pose at q_init the grasp needs)")
    if not any(f > 0 for f in fractions.values()):
        out.append(f"{w}.waypoints: no waypoint beyond q_init")
    for key, wp in art.waypoints.items():
        for name in ("approach", "pinch"):
            p = _unit_problem(getattr(wp, name), f"{w}.waypoints[{key!r}].{name}")
            if p:
                out.append(p)
    for key in art.handle_reach:
        if key not in art.waypoints:
            out.append(f"{w}.handle_reach: key {key!r} has no waypoint")
    for key, f in fractions.items():
        if f > 0 and key not in art.handle_reach:
            out.append(f"{w}.handle_reach: no entry for waypoint {key!r}")
    if art.passive_from is not None:
        if not any(abs(f - art.passive_from) < 1e-9 for f in fractions.values()):
            out.append(f"{w}.passive_from: {art.passive_from} is not a recorded waypoint fraction")
        elif art.passive_from <= 0:
            out.append(f"{w}.passive_from: must be > 0")
    if not out:
        for f, key in art.followed_fractions():
            if not art.handle_reach.get(key, False):
                out.append(f"{w}.handle_reach[{key!r}]: a waypoint the oracle drives through is recorded unreachable")
    if art.follow_mode in ("base", "hybrid"):
        if art.base_follow is None:
            out.append(f"{w}.base_follow: required in follow_mode {art.follow_mode!r}")
        else:
            for key in art.base_follow:
                if key not in art.waypoints:
                    out.append(f"{w}.base_follow: key {key!r} has no waypoint")
            if not out:
                for f, key in art.followed_fractions():
                    if key not in art.base_follow:
                        out.append(f"{w}.base_follow: no base pose for waypoint {key!r}")
    elif art.base_follow is not None:
        out.append(f"{w}.base_follow: present but follow_mode is {art.follow_mode!r}")
    if art.follow_mode == "hybrid":
        out.extend(_follow_path_problems(art.extra.get("base_follow_path"), f"{w}.base_follow_path"))
    return out


def _follow_path_problems(path: Any, w: str) -> List[str]:
    """The hybrid follow's dense path: ``poses[]`` (>= 2) of ``{q, base_pose2d [x, y, yaw], arm_q [7 joints]}``, the
    first pose at the manipulation stance (``k`` 0), ``k`` non-decreasing."""
    if not isinstance(path, dict):
        return [f"{w}: required in follow_mode 'hybrid' (a dict with poses[])"]
    poses = path.get("poses")
    if not isinstance(poses, list) or len(poses) < 2:
        return [f"{w}.poses: expected a list of >= 2 poses"]
    out: List[str] = []
    prev_k = None
    for i, p in enumerate(poses):
        if not isinstance(p, dict):
            out.append(f"{w}.poses[{i}]: expected a dict")
            continue
        for key, n in (("base_pose2d", 3), ("arm_q", 7)):
            v = p.get(key)
            if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in v):
                out.append(f"{w}.poses[{i}].{key}: expected {n} finite numbers")
        if not isinstance(p.get("q"), (int, float)):
            out.append(f"{w}.poses[{i}].q: expected a number")
        k = p.get("k")
        if isinstance(k, (int, float)):
            if prev_k is not None and k < prev_k - 1e-9:
                out.append(f"{w}.poses[{i}].k: {k} decreases")
            prev_k = float(k)
    if not out and isinstance(poses[0].get("k"), (int, float)) and abs(float(poses[0]["k"])) > 1e-9:
        out.append(f"{w}.poses[0].k: the path must start at the grasp (k = 0), got {poses[0]['k']}")
    return out


def _block_problems(blocks: List[BlockRecord]) -> List[str]:
    out: List[str] = []
    orders = sorted(b.order for b in blocks)
    if orders != list(range(len(blocks))):
        out.append(f"blocks: orders {orders} are not 0..{len(blocks) - 1}")
    names = [b.instance for b in blocks]
    if len(set(names)) != len(names):
        out.append("blocks: duplicate instance names")
    for i, b in enumerate(blocks):
        w = f"blocks[{i}]"
        if b.face_pair not in FACE_PAIRS:
            out.append(f"{w}.face_pair: expected one of {FACE_PAIRS}, got {b.face_pair!r}")
        if b.flip_sign not in (-1, 0, 1):
            out.append(f"{w}.flip_sign: expected -1, 0 or 1, got {b.flip_sign}")
        if (b.flip_sign == 0) != (b.flip_axis is None):
            out.append(f"{w}: flip_axis must be present exactly when flip_sign != 0")
        if b.flip_axis is not None:
            p = _unit_problem(b.flip_axis, f"{w}.flip_axis")
            if p:
                out.append(p)
        if b.carry_height < 0:
            out.append(f"{w}.carry_height: negative")
        for kind in ("row", "col"):
            brace = getattr(b, f"brace_{kind}")
            push = getattr(b, f"push_{kind}")
            if (brace is None) != (push is None):
                out.append(f"{w}: brace_{kind} and push_{kind} must be recorded together")
            out.extend(_contact_problems(push, brace, f"{w}.{kind}"))
        if b.waypoints is not None and not b.waypoints:
            out.append(f"{w}.waypoints: empty list")
    return out


__all__ = [
    "SCHEMA", "ARMS", "SUBTASKS", "REQUIRED_BLOCKS", "SidecarError", "Pose", "Stance", "Transition", "Path",
    "GraspRecord", "PushContact", "BraceContact", "Contacts", "HandleWaypoint", "Articulation", "BlockRecord", "Goal",
    "ObjectLanguage", "Language", "Sidecar", "alias_templates", "subtask_of_alias", "family_of_subtask",
    "sha256_of_file",
]
