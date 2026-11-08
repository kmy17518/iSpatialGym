"""Stance / transition / path records of the v2 sampler (spec §3.3, §3.7, §3.8, §7) and the frame helpers they share.

Pure module: no simulator import. Every record serialises to the sidecar blocks of spec §7 (``to_sidecar()``) and is
what the task modules, ``navigation``, ``ik`` and ``program_check`` exchange. Until ``ispatialgym.oracle.sidecar`` (F6)
lands, :class:`SidecarDraft` is the thin dict-backed adapter of the §7 schema; ``core.write_instance`` finalises it
(``tro_sha256``). When F6 lands, ``SidecarDraft.to_dict()`` is what its dataclasses are built from.

Frames (spec §3.3): base frame x ahead, lateral + = robot's left. Edge frame of a tabletop stance: ``inward`` = unit xy
normal from the edge into the support (the robot faces along it), ``tangent`` = unit xy along the edge, robot's left when
facing inward. ``local()`` / ``world()`` convert between world xy and (depth into the support from the front edge,
lateral offset from the robot's centre line), as the v1 ``Stance`` did.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

Pose = Tuple[th.Tensor, th.Tensor]

SIDECAR_SCHEMA = "isg-oracle-v1"
ARMS = ("left", "right")


def _t(x) -> th.Tensor:
    return x if isinstance(x, th.Tensor) else th.tensor(x, dtype=th.float32)


def _lst(x):
    if x is None:
        return None
    if isinstance(x, th.Tensor):
        return [float(v) for v in x.flatten().tolist()]
    if isinstance(x, (list, tuple)):
        return [_lst(v) if isinstance(v, (list, tuple, th.Tensor)) else v for v in x]
    return x


def pose_to_sidecar(pose: Optional[Pose]) -> Optional[Dict[str, List[float]]]:
    """``{"position": [x, y, z], "orientation": [x, y, z, w]}`` (world frame) or None."""
    if pose is None:
        return None
    return {"position": _lst(pose[0]), "orientation": _lst(pose[1])}


def pose_from_sidecar(d: Optional[Dict[str, Any]]) -> Optional[Pose]:
    if not d:
        return None
    return _t(d["position"]), _t(d["orientation"])


# ----------------------------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class EdgeFrame:
    """One edge of a support top (spec §4.2): ``mid`` world xy of the edge midpoint, ``inward`` unit xy normal pointing
    from the edge into the support, ``tangent`` = robot's left when facing inward, ``length`` of the edge, ``overhang``
    = AABB overhang beyond the raster edge when >= 5 cm (added to the stand-off, else 0)."""

    name: str
    mid: th.Tensor
    inward: th.Tensor
    length: float
    overhang: float = 0.0

    @property
    def tangent(self) -> th.Tensor:
        return th.tensor([-float(self.inward[1]), float(self.inward[0])])

    @property
    def yaw(self) -> float:
        """Base yaw of a stance facing the edge (= direction of ``inward``)."""
        return math.atan2(float(self.inward[1]), float(self.inward[0]))

    @staticmethod
    def from_edge(name: str, mid, outward_normal, length: float, overhang: float = 0.0) -> "EdgeFrame":
        """From a ``SupportTop.edges()`` tuple (name, midpoint, *outward* normal, length)."""
        n = _t(outward_normal)[:2]
        return EdgeFrame(name, _t(mid)[:2].clone(), (-n).clone(), float(length), float(overhang))

    def base_xy(self, stand_off: float, lateral: float = 0.0) -> th.Tensor:
        """Base centre ``stand_off`` (+ overhang) outside the edge, shifted ``lateral`` along the tangent."""
        return self.mid - self.inward * (stand_off + self.overhang) + self.tangent * lateral

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "mid": _lst(self.mid), "inward": _lst(self.inward), "tangent": _lst(self.tangent),
                "length_m": self.length, "overhang_m": self.overhang}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "EdgeFrame":
        return EdgeFrame(d["name"], _t(d["mid"]), _t(d["inward"]), float(d["length_m"]), float(d.get("overhang_m", 0.0)))


@dataclass
class FaceFrame:
    """Face frame of the large-scale tasks (spec §3.3): ``n`` outward unit normal of the manipulation face, ``P``
    reference point (world, 3-D), ``d`` distance of the base centre from the face plane along n, ``lam`` lateral offset
    from P along the face tangent (+ = robot's left when facing the face)."""

    name: str
    n: th.Tensor
    P: th.Tensor
    d: float = 0.0
    lam: float = 0.0

    @property
    def tangent(self) -> th.Tensor:
        """Robot's left when facing the face (the robot faces along -n)."""
        return th.tensor([float(self.n[1]), -float(self.n[0])])

    @property
    def yaw(self) -> float:
        return math.atan2(-float(self.n[1]), -float(self.n[0]))

    def base_xy(self, d: Optional[float] = None, lam: Optional[float] = None) -> th.Tensor:
        d = self.d if d is None else d
        lam = self.lam if lam is None else lam
        return self.P[:2] + self.n[:2] * d + self.tangent * lam

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "n": _lst(self.n), "P": _lst(self.P), "d": self.d, "lambda": self.lam}


# ----------------------------------------------------------------------------------------------------------------
# Stance
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Stance:
    """One recorded robot stance (spec §7 ``stances.<name>``).

    ``base_pose2d`` = (x, y, yaw) world; ``translate`` / ``tilt`` = JoyLo trunk; ``stand_off`` s (m from the edge / face);
    ``edge`` (tabletop) or ``face`` (large-scale) frame; ``lateral`` = offset along the edge from its midpoint (m,
    + = robot's left); ``focus`` = the world point the manipulation stance looks at (x, y, z); ``arms`` = the idle
    posture name (``"untucked"`` default; ``"tucked"`` / ``"reset"`` = JoyLo reset arms, ``"ready"``) or the list of
    working arms; ``contacts`` = names of the contact records made from this stance; ``q`` = the **recorded
    configuration** of the stance (28-D): its arm joints and finger openings are what ``IKChecker.q_locked`` locks and
    seeds with (§3.3 exceptions — the held-object hold, the camera-relocalization reset arms), the base virtual joints
    are re-derived at every use (they belong to the robot root at the time ``q`` was built); ``fingers`` = gripper
    opening when it is not the default open 0.05 (closed pads for push / brace stances); ``carried`` = bddl instance
    carried at this stance (None = hands empty) — attached (posed in the hand per its grasp record) for the checks.
    """

    name: str
    base_pose2d: Tuple[float, float, float]
    translate: float
    tilt: float = 0.0
    stand_off: Optional[float] = None
    edge: Optional[EdgeFrame] = None
    face: Optional[FaceFrame] = None
    lateral: float = 0.0
    focus: Optional[th.Tensor] = None
    arms: Any = "untucked"
    contacts: List[str] = field(default_factory=list)
    q: Optional[th.Tensor] = None
    carried: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    fingers: Optional[float] = None
    #: A6: posture of the idle arm at a manipulation stance (``"untucked"`` / ``"ready"`` / ``"tucked"``), set by the
    #: stance search when the untucked idle arm collides; written to the sidecar as ``idle_posture`` + ``arms_q``
    idle_posture: Optional[str] = None

    @property
    def posture(self) -> str:
        """The posture name for both arms (``arms`` when it is a posture string, else ``"untucked"``)."""
        return self.arms if isinstance(self.arms, str) and self.arms in ("untucked", "tucked", "reset", "ready") else "untucked"

    @property
    def working_arms(self) -> List[str]:
        """The arms this stance was solved for (``arms`` as a list; empty for a posture string)."""
        return list(self.arms) if isinstance(self.arms, (list, tuple)) else []

    @property
    def idle_arm(self) -> Optional[str]:
        """The single idle arm of a one-arm manipulation stance (None for init stances and two-arm stances)."""
        w = self.working_arms
        if len(w) != 1:
            return None
        return "left" if w[0] == "right" else "right"

    # -- frame helpers ------------------------------------------------------------------------------------------
    @property
    def base_xy(self) -> th.Tensor:
        return th.tensor([float(self.base_pose2d[0]), float(self.base_pose2d[1])])

    @property
    def yaw(self) -> float:
        return float(self.base_pose2d[2])

    @property
    def forward(self) -> th.Tensor:
        """Unit xy along the base +x (= inward for a tabletop stance)."""
        return th.tensor([math.cos(self.yaw), math.sin(self.yaw)])

    @property
    def left(self) -> th.Tensor:
        """Unit xy along the base +y (robot's left)."""
        return th.tensor([-math.sin(self.yaw), math.cos(self.yaw)])

    @property
    def inward(self) -> th.Tensor:
        return self.edge.inward if self.edge is not None else self.forward

    @property
    def tangent(self) -> th.Tensor:
        return self.edge.tangent if self.edge is not None else self.left

    @property
    def edge_depth(self) -> float:
        """Distance from the base centre to the front edge along ``inward`` (= stand_off + overhang for an edge stance)."""
        if self.edge is None:
            return 0.0
        return float(th.dot(self.edge.mid - self.base_xy, self.edge.inward))

    def local(self, xy) -> Tuple[float, float]:
        """World xy -> (depth into the support from the front edge, lateral from the robot's centre line, + = left)."""
        rel = _t(xy)[:2] - self.base_xy
        return float(rel @ self.inward) - self.edge_depth, float(rel @ self.tangent)

    def world(self, depth: float, lateral: float) -> th.Tensor:
        """(depth from the front edge, lateral from the centre line) -> world xy."""
        return self.base_xy + self.inward * (self.edge_depth + depth) + self.tangent * lateral

    def base_to_world(self, xy_base) -> th.Tensor:
        """Base-frame xy (x ahead, + lateral left) -> world xy."""
        p = _t(xy_base)
        return self.base_xy + self.forward * float(p[0]) + self.left * float(p[1])

    def world_to_base(self, xy_world) -> th.Tensor:
        rel = _t(xy_world)[:2] - self.base_xy
        return th.tensor([float(rel @ self.forward), float(rel @ self.left)])

    def with_trunk(self, translate: float, tilt: float, name: Optional[str] = None) -> "Stance":
        """Copy with another trunk (``q`` dropped: it must be recomputed)."""
        return Stance(name or self.name, tuple(self.base_pose2d), float(translate), float(tilt), self.stand_off, self.edge, self.face,
                      self.lateral, self.focus, self.arms, list(self.contacts), None, self.carried, dict(self.extras), self.fingers, self.idle_posture)

    def shifted(self, stand_off: Optional[float] = None, lateral: Optional[float] = None, name: Optional[str] = None) -> "Stance":
        """Copy moved along the edge frame to another (stand_off, lateral); ``q`` dropped."""
        if self.edge is None:
            raise ValueError("shifted() needs an edge frame")
        s = self.stand_off if stand_off is None else float(stand_off)
        lat = self.lateral if lateral is None else float(lateral)
        xy = self.edge.base_xy(s, lat)
        return Stance(name or self.name, (float(xy[0]), float(xy[1]), self.edge.yaw), self.translate, self.tilt, s, self.edge, self.face, lat,
                      self.focus, self.arms, list(self.contacts), None, self.carried, dict(self.extras), self.fingers, self.idle_posture)

    # -- serialisation ------------------------------------------------------------------------------------------
    def to_sidecar(self) -> Dict[str, Any]:
        """The ``stances.<name>`` block of spec §7 (only the keys a stance has) plus the task's ``extras`` (``goal_cell``,
        ``init_surface``, ``path_length_m``, ``bearing_deg``, ``deviation_deg``, ...; the F6 ``Sidecar`` keeps unknown keys
        in ``extra``). A known key is never overridden by an extra."""
        # F6 contract (``oracle.sidecar.Stance``): ``arms`` is the list of working arms or the string ``"untucked"`` (both arms
        # idle); a stance solved in another posture for both arms (the tucked floor-init A of IORS large-scale, ``1dd3820``)
        # carries that posture in ``idle_posture`` below, which the oracle reads for every arm of a posture stance
        # (0919-233032: the first ontop large-scale demo refused the sidecar on ``arms: "tucked"``)
        d: Dict[str, Any] = {"base_pose2d": [float(v) for v in self.base_pose2d], "translate": float(self.translate), "tilt": float(self.tilt),
                             "arms": self.arms if isinstance(self.arms, (list, tuple)) else "untucked"}
        if self.stand_off is not None:
            d["stand_off"] = float(self.stand_off)
        if self.edge is not None:
            d["edge"] = self.edge.name
            d["lateral"] = float(self.lateral)
        if self.face is not None:
            d["face"] = self.face.name
            d["d"] = float(self.face.d)
            d["lambda"] = float(self.face.lam)
        if self.focus is not None:
            d["focus"] = _lst(self.focus)
        if self.contacts:
            d["contacts"] = list(self.contacts)
        if self.carried is not None:
            d["carried"] = self.carried
        # A6: the idle arm's posture (the oracle moves the arm there before the trunk motion); its joint values follow
        # as ``arms_q`` once core annotated the stance (``SamplerV2.annotate_stance_arms``)
        if self.idle_arm is not None:
            d["idle_posture"] = self.idle_posture or "untucked"
        elif self.posture != "untucked":
            d["idle_posture"] = self.posture
        for k, v in self.extras.items():
            if k not in d and not str(k).startswith("_"):
                d[k] = _jsonable(v)
        return d

    def arm_joints(self, arm: str, arm_control_idx) -> Optional[th.Tensor]:
        """The recorded joint vector of ``arm`` when the stance carries a configuration (``q``), else None."""
        if self.q is None:
            return None
        return G_index(self.q, arm_control_idx[arm])

    def to_meta(self) -> Dict[str, Any]:
        """Everything (for ``template_meta.json`` / summaries): the sidecar block plus the frames."""
        d = self.to_sidecar()
        d["name"] = self.name
        if self.edge is not None:
            d["edge_frame"] = self.edge.to_json()
            d["edge_depth"] = self.edge_depth
        if self.face is not None:
            d["face_frame"] = self.face.to_json()
        if self.extras:
            d["extras"] = _jsonable(self.extras)
        return d

    #: the ``stances.<name>`` keys :meth:`to_sidecar` writes from the stance's own fields; every other key is an extra
    SIDECAR_FIELDS = ("base_pose2d", "translate", "tilt", "arms", "stand_off", "edge", "lateral", "face", "d", "lambda", "focus", "contacts", "carried",
                      "idle_posture")

    @staticmethod
    def from_sidecar(name: str, d: Dict[str, Any], edge: Optional[EdgeFrame] = None, face: Optional[FaceFrame] = None) -> "Stance":
        """The stance a written sidecar recorded, with its extras - among them ``arms_q``, the arm joint values the sampler's
        checks used (:func:`ik.stance_arm_vectors` reads them when there is no recorded ``q``) - ``fingers`` and
        ``idle_posture``."""
        extras = {k: v for k, v in d.items() if k not in Stance.SIDECAR_FIELDS}
        return Stance(name, tuple(float(v) for v in d["base_pose2d"]), float(d.get("translate", 0.5)), float(d.get("tilt", 0.0)), d.get("stand_off"),
                      edge, face, float(d.get("lateral", 0.0)), None if d.get("focus") is None else _t(d["focus"]), d.get("arms", "untucked"),
                      list(d.get("contacts", [])), None, d.get("carried"), extras, None if d.get("fingers") is None else float(d["fingers"]),
                      d.get("idle_posture"))

    # -- cache round trip (2026-10-31 train-fast, SPA_CACHE: the lateral-cell memo of IORS small's start per attempt) -------------
    #: the key under which :meth:`to_cache` keeps the extras :meth:`to_sidecar` leaves out (``_``-prefixed): ``_position`` is the split
    #: cone's lattice position of a Phase-2 candidate (``tabletop.split_cone_draw`` groups the candidates by it, ``core.record_stance_travel``
    #: reads it as the edge-travel mark; ints, kept a tuple)
    CACHE_PRIVATE_EXTRAS_KEY = "_private_extras"

    def to_cache(self) -> Dict[str, Any]:
        """Everything :meth:`from_cache` needs to rebuild this stance as the sampler holds it in memory: the sidecar block of
        :meth:`to_sidecar` plus what that block leaves out - ``name``, the raw ``arms`` (a posture string stays itself), ``fingers``,
        ``idle_posture`` as is (None stays None, the sidecar's ``"untucked"`` default is not written) and the private ``_``-prefixed
        extras. The edge / face frames are not included (the caller keeps them and passes them back); ``q`` is never cached."""
        d = self.to_sidecar()
        d["name"] = self.name
        d["arms"] = list(self.arms) if isinstance(self.arms, (list, tuple)) else self.arms
        if self.fingers is not None:
            d["fingers"] = float(self.fingers)
        if self.idle_posture is None:
            d.pop("idle_posture", None)
        else:
            d["idle_posture"] = self.idle_posture
        private = {str(k): _jsonable(v) for k, v in self.extras.items() if str(k).startswith("_")}
        if private:
            d[self.CACHE_PRIVATE_EXTRAS_KEY] = private
        return d

    @staticmethod
    def from_cache(d: Dict[str, Any], edge: Optional[EdgeFrame] = None, face: Optional[FaceFrame] = None) -> "Stance":
        """The inverse of :meth:`to_cache`: a fresh stance whose containers are its own (``d`` is deep-copied: the nested ``start_region`` /
        ``start_region_arms`` lists are never shared with the cached dict, so later mutations - ``drop_candidate``, the idle-posture retry -
        stay with the copy), ``focus`` a tensor again and ``_position`` a tuple of ints again."""
        d = copy.deepcopy(d)
        private = d.pop(Stance.CACHE_PRIVATE_EXTRAS_KEY, None) or {}
        name = str(d.pop("name", "stance"))
        st = Stance.from_sidecar(name, d, edge=edge, face=face)
        st.extras.pop("fingers", None)  # a field, not an extra (from_sidecar keeps every non-field key as an extra)
        for k, v in private.items():
            st.extras[k] = tuple(int(i) for i in v) if k == "_position" and isinstance(v, (list, tuple)) else v
        return st


# ----------------------------------------------------------------------------------------------------------------
# Transition / path
# ----------------------------------------------------------------------------------------------------------------

TRANSITION_ORDER = ("drive-back", "trunk", "drive-in")


@dataclass
class Transition:
    """Spec §3.8: the trunk change between two consecutive stances happens at stand-off ``stand_off`` (s_T), after
    driving back from the arriving stance along -inward and before driving in again.

    ``at`` (owner 2026‑09‑19 21:56, IORS large-scale): ``"arrival"`` = the §3.8 transition above (s_T on the *arriving*
    stance's axis, trunk = the arriving stance's); ``"departure"`` = the mirror image at the *departing* stance — drive
    back ``s_T - frm.stand_off`` along the departing stance's own -forward (the way it arrived), sweep the trunk to the
    **travel trunk** ``trunk_to`` (torso straight, tilt 0), no drive-in; the recorded path ``frm -> to`` then starts at
    that drive-back pose and is driven at ``trunk_to``. A pair of stances may record one of each (departure first)."""

    frm: str
    to: str
    stand_off: float
    order: Tuple[str, ...] = TRANSITION_ORDER
    trunk_from: Optional[Tuple[float, float]] = None
    trunk_to: Optional[Tuple[float, float]] = None
    n_samples: int = 5
    holding: Optional[str] = None  # bddl instance carried through the transition (attachment: the second base pose after the male grasp)
    extras: Dict[str, Any] = field(default_factory=dict)
    at: str = "arrival"

    @property
    def departure(self) -> bool:
        return self.at == "departure"

    def to_sidecar(self) -> Dict[str, Any]:
        d = {"from": self.frm, "to": self.to, "stand_off": float(self.stand_off), "order": list(self.order)}
        if self.holding is not None:
            d["holding"] = self.holding
        if self.departure:
            d["at"] = "departure"
            if self.trunk_to is not None:
                d["trunk_to"] = [float(self.trunk_to[0]), float(self.trunk_to[1])]
        for k, v in (self.extras or {}).items():
            if k not in d and v is not None and not str(k).startswith("_"):
                d[k] = v
        return d


@dataclass
class PathRecord:
    """Spec §3.7 / §7 ``paths[]``: the base path the oracle drives, ``mode`` ``local`` (un-eroded shortest path),
    ``global`` (eroded global path + straight local connections) or ``edge_slide`` (owner 2026-09-24: drive back / slide
    along the support edge / drive in between two stances at the same edge, the body facing the edge throughout - every
    waypoint carries the arriving yaw, so the holonomic base slides without turning), ``waypoints`` [[x, y, yaw], ...] world, ``length_m``,
    ``posture`` (``untucked``), ``carried`` bddl instance or None, ``trunk`` (translate, tilt) the checks were made at.
    ``style`` (optional, ``robot.base_path.STYLES``): how the oracle drives the waypoints — ``face_travel`` = the shared
    ``robot.base_path`` schedule the record was collision-checked as (camera relocalization, owner 2026-09-22); absent =
    the polyline as recorded."""

    frm: str
    to: str
    mode: str
    waypoints: List[Tuple[float, float, float]]
    length_m: float
    posture: str = "untucked"
    carried: Optional[str] = None
    trunk: Optional[Tuple[float, float]] = None
    stride_m: float = 0.1
    n_checked: int = 0
    style: Optional[str] = None
    #: owner 2026-09-24: the arm joints (arm -> 7 values) the drive was swept with when they are not the stance posture - the arm
    #: left where the previous motion ended it (``program_check.choose_departure_arms``); the oracle re-poses only a deviating arm
    arms_q: Optional[Dict[str, List[float]]] = None
    #: owner 2026-10-13 (disembedding): the leg into the place pose keeps the heading and moves only forward / backward or left /
    #: right (``navigation.plan_axis_leg``): ``{"back_m": the drive-back used, "tried_back_m": [...], "moves": [[axis, m], ...]}``
    axis_leg: Optional[Dict[str, Any]] = None
    #: oracle-fix 2026-11-06 (``generation.face_travel_fallback_cfgs``): the ``robot.base_path.BasePathConfig`` overrides of the
    #: face_travel schedule the record was checked as when the default schedule collided (e.g. simplify / fillet 3 cm); the
    #: oracle builds its schedule with them (``robot.base_path.config_with``); None = the default schedule
    face_travel_cfg: Optional[Dict[str, float]] = None

    def to_sidecar(self) -> Dict[str, Any]:
        out = {"from": self.frm, "to": self.to, "mode": self.mode, "waypoints": [[float(a) for a in w] for w in self.waypoints],
               "length_m": float(self.length_m), "posture": self.posture, "carried": self.carried}
        if self.style is not None:
            out["style"] = self.style
        if self.style == "face_travel" and self.face_travel_cfg:
            out["face_travel_cfg"] = {str(k): float(v) for k, v in self.face_travel_cfg.items()}
        if self.arms_q:
            out["arms_q"] = {str(a): [float(v) for v in q] for a, q in self.arms_q.items()}
        if self.axis_leg:
            out["axis_leg"] = dict(self.axis_leg)
        return out

    @staticmethod
    def schema_ok(d: Dict[str, Any]) -> bool:
        """Whether a dict is a well-formed ``paths[]`` entry (used by tests and the program-check stub)."""
        try:
            return (d["mode"] in ("local", "global", "edge_slide") and isinstance(d["waypoints"], list) and len(d["waypoints"]) >= 1
                    and all(len(w) == 3 for w in d["waypoints"]) and float(d["length_m"]) >= 0.0 and isinstance(d["posture"], str)
                    and isinstance(d["from"], str) and isinstance(d["to"], str) and "carried" in d)
        except (KeyError, TypeError, ValueError):
            return False


def path_length(waypoints: Sequence[Sequence[float]]) -> float:
    n = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return n


# ----------------------------------------------------------------------------------------------------------------
# Grasp / contact records
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class GraspRecord:
    """Spec §7 ``grasps[]``: the drawn library grasp of one object plus the eef poses derived from it at the actual
    object pose (world): pre-grasp, grasp, pre-place, place; ``carry_height`` = lift above the grasp pose."""

    object: str
    arm: str
    grasp_key: Optional[str]
    grasp_name: Optional[str]
    yaw: float
    pre_grasp: Optional[Pose] = None
    grasp: Optional[Pose] = None
    pre_place: Optional[Pose] = None
    place: Optional[Pose] = None
    carry_height: float = 0.12
    finger_q: Optional[List[float]] = None
    eef_in_object: Optional[Pose] = None
    stance: str = "manipulation"
    place_on: Optional[str] = None  # bddl instance the object is placed on / into / attached to (F6 programs)
    waypoints: Optional[List[Pose]] = None  # in-hand motion key poses (held-object tasks, attachment), last = end pose
    carry: Optional[List[Pose]] = None  # FOV rule (b): waypoint(s) of the lift-end -> pre-place carry leg when the straight leg fails
    extras: Dict[str, Any] = field(default_factory=dict)  # diagnostics / task facts (F6 keeps unknown keys in ``extra``)

    def to_sidecar(self) -> Dict[str, Any]:
        d = {"object": self.object, "arm": self.arm, "grasp_key": self.grasp_key, "grasp_name": self.grasp_name, "yaw": float(self.yaw),
             "grasp": pose_to_sidecar(self.grasp), "carry_height": float(self.carry_height), "stance": self.stance}
        for k in ("pre_grasp", "pre_place", "place"):
            v = getattr(self, k)
            if v is not None:
                d[k] = pose_to_sidecar(v)
        if self.place_on is not None:
            d["place_on"] = self.place_on
        if self.waypoints is not None:
            d["waypoints"] = [pose_to_sidecar(p) for p in self.waypoints]
        if self.carry is not None:
            d["carry"] = [pose_to_sidecar(p) for p in self.carry]
        if self.finger_q is not None:
            d["finger_q"] = [float(v) for v in self.finger_q]
        if self.eef_in_object is not None:
            d["eef_in_object"] = pose_to_sidecar(self.eef_in_object)
        for k, v in self.extras.items():
            if k not in d and not str(k).startswith("_"):
                d[k] = v
        return d


@dataclass
class ContactRecord:
    """Spec §7 ``contacts.push`` / ``contacts.brace``: arm, kind, pre-contact and contact eef poses (world), direction
    (unit, world) and distance for pushes, the ``mid`` / ``goal`` poses of a push arc, the object touched."""

    kind: str  # "push" | "brace"
    arm: str
    pre: Optional[Pose] = None
    contact: Optional[Pose] = None
    mid: Optional[Pose] = None
    goal: Optional[Pose] = None
    direction: Optional[th.Tensor] = None
    distance_m: Optional[float] = None
    brace_kind: Optional[str] = None
    object: Optional[str] = None
    stance: str = "manipulation"
    extras: Dict[str, Any] = field(default_factory=dict)  # blade / pose-class facts (F6 keeps unknown keys in ``extra``)

    def to_sidecar(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"arm": self.arm, "pre": pose_to_sidecar(self.pre), "stance": self.stance}
        if self.object is not None:
            d["object"] = self.object
        if self.kind.startswith("push"):  # "push" and the extra push slots ("push_prep": the preparatory press of a two-flap carton, 2026-09-19)
            d.update({"init": pose_to_sidecar(self.contact), "mid": pose_to_sidecar(self.mid), "goal": pose_to_sidecar(self.goal),
                      "direction": _lst(self.direction), "distance_m": None if self.distance_m is None else float(self.distance_m)})
        else:
            d.update({"kind": self.brace_kind, "contact": pose_to_sidecar(self.contact), "direction": _lst(self.direction)})
        for k, v in (self.extras or {}).items():
            if k not in d and v is not None and not str(k).startswith("_"):
                d[k] = v
        return d


# ----------------------------------------------------------------------------------------------------------------
# Sidecar draft (F6 adapter)
# ----------------------------------------------------------------------------------------------------------------


class SidecarDraft:
    """Dict-backed draft of the §7 sidecar filled while sampling; ``finalize(iid, tro_path)`` adds ``tro_sha256``.

    Blocks: ``stances`` (name -> Stance), ``transitions`` (list), ``paths`` (list), ``grasps`` (list), ``contacts``
    (``push`` / ``brace`` -> ContactRecord), ``articulation``, ``blocks``, ``goal``, ``language``, ``waypoint_policy``.
    Only the blocks a task filled are written. The task modules add records through the ``add_*`` methods; ``core``
    and ``program_check`` read them back with the typed accessors.
    """

    def __init__(self, variant: str, task_family: str, instance_id: str = "draft"):
        self.variant = variant
        self.task_family = task_family
        self.instance_id = instance_id
        self.path_bucket_m: Optional[float] = None
        self.stances: Dict[str, Stance] = {}
        self.transitions: List[Transition] = []
        self.paths: List[PathRecord] = []
        self.grasps: List[GraspRecord] = []
        self.contacts: Dict[str, ContactRecord] = {}
        self.articulation: Optional[Dict[str, Any]] = None
        self.blocks: List[Dict[str, Any]] = []
        self.goal: Optional[Dict[str, Any]] = None
        self.language: Dict[str, Any] = {"objects": {}, "goal_numbers": {}}
        self.waypoint_policy: str = "checked"
        self.checked_waypoints: List[Dict[str, Any]] = []  # what §3.9 checked (debug / summary)
        #: the program check's joint solutions per action end pose (``program_check.run``): D5 verifies them as witnesses
        self.ik_witnesses: List[Dict[str, Any]] = []
        #: task-specific top-level facts the oracle reads through ``Sidecar.extra`` (attachment: ``attach_correction``, the
        #: per-task correction regime of ``generation.attach_correction``; owner 2026-09-20 23:39)
        self.extra: Dict[str, Any] = {}
        self.tro_sha256: Optional[str] = None

    # -- writers ------------------------------------------------------------------------------------------------
    def add_stance(self, stance: Stance) -> Stance:
        self.stances[stance.name] = stance
        return stance

    def add_transition(self, tr: Transition) -> Transition:
        self.transitions.append(tr)
        return tr

    def add_path(self, path: PathRecord) -> PathRecord:
        self.paths.append(path)
        return path

    def add_grasp(self, g: GraspRecord) -> GraspRecord:
        self.grasps.append(g)
        return g

    def add_contact(self, c: ContactRecord) -> ContactRecord:
        self.contacts[c.kind] = c
        return c

    def set_goal(self, base_pose2d, translate: float, tilt: float, arms: str = "same") -> Dict[str, Any]:
        self.goal = {"base_pose2d": [float(v) for v in base_pose2d], "translate": float(translate), "tilt": float(tilt), "arms": arms}
        return self.goal

    def describe(self, instance: str, display_name: str, descriptor: Optional[str] = None, **numbers):
        self.language["objects"][instance] = {"display_name": display_name, "descriptor": descriptor or display_name}
        self.language["goal_numbers"].update({k: float(v) for k, v in numbers.items()})

    # -- readers ------------------------------------------------------------------------------------------------
    def stance(self, name: str) -> Stance:
        if name not in self.stances:
            raise KeyError(f"sidecar has no stance {name!r}; have {sorted(self.stances)}")
        return self.stances[name]

    def grasp_of(self, obj: str) -> Optional[GraspRecord]:
        for g in self.grasps:
            if g.object == obj:
                return g
        return None

    # -- serialisation ------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"schema": SIDECAR_SCHEMA, "instance_id": self.instance_id, "tro_sha256": self.tro_sha256, "task_family": self.task_family,
                             "variant": self.variant, "path_bucket_m": self.path_bucket_m,
                             "stances": {k: v.to_sidecar() for k, v in self.stances.items()}}
        if self.transitions:
            d["transitions"] = [t.to_sidecar() for t in self.transitions]
        if self.paths:
            d["paths"] = [p.to_sidecar() for p in self.paths]
        if self.grasps:
            d["grasps"] = [g.to_sidecar() for g in self.grasps]
        if self.contacts:
            d["contacts"] = {k: v.to_sidecar() for k, v in self.contacts.items()}
        if self.articulation:
            d["articulation"] = _jsonable(self.articulation)
        if self.blocks:
            d["blocks"] = _jsonable(self.blocks)
        if self.goal:
            d["goal"] = dict(self.goal)
        d["language"] = _jsonable(self.language)
        d["waypoint_policy"] = self.waypoint_policy
        if self.ik_witnesses:
            d["ik_witnesses"] = _jsonable(self.ik_witnesses)
        for k, v in (self.extra or {}).items():
            d.setdefault(k, _jsonable(v))
        return d

    def finalize(self, instance_id: str, tro_path: str) -> Dict[str, Any]:
        self.instance_id = instance_id
        h = hashlib.sha256()
        with open(tro_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.tro_sha256 = h.hexdigest()
        return self.to_dict()

    def write(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def reset_instance(self):
        """Drop the per-instance records (keep nothing: every attempt fills the draft anew)."""
        self.stances.clear()
        self.transitions.clear()
        self.paths.clear()
        self.grasps.clear()
        self.contacts.clear()
        self.articulation = None
        self.blocks = []
        self.goal = None
        self.language = {"objects": {}, "goal_numbers": {}}
        self.checked_waypoints = []
        self.ik_witnesses = []
        self.extra = {}
        self.path_bucket_m = None
        self.tro_sha256 = None


def G_index(q: th.Tensor, idx) -> th.Tensor:
    return q[idx].clone()


# ----------------------------------------------------------------------------------------------------------------
# Band membership (A1 joint placement; pure, base frame: x ahead, lateral + = robot's left)
# ----------------------------------------------------------------------------------------------------------------

BAND_CELL_TOL = 0.06  # a point counts as "in the band" within this distance of a band cell (8 cm lattice: half a diagonal)


def in_band(point_base, cells, tol: float = BAND_CELL_TOL) -> bool:
    """Whether a base-frame point lies within ``tol`` of a cell of a reach band (N x 2 base-frame cells, numpy or torch);
    an empty / missing band admits nothing."""
    if cells is None or len(cells) == 0:
        return False
    c = th.as_tensor(np.asarray(cells, dtype=float)).reshape(-1, 2)
    p = th.tensor([float(point_base[0]), float(point_base[1])], dtype=c.dtype)
    return bool(th.min(th.linalg.norm(c - p, dim=1)) <= tol)


def brace_placement_admissible(work_point_base, brace_point_base, right_cells, left_cells, tol: float = BAND_CELL_TOL) -> bool:
    """A1 (owner decision 2026-09-16): the pushed end / work point of the right arm lies in the **right** band and the
    brace point of the left arm in the **left** contact band *of the same trunk*; the exact two-arm IK of §4.6 step 6
    follows. Used by touching (partner far side), axial (stop edge), attachment (partner brace), block design (braces)."""
    return in_band(work_point_base, right_cells, tol) and in_band(brace_point_base, left_cells, tol)


def nearer_side_lateral(lat_init: float, partner_lat: Tuple[float, float], target_lat: Tuple[float, float], gap: float) -> Tuple[str, float]:
    """§6.11 nextto / touching goal (owner review 2026-09-16): the target goes on the partner side that **faces its init
    position**, so it is never lifted over the partner. Laterals are along the stance tangent (+ = robot's left):
    ``partner_lat`` = (min, max) of the partner hull about the base, ``target_lat`` = (min, max) of the target hull about
    its own origin, ``gap`` = the AABB gap to leave. Returns ``("left" | "right", lateral of the target origin)``: the
    target's near face sits ``gap`` outside the partner's facing face."""
    p_min, p_max = float(partner_lat[0]), float(partner_lat[1])
    t_min, t_max = float(target_lat[0]), float(target_lat[1])
    if float(lat_init) > 0.5 * (p_min + p_max):
        return "left", p_max + float(gap) - t_min
    return "right", p_min - float(gap) - t_max


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, th.Tensor):
        return x.tolist()
    if hasattr(x, "to_sidecar"):
        return x.to_sidecar()
    if hasattr(x, "__dataclass_fields__"):
        return _jsonable(asdict(x))
    return x
