"""Layer 2 of the v2 oracle: the action vocabulary (``oracle_primitives_design.md`` §3.6, behaviour §6.1).

An action is the unit of checkpoint, annotation and QA in the executor loop (§3.8). This module (TODO F6) declares
the contract and the vocabulary — ``Navigate``, ``Transition``, ``SetTorso``, ``Grasp``, ``Lift``, ``Reorient``,
``Place``, ``Push``, ``Brace``, ``BaseFollow``, ``Retreat`` — and implements ``targets()`` and ``annotate()``;
``execute()`` / ``postcondition()`` are declared and implemented by F2 (Layer-1 motions).

``targets(sidecar, state) -> dict`` resolves every pose / waypoint list / base path the action will track from the
sidecar facts through the shared conventions module (``ispatialgym.robot.conventions``, TODO F5, §3.5) — never from
a copy of its geometry. The sampler's program check (spec §3.5) IK- and collision-checks every value of the returned
dict, so the keys are fixed per class (``TARGET_KEYS``) and documented in each class docstring. Values are
:class:`~ispatialgym.oracle.sidecar.Pose` (world frame, quaternion xyzw), lists of poses, ``[x, y, yaw]`` base poses
or ``(translate, tilt)`` trunk pairs; ``None`` marks a value that does not apply (e.g. ``base_poses`` in arm mode).

The only privileged reads ``targets()`` makes go through ``state`` (:class:`State`): the world pose of a bddl
instance (object poses at init for the sampler's check, the restored state for a retry). Everything else is a
sidecar fact, so a program stays pure and the chain of eef poses through consecutive actions is derived, not
measured (``end_pose()``); the tracking residual at run time is closed by the Layer-1 motions (correction, not a
fallback, §3 rules).

Conventions used (``ispatialgym/robot/conventions.py``; tensors are float32 CPU, poses ``(pos (3,), quat_xyzw (4,))``
tuples — :func:`_as_pose` also accepts ``FramedPose``, ``{"position", "orientation"}`` dicts and 7-vectors). Every
waypoint generator returns the poses at fractions k/n, k = 1..n (start excluded, goal last):

=========================================================  ========================================================
``grasp_pose(obj_pose, grasp_key, grasp_name, yaw=0)``      library grasp at the object's *current* pose (yaw 0: the
                                                            recorded ``yaw`` was the sampler's draw, the settled pose
                                                            already carries it) — ``Grasp[library]``
``cube_grasp(block_pose, face_pair, tilt_pick_deg, yaw)``   ``.eef`` of the cube grasp — ``Grasp[cube]``
``handle_eef(contact, approach, pinch)``                    eef of a recorded handle waypoint h(q_k) (contact −
                                                            0.012·a) — ``Grasp[handle]``, ``BaseFollow``
``pre_pose(pose, dist)``                                    back along −a — pre-grasp / pre-place / retreats
``line_waypoints(p0, p1, n)``                               straight segments (approach, lift, descent, retreat,
                                                            straight pushes, the held object's camera ray)
``slerp_waypoints(q0, q1, n)``                              object orientation path of an in-hand rotation +
                                                            translation (attachment)
``orbit_waypoints(pose, pivot, delta_quat, n)``             rotation about the object centre (mental rotation, block
                                                            yaw-align)
``flip_waypoints(eef, pivot, flip_axis, flip_sign, yaw)``   block yaw-align + 90° flip (``Reorient[flip]``)
``transition_poses(stance_from, stance_to, s_T)``           drive-back / trunk / drive-in of a ``Transition``
``eef_base_transform(eef, base)``, ``base_follow_poses``    b_k = h(q_k) ∘ T_eef→base(q*) (``BaseFollow`` base mode)
=========================================================  ========================================================

Two §3.5 functions are the sampler's and not re-evaluated here because spec §7 records their *outputs*:
``handle_waypoints`` (the sidecar carries the moved contact / approach / pinch per k, turned into eef poses with
``handle_eef``) and ``arc_waypoints`` (the push record carries ``mid`` / ``goal``, not the arc centre / axis / δ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

from ispatialgym.oracle.sidecar import (
    ARMS,
    Articulation,
    BlockRecord,
    BraceContact,
    Goal,
    GraspRecord,
    Path as PathRecord,
    Pose,
    PushContact,
    Sidecar,
    SidecarError,
    Stance,
    Transition as TransitionRecord,
)

# Geometry constants of the action behaviour (§6.1, spec §3.6); the eef offsets themselves live in the conventions.
PRE_DIST = 0.10  # pre-grasp / pre-contact / pre-place stand-off along −a
RETREAT_DIST = 0.10  # retreat along −a after a release / contact
RETREAT_LITTLE = 0.05  # "move contacting grippers away a little" after a place (§6.1)
N_WAYPOINTS = 3  # intermediate waypoints of a straight / slerp segment (spec §3.9)

QA_FAILED_GRASP = "failed_grasp"


class State(Protocol):
    """The privileged state ``targets()`` may read (the sampler's world at init, the executor's restored state)."""

    def object_pose(self, name: str) -> Any:
        """World pose of the bddl instance ``name`` as ``(pos, quat_xyzw)`` tensors (or a :class:`Pose`)."""


# ----------------------------------------------------------------------------------------------------------------
# Small pure helpers: quaternions (xyzw), pose composition, conventions adapters
# ----------------------------------------------------------------------------------------------------------------


def _conventions():
    """The shared conventions module (imported lazily: F5 lands in parallel and pulls in torch)."""
    from ispatialgym.robot import conventions as C  # noqa: N812

    return C


def _to_list(x: Any) -> List[float]:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _as_pose(x: Any) -> Pose:
    """Normalise a conventions / state return value to :class:`Pose`."""
    if isinstance(x, Pose):
        return x
    if isinstance(x, dict):
        return Pose.from_dict(x, "conventions")
    if hasattr(x, "pos") and hasattr(x, "quat"):
        return Pose(_to_list(x.pos), _to_list(x.quat))
    if hasattr(x, "position") and hasattr(x, "orientation"):
        return Pose(_to_list(x.position), _to_list(x.orientation))
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return Pose(_to_list(x[0]), _to_list(x[1]))
    flat = _to_list(x)
    if len(flat) == 7:
        return Pose(flat[:3], flat[3:])
    raise TypeError(f"cannot interpret {type(x).__name__} as a pose")


def _as_pose_list(x: Any, quat: Optional[Sequence[float]] = None) -> List[Pose]:
    """Normalise a waypoint list: ``(n, 3)`` positions (with ``quat``), ``(n, 7)`` rows, or a sequence of poses."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist") and hasattr(x, "shape") and len(getattr(x, "shape", ())) == 2:
        rows = x.tolist()
        width = len(rows[0]) if rows else 0
        if width == 3:
            if quat is None:
                raise TypeError("position waypoints need an orientation")
            return [Pose([float(v) for v in r], list(quat)) for r in rows]
        if width == 7:
            return [Pose([float(v) for v in r[:3]], [float(v) for v in r[3:]]) for r in rows]
        raise TypeError(f"cannot interpret waypoints of width {width}")
    out = []
    for item in x:
        if isinstance(item, (Pose, dict)) or hasattr(item, "pos") or hasattr(item, "position"):
            out.append(_as_pose(item))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append(_as_pose(item))
            continue
        flat = _to_list(item)
        if len(flat) == 3 and quat is not None:
            out.append(Pose(flat, list(quat)))
        elif len(flat) == 7:
            out.append(Pose(flat[:3], flat[3:]))
        else:
            raise TypeError(f"cannot interpret a waypoint of {len(flat)} values")
    return out


def _as_pose2d(x: Any) -> List[float]:
    v = _to_list(x)
    if len(v) != 3:
        raise TypeError(f"expected [x, y, yaw], got {len(v)} values")
    return v


def _as_pose2d_list(x: Any) -> List[List[float]]:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist") and hasattr(x, "shape") and len(getattr(x, "shape", ())) == 2:
        return [[float(v) for v in r] for r in x.tolist()]
    return [_as_pose2d(r) for r in x]


def _tensor(v: Sequence[float]):
    import torch as th

    return th.tensor([float(x) for x in v], dtype=th.float32)


def _handle_eef(C, wp) -> Pose:
    """eef pose of a recorded handle waypoint h(q_k) (``HandleWaypoint`` record) through ``conventions.handle_eef``."""
    return _as_pose(C.handle_eef(_tensor(wp.contact), approach=_tensor(wp.approach), closing=_tensor(wp.pinch)))


def _qmul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _qconj(q: Sequence[float]) -> List[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _qrot(q: Sequence[float], v: Sequence[float]) -> List[float]:
    return _qmul(_qmul(list(q), [v[0], v[1], v[2], 0.0]), _qconj(q))[:3]


def _compose(a: Pose, b: Pose) -> Pose:
    """T_a ∘ T_b (apply b in a's frame)."""
    p = _qrot(a.orientation, b.position)
    return Pose([a.position[i] + p[i] for i in range(3)], _qmul(a.orientation, b.orientation))


def _inverse(a: Pose) -> Pose:
    qi = _qconj(a.orientation)
    p = _qrot(qi, a.position)
    return Pose([-v for v in p], qi)


def _relative(target: Pose, frame: Pose) -> Pose:
    """``target`` expressed in ``frame`` (both in the same parent frame)."""
    return _compose(_inverse(frame), target)


def _quat_angle(q1: Sequence[float], q2: Sequence[float]) -> float:
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, d)))


def _distance(p: Sequence[float], q: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def _yaw_pose(pose2d: Sequence[float]) -> Pose:
    x, y, yaw = pose2d
    return Pose([float(x), float(y), 0.0], [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def _state_pose(state: Any, name: str) -> Pose:
    if state is None or not hasattr(state, "object_pose"):
        raise SidecarError(f"targets() needs a state with object_pose() to locate {name!r}")
    return _as_pose(state.object_pose(name))


def _line(C, a: Pose, b: Pose, n: int = N_WAYPOINTS) -> List[Pose]:
    """Straight eef segment a → b (positions from the conventions, orientation of ``b``)."""
    pts = C.line_waypoints(_tensor(a.position), _tensor(b.position), n)
    return _as_pose_list(pts, quat=b.orientation)


def _pre(C, pose: Pose, dist: float) -> Pose:
    return _as_pose(C.pre_pose(pose.tensors(), dist))


def _hand(arm: str) -> str:
    return f"{arm} hand"


def _cm(m: float) -> str:
    cm = abs(m) * 100.0
    return f"{cm:.1f} cm" if cm < 10.0 else f"{cm:.0f} cm"


def _deg(rad: float) -> str:
    return f"{math.degrees(rad):.0f}°"


def _xyz(p: Sequence[float]) -> str:
    return f"x={p[0]:.2f} m, y={p[1]:.2f} m, z={p[2]:.2f} m"


def direction_word(direction: Sequence[float], base_yaw: Optional[float]) -> str:
    """Dominant robot-relative direction word of a world-frame direction (``base_yaw`` = the stance yaw; without it
    the world axes are named)."""
    vx, vy, vz = direction
    if base_yaw is None:
        comps = [(abs(vx), "+x" if vx > 0 else "-x"), (abs(vy), "+y" if vy > 0 else "-y"), (abs(vz), "up" if vz > 0 else "down")]
        return max(comps)[1]
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    fx = c * vx + s * vy
    fy = -s * vx + c * vy
    comps = [
        (abs(fx), "forward" if fx > 0 else "backward"),
        (abs(fy), "to the left" if fy > 0 else "to the right"),
        (abs(vz), "up" if vz > 0 else "down"),
    ]
    return max(comps)[1]


_DEST_WORDS = {
    "init": "the initial position",
    "manipulation": "the manipulation stance",
    "placement": "the placement stance",
    "goal": "the goal position",
}


def _dest(name: str) -> str:
    return _DEST_WORDS.get(name, f"the {name} position")


# ----------------------------------------------------------------------------------------------------------------
# Contract
# ----------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Action:
    """One row of the executor loop (§3.6): one checkpoint, one annotation pair, one QA verdict.

    ``arms``: arms involved (QA gripper-in-FOV monitor; failed-grasp exemption). ``objects``: bddl instances involved
    (annotation; the intended-contact whitelist of the collision monitor — never list an object the action must not
    touch). ``qa_exempt``: QA categories of §6.4 that do not apply (``{"failed_grasp"}`` for ``Push`` / ``Brace``).
    """

    name: str
    arms: Tuple[str, ...] = ()
    objects: Tuple[str, ...] = ()
    qa_exempt: frozenset = frozenset()

    TARGET_KEYS: ClassVar[Tuple[str, ...]] = ()

    def targets(self, sidecar: Sidecar, state: Optional[State]) -> Dict[str, Any]:
        """Poses / waypoints / paths the action tracks, via the conventions (§3.5); pure, keys = ``TARGET_KEYS``."""
        raise NotImplementedError

    def execute(self, motions: Any, state: Any) -> Iterator[Any]:
        """Layer-1 motions only; yields 23-D JoyLo actions (implemented by TODO F2)."""
        raise NotImplementedError(f"{type(self).__name__}.execute is implemented by TODO F2 (Layer-1 motions)")

    def postcondition(self, state: Any) -> Optional[str]:
        """``None`` = ok, else a QA category of §6.4 (implemented by TODO F2)."""
        raise NotImplementedError(f"{type(self).__name__}.postcondition is implemented by TODO F2")

    def annotate(self, sidecar: Sidecar) -> Tuple[str, str]:
        """``(fine-grained text, the same with the concrete numbers)`` over the involved objects' display names /
        descriptors (§6.3)."""
        raise NotImplementedError

    def end_pose(self, sidecar: Sidecar, state: Optional[State]) -> Optional[Pose]:
        """Where the action leaves the working arm's eef (``None`` for base / trunk actions); chains ``targets()``."""
        return None

    def __post_init__(self) -> None:
        for arm in self.arms:
            if arm not in ARMS:
                raise ValueError(f"{self.name}: unknown arm {arm!r}")
        self.arms = tuple(self.arms)
        self.objects = tuple(self.objects)
        self.qa_exempt = frozenset(self.qa_exempt)

    def summary(self) -> str:
        return f"{self.name}[{','.join(self.arms)}]({', '.join(self.objects)})"


# ----------------------------------------------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class Navigate(Action):
    """Follow a recorded base path (``paths[i]``; motion ``track_path``).

    targets: ``{"waypoints": [[x, y, yaw], ...], "goal": [x, y, yaw], "posture": str, "carried": str | None}`` —
    the recorded waypoints (whole-body collision-checked at ``posture`` with ``carried`` attached), copied from the
    record; the tracker follows them exactly (spec §3.7). ``obstacles`` (camera relocalization — obstructed) name
    the objects the path avoids for the annotation; they are deliberately **not** in ``objects`` (never touched).
    """

    name: str = "navigate"
    path: PathRecord
    obstacles: Tuple[str, ...] = ()
    TARGET_KEYS = ("waypoints", "goal", "posture", "carried")

    def targets(self, sidecar, state):
        return {
            "waypoints": [list(w) for w in self.path.waypoints],
            "goal": self.path.goal,
            "posture": self.path.posture,
            "carried": self.path.carried,
        }

    def annotate(self, sidecar):
        lang = sidecar.language
        dest = _dest(self.path.to)
        if self.path.carried:
            fine = f"carry {lang.phrase(self.path.carried)} to {dest}"
        elif self.obstacles:
            fine = f"drive around {' and '.join(lang.phrase(o) for o in self.obstacles)} to {dest}"
        else:
            fine = f"drive to {dest}"
        x, y, yaw = self.path.goal
        numbers = f"{fine} ({self.path.length_m:.2f} m recorded path to x={x:.2f} m, y={y:.2f} m, yaw={math.degrees(yaw):.0f}°)"
        return fine, numbers


@dataclass(kw_only=True)
class Transition(Action):
    """Stance transition (``transitions[i]`` + the two stances, spec §3.8): drive back to the transition stand-off
    s_T, move the trunk, drive in (motions ``track_path`` → ``trunk_to`` → ``track_path``).

    targets: ``{"drive_back": [x, y, yaw], "trunk": (translate, tilt), "drive_in": [x, y, yaw]}`` from
    ``conventions.transition_poses(stance_from, stance_to, s_T)``.
    """

    name: str = "transition"
    transition: TransitionRecord
    stance_from: Stance
    stance_to: Stance
    carried: Optional[str] = None
    TARGET_KEYS = ("drive_back", "trunk", "drive_in")

    def targets(self, sidecar, state):
        C = _conventions()
        tp = C.transition_poses(self.stance_from.to_dict(), self.stance_to.to_dict(), self.transition.stand_off)
        if hasattr(tp, "drive_back"):
            back, trunk, drive_in = tp.drive_back, tp.trunk, tp.drive_in
        else:
            back, trunk, drive_in = tp
        translate, tilt = _to_list(trunk)
        return {"drive_back": _as_pose2d(back), "trunk": (translate, tilt), "drive_in": _as_pose2d(drive_in)}

    def annotate(self, sidecar):
        purpose = "placing" if self.transition.to == "placement" else "manipulation"
        with_obj = f" with {sidecar.language.phrase(self.carried)}" if self.carried else ""
        fine = f"back off{with_obj}, set the trunk for {purpose} and move in to {_dest(self.transition.to)}"
        t0, k0 = self.stance_from.trunk
        t1, k1 = self.stance_to.trunk
        s = self.stance_to.stand_off
        arrive = f", drive in to {s:.2f} m stand-off" if s is not None else ", drive in"
        numbers = (
            f"drive back{with_obj} to {self.transition.stand_off:.2f} m stand-off, set trunk translate {t1:.2f} / tilt {k1:.2f}"
            f" (from {t0:.2f} / {k0:.2f}){arrive}"
        )
        return fine, numbers


@dataclass(kw_only=True)
class SetTorso(Action):
    """Camera-relocalization goal trunk (``goal.{translate, tilt, arms}``): lift the arms to untucked first when the
    sidecar says so (self-collision at the goal trunk), then ``trunk_to``; postcondition: head-camera pose within
    tolerance, one corrective ``track_path`` on the residual allowed (correction, not fallback).

    targets: ``{"base": [x, y, yaw], "trunk": (translate, tilt), "arms": "same" | "untucked"}``. The untucked joint
    vector is the robot constant ``untucked_default_joint_pos`` (spec §3.3), not a sidecar fact.
    """

    name: str = "set_torso"
    goal: Goal
    TARGET_KEYS = ("base", "trunk", "arms")

    def __post_init__(self):
        if self.goal.arms == "untucked":
            self.arms = ("left", "right")  # both arms lift to the untucked posture before the trunk moves
        super().__post_init__()

    def targets(self, sidecar, state):
        return {"base": list(self.goal.base_pose2d), "trunk": self.goal.trunk, "arms": self.goal.arms}

    def annotate(self, sidecar):
        lift = ", lifting both arms to the untucked posture first" if self.goal.arms == "untucked" else ""
        fine = f"set the torso to the goal camera pose{lift}"
        x, y, yaw = self.goal.base_pose2d
        numbers = (
            f"set trunk translate {self.goal.translate:.2f} / tilt {self.goal.tilt:.2f} at base x={x:.2f} m, y={y:.2f} m,"
            f" yaw={math.degrees(yaw):.0f}°{lift}"
        )
        return fine, numbers


@dataclass(kw_only=True)
class Grasp(Action):
    """Grasp an object (``grasps[i]``), a block (``blocks[i]`` cube grasp) or an articulation handle (``articulation``
    at q_init): ``gripper(open)`` → ``arm_to(pre_grasp)`` → ``arm_along(approach)`` → ``gripper(close)``;
    postcondition: assisted grasp attached (``failed_grasp`` otherwise).

    ``kind``: ``"library"`` (``grasp_pose(obj_pose, grasp_key, grasp_name)`` at the object's current pose — the
    recorded ``yaw`` was the sampler's draw and is already part of the settled pose), ``"cube"``
    (``cube_grasp(block_pose, face_pair, tilt_pick_deg, yaw).eef``), ``"handle"`` (``handle_eef`` of h(q_init)).

    targets: ``{"pre_grasp": Pose, "grasp": Pose, "approach": [Pose, ...]}`` — ``pre_grasp = pre_pose(grasp,
    PRE_DIST)``, ``approach`` = straight segment pre_grasp → grasp (``line_waypoints``).
    """

    name: str = "grasp"
    kind: str
    arm: str
    object: str
    grasp: Optional[GraspRecord] = None
    block: Optional[BlockRecord] = None
    articulation: Optional[Articulation] = None
    TARGET_KEYS = ("pre_grasp", "grasp", "approach")

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()
        needed = {"library": self.grasp, "cube": self.block, "handle": self.articulation}
        if self.kind not in needed:
            raise ValueError(f"grasp: unknown kind {self.kind!r}")
        if needed[self.kind] is None:
            raise ValueError(f"grasp[{self.kind}]: missing its record")

    def grasp_pose(self, sidecar, state) -> Pose:
        C = _conventions()
        if self.kind == "library":
            g = self.grasp
            obj = _state_pose(state, self.object)
            return _as_pose(C.grasp_pose(obj.tensors(), g.grasp_key, g.grasp_name))
        if self.kind == "cube":
            b = self.block
            obj = _state_pose(state, self.object)
            cube = C.cube_grasp(obj.tensors(), face_pair=b.face_pair, tilt_pick_deg=b.tilt_pick_deg, yaw=math.radians(b.yaw_deg))
            return _as_pose(getattr(cube, "eef", cube))
        art = self.articulation
        return _handle_eef(C, art.waypoints[art.key_at(0.0)])

    def targets(self, sidecar, state):
        C = _conventions()
        grasp = self.grasp_pose(sidecar, state)
        pre = _pre(C, grasp, PRE_DIST)
        return {"pre_grasp": pre, "grasp": grasp, "approach": _line(C, pre, grasp)}

    def end_pose(self, sidecar, state):
        return self.grasp_pose(sidecar, state)

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        approach = f"approach {_cm(PRE_DIST)} along the grasp axis"
        if self.kind == "handle":
            art = self.articulation
            fine = f"grasp the handle of {obj} with the {_hand(self.arm)}"
            numbers = f"{fine} (joint {art.joint} at q={art.q_init:.2f}, {approach})"
        elif self.kind == "cube":
            b = self.block
            fine = f"grasp {obj} with the {_hand(self.arm)}"
            numbers = f"{fine} (pinch across the {b.face_pair} faces, pick tilt {b.tilt_pick_deg:.0f}°, yaw {b.yaw_deg:.0f}°, {approach})"
        else:
            g = self.grasp
            fine = f"grasp {obj} with the {_hand(self.arm)}"
            numbers = f"{fine} ({g.grasp_name} of {g.grasp_key} at yaw {_deg(g.yaw)}, {approach})"
        return fine, numbers


@dataclass(kw_only=True)
class Lift(Action):
    """Lift the grasped object straight up by ``carry_height`` (``grasps[i].carry_height`` / the block's), object
    orientation kept (``arm_along(vertical)``).

    targets: ``{"lift": [Pose, ...]}`` — straight segment from the grasp pose (``after.end_pose``) to the carry pose
    (``line_waypoints``).
    """

    name: str = "lift"
    arm: str
    object: str
    carry_height: float
    after: Action
    TARGET_KEYS = ("lift",)

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()

    def _segment(self, sidecar, state) -> Tuple[Pose, Pose]:
        start = self.after.end_pose(sidecar, state)
        if start is None:
            raise SidecarError(f"lift: the preceding action {self.after.name} leaves no eef pose")
        return start, start.translated([0.0, 0.0, self.carry_height])

    def targets(self, sidecar, state):
        C = _conventions()
        start, end = self._segment(sidecar, state)
        return {"lift": _line(C, start, end)}

    def end_pose(self, sidecar, state):
        return self._segment(sidecar, state)[1]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        return f"lift {obj}", f"lift {obj} {_cm(self.carry_height)} straight up"


@dataclass(kw_only=True)
class Reorient(Action):
    """Move the held object through recorded in-hand key poses with the object attached (``arm_along``; §6.1):
    ``mode`` ``"orbit"`` — rotation about the object's centre (mental rotation, ``orbit_waypoints``); ``"line"`` —
    translation with the orientation kept (object scaling along the camera ray, ``line_waypoints``); ``"slerp"`` —
    rotation + translation of the held object (attachment: toward the pre-attach pose, ``slerp_waypoints`` +
    ``line_waypoints`` on the object pose); ``"flip"`` — the block yaw-align + 90° flip (``flip_waypoints``).
    Postcondition: object pose within tolerance of the waypoint end.

    Start = ``start_pose`` (the recorded hold of the held-object tasks) or ``after.end_pose`` (a lifted grasp);
    ``waypoints`` = the sampler's recorded key eef poses (last = the end; required except for ``"flip"``).

    targets: ``{"waypoints": [Pose, ...], "recorded": [Pose, ...] | None, "pivot": [x, y, z], "end": Pose}`` —
    ``waypoints`` derived through the conventions from start / end / pivot (what the oracle tracks), ``recorded``
    the sampler's key poses (both are IK-checked; they must agree), ``pivot`` the object centre at the start.
    """

    name: str = "reorient"
    mode: str
    arm: str
    object: str
    waypoints: Optional[List[Pose]] = None
    start_pose: Optional[Pose] = None
    after: Optional[Action] = None
    block: Optional[BlockRecord] = None
    n: int = N_WAYPOINTS
    TARGET_KEYS = ("waypoints", "recorded", "pivot", "end")
    MODES: ClassVar[Tuple[str, ...]] = ("orbit", "line", "slerp", "flip")

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()
        if self.mode not in self.MODES:
            raise ValueError(f"reorient: unknown mode {self.mode!r}")
        if (self.start_pose is None) == (self.after is None):
            raise ValueError("reorient: exactly one of start_pose / after must be given")
        if self.mode == "flip" and self.block is None:
            raise ValueError("reorient[flip]: needs the block record")
        if self.mode != "flip" and not self.waypoints:
            raise ValueError(f"reorient[{self.mode}]: needs the recorded waypoints")

    def _start(self, sidecar, state) -> Pose:
        if self.start_pose is not None:
            return self.start_pose
        start = self.after.end_pose(sidecar, state)
        if start is None:
            raise SidecarError(f"reorient: the preceding action {self.after.name} leaves no eef pose")
        return start

    def _grasp_pose(self, sidecar, state) -> Optional[Pose]:
        """The eef pose at the grasp (first link of the chain), to place the object centre after the lift."""
        action = self.after
        while action is not None:
            if isinstance(action, Grasp):
                return action.end_pose(sidecar, state)
            action = getattr(action, "after", None)
        return None

    def _pivot(self, sidecar, state, start: Pose) -> List[float]:
        obj = _state_pose(state, self.object)
        grasp = self._grasp_pose(sidecar, state)
        if grasp is None:
            return list(obj.position)
        return [obj.position[i] + start.position[i] - grasp.position[i] for i in range(3)]

    def targets(self, sidecar, state):
        C = _conventions()
        start = self._start(sidecar, state)
        pivot = self._pivot(sidecar, state, start)
        if self.mode == "flip":
            b = self.block
            yaw = -math.radians(b.yaw_deg)  # the block sits at yaw ψ off the table axes; the smallest align is −ψ
            if b.flips:
                wps = _as_pose_list(C.flip_waypoints(start.tensors(), _tensor(pivot), yaw=yaw, flip_axis=_tensor(b.flip_axis), flip_sign=b.flip_sign))
            else:
                wps = _as_pose_list(C.orbit_waypoints(start.tensors(), _tensor(pivot), C.quat_from_yaw(yaw), n=self.n))
        else:
            end = self.waypoints[-1]
            if self.mode == "orbit":
                delta = _qmul(end.orientation, _qconj(start.orientation))  # world-frame rotation start -> end
                wps = _as_pose_list(C.orbit_waypoints(start.tensors(), _tensor(pivot), _tensor(delta), n=self.n))
            else:
                # object-centred: the object pose moves on a line / slerp, the eef follows through the rigid in-hand
                # transform (for "line" the slerp is trivial: orientation kept)
                obj0 = Pose(pivot, _state_pose(state, self.object).orientation)
                t_obj_eef = _relative(start, obj0)
                obj1 = _compose(end, _inverse(t_obj_eef))
                pts = _as_pose_list(C.line_waypoints(_tensor(obj0.position), _tensor(obj1.position), self.n), quat=obj1.orientation)
                quats = C.slerp_waypoints(_tensor(obj0.orientation), _tensor(obj1.orientation), self.n)
                quats = [[float(v) for v in q] for q in (quats.tolist() if hasattr(quats, "tolist") else quats)]
                wps = [_compose(Pose(p.position, q), t_obj_eef) for p, q in zip(pts, quats)]
        end_pose = self.waypoints[-1] if self.waypoints else wps[-1]
        return {"waypoints": wps, "recorded": list(self.waypoints) if self.waypoints else None, "pivot": pivot, "end": end_pose}

    def end_pose(self, sidecar, state):
        if self.waypoints:
            return self.waypoints[-1]
        return self.targets(sidecar, state)["end"]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        if self.mode == "flip":
            b = self.block
            what = []
            if abs(b.yaw_deg) > 1e-6:
                what.append(f"yaw it {b.yaw_deg:.0f}° to the table axes")
            if b.flips:
                what.append(f"flip it 90° about the pinch axis ({'+' if b.flip_sign > 0 else '-'}{b.face_pair}, tilt {b.tilt_pick_deg:.0f}° → {b.tilt_place_deg:.0f}°)")
            fine = f"turn {obj} in hand to its goal face" + (" by yawing and flipping it" if len(what) == 2 else "")
            return fine, f"turn {obj} in hand: " + ", then ".join(what)
        start = self.start_pose if self.start_pose is not None else None
        end = self.waypoints[-1]
        if self.mode == "orbit":
            fine = f"rotate {obj} in hand to the goal orientation, keeping it in place"
            angle = _quat_angle(start.orientation, end.orientation) if start is not None else None
            numbers = f"rotate {obj} {_deg(angle)} in hand about its centre" if angle is not None else fine
            return fine, numbers
        if self.mode == "line":
            fine = f"move {obj} along the camera axis to the goal distance, keeping its orientation"
            dist = _distance(start.position, end.position) if start is not None else None
            numbers = f"move {obj} {_cm(dist)} along the camera axis" if dist is not None else fine
            return fine, numbers
        fine = f"turn {obj} in hand toward the attach orientation while bringing it to the pre-attach pose"
        numbers = f"{fine} at {_xyz(end.position)} ({len(self.waypoints)} recorded key poses)"
        return fine, numbers


@dataclass(kw_only=True)
class Place(Action):
    """Place the held object: ``arm_to(pre_place)`` → ``arm_along(descent)`` → ``gripper(open)`` →
    ``arm_along(retreat −a)``. ``kind``: ``"support"`` (set down; postcondition object settled on the support),
    ``"drop"`` (IORS inside: release above the rim, the object drops), ``"attach"`` (attachment: descent = insertion
    along the female axis; postcondition ``attached``). ``on`` = the instance placed on / into / attached to;
    ``relation`` = the annotation's relation word toward it ("on", "next to", "under"; without ``on`` the whole
    phrase, e.g. "at its slot in the goal grid").

    targets: ``{"pre_place": Pose, "place": Pose, "descent": [Pose, ...], "retreat": [Pose, ...]}`` — recorded
    ``place`` (and ``pre_place``, else ``pre_pose(place, PRE_DIST)``), straight descent and the retreat to
    ``pre_pose(place, RETREAT_DIST)`` (``line_waypoints``).
    """

    name: str = "place"
    kind: str
    arm: str
    object: str
    place: Pose
    pre_place: Optional[Pose] = None
    on: Optional[str] = None
    relation: str = "on"
    TARGET_KEYS = ("pre_place", "place", "descent", "retreat")
    KINDS: ClassVar[Tuple[str, ...]] = ("support", "drop", "attach")

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,) + ((self.on,) if self.on else ())
        super().__post_init__()
        if self.kind not in self.KINDS:
            raise ValueError(f"place: unknown kind {self.kind!r}")

    def targets(self, sidecar, state):
        C = _conventions()
        pre = self.pre_place if self.pre_place is not None else _pre(C, self.place, PRE_DIST)
        back = _pre(C, self.place, RETREAT_DIST)
        return {"pre_place": pre, "place": self.place, "descent": _line(C, pre, self.place), "retreat": _line(C, self.place, back)}

    def end_pose(self, sidecar, state):
        return _pre(_conventions(), self.place, RETREAT_DIST)

    def annotate(self, sidecar):
        lang = sidecar.language
        obj = lang.phrase(self.object)
        on = lang.phrase(self.on) if self.on else None
        if self.kind == "attach":
            fine = f"insert {obj} into {on} until it attaches" if on else f"insert {obj} until it attaches"
        elif self.kind == "drop":
            fine = f"lower {obj} into {on} and let it drop" if on else f"lower {obj} into its container and let it drop"
        elif on:
            fine = f"place {obj} {self.relation} {on}"
        else:
            fine = f"place {obj} {self.relation if self.relation != 'on' else 'at its goal position'}"
        fine += f" and release it with the {_hand(self.arm)}"
        numbers = f"{fine} ({_xyz(self.place.position)}, descent {_cm(PRE_DIST)}, retreat {_cm(RETREAT_DIST)})"
        return fine, numbers


@dataclass(kw_only=True)
class Push(Action):
    """Push with the closed gripper (``contacts.push`` / a block's ``push_row`` / ``push_col``): ``gripper(close)`` →
    ``arm_to(pre)`` → ``arm_along(init → mid → goal)``; postcondition: eef at the goal contact and the pushed object
    moved (``no_motion`` otherwise). ``failed_grasp`` is not checked (``qa_exempt``).

    targets: ``{"pre": Pose, "waypoints": [Pose, ...], "direction": [x, y, z], "distance_m": float}`` — the
    recorded contact key poses ``init, (mid,) goal`` (the sampler's ``arc_waypoints`` outputs); a straight push
    (no ``mid``) is densified with ``line_waypoints``. ``base_yaw`` (stance yaw) turns ``direction`` into a
    robot-relative word for the annotation.
    """

    name: str = "push"
    contact: PushContact
    base_yaw: Optional[float] = None
    qa_exempt: frozenset = frozenset({QA_FAILED_GRASP})
    TARGET_KEYS = ("pre", "waypoints", "direction", "distance_m")

    def __post_init__(self):
        self.arms = (self.contact.arm,)
        self.objects = (self.contact.object,)
        super().__post_init__()

    def targets(self, sidecar, state):
        C = _conventions()
        c = self.contact
        wps = c.waypoints if c.mid is not None else _line(C, c.init, c.goal)
        return {"pre": c.pre, "waypoints": wps, "direction": list(c.direction), "distance_m": c.distance_m}

    def end_pose(self, sidecar, state):
        return self.contact.goal

    def annotate(self, sidecar):
        c = self.contact
        obj = sidecar.language.phrase(c.object)
        word = direction_word(c.direction, self.base_yaw)
        arc = " along the recorded arc" if c.mid is not None else ""
        fine = f"push {obj} {word} with the closed {_hand(c.arm)}"
        numbers = f"push {obj} {_cm(c.distance_m)} {word}{arc} with the closed {_hand(c.arm)} from {_xyz(c.init.position)}"
        return fine, numbers


_BRACE_WORDS = {
    "below_high": "from below at its high end",
    "above_low": "from above at its low end",
    "side_high": "against the outer edge of its high end",
    "opposite_face": "on the face opposite to the push",
    "top_press": "by pressing down on its top",
    "stop_edge": "at its stop edge",
    "far_side": "on its far side",
    "row_left": "at the left end of the placed row",
    "column_top": "at the top of the placed column",
    "partner_face": "on the face opposite to the insertion",
}


@dataclass(kw_only=True)
class Brace(Action):
    """Brace with the closed gripper (``contacts.brace`` / a block's ``brace_row`` / ``brace_col``): ``gripper(close)``
    → ``arm_to(pre)`` → ``arm_along(pre → contact)``; placed before the working arm moves and held (no further
    command) until that arm's ``Retreat``. ``failed_grasp`` is not checked (``qa_exempt``).

    targets: ``{"pre": Pose, "contact": Pose, "approach": [Pose, ...]}`` (``line_waypoints`` pre → contact).
    """

    name: str = "brace"
    contact: BraceContact
    base_yaw: Optional[float] = None
    qa_exempt: frozenset = frozenset({QA_FAILED_GRASP})
    TARGET_KEYS = ("pre", "contact", "approach")

    def __post_init__(self):
        self.arms = (self.contact.arm,)
        self.objects = (self.contact.object,)
        super().__post_init__()

    def targets(self, sidecar, state):
        C = _conventions()
        c = self.contact
        return {"pre": c.pre, "contact": c.contact, "approach": _line(C, c.pre, c.contact)}

    def end_pose(self, sidecar, state):
        return self.contact.contact

    def annotate(self, sidecar):
        c = self.contact
        obj = sidecar.language.phrase(c.object)
        where = _BRACE_WORDS.get(c.kind, f"at its {c.kind.replace('_', ' ')}")
        fine = f"brace {obj} {where} with the closed {_hand(c.arm)}"
        numbers = f"{fine} at {_xyz(c.contact.position)}, pressing {direction_word(c.direction, self.base_yaw)} ({c.kind})"
        return fine, numbers


@dataclass(kw_only=True)
class BaseFollow(Action):
    """Move an articulated joint with the handle grasped (``articulation.{base_follow, waypoints, follow_mode,
    passive_from}``): ``follow_mode`` ``"base"`` tracks the base poses b_k with the arm locked (``base_follow``),
    ``"arm"`` moves the arm along h(q_k) (``arm_along``); up to ``passive_from`` (the link continues on its own) or
    q_goal. Postcondition: joint at q_goal, or ``passive_from`` reached and the joint continued passively.

    targets: ``{"mode": str, "fractions": [k, ...], "q_values": [q_k, ...], "handle_poses": [Pose, ...],
    "base_poses": [[x, y, yaw], ...] | None}`` — ``handle_poses`` = ``handle_eef`` of the recorded h(q_k) records;
    ``base_poses`` (base mode) = ``base_follow_poses(handle_poses, T_eef_base)`` with ``T_eef_base =
    eef_base_transform(h(q_init) eef, stance base pose)`` (the arm locked at its grasp solution).
    """

    name: str = "base_follow"
    arm: str
    articulation: Articulation
    stance: Stance
    grasp_action: Grasp
    direction: str = "move"
    TARGET_KEYS = ("mode", "fractions", "q_values", "handle_poses", "base_poses")

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.articulation.object,)
        super().__post_init__()

    def targets(self, sidecar, state):
        C = _conventions()
        art = self.articulation
        followed = art.followed_fractions()
        if not followed:
            raise SidecarError("base_follow: no waypoint beyond q_init to drive through")
        fractions = [f for f, _ in followed]
        handle_poses = [_handle_eef(C, art.waypoints[key]) for _, key in followed]
        base_poses = None
        if art.follow_mode == "base":
            eef0 = self.grasp_action.grasp_pose(sidecar, state)
            t_eef_base = _as_pose(C.eef_base_transform(eef0.tensors(), _yaw_pose(self.stance.base_pose2d).tensors()))
            base_poses = _as_pose2d_list(C.base_follow_poses([p.tensors() for p in handle_poses], t_eef_base.tensors()))
        return {
            "mode": art.follow_mode,
            "fractions": fractions,
            "q_values": [art.q_at(f) for f in fractions],
            "handle_poses": handle_poses,
            "base_poses": base_poses,
        }

    def end_pose(self, sidecar, state):
        return self.targets(sidecar, state)["handle_poses"][-1]

    def annotate(self, sidecar):
        art = self.articulation
        obj = sidecar.language.phrase(art.object)
        how = "driving the base" if art.follow_mode == "base" else "moving the arm"
        fine = f"{self.direction} {obj} by {how} while holding its handle with the {_hand(self.arm)}"
        if art.joint_type == "prismatic":
            span = f"{art.q_init:.2f} m to {art.q_goal:.2f} m ({_cm(art.q_goal - art.q_init)})"
        else:
            span = f"{art.q_init:.2f} rad to {art.q_goal:.2f} rad ({_deg(abs(art.q_goal - art.q_init))})"
        passive = f", letting go at {art.passive_from * 100:.0f} % where it continues on its own" if art.passive_from is not None else ""
        numbers = f"{fine}: joint {art.joint} from {span}{passive}"
        return fine, numbers


@dataclass(kw_only=True)
class Retreat(Action):
    """Move the gripper clear of the object along −a from where the preceding action left it (``after.end_pose``):
    ``gripper(open)`` first when ``release`` (a brace or a held handle), then ``arm_along(retreat)``; postcondition:
    eef clear by ``distance`` (``RETREAT_DIST``, or ``RETREAT_LITTLE`` "away a little" after a place).

    targets: ``{"retreat": [Pose, ...]}`` — straight segment to ``pre_pose(start, distance)`` (``line_waypoints``).
    """

    name: str = "retreat"
    arm: str
    object: str
    after: Action
    distance: float = RETREAT_DIST
    release: bool = False
    TARGET_KEYS = ("retreat",)

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()

    def _segment(self, sidecar, state) -> Tuple[Pose, Pose]:
        C = _conventions()
        start = self.after.end_pose(sidecar, state)
        if start is None:
            raise SidecarError(f"retreat: the preceding action {self.after.name} leaves no eef pose")
        return start, _pre(C, start, self.distance)

    def targets(self, sidecar, state):
        C = _conventions()
        start, end = self._segment(sidecar, state)
        return {"retreat": _line(C, start, end)}

    def end_pose(self, sidecar, state):
        return self._segment(sidecar, state)[1]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        release = f"release {obj} and " if self.release else ""
        fine = f"{release}move the {_hand(self.arm)} away from {obj}"
        numbers = f"{fine} by {_cm(self.distance)} along the approach axis"
        return fine, numbers


VOCABULARY: Tuple[type, ...] = (Navigate, Transition, SetTorso, Grasp, Lift, Reorient, Place, Push, Brace, BaseFollow, Retreat)
ACTION_NAMES: Tuple[str, ...] = tuple(cls.__dataclass_fields__["name"].default for cls in VOCABULARY)

__all__ = [
    "PRE_DIST", "RETREAT_DIST", "RETREAT_LITTLE", "N_WAYPOINTS", "QA_FAILED_GRASP", "State", "Action", "Navigate",
    "Transition", "SetTorso", "Grasp", "Lift", "Reorient", "Place", "Push", "Brace", "BaseFollow", "Retreat",
    "VOCABULARY", "ACTION_NAMES", "direction_word",
]
