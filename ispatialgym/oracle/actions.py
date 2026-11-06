"""Layer 2 of the v2 oracle: the action vocabulary (``oracle_primitives_design.md`` §3.6, behaviour §6.1).

An action is the unit of checkpoint, annotation and QA in the executor loop (§3.8). This module (TODO F6) declares
the contract and the vocabulary — ``Navigate``, ``Transition``, ``SetTorso``, ``Grasp``, ``Lift``, ``Reorient``,
``Place``, ``Push``, ``Brace``, ``BaseFollow``, ``PushClose``, ``BasePush``, ``Retreat`` — and implements ``targets()`` and ``annotate()``;
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

Conventions used (``ispatialgym/robot/conventions.py``, F5; tensors are float32 CPU, poses ``(pos (3,), quat_xyzw
(4,))`` tuples — :func:`_as_pose` also accepts ``FramedPose``, ``{"position", "orientation"}`` dicts and 7-vectors).
F5 waypoint rule (spec §3.9): every generator returns the ``n`` poses at fractions k/n, k = 1..n — the start excluded,
the goal last — so "3 intermediate waypoints" is ``n = 4`` (:data:`N_INHAND`). Signatures as called here:

==============================================================  ===================================================
``grasp_pose(obj_pose, grasp_key, grasp_name)``                  library grasp at the object's *current* (settled)
                                                                 pose; ``yaw`` omitted (``None`` = the object's own
                                                                 orientation — the recorded ``yaw`` was the sampler's
                                                                 draw and is already part of the settled pose; a
                                                                 float would *replace* the orientation with the
                                                                 nominal ``R_z(yaw) ∘ rest_quat``) — ``Grasp[library]``
``cube_grasp(block_pose, face_pair, tilt_pick_deg, yaw).eef``    ``tilt_pick_deg`` in degrees, ``yaw`` = the wrist yaw
                                                                 in radians (``blocks[].yaw_deg`` → rad) — ``Grasp[cube]``
``handle_eef(contact, approach=, closing=pinch)``                eef of a recorded handle waypoint h(q_k) (contact −
                                                                 0.012·a) — ``Grasp[handle]``, ``BaseFollow``
``pre_pose(pose, dist)``                                         back along −a — pre-grasp / pre-place / retreats
``line_waypoints(p0, p1, n)``                                    straight segments (positions in → ``(n, 3)``; poses in
                                                                 → orientation SLERPed alongside): approach, lift,
                                                                 descent, retreat, straight pushes, the held object's
                                                                 camera ray, the attachment's eef path
``orbit_waypoints(pose, pivot, delta_quat, n)``                  rotation about the object centre (mental rotation,
                                                                 block yaw-align), ``delta_quat`` in the world frame
``flip_waypoints(eef, pivot, yaw, flip_axis, flip_sign,``        block yaw-align ψ (rad, ``blocks[].yaw_align_deg``) +
``               n_yaw=2, n_flip=3)``                            one 90° flip (``Reorient[flip]``); 2 + 3 poses
``transition_poses(stance_from, stance_to, s_T)``                ``TransitionPoses`` dataclass: ``.drive_back``,
                                                                 ``.trunk``, ``.drive_in`` of a ``Transition``
``eef_base_transform(eef, base)``, ``base_follow_poses``         b_k = h(q_k) ∘ T_eef→base(q*) (``BaseFollow`` base mode)
==============================================================  ===================================================

Two §3.5 functions are the sampler's and not re-evaluated here because spec §7 records their *outputs*:
``handle_waypoints(handle, joint_info, q_values, obj_pose)`` (the ``articulation`` block carries the moved
``contact`` / ``approach`` / ``pinch`` per k — ``HandleWaypoint.to_sidecar()`` —, not the handle annotation and the
joint info it takes; the eef poses are ``handle_eef`` of those records, identical to ``HandleWaypoint.eef``) and
``arc_waypoints`` (the push record carries ``mid`` / ``goal``, not the arc centre / axis / δ). Recorded key poses win
over regeneration: where the sidecar carries an in-hand motion's waypoints (``grasps[].waypoints`` of the
held-object tasks and the attachment, ``blocks[].waypoints``) ``Reorient`` tracks exactly those and exposes its own
regeneration as ``derived`` for the program check to compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
from ispatialgym.robot import base_path as BP

# Geometry constants of the action behaviour (§6.1, spec §3.6); the eef offsets themselves live in the conventions.
PRE_DIST = 0.10  # pre-grasp / pre-contact / pre-place stand-off along −a
RETREAT_DIST = 0.10  # retreat along −a after a release / contact
RETREAT_LITTLE = 0.05  # "move contacting grippers away a little" after a place (§6.1)
# Waypoint counts follow the F5 rule (poses at k/n, the goal last): ``n`` poses = ``n − 1`` intermediate + the goal.
N_WAYPOINTS = 3  # poses of a short straight segment (approach, lift, descent, retreat, straight push): 2 + the goal
CARRY_WAYPOINT_SPACING = 0.05  # the carry leg lift end -> pre-place is a longer straight leg: a pose every 5 cm
#: oracle-fix 2026-11-05: the carry waypoints tried when a straight carry leg does not track - ``(where, raise m, lateral m, orientation
#: fraction toward the pre-place's)``: ``start`` = at the hand (rotate first), ``mid`` = the leg's midpoint, ``end`` = at the pre-place
#: (translate first); lateral is horizontal, across the leg. The sampler's search at C (``IORSLarge._carry_track_at_c``) and the oracle's
#: online search (``execute._carry_online``) read the same table
CARRY_VIAS = (("start", 0.0, 0.0, 1.0), ("start", 0.05, 0.0, 1.0), ("start", 0.10, 0.0, 1.0),
              ("mid", 0.05, 0.0, 1.0), ("mid", 0.10, 0.0, 1.0), ("mid", 0.05, 0.05, 1.0), ("mid", 0.05, -0.05, 1.0),
              ("mid", 0.0, 0.0, 0.5), ("mid", 0.05, 0.0, 0.5), ("mid", 0.10, 0.0, 0.5),
              ("end", 0.05, 0.0, 0.0), ("end", 0.10, 0.0, 0.0))
N_INHAND = 4  # in-hand motions of the held-object tasks / attachment: 3 intermediate + the goal (spec §3.9, §6.3)
N_YAW_ALIGN = 2  # block yaw-align stage (``flip_waypoints`` ``n_yaw``; the sampler's ``orbit_waypoints(..., n=2)``)
N_FLIP = 3  # block 90° flip stage (``flip_waypoints`` ``n_flip``)

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


def carry_via_candidates(h0, pre) -> List[Tuple[str, Any]]:
    """The candidate carry waypoints of :data:`CARRY_VIAS` between the hand pose ``h0`` and the pre-place ``pre`` (``(pos, quat)``),
    labelled ``"<where> raise <m> lateral <m> turn <frac>"``."""
    import torch as th

    C = _conventions()
    p0, q0 = th.as_tensor(h0[0]).float().reshape(-1)[:3], th.as_tensor(h0[1]).float().reshape(-1)
    p1, q1 = th.as_tensor(pre[0]).float().reshape(-1)[:3], th.as_tensor(pre[1]).float().reshape(-1)
    d = p1 - p0
    across = th.tensor([-float(d[1]), float(d[0]), 0.0])
    across = across / max(float(th.linalg.norm(across)), 1e-6)
    out: List[Tuple[str, Any]] = []
    for where, raise_m, lat, frac in CARRY_VIAS:
        base = {"start": p0, "mid": 0.5 * (p0 + p1), "end": p1}[where]
        pos = base + th.tensor([0.0, 0.0, float(raise_m)]) + across * float(lat)
        out.append((f"{where} raise {raise_m:.2f} lateral {lat:+.2f} turn {frac:.1f}", (pos, C.slerp(q0, q1, float(frac)).float())))
    return out


def carry_leg_poses(chain: Sequence[Any]) -> List[Tuple[Any, Any]]:
    """The carry leg's poses through ``chain`` (``(pos, quat)`` each) as :meth:`Place._carry_leg` builds them: a pose every
    :data:`CARRY_WAYPOINT_SPACING`, at least :data:`N_WAYPOINTS` per segment."""
    import torch as th

    C = _conventions()
    out: List[Tuple[Any, Any]] = []
    for a, b in zip(chain[:-1], chain[1:]):
        length = float(th.linalg.norm(th.as_tensor(b[0]).float() - th.as_tensor(a[0]).float()))
        n = max(N_WAYPOINTS, int(math.ceil(length / CARRY_WAYPOINT_SPACING)))
        out += [tuple(w) for w in C.line_waypoints(a, b, n)]
    return out


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


def _handle_eef(C, wp, delta: Optional[Tuple[List[float], List[float]]] = None, joint_rot: Optional[Tuple[Any, Any, float]] = None) -> Pose:
    """eef pose of a recorded handle waypoint h(q_k) (sidecar ``HandleWaypoint``: the world-frame ``contact`` /
    ``approach`` / ``pinch`` the sampler wrote with ``conventions.HandleWaypoint.to_sidecar()``) through
    ``conventions.handle_eef(contact, approach=, closing=)`` — the same eef ``handle_waypoints(handle, joint_info,
    q_values, obj_pose)`` computed for the sampler (its inputs are not in the sidecar, its outputs are). ``delta``
    (``_handle_delta``: the object's displacement since the record) moves the record with the object; ``joint_rot``
    (``_handle_joint_rot``: ``(hinge, axis, angle)`` in the live world) turns it with the link about its hinge."""
    contact, approach, pinch = _tensor(wp.contact), _tensor(wp.approach), _tensor(wp.pinch)
    if delta is not None:
        dp, dq = _tensor(delta[0]), _tensor(delta[1])
        contact = C.quat_rotate(dq, contact) + dp
        approach, pinch = C.quat_rotate(dq, approach), C.quat_rotate(dq, pinch)
    if joint_rot is not None:
        hinge, axis, ang = joint_rot
        rq = C.quat_from_axis_angle(axis, float(ang))
        contact = hinge + C.quat_rotate(rq, contact - hinge)
        approach, pinch = C.quat_rotate(rq, approach), C.quat_rotate(rq, pinch)
    return _as_pose(C.handle_eef(contact, approach=approach, closing=pinch))


def _handle_joint_rot(art, state, delta: Optional[Tuple[List[float], List[float]]] = None) -> Optional[Tuple[Any, Any, float]]:
    """Owner 2026-09-20 21:20: a handle grasp is derived from the link's LIVE joint angle, not the recorded one - the lever's
    lift under-delivers (0.2 of 0.35 rad, 0920-211200) and the recorded pinch closed above the edge. The waypoint at k = 0
    belongs to ``articulation.q_init``; the record's ``hinge_world`` / ``axis_world`` (moved with the object by ``delta``)
    and the state's joint value give the rotation ``(hinge, axis, q_live - q_init)``. None without the record / a state."""
    extra = getattr(art, "extra", None) or {}
    if extra.get("hinge_world") is None or extra.get("axis_world") is None or state is None or not hasattr(state, "joint_position"):
        return None
    try:
        q_live = float(state.joint_position(art.object, art.joint))
    except Exception:  # noqa: BLE001
        return None
    dq_j = q_live - float(art.q_init)
    if abs(dq_j) < 0.01:
        return None
    C = _conventions()
    hinge, axis = _tensor(extra["hinge_world"]), C.unit(_tensor(extra["axis_world"]))
    if delta is not None:
        dp, dq = _tensor(delta[0]), _tensor(delta[1])
        hinge = C.quat_rotate(dq, hinge) + dp
        axis = C.quat_rotate(dq, axis)
    return hinge, axis, dq_j


def _handle_delta(art, state) -> Optional[Tuple[List[float], List[float]]]:
    """The articulated object's rigid displacement since the sampler's record, ``T_live ∘ T_rec⁻¹`` as ``(dp, dq)`` (2026-09-19
    23:55: the fist pressing the other flap shoved the carton 5 cm and the recorded pinch closed on air, 0919-231459 /
    233259): ``articulation.object_pose`` (the root pose the waypoints were computed at) against ``state.object_pose``.
    None without the record or a state (the program check's stub), or when the displacement is negligible."""
    rec = (getattr(art, "extra", None) or {}).get("object_pose")
    if not isinstance(rec, dict) or state is None or not hasattr(state, "object_pose"):
        return None
    try:
        live = _as_pose(state.object_pose(art.object))
        p_rec, q_rec = [float(v) for v in rec["position"]], [float(v) for v in rec["orientation"]]
    except Exception:  # noqa: BLE001
        return None
    dq = _qmul(list(live.orientation), _qconj(q_rec))
    n = math.sqrt(sum(v * v for v in dq)) or 1.0
    dq = [v / n for v in dq]
    C = _conventions()
    rp = C.quat_rotate(_tensor(dq), _tensor(p_rec))
    dp = [float(live.position[i]) - float(rp[i]) for i in range(3)]
    ang = 2.0 * math.acos(max(-1.0, min(1.0, abs(dq[3]))))
    if math.sqrt(sum(v * v for v in dp)) < 0.002 and ang < 0.01:
        return None
    return dp, dq


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


def block_centre_pose(state: Any, block: BlockRecord) -> Pose:
    """The block's **geometric centre** pose ``root_pose ∘ (centre_offset, I)`` at its current pose — the frame the cube
    grasp and the in-hand flip pivot are defined in (``blocks[].centre_offset``, the offset of the centre from the
    root link in the block frame, written by the block-design sampler; the root pose when absent)."""
    root = _state_pose(state, block.instance)
    off = block.extra.get("centre_offset") if isinstance(block.extra, dict) else None
    if off is None:
        return root
    C = _conventions()
    return _as_pose(C.compose(root.tensors(), (_tensor(off), _tensor([0.0, 0.0, 0.0, 1.0]))))


def yaw_align_deg(block: BlockRecord) -> float:
    """ψ (degrees) of a block record: the smallest yaw that aligns the block with the table axes (``R_z(ψ) ∘ q_block``,
    the sampler's ``cube_flip_plan``), written by the block-design sampler as ``yaw_align_deg`` **next to** the wrist
    yaw ``yaw_deg`` of the cube grasp (both in degrees; spec §7 lists ``yaw_deg`` only). Read from the record's own
    field when the schema carries one, else from ``extra``; a record without it records no yaw-align (ψ = 0)."""
    value = getattr(block, "yaw_align_deg", None)
    if value is None:
        value = block.extra.get("yaw_align_deg", 0.0)
    return float(value)


def block_reorients(block: BlockRecord) -> bool:
    """Whether a block needs an in-hand reorientation between the lift and the place: a flip, or a non-zero yaw-align
    ψ (``yaw_align_deg``) — the wrist yaw ``yaw_deg`` of the cube grasp is not a rotation of the block."""
    return block.flips or abs(yaw_align_deg(block)) > 1e-6


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


def _arms_q_of(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[float]]]:
    """``arms_q`` of a path / transition record (arm -> 7 joint values as floats), None when absent or malformed."""
    aq = (extra or {}).get("arms_q")
    if not isinstance(aq, dict):
        return None
    out: Dict[str, List[float]] = {}
    for a, v in aq.items():
        try:
            vals = [float(x) for x in v]
        except (TypeError, ValueError):
            return None
        if len(vals) != 7:
            return None
        out[str(a)] = vals
    return out or None


def _preshape_of(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """``preshape_m`` of a transition record (arm -> finger opening in m, oracle-fix 2026-11-06), None when absent or malformed."""
    ps = (extra or {}).get("preshape_m")
    if not isinstance(ps, dict):
        return None
    out: Dict[str, float] = {}
    for a, v in ps.items():
        try:
            out[str(a)] = float(v)
        except (TypeError, ValueError):
            return None
    return out or None


@dataclass(kw_only=True)
class Navigate(Action):
    """Follow a recorded base path (``paths[i]``; motion ``track_path``).

    targets: ``{"waypoints": [[x, y, yaw], ...], "goal": [x, y, yaw], "posture": str, "carried": str | None,
    "style": str}`` — the recorded waypoints (whole-body collision-checked at ``posture`` with ``carried`` attached),
    copied from the record, and how they are driven (spec §3.7): ``"recorded"`` tracks the polyline as recorded
    (interior yaws blended per segment); ``"face_travel"`` drives the shared schedule of ``robot.base_path`` (grid
    staircase simplified within one map cell, corners filleted, the body facing the way with the hybrid end turns —
    what the camera-relocalization sampler collision-checked, owner 2026-09-22). The program sets ``style``; when it
    does not, the record's own ``style`` (sidecar ``paths[].style``) decides, else ``"recorded"``. ``obstacles`` (camera
    relocalization — obstructed) name the objects the path avoids for the annotation; they are deliberately **not** in
    ``objects`` (never touched). ``arms_q`` (sidecar ``paths[].arms_q``, owner 2026-09-24; None when absent): the arm
    joints (arm -> 7) the sampler swept the drive with when it left an arm where the previous motion ended it instead of
    the stance posture - the executor re-poses only an arm that deviates from them. ``axis_leg`` (owner 2026-10-13, the
    disembedding leg into the place pose; the program sets it after ``robot.base_path.axis_leg_violation`` passed): the
    heading is kept and the moves are forward / backward or left / right only - the tracker stops at every corner and the
    postcondition checks the driven heading and corridor. ``face_travel_cfg`` (sidecar ``paths[].face_travel_cfg``, oracle-fix
    2026-11-06; None when absent or not driven as ``face_travel``): the record's own ``robot.base_path.BasePathConfig`` overrides
    (``robot.base_path.config_with``) - the schedule the sampler checked when the default one collided.
    """

    name: str = "navigate"
    path: PathRecord
    obstacles: Tuple[str, ...] = ()
    style: Optional[str] = None
    axis_leg: bool = False
    TARGET_KEYS = ("waypoints", "goal", "posture", "carried", "style", "arms_q", "axis_leg", "face_travel_cfg")

    def __post_init__(self):
        super().__post_init__()
        if self.style is not None and self.style not in BP.STYLES:
            raise ValueError(f"navigate: unknown style {self.style!r} (one of {BP.STYLES})")
        if self.axis_leg and self.style not in (None, BP.RECORDED):
            raise ValueError(f"navigate: an axis-aligned leg is driven as recorded, not {self.style!r}")
        if (self.path.extra or {}).get("face_travel_cfg"):
            BP.config_with(self.path.extra["face_travel_cfg"])  # ValueError for a key robot.base_path.BasePathConfig does not have

    def targets(self, sidecar, state):
        style = BP.RECORDED if self.axis_leg else (self.style or (self.path.extra or {}).get("style") or BP.RECORDED)
        cfg = (self.path.extra or {}).get("face_travel_cfg") if style == BP.FACE_TRAVEL else None
        return {
            "waypoints": [list(w) for w in self.path.waypoints],
            "goal": self.path.goal,
            "posture": self.path.posture,
            "carried": self.path.carried,
            "style": style,
            "arms_q": _arms_q_of(self.path.extra),
            "axis_leg": bool(self.axis_leg),
            "face_travel_cfg": {str(k): float(v) for k, v in cfg.items()} if isinstance(cfg, dict) and cfg else None,
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
        if self.path.mode == "edge_slide" and len(self.path.waypoints) >= 2:
            # the §3.7 edge slide (sampler 2026-09-24): the body faces the table throughout, the base slides along the edge;
            # the lateral displacement in the robot's frame (+ = its left) and the stand-off change (along its forward axis)
            x0, y0, yaw0 = (float(v) for v in self.path.waypoints[0])
            lat = -(x - x0) * math.sin(yaw0) + (y - y0) * math.cos(yaw0)
            fwd = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
            side = "left" if lat >= 0 else "right"
            fine = fine.replace("carry ", "slide sideways carrying ", 1) if self.path.carried else fine.replace("drive to", "slide sideways to", 1)
            moves = BP.axis_leg_moves(self.path.waypoints)
            numbers = (f"{fine} ({abs(lat):.2f} m to the {side} along the edge facing the table"
                       + (f", stand-off {'+' if fwd < 0 else '-'}{abs(fwd):.2f} m" if abs(fwd) > 0.005 else "")
                       + (f", moves {' then '.join(f'{a} {m:.2f} m' for a, m in moves)}" if len(moves) > 1 else "")
                       + f", {self.path.length_m:.2f} m path, yaw held at {math.degrees(yaw):.0f}°)")
            return fine, numbers
        numbers = f"{fine} ({self.path.length_m:.2f} m recorded path to x={x:.2f} m, y={y:.2f} m, yaw={math.degrees(yaw):.0f}°)"
        return fine, numbers


@dataclass(kw_only=True)
class Transition(Action):
    """Stance transition (``transitions[i]`` + the two stances, spec §3.8): drive back to the transition stand-off
    s_T, move the trunk, drive in (motions ``track_path`` → ``trunk_to`` → ``track_path``).

    targets: ``{"drive_back": [x, y, yaw], "trunk": (translate, tilt), "drive_in": [x, y, yaw], "arms_q": {arm: [7]} | None,
    "preshape": {arm: m} | None}`` — the fields ``drive_back`` / ``trunk`` / ``drive_in`` of the ``TransitionPoses`` dataclass
    ``conventions.transition_poses(stance_from, stance_to, s_T)`` returns (stances passed as their sidecar dicts);
    ``arms_q`` (sidecar ``transitions[].arms_q``, owner 2026-09-24): the arm joints the sampler swept the transition with when
    it left an arm where the previous motion ended it - the executor's ``move_idle_arms`` then targets these instead of
    the arriving stance's posture. ``preshape`` (sidecar ``transitions[].preshape_m``, oracle-fix 2026-11-06, the shelf back-off: the
    record carries the arm to its pre-grasp): the finger opening per arm the executor pre-shapes to before the arms move.
    """

    name: str = "transition"
    transition: TransitionRecord
    stance_from: Stance
    stance_to: Stance
    carried: Optional[str] = None
    TARGET_KEYS = ("drive_back", "trunk", "drive_in", "arms_q", "preshape")

    def targets(self, sidecar, state):
        C = _conventions()
        if getattr(self.transition, "departure", False):
            # owner 2026-09-19 21:56: the trunk change at the DEPARTING stance (drive back along its own axis, sweep to the travel
            # trunk, no drive-in); the recorded path that follows starts at ``drive_back``
            tp = C.departure_transition_poses(self.stance_from.to_dict(), self.transition.stand_off, self.transition.trunk_to)
        else:
            tp = C.transition_poses(self.stance_from.to_dict(), self.stance_to.to_dict(), self.transition.stand_off)
        translate, tilt = (float(v) for v in tp.trunk)
        return {"drive_back": _as_pose2d(tp.drive_back), "trunk": (translate, tilt), "drive_in": _as_pose2d(tp.drive_in),
                "arms_q": _arms_q_of(getattr(self.transition, "extra", None)), "preshape": _preshape_of(getattr(self.transition, "extra", None))}

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

    ``kind``:

    * ``"library"`` — ``conventions.grasp_pose(obj_pose, grasp_key, grasp_name)`` at the object's **current** pose
      (``state.object_pose``: the settled init pose for the sampler's check, the restored / drifted pose at run time)
      with ``yaw`` **omitted**. The sidecar's ``grasps[].yaw`` is the object yaw the sampler drew (spec §4.6) and the
      settled pose already carries it; F5 takes a float ``yaw`` as "replace the orientation by the nominal
      ``R_z(yaw) ∘ rest_quat(grasp_key)``", so passing the recorded yaw — or 0 — would grasp the un-drifted nominal
      object instead of the one in front of the robot. ``yaw`` stays an annotation fact only.
    * ``"cube"`` — ``conventions.cube_grasp(block_pose, face_pair, tilt_pick_deg, yaw).eef`` with ``tilt_pick_deg``
      in degrees as recorded and ``yaw`` = the recorded **wrist yaw** ``blocks[].yaw_deg`` converted to radians (F5
      snaps it to the nearer admissible value; the table-alignment yaw ``yaw_align_deg`` belongs to ``Reorient``).
    * ``"handle"`` — ``conventions.handle_eef`` of the recorded h(q_init) (``articulation.waypoints["0.0"]``).

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
    #: instances the grasp may touch besides the grasped object (QA whitelist): ``grasps[].may_touch`` (disembedding's parts and
    #: distractors, owner 2026-10-12 A2b)
    touches: Tuple[str, ...] = ()
    #: block design after a between-block base move (``programs.block_repose``): ``{"arm", "target": "pre_grasp", "q": [7], "pose": Pose,
    #: "from_stance"}`` - the arm first makes a planned joint-space move to ``q`` (this grasp's recorded pre-grasp configuration), then
    #: the transit; None = the transit from wherever the arm is
    repose: Optional[Dict[str, Any]] = None
    TARGET_KEYS = ("pre_grasp", "grasp", "approach")

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        self.touches = tuple(self.touches)
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
            # the sampler's recorded object-frame eef transform (``grasps[].eef_in_object``, spec §7) carries every adjustment
            # of the library pinch the sampler made and program-checked - the re-centred pinch, the raised knob grasp of
            # ``grasp_height_extra_m`` / ``contact_height_extra_m`` (attachment 0919-171109: the sidecar recorded the lid grasp
            # 8 mm higher, the oracle re-derived the library pose and closed at the old height) - applied to the object's
            # CURRENT pose exactly like the library transform; the library transform is the fallback for a record without it
            eio = g.extra.get("eef_in_object") if isinstance(g.extra, dict) else None
            if isinstance(eio, dict) and eio.get("position") is not None and eio.get("orientation") is not None:
                obj_t = obj.tensors()
                ov = g.extra.get("pick_overhang")
                if isinstance(ov, dict) and ov.get("yaw_free") and isinstance(ov.get("post_push_pose"), dict) and ov["post_push_pose"].get("orientation") is not None:
                    # the overhang pick of a body of revolution about z (owner 2026-10-13 ~03:15 item 4): the grasp at the object's current
                    # position with the yaw the sampler settled it at - a push-induced turn of a round plate does not move the pinch off the overhang
                    q_cur = obj_t[1]
                    dy = C.wrap_angle(C.yaw_of(_tensor(ov["post_push_pose"]["orientation"])) - C.yaw_of(q_cur))
                    obj_t = (obj_t[0], C.qmul(C.quat_from_yaw(dy), q_cur))
                return _as_pose(C.grasp_pose_from_transform(obj_t, _as_pose(eio).tensors()))
            # yaw omitted on purpose: the settled pose carries the drawn yaw (see the class docstring)
            return _as_pose(C.grasp_pose(obj.tensors(), g.grasp_key, g.grasp_name))
        if self.kind == "cube":
            b = self.block
            centre = block_centre_pose(state, b)  # the cube grasp is defined about the block's centre, not its root link
            # the pinch depth follows the recorded tip clearance (``blocks[].tip_clearance_m``, the sampler's ladder /
            # lean-dependent finger clearance); the conventions default (3 mm) when the record carries none
            extra = b.extra if isinstance(b.extra, dict) else {}
            kw = {}
            if extra.get("tip_clearance_m") is not None:
                kw["tip_clearance"] = float(extra["tip_clearance_m"])
            if extra.get("edge_m") is not None:
                kw["edge"] = float(extra["edge_m"])
            cube = C.cube_grasp(centre.tensors(), face_pair=b.face_pair, tilt_pick_deg=float(b.tilt_pick_deg), yaw=math.radians(b.yaw_deg), **kw)
            return _as_pose(cube.eef)
        art = self.articulation
        delta = _handle_delta(art, state)
        return _handle_eef(C, art.waypoints[art.key_at(0.0)], delta=delta, joint_rot=_handle_joint_rot(art, state, delta))

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
            numbers = f"{fine} (pinch across the {b.face_pair} faces, pick tilt {b.tilt_pick_deg:.0f}°, wrist yaw {b.yaw_deg:.0f}°, {approach})"
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
    #: oracle-fix 2026-11-04 round 4 (``grasps[i].lift_away``, touching#03's shelf tip release): a world offset (m) added to the lift's end - the book
    #: leaning on a side board is lifted away from it; None = straight up
    away: Optional[Sequence[float]] = None
    TARGET_KEYS = ("lift",)

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()

    def _segment(self, sidecar, state) -> Tuple[Pose, Pose]:
        """Straight up from where the hand **is** (``state.eef_pose`` at run time — the grasp may have closed a few mm
        off the recorded pose, and asking the lift to correct that residual first stalled the differential IK at the
        shoulder singularity of the recorded configuration), else from the recorded grasp pose (the sampler's
        check, the stub state)."""
        start = None
        eef = getattr(state, "eef_pose", None)
        if callable(eef):
            try:
                start = _as_pose(eef(self.arm))
            except Exception:  # noqa: BLE001
                start = None
        if start is None:
            start = self.after.end_pose(sidecar, state)
        if start is None:
            raise SidecarError(f"lift: the preceding action {self.after.name} leaves no eef pose")
        dx, dy, dz = (float(v) for v in (self.away or (0.0, 0.0, 0.0)))
        return start, start.translated([dx, dy, self.carry_height + dz])

    def targets(self, sidecar, state):
        C = _conventions()
        start, end = self._segment(sidecar, state)
        return {"lift": _line(C, start, end)}

    def end_pose(self, sidecar, state):
        return self._segment(sidecar, state)[1]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        if self.away:
            return f"lift {obj}", f"lift {obj} {_cm(self.carry_height)} up and {_cm(math.hypot(float(self.away[0]), float(self.away[1])))} away from the board"
        return f"lift {obj}", f"lift {obj} {_cm(self.carry_height)} straight up"


@dataclass(kw_only=True)
class Reorient(Action):
    """Move the held object through in-hand key poses with the object attached (``arm_along``; §6.1): ``mode``
    ``"orbit"`` — rotation about the object's centre, position kept (mental rotation, ``orbit_waypoints``);
    ``"line"`` — translation with the orientation kept (object scaling along the camera ray, ``line_waypoints``);
    ``"slerp"`` — rotation + translation of the held object (attachment: the eef on a straight line to the pre-attach
    pose with its orientation SLERPed alongside, ``line_waypoints`` on poses); ``"flip"`` — the block yaw-align ψ +
    one 90° flip (``flip_waypoints``; ``orbit_waypoints`` for a yaw-align without a flip). Postcondition: object pose
    within tolerance of the waypoint end.

    Start = ``start_pose`` (the recorded hold of the held-object tasks) or ``after.end_pose`` (a lifted grasp);
    ``waypoints`` = the sampler's recorded key eef poses in order, last = the end (required except for ``"flip"``,
    where the block record may or may not carry them). ``n`` = the poses of a regenerated in-hand motion at
    fractions k/n, the goal last: :data:`N_INHAND` = 4 gives the 3 intermediate waypoints of spec §3.9 / §6.3, the
    count the held-object sampler records (``line_waypoints(p0, p1, 4)`` / ``orbit_waypoints(..., n=4)``).

    targets: ``{"waypoints": [Pose, ...], "derived": [Pose, ...], "pivot": [x, y, z], "end": Pose}`` —
    ``waypoints`` is what the oracle tracks: the **recorded** key poses when the sidecar carries them (the program
    check and the executor then use exactly the poses the sampler validated), else ``derived``; ``derived`` is the
    same motion regenerated through the conventions from start / end / pivot (always present; IK-checked alongside,
    it must agree with the recorded poses); ``pivot`` the object root position at the start (the orbit centre);
    ``end`` = ``waypoints[-1]``.
    """

    name: str = "reorient"
    mode: str
    arm: str
    object: str
    waypoints: Optional[List[Pose]] = None
    start_pose: Optional[Pose] = None
    after: Optional[Action] = None
    block: Optional[BlockRecord] = None
    n: int = N_INHAND
    TARGET_KEYS = ("waypoints", "derived", "pivot", "end")
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
        """The in-hand rotation pivot: the object's centre — for a block ``root ∘ centre_offset`` — carried to where
        the lifted eef holds it (``start`` vs the grasp eef pose at the object's current pose)."""
        obj = block_centre_pose(state, self.block) if self.block is not None else _state_pose(state, self.object)
        grasp = self._grasp_pose(sidecar, state)
        if grasp is None:
            return list(obj.position)
        return [obj.position[i] + start.position[i] - grasp.position[i] for i in range(3)]

    def derived_waypoints(self, sidecar, state, start: Pose, pivot: List[float]) -> List[Pose]:
        """The in-hand motion regenerated through the conventions (the sampler's own construction per mode)."""
        C = _conventions()
        if self.mode == "flip":
            b = self.block
            psi = math.radians(yaw_align_deg(b))  # +ψ: R_z(ψ) ∘ q_block aligns the block (sampler's cube_flip_plan)
            if b.flips:
                wps = C.flip_waypoints(start.tensors(), _tensor(pivot), yaw=psi, flip_axis=_tensor(b.flip_axis), flip_sign=b.flip_sign,
                                       n_yaw=N_YAW_ALIGN, n_flip=N_FLIP)
            else:
                wps = C.orbit_waypoints(start.tensors(), _tensor(pivot), C.quat_from_yaw(psi), n=N_YAW_ALIGN)
            return _as_pose_list(wps)
        end = self.waypoints[-1]
        if self.mode == "orbit":
            delta = _qmul(end.orientation, _qconj(start.orientation))  # world-frame rotation start -> end
            return _as_pose_list(C.orbit_waypoints(start.tensors(), _tensor(pivot), _tensor(delta), n=self.n))
        # "line" / "slerp": the eef on a straight line start -> end, orientation SLERPed alongside (constant for "line")
        return _as_pose_list(C.line_waypoints(start.tensors(), end.tensors(), self.n))

    def targets(self, sidecar, state):
        start = self._start(sidecar, state)
        pivot = self._pivot(sidecar, state, start)
        derived = self.derived_waypoints(sidecar, state, start, pivot)
        tracked = list(self.waypoints) if self.waypoints else derived
        return {"waypoints": tracked, "derived": derived, "pivot": pivot, "end": tracked[-1]}

    def end_pose(self, sidecar, state):
        if self.waypoints:
            return self.waypoints[-1]
        return self.targets(sidecar, state)["end"]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        if self.mode == "flip":
            b = self.block
            psi = yaw_align_deg(b)
            what = []
            if abs(psi) > 1e-6:
                what.append(f"yaw it {psi:.0f}° to the table axes")
            if b.flips:
                what.append(f"flip it 90° about the pinch axis ({'+' if b.flip_sign > 0 else '-'}{b.face_pair}, tilt {b.tilt_pick_deg:.0f}° → {b.tilt_place_deg:.0f}°)")
            if not what:
                what.append("no in-hand rotation recorded (0° yaw-align, no flip)")
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
            # the sampler's signed depth change (< 0: toward the camera); sidecars written before it name no direction
            change = (getattr(sidecar.language, "goal_numbers", None) or {}).get("depth_change_cm")
            way = "along the camera axis" if change is None else ("away from the camera" if float(change) > 0 else "toward the camera")
            fine = f"move {obj} {way} to the goal distance, keeping its orientation"
            dist = _distance(start.position, end.position) if start is not None else None
            numbers = f"move {obj} {_cm(dist)} {way}" if dist is not None else fine
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

    targets: ``{"pre_place": Pose, "place": Pose, "descent": [Pose, ...], "retreat": [Pose, ...], "carry": [Pose, ...]}``
    — recorded ``place`` (and ``pre_place``, else ``pre_pose(place, PRE_DIST)``), straight descent and the retreat to
    ``pre_pose(place, RETREAT_DIST)`` (``line_waypoints``); ``carry`` = the **carry leg** from where the hand is (the
    lift end, ``state.eef_pose`` at run time) to the pre-place — straight, orientation SLERPed, through the recorded
    ``grasps[k].carry`` waypoint when the sampler wrote one (its straight leg was infeasible; ``carry_via`` lists those
    recorded waypoints) — so the fingertips stay in the head camera frame between two recorded in-frame poses (FOV
    rule (b), owner 2026-09-16 17:10); ``[]`` when the state has no eef pose (the sampler's check, the stub state).
    """

    name: str = "place"
    kind: str
    arm: str
    object: str
    place: Pose
    pre_place: Optional[Pose] = None
    on: Optional[str] = None
    relation: str = "on"
    #: instances the action must not touch (QA: forbidden contacts beyond the drop container), e.g. the partner a
    #: push-under drop lands in front of
    avoid: Tuple[str, ...] = ()
    TARGET_KEYS = ("pre_place", "place", "descent", "retreat", "carry", "carry_via")
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
        carry, via = self._carry_leg(C, sidecar, state, pre)
        return {"pre_place": pre, "place": self.place, "descent": _line(C, pre, self.place), "retreat": _line(C, self.place, back),
                "carry": carry, "carry_via": via}

    def carry_waypoints(self, sidecar) -> List[Pose]:
        """The sampler's ``grasps[k].carry`` poses for this object / arm (``[]`` when none was recorded)."""
        for g in getattr(sidecar, "grasps", None) or []:
            if getattr(g, "object", None) != self.object or getattr(g, "arm", None) not in (None, self.arm):
                continue
            raw = (getattr(g, "extra", None) or {}).get("carry") or getattr(g, "carry", None) or []
            return [_as_pose(w) if not isinstance(w, dict) else Pose.from_dict(w, "grasps[].carry") for w in raw]
        return []

    def _carry_leg(self, C, sidecar, state, pre: Pose) -> Tuple[List[Pose], List[Pose]]:
        """``(carry leg poses, the recorded carry waypoints it passes through)``."""
        start = None
        eef = getattr(state, "eef_pose", None)
        if callable(eef):
            try:
                start = _as_pose(eef(self.arm))
            except Exception:  # noqa: BLE001
                start = None
        if start is None:
            return [], []
        via = self.carry_waypoints(sidecar)
        legs: List[Pose] = []
        a = start
        for b in via + [pre]:
            length = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(b.position, a.position)))
            n = max(N_WAYPOINTS, int(math.ceil(length / CARRY_WAYPOINT_SPACING)))
            legs += _as_pose_list(C.line_waypoints(a.tensors(), b.tensors(), n))
            a = b
        return legs, via

    def end_pose(self, sidecar, state):
        return _pre(_conventions(), self.place, RETREAT_DIST)

    def annotate(self, sidecar):
        lang = sidecar.language
        obj = lang.phrase(self.object)
        on = lang.phrase(self.on) if self.on else None
        if self.kind == "attach":
            fine = f"insert {obj} into {on} until it attaches" if on else f"insert {obj} until it attaches"
        elif self.kind == "drop" and self.relation in ("into", "on"):
            fine = f"lower {obj} into {on} and let it drop" if on else f"lower {obj} into its container and let it drop"
        elif self.kind == "drop":
            fine = f"lower {obj} to {on} and let it drop" if on else f"lower {obj} and let it drop"
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
    #: instances the push may touch besides the pushed object (QA whitelist): the floor of a ``push_under``
    touches: Tuple[str, ...] = ()
    #: instances the push must not touch (QA forbidden contacts): the partner of a ``push_under``
    avoid: Tuple[str, ...] = ()
    #: the instance the pushed object's goal was sampled against (the touching partner): when it has moved since the program
    #: start, the executor moves the goal with it
    follow: Optional[str] = None
    qa_exempt: frozenset = frozenset({QA_FAILED_GRASP})
    TARGET_KEYS = ("pre", "waypoints", "direction", "distance_m")

    def __post_init__(self):
        self.arms = (self.contact.arm,)
        self.objects = (self.contact.object,)
        super().__post_init__()

    @property
    def kind(self) -> str:
        """``contacts.push.kind``: ``"push"`` (default), ``"push_under"`` (the floor-level push of IORS under, spec
        §5.6 owner decision 2026-09-16 item 7) or ``"push_touching"`` (the floor-level push of IORS touching against the
        partner's face, owner 2026-10-12)."""
        return str(self.contact.extra.get("kind") or "push")

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
        if self.kind == "push_under":
            partner = c.extra.get("partner")
            under = f" under {sidecar.language.phrase(partner)}" if partner else ""
            fine = f"push {obj} along the floor{under} with the closed {_hand(c.arm)}"
            numbers = f"push {obj} {_cm(c.distance_m)} {word} along the floor{under} with the closed {_hand(c.arm)} from {_xyz(c.init.position)}"
            return fine, numbers
        if self.kind == "push_touching":
            partner = c.extra.get("partner")
            until = f" until it touches {sidecar.language.phrase(partner)}" if partner else ""
            fine = f"push {obj} along the floor{until} with the closed {_hand(c.arm)}"
            numbers = f"push {obj} {_cm(c.distance_m)} {word} along the floor{until} with the closed {_hand(c.arm)} from {_xyz(c.init.position)}"
            return fine, numbers
        if self.kind == "pick_overhang":
            support = c.extra.get("support")
            along = f" along {sidecar.language.phrase(support)}" if support in sidecar.language.objects else " along its support"
            hand = f"the {'open' if c.extra.get('fingers') == 'open' else 'closed'} {_hand(c.arm)}"
            fine = f"push {obj}{along} until it overhangs the edge, with {hand}"
            over = c.extra.get("overhang_m")
            numbers = (f"push {obj} {_cm(c.distance_m)} {word}{along} with {hand}" + (f" until it overhangs the edge by {_cm(float(over))}" if over is not None else "")
                       + f" from {_xyz(c.init.position)}")
            return fine, numbers
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
    command) until that arm's ``Retreat``. ``failed_grasp`` is not checked (``qa_exempt``). A wide stop (the record's
    ``fingers_opening_m``, :func:`brace_is_open`) pre-shapes the fingers open instead of closing them, and is annotated so.

    targets: ``{"pre": Pose, "contact": Pose, "approach": [Pose, ...]}`` (``line_waypoints`` pre → contact).

    ``follow``: the program has not moved the braced object before the brace, so its init pose is the pose the stop was
    recorded at; the executor moves the targets with the object when something shoved it since the program start.
    """

    name: str = "brace"
    contact: BraceContact
    base_yaw: Optional[float] = None
    follow: bool = False
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
        fine = f"brace {obj} {where} with the {'open' if brace_is_open(c) else 'closed'} {_hand(c.arm)}"
        numbers = f"{fine} at {_xyz(c.contact.position)}, pressing {direction_word(c.direction, self.base_yaw)} ({c.kind})"
        return fine, numbers


def brace_is_open(contact) -> bool:
    """A wide stop: the brace record pre-shapes the fingers to ``fingers_opening_m`` (> 0) so both fingertips meet the braced face
    (``execute_brace`` opens the gripper to it before the transit); every other brace closes the gripper."""
    opening = (getattr(contact, "extra", None) or {}).get("fingers_opening_m")
    try:
        return opening is not None and float(opening) > 0.0
    except (TypeError, ValueError):
        return False


@dataclass(kw_only=True)
class BaseFollow(Action):
    """Move an articulated joint with the handle grasped (``articulation.{base_follow, waypoints, follow_mode,
    passive_from}``): ``follow_mode`` ``"base"`` tracks the base poses b_k with the arm locked (``base_follow``),
    ``"arm"`` moves the arm along h(q_k) (``arm_along``); up to ``passive_from`` (the link continues on its own) or
    q_goal. Postcondition: joint at q_goal, or ``passive_from`` reached and the joint continued passively.

    targets: ``{"mode": str, "fractions": [k, ...], "q_values": [q_k, ...], "handle_poses": [Pose, ...],
    "base_poses": [[x, y, yaw], ...] | None, "follow_path": [...] | None}`` — ``handle_poses`` = ``conventions.handle_eef``
    of the recorded h(q_k) records (``articulation.waypoints[k]``: the ``contact`` / ``approach`` / ``pinch`` the
    sampler's ``handle_waypoints(handle, joint_info, q_values, obj_pose)`` produced, in the world frame — the sidecar
    carries those outputs, not the handle annotation / joint info, so the eef is re-derived from the records, never
    from ``handle_waypoints``); ``base_poses`` (base mode) = ``base_follow_poses(handle_poses, T_eef_base)`` with
    ``T_eef_base = eef_base_transform(h(q_init) eef, stance base pose)`` (the arm locked at its grasp solution).

    ``"hybrid"`` (sampler ``articulation_hybrid``, D-011 C): ``base_poses`` = the **recorded** ``articulation.base_follow[k]``
    per followed fraction (the base does not orbit rigidly: the arm re-solves per pose), and ``follow_path`` = the dense
    ``articulation.base_follow_path.poses[]`` records ``{"k", "q", "base_pose2d": [x, y, yaw], "arm_q": [7 arm joints],
    "eef": {position, orientation}}`` from the grasp (k = 0, the manipulation stance) to the last followed waypoint — the
    executor drives the base along ``base_pose2d`` and moves the arm joints to ``arm_q`` **together** (both interpolated
    over the same segment) so the eef stays on the handle arc; no re-planning.
    """

    name: str = "base_follow"
    arm: str
    articulation: Articulation
    stance: Stance
    grasp_action: Grasp
    direction: str = "move"
    TARGET_KEYS = ("mode", "fractions", "q_values", "handle_poses", "base_poses", "follow_path", "follow_segments")

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
        delta = _handle_delta(art, state)  # the object moved since the record (a press shoved it): the follow moves with it
        handle_poses = [_handle_eef(C, art.waypoints[key], delta=delta) for _, key in followed]
        base_poses = None
        follow_path = None
        follow_segments = None
        if art.follow_mode == "base":
            eef0 = self.grasp_action.grasp_pose(sidecar, state)
            t_eef_base = _as_pose(C.eef_base_transform(eef0.tensors(), _yaw_pose(self.stance.base_pose2d).tensors()))
            base_poses = _as_pose2d_list(C.base_follow_poses([p.tensors() for p in handle_poses], t_eef_base.tensors()))
        elif art.follow_mode == "hybrid":
            if not art.base_follow:
                raise SidecarError("base_follow: follow_mode 'hybrid' needs the recorded base_follow[k] poses")
            missing = [key for _, key in followed if key not in art.base_follow]
            if missing:
                raise SidecarError(f"base_follow: no recorded base pose for the followed waypoints {missing}")
            base_poses = [[float(v) for v in art.base_follow[key]] for _, key in followed]
            path = (art.extra or {}).get("base_follow_path") or {}
            follow_path = [dict(p) for p in (path.get("poses") or [])]
            if len(follow_path) < 2:
                raise SidecarError("base_follow_path.poses: follow_mode 'hybrid' needs the recorded dense path (>= 2 poses)")
            if path.get("policy") == "sequential" and path.get("segments"):  # owner 2026-09-20 20:19: typed segments, driven strictly one at a time
                follow_segments = [dict(g) for g in path["segments"]]
        return {
            "mode": art.follow_mode,
            "fractions": fractions,
            "q_values": [art.q_at(f) for f in fractions],
            "handle_poses": handle_poses,
            "base_poses": base_poses,
            "follow_path": follow_path,
            "follow_segments": follow_segments,
        }

    def end_pose(self, sidecar, state):
        return self.targets(sidecar, state)["handle_poses"][-1]

    def annotate(self, sidecar):
        art = self.articulation
        obj = sidecar.language.phrase(art.object)
        how = {"base": "driving the base", "hybrid": "driving the base and moving the arm together"}.get(art.follow_mode, "moving the arm")
        fine = f"{self.direction} {obj} by {how} while holding its handle with the {_hand(self.arm)}"
        # a stage-2 hand-over (``articulation.follow_to``): the hand's part ends at q(follow_to), the program's push does the rest
        handover = (art.extra or {}).get("follow_to")
        q_end = art.q_at(float(handover)) if handover is not None else art.q_goal
        if art.joint_type == "prismatic":
            span = f"{art.q_init:.2f} m to {q_end:.2f} m ({_cm(q_end - art.q_init)})"
        else:
            span = f"{art.q_init:.2f} rad to {q_end:.2f} rad ({_deg(abs(q_end - art.q_init))})"
        if handover is not None:
            passive = ", letting go there for the push that takes it on"
        else:
            passive = f", letting go at {art.passive_from * 100:.0f} % where it continues on its own" if art.passive_from is not None else ""
        numbers = f"{fine}: joint {art.joint} from {span}{passive}"
        return fine, numbers


@dataclass(kw_only=True)
class PushClose(Action):
    """Close an open room door by pushing its **outer** face (the side the robot stands on) with the closed hand, no
    grasp (owner decision B, 2026-09-20 20:19): per chunk of ``chunk_deg`` a base pose in front of (or beside) the
    contact is chosen by the backend IK of the pre-contact, the contact and the arc waypoints, the base drives there
    with the arms at "ready", the hand approaches the face guarded along −m, pushes along the arc about the hinge to the
    arm's limit and backs off; repeated until the joint is within ``closed_tol_rad`` of ``q_lower`` or a chunk gains
    nothing. The leaf geometry is derived from the recorded handle waypoints (``programs.door_leaf_geometry``).

    targets: ``{"leaf": {"hinge_xy", "handle_radius_m", "q_from", "q_to"}}`` — geometry only: the push stances and
    contacts depend on where the leaf is at the start of each chunk, so none of them is an eef pose of the record.

    ``plan`` (owner 2026-10-12 ~04:20, a cabinet door's push finish; ``programs.push_finish_chunks``): the sampler's chunks
    (``articulation.push_finish``: stance base pose, fist contact at ``q_from``, hinge, swing sign, landing ``q_to``) replace the
    stance search - the recorded contact turns with the leaf to its measured angle, the arc runs to ``q_to``, no base drag; the
    last chunk is repeated ``plan_repeats`` times while it still gains. ``tol_span_rad``: the span the postcondition's §8
    tolerance is taken from (None: ``|q_from − q_lower|``; the finish: the instance's ``|q_goal − q_init|``).

    The leaf of a cabinet door closed room-door style (owner 2026-10-12, ``programs.push_close_options``; None / 0 = the room door's
    estimate): ``leaf_width_m`` hinge line → free edge (None: the handle radius + 3.5 cm), ``face_offset_m`` the outer face in
    front of the hinge line (None: half the leaf link's AABB thickness), ``handle_standoff_m`` the handle contact in front of the
    hinge line (None: ``execute.HANDLE_STANDOFF_M``, a lever 6 cm off the face), ``q_closed`` the joint value of the closed leaf
    (``handle_xy_closed``), ``contact_clearance_m`` the push contact in front of the outer face (the pad centre goes on it; 0 for a
    room door).
    """

    name: str = "push_close"
    arm: str
    object: str
    joint: str
    hinge_xy: Tuple[float, float]
    handle_xy_closed: Tuple[float, float]
    approach_closed: Tuple[float, float, float]
    handle_radius_m: float
    handle_z: float
    q_from: float
    q_lower: float = 0.0
    leaf_link: str = "leaf"
    push_inset_m: float = 0.15
    push_height_drop_m: float = 0.05
    hook_gap_m: float = 0.035
    push_pre_m: float = 0.10
    chunk_deg: float = 20.0
    push_n: int = 5
    closed_tol_rad: float = 0.05
    max_chunks: int = 10
    drag_m: float = 0.20  # base drive along −m with the fist on the face when the arm's arc gains < 2°
    push_arms: Tuple[str, ...] = ("right", "left")
    plan: Optional[List[Dict[str, Any]]] = None
    plan_repeats: int = 2
    tol_span_rad: Optional[float] = None
    leaf_width_m: Optional[float] = None
    face_offset_m: Optional[float] = None
    handle_standoff_m: Optional[float] = None
    q_closed: float = 0.0
    contact_clearance_m: float = 0.0
    qa_exempt: frozenset = frozenset({QA_FAILED_GRASP})
    TARGET_KEYS = ("leaf",)

    def __post_init__(self):
        self.arms = ("left", "right")
        self.objects = (self.object,)
        super().__post_init__()

    def targets(self, sidecar, state):
        leaf = {"hinge_xy": [float(v) for v in self.hinge_xy], "handle_radius_m": float(self.handle_radius_m), "q_from": float(self.q_from), "q_to": float(self.q_lower)}
        if self.plan:
            leaf["planned_chunks"] = len(self.plan)
        return {"leaf": leaf}

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        if self.plan:
            fine = f"push {obj} the rest of the way shut with the closed {_hand(self.arm)} on its outer face"
            return fine, f"{fine}: joint {self.joint} from {self.q_from:.2f} rad to {self.q_lower:.2f} rad ({_deg(abs(self.q_from - self.q_lower))}), in one arc about the hinge"
        fine = f"close {obj} by pushing its outer face with the closed hand"
        return fine, f"{fine}: joint {self.joint} from {self.q_from:.2f} rad to {self.q_lower:.2f} rad ({_deg(abs(self.q_from - self.q_lower))}), pushed in arcs of {self.chunk_deg:.0f}° about the hinge"


#: ``contacts.push.kind`` of a base push (IORS nextto large-scale, sampler ``iors_large_nextto``)
BASE_PUSH_KIND = "base_push"


@dataclass(kw_only=True)
class BasePush(Action):
    """Push a floor-standing object along a straight floor line by driving the base, the closed hand on its push face
    (``contacts.push`` with ``kind: "base_push"``, owner 2026-10-12 nextto_large): ``gripper(close)`` → ``arm_to(pre)`` → guarded
    approach along the push direction until the object moves or the hand is blocked (the room door's evidence rule: a blocked
    hand counts only when a further 1 cm push moves the object) → the arm held fixed relative to the base while the base drives
    along the recorded line ``base_start → base_end`` for ``distance_m``; a lagging object is re-contacted, a sideways drift beyond
    ``drift_tol_m`` (``contacts.push.drift_tol_m``) stops the drive; the drive stops ``nextto_push_on_m`` after ``partner`` is
    NextTo the object, or when the distance is done. Postcondition: the object moved (``no_motion`` otherwise).

    targets: ``{"pre": Pose, "approach": [Pose, ...], "contact": Pose, "base_path": [[x, y, yaw], [x, y, yaw]], "direction": [x, y, z],
    "distance_m": float, "second": None | {"arm": str, "pre": Pose, "approach": [Pose, ...], "contact": Pose}}`` — the pre-contact and the
    contact at the push stance (the approach densified with ``line_waypoints``), the base line the drive follows, and the other hand's
    of a two-hand push. ``end_stance``: the recorded stance where the base ends (the program check moves there).
    :meth:`end_pose` is where the hand is when the state can tell (the drive may stop before the recorded end), else the recorded
    end contact ``contacts.push.goal``.

    ``second`` (owner 2026-10-12 ~20:17 item C, ``contacts.push.second_hand``): the other closed hand on the same face (its pre / init /
    goal poses; the direction and distance are the push's) - both hands are brought to their pre-contacts, approach until both touch
    and are held through the drive; :meth:`hand` is the push as one hand sees it (that hand's ``Retreat.after``)."""

    name: str = "base_push"
    contact: PushContact
    base_yaw: Optional[float] = None
    partner: Optional[str] = None
    end_stance: Optional[str] = None
    second: Optional[PushContact] = None
    qa_exempt: frozenset = frozenset({QA_FAILED_GRASP})
    TARGET_KEYS = ("pre", "approach", "contact", "base_path", "direction", "distance_m", "second")

    def __post_init__(self):
        if self.second is not None and self.second.arm == self.contact.arm:
            raise SidecarError(f"base_push: the second hand must be the other arm, both are {self.contact.arm!r}")
        self.arms = (self.contact.arm,) + ((self.second.arm,) if self.second is not None else ())
        self.objects = (self.contact.object,)
        super().__post_init__()

    @property
    def contacts(self) -> List[PushContact]:
        """The hands' records, the first hand first."""
        return [self.contact] + ([self.second] if self.second is not None else [])

    def hand(self, arm: str) -> "BasePush":
        """The push as ``arm``'s hand sees it: this action for the first hand, a one-hand copy on the second hand's record for the other."""
        if arm == self.contact.arm:
            return self
        if self.second is None or self.second.arm != arm:
            raise SidecarError(f"base_push: no {arm} hand on {self.contact.object}")
        return replace(self, contact=self.second, second=None)

    @property
    def base_line(self) -> Tuple[List[float], List[float]]:
        x = self.contact.extra
        for k in ("base_start", "base_end"):
            v = x.get(k)
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise SidecarError(f"contacts.push.{k}: a base push records [x, y, yaw], got {v!r}")
        return [float(v) for v in x["base_start"]], [float(v) for v in x["base_end"]]

    def targets(self, sidecar, state):
        C = _conventions()
        c, s = self.contact, self.second
        start, end = self.base_line
        second = None if s is None else {"arm": s.arm, "pre": s.pre, "approach": _line(C, s.pre, s.init), "contact": s.init}
        return {"pre": c.pre, "approach": _line(C, c.pre, c.init), "contact": c.init, "base_path": [start, end], "direction": list(c.direction),
                "distance_m": c.distance_m, "second": second}

    def end_pose(self, sidecar, state):
        eef = getattr(state, "eef_pose", None)
        if callable(eef):
            try:
                return _as_pose(eef(self.contact.arm))
            except Exception:  # noqa: BLE001 - no live hand (the sampler's check after a travel / contact): the record
                pass
        return self.contact.goal

    def annotate(self, sidecar):
        c = self.contact
        obj = sidecar.language.phrase(c.object)
        to = f" next to {sidecar.language.phrase(self.partner)}" if self.partner else ""
        word = direction_word(c.direction, self.base_yaw)
        hands = f"the closed {_hand(c.arm)}" if self.second is None else "both closed hands"
        fine = f"push {obj} {word} across the floor{to} by driving the base, {hands} on it"
        h = c.extra.get("push_height_m")
        at = f" at {_cm(float(h))} height" if isinstance(h, (int, float)) else ""
        return fine, f"push {obj} {_cm(c.distance_m)} {word} across the floor{to} by driving the base, {hands} on it{at} from {_xyz(c.init.position)}"


@dataclass(kw_only=True)
class Retreat(Action):
    """Move the gripper clear of the object along −a from where the preceding action left it (``after.end_pose``):
    ``gripper(open)`` first when ``release`` (a brace or a held handle), then ``arm_along(retreat)``; postcondition:
    eef clear by ``distance`` (``RETREAT_DIST``, or ``RETREAT_LITTLE`` "away a little" after a place).

    targets: ``{"retreat": [Pose, ...]}`` — straight segment to ``pre_pose(start, distance)`` (``line_waypoints``); two legs with
    ``lift_first_m`` (up, then along ``slide_dir``) or ``lift_after_m`` (along ``slide_dir``, then up).
    """

    name: str = "retreat"
    arm: str
    object: str
    after: Action
    distance: float = RETREAT_DIST
    release: bool = False
    #: block design packing tail (owner 2026-09-19 21:47): the radial distance of the contact escape (``execute_retreat``) when
    #: it must be shorter than ``max(distance, 2 x contact_retreat_escape_m)`` - the 10 cm escape opposite a wall push ran the
    #: arm 14 mm past its reach (0920-192950 action 33)
    escape_far: Optional[float] = None
    #: oracle-fix 2026-11-05 (the hardback's overhang pick, diagnostic hbD1: the 5 cm escape off the desk left the hand at the book's far
    #: end once the base backed off to the grasp stance, and no transit to the pre-grasp planned with the desk present): the contact
    #: escape's leg off the supports (``escape_direction``) is this long instead of ``contact_retreat_escape_m`` when set
    escape_up_m: Optional[float] = None
    #: owner 2026-09-20 21:20 (the lever's fist leaving the flaps): first straight UP by ``lift_first_m`` (off the flap the palm
    #: presses), then ``distance`` along ``slide_dir`` (world unit vector; the leg along -a when None) - so the palm does not drag
    #: the box against the other hand's grasp
    lift_first_m: Optional[float] = None
    slide_dir: Optional[List[float]] = None
    #: owner 2026-10-13 ~04:30 item 1a (the push-under retreat): the other order - first ``distance`` along ``slide_dir`` at the height of
    #: the preceding action's recorded end (back out through the corridor the push swept), then straight UP by ``lift_after_m``; both
    #: legs end at poses taken from that recorded end, the poses the sampler checks and records the witness of
    lift_after_m: Optional[float] = None
    TARGET_KEYS = ("retreat",)

    def __post_init__(self):
        self.arms = (self.arm,)
        self.objects = (self.object,)
        super().__post_init__()
        if self.lift_after_m is not None and (self.slide_dir is None or self.lift_first_m is not None):
            raise ValueError("retreat: lift_after_m needs slide_dir and excludes lift_first_m")

    def _slid_path(self, C, start: Pose, recorded: Pose) -> Optional[List[Pose]]:
        """The slide-then-lift retreat: from where the hand is to ``distance`` along ``slide_dir`` from the recorded end, at its height,
        then up ``lift_after_m``."""
        if self.lift_after_m is None:
            return None
        d = C.unit(_tensor(self.slide_dir).reshape(-1)[:3].float())
        p_mid = _tensor(recorded.position).reshape(-1)[:3].float() + d * float(self.distance)
        mid = Pose([float(v) for v in p_mid.tolist()], list(recorded.orientation))
        end = Pose([float(v) for v in (p_mid + _tensor([0.0, 0.0, float(self.lift_after_m)])).tolist()], list(recorded.orientation))
        return _line(C, start, mid) + _line(C, mid, end)

    def _lifted_path(self, C, start: Pose, recorded: Pose) -> Optional[List[Pose]]:
        """The two-leg retreat: up ``lift_first_m`` from where the hand is, then ``distance`` along ``slide_dir`` (or -a)."""
        if self.lift_first_m is None:
            return None
        up = _tensor([0.0, 0.0, float(self.lift_first_m)])
        p0 = _tensor(start.position).reshape(-1)[:3].float()
        p1 = p0 + up
        mid = Pose([float(v) for v in p1.tolist()], list(start.orientation))
        if self.slide_dir is not None:
            d = C.unit(_tensor(self.slide_dir).reshape(-1)[:3].float())
            p2 = p1 + d * float(self.distance)
            end = Pose([float(v) for v in p2.tolist()], list(start.orientation))
        else:
            end = _pre(C, Pose([float(v) for v in (_tensor(recorded.position).reshape(-1)[:3].float() + up).tolist()], list(recorded.orientation)), self.distance)
        return _line(C, start, mid) + _line(C, mid, end)[1:]

    def _segment(self, sidecar, state) -> Tuple[Pose, Pose]:
        """From where the hand **is** (``state.eef_pose`` at run time: a brace / push pad pressing on its object sits
        a few cm short of the recorded contact under the load — a retreat that first drove to the recorded contact
        shoved the painting 0.6 rad off its goal) straight to the **recorded** clear pose ``pre_pose(recorded end,
        distance)`` — the pose the sampler validated and the retreat witness belongs to (a leg ending ``distance``
        from the actual pose instead ran into a joint limit 1.2 cm short, axial 0916-202403); from the recorded end
        pose when the state has none (the sampler's check, the stub state)."""
        C = _conventions()
        recorded = self.after.end_pose(sidecar, state)
        if recorded is None:
            raise SidecarError(f"retreat: the preceding action {self.after.name} leaves no eef pose")
        clear = _pre(C, recorded, self.distance)
        start = None
        eef = getattr(state, "eef_pose", None)
        if callable(eef):
            try:
                start = _as_pose(eef(self.arm))
            except Exception:  # noqa: BLE001
                start = None
        if start is None:
            return recorded, clear
        return start, clear

    def targets(self, sidecar, state):
        C = _conventions()
        start, end = self._segment(sidecar, state)
        recorded = self.after.end_pose(sidecar, state) or start
        lifted = self._lifted_path(C, start, recorded)
        if lifted is not None:
            return {"retreat": lifted}
        slid = self._slid_path(C, start, recorded)
        if slid is not None:
            return {"retreat": slid}
        slide = self._edge_escape(C, start)
        if slide is not None:
            return {"retreat": _line(C, start, slide) + _line(C, slide, end)[1:]}
        return {"retreat": _line(C, start, end)}

    EDGE_ESCAPE_M: ClassVar[float] = 0.04  # the pads sit 1.2 cm past a contact 1.5 cm inside the edge: 4 cm clears (6 cm was untrackable, 0919-065124)

    def _edge_escape(self, C, start: Pose) -> Optional[Pose]:
        """A released **edge pinch** (a flap / lid pinched across its thickness, ``articulation.handle_kind`` ``edge_lid``)
        first slides ``EDGE_ESCAPE_M`` radially away from the hinge in the flap's plane so the open pads clear the free
        edge; the clear leg along the approach axis alone moved the upper pad over the rising side of the tilted flap and
        pressed it down 0.25 rad (cdmmwy 0919-062149 / 063134). Needs ``articulation.hinge_world`` / ``axis_world``."""
        after = self.after
        art = getattr(after, "articulation", None) if after is not None and after.name == "base_follow" else None
        extra = (getattr(art, "extra", None) or {}) if art is not None else {}
        if extra.get("handle_kind") != "edge_lid" or extra.get("hinge_world") is None or extra.get("axis_world") is None:
            return None
        hinge = C.as_tensor(extra["hinge_world"]).reshape(-1)[:3].float()
        axis = C.unit(C.as_tensor(extra["axis_world"]).reshape(-1)[:3].float())
        p = _tensor(start.position).reshape(-1)[:3].float()
        radial = p - hinge
        radial = radial - axis * float(radial @ axis)
        if float(radial.norm()) < 1e-6:
            return None
        radial = radial / radial.norm()
        return Pose([float(v) for v in (p + radial * self.EDGE_ESCAPE_M).tolist()], list(start.orientation))

    def end_pose(self, sidecar, state):
        if self.lift_first_m is not None or self.lift_after_m is not None:
            C = _conventions()
            start, _ = self._segment(sidecar, state)
            recorded = self.after.end_pose(sidecar, state) or start
            path = self._lifted_path(C, start, recorded) or self._slid_path(C, start, recorded)
            if path:
                return path[-1]
        return self._segment(sidecar, state)[1]

    def annotate(self, sidecar):
        obj = sidecar.language.phrase(self.object)
        release = f"release {obj} and " if self.release else ""
        fine = f"{release}move the {_hand(self.arm)} away from {obj}"
        if self.lift_after_m is not None:
            back = "back along the push" if getattr(self.after, "name", "") == "push" else "sideways"
            return fine, f"{fine}: {_cm(self.distance)} {back} at its height, then {_cm(self.lift_after_m)} up"
        numbers = f"{fine} by {_cm(self.distance)} along the approach axis"
        return fine, numbers


VOCABULARY: Tuple[type, ...] = (Navigate, Transition, SetTorso, Grasp, Lift, Reorient, Place, Push, Brace, BaseFollow, PushClose, BasePush, Retreat)
ACTION_NAMES: Tuple[str, ...] = tuple(cls.__dataclass_fields__["name"].default for cls in VOCABULARY)

__all__ = [
    "PRE_DIST", "RETREAT_DIST", "RETREAT_LITTLE", "N_WAYPOINTS", "N_INHAND", "N_YAW_ALIGN", "N_FLIP", "QA_FAILED_GRASP",
    "State", "Action", "Navigate", "Transition", "SetTorso", "Grasp", "Lift", "Reorient", "Place", "Push", "Brace",
    "BaseFollow", "PushClose", "BasePush", "BASE_PUSH_KIND", "Retreat", "VOCABULARY", "ACTION_NAMES", "direction_word", "yaw_align_deg", "block_reorients", "block_centre_pose",
]
