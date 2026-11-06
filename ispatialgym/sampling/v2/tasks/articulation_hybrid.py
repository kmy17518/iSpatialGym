"""Hybrid base-follow for large-scale articulation doors (spec §5.3, owner decision A4 ``room_door`` → hybrid; D-011 C,
2026‑09‑17 01:00: the hybrid follow for every door class behind ``generation.follow_mode_override: "hybrid"``).

Why: from a fixed manipulation stance the arm alone follows 9–33 % of a door swing (top_cabinet 9–18 %, dishwasher
11–33 %, oven 0 %); the rigid eef→base sweep of ``follow_mode base`` (the base orbits the hinge at ≈ 1 m) is blocked by
the kitchen for revolute doors. The hybrid follow lets the **arm absorb what it can and moves the base only when it must**:
along a dense grid of joint values q_i from q_init to the last required waypoint (sub-steps ≤ 10° / 5 cm that contain
the §5.3 fractions), the handle pose h(q_i) is solved from the current base b_{i−1} first; when the arm does not reach
it, a new base pose b_i is searched **near the current base** — the interpolations toward the rigid-sweep pose
b_rigid(q_i) = h(q_i) ∘ T_eef→base(q*) (the smallest fraction that works = minimum displacement), then a lattice of
(x, y, yaw) offsets around b_{i−1} in order of displacement. Every candidate must stand on free floor (un-eroded), keep
its footprint off the door leaf at q_i and off the fixed body (planar), admit an exact arm IK for h(q_i) with the
joint-limit margin (``IKChecker.away_from_limits``), be whole-body collision-free at the arriving configuration in the
world with the **leaf removed but the body present** (link-level ignore: the gripper straddles the handle), and the base
path b_{i−1} → b_i (0.1 m strides, arm joints interpolated) must be collision-free with the leaf clear of the footprint
at both ends.

What is recorded (``articulation`` block, mirrors ``follow_mode base``): ``follow_mode: "hybrid"``, ``base_follow``
= ``{k: [x, y, yaw]}`` for the §5.3 fractions the oracle drives through (as the drawer's base path), and
``base_follow_path`` = the dense path ``poses[]`` (``k``, ``q``, ``base_pose2d``, ``arm_q`` 7 arm joints, the eef pose,
the witness quality) the oracle's ``BaseFollow`` drives + tracks without re-planning: between consecutive poses the
base pose and the arm joints are interpolated together (≤ 10° of joint motion per segment keeps the eef within a few
mm of the handle arc). The program check (:func:`check_program_follow`) verifies every recorded pose as a witness — FK
onto h(q_i), collision-free with the leaf at q_i, floor, margin — and hands the end configuration to the tracker so the
retreat is checked from where the base ends, not from the manipulation stance.

Single call site: :meth:`ArticulationLarge._solve_from` (``articulation_large.py``) calls :func:`search_follow_path`
when ``self.mode == "hybrid"``; :func:`program_check.check_action_targets` calls :func:`check_program_follow` for a
``base_follow`` action whose targets carry ``mode == "hybrid"``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import prompt0 as P0
from ispatialgym.sampling.v2.stances import Stance

log = logging.getLogger("ispatialgym.sampling.v2")

Pose2D = Tuple[float, float, float]

#: ``generation.hybrid_follow.*`` knobs and their defaults
DEFAULTS: Dict[str, Any] = {
    "step_revolute_deg": 10.0,  # dense joint grid: sub-steps of the §5.3 fraction intervals no larger than this
    "step_prismatic_m": 0.05,
    "alphas": [0.25, 0.5, 0.75, 1.0],  # interpolation fractions b_{i-1} -> b_rigid(q_i) tried first (smallest = min displacement)
    "lattice_step_m": 0.10,  # fallback lattice of base positions around b_{i-1}
    "lattice_radius_m": 0.50,
    "yaw_step_deg": 15.0,  # yaw candidates between the current yaw and the rigid-sweep yaw, this far apart
    "yaw_extra_deg": 15.0,  # one extra yaw step beyond either end
    "yaw_weight_m_per_rad": 0.30,  # displacement cost = |dxy| + w * |dyaw|
    "max_ik_per_step": 40,  # exact IK calls per dense step before the step is infeasible
    "min_margin_rad": 0.15,  # preferred joint-limit margin of a follow pose: the closest-to-seed IK solution with >= this wins,
    # else the best-margin one >= IKChecker.JOINT_LIMIT_MARGIN (top_cabinet 0917-014213: the retreat from the path's end
    # found left_arm_joint7 within 14 mrad of its limit on every solution -> ik_joint_limit at the program check)
    "hard_margin_rad": 0.05,  # the floor below which no IK solution is taken (IKChecker.JOINT_LIMIT_MARGIN; the program check's floor is 0.025)
    "check_retreat": True,  # the last pose must also admit the retreat pose (pre_pose of h(q_last), RETREAT 0.10 m back)
    # base-move policy when the arm alone cannot reach the next pose: "min_displacement" = the first feasible candidate in
    # order of displacement (many small moves); "lookahead" = among the first ``lookahead_candidates`` feasible ones, the base
    # pose from which the most of the next ``lookahead_steps`` dense poses are reachable arm-only (owner 2026-09-20 15:54:
    # "continuous pull until the limit, one base move, pull again" - fewest base moves)
    "base_move_policy": "min_displacement",
    # follow policy (owner 2026-09-20 20:19): "hybrid" = the tracker above (base and arm move together per dense step);
    # "sequential" = pull with the arm to its limit, stop, move the base with the door stationary (anchored: the hand stays
    # on the handle while the base drives; else release -> move -> re-grasp), pull again - the fewest base moves
    "policy": "hybrid",
    "move_stride_m": 0.05,  # base path strides of a release -> move -> re-grasp move (whole-body collision check per stride)
    "retreat_margin_slack_rad": 0.012,  # the search's retreat probe demands this much more margin than the program check's floor
    # sequential release: the hand retreats along -a by the first of these distances whose pose has an IK with the margin (the
    # path's final pose keeps the program's PRE_DIST retreat). The 0.10 m retreat alone cut every arm segment to one pose - the
    # elbow folds past its limit when the handle has come close - and cost a base move per 8 deg (dishwasher_wide 0921-030918)
    "release_retreat_m": [C.PRE_DIST, 0.07, 0.05],
    "min_move_m": 0.10,  # a sequential base move shorter than this AND turning less than min_move_deg is not worth its release / re-grasp
    # oracle-fix 2026-11-03 (owner: the dishwasher follows ARM ONLY, else SEQUENTIAL - never the base): a sequential arm segment ends once
    # it pulled the joint this far from where it began (rad; None = to the arm's limit, as before). The torso sags with the length of an
    # arm-only pull (dishwasher open 0075: past the 1 deg robot_wobble gate after 0.43 rad of travel, 0074 0.91 deg after 0.45 rad)
    "seq_max_pull_rad": None,
    "min_move_deg": 10.0,
    # every IK solution of the search must clear cuRobo's self-collision padding by this much (the arm embodiment, the trunk
    # where the stance puts it): the demo's trunk is not exactly there and the oracle refuses to plan from a start
    # configuration inside the padding (dishwasher_wide 0921-044216, link5 - torso_link4 by 9 mm at the recorded retreat)
    "self_clearance_m": 0.02,
    "lookahead_candidates": 5,
    "lookahead_steps": 8,
    "check_fov": True,  # the fingertips at h(q) must lie in the head camera frame from the base pose (QA gripper_out_of_fov, spec 3.4;
    # dishwasher close 0920-011633: the hand rose out of the frame's lower right, depth 0.38 m, as the base moved in)
    "fov_margin": 0.08,  # frame margin (fraction) for that check; the QA monitor's is 0.05
    "reach_fraction": 1.0,  # torso-top-to-eef distance pre-filter: <= this fraction of the straight arm's + reach_extra_m
    "reach_extra_m": 0.10,  # (the arm mounts off the torso top: the straight-arm distance measured there is not the maximum)
    "min_reach_m": 0.15,
    "base_radius_m": 0.33,  # footprint disc of the leaf stand-in (R1Pro chassis half-diagonal)
    "body_radius_m": 0.30,  # footprint disc against the fixed body's planar AABBs
    "body_height_m": 1.20,  # fixed-body links above this height (over the floor) do not block the base (a top cabinet)
    "budget_s": 600.0,  # per-search wall-clock budget
    "record_eef": True,
}


def knobs(gen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``DEFAULTS`` updated by ``generation.hybrid_follow`` (unknown keys kept for the record)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (gen or {}).items() if v is not None})
    return out


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def dense_values(q_init: float, q_last: float, marks: Sequence[float], step: float) -> List[float]:
    """Joint values from ``q_init`` (excluded) to ``q_last`` (included) that contain every mark in ``marks`` (the §5.3
    waypoint values between them, either direction) and split every interval into equal sub-steps no larger than
    ``step``. Empty when ``q_last == q_init``."""
    q0, q1 = float(q_init), float(q_last)
    if abs(q1 - q0) < 1e-12:
        return []
    sign = 1.0 if q1 > q0 else -1.0
    nodes = sorted({q0, q1, *[float(m) for m in marks if (m - q0) * sign > 1e-9 and (q1 - m) * sign > -1e-9]}, key=lambda v: sign * v)
    out: List[float] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        n = max(1, int(math.ceil(abs(b - a) / max(float(step), 1e-9) - 1e-9)))
        out += [a + (b - a) * j / n for j in range(1, n + 1)]
    return [float(v) for v in out]


def pose2d_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """``(planar distance m, |yaw change| rad)`` between two base poses."""
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])), abs(C.wrap_angle(float(b[2]) - float(a[2])))


def displacement_cost(a: Sequence[float], b: Sequence[float], yaw_weight: float) -> float:
    d, dyaw = pose2d_delta(a, b)
    return d + float(yaw_weight) * dyaw


def pose2d_lerp(a: Sequence[float], b: Sequence[float], f: float) -> Pose2D:
    """Planar interpolation ``a → b`` (yaw the shortest way)."""
    f = float(f)
    dyaw = C.wrap_angle(float(b[2]) - float(a[2]))
    return (float(a[0]) + (float(b[0]) - float(a[0])) * f, float(a[1]) + (float(b[1]) - float(a[1])) * f, C.wrap_angle(float(a[2]) + dyaw * f))


def yaw_candidates(yaw_from: float, yaw_to: float, step: float, extra: float) -> List[float]:
    """Yaws from ``yaw_from`` toward ``yaw_to`` every ``step`` rad (both ends included), plus one ``extra`` step beyond
    each end; ``yaw_from`` first, then increasing distance from it."""
    d = C.wrap_angle(float(yaw_to) - float(yaw_from))
    sign = 1.0 if d >= 0 else -1.0
    vals = [0.0]
    n = int(math.floor(abs(d) / max(float(step), 1e-9) + 1e-9))
    vals += [sign * float(step) * j for j in range(1, n + 1)]
    if abs(abs(d) - float(step) * n) > 1e-6:
        vals.append(d)
    if extra > 0:
        vals.append(d + sign * float(extra))
        vals.append(-sign * float(extra))
    seen: List[float] = []
    for v in sorted(vals, key=abs):
        if all(abs(v - s) > 1e-6 for s in seen):
            seen.append(v)
    return [C.wrap_angle(float(yaw_from) + v) for v in seen]


def lattice_candidates(b_prev: Sequence[float], b_rigid: Sequence[float], step_m: float, radius_m: float, yaw_step: float, yaw_extra: float,
                       yaw_weight: float, alphas: Sequence[float] = ()) -> List[Tuple[float, Pose2D, str]]:
    """Base-pose candidates for one dense step, ``(cost, pose, tag)`` sorted by the displacement cost from ``b_prev``:
    the interpolations ``b_prev → b_rigid`` at ``alphas`` (tag ``alpha=<f>``) and a lattice of positions on a
    ``step_m`` grid within ``radius_m`` of ``b_prev`` (world axes) crossed with :func:`yaw_candidates` between the
    current yaw and the rigid-sweep yaw (tag ``lattice``); ``b_prev`` itself is not a candidate. Duplicates (same pose
    within 5 mm / 0.5°) are dropped, the earlier tag wins."""
    out: List[Tuple[float, Pose2D, str]] = []
    for f in alphas:
        p = pose2d_lerp(b_prev, b_rigid, f)
        out.append((displacement_cost(b_prev, p, yaw_weight), p, f"alpha={float(f):g}"))
    yaws = yaw_candidates(float(b_prev[2]), float(b_rigid[2]), yaw_step, yaw_extra)
    n = int(math.floor(float(radius_m) / max(float(step_m), 1e-9) + 1e-9))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx, dy = i * float(step_m), j * float(step_m)
            if math.hypot(dx, dy) > float(radius_m) + 1e-9:
                continue
            for yaw in yaws:
                p = (float(b_prev[0]) + dx, float(b_prev[1]) + dy, yaw)
                if i == 0 and j == 0 and abs(C.wrap_angle(yaw - float(b_prev[2]))) < 1e-9:
                    continue
                out.append((displacement_cost(b_prev, p, yaw_weight), p, "lattice"))
    out.sort(key=lambda t: t[0])
    kept: List[Tuple[float, Pose2D, str]] = []
    seen = set()
    for c in out:
        key = (round(c[1][0] / 5e-3), round(c[1][1] / 5e-3), round(C.wrap_angle(c[1][2]) / math.radians(0.5)))
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


def disc_clear_of_boxes(xy: Sequence[float], radius: float, boxes: Iterable[Sequence[float]]) -> bool:
    """No planar AABB ``(xmin, ymin, xmax, ymax)`` in ``boxes`` intersects the disc of ``radius`` about ``xy``."""
    x, y = float(xy[0]), float(xy[1])
    for bx in boxes:
        dx = max(float(bx[0]) - x, 0.0, x - float(bx[2]))
        dy = max(float(bx[1]) - y, 0.0, y - float(bx[3]))
        if math.hypot(dx, dy) < float(radius):
            return False
    return True


def leaf_clear_of_base(leaf_pts_xy, base_xy: Sequence[float], radius: float) -> bool:
    """Planar stand-in of "the grasped link ignored for the gripper only" (as ``articulation_large.leaf_clear_of_base``):
    no point of the moving link's footprint lies inside the base's footprint disc."""
    if leaf_pts_xy is None or len(leaf_pts_xy) == 0:
        return True
    pts = th.as_tensor(leaf_pts_xy, dtype=th.float32).reshape(-1, 2)
    d = th.linalg.norm(pts - th.tensor([float(base_xy[0]), float(base_xy[1])]), dim=1)
    return bool((d > float(radius)).all())


def interpolate_arm(qa: th.Tensor, qb: th.Tensor, f: float) -> th.Tensor:
    return th.as_tensor(qa, dtype=th.float32) + (th.as_tensor(qb, dtype=th.float32) - th.as_tensor(qa, dtype=th.float32)) * float(f)


def fraction_of(q: float, q_init: float, q_goal: float) -> float:
    span = float(q_goal) - float(q_init)
    return 0.0 if abs(span) < 1e-12 else (float(q) - float(q_init)) / span


def base_follow_dict(poses: Sequence["FollowPose"], marks: Dict[str, float], tol: float = 1e-6) -> Dict[str, List[float]]:
    """``{fraction key: [x, y, yaw]}`` — the base pose of the dense path at each §5.3 waypoint value in ``marks``
    (``key -> q_k``) that the path reaches (the drawer's ``base_follow`` record)."""
    out: Dict[str, List[float]] = {}
    for key, qk in marks.items():
        for p in poses:
            if abs(p.q - float(qk)) <= tol:
                out[key] = [float(v) for v in p.base_pose2d]
                break
    return out


# ----------------------------------------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class FollowPose:
    """One pose of the dense hybrid path: joint value ``q`` (fraction ``k`` of q_init → q_goal), the base pose the robot
    stands at and the 7 arm joints (``M.ARM_JOINTS`` order) that put the eef on h(q); ``q_full`` = the 28-D
    configuration in the sampler's root frame (not written), ``eef`` = h(q).eef (world)."""

    k: float
    q: float
    base_pose2d: Pose2D
    arm_q: List[float]
    q_full: Optional[th.Tensor] = None
    eef: Optional[Tuple[th.Tensor, th.Tensor]] = None
    fk_residual_m: Optional[float] = None
    joint_limit_margin_rad: Optional[float] = None
    base_step_m: float = 0.0
    base_step_rad: float = 0.0
    chosen: str = "stance"
    n_ik: int = 0

    def to_sidecar(self, record_eef: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {"k": round(float(self.k), 6), "q": float(self.q), "base_pose2d": [float(v) for v in self.base_pose2d],
                             "arm_q": [round(float(v), 6) for v in self.arm_q], "base_step_m": round(float(self.base_step_m), 5),
                             "base_step_rad": round(float(self.base_step_rad), 5), "chosen": self.chosen, "n_ik": int(self.n_ik)}
        if self.fk_residual_m is not None:
            d["fk_residual_m"] = round(float(self.fk_residual_m), 6)
        if self.joint_limit_margin_rad is not None:
            d["joint_limit_margin_rad"] = round(float(self.joint_limit_margin_rad), 5)
        if record_eef and self.eef is not None:
            d["eef"] = {"position": [round(float(v), 6) for v in G.t(self.eef[0]).tolist()], "orientation": [round(float(v), 7) for v in G.t(self.eef[1]).tolist()]}
        return d


@dataclass
class HybridPath:
    """The result of :func:`search_follow_path`."""

    arm: str
    trunk: Tuple[float, float]
    poses: List[FollowPose]
    base_follow: Dict[str, List[float]]
    idle_arm: Optional[str] = None
    idle_arm_q: Optional[List[float]] = None
    fingers: Optional[float] = None
    step: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    knobs: Dict[str, Any] = field(default_factory=dict)
    policy: str = "hybrid"
    segments: Optional[List[Dict[str, Any]]] = None  # sequential policy: typed segments (arm / release_move_regrasp)
    retreat_arm_q: Optional[List[float]] = None  # the search's retreat configuration (7 arm joints) at the final pose: the program check's witness

    @property
    def end_base(self) -> Pose2D:
        return self.poses[-1].base_pose2d

    def to_sidecar(self, record_eef: bool = True) -> Dict[str, Any]:
        """The ``articulation.base_follow_path`` block (``poses`` always - the program check, validation and annotation
        read them; ``segments`` with the sequential policy - the executor drives those)."""
        out = {"mode": "hybrid", "policy": self.policy, "arm": self.arm, "trunk": [float(self.trunk[0]), float(self.trunk[1])], "step": dict(self.step),
               "arm_q_order": "arm joints 1..7 (r1pro_model.ARM_JOINTS[arm])", "idle_arm": self.idle_arm,
               "idle_arm_q": None if self.idle_arm_q is None else [round(float(v), 6) for v in self.idle_arm_q],
               "fingers": None if self.fingers is None else float(self.fingers),
               "poses": [p.to_sidecar(record_eef) for p in self.poses], "stats": dict(self.stats), "knobs": {k: v for k, v in self.knobs.items() if k != "record_eef"}}
        if self.segments is not None:
            out["segments"] = self.segments
        if self.retreat_arm_q is not None:
            out["retreat_arm_q"] = [round(float(v), 6) for v in self.retreat_arm_q]
        return out


def path_stats(poses: Sequence[FollowPose], q_init: float, q_goal: float) -> Dict[str, Any]:
    moves = [p for p in poses[1:] if p.base_step_m > 1e-6 or p.base_step_rad > 1e-6]
    return {"n_poses": len(poses), "n_base_moves": len(moves), "base_path_m": round(sum(p.base_step_m for p in poses), 4),
            "base_turn_rad": round(sum(p.base_step_rad for p in poses), 4), "max_step_m": round(max([p.base_step_m for p in poses] or [0.0]), 4),
            "max_step_rad": round(max([p.base_step_rad for p in poses] or [0.0]), 4), "n_ik": int(sum(p.n_ik for p in poses)),
            "followed_fraction": round(fraction_of(poses[-1].q, q_init, q_goal), 4) if poses else 0.0,
            "arm_only_fraction": round(fraction_of(next((p.q for p in poses[1:] if p.chosen != "arm_only"), poses[-1].q), q_init, q_goal), 4) if len(poses) > 1 else 0.0,
            "total_displacement_m": round(pose2d_delta(poses[0].base_pose2d, poses[-1].base_pose2d)[0], 4) if poses else 0.0}


# ----------------------------------------------------------------------------------------------------------------
# simulator side
# ----------------------------------------------------------------------------------------------------------------


class _LinkSet:
    """Stand-in object for ``IKChecker.ignoring`` / ``_Ignoring`` that carries only some links of an articulated object
    (the moving leaf and its sub-links): the collision meshes of exactly those links leave the cuRobo world."""

    def __init__(self, name: str, links: Dict[str, Any]):
        self.name = name
        self.links = dict(links)


def leaf_links(target, jinfo: Dict[str, Any], joint) -> Dict[str, Any]:
    """The moving link of the joint (``jinfo.child_link``, else the joint prim's child body) and its recorded
    ``sub_links`` (handles fixed to the leaf), as ``name -> link``."""
    names: List[str] = []
    child = jinfo.get("child_link")
    if not child and joint is not None:
        try:
            child = str(joint.body1).split("/")[-1]
        except Exception:  # noqa: BLE001
            child = None
    if child:
        names.append(child)
    names += [s.get("link") for s in (jinfo.get("sub_links") or []) if isinstance(s, dict) and s.get("link")]
    # every link fixed to the moving link (transitively) swings with it: door_crtrpa_0's ``glass`` panel hangs on the
    # ``leaf`` by a fixed joint (not in OmniGibson's ``joints``, which lists the DoFs) and is not in the inventory's
    # ``sub_links`` - left in the collision world it moved into the arm at every dense step (IK failures) and into the
    # body at the start of every base path (0919-114353). The fixed joints are read from the object's USD ``joints`` scope.
    pairs = fixed_joint_pairs(target)
    grew = True
    while grew and pairs:
        grew = False
        for parent, kid in pairs:
            if parent in names and kid not in names:
                names.append(kid)
                grew = True
    return {n: target.links[n] for n in names if n in getattr(target, "links", {})}


def fixed_joint_pairs(target) -> List[Tuple[str, str]]:
    """``(parent link, child link)`` of every fixed joint prim under ``<target>/joints`` (USD; empty without a stage)."""
    out: List[Tuple[str, str]] = []
    try:
        import omnigibson as og

        stage = og.sim.stage
        scope = stage.GetPrimAtPath(f"{target.prim_path}/joints")
        if not scope or not scope.IsValid():
            return out
        for prim in scope.GetChildren():
            if prim.GetTypeName() != "PhysicsFixedJoint":
                continue
            b0 = prim.GetRelationship("physics:body0").GetTargets()
            b1 = prim.GetRelationship("physics:body1").GetTargets()
            if b0 and b1:
                out.append((str(b0[0]).split("/")[-1], str(b1[0]).split("/")[-1]))
    except Exception:  # noqa: BLE001 — no simulator (tests) or an object without a joints scope
        return out
    return out


def body_boxes(target, leaf: Dict[str, Any], floor_z: float, height: float) -> List[Tuple[float, float, float, float]]:
    """Planar AABBs of the target's fixed links (not in ``leaf``) whose extent overlaps the base / torso height band
    ``[floor_z + 0.05, floor_z + height]``."""
    out = []
    for name, link in getattr(target, "links", {}).items():
        if name in leaf:
            continue
        try:
            lo, hi = link.aabb
        except Exception:  # noqa: BLE001
            continue
        lo, hi = G.t(lo), G.t(hi)
        if float(hi[2]) < floor_z + 0.05 or float(lo[2]) > floor_z + height:
            continue
        out.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    return out


class LeafFootprint:
    """Planar footprint of the moving leaf at any joint value: the corners of the leaf links' AABBs captured at the
    reference joint value ``q_ref`` (the closed / init state, where the boxes are tight) moved with the joint — rotated
    about the hinge (``hinge_point_obj`` / ``axis_dir_obj`` in the world) by ``q − q_ref`` or translated along the axis —
    and projected to xy. Tighter than the world AABB of the rotated leaf (the stand-in of ``follow_mode base``: a slab at
    60° has a bounding box the size of the room corner); the sampler's AABB at q is the fallback when the record has no
    hinge (``at`` then returns the caller's ``fallback`` points)."""

    def __init__(self, target, leaf: Dict[str, Any], jinfo: Dict[str, Any], obj_pose, q_ref: float):
        pts = []
        for link in leaf.values():
            try:
                lo, hi = link.aabb
            except Exception:  # noqa: BLE001
                continue
            pts.append(G.aabb_corners(G.t(lo), G.t(hi)))
        self.corners0 = th.cat(pts) if pts else th.zeros(0, 3)
        self.q_ref = float(q_ref)
        kind = str(jinfo.get("type", "")).lower()
        self.prismatic = "prismatic" in kind or "slider" in kind
        op, oq = G.t(obj_pose[0]), G.t(obj_pose[1])
        ax = jinfo.get("axis_dir_obj")
        self.axis_w = C.unit(G.rotate(oq, G.t(ax))) if ax is not None else None
        hp = jinfo.get("hinge_point_obj")
        self.hinge_w = (op + G.rotate(oq, G.t(hp))) if hp is not None else None
        self.analytic = self.axis_w is not None and (self.prismatic or self.hinge_w is not None) and len(self.corners0) > 0

    def at(self, q: float, fallback=None) -> th.Tensor:
        """World xy points of the leaf footprint at joint value ``q`` (``fallback`` when not analytic)."""
        if not self.analytic:
            return G.t(fallback)[:, :2] if fallback is not None and len(fallback) else th.zeros(0, 2)
        d = float(q) - self.q_ref
        if self.prismatic:
            return (self.corners0 + d * self.axis_w)[:, :2]
        R = C.quat_to_mat(C.quat_from_axis_angle(self.axis_w, d))
        return ((self.corners0 - self.hinge_w) @ R.T + self.hinge_w)[:, :2]


def _ignore_set(sampler, obj, leaf: Dict[str, Any]) -> List[Any]:
    """What leaves the collision world for the handle IK / whole-body checks: the leaf links only when the backend has
    the per-mesh switch, else the whole object (the ``RobotKit.ignoring`` fallback works on objects)."""
    be = sampler.kit.backend
    if leaf and hasattr(be, "_mesh_enable") and hasattr(be, "_world_mesh_names"):
        return [_LinkSet(f"{obj.name}:leaf", leaf)]
    return [obj]


@dataclass
class _Sol:
    q: th.Tensor
    residual: float
    margin: float
    n_ik: int


class FollowSearch:
    """The per-attempt search state (see the module docstring). ``s`` is the :class:`ArticulationLarge` sampler."""

    def __init__(self, s, manip: Stance, q_star: th.Tensor, h0: C.HandleWaypoint, q_init: float, q_goal: float, gen: Optional[Dict[str, Any]] = None):
        self.s = s
        self.kit = s.kit
        self.ikc = s.ikc
        self.obj = s.target
        self.arm = s.arm
        self.manip = manip
        self.q_star = G.t(q_star).clone()
        self.h0 = h0
        self.q_init, self.q_goal = float(q_init), float(q_goal)
        self.k = knobs(gen if gen is not None else (s.gp.gen("hybrid_follow") or {}))
        self.bz = P0.base_z(s)
        self.trunk = (float(manip.translate), float(manip.tilt))
        self.idx = self.kit.robot.arm_control_idx[self.arm]
        self.leaf = leaf_links(self.obj, s.jinfo, s.joint)
        self.ignore = _ignore_set(s, self.obj, self.leaf)
        self.boxes = body_boxes(self.obj, self.leaf, float(self.kit.floor_z), float(self.k["body_height_m"]))
        self.leaf_fp = LeafFootprint(self.obj, self.leaf, s.jinfo, s.obj_pose(), self.q_init)  # the object stands at q_init here
        self._fp_warned = False
        # reference posture: the eef->base transform of the grasp solution q* (the rigid sweep of §5.3 base mode)
        self.T = C.eef_base_transform(h0.eef, C.pose2d_to_pose(*manip.base_pose2d, z=self.bz))
        # shoulder-distance pre-filter: torso_link4 (both arms mount there) in the base frame, and the straight arm's reach
        from ispatialgym.oracle import r1pro_model as M

        sh = self.kit.fk([M.TORSO_TOP_LINK], self.q_star)[M.TORSO_TOP_LINK]
        self.shoulder_in_base = P0.to_base_frame(G.t(sh[0]), manip.base_pose2d, self.bz)
        q_straight = self.q_star.clone()
        q_straight[self.idx] = 0.0
        fk = self.kit.fk([M.TORSO_TOP_LINK, M.EEF_LINK[self.arm]], q_straight)
        r_zero = float(th.linalg.norm(G.t(fk[M.EEF_LINK[self.arm]][0]) - G.t(fk[M.TORSO_TOP_LINK][0])))
        r_now = float(th.linalg.norm(G.t(h0.eef[0]) - G.t(sh[0])))
        self.r_max = max(r_zero, r_now) * float(self.k["reach_fraction"]) + float(self.k["reach_extra_m"])
        self.n_ik = 0
        self.t0 = time.time()
        self.log: List[Dict[str, Any]] = []

    # -- geometry -------------------------------------------------------------------------------------------------
    def rigid_pose(self, eef: Tuple[th.Tensor, th.Tensor]) -> Pose2D:
        return C.pose_to_pose2d(C.compose(eef, self.T))

    def shoulder_world(self, b: Sequence[float]) -> th.Tensor:
        x, y, yaw = (float(v) for v in b[:3])
        c, s_ = math.cos(yaw), math.sin(yaw)
        p = self.shoulder_in_base
        return th.tensor([x + c * float(p[0]) - s_ * float(p[1]), y + s_ * float(p[0]) + c * float(p[1]), self.bz + float(p[2])])

    def reach_ok(self, b: Sequence[float], eef_pos) -> bool:
        d = float(th.linalg.norm(G.t(eef_pos) - self.shoulder_world(b)))
        return float(self.k["min_reach_m"]) <= d <= self.r_max

    def footprint_at(self, q: float) -> th.Tensor:
        """Leaf footprint at ``q`` (the object must stand at ``q``): the oriented analytic footprint when its centroid lies
        inside the leaf's actual world AABB (inflated 10 cm), else that AABB's corners (loose but safe)."""
        aabb = self.s.moving_link_footprint()
        pts = self.leaf_fp.at(q, fallback=aabb)
        if self.leaf_fp.analytic and len(aabb) and len(pts):
            c = pts.mean(dim=0)
            lo, hi = aabb.min(dim=0).values - 0.10, aabb.max(dim=0).values + 0.10
            if not bool(((c >= lo) & (c <= hi)).all()):
                if not self._fp_warned:
                    log.warning("%s: hybrid follow: the analytic leaf footprint at q = %.3f (centroid %s) leaves the leaf's AABB %s..%s; using the AABB", self.s.gp.variant, q,
                                [round(float(v), 3) for v in c.tolist()], [round(float(v), 2) for v in lo.tolist()], [round(float(v), 2) for v in hi.tolist()])
                    self._fp_warned = True
                return aabb
        return pts

    def footprint_ok(self, b: Sequence[float], leaf_pts) -> Tuple[bool, str]:
        if not self.kit.trav_free(b[:2]):
            return False, "floor"
        if not disc_clear_of_boxes(b[:2], float(self.k["body_radius_m"]), self.boxes):
            return False, "body"
        if not leaf_clear_of_base(leaf_pts, b[:2], float(self.k["base_radius_m"])):
            return False, "leaf"
        return True, ""

    def stance_at(self, b: Sequence[float], name: str = "follow") -> Stance:
        return dataclasses.replace(self.manip, name=name, base_pose2d=(float(b[0]), float(b[1]), float(b[2])), q=None)

    # -- IK ------------------------------------------------------------------------------------------------------
    def _ik(self, eef: Tuple[th.Tensor, th.Tensor], q_seed: th.Tensor, hard_margin: Optional[float] = None) -> Optional[th.Tensor]:
        """Exact arm IK (every successful seed of the 32): the solution **closest to the seed** among those with a
        joint-limit margin >= ``min_margin_rad`` (arm continuity between consecutive poses), else the largest-margin one
        when it keeps ``IKChecker.JOINT_LIMIT_MARGIN``; None otherwise. Counted as one IK call."""
        from ispatialgym.oracle import r1pro_model as M

        self.kit._ensure_synced()
        t0 = time.time()
        sols = self.kit.backend.solve_ik(f"{self.arm}_arm", {M.EEF_LINK[self.arm]: (G.t(eef[0]), G.t(eef[1]))}, q_seed=q_seed, return_all=True, sync_world=False)
        self.kit.stats["ik_calls"] = self.kit.stats.get("ik_calls", 0) + 1
        self.kit.stats["ik_s"] = self.kit.stats.get("ik_s", 0.0) + (time.time() - t0)
        self.n_ik += 1
        if sols is None or len(sols) == 0:
            return None
        sols = G.t(sols).reshape(-1, q_seed.shape[0])
        clear = float(self.k.get("self_clearance_m", 0.0))
        if clear > 0.0:
            # the demo's trunk stands a centimetre off the recorded one and the oracle plans with the ARM embodiment (the
            # trunk locked where it really is): a witness that just clears cuRobo's self-collision padding here is inside it
            # there and no plan may start from it (dishwasher_wide 0921-044216: link5 - torso_link4 by 9 mm at the retreat)
            keep = [k for k in range(len(sols)) if not self.kit.backend.self_collision_pairs(sols[k], f"{self.arm}_arm", margin=clear)]
            if not keep:
                n_sc = getattr(self, "_n_selfclear_debug", 0)
                if n_sc < 3:
                    self._n_selfclear_debug = n_sc + 1
                    worst = self.kit.backend.self_collision_pairs(sols[0], f"{self.arm}_arm", margin=clear)[:2]
                    log.info("%s: hybrid IK rejected by the self-collision clearance (%d solutions, none %.0f mm clear of the padding; e.g. %s)", self.s.gp.variant, len(sols),
                             1000 * clear, [(p["links"], round(1000 * float(p["penetration"]) - 1000 * clear, 1)) for p in worst])
                return None
            sols = sols[keep]
        margins = th.tensor([self.ikc.joint_limit_margin(q, self.arm)[0] for q in sols])
        dist = th.linalg.norm(sols[:, self.idx] - G.t(q_seed)[self.idx], dim=1)
        good = margins >= float(self.k["min_margin_rad"])
        if bool(good.any()):
            return sols[int(th.argmin(th.where(good, dist, th.full_like(dist, float("inf")))))].clone()
        j = int(th.argmax(margins))
        # the hard floor: generation.hybrid_follow.hard_margin_rad (default IKChecker.JOINT_LIMIT_MARGIN 0.05; the program
        # check's own floor is 0.025) - door_crtrpa_0 0919-074439: 36 / 40 candidates at 17 deg had solutions, none with margin
        hard = float(self.k.get("hard_margin_rad", self.ikc.JOINT_LIMIT_MARGIN)) if hard_margin is None else float(hard_margin)
        if float(margins[j]) >= hard:
            return sols[j].clone()
        n = getattr(self, "_n_margin_debug", 0)
        if n < 3:
            self._n_margin_debug = n + 1
            nearest = [self.ikc.joint_limit_margin(q, self.arm)[1] for q in sols]
            log.info("%s: hybrid IK rejected by the joint-limit margin (%d solutions, best %.3f rad < %.3f at %s; nearest limits %s)", self.s.gp.variant, len(sols),
                     float(margins[j]), hard, nearest[j], sorted(set(nearest)))
        return None

    def _debug_polish(self, q: th.Tensor, eef: Tuple[th.Tensor, th.Tensor], residual: float) -> None:
        """Why a follow pose's polish left > 5 mm (door 0919-044806: 31 of 40 candidates at the first dense step): the raw
        IK residual, the diff-IK polish verdict, the polished configuration's collision / joint-limit-margin verdicts.
        First three per search, INFO."""
        n = getattr(self, "_n_polish_debug", 0)
        if n >= 3:
            return
        self._n_polish_debug = n + 1
        try:
            from ispatialgym.oracle import r1pro_model as M

            raw = self.ikc.fk_residual(q, eef, self.arm)
            res = self.kit.backend.solve_ik_path(f"{self.arm}_arm", [{M.EEF_LINK[self.arm]: (G.t(eef[0]), G.t(eef[1]))}], q_start=G.t(q).clone(), pos_tol=1e-3, rot_tol=5e-3,
                                                 max_iters=40, check_collisions=False, sync_world=False, max_joint_step=0.3)
            info: Dict[str, Any] = {"raw_residual_mm": round(1000 * raw, 2), "polish_success": bool(res.success), "status": str(getattr(res, "status", ""))[:80]}
            # the two kinematics the search mixes: the arm embodiment (IK / polish) vs the whole body (fk_residual) on the SAME q
            link = M.EEF_LINK[self.arm]
            p_body = G.t(self.kit.backend.fk([link], q_full=G.t(q), emb_name="whole_body")[link][0])
            try:
                p_arm = G.t(self.kit.backend.fk([link], q_full=G.t(q), emb_name=f"{self.arm}_arm")[link][0])
                info["arm_vs_body_fk_mm"] = [round(1000 * float(v), 2) for v in (p_arm - p_body).tolist()]
            except Exception as e:  # noqa: BLE001
                info["arm_vs_body_fk_mm"] = f"n/a ({type(e).__name__})"
            info["body_fk_minus_target_mm"] = [round(1000 * float(v), 2) for v in (p_body - G.t(eef[0])).tolist()]
            rp = self.kit.backend.root_pose()[0]
            info["root_z"] = round(float(rp[2]), 4)
            info["floor_z"] = round(float(self.kit.floor_z), 4)
            info["seed_base_z_joint"] = round(float(G.t(q)[self.kit.jidx["base_footprint_z_joint"]]), 4)
            if res.success and res.q_traj is not None:
                cand = res.q_traj[-1]
                info["polished_residual_mm"] = round(1000 * self.ikc.fk_residual(cand, eef, self.arm), 2)
                info["polished_free_margin0"] = bool(self.kit.collision_free(cand, margin=0.0)[0])
                m, j = self.ikc.joint_limit_margin(cand, self.arm)
                info["polished_margin_rad"] = round(float(m), 4)
                info["nearest_limit_joint"] = j
                if not info["polished_free_margin0"]:
                    info["culprits"] = P0.debug_collision_q(self.s, G.t(cand), "hybrid polish", margin=0.0).get("objects_at_spheres")
            log.info("%s: hybrid polish failure %d (residual %.1f mm): %s", self.s.gp.variant, self._n_polish_debug, 1000 * residual, info)
        except Exception as e:  # noqa: BLE001 — diagnostics never fail the search
            log.info("%s: hybrid polish debug failed: %r", self.s.gp.variant, e)

    def solve(self, b: Sequence[float], eef: Tuple[th.Tensor, th.Tensor], q_prev: th.Tensor, leaf_pts, retreat: bool = False) -> Optional[_Sol]:
        """Exact arm IK of ``eef`` from base ``b`` seeded with the previous configuration (:meth:`_ik`: margin-aware),
        polished onto the pose, whole-body collision-free at ``COLLISION_MARGIN`` (leaf removed), the leaf clear of the
        footprint; with ``retreat`` the retreat pose (``pre_pose`` of ``eef``, 0.10 m back along −a) must be reachable
        from the same base too. None when any of it fails."""
        q_seed = self.kit.q_at_base_pose(q_prev, float(b[0]), float(b[1]), float(b[2]))
        n0 = int(self.kit.stats.get("ik_calls", 0))
        self.last_solve_fail = None  # why the last solve() returned None: ik / polish / collision / retreat_ik (door 0919-031559: 39 'ik')
        with self.ikc.ignoring(self.ignore):
            q = self._ik(eef, q_seed)
            if q is None:
                self.last_solve_fail = "ik"
                return None
            q_pol, residual = self.ikc.polish_witness(q, eef, self.arm, ignore=())
            if residual > 5e-3:  # the solver's own acceptance; the program check polishes again to <= 2 mm
                self.last_solve_fail = "polish"
                self._debug_polish(q, eef, residual)
                return None
            free = bool(self.kit.collision_free(q_pol, margin=P0.COLLISION_MARGIN)[0])
            if free and retreat and self._ik(C.pre_pose(eef, C.PRE_DIST), q_pol, hard_margin=float(self.k.get("hard_margin_rad", self.ikc.JOINT_LIMIT_MARGIN)) + float(self.k["retreat_margin_slack_rad"])) is None:
                self.last_solve_fail = "retreat_ik"
                return None
        if not free:
            self.last_solve_fail = "collision"
            return None
        if bool(self.k["check_fov"]) and not self.in_view(b, eef):
            self.last_solve_fail = "fov"
            return None
        margin, _ = self.ikc.joint_limit_margin(q_pol, self.arm)
        return _Sol(q_pol, float(residual), float(margin), int(self.kit.stats.get("ik_calls", 0)) - n0)

    def in_view(self, b: Sequence[float], eef: Tuple[th.Tensor, th.Tensor]) -> bool:
        """The fingertips of the hand on ``eef`` lie in the head camera frame of the stance's trunk at base ``b``."""
        try:
            cam = P0.camera_poses(self.s, [self.stance_at(b, "view")])[0]
            return bool(P0.in_frame(self.s, cam, C.fingertip(eef).reshape(1, 3), float(self.k["fov_margin"])))
        except Exception as e:  # noqa: BLE001 - a camera model failure must not veto the path
            log.debug("hybrid in_view failed (%r): accepted", e)
            return True

    def path_ok(self, b_prev: Sequence[float], b: Sequence[float], q_prev: th.Tensor, q_new: th.Tensor, leaf_prev, leaf_now,
                joint_from: Optional[float] = None, joint_to: Optional[float] = None) -> Tuple[bool, str]:
        """The base path b_prev -> b in 0.1 m strides with the arm joints interpolated between the two solutions:
        whole-body collision-free (leaf removed) and the leaf clear of every stride footprint. The base and the door move
        **together** (the executor interpolates both over the segment), so with the joint values given and an analytic
        footprint the leaf is taken at the interpolated joint value per stride; without them the leaf at both ends is
        tested at every stride (door_crtrpa_0 0919-114353: the leaf at its NEW angle against the base's START pose rejected
        every IK-feasible candidate at stride 0 - the door had swung into where the base still stood)."""
        poses = NAV.stride_poses([tuple(b_prev), tuple(b)], NAV.STRIDE_M)
        n = max(len(poses) - 1, 1)
        interp = joint_from is not None and joint_to is not None and self.leaf_fp.analytic
        qs = []
        for j, p in enumerate(poses):
            q = q_prev.clone()
            q[self.idx] = interpolate_arm(q_prev[self.idx], q_new[self.idx], j / n)
            qs.append(self.kit.q_at_base_pose(q, *p))
            if interp:
                q_leaf = float(joint_from) + (float(joint_to) - float(joint_from)) * (j / n)
                if not leaf_clear_of_base(self.leaf_fp.at(q_leaf), p[:2], float(self.k["base_radius_m"])):
                    return False, f"leaf (q {q_leaf:.3f}) in the footprint at stride {j}/{len(poses)}"
                continue
            for pts, tag in ((leaf_prev, "leaf_prev"), (leaf_now, "leaf")):
                if not leaf_clear_of_base(pts, p[:2], float(self.k["base_radius_m"])):
                    return False, f"{tag} in the footprint at stride {j}/{len(poses)}"
            if not self.kit.trav_free(p[:2]):
                return False, f"off the floor at stride {j}/{len(poses)}"
        with self.ikc.ignoring(self.ignore):
            free = self.kit.collision_free(th.stack(qs), margin=P0.COLLISION_MARGIN)
        if not bool(free.all()):
            return False, f"whole body collides at stride {int(th.nonzero(~free)[0])}/{len(poses)}"
        return True, ""

    # -- the search ----------------------------------------------------------------------------------------------
    def run(self, q_values: Sequence[float], marks: Dict[str, float]) -> HybridPath:
        """Follow h(q) through ``q_values`` (dense, from the sampler's current joint state q_init); ``marks`` = the §5.3
        fraction keys -> q_k the ``base_follow`` record needs. Raises ``Reject("hybrid_follow_infeasible")`` (the
        reached fraction in the message and in ``s.template_meta["hybrid_follow"]``) or ``Reject("time_budget")``."""
        s = self.s
        b_prev: Pose2D = tuple(float(v) for v in self.manip.base_pose2d)  # type: ignore[assignment]
        q_prev = self.kit.q_at_base_pose(self.q_star, *b_prev)
        s.pose_joint(self.q_init)
        leaf_prev = self.footprint_at(self.q_init)
        margin0, _ = self.ikc.joint_limit_margin(q_prev, self.arm)
        poses = [FollowPose(0.0, self.q_init, b_prev, [float(v) for v in q_prev[self.idx].tolist()], q_prev.clone(), self.h0.eef,
                            self.ikc.fk_residual(q_prev, self.h0.eef, self.arm), margin0, chosen="stance")]
        ok0, why0 = self.footprint_ok(b_prev, leaf_prev)
        if not ok0:
            log.info("%s: hybrid follow: the manipulation stance itself fails the planar %s stand-in (kept: the stance passed the whole-body check)", s.gp.variant, why0)
        try:
            for i_q, q_i in enumerate(q_values):
                if time.time() - self.t0 > float(self.k["budget_s"]):
                    self._fail(poses, "time_budget", f"hybrid follow search over {self.k['budget_s']:.0f} s at q = {q_i:.3f}")
                s.pose_joint(q_i)
                w = s.h([q_i])[0]
                leaf_now = self.footprint_at(q_i)
                k_i = fraction_of(q_i, self.q_init, self.q_goal)
                b_rigid = self.rigid_pose(w.eef)
                n_before = self.n_ik
                sol = None
                chosen = "arm_only"
                b_new: Pose2D = b_prev
                last = i_q == len(q_values) - 1 and bool(self.k["check_retreat"])  # the retreat must leave from the final pose
                # 1. the arm alone from the current base (the leaf at q_i must stay clear of the footprint)
                tried: Dict[str, int] = {}
                if leaf_clear_of_base(leaf_now, b_prev[:2], float(self.k["base_radius_m"])) and self.reach_ok(b_prev, w.eef[0]):
                    sol = self.solve(b_prev, w.eef, q_prev, leaf_now, retreat=last)
                    if sol is None:
                        tried["arm_only_" + (getattr(self, "last_solve_fail", None) or "ik")] = 1
                else:
                    tried["arm_only_skipped"] = 1
                # 2. move the base: the rigid-sweep interpolations, then the lattice, in order of displacement
                if sol is None:
                    cands = lattice_candidates(b_prev, b_rigid, float(self.k["lattice_step_m"]), float(self.k["lattice_radius_m"]),
                                               math.radians(float(self.k["yaw_step_deg"])), math.radians(float(self.k["yaw_extra_deg"])),
                                               float(self.k["yaw_weight_m_per_rad"]), alphas=[float(a) for a in self.k["alphas"]])
                    for cost, b, tag in cands:
                        if self.n_ik - n_before >= int(self.k["max_ik_per_step"]):
                            tried["ik_cap"] = tried.get("ik_cap", 0) + 1
                            break
                        ok, why = self.footprint_ok(b, leaf_now)
                        if not ok:
                            tried[why] = tried.get(why, 0) + 1
                            continue
                        if not self.reach_ok(b, w.eef[0]):
                            tried["reach"] = tried.get("reach", 0) + 1
                            continue
                        cand = self.solve(b, w.eef, q_prev, leaf_now, retreat=last)
                        if cand is None:
                            why_s = getattr(self, "last_solve_fail", None) or "ik"
                            tried[why_s] = tried.get(why_s, 0) + 1
                            continue
                        ok, why = self.path_ok(b_prev, b, q_prev, cand.q, leaf_prev, leaf_now, joint_from=float(poses[-1].q), joint_to=float(q_i))
                        if not ok:
                            # door 0919-104121: 20 IK-feasible candidates at 18 deg all failed here - which gate (leaf / body / floor)
                            kind = "path_leaf" if "footprint" in why else ("path_floor" if "floor" in why else "path_body")
                            tried[kind] = tried.get(kind, 0) + 1
                            n_dbg = getattr(self, "_n_path_debug", 0)
                            if n_dbg < 3:
                                self._n_path_debug = n_dbg + 1
                                log.info("%s: hybrid candidate %s (%s) rejected by the base path: %s", s.gp.variant, [round(v, 3) for v in b], tag, why)
                            continue
                        if str(self.k["base_move_policy"]) == "lookahead":
                            # score = how many of the following dense poses the arm reaches from this base without moving it
                            score = self.lookahead(b, cand.q, q_values[i_q + 1:], int(self.k["lookahead_steps"]), i_q + 1 == len(q_values))
                            scored = getattr(self, "_scored", None)
                            if scored is None or not scored or scored[-1][2] != i_q:
                                scored = []
                                self._scored = scored
                            scored.append((score, cost, i_q, cand, b, tag))
                            if len(scored) < int(self.k["lookahead_candidates"]) and self.n_ik - n_before < int(self.k["max_ik_per_step"]):
                                continue
                            best = max(scored, key=lambda t: (t[0], -t[1]))
                            sol, b_new, chosen = best[3], best[4], f"{best[5]}+lookahead{best[0]}"
                            tried["lookahead_scores"] = sorted((t[0] for t in scored), reverse=True)
                            self._scored = []
                            break
                        sol, b_new, chosen = cand, b, tag
                        break
                    if sol is None and str(self.k["base_move_policy"]) == "lookahead" and getattr(self, "_scored", None):
                        best = max(self._scored, key=lambda t: (t[0], -t[1]))  # fewer than lookahead_candidates were feasible
                        sol, b_new, chosen = best[3], best[4], f"{best[5]}+lookahead{best[0]}"
                        tried["lookahead_scores"] = sorted((t[0] for t in self._scored), reverse=True)
                        self._scored = []
                if sol is None:
                    self._fail(poses, "hybrid_follow_infeasible", f"no base pose within {self.k['lattice_radius_m']:.2f} m of {[round(v, 3) for v in b_prev]} reaches "
                                                                 f"h(q = {q_i:.3f}) (k = {k_i:.2f}); candidates rejected: {tried}; {self.n_ik - n_before} IK calls")
                d_m, d_yaw = pose2d_delta(b_prev, b_new)
                poses.append(FollowPose(k_i, float(q_i), b_new, [float(v) for v in sol.q[self.idx].tolist()], sol.q.clone(), w.eef, sol.residual, sol.margin,
                                        d_m, d_yaw, chosen, self.n_ik - n_before))
                self.log.append({"q": round(float(q_i), 4), "k": round(k_i, 3), "chosen": chosen, "base_step_m": round(d_m, 3), "base_step_rad": round(d_yaw, 3),
                                 "n_ik": self.n_ik - n_before, "rejected": tried, "b_rigid": [round(v, 3) for v in b_rigid]})
                log.info("%s: hybrid follow q = %.3f (k %.2f): %s, base step %.3f m / %.3f rad, %d IK calls%s", s.gp.variant, q_i, k_i, chosen, d_m, d_yaw,
                         self.n_ik - n_before, f", rejected {tried}" if tried else "")
                b_prev, q_prev, leaf_prev = b_new, sol.q, leaf_now
        finally:
            s.pose_joint(self.q_init)
        idle = "left" if self.arm == "right" else "right"
        idle_idx = self.kit.robot.arm_control_idx[idle]
        fingers = float(self.q_star[self.kit.robot.gripper_control_idx[self.arm]][0])
        st = path_stats(poses, self.q_init, self.q_goal)
        st.update({"time_s": round(time.time() - self.t0, 1), "r_max_m": round(self.r_max, 3), "ignored": "leaf links" if self.ignore and isinstance(self.ignore[0], _LinkSet) else "object",
                   "leaf_links": sorted(self.leaf), "n_body_boxes": len(self.boxes), "leaf_footprint": ("analytic" if self.leaf_fp.analytic else "aabb") + (" (aabb fallback used)" if self._fp_warned else "")})
        s.template_meta["hybrid_follow"] = {"ok": True, **st, "log": self.log}
        step = {"revolute_rad": math.radians(float(self.k["step_revolute_deg"]))} if s.joint.is_revolute else {"prismatic_m": float(self.k["step_prismatic_m"])}
        return HybridPath(self.arm, self.trunk, poses, base_follow_dict(poses, marks), idle, [float(v) for v in self.q_star[idle_idx].tolist()], fingers, step, st, dict(self.k),
                          retreat_arm_q=self.retreat_witness(poses))

    def lookahead(self, b: Sequence[float], q_at_b: th.Tensor, q_next: Sequence[float], max_steps: int, is_last: bool,
                  retreat_ladder: Optional[Sequence[float]] = None) -> int:
        """How many of the next dense joint values (in order, stopping at the first miss) the arm reaches from base ``b``
        without moving it: the leaf clear of the footprint, the reach pre-filter and the exact IK chained from ``q_at_b``
        (the retreat checked on the path's final pose). The joint is restored to the value it had on entry.
        ``retreat_ladder`` (the sequential policy): a pose counts only up to the last one whose release is feasible - a
        retreat pose of one of these distances along -a has an IK with the margin (the path's final pose: PRE_DIST) -
        since an arm segment can end only there (dishwasher_wide 0921-034952: a 1 cm move scored 3 by reach alone,
        then its segment was cut to one pose by the retreat)."""
        s = self.s
        q_entry = float(s.joint_value())
        count, count_releasable = 0, 0
        q_seed = q_at_b
        n_rest = len(list(q_next))
        hard = float(self.k.get("hard_margin_rad", self.ikc.JOINT_LIMIT_MARGIN)) + float(self.k["retreat_margin_slack_rad"])
        try:
            for j, q_j in enumerate(list(q_next)[:max_steps]):
                s.pose_joint(float(q_j))
                w = s.h([float(q_j)])[0]
                leaf_j = self.footprint_at(float(q_j))
                if not leaf_clear_of_base(leaf_j, b[:2], float(self.k["base_radius_m"])) or not self.reach_ok(b, w.eef[0]):
                    break
                # a cheap probe: the margin-aware IK alone (the chosen step is later gated by the full solve: polish,
                # collision, retreat, fov) - the full solve per look-ahead step blew the 600 s search budget (0920-160045)
                q_seed_b = self.kit.q_at_base_pose(q_seed, float(b[0]), float(b[1]), float(b[2]))
                with self.ikc.ignoring(self.ignore):
                    q_j_sol = self._ik(w.eef, q_seed_b)
                if q_j_sol is None:
                    break
                q_seed = q_j_sol
                count += 1
                if retreat_ladder is not None:
                    dists = [float(C.PRE_DIST)] if j == n_rest - 1 else [float(v) for v in retreat_ladder]
                    with self.ikc.ignoring(self.ignore):
                        if any(self._ik(C.pre_pose(w.eef, d_), q_j_sol, hard_margin=hard) is not None for d_ in dists):
                            count_releasable = count
        finally:
            s.pose_joint(q_entry)
        return count_releasable if retreat_ladder is not None else count

    def retreat_witness(self, poses: List[FollowPose]) -> Optional[List[float]]:
        """The retreat configuration (7 arm joints) at the path's final pose, solved as the search solves it (the free
        margin-aware IK) - recorded so the program check verifies THIS configuration instead of re-solving the retreat as a
        tracked chain (the two solvers disagreed by 8-17 mrad of margin: 0919-213438, 0921-022826)."""
        if not poses or poses[-1].q_full is None or poses[-1].eef is None:
            return None
        hard = float(self.k.get("hard_margin_rad", self.ikc.JOINT_LIMIT_MARGIN)) + float(self.k["retreat_margin_slack_rad"])
        try:
            self.s.pose_joint(float(poses[-1].q))
            with self.ikc.ignoring(self.ignore):
                q_ret = self._ik(C.pre_pose(poses[-1].eef, C.PRE_DIST), poses[-1].q_full, hard_margin=hard)
        finally:
            self.s.pose_joint(self.q_init)
        if q_ret is None:
            return None
        if not bool(self.kit.collision_free(q_ret, margin=P0.COLLISION_MARGIN)[0]):
            return None
        return [float(v) for v in q_ret[self.idx].tolist()]

    # -- sequential policy (owner 2026-09-20 20:19) -----------------------------------------------------------------
    def _arm_q(self, q_full: th.Tensor) -> List[float]:
        return [float(v) for v in q_full[self.idx].tolist()]

    @staticmethod
    def _eef_rec(eef) -> Dict[str, List[float]]:
        return {"position": [round(float(v), 6) for v in G.t(eef[0]).tolist()], "orientation": [round(float(v), 7) for v in G.t(eef[1]).tolist()]}

    def release_move_regrasp(self, b_prev: Pose2D, b_new: Pose2D, w: C.HandleWaypoint, q_prev: th.Tensor, leaf_pts,
                             retreat_m: float = C.PRE_DIST) -> Optional[Dict[str, Any]]:
        """Release at the current joint value (retreat pose IK from the grasp configuration, ``retreat_m`` along -a),
        drive ``b_prev -> b_new`` with the arm frozen at the retreat configuration (whole-body collision-free at every
        stride, else the stance's idle posture), re-grasp ``w`` from ``b_new`` (pre-grasp + grasp exact-IK chain with
        the margins). None when a step fails."""
        retreat_eef = C.pre_pose(w.eef, float(retreat_m))
        q_ret = self._ik(retreat_eef, q_prev)  # the full world: the hand leaves the handle, the leaf is an obstacle again
        if q_ret is None or not bool(self.kit.collision_free(q_ret, margin=P0.COLLISION_MARGIN)[0]):
            self.last_move_fail = "retreat_ik" if q_ret is None else "retreat_collision"
            return None
        strides = NAV.stride_poses([tuple(b_prev), tuple(b_new)], float(self.k["move_stride_m"]))
        if len(strides) < 2:
            strides = [tuple(b_prev), tuple(b_new)]
        for p in strides:
            if not leaf_clear_of_base(leaf_pts, p[:2], float(self.k["base_radius_m"])) or not self.footprint_ok(p, leaf_pts)[0]:
                self.last_move_fail = "path_footprint"
                return None
        hold_posture, q_hold = "retreat", q_ret
        qs = th.stack([self.kit.q_at_base_pose(q_hold, *p) for p in strides])
        if not bool(self.kit.collision_free(qs, margin=P0.COLLISION_MARGIN).all()):
            q_hold = self.ikc.q_locked(self.stance_at(b_prev, "idle"))  # the stance's recorded idle posture for every arm
            qs = th.stack([self.kit.q_at_base_pose(q_hold, *p) for p in strides])
            if not bool(self.kit.collision_free(qs, margin=P0.COLLISION_MARGIN).all()):
                self.last_move_fail = "path_collision"
                return None
            hold_posture = "idle"
        st_new = self.stance_at(b_new, "regrasp")
        res = self.ikc.exact([w.pre_grasp, w.eef], st_new, self.arm, ignore_for_last=[self.obj], cache=False)
        if not res.ok or res.q is None:
            self.last_move_fail = f"regrasp: {res.reason[:80]}"
            return None
        q_pre = self.ikc.solutions.get((self.arm, self.ikc._pose_key(w.pre_grasp)))
        clear = float(self.k.get("self_clearance_m", 0.0))
        if clear > 0.0:
            for tag_, q_ in (("regrasp", res.q), ("regrasp_pre", q_pre), ("hold", q_hold)):
                if q_ is not None and self.kit.backend.self_collision_pairs(q_, f"{self.arm}_arm", margin=clear):
                    self.last_move_fail = f"{tag_}_self_clearance"
                    return None
        self.last_move_fail = None
        return {"type": "release_move_regrasp", "retreat_eef": self._eef_rec(retreat_eef), "retreat_m": round(float(retreat_m), 3),
                "arm_q_retreat": [round(float(v), 6) for v in q_ret[self.idx].tolist()],
                "base_path": [[float(v) for v in p] for p in strides[1:]], "hold_posture": hold_posture, "arm_q_hold": [round(float(v), 6) for v in q_hold[self.idx].tolist()],
                "pre_grasp_eef": self._eef_rec(w.pre_grasp), "grasp_eef": self._eef_rec(w.eef),
                "arm_q_pre": None if q_pre is None else [round(float(v), 6) for v in q_pre[self.idx].tolist()],
                "arm_q_regrasp": [round(float(v), 6) for v in res.q[self.idx].tolist()], "q_end": res.q}

    def run_sequential(self, q_values: Sequence[float], marks: Dict[str, float]) -> HybridPath:
        """Owner policy 2026-09-20 20:19 / 23:58: arm segments to the arm's limit (+ margin), then one base move chosen by
        the look-ahead (fewest base moves) as release / move / re-grasp with the door stationary; typed segments for the
        executor, the flat ``poses`` list for the program check and the record."""
        s = self.s
        b: Pose2D = tuple(float(v) for v in self.manip.base_pose2d)  # type: ignore[assignment]
        q_prev = self.kit.q_at_base_pose(self.q_star, *b)
        s.pose_joint(self.q_init)
        margin0, _ = self.ikc.joint_limit_margin(q_prev, self.arm)
        poses = [FollowPose(0.0, self.q_init, b, self._arm_q(q_prev), q_prev.clone(), self.h0.eef, self.ikc.fk_residual(q_prev, self.h0.eef, self.arm), margin0, chosen="stance")]
        segments: List[Dict[str, Any]] = []
        n = len(q_values)
        i = 0
        q_cur, w_cur = float(self.q_init), self.h0
        visited: Dict[Tuple[float, float, float, float], int] = {}  # (q_cur, base) -> visits: a base move must not return to a pose already tried at this joint value
        try:
            while i < n:
                if time.time() - self.t0 > float(self.k["budget_s"]):
                    self._fail(poses, "time_budget", f"sequential follow search over {self.k['budget_s']:.0f} s at q = {q_values[i]:.3f}")
                # 1. arm segment: advance while the arm alone reaches h(q) with the margin; the segment then ends at the LAST pose
                # whose retreat pose (0.10 m along -a) is reachable too - a release, or the path's end, follows every segment
                # (dishwasher_wide 0921-011609: the arm pulled to 23.6 deg where no retreat pose had an IK - no move possible;
                # requiring the retreat at every pose instead shrank the segments to one pose each, 0921-020027)
                seg: List[FollowPose] = []
                seg_state: List[Tuple[th.Tensor, float, Any, int, Optional[float]]] = []  # (q_full, q, waypoint, i_next, retreat distance or None) per pose
                q_from, arm_from = q_cur, self._arm_q(q_prev)
                hard = float(self.k.get("hard_margin_rad", self.ikc.JOINT_LIMIT_MARGIN)) + float(self.k["retreat_margin_slack_rad"])
                ladder = [float(v) for v in (self.k.get("release_retreat_m") or [C.PRE_DIST])]
                i_seg0 = i
                seq_cap = self.k.get("seq_max_pull_rad")
                while i < n:
                    q_i = float(q_values[i])
                    if seq_cap is not None and seg and abs(q_i - float(q_from)) > float(seq_cap) + 1e-9:
                        break  # the pull is long enough: release / move / re-grasp next (seq_max_pull_rad)
                    s.pose_joint(q_i)
                    w = s.h([q_i])[0]
                    leaf_i = self.footprint_at(q_i)
                    n0 = self.n_ik
                    if not (leaf_clear_of_base(leaf_i, b[:2], float(self.k["base_radius_m"])) and self.reach_ok(b, w.eef[0])):
                        break
                    sol = self.solve(b, w.eef, q_prev, leaf_i, retreat=False)
                    if sol is None:
                        break
                    # the retreat that a release here would use: the program's PRE_DIST at the path's final pose, else the
                    # first distance of the ladder with an IK (the leaf ignored: the hand is still on / at the handle)
                    ret_m: Optional[float] = None
                    if not bool(self.k["check_retreat"]):
                        ret_m = float(C.PRE_DIST)
                    else:
                        with self.ikc.ignoring(self.ignore):
                            for dist in ([float(C.PRE_DIST)] if i == n - 1 else ladder):
                                if self._ik(C.pre_pose(w.eef, dist), sol.q, hard_margin=hard) is not None:
                                    ret_m = dist
                                    break
                    k_i = fraction_of(q_i, self.q_init, self.q_goal)
                    poses.append(FollowPose(k_i, q_i, b, self._arm_q(sol.q), sol.q.clone(), w.eef, sol.residual, sol.margin, 0.0, 0.0, "arm_only", self.n_ik - n0))
                    seg.append(poses[-1])
                    seg_state.append((sol.q, q_i, w, i + 1, ret_m))
                    q_prev, q_cur, w_cur = sol.q, q_i, w
                    i += 1
                retreat_m_cur: float = float(C.PRE_DIST)
                if seg:
                    last_ok = max((j for j, st_ in enumerate(seg_state) if st_[4] is not None), default=-1)
                    if last_ok >= 0:
                        retreat_m_cur = float(seg_state[last_ok][4])  # type: ignore[arg-type]
                    if last_ok < len(seg) - 1:
                        dropped = len(seg) - 1 - last_ok
                        del poses[len(poses) - dropped:]
                        seg = seg[: last_ok + 1]
                        if last_ok >= 0:
                            q_prev, q_cur, w_cur, i = seg_state[last_ok][0], seg_state[last_ok][1], seg_state[last_ok][2], seg_state[last_ok][3]
                        else:
                            q_prev_keep = poses[-1].q_full
                            q_prev, i = (q_prev_keep if q_prev_keep is not None else q_prev), i_seg0
                            q_cur = float(poses[-1].q)
                            s.pose_joint(q_cur)
                            w_cur = s.h([q_cur])[0]
                        log.info("%s: sequential: arm segment truncated by %d pose(s) to the last pose with a reachable retreat (q = %.3f, retreat %.2f m)", s.gp.variant, dropped, q_cur, retreat_m_cur)
                if seg:
                    segments.append({"type": "arm", "q_from": float(q_from), "q_to": float(q_cur), "base_pose2d": [float(v) for v in b], "arm_q_from": [round(v, 6) for v in arm_from],
                                     "arm_q_to": [round(v, 6) for v in self._arm_q(q_prev)], "poses": [p.to_sidecar(True) for p in seg]})
                    log.info("%s: sequential: arm segment %.3f -> %.3f (%d poses) from base %s, retreat %.2f m", s.gp.variant, q_from, q_cur, len(seg), [round(v, 3) for v in b], retreat_m_cur)
                elif not segments:
                    # the stance itself admits no arm pull that can be released (door 0921-022828 / 030920: the arm reached 10 deg
                    # but the retreat from there folded the elbow past its limit) - a grasp followed by an immediate release and a
                    # base move is not a demo; reject the stance so the sampler draws the next one (a farther stand-off)
                    self._fail(poses, "hybrid_follow_infeasible", f"sequential: the stance {[round(v, 3) for v in b]} admits no arm segment with a reachable retreat "
                                                                 f"(first pose q = {q_values[i_seg0]:.3f}: {'arm IK miss: ' + str(getattr(self, 'last_solve_fail', None)) if not seg_state else f'no retreat pose ({ladder} m along -a) has an IK with the margin'})")
                elif segments and segments[-1]["type"] != "arm":
                    self._fail(poses, "hybrid_follow_infeasible", f"sequential: no arm progress after the {segments[-1]['type']} at q = {q_cur:.3f} (the full solve of the next pose failed: "
                                                                 f"{getattr(self, 'last_solve_fail', None)})")
                if i >= n:
                    break
                # 2. base move with the door stationary at q_cur: candidates scored by the look-ahead from the next pose
                s.pose_joint(q_cur)
                leaf_cur = self.footprint_at(q_cur)
                q_next = float(q_values[i])
                s.pose_joint(q_next)
                w_next = s.h([q_next])[0]
                leaf_next = self.footprint_at(q_next)
                b_rigid = self.rigid_pose(w_next.eef)
                cands = lattice_candidates(b, b_rigid, float(self.k["lattice_step_m"]), float(self.k["lattice_radius_m"]), math.radians(float(self.k["yaw_step_deg"])),
                                           math.radians(float(self.k["yaw_extra_deg"])), float(self.k["yaw_weight_m_per_rad"]), alphas=[float(a) for a in self.k["alphas"]])
                n_before = self.n_ik
                scored: List[Tuple[int, float, Pose2D, str]] = []
                tried: Dict[str, int] = {}
                visited[(round(q_cur, 4), round(b[0], 3), round(b[1], 3), round(b[2], 3))] = visited.get((round(q_cur, 4), round(b[0], 3), round(b[1], 3), round(b[2], 3)), 0) + 1
                t_move, t_note, n_cand = time.time(), time.time(), 0
                for cost, b_c, tag in cands:
                    n_cand += 1
                    if time.time() - t_note > 60.0:  # door 0921-030920: 30 min of silence inside this loop - say where the time goes
                        t_note = time.time()
                        log.info("%s: sequential: base-move search at q = %.3f running %.0f s: %d / %d candidates, %d IK calls, %d scored, rejected %s", s.gp.variant, q_cur,
                                 time.time() - t_move, n_cand, len(cands), self.n_ik - n_before, len(scored), tried)
                    if self.n_ik - n_before >= int(self.k["max_ik_per_step"]):
                        tried["ik_cap"] = tried.get("ik_cap", 0) + 1
                        break
                    if pose2d_delta(b, b_c)[0] < float(self.k["min_move_m"]) and pose2d_delta(b, b_c)[1] < math.radians(float(self.k["min_move_deg"])):
                        tried["too_small"] = tried.get("too_small", 0) + 1  # a release / re-grasp cycle for a 1 cm move (0921-034952) buys nothing
                        continue
                    if visited.get((round(q_cur, 4), round(float(b_c[0]), 3), round(float(b_c[1]), 3), round(float(b_c[2]), 3))):
                        tried["visited"] = tried.get("visited", 0) + 1
                        continue
                    ok, why = self.footprint_ok(b_c, leaf_cur)
                    if not ok or not self.footprint_ok(b_c, leaf_next)[0]:
                        tried[why or "footprint"] = tried.get(why or "footprint", 0) + 1
                        continue
                    if not self.reach_ok(b_c, w_next.eef[0]):
                        tried["reach"] = tried.get("reach", 0) + 1
                        continue
                    score = self.lookahead(b_c, q_prev, q_values[i:], int(self.k["lookahead_steps"]), False, retreat_ladder=ladder if bool(self.k["check_retreat"]) else None)
                    if score <= 0:
                        tried["ik"] = tried.get("ik", 0) + 1
                        continue
                    # the cheap look-ahead probe is not the gate: the next pose must pass the full solve from this base (polish,
                    # collision, fov, the retreat at the path's end) - else the arm segment after the move makes no progress and
                    # the search ping-pongs between two bases (dishwasher_wide 0920-203527 at q 0.471)
                    s.pose_joint(q_next)
                    sol_next = self.solve(b_c, w_next.eef, self.kit.q_at_base_pose(q_prev, *b_c), leaf_next, retreat=bool(self.k["check_retreat"]))
                    s.pose_joint(q_cur)
                    if sol_next is None:
                        tried["ik_full_" + (getattr(self, "last_solve_fail", None) or "ik")] = tried.get("ik_full_" + (getattr(self, "last_solve_fail", None) or "ik"), 0) + 1
                        continue
                    scored.append((score, cost, tuple(float(v) for v in b_c), tag))
                    if len(scored) >= int(self.k["lookahead_candidates"]):
                        break
                s.pose_joint(q_cur)
                if not scored:
                    self._fail(poses, "hybrid_follow_infeasible", f"sequential: no base pose within {self.k['lattice_radius_m']:.2f} m of {[round(v, 3) for v in b]} reaches "
                                                                 f"h(q = {q_next:.3f}) with the door at {q_cur:.3f}; candidates rejected: {tried}; {self.n_ik - n_before} IK calls")
                move: Optional[Dict[str, Any]] = None
                b_new: Pose2D = b
                for score, cost, b_c, tag in sorted(scored, key=lambda t: (-t[0], t[1])):
                    # owner 2026-09-20 23:58: the only base move is release -> move -> re-grasp (the anchored move and its
                    # per-stride IK are gone: 25-125 IK calls per move and the 7.467 <-> 7.567 ping-pong, 0920-203527)
                    rec = self.release_move_regrasp(b, b_c, w_cur, q_prev, leaf_cur, retreat_m=retreat_m_cur)
                    if rec is None:
                        why_m = getattr(self, "last_move_fail", None) or "move"
                        tried["move_" + why_m.split(":")[0]] = tried.get("move_" + why_m.split(":")[0], 0) + 1
                        log.info("%s: sequential: release/move/re-grasp %s -> %s at q = %.3f rejected: %s", s.gp.variant, [round(v, 3) for v in b], [round(v, 3) for v in b_c], q_cur, why_m)
                    if rec is not None:
                        rec.update({"q": float(q_cur), "lookahead": int(score), "base_from": [float(v) for v in b], "base_to": [float(v) for v in b_c], "candidate": tag})
                        move, b_new = rec, b_c
                        break
                if move is None:
                    self._fail(poses, "hybrid_follow_infeasible", f"sequential: the door at {q_cur:.3f}: none of {len(scored)} scored base poses admits a release / move / re-grasp ({tried})")
                q_end = move.pop("q_end")
                d_m, d_yaw = pose2d_delta(b, b_new)
                margin_e, _ = self.ikc.joint_limit_margin(q_end, self.arm)
                poses.append(FollowPose(fraction_of(q_cur, self.q_init, self.q_goal), float(q_cur), b_new, self._arm_q(q_end), q_end.clone(), w_cur.eef,
                                        self.ikc.fk_residual(q_end, w_cur.eef, self.arm), margin_e, d_m, d_yaw, move["type"], self.n_ik - n_before))
                segments.append(move)
                self.log.append({"q": round(float(q_cur), 4), "chosen": move["type"], "base_step_m": round(d_m, 3), "base_step_rad": round(d_yaw, 3), "lookahead": int(move["lookahead"]),
                                 "n_ik": self.n_ik - n_before, "rejected": tried})
                log.info("%s: sequential: %s at q = %.3f, base %s -> %s (%.2f m / %.2f rad, look-ahead %d, %d IK calls, rejected %s)", s.gp.variant, move["type"], q_cur,
                         [round(v, 3) for v in b], [round(v, 3) for v in b_new], d_m, d_yaw, move["lookahead"], self.n_ik - n_before, tried)
                b, q_prev = b_new, q_end
        finally:
            s.pose_joint(self.q_init)
        idle = "left" if self.arm == "right" else "right"
        idle_idx = self.kit.robot.arm_control_idx[idle]
        fingers = float(self.q_star[self.kit.robot.gripper_control_idx[self.arm]][0])
        st = path_stats(poses, self.q_init, self.q_goal)
        st.update({"time_s": round(time.time() - self.t0, 1), "r_max_m": round(self.r_max, 3), "policy": "sequential", "n_segments": len(segments),
                   "n_regrasp_moves": sum(1 for g in segments if g["type"] == "release_move_regrasp"),
                   "leaf_links": sorted(self.leaf), "n_body_boxes": len(self.boxes)})
        s.template_meta["hybrid_follow"] = {"ok": True, **st, "log": self.log}
        step = {"revolute_rad": math.radians(float(self.k["step_revolute_deg"]))} if s.joint.is_revolute else {"prismatic_m": float(self.k["step_prismatic_m"])}
        return HybridPath(self.arm, self.trunk, poses, base_follow_dict(poses, marks), idle, [float(v) for v in self.q_star[idle_idx].tolist()], fingers, step, st, dict(self.k),
                          policy="sequential", segments=segments, retreat_arm_q=self.retreat_witness(poses))

    def _fail(self, poses: List[FollowPose], reason: str, msg: str):
        st = path_stats(poses, self.q_init, self.q_goal)
        st.update({"time_s": round(time.time() - self.t0, 1)})
        s = self.s
        meta = s.template_meta.setdefault("hybrid_follow", {})
        meta.update({"ok": False, "reason": reason, "message": msg, **st, "log": self.log})
        rf = s.template_meta.setdefault("hybrid_follow_fraction", {})
        key = f"d{self.manip.stand_off:.2f}_l{self.manip.lateral:+.2f}_t{self.manip.translate:.2f}_{self.manip.tilt:+.2f}" if self.manip.stand_off is not None else self.manip.name
        rf[key] = st["followed_fraction"]
        s.template_meta["hybrid_follow_fraction_max"] = max(float(v) for v in rf.values())
        raise Reject(reason, f"{msg}; followed {st['followed_fraction']:.2f} of the swing ({st['n_base_moves']} base moves, {st['base_path_m']:.2f} m)")


def search_follow_path(s, manip: Stance, q_star: th.Tensor, h0: C.HandleWaypoint, q_init: float, q_goal: float, fractions: Sequence[float],
                       required: Sequence[float], fraction_key, policy: Optional[str] = None) -> HybridPath:
    """The single entry point of the sampler (``ArticulationLarge._solve_from``): the hybrid path from q_init to the last
    required §5.3 waypoint. ``fraction_key`` = ``articulation_large.fraction_key`` (the sidecar key of a fraction);
    ``policy`` = the follow policy the caller resolved (``articulation_large.base_follow_policy``) over
    ``generation.hybrid_follow.policy``."""
    if not required:
        raise Reject("hybrid_follow_infeasible", "no required waypoint to follow")
    k_last = max(float(k) for k in required)
    q_last = float(q_init) + k_last * (float(q_goal) - float(q_init))
    marks = {fraction_key(k): float(q_init) + float(k) * (float(q_goal) - float(q_init)) for k in fractions if float(k) <= k_last + 1e-9}
    gen = dict(s.gp.gen("hybrid_follow") or {})
    if policy is not None:
        gen["policy"] = str(policy)
    fs = FollowSearch(s, manip, q_star, h0, q_init, q_goal, gen=gen)
    step = math.radians(float(fs.k["step_revolute_deg"])) if s.joint.is_revolute else float(fs.k["step_prismatic_m"])
    q_values = dense_values(q_init, q_last, list(marks.values()), step)
    log.info("%s: %s follow search %s from %.3f to %.3f (%d dense steps of <= %.3f, fractions %s), arm %s, leaf links %s, %d body boxes, reach <= %.2f m", s.gp.variant,
             fs.k["policy"], s.joint_name, q_init, q_last, len(q_values), step, sorted(marks), fs.arm, sorted(fs.leaf), len(fs.boxes), fs.r_max)
    if str(fs.k["policy"]) == "sequential":
        return fs.run_sequential(q_values, marks)
    return fs.run(q_values, marks)


# ----------------------------------------------------------------------------------------------------------------
# program check (spec §3.5): the recorded path as witnesses
# ----------------------------------------------------------------------------------------------------------------


def check_program_follow(sidecar, sampler, action, targets: Dict[str, Any], stance: Stance, arm: str, witnesses: Dict[str, Any]) -> int:
    """Hybrid ``base_follow``: every recorded pose of ``articulation.base_follow_path.poses`` is verified as a witness —
    the arm joints at the recorded base pose reach the recorded handle pose h(q_i) (FK residual <= ``WITNESS_RESIDUAL_MAX``
    after polishing), whole-body collision-free with the object's joint at q_i (leaf links removed, body present), the
    base on free floor, the joint-limit margin kept, the base path between consecutive poses collision-free; the
    fraction record ``base_follow[k]`` must be the path's pose at q_k. Witnesses ``handle_poses[i]`` per pose and
    ``follow_end`` (the configuration at the end of the follow: the tracker checks the retreat from there). Returns the
    number of poses checked."""
    art = getattr(action, "articulation", None)
    path = (getattr(art, "extra", None) or {}).get("base_follow_path") if art is not None else None
    recs = list((path or {}).get("poses") or [])
    if len(recs) < 2:
        raise Reject("program_check_failed", "hybrid base_follow: articulation.base_follow_path.poses needs >= 2 poses", action=getattr(action, "name", "base_follow"))
    obj = sampler.objects.get(getattr(art, "object", None))
    if obj is None:
        raise Reject("program_check_failed", f"hybrid base_follow: object {getattr(art, 'object', None)!r} not in the world", action="base_follow")
    joint_name = getattr(art, "joint", None)
    if joint_name not in getattr(obj, "joints", {}):
        raise Reject("program_check_failed", f"hybrid base_follow: joint {joint_name!r} not on {obj.name}", action="base_follow")
    jinfo = getattr(sampler, "jinfo", None) or {}
    leaf = leaf_links(obj, jinfo, obj.joints[joint_name])
    ignore = _ignore_set(sampler, obj, leaf)
    idx = sampler.kit.robot.arm_control_idx[arm]
    joint_idx = list(obj.joints.keys()).index(joint_name)
    q_saved = obj.get_joint_positions().clone()
    kn = knobs(sampler.gp.gen("hybrid_follow") if hasattr(sampler, "gp") else None)
    op, oq = obj.get_position_orientation()
    fp = LeafFootprint(obj, leaf, jinfo, (op.float(), oq.float()), float(q_saved[joint_idx]))  # the object stands at q_init (the draft)
    q_by_key = {key: float(art.q_at(float(key))) for key in (art.base_follow or {})}
    n = 0
    prev_q = None
    prev_b = None
    prev_leaf = None
    prev_joint = None
    try:
        for i, rec in enumerate(recs):
            b = tuple(float(v) for v in rec["base_pose2d"])
            q_i = float(rec["q"])
            q_all = q_saved.clone()
            q_all[joint_idx] = q_i
            obj.set_joint_positions(q_all)
            obj.keep_still()
            sampler.world.step_physics(1, still=[sampler.robot])
            obj.set_joint_positions(q_all)
            obj.keep_still()
            sampler.kit.sync_world()
            aabb = sampler.moving_link_footprint() if hasattr(sampler, "moving_link_footprint") else None
            leaf_pts = fp.at(q_i, fallback=aabb)
            eef = _as_pose(rec.get("eef")) if rec.get("eef") else _as_pose(sampler.h([q_i])[0].eef)
            st_b = dataclasses.replace(stance, name=f"follow_{i}", base_pose2d=b, q=None)
            arm_q = th.tensor([float(v) for v in rec["arm_q"]], dtype=th.float32)
            q = sampler.ikc.q_locked(st_b, right_arm=arm_q if arm == "right" else None, left_arm=arm_q if arm == "left" else None)
            if path.get("idle_arm_q") is not None and path.get("idle_arm") in ("left", "right"):
                q[sampler.kit.robot.arm_control_idx[path["idle_arm"]]] = th.tensor([float(v) for v in path["idle_arm_q"]], dtype=th.float32)
            # oracle-fix 2026-11-04 round 3 (articulation #03, bottom_cabinet_rvunhj_0): poses[0] IS the grasp at q_init, which step 5a checked
            # with the whole object out of the world (the open fingers 70 mm from the neighbouring door's bar); opt-in
            # ``generation.follow_first_pose_ignore_object`` checks it the same way instead of with the body present
            first_as_grasp = i == 0 and bool(sampler.gp.gen("follow_first_pose_ignore_object", False)) if hasattr(sampler, "gp") else False
            with sampler.ikc.ignoring(list(ignore) + ([obj] if first_as_grasp else [])):
                q_pol, residual = sampler.ikc.polish_witness(q, eef, arm, ignore=())
                free = bool(sampler.kit.collision_free(q_pol, margin=P0.COLLISION_MARGIN)[0])
            n += 1
            where = f"base_follow_path.poses[{i}] (q = {q_i:.3f}, base {[round(v, 3) for v in b]})"
            # a follow pose is accepted by the search at the solver's 5 mm (``solve``: a polish that would sit inside the
            # joint-limit margin is discarded and the raw IK residual stands); the executor ramps the recorded joints while
            # the grasp constraint carries a few mm of strain anyway (dishwasher 0917-213723: 3.4 mm) - the 2 mm end-pose
            # standard rejected pose 2 of the door path at 4.8 mm (0919-195417). ``generation.witness_residual_follow_m``.
            limit = float(sampler.gp.gen("witness_residual_follow_m", 5e-3))
            if residual > limit:
                raise Reject("program_check_failed", f"{where}: FK residual {residual * 1000:.1f} mm > {limit * 1000:.0f} mm", action="base_follow")
            if not free:
                raise Reject("program_check_failed", f"{where}: whole body collides (leaf removed, body present)", action="base_follow")
            if not sampler.kit.trav_free(b[:2]):
                raise Reject("program_check_failed", f"{where}: off the free floor", action="base_follow")
            if not leaf_clear_of_base(leaf_pts, b[:2], float(kn["base_radius_m"])):
                raise Reject("program_check_failed", f"{where}: the leaf sweeps into the base footprint", action="base_follow")
            margin, jn = sampler.ikc.joint_limit_margin(q_pol, arm)
            if margin < sampler.ikc.JOINT_LIMIT_MARGIN * 0.5:
                raise Reject("program_check_failed", f"{where}: {jn} within {margin * 1000:.0f} mrad of its limit", action="base_follow")
            stationary_move = prev_joint is not None and abs(float(prev_joint) - q_i) < 1e-9 and bool(path.get("segments"))
            if prev_q is not None and not stationary_move and (pose2d_delta(prev_b, b)[0] > 1e-6 or pose2d_delta(prev_b, b)[1] > 1e-6):
                poses2d = NAV.stride_poses([prev_b, b], NAV.STRIDE_M)
                m = max(len(poses2d) - 1, 1)
                qs = []
                for j, p in enumerate(poses2d):
                    qj = prev_q.clone()
                    qj[idx] = interpolate_arm(prev_q[idx], q_pol[idx], j / m)
                    qs.append(sampler.kit.q_at_base_pose(qj, *p))
                    # the base and the door move together along the segment (as the search's path_ok): the leaf at the
                    # interpolated joint value per stride when the footprint is analytic, else at both ends
                    if fp.analytic and prev_joint is not None:
                        q_leaf = float(prev_joint) + (float(q_i) - float(prev_joint)) * (j / m)
                        if not leaf_clear_of_base(fp.at(q_leaf), p[:2], float(kn["base_radius_m"])):
                            raise Reject("program_check_failed", f"{where}: the leaf (q {q_leaf:.3f}) is in the footprint along the base path (stride {j})", action="base_follow")
                        continue
                    for pts in (prev_leaf, leaf_pts):
                        if not leaf_clear_of_base(pts, p[:2], float(kn["base_radius_m"])):
                            raise Reject("program_check_failed", f"{where}: the leaf is in the footprint along the base path (stride {j})", action="base_follow")
                with sampler.ikc.ignoring(ignore):
                    ok = sampler.kit.collision_free(th.stack(qs), margin=P0.COLLISION_MARGIN)
                n += len(qs)
                if not bool(ok.all()):
                    raise Reject("program_check_failed", f"{where}: the base path from the previous pose collides at stride {int(th.nonzero(~ok)[0])}", action="base_follow")
            for key, qk in q_by_key.items():
                if abs(qk - q_i) < 1e-6:
                    bases_at_qk = [tuple(float(v) for v in r["base_pose2d"]) for r in recs if abs(float(r["q"]) - qk) < 1e-6]  # a stationary-door base move leaves two poses at q_k
                    if not any(all(abs(float(u) - float(v)) <= 1e-4 for u, v in zip(art.base_follow[key], bb)) for bb in bases_at_qk):
                        raise Reject("program_check_failed", f"articulation.base_follow[{key!r}] {art.base_follow[key]} differs from the path pose(s) at q_k {bases_at_qk}", action="base_follow")
            witnesses[f"handle_poses[{i}]"] = {"pose": {"position": [round(float(v), 5) for v in eef[0].tolist()], "orientation": [round(float(v), 6) for v in eef[1].tolist()]},
                                              "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, st_b).tolist()],
                                              "q_frame": "root at the world origin (identity pose, z = 0)", "base_pose2d": [float(v) for v in b],
                                              "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": jn, "joint_q": q_i}
            prev_q, prev_b, prev_leaf, prev_joint = q_pol, b, leaf_pts, float(q_i)
        witnesses["follow_end"] = {**witnesses[f"handle_poses[{len(recs) - 1}]"], "translate": float(stance.translate), "tilt": float(stance.tilt), "arm": arm}
        n += _check_segments(sidecar, sampler, art, path, stance, arm, witnesses, obj, joint_idx, q_saved, fp, ignore, kn)
    finally:
        obj.set_joint_positions(q_saved)
        obj.keep_still()
        sampler.world.step_physics(1, still=[sampler.robot])
        obj.set_joint_positions(q_saved)
        obj.keep_still()
        sampler.kit.sync_world()
    return n


def _check_segments(sidecar, sampler, art, path: Dict[str, Any], stance: Stance, arm: str, witnesses: Dict[str, Any], obj, joint_idx: int, q_saved: th.Tensor,
                    fp: "LeafFootprint", ignore, kn: Dict[str, Any]) -> int:
    """Sequential policy: the base moves happen with the door stationary and are verified per segment — an anchored move
    at every recorded stride (arm on the handle: polish <= 5 mm, collision-free with the leaf removed, footprint, margin),
    a release / move / re-grasp move at its retreat pose, along its path with the held arm (full world) and at the
    re-grasp chain (exact IK, margins). Witnesses ``regrasp_pre[j]`` / ``regrasp[j]`` for the executor's arm_to hints."""
    segs = list(path.get("segments") or [])
    if not segs:
        return 0
    limit = float(sampler.gp.gen("witness_residual_follow_m", 5e-3))
    n = 0

    def set_joint(q_i: float):
        q_all = q_saved.clone()
        q_all[joint_idx] = q_i
        obj.set_joint_positions(q_all)
        obj.keep_still()
        sampler.world.step_physics(1, still=[sampler.robot])
        obj.set_joint_positions(q_all)
        obj.keep_still()
        sampler.kit.sync_world()

    for j, seg in enumerate(segs):
        kind = seg.get("type")
        if kind == "arm":
            continue
        q_i = float(seg["q"])
        set_joint(q_i)
        aabb = sampler.moving_link_footprint() if hasattr(sampler, "moving_link_footprint") else None
        leaf_pts = fp.at(q_i, fallback=aabb)
        where = f"base_follow_path.segments[{j}] ({kind} at q = {q_i:.3f})"
        if kind == "release_move_regrasp":
            b0 = tuple(float(v) for v in seg["base_from"])
            b1 = tuple(float(v) for v in seg["base_to"])
            st0 = dataclasses.replace(stance, name=f"release_{j}", base_pose2d=b0, q=None)
            st1 = dataclasses.replace(stance, name=f"regrasp_{j}", base_pose2d=b1, q=None)
            aq_ret = th.tensor(seg["arm_q_retreat"], dtype=th.float32)
            q_ret = sampler.ikc.q_locked(st0, right_arm=aq_ret if arm == "right" else None, left_arm=aq_ret if arm == "left" else None)
            q_pol, residual = sampler.ikc.polish_witness(q_ret, _as_pose(seg["retreat_eef"]), arm, ignore=())
            n += 1
            if residual > limit or not bool(sampler.kit.collision_free(q_pol, margin=P0.COLLISION_MARGIN)[0]):
                raise Reject("program_check_failed", f"{where}: the retreat pose fails (residual {residual * 1000:.1f} mm)", action="base_follow")
            aq_hold = th.tensor(seg["arm_q_hold"], dtype=th.float32)
            q_hold = sampler.ikc.q_locked(st0, right_arm=aq_hold if arm == "right" else None, left_arm=aq_hold if arm == "left" else None)
            qs = th.stack([sampler.kit.q_at_base_pose(q_hold, *tuple(float(v) for v in b)) for b in seg["base_path"]])
            n += len(seg["base_path"])
            ok = sampler.kit.collision_free(qs, margin=P0.COLLISION_MARGIN)
            if not bool(ok.all()):
                raise Reject("program_check_failed", f"{where}: the base path with the held arm collides at stride {int(th.nonzero(~ok)[0])}", action="base_follow")
            for b in seg["base_path"]:
                b = tuple(float(v) for v in b)
                if not sampler.kit.trav_free(b[:2]) or not leaf_clear_of_base(leaf_pts, b[:2], float(kn["base_radius_m"])):
                    raise Reject("program_check_failed", f"{where}: base off the free floor or in the leaf's footprint along the path", action="base_follow")
            res = sampler.ikc.exact([_as_pose(seg["pre_grasp_eef"]), _as_pose(seg["grasp_eef"])], st1, arm, ignore_for_last=[obj], cache=False)
            n += 2
            if not res.ok or res.q is None:
                raise Reject("program_check_failed", f"{where}: the re-grasp chain fails the exact IK ({res.reason})", action="base_follow")
            q_pre = sampler.ikc.solutions.get((arm, sampler.ikc._pose_key(_as_pose(seg["pre_grasp_eef"]))))
            witnesses[f"regrasp[{j}]"] = {"q": [round(float(v), 5) for v in sampler.ikc.canonical_q(res.q, st1).tolist()], "base_pose2d": list(b1), "joint_q": q_i}
            if q_pre is not None:
                witnesses[f"regrasp_pre[{j}]"] = {"q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pre, st1).tolist()], "base_pose2d": list(b1), "joint_q": q_i}
        else:
            raise Reject("program_check_failed", f"{where}: unknown segment type", action="base_follow")
    return n


def end_stance_from_witness(w: Dict[str, Any], name: str = "follow_end") -> Optional[Stance]:
    """The in-memory stance the program check continues from after a hybrid follow (``follow_end`` witness): the base
    pose where the follow ends, the manipulation trunk, the working arm, the end configuration as ``q``."""
    if not w or "base_pose2d" not in w:
        return None
    q = th.tensor([float(v) for v in w["q"]], dtype=th.float32) if w.get("q") is not None else None
    return Stance(name, tuple(float(v) for v in w["base_pose2d"]), float(w.get("translate", 0.5)), float(w.get("tilt", 0.0)), arms=[w["arm"]] if w.get("arm") else "untucked", q=q)


def _as_pose(v) -> Tuple[th.Tensor, th.Tensor]:
    if isinstance(v, dict):
        return th.as_tensor(v["position"], dtype=th.float32), th.as_tensor(v["orientation"], dtype=th.float32)
    return G.t(v[0]), G.t(v[1])


__all__ = ["DEFAULTS", "knobs", "dense_values", "pose2d_delta", "displacement_cost", "pose2d_lerp", "yaw_candidates", "lattice_candidates", "disc_clear_of_boxes",
           "leaf_clear_of_base", "interpolate_arm", "fraction_of", "base_follow_dict", "FollowPose", "HybridPath", "path_stats", "leaf_links", "body_boxes",
           "FollowSearch", "search_follow_path", "check_program_follow", "end_stance_from_witness"]
