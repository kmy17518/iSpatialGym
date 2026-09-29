"""Prompt 0 — the large-scale stance skeleton (spec §5.2; caches §3.10; used by §5.3 / §5.5 / §5.6).

Every large-scale task (a scene articulated object, a target hung on a wall, a big fixed IORS partner) builds its
robot init and manipulation stances from the two phases here, in the **face frame** of §3.3 (``n`` outward unit normal
of the manipulation face, ``P`` reference point, ``d`` distance of the base centre from the face plane along n, ``λ``
lateral offset from P along the face tangent, + = robot's left when facing the face):

* **Phase 1 — init pose** (:func:`phase1_init_pose`): approach ray at bearing β ~ U(−β_max, β_max) about n through P,
  base yaw facing P, tilt 0; d scanned from d_min in 5 cm steps to d_max; at each d the translates {0, 0.25, 0.5, 0.75,
  0.95} are tested for (1) base on free floor (un-eroded map) inside the task rooms, (2) every target corner in frame
  (5 %), (3) whole-body collision-free (1 cm, untucked arms, open grippers), (4) the trunk sweep at tilt 0 free with the
  arms excluded. d* = the first d with a feasible t, t* = the feasible t whose optical axis is closest to the target
  centre (tie → smaller t). No feasible d → β redrawn (≤ 6 times) → ``Reject("no_stance")``.
* **Phase 2 — manipulation stance** (:func:`phase2_manipulation`): base square to the face (yaw = −n) on the lattice
  d ∈ {0.35, 0.40, …} × λ ∈ Λ (5 cm) × translate × grid tilt (the *band*; the tilt is drawn continuously inside the band
  at step 4). Step 1 base free / in room / collision-free with the target in its init state (single-arm: the idle arm
  untucked is part of the body; two-arm: arms excluded), step 2 every required contact (pose + pre-pose, per arm)
  feasible in the robot-only **reach volume** (§5.1), step 3 the focus in frame (5 %) and the optical axis within 8°
  of it (else 20°; the survivors of the best class are kept), step 4 the uniform draw among the survivors with a
  continuous tilt inside the band (:func:`draw_manipulation`), step 5 the exact IK of the whole contact set
  (:func:`exact_contacts`), step 6 the path init → manipulation (≤ 3 m, §3.7) and the transition (§3.8)
  (:func:`connect`).

**Caches** (§3.10): the step-1 survivors per (scene, target, init state, lattice, arm rule) and, when the caller asks
for it, the step-1–3 survivors per contact set — through ``sampling.cache`` (``ISPATIALGYM_SAMPLING_CACHE``); the
per-instance parts (the draw, step 5, step 6) are never cached.

**Arms excluded** (Phase 1 step 4, Phase 2 step 1 of two-arm tasks): the cuRobo backend has no link mask, so the
check runs the whole body with the arms hanging straight down (``arms="tucked"``) and self-collision off — the base,
torso and head against the scene with the arms folded into the body's own footprint (documented stand-in; a
link-masked ``check_collision`` is requested from the core).

Pure helpers (face frames, the ray, the scans, the base-frame lookup of the reach volume, the tilt band, the angle
classes) take no simulator and are unit-tested in ``tests/test_v2_large_scale.py``; everything that takes the sampler
``s`` needs the live world.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put
from ispatialgym.sampling.eef_reach_volume import CLASSES, ReachVolume, TILTS, TRANSLATES
from ispatialgym.sampling.v2.stances import FaceFrame, PathRecord, Pose, Stance, Transition

log = logging.getLogger("ispatialgym.sampling.v2")

#: §5.2 Phase 1 d scan step and Phase 2 lattice steps
D_STEP = 0.05
LAMBDA_STEP = 0.05
#: §5.2 Phase 2 default d scan {0.35, …, 0.90}; tasks extend it (articulation: 1.10, an open door protrudes)
D_RANGE_DEFAULT = (0.35, 0.90)
#: §3.4 in-frame margin (fraction of the image size)
IN_FRAME_MARGIN = 0.05
#: §5.2 step 3 optical-axis classes: within 8° (class 0), else within 20° (class 1)
AXIS_TIGHT_DEG = 8.0
AXIS_LOOSE_DEG = 20.0
#: §5.2 Phase 1: β redrawn at most 6 times
BETA_REDRAWS = 6
#: §5.2 step 6 / §3.7: path init → manipulation ≤ 3 m
PATH_MAX_M = 3.0
#: half width of a tilt band around its grid tilt (the grid is 0.3 apart)
TILT_BAND_HALF = 0.15
#: Phase 1 step 4 / §4.4: translates of the trunk sweep at tilt 0 (1.0 included: self-collision is off in that check)
SWEEP_TRANSLATES = (0.0, 0.25, 0.5, 0.75, 1.0)
#: cuRobo margin of the stance checks (§3.3: 1 cm for non-floor links)
COLLISION_MARGIN = 0.01

Pose2D = Tuple[float, float, float]


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (no simulator)
# ----------------------------------------------------------------------------------------------------------------


def _t(x) -> th.Tensor:
    return x if isinstance(x, th.Tensor) else th.tensor(x, dtype=th.float32)


def unit_xy(v) -> th.Tensor:
    """Unit horizontal (xy) vector of a 2-/3-vector."""
    v = _t(v)[:2].clone()
    n = float(th.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("face normal has no horizontal component")
    return v / n


def make_face(name: str, n, P, d: float = 0.0, lam: float = 0.0) -> FaceFrame:
    """Face frame (§3.3): ``n`` is made horizontal and unit (a face is vertical), ``P`` kept in 3-D."""
    nxy = unit_xy(n)
    return FaceFrame(name, th.tensor([float(nxy[0]), float(nxy[1]), 0.0]), _t(P).clone(), float(d), float(lam))


def face_yaw(face: FaceFrame) -> float:
    """Base yaw of a stance square to the face: the robot faces along −n."""
    return math.atan2(-float(face.n[1]), -float(face.n[0]))


def approach_ray_base(face: FaceFrame, beta: float, d: float) -> Pose2D:
    """§5.2 Phase 1a: ``b(d) = P_xy + R_z(β)·n_xy·d`` with the base yaw facing P."""
    c, s = math.cos(beta), math.sin(beta)
    nx, ny = float(face.n[0]), float(face.n[1])
    dx, dy = c * nx - s * ny, s * nx + c * ny
    bx, by = float(face.P[0]) + dx * d, float(face.P[1]) + dy * d
    yaw = math.atan2(float(face.P[1]) - by, float(face.P[0]) - bx)
    return bx, by, yaw


def d_scan(d_min: float, d_max: float, step: float = D_STEP) -> List[float]:
    """§5.2 Phase 1b: ``d_min, d_min + 5 cm, …, ≤ d_max``."""
    n = int(math.floor((d_max - d_min) / step + 1e-9))
    return [round(d_min + k * step, 6) for k in range(max(n, 0) + 1)]


def lattice_values(lo: float, hi: float, step: float) -> List[float]:
    """Lattice values ``lo, lo + step, …, ≤ hi`` (inclusive with a rounding tolerance)."""
    if hi < lo:
        return []
    n = int(math.floor((hi - lo) / step + 1e-9))
    return [round(lo + k * step, 6) for k in range(n + 1)]


def square_stance(face: FaceFrame, d: float, lam: float, translate: float, tilt: float, name: str = "manipulation") -> Stance:
    """Stance square to the face at (d, λ) with the face frame attached (``stand_off = d``, ``lateral = λ``)."""
    f = FaceFrame(face.name, face.n.clone(), face.P.clone(), float(d), float(lam))
    xy = f.base_xy()
    return Stance(name, (float(xy[0]), float(xy[1]), face_yaw(face)), float(translate), float(tilt), float(d), None, f, float(lam))


def ray_stance(face: FaceFrame, beta: float, d: float, translate: float, tilt: float = 0.0, name: str = "init") -> Stance:
    """Init stance on the approach ray (Phase 1): ``base_pose2d = approach_ray_base``; ``stand_off = d``; the face
    frame records d and λ = 0 (the ray passes through P); ``extras`` keep ``bearing_deg`` / ``distance``."""
    x, y, yaw = approach_ray_base(face, beta, d)
    f = FaceFrame(face.name, face.n.clone(), face.P.clone(), float(d), 0.0)
    st = Stance(name, (x, y, yaw), float(translate), float(tilt), float(d), None, f, 0.0)
    st.extras.update({"bearing_deg": math.degrees(beta), "distance": float(d), "face_normal": [float(v) for v in face.n[:2].tolist()]})
    return st


def to_base_frame(p_world, base_pose2d: Sequence[float], base_z: float) -> th.Tensor:
    """World point → base frame of a stance: ``R_z(−ψ)(p − b)`` with ``b = (x, y, base_z)`` (§5.1 lookup)."""
    p = _t(p_world)
    x, y, yaw = (float(v) for v in base_pose2d[:3])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = float(p[0]) - x, float(p[1]) - y, float(p[2]) - base_z
    return th.tensor([c * dx + s * dy, -s * dx + c * dy, dz])


def axes_in_base(quat_world, yaw: float) -> Tuple[th.Tensor, th.Tensor]:
    """(a, c) of a world eef orientation expressed in the base frame of yaw ``yaw`` (a = eef +z, c = eef +y)."""
    R = C.quat_to_mat(quat_world)
    a, c = R[:, 2], R[:, 1]
    cs, sn = math.cos(yaw), math.sin(yaw)
    Rz = th.tensor([[cs, sn, 0.0], [-sn, cs, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ a, Rz @ c


def reach_class(quat_world, yaw: float, allowed: Optional[Sequence[str]] = None) -> Optional[str]:
    """Reach-volume class of a world eef orientation at a base of yaw ``yaw`` (§5.1: pinch classes within 20° with the
    closing axis up to sign, push classes the nearest of the four); ``allowed`` restricts the candidates (a knob may
    use either pinch class). ``None`` = no class → infeasible for screening."""
    a, c = axes_in_base(quat_world, yaw)
    cls = ReachVolume.classify(a, c)
    if cls is None:
        return None
    if allowed is not None and cls not in allowed:
        # the nearest allowed class when the axes fit it (a knob: v/h are both admissible)
        for alt in allowed:
            spec = CLASSES[alt]
            ang_a = math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(a, th.tensor(spec["a"])))))))
            ang_c = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(th.dot(c, th.tensor(spec["c"]))))))))
            tol_a = 20.0 if spec["group"] == "pinch" else 45.0
            if ang_a <= tol_a and ang_c <= 20.0:
                return alt
        return None
    return cls


def tilt_min(translate: float) -> float:
    """Most negative tilt (forward / down) the ``torso_joint3`` limit admits on top of the keyframe curve at ``translate``."""
    j = 2
    return float(C.TORSO_JOINT_LIMITS[0][j]) - float(C.translate_to_torso_q(translate)[j])


def tilt_band(tilt_grid: float, translate: float, half: float = TILT_BAND_HALF) -> Tuple[float, float]:
    """Continuous tilt band of a grid tilt (§5.2 step 4): ``[k − 0.15, k + 0.15]`` clipped to ``[tilt_min, 0]``."""
    lo = max(tilt_min(translate), float(tilt_grid) - half)
    hi = min(0.0, float(tilt_grid) + half)
    return (lo, hi) if lo <= hi else (float(tilt_grid), float(tilt_grid))


def angle_class(angle_rad: float, tight_deg: float = AXIS_TIGHT_DEG, loose_deg: float = AXIS_LOOSE_DEG) -> Optional[int]:
    """0 within ``tight_deg`` of the focus, 1 within ``loose_deg``, None beyond (§5.2 step 3 / §4.5)."""
    a = math.degrees(float(angle_rad))
    if a <= tight_deg:
        return 0
    if a <= loose_deg:
        return 1
    return None


def best_class(items: Sequence[Tuple[Any, int]]) -> List[Any]:
    """The items of the best (smallest) angle class (§5.2 step 4: never always the smallest angle)."""
    if not items:
        return []
    k = min(c for _, c in items)
    return [x for x, c in items if c == k]


def phase1_choice(feasible: Sequence[Tuple[float, float]]) -> Optional[float]:
    """t* among feasible ``(translate, axis angle)`` pairs: the smallest angle, ties → the smaller translate."""
    if not feasible:
        return None
    return min(feasible, key=lambda ta: (round(ta[1], 6), ta[0]))[0]


# ----------------------------------------------------------------------------------------------------------------
# contact specs
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ContactSpec:
    """One required contact of a task's Phase-2 contact set (§5.2 step 2 / step 5).

    ``pose`` = the final world eef pose (grasp / contact), ``pre`` its pre-pose (default: ``pre_pose(pose)``),
    ``waypoints`` = further poses between ``pre`` and ``pose`` … or after it (handle poses at q_k, arc waypoints) that
    step 5 screens (§3.9) — they are listed in execution order *after* ``pose`` when ``after=True``; ``classes`` = the
    admissible reach-volume classes (``None`` = classify from the pose), ``ignore_for_last`` = objects removed from
    the collision world for the final pose only, ``ignore_waypoints`` = removed for every waypoint pose, ``screen_only``
    marks contacts the volume screens but step 5 does not solve (e.g. optional brace candidates handled by the task).
    """

    name: str
    arm: str
    pose: Pose
    pre: Optional[Pose] = None
    waypoints: List[Pose] = field(default_factory=list)
    after: bool = True
    classes: Optional[Tuple[str, ...]] = None
    ignore_for_last: List[Any] = field(default_factory=list)
    ignore_waypoints: List[Any] = field(default_factory=list)
    screen_only: bool = False

    def pre_pose(self) -> Pose:
        return self.pre if self.pre is not None else C.pre_pose(self.pose)

    def screen_poses(self) -> List[Pose]:
        """The poses the reach volume must admit: the final pose (the volume already encodes its pre-pose) and every
        waypoint (each one with its own pre-pose inside the volume's definition)."""
        return [self.pose] + list(self.waypoints)

    def volume_ok(self, rv: ReachVolume, stance: Stance, base_z: float) -> bool:
        """§5.2 step 2: every screen pose of this contact feasible in the reach volume at the stance's base and trunk."""
        yaw = stance.yaw
        for p in self.screen_poses():
            cls = reach_class(p[1], yaw, self.classes)
            if cls is None:
                return False
            pb = to_base_frame(p[0], stance.base_pose2d, base_z)
            if not rv.reachable(self.arm, stance.translate, stance.tilt, cls, pb):
                return False
        return True

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "arm": self.arm, "classes": list(self.classes) if self.classes else None,
                "pose": [[round(float(v), 4) for v in self.pose[0].tolist()], [round(float(v), 4) for v in self.pose[1].tolist()]],
                "n_waypoints": len(self.waypoints)}


def contact_set_key(contacts: Sequence[ContactSpec]) -> List[Any]:
    """Cache-key material of a contact set (arms, classes, poses rounded to the lattice)."""
    out = []
    for c in contacts:
        out.append([c.name, c.arm, list(c.classes) if c.classes else None,
                    [[round(float(v), 3) for v in p[0].tolist()] + [round(float(v), 3) for v in p[1].tolist()] for p in c.screen_poses()]])
    return out


# ----------------------------------------------------------------------------------------------------------------
# simulator-side helpers (take the sampler ``s``)
# ----------------------------------------------------------------------------------------------------------------

_RV: Optional[ReachVolume] = None


def reach_volume() -> ReachVolume:
    """The robot-only reach volume (§5.1), loaded once per process (``ISPATIALGYM_REACH_VOLUME`` / the committed run)."""
    global _RV
    if _RV is None:
        _RV = ReachVolume.load()
    return _RV


def base_z(s) -> float:
    """World height of ``base_link`` when the robot stands on the task floor (the reach-volume lattice is base-relative)."""
    return float(s.kit.floor_z) + float(C.NOMINAL_BASE_Z)


def in_task_rooms(s, xy) -> bool:
    """Whether a world xy lies in one of the task rooms (§5.2 Phase 1 (1) "inside the task room"). With
    ``generation.stance_rooms_include_adjacent`` (default off — an owner decision, see the Rs_int drawer finding: the
    seg-map boundary between the kitchen and the living room runs along the counter fronts, so the floor in front of a
    kitchen unit belongs to the living room) the loaded adjacent rooms (``world.eval_rooms``) count as well."""
    rooms = getattr(s.world, "rooms", None)
    if not rooms:
        return True
    seg = getattr(s.world.scene, "seg_map", None)
    if seg is None:
        return True
    allowed = set(rooms)
    try:
        if bool(s.gp.gen("stance_rooms_include_adjacent", False)):
            allowed |= set(getattr(s.world, "eval_rooms", None) or [])
    except Exception:  # noqa: BLE001 - samplers without a customs entry
        pass
    try:
        return seg.get_room_instance_by_point(_t(xy)[:2]) in allowed
    except Exception:  # noqa: BLE001 - outside the segmentation map
        return False


def base_ok(s, xy) -> bool:
    """Phase 1 (1) / Phase 2 step 1: base centre on free floor (un-eroded map) and inside the task rooms."""
    return bool(s.kit.trav_free(xy)) and in_task_rooms(s, xy)


def body_free(s, stances: Sequence[Stance], arms_excluded: bool = False, margin: float = COLLISION_MARGIN, chunk: int = 256,
              posture: Optional[str] = None, fingers: Optional[float] = None) -> th.Tensor:
    """Whole-body collision check of many stances (1 cm). ``arms_excluded`` → the documented stand-in (arms hanging,
    self-collision off): base + torso + head against the scene. ``posture`` (A6, §3.6) → the honest check of a named
    recorded arm posture (``untucked`` / ``tucked`` / ``ready``, self-collision on, ``fingers`` opening): the posture the
    stance will record in ``q`` and drive in with; it overrides ``arms_excluded``."""
    if not stances:
        return th.zeros(0, dtype=th.bool)
    out = []
    for i in range(0, len(stances), chunk):
        part = stances[i:i + chunk]
        if posture is not None:
            qs = th.stack([s.ikc.q_locked(st, arms=posture, fingers=fingers) for st in part])
            out.append(s.kit.collision_free(qs, margin=margin))
        elif arms_excluded:
            qs = th.stack([s.ikc.q_locked(st, arms="tucked") for st in part])
            s.kit._ensure_synced()
            out.append(~s.kit.backend.check_collision(qs, emb_name="whole_body", sync_world=False, self_collision=False, margin=margin))
        else:
            out.append(s.ikc.stances_collision_free(part, margin=margin))
    return th.cat(out)


def debug_collisions(s, stances: Sequence[Stance], arms_excluded: bool, margin: float = COLLISION_MARGIN) -> None:
    """Diagnostic log when a whole lattice collides: per stance the verdict of every arm posture and the penetrating
    spheres (base-frame position, radius) of the checked posture."""
    try:
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState
    except Exception:  # noqa: BLE001
        return
    be = s.kit.backend
    entry = be.kin_entry("whole_body")
    dev = be.device_cfg.device
    for st in stances:
        verdicts = {}
        for arms, selfc in (("untucked", True), ("tucked", True), ("tucked", False)):
            q = s.ikc.q_locked(st, arms=arms)
            verdicts[f"{arms}/self={selfc}"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=selfc, margin=margin)[0])
        q = s.ikc.q_locked(st, arms="tucked")
        verdicts["tucked/margin=0"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=0.0)[0])
        try:
            be._ground_obstacles_enabled(False)
            verdicts["tucked/no_ground"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)[0])
        finally:
            be._ground_obstacles_enabled(True)
        q = s.ikc.q_locked(st, arms="tucked" if arms_excluded else "untucked")
        js = be._active_js(entry, q.unsqueeze(0))
        state = entry.fk.compute_kinematics(js)
        spheres = state.robot_spheres.reshape(1, 1, -1, 4).contiguous()
        buf = CollisionBuffer.from_shape(spheres.shape, be.device_cfg)
        w = th.ones(1, device=dev)
        d0 = be.checker.get_sphere_collision(KinematicsState(robot_spheres=spheres), buf, w, th.full((1,), float(margin), device=dev)).reshape(-1)
        sp = spheres.reshape(-1, 4).cpu()
        bad = (d0 > 0).nonzero().flatten().tolist()
        rp = be.root_pose()[0]
        desc = [([round(float(v), 2) for v in (sp[i, :3] + rp.cpu()[:3]).tolist()], round(float(sp[i, 3]), 3)) for i in bad[:6]]
        log.info("collision debug %s d=%.2f λ=%.2f t=%.2f k=%.2f: %s; %d penetrating spheres (world pos, r) %s", st.name, st.stand_off or 0.0, st.lateral, st.translate,
                 st.tilt, verdicts, len(bad), desc)


def debug_collision_q(s, q: th.Tensor, label: str, margin: float = COLLISION_MARGIN, ignore: Sequence[Any] = ()) -> Dict[str, Any]:
    """Diagnostic of one 28-D configuration: verdicts with / without the ground obstacles and at margin 0, the penetrating
    spheres (world position, radius, z) and the scene objects whose AABB (inflated 5 cm) contains them."""
    out: Dict[str, Any] = {}
    try:
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState
    except Exception:  # noqa: BLE001
        return out
    be = s.kit.backend
    entry = be.kin_entry("whole_body")
    dev = be.device_cfg.device
    with s.ikc.ignoring(list(ignore)) if ignore else contextlib.nullcontext():
        s.kit._ensure_synced()
        out["collides"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=True, margin=margin)[0])
        out["no_self"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)[0])
        out["margin0"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=0.0)[0])
        try:
            be._ground_obstacles_enabled(False)
            out["no_ground"] = bool(be.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=margin)[0])
        finally:
            be._ground_obstacles_enabled(True)
        js = be._active_js(entry, q.unsqueeze(0))
        state = entry.fk.compute_kinematics(js)
        spheres = state.robot_spheres.reshape(1, 1, -1, 4).contiguous()
        buf = CollisionBuffer.from_shape(spheres.shape, be.device_cfg)
        d0 = be.checker.get_sphere_collision(KinematicsState(robot_spheres=spheres), buf, th.ones(1, device=dev), th.full((1,), float(margin), device=dev)).reshape(-1)
    sp = spheres.reshape(-1, 4).cpu()
    bad = (d0 > 0).nonzero().flatten().tolist()
    rp = be.root_pose()[0].cpu()
    pts = [(sp[i, :3] + rp[:3]) for i in bad]
    out["n_penetrating"] = len(bad)
    out["spheres"] = [([round(float(v), 2) for v in p.tolist()], round(float(sp[i, 3]), 3)) for p, i in zip(pts[:6], bad[:6])]
    hits: Dict[str, int] = {}
    for obj in getattr(s.world.scene, "objects", []):
        try:
            lo, hi = obj.aabb
        except Exception:  # noqa: BLE001
            continue
        lo, hi = th.as_tensor(lo) - 0.05, th.as_tensor(hi) + 0.05
        n = sum(1 for p in pts if bool(((p >= lo) & (p <= hi)).all()))
        if n:
            hits[obj.name] = n
    out["objects_at_spheres"] = dict(sorted(hits.items(), key=lambda kv: -kv[1])[:5])
    log.info("collision debug %s: %s", label, out)
    return out


def trunk_sweep_free(s, base_pose2d: Pose2D, translates: Sequence[float] = SWEEP_TRANSLATES, margin: float = COLLISION_MARGIN) -> bool:
    """Phase 1 (4) / §4.4 (3): base + torso links + head collision-free at every translate of the sweep, tilt 0, arms
    excluded (stand-in: arms hanging, self-collision off)."""
    stances = [Stance("sweep", tuple(base_pose2d), float(t), 0.0) for t in translates]
    return bool(body_free(s, stances, arms_excluded=True, margin=margin).all())


def camera_poses(s, stances: Sequence[Stance]) -> List[Pose]:
    """Head-camera poses of many stances (FK batch; arms untucked — the camera does not depend on them)."""
    if not stances:
        return []
    qs = th.stack([s.ikc.q_locked(st) for st in stances])
    return s.kit.camera_poses_batch(qs)


def in_frame(s, cam: Pose, points: th.Tensor, margin: float = IN_FRAME_MARGIN) -> bool:
    return bool(s.world.camera_model.all_inside(cam, _t(points).reshape(-1, 3), margin))


def axis_angle(s, cam: Pose, point) -> float:
    return float(s.world.camera_model.angle_off_axis(cam, _t(point)))


# ----------------------------------------------------------------------------------------------------------------
# Phase 1 — init pose (§5.2)
# ----------------------------------------------------------------------------------------------------------------


def phase1_scan(s, face: FaceFrame, beta: float, d_min: float, d_max: float, target_extent_fn: Callable[[], Tuple[th.Tensor, th.Tensor]],
                translates: Sequence[float] = TRANSLATES, in_frame_margin: float = IN_FRAME_MARGIN, stats: Optional[Dict[str, int]] = None,
                name: str = "init", tilts: Sequence[float] = (0.0,)) -> Optional[Stance]:
    """One bearing β: the d scan of §5.2 Phase 1b. Returns the init stance at (d*, t*) or None when no d is feasible.
    ``target_extent_fn() -> (corners (N, 3) world, centre (3,))`` of the target in its init state. ``tilts``: trunk
    tilts tried per translate, upright first (``generation.prompt0.init_tilts``; a lean forward brings a low, near
    target into the frame — the open dishwasher door of the close task fits the frame only from 1.15 m upright, where
    the base backs into the kitchen bar, 0919-232646)."""
    corners, centre = target_extent_fn()
    corners = _t(corners).reshape(-1, 3)
    st_ = stats if stats is not None else {}
    for d in d_scan(d_min, d_max):
        x, y, yaw = approach_ray_base(face, beta, d)
        st_["d_tested"] = st_.get("d_tested", 0) + 1
        if not base_ok(s, (x, y)):
            # which half of base_ok fails (Rs_int drawer 17:44: 287/287 (d, t) failed base_ok in 3 s — floor map or room?)
            k = "off_trav_map" if not bool(s.kit.trav_free((x, y))) else "outside_rooms"
            st_[k] = st_.get(k, 0) + 1
            lo_k, hi_k = st_.get(k + "_d", (d, d))
            st_[k + "_d"] = (round(min(lo_k, d), 2), round(max(hi_k, d), 2))  # the d range of that failure
            if k == "outside_rooms" and "room_seen" not in st_:
                try:
                    st_["room_seen"] = str(s.world.scene.seg_map.get_room_instance_by_point(th.tensor([x, y])))
                except Exception:  # noqa: BLE001
                    st_["room_seen"] = "?"
            continue
        st_["base_ok"] = st_.get("base_ok", 0) + 1
        cands = [ray_stance(face, beta, d, t, float(tilt), name) for t in translates for tilt in tilts]
        cams = camera_poses(s, cands)
        seen = [(st, cam) for st, cam in zip(cands, cams) if in_frame(s, cam, corners, in_frame_margin)]
        if not seen:
            continue
        st_["in_frame"] = st_.get("in_frame", 0) + len(seen)
        free = body_free(s, [st for st, _ in seen])
        if not bool(free.any()) and "collision_debug" not in st_:
            # one diagnostic per scan: what the body hits at the first in-frame candidate (door_crtrpa_0 0919-013815: 48 in-frame
            # stances in the 1.65 m garden strip, 0 collision-free, no culprit in the rejection)
            st0 = seen[0][0]
            try:
                dbg = debug_collision_q(s, s.ikc.q_locked(st0), f"phase1 {name} d={d:.2f}")
                st_["collision_debug"] = {"d": round(d, 2), "translate": st0.translate, "objects": dbg.get("objects_at_spheres"), "no_ground": dbg.get("no_ground"),
                                          "margin0": dbg.get("margin0"), "no_self": dbg.get("no_self"), "n_spheres": dbg.get("n_penetrating")}
            except Exception as e:  # noqa: BLE001 — diagnostics never fail the scan
                st_["collision_debug"] = {"error": repr(e)[:120]}
        seen = [(st, cam) for (st, cam), f in zip(seen, free.tolist()) if f]
        if not seen:
            continue
        st_["collision_free"] = st_.get("collision_free", 0) + len(seen)
        if not trunk_sweep_free(s, (x, y, yaw)):
            continue
        st_["sweep_free"] = st_.get("sweep_free", 0) + 1
        t_star = phase1_choice([(st.translate, axis_angle(s, cam, centre)) for st, cam in seen])
        for st, cam in seen:  # the first tilt of ``tilts`` (upright) at the chosen translate
            if abs(st.translate - t_star) < 1e-9:
                st.extras["axis_angle_deg"] = math.degrees(axis_angle(s, cam, centre))
                st.extras["target_centre"] = [round(float(v), 4) for v in _t(centre).tolist()]
                return st
    return None


def phase1_init_pose(s, face: FaceFrame, beta_max: float, d_min: float, d_max: float, target_extent_fn: Callable[[], Tuple[th.Tensor, th.Tensor]],
                     rng: np.random.Generator, translates: Sequence[float] = TRANSLATES, in_frame_margin: float = IN_FRAME_MARGIN,
                     redraws: int = BETA_REDRAWS, name: str = "init", tilts: Sequence[float] = (0.0,)) -> Stance:
    """§5.2 Phase 1: β ~ U(−β_max, β_max) (radians), the d scan, β redrawn at most ``redraws`` times;
    ``Reject("no_stance")`` with the per-stage counts when every bearing fails."""
    from ispatialgym.sampling.sim_utils import Reject

    stats: Dict[str, int] = {}
    betas = []
    for _ in range(redraws + 1):
        beta = float(rng.uniform(-beta_max, beta_max)) if beta_max > 0 else 0.0
        betas.append(round(math.degrees(beta), 1))
        st = phase1_scan(s, face, beta, d_min, d_max, target_extent_fn, translates, in_frame_margin, stats, name, tilts=tilts)
        if st is not None:
            st.arms = "untucked"
            return st
    raise Reject("no_stance", f"Phase 1: no (d, t) on {len(betas)} approach rays (bearings {betas}°, d {d_min}–{d_max} m) sees the whole target "
                              f"collision-free; stages {stats}", face=face.name)


# ----------------------------------------------------------------------------------------------------------------
# Phase 2 — manipulation stance (§5.2)
# ----------------------------------------------------------------------------------------------------------------


def phase2_lattice(face: FaceFrame, d_range: Tuple[float, float], lam_range: Tuple[float, float], translates: Sequence[float] = TRANSLATES,
                   tilts: Sequence[float] = TILTS, d_step: float = D_STEP, lam_step: float = LAMBDA_STEP) -> List[Stance]:
    """The Phase-2 candidate lattice d × λ × translate × grid tilt as stances square to the face (``extras["band"]`` =
    the continuous tilt band of the grid tilt)."""
    out = []
    for d in lattice_values(d_range[0], d_range[1], d_step):
        for lam in lattice_values(lam_range[0], lam_range[1], lam_step):
            for tr in translates:
                for ti in tilts:
                    st = square_stance(face, d, lam, tr, ti)
                    st.extras["band"] = list(tilt_band(ti, tr))
                    out.append(st)
    return out


def phase2_step1(s, face: FaceFrame, d_range: Tuple[float, float], lam_range: Tuple[float, float], arms_excluded: bool,
                 cache_scope: Optional[Dict[str, Any]] = None, translates: Sequence[float] = TRANSLATES, tilts: Sequence[float] = TILTS,
                 posture: Optional[str] = None, fingers: Optional[float] = None) -> List[Stance]:
    """§5.2 Phase 2 step 1 over the lattice: base on free floor and in room, whole-body collision-free (1 cm) against the
    scene with the target in its init state (the caller synced the world). Cached per ``cache_scope`` (scene, target,
    init state, lattice, arm rule) — the survivors do not depend on any per-instance draw. ``posture`` (A6): the named
    arm posture the stance records and drives in with (self-collision on) instead of the arms-excluded stand-in."""
    key = None
    if cache_scope is not None:
        key = cache_key("prompt0_step1", {**cache_scope, "face": face.to_json(), "d_range": list(d_range), "lam_range": list(lam_range),
                                          "translates": list(translates), "tilts": list(tilts), "arms_excluded": arms_excluded,
                                          "lattice": [D_STEP, LAMBDA_STEP], "margin": COLLISION_MARGIN, "posture": posture, "fingers": fingers,
                                          "working_arms": sorted(getattr(s.ikc, "default_working_arms", []) or [])})
        hit = cache_get(key)
        if hit is not None:
            out = []
            for e in hit["survivors"]:
                st = square_stance(face, e["d"], e["lam"], e["translate"], e["tilt"])
                st.extras["band"] = list(tilt_band(e["tilt"], e["translate"]))
                st.idle_posture = e.get("idle_posture")  # A6: the posture step 1 passed with (see phase2_manipulation)
                out.append(st)
            log.info("Phase 2 step 1: %d survivors from the cache", len(out))
            return out
    cands = phase2_lattice(face, d_range, lam_range, translates, tilts)
    base_free: Dict[Tuple[float, float], bool] = {}
    keep = []
    for st in cands:
        k = (st.stand_off, st.lateral)
        if k not in base_free:
            base_free[k] = base_ok(s, st.base_xy)
        if base_free[k]:
            keep.append(st)
    free = body_free(s, keep, arms_excluded=arms_excluded, posture=posture, fingers=fingers)
    out = [st for st, f in zip(keep, free.tolist()) if f]
    log.info("Phase 2 step 1: %d lattice candidates, %d on free floor, %d collision-free (arms %s)", len(cands), len(keep), len(out),
             posture or ("excluded" if arms_excluded else "untucked"))
    if not out and keep:
        flat = [st for st in keep if abs(st.tilt) < 1e-9 and abs(st.translate - 0.5) < 1e-9]
        debug_collisions(s, [flat[i] for i in range(0, len(flat), max(1, len(flat) // 6))][:6], arms_excluded)
    if key is not None and out:  # an empty set is not cached (it is diagnosed, and may come from a transient world state)
        cache_put(key, {"survivors": [{"d": st.stand_off, "lam": st.lateral, "translate": st.translate, "tilt": st.tilt, "idle_posture": st.idle_posture} for st in out]})
    return out


def phase2_screen(s, cands: Sequence[Stance], contact_set: Sequence[ContactSpec], focus, extra_visibility: Optional[Callable[[Pose], bool]] = None,
                  in_frame_margin: float = IN_FRAME_MARGIN, stats: Optional[Dict[str, int]] = None) -> List[Stance]:
    """§5.2 Phase 2 steps 2–3 on step-1 survivors: every required contact feasible in the reach volume at the base and
    trunk (step 2), then the focus in frame (5 %), ``extra_visibility(cam)`` (the task's extra points) and the optical
    axis within 8° (else 20°) of the focus (step 3). Returns the survivors of the **best angle class** with ``focus``,
    ``extras["axis_angle_deg"]`` and ``extras["angle_class"]`` set."""
    rv = reach_volume()
    bz = base_z(s)
    foc = _t(focus)
    st_ = stats if stats is not None else {}
    reach_ok = [st for st in cands if all(c.volume_ok(rv, st, bz) for c in contact_set)]
    st_["step2_reach"] = len(reach_ok)
    if not reach_ok:
        return []
    cams = camera_poses(s, reach_ok)
    classed: List[Tuple[Stance, int]] = []
    for st, cam in zip(reach_ok, cams):
        if not in_frame(s, cam, foc.reshape(1, 3), in_frame_margin):
            continue
        if extra_visibility is not None and not extra_visibility(cam):
            continue
        ang = axis_angle(s, cam, foc)
        k = angle_class(ang)
        if k is None:
            continue
        st.focus = foc.clone()
        st.extras["axis_angle_deg"] = math.degrees(ang)
        st.extras["angle_class"] = k
        classed.append((st, k))
    st_["step3_visible"] = len(classed)
    out = best_class(classed)
    st_["best_class"] = out[0].extras["angle_class"] if out else None
    return out


def phase2_manipulation(s, face: FaceFrame, contact_set: Sequence[ContactSpec], lam_range: Tuple[float, float], arms: Sequence[str],
                        d_range: Tuple[float, float] = D_RANGE_DEFAULT, extra_visibility: Optional[Callable[[Pose], bool]] = None,
                        focus=None, cache_scope: Optional[Dict[str, Any]] = None, cache_contacts: bool = False,
                        translates: Sequence[float] = TRANSLATES, tilts: Sequence[float] = TILTS, stats: Optional[Dict[str, int]] = None,
                        posture: Optional[str] = None, fingers: Optional[float] = None,
                        step1_memo: Optional[Dict[str, Any]] = None) -> List[Stance]:
    """§5.2 Phase 2 steps 1–3: the survivors of the best angle class (each a grid-trunk stance with its tilt band and
    focus). ``arms`` = the working arms of the task (two arms → arms excluded from the stance collision check unless
    ``posture`` names the recorded arm posture of the stance, A6); ``focus`` defaults to the first contact's pad point.
    ``cache_scope`` caches step 1; with ``cache_contacts`` the steps 1–3 survivors are cached under the contact set too
    (articulation: the contact set is fixed per (target, joint, q_init); alignment: per-instance δ → not cached).
    ``step1_memo``: a dict shared by calls whose step 1 is the same (same face, ranges, lattice, arms, posture and
    world, e.g. the bar offsets of one attempt): step 1 runs on the first call that needs it and the later calls screen
    copies of its survivors (step 3 annotates the stances it screens)."""
    arms_excluded = len(set(arms)) > 1
    foc = _t(focus) if focus is not None else C.pad_center(contact_set[0].pose)
    key = None
    if cache_scope is not None and cache_contacts:
        key = cache_key("prompt0_step123", {**cache_scope, "face": face.to_json(), "d_range": list(d_range), "lam_range": list(lam_range),
                                            "translates": list(translates), "tilts": list(tilts), "arms": sorted(set(arms)),
                                            "contacts": contact_set_key(contact_set), "focus": foc, "margin": IN_FRAME_MARGIN,
                                            "classes": [AXIS_TIGHT_DEG, AXIS_LOOSE_DEG], "posture": posture, "fingers": fingers,
                                            "working_arms": sorted(getattr(s.ikc, "default_working_arms", []) or [])})
        hit = cache_get(key)
        if hit is not None:
            out = []
            for e in hit["survivors"]:
                st = square_stance(face, e["d"], e["lam"], e["translate"], e["tilt"])
                st.extras.update({"band": list(tilt_band(e["tilt"], e["translate"])), "axis_angle_deg": e["axis_angle_deg"], "angle_class": e["angle_class"]})
                st.focus = foc.clone()
                # A6: the idle-arm posture step 1 passed with travels through the cache (a survivor that needed the tucked idle
                # arm came back untucked and failed every re-check draw by collision: door_crtrpa_0 0919-023637)
                st.idle_posture = e.get("idle_posture")
                st.arms = list(arms)
                out.append(st)
            log.info("Phase 2 steps 1–3: %d survivors from the cache", len(out))
            return out
    if step1_memo is not None and "survivors" in step1_memo:
        step1 = copy.deepcopy(step1_memo["survivors"])
        log.info("Phase 2 step 1: %d survivors of this attempt's step 1 (same lattice and world)", len(step1))
    else:
        step1 = phase2_step1(s, face, d_range, lam_range, arms_excluded, cache_scope, translates, tilts, posture=posture, fingers=fingers)
        if step1_memo is not None:
            step1_memo["survivors"] = copy.deepcopy(step1)
    st_ = stats if stats is not None else {}
    st_["step1"] = len(step1)
    out = phase2_screen(s, step1, contact_set, foc, extra_visibility, stats=st_)
    log.info("Phase 2: %d step-1 survivors, %d reach the contact set, %d visible, %d in the best class %s", len(step1), st_.get("step2_reach", 0),
             st_.get("step3_visible", 0), len(out), st_.get("best_class"))
    if key is not None and out:
        cache_put(key, {"survivors": [{"d": st.stand_off, "lam": st.lateral, "translate": st.translate, "tilt": st.tilt,
                                       "axis_angle_deg": st.extras["axis_angle_deg"], "angle_class": st.extras["angle_class"],
                                       "idle_posture": st.idle_posture} for st in out]})
    return out


CLEARANCE_MARGINS = (0.05, 0.03)


def clearance_tier_indices(s, survivors: Sequence[Stance], ignore: Sequence[Any], margins: Sequence[float] = CLEARANCE_MARGINS) -> List[int]:
    """Indices of the survivors in the best base-clearance tier: base + torso + head collision-free against the scene
    (``ignore`` — the target — removed) at the largest margin of ``margins`` that any survivor passes; all indices when
    none passes 3 cm. The pilot's executor tracks the recorded path / drive-in with more than the 1 cm cuRobo margin of
    error: the Rs_int drawer stance drawn 2 cm from coffee_table_fqluyq_0 (20:57) failed its first navigate with
    ``base_link in contact with coffee_table_fqluyq_0``."""
    idx = list(range(len(survivors)))
    if len(idx) <= 1:
        return idx
    ctx = s.ikc.ignoring(list(ignore)) if ignore else contextlib.nullcontext()
    with ctx:
        for m in margins:
            free = body_free(s, list(survivors), arms_excluded=True, margin=float(m))
            keep = [i for i, f in zip(idx, free.tolist()) if f]
            if keep:
                return keep
    return idx


def draw_manipulation(s, survivors: Sequence[Stance], rng: np.random.Generator, arms: Sequence[str], focus=None,
                      extra_visibility: Optional[Callable[[Pose], bool]] = None, arms_excluded: Optional[bool] = None,
                      in_frame_margin: float = IN_FRAME_MARGIN, name: str = "manipulation", tilt_draws: int = 3,
                      posture: Optional[str] = None, fingers: Optional[float] = None,
                      clearance_ignore: Optional[Sequence[Any]] = None, stats: Optional[Dict[str, int]] = None) -> Tuple[int, Optional[Stance]]:
    """§5.2 step 4: one survivor drawn uniformly (its index is returned so the caller can drop it) with the tilt drawn
    continuously inside its band; the drawn stance is re-checked (collision, focus in frame, extra visibility, optical
    axis within its class). The tilt is redrawn up to ``tilt_draws`` times inside the band (the grid tilt itself, which
    passed steps 1–3, is the last draw); when every draw fails the re-check ``(index, None)`` is returned and the
    caller takes the next survivor. ``posture`` (A6): the stance is checked in that named arm posture and records it
    (``q`` = the configuration the checks locked, ``fingers`` its gripper opening, ``extras["arm_posture"]``).
    ``clearance_ignore`` (the target): the draw is uniform among the survivors of the best base-clearance tier
    (:func:`clearance_tier_indices`; the tier is cached on the survivors' ``extras["clearance_tier"]``)."""
    if not survivors:
        return -1, None
    if clearance_ignore is not None:
        if any("clearance_tier" not in st.extras for st in survivors):
            keep = set(clearance_tier_indices(s, survivors, clearance_ignore))
            for j, st in enumerate(survivors):
                st.extras["clearance_tier"] = 0 if j in keep else 1
        top = [j for j, st in enumerate(survivors) if st.extras.get("clearance_tier") == 0] or list(range(len(survivors)))
        i = int(top[int(rng.integers(len(top)))])
    else:
        i = int(rng.integers(len(survivors)))
    grid = survivors[i]
    lo, hi = grid.extras.get("band", tilt_band(grid.tilt, grid.translate))
    excl = (len(set(arms)) > 1) if arms_excluded is None else arms_excluded
    draws = [float(rng.uniform(lo, hi)) if hi > lo else grid.tilt for _ in range(max(1, int(tilt_draws)) - 1)] + [float(grid.tilt)]
    st_ = stats if stats is not None else {}

    def fail(kind: str) -> None:  # why the drawn tilts fail (door 0919-023637: every draw of 19 survivors failed with no reason)
        st_[kind] = st_.get(kind, 0) + 1

    for tilt in draws:
        st = grid.with_trunk(grid.translate, tilt, name)
        st.focus = grid.focus if focus is None else _t(focus)
        st.arms = list(arms)
        st.extras = {k: v for k, v in grid.extras.items() if k in ("band", "angle_class", "clearance_tier", "bar_offset_m")}
        st.extras["grid_tilt"] = grid.tilt
        if not bool(body_free(s, [st], arms_excluded=excl, posture=posture, fingers=fingers)[0]):
            fail("collision")
            continue
        if posture is not None:
            st.fingers = fingers
            st.q = s.ikc.q_locked(st, arms=posture, fingers=fingers)
            st.extras["arm_posture"] = posture
        cam = camera_poses(s, [st])[0]
        if st.focus is not None:
            if not in_frame(s, cam, st.focus.reshape(1, 3), in_frame_margin):
                fail("focus_out_of_frame")
                continue
            ang = axis_angle(s, cam, st.focus)
            k = angle_class(ang)
            if k is None or k > int(grid.extras.get("angle_class", 1)):
                fail("axis_class")
                continue
            st.extras["axis_angle_deg"] = math.degrees(ang)
        if extra_visibility is not None and not extra_visibility(cam):
            fail("extra_visibility")
            continue
        return i, st
    return i, None


# ----------------------------------------------------------------------------------------------------------------
# step 5 — exact IK of the contact set; step 6 — path + transition
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ContactSolution:
    ok: bool
    q: Optional[th.Tensor] = None  # combined configuration (every solved arm at its solution)
    arm_q: Dict[str, th.Tensor] = field(default_factory=dict)
    reason: str = ""
    n_poses: int = 0


def exact_contacts(s, stance: Stance, contact_set: Sequence[ContactSpec], order: Optional[Sequence[str]] = None,
                   ignore_combined: Sequence[Any] = (), scale_key: Any = None) -> ContactSolution:
    """§5.2 step 5: the exact IK of the whole contact set from the drawn stance — locked trunk, idle arm untucked and
    locked; arm after arm in ``order`` (default: the contacts' order), every later arm seeded with the earlier arms at
    their solutions; per contact ``[pre, pose]`` exactly (``ignore_for_last`` for the pose only) and the waypoints per
    §3.9 (screening, exact endpoints; ``ignore_waypoints`` for every waypoint); finally one whole-body collision check
    of the combined configuration (``ignore_combined`` removed). ``screen_only`` contacts are skipped."""
    solve = [c for c in contact_set if not c.screen_only]
    arms = list(order) if order else list(dict.fromkeys(c.arm for c in solve))
    q_seed = s.ikc.q_locked(stance)
    arm_q: Dict[str, th.Tensor] = {}
    n = 0
    for arm in arms:
        for c in [c for c in solve if c.arm == arm]:
            pre = c.pre_pose()
            res = s.ikc.exact([pre, c.pose], stance, arm, ignore_for_last=list(c.ignore_for_last), q_seed=q_seed, scale_key=scale_key, cache=False)
            n += 2
            if not res.ok:
                return ContactSolution(False, None, arm_q, f"{c.name} ({arm}): pre / final pose fail the exact IK", n)
            q_last = res.q
            if c.waypoints:
                wps = ([c.pose] + list(c.waypoints)) if c.after else ([pre] + list(c.waypoints) + [c.pose])
                r2 = s.ikc.waypoints_ok(wps, stance, arm, ignore_for_last=list(c.ignore_for_last), ignore_all=list(c.ignore_waypoints), q_seed=q_seed,
                                        scale_key=scale_key)
                n += len(wps)
                if not r2.ok:
                    return ContactSolution(False, None, arm_q, f"{c.name} ({arm}): waypoints: {r2.reason}", n)
                if c.after:
                    q_last = r2.q if r2.q is not None else q_last
            # lock this arm at its solution for the next arm (the seed carries the arm joints)
            q_seed = q_seed.clone()
            q_seed[s.kit.robot.arm_control_idx[arm]] = res.q[s.kit.robot.arm_control_idx[arm]]
            arm_q[arm] = res.q[s.kit.robot.arm_control_idx[arm]].clone()
            _ = q_last
    if not arm_q:
        return ContactSolution(True, q_seed, {}, "nothing to solve", 0)
    with s.kit.ignoring(list(ignore_combined)):
        free = bool(s.kit.collision_free(q_seed, margin=COLLISION_MARGIN)[0])
    if not free:
        return ContactSolution(False, q_seed, arm_q, "combined configuration collides", n)
    return ContactSolution(True, q_seed, arm_q, "", n)


def connect(s, init: Stance, manip: Stance, path_max_m: float = PATH_MAX_M, carried: Optional[str] = None, arm: str = "right",
            blocked: Sequence[Any] = ()) -> Tuple[PathRecord, Transition]:
    """§5.2 step 6 in the §3.8 order: the base path init → manipulation (§3.7, ≤ ``path_max_m``) and the transition,
    both recorded in the sidecar; ``Reject`` (``no_path`` / ``path_collision`` / ``path_too_long`` / ``no_transition``)
    otherwise.

    The transition stand-off s_T is chosen first, smallest candidate first (§3.8 ``{s, s+0.1, …, s+0.5}``): at s_T the
    trunk sweep **and the drive-in s_T → s** are checked with the *arriving* stance's arms (its recorded configuration —
    the A6 posture of a two-arm stance, the idle posture of a one-arm stance — via ``ik.stance_arm_vectors``) and the
    arriving trunk. The path is then planned from ``init`` to the **drive-back pose at s_T** and checked at the
    *departing* stance's trunk and arms (the robot drives before the trunk / arm change). The recorded path therefore
    ends at the drive-back pose (``Path.to`` stays the manipulation stance: the oracle's Transition drive-back is a
    no-op there); driving the untucked arms all the way to a stance 0.5 m from a wall would otherwise put the hands
    into the wall, which is what blocked every two-arm wall stance in the pilot."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import ik as IK
    from ispatialgym.sampling.v2 import navigation as NAV

    carried_obj, arm, eef_in_object = s.carried_attachment(carried or manip.carried or init.carried, arm)
    tf, tt = (init.translate, init.tilt), (manip.translate, manip.tilt)
    s_val = float(manip.stand_off if manip.stand_off is not None else 0.0)
    inward = manip.inward
    x0, y0, yaw = (float(v) for v in manip.base_pose2d)
    right_arm, left_arm, fingers = IK.stance_arm_vectors(s.kit, manip)
    arm_q_to = {"right": right_arm, "left": left_arm}
    sweep = NAV.trunk_sweep(tf, tt, NAV.TRANSITION_SAMPLES)
    same_trunk = abs(tf[0] - tt[0]) < 1e-6 and abs(tf[1] - tt[1]) < 1e-6
    rooms = s.gp.rooms(s.scene) or s.world.rooms
    why: List[str] = []
    path_err: Optional[NAV.NavigationError] = None
    for s_T in NAV.transition_standoffs(s_val):
        back = s_T - s_val
        bx, by = x0 - float(inward[0]) * back, y0 - float(inward[1]) * back
        if back > 0 and not s.kit.trav_free((bx, by)):
            why.append(f"s_T {s_T:.2f}: drive-back pose off the free floor")
            continue
        with NAV.attached(s.kit, arm if carried_obj is not None else None, carried_obj, eef_in_object=eef_in_object):
            if not same_trunk:  # (a) the trunk sweep at s_T with the arriving arms
                qs = th.stack([s.kit.q_at_base_pose(s.kit.q_config(arms="untucked", translate=tr, tilt=ti, right_arm=right_arm, left_arm=left_arm, fingers=fingers),
                                                    bx, by, yaw) for tr, ti in sweep])
                if not bool(s.kit.collision_free(qs, margin=COLLISION_MARGIN).all()):
                    why.append(f"s_T {s_T:.2f}: trunk sweep collides")
                    continue
            if back > 0:  # (b) the drive-in s_T → s at the arriving trunk and arms
                drive = NAV.stride_poses([(bx, by, yaw), (x0, y0, yaw)], NAV.STRIDE_M)
                ok, bad = NAV.path_collision_free(s.kit, drive, tt, "untucked", margin=COLLISION_MARGIN, q_arms=arm_q_to)
                if not ok:
                    why.append(f"s_T {s_T:.2f}: drive-in collides at stride {bad}/{len(drive)}")
                    continue
        # (c) the path init → drive-back pose at the departing trunk and arms (§3.7)
        back_stance = Stance(manip.name, (bx, by, yaw), manip.translate, manip.tilt, s_T, manip.edge, manip.face, manip.lateral, manip.focus, manip.arms,
                             list(manip.contacts), None, manip.carried, dict(manip.extras), manip.fingers)
        try:
            # owner 2026-09-26: ``generation.base_path_style`` - the approach of an opted-in family (large articulation, alignment) is checked and
            # recorded as the face_travel schedule when uncarried (recorded fallback when the schedule collides); a carried approach stays recorded
            is_carried = carried_obj is not None
            path = NAV.plan_path(s.kit, init, back_stance, carried=carried, carried_obj=carried_obj, carried_arm=arm, rooms=rooms, blocked=blocked,
                                 max_length_m=path_max_m, trunk=tf, eef_in_object=eef_in_object,
                                 style="recorded" if is_carried else str(s.gp.gen("base_path_style", "recorded") or "recorded"), probe_face_travel=is_carried)
        except NAV.NavigationError as e:
            why.append(f"s_T {s_T:.2f}: {e.reason}")
            path_err = e
            continue
        s.sidecar.add_path(path)
        tr = Transition(init.name, manip.name, s_T, trunk_from=tf, trunk_to=tt, n_samples=0 if same_trunk else NAV.TRANSITION_SAMPLES)
        s.sidecar.add_transition(tr)
        log.info("step 6: path %s -> %s %.2f m (%s) to the drive-back pose at s_T %.2f, drive-in %.2f m with the arriving arms", init.name, manip.name,
                 path.length_m, path.mode, s_T, back)
        return path, tr
    if path_err is not None:
        raise Reject(path_err.reason, f"{init.name} -> {manip.name}: {path_err} ({'; '.join(why[-3:])})")
    raise Reject("no_transition", f"{init.name} -> {manip.name}: no s_T in {NAV.transition_standoffs(s_val)} passes the trunk sweep + drive-in with the arriving arms "
                                  f"({'; '.join(why[-4:])})")


def summarize_stance(st: Stance) -> Dict[str, Any]:
    """Compact per-instance summary of a Prompt-0 stance (template_meta / instance summary)."""
    d = {"base_pose2d": [round(float(v), 4) for v in st.base_pose2d], "translate": st.translate, "tilt": round(st.tilt, 4), "stand_off": st.stand_off,
         "lateral": round(float(st.lateral), 4), "arms": st.arms}
    if st.focus is not None:
        d["focus"] = [round(float(v), 4) for v in st.focus.tolist()]
    for k in ("bearing_deg", "distance", "axis_angle_deg", "angle_class", "grid_tilt", "band"):
        if k in st.extras:
            d[k] = st.extras[k]
    return d
