"""IK conventions of the v2 sampler (spec §3.6, §3.9, §3.10) — the only place cuRobo IK is called from v2.

* **Screening** = 12-seed batched IK per pose (``RobotKit.ik_batch``); **exact** = 32-seed IK (``RobotKit.ik``).
* **Locked joints** = the untucked init configuration at the stance's trunk (``q_locked``); the idle arm is untucked and
  part of the collision body; the working arm's solution is what the oracle tracks.
* **"Support ignored for the grasp pose only"**: ``ignore`` objects leave the cuRobo world for the *last* pose of a
  sequence (grasp / contact) and for nothing else — every other object (partner, distractors, scene) stays.
* **Waypoints** (§3.9): ``waypoints_ok`` screens the intermediate poses and confirms the endpoints exactly; the
  ``policy`` string is what the sidecar records as ``waypoint_policy``.
* **eef in frame** (§3.9): the eef body (gripper + fingertips) at the final pose projects inside the image with 5 %.
* **Bands' feasible sets F** (§4.6 / §3.10): ``feasible_sets`` screens (yaw, grasp) pairs over cells in one batch per
  trunk and returns F per cell.
* **Exact-IK cache** of v1 (``ISPATIALGYM_SAMPLING_CACHE``): every key includes the spawn scale vector of the object.

Geometry (grasp / pre / contact poses, waypoints) is ``ispatialgym.robot.conventions`` (F5, commit ``dc91cfb``); the
thin wrappers below only fix the argument shapes the task modules use (tensor pairs in, tensor pairs out) — v2 never
re-implements a conventions function (README contract 2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.cache import cache_get, cache_key, cache_put
from ispatialgym.sampling.v2.stances import Pose, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SCREEN_SEEDS = 12
EXACT_SEEDS = 32
PRE_DIST = C.PRE_DIST
HANDLE_OFFSET = C.HANDLE_EEF_OFFSET
IN_FRAME_MARGIN = 0.05
FINGER_OPEN_DEFAULT = C.FINGER_OPEN
FINGER_CLOSED = C.FINGER_CLOSED
#: a segment between consecutive chain poses counts as a straight *approach* (tracked from the previous solution) when
#: the orientation is the same within this angle (rad) and the distance at most this (m): pre -> grasp / place /
#: contact (``PRE_DIST`` 0.10), the wall brace chain, lifts and descents recorded as pose lists
APPROACH_SAME_ORIENT_RAD = 0.05
APPROACH_MAX_M = 0.25


def is_approach(a: Pose, b: Pose) -> bool:
    """Whether ``a -> b`` is a straight same-orientation segment of at most ``APPROACH_MAX_M`` (a pre-pose to its end pose)."""
    d = float(th.linalg.norm(G.t(b[0]) - G.t(a[0])))
    return 1e-4 < d <= APPROACH_MAX_M and G.quat_angle(G.t(a[1]), G.t(b[1])) <= APPROACH_SAME_ORIENT_RAD

# ----------------------------------------------------------------------------------------------------------------
# conventions (F5) — argument-shape wrappers
# ----------------------------------------------------------------------------------------------------------------


def _pair(pose) -> Pose:
    p, q = C.as_pose(pose)
    return G.t(p), G.t(q)


def approach_axis(pose: Pose) -> th.Tensor:
    """a = eef +z in world (conventions ``approach_axis``)."""
    return C.approach_axis(G.t(pose[1]))


def pre_pose(pose: Pose, dist: float = PRE_DIST) -> Pose:
    """Pre-grasp / pre-contact / pre-place: the pose moved ``dist`` back along -a (spec §3.6; conventions ``pre_pose``)."""
    return _pair(C.pre_pose((G.t(pose[0]), G.t(pose[1])), dist))


def grasp_pose(obj_pose: Pose, eef_in_object: Pose) -> Pose:
    """eef pose = object pose ∘ T_obj→eef (conventions ``grasp_pose_from_transform``; the grasp-library entry gives
    T_obj→eef; the recorded ``grasps[].yaw`` is the object yaw, so consumers re-deriving from the settled object pose
    call ``conventions.grasp_pose(obj_pose, key, name)`` with ``yaw = 0``)."""
    return _pair(C.grasp_pose_from_transform((G.t(obj_pose[0]), G.t(obj_pose[1])), (G.t(eef_in_object[0]), G.t(eef_in_object[1]))))


def contact_eef(point, approach_into_object, closing=None) -> Pose:
    """Push / brace contact: pad centre on ``point``, approach a along ``approach_into_object`` (= -outward normal),
    pads closing along ``closing`` (default horizontal ⊥ a), eef origin = point − CONTACT_CENTER_Z·a (spec §3.6;
    conventions ``contact_eef``)."""
    return _pair(C.contact_eef(G.t(point), G.t(approach_into_object), None if closing is None else G.t(closing)))


def handle_eef(contact, approach, closing=None) -> Pose:
    """Handle grasp: approach a along ``approach`` (into the handle), pads closing along ``closing`` (default: horizontal
    ⊥ a), eef origin = contact − 0.012·a (spec §3.6; conventions ``handle_eef``)."""
    a = C.unit(G.t(approach))
    return _pair(C.handle_eef(G.t(contact), a, C._default_closing(a) if closing is None else G.t(closing)))


def line_waypoints(p0: Pose, p1: Pose, n: int = 3, endpoints: bool = False) -> List[Pose]:
    """Straight eef motion p0 -> p1 at fractions k/n (start excluded, goal last; ``endpoints=True`` prepends the start),
    orientation SLERPed (conventions ``line_waypoints``)."""
    return [_pair(p) for p in C.line_waypoints((G.t(p0[0]), G.t(p0[1])), (G.t(p1[0]), G.t(p1[1])), n, endpoints)]


# ----------------------------------------------------------------------------------------------------------------
# stance arm postures (A6)
# ----------------------------------------------------------------------------------------------------------------

POSTURES = {"untucked": "untucked", "tucked": "tucked", "reset": "tucked", "ready": "ready"}


def stance_arm_vectors(kit, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                       fingers: Optional[float] = None) -> Tuple[th.Tensor, th.Tensor, float]:
    """``(right_arm, left_arm, fingers)`` of a stance, in this order of precedence: the explicit arguments, the stance's
    **recorded configuration** ``stance.q`` (§3.3 exceptions: the held-object hold, the camera-relocalization reset arms)
    and ``stance.fingers``, then per arm: the working arms (``stance.arms`` list) untucked, the idle arm in
    ``stance.idle_posture`` (A6: untucked / ready / tucked), a posture string ``arms`` / ``stance.arms`` for both."""
    robot = kit.robot
    if stance.q is not None:
        if right_arm is None:
            right_arm = stance.q[robot.arm_control_idx["right"]].clone()
        if left_arm is None:
            left_arm = stance.q[robot.arm_control_idx["left"]].clone()
        if fingers is None and stance.fingers is None:
            fingers = float(stance.q[robot.gripper_control_idx["right"]][0])
    if fingers is None:
        fingers = stance.fingers if stance.fingers is not None else FINGER_OPEN_DEFAULT
    both = POSTURES.get(arms or stance.posture, "untucked")
    working = stance.working_arms
    for arm in ("right", "left"):
        cur = right_arm if arm == "right" else left_arm
        if cur is not None:
            continue
        name = POSTURES.get(stance.idle_posture, "untucked") if (working and arm not in working and stance.idle_posture) else both
        vec = kit.arm_q(name, arm)
        if arm == "right":
            right_arm = vec
        else:
            left_arm = vec
    return right_arm, left_arm, float(fingers)


# ----------------------------------------------------------------------------------------------------------------
# IK checker
# ----------------------------------------------------------------------------------------------------------------


class _Ignoring:
    """Switch the collision meshes of ``objs`` off in the cuRobo world for the block; meshes disabled for another
    reason (attached objects, removed objects) are left alone. Falls back to ``RobotKit.ignoring`` (two world syncs)
    when the backend has no per-mesh switch."""

    def __init__(self, kit, objs: List[Any]):
        self.kit = kit
        self.objs = objs
        self.names: List[str] = []
        self._fallback = None

    def __enter__(self):
        if not self.objs:
            return self
        backend = self.kit.backend
        if not (hasattr(backend, "_mesh_enable") and hasattr(backend, "_world_mesh_names")):
            self._fallback = self.kit.ignoring(self.objs)
            self._fallback.__enter__()
            return self
        self.kit._ensure_synced()
        disabled = getattr(backend, "_disabled_obstacles", set())
        active = _ACTIVE_IGNORES.setdefault(id(backend), {})
        for o in self.objs:
            for link in getattr(o, "links", {}).values():
                for cm in getattr(link, "collision_meshes", {}).values():
                    name = cm.prim_path
                    if name in backend._world_mesh_names and name not in disabled:
                        # nested contexts may ignore the same mesh: it is switched off once and back on when the last
                        # context that asked for it exits (refcount)
                        if active.get(name, 0) == 0:
                            backend._mesh_enable(name, False)
                        active[name] = active.get(name, 0) + 1
                        self.names.append(name)
        return self

    def __exit__(self, *exc):
        if self._fallback is not None:
            self._fallback.__exit__(*exc)
            return False
        active = _ACTIVE_IGNORES.setdefault(id(self.kit.backend), {})
        for name in self.names:
            active[name] = active.get(name, 1) - 1
            if active[name] <= 0:
                active.pop(name, None)
                self.kit.backend._mesh_enable(name, True)
        self.names = []
        return False


#: per backend: collision meshes currently switched off by active ``_Ignoring`` contexts -> number of contexts asking
_ACTIVE_IGNORES: Dict[int, Dict[str, int]] = {}


@dataclass
class IKResult:
    ok: bool
    q: Optional[th.Tensor] = None
    reason: str = ""
    n_checked: int = 0

    @property
    def code(self) -> str:
        """Distinct rejection code carried by the reason (``approach_untrackable``: a straight approach whose end pose has
        no configuration in the pre pose's IK branch), else ``""`` — callers use ``res.code or "<their reason>"``."""
        return "approach_untrackable" if self.reason.startswith("approach_untrackable") else ""


class IKChecker:
    """§3.6 / §3.9 IK checks over a ``RobotKit``. Stateless apart from the kit and the cache namespace."""

    def __init__(self, kit, cache_scope: Optional[Dict[str, Any]] = None, default_working_arms: Sequence[str] = ("right",)):
        self.kit = kit
        self.cache_scope = dict(cache_scope or {})  # scene / rooms / placed objects: part of every cache key
        #: the task's working arm(s) (``generation.arm``; core sets it) for stances the search checks before the module names them
        self.default_working_arms = list(default_working_arms)
        #: (arm, rounded pose) -> 28-D solution of a successful exact solve; cuRobo's remaining seeds are random, so a
        #: marginal pose can pass once and fail when re-solved (the program check re-solves every recorded pose): the
        #: sampler's own solution is used as the seed configuration then. Cleared per attempt (``core``).
        self.solutions: Dict[Tuple[str, Tuple[float, ...]], th.Tensor] = {}
        self._last_track_start: Optional[th.Tensor] = None  # pre-pose configuration of the last tracked approach

    def ignoring(self, objs: Sequence[Any]):
        """Context manager: ``objs`` leave the cuRobo collision world for the queries inside (§3.6 "support ignored for the
        grasp pose only"). Their collision meshes are switched off by name (``R1ProCuRobo._mesh_enable``, what
        ``attach_object`` does) instead of the two full world re-syncs of ``RobotKit.ignoring`` (0.5–13 s each)."""
        return _Ignoring(self.kit, list(objs))

    @staticmethod
    def _pose_key(pose: Pose) -> Tuple[float, ...]:
        p, q = G.t(pose[0]), G.t(pose[1])
        return tuple(round(float(v), 3) for v in p.tolist()) + tuple(round(float(v), 3) for v in q.tolist())

    SEED_POS_TOL = 0.03
    SEED_ANG_TOL = 0.15

    #: the solver's own acceptance (cuRobo ``position_tolerance`` 5 mm / ``orientation_tolerance`` 0.05 rad)
    WITNESS_POS_TOL = 0.005
    WITNESS_ANG_TOL = 0.05

    def _witness_ok(self, q: th.Tensor, pose: Pose, arm: str) -> bool:
        """``q`` reaches ``pose`` within the IK solver's acceptance (5 mm / 0.05 rad) and is collision-free in the current
        world at the solver's own margin (0: the IK's collision term has no extra clearance) — the same criterion the
        solver applies to its own solutions, evaluated deterministically."""
        from ispatialgym.oracle import r1pro_model as M

        try:
            eef = self.kit.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]
        except Exception:
            return False
        if float(th.linalg.norm(G.t(eef[0]) - G.t(pose[0]))) > self.WITNESS_POS_TOL or G.quat_angle(eef[1], G.t(pose[1])) > self.WITNESS_ANG_TOL:
            return False
        return bool(self.kit.collision_free(q, margin=0.0)[0])

    def _seeded(self, q0: th.Tensor, pose: Pose, arm: str) -> th.Tensor:
        """``q0`` with the working arm set to the remembered solution of the nearest recorded pose of ``arm`` within
        (3 cm, 0.15 rad) — the same pose, or the same pose after the settle moved the object by a few mm (the other
        joints stay ``q0``'s: the base virtual joints must belong to the current robot root)."""
        p, q = G.t(pose[0]), G.t(pose[1])
        items = [(key, q_rec) for (a, key), q_rec in self.solutions.items() if a == arm]
        if not items:
            return q0
        keys = th.tensor([k for k, _ in items], dtype=th.float32)  # (M, 7): position + quaternion (vectorised search)
        d = th.linalg.norm(keys[:, :3] - p, dim=1)
        kq = keys[:, 3:] / th.linalg.norm(keys[:, 3:], dim=1, keepdim=True).clamp(min=1e-9)
        dots = th.clamp((kq * (q / th.linalg.norm(q).clamp(min=1e-9))).sum(dim=1).abs(), max=1.0)
        ang = 2.0 * th.acos(dots)
        ok = (d <= self.SEED_POS_TOL) & (ang <= self.SEED_ANG_TOL)
        if not bool(ok.any()):
            return q0
        d = th.where(ok, d, th.full_like(d, float("inf")))
        best = items[int(th.argmin(d))][1]
        out = q0.clone()
        idx = self.kit.robot.arm_control_idx[arm]
        out[idx] = best[idx]
        return out

    #: wide radius for *seeding* (not witnessing) a re-solve: the chain's other pose (PRE_DIST 0.10 away), an earlier draw
    RETRY_SEED_POS_TOL = 0.20
    RETRY_SEED_ANG_TOL = 0.8

    def _retry_seeds(self, q0: th.Tensor, pose: Pose, arm: str, chain: Sequence[th.Tensor] = ()) -> List[th.Tensor]:
        """Seeds for a re-solve after the random seeds missed ``pose``: the configurations found earlier in the same chain
        (nearest pose first) and the remembered solutions of ``arm`` within the wide radius, nearest first, up to 4."""
        p, q = G.t(pose[0]), G.t(pose[1])
        idx = self.kit.robot.arm_control_idx[arm]
        out: List[th.Tensor] = []
        for q_c in reversed(list(chain)):
            q_s = q0.clone()
            q_s[idx] = G.t(q_c)[idx]
            out.append(q_s)
        items = [(key, q_rec) for (a, key), q_rec in self.solutions.items() if a == arm]
        if items:
            keys = th.tensor([k for k, _ in items], dtype=th.float32)
            d = th.linalg.norm(keys[:, :3] - p, dim=1)
            kq = keys[:, 3:] / th.linalg.norm(keys[:, 3:], dim=1, keepdim=True).clamp(min=1e-9)
            ang = 2.0 * th.acos(th.clamp((kq * (q / th.linalg.norm(q).clamp(min=1e-9))).sum(dim=1).abs(), max=1.0))
            ok = (d <= self.RETRY_SEED_POS_TOL) & (ang <= self.RETRY_SEED_ANG_TOL)
            order = th.argsort(th.where(ok, d, th.full_like(d, float("inf"))))
            for j in order.tolist():
                if not bool(ok[j]) or len(out) >= 4:
                    break
                q_s = q0.clone()
                q_s[idx] = items[j][1][idx]
                out.append(q_s)
        return out

    # -- configurations -----------------------------------------------------------------------------------------
    POSTURES = {"untucked": "untucked", "tucked": "tucked", "reset": "tucked", "ready": "ready"}
    #: A6 (owner decision 2026-09-16): the idle arm of a manipulation stance may take any recorded collision-free posture;
    #: the alternatives tried, in order, when the untucked idle arm collides
    IDLE_POSTURES = ("untucked", "ready", "tucked")

    def arm_vectors(self, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                    fingers: Optional[float] = None) -> Tuple[th.Tensor, th.Tensor, float]:
        """``(right_arm, left_arm, fingers)`` of a stance — see :func:`stance_arm_vectors`."""
        return stance_arm_vectors(self.kit, stance, arms, right_arm, left_arm, fingers)

    def q_locked(self, stance: Stance, arms: Optional[str] = None, right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None,
                 fingers: Optional[float] = None) -> th.Tensor:
        """28-D configuration at the stance (base pose + trunk) — the locked-joint source and IK seed of every check from
        it; arms and fingers per :meth:`arm_vectors`. The base virtual joints are always re-derived for the robot's
        current root."""
        r, l, f = self.arm_vectors(stance, arms, right_arm, left_arm, fingers)
        q = self.kit.q_config(arms="untucked", translate=stance.translate, tilt=stance.tilt, right_arm=r, left_arm=l, fingers=f)
        return self.kit.q_at_base_pose(q, *stance.base_pose2d)

    def stance_collision_free(self, stance: Stance, margin: float = 0.01, arms: Optional[str] = None, **arm_q) -> bool:
        return bool(self.kit.collision_free(self.q_locked(stance, arms, **arm_q), margin=margin)[0])

    def stances_collision_free(self, stances: Sequence[Stance], margin: float = 0.01, try_idle_postures: bool = True) -> th.Tensor:
        """Whole-body collision mask of the stances at their configuration. A6: a manipulation stance (``arms`` = list of
        working arms) whose untucked idle arm collides is retried with the alternative idle postures (``IDLE_POSTURES``,
        batched per posture); a survivor records the posture that passed in ``stance.idle_posture`` (its joint values go
        to the sidecar as ``arms_q``). Init stances (``arms = "untucked"``) are never re-postured."""
        if not stances:
            return th.zeros(0, dtype=th.bool)
        free = self.kit.collision_free(th.stack([self.q_locked(s) for s in stances]), margin=margin).clone()
        if not try_idle_postures:
            return free

        def idle_of(s: Stance) -> Optional[str]:
            # the stance search (tabletop Phase 2 / prompt0) checks candidates before the task module names the working
            # arm: a non-init stance without one uses the task's default working arm (``default_working_arms``)
            working = s.working_arms or (list(self.default_working_arms) if s.name not in ("init", "goal") and s.posture == "untucked" else [])
            if len(working) != 1:
                return None
            return "left" if working[0] == "right" else "right"

        retry = [i for i, s in enumerate(stances) if not bool(free[i]) and s.q is None and idle_of(s) is not None]
        for name in self.IDLE_POSTURES:
            if not retry:
                break
            todo = [i for i in retry if (stances[i].idle_posture or "untucked") != name]
            if not todo:
                continue
            qs = []
            for i in todo:
                s = stances[i]
                idle = idle_of(s)
                vec = self.kit.arm_q(self.POSTURES[name], idle)
                qs.append(self.q_locked(s, right_arm=vec if idle == "right" else None, left_arm=vec if idle == "left" else None))
            ok = self.kit.collision_free(th.stack(qs), margin=margin)
            for i, f in zip(todo, ok.tolist()):
                if f:
                    stances[i].idle_posture = name
                    if not stances[i].working_arms:  # record which arm the posture belongs to
                        stances[i].arms = list(self.default_working_arms)
                    free[i] = True
            retry = [i for i in retry if not bool(free[i])]
        return free

    # -- screening / exact --------------------------------------------------------------------------------------
    def screen(self, poses: Sequence[Pose], stance: Stance, arm: str = "right", ignore: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None,
               fingers: Optional[float] = None, seeds: Sequence[th.Tensor] = (), witnesses: bool = True) -> th.Tensor:
        """12-seed batched IK per pose from the stance (locked joints = the stance's recorded configuration / posture at
        its trunk, ``fingers`` overrides the opening: closed for push / brace contacts); ``ignore`` objects are removed
        from the world for *these* poses (callers pass the support for grasp poses only).

        Before the solver is asked, a **remembered solution** of the same pose (the sampler's exact solve of that
        waypoint) is verified as a witness — FK + collision, deterministic — and counts as a pass; the 12 random seeds
        around a tucked / ready seed otherwise miss poses the 32-seed exact IK finds (the wall brace chain, 2 cm off the
        wall). Poses the batched solve still misses are retried seeded from the nearest remembered solution and from
        ``seeds`` (callers pass the endpoint solutions of a waypoint chain). Solutions found are remembered."""
        if len(poses) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        poses = list(poses)
        out = th.zeros(len(poses), dtype=th.bool)
        idx = self.kit.robot.arm_control_idx[arm]
        with self.ignoring(ignore):
            todo: List[int] = []
            for i, p in enumerate(poses):
                q_near = self._seeded(q0, p, arm) if witnesses else q0
                if not th.equal(q_near, q0) and self._witness_ok(q_near, p, arm):
                    out[i] = True
                else:
                    todo.append(i)
            if todo:
                ok, qs = self.kit.ik_batch_solve([poses[i] for i in todo], q0, arm=arm)
                for j, i in enumerate(todo):
                    if bool(ok[j]):
                        out[i] = True
                        self.solutions[(arm, self._pose_key(poses[i]))] = qs[j].clone()
            for i in [i for i in todo if not bool(out[i])]:
                # alternative seeds: the nearest remembered solution (a neighbouring waypoint), then the caller's seeds
                cands: List[th.Tensor] = []
                q_near = self._seeded(q0, poses[i], arm)
                if not th.equal(q_near, q0):
                    cands.append(q_near)
                for s in seeds:
                    if s is not None:
                        q_s = q0.clone()
                        q_s[idx] = G.t(s)[idx]
                        cands.append(q_s)
                for q_s in cands:
                    ok1, q1 = self.kit.ik_batch_solve([poses[i]], q_s, arm=arm)
                    if bool(ok1[0]):
                        out[i] = True
                        self.solutions[(arm, self._pose_key(poses[i]))] = q1[0].clone()
                        break
        return out

    def screen_pairs(self, pairs: Sequence[Sequence[Pose]], stance: Stance, arm: str = "right", ignore_for_last: Sequence[Any] = (),
                     q_seed: Optional[th.Tensor] = None, fingers: Optional[float] = None) -> th.Tensor:
        """Batched screening of [pre, final] pose pairs: pre against the full world, final with ``ignore_for_last`` removed."""
        if len(pairs) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        pres = [p[0] for p in pairs]
        finals = [p[-1] for p in pairs]
        ok_pre = self.kit.ik_batch(pres, q0, arm=arm)
        idx = [i for i, s in enumerate(ok_pre.tolist()) if s]
        out = th.zeros(len(pairs), dtype=th.bool)
        if idx:
            with self.ignoring(ignore_for_last):
                ok_f = self.kit.ik_batch([finals[i] for i in idx], q0, arm=arm)
            for i, s in zip(idx, ok_f.tolist()):
                out[i] = bool(s)
        return out

    def exact(self, poses: Sequence[Pose], stance: Stance, arm: str = "right", ignore_for_last: Sequence[Any] = (),
              q_seed: Optional[th.Tensor] = None, scale_key: Any = None, cache: bool = True, fingers: Optional[float] = None,
              approached: Optional[Sequence[Any]] = None) -> IKResult:
        """32-seed IK of every pose in order from the stance (locked joints = the stance's recorded configuration /
        posture, ``fingers`` overrides the gripper opening); the last pose with ``ignore_for_last`` removed (the support
        under a grasp). Returns the configuration at the last pose. ``scale_key`` (the spawn scale vector of the object
        the poses belong to) is part of the cache key (§3.10).

        ``approached`` (owner decision 2026-09-16 10:15, A): the object the tracked approach segments close in on — the
        grasped target, the placed-on partner, the touched object — is removed from the world along the **whole** segment
        (side / top grasps straddle it over the last 10 cm by construction); the supports of ``ignore_for_last`` stay
        removed for the last waypoint only, the pre pose itself keeps the full world. ``None`` infers it from the end
        pose (``approached_objects``: the spawned objects whose expanded AABB holds the pad centre / fingertip)."""
        if len(poses) == 0:
            return IKResult(True, None, "no poses", 0)
        q0 = self.q_locked(stance, fingers=fingers) if q_seed is None else q_seed
        key = None
        if cache:
            key = cache_key("exact_ik_v3", {**self.cache_scope, "base": list(stance.base_pose2d), "trunk": [stance.translate, stance.tilt], "arm": arm,
                                            "poses": [list(G.t(p[0]).tolist()) + list(G.t(p[1]).tolist()) for p in poses],
                                            "ignore": sorted(getattr(o, "name", str(o)) for o in ignore_for_last), "scale": scale_key,
                                            "seeds": EXACT_SEEDS, "seed_arms": q0[self.kit.robot.arm_control_idx[arm]],
                                            "locked": q0[[i for a in ("left", "right") for i in self.kit.robot.arm_control_idx[a].tolist()] + self.kit.robot.gripper_control_idx["right"].tolist()]})
            hit = cache_get(key)
            if hit is not None:
                q_hit = None if hit.get("q") is None else th.tensor(hit["q"])
                if q_hit is not None:
                    self.solutions[(arm, self._pose_key(poses[-1]))] = q_hit.clone()
                # every pose of the chain keeps its witness (entries written before 2026-09-16 14:45 carry the last only):
                # without them a later re-solve of the pre pose at the settled object pose starts from cold seeds
                for p_i, q_i in zip(poses, hit.get("qs") or []):
                    if q_i is not None:
                        self.solutions[(arm, self._pose_key(p_i))] = th.tensor(q_i)
                log.debug("exact IK cache hit (%d poses, arm %s, witnesses for %d)", len(poses), arm, len(hit.get("qs") or []))
                return IKResult(bool(hit["ok"]), q_hit, "cache", len(poses))
        q = None
        qs: List[Optional[th.Tensor]] = []
        n = len(poses)
        why = "exact IK failed"
        for i, p in enumerate(poses):
            ign = list(ignore_for_last) if i == n - 1 else []
            if i > 0 and q is not None and is_approach(poses[i - 1], p):
                # a straight same-orientation segment from the previous pose (pre-grasp -> grasp, pre-place -> place,
                # pre-contact -> contact): the end pose is *tracked* from the previous solution so both witnesses lie in
                # one IK branch and the oracle's straight approach is executable (oracle finding: the french-fries pre_grasp
                # / grasp witnesses 3.16 rad apart -> plan_failed); an untrackable approach fails the chain
                near = list(approached) if approached is not None else self.approached_objects(p)
                q, why = self.track_approach(q, poses[i - 1], p, arm, ignore_end=ign, ignore_mid=near, q_hint=q0)
                if q is None:
                    break
                qs.append(q)
                if qs and len(qs) >= 2:
                    qs[-2] = self._last_track_start.clone()
                self.solutions[(arm, self._pose_key(poses[i - 1]))] = self._last_track_start.clone()
            else:
                with self.ignoring(ign):
                    # a remembered solution of this very pose is verified as a witness (FK + collision) before the solver
                    # is asked again (cuRobo's IK is randomised, a witness check is not); otherwise it seeds the solver
                    q_near = self._seeded(q0, p, arm) if q_seed is None else q0
                    if q_seed is None and not th.equal(q_near, q0) and self._witness_ok(q_near, p, arm):
                        q = q_near
                    else:
                        q = self.kit.ik(p, q_near, arm=arm)
                    if q is None:
                        # the 32 random seeds missed: retry seeded from the nearest remembered solution within the wide
                        # radius (the chain's other pose 10 cm away, an earlier draw of the same grasp) and from the
                        # solutions already found in this chain
                        for q_alt in self._retry_seeds(q0, p, arm, [x for x in qs if x is not None]):
                            q = self.kit.ik(p, q_alt, arm=arm)
                            if q is not None:
                                break
                if q is None:
                    why = f"exact IK failed at pose {i + 1} of {n}"
                    break
            qs.append(q)
            self.solutions[(arm, self._pose_key(p))] = q.clone()
        res = IKResult(q is not None, q, "" if q is not None else why, len(poses))
        if key is not None and res.ok:  # a failure may be a random miss of the solver: only witnesses are cached
            cache_put(key, {"ok": True, "q": q.tolist(), "qs": [None if x is None else x.tolist() for x in qs]})
        return res

    # -- straight approaches (branch consistency) -----------------------------------------------------------------
    #: dense tracking step along an approach (m), the largest joint change allowed per dense step (rad), the largest
    #: joint excursion allowed between two §3.9 waypoints (~3.3 cm apart; the oracle's finishing-motion bound), and the
    #: longest segment treated as an approach
    APPROACH_STEP_M = 0.01
    APPROACH_MAX_STEP = 0.2
    APPROACH_MAX_EXCURSION = 0.6
    APPROACH_WAYPOINT_M = 0.1 / 3

    #: an object counts as approached by an end pose when its AABB, grown by this, holds the pad centre or the fingertip
    APPROACHED_AABB_PAD = 0.03

    def approached_objects(self, end_pose: Pose) -> List[Any]:
        """Spawned objects an approach ending at ``end_pose`` closes in on: those whose AABB (hull at the PhysX pose when
        the world caches it, else ``obj.aabb``) grown by ``APPROACHED_AABB_PAD`` contains the pad centre or the fingertip
        point of the end pose. A grasp straddles the target, a place hovers 1 cm over the partner, a contact sits 5 mm
        off the touched face: all three land inside. Objects attached to the planner are already out of the world."""
        world = getattr(self.kit, "world", None)
        spawned = list(getattr(world, "spawned", {}).values()) if world is not None else []
        if not spawned:
            return []
        if hasattr(world, "sync_render_transforms"):
            world.sync_render_transforms()  # ``obj.aabb`` reads the render-side transform (stale after physics-only steps)
        pts = th.stack([C.pad_center((G.t(end_pose[0]), G.t(end_pose[1]))), C.fingertip((G.t(end_pose[0]), G.t(end_pose[1])))])
        out = []
        for o in spawned:
            try:
                lo, hi = o.aabb
            except Exception:  # noqa: BLE001
                continue
            lo, hi = G.t(lo) - self.APPROACHED_AABB_PAD, G.t(hi) + self.APPROACHED_AABB_PAD
            if bool(((pts >= lo) & (pts <= hi)).all(dim=1).any()):
                out.append(o)
        return out

    def track_segment(self, q_from: th.Tensor, pose_from: Pose, pose_to: Pose, arm: str, ignore_mid: Sequence[Any] = (),
                      ignore_end: Sequence[Any] = (), ignore_start: Sequence[Any] = ()) -> Tuple[Optional[th.Tensor], str]:
        """Differential-IK tracking (``backend.solve_ik_path``) of the straight segment ``pose_from -> pose_to`` from the
        configuration ``q_from`` (which reaches ``pose_from``) in ``APPROACH_STEP_M`` steps: continuous joint path, no
        step above ``APPROACH_MAX_STEP`` rad, joint excursion between consecutive §3.9 waypoints ≤ ``APPROACH_MAX_EXCURSION``,
        every configuration collision-free with ``ignore_mid`` removed — the ones within ``APPROACH_WAYPOINT_M`` of the end
        pose additionally with ``ignore_end`` removed (the last centimetres of a grasp descent run inside the support's
        sphere margin, as the end pose does: §3.6 "support ignored for the grasp pose only" at the §3.9 waypoint
        granularity), the ones within it of the start pose with ``ignore_start`` removed (backward tracking).
        Returns ``(q_end, "ok")`` or ``(None, reason)``."""
        from ispatialgym.oracle import r1pro_model as M

        p0, p1 = (G.t(pose_from[0]), G.t(pose_from[1])), (G.t(pose_to[0]), G.t(pose_to[1]))
        dist = float(th.linalg.norm(p1[0] - p0[0]))
        n = max(2, int(math.ceil(dist / self.APPROACH_STEP_M)))
        wps = [tuple(w) for w in C.line_waypoints(p0, p1, n)]
        self.kit._ensure_synced()
        with self.ignoring(ignore_mid):
            res = self.kit.backend.solve_ik_path(f"{arm}_arm", [{M.EEF_LINK[arm]: (G.t(w[0]), G.t(w[1]))} for w in wps], q_start=G.t(q_from),
                                                 check_collisions=False, sync_world=False, max_joint_step=self.APPROACH_MAX_STEP, max_iters=60)
        self.kit.stats["ik_calls"] = self.kit.stats.get("ik_calls", 0) + 1
        if not res.success:
            return None, f"approach_untrackable: {res.status}"
        traj = res.q_traj  # (n + 1, 28): the start and one configuration per dense waypoint
        idx = self.kit.robot.arm_control_idx[arm]
        k = max(1, int(round(self.APPROACH_WAYPOINT_M / self.APPROACH_STEP_M)))
        coarse = list(range(0, len(traj) - 1, k)) + [len(traj) - 1]
        exc = max(float(th.max(th.abs(traj[b, idx] - traj[a, idx]))) for a, b in zip(coarse[:-1], coarse[1:]))
        if exc > self.APPROACH_MAX_EXCURSION:
            return None, f"approach_untrackable: joint excursion {exc:.2f} rad between consecutive waypoints (> {self.APPROACH_MAX_EXCURSION})"
        # configuration i (1..n) sits i / n of the way; the tails near the start / end poses take those poses' ignore sets
        m = len(traj) - 1
        is_end = [dist * (m - i) / m < self.APPROACH_WAYPOINT_M - 1e-9 for i in range(m + 1)]
        is_start = [dist * i / m < self.APPROACH_WAYPOINT_M - 1e-9 for i in range(m + 1)]
        # a short segment (the §3.9 waypoint spacing itself) lies within range of both poses: those configurations take
        # both ignore sets (the grasp's support stays removed 8 mm from the grasp whichever way the segment is tracked)
        both = [i for i in range(1, m) if is_end[i] and is_start[i]]
        near_end = [i for i in range(1, m) if is_end[i] and not is_start[i]]
        near_start = [i for i in range(1, m) if is_start[i] and not is_end[i]]
        body = [i for i in range(1, m) if not is_end[i] and not is_start[i]]
        for idxs, extra, what in ((body, (), "an intermediate configuration of the approach collides"),
                                  (near_start, ignore_start, "a configuration just after the start pose collides"),
                                  (near_end, ignore_end, "a configuration just before the end pose collides"),
                                  (both, list(ignore_start) + list(ignore_end), "a configuration of the short segment collides")):
            if not idxs:
                continue
            ign = list(ignore_mid) + list(extra)
            with self.ignoring(ign):
                free = self.kit.collision_free(traj[idxs], margin=0.0)
            if not bool(free.all()):
                bad = [idxs[j] for j, f in enumerate(free.tolist()) if not f]
                return None, f"approach_untrackable: {what} ({len(bad)}/{len(idxs)}, {[round(dist * i / m, 3) for i in bad[:3]]} m from the start; {self._collision_blame(traj[bad[0]], ign)})"
        with self.ignoring(list(ignore_mid) + list(ignore_end)):
            if not bool(self.kit.collision_free(traj[-1], margin=0.0)[0]):
                return None, "approach_untrackable: the end configuration collides"
        return traj[-1].clone(), "ok"

    def _collision_blame(self, q: th.Tensor, ignored: Sequence[Any]) -> str:
        """Diagnostic: which spawned object (or 'scene / self') a colliding configuration collides with — each spawned
        object is removed in turn on top of ``ignored``."""
        world = getattr(self.kit, "world", None)
        spawned = list(getattr(world, "spawned", {}).values()) if world is not None else []
        for o in spawned:
            if any(o is x for x in ignored):
                continue
            with self.ignoring(list(ignored) + [o]):
                if bool(self.kit.collision_free(q, margin=0.0)[0]):
                    return f"collides with {getattr(o, 'name', o)}"
        try:  # world-only check: passing here means the robot (or its attached object) collides with itself
            self.kit._ensure_synced()
            with self.ignoring(list(ignored)):
                world_only = bool(self.kit.backend.check_collision(G.t(q).unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=0.0)[0])
            if not world_only:
                return "self-collision (arm / attached object against the robot's own links)"
        except Exception:  # noqa: BLE001
            pass
        return "collides with the scene (a non-spawned object)"

    def track_approach(self, q_pre: th.Tensor, pre: Pose, end: Pose, arm: str, ignore_end: Sequence[Any] = (), ignore_mid: Sequence[Any] = (),
                       q_hint: Optional[th.Tensor] = None) -> Tuple[Optional[th.Tensor], str]:
        """End-pose configuration of a straight approach, branch-consistent with the pre-pose configuration ``q_pre``:
        the segment is tracked forward from ``q_pre``; when that fails the end pose is solved on its own (32 seeds around
        ``q_hint``) and the segment tracked *backward* to the pre pose — the pre configuration is then replaced by the
        end's branch (``self._last_track_start``, recorded by the caller as the pre-pose witness). ``(None, reason)`` when
        neither branch tracks (``approach_untrackable``)."""
        self._last_track_start = G.t(q_pre).clone()
        q_end, why = self.track_segment(q_pre, pre, end, arm, ignore_mid=ignore_mid, ignore_end=ignore_end)
        if q_end is not None:
            return q_end, "ok"
        seed = q_hint if q_hint is not None else q_pre
        with self.ignoring(list(ignore_mid) + list(ignore_end)):
            q_alt = self.kit.ik(end, self._seeded(G.t(seed), end, arm), arm=arm)
        if q_alt is None:
            return None, f"{why}; the end pose has no solution on its own either"
        q_pre2, why2 = self.track_segment(q_alt, end, pre, arm, ignore_mid=ignore_mid, ignore_end=(), ignore_start=ignore_end)
        if q_pre2 is None:
            return None, f"{why}; from the end's branch: {why2}"
        self._last_track_start = q_pre2.clone()
        return q_alt, "ok"

    def exact_both_arms(self, work: Tuple[str, Sequence[Pose]], other: Tuple[str, Sequence[Pose]], stance: Stance, ignore_for_last: Sequence[Any] = (),
                        ignore_other_last: Sequence[Any] = (), fingers: Optional[float] = None,
                        brace_only_check: Optional[Tuple[Sequence[Any], float]] = None) -> IKResult:
        """Brace / second-arm rule (§4.6 step 6): the working arm's poses first; then the other arm with the working arm
        locked at its solution; combined whole-body check of the final configuration **with the objects the two final
        poses touch removed** (``ignore_for_last`` ∪ ``ignore_other_last``: the support and the pushed / braced object —
        §3.6 "ignored for the contact pose only"; with them in the world every pad-on-face contact read "combined
        configuration collides", which is why ``tabletop.brace_second_arm`` bypassed this method). ``fingers`` = gripper
        opening of both hands (closed for push + brace contacts). ``brace_only_check = (ignored objects, margin)``: an
        extra check of the brace arm's final configuration with the working arm back at the stance posture against
        everything but the ignored objects at ``margin`` (a stop blade against the target at its goal pose while the
        push hand, which touches the target, is out of the picture)."""
        arm_w, poses_w = work
        arm_o, poses_o = other
        r1 = self.exact(poses_w, stance, arm_w, ignore_for_last, cache=False, fingers=fingers)
        if not r1.ok:
            return IKResult(False, None, f"{arm_w}: {r1.reason}")
        q_seed = r1.q.clone()
        r2 = self.exact(poses_o, stance, arm_o, ignore_other_last, q_seed=q_seed, cache=False, fingers=fingers)
        if not r2.ok:
            return IKResult(False, None, f"{arm_o} (brace): {r2.reason}")
        touched = list(ignore_for_last) + [o for o in ignore_other_last if all(o is not w for w in ignore_for_last)]
        with self.ignoring(touched):
            free = bool(self.kit.collision_free(r2.q, margin=0.01)[0])
        if not free:
            return IKResult(False, None, "combined configuration collides")
        if brace_only_check is not None:
            ignored, margin = brace_only_check
            q_b = self.q_locked(stance, fingers=fingers).clone()
            idx = self.kit.robot.arm_control_idx[arm_o]
            q_b[idx] = r2.q[idx]
            with self.ignoring(list(ignored)):
                ok_b = bool(self.kit.collision_free(q_b, margin=float(margin))[0])
            if not ok_b:
                return IKResult(False, None, f"{arm_o} (brace) intersects the target at its goal pose (margin {float(margin) * 1000:.0f} mm)")
        return IKResult(True, r2.q, "", len(poses_w) + len(poses_o))

    # -- waypoints ----------------------------------------------------------------------------------------------
    def waypoints_ok(self, waypoints: Sequence[Pose], stance: Stance, arm: str = "right", policy: str = "screen_mid_exact_ends",
                     ignore_for_last: Sequence[Any] = (), ignore_all: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None,
                     scale_key: Any = None, cache: bool = True, fingers: Optional[float] = None, approached: Optional[Sequence[Any]] = None) -> IKResult:
        """§3.9: screening IK at the intermediate waypoints, exact IK at the two endpoints (``policy`` names what was
        checked and is recorded as ``waypoint_policy``). ``ignore_all`` objects are removed for every waypoint (the
        carried object's own support along a lift), ``ignore_for_last`` for the final pose only. ``cache=False`` for
        re-checks that must not hit a cached verdict (program check, ``validate_instances``)."""
        wps = list(waypoints)
        if len(wps) < 2:
            return self.exact(wps, stance, arm, ignore_for_last, q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers, approached=approached)
        if all(is_approach(a, b) for a, b in zip(wps[:-1], wps[1:])):
            # a straight same-orientation chain (approach, lift, descent, retreat, the wall brace chain): solved by
            # ``exact`` as one tracked motion from the first pose's solution (branch-consistent witnesses at every
            # waypoint; ``approach_untrackable`` otherwise) with ``ignore_all`` removed throughout
            with self.ignoring(list(ignore_all)):
                res = self.exact(wps, stance, arm, list(ignore_for_last), q_seed=q_seed, scale_key=scale_key, cache=False, fingers=fingers, approached=approached)
            return IKResult(res.ok, res.q, policy if res.ok else res.reason, len(wps))
        # endpoints first (exact): their solutions seed the screening of the intermediate waypoints, which otherwise
        # misses poses near the ends that the 32-seed solver finds (the wall brace chain)
        r0 = self.exact([wps[0]], stance, arm, list(ignore_all), q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers)
        if not r0.ok:
            return IKResult(False, None, "first waypoint fails the exact IK", len(wps))
        r1 = self.exact([wps[-1]], stance, arm, list(ignore_for_last) + list(ignore_all), q_seed=q_seed, scale_key=scale_key, cache=cache, fingers=fingers)
        if not r1.ok:
            return IKResult(False, None, "last waypoint fails the exact IK", len(wps))
        mids = wps[1:-1]
        if mids:
            ok = self.screen(mids, stance, arm, ignore=list(ignore_all), q_seed=q_seed, fingers=fingers, seeds=[r0.q, r1.q])
            if not bool(ok.all()):
                bad = [i + 1 for i, s in enumerate(ok.tolist()) if not s]
                return IKResult(False, None, f"waypoints {bad} of {len(wps)} fail the screening IK", len(wps))
        return IKResult(True, r1.q, policy, len(wps))

    # -- attached objects ---------------------------------------------------------------------------------------
    def attachable(self, obj, eef_in_object: Pose, arm: str = "right") -> bool:
        """Whether the object, attached to ``arm`` per the grasp transform ``T_obj→eef``, is free of **self**-collision
        with the robot in an untucked free-space configuration (the attached spheres vs the palm / wrist, which the
        robot model does not exempt — only the finger links are). A grasp that fails here fails every attached check
        (lift, carry, place) whatever the pose, so task modules drop it from the grasp set once per template."""
        from ispatialgym.sampling.v2.navigation import attached, attached_checks_enabled

        if not attached_checks_enabled():
            return True
        q = self.kit.q_at_base_pose(self.kit.q_config(arms="untucked", translate=0.5), *self.kit.world_base_pose2d())
        with attached(self.kit, arm, obj, eef_in_object=eef_in_object):
            self.kit._ensure_synced()
            with_self = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=True, margin=0.01)[0])
            without = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, self_collision=False, margin=0.01)[0])
        return not (with_self and not without)

    # -- gripper visibility -------------------------------------------------------------------------------------
    def eef_in_frame(self, pose: Pose, camera_pose: Pose, margin: float = IN_FRAME_MARGIN, eef_extent: float = 0.06) -> bool:
        """§3.9 gripper in FOV: the eef body (a box of half-extent ``eef_extent`` around the eef origin, fingertips
        along +a) projects inside the image of ``camera_pose`` with ``margin`` (fraction of the image size)."""
        p, q = G.t(pose[0]), G.t(pose[1])
        pts = []
        for dx in (-eef_extent, eef_extent):
            for dy in (-eef_extent, eef_extent):
                for dz in (-0.02, 0.06):
                    pts.append(p + G.rotate(q, th.tensor([dx, dy, dz])))
        return self.kit.world.camera_model.all_inside(camera_pose, th.stack(pts), margin)

    def camera_pose(self, stance: Stance) -> Pose:
        """Head-camera pose at a stance through the shared FK (``conventions.camera_pose``, base on the measured floor, the
        kit's measured sensor offset) — the same model the oracle's FOV monitor projects with."""
        q = self.q_locked(stance)
        x, y, yaw = (float(v) for v in stance.base_pose2d)
        base = (th.tensor([x, y, float(self.kit.floor_z) + C.NOMINAL_BASE_Z]), G.quat_from_yaw(yaw))
        return C.camera_pose(q, base_pose=base, cam_in_zed=self.kit.cam_in_zed)

    # -- bands: feasible sets ------------------------------------------------------------------------------------
    def feasible_sets(self, cells: Sequence[Any], yaws: Sequence[float], grasps: Sequence[Any], pose_at: Callable[[Any, float], Pose], stance: Stance,
                      arm: str = "right", ignore_for_grasp: Sequence[Any] = (), eef_of_grasp: Optional[Callable[[Any], Pose]] = None,
                      scale_key: Any = None, cache: bool = True) -> Dict[int, List[Tuple[float, Any]]]:
        """§4.6 step 1 over many cells in one batch per trunk: for every cell index the feasible set F of ``(yaw, grasp)``
        pairs whose [pre-grasp, grasp] pass the screening IK from ``stance`` (grasp pose with ``ignore_for_grasp``
        removed). ``pose_at(cell, yaw) -> object pose``; ``eef_of_grasp(grasp) -> T_obj→eef`` (default ``grasp.pose``).
        Cached (§3.10 band cache of F) under the stance, cells, yaws, grasp set, object poses at yaw 0 and ``scale_key``;
        a cell with empty F is absent from the result (the caller removes it from the band)."""
        eef_of = eef_of_grasp or (lambda g: g.pose)
        by_name = {getattr(g, "name", str(k)): g for k, g in enumerate(grasps)}
        key = None
        if cache:
            key = cache_key("feasible_sets_v2", {**self.cache_scope, "base": list(stance.base_pose2d), "trunk": [stance.translate, stance.tilt], "arm": arm,
                                                 "cells": [list(G.t(c).tolist()) for c in cells], "yaws": list(yaws), "scale": scale_key,
                                                 "grasps": [(getattr(g, "key", None), getattr(g, "name", None), list(G.t(eef_of(g)[0]).tolist()), list(G.t(eef_of(g)[1]).tolist())) for g in grasps],
                                                 "poses0": [list(G.t(pose_at(c, 0.0)[0]).tolist()) + list(G.t(pose_at(c, 0.0)[1]).tolist()) for c in cells],
                                                 "ignore": sorted(getattr(o, "name", str(o)) for o in ignore_for_grasp), "seeds": SCREEN_SEEDS})
            hit = cache_get(key)
            if hit is not None:
                return {int(i): [(float(yaw), by_name[gn]) for yaw, gn in lst if gn in by_name] for i, lst in hit.items()}
        pairs, index = [], []
        for i, cell in enumerate(cells):
            for yaw in yaws:
                obj_pose = pose_at(cell, yaw)
                for g in grasps:
                    eef = grasp_pose(obj_pose, eef_of(g))
                    pairs.append([pre_pose(eef), eef])
                    index.append((i, yaw, g))
        feas = self.screen_pairs(pairs, stance, arm, ignore_for_last=ignore_for_grasp)
        out: Dict[int, List[Tuple[float, Any]]] = {}
        for (i, yaw, g), s in zip(index, feas.tolist()):
            if s:
                out.setdefault(i, []).append((yaw, g))
        if key is not None:
            cache_put(key, {str(i): [(yaw, getattr(g, "name", None)) for yaw, g in lst] for i, lst in out.items()})
        return out
