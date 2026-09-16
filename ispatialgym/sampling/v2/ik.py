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
        for o in self.objs:
            for link in getattr(o, "links", {}).values():
                for cm in getattr(link, "collision_meshes", {}).values():
                    name = cm.prim_path
                    if name in backend._world_mesh_names and name not in disabled:
                        backend._mesh_enable(name, False)
                        self.names.append(name)
        return self

    def __exit__(self, *exc):
        if self._fallback is not None:
            self._fallback.__exit__(*exc)
            return False
        for name in self.names:
            self.kit.backend._mesh_enable(name, True)
        self.names = []
        return False


@dataclass
class IKResult:
    ok: bool
    q: Optional[th.Tensor] = None
    reason: str = ""
    n_checked: int = 0


class IKChecker:
    """§3.6 / §3.9 IK checks over a ``RobotKit``. Stateless apart from the kit and the cache namespace."""

    def __init__(self, kit, cache_scope: Optional[Dict[str, Any]] = None):
        self.kit = kit
        self.cache_scope = dict(cache_scope or {})  # scene / rooms / placed objects: part of every cache key
        #: (arm, rounded pose) -> 28-D solution of a successful exact solve; cuRobo's remaining seeds are random, so a
        #: marginal pose can pass once and fail when re-solved (the program check re-solves every recorded pose): the
        #: sampler's own solution is used as the seed configuration then. Cleared per attempt (``core``).
        self.solutions: Dict[Tuple[str, Tuple[float, ...]], th.Tensor] = {}

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

    def _seeded(self, q0: th.Tensor, pose: Pose, arm: str) -> th.Tensor:
        """``q0`` with the working arm set to the remembered solution of the nearest recorded pose of ``arm`` within
        (3 cm, 0.15 rad) — the same pose, or the same pose after the settle moved the object by a few mm (the other
        joints stay ``q0``'s: the base virtual joints must belong to the current robot root)."""
        p, q = G.t(pose[0]), G.t(pose[1])
        best, best_d = None, None
        for (a, key), q_rec in self.solutions.items():
            if a != arm:
                continue
            kp, kq = th.tensor(key[:3]), th.tensor(key[3:])
            d = float(th.linalg.norm(kp - p))
            if d > self.SEED_POS_TOL or G.quat_angle(kq, q) > self.SEED_ANG_TOL:
                continue
            if best_d is None or d < best_d:
                best, best_d = q_rec, d
        if best is None:
            return q0
        out = q0.clone()
        idx = self.kit.robot.arm_control_idx[arm]
        out[idx] = best[idx]
        return out

    # -- configurations -----------------------------------------------------------------------------------------
    def q_locked(self, stance: Stance, arms: str = "untucked", right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None) -> th.Tensor:
        """28-D configuration at the stance (base pose + trunk), arms untucked (or the given arm vectors), grippers open —
        the locked-joint source and IK seed of every check from this stance."""
        q = self.kit.q_config(arms=arms, translate=stance.translate, tilt=stance.tilt, right_arm=right_arm, left_arm=left_arm)
        return self.kit.q_at_base_pose(q, *stance.base_pose2d)

    def stance_collision_free(self, stance: Stance, margin: float = 0.01, arms: str = "untucked", **arm_q) -> bool:
        return bool(self.kit.collision_free(self.q_locked(stance, arms, **arm_q), margin=margin)[0])

    def stances_collision_free(self, stances: Sequence[Stance], margin: float = 0.01) -> th.Tensor:
        if not stances:
            return th.zeros(0, dtype=th.bool)
        return self.kit.collision_free(th.stack([self.q_locked(s) for s in stances]), margin=margin)

    # -- screening / exact --------------------------------------------------------------------------------------
    def screen(self, poses: Sequence[Pose], stance: Stance, arm: str = "right", ignore: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None) -> th.Tensor:
        """12-seed batched IK per pose from the stance (locked joints = untucked at the stance trunk); ``ignore``
        objects are removed from the world for *these* poses (callers pass the support for grasp poses only)."""
        if len(poses) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance) if q_seed is None else q_seed
        with self.ignoring(ignore):
            return self.kit.ik_batch(list(poses), q0, arm=arm)

    def screen_pairs(self, pairs: Sequence[Sequence[Pose]], stance: Stance, arm: str = "right", ignore_for_last: Sequence[Any] = (),
                     q_seed: Optional[th.Tensor] = None) -> th.Tensor:
        """Batched screening of [pre, final] pose pairs: pre against the full world, final with ``ignore_for_last`` removed."""
        if len(pairs) == 0:
            return th.zeros(0, dtype=th.bool)
        q0 = self.q_locked(stance) if q_seed is None else q_seed
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
              q_seed: Optional[th.Tensor] = None, scale_key: Any = None, cache: bool = True) -> IKResult:
        """32-seed IK of every pose in order from the stance; the last pose with ``ignore_for_last`` removed (the support
        under a grasp). Returns the configuration at the last pose. ``scale_key`` (the spawn scale vector of the object
        the poses belong to) is part of the cache key (§3.10)."""
        if len(poses) == 0:
            return IKResult(True, None, "no poses", 0)
        q0 = self.q_locked(stance) if q_seed is None else q_seed
        if q_seed is None:
            q0 = self._seeded(q0, poses[-1], arm)
        key = None
        if cache:
            key = cache_key("exact_ik_v2", {**self.cache_scope, "base": list(stance.base_pose2d), "trunk": [stance.translate, stance.tilt], "arm": arm,
                                            "poses": [list(G.t(p[0]).tolist()) + list(G.t(p[1]).tolist()) for p in poses],
                                            "ignore": sorted(getattr(o, "name", str(o)) for o in ignore_for_last), "scale": scale_key,
                                            "seeds": EXACT_SEEDS, "seed_arms": q0[self.kit.robot.arm_control_idx[arm]]})
            hit = cache_get(key)
            if hit is not None:
                q_hit = None if hit.get("q") is None else th.tensor(hit["q"])
                if q_hit is not None:
                    self.solutions[(arm, self._pose_key(poses[-1]))] = q_hit.clone()
                return IKResult(bool(hit["ok"]), q_hit, "cache", len(poses))
        q = None
        n = len(poses)
        for i, p in enumerate(poses):
            if i == n - 1 and ignore_for_last:
                with self.ignoring(ignore_for_last):
                    q = self.kit.ik(p, q0, arm=arm)
            else:
                q = self.kit.ik(p, q0, arm=arm)
            if q is None:
                break
        res = IKResult(q is not None, q, "" if q is not None else "exact IK failed", len(poses))
        if q is not None:
            self.solutions[(arm, self._pose_key(poses[-1]))] = q.clone()
        if key is not None:
            cache_put(key, {"ok": res.ok, "q": None if q is None else q.tolist()})
        return res

    def exact_both_arms(self, work: Tuple[str, Sequence[Pose]], other: Tuple[str, Sequence[Pose]], stance: Stance, ignore_for_last: Sequence[Any] = (),
                        ignore_other_last: Sequence[Any] = ()) -> IKResult:
        """Brace / second-arm rule (§4.6 step 6): the working arm's poses first; then the other arm with the working arm
        locked at its solution; combined whole-body check of the final configuration."""
        arm_w, poses_w = work
        arm_o, poses_o = other
        r1 = self.exact(poses_w, stance, arm_w, ignore_for_last, cache=False)
        if not r1.ok:
            return IKResult(False, None, f"{arm_w}: {r1.reason}")
        q_seed = r1.q.clone()
        r2 = self.exact(poses_o, stance, arm_o, ignore_other_last, q_seed=q_seed, cache=False)
        if not r2.ok:
            return IKResult(False, None, f"{arm_o} (brace): {r2.reason}")
        if not bool(self.kit.collision_free(r2.q, margin=0.01)[0]):
            return IKResult(False, None, "combined configuration collides")
        return IKResult(True, r2.q, "", len(poses_w) + len(poses_o))

    # -- waypoints ----------------------------------------------------------------------------------------------
    def waypoints_ok(self, waypoints: Sequence[Pose], stance: Stance, arm: str = "right", policy: str = "screen_mid_exact_ends",
                     ignore_for_last: Sequence[Any] = (), ignore_all: Sequence[Any] = (), q_seed: Optional[th.Tensor] = None,
                     scale_key: Any = None) -> IKResult:
        """§3.9: screening IK at the intermediate waypoints, exact IK at the two endpoints (``policy`` names what was
        checked and is recorded as ``waypoint_policy``). ``ignore_all`` objects are removed for every waypoint (the
        carried object's own support along a lift), ``ignore_for_last`` for the final pose only."""
        wps = list(waypoints)
        if len(wps) < 2:
            return self.exact(wps, stance, arm, ignore_for_last, q_seed=q_seed, scale_key=scale_key)
        mids = wps[1:-1]
        if mids:
            ok = self.screen(mids, stance, arm, ignore=list(ignore_all), q_seed=q_seed)
            if not bool(ok.all()):
                bad = [i + 1 for i, s in enumerate(ok.tolist()) if not s]
                return IKResult(False, None, f"waypoints {bad} of {len(wps)} fail the screening IK", len(wps))
        r0 = self.exact([wps[0]], stance, arm, list(ignore_all), q_seed=q_seed, scale_key=scale_key)
        if not r0.ok:
            return IKResult(False, None, "first waypoint fails the exact IK", len(wps))
        r1 = self.exact([wps[-1]], stance, arm, list(ignore_for_last) + list(ignore_all), q_seed=q_seed, scale_key=scale_key)
        if not r1.ok:
            return IKResult(False, None, "last waypoint fails the exact IK", len(wps))
        return IKResult(True, r1.q, policy, len(wps))

    # -- attached objects ---------------------------------------------------------------------------------------
    def attachable(self, obj, eef_in_object: Pose, arm: str = "right") -> bool:
        """Whether the object, attached to ``arm`` per the grasp transform ``T_obj→eef``, is free of **self**-collision
        with the robot in an untucked free-space configuration (the attached spheres vs the palm / wrist, which the
        robot model does not exempt — only the finger links are). A grasp that fails here fails every attached check
        (lift, carry, place) whatever the pose, so task modules drop it from the grasp set once per template."""
        from ispatialgym.sampling.v2.navigation import attached

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
        return self.kit.camera_pose(self.q_locked(stance))

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
