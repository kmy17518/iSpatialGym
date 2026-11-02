"""R1Pro kinematics / reachability / viewing helpers for the sampler, built on the oracle's cuRobo backend.

Everything here reasons about *hypothetical* robot configurations: the robot stays parked while candidate
base poses are expressed through the virtual base joints (``base_footprint_{x,y,rz}_joint``) in the root
frame, exactly like ``Skills.q_with_base_pose``. Only the chosen configuration is teleported.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: F401  (must precede omnigibson import)

import omnigibson.utils.transform_utils as T  # noqa: E402

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.robot.conventions import CONTACT_CENTER_Z, FINGER_OPEN, NOMINAL_BASE_Z, camera_offset_in_zed, q_with_base_pose, torso_q, torso_q_to_translate_tilt  # noqa: E402,F401  (moved to robot/conventions, F5)
from ispatialgym.sampling import geometry as G  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


@dataclass
class ViewSolution:
    translate: float
    tilt: float
    angle: float  # angle between the optical axis and the ray to the target (rad)
    cam_pose: G.Pose
    q: th.Tensor


class RobotKit:
    def __init__(self, world, prebuild: Sequence[str] = ("right_arm",), mesh_cache_size: Optional[int] = None, sdf_grid: Optional[Dict[str, object]] = None):
        """``mesh_cache_size``: capacity of the cuRobo mesh store (default: the backend's 8192). The store's capacity, not
        its fill, sizes every collision kernel launch, so a caller that knows how many collision meshes its world holds
        (``count_collision_meshes``) should pass a snug value - see ``R1ProCuRobo.__init__``. ``sdf_grid``: the opt-in
        voxel-grid collision world (``R1ProCuRobo(sdf_grid=...)``; ``reach_bands --collision sdf-grid``)."""
        self.world = world
        self.robot = world.robot
        kw = {"mesh_cache_size": int(mesh_cache_size)} if mesh_cache_size else {}
        if sdf_grid:
            kw["sdf_grid"] = sdf_grid
        self.backend = R1ProCuRobo(self.robot, prebuild=prebuild, **kw)
        if hasattr(world, "obstacle_filter"):
            self.backend.obstacle_filter = world.obstacle_filter
        self.jidx = self.backend._jidx
        # a camera-less world (SimWorld(cameras=False)) has no sensor to measure: the USD constant (camera_pose / solve_view are not used there)
        self.cam_in_zed = camera_offset_in_zed(self.robot, world.cam_key) if getattr(world, "cam_key", None) is not None else camera_offset_in_zed(None)
        idx = self.robot.trunk_control_idx
        self.trunk_limits = (self.robot.joint_lower_limits[idx].clone(), self.robot.joint_upper_limits[idx].clone())
        self._trav_cache: Dict[int, th.Tensor] = {}
        self.floor_z = float(getattr(world, "floor_z", 0.0))
        #: call / time counters for the expensive queries (reported per template in template_meta.timings.kit)
        self.stats: Dict[str, float] = {"ik_calls": 0, "ik_s": 0.0, "ik_batch_calls": 0, "ik_batch_problems": 0, "ik_batch_s": 0.0,
                                        "collision_calls": 0, "collision_configs": 0, "collision_s": 0.0,
                                        # train-fast 2026-10-31: the formerly uninstrumented costs - differential IK (``ik_path``: approach / chain
                                        # tracking, the witness polish), world syncs (``sync_world`` / ``ignoring``; ``sync_full_calls`` = passes over
                                        # every mesh), attachments (``attach_object``; ``fit_cache_hits`` = SEEDED_SPHERE_FIT=cache hits) and the
                                        # return-all IK solves (``ik_all``; counted in ``ik_calls`` / ``ik_s`` as well)
                                        "ik_path_calls": 0, "ik_path_s": 0.0, "ik_path_waypoints": 0, "sync_calls": 0, "sync_s": 0.0, "sync_full_calls": 0,
                                        "attach_calls": 0, "attach_s": 0.0, "fit_cache_hits": 0, "ik_all_calls": 0}

    def stats_snapshot(self) -> Dict[str, float]:
        return dict(self.stats)

    def _bump(self, key: str, n: float = 1) -> None:
        """``stats[key] += n`` (the key created when a subclass built its own ``stats``; train-fast 2026-10-31)."""
        self.stats[key] = self.stats.get(key, 0) + n

    @staticmethod
    def count_collision_meshes(world) -> int:
        """Number of collision meshes ``R1ProCuRobo.update_world`` would load from ``world`` now (scene objects passing the
        obstacle filter, spawned objects; the robot and visual-only objects excluded) - what the cuRobo mesh cache must hold."""
        n = 0
        for o in world.scene.objects:
            if o is world.robot or getattr(o, "visual_only", False) or (hasattr(world, "obstacle_filter") and not world.obstacle_filter(o)):
                continue
            n += sum(len(link.collision_meshes) for link in o.links.values())
        return n

    @staticmethod
    def stats_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        return {k: round(after[k] - before.get(k, 0), 3) for k in after}

    # ------------------------------------------------------------------ configurations

    def arm_q(self, mode: str, arm: str) -> th.Tensor:
        if mode == "untucked":
            return self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[arm]].clone()
        if mode == "tucked":
            return th.zeros(7)
        if mode == "ready":
            return th.tensor(M.ARM_READY_Q[arm], dtype=th.float32)
        raise ValueError(mode)

    def q_config(self, arms: str = "untucked", translate: Optional[float] = None, tilt: float = 0.0, torso_zero: bool = False,
                 right_arm: Optional[th.Tensor] = None, left_arm: Optional[th.Tensor] = None, fingers: float = FINGER_OPEN) -> th.Tensor:
        """28-D configuration; the virtual base joints place base_link upright at the nominal standing height above
        the floor (``floor_z + NOMINAL_BASE_Z``) so that collision checks do not depend on how far the simulated
        robot has currently sunk into or hovers above the floor. ``q_at_base_pose`` then sets x / y / yaw."""
        q = th.zeros(28)
        rp, _ = self.backend.root_pose()
        q[self.jidx["base_footprint_z_joint"]] = self.floor_z + NOMINAL_BASE_Z - float(rp[2])
        if not torso_zero:
            q[self.robot.trunk_control_idx] = torso_q(M.DEFAULT_TRUNK_TRANSLATE if translate is None else translate, tilt, self.trunk_limits)
        for arm in M.ARMS:
            q[self.robot.arm_control_idx[arm]] = self.arm_q(arms, arm)
            q[self.robot.gripper_control_idx[arm]] = fingers
        if right_arm is not None:
            q[self.robot.arm_control_idx["right"]] = G.t(right_arm)
        if left_arm is not None:
            q[self.robot.arm_control_idx["left"]] = G.t(left_arm)
        return q

    def with_trunk(self, q: th.Tensor, translate: float, tilt: float) -> th.Tensor:
        qk = q.clone()
        qk[self.robot.trunk_control_idx] = torso_q(translate, tilt, self.trunk_limits)
        return qk

    def trunk_of(self, q: th.Tensor) -> Tuple[float, float]:
        return torso_q_to_translate_tilt(q[self.robot.trunk_control_idx])

    def q_at_base_pose(self, q: th.Tensor, x: float, y: float, yaw: float) -> th.Tensor:
        """Copy of ``q`` whose virtual base joints put base_link at world (x, y, yaw) given the current root pose.

        The articulation root stays where the robot was spawned (possibly metres above the floor); the z / roll / pitch
        virtual joints always put base_link upright at the nominal standing height, whatever ``q`` carried."""
        rp, rq = self.backend.root_pose()
        return q_with_base_pose(q, x, y, yaw, root_pose2d=(float(rp[0]), float(rp[1]), G.yaw_of(rq)), base_z=self.floor_z + NOMINAL_BASE_Z,
                                root_z=float(rp[2]), joint_index=self.jidx)

    def q_current(self) -> th.Tensor:
        return self.backend.q_full()

    def sync_world(self, ignore_objects=(), only_objects=None):
        """``only_objects``: refresh those objects' meshes alone (cheap; the rest of the world has not moved) - a full
        sync when the world is not loaded yet or the robot root drifted since the last full sync."""
        t0 = time.time()  # train-fast 2026-10-31: timed (sync_calls / sync_s; sync_full_calls = the passes over every mesh)
        self._bump("sync_calls")
        try:
            if only_objects is not None and getattr(self.backend, "_world_loaded", False) and getattr(self, "_synced_root", None) is not None \
                    and float(th.linalg.norm(self.backend.root_pose()[0] - self._synced_root)) <= 0.01:
                self.backend.update_world(ignore_objects=ignore_objects, only_objects=only_objects)
                return
            self._bump("sync_full_calls")
            self.backend.update_world(ignore_objects=ignore_objects)
            self._synced_root = self.backend.root_pose()[0].clone()
        finally:
            self._bump("sync_s", time.time() - t0)

    def _ensure_synced(self):
        """cuRobo stores obstacles relative to the robot root at sync time; if the simulated root moved since (e.g. the
        parked robot sank), every query would see a shifted world. Re-sync when the drift exceeds 1 cm."""
        root = self.backend.root_pose()[0]
        prev = getattr(self, "_synced_root", None)
        if prev is None or float(th.linalg.norm(root - prev)) > 0.01:
            if prev is not None:
                log.warning("robot root moved %.3f m since the last cuRobo sync; re-syncing", float(th.linalg.norm(root - prev)))
            self.sync_world()

    # ------------------------------------------------------------------ kinematics

    def fk(self, frames: Sequence[str], q: th.Tensor) -> Dict[str, G.Pose]:
        return self.backend.fk(list(frames), q_full=q, emb_name="whole_body")

    def camera_pose(self, q: th.Tensor) -> G.Pose:
        zed = self.fk([M.HEAD_CAMERA_LINK], q)[M.HEAD_CAMERA_LINK]
        return G.compose(zed, self.cam_in_zed)

    def camera_poses_batch(self, qs: th.Tensor) -> List[G.Pose]:
        pos, quat = self.backend.fk_batch(M.HEAD_CAMERA_LINK, qs, emb_name="whole_body")
        return [G.compose((pos[i], quat[i]), self.cam_in_zed) for i in range(pos.shape[0])]

    def eef_pose(self, q: th.Tensor, arm: str = "right") -> G.Pose:
        return self.fk([M.EEF_LINK[arm]], q)[M.EEF_LINK[arm]]

    def pad_center(self, eef_pose: G.Pose) -> th.Tensor:
        return G.t(eef_pose[0]) + G.rotate(eef_pose[1], th.tensor([0.0, 0.0, CONTACT_CENTER_Z]))

    # ------------------------------------------------------------------ viewing

    def solve_view(self, q_base: th.Tensor, target, translates=None, tilts=None, prefer_translate: float = M.DEFAULT_TRUNK_TRANSLATE,
                   top_k: int = 1, max_angle: float = math.radians(8.0)) -> List[ViewSolution]:
        """Trunk (translate, tilt) candidates whose optical axis points at ``target`` (world) from the base pose encoded in
        ``q_base``, best first (axis angle plus a small preference toward ``prefer_translate``). Only candidates within
        ``max_angle`` of the target are returned (at least one)."""
        translates = th.linspace(0.0, 1.5, 31) if translates is None else G.t(translates)
        tilts = th.linspace(-0.9, 0.9, 37) if tilts is None else G.t(tilts)
        qs, keys = [], []
        for tr in translates.tolist():
            for ti in tilts.tolist():
                qs.append(self.with_trunk(q_base, tr, ti))
                keys.append((tr, ti))
        cams = self.camera_poses_batch(th.stack(qs))
        tgt = G.t(target)
        scored = []
        for (tr, ti), cam, qk in zip(keys, cams, qs):
            ang = self.world.camera_model.angle_off_axis(cam, tgt)
            scored.append((ang + 0.02 * abs(tr - prefer_translate), ViewSolution(tr, ti, ang, cam, qk)))
        scored.sort(key=lambda s: s[0])
        out = [v for _, v in scored if v.angle <= max_angle][:top_k]
        return out or [scored[0][1]]

    # ------------------------------------------------------------------ collision

    def links_clear(self, qs: th.Tensor, links: Sequence[str], margin: float) -> th.Tensor:
        """``(N,)`` bool: True where none of ``links``' spheres lies within ``margin`` of a world obstacle (ignored objects excluded)."""
        qs = G.t(qs)
        if qs.ndim == 1:
            qs = qs.unsqueeze(0)
        self._ensure_synced()
        return ~self.backend.link_proximity(qs, list(links), float(margin), emb_name="whole_body", sync_world=False)

    def collision_free(self, qs: th.Tensor, margin: float = 0.01, sync: bool = False) -> th.Tensor:
        qs = G.t(qs)
        if qs.ndim == 1:
            qs = qs.unsqueeze(0)
        if not sync:
            self._ensure_synced()
        t0 = time.time()
        out = ~self.backend.check_collision(qs, emb_name="whole_body", sync_world=sync, margin=margin)
        self.stats["collision_calls"] += 1
        self.stats["collision_configs"] += int(qs.shape[0])
        self.stats["collision_s"] += time.time() - t0
        return out

    # ------------------------------------------------------------------ IK

    def ik(self, eef_pose: G.Pose, q_seed: th.Tensor, arm: str = "right", torso: bool = False) -> Optional[th.Tensor]:
        emb = f"{arm}_arm_torso" if torso else f"{arm}_arm"
        self._ensure_synced()
        t0 = time.time()
        out = self.backend.solve_ik(emb, {M.EEF_LINK[arm]: (G.t(eef_pose[0]), G.t(eef_pose[1]))}, q_seed=q_seed, sync_world=False)
        self.stats["ik_calls"] += 1
        self.stats["ik_s"] += time.time() - t0
        return out

    def ik_all(self, eef_pose: G.Pose, q_seed: th.Tensor, arm: str = "right", torso: bool = False) -> Optional[th.Tensor]:
        """Every successful solution of one 32-seed solve, ``(K, 28)`` (locked joints from ``q_seed``), or None - the same solve as
        :meth:`ik` (``backend.solve_ik(return_all=True)``), which keeps only the solution nearest to the seed; the caller picks
        (``IKChecker`` prefers the one clear of the joint limits). Timed like :meth:`ik` (``ik_calls`` / ``ik_s``) plus ``ik_all_calls``.
        Train-fast 2026-10-31 (LIMIT_RETRY_RETURN_ALL)."""
        emb = f"{arm}_arm_torso" if torso else f"{arm}_arm"
        self._ensure_synced()
        t0 = time.time()
        out = self.backend.solve_ik(emb, {M.EEF_LINK[arm]: (G.t(eef_pose[0]), G.t(eef_pose[1]))}, q_seed=q_seed, sync_world=False, return_all=True)
        self.stats["ik_calls"] += 1
        self.stats["ik_s"] += time.time() - t0
        self._bump("ik_all_calls")
        if out is None:
            return None
        return out.reshape(-1, out.shape[-1]) if out.ndim > 1 else out.unsqueeze(0)

    def ik_path(self, emb: str, waypoints: Sequence[Dict[str, G.Pose]], q_start: Optional[th.Tensor] = None, **kw):
        """``backend.solve_ik_path(emb, waypoints, q_start=q_start, **kw)`` (differential IK along dense waypoints: the approach /
        chain tracking and the witness polish of ``IKChecker``), timed into ``ik_path_calls`` / ``ik_path_s`` / ``ik_path_waypoints``
        (train-fast 2026-10-31: the largest formerly uninstrumented cost). The world is NOT re-synced here (the callers hold their
        ignore contexts); pass ``sync_world`` explicitly to change that."""
        t0 = time.time()
        try:
            return self.backend.solve_ik_path(emb, waypoints, q_start=q_start, **kw)
        finally:
            self._bump("ik_path_calls")
            self._bump("ik_path_waypoints", len(waypoints))
            self._bump("ik_path_s", time.time() - t0)

    # ------------------------------------------------------------------ attachment (timed; train-fast 2026-10-31)

    def attach_object(self, arm: str, obj, **kw) -> None:
        """``backend.attach_object(arm, obj, **kw)`` timed into ``attach_calls`` / ``attach_s``; ``fit_cache_hits`` counts the sphere fits
        served from the SEEDED_SPHERE_FIT=cache store (0 unless that env is set)."""
        from ispatialgym.oracle import curobo_planner as CP

        hits0 = int(getattr(CP, "_FIT_COUNTS", {}).get("hits", 0))
        t0 = time.time()
        try:
            self.backend.attach_object(arm, obj, **kw)
        finally:
            self._bump("attach_calls")
            self._bump("attach_s", time.time() - t0)
            self._bump("fit_cache_hits", int(getattr(CP, "_FIT_COUNTS", {}).get("hits", 0)) - hits0)

    def detach_object(self, arm: str) -> None:
        self.backend.detach_object(arm)

    def reachable(self, eef_poses: Sequence[G.Pose], q_seed: th.Tensor, arm: str = "right", torso: bool = False, ignore_for_last=()) -> Optional[th.Tensor]:
        """Configuration reaching the *last* pose if every pose in the list is IK-feasible from ``q_seed``, else None.
        ``ignore_for_last``: objects ignored (collision) for the last pose only (the support under a grasp)."""
        q = None
        n = len(eef_poses)
        for i, p in enumerate(eef_poses):
            if i == n - 1 and ignore_for_last:
                with self.ignoring(ignore_for_last):
                    q = self.ik(p, q_seed, arm=arm, torso=torso)
            else:
                q = self.ik(p, q_seed, arm=arm, torso=torso)
            if q is None:
                return None
        return q

    def reachable_any_translate(self, eef_poses: Sequence[G.Pose], q_stance: th.Tensor, translates: Sequence[float], arm: str = "right") -> Optional[float]:
        """First trunk translate (tilt 0) in ``translates`` from which every pose is reachable (arm only)."""
        for tr in translates:
            q = self.with_trunk(q_stance, float(tr), 0.0)
            if self.reachable(eef_poses, q, arm=arm) is not None:
                return float(tr)
        return None

    # ------------------------------------------------------------------ grasp poses

    @staticmethod
    def top_down_quat(yaw: float, tilt_deg: float = 0.0, toward=None) -> th.Tensor:
        """Approach axis (eef +z) pointing down, pitched ``tilt_deg`` toward the unit xy direction ``toward``; wrist yaw."""
        z = th.tensor([0.0, 0.0, -1.0])
        if tilt_deg and toward is not None:
            tw = G.t([float(toward[0]), float(toward[1]), 0.0])
            tw = tw / th.linalg.norm(tw)
            tr = math.radians(tilt_deg)
            z = z * math.cos(tr) + tw * math.sin(tr)
        x_ref = th.tensor([math.cos(yaw), math.sin(yaw), 0.0])
        x = x_ref - th.dot(x_ref, z) * z
        x = x / th.linalg.norm(x)
        y = th.cross(z, x, dim=0)
        return T.mat2quat(th.stack([x, y, z], dim=1))

    def grasp_candidates_at(self, point, yaw_choices: Sequence[float], tilts: Sequence[float] = (0.0, 30.0), toward=None, pregrasp: float = 0.10) -> List[List[G.Pose]]:
        """[pre-grasp, grasp] eef pose pairs above ``point`` for several wrist yaws / approach tilts."""
        out = []
        p = G.t(point)
        for tilt in tilts:
            for yaw in yaw_choices:
                q = self.top_down_quat(yaw, tilt, toward)
                axis = G.rotate(q, th.tensor([0.0, 0.0, 1.0]))
                out.append([(p - axis * pregrasp, q), (p, q)])
        return out

    def ik_batch(self, poses: Sequence[G.Pose], q_seed: th.Tensor, arm: str = "right", batch_size: int = 64, ignore=(), torso: bool = False,
                 seeds: Optional[int] = None) -> th.Tensor:
        """Screening IK for many eef poses at once (fewer seeds than :meth:`ik`); returns a bool mask.

        ``ignore``: objects whose collision meshes are disabled for these queries (e.g. the support surface for the
        final grasp pose, which the oracle approaches without collision checking); re-enabled afterwards.
        ``torso``: use the arm + torso embodiment (single tool frame as well, so batching applies).
        ``seeds`` (train-fast 2026-10-31): seeds per pose of the batched solver (default: the backend's ``batch_ik_seeds``, 12).
        """
        return self.ik_batch_solve(poses, q_seed, arm=arm, batch_size=batch_size, ignore=ignore, torso=torso, seeds=seeds)[0]

    def ik_batch_solve(self, poses: Sequence[G.Pose], q_seed: th.Tensor, arm: str = "right", batch_size: int = 64, ignore=(),
                       torso: bool = False, seeds: Optional[int] = None) -> Tuple[th.Tensor, th.Tensor]:
        """Like :meth:`ik_batch` but also returns the solutions ``(success (N,), q (N, 28))`` (locked joints from
        ``q_seed``). The screening solutions are collision-checked by the same checker as the exact solver, so callers
        that need a configuration (not just feasibility) can use them directly and skip the exact solve.

        Note: every call with a *different* set of locked-joint values (trunk configuration) re-derives the fixed
        transforms of the batched solver, which costs a few seconds; group queries by trunk configuration.
        ``seeds`` (train-fast 2026-10-31): seeds per pose (default: the backend's ``batch_ik_seeds``); a solver is cached per
        (embodiment, batch size, seeds), so a small anchor batch may ask for 32 seeds without touching the screening solver."""
        if len(poses) == 0:
            return th.zeros(0, dtype=th.bool), th.zeros(0, 28)
        self._ensure_synced()
        t0 = time.time()
        emb = f"{arm}_arm_torso" if torso else f"{arm}_arm"
        with self.ignoring(ignore):
            succ, q_out = self.backend.solve_ik_batch(emb, M.EEF_LINK[arm], list(poses), q_seed, batch_size=batch_size,
                                                      **({"seeds": int(seeds)} if seeds is not None else {}))
        self.stats["ik_batch_calls"] += 1
        self.stats["ik_batch_problems"] += len(poses)
        self.stats["ik_batch_s"] += time.time() - t0
        return succ, q_out

    def ignoring(self, objs):
        """Context manager: temporarily drop ``objs`` from the cuRobo collision world."""
        kit = self

        class _Ctx:
            def __enter__(self_inner):
                if objs:
                    self_inner._timed(ignore_objects=list(objs))

            def __exit__(self_inner, *exc):
                if objs:
                    self_inner._timed()

            @staticmethod
            def _timed(**kw):
                # train-fast 2026-10-31: both are full passes over every mesh (sync_calls / sync_full_calls / sync_s)
                t0 = time.time()
                try:
                    kit.backend.update_world(**kw)
                finally:
                    kit._bump("sync_calls")
                    kit._bump("sync_full_calls")
                    kit._bump("sync_s", time.time() - t0)

        return _Ctx()

    def pairs_feasible(self, pairs: Sequence[Sequence[G.Pose]], q_seed: th.Tensor, ignore_for_grasp=()) -> th.Tensor:
        """Batched screening of [pre-grasp, grasp] pairs: pre-grasps against the full world, grasps with
        ``ignore_for_grasp`` (typically the support) disabled. Returns a bool mask per pair."""
        pres = [p[0] for p in pairs]
        grasps = [p[-1] for p in pairs]
        ok_pre = self.ik_batch(pres, q_seed)
        idx = [i for i, s in enumerate(ok_pre.tolist()) if s]
        out = th.zeros(len(pairs), dtype=th.bool)
        if idx:
            ok_g = self.ik_batch([grasps[i] for i in idx], q_seed, ignore=ignore_for_grasp)
            for i, s in zip(idx, ok_g.tolist()):
                out[i] = bool(s)
        return out

    def reach_grid(self, q_stance: th.Tensor, cells_xy: th.Tensor, z: float, yaw_choices: Sequence[float], translates: Sequence[float],
                   toward=None, tilts: Sequence[float] = (0.0, 30.0), pregrasp: float = 0.10, ignore_for_grasp=()) -> Tuple[th.Tensor, List[Optional[float]]]:
        """Per cell: whether some (translate, yaw, tilt) makes grasp + pre-grasp IK-feasible; returns (mask, translate).

        All poses of one trunk translate are solved in batched IK calls (the locked torso joints differ per translate).
        """
        n = len(cells_xy)
        ok = th.zeros(n, dtype=th.bool)
        tr_of: List[Optional[float]] = [None] * n
        for tr in translates:
            todo = [i for i in range(n) if not ok[i]]
            if not todo:
                break
            q = self.with_trunk(q_stance, float(tr), 0.0)
            pairs, index = [], []
            for i in todo:
                p = th.tensor([float(cells_xy[i][0]), float(cells_xy[i][1]), float(z)])
                for k, pair in enumerate(self.grasp_candidates_at(p, yaw_choices, tilts, toward, pregrasp)):
                    pairs.append(pair)
                    index.append((i, k))
            feas = self.pairs_feasible(pairs, q, ignore_for_grasp)
            for (i, k), s in zip(index, feas.tolist()):
                if s and not ok[i]:
                    ok[i] = True
                    tr_of[i] = float(tr)
        return ok, tr_of

    # ------------------------------------------------------------------ traversability

    def eroded_trav_map(self, floor: int = 0) -> th.Tensor:
        """BEHAVIOR's navigation map: eroded by half the robot's chassis diagonal + 0.2 m (``TraversableMap``)."""
        if floor not in self._trav_cache:
            trav = self.world.scene.trav_map
            self._trav_cache[floor] = trav._erode_trav_map(th.clone(trav.floor_map[floor]), robot=self.robot)
        return self._trav_cache[floor]

    #: radius of the base footprint that must lie inside the task rooms (R1Pro chassis half-diagonal ~0.33 m)
    BASE_ROOM_RADIUS = 0.33

    def trav_free(self, xy, floor: int = 0, eroded: bool = False, in_task_rooms: bool = False) -> bool:
        """Whether ``xy`` is on free floor. ``eroded=True`` uses the navigation map (robot footprint + margin); the
        default un-eroded map only asserts that there is floor to stand on (stances next to furniture are validated
        with the whole-body collision check instead). With ``in_task_rooms`` the whole base footprint must also lie
        inside the task rooms (open-plan boundaries such as dining / living room otherwise let the base rest on two
        ``floors`` objects; ``ontop agent floor`` still evaluates true for the floor under the base centre once the
        robot has settled, so this is off by default - it removed every disembedding stance on the test table)."""
        trav = self.world.scene.trav_map
        m = self.eroded_trav_map(floor) if eroded else trav.floor_map[floor]
        p = G.t(xy)[:2]
        ij = trav.world_to_map(p)
        i, j = int(ij[0]), int(ij[1])
        if i < 0 or j < 0 or i >= m.shape[0] or j >= m.shape[1]:
            return False
        if not bool(m[i, j] == 255):
            return False
        rooms = getattr(self.world, "rooms", None)
        if in_task_rooms and rooms:
            seg = self.world.scene.seg_map
            r = self.BASE_ROOM_RADIUS
            for d in (th.tensor([0.0, 0.0]), th.tensor([r, 0.0]), th.tensor([-r, 0.0]), th.tensor([0.0, r]), th.tensor([0.0, -r])):
                if seg.get_room_instance_by_point(p + d) not in rooms:
                    return False
        return True

    def geodesic(self, a_xy, b_xy, floor: int = 0):
        trav = self.world.scene.trav_map
        path, dist = trav.get_shortest_path(floor, G.t(a_xy)[:2], G.t(b_xy)[:2], entire_path=True, robot=self.robot)
        return path, (None if dist is None else float(dist))

    def random_free_points(self, rng, n: int, floor: int = 0, room_filter=None) -> List[th.Tensor]:
        trav = self.world.scene.trav_map
        m = self.eroded_trav_map(floor)
        free = th.nonzero(m == 255)
        out = []
        tries = 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            k = int(rng.integers(len(free)))
            xy = trav.map_to_world(free[k])
            if room_filter is not None and not room_filter(xy):
                continue
            out.append(xy)
        return out

    # ------------------------------------------------------------------ sim application

    def apply(self, q: th.Tensor, base_pose2d: Optional[Tuple[float, float, float]] = None):
        """Teleport the robot to ``q`` (arm / trunk joints) and ``base_pose2d`` (world), then re-sync the cuRobo world."""
        qk = G.t(q).clone()
        qk[[self.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
        self.world.teleport_robot(base_pose2d if base_pose2d is not None else self.world_base_pose2d(), qk)
        self.sync_world()

    def world_base_pose2d(self) -> Tuple[float, float, float]:
        p, q = self.robot.get_position_orientation()
        return float(p[0]), float(p[1]), G.yaw_of(q)

    def joint_dict(self, q: Optional[th.Tensor] = None, include_base: bool = False) -> Dict[str, float]:
        """Joint positions by name. The six virtual base joints are excluded by default: the loader restores the base
        pose from ``robot_poses`` and the virtual joints only hold the offset to the (arbitrary) articulation root."""
        q = self.robot.get_joint_positions() if q is None else G.t(q)
        return {n: float(q[i]) for i, n in enumerate(M.ALL_JOINTS) if include_base or n not in M.BASE_VIRTUAL_JOINTS}
