"""Camera relocalization (``camera_relocalization-0``): move from a tucked init base pose to the base / trunk
configuration whose head-camera pose is the goal (shown as the goal image).

Roles: fixed ``floor``, plus the spawned group ``obstacles`` in the *obstructed* variant (spec section 6.2). Everything
is sampled on the BEHAVIOR-eroded traversability map of the loaded floor, restricted to the spec's room; candidates
are filtered analytically (Euclidean, geodesic, heading, collision) and only survivors are rendered (two
seg_instance renders per candidate pair).

Obstructed variant: once init, goal and the free shortest path are chosen, every obstacle is stood upright (canonical
+z-up pose, random yaw) on a free floor cell of the current shortest path, roughly midway, and settled (60 steps,
<= 5 mm drift, no tipping). Its floor footprint (AABB grown by ``obstacle_margin_m``) is blocked in a copy of the
traversability map and the path is re-planned on it: the new path must exist (the eroded map stays connected), be at
least ``obstacle_min_detour_m`` longer than the free path and at most ``max_geodesic_m``, a corridor of
``obstacle_clearance_m`` must stay free on one side of the obstacle, and the obstacle must be inside the head-camera
frame (``obstacle_view_margin``) at some waypoint of the new path **before the path departs from the unobstructed
shortest path** (owner decision 2026-09-15: seeing the obstacle only after the detour, e.g. at the last waypoint when
the base turns to the goal yaw, would let a policy learn the detour without its cause; 0.2 m strides, init trunk, base
heading along the unobstructed path). Init and goal are re-checked for whole-body collisions with the obstacles in the world. The TRO scene block
records ``scene_variant.blocked_footprints`` (the counterpart of the chair variant's ``freed_footprints``); the
summary / template meta record the free and blocked path lengths and the waypoint index where each obstacle is seen.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import knowledge as K
from ispatialgym.sampling import scene_variants as SV
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, TaskSampler
from ispatialgym.sampling.tro_writer import pose_target

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


class CameraRelocalizationSampler(TaskSampler):
    #: obstacle placement (spec 6.2): settle steps and the drift / tilt beyond which a cell is rejected
    OBSTACLE_SETTLE_STEPS = 60
    OBSTACLE_MAX_DRIFT_M = 0.005
    OBSTACLE_MAX_TILT_RAD = 0.05
    #: generation keys of the obstructed variant with their defaults
    OBSTACLE_DEFAULTS = (("obstacle_min_detour_m", 0.3), ("obstacle_view_margin", 0.05), ("obstacle_clearance_m", 0.75),
                         ("obstacle_margin_m", 0.02), ("obstacle_max_cells", 8))

    def __init__(self, ctx):
        super().__init__(ctx)
        self.room = None
        self.floor_z = float(self.world.floor_z)
        self.obst_hull: Dict[str, th.Tensor] = {}  # obstacle instance -> collision hull in its root frame
        self.obst_geom: Dict[str, Dict[str, Any]] = {}
        self._scene_aabbs: List[Tuple[str, th.Tensor, th.Tensor]] = []

    def bind_fixed_roles(self):
        self.bind_floor()
        req = self.scene_spec.fixed.get("floor")
        self.room = req.get("room") if isinstance(req, dict) else (self.scene_spec.rooms[0] if self.scene_spec.rooms else None)

    def prepare_template(self):
        self.q_init_rel = self.kit.q_config(arms=self.init_arms(), torso_zero=True)
        self.template_meta["init_config"] = f"{self.init_arms()} arms, torso joints zero"
        if self.obstacle_names:
            self._prepare_obstacles()

    def _in_room(self, xy) -> bool:
        if self.room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == self.room

    def _config(self, base2d, translate: float, tilt: float, arms: Optional[str] = None) -> th.Tensor:
        arms = self.init_arms() if arms is None else arms
        q = self.kit.q_config(arms=arms, translate=translate, tilt=tilt) if translate is not None else self.q_init_rel
        return self.kit.q_at_base_pose(q, *base2d)

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        max_e = float(self.gen("max_euclidean_m", 5.0))
        max_g = float(self.gen("max_geodesic_m", 10.0))
        max_h = math.radians(float(self.gen("max_heading_deg", 60.0)))
        tr_lo, tr_hi = self.gen("goal_translate_range", [0.0, 1.25])
        ti_lo, ti_hi = self.gen("goal_tilt_range", [-0.5, 0.5])
        min_objs = int(self.gen("min_goal_objects", 3))
        min_px = int(self.gen("min_object_pixels", 200))
        min_shared = float(self.gen("min_shared_visible_fraction", 0.3))
        pts = self.kit.random_free_points(rng, 40, room_filter=self._in_room)
        if len(pts) < 2:
            raise Reject("no_stance", "not enough free points in the room")
        # init: tucked, collision free
        init = None
        for xy in pts:
            yaw = float(rng.uniform(-math.pi, math.pi))
            base = (float(xy[0]), float(xy[1]), yaw)
            q = self._config(base, None, 0.0)
            if bool(self.kit.collision_free(q, margin=0.02)[0]):
                init = (base, q)
                break
        if init is None:
            raise Reject("no_stance", "no collision-free init pose")
        base0, q0 = init
        cam0 = self.kit.camera_pose(q0)
        axis0 = self.world.camera_model.optical_axis(cam0)
        # goal candidates: analytic filters first
        cands = []
        for xy in pts:
            e = float(th.linalg.norm(G.t(xy)[:2] - th.tensor(base0[:2])))
            if e < 0.3 or e > max_e:
                continue
            for _ in range(3):
                yaw1 = base0[2] + float(rng.uniform(-max_h, max_h))
                tr1 = float(rng.uniform(tr_lo, tr_hi))
                ti1 = float(rng.uniform(ti_lo, ti_hi))
                base1 = (float(xy[0]), float(xy[1]), yaw1)
                q1 = self._config(base1, tr1, ti1)
                cam1 = self.kit.camera_pose(q1)
                axis1 = self.world.camera_model.optical_axis(cam1)
                heading = math.acos(max(-1.0, min(1.0, float(th.dot(axis0, axis1)))))
                if heading > max_h:
                    continue
                cands.append((e, heading, base1, tr1, ti1, q1, cam1))
        if not cands:
            raise Reject("goal_sampling", "no goal candidate passes the analytic filters")
        order = rng.permutation(len(cands))
        chosen = None
        n_rendered = 0
        for i in order[:12]:
            e, heading, base1, tr1, ti1, q1, cam1 = cands[i]
            if not bool(self.kit.collision_free(q1, margin=0.02)[0]):
                continue
            path, geo = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
            if geo is None or geo > max_g:
                continue
            # renders: goal first (needs >= min_objs non-structural objects), then init (shared fraction)
            self.kit.apply(q1, base1)
            goal_vis = self.non_structural_visible(min_px)
            n_rendered += 1
            if len(goal_vis) < min_objs:
                continue
            self.kit.apply(q0, base0)
            init_vis = self.non_structural_visible(min_px)
            shared = len(set(goal_vis) & set(init_vis)) / max(len(goal_vis), 1)
            if shared < min_shared:
                continue
            chosen = (e, heading, base1, tr1, ti1, q1, cam1, geo, goal_vis, init_vis, shared, path)
            break
        if chosen is None:
            raise Reject("goal_sampling", f"no goal candidate passed collision / geodesic / visibility ({n_rendered} rendered)")
        e, heading, base1, tr1, ti1, q1, cam1, geo, goal_vis, init_vis, shared, path = chosen
        d.robot_q = q0
        d.base_pose2d = base0
        d.settle_free = []
        d.extras["goal_robot_q"] = q1
        d.extras["goal_base_pose2d"] = base1
        d.pose_targets[AGENT_INSTANCE] = pose_target(position=G.t(cam1[0]).tolist(), orientation=G.t(cam1[1]).tolist(), position_metric="full",
                                                     orientation_metric="full", position_tolerance=self.tol("position", 0.05),
                                                     orientation_tolerance=self.tol("orientation", 0.10), link=self.world.cam_key)
        d.summary.update({"euclidean_m": e, "geodesic_m": geo, "heading_deg": math.degrees(heading), "goal_base_pose2d": list(base1),
                          "goal_trunk": [tr1, ti1], "goal_visible_objects": goal_vis, "init_visible_objects": init_vis, "shared_fraction": shared})
        if self.obstacle_names:
            self._place_obstacles(d, rng, base0, q0, cam0, base1, q1, cam1, path, float(geo), max_g)
        return d

    def check_goal_feasible(self, draft: Draft):
        state = self.world.snapshot()
        before = {n: tuple(x.clone() for x in self.objects[n].get_position_orientation()) for n in draft.object_poses}
        try:
            self.kit.apply(draft.extras["goal_robot_q"], draft.extras["goal_base_pose2d"])
            self.world.step_physics(2, still=[self.robot])
            cam = self.world.camera_pose()
            tgt = draft.pose_targets[AGENT_INSTANCE]
            # the sensor pose realised in simulation must match the FK prediction (else the target is inconsistent)
            err_p = float(th.linalg.norm(G.t(cam[0]) - th.tensor(tgt["position"])))
            err_o = G.quat_angle(G.t(cam[1]), th.tensor(tgt["orientation"]))
            if err_p > 0.5 * self.tol("position", 0.05) or err_o > 0.5 * self.tol("orientation", 0.10):
                # snap the target to the realised sensor pose
                tgt["position"], tgt["orientation"] = G.t(cam[0]).tolist(), G.t(cam[1]).tolist()
            # the obstacles (never manipulated) must not be displaced by the robot standing in the goal configuration
            for name, (p0, r0) in before.items():
                p, r = self.objects[name].get_position_orientation()
                if float(th.linalg.norm(p - p0)) > self.OBSTACLE_MAX_DRIFT_M or G.quat_angle(r, r0) > self.OBSTACLE_MAX_TILT_RAD:
                    raise Reject("obstacle", f"{name} displaced by the goal configuration ({float(th.linalg.norm(p - p0)) * 1000:.1f} mm)")
            self.configure_targets(draft)
            holds, status = self.binary_goal_holds()
            if not holds:
                failed = [c["bddl"] for c in self.goal_report(status) if not c["satisfied"]]
                raise Reject("goal_infeasible", f"unsatisfied at the goal configuration: {failed}")
            draft.summary["goal_fk_error"] = {"position_m": err_p, "orientation_rad": err_o}
            draft.summary["goal_check"] = {"goal": self.goal_report(status)}
        finally:
            self.world.restore(state)
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def refine_targets_from_goal(self, draft: Draft):
        pass

    # ------------------------------------------------------------------ obstacles (obstructed variant, spec 6.2)

    @property
    def obstacle_names(self) -> List[str]:
        return [b.name for b in self.bindings.many("obstacles")]

    def _obstacle_params(self) -> Dict[str, float]:
        return {k: self.gen(k, default) for k, default in self.OBSTACLE_DEFAULTS}

    def _prepare_obstacles(self):
        """Per-template geometry of the obstacles (collision hull in the canonical +z-up pose) and the AABBs of the
        non-structural scene objects an obstacle may not intersect."""
        z_floor = float(self.obj("floor").aabb[1][2])
        if abs(z_floor - float(self.world.floor_z)) < 0.2:
            self.floor_z = z_floor
        for name in self.obstacle_names:
            obj = self.objects[name]
            hull = self.world.local_hull(obj)
            if len(hull) == 0:
                raise Reject("spec", f"{name} ({obj.category}/{obj.model}) has no collision geometry")
            pts = G.local_points_world(hull, (th.zeros(3), G.IDENTITY_QUAT))  # canonical +z-up pose
            lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
            self.obst_hull[name] = hull
            self.obst_geom[name] = {"scene_object_name": obj.name, "category": obj.category, "model": obj.model,
                                    "bottom_offset_m": -float(lo[2]), "height_m": float(hi[2] - lo[2]),
                                    "footprint_m": [float(hi[0] - lo[0]), float(hi[1] - lo[1])], "radius_m": float(pts[:, :2].norm(dim=1).max())}
        # the scene is reset before every attempt, so the AABBs are the same for every instance; carpets stay in the
        # list on purpose (`ontop obstacle floor` needs contact with the floor object, not with a rug)
        self._scene_aabbs = []
        for o in self.world.scene.objects:
            if o is self.robot or o.name in self.world.spawned or o.category in K.STRUCTURE_CATEGORIES:
                continue
            lo, hi = o.aabb
            self._scene_aabbs.append((o.name, lo.clone(), hi.clone()))
        params = self._obstacle_params()
        self.template_meta["obstacles"] = {"instances": dict(self.obst_geom), "generation": params, "settle_steps": self.OBSTACLE_SETTLE_STEPS,
                                           "max_drift_m": self.OBSTACLE_MAX_DRIFT_M, "max_tilt_rad": self.OBSTACLE_MAX_TILT_RAD, "per_instance": {}}
        log.info("%s: obstacles %s (%d scene AABBs to avoid, floor z %.3f, %s)", self.spec.name,
                 {n: f"{g['category']}/{g['model']} {g['footprint_m'][0]:.2f}x{g['footprint_m'][1]:.2f}x{g['height_m']:.2f} m" for n, g in self.obst_geom.items()},
                 len(self._scene_aabbs), self.floor_z, params)

    def _park_obstacle(self, name: str):
        obj = self.objects[name]
        self.world.park(obj, list(self.world.spawned).index(obj.name))

    def _obstacle_pose(self, name: str, xy, yaw: float) -> G.Pose:
        """Upright pose (canonical +z-up, ``yaw`` about z) with the lowest collision point 3 mm above the floor."""
        quat = G.quat_from_yaw(yaw)
        z = self.floor_z + self.world.bottom_offset(self.obst_hull[name], quat) + 0.003
        return th.tensor([float(xy[0]), float(xy[1]), z]), quat

    def _obstacle_aabb(self, name: str, pose: G.Pose) -> Tuple[th.Tensor, th.Tensor]:
        pts = self.world.hull_at(self.obst_hull[name], pose)
        return pts.min(dim=0).values, pts.max(dim=0).values

    def _footprints(self, poses: Dict[str, G.Pose], margin: float) -> Dict[str, List]:
        """``blocked_footprints`` record: every obstacle's AABB projected to the floor, grown by ``margin``."""
        return SV.blocked_footprints([self._obstacle_aabb(n, p) for n, p in poses.items()], margin)

    def _blocked_cells(self, footprints: Dict[str, List]) -> set:
        """Map cells the footprints block: what the planner actually sees (two footprints that block the same cells
        give the same path, so a re-plan is only needed when this set changes)."""
        trav = self.world.scene.trav_map
        out = set()
        for poly in footprints.get("block", []):
            out.update(map(tuple, SV.polygon_cells(trav, poly).tolist()))
        return out

    def _intersecting_object(self, lo: th.Tensor, hi: th.Tensor) -> Optional[str]:
        for name, olo, ohi in self._scene_aabbs:
            if bool((hi >= olo).all()) and bool((lo <= ohi).all()):
                return name
        return None

    @contextmanager
    def _blocked_map(self, footprints: Dict[str, List]):
        """Plan on a copy of the traversability map with ``footprints`` blocked. ``floor_map[0]`` is swapped in for the
        duration (``kit.geodesic`` reads it) and restored afterwards: later templates and the kit's eroded-map cache
        (``random_free_points``) never see the obstacles."""
        trav = self.world.scene.trav_map
        orig = trav.floor_map[0]
        trav.floor_map[0] = orig.clone()
        try:
            SV.block_trav_map(self.world.scene, footprints)
            yield trav.floor_map[0]
        finally:
            trav.floor_map[0] = orig

    @staticmethod
    def _path_cells(rng, path: th.Tensor, max_cells: int) -> List[Tuple[int, th.Tensor]]:
        """Waypoints (index, xy) of the middle 30..70 % of the path (every interior waypoint on short paths), shuffled."""
        n = len(path)
        if n < 3:
            return []
        lo, hi = int(math.floor(0.3 * (n - 1))), int(math.ceil(0.7 * (n - 1)))
        idx = list(range(max(lo, 1), min(hi, n - 2) + 1)) or list(range(1, n - 1))
        perm = rng.permutation(len(idx))
        return [(idx[i], path[idx[i]]) for i in perm[:max_cells]]

    @staticmethod
    def _path_yaws(path: th.Tensor, final_yaw: float) -> List[float]:
        """Heading along the path at every waypoint (the last one keeps the goal yaw, zero-length steps the previous heading)."""
        yaws: List[float] = []
        for i in range(len(path)):
            if i == len(path) - 1:
                yaws.append(float(final_yaw))
                continue
            step = path[i + 1] - path[i]
            if float(th.linalg.norm(step)) < 1e-6:
                yaws.append(yaws[-1] if yaws else float(final_yaw))
            else:
                yaws.append(math.atan2(float(step[1]), float(step[0])))
        return yaws

    @staticmethod
    def _divergence_index(path_before: th.Tensor, path_after: th.Tensor, tol: float) -> int:
        """Index of the last waypoint the two paths share (their common prefix, waypoints within ``tol`` of each other):
        the robot on ``path_after`` departs from ``path_before`` right after this waypoint. 0 when they part immediately."""
        n = min(len(path_before), len(path_after))
        k = 0
        while k + 1 < n and float(th.linalg.norm(path_before[k + 1] - path_after[k + 1])) <= tol:
            k += 1
        return k

    def _view_indices(self, path: th.Tensor, final_yaw: float, corners: th.Tensor, margin: float,
                      path_before: Optional[th.Tensor] = None) -> List[int]:
        """Waypoints of ``path`` (0.2 m strides) at which the whole box ``corners`` (8 x 3) is inside the head-camera
        image with ``margin``, the robot in its init trunk / arm configuration and the base facing along the path.

        With ``path_before`` (the shortest path the robot would take without the obstacle) only the waypoints **up to
        the divergence point** count (owner decision 2026-09-15): the obstacle must be seen before the robot's path
        departs from the unobstructed path, otherwise a policy could learn the detour without ever seeing its cause
        (e.g. an obstacle that only enters the frame at the last waypoint, when the base turns to the goal yaw). The
        heading at those waypoints is the heading of ``path_before`` (at the divergence waypoint: toward where the
        unobstructed path would have continued)."""
        if len(path) == 0:
            return []
        if path_before is None:
            yaws = self._path_yaws(path, final_yaw)
            pts = path
        else:
            k = self._divergence_index(path_before, path, tol=0.5 * float(self.world.scene.trav_map.map_resolution) + 1e-6)
            yaws = self._path_yaws(path_before, final_yaw)[: k + 1]
            pts = path[: k + 1]
        qs = th.stack([self.kit.q_at_base_pose(self.q_init_rel, float(p[0]), float(p[1]), yaw) for p, yaw in zip(pts, yaws)])
        cams = self.kit.camera_poses_batch(qs)
        return [i for i, cam in enumerate(cams) if self.frustum_ok(cam, corners, margin)]

    def _corridor_widths(self, blocked_map: th.Tensor, xy: th.Tensor, direction: th.Tensor, max_m: float = 3.0) -> List[float]:
        """Free floor run (m) on the blocked un-eroded map on either side of the obstacle, perpendicular to the path
        direction at its cell: from the obstacle centre outward, past its own footprint, up to the first blocked cell."""
        trav = self.world.scene.trav_map
        res = float(trav.map_resolution)
        if float(th.linalg.norm(direction)) < 1e-6:
            direction = th.tensor([1.0, 0.0])
        perp = th.tensor([-float(direction[1]), float(direction[0])])
        perp = perp / th.linalg.norm(perp)
        out = []
        for sign in (1.0, -1.0):
            run, started = 0.0, False
            for k in range(1, int(max_m / res) + 1):
                ij = trav.world_to_map(G.t(xy)[:2] + perp * (sign * k * res))
                i, j = int(ij[0]), int(ij[1])
                free = 0 <= i < blocked_map.shape[0] and 0 <= j < blocked_map.shape[1] and bool(blocked_map[i, j] == 255)
                if not started:
                    started = free
                    run = res if free else 0.0
                    continue
                if not free:
                    break
                run += res
            out.append(round(run, 3))
        return out

    @staticmethod
    def _detour_failure(geo_blocked: Optional[float], geo_before: float, max_g: float, min_detour: float) -> Optional[str]:
        if geo_blocked is None:
            return "disconnected"
        if float(geo_blocked) - geo_before < min_detour:
            return "short_detour"
        if float(geo_blocked) > max_g:
            return "too_long"
        return None

    def _view_record(self, poses: Dict[str, G.Pose], path_b: th.Tensor, final_yaw: float, margin: float,
                     ref_paths: Dict[str, th.Tensor]) -> Dict[str, Any]:
        """Per obstacle the waypoints of ``path_b`` at which it is inside the head-camera frame **before the path departs
        from the obstacle's reference path** (the shortest path before that obstacle was placed; at least one, else
        Reject)."""
        view, div = {}, {}
        for name, pose in poses.items():
            ref = ref_paths[name]
            hits = self._view_indices(path_b, final_yaw, G.aabb_corners(*self._obstacle_aabb(name, pose)), margin, path_before=ref)
            div[name] = self._divergence_index(ref, path_b, tol=0.5 * float(self.world.scene.trav_map.map_resolution) + 1e-6)
            if not hits:
                raise Reject("obstacle", f"{name} is not inside the head camera frame at any waypoint before the path departs from the "
                                         f"unobstructed path (divergence at waypoint {div[name]} of {len(path_b)})")
            view[name] = hits
        return {"view_waypoint_index": {n: h[0] for n, h in view.items()}, "view_waypoints": view, "divergence_waypoint_index": div}

    def _navigation_record(self, poses: Dict[str, G.Pose], base0, base1, geo_free: float, max_g: float, P: Dict[str, float],
                           ref_paths: Dict[str, th.Tensor], planned: Optional[Tuple[th.Tensor, float]] = None) -> Dict[str, Any]:
        """Blocked footprints of ``poses``, the path on the blocked map (must divert by ``min_detour`` and stay within
        ``max_g``) and the waypoints at which every obstacle is in view before the departure from its reference path
        (``ref_paths[name]`` = the shortest path before ``name`` was placed). ``planned`` = (path, length) already planned
        on these footprints (one A* on the 862^2 house map costs ~4 s; it is skipped when given)."""
        fps = self._footprints(poses, P["obstacle_margin_m"])
        if planned is not None:
            path_b, geo_b = planned
        else:
            with self._blocked_map(fps):
                path_b, geo_b = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
        why = self._detour_failure(geo_b, geo_free, max_g, P["obstacle_min_detour_m"])
        if why:
            raise Reject("obstacle", f"blocked-map path {why} (free {geo_free:.2f} m, blocked {geo_b})")
        rec = {"blocked_footprints": fps, "free_geodesic_m": float(geo_free), "blocked_geodesic_m": float(geo_b), "detour_m": float(geo_b) - float(geo_free),
               "path_blocked": [[round(float(v), 3) for v in p] for p in path_b], "n_waypoints_blocked": int(len(path_b))}
        rec.update(self._view_record(poses, path_b, base1[2], P["obstacle_view_margin"], ref_paths))
        return rec

    def _try_obstacle_cell(self, name: str, xy, yaw: float, direction: th.Tensor, placed: Dict[str, G.Pose], base0, q0, base1, q1,
                           geo_before: float, max_g: float, P: Dict[str, float], path_before: Optional[th.Tensor] = None):
        """One placement candidate for ``name``: (record, None) on success (the obstacle stands settled in the world),
        (None, failure tag) otherwise (the obstacle is parked again). Cheap geometric filters first, then the settle,
        then the single A* on the blocked map, the corridor, the view along the new path and the collision checks."""
        xy = G.t(xy)[:2]
        geom = self.obst_geom[name]
        if not self._in_room(xy):
            return None, "room"
        keep_out = geom["radius_m"] + P["obstacle_margin_m"] + self.kit.BASE_ROOM_RADIUS + 0.10
        if any(float(th.linalg.norm(xy - th.tensor(b[:2]))) < keep_out for b in (base0, base1)):
            return None, "near_robot"
        pose = self._obstacle_pose(name, xy, yaw)
        lo, hi = self._obstacle_aabb(name, pose)
        fp = SV.aabb_footprint(lo, hi, P["obstacle_margin_m"])
        if not SV.footprint_free(self.world.scene.trav_map, fp):
            return None, "floor"
        if self._intersecting_object(lo, hi) is not None:
            return None, "aabb"
        if any(G.polygons_overlap(th.tensor(fp), th.tensor(SV.aabb_footprint(*self._obstacle_aabb(n, p), P["obstacle_margin_m"]))) for n, p in placed.items()):
            return None, "aabb"
        # stand it up and settle
        obj = self.objects[name]
        self.world.place(obj, xy, pose[1], self.floor_z, local_hull=self.obst_hull[name])
        err = self.world.settle([obj], steps=self.OBSTACLE_SETTLE_STEPS)
        p1, r1 = obj.get_position_orientation()
        drift, tilt, dz = float(th.linalg.norm(p1[:2] - pose[0][:2])), G.quat_angle(r1, pose[1]), float(p1[2] - pose[0][2])
        if err or drift > self.OBSTACLE_MAX_DRIFT_M or tilt > self.OBSTACLE_MAX_TILT_RAD or abs(dz) > 0.02:
            self._park_obstacle(name)
            return None, "unstable"
        pose = (p1.clone(), r1.clone())
        # plan with the settled footprint blocked: divert without closing; corridor on at least one side; in view along
        # the new path; init and goal collision free with the obstacle in the world
        fps = self._footprints({**placed, name: pose}, P["obstacle_margin_m"])
        with self._blocked_map(fps) as blocked:
            path_b, geo_b = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
            corridor = self._corridor_widths(blocked, pose[0][:2], direction)
        why = self._detour_failure(geo_b, geo_before, max_g, P["obstacle_min_detour_m"])
        hits: List[int] = []
        if why is None and max(corridor) < P["obstacle_clearance_m"]:
            why = "corridor"
        if why is None:
            # the obstacle must be seen before the robot departs from the path it would have taken without it
            hits = self._view_indices(path_b, base1[2], G.aabb_corners(*self._obstacle_aabb(name, pose)), P["obstacle_view_margin"],
                                      path_before=path_before)
            if not hits:
                why = "view"
        if why is None:
            self.kit.sync_world()  # the obstacle enters the collision world
            qs = th.stack([self.kit.q_at_base_pose(q0, *base0), self.kit.q_at_base_pose(q1, *base1)])
            free = self.kit.collision_free(qs, margin=0.02).tolist()
            if not all(free):
                why = "collision_init" if not free[0] else "collision_goal"
        if why is not None:
            self._park_obstacle(name)
            return None, why
        return {"pose": pose, "path": path_b, "geodesic_m": float(geo_b), "corridor_m": corridor, "view_indices": hits, "yaw": yaw,
                "settle_drift_m": drift, "settle_tilt_rad": tilt}, None

    def _place_obstacles(self, d: Draft, rng, base0, q0, cam0: G.Pose, base1, q1, cam1: G.Pose, path_free: th.Tensor, geo_free: float, max_g: float):
        """Stand every obstacle on the current shortest path (each one must divert it by ``obstacle_min_detour_m``),
        then validate the final blocked-map path and record everything in the draft."""
        P = self._obstacle_params()
        names = self.obstacle_names
        poses: Dict[str, G.Pose] = {}
        placements: Dict[str, Dict[str, Any]] = {}
        ref_paths: Dict[str, th.Tensor] = {}  # per obstacle: the shortest path before it was placed (the view rule's reference)
        cur_path, cur_geo = path_free, geo_free
        fails: Dict[str, int] = {}
        n_tried = 0
        for name in names:
            placed = None
            for idx, xy in self._path_cells(rng, cur_path, int(P["obstacle_max_cells"])):
                n_tried += 1
                yaw = float(rng.uniform(-math.pi, math.pi))
                direction = cur_path[min(idx + 1, len(cur_path) - 1)] - cur_path[max(idx - 1, 0)]
                rec, why = self._try_obstacle_cell(name, xy, yaw, direction, poses, base0, q0, base1, q1, cur_geo, max_g, P, path_before=cur_path)
                if rec is None:
                    fails[why] = fails.get(why, 0) + 1
                    continue
                placed = rec
                break
            if placed is None:
                raise Reject("obstacle", f"no cell of the shortest path admits {name} ({n_tried} cells tried: {fails}; free path {geo_free:.2f} m, "
                                         f"{len(cur_path)} waypoints)")
            poses[name] = placed["pose"]
            ref_paths[name] = cur_path
            corners = G.aabb_corners(*self._obstacle_aabb(name, placed["pose"]))
            placements[name] = {"position": [round(float(v), 4) for v in placed["pose"][0]], "orientation": [round(float(v), 5) for v in placed["pose"][1]],
                                "yaw": placed["yaw"], "corridor_m": placed["corridor_m"], "settle_drift_m": round(placed["settle_drift_m"], 4),
                                "settle_tilt_rad": round(placed["settle_tilt_rad"], 4), "detour_m": round(placed["geodesic_m"] - cur_geo, 3),
                                "in_view_at_init": self.frustum_ok(cam0, corners, P["obstacle_view_margin"]),
                                "in_view_at_goal": self.frustum_ok(cam1, corners, P["obstacle_view_margin"])}
            cur_path, cur_geo = placed["path"], placed["geodesic_m"]
        # the final path (all obstacles blocked) must divert from the *free* path, stay in range and show every obstacle;
        # the last cell check planned with every obstacle blocked, so its path is the final one (no second A*)
        rec = self._navigation_record(poses, base0, base1, geo_free, max_g, P, ref_paths, planned=(cur_path, cur_geo))
        for name, pose in poses.items():
            d.object_poses[name] = pose
        d.settle_free = list(names)
        d.extras["blocked_footprints"] = rec.pop("blocked_footprints")
        d.extras["obstacle_planned_poses"] = {n: (p[0].clone(), p[1].clone()) for n, p in poses.items()}
        d.extras["obstacle_path"] = (cur_path, cur_geo)
        d.extras["obstacle_ref_paths"] = ref_paths
        d.summary["geodesic_blocked_m"] = rec["blocked_geodesic_m"]
        d.summary["obstacles"] = {"placements": placements, "cells_tried": n_tried, "cell_failures": fails,
                                  "path_free": [[round(float(v), 3) for v in p] for p in path_free], **rec}
        log.info("%s: %d obstacle(s) placed after %d cells (%s): free path %.2f m -> blocked %.2f m, in view at waypoint %s of %d",
                 self.spec.name, len(names), n_tried, fails, geo_free, rec["blocked_geodesic_m"], rec["view_waypoint_index"], rec["n_waypoints_blocked"])

    # ------------------------------------------------------------------ pipeline hooks (obstructed variant)

    def after_settle(self, draft: Draft):
        if not draft.object_poses:
            return
        P = self._obstacle_params()
        planned = draft.extras["obstacle_planned_poses"]
        for name, (p0, r0) in planned.items():
            p, r = draft.object_poses[name]  # settled poses (settle_and_validate refreshed them)
            drift, tilt = float(th.linalg.norm(G.t(p)[:2] - p0[:2])), G.quat_angle(r, r0)
            if drift > self.OBSTACLE_MAX_DRIFT_M or tilt > self.OBSTACLE_MAX_TILT_RAD:
                raise Reject("obstacle", f"{name} moved {drift * 1000:.1f} mm / {tilt:.3f} rad during the settle")
        # footprints, path and view indices follow the settled poses; the path is re-planned only when the (<= 5 mm)
        # motion changed the set of blocked map cells (the planner sees nothing else)
        fps = self._footprints(draft.object_poses, P["obstacle_margin_m"])
        same_cells = self._blocked_cells(fps) == self._blocked_cells(draft.extras["blocked_footprints"])
        rec = self._navigation_record(draft.object_poses, draft.base_pose2d, draft.extras["goal_base_pose2d"], float(draft.summary["geodesic_m"]),
                                      float(self.gen("max_geodesic_m", 10.0)), P, draft.extras["obstacle_ref_paths"],
                                      planned=draft.extras["obstacle_path"] if same_cells else None)
        rec["replanned_after_settle"] = not same_cells
        draft.extras["blocked_footprints"] = rec.pop("blocked_footprints")
        draft.summary["geodesic_blocked_m"] = rec["blocked_geodesic_m"]
        draft.summary["obstacles"].update(rec)

    def tro_meta_block(self, draft: Draft, instance_id: str, seed: int) -> Dict[str, Any]:
        meta = super().tro_meta_block(draft, instance_id, seed)
        fps = draft.extras.get("blocked_footprints")
        if fps:
            sv = dict(meta["scene"].get("scene_variant") or {})
            sv["blocked_footprints"] = fps
            meta["scene"]["scene_variant"] = sv
        return meta

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        if draft.extras.get("blocked_footprints"):
            obs = draft.summary["obstacles"]
            self.template_meta["obstacles"]["per_instance"][self.instance_id(index)] = {
                "blocked_footprints": draft.extras["blocked_footprints"], "free_geodesic_m": obs["free_geodesic_m"],
                "blocked_geodesic_m": obs["blocked_geodesic_m"], "view_waypoint_index": obs["view_waypoint_index"],
                "positions": {n: [round(float(v), 4) for v in p[0]] for n, p in draft.object_poses.items()}}
        return path

    def on_reject(self, err: Reject):
        super().on_reject(err)  # parks the obstacles
        if self.obstacle_names:
            self.kit.sync_world()  # the next attempt's init / goal collision checks must not see a phantom obstacle
