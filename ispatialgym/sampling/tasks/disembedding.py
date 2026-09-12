"""Disembedding (``disembedding-0``): find the target among distractors on the source tray and move it to the
destination tray.

Roles: ``target``, ``distractors`` (drawn per template from ``same_category_pool`` / ``distractor_pool``),
``tray_source``, ``tray_dest``, fixed ``table``, ``floor``.

Template geometry (once): canonical stance, source tray centred in front of the robot with its long axis parallel
to the front edge, destination tray at the same depth on one side so that only ``dest_visible_width`` of it enters
the image, inner-floor rasters of both trays, reachability of library grasps over the source floor and of the
placement at the destination centre (from the laterally shifted stance).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import knowledge as K
from ispatialgym.sampling.bddl_writer import GROUP_NAMES
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.spec import ModelChoice
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, Stance, TaskSampler
from ispatialgym.sampling.tro_writer import full_pose_target

log = logging.getLogger("ispatialgym.sampling")


class DisembeddingSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.top = None
        self.stance = None
        self.place_stance = None
        self.hull: Dict[str, th.Tensor] = {}
        self.radius: Dict[str, float] = {}
        self.height: Dict[str, float] = {}
        self.src_floor: Optional[SupportTop] = None
        self.dst_floor: Optional[SupportTop] = None
        self.grasps = []
        self.combos = {}
        self.cells = None
        self.variant_overrides: List[str] = []

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def spawn_roles(self):
        super().spawn_roles()  # target, tray_source, tray_dest
        rng = np.random.default_rng(self.ctx.base_seed * 31 + 17)
        target = self.choices["target"]
        same = rng.random() < float(self.gen("same_category_prob", 0.5))
        pool_role = self.spec.roles.get("same_category_pool" if same else "distractor_pool")
        pool = [c for c in (pool_role.choices() if pool_role else []) if not (c.category == target.category and c.model == target.model)]
        if not pool and not same:
            pool_role = self.spec.roles.get("same_category_pool")
            pool = [c for c in (pool_role.choices() if pool_role else []) if c.model != target.model]
        if not pool:
            raise Reject("spec", "empty distractor pool")
        n_max = min(int(self.gen("max_distractors", 3)), len(pool))
        n = int(rng.integers(1, n_max + 1))
        order = [pool[i] for i in rng.permutation(len(pool))]
        # the set must fit on the source tray with the target: footprints (canonical bounding boxes, 1 cm margin) may
        # not exceed ``distractor_area_fraction`` of the tray floor (otherwise every instance fails placement); the
        # draw drops to smaller distractors / fewer of them
        tray = self.choices["tray_source"]
        tray_bb = K.canonical_bbox(tray.model) or [0.4, 0.3, 0.05]
        budget = float(self.gen("distractor_area_fraction", 0.45)) * tray_bb[0] * tray_bb[1]

        def area(c):
            bb = K.canonical_bbox(c.model) or [0.1, 0.1, 0.1]
            return (bb[0] + 0.02) * (bb[1] + 0.02)

        used = area(target)
        picks = []
        for c in order:
            if len(picks) >= n:
                break
            if used + area(c) <= budget:
                picks.append(c)
                used += area(c)
        if not picks:
            picks = [min(order, key=area)]
        for c in picks:
            self.spawn_role("distractors", c)
        self.template_meta["distractors"] = {"same_category": bool(same), "models": [c.tag for c in picks], "requested": n,
                                             "footprint_fraction": round(used / (tray_bb[0] * tray_bb[1]), 2)}

    def prepare_bindings(self):
        self._load_grasps()
        for role in ("target", "tray_source", "tray_dest"):
            self._measure(role, self.obj(role))
        for b in self.bindings.many("distractors"):
            self._measure(b.name, self.objects[b.name])
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
                for o in self.world.spawned.values():
                    self.world.park(o, list(self.world.spawned).index(o.name))
        raise Reject("no_support", "; ".join(errors))

    def _measure(self, key: str, obj):
        hull = self.world.local_hull(obj)
        self.hull[key] = hull
        self.radius[key] = float(hull[:, :2].norm(dim=1).max())
        self.height[key] = float(hull[:, 2].max() - hull[:, 2].min())

    def _tray_yaw(self, key: str, stance: Stance) -> float:
        """Yaw putting the tray's longer footprint axis along the front edge (stance tangent)."""
        h = self.hull[key]
        ext = h.max(dim=0).values - h.min(dim=0).values
        tangent_yaw = math.atan2(float(stance.tangent[1]), float(stance.tangent[0]))
        return tangent_yaw if float(ext[0]) >= float(ext[1]) else tangent_yaw - math.pi / 2

    def _pose_on(self, key: str, xy, yaw: float, z: float) -> G.Pose:
        quat = G.quat_from_yaw(yaw)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull[key], quat) + 0.003])
        return pos, quat

    def _tray_floor(self, tray) -> SupportTop:
        """Raster of the tray's inner floor (cells whose downward ray hits the tray at its lowest inner level)."""
        lo, hi = tray.aabb
        res = float(self.gen("tray_resolution", 0.01))
        xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
        ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), float(hi[2]) + 0.10)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(lo[2]) - 0.02
        hits = self.world.raycast(starts, ends)
        prims = {l.prim_path for l in tray.links.values()}
        zs = th.full((len(xs), len(ys)), float("nan"))
        for k, h in enumerate(hits):
            if h["hit"] and h["rigidBody"] in prims:
                i, j = divmod(k, len(ys))
                zs[i, j] = float(h["position"][2])
        valid = zs[~th.isnan(zs)]
        if len(valid) == 0:
            raise Reject("support", f"{tray.name}: no raycast hits")
        floor_z = float(th.quantile(valid, 0.10))
        mask = (~th.isnan(zs)) & ((zs - floor_z).abs() < 0.004)
        top = SupportTop(tray.name, xs, ys, mask, th.zeros_like(mask), floor_z, G.yaw_of(tray.get_position_orientation()[1]), [], res)
        if int(mask.sum()) < 40:
            raise Reject("support", f"{tray.name}: inner floor too small ({int(mask.sum())} cells)")
        return top

    def _try_support(self, table):
        errors = []
        for arms in ["untucked"] + [a for a in self.gen("view_arm_fallbacks", ["ready"]) if a != "untucked"]:
            try:
                self._try_support_arms(table, arms)
                self.template_meta["view_arms"] = arms
                return
            except Reject as e:
                errors.append(f"{arms}: {e}")
                for o in self.world.spawned.values():
                    self.world.park(o, list(self.world.spawned).index(o.name))
        raise Reject("support", "; ".join(errors))

    def _try_support_arms(self, table, arms: str):
        self.kit.sync_world()
        top = self.support_top(table)
        log.info("%s: support %s z %.3f, %d free cells, occupants %s", self.spec.name, top.name, top.z, int(top.free.sum()), top.occupants)
        pref = self.scene_spec.fixed.get("table_front_edge")
        edge_names = [pref] if pref else []
        edge_names += [n for n, _, _, _ in sorted(top.edges(), key=lambda e: -e[3]) if n not in edge_names]
        # both trays must fit side by side along the front edge; on narrow edges the robot (and the source tray in
        # front of it) moves off the edge centre so that the destination tray has room on the other side. Round 1
        # requires the robot square to the edge (destination "barely visible" by its lateral position alone) on any
        # edge; round 2 allows the base-yaw fallback.
        errors = []
        for allow_yaw in (False, True):
            for edge in edge_names:
                try:
                    stance0 = self.choose_stance(top, preferred_edge=edge, arms=arms)
                except Reject as e:
                    errors.append(f"edge {edge}: {e}")
                    continue
                for s_lat in self.gen("source_lateral_offsets", [0.0, 0.25, -0.25, 0.35, -0.35]):
                    try:
                        if s_lat == 0.0:
                            stance = stance0
                        else:
                            focus = stance0.edge_mid + stance0.tangent * float(s_lat)
                            stance = self.choose_stance(top, focus_xy=focus, preferred_edge=stance0.edge_name, arms=arms)
                            if abs(getattr(stance, "lateral", 0.0) - float(s_lat)) > 0.08:
                                raise Reject("no_stance", f"no stance near lateral {s_lat:+.2f} on edge {stance0.edge_name}")
                        self._try_support_stance(table, top, stance, arms, allow_yaw=allow_yaw)
                        self.template_meta["source_lateral_offset"] = float(s_lat)
                        return
                    except Reject as e:
                        errors.append(f"edge {edge} lateral {s_lat:+.2f}{' (yaw allowed)' if allow_yaw else ''}: {e}")
                        for o in self.world.spawned.values():
                            self.world.park(o, list(self.world.spawned).index(o.name))
        raise Reject("support", "; ".join(errors))

    def _try_support_stance(self, table, top, stance, arms: str, allow_yaw: bool = True):
        src, dst = self.obj("tray_source"), self.obj("tray_dest")
        # source tray: centred on the robot's centre line, near edge tray_edge_offset behind the front edge
        yaw_s = self._tray_yaw("tray_source", stance)
        fp0 = self.world.footprint(self.hull["tray_source"], (th.zeros(3), G.quat_from_yaw(yaw_s)))
        half_depth_s = float((fp0 @ stance.inward).abs().max())
        depth_s = float(self.gen("tray_edge_offset", 0.08)) + half_depth_s
        s_pose = self._pose_on("tray_source", stance.world(depth_s, 0.0), yaw_s, top.z)
        log.debug("%s: stance edge %s lateral %.3f base %s; source tray depth %.3f at %s (yaw %.2f)", self.spec.name, stance.edge_name,
                  float(getattr(stance, "lateral", 0.0)), [round(v, 3) for v in stance.base_pose2d], depth_s, [round(float(v), 3) for v in s_pose[0]], yaw_s)
        if not top.polygon_supported(self.world.footprint(self.hull["tray_source"], s_pose)):
            fp = self.world.footprint(self.hull["tray_source"], s_pose)
            raise Reject("support", f"source tray not fully supported at the canonical position (stance lateral {float(getattr(stance, 'lateral', 0.0)):+.2f}, "
                                    f"footprint u {float((fp @ stance.tangent).min()):.2f}..{float((fp @ stance.tangent).max()):.2f} vs support {top.local_bounds()})")
        src.set_position_orientation(*s_pose)
        self.world.activate(src)
        self.world.step_physics(30, still=[self.robot])
        s_pose = tuple(x.clone() for x in src.get_position_orientation())
        src_floor = self._tray_floor(src)
        s_centre = th.tensor([float(src_floor.cells().mean(dim=0)[0]), float(src_floor.cells().mean(dim=0)[1]), src_floor.z])
        # view: source tray centred (within 20 deg: the tray is close, exact centring needs trunk configurations that
        # lean over the edge), whole tray in the image; a standing trunk keeps the camera high and behind the edge
        translates = th.tensor(self.gen("view_translates", [0.0, 0.25, 0.5, 0.75, 1.0]))
        views = self.kit.solve_view(stance.q_stance, s_centre, translates=translates, top_k=40, prefer_translate=float(self.gen("view_prefer_translate", 0.0)),
                                    max_angle=math.radians(float(self.gen("view_max_angle_deg", 20.0))))
        qs = th.stack([self.kit.with_trunk(stance.q_stance, v.translate, v.tilt) for v in views])
        free = self.kit.collision_free(qs, margin=0.01)
        src_corners = self.hull_corners_world(self.hull["tray_source"], s_pose)
        floor_cells = src_floor.cells()
        yaw_d = self._tray_yaw("tray_dest", stance)
        fp0d = self.world.footprint(self.hull["tray_dest"], (th.zeros(3), G.quat_from_yaw(yaw_d)))
        half_lat_d = float((fp0d @ stance.tangent).abs().max())
        half_depth_d = float((fp0d @ stance.inward).abs().max())
        s_fp = self.world.footprint(self.hull["tray_source"], s_pose)
        first = self.gen("dest_side", "right")
        if first == "random":
            first = "right" if np.random.default_rng(self.ctx.base_seed * 31 + 23).random() < 0.5 else "left"
        sides = [first, "left" if first == "right" else "right"]
        depth_d = float(self.gen("tray_edge_offset", 0.08)) + half_depth_d
        half_lat_s = float((fp0 @ stance.tangent).abs().max())
        w = float(self.gen("dest_visible_width", 0.04))
        d_top_off = self.world.top_offset(self.hull["tray_dest"], G.quat_from_yaw(yaw_d)) + self.world.bottom_offset(self.hull["tray_dest"], G.quat_from_yaw(yaw_d))
        # destination tray: furthest supported, non-overlapping lateral position on the chosen side (a 99 deg camera sees
        # far more table than most tables offer, so the tray goes as far out as the support allows and the *base yaw*
        # is then turned away from it until only `w` of the tray remains inside the image)
        dest_options = []
        for side in sides:
            sign = -1.0 if side == "right" else 1.0  # stance.tangent = robot's left
            lat_c = None
            for l in th.arange(half_lat_s + half_lat_d + 0.04, 3.0, 0.02).tolist():
                d_pose = self._pose_on("tray_dest", stance.world(depth_d, sign * l), yaw_d, top.z)
                d_fp = self.world.footprint(self.hull["tray_dest"], d_pose)
                if not top.polygon_supported(d_fp) or G.polygons_overlap(s_fp, d_fp, margin=0.02):
                    break
                lat_c = sign * l
            if lat_c is not None:
                dest_options.append((side, sign, lat_c))
        if not dest_options:
            raise Reject("support", f"no lateral room for the destination tray on either side (support bounds {top.local_bounds()})")
        chosen = None
        diag = {"views": len(views), "collision_free": int(free.sum()), "src_in_view": 0, "lateral_out_of_range": 0, "yaw_search_failed": 0,
                "yaw_collision": 0, "yaw_src_out": 0}
        x0, y0, yaw0 = stance.base_pose2d
        l_min = half_lat_s + half_lat_d + 0.04
        bound_of = {"right": self.world.camera_model.width, "left": 0.0}

        def border_gap(cam_pose, side, sign, lat):
            """Signed image-border distance of the point `w` inside the destination tray's inner edge (at the tray top):
            <= 0 while at least `w` of the tray is inside the image, > 0 once less than `w` shows."""
            p_w = stance.world(depth_d, lat - sign * (half_lat_d - w))
            p_w = th.tensor([float(p_w[0]), float(p_w[1]), top.z + d_top_off])
            uv, _ = self.world.camera_model.project(cam_pose, p_w.unsqueeze(0))
            return (float(uv[0, 0]) - bound_of[side]) * (1.0 if side == "right" else -1.0)

        # pass 1: robot square to the edge (no yaw change); the destination tray's lateral position is bisected between
        # the closest non-overlapping and the furthest supported position until exactly `w` of it is inside the image.
        # pass 2 (fallback, e.g. table ends with no lateral room): tray at the furthest position and the *base yaw*
        # turned away from it until only `w` remains in view.
        for yaw_mode in ((False, True) if allow_yaw else (False,)):
            for v, q, f in zip(views, qs, free.tolist()):
                if not f:
                    continue
                cam = self.kit.camera_pose(q)
                if not self.frustum_ok(cam, src_corners, 0.05):
                    continue
                if not yaw_mode:
                    diag["src_in_view"] += 1
                    vis = self.visible_cells(floor_cells, cam, src_floor.z, self.height["target"], 0.03, self.radius["target"])
                    if float(vis.float().mean()) < 0.25:
                        continue
                for side, sign, lat_c in dest_options:
                    l_max = abs(lat_c)
                    if not yaw_mode:
                        g_lo, g_hi = border_gap(cam, side, sign, sign * l_min), border_gap(cam, side, sign, sign * l_max)
                        if not (g_lo <= 0 <= g_hi):  # even the furthest position shows more than `w` (or the closest less)
                            diag["lateral_out_of_range"] += 1
                            continue
                        lo, hi = l_min, l_max
                        for _ in range(16):
                            mid = 0.5 * (lo + hi)
                            if border_gap(cam, side, sign, sign * mid) <= 0:
                                lo = mid
                            else:
                                hi = mid
                        lat = sign * 0.5 * (lo + hi)
                        d_pose = self._pose_on("tray_dest", stance.world(depth_d, lat), yaw_d, top.z)
                        if not top.polygon_supported(self.world.footprint(self.hull["tray_dest"], d_pose)):
                            diag["lateral_out_of_range"] += 1
                            continue
                        chosen = (v, q, cam, side, d_pose, lat, 0.0)
                        break
                    d_pose = self._pose_on("tray_dest", stance.world(depth_d, lat_c), yaw_d, top.z)

                    def g(psi):
                        cam_psi = self.kit.camera_pose(self.kit.q_at_base_pose(q, x0, y0, yaw0 + psi))
                        return border_gap(cam_psi, side, sign, lat_c), cam_psi

                    # turning away from the destination pushes the tray toward the border: the tray on the robot's right
                    # (sign < 0) leaves the image when the base yaws left (+psi)
                    lo, hi = 0.0, -sign * math.radians(float(self.gen("max_yaw_adjust_deg", 60.0)))
                    g_lo, _ = g(lo)
                    g_hi, _ = g(hi)
                    if g_lo > 0:  # already less than `w` visible: turn toward the tray instead
                        lo, hi = -hi, 0.0
                        g_lo, g_hi = g(lo)[0], g_lo
                    if not (g_lo <= 0 <= g_hi):
                        diag["yaw_search_failed"] += 1
                        continue
                    for _ in range(14):
                        mid = 0.5 * (lo + hi)
                        gm, _ = g(mid)
                        if gm <= 0:
                            lo, g_lo = mid, gm
                        else:
                            hi, g_hi = mid, gm
                    psi = 0.5 * (lo + hi)
                    q_psi = self.kit.q_at_base_pose(q, x0, y0, yaw0 + psi)
                    cam_psi = self.kit.camera_pose(q_psi)
                    # the whole source tray must stay inside the image after the yaw adjustment, and the target must fit
                    # somewhere on it: at least a quarter of the floor cells keep a target-sized box fully in view
                    if not self.frustum_ok(cam_psi, src_corners, 0.05):
                        diag["yaw_src_out"] += 1
                        continue
                    vis = self.visible_cells(floor_cells, cam_psi, src_floor.z, self.height["target"], 0.03, self.radius["target"])
                    if float(vis.float().mean()) < 0.25:
                        diag["yaw_src_out"] += 1
                        continue
                    if not bool(self.kit.collision_free(q_psi, margin=0.01)[0]):
                        diag["yaw_collision"] += 1
                        continue
                    chosen = (v, q_psi, cam_psi, side, d_pose, lat_c, psi)
                    break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None:
            raise Reject("support", f"no view / side keeps the source tray in view with the destination tray barely visible ({diag}; source tray at depth "
                                    f"{depth_s:.2f}, dest options {[(s, round(l, 2)) for s, _, l in dest_options]}, support bounds {top.local_bounds()})")
        v, q, cam, side, d_pose, lat_c, psi = chosen
        # the stance keeps its position and edge frame; only the base yaw (and the camera) may have changed (pass 2)
        stance.base_pose2d = (x0, y0, yaw0 + psi)
        stance.view, stance.q_stance = v, q
        stance.view.cam_pose = cam
        self.template_meta["yaw_adjust_deg"] = math.degrees(psi)
        self.template_meta["dest_lateral_m"] = float(lat_c)
        if abs(psi) > 1e-6:
            log.warning("%s: no lateral position keeps the destination tray barely visible with the robot square to the edge %s; base yaw turned %.1f deg",
                        self.spec.name, stance.edge_name, math.degrees(psi))
        dst.set_position_orientation(*d_pose)
        self.world.activate(dst)
        self.world.step_physics(30, still=[self.robot])
        d_pose = tuple(x.clone() for x in dst.get_position_orientation())
        dst_floor = self._tray_floor(dst)
        # render check: the destination tray is barely visible
        self.kit.apply(stance.q_stance, stance.base_pose2d)
        self.world.step_physics(1, still=[self.robot])
        px = self.world.visible_pixels(exclude_structure=False)
        px_d, px_s = px.get(dst.name, 0), px.get(src.name, 0)
        if px_d == 0 or px_d > float(self.gen("dest_max_visible_fraction", 0.35)) * max(px_s, 1):
            raise Reject("support", f"destination tray visibility out of range (dest {px_d} px, source {px_s} px)")
        # placement stance: shifted sideways toward the destination tray, facing the table edge again
        place_stance = None
        for frac in (1.0, 0.75, 0.5, 0.25):
            bxy = stance.base_xy + stance.tangent * lat_c * frac
            if not self.kit.trav_free(bxy):
                continue
            q_ps = self.kit.q_at_base_pose(self.kit.q_config(arms=self.REACH_ARMS), float(bxy[0]), float(bxy[1]), yaw0)
            if not bool(self.kit.collision_free(q_ps, margin=0.01)[0]):
                continue
            place_stance = Stance(stance.edge_name, stance.edge_mid, stance.inward, stance.tangent, (float(bxy[0]), float(bxy[1]), yaw0), q_ps)
            break
        if place_stance is None:
            raise Reject("no_stance", "no traversable stance in front of the destination tray")
        # target cells: inside the tray floor with the target's radius, and the target-sized box fully in view there
        r_t = self.radius["target"]
        cells = th.stack([c for c in src_floor.cells() if src_floor.contains(c, r_t * 0.8)]) if int(src_floor.free.sum()) else th.zeros(0, 2)
        if len(cells) == 0:
            raise Reject("support", "source tray too small for the target")
        vis = self.visible_cells(cells, stance.view.cam_pose, src_floor.z, self.height["target"], 0.03, r_t)
        cells = cells[vis]
        if len(cells) == 0:
            raise Reject("support", "no target position on the source tray keeps the whole target in view")
        coarse = self.coarse_cells(cells, float(self.gen("reach_resolution", 0.04)), src_floor.resolution)
        self.grasp_z_offset = max(0.6 * self.height["target"], 0.03)
        mask, tr = self.reachable_cells(coarse, stance, src_floor.z + self.grasp_z_offset)
        if int(mask.sum()) == 0:
            raise Reject("unreachable", "no reachable cell on the source tray")
        self.top, self.stance, self.place_stance = top, stance, place_stance
        self.src_floor, self.dst_floor = src_floor, dst_floor
        self.tray_poses = {"tray_source": s_pose, "tray_dest": d_pose}
        self.cells = coarse[mask]
        self.dest_side = side
        d_cells = dst_floor.cells()
        self.dst_centre = th.tensor([float(d_cells[:, 0].mean()), float(d_cells[:, 1].mean())])
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "place_stance": place_stance.to_json(), "dest_side": side,
                                   "dest_pixels": px_d, "source_pixels": px_s, "src_floor_cells": int(src_floor.free.sum()), "dst_floor_cells": int(dst_floor.free.sum()),
                                   "n_reachable_cells": int(mask.sum())})
        self._reach_tables()

    def prepare_template(self):
        # grasps were loaded in prepare_bindings and the grasp / placement tables computed with the stance
        self.template_meta.update({"n_grasp_cells": len(self.cells), "n_place_combos": len(self.place_ok)})

    def _load_grasps(self):
        choice = self.choices["target"]
        try:
            key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, ["floating", "tabletop"])
            self.grasps = GL.load_grasps(key)
            self.template_meta["grasp_key"] = key
        except (FileNotFoundError, ValueError) as e:
            raise Reject("spec", f"no grasp library entry for the target: {e}")

    def _reach_tables(self):
        """Grasp combination table over the source tray cells plus placement feasibility at the destination centre from
        the shifted stance; keeps only cells with a (yaw, grasp) that is both graspable and placeable. Part of the stance
        search: a stance without such a combination is rejected and the next edge / lateral option is tried."""
        translates = list(self.gen("reach_translates", self.DEFAULT_TRANSLATES))
        yaws = list(th.linspace(-math.pi, math.pi, 9)[:-1].tolist())
        combos = self.grasp_combo_table(self.stance, self.cells, lambda xy, yaw: self._pose_on("target", xy, yaw, self.src_floor.z), self.grasps, translates, yaws=yaws)
        place_ok: Dict[Tuple[float, str], float] = {}
        pairs, index = [], []
        for yaw in yaws:
            goal = self._pose_on("target", self.dst_centre, yaw, self.dst_floor.z)
            for g in self.grasps:
                pairs.append(self.grasp_eef_poses(goal, g))
                index.append((yaw, g.name))
        for tr in self.stance_free_translates(self.place_stance, translates):
            feas = self.kit.pairs_feasible(pairs, self.reach_seed(self.place_stance, tr), ignore_for_grasp=self.grasp_ignore() + [self.obj("tray_dest")])
            for (yaw, gname), s in zip(index, feas.tolist()):
                if s and (yaw, gname) not in place_ok:
                    place_ok[(yaw, gname)] = float(tr)
        keep = {}
        for i, lst in combos.items():
            good = [(yaw, g, tr) for (yaw, g, tr) in lst if (yaw, g.name) in place_ok]
            if good:
                keep[i] = good
        if not keep:
            raise Reject("unreachable", "no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray")
        idx = sorted(keep)
        self.cells = self.cells[idx]
        self.combos = {j: keep[i] for j, i in enumerate(idx)}
        self.place_ok = place_ok

    def grasp_ignore(self):
        return [self.objects[b.name] for b in self.bindings.roles.get("table", [])] + [self.obj("tray_source")]

    # ------------------------------------------------------------------ instance

    def _occluded_variant(self, rng) -> bool:
        if "occluded" in self.variant_overrides:
            return True
        if "unoccluded" in self.variant_overrides:
            return False
        return bool(rng.random() < float((self.gen("occlusion") or {}).get("prob", 0.5)))

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        st, cam = self.stance, self.stance.view.cam_pose
        tname = self.bindings.one("target").name
        dnames = [b.name for b in self.bindings.many("distractors")]
        occ = self.gen("occlusion") or {}
        occluded = self._occluded_variant(rng)
        # target: reachable & placeable (yaw, grasp) at a cell; occluded instances need room in front of the target
        if occluded and len(self.cells) > 1:
            depths = th.tensor([st.local(c)[0] for c in self.cells])
            deep = th.nonzero(depths >= depths.median()).flatten().tolist()
            k = int(deep[int(rng.integers(len(deep)))])
        else:
            k = int(rng.integers(len(self.cells)))
        combos = list(self.combos[k])
        perm = rng.permutation(len(combos))
        t_pose, chosen = None, None
        fails = {"support": 0, "view": 0, "grasp_ik": 0, "place_ik": 0}
        for i in perm[:6]:
            yaw0, g, tr = combos[i]
            yaw = yaw0 + float(rng.uniform(-0.15, 0.15))
            pose = self._pose_on("target", self.cells[k], yaw, self.src_floor.z)
            if not self.src_floor.polygon_supported(self.world.footprint(self.hull["target"], pose), margin=0.005):
                fails["support"] += 1
                continue
            if not self.frustum_ok(cam, self.hull_corners_world(self.hull["target"], pose), 0.03):
                fails["view"] += 1
                continue
            tr_ok = self.reachable_poses(st, self.grasp_eef_poses(pose, g), [tr] + [t for t in self.gen("reach_translates", self.DEFAULT_TRANSLATES) if t != tr])
            if tr_ok is None:
                fails["grasp_ik"] += 1
                continue
            goal = self._pose_on("target", self.dst_centre, yaw, self.dst_floor.z)
            tr_place = self.place_ok[(yaw0, g.name)]
            tr_p = self.reachable_poses(self.place_stance, self.grasp_eef_poses(goal, g),
                                        [tr_place] + [t for t in self.gen("reach_translates", self.DEFAULT_TRANSLATES) if t != tr_place],
                                        ignore_for_last=self.grasp_ignore() + [self.obj("tray_dest")])
            if tr_p is None:
                fails["place_ik"] += 1
                continue
            t_pose, chosen = pose, {"key": g.key, "name": g.name, "translate": tr_ok, "place_translate": tr_p, "yaw": yaw}
            break
        if t_pose is None:
            raise Reject("target_placement", f"no cached (yaw, grasp) survives support / view / exact IK / placement ({fails}, cell {self.cells[k].tolist()})")
        d.object_poses[tname] = t_pose
        d.summary["grasp"] = chosen
        # distractors
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        to_cam = G.t(cam[0])[:2] - t_pose[0][:2]
        to_cam = to_cam / th.linalg.norm(to_cam)
        placed = {tname: t_fp}
        order = list(dnames)
        occluder = None
        if occluded and dnames:
            # tallest distractor first (short ones cannot hide a tall target)
            order.sort(key=lambda n: -self.height[n])
            occluder = order[0]
            o_yaw = float(rng.uniform(-math.pi, math.pi))
            base_xy, dist = self._front_position(occluder, o_yaw, t_pose, t_fp, to_cam)
            if base_xy is None:
                if "occluded" in self.variant_overrides:
                    raise Reject("occlusion", "no supported position on the camera ray in front of the target")
                # no room in front of the target on this tray: produce an unoccluded instance instead of wasting the attempt
                occluded, occluder = False, None
                d.summary["occlusion_fallback"] = "no room in front of the target"
            else:
                d.object_poses[occluder] = self._pose_on(occluder, base_xy, o_yaw, self.src_floor.z)
                placed[occluder] = self.world.footprint(self.hull[occluder], d.object_poses[occluder])
        for n in order:
            if n == occluder:
                continue
            pose = self._random_free_position(rng, n, placed, t_pose, to_cam, avoid_front=True)
            if pose is None:
                raise Reject("distractor_placement", f"no free position for {n}")
            d.object_poses[n] = pose
            placed[n] = self.world.footprint(self.hull[n], pose)
        for n in ("tray_source", "tray_dest"):
            d.object_poses[self.bindings.one(n).name] = self.tray_poses[n]
        d.robot_q = st.q_stance
        d.base_pose2d = st.base_pose2d
        d.summary.update({"occluded_variant": occluded, "occluder": occluder, "distractors": dnames, "dest_side": self.dest_side})
        # occlusion tuning by renders on the drafted (unsettled) poses
        self._teleport(d)
        if occluded and occluder is not None:
            self._tune_occluder(d, rng, occluder, tname, dnames, to_cam, occ)
        else:
            max_ratio = float(occ.get("unoccluded_max_ratio", 0.10))
            ratio, vis, ref = self._occlusion(tname, dnames)
            # perspective (the camera looks down from above) lets a distractor beside the target still overlap it in the
            # image: re-draw the distractor positions a few times before giving the attempt up
            for _ in range(int(occ.get("unoccluded_redraws", 3))):
                if ratio <= max_ratio or not dnames:
                    break
                placed = {tname: t_fp}
                for n in dnames:
                    pose = self._random_free_position(rng, n, placed, t_pose, to_cam, avoid_front=True)
                    if pose is None:
                        raise Reject("distractor_placement", f"no free position for {n}")
                    d.object_poses[n] = pose
                    placed[n] = self.world.footprint(self.hull[n], pose)
                self._teleport(d)
                ratio, vis, ref = self._occlusion(tname, dnames)
            if ratio > max_ratio:
                raise Reject("occlusion", f"unoccluded instance has ratio {ratio:.2f}")
            d.extras["initial_occlusion"] = self._occlusion_record(tname, dnames, occluded, ratio, vis, ref)
        goal = self._pose_on("target", self.dst_centre, G.yaw_of(t_pose[1]), self.dst_floor.z)
        d.goal_poses[tname] = goal
        d.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        return d

    def _front_position(self, name: str, yaw: float, t_pose: G.Pose, t_fp: th.Tensor, to_cam: th.Tensor):
        """Closest supported, non-overlapping position on the camera ray in front of the target."""
        r = self.radius[name]
        start = max(0.02, float(((t_fp - t_pose[0][:2]) @ to_cam).max()) + 0.5 * r)
        hull = self.hull[name]
        for s in th.arange(start, start + 0.40, 0.01).tolist():
            xy = t_pose[0][:2] + to_cam * s
            pose = self._pose_on(name, xy, yaw, self.src_floor.z)
            fp = self.world.footprint(hull, pose)
            if G.polygons_overlap(t_fp, fp, margin=0.01):
                continue
            if not self.src_floor.polygon_supported(fp, margin=0.005):
                return None, None
            return xy, s
        return None, None

    def _random_free_position(self, rng, name: str, placed: Dict[str, th.Tensor], t_pose: G.Pose, to_cam: th.Tensor, avoid_front: bool) -> Optional[G.Pose]:
        cells = self.src_floor.cells()
        r = self.radius[name]
        order = rng.permutation(len(cells))[:80]
        for i in order:
            xy = cells[i]
            if not self.src_floor.contains(xy, r * 0.8):
                continue
            rel = xy - t_pose[0][:2]
            dist = float(th.linalg.norm(rel))
            if dist < 1e-3:
                continue
            if avoid_front and float(th.dot(rel / dist, to_cam)) > math.cos(math.radians(40.0)):
                continue  # in the cone between the target and the camera: would occlude
            pose = self._pose_on(name, xy, float(rng.uniform(-math.pi, math.pi)), self.src_floor.z)
            fp = self.world.footprint(self.hull[name], pose)
            if not self.src_floor.polygon_supported(fp, margin=0.005):
                continue
            if any(G.polygons_overlap(fp, other, margin=0.01) for other in placed.values()):
                continue
            return pose
        return None

    def _teleport(self, draft: Draft):
        for name, pose in draft.object_poses.items():
            obj = self.objects[name]
            obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            obj.wake()  # sleeping bodies keep a stale render transform after a physics-side teleport
            obj.keep_still()
        self.kit.apply(draft.robot_q, draft.base_pose2d)
        self.world.step_physics(1, still=[self.robot] + [self.objects[n] for n in draft.object_poses])

    def _occlusion(self, tname: str, dnames: List[str]) -> Tuple[float, int, int]:
        target = self.objects[tname]
        vis = self.world.object_pixels(target)
        ref = self.world.object_pixels(target, hidden=[self.objects[n] for n in dnames])
        if ref == 0:
            # the target is in view by construction; a zero reference means the segmentation label map does not know
            # the recently spawned object yet
            self.world.ensure_segmented(target)
            vis = self.world.object_pixels(target)
            ref = self.world.object_pixels(target, hidden=[self.objects[n] for n in dnames])
        if ref == 0:
            raise Reject("seg_missing", f"{target.name} has no instance-segmentation pixels even with the distractors hidden")
        ratio = 1.0 - vis / max(ref, 1)
        return ratio, vis, ref

    def _occlusion_record(self, tname, dnames, occluded, ratio, vis, ref) -> Dict[str, Any]:
        return {"target": tname, "occluders": list(dnames), "occluded_variant": bool(occluded), "ratio": float(ratio),
                "visible_pixels": int(vis), "reference_pixels": int(ref), "camera": "robot_r1:zed_link:Camera:0"}

    def _tune_occluder(self, draft: Draft, rng, occluder: str, tname: str, dnames: List[str], to_cam: th.Tensor, occ: Dict[str, Any]):
        """Bisect the occluder's lateral offset from the camera ray until the external occlusion ratio is in range."""
        lo_r, hi_r = occ.get("ratio_range", [0.30, 0.70])
        min_vis = int(occ.get("min_visible_pixels", 400))
        side = th.tensor([-float(to_cam[1]), float(to_cam[0])])
        sign = 1.0 if rng.random() < 0.5 else -1.0
        base_pose = draft.object_poses[occluder]
        t_fp = self.world.footprint(self.hull[tname], draft.object_poses[tname])
        others = {n: self.world.footprint(self.hull[n], draft.object_poses[n]) for n in draft.object_poses if n not in (occluder, tname)}
        lo, hi = 0.0, self.radius[occluder] + self.radius[tname] + 0.05
        off = 0.0
        best = None
        for it in range(int(occ.get("max_renders", 6))):
            xy = base_pose[0][:2] + side * (sign * off)
            pose = (th.tensor([float(xy[0]), float(xy[1]), float(base_pose[0][2])]), base_pose[1])
            fp = self.world.footprint(self.hull[occluder], pose)
            valid = self.src_floor.polygon_supported(fp, margin=0.005) and not G.polygons_overlap(fp, t_fp, margin=0.01) \
                and not any(G.polygons_overlap(fp, o, margin=0.01) for o in others.values())
            if not valid:
                if it == 0:
                    sign = -sign
                    continue
                hi = off  # moved out of the tray: come back
                off = (lo + hi) / 2
                continue
            self.objects[occluder].set_position_orientation(*pose)
            self.objects[occluder].keep_still()
            self.world.step_physics(1, still=[self.robot] + [self.objects[n] for n in draft.object_poses])
            ratio, vis, ref = self._occlusion(tname, dnames)
            draft.object_poses[occluder] = pose
            if lo_r <= ratio <= hi_r and vis >= min_vis:
                best = (ratio, vis, ref)
                break
            if ratio > hi_r or vis < min_vis:
                lo = off  # too much occlusion: move further off the ray
            else:
                hi = off
            off = (lo + hi) / 2
        if best is None:
            raise Reject("occlusion", f"could not bring the occlusion ratio into [{lo_r}, {hi_r}] (last {ratio:.2f})")
        draft.extras["initial_occlusion"] = self._occlusion_record(tname, dnames, True, *best)
        draft.summary["occluder_offset"] = off

    # ------------------------------------------------------------------ pipeline hooks

    def settle_entities(self, draft: Draft):
        return [self.objects[n] for n in draft.object_poses]

    def after_settle(self, draft: Draft):
        tname = self.bindings.one("target").name
        t_pose = draft.object_poses[tname]
        goal = self._pose_on("target", self.dst_centre, G.yaw_of(t_pose[1]), self.dst_floor.z)
        goal = (goal[0], t_pose[1])
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))

    def extra_checks(self, draft: Draft):
        tname = self.bindings.one("target").name
        dnames = [b.name for b in self.bindings.many("distractors")]
        occ = self.gen("occlusion") or {}
        ratio, vis, ref = self._occlusion(tname, dnames)
        occluded = bool(draft.summary.get("occluded_variant"))
        lo_r, hi_r = occ.get("ratio_range", [0.30, 0.70])
        if occluded and not (lo_r <= ratio <= hi_r and vis >= int(occ.get("min_visible_pixels", 400))):
            raise Reject("occlusion", f"after settling the occlusion ratio is {ratio:.2f} ({vis} px visible)")
        if not occluded and ratio > float(occ.get("unoccluded_max_ratio", 0.10)):
            raise Reject("occlusion", f"after settling the unoccluded ratio is {ratio:.2f}")
        draft.extras["initial_occlusion"] = self._occlusion_record(tname, dnames, occluded, ratio, vis, ref)
        px = self.world.visible_pixels(exclude_structure=False)
        draft.summary.update({"view": {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt},
                              "dest_pixels": px.get(self.obj("tray_dest").name, 0), "source_pixels": px.get(self.obj("tray_source").name, 0)})
        if draft.summary["dest_pixels"] == 0:
            raise Reject("dest_not_visible", "destination tray not visible after settling")
