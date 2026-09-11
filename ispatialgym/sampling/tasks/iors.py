"""Inter-object relation satisfaction (``inter_object_relation_satisfaction-0..4``).

Roles: ``target`` (the object that moves), ``partner`` (stays), fixed ``table`` (init support of both for the
small/small case) or fixed ``partner`` (big scene object, ``under`` small-under-big case), ``floor``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.bddl_writer import IORS_RELATIONS
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler
from ispatialgym.sampling.tro_writer import full_pose_target

import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.object_states import Touching  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


class IORSSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = IORS_RELATIONS[self.definition_id]
        self.big_partner = "partner" in [r.name for r in self.spec.fixed_roles()]
        self.top = None
        self.stance = None
        self.cells = None
        self.reach_mask = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        self.grasp_z_offset = 0.0
        self.floor_z = 0.0

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        if self.big_partner:
            cands = self.fixed_candidates("partner", ("bed", "couch", "sofa", "breakfast_table", "desk"))
            if not cands:
                raise Reject("spec", "no partner candidates")
            self.bind_scene_object("partner", cands[0])
        # the target's init support is chosen in prepare_bindings (needs geometry)
        self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)
        if self.big_partner:
            # last resort for cramped rooms: the big partner's own top (bed, couch) is the init support -- "take the toy
            # from the bed and put it under the bed"
            partner = self.obj("partner")
            if partner not in self._table_candidates:
                self._table_candidates.append(partner)

    def spawn_roles(self):
        for role in self.spec.spawned_roles():
            if role.name in ("target", "partner") and role.name in self.choices:
                self.spawn_role(role.name, self.choices[role.name])

    def prepare_bindings(self):
        # the table binding must exist before the BDDL is built, so choose it here (geometry + stance)
        self._choose_support()

    def _choose_support(self):
        target = self.obj("target")
        self.hull["target"] = self.world.local_hull(target)
        self.rest["target"] = self._rest_quat("target")
        if not self.big_partner:
            partner = self.obj("partner")
            self.hull["partner"] = self.world.local_hull(partner)
            self.rest["partner"] = self._rest_quat("partner")
        # library grasps first: a support / stance is only accepted if some cell admits one of them
        choice = self.choices["target"]
        self.grasps = []
        try:
            key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, ["tabletop", "insert"])
            self.grasps = GL.load_grasps(key)
            self.template_meta["grasp_key"] = key
        except (FileNotFoundError, ValueError) as e:
            log.warning("%s: no grasp library entry for the target (%s); reachability uses generic top-down grasps", self.spec.name, e)
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                if self.big_partner and table is self.obj("partner"):
                    self.alias_role("table", self.bindings.one("partner"))
                else:
                    b = self.bind_scene_object("table", table)
                    if not self.big_partner:
                        self.alias_role("partner_support", b)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def grasp_ignore(self):
        # the support under evaluation is not bound yet while the stance / reach are computed
        cur = getattr(self, "_current_table", None)
        return [cur] if cur is not None else super().grasp_ignore()

    def _try_support(self, table):
        self.kit.sync_world()
        self._current_table = table
        top = self.support_top(table)
        if int(top.free.sum()) < 50:
            raise Reject("support", f"{table.name} has too little free top surface")
        # the default stand-off first; closer for supports whose frame keeps the robot far from the top (beds), further
        # for small furniture in tight rooms (nightstands). With ``try_all_edges`` every edge of the support is tried
        # (the preferred stance edge is not necessarily the one with reachable cells).
        errors = []
        edges = [self.scene_spec.fixed.get("table_front_edge")]
        if self.gen("try_all_edges", False):
            edges += [name for name, *_ in top.edges() if name != edges[0]]
        for stand_off in self.gen("stand_offs", [0.45, 0.35, 0.55, 0.65]):
            for edge in edges:
                try:
                    self._try_support_at(table, top, float(stand_off), edge)
                    self.template_meta.update({"stand_off": float(stand_off), "edge": edge})
                    return
                except Reject as e:
                    errors.append(f"stand_off {stand_off} edge {edge}: {e}")
        raise Reject("no_stance", "; ".join(errors))

    def _try_support_at(self, table, top, stand_off: float, preferred_edge):
        depth_range = tuple(self.gen("depth_range", [0.10, 0.45]))
        lateral_half = float(self.gen("lateral_half", 0.35))
        stance = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=preferred_edge, stand_off=stand_off)
        cells = self.region_cells(top, stance, depth_range, lateral_half, margin=float(self.gen("region_margin", 0.04)))
        if len(cells) == 0:
            raise Reject("support", "placement region empty")
        t_h = self.world.top_offset(self.hull["target"], self.rest["target"]) + self.world.bottom_offset(self.hull["target"], self.rest["target"])
        # generic pinch height: mid-high on the object but never so low that the fingertips (1.9 cm below the eef origin)
        # would have to enter the support
        self.grasp_z_offset = max(0.6 * t_h, 0.03)
        # visible region (frustum test on the object's extent), then reachability on a coarser lattice
        t_pts = G.local_points_world(self.hull["target"], (th.zeros(3), self.rest["target"]))
        t_radius = float(t_pts[:, :2].norm(dim=1).max())
        vis = self.visible_cells(cells, stance.view.cam_pose, top.z, t_h, 0.05, t_radius)
        cells_vis = cells[vis]
        if len(cells_vis) == 0:
            raise Reject("support", "no visible cell in the placement region")
        coarse = self.coarse_cells(cells_vis, float(self.gen("reach_resolution", 0.08)), top.resolution)
        mask, tr = self.reachable_cells(coarse, stance, top.z + self.grasp_z_offset)
        if int(mask.sum()) == 0:
            raise Reject("unreachable", "no reachable cell on the support")
        if not self.big_partner:
            p_pts = G.local_points_world(self.hull["partner"], (th.zeros(3), self.rest["partner"]))
            p_h = float(p_pts[:, 2].max() - p_pts[:, 2].min())
            p_radius = float(p_pts[:, :2].norm(dim=1).max())
            pv = self.visible_cells(cells, stance.view.cam_pose, top.z, p_h, 0.05, p_radius)
            self.partner_cells = cells[pv]
            if len(self.partner_cells) == 0:
                raise Reject("support", "no visible cell for the partner")
        # big partner (under): the partner must be visible from the stance and close
        if self.big_partner:
            partner = self.obj("partner")
            lo, hi = partner.aabb
            corners = G.aabb_corners(lo, hi)
            inside = self.world.camera_model.inside(stance.view.cam_pose, corners, 0.0)
            if int(inside.sum()) < (2 if table is partner else 4):
                raise Reject("partner_not_visible", f"{partner.name} corners in view: {int(inside.sum())}/8")
            if table is not partner:
                # navigation distance from the stance to the partner's near side (the partner's footprint itself is not
                # traversable, so aim at the point of its AABB boundary closest to the stance)
                lo2, hi2 = lo[:2], hi[:2]
                bxy = th.tensor(stance.base_pose2d[:2])
                near = th.minimum(th.maximum(bxy, lo2), hi2)
                d = bxy - near
                goal_xy = near + (d / max(float(th.linalg.norm(d)), 1e-6)) * 0.5
                path, dist = self.kit.geodesic(bxy, goal_xy)
                if dist is None or dist > float(self.gen("max_partner_geodesic", 3.0)) + 1.0:
                    raise Reject("partner_far", f"geodesic to {partner.name}: {dist}")
        self.top, self.stance, self.cells = top, stance, coarse[mask]
        self.reach_translates = [t for t, m in zip(tr, mask.tolist()) if m]
        # per-cell feasible (yaw, grasp, translate) combinations of the library grasps: sampling then never searches
        self.combos = {}
        if self.grasps:
            translates = list(self.gen("reach_translates", [0.5, 0.75, 1.0, 1.25]))
            self.combos = self.grasp_combo_table(self.stance, self.cells, lambda xy, yaw: self._pose_on("target", xy, yaw, self.top.z),
                                                 self.grasps, translates)
            keep = sorted(self.combos)
            if not keep:
                raise Reject("unreachable", "no cell admits a reachable library grasp")
            self.cells = self.cells[keep]
            self.reach_translates = [self.reach_translates[i] for i in keep]
            self.combos = {j: self.combos[i] for j, i in enumerate(keep)}
            self.template_meta["n_grasp_cells"] = len(keep)
            self.template_meta["n_grasp_combos"] = sum(len(v) for v in self.combos.values())
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_reachable_cells": int(mask.sum()),
                                   "n_region_cells": int(len(cells)), "n_visible_cells": int(len(cells_vis))})

    def _rest_quat(self, role: str) -> th.Tensor:
        choice = self.choices.get(role)
        if choice is not None and choice.grasp_key:
            try:
                rq = GL.rest_quat_xyzw(choice.grasp_key)
                if rq is not None:
                    return rq
            except FileNotFoundError:
                pass
        return G.IDENTITY_QUAT.clone()

    def prepare_template(self):
        self.floor_z = float(self.obj("floor").aabb[1][2])

    # ------------------------------------------------------------------ instance

    def _pose_on(self, role: str, xy, yaw: float, z: float) -> G.Pose:
        quat = G.qmul(G.quat_from_yaw(yaw), self.rest[role])
        pos = th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull[role], quat) + 0.003])
        return pos, quat

    def _footprint(self, role: str, pose: G.Pose) -> th.Tensor:
        return self.world.footprint(self.hull[role], pose)

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        st, top, cam = self.stance, self.top, self.stance.view.cam_pose
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        # target cell (reachable & visible lattice), then a (yaw, grasp) combination that is reachable there
        k = int(rng.integers(len(self.cells)))
        t_xy = self.cells[k]
        translates = [self.reach_translates[k]] + [t for t in self.gen("reach_translates", [0.5, 0.75, 1.0, 1.25]) if t != self.reach_translates[k]]
        t_pose, chosen = None, None
        if self.grasps:
            combos = list(self.combos[k])
            perm = rng.permutation(len(combos))
            fails = {"support": 0, "view": 0, "ik": 0}
            for i in perm[:6]:
                yaw, g, tr = combos[i]
                yaw = yaw + float(rng.uniform(-0.15, 0.15))  # small jitter around the screened yaw
                pose = self._pose_on("target", t_xy, yaw, top.z)
                if not top.polygon_supported(self._footprint("target", pose), margin=float(self.gen("support_margin", 0.02))):
                    fails["support"] += 1
                    continue
                if not self.frustum_ok(cam, self.hull_corners_world(self.hull["target"], pose), 0.05):
                    fails["view"] += 1
                    continue
                tr_ok = self.reachable_poses(st, self.grasp_eef_poses(pose, g), [tr] + [t for t in translates if t != tr])
                if tr_ok is None:
                    fails["ik"] += 1
                    continue
                t_pose, chosen = pose, {"key": g.key, "name": g.name, "translate": tr_ok, "yaw": yaw}
                break
            if t_pose is None:
                raise Reject("target_placement", f"no cached (yaw, grasp) combination survives support / view / exact IK ({fails}, cell {t_xy.tolist()})")
        else:
            for _ in range(8):
                yaw = float(rng.uniform(-math.pi, math.pi))
                pose = self._pose_on("target", t_xy, yaw, top.z)
                if not top.polygon_supported(self._footprint("target", pose)) or not self.frustum_ok(cam, self.hull_corners_world(self.hull["target"], pose), 0.05):
                    continue
                centre = th.tensor([float(t_xy[0]), float(t_xy[1]), top.z + self.grasp_z_offset])
                tr_ok = None
                for pair in self.kit.grasp_candidates_at(centre, [st.base_pose2d[2], st.base_pose2d[2] + math.pi / 2], (0.0, 30.0), st.inward):
                    tr_ok = self.reachable_poses(st, pair, translates)
                    if tr_ok is not None:
                        break
                if tr_ok is not None:
                    t_pose, chosen = pose, {"key": None, "translate": tr_ok, "yaw": yaw}
                    break
            if t_pose is None:
                raise Reject("target_placement", "no supported, visible, reachable target pose")
        d.summary["grasp"] = chosen
        t_fp = self._footprint("target", t_pose)
        if not self.big_partner:
            p_pose = self._sample_partner(rng, t_pose, t_fp, top, st)
            d.object_poses[pname] = p_pose
        d.object_poses[tname] = t_pose
        d.robot_q = st.q_stance
        d.base_pose2d = st.base_pose2d
        d.pose_targets[pname] = full_pose_target(*(d.object_poses[pname] if pname in d.object_poses else self.obj("partner").get_position_orientation()),
                                                 self.tol("position", 0.03), self.tol("orientation", 0.20))
        d.pose_targets[tname] = full_pose_target(t_pose[0], t_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        d.summary["relation"] = self.relation
        return d

    def _sample_partner(self, rng, t_pose, t_fp, top, st) -> G.Pose:
        """Partner position from the visible cells, subject to the relation's side constraint and a free footprint."""
        t_depth, t_lat = st.local(t_pose[0])
        cells = self.partner_cells
        rel = cells - st.base_xy
        depth = rel @ st.inward - st.edge_depth
        lat = rel @ st.tangent
        side_by_side = self.relation in ("nextto", "touching")
        if side_by_side:
            # partner on the robot's left; the AABB gap (what NextTo measures) must exceed the NextTo bound at init,
            # so the lateral separation is chosen from the footprints, not from the centres
            t_lat_max = float(((t_fp - st.base_xy) @ st.tangent).max())
            gap_lo, gap_hi = self.gen("init_aabb_gap", [0.10, 0.25])
            keep = ((depth - t_depth).abs() <= 0.12) & (lat > t_lat)
        else:
            keep = ((lat - t_lat).abs() <= 0.35) & (th.linalg.norm(cells - t_pose[0][:2], dim=1) >= 0.05)
        cand = cells[keep]
        if len(cand) == 0:
            raise Reject("partner_placement", "no visible partner cell satisfies the side constraint")
        order = rng.permutation(len(cand))[:60]
        for i in order:
            yaw = float(rng.uniform(-math.pi, math.pi))
            p_pose = self._pose_on("partner", cand[i], yaw, top.z)
            p_fp = self._footprint("partner", p_pose)
            if side_by_side:
                p_lat_min = float(((p_fp - st.base_xy) @ st.tangent).min())
                gap = p_lat_min - t_lat_max
                if gap < gap_lo or gap > gap_hi:
                    continue
            if not top.polygon_supported(p_fp):
                continue
            if G.polygons_overlap(t_fp, p_fp, margin=float(self.gen("min_gap", 0.03))):
                continue
            if not self.frustum_ok(st.view.cam_pose, self.hull_corners_world(self.hull["partner"], p_pose), 0.05):
                continue
            return p_pose
        raise Reject("partner_placement", "no free partner position")

    # ------------------------------------------------------------------ goal

    def after_settle(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        t_pose = draft.object_poses[tname]
        p_obj = self.obj("partner")
        p_pose = draft.object_poses.get(pname) or tuple(x.clone() for x in p_obj.get_position_orientation())
        draft.pose_targets[pname] = full_pose_target(p_pose[0], p_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        goal = self._goal_pose(draft, t_pose, p_pose)
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))

    def _goal_pose(self, draft: Draft, t_pose: G.Pose, p_pose: G.Pose) -> G.Pose:
        rel = self.relation
        st = self.stance
        t_quat = t_pose[1]
        t_hull_w = self.world.hull_at(self.hull["target"], (th.zeros(3), t_quat))  # target hull about its origin
        p_obj = self.obj("partner")
        if rel == "ontop":
            lo, hi = p_obj.aabb
            z = float(hi[2]) - float(t_hull_w[:, 2].min()) + 0.01
            xy = p_obj.aabb_center[:2]
            return th.tensor([float(xy[0]), float(xy[1]), z]), t_quat
        if rel == "inside":
            centre, drop_z = self._container_volume(p_obj)
            z = drop_z - float(t_hull_w[:, 2].min())
            return th.tensor([float(centre[0]), float(centre[1]), z]), t_quat
        if rel in ("nextto", "touching"):
            gap = float(self.gen("nextto_gap", 0.02)) if rel == "nextto" else 0.0
            p_pts = self.world.hull_at(self.hull["partner"], p_pose)[:, :2]
            p_min = float(((p_pts - st.base_xy) @ st.tangent).min())  # partner's right-most lateral coordinate
            t_max = float((t_hull_w[:, :2] @ st.tangent).max())  # target hull extent toward its left (about its origin)
            lat = p_min - gap - t_max  # target origin lateral coordinate
            p_depth, _ = st.local(p_pose[0])
            xy = st.world(p_depth, lat)
            z = float(t_pose[0][2])
            return th.tensor([float(xy[0]), float(xy[1]), z]), t_quat
        if rel == "under":
            return self._under_goal(t_pose, p_obj)
        raise ValueError(rel)

    def _container_volume(self, obj) -> Tuple[th.Tensor, float]:
        """Centroid (xy) of the container meta-link volume (``fillable``; its link origin) and the height from which
        the object is dropped: the container's top, so the object falls into the cavity and settles on its floor."""
        from omnigibson.macros import macros

        lo, hi = obj.aabb
        centre = (lo + hi) / 2
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                p, _ = link.get_position_orientation()
                centre = th.tensor([float(p[0]), float(p[1]), float(centre[2])])
                break
        return centre, float(hi[2]) + 0.005

    def _under_goal(self, t_pose: G.Pose, big) -> G.Pose:
        lo, hi = big.aabb
        t_pts = self.world.hull_at(self.hull["target"], (th.zeros(3), t_pose[1]))
        t_half = float(max(t_pts[:, 0].max() - t_pts[:, 0].min(), t_pts[:, 1].max() - t_pts[:, 1].min())) / 2
        t_height = float(t_pts[:, 2].max() - t_pts[:, 2].min())
        res = 0.05
        xs = th.arange(float(lo[0]) + t_half + 0.05, float(hi[0]) - t_half - 0.05, res)
        ys = th.arange(float(lo[1]) + t_half + 0.05, float(hi[1]) - t_half - 0.05, res)
        if len(xs) == 0 or len(ys) == 0:
            raise Reject("under_goal", "big object footprint too small")
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
        d = th.linalg.norm(pts - t_pose[0][:2], dim=1)
        order = th.argsort(d)
        pts = pts[order][:400]
        starts = th.cat([pts, th.full((len(pts), 1), self.floor_z + 0.005)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(hi[2]) + 0.5
        hits = self.world.raycast(starts, ends)
        prims = {l.prim_path for l in big.links.values()}
        for k, h in enumerate(hits):
            if not h["hit"] or h["rigidBody"] not in prims:
                continue
            clearance = float(h["position"][2]) - self.floor_z
            if clearance < t_height + 0.03:
                continue
            xy = pts[k]
            z = self.floor_z - float(t_pts[:, 2].min()) + 0.003
            return th.tensor([float(xy[0]), float(xy[1]), z]), t_pose[1]
        raise Reject("under_goal", "no clearance under the big object")

    def apply_goal(self, draft: Draft):
        super().apply_goal(draft)

    def refine_targets_from_goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if self.relation == "touching":
            self._nudge_until_touching(tname, pname)
        super().refine_targets_from_goal(draft)
        # partner must not have moved during the goal settle
        p, q = self.obj("partner").get_position_orientation()
        tgt = draft.pose_targets[pname]
        if float(th.linalg.norm(p - th.tensor(tgt["position"]))) > 0.005:
            raise Reject("goal_partner_moved", f"partner moved {float(th.linalg.norm(p - th.tensor(tgt['position']))):.3f} m at the goal")

    def _nudge_until_touching(self, tname: str, pname: str, step: float = 0.001, max_steps: int = 25):
        t, p = self.objects[tname], self.objects[pname]
        direction = self.stance.tangent  # toward the partner (robot's left)
        for _ in range(max_steps):
            self.world.wake([t, p])
            if t.states[Touching].get_value(p):
                return
            pos, quat = t.get_position_orientation()
            t.set_position_orientation(pos + th.tensor([float(direction[0]) * step, float(direction[1]) * step, 0.0]), quat)
            t.keep_still()
            self.world.step_physics(3, still=[self.robot, p])
        raise Reject("goal_infeasible", "could not establish contact for touching")

    # ------------------------------------------------------------------ misc

    def extra_checks(self, draft: Draft):
        draft.summary["view"] = {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt}
