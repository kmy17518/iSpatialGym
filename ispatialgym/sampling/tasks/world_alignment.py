"""World alignment: table surface (``world_alignment-0``), world vertical (``world_alignment-1``) and world horizontal
(``world_alignment-2``).

All three hinge the target at its centre of mass (TRO ``constraints``) so that only its orientation can change; the
goal is the aligned orientation (table axes / world vertical / level along the wall) and the init is a sampled
deviation from it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.constraints import Hinge, create_hinge, remove_hinge
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler
from ispatialgym.sampling.tro_writer import pose_target

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def _axis(name: str) -> th.Tensor:
    name = name if name[0] in "+-" else "+" + name
    return th.tensor(AXES[name], dtype=th.float32)


class _HingedSampler(TaskSampler):
    """Shared hinge bookkeeping: the hinge is (re)created for every draft and removed on rejection / completion."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.hinge: Optional[Hinge] = None
        self.hull = None
        self.com_local = th.zeros(3)

    def _clear_hinge(self):
        remove_hinge(self.hinge)
        self.hinge = None

    def on_reject(self, err: Reject):
        self._clear_hinge()
        super().on_reject(err)

    def teardown(self):
        self._clear_hinge()

    def target_param(self, key: str, default=None):
        """``generation.<key>`` for the bound target: a single value for every target, or a dict keyed by
        ``"<category>/<model>"``, ``"<category>"`` or ``"default"``."""
        v = self.gen(key, default)
        if isinstance(v, dict):
            b = self.bindings.one("target")
            v = v.get(f"{b.category}/{b.model}", v.get(b.category, v.get("default", default)))
        return v

    def axis_param(self, key: str, default: str) -> str:
        """``generation.<key>`` (``front_axis`` / ``long_axis`` / ``up_axis`` / ``level_axis``) as an axis name
        (``"+x"``), per target through :meth:`target_param` (the canonical frames differ per category: books lie with
        the cover text bottom at +x, pictures lie flat with their up at -x, ...; see
        runlogs/world_alignment_selection_20260911/selection.md and runlogs/world_horizontal_selection_20260912/selection.md)."""
        return str(self.target_param(key, default))

    def com_world(self, obj) -> th.Tensor:
        p, q = obj.get_position_orientation()
        return p.float() + G.rotate(q.float(), self.com_local)

    def _com_local(self, obj) -> th.Tensor:
        """Centre of mass in the root frame (link COM offsets from PhysX, mass-weighted)."""
        try:
            total, acc = 0.0, th.zeros(3)
            rp, rq = obj.get_position_orientation()
            for link in obj.links.values():
                m = float(link.mass)
                if m <= 0:
                    continue
                c_local = link.center_of_mass  # in the link frame
                lp, lq = link.get_position_orientation()
                c_world = lp.float() + G.rotate(lq.float(), G.t(c_local))
                acc += m * c_world
                total += m
            if total > 0:
                c = acc / total
                return G.rotate(G.qinv(rq.float()), c - rp.float())
        except Exception:
            pass
        lo, hi = obj.aabb
        rp, rq = obj.get_position_orientation()
        return G.rotate(G.qinv(rq.float()), (lo + hi) / 2 - rp.float())

    def _install_hinge(self, obj, parent_link, axis_world, sink: float = 0.0):
        """Create the hinge at the object's current COM (optionally after lowering the object by ``sink`` metres so
        it presses onto its support and keeps reporting contact)."""
        if sink:
            p, q = obj.get_position_orientation()
            obj.set_position_orientation(p - th.tensor([0.0, 0.0, sink]), q)
            obj.keep_still()
        anchor = self.com_world(obj)
        self.hinge = create_hinge(obj, anchor, axis_world, parent_link=parent_link)
        obj.keep_still()
        return self.hinge

    def rotate_about_com(self, obj, dquat_world):
        """Set the object's orientation to ``dquat_world * q`` keeping its COM fixed (consistent with the hinge)."""
        p, q = obj.get_position_orientation()
        com = self.com_world(obj)
        q_new = G.qmul(dquat_world, q.float())
        p_new = com - G.rotate(q_new, self.com_local)
        obj.set_position_orientation(p_new, q_new)
        obj.keep_still()
        return p_new, q_new


# ----------------------------------------------------------------------------------------------------------------
# Table surface
# ----------------------------------------------------------------------------------------------------------------


class TableSurfaceSampler(_HingedSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.top = None
        self.stance = None
        self.cells = None

    def bind_fixed_roles(self):
        self.bind_floor()
        self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def prepare_bindings(self):
        target = self.obj("target")
        self.hull = self.world.local_hull(target)
        self.com_local = self._com_local(target)
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def _extents(self) -> Tuple[float, float, float]:
        pts = self.hull
        return float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 1].max() - pts[:, 1].min()), float(pts[:, 2].max() - pts[:, 2].min())

    def _try_support(self, table):
        self.kit.sync_world()
        top = self.support_top(table)
        ex, ey, ez = self._extents()
        half_diag = 0.5 * math.hypot(ex, ey)
        depth_range = tuple(self.gen("depth_range", [half_diag + 0.03, half_diag + 0.20]))
        stance = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=self.scene_spec.fixed.get("table_front_edge"))
        cells = self.region_cells(top, stance, depth_range, float(self.gen("lateral_half", 0.15)), margin=half_diag + 0.02)
        if len(cells) == 0:
            raise Reject("support", "placement region empty (object too large for the support depth)")
        cam = stance.view.cam_pose
        # the object must be centred and fully visible, and the front edge + one side edge of the support in view
        vis = self.visible_cells(cells, cam, top.z, ez, 0.05, half_diag)
        cells = cells[vis]
        if len(cells) == 0:
            raise Reject("support", "no cell keeps the whole object in view")
        umin, umax, vmin, vmax = top.local_bounds()
        corners = th.tensor([[u, v] for u in (umin, umax) for v in (vmin, vmax)])
        ax, ay = top.local_axes()
        corners_w = th.stack([c[0] * ax + c[1] * ay for c in corners])
        # front edge = the two corners closest to the robot along the inward axis
        d = (corners_w - stance.base_xy) @ stance.inward
        order = th.argsort(d)
        front = corners_w[order[:2]]
        # a sufficient run of the front edge must be visible (the reference the robot aligns the object with)
        ts = th.linspace(0.0, 1.0, 9).unsqueeze(1)
        edge_pts = front[0] * (1 - ts) + front[1] * ts
        pts = th.cat([edge_pts, th.full((len(edge_pts), 1), top.z)], dim=1)
        n_vis = int(self.world.camera_model.inside(cam, pts, 0.02).sum())
        if n_vis < 3:
            raise Reject("support", f"front edge barely in view ({n_vis}/9 samples)")
        # reachability raster for pushing (top-down poses at the object's height) over the whole near strip: the
        # object's *ends* must be reachable, not its centre (which sits deeper than the arm reaches for long objects)
        strip = self.region_cells(top, stance, (0.02, depth_range[1] + 0.5 * max(ex, ey)), lateral_half=float(self.gen("lateral_half", 0.15)) + 0.5 * max(ex, ey), margin=0.0)
        coarse = self.coarse_cells(strip, float(self.gen("reach_resolution", 0.08)), top.resolution)
        mask, tr = self.reachable_cells(coarse, stance, top.z + max(0.6 * ez, 0.03), tilts=(0.0, 30.0))
        if int(mask.sum()) == 0:
            raise Reject("unreachable", "no reachable cell in the near strip")
        self.reach_pts = coarse[mask]
        self.reach_translates = [t for t, m in zip(tr, mask.tolist()) if m]
        self.top, self.stance, self.cells = top, stance, self.coarse_cells(cells, float(self.gen("cell_resolution", 0.04)), top.resolution)
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_cells": int(len(self.cells)), "n_reachable_strip_cells": int(mask.sum())})

    def _aligned_yaw(self) -> float:
        st = self.stance
        front = _axis(self.axis_param("front_axis", "+x"))[:2]
        target_dir = -st.inward  # front of the object toward the robot
        return math.atan2(float(target_dir[1]), float(target_dir[0])) - math.atan2(float(front[1]), float(front[0]))

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        st, top = self.stance, self.top
        tname = self.bindings.one("target").name
        yaw_goal = self._aligned_yaw()
        lo, hi = self.gen("deviation_range_deg", [20.0, 70.0])
        long_axis = _axis(self.axis_param("long_axis", "y"))
        ex, ey, ez = self._extents()
        half_len = 0.5 * (ex if long_axis[0] != 0 else ey) - 0.03
        z_push = top.z + max(0.6 * ez, 0.03)
        # (cell, deviation) such that the object is supported, in view, and one end lies on a reachable strip cell
        chosen = None
        for _ in range(30):
            delta = math.radians(float(rng.uniform(lo, hi))) * (1 if rng.random() < 0.5 else -1)
            yaw0 = yaw_goal + delta
            xy = self.cells[int(rng.integers(len(self.cells)))]
            quat0 = G.quat_from_yaw(yaw0)
            pos0 = th.tensor([float(xy[0]), float(xy[1]), top.z + self.world.bottom_offset(self.hull, quat0) + 0.003])
            if not top.polygon_supported(self.world.footprint(self.hull, (pos0, quat0))):
                continue
            if not self.frustum_ok(st.view.cam_pose, self.hull_corners_world(self.hull, (pos0, quat0)), 0.05):
                continue
            ends = [pos0 + G.rotate(quat0, long_axis * s * half_len) for s in (1.0, -1.0)]
            near = [float(th.linalg.norm(self.reach_pts - e[:2], dim=1).min()) for e in ends]
            if min(near) > 0.06:
                continue
            chosen = (delta, yaw0, pos0, quat0, ends, near)
            break
        if chosen is None:
            raise Reject("target_placement", "no (cell, deviation) with a reachable end")
        delta, yaw0, pos0, quat0, ends, near = chosen
        # exact pushing reach: top-down poses above the reachable end(s)
        reach = []
        for e, dist in zip(ends, near):
            ok = None
            if dist <= 0.06:
                p = th.tensor([float(e[0]), float(e[1]), z_push])
                for pair in self.kit.grasp_candidates_at(p, [st.base_pose2d[2], st.base_pose2d[2] + math.pi / 2], (0.0, 30.0), st.inward):
                    ok = self.reachable_poses(st, pair)
                    if ok is not None:
                        break
            reach.append(ok)
        if all(r is None for r in reach):
            raise Reject("grasp_unreachable", "neither end of the object is reachable")
        d.summary["push_reach_translates"] = reach
        d.object_poses[tname] = (pos0, quat0)
        d.robot_q = st.q_stance
        d.base_pose2d = st.base_pose2d
        d.summary.update({"yaw_goal": yaw_goal, "yaw_init": yaw0, "deviation_deg": math.degrees(delta)})
        d.pose_targets[tname] = pose_target(orientation=G.quat_from_yaw(yaw_goal), orientation_metric="full",
                                            orientation_tolerance=self.tol("orientation", 0.10), equivalences=self.gen("orientation_equivalences", []))
        return d

    def after_settle(self, draft: Draft):
        """Install the hinge at the settled COM (pressing 1 mm into the support) and record the constraint."""
        target = self.obj("target")
        table = self.obj("table")
        self._clear_hinge()
        self._install_hinge(target, table.root_link, th.tensor([0.0, 0.0, 1.0]), sink=0.001)
        og.sim.step_physics()
        target.keep_still()
        tname, table_name = self.bindings.one("target").name, self.bindings.one("table").name
        draft.extras["constraints"] = [self.hinge.to_tro(tname, table_name)]
        p, q = target.get_position_orientation()
        draft.object_poses[tname] = (p.clone(), q.clone())
        # goal: rotate about the hinge to the aligned yaw
        yaw_goal = draft.summary["yaw_goal"]
        dq = G.qmul(G.quat_from_yaw(yaw_goal), G.qinv(q.float()))
        com = self.com_world(target)
        q_goal = G.quat_from_yaw(yaw_goal)
        p_goal = com - G.rotate(q_goal, self.com_local)
        draft.goal_poses[tname] = (p_goal, q_goal)

    def refine_targets_from_goal(self, draft: Draft):
        pass  # orientation-only target; the goal orientation is exact by construction

    def extra_checks(self, draft: Draft):
        draft.summary["view"] = {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt}

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self._clear_hinge()
        return path


# ----------------------------------------------------------------------------------------------------------------
# World vertical
# ----------------------------------------------------------------------------------------------------------------


class WorldVerticalSampler(_HingedSampler):
    tag = "world_vertical"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.wall_spots: List[Dict[str, Any]] = []
        self.room = None

    def bind_fixed_roles(self):
        self.bind_floor()
        req = self.scene_spec.fixed.get("floor")
        self.room = req.get("room") if isinstance(req, dict) else (self.scene_spec.rooms[0] if self.scene_spec.rooms else None)

    def _in_room(self, xy) -> bool:
        if self.room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == self.room

    def prepare_bindings(self):
        target = self.obj("target")
        self.hull = self.world.local_hull(target)
        self.com_local = self._com_local(target)
        self.kit.sync_world()
        # optional spec restrictions (``scenes.<scene>.fixed``): ``wall`` = allowed wall object names,
        # ``wall_regions`` = surveyed wall segments (runlogs/world_alignment_selection_20260911/wall_candidates.json);
        # with regions the spots are seeded along the segments instead of by rays from the room centre
        req = self.scene_spec.fixed.get("wall")
        allowed_walls = set(req) if isinstance(req, list) else None
        regions = self.scene_spec.fixed.get("wall_regions") or []
        if self.room is not None:
            regions = [r for r in regions if r.get("room") in (None, self.room)]
        spots = self._region_spots(regions) if regions else self._find_wall_spots()
        spots = [s for s in spots if self._in_room(s["point"][:2] + s["normal"][:2] * 0.8)]
        if allowed_walls is not None:
            spots = [s for s in spots if s["wall_name"] in allowed_walls]
        if not spots:
            raise Reject("no_wall", "no flat, free wall patch found in the room" + (f" (restricted to walls {sorted(allowed_walls)})" if allowed_walls else "")
                         + (f" ({len(regions)} spec wall regions)" if regions else ""))
        # per spot: collision-free, traversable base poses on the arc (computed once) and pushing reach of the aligned
        # object's bottom corners from at least one of them
        good = []
        n_bases = 0
        for s in spots:
            bases = self._arc_bases(s)
            n_bases += len(bases)
            if not bases:
                continue
            # goal (aligned) pose: one of the bottom corners must be pushable from the base
            p_goal, q_goal = self._hang_pose(s, 0.0)
            corners = [self._lowest_corner((p_goal, q_goal), s["normal"], side) for side in (1.0, -1.0)]
            keep, reach = [], []
            for b in bases:
                tr = None
                for c in corners:
                    tr = self._reach_contact(b, c, s["normal"])
                    if tr is not None:
                        break
                if tr is not None:
                    keep.append(b)
                    reach.append(tr)
            if not keep:
                continue
            s["bases"], s["goal_reach"] = keep, reach
            good.append(s)
        if not good:
            raise Reject("no_stance", f"{len(spots)} wall patches, {n_bases} traversable collision-free arc positions, none with a pushable goal corner")
        # one wall object per template (one BDDL wall instance): the one with the most usable spots
        counts: Dict[str, int] = {}
        for s in good:
            counts[s["wall_name"]] = counts.get(s["wall_name"], 0) + 1
        wall_name = sorted(counts, key=lambda n: (-counts[n], n))[0]
        wall_obj = self.world.obj(wall_name)
        if len(counts) > 1:
            log.info("%s: spots on several walls %s; keeping those on %s", self.tag, counts, wall_obj.name)
        self.wall_spots = [s for s in good if s["wall_name"] == wall_name]
        self.bind_scene_object("wall", wall_obj)
        self.template_meta["wall_spots"] = [{"point": s["point"].tolist(), "normal": s["normal"].tolist(), "n_bases": len(s["bases"])} for s in self.wall_spots]

    def _push_quat(self, normal) -> th.Tensor:
        """eef orientation for pushing: approach (z) into the wall, fingers closing horizontally along the wall."""
        zq = G.t(normal * -1.0)
        x_ref = th.tensor([0.0, 0.0, 1.0])
        xx = x_ref - th.dot(x_ref, zq) * zq
        xx = xx / th.linalg.norm(xx)
        yy = th.cross(zq, xx, dim=0)
        return T.mat2quat(th.stack([xx, yy, zq], dim=1))

    def _lowest_corner(self, pose: G.Pose, normal, side: float = 0.0) -> th.Tensor:
        """Mean of the lowest hull points; ``side`` (+1 / -1) prefers the corner on that side along the wall tangent."""
        tangent = th.tensor([-float(normal[1]), float(normal[0]), 0.0])
        pts = self.world.hull_at(self.hull, pose)
        score = pts[:, 2] - 0.3 * side * (pts @ tangent)
        return pts[th.argsort(score)[: max(3, len(pts) // 20)]].mean(dim=0)

    def _reach_contact(self, base: Tuple[float, float, float], contact: th.Tensor, normal) -> Optional[float]:
        """First trunk translate from which [pre-push (20 cm off the wall), push standoff (10 cm)] are IK-feasible."""
        quat = self._push_quat(normal)
        standoff = (contact + normal * 0.10, quat)
        pre = (contact + normal * 0.20, quat)
        q_stance = self.kit.q_at_base_pose(self.kit.q_config(arms=self.init_arms()), *base)
        for tr in self.gen("reach_translates", [0.0, 0.25, 0.5, 0.75]):
            if self.kit.reachable([pre, standoff], self.kit.with_trunk(q_stance, tr, 0.0), ignore_for_last=[self.obj("target")]) is not None:
                return float(tr)
        return None

    def _arc_bases(self, spot) -> List[Tuple[float, float, float]]:
        normal = spot["normal"]
        centre_xy = spot["point"][:2]
        r_lo, r_hi = self.gen("arc_radius_range", [0.6, 0.95])
        half = math.radians(float(self.gen("arc_half_angle_deg", 35.0)))
        cands = []
        for r in th.linspace(r_lo, r_hi, 4).tolist():
            for phi in th.linspace(-half, half, 7).tolist():
                dir_out = G.rotate(G.quat_from_yaw(phi), normal)
                base_xy = centre_xy + dir_out[:2] * r
                yaw = math.atan2(float(-dir_out[1]), float(-dir_out[0]))
                if not self.kit.trav_free(base_xy) or not self._in_room(base_xy):
                    continue
                cands.append((float(base_xy[0]), float(base_xy[1]), yaw))
        if not cands:
            return []
        qs = th.stack([self.kit.q_at_base_pose(self.kit.q_config(arms=self.init_arms()), *c) for c in cands])
        free = self.kit.collision_free(qs, margin=0.02)
        return [c for c, f in zip(cands, free.tolist()) if f]

    # -- geometry helpers -------------------------------------------------------------------------------------------

    def _frame_axes(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Object-frame unit vectors (front = toward the room / out of the wall, side = along the wall, up = world up
        when hung) from ``generation.front_axis`` (default +x) and ``generation.up_axis`` (default +z), per category /
        model when given as dicts. The defaults are the frame of the paintings (thin x, standing upright); pictures,
        license plates or guitars that lie flat in their canonical frame use e.g. front +z / up -x."""
        front = _axis(self.axis_param("front_axis", "+x"))
        up = _axis(self.axis_param("up_axis", "+z"))
        if abs(float(th.dot(front, up))) > 1e-6:
            raise Reject("spec", f"front_axis {self.axis_param('front_axis', '+x')} and up_axis {self.axis_param('up_axis', '+z')} are not orthogonal")
        side = th.cross(up, front, dim=0)  # right-handed: front x side = up ... (front, side, up) ~ (x, y, z)
        return front, side, up

    def _hang_quat(self, normal) -> th.Tensor:
        """Rotation that maps the object's front axis onto the wall normal (horizontal) and its up axis onto world +z."""
        n = G.t(normal).clone()
        n[2] = 0.0
        n = n / th.linalg.norm(n)
        z = th.tensor([0.0, 0.0, 1.0])
        t = th.cross(z, n, dim=0)
        W = th.stack([n, t, z], dim=1)  # world basis columns
        front, side, up = self._frame_axes()
        O = th.stack([front, side, up], dim=1)  # object basis columns
        return T.mat2quat(W @ O.T)

    def _object_dims(self) -> Tuple[float, float, float]:
        """(thickness along the front axis, width along the wall, height along the up axis) of the canonical hull."""
        pts = self.hull
        front, side, up = self._frame_axes()
        return tuple(float((pts @ a).max() - (pts @ a).min()) for a in (front, side, up))

    def _patch_extents(self) -> Tuple[float, float]:
        """(width along the wall, height) the wall patch must accommodate: the aligned object's extents here; the
        horizontal problem overrides this with the extents swept over the deviation range."""
        _, width, height = self._object_dims()
        return width, height

    def _goal_equivalences(self) -> List[List[float]]:
        """Object-frame symmetry quaternions written into the target's ``orientation_equivalences``."""
        return [list(q) for q in self.gen("orientation_equivalences", [])]

    def _find_wall_spots(self) -> List[Dict[str, Any]]:
        floor = self.obj("floor")
        lo, hi = floor.aabb
        centre = th.tensor([float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[2]) + float(self.gen("height", 1.45))])
        n_dirs = int(self.gen("wall_ray_directions", 24))
        starts = centre.unsqueeze(0).repeat(n_dirs, 1)
        dirs = th.stack([th.tensor([math.cos(a), math.sin(a), 0.0]) for a in th.linspace(0, 2 * math.pi, n_dirs + 1)[:-1].tolist()])
        ends = starts + dirs * 12.0
        hits = self.world.raycast(starts, ends)
        walls = {l.prim_path: o for o in self.world.scene.objects if o.category == "walls" for l in o.links.values()}
        width, height = self._patch_extents()
        spots = []
        for h in hits:
            if not h["hit"] or h["rigidBody"] not in walls:
                continue
            wall = walls[h["rigidBody"]]
            point = G.t(h["position"]).clone()
            normal = G.t(h["normal"]).clone()
            normal[2] = 0.0
            if float(th.linalg.norm(normal)) < 0.5:
                continue
            normal = normal / th.linalg.norm(normal)
            if not self._flat_patch(point, normal, width + 0.10, height + 0.10, wall):
                continue
            if not self._front_clear(point, normal, width, height):
                continue
            spots.append({"wall_name": wall.name, "wall_link": h["rigidBody"], "point": point, "normal": normal})
        return spots

    def _region_spots(self, regions, step: float = 0.25) -> List[Dict[str, Any]]:
        """Candidate spots along spec wall regions (``{"p0": [x, y], "p1": [x, y], "normal": [nx, ny], "hang_z": z}``
        in world coordinates; ``hang_z`` defaults to floor top + ``generation.height``). Each point is confirmed by a
        ray onto the wall and by the same flat-patch / front-clearance tests as the ray-found spots."""
        floor = self.obj("floor")
        z_default = float(floor.aabb[1][2]) + float(self.gen("height", 1.45))
        walls = {l.prim_path: o for o in self.world.scene.objects if o.category == "walls" for l in o.links.values()}
        width, height = self._patch_extents()
        spots = []
        for r in regions:
            p0, p1 = th.tensor([float(v) for v in r["p0"]]), th.tensor([float(v) for v in r["p1"]])
            n = th.tensor([float(r["normal"][0]), float(r["normal"][1]), 0.0])
            n = n / th.linalg.norm(n)
            z = float(r.get("hang_z", z_default))
            length = float(th.linalg.norm(p1 - p0))
            k = max(1, int(round(length / step)))
            for t in th.linspace(0.0, 1.0, k + 1).tolist():
                xy = p0 * (1 - t) + p1 * t
                start = th.tensor([float(xy[0]), float(xy[1]), z]) + n * 0.5
                end = th.tensor([float(xy[0]), float(xy[1]), z]) - n * 0.15
                h = self.world.raycast(start.unsqueeze(0), end.unsqueeze(0))[0]
                if not h["hit"] or h["rigidBody"] not in walls:
                    continue
                if r.get("wall") and walls[h["rigidBody"]].name != r["wall"]:
                    continue
                point = G.t(h["position"]).clone()
                point[2] = z
                normal = G.t(h["normal"]).clone()
                normal[2] = 0.0
                if float(th.linalg.norm(normal)) < 0.5:
                    continue
                normal = normal / th.linalg.norm(normal)
                wall = walls[h["rigidBody"]]
                if not self._flat_patch(point, normal, width + 0.10, height + 0.10, wall):
                    continue
                if not self._front_clear(point, normal, width, height):
                    continue
                spots.append({"wall_name": wall.name, "wall_link": h["rigidBody"], "point": point, "normal": normal, "region": r.get("id")})
        log.info("%s: %d spots from %d spec wall regions", self.tag, len(spots), len(regions))
        return spots

    def _flat_patch(self, point, normal, w, h, wall) -> bool:
        tangent = th.tensor([-float(normal[1]), float(normal[0]), 0.0])
        up = th.tensor([0.0, 0.0, 1.0])
        starts, ends = [], []
        for u in th.linspace(-w / 2, w / 2, 5).tolist():
            for v in th.linspace(-h / 2, h / 2, 5).tolist():
                p = point + tangent * u + up * v
                starts.append(p + normal * 0.5)
                ends.append(p - normal * 0.1)
        hits = self.world.raycast(th.stack(starts), th.stack(ends))
        prims = {l.prim_path for l in wall.links.values()}
        for hh in hits:
            if not hh["hit"] or hh["rigidBody"] not in prims:
                return False
            if abs(float(hh["distance"]) - 0.5) > 0.01:
                return False
        return True

    def _front_clear(self, point, normal, w, h, depth: float = 0.35) -> bool:
        """No object (other than walls / floors / ceilings) intersects the box in front of the patch."""
        tangent = th.tensor([-float(normal[1]), float(normal[0]), 0.0])
        c = point + normal * depth / 2
        half = th.tensor([0.0, 0.0, 0.0])
        # box corners in world (axis-aligned bound of the oriented box is enough here)
        corners = []
        for su in (-1, 1):
            for sv in (-1, 1):
                for sd in (-1, 1):
                    corners.append(c + tangent * su * (w / 2 + 0.05) + th.tensor([0.0, 0.0, sv * (h / 2 + 0.05)]) + normal * sd * depth / 2)
        corners = th.stack(corners)
        blo, bhi = corners.min(dim=0).values, corners.max(dim=0).values
        for o in self.world.scene.objects:
            if o.category in ("walls", "floors", "ceilings") or o is self.robot or o.name in self.world.spawned:
                continue
            lo, hi = o.aabb
            if bool((lo <= bhi).all() and (hi >= blo).all()):
                return False
        return True

    # -- sampling -----------------------------------------------------------------------------------------------

    def _hang_pose(self, spot, angle: float) -> G.Pose:
        """Object pose hanging on the wall: canonical +z up, thin axis (x) along the wall normal (front face toward the
        room, ``front_axis`` from the spec), rotated in-plane by ``angle`` about the normal through the COM."""
        normal = spot["normal"]
        q_up = self._hang_quat(normal)
        q = G.qmul(G.quat_about_axis(normal, angle), q_up)
        thickness, width, height = self._object_dims()
        centre_target = spot["point"] + normal * (thickness / 2 + 0.01)
        centre_target[2] = float(self.gen("height", 1.45))
        p = centre_target - G.rotate(q, self.com_local)
        return p, q

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname = self.bindings.one("target").name
        weights = th.tensor([float(len(s["bases"])) for s in self.wall_spots])
        spot = self.wall_spots[int(rng.choice(len(self.wall_spots), p=(weights / weights.sum()).numpy()))]
        lo, hi = self.gen("deviation_range_deg", [15.0, 45.0])
        delta = math.radians(float(rng.uniform(lo, hi))) * (1 if rng.random() < 0.5 else -1)
        p0, q0 = self._hang_pose(spot, delta)
        p_goal, q_goal = self._hang_pose(spot, 0.0)
        # robot on an arc in front of the object: cached traversable, collision-free bases of this spot
        normal = spot["normal"]
        obj_centre = p0 + G.rotate(q0, self.com_local)
        chosen = None
        fails = {"view_collision": 0, "frustum": 0}
        order = rng.permutation(len(spot["bases"]))
        for i in order[:12]:
            bx, by, yaw = spot["bases"][i]
            base_xy = th.tensor([bx, by])
            q_base = self.kit.q_at_base_pose(self.kit.q_config(arms=self.init_arms()), bx, by, yaw)
            views = self.kit.solve_view(q_base, obj_centre, translates=th.tensor(self.gen("view_translates", [0.0, 0.25, 0.5, 0.75])), top_k=12)
            qs = th.stack([self.kit.with_trunk(q_base, v.translate, v.tilt) for v in views])
            free = self.kit.collision_free(qs, margin=0.02)
            if not bool(free.any()):
                fails["view_collision"] += 1
                continue
            for v, q, f in zip(views, qs, free.tolist()):
                if not f:
                    continue
                cam = self.kit.camera_pose(q)
                if not self.frustum_ok(cam, self.hull_corners_world(self.hull, (p0, q0)), 0.05):
                    continue
                chosen = (base_xy, yaw, q, v, cam)
                break
            if chosen is not None:
                break
            fails["frustum"] += 1
        if chosen is None:
            raise Reject("no_stance", f"no arc position sees the whole object collision free (spot {spot['wall_name']} at {spot['point'].tolist()}, fails {fails})")
        base_xy, yaw, q_robot, view, cam = chosen
        # pushing reach: the lowest corner of the frame at init (the goal corners were checked for this base at
        # template time) must be IK-feasible with a front approach
        base = (float(base_xy[0]), float(base_xy[1]), yaw)
        reach = {"init": self._reach_contact(base, self._lowest_corner((p0, q0), normal), normal), "goal": spot["goal_reach"][spot["bases"].index(base)]}
        if reach["init"] is None:
            raise Reject("grasp_unreachable", f"init pushing contact unreachable (base {base} r={float(th.linalg.norm(base_xy - obj_centre[:2])):.2f})")
        d.object_poses[tname] = (p0, q0)
        d.goal_poses[tname] = (p_goal, q_goal)
        d.robot_q = q_robot
        d.base_pose2d = (float(base_xy[0]), float(base_xy[1]), yaw)
        d.settle_free = []  # the object hangs on the hinge; nothing rests freely
        d.summary.update({"wall": spot["wall_name"], "deviation_deg": math.degrees(delta), "arc": {"radius": float(th.linalg.norm(base_xy - obj_centre[:2])), "yaw": yaw},
                          "view": {"translate": view.translate, "tilt": view.tilt}, "push_reach_translates": reach})
        d.pose_targets[tname] = pose_target(orientation=q_goal, orientation_metric="full", orientation_tolerance=self.tol("orientation", 0.08),
                                            equivalences=self._goal_equivalences())
        self._spot = spot
        return d

    def apply_draft_extras(self, draft: Draft):
        target = self.obj("target")
        wall = self.obj("wall")
        self._clear_hinge()
        link = None
        for l in wall.links.values():
            if l.prim_path == self._spot["wall_link"]:
                link = l
        self._install_hinge(target, link, self._spot["normal"])
        target.enable_gravity()
        og.sim.step_physics()
        target.keep_still()
        draft.extras["constraints"] = [self.hinge.to_tro(self.bindings.one("target").name, self.bindings.one("wall").name)]

    def settle_entities(self, draft: Draft):
        return [self.obj("target")]

    def extra_checks(self, draft: Draft):
        # the object must not have swung away from its init angle (hinge at the COM: no gravity torque)
        tname = self.bindings.one("target").name
        p, q = self.obj("target").get_position_orientation()
        if G.quat_angle(q, draft.object_poses[tname][1]) > 0.03:
            raise Reject("hinge_drift", f"object rotated {G.quat_angle(q, draft.object_poses[tname][1]):.3f} rad on the hinge while settling")
        # at least N objects (structure included) visible from the init view
        px = self.world.visible_pixels(exclude_structure=False)
        visible = [n for n, c in px.items() if c >= int(self.gen("min_object_pixels", 200)) and n != self.robot.name]
        if len(visible) < int(self.gen("min_visible_objects", 5)):
            raise Reject("too_few_visible", f"{len(visible)} objects visible: {visible}")
        draft.summary["visible_objects"] = visible

    def refine_targets_from_goal(self, draft: Draft):
        pass

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self._clear_hinge()
        return path


# ----------------------------------------------------------------------------------------------------------------
# World horizontal
# ----------------------------------------------------------------------------------------------------------------


def _swept_extent(p: float, q: float, delta: float) -> float:
    """max over ``a in [0, delta]`` of ``p cos a + q sin a``: the extent along one wall axis of a ``p x q`` rectangle
    swept between the aligned pose and a tilt of ``delta`` (the diagonal when the tilt reaches ``atan(q / p)``)."""
    if delta <= 0:
        return p
    if math.atan2(q, p) <= delta:
        return math.hypot(p, q)
    return p * math.cos(delta) + q * math.sin(delta)


class WorldHorizontalSampler(WorldVerticalSampler):
    """``world_alignment-2``: the target hangs on the wall hinge like in world vertical, but it is an elongated wall
    item (towel bar, hook rail, shelf, wide sign / screen, a rod on display hooks) that starts tilted in the wall plane
    and must be pushed back to **level**. Spec (``generation``): ``front_axis`` / ``up_axis`` as for world vertical, the
    optional ``level_axis`` (the reviewers' reading of the axis that lies along the wall; checked against the frame),
    ``symmetric_180`` (bool or per-model dict: the object looks the same upside down, so the half turn about its front
    axis becomes an orientation equivalence of the goal) and ``swept_patch`` (default true: the wall patch must be flat
    and clear for the extents the tilted object sweeps between its largest init tilt and level)."""

    tag = "world_horizontal"

    def _frame_axes(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        front, side, up = super()._frame_axes()
        lv = self.target_param("level_axis", None)
        if lv is not None:
            want = str(lv).strip().lstrip("+-").lower()
            got = "xyz"[int(th.argmax(side.abs()))]
            if want != got:
                raise Reject("spec", f"level_axis {lv} is not the axis along the wall implied by front_axis "
                                     f"{self.axis_param('front_axis', '+x')} / up_axis {self.axis_param('up_axis', '+z')} ({got})")
        return front, side, up

    def _max_deviation(self) -> float:
        lo, hi = self.gen("deviation_range_deg", [15.0, 45.0])
        return math.radians(float(max(abs(float(lo)), abs(float(hi)))))

    def _patch_extents(self) -> Tuple[float, float]:
        _, width, height = self._object_dims()
        if not bool(self.gen("swept_patch", True)):
            return width, height
        d = self._max_deviation()
        return _swept_extent(width, height, d), _swept_extent(height, width, d)

    def _symmetric_180(self) -> bool:
        return bool(self.target_param("symmetric_180", False))

    def _goal_equivalences(self) -> List[List[float]]:
        eqs = super()._goal_equivalences()
        if self._symmetric_180():
            front, _, _ = self._frame_axes()
            eqs.append([float(round(float(v), 6)) or 0.0 for v in G.quat_about_axis(front, math.pi)])  # exact (1, 0, 0, 0) for +x
        return eqs

    def prepare_bindings(self):
        # frame / swept extents first (recorded even when no wall spot admits the object)
        target = self.obj("target")
        self.hull = self.world.local_hull(target)
        self.com_local = self._com_local(target)
        front, side, up = self._frame_axes()
        _, width, height = self._object_dims()
        sw, sh = self._patch_extents()
        self.template_meta["frame"] = {"front_axis": self.axis_param("front_axis", "+x"), "up_axis": self.axis_param("up_axis", "+z"),
                                       "level_axis": "xyz"[int(th.argmax(side.abs()))], "symmetric_180": self._symmetric_180()}
        self.template_meta["swept_dims"] = {"level_width": round(width, 4), "level_height": round(height, 4), "patch_width": round(sw, 4),
                                            "patch_height": round(sh, 4), "max_deviation_deg": round(math.degrees(self._max_deviation()), 2)}
        log.info("%s: %s level %.2f x %.2f m, wall patch %.2f x %.2f m (+0.10 margin), symmetric_180=%s", self.tag,
                 self.bindings.one("target").name, width, height, sw, sh, self._symmetric_180())
        super().prepare_bindings()

    def sample(self, rng: np.random.Generator) -> Draft:
        d = super().sample(rng)
        d.summary["symmetric_180"] = self._symmetric_180()
        return d
