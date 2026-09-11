"""Configuration matching / attachment (``configuration_matching-3``): attach the male part to the female part.

Roles: ``male`` (grasped, e.g. cap), ``female`` (e.g. bottle), fixed ``table``, ``floor``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.bddl_writer import GROUP_NAMES
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.object_states import AttachedTo  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


class AttachmentSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.top = None
        self.stance = None
        self.cells = None
        self.hull: Dict[str, th.Tensor] = {}
        self.grasps = []
        self.combos = {}
        self.pair = None  # (child_link, parent_link)

    def bind_fixed_roles(self):
        self.bind_floor()
        self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def spawn_roles(self):
        # AttachedTo is only instantiated for "attachable" objects; the taxonomy does not flag every model that has
        # attachment meta links, so both parts are spawned with the ability added
        for role in self.spec.spawned_roles():
            if role.name in self.choices:
                self.spawn_role(role.name, self.choices[role.name], count=role.count, extra_abilities={"attachable": {}} if role.name in ("male", "female") else None)

    def prepare_bindings(self):
        male, female = self.obj("male"), self.obj("female")
        if AttachedTo not in male.states:
            raise Reject("spec", f"{male.name} has no AttachedTo state (no attachment meta links?)")
        self.hull["male"] = self.world.local_hull(male)
        self.hull["female"] = self.world.local_hull(female)
        self.pair = self._attachment_links(male, female)
        if self.pair is None:
            raise Reject("spec", f"{male.name} has no male attachment link matching a female link of {female.name}")
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def _attachment_links(self, male, female):
        cands = male.states[AttachedTo]._get_parent_candidates(female)
        if not cands:
            return None
        for child_name, parent_names in cands.items():
            for parent_name in parent_names:
                return male.states[AttachedTo].links[child_name], female.states[AttachedTo].links[parent_name]
        return None

    def _try_support(self, table):
        self.kit.sync_world()
        top = self.support_top(table)
        depth_range = tuple(self.gen("depth_range", [0.10, 0.40]))
        stance = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=self.scene_spec.fixed.get("table_front_edge"))
        cells = self.region_cells(top, stance, depth_range, float(self.gen("lateral_half", 0.30)))
        if len(cells) == 0:
            raise Reject("support", "placement region empty")
        m_pts, f_pts = self.hull["male"], self.hull["female"]
        m_h = float(m_pts[:, 2].max() - m_pts[:, 2].min())
        f_h = float(f_pts[:, 2].max() - f_pts[:, 2].min())
        vis_m = self.visible_cells(cells, stance.view.cam_pose, top.z, m_h, 0.05, float(m_pts[:, :2].norm(dim=1).max()))
        vis_f = self.visible_cells(cells, stance.view.cam_pose, top.z, f_h, 0.05, float(f_pts[:, :2].norm(dim=1).max()))
        self.male_cells_fine = cells[vis_m]
        self.female_cells = cells[vis_f]
        if len(self.male_cells_fine) == 0 or len(self.female_cells) == 0:
            raise Reject("support", "no visible cells")
        coarse = self.coarse_cells(self.male_cells_fine, float(self.gen("reach_resolution", 0.08)), top.resolution)
        mask, tr = self.reachable_cells(coarse, stance, top.z + max(0.6 * m_h, 0.03))
        if int(mask.sum()) == 0:
            raise Reject("unreachable", "no reachable cell for the male part")
        self.top, self.stance = top, stance
        self.cells = coarse[mask]
        self.reach_translates = [t for t, m in zip(tr, mask.tolist()) if m]
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_reachable_cells": int(mask.sum())})

    def prepare_template(self):
        choice = self.choices["male"]
        try:
            key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, ["attachment", "tabletop"])
            self.grasps = GL.load_grasps(key)
            self.template_meta["grasp_key"] = key
        except (FileNotFoundError, ValueError) as e:
            raise Reject("spec", f"no grasp library entry for the male part: {e}")
        translates = list(self.gen("reach_translates", [0.5, 0.75, 1.0, 1.25]))
        self.combos = self.grasp_combo_table(self.stance, self.cells, lambda xy, yaw: self._pose_on("male", xy, yaw, self.top.z), self.grasps, translates)
        keep = sorted(self.combos)
        if not keep:
            raise Reject("unreachable", "no cell admits a reachable library grasp of the male part")
        self.cells = self.cells[keep]
        self.combos = {j: self.combos[i] for j, i in enumerate(keep)}
        # attachment goal frame: child root pose in the parent link frame (rigid, computed once)
        child_link, parent_link = self.pair
        male = self.obj("male")
        # OmniGibson's AttachedTo no longer filters child / parent collisions on attach; an attached cap whose collision
        # hull overlaps the bottle neck then jitters violently. Filter the pair once (stop / play inside) so the goal
        # settle measures the attachment, not the interpenetration. The evaluator must do the same on attach.
        male.states[AttachedTo]._disable_collision_between_child_and_parent(male, self.obj("female"))
        og.sim.step_physics()
        self.kit.sync_world()
        cp, cq = child_link.get_position_orientation()
        rp, rq = male.get_position_orientation()
        self.root_in_child_link = G.relative((rp.float(), rq.float()), (cp.float(), cq.float()))
        self.symmetry = self._symmetry_equivalences(male, child_link)
        self.template_meta.update({"n_grasp_cells": len(keep), "attachment_links": [child_link.body_name, parent_link.body_name],
                                   "n_symmetry_equivalences": len(self.symmetry)})

    def _symmetry_equivalences(self, male, child_link) -> List[List[float]]:
        """Rotations about the attachment axis (child link z) that leave the male part visually unchanged.

        Explicit ``generation.orientation_equivalences`` win. Otherwise the part is tested for being *revolute* about
        the attachment axis: in every 5 mm slab along the axis the hull points' radial distances vary by less than
        ``symmetry_radial_tolerance`` (default 10 %); a revolute part gets every multiple of ``symmetry_step_deg``.
        """
        explicit = self.gen("orientation_equivalences")
        if explicit:
            return [list(map(float, q)) for q in explicit]
        rp, rq = male.get_position_orientation()
        cp, cq = child_link.get_position_orientation()
        axis_local = G.rotate(G.qinv(rq.float()), G.rotate(cq.float(), th.tensor([0.0, 0.0, 1.0])))
        centre_local = G.rotate(G.qinv(rq.float()), cp.float() - rp.float())
        pts = self.hull["male"] - centre_local
        h = pts @ axis_local
        radial = pts - h.unsqueeze(1) * axis_local
        r = radial.norm(dim=1)
        tol = float(self.gen("symmetry_radial_tolerance", 0.10))
        for lo in th.arange(float(h.min()), float(h.max()) + 1e-6, 0.005).tolist():
            sel = (h >= lo) & (h < lo + 0.005)
            if int(sel.sum()) < 4:
                continue
            rs = r[sel]
            if float(rs.max()) > 0.004 and float(rs.max() - rs.min()) / float(rs.max()) > tol:
                return []
        step = math.radians(float(self.gen("symmetry_step_deg", 15.0)))
        n = int(round(2 * math.pi / step))
        return [[float(v) for v in G.quat_about_axis(axis_local, k * step)] for k in range(1, n)]

    def _pose_on(self, role: str, xy, yaw: float, z: float) -> G.Pose:
        quat = G.quat_from_yaw(yaw)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull[role], quat) + 0.003])
        return pos, quat

    # ------------------------------------------------------------------ instance

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        st, top, cam = self.stance, self.top, self.stance.view.cam_pose
        mname, fname = self.bindings.one("male").name, self.bindings.one("female").name
        k = int(rng.integers(len(self.cells)))
        m_xy = self.cells[k]
        combos = list(self.combos[k])
        perm = rng.permutation(len(combos))
        m_pose, chosen = None, None
        for i in perm[:6]:
            yaw, g, tr = combos[i]
            pose = self._pose_on("male", m_xy, yaw, top.z)
            if not top.polygon_supported(self.world.footprint(self.hull["male"], pose)):
                continue
            if not self.frustum_ok(cam, self.hull_corners_world(self.hull["male"], pose), 0.05):
                continue
            tr_ok = self.reachable_poses(st, self.grasp_eef_poses(pose, g), [tr] + [t for t in self.gen("reach_translates", [0.5, 0.75, 1.0, 1.25]) if t != tr])
            if tr_ok is None:
                continue
            m_pose, chosen = pose, (g, tr_ok, yaw)
            break
        if m_pose is None:
            raise Reject("target_placement", "no cached grasp combination survives")
        g, tr_ok, yaw = chosen
        # female part: upright, within the visible cells, not overlapping the male part, attach site reachable
        m_fp = self.world.footprint(self.hull["male"], m_pose)
        f_pose = None
        order = rng.permutation(len(self.female_cells))[:60]
        for i in order:
            pose = self._pose_on("female", self.female_cells[i], float(rng.uniform(-math.pi, math.pi)), top.z)
            fp = self.world.footprint(self.hull["female"], pose)
            if float(th.linalg.norm(pose[0][:2] - m_pose[0][:2])) < 0.08:
                continue
            if not top.polygon_supported(fp) or G.polygons_overlap(m_fp, fp, margin=float(self.gen("min_gap", 0.04))):
                continue
            if not self.frustum_ok(cam, self.hull_corners_world(self.hull["female"], pose), 0.05):
                continue
            goal = self._attached_pose(pose)
            eef_goal = self.grasp_eef_poses(goal, g)  # [10 cm above the attach pose, attach pose] along the grasp approach
            if self.reachable_poses(st, eef_goal) is None:
                continue
            f_pose = pose
            break
        if f_pose is None:
            raise Reject("partner_placement", "no female position with a reachable attach site")
        goal = self._attached_pose(f_pose)
        d.object_poses[mname] = m_pose
        d.object_poses[fname] = f_pose
        d.goal_poses[mname] = goal
        d.robot_q = st.q_stance
        d.base_pose2d = st.base_pose2d
        d.summary["grasp"] = {"key": g.key, "name": g.name, "translate": tr_ok, "yaw": yaw}
        d.groups[GROUP_NAMES["attachment"]] = self._group_target(f_pose, goal, mname, fname)
        child_link, parent_link = self.pair
        d.extras["attachment_targets"] = [{"child": mname, "parent": fname, "child_link": child_link.body_name, "parent_link": parent_link.body_name, "joint_type": "fixed"}]
        return d

    def _attached_pose(self, female_pose: G.Pose) -> G.Pose:
        """Male root pose when its attachment link coincides with the female attachment link (female at ``female_pose``)."""
        child_link, parent_link = self.pair
        female = self.obj("female")
        fp, fq = female.get_position_orientation()
        pp, pq = parent_link.get_position_orientation()
        parent_link_in_female = G.relative((pp.float(), pq.float()), (fp.float(), fq.float()))
        parent_link_world = G.compose(female_pose, parent_link_in_female)
        return G.compose(parent_link_world, self.root_in_child_link)

    def _group_target(self, f_pose: G.Pose, m_goal: G.Pose, mname: str, fname: str) -> Dict[str, Any]:
        rel_p, rel_q = G.relative(m_goal, f_pose)
        return {
            "members": [mname, fname],
            "assignment": {"mode": "fixed", "member_to_slot": {mname: "male", fname: "female"}},
            "orientation_equivalences": self.symmetry,
            "target_slots": {
                "male": {"position": [float(v) for v in m_goal[0]], "orientation": [float(v) for v in m_goal[1]]},
                "female": {"position": [float(v) for v in f_pose[0]], "orientation": [float(v) for v in f_pose[1]]},
            },
            "pose_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
            "layout_relations": [{"from": "female", "to": "male", "position": [float(v) for v in rel_p], "orientation": [float(v) for v in rel_q]}],
            "layout_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
        }

    def after_settle(self, draft: Draft):
        # the female part settled: recompute the attach pose and the group target from its settled pose
        mname, fname = self.bindings.one("male").name, self.bindings.one("female").name
        f_pose = draft.object_poses[fname]
        goal = self._attached_pose(f_pose)
        draft.goal_poses[mname] = goal
        draft.groups[GROUP_NAMES["attachment"]] = self._group_target(f_pose, goal, mname, fname)

    def apply_goal(self, draft: Draft):
        super().apply_goal(draft)
        male, female = self.obj("male"), self.obj("female")
        og.sim.step_physics()
        ok = male.states[AttachedTo].set_value(female, True)
        if not ok:
            raise Reject("goal_infeasible", "AttachedTo.set_value failed at the computed attach pose")

    def refine_targets_from_goal(self, draft: Draft):
        mname, fname = self.bindings.one("male").name, self.bindings.one("female").name
        m_pose = tuple(x.clone() for x in self.obj("male").get_position_orientation())
        f_pose = tuple(x.clone() for x in self.obj("female").get_position_orientation())
        if float(th.linalg.norm(f_pose[0] - draft.object_poses[fname][0])) > 0.005:
            raise Reject("goal_partner_moved", "female part moved while attaching")
        draft.groups[GROUP_NAMES["attachment"]] = self._group_target(f_pose, m_pose, mname, fname)

    def check_goal_feasible(self, draft: Draft):
        try:
            super().check_goal_feasible(draft)
        finally:
            male, female = self.obj("male"), self.obj("female")
            if male.states[AttachedTo].get_value(female):
                male.states[AttachedTo].set_value(female, False)
            og.sim.step_physics()
            self.world.keep_still(self.objects.values())

    def extra_checks(self, draft: Draft):
        draft.summary["view"] = {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt}
