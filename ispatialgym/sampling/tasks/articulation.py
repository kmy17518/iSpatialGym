"""Configuration matching / articulation (``configuration_matching-1`` closed -> open, ``-2`` open -> closed).

Roles: ``artic`` (spawned non-scene object placed on the fixed ``table``, or a fixed scene object), ``floor``.
Handle geometry comes from the articulation pass of the grasp library (``artic/handles/<key>.json``): contact
point, pinch axis and approach direction in the object frame at the closed configuration.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.grasp_library import library_root
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler
from ispatialgym.sampling.tro_writer import full_pose_target

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.object_states import Open  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


class ArticulationSampler(TaskSampler):
    #: handles sit well above the table top, so standing trunk translates are tried first
    DEFAULT_REACH = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.direction = "open" if self.definition_id == 1 else "close"
        self.scene_object = "artic" in [r.name for r in self.spec.fixed_roles()]
        self.top = None
        self.stance = None
        self.cells = None
        self.hull = None
        self.handle = None
        self.joint_name = None
        self.stable_angles: List[float] = []

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        if self.scene_object:
            cands = self.fixed_candidates("artic", ())
            if not cands:
                raise Reject("spec", "no scene articulated object bound")
            self.bind_scene_object("artic", cands[0])
        else:
            self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def _load_handle(self, obj):
        role = self.spec.roles["artic"]
        key = role.options.get("handle_key", f"{obj.category}__{obj.model}")
        path = os.path.join(library_root(), "artic", "handles", f"{key}.json")
        if not os.path.exists(path):
            raise Reject("spec", f"no handle annotation for {key} at {path}")
        with open(path) as f:
            data = json.load(f)
        joint_name = role.options.get("joint")
        joints = [j for j in data["joints"] if joint_name is None or j["joint"] == joint_name]
        if not joints:
            raise Reject("spec", f"joint {joint_name!r} not in handle annotation {key}")
        j = joints[0]
        handles = [h for h in j.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([j["fallback"]] if j.get("fallback") else [])
        if not handles:
            raise Reject("spec", f"no graspable handle for {key}/{j['joint']}")
        self.joint_name = j["joint"]
        self.handle = handles[0]
        self.joint_info = j

    def prepare_bindings(self):
        artic = self.obj("artic")
        self._load_handle(artic)
        self.hull = self.world.local_hull(artic)
        if self.scene_object:
            self._prepare_scene_object(artic)
            return
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def _handle_eef_poses(self, obj_pose: G.Pose, pregrasp: float = 0.10) -> List[G.Pose]:
        """[pre-grasp, grasp] eef poses for the annotated handle: approach along ``approach_dir``, fingers closing
        along ``pinch_axis`` (object frame, closed configuration)."""
        h = self.handle
        contact = G.compose(obj_pose, (th.tensor(h["contact"], dtype=th.float32), G.IDENTITY_QUAT))[0]
        approach = G.rotate(obj_pose[1], th.tensor(h["approach_dir"], dtype=th.float32))
        pinch = G.rotate(obj_pose[1], th.tensor(h["pinch_axis"], dtype=th.float32))
        z = approach / th.linalg.norm(approach)
        y = pinch - th.dot(pinch, z) * z
        y = y / th.linalg.norm(y)
        x = th.cross(y, z, dim=0)
        quat = T.mat2quat(th.stack([x, y, z], dim=1))
        eef = contact - z * 0.012  # pad centre 1.2 cm behind the eef origin along the approach
        return [(eef - z * pregrasp, quat), (eef, quat)]

    def _facing_dir_local(self) -> th.Tensor:
        """Object-frame horizontal direction that should point at the robot: the outward normal of the moving face
        (doors / drawers); for lids (vertical outward normal) the direction from the hinge line to the object centre,
        so the lid swings away from the robot."""
        outward = th.tensor(self.joint_info.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
        if abs(float(outward[2])) > 0.9:
            axis = th.tensor(self.joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
            hinge = th.tensor(self.joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            perp = th.cross(axis, th.tensor([0.0, 0.0, 1.0]), dim=0)
            centre = self.hull.mean(dim=0)
            if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
                perp = -perp
            outward = perp
        outward[2] = 0.0
        return outward / th.linalg.norm(outward)

    def _try_support(self, table):
        self.kit.sync_world()
        top = self.support_top(table)
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        facing = self._facing_dir_local()
        # half extents of the (yaw-aligned) object along the robot's depth / lateral directions
        pts = self.hull[:, :2]
        d_half = float((pts @ facing[:2]).abs().max())
        side = th.tensor([-float(facing[1]), float(facing[0])])
        l_half = float((pts @ side).abs().max())
        depth_range = tuple(self.gen("depth_range", [d_half + 0.03, d_half + 0.20]))
        # wide objects need the camera high and behind the edge (standing trunk) to fit in the image while the handle
        # stays within arm reach; try the stance stand-offs in order
        errors = []
        stance, cells = None, None
        for stand_off in self.gen("stand_offs", [0.45, 0.55]):
            try:
                st = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=self.scene_spec.fixed.get("table_front_edge"), focus_z=float(ext[2]) / 2,
                                           stand_off=float(stand_off), prefer_translate=float(self.gen("view_prefer_translate", 0.0)))
            except Reject as e:
                errors.append(str(e))
                continue
            # the handle is worked with the right arm, whose reachable strip lies to the robot's right of the centre
            # line: offset the object that way (lateral_center < 0 = robot's right)
            region = self.region_cells(top, st, depth_range, float(self.gen("lateral_half", 0.15)), margin=min(d_half, l_half) + 0.02,
                                       lateral_center=float(self.gen("lateral_center", -0.15)))
            if len(region) == 0:
                errors.append(f"stand_off {stand_off}: placement region empty")
                continue
            # object yaw such that ``facing`` points at the robot (= -inward); keep cells where the yawed object is fully
            # supported and its whole hull is inside the image (exact hull, not a bounding disc: the object is wide)
            face_yaw = math.atan2(float(-st.inward[1]), float(-st.inward[0])) - math.atan2(float(facing[1]), float(facing[0]))
            keep, n_vis = [], 0
            for c in region:
                pose = self._pose_on(c, face_yaw, top.z)
                if not self.frustum_ok(st.view.cam_pose, self.hull_corners_world(self.hull, pose), float(self.gen("view_margin", 0.03))):
                    continue
                n_vis += 1
                if top.polygon_supported(self.world.footprint(self.hull, pose)):
                    keep.append(c)
            if keep:
                stance, cells, self.face_yaw = st, th.stack(keep), face_yaw
                break
            errors.append(f"stand_off {stand_off}: {n_vis} visible of {len(region)} cells, none supported+visible (view translate {st.view.translate})")
        if stance is None:
            raise Reject("support", "; ".join(errors))
        # handle reachability per coarse cell (exact handle poses, one yaw). Wide objects are viewed from further back
        # than the arm reaches, so the manipulation stance may be the viewing stance driven straight forward by up to
        # ``approach_forward`` (the oracle drives before reaching); the first (forward, translate) that works is kept.
        coarse = self.coarse_cells(cells, float(self.gen("reach_resolution", 0.06)), top.resolution)
        translates = list(self.gen("reach_translates", self.DEFAULT_REACH))
        pairs = [self._handle_eef_poses(self._pose_on(xy, self.face_yaw, top.z)) for xy in coarse]
        ok_cells, tr_of, fwd_of = [], [], []
        for fwd in self.gen("approach_forward", [0.0, 0.10, 0.20]):
            st_f = self._shifted(stance, float(fwd))
            if fwd and not self.kit.trav_free(st_f.base_xy):
                continue
            for tr in self.stance_free_translates(st_f, translates):
                todo = [i for i in range(len(pairs)) if i not in ok_cells]
                if not todo:
                    break
                feas = self.kit.pairs_feasible([pairs[i] for i in todo], self.reach_seed(st_f, tr), ignore_for_grasp=[table])
                for i, s in zip(todo, feas.tolist()):
                    if s:
                        ok_cells.append(i)
                        tr_of.append(float(tr))
                        fwd_of.append(float(fwd))
            if ok_cells:
                break
        if not ok_cells:
            h0 = pairs[0][-1][0].tolist() if pairs else None
            raise Reject("unreachable", f"handle not reachable from any of {len(coarse)} cells (e.g. handle at {h0}, base {stance.base_pose2d}, "
                                        f"translates {self.stance_free_translates(stance, translates)})")
        self.top, self.stance = top, stance
        self.cells = coarse[ok_cells]
        self.reach_translates = tr_of
        self.reach_forward = fwd_of
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_reachable_cells": len(ok_cells), "handle": self.handle, "joint": self.joint_name,
                                   "approach_forward": sorted(set(fwd_of))})

    def _shifted(self, stance, forward: float):
        """Copy of ``stance`` with the base driven ``forward`` metres toward the support."""
        if not forward:
            return stance
        from ispatialgym.sampling.tasks.base import Stance

        bxy = stance.base_xy + stance.inward * forward
        base = (float(bxy[0]), float(bxy[1]), stance.base_pose2d[2])
        return Stance(stance.edge_name, stance.edge_mid, stance.inward, stance.tangent, base, self.kit.q_at_base_pose(stance.q_stance, *base), stance.view)

    def _pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        quat = G.quat_from_yaw(yaw)
        pos = th.tensor([float(xy[0]), float(xy[1]), z + self.world.bottom_offset(self.hull, quat) + 0.003])
        return pos, quat

    def _prepare_scene_object(self, artic):
        """Scene object: stance in front of the opening side with the handle reachable (may drive straight forward)."""
        self.kit.sync_world()
        pose = tuple(x.float() for x in artic.get_position_orientation())
        outward = G.rotate(pose[1], th.tensor(self.joint_info.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32))
        outward[2] = 0.0
        outward = outward / th.linalg.norm(outward)
        centre = artic.aabb_center.float()
        handle_pairs = self._handle_eef_poses(pose)
        best = None
        for dist in self.gen("scene_distances", [0.6, 0.7, 0.8, 0.9, 1.0]):
            for lateral in (0.0, -0.2, 0.2, -0.35):
                tangent = th.tensor([-float(outward[1]), float(outward[0])])
                base_xy = centre[:2] + outward[:2] * dist + tangent * lateral
                yaw = math.atan2(float(-outward[1]), float(-outward[0]))
                if not self.kit.trav_free(base_xy):
                    continue
                q_base = self.kit.q_at_base_pose(self.kit.q_config(arms="untucked"), float(base_xy[0]), float(base_xy[1]), yaw)
                if not bool(self.kit.collision_free(q_base, margin=0.01)[0]):
                    continue
                views = self.kit.solve_view(q_base, centre, translates=th.tensor(self.gen("view_translates", [0.0, 0.25, 0.5, 0.75])), top_k=12)
                for v in views:
                    q = self.kit.with_trunk(q_base, v.translate, v.tilt)
                    if not bool(self.kit.collision_free(q, margin=0.01)[0]):
                        continue
                    cam = self.kit.camera_pose(q)
                    lo, hi = artic.aabb
                    if not self.frustum_ok(cam, G.aabb_corners(lo, hi), 0.03):
                        continue
                    # reach: from the stance or after driving forward up to 0.3 m
                    for fwd in (0.0, 0.15, 0.3):
                        bxy = base_xy - outward[:2] * fwd
                        stance_q = self.kit.q_at_base_pose(self.kit.q_config(arms=self.REACH_ARMS), float(bxy[0]), float(bxy[1]), yaw)
                        tr_ok = None
                        for tr in self.gen("reach_translates", self.DEFAULT_REACH):
                            if self.kit.reachable(handle_pairs, self.kit.with_trunk(stance_q, tr, 0.0), ignore_for_last=[artic]) is not None:
                                tr_ok = tr
                                break
                        if tr_ok is not None:
                            best = {"base_pose2d": (float(base_xy[0]), float(base_xy[1]), yaw), "q": q, "view": v, "forward": fwd, "translate": tr_ok}
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            raise Reject("unreachable", f"no stance in front of {artic.name} reaches the handle")
        self.scene_stance = best
        self.template_meta.update({"scene_stance": {k: (v if not isinstance(v, th.Tensor) else v.tolist()) for k, v in best.items() if k not in ("q", "view")}})

    def prepare_template(self):
        artic = self.obj("artic")
        self.joint = artic.joints[self.joint_name]
        self.lower, self.upper = float(self.joint.lower_limit), float(self.joint.upper_limit)
        self.stable_angles = [self.upper] if os.environ.get("ISG_SKIP_STABLE_PROBE") else self._stable_angles(artic)
        if not self.stable_angles:
            raise Reject("articulation", f"{artic.name}/{self.joint_name} holds no open angle passively")
        self.template_meta["stable_open_fractions"] = [(a - self.lower) / (self.upper - self.lower) for a in self.stable_angles]

    def _stable_angles(self, artic) -> List[float]:
        """Joint positions (fractions of the range from the spec) at which the moving link holds still for 120 physics
        steps; the upper limit is always tested last (resting on the stop)."""
        state = self.world.snapshot()
        lo, hi = self.gen("open_fraction_range", [0.5, 0.95])
        fracs = [float(f) for f in th.linspace(lo, hi, 7).tolist()] + [1.0]
        tol = self._joint_tol()
        out = []
        idx = list(artic.joints).index(self.joint_name)
        try:
            if not self.scene_object:
                # test on a stance cell (gravity direction is what matters, not the exact place)
                pose = self._pose_on(self.cells[0], self.face_yaw, self.top.z)
                artic.set_position_orientation(*pose)
                self.world.activate(artic)
                self.world.step_physics(30, still=[self.robot])
            for f in fracs:
                q = self.lower + f * (self.upper - self.lower)
                self._set_joint(artic, q)
                self.world.step_physics(120, still=[self.robot])
                q_after = float(artic.get_joint_positions()[idx])
                log.info("%s: joint %s at fraction %.2f (%.3f) -> %.3f after 120 steps", self.spec.name, self.joint_name, f, q, q_after)
                if abs(q_after - q) <= 0.5 * tol:
                    out.append(q)
        finally:
            self.world.restore(state)
            if not self.scene_object:
                self.world.park(artic, list(self.world.spawned).index(artic.name))
        return out

    def _set_joint(self, artic, q: float):
        qs = artic.get_joint_positions().clone()
        qs[list(artic.joints).index(self.joint_name)] = q
        artic.set_joint_positions(qs)
        artic.set_joint_velocities(th.zeros_like(qs))
        artic.keep_still()

    def _joint_tol(self) -> float:
        tols = self.gen("tolerances", {})
        return float(tols.get("revolute" if self.joint.is_revolute else "prismatic", 0.10 if self.joint.is_revolute else 0.02))

    # ------------------------------------------------------------------ instance

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        aname = self.bindings.one("artic").name
        artic = self.obj("artic")
        joint_names = list(artic.joints)
        q_closed = artic.get_joint_positions().clone()
        for n, j in artic.joints.items():
            q_closed[joint_names.index(n)] = float(j.lower_limit)
        q_open_val = self.stable_angles[int(rng.integers(len(self.stable_angles)))]
        if self.direction == "open":
            q_init, q_goal_val = q_closed.clone(), q_open_val
        else:
            q_init = q_closed.clone()
            q_init[joint_names.index(self.joint_name)] = q_open_val
            q_goal_val = float(self.joint.lower_limit)
        q_goal = q_closed.clone()
        q_goal[joint_names.index(self.joint_name)] = q_goal_val
        d.object_joints[aname] = q_init
        d.goal_joints[aname] = q_goal
        tol = self._joint_tol()
        d.articulation_targets[aname] = {"joints": {n: {"position": float(q_goal[joint_names.index(n)]), "tolerance": tol} for n in joint_names}}
        if self.scene_object:
            st = self.scene_stance
            d.robot_q = st["q"]
            d.base_pose2d = st["base_pose2d"]
            d.settle_free = []
            d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_open": q_open_val, "scene_stance": {k: v for k, v in st.items() if k in ("forward", "translate")}})
            return d
        k = int(rng.integers(len(self.cells)))
        pose = self._pose_on(self.cells[k], self.face_yaw + float(rng.uniform(-0.1, 0.1)), self.top.z)
        if not self.top.polygon_supported(self.world.footprint(self.hull, pose)):
            raise Reject("target_unsupported")
        if not self.frustum_ok(self.stance.view.cam_pose, self.hull_corners_world(self.hull, pose), 0.05):
            raise Reject("out_of_view")
        st_reach = self._shifted(self.stance, self.reach_forward[k])
        tr_ok = self.reachable_poses(st_reach, self._handle_eef_poses(pose), [self.reach_translates[k]] + list(self.gen("reach_translates", self.DEFAULT_REACH)))
        if tr_ok is None:
            raise Reject("grasp_unreachable", "handle not reachable at the sampled pose")
        d.object_poses[aname] = pose
        d.robot_q = self.stance.q_stance
        d.base_pose2d = self.stance.base_pose2d
        d.pose_targets[aname] = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_open": q_open_val, "handle_translate": tr_ok,
                          "approach_forward": self.reach_forward[k], "view": {"translate": self.stance.view.translate, "tilt": self.stance.view.tilt}})
        return d

    def apply_draft(self, draft: Draft):
        super().apply_draft(draft)
        # hold the moving link at its init position during settling (position control is not available on objects)
        artic = self.obj("artic")
        aname = self.bindings.one("artic").name
        self._set_joint(artic, float(draft.object_joints[aname][list(artic.joints).index(self.joint_name)]))

    def after_settle(self, draft: Draft):
        aname = self.bindings.one("artic").name
        if aname in draft.pose_targets:
            p, q = self.obj("artic").get_position_orientation()
            draft.pose_targets[aname] = full_pose_target(p, q, self.tol("position", 0.03), self.tol("orientation", 0.20))
        q_now = float(self.obj("artic").get_joint_positions()[list(self.obj("artic").joints).index(self.joint_name)])
        q_want = float(draft.object_joints[aname][list(self.obj("artic").joints).index(self.joint_name)])
        if abs(q_now - q_want) > 0.5 * self._joint_tol():
            raise Reject("articulation", f"init joint drifted from {q_want:.3f} to {q_now:.3f}")

    def apply_goal(self, draft: Draft):
        super().apply_goal(draft)

    def refine_targets_from_goal(self, draft: Draft):
        aname = self.bindings.one("artic").name
        artic = self.obj("artic")
        q = artic.get_joint_positions()
        names = list(artic.joints)
        tgt = draft.articulation_targets[aname]["joints"]
        for n in names:
            if abs(float(q[names.index(n)]) - tgt[n]["position"]) > 0.5 * tgt[n]["tolerance"]:
                raise Reject("goal_unstable", f"joint {n} drifted to {float(q[names.index(n)]):.3f} (target {tgt[n]['position']:.3f})")
        if aname in draft.pose_targets:
            super().refine_targets_from_goal(draft)
