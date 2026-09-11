"""Camera relocalization (``camera_relocalization-0``): move from a tucked init base pose to the base / trunk
configuration whose head-camera pose is the goal (shown as the goal image).

Roles: fixed ``floor`` only. Everything is sampled on the BEHAVIOR-eroded traversability map of the loaded floor,
restricted to the spec's room; candidates are filtered analytically (Euclidean, geodesic, heading, collision) and
only survivors are rendered (two seg_instance renders per candidate pair).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, TaskSampler
from ispatialgym.sampling.tro_writer import pose_target

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")


class CameraRelocalizationSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.room = None

    def bind_fixed_roles(self):
        self.bind_floor()
        req = self.scene_spec.fixed.get("floor")
        self.room = req.get("room") if isinstance(req, dict) else (self.scene_spec.rooms[0] if self.scene_spec.rooms else None)

    def prepare_template(self):
        self.q_init_rel = self.kit.q_config(arms="tucked", torso_zero=True)
        self.template_meta["init_config"] = "tucked (all joints zero)"

    def _in_room(self, xy) -> bool:
        if self.room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == self.room

    def _config(self, base2d, translate: float, tilt: float, arms: str = "tucked") -> th.Tensor:
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
            chosen = (e, heading, base1, tr1, ti1, q1, cam1, geo, goal_vis, init_vis, shared)
            break
        if chosen is None:
            raise Reject("goal_sampling", f"no goal candidate passed collision / geodesic / visibility ({n_rendered} rendered)")
        e, heading, base1, tr1, ti1, q1, cam1, geo, goal_vis, init_vis, shared = chosen
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
        return d

    def apply_draft(self, draft: Draft):
        self.kit.apply(draft.robot_q, draft.base_pose2d)

    def check_goal_feasible(self, draft: Draft):
        state = self.world.snapshot()
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
            self.configure_targets(draft)
            holds, status = self.binary_goal_holds()
            if not holds:
                failed = [c["bddl"] for c in self.goal_report(status) if not c["satisfied"]]
                raise Reject("goal_infeasible", f"unsatisfied at the goal configuration: {failed}")
            draft.summary["goal_fk_error"] = {"position_m": err_p, "orientation_rad": err_o}
        finally:
            self.world.restore(state)
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def refine_targets_from_goal(self, draft: Draft):
        pass
