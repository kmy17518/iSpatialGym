"""Held-object tasks: object scaling (``object_scaling-0``) and mental rotation (``mental_rotation-0``).

Init: the robot holds the target in the right gripper at the *canonical held pose* -- pad centre on the head camera's
optical axis at ``init_depth``, approach axis horizontal pointing to the robot's left, closing axis vertical. The
28-D hold configuration is solved once per template and shared by all instances (only the base pose differs).
The grasp is a validated library grasp, established with assisted grasping exactly as in the grasp library run
(gravity off -> close -> attached -> gravity on -> still held).

Goal (object scaling): the object at another depth on the same camera ray, orientation unchanged
(``pose_targets[target]``: frame = head camera, ``position_metric = depth``).
Goal (mental rotation): the object rotated about a sampled axis through its root, position unchanged
(``orientation_metric = full``). Both goals are checked for IK feasibility (right arm, else right arm + torso) and
for self-occlusion / full visibility in the head camera at init and goal.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.robot_utils import CONTACT_CENTER_Z
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft, TaskSampler
from ispatialgym.sampling.tro_writer import pose_target

import omnigibson as og  # noqa: E402
from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

ARM = "right"


class HeldObjectSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.hull = None
        self.grasps = []
        self.q_hold_rel: Optional[th.Tensor] = None  # 28-D with zero base joints, at the hold trunk
        self.eef_in_base: Optional[G.Pose] = None
        self.room = None
        self.grasp_stats: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        req = self.scene_spec.fixed.get("floor")
        self.room = req.get("room") if isinstance(req, dict) else (self.scene_spec.rooms[0] if self.scene_spec.rooms else None)

    def prepare_bindings(self):
        self.hull = self.world.local_hull(self.obj("target"))
        self.radius = float(self.hull.norm(dim=1).max())

    def prepare_template(self):
        choice = self.choices["target"]
        try:
            key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, ["floating", "tabletop"])
            self.grasps = GL.load_grasps(key)
            self.template_meta["grasp_key"] = key
        except (FileNotFoundError, ValueError) as e:
            raise Reject("spec", f"no grasp library entry for the target: {e}")
        self._solve_hold()
        self.floor_z = float(self.obj("floor").aabb[1][2])
        # scene obstacles for the held-object clearance test (everything but floors / ceilings)
        self.obstacles = [(o.aabb[0].clone(), o.aabb[1].clone()) for o in self.world.scene.objects
                          if o.category not in ("floors", "ceilings") and o.name not in self.world.spawned and o is not self.robot]
        self._precompute_options()

    def _solve_hold(self):
        """Canonical hold configuration: pad centre on the optical axis at ``init_depth``, approach to the robot's left,
        closing axis vertical (rolled about the approach axis in 15 deg steps until IK succeeds)."""
        import omnigibson.utils.transform_utils as T

        depth0 = float(self.gen("init_depth", 0.55))
        x, y, yaw = self.kit.world_base_pose2d()
        left = G.rotate(G.quat_from_yaw(yaw), th.tensor([0.0, 1.0, 0.0]))
        z = left / th.linalg.norm(left)
        # (depth, trunk) candidates: the spec depth at the JoyLo default trunk first. The head camera sits ~0.3 m ahead of
        # the base and its axis stays fairly level (JoyLo tilt is clamped by torso_joint3), so 0.55 m along the axis is
        # ~0.75 m ahead of the base -- beyond the right arm; shorter depths and lower / tilted trunks are tried next.
        depths = [depth0] + [d for d in self.gen("init_depth_fallbacks", [0.5, 0.45, 0.4, 0.35]) if d < depth0]
        trunks = [tuple(map(float, t)) for t in self.gen("hold_trunks", [[0.5, 0.0], [0.5, -0.2], [0.5, -0.4], [0.5, -0.6], [0.25, -0.4], [0.25, -0.6],
                                                                          [0.0, -0.6], [0.75, -0.4], [0.75, -0.6], [0.5, 0.2]])]
        solution = None
        tried = 0
        pads = []
        rolls = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
        # one batched IK call per trunk configuration (every call with new locked joints costs seconds): all depths and
        # all 26 orientations of a trunk go into the same batch; candidates are ranked depth-first (largest depth, then
        # the smallest roll) and the first collision-free solution wins
        for translate, tilt in trunks:
            q0 = self.kit.q_at_base_pose(self.kit.q_config(arms="ready", translate=translate, tilt=tilt), x, y, yaw)
            cam = self.kit.camera_pose(q0)
            axis = self.world.camera_model.optical_axis(cam)
            cands = []
            for depth in depths:
                pad = G.t(cam[0]) + axis * depth
                pads.append((depth, translate, tilt, [round(float(v), 2) for v in G.relative((pad, G.IDENTITY_QUAT), (th.tensor([x, y, self.floor_z_of_robot()]), G.quat_from_yaw(yaw)))[0]]))
                for roll_deg in rolls:
                    for up_sign in (-1.0, 1.0):
                        y_axis = th.tensor([0.0, 0.0, up_sign])
                        y_axis = G.rotate(G.quat_about_axis(z, math.radians(roll_deg)), y_axis)
                        y_axis = y_axis - th.dot(y_axis, z) * z
                        y_axis = y_axis / th.linalg.norm(y_axis)
                        x_axis = th.cross(y_axis, z, dim=0)
                        quat = T.mat2quat(th.stack([x_axis, y_axis, z], dim=1))
                        eef = pad - G.rotate(quat, th.tensor([0.0, 0.0, CONTACT_CENTER_Z]))
                        cands.append((depth, roll_deg, (eef, quat), pad))
            tried += len(cands)
            ok, q_out = self.kit.ik_batch_solve([c[2] for c in cands], q0, arm=ARM)
            for (depth, roll_deg, pose, pad), s, q in zip(cands, ok.tolist(), q_out):
                if not s or not bool(self.kit.collision_free(q, margin=0.01)[0]):
                    continue
                solution = (depth, translate, tilt, roll_deg, q.clone(), pose, cam, pad)
                break
            if solution:
                break
        if solution is None:
            raise Reject("hold_pose", f"no IK solution for the canonical held pose ({tried} eef poses over {len(depths)} depths x {len(trunks)} trunk "
                                      f"configurations; pad centre in the base frame per (depth, translate, tilt): {pads})")
        depth, translate, tilt, roll_deg, q, eef_world, cam, pad = solution
        self.hold_depth = depth
        base_pose = (th.tensor([x, y, self.floor_z_of_robot()]), G.quat_from_yaw(yaw))
        self.eef_in_base = G.relative(eef_world, base_pose)
        self.pad_in_base = G.relative((pad, eef_world[1]), base_pose)
        q_rel = q.clone()
        q_rel[[self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
        self.q_hold_rel = q_rel
        self.cam_in_base = G.relative(cam, base_pose)
        log.info("%s: canonical hold at trunk (%.2f, %.2f), roll %d deg, depth %.2f", self.spec.name, translate, tilt, roll_deg, depth)
        self.template_meta["hold"] = {"depth": depth, "translate": translate, "tilt": tilt, "roll_deg": roll_deg,
                                      "eef_in_base": [self.eef_in_base[0].tolist(), self.eef_in_base[1].tolist()]}

    def floor_z_of_robot(self) -> float:
        return float(self.robot.get_position_orientation()[0][2])

    # ------------------------------------------------------------------ instance geometry

    def _base_pose3d(self, base2d) -> G.Pose:
        x, y, yaw = base2d
        return th.tensor([x, y, self.floor_z_of_robot()]), G.quat_from_yaw(yaw)

    def _in_room(self, xy) -> bool:
        if self.room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == self.room

    def _sample_base(self, rng) -> Tuple[float, float, float]:
        pts = self.kit.random_free_points(rng, 30, room_filter=self._in_room)
        for xy in pts:
            yaw = float(rng.uniform(-math.pi, math.pi))
            base2d = (float(xy[0]), float(xy[1]), yaw)
            q = self.kit.q_at_base_pose(self.q_hold_rel, *base2d)
            if not bool(self.kit.collision_free(q, margin=0.02)[0]):
                continue
            pad_w = G.compose(self._base_pose3d(base2d), self.pad_in_base)[0]
            if not self._clear(pad_w, self.radius + 0.10):
                continue
            # free floor ahead so that the goal motions (and the arm) have room
            fwd = th.tensor([math.cos(yaw), math.sin(yaw)])
            if not self.kit.trav_free(xy[:2] + fwd * 0.6, eroded=True):
                continue
            return base2d
        raise Reject("no_stance", "no collision-free base pose with clearance for the held object")

    def _clear(self, centre: th.Tensor, radius: float) -> bool:
        for lo, hi in self.obstacles:
            d = th.clamp(lo - centre, min=0.0) + th.clamp(centre - hi, min=0.0)
            if float(th.linalg.norm(d)) < radius:
                return False
        return True

    def _object_pose(self, base2d, grasp) -> G.Pose:
        eef_w = G.compose(self._base_pose3d(base2d), self.eef_in_base)
        return G.compose(eef_w, G.invert(grasp.pose))

    def _visible_whole(self, cam: G.Pose, obj_pose: G.Pose) -> bool:
        return self.frustum_ok(cam, self.hull_corners_world(self.hull, obj_pose), 0.05)

    def _goal_candidates(self, rng, cam: G.Pose, obj_pose: G.Pose) -> List[Tuple[G.Pose, Dict[str, Any]]]:
        """Goal object poses (world, at the template's reference base) with their descriptions."""
        raise NotImplementedError

    #: whether the goal may use the torso (moves the head camera; the depth target of object scaling is defined in the
    #: camera frame, so scaling keeps the camera where it is)
    ALLOW_TORSO = True

    def _precompute_options(self):
        """Per library grasp: whether the held object is fully in view at the canonical hold, and which goal candidates
        are IK-feasible (right arm, else right arm + torso). Everything is expressed in the base frame, so the options
        hold for any base pose (world collisions are re-checked per instance)."""
        rng = np.random.default_rng(self.ctx.base_seed * 13 + 5)
        ref = self.kit.world_base_pose2d()
        base3 = self._base_pose3d(ref)
        cam = G.compose(base3, self.cam_in_base)
        q_hold = self.kit.q_at_base_pose(self.q_hold_rel, *ref)
        self.options: Dict[str, List[Dict[str, Any]]] = {}
        n_cand = int(self.gen("goal_candidates", 24))
        stats = {"grasps": len(self.grasps), "held_out_of_view": 0, "candidates": 0, "goal_out_of_view": 0, "ik_failed": 0, "feasible": 0}
        # gather every (grasp, goal candidate) first, screen them in two batches (arm, then arm + torso for the arm
        # failures) and run the exact solver only on the survivors
        entries = []  # (grasp, goal, info)
        for g in self.grasps:
            obj_pose = self._object_pose(ref, g)
            if not self._visible_whole(cam, obj_pose):
                stats["held_out_of_view"] += 1
                continue
            for goal, info in self._goal_candidates(rng, cam, obj_pose)[:n_cand]:
                entries.append((g, goal, info))
        stats["candidates"] = len(entries)
        eefs = [G.compose(goal, g.pose) for g, goal, _ in entries]
        # the batched solutions are used directly (same collision checker as the exact solver): exact torso IK costs
        # ~0.7 s per call, the batch a few ms per pose
        sols: List[Optional[Tuple[th.Tensor, str]]] = [None] * len(entries)
        with self.kit.ignoring([self.obj("target")]):
            arm_ok, arm_q = self.kit.ik_batch_solve(eefs, q_hold, arm=ARM)
            for i, ok in enumerate(arm_ok.tolist()):
                if ok:
                    sols[i] = (arm_q[i].clone(), f"{ARM}_arm")
            if self.ALLOW_TORSO:
                idx = [i for i, s in enumerate(sols) if s is None]
                if idx:
                    t_ok, t_q = self.kit.ik_batch_solve([eefs[i] for i in idx], q_hold, arm=ARM, torso=True)
                    for j, i in enumerate(idx):
                        if bool(t_ok[j]):
                            sols[i] = (t_q[j].clone(), f"{ARM}_arm_torso")
        for (g, goal, info), sol in zip(entries, sols):
            if sol is None:
                stats["ik_failed"] += 1
                continue
            q_goal, emb = sol
            # torso solutions move the head camera: visibility and the depth target refer to the goal-time camera
            cam_goal = self.kit.camera_pose(q_goal)
            if not self._visible_whole(cam_goal, goal):
                stats["goal_out_of_view"] += 1
                continue
            q_rel = q_goal.clone()
            q_rel[[self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
            self.options.setdefault(g.name, []).append({"goal_in_base": G.relative(goal, base3), "cam_goal_in_base": G.relative(cam_goal, base3),
                                                        "info": info, "emb": emb, "q_rel": q_rel})
            stats["feasible"] += 1
        self.template_meta["goal_options"] = {**stats, "grasps_with_options": len(self.options)}
        if not self.options:
            raise Reject("goal_infeasible", f"no library grasp has a visible held pose with an IK-feasible goal ({stats})")
        log.info("%s: %d grasps with goal options (%s)", self.spec.name, len(self.options), stats)

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname = self.bindings.one("target").name
        base2d = self._sample_base(rng)
        base3 = self._base_pose3d(base2d)
        q_init = self.kit.q_at_base_pose(self.q_hold_rel, *base2d)
        cam = G.compose(base3, self.cam_in_base)
        # grasps whose assisted grasp attached before are preferred (per-template statistics), ties in random order
        names = list(self.options)
        names = [names[i] for i in rng.permutation(len(names))]
        names.sort(key=lambda n: self.grasp_stats.get(n, {}).get("failed", 0) - 2 * self.grasp_stats.get(n, {}).get("ok", 0))
        chosen = None
        for gname in names[:4]:
            g = next(x for x in self.grasps if x.name == gname)
            opts = self.options[gname]
            for oi in rng.permutation(len(opts))[:4]:
                opt = opts[int(oi)]
                q_goal = self.kit.q_at_base_pose(opt["q_rel"], *base2d)
                if not bool(self.kit.collision_free(q_goal, margin=0.01)[0]):
                    continue
                goal = G.compose(base3, opt["goal_in_base"])
                chosen = (g, self._object_pose(base2d, g), goal, dict(opt["info"]), q_goal, opt["emb"], G.compose(base3, opt["cam_goal_in_base"]))
                break
            if chosen:
                break
        if chosen is None:
            raise Reject("goal_infeasible", "no precomputed goal option is collision free at this base pose")
        g, obj_pose, goal, info, q_goal, emb, cam_goal = chosen
        if "axis_base" in info:
            info["axis_world"] = G.rotate(G.quat_from_yaw(base2d[2]), th.tensor(info["axis_base"], dtype=th.float32)).tolist()
        d.object_poses[tname] = obj_pose
        d.goal_poses[tname] = goal
        d.robot_q = q_init
        d.base_pose2d = base2d
        d.settle_free = []
        d.extras["grasps"] = [{"agent": AGENT_INSTANCE, "arm": ARM, "object": tname, "grasp_key": g.key, "grasp_name": g.name,
                               "object_pose_in_eef_frame": {"position": G.invert(g.pose)[0].tolist(), "orientation": G.invert(g.pose)[1].tolist()}}]
        d.extras["goal_robot_q"] = q_goal
        d.summary.update({"grasp": {"key": g.key, "name": g.name}, "goal": info, "feasibility": {"embodiment": emb}})
        d.pose_targets[tname] = self._pose_target(cam_goal, goal)
        return d

    def _pose_target(self, cam: G.Pose, goal: G.Pose) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------ pipeline hooks

    def apply_draft(self, draft: Draft):
        tname = self.bindings.one("target").name
        target = self.obj("target")
        self.kit.apply(draft.robot_q, draft.base_pose2d)
        pose = draft.object_poses[tname]
        target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
        target.disable_gravity()
        target.wake()  # sleeping bodies keep a stale render transform after a physics-side teleport
        target.keep_still()
        gname = draft.extras["grasps"][0]["grasp_name"]
        stats = self.grasp_stats.setdefault(gname, {"ok": 0, "failed": 0})
        try:
            self._establish_grasp(target, pose)
        except Reject:
            stats["failed"] += 1
            raise
        stats["ok"] += 1
        # the realised object pose (assisted grasping pulls the object into the pads, up to a few cm / 0.2 rad) and the
        # closed finger positions; the goal is re-derived from the realised eef->object transform so that the object at
        # the goal is exactly where the hand at the goal configuration carries it
        p, q = target.get_position_orientation()
        draft.object_poses[tname] = (p.clone(), q.clone())
        draft.robot_q = self.kit.q_current()
        eef = tuple(x.float() for x in self.robot.eef_links[ARM].get_position_orientation())
        rel = G.relative((p.float(), q.float()), eef)
        draft.extras["grasps"][0]["object_pose_in_eef_frame"] = {"position": rel[0].tolist(), "orientation": rel[1].tolist()}
        draft.extras["grasps"][0]["finger_q"] = self.robot.get_joint_positions()[self.robot.gripper_control_idx[ARM]].tolist()
        q_goal = G.t(draft.extras["goal_robot_q"])
        eef_goal = self.kit.eef_pose(q_goal, ARM)
        goal = G.compose(eef_goal, rel)
        draft.goal_poses[tname] = goal
        cam_goal = self.kit.camera_pose(q_goal)
        draft.pose_targets[tname] = self._pose_target(cam_goal, goal)
        draft.summary["goal_realised"] = {"position": goal[0].tolist(), "orientation": goal[1].tolist(), "eef_offset_deg": math.degrees(G.quat_angle(rel[1], G.invert(self._grasp_of(draft).pose)[1]))}

    def _grasp_of(self, draft: Draft):
        name = draft.extras["grasps"][0]["grasp_name"]
        return next(g for g in self.grasps if g.name == name)

    def _establish_grasp(self, target, pose: G.Pose):
        state = ActionState(self.robot)
        state.hold_current()
        state.set_gripper(ARM, close=False)
        # controller steps without rendering or observation reads (see SimWorld.control_step); assisted grasping runs
        # inside the simulator step, so this is the grasp library's procedure minus the image pipeline
        for _ in range(8):
            target.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
            target.keep_still()
            self.world.control_step(state.action())
        state.set_gripper(ARM, close=True)
        attached = False
        for _ in range(int(self.gen("grasp_close_steps", 60))):
            if self.robot._ag_obj_in_hand[ARM] is None:
                target.keep_still()
            self.world.control_step(state.action())
            if self.robot._ag_obj_in_hand[ARM] is target:
                attached = True
                break
        if not attached:
            raise Reject("grasp_failed", "assisted grasping did not attach the object")
        target.enable_gravity()
        self.world.control_step(state.action(), n=30)
        if self.robot._ag_obj_in_hand[ARM] is not target:
            raise Reject("grasp_failed", "object dropped after gravity was re-enabled")
        self.robot.keep_still()
        self.world.full_step(render=True)  # sync transforms / caches after the non-rendering control steps
        self.robot.keep_still()
        self.kit.sync_world()

    def _release(self):
        try:
            if self.robot._ag_obj_in_hand.get(ARM) is not None:
                self.robot.release_grasp_immediately(ARM)
        except Exception:
            pass
        state = ActionState(self.robot)
        state.hold_current()
        state.set_gripper(ARM, close=False)
        self.world.control_step(state.action(), n=10)

    def on_reject(self, err: Reject):
        self._release()
        super().on_reject(err)

    def teardown(self):
        # an object still held when the template ends would leave the robot's assisted-grasp state pointing at an
        # object that is about to be removed (the next spawn's load_state then fails)
        self._release()

    def extra_checks(self, draft: Draft):
        target = self.obj("target")
        occ = self._self_occlusion(target)
        if occ["ratio"] > float(self.gen("max_self_occlusion", 0.6)):
            raise Reject("self_occlusion", f"init self-occlusion {occ['ratio']:.2f} ({occ['visible_pixels']} of {occ['reference_pixels']} px visible)")
        cam = self.world.camera_pose()
        if not self._visible_whole(cam, tuple(x.float() for x in target.get_position_orientation())):
            raise Reject("out_of_view", "held object not fully inside the image after grasping")
        draft.summary["init_self_occlusion"] = occ

    def _self_occlusion(self, target) -> Dict[str, Any]:
        vis = self.world.object_pixels(target)
        ref = self.world.object_pixels(target, hidden=[self.robot])
        if ref == 0:
            # the object is in view by construction: a zero reference means the instance-segmentation label map does
            # not know the (recently spawned) object yet; refresh it and measure again
            self.world.ensure_segmented(target)
            vis = self.world.object_pixels(target)
            ref = self.world.object_pixels(target, hidden=[self.robot])
        if ref == 0:
            raise Reject("seg_missing", f"{target.name} has no instance-segmentation pixels even with the robot hidden")
        return {"visible_pixels": int(vis), "reference_pixels": int(ref), "ratio": 1.0 - vis / max(ref, 1)}

    def check_goal_feasible(self, draft: Draft):
        """Goal = robot at the IK solution with the object in hand: collision free (checked when the option was built),
        whole object in view, self-occlusion within bounds and ``pose_reached`` true there.

        Robot joints and object pose are written together *before* the single physics step that refreshes link poses:
        stepping with the fingers teleported away from the object would let the assisted-grasp joint yank the object
        (and release it), and that joint is what `grasped` checks. `grasped` itself is not re-evaluated at the goal.
        """
        from omnigibson.object_states import PoseReached

        tname = self.bindings.one("target").name
        target = self.obj("target")
        state = self.world.snapshot()
        try:
            # the holonomic base is encoded in the six virtual joints (the articulation root never moves): keep their
            # current values so that only the arm / trunk change
            q_goal = G.t(draft.extras["goal_robot_q"]).clone()
            base_idx = [self.kit.jidx[n] for n in M.BASE_VIRTUAL_JOINTS]
            q_goal[base_idx] = self.robot.get_joint_positions()[base_idx]
            goal = draft.goal_poses[tname]
            self.robot.set_joint_positions(q_goal)
            self.robot.set_joint_velocities(th.zeros_like(q_goal))
            self.robot.keep_still()
            target.set_position_orientation(G.t(goal[0]), G.t(goal[1]))
            target.keep_still()
            og.sim.step_physics()
            self.robot.keep_still()
            target.keep_still()
            self.world.render(3)
            cam = self.world.camera_pose()
            obj_now = tuple(x.float() for x in target.get_position_orientation())
            if not self._visible_whole(cam, obj_now):
                cam_fk = self.kit.camera_pose(G.t(draft.extras["goal_robot_q"]))
                uv, _ = self.world.camera_model.project(cam, self.hull_corners_world(self.hull, obj_now))
                raise Reject("goal_out_of_view", f"object leaves the image at the goal (sim cam {[round(v, 3) for v in cam[0].tolist()]}, fk cam "
                                                 f"{[round(v, 3) for v in cam_fk[0].tolist()]}, obj {[round(v, 3) for v in obj_now[0].tolist()]} vs goal "
                                                 f"{[round(float(v), 3) for v in goal[0]]}, uv range {[round(float(v)) for v in uv.min(dim=0).values]}..{[round(float(v)) for v in uv.max(dim=0).values]})")
            occ = self._self_occlusion(target)
            if occ["ratio"] > float(self.gen("max_self_occlusion", 0.6)):
                raise Reject("goal_self_occlusion", f"goal self-occlusion {occ['ratio']:.2f}")
            self.configure_targets(draft)
            st = target.states[PoseReached]
            if not bool(st.get_value()):
                raise Reject("goal_infeasible", f"pose_reached false at the goal configuration (errors {st.get_error()})")
            # full goal expression at this configuration (no extra sim step: the assisted grasp must not be disturbed)
            _, status = self.compiled.check_goal(self._evaluate)
            report = self.goal_report(status)
            failed = [c["bddl"] for c in report if not c["satisfied"] and "pose_settled" not in c["bddl"]]
            if [f for f in failed if "grasped" not in f]:
                raise Reject("goal_infeasible", f"unsatisfied at the goal configuration: {failed}")
            if failed:  # AG bookkeeping after the teleport; the grasp itself was established in simulation at init
                log.warning("%s: 'grasped' reads false right after the goal teleport: %s", self.spec.name, failed)
            draft.summary["goal_check"] = {"goal": report}
            draft.summary["goal_self_occlusion"] = occ
        finally:
            self.world.restore(state)
            self.robot.keep_still()
            target.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self._release()
        target = self.obj("target")
        self.world.park(target, list(self.world.spawned).index(target.name))
        og.sim.step_physics()
        return path


class ObjectScalingSampler(HeldObjectSampler):
    ALLOW_TORSO = False  # the depth target lives in the head-camera frame: the camera must not move with the goal

    def _goal_candidates(self, rng, cam, obj_pose):
        """Goal depths on a 5 cm grid over ``goal_depth_range`` at least ``min_depth_change`` from the init depth (near
        depths first: they are the reachable ones), each jittered by up to 1 cm."""
        axis = self.world.camera_model.optical_axis(cam)
        d0 = float(th.dot(G.t(obj_pose[0]) - G.t(cam[0]), axis))
        lo, hi = self.gen("goal_depth_range", [0.35, 0.85])
        min_change = float(self.gen("min_depth_change", 0.10))
        depths = [float(d) for d in th.arange(lo, hi + 1e-6, 0.05).tolist() if abs(float(d) - d0) >= min_change]
        depths.sort(key=lambda d: abs(d - d0))
        out = []
        for d1 in depths:
            d1 = min(max(d1 + float(rng.uniform(-0.01, 0.01)), lo), hi)
            goal = (G.t(obj_pose[0]) + axis * (d1 - d0), G.t(obj_pose[1]).clone())
            out.append((goal, {"init_depth": d0, "goal_depth": d1}))
        return out

    def _pose_target(self, cam, goal):
        p_cam = G.relative((G.t(goal[0]), G.IDENTITY_QUAT), cam)[0]  # goal root position in the camera frame
        return pose_target(position=p_cam.tolist(), position_metric="depth", position_tolerance=float(self.gen("depth_tolerance", 0.03)),
                           frame=f"{AGENT_INSTANCE}::{self.world.cam_key}")

    def refine_targets_from_goal(self, draft: Draft):
        pass


class MentalRotationSampler(HeldObjectSampler):
    def _goal_candidates(self, rng, cam, obj_pose):
        """Rotations about ``goal_candidates`` random axes (uniform on the sphere, or ``generation.axes`` expressed in the
        robot base frame) by angles in ``goal_angle_range_deg``; the object root stays where it is."""
        axes = self.gen("axes")
        lo, hi = self.gen("goal_angle_range_deg", [30.0, 120.0])
        base_q = G.quat_from_yaw(self.kit.world_base_pose2d()[2])
        out = []
        for _ in range(int(self.gen("goal_candidates", 24))):
            if axes:
                ax = G.rotate(base_q, th.tensor(axes[int(rng.integers(len(axes)))], dtype=th.float32))
            else:
                v = rng.normal(size=3)
                ax = th.tensor(v / np.linalg.norm(v), dtype=th.float32)
            ang = math.radians(float(rng.uniform(lo, hi)))
            dq = G.quat_about_axis(ax, ang)
            goal = (G.t(obj_pose[0]).clone(), G.qmul(dq, G.t(obj_pose[1])))
            out.append((goal, {"axis_base": G.rotate(G.qinv(base_q), ax).tolist(), "angle_deg": math.degrees(ang)}))
        return out

    def _pose_target(self, cam, goal):
        return pose_target(orientation=G.t(goal[1]).tolist(), orientation_metric="full",
                           orientation_tolerance=float(self.gen("orientation_tolerance", 0.15)),
                           equivalences=self.gen("orientation_equivalences", []))

    def refine_targets_from_goal(self, draft: Draft):
        pass
