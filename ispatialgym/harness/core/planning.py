"""Arm motion planning for move_eef (needs the simulator's cuRobo backend), as RoboDojo's ``CuroboPlanner.plan_path``, the
planner behind RoboProbe's move_eef, does it: cuRobo's ``MotionPlanner.plan_pose`` from the current joints to the target
pose with RoboDojo's settings (32 IK seeds, 4 trajectory seeds, 5 attempts, no graph planner, self-collision on), against
the slab under the robot's mounting surface (:mod:`world`: R1Pro's floor; RoboDojo's table top). Nothing of the scene is in
it.

One planner per embodiment (``left_arm`` / ``right_arm`` / ``both_arms``: both arms named are planned as one unit, since
R1Pro's arms share a torso), on the oracle's R1Pro model and built on first use; torso, base and every arm joint not
planned are locked at their current values. The world is given in the robot's root frame, so the planners are rebuilt if
the root ever moves."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.harness.core.world import Box, scene_model
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot.conventions import as_tensor

log = logging.getLogger(__name__)

Pose = Tuple[th.Tensor, th.Tensor]


class ArmPlanner:
    def __init__(self, backend, boxes: Sequence[Box], max_attempts: int = 5, num_ik_seeds: int = 32, num_trajopt_seeds: int = 4):
        self.backend = backend
        self.boxes: List[Box] = list(boxes)
        self.max_attempts = int(max_attempts)
        self.num_ik_seeds = int(num_ik_seeds)
        self.num_trajopt_seeds = int(num_trajopt_seeds)
        #: why the last plan failed, for the trace (the model is told only RoboDojo's status)
        self.last_detail: Optional[str] = None
        self._entries: Dict[str, Any] = {}
        self._root: Optional[Pose] = None
        self._scene: Optional[Dict[str, Any]] = None

    def _sync_root(self) -> None:
        rp, rq = (t.detach().cpu().float() for t in self.backend.root_pose())
        if self._root is not None:
            moved = float(th.linalg.norm(rp - self._root[0]))
            turned = 2.0 * math.degrees(math.acos(min(1.0, abs(float(th.dot(rq, self._root[1]))))))
            if moved < 1e-3 and turned < 0.1:
                return
            log.warning("robot root moved %.1f mm / %.2f deg: rebuilding the arm planners' world", 1000 * moved, turned)
            self._entries.clear()
        self._root = (rp, rq)
        self._scene = scene_model(self.boxes, rp.tolist(), rq.tolist()) if self.boxes else None

    def _entry(self, emb_name: str):
        if emb_name in self._entries:
            return self._entries[emb_name]
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
        from curobo.kinematics import Kinematics
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        from ispatialgym.oracle.curobo_planner import _PlannerEntry

        b = self.backend
        t0 = time.time()
        emb = M.EMBODIMENTS[emb_name]
        q = b.q_full()
        lock_values = {n: float(q[b._jidx[n]]) for n in emb.locked_joints}
        robot_dict = M.build_curobo_robot_dict(emb, b.urdf_path, lock_values, b._default_q, b._v1_kin)
        cfg = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=self._scene,
            self_collision_check=True,
            device_cfg=b.device_cfg,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            position_tolerance=b.position_tolerance,
            orientation_tolerance=b.orientation_tolerance,
            use_cuda_graph=b.use_cuda_graph,
            optimizer_collision_activation_distance=b.collision_activation_distance,
        )
        cfg.trajopt_solver_config.interpolation_dt = b.control_dt
        cfg.graph_planner_config = None
        planner = MotionPlanner(cfg)
        entry = _PlannerEntry(embodiment=emb, planner=planner)
        entry.fk = Kinematics(planner.kinematics.config)
        b._fill_lock_info(entry, entry.fk)
        entry.default_criteria = {f: ToolPoseCriteria(device_cfg=b.device_cfg) for f in planner.tool_frames}
        self._entries[emb_name] = entry
        log.info("Built arm planner '%s' (%d world boxes) in %.1fs", emb_name, len(self.boxes), time.time() - t0)
        return entry

    def plan(self, targets_world: Dict[str, Pose], q_start) -> Tuple[Optional[th.Tensor], str]:
        """Joint path ``(T, 28)`` from ``q_start`` bringing each named arm's eef link to its world pose, and RoboDojo's
        status (``Success`` / ``Fail``); :attr:`last_detail` says which stage failed."""
        from curobo.types import GoalToolPose

        self._sync_root()
        b = self.backend
        arms = sorted(targets_world)
        entry = self._entry("both_arms" if len(arms) == 2 else f"{arms[0]}_arm")
        planner = entry.planner
        q_start = as_tensor(q_start)
        b._refresh_locked_entry(entry, q_start)
        goal = GoalToolPose.from_poses({M.EEF_LINK[arm]: b._cu_pose_from_world([targets_world[arm]]) for arm in arms},
                                       ordered_tool_frames=list(planner.tool_frames), num_goalset=1)
        cur = entry.fk.get_active_js(b._js_full(q_start.unsqueeze(0)))
        result = planner.plan_pose(goal, cur, max_attempts=self.max_attempts, enable_graph_attempt=self.max_attempts)
        if result is None:
            self.last_detail = "IK found no goal configuration clear of the robot and the world"
            return None, "Fail"
        if result.success is None or not bool(result.success.any()):
            self.last_detail = "trajectory optimization found no collision-free path to the IK goals"
            return None, "Fail"
        self.last_detail = None
        interp = result.get_interpolated_plan()
        names = list(interp.joint_names)
        return b._full_from_active(entry, interp.position.reshape(-1, len(names)), names, q_start), "Success"
