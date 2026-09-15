"""Camera Relocalization oracle.

Goal: reproduce a target head-camera pose. The R1Pro head camera is rigid on ``torso_link4``, so the
camera pose is a function of the planar base pose (x, y, yaw) and the two JoyLo trunk scalars
(translate, tilt). :func:`solve_viewpoint` inverts this map: it grid-searches (translate, tilt) so
that the base pose implied by ``goal = base * T_base->camera(translate, tilt)`` is planar (no
roll / pitch / height), then refines locally. The oracle then executes ``set_torso`` and
``navigate_to`` (order chosen for collision-freeness) and finishes with a small corrective
navigation on the residual error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch as th

import omnigibson.utils.transform_utils as T

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.poses import Pose, as_tensor, compose, invert, quat_angle
from ispatialgym.oracle.skills import Skills
from ispatialgym.robot.conventions import camera_offset_in_zed, torso_q  # noqa: F401  (moved to robot/conventions, F5)

log = logging.getLogger(__name__)


@dataclass
class Viewpoint:
    x: float
    y: float
    yaw: float
    translate: float
    tilt: float
    residual: float  # planar-consistency residual (m + rad) of the solution

    @property
    def pose2d(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)


def camera_pose_from_state(skills: Skills, q_full: th.Tensor, cam_in_zed: Pose) -> Pose:
    zed = skills.backend.fk([M.HEAD_CAMERA_LINK], q_full=q_full, emb_name="whole_body")[M.HEAD_CAMERA_LINK]
    return compose(zed, cam_in_zed)


def solve_viewpoint(
    skills: Skills,
    goal_cam: Pose,
    cam_in_zed: Optional[Pose] = None,
    translates: Optional[th.Tensor] = None,
    tilts: Optional[th.Tensor] = None,
    refine_iters: int = 2,
) -> Viewpoint:
    """Base pose + trunk scalars reproducing the world camera pose ``goal_cam``.

    For every (translate, tilt) candidate the camera pose relative to ``base_link`` is obtained by
    FK (base joints zeroed); the implied base pose ``goal * inv(T_base_cam)`` must be planar
    (roll = pitch = 0, z = 0). The candidate with the smallest planarity residual is refined on a
    finer grid; its planar part gives (x, y, yaw).
    """
    be = skills.backend
    robot = skills.robot
    cam_in_zed = camera_offset_in_zed(robot) if cam_in_zed is None else cam_in_zed
    goal_p, goal_q = as_tensor(goal_cam[0]), as_tensor(goal_cam[1])
    limits = skills._joint_limits
    q0 = be.q_full()
    j = be._jidx
    for n in M.BASE_VIRTUAL_JOINTS:
        q0[j[n]] = 0.0  # base_link == root frame

    translates = th.linspace(0.0, 2.0, 41) if translates is None else as_tensor(translates)
    tilt_lo, tilt_hi = float(limits[0][M.TRUNK_TILT_JOINT_INDEX]), float(limits[1][M.TRUNK_TILT_JOINT_INDEX])
    tilts = th.linspace(-0.8, 0.8, 33) if tilts is None else as_tensor(tilts)
    root_pose = be.root_pose()

    def evaluate(tr_vals, ti_vals):
        qs = []
        keys = []
        for tr in tr_vals.tolist():
            for ti in ti_vals.tolist():
                qk = q0.clone()
                qk[robot.trunk_control_idx] = torso_q(tr, ti, limits)
                qs.append(qk)
                keys.append((tr, ti))
        qs = th.stack(qs)
        pos, quat = be.fk_batch(M.HEAD_CAMERA_LINK, qs, emb_name="whole_body")  # world (root at origin -> root frame)
        best = None
        for k, (tr, ti) in enumerate(keys):
            # camera pose in the root frame (root == base_link here)
            zed_root = T.relative_pose_transform(pos[k], quat[k], root_pose[0], root_pose[1])
            cam_root = compose(zed_root, cam_in_zed)  # T_base_cam
            base = compose((goal_p, goal_q), invert(cam_root))  # implied base pose (world)
            roll, pitch, yaw = T.quat2euler(base[1]).tolist()
            res = abs(roll) + abs(pitch) + abs(float(base[0][2]))
            if best is None or res < best.residual:
                best = Viewpoint(float(base[0][0]), float(base[0][1]), float(yaw), tr, ti, res)
        return best

    best = evaluate(translates, tilts)
    step_tr, step_ti = float(translates[1] - translates[0]) if len(translates) > 1 else 0.05, float(tilts[1] - tilts[0]) if len(tilts) > 1 else 0.05
    for _ in range(refine_iters):
        tr_vals = th.clamp(th.linspace(best.translate - step_tr, best.translate + step_tr, 9), 0.0, 2.0)
        ti_vals = th.clamp(th.linspace(best.tilt - step_ti, best.tilt + step_ti, 9), tilt_lo - 2.0, tilt_hi + 2.0)
        best = evaluate(tr_vals, ti_vals)
        step_tr, step_ti = step_tr / 4.0, step_ti / 4.0
    return best


class CameraRelocalizationOracle:
    """Move base + trunk so that the head camera reproduces ``goal_cam`` (world pose, xyzw)."""

    def __init__(self, skills: Skills, pos_tol: float = 0.03, ang_tol: float = 0.05, max_corrections: int = 2):
        self.skills = skills
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.max_corrections = max_corrections
        self.cam_in_zed = camera_offset_in_zed(skills.robot)

    def camera_pose(self) -> Pose:
        key = [k for k in self.skills.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
        p, q = self.skills.robot.sensors[key].get_position_orientation()
        return p.float(), q.float()

    def error(self, goal_cam: Pose) -> Tuple[float, float]:
        p, q = self.camera_pose()
        return float(th.linalg.norm(p - as_tensor(goal_cam[0]))), quat_angle(q, goal_cam[1])

    def run(self, goal_cam: Pose) -> Iterator[th.Tensor]:
        sk = self.skills
        vp = solve_viewpoint(sk, goal_cam, self.cam_in_zed)
        log.info("camera relocalization: viewpoint base=(%.2f, %.2f, %.2f) translate=%.2f tilt=%.2f residual=%.4f", vp.x, vp.y, vp.yaw, vp.translate, vp.tilt, vp.residual)
        if vp.residual > 0.05:
            log.warning("goal camera pose is not exactly reproducible by (base, translate, tilt); residual %.3f", vp.residual)
        # trunk and base ordering, as in Skills.go_to_stance
        yield from sk.go_to_stance((vp.pose2d, vp.translate))
        yield from sk.set_torso(translate=vp.translate, tilt=vp.tilt)
        for _ in range(self.max_corrections):
            dp, da = self.error(goal_cam)
            log.info("camera relocalization: residual error %.3f m / %.3f rad", dp, da)
            if dp < self.pos_tol and da < self.ang_tol:
                break
            # re-solve from the achieved state (compensates tracking error) and correct the base only
            vp2 = solve_viewpoint(sk, goal_cam, self.cam_in_zed, translates=th.tensor([vp.translate]), tilts=th.tensor([vp.tilt]), refine_iters=0)
            yield from sk.navigate_to(vp2.pose2d)
        yield from sk.settle()
