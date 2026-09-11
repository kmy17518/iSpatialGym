"""Oracles for the two held-object tasks (the grasp is part of the instance's initial state).

* Object Scaling: translate the held object toward / away from the head camera until it reaches
  the target position (apparent size = focal_length * size / distance, so a target apparent size
  maps to a target distance along the camera->object ray). Orientation is kept fixed.
* Mental Rotation: rotate the held object about its own centre until its orientation matches the
  target. The eef orbits the object centre so the object stays in view.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional

import torch as th

import omnigibson.utils.transform_utils as T

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.poses import as_tensor, quat_angle
from ispatialgym.oracle.skills import SkillError, Skills

log = logging.getLogger(__name__)


def _head_camera_position(skills: Skills) -> th.Tensor:
    key = [k for k in skills.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
    return skills.robot.sensors[key].get_position_orientation()[0].float()


def _translate_with_fallbacks(sk: Skills, arm: str, delta: th.Tensor) -> Iterator[th.Tensor]:
    """Straight-line translation of the eef (fixed orientation); if the arm alone cannot track it,
    let the torso participate, and as a last resort use planned motions to the same end pose."""
    p, q = sk.eef_pose(arm)
    attempts = (
        ("cartesian arm", lambda: sk.move_eef_cartesian(arm, delta_pos=delta, delta_frame="world")),
        ("cartesian arm+torso", lambda: sk.move_eef_cartesian(arm, delta_pos=delta, delta_frame="world", torso=True)),
        ("planned arm", lambda: sk.move_eef(arm, (p + delta, q))),
        ("planned arm+torso", lambda: sk.move_eef(arm, (p + delta, q), torso=True)),
    )
    last = None
    for label, make in attempts:
        try:
            yield from make()
            return
        except SkillError as e:
            log.info("translate: %s failed (%s)", label, e)
            last = e
    raise SkillError("planning", "held-object translation not feasible", cause=str(last))


def _rotate_with_fallbacks(sk: Skills, arm: str, obj, delta_quat: th.Tensor) -> Iterator[th.Tensor]:
    """Rotate the held object about its centre; same fallback chain as :func:`_translate_with_fallbacks`."""
    p, q = sk.eef_pose(arm)
    piv = obj.aabb_center.float()
    Rc = T.quat2mat(delta_quat)
    goal = (piv + Rc @ (p - piv), T.mat2quat(Rc @ T.quat2mat(q)))
    attempts = (
        ("orbit arm", lambda: sk.rotate_eef(arm, delta_quat=delta_quat, pivot="object", frame="world")),
        ("orbit arm+torso", lambda: sk.rotate_eef(arm, delta_quat=delta_quat, pivot="object", frame="world", torso=True)),
        ("planned arm", lambda: sk.move_eef(arm, goal)),
        ("planned arm+torso", lambda: sk.move_eef(arm, goal, torso=True)),
    )
    last = None
    for label, make in attempts:
        try:
            yield from make()
            return
        except SkillError as e:
            log.info("rotate: %s failed (%s)", label, e)
            last = e
    raise SkillError("planning", "held-object rotation not feasible", cause=str(last))


class ObjectScalingOracle:
    def __init__(self, skills: Skills, arm: Optional[str] = None, pos_tol: float = 0.01):
        self.skills = skills
        self.arm = arm or skills.arm
        self.pos_tol = pos_tol

    def target_from_distance(self, distance: float) -> th.Tensor:
        """Object position at ``distance`` metres from the head camera along the current camera->object ray."""
        obj = self.skills.held_object(self.arm)
        if obj is None:
            raise SkillError("no_object", "object scaling requires a held object")
        cam = _head_camera_position(self.skills)
        centre = obj.aabb_center.float()
        ray = centre - cam
        return cam + ray / th.linalg.norm(ray) * float(distance)

    def target_from_scale(self, apparent_scale: float) -> th.Tensor:
        """Object position whose apparent size is ``apparent_scale`` times the current one."""
        cam = _head_camera_position(self.skills)
        obj = self.skills.held_object(self.arm)
        d = float(th.linalg.norm(obj.aabb_center.float() - cam))
        return self.target_from_distance(d / float(apparent_scale))

    def run(self, target_position, max_corrections: int = 2) -> Iterator[th.Tensor]:
        """Translate the held object (fixed orientation) so that its centre reaches ``target_position``."""
        sk = self.skills
        obj = sk.held_object(self.arm)
        if obj is None:
            raise SkillError("no_object", "object scaling requires a held object")
        target = as_tensor(target_position)
        for _ in range(1 + max_corrections):
            delta = target - obj.aabb_center.float()
            if float(th.linalg.norm(delta)) < self.pos_tol:
                break
            log.info("object scaling: moving held object by %s", delta.numpy().round(3))
            yield from _translate_with_fallbacks(sk, self.arm, delta)
        yield from sk.settle()


class MentalRotationOracle:
    def __init__(self, skills: Skills, arm: Optional[str] = None, ang_tol: float = 0.03, max_step_deg: float = 60.0):
        self.skills = skills
        self.arm = arm or skills.arm
        self.ang_tol = ang_tol
        self.max_step = math.radians(max_step_deg)

    def run(self, target_quat, max_corrections: int = 2) -> Iterator[th.Tensor]:
        """Rotate the held object about its centre until its orientation matches ``target_quat`` (xyzw)."""
        sk = self.skills
        obj = sk.held_object(self.arm)
        if obj is None:
            raise SkillError("no_object", "mental rotation requires a held object")
        tq = as_tensor(target_quat)
        for _ in range(1 + max_corrections):
            _, oq = obj.get_position_orientation()
            oq = oq.float()
            ang = quat_angle(oq, tq)
            if ang < self.ang_tol:
                break
            # rotation taking the current object orientation to the target, in world axes
            delta = T.quat_multiply(tq, T.quat_inverse(oq))
            # large rotations in chunks so every chunk stays inside the wrist's range (waypoints are
            # slerped inside rotate_eef; chunking only bounds how far a single Cartesian path goes)
            n_chunks = max(1, int(math.ceil(ang / self.max_step)))
            aa = T.quat2axisangle(delta) / n_chunks
            chunk = T.axisangle2quat(aa)
            log.info("mental rotation: %.1f deg remaining in %d chunk(s)", math.degrees(ang), n_chunks)
            for _c in range(n_chunks):
                yield from _rotate_with_fallbacks(sk, self.arm, obj, chunk)
        yield from sk.settle()
