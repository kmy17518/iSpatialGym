"""Loading an eval instance into an OmniGibson environment.

:class:`InstanceLoader` builds the environment of an instance's template (R1Pro in the JoyLo action space, the evaluation
camera profile ``isg-v1``: head camera 1080 x 1080 with a 40 mm aperture, one render per step) and restores the
instance: the TRO's object states and robot state, then the setup records the task does not restore itself
(``constraints``: hinges, joint brakes, masses; ``grasps``: objects held in hand), re-applied after every reset.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import torch as th

from ispatialgym import single_env as SE
from ispatialgym.eval import geometry as G
from ispatialgym.eval.instances import TRO_SUFFIX, template_files
from ispatialgym.eval.tro import TROFile, load_tro
from ispatialgym.robot import r1pro
from ispatialgym.robot.action import ActionState, apply_recorded_targets

log = logging.getLogger(__name__)

EVALUATION_PROFILE = "isg-v1"
DEFAULT_IMAGE_SIZE = 1080
DEFAULT_HEAD_APERTURE_MM = 40.0
DEFAULT_ROBOT_NAME = "robot_r1"
ACTION_FREQUENCY = 30
PHYSICS_FREQUENCY = 120

#: BEHAVIOR's replay camera over the robot's shoulder (``obs["external"][EXTERNAL_CAMERA_NAME]``)
EXTERNAL_CAMERA_NAME = "external_sensor0"
EXTERNAL_CAMERA_HW = (960, 1440)

CONSTRAINT_TYPES = ("revolute", "joint_friction", "mass")
JOINT_BRAKE_KEYS = ("friction", "stiffness", "damping", "max_effort", "lower_limit_rad", "upper_limit_rad")


class LoadError(RuntimeError):
    """An instance that cannot be restored; ``reason`` is ``failed_grasp`` or ``error``."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason
        self.message = message


def resolve_instance(tro_path: str) -> tuple[str, dict[str, str], dict, str, int]:
    """``(instance id, template files, template metadata, activity, definition id)`` of an instance file."""
    tro_path = os.path.abspath(tro_path)
    if not tro_path.endswith(TRO_SUFFIX):
        raise ValueError(f"an instance file ends with {TRO_SUFFIX}, got {tro_path}")
    files = template_files(tro_path)
    with open(files["template_meta"]) as f:
        meta = json.load(f)
    activity, definition_id = meta["problem"].rsplit("-", 1)
    return os.path.basename(tro_path)[: -len(TRO_SUFFIX)], files, meta, activity, int(definition_id)


def robot_name_from_tro(tro: TROFile, default: str = DEFAULT_ROBOT_NAME) -> str:
    """The robot name the instance's sensor keys use (``<robot>:<link>:Camera:0``)."""
    key = (tro.meta.goal_observation or {}).get("sensor_key")
    if not key:
        for target in tro.meta.pose_targets.values():
            if target.link and ":" in target.link:
                key = target.link
                break
            if target.frame and "::" in target.frame:
                key = target.frame.split("::", 1)[1]
                break
    return key.split(":", 1)[0] if key and ":" in key else default


def build_env_config(
    scene_file: str,
    meta: dict,
    tro: TROFile,
    tro_path: str,
    activity: str,
    definition_id: int,
    robot_name: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    external_camera: bool = False,
) -> dict:
    """The environment of an instance, with the robot spawned at the TRO's base pose."""
    robot_kw = {}
    poses = {k.lower(): v for k, v in (tro.robot_poses or {}).items()}
    pose = (poses.get("robot") or poses.get("r1pro") or [None])[0]
    if pose is not None:
        robot_kw = {
            "position": [float(v) for v in pose["position"]],
            "orientation": [float(v) for v in pose["orientation"]],
        }
    sensors = {
        "obs_modalities": ["rgb", "proprio"],
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": image_size, "image_width": image_size}}},
    }
    cfg = {
        "env": {"action_frequency": ACTION_FREQUENCY, "physics_frequency": PHYSICS_FREQUENCY},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": scene_file,
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        "robots": [r1pro.robot_config(robot_name, **sensors, **robot_kw)],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": definition_id,
            "tro_path": tro_path,
            "termination_config": {"max_steps": 10**6},  # the runner owns the step budget
        },
    }
    if external_camera:
        cfg["env"]["external_sensors"] = [
            {
                "sensor_type": "VisionSensor",
                "name": EXTERNAL_CAMERA_NAME,
                "relative_prim_path": f"/controllable__r1pro__{robot_name}/base_link/{EXTERNAL_CAMERA_NAME}",
                "modalities": ["rgb"],
                "sensor_kwargs": {
                    "image_height": EXTERNAL_CAMERA_HW[0],
                    "image_width": EXTERNAL_CAMERA_HW[1],
                    "horizontal_aperture": 40.0,
                },
                "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
                "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
                "pose_frame": "parent",
            }
        ]
    return cfg


def gripper_command_for(robot, arm: str, finger_q) -> float:
    """The smooth gripper command that holds the fingers at ``finger_q``."""
    idx = robot.gripper_control_idx[arm]
    lo, hi = robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]
    q = th.as_tensor(finger_q, dtype=th.float32)
    return float(th.clamp(2.0 * (q.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))


class InstanceLoader:
    """Loads one instance and re-applies its setup records after every reset."""

    def __init__(
        self,
        tro_path: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        head_aperture: float = DEFAULT_HEAD_APERTURE_MM,
        external_camera: bool = False,
    ):
        self.tro_path = os.path.abspath(tro_path)
        self.iid, self.files, self.tmeta, self.activity, self.definition_id = resolve_instance(self.tro_path)
        self.tro = load_tro(self.tro_path)
        self.robot_name = robot_name_from_tro(self.tro)
        self.image_size = int(image_size)
        self.head_aperture = float(head_aperture)
        self.external_camera = bool(external_camera)
        self.env = self.task = self.robot = None
        self.cam_key: str | None = None
        self.physical: dict[str, Any] = {}
        self.report: dict[str, Any] = {}
        self.hinges: list[Any] = []
        self.grasps: list[dict] = []
        self.hold: ActionState | None = None
        self._hinge_records: list[dict] = []
        self._joint_brake_records: list[dict] = []
        self._mass_records: list[dict] = []
        self._articulation_q: dict[str, th.Tensor] = {}

    @property
    def goal_image_path(self) -> str | None:
        """The instance's goal image (the TRO's ``goal_observation.rgb_path``)."""
        path = (self.tro.meta.goal_observation or {}).get("rgb_path")
        return None if not path else os.path.join(os.path.dirname(self.tro_path), path)

    @property
    def tracked(self) -> dict[str, Any]:
        """The task's physical instances except the robot and the floors."""
        from ispatialgym.activity.problem import AGENT_INSTANCE

        return {
            k: v for k, v in self.physical.items() if k != AGENT_INSTANCE and getattr(v, "category", None) != "floors"
        }

    def build_env(self):
        import omnigibson as og
        from omnigibson.macros import gm

        import ispatialgym.tasks  # noqa: F401 - registers ISpatialGymTask
        from ispatialgym.eval.scene_variants import block_trav_map_from_meta, patch_trav_map_from_meta

        for name, value in (
            ("ENABLE_OBJECT_STATES", True),
            ("USE_GPU_DYNAMICS", False),
            ("ENABLE_TRANSITION_RULES", False),
        ):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.cfg = build_env_config(
            self.files["template"],
            self.tmeta,
            self.tro,
            self.tro_path,
            self.activity,
            self.definition_id,
            self.robot_name,
            self.image_size,
            external_camera=self.external_camera,
        )
        self.env = og.Environment(configs=self.cfg)
        self.report["trav_cells_freed"] = patch_trav_map_from_meta(self.env.scene, self.tmeta)
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.task = self.env.task
        self.robot = self.env.scene.robots[0]
        self.report["base_link_mass_kg"] = r1pro.apply_eval_base_mass(self.robot)
        self.cam_key = next((k for k in self.robot.sensors if k.endswith(r1pro.HEAD_CAMERA_SENSOR_SUFFIX)), None)
        if self.cam_key is None:
            raise RuntimeError(f"robot {self.robot.name} has no head camera ({r1pro.HEAD_CAMERA_SENSOR_SUFFIX})")
        self.robot.sensors[self.cam_key].horizontal_aperture = self.head_aperture
        self.report["head_camera_key"] = self.cam_key
        self.report["head_camera_aperture_mm"] = float(self.robot.sensors[self.cam_key].horizontal_aperture)
        self.report["evaluation_profile"] = self.tro.meta.extras.get("evaluation_profile") or EVALUATION_PROFILE
        return self.env

    def restore(self, settle_physics_steps: int = 25):
        """Apply the instance, settle it with the task entities held still, snapshot it as the scene's initial state
        and reset; returns the reset's ``(obs, info)``."""
        import omnigibson as og

        from ispatialgym.eval.group_entity import GroupEntity

        env, task, robot = self.env, self.task, self.robot
        task.load_tro_instance(env, tro=self.tro, settle_steps=0)
        self.physical = {k: v for k, v in task.object_scope.items() if v is not None and not isinstance(v, GroupEntity)}
        extras = self.tro.meta.extras
        if extras.get("attachment_targets"):
            raise LoadError("error", "attachment_targets are not supported by this release")
        self.report["constraints"] = self._restore_constraints(extras.get("constraints") or [])
        self._articulation_q = {
            inst: self.physical[inst].get_joint_positions().clone()
            for inst in self.tro.meta.articulation_targets
            if inst in self.physical
        }
        self.grasps = self._restore_grasps(extras.get("grasps") or [])
        self.report["grasps"] = self.grasps

        # a gripper command must exist or the assisted grasp releases the object at the first physics step
        self.hold = ActionState(robot)
        for g in self.grasps:
            if g.get("finger_q"):
                self.hold.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        recorded = (self.tro.meta.robot_state or {}).get("controller_targets") or {}
        self.report["controller_targets_restored"] = (
            apply_recorded_targets(self.hold, robot, recorded) if recorded else []
        )
        robot.apply_action(self.hold.action())
        for _ in range(settle_physics_steps):
            og.sim.step_physics()
            self._hold_entities()
        for _ in range(2):  # full steps refresh the per-step caches (AABBs, contacts)
            og.sim.step()
            self._hold_entities()
        env.scene.update_initial_file()
        task._reset_tracking()
        obs, info = SE.reset(env)
        self.report["after_reset"] = self.restore_after_reset()
        if self.report["controller_targets_restored"]:
            robot.apply_action(self.hold.action())  # the recorded targets hold until the first action
        return obs, info

    def restore_after_reset(self) -> dict[str, Any]:
        """Re-apply what a ``scene.reset()`` may undo: hinge prims, joint brakes, masses, grasps and articulation joints."""
        import omnigibson as og

        out: dict[str, Any] = {"hinges_recreated": 0, "grasps": []}
        for i, hinge in enumerate(list(self.hinges)):
            prim = og.sim.stage.GetPrimAtPath(hinge.prim_path)
            if prim is None or not prim.IsValid():
                self.hinges[i] = self._create_hinge(self._hinge_records[i])
                out["hinges_recreated"] += 1
        for rec in self._joint_brake_records:
            self._apply_joint_brake(rec)
        for rec in self._mass_records:
            self._apply_mass(rec)
        for g in self.grasps:
            obj = self.physical[g["object"]]
            if self.robot._ag_obj_in_hand.get(g["arm"]) is not obj:
                self._establish(g["arm"], obj, g.get("ag_anchor_in_eef"))
            attached = self.robot._ag_obj_in_hand.get(g["arm"]) is obj
            out["grasps"].append({"arm": g["arm"], "object": g["object"], "attached": attached})
            if not attached:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} could not be restored after the reset")
        for inst, q0 in self._articulation_q.items():
            entity = self.physical[inst]
            if float(th.max(th.abs(entity.get_joint_positions() - q0))) > 1e-3:
                entity.set_joint_positions(q0)
                entity.set_joint_velocities(th.zeros_like(q0))
                out.setdefault("articulation_reset", []).append(inst)
        return out

    def _hold_entities(self) -> None:
        for inst, entity in self.physical.items():
            entity.keep_still()
            q0 = self._articulation_q.get(inst)
            if q0 is not None:
                entity.set_joint_positions(q0)
                entity.set_joint_velocities(th.zeros_like(q0))

    def _bound(self, inst: str, what: str):
        entity = self.physical.get(inst)
        if entity is None:
            raise LoadError("error", f"{what} names {inst!r}, which is not a bound physical instance")
        return entity

    def _restore_constraints(self, records: list[dict]) -> list[dict]:
        import omnigibson as og

        out = []
        self.hinges, self._hinge_records, self._joint_brake_records, self._mass_records = [], [], [], []
        for rec in records:
            kind = rec.get("type", "revolute")
            if kind not in CONSTRAINT_TYPES:
                raise LoadError("error", f"constraint type {kind!r} is not supported by this release")
            if kind == "revolute":
                hinge = self._create_hinge(rec)
                self.hinges.append(hinge)
                self._hinge_records.append(rec)
                out.append(
                    {
                        "type": kind,
                        "child": rec["child"],
                        "parent": rec.get("parent"),
                        "friction": hinge.friction,
                        "damping": hinge.damping,
                    }
                )
            elif kind == "joint_friction":
                out.append(self._apply_joint_brake(rec))
                self._joint_brake_records.append(rec)
            else:
                out.append(self._apply_mass(rec))
                self._mass_records.append(rec)
        if self.hinges:
            og.sim.update_handles()
        return out

    def _create_hinge(self, rec: dict):
        from ispatialgym.eval.constraints import create_hinge, hinge_brake_damping

        child = self._bound(rec["child"], "constraint")
        parent_link = None
        if rec.get("parent"):
            parent = self.physical.get(rec["parent"]) or self.env.scene.object_registry("name", rec["parent"])
            if parent is None:
                raise LoadError("error", f"constraint parent {rec['parent']!r} not found")
            parent_link = parent.links[rec.get("parent_link") or parent.root_link_name]
        friction = rec.get("friction")
        return create_hinge(
            child,
            th.tensor(rec["anchor_position"], dtype=th.float32),
            th.tensor(rec["axis"], dtype=th.float32),
            parent_link=parent_link,
            friction=friction,
            damping=hinge_brake_damping(friction, rec.get("damping")),
        )

    def _apply_joint_brake(self, rec: dict) -> dict:
        from ispatialgym.eval.constraints import apply_joint_brake

        entity = self._bound(rec["object"], "joint_friction constraint")
        if rec["joint"] not in entity.joints:
            raise LoadError("error", f"joint_friction constraint: {rec['object']} has no joint {rec['joint']!r}")
        if rec.get("openable_joints"):
            raise LoadError("error", "joint_friction openable_joints are not supported by this release")
        spec = {k: rec[k] for k in JOINT_BRAKE_KEYS if rec.get(k) is not None}
        return {
            "type": "joint_friction",
            "object": rec["object"],
            "joint": rec["joint"],
            **apply_joint_brake(entity.joints[rec["joint"]], spec),
        }

    def _apply_mass(self, rec: dict) -> dict:
        from ispatialgym.eval.constraints import apply_mass

        entity = self._bound(rec["object"], "mass constraint")
        return {"type": "mass", "object": rec["object"], **apply_mass(entity, float(rec["mass_kg"]), rec.get("link"))}

    def _establish(self, arm: str, obj, anchor_in_eef=None) -> None:
        """The assisted-grasp joint of ``obj`` in ``arm``'s hand, anchored at ``anchor_in_eef`` (eef frame) or the eef."""
        robot = self.robot
        link = obj.root_link_name
        joint_type = robot._get_assisted_grasp_joint_type(obj, link)
        if joint_type is None:
            raise LoadError("failed_grasp", f"{obj.name} cannot be assisted-grasped")
        held = robot._ag_obj_in_hand.get(arm)
        if held is not None and held is not obj:
            robot.release_grasp_immediately(arm)
        if robot._ag_obj_in_hand.get(arm) is not obj:
            eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
            contact = eef[0]
            if anchor_in_eef is not None:
                contact = G.compose_pose(eef, (th.tensor(anchor_in_eef, dtype=th.float32), G.IDENTITY_QUAT))[0]
            robot._establish_grasp(obj, link, arm, contact, joint_type)

    def _restore_grasps(self, records: list[dict]) -> list[dict]:
        """Objects held in hand: each is placed at ``eef * object_pose_in_eef_frame``, the recorded finger joints are
        set and the assisted grasp is attached before the first step; a failed attach fails the load."""
        robot = self.robot
        out = []
        for g in records:
            arm, inst = g["arm"], g["object"]
            obj = self.physical.get(inst)
            if obj is None:
                raise LoadError("failed_grasp", f"grasp record names {inst!r}, which is not a bound physical instance")
            rec = {
                "arm": arm,
                "object": inst,
                "scene_object": obj.name,
                "finger_q": g.get("finger_q"),
                "ag_anchor_in_eef": g.get("ag_anchor_in_eef"),
            }
            ref = g.get("object_pose_in_eef_frame")
            if ref:
                eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
                want = G.compose_pose(
                    eef, (th.tensor(ref["position"], dtype=th.float32), th.tensor(ref["orientation"], dtype=th.float32))
                )
                obj.set_position_orientation(want[0], want[1])
                obj.keep_still()
            if g.get("finger_q"):
                q = robot.get_joint_positions()
                q[robot.gripper_control_idx[arm]] = th.tensor(g["finger_q"], dtype=th.float32)
                robot.set_joint_positions(q)
            self._establish(arm, obj, g.get("ag_anchor_in_eef"))
            if ref:
                eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
                rel = G.relative_pose(eef, tuple(x.float() for x in obj.get_position_orientation()))
                rec["object_in_eef_dev_mm"] = 1000.0 * float(th.linalg.norm(rel[0] - th.tensor(ref["position"])))
                rec["object_in_eef_dev_deg"] = math.degrees(G.quat_angle(rel[1], th.tensor(ref["orientation"])))
            rec["attached"] = robot._ag_obj_in_hand.get(arm) is obj
            if not rec["attached"]:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} did not attach at load")
            out.append(rec)
        return out
