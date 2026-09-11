"""Assembly and validation of iSpatialGym TRO files (``isg-tro-v1``) from a sampled instance."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from omnigibson.utils.config_utils import TorchEncoder

from ispatialgym.eval.targets import SCHEMA_VERSION, TRO_KEY, ISGTroMeta
from ispatialgym.eval.tro import TROFile, validate_tro_against_problem

EVALUATION_PROFILE = "isg-v1"
HEAD_SENSOR_KEY = "robot_r1:zed_link:Camera:0"


def pose_target(position=None, orientation=None, position_metric=None, orientation_metric=None, position_tolerance=None,
                orientation_tolerance=None, frame: str = "world", link: Optional[str] = None, equivalences=None) -> Dict[str, Any]:
    if position_metric is None and position is not None:
        position_metric = "full"
    if orientation_metric is None and orientation is not None:
        orientation_metric = "full"
    return {
        "link": link,
        "frame": frame,
        "position": None if position is None else [float(v) for v in position],
        "orientation": None if orientation is None else [float(v) for v in orientation],
        "position_metric": position_metric,
        "orientation_metric": orientation_metric,
        "position_tolerance": None if position_metric is None else float(position_tolerance),
        "orientation_tolerance": None if orientation_metric is None else float(orientation_tolerance),
        "orientation_equivalences": [[float(v) for v in q] for q in (equivalences or [])],
    }


def full_pose_target(position, orientation, position_tolerance, orientation_tolerance, equivalences=None) -> Dict[str, Any]:
    return pose_target(position, orientation, "full", "full", position_tolerance, orientation_tolerance, equivalences=equivalences)


def build_tro(
    *,
    problem_name: str,
    activity: str,
    definition_id: int,
    task_variant: Optional[str],
    instance_id: str,
    generation_seed: int,
    bddl_sha256: str,
    task_spec_sha256: str,
    scene: Dict[str, Any],
    object_bindings: Dict[str, Dict[str, Any]],
    object_states: Dict[str, Any],
    robot_pose: Dict[str, Any],
    robot_state: Dict[str, Any],
    pose_targets: Optional[Dict[str, Any]] = None,
    groups: Optional[Dict[str, Any]] = None,
    articulation_targets: Optional[Dict[str, Any]] = None,
    grasps=None,
    constraints=None,
    attachment_targets=None,
    initial_occlusion=None,
    layout_template_id=None,
    goal_observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "activity": activity,
        "activity_definition_id": int(definition_id),
        "task_variant": task_variant,
        "instance_id": instance_id,
        "generation_seed": int(generation_seed),
        "evaluation_profile": EVALUATION_PROFILE,
        "bddl_sha256": bddl_sha256,
        "task_spec_sha256": task_spec_sha256,
        "layout_template_id": layout_template_id,
        "scene": scene,
        "object_bindings": object_bindings,
        "robot_state": robot_state,
        "goal_observation": goal_observation
        or {
            "sensor_key": HEAD_SENSOR_KEY,
            "rgb_path": None,
            "rgb_sha256": None,
            "resolution": [1080, 1080],
            "intrinsics": None,
            "camera_pose_world": None,
        },
        "pose_targets": pose_targets or {},
        "groups": groups or {},
        "articulation_targets": articulation_targets or {},
    }
    for key, val in (("grasps", grasps), ("constraints", constraints), ("attachment_targets", attachment_targets),
                     ("initial_occlusion", initial_occlusion)):
        if val:
            meta[key] = val
    tro: Dict[str, Any] = dict(object_states)
    tro["robot_poses"] = {"robot": [robot_pose]}
    tro[TRO_KEY] = meta
    return tro


def validate_and_write(tro: Dict[str, Any], compiled, path: str) -> TROFile:
    """Round-trip the TRO through the JSON encoder, validate it against the compiled problem, write it."""
    text = json.dumps(tro, cls=TorchEncoder, indent=2)
    data = json.loads(text)
    tro_file = TROFile.from_dict(data, path=path)
    validate_tro_against_problem(tro_file.meta, compiled, require_bindings=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return tro_file


def meta_from_dict(d: Dict[str, Any]) -> ISGTroMeta:
    return ISGTroMeta.from_dict(json.loads(json.dumps(d, cls=TorchEncoder)))
