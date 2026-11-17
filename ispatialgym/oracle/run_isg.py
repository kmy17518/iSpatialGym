"""``run_isg`` -- v2 driver: load one iSpatialGym instance, run the oracle program (or a replay / a policy) through
the executor, record, evaluate (``oracle_primitives_design.md`` §3.8, §6.5; TODO F1 / F3 / F4). One process = one
instance = one episode::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.oracle.run_isg --instance <iid>-tro_state.json \\
        [--sidecar <iid>.oracle.json] --mode oracle|replay|policy [--out <dir>] [--record-view head,...] [--no-qa] [--goal-image]

What it does

1. **Loader (spec §3.14).** The template scene (``template.json``, partial rooms) with the R1Pro JoyLo
   configuration and the evaluation camera profile ``isg-v1`` (head camera 1080 x 1080, 40 mm aperture,
   ``n_render_iterations = 1``); an :class:`ISpatialGymTask` from the template's ``activity_definitions``; the TRO
   applied with ``load_tro_instance`` and then everything the task loader does not restore yet, re-applied after
   every ``scene.reset()`` (:meth:`InstanceLoader.restore_after_reset`): ``constraints`` (the COM hinges of the
   alignment tasks, re-created as PhysX revolute joints; ``joint_positions``: a partner's joints the sampler set at init,
   held like the articulation targets and reported at the end of the episode), ``grasps`` (held-object tasks: the object at
   ``object_pose_in_eef_frame``, the recorded finger joints, the assisted grasp attached before step 0 — an attach
   failure fails the load with ``failed_grasp``), ``attachment_targets`` (the child / parent collision filter, at
   load and therefore also when the policy attaches), ``articulation_targets`` (the target's initial joint values held
   during the settle), plus the trav-map patches of the scene variant (``freed_footprints`` of the template,
   ``blocked_footprints`` of the instance) on the planner map. ``pose_targets`` / ``groups`` /
   ``orientation_equivalences`` / ``evaluation_profile`` are the task's (``configure_targets``). The physics of
   ``run_isg_vec`` (:mod:`~ispatialgym.oracle.sleep_assist`, the same flags and defaults): ``--rigid-sleep-threshold``
   on the free objects of the scene at the build and on the task objects at the restore.
2. **Sidecar** (``--sidecar``, default ``<iid>.oracle.json`` next to the TRO): loaded, validated, ``tro_sha256``
   verified (``--allow-sha-mismatch`` for hand-built sidecars), ``expand(variant, sidecar)`` → the action list.
3. **Executor** (:class:`~ispatialgym.oracle.executor.Executor`): per action checkpoint → execute under the retry
   rules (:class:`~ispatialgym.oracle.qa.Monitors`) and the ``--qa`` metrics (:class:`~ispatialgym.oracle.qa.QA`: a HARD
   metric ends the oracle's demo as failed) → postcondition → one retry → annotate; epilogue: exactly ``--settle-steps``
   no-op steps (reserved in the step budget) → the task's metrics decide (``ispatialgym.eval.metrics``: symbolic / geometric
   success, ``--settle-eval``, ``--success-metric``) → goal image (``--goal-image``: ≥ 3 renders,
   ``goal_observation.camera_pose_world``, profile ``isg-v1``) → TRO goal override with the settled values when within
   tolerance (written back into the TRO, the sidecar's ``tro_sha256`` updated; ``--no-goal-override`` to disable). The
   tolerances are the ``isg-v2`` profile's (``ispatialgym.eval.tolerance_profile``, ``--tolerance-scale``), computed after the
   loader's settle and before the first step. ``--sleep-assist`` after every control step, as in ``run_isg_vec``
   (``meta.sleep_assist``).
4. **Recording** (``--record full|hdf5|video``): ``demo.hdf5`` (BEHAVIOR-replayable) + the LeRobot v3 dataset under
   ``lerobot/`` (``full``), ``<view>_rgb.mp4`` per ``--record-view`` view (``tiled``, ``head``, ``external``,
   ``left_wrist``, ``right_wrist``), ``init_rgb.png`` / ``final_rgb.png`` / ``goal_rgb.png``, ``annotations.json`` (the
   three §6.3 levels, QA verdicts) and ``meta.json`` (``meta.qa``: every ``--qa`` metric with the steps it fired at).
5. Prints one line ``RUN_ISG_RESULT {...}`` and exits with ``0`` success, ``2`` oracle / QA failure (``reason`` one
   of ``qa.CATEGORIES``), ``3`` ``no_oracle`` (no sidecar / no program for the variant; the instance is still loaded
   and its init view rendered), ``4`` ``error`` (traceback in ``meta.json``).

``--mode replay`` feeds the actions of a ``demo.hdf5`` (or a LeRobot episode, ``--replay <dataset root> [--episode N]``)
through the same executor: the demo's action rows and the first ``--settle-steps`` of its settle rows (default all);
``--mode policy`` calls ``--policy module:callable`` as ``policy(obs, info) -> 23-D action`` every step until the geometric
success or ``--max-steps``. Both are thin: no program, no checkpoints, no epilogue, the same recorder; their ``--qa`` metrics
are only logged.
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import importlib
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNIGIBSON_CHANGE_DETECTION_NUMPY", "1")
# the exact opt-ins run_isg_vec defaults on (the fast flow's job env sets them too): the potential reuses the termination's goal
# evaluation, the batched symmetry search, one relation_error per member pair within an evaluation
os.environ.setdefault("ISPATIALGYM_SHARED_GOAL_EVAL", "1")
os.environ.setdefault("ISPATIALGYM_FAST_GEOMETRY", "1")
os.environ.setdefault("ISPATIALGYM_FAST_ASSIGN", "1")

import torch as th  # noqa: E402

from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: E402  (import before omnigibson: warp pinning)
from ispatialgym import single_env as SE  # noqa: E402

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.eval.group_entity import GroupEntity  # noqa: E402
from ispatialgym.eval.targets import TRO_KEY  # noqa: E402
from ispatialgym.eval.tro import TROFile, load_tro  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState, apply_recorded_targets  # noqa: E402
from ispatialgym.oracle.executor import Executor, ExecutorConfig, SimState, apply_goal_override  # noqa: E402
from ispatialgym.oracle.motions import Motions, MotionsConfig  # noqa: E402
from ispatialgym.oracle import physics_only as PO  # noqa: E402
from ispatialgym.oracle import sleep_assist as SA  # noqa: E402
from ispatialgym.fastflags import fast_flags_seen, fast_on, fast_value  # noqa: E402  (2026-10-31 train-fast switches)
from ispatialgym.oracle.programs import expand  # noqa: E402
from ispatialgym.oracle.qa import Monitors, QAConfig  # noqa: E402
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE, VIEWS, EpisodeAnnotations, Recorder, RecorderConfig, dump_json, json_default, save_png  # noqa: E402
from ispatialgym.eval import instances as INST
from ispatialgym.oracle import start_state
from ispatialgym.oracle.sidecar import IDDL_ROOT, Sidecar, SidecarError, family_of_subtask, from_legacy_keys, sha256_of_file  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.tasks import ISpatialGymTask  # noqa: E402, F401  (registers the task type)

log = logging.getLogger("ispatialgym.run_isg")

TRO_SUFFIX = "-tro_state.json"
SIDECAR_SUFFIX = ".oracle.json"
DEFAULT_IMAGE_SIZE = 1080  # evaluation camera profile (isg-v1)
DEFAULT_HEAD_APERTURE_MM = 40.0
DEFAULT_ROBOT_NAME = "robot_r1"
EVALUATION_PROFILE = "isg-v1"

EXIT_SUCCESS, EXIT_ORACLE_FAILED, EXIT_NO_ORACLE, EXIT_ERROR = 0, 2, 3, 4

#: ISPATIALGYM_GC_FREEZE=1 (opt-in): gc.freeze() once the world is built and again after the first planner build, so the
#: collector's later passes skip the long-lived scene / Kit / planner objects (torch runs a full gc.collect() before every
#: CUDA graph capture: ~0.5 s each over the whole heap). Garbage cycles among the frozen objects are never collected; a
#: process keeps one world (run_queue exits for another template).
GC_FREEZE = os.environ.get("ISPATIALGYM_GC_FREEZE") == "1"

#: largest ISG_HINGE_FRICTION_SCALE applied to a wall hinge record (2026-09-22, InstanceLoader._create_hinge): x8 saturated
#: the push arm's drives (0918-000750); the damping scale is not applied at all
HINGE_FRICTION_SCALE_MAX = 2.0

#: §6.3 (b): the family-level description shared by every instance of a task family
FAMILY_DESCRIPTIONS = {
    "camera_relocalization": "Drive and set the torso so that the head camera reproduces the goal view.",
    "object_scaling": "Move the held object toward or away from the camera until it appears at the goal size.",
    "mental_rotation": "Rotate the held object in hand to the goal orientation.",
    "disembedding": "Find the target among the clutter, pick it up and place it on the destination tray.",
    "configuration_matching": "Reproduce the pictured configuration: a block layout, an articulation state, or an attachment.",
    "alignment": "Push the object until it is aligned: upright, level, or along the table axis.",
    "inter_object_relation_satisfaction": "Move the target so that it satisfies the pictured relation with the partner object.",
}


class LoadError(RuntimeError):
    """A loader-contract violation (spec §3.14); ``reason`` is a QA category (``failed_grasp``) or ``error``."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason
        self.message = message


#: i7-attachfix: a TRO object further than this from its TRO root pose after the restore is a load error (the settle under keep_still
#: moves an object < 0.1 mm; the reuse bug moved them 40-549 mm)
TRO_POSE_TOL_M = 0.005
TRO_POSE_TOL_RAD = 0.05


def _quat_angle_xyzw(a, b) -> float:
    d = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, d))


def tro_pose_mismatches(expected: Dict[str, Any], live: Dict[str, Any], pos_tol: float = TRO_POSE_TOL_M,
                        rot_tol: float = TRO_POSE_TOL_RAD) -> List[Dict[str, Any]]:
    """i7-attachfix: the instances whose live root pose (``live[inst] = (pos, quat xyzw)``) is off the TRO's (``expected``, same form)
    by more than ``pos_tol`` (m) or ``rot_tol`` (rad): ``[{"instance", "pos_m", "rot_rad"}]`` (sorted by instance)."""
    out = []
    for inst in sorted(expected):
        if inst not in live:
            continue
        (pe, qe), (pl, ql) = expected[inst], live[inst]
        dp = math.dist([float(x) for x in pe], [float(x) for x in pl])
        da = _quat_angle_xyzw(qe, ql)
        if dp > pos_tol or da > rot_tol:
            out.append({"instance": inst, "pos_m": round(dp, 4), "rot_rad": round(da, 4)})
    return out


def detach_scene_attachments(scene) -> List[Dict[str, str]]:
    """i7-attachfix: remove every BEHAVIOR attachment joint of ``scene`` (each object whose ``AttachedTo`` has a parent):
    ``set_value(parent, False)``, outside any state load. Returns ``[{"child", "parent"}]``."""
    try:
        from omnigibson.object_states import AttachedTo
    except Exception:  # noqa: BLE001 - no simulator (the test stubs)
        return []
    out = []
    for obj in list(getattr(scene, "objects", []) or []):
        st = (getattr(obj, "states", None) or {}).get(AttachedTo) if hasattr(getattr(obj, "states", None), "get") else None
        if st is None or getattr(st, "parent", None) is None:
            continue
        parent = st.parent
        st.set_value(parent, False)
        out.append({"child": obj.name, "parent": parent.name})
    return out


# ==========================================================================================
# instance resolution / env config (from the v1 driver)
# ==========================================================================================


def resolve_instance(path: str) -> Tuple[str, str, dict, str, int]:
    """``(instance id, template dir, template_meta, activity, definition id)`` from a ``*-tro_state.json`` path: a sampler
    output (``<template>/instances/<iid>-tro_state.json``) or an instance of the eval dataset (ispatialgym.eval.instances),
    whose template-level files sit next to its instances folder; for the latter ``template_meta["_paths"]`` names them."""
    path = os.path.abspath(path)
    if not path.endswith(TRO_SUFFIX):
        raise ValueError(f"--instance must point at a *{TRO_SUFFIX} file, got {path}")
    iid = os.path.basename(path)[: -len(TRO_SUFFIX)]
    from ispatialgym.eval.instances import template_files

    dataset = template_files(path)
    if dataset is not None:
        with open(dataset["template_meta"]) as f:
            meta = from_legacy_keys(json.load(f))
        meta["_paths"] = {"scene_file": dataset["template"], "activity_definitions_root": dataset["activity_definitions"], "dataset_root": dataset["root"]}
        # an eval dataset instance is lit as its goal image and training-format renders were: the stock lights (the batched replay ignored
        # collection lighting draws; the dataset's eval_lighting "stock"); ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING=0 restores them
        os.environ.setdefault("ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING", "1")
        activity, did = meta["problem"].rsplit("-", 1)
        return iid, os.path.dirname(dataset["template"]), meta, activity, int(did)
    template_dir = os.path.dirname(os.path.dirname(path))
    meta_path = os.path.join(template_dir, "template_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no template_meta.json next to the instance (expected {meta_path})")
    with open(meta_path) as f:
        meta = from_legacy_keys(json.load(f))
    activity, did = meta["problem"].rsplit("-", 1)
    return iid, template_dir, meta, activity, int(did)


def default_sidecar_path(tro_path: str) -> str:
    return tro_path[: -len(TRO_SUFFIX)] + SIDECAR_SUFFIX


def robot_name_from_tro(tro: TROFile, default: str = DEFAULT_ROBOT_NAME) -> str:
    go = tro.meta.goal_observation or {}
    key = go.get("sensor_key")
    if not key:
        for tgt in tro.meta.pose_targets.values():
            if tgt.link and ":" in tgt.link:
                key = tgt.link
                break
            if tgt.frame and "::" in tgt.frame:
                key = tgt.frame.split("::", 1)[1]
                break
    return key.split(":", 1)[0] if key and ":" in key else default


#: the JoyLo replay over-the-shoulder camera (``--record-view external`` / ``tiled``), BEHAVIOR's size
EXTERNAL_CAMERA_NAME = "external_sensor0"
EXTERNAL_CAMERA_HW = (960, 1440)


def rooms_for_template(meta: dict):
    """The room instances a world of this template loads: the template's ``load_room_instances`` (task rooms + every adjacent
    room), unless ``ISG_ROOMS_OVERRIDE`` names a JSON file ({alias or variant: [rooms] | {"rooms": [...], "variant": ...}},
    campaign rooms 2026-10-28: the rooms the demos can see) with an entry for this template's ``variant`` - the single-env
    evaluator and the vectorized one (``VecInstanceLoader(load_rooms=...)``, which resolves the alias itself) then load the
    same rooms. The chosen list is logged once per template."""
    rooms = meta.get("load_room_instances") or None
    path = os.environ.get("ISG_ROOMS_OVERRIDE")
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            d = d.get("overrides", d) if isinstance(d, dict) else {}
            variant = meta.get("variant")
            hit = None
            for key, e in d.items():
                ev = e.get("variant") if isinstance(e, dict) else None
                if key == variant or ev == variant:
                    hit = e
                    break
            if hit is not None:
                override = hit.get("rooms") if isinstance(hit, dict) else hit
                if override:
                    log.info("rooms override for %s: %s (template: %s)", variant, override, rooms)
                    return [str(r) for r in override]
        except Exception as ex:  # noqa: BLE001
            log.warning("ISG_ROOMS_OVERRIDE %s not applied: %r", path, ex)
    return rooms

#: 2026-11-01 train-fast (PARTIAL_SCENE): the sampler's cropped world dump next to template.json (template meta ``scene_crop``)
PARTIAL_SCENE_FILE = "template-partial_rooms.json"


def scene_crop_of(tmeta: dict) -> Optional[Dict[str, Any]]:
    """2026-11-01 train-fast (PARTIAL_SCENE): the template meta's ``scene_crop`` when the sampler loaded a cropped world (mode other
    than ``full``) and ``ISPATIALGYM_FAST_PARTIAL_SCENE`` is on (default) - the collector then loads exactly the sampler's world
    (``loaded_scene_file`` with ``loaded_rooms``) instead of the full template; None (absent / ``full`` / switch off): today's world."""
    crop = (tmeta or {}).get("scene_crop")
    if not isinstance(crop, dict) or (crop.get("mode") or "full") == "full":
        return None
    if (tmeta or {}).get("eval"):
        # policy eval 2026-11-15: an eval dataset template (scripts/batched_replay/export_eval.py writes the "eval" block) ships the full
        # template.json only, and its goal image / training-format renders come from the full world (batched replay --scene full)
        return None
    return crop if fast_on("PARTIAL_SCENE") else None


def loaded_scene_path(template_dir: str, tmeta: Optional[dict] = None) -> str:
    """The scene file the collector's world is built from: ``template.json``, or (PARTIAL_SCENE) the cropped world dump."""
    if tmeta is None:
        with open(os.path.join(template_dir, "template_meta.json")) as f:
            tmeta = json.load(f)
    crop = scene_crop_of(tmeta)
    return os.path.join(template_dir, crop.get("loaded_scene_file") or PARTIAL_SCENE_FILE) if crop else os.path.join(template_dir, "template.json")


def build_env_config(template_dir: str, meta: dict, tro: TROFile, tro_path: str, activity: str, did: int, robot_name: str, image_size: int, proprio: bool = True,
                     behavior_output: bool = False, scene_crop: Optional[Dict[str, Any]] = None, behavior_proprio: bool = False) -> dict:
    """Evaluator configuration (mirrors ``validate_instances.py``) with the robot spawned at the TRO base pose so the
    holonomic base's virtual joints stay near zero. ``behavior_output``: the robot carries BEHAVIOR replay_obs.py's sensors,
    modalities and proprioception (``behavior_lerobot.robot_overrides``) instead of the evaluation camera profile.
    ``scene_crop`` (2026-11-01 train-fast PARTIAL_SCENE, :func:`scene_crop_of`; None, the default: the full template as before): the
    world is the sampler's cropped one - its dump ``loaded_scene_file`` loaded with its ``loaded_rooms``."""
    scene_file = (meta.get("_paths") or {}).get("scene_file") or os.path.join(template_dir, "template.json")
    rooms = rooms_for_template(meta)  # the template's rooms, or the ISG_ROOMS_OVERRIDE entry of its variant (vec_next)
    if scene_crop:
        scene_file = os.path.join(template_dir, scene_crop.get("loaded_scene_file") or PARTIAL_SCENE_FILE)
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"template meta scene_crop (mode {scene_crop.get('mode')!r}) names {scene_file}, which does not exist "
                                    "(ISPATIALGYM_FAST_PARTIAL_SCENE=0 loads the full template.json)")
        rooms = scene_crop.get("loaded_rooms") or rooms
    robot_kw = {}
    poses = {k.lower(): v for k, v in (tro.robot_poses or {}).items()}
    pose = (poses.get("robot") or poses.get("r1pro") or [None])[0]
    if pose is not None:
        robot_kw = {"position": [float(v) for v in pose["position"]], "orientation": [float(v) for v in pose["orientation"]]}
    if behavior_output:
        from ispatialgym.oracle.behavior_lerobot import robot_overrides

        sensors = robot_overrides()
    else:
        sensors = {"obs_modalities": ["rgb", "proprio"] if proprio else ["rgb"],
                   "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": image_size, "image_width": image_size}}}, **M.camera_overrides()}
        if behavior_proprio:
            # policy eval 2026-11-16 (run_isg --mode policy): the proprioception the training data holds and the policy servers read -
            # BEHAVIOR's 61-D layout (PROPRIOCEPTION_INDICES["R1Pro"], the batched replay's robot_overrides) instead of the robot's
            # default 48-D one; an observation-only change (no physics, controller or camera setting differs)
            from ispatialgym.oracle.behavior_lerobot import ROBOT_PROPRIO_KEYS

            sensors["proprio_obs"] = list(ROBOT_PROPRIO_KEYS)
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": scene_file,  # 2026-11-01 train-fast PARTIAL_SCENE: template.json unless the template is cropped
            "load_room_instances": rooms,
        },
        "robots": [M.joylo_robot_config(robot_name, **sensors, **robot_kw)],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": did,
            "activity_definitions_root": (meta.get("_paths") or {}).get("activity_definitions_root") or os.path.join(template_dir, "activity_definitions"),
            "tro_path": tro_path,
            "termination_config": {"max_steps": 10**6},  # the executor owns the step budget
        },
    }


def executor_mesh_cache_kw(robot, make_obstacle_filter=None) -> Dict[str, int]:
    """ISPATIALGYM_EXECUTOR_MESH_CACHE (opt-in; unset: cuRobo's 8192): capacity of the executor's cuRobo mesh store,
    ``auto`` = the scene's collision meshes (objects other than the robot, not visual-only) + reach_bands'
    MESH_CACHE_HEADROOM, at least MESH_CACHE_MIN, a multiple of 64 (as the sampler's ISPATIALGYM_SAMPLER_MESH_CACHE);
    ``room`` = likewise over the objects the executor's obstacle filter keeps (``make_obstacle_filter()``: what
    update_world loads; the headroom takes objects that enter the task rooms later); or a number. The capacity sizes every
    collision kernel launch; the thread layout can change the order of cuRobo's atomic float sums. In queue mode the
    store serves every instance of the loaded world (removed objects' meshes stay, disabled)."""
    value = os.environ.get("ISPATIALGYM_EXECUTOR_MESH_CACHE", "").strip().lower()
    if not value:
        return {}
    if value not in ("auto", "room"):
        return {"mesh_cache_size": int(value)}
    from ispatialgym.sampling import reach_bands as RB

    keep = make_obstacle_filter() if value == "room" and make_obstacle_filter is not None else None
    n = sum(len(link.collision_meshes) for o in robot.scene.objects
            if o is not robot and not getattr(o, "visual_only", False) and (keep is None or keep(o)) for link in o.links.values())
    cache = max(RB.MESH_CACHE_MIN, 64 * math.ceil((n + RB.MESH_CACHE_HEADROOM) / 64))
    log.info("executor cuRobo mesh store %d for %d scene collision meshes (ISPATIALGYM_EXECUTOR_MESH_CACHE)", cache, n)
    return {"mesh_cache_size": cache}


def task_bounds(scene, rooms: Sequence[str], margin: float = 1.0):
    los, his = [], []
    for o in scene.objects:
        if getattr(o, "category", None) == "floors" and set(o.in_rooms or []) & set(rooms):
            lo, hi = o.aabb
            los.append(lo)
            his.append(hi)
    if not los:
        return None
    lo = th.stack(los).min(dim=0).values - margin
    hi = th.stack(his).max(dim=0).values + margin
    hi[2] = 10.0
    return lo, hi


def make_obstacle_filter(bounds, robot, always: Sequence[str]):
    always = set(always)

    def f(obj):
        if bounds is None or obj is robot or obj.name in always:
            return True
        lo, hi = obj.aabb
        return bool(((hi >= bounds[0]) & (lo <= bounds[1])).all())

    return f


def gripper_command_for(robot, arm: str, finger_q) -> float:
    """``MultiFingerGripperController`` (smooth mode) command holding the fingers at ``finger_q``."""
    idx = robot.gripper_control_idx[arm]
    lo, hi = robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]
    q = th.as_tensor(finger_q, dtype=th.float32)
    return float(th.clamp(2.0 * (q.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))



def _copy_map(m):
    return m.clone() if hasattr(m, "clone") else m.copy()


#: InstanceLoader.attach() loads the scene state taken after build_env before the next instance (queue mode)
QUEUE_SCENE_RESTORE = True
_PROVENANCE: Optional[Dict[str, Any]] = None


def provenance() -> Dict[str, Any]:
    """The code this process runs, for every results row and meta.json: the snapshot (``PROD_SNAP`` of a produce job, else
    the checkout), the iSpatialGym and BEHAVIOR-1K commits and dirty flags of the packages actually imported, whether
    OmniGibson's tensorized-state change detection is on (``gm.TRACK_STATE_UPDATES``; builds without the switch always
    track) and whether queue mode restores the scene between instances."""
    global _PROVENANCE
    if _PROVENANCE is None:
        import ispatialgym

        isg_root = os.path.dirname(os.path.dirname(os.path.abspath(ispatialgym.__file__)))
        b1k_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(og.__file__))))

        def git(root, *a):
            try:
                return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=60).stdout.strip() or None
            except Exception:  # noqa: BLE001
                return None

        _PROVENANCE = {
            "snapshot": os.environ.get("PROD_SNAP") or os.path.dirname(isg_root),
            "isg_root": isg_root, "isg_commit": git(isg_root, "rev-parse", "HEAD"),
            "isg_dirty": bool(git(isg_root, "status", "--porcelain", "--", "ispatialgym", "scripts")),
            "b1k_root": b1k_root, "b1k_commit": git(b1k_root, "rev-parse", "HEAD"),
            "b1k_dirty": bool(git(b1k_root, "status", "--porcelain", "--", "OmniGibson/omnigibson", "bddl3")),
            "track_state_updates": bool(getattr(gm, "TRACK_STATE_UPDATES", True)),
            "queue_scene_restore": QUEUE_SCENE_RESTORE,
        }
    return dict(_PROVENANCE)


def _first_key(d: Dict[str, Any], suffix: str) -> Optional[str]:
    for k in d:
        if k.endswith(suffix):
            return k
    return None


# ==========================================================================================
# loader (spec §3.14)
# ==========================================================================================


class InstanceLoader:
    """Loads one instance and re-applies what the TRO must let ``run_isg`` restore after every reset."""

    def __init__(self, tro_path: str, robot_name: Optional[str] = None, image_size: int = DEFAULT_IMAGE_SIZE, head_aperture: float = DEFAULT_HEAD_APERTURE_MM, *, restoration_policy=None, pool_consumers=None, pool_world=None):
        self.tro_path = os.path.abspath(tro_path)
        self.iid, self.template_dir, self.tmeta, self.activity, self.did = resolve_instance(self.tro_path)
        # 2026-11-01 train-fast (PARTIAL_SCENE): the sampler's cropped world when the template was sampled in one (None: the full template)
        self.scene_crop = scene_crop_of(self.tmeta)
        self.tro = load_tro(self.tro_path)
        self.pool_world = pool_world
        if pool_world is not None:
            from ispatialgym.oracle.pool_world import PoolWorld
            if type(pool_world) is not PoolWorld or restoration_policy is not None:
                raise ValueError("native pool needs its own reviewed lifecycle; combined restoration is not bound")
            if pool_consumers is not None and pool_consumers is not pool_world.consumers:
                raise ValueError("native pool and recording consumers differ")
            pool_consumers = pool_world.consumers
            pool_world.bind_loader(self)
        self.pool_consumers = pool_consumers
        if pool_consumers is not None and pool_world is None:
            from ispatialgym.oracle.pool_consumers import PoolConsumers
            if type(pool_consumers) is not PoolConsumers:
                raise ValueError("loader needs exact native pool consumers")
            pool_consumers.bind_loader(self)
        from pathlib import Path
        from scripts.campaign_20261110.training_canary_scope import restoration_execution

        self._restoration_execution = restoration_execution(
            self.tro.meta.raw.get("campaign") or {}, self.tro_path, Path(__file__).resolve().parents[2])
        from ispatialgym.oracle.restoration_policy import validate as validate_restoration_policy

        self._restoration_policy = validate_restoration_policy(
            restoration_policy, self.tro_path, Path(__file__).resolve().parents[2])
        self.robot_name = robot_name or robot_name_from_tro(self.tro)
        self.image_size = int(image_size)
        self.head_aperture = float(head_aperture)
        self.env = None
        self.task = None
        self.robot = None
        self.cam_key: Optional[str] = None
        self.physical: Dict[str, Any] = {}
        self.report: Dict[str, Any] = {}
        self.hinges: List[Any] = []
        self.grasps: List[dict] = []
        self.hold: Optional[ActionState] = None
        self._articulation_q: Dict[str, th.Tensor] = {}
        #: the rigid-body physics overrides (``sleep_assist.rigid_overrides`` of the run's flags, set by run_instance; empty: stock):
        #: the scene's free objects at the build, the task objects at every restore, as ``VecInstanceLoader``
        self.physics_overrides: Dict[str, Any] = {}

    # -- environment --------------------------------------------------------------------------------------------

    def build_env(self):
        # 2026-11-01 train-fast COLLECTOR_GPU_DYNAMICS (default off; owner follow-up F1): GPU PhysX for this single-scene collector, the
        # simulator device staying "cpu" (PhysX steps on the GPU, the tensor API hands CPU copies), for the CPU vs GPU dynamics A/B
        gpu_dynamics = fast_on("COLLECTOR_GPU_DYNAMICS", default=False)
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", gpu_dynamics), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.cfg = build_env_config(self.template_dir, self.tmeta, self.tro, self.tro_path, self.activity, self.did, self.robot_name, self.image_size,
                                    behavior_output=getattr(self, "behavior_output", False), scene_crop=self.scene_crop,
                                    behavior_proprio=getattr(self, "behavior_proprio", False))
        if self.pool_world is not None:
            self.cfg = self.pool_world.configuration(self.cfg)
        if self.scene_crop:
            # 2026-11-01 train-fast (PARTIAL_SCENE): the demo is recorded in the sampler's cropped world; the replayer combines it with the
            # full template (recorder attrs isg_scene_crop / isg_full_scene_file, replay --scene full)
            log.info("train-fast PARTIAL_SCENE: collector world = %s (crop %s, radius %s m around %s, %s dropped), rooms %s (full template: %s)",
                     self.cfg["scene"]["scene_file"], self.scene_crop.get("mode"), self.scene_crop.get("radius_m"), self.scene_crop.get("center_object"),
                     self.scene_crop.get("n_dropped"), self.cfg["scene"]["load_room_instances"], self.tmeta.get("load_room_instances"))
        if getattr(self, "external_camera", False):
            # the BEHAVIOR / JoyLo replay external viewpoint (joylo/scripts/replay_data.py): a camera on the robot's base
            # link over its shoulder, looking down-forward
            self.cfg["env"]["external_sensors"] = [{
                "sensor_type": "VisionSensor",
                "name": EXTERNAL_CAMERA_NAME,
                "relative_prim_path": f"/controllable__r1pro__{self.robot_name}/base_link/{EXTERNAL_CAMERA_NAME}",
                "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": EXTERNAL_CAMERA_HW[0], "image_width": EXTERNAL_CAMERA_HW[1], "horizontal_aperture": 40.0},
                "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
                "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
                "pose_frame": "parent",
            }]
        self.env = og.Environment(configs=self.cfg)
        # 2026-11-02 train-fast COLLECTOR_PHYSX_THREADS (default 0 = PhysX's own count; packing test E16): the CPU PhysX worker threads
        # of this collector, for jobs that run several collectors on few CPUs
        import omnigibson.lazy as lazy

        settings = lazy.carb.settings.get_settings()
        self.report["physx_threads"] = {"default": settings.get("/persistent/physics/numThreads")}
        physx_threads = fast_value("COLLECTOR_PHYSX_THREADS", 0, int)
        if physx_threads > 0:
            from omni.physx import get_physx_interface

            settings.set_int("/persistent/physics/numThreads", physx_threads)
            get_physx_interface().set_thread_count(physx_threads)
            self.report["physx_threads"]["set"] = physx_threads
        log.info("train-fast COLLECTOR_PHYSX_THREADS: %s", self.report["physx_threads"])
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta, patch_trav_map_from_meta

        # the scene variant's freed footprints (template) and the instance's blocked footprints on the planner map
        if getattr(self, '_sequential_augmentation', None) is not None:
            self._floor_map_before_augmentation = _copy_map(self.env.scene.trav_map.floor_map[0])
        self.report["trav_cells_freed"] = patch_trav_map_from_meta(self.env.scene, self.tmeta)
        self._floor_map_template = _copy_map(self.env.scene.trav_map.floor_map[0])
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.task = self.env.task
        self.robot = self.env.scene.robots[0]
        # the BEHAVIOR-1K evaluator's 250 kg chassis (owner 2026-09-21 23:49): the stock 47.6 kg base tipped 0.8 deg on
        # its wheel spheres under the dishwasher pull and snapped back at the release - the grader's physics, not ours. The rigid-body
        # overrides of the scene's free objects in the same stop / play (VecInstanceLoader.build_env)
        ov = self.physics_overrides

        # oracle-fix 2026-11-07 (owner prompt B, opt-in ISPATIALGYM_PHYSX_FILTER_ELBOW=1): the elbow pair link3 / link5 of each arm filtered in
        # PhysX in the same stop (the hulls graze within the joint-4 limit; the fix3_hb pitch-40 push with the QA exclusion: 29-30 contact steps,
        # the hand slowed 0.103 -> 0.028 m/s and joint 4 chattered at up to 21 rad/s^2); recorded as meta.loader.physx_filtered_robot_pairs
        elbow = os.environ.get(M.PHYSX_FILTER_ELBOW_SWITCH, "") == "1"
        self.report["physx_filtered_robot_pairs"] = []

        def _rigid():
            if ov:
                self.report["physics_overrides"] = SA.apply_to_scenes([self.env.scene], ov)
            if elbow:
                from ispatialgym.oracle import qa as QA

                self.report["physx_filtered_robot_pairs"] = M.filter_link_pairs_stopped(self.robot, QA.ELBOW_GRAZE_PAIRS)
            if self.pool_world is not None:
                self.pool_world.cold_stopped(og.sim)

        self.report["base_link_mass_kg"] = M.apply_eval_base_mass(self.robot, while_stopped=_rigid if (ov or elbow or self.pool_world is not None) else None)
        log.info("robot base_link mass %.2f -> %.2f kg (BEHAVIOR-1K eval: %.0f)", self.report["base_link_mass_kg"]["before"], self.report["base_link_mass_kg"]["after"], M.EVAL_BASE_LINK_MASS_KG)
        self.cam_key = _first_key(self.robot.sensors, M.HEAD_CAMERA_SENSOR_SUFFIX)
        if self.cam_key is None:
            raise RuntimeError(f"robot {self.robot.name} has no head camera sensor ({M.HEAD_CAMERA_SENSOR_SUFFIX})")
        if self.head_aperture:
            self.robot.sensors[self.cam_key].horizontal_aperture = self.head_aperture
        self.report["head_camera_key"] = self.cam_key
        self.report["head_camera_aperture_mm"] = float(self.robot.sensors[self.cam_key].horizontal_aperture)
        if getattr(self, "behavior_output", False):
            from ispatialgym.oracle.behavior_lerobot import warm_camera_params

            warm_camera_params(self.env)
        self.report["evaluation_profile"] = (self.tro.meta.extras.get("evaluation_profile") or EVALUATION_PROFILE)
        # the whole scene as the template file loads it, for attach(): load_tro_instance rewrites only the task objects and
        # the robot, and restore()'s update_initial_file would carry whatever a demo moved into the next instance's start
        self._scene_state0 = copy.deepcopy(self.env.scene.dump_state(serialized=False))
        if getattr(self, '_sequential_augmentation', None) is not None:
            from ispatialgym.oracle.campaign_batch import capture_robot_anchor
            self._cold_robot_anchor = capture_robot_anchor(self)
        from ispatialgym.oracle.campaign_restore import capture_pristine_mass, mass_constraint_objects

        campaign = self.tro.meta.raw.get("campaign") or {}
        self._pristine_mass0 = capture_pristine_mass(campaign, mass_constraint_objects(
            campaign, self.task.object_scope, (self.tro.meta.extras or {}).get("constraints") or [],
            execution=self._restoration_execution), execution=self._restoration_execution)
        from ispatialgym.oracle.restoration_policy import capture as capture_fixed_frames

        capture_fixed_frames(self)
        if self.pool_world is not None:
            self.pool_world.initialized()
        return self.env

    def share(self, shared: Dict[str, Any]) -> None:
        """``--queue``: hand this loader's world to the next instances of the template."""
        shared.update({"env": self.env, "task": self.task, "robot": self.robot, "cam_key": self.cam_key, "cfg": self.cfg,
                       "template_dir": self.template_dir, "template_key": (template_key(self.template_dir, self.tmeta) if fast_on("COLLECTOR_PREBUILD") else None), "floor_map": self._floor_map_template, "scene_state0": self._scene_state0,
                       "pristine_mass0": getattr(self, "_pristine_mass0", None),
                       "build_report": {k: v for k, v in self.report.items() if k != "trav_cells_blocked"}})
        from ispatialgym.oracle.restoration_policy import share as share_fixed_frames

        share_fixed_frames(self, shared)
        if self.pool_world is not None:
            self.pool_world.share(shared)
        if getattr(self, '_sequential_augmentation', None) is not None:
            shared['floor_map_before_augmentation'] = self._floor_map_before_augmentation
            shared['cold_robot_anchor'] = self._cold_robot_anchor

    def attach(self, shared: Dict[str, Any]):
        """``--queue``: this instance in the world of the previous one (same template, :func:`instance_signature`) instead of
        :meth:`build_env` - the previous instance's hinge prims, assisted grasps and planner attachments are removed, the
        whole scene goes back to its state right after :meth:`build_env` and the planner map to the template's before this
        instance's footprints are blocked; :meth:`restore` follows."""
        from ispatialgym.sampling.constraints import remove_hinge
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta

        if os.path.abspath(shared["template_dir"]) != os.path.abspath(self.template_dir):
            # 2026-10-31 train-fast (COLLECTOR_PREBUILD): another copy of the same template (equal template_key) is the same world
            same = fast_on("COLLECTOR_PREBUILD") and os.path.abspath(self.template_dir) in (shared.get("compatible_template_dirs") or ())
            if self.pool_world is not None:
                same = self.pool_world.compatible(shared)
            if not same:
                raise RuntimeError(f"--queue: {self.iid} belongs to {self.template_dir}, the world to {shared['template_dir']}")
            self.report["world_built_from"] = shared["template_dir"]
        self.env, self.task, self.robot, self.cam_key, self.cfg = (shared[k] for k in ("env", "task", "robot", "cam_key", "cfg"))
        from ispatialgym.oracle.restoration_policy import attach as attach_fixed_frames

        attach_fixed_frames(self, shared)
        prev = shared.get("loader")
        for h in list(getattr(prev, "hinges", None) or []):
            remove_hinge(h)
        for arm, held in list(self.robot._ag_obj_in_hand.items()):
            if held is not None:
                self.robot.release_grasp_immediately(arm)
        backend = shared.get("backend")
        for arm in list(getattr(backend, "_attached", {}) or {}):
            backend.detach_object(arm)
        # i7-attachfix 2026-11-08: the previous episode's BEHAVIOR attachment joints (a lid / cork the attach rule joined to its partner)
        # are removed HERE, before the state load. Left in place, the scene state load below detaches them itself (AttachedTo._load_state
        # -> _detach -> og.sim.update_handles() in the middle of the load) and the instance started with the pair off its TRO
        # (i7_attswept: attachment_01 300114 lid 103.8 mm / teapot 81.6 mm, 300115 lid 76.6 mm, attachment_04 300116 lid 549 mm,
        # attachment_03 300020 bottle 40 mm - each the instance after a joined episode on the same collector world)
        self.report["attachments_detached_before_reuse"] = detach_scene_attachments(self.env.scene)
        # ... and the attach rule's pair records reset (as the vectorized loader's cleanup does): a joined pair stays collision-FILTERED in
        # USD across the state load and its deferred un-filter (a stop / play) ran in the next episode's restore - i7af_v (trunk_b4d8f3d4)
        # still met the pair 95-240 mm / 0.6-3.1 rad off its TRO there (the pose check re-applied it); the un-filter now runs here, before
        # the state load
        from ispatialgym.sampling import predicates as PRED

        self.report["pair_records_reset"] = PRED.reset_pair_records(self.env.scene)
        from ispatialgym.oracle.campaign_restore import restart_reused_world

        scene_state = copy.deepcopy(shared['scene_state0'])
        if self.pool_world is not None:
            if shared.get('sequential_augmentation') is not None:
                raise ValueError('combined pool and augmentation lifecycle is not bound')
            scene_state = self.pool_world.prepare_incoming(shared)
        anchor_callback = None
        if shared.get('sequential_augmentation') is not None:
            from ispatialgym.oracle.campaign_batch import prepare_robot_anchor, restore_robot_anchor_stopped
            scene_state, anchor_record = prepare_robot_anchor(self, shared, shared['sequential_augmentation'])
            self.report['sequential_robot_anchor'] = anchor_record
            anchor_callback = lambda: restore_robot_anchor_stopped(self.robot, og.sim, anchor_record)
            # A cold world parses the original USD mass, not the preceding
            # episode's tray override. Restore the captured native mass and USD
            # authorship before the existing stop/play rebuild; the ordinary
            # pre-reset check below still runs and must be idempotent.
            from ispatialgym.oracle.campaign_restore import mass_constraint_objects, restore_pristine_mass
            campaign = self.tro.meta.raw.get('campaign') or {}
            self.report['pre_lifecycle_pristine_mass'] = restore_pristine_mass(
                campaign, mass_constraint_objects(campaign, self.task.object_scope,
                    (self.tro.meta.extras or {}).get('constraints') or [], execution=self._restoration_execution),
                shared.get('pristine_mass0'), execution=self._restoration_execution)
        lifecycle = restart_reused_world(self.tro.meta.raw.get("campaign") or {}, og.sim,
                                         execution=self._restoration_execution, while_stopped=anchor_callback)
        if anchor_callback is not None and lifecycle is None:
            raise RuntimeError('sequential robot anchor requires the authorized native lifecycle')
        if lifecycle is not None:
            self.report["reuse_native_lifecycle"] = lifecycle
        # every scene object (poses, velocities, joints of doors / drawers, object states) back to the template's, so that
        # only the TRO decides where this instance starts - as in a fresh process
        self._load_reused_scene_state(scene_state)
        from ispatialgym.oracle.restoration_policy import restore_background_roots
        restore_background_roots(self, scene_state, og.sim, phase='after_attach_state_restore')
        if anchor_callback is not None:
            from ispatialgym.oracle.campaign_batch import verify_robot_anchor
            verify_robot_anchor(self.robot, anchor_record)
        self._scene_state0 = shared["scene_state0"]
        self._pristine_mass0 = shared.get("pristine_mass0")
        self.env.scene.trav_map.floor_map[0] = _copy_map(shared["floor_map"])
        self._floor_map_template = shared["floor_map"]
        self.report.update(shared["build_report"])
        if shared.get('sequential_augmentation') is not None:
            from ispatialgym.oracle.campaign_batch import restore_parking
            from ispatialgym.sampling.scene_variants import patch_trav_map_from_meta

            self.report['sequential_augmentation'] = restore_parking(self, shared['sequential_augmentation'])
            self.env.scene.trav_map.floor_map[0] = _copy_map(shared['floor_map_before_augmentation'])
            self.report['trav_cells_freed'] = patch_trav_map_from_meta(self.env.scene, self.tmeta)
            self._floor_map_template = _copy_map(self.env.scene.trav_map.floor_map[0])
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.report["reused_world"] = True
        if self.pool_world is not None:
            self.pool_world.attached()
        return self.env

    def _load_reused_scene_state(self, scene_state):
        """Load the incoming snapshot after attachment cleanup has finished.

        Cleanup may independently restore its own native snapshot. Keep that
        lifecycle separate from the single incoming episode state boundary.
        """
        if self.pool_world is not None:
            return self.pool_world.load_incoming(scene_state, og.sim)
        return self.env.scene.load_state(scene_state, serialized=False)

    # -- the loader contract --------------------------------------------------------------------------------------

    def restore(self, settle_physics_steps: int = 25):
        """Restore with complete native snapshots, then reinstate the episode filter."""
        from ispatialgym.oracle.campaign_restore import complete_scene_snapshot_scope
        from ispatialgym.oracle.restoration_policy import restore as restore_fixed_frames
        from ispatialgym.oracle.restoration_policy import complete_snapshot_scope

        restore_fixed_frames(self, og.sim)
        if self._restoration_policy is not None:
            with complete_snapshot_scope(self):
                return self._restore_complete_snapshot(settle_physics_steps)

        with complete_scene_snapshot_scope(self.tro.meta.raw.get("campaign") or {},
                                           self.env.scene, self.report,
                                           execution=self._restoration_execution):
            result = self._restore_complete_snapshot(settle_physics_steps)
        if self.pool_world is not None:
            self.pool_world.observe('after_complete_native_loader')
        return result

    def _restore_complete_snapshot(self, settle_physics_steps: int = 25):
        """TRO → simulator, then the §3.14 restores, a settle with the task entities held, the snapshot, one reset."""
        env, task, robot = self.env, self.task, self.robot
        try:  # i7-attachfix: the attach rule's join list is per process - this episode's joins start here
            from ispatialgym.sampling import predicates as PRED

            self.report["attach_joins_start"] = len(PRED.JOINS)
        except Exception:  # noqa: BLE001
            pass
        self.report["floor_collision_filter"] = self._apply_floor_filters()
        if self.tro.meta.raw.get("campaign"):
            from ispatialgym.oracle.campaign_restore import mass_constraint_objects, restore_pristine_mass, restore_serialized_mass_before_reset

            campaign = self.tro.meta.raw["campaign"]
            pristine_mass = restore_pristine_mass(campaign, mass_constraint_objects(
                campaign, task.object_scope, (self.tro.meta.extras or {}).get("constraints") or [],
                execution=self._restoration_execution),
                getattr(self, "_pristine_mass0", None), execution=self._restoration_execution)
            if pristine_mass is not None:
                self.report["pre_reset_pristine_mass"] = pristine_mass
            early_mass = restore_serialized_mass_before_reset(
                self.tro.meta.raw["campaign"], task.object_scope, self.tro.meta.object_bindings or {},
                (self.tro.meta.extras or {}).get("constraints") or [], execution=self._restoration_execution)
            if early_mass is not None:
                self.report["pre_reset_serialized_mass"] = early_mass
        from ispatialgym.oracle.restoration_policy import needs_reconstruction, reconstruction_reset_scope

        reset_options = ({'reset_context': lambda: reconstruction_reset_scope(self, og.sim)}
                         if needs_reconstruction(self) else {})
        if self.pool_world is not None:
            self.pool_world.observe_reset('before_task_restore', og.sim)
        task.load_tro_instance(env, tro=self.tro, settle_steps=0, **reset_options)
        if self.pool_world is not None:
            self.pool_world.observe_reset('after_task_restore', og.sim)
        self.physical = {k: v for k, v in task.object_scope.items() if v is not None and not isinstance(v, GroupEntity)}
        if self.tro.meta.raw.get("campaign"):
            from ispatialgym.sampling.campaign_runtime import verify_loaded

            self.report["campaign_restoration"] = verify_loaded(self.tro.meta.raw["campaign"], self.physical,
                                                                  scene=env.scene, sim=og.sim)
        # the rigid-body overrides of the task objects the TRO placed (the build's pass reaches the scene's own objects only), before
        # the constraints and the settle, as VecInstanceLoader.restore
        self.report["rigid_sleep_threshold"] = SA.apply_to_objects(self.physical.values(), self.physics_overrides)
        extras = self.tro.meta.extras
        self.report["constraints"] = self._restore_constraints(extras.get("constraints") or [])
        self.report["attachments"] = self._attachment_report(extras.get("attachment_targets") or [])
        self._articulation_q = self._held_articulations(self.physical)
        self.grasps = self._restore_grasps(extras.get("grasps") or [])
        self.report["grasps"] = self.grasps
        # a gripper command must exist or assisted grasping releases the object at the first physics step
        self.hold = ActionState(robot)
        self.hold.hold_current()
        for g in self.grasps:
            if g.get("finger_q"):
                self.hold.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        recorded = (self.tro.meta.robot_state or {}).get("controller_targets") or {}
        self.report["controller_targets_restored"] = apply_recorded_targets(self.hold, robot, recorded) if recorded else []
        robot.apply_action(self.hold.action())
        if self.pool_world is not None:
            self.pool_world.observe_reset('before_native_settle', og.sim)
        for _ in range(settle_physics_steps):
            og.sim.step_physics()
            self._hold_entities()
        for _ in range(2):  # full steps refresh the per-step caches (AABBs, contacts, kinematic 'changed' flags)
            og.sim.step()
            self._hold_entities()
        if self.pool_world is not None:
            self.pool_world.observe_reset('after_native_settle_before_reset', og.sim)
        from contextlib import nullcontext

        with self.pool_world.trace_reset(og.sim) if self.pool_world is not None else nullcontext() as reset_trace:
            env.scene.update_initial_file()
            task._reset_tracking()
            if PO.trim():
                # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): Environment.reset renders 3 frames before it reads the observation (no
                # config key for the count in OmniGibson 3.9); a physics-only run keeps no frame - the same reset and step, the
                # observation read without the renders (its images are stale, nothing reads them)
                env.reset(get_obs=False)
                og.sim.step()
                obs, obs_info = SE.get_obs(env)
                info = {"obs_info": obs_info}
                log.info("train-fast PHYSICS_ONLY_TRIM: reset without the 3 observation renders")
            else:
                obs, info = SE.reset(env)  # returns to the snapshot, activates pose_settled tracking, renders
            self.report["after_reset"] = self.restore_after_reset()
            # i7-attachfix: every TRO object starts where the TRO puts it (the plan, its grasps and its IK witnesses assume it); a world
            # reused after a joined episode left the pair off its TRO pose and nothing compared them (meta.start_state was only recorded)
            self.report["tro_pose_check"] = self._check_tro_poses()
            if self.report["controller_targets_restored"]:
                robot.apply_action(self.hold.action())  # the recorded targets also hold after the reset, until the first action
            if self.tro.meta.raw.get("campaign"):
                from ispatialgym.oracle.campaign_restore import restore_serialized_root_motion

                restored = restore_serialized_root_motion(self.tro.meta.raw["campaign"], self.physical,
                                                          self.tro.object_states, execution=self._restoration_execution)
                if restored is not None:
                    self.report["campaign_restoration"]["serialized_root_motion"] = restored
            if self.pool_world is not None:
                self.pool_world.observe_reset('after_native_reset', og.sim)
        if self.pool_world is not None:
            self.report["model_pool_reset_internal_trace"] = reset_trace
        return obs, info

    def _tro_pose_pairs(self) -> Dict[str, Tuple[Any, Any]]:
        """``{instance: (TRO root pose, live root pose)}`` of the TRO's physical objects (position, quaternion xyzw)."""
        out = {}
        for inst, st in (self.tro.object_states or {}).items():
            ent = self.physical.get(inst)
            rl = (st or {}).get("root_link") if isinstance(st, dict) else None
            if ent is None or not rl or rl.get("pos") is None or rl.get("ori") is None:
                continue
            pos, quat = ent.get_position_orientation()
            out[inst] = ((rl["pos"], rl["ori"]), (pos.tolist(), quat.tolist()))
        return out

    def _check_tro_poses(self) -> Dict[str, Any]:
        """i7-attachfix: compare every TRO object's live root pose with the TRO (:func:`tro_pose_mismatches`). A mismatch is re-applied
        once (the TRO states of those objects, held still, the initial file and the reset again); one that persists is a LoadError -
        the episode must not start from a world the plan does not describe."""
        pairs = self._tro_pose_pairs()
        bad = tro_pose_mismatches({k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()})
        rec: Dict[str, Any] = {"n_objects": len(pairs), "mismatch": bad, "reapplied": False}
        if not bad:
            return rec
        log.warning("TRO_POSE_CHECK: %d object(s) off their TRO pose after the restore: %s - re-applying the TRO states", len(bad), bad)
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        for m in bad:
            self.physical[m["instance"]].load_state(recursively_convert_to_torch(copy.deepcopy(self.tro.object_states[m["instance"]])), serialized=False)
        for _ in range(2):
            og.sim.step_physics()
            self._hold_entities()
        og.sim.step()
        self.env.scene.update_initial_file()
        self.env.scene.reset()
        self.restore_after_reset()
        pairs = self._tro_pose_pairs()
        still = tro_pose_mismatches({k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()})
        rec.update({"reapplied": True, "after_reapply": still})
        if still:
            raise LoadError("error", f"TRO objects off their TRO pose after the restore and a re-apply: {still}")
        return rec

    def _held_articulations(self, physical: Dict[str, Any]) -> Dict[str, th.Tensor]:
        """The joint vectors the settle holds and every reset restores: the ``articulation_targets`` instances' (the target's initial
        joints) and the objects of ``joint_positions`` constraints (a partner's door the sampler opened)."""
        insts = list(self.tro.meta.articulation_targets) + [r["object"] for r in getattr(self, "_joint_position_records", [])]
        return {inst: physical[inst].get_joint_positions().clone() for inst in dict.fromkeys(insts) if inst in physical}

    def joint_positions_report(self) -> Dict[str, Any]:
        """``{instance: {scene_object, init, end}}`` for the ``joint_positions`` constraints: the joints as restored and as they are now
        (read at the end of the episode, for the report); empty without such a record."""
        out: Dict[str, Any] = {}
        for rec in getattr(self, "_joint_position_records", []):
            ent = self.physical.get(rec["object"])
            if ent is None:
                continue
            names = list(ent.joints)
            q = ent.get_joint_positions()
            out[rec["object"]] = {"scene_object": ent.name, "init": {n: float(v) for n, v in rec["joints"].items()},
                                  "end": {n: round(float(q[names.index(n)]), 4) for n in rec["joints"] if n in names}}
        return out

    def _hold_entities(self):
        for inst, ent in self.physical.items():
            ent.keep_still()
            q0 = self._articulation_q.get(inst)
            if q0 is not None:
                ent.set_joint_positions(q0)
                ent.set_joint_velocities(th.zeros_like(q0))

    def restore_after_reset(self) -> Dict[str, Any]:
        """Re-apply the restores a ``scene.reset()`` may have undone (idempotent): hinge prims, assisted grasps and the
        articulation joint values."""
        out: Dict[str, Any] = {"hinges_recreated": 0, "grasps": []}
        stage = og.sim.stage
        for i, h in enumerate(list(self.hinges)):
            prim = stage.GetPrimAtPath(h.prim_path)
            if prim is None or not prim.IsValid():
                rec = self._hinge_records[i]
                self.hinges[i] = self._create_hinge(rec)
                out["hinges_recreated"] += 1
        for rec in getattr(self, "_joint_friction_records", []):
            self._apply_joint_friction(rec)
        for rec in getattr(self, "_counterbalance_records", []):
            self._apply_counterbalance(rec)
        for rec in getattr(self, "_mass_records", []):
            self._apply_mass(rec)
        for g in self.grasps:
            obj = self.physical[g["object"]]
            attached = self.robot._ag_obj_in_hand.get(g["arm"]) is obj
            if not attached:
                self._establish(g["arm"], obj, g.get("ag_anchor_in_eef"))
                attached = self.robot._ag_obj_in_hand.get(g["arm"]) is obj
            g["attached_after_reset"] = attached
            out["grasps"].append({"arm": g["arm"], "object": g["object"], "attached": attached})
            if not attached:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} could not be restored after the reset")
        for inst, q0 in self._articulation_q.items():
            ent = self.physical[inst]
            q = ent.get_joint_positions()
            if float(th.max(th.abs(q - q0))) > 1e-3:
                ent.set_joint_positions(q0)
                ent.set_joint_velocities(th.zeros_like(q0))
                out.setdefault("articulation_reset", []).append(inst)
        return out

    # -- constraints (hinges) -------------------------------------------------------------------------------------

    def _create_hinge(self, rec: dict):
        from ispatialgym.sampling.constraints import HINGE_BRAKE_OMEGA_RAD_S, create_hinge, hinge_brake_damping

        child = self.physical.get(rec["child"])
        if child is None:
            raise LoadError("error", f"constraint child {rec['child']!r} is not a bound physical instance")
        parent_link = None
        if rec.get("parent"):
            parent = self.physical.get(rec["parent"]) or self.env.scene.object_registry("name", rec["parent"])
            if parent is None:
                raise LoadError("error", f"constraint parent {rec['parent']!r} not found")
            link_name = rec.get("parent_link") or parent.root_link_name
            parent_link = parent.links[link_name]
        # §7 hinge friction / damping (owner 2026-09-16 22:05): the wall hinge record carries a Coulomb torque + viscous
        # damping, applied as the joint's angular drive (sampling.constraints.apply_hinge_drive); absent = frictionless (axial)
        # rush 2026-09-17 (branch rush-20260917-2100): ISG_HINGE_FRICTION_SCALE / ISG_HINGE_DAMPING_SCALE multiply the record's
        # values at load time (painting 0917-211738: push in tolerance at +0.037 rad, then the departing pads shoved it
        # to -0.39 rad - the record's resistance holds a coasting painting, not a shoved one). Default 1 = the record.
        # 2026-09-22 (0918-000750, friction x8 / damping x3): the scaled hinge needed ~40 N at the pad instead of the record's
        # ~9 N (torque = friction + damping * omega at 0.6 rad/s); the elbow drive saturated at its 25 N.m cap (0.097 rad
        # behind its target), the wrist drives went bang-bang against the contact and the painting shook / was kicked back
        # 0.8 deg at the stop. The friction scale is therefore capped at HINGE_FRICTION_SCALE_MAX and the damping scale is no
        # longer applied (the viscous term dominates at push speed); the departing-pad shove is handled by the radial escape
        # of the retreat. A stiffer hinge belongs in the record, where the sampler's feasibility checks see the same value.
        friction, damping = rec.get("friction"), rec.get("damping")
        f_scale, d_scale = float(os.environ.get("ISG_HINGE_FRICTION_SCALE", "1")), float(os.environ.get("ISG_HINGE_DAMPING_SCALE", "1"))
        if not (0.0 <= f_scale <= HINGE_FRICTION_SCALE_MAX):
            log.warning("hinge %s: ISG_HINGE_FRICTION_SCALE=%.2f clamped to [0, %.1f]", rec["child"], f_scale, HINGE_FRICTION_SCALE_MAX)
            f_scale = min(max(f_scale, 0.0), HINGE_FRICTION_SCALE_MAX)
        if d_scale != 1.0:
            log.warning("hinge %s: ISG_HINGE_DAMPING_SCALE=%.2f ignored (the damping of the record is applied as is)", rec["child"], d_scale)
        if friction is not None and f_scale != 1.0:
            friction = friction * f_scale
            log.warning("hinge %s: friction x%.1f (ISG_HINGE_FRICTION_SCALE override)", rec["child"], f_scale)
        # brake (2026-09-22, sampling.constraints module docstring): a record written before the brake model carries the plain
        # damper I_n / tau, under which a residual torque (the anchor a millimetre off the COM) creeps the hinge forever (0922-180242:
        # the levelled painting drifted back 0.25 deg/s out of its tolerance during the settle window) - its damping is raised to
        # friction / HINGE_BRAKE_OMEGA_RAD_S, the same value the sampler writes now; the friction the pad works against is untouched
        braked = hinge_brake_damping(friction, damping)
        if braked is not None and (damping is None or braked > float(damping) + 1e-9):
            log.warning("hinge %s: record damping %s N·m·s/rad predates the brake model - applied friction / %.4f rad/s = %.1f N·m·s/rad",
                        rec["child"], None if damping is None else round(float(damping), 3), HINGE_BRAKE_OMEGA_RAD_S, braked)
            damping = braked
        return create_hinge(child, th.tensor(rec["anchor_position"], dtype=th.float32), th.tensor(rec["axis"], dtype=th.float32), parent_link=parent_link,
                            friction=friction, damping=damping)

    def _apply_joint_friction(self, rec: dict) -> dict:
        """``{"type": "joint_friction", "object", "joint", friction | stiffness / damping / max_effort}`` (small articulation,
        2026-09-18): the resistance of the moved joint as the sampler probed the stable set and the goal settle with it
        (:func:`ispatialgym.sampling.constraints.apply_joint_brake`; the stock 0.2 friction lets a hinged lid fall to its
        gravitational equilibrium, damping + max_effort make the joint drive a Coulomb brake). Idempotent; re-applied after
        every reset."""
        from ispatialgym.sampling.constraints import apply_joint_brake

        ent = self.physical.get(rec["object"])
        if ent is None:
            raise LoadError("error", f"joint_friction constraint names an unbound instance {rec['object']!r}")
        if rec["joint"] not in ent.joints:
            raise LoadError("error", f"joint_friction constraint: {rec['object']} has no joint {rec['joint']!r}")
        spec = {k: rec[k] for k in ("friction", "stiffness", "damping", "max_effort", "lower_limit_rad", "upper_limit_rad") if rec.get(k) is not None}
        applied = apply_joint_brake(ent.joints[rec["joint"]], spec)
        if rec.get("openable_joints"):
            # a flap with an extended lower limit would read as open at its closed position (OG's Open threshold is lower + 5 %
            # of the range): the record names the joints that decide openness (the target joint), 2026-09-19
            from ispatialgym.sampling.constraints import restrict_openable_joints

            restrict_openable_joints(ent, list(rec["openable_joints"]))
        return {"type": "joint_friction", "object": rec["object"], "joint": rec["joint"], **applied}

    def _apply_joint_positions(self, rec: dict) -> dict:
        """``{"type": "joint_positions", "object", "joints": {joint: position}}`` (IORS large-scale, ``generation.partner_joint_init``,
        owner 2026-10-12 ~16:05: the fridge door open at init): the joints set (the TRO's object state already holds them; this makes the
        restore explicit), held through the settle and restored after a reset with the articulation targets (:meth:`_held_articulations`)."""
        ent = self.physical.get(rec["object"])
        if ent is None:
            raise LoadError("error", f"joint_positions constraint names an unbound instance {rec['object']!r}")
        names = list(ent.joints)
        unknown = sorted(set(rec.get("joints") or {}) - set(names))
        if unknown:
            raise LoadError("error", f"joint_positions constraint: {rec['object']} has no joints {unknown}")
        q = ent.get_joint_positions().clone()
        for n, v in (rec.get("joints") or {}).items():
            q[names.index(n)] = float(v)
        ent.set_joint_positions(q)
        ent.set_joint_velocities(th.zeros_like(q))
        return {"type": "joint_positions", "object": rec["object"], "joints": {n: float(v) for n, v in (rec.get("joints") or {}).items()}}

    def _apply_counterbalance(self, rec: dict) -> dict:
        """``{"type": "counterbalance", "object", "joint", "links", "damping"}`` (large articulation, owner decision 2026-09-19): the
        spring-balanced appliance door - gravity off on the leaf assembly, viscous joint damping (:func:`ispatialgym.sampling.constraints.apply_counterbalance`);
        the sampler probed the stable set and the goal settle with it. Idempotent; re-applied after every reset."""
        from ispatialgym.sampling.constraints import apply_counterbalance

        ent = self.physical.get(rec["object"])
        if ent is None:
            raise LoadError("error", f"counterbalance constraint names an unbound instance {rec['object']!r}")
        if rec["joint"] not in ent.joints:
            raise LoadError("error", f"counterbalance constraint: {rec['object']} has no joint {rec['joint']!r}")
        applied = apply_counterbalance(ent, ent.joints[rec["joint"]], list(rec.get("links") or []), rec.get("damping"))
        return {"type": "counterbalance", "object": rec["object"], "joint": rec["joint"], **applied}

    def _apply_mass(self, rec: dict) -> dict:
        """``{"type": "mass", "object", "mass_kg"[, "link"]}`` (disembedding, owner 2026-09-25): the mass the sampler gave a
        support object - the trays under 1 kg bottles, whose 0.39 kg category mass had PhysX chattering the stack
        (:func:`ispatialgym.sampling.constraints.apply_mass`); the sampler's stillness gate ran with it. Idempotent; re-applied
        after every reset."""
        from ispatialgym.sampling.constraints import apply_mass

        ent = self.physical.get(rec["object"])
        if ent is None:
            raise LoadError("error", f"mass constraint names an unbound instance {rec['object']!r}")
        applied = apply_mass(ent, float(rec["mass_kg"]), rec.get("link"))
        return {"type": "mass", "object": rec["object"], **applied}

    def _apply_floor_filters(self) -> List[str]:
        """Oracle-fix 2026-11-04 round 3 (articulation #03, bottom_cabinet_rvunhj_0): the collision between each articulation-target object of
        the TRO and the TRO's floor instance(s) filtered, before the TRO load - when the TRO records a ``floor_collision_filter`` constraint or
        the variant's customs set ``generation.floor_collision_filter``. A probe found the cabinet's lower door slab 5.6 mm into
        floors_ovpbsn_0 at every angle (its only contact; the floor's friction pinned the door against 150 N m; free with the pair filtered).
        Returns the filtered pairs."""
        recs = [r for r in ((self.tro.meta.extras or {}).get("constraints") or []) if isinstance(r, dict) and r.get("type") == "floor_collision_filter"]
        on = bool(recs)
        if not on:
            from types import SimpleNamespace

            from ispatialgym.oracle.execute import variant_generation

            variant = (self.tmeta or {}).get("variant") if isinstance(getattr(self, "tmeta", None), dict) else None
            variant = variant or getattr(self.tro.meta, "variant", None) or os.path.basename(os.path.dirname(str(self.template_dir or "")))
            on = bool(variant_generation({"sidecar": SimpleNamespace(variant=variant)}).get("floor_collision_filter"))
        if not on:
            return []
        from ispatialgym.sampling import predicates as PRED

        b = self.tro.meta.object_bindings or {}
        names = {k: (v.get("scene_object_name") if isinstance(v, dict) else v) for k, v in b.items()}
        targets = [names.get(i) for i in (self.tro.meta.articulation_targets or {})]
        floors = [n for i, n in names.items() if i.split(".")[0] == "floor" and n]
        out = []
        for t in targets:
            for f in floors:
                o, fl = self.env.scene.object_registry("name", t), self.env.scene.object_registry("name", f)
                if o is None or fl is None:
                    log.warning("floor collision filter: %s / %s not in the scene", t, f)
                    continue
                PRED.set_pair_collision_filter(o, fl, True)
                out.append(f"{t} x {f}")
                log.info("floor collision filter: %s x %s filtered", t, f)
        return out

    def _restore_constraints(self, records: List[dict]) -> List[dict]:
        out = []
        self.hinges = []
        self._hinge_records = []  # the hinge records in self.hinges order (restore_after_reset recreates a lost prim from its record)
        self._joint_friction_records = []
        self._counterbalance_records = []
        self._mass_records = []
        self._joint_position_records = []
        for rec in records:
            kind = rec.get("type", "revolute")
            if kind == "joint_positions":
                out.append(self._apply_joint_positions(rec))
                self._joint_position_records.append(rec)
                log.info("joint positions applied: %s %s", rec["object"], out[-1]["joints"])
                continue
            if kind == "mass":
                out.append(self._apply_mass(rec))
                self._mass_records.append(rec)
                log.info("mass applied: %s = %s", rec["object"], out[-1])
                continue
            if kind == "floor_collision_filter":  # applied before the TRO load (_apply_floor_filters)
                out.append({"type": kind, "object": rec.get("object"), "applied": "before the load"})
                continue
            if kind == "counterbalance":
                out.append(self._apply_counterbalance(rec))
                self._counterbalance_records.append(rec)
                log.info("counterbalance applied: %s/%s = %s", rec["object"], rec["joint"], out[-1])
                continue
            if kind == "joint_friction":
                out.append(self._apply_joint_friction(rec))
                self._joint_friction_records.append(rec)
                log.info("joint resistance applied: %s/%s = %s (stock %s)", rec["object"], rec["joint"], out[-1], rec.get("stock", rec.get("stock_friction")))
                continue
            if kind != "revolute":
                raise LoadError("error", f"unsupported constraint type {rec.get('type')!r}")
            h = self._create_hinge(rec)
            self.hinges.append(h)
            self._hinge_records.append(rec)
            out.append({"child": rec["child"], "parent": rec.get("parent"), "prim_path": h.prim_path, "friction": h.friction, "damping": h.damping})
            log.info("hinge restored: %s about %s at %s (friction %s N·m, damping %s N·m·s/rad)", rec["child"], [round(v, 3) for v in rec["axis"]],
                     [round(v, 3) for v in rec["anchor_position"]], None if h.friction is None else round(h.friction, 4), None if h.damping is None else round(h.damping, 4))
        if self.hinges:
            og.sim.update_handles()
        return out

    # -- attachments ----------------------------------------------------------------------------------------------

    def _attachment_report(self, records: List[dict]) -> List[dict]:
        """The TRO's attachment pairs. Their collision stays on until the task's attach rule joins them (owner 2026-10-10;
        ``predicates.attach_if_aligned`` filters it in that instant), so nothing is filtered at load."""
        out = []
        if not records:
            return out
        from omnigibson.object_states import AttachedTo

        for rec in records:
            child, parent = self.physical.get(rec["child"]), self.physical.get(rec["parent"])
            if child is None or parent is None:
                raise LoadError("error", f"attachment_targets names unbound instances {rec}")
            if AttachedTo not in child.states:
                raise LoadError("error", f"{child.name} has no AttachedTo state (not attachable)")
            out.append({"child": rec["child"], "parent": rec["parent"], "child_link": rec.get("child_link"), "parent_link": rec.get("parent_link"), "collision_filtered": False})
        return out

    # -- grasps -----------------------------------------------------------------------------------------------------

    def _establish(self, arm: str, obj, anchor_in_eef=None) -> None:
        """The assisted-grasp joint of ``obj`` in ``arm``'s hand, anchored at ``anchor_in_eef`` (the recorded joint anchor in
        the eef frame; the grasp record's ``ag_anchor_in_eef``) or at the eef origin without one."""
        robot = self.robot
        link = obj.root_link_name
        joint_type = robot._get_assisted_grasp_joint_type(obj, link)
        if joint_type is None:
            raise LoadError("failed_grasp", f"{obj.name} cannot be assisted-grasped (too heavy?)")
        held = robot._ag_obj_in_hand.get(arm)
        if held is not None and held is not obj:
            robot.release_grasp_immediately(arm)
        if robot._ag_obj_in_hand.get(arm) is not obj:
            eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
            contact = eef[0] if anchor_in_eef is None else C.compose(eef, (th.tensor(anchor_in_eef, dtype=th.float32), C.IDENTITY_QUAT))[0]
            robot._establish_grasp(obj, link, arm, contact, joint_type)

    def _restore_grasps(self, records: List[dict]) -> List[dict]:
        """spec §3.14: the object at ``object_pose_in_eef_frame``, the recorded finger joints, the assisted grasp
        attached before the first policy step; an attach failure fails the load. The held object is re-placed at
        ``eef ∘ object_pose_in_eef_frame`` **unconditionally** before the grasp is established: the §3.5 reload
        check (``validate_instances``) holds every object to 1 mm / 0.005 rad, so a "close enough" gate above that
        (the earlier 2 mm / 0.02 rad) would let a settle drift into the in-hand pose the sampler recorded; the
        deviation found is still reported (``object_pose_dev_before_*``)."""
        robot = self.robot
        out = []
        for g in records:
            arm, inst = g["arm"], g["object"]
            obj = self.physical.get(inst)
            if obj is None:
                raise LoadError("failed_grasp", f"grasp record names {inst!r}, which is not a bound physical instance")
            eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
            ref = g.get("object_pose_in_eef_frame")
            rec = {"arm": arm, "object": inst, "scene_object": obj.name, "grasp_key": g.get("grasp_key"), "grasp_name": g.get("grasp_name"), "finger_q": g.get("finger_q"),
                   "ag_anchor_in_eef": g.get("ag_anchor_in_eef")}
            if ref:
                want = C.compose(eef, (th.tensor(ref["position"], dtype=th.float32), th.tensor(ref["orientation"], dtype=th.float32)))
                have = tuple(x.float() for x in obj.get_position_orientation())
                dp, da = float(th.linalg.norm(want[0] - have[0])), C.quat_angle(want[1], have[1])
                rec["object_pose_dev_before_mm"] = 1000.0 * dp
                rec["object_pose_dev_before_deg"] = math.degrees(da)
                obj.set_position_orientation(want[0], want[1])
                obj.keep_still()
                rec["object_replaced"] = True
            if g.get("finger_q"):
                q = robot.get_joint_positions()
                q[robot.gripper_control_idx[arm]] = th.tensor(g["finger_q"], dtype=th.float32)
                robot.set_joint_positions(q)
            self._establish(arm, obj, g.get("ag_anchor_in_eef"))
            rel = C.relative(tuple(x.float() for x in obj.get_position_orientation()), tuple(x.float() for x in robot.eef_links[arm].get_position_orientation()))
            if ref:
                rec["object_in_eef_dev_mm"] = 1000.0 * float(th.linalg.norm(rel[0] - th.tensor(ref["position"])))
                rec["object_in_eef_dev_deg"] = math.degrees(C.quat_angle(rel[1], th.tensor(ref["orientation"])))
            rec["attached"] = robot._ag_obj_in_hand.get(arm) is obj
            if not rec["attached"]:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} did not attach at load")
            out.append(rec)
            log.info("grasp restored: %s in %s hand (dev %.1f mm / %.1f deg)", obj.name, arm, rec.get("object_in_eef_dev_mm", 0.0), rec.get("object_in_eef_dev_deg", 0.0))
        return out

    # -- accessors ------------------------------------------------------------------------------------------------

    @property
    def goal_image_path(self) -> Optional[str]:
        """The instance's goal image (the TRO's ``goal_observation.rgb_path``)."""
        path = (self.tro.meta.goal_observation or {}).get("rgb_path")
        return None if not path else os.path.join(os.path.dirname(self.tro_path), path)

    @property
    def tracked(self) -> Dict[str, Any]:
        return {k: v for k, v in self.physical.items() if k != AGENT_INSTANCE and getattr(v, "category", None) != "floors"}

    @property
    def rooms(self) -> List[str]:
        return list((self.tro.meta.scene or {}).get("task_rooms") or self.tmeta.get("task_rooms") or [])


# ==========================================================================================
# annotations (§6.3 demo level)
# ==========================================================================================


def iddl_header(subtask: str) -> str:
    """The leading ``;`` comment of the subtask's iddl (the template-level description of §6.3)."""
    try:
        family = family_of_subtask(subtask)
        short = subtask.split("-", 1)[1] if "-" in subtask else subtask
        path = os.path.join(IDDL_ROOT, family, f"{short}.iddl")
        lines = []
        with open(path) as f:
            for line in f:
                if line.startswith(";"):
                    lines.append(line[1:].strip())
                elif lines:
                    break
        return " ".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def episode_annotations(sidecar: Sidecar) -> EpisodeAnnotations:
    subtask = sidecar.subtask
    family = sidecar.task_family
    objects = ", ".join(sorted({o.display_name for o in sidecar.language.objects.values()})) or "no manipulated object"
    template = f"{iddl_header(subtask)} Objects: {objects}.".strip()
    numbers = ", ".join(f"{k} = {v}" for k, v in sorted(sidecar.language.goal_numbers.items()))
    instance = f"{template} Goal: {numbers}." if numbers else template
    return EpisodeAnnotations(
        family=family,
        family_description=FAMILY_DESCRIPTIONS.get(family, ""),
        template_description=template,
        instance_description=instance,
        variant=sidecar.variant,
        instance_id=sidecar.instance_id,
    )


# ==========================================================================================
# replay / policy sources
# ==========================================================================================


def replay_actions(path: str, episode: int = 0) -> Iterator[th.Tensor]:
    """Actions of a recorded episode: a LeRobot dataset root (``meta/info.json`` inside) or a ``demo.hdf5``."""
    if os.path.isdir(path):
        from lerobot.datasets import LeRobotDataset

        ds = LeRobotDataset(repo_id=os.path.basename(path.rstrip("/")), root=path)
        idx = th.nonzero(th.as_tensor(ds.hf_dataset["episode_index"]) == episode).reshape(-1).tolist()
        for i in idx:
            yield th.as_tensor(ds[i]["action"], dtype=th.float32)
        return
    import h5py

    with h5py.File(path, "r") as f:
        actions = f["data"][f"demo_{episode}"]["action"][()]
    for a in actions:
        yield th.as_tensor(a, dtype=th.float32)


def replay_phases(path: str, episode: int = 0) -> Optional[List[int]]:
    """The recorded phase (oracle / settle) of every action of a ``demo.hdf5`` episode (``isg/phase``); None for a LeRobot
    dataset or a demo without it."""
    if os.path.isdir(path):
        return None
    import h5py

    with h5py.File(path, "r") as f:
        g = f["data"][f"demo_{episode}"]
        return [int(p) for p in g["isg/phase"][()]] if "isg/phase" in g else None


def replay_counts(path: str, episode: int = 0, recorded: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """``(action rows, settle rows)`` of a demo: a ``demo.hdf5``'s ``isg/phase`` (0 the program's actions, 1 its settle window,
    in that order), else - a LeRobot episode - the step counts the eval dataset records for the instance's demo (``recorded``:
    its ``episodes.jsonl`` line); raises ValueError without them."""
    phases = replay_phases(path, episode)
    if phases is not None:
        n_action, n_settle = phases.count(PHASE_ORACLE), phases.count(PHASE_SETTLE)
        order = [PHASE_ORACLE] * n_action + [PHASE_SETTLE] * n_settle
        if [p for p in phases if p in (PHASE_ORACLE, PHASE_SETTLE)] != order:
            raise ValueError(f"{path} episode {episode}: isg/phase does not hold the action rows then the settle rows")
        return n_action, n_settle
    if recorded and recorded.get("action_steps") is not None and recorded.get("settle_steps") is not None:
        return int(recorded["action_steps"]), int(recorded["settle_steps"])
    raise ValueError(f"--mode replay of {path}: no action / settle row counts (a demo.hdf5 records isg/phase; a LeRobot episode needs the "
                     "eval dataset's episodes.jsonl counts of its instance)")


def replay_rows(path: str, episode: int, n_action: int, n_settle: int) -> Iterator[th.Tensor]:
    """A demo's ``n_action`` action rows, then the first ``n_settle`` of its settle rows (``run_isg --settle-steps``)."""
    phases = replay_phases(path, episode)
    if phases is None:
        keep = set(range(n_action + n_settle))
    else:
        settle = [i for i, p in enumerate(phases) if p == PHASE_SETTLE]
        keep = {i for i, p in enumerate(phases) if p == PHASE_ORACLE} | set(settle[:n_settle])
    for i, a in enumerate(replay_actions(path, episode)):
        if i in keep:
            yield a


def final_target_poses(task, tro_meta) -> Dict[str, Any]:
    """Every pose target's evaluated pose at the end of the episode, in the target's own frame and link as ``pose_reached``
    compares it (a camera-frame target: in the head camera's frame): ``{inst: {"frame", "link", "position", "orientation",
    "in_goal", "in_measure"}}`` - what ``python -m ispatialgym.eval.instances set-goal --measured-targets`` re-pins."""
    from omnigibson.object_states import PoseReached

    out: Dict[str, Any] = {}
    compiled = getattr(task, "compiled_task", None)
    for inst, tgt in tro_meta.pose_targets.items():
        ent = (task.object_scope or {}).get(inst)
        st = ent.states[PoseReached] if ent is not None and PoseReached in getattr(ent, "states", {}) else None
        if st is None:
            continue
        try:
            pos, quat = st._pose_in_frame(*st._get_current_pose())
        except Exception as e:  # noqa: BLE001  (a fact for set-goal, never a failure)
            out[inst] = {"error": repr(e)}
            continue
        out[inst] = {"frame": tgt.frame, "link": tgt.link, "position": [float(v) for v in pos], "orientation": [float(v) for v in quat],
                     "in_goal": bool(compiled.is_gating("pose_reached", inst)) if compiled is not None else None,
                     "in_measure": bool(compiled.in_measure("pose_reached", inst)) if compiled is not None else None}
    return out


def _dataset_record(loader) -> Optional["INST.InstanceRecord"]:
    """The eval dataset's record of the instance ``loader`` loaded (its ``template_meta["_paths"]`` names the dataset)."""
    tro = os.path.abspath(loader.tro_path)
    return next((r for r in INST.load_index(loader.tmeta["_paths"]["dataset_root"]) if os.path.abspath(r.path("tro_state")) == tro), None)


def load_policy(spec: str, kwargs: Optional[Dict[str, Any]] = None):
    """``module:callable`` → ``policy(obs, info) -> action``."""
    from ispatialgym.eval.policy import import_policy, make_policy

    return make_policy(import_policy(spec), kwargs)


def policy_actions(policy, executor: Executor, obs0, info0, max_steps: int) -> Iterator[th.Tensor]:
    obs, info = obs0, info0
    for _ in range(int(max_steps)):
        a = policy(obs, info)
        if a is None:
            return
        yield th.as_tensor(a, dtype=th.float32).reshape(-1)
        obs, info = executor.last_obs, executor.last_info


# ==========================================================================================
# main
# ==========================================================================================


INSTANCE_OPTIONS = ("task", "instance_indices", "instances_root", "instance", "queue", "claims", "done_file", "results", "deadline", "max_instances", "poll_s", "instance_list")


def record_views(text: str) -> Tuple[str, ...]:
    """``--record-view``: a comma-separated list of :data:`~ispatialgym.oracle.recorder.VIEWS`."""
    views = tuple(dict.fromkeys(v.strip() for v in text.split(",") if v.strip()))
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown view(s) {', '.join(unknown)}; choose from {', '.join(VIEWS)}")
    return views


def add_lean_step_argument(p) -> None:
    """``--lean-step`` of both drivers (``run_isg_vec``'s default): ``ISG_VEC_LEAN_STEP`` for the overlay's ``Simulator._non_physics_step``."""
    p.add_argument("--lean-step", action=argparse.BooleanOptionalAction, default=True,
                   help="ISG_VEC_LEAN_STEP=1 for the overlay's Simulator._non_physics_step: no per-object update_visuals() loop (the texture "
                        "changes of Burnt / Cooked / Frozen / Saturated / ToggledOn and their emitters; no ISG task changes a visual state, "
                        "nothing the physics or the verdicts read). run_isg_vec's default; --no-lean-step = stock")


def apply_driver_env(args) -> None:
    """The environment the drivers' physics flags set for the code that reads it: ``ISG_SLEEP_ASSIST*`` (``sleep_assist.export_env``)
    and ``ISG_VEC_LEAN_STEP``; :func:`_run_with_args` calls it for both entry points (this module's and ``ispatialgym.run_isg``)."""
    SA.export_env(args)
    os.environ["ISG_VEC_LEAN_STEP"] = "1" if getattr(args, "lean_step", True) else "0"


def qa_config_for_budget(budget, mode, max_action_steps=None):
    """Resolve a finite action timeout while preserving every physical QA rule."""
    cfg = QAConfig(max_episode_steps=budget["episode_cap"])
    if max_action_steps is not None:
        if type(max_action_steps) is not int or not 1 <= max_action_steps <= budget["max_steps"]:
            raise ValueError("--max-action-steps must be a positive integer no larger than --max-steps")
        cfg.max_action_steps = max_action_steps
    elif mode != "oracle":
        # A replay / policy episode is one stream action.
        cfg.max_action_steps = budget["episode_cap"]
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML file of option values; command-line options override it")
    p.add_argument("--instance", default=None, help="path to <iid>-tro_state.json of a sampled instance")
    p.add_argument("--task", default=None,
                   help="instead of --instance: a task family, task or subtask of the eval dataset (ispatialgym.eval.instances); runs "
                        "its instances --instance-indices (one in this process, several each in a process of its own)")
    p.add_argument("--instance-indices", type=int, nargs="+", default=[0],
                   help="--task: instance indices (a task's instances are its subtasks' instances in the dataset's order)")
    p.add_argument("--instances-root", default=None,
                   help="--task: dataset root (default $ISPATIALGYM_INSTANCES_PATH or $OMNIGIBSON_DATA_PATH/ispatialgym-eval-instances)")
    p.add_argument("--instance-list", default=None, metavar="FILE", help="instead of --instance: a file of eval dataset instance files (one *-tro_state.json per line) run one after the other in one world (the first builds it, the others attach to it: run_queue's world reuse); every instance must share the first one's scene and BDDL problem, and its template must be compatible (scripts/policy_eval/world_groups.py groups them)")
    p.add_argument("--queue", default=None, metavar="INST_ROOT",
                   help="instead of --instance: replay every instance a sampler writes under INST_ROOT (sample_isg --out), one after "
                        "the other in this process's world (run_queue)")
    p.add_argument("--claims", default=None, help="--queue: directory of the claim files (default: <INST_ROOT>/../claims)")
    p.add_argument("--done-file", default=None, help="--queue: once this file exists, exit when no ready instance is left unclaimed")
    p.add_argument("--results", default=None, help="--queue: append one JSON line per instance to this file")
    p.add_argument("--deadline", type=float, default=None, help="--queue: epoch seconds after which no instance starts")
    p.add_argument("--max-instances", type=int, default=None, help="--queue: instances per process")
    p.add_argument("--poll-s", type=float, default=10.0, help="--queue: wait between scans for new instances")
    p.add_argument("--prebuild", default=None, metavar="TRO",
                   help="--queue (train-fast COLLECTOR_PREBUILD): build the world from this instance of the same template (e.g. the counted eval "
                        "instance) before the queue has any, so the scene load overlaps the sampler's; the first real instance must share its "
                        "template_key, else <claims>/../prebuild.mismatch is written and the process exits for a restart without it")
    p.add_argument("--restoration-policy-manifest", default=None, metavar="JSON",
                   help="Exact bounded per-source shared-loader policy manifest; does not release production")
    p.add_argument("--restoration-policy-manifest-sha256", default=None, metavar="SHA256",
                   help="Immutable SHA256 of --restoration-policy-manifest")
    p.add_argument("--pool-world-request", default=None, metavar="JSON",
                   help="Bounded exact-source native pool replay request; source arguments and driver settings are preserved")
    p.add_argument("--pool-world-request-sha256", default=None, metavar="SHA256",
                   help="Immutable SHA256 of --pool-world-request")
    p.add_argument("--sidecar", default=None, help="oracle sidecar (default: <iid>.oracle.json next to the TRO)")
    p.add_argument("--mode", default="oracle", choices=["oracle", "replay", "policy"])
    p.add_argument("--replay", default=None, help="replay mode: LeRobot dataset root or demo.hdf5 (default for an eval dataset instance: its demo)")
    p.add_argument("--episode", type=int, default=0, help="replay mode: episode index")
    p.add_argument("--policy", default=None, help="policy mode: module:callable -> policy(obs, info) -> action")
    p.add_argument("--policy-kwargs", type=json.loads, default={}, metavar="JSON", help="keyword arguments of the policy class")
    p.add_argument("--out", default=None, help="output root; the run goes to <out>/<iid>/ (default: <template_dir>/runs)")
    p.add_argument("--record", default="full", choices=["full", "hdf5", "video"],
                   help="full = hdf5 + LeRobot; hdf5 = hdf5 only; video = the --record-view videos and meta only")
    p.add_argument("--record-view", type=record_views, default=(), metavar="VIEW[,VIEW...]",
                   help="videos to record, each to <out>/<iid>/<view>_rgb.mp4: head (head camera), external (BEHAVIOR's JoyLo replay "
                        "camera over the robot's shoulder, 1440 x 960), left_wrist / right_wrist (wrist cameras), tiled (BEHAVIOR's "
                        "JoyLo replay video: left wrist | head | right wrist in 480 px tiles over the external camera, 1440 x 1440)")
    p.add_argument("--video-size", type=int, default=None, help="scale every video to this width (aspect kept)")
    p.add_argument("--goal-image", action="store_true", help="capture the goal image after the settled check (>= 3 renders) and write goal_observation into the TRO")
    p.add_argument("--cross-goal", action="store_true", help="policy mode, eval dataset: condition the policy on the instance's cross-episode goal image (files.cross_episode_goal_rgb: the goal image of another episode of its goal group) instead of its own goal image; an instance without one (singleton goal group) keeps its own, recorded in meta.policy_client.goal_kind")
    p.add_argument("--no-goal-override", action="store_true", help="do not write the settled values back into the TRO goal poses")
    p.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True,
                   help="the QA metrics of BEHAVIOR's demo QA plus the demo gates (ispatialgym.oracle.qa.QA_METRICS), logged in meta.qa "
                        "with the steps each fired at. HARD: robot touching objects / itself, pause, jerky motion, ghost hand, object "
                        "tilt, goal occlusion, robot wobble, bad release, arm stutter; SOFT: held object drift and stalled, gripper in view and failed "
                        "grasp (oracle only) - the oracle's retry rules, retried as before and every verdict logged -, head camera "
                        "upright, arm / hand contacts. --mode oracle: a HARD metric ends the demo as failed; replay / policy: every "
                        "metric is only logged. --no-qa: none of them (the oracle's retry rules still apply)")
    p.add_argument("--carry-leg", default="straight", choices=["straight", "trajopt"],
                   help="the carry leg lift end -> pre-place: straight Cartesian leg through the recorded carry waypoint (FOV rule (b), default) or the former free trajopt transit (regression reference)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="steps of the actions (policy steps, a replay, the oracle program; 30 Hz control steps); default: an eval dataset "
                        "instance max(300, round(1.5 x its demo's length)), else 3000. An oracle episode is cut at --max-steps + --settle-steps "
                        "(its epilogue), a replay / policy episode at --max-steps; without geometric success a policy runs to it")
    p.add_argument("--max-action-steps", type=int, default=None,
                   help="finite step limit for each action, no larger than --max-steps; default: 3000 for oracle, "
                        "the episode budget for replay / policy. Other QA thresholds are unchanged")
    p.add_argument("--settle-steps", type=int, default=None,
                   help="the pose_settled window and the --settle-eval window, in steps, in every mode: oracle = its epilogue, exactly this "
                        "long and reserved in the step budget (default 45); replay = the demo's action rows and only the first this-many of "
                        "its settle rows are streamed (default: all of them; more than the demo has is an error); policy = the window only "
                        "(default 45)")
    p.add_argument("--wallclock", type=float, default=None,
                   help="seconds the actions may take (default 1 s x --max-steps): a replay / policy stream ends there and the task's "
                        "metrics judge the episode; the oracle program gets a timeout verdict")
    p.add_argument("--settle-eval", default="auto", choices=["auto", "on", "off"],
                   help="how symbolic and geometric success are decided (ispatialgym.eval.metrics): off = the first step the goal holds "
                        "(every goal atom but pose_settled; geometric: with every (:measure ...) atom within tolerance at that step); on = "
                        "the first step every pose_settled atom holds with the goal (geometric: and the measured targets) held at every step "
                        "of its --settle-steps window; auto = off for --mode policy, on for oracle / replay. With on, the off mode is "
                        "reported too (meta.other_mode; null with off)")
    p.add_argument("--success-metric", default="geometric", choices=["geometric", "symbolic"],
                   help="what success, status and the exit code follow: geometric = the BDDL goal plus every (:measure ...) atom within "
                        "tolerance (default), symbolic = the BDDL goal; the episode ends at the geometric success either way")
    p.add_argument("--tolerance-scale", type=float, default=None,
                   help="multiplies every isg-v2 tolerance, overrides included (ispatialgym.eval.tolerance_profile); default 3 with "
                        "--mode policy, else 1")
    p.add_argument("--retries", type=int, default=1, help="retries per action from its checkpoint")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--head-aperture", type=float, default=DEFAULT_HEAD_APERTURE_MM)
    p.add_argument("--robot-name", default=None)
    p.add_argument("--allow-sha-mismatch", action="store_true", help="run a sidecar whose tro_sha256 does not match the TRO (hand-built sidecars)")
    p.add_argument("--frame-buffer", default="jpeg", choices=["jpeg", "raw"], help="how pending rgb frames are buffered until an action commits")
    p.add_argument("--all-cameras", action="store_true", help="LeRobot: every rgb sensor (default: the head camera only)")
    p.add_argument("--behavior-shard", default=None,
                   help="<root>/<repo_id>: append each successful demo without a mid-episode retry to this LeRobot v3 dataset in BEHAVIOR's "
                        "replay_obs.py format (robot with replay_obs's sensors: head 720 px, wrists 480 px, rgb + depth_linear + proprio)")
    # the physics of run_isg_vec, the same flags and defaults (demos are generated here, policies evaluated there)
    SA.add_arguments(p)
    add_lean_step_argument(p)
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    p.add_argument("--load-only", action="store_true", help="load the instance, record its start state (meta.start_state) and exit: no program, no demo, no "
                                                            "TRO writes - the fresh-reload reference a warm demo's start state is compared with")
    return p


def config_argv(path: str, parser: argparse.ArgumentParser) -> List[str]:
    """A YAML config as command-line tokens, so argparse converts and validates its values like the options."""
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: a config is a mapping of option names to values")
    known = {
        a.dest: a.option_strings[0] for a in parser._actions if a.option_strings and a.dest not in ("help", "config")
    }
    unknown = sorted(set(config) - set(known))
    if unknown:
        raise SystemExit(f"{path}: unknown option(s) {unknown}; known: {sorted(known)}")
    negatable = {a.dest for a in parser._actions if isinstance(a, argparse.BooleanOptionalAction)}  # --qa / --no-qa, --lean-step / --no-lean-step
    argv: List[str] = []
    for key, value in config.items():
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue
        if key == "settle_eval" and isinstance(value, bool):  # YAML reads a bare on / off as a boolean
            value = "on" if value else "off"
        if key == "sleep_assist" and isinstance(value, bool):  # likewise: off, or on = the default window
            value = parser.get_default("sleep_assist") if value else "off"
        if key == "record_view" and isinstance(value, (list, tuple)):
            value = ",".join(value)
        if isinstance(value, bool):
            if key in negatable:
                argv.append(known[key] if value else "--no-" + known[key][2:])
            elif value:
                argv.append(known[key])
            continue
        if isinstance(value, dict):
            value = json.dumps(value)
        argv += [known[key], *map(str, value)] if isinstance(value, (list, tuple)) else [known[key], str(value)]
    return argv


def run_options(args: argparse.Namespace) -> Dict[str, Any]:
    """The resolved options of one instance's run (``config.yaml``)."""
    skip = set(INSTANCE_OPTIONS) | {"config"}
    out = {k: v for k, v in vars(args).items() if k not in skip}
    out["record_view"] = list(out["record_view"])
    return out


def parse_args(argv=None, default_policy: bool = False):
    p = build_parser()
    if default_policy:
        p.set_defaults(mode="policy", out="runs", settle_eval="off", settle_steps=45, tolerance_scale=3.0)
    argv = list(sys.argv[1:] if argv is None else argv)
    config = p.parse_known_args(argv)[0].config
    args = p.parse_args((config_argv(config, p) if config else []) + argv)
    if sum(x is not None for x in (args.instance, args.queue, args.task, getattr(args, "instance_list", None))) != 1:
        p.error("exactly one of --instance, --queue, --task and --instance-list")
    if args.mode == "policy" and not args.policy:
        p.error("--policy is required")
    if args.mode == "policy" and not isinstance(args.policy_kwargs, dict):
        p.error("--policy-kwargs must be a JSON object")
    if args.task and args.replay and len(args.instance_indices) > 1:
        p.error("--replay names one demo: give one --instance-indices value (without --replay each instance replays its own demo)")
    if args.all_cameras and M.head_camera_only():
        p.error("--all-cameras records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    wrist_views = [v for v in args.record_view if v in ("tiled", "left_wrist", "right_wrist")]
    if wrist_views and M.head_camera_only():
        p.error(f"--record-view {','.join(wrist_views)} records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    if args.behavior_shard and M.head_camera_only():
        p.error("--behavior-shard records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    if args.behavior_shard and args.record not in ("full", "hdf5"):
        p.error("--behavior-shard needs demo.hdf5 (--record hdf5 or full): retried demos are replayed from it")
    return args


def _result_line(result: dict) -> None:
    print("RUN_ISG_RESULT " + json.dumps(result, default=json_default), flush=True)


def write_tro_updates(tro_path: str, sidecar_path: Optional[str], sidecar: Optional[Sidecar], override: Optional[Dict[str, Any]], goal_obs: Optional[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    """The epilogue's writes into the TRO (goal override, goal image) and the sidecar's ``tro_sha256`` refresh."""
    with open(tro_path) as f:
        raw = json.load(f)
    changed: List[str] = []
    if override is not None:
        changed += apply_goal_override(raw, override, TRO_KEY)
    if goal_obs is not None and goal_obs.get("rgb") is not None:
        png = os.path.join(out_dir, "goal_rgb.png")
        save_png(png, goal_obs["rgb"])
        block = raw[TRO_KEY].setdefault("goal_observation", {}) or {}
        block.update({
            "sensor_key": goal_obs["sensor_key"],
            "rgb_path": os.path.relpath(png, os.path.dirname(tro_path)),
            "rgb_sha256": sha256_of_file(png),
            "resolution": goal_obs["resolution"],
            "intrinsics": goal_obs.get("intrinsics"),
            "camera_pose_world": goal_obs["camera_pose_world"],
            "evaluation_profile": EVALUATION_PROFILE,
            "n_render_iterations": 1,
            "n_renders": goal_obs.get("n_renders"),
        })
        raw[TRO_KEY]["goal_observation"] = block
        changed.append("goal_observation")
    out = {"changed": changed, "tro_sha256_before": sha256_of_file(tro_path)}
    if changed:
        with open(tro_path, "w") as f:
            json.dump(raw, f, indent=2, default=json_default)
        out["tro_sha256_after"] = sha256_of_file(tro_path)
        if sidecar is not None and sidecar_path and os.path.exists(sidecar_path):
            sidecar.tro_sha256 = out["tro_sha256_after"]
            sidecar.save(sidecar_path)
            out["sidecar_updated"] = sidecar_path
    return out


def _apply_program_hook(spec: str, actions, sidecar, loader):
    """``ISG_PROGRAM_HOOK``: ``/path/to/file.py:callable`` or ``package.module:callable``; the callable gets
    ``(actions, sidecar, loader)`` and returns the program to run."""
    import importlib
    import importlib.util

    target, _, fn_name = spec.rpartition(":")
    if target.endswith(".py"):
        spec_ = importlib.util.spec_from_file_location("isg_program_hook", target)
        mod = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = mod  # dataclasses with postponed annotations resolve them through sys.modules
        spec_.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = importlib.import_module(target)
    out = getattr(mod, fn_name)(list(actions), sidecar, loader)
    log.info("program hook %s: %d -> %d actions (%s)", spec, len(actions), len(out), [a.name for a in out])
    return out


def main(argv=None, default_policy: bool = False) -> int:
    args = parse_args(argv, default_policy=default_policy)
    return _run_with_args(args, argv)


def _run_with_args(args, argv=None) -> int:
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    from ispatialgym.oracle.pool_inputs import enabled as pool_enabled, run as run_pool

    if pool_enabled(args) and not args.queue:
        return run_pool(sys.modules[__name__], args)
    apply_driver_env(args)
    if getattr(args, "instance_list", None):
        return run_instance_list(args)
    if args.queue:
        return run_queue(args)
    if args.task:
        return run_task(args, argv)
    return run_instance(args, args.instance)[0]


def run_instance_list(args) -> int:
    """``--instance-list FILE`` (policy eval 2026-11-16): eval dataset instances one after the other in one world - the first builds it, each
    next one attaches to it (:meth:`InstanceLoader.attach`, run_queue's world reuse: the previous episode's hinges / grasps / attachment
    joints removed, the whole scene back to its state after the build, the planner map back to the template's, then the instance's own
    restore). Every instance must name the first one's scene directory and BDDL problem (the caller groups instances whose templates are
    compatible outside their task objects: scripts/policy_eval/world_groups.py). One ``RUN_ISG_RESULT`` line per instance; stops at the
    first instance that errors (the world may be inconsistent), returning its exit code, else the largest exit code."""
    with open(args.instance_list) as f:
        tros = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    first = resolve_instance(tros[0])
    for tro in tros[1:]:
        r = resolve_instance(tro)
        if os.path.abspath(r[1]) != os.path.abspath(first[1]) or r[2].get("problem") != first[2].get("problem"):
            raise SystemExit(f"--instance-list: {tro} has scene dir / problem {r[1]} / {r[2].get('problem')}, the world "
                             f"{first[1]} / {first[2].get('problem')}")
    shared: Dict[str, Any] = {}
    worst = EXIT_SUCCESS
    for k, tro in enumerate(tros):
        t0 = time.time()
        code, result = run_instance(args, tro, shared)
        log.info("--instance-list %d/%d %s: rc %s in %.0fs (reused world: %s)", k + 1, len(tros), os.path.basename(tro), code,
                 time.time() - t0, k > 0)
        if code == EXIT_ERROR:
            return code
        worst = max(worst, code)
    return worst


def run_task(args, argv=None) -> int:
    """``--task``: the eval dataset instances the name and indices resolve to; one runs in this process, several each in a
    process of its own (an Isaac Sim process builds one environment)."""
    import subprocess

    from ispatialgym.eval import instances as INST

    recs = INST.resolve(args.task, args.instance_indices, args.instances_root)
    for i, rec in zip(args.instance_indices, recs):
        log.info("--task %s index %d -> subtask %s instance %d (%s)", args.task, i, rec.subtask, rec.instance, rec.files["tro_state"])
    if len(recs) == 1:
        return run_instance(args, recs[0].path("tro_state"))[0]
    rest, src, i = [], list(sys.argv[1:] if argv is None else argv), 0
    while i < len(src):
        if src[i] in ("--task", "--instances-root"):
            i += 2
            continue
        if src[i] == "--instance-indices":
            i += 1
            while i < len(src) and src[i].lstrip("-").isdigit():
                i += 1
            continue
        rest.append(src[i])
        i += 1
    codes = [subprocess.run([sys.executable, "-m", "ispatialgym.oracle.run_isg", *rest, "--instance", rec.path("tro_state")]).returncode for rec in recs]
    return max(codes)


def run_instance(args, tro_arg: str, shared: Optional[Dict[str, Any]] = None, *, restoration_policy=None, pool_consumers=None, pool_world=None) -> Tuple[int, Dict[str, Any]]:
    """One instance end to end: load, oracle stack, episode, recording, ``meta.json`` and the ``RUN_ISG_RESULT`` line; returns
    ``(exit code, result)``. ``shared`` (``--queue``): the world and planner of the previous instance of the same template,
    reused (:meth:`InstanceLoader.attach`); the first instance fills it in."""
    t_start = time.time()
    timings: Dict[str, float] = {}
    fast_meta: Dict[str, Any] = {}  # 2026-10-31 train-fast: what the switches did this run (meta["fast"], written in the finally)
    meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "provenance": provenance(),
                            "reused_world": bool(shared and shared.get("env") is not None)}
    result: Dict[str, Any] = {"status": "error", "mode": args.mode, "provenance": meta["provenance"]}
    # ISPATIALGYM_GROUP_ASSIGN_POSE_TIEBREAK (i7-blockdiag): this episode's own counters (group_entity imports omnigibson, so it
    # is only touched once loaded)
    _ge = sys.modules.get("ispatialgym.eval.group_entity")
    if _ge is not None:
        _ge.pose_tiebreak_reset()
    exit_code = EXIT_ERROR
    out_dir = None
    recorder: Optional[Recorder] = None
    executor: Optional[Executor] = None
    sidecar: Optional[Sidecar] = None
    sidecar_path: Optional[str] = None

    try:
        from ispatialgym.oracle.restoration_policy_inputs import from_arguments

        restoration_policy = from_arguments(args, tro_arg,
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), provided=restoration_policy)
        image_size = int(args.image_size)
        if PO.trim() and image_size == DEFAULT_IMAGE_SIZE:
            # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): no frame is recorded, yet every step copies each camera buffer - the cameras
            # run small (an explicit --image-size is kept); the head rgb sensor stays, so cam_key / camera_pose / the FOV monitor work
            image_size = int(fast_value("PHYSICS_ONLY_IMAGE_SIZE", 128))
            fast_meta["physics_only_trim"] = {"image_size": image_size, "skipped": []}
            log.info("train-fast PHYSICS_ONLY_TRIM: cameras at %d x %d (physics-only: no frame is recorded)", image_size, image_size)
        loader = InstanceLoader(tro_arg, robot_name=args.robot_name, image_size=image_size, head_aperture=args.head_aperture,
                                restoration_policy=restoration_policy, pool_consumers=pool_consumers, pool_world=pool_world)
        loader.external_camera = bool({"external", "tiled"} & set(args.record_view))
        loader.behavior_output = bool(args.behavior_shard)
        loader.behavior_proprio = args.mode == "policy"  # policy eval: the training data's 61-D proprioception
        loader.physics_overrides = SA.rigid_overrides(args)
        # run_isg_vec's physics (oracle.sleep_assist; ISG_VEC_LEAN_STEP): what this run used, for the demo's provenance
        assist = {"config": os.environ.get("ISG_SLEEP_ASSIST"), "mode": SA.sleep_assist_mode(), "max_mass_kg": SA.sleep_assist_max_mass(),
                  "wake_radius_m": SA.sleep_assist_wake_radius()}
        if SA.restart_on_wake():  # i7-sleep (A8b): recorded only when on
            assist["restart_on_wake"] = True
        meta["driver_physics"] = {"rigid_overrides": dict(loader.physics_overrides), "sleep_assist": assist if SA.sleep_assist_config() is not None else None,
                                  "lean_step": os.environ.get("ISG_VEC_LEAN_STEP") == "1"}
        meta["record_view"] = list(args.record_view)
        iid, tro_path = loader.iid, loader.tro_path
        # an eval dataset instance (template_meta "_paths") never gets runs written into the dataset
        out_root = args.out or (os.path.join(os.getcwd(), "runs") if loader.tmeta.get("_paths") else os.path.join(loader.template_dir, "runs"))
        out_dir = os.path.join(out_root, iid)
        os.makedirs(out_dir, exist_ok=True)
        try:
            import yaml

            with open(os.path.join(out_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(run_options(args), f, sort_keys=False)
        except Exception as e:  # noqa: BLE001
            log.warning("config.yaml not written: %r", e)
        meta["config"] = run_options(args)
        meta.update({
            "instance_id": iid, "template_dir": loader.template_dir, "template_id": loader.tmeta.get("template_id"), "problem": loader.tmeta["problem"],
            "activity": loader.activity, "activity_definition_id": loader.did, "variant": loader.tmeta.get("variant") or loader.tro.meta.task_variant,
            "scene": loader.tmeta.get("scene"), "load_room_instances": loader.tmeta.get("load_room_instances"), "tro_path": tro_path,
            "goal_image": loader.goal_image_path, "robot_name": loader.robot_name, "out_dir": out_dir,
        })
        # task family / task / subtask (isg_customs.json is keyed by subtask): the eval dataset's names, else from the sampling key
        names = {k: v for k, v in (loader.tmeta.get("eval") or {}).items() if k in ("task_family", "task", "subtask")}
        if not names and meta.get("variant"):
            from ispatialgym.eval import names as NAMES

            try:
                names = {"task_family": NAMES.task_family(meta["variant"]), "task": NAMES.task_name(meta["variant"]), "subtask": NAMES.subtask_name(meta["variant"])}
            except Exception as e:  # noqa: BLE001  (a key outside the iddl name templates: no customs entry)
                log.warning("no subtask name for %s (%r)", meta["variant"], e)
        meta.update(names)
        # an eval dataset instance's demo: the default --replay, and its length the step budget
        dataset_rec = _dataset_record(loader) if loader.tmeta.get("_paths") else None
        if dataset_rec is not None:  # the instance's identity in the eval dataset (scripts/policy_eval/summarize.py groups a results dir by it)
            meta["eval_instance"] = {"index": dataset_rec.position, "instance_id": dataset_rec.instance_id,
                                     "class_id": (dataset_rec.source or {}).get("class_id"), "task_family": dataset_rec.task_family,
                                     "task": dataset_rec.task, "subtask": dataset_rec.subtask, "dataset_root": dataset_rec.root}
        replay, replay_n = None, None
        if args.mode == "replay":
            replay = args.replay or (dataset_rec.demo_path() if dataset_rec else None)
            if not replay:
                raise ValueError("--mode replay needs --replay <dataset root | demo.hdf5> (an eval dataset instance replays its own demo)")
            replay_n = replay_counts(replay, args.episode, dataset_rec.episode if dataset_rec else None)
        length = dataset_rec.episode.get("length") if dataset_rec else (sum(replay_n) if replay_n else None)
        budget = INST.step_budget(length, args.max_steps, args.settle_steps, args.wallclock, mode=args.mode,
                                  demo_settle_steps=replay_n[1] if replay_n else None,
                                  instance_max_steps=dataset_rec.max_steps if dataset_rec else None)
        qa_cfg = qa_config_for_budget(budget, args.mode, getattr(args, "max_action_steps", None))
        budget["max_action_steps"] = qa_cfg.max_action_steps
        settle_eval = args.settle_eval == "on" or (args.settle_eval == "auto" and args.mode != "policy")
        meta.update({"budget": budget, "settle_eval": settle_eval, "settle_steps": budget["settle_steps"], "success_metric": args.success_metric})
        result.update({"instance": iid, **names, "activity": loader.activity, "variant": meta["variant"], "out": out_dir})

        # ---------------------------------------------------------------- sidecar + program (before the sim: cheap failures first)
        from ispatialgym.sampling.campaign_class_routes import verify_recorded
        class_route = verify_recorded(loader.tro.meta.raw.get('campaign') or {},
                                      phase='oracle' if args.mode == 'oracle' else 'replay')
        if class_route is not None:
            meta['class_route'] = class_route
        actions = None
        program_error = None
        if args.mode == "oracle":
            sidecar_path = os.path.abspath(args.sidecar) if args.sidecar else default_sidecar_path(tro_path)
            if not os.path.exists(sidecar_path):
                program_error = f"no oracle sidecar at {sidecar_path}"
            else:
                sidecar = Sidecar.load(sidecar_path).validate(check_program=True)
                meta["sidecar"] = sidecar_path
                try:
                    sidecar.verify_tro(tro_path)
                    meta["tro_sha256_verified"] = True
                except SidecarError as e:
                    if not args.allow_sha_mismatch:
                        raise
                    meta["tro_sha256_verified"] = False
                    log.warning("%s (--allow-sha-mismatch)", e)
                variant = sidecar.variant
                try:
                    actions = expand(variant, sidecar)
                except SidecarError as e:
                    program_error = f"no program for {variant!r}: {e}"
                if actions is not None and os.environ.get("ISG_PROGRAM_HOOK"):
                    # experiments (owner 2026-09-19 night: the door pull-back / push-open study): ``path.py:callable`` or
                    # ``module:callable`` rewriting the expanded program; recorded in meta, never used by the pilot runs
                    actions = _apply_program_hook(os.environ["ISG_PROGRAM_HOOK"], actions, sidecar, loader)
                    meta["program_hook"] = os.environ["ISG_PROGRAM_HOOK"]
                meta["iddl"] = sidecar.subtask  # the task's iddl id (meta["subtask"] is the subtask name, the isg_customs key)
                meta["program"] = [a.summary() for a in actions] if actions else None
        qa_program = actions
        if args.mode != "oracle" and args.qa:
            # a stream has no program: object_tilt watches the free-standing objects the instance's oracle program opens or closes
            qa_sidecar = os.path.abspath(args.sidecar) if args.sidecar else default_sidecar_path(tro_path)
            try:
                _sc = Sidecar.load(qa_sidecar).validate(check_program=True)
                qa_program = expand(_sc.variant, _sc)
            except Exception as e:  # noqa: BLE001
                log.info("--qa object_tilt: no program for this instance (%r); no object is watched", e)
            meta["qa_program_sidecar"] = qa_sidecar if qa_program else None
        log.info("run_isg v2: %s (%s, mode %s, %s actions) -> %s", iid, loader.activity, args.mode, None if actions is None else len(actions), out_dir)

        # ---------------------------------------------------------------- environment + loader contract
        t0 = time.time()
        if shared is not None and shared.get('sequential_augmentation') is not None:
            loader._sequential_augmentation = shared['sequential_augmentation']
        if shared is not None and shared.get("env") is not None:
            env = loader.attach(shared)
        else:
            env = loader.build_env()
            if shared is not None:
                loader.share(shared)
            if GC_FREEZE:
                gc.collect()
                gc.freeze()
        if shared is not None:
            shared["loader"] = loader
        # the rooms the world actually loaded (rooms_for_template: the template's list, or the ISG_ROOMS_OVERRIDE entry of this variant;
        # a PARTIAL_SCENE crop's own rooms) - recorded once the world exists: the loader's cfg is built by build_env / taken by attach
        loaded = loader.cfg["scene"].get("load_room_instances")
        meta["load_rooms"] = {"rooms": loaded, "template_rooms": loader.tmeta.get("load_room_instances"),
                              "mode": "env-override" if loaded != loader.tmeta.get("load_room_instances") else "template",
                              "source": os.environ.get("ISG_ROOMS_OVERRIDE")}
        crop = (loader.tmeta.get("scene_crop") or {}) if isinstance(loader.tmeta.get("scene_crop"), dict) else {}
        if loader.scene_crop:
            # 2026-11-01 train-fast (PARTIAL_SCENE): which world the demo was collected in ("scene" stays the scene model name)
            meta["scene_crop"] = {"crop": loader.scene_crop.get("mode"), "loaded_scene_file": loader.cfg["scene"]["scene_file"],
                                  "loaded_rooms": loader.cfg["scene"]["load_room_instances"], "n_objects_loaded": len(env.scene.objects),
                                  "radius_m": loader.scene_crop.get("radius_m"), "center_object": loader.scene_crop.get("center_object"),
                                  "n_dropped": loader.scene_crop.get("n_dropped")}
        elif (crop.get("mode") or "full") != "full":
            meta["scene_crop"] = {"crop": "full", "template_crop": crop.get("mode"), "why": "ISPATIALGYM_FAST_PARTIAL_SCENE off",
                                  "n_objects_loaded": len(env.scene.objects)}
        loader.task.configure_evaluation(settle_eval, budget["settle_steps"])  # before the reset in restore activates pose_settled
        try:
            obs, info = loader.restore()
        except LoadError as e:
            meta.update({"status": "oracle_failed", "reason": e.reason, "load_error": e.message, "loader": loader.report})
            result.update({"status": "oracle_failed", "success": False, "reason": e.reason, "n_steps": 0, "error": e.message})
            exit_code = EXIT_ORACLE_FAILED
            return exit_code, result
        timings["load_s"] = time.time() - t0
        task, robot, cam_key = loader.task, loader.robot, loader.cam_key
        meta["loader"] = loader.report
        meta["start_state"] = start_state.fingerprint(loader)  # what the executor starts from (engineering track: warm vs fresh loads)
        if args.load_only:
            meta.update({"status": "loaded"})
            result.update({"status": "loaded", "success": True, "start_state_sha256": meta["start_state"]["sha256"]})
            exit_code = EXIT_SUCCESS
            return exit_code, result
        # the isg-v2 tolerances, from the state after the loader's settle, before the first step (the dataset is never written)
        tolerance_profile = task.apply_tolerance_profile(args.mode, scale=args.tolerance_scale)
        meta["tolerance_profile"] = tolerance_profile
        rp, rq = robot.get_position_orientation()
        meta["robot_base_init"] = {"position": rp.tolist(), "orientation": rq.tolist()}
        # owner 2026-10-11 11:46: textures stream in after the load - render (never step) until they are loaded, then capture
        # init_rgb.png and the recorder's first frame from that render
        from ispatialgym.oracle import texture_wait as _TW

        t_tex = time.time()
        if PO.trim():
            # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): the texture wait (render-only frames, up to ISG_TEXTURE_WAIT_MAX_S) and
            # init_rgb.png exist for the recorded frames; a physics-only run records none - skipped, noted in meta
            meta["texture_wait"] = {"skipped": "physics_only", "complete": None, "renders": 0, "wait_s": 0.0}
            meta["init_rgb"] = {"skipped": "physics_only"}
            fast_meta.setdefault("physics_only_trim", {"skipped": []})["skipped"] += ["texture_wait", "init_rgb.png", "reset_renders"]
            log.info("train-fast PHYSICS_ONLY_TRIM: texture wait and init_rgb.png skipped (physics-only: no frame is recorded)")
        else:
            meta["texture_wait"], obs_tex = _TW.wait_for_textures(lambda: SE.get_obs(env)[0], og.sim.render, robot.name, EXTERNAL_CAMERA_NAME if loader.external_camera else None)
            if obs_tex is not None:
                obs = obs_tex
            try:
                save_png(os.path.join(out_dir, "init_rgb.png"), obs[robot.name][cam_key]["rgb"])
            except Exception as e:  # noqa: BLE001
                log.warning("init render not saved: %r", e)
        timings["texture_wait_s"] = time.time() - t_tex
        tracked = loader.tracked
        meta["tracked_objects"] = {k: v.name for k, v in tracked.items()}
        log.info("instance loaded in %.1fs (base %s, %d grasps, %d hinges)", timings["load_s"], [round(float(v), 3) for v in rp], len(loader.grasps), len(loader.hinges))

        if args.mode == "oracle" and actions is None:
            meta.update({"status": "no_oracle", "reason": program_error})
            result.update({"status": "no_oracle", "reason": program_error, "success": False, "n_steps": 0})
            exit_code = EXIT_NO_ORACLE
            return exit_code, result

        # ---------------------------------------------------------------- oracle stack
        t0 = time.time()
        arms = sorted({arm for a in (actions or []) for arm in a.arms}) or ["right"]
        prebuild = tuple(f"{arm}_arm" for arm in arms)
        if shared is not None and shared.get("backend") is not None:
            backend = shared["backend"]
        else:
            room_filter = lambda: make_obstacle_filter(task_bounds(env.scene, loader.rooms) if loader.rooms else None, robot,  # noqa: E731
                                                       [o.name for o in tracked.values()])
            if loader.pool_consumers is not None:
                native_room_filter = room_filter
                room_filter = lambda: loader.pool_consumers.obstacle_filter(native_room_filter())
            planner_kw = executor_mesh_cache_kw(robot, room_filter)
            if fast_on("PLAN_LADDER"):
                # 2026-10-31 train-fast (PLAN_LADDER): fewer trajopt seeds per attempt (cuRobo default 4); a plan costs ~37 s on a TITAN RTX
                planner_kw["num_trajopt_seeds"] = int(fast_value("TRAJOPT_SEEDS", 2))
            backend = R1ProCuRobo(robot, prebuild=prebuild, **planner_kw)
            if shared is not None:
                shared["backend"] = backend
            if GC_FREEZE:
                gc.freeze()
        if os.environ.get("ISPATIALGYM_EXECUTOR_MESH_CACHE"):
            meta["executor_mesh_cache"] = int(backend.mesh_cache_size)
        bounds = task_bounds(env.scene, loader.rooms) if loader.rooms else None
        backend.obstacle_filter = make_obstacle_filter(bounds, robot, [o.name for o in tracked.values()])
        if loader.pool_consumers is not None:
            backend.obstacle_filter = loader.pool_consumers.obstacle_filter(backend.obstacle_filter)
        # load the collision world *before* attaching the held objects (their meshes must be known to be disabled)
        backend.update_world()
        if loader.pool_consumers is not None:
            loader.pool_consumers.observe_planner(backend)
        mcfg = MotionsConfig()
        if fast_on("PLAN_LADDER"):
            # 2026-10-31 train-fast (PLAN_LADDER): a shorter ladder in arm_to - attempts per round (4 -> 2), retry rounds to a witness (4 -> 1)
            mcfg.plan_attempts = int(fast_value("PLAN_ATTEMPTS", 2))
            mcfg.witness_plan_retries = int(fast_value("WITNESS_PLAN_RETRIES", 1))
            fast_meta["plan_ladder"] = {"num_trajopt_seeds": getattr(backend, "num_trajopt_seeds", None), "plan_attempts": mcfg.plan_attempts,
                                        "witness_plan_retries": mcfg.witness_plan_retries}
            log.info("train-fast PLAN_LADDER: trajopt seeds %s, plan_attempts %d, witness_plan_retries %d", fast_meta["plan_ladder"]["num_trajopt_seeds"],
                     mcfg.plan_attempts, mcfg.witness_plan_retries)
        # 2026-11-01 train-fast (always on): count and time the collision-world syncs of this instance (meta["fast"]["world_sync"])
        if not getattr(backend, "_isg_sync_timed", False):
            _uw = backend.update_world

            def _timed_update_world(*a, **kw):
                st = backend.__dict__.setdefault("_isg_sync_stats", {"calls": 0, "s": 0.0, "partial_calls": 0})
                t = time.perf_counter()
                try:
                    return _uw(*a, **kw)
                finally:
                    st["calls"] += 1
                    st["partial_calls"] += int(kw.get("only_objects") is not None)
                    st["s"] += time.perf_counter() - t

            backend.update_world = _timed_update_world
            backend._isg_sync_timed = True
        backend.__dict__["_isg_sync_stats"] = {"calls": 0, "s": 0.0, "partial_calls": 0}
        motions = Motions(env, backend, cfg=mcfg)
        for g in loader.grasps:
            backend.attach_object(g["arm"], loader.physical[g["object"]])
            if g.get("finger_q"):
                motions.state.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        timings["planner_build_s"] = time.time() - t0
        meta["collision_world_rooms"] = loader.rooms
        meta["collision_world_meshes"] = len(backend._world_mesh_names)
        meta["prebuilt_embodiments"] = list(prebuild)

        cam_in_zed = C.camera_offset_in_zed(robot, cam_key)
        meta["camera_in_zed"] = {"position": cam_in_zed[0].tolist(), "orientation": cam_in_zed[1].tolist(),
                                 "constant_dev_mm": 1000.0 * float(th.linalg.norm(cam_in_zed[0] - C.CAMERA_IN_ZED[0]))}
        state = SimState(env, task, robot, cam_key, cam_in_zed=cam_in_zed)
        monitors = Monitors(env, robot, qa_cfg, cam_key=cam_key, cam_in_zed=cam_in_zed)
        rec_cfg = RecorderConfig(hdf5=args.record in ("full", "hdf5"), lerobot=args.record == "full", views=tuple(args.record_view),
                                 video_size=args.video_size, all_cameras=args.all_cameras, frame_buffer=args.frame_buffer,
                                 external_camera=EXTERNAL_CAMERA_NAME if loader.external_camera else None,
                                 behavior_shard=args.behavior_shard, behavior_task=meta.get("variant") or loader.activity)
        recorder = Recorder(out_dir, env, robot, cam_key, tracked, fps=int(loader.cfg["env"]["action_frequency"]), cfg=rec_cfg,
                            task_name=loader.activity.replace("_", " "), instance_id=iid,
                            pool_consumers=loader.pool_consumers)
        if loader.pool_consumers is not None:
            loader.pool_consumers.prepare_recording(build_env_config(
                loader.template_dir, loader.tmeta, loader.tro, loader.tro_path,
                loader.activity, loader.did, loader.robot_name, loader.image_size,
                behavior_output=loader.behavior_output, scene_crop=loader.scene_crop, behavior_proprio=loader.behavior_proprio))
        if loader.scene_crop and hasattr(recorder, "scene_crop"):
            # 2026-11-01 train-fast (PARTIAL_SCENE): the hdf5 says its recorded scene file is the cropped world and where the full one is
            recorder.scene_crop = dict(loader.scene_crop)
            recorder.full_scene_file = os.path.abspath(os.path.join(loader.template_dir, "template.json"))
            recorder.full_load_room_instances = loader.tmeta.get("load_room_instances")
        if image_size != int(args.image_size) and hasattr(recorder, "nominal_image_size"):
            # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): the hdf5 header's env config keeps the evaluation camera size the run would
            # have had (a playback builds its env from it), not the shrunk live cameras
            recorder.nominal_image_size = int(args.image_size)
        if image_size != int(args.image_size):
            # 2026-10-31 train-fast (PHYSICS_ONLY_TRIM): the hdf5 header's config keeps the evaluation camera size (recorder._snapshot_scene)
            recorder.nominal_image_size = int(args.image_size)
            fast_meta.setdefault("physics_only_trim", {"skipped": []})["hdf5_config_image_size"] = int(args.image_size)
        if sidecar is not None:
            recorder.episode = episode_annotations(sidecar)
        online_deadline = getattr(args, "online_member_deadline", None)
        if online_deadline is not None:
            remaining = max(0.001, online_deadline - time.time())
            budget["wallclock_s"] = min(budget["wallclock_s"], remaining) if budget["wallclock_s"] else remaining
        ex_cfg = ExecutorConfig(max_steps=budget["episode_cap"], max_wall_s=budget["wallclock_s"], retries=args.retries, settle_steps=budget["settle_steps"],
                                success_metric=args.success_metric, stop_on_success=args.mode == "policy", goal_image=args.goal_image,
                                record_state=rec_cfg.hdf5, qa=args.qa, qa_stream_fov=args.mode == "policy")
        from ispatialgym.oracle import execute as _execute

        _execute.CONFIG.carry_leg = args.carry_leg
        meta["carry_leg"] = args.carry_leg
        if os.environ.get("ISG_PUSH_BACKOFF_M") is not None:  # rush 2026-09-17: 0 keeps the push pad on the target until its retreat
            _execute.CONFIG.push_backoff_m = float(os.environ["ISG_PUSH_BACKOFF_M"])
            meta["push_backoff_m"] = _execute.CONFIG.push_backoff_m
        if os.environ.get("ISG_AIM_BIAS_RAD") is not None:  # rush: the braced-push ladder's arc aim (overshoot side, rad)
            ex_cfg.brace_retry_aim_bias_rad = float(os.environ["ISG_AIM_BIAS_RAD"])
        if os.environ.get("ISG_AIM_FRAC") is not None:
            ex_cfg.brace_retry_aim_frac = float(os.environ["ISG_AIM_FRAC"])
        meta["rush_env"] = {k: v for k, v in os.environ.items() if k.startswith("ISG_")}
        executor = Executor(env, task, sidecar, recorder=recorder, motions=motions, monitors=monitors, cfg=ex_cfg, sim=og.sim, state=state,
                            tracked_objects=tracked, cam_key=cam_key, task_variant=meta["variant"],
                            pool_consumers=loader.pool_consumers)
        meta["qa_task_variant"] = executor.task_variant
        executor.qa_program = qa_program
        executor.last_obs, executor.last_info = obs, info
        # owner 2026-10-11: the template's objects start on an interior support surface (a shelf inside a bookcase / wardrobe / hall
        # tree, template_meta template_facts.support_surface): the support stays an obstacle but for the board under the object
        executor.support_surface = _execute.interior_support_surface(loader.tmeta)
        if executor.support_surface is not None:
            meta["interior_support_surface"] = {k: executor.support_surface.get(k) for k in ("support", "z_world", "z_above_floor", "open_edges", "why")}
            log.info("interior support surface: %s at z %.3f - the oracle keeps it as an obstacle except its board pieces under the object",
                     executor.support_surface["support"], float(executor.support_surface["z_world"]))
        # oracle.protection (owner 2026-09-26): the goal objects the task placed are protected during transits - plan-time
        # clearance check with a via-above re-plan, run-time contact / movement verdicts (ISG_PROTECT=off disables both)
        if os.environ.get("ISG_PROTECT", "on").lower() not in ("off", "0", "false"):
            from ispatialgym.oracle import protection as _PROT

            try:
                _goal = list(getattr(getattr(task, "problem", None), "goal", None) or [])
                executor.protected_goal = _PROT.goal_protected(_goal, loader.physical, _PROT.variant_override(meta.get("variant")))
            except Exception as e:  # noqa: BLE001
                log.warning("protected objects unavailable (%r): transits unchecked", e)
                executor.protected_goal = []
        meta["protected_objects"] = {b: getattr(o, "name", b) for b, o in executor.protected_goal}
        log.info("protected during transits (goal objects placed by the task): %s", meta["protected_objects"] or "none")
        # the scene file start() saves must hold every object's state (BEHAVIOR's playback loads the scene from it); in queue
        # mode the previous instance left the per-step filter on
        executor.enable_recording_filter(False)
        recorder.start(obs, info, state0=executor.initial_recording_state())
        if rec_cfg.hdf5:
            executor.enable_recording_filter(True)

        # ---------------------------------------------------------------- run
        t0 = time.time()
        if args.mode == "oracle":
            episode = executor.run_program(actions)
        elif args.mode == "replay":
            n_action, n_settle = replay_n[0], budget["settle_steps"]
            meta["replay"] = replay
            meta["replay_rows"] = {"action": n_action, "settle": n_settle, "demo_settle": replay_n[1]}
            executor.stream_phases = [PHASE_ORACLE] * n_action + [PHASE_SETTLE] * n_settle
            episode = executor.run_stream(replay_rows(replay, args.episode, n_action, n_settle))
        else:
            if not args.policy:
                raise ValueError("--mode policy needs --policy module:callable")
            policy = load_policy(args.policy, args.policy_kwargs)
            bind = getattr(policy, "bind", None)
            if callable(bind):  # a policy driving the robot-level motion layer (ispatialgym.harness) gets the executor
                bind(executor=executor, loader=loader, meta=meta, out_dir=out_dir, tro_path=tro_path)
            episode = executor.run_stream(policy_actions(policy, executor, obs, info, budget["max_steps"]))
        timings["episode_s"] = time.time() - t0
        timings.update({f"executor_{k}": v for k, v in episode.get("timings", {}).items()})
        if args.mode == "policy" and callable(getattr(policy, "finalize", None)):
            policy.finalize(episode)
        if episode.get("forced_failure"):
            episode.update({"success": False, "reason": episode.get("reason") or "forced_failure"})
        meta["final_target_poses"] = final_target_poses(task, loader.tro.meta)
        from ispatialgym.eval.round_symmetry import scoring_provenance
        meta["scoring_revision"] = scoring_provenance()
        # i7-measure (owner A4): the targets whose rotation about their round axis the success measure freed (ISPATIALGYM_SUCCESS_YAW_FREE_SYMMETRIC)
        yf = getattr(loader.tro.meta, "yaw_free_report", None)
        if yf:
            meta["success_yaw_free"] = yf
        joints_end = loader.joint_positions_report()  # the TRO's joint_positions constraints (a door opened at init) as the episode leaves them
        if joints_end:
            meta["joint_positions"] = joints_end
            log.info("joint positions at the end of the episode: %s", {k: {"init": v["init"], "end": v["end"]} for k, v in joints_end.items()})

        # ---------------------------------------------------------------- epilogue writes (goal override, goal image)
        goal_obs = episode.get("goal_observation")
        override = None
        if episode["success"] and not args.no_goal_override:
            try:
                override = executor.goal_override(loader.tro.meta, task.compiled_task)
            except Exception as e:  # noqa: BLE001
                log.warning("goal override not computed: %r", e)
                meta["goal_override_error"] = repr(e)
        if episode["success"] and (override is not None or (goal_obs and goal_obs.get("rgb") is not None)):
            if loader.tmeta.get("_paths"):  # the eval dataset's instance files stay as released
                meta["tro_updates"] = {"changed": [], "skipped": "eval dataset instance"}
            else:
                meta["tro_updates"] = write_tro_updates(tro_path, sidecar_path, sidecar, override, goal_obs if args.goal_image else None, out_dir)
        if goal_obs is not None:
            goal_obs = {k: v for k, v in goal_obs.items() if k != "rgb"}
        meta["goal_observation"] = goal_obs
        meta["goal_override"] = None if override is None else {k: v for k, v in override.items()}
        if recorder.last_rgb is not None:
            save_png(os.path.join(out_dir, "final_rgb.png"), recorder.last_rgb)

        success = bool(episode["success"])
        reason = episode.get("reason")
        meta.update({k: v for k, v in episode.items() if k not in ("goal_observation", "timings")})
        meta["status"] = "success" if success else "oracle_failed"
        meta["qa_reason"] = reason
        if loader.report.get("attachments"):
            from ispatialgym.sampling import predicates as PRED

            # the attach rule's joins in this episode (owner 2026-10-10): trigger, alignment, contact, snap size
            meta["attach_joins"] = list(PRED.JOINS[int(loader.report.get("attach_joins_start") or 0):])  # this episode's only (i7-attachfix: the list is per process)
            meta["attach_join_mode"] = PRED.join_mode()
        exit_code = EXIT_SUCCESS if success else EXIT_ORACLE_FAILED
        result.update({
            "status": meta["status"], "success": success, "reason": reason, "success_metric": args.success_metric, "settle_eval": settle_eval,
            "settle_steps": budget["settle_steps"],
            **{k: episode.get(k) for k in ("symbolic_success", "symbolic_success_step", "symbolic_first_step", "geometric_success", "geometric_success_step",
                                            "geometric_first_step", "evaluation_step", "evaluation_kind", "evaluation_settled", "success_final", "error_metric",
                                            "closest_approach", "closest_goal_approach", "errors", "errors_final_step", "measured", "other_mode")},
            "tolerance_profile": tolerance_profile,
            "n_steps": episode["n_steps"], "n_action_steps": episode["n_action_steps"], "binary_branch": episode.get("binary_branch"),
            "collisions": (episode.get("monitors") or {}).get("collisions", {}).get("n_events"),
            "qa_by_category": (episode.get("monitors") or {}).get("stats", {}).get("by_category"),
            "qa_hard": (episode.get("qa") or {}).get("hard_triggered"), "qa_soft": (episode.get("qa") or {}).get("soft_triggered"),
            "goal_override": None if override is None else override.get("applied"),
            "max_steps": budget["max_steps"], "wallclock_s": budget["wallclock_s"],
            "wall_clock_exceeded": episode.get("wall_clock_exceeded"),
            **({"joint_positions": {k: {"init": v["init"], "end": v["end"]} for k, v in joints_end.items()}} if joints_end else {}),
        })
        return exit_code, result
    except BaseException as e:  # noqa: BLE001  (KeyboardInterrupt / SystemExit included: the verdict must be written)
        meta["status"] = "error"
        meta["error"] = repr(e)
        meta["traceback"] = traceback.format_exc()
        if locals().get("loader") is not None:
            meta["loader"] = loader.report
        result.update({"status": "error", "success": False, "reason": "error", "error": repr(e)})
        traceback.print_exc()
        exit_code = EXIT_ERROR
        return exit_code, result
    finally:
        timings["total_s"] = time.time() - t_start
        meta["execution_started_at_unix"] = t_start
        meta["execution_finished_at_unix"] = time.time()
        meta["timings"] = timings
        meta["wall_time_s"] = timings["total_s"]
        # 2026-10-31 train-fast: the switches / tunables this process consulted (ispatialgym.fastflags) and the straight-arm counters
        _motions = locals().get("motions")
        fast_meta["straight_arm"] = dict(getattr(_motions, "straight_arm_stats", None) or {})
        _ex = locals().get("executor")
        if _ex is not None and getattr(_ex, "step_costs", None):
            fast_meta["step_costs"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _ex.step_costs.items()}
        _bk = getattr(_motions, "backend", None)
        if _bk is not None and getattr(_bk, "_isg_sync_stats", None):
            fast_meta["world_sync"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _bk._isg_sync_stats.items()}
            fast_meta["world_sync"]["n_meshes"] = len(getattr(_bk, "_world_mesh_names", ()) or ())
        meta["fast"] = {"flags": fast_flags_seen(), **fast_meta}
        _ge = sys.modules.get("ispatialgym.eval.group_entity")
        if _ge is not None and _ge.pose_tiebreak_on():  # i7-blockdiag: where the group assignment's pose tie-break acted (absent when off)
            meta["group_assign_pose_tiebreak"] = _ge.pose_tiebreak_report()
        _backend = locals().get("backend")
        if os.environ.get("ISPATIALGYM_EXECUTOR_MESH_CACHE") and _backend is not None:
            # the store cannot grow: meshes held (loaded ones stay, disabled, once their object leaves) vs capacity
            meta["executor_mesh_store"] = {"capacity": int(_backend.mesh_cache_size), "meshes": len(_backend._world_mesh_names),
                                           "free": int(_backend.mesh_cache_size) - len(_backend._world_mesh_names)}
        result["wall_time_s"] = round(timings["total_s"], 1)
        result["exit_code"] = exit_code
        if recorder is not None:
            try:
                qa = None
                if executor is not None and executor.result:
                    qa = {"stats": (executor.result.get("monitors") or {}).get("stats"), "metrics": executor.result.get("qa")}
                paths = recorder.close(meta, qa=qa, success=bool(result.get("success")))
                meta.update({"hdf5": paths.get("hdf5"), "videos": paths.get("views"), "lerobot": paths.get("lerobot"), "annotations": paths.get("annotations"),
                             "video_errors": paths.get("view_errors"), "lerobot_error": paths.get("lerobot_error"),
                             "behavior_lerobot": paths.get("behavior_lerobot")})
                result["behavior_lerobot"] = paths.get("behavior_lerobot")
                result["hdf5_ok"] = bool(paths.get("hdf5")) if rec_cfg.hdf5 else None
            except Exception as e:  # noqa: BLE001
                meta["recorder_error"] = repr(e)
                traceback.print_exc()
        if out_dir is not None:
            try:
                _loader = locals().get("loader")
                if _loader is not None and _loader.tro.meta.raw.get("campaign") and result.get("success") is not True:
                    from ispatialgym.sampling.campaign_runtime import record_execution_failure

                    record_execution_failure(tro_arg, out_dir, result)
                dump_json(os.path.join(out_dir, "meta.json"), meta)
                # policy mode: the goal image the policy was conditioned on, next to init_rgb / final_rgb
                goal_path = (meta.get("policy_client") or {}).get("goal_path")
                if goal_path and os.path.isfile(goal_path):
                    import shutil

                    shutil.copyfile(goal_path, os.path.join(out_dir, "goal_rgb.png"))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                result["meta_error"] = repr(e)
            try:
                dump_json(os.path.join(out_dir, "result.json"), result)  # the RUN_ISG_RESULT record
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        _result_line(result)


# ==========================================================================================
# --queue: many instances of one template in one world
# ==========================================================================================

#: ``run_queue``: the next instance needs a world of its own (another template, other simulator-side records); its claim is
#: released for a fresh process
EXIT_QUEUE_RESTART = 75
_SUMMARY_SUFFIX = ".summary.json"


def queue_ready(inst_root: str) -> List[Tuple[str, str]]:
    """``[(instance id, tro path)]`` under a ``sample_isg --out`` root whose summary exists (the sampler writes it after the TRO
    and the sidecar) and whose template meta is there."""
    out = []
    for s in sorted(glob.glob(os.path.join(inst_root, "*", "*", "instances", "*" + _SUMMARY_SUFFIX))):
        iid = os.path.basename(s)[: -len(_SUMMARY_SUFFIX)]
        tro = os.path.join(os.path.dirname(s), iid + TRO_SUFFIX)
        if os.path.exists(tro) and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(s)), "template_meta.json")):
            out.append((iid, tro))
    return out


def _claim(claims: str, iid: str) -> bool:
    try:
        fd = os.open(os.path.join(claims, iid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
    os.close(fd)
    return True


def _tro_object_names(tro_path: str) -> List[str]:
    """The scene objects an instance's TRO sets (restore rewrites their state): the ``ispatialgym.object_bindings`` scene object names of
    its BDDL scope (task objects, support, floor) plus any top-level entry name already in scene-object form."""
    with open(tro_path) as f:
        raw = json.load(f)
    names = {str(b.get("scene_object_name")) for b in ((raw.get("ispatialgym") or {}).get("object_bindings") or {}).values()
             if isinstance(b, dict) and b.get("scene_object_name")}
    names |= {k for k, v in raw.items() if isinstance(v, dict) and "root_link" in v}
    return sorted(names)


def _scene_leaves(template_dir: str, tro_path: str) -> Dict[str, Any]:
    """The template scene as the instance's world keeps it: ``template.json`` flattened to ``{path: value}`` without the entities the
    instance's restore rewrites (the TRO's bound objects, the robot), the robot poses of the task metadata, the file's own path, the
    ``is_asleep`` flags, the linear / angular velocities (PhysX state: an object still swaying when one copy was saved) and the cached
    ``non_kin/AABB`` bounding boxes (derived from the pose and the geometry; they differ between saves of an unmoved object)."""
    names = set(_tro_object_names(tro_path))
    # 2026-11-01 train-fast (PARTIAL_SCENE): the file the world is actually built from (the cropped dump of a cropped template)
    with open(loaded_scene_path(template_dir)) as f:
        doc = json.load(f)
    for sect in ((((doc.get("state") or {}).get("registry") or {}).get("object_registry") or {}), (((doc.get("objects_info") or {}).get("init_info")) or {})):
        for n in list(sect):
            if n in names or n.startswith("robot"):
                sect.pop(n, None)
    ((doc.get("metadata") or {}).get("task") or {}).pop("robot_poses", None)
    ((doc.get("init_info") or {}).get("args") or {}).pop("scene_file", None)
    out: Dict[str, Any] = {}

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{p}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{p}[{i}]")
        elif not (p.endswith("/is_asleep") or "/lin_vel[" in p or "/ang_vel[" in p or "/non_kin/AABB/" in p):
            # PhysX state and the cached bounding box (a derived value: equal poses, boxes 1.3 cm apart between an eval and a train save);
            # poses and joints are compared
            out[p] = x

    walk(doc, "")
    return out


def template_key(template_dir: str, tmeta: dict, tro_path: Optional[str] = None, robot_name: Optional[str] = None) -> str:
    """2026-10-31 train-fast (COLLECTOR_PREBUILD): the template's identity independent of where its copy lives - template id, the activity's
    BDDL problem and the meta fields the world build reads (scene, load_room_instances, problem, scene_variant). Two copies with one key
    are the same template; whether their saved scenes agree is :func:`template_compatible`'s question."""
    h = hashlib.sha256()
    activity, did = tmeta["problem"].rsplit("-", 1)
    bddl = os.path.join(template_dir, "activity_definitions", activity, f"problem{did}.bddl")
    if os.path.exists(bddl):
        h.update(open(bddl, "rb").read())
    h.update(json.dumps({k: tmeta.get(k) for k in ("template_id", "scene", "load_room_instances", "problem", "scene_variant")}, sort_keys=True,
                        default=str).encode())
    crop = scene_crop_of(tmeta)
    if crop:
        # 2026-11-01 train-fast (PARTIAL_SCENE): a cropped and an uncropped copy (or two crops) are different worlds; an uncropped
        # template hashes exactly as before
        h.update(json.dumps({"scene_crop": {"mode": crop.get("mode"), "radius_m": crop.get("radius_m"), "center_object": crop.get("center_object"),
                                            "dropped": sorted(str(n) for n in (crop.get("dropped") or [])),
                                            "loaded_rooms": crop.get("loaded_rooms")}}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def template_compatible(dir_a: str, tro_a: str, dir_b: str, tro_b: str, tol: Optional[float] = None) -> Tuple[bool, str]:
    """2026-10-31 train-fast (COLLECTOR_PREBUILD): may a world built from template copy ``dir_a`` (via its instance ``tro_a``) run the
    instances of copy ``dir_b``? Yes when their scenes (:func:`_scene_leaves`) have the same entries, equal non-numeric values and numbers
    within ``tol`` (``ISPATIALGYM_FAST_TEMPLATE_TOL_M``, 5 mm / 5e-3: two saves of one template settle untouched objects ~1e-6 apart; a
    scene object pushed 1-2 cm or a scene variant that moved chairs away is a different world). Returns (ok, why)."""
    tol = float(fast_value("TEMPLATE_TOL_M", 0.005)) if tol is None else float(tol)
    A, B = _scene_leaves(dir_a, tro_a), _scene_leaves(dir_b, tro_b)
    if set(A) != set(B):
        extra = sorted(set(A) ^ set(B))
        return False, f"{len(extra)} scene entries only in one copy (e.g. {extra[0][:120]})"
    worst, where = 0.0, ""
    for k, va in A.items():
        vb = B[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
            d = abs(float(va) - float(vb))
            if d > worst:
                worst, where = d, k
        elif va != vb:
            return False, f"{k[:120]}: {str(va)[:40]} vs {str(vb)[:40]}"
    if worst > tol:
        return False, f"{where[:120]} differs by {worst:.4f} (> {tol})"
    return True, f"{len(A)} scene entries agree within {worst:.2g}"


def instance_signature(tro_path: str) -> str:
    """What an instance must share with the previous one to reuse its world: the template, the joint resistance /
    counterbalance / mass records and the attachment filters (simulator edits the next instance only overwrites). Hinge
    prims and assisted grasps are removed / re-established per instance and do not count."""
    _, template_dir, tmeta, _, _ = resolve_instance(tro_path)
    tro = load_tro(tro_path)
    extras = tro.meta.extras or {}
    # 2026-10-31 train-fast (COLLECTOR_PREBUILD): the template by content (template_key), so a world prebuilt from another copy of the template
    # (the counted eval instance) serves this job's instances; off: the template's path, as before
    tkey = template_key(template_dir, tmeta) if fast_on("COLLECTOR_PREBUILD") else os.path.abspath(template_dir)
    records = [{k: v for k, v in rec.items() if k not in ("stock", "stock_friction")}
               for rec in (extras.get("constraints") or []) if rec.get("type", "revolute") != "revolute"]
    attachments = sorted((str(r.get("child")), str(r.get("parent"))) for r in (extras.get("attachment_targets") or []))
    return json.dumps({"template": tkey, "records": records, "attachments": attachments}, sort_keys=True, default=str)


def run_queue(args) -> int:
    """``--queue <inst root>``: replay the instances a sampler writes under the root one after the other in this process's
    world (one scene load and one planner build for all of them, :meth:`InstanceLoader.attach`). Each instance is claimed
    (an ``O_EXCL`` file under ``--claims``) so that several runners can share the queue; ``--results`` gets one JSON line per
    instance. Returns 0 once ``--done-file`` exists and no ready instance is left unclaimed (or at ``--deadline`` /
    ``--max-instances``), :data:`EXIT_QUEUE_RESTART` when the next instance needs a world of its own, and the instance's
    exit code after an error (the world may be inconsistent)."""
    from ispatialgym.oracle.pool_inputs import enabled as pool_enabled, run as run_pool

    if pool_enabled(args):
        return run_pool(sys.modules[__name__], args)
    claims = args.claims or os.path.join(os.path.dirname(os.path.abspath(args.queue.rstrip("/"))), "claims")
    os.makedirs(claims, exist_ok=True)
    shared: Dict[str, Any] = {}
    n = 0
    prebuilt_sig = None
    if getattr(args, "prebuild", None) and fast_on("COLLECTOR_PREBUILD"):
        # 2026-10-31 train-fast (COLLECTOR_PREBUILD): the scene load (430-520 s) while the sampler loads its own world and draws the first
        # instance, instead of after it; the world is then reused as for any later instance of the queue (InstanceLoader.attach)
        t0 = time.time()
        image_size = int(args.image_size)
        if PO.trim() and image_size == DEFAULT_IMAGE_SIZE:
            image_size = int(fast_value("PHYSICS_ONLY_IMAGE_SIZE", 128))
        from ispatialgym.oracle.restoration_policy_inputs import from_arguments

        pre_policy = from_arguments(args, args.prebuild,
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        pre = InstanceLoader(args.prebuild, robot_name=args.robot_name, image_size=image_size,
                             head_aperture=args.head_aperture, restoration_policy=pre_policy)
        pre.external_camera = bool({"external", "tiled"} & set(args.record_view))
        pre.behavior_output = bool(args.behavior_shard)
        pre.build_env()
        pre.share(shared)
        shared["loader"] = pre
        if GC_FREEZE:
            gc.collect()
            gc.freeze()
        prebuilt_sig = instance_signature(args.prebuild)
        shared["signature"] = prebuilt_sig
        shared["prebuilt_from"] = os.path.abspath(args.prebuild)
        log.info("train-fast COLLECTOR_PREBUILD: world built from %s in %.1fs before the first queued instance", args.prebuild, time.time() - t0)
    while True:
        if (args.deadline and time.time() > args.deadline) or (args.max_instances and n >= args.max_instances):
            return EXIT_SUCCESS
        nxt = None
        for iid, tro in queue_ready(args.queue):
            if not os.path.exists(os.path.join(claims, iid)) and _claim(claims, iid):
                nxt = (iid, tro)
                break
        if nxt is None:
            if args.done_file and os.path.exists(args.done_file):
                if all(os.path.exists(os.path.join(claims, i)) for i, _ in queue_ready(args.queue)):
                    return EXIT_SUCCESS
                continue
            time.sleep(args.poll_s)
            continue
        iid, tro = nxt
        sig = instance_signature(tro)
        if shared.get("signature") not in (None, sig):
            os.remove(os.path.join(claims, iid))
            if n == 0 and prebuilt_sig is not None:
                # COLLECTOR_PREBUILD: the prebuilt world does not fit this job's template - tell the job script to restart without --prebuild
                marker = os.path.join(os.path.dirname(os.path.abspath(claims)), "prebuild.mismatch")
                with open(marker, "a") as f:
                    f.write(json.dumps({"instance": iid, "prebuild": shared.get("prebuilt_from"), "instance_signature": sig, "prebuilt_signature": prebuilt_sig,
                                        "pid": os.getpid(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                log.warning("train-fast COLLECTOR_PREBUILD: %s does not share the prebuilt world's signature; %s written", iid, marker)
            log.warning("queue: %s needs a world of its own (another template or other simulator-side records than the %d instance(s) run in this "
                        "world): its claim is released and this process exits with rc %d for a fresh one", iid, n, EXIT_QUEUE_RESTART)
            return EXIT_QUEUE_RESTART
        if prebuilt_sig is not None and sig == prebuilt_sig:
            tdir = os.path.dirname(os.path.dirname(os.path.abspath(tro)))
            if tdir not in (shared.get("compatible_template_dirs") or ()):
                pdir = os.path.dirname(os.path.dirname(shared["prebuilt_from"]))
                ok, why = template_compatible(pdir, shared["prebuilt_from"], tdir, tro)
                if not ok:
                    os.remove(os.path.join(claims, iid))
                    marker = os.path.join(os.path.dirname(os.path.abspath(claims)), "prebuild.mismatch")
                    with open(marker, "a") as f:
                        f.write(json.dumps({"instance": iid, "prebuild": shared.get("prebuilt_from"), "why": why, "pid": os.getpid(),
                                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                    log.warning("train-fast COLLECTOR_PREBUILD: the prebuilt world does not fit %s (%s); %s written, exit %d for a fresh world",
                                iid, why, marker, EXIT_QUEUE_RESTART)
                    return EXIT_QUEUE_RESTART
                shared.setdefault("compatible_template_dirs", set()).add(tdir)
                log.info("train-fast COLLECTOR_PREBUILD: %s runs in the prebuilt world (%s)", iid, why)
        t0 = time.time()
        # A grouped worker's member budget includes sampling and queue wait.
        # Preserve the original physics/program and clamp only wallclock.
        online = (load_tro(tro).meta.raw.get("campaign") or {}).get("runtime", {}).get("online_group")
        previous_wallclock = args.wallclock
        previous_online_deadline = getattr(args, "online_member_deadline", None)
        executed = False
        # Train (ungated demos, handoff 2026-11-14 section 3a): a published member's demo is bounded only by its
        # demo_deadline (publication + per-class safety cap); eval keeps the member deadline.
        online_deadline = float(online.get("demo_deadline", online["member_deadline"])) if online else None
        if online and t0 >= online_deadline:
            code, result = 0, {"status": "expired", "success": False, "reason": "online_member_deadline_before_load"}
        else:
            if online:
                args.online_member_deadline = online_deadline
                remaining = args.online_member_deadline - t0
                args.wallclock = min(previous_wallclock, remaining) if previous_wallclock else remaining
            try:
                code, result = run_instance(args, tro, shared)
                executed = True
            finally:
                args.wallclock = previous_wallclock
                args.online_member_deadline = previous_online_deadline
        if executed:
            shared["signature"] = sig
            n += 1
        if args.results:
            prov = provenance()
            row = {"instance": iid, "rc": code, "status": result.get("status"), "success": bool(result.get("success")), "reason": result.get("reason"),
                   "n_steps": result.get("n_steps"), "wall_s": round(time.time() - t0, 1), "tro": tro, "out": result.get("out"), "reused_world": n > 1,
                   "provisional": bool(n > 1 and not prov["queue_scene_restore"]), "provenance": prov,
                   "hdf5_ok": result.get("hdf5_ok"), "behavior_lerobot": result.get("behavior_lerobot"),
                   "host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "finished_at_unix": time.time()}
            # 2026-10-31 train-fast: k runners append to one results file - one O_APPEND write per line keeps the lines intact
            # (a buffered ``open(..., "a").write`` of several processes can interleave)
            line = (json.dumps(row, default=json_default) + "\n").encode()
            fd = os.open(args.results, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        if code == EXIT_ERROR:
            return code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if og.app is not None:
        try:
            og.shutdown()
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Isaac Sim teardown must not change the exit code
