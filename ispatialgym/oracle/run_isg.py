"""``run_isg`` -- v2 driver: load one iSpatialGym instance, run the oracle program (or a replay / a policy) through
the executor, record, evaluate (``oracle_primitives_design.md`` §3.8, §6.5; TODO F1 / F3 / F4). One process = one
instance = one episode::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.oracle.run_isg --instance <iid>-tro_state.json \\
        [--sidecar <iid>.oracle.json] --mode oracle|replay|policy [--out <dir>] [--video] [--goal-image]

What it does

1. **Loader (spec §3.14).** The template scene (``template.json``, partial rooms) with the R1Pro JoyLo
   configuration and the evaluation camera profile ``isg-v1`` (head camera 1080 x 1080, 40 mm aperture,
   ``n_render_iterations = 1``); an :class:`ISpatialGymTask` from the template's ``activity_definitions``; the TRO
   applied with ``load_tro_instance`` and then everything the task loader does not restore yet, re-applied after
   every ``scene.reset()`` (:meth:`InstanceLoader.restore_after_reset`): ``constraints`` (the COM hinges of the
   alignment tasks, re-created as PhysX revolute joints), ``grasps`` (held-object tasks: the object at
   ``object_pose_in_eef_frame``, the recorded finger joints, the assisted grasp attached before step 0 — an attach
   failure fails the load with ``failed_grasp``), ``attachment_targets`` (the child / parent collision filter, at
   load and therefore also when the policy attaches), ``articulation_targets`` (the target's initial joint values held
   during the settle), plus the trav-map patches of the scene variant (``freed_footprints`` of the template,
   ``blocked_footprints`` of the instance) on the planner map. ``pose_targets`` / ``groups`` /
   ``orientation_equivalences`` / ``evaluation_profile`` are the task's (``configure_targets``).
2. **Sidecar** (``--sidecar``, default ``<iid>.oracle.json`` next to the TRO): loaded, validated, ``tro_sha256``
   verified (``--allow-sha-mismatch`` for hand-built sidecars), ``expand(alias, sidecar)`` → the action list.
3. **Executor** (:class:`~ispatialgym.oracle.executor.Executor`): per action checkpoint → execute under the QA
   monitors → postcondition → one retry → annotate; epilogue: binary success → ``max(50, 20 %)`` no-op → settled →
   goal image (``--goal-image``: ≥ 3 renders, ``goal_observation.camera_pose_world``, profile ``isg-v1``) → TRO goal
   override with the settled values when within tolerance (written back into the TRO, the sidecar's ``tro_sha256``
   updated; ``--no-goal-override`` to disable).
4. **Recording** (``--record full|hdf5|video``): ``demo.hdf5`` (BEHAVIOR-replayable) + the LeRobot v3 dataset under
   ``lerobot/`` (``full``), ``head_rgb.mp4`` with ``--video``, ``init_rgb.png`` / ``final_rgb.png`` / ``goal_rgb.png``,
   ``annotations.json`` (the three §6.3 levels, QA verdicts) and ``meta.json``.
5. Prints one line ``RUN_ISG_RESULT {...}`` and exits with ``0`` success, ``2`` oracle / QA failure (``reason`` one
   of ``qa.CATEGORIES``), ``3`` ``no_oracle`` (no sidecar / no program for the alias; the instance is still loaded
   and its init view rendered), ``4`` ``error`` (traceback in ``meta.json``).

``--mode replay`` feeds the actions of a LeRobot episode (``--replay <dataset root> [--episode N]``) or of a
``demo.hdf5`` through the same executor; ``--mode policy`` calls ``--policy module:callable`` as
``policy(obs, info) -> 23-D action`` every step (``--policy-steps``). Both are thin: no program, no checkpoints, the
same monitors, recorder and epilogue.
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import importlib
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
from ispatialgym.oracle.motions import Motions  # noqa: E402
from ispatialgym.oracle.programs import expand  # noqa: E402
from ispatialgym.oracle.qa import Monitors, QAConfig  # noqa: E402
from ispatialgym.oracle.recorder import EpisodeAnnotations, Recorder, RecorderConfig, dump_json, json_default, save_png  # noqa: E402
from ispatialgym.oracle.sidecar import IDDL_ROOT, Sidecar, SidecarError, family_of_subtask, sha256_of_file  # noqa: E402
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


# ==========================================================================================
# instance resolution / env config (from the v1 driver)
# ==========================================================================================


def resolve_instance(path: str) -> Tuple[str, str, dict, str, int]:
    """``(instance id, template dir, template_meta, activity, definition id)`` from a ``*-tro_state.json`` path."""
    path = os.path.abspath(path)
    if not path.endswith(TRO_SUFFIX):
        raise ValueError(f"--instance must point at a *{TRO_SUFFIX} file, got {path}")
    iid = os.path.basename(path)[: -len(TRO_SUFFIX)]
    template_dir = os.path.dirname(os.path.dirname(path))
    meta_path = os.path.join(template_dir, "template_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no template_meta.json next to the instance (expected {meta_path})")
    with open(meta_path) as f:
        meta = json.load(f)
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


EXTERNAL_CAMERA_NAME = "external_sensor0"  # --dev: the JoyLo replay over-the-shoulder viewpoint


def build_env_config(template_dir: str, meta: dict, tro: TROFile, tro_path: str, activity: str, did: int, robot_name: str, image_size: int, proprio: bool = True,
                     behavior_output: bool = False) -> dict:
    """Evaluator configuration (mirrors ``validate_instances.py``) with the robot spawned at the TRO base pose so the
    holonomic base's virtual joints stay near zero. ``behavior_output``: the robot carries BEHAVIOR replay_obs.py's sensors,
    modalities and proprioception (``behavior_lerobot.robot_overrides``) instead of the evaluation camera profile."""
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
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": os.path.join(template_dir, "template.json"),
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        "robots": [M.joylo_robot_config(robot_name, **sensors, **robot_kw)],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": did,
            "activity_definitions_root": os.path.join(template_dir, "activity_definitions"),
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

    def __init__(self, tro_path: str, robot_name: Optional[str] = None, image_size: int = DEFAULT_IMAGE_SIZE, head_aperture: float = DEFAULT_HEAD_APERTURE_MM):
        self.tro_path = os.path.abspath(tro_path)
        self.iid, self.template_dir, self.tmeta, self.activity, self.did = resolve_instance(self.tro_path)
        self.tro = load_tro(self.tro_path)
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

    # -- environment --------------------------------------------------------------------------------------------

    def build_env(self):
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.cfg = build_env_config(self.template_dir, self.tmeta, self.tro, self.tro_path, self.activity, self.did, self.robot_name, self.image_size,
                                    behavior_output=getattr(self, "behavior_output", False))
        if getattr(self, "external_camera", False):
            # --dev (owner 2026-09-17 23:39): the BEHAVIOR / JoyLo replay external viewpoint - a camera on the robot's
            # base link over its shoulder, looking down-forward (render_instances.render_template) - recorded next to
            # the head video as external_rgb.mp4
            self.cfg["env"]["external_sensors"] = [{
                "sensor_type": "VisionSensor",
                "name": EXTERNAL_CAMERA_NAME,
                "relative_prim_path": f"/controllable__r1pro__{self.robot_name}/base_link/{EXTERNAL_CAMERA_NAME}",
                "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": self.image_size, "image_width": self.image_size, "horizontal_aperture": 40.0},
                "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
                "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
                "pose_frame": "parent",
            }]
        self.env = og.Environment(configs=self.cfg)
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta, patch_trav_map_from_meta

        # the scene variant's freed footprints (template) and the instance's blocked footprints on the planner map
        self.report["trav_cells_freed"] = patch_trav_map_from_meta(self.env.scene, self.tmeta)
        self._floor_map_template = _copy_map(self.env.scene.trav_map.floor_map[0])
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.task = self.env.task
        self.robot = self.env.scene.robots[0]
        # the BEHAVIOR-1K evaluator's 250 kg chassis (owner 2026-09-21 23:49): the stock 47.6 kg base tipped 0.8 deg on
        # its wheel spheres under the dishwasher pull and snapped back at the release - the grader's physics, not ours
        self.report["base_link_mass_kg"] = M.apply_eval_base_mass(self.robot)
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
        return self.env

    def share(self, shared: Dict[str, Any]) -> None:
        """``--queue``: hand this loader's world to the next instances of the template."""
        shared.update({"env": self.env, "task": self.task, "robot": self.robot, "cam_key": self.cam_key, "cfg": self.cfg,
                       "template_dir": self.template_dir, "floor_map": self._floor_map_template, "scene_state0": self._scene_state0,
                       "build_report": {k: v for k, v in self.report.items() if k != "trav_cells_blocked"}})

    def attach(self, shared: Dict[str, Any]):
        """``--queue``: this instance in the world of the previous one (same template, :func:`instance_signature`) instead of
        :meth:`build_env` - the previous instance's hinge prims, assisted grasps and planner attachments are removed, the
        whole scene goes back to its state right after :meth:`build_env` and the planner map to the template's before this
        instance's footprints are blocked; :meth:`restore` follows."""
        from ispatialgym.sampling.constraints import remove_hinge
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta

        if os.path.abspath(shared["template_dir"]) != os.path.abspath(self.template_dir):
            raise RuntimeError(f"--queue: {self.iid} belongs to {self.template_dir}, the world to {shared['template_dir']}")
        self.env, self.task, self.robot, self.cam_key, self.cfg = (shared[k] for k in ("env", "task", "robot", "cam_key", "cfg"))
        prev = shared.get("loader")
        for h in list(getattr(prev, "hinges", None) or []):
            remove_hinge(h)
        for arm, held in list(self.robot._ag_obj_in_hand.items()):
            if held is not None:
                self.robot.release_grasp_immediately(arm)
        backend = shared.get("backend")
        for arm in list(getattr(backend, "_attached", {}) or {}):
            backend.detach_object(arm)
        # every scene object (poses, velocities, joints of doors / drawers, object states) back to the template's, so that
        # only the TRO decides where this instance starts - as in a fresh process
        self.env.scene.load_state(copy.deepcopy(shared["scene_state0"]), serialized=False)
        self._scene_state0 = shared["scene_state0"]
        self.env.scene.trav_map.floor_map[0] = _copy_map(shared["floor_map"])
        self._floor_map_template = shared["floor_map"]
        self.report.update(shared["build_report"])
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.report["reused_world"] = True
        return self.env

    # -- the loader contract --------------------------------------------------------------------------------------

    def restore(self, settle_physics_steps: int = 25):
        """TRO → simulator, then the §3.14 restores, a settle with the task entities held, the snapshot, one reset."""
        env, task, robot = self.env, self.task, self.robot
        task.load_tro_instance(env, tro=self.tro, settle_steps=0)
        self.physical = {k: v for k, v in task.object_scope.items() if v is not None and not isinstance(v, GroupEntity)}
        extras = self.tro.meta.extras
        self.report["constraints"] = self._restore_constraints(extras.get("constraints") or [])
        self.report["attachments"] = self._restore_attachment_filters(extras.get("attachment_targets") or [])
        self._articulation_q = {inst: self.physical[inst].get_joint_positions().clone() for inst in self.tro.meta.articulation_targets if inst in self.physical}
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
        for _ in range(settle_physics_steps):
            og.sim.step_physics()
            self._hold_entities()
        for _ in range(2):  # full steps refresh the per-step caches (AABBs, contacts, kinematic 'changed' flags)
            og.sim.step()
            self._hold_entities()
        env.scene.update_initial_file()
        task._reset_tracking()
        obs, info = SE.reset(env)  # returns to the snapshot, activates pose_settled tracking, renders
        self.report["after_reset"] = self.restore_after_reset()
        if self.report["controller_targets_restored"]:
            robot.apply_action(self.hold.action())  # the recorded targets also hold after the reset, until the first action
        return obs, info

    def _hold_entities(self):
        for inst, ent in self.physical.items():
            ent.keep_still()
            q0 = self._articulation_q.get(inst)
            if q0 is not None:
                ent.set_joint_positions(q0)
                ent.set_joint_velocities(th.zeros_like(q0))

    def restore_after_reset(self) -> Dict[str, Any]:
        """Re-apply the restores a ``scene.reset()`` may have undone (idempotent): hinge prims, assisted grasps, the
        attachment collision filters (persistent USD state) and the articulation joint values."""
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

    def _restore_constraints(self, records: List[dict]) -> List[dict]:
        out = []
        self.hinges = []
        self._hinge_records = []  # the hinge records in self.hinges order (restore_after_reset recreates a lost prim from its record)
        self._joint_friction_records = []
        self._counterbalance_records = []
        self._mass_records = []
        for rec in records:
            kind = rec.get("type", "revolute")
            if kind == "mass":
                out.append(self._apply_mass(rec))
                self._mass_records.append(rec)
                log.info("mass applied: %s = %s", rec["object"], out[-1])
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

    def _restore_attachment_filters(self, records: List[dict]) -> List[dict]:
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
            child.states[AttachedTo]._disable_collision_between_child_and_parent(child, parent)
            out.append({"child": rec["child"], "parent": rec["parent"], "child_link": rec.get("child_link"), "parent_link": rec.get("parent_link"), "collision_filtered": True})
            log.info("attachment collision filter: %s / %s", child.name, parent.name)
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
        alias=sidecar.alias,
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


def load_policy(spec: str):
    """``module:callable`` → ``policy(obs, info) -> action``."""
    mod, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(mod), name or "policy")
    return fn() if isinstance(fn, type) else fn


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default=None, help="path to <iid>-tro_state.json of a sampled instance")
    p.add_argument("--queue", default=None, metavar="INST_ROOT",
                   help="instead of --instance: replay every instance a sampler writes under INST_ROOT (sample_isg --out), one after "
                        "the other in this process's world (run_queue)")
    p.add_argument("--claims", default=None, help="--queue: directory of the claim files (default: <INST_ROOT>/../claims)")
    p.add_argument("--done-file", default=None, help="--queue: once this file exists, exit when no ready instance is left unclaimed")
    p.add_argument("--results", default=None, help="--queue: append one JSON line per instance to this file")
    p.add_argument("--deadline", type=float, default=None, help="--queue: epoch seconds after which no instance starts")
    p.add_argument("--max-instances", type=int, default=None, help="--queue: instances per process")
    p.add_argument("--poll-s", type=float, default=10.0, help="--queue: wait between scans for new instances")
    p.add_argument("--sidecar", default=None, help="oracle sidecar (default: <iid>.oracle.json next to the TRO)")
    p.add_argument("--mode", default="oracle", choices=["oracle", "replay", "policy"])
    p.add_argument("--replay", default=None, help="replay mode: LeRobot dataset root or demo.hdf5")
    p.add_argument("--episode", type=int, default=0, help="replay mode: episode index")
    p.add_argument("--policy", default=None, help="policy mode: module:callable -> policy(obs, info) -> action")
    p.add_argument("--policy-steps", type=int, default=3000, help="policy mode: step budget")
    p.add_argument("--out", default=None, help="output root; the run goes to <out>/<iid>/ (default: <template_dir>/runs)")
    p.add_argument("--record", default="full", choices=["full", "hdf5", "video"], help="full = hdf5 + LeRobot; hdf5 = hdf5 only; video = mp4 / meta only")
    p.add_argument("--video", action="store_true", help="record the head camera RGB to <out>/<iid>/head_rgb.mp4")
    p.add_argument("--dev", action="store_true", help="development recording: also the JoyLo over-the-shoulder external viewpoint (external_rgb.mp4, with --video)")
    p.add_argument("--video-size", type=int, default=None, help="downscale video frames to this square size")
    p.add_argument("--goal-image", action="store_true", help="capture the goal image after the settled check (>= 3 renders) and write goal_observation into the TRO")
    p.add_argument("--no-goal-override", action="store_true", help="do not write the settled values back into the TRO goal poses")
    p.add_argument("--demo-gating", default="auto", choices=["auto", "placed_tilt", "placed_orientation", "root_moved", "none"],
                   help="demo-collection gate after the epilogue: placed_orientation = every placed object within its recorded orientation tolerance (full error; "
                        "an object with a continuous symmetry about its long axis per the ICL survey - marker, pencil, bottles - modulo rotation about that axis); "
                        "root_moved = the root of a free-standing articulated object whose joint the program works stays within 5 mm / 0.02 rad at the end and "
                        "10 mm / 0.04 rad of tilt throughout (auto = both on for --mode oracle, off for replay / policy); placed_tilt = the retracted "
                        "principal-axis tilt variant (unused by default); none")
    p.add_argument("--carry-leg", default="straight", choices=["straight", "trajopt"],
                   help="the carry leg lift end -> pre-place: straight Cartesian leg through the recorded carry waypoint (FOV rule (b), default) or the former free trajopt transit (regression reference)")
    p.add_argument("--max-steps", type=int, default=6000, help="episode step budget (30 Hz control steps)")
    p.add_argument("--settle-steps", type=int, default=50, help="no-op window after the program: max(this, 20%% of the action steps)")
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
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)
    if (args.instance is None) == (args.queue is None):
        p.error("exactly one of --instance and --queue")
    if args.all_cameras and M.head_camera_only():
        p.error("--all-cameras records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
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


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    if args.queue:
        return run_queue(args)
    return run_instance(args, args.instance)[0]


def run_instance(args, tro_arg: str, shared: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
    """One instance end to end: load, oracle stack, episode, recording, ``meta.json`` and the ``RUN_ISG_RESULT`` line; returns
    ``(exit code, result)``. ``shared`` (``--queue``): the world and planner of the previous instance of the same template,
    reused (:meth:`InstanceLoader.attach`); the first instance fills it in."""
    t_start = time.time()
    timings: Dict[str, float] = {}
    meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "provenance": provenance(),
                            "reused_world": bool(shared and shared.get("env") is not None)}
    result: Dict[str, Any] = {"status": "error", "mode": args.mode, "provenance": meta["provenance"]}
    exit_code = EXIT_ERROR
    out_dir = None
    recorder: Optional[Recorder] = None
    executor: Optional[Executor] = None
    sidecar: Optional[Sidecar] = None
    sidecar_path: Optional[str] = None

    try:
        loader = InstanceLoader(tro_arg, robot_name=args.robot_name, image_size=args.image_size, head_aperture=args.head_aperture)
        loader.external_camera = bool(args.dev)
        loader.behavior_output = bool(args.behavior_shard)
        meta["dev"] = bool(args.dev)
        iid, tro_path = loader.iid, loader.tro_path
        out_root = args.out or os.path.join(loader.template_dir, "runs")
        out_dir = os.path.join(out_root, iid)
        os.makedirs(out_dir, exist_ok=True)
        meta.update({
            "instance_id": iid, "template_dir": loader.template_dir, "template_id": loader.tmeta.get("template_id"), "problem": loader.tmeta["problem"],
            "activity": loader.activity, "activity_definition_id": loader.did, "alias": loader.tmeta.get("alias") or loader.tro.meta.task_variant,
            "scene": loader.tmeta.get("scene"), "load_room_instances": loader.tmeta.get("load_room_instances"), "tro_path": tro_path,
            "robot_name": loader.robot_name, "out_dir": out_dir,
        })
        result.update({"instance": iid, "activity": loader.activity, "alias": meta["alias"], "out": out_dir})

        # ---------------------------------------------------------------- sidecar + program (before the sim: cheap failures first)
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
                alias = sidecar.alias
                try:
                    actions = expand(alias, sidecar)
                except SidecarError as e:
                    program_error = f"no program for {alias!r}: {e}"
                if actions is not None and os.environ.get("ISG_PROGRAM_HOOK"):
                    # experiments (owner 2026-09-19 night: the door pull-back / push-open study): ``path.py:callable`` or
                    # ``module:callable`` rewriting the expanded program; recorded in meta, never used by the pilot runs
                    actions = _apply_program_hook(os.environ["ISG_PROGRAM_HOOK"], actions, sidecar, loader)
                    meta["program_hook"] = os.environ["ISG_PROGRAM_HOOK"]
                meta["subtask"] = sidecar.subtask
                meta["program"] = [a.summary() for a in actions] if actions else None
        log.info("run_isg v2: %s (%s, mode %s, %s actions) -> %s", iid, loader.activity, args.mode, None if actions is None else len(actions), out_dir)

        # ---------------------------------------------------------------- environment + loader contract
        t0 = time.time()
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
        rp, rq = robot.get_position_orientation()
        meta["robot_base_init"] = {"position": rp.tolist(), "orientation": rq.tolist()}
        try:
            save_png(os.path.join(out_dir, "init_rgb.png"), obs[robot.name][cam_key]["rgb"])
        except Exception as e:  # noqa: BLE001
            log.warning("init render not saved: %r", e)
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
            backend = R1ProCuRobo(robot, prebuild=prebuild, **executor_mesh_cache_kw(robot, room_filter))
            if shared is not None:
                shared["backend"] = backend
            if GC_FREEZE:
                gc.freeze()
        if os.environ.get("ISPATIALGYM_EXECUTOR_MESH_CACHE"):
            meta["executor_mesh_cache"] = int(backend.mesh_cache_size)
        bounds = task_bounds(env.scene, loader.rooms) if loader.rooms else None
        backend.obstacle_filter = make_obstacle_filter(bounds, robot, [o.name for o in tracked.values()])
        # load the collision world *before* attaching the held objects (their meshes must be known to be disabled)
        backend.update_world()
        motions = Motions(env, backend)
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
        monitors = Monitors(env, robot, QAConfig(max_episode_steps=args.max_steps), cam_key=cam_key, cam_in_zed=cam_in_zed)
        rec_cfg = RecorderConfig(hdf5=args.record in ("full", "hdf5"), lerobot=args.record == "full", video=args.video, video_size=args.video_size,
                                 all_cameras=args.all_cameras, frame_buffer=args.frame_buffer,
                                 external_camera=EXTERNAL_CAMERA_NAME if args.dev else None,
                                 behavior_shard=args.behavior_shard, behavior_task=meta.get("alias") or loader.activity)
        recorder = Recorder(out_dir, env, robot, cam_key, tracked, fps=int(loader.cfg["env"]["action_frequency"]), cfg=rec_cfg,
                            task_name=loader.activity.replace("_", " "), instance_id=iid)
        if sidecar is not None:
            recorder.episode = episode_annotations(sidecar)
        if args.demo_gating == "auto":
            # the full placed_orientation gate (owner 2026-09-16 17:10 item 4: placed_tilt retracted; cylinders are compared
            # modulo rotation about their long axis inside the gate, ExecutorConfig.demo_revolution_axes) and the operated
            # articulated roots (owner 2026-09-27: the body of an object whose lid / flap / door the robot works stays put)
            gating = ("placed_orientation", "root_moved") if args.mode == "oracle" else ()
        else:
            gating = () if args.demo_gating == "none" else (args.demo_gating,)
        ex_cfg = ExecutorConfig(max_steps=args.max_steps, retries=args.retries, settle_min_steps=args.settle_steps, goal_image=args.goal_image,
                                record_state=rec_cfg.hdf5, demo_gating=gating)
        meta["demo_gating_enabled"] = list(gating)
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
                            tracked_objects=tracked, cam_key=cam_key)
        executor.last_obs, executor.last_info = obs, info
        # oracle.protection (owner 2026-09-26): the goal objects the task placed are protected during transits - plan-time
        # clearance check with a via-above re-plan, run-time contact / movement verdicts (ISG_PROTECT=off disables both)
        if os.environ.get("ISG_PROTECT", "on").lower() not in ("off", "0", "false"):
            from ispatialgym.oracle import protection as _PROT

            try:
                _goal = list(getattr(getattr(task, "problem", None), "goal", None) or [])
                executor.protected_goal = _PROT.goal_protected(_goal, loader.physical, _PROT.alias_override(meta.get("alias")))
            except Exception as e:  # noqa: BLE001
                log.warning("protected objects unavailable (%r): transits unchecked", e)
                executor.protected_goal = []
        meta["protected_objects"] = {b: getattr(o, "name", b) for b, o in executor.protected_goal}
        log.info("protected during transits (goal objects placed by the task): %s", meta["protected_objects"] or "none")
        # the scene file start() saves must hold every object's state (BEHAVIOR's playback loads the scene from it); in queue
        # mode the previous instance left the per-step filter on
        executor.enable_recording_filter(False)
        recorder.start(obs, info, state0=executor._dump_state(full=True))
        if rec_cfg.hdf5:
            executor.enable_recording_filter(True)

        # ---------------------------------------------------------------- run
        t0 = time.time()
        if args.mode == "oracle":
            episode = executor.run_program(actions)
        elif args.mode == "replay":
            if not args.replay:
                raise ValueError("--mode replay needs --replay <dataset root | demo.hdf5>")
            episode = executor.run_stream(replay_actions(args.replay, args.episode))
        else:
            if not args.policy:
                raise ValueError("--mode policy needs --policy module:callable")
            policy = load_policy(args.policy)
            episode = executor.run_stream(policy_actions(policy, executor, obs, info, args.policy_steps))
        timings["episode_s"] = time.time() - t0
        timings.update({f"executor_{k}": v for k, v in episode.get("timings", {}).items()})

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
            meta["tro_updates"] = write_tro_updates(tro_path, sidecar_path, sidecar, override, goal_obs if args.goal_image else None, out_dir)
        if goal_obs is not None:
            goal_obs = {k: v for k, v in goal_obs.items() if k != "rgb"}
        meta["goal_observation"] = goal_obs
        meta["goal_override"] = None if override is None else {k: v for k, v in override.items()}
        if recorder.last_rgb is not None:
            save_png(os.path.join(out_dir, "final_rgb.png"), recorder.last_rgb)

        success = bool(episode["success"])
        reason = episode.get("reason")
        errors = episode.get("errors") or {}
        last_errors = episode.get("errors_final_step") or {}
        meta.update({k: v for k, v in episode.items() if k not in ("goal_observation", "timings")})
        meta["status"] = "success" if success else "oracle_failed"
        meta["qa_reason"] = reason
        exit_code = EXIT_SUCCESS if success else EXIT_ORACLE_FAILED
        result.update({
            "status": meta["status"], "success": success, "reason": reason, "n_steps": episode["n_steps"], "n_action_steps": episode["n_action_steps"],
            "max_position_error": errors.get("max_position_error"), "max_orientation_error": errors.get("max_orientation_error"),
            "final_step_max_position_error": last_errors.get("max_position_error"), "final_step_max_orientation_error": last_errors.get("max_orientation_error"),
            "settled_success": episode.get("settled_success"), "binary_branch": episode.get("binary_branch"), "collisions": (episode.get("qa") or {}).get("collisions", {}).get("n_events"),
            "qa_by_category": (episode.get("qa") or {}).get("stats", {}).get("by_category"), "goal_override": None if override is None else override.get("applied"),
        })
        return exit_code, result
    except BaseException as e:  # noqa: BLE001  (KeyboardInterrupt / SystemExit included: the verdict must be written)
        meta["status"] = "error"
        meta["error"] = repr(e)
        meta["traceback"] = traceback.format_exc()
        result.update({"status": "error", "success": False, "reason": "error", "error": repr(e)})
        traceback.print_exc()
        exit_code = EXIT_ERROR
        return exit_code, result
    finally:
        timings["total_s"] = time.time() - t_start
        meta["timings"] = timings
        meta["wall_time_s"] = timings["total_s"]
        _backend = locals().get("backend")
        if os.environ.get("ISPATIALGYM_EXECUTOR_MESH_CACHE") and _backend is not None:
            # the store cannot grow: meshes held (loaded ones stay, disabled, once their object leaves) vs capacity
            meta["executor_mesh_store"] = {"capacity": int(_backend.mesh_cache_size), "meshes": len(_backend._world_mesh_names),
                                           "free": int(_backend.mesh_cache_size) - len(_backend._world_mesh_names)}
        result["wall_time_s"] = round(timings["total_s"], 1)
        result["exit_code"] = exit_code
        if recorder is not None:
            try:
                qa = (executor.result.get("qa") or {}).get("stats") if executor is not None and executor.result else None
                paths = recorder.close(meta, qa=qa, success=bool(result.get("success")))
                meta.update({"hdf5": paths.get("hdf5"), "video": paths.get("video"), "lerobot": paths.get("lerobot"), "annotations": paths.get("annotations"),
                             "video_error": paths.get("video_error"), "lerobot_error": paths.get("lerobot_error"),
                             "behavior_lerobot": paths.get("behavior_lerobot")})
                result["behavior_lerobot"] = paths.get("behavior_lerobot")
                result["hdf5_ok"] = bool(paths.get("hdf5")) if rec_cfg.hdf5 else None
            except Exception as e:  # noqa: BLE001
                meta["recorder_error"] = repr(e)
                traceback.print_exc()
        if out_dir is not None:
            try:
                dump_json(os.path.join(out_dir, "meta.json"), meta)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                result["meta_error"] = repr(e)
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


def instance_signature(tro_path: str) -> str:
    """What an instance must share with the previous one to reuse its world: the template, the joint resistance /
    counterbalance / mass records and the attachment filters (simulator edits the next instance only overwrites). Hinge
    prims and assisted grasps are removed / re-established per instance and do not count."""
    _, template_dir, _, _, _ = resolve_instance(tro_path)
    extras = load_tro(tro_path).meta.extras or {}
    records = [{k: v for k, v in rec.items() if k not in ("stock", "stock_friction")}
               for rec in (extras.get("constraints") or []) if rec.get("type", "revolute") != "revolute"]
    attachments = sorted((str(r.get("child")), str(r.get("parent"))) for r in (extras.get("attachment_targets") or []))
    return json.dumps({"template": os.path.abspath(template_dir), "records": records, "attachments": attachments}, sort_keys=True, default=str)


def run_queue(args) -> int:
    """``--queue <inst root>``: replay the instances a sampler writes under the root one after the other in this process's
    world (one scene load and one planner build for all of them, :meth:`InstanceLoader.attach`). Each instance is claimed
    (an ``O_EXCL`` file under ``--claims``) so that several runners can share the queue; ``--results`` gets one JSON line per
    instance. Returns 0 once ``--done-file`` exists and no ready instance is left unclaimed (or at ``--deadline`` /
    ``--max-instances``), :data:`EXIT_QUEUE_RESTART` when the next instance needs a world of its own, and the instance's
    exit code after an error (the world may be inconsistent)."""
    claims = args.claims or os.path.join(os.path.dirname(os.path.abspath(args.queue.rstrip("/"))), "claims")
    os.makedirs(claims, exist_ok=True)
    shared: Dict[str, Any] = {}
    n = 0
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
            log.info("queue: %s needs a world of its own (another template or other simulator-side records)", iid)
            return EXIT_QUEUE_RESTART
        t0 = time.time()
        code, result = run_instance(args, tro, shared)
        shared["signature"] = sig
        n += 1
        if args.results:
            prov = provenance()
            row = {"instance": iid, "rc": code, "status": result.get("status"), "success": bool(result.get("success")), "reason": result.get("reason"),
                   "n_steps": result.get("n_steps"), "wall_s": round(time.time() - t0, 1), "tro": tro, "out": result.get("out"), "reused_world": n > 1,
                   "provisional": bool(n > 1 and not prov["queue_scene_restore"]), "provenance": prov,
                   "hdf5_ok": result.get("hdf5_ok"), "behavior_lerobot": result.get("behavior_lerobot"),
                   "host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(args.results, "a") as f:
                f.write(json.dumps(row, default=json_default) + "\n")
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
