"""``VecInstanceLoader``: ``run_isg.InstanceLoader``'s world and loader contract for N instances in one vectorized
``og.Environment`` (``num_envs`` cloned scenes of one template, a ``VecISpatialGymTask``).

Per batch (``restore_batch``): every slot's TRO goes into its own clone - the scene back to the template's state when the world
is reused, the instance's blocked footprints on that scene's planner map, ``VecISpatialGymTask.load_tro_instance(env, tro,
env_idx)`` (object states with their root poses in the clone's frame, the robot pose set scene-relative, the joints, the
targets in the clone's frame), the robot's hold (``ActionState.hold_current`` + the TRO's recorded controller targets) - then
one batched settle (``settle_physics_steps`` physics substeps with every loaded env's task entities kept still, two full
steps for the caches), ``scene.update_initial_file()`` per loaded scene, ``env.reset(env_indices=loaded)``, and the robots'
holds re-applied where the TRO recorded targets. This is the single-env ``InstanceLoader.restore`` done once for the batch
(BEHAVIOR's ``_settle_and_finalize``).

Every TRO kind: the §3.14 restores of ``run_isg.InstanceLoader`` run per clone through :class:`SlotLoader` (the single-env
methods bound to that clone's scene, robot and scope): hinge constraints (``anchor_position`` translated into the clone's
frame, the axis unchanged - clones are translated copies -, the parent link from that scene's object), joint-friction /
counterbalance / mass records, the attachment pairs' report (their collision stays on until the task's attach rule joins
them, as in the single-env loader), the assisted grasps on that clone's
robot (``ag_anchor_in_eef`` and the object-in-eef placement are eef-relative), ``restore_after_reset`` per env after the
batched ``env.reset`` (lost hinge prims recreated, grasps re-established, articulation joints re-checked), and the reuse
cleanup per scene (that scene's hinges removed, that robot's grasps released) before the scene goes back to the template's
state. A load failure of one slot (``LoadError``: failed grasp, unbound instance) fails that slot's episode only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

import omnigibson as og
from omnigibson.macros import gm

from ispatialgym.eval.group_entity import GroupEntity
from ispatialgym.eval.tro import TROFile, load_tro
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import sleep_assist as SA
from ispatialgym.oracle.action import ActionState, apply_recorded_targets
from ispatialgym.oracle.run_isg import (DEFAULT_HEAD_APERTURE_MM, DEFAULT_IMAGE_SIZE, InstanceLoader, LoadError, _copy_map, _first_key, build_env_config, gripper_command_for,
                                        resolve_instance, robot_name_from_tro)
from ispatialgym.eval.clone_frames import scene_translation
from ispatialgym.tasks import VecISpatialGymTask  # noqa: F401  (registers the task type)

log = logging.getLogger(__name__)


def template_paths(template_dir: str, tmeta: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """``(template.json, template_meta.json)`` of a template: the sampler's layout (both next to ``instances/``) or the eval
    dataset's (``tmeta["_paths"]`` of :func:`run_isg.resolve_instance`: ``<scene>_task_<subtask>_0_0_template.json`` and its
    ``-meta.json``)."""
    paths = (tmeta or {}).get("_paths") or {}
    if paths.get("scene_file"):
        t = paths["scene_file"]
        return t, t[: -len(".json")] + "-meta.json"
    return os.path.join(template_dir, "template.json"), os.path.join(template_dir, "template_meta.json")


def world_payload(template_dir: str, tmeta: Optional[Dict[str, Any]] = None, ignore_objects: Sequence[str] = ()) -> Dict[str, Any]:
    """What makes two templates the same world for the batch: the scene model, the room filter, the problem's activity, the set of
    scene objects with their categories / models, and their resting poses / joint positions. The robot's own entry (pose, virtual base
    joints, AABB) is excluded: the TRO restore overwrites it; AABBs, sleep flags and residual velocities are settle noise.
    ``ignore_objects``: scene object names whose state is left out (the task objects a TRO re-places)."""
    t_path, m_path = template_paths(template_dir, tmeta)
    with open(t_path) as f:
        t = json.load(f)
    with open(m_path) as f:
        meta = json.load(f)
    skip = set(ignore_objects)
    init = (t.get("objects_info") or {}).get("init_info") or {}
    objs = {k: {"class": v.get("class_name"), "args": {a: v.get("args", {}).get(a) for a in ("category", "model", "name", "scale", "fixed_base")}}
            for k, v in init.items() if not str(k).startswith("robot")}
    reg = ((t.get("state") or {}).get("registry") or {}).get("object_registry") or {}

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("non_kin", "is_asleep") and not str(k).endswith("_vel")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    state = {k: strip(v) for k, v in reg.items() if not str(k).startswith("robot") and k not in skip}
    problem = meta.get("problem") or ""
    return {"scene": meta.get("scene"), "rooms": meta.get("load_room_instances"), "problem": problem.rsplit("-", 1)[0] if "-" in problem else problem,
            "objects": objs, "state": state}


def _leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f"{prefix}/{k}")
    elif isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x):
        yield prefix, x
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def worlds_compatible(a: Dict[str, Any], b: Dict[str, Any], tol: float = 5e-3) -> Tuple[bool, List[str]]:
    """Same scene / rooms / problem / objects, and every numeric state leaf within ``tol`` (m, rad, quaternion components):
    the same template settled on another machine differs by ~1e-7 (the single-env worker reused such worlds)."""
    bad = []
    for k in ("scene", "rooms", "problem"):
        if a.get(k) != b.get(k):
            bad.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    if a.get("objects") != b.get("objects"):
        bad.append("objects differ")
    la, lb = dict(_leaves(a.get("state") or {})), dict(_leaves(b.get("state") or {}))
    if set(la) != set(lb):
        bad.append(f"state keys differ ({len(set(la) ^ set(lb))})")
    for k in sorted(set(la) & set(lb)):
        va, vb = la[k], lb[k]
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb) or any(abs(float(x) - float(y)) > tol for x, y in zip(va, vb)):
                bad.append(f"{k}: {va} vs {vb}")
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > tol:
                bad.append(f"{k}: {va} vs {vb}")
        elif va != vb:
            bad.append(f"{k}: {va} vs {vb}")
        if len(bad) > 10:
            break
    return (not bad), bad


def tro_task_objects(tro_path: str) -> List[str]:
    """The scene object names a TRO's ``object_bindings`` name (the objects its restore re-places)."""
    try:
        with open(tro_path) as f:
            raw = json.load(f)
        meta = raw.get("ispatialgym") or raw.get("metadata") or {}
        out = []
        for v in (meta.get("object_bindings") or {}).values():
            name = v.get("scene_object_name") if isinstance(v, dict) else v
            if name:
                out.append(str(name))
        return out
    except Exception:  # noqa: BLE001
        return []


def world_signature(template_dir: str, tmeta: Optional[Dict[str, Any]] = None) -> str:
    """A label of the world (scene, rooms, problem, object set: the parts compared exactly); the state is compared with a tolerance
    by :func:`worlds_compatible`, so two templates with equal signatures may still be rejected."""
    p = world_payload(template_dir, tmeta)
    return hashlib.sha256(json.dumps({k: p[k] for k in ("scene", "rooms", "problem", "objects")}, sort_keys=True, default=str).encode()).hexdigest()[:16]


#: the carb render settings the tiled-exposure experiment reads (auto-exposure histogram, tone mapping, AA, DLSS, render mode, path tracing)
RENDER_SETTING_KEYS = ["/rtx/rendermode", "/rtx/post/histogram/enabled", "/rtx/post/histogram/minEV", "/rtx/post/histogram/maxEV", "/rtx/post/histogram/tau",
                       "/rtx/post/histogram/whiteScale", "/rtx/post/histogram/useExposureClamping", "/rtx/post/tonemap/op", "/rtx/post/tonemap/filmIso",
                       "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber", "/rtx/post/tonemap/whitepoint", "/rtx/post/tonemap/cm2Factor",
                       "/rtx/post/tonemap/enableSrgbToGamma", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/dlss/enabled", "/rtx/post/dlss/auto",
                       "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/spp", "/rtx/pathtracing/optixDenoiser/enabled", "/rtx/pathtracing/adaptiveSampling/enabled",
                       "/rtx/directLighting/sampledLighting/enabled", "/rtx/indirectDiffuse/enabled", "/rtx/reflections/enabled", "/rtx/ambientOcclusion/enabled",
                       "/rtx/post/scaling/enabled", "/rtx/post/scaling/staticRatio", "/rtx/fractionalCutoutOpacity", "/rtx/post/lensFlares/enabled",
                       "/rtx/post/motionblur/enabled", "/rtx/post/dof/enabled", "/rtx/post/colorcorr/enabled", "/rtx/post/colorgrad/enabled"]


class SceneView:
    """The environment seen from one clone: ``.scene`` is ``env.scenes[idx]``; everything else is the environment's (what
    ``Monitors``, ``SimState``, ``Recorder`` and the §3.14 restores read)."""

    def __init__(self, env, idx: int):
        self._env, self.idx = env, int(idx)

    @property
    def scene(self):
        return self._env.scenes[self.idx]

    def __getattr__(self, name):
        return getattr(self._env, name)


class TaskView:
    """The task seen from one clone: ``object_scope`` is that env's scope (``SimState.object`` resolves names through it)."""

    def __init__(self, task, idx: int):
        self._task, self.idx = task, int(idx)

    @property
    def object_scope(self):
        return self._task.scope_of(self.idx)

    def __getattr__(self, name):
        return getattr(self._task, name)


class SlotLoader(InstanceLoader):
    """``InstanceLoader``'s §3.14 restores for one clone (the base class's methods unchanged, bound to the clone's scene view,
    robot and scope; its ``__init__`` is not run - the world is the ``VecInstanceLoader``'s). The only geometric adaptation:
    a hinge record's world-frame ``anchor_position`` (authored in env 0's frame) is translated into the clone's frame before
    ``create_hinge``; the axis is unchanged (clones are translated copies); ``restore_after_reset`` recreates lost prims
    through the same override."""

    def __init__(self, env_view: SceneView, robot, tro, physical: Dict[str, Any], offset):
        self.env, self.robot, self.tro = env_view, robot, tro
        self.physical = dict(physical)
        self.offset = None if offset is None else th.as_tensor(offset, dtype=th.float32).reshape(3)
        self.report: Dict[str, Any] = {}
        self.hinges: List[Any] = []
        self._hinge_records: List[dict] = []
        self._joint_friction_records: List[dict] = []
        self._counterbalance_records: List[dict] = []
        self._mass_records: List[dict] = []
        self._joint_position_records: List[dict] = []
        self.grasps: List[dict] = []
        self.hold: Optional[ActionState] = None
        self._articulation_q: Dict[str, th.Tensor] = {}

    def _create_hinge(self, rec: dict):
        if self.offset is not None and float(th.norm(self.offset)) > 0:
            rec = dict(rec)
            rec["anchor_position"] = [float(v) for v in (th.as_tensor(rec["anchor_position"], dtype=th.float32) + self.offset)]
        return super()._create_hinge(rec)

    def cleanup(self) -> None:
        """Before the clone goes back to the template's state: its hinge prims removed, its robot's grasps released, the
        attachment pairs it filtered restored and its per-pair records forgotten (``predicates.reset_pair_records``)."""
        from ispatialgym.sampling import predicates as PRED
        from ispatialgym.sampling.constraints import remove_hinge

        for h in list(self.hinges):
            remove_hinge(h)
        self.hinges = []
        for arm, held in list(getattr(self.robot, "_ag_obj_in_hand", {}).items()):
            if held is not None:
                self.robot.release_grasp_immediately(arm)
        self.report["pair_records_reset"] = PRED.reset_pair_records(self.env.scene)


class VecInstanceLoader:
    """N clones of one template; see the module docstring."""

    def __init__(self, num_envs: int, robot_name: Optional[str] = None, image_size: int = DEFAULT_IMAGE_SIZE, head_aperture: float = DEFAULT_HEAD_APERTURE_MM,
                 behavior_output: bool = True, tiled_rendering: bool = False, settle_physics_steps: int = 25, warm_renders: int = 100, enhanced_determinism: Optional[bool] = None,
                 fixed_handle_ag: bool = True, physics_overrides: Optional[Dict[str, Any]] = None, dummy_bodies: int = 0, dummy_bodies_x: float = 63.0,
                 render_overrides: Optional[Dict[str, Any]] = None, scene_margin: Optional[float] = None, drop_depth: bool = True,
                 external_camera_hw: Optional[Tuple[int, int]] = None, layout: str = "grid", load_rooms: Optional[Sequence[str]] = None,
                 log_render_settings: bool = False, seg_instance: bool = False):
        """``tiled_rendering``: one tiled render product for all clones' cameras (off by default: the tiles share one exposure, so
        a clone's frames stay 15-20 % off its own per-camera render). ``warm_renders``: renders after the batch's reset before
        the first observation is taken - the real-time renderer adapts to a camera's new view over ~20-60 frames (a robot moved
        to its instance pose renders ~6 % darker at first), and a single-scene world got that warm-up for free during the build."""
        self.num_envs = int(num_envs)
        self.warm_renders = int(warm_renders)
        # PhysX enhanced determinism (physxScene:enableEnhancedDeterminism): the solver's results no longer depend on how the actors
        # are grouped into islands / threads, i.e. on how many scenes share the simulation. Default: on for a multi-scene world, off
        # for num_envs = 1 (the single-env evaluator runs without it; bit-identity with it is kept).
        self.enhanced_determinism = bool(self.num_envs > 1) if enhanced_determinism is None else bool(enhanced_determinism)
        # the demo collector's assisted-grasp semantics (run_isg --mode oracle, ExecutorConfig.handle_grasp_fixed_ag): a handle grasp -
        # a link behind a movable joint (a lid, a leaf) - attaches with a FixedJoint, where OmniGibson's stock rule picks a
        # SphericalJoint under which the hand turns without turning the link. A stream has no action boundaries to install the
        # oracle's per-closure override, so it holds for the whole episode (rigid bodies keep the stock FixedJoint). Without it the
        # blender demos replay to the lid turning in the hand (item 1b, 2026-10-08); with it all four replay to the demo's success step.
        self.fixed_handle_ag = bool(fixed_handle_ag)
        # item 2 experiments (2026-10-08): physics settings applied while the simulator is stopped after the world build
        # ({"broadphase": "SAP"|"MBP"|"GPU", "solver": "TGS"|"PGS", "gpu_dynamics": bool, "num_threads": int, "enhanced_determinism": bool}) and
        # dummy rigid bodies (static cubes) added to scene 0 at x = dummy_bodies_x: does the base scene's physics depend on the world's composition?
        self.physics_overrides = dict(physics_overrides or {})
        # item 1c experiments: carb render settings applied right after the simulator launch, before the world (e.g. {"/rtx/post/aa/op": 0})
        self.render_overrides = dict(render_overrides or {})
        # the gap between cloned scenes along +x (omnigibson.simulator.m.SCENE_MARGIN, default 10 m; the spacing is this margin plus the
        # scene's AABB width, ~53 m for house_single_floor); None = OmniGibson's default
        self.scene_margin = None if scene_margin is None else float(scene_margin)
        self.dummy_bodies = int(dummy_bodies)
        self.dummy_bodies_x = float(dummy_bodies_x)
        # the robot's depth annotators are dropped from the BEHAVIOR observation layout (nothing of a policy run reads them; they
        # cost a render readback per camera per step); ``external_camera_hw``: BEHAVIOR's JoyLo replay camera over the robot's
        # shoulder on every clone (run_isg --record-view external / tiled) at this (height, width)
        self.drop_depth = bool(drop_depth)
        self.external_camera_hw = None if external_camera_hw is None else (int(external_camera_hw[0]), int(external_camera_hw[1]))
        # clone placement (the vec-eval Behavior-1K overlay, scenes/scene_base.py): the scene measured by its live objects (the
        # sampler parks removed objects 100 m underground and far away, which stretched house_single_floor to ~190 m and put
        # clone 6 of a line past x = 1500 m, where float32 world coordinates resolve 0.1 mm) and, with "grid", the clones on a
        # ceil(sqrt(N))-column grid instead of one line along +x; "line" keeps OmniGibson's line (still with the live box)
        self.layout = str(layout)
        # room loading (campaign rooms, 2026-10-28): None = the template's ``load_room_instances`` (the task rooms + every
        # room sharing a wall / doorway, ``sampling.sim_utils.adjacent_rooms``), or what ``run_isg.rooms_for_template`` put into
        # the config from ISG_ROOMS_OVERRIDE (by template variant); a list = exactly these room instances (an override: the
        # rooms the demos can actually see; the driver resolves --rooms-override / --load-rooms into it). OmniGibson always
        # loads walls / ceilings / roofs / fences / doors whatever the list (``InteractiveTraversableScene._should_load_object``);
        # floors, windows, lights and furniture follow the list. Recorded in ``report["load_rooms"]`` and every slot's meta.
        self.load_rooms = None if load_rooms is None else [str(r) for r in load_rooms]
        # ``seg_instance`` on every robot camera (the room-visibility probe of run_isg_vec --seg-visibility): per-pixel
        # object ids; run with the sampler's crash guard (``/rtx/raytracing/showLights`` = 0, SimWorld.LIGHT_SHAPES_SETTING)
        self.seg_instance = bool(seg_instance)
        # read the exposure / tone-mapping / AA / render-mode carb settings after the launch into report["render_settings"]
        # (the tiled-exposure experiment, 2026-10-27); forces an explicit og.launch before the world like render_overrides
        self.log_render_settings = bool(log_render_settings)
        self._render_disabled: set = set()  # clones whose render products are paused (lever S1-finished)
        self.robot_name_arg = robot_name
        self.image_size = int(image_size)
        self.head_aperture = float(head_aperture)
        self.behavior_output = bool(behavior_output)
        self.tiled_rendering = bool(tiled_rendering)
        self.settle_physics_steps = int(settle_physics_steps)
        self.env = None
        self.task = None
        self.robots: List[Any] = []
        self.cam_key: Optional[str] = None
        self.template_dir: Optional[str] = None
        self.signature: Optional[str] = None
        self.report: Dict[str, Any] = {}
        self.scene_state0: List[Any] = []
        self.floor_map_template: List[Any] = []
        self.tros: List[Optional[TROFile]] = [None] * self.num_envs
        self.tro_paths: List[Optional[str]] = [None] * self.num_envs
        self.slots: List[Optional[SlotLoader]] = [None] * self.num_envs
        self.slot_reports: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.failed: Dict[int, Dict[str, Any]] = {}
        self.scene_loads = 0
        self.batches = 0
        self.task_objects: List[str] = []
        self._compatible: set = set()

    # ------------------------------------------------------------------ world

    def build_env(self, first_tro_path: str):
        """The world: N clones of the template of ``first_tro_path`` (the robots spawned at its TRO base pose, scene-relative)."""
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", self.tiled_rendering)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        iid, template_dir, tmeta, activity, did = resolve_instance(os.path.abspath(first_tro_path))
        tro = load_tro(os.path.abspath(first_tro_path))
        self.template_dir = template_dir
        self.template_file = os.path.abspath(template_paths(template_dir, tmeta)[0])
        self.task_objects = tro_task_objects(os.path.abspath(first_tro_path))
        self.payload = world_payload(template_dir, tmeta, ignore_objects=self.task_objects)
        self.signature = world_signature(template_dir, tmeta)
        self.tmeta, self.activity, self.did = tmeta, activity, did
        self.robot_name = self.robot_name_arg or robot_name_from_tro(tro)
        self.cfg = build_env_config(template_dir, tmeta, tro, os.path.abspath(first_tro_path), activity, did, self.robot_name, self.image_size, behavior_output=self.behavior_output)
        self.cfg["task"]["type"] = "VecISpatialGymTask"
        self.cfg["env"]["num_envs"] = self.num_envs
        template_rooms = tmeta.get("load_room_instances")
        if self.load_rooms is not None:
            self.cfg["scene"]["load_room_instances"] = list(self.load_rooms)
            self.report["load_rooms"] = {"mode": "override", "rooms": list(self.load_rooms), "template_rooms": template_rooms,
                                         "dropped": [r for r in (template_rooms or []) if r not in self.load_rooms],
                                         "added": [r for r in self.load_rooms if r not in (template_rooms or [])]}
        else:
            # build_env_config applied ISG_ROOMS_OVERRIDE (by template variant) when set: report what the world will load
            cfg_rooms = self.cfg["scene"].get("load_room_instances")
            if cfg_rooms != template_rooms:
                self.report["load_rooms"] = {"mode": "env-override", "rooms": cfg_rooms, "template_rooms": template_rooms,
                                             "dropped": [r for r in (template_rooms or []) if r not in (cfg_rooms or [])], "source": os.environ.get("ISG_ROOMS_OVERRIDE")}
            else:
                self.report["load_rooms"] = {"mode": "template", "rooms": template_rooms}
        if self.drop_depth:
            for r in self.cfg["robots"]:
                r["obs_modalities"] = [m for m in r.get("obs_modalities", []) if m != "depth_linear"]
        if self.seg_instance:
            for r in self.cfg["robots"]:
                if "seg_instance" not in r.get("obs_modalities", []):
                    r["obs_modalities"] = list(r.get("obs_modalities", [])) + ["seg_instance"]
        if self.external_camera_hw is not None:
            from ispatialgym.oracle.run_isg import EXTERNAL_CAMERA_NAME

            self.cfg["env"]["external_sensors"] = [{
                "sensor_type": "VisionSensor", "name": EXTERNAL_CAMERA_NAME,
                "relative_prim_path": f"/controllable__r1pro__{self.robot_name}/base_link/{EXTERNAL_CAMERA_NAME}",
                "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": self.external_camera_hw[0], "image_width": self.external_camera_hw[1], "horizontal_aperture": 40.0},
                "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
                "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
                "pose_frame": "parent",
            }]
        t0 = time.time()
        # vec_next (2026-10-30, the small-object trembling): macro-level physics overrides applied before the world is built -
        # sleep_threshold (every entity's physxRigidBody / physxArticulation sleepThreshold, stock 5e-5), solver_position_iterations /
        # solver_velocity_iterations (every object's solver iteration counts, stock 32 / 1)
        if self.physics_overrides:
            import omnigibson.prims.entity_prim as _ep
            import omnigibson.objects.usd_object as _uo
            if "sleep_threshold" in self.physics_overrides:
                with _ep.m.unlocked():
                    _ep.m.DEFAULT_SLEEP_THRESHOLD = float(self.physics_overrides["sleep_threshold"])
                self.report["sleep_threshold"] = float(_ep.m.DEFAULT_SLEEP_THRESHOLD)
            if "solver_position_iterations" in self.physics_overrides:
                with _uo.m.unlocked():
                    _uo.m.DEFAULT_SOLVER_POSITION_ITERATIONS = int(self.physics_overrides["solver_position_iterations"])
                self.report["solver_position_iterations"] = int(_uo.m.DEFAULT_SOLVER_POSITION_ITERATIONS)
            if "solver_velocity_iterations" in self.physics_overrides:
                with _uo.m.unlocked():
                    _uo.m.DEFAULT_SOLVER_VELOCITY_ITERATIONS = int(self.physics_overrides["solver_velocity_iterations"])
                self.report["solver_velocity_iterations"] = int(_uo.m.DEFAULT_SOLVER_VELOCITY_ITERATIONS)
        import math as _math

        os.environ.setdefault("ISG_SCENE_LIVE_AABB", "1")
        # "grid-centered" (2026-10-27): the same grid centred on scene 0 (ISG_SCENE_GRID_CENTER=1 in the overlay) - the clones
        # take the cells around the origin nearest first, so the farthest clone origin is (px, py) instead of (2 px, 2 py)
        cols = int(_math.ceil(_math.sqrt(self.num_envs))) if (self.layout in ("grid", "grid-centered") and self.num_envs > 1) else 0
        os.environ["ISG_SCENE_GRID_COLS"] = str(cols)
        os.environ["ISG_SCENE_GRID_CENTER"] = "1" if self.layout == "grid-centered" else "0"
        self.report["clone_layout"] = {"layout": self.layout, "grid_cols": cols, "grid_center": self.layout == "grid-centered",
                                       "live_aabb": os.environ.get("ISG_SCENE_LIVE_AABB") == "1"}
        import omnigibson.simulator as _S

        if self.scene_margin is not None:
            with _S.m.unlocked():
                _S.m.SCENE_MARGIN = self.scene_margin
        self.report["scene_margin"] = float(_S.m.SCENE_MARGIN)
        if (self.render_overrides or self.log_render_settings) and og.sim is None:
            import omnigibson.lazy as lazy

            env_cfg = self.cfg["env"]
            og.launch(physics_dt=1.0 / env_cfg["physics_frequency"], rendering_dt=1.0 / env_cfg.get("rendering_frequency", 30), sim_step_dt=1.0 / env_cfg["action_frequency"])
            settings = lazy.carb.settings.get_settings()
            if self.log_render_settings:
                self.report["render_settings_before_overrides"] = {k: settings.get(k) for k in RENDER_SETTING_KEYS}
            applied = {}
            for key, value in self.render_overrides.items():
                if isinstance(value, bool):
                    settings.set_bool(key, value)
                elif isinstance(value, int):
                    settings.set_int(key, value)
                elif isinstance(value, float):
                    settings.set_float(key, value)
                else:
                    settings.set_string(key, str(value))
                applied[key] = settings.get(key)
            if self.render_overrides:
                self.report["render_overrides"] = applied
            if self.log_render_settings:
                self.report["render_settings"] = {k: settings.get(k) for k in RENDER_SETTING_KEYS}
        self.env = og.Environment(configs=self.cfg)
        self.scene_loads += 1
        from ispatialgym.sampling.scene_variants import patch_trav_map_from_meta

        self.task = self.env.task
        assert isinstance(self.task, VecISpatialGymTask), type(self.task)
        self.robots = [scene.robots[0] for scene in self.env.scenes]
        self.report["trav_cells_freed"] = [patch_trav_map_from_meta(scene, tmeta) for scene in self.env.scenes]
        self.floor_map_template = [_copy_map(scene.trav_map.floor_map[0]) for scene in self.env.scenes]
        # the BEHAVIOR-1K evaluator's 250 kg chassis on every clone's robot (one stop / play for all)
        masses = []
        mass_kg = float(os.environ.get("ISG_BASE_MASS_KG", M.EVAL_BASE_LINK_MASS_KG))
        links = [(r.base_footprint_link if hasattr(r, "base_footprint_link") else r.links[M.BASE_LINK]) for r in self.robots]
        need_mass = mass_kg > 0 and any(abs(float(l.mass) - mass_kg) > 1e-6 for l in links)
        if need_mass or self.enhanced_determinism or self.physics_overrides or self.dummy_bodies:
            og.sim.stop()
            if need_mass:
                for l in links:
                    masses.append({"before": float(l.mass)})
                    l.mass = mass_kg
            if self.enhanced_determinism or "enhanced_determinism" in self.physics_overrides:
                self.report["enhanced_determinism"] = self._set_enhanced_determinism(bool(self.physics_overrides.get("enhanced_determinism", self.enhanced_determinism)))
            if self.physics_overrides:
                self.report["physics_overrides"] = self._apply_physics_overrides(self.physics_overrides)
            if self.dummy_bodies:
                self.report["dummy_bodies"] = self._add_dummy_bodies(self.dummy_bodies, self.dummy_bodies_x)
            og.sim.play()
            if hasattr(og.sim, "update_handles"):
                og.sim.update_handles()
            for m, l in zip(masses, links):
                m["after"] = float(l.mass)
        else:
            masses = [{"before": float(l.mass), "after": float(l.mass)} for l in links]
        self.report["base_link_mass_kg"] = masses
        self.cam_key = _first_key(self.robots[0].sensors, M.HEAD_CAMERA_SENSOR_SUFFIX)
        if self.cam_key is None:
            raise RuntimeError(f"robot {self.robots[0].name} has no head camera sensor ({M.HEAD_CAMERA_SENSOR_SUFFIX})")
        if self.head_aperture:
            for r in self.robots:
                r.sensors[self.cam_key].horizontal_aperture = self.head_aperture
        self.report["head_camera_key"] = self.cam_key
        self.report["head_camera_aperture_mm"] = [float(r.sensors[self.cam_key].horizontal_aperture) for r in self.robots]
        if self.behavior_output:
            self._warm_camera_params()
        if self.fixed_handle_ag:
            from ispatialgym.oracle.execute import _patch_ag_joint_type

            for r in self.robots:
                _patch_ag_joint_type(r, "FixedJoint")
        self.report["fixed_handle_ag"] = self.fixed_handle_ag
        self.report["tiled_rendering"] = bool(getattr(self.env, "_tiled_sensor", None) is not None)
        self.report["num_envs"] = self.num_envs
        self.report["scene_offsets"] = [[float(v) for v in scene.pose[:3, 3]] for scene in self.env.scenes]
        self.report["build_s"] = round(time.time() - t0, 1)
        self.scene_state0 = [copy.deepcopy(scene.dump_state(serialized=False)) for scene in self.env.scenes]
        self._fresh = True
        return self.env

    @staticmethod
    def _is_free_rigid(obj) -> bool:
        """:func:`ispatialgym.oracle.sleep_assist.is_free_rigid` (shared with run_isg)."""
        return SA.is_free_rigid(obj)

    def _apply_rigid_sleep_threshold(self, objs) -> Optional[Dict[str, Any]]:
        """vec_next (2026-10-30): the rigid-body physics overrides for a slot's TASK objects, which the TRO restore adds after the world build
        (the build-time pass in ``_apply_physics_overrides`` only reaches the scene's own objects): every free, non-articulated object of
        ``objs`` gets ``rigid_sleep_threshold`` (physxRigidBody / physxArticulation sleepThreshold) and ``rigid_solver_position_iterations`` /
        ``rigid_solver_velocity_iterations`` (the solver iteration counts; the teapot-on-bar chatter that never sleeps at OmniGibson's 32 / 1
        sleeps within 0.5 s at 64 / 8, probe R7 2026-10-30). Returns the report entry (None when no override is on)."""
        return SA.apply_to_objects(objs, self.physics_overrides or {})

    @staticmethod
    def _apply_physics_overrides(ov: Dict[str, Any]) -> Dict[str, Any]:
        """Physics-engine settings for the item-2 experiments (simulator stopped)."""
        import omnigibson.lazy as lazy

        ctx = og.sim._physics_context
        out: Dict[str, Any] = {}
        with og.sim.editing_usd():  # the physics scene attributes are USD edits
            if "broadphase" in ov:
                ctx.set_broadphase_type(str(ov["broadphase"]))
                out["broadphase"] = ctx.get_broadphase_type()
            if "solver" in ov:
                ctx.set_solver_type(str(ov["solver"]))
                out["solver"] = str(ov["solver"])
            if "gpu_dynamics" in ov:
                ctx.enable_gpu_dynamics(bool(ov["gpu_dynamics"]))
                out["gpu_dynamics"] = bool(ov["gpu_dynamics"])
        out.update(SA.apply_to_scenes(og.sim.scenes, ov))  # rigid_sleep_threshold / rigid_solver_* (shared with run_isg)
        if "stabilization" in ov:
            # physxScene:enableStabilization on the physics scene prim + physxRigidBody:stabilizationThreshold on every free object
            thr = float(ov["stabilization"])
            scene_prim = None
            for prim in og.sim.stage.Traverse():
                if prim.GetTypeName() == "PhysicsScene":
                    scene_prim = prim
                    break
            with og.sim.editing_usd():
                if scene_prim is not None:
                    attr = scene_prim.GetAttribute("physxScene:enableStabilization")
                    if not attr or not attr.IsValid():
                        attr = scene_prim.CreateAttribute("physxScene:enableStabilization", lazy.pxr.Sdf.ValueTypeNames.Bool)
                    attr.Set(thr > 0)
            n_set = 0
            for scene in og.sim.scenes:
                for obj in scene.objects:
                    if getattr(obj, "fixed_base", False) or getattr(obj, "kinematic_only", False) or getattr(obj, "category", "") == "agent":
                        continue
                    try:
                        obj.stabilization_threshold = thr
                        n_set += 1
                    except Exception:  # noqa: BLE001
                        pass
            out["stabilization"] = {"enabled": thr > 0, "threshold": thr, "objects": n_set}
        if "num_threads" in ov:
            settings = lazy.carb.settings.get_settings()
            settings.set_int("/persistent/physics/numThreads", int(ov["num_threads"]))
            settings.set_int("/physics/numThreads", int(ov["num_threads"]))
            out["num_threads"] = int(ov["num_threads"])
        return out

    def _add_dummy_bodies(self, n: int, x: float) -> Dict[str, Any]:
        """``n`` static 10 cm cubes in scene 0 at (x + i, 0, 1): actors that never touch anything of the task (item 2)."""
        from omnigibson.objects.primitive_object import PrimitiveObject

        scene = self.env.scenes[0]
        names = []
        for i in range(n):
            obj = PrimitiveObject(name=f"isg_dummy_{i}", primitive_type="Cube", size=0.1, fixed_base=True, visual_only=False)
            scene.add_object(obj)
            obj.set_position_orientation(th.tensor([x + float(i), 0.0, 1.0 + 0.2 * i]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            names.append(obj.name)
        return {"n": n, "x": x, "names": names}

    @staticmethod
    def _set_enhanced_determinism(value: bool) -> Dict[str, Any]:
        """``physxScene:enableEnhancedDeterminism`` on the simulator's physics scene prim (while stopped)."""
        import omnigibson.lazy as lazy

        ctx = og.sim._physics_context
        path = getattr(ctx, "prim_path", None) or "/physicsScene"
        prim = og.sim.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return {"error": f"no physics scene prim at {path}"}
        api_cls = lazy.pxr.PhysxSchema.PhysxSceneAPI
        with og.sim.editing_usd():
            api = api_cls(prim) if prim.HasAPI(api_cls) else api_cls.Apply(prim)
            attr = api.GetEnableEnhancedDeterminismAttr() or api.CreateEnableEnhancedDeterminismAttr()
            attr.Set(bool(value))
        return {"prim": path, "enableEnhancedDeterminism": bool(attr.Get())}

    def _warm_camera_params(self, max_renders: int = 20) -> None:
        from omnigibson.sensors.vision_sensor import VisionSensor

        for r in self.robots:
            for sensor in r.sensors.values():
                if not isinstance(sensor, VisionSensor):
                    continue
                for _ in range(max_renders):
                    if float(th.as_tensor(sensor.camera_parameters["cameraProjection"]).abs().sum()) > 0:
                        break
                    og.sim.render()

    # ------------------------------------------------------------------ loader contract (per batch)

    def check_template(self, tro_path: str) -> str:
        """The instance's template must be this world's: the same template dir, or one whose scene / rooms / activity / object set
        and resting state (the TRO's own task objects excepted) match (:func:`worlds_compatible`)."""
        _, template_dir, tmeta, _, _ = resolve_instance(os.path.abspath(tro_path))
        key = os.path.abspath(template_paths(template_dir, tmeta)[0])
        if key == self.template_file:
            return template_dir
        if key in self._compatible:
            return template_dir
        ok, bad = worlds_compatible(self.payload, world_payload(template_dir, tmeta, ignore_objects=set(self.task_objects) | set(tro_task_objects(os.path.abspath(tro_path)))))
        if not ok:
            raise RuntimeError(f"{tro_path}: template {template_dir} is another world than {self.template_dir}: {bad[:3]}")
        self._compatible.add(key)
        return template_dir

    def restore_batch(self, tro_paths: Sequence[Optional[str]], settle_eval: bool = True, settle_steps: Optional[Sequence[Optional[int]]] = None,
                      mode: str = "replay", tolerance_scale: Optional[float] = None) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """Load ``tro_paths[i]`` into clone ``i`` (None = idle slot, left as it is); returns ``({env_idx: obs}, {env_idx: info})``
        of the loaded envs after the reset. A slot whose §3.14 restore fails (``LoadError``) is recorded in ``self.failed[i]``
        (``{"reason", "message"}``), left holding, and excluded from the returned dicts. Every loaded env gets the run's
        ``settle_eval`` and its window ``settle_steps[i]`` (default the task's) before the reset and its ``isg-v2`` tolerances
        (``mode`` / ``tolerance_scale``, ``slot_reports[i]["tolerance_profile"]``) after it, as ``run_isg`` does."""
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta

        assert len(tro_paths) == self.num_envs, (len(tro_paths), self.num_envs)
        env, task = self.env, self.task
        wanted = [i for i, p in enumerate(tro_paths) if p]
        self.failed = {}
        for i in sorted(self._render_disabled):  # lever S1-finished: every paused clone renders again before the warm renders
            self.set_render_enabled(i, True)
        if not wanted:
            return {}, {}
        t0 = time.time()
        loaded: List[int] = []
        # phase 1 (vec_next 2026-10-31): every clone's TRO restored WITHOUT the physics step Scene.reset ends with (that step advances
        # the whole world: restoring the clones one after another gave the clone restored first N-1 extra unheld steps), then ONE
        # world step - per clone exactly the N=1 sequence (restore, one step, then the per-slot restores of phase 2).
        staged: Dict[int, Tuple[Any, str, Dict[str, Any], Dict[str, Any]]] = {}
        for i in wanted:
            scene = env.scenes[i]
            path = os.path.abspath(tro_paths[i])
            template_dir = self.check_template(path)
            tro = load_tro(path)
            rep: Dict[str, Any] = {"tro_path": path, "template_dir": template_dir}
            extras = tro.meta.extras or {}
            try:
                if not self._fresh:
                    # the previous episode's hinge prims / grasps of this clone away, then the whole clone back to the template's state
                    if self.slots[i] is not None:
                        self.slots[i].cleanup()
                    scene.load_state(copy.deepcopy(self.scene_state0[i]), serialized=False)
                    scene.trav_map.floor_map[0] = _copy_map(self.floor_map_template[i])
                rep["trav_cells_blocked"] = block_trav_map_from_meta(scene, tro.meta.scene or {})
                self._restore_probe(f"before_load_{i}", list(range(self.num_envs)))
                task.load_tro_instance(env, tro=tro, env_idx=i, settle_steps=0, defer_reset_step=True)
                self._restore_probe(f"after_load_{i}", list(range(self.num_envs)))
                self.tros[i], self.tro_paths[i] = tro, path
            except LoadError as e:
                log.error("slot %d (%s): load failed: %s: %s", i, os.path.basename(path), e.reason, e.message)
                self.failed[i] = {"reason": e.reason, "message": e.message}
                rep.update({"load_error": e.message, "load_error_reason": e.reason})
                self.slot_reports[i] = rep
                continue
            except Exception as e:  # noqa: BLE001  (a TRO the task rejects, a missing object: this slot's episode only)
                log.error("slot %d (%s): load failed: %r", i, os.path.basename(path), e)
                self.failed[i] = {"reason": "error", "message": repr(e)}
                rep.update({"load_error": repr(e), "load_error_reason": "error"})
                self.slot_reports[i] = rep
                continue
            staged[i] = (tro, path, rep, extras)
        if staged:
            if any(tro.meta.raw.get("campaign") for tro, _, _, _ in staged.values()):
                from ispatialgym.sampling.campaign_augmentation import require_shared_skybox

                require_shared_skybox([tro.meta.raw.get("campaign") for tro, _, _, _ in staged.values()])
            task.finish_deferred_reset(sorted(staged))
            self._restore_probe("after_reset_step", sorted(staged))
        # phase 2: the per-slot restores (sleep threshold, hinges, attachments, grasps) and the robot's hold
        for i in sorted(staged):
            tro, path, rep, extras = staged[i]
            scene = env.scenes[i]
            try:
                physical = {k: v for k, v in task.physical_scope_of(i).items() if v is not None and not isinstance(v, GroupEntity)}
                if tro.meta.raw.get("campaign"):
                    from ispatialgym.sampling.campaign_runtime import verify_loaded

                    rep["campaign_restoration"] = verify_loaded(tro.meta.raw["campaign"], physical, scene=scene, sim=og.sim)
                rep["rigid_sleep_threshold"] = self._apply_rigid_sleep_threshold(physical.values())
                sl = SlotLoader(SceneView(env, i), self.robots[i], tro, physical, scene_translation(scene))
                self.slots[i] = sl
                rep["constraints"] = sl._restore_constraints(extras.get("constraints") or [])
                rep["attachments"] = sl._attachment_report(extras.get("attachment_targets") or [])
                sl._articulation_q = sl._held_articulations(physical)
                sl.grasps = sl._restore_grasps(extras.get("grasps") or [])
                rep["grasps"] = sl.grasps
            except LoadError as e:
                log.error("slot %d (%s): load failed: %s: %s", i, os.path.basename(path), e.reason, e.message)
                self.failed[i] = {"reason": e.reason, "message": e.message}
                rep.update({"load_error": e.message, "load_error_reason": e.reason})
                self.slot_reports[i] = rep
                continue
            except Exception as e:  # noqa: BLE001
                log.error("slot %d (%s): load failed: %r", i, os.path.basename(path), e)
                self.failed[i] = {"reason": "error", "message": repr(e)}
                rep.update({"load_error": repr(e), "load_error_reason": "error"})
                self.slot_reports[i] = rep
                continue
            robot = self.robots[i]
            hold = ActionState(robot)
            hold.hold_current()
            for g in sl.grasps:
                if g.get("finger_q"):
                    hold.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
            recorded = (tro.meta.robot_state or {}).get("controller_targets") or {}
            rep["controller_targets_restored"] = apply_recorded_targets(hold, robot, recorded) if recorded else []
            robot.apply_action(hold.action())
            sl.hold = hold
            self.slot_reports[i] = rep
            loaded.append(i)
            self._restore_probe(f"after_slot_{i}", list(range(self.num_envs)))
        self._fresh = False
        if not loaded:
            self.batches += 1
            return {}, {}
        # one settle for the batch: physics substeps with every loaded env's task entities held, two full steps for the caches
        self._restore_probe("before_settle", loaded)
        for _ in range(self.settle_physics_steps):
            og.sim.step_physics()
            self._hold_entities(loaded)
        self._restore_probe("after_settle_physics", loaded)
        for _ in range(2):
            og.sim.step()
            self._hold_entities(loaded)
        self._restore_probe("after_settle_steps", loaded)
        for i in loaded:
            env.scenes[i].update_initial_file()
            n = settle_steps[i] if settle_steps is not None and settle_steps[i] is not None else task.settle_steps
            task.configure_evaluation(settle_eval, int(n), env_indices=[i])  # before the reset activates its pose_settled windows
        idx = th.tensor(loaded, dtype=th.long)
        obs_list, info = env.reset(env_indices=idx)  # the loaded scenes back to their snapshots, pose_settled tracking on, a render
        self._restore_probe("after_reset", loaded)
        info_list = info.get("obs_info") or [None] * len(loaded)
        if self.warm_renders > 0:
            # the renderer's adaptation to every clone's new view (see __init__); physics untouched, the observation re-read
            for _ in range(self.warm_renders):
                og.sim.render()
            obs_list, info_list = env.get_obs(env_indices=idx)
        obs = {i: obs_list[k] for k, i in enumerate(loaded)}
        infos = {i: {"obs_info": info_list[k]} for k, i in enumerate(loaded)}
        for k, i in enumerate(loaded):
            sl = self.slots[i]
            try:
                self.slot_reports[i]["after_reset"] = sl.restore_after_reset()
            except LoadError as e:
                log.error("slot %d: restore after reset failed: %s: %s", i, e.reason, e.message)
                self.failed[i] = {"reason": e.reason, "message": e.message}
                self.slot_reports[i].update({"load_error": e.message, "load_error_reason": e.reason})
                obs.pop(i, None)
                infos.pop(i, None)
                continue
            if self.slot_reports[i]["controller_targets_restored"]:
                self.robots[i].apply_action(sl.hold.action())
            self.slot_reports[i]["tolerance_profile"] = task.apply_tolerance_profile(mode, scale=tolerance_scale, env_idx=i, scene=env.scenes[i],
                                                                                     meta=self.tros[i].meta)
            rp, rq = self.robots[i].get_position_orientation()
            t_i = scene_translation(env.scenes[i])
            if t_i is not None:
                rp = rp - t_i
            self.slot_reports[i]["robot_base_init_scene"] = {"position": [float(v) for v in rp], "orientation": [float(v) for v in rq]}
        self._restore_probe("after_restore_after_reset", loaded)
        if os.environ.get("ISG_RESTORE_PROBE_EXIT") == "1":
            os._exit(0)
        self.batches += 1
        self.report["last_restore_s"] = round(time.time() - t0, 1)
        self.report["warm_renders"] = self.warm_renders
        return obs, infos

    def _restore_probe(self, phase: str, slots: Sequence[int]) -> None:
        """Diagnostic (ISG_RESTORE_PROBE=<jsonl path>, off by default): every listed slot's robot base / joints and task objects
        (scene-relative, float64 lists) and the sim clock, appended as one JSON line per phase of the batch restore."""
        path = os.environ.get("ISG_RESTORE_PROBE")
        if not path:
            return
        rec = {"phase": phase, "t": float(og.sim.current_time), "slots": {}}
        for i in slots:
            try:
                tr = scene_translation(self.env.scenes[i])
                tr = th.zeros(3, dtype=th.float64) if tr is None else th.as_tensor(tr, dtype=th.float64)
                d = {}
                robot = self.robots[i]
                p, q = robot.get_position_orientation()
                d["robot_base"] = (th.as_tensor(p, dtype=th.float64) - tr).tolist() + th.as_tensor(q, dtype=th.float64).tolist()
                d["joints"] = th.as_tensor(robot.get_joint_positions(), dtype=th.float64).tolist()
                for name, obj in (self.task.physical_scope_of(i) or {}).items():
                    if obj is None or isinstance(obj, GroupEntity) or not hasattr(obj, "get_position_orientation"):
                        continue
                    p, q = obj.get_position_orientation()
                    d[name] = (th.as_tensor(p, dtype=th.float64) - tr).tolist() + th.as_tensor(q, dtype=th.float64).tolist()
                rec["slots"][str(i)] = d
            except Exception as e:  # noqa: BLE001
                rec["slots"][str(i)] = {"error": repr(e)}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _hold_entities(self, loaded: Sequence[int]) -> None:
        for i in loaded:
            self.slots[i]._hold_entities()

    # ------------------------------------------------------------------ helpers

    def render_products(self, i: int) -> List[Any]:
        """The per-camera render products of clone ``i`` (its robot's VisionSensors and its external camera); none with tiled
        rendering (one shared product per camera name)."""
        if self.env is None or getattr(self.env, "_tiled_sensor", None) is not None:
            return []
        out = []
        for sensor in getattr(self.robots[i], "sensors", {}).values():
            rp = getattr(sensor, "render_product", None)
            if rp is not None:
                out.append(rp)
        for sensors in (getattr(self.env, "_ext_sensor_map", None) or {}).values():
            try:
                rp = getattr(sensors[i], "render_product", None)
            except Exception:  # noqa: BLE001
                rp = None
            if rp is not None:
                out.append(rp)
        return out

    #: how a clone's cameras are paused (lever S1-finished): "hydra-sync" = HydraTexture.set_updates_enabled inside og.sim.editing_usd()
    #: (the 10-27 implementation: ends with a full USD->Fabric synchronisation), "hydra-nosync" = the same call with OmniGibson's USD-edit
    #: guard muted and no Fabric sync (2026-10-28: is the other slots' brightness transient the sync or the switch?), "detach" = the
    #: sensors' annotators detached from their render products instead (the products keep rendering; no AOV readback)
    render_switch_method: str = "hydra-sync"

    def _sensors(self, i: int) -> List[Any]:
        out = list(getattr(self.robots[i], "sensors", {}).values())
        for sensors in (getattr(self.env, "_ext_sensor_map", None) or {}).values():
            try:
                out.append(sensors[i])
            except Exception:  # noqa: BLE001
                pass
        return out

    def set_render_enabled(self, i: int, enabled: bool) -> int:
        """Pause / resume the rendering of clone ``i``'s cameras (``HydraTexture.set_updates_enabled``, lever S1-finished, 2026-10-27):
        a finished slot's cameras need not be rendered until the next batch restores it. Returns the number of products switched.
        ``render_switch_method`` selects the mechanism (module attribute above)."""
        n = 0
        method = self.render_switch_method
        if method == "detach":
            for sensor in self._sensors(i):
                rp = getattr(sensor, "render_product", None)
                anns = getattr(sensor, "_annotators", None) or {}
                if rp is None:
                    continue
                for ann in anns.values():
                    try:
                        if enabled:
                            ann.attach([rp])
                        else:
                            ann.detach([rp])
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
        else:
            rps = self.render_products(i)
            if rps:
                if method == "hydra-nosync":
                    # the guard (a Tf.Notice listener) is muted for the call; no SynchronizeToFabric afterwards
                    guard = getattr(og.sim, "_usd_guard_enabled", None)
                    try:
                        if guard is not None:
                            og.sim._usd_guard_enabled = False
                        for rp in rps:
                            ht = getattr(rp, "hydra_texture", None)
                            if ht is not None and hasattr(ht, "set_updates_enabled"):
                                ht.set_updates_enabled(bool(enabled))
                                n += 1
                    finally:
                        if guard is not None:
                            og.sim._usd_guard_enabled = guard
                            og.sim._deferred_usd_guard_error = None
                else:
                    # the switch edits the render product's /Render prim: inside OmniGibson's USD-edit guard (og.sim.editing_usd), else the
                    # next og.sim.step raises "USD edit detected outside of og.sim.editing_usd() context" (seen 2026-10-27)
                    with og.sim.editing_usd():
                        for rp in rps:
                            ht = getattr(rp, "hydra_texture", None)
                            if ht is not None and hasattr(ht, "set_updates_enabled"):
                                ht.set_updates_enabled(bool(enabled))
                                n += 1
        if enabled:
            self._render_disabled.discard(i)
        elif n:
            self._render_disabled.add(i)
        return n

    def tracked(self, i: int) -> Dict[str, Any]:
        sl = self.slots[i]
        return {} if sl is None else sl.tracked

    def physical_of(self, i: int) -> Dict[str, Any]:
        sl = self.slots[i]
        return {} if sl is None else dict(sl.physical)

    def view(self, i: int) -> SceneView:
        return SceneView(self.env, i)

    def task_view(self, i: int) -> TaskView:
        return TaskView(self.task, i)

    def hold_action(self, i: int) -> th.Tensor:
        """A hold of the current joints for an idle slot's robot (no episode in it)."""
        st = ActionState(self.robots[i])
        st.hold_current()
        return st.action()


__all__ = ["VecInstanceLoader", "SlotLoader", "SceneView", "TaskView", "world_signature", "world_payload", "worlds_compatible"]
