"""Demo recording of the v2 executor (``oracle_primitives_design.md`` §6.5; TODO F4).

One :class:`Recorder` hangs off the executor's ``env.step`` and writes, in **one pass**:

* ``demo.hdf5`` — the BEHAVIOR collection layout ``DataPlaybackWrapper.playback_episode`` replays
  (``/data`` attrs ``config`` / ``scene_file`` / ``omnigibson_git_hash`` / ``n_episodes`` / ``n_steps``; group
  ``demo_0`` with ``num_samples``, ``transitions``, datasets ``action`` (T, 23), ``state`` (T, S) padded serialized
  ``og.sim.dump_state`` — ``state[i]`` is the state *before* ``action[i]`` — ``state_size``, ``reward``,
  ``terminated``, ``truncated``, empty ``init_metadata``), plus iSpatialGym's numeric per-step datasets under
  ``demo_0/isg`` (joint positions / velocities, base / eef / head-camera poses, task-object poses, evaluator flags
  and errors, phase, action index, attempt, proprio) — numeric only, so ``h5py_group_to_torch`` still converts the
  whole group. Rolled-back attempts go to ``demo_0/rollbacks/action_<i>_attempt_<k>`` (BEHAVIOR's
  ``keep_checkpoint_rollback_data`` layout). Strings (annotations, QA verdicts, meta) are JSON **attrs**.
* the LeRobot v3 ingredients — a ``LeRobotDataset`` created with the feature mapping of
  ``OmniGibson/omnigibson/envs/lerobot_data_wrapper.py`` (``observation.images.<sensor>`` videos,
  ``observation.state`` proprio, ``action``, ``next.reward`` / ``next.terminated`` / ``next.truncated``, per-frame
  ``task`` = the fine-grained annotation of the running action, ``annotation.action_index``), frame ``i`` pairing
  the observation *before* ``action[i]`` exactly as the wrapper does; written under ``<out>/lerobot/``.
* ``<view>_rgb.mp4`` — the rollout videos of ``run_isg --record-view`` (:data:`VIEWS`: the head camera, the external
  camera over the robot's shoulder, the two wrist cameras, and BEHAVIOR's JoyLo replay tiling of all four,
  :func:`tiled_frame`),
* ``annotations.json`` — the three annotation levels of §6.3 per action (fine-grained, with numbers, the demo-level
  family / template / instance descriptions) with step ranges and QA verdicts; the same JSON goes into the hdf5 attrs.

Checkpoint semantics (§3.8): rows are buffered per *segment* (the steps since the last checkpoint) and reach the
persistent sinks only on :meth:`commit`; :meth:`rollback` moves the segment to the rollback store instead, so a
retried action leaves no trace in the demo, the video or the LeRobot episode. Buffered rgb frames are JPEG-compressed
(``frame_buffer="jpeg"``) to bound memory; ``"raw"`` keeps uint8 frames.

Simulator-free: ``env`` / ``robot`` / objects are duck-typed; h5py / imageio / lerobot are imported lazily.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.oracle import r1pro_model as M

log = logging.getLogger(__name__)

PHASE_ORACLE, PHASE_SETTLE, PHASE_GOAL = 0, 1, 2
PHASE_CODES = {"oracle": PHASE_ORACLE, "settle": PHASE_SETTLE, "goal": PHASE_GOAL}

#: the views ``run_isg --record-view`` records, each to ``<view>_rgb.mp4``
VIEWS: Tuple[str, ...] = ("tiled", "head", "external", "left_wrist", "right_wrist")
#: BEHAVIOR's JoyLo replay video (``joylo/scripts/replay_data.py``): 480 px tiles
TILE_PX = 480


def _resized(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(frame)).resize((int(width), int(height)), Image.BILINEAR))


def tiled_frame(left_wrist: Optional[np.ndarray], head: Optional[np.ndarray], right_wrist: Optional[np.ndarray], external: Optional[np.ndarray],
                tile: int = TILE_PX) -> np.ndarray:
    """BEHAVIOR's JoyLo replay frame: left wrist | head | right wrist (``tile`` px squares) over the external camera
    (``3 tile`` x ``2 tile``), ``3 tile`` px square in all; a missing view is black."""
    def fit(frame, width, height):
        if frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return _resized(np.asarray(frame)[..., :3], width, height)

    top = np.concatenate([fit(left_wrist, tile, tile), fit(head, tile, tile), fit(right_wrist, tile, tile)], axis=1)
    return np.concatenate([top, fit(external, 3 * tile, 2 * tile)], axis=0)


def json_default(o):
    if isinstance(o, th.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def dump_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=json_default)


def to_uint8_rgb(img) -> np.ndarray:
    arr = img.detach().cpu().numpy() if isinstance(img, th.Tensor) else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def save_png(path: str, img) -> None:
    from PIL import Image

    Image.fromarray(to_uint8_rgb(img)).save(path)


def pose7(pos, quat) -> np.ndarray:
    return np.concatenate([np.asarray(th.as_tensor(pos).detach().cpu(), dtype=np.float32), np.asarray(th.as_tensor(quat).detach().cpu(), dtype=np.float32)])


#: ISPATIALGYM_RECORD_LINK_POSES=1 (opt-in, adds a dataset): ``isg/link_poses`` (T, L, 7) - every step's world pose (pos
#: xyz, quat xyzw) of each link of the scene's movable objects (the robot, articulated or free objects; fixed-base
#: objects without joints never move), read from the physics state like the other poses here, so it works without
#: rendering; the link prim paths in the data group's ``isg_link_pose_paths`` attribute. What a Fabric-direct replayer
#: writes per tick instead of loading the recorded state.
RECORD_LINK_POSES = os.environ.get("ISPATIALGYM_RECORD_LINK_POSES") == "1"


def sensor_name(key: str) -> str:
    """LeRobot observation name of a sensor key, as ``LeRobotDataWrapper.get_lerobot_obs_mapping``:
    ``robot_r1:zed_link:Camera:0`` -> ``zed_link_camera_0``."""
    parts = key.split(":")
    if len(parts) == 4:
        parts = parts[1:]
    return "_".join(parts).lower()


# ==========================================================================================
# annotations (§6.3)
# ==========================================================================================


@dataclass
class ActionAnnotation:
    index: int
    name: str
    arms: List[str]
    objects: List[str]
    fine: str
    numbers: str
    step_start: int
    step_end: int
    attempts: int = 1
    verdict: Optional[Dict[str, Any]] = None
    rolled_back: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "name": self.name, "arms": list(self.arms), "objects": list(self.objects),
            "fine": self.fine, "numbers": self.numbers, "step_start": self.step_start, "step_end": self.step_end,
            "attempts": self.attempts, "verdict": self.verdict, "rolled_back": list(self.rolled_back),
        }


@dataclass
class EpisodeAnnotations:
    """Demo-level texts of §6.3: (a) family name, (b) family-level description, (c) template-level description with
    the involved categories, (d) instance-level description with the goal numbers."""

    family: str = ""
    family_description: str = ""
    template_description: str = ""
    instance_description: str = ""
    variant: str = ""
    instance_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family, "family_description": self.family_description, "template_description": self.template_description,
            "instance_description": self.instance_description, "variant": self.variant, "instance_id": self.instance_id,
        }


# ==========================================================================================
# frame buffer
# ==========================================================================================


class _Frame:
    """One buffered rgb frame (JPEG bytes or a raw uint8 array)."""

    __slots__ = ("data", "shape")

    def __init__(self, arr: np.ndarray, mode: str):
        self.shape = arr.shape
        if mode == "jpeg":
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=92)
            self.data = buf.getvalue()
        else:
            self.data = np.ascontiguousarray(arr)

    def array(self) -> np.ndarray:
        if isinstance(self.data, bytes):
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(self.data)).convert("RGB"))
        return self.data


class _StepFrames(dict):
    """``RecorderConfig.async_frames``: one pending frame per camera key of a step's robot observation, made on first use."""

    def __init__(self, pipe, robs: Dict[str, Any]):
        super().__init__()
        self._pipe, self._robs = pipe, robs

    def get(self, key, default=None):
        if key not in self:
            try:
                self[key] = self._pipe.frame(self._robs[key]["rgb"])
            except Exception:  # noqa: BLE001  (as the synchronous path: a camera without an rgb image has no frame)
                self[key] = None
        v = dict.get(self, key)
        return default if v is None else v


# ==========================================================================================
# recorder
# ==========================================================================================


@dataclass
class RecorderConfig:
    hdf5: bool = True
    lerobot: bool = True
    #: the videos to write (:data:`VIEWS`), each to ``<view>_rgb.mp4``
    views: Tuple[str, ...] = ()
    #: every video scaled to this width (aspect kept)
    video_size: Optional[int] = None
    all_cameras: bool = False  # LeRobot: every rgb sensor, else the head camera only
    keep_rollbacks: bool = True
    frame_buffer: str = "jpeg"  # "jpeg" | "raw"
    video_crf: int = 20
    lerobot_repo_id: Optional[str] = None
    lerobot_robot_type: str = "r1pro"
    lerobot_vcodec: str = "hevc"
    #: name of the env external sensor (``obs["external"][name]["rgb"]``) of the ``external`` / ``tiled`` views - the JoyLo
    #: replay viewpoint over the robot's shoulder
    external_camera: Optional[str] = None
    #: ``<root>/<repo_id>`` of the LeRobot v3 shard the episode is appended to in BEHAVIOR's replay_obs format
    #: (:mod:`ispatialgym.oracle.behavior_lerobot`; the robot must carry replay_obs's sensors); None: not written
    behavior_shard: Optional[str] = None
    #: task name of the shard's episodes (the variant)
    behavior_task: Optional[str] = None
    #: the shard's segments handed to BEHAVIOR's LeRobot writer from a background thread (same frames, same order), so the
    #: demo does not wait on the video encoders' queues. Default: ISPATIALGYM_ASYNC_SHARD=1 (off when unset)
    async_shard: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_ASYNC_SHARD", "").lower() in ("1", "true", "yes"))
    #: frame conversion / encode off the stepping thread, once per step for all sinks, and the committed frames written
    #: to the video / LeRobot sinks from a background thread (:mod:`ispatialgym.oracle.frame_pipeline`; same bytes, same
    #: order). Default: ISPATIALGYM_ASYNC_FRAMES=1 (off when unset)
    async_frames: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_ASYNC_FRAMES", "").lower() in ("1", "true", "yes"))
    #: physics-only collection (:mod:`ispatialgym.oracle.physics_only`): demo.hdf5 only, no frames kept (the images would
    #: be stale). Default: ISPATIALGYM_PHYSICS_ONLY=1 (off when unset)
    physics_only: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_PHYSICS_ONLY", "").lower() in ("1", "true", "yes"))
    #: ``isg/behavior_proprio``: BEHAVIOR's proprioception of the state before each action (row i pairs with state[i] and
    #: action[i]), for a replay that writes collection-time ``observation.state``. Default: physics-only collection or
    #: ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO=1
    behavior_proprio: bool = field(default_factory=lambda: any(os.environ.get(k, "").lower() in ("1", "true", "yes")
                                                               for k in ("ISPATIALGYM_PHYSICS_ONLY", "ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO")))
    #: keep the head camera's last frame as a device copy and convert it when read (``last_rgb``) instead of converting
    #: every step, when no video sink takes the frames. Default: ISPATIALGYM_LAZY_FRAMES=1 (off when unset)
    lazy_frames: bool = field(default_factory=lambda: os.environ.get("ISPATIALGYM_LAZY_FRAMES", "").lower() in ("1", "true", "yes"))


class _LazyRGB:
    """``RecorderConfig.lazy_frames``: a frame kept as a copy of the camera buffer (on its device), converted to uint8 rgb
    the first time it is read."""

    __slots__ = ("_img", "_rgb")

    def __init__(self, img):
        self._img = img.detach().clone() if isinstance(img, th.Tensor) else np.array(img, copy=True)
        self._rgb = None

    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = to_uint8_rgb(self._img)
            self._img = None
        return self._rgb


def scene_crop_attrs(scene_crop: Optional[Dict[str, Any]], full_scene_file: Optional[str], full_load_room_instances=None) -> Dict[str, str]:
    """2026-11-01 train-fast (PARTIAL_SCENE): the ``/data`` attrs of a demo collected in a cropped world (JSON strings) - ``isg_scene_crop``
    (the template meta's ``scene_crop``), ``isg_full_scene_file`` (absolute path of the full ``template.json``), ``isg_full_scene_sha256``
    (its content, when readable, so a replay elsewhere can check its copy) and ``isg_full_load_room_instances`` (the evaluation rooms).
    The recorded ``scene_file`` stays the dump of the cropped world and the per-step states keep their format (self-describing per
    object): the replayer (``multienv.replay --scene full``) merges them into the full template. Empty without a crop (mode ``full``)."""
    if not isinstance(scene_crop, dict) or (scene_crop.get("mode") or "full") == "full":
        return {}
    out = {"isg_scene_crop": json.dumps(scene_crop, default=json_default),
           "isg_full_scene_file": json.dumps(os.path.abspath(full_scene_file) if full_scene_file else None),
           "isg_full_load_room_instances": json.dumps(list(full_load_room_instances) if full_load_room_instances else None)}
    if full_scene_file and os.path.exists(full_scene_file):
        import hashlib

        h = hashlib.sha256()
        with open(full_scene_file, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out["isg_full_scene_sha256"] = json.dumps(h.hexdigest())
    return out


class Recorder:
    def __init__(self, out_dir: str, env, robot, cam_key: str, objects: Dict[str, Any], fps: int = 30, cfg: Optional[RecorderConfig] = None,
                 task_name: Optional[str] = None, instance_id: Optional[str] = None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.env, self.robot, self.cam_key = env, robot, cam_key
        self.objects = dict(objects)
        self.fps = int(fps)
        self.cfg = cfg or RecorderConfig()
        self.task_name = task_name or "ispatialgym"
        self.instance_id = instance_id or "episode"
        # committed rows
        self.rows: Dict[str, list] = defaultdict(list)
        self.states: List[th.Tensor] = []  # state[0] = reset state, then one per committed step
        self.frames: Dict[str, List[_Frame]] = defaultdict(list)  # committed frames not yet flushed to the sinks
        self.n = 0  # committed steps
        # pending segment
        self._pending_rows: Dict[str, list] = defaultdict(list)
        self._pending_states: List[th.Tensor] = []
        self._pending_frames: Dict[str, List[_Frame]] = defaultdict(list)
        self._pending_lerobot: List[Dict[str, Any]] = []
        self._pending_task: List[str] = []
        self.rollbacks: List[Dict[str, Any]] = []
        # annotations
        self.actions: List[ActionAnnotation] = []
        self.episode = EpisodeAnnotations()
        self.current_task = self.task_name
        # sinks: one video writer per view
        unknown = [v for v in self.cfg.views if v not in VIEWS]
        if unknown:
            raise ValueError(f"unknown record views {unknown}; choose from {', '.join(VIEWS)}")
        self.view_paths: Dict[str, str] = {v: os.path.join(out_dir, f"{v}_rgb.mp4") for v in dict.fromkeys(self.cfg.views)}
        self.view_errors: Dict[str, str] = {}
        self._writers: Dict[str, Any] = {}
        self._wrist_keys: Dict[str, Optional[str]] = {}
        self.last_rgb: Optional[np.ndarray] = None
        self.reset_obs = None
        self._last_obs_lerobot: Optional[Dict[str, Any]] = None
        self.dataset = None
        self.lerobot_error: Optional[str] = None
        self.lerobot_root = os.path.join(out_dir, "lerobot")
        self._obs_mapping: Dict[str, str] = {}
        self._video_keys: List[str] = []
        self.scene_file = None
        self.env_config = None
        # 2026-10-31 train-fast PHYSICS_ONLY_TRIM (set by run_isg): the run's cameras were shrunk; the hdf5 header's config keeps this size
        self.nominal_image_size: Optional[int] = None
        # 2026-11-01 train-fast PARTIAL_SCENE (set by run_isg): the world was the sampler's cropped one - the template meta's scene_crop,
        # the full template.json and its rooms go into the hdf5 header (scene_crop_attrs) for the replayer's --scene full
        self.scene_crop: Optional[Dict[str, Any]] = None
        self.full_scene_file: Optional[str] = None
        self.full_load_room_instances: Optional[List[str]] = None
        self.hdf5_path = os.path.join(out_dir, "demo.hdf5") if self.cfg.hdf5 else None
        self.behavior = None  # behavior_lerobot.ShardWriter of the episode
        self.behavior_status: Optional[str] = None
        if self.cfg.physics_only:
            from ispatialgym.oracle import physics_only as PO

            PO.check_recording(bool(self.cfg.views), self.cfg.lerobot, self.cfg.behavior_shard)
        # behavior_proprio: the proprioception of the state the next recorded step starts from, the one of the last
        # committed state (a rollback to an empty pending segment returns to it) and the one after each pending step
        self._bproprio_last: Optional[np.ndarray] = None
        self._bproprio_committed: Optional[np.ndarray] = None
        self._pending_bproprio: List[np.ndarray] = []
        self._pipe = None
        if self.cfg.async_frames:
            from ispatialgym.oracle.frame_pipeline import FramePipeline

            self._pipe = FramePipeline(self.cfg.frame_buffer)

    @property
    def last_rgb(self) -> Optional[np.ndarray]:
        v = self._last_rgb
        rgb = getattr(v, "rgb", None)
        return rgb() if callable(rgb) else v

    @last_rgb.setter
    def last_rgb(self, v) -> None:
        self._last_rgb = v

    def _step_frames(self, robs: Dict[str, Any]) -> Optional["_StepFrames"]:
        """``async_frames``: the step's frames, one per camera key, shared by every sink; None otherwise, and when no
        video or LeRobot sink takes frames (then only ``last_rgb`` is kept, as on the synchronous path)."""
        if self._pipe is None or not (self.cfg.views or self.cfg.lerobot):
            return None
        return _StepFrames(self._pipe, robs)

    # ------------------------------------------------------------------ start

    def _pose_links(self) -> list:
        links = getattr(self, "_pose_links_cache", None)
        if links is None:
            objs = [o for o in self.robot.scene.objects if not (getattr(o, "fixed_base", False) and not getattr(o, "joints", None))]
            links = sorted((link for o in objs for link in o.links.values()), key=lambda link: link.prim_path)
            self._pose_links_cache = links
        return links

    def _robot_obs(self, obs) -> Dict[str, Any]:
        if obs is None:
            return {}
        return obs.get(self.robot.name, obs) if isinstance(obs, dict) else {}

    def head_frame(self, obs) -> Optional[np.ndarray]:
        try:
            return to_uint8_rgb(self._robot_obs(obs)[self.cam_key]["rgb"])
        except Exception:  # noqa: BLE001
            return None

    def _keep_head_frame(self, obs) -> None:
        """``last_rgb`` from the synchronous path: nothing in physics-only collection (no render: the image is stale), a
        lazily converted copy with ``lazy_frames`` when no video sink needs the frame now, else the converted frame."""
        if self.cfg.physics_only:
            return
        if self.cfg.lazy_frames and not self._writers:
            try:
                self.last_rgb = _LazyRGB(self._robot_obs(obs)[self.cam_key]["rgb"])
            except Exception:  # noqa: BLE001  (as head_frame: a camera without an rgb image has no frame)
                pass
            return
        frame = self.head_frame(obs)
        if frame is not None:
            self.last_rgb = frame

    def _behavior_proprio_now(self) -> np.ndarray:
        from ispatialgym.oracle import physics_only as PO

        return PO.behavior_proprio(self.robot)

    def external_frame(self, obs) -> Optional[np.ndarray]:
        """The rgb of the configured external sensor (``obs["external"][name]``), None without one."""
        if not self.cfg.external_camera or not isinstance(obs, dict):
            return None
        try:
            return to_uint8_rgb(obs["external"][self.cfg.external_camera]["rgb"])
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _camera_frame(robs: Dict[str, Any], key: Optional[str]) -> Optional[np.ndarray]:
        if key is None:
            return None
        try:
            return to_uint8_rgb(robs[key]["rgb"])
        except Exception:  # noqa: BLE001
            return None

    def _rgb_keys(self, robs: Dict[str, Any]) -> List[str]:
        keys = [k for k, v in robs.items() if isinstance(v, dict) and "rgb" in v]
        if not self.cfg.all_cameras:
            keys = [k for k in keys if k == self.cam_key]
        return keys

    def _proprio(self, robs: Dict[str, Any]) -> Optional[np.ndarray]:
        p = robs.get("proprio")
        if p is None:
            return None
        return np.asarray(th.as_tensor(p).detach().cpu(), dtype=np.float32).reshape(-1)

    def start(self, obs0, info0=None, state0: Optional[th.Tensor] = None) -> None:
        """Record the reset observation and the initial serialized state (``state[0]``); create the sinks."""
        self.reset_obs = obs0
        if state0 is not None:
            self.states = [th.as_tensor(state0).detach().cpu().float().clone()]
        robs = self._robot_obs(obs0)
        self._video_keys = self._rgb_keys(robs)
        frames = self._step_frames(robs)
        self._last_obs_lerobot = self._lerobot_obs(robs, frames)
        if frames is None:
            self._keep_head_frame(obs0)
        else:
            frame = frames.get(self.cam_key)
            if frame is not None:
                self.last_rgb = frame
        if self.cfg.behavior_proprio:
            self._bproprio_last = self._bproprio_committed = self._behavior_proprio_now()
            self._pending_bproprio = []
        if self.cfg.hdf5:
            self._snapshot_scene()
        if self.view_paths:
            self._open_views(robs)
        if self.cfg.lerobot:
            self._create_lerobot(robs)
        if self.cfg.behavior_shard:
            try:
                from ispatialgym.oracle.behavior_lerobot import ShardWriter

                root, repo_id = os.path.split(os.path.abspath(self.cfg.behavior_shard))
                self.behavior = ShardWriter(self.env, root, repo_id, self.cfg.behavior_task or self.task_name or "", async_flush=self.cfg.async_shard)
                self.behavior.begin(obs0, info0)
                self.behavior_status = "recording"
            except Exception as e:  # noqa: BLE001
                log.warning("BEHAVIOR LeRobot shard not opened: %r", e)
                self.behavior, self.behavior_status = None, f"error: {e!r}"

    def _snapshot_scene(self) -> None:
        try:
            import copy

            task = getattr(self.env, "task", None)
            if task is not None and hasattr(task, "update_bddl_scope_metadata"):
                task.update_bddl_scope_metadata(self.env)
            self.scene_file = self.env.scene.save()
            cfg = copy.deepcopy(getattr(self.env, "config", None))
            # BEHAVIOR's playback builds its env from this config: a demo recorded in one of several scenes
            # (ispatialgym.multienv) replays in a one-scene simulator
            env_cfg = cfg.get("env") if isinstance(cfg, dict) else None
            if isinstance(env_cfg, dict) and env_cfg.get("num_envs", 1) != 1:
                env_cfg["num_envs"] = 1
            if self.nominal_image_size and isinstance(cfg, dict):
                # 2026-10-31 train-fast PHYSICS_ONLY_TRIM: no image is recorded and the live cameras were shrunk (run_isg); the config a
                # playback builds its env from keeps the evaluation camera size (the hdf5 contract is unchanged)
                for rcfg in cfg.get("robots") or []:
                    kw = ((rcfg.get("sensor_config") or {}).get("VisionSensor") or {}).get("sensor_kwargs") if isinstance(rcfg, dict) else None
                    if isinstance(kw, dict):
                        for k in ("image_height", "image_width"):
                            if k in kw:
                                kw[k] = int(self.nominal_image_size)
            self.env_config = cfg
        except Exception as e:  # noqa: BLE001
            log.warning("scene snapshot for the hdf5 header failed: %r", e)

    def _open_views(self, robs: Dict[str, Any]) -> None:
        for arm in M.ARMS:
            self._wrist_keys[arm] = next((k for k, v in robs.items() if f"{arm}_realsense_link" in k and isinstance(v, dict) and "rgb" in v), None)
        sources = {"head": self.cam_key in robs, "external": bool(self.cfg.external_camera),
                   "left_wrist": self._wrist_keys["left"] is not None, "right_wrist": self._wrist_keys["right"] is not None}
        for view, path in self.view_paths.items():
            if not sources.get(view, True):
                self.view_errors[view] = "no camera" if view != "external" else "no external camera configured"
                log.warning("%s video not recorded: %s", view, self.view_errors[view])
                continue
            if view == "tiled" and not all(sources.values()):
                log.warning("tiled video: no %s camera, its tile stays black", ", ".join(v for v, ok in sources.items() if not ok))
            try:
                import imageio.v2 as imageio

                self._writers[view] = imageio.get_writer(
                    path, fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                    ffmpeg_params=["-crf", str(self.cfg.video_crf), "-preset", "veryfast"],
                )
            except Exception as e:  # noqa: BLE001
                self.view_errors[view] = repr(e)
                log.warning("%s video writer unavailable (%r); recording without it", view, e)

    # ------------------------------------------------------------------ LeRobot

    def _lerobot_features(self, robs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        action_dim = int(getattr(self.robot, "action_dim", M.ACTION_DIM))
        features: Dict[str, Dict[str, Any]] = {
            "action": {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
            "next.terminated": {"dtype": "bool", "shape": (1,), "names": ["done"]},
            "next.truncated": {"dtype": "bool", "shape": (1,), "names": ["done"]},
            "annotation.action_index": {"dtype": "int64", "shape": (1,), "names": ["action_index"]},
        }
        self._obs_mapping = {}
        for key in self._video_keys:
            rgb = robs[key]["rgb"]
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            name = f"observation.images.{sensor_name(key)}"
            features[name] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"], "info": {"is_depth_map": False}}
            self._obs_mapping[key] = name
        proprio = self._proprio(robs)
        if proprio is not None:
            features["observation.state"] = {"dtype": "float32", "shape": (int(proprio.shape[0]),), "names": None}
            self._obs_mapping["proprio"] = "observation.state"
        return features

    def _lerobot_obs(self, robs: Dict[str, Any], frames: Optional[_StepFrames] = None) -> Dict[str, Any]:
        """The observation part of a LeRobot frame (frames buffered, proprio as float32)."""
        out: Dict[str, Any] = {}
        for key in self._video_keys:
            if frames is not None:
                fr = frames.get(key)
                if fr is not None:
                    out[key] = fr
                continue
            try:
                out[key] = _Frame(to_uint8_rgb(robs[key]["rgb"]), self.cfg.frame_buffer)
            except Exception:  # noqa: BLE001
                pass
        p = self._proprio(robs)
        if p is not None:
            out["proprio"] = p
        return out

    def _create_lerobot(self, robs: Dict[str, Any]) -> None:
        try:
            from lerobot.configs import VideoEncoderConfig
            from lerobot.datasets import LeRobotDataset

            features = self._lerobot_features(robs)
            repo_id = self.cfg.lerobot_repo_id or f"ispatialgym/{self.instance_id}"
            root = os.path.join(self.lerobot_root, repo_id)
            if os.path.exists(root):
                import shutil

                shutil.rmtree(root)
            encoder = VideoEncoderConfig(vcodec=self.cfg.lerobot_vcodec, pix_fmt="yuv420p", g=8, crf=30, extra_options={"x265-params": "log-level=0:bframes=0"})
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id, fps=self.fps, features=features, root=root, robot_type=self.cfg.lerobot_robot_type,
                use_videos=True, streaming_encoding=True, camera_encoder=encoder,
            )
        except Exception as e:  # noqa: BLE001
            self.lerobot_error = repr(e)
            self.dataset = None
            log.warning("LeRobot dataset unavailable (%r); recording hdf5 / video only", e)

    def _lerobot_frame(self, prev_obs: Dict[str, Any], action: th.Tensor, reward: float, terminated: bool, truncated: bool, action_index: int, task: str) -> Dict[str, Any]:
        frame: Dict[str, Any] = {
            "action": th.as_tensor(action).detach().cpu().float(),
            "next.reward": th.tensor([float(reward)], dtype=th.float32),
            "next.terminated": th.tensor([bool(terminated)]),
            "next.truncated": th.tensor([bool(truncated)]),
            "annotation.action_index": th.tensor([int(action_index)], dtype=th.int64),
            "task": task,
        }
        for key, name in self._obs_mapping.items():
            v = prev_obs.get(key)
            if v is None:
                continue
            frame[name] = v  # _Frame or float32 array; materialised at commit
        return frame

    # ------------------------------------------------------------------ per step

    def record(self, step: int, action: th.Tensor, obs, info: dict, state: Optional[th.Tensor] = None, phase: int = PHASE_ORACLE,
               action_index: int = -1, attempt: int = 0, reward: float = 0.0, terminated: bool = False, truncated: bool = False,
               sim_time: Optional[float] = None) -> None:
        """Buffer one executed step into the pending segment."""
        r = self._pending_rows
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        r["step"].append(int(step))
        r["sim_time"].append(float("nan") if sim_time is None else float(sim_time))
        r["phase"].append(int(phase))
        r["action_index"].append(int(action_index))
        r["attempt"].append(int(attempt))
        r["action"].append(np.asarray(th.as_tensor(action).detach().cpu(), dtype=np.float32))
        r["reward"].append(float(reward))
        r["terminated"].append(bool(terminated))
        r["truncated"].append(bool(truncated))
        r["joint_positions"].append(np.asarray(q.detach().cpu(), dtype=np.float32))
        r["joint_velocities"].append(np.asarray(dq.detach().cpu(), dtype=np.float32))
        r["base_pose"].append(pose7(*self.robot.get_position_orientation()))
        for arm in M.ARMS:
            r[f"eef_pose_{arm}"].append(pose7(*self.robot.eef_links[arm].get_position_orientation()))
        if self.cam_key in getattr(self.robot, "sensors", {}):
            r["head_camera_pose"].append(pose7(*self.robot.sensors[self.cam_key].get_position_orientation()))
        for inst, obj in self.objects.items():
            r[f"object_pose/{inst}"].append(pose7(*obj.get_position_orientation()))
        if RECORD_LINK_POSES:
            r["link_poses"].append(np.stack([pose7(*link.get_position_orientation()) for link in self._pose_links()]))
        isg = (info or {}).get("done", {}).get("ispatialgym") or (info or {}).get("ispatialgym") or {}
        # the task's facts of the step (ispatialgym.eval.metrics): G, S, M and the error metric of the (:measure ...) atoms
        r["eval/symbolic"].append(bool(isg.get("goal", False)))
        r["eval/settled"].append(bool(isg.get("settled", False)))
        r["eval/geometric_within"].append(bool(isg.get("measure_within", False)))
        metric = isg.get("error_metric") or {}
        for key in ("position_error", "orientation_error"):
            v = metric.get(key)
            r[f"eval/{key}"].append(float("nan") if v is None else float(v))
        robs = self._robot_obs(obs)
        p = self._proprio(robs)
        if p is not None:
            r["proprio"].append(p)
        if self.cfg.behavior_proprio and self._bproprio_last is not None:
            r["behavior_proprio"].append(self._bproprio_last)
            self._bproprio_last = self._behavior_proprio_now()
            self._pending_bproprio.append(self._bproprio_last)
        if state is not None:
            self._pending_states.append(th.as_tensor(state).detach().cpu().float().clone())
        frames = self._step_frames(robs)
        head = None
        if frames is not None:
            frame = frames.get(self.cam_key)
            if frame is not None:
                self.last_rgb = frame
                if "head" in self._writers:
                    self._pending_frames["head"].append(frame)
        elif self._writers:
            head = self.head_frame(obs)
            if head is not None:
                self.last_rgb = head
                if "head" in self._writers:
                    self._pending_frames["head"].append(_Frame(head, self.cfg.frame_buffer))
        else:
            self._keep_head_frame(obs)
        if self._writers:
            self._buffer_views(obs, robs, head, frames)
        if self.dataset is not None and self._last_obs_lerobot is not None:
            self._pending_lerobot.append(self._lerobot_frame(self._last_obs_lerobot, action, reward, terminated, truncated, action_index, self.current_task))
        if self.dataset is not None:
            self._last_obs_lerobot = self._lerobot_obs(robs, frames)
        if self.behavior is not None:
            try:
                self.behavior.step(r["action"][-1], obs, reward, terminated, truncated, info)
            except Exception as e:  # noqa: BLE001
                log.warning("BEHAVIOR LeRobot step not written: %r", e)
                self.behavior.abort()
                self.behavior_status = f"error: {e!r}"

    def _buffer_views(self, obs, robs: Dict[str, Any], head: Optional[np.ndarray], frames: Optional[_StepFrames]) -> None:
        """Buffer the step's frame of every view but the head one (buffered with ``last_rgb``)."""
        views = [v for v in self._writers if v != "head"]
        if not views:
            return
        wrist = {arm: self._camera_frame(robs, self._wrist_keys.get(arm)) for arm in M.ARMS
                 if f"{arm}_wrist" in views or "tiled" in views}
        external = self.external_frame(obs) if ("external" in views or "tiled" in views) else None
        for view in views:
            if view == "tiled":
                if head is None and frames is not None:
                    frame = frames.get(self.cam_key)
                    head = None if frame is None else frame.array()
                frame = tiled_frame(wrist.get("left"), head, wrist.get("right"), external)
            elif view == "external":
                frame = external
            else:
                frame = wrist.get(view.split("_")[0])
            if frame is not None:
                self._pending_frames[view].append(_Frame(frame, self.cfg.frame_buffer))

    @property
    def n_pending(self) -> int:
        return len(self._pending_rows["step"])

    @property
    def n_total(self) -> int:
        return self.n + self.n_pending

    # ------------------------------------------------------------------ segments

    def commit(self) -> int:
        """Move the pending segment to the persistent sinks; returns the number of committed steps."""
        n = self.n_pending
        if n == 0:
            return 0
        for k, v in self._pending_rows.items():
            self.rows[k].extend(v)
        self.states.extend(self._pending_states)
        if self.cfg.behavior_proprio:
            self._bproprio_committed = self._bproprio_last
        if self._pipe is not None:
            # the sink thread writes them in this order; close() drains it before closing the sinks
            for view in list(self._writers):
                for fr in self._pending_frames[view]:
                    self._pipe.sink(self._sink_view_frame, fr, view)
            if self.dataset is not None:
                for frame in self._pending_lerobot:
                    self._pipe.sink(self._sink_lerobot_frame, frame)
        else:
            for view in list(self._writers):
                for fr in self._pending_frames[view]:
                    self._write_view_frame(view, fr.array())
            if self.dataset is not None:
                for frame in self._pending_lerobot:
                    self._add_lerobot_frame(frame)
        self.n += n
        self._clear_pending()
        return n

    def mark(self) -> int:
        """Position in the pending segment (number of pending steps) a later ``rollback(mark=...)`` truncates to."""
        return self.n_pending

    def rollback(self, action_index: int, attempt: int, verdict: Optional[Dict[str, Any]] = None, state_before: Optional[th.Tensor] = None, mark: int = 0) -> int:
        """Drop the pending steps recorded after ``mark`` (a failed attempt; the whole pending segment by default);
        kept under ``rollbacks`` when configured (``state_before`` = the checkpoint state, so the stored segment keeps
        the BEHAVIOR pairing state[i] before action[i]). Steps before ``mark`` (an earlier, deferred-commit action of
        the same group — the brace of a braced push) stay pending."""
        mark = max(0, min(int(mark), self.n_pending))
        n = self.n_pending - mark
        if n and self.behavior is not None and not self.behavior.aborted:
            # a mid-episode retry: the demo gets its BEHAVIOR episode by replaying demo.hdf5 (replay_obs_isg.py)
            self.behavior.abort()
            self.behavior_status = "rollback"
        if n and self.cfg.keep_rollbacks:
            states = list(self._pending_states[mark:])
            if state_before is not None:
                states = [th.as_tensor(state_before).detach().cpu().float().clone()] + states
            self.rollbacks.append({
                "action_index": int(action_index), "attempt": int(attempt), "verdict": verdict, "n_steps": n,
                "rows": {k: list(v[mark:]) for k, v in self._pending_rows.items()}, "states": states,
            })
        if self.cfg.behavior_proprio and self._bproprio_last is not None:
            # the proprioception the kept steps end at, as the kept states: the step before the mark, or the committed state
            self._bproprio_last = self._pending_bproprio[mark - 1] if mark > 0 else self._bproprio_committed
        if mark == 0:
            self._clear_pending()
        else:
            for k in list(self._pending_rows):
                del self._pending_rows[k][mark:]
            del self._pending_states[mark:]
            del self._pending_bproprio[mark:]
            for k in list(self._pending_frames):
                del self._pending_frames[k][mark:]
            del self._pending_lerobot[mark:]
        self._last_obs_lerobot = None  # re-armed by the executor with the checkpoint observation (set_observation)
        return n

    def set_observation(self, obs) -> None:
        """The observation of the restored state (pairs with the first action after a rollback)."""
        robs = self._robot_obs(obs)
        frames = self._step_frames(robs)
        self._last_obs_lerobot = self._lerobot_obs(robs, frames) if self.dataset is not None else None
        if frames is None:
            self._keep_head_frame(obs)
        else:
            frame = frames.get(self.cam_key)
            if frame is not None:
                self.last_rgb = frame

    def _clear_pending(self) -> None:
        self._pending_rows = defaultdict(list)
        self._pending_states = []
        self._pending_bproprio = []
        self._pending_frames = defaultdict(list)
        self._pending_lerobot = []

    def _write_view_frame(self, view: str, frame: np.ndarray) -> None:
        writer = self._writers.get(view)
        if writer is None:
            return
        try:
            size = self.cfg.video_size
            if size and frame.shape[1] != size:
                frame = _resized(frame, size, int(round(frame.shape[0] * size / frame.shape[1])))
            writer.append_data(frame)
        except Exception as e:  # noqa: BLE001
            self.view_errors[view] = repr(e)
            log.warning("%s video frame dropped (%r); closing that writer", view, e)
            self._close_writer(view)

    def _sink_view_frame(self, fr: _Frame, view: str) -> None:
        if view in self._writers:
            self._write_view_frame(view, fr.array())

    def _sink_lerobot_frame(self, frame: Dict[str, Any]) -> None:
        if self.dataset is not None:
            self._add_lerobot_frame(frame)

    def _add_lerobot_frame(self, frame: Dict[str, Any]) -> None:
        try:
            out = {}
            for k, v in frame.items():
                if isinstance(v, _Frame):
                    out[k] = th.from_numpy(np.ascontiguousarray(v.array()))
                elif isinstance(v, np.ndarray):
                    out[k] = th.from_numpy(v.astype(np.float32))
                else:
                    out[k] = v
            self.dataset.add_frame(out)
        except Exception as e:  # noqa: BLE001
            self.lerobot_error = repr(e)
            log.warning("LeRobot add_frame failed (%r); disabling the LeRobot sink", e)
            self.dataset = None

    # ------------------------------------------------------------------ annotations

    def begin_action_annotation(self, index: int, action, fine: str, numbers: str) -> None:
        self.current_task = fine or self.task_name
        self.actions.append(ActionAnnotation(
            index=int(index), name=action.name, arms=list(action.arms), objects=list(action.objects), fine=fine, numbers=numbers,
            step_start=self.n_total, step_end=self.n_total,
        ))

    def end_action_annotation(self, verdict: Optional[Dict[str, Any]] = None, attempts: int = 1) -> None:
        if not self.actions:
            return
        a = self.actions[-1]
        a.step_end = self.n_total
        a.attempts = int(attempts)
        a.verdict = verdict
        self.current_task = self.task_name

    def note_rollback(self, verdict: Optional[Dict[str, Any]]) -> None:
        if self.actions:
            self.actions[-1].rolled_back.append(verdict or {})

    def annotations(self, qa: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "episode": self.episode.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "qa": qa,
            "phase_codes": PHASE_CODES,
            "n_steps": self.n,
        }

    # ------------------------------------------------------------------ finish

    def _close_writer(self, view: Optional[str] = None) -> None:
        for v in ([view] if view is not None else list(self._writers)):
            writer = self._writers.pop(v, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass

    def written_views(self) -> Dict[str, str]:
        """The closed, complete videos: view -> path."""
        return {v: p for v, p in self.view_paths.items() if v not in self._writers and v not in self.view_errors and os.path.exists(p)}

    def close(self, meta: dict, qa: Optional[Dict[str, Any]] = None, success: bool = False) -> Dict[str, Any]:
        """Flush everything; returns the paths written (``None`` where a sink was off or failed)."""
        self.commit()
        if self._pipe is not None:
            self._pipe.drain()
        self._close_writer()
        out: Dict[str, Any] = {"video": None, "hdf5": None, "lerobot": None, "annotations": None}
        ann = self.annotations(qa)
        try:
            path = os.path.join(self.out_dir, "annotations.json")
            dump_json(path, ann)
            out["annotations"] = path
        except Exception as e:  # noqa: BLE001
            log.warning("annotations.json not written: %r", e)
        out["views"] = self.written_views()
        out["video"] = out["views"].get("head")
        if self.view_errors:
            out["view_errors"] = dict(self.view_errors)
        if self.cfg.hdf5 and self.n > 0:
            try:
                out["hdf5"] = self._write_hdf5(meta, ann)
            except Exception as e:  # noqa: BLE001
                log.warning("demo.hdf5 not written: %r", e)
                meta["hdf5_error"] = repr(e)
        if self.dataset is not None:
            try:
                if self.n > 0:
                    self.dataset.save_episode()
                self.dataset.finalize()
                out["lerobot"] = self.dataset.root if isinstance(self.dataset.root, str) else str(self.dataset.root)
                dump_json(os.path.join(out["lerobot"], "annotations.json"), ann)
            except Exception as e:  # noqa: BLE001
                self.lerobot_error = repr(e)
                log.warning("LeRobot episode not saved: %r", e)
        if self.behavior is not None or self.behavior_status:
            index = None
            if self.behavior is not None:
                if self.cfg.hdf5 and not out.get("hdf5"):
                    self.behavior_status = "no_hdf5"  # the demo is recorded again; its episode would come twice
                try:
                    index = self.behavior.end(save=bool(success) and self.behavior_status == "recording")
                    self.behavior.close()
                except Exception as e:  # noqa: BLE001
                    log.warning("BEHAVIOR LeRobot episode not saved: %r", e)
                    self.behavior_status = f"error: {e!r}"
                self.behavior = None
            if index is not None:
                self.behavior_status = "saved"
            elif self.behavior_status == "recording":
                self.behavior_status = "not_saved"  # a failed demo
            out["behavior_lerobot"] = {"shard": os.path.abspath(self.cfg.behavior_shard) if self.cfg.behavior_shard else None,
                                       "episode_index": index, "status": self.behavior_status,
                                       "needs_replay": bool(success) and index is None}
        if self._pipe is not None:
            self._pipe.close()
        out["video_error"] = self.view_errors.get("head")
        out["lerobot_error"] = self.lerobot_error
        return out

    def _write_hdf5(self, meta: dict, annotations: Dict[str, Any]) -> str:
        import h5py

        path = self.hdf5_path
        T = self.n
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs["config"] = json.dumps(self.env_config, default=json_default) if self.env_config is not None else "{}"
            if self.scene_file is not None:
                data.attrs["scene_file"] = json.dumps(self.scene_file, default=json_default) if isinstance(self.scene_file, dict) else self.scene_file
            try:
                from omnigibson.utils.asset_utils import get_omnigibson_git_hash

                data.attrs["omnigibson_git_hash"] = get_omnigibson_git_hash()
            except Exception:  # noqa: BLE001
                data.attrs["omnigibson_git_hash"] = ""
            data.attrs["n_episodes"] = 1
            data.attrs["n_steps"] = T
            data.attrs["isg_meta"] = json.dumps(meta, default=json_default)
            data.attrs["isg_schema"] = "isg-demo-v2"
            data.attrs["controller_order"] = json.dumps(M.CONTROLLER_ORDER)
            data.attrs["controller_dims"] = json.dumps(M.CONTROLLER_DIMS)
            data.attrs["joint_names"] = json.dumps(list(getattr(self.robot, "joints", {}).keys()))
            data.attrs["pose_layout"] = "pos xyz, quat xyzw"
            if RECORD_LINK_POSES:
                data.attrs["isg_link_pose_paths"] = json.dumps([link.prim_path for link in self._pose_links()])
            data.attrs["fps"] = self.fps
            data.attrs["robot_name"] = self.robot.name
            data.attrs["head_camera_key"] = self.cam_key
            for k, v in scene_crop_attrs(self.scene_crop, self.full_scene_file, self.full_load_room_instances).items():
                data.attrs[k] = v  # 2026-11-01 train-fast PARTIAL_SCENE (none without a crop)
            views = self.written_views()
            if "head" in views:
                data.attrs["head_rgb_video"] = os.path.basename(views["head"])
            if views:
                data.attrs["isg_videos"] = json.dumps({v: os.path.basename(p) for v, p in views.items()})
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = T
            demo.attrs["transitions"] = json.dumps({})
            demo.attrs["isg_annotations"] = json.dumps(annotations, default=json_default)
            demo.create_group("init_metadata")
            self._write_behavior_datasets(demo, self.rows, self.states, T)
            isg = demo.create_group("isg")
            for key, rows in self.rows.items():
                if key in ("action", "reward", "terminated", "truncated"):
                    continue
                arr = np.asarray(rows)
                isg.create_dataset(key, data=arr, compression="gzip" if arr.ndim > 1 else None)
            if self.rollbacks:
                rb = demo.create_group("rollbacks")
                for r in self.rollbacks:
                    g = rb.create_group(f"action_{r['action_index']}_attempt_{r['attempt']}")
                    g.attrs["verdict"] = json.dumps(r["verdict"], default=json_default)
                    self._write_behavior_datasets(g, r["rows"], r["states"], r["n_steps"])
                    for key, rows in r["rows"].items():
                        if key not in ("action", "reward", "terminated", "truncated"):
                            g.create_dataset(f"isg/{key}", data=np.asarray(rows))
        return path

    @staticmethod
    def _write_behavior_datasets(grp, rows: Dict[str, list], states: Sequence[th.Tensor], T: int) -> None:
        grp.create_dataset("action", data=np.asarray(rows["action"], dtype=np.float32)[:T])
        grp.create_dataset("reward", data=np.asarray(rows["reward"], dtype=np.float32)[:T])
        grp.create_dataset("terminated", data=np.asarray(rows["terminated"], dtype=bool)[:T])
        grp.create_dataset("truncated", data=np.asarray(rows["truncated"], dtype=bool)[:T])
        if states:
            # BEHAVIOR pairing: state[i] is the state before action[i]; the list holds the pre-segment state first
            sel = list(states[:T])
            if sel:
                smax = max(int(s.numel()) for s in sel)
                padded = np.zeros((len(sel), smax), dtype=np.float32)
                sizes = np.zeros((len(sel),), dtype=np.int64)
                for i, s in enumerate(sel):
                    arr = s.numpy().astype(np.float32)
                    padded[i, : arr.shape[0]] = arr
                    sizes[i] = arr.shape[0]
                grp.create_dataset("state", data=padded, compression="gzip")
                grp.create_dataset("state_size", data=sizes)


__all__ = [
    "PHASE_ORACLE", "PHASE_SETTLE", "PHASE_GOAL", "PHASE_CODES", "Recorder", "RecorderConfig", "ActionAnnotation",
    "EpisodeAnnotations", "json_default", "dump_json", "save_png", "to_uint8_rgb", "pose7", "sensor_name",
]
