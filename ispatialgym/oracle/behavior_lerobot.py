"""BEHAVIOR-identical LeRobot v3 output written while a demo is collected.

BEHAVIOR's ``OmniGibson/scripts/learning/replay_obs.py`` turns a recorded ``demo.hdf5`` into LeRobot v3 by playing it back
through ``LeRobotPlaybackWrapper``. :class:`ShardWriter` produces the same episodes from the live demo instead: it is
BEHAVIOR's ``LeRobotDataWrapper`` (features, ``_process_obs``, video / depth encoders, frame pairing, the streaming
``flush_partial_traj`` of the playback wrapper) fed by :class:`~ispatialgym.oracle.recorder.Recorder` rather than stepping
the environment itself, and the robot is configured with replay_obs's sensors, modalities and proprioception keys
(:func:`robot_overrides`). One shard (one LeRobot v3 dataset, many episodes) per runner process: every episode is
finalized when it is saved and the dataset resumed for the next, so a process that dies loses at most the episode it was
recording. An episode with a mid-episode retry (a rollback) is dropped here; its ``demo.hdf5`` is replayed with
``scripts/isg/replay_obs_isg.py`` instead.
"""

from __future__ import annotations

import glob
import os
import queue
import shutil
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the shards are local; lerobot must never fetch a dataset from the Hub

#: Opt-in shard formats, both changing recorded data. One setting per dataset: shards with and without depth cannot be
#: aggregated (different features), and x265 / NVENC episodes must not be concatenated into one video by stream copy.
#: ISPATIALGYM_SHARD_NO_DEPTH=1: the robot records proprio + rgb only (no depth annotators, readback or streams) and the
#: datasets keep observation.robot2cam_pose.<camera> for every camera (BEHAVIOR's writer adds them only alongside depth).
#: ISPATIALGYM_SHARD_RGB_ENCODER="hevc_nvenc,qp=<n>,preset=<p>": the rgb streams' encoder (GOP 8, yuv420p, no B-frames).
#: Both are passed on to BEHAVIOR's writer, which both this module and scripts/isg/replay_obs_isg.py (it imports
#: ROBOT_OBS_MODALITIES from here) build their datasets with.
SHARD_NO_DEPTH = os.environ.get("ISPATIALGYM_SHARD_NO_DEPTH") == "1"
if SHARD_NO_DEPTH:
    os.environ["OMNIGIBSON_LEROBOT_ROBOT2CAM_WITH_RGB"] = "1"
if os.environ.get("ISPATIALGYM_SHARD_RGB_ENCODER"):
    os.environ["OMNIGIBSON_LEROBOT_RGB_ENCODER"] = os.environ["ISPATIALGYM_SHARD_RGB_ENCODER"]

import gymnasium as gym
import torch as th

from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.utils.gym_utils import recursively_generate_flat_dict

from ispatialgym.oracle import lerobot_compat

lerobot_compat.install()

#: replay_obs.replay_hdf5_file's playback settings (robot_sensor_config, robot_obs_modalities, robot_proprio_keys, robot_type)
ROBOT_SENSOR_CONFIG = {
    "VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}},
    "zed_link:Camera:0": {"sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": 720, "image_width": 720}},
}
ROBOT_OBS_MODALITIES = ["proprio", "rgb"] if SHARD_NO_DEPTH else ["proprio", "rgb", "depth_linear"]
ROBOT_PROPRIO_KEYS = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
ROBOT_TYPE = "R1Pro"
HEAD_IMAGE_SIZE = 720
#: frames handed to the LeRobot writer every this many steps (streaming encoding keeps memory bounded; the output does not
#: depend on it)
FLUSH_EVERY_N_STEPS = 32


def warm_camera_params(env, max_renders: int = 20) -> None:
    """Make every robot camera's intrinsics readable before the writer reads them (``create_dataset`` asserts they are not
    degenerate). ``VisionSensor.camera_parameters`` attaches the camera-params annotator and renders 4 frames on first use,
    which left a wrist camera's projection still zero during collection; renders (no physics step) until none is."""
    import omnigibson as og
    from omnigibson.sensors.vision_sensor import VisionSensor

    for sensor in env.scene.robots[0].sensors.values():
        if not isinstance(sensor, VisionSensor):
            continue
        for _ in range(max_renders):
            if float(th.as_tensor(sensor.camera_parameters["cameraProjection"]).abs().sum()) > 0:
                break
            og.sim.render()


def robot_overrides() -> Dict[str, Any]:
    """``joylo_robot_config`` overrides giving the robot the sensors and observations replay_obs.py records with (what
    ``DataPlaybackWrapper.create_from_hdf5`` sets on the recorded robot config)."""
    return {"obs_modalities": list(ROBOT_OBS_MODALITIES), "proprio_obs": list(ROBOT_PROPRIO_KEYS), "sensor_config": ROBOT_SENSOR_CONFIG,
            "include_sensor_names": None, "exclude_sensor_names": None}


class FlatObsView(EnvironmentWrapper):
    """The environment as BEHAVIOR's playback environment presents itself to the writer: ``flatten_obs_space`` and no task
    observations, i.e. the flattened space of the robot's proprio / rgb / depth_linear observations."""

    def __init__(self, env):
        super().__init__(env=env)
        robot = env.scene.robots[0].name
        flat = recursively_generate_flat_dict(dic=env.observation_space)
        keys = [k for k in flat if k.split("::")[0] == robot and k.split("::")[-1] in ROBOT_OBS_MODALITIES]
        object.__setattr__(self, "_flat_keys", keys)
        object.__setattr__(self, "_flat_space", gym.spaces.Dict({k: flat[k] for k in keys}))

    @property
    def observation_space(self):
        return self._flat_space

    def flat(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        f = recursively_generate_flat_dict(dic=obs)
        return {k: f[k] for k in self._flat_keys}


class _BackgroundFlush:
    """``ShardWriter(async_flush=True)``: one worker thread that hands flushed trajectory segments to the LeRobot writer,
    in submission order. Up to ``max_pending`` segments wait (each holds 32 steps of images); beyond that the stepping
    thread waits, as it waits on the encoders' queues without it. A writer error is kept and raised on the stepping
    thread at its next call."""

    def __init__(self, max_pending: int = 2):
        self._jobs: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_pending)
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="isg-shard-flush", daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> None:
        self._jobs.put((fn, args))

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                if self.error is None:
                    fn, args = job
                    fn(*args)
            except BaseException as e:  # noqa: BLE001
                self.error = e
            finally:
                self._jobs.task_done()

    def drain(self, cancel: bool = False) -> None:
        """Wait until every submitted segment is written; ``cancel``: drop the ones not started yet."""
        if cancel:
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break
                self._jobs.task_done()
        self._jobs.join()

    def raise_error(self) -> None:
        if self.error is not None:
            e, self.error = self.error, None
            raise e

    def close(self) -> None:
        self.drain()
        self._jobs.put(None)
        self._thread.join(timeout=60)


class ShardWriter(LeRobotDataWrapper):
    """One LeRobot v3 dataset (``<root>/<repo_id>``) written episode by episode from the recorder's steps.

    ``async_flush``: every ``FLUSH_EVERY_N_STEPS`` steps the segment goes to :class:`_BackgroundFlush`, which runs the
    playback wrapper's ``flush_partial_traj`` on it (the same frames, in the same order) while the demo steps on."""

    def __init__(self, env, root: str, repo_id: str, task_name: str, async_flush: bool = False):
        warm_camera_params(env)
        view = FlatObsView(env)
        path = os.path.join(root, repo_id)
        if os.path.exists(path) and not glob.glob(os.path.join(path, "meta", "episodes", "*", "*.parquet")):
            # no episode saved yet (the runner's first one was dropped): lerobot cannot resume it from the local files and
            # falls back to the Hub; start the shard over
            shutil.rmtree(path)
        super().__init__(env=view, output_path=repo_id, root_dir=root, overwrite=not os.path.exists(path),
                         only_successes=False, flush_every_n_traj=1, robot_type=ROBOT_TYPE, task_name=task_name)
        self.flush_every_n_steps = FLUSH_EVERY_N_STEPS
        self.current_episode_step_count = 0
        self.aborted = False
        self.root, self.repo_id = root, repo_id
        self._bg_flush = _BackgroundFlush() if async_flush else None

    @property
    def should_save_current_episode(self) -> bool:
        # the recorder decides (successful demos without a retry); an episode needs at least one step
        return len(self.current_traj_history) > 1 or self.current_episode_step_count > 0

    def begin(self, obs, info: Optional[dict] = None) -> None:
        """The observation before the first recorded action (``state[0]``)."""
        self.current_traj_history = [{"obs": self._process_obs(self.env.flat(obs), info or {})}]
        self.current_episode_step_count = 0
        self.aborted = False

    def step(self, action, obs, reward: float, terminated: bool, truncated: bool, info: Optional[dict] = None) -> None:
        """One recorded step: the action and the observation it led to (paired with the next action by the writer)."""
        if self.aborted or not self.current_traj_history:
            return
        if self._bg_flush is not None:
            self._bg_flush.raise_error()
        action = th.as_tensor(action, dtype=th.float32).detach().cpu()
        self._record_step_trajectory(action, self.env.flat(obs), float(reward), bool(terminated), bool(truncated), info or {})
        self.current_episode_step_count += 1
        if len(self.current_traj_history) > self.flush_every_n_steps:
            if self._bg_flush is None:
                LeRobotPlaybackWrapper.flush_partial_traj(self, step_idx=self.current_episode_step_count, total_steps=0, step_data=None)
            else:
                # flush_partial_traj on a view holding only this segment; the stepping thread keeps its last observation
                # (paired with the next action), as flush_partial_traj leaves it
                segment = SimpleNamespace(current_traj_history=self.current_traj_history, flush_every_n_steps=self.flush_every_n_steps,
                                          current_episode_step_count=self.current_episode_step_count, task_name=self.task_name,
                                          dataset=self.dataset)
                self.current_traj_history = [{"obs": segment.current_traj_history[-1]["obs"]}]
                self._bg_flush.submit(LeRobotPlaybackWrapper.flush_partial_traj, segment, segment.current_episode_step_count, 0, None)

    def abort(self) -> None:
        """Drop the episode being written (a rollback, or a failed demo)."""
        if self._bg_flush is not None:
            self._bg_flush.drain(cancel=True)
            self._bg_flush.error = None
        if self.dataset is not None and self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self.current_traj_history = []
        self.aborted = True

    def end(self, save: bool) -> Optional[int]:
        """Save (``save``) or drop the episode; returns its episode index in the shard when saved."""
        if self._bg_flush is not None:
            self._bg_flush.drain(cancel=self.aborted or not save or self.current_episode_step_count == 0)
            if self._bg_flush.error is not None:
                # without async_flush this error would have come from the step that flushed, and the recorder would have
                # dropped the episode and still closed the shard: do both here, then report it
                err, self._bg_flush.error = self._bg_flush.error, None
                self.abort()
                self.close()
                raise err
        if self.aborted or not save or self.current_episode_step_count == 0:
            self.abort()
            return None
        index = int(self.dataset.num_episodes)
        self.flush_current_traj()
        return index

    def close(self) -> None:
        if self._bg_flush is not None:
            self._bg_flush.close()
            self._bg_flush = None
        if self.dataset is not None:
            self.close_dataset()
            self.dataset = None


__all__ = ["ROBOT_SENSOR_CONFIG", "ROBOT_OBS_MODALITIES", "ROBOT_PROPRIO_KEYS", "ROBOT_TYPE", "HEAD_IMAGE_SIZE", "robot_overrides",
           "FlatObsView", "ShardWriter"]
