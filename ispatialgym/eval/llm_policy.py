"""Base class for an external (e.g. LLM-harness) policy under ``run_isg --mode policy`` (scripts/llm_harness/README.md).

Subclass :class:`HarnessPolicy` and implement :meth:`HarnessPolicy.act`; ``scripts/llm_harness/run_eval.py --policy
my_module:MyPolicy`` runs it. Every control step (30 Hz) the policy gets the robot's observation and returns one 23-D action;
it sees only what a learned policy sees - the three 224 x 224 camera images, the 61-D proprioception, and the task's conditioning
(a goal image, or a text instruction) - never the simulator state or the success signal.

Conditioning (``ISG_HARNESS_CONDITION``, set by run_eval.py ``--condition``):

* ``goal``: :attr:`goal_image` = the instance's own goal image (the last head-camera frame of its demo, 224 x 224 x 3 uint8);
* ``cross_goal``: :attr:`goal_image` = its cross-episode goal image (the goal image of another demo of the same goal: same goal
  state, other start / camera, possibly other appearance), run with ``run_isg --cross-goal``; an instance without one keeps its own;
* ``text``: :attr:`prompt` = the task's text instruction (``text_prompts_iors.json``: the exact prompts the text-only iors
  checkpoint was trained with), no goal image.

Action (``ACTION_SLICES``, R1Pro's BEHAVIOR / JoyLo layout): base ``[0:3]`` velocity (vx, vy, wz) normalised to [-1, 1]
(x 0.75 m/s, 0.75 m/s, 1.0 rad/s), trunk ``[3:7]`` absolute joint positions (rad), left arm ``[7:14]`` absolute joint
positions, left gripper ``[14]`` in [-1, 1] (-1 closed, +1 open), right arm ``[15:22]``, right gripper ``[22]``.
Observation (:meth:`observation`): ``head_rgb`` / ``left_wrist_rgb`` / ``right_wrist_rgb`` (224, 224, 3) uint8, ``proprio`` (61,)
float32 and ``proprio_named`` (BEHAVIOR's R1Pro proprioception indices: ``PROPRIO_SLICES``), ``step`` (0-based).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

#: R1Pro's 23-D action (BEHAVIOR / JoyLo controller order)
ACTION_SLICES = {"base": slice(0, 3), "trunk": slice(3, 7), "arm_left": slice(7, 14), "gripper_left": slice(14, 15),
                 "arm_right": slice(15, 22), "gripper_right": slice(22, 23)}
ACTION_DIM = 23
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
#: BEHAVIOR's R1Pro proprioception (omnigibson/eval/utils/eval_utils.py PROPRIOCEPTION_INDICES); eef poses in the robot base frame
PROPRIO_SLICES = {"base_qvel": slice(0, 3), "arm_left_qpos": slice(3, 10), "arm_left_qvel": slice(10, 17), "eef_left_pos": slice(17, 20),
                  "eef_left_quat": slice(20, 24), "gripper_left_qpos": slice(24, 26), "gripper_left_qvel": slice(26, 28),
                  "arm_right_qpos": slice(28, 35), "arm_right_qvel": slice(35, 42), "eef_right_pos": slice(42, 45),
                  "eef_right_quat": slice(45, 49), "gripper_right_qpos": slice(49, 51), "gripper_right_qvel": slice(51, 53),
                  "trunk_qpos": slice(53, 57), "trunk_qvel": slice(57, 61)}
CAMERAS = {"head_rgb": "zed_link", "left_wrist_rgb": "left_realsense_link", "right_wrist_rgb": "right_realsense_link"}
CONDITIONS = ("goal", "cross_goal", "text")
PROMPTS_FILE = Path(__file__).resolve().parents[2] / "scripts" / "llm_harness" / "text_prompts_iors.json"


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.ascontiguousarray(np.asarray(x))


def load_rgb(path: str) -> np.ndarray:
    """A PNG as uint8 HWC rgb, decoded with PyAV as the policy servers decoded their goal images (ispatialgym.eval.ws_policy)."""
    import av

    with av.open(path) as container:
        frame = next(container.decode(video=0))
        return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))


class HarnessPolicy:
    """Subclass and implement :meth:`act` (and optionally :meth:`reset` / :meth:`on_episode_end`).

    Set by :meth:`bind` before the first step: :attr:`condition`, :attr:`goal_image` (None in ``text``), :attr:`prompt` (None
    unless ``text``), :attr:`task` (the dataset's ``source.class_id``, e.g. ``alignment-axial_01``), :attr:`task_family`,
    :attr:`dataset_task`, :attr:`instance_id`, :attr:`step_budget` (the episode's steps: the stream always runs to it),
    :attr:`out_dir` (this instance's results folder; files the policy writes there are kept with the run).
    """

    robot = "robot_r1"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.condition = os.environ.get("ISG_HARNESS_CONDITION", "goal")
        if self.condition not in CONDITIONS:
            raise ValueError(f"ISG_HARNESS_CONDITION={self.condition!r}: one of {CONDITIONS}")
        self.goal_image: Optional[np.ndarray] = None
        self.prompt: Optional[str] = None
        self.task = self.task_family = self.dataset_task = self.instance_id = None
        self.step_budget: Optional[int] = None
        self.out_dir: Optional[str] = None
        self.step = 0
        self.last_action: Optional[np.ndarray] = None
        self._keys: Optional[Dict[str, str]] = None
        self._meta: Optional[dict] = None
        self._act_s: list = []
        self._initial_hold: Optional[np.ndarray] = None

    # -- hooks for subclasses ------------------------------------------------------------------------------------------

    def reset(self) -> None:
        """Once per episode after :meth:`bind` (conditioning set), before the first :meth:`act`."""

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """The next 23-D action for ``obs`` (:meth:`observation`). Called every control step (30 Hz) until the step budget ends
        the episode; keep returning actions (e.g. :meth:`hold_action`) after the task is done - the success is judged on a held
        final state."""
        raise NotImplementedError

    def on_episode_end(self, result: Dict[str, Any]) -> None:
        """Once after the episode (``result``: the run's record, success fields included); e.g. close a session."""

    # -- helpers -------------------------------------------------------------------------------------------------------

    def hold_action(self) -> np.ndarray:
        """The last action with the base velocity zeroed (arms, trunk and grippers keep their commands): the robot stays where
        it is. Before the first action: the robot's start configuration."""
        a = (self.last_action if self.last_action is not None else self._initial_hold).copy()
        a[ACTION_SLICES["base"]] = 0.0
        return a

    def observation(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        robot_obs = env_obs[self.robot]
        if self._keys is None:
            self._keys = {}
            for name, cam in CAMERAS.items():
                self._keys[name] = next(k for k in robot_obs if str(k).endswith(f"{cam}:Camera:0"))
        out: Dict[str, Any] = {}
        for name, key in self._keys.items():
            img = _np(robot_obs[key]["rgb"])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            out[name] = np.ascontiguousarray(img[..., :3])
        proprio = _np(robot_obs["proprio"]).astype(np.float32).reshape(-1)
        out["proprio"] = proprio
        out["proprio_named"] = {k: proprio[s] for k, s in PROPRIO_SLICES.items()} if proprio.shape[0] == 61 else {}
        out["step"] = self.step
        return out

    # -- run_isg plumbing ----------------------------------------------------------------------------------------------

    def bind(self, executor=None, loader=None, meta=None, out_dir=None, **_):
        from ispatialgym.oracle import run_isg as R

        rec = R._dataset_record(loader)
        if rec is None:
            raise RuntimeError(f"{type(self).__name__}: {loader.tro_path} is not an eval dataset instance")
        self.task = (rec.source or {}).get("class_id")
        self.task_family, self.dataset_task, self.instance_id = rec.task_family, rec.task, rec.instance_id
        self.step_budget = int(((meta or {}).get("budget") or {}).get("max_steps") or 0) or None
        self.out_dir = str(out_dir) if out_dir else None
        cross_flag = bool(((meta or {}).get("config") or {}).get("cross_goal"))
        if (self.condition == "cross_goal") != cross_flag:
            raise RuntimeError(f"condition {self.condition} needs run_isg {'--cross-goal' if self.condition == 'cross_goal' else 'without --cross-goal'} "
                               "(run it through scripts/llm_harness/run_eval.py)")
        goal_path, goal_kind, cross = None, None, None
        if self.condition == "text":
            prompts = json.load(open(os.environ.get("ISG_HARNESS_PROMPTS", PROMPTS_FILE)))["prompts"]
            if self.task not in prompts:
                raise RuntimeError(f"no text prompt for task {self.task} (text prompts exist for the 35 iors tasks)")
            self.prompt, goal_kind = prompts[self.task], "none (text prompt)"
        else:
            cross_path = rec.path("cross_episode_goal_rgb") if rec.files.get("cross_episode_goal_rgb") else None
            if self.condition == "cross_goal" and cross_path and os.path.exists(cross_path):
                goal_path, goal_kind = cross_path, "cross_episode"
                cross = ((rec.extra or {}).get("cross_episode_goal") or {}).get("difficulty")
            else:
                goal_path = loader.goal_image_path
                goal_kind = "own (no cross-episode goal)" if self.condition == "cross_goal" else "own"
            self.goal_image = load_rgb(goal_path)
        hold = getattr(loader, "hold", None)
        if hold is not None and callable(getattr(hold, "action", None)):
            self._initial_hold = _np(hold.action()).astype(np.float32).reshape(-1)
        else:
            self._initial_hold = np.zeros(ACTION_DIM, dtype=np.float32)
        self._meta = meta
        if meta is not None:  # what summarize.py and the results folder read (goal kind, goal image copy)
            meta["policy_client"] = {"harness": type(self).__module__ + ":" + type(self).__name__, "condition": self.condition,
                                     "alias": self.task, "goal_path": goal_path, "goal_kind": goal_kind, "cross_episode_goal": cross,
                                     "prompt": self.prompt, "kwargs": {k: v for k, v in self.kwargs.items() if isinstance(v, (str, int, float, bool))}}
        if self.out_dir and self.prompt:
            Path(self.out_dir, "prompt.txt").write_text(self.prompt + "\n")
        self.step = 0
        self.reset()

    def __call__(self, obs, info):
        import torch as th

        t0 = time.time()
        a = self.act(self.observation(obs))
        self._act_s.append(time.time() - t0)
        if a is None:
            raise RuntimeError("act() returned None: keep returning actions until the step budget ends the episode (hold_action())")
        a = np.asarray(_np(a), dtype=np.float32).reshape(-1)
        if a.shape[0] != ACTION_DIM or not np.all(np.isfinite(a)):
            raise ValueError(f"act() must return {ACTION_DIM} finite values, got shape {a.shape}")
        self.last_action = a
        self.step += 1
        return th.as_tensor(a)

    def finalize(self, result) -> None:
        stats = {"n_actions": self.step, "act_s_mean": float(np.mean(self._act_s)) if self._act_s else None,
                 "act_s_max": float(np.max(self._act_s)) if self._act_s else None}
        merged = dict((self._meta or {}).get("policy_client") or {})
        merged.update(stats)
        if self._meta is not None:
            self._meta["policy_client"] = merged
        result["policy_client"] = merged  # run_isg copies the episode's policy_client over meta's
        self.on_episode_end(result)


class HoldStillPolicy(HarnessPolicy):
    """Example / smoke test: holds the start configuration for the whole episode (expected to fail every task)."""

    def act(self, obs):
        return self.hold_action()


__all__ = ["HarnessPolicy", "HoldStillPolicy", "ACTION_SLICES", "ACTION_DIM", "PROPRIO_SLICES", "GRIPPER_OPEN", "GRIPPER_CLOSED",
           "CAMERAS", "CONDITIONS"]
