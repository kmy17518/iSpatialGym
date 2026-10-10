"""ACT policy-server client as a ``run_isg --mode policy`` callable (``--policy lab_tools.act_policy:ActPolicy``).

Protocol: BEHAVIOR's NumPy-MessagePack websocket (``b1k_server.py`` of kmy17518/act). One connection per episode. Every
control step (30 Hz) the observation sent is

* ``robot_r1::proprio``: the robot's raw proprioception vector (61) as OmniGibson returns it (``obs[robot]["proprio"]``);
* ``robot_r1::robot_r1:{zed_link,left_realsense_link,right_realsense_link}:Camera:0::rgb``: uint8 HWC at native size;
* ``goal::robot_r1::robot_r1:<view>:Camera:0::rgb`` for every view in the server's ``goal_views`` (the zed goal PNG 720 x 720;
  the wrist goals 480 x 480 from ``lab_tools/wrist_goals.py`` for checkpoints trained with wrist goal views);
* ``task_id``: the task's ``train_task_index``.

The server answers ``{"action": (1, 23)}``; the row is returned to the executor unchanged (the dataset's ``action``
layout). The episode context (goal image, task id, horizon, server URL) is set by the driver in :data:`CONTEXT` before
each episode; the callable returns ``None`` (ends the stream) at the horizon or, when ``stop_on_success`` is set, once the
task reports ``official_success``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import msgpack
import numpy as np

CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
OBS_KEYS = {cam: f"robot_r1::robot_r1:{cam}:Camera:0::rgb" for cam in CAMERAS}
GOAL_KEY = "goal::robot_r1::robot_r1:zed_link:Camera:0::rgb"
PROPRIO_KEY = "robot_r1::proprio"

#: set by the driver before every episode
CONTEXT: Dict[str, Any] = {}
#: the policy object of the current episode (its websocket is closed when the next episode begins)
ACTIVE: Optional["ActPolicy"] = None


# -- msgpack-numpy (identical to b1k_server.py) --------------------------------------------------------------------

def pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value)}")


def unpack_array(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def packb(value):
    return msgpack.packb(value, default=pack_array, use_bin_type=True)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=unpack_array, raw=False, strict_map_key=False)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.ascontiguousarray(np.asarray(x))


def task_info(info) -> Dict[str, Any]:
    """The task's ``ispatialgym`` info block of an ``env.step`` info dict. OmniGibson's task nests it under ``info["done"]``
    (``BaseTask.step``: ``{"reward": ..., "done": {..., "ispatialgym": ...}}``); the executor's ``isg`` property reads it
    the same way. (Until 2026-10-07 the bridge read ``info["ispatialgym"]`` - always absent - so stop-on-success never
    fired; no policy episode had reached the goal, so no recorded result was affected.)"""
    info = info or {}
    done = info.get("done") if isinstance(info.get("done"), dict) else None
    return (done or {}).get("ispatialgym") or info.get("ispatialgym") or {}


def record_step_error(ctx: Dict[str, Any], isg: Dict[str, Any]) -> None:
    """Append the current [position_error_m, orientation_error_rad] of the first pose target (relocalization: the head camera
    vs the goal camera pose) to ``ctx["step_errors"]``; NaNs when the info has no pose report (first call before any step)."""
    errs = ((isg.get("errors") or {}).get("pose_errors") or {}) if isg else {}
    rep = next(iter(errs.values()), None) if errs else None
    if rep is None:
        ctx.setdefault("step_errors", []).append([float("nan"), float("nan")])
        return
    pe, oe = rep.get("position_error"), rep.get("orientation_error")
    ctx.setdefault("step_errors", []).append([float("nan") if pe is None else float(pe), float("nan") if oe is None else float(oe)])


def sensor_key(robot_obs: Dict[str, Any], camera: str) -> str:
    suffix = f"{camera}:Camera:0"
    for k in robot_obs:
        if k.endswith(suffix):
            return k
    raise KeyError(f"no sensor {suffix!r} in {sorted(robot_obs)}")


class ActPolicy:
    """``policy(obs, info) -> action | None``; a websocket session per episode (``CONTEXT`` identity change = new episode).

    ``ctx``: an explicit episode context instead of the module's :data:`CONTEXT` (the vectorized evaluator runs one policy
    per environment slot, each with its own context and websocket session; such a policy never closes another's socket)."""

    def __init__(self, ctx: Optional[dict] = None):
        self.ctx = ctx
        self.ws = None
        self.metadata: Optional[dict] = None
        self.episode_ctx: Optional[dict] = None
        self.steps = 0
        self.robot_name: Optional[str] = None
        self.keys: Dict[str, str] = {}
        self.request_s: list = []
        self.stop_reason: Optional[str] = None
        self.goal_msg: Dict[str, np.ndarray] = {}

    # -- connection ------------------------------------------------------------------------------------------------

    def connect(self, url: str, timeout: float = 60.0):
        from websockets.sync.client import connect

        self.close()
        self.ws = connect(url, max_size=64 * 1024 * 1024, compression=None, open_timeout=timeout, close_timeout=5)
        self.metadata = unpackb(self.ws.recv(timeout=timeout))
        return self.metadata

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        self.ws = None

    def begin_episode(self, ctx: dict):
        global ACTIVE
        if self.ctx is None:
            if ACTIVE is not None and ACTIVE is not self:
                ACTIVE.close()
            ACTIVE = self
        self.episode_ctx = ctx
        self.steps = 0
        self.request_s = []
        self.stop_reason = None
        self.keys = {}
        self.robot_name = None
        meta = self.connect(ctx["server"])
        ctx["server_metadata"] = meta
        ah = meta.get("action_horizon")
        if ctx.get("expect_action_horizon") is not None and ah != ctx["expect_action_horizon"]:
            raise RuntimeError(f"server action_horizon {ah} != expected {ctx['expect_action_horizon']}")
        # goal images per view the checkpoint was trained with (server metadata goal_views; zed_link only for most runs, the
        # wrist cameras too for the "wrist" runs): ctx["goal_rgbs"] = {view: uint8 HWC}, ctx["goal_rgb"] = the zed image (legacy)
        views = list(meta.get("goal_views") or ["zed_link"])
        rgbs = dict(ctx.get("goal_rgbs") or {})
        if "zed_link" not in rgbs and ctx.get("goal_rgb") is not None:
            rgbs["zed_link"] = ctx["goal_rgb"]
        missing = [v for v in views if rgbs.get(v) is None]
        if missing:
            raise RuntimeError(f"the checkpoint needs goal images for views {views}; none for {missing} (lab_tools/wrist_goals.py makes the wrist ones)")
        keys = meta.get("goal_observation_keys") or [f"goal::robot_r1::robot_r1:{v}:Camera:0::rgb" for v in views]
        self.goal_msg = {k: np.ascontiguousarray(rgbs[v])[None, ..., :3] for k, v in zip(keys, views)}
        ctx["goal_views_sent"] = views
        import hashlib

        ctx["goal_sha256"] = {v: hashlib.sha256(self.goal_msg[k].tobytes()).hexdigest()[:16] for k, v in zip(keys, views)}  # what this session sends (per-slot isolation evidence)
        ctx["task_id_sent"] = int(ctx["task_id"])
        ctx["step_errors"] = []  # per step: [position_error_m, orientation_error_rad] of the task's pose target (camera vs goal)

    def end_stream(self, reason: str):
        ctx = self.episode_ctx or {}
        self.stop_reason = reason
        ctx["stop_reason"] = reason
        ctx["steps"] = self.steps
        if self.request_s:
            ctx["request_s_mean"] = float(np.mean(self.request_s))
            ctx["request_s_max"] = float(np.max(self.request_s))
        self.close()
        return None

    # -- the callable ------------------------------------------------------------------------------------------------

    def __call__(self, obs, info):
        ctx = self.ctx if self.ctx is not None else CONTEXT
        if not ctx:
            raise RuntimeError("act_policy.CONTEXT is empty: the driver must set it before the episode")
        if ctx is not self.episode_ctx:
            self.begin_episode(ctx)
        if self.steps >= int(ctx["horizon"]):
            return self.end_stream("horizon")
        isg = task_info(info)
        record_step_error(ctx, isg)
        if ctx.get("stop_on_success", True) and isg.get("official_success"):
            return self.end_stream("success")
        if self.robot_name is None:
            self.robot_name = ctx.get("robot") or next(iter(k for k in obs if isinstance(obs[k], dict) and "proprio" in obs[k]))
            robot_obs = obs[self.robot_name]
            self.keys = {cam: sensor_key(robot_obs, cam) for cam in CAMERAS}
        robot_obs = obs[self.robot_name]
        proprio = _np(robot_obs["proprio"]).astype(np.float32).reshape(-1)
        if proprio.shape[0] != 61:
            raise RuntimeError(f"proprio has {proprio.shape[0]} values, the server needs 61")
        msg = {PROPRIO_KEY: proprio[None], "task_id": np.int64(ctx["task_id"]), **self.goal_msg}
        for cam in CAMERAS:
            img = _np(robot_obs[self.keys[cam]]["rgb"])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            msg[OBS_KEYS[cam]] = img[None, ..., :3]
        t0 = time.time()
        self.ws.send(packb(msg))
        reply = unpackb(self.ws.recv(timeout=120))
        self.request_s.append(time.time() - t0)
        action = np.asarray(reply["action"], dtype=np.float32)
        if action.ndim == 2:
            action = action[0]
        if action.shape != (23,) or not np.isfinite(action).all():
            raise RuntimeError(f"bad action from the server: shape {action.shape}")
        self.steps += 1
        ctx["steps"] = self.steps
        ctx.setdefault("actions", []).append(action.astype(np.float32))
        ctx["request_s_mean"] = float(np.mean(self.request_s))
        ctx["request_s_max"] = float(np.max(self.request_s))
        if self.steps >= int(ctx["horizon"]):
            ctx["stop_reason"] = "horizon"  # the executor's stream has exactly `horizon` slots: no further call comes
        return action
