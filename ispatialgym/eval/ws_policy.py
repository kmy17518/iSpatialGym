"""Batched websocket policy client of the vectorized evaluator (``run_isg_vec --mode policy``).

One websocket session serves every slot of a batch: each control step the slots' observations go out as **one** batched
request - ``robot_r1::proprio`` ``(B, 61)``, the three camera keys ``(B, H, W, 3)`` uint8, ``task_id`` ``(B,)``, the goal
images ``goal::<camera key>`` ``(B, Hg, Wg, 3)`` - and the reply's ``action`` ``(B, 23)`` is split back per slot. The batch
is kept at a fixed size and order for the whole session (a slot that has finished keeps sending its last observation and
its action is dropped): the ACT server's temporal-aggregation history and the openpi wrapper's plan buffers are per batch
position and are reset by a change of batch size.

Protocols (BEHAVIOR's NumPy-MessagePack wire format, the server's metadata decides):

* ``act`` - ``b1k_server.py`` of kmy17518/act: a request every step; the server executes its chunk (``--action-horizon``) or
  aggregates overlapping chunks (``--temporal-agg``) itself.
* ``openpi`` - openpi's ``websocket_b1k_server.py`` (``serve_b1k.py``): a request carries ``__action_chunk_size__ = n`` and the
  reply an ``action_chunk`` ``(B, n, 23)`` that the client executes open loop before the next request (BEHAVIOR's
  ``WebsocketClientPolicy(action_chunk_size=n)``).

What is sent is what the single-env bridge sent (``lab_tools/act_policy.py``): the robot's raw 61-D proprioception as
OmniGibson returns it and the cameras' uint8 frames at their native size; the server does the training-time preprocessing.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

import msgpack
import numpy as np

CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
PROPRIO_KEY = "robot_r1::proprio"
OBS_KEYS = {cam: f"robot_r1::robot_r1:{cam}:Camera:0::rgb" for cam in CAMERAS}
GOAL_KEYS = {cam: f"goal::{key}" for cam, key in OBS_KEYS.items()}
ACTION_CHUNK_REQUEST_KEY = "__action_chunk_size__"
ACTION_DIM = 23


# -- msgpack-numpy (b1k_server.py / openpi's msgpack_numpy) --------------------------------------------------------------

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


def to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.ascontiguousarray(np.asarray(x))


def sensor_key(robot_obs: Dict[str, Any], camera: str) -> str:
    suffix = f"{camera}:Camera:0"
    for k in robot_obs:
        if k.endswith(suffix):
            return k
    raise KeyError(f"no sensor {suffix!r} in {sorted(robot_obs)}")


def load_rgb(path: str) -> np.ndarray:
    """A PNG / JPEG as uint8 HWC rgb (PyAV, as the servers load their fixed goal images)."""
    import av

    with av.open(path) as container:
        frame = next(container.decode(video=0))
        return np.ascontiguousarray(frame.to_ndarray(format="rgb24"))


def resize_quantize(images, size, device: str = "cuda"):
    """ACT's training-time image path for a batch of uint8 HWC frames: ``preprocess_image`` (float / 255, bilinear antialiased
    resize to ``size``) then ``quantize_image`` (round to the nearest uint8) - what the frame cache stored and the server
    turns back into ``uint8 / 255`` without resizing (its ``preprocess_image`` skips the resize at the training size). Computed
    in one batched torch call (on ``device``: CUDA's antialiased bilinear differs from the CPU kernel by at most one LSB on a
    negligible fraction of pixels). ``images``: list of uint8 HWC arrays / tensors of one size; returns uint8 (B, h, w, 3)."""
    import torch as th
    import torch.nn.functional as F

    ts = []
    for im in images:
        t = im if isinstance(im, th.Tensor) else th.from_numpy(np.ascontiguousarray(im))
        ts.append(t[..., :3])
    x = th.stack(ts).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False, antialias=True)
    q = th.round(x * 255).clamp_(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous()
    return q.cpu().numpy()


def detect_protocol(metadata: Dict[str, Any]) -> str:
    """``act`` for b1k_server.py's handshake (``protocol: behavior-numpy-msgpack`` / ``execution_mode``), else ``openpi``."""
    if metadata.get("protocol") == "behavior-numpy-msgpack" or "execution_mode" in metadata or "temporal_agg" in metadata:
        return "act"
    return "openpi"


class BatchedWSPolicy:
    """One websocket session for the slots of a batch (module docstring).

    ``slots``: the slot indices in batch order; ``task_ids[slot]``; ``goals[slot] = {camera: uint8 HWC}`` (the views the server
    asks for in its metadata ``goal_views``; a missing view raises); ``action_chunk``: openpi's chunk (default: the server's
    ``action_horizon`` metadata, else 16); ``robot``: the robot's name in the observation dicts."""

    def __init__(self, url: str, slots: Sequence[int], task_ids: Dict[int, int], goals: Dict[int, Dict[str, np.ndarray]], robot: str = "robot_r1",
                 protocol: str = "auto", action_chunk: Optional[int] = None, timeout: float = 120.0, open_timeout: float = 120.0,
                 client_image_size: Optional[Sequence[int]] = None, preprocess_device: str = "cuda"):
        """``client_image_size``: ACT only - the frames (and goals) are resized + quantized on the client with the training-time
        path (:func:`resize_quantize`) and sent at that size, so the server's CPU resize of N x 4 full-size images per step
        (~12 ms each) is replaced by one batched GPU op here; None sends the cameras' native frames."""
        self.url = url
        self.client_image_size = tuple(int(v) for v in client_image_size) if client_image_size else None
        self.preprocess_device = preprocess_device
        self.slots = [int(s) for s in slots]
        self.task_ids = {int(k): int(v) for k, v in task_ids.items()}
        self.robot = robot
        self.timeout = float(timeout)
        self.open_timeout = float(open_timeout)
        self.ws = None
        self.metadata: Dict[str, Any] = {}
        self.protocol = protocol
        self.action_chunk = action_chunk
        self.keys: Dict[str, str] = {}
        self.last_obs: Dict[int, Any] = {}
        self.request_s: List[float] = []
        self.n_requests = 0
        self._chunks: Dict[int, np.ndarray] = {}
        self._chunk_pos = 0
        self._goal_msg: Dict[str, np.ndarray] = {}
        self.goal_sha256: Dict[int, Dict[str, str]] = {}
        self._goals_in = {int(k): dict(v) for k, v in goals.items()}
        self.default_goal_views: Optional[List[str]] = None
        self.goal_views: List[str] = []

    # -- connection ------------------------------------------------------------------------------------------------

    def connect(self) -> Dict[str, Any]:
        from websockets.sync.client import connect

        self.close()
        self.ws = connect(self.url, max_size=None, compression=None, open_timeout=self.open_timeout, close_timeout=5, ping_interval=60, ping_timeout=300)
        self.metadata = unpackb(self.ws.recv(timeout=self.open_timeout))
        if self.protocol == "auto":
            self.protocol = detect_protocol(self.metadata)
        if self.protocol == "openpi" and self.action_chunk is None:
            self.action_chunk = int(self.metadata.get("action_horizon") or 16)
        if self.protocol != "act":
            self.client_image_size = None  # openpi's wrapper does its own resize_with_pad on the native frames
        self._build_goal_msg()
        self.ws.send(packb({"reset": True}))  # no acknowledgement (both servers)
        return self.metadata

    def _build_goal_msg(self) -> None:
        views = list(self.metadata.get("goal_views") or getattr(self, "default_goal_views", None) or ["zed_link"])
        if self.protocol == "openpi":
            # openpi's robot-config goal view names (goal_image_0 = the head camera) are mapped to our camera names in order
            views = ["zed_link", "left_realsense_link", "right_realsense_link"][: len(views)] if views and not all(v in CAMERAS for v in views) else views
        keys = self.metadata.get("goal_observation_keys") or [GOAL_KEYS[v] for v in views]
        self._goal_msg = {}
        self.goal_views = views
        for key, view in zip(keys, views):
            imgs = []
            for s in self.slots:
                g = self._goals_in.get(s, {}).get(view)
                if g is None:
                    raise RuntimeError(f"slot {s}: the server needs a goal image for view {view!r} (goal_views {views}); none given")
                g = np.ascontiguousarray(np.asarray(g)[..., :3])
                imgs.append(g)
                self.goal_sha256.setdefault(s, {})[view] = hashlib.sha256(g.tobytes()).hexdigest()[:16]
            shapes = {i.shape for i in imgs}
            if len(shapes) != 1:
                raise RuntimeError(f"goal images of view {view!r} differ in size across the slots: {sorted(shapes)}")
            self._goal_msg[key] = resize_quantize(imgs, self.client_image_size, self.preprocess_device) if self.client_image_size else np.stack(imgs)

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        self.ws = None

    # -- requests --------------------------------------------------------------------------------------------------

    def _message(self, obs_by_slot: Dict[int, Any]) -> Dict[str, Any]:
        for s in self.slots:
            if obs_by_slot.get(s) is not None:
                self.last_obs[s] = obs_by_slot[s]
        missing = [s for s in self.slots if s not in self.last_obs]
        if missing:
            raise RuntimeError(f"no observation yet for slots {missing}")
        first = self.last_obs[self.slots[0]][self.robot]
        if not self.keys:
            self.keys = {cam: sensor_key(first, cam) for cam in CAMERAS}
        proprio = np.stack([to_np(self.last_obs[s][self.robot]["proprio"]).astype(np.float32).reshape(-1) for s in self.slots])
        if proprio.shape[1] != 61:
            raise RuntimeError(f"proprio has {proprio.shape[1]} values, the server needs 61")
        msg: Dict[str, Any] = {PROPRIO_KEY: proprio, "task_id": np.asarray([self.task_ids[s] for s in self.slots], dtype=np.int64), **self._goal_msg}
        for cam in CAMERAS:
            raw = [self.last_obs[s][self.robot][self.keys[cam]]["rgb"] for s in self.slots]
            if self.client_image_size:
                msg[OBS_KEYS[cam]] = resize_quantize(raw, self.client_image_size, self.preprocess_device)
                continue
            imgs = []
            for img in raw:
                img = to_np(img)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                imgs.append(np.ascontiguousarray(img[..., :3]))
            msg[OBS_KEYS[cam]] = np.stack(imgs)
        return msg

    def _request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        self.ws.send(packb(msg))
        reply = self.ws.recv(timeout=self.timeout)
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error: {reply[:500]}")
        reply = unpackb(reply)
        self.request_s.append(time.time() - t0)
        self.n_requests += 1
        if not isinstance(reply, dict) or "action" not in reply:
            raise RuntimeError(f"policy server reply without 'action': {str(reply)[:200]}")
        return reply

    def act(self, obs_by_slot: Dict[int, Any]) -> Dict[int, np.ndarray]:
        """The next action of every slot in ``obs_by_slot`` (the others' observations are the last ones seen)."""
        B = len(self.slots)
        if self.protocol == "openpi":
            if not self._chunks or self._chunk_pos >= self.action_chunk:
                msg = self._message(obs_by_slot)
                msg[ACTION_CHUNK_REQUEST_KEY] = int(self.action_chunk)
                reply = self._request(msg)
                chunk = np.asarray(reply.get("action_chunk", np.asarray(reply["action"])[:, None, :]), dtype=np.float32)
                if chunk.shape != (B, int(self.action_chunk), ACTION_DIM):
                    raise RuntimeError(f"action_chunk shape {chunk.shape} != {(B, self.action_chunk, ACTION_DIM)}")
                self._chunks = {s: chunk[k] for k, s in enumerate(self.slots)}
                self._chunk_pos = 0
            out = {s: self._chunks[s][self._chunk_pos].copy() for s in obs_by_slot if s in self._chunks}
            self._chunk_pos += 1
            return out
        reply = self._request(self._message(obs_by_slot))
        action = np.asarray(reply["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None]
        if action.shape != (B, ACTION_DIM) or not np.isfinite(action).all():
            raise RuntimeError(f"bad action batch from the server: shape {action.shape}")
        return {s: action[k].copy() for k, s in enumerate(self.slots) if s in obs_by_slot}

    def stats(self) -> Dict[str, Any]:
        r = np.asarray(self.request_s, dtype=np.float64)
        return {"protocol": self.protocol, "n_requests": self.n_requests, "client_image_size": self.client_image_size, "preprocess_device": self.preprocess_device if self.client_image_size else None,
                "request_s_mean": float(r.mean()) if r.size else None,
                "request_s_max": float(r.max()) if r.size else None, "action_chunk": self.action_chunk if self.protocol == "openpi" else None,
                "goal_views_sent": list(getattr(self, "goal_views", [])), "batch_size": len(self.slots)}


__all__ = ["BatchedWSPolicy", "load_rgb", "detect_protocol", "CAMERAS", "OBS_KEYS", "GOAL_KEYS", "PROPRIO_KEY"]
