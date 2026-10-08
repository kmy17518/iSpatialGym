"""Linear depth of a robot camera without touching its observations.

A camera without a ``depth_linear`` modality gets one extra depth annotator on its render product, attached the first
time it is needed and kept for the process; it is never registered as a modality and never detached. Adding a modality
changes the observations ``Environment.reset`` checks against the observation space, and detaching an annotator again
left the robot's articulation view invalid (``get_joint_positions`` -> None at the next instance of a reused world). A
camera that records depth anyway is read through its own annotator.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch as th


def depth_annotator(sensor):
    """The annotator a depth read of ``sensor`` uses (its own ``depth_linear`` one, else the kept extra one)."""
    own = (getattr(sensor, "_annotators", None) or {}).get("depth_linear")
    if own is not None:
        return own
    ann = getattr(sensor, "_isg_depth_annotator", None)
    if ann is None:
        import omnigibson as og
        import omnigibson.lazy as lazy

        with og.sim.editing_usd():
            ann = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(sensor.raw_sensor_types["depth_linear"])
            ann.attach([sensor.render_product])
        sensor._isg_depth_annotator = ann
    return ann


def read_depth(sensor) -> th.Tensor:
    """The annotator's current buffer as a float tensor on the CPU (empty while it has no data yet)."""
    import omnigibson as og
    import omnigibson.lazy as lazy

    raw = depth_annotator(sensor).get_data(device=og.sim.device)
    data = raw["data"] if isinstance(raw, dict) else raw
    t = th.from_numpy(np.asarray(data)) if og.sim.device == "cpu" else lazy.warp.to_torch(data)
    return t.float().cpu().clone()


def settled_depth(sensor, render: Callable[[], None], min_renders: int = 4, max_renders: int = 24, read: Callable = read_depth) -> np.ndarray:
    """The camera's linear depth ``(H, W)`` in m, rendered until two consecutive reads of a non-empty buffer agree: annotator
    data lags the scene by one to two frames (``SimWorld.seg_instance``), and a freshly attached annotator delivers nothing
    for its first frames. ``RuntimeError`` when no frame came within ``max_renders``."""
    prev = None
    cur = None
    for n in range(1, int(max_renders) + 1):
        render()
        cur = read(sensor)
        if n >= min_renders and cur.numel() > 0 and prev is not None and cur.shape == prev.shape and th.equal(cur, prev):
            break
        prev = cur
    if cur is None or cur.numel() == 0:
        raise RuntimeError(f"no depth from {getattr(sensor, 'name', sensor)} after {max_renders} frames")
    return cur.reshape(int(sensor.image_height), int(sensor.image_width)).numpy()
