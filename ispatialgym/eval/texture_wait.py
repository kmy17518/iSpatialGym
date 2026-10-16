"""Wait for streamed textures before the first observation.

Kit streams textures in after a scene load, so the first frames are blurry. :func:`wait_for_textures` renders (physics
never steps) until the USD context has no file loads in flight and the sharpness (Laplacian variance) of every camera
image, whole frame and bottom-centre crop, has stopped rising for :data:`STABLE_FRAMES` renders, or :data:`MAX_S` passed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MAX_S = 240.0
MIN_FRAMES = 5
STABLE_FRAMES = 12
REL_TOL = 0.01
CROP_PX = 240


def _gray(rgb) -> np.ndarray:
    a = np.asarray(rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else rgb)
    if a.ndim == 3 and a.shape[-1] >= 3:
        a = a[..., :3]
    a = a.astype(np.float32)
    if a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a.mean(axis=-1) if a.ndim == 3 else a


def sharpness(rgb) -> float:
    """Variance of the 4-neighbour Laplacian of the grey image (0-255 scale)."""
    g = _gray(rgb)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = 4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def bottom_centre(rgb, px: int = CROP_PX):
    """The ``px`` x ``px`` crop at the bottom centre of the image (the floor in front of the robot)."""
    a = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    h, w = a.shape[0], a.shape[1]
    p = min(px, h, w)
    x0 = max(0, (w - p) // 2)
    return a[h - p : h, x0 : x0 + p]


def stage_loading() -> tuple[int, int] | None:
    """``(files loaded, total files)`` of the USD context's loading status, None without Kit."""
    try:
        import omni.usd

        _, loaded, total = omni.usd.get_context().get_stage_loading_status()
        return int(loaded), int(total)
    except Exception:  # noqa: BLE001 - no Kit or no stage
        return None


def camera_images(obs: dict[str, Any], robot_name: str, external: str | None = None) -> dict[str, Any]:
    """Every rgb image of the robot's cameras and of the external camera ``external``."""
    out: dict[str, Any] = {}
    robot_obs = obs.get(robot_name, {}) if isinstance(obs, dict) else {}
    for k, v in robot_obs.items() if isinstance(robot_obs, dict) else ():
        if isinstance(v, dict) and v.get("rgb") is not None:
            out[k] = v["rgb"]
    ext = obs.get("external") if isinstance(obs, dict) else None
    if (
        external
        and isinstance(ext, dict)
        and isinstance(ext.get(external), dict)
        and ext[external].get("rgb") is not None
    ):
        out[f"external:{external}"] = ext[external]["rgb"]
    return out


def wait_for_textures(
    get_obs: Callable[[], dict[str, Any]],
    render: Callable[[], None],
    robot_name: str,
    external: str | None = None,
    max_s: float = MAX_S,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render until the textures are loaded; returns ``(status, obs of the last render)``."""
    t0 = time.time()
    images = camera_images(get_obs(), robot_name, external)
    best = {k: [sharpness(v), sharpness(bottom_centre(v))] for k, v in images.items()}
    since = {k: 0 for k in images}
    n, complete, why = 0, False, "timeout"
    while True:
        render()
        n += 1
        obs = get_obs()
        for k, v in camera_images(obs, robot_name, external).items():
            s = (sharpness(v), sharpness(bottom_centre(v)))
            b = best.setdefault(k, [0.0, 0.0])
            rose = s[0] > b[0] * (1.0 + REL_TOL) or s[1] > b[1] * (1.0 + REL_TOL)
            b[0], b[1] = max(b[0], s[0]), max(b[1], s[1])
            since[k] = 0 if rose else since.get(k, 0) + 1
        loading = stage_loading()
        busy = loading is not None and (loading[0] > 0 or loading[1] > 0)
        if n >= MIN_FRAMES and not busy and since and all(c >= STABLE_FRAMES for c in since.values()):
            complete, why = True, "stage loaded, sharpness stable"
            break
        if time.time() - t0 > max_s:
            break
    status = {"complete": complete, "why": why, "renders": n, "wait_s": round(time.time() - t0, 2)}
    log.info("texture wait: %s after %d renders, %.1f s", why, n, status["wait_s"])
    return status, obs
