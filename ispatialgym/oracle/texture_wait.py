"""Texture loading wait before the first capture (owner 2026-10-11 11:46).

Kit streams textures in after a scene load (lower mip levels first, the full resolution over the next frames): the starting
images of the s1_3tasks demos and replays were captured with a blurry floor (init_rgb Laplacian variance ~1-5 against ~40-80
at the end of the same camera's episode), and the blur ran into the first seconds of the videos. Before ``run_isg`` captures
``init_rgb.png`` and the recorder's first frame, :func:`wait_for_textures` renders - ``og.sim.render``, Isaac's render-only
update with ``/app/player/playSimulations`` off, so physics never steps - until

* the USD context has no file loads in flight (``UsdContext.get_stage_loading_status``: files loaded / total both 0), and
* the sharpness (Laplacian variance) of every recorded camera image, the whole frame and its bottom-centre crop, has not risen
  by more than :data:`REL_TOL` for :data:`STABLE_FRAMES` consecutive renders (after at least :data:`MIN_FRAMES`),

capped at :data:`MAX_S` (``ISG_TEXTURE_WAIT_MAX_S``). The status (``complete``, why, renders, seconds, the loading status and the
sharpness at the start and at the capture) goes to meta.json ``texture_wait`` and the log.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("ispatialgym.oracle.texture_wait")

MAX_S = float(os.environ.get("ISG_TEXTURE_WAIT_MAX_S", "240"))
MIN_FRAMES = 5
STABLE_FRAMES = 12
REL_TOL = 0.01
CROP_PX = 240


def _gray(rgb) -> np.ndarray:
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    a = np.asarray(rgb)
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
    return a[h - p:h, x0:x0 + p]


def stage_loading() -> Optional[Tuple[int, int]]:
    """``(files loaded, total files)`` of the USD context's loading status (both 0 when nothing is loading); None without Kit."""
    try:
        import omni.usd

        _, loaded, total = omni.usd.get_context().get_stage_loading_status()
        return int(loaded), int(total)
    except Exception:  # noqa: BLE001 - no Kit (tests) or no stage
        return None


def camera_images(obs: Dict[str, Any], robot_name: str, external: Optional[str] = None) -> Dict[str, Any]:
    """Every rgb image the recorder can write: the robot's cameras (``obs[robot][sensor]["rgb"]``) and the external camera."""
    out: Dict[str, Any] = {}
    robs = obs.get(robot_name, {}) if isinstance(obs, dict) else {}
    for k, v in (robs.items() if isinstance(robs, dict) else []):
        if isinstance(v, dict) and v.get("rgb") is not None:
            out[k] = v["rgb"]
    ext = obs.get("external") if isinstance(obs, dict) else None
    if external and isinstance(ext, dict) and isinstance(ext.get(external), dict) and ext[external].get("rgb") is not None:
        out[f"external:{external}"] = ext[external]["rgb"]
    return out


def wait_for_textures(get_obs: Callable[[], Dict[str, Any]], render: Callable[[], None], robot_name: str, external: Optional[str] = None,
                      max_s: float = MAX_S) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Render-only frames until textures are loaded (module docstring). Returns ``(status, obs)``: ``obs`` = the observation
    of the last render (None when nothing was rendered)."""
    t0 = time.time()
    obs = get_obs()
    imgs = camera_images(obs, robot_name, external)
    first = {k: (round(sharpness(v), 2), round(sharpness(bottom_centre(v)), 2)) for k, v in imgs.items()}
    best = {k: list(s) for k, s in first.items()}
    since = {k: 0 for k in imgs}
    loading0 = stage_loading()
    n, complete, why = 0, False, "timeout"
    loading = loading0
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
            why = f"timeout after {max_s:.0f} s ({'files loading ' + str(loading) if busy else 'sharpness still rising: ' + str([k for k, c in since.items() if c < STABLE_FRAMES])})"
            break
    final = {k: (round(sharpness(v), 2), round(sharpness(bottom_centre(v)), 2)) for k, v in camera_images(obs, robot_name, external).items()}
    status = {"complete": complete, "why": why, "renders": n, "wait_s": round(time.time() - t0, 2), "stage_loading_start": loading0,
              "stage_loading_capture": loading, "sharpness_start": first, "sharpness_capture": final,
              "rule": {"min_frames": MIN_FRAMES, "stable_frames": STABLE_FRAMES, "rel_tol": REL_TOL, "crop_px": CROP_PX, "max_s": max_s}}
    log.info("texture wait: complete=%s (%s) after %d renders, %.1f s; stage loading %s -> %s; sharpness (frame, floor crop) %s -> %s", complete, why, n,
             status["wait_s"], loading0, loading, first, final)
    return status, obs
