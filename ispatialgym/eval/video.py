"""Rollout videos: one frame per control step for each requested view, each to ``<out>/<view>_rgb.mp4``.

Views: ``head`` (head camera), ``external`` (BEHAVIOR's replay camera over the robot's shoulder), ``left_wrist`` /
``right_wrist`` (wrist cameras) and ``tiled`` (left wrist | head | right wrist in 480 px tiles over the external camera).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ispatialgym.robot import r1pro

log = logging.getLogger(__name__)

VIEWS = ("tiled", "head", "external", "left_wrist", "right_wrist")
TILE_PX = 480
VIDEO_CRF = 20


def to_uint8_rgb(img) -> np.ndarray:
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def save_png(path: str, img) -> None:
    from PIL import Image

    Image.fromarray(to_uint8_rgb(img)).save(path)


def _resized(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(frame)).resize((int(width), int(height)), Image.BILINEAR))


def tiled_frame(left_wrist, head, right_wrist, external, tile: int = TILE_PX) -> np.ndarray:
    """Left wrist | head | right wrist (``tile`` px squares) over the external camera; a missing view is black."""

    def fit(frame, width, height):
        if frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return _resized(np.asarray(frame)[..., :3], width, height)

    top = np.concatenate([fit(left_wrist, tile, tile), fit(head, tile, tile), fit(right_wrist, tile, tile)], axis=1)
    return np.concatenate([top, fit(external, 3 * tile, 2 * tile)], axis=0)


class VideoRecorder:
    def __init__(
        self,
        out_dir: str,
        views: Sequence[str],
        robot_name: str,
        cam_key: str,
        external_camera: str | None = None,
        fps: int = 30,
        width: int | None = None,
    ):
        unknown = [v for v in views if v not in VIEWS]
        if unknown:
            raise ValueError(f"unknown record views {unknown}; choose from {', '.join(VIEWS)}")
        self.paths = {v: os.path.join(out_dir, f"{v}_rgb.mp4") for v in dict.fromkeys(views)}
        self.robot_name, self.cam_key, self.external_camera = robot_name, cam_key, external_camera
        self.fps, self.width = int(fps), width
        self.errors: dict[str, str] = {}
        self._writers: dict[str, Any] = {}
        self._wrist_keys: dict[str, str | None] = {}

    def _robot_obs(self, obs) -> dict[str, Any]:
        return obs.get(self.robot_name, {}) if isinstance(obs, dict) else {}

    @staticmethod
    def _rgb(sensor_obs) -> np.ndarray | None:
        return (
            to_uint8_rgb(sensor_obs["rgb"])
            if isinstance(sensor_obs, dict) and sensor_obs.get("rgb") is not None
            else None
        )

    def start(self, obs) -> None:
        import imageio.v2 as imageio

        robot_obs = self._robot_obs(obs)
        for arm in r1pro.ARMS:
            self._wrist_keys[arm] = next(
                (k for k in robot_obs if f"{arm}_realsense_link" in k and self._rgb(robot_obs[k]) is not None), None
            )
        sources = {
            "head": self.cam_key in robot_obs,
            "external": bool(self.external_camera),
            "left_wrist": self._wrist_keys["left"] is not None,
            "right_wrist": self._wrist_keys["right"] is not None,
        }
        for view, path in self.paths.items():
            if not sources.get(view, True):
                self.errors[view] = "no camera"
                continue
            if view == "tiled" and not all(sources.values()):
                log.warning(
                    "tiled video: no %s camera, its tile stays black",
                    ", ".join(v for v, ok in sources.items() if not ok),
                )
            try:
                self._writers[view] = imageio.get_writer(
                    path,
                    fps=self.fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=8,
                    ffmpeg_params=["-crf", str(VIDEO_CRF), "-preset", "veryfast"],
                )
            except Exception as e:  # noqa: BLE001 - recording without this view
                self.errors[view] = repr(e)
        for view, error in self.errors.items():
            log.warning("%s video not recorded: %s", view, error)

    def record(self, obs) -> None:
        if not self._writers:
            return
        robot_obs = self._robot_obs(obs)
        head = self._rgb(robot_obs.get(self.cam_key))
        wrists = {arm: self._rgb(robot_obs.get(key)) if key else None for arm, key in self._wrist_keys.items()}
        external = None
        if self.external_camera and isinstance(obs, dict):
            external = self._rgb((obs.get("external") or {}).get(self.external_camera))
        frames = {"head": head, "external": external, "left_wrist": wrists["left"], "right_wrist": wrists["right"]}
        for view in list(self._writers):
            frame = tiled_frame(wrists["left"], head, wrists["right"], external) if view == "tiled" else frames[view]
            if frame is not None:
                self._write(view, frame)

    def _write(self, view: str, frame: np.ndarray) -> None:
        try:
            if self.width and frame.shape[1] != self.width:
                frame = _resized(frame, self.width, int(round(frame.shape[0] * self.width / frame.shape[1])))
            self._writers[view].append_data(frame)
        except Exception as e:  # noqa: BLE001 - one broken view does not end the episode
            self.errors[view] = repr(e)
            log.warning("%s video frame dropped (%r); closing that writer", view, e)
            self._close(view)

    def _close(self, view: str) -> None:
        writer = self._writers.pop(view, None)
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> dict[str, str]:
        """Close every writer; returns the complete videos (view -> path)."""
        for view in list(self._writers):
            self._close(view)
        return {v: p for v, p in self.paths.items() if v not in self.errors and os.path.exists(p)}
