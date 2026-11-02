"""Videos of the vectorized evaluator (``run_isg_vec``): per slot a BEHAVIOR-style tiled frame (left wrist | head | right wrist
over the external camera, every tile ``tile_px`` wide) streamed into an H.264 file as the episode runs, and per batch one grid
video of every slot's tile (for inspection), both encoded on a background thread so the stepping thread only composes frames.

The single-env :class:`~ispatialgym.oracle.recorder.Recorder` keeps every frame of a view in memory (JPEG) until the stream
commits and encodes the video at the end; here the frames go to the encoder every step, which bounds memory for N slots and
removes the end-of-episode stall.
"""

from __future__ import annotations

import math
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def to_uint8_rgb(img) -> Optional[np.ndarray]:
    """A camera image (torch / numpy, HWC rgb or rgba, uint8 or float) as a contiguous uint8 HWC rgb array."""
    if img is None:
        return None
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] >= 3:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6 else a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    import cv2

    interp = cv2.INTER_AREA if (frame.shape[1] > width or frame.shape[0] > height) else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)


def fit(frame: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    """``frame`` scaled to fit ``width x height`` (aspect kept, letterboxed in black); a black tile without one."""
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None:
        return out
    h, w = frame.shape[:2]
    s = min(width / w, height / h)
    nw, nh = max(2, int(round(w * s)) // 2 * 2), max(2, int(round(h * s)) // 2 * 2)
    r = _resize(frame, nw, nh)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def tiled_frame(left: Optional[np.ndarray], head: Optional[np.ndarray], right: Optional[np.ndarray], external: Optional[np.ndarray],
                tile: int = 240, external_rows: Optional[int] = None) -> np.ndarray:
    """left wrist | head | right wrist (``tile`` px squares) over the external camera (``3 tile`` wide, ``external_rows`` high;
    default ``2 tile``); without an external camera the top row alone."""
    top = np.concatenate([fit(left, tile, tile), fit(head, tile, tile), fit(right, tile, tile)], axis=1)
    if external is None and external_rows is None:
        return top
    rows = int(external_rows) if external_rows is not None else 2 * tile
    return np.concatenate([top, fit(external, 3 * tile, rows)], axis=0)


def label(frame: np.ndarray, text: str, org: Tuple[int, int] = (6, 18), scale: float = 0.5) -> np.ndarray:
    try:
        import cv2

        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:  # noqa: BLE001
        pass
    return frame


def grid_frame(tiles: Sequence[Optional[np.ndarray]], max_width: int = 1920, tile_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """The slots' tiles in a near-square grid, scaled so the grid is at most ``max_width`` wide (an even size)."""
    n = len(tiles)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    ref = next((t for t in tiles if t is not None), None)
    th_, tw = (ref.shape[0], ref.shape[1]) if tile_hw is None and ref is not None else (tile_hw or (240, 720))
    s = min(1.0, max_width / (cols * tw))
    w, h = max(2, int(tw * s) // 2 * 2), max(2, int(th_ * s) // 2 * 2)
    out = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, t in enumerate(tiles):
        if t is None:
            continue
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = _resize(t, w, h)
    return out


class _Encoder:
    """One H.264 file (PyAV / libx264) fed frame by frame; frames must share one size (even dimensions)."""

    def __init__(self, path: str, fps: int, crf: int = 23, preset: str = "veryfast"):
        import av

        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=int(fps))
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(int(crf)), "preset": preset, "tune": "zerolatency"}
        self._opened = False
        self.n = 0
        self.error: Optional[str] = None

    def write(self, frame: np.ndarray) -> None:
        import av

        if self.error is not None:
            return
        h, w = frame.shape[:2]
        if not self._opened:
            self.stream.width, self.stream.height = int(w) // 2 * 2, int(h) // 2 * 2
            self._opened = True
        if frame.shape[1] != self.stream.width or frame.shape[0] != self.stream.height:
            frame = _resize(frame, self.stream.width, self.stream.height)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in self.stream.encode(vf):
            self.container.mux(packet)
        self.n += 1

    def close(self) -> None:
        try:
            if self._opened:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            self.container.close()
        except Exception as e:  # noqa: BLE001
            self.error = self.error or repr(e)


class VideoSink:
    """Named encoders written from one background thread in submission order (``put(name, frame)``), closed by ``close()``.
    ``closed_names`` lists what was written and how many frames."""

    def __init__(self, fps: int, crf: int = 23, preset: str = "veryfast", max_pending: int = 256):
        self.fps, self.crf, self.preset = int(fps), int(crf), preset
        self._paths: Dict[str, str] = {}
        self._enc: Dict[str, _Encoder] = {}
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_pending)
        self._errors: Dict[str, str] = {}
        # lever S3-recording (2026-10-27): slots registered for composition on this thread (tile size, external rows) and their last tiles
        self._tile_cfg: Dict[str, Tuple[int, Optional[int]]] = {}
        self._last_tiles: Dict[str, np.ndarray] = {}
        self.max_pending = 0
        self._thread = threading.Thread(target=self._run, name="isg-vec-video", daemon=True)
        self._thread.start()

    def open(self, name: str, path: str) -> None:
        self._paths[name] = path

    def _enqueue(self, job: tuple) -> None:
        self._q.put(job)
        n = self._q.qsize()
        if n > self.max_pending:
            self.max_pending = n

    def put(self, name: str, frame: np.ndarray) -> None:
        if name in self._paths:
            self._enqueue((name, frame))

    # -- composition on the sink's thread --------------------------------------------------------------------------
    def register_tile(self, slot: str, tile: int, external_rows: Optional[int]) -> None:
        """Slot ``slot`` (``"s<idx>"``) will send raw frames (:meth:`put_raw`); its tile is composed here."""
        self._tile_cfg[slot] = (int(tile), external_rows)

    def put_raw(self, slot: str, frames: Dict[str, Optional[np.ndarray]], step_text: str, views: Sequence[str]) -> None:
        """This step's raw camera frames of ``slot`` (uint8 HWC host arrays owned by the caller no longer): the thread composes the
        tile (left | head | right over external, labelled) and writes the slot's videos."""
        self._enqueue(("__raw__", slot, frames, step_text, tuple(views)))

    def put_grid(self, name: str, slot_order: Sequence[str], done_labels: Dict[int, str], max_width: int = 1920) -> None:
        """One grid frame of the registered slots' last tiles (in ``slot_order``), ``done_labels`` (slot idx -> text) stamped."""
        if name in self._paths:
            self._enqueue(("__grid__", name, tuple(slot_order), dict(done_labels), int(max_width)))

    def _write(self, name: str, frame: np.ndarray) -> None:
        enc = self._enc.get(name)
        if enc is None:
            if name not in self._paths:
                return
            try:
                enc = self._enc[name] = _Encoder(self._paths[name], self.fps, self.crf, self.preset)
            except Exception as e:  # noqa: BLE001
                self._errors[name] = repr(e)
                self._paths.pop(name, None)
                return
        enc.write(frame)

    def _run(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job is None:
                    return
                name, frame = job[0], job[1]
                if name == "__close__":
                    enc = self._enc.pop(frame, None)
                    if enc is not None:
                        enc.close()
                        if enc.error:
                            self._errors[frame] = enc.error
                    continue
                if name == "__raw__":
                    _, slot, fr, step_text, views = job
                    fr = {k: to_uint8_rgb(v) for k, v in fr.items()}  # RGBA host / device arrays -> uint8 RGB here, off the stepping thread
                    tile_px, ext_rows = self._tile_cfg.get(slot, (240, None))
                    tile = None
                    for v in views:
                        if v == "tiled":
                            tile = tiled_frame(fr.get("left"), fr.get("head"), fr.get("right"), fr.get("external"), tile=tile_px,
                                               external_rows=ext_rows if fr.get("external") is not None else None)
                            if step_text:
                                label(tile, step_text)
                            self._write(f"{slot}:tiled", tile)
                        elif v == "head" and fr.get("head") is not None:
                            self._write(f"{slot}:head", fr["head"])
                        elif v == "external" and fr.get("external") is not None:
                            self._write(f"{slot}:external", fr["external"])
                        elif v in ("left_wrist", "right_wrist") and fr.get(v.split("_")[0]) is not None:
                            self._write(f"{slot}:{v}", fr[v.split("_")[0]])
                    if tile is None and fr.get("head") is not None:
                        tile = tiled_frame(fr.get("left"), fr.get("head"), fr.get("right"), fr.get("external"), tile=tile_px,
                                           external_rows=ext_rows if fr.get("external") is not None else None)
                    if tile is not None:
                        self._last_tiles[slot] = tile
                    continue
                if name == "__grid__":
                    _, gname, order, done_labels, max_width = job
                    tiles = []
                    for k, slot in enumerate(order):
                        t = self._last_tiles.get(slot)
                        idx = int(slot[1:]) if slot[1:].isdigit() else k
                        if t is not None and idx in done_labels:
                            t = t.copy()
                            label(t, done_labels[idx], org=(6, t.shape[0] - 10))
                        tiles.append(t)
                    self._write(gname, grid_frame(tiles, max_width=max_width))
                    continue
                self._write(name, frame)
            except Exception as e:  # noqa: BLE001
                try:
                    self._errors[str(job[1]) if job[0] in ("__raw__", "__grid__") else job[0]] = repr(e)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._q.task_done()

    def close_name(self, name: str) -> None:
        """Finish one video (its frames queued so far are written first)."""
        if name in self._paths:
            self._q.put(("__close__", name))

    def close(self) -> Dict[str, Any]:
        for name in list(self._paths):
            self.close_name(name)
        self._q.put(None)
        self._thread.join()
        return {"paths": dict(self._paths), "errors": dict(self._errors), "max_pending": self.max_pending}

    @property
    def pending(self) -> int:
        return self._q.qsize()


__all__ = ["VideoSink", "tiled_frame", "grid_frame", "fit", "label", "to_uint8_rgb"]
