"""Off-thread frame handling for the :class:`~ispatialgym.oracle.recorder.Recorder` (``RecorderConfig.async_frames``).

Per step the recorder takes the head camera's rgb for the video and the LeRobot dataset. Synchronously that is, for each
sink, a device-to-host copy, an RGBA -> RGB copy and a JPEG encode (the pending-segment buffer), then at commit a JPEG
decode, a resize and the sink write, all on the thread that steps the simulator. Here:

* :meth:`FramePipeline.frame` copies the image off the renderer's buffer on the calling thread (the next render reuses
  it) and hands the conversion and the encode to a worker; the frame is taken once per step and shared by every sink.
* :meth:`FramePipeline.sink` runs committed writes on one background thread, in submission order;
  :meth:`FramePipeline.drain` waits for them (the recorder drains before closing its writers).

The bytes are the synchronous path's: the same ``to_uint8_rgb`` conversion and the same JPEG encoder settings, so the
decoded frames, and every file written from them, are identical.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np
import torch as th

log = logging.getLogger(__name__)


def _host_copy(img):
    """A private host copy of ``img`` (a device tensor, a host tensor or an array the renderer may overwrite)."""
    if isinstance(img, th.Tensor):
        return img.detach().to("cpu", copy=True)
    return np.array(img, copy=True)


class FramePipeline:
    def __init__(self, mode: str = "jpeg", workers: int = 2):
        self.mode = mode
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="isg-frame")
        self._jobs: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._sink = threading.Thread(target=self._run_sink, name="isg-frame-sink", daemon=True)
        self._sink.start()
        self._closed = False

    # ------------------------------------------------------------------ frames

    def frame(self, img) -> "PendingFrame":
        """A frame of ``img`` (an rgb(a) image, uint8 or float in [0, 1]) whose conversion and encode run on a worker."""
        from ispatialgym.oracle.recorder import to_uint8_rgb

        host = _host_copy(img)
        return PendingFrame(self._pool.submit(_encode, host, self.mode, to_uint8_rgb))

    # ------------------------------------------------------------------ sink

    def sink(self, fn: Callable[..., Any], *args, **kwargs) -> None:
        """Run ``fn(*args, **kwargs)`` on the sink thread, after every job submitted before it."""
        self._jobs.put((fn, args, kwargs))

    def drain(self) -> None:
        self._jobs.join()

    def _run_sink(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                fn, args, kwargs = job
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001  (the sink functions record their own errors; never kill the thread)
                    log.exception("frame sink job failed")
            finally:
                self._jobs.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.drain()
        self._jobs.put(None)
        self._sink.join(timeout=60)
        self._pool.shutdown(wait=True)


def _encode(host, mode: str, to_uint8_rgb):
    rgb = to_uint8_rgb(host)
    if mode == "jpeg":
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
        return rgb, buf.getvalue()
    return rgb, np.ascontiguousarray(rgb)


def _pending_frame_base():
    from ispatialgym.oracle.recorder import _Frame

    return _Frame


class PendingFrame(_pending_frame_base()):
    """A :class:`~ispatialgym.oracle.recorder._Frame` whose conversion / encode may still be running."""

    __slots__ = ("_future",)

    def __init__(self, future: Future):  # noqa: D107  (no super().__init__: the encode is the future's)
        self._future = future

    @property
    def data(self):
        return self._future.result()[1]

    @property
    def shape(self):
        return self._future.result()[0].shape

    def rgb(self) -> np.ndarray:
        """The uint8 rgb image before buffering (what the synchronous path keeps as ``last_rgb``)."""
        return self._future.result()[0]

    def array(self) -> np.ndarray:
        data = self.data
        if isinstance(data, bytes):
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return data
