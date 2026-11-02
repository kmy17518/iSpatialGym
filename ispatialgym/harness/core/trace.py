"""Per-episode trace: ``transcript.jsonl`` (every request summary, model output, tool call, motion and stop, images by
file reference), ``frames/`` (each image as sent to the model) and ``summary.json``."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from functools import lru_cache
from typing import Any, Dict, Optional

HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


@lru_cache(maxsize=None)
def source_revision() -> Dict[str, Optional[str]]:
    """Which code drove the episode: the checkout's commit (and whether it was dirty) and a hash of the harness package's
    own files, which identifies uncommitted harness code too."""
    h = hashlib.sha1()
    for root, dirs, files in os.walk(HARNESS_DIR):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith((".py", ".md")):
                path = os.path.join(root, name)
                h.update(os.path.relpath(path, HARNESS_DIR).encode())
                with open(path, "rb") as f:
                    h.update(f.read())
    out: Dict[str, Optional[str]] = {"harness_sha1": h.hexdigest(), "commit": None, "dirty": None}
    try:
        repo = os.path.dirname(os.path.dirname(HARNESS_DIR))
        out["commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
        status = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout
        out["dirty"] = str(bool(status.strip())).lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


class EpisodeTrace:
    def __init__(self, out_dir: Optional[str]):
        self.dir = out_dir
        self.t0 = time.time()
        self.events = 0
        if out_dir:
            os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
            self._f = open(os.path.join(out_dir, "transcript.jsonl"), "a")
        else:
            self._f = None

    def event(self, event_type: str, /, **fields: Any) -> None:
        self.events += 1
        if self._f is None:
            return
        rec = {"t": round(time.time() - self.t0, 3), "type": event_type}
        rec.update(_jsonable(fields))
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def frame(self, name: str, jpeg: bytes) -> Optional[str]:
        """Save one image as sent; returns its path relative to the trace dir."""
        if not self.dir:
            return None
        rel = os.path.join("frames", name)
        with open(os.path.join(self.dir, rel), "wb") as f:
            f.write(jpeg)
        return rel

    def summary(self, data: Dict[str, Any]) -> None:
        if self.dir:
            with open(os.path.join(self.dir, "summary.json"), "w") as f:
                json.dump(_jsonable(data), f, indent=1)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
