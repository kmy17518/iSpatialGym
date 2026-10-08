"""The BEHAVIOR shard's format (changes recorded data) and the marker that keeps one format per dataset.

One format per dataset: shards with and without depth cannot be aggregated (different features), and x265 and NVENC
episodes must not be concatenated into one video by stream copy - their LeRobot metadata is identical (hevc, yuv420p),
so nothing in LeRobot refuses a mixed aggregate. Every shard therefore gets a marker next to it,
``<shard dir>.format.json``, and a shard is only resumed by a process writing the same format.

Formats (``ISPATIALGYM_SHARD_FORMAT``):
  legacy         the default, production's format: depth_linear streams, rgb by x265 crf 30.
  nodepth_nvenc  opt-in (v2.0-multienv's default): the robot records proprio + rgb only (no depth annotators, readback or
                 streams), the datasets keep observation.robot2cam_pose.<camera> for every camera, and the rgb streams
                 are encoded by hevc_nvenc (QP 33, preset p7, GOP 8, yuv420p, no B-frames).
For experiments ``ISPATIALGYM_SHARD_NO_DEPTH=0|1`` and ``ISPATIALGYM_SHARD_RGB_ENCODER=<spec>|x265`` override one part.
No omnigibson import here: scripts can check markers without loading the simulator.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

FORMATS: Dict[str, Dict[str, Any]] = {
    "nodepth_nvenc": {"depth": False, "rgb_encoder": "hevc_nvenc,qp=33,preset=p7"},
    "legacy": {"depth": True, "rgb_encoder": "x265"},
}
DEFAULT = "legacy"
MARKER_SUFFIX = ".format.json"
#: the parts that decide whether two shards may meet in one dataset
KEYS = ("depth", "rgb_encoder")


def effective(environ=os.environ) -> Dict[str, Any]:
    """The format this process writes: ``{"name", "depth", "rgb_encoder"}`` (name "custom" when an override changes a
    named format)."""
    name = environ.get("ISPATIALGYM_SHARD_FORMAT", DEFAULT)
    if name not in FORMATS:
        raise ValueError(f"ISPATIALGYM_SHARD_FORMAT={name!r}: one of {sorted(FORMATS)}")
    fmt = dict(FORMATS[name])
    if environ.get("ISPATIALGYM_SHARD_NO_DEPTH") in ("0", "1"):
        fmt["depth"] = environ["ISPATIALGYM_SHARD_NO_DEPTH"] == "0"
    if environ.get("ISPATIALGYM_SHARD_RGB_ENCODER"):
        fmt["rgb_encoder"] = environ["ISPATIALGYM_SHARD_RGB_ENCODER"]
    if fmt != FORMATS[name]:
        name = "custom"
    return {"name": name, **fmt}


def marker_path(shard: str) -> str:
    return shard.rstrip("/") + MARKER_SUFFIX


def write_marker(shard: str, fmt: Dict[str, Any]) -> None:
    with open(marker_path(shard), "w") as f:
        json.dump({k: fmt[k] for k in ("name",) + KEYS}, f, indent=1)


def read_marker(shard: str):
    try:
        with open(marker_path(shard)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def has_depth_features(shard: str) -> bool:
    with open(os.path.join(shard, "meta", "info.json")) as f:
        return any(".depth" in k for k in json.load(f).get("features", {}))


def check_resume(shard: str, fmt: Dict[str, Any]) -> None:
    """Raise RuntimeError unless a process writing ``fmt`` may append to the existing ``shard``. A shard without a marker
    is accepted only as production's format (it has depth features, and x265 is all production writes) and only by a
    process writing that format; it then gets its marker."""
    want = {k: fmt[k] for k in KEYS}
    have = read_marker(shard)
    if have is not None:
        have = {k: have.get(k) for k in KEYS}
        if have != want:
            raise RuntimeError(f"shard {shard} was written as {have}; this process writes {want}: start a new shard")
        return
    depth = has_depth_features(shard)
    if not (depth and want == {k: FORMATS["legacy"][k] for k in KEYS}):
        raise RuntimeError(f"shard {shard} has no format marker (depth features: {depth}); this process writes {want}: "
                           "refusing to resume it")
    write_marker(shard, {"name": "legacy", **want})
