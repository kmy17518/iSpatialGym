"""On-disk cache for per-template precomputations (reach grids, grasp combination tables).

The expensive part of a template is batched cuRobo IK over the placement lattice. Its inputs are fully described by
the scene (model + loaded rooms, which fix the collision world while task objects are parked), the stance, the lattice,
the query poses and the IK settings, so results are keyed by a hash of exactly those and reused across runs of the same
spec (regenerating more instances of an existing template costs no IK at all).

Location: ``$ISPATIALGYM_SAMPLING_CACHE`` (default ``~/.cache/ispatialgym/sampling``); ``ISPATIALGYM_SAMPLING_CACHE=off``
disables it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

import torch as th

log = logging.getLogger("ispatialgym.sampling")

#: bump when the semantics of a cached computation change (invalidates every entry)
CACHE_SCHEMA = 1


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, th.Tensor):
        return [round(float(v), 4) for v in x.flatten().tolist()]
    if isinstance(x, float):
        return round(x, 4)
    return x


def cache_root() -> Optional[str]:
    root = os.environ.get("ISPATIALGYM_SAMPLING_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "ispatialgym", "sampling"))
    if root.lower() in ("off", "0", "none", ""):
        return None
    return root


def cache_key(kind: str, payload: Any) -> str:
    blob = json.dumps({"schema": CACHE_SCHEMA, "kind": kind, "payload": _plain(payload)}, sort_keys=True)
    return f"{kind}-{hashlib.sha256(blob.encode()).hexdigest()[:24]}"


def cache_get(key: str) -> Optional[Any]:
    root = cache_root()
    if root is None:
        return None
    path = os.path.join(root, key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # corrupt entry: ignore it
        log.warning("ignoring unreadable cache entry %s (%r)", path, e)
        return None


def cache_put(key: str, value: Any) -> None:
    root = cache_root()
    if root is None:
        return
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, key + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_plain(value), f)
    os.replace(tmp, path)
