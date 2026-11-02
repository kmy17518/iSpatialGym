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

import numpy as np
import torch as th

log = logging.getLogger("ispatialgym.sampling")

#: bump when the semantics of a cached computation change (invalidates every entry)
CACHE_SCHEMA = 1


def _plain(x, round_floats: bool = True):
    """JSON-able copy of ``x``: dicts with sorted string keys, tensors / tuples as lists, floats rounded to 4 decimals (the key hash and
    today's entries). ``round_floats=False`` (2026-10-31 train-fast: entries that hold stances the sampler re-uses as they were, the
    start-per-attempt lateral cells) keeps every float and tensor value exact."""
    if isinstance(x, dict):
        return {str(k): _plain(v, round_floats) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v, round_floats) for v in x]
    if isinstance(x, th.Tensor):
        return [round(float(v), 4) for v in x.flatten().tolist()] if round_floats else x.flatten().tolist()
    if isinstance(x, float):
        return round(x, 4) if round_floats else x
    return x


def _json_default(x):
    """``json.dump`` fallback for the numpy / torch scalars a value may still carry (a ``np.bool_`` mask verdict, a 0-d tensor angle)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, th.Tensor):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def cache_root() -> Optional[str]:
    root = os.environ.get("ISPATIALGYM_SAMPLING_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "ispatialgym", "sampling"))
    if root.lower() in ("off", "0", "none", ""):
        return None
    return root


def cache_key(kind: str, payload: Any) -> str:
    blob = json.dumps({"schema": CACHE_SCHEMA, "kind": kind, "payload": _plain(payload)}, sort_keys=True, default=_json_default)
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


def cache_put(key: str, value: Any, *, round_floats: bool = True) -> None:
    """Write ``value`` under ``key`` (atomic: a temp file renamed into place). 2026-10-31 train-fast: the temp name is unique per process
    (``.tmp.<pid>``) - many train workers write the same entry at once on NFS, and a shared ``.tmp`` name let one writer rename
    another's half-written file -, and a temp file that vanished under us (another writer's rename raced ours on a non-POSIX mount) is
    tolerated: the entry on disk is then the other writer's, computed from the same key. ``round_floats`` as :func:`_plain`."""
    root = cache_root()
    if root is None:
        return
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, key + ".json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(_plain(value, round_floats), f, default=_json_default)
        os.replace(tmp, path)
    except FileNotFoundError as e:
        log.warning("cache entry %s: temp file vanished before it was renamed (a concurrent writer of the same entry?): %r", path, e)
    except OSError as e:  # a stale NFS handle / a full or read-only mount: the entry is simply not written this time, the sampler goes on
        log.warning("cache entry %s not written: %r", path, e)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
