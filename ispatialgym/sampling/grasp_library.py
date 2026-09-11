"""Access to the validated R1Pro grasp library (``runlogs/grasp_library_20260905``).

A *key* names one (category, model, pipeline) run, e.g. ``bottle_of_beer__gxxbhh`` (GraspGen floating grasps),
``pack_of_kielbasa__dgodnn__t`` (tabletop sampler), ``lemon__ovbcff__i`` (tabletop insert), ``cap__xxqyrt``
(attachment-free). For every key:

* ``cands/<key>.yaml``  - isaac_grasp candidates: eef (``right_eef_link``) pose in the object ``base_link`` frame,
* ``results/<key>.json`` - assisted-grasp outcome per candidate (attached + held under gravity),
* ``review/wave*.json``  - reviewer verdict per validated row (rank order = order of the valid rows in results).

:func:`load_grasps` returns the grasps that attached, held, and were not flagged by the reviewer (``one_finger`` is
tolerated: it is an assisted-grasping artefact, see the library README).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import torch as th
import yaml

from ispatialgym import REPO_ROOT

DEFAULT_LIBRARY = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905")
TOLERATED_FLAGS = {"one_finger"}


def library_root() -> str:
    return os.environ.get("ISPATIALGYM_GRASP_LIBRARY", DEFAULT_LIBRARY)


@dataclass
class Grasp:
    key: str
    name: str
    #: eef pose in the object base_link frame (xyzw)
    position: th.Tensor
    quat: th.Tensor
    finger_q: List[float]
    meta: Dict = field(default_factory=dict)
    verdict: str = "ok"
    flags: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def pose(self):
        return self.position, self.quat

    def to_json(self) -> Dict:
        return {
            "key": self.key,
            "name": self.name,
            "eef_in_object": {"position": self.position.tolist(), "orientation": self.quat.tolist()},
            "finger_q": list(self.finger_q),
            "verdict": self.verdict,
            "flags": list(self.flags),
            "meta": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.meta.items()},
        }


@lru_cache(maxsize=1)
def _reviews() -> Dict[str, dict]:
    """Latest reviewer entry per key (later waves override earlier ones)."""
    out: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(library_root(), "review", "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("models", data.get("results", []))
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("key"):
                out[it["key"]] = it
    return out


def _candidates(key: str) -> Dict[str, dict]:
    path = os.path.join(library_root(), "cands", f"{key}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no grasp candidates for {key!r} at {path}")
    with open(path) as f:
        d = yaml.safe_load(f)
    return d


def rest_quat_xyzw(key: str) -> Optional[th.Tensor]:
    """Orientation the object had to be placed in (side rest pose) for the tabletop grasps, or None (canonical)."""
    d = _candidates(key)
    rq = (d.get("ispatialgym") or {}).get("rest_quat_wxyz")
    if rq is None or (d.get("ispatialgym") or {}).get("rest_pose", "canonical") == "canonical":
        return None
    w, x, y, z = rq
    return th.tensor([x, y, z, w], dtype=th.float32)


def load_grasps(key: str, require_clean: bool = True) -> List[Grasp]:
    root = library_root()
    cands = _candidates(key).get("grasps") or {}
    res_path = os.path.join(root, "results", f"{key}.json")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no grasp results for {key!r} at {res_path}")
    with open(res_path) as f:
        results = json.load(f)
    review = _reviews().get(key, {})
    verdicts = {g.get("rank"): g for g in review.get("grasps", [])}
    out: List[Grasp] = []
    rank = 0
    for rec in results.get("tested", []):
        if not (rec.get("ag_attached") and rec.get("held_after_gravity")):
            continue
        rank += 1
        c = cands.get(rec["grasp"])
        if c is None:
            continue
        v = verdicts.get(rank, {})
        flags = list(v.get("flags", []))
        verdict = v.get("verdict", "ok")
        if require_clean and (verdict != "ok" and set(flags) - TOLERATED_FLAGS):
            continue
        q = c["orientation"]
        out.append(
            Grasp(
                key=key,
                name=rec["grasp"],
                position=th.tensor(c["position"], dtype=th.float32),
                quat=th.tensor([*q["xyz"], q["w"]], dtype=th.float32),
                finger_q=[float(x) for x in rec.get("finger_q", [])],
                meta=dict(rec.get("meta", {})),
                verdict=verdict,
                flags=flags,
                note=v.get("note", ""),
            )
        )
    if not out:
        raise ValueError(f"grasp key {key!r} has no validated (clean) grasps")
    return out


def default_key(category: str, model: str, task_kind: str) -> str:
    """Library key convention: floating (GraspGen) keys have no suffix, tabletop ``__t``, insert ``__i``,
    attachment-free ``__a``; round-1 tabletop runs have no suffix either, so callers should prefer explicit keys."""
    suffix = {"floating": "", "tabletop": "__t", "insert": "__i", "attachment": "__a"}.get(task_kind, "")
    return f"{category}__{model}{suffix}"


def resolve_key(category: str, model: str, preferred: Optional[str], kinds: List[str]) -> str:
    """First existing key among ``preferred`` and the conventional keys for ``kinds``."""
    root = library_root()
    tried = []
    for key in ([preferred] if preferred else []) + [default_key(category, model, k) for k in kinds] + [f"{category}__{model}"]:
        if key in tried:
            continue
        tried.append(key)
        if os.path.exists(os.path.join(root, "results", f"{key}.json")):
            return key
    raise FileNotFoundError(f"no grasp library entry for {category}/{model}; tried {tried}")
