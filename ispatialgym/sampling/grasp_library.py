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
import math
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


# ------------------------------------------------------------------------------------------------ v2 (opt-in)
#: ``ISPATIALGYM_GRASP_LIBRARY_V2=<root>`` (``ispatialgym.grasping``): a legacy key that a v2 entry replaces (its ``legacy_key``) is
#: served from v2 - the physics-validated representatives with their MEASURED rest (``rest_quat_xyzw`` returns it). Unset: the
#: legacy library exactly as before. A v2 entry is never synthesised from a legacy one (no measured rest), and the sampler and the
#: oracle must run with the same setting (grasp names differ between the libraries).


def v2_root() -> Optional[str]:
    return os.environ.get("ISPATIALGYM_GRASP_LIBRARY_V2") or None


@lru_cache(maxsize=4)
def _v2_index(root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in glob.glob(os.path.join(root, "models", "*", "library.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("legacy_key"):
            out[d["legacy_key"]] = p
    return out


def v2_entry(key: str):
    """The v2 ``ModelLibrary`` that replaces legacy ``key`` (None: no v2 root, or no entry)."""
    root = v2_root()
    if not root:
        return None
    p = _v2_index(root).get(key)
    if p is None:
        return None
    from ispatialgym.grasping import schema as S

    with open(p) as f:
        d = json.load(f)
    return S.load_model(d["category"], d["model"], d.get("scale"), root, constraint=d.get("constraint") or "tabletop",
                        rest_hint_wxyz=d.get("rest_hint_wxyz") if "rest_hint_wxyz" in d else _v2_hint(p))


def _v2_hint(path: str):
    c = os.path.join(os.path.dirname(path), "candidates.json")
    if os.path.exists(c):
        with open(c) as f:
            return json.load(f).get("rest_hint_wxyz")
    return None


def grasps_from_v2(lib, key: str, rest_id: Optional[str] = None) -> List["Grasp"]:
    out = []
    for g in lib.grasps(rest_id):
        T = g["T_object_eef"]
        out.append(Grasp(key=key, name=g["name"], position=th.tensor(T["position"], dtype=th.float32), quat=th.tensor(T["orientation_xyzw"], dtype=th.float32),
                         finger_q=[float(v) for v in g.get("finger_q", [])],
                         meta={"width": float(g.get("width_m", 0.0)), "rest_id": g.get("rest_id"), "family_id": g.get("family_id"), "mode": g.get("mode"),
                               "library": "v2", "physics": g.get("physics", "tested")}))
    return out


def rest_quat_xyzw(key: str) -> Optional[th.Tensor]:
    """Orientation the object had to be placed in (side rest pose) for the tabletop grasps, or None (canonical). With a v2 entry
    (``ISPATIALGYM_GRASP_LIBRARY_V2``): its measured rest (xyzw, reference yaw included)."""
    lib = v2_entry(key)
    if lib is not None and lib.rests:
        rid = sorted(lib.rests, key=lambda r: (not lib.rests[r].stable, r))[0]
        return th.tensor([float(v) for v in lib.rests[rid].quat_xyzw], dtype=th.float32)
    d = _candidates(key)
    rq = (d.get("ispatialgym") or {}).get("rest_quat_wxyz")
    if rq is None or (d.get("ispatialgym") or {}).get("rest_pose", "canonical") == "canonical":
        return None
    w, x, y, z = rq
    return th.tensor([x, y, z, w], dtype=th.float32)


def load_grasps(key: str, require_clean: bool = True) -> List[Grasp]:
    lib = v2_entry(key)
    if lib is not None:
        out = grasps_from_v2(lib, key)
        if not out:
            raise ValueError(f"grasp key {key!r}: its v2 entry {lib.path} has no physics-validated grasps")
        return out
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


def twin(g) -> Grasp:
    """The same pinch with the fingers swapped: the eef turned π about its approach axis (``conventions.twin_pose``),
    named ``<row>~r180`` (``conventions.twin_name``), ``finger_q`` reversed, the row's ``meta`` (pinch width) with
    ``twin_of`` = the row. The twin of a twin is the row again (same name and rotation)."""
    from ispatialgym.robot import conventions as C

    base, flipped = C.split_twin(g.name)
    pos, quat = C.twin_pose((g.position, g.quat))
    meta = dict(getattr(g, "meta", None) or {})
    if flipped:
        meta.pop("twin_of", None)
    else:
        meta["twin_of"] = base
    return Grasp(key=getattr(g, "key", None), name=C.twin_name(g.name), position=pos, quat=quat,
                 finger_q=list(reversed([float(v) for v in (getattr(g, "finger_q", None) or [])])), meta=meta,
                 verdict=getattr(g, "verdict", "ok"), flags=list(getattr(g, "flags", None) or []), note=getattr(g, "note", ""))


def stored_twin(g, grasps: List[Grasp], pos_tol: float = 0.015, ang_tol_deg: float = 20.0) -> Optional[Grasp]:
    """A row of ``grasps`` other than ``g`` that already is ``g``'s twin within ``pos_tol`` (m) / ``ang_tol_deg``: the
    candidate NMS compares rotations without the half turn, so ~40 % of the validated rows have one."""
    from ispatialgym.robot import conventions as C

    t = twin(g)
    for h in grasps:
        if h is g or h.name == g.name:
            continue
        if float(th.linalg.norm(h.position - t.position)) <= pos_tol and C.quat_angle(h.quat, t.quat) <= math.radians(ang_tol_deg):
            return h
    return None


def with_twins(grasps: List[Grasp]) -> List[Grasp]:
    """Both finger assignments of every pinch: the rows, then the twin of each row that has no stored twin among them."""
    rows = list(grasps)
    return rows + [twin(g) for g in rows if stored_twin(g, rows) is None]


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
        if os.path.exists(os.path.join(root, "results", f"{key}.json")) or v2_entry(key) is not None:
            return key
    raise FileNotFoundError(f"no grasp library entry for {category}/{model}; tried {tried}")
