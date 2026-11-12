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
from typing import Any, Dict, List, Optional, Tuple

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
    """legacy key -> library.json path. of-v2maps 2026-11-07 (the isg1020 spec's lookup rule, P2 §11): deterministic when several entries
    share a legacy key (a retry / scale variant copied into the same root) - the entry with a stable rest and valid grasps wins, then the
    shortest id, then the path (was: the last file of the glob)."""
    cand: Dict[str, List[Tuple[Tuple[Any, ...], str]]] = {}
    for p in sorted(glob.glob(os.path.join(root, "models", "*", "library.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        # of-v2maps 2026-11-07 (prompt I): ``legacy_keys`` - the campaign's links of an entry to the sampler keys it serves (a roles_all /
        # retry entry has legacy_key None; a held-object floating key or a task variant's key names the same model)
        keys = [k for k in [d.get("legacy_key"), *(d.get("legacy_keys") or [])] if k]
        if not keys:
            continue
        if _is_lean_library(d) or _is_overhang_library(d):
            continue  # i7-lean / i7-overhang 2026-11-07: a lean / overhang entry never serves a stable key (their own indexes list them)
        rests = d.get("rests") or {}
        stable = any(bool((r or {}).get("stable")) for r in rests.values())
        eid = os.path.basename(os.path.dirname(p))
        rank = (not stable, not bool(d.get("grasps")), len(eid), p)
        for k in dict.fromkeys(keys):
            cand.setdefault(k, []).append((rank, p))
    return {k: sorted(v)[0][1] for k, v in cand.items()}


def _is_lean_library(d: Dict[str, Any]) -> bool:
    """i7-lean 2026-11-07: a library.json of a lean entry (``rest_kind: "lean"``, or any lean rest record)."""
    if str(d.get("rest_kind") or "stable") == "lean":
        return True
    return any(str((r or {}).get("rest_kind") or "stable") == "lean" for r in (d.get("rests") or {}).values())


def _is_overhang_library(d: Dict[str, Any]) -> bool:
    """i7-overhang 2026-11-07: a library.json of an overhang entry (``rest_kind: "overhang"``, or any overhang rest record)."""
    if str(d.get("rest_kind") or "stable") == "overhang":
        return True
    return any(str((r or {}).get("rest_kind") or "stable") == "overhang" for r in (d.get("rests") or {}).values())


@lru_cache(maxsize=4)
def _v2_overhang_index(root: str) -> Dict[Tuple[str, str], str]:
    """i7-overhang 2026-11-07: ``(base key, overhang tag) -> library.json`` of the OVERHANG entries (``overhang_keys`` + ``legacy_key``; the
    tag is the entry's ``overhang.tag``: side, distance, edge, slab). Deterministic: the entry with valid grasps, then the shortest id, then
    the path."""
    cand: Dict[Tuple[str, str], List[Tuple[Tuple[Any, ...], str]]] = {}
    for p in sorted(glob.glob(os.path.join(root, "models", "*", "library.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if not _is_overhang_library(d) or not isinstance(d.get("overhang"), dict):
            continue
        tag = d["overhang"].get("tag")
        if not tag:
            continue
        eid = os.path.basename(os.path.dirname(p))
        for k in dict.fromkeys(k for k in [d.get("legacy_key"), *(d.get("overhang_keys") or [])] if k):
            cand.setdefault((k, tag), []).append(((not bool(d.get("grasps")), len(eid), p), p))
    return {k: sorted(v)[0][1] for k, v in cand.items()}


def overhang_key_parts(key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(base key, overhang tag or None)`` of a sampler grasp key (``<base>@<overhang tag>``: ``grasping.overhang.overhang_key``)."""
    from ispatialgym.grasping import overhang as OV

    return OV.split_overhang_key(key)


def v2_overhang_tags(base_key: str) -> List[Tuple[str, Dict[str, Any]]]:
    """i7-overhang: the overhang rests the v2 library holds for ``base_key``: ``[(tag, overhang json)]`` sorted by (side, distance) - only
    entries with physics-validated grasps."""
    root = v2_root()
    if not root:
        return []
    out = []
    for (k, tag), p in _v2_overhang_index(root).items():
        if k != base_key:
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("grasps"):
            out.append((tag, d["overhang"]))
    return sorted(out, key=lambda t: (t[1].get("side"), float(t[1].get("distance_m", 0.0)), t[0]))


@lru_cache(maxsize=4)
def _v2_lean_index(root: str) -> Dict[Tuple[str, str], str]:
    """i7-lean 2026-11-07: ``(base key, lean tag) -> library.json`` of the LEAN entries (``lean_keys`` + ``legacy_key``; the tag is the
    entry's ``lean.tag``: angle, side, board, gap). Deterministic: the entry with valid grasps, then the shortest id, then the path."""
    cand: Dict[Tuple[str, str], List[Tuple[Tuple[Any, ...], str]]] = {}
    for p in sorted(glob.glob(os.path.join(root, "models", "*", "library.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if not _is_lean_library(d) or not isinstance(d.get("lean"), dict):
            continue
        tag = d["lean"].get("tag")
        if not tag:
            continue
        eid = os.path.basename(os.path.dirname(p))
        for k in dict.fromkeys(k for k in [d.get("legacy_key"), *(d.get("lean_keys") or [])] if k):
            cand.setdefault((k, tag), []).append(((not bool(d.get("grasps")), len(eid), p), p))
    return {k: sorted(v)[0][1] for k, v in cand.items()}


def lean_key_parts(key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(base key, lean tag or None)`` of a sampler grasp key (``<base>@<lean tag>``: ``grasping.lean.lean_key``)."""
    from ispatialgym.grasping import lean as LN

    return LN.split_lean_key(key)


def v2_lean_tags(base_key: str) -> List[Tuple[str, Dict[str, Any]]]:
    """i7-lean: the lean rests the v2 library holds for ``base_key``: ``[(tag, lean json)]`` sorted by (side, angle) - only entries with
    physics-validated grasps."""
    root = v2_root()
    if not root:
        return []
    out = []
    for (k, tag), p in _v2_lean_index(root).items():
        if k != base_key:
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("grasps"):
            out.append((tag, d["lean"]))
    return sorted(out, key=lambda t: (t[1].get("side"), float(t[1].get("deg", 0.0)), t[0]))


def v2_entry(key: str):
    """The v2 ``ModelLibrary`` that replaces legacy ``key`` (None: no v2 root, or no entry). i7-lean: a lean key (``<base>@<lean tag>``)
    is served by the lean entry of exactly that tag, a plain key never by a lean entry."""
    root = v2_root()
    if not root:
        return None
    base, tag = lean_key_parts(key)
    obase, otag = overhang_key_parts(key)  # i7-overhang: an overhang key is served by the overhang entry of exactly that tag
    if otag is not None:
        p = _v2_overhang_index(root).get((obase, otag))
    else:
        p = _v2_lean_index(root).get((base, tag)) if tag is not None else _v2_index(root).get(key)
    if p is None:
        return None
    from ispatialgym.grasping import schema as S

    with open(p) as f:
        d = json.load(f)
    lib = S.load_model(d["category"], d["model"], d.get("scale"), root, constraint=d.get("constraint") or "tabletop",
                       rest_hint_wxyz=d.get("rest_hint_wxyz") if "rest_hint_wxyz" in d else _v2_hint(p),
                       lean=d.get("lean") if tag is not None else None, overhang=d.get("overhang") if otag is not None else None,
                       validate_arm=d.get("validate_arm") if tag is not None else None,
                       access_region=(d.get("access_region") or {}).get("tag") if tag is not None and isinstance(d.get("access_region"), dict) else None)
    _check_v2_policy(lib, d)
    return lib


def _check_v2_policy(lib, d: Dict[str, Any]) -> None:
    """of-v2maps 2026-11-07 (P1 §2 / P2 §11): an entry validated under another contact policy than the shared one does not serve
    (``ValueError``); an entry without a policy key (older schema) is accepted as before."""
    pk = d.get("policy_key")
    if not pk:
        return
    from ispatialgym.grasping import contact_policy as CP

    want = CP.DEFAULT.key(d.get("arm") or "right")
    if pk != want:
        raise ValueError(f"v2 entry {getattr(lib, 'path', '?')}: policy {pk} != the shared {want}")


def v2_rest_id(lib, lean: Optional[str] = None, overhang: Optional[str] = None) -> Optional[str]:
    """The rest a v2 entry is used at (of-v2maps 2026-11-07): its stable rests first, then the id - deterministic; None without a stable rest
    (the spec: a measured, stable rest is required). i7-lean 2026-11-07: only rests of kind "stable" unless the caller asks for a lean rest
    (``lean`` = its tag): then that lean rest only (stable, of that tag) - a lean rest never serves a stable lookup, nor the reverse."""
    if lib is None or not lib.rests:
        return None
    if overhang is not None:  # i7-overhang: that overhang rest only (stable, of that tag)
        r = lib.rests.get(overhang)
        return overhang if (r is not None and r.stable and getattr(r, "kind", "stable") == "overhang") else None
    if lean is not None:
        r = lib.rests.get(lean)
        return lean if (r is not None and r.stable and getattr(r, "kind", "stable") == "lean") else None
    stable = [r for r in lib.rests if lib.rests[r].stable and getattr(lib.rests[r], "kind", "stable") == "stable"]
    return sorted(stable)[0] if stable else None


def v2_rest_id_for_key(lib, key: Optional[str]) -> Optional[str]:
    """:func:`v2_rest_id` of the rest a grasp key names (a lean key: its lean tag; an overhang key: its overhang tag; a plain key: the
    stable rest)."""
    return v2_rest_id(lib, lean=lean_key_parts(key)[1], overhang=overhang_key_parts(key)[1])


def _v2_hint(path: str):
    c = os.path.join(os.path.dirname(path), "candidates.json")
    if os.path.exists(c):
        with open(c) as f:
            return json.load(f).get("rest_hint_wxyz")
    return None


#: a family member's grasp name: ``<row>@<param>`` (of-v2maps 2026-11-07, map-seeded F: ``v2/map_f.py``)
MEMBER_SEP = "@"


def family_member(g: "Grasp", family: Dict[str, Any], param: float) -> "Grasp":
    """The member of ``g``'s v2 family at ``param`` (rad / m from the representative, ``schema.family_member_T``) as a grasp of its own,
    named ``<row>@<param>`` - same key, pinch width and rest; ``meta`` adds ``family_param`` / ``family_base``. The oracle executes the
    recorded ``eef_in_object``; :func:`conventions.grasp_entry` resolves the name."""
    import copy

    import numpy as np

    from ispatialgym.grasping import schema as S

    T = S.pose_mat(g.position.tolist(), g.quat.tolist())
    Tm = S.family_member_T(family, T, float(param))
    pose = S.mat_pose(Tm)
    out = copy.copy(g)
    out.name = f"{g.name}{MEMBER_SEP}{float(param):+.4f}"
    out.position = th.tensor(np.asarray(pose["position"], dtype=float), dtype=th.float32)
    out.quat = th.tensor(np.asarray(pose["orientation_xyzw"], dtype=float), dtype=th.float32)
    out.meta = {**(g.meta or {}), "family_param": round(float(param), 6), "family_base": g.name}
    return out


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
    if lib is not None and getattr(lib, "floating", False):
        # i7-floating 2026-11-07: a floating entry (held-object grasps) has no rest - it never serves a placement on a support
        raise ValueError(f"grasp key {key!r}: its v2 entry {lib.path} is a floating entry (no rest)")
    if lib is not None and lib.rests:
        rid = v2_rest_id_for_key(lib, key)
        if rid is None:
            raise ValueError(f"grasp key {key!r}: its v2 entry {lib.path} has no stable rest" + (" of that lean" if lean_key_parts(key)[1] else "")
                             + (" of that overhang" if overhang_key_parts(key)[1] else ""))
        return th.tensor([float(v) for v in lib.rests[rid].quat_xyzw], dtype=th.float32)
    if lean_key_parts(key)[1] is not None:
        raise ValueError(f"lean grasp key {key!r}: no v2 lean entry (ISPATIALGYM_GRASP_LIBRARY_V2 {v2_root()!r})")
    if overhang_key_parts(key)[1] is not None:
        raise ValueError(f"overhang grasp key {key!r}: no v2 overhang entry (ISPATIALGYM_GRASP_LIBRARY_V2 {v2_root()!r})")
    d = _candidates(key)
    rq = (d.get("ispatialgym") or {}).get("rest_quat_wxyz")
    if rq is None or (d.get("ispatialgym") or {}).get("rest_pose", "canonical") == "canonical":
        return None
    w, x, y, z = rq
    return th.tensor([x, y, z, w], dtype=th.float32)


def v2_only() -> bool:
    """``ISPATIALGYM_V2_ONLY=1`` (``submit_fast.py --grasps v2only``; ``reach_bands.v2_only``): no legacy grasps either."""
    return os.environ.get("ISPATIALGYM_V2_ONLY", "") == "1"


def load_grasps(key: str, require_clean: bool = True) -> List[Grasp]:
    lib = v2_entry(key)
    if lib is None and lean_key_parts(key)[1] is not None:
        # i7-lean 2026-11-07: a lean key is served by its lean entry only (never the legacy library, never a stable entry)
        raise ValueError(f"lean grasp key {key!r} has no v2 lean entry with that lean rest")
    if lib is None and overhang_key_parts(key)[1] is not None:
        # i7-overhang 2026-11-07: an overhang key is served by its overhang entry only (never the legacy library, never a stable entry)
        raise ValueError(f"overhang grasp key {key!r} has no v2 overhang entry with that overhang rest")
    if lib is None and v2_only() and v2_root() and not str(key).startswith(("attach:", "generic")):
        # of-v2maps 2026-11-07 (prompt I; owner: only the v2 grasps): v2only served the LEGACY library for a key without a v2 entry -
        # v2only_before wrote 20 aliases (mental_rotation, object_scaling) on legacy grasps
        import logging

        logging.getLogger("ispatialgym.sampling.v2").info("LEGACY_GRASPS_BLOCKED %s", key)
        raise ValueError(f"v2 only: grasp key {key!r} has no v2 library entry (legacy grasps are not served)")
    if lib is not None:
        # of-v2maps 2026-11-07 (P1 §3a: each grasp belongs to its rest_id): only the grasps of the rest the object is placed at
        tag = lean_key_parts(key)[1]
        otag = overhang_key_parts(key)[1]
        if otag is not None:  # i7-overhang: an overhang key's own overhang rest only
            rid = v2_rest_id(lib, overhang=otag)
            out = grasps_from_v2(lib, key, rest_id=rid) if rid is not None else []
        elif tag is None:
            out = grasps_from_v2(lib, key, rest_id=v2_rest_id(lib))  # a plain key: exactly as before
        else:  # i7-lean: a lean key's own lean rest only
            rid = v2_rest_id(lib, lean=tag)
            out = grasps_from_v2(lib, key, rest_id=rid) if rid is not None else []
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
    """First compatible key among the preferred and conventional keys.

    An insert consumer must not inherit an ordinary tabletop entry through an
    explicit selected-model hint or the unsuffixed fallback. ``insert`` is the
    existing role family for both insert and flower-stem libraries; an explicit
    constraint narrows it further. Inspect library metadata, never key suffixes.
    Other role families retain their existing resolution behavior.
    """
    root = library_root()
    specialized = {"insert": {"tabletop_insert", "flower_stem_top"},
                   "i": {"tabletop_insert", "flower_stem_top"},
                   "fl": {"tabletop_insert", "flower_stem_top"},
                   "tabletop_insert": {"tabletop_insert"}, "flower_stem_top": {"flower_stem_top"}}
    allowed = set().union(*(specialized[k] for k in kinds)) if kinds and all(k in specialized for k in kinds) else None
    tried, incompatible = [], []
    for key in ([preferred] if preferred else []) + [default_key(category, model, k) for k in kinds] + [f"{category}__{model}"]:
        if key in tried:
            continue
        tried.append(key)
        legacy_path = os.path.join(root, "results", f"{key}.json")
        lib = v2_entry(key)
        if allowed is not None and (lib is not None or os.path.exists(legacy_path)):
            if lib is not None:
                constraint = lib.data.get("constraint") or "tabletop"
            else:
                with open(legacy_path) as f:
                    constraint = json.load(f).get("constraint")
                if not constraint:
                    try:
                        constraint = (_candidates(key).get("ispatialgym") or {}).get("constraint")
                    except FileNotFoundError:
                        pass
            if constraint not in allowed:
                incompatible.append(f"{key}:{constraint or 'unrecorded constraint'}")
                continue
        if os.path.exists(legacy_path) or lib is not None:
            return key
    raise FileNotFoundError(f"no compatible grasp library entry for {category}/{model}; tried {tried}; incompatible {incompatible}")
