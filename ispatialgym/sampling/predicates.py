"""Robust BDDL predicate evaluation shared by the sampler's goal check and the task evaluator (D-007, option A, owner
2026-09-16 21:17).

``under`` / ``ontop`` through OmniGibson alone are fragile on a physically correct placement: ``Under`` / ``OnTop`` shoot a
single vertical ray from the supporting object's own surface point into the resting object, and whether that ray
registers the object depends on sub-millimetre contact / penetration (teacup on the saucer well, 0916-172359 and
0916-192655: the cup 1.4 mm from its goal, 2.7 deg tilt, ``binary_success_ever`` False; the same geometry passed at 18:41).

``evaluate_predicate(predicate_cls, *entities)`` is a drop-in for ``omnigibson.utils.bddl_utils.evaluate_bddl_predicate``:

1. **omnigibson** — OmniGibson's verdict; True ends the evaluation.
2. **rays** (``under`` / ``ontop`` only) — several rays across the contact patch: the resting object's footprint centre and a
   ring at half its footprint radius; at each point the supporting object's surface under the resting object is found with
   a downward ray (the resting object's bodies ignored), the probe ray starts ``RAY_BELOW_SURFACE_M`` below that surface and
   goes up (the supporting object's bodies ignored); any hit on the resting object counts.
3. **geometry** — the resting object's AABB bottom within ``BOTTOM_GAP_MAX_M`` of the supporting surface under its footprint
   centre, the footprint centre inside the supporting object's top footprint, the resting object's up axis within
   ``UP_AXIS_MAX_DEG`` of vertical (a concave partner whose collision mesh lets the rays through).

Every evaluation appends ``{"predicate", "entities", "branch", "ok", "detail"}`` to the branch log (``reset_branches`` /
``branches``); the callers copy it into ``goal_check.binary_branch`` (sampler) and ``meta.binary_branch`` (evaluator) so the
ledger shows which branch decided. Other predicates pass straight through.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

RAY_BELOW_SURFACE_M = 0.003
RAY_LENGTH_M = 0.40
BOTTOM_GAP_MAX_M = 0.008
UP_AXIS_MAX_DEG = 15.0
RING_POINTS = 8

_BRANCHES: List[Dict[str, Any]] = []


def reset_branches() -> None:
    _BRANCHES.clear()


def branches() -> List[Dict[str, Any]]:
    return list(_BRANCHES)


MAX_LOG = 200  # the evaluator calls every step: keep the latest entries only


def _log(predicate: str, entities: Sequence[Any], branch: str, ok: bool, detail: Optional[Dict[str, Any]] = None) -> None:
    _BRANCHES.append({"predicate": predicate, "entities": [getattr(e, "name", str(e)) for e in entities], "branch": branch, "ok": bool(ok),
                      "detail": detail or {}})
    if len(_BRANCHES) > MAX_LOG:
        del _BRANCHES[: len(_BRANCHES) - MAX_LOG]


def decided_by() -> Dict[str, str]:
    """``{"<predicate>(<a>, <b>)": branch}`` of the latest evaluation of every under / ontop atom in the log — what a caller
    writes as ``binary_branch``."""
    out: Dict[str, str] = {}
    for e in _BRANCHES:
        if e["predicate"] in ("under", "ontop"):
            out[f"{e['predicate']}({', '.join(e['entities'])})"] = e["branch"]
    return out


def evaluate_predicate(predicate_cls, *entities) -> bool:
    """Drop-in for ``evaluate_bddl_predicate`` with the robust ``under`` / ``ontop`` re-checks (module docstring)."""
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate

    name = str(getattr(predicate_cls, "__name__", predicate_cls)).lower()
    ok = bool(evaluate_bddl_predicate(predicate_cls, *entities))
    if name not in ("under", "ontop") or len(entities) != 2 or any(e is None or not _is_object(e) for e in entities):
        _log(name, entities, "omnigibson", ok)
        return ok
    if ok:
        _log(name, entities, "omnigibson", True)
        return True
    # under(a, b): a is under b -> a supports b; ontop(a, b): a rests on b -> b supports a
    lower, upper = (entities[0], entities[1]) if name == "under" else (entities[1], entities[0])
    try:
        r = support_rays(lower, upper)
    except Exception as e:  # noqa: BLE001 - a raycast failure must not turn into a verdict
        r = {"ok": False, "error": repr(e)}
    if r.get("ok"):
        _log(name, entities, "rays", True, r)
        return True
    try:
        g = support_geometry(lower, upper)
    except Exception as e:  # noqa: BLE001
        g = {"ok": False, "error": repr(e)}
    if g.get("ok"):
        _log(name, entities, "geometry", True, g)
        return True
    _log(name, entities, "none", False, {"rays": r, "geometry": g})
    return False


# ----------------------------------------------------------------------------------------------------------------
# branches
# ----------------------------------------------------------------------------------------------------------------


def _is_object(e: Any) -> bool:
    return hasattr(e, "links") and hasattr(e, "get_position_orientation")


def _link_paths(obj: Any) -> List[str]:
    return [link.prim_path for link in getattr(obj, "links", {}).values()]


def _aabb(obj: Any) -> Tuple[th.Tensor, th.Tensor]:
    lo, hi = obj.aabb
    return th.as_tensor(lo).float(), th.as_tensor(hi).float()


def _footprint_points(upper: Any) -> List[Tuple[float, float]]:
    """The resting object's footprint centre and a ring at half its footprint radius (world xy)."""
    lo, hi = _aabb(upper)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    r = 0.25 * float(min(hi[0] - lo[0], hi[1] - lo[1]))  # half the footprint radius
    pts = [(cx, cy)]
    for k in range(RING_POINTS):
        a = 2 * math.pi * k / RING_POINTS
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _surface_under(lower: Any, upper: Any, x: float, y: float) -> Optional[float]:
    """z of the supporting object's surface under world (x, y): a downward ray from above its AABB top with the resting
    object's bodies ignored; None when the first hit is not the supporting object."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_l, hi_l = _aabb(lower)
    start = [x, y, float(hi_l[2]) + 0.05]
    end = [x, y, float(lo_l[2]) - 0.02]
    hits = raytest_batch([start], [end], only_closest=True, ignore_bodies=_link_paths(upper) or None)
    h = hits[0] if hits else {}
    if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(tuple(_link_paths(lower))):
        return None
    return float(h["position"][2])


def support_rays(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 2: probe rays from just below the supporting surface up into the resting object across its contact patch."""
    from omnigibson.utils.sampling_utils import raytest_batch

    upper_paths = tuple(_link_paths(upper))
    lower_paths = _link_paths(lower)
    starts, ends = [], []
    n_surface = 0
    for x, y in _footprint_points(upper):
        z = _surface_under(lower, upper, x, y)
        if z is None:
            continue
        n_surface += 1
        starts.append([x, y, z - RAY_BELOW_SURFACE_M])
        ends.append([x, y, z - RAY_BELOW_SURFACE_M + RAY_LENGTH_M])
    if not starts:
        return {"ok": False, "n_rays": 0, "n_surface": 0, "why": "no supporting surface under the footprint"}
    hits = raytest_batch(starts, ends, only_closest=True, ignore_bodies=lower_paths or None)
    n_hit = sum(1 for h in hits if h.get("hit") and str(h.get("rigidBody", "")).startswith(upper_paths))
    return {"ok": n_hit > 0, "n_rays": len(starts), "n_surface": n_surface, "n_hits_on_upper": n_hit}


def support_geometry(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 3: bottom-to-surface gap under the footprint centre, footprint centre inside the supporting top, up axis."""
    lo_u, hi_u = _aabb(upper)
    lo_l, hi_l = _aabb(lower)
    cx, cy = float((lo_u[0] + hi_u[0]) / 2), float((lo_u[1] + hi_u[1]) / 2)
    z_surface = _surface_under(lower, upper, cx, cy)
    if z_surface is None:
        return {"ok": False, "why": "no supporting surface under the footprint centre"}
    gap = float(lo_u[2]) - z_surface
    inside = bool(lo_l[0] <= cx <= hi_l[0] and lo_l[1] <= cy <= hi_l[1])
    _, q = upper.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the rotated local z axis
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    ok = abs(gap) <= BOTTOM_GAP_MAX_M and inside and tilt <= UP_AXIS_MAX_DEG
    return {"ok": ok, "bottom_gap_mm": round(gap * 1000, 1), "centre_inside_support": inside, "tilt_deg": round(tilt, 1)}
