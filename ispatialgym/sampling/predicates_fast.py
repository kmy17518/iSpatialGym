"""The ``under`` / ``ontop`` robust re-checks of :mod:`ispatialgym.sampling.predicates` with each AABB computed once per
check (opt-in: ``ISPATIALGYM_FAST_SUPPORT=1``; ``=check`` runs both versions on every call, returns the reference's
result and prints ``SUPPORT_FAST_CHECK`` lines with the call / mismatch counts).

``predicates.support_rays`` finds the supporting surface under each of its nine footprint points with
``_surface_under``, which recomputes the supporting object's AABB (every link's collision points transformed to the
world) for every point, and ``support_geometry`` computes it twice more: 13 AABBs per atom when both run (block
design: 189k AABB computations, a third of its goal check). Here each object's AABB and link paths are computed once
per check and the nine downward rays go to one ``raytest_batch`` call (which casts them one by one, as before). The
rays, their order, their ignore lists and every comparison are the reference's, and nothing changes the scene between
them, so the results are the same by construction; the ``check`` mode verifies it on real episodes.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch as th

_COUNTS = {"calls": 0, "mismatches": 0, "ref_s": 0.0, "fast_s": 0.0}


def _surface_hits(P, lower_paths: List[str], upper_paths: List[str], lo_l: th.Tensor, hi_l: th.Tensor,
                  points: List[Tuple[float, float]]) -> List[Optional[float]]:
    """``P._surface_under`` for every point with the supporting object's AABB / link paths given: z or None per point."""
    from omnigibson.utils.sampling_utils import raytest_batch

    z_top, z_bot = float(hi_l[2]) + 0.05, float(lo_l[2]) - 0.02
    hits = raytest_batch([[x, y, z_top] for x, y in points], [[x, y, z_bot] for x, y in points], only_closest=True,
                         ignore_bodies=upper_paths or None)
    prefix = tuple(lower_paths)
    out = []
    for h in hits:
        if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(prefix):
            out.append(None)
        else:
            out.append(float(h["position"][2]))
    return out


def support_rays(P, lower: Any, upper: Any) -> Dict[str, Any]:
    from omnigibson.utils.sampling_utils import raytest_batch

    upper_list = P._link_paths(upper)
    lower_list = P._link_paths(lower)
    lo_l, hi_l = P._aabb(lower)
    points = P._footprint_points(upper)
    starts, ends = [], []
    n_surface = 0
    for (x, y), z in zip(points, _surface_hits(P, lower_list, upper_list, lo_l, hi_l, points)):
        if z is None:
            continue
        n_surface += 1
        starts.append([x, y, z - P.RAY_BELOW_SURFACE_M])
        ends.append([x, y, z - P.RAY_BELOW_SURFACE_M + P.RAY_LENGTH_M])
    if not starts:
        return {"ok": False, "n_rays": 0, "n_surface": 0, "why": "no supporting surface under the footprint"}
    hits = raytest_batch(starts, ends, only_closest=True, ignore_bodies=lower_list or None)
    upper_paths = tuple(upper_list)
    n_hit = sum(1 for h in hits if h.get("hit") and str(h.get("rigidBody", "")).startswith(upper_paths))
    return {"ok": n_hit > 0, "n_rays": len(starts), "n_surface": n_surface, "n_hits_on_upper": n_hit}


def support_geometry(P, lower: Any, upper: Any) -> Dict[str, Any]:
    lo_u, hi_u = P._aabb(upper)
    lo_l, hi_l = P._aabb(lower)
    cx, cy = float((lo_u[0] + hi_u[0]) / 2), float((lo_u[1] + hi_u[1]) / 2)
    z_surface = _surface_hits(P, P._link_paths(lower), P._link_paths(upper), lo_l, hi_l, [(cx, cy)])[0]
    if z_surface is None:
        return {"ok": False, "why": "no supporting surface under the footprint centre"}
    gap = float(lo_u[2]) - z_surface
    inside = bool(lo_l[0] <= cx <= hi_l[0] and lo_l[1] <= cy <= hi_l[1])
    _, q = upper.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    ok = abs(gap) <= P.BOTTOM_GAP_MAX_M and inside and tilt <= P.UP_AXIS_MAX_DEG
    return {"ok": ok, "bottom_gap_mm": round(gap * 1000, 1), "centre_inside_support": inside, "tilt_deg": round(tilt, 1)}


def _checked(name, ref_fn, fast_fn):
    def f(lower, upper):
        t0 = time.perf_counter()
        ref = ref_fn(lower, upper)
        t1 = time.perf_counter()
        try:
            got = fast_fn(lower, upper)
        except Exception as e:  # noqa: BLE001
            got = {"error": repr(e)}
        _COUNTS["ref_s"] += t1 - t0
        _COUNTS["fast_s"] += time.perf_counter() - t1
        _COUNTS["calls"] += 1
        if got != ref:
            _COUNTS["mismatches"] += 1
            if _COUNTS["mismatches"] <= 20:
                print(f"SUPPORT_FAST_CHECK mismatch {name}: reference {ref!r} fast {got!r}", flush=True)
        if _COUNTS["calls"] % 1000 == 0:
            print(f"SUPPORT_FAST_CHECK calls {_COUNTS['calls']} mismatches {_COUNTS['mismatches']} reference_s {_COUNTS['ref_s']:.2f} "
                  f"fast_s {_COUNTS['fast_s']:.2f}", flush=True)
        return ref

    return f


def install(P, check: bool = False) -> None:
    """Replace ``P.support_rays`` / ``P.support_geometry`` (P = the predicates module) with these versions, or with
    checking wrappers that return the reference's result."""
    ref_rays, ref_geom = P.support_rays, P.support_geometry

    def fast_rays(lower, upper):
        return support_rays(P, lower, upper)

    def fast_geom(lower, upper):
        return support_geometry(P, lower, upper)

    if check:
        P.support_rays = _checked("support_rays", ref_rays, fast_rays)
        P.support_geometry = _checked("support_geometry", ref_geom, fast_geom)
    else:
        P.support_rays = fast_rays
        P.support_geometry = fast_geom
