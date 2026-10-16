"""BDDL predicate evaluation with geometric fallbacks for the predicates OmniGibson decides fragilely.

:func:`evaluate_predicate` is a drop-in for ``omnigibson.utils.bddl_utils.evaluate_bddl_predicate``. OmniGibson's
verdict is taken when it is True; when it is False:

* ``under`` / ``ontop``: rays across the resting object's contact patch (its footprint centre and a ring at half its
  footprint radius), started just below the supporting surface; then a geometric check (bottom within
  ``BOTTOM_GAP_MAX_M`` of the surface under the footprint centre, the centre inside the support's footprint, the resting
  object's up axis within ``UP_AXIS_MAX_DEG`` of vertical). OmniGibson shoots a single ray from one surface point, which
  misses physically correct placements depending on sub-millimetre contact.
* ``touching``: the two objects' convex hulls within ``TOUCHING_GAP_MAX_M``. OmniGibson's contact report flickers between
  resting objects with zero normal force.
* ``inside`` of a container without a fillable meta link (OmniGibson only tests those): the inner object's AABB centre
  inside the container's AABB with the container on both sides of it along at least ``INSIDE_MIN_AXES`` axes.
* ``pose_settled``: the configuration drift over the ``PoseSettled`` window alone. The velocity gate of OmniGibson's rule
  never passes for a body in resting contact or on a driven joint, whose solver residual reads 1-3 mm/s at micrometres
  of motion; the window size and drift thresholds are set by the task.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch as th

RAY_BELOW_SURFACE_M = 0.003
RAY_LENGTH_M = 0.40
BOTTOM_GAP_MAX_M = 0.008
UP_AXIS_MAX_DEG = 15.0
RING_POINTS = 8
TOUCHING_GAP_MAX_M = 0.003
INSIDE_MIN_AXES = 2
INSIDE_RAY_MARGIN_M = 0.05
CONTAINER_META_LINK_TYPES = ("fillable", "openfillable")


def evaluate_predicate(predicate_cls, *entities) -> bool:
    """Drop-in for ``evaluate_bddl_predicate`` with the fallbacks of the module docstring."""
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate

    if evaluate_bddl_predicate(predicate_cls, *entities):
        return True
    name = str(getattr(predicate_cls, "__name__", predicate_cls)).lower()
    name = {"posesettled": "pose_settled"}.get(name, name)
    if any(e is None or not _is_object(e) for e in entities):
        return False
    if name == "pose_settled" and len(entities) == 1:
        return _ok(settled_by_drift, entities[0])
    if len(entities) != 2:
        return False
    if name == "inside":
        return not has_container_link(entities[1]) and _ok(inside_rays, *entities)
    if name == "touching":
        return _ok(touching_geometry, *entities)
    if name in ("under", "ontop"):
        # under(a, b): a supports b; ontop(a, b): b supports a
        lower, upper = entities if name == "under" else entities[::-1]
        return _ok(support_rays, lower, upper) or _ok(support_geometry, lower, upper)
    return False


def _ok(check, *args) -> bool:
    try:
        return bool(check(*args).get("ok"))
    except Exception:  # noqa: BLE001 - a raycast or geometry failure is not a verdict
        return False


def _is_object(e: Any) -> bool:
    return hasattr(e, "links") and hasattr(e, "get_position_orientation")


def _qangle(a: th.Tensor, b: th.Tensor) -> float:
    d = abs(float(th.dot(a / th.linalg.norm(a), b / th.linalg.norm(b))))
    return 2.0 * math.acos(min(1.0, d))


def settled_by_drift(obj: Any) -> dict[str, Any]:
    """``pose_settled`` from the object's ``PoseSettled`` window: active, full (``n_steps`` samples) and every sample's root
    pose (and joint positions) within the state's tolerances of the window's first sample."""
    from omnigibson.object_states import PoseSettled

    st = obj.states.get(PoseSettled) if hasattr(obj.states, "get") else None
    if st is None or not getattr(st, "is_active", False):
        return {"ok": False}
    hist = list(getattr(st, "_history", []) or [])
    n_steps = int(getattr(st, "n_steps", len(hist)))
    if not hist or len(hist) < n_steps:
        return {"ok": False}
    _, anchor_pos, anchor_quat, anchor_jpos, _ = hist[0]
    max_pos = max_ori = max_joint = 0.0
    for _, pos, quat, jpos, _ in hist:
        max_pos = max(max_pos, float(th.linalg.norm(pos - anchor_pos)))
        max_ori = max(max_ori, _qangle(quat.double(), anchor_quat.double()))
        if anchor_jpos is not None and jpos is not None and jpos.numel() == anchor_jpos.numel() and jpos.numel() > 0:
            max_joint = max(max_joint, float(th.max(th.abs(jpos - anchor_jpos))))
    ok = (
        max_pos <= float(getattr(st, "_pos_tolerance", 0.01))
        and max_ori <= float(getattr(st, "_ori_tolerance", 0.05))
        and max_joint <= float(getattr(st, "_joint_tolerance", 0.05))
    )
    return {"ok": ok, "max_pos_drift_m": max_pos, "max_ori_drift_rad": max_ori, "max_joint_drift": max_joint}


def _boundary_points(obj: Any) -> th.Tensor:
    """World collision boundary points (the convex-hull vertices of every link's collision geometry)."""
    import omnigibson as og

    og.sim.sync_physx_to_fabric()  # the boundary points read the render-side transforms
    pts = [p for p in (link.collision_boundary_points_world for link in obj.links.values()) if p is not None and len(p)]
    return th.cat([th.as_tensor(p).float() for p in pts], dim=0) if pts else th.zeros(0, 3)


def hull_gap(points_a: th.Tensor, points_b: th.Tensor) -> float:
    """Separation of the convex hulls of two point sets (negative when they interpenetrate).

    The larger signed distance of one set's surface points to the other hull's half-spaces, minimised over the points,
    taken both ways: exact for a face-to-face contact, a lower bound for edge or corner contacts.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    hulls = {id(p): ConvexHull(p.numpy().astype(np.float64)) for p in (points_a, points_b)}

    def surface_points(pts: th.Tensor) -> th.Tensor:
        # vertices plus triangle centroids and edge midpoints: partially overlapping faces may have no vertex inside
        tri = pts[th.as_tensor(hulls[id(pts)].simplices, dtype=th.long)]
        extra = [tri.mean(dim=1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 0] + tri[:, 2]) / 2]
        return th.cat([pts, *extra], dim=0)

    def one_way(pts: th.Tensor, hull_pts: th.Tensor) -> float:
        eq = th.as_tensor(hulls[id(hull_pts)].equations).float()  # n . x + d <= 0 inside
        signed = surface_points(pts) @ eq[:, :3].T + eq[:, 3]
        return float(signed.max(dim=1).values.min())

    return min(one_way(points_a, points_b), one_way(points_b, points_a))


def touching_geometry(a: Any, b: Any) -> dict[str, Any]:
    pa, pb = _boundary_points(a), _boundary_points(b)
    if len(pa) < 4 or len(pb) < 4:
        return {"ok": False}
    gap = hull_gap(pa, pb)
    return {"ok": gap <= TOUCHING_GAP_MAX_M, "gap_m": gap}


def _link_paths(obj: Any) -> list[str]:
    return [link.prim_path for link in obj.links.values()]


def has_container_link(obj: Any) -> bool:
    """Whether ``obj`` has a fillable / openfillable meta link, the volume OmniGibson's ``Inside`` tests."""
    try:
        from omnigibson.object_states.contains import m as contains_m

        types = set(contains_m.CONTAINER_META_LINK_TYPES)
    except Exception:  # noqa: BLE001 - the macro moved in this OmniGibson version
        types = set(CONTAINER_META_LINK_TYPES)
    return any(
        getattr(link, "is_meta_link", False) and getattr(link, "meta_link_type", None) in types
        for link in obj.links.values()
    )


def inside_both_sides(sides: dict[str, Sequence[bool]], min_axes: int = INSIDE_MIN_AXES) -> tuple[bool, list[str]]:
    """Whether the container is on both sides of the inner object along at least ``min_axes`` axes."""
    axes = [k for k, v in sides.items() if len(v) == 2 and bool(v[0]) and bool(v[1])]
    return len(axes) >= int(min_axes), axes


def _aabb(obj: Any) -> tuple[th.Tensor, th.Tensor]:
    lo, hi = obj.aabb
    return th.as_tensor(lo).float(), th.as_tensor(hi).float()


def inside_rays(inner: Any, container: Any) -> dict[str, Any]:
    """Rays from the inner object's AABB centre up / down and along the container's own x / y axes (the inner object's
    bodies ignored), :func:`inside_both_sides` on the sides where a ray meets the container."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_i, hi_i = _aabb(inner)
    lo_c, hi_c = _aabb(container)
    c = (lo_i + hi_i) / 2
    if not bool(((c >= lo_c) & (c <= hi_c)).all()):
        return {"ok": False}
    _, q = container.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    ca, sa = math.cos(yaw), math.sin(yaw)
    reach = float(th.linalg.norm(hi_c - lo_c)) + INSIDE_RAY_MARGIN_M
    keys, starts, ends = [], [], []
    for k, a in (("z", (0.0, 0.0, 1.0)), ("x", (ca, sa, 0.0)), ("y", (-sa, ca, 0.0))):
        for side, sgn in enumerate((1.0, -1.0)):
            keys.append((k, side))
            starts.append([float(c[0]), float(c[1]), float(c[2])])
            ends.append([float(c[i]) + sgn * reach * a[i] for i in range(3)])
    hits = raytest_batch(starts, ends, only_closest=False, ignore_bodies=_link_paths(inner) or None)
    paths = tuple(_link_paths(container))
    sides: dict[str, list[bool]] = {"z": [False, False], "x": [False, False], "y": [False, False]}
    for (k, side), hs in zip(keys, hits, strict=True):
        hs = hs if isinstance(hs, list) else [hs]
        sides[k][side] = any(h.get("hit") and str(h.get("rigidBody", "")).startswith(paths) for h in hs)
    ok, axes = inside_both_sides(sides)
    return {"ok": ok, "axes_both_sides": axes}


def _footprint_points(upper: Any) -> list[tuple[float, float]]:
    """The resting object's footprint centre and a ring at half its footprint radius (world xy)."""
    lo, hi = _aabb(upper)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    r = 0.25 * float(min(hi[0] - lo[0], hi[1] - lo[1]))
    pts = [(cx, cy)]
    for k in range(RING_POINTS):
        a = 2 * math.pi * k / RING_POINTS
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _surface_under(lower: Any, upper: Any, x: float, y: float) -> float | None:
    """z of the supporting object's surface under world ``(x, y)`` (the resting object's bodies ignored), None when the
    first hit is not the supporting object."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_l, hi_l = _aabb(lower)
    start, end = [x, y, float(hi_l[2]) + 0.05], [x, y, float(lo_l[2]) - 0.02]
    hits = raytest_batch([start], [end], only_closest=True, ignore_bodies=_link_paths(upper) or None)
    h = hits[0] if hits else {}
    if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(tuple(_link_paths(lower))):
        return None
    return float(h["position"][2])


def support_rays(lower: Any, upper: Any) -> dict[str, Any]:
    """Probe rays from just below the supporting surface up into the resting object across its contact patch."""
    from omnigibson.utils.sampling_utils import raytest_batch

    starts, ends = [], []
    for x, y in _footprint_points(upper):
        z = _surface_under(lower, upper, x, y)
        if z is not None:
            starts.append([x, y, z - RAY_BELOW_SURFACE_M])
            ends.append([x, y, z - RAY_BELOW_SURFACE_M + RAY_LENGTH_M])
    if not starts:
        return {"ok": False}
    hits = raytest_batch(starts, ends, only_closest=True, ignore_bodies=_link_paths(lower) or None)
    upper_paths = tuple(_link_paths(upper))
    n_hit = sum(1 for h in hits if h.get("hit") and str(h.get("rigidBody", "")).startswith(upper_paths))
    return {"ok": n_hit > 0, "n_rays": len(starts), "n_hits": n_hit}


def support_geometry(lower: Any, upper: Any) -> dict[str, Any]:
    """Bottom-to-surface gap under the footprint centre, footprint centre inside the supporting top, up axis tilt."""
    lo_u, hi_u = _aabb(upper)
    lo_l, hi_l = _aabb(lower)
    cx, cy = float((lo_u[0] + hi_u[0]) / 2), float((lo_u[1] + hi_u[1]) / 2)
    z_surface = _surface_under(lower, upper, cx, cy)
    if z_surface is None:
        return {"ok": False}
    gap = float(lo_u[2]) - z_surface
    inside = bool(lo_l[0] <= cx <= hi_l[0] and lo_l[1] <= cy <= hi_l[1])
    _, q = upper.get_position_orientation()
    x, y, _z, _w = [float(v) for v in th.as_tensor(q).float()]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))  # local z vs world z
    return {"ok": abs(gap) <= BOTTOM_GAP_MAX_M and inside and tilt <= UP_AXIS_MAX_DEG, "gap_m": gap, "tilt_deg": tilt}
