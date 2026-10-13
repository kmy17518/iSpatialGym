"""An articulated scene object posed by the alias before the stances (owner 2026-10-12 ~16:05, inside_large_scale#02: the fridge's left
door open at init, ``generation.partner_joint_init``; the band CLI's ``--support-joints``): the joint request parsed and clamped, the
joints set, the links that moved, and what the moved links change for the geometry - the AABB without them (stance stand-offs), the
raster cells under them (an interior surface's sides) and their floor footprints (the path planner's map). Duck-typed on OmniGibson's
``EntityPrim`` (``joints`` / ``links`` / ``get_joint_positions`` / ``set_joint_positions`` / ``set_joint_velocities`` / ``keep_still``)."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch as th

#: a link counts as moved by the joint request when it travelled this far or turned this much
MOVED_LINK_M = 0.01
MOVED_LINK_RAD = 0.05
#: two joint requests are the same (band lookup) within this (rad / m)
SPEC_MATCH_TOL = 0.02

Box = Tuple[th.Tensor, th.Tensor]


def _t(x) -> th.Tensor:
    return x.float() if isinstance(x, th.Tensor) else th.as_tensor(x, dtype=th.float32)


def _quat_angle(q1, q2) -> float:
    """Rotation angle (rad) between two xyzw quaternions."""
    d = abs(float(th.dot(_t(q1) / th.linalg.norm(_t(q1)), _t(q2) / th.linalg.norm(_t(q2)))))
    return 2.0 * math.acos(min(1.0, d))


def parse_joint_spec(value: Any, joint_names: Sequence[str], limits: Optional[Mapping[str, Tuple[float, float]]] = None) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """``{joint: position}`` (rad / m) -> ``(spec, clamped)``: the positions, each clamped to its joint's ``limits`` (``clamped`` =
    ``{joint: {requested, applied}}`` for the ones that were). ``ValueError``: not a non-empty mapping, an unknown joint, a value that is
    not a finite number."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"expected a non-empty {{joint: position}} mapping, got {value!r}")
    names = list(joint_names)
    spec: Dict[str, float] = {}
    clamped: Dict[str, Dict[str, float]] = {}
    for name, v in value.items():
        if name not in names:
            raise ValueError(f"no joint {name!r} (joints {names})")
        try:
            q = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {v!r} is not a number") from None
        if not math.isfinite(q):
            raise ValueError(f"{name}: {v!r} is not finite")
        lo, hi = (limits or {}).get(name, (-math.inf, math.inf))
        q_in = min(max(q, float(lo)), float(hi))
        if q_in != q:
            clamped[name] = {"requested": q, "applied": q_in}
        spec[name] = q_in
    return spec, clamped


def joint_limits(obj) -> Dict[str, Tuple[float, float]]:
    """``{joint: (lower, upper)}`` of ``obj``'s single-DOF joints (the others are left out: no clamp)."""
    out = {}
    for name, j in obj.joints.items():
        try:
            out[name] = (float(j.lower_limit), float(j.upper_limit))
        except Exception:  # noqa: BLE001 - a multi-DOF joint asserts
            continue
    return out


def link_poses(obj) -> Dict[str, Tuple[th.Tensor, th.Tensor]]:
    return {name: tuple(x.clone().float() for x in link.get_position_orientation()) for name, link in obj.links.items()}


def moved_links(before: Mapping[str, Tuple[th.Tensor, th.Tensor]], after: Mapping[str, Tuple[th.Tensor, th.Tensor]], dist_m: float = MOVED_LINK_M,
                angle_rad: float = MOVED_LINK_RAD) -> List[str]:
    """The links whose pose changed between two :func:`link_poses` readings by more than ``dist_m`` or ``angle_rad``."""
    out = []
    for name, (p1, q1) in after.items():
        if name not in before:
            continue
        p0, q0 = before[name]
        if float(th.linalg.norm(p1 - p0)) > dist_m or _quat_angle(q1, q0) > angle_rad:
            out.append(name)
    return out


def set_joints(obj, spec: Mapping[str, float]) -> th.Tensor:
    """Set the joints of ``spec`` (the others keep their values), velocities zero, the object held still. Returns the joint vector set."""
    names = list(obj.joints)
    q = obj.get_joint_positions().clone()
    for name, v in spec.items():
        q[names.index(name)] = float(v)
    obj.set_joint_positions(q)
    obj.set_joint_velocities(th.zeros_like(q))
    obj.keep_still()
    return q


def joint_dict(obj) -> Dict[str, float]:
    q = obj.get_joint_positions()
    return {name: float(q[i]) for i, name in enumerate(obj.joints)}


def link_piece_aabbs(obj, link_names: Iterable[str]) -> List[Box]:
    """World AABBs of the collision meshes of ``obj``'s links ``link_names`` (the link's own AABB when its meshes cannot be read)."""
    from omnigibson.utils import transform_utils as T

    out: List[Box] = []
    for name in link_names:
        link = obj.links[name]
        boxes = []
        try:
            tf = link.scaled_transform
            for mesh in (link.collision_meshes or {}).values():
                pts = mesh.points_in_parent_frame
                if pts is not None and len(pts):
                    w = T.transform_points(pts, tf)
                    boxes.append((w.min(dim=0).values.float(), w.max(dim=0).values.float()))
        except Exception:  # noqa: BLE001 - a mesh type without readable points
            boxes = []
        if not boxes:
            lo, hi = link.aabb
            boxes = [(th.as_tensor(lo, dtype=th.float32), th.as_tensor(hi, dtype=th.float32))]
        out += boxes
    return out


def union_box(boxes: Sequence[Box]) -> Optional[Box]:
    if not boxes:
        return None
    return th.stack([_t(b[0]).float() for b in boxes]).min(dim=0).values, th.stack([_t(b[1]).float() for b in boxes]).max(dim=0).values


def static_aabb(obj, moved: Sequence[str]) -> Box:
    """``obj``'s AABB without its ``moved`` links (the open door left out of a fridge's stand-off): the union of its other links' collision
    pieces; ``obj.aabb`` when nothing moved or no other link has geometry."""
    lo, hi = obj.aabb
    if not moved:
        return _t(lo).float().clone(), _t(hi).float().clone()
    box = union_box(link_piece_aabbs(obj, [n for n in obj.links if n not in set(moved)]))
    return box if box is not None else (_t(lo).float().clone(), _t(hi).float().clone())


def mask_out_boxes(top, boxes: Sequence[Box], pad: float = 0.0) -> int:
    """Clear the cells of a raster (``SupportTop`` / ``SupportInterior``: ``xs`` / ``ys`` / ``mask``) whose centre lies inside the floor
    projection of any of ``boxes`` grown by ``pad``. Returns the number of cells cleared."""
    if not boxes:
        return 0
    gx, gy = th.meshgrid(_t(top.xs).float(), _t(top.ys).float(), indexing="ij")
    inside = th.zeros_like(top.mask, dtype=th.bool)
    for lo, hi in boxes:
        lo, hi = _t(lo).float(), _t(hi).float()
        inside |= (gx >= float(lo[0]) - pad) & (gx <= float(hi[0]) + pad) & (gy >= float(lo[1]) - pad) & (gy <= float(hi[1]) + pad)
    hit = top.mask.bool() & inside
    n = int(hit.sum())
    if n:
        top.mask = top.mask.bool() & ~inside
    return n


def footprints(boxes: Sequence[Box], margin: float = 0.02) -> List[List[List[float]]]:
    """Floor footprints (counter-clockwise world xy corners) of ``boxes`` grown by ``margin`` - ``plan_path``'s ``blocked`` polygons."""
    out = []
    for lo, hi in boxes:
        x0, y0, x1, y1 = float(lo[0]) - margin, float(lo[1]) - margin, float(hi[0]) + margin, float(hi[1]) + margin
        out.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return out


def specs_match(a: Optional[Mapping[str, float]], b: Optional[Mapping[str, float]], tol: float = SPEC_MATCH_TOL) -> bool:
    """Two joint requests are the same: both empty, or the same joints within ``tol``."""
    a, b = dict(a or {}), dict(b or {})
    if set(a) != set(b):
        return False
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in a)
