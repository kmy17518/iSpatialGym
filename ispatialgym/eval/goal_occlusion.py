"""Occlusion of the task objects in the goal image: the head camera's frame at the settled end of an episode.

A demo is not valid when one of its target / partner objects (scene names ``target_*``, ``partner_*``, ``targets_*``) is at
least :data:`THRESHOLD` hidden in that frame; supports, distractors and every other object are not judged, and camera
relocalization (no target / partner) is exempt. The target of an inside goal is measured but not judged (owner 2026-10-12, A3):
an opaque container hides it by design, so its container - the partner - is what has to be visible. Built-in fixtures (scene
names ``dishwasher_*`` / ``door_*``) count only what the robot covers of their moving panel: the counter above a dishwasher and
a door's frame stand in front of the panel in every frame.

Measures (all ratios of the in-image silhouette; an object outside the image is not judged):

* :func:`box_occlusion` - the screen of ``isg_home/tmp/occlusion/occ.py``: the object's box (asset ``bbox_size`` and
  ``base_link_offset`` times its scale, at its pose) rasterized in the camera; a silhouette pixel is occluded when the
  depth frame is more than :data:`DEPTH_TOL_M` in front of the box's entry depth along that pixel's ray.
* :func:`panel_occlusion` - ``seg_check.py``'s fixture screen: the panel's world box, depth with and without the robot;
  a panel pixel is occluded when the robot's render is more than :data:`DEPTH_TOL_M` nearer.

The sampler applies them before any demo is recorded, to the settled goal poses with the robot where the program check
ends (``SamplerV2.check_goal_occlusion``: an occluded instance is never written); the oracle applies them to the live
last frame of every demo (``Executor`` demo gate ``goal_occlusion``: an occluded demo is not valid). No simulator import
here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

#: a target / partner hidden at least this much makes the demo invalid
THRESHOLD = 0.5
#: depth tolerance of the box / panel screens (m)
DEPTH_TOL_M = 0.02
ROLE_PREFIXES = ("target_", "partner_", "targets_")
FIXTURE_CATEGORIES = ("dishwasher", "door")
#: subtasks without a target or partner
EXEMPT_SUBTASK_PREFIXES = ("camera_relocalization",)
#: subtasks whose goal puts the target inside its partner: the target is not judged, the partner (the container) is (inside#02's candy
#: was 100 % hidden in its jar in every check)
CONTAINED_TARGET_SUBTASK_PREFIXES = ("inter_object_relation_satisfaction-inside_small_scale",)


def enabled(environ=os.environ) -> bool:
    """``ISPATIALGYM_GOAL_OCCLUSION=off`` switches both the sampler's prediction and the oracle's gate off."""
    return environ.get("ISPATIALGYM_GOAL_OCCLUSION", "on").lower() not in ("off", "0", "false", "no")


def role(scene_name: str) -> Optional[str]:
    """``"object"`` for a judged target / partner, ``"fixture"`` for a built-in fixture, None otherwise."""
    name = str(scene_name or "")
    if name.startswith(ROLE_PREFIXES):
        return "object"
    if name.split("_")[0] in FIXTURE_CATEGORIES:
        return "fixture"
    return None


def exempt(subtask_or_variant: Optional[str]) -> bool:
    return str(subtask_or_variant or "").startswith(EXEMPT_SUBTASK_PREFIXES)


def contained_target(subtask_or_variant: Optional[str]) -> bool:
    """Whether the goal of the subtask (or a variant of it) puts the target inside its partner (:data:`CONTAINED_TARGET_SUBTASK_PREFIXES`)."""
    return str(subtask_or_variant or "").startswith(CONTAINED_TARGET_SUBTASK_PREFIXES)


def quat_xyzw_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


@lru_cache(maxsize=8)
def _rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Per pixel (row-major) the ray in the camera frame (-z forward, +y up) scaled so that its parameter is the depth
    along the optical axis - the convention of OmniGibson's ``depth_linear``."""
    vs, us = np.mgrid[0:height, 0:width]
    return np.stack([(us.ravel() - cx) / fx, -(vs.ravel() - cy) / fy, -np.ones(height * width)], 1)


def _clean_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    return np.where(np.isfinite(d) & (d > 0), d, np.inf)


def box_occlusion(depth: np.ndarray, cam_pose: Tuple[Iterable[float], Iterable[float]], obj_pose: Tuple[Iterable[float], Iterable[float]],
                  size: Iterable[float], offset: Iterable[float], intrinsics: Tuple[int, int, float, float, float, float],
                  tol: float = DEPTH_TOL_M) -> Dict[str, Any]:
    """The box screen of one object (poses are ``(position, quaternion xyzw)`` in the world, ``size`` / ``offset`` already
    scaled). ``{"status": "ok", "in_frame_px", "occluded", "box_depth_m"}`` or ``{"status": "out_of_frame"}``."""
    width, height = intrinsics[0], intrinsics[1]
    tc, Rc = np.asarray(cam_pose[0], float), quat_xyzw_to_matrix(cam_pose[1])
    to, Ro = np.asarray(obj_pose[0], float), quat_xyzw_to_matrix(obj_pose[1])
    half = np.asarray(size, float) / 2.0
    centre = to + Ro @ np.asarray(offset, float)
    d_b = (_rays(*intrinsics) @ Rc.T) @ Ro
    o_b = Ro.T @ (tc - centre)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o_b) / d_b
        t2 = (half - o_b) / d_b
    tnear = np.nanmax(np.minimum(t1, t2), 1)
    tfar = np.nanmin(np.maximum(t1, t2), 1)
    hit = (tnear <= tfar) & (tfar > 0)
    n_hit = int(hit.sum())
    if n_hit == 0:
        return {"status": "out_of_frame"}
    d = _clean_depth(depth)
    if d.size != width * height:
        raise ValueError(f"depth has {d.size} pixels, the intrinsics {width}x{height}")
    entry = np.maximum(tnear[hit], 0.0)
    occ = d[hit] < entry - tol
    return {"status": "ok", "in_frame_px": n_hit, "occluded": round(float(occ.mean()), 4), "box_depth_m": round(float(np.median(entry)), 3)}


def panel_occlusion(depth_with: np.ndarray, depth_without: np.ndarray, cam_pose: Tuple[Iterable[float], Iterable[float]], lo: Iterable[float],
                    hi: Iterable[float], intrinsics: Tuple[int, int, float, float, float, float], tol: float = DEPTH_TOL_M) -> Dict[str, Any]:
    """The fixture screen: the panel's world-aligned box ``[lo, hi]``; a pixel shows the panel when, without the robot, the
    first surface along its ray lies inside that box, and is occluded when the render with the robot is more than ``tol``
    nearer. ``{"status": "ok", "panel_px", "occluded_px", "occluded"}`` or ``{"status": "panel_not_in_view"}``."""
    tc, Rc = np.asarray(cam_pose[0], float), quat_xyzw_to_matrix(cam_pose[1])
    d = _rays(*intrinsics) @ Rc.T
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - tc) / d
        t2 = (hi - tc) / d
    tnear = np.maximum(np.nanmax(np.minimum(t1, t2), 1), 0.0)
    tfar = np.nanmin(np.maximum(t1, t2), 1)
    hit = (tnear <= tfar) & (tfar > 0)
    dw, dr = _clean_depth(depth_without), _clean_depth(depth_with)
    panel = hit & (dw >= tnear - tol) & (dw <= tfar + tol)
    n_panel = int(panel.sum())
    if n_panel == 0:
        return {"status": "panel_not_in_view"}
    n_occ = int((panel & (dr < dw - tol)).sum())
    return {"status": "ok", "panel_px": n_panel, "occluded_px": n_occ, "occluded": round(n_occ / n_panel, 4)}


def intrinsics_of(camera_model: Any) -> Tuple[int, int, float, float, float, float]:
    """``(width, height, fx, fy, cx, cy)`` of a pinhole camera model (``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy``)."""
    m = camera_model
    return int(m.width), int(m.height), float(m.fx), float(m.fy), float(m.cx), float(m.cy)


def _vec(v: Any) -> np.ndarray:
    return np.asarray(v.tolist() if hasattr(v, "tolist") else v, dtype=float).reshape(-1)


def object_box(obj: Any) -> Tuple[Tuple[list, list], list, list]:
    """``(pose, size, offset)`` of a dataset object's box: its world pose, asset ``bbox_size`` (``native_bbox``) and
    ``base_link_offset`` times its scale."""
    p, q = obj.get_position_orientation()
    scale = _vec(obj.scale)
    return (_vec(p).tolist(), _vec(q).tolist()), (_vec(obj.native_bbox) * scale).tolist(), (_vec(obj.base_link_offset) * scale).tolist()


def panel_aabb(obj: Any, revolute: Any) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], list]:
    """The world box of the links a fixture's revolute joints move (the door leaf, the dishwasher front) and their names;
    ``revolute`` is the simulator's revolute joint type."""
    paths = {j.body1 for j in obj.joints.values() if j.joint_type == revolute}
    links = {name: link for name, link in obj.links.items() if link.prim_path in paths}
    if not links:
        return None, []
    lo = np.min([_vec(link.aabb[0]) for link in links.values()], 0)
    hi = np.max([_vec(link.aabb[1]) for link in links.values()], 0)
    return (lo, hi), sorted(links)


def screen_scene(depth: np.ndarray, cam_pose: Tuple[Iterable[float], Iterable[float]], intrinsics: Tuple[int, int, float, float, float, float],
                 judged: Iterable[Any], depth_without_robot: Optional[np.ndarray] = None, revolute: Any = None,
                 subtask: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Per judged object (by :func:`role`): the box screen of a target / partner, the panel screen of a fixture (needs the
    depth rendered with the robot hidden). For an inside goal (:func:`contained_target` of ``subtask``, a subtask id or a variant)
    the target's record carries ``"judged": False``: measured, left out of :func:`worst`."""
    out: Dict[str, Dict[str, Any]] = {}
    for obj in judged:
        kind = role(obj.name)
        if kind == "object":
            pose, size, offset = object_box(obj)
            out[obj.name] = box_occlusion(depth, cam_pose, pose, size, offset, intrinsics)
            if obj.name.startswith("target_") and contained_target(subtask):
                out[obj.name].update({"judged": False, "why": "inside goal: the container hides the target by design, the partner is judged"})
        elif kind == "fixture":
            box, part = panel_aabb(obj, revolute) if depth_without_robot is not None else (None, [])
            if depth_without_robot is None:
                out[obj.name] = {"status": "not_measured"}
            elif box is None:
                out[obj.name] = {"status": "no_panel"}
            else:
                out[obj.name] = {**panel_occlusion(depth, depth_without_robot, cam_pose, box[0], box[1], intrinsics), "part": part}
    return out


def worst(objects: Dict[str, Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    """The largest occluded share among the judged objects and its scene name (None when nothing was judged)."""
    best, who = None, None
    for name, rec in (objects or {}).items():
        v = rec.get("occluded") if rec.get("status") == "ok" and rec.get("judged", True) else None
        if v is not None and (best is None or v > best):
            best, who = float(v), name
    return best, who


def report(objects: Dict[str, Dict[str, Any]], method: str, threshold: float = THRESHOLD, **extra) -> Dict[str, Any]:
    """The record kept in a demo's meta.json / an instance's summary: per object, the largest share, and ``occluded``
    (whether it reaches ``threshold``)."""
    share, who = worst(objects)
    return {"method": method, "threshold": threshold, "objects": objects, "max_occluded": share, "worst": who,
            "occluded": share is not None and share >= threshold, **extra}
