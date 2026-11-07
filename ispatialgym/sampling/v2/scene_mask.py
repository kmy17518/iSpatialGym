"""The band view's scene mask (of-v2maps 2026-11-07, owner prompt "mask the band view by the real scene so F stays honest"; opt-in).

The reach maps are computed on a flat synthetic slab at a height anchor and hold ~3x the legacy bands' feasible cells; the pairs the real
scene rules out die later, at the draw-time support check or the exact IK. The mask removes the (cell, yaw, grasp) pairs that are
CERTAINLY infeasible from geometry fixed by the cell, the yaw and the grasp alone (never anything that depends on the IK branch):

``support``   the object's footprint at (cell, yaw) off the support's free raster (``top.polygon_supported``, margin 0)
``height``    the local top height under the cell (``NB.z_at``) more than :data:`HEIGHT_TOL_M` from the map's anchor height
``trav``      the stance's base outside the eroded traversability map (``kit.trav_free(eroded=True)``: OmniGibson's ``_erode_trav_map``)
``static``    the object's hull points at the pose inside a static scene obstacle (the support and the task objects out)
``hand``      the eef-fixed links (gripper / fingers at the grasp's pre-shape / wrist camera / arm_link7: rigid in the eef frame) at the
              grasp (support out, as the exact IK's last pose), at the pre-grasp and along the straight corridor between them, against the
              static scene (task objects out) - one batched cuRobo sphere query (``kit.world_sphere_hits``), no IK
``features``  the same spheres against the SUPPORT alone, counting only spheres whose centre is above the top by more than their radius +
              :data:`FEATURE_CLEAR_M` (a lip, a back panel, a higher tier under the hand's descent; the top plane itself is the support policy's)
``frame``     the grasp's eef box out of the stance's head-camera frame (``IKChecker.eef_in_frame``) - only where the task applies that rule
              to the grasp (``frame_rule_at_grasp``)

Switches: ``ISPATIALGYM_MAP_SCENE_MASK=1`` (remove), ``ISPATIALGYM_MAP_SCENE_MASK_DIAG=1`` (evaluate and log, remove nothing; also the
pose-level outcome log and the false-negative check), ``ISPATIALGYM_MAP_SCENE_MASK_CATS`` (comma list, default all of :data:`CATEGORIES`).
The draw-time support check and the exact IK stay unchanged, as the guards."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G

log = logging.getLogger("ispatialgym.sampling.v2")

CATEGORIES = ("support", "height", "trav", "static", "hand", "features", "frame")
HEIGHT_TOL_M = 0.03
SPHERE_MARGIN_M = -0.003  # a sphere counts when it penetrates an obstacle by > 3 mm (robust to the sphere approximation)
FEATURE_CLEAR_M = 0.01
CORRIDOR_POINTS = 3
HULL_SPHERE_R = 0.004
FN_CHECK_MAX = 200  # masked pairs per process tested with the exact IK (DIAG)


def mask_on() -> bool:
    return os.environ.get("ISPATIALGYM_MAP_SCENE_MASK", "") == "1"


def diag_on() -> bool:
    return os.environ.get("ISPATIALGYM_MAP_SCENE_MASK_DIAG", "") == "1"


def categories() -> List[str]:
    v = os.environ.get("ISPATIALGYM_MAP_SCENE_MASK_CATS", "").strip()
    cats = [c.strip() for c in v.split(",") if c.strip()] if v else list(CATEGORIES)
    return [c for c in cats if c in CATEGORIES]


def active() -> bool:
    return mask_on() or diag_on()


def _count(s, key: str, n: int = 1) -> None:
    st = getattr(getattr(s, "kit", None), "stats", None)
    if isinstance(st, dict):
        st[key] = st.get(key, 0) + n


def _mat(pose) -> th.Tensor:
    T = th.eye(4)
    T[:3, :3] = C.quat_to_mat(G.t(pose[1]).float()).float()
    T[:3, 3] = G.t(pose[0]).float()
    return T


def hand_local(s, stance, arm: str, opening: Optional[float]) -> Tuple[th.Tensor, th.Tensor, List[str]]:
    """The eef-fixed links' collision spheres in the eef frame (centres (K, 3), radii (K,), links) at finger ``opening`` (m per finger;
    None = the stance configuration's); from one forward kinematics of the stance configuration, cached per (arm, opening)."""
    key = (arm, None if opening is None else round(float(opening), 4))
    cache = s.__dict__.setdefault("_scene_mask_hand", {})
    if key in cache:
        return cache[key]
    q = s.ikc.q_locked(stance).clone()
    if opening is not None:
        q[s.kit.robot.gripper_control_idx[arm]] = float(opening)
    sw, links = s.kit.backend.robot_spheres_world(q)
    sw = sw[0]
    want = lambda l: l.startswith(f"{arm}_gripper") or l.startswith(f"{arm}_realsense") or l == f"{arm}_arm_link7"  # noqa: E731
    idx = [i for i, l in enumerate(links) if want(l) and float(sw[i, 3]) > 0]
    T = _mat(s.kit.eef_pose(q, arm))
    Ti = th.linalg.inv(T)
    c = sw[idx, :3].float() @ Ti[:3, :3].T + Ti[:3, 3]
    out = (c, sw[idx, 3].float(), [links[i] for i in idx])
    cache[key] = out
    return out


def _spheres_at(local: Tuple[th.Tensor, th.Tensor, List[str]], poses: Sequence[Any]) -> th.Tensor:
    c, r, _ = local
    out = []
    for p in poses:
        T = _mat(p)
        w = c @ T[:3, :3].T + T[:3, 3]
        out.append(th.cat([w, r[:, None]], dim=1))
    return th.stack(out) if out else th.zeros(0, len(r), 4)


def evaluate(s, stance, top, arm: str, cells: th.Tensor, yaws: Sequence[float], grasps: Sequence[Any], pose_at: Callable, eef_of: Callable,
             support_obj: Any, hull: Optional[th.Tensor] = None, ignore: Sequence[Any] = (), anchor_z: Optional[float] = None, opening_of: Optional[Callable] = None,
             frame_at_grasp: bool = False, cats: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The categories for every (cell, yaw, grasp): ``{"masks": {cat: bool (N, Y, G)}, "any": bool (N, Y, G), "n": N*Y*G, "removed": {cat:
    count}, "s": seconds}``; True = certainly infeasible. ``pose_at(cell, yaw)`` the object pose the sampler would draw (its
    ``phase3_target.pose_at``), ``eef_of(g)`` T_obj->eef, ``ignore`` the task objects (out of the static scene), ``anchor_z`` the map's
    height above the floor (``height``), ``opening_of(g)`` the grasp's pre-shape (``tabletop.policy_opening_of``)."""
    from ispatialgym.sampling.v2 import ik as IKm
    from ispatialgym.sampling.v2 import neighbors as NB

    t0 = time.time()
    cats = list(cats or categories())
    cells = G.t(cells).float().reshape(-1, 2)
    N, Y, Gn = len(cells), len(yaws), len(grasps)
    masks = {c: th.zeros(N, Y, Gn, dtype=th.bool) for c in cats}
    poses = [[pose_at(cells[i], float(y)) for y in yaws] for i in range(N)]
    if "support" in masks:
        for i in range(N):
            for j in range(Y):
                fp = s.world.footprint(hull, poses[i][j]) if hull is not None else None
                if fp is not None and not top.polygon_supported(fp, margin=0.0):
                    masks["support"][i, j, :] = True
    if "height" in masks and anchor_z is not None:
        floor_z = float(getattr(s.world, "floor_z", 0.0))
        for i in range(N):
            if abs(NB.z_at(top, cells[i]) - floor_z - float(anchor_z)) > HEIGHT_TOL_M:
                masks["height"][i] = True
    if "trav" in masks:
        try:
            if not s.kit.trav_free(stance.base_xy, eroded=True):
                masks["trav"][:] = True
        except Exception:  # noqa: BLE001  (no trav map: the category is not measured)
            pass
    need_spheres = any(c in masks for c in ("static", "hand", "features"))
    if need_spheres:
        IK = IKm
        top_z = float(getattr(top, "z", 0.0))
        with s.ikc.ignoring(list(ignore) + ([support_obj] if support_obj is not None else [])):
            if "static" in masks and hull is not None:
                pts = [G.local_points_world(hull, poses[i][j]) for i in range(N) for j in range(Y)]
                sp = th.stack([th.cat([p, th.full((len(p), 1), HULL_SPHERE_R)], dim=1) for p in pts])
                hit = s.kit.world_sphere_hits(sp, SPHERE_MARGIN_M).any(dim=1).reshape(N, Y)
                masks["static"] |= hit[:, :, None].expand(N, Y, Gn)
            for k, g in enumerate(grasps):
                op = opening_of(g) if opening_of is not None else None
                loc = hand_local(s, stance, arm, op)
                eio = eef_of(g)
                eefs = [IK.grasp_pose(poses[i][j], eio) for i in range(N) for j in range(Y)]
                pres = [IK.pre_pose(e) for e in eefs]
                corr = [[(G.t(p[0]) + (G.t(e[0]) - G.t(p[0])) * (m + 1) / (CORRIDOR_POINTS + 1), G.t(e[1])) for m in range(CORRIDOR_POINTS)]
                        for e, p in zip(eefs, pres)]
                if "hand" in masks:
                    h_grasp = s.kit.world_sphere_hits(_spheres_at(loc, eefs), SPHERE_MARGIN_M).any(dim=1)
                    masks["hand"][:, :, k] |= h_grasp.reshape(N, Y)
                # pre-grasp and corridor: the support present (the exact IK keeps it for the pre-grasp): 'hand' counts the static scene
                # without it, 'features' the support alone above its top
                if "hand" in masks:
                    flat = [q for e, p, cs in zip(eefs, pres, corr) for q in [p] + cs]
                    hc = s.kit.world_sphere_hits(_spheres_at(loc, flat), SPHERE_MARGIN_M).any(dim=1).reshape(N * Y, -1).any(dim=1)
                    masks["hand"][:, :, k] |= hc.reshape(N, Y)
                masks.setdefault("_eefs", {})[k] = (eefs, pres, corr)
        if "features" in masks and support_obj is not None:
            others = [o for o in _scene_objects(s) if o is not support_obj]
            with s.ikc.ignoring(list(ignore) + others):
                for k, g in enumerate(grasps):
                    eefs, pres, corr = masks["_eefs"][k]
                    loc = hand_local(s, stance, arm, opening_of(g) if opening_of is not None else None)
                    flat = [q for e, p, cs in zip(eefs, pres, corr) for q in [e, p] + cs]
                    sp = _spheres_at(loc, flat)
                    hit = s.kit.world_sphere_hits(sp, SPHERE_MARGIN_M) & (sp[..., 2] - sp[..., 3] > top_z + FEATURE_CLEAR_M)
                    masks["features"][:, :, k] |= hit.any(dim=1).reshape(N * Y, -1).any(dim=1).reshape(N, Y)
    if "frame" in masks and frame_at_grasp:
        cam = s.camera_pose_at(stance)
        for k, g in enumerate(grasps):
            eio = eef_of(g)
            for i in range(N):
                for j in range(Y):
                    if not s.ikc.eef_in_frame(IKm.grasp_pose(poses[i][j], eio), cam):
                        masks["frame"][i, j, k] = True
    masks.pop("_eefs", None)
    anym = th.zeros(N, Y, Gn, dtype=th.bool)
    for m in masks.values():
        anym |= m
    return {"masks": masks, "any": anym, "n": N * Y * Gn, "removed": {c: int(m.sum()) for c, m in masks.items()}, "removed_any": int(anym.sum()),
            "s": round(time.time() - t0, 3)}


def _scene_objects(s) -> List[Any]:
    sc = getattr(getattr(s, "world", None), "scene", None)
    try:
        return list(sc.objects)
    except Exception:  # noqa: BLE001
        return []


def frame_rule_at_grasp(s) -> bool:
    """Whether the sampler's task applies the head-camera frame rule to the GRASP pose (disembedding, iors_large / _inside; not
    iors_small, which tests the place and pre-place only) - the frame category masks grasps only there (it must not remove what the
    sampler would accept)."""
    return bool(getattr(s, "FRAME_RULE_AT_GRASP", False))


def log_record(s, where: str, ev: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    rec = {"where": where, "n": ev["n"], "removed": ev["removed"], "removed_any": ev["removed_any"], "s": ev["s"], "mode": "mask" if mask_on() else "diag"}
    if extra:
        rec.update(extra)
    for c, n in ev["removed"].items():
        _count(s, f"scene_mask.{c}", n)
    _count(s, "scene_mask.pairs", ev["n"])
    _count(s, "scene_mask.removed_any", ev["removed_any"])
    log.info("SCENE_MASK %s", json.dumps(rec))
