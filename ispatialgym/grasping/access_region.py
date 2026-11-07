"""The ACCESSIBLE REGION of a grasp-library entry used at an INTERIOR goal (owner decision 2026-11-07, i7-overhang: a lean or stable rest in a
compartment / shelf / basin - the interior map kind). A candidate-generation constraint, not a relaxed gate: the v2 gates and the physics
validation stay as they are.

The region is derived from the goal geometry the entry is built for (``interiors.InteriorSpec``: its open side, its boards / walls /
headroom) and from the gates that judge the grasp there:

* reachable from the OPEN FRONT (the compartment's front plane, -x of the rest frame: the robot's side) - the approach points into the
  compartment (``approach_min_cos`` with +x) and the contact lies within the pads' reach of the object's front face (``front_depth_m`` =
  the pads' span from the fingertip, ``FINGERTIP_LENGTH - PAD_Z_RANGE[0]``);
* the hand's collision spheres (fingers included) clear every board of the goal - the lean board of a lean rest, the opposite side wall,
  the board above - by the obstacle gate's margin (``contact_policy.OBSTACLE_MARGIN_M``) at the opening the validator's placement IK checks
  them with (``clearance_opening_q`` = the fully open gripper, ``validate_tabletop.TabletopWorld.plan``'s configuration) with the model's
  sphere buffer.

Measured on hardback__bqvvvx__bu at 16.7 deg (``$OFH/overhang_20261107/board_clear.jsonl``, 87 candidates of 7 runs): every physics-valid
lean grasp has those spheres +0.3..+5.0 mm clear of the board, every placement failure -5.8..-9.2 mm (``no_placement:board:<arm>_gripper_finger_
link1``): the gate is that sphere clearance, and the clearing spine pinches sit >= 6.4 cm below the book's centre.

The candidates of such an entry are restricted to the region and RANKED by that board clearance (``candidates._generate``), so the
representatives the validator tests are the accessible pinches. The region's identity (:func:`region_identity`, digest / tag) is in the
job, the entry id (``__<tag>``), the library and its identity (``schema``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import schema as S

SCHEMA = "isg-access-region/1"
TAG_PREFIX = "ar"
#: the approach must point into the compartment from its open front within this angle
APPROACH_MAX_DEG = 45.0
#: the model's cuRobo sphere buffer (``collision_sphere_buffer``)
SPHERE_BUFFER_M = 0.002


def front_depth_m() -> float:
    from ispatialgym.robot import conventions as C

    return round(float(C.FINGERTIP_LENGTH) - float(C.PAD_Z_RANGE[0]), 4)


def region_identity(spec) -> Dict[str, Any]:
    """The region rule of an entry used in the compartment ``spec`` (``interiors.InteriorSpec``) - every factor that decides it."""
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.robot import conventions as C

    d = {"schema": SCHEMA, "rule": "open-front pinch, every hand sphere clear of the goal's boards at the placement opening",
         "interior": spec.name, "interior_digest": spec.digest(), "open_side": "front", "frame": "rest frame: +x into the compartment (the open front at -x)",
         "approach_min_cos": round(math.cos(math.radians(APPROACH_MAX_DEG)), 6), "front_depth_m": front_depth_m(),
         "clearance_opening_q": float(C.FINGER_OPEN), "margin_m": float(CP.OBSTACLE_MARGIN_M), "sphere_buffer_m": SPHERE_BUFFER_M,
         "boxes": ["the lean board (a lean rest)", "the opposite side wall", "the board above"],
         "width_m": round(float(spec.width_m), 4), "headroom_m": round(float(spec.headroom_m), 4), "wall_thickness_m": round(float(spec.wall_thickness_m), 4),
         "top_thickness_m": round(float(spec.top_thickness_m), 4)}
    d["digest"] = S.digest({k: v for k, v in d.items() if k != "digest"}, 12)
    d["tag"] = f"{TAG_PREFIX}{d['digest'][:6]}"
    return d


def region_of_job(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The access region of a job (``access_region``) - lean jobs only (the interior goals of the owner's rule built so far); else None."""
    if not job or job.get("rest_kind") != "lean":
        return None
    r = job.get("access_region")
    return r if isinstance(r, dict) and r.get("digest") else None


def goal_boxes(region: Dict[str, Any], lean_rest: Optional[Dict[str, Any]], z_top: float, side_sign: float):
    """The goal's boards as ``contact_policy.Support`` boxes in the REST frame (the root at the origin, the support top at ``z_top``): the lean
    board (``lean_rest.board_in_root``), the opposite side wall (its inner face ``width_m`` from the lean board's inner face) and the board
    above (``headroom_m`` above the support top); along x long boxes (the compartment's depth relative to the root varies by cell)."""
    from ispatialgym.grasping import contact_policy as CP

    out = []
    face = 0.0
    if lean_rest and lean_rest.get("board_in_root"):
        b = lean_rest["board_in_root"]
        out.append(CP.Support.box(b["centre"], b["half"], name="lean_board"))
        face = float(lean_rest["board_face_y_m"])
    sg = float(side_sign)
    w, t, hr, tt = (float(region[k]) for k in ("width_m", "wall_thickness_m", "headroom_m", "top_thickness_m"))
    y_in = face - sg * w  # the opposite wall's inner face
    out.append(CP.Support.box([0.0, y_in - sg * 0.5 * t, z_top + 0.5 * hr], [0.6, 0.5 * t, 0.5 * hr], name="opposite_wall"))
    yc = face - sg * 0.5 * w
    out.append(CP.Support.box([0.0, yc, z_top + hr + 0.5 * tt], [0.6, 0.5 * w + t, 0.5 * tt], name="board_above"))
    return out


def evaluate(T: np.ndarray, contact_z: np.ndarray, V_rest: np.ndarray, boxes: Sequence[Any], region: Dict[str, Any], arm: str = "right") -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """``(in region (N,), board clearance (N,) m, counts)`` of eef poses ``T`` (rest frame) whose pads touch at ``contact_z`` along the approach:
    the approach into the compartment, the contact within ``front_depth_m`` of the object's front face (min x of ``V_rest``), every hand
    sphere at ``clearance_opening_q`` (+ the sphere buffer) clear of ``boxes`` by ``margin_m``."""
    import torch as th

    from ispatialgym.grasping import hand_geometry as H

    T = np.asarray(T, dtype=float)
    a = T[:, :3, 2]
    ok_a = a[:, 0] >= float(region["approach_min_cos"]) - 1e-9
    c = T[:, :3, 3] + a * np.asarray(contact_z, dtype=float).reshape(-1, 1)
    x_front = float(np.asarray(V_rest)[:, 0].min())
    ok_d = (c[:, 0] - x_front) <= float(region["front_depth_m"]) + 1e-9
    q = float(region["clearance_opening_q"])
    Cs, Rs, _ = H.hand_spheres_eef(arm, (q, q), buffer=float(region["sphere_buffer_m"]))
    P = th.as_tensor(np.einsum("nij,kj->nki", T[:, :3, :3], Cs) + T[:, None, :3, 3], dtype=th.float32)
    r = th.as_tensor(Rs, dtype=th.float32)
    clr = th.full(P.shape[:2], float("inf"))
    for b in boxes:
        clr = th.minimum(clr, b.sdf(P) - r[None])
    clear = clr.min(dim=1).values.numpy().astype(float)
    ok_c = clear >= float(region["margin_m"]) - 1e-9
    ok = ok_a & ok_d & ok_c
    return ok, clear, {"approach": int(ok_a.sum()), "front_depth": int((ok_a & ok_d).sum()), "clear": int(ok.sum())}


def tag_of(region: Optional[Dict[str, Any]]) -> Optional[str]:
    return None if not region else str(region.get("tag") or f"{TAG_PREFIX}{str(region['digest'])[:6]}")


def pick_compatibility(goal_grasps: Sequence[Dict[str, Any]], pick_grasps: Sequence[Dict[str, Any]], pos_tol_m: float = 0.015,
                       ang_tol_deg: float = 30.0) -> List[Dict[str, Any]]:
    """Owner rule 4 (the grasp is held from pick to place): per goal-entry grasp (``T_object_eef`` in the object frame) the nearest pick-entry
    grasp or its twin - position distance, rotation angle and whether both are within tolerance (the same pinch could be held)."""
    from ispatialgym.grasping import symmetry as SY

    out = []
    pk = []
    for g in pick_grasps:
        T = S.pose_of(g["T_object_eef"])
        pk += [(g["name"], T), (g["name"] + "~r180", SY.twin_T(T))]
    for g in goal_grasps:
        T = S.pose_of(g["T_object_eef"])
        best = None
        for n, Tp in pk:
            dp = float(np.linalg.norm(T[:3, 3] - Tp[:3, 3]))
            R = T[:3, :3].T @ Tp[:3, :3]
            da = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
            if best is None or (dp, da) < (best["pos_m"], best["ang_deg"]):
                best = {"goal": g["name"], "pick": n, "pos_m": round(dp, 4), "ang_deg": round(da, 1)}
        if best is not None:
            best["compatible"] = best["pos_m"] <= pos_tol_m and best["ang_deg"] <= ang_tol_deg
            out.append(best)
    return out
