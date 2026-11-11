"""Interior PLACE maps (i7-interior 2026-11-07): the goal-side interior reach map of a STABLE rest in a fixed compartment, computed as the
place stance C will exactly check - so that the map is a superset of that check, never stricter.

The interior kind of :mod:`reach_maps` (i7-lean) judged a goal-side place like a GRASP of the object resting on the board: the eef at the
object's rest pose, the pre-grasp 0.10 m back along the gripper's approach, the hand at its pre-shape opening, the WHOLE robot (idle arm,
torso) against the compartment's boxes in ONE idle posture per trunk, the compartment at one lateral offset (the spec's ``centre_y_m``)
and the 0.08 m lattice. Stance C (``iors_large_inside``) and the exact place check after it do something else, and the map removed the
cells the legacy stance C used (inside_large_02 fridge: 29 feasible entries, all at d_e 0.55 in the board's first row; inside_large_03
sink: 1 cell per arm at d_e 0.35):

* the static body (torso, the idle arm, the head) is judged EXACTLY by stance C in the real scene, in the first free posture of
  ``inside_stance_postures`` per candidate (:func:`iors_large_inside.free_in_postures`); the map judges only the WORKING arm and its hand
  against the compartment (rule field ``body`` "arm"; "robot" = the former whole-robot check, for measurements only);
* the idle posture: a problem is feasible when it is under ANY self-collision-free posture of ``cfg.postures`` (the IK's self-collision
  with the idle arm is the posture's only effect left);
* the place is the RELEASE (``InsideMode.goal_pose_at``): the object's bottom ``drop_m`` above the board, raised until the open gripper's
  lowest point of every library grasp clears the board by ``clearance_m`` (``release_above``; ``lift_per_grasp``: the grasp's own), at the
  level orientation and at each release tilt ``tilts_deg`` (``generation.release_tilt_deg``: about the horizontal axis perpendicular to the
  grasp's radial direction, the grasp side up - :func:`iors_large_inside.tilted_quat`); an entry is feasible when any of them is;
* the pre-place (``InsideMode.pre_place_of``): ``outward`` = the release moved ``pre_m`` straight out of the opening (``generation.
  inside_pre_outward``), ``approach`` = back along the gripper's approach (``ik.pre_pose``);
* the hand carries the object CLOSED (each grasp's contact opening) along the pre-place -> release line and at the release; it opens
  there - the exact check judges the open gripper only against the surface (the release height above), and so does the map;
* the compartment at every lateral offset ``laterals_m`` of the base along the opening (stance C's laterals: ``lattice_values(-half_run,
  half_run, 0.10)`` - inside_large_02's single lateral is -0.035, not 0): its centre at ``centre_y_m - lateral`` in the base frame; a cell
  is feasible when it is at any of them (the consumer snaps the real goal cells, which lie in the real compartment, to the map's cells);
* the side walls only along the board's depth (``walls`` "board"), where ``interiors.measure`` takes its wall rays: the former map boxed
  them from the front plane, but the fridge's cabinet walls start ~0.13-0.15 m behind it (probe 2026-11-07, rays from the compartment's
  centre line: the open left door at 0.20-0.30 m and the closed RIGHT door at 0.222-0.245 m to the sides for x < 0.13, the cabinet at
  0.172-0.179 m from 0.15 m on) - the phantom 0.02 m wall boxes held the working forearm (right_arm_link4 / 5) at every lateral 0 stance;
* a finer lattice (``lattice_m`` 0.04) restricted to the compartment's span: the consumer admits goal cells within 0.51 lattice of a
  feasible cell (the 0.08 lattice's first row at d_e 0.55 sat 0.041 m in front of the fridge's first eroded goal cell - just outside).

Every exact check stays: stance C's body check, the goal cells, the camera frame, the F_goal screen and the exact place IK with the
object attached. The rule (:func:`place_rule`) is part of the map's identity (``MapConfig.interior_place`` in the config -> the key);
``ReachMap.find(..., interior_place=)`` serves a lookup only from a map whose rule covers it (:func:`covers`); maps without the field are
the former rule and keep their keys (tabletop / floor / generic / lean-interior keys unchanged).

i7-interior2 2026-11-08 (owner 5x.2): the rule field ``joint_margin_rad`` (opt-in, default 0 = absent from the rule and the key): an IK
solution of the pre-place or of the release counts only when every WORKING-arm joint is at least that far from its limits - the exact
place check's ``ik.JOINT_LIMIT_MARGIN`` (0.05 rad: ``away_from_limits`` at the pose, the tracked approach's end). Without it the fridge's
map admitted cells whose only solutions had arm joint 2 at its limit (268 of ~350 stance-C place-IK failures, median 3 mrad). A margin map
serves the same lookups (the field is not compared: it only removes entries the exact check rejects); ``meta["joint_margin"]`` counts them.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

RULE = "interior-place/1"
#: the base's lateral offsets along the opening the compartment is evaluated at (stance C's laterals; the fridge's is -0.035)
DEFAULT_LATERALS_M: Tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08)
DEFAULT_LATTICE_M = 0.04
PRE_RULES = ("approach", "outward")
BODY_RULES = ("arm", "robot")
RELEASE_RULES = ("lift", "rest")
#: the side walls' extent (``InteriorSpec.boxes(walls=)``): "board" = along the board's depth, where they were measured; "front" = from the
#: front plane (the former interior map: the fridge's walls 0.23 m in front of its board, where the doors are - probe 2026-11-07)
WALL_RULES = ("board", "front")
#: stance C's own defaults (``iors_large_inside``: DROP_HEIGHT_M, RELEASE_CLEARANCE_M; ``ik.PRE_DIST``)
DROP_M = 0.005
CLEARANCE_M = 0.02
PRE_M = 0.10
#: the exact place check's arm joint margin (``ik.IKCache.JOINT_LIMIT_MARGIN``) - the value of the opt-in rule field ``joint_margin_rad``
EXACT_JOINT_MARGIN_RAD = 0.05


def place_rule(pre: str = "approach", pre_m: float = PRE_M, drop_m: float = DROP_M, clearance_m: float = CLEARANCE_M, tilts_deg: Sequence[float] = (),
               lift_per_grasp: bool = False, laterals_m: Sequence[float] = DEFAULT_LATERALS_M, lattice_m: float = DEFAULT_LATTICE_M, body: str = "arm",
               release: str = "lift", walls: str = "board", joint_margin_rad: float = 0.0) -> Dict[str, Any]:
    """The canonical place rule (the map identity's ``interior_place``). ``body`` "robot", ``release`` "rest" and ``walls`` "front" are the
    former interior map's assumptions, kept for measurements (a lookup never asks for them). ``joint_margin_rad`` > 0 (owner 5x.2, opt-in):
    the working arm's joint margin of the pre-place and release IK, in the rule (and so the key) only when set."""
    if pre not in PRE_RULES:
        raise ValueError(f"interior place: pre {pre!r} (one of {PRE_RULES})")
    if body not in BODY_RULES:
        raise ValueError(f"interior place: body {body!r} (one of {BODY_RULES})")
    if release not in RELEASE_RULES:
        raise ValueError(f"interior place: release {release!r} (one of {RELEASE_RULES})")
    if walls not in WALL_RULES:
        raise ValueError(f"interior place: walls {walls!r} (one of {WALL_RULES})")
    tilts = sorted({round(float(t), 2) for t in tilts_deg if float(t) != 0.0})
    if any(not 0.0 < t < 90.0 for t in tilts):
        raise ValueError(f"interior place: tilts_deg {list(tilts_deg)} (0 < tilt < 90)")
    lats = sorted({round(float(v), 4) for v in laterals_m}) or [0.0]
    if not float(lattice_m) > 0:
        raise ValueError(f"interior place: lattice_m {lattice_m}")
    jm = round(float(joint_margin_rad or 0.0), 4)
    if not 0.0 <= jm < 0.5:
        raise ValueError(f"interior place: joint_margin_rad {joint_margin_rad} (0 <= margin < 0.5)")
    extra = {"joint_margin_rad": jm} if jm > 0 else {}
    return {"rule": RULE, "pre": pre, "pre_m": round(float(pre_m), 4), "drop_m": round(float(drop_m), 4), "clearance_m": round(float(clearance_m), 4),
            "tilts_deg": tilts, "lift_per_grasp": bool(lift_per_grasp), "laterals_m": lats, "lattice_m": round(float(lattice_m), 4), "body": body,
            "release": release, "walls": walls, **extra}


def parse_rule(value: Any) -> Optional[Dict[str, Any]]:
    """A place rule from the CLI / a manifest: None, "", "none" -> None (the former interior map); a JSON object (string or dict) of
    :func:`place_rule`'s fields."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    d = json.loads(value) if isinstance(value, str) else dict(value)
    d.pop("rule", None)
    return place_rule(**d)


#: the fields a map's rule must equal a lookup's (the tilts must contain the lookup's; laterals / lattice are the builder's resolution)
_EQUAL_FIELDS = ("rule", "pre", "pre_m", "drop_m", "clearance_m", "lift_per_grasp", "body", "release", "walls")


def covers(have: Optional[Dict[str, Any]], want: Optional[Dict[str, Any]]) -> Optional[str]:
    """None when a map of rule ``have`` serves a lookup of rule ``want``, else why not. ``want`` None (a lean / former interior lookup):
    only a map without a place rule; a place lookup: a map of the same rule whose tilts contain the lookup's."""
    if want is None:
        return None if have is None else "a place map (the lookup reads the former interior rule)"
    if have is None:
        return "no place rule (an interior map before i7-interior: the whole robot at the rest pose)"
    for k in _EQUAL_FIELDS:
        if have.get(k) != want.get(k):
            return f"{k} {have.get(k)!r} vs {want.get(k)!r}"
    miss = sorted(set(want.get("tilts_deg") or []) - set(have.get("tilts_deg") or []))
    if miss:
        return f"tilts {have.get('tilts_deg')} lack {miss}"
    return None


# ------------------------------------------------------------------------------------------------ the release poses (pure geometry)


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def tilted_R(R_level: np.ndarray, eef_in_object_pos: Sequence[float], deg: float) -> np.ndarray:
    """``iors_large_inside.tilted_quat`` as a matrix: the level orientation turned ``deg`` about the horizontal axis perpendicular to the
    grasp point's radial direction (the grasp side up); the level one at 0 deg or without a radial direction."""
    if not deg:
        return R_level
    p = R_level @ np.asarray(eef_in_object_pos, dtype=float)
    n = math.hypot(float(p[0]), float(p[1]))
    if n < 1e-6:
        return R_level
    return _rot(np.array([p[1] / n, -p[0] / n, 0.0]), math.radians(float(deg))) @ R_level


def open_gripper_low(R_eef: np.ndarray, p_eef: np.ndarray) -> float:
    """The lowest z of the open gripper's finger envelope at an eef pose (``iors_small.gripper_corners_world`` with its defaults)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    a, c, x = R_eef[:, 2], R_eef[:, 1], R_eef[:, 0]
    half_c = 0.5 * float(C.GRIPPER_OPEN_M) + float(IS.FINGER_THICKNESS)
    low = math.inf
    for za in (float(C.PAD_Z_RANGE[0]), float(C.FINGERTIP_LENGTH)):
        for sc in (-1.0, 1.0):
            for sx in (-1.0, 1.0):
                low = min(low, float((p_eef + za * a + sc * half_c * c + sx * float(IS.FINGER_HALF_WIDTH) * x)[2]))
    return low


def release_lift(lows: Sequence[float], surface_z: float, clearance: float) -> float:
    """``iors_large_inside.release_lift``: how far the release rises so that every lowest open-gripper point clears the surface."""
    if not lows:
        return 0.0
    return max(0.0, float(surface_z) + float(clearance) - min(float(v) for v in lows))


def arm_joint_bounds(world, entry, arm: str, margin: float) -> Optional[Tuple[th.Tensor, th.Tensor, th.Tensor]]:
    """``(active indices of the arm's joints in the IK entry, lower + margin, upper - margin)`` for :func:`margin_ok`; None without a margin."""
    if not margin > 0:
        return None
    names = list(entry.rsc.joint_names)
    sel = [i for i, n in enumerate(names) if n.startswith(f"{arm}_arm_joint")]
    if not sel:
        raise ValueError(f"interior place: no {arm}_arm_joint* among the IK entry's joints {names}")
    full = th.tensor([world.jidx[names[i]] for i in sel])
    lo = world.robot.joint_lower_limits[full].float().to(world.dev) + float(margin)
    hi = world.robot.joint_upper_limits[full].float().to(world.dev) - float(margin)
    return th.tensor(sel, device=world.dev), lo, hi


def margin_ok(q: th.Tensor, bounds: Optional[Tuple[th.Tensor, th.Tensor, th.Tensor]]) -> th.Tensor:
    """(...,) True where every arm joint of the active configurations ``q`` (..., dof) keeps the margin of ``bounds`` (all True without)."""
    if bounds is None:
        return th.ones(q.shape[:-1], dtype=th.bool, device=q.device)
    idx, lo, hi = bounds
    qa = q[..., idx]
    return ((qa >= lo) & (qa <= hi)).all(dim=-1)


def release_table(V: np.ndarray, R_rest: np.ndarray, yaws: Sequence[float], T_og: np.ndarray, base_rows: Sequence[int], height: float,
                  rule: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """The release orientations and heights per (yaw, tilt, grasp): ``(R (Y, T, G, 3, 3), z_root (Y, T, G), lift (Y, T, G), tilts)``
    with ``tilts`` = [0] + the rule's (the level release first). ``V`` the object's collision vertices (object frame), ``T_og`` (G, 4, 4)
    the map's grasps, ``base_rows`` the library rows among them (``release_above``'s "every library grasp"; a twin's open envelope is the
    row's). ``release`` "rest": the object resting on the board, level only (the former interior map)."""
    tilts = [0.0] + [float(t) for t in (rule.get("tilts_deg") or [])] if rule.get("release", "lift") == "lift" else [0.0]
    nY, nT, nG = len(yaws), len(tilts), int(T_og.shape[0])
    R = np.zeros((nY, nT, nG, 3, 3))
    z = np.zeros((nY, nT, nG))
    lift = np.zeros((nY, nT, nG))
    drop, clr = float(rule.get("drop_m", DROP_M)), float(rule.get("clearance_m", CLEARANCE_M))
    for yi, yw in enumerate(yaws):
        Rz = np.array([[math.cos(yw), -math.sin(yw), 0.0], [math.sin(yw), math.cos(yw), 0.0], [0.0, 0.0, 1.0]])
        R_lvl = Rz @ R_rest
        for ti, t in enumerate(tilts):
            for gi in range(nG):
                Rt = tilted_R(R_lvl, T_og[gi, :3, 3], t)
                R[yi, ti, gi] = Rt
                bottom = -float((V @ Rt.T)[:, 2].min())
                if rule.get("release", "lift") == "rest":
                    z[yi, ti, gi] = height + bottom
                    continue
                z0 = height + bottom + drop
                rows = [gi] if rule.get("lift_per_grasp") else list(base_rows)
                lows = [open_gripper_low(Rt @ T_og[r, :3, :3], np.array([0.0, 0.0, z0]) + Rt @ T_og[r, :3, 3]) for r in rows]
                lift[yi, ti, gi] = release_lift(lows, height, clr)
                z[yi, ti, gi] = z0 + lift[yi, ti, gi]
    return R, z, lift, tilts


def working_link_mask(links: Sequence[str], arm: str, policy) -> List[bool]:
    """The spheres a place map judges against the compartment: the working arm's links, its gripper, fingers and wrist camera (the links
    the place IK moves). Everything else is static at the stance - stance C's exact check in the real scene."""
    hand = set(policy.finger_links(arm)) | set(policy.hand_rigid_links(arm))
    pre = (f"{arm}_arm", f"{arm}_gripper", f"{arm}_realsense", f"{arm}_eef", f"{arm}_hand")
    return [bool(l) and (l in hand or l.startswith(pre)) for l in links]


def place_config_fields(spec, rule: Dict[str, Any], edge_distances: Sequence[float]) -> Dict[str, Any]:
    """The lattice of a place map: ``rule["lattice_m"]``, x over the boards of every stand-off, y over the compartment at every lateral."""
    lat = float(rule["lattice_m"])
    ymax = abs(float(spec.centre_y_m)) + 0.5 * float(spec.width_m) + max(abs(float(v)) for v in rule["laterals_m"])
    return {"lattice_m": lat, "x_min_m": round(min(edge_distances) + float(spec.board_setback_m), 4),
            "x_max_m": round(max(edge_distances) + float(spec.depth_m), 4), "y_half_m": round(math.ceil(ymax / lat) * lat, 4)}


# ------------------------------------------------------------------------------------------------ the map


def compute_place_map(world, lib, rest_id: str, height: float, anchor: Dict[str, Any], geom, cfg, policy, interior) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One interior PLACE map (module docstring): ``reach_maps.compute_map``'s ``(meta, arrays)`` with the same status arrays
    ``s{d_e}_{trunk}`` (cells x yaws x grasps; the tilts and laterals folded in: feasible when any is, else the level release's reason at
    the spec's own lateral)."""
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import reach_maps as RM
    from ispatialgym.sampling.reach_bands import trunk_grid

    rule = dict(cfg.interior_place)
    walls = rule.get("walls", "front")  # a rule without the field (built before it) had the walls from the front plane
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    rest = lib.rests[rest_id]
    if getattr(rest, "kind", "stable") != "stable":
        raise ValueError(f"an interior place map is computed at a stable rest (rest {rest_id!r} is a {getattr(rest, 'kind', '?')} rest)")
    if geom is None:
        raise ValueError("an interior place map needs the object's collision geometry")
    if abs(float(height) - float(interior.height_m)) > 1e-9:
        raise ValueError(f"an interior map is computed at its board height {interior.height_m} (got {height})")
    STATUS = RM.STATUS
    grasps = RM.grasp_set(lib, rest_id, cfg)
    nG = len(grasps)
    if nG == 0:
        raise ValueError(f"{lib.path}: no validated grasps on rest {rest_id}")
    cells = RM.lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY = len(cells), len(yaws)
    T_og = np.stack([g["T"] for g in grasps])
    base_rows = [i for i, g in enumerate(grasps) if not g.get("twin") and abs(float(g.get("param", 0.0))) < 1e-9] or list(range(nG))
    Rr = S.quat_to_mat(rest.quat_xyzw)
    V = np.asarray(geom.union.vertices)  # object frame (the rest and the yaw / tilt applied in the table)
    t0 = time.time()
    R_tab, z_tab, lift_tab, tilts = release_table(V, Rr, yaws, T_og, base_rows, float(height), rule)
    nT = len(tilts)
    N = nC * nY * nT * nG
    # object pose per (yaw, tilt, grasp) at the origin xy; the eef = object pose @ T_og; problems (cell, yaw, tilt, grasp) row-major
    To = np.zeros((nY, nT, nG, 4, 4))
    To[..., :3, :3] = R_tab
    To[..., 2, 3] = z_tab
    To[..., 3, 3] = 1.0
    TE = np.einsum("ytgij,gjk->ytgik", To, T_og)
    TE_t = th.as_tensor(TE, dtype=th.float32, device=dev).reshape(1, nY, nT, nG, 4, 4).repeat(nC, 1, 1, 1, 1, 1)
    cx = th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1, 1)
    cy = th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1, 1)
    TE_t[..., 0, 3] += cx
    TE_t[..., 1, 3] += cy
    TE_flat = TE_t.reshape(-1, 4, 4)
    del TE_t
    TP_flat = TE_flat.clone()
    pre_m = float(rule["pre_m"])
    if rule["pre"] == "outward":
        TP_flat[:, 0, 3] -= pre_m  # straight out of the opening: the robot faces +x, the opening faces -x
    else:
        TP_flat[:, :3, 3] -= pre_m * TP_flat[:, :3, 2]
    g_of = th.arange(N, device=dev) % nG
    t_of = (th.arange(N, device=dev) // nG) % nT
    y_of = (th.arange(N, device=dev) // (nG * nT)) % nY
    c_of = th.arange(N, device=dev) // (nG * nT * nY)
    cq_dev = th.as_tensor([float(g["contact_q"]) for g in grasps], dtype=th.float32, device=dev)
    tim["targets_s"] = round(time.time() - t0, 3)
    # the footprint at the level rest per yaw (the support mask: on the board, between the walls at the lateral's centre)
    Ry = RM._yaw_R(yaws)
    Vr = V @ Rr.T
    reach_back = np.array([float((Vr @ R.T)[:, 0].min()) for R in Ry])
    reach_front = np.array([float((Vr @ R.T)[:, 0].max()) for R in Ry])
    reach_ylo = np.array([float((Vr @ R.T)[:, 1].min()) for R in Ry])
    reach_yhi = np.array([float((Vr @ R.T)[:, 1].max()) for R in Ry])
    lats = [float(v) for v in rule["laterals_m"]]
    centres = [float(interior.centre_y_m) - v for v in lats]  # the compartment's centre in the base frame at each lateral of the base
    d_es = [float(d) for d in cfg.edge_distances]
    onsup: Dict[Tuple[float, int], th.Tensor] = {}
    for d_e in d_es:
        for li, yc in enumerate(centres):
            m = RM.support_mask(cells, reach_back, d_e, "interior", reach_front=reach_front, interior=interior, lateral=(reach_ylo, reach_yhi, yc))  # (C, Y)
            onsup[(d_e, li)] = th.as_tensor(m, device=dev)[c_of, y_of]
    # -- the hand (closed on the object: each grasp's contact opening) vs the compartment, along the pre-place -> release line and at the release
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)

    def hand_ok_at(T: th.Tensor, pid: th.Tensor, sb, obs, yc: float) -> th.Tensor:
        """Hand hulls of poses ``T`` (problems ``pid``) at their grasps' contact opening, in the compartment frame (centre ``yc``)."""
        out = th.ones(T.shape[0], dtype=th.bool, device=dev)
        if T.shape[0] == 0:
            return out
        T = T.clone()
        T[:, 1, 3] -= yc
        vals = th.round(cq_dev[g_of[pid]] * 1e5) / 1e5
        for v in th.unique(vals).tolist():
            m = vals == v
            out[m] = hc.check_with_obstacles(T[m], sb, obs, (float(v), float(v)))[0].to(dev)
        return out

    hand_ok: Dict[Tuple[float, int], th.Tensor] = {}
    step = policy.approach_step_m * 2
    for d_e in d_es:
        sb = interior.policed(height, d_e)
        obs = interior.obstacles(height, d_e, walls)
        for li, yc in enumerate(centres):
            ok = th.zeros(N, dtype=th.bool, device=dev)
            idx = th.nonzero(onsup[(d_e, li)]).flatten()
            for c0 in range(0, idx.numel(), 16384):
                ii = idx[c0:c0 + 16384]
                path = CP.approach_path(TP_flat[ii], TE_flat[ii], step)
                o = th.ones(ii.numel(), dtype=th.bool, device=dev)
                for k in range(path.shape[0]):
                    o &= hand_ok_at(path[k], ii, sb, obs, yc)
                ok[ii] = o
            hand_ok[(d_e, li)] = ok
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    any_sup = {d_e: th.stack([onsup[(d_e, li)] for li in range(len(centres))]).any(dim=0) for d_e in d_es}
    any_hand = {d_e: th.stack([hand_ok[(d_e, li)] & onsup[(d_e, li)] for li in range(len(centres))]).any(dim=0) for d_e in d_es}
    # -- the arm vs the object at the release (the object's pose per problem)
    pieces = [(th.as_tensor(pc.face_normals, dtype=th.float32, device=dev), th.as_tensor(np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0]),
                                                                                            dtype=th.float32, device=dev)) for pc in geom.pieces]
    hand_links = set(policy.finger_links(cfg.arm)) | set(policy.hand_rigid_links(cfg.arm))
    entry = world.entry()
    jm = float(rule.get("joint_margin_rad", 0.0) or 0.0)
    jbounds = arm_joint_bounds(world, entry, cfg.arm, jm)
    jm_stats = {"pre_seeds_dropped": 0, "release_seeds_dropped": 0, "pre_problems_lost": 0, "release_problems_lost": 0}
    links = world.link_of_sphere()
    arm_mask = th.tensor([(l not in hand_links) and bool(l) and l.startswith(f"{cfg.arm}_arm") for l in links], device=dev)
    body = rule["body"]
    work = working_link_mask(links, cfg.arm, policy) if body == "arm" else [bool(l) for l in links]
    work_t = th.tensor(work, dtype=th.bool, device=dev)
    work_links = [l for l, w in zip(links, work) if w]
    R_t = th.as_tensor(R_tab, dtype=th.float32, device=dev)  # (Y, T, G, 3, 3)
    z_t = th.as_tensor(z_tab, dtype=th.float32, device=dev)
    cells_t = th.as_tensor(cells, dtype=th.float32, device=dev)
    sphere_hits: Dict[str, int] = {}

    li_ref = int(np.argmin([abs(v) for v in lats]))  # the hits are counted at the lateral nearest 0 (the former map's one)

    def spheres_ok(sph: th.Tensor, d_e: float, li: int) -> th.Tensor:
        """The judged spheres (the working arm's; ``body`` "robot": all) clear of the board (rule 3, the fingers excepted) and of the
        obstacles, in the compartment frame of lateral ``li``; failing links counted (at ``li_ref``)."""
        s = sph[:, work_t].clone()
        s[..., 1] -= centres[li]
        ok, worst = CP.sphere_support_ok(s, work_links, interior.policed(height, d_e), cfg.arm, policy)
        ok_o, worst_o = CP.spheres_clear(s, work_links, interior.obstacles(height, d_e, walls))
        if li == li_ref:
            for k in [f"board:{w}" if w else f"obstacle:{wo}" for w, o, wo in zip(worst, ok_o.tolist(), worst_o) if w or not o]:
                sphere_hits[k] = sphere_hits.get(k, 0) + 1
        return ok.to(dev) & ok_o.to(dev)

    def object_clear(q_full: th.Tensor, pid: th.Tensor) -> th.Tensor:
        o = th.ones(pid.numel(), dtype=th.bool, device=dev)
        for c0 in range(0, pid.numel(), 2048):
            pc_ = pid[c0:c0 + 2048]
            sp = world.spheres(q_full[c0:c0 + 2048])[:, arm_mask]
            Ro = R_t[y_of[pc_], t_of[pc_], g_of[pc_]]
            to = th.cat([cells_t[c_of[pc_]], z_t[y_of[pc_], t_of[pc_], g_of[pc_]].unsqueeze(1)], 1)
            Po = th.einsum("nji,nsj->nsi", Ro, sp[..., :3] - to[:, None])
            deep = th.full(Po.shape[:2], float("inf"), device=dev)
            for nrm, dd in pieces:
                deep = th.minimum(deep, (Po @ nrm.T - dd).max(dim=-1).values)
            o[c0:c0 + 2048] = (((deep - sp[..., 3]) >= 0.0) | (sp[..., 3] <= 0)).all(dim=1)
        return o

    meta_trunks: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32),
                                     "release_lift": lift_tab.astype(np.float32), "release_z": z_tab.astype(np.float32), "tilts": np.array(tilts, dtype=np.float32)}
    t_ik = t_chk = 0.0
    n_ik = 0
    from ispatialgym.sampling.interior_shards import ENV, select
    trunks_to_compute, trunk_shard = select(trunk_grid(height), os.environ.get(ENV))
    for tr, ti in trunks_to_compute:
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        # the idle postures: every one free of self-collision (stance C judges the static body in the real scene and picks per candidate)
        postures = []
        for p in cfg.postures:
            q = world.q_config(tr, ti, p)
            if not bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                postures.append((p, q))
        rec["postures"] = [p for p, _ in postures]
        rec["posture"] = postures[0][0] if postures else None
        if not postures:
            rec["eligible"] = False
            rec["reason"] = "every idle-arm posture collides with the robot itself"
            for d_e in d_es:
                arrays[f"s{d_e:.2f}_{key}"] = np.full((nC, nY, nG), STATUS["trunk_ineligible"], dtype=np.uint8)
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        reach = L + cfg.reach_margin_m
        in_reach = (th.linalg.norm(TE_flat[:, :3, 3] - sh_t, dim=1) <= reach) & (th.linalg.norm(TP_flat[:, :3, 3] - sh_t, dim=1) <= reach)
        rec.update({"shoulder": [round(float(v), 4) for v in sh], "arm_len": round(L, 4)})
        cand = {d_e: in_reach & any_hand[d_e] for d_e in d_es}
        feas = {d_e: th.zeros(N, dtype=th.bool, device=dev) for d_e in d_es}
        why_de = {d_e: th.full((N,), STATUS["ik_pre"], dtype=th.uint8, device=dev) for d_e in d_es}
        by_posture: Dict[str, int] = {}
        n_prob = 0
        for p, q_lock in postures:
            todo = th.zeros(N, dtype=th.bool, device=dev)
            for d_e in d_es:
                todo |= cand[d_e] & ~feas[d_e]
            idx = th.nonzero(todo).flatten()
            if idx.numel() == 0:
                break
            n_prob = max(n_prob, int(idx.numel()))
            ta = time.time()
            succ_pre, q_pre = RM._solve(world, entry, TP_flat[idx], q_lock)
            n_ik += int(idx.numel())
            t_ik += time.time() - ta
            if jbounds is not None:  # owner 5x.2: the exact check's joint margin at the pre-place
                keep = succ_pre & margin_ok(q_pre, jbounds)
                jm_stats["pre_seeds_dropped"] += int((succ_pre & ~keep).sum())
                jm_stats["pre_problems_lost"] += int((succ_pre.any(dim=1) & ~keep.any(dim=1)).sum())
                succ_pre = keep
            pre_any = succ_pre.any(dim=1)
            first_seed = th.argmax(succ_pre.int(), dim=1)
            q_pre1 = q_pre[th.arange(idx.numel(), device=dev), first_seed]
            sel = th.nonzero(pre_any).flatten()
            if sel.numel() == 0:
                continue
            ta = time.time()
            succ_g, q_g = RM._solve(world, entry, TE_flat[idx[sel]], q_lock, seed_active=q_pre1[sel])
            n_ik += int(sel.numel())
            t_ik += time.time() - ta
            if jbounds is not None:  # owner 5x.2: the exact check's joint margin at the release
                keep = succ_g & margin_ok(q_g, jbounds)
                jm_stats["release_seeds_dropped"] += int((succ_g & ~keep).sum())
                jm_stats["release_problems_lost"] += int((succ_g.any(dim=1) & ~keep.any(dim=1)).sum())
                succ_g = keep
            tb = time.time()
            for d_e in d_es:
                why_de[d_e][idx[sel]] = th.where(feas[d_e][idx[sel]], why_de[d_e][idx[sel]], th.full_like(why_de[d_e][idx[sel]], STATUS["ik_grasp"]))
            qp = q_pre1[sel]
            pid_all = idx[sel]
            got = {d_e: th.zeros(sel.numel(), dtype=th.bool, device=dev) for d_e in d_es}
            for s in range(succ_g.shape[1]):
                ok_s = succ_g[:, s]
                if not bool(ok_s.any()):
                    continue
                qg = q_g[:, s]
                jump = (qg - qp).abs().max(dim=1).values > cfg.branch_max_rad
                cand_s = ok_s & ~jump
                for d_e in d_es:
                    wj = ok_s & jump & ~got[d_e]
                    why_de[d_e][pid_all[wj]] = STATUS["branch_jump"]
                need = th.zeros(sel.numel(), dtype=th.bool, device=dev)
                for d_e in d_es:
                    need |= cand_s & ~got[d_e] & cand[d_e][pid_all]
                ii = th.nonzero(need).flatten()
                if ii.numel() == 0:
                    continue
                pid = pid_all[ii]
                # the judged spheres along the pre-place -> release joint path, per (stand-off, lateral)
                ok_sph = {(d_e, li): th.ones(ii.numel(), dtype=th.bool, device=dev) for d_e in d_es for li in range(len(centres))}
                for t in [0.0] + [k / (cfg.approach_configs + 1) for k in range(1, cfg.approach_configs + 1)] + [1.0]:
                    for c0 in range(0, ii.numel(), 8192):
                        iic = ii[c0:c0 + 8192]
                        qq = qp[iic] * (1 - t) + qg[iic] * t
                        sp = world.spheres(world.full_q(entry, qq, q_lock))
                        for d_e in d_es:
                            for li in range(len(centres)):
                                ok_sph[(d_e, li)][c0:c0 + 8192] &= spheres_ok(sp, d_e, li)
                # the hand hulls at the FK eef of the release and the pre-place (IK tolerance), the object in the hand (contact opening)
                E_g = world.eef(world.full_q(entry, qg[ii], q_lock))
                E_p = world.eef(world.full_q(entry, qp[ii], q_lock))
                obj_ok = object_clear(world.full_q(entry, qg[ii], q_lock), pid)
                for d_e in d_es:
                    sb, obs = interior.policed(height, d_e), interior.obstacles(height, d_e, walls)
                    hit = th.zeros(ii.numel(), dtype=th.bool, device=dev)
                    w_sph = th.ones(ii.numel(), dtype=th.bool, device=dev)
                    w_hand = th.ones(ii.numel(), dtype=th.bool, device=dev)
                    for li, yc in enumerate(centres):
                        base_ok = onsup[(d_e, li)][pid] & hand_ok[(d_e, li)][pid] & cand_s[ii]
                        if not bool(base_ok.any()):
                            continue
                        s_ok = ok_sph[(d_e, li)]
                        h_ok = hand_ok_at(E_g, pid, sb, obs, yc) & hand_ok_at(E_p, pid, sb, obs, yc)
                        good = base_ok & s_ok & h_ok & obj_ok
                        hit |= good
                        w_sph &= ~(base_ok & s_ok)
                        w_hand &= ~(base_ok & s_ok & h_ok)
                    newly = hit & ~got[d_e][ii]
                    got[d_e][ii] |= hit
                    w = why_de[d_e][pid]
                    w = th.where(w_sph, th.full_like(w, STATUS["slab_spheres"]), w)
                    w = th.where(~w_sph & w_hand, th.full_like(w, STATUS["hand_policy"]), w)
                    w = th.where(~w_sph & ~w_hand & ~obj_ok, th.full_like(w, STATUS["object_arm"]), w)
                    why_de[d_e][pid] = th.where(got[d_e][ii], why_de[d_e][pid], w)
                    by_posture[p] = by_posture.get(p, 0) + int(newly.sum())
            for d_e in d_es:
                feas[d_e][pid_all] |= got[d_e]
            t_chk += time.time() - tb
        rec["n_problems"] = n_prob
        rec["feasible_by_posture"] = by_posture
        for d_e in d_es:
            st = th.full((N,), STATUS["unevaluated"], dtype=th.uint8, device=dev)
            st[~in_reach] = STATUS["reach_bound"]
            st[in_reach & ~any_hand[d_e]] = STATUS["hand_policy"]
            st[~any_sup[d_e]] = STATUS["off_support"]
            ev = cand[d_e]
            st[ev] = why_de[d_e][ev]
            st[feas[d_e]] = STATUS["feasible"]
            st5 = st.view(nC, nY, nT, nG)
            f_any = (st5 == STATUS["feasible"]).any(dim=2)
            out = th.where(f_any, th.full_like(st5[:, :, 0], STATUS["feasible"]), st5[:, :, 0])
            arrays[f"s{d_e:.2f}_{key}"] = out.cpu().numpy()
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int(f_any.sum())
            rec.setdefault("n_feasible_by_tilt", {})[f"{d_e:.2f}"] = {f"{tilts[k]:g}": int((st5[:, :, k] == STATUS["feasible"]).sum()) for k in range(nT)}
        meta_trunks.append(rec)
        print("INTERIOR_TRUNK_DONE " + json.dumps({"key": key, "feasible": rec.get("n_feasible"),
              "elapsed_s": round(time.time() - t_all, 2), "shard": trunk_shard}), flush=True)
    tim.update({"ik_s": round(t_ik, 2), "checks_s": round(t_chk, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)})
    meta = {"schema": RM.SCHEMA, "kind": "interior", "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": lib.identity(), "object_key": lib.identity_key(), "rest_id": rest_id,
            "height": {"value": height, "anchor_key": anchor.get("key"), "anchor_surfaces": anchor.get("surfaces", [])[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "policy": policy.meta(cfg.arm),
            "slab": None, "plane": RM.PLANE_OF_KIND["interior"],
            "ik": {"seeds": cfg.seeds, "position_tolerance": world.backend.position_tolerance, "orientation_tolerance": world.backend.orientation_tolerance,
                   "collision_world": "floor cuboid + self-collision (the compartment and the object checked explicitly)", "pre_dist": pre_m},
            "approach_check": (f"hand hulls at each grasp's contact opening at the {rule['pre']} pre-place, every {step:.3f} m of the straight line to the "
                               f"release and at the release; the {'working arm' if body == 'arm' else 'whole robot'}'s spheres at "
                               f"{cfg.approach_configs} joint-interpolated configurations"),
            "screening": cfg.mode, "exhaustive": True,
            "grasps": [{k: v for k, v in g.items() if k != "T"} | {"T_object_eef": S.mat_pose(g["T"])} for g in grasps],
            "brace": None, "trunks": meta_trunks, "status_codes": STATUS, "robot": RM.robot_digest(), "timing": tim, "created": S.now(),
            "interior": interior.to_json(), "interior_placement": "fixed_place", "interior_place": rule,
            **({"joint_margin": {"rad": jm, **jm_stats}} if jbounds is not None else {}),
            "interior_sphere_hits": dict(sorted(sphere_hits.items(), key=lambda kv: -kv[1])[:12]),
            "release": {"tilts_deg": tilts, "lift_m": {f"{t:g}": [round(float(lift_tab[:, k].min()), 4), round(float(lift_tab[:, k].max()), 4)] for k, t in enumerate(tilts)},
                        "base_rows": [grasps[i]["name"] for i in base_rows], "centres_y_m": [round(v, 4) for v in centres]},
            "interior_note": ("an interior PLACE map (i7-interior, interior_place.py): the stable rest's release in a compartment fixed in the map, at "
                              "every lateral of laterals_m (its centre at centre_y_m - lateral); the release lifted / tilted as the exact place; the "
                              "working arm and its hand (closed on the object) against the board (policed) and the other boxes; the static body is "
                              "stance C's exact check; feasible under any self-collision-free idle posture, at any tilt / lateral")}
    if trunk_shard is not None:
        meta["interior_trunk_shard"] = trunk_shard
        meta["exhaustive"] = False
    meta["key"] = RM.map_key(meta)
    return meta, arrays
