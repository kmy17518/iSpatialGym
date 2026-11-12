"""ATTACH reach maps (i7-attachmap 2026-11-07, owner: "no v2 attach map" for configuration_matching-attachment_01 / 04).

The legacy attach-reach band (``tasks/attachment.py``: ``ReachBands.load(scene, support, <partner>, "attach:<lid key>",
placed=["attach:<lid>"])`` or ``tabletop.compute_bands`` of the pseudo-grasps ``T_partner_male ∘ T_object_eef``) is a per-scene band of the
PARTNER: a cell (the partner root on the support top) is in it when one (partner yaw, male grasp) pair has its seat eef pose and the
10 cm pre-grasp back along the gripper's approach IK-reachable, the support ignored at the seat, the partner not in the scene. The task
reads it at a stance only as cells - ``band_cells_world`` (the partner cells ``_partner_room`` / ``_attach_and_brace`` try) and
``joint_placement`` (the work cell; ``classes`` None for library grasps) - and decides everything else exactly.

An attach map is its scene-independent replacement, the reach map of the held male at its SEAT on the partner, computed in the synthetic
world of :mod:`reach_maps` (the R1Pro alone; the support a synthetic slab at the anchor height, its near edge at ``d_e``):

* **cell** = the partner's ROOT xy in the map frame (the base on the floor, +x ahead), at its explicit rest (identity by default)
  at ``n_yaws`` yaw slots, its root at ``h + bottom offset + 3 mm`` (``tabletop.object_pose_on``); **grasps** = the male's v2 grasps of
  its stable rest with their twins / family members (``reach_maps.grasp_set``: the pick map's grasp axis);
* the pair (:class:`AttachSpec`, measured in Isaac Sim by :func:`measure_pair` with the task's own code): the male root in the partner
  root frame when attached (``T_partner_male = parent_link_in_female ∘ root_in_child_link``), the sampler's insertion axis
  (``attachment.insertion_axis_rule``), the oracle's seat axis (``predicates.seat_axis``), the partner's collision pieces and bottom offset;
* per (d_e, trunk, cell, yaw, grasp) the CHAIN the oracle executes after the carry (``execute.execute_place`` for ``kind="attach"``), the
  seat eef ``T_e = T_partner ∘ T_partner_male ∘ T_object_eef`` translated along the axes, one orientation:
  ``pre`` (the recorded pre-attach: ``pre_attach_back_m`` back along the insertion axis) -> ``hover`` (``attach_hover_targets``: the seat
  raised along the seat axis by the HOVER RULE's height) -> ``touchdown`` (``_touch_down``'s leg end: the hover lowered by the part's
  clearance + ``attach_touchdown_extra_m``) -> ``seat`` (the sampler's exact check reaches the seat pose itself); batched IK of the pre
  (``seeds`` seeds), each next pose seeded from the previous one (same branch: the smallest joint step <= ``branch_max_rad``); the hand's
  hulls (the fingers closed at the grasp's measured opening) against the slab at every chain pose; per d_e every non-finger sphere clear
  of the slab at the chain's configurations and ``approach_configs`` interpolated pre -> hover ones (the descent); the arm (not the hand)
  clear of the partner's pieces at pre / descent / hover / touchdown (the exact check ignores the partner at the seat, as here).

**Hover rules** (:data:`HOVER_RULES`, part of the identity; the oracle's ``ExecuteConfig.attach_hover_rule`` where the oracle has it,
i7-attachhover a982690a ``ISPATIALGYM_ATTACH_HOVER``, else ``box``): ``box`` (today) - the lowest point of the part and the fingertips
``attach_hover_clear_m`` above the highest partner vertex of every partner piece whose xy box meets theirs (widened by
``attach_hover_footprint_margin_m``), divided by ``max(axis_z, 0.5)``; ``swept`` (option B) - ``attach_hover_clear_m`` above the FIRST TOUCH
of the part / fingertips swept down the seat axis onto the partner's pieces (:func:`first_touch`, an exact Minkowski-difference ray cast
of convex pieces; no touch -> the box rule). The touchdown depth is the oracle's: the part's clearance (box: vertical, ``low_part + dz -
top``; swept: ``h - t_part``; 0 -> ``attach_hover_clear_m``) + ``attach_touchdown_extra_m``.

Identity (:func:`map_key`): kind, arm, the male (v2 entry, its identity key, rest, the grasp set), the partner (category / model / scale /
geometry digest), the pair spec digest (T_partner_male, insertion + seat axes, bottom offset, drop, both geometry digests), the hover rule
and its parameters, ``pre_attach_back_m``, the height, the edge distances, the config, the contact policy key and the robot digest.
Files ``<maps root>/<male entry>/amap__attach__<partner category>__<partner model>__h<h>__<key>.{json,npz}`` (``ReachMap.find`` reads
``map__*`` only: an attach map is never served for a tabletop lookup). A map is a proposal filter: the sampler's checks stay exact.

    python -m ispatialgym.sampling.attach_maps build --pairs PAIRS.json --glib GLIB --anchors anchors.json --out OUT [--rules box swept] [--arm right]
    python -m ispatialgym.sampling.attach_maps info MAP.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import reach_maps as RM

SCHEMA = "isg-attach-map/1"
SPEC_SCHEMA = "isg-attach-spec/1"
KIND = "attach"
FILE_PREFIX = f"amap__{KIND}__"
#: the oracle's hover rules (``execute.ATTACH_HOVER_RULES`` on i7-attachhover); a map is computed for one
HOVER_RULES = ("box", "swept")
#: the chain the oracle runs after the carry, in order (the eef at one orientation, translated along the pair's axes)
CHAIN = ("pre", "hover", "touchdown", "seat")
#: the reach-map codes plus the chain's later IK steps and the arm against the partner
STATUS = dict(RM.STATUS, ik_hover=12, ik_touchdown=13, ik_seat=14, partner_arm=15)
STATUS_NAME = {v: k for k, v in STATUS.items()}
#: the oracle's hover divides the vertical clearance by ``max(axis_z, AXIS_Z_MIN)`` (``attach_hover_targets``)
AXIS_Z_MIN = 0.5
#: the pair drop of a partner placed on a support (``tabletop.Z_OFFSET``: z = top + bottom offset + 3 mm)
PARTNER_DROP_M = 0.003
#: the live pair (sampler template) vs the map's measured spec: the male's seat position / rotation, the insertion axis, the bottom offset
SPEC_TOL = {"pos_m": 0.001, "rot_rad": 0.01, "axis_rad": 0.01, "bottom_m": 0.002}


def canonical_partner_rest(value=None) -> List[float]:
    """Explicit measured partner rest; omitted values preserve old identity maps."""
    q = np.asarray([0., 0., 0., 1.] if value is None else value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(float(np.linalg.norm(q))-1.) > 1e-5:
        raise ValueError('partner rest must be a finite unit xyzw quaternion')
    q = q / np.linalg.norm(q)
    # q and -q name one physical rest and must share a cache identity.
    for index in (3, 0, 1, 2):
        if abs(q[index]) > 1e-12:
            if q[index] < 0: q = -q
            break
    return [0. if abs(x) < 1e-12 else float(x) for x in q]


# ------------------------------------------------------------------------------------------------ small math (numpy, 4x4)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def _tf(T: np.ndarray, P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def _angle(R: np.ndarray) -> float:
    return math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def digest(obj: Any, n: int = 12) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def pieces_digest(pieces: Sequence[np.ndarray]) -> str:
    """Content digest of collision pieces (vertices rounded to 1e-6 m)."""
    h = hashlib.sha1()
    for p in pieces:
        h.update(np.round(np.asarray(p, dtype=np.float64), 6).tobytes())
        h.update(b"|")
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ convex pieces, first touch


@dataclass
class Hull:
    """A convex piece: its hull vertices and its faces as halfspaces ``normals @ x <= offsets``."""

    verts: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray


def hull_of(points) -> Optional[Hull]:
    """The convex hull of ``points`` (N, 3); a flat piece gets a joggled (paper-thin) hull; None with fewer than 4 points."""
    from scipy.spatial import ConvexHull

    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 4:
        return None
    for opts in (None, "QJ"):
        try:
            h = ConvexHull(P, qhull_options=opts) if opts else ConvexHull(P)
            return Hull(verts=P[h.vertices], normals=h.equations[:, :3].copy(), offsets=-h.equations[:, 3].copy())
        except Exception:  # noqa: BLE001 - QhullError (flat input): joggled next
            continue
    return None


def _perp(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def pair_first_touch(A: np.ndarray, B: np.ndarray, u: np.ndarray, tol: float = 1e-9) -> Optional[float]:
    """The largest ``t`` at which the convex hull of ``A`` (moving) translated by ``t u`` meets the hull of ``B`` (fixed), None when never:
    ``t u`` must lie in the Minkowski difference ``B ⊖ A = hull{b - a}`` - a ray cast along ``u`` through its halfspaces."""
    D = (np.asarray(B, dtype=np.float64)[:, None, :] - np.asarray(A, dtype=np.float64)[None, :, :]).reshape(-1, 3)
    h = hull_of(D)
    if h is None:
        return None
    k = h.normals @ u
    pos, neg = k > 1e-12, k < -1e-12
    par = ~(pos | neg)
    if par.any() and (h.offsets[par] < -tol).any():
        return None
    hi = float(np.min(h.offsets[pos] / k[pos])) if pos.any() else math.inf
    lo = float(np.max(h.offsets[neg] / k[neg])) if neg.any() else -math.inf
    if not math.isfinite(hi) or lo > hi + tol:
        return None
    return hi


def first_touch(moving: Sequence[np.ndarray], fixed: Sequence[np.ndarray], axis, tol: float = 1e-9) -> Dict[str, Any]:
    """The largest ``t`` at which some piece of ``moving`` (vertex arrays, world) translated by ``t * axis`` meets some piece of ``fixed``
    (``t`` None when no pair meets: their footprints across the axis miss), the pair that sets it and the pair counts. Pairs whose boxes in
    the plane normal to the axis miss are skipped; the rest are visited by their upper bound (the fixed piece's top minus the moving
    piece's bottom along the axis) and the search stops once no pair left can beat the best touch."""
    u = _unit(axis)
    e1, e2 = _perp(u)

    def box(P):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        x, y, z = P @ e1, P @ e2, P @ u
        return x.min(), x.max(), y.min(), y.max(), z.min(), z.max()

    bm = [(i, box(p)) for i, p in enumerate(moving) if p is not None and len(p) >= 4]
    bf = [(j, box(p)) for j, p in enumerate(fixed) if p is not None and len(p) >= 4]
    pairs = []
    for i, a in bm:
        for j, b in bf:
            if a[0] > b[1] + tol or b[0] > a[1] + tol or a[2] > b[3] + tol or b[2] > a[3] + tol:
                continue
            pairs.append((b[5] - a[4], i, j))
    pairs.sort(key=lambda x: -x[0])
    best, arg, n = None, None, 0
    for ub, i, j in pairs:
        if best is not None and ub <= best + tol:
            break
        n += 1
        t = pair_first_touch(moving[i], fixed[j], u, tol)
        if t is not None and (best is None or t > best):
            best, arg = t, (i, j)
    return {"t": best, "pair": arg, "pairs_total": len(bm) * len(bf), "pairs_overlapping": len(pairs), "pairs_tested": n}


# ------------------------------------------------------------------------------------------------ the oracle's hover rules (pure)


def hover_box(part: Sequence[np.ndarray], fingers: Sequence[np.ndarray], partner: Sequence[np.ndarray], axis, clear_m: float,
              footprint_margin_m: float) -> Dict[str, Any]:
    """``execute.attach_hover_targets``' box rule at the seated hand pose (every input in one world frame, the part and the fingertips AT
    THE SEAT): the partner's top = the highest vertex of the partner pieces whose xy box meets the part + fingertips' box widened by
    ``footprint_margin_m`` (all pieces when none does); the height above the seat along the axis = ``max(0, top + clear - lowest) /
    max(axis_z, 0.5)``; the part's clearance above the top at the hover (vertical) and which is lowest."""
    a = _unit(axis)
    P = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in part])
    F = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in fingers]) if fingers else None
    moving = P if F is None else np.concatenate([P, F])
    lo = moving[:, :2].min(axis=0) - footprint_margin_m
    hi = moving[:, :2].max(axis=0) + footprint_margin_m
    meshes = [np.asarray(m, dtype=np.float64).reshape(-1, 3) for m in partner]
    under = [m for m in meshes if bool((m[:, :2].min(axis=0) <= hi).all() and (m[:, :2].max(axis=0) >= lo).all())]
    top = max(float(m[:, 2].max()) for m in (under or meshes))
    low_part = float(P[:, 2].min())
    low_fingers = float(F[:, 2].min()) if F is not None else math.inf
    h = max(0.0, top + clear_m - min(low_part, low_fingers)) / max(float(a[2]), AXIS_Z_MIN)
    dz = h * float(a[2])
    return {"h_m": h, "top_z": top, "part_clearance_m": low_part + dz - top, "finger_clearance_m": None if F is None else low_fingers + dz - top,
            "lowest": "fingertips" if low_fingers < low_part else "part", "partner_under": len(under), "partner_pieces": len(meshes)}


def hover_of_rule(rule: str, box: Dict[str, Any], t_part: Optional[float], t_fingers: Optional[float], clear_m: float,
                  touchdown_extra_m: float) -> Dict[str, Any]:
    """The hover the oracle uses under ``rule`` (``box`` / ``swept``) and its touchdown: ``h_m`` above the seat along the seat axis, the
    part's clearance, the touchdown depth (``(part clearance or clear) + extra``: ``_touch_down``) and its end ``touchdown_m = h - depth``
    above the seat. Swept: ``clear`` above the first touch of the part or the fingertips, whichever is higher; no touch -> the box rule."""
    if rule not in HOVER_RULES:
        raise ValueError(f"hover rule {rule!r} is not one of {HOVER_RULES}")
    out: Dict[str, Any] = {"rule": rule, "box_h_m": box["h_m"], "first_touch_m": t_part, "first_touch_fingers_m": t_fingers}
    if rule == "swept" and t_part is not None:
        first = t_part if t_fingers is None else max(t_part, t_fingers)
        h = max(0.0, first + clear_m)
        part_clearance = h - t_part
        out.update(source="swept", lowest="fingertips" if t_fingers is not None and t_fingers > t_part else "part")
    else:
        h, part_clearance = box["h_m"], box["part_clearance_m"]
        out.update(source="box", lowest=box["lowest"])
        if rule == "swept":
            out["swept_fallback"] = "no piece of the part meets a piece of the partner along the seat axis"
    depth = float(part_clearance or clear_m) + float(touchdown_extra_m)
    out.update(h_m=h, part_clearance_m=part_clearance, depth_m=depth, touchdown_m=h - depth,
               gap_to_first_touch_m=None if t_part is None else h - t_part)
    return out


# ------------------------------------------------------------------------------------------------ the pair spec


@dataclass
class AttachSpec:
    """One (male entry, partner model / scale) pair, measured by :func:`measure_pair` (Isaac Sim, the task's code) - the record a map is
    computed from and checked against (its :meth:`digest` is part of the map identity)."""

    data: Dict[str, Any]
    partner_pieces: List[np.ndarray]

    @property
    def partner_rest(self) -> List[float]:
        return canonical_partner_rest(self.data['partner'].get('rest_quat'))

    def partner_transform(self, yaw: float) -> np.ndarray:
        rotation = _rz(float(yaw))
        rest = self.partner_rest
        return rotation if rest == [0., 0., 0., 1.] else rotation @ S.pose_mat([0., 0., 0.], rest)

    @property
    def T_partner_male(self) -> np.ndarray:
        return S.pose_of(self.data["attachment"]["T_partner_male"])

    @property
    def insertion_axis(self) -> np.ndarray:
        return _unit(self.data["insertion_axis"]["local"])

    @property
    def seat_axis(self) -> np.ndarray:
        return _unit(self.data["seat_axis"]["local"])

    @property
    def bottom_offset_m(self) -> float:
        return float(self.data["partner"]["bottom_offset_m"])

    @property
    def drop_m(self) -> float:
        return float(self.data["partner"].get("drop_m", PARTNER_DROP_M))

    def identity(self) -> Dict[str, Any]:
        d = self.data
        identity = {"schema": d.get("schema"), "male": {k: d["male"].get(k) for k in ("entry", "category", "model", "scale", "geometry_digest")},
                "partner": {k: d["partner"].get(k) for k in ("category", "model", "scale", "geometry_digest", "bottom_offset_m", "drop_m")},
                "attachment": {k: d["attachment"].get(k) for k in ("T_partner_male", "child_link", "parent_link")},
                "insertion_axis": [round(float(v), 6) for v in d["insertion_axis"]["local"]], "seat_axis": [round(float(v), 6) for v in d["seat_axis"]["local"]]}
        if self.partner_rest != [0., 0., 0., 1.]:
            identity['partner']['rest_quat'] = self.partner_rest
        return identity

    def digest(self) -> str:
        return digest(self.identity(), 16)


def spec_path(spec_root: str, pair_id: str) -> str:
    return os.path.join(spec_root, pair_id, "attach_spec.json")


def save_spec(spec_root: str, pair_id: str, data: Dict[str, Any], partner_pieces: Sequence[np.ndarray]) -> str:
    d = os.path.join(spec_root, pair_id)
    os.makedirs(d, exist_ok=True)
    S.save_json(os.path.join(d, "partner_geometry.json"), {"schema": SPEC_SCHEMA, "frame": "partner root", "pieces": [np.round(np.asarray(p), 6).tolist() for p in partner_pieces]})
    data = dict(data, partner_geometry="partner_geometry.json")
    p = os.path.join(d, "attach_spec.json")
    S.save_json(p, data)
    return p


def load_spec(path: str) -> AttachSpec:
    with open(path) as f:
        data = json.load(f)
    if data.get("schema") != SPEC_SCHEMA:
        raise S.IncompatibleRecord(f"{path}: schema {data.get('schema')!r} is not {SPEC_SCHEMA}")
    with open(os.path.join(os.path.dirname(path), data.get("partner_geometry", "partner_geometry.json"))) as f:
        g = json.load(f)
    pieces = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in g["pieces"]]
    if pieces_digest(pieces) != data["partner"]["geometry_digest"]:
        raise S.IncompatibleRecord(f"{path}: the partner geometry does not match its digest")
    return AttachSpec(data, pieces)


def male_pieces(model_dir: str) -> List[np.ndarray]:
    """The male's collision pieces (root frame, the entry's scale) from its v2 entry's ``geometry.json`` (``object_geometry.export_loaded``)."""
    d = S.load_json(os.path.join(model_dir, "geometry.json"))
    if d is None:
        raise FileNotFoundError(f"{model_dir}: no geometry.json")
    return [np.asarray(p["vertices"], dtype=np.float64).reshape(-1, 3) for p in d["pieces"]]


def finger_pieces(hand, finger_q: Sequence[float]) -> List[np.ndarray]:
    """The fingers' collision pieces (``hand_geometry``: eef frame) at the joint values ``finger_q`` (link1, link2)."""
    out = []
    for lk in hand.by_role("finger"):
        q = float(finger_q[0] if lk.link.endswith("1") else finger_q[1])
        for pc in (lk.pieces or [lk.points]):
            out.append(np.asarray(pc, dtype=np.float64) + (q * np.asarray(lk.axis) if lk.axis is not None else 0.0))
    return out


# ------------------------------------------------------------------------------------------------ the grasp axis and the hover table


def attach_grasps(lib: S.ModelLibrary, rest_id: str, cfg: RM.MapConfig) -> List[Dict[str, Any]]:
    """The map's grasp axis = the pick map's (``reach_maps.grasp_set``: every validated grasp of the rest, its twin, family members), each
    with the fingers' joint values while it holds the male (``finger_q``: the row's measured values, reversed for a twin; the contact
    opening on both fingers without them)."""
    rows = {g["name"]: g for g in lib.grasps(rest_id)}
    out = []
    for g in RM.grasp_set(lib, rest_id, cfg):
        fq = rows.get(g["base"], {}).get("finger_q")
        fq = [float(v) for v in fq] if fq and len(fq) == 2 else [float(g["contact_q"])] * 2
        out.append(dict(g, finger_q=list(reversed(fq)) if g["twin"] else fq))
    return out


def grasp_set_digest(grasps: Sequence[Dict[str, Any]]) -> str:
    return digest([(g["name"], np.round(np.asarray(g["T"], dtype=np.float64), 6).tolist(), [round(float(v), 6) for v in g["finger_q"]]) for g in grasps])


def hover_table(spec: AttachSpec, male: Sequence[np.ndarray], hand, grasps: Sequence[Dict[str, Any]], yaws: Sequence[float],
                params: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """``table[y][g]``: the hover / touchdown of grasp ``g`` with the partner at yaw slot ``y`` (:func:`hover_box`, :func:`first_touch`,
    :func:`hover_of_rule`), in the partner's frame turned by the yaw. The box rule reads world xy boxes (it depends on the yaw); the
    swept first touch is invariant to a common turn about z (computed at the first yaw, reused)."""
    T_pm = spec.T_partner_male
    out: List[List[Dict[str, Any]]] = []
    swept: Dict[int, Tuple[Optional[float], Optional[float], Any]] = {}
    for yi, yaw in enumerate(yaws):
        R = spec.partner_transform(yaw)
        axis = R[:3, :3] @ spec.seat_axis
        partner = [_tf(R, p) for p in spec.partner_pieces]
        row = []
        for gi, g in enumerate(grasps):
            T_m = R @ T_pm
            part = [_tf(T_m, p) for p in male]
            fing = [_tf(T_m @ np.asarray(g["T"], dtype=np.float64), p) for p in finger_pieces(hand, g["finger_q"])]
            bx = hover_box(part, fing, partner, axis, params["clear_m"], params["footprint_margin_m"])
            if gi not in swept:
                if params["rule"] == "swept" or params.get("sweep_always", True):
                    rp, rf = first_touch(part, partner, axis), first_touch(fing, partner, axis)
                    swept[gi] = (rp["t"], rf["t"], {"part_pair": rp["pair"], "fingers_pair": rf["pair"], "pairs_tested": rp["pairs_tested"] + rf["pairs_tested"]})
                else:
                    swept[gi] = (None, None, None)
            t_p, t_f, info = swept[gi]
            hv = hover_of_rule(params["rule"], bx, t_p, t_f, params["clear_m"], params["touchdown_extra_m"])
            hv.update(yaw_deg=round(math.degrees(float(yaw)), 3), grasp=g["name"], box_top_z=round(bx["top_z"], 5), box_partner_under=bx["partner_under"],
                      sweep=info)
            row.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in hv.items()})
        out.append(row)
    return out


def chain_offsets(spec: AttachSpec, hover: List[List[Dict[str, Any]]], yaws: Sequence[float], pre_back_m: float) -> np.ndarray:
    """(Y, G, K, 3): each chain pose's translation from the seat eef (world-aligned, the partner at yaw slot ``y``): ``pre`` = ``pre_back_m``
    along the insertion axis, ``hover`` / ``touchdown`` = their heights along the seat axis, ``seat`` = 0."""
    nY, nG = len(yaws), len(hover[0]) if hover else 0
    out = np.zeros((nY, nG, len(CHAIN), 3))
    for yi, yaw in enumerate(yaws):
        R = spec.partner_transform(yaw)[:3, :3]
        a_ins, a_seat = R @ spec.insertion_axis, R @ spec.seat_axis
        for gi in range(nG):
            hv = hover[yi][gi]
            out[yi, gi, 0] = pre_back_m * a_ins
            out[yi, gi, 1] = float(hv["h_m"]) * a_seat
            out[yi, gi, 2] = float(hv["touchdown_m"]) * a_seat
    return out


def seat_eef(spec: AttachSpec, grasps: Sequence[Dict[str, Any]], yaws: Sequence[float]) -> np.ndarray:
    """(Y, G, 4, 4): the seat eef with the partner ROOT at the origin xy, z 0 (the support at 0), at yaw slot ``y``."""
    T_pm = spec.T_partner_male
    out = np.zeros((len(yaws), len(grasps), 4, 4))
    for yi, yaw in enumerate(yaws):
        Tp = spec.partner_transform(yaw)
        Tp[2, 3] = spec.bottom_offset_m + spec.drop_m
        for gi, g in enumerate(grasps):
            out[yi, gi] = Tp @ T_pm @ np.asarray(g["T"], dtype=np.float64)
    return out


# ------------------------------------------------------------------------------------------------ the oracle's parameters


def oracle_hover_rule(cfg=None) -> str:
    """The hover rule the oracle executes: ``ExecuteConfig.attach_hover_rule`` through the oracle's own resolver where it has one
    (i7-attachhover a982690a: ``ISPATIALGYM_ATTACH_HOVER`` box | swept, anything else box), else ``box`` (an oracle without the switch)."""
    from ispatialgym.oracle import execute as E

    cfg = cfg if cfg is not None else E.ExecuteConfig()
    fn = getattr(E, "attach_hover_rule", None)
    if fn is None or not hasattr(cfg, "attach_hover_rule"):
        return "box"
    rule = str(fn(cfg))
    return rule if rule in HOVER_RULES else "box"


def oracle_hover_params(rule: Optional[str] = None, cfg=None) -> Dict[str, Any]:
    """The hover / touchdown parameters of the oracle's ``ExecuteConfig`` (part of the map identity) for ``rule`` (default: the rule the
    oracle executes, :func:`oracle_hover_rule`)."""
    from ispatialgym.oracle import execute as E

    cfg = cfg if cfg is not None else E.ExecuteConfig()
    r = rule or oracle_hover_rule(cfg)
    if r not in HOVER_RULES:
        raise ValueError(f"hover rule {r!r} is not one of {HOVER_RULES}")
    return {"rule": r, "clear_m": float(cfg.attach_hover_clear_m), "footprint_margin_m": float(cfg.attach_hover_footprint_margin_m),
            "touchdown_extra_m": float(cfg.attach_touchdown_extra_m), "axis_z_min": AXIS_Z_MIN,
            "method": "box: execute.attach_hover_targets; swept: exact Minkowski ray cast of the convex pieces (convex_sweep.first_touch's result)"}


def hover_identity(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: params[k] for k in ("rule", "clear_m", "footprint_margin_m", "touchdown_extra_m", "axis_z_min")}


# ------------------------------------------------------------------------------------------------ the computation (cuRobo, the synthetic world)


def compute_attach_map(world: RM.SynthWorld, spec: AttachSpec, lib: S.ModelLibrary, rest_id: str, height: float, anchor: Dict[str, Any], *,
                       male: Sequence[np.ndarray], hand, params: Dict[str, Any], pre_back_m: float, cfg: Optional[RM.MapConfig] = None,
                       policy: CP.ContactPolicy = CP.DEFAULT) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One attach map: ``(meta, arrays)`` (module docstring). ``male``: the male's collision pieces (root frame); ``hand``: the hand's
    collision geometry (``hand_geometry.load(arm)``); ``params``: :func:`oracle_hover_params`."""
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    grasps = attach_grasps(lib, rest_id, cfg)
    if not grasps:
        raise ValueError(f"{lib.path}: no validated grasps on rest {rest_id}")
    cells = RM.lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY, nG, K = len(cells), len(yaws), len(grasps), len(CHAIN)
    t0 = time.time()
    hover = hover_table(spec, male, hand, grasps, yaws, params)
    tim["hover_s"] = round(time.time() - t0, 2)
    # -- the chain's eef targets (cells x yaws x grasps, row-major), the map frame
    TS = seat_eef(spec, grasps, yaws)  # (Y, G, 4, 4), the partner root at the origin
    off = chain_offsets(spec, hover, yaws, pre_back_m)  # (Y, G, K, 3)
    TE = np.repeat(TS[:, :, None], K, axis=2).copy()  # (Y, G, K, 4, 4)
    TE[..., :3, 3] += off
    TE_t = th.as_tensor(TE, dtype=th.float32, device=dev).reshape(1, nY, nG, K, 4, 4).repeat(nC, 1, 1, 1, 1, 1)
    TE_t[..., 0, 3] += th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1, 1)
    TE_t[..., 1, 3] += th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1, 1)
    TE_t[..., 2, 3] += float(height)
    chain = TE_t.reshape(-1, K, 4, 4)  # (N, K, 4, 4)
    N = chain.shape[0]
    g_of = th.arange(N, device=dev) % nG
    y_of = (th.arange(N, device=dev) // nG) % nY
    c_of = th.arange(N, device=dev) // (nG * nY)
    fq = th.as_tensor([g["finger_q"] for g in grasps], dtype=th.float32, device=dev)  # (G, 2)
    # the partner's footprint toward the robot at each yaw (support fit at d_e), its pieces' planes in its root frame
    pts = np.concatenate(spec.partner_pieces)
    reach_back = np.array([float(_tf(spec.partner_transform(y), pts)[:, 0].min()) for y in yaws])
    hulls = [h for h in (hull_of(p) for p in spec.partner_pieces) if h is not None]
    planes = [(th.as_tensor(h.normals, dtype=th.float32, device=dev), th.as_tensor(h.offsets, dtype=th.float32, device=dev)) for h in hulls]
    # -- the hand vs the slab per d_e at every chain pose (IK-independent), the fingers closed at the grasp's opening
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)

    def hand_check(T, g_idx, sb):
        out = th.ones(T.shape[0], dtype=th.bool, device=dev)
        for gi in th.unique(g_idx).tolist():
            m = g_idx == gi
            out[m] = hc.check(T[m], sb, (float(fq[gi, 0]), float(fq[gi, 1])))[0]
        return out

    hand_ok: Dict[float, th.Tensor] = {}
    onsup: Dict[float, th.Tensor] = {}
    for d_e in cfg.edge_distances:
        sb = RM.slab(height, d_e, cfg)
        ok = th.ones(N, dtype=th.bool, device=dev)
        for k in range(K):
            ok &= hand_check(chain[:, k], g_of, sb)
        hand_ok[d_e] = ok
        m = RM.support_mask(cells, reach_back, d_e)  # (C, Y)
        onsup[d_e] = th.as_tensor(np.repeat(m[:, :, None], nG, axis=2).reshape(-1), device=dev)
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    entry = world.entry()
    links = world.link_of_sphere()
    hand_links = set(policy.finger_links(cfg.arm)) | set(policy.hand_rigid_links(cfg.arm))
    arm_mask = th.tensor([(l not in hand_links) and bool(l) and l.startswith(f"{cfg.arm}_arm") for l in links], device=dev)
    yaw_t = th.as_tensor(yaws, dtype=th.float32, device=dev)
    root_z = float(height) + spec.bottom_offset_m + spec.drop_m
    cells_t = th.as_tensor(cells, dtype=th.float32, device=dev)
    rest_rotation = (th.as_tensor(S.pose_mat([0., 0., 0.], spec.partner_rest)[:3, :3], dtype=th.float32, device=dev)
                     if spec.partner_rest != [0., 0., 0., 1.] else None)

    def partner_clear(qq: th.Tensor, rows: th.Tensor) -> th.Tensor:
        """The arm's spheres (not the hand's) clear of every partner piece, problems ``rows`` at configurations ``qq``."""
        o = th.ones(rows.numel(), dtype=th.bool, device=dev)
        for c0 in range(0, rows.numel(), 2048):
            r = rows[c0:c0 + 2048]
            sp = world.spheres(world.full_q(entry, qq[c0:c0 + 2048], q_lock))[:, arm_mask]
            a = yaw_t[y_of[r]]
            ca, sa = th.cos(a), th.sin(a)
            rel = sp[..., :3] - th.stack([cells_t[c_of[r], 0], cells_t[c_of[r], 1], th.full_like(a, root_z)], 1)[:, None]
            loc = th.stack([ca[:, None] * rel[..., 0] + sa[:, None] * rel[..., 1], -sa[:, None] * rel[..., 0] + ca[:, None] * rel[..., 1], rel[..., 2]], -1)
            if rest_rotation is not None:
                loc = loc @ rest_rotation
            deep = th.full(loc.shape[:2], float("inf"), device=dev)
            for nrm, dd in planes:
                deep = th.minimum(deep, (loc @ nrm.T - dd).max(dim=-1).values)
            o[c0:c0 + 2048] = (((deep - sp[..., 3]) >= 0.0) | (sp[..., 3] <= 0)).all(dim=1)
        return o

    def spheres_ok(qq: th.Tensor, sb) -> th.Tensor:
        o = th.ones(qq.shape[0], dtype=th.bool, device=dev)
        for c0 in range(0, qq.shape[0], 8192):
            ok_, _ = CP.sphere_support_ok(world.spheres(world.full_q(entry, qq[c0:c0 + 8192], q_lock)), links, sb, cfg.arm, policy)
            o[c0:c0 + 8192] = ok_.to(dev)
        return o

    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32)}
    meta_trunks: List[Dict[str, Any]] = []
    t_ik = t_chk = 0.0
    n_ik = 0
    ik_codes = [STATUS["ik_pre"], STATUS["ik_hover"], STATUS["ik_touchdown"], STATUS["ik_seat"]]
    for tr, ti in trunk_grid(height):
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        posture, q_lock = None, None
        for p in cfg.postures:
            q = world.q_config(tr, ti, p)
            if bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                continue
            sph, ln = world.backend.robot_spheres_world(q.unsqueeze(0))
            slab_free = [bool(CP.sphere_support_ok(sph, ln, RM.slab(height, d_e, cfg), cfg.arm, policy)[0][0]) for d_e in cfg.edge_distances]
            if all(slab_free):
                posture, q_lock = p, q
                break
            rec.setdefault("slab_collide_by_posture", {})[p] = slab_free
        rec["posture"] = posture
        if q_lock is None:
            rec.update(eligible=False, reason="every idle-arm posture collides (self or the slab at some d_e)")
            for d_e in cfg.edge_distances:
                arrays[f"s{d_e:.2f}_{key}"] = np.full((nC, nY, nG), STATUS["trunk_ineligible"], dtype=np.uint8)
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        in_reach = (th.linalg.norm(chain[:, :, :3, 3] - sh_t, dim=2) <= L + cfg.reach_margin_m).all(dim=1)
        rec.update(shoulder=[round(float(v), 4) for v in sh], arm_len=round(L, 4))
        any_ok = th.zeros(N, dtype=th.bool, device=dev)
        for d_e in cfg.edge_distances:
            any_ok |= hand_ok[d_e] & onsup[d_e]
        idx = th.nonzero(in_reach & any_ok).flatten()
        rec["n_problems"] = int(idx.numel())
        # -- the chain's IK: the pre with random seeds, each next pose seeded from the previous one (the smallest joint step on the branch)
        ta = time.time()
        why = th.full((idx.numel(),), STATUS["ik_pre"], dtype=th.uint8, device=dev)
        alive = th.zeros(idx.numel(), dtype=th.bool, device=dev)
        qs: List[th.Tensor] = []
        if idx.numel():
            succ, q0 = RM._solve(world, entry, chain[idx, 0], q_lock)
            n_ik += int(idx.numel())
            alive = succ.any(dim=1)
            qs.append(q0[th.arange(idx.numel(), device=dev), th.argmax(succ.int(), dim=1)])
            for k in range(1, K):
                sel = th.nonzero(alive).flatten()
                qk = qs[-1].clone()
                if sel.numel():
                    s2, q2 = RM._solve(world, entry, chain[idx[sel], k], q_lock, seed_active=qs[-1][sel])
                    n_ik += int(sel.numel())
                    step = (q2 - qs[-1][sel][:, None, :]).abs().max(dim=2).values  # (n, S)
                    good = s2 & (step <= cfg.branch_max_rad)
                    pick = th.argmin(th.where(good, step, th.full_like(step, float("inf"))), dim=1)
                    ok2 = good.any(dim=1)
                    fail = ~ok2
                    why[sel[fail & s2.any(dim=1)]] = STATUS["branch_jump"]
                    why[sel[fail & ~s2.any(dim=1)]] = ik_codes[k]
                    alive[sel[fail]] = False
                    qk[sel] = q2[th.arange(sel.numel(), device=dev), pick]
                qs.append(qk)
        t_ik += time.time() - ta
        # -- the descent's interpolated configurations, the spheres / hand hulls against the slab per d_e, the arm against the partner
        tb = time.time()
        sel = th.nonzero(alive).flatten()
        rows = idx[sel]
        confs: List[th.Tensor] = []
        if sel.numel():
            confs = [qs[0][sel] * (1 - t) + qs[1][sel] * t for t in [j / (cfg.approach_configs + 1) for j in range(1, cfg.approach_configs + 1)]]
            confs += [q[sel] for q in qs]
            p_ok = th.ones(sel.numel(), dtype=th.bool, device=dev)
            for qq in confs[:-1]:  # every configuration but the seat (the exact check ignores the partner there)
                p_ok &= partner_clear(qq, rows)
        for d_e in cfg.edge_distances:
            w = why.clone()
            if sel.numel():
                sb = RM.slab(height, d_e, cfg)
                good = th.ones(sel.numel(), dtype=th.bool, device=dev)
                for qq in confs:
                    good &= spheres_ok(qq, sb)
                w_slab = ~good
                for qq in qs:
                    eef = world.eef(world.full_q(entry, qq[sel], q_lock))
                    good &= hand_check(eef, g_of[rows], sb)
                w_hand = ~good & ~w_slab
                good &= p_ok
                w_sel = th.full((sel.numel(),), STATUS["feasible"], dtype=th.uint8, device=dev)
                w_sel = th.where(~p_ok, th.full_like(w_sel, STATUS["partner_arm"]), w_sel)
                w_sel = th.where(w_hand, th.full_like(w_sel, STATUS["hand_policy"]), w_sel)
                w_sel = th.where(w_slab, th.full_like(w_sel, STATUS["slab_spheres"]), w_sel)
                w[sel] = w_sel
            vals = th.full((N,), STATUS["unevaluated"], dtype=th.uint8, device=dev)
            vals[~in_reach] = STATUS["reach_bound"]
            vals[in_reach & ~hand_ok[d_e]] = STATUS["hand_policy"]
            vals[~onsup[d_e]] = STATUS["off_support"]
            here = hand_ok[d_e][idx] & onsup[d_e][idx]
            vals[idx] = th.where(here, w, vals[idx])
            a = vals.view(nC, nY, nG).cpu().numpy()
            arrays[f"s{d_e:.2f}_{key}"] = a
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int((a == STATUS["feasible"]).sum())
            rec.setdefault("n_cells", {})[f"{d_e:.2f}"] = int((a == STATUS["feasible"]).any(axis=(1, 2)).sum())
        t_chk += time.time() - tb
        meta_trunks.append(rec)
    tim.update(ik_s=round(t_ik, 2), checks_s=round(t_chk, 2), ik_problems=n_ik, total_s=round(time.time() - t_all, 2))
    d = spec.data
    meta = {"schema": SCHEMA, "kind": KIND, "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "cell": ("the partner ROOT xy (upright: rest identity, turned by the yaw slot; root z = h + bottom offset + drop)"
                     if spec.partner_rest == [0., 0., 0., 1.] else
                     "the partner ROOT xy (yaw composed with declared rest; root z = h + rotated bottom offset + drop)"),
            "male": {"entry": d["male"]["entry"], "category": lib.category, "model": lib.model, "scale": list(lib.scale), "rest_id": rest_id,
                     "object_key": lib.identity_key(), "geometry_digest": d["male"].get("geometry_digest")},
            "partner": spec.identity()['partner'],
            "spec": {"digest": spec.digest(), "id": d.get("id"), "T_partner_male": d["attachment"]["T_partner_male"], "child_link": d["attachment"].get("child_link"),
                     "parent_link": d["attachment"].get("parent_link"), "insertion_axis": d["insertion_axis"], "seat_axis": d["seat_axis"],
                     "measured": d.get("measured")},
            "hover": dict(params, table=hover), "pre_attach_back_m": float(pre_back_m),
            "chain": "pre (pre_attach_back_m along the insertion axis) -> hover -> touchdown (along the seat axis) -> seat: IK seeded along the chain, "
                     "the smallest joint step <= branch_max_rad; spheres vs the slab at the chain + the pre -> hover interpolations; the arm vs the "
                     "partner at all but the seat; the hand hulls (fingers at the grasp's finger_q) vs the slab at every chain pose",
            "grasps": [{k: v for k, v in g.items() if k != "T"} | {"T_object_eef": S.mat_pose(g["T"])} for g in grasps],
            "grasp_set_digest": grasp_set_digest(grasps), "height": {"value": float(height), "anchor_key": anchor.get("key"), "anchor_surfaces": list(anchor.get("surfaces", []))[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "robot": RM.robot_digest(),
            "slab": {"depth_m": cfg.slab_depth_m, "long_m": cfg.slab_long_m, "thickness_m": cfg.slab_thickness_m},
            "ik": {"seeds": cfg.seeds, "position_tolerance": getattr(world.backend, "position_tolerance", None),
                   "orientation_tolerance": getattr(world.backend, "orientation_tolerance", None),
                   "collision_world": "floor cuboid + self-collision (the slab, the partner checked explicitly; the held male ignored)"},
            "trunks": meta_trunks, "status_codes": STATUS, "timing": tim, "created": S.now()}
    meta["key"] = map_key(meta)
    return meta, arrays


# ------------------------------------------------------------------------------------------------ identity, files, reading


def map_key(meta: Dict[str, Any]) -> str:
    """The map identity (module docstring)."""
    ident = {k: meta[k] for k in ("schema", "kind", "arm", "edge_distances", "config", "policy_key", "robot", "pre_attach_back_m", "grasp_set_digest")}
    ident.update(male={k: meta["male"].get(k) for k in ("entry", "object_key", "rest_id")},
                 partner={k: meta["partner"].get(k) for k in ("category", "model", "scale", "geometry_digest")},
                 spec=meta["spec"]["digest"], hover=hover_identity(meta["hover"]), height=meta["height"]["value"])
    rest = canonical_partner_rest(meta['partner'].get('rest_quat'))
    if rest != [0., 0., 0., 1.]:
        ident['partner']['rest_quat'] = rest
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def save(out_root: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    """``<out_root>/<male entry>/amap__attach__<partner category>__<partner model>__h<h>__<key>.{json,npz}``."""
    d = os.path.join(out_root, str(meta["male"]["entry"]))
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{FILE_PREFIX}{meta['partner']['category']}__{meta['partner']['model']}__h{float(meta['height']['value']):.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    S.save_json(base + ".json", meta)
    return base + ".json"


class AttachMap:
    """One attach map (``.json`` + ``.npz``): status per (edge distance, trunk) of (cell, yaw slot, grasp); reads like a
    ``reach_maps.ReachMap`` for ``map_sampling.as_bands``."""

    def __init__(self, path_json: str):
        self.path = path_json
        with open(path_json) as f:
            self.meta = json.load(f)
        if self.meta.get("schema") != SCHEMA:
            raise S.IncompatibleRecord(f"{path_json}: schema {self.meta.get('schema')!r} is not {SCHEMA}")
        z = np.load(path_json[:-5] + ".npz")
        self.arrays = {k: z[k] for k in z.files}
        self.cells, self.yaws = self.arrays["cells"], self.arrays["yaws"]
        self.lookup: Dict[str, Any] = {}

    @staticmethod
    def find(entry_dir: str, height: float, *, partner: Tuple[str, str, Any], rest_id: str, object_key: Optional[str] = None, arm: str = "right",
             hover: Optional[Dict[str, Any]] = None, pre_back_m: Optional[float] = None, policy_key: Optional[str] = None, robot: Optional[str] = None,
             anchors=None, config_require: Optional[Dict[str, Any]] = None, partner_rest_quat=None) -> Optional["AttachMap"]:
        """The attach map in ``entry_dir`` (the male entry's maps directory) of the partner ``(category, model, scale)`` at the anchor covering
        ``height`` (``anchors``: exact or within the anchor's clipped range; None: the map's own height within 1 mm), the male's rest (and
        identity key), the arm, the hover rule + parameters (:func:`hover_identity`), ``pre_attach_back_m``, the contact policy, the robot,
        the required config; the map's own key over its identity must hold (a hand-edited identity does not serve). None when none matches;
        the hit carries ``lookup`` = ``{how, query_h, map_h}``."""
        target, how = float(height), "height"
        if anchors is not None:
            a, how = anchors.lookup(height)
            if a is None:
                return None
            target = float(a["height"])
        cat, model, scale = partner
        want_scale = list(S.canonical_scale(scale))
        want_rest = canonical_partner_rest(partner_rest_quat)
        want_hover = hover_identity(hover) if hover is not None else None
        for fn in sorted(os.listdir(entry_dir)) if os.path.isdir(entry_dir) else []:
            if not (fn.startswith(FILE_PREFIX) and fn.endswith(".json")):
                continue
            p = os.path.join(entry_dir, fn)
            try:
                with open(p) as f:
                    m = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if m.get("schema") != SCHEMA or m.get("kind") != KIND or m.get("arm") != arm:
                continue
            pm = m.get("partner") or {}
            if pm.get("category") != cat or pm.get("model") != model or list(S.canonical_scale(pm.get("scale"))) != want_scale:
                continue
            try:
                if canonical_partner_rest(pm.get('rest_quat')) != want_rest:
                    continue
            except (TypeError, ValueError):
                continue
            if (m.get("male") or {}).get("rest_id") != rest_id or (object_key is not None and (m.get("male") or {}).get("object_key") != object_key):
                continue
            if want_hover is not None and hover_identity(m.get("hover") or {}) != want_hover:
                continue
            if pre_back_m is not None and abs(float(m.get("pre_attach_back_m", -1.0)) - float(pre_back_m)) > 1e-9:
                continue
            if policy_key is not None and m.get("policy_key") != policy_key:
                continue
            if robot is not None and m.get("robot") != robot:
                continue
            if config_require and any(RM.config_value(m, k) != v for k, v in config_require.items()):
                continue
            if abs(float(m["height"]["value"]) - target) > (1e-6 if anchors is not None else 1e-3):
                continue
            try:
                if m.get("key") != map_key(m):
                    continue
            except (KeyError, TypeError):
                continue
            am = AttachMap(p)
            am.lookup = {"how": how, "query_h": round(float(height), 4), "map_h": float(m["height"]["value"])}
            return am
        return None

    def trunks(self) -> List[Tuple[float, float]]:
        return [(t["translate"], t["tilt"]) for t in self.meta["trunks"] if t.get("eligible")]

    def status(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        d = min(self.meta["edge_distances"], key=lambda v: abs(v - edge_distance))
        if abs(d - edge_distance) > 1e-6:
            raise KeyError(f"edge distance {edge_distance} is not in the map ({self.meta['edge_distances']})")
        return self.arrays[f"s{d:.2f}_t{translate:.2f}_k{tilt:+.1f}"]

    def feasible_cells(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        return self.cells[(self.status(edge_distance, translate, tilt) == STATUS["feasible"]).any(axis=(1, 2))]


def as_bands(am: AttachMap):
    """The attach map as the attach-reach band view the task reads (``map_sampling.as_bands``: one stand-off per edge distance, overhang 0,
    the cells = the partner roots in the base frame, pairs ``[cell, yaw slot, grasp]``), tagged ``map_kind`` "attach"."""
    from ispatialgym.sampling import map_sampling as MS

    view = MS.as_bands(am, overhang=0.0)
    view.meta.update({"map_kind": KIND, "hover_rule": (am.meta.get("hover") or {}).get("rule"), "anchor_lookup": dict(am.lookup or {}),
                      "partner": am.meta.get("partner"), "spec_digest": (am.meta.get("spec") or {}).get("digest")})
    return view


def live_mismatch(meta: Dict[str, Any], live: Dict[str, Any], tol: Dict[str, float] = SPEC_TOL) -> Optional[str]:
    """The map's pair spec against the live pair of the sampler's template (``T_partner_male`` pose, ``insertion_axis`` in the partner root
    frame, ``partner_bottom_offset_m``; optional ``child_link`` / ``parent_link`` names): None when they agree within ``tol``, else why."""
    sp = meta.get("spec") or {}
    if canonical_partner_rest((meta.get('partner') or {}).get('rest_quat')) != canonical_partner_rest(live.get('partner_rest_quat')):
        return "the partner rest orientation differs from the map's"
    T_map = S.pose_of(sp["T_partner_male"])
    if live.get("T_partner_male") is not None:
        p, q = live["T_partner_male"]
        T_live = S.pose_mat([float(v) for v in p], [float(v) for v in q])
        dp = float(np.linalg.norm(T_map[:3, 3] - T_live[:3, 3]))
        dr = _angle(T_map[:3, :3].T @ T_live[:3, :3])
        if dp > tol["pos_m"] or dr > tol["rot_rad"]:
            return f"T_partner_male differs from the map's by {dp * 1000:.1f} mm / {math.degrees(dr):.2f} deg"
    if live.get("insertion_axis") is not None:
        a, b = _unit(sp["insertion_axis"]["local"]), _unit([float(v) for v in live["insertion_axis"]])
        ang = math.acos(max(-1.0, min(1.0, float(a @ b))))
        if ang > tol["axis_rad"]:
            return f"the insertion axis differs from the map's by {math.degrees(ang):.2f} deg"
    if live.get("partner_bottom_offset_m") is not None:
        d = abs(float(live["partner_bottom_offset_m"]) - float((meta.get("partner") or {}).get("bottom_offset_m", math.nan)))
        if not d <= tol["bottom_m"]:
            return f"the partner's bottom offset differs from the map's by {d * 1000:.1f} mm"
    for k in ("child_link", "parent_link"):
        if live.get(k) is not None and sp.get(k) is not None and str(live[k]) != str(sp[k]):
            return f"{k} {live[k]} is not the map's {sp[k]}"
    return None


def sampler_view(s, *, grasp_key: str, partner: Tuple[str, str, Any], top, live: Dict[str, Any], pre_back_m: float, arm: str = "right"):
    """The attach map of the sampler's pair as the attach-reach band view: ``(view or None, MAP_REQUEST fields, why)``. Looked up in
    ``<ISPATIALGYM_REACH_MAPS>/<male entry>`` (the male's v2 entry and stable rest, ``grasp_library``) at the top's height above the floor
    (the anchors of ``ISPATIALGYM_REACH_MAP_ANCHORS``), for ``partner`` = ``(category, model, scale)``, the arm, the oracle's hover rule and
    parameters (:func:`oracle_hover_params`), ``pre_back_m``, the shared contact policy, the robot, ``ISPATIALGYM_REACH_MAP_REQUIRE``; then the
    map's pair spec against ``live`` (:func:`live_mismatch`)."""
    import logging

    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.height_anchors import AnchorSet
    from ispatialgym.sampling.v2 import tabletop as TT

    log = logging.getLogger("ispatialgym.sampling.v2")
    cat, model, scale = partner
    h = float(top.z) - float(getattr(s.world, "floor_z", 0.0))
    params = oracle_hover_params()
    lib = GL.v2_entry(grasp_key) if grasp_key else None
    rid = GL.v2_rest_id(lib) if lib is not None else None
    entry = os.path.basename(os.path.dirname(lib.path)) if lib is not None else None
    req = dict(kind=KIND, entry=entry, rest_id=rid, partner=f"{cat}/{model}", partner_scale=list(S.canonical_scale(scale)), h=round(h, 4), arm=arm,
               hover_rule=params["rule"], pre_back_m=float(pre_back_m), support=str(getattr(top, "name", "")), scene=str(getattr(s.world, "scene_model", "")))
    if canonical_partner_rest(live.get('partner_rest_quat')) != [0., 0., 0., 1.]:
        req['partner_rest_quat'] = canonical_partner_rest(live['partner_rest_quat'])
    if lib is None or rid is None:
        return None, req, f"the male {grasp_key} has no v2 entry with a stable rest"
    root = TT.map_sampling_root()
    if root is None:
        return None, req, "no maps root (ISPATIALGYM_REACH_MAPS)"
    anchors = None
    ap = os.environ.get("ISPATIALGYM_REACH_MAP_ANCHORS")
    if ap and os.path.exists(ap):
        with open(ap) as f:
            an = json.load(f)
        anchors = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
    cfg_req = json.loads(os.environ.get("ISPATIALGYM_REACH_MAP_REQUIRE") or "{}") or None
    am = AttachMap.find(os.path.join(root, entry), h, partner=(cat, model, scale), rest_id=rid, object_key=lib.identity_key(), arm=arm, hover=params,
                        pre_back_m=pre_back_m, policy_key=CP.DEFAULT.key(arm), robot=RM.robot_digest(), anchors=anchors, config_require=cfg_req,
                        partner_rest_quat=live.get('partner_rest_quat'))
    if am is None:
        log.info("attach map miss for %s on %s/%s at h %.4f, %s arm, hover %s (%s)", entry, cat, model, h, arm, params["rule"], os.path.join(root, entry))
        return None, req, "no compatible attach map"
    why = live_mismatch(am.meta, live)
    if why is not None:
        log.info("attach map %s does not serve the live pair: %s", am.path, why)
        return None, dict(req, mismatch=why), f"the map's pair spec does not match the live pair: {why}"
    view = as_bands(am)
    log.info("attach map hit for %s on %s/%s at h %.4f (%s, map h %.4f), %s arm, hover %s: %s", entry, cat, model, h, am.lookup.get("how"),
             float(am.meta["height"]["value"]), arm, params["rule"], am.path)
    return view, req, None


# ------------------------------------------------------------------------------------------------ the pair measurement (Isaac Sim)


def _local_hull(og, obj) -> th.Tensor:
    """``SimWorld.local_hull``: the collision boundary points of every link in the object's root frame (render-side transforms synced)."""
    from ispatialgym.robot import conventions as C

    og.sim.sync_physx_to_fabric()
    rp, rq = obj.get_position_orientation()
    pts = [th.as_tensor(p).float() for p in (link.collision_boundary_points_world for link in obj.links.values()) if p is not None and len(p)]
    world = th.cat(pts, dim=0) if pts else th.zeros(0, 3)
    return C.transform_points(C.invert((rp.float(), rq.float())), world)


def _local_pieces(og, obj) -> List[np.ndarray]:
    """``SimWorld.local_collision_pieces``: the points of every collision mesh of every link, one array per mesh, in the root frame."""
    from ispatialgym.robot import conventions as C

    og.sim.sync_physx_to_fabric()
    rp, rq = obj.get_position_orientation()
    inv = C.invert((rp.float(), rq.float()))
    out = []
    for link in obj.links.values():
        for mesh in link.collision_meshes.values():
            p = mesh.points_in_parent_frame
            if p is None or len(p) == 0:
                continue
            out.append(C.transform_points(inv, link.transform_local_points_to_world(p).float()).numpy().astype(np.float64))
    return out


def _spawn(og, scene, name: str, category: str, model: str, scale, xy):
    from omnigibson.objects.dataset_object import DatasetObject

    from ispatialgym.sampling import knowledge as K

    kw: Dict[str, Any] = {}
    sc = S.canonical_scale(scale)
    if sc != (1.0, 1.0, 1.0):
        kw["scale"] = th.tensor(sc, dtype=th.float32)
    abilities = K.abilities_of_category(category)
    abilities["attachable"] = {}
    obj = DatasetObject(name=name, category=category, model=model, dataset_name="behavior-1k-assets", fixed_base=False, abilities=abilities, **kw)
    scene.add_object(obj)
    obj.set_position_orientation(th.tensor([float(xy[0]), float(xy[1]), 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj.keep_still()
    return obj


def measure_pair(og, scene, job: Dict[str, Any], glib_root: str) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """The pair spec of ``job`` (``{"id", "male_entry", "partner": {"category", "model", "scale"}}``), measured with both objects spawned
    ``attachable`` in ``scene`` (far from the robot, removed afterwards) by the task's own code: ``attachment.attachment_links`` (the pair),
    ``root_in_child_link`` / ``parent_link_in_female`` as the template reads them, ``attachment.insertion_axis_rule`` on the partner's
    ``local_hull``, ``predicates.seat_axis`` (the oracle's), the partner's collision pieces (hull vertices) and bottom offset; the male's
    live pieces against its entry's ``geometry.json`` (``measured.male_geometry_max_dev_m``). Returns ``(spec data, partner pieces)``."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import predicates as PRED
    from ispatialgym.sampling.v2.tasks import attachment as AT

    mdir = os.path.join(glib_root, "models", job["male_entry"])
    lib = S.load_json(os.path.join(mdir, "library.json"))
    if lib is None:
        raise FileNotFoundError(f"{mdir}: no library.json")
    pj = job["partner"]
    tag = int(time.time() * 1000) % 100000
    male = _spawn(og, scene, f"am_male_{tag}", lib["category"], lib["model"], lib.get("scale"), (30.0, 30.0))
    female = _spawn(og, scene, f"am_female_{tag}", pj["category"], pj["model"], pj.get("scale"), (30.0, 32.0))
    try:
        og.sim.step()
        og.sim.step_physics()
        og.sim.sync_physx_to_fabric()
        from omnigibson.object_states import AttachedTo

        if AttachedTo not in male.states or AttachedTo not in female.states:
            raise ValueError(f"{job['id']}: the pair has no AttachedTo state")
        pair = AT.attachment_links(male, female)
        if pair is None:
            raise ValueError(f"{job['id']}: {lib['category']}/{lib['model']} has no male link matching a female link of {pj['category']}/{pj['model']}")
        child_link, parent_link = pair
        cp, cq = child_link.get_position_orientation()
        rp, rq = male.get_position_orientation()
        root_in_child_link = C.relative((rp.float(), rq.float()), (cp.float(), cq.float()))
        fp, fq = female.get_position_orientation()
        pp, pq = parent_link.get_position_orientation()
        parent_link_in_female = C.relative((pp.float(), pq.float()), (fp.float(), fq.float()))
        T_pm = C.compose(parent_link_in_female, root_in_child_link)
        hull_f = _local_hull(og, female)
        axis_ins, facts = AT.insertion_axis_rule(parent_link_in_female, hull_f)
        seat_w = PRED.seat_axis(male, female, parent_link.body_name).float()
        seat_local = C.quat_rotate(C.qinv(fq.float()), seat_w)
        pieces = [hull_of(p).verts for p in _local_pieces(og, female) if hull_of(p) is not None]
        live_male = _local_pieces(og, male)
        try:
            glib_male = male_pieces(mdir)
            dev = max(float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max()) for a, b in zip(live_male, glib_male)) if len(live_male) == len(glib_male) else None
        except (FileNotFoundError, ValueError):
            glib_male, dev = None, None
        rest = canonical_partner_rest(pj.get('rest_quat'))
        bottom = -float(hull_f[:, 2].min() if rest == [0., 0., 0., 1.] else
                        C.quat_rotate(th.tensor(rest), hull_f)[:, 2].min())
        data = {"schema": SPEC_SCHEMA, "id": job["id"], "frame": "partner root (upright, rest identity)",
                "male": {"entry": job["male_entry"], "category": lib["category"], "model": lib["model"], "scale": list(S.canonical_scale(lib.get("scale"))),
                         "geometry_digest": pieces_digest(glib_male) if glib_male is not None else None},
                "partner": {"category": pj["category"], "model": pj["model"], "scale": list(S.canonical_scale(pj.get("scale"))),
                            "geometry_digest": pieces_digest(pieces), "bottom_offset_m": round(bottom, 6), "drop_m": PARTNER_DROP_M, "n_pieces": len(pieces)},
                "attachment": {"child_link": child_link.body_name, "parent_link": parent_link.body_name,
                               "T_partner_male": S.mat_pose(S.pose_mat(T_pm[0].tolist(), T_pm[1].tolist())),
                               "root_in_child_link": S.mat_pose(S.pose_mat(root_in_child_link[0].tolist(), root_in_child_link[1].tolist())),
                               "parent_link_in_partner": S.mat_pose(S.pose_mat(parent_link_in_female[0].tolist(), parent_link_in_female[1].tolist()))},
                "insertion_axis": {"local": [round(float(v), 6) for v in axis_ins], "facts": facts, "rule": "attachment.insertion_axis_rule (defaults)"},
                "seat_axis": {"local": [round(float(v), 6) for v in seat_local], "rule": "predicates.seat_axis"},
                "measured": {"male_geometry_max_dev_m": dev, "male_live_pieces": len(live_male), "partner_hull_points": int(hull_f.shape[0]),
                             "seat_vs_insertion_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(_unit(seat_local) @ _unit(axis_ins)))))), 3),
                             "created": S.now()}}
        if rest != [0., 0., 0., 1.]:
            data['partner']['rest_quat'] = rest
            data['frame'] = 'partner root (yaw composed with explicitly measured rest)'
        return data, pieces
    finally:
        for o in (male, female):
            try:
                scene.remove_object(o)
            except Exception:  # noqa: BLE001
                pass
        og.sim.step()


# ------------------------------------------------------------------------------------------------ CLI


def _build_cli(a) -> None:
    """``build``: per pair of ``--pairs`` (``[{"id", "male_entry", "partner": {"category", "model", "scale"}, "heights": [anchor keys]}]``) the
    spec (``<out>/specs/<id>/attach_spec.json``, measured when missing or with ``--remeasure``), then one map per height x ``--rules`` for
    ``--arm`` into ``<out>/maps/<male entry>/``. Logs ``ATTACH_SPEC`` / ``ATTACH_MAP`` json lines (``ok`` true / false) and ``ATTACH_DONE``."""
    from ispatialgym.grasping import hand_geometry as H
    from ispatialgym.sampling.generation_defaults import defaults_for
    from ispatialgym.sampling.height_anchors import AnchorSet

    with open(a.pairs) as f:
        jobs = json.load(f)
    if a.only:
        jobs = [j for j in jobs if j["id"] in set(a.only)]
    with open(a.anchors) as f:
        an = json.load(f)
    aset = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
    by_key = {x["key"]: x for x in aset.anchors}
    pre_back = float(a.pre_back) if a.pre_back is not None else float(defaults_for("configuration_matching-attachment")["pre_attach_back_m"])
    cfg = RM.MapConfig(mode="full", lattice_m=a.lattice, seeds=a.seeds, batch=a.batch, arm=a.arm)
    world = RM.SynthWorld(cfg)
    hand = H.load(a.arm)
    spec_root, maps_root = os.path.join(a.out, "specs"), os.path.join(a.out, "maps")
    for j in jobs:
        sp = spec_path(spec_root, j["id"])
        try:
            if a.remeasure or not os.path.exists(sp):
                data, pieces = measure_pair(world.og, world.env.scene, j, a.glib)
                save_spec(spec_root, j["id"], data, pieces)
            spec = load_spec(sp)
            print(f"ATTACH_SPEC {json.dumps({'id': j['id'], 'ok': True, 'path': sp, 'digest': spec.digest(), 'T_partner_male': spec.data['attachment']['T_partner_male'], 'insertion_axis': spec.data['insertion_axis'], 'seat_axis': spec.data['seat_axis'], 'partner': spec.data['partner'], 'measured': spec.data.get('measured')})}", flush=True)
        except Exception as e:  # noqa: BLE001 - the next pair goes on; the log names the failure
            import traceback

            print(f"ATTACH_SPEC {json.dumps({'id': j['id'], 'ok': False, 'error': repr(e)[:500], 'tb': traceback.format_exc()[-1500:]})}", flush=True)
            continue
        mdir = os.path.join(a.glib, "models", j["male_entry"])
        d = S.load_json(os.path.join(mdir, "library.json"))
        from ispatialgym.sampling import grasp_library as GL

        # the entry loaded as the sampler's lookup loads it (grasp_library.v2_entry: the rest hint of the record, else its candidates.json)
        lib = S.load_model(d["category"], d["model"], d.get("scale"), a.glib, constraint=d.get("constraint") or "tabletop",
                           rest_hint_wxyz=d.get("rest_hint_wxyz") if "rest_hint_wxyz" in d else GL._v2_hint(os.path.join(mdir, "library.json")))
        if os.path.realpath(os.path.dirname(lib.path)) != os.path.realpath(mdir):
            print(f"ATTACH_MAP {json.dumps({'id': j['id'], 'ok': False, 'reason': f'the entry resolves to {lib.path}, not {mdir}'})}", flush=True)
            continue
        rid = GL.v2_rest_id(lib)
        male = male_pieces(mdir)
        for hk in j.get("heights") or a.heights or []:
            anc = by_key.get(hk)
            for rule in a.rules:
                rec = {"id": j["id"], "male_entry": j["male_entry"], "h": hk, "arm": a.arm, "rule": rule}
                if anc is None:
                    print(f"ATTACH_MAP {json.dumps(dict(rec, ok=False, reason='not an anchor key of --anchors'))}", flush=True)
                    continue
                try:
                    params = oracle_hover_params(rule)
                    meta, arrays = compute_attach_map(world, spec, lib, rid, float(anc["height"]), anc, male=male, hand=hand, params=params,
                                                      pre_back_m=pre_back, cfg=cfg)
                    p = save(maps_root, meta, arrays)
                    nf = {t["key"]: t.get("n_cells") for t in meta["trunks"] if t.get("eligible")}
                    hv = [[{k: c.get(k) for k in ("grasp", "h_m", "touchdown_m", "first_touch_m", "box_h_m", "source")} for c in row] for row in meta["hover"]["table"][:1]]
                    print(f"ATTACH_MAP {json.dumps(dict(rec, ok=True, path=p, key=meta['key'], timing=meta['timing'], n_grasps=len(meta['grasps']), cells=nf, hover_yaw0=hv))}", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print(f"ATTACH_MAP {json.dumps(dict(rec, ok=False, error=repr(e)[:500], tb=traceback.format_exc()[-1500:]))}", flush=True)
    print("ATTACH_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ispatialgym.sampling.attach_maps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="measure the pairs' specs and compute their attach maps (Isaac Sim + cuRobo)")
    b.add_argument("--pairs", required=True)
    b.add_argument("--glib", required=True, help="the v2 library root the male entries are in (<glib>/models/<entry>)")
    b.add_argument("--anchors", required=True)
    b.add_argument("--heights", nargs="*", default=None, help="anchor keys for pairs without their own 'heights'")
    b.add_argument("--rules", nargs="+", default=["box", "swept"], choices=list(HOVER_RULES))
    b.add_argument("--arm", default="right")
    b.add_argument("--out", required=True, help="<out>/specs/<id>/, <out>/maps/<male entry>/")
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--remeasure", action="store_true")
    b.add_argument("--pre-back", type=float, default=None, help="pre_attach_back_m (default: the attachment generation default)")
    b.add_argument("--lattice", type=float, default=0.08)
    b.add_argument("--seeds", type=int, default=8)
    b.add_argument("--batch", type=int, default=2048)
    i = sub.add_parser("info", help="an attach map's summary")
    i.add_argument("map")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        _build_cli(a)
        try:
            import omnigibson as og

            og.shutdown()
        except Exception:  # noqa: BLE001
            pass
    else:
        m = AttachMap(a.map)
        print(json.dumps({k: m.meta[k] for k in ("kind", "arm", "male", "partner", "height", "edge_distances", "pre_attach_back_m", "key")}, indent=1))
        print(json.dumps({k: v for k, v in m.meta["hover"].items() if k != "table"}))
        print(json.dumps([{"trunk": t["key"], "n_cells": t.get("n_cells")} for t in m.meta["trunks"] if t.get("eligible")]))


if __name__ == "__main__":
    main()
