"""Settle first, then generate (of-v2maps 2026-11-07, owner prompt "settle before generating candidates"; opt-in ``GB_SETTLE_FIRST=1``).

Today the candidates are generated geometrically at the DESIGNATED rest and physics settles the object only afterwards, in validation;
an object that settles elsewhere turns into measured-rest rejections (roles_all_20261102: ``geometry_at_measured_rest:finger_penetration``
413 trials, 389 of them at a non-intended rest; ``palm_contact`` 228, 196). With the switch the pipeline first drops the object on the
slab several times (:data:`SURVEY_YAWS_DEG` x :data:`SURVEY_DROPS_M`), settles it, and clusters the outcomes into distinct stable rests
(:func:`cluster_rests`); candidates are then generated at each measured rest (``pipeline.stage_settle``), and validation's own settle
stays as the guard (its ``geometry_at_measured_rest`` rejections count how often it still fires).

Clustering: two outcomes are one rest when the object-frame up vectors are within ``tilt_tol_rad``, or when they differ only by a roll
about a near-round axis of the object (:func:`roll_equivalent`: both up vectors perpendicular to one principal axis within
``axis_tol_rad``, the heights within ``h_tol_m``, and the hull's cross-section about that axis round within ``round_tol`` - a
tolerance, not exact symmetry detection, since a not-quite-round collision mesh (the plate: rim 0.113-0.128 m) defeats the exact one).
Yaw is free: a rest is stored yaw-free (``q = R_z(-yaw(q)) q``)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import schema as S

METHOD = "settle_first/1"
SURVEY_YAWS_DEG: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
SURVEY_DROPS_M: Tuple[float, ...] = (0.003, 0.02, 0.05)
TILT_TOL_RAD = 0.10
H_TOL_M = 0.003
AXIS_TOL_RAD = math.radians(8.0)
ROUND_TOL = 0.15
MIN_OUTCOMES = 2  # a rest seen fewer times (and not the most frequent) gets no job of its own


def survey_plan(yaws_deg: Sequence[float] = SURVEY_YAWS_DEG, drops_m: Sequence[float] = SURVEY_DROPS_M) -> List[Tuple[float, float]]:
    """The drops of the survey: (initial yaw rad, drop height m)."""
    return [(math.radians(float(y)), float(d)) for d in drops_m for y in yaws_deg]


def yaw_free(q_xyzw) -> np.ndarray:
    """``q`` with its yaw about +z removed (``R_z(-yaw) q``), xyzw, w >= 0."""
    q = np.asarray(q_xyzw, dtype=float)
    r = S.quat_mul(S.quat_yaw(-S.yaw_of_quat(q)), q)
    r = r / np.linalg.norm(r)
    return -r if r[3] < 0 else r


def up_in_object(q_xyzw) -> np.ndarray:
    """The world +z in the object frame at orientation ``q`` (the support normal as the object sees it)."""
    return S.quat_to_mat(np.asarray(q_xyzw, dtype=float)).T @ np.array([0.0, 0.0, 1.0])


def principal_axes(V: np.ndarray) -> np.ndarray:
    """The hull's principal axes (rows, unit), from the vertex covariance about the centroid."""
    X = np.asarray(V, dtype=float)
    X = X - X.mean(axis=0)
    w, E = np.linalg.eigh(X.T @ X / max(len(X), 1))
    return E.T


def roundness(V: np.ndarray, axis: np.ndarray, n: int = 36) -> float:
    """How far the hull's cross-section about ``axis`` (through the centroid) is from round: the relative spread (max - min) / max of its
    support function over ``n`` directions perpendicular to the axis (a box's corners are all equidistant from its axis; its support
    function is not)."""
    X = np.asarray(V, dtype=float)
    X = X - X.mean(axis=0)
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    h = np.array([float((X @ (math.cos(t) * e1 + math.sin(t) * e2)).max()) for t in np.linspace(0.0, 2 * math.pi, n, endpoint=False)])
    return 1.0 if h.max() <= 1e-9 else float((h.max() - h.min()) / h.max())


def roll_equivalent(u1: np.ndarray, u2: np.ndarray, h1: float, h2: float, V: np.ndarray, axis_tol_rad: float = AXIS_TOL_RAD,
                    h_tol_m: float = H_TOL_M, round_tol: float = ROUND_TOL) -> Tuple[bool, Dict[str, Any]]:
    """Whether two rests (object-frame up vectors ``u1`` / ``u2``, root heights ``h1`` / ``h2``) differ only by a roll about a near-round
    principal axis of the hull ``V``: ``(equivalent, facts)``."""
    if abs(float(h1) - float(h2)) > h_tol_m:
        return False, {"why": "height"}
    s = math.sin(axis_tol_rad)
    for k, a in enumerate(principal_axes(V)):
        if abs(float(np.dot(u1, a))) <= s and abs(float(np.dot(u2, a))) <= s:
            rn = roundness(V, a)
            if rn <= round_tol:
                return True, {"axis": [round(float(v), 4) for v in a], "axis_index": k, "roundness": round(rn, 4)}
            return False, {"why": "not round", "roundness": round(rn, 4)}
    return False, {"why": "no common axis"}


def cluster_rests(outcomes: Sequence[Dict[str, Any]], V: np.ndarray, designated_xyzw=None, tilt_tol_rad: float = TILT_TOL_RAD,
                  h_tol_m: float = H_TOL_M, axis_tol_rad: float = AXIS_TOL_RAD, round_tol: float = ROUND_TOL) -> List[Dict[str, Any]]:
    """The survey's outcomes (validate_tabletop rest records + ``yaw0`` / ``drop_m``) clustered into distinct rests, most frequent first
    (ties: the one nearer the designated rest). Each cluster: ``rest_id`` (s0, s1, ...), ``quat_xyzw`` (the yaw-free orientation of its
    median-height stable member), ``h_root`` (mean) / ``h_root_spread_m``, ``n`` outcomes, ``n_stable``, ``stability`` (n_stable / n),
    ``tilt_from_designated_rad``, ``roll_merged`` (outcomes joined by :func:`roll_equivalent`, with its facts), ``members``, ``method``."""
    clusters: List[Dict[str, Any]] = []
    for o in outcomes:
        u = up_in_object(o["quat_xyzw"])
        h = float(o["h_root"])
        home, how = None, None
        for c in clusters:
            if math.acos(max(-1.0, min(1.0, float(np.dot(u, c["_u"]))))) <= tilt_tol_rad and abs(h - c["_h"]) <= h_tol_m:
                home, how = c, None
                break
            eq, facts = roll_equivalent(u, c["_u"], h, c["_h"], V, axis_tol_rad, h_tol_m, round_tol)
            if eq:
                home, how = c, facts
                break
        if home is None:
            home = {"_u": u, "_h": h, "members": [], "roll_merged": []}
            clusters.append(home)
        home["members"].append(o)
        if how is not None:
            home["roll_merged"].append({"yaw0": o.get("yaw0"), "drop_m": o.get("drop_m"), **how})
    q_des = None if designated_xyzw is None else yaw_free(designated_xyzw)
    out = []
    for c in clusters:
        ms = c["members"]
        st = [m for m in ms if m.get("stable")]
        # the representative: the stable member nearest the designated rest (a roll-merged cluster's members differ by the roll), else the
        # median height
        pool = st or ms
        pick = (min(pool, key=lambda m: S.tilt_distance(yaw_free(m["quat_xyzw"]), q_des)) if q_des is not None
                else sorted(pool, key=lambda m: float(m["h_root"]))[len(pool) // 2])
        q = yaw_free(pick["quat_xyzw"])
        hs = [float(m["h_root"]) for m in ms]
        rec = {"quat_xyzw": [round(float(v), 7) for v in q], "h_root": round(float(np.mean(hs)), 6),
               "h_root_spread_m": round(float(max(hs) - min(hs)), 6), "n": len(ms), "n_stable": len(st),
               "stability": round(len(st) / len(ms), 3), "roll_merged": c["roll_merged"],
               "tilt_from_designated_rad": None if q_des is None else round(float(S.tilt_distance(q, q_des)), 5),
               "members": [{"yaw0": m.get("yaw0"), "drop_m": m.get("drop_m"), "stable": bool(m.get("stable")), "h_root": m.get("h_root"),
                            "tilt_from_intended_rad": m.get("tilt_from_intended_rad")} for m in ms],
               "method": f"{METHOD}: drops {len(outcomes)}, tilt_tol {tilt_tol_rad} rad, h_tol {h_tol_m} m, roll about a principal axis "
                         f"(axis_tol {round(axis_tol_rad, 4)} rad, round_tol {round_tol})"}
        out.append(rec)
    out.sort(key=lambda r: (-r["n_stable"], -r["n"], r["tilt_from_designated_rad"] if r["tilt_from_designated_rad"] is not None else 0.0))
    for k, r in enumerate(out):
        r["rest_id"] = f"s{k}"
    return out


def jobs_for_rests(job: Dict[str, Any], clusters: Sequence[Dict[str, Any]], min_outcomes: int = MIN_OUTCOMES) -> List[Dict[str, Any]]:
    """The jobs generated from a survey: the job itself designated at its most frequent stable rest (same id: the downstream maps /
    sampler keep their entry), and a derived job (its own id, ``derived_from``) per other stable rest seen at least ``min_outcomes``
    times. The main job carries the measured rest as ``generation_rest_wxyz`` (its ``rest_hint_wxyz``, which names its directory, unchanged),
    a derived job as its ``rest_hint_wxyz``; both carry ``settle_first`` (the cluster without its members)."""
    out: List[Dict[str, Any]] = []
    stable = [c for c in clusters if c["n_stable"] > 0]
    for k, c in enumerate(stable):
        if k > 0 and c["n_stable"] < min_outcomes:
            continue
        x, y, z, w = c["quat_xyzw"]
        hint = [round(w, 6), round(x, 6), round(y, 6), round(z, 6)]
        meta = {kk: v for kk, v in c.items() if kk != "members"}
        if k == 0:
            # the job's own id AND directory (schema.job_dir derives both from rest_hint_wxyz): the designated hint stays, the measured rest is
            # generation_rest_wxyz (candidates.generate / validate_tabletop.run_model read it first)
            out.append(dict(job, generation_rest_wxyz=hint, rest_pose="settle_first", settle_first=meta, designated_rest_hint_wxyz=job.get("rest_hint_wxyz")))
        else:
            nj = dict(job, rest_hint_wxyz=hint, rest_pose="settle_first", settle_first=meta, derived_from=job["id"],
                      designated_rest_hint_wxyz=job.get("rest_hint_wxyz"))
            nj["id"] = S.entry_id(nj["category"], nj["model"], nj["scale"], nj.get("constraint") or "tabletop", hint)
            out.append(nj)
    return out
