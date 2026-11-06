"""Grasp symmetry families: one candidate stands for a continuum of geometrically equivalent grasps.

* **rotational** - the grasp turned about an axis through a centre (a cup rim, a round lid's edge, a bottle's body): ``axis``,
  ``center`` and the valid angular ``interval`` (rad, relative to the representative) in the object frame;
* **translational** - the grasp slid along an axis (a straight handle or bar, a rod): ``axis`` and the valid ``interval`` (m).

Detection is geometric and LOCAL (:func:`detect`): the object's surface around the pads (``local_radius_m`` of the pad centre) must
map onto the object's surface under the transformation (p95 deviation <= ``tol_m``), and the transformed grasp must keep its
contact (width within ``width_tol_m``) and its clearance from the object. The interval grows in ``step`` increments from the
representative until a test fails - a cup's rim family stops before the handle. Rotation axes are fitted to the local cross-section
(a circle fit of the surface points in the plane normal to the axis at the pad height), so a handle that shifts the centroid does not
shift the rim's axis.

**One family = one candidate**: :func:`dedup` folds every candidate that is a family member (or a stored finger-swap twin) of an
earlier, better-ranked one into that family. Only the representative is physics-tested; a family record says so
(``physics: "representative_only"``) - the members are identified geometrically / by review, and the reach-band and sampling stages
check reachability and scene constraints for the member they choose.

A rotational family whose axis is parallel to the rest's vertical is ``vertical_yaw_equivalent``: its member at angle t at object yaw
psi has the same world eef pose as the representative at yaw psi + t with the object root moved by ``R(psi)c_h - R(psi + t)c_h``
(``c_h`` = the axis offset from the root in the horizontal plane), so reach maps store the representative per yaw plus that mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import schema as S

METHOD = "local_surface_invariance/1"


@dataclass
class SymmetryConfig:
    tol_m: float = 0.0015
    local_radius_m: float = 0.05
    width_tol_m: float = 0.003
    rot_step_rad: float = math.radians(10.0)
    trans_step_m: float = 0.005
    min_rot_interval_rad: float = math.radians(30.0)
    min_trans_interval_m: float = 0.02
    fit_slice_m: float = 0.005
    max_trans_m: float = 0.40
    n_surface: int = 20000
    dedup_pos_m: float = 0.015
    dedup_rot_deg: float = 20.0


class SurfaceIndex:
    """Dense surface samples of a mesh with a KD-tree: distance-to-surface queries (one-sided, sample spacing ~ sqrt(area / n))."""

    def __init__(self, mesh, n: int = 20000, seed: int = 0):
        from scipy.spatial import cKDTree

        pts, _ = _sample_surface(mesh, n, seed)
        self.points = np.concatenate([pts, np.asarray(mesh.vertices)])
        self.tree = cKDTree(self.points)
        self.spacing = math.sqrt(max(mesh.area, 1e-8) / n)

    def distance(self, p: np.ndarray) -> np.ndarray:
        d, _ = self.tree.query(p, k=1)
        return d

    def near(self, c: np.ndarray, r: float) -> np.ndarray:
        idx = self.tree.query_ball_point(c, r)
        return self.points[idx]


def _sample_surface(mesh, n: int, seed: int):
    import trimesh

    return trimesh.sample.sample_surface(mesh, n, seed=seed)


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def fit_axis_center(points: np.ndarray, axis: np.ndarray, c0: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Circle fit (Kasa) of ``points`` projected on the plane normal to ``axis``: ``(centre on the plane through c0, rms residual)``."""
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    P = points - c0
    x, y = P @ u, P @ v
    if len(x) < 8:
        return None, float("inf")
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    res = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    if r < 1e-3:
        return None, float("inf")
    return c0 + cx * u + cy * v, float(np.sqrt(np.mean(res ** 2)))


def candidate_axes(union, rest_R: Optional[np.ndarray] = None) -> List[Tuple[str, np.ndarray]]:
    """Rotation-axis directions to test (object frame): the rest's vertical and the mesh's principal axes."""
    axes: List[Tuple[str, np.ndarray]] = []
    if rest_R is not None:
        axes.append(("rest_vertical", np.asarray(rest_R)[:3, :3].T @ np.array([0, 0, 1.0])))
    else:
        axes.append(("z", np.array([0, 0, 1.0])))
    try:
        _, vecs = np.linalg.eigh(np.cov((union.vertices - union.vertices.mean(0)).T))
        for k in range(3):
            v = vecs[:, k]
            if all(abs(float(np.dot(v, a))) < 0.98 for _, a in axes):
                axes.append((f"principal_{k}", v / np.linalg.norm(v)))
    except Exception:  # noqa: BLE001
        pass
    return axes


def _local_ok(idx: SurfaceIndex, local: np.ndarray, M: np.ndarray, tol: float) -> bool:
    if len(local) < 8:
        return False
    q = local @ M[:3, :3].T + M[:3, 3]
    d = idx.distance(q)
    return float(np.percentile(d, 95)) <= tol + idx.spacing


def detect(T: np.ndarray, union, idx: SurfaceIndex, width_fn, clearance_fn, rest_R: Optional[np.ndarray] = None,
           cfg: SymmetryConfig = SymmetryConfig(), contact_z: float = -0.01186, support_fn=None) -> Dict[str, Any]:
    """The largest family of the grasp ``T`` (object frame): ``{"type": "none"}`` when neither a rotational nor a translational
    interval reaches its minimum. ``width_fn(Ts) -> widths``, ``clearance_fn(Ts) -> object clearances`` judge transformed grasps;
    ``support_fn(Ts) -> bool`` (the contact policy against the rest's support, pre-grasp / approach / sweep) bounds the interval too:
    turning a grasp about a horizontal axis tilts the hand toward the table."""
    c = T[:3, 3] + T[:3, 2] * contact_z
    local = idx.near(c, cfg.local_radius_m)
    w0 = float(width_fn(T[None])[0])
    best: Dict[str, Any] = {"type": "none"}
    best_measure = 0.0

    def member_ok(M: np.ndarray) -> bool:
        if not _local_ok(idx, local, M, cfg.tol_m):
            return False
        Tm = M @ T
        w = float(width_fn(Tm[None])[0])
        if not math.isfinite(w) or abs(w - w0) > cfg.width_tol_m:
            return False
        if float(clearance_fn(Tm[None])[0]) < 0.0:
            return False
        return support_fn is None or bool(support_fn(Tm[None])[0])

    for name, a in candidate_axes(union, rest_R):
        an = a / np.linalg.norm(a)
        ring = local[np.abs((local - c) @ an) <= cfg.fit_slice_m]  # the cross-section at the pads' height along the axis
        ctr, rms = fit_axis_center(ring, a, c)
        if ctr is None or rms > 2 * cfg.tol_m:
            continue
        lim = []
        for sgn in (1.0, -1.0):
            k = 0
            while k * cfg.rot_step_rad < math.pi:
                ang = sgn * (k + 1) * cfg.rot_step_rad
                M = np.eye(4)
                R = _rot(a, ang)
                M[:3, :3] = R
                M[:3, 3] = ctr - R @ ctr
                if not member_ok(M):
                    break
                k += 1
            lim.append(sgn * k * cfg.rot_step_rad)
        lo, hi = min(lim), max(lim)
        if hi - lo >= 2 * math.pi - cfg.rot_step_rad - 1e-9:
            lo, hi = -math.pi, math.pi
        if hi - lo >= cfg.min_rot_interval_rad:
            radius = float(np.linalg.norm((c - ctr) - np.dot(c - ctr, a) * a))
            measure = (hi - lo) * max(radius, 0.01)  # arc length swept by the pad centre
            if measure > best_measure:
                vert = rest_R is not None and abs(float(np.dot(a, np.asarray(rest_R)[:3, :3].T @ np.array([0, 0, 1.0])))) > 0.995
                full = hi - lo >= 2 * math.pi - 1e-9
                best = {"type": "rotational", "axis": a.round(6).tolist(), "center": ctr.round(6).tolist(),
                        "interval": [-math.pi, math.pi] if full else [round(lo, 5), round(hi, 5)], "full_turn": bool(full),
                        "axis_source": name, "fit_rms_m": round(rms, 5), "radius_m": round(radius, 4), "vertical_yaw_equivalent": bool(vert)}
                best_measure = measure
    # translation along the pad-width axis (eef x: a bar crossed by the pinch) and along the approach-normal object axes
    for name, t in (("eef_x", T[:3, 0]),):
        lim = []
        for sgn in (1.0, -1.0):
            k = 0
            while (k + 1) * cfg.trans_step_m <= cfg.max_trans_m:
                M = np.eye(4)
                M[:3, 3] = t * sgn * (k + 1) * cfg.trans_step_m
                if not member_ok(M):
                    break
                k += 1
            lim.append(sgn * k * cfg.trans_step_m)
        lo, hi = min(lim), max(lim)
        if hi - lo >= cfg.min_trans_interval_m and (hi - lo) > best_measure:
            best = {"type": "translational", "axis": (t / np.linalg.norm(t)).round(6).tolist(), "origin": c.round(6).tolist(),
                    "interval": [round(lo, 5), round(hi, 5)], "axis_source": name}
            best_measure = hi - lo
    if best["type"] != "none":
        best["provenance"] = {"method": METHOD, "tol_m": cfg.tol_m, "local_radius_m": cfg.local_radius_m,
                              "rot_step_rad": cfg.rot_step_rad, "trans_step_m": cfg.trans_step_m, "reviewed": None}
    return best


def _pose_close(A: np.ndarray, B: np.ndarray, pos_m: float, rot_deg: float) -> bool:
    if np.linalg.norm(A[:3, 3] - B[:3, 3]) > pos_m:
        return False
    R = A[:3, :3].T @ B[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return ang <= rot_deg


def twin_T(T: np.ndarray) -> np.ndarray:
    """The finger-swapped twin (``conventions.twin_pose``): the eef turned pi about its approach (+z)."""
    Rz = np.diag([-1.0, -1.0, 1.0, 1.0])
    return T @ Rz


def member_param(family: Dict[str, Any], T_rep: np.ndarray, T: np.ndarray, cfg: SymmetryConfig) -> Optional[float]:
    """The family parameter at which ``T`` coincides with a member (pose tolerance of ``cfg``), else None."""
    kind = family.get("type")
    if kind not in ("rotational", "translational"):
        return None
    lo, hi = family["interval"]
    if kind == "rotational":
        grid = np.arange(lo, hi + 1e-9, math.radians(2.0))
    else:
        grid = np.arange(lo, hi + 1e-9, 0.0025)
    for p in grid:
        if _pose_close(S.family_member_T(family, T_rep, float(p)), T, cfg.dedup_pos_m, cfg.dedup_rot_deg):
            return float(p)
    return None


def dedup(cands: List[Dict[str, Any]], families: Dict[str, Dict[str, Any]], cfg: SymmetryConfig = SymmetryConfig()) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Fold family members and stored twins into the earliest (best-ranked) representative. ``cands`` (score order) carry
    ``name`` and ``T`` (4x4 object frame); ``families[name]`` is the detected family of each. Returns the representatives (with
    ``family_id``) and the family records (``members`` = folded candidate names with their parameter, ``twins`` = folded twins)."""
    reps: List[Dict[str, Any]] = []
    fams: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        T = np.asarray(c["T"], dtype=float)
        hit = None
        for r in reps:
            fam = fams[r["family_id"]]
            Tr = np.asarray(r["T"], dtype=float)
            if _pose_close(Tr, T, cfg.dedup_pos_m, cfg.dedup_rot_deg):
                hit = (r, "duplicate", 0.0)
                break
            if _pose_close(twin_T(Tr), T, cfg.dedup_pos_m, cfg.dedup_rot_deg):
                hit = (r, "twin", 0.0)
                break
            p = member_param(fam, Tr, T, cfg)
            if p is None:
                p_t = member_param(fam, twin_T(Tr), T, cfg)
                if p_t is not None:
                    hit = (r, "twin_member", p_t)
                    break
            else:
                hit = (r, "member", p)
                break
        if hit is not None:
            r, how, p = hit
            fams[r["family_id"]].setdefault("members", []).append({"name": c["name"], "relation": how, "param": round(float(p), 5)})
            continue
        fid = f"f{len(fams)}"
        fam = dict(families.get(c["name"], {"type": "none"}))
        fam.update({"family_id": fid, "schema": S.FAMILY, "frame": "object", "representative": c["name"], "members": [],
                    "physics": "representative_only"})
        fams[fid] = fam
        c = dict(c)
        c["family_id"] = fid
        reps.append(c)
    return reps, fams
