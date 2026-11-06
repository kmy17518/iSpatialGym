"""Tabletop and attachment grasp candidates v2: top-down AND side grasps, screened under the shared contact policy, symmetry families
folded to one candidate each (port of ``runlogs/grasp_library_20260905/scripts/gen_candidates.py``, pipeline ``a1``).

Per model (its exported collision pieces, :mod:`object_geometry`) and designated rest (canonical or the legacy key's rest hint):

1. Isaac's antipodal sampler on the collision union (``n_candidates`` x ``n_orientations`` about the closing axis: approach directions
   all around the pinch line, so side approaches are sampled with the top-down ones);
2. per mode - ``top`` (approach within ``top_max_tilt_deg`` of straight down) and ``side`` (approach between ``side_tilt_deg`` from
   vertical: about parallel to the tabletop) - width / antipodal-cone / coverage features (rays from the pad centre along the closing
   axis, as before);
3. the support: the rest's tabletop plane under the object. The hand's exact collision hulls (fingers pre-shaped to
   ``width / 2 + preshape margin`` like the oracle) are measured against it (:class:`contact_policy.HandChecker`): a top grasp whose
   fingers sink deeper than the policy's 2 mm retreats along its approach (as before: at most ``tip_max_m`` of pad depth), a side
   grasp is lifted along the support normal (the oracle's ``lift_grasp_off_support`` move) as long as the pads stay on the object;
4. object clearance of the hand's cuRobo spheres (``obj_margin_m``) with the object-aware retreat for top grasps (knobs, rims), and
   the approach from the pre-grasp (``conventions.PRE_DIST`` back along the approach) clear of the object;
5. the closing sweep (the pads meet the intended contact first), the lever-arm cap, the constraint (``attachment_free``: hand spheres
   2 cm from the mating points and the contact off the mating side; ``flower_stem_top``);
6. ranking (the old tabletop heuristic; the vertical-approach term only for top grasps) and NMS to ``topk`` per mode;
7. the full contact-policy geometry (pre-grasp, approach waypoints, closing sweep: :func:`contact_policy.check_grasp_geometry`) - a
   failing candidate is kept in the file with its reason but never becomes a representative;
8. symmetry families (:mod:`symmetry`) and deduplication: the representatives are what physics validation tests.

Output ``candidates.json`` (schema ``CANDIDATES``) in the model's library directory.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import hand_geometry as H
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import symmetry as SY

PIPELINE = "a1v2"
CONTACT_Z = -0.01186  # conventions.CONTACT_CENTER_Z (pad centre behind the eef origin along the approach)
GRIPPER_MAX_OPENING = 0.0969
PRESHAPE_MARGIN = 0.010  # motions.MotionsConfig.gripper_preshape_margin
FINGER_OPEN = 0.05


@dataclass
class CandidateConfig:
    n_candidates: int = 6000
    n_orientations: int = 12
    seed: int = 0
    aperture_m: float = GRIPPER_MAX_OPENING - 0.008
    min_width_m: float = 0.003
    cone_deg: float = 30.0
    top_max_tilt_deg: float = 40.0
    side_tilt_deg: Tuple[float, float] = (70.0, 100.0)
    obj_margin_m: float = 0.001
    tip_max_m: float = 0.017  # the contact at most this far beyond the eef origin along the approach after a retreat (the fingertips: 0.0188)
    retreats_m: Tuple[float, ...] = (0.005, 0.01, 0.015, 0.02, 0.025)
    side_lift_max_m: float = 0.03
    sweep_margin_m: float = 0.008
    lever_cap: float = 0.40
    topk: int = 12
    nms_pos_m: float = 0.015
    nms_rot_deg: float = 30.0
    approach_obj_check: bool = True
    modes: Tuple[str, ...] = ("top", "side")
    max_representatives: int = 10
    weights: Dict[str, float] = field(default_factory=lambda: {"com": 5.0, "antipodal": 1.0, "coverage": 1.0, "width": 0.5, "clearance": 1.0, "vertical": 3.0})


FLOWER_WINDOW = {"lily": (0.55, 0.85, 0.70)}
FLOWER_WINDOW_DEFAULT = (0.2, 0.7, 0.35)
STEM_MAX_WIDTH = 0.030


# ------------------------------------------------------------------------------------------------ the Isaac antipodal sampler


def _isaac_sampler():
    """``isaacsim.replicator.grasping.sampler_utils`` without Kit (its pxr import is stubbed; only trimesh paths are used)."""
    spec = importlib.util.find_spec("isaacsim")
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    for r in roots:
        p = os.path.join(r, "exts", "isaacsim.replicator.grasping", "isaacsim", "replicator", "grasping", "sampler_utils.py")
        if os.path.exists(p):
            break
    else:
        raise ImportError("isaacsim.replicator.grasping sampler_utils not found")
    if "pxr" not in sys.modules:
        class _Any:
            def __getattr__(self, k):
                return object
        pxr = types.ModuleType("pxr")
        pxr.Gf = _Any()
        pxr.UsdGeom = _Any()
        sys.modules["pxr"] = pxr
    s = importlib.util.spec_from_file_location("isg_sampler_utils", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class _RayProxy:
    """embree returns only the self-hit for rays starting on a face: offset origins 0.1 mm along the ray."""

    def __init__(self, ray):
        self.ray = ray

    def intersects_location(self, origins, dirs, multiple_hits=True):
        o = np.asarray(origins, float) + 1e-4 * np.asarray(dirs, float)
        return self.ray.intersects_location(o, dirs, multiple_hits=multiple_hits)


class _MeshProxy:
    def __init__(self, m):
        self.m = m
        self.ray = _RayProxy(m.ray)
        self.face_normals = m.face_normals

    def sample(self, *a, **k):
        return self.m.sample(*a, **k)


# ------------------------------------------------------------------------------------------------ geometry features


def hits(union, origins, dirs):
    n = len(origins)
    d = np.full(n, np.inf)
    cosang = np.zeros(n)
    if n == 0:
        return d, cosang
    loc, idx, tri = union.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(idx):
        dist = np.linalg.norm(loc - origins[idx], axis=1)
        order = np.argsort(dist)
        idx, tri, dist = idx[order], tri[order], dist[order]
        first = np.unique(idx, return_index=True)[1]
        idx, tri, dist = idx[first], tri[first], dist[first]
        d[idx] = dist
        cosang[idx] = np.abs(np.einsum("ij,ij->i", union.face_normals[tri], dirs[idx]))
    return d, cosang


def contact_features(union, T, cz):
    cz = np.asarray(cz, dtype=float).reshape(-1, 1) if np.ndim(cz) else cz
    c = T[:, :3, 3] + T[:, :3, 2] * cz
    y, x = T[:, :3, 1], T[:, :3, 0]
    dp, cp = hits(union, c, y)
    dm, cm = hits(union, c, -y)
    width = dp + dm
    antipodal = np.minimum(cp, cm)
    cov = np.zeros(len(T))
    for off in (-0.01, -0.005, 0.005, 0.01):
        o = c + x * off
        d1, _ = hits(union, o, y)
        d2, _ = hits(union, o, -y)
        with np.errstate(invalid="ignore"):
            cov += (np.isfinite(d1 + d2) & (np.abs(d1 + d2 - width) < 0.01)).astype(float)
    return c, width, antipodal, cov / 4.0


PAD_Z = np.array([-0.035, -0.02, -0.005, 0.01])
PAD_X = np.array([-0.01, 0.0, 0.01])
PAD_OPEN_HALF = 0.0485


def pad_sweep(union, T):
    """Largest half-width at which a closing pad meets the object (rays from the open pad faces inward over the pad footprint)."""
    n = len(T)
    best = np.zeros(n)
    y, x, z = T[:, :3, 1], T[:, :3, 0], T[:, :3, 2]
    base = T[:, :3, 3]
    for dz in PAD_Z:
        for dx in PAD_X:
            o = base + z * dz + x * dx
            for sgn in (1.0, -1.0):
                d, _ = hits(union, o + y * (sgn * PAD_OPEN_HALF), -sgn * y)
                best = np.maximum(best, np.where(np.isfinite(d), PAD_OPEN_HALF - d, 0.0))
    return best


class SphereClearance:
    """Signed clearance of the hand's cuRobo spheres from the object's convex pieces (negative: inside a piece)."""

    def __init__(self, geom: OG.ObjectGeometry, arm: str = "right"):
        self.geom = geom
        self.planes = []
        for pc in geom.pieces:
            n = pc.face_normals
            self.planes.append((n, np.einsum("ij,ij->i", n, pc.triangles[:, 0])))
        self.arm = arm

    def spheres(self, opening: float):
        Cs, Rs, L = H.hand_spheres_eef(self.arm, (opening, opening))
        return Cs, Rs, L

    def __call__(self, T, opening: float = FINGER_OPEN, exclude_fingers: bool = False) -> np.ndarray:
        import trimesh

        T = np.asarray(T)
        Cs, Rs, L = self.spheres(opening)
        if exclude_fingers:
            keep = np.array(["finger" not in l for l in L])
            Cs, Rs = Cs[keep], Rs[keep]
        P = np.einsum("nij,kj->nki", T[:, :3, :3], Cs) + T[:, None, :3, 3]
        flat = P.reshape(-1, 3)
        inside = np.zeros(len(flat), dtype=bool)
        for n, off in self.planes:
            inside |= ((flat @ n.T - off[None]) <= 0).all(axis=1)
        _, dist, _ = trimesh.proximity.closest_point(self.geom.union, flat)
        dist = np.where(inside, -dist, dist)
        return (dist.reshape(len(T), -1) - Rs[None]).min(axis=1)


def tilt_deg(T):
    """Angle of the approach (eef +z) from straight down."""
    return np.degrees(np.arccos(np.clip(-T[:, :3, 2][:, 2], -1, 1)))


def nms(T, scores, k, pos_m, rot_deg):
    order = np.argsort(-scores)
    keep = []
    for i in order:
        ok = True
        for j in keep:
            R = T[i, :3, :3].T @ T[j, :3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if np.linalg.norm(T[i, :3, 3] - T[j, :3, 3]) < pos_m and ang < rot_deg:
                ok = False
                break
        if ok:
            keep.append(i)
        if len(keep) >= k:
            break
    return np.array(keep, dtype=int)


def attachment_points(info):
    pts = []
    for link, ml in (info.get("meta_links") or {}).items():
        if not isinstance(ml, dict):
            continue
        for k, v in ml.get("attachment", {}).items():
            for e in (v.values() if isinstance(v, dict) else v):
                if isinstance(e, dict) and "position" in e:
                    pts.append(np.array(e["position"], dtype=float))
    if not pts:  # no mating meta link: the bottom centre
        lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"])
        c = (lo + hi) / 2
        c[2] = lo[2]
        pts.append(c)
    return np.array(pts)


def attachment_frames(info, R: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``[(point, mating direction)]`` of the attachment meta links in the rest frame (``R``): the direction is the meta link frame's -z
    (identity orientation: straight down, onto the partner). No meta link: the bottom centre, mating downward."""
    out = []
    for link, ml in (info.get("meta_links") or {}).items():
        if not isinstance(ml, dict):
            continue
        for k, v in ml.get("attachment", {}).items():
            for e in (v.values() if isinstance(v, dict) else v):
                if isinstance(e, dict) and "position" in e:
                    q = e.get("orientation") or [0.0, 0.0, 0.0, 1.0]
                    Rm = S.quat_to_mat(q)
                    out.append((R[:3, :3] @ np.asarray(e["position"], dtype=float), R[:3, :3] @ (Rm @ np.array([0.0, 0.0, -1.0]))))
    if not out:
        lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"])
        c = (lo + hi) / 2
        c[2] = lo[2]
        out.append((c, np.array([0.0, 0.0, -1.0])))
    return out


def stem_axis(union, lo, hi):
    ext = hi - lo
    ax = int(np.argmax(ext))
    v = union.vertices
    t = (v[:, ax] - lo[ax]) / ext[ax]
    perp = np.delete(v, ax, axis=1)

    def spread(mask):
        return float(np.ptp(perp[mask], axis=0).max()) if mask.sum() > 3 else 0.0

    lo_end, hi_end = spread(t < 0.3), spread(t > 0.7)
    d = np.zeros(3)
    d[ax] = 1.0
    head = hi.copy() if hi_end >= lo_end else lo.copy()
    axis_dir = -d if hi_end >= lo_end else d
    head_pt = (lo + hi) / 2
    head_pt[ax] = head[ax]
    return head_pt, axis_dir, float(ext[ax])


# ------------------------------------------------------------------------------------------------ generation


def rest_rotation(hint_wxyz: Optional[Sequence[float]]) -> np.ndarray:
    R = np.eye(4)
    if hint_wxyz is not None:
        w, x, y, z = hint_wxyz
        R[:3, :3] = S.quat_to_mat([x, y, z, w])
    return R


def _score(T, com, width, antip, cov, clear, size, cz, mode, W):
    c = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
    v = com[None] - c
    along = np.einsum("ij,ij->i", v, T[:, :3, 1])
    perp = np.abs(np.einsum("ij,ij->i", v, T[:, :3, 0]))
    scale = max(0.01, 0.25 * size)
    f_com = np.exp(-perp / scale) * (0.6 + 0.4 * np.exp(-np.abs(along) / scale))
    f_width = np.clip(1.0 - np.maximum(0.0, width - 0.07) / 0.03, 0, 1) * np.clip((width - 0.008) / 0.007, 0, 1)
    f_clear = np.clip(clear / 0.02, 0, 1)
    f_vert = np.clip(-T[:, :3, 2][:, 2], 0, 1) if mode == "top" else np.zeros(len(T))
    return (W["com"] * f_com + W["antipodal"] * antip + W["coverage"] * cov + W["width"] * f_width + W["clearance"] * f_clear
            + W["vertical"] * f_vert), perp


def generate(job: Dict[str, Any], model_dir: str, cfg: CandidateConfig = CandidateConfig(), policy: CP.ContactPolicy = CP.DEFAULT,
             sym_cfg: SY.SymmetryConfig = SY.SymmetryConfig(), arm: str = "right", densify: Sequence[Tuple[int, int]] = ((4, 2), (8, 4))) -> Dict[str, Any]:
    """Candidates of one model (``job``: category, model, scale, constraint, rest_hint_wxyz) -> ``candidates.json``. A model whose
    representatives come out empty is sampled again denser (``densify``: candidate x orientation multipliers; small features such as
    a lid's knob draw few of the uniform surface samples - the legacy lid__pbmsji__s065 run used 48000 x 48)."""
    out = _generate(job, model_dir, cfg, policy, sym_cfg, arm)
    for mc, mo in densify:
        if out["representatives"]:
            break
        dense = CandidateConfig(**{**asdict(cfg), "n_candidates": cfg.n_candidates * mc, "n_orientations": cfg.n_orientations * mo})
        prev = out["log"]
        out = _generate(job, model_dir, dense, policy, sym_cfg, arm)
        out["log"]["densified_from"] = prev
    return out


def _generate(job: Dict[str, Any], model_dir: str, cfg: CandidateConfig, policy: CP.ContactPolicy, sym_cfg: SY.SymmetryConfig, arm: str) -> Dict[str, Any]:
    import torch as th

    t0 = time.time()
    geom0 = OG.load(model_dir)
    if geom0 is None:
        raise FileNotFoundError(f"no geometry.json in {model_dir} (run the export stage)")
    constraint = job.get("constraint") or "tabletop"
    R = rest_rotation(job.get("rest_hint_wxyz"))
    geom = geom0.rotated(R)  # the rest frame: the tabletop is z = z_min
    union = geom.union
    lo, hi = union.vertices.min(0), union.vertices.max(0)
    z_min = float(lo[2])
    size = float(np.max(hi - lo))
    com = union.center_mass if union.volume > 1e-9 else union.centroid
    info = dict(geom0.info)
    info["coll_bbox_min"], info["coll_bbox_max"] = lo.tolist(), hi.tolist()
    if constraint == "attachment_free":
        for link, ml in (info.get("meta_links") or {}).items():  # mating points into the rest frame
            if isinstance(ml, dict):
                for k, v in ml.get("attachment", {}).items():
                    for e in (v.values() if isinstance(v, dict) else v):
                        if isinstance(e, dict) and "position" in e:
                            e["position"] = (R[:3, :3] @ np.array(e["position"])).tolist()
    su = _isaac_sampler()
    T_all = np.array(su.sample_antipodal(_MeshProxy(union), num_candidates=cfg.n_candidates, num_orientations=cfg.n_orientations,
                                         gripper_maximum_aperture=cfg.aperture_m, gripper_standoff_fingertips=CONTACT_Z,
                                         gripper_approach_direction=np.array([0, 0, 1.0]), grasp_align_axis=np.array([0, 1.0, 0]),
                                         orientation_sample_axis=np.array([0, 1.0, 0]), random_seed=cfg.seed))
    log: Dict[str, Any] = {"n_sampled": int(len(T_all)), "z_min": round(z_min, 5), "size": round(size, 4)}
    support = CP.Support.plane(z_min, name="tabletop")
    hc = CP.HandChecker(arm, policy, spacing=0.004)
    clear_fn = SphereClearance(geom, arm)
    out_cands: List[Dict[str, Any]] = []
    for mode in cfg.modes:
        L: Dict[str, Any] = {}
        if len(T_all) == 0:
            log[mode] = L
            continue
        tl = tilt_deg(T_all)
        m = tl <= cfg.top_max_tilt_deg if mode == "top" else (tl >= cfg.side_tilt_deg[0]) & (tl <= cfg.side_tilt_deg[1])
        T = T_all[m].copy()
        L["n_tilt"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        _, width, antip, cov = contact_features(union, T, CONTACT_Z)
        keep = np.isfinite(width) & (width >= cfg.min_width_m) & (width <= cfg.aperture_m) & (antip >= math.cos(math.radians(cfg.cone_deg)))
        T, width = T[keep], width[keep]
        L["n_width_antipodal"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        # 3) the support under the hand: policy measurement on the exact hulls at the pre-shaped opening
        shift = np.zeros(len(T))
        pre_q = np.clip(0.5 * width + PRESHAPE_MARGIN, 0.0, FINGER_OPEN)
        deficit = np.zeros(len(T))
        margin = 0.001  # aim 1 mm inside every allowance: the measured rest sits a fraction of a mm off the canonical one
        for q in np.unique(np.round(pre_q, 3)):
            idx = np.where(np.round(pre_q, 3) == q)[0]
            cl = hc.clearance(th.as_tensor(T[idx], dtype=th.float32), support, (float(q), float(q)))
            need_f = -policy.finger_support_max_penetration_m + margin - cl["finger"].numpy()
            hand = np.minimum(np.minimum(cl["palm"].numpy(), cl["wrist_camera"].numpy()), cl["wrist"].numpy())
            need_h = policy.hand_support_margin_m + margin - hand
            # only a part below its allowance asks for a move; the move then clears every part by the margin
            need = np.where((cl["finger"].numpy() < -policy.finger_support_max_penetration_m) | (hand < policy.hand_support_margin_m),
                            np.maximum(need_f, need_h), 0.0)
            deficit[idx] = np.maximum(0.0, need)
        a = T[:, :3, 2]
        if mode == "top":  # retreat along the approach (as before; the support's normal component of the retreat lifts the hand)
            s = deficit / np.maximum(-a[:, 2], 1e-3)
            ok = (CONTACT_Z + s) <= cfg.tip_max_m + 1e-9  # the contact no further out than the tip surface (gen_candidates TIP_MAX)
            T, s, width, pre_q = T[ok], s[ok], width[ok], pre_q[ok]
            T[:, :3, 3] -= T[:, :3, 2] * s[:, None]
            shift = s
        else:  # lift along the support normal (lift_grasp_off_support's move); the pads must still close on the object (sweep below)
            ok = deficit <= cfg.side_lift_max_m
            T, width, pre_q, deficit = T[ok], width[ok], pre_q[ok], deficit[ok]
            T[:, 2, 3] += deficit
            shift = deficit
        L["n_support"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        # 4) object clearance (spheres) with the object-aware retreat of top grasps
        clear = _clear_grouped(clear_fn, T, pre_q)
        ok = clear >= cfg.obj_margin_m
        if mode == "top":
            for ds in cfg.retreats_m:
                bad = np.where(~ok & ((CONTACT_Z + shift + ds) <= cfg.tip_max_m))[0]
                if len(bad) == 0:
                    break
                Tb = T[bad].copy()
                Tb[:, :3, 3] -= Tb[:, :3, 2] * ds
                cb = _clear_grouped(clear_fn, Tb, pre_q[bad])
                good = cb >= cfg.obj_margin_m
                idx = bad[good]
                T[idx] = Tb[good]
                clear[idx] = cb[good]
                shift[idx] += ds
                ok[idx] = True
                shift[bad[~good]] += ds
        T, clear, shift, width, pre_q = T[ok], clear[ok], shift[ok], width[ok], pre_q[ok]
        L["n_object_clear"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        cz = CONTACT_Z + (shift if mode == "top" else 0.0)  # the pads' contact height after the retreat
        cz = np.broadcast_to(np.asarray(cz, dtype=float), (len(T),)).copy()
        c, width, antip, cov = contact_features(union, T, cz)
        touch = pad_sweep(union, T)
        keep = np.isfinite(width) & (width >= cfg.min_width_m) & (touch <= width / 2 + cfg.sweep_margin_m)
        T, clear, shift, cz, width, antip, cov, pre_q = T[keep], clear[keep], shift[keep], cz[keep], width[keep], antip[keep], cov[keep], pre_q[keep]
        L["n_sweep"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        score, perp = _score(T, com, width, antip, cov, clear, size, cz, mode, cfg.weights)
        feats = {"com_perp": perp, "width": width, "antipodal": antip, "coverage": cov, "clearance": clear, "tilt_deg": tilt_deg(T), "shift_m": shift}
        if constraint != "flower_stem_top":
            keep = perp <= cfg.lever_cap * size
            T, score, cz, pre_q = T[keep], score[keep], cz[keep], pre_q[keep]
            feats = {k: v[keep] for k, v in feats.items()}
            L["n_lever"] = int(len(T))
        if constraint == "attachment_free" and len(T):
            # the mating side from the attachment meta link: its point p (an anchor, often inside the body) and its frame's -z (the
            # direction the object moves onto its partner). A lid / cap mates with its underside: the contact must lie on the far
            # side of p from the mating direction, and no finger may reach past the object's mating face (the partner's volume).
            # The generic "hand spheres >= 1 cm from p, contact off the side of p" rule rejected every knob pinch of the lids (their
            # anchor sits above the CoM, inside the knob's reach) - the legacy lid__pbmsji__s065 run replaced it with a rim rule.
            mates = attachment_frames(info, R)
            cc = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
            keep = np.ones(len(T), dtype=bool)
            side_m = np.full(len(T), np.inf)
            face_gap = np.full(len(T), np.inf)
            Vm_max = {}
            for k, (pm, m) in enumerate(mates):
                side = (cc - pm[None]) @ (-m)  # > 0: the contact on the side away from the mating direction
                Vm_max[k] = float((union.vertices @ m).max())
                keep &= side >= 0.0
                side_m = np.minimum(side_m, side)
                for i in range(len(T)):
                    P, _ = hc.geom.points(("finger",), (float(pre_q[i]), float(pre_q[i])))
                    Pw = P @ T[i, :3, :3].T + T[i, :3, 3]
                    face_gap[i] = min(face_gap[i], Vm_max[k] - float((Pw @ m).max()))
            keep &= face_gap >= 0.002
            T, score, cz, pre_q, side_m, face_gap = T[keep], score[keep], cz[keep], pre_q[keep], side_m[keep], face_gap[keep]
            feats = {k: v[keep] for k, v in feats.items()}
            feats["attach_side_m"], feats["attach_face_gap_m"] = side_m, face_gap
            score = score + 2.0 * np.clip(side_m / 0.03, 0, 1)
            L["n_attachment_free"] = int(len(T))
        elif constraint == "flower_stem_top" and len(T):
            head_pt, axis_dir, Ls = stem_axis(union, lo, hi)
            cc = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
            t = np.einsum("ij,j->i", cc - head_pt[None], axis_dir) / Ls
            across = np.abs(T[:, :3, 1] @ axis_dir)
            w_lo, w_hi, w_c = FLOWER_WINDOW.get(job["category"], FLOWER_WINDOW_DEFAULT)
            keep = (t >= w_lo) & (t <= w_hi) & (across <= 0.5) & (feats["width"] <= STEM_MAX_WIDTH)
            T, score, cz, pre_q, t, across = T[keep], score[keep], cz[keep], pre_q[keep], t[keep], across[keep]
            feats = {k: v[keep] for k, v in feats.items()}
            score = score + 5.0 * np.exp(-((t - w_c) / 0.15) ** 2) - 1.0 * across
            feats["stem_t"], feats["pinch_across"] = t, across
            L["n_flower"] = int(len(T))
        if len(T) == 0:
            log[mode] = L
            continue
        sel = nms(T, score, cfg.topk, cfg.nms_pos_m, cfg.nms_rot_deg)
        L["n_selected"] = int(len(sel))
        # 7) the full policy geometry: pre-grasp, approach, closing sweep (fingers from the pre-shape to the contact opening)
        Ts = T[sel]
        contact_q = np.clip(0.5 * feats["width"][sel] + pad_face_offset(arm), 0.0, FINGER_OPEN)  # joint value at the pads' first touch
        geo = CP.check_grasp_geometry(hc, th.as_tensor(Ts, dtype=th.float32), support, contact_q, open_q=float(np.max(pre_q[sel])))
        appr = _approach_object_clear(clear_fn, Ts, pre_q[sel], cfg) if cfg.approach_obj_check else np.ones(len(sel), dtype=bool)
        blades, deep = finger_blades_clear(geom, hc, Ts, pre_q[sel], contact_q)
        for r, i in enumerate(sel):
            T_obj = np.linalg.inv(R) @ Ts[r]  # back to the canonical object frame
            ok_geo = bool(geo["ok"][r]) and bool(appr[r]) and bool(blades[r])
            reason = (geo["reason"][r] if not bool(geo["ok"][r]) else ("approach_hits_object" if not appr[r] else
                      ("finger_blade_in_object" if not blades[r] else "ok")))
            out_cands.append({
                "name": f"{mode}_{len([c for c in out_cands if c['mode'] == mode])}", "mode": mode, "score": round(float(score[i]), 4),
                "T_object_eef": S.mat_pose(T_obj), "T_rest_eef": S.mat_pose(Ts[r]), "width_m": round(float(feats["width"][i]), 5),
                "preshape_q": round(float(pre_q[i]), 5), "contact_q": round(float(contact_q[r]), 5),
                "features": {k: round(float(v[i]), 5) for k, v in feats.items()},
                "geometry": {"ok": ok_geo, "reason": reason, "stage": geo["stage"][r], "finger_min_m": round(float(geo["finger_min"][r]), 5),
                             "hand_min_m": round(float(geo["hand_min"][r]), 5), "blade_object_depth_m": round(float(deep[r]), 5)},
            })
        log[mode] = L
    # 8) symmetry families of the geometric survivors, deduplicated (one candidate per family)
    t_sym = time.time()
    good = sorted([c for c in out_cands if c["geometry"]["ok"]], key=lambda c: -c["score"])
    fams: Dict[str, Dict[str, Any]] = {}
    if good:
        idx = SY.SurfaceIndex(union, n=sym_cfg.n_surface)

        def width_fn(Tm):
            _, w, _, _ = contact_features(union, Tm, CONTACT_Z)
            return w

        for c in good:
            Tr = S.pose_of(c["T_rest_eef"])

            def clear_obj(Tm, q=c["preshape_q"]):  # the member's hand at the representative's pre-shape, as generation judged it
                return clear_fn(Tm, q) - cfg.obj_margin_m + 1e-9

            def support_ok(Tm, c=c):  # the member under the contact policy against the rest's tabletop (pre-grasp, approach, sweep)
                return CP.check_grasp_geometry(hc, th.as_tensor(Tm, dtype=th.float32), support, [c["contact_q"]] * len(Tm), open_q=c["preshape_q"])["ok"].numpy()

            fam = SY.detect(Tr, union, idx, width_fn, clear_obj, rest_R=np.eye(4), cfg=sym_cfg, contact_z=CONTACT_Z, support_fn=support_ok)
            fams[c["name"]] = _family_to_object(fam, R)
    reps, families = SY.dedup([{"name": c["name"], "T": S.pose_of(c["T_object_eef"])} for c in good], fams, sym_cfg)
    rep_names = [r["name"] for r in reps][: cfg.max_representatives]
    by_name = {c["name"]: c for c in out_cands}
    for fid, f in families.items():
        for mem in f.get("members", []):
            by_name[mem["name"]]["family_id"] = fid
            by_name[mem["name"]]["family_relation"] = mem["relation"]
        by_name[f["representative"]]["family_id"] = fid
        by_name[f["representative"]]["family_relation"] = "representative"
    log["symmetry_s"] = round(time.time() - t_sym, 2)
    out = {"schema": S.CANDIDATES, "pipeline": PIPELINE, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
           "dataset": job.get("dataset"), "constraint": constraint, "rest_hint_wxyz": job.get("rest_hint_wxyz"), "legacy_key": job.get("legacy_key"),
           "arm": arm, "policy": policy.meta(arm), "policy_key": policy.key(arm), "config": asdict(cfg), "symmetry_config": asdict(sym_cfg),
           "candidates": out_cands, "families": families, "representatives": rep_names, "log": log, "t_s": round(time.time() - t0, 2),
           "created": S.now(), "review_instructions": "docs/GRASP_REVIEW.md (symmetry annotations required)"}
    S.save_json(os.path.join(model_dir, "candidates.json"), out)
    return out


def _clear_grouped(clear_fn, T, pre_q):
    out = np.zeros(len(T))
    for q in np.unique(np.round(pre_q, 3)):
        idx = np.where(np.round(pre_q, 3) == q)[0]
        out[idx] = clear_fn(T[idx], float(q))
    return out


def _approach_object_clear(clear_fn, T, pre_q, cfg) -> np.ndarray:
    """The straight approach from the pre-grasp (``conventions.PRE_DIST``) at the pre-shaped opening keeps every hand sphere clear of
    the object down to the last waypoint before the pads reach the contact (the closing sweep covers the rest)."""
    from ispatialgym.robot import conventions as C

    ok = np.ones(len(T), dtype=bool)
    for t in np.linspace(1.0, 0.25, 6):  # fraction of PRE_DIST still to go
        Tw = T.copy()
        Tw[:, :3, 3] -= Tw[:, :3, 2] * (C.PRE_DIST * t)
        ok &= _clear_grouped(clear_fn, Tw, pre_q) >= cfg.obj_margin_m
    return ok


def pad_face_offset(arm: str = "right") -> float:
    """Joint value minus the pad's inner face position along the closing axis (the closed pads overlap the centre line by ~0.9 mm):
    a pinch of width w is touched at ``q = w / 2 + pad_face_offset``."""
    g = H.load(arm)
    f1 = [l for l in g.by_role("finger") if l.link.endswith("1")][0]
    P = f1.at(0.0)
    m = (P[:, 2] > -0.0381) & (P[:, 2] < 0.0119)  # conventions.PAD_Z_RANGE
    return float(-P[m, 1].min())


def finger_blades_clear(geom: OG.ObjectGeometry, hc: CP.HandChecker, T, pre_q, contact_q, steps: int = 6, face_band_m: float = 0.0015,
                        tol_m: float = 0.0005, contact_z: float = CONTACT_Z) -> Tuple[np.ndarray, np.ndarray]:
    """The finger blades' exact hulls stay outside the object's collision pieces (within ``tol_m``) from the pre-shape down to the pads'
    first touch (``contact_q`` + 0.5 mm) - the lid__pbmsji__s065 rule (sphere-cleared knob pinches put the blades 0.3-1.4 mm into the
    lid's dome; that run cleared "every piece but the pinched one by >= 1 mm"). Left out, as what the closure meets by design: the pads'
    inner faces (points within ``face_band_m`` of each pad's inner plane inside ``PAD_Z_RANGE``; the closing sweep judges them) and the
    pinched pieces (within 2 mm of the two contact points - a flat pad meets a curved rim at its corners first). Signed depth of a point
    in a convex piece = max over its face planes (negative inside). ``(ok (N,), deepest (N,))``."""
    import torch as th
    import trimesh

    planes = []
    for pc in geom.pieces:
        n = pc.face_normals
        planes.append((th.as_tensor(n, dtype=th.float32), th.as_tensor(np.einsum("ij,ij->i", n, pc.triangles[:, 0]), dtype=th.float32)))
    off = pad_face_offset(hc.arm)
    Tn = np.asarray(T)
    c = Tn[:, :3, 3] + Tn[:, :3, 2] * contact_z
    y = Tn[:, :3, 1]
    dp, _ = hits(geom.union, c, y)
    dm, _ = hits(geom.union, c, -y)
    T = th.as_tensor(Tn, dtype=th.float32)
    deep = np.full(len(T), np.inf)
    for i in range(len(T)):
        pts = np.array([p for p in (c[i] + y[i] * dp[i], c[i] - y[i] * dm[i]) if np.isfinite(p).all()])
        pinched = set()
        if len(pts):
            for k, pc in enumerate(geom.pieces):
                if float(np.abs(trimesh.proximity.closest_point(pc, pts)[1]).min()) <= 0.002:
                    pinched.add(k)
        for q in np.linspace(float(pre_q[i]), float(contact_q[i]) + 0.0005, steps):
            P, _ = hc.geom.points(("finger",), (float(q), float(q)))
            face = q - off  # the inner pad planes at y = +-face
            pad = (P[:, 2] > -0.0381) & (P[:, 2] < 0.0119)
            inner = pad & ((np.abs(P[:, 1] - face) <= face_band_m) | (np.abs(P[:, 1] + face) <= face_band_m))
            Pw = th.as_tensor(P[~inner], dtype=th.float32) @ T[i, :3, :3].T + T[i, :3, 3]
            for k, (n, d) in enumerate(planes):
                if k in pinched:
                    continue
                sd = (Pw @ n.T - d[None]).max(dim=1).values
                deep[i] = min(deep[i], float(sd.min()))
    return deep >= -tol_m, deep


def _family_to_object(fam: Dict[str, Any], R: np.ndarray) -> Dict[str, Any]:
    """A family detected in the rest frame, in the canonical object frame (axes / centres rotated by R^-1)."""
    if fam.get("type") in (None, "none"):
        return fam
    Ri = np.linalg.inv(R)[:3, :3]
    out = dict(fam)
    out["axis"] = (Ri @ np.asarray(fam["axis"])).round(6).tolist()
    for k in ("center", "origin"):
        if k in fam:
            out[k] = (Ri @ np.asarray(fam[k])).round(6).tolist()
    return out
