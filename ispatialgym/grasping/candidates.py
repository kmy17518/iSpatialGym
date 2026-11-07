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
    # oracle-fix 2026-11-04 (owner item 5): ISG_GLIB_CONE_DEG overrides (the pillow's tapered edge needs 45 deg; unset = 30)
    cone_deg: float = field(default_factory=lambda: float(os.environ.get("ISG_GLIB_CONE_DEG") or 30.0))
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
    # i7-plate 2026-11-07 (owner: recover plate luhkiz): opt-in structured candidate FAMILIES (:data:`FAMILIES`), each its own pool after
    # the modes, through the same gates as a top grasp (retreat along the approach). Default () = off; enable per object with the
    # manifest's ``cand_overrides`` ({"families": ["lip_shallow", "rim_straddle"]}) or per run with ISG_GLIB_CAND_OVERRIDES.
    families: Tuple[str, ...] = ()
    lip_band_m: float = 0.012  # lip_shallow: the pad contact within this of the largest radius about the plate axis
    lip_tilt_deg: Tuple[float, float] = (0.0, 70.0)  # lip_shallow: the tilt window (the top mode stops at 40, the side mode starts at 70)
    lip_pad_offsets_m: Tuple[float, ...] = (0.0, 0.008, 0.016, 0.024)  # lip_shallow: eef raised back along the approach (contact -> tips)
    straddle_n_azimuth: int = 72
    straddle_n_heights: int = 10
    straddle_tilts_deg: Tuple[float, ...] = (0.0, 10.0, -10.0, 20.0, -20.0)  # rim_straddle: approach tilt in the radial plane (+ leans outward)
    straddle_pad_offsets_m: Tuple[float, ...] = (0.0, 0.007, 0.014, 0.021, 0.028)  # rim_straddle: the contact this far toward the tips
    family_max_pool: int = 40000  # a family pool larger than this is subsampled (seeded)
    # ISG_GLIB_GATE_DUMP=1: per pool and gate, the counts and the measures of the dropped candidates -> gate_dump.json (diagnostics only)
    gate_dump: bool = field(default_factory=lambda: os.environ.get("ISG_GLIB_GATE_DUMP", "") == "1")
    # OWNER DECISION, ADOPTED 2026-11-07 (i7-switches: default True for new builds after the count on the campaign library; False = the
    # behaviour before, which every entry generated without the field reads as - schema.candidate_switches): finger_blades_clear's pinched
    # pieces looked up at the retreated contact (CONTACT_Z + the real retreat) instead of CONTACT_Z. Without it a retreated top grasp's real
    # pinched piece is not left out and the pad's own lip contact counts as a blade in the object (luhkiz L3: 4 of its 5 lip pinches pass at
    # the real contact). The state is part of the entry identity (schema.ModelLibrary.identity when on).
    blade_contact_at_retreat: bool = True
    # OWNER DECISION, ADOPTED 2026-11-07 (i7-switches, as above): the top mode's object-clearance retreats keep the shift equal to the eef's
    # real displacement. Without it a failed try (absolute retreats 5, 10, ... mm from the unmoved pose) still adds to the shift, so a
    # grasp cleared at 10 mm records 15 mm and the closing sweep re-measures the width 5 mm off its contact (luhkiz: 57 of 62 top grasps
    # lose their width there; 59 of 62 pass the sweep at the real displacement).
    retreat_shift_exact: bool = True


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


# ------------------------------------------------------------------------------------------------ opt-in candidate families (i7-plate)
#
# Owner decision 2026-11-07 (plate luhkiz: 7 sloped rim sectors + a foot disk as convex hulls, no grasp passed): structured pools that go
# through EVERY gate of a top grasp unchanged (width / antipodal cone, support retreat, sphere clearance, closing sweep, lever, NMS, policy
# geometry, finger blades). A family only proposes poses; it never relaxes a gate. Default off (``CandidateConfig.families`` = ()).


def plate_axis(union) -> Tuple[float, float, float]:
    """The rest-frame vertical axis of a plate-like object: ``(cx, cy, r_max)`` - the xy centre of its collision bbox and the largest
    vertex radius about it."""
    v = np.asarray(union.vertices)
    lo, hi = v.min(0), v.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    return float(cx), float(cy), float(np.hypot(v[:, 0] - cx, v[:, 1] - cy).max())


def _inside_any(P: np.ndarray, planes) -> np.ndarray:
    """Points inside (or on) any convex piece given by its face planes ``[(normals (F, 3), offsets (F,))]``."""
    inside = np.zeros(len(P), dtype=bool)
    for n, off in planes:
        inside |= ((P @ n.T - off[None]) <= 1e-9).all(axis=1)
    return inside


def _frames(x: np.ndarray, y: np.ndarray, z: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = x, y, z, p
    return T


def lip_family_poses(T_all: np.ndarray, geom: OG.ObjectGeometry, cfg: CandidateConfig) -> Tuple[np.ndarray, np.ndarray]:
    """``lip_shallow`` (owner option 2): the sampled antipodal pinches whose pad contact lies on the OUTER LIP - within ``lip_band_m`` of the
    largest radius about the plate axis (:func:`plate_axis`) - at tilts in ``lip_tilt_deg`` (the top mode ends at ``top_max_tilt_deg``,
    the side mode starts at ``side_tilt_deg[0]``; the window between them is proposed here too), each at every ``lip_pad_offsets_m``: the
    eef raised back along its approach by the offset with the contact point kept (a higher eef - the contact nearer the fingertips, so
    less blade reaches past the lip). An offset-0 pose inside the top mode's own window is left out (the top mode makes it).
    Returns ``(T (N, 4, 4), initial shift (N,))``: the contact sits at ``CONTACT_Z + shift`` along the approach."""
    empty = (np.zeros((0, 4, 4)), np.zeros(0))
    if len(T_all) == 0:
        return empty
    cx, cy, r_max = plate_axis(geom.union)
    c = T_all[:, :3, 3] + T_all[:, :3, 2] * CONTACT_Z
    tl = tilt_deg(T_all)
    m = (np.hypot(c[:, 0] - cx, c[:, 1] - cy) >= r_max - cfg.lip_band_m) & (tl >= cfg.lip_tilt_deg[0]) & (tl <= cfg.lip_tilt_deg[1])
    T, tl = T_all[m], tl[m]
    offs = [float(s) for s in cfg.lip_pad_offsets_m if CONTACT_Z + float(s) <= cfg.tip_max_m + 1e-9]
    cap = max(1, int(cfg.family_max_pool) // max(1, len(offs)))
    if len(T) > cap:
        pick = np.sort(np.random.default_rng(cfg.seed).choice(len(T), cap, replace=False))
        T, tl = T[pick], tl[pick]
    Ts, ss = [], []
    for s0 in offs:
        k = (tl > cfg.top_max_tilt_deg) if (s0 <= 0.0 and "top" in cfg.modes) else np.ones(len(T), dtype=bool)
        Tk = T[k].copy()
        Tk[:, :3, 3] -= Tk[:, :3, 2] * s0
        Ts.append(Tk)
        ss.append(np.full(len(Tk), s0))
    return (np.concatenate(Ts), np.concatenate(ss)) if Ts else empty


def rim_profile(geom: OG.ObjectGeometry, theta: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    """The rim's inner and outer faces about the plate axis at azimuths ``theta`` and heights ``z`` (paired arrays): a ray from the axis
    outward meets the inner face (``r_in``; only where the axis point is free - inside no piece, an open centre), a ray from outside inward
    the outer face (``r_out``). ``{"r_in", "r_out", "free", "radial" (N, 3), "axis" (N, 3)}``; NaN where a ray misses."""
    union = geom.union
    cx, cy, r_max = plate_axis(union)
    planes = [(pc.face_normals, np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0])) for pc in geom.pieces]
    rad = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    axis = np.stack([np.full_like(theta, cx), np.full_like(theta, cy), np.asarray(z, dtype=float)], axis=1)
    free = ~_inside_any(axis, planes)
    far = r_max + 0.05
    d_in, _ = hits(union, axis, rad)
    d_out, _ = hits(union, axis + rad * far, -rad)
    r_in = np.where(free & np.isfinite(d_in), d_in, np.nan)
    r_out = np.where(np.isfinite(d_out), far - d_out, np.nan)
    return {"r_in": r_in, "r_out": r_out, "free": free, "radial": rad, "axis": axis}


def straddle_family_poses(T_all: np.ndarray, geom: OG.ObjectGeometry, cfg: CandidateConfig) -> Tuple[np.ndarray, np.ndarray]:
    """``rim_straddle`` (owner option 3): approach about vertical (``straddle_tilts_deg`` in the radial plane, + leaning outward), the
    closing axis radial - one finger in the open centre, one outside the rim, the pads across the rim wall. Per azimuth
    (``straddle_n_azimuth``) and height (``straddle_n_heights`` over the object's z range) the pad contact centre is midway between the
    rim's inner and outer faces (:func:`rim_profile`), at every ``straddle_pad_offsets_m`` toward the fingertips (the tips stay above the
    support / the centre's floor while the pads meet the rim low on the pad). ``T_all`` is unused (the family is constructed).
    Returns ``(T (N, 4, 4), initial shift (N,))``."""
    union = geom.union
    lo, hi = union.vertices.min(0), union.vertices.max(0)
    th_ = np.linspace(0.0, 2.0 * np.pi, int(cfg.straddle_n_azimuth), endpoint=False)
    zs = np.linspace(lo[2] + 0.001, hi[2] - 0.001, int(cfg.straddle_n_heights))
    TH, ZZ = (a.ravel() for a in np.meshgrid(th_, zs, indexing="ij"))
    pr = rim_profile(geom, TH, ZZ)
    ok = np.isfinite(pr["r_in"]) & np.isfinite(pr["r_out"]) & (pr["r_out"] - pr["r_in"] >= cfg.min_width_m)
    rad, axis = pr["radial"][ok], pr["axis"][ok]
    c = axis + rad * (0.5 * (pr["r_in"][ok] + pr["r_out"][ok]))[:, None]
    up = np.array([0.0, 0.0, 1.0])
    Ts, ss = [], []
    for a_deg in cfg.straddle_tilts_deg:
        a = math.radians(float(a_deg))
        z = -math.cos(a) * up[None] + math.sin(a) * rad  # the approach: down, leaning outward for a > 0
        y = math.cos(a) * rad + math.sin(a) * up[None]   # the closing axis: radial, perpendicular to the approach
        x = np.cross(y, z)
        for s0 in cfg.straddle_pad_offsets_m:
            s0 = float(s0)
            if CONTACT_Z + s0 > cfg.tip_max_m + 1e-9:
                continue
            Ts.append(_frames(x, y, z, c - z * (CONTACT_Z + s0)))
            ss.append(np.full(len(c), s0))
    if not Ts:
        return np.zeros((0, 4, 4)), np.zeros(0)
    T, s = np.concatenate(Ts), np.concatenate(ss)
    if len(T) > cfg.family_max_pool:
        pick = np.sort(np.random.default_rng(cfg.seed).choice(len(T), int(cfg.family_max_pool), replace=False))
        T, s = T[pick], s[pick]
    return T, s


FAMILIES = {"lip_shallow": lip_family_poses, "rim_straddle": straddle_family_poses}


# ------------------------------------------------------------------------------------------------ gate dump (ISG_GLIB_GATE_DUMP=1)

GATE_DUMP_FILE = "gate_dump.json"
GATE_DUMP_SCHEMA = "isg-glib-gate-dump/1"
GATE_DUMP_NOTES = ("Per generation pass (the last one wrote candidates.json), per pool (a mode or a family) and gate: n_in / n_out, dropped_by "
                   "(why the dropped ones failed), stats (percentiles 0/10/50/90/100 of every numeric measure over the dropped ones) and up to "
                   "max_records dropped candidates with the measures the gate decided on. 'sweep' holds two tests: width_lost (the pinch line "
                   "meets no object after the support move, e.g. a side lift above the object) and touch (the pad sweep: a pad meets the object "
                   "further out than width / 2 + sweep_margin_m). 'geometry' lists every selected candidate with its final reason; "
                   "blade_depth_at_real_contact_mm is diagnostic (finger_blades_clear with the pinched pieces looked up at the real contact, "
                   "CONTACT_Z + the eef's real retreat; the call as written uses CONTACT_Z); shift_minus_real_mm > 0 = failed object-clearance "
                   "retreat tries counted in the shift (the sweep then re-measures the width that far off the contact). "
                   "Lengths in mm, contact_offset_mm = the contact's offset toward the fingertips (the retreat).")
SUPPORT_PARTS = ("finger", "palm", "wrist_camera", "wrist")


def _js(v):
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2) if math.isfinite(float(v)) else None
    return str(v)


class _GateDump:
    """Diagnostics only (no gate reads it): what each gate dropped and the measures it decided on (:data:`GATE_DUMP_NOTES`)."""

    def __init__(self, on: bool, max_records: int = 400):
        self.on, self.max_records = bool(on), int(max_records)
        self.pools: Dict[str, List[Dict[str, Any]]] = {}

    def gate(self, pool: str, gate: str, n_in: int, n_out: int, drop_idx=None, measures: Optional[Dict[str, np.ndarray]] = None, why=None,
             summary=None, records=None) -> None:
        if not self.on:
            return
        from collections import Counter

        st: Dict[str, Any] = {"gate": gate, "n_in": int(n_in), "n_out": int(n_out)}
        if why is not None:
            st["dropped_by"] = dict(Counter(str(w) for w in why).most_common())
        if summary is not None:
            st["summary"] = dict(Counter(str(w) for w in summary).most_common())
        if drop_idx is not None and measures:
            idx = np.asarray(drop_idx, dtype=int)
            stats = {}
            for k, v in measures.items():
                v = np.asarray(v)
                if v.ndim == 1 and v.dtype.kind in "fi" and len(idx):
                    x = v[idx].astype(float)
                    x = x[np.isfinite(x)]
                    stats[k] = {"n_finite": int(len(x)), "pct": _js(np.percentile(x, [0, 10, 50, 90, 100])) if len(x) else None}
            st["stats"] = stats
            st["dropped"] = [{k: _js(np.asarray(v)[i]) for k, v in measures.items()} for i in idx[: self.max_records]]
        if records is not None:
            st["records"] = records
        self.pools.setdefault(pool, []).append(st)


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
    dumps: List[Dict[str, Any]] = []
    out = _generate(job, model_dir, cfg, policy, sym_cfg, arm, dumps)
    for mc, mo in densify:
        if out["representatives"]:
            break
        dense = CandidateConfig(**{**asdict(cfg), "n_candidates": cfg.n_candidates * mc, "n_orientations": cfg.n_orientations * mo})
        prev = out["log"]
        out = _generate(job, model_dir, dense, policy, sym_cfg, arm, dumps)
        out["log"]["densified_from"] = prev
    if cfg.gate_dump:
        S.save_json(os.path.join(model_dir, GATE_DUMP_FILE), {"schema": GATE_DUMP_SCHEMA, "id": job.get("id"), "category": job.get("category"),
                                                           "model": job.get("model"), "families": list(cfg.families), "passes": dumps,
                                                           "final_pass": len(dumps) - 1, "notes": GATE_DUMP_NOTES, "created": S.now()})
    return out


def _generate(job: Dict[str, Any], model_dir: str, cfg: CandidateConfig, policy: CP.ContactPolicy, sym_cfg: SY.SymmetryConfig, arm: str,
              dumps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import torch as th

    t0 = time.time()
    geom0 = OG.load(model_dir)
    if geom0 is None:
        raise FileNotFoundError(f"no geometry.json in {model_dir} (run the export stage)")
    constraint = job.get("constraint") or "tabletop"
    # of-v2maps 2026-11-07 (GB_SETTLE_FIRST): a surveyed rest generates the candidates (generation_rest_wxyz), the designated hint names the entry
    R = rest_rotation(job.get("generation_rest_wxyz") or job.get("rest_hint_wxyz"))
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
    dump = _GateDump(cfg.gate_dump)
    # the pools: the modes first (as before: their names, order and numbers are unchanged), then the opt-in families (top-type)
    pools: List[Tuple[str, str, Optional[np.ndarray], Optional[np.ndarray]]] = []
    for mode in cfg.modes:
        if len(T_all) == 0:
            pools.append((mode, mode, None, None))
            continue
        tl = tilt_deg(T_all)
        m = tl <= cfg.top_max_tilt_deg if mode == "top" else (tl >= cfg.side_tilt_deg[0]) & (tl <= cfg.side_tilt_deg[1])
        pools.append((mode, mode, T_all[m].copy(), None))
    for fam in cfg.families:
        if fam not in FAMILIES:
            raise ValueError(f"unknown candidate family {fam!r} (known: {sorted(FAMILIES)})")
        Tf, sf = FAMILIES[fam](T_all, geom, cfg)
        pools.append((fam, "top", Tf, sf))
    n_named: Dict[str, int] = {}
    top_z = float(hi[2])
    ax_x, ax_y, _ = plate_axis(union)

    def _where(Tm, czm):  # dump: where a candidate pinches (rest frame, about the plate axis), its tilt and its contact offset toward the tips
        czm = np.broadcast_to(np.asarray(czm, dtype=float), (len(Tm),))
        cm = Tm[:, :3, 3] + Tm[:, :3, 2] * czm[:, None]
        return {"contact_mm": cm * 1000, "contact_r_mm": np.hypot(cm[:, 0] - ax_x, cm[:, 1] - ax_y) * 1000,
                "contact_azimuth_deg": np.degrees(np.arctan2(cm[:, 1] - ax_y, cm[:, 0] - ax_x)), "tilt_deg": tilt_deg(Tm),
                "contact_offset_mm": (czm - CONTACT_Z) * 1000}

    for pool, mode, T, shift0 in pools:
        L: Dict[str, Any] = {}
        if T is None:
            log[pool] = L
            continue
        is_family = shift0 is not None
        L["n_pool" if is_family else "n_tilt"] = int(len(T))
        dump.gate(pool, "pool", len(T) if is_family else len(T_all), len(T))  # a family pool is proposed, not filtered from the samples
        if len(T) == 0:
            log[pool] = L
            continue
        cz0 = CONTACT_Z + shift0 if is_family else CONTACT_Z  # a family pose may start with its contact toward the tips
        if not is_family:
            shift0 = np.zeros(len(T))
        _, width, antip, cov = contact_features(union, T, cz0)
        keep = np.isfinite(width) & (width >= cfg.min_width_m) & (width <= cfg.aperture_m) & (antip >= math.cos(math.radians(cfg.cone_deg)))
        if dump.on:
            ok_w = np.isfinite(width) & (width >= cfg.min_width_m) & (width <= cfg.aperture_m)
            why = np.where(~np.isfinite(width), "no_width", np.where(~ok_w, "width_range", "antipodal"))
            dump.gate(pool, "width_antipodal", len(T), int(keep.sum()), np.where(~keep)[0],
                      {**_where(T, cz0), "width_mm": width * 1000, "antipodal": antip, "why": why}, why[~keep])
        T, width, shift0 = T[keep], width[keep], shift0[keep]
        L["n_width_antipodal"] = int(len(T))
        if len(T) == 0:
            log[pool] = L
            continue
        # 3) the support under the hand: policy measurement on the exact hulls at the pre-shaped opening
        shift = np.zeros(len(T))
        pre_q = np.clip(0.5 * width + PRESHAPE_MARGIN, 0.0, FINGER_OPEN)
        deficit = np.zeros(len(T))
        part = np.zeros(len(T), dtype=int)  # dump: the hand part furthest below its allowance (index into SUPPORT_PARTS)
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
            if dump.on:
                part[idx] = np.argmin(np.stack([cl["finger"].numpy() + policy.finger_support_max_penetration_m] +
                                               [cl[r].numpy() - policy.hand_support_margin_m for r in SUPPORT_PARTS[1:]]), axis=0)
        a = T[:, :3, 2]
        if mode == "top":  # retreat along the approach (as before; the support's normal component of the retreat lifts the hand)
            s = deficit / np.maximum(-a[:, 2], 1e-3)
            ok = (CONTACT_Z + shift0 + s) <= cfg.tip_max_m + 1e-9  # the contact no further out than the tip surface (gen_candidates TIP_MAX)
            if dump.on:
                dump.gate(pool, "support_retreat", len(T), int(ok.sum()), np.where(~ok)[0],
                          {**_where(T, CONTACT_Z + shift0), "deficit_mm": deficit * 1000, "retreat_mm": s * 1000, "part": np.array(SUPPORT_PARTS)[part]},
                          np.array(SUPPORT_PARTS)[part][~ok])
            T, s, width, pre_q, shift0 = T[ok], s[ok], width[ok], pre_q[ok], shift0[ok]
            T[:, :3, 3] -= T[:, :3, 2] * s[:, None]
            shift = shift0 + s
        else:  # lift along the support normal (lift_grasp_off_support's move); the pads must still close on the object (sweep below)
            ok = deficit <= cfg.side_lift_max_m
            if dump.on:
                dump.gate(pool, "support_lift", len(T), int(ok.sum()), np.where(~ok)[0],
                          {**_where(T, CONTACT_Z), "lift_mm": deficit * 1000, "part": np.array(SUPPORT_PARTS)[part]}, np.array(SUPPORT_PARTS)[part][~ok])
            lift_part = np.array(SUPPORT_PARTS)[part][ok]
            T, width, pre_q, deficit, shift0 = T[ok], width[ok], pre_q[ok], deficit[ok], shift0[ok]
            T[:, 2, 3] += deficit
            shift = deficit
        L["n_support"] = int(len(T))
        if len(T) == 0:
            log[pool] = L
            continue
        # 4) object clearance (spheres) with the object-aware retreat of top grasps
        clear = _clear_grouped(clear_fn, T, pre_q)
        ok = clear >= cfg.obj_margin_m
        # the eef's real displacement along the approach (the shift as written may count failed retreat tries)
        real = _object_retreats(clear_fn, T, pre_q, shift, clear, ok, cfg) if mode == "top" else shift.copy()
        if dump.on:
            dump.gate(pool, "object_clear", len(T), int(ok.sum()), np.where(~ok)[0],
                      {**_where(T, CONTACT_Z + (shift if mode == "top" else 0.0)), "sphere_clear_mm": clear * 1000, "shift_mm": shift * 1000})
        T, clear, shift, width, pre_q, shift0, real = T[ok], clear[ok], shift[ok], width[ok], pre_q[ok], shift0[ok], real[ok]
        if mode != "top":
            lift_part = lift_part[ok]
        L["n_object_clear"] = int(len(T))
        if len(T) == 0:
            log[pool] = L
            continue
        cz = CONTACT_Z + (shift if mode == "top" else 0.0)  # the pads' contact height after the retreat
        cz = np.broadcast_to(np.asarray(cz, dtype=float), (len(T),)).copy()
        c, width, antip, cov = contact_features(union, T, cz)
        touch = pad_sweep(union, T)
        keep = np.isfinite(width) & (width >= cfg.min_width_m) & (touch <= width / 2 + cfg.sweep_margin_m)
        if dump.on:
            # the stage holds two tests: the pinch line still meets the object after the support move (width_lost: e.g. a side lift
            # above the object), and the pad sweep proper (touch: the pads meet the object no further out than width / 2 + margin)
            lost = ~(np.isfinite(width) & (width >= cfg.min_width_m))
            why = np.where(lost, "width_lost", "touch")
            ms = {**_where(T, cz), "why": why, "width_mm": width * 1000, "touch_mm": touch * 1000, "limit_mm": (width / 2 + cfg.sweep_margin_m) * 1000,
                  "contact_above_object_top_mm": (c[:, 2] - top_z) * 1000, "shift_mm": shift * 1000, "shift_minus_real_mm": (shift - real) * 1000}
            if mode != "top":
                ms["lift_part"] = lift_part
            dump.gate(pool, "sweep", len(T), int(keep.sum()), np.where(~keep)[0], ms, why[~keep])
        T, clear, shift, cz, width, antip, cov, pre_q, shift0, real = (T[keep], clear[keep], shift[keep], cz[keep], width[keep], antip[keep],
                                                                       cov[keep], pre_q[keep], shift0[keep], real[keep])
        L["n_sweep"] = int(len(T))
        if len(T) == 0:
            log[pool] = L
            continue
        score, perp = _score(T, com, width, antip, cov, clear, size, cz, mode, cfg.weights)
        feats = {"com_perp": perp, "width": width, "antipodal": antip, "coverage": cov, "clearance": clear, "tilt_deg": tilt_deg(T), "shift_m": shift}
        if is_family:
            feats["pad_offset_m"] = shift0
        if constraint != "flower_stem_top":
            keep = perp <= cfg.lever_cap * size
            if dump.on:
                dump.gate(pool, "lever", len(T), int(keep.sum()), np.where(~keep)[0], {**_where(T, cz), "com_perp_mm": perp * 1000,
                                                                                      "cap_mm": np.full(len(T), cfg.lever_cap * size * 1000)})
            T, score, cz, pre_q, real = T[keep], score[keep], cz[keep], pre_q[keep], real[keep]
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
            T, score, cz, pre_q, side_m, face_gap, real = T[keep], score[keep], cz[keep], pre_q[keep], side_m[keep], face_gap[keep], real[keep]
            feats = {k: v[keep] for k, v in feats.items()}
            feats["attach_side_m"], feats["attach_face_gap_m"] = side_m, face_gap
            score = score + 2.0 * np.clip(side_m / 0.03, 0, 1)
            L["n_attachment_free"] = int(len(T))
            dump.gate(pool, "attachment_free", L.get("n_lever", L["n_sweep"]), len(T))
        elif constraint == "flower_stem_top" and len(T):
            head_pt, axis_dir, Ls = stem_axis(union, lo, hi)
            cc = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
            t = np.einsum("ij,j->i", cc - head_pt[None], axis_dir) / Ls
            across = np.abs(T[:, :3, 1] @ axis_dir)
            w_lo, w_hi, w_c = FLOWER_WINDOW.get(job["category"], FLOWER_WINDOW_DEFAULT)
            keep = (t >= w_lo) & (t <= w_hi) & (across <= 0.5) & (feats["width"] <= STEM_MAX_WIDTH)
            T, score, cz, pre_q, t, across, real = T[keep], score[keep], cz[keep], pre_q[keep], t[keep], across[keep], real[keep]
            feats = {k: v[keep] for k, v in feats.items()}
            score = score + 5.0 * np.exp(-((t - w_c) / 0.15) ** 2) - 1.0 * across
            feats["stem_t"], feats["pinch_across"] = t, across
            L["n_flower"] = int(len(T))
            dump.gate(pool, "flower_stem_top", L["n_sweep"], len(T))
        if len(T) == 0:
            log[pool] = L
            continue
        sel = nms(T, score, cfg.topk, cfg.nms_pos_m, cfg.nms_rot_deg)
        L["n_selected"] = int(len(sel))
        dump.gate(pool, "nms", len(T), len(sel))
        # 7) the full policy geometry: pre-grasp, approach, closing sweep (fingers from the pre-shape to the contact opening)
        Ts = T[sel]
        contact_q = np.clip(0.5 * feats["width"][sel] + pad_face_offset(arm), 0.0, FINGER_OPEN)  # joint value at the pads' first touch
        geo = CP.check_grasp_geometry(hc, th.as_tensor(Ts, dtype=th.float32), support, contact_q, open_q=float(np.max(pre_q[sel])))
        appr = _approach_object_clear(clear_fn, Ts, pre_q[sel], cfg) if cfg.approach_obj_check else np.ones(len(sel), dtype=bool)
        cz_real = (CONTACT_Z + (real if mode == "top" else 0.0)) * np.ones(len(T))  # the pads' real contact offset (the eef's real retreat)
        blade_kw = {"contact_z": cz_real[sel].reshape(-1, 1)} if cfg.blade_contact_at_retreat else {}  # OWNER DECISION switch (default on since i7-switches)
        blades, deep = finger_blades_clear(geom, hc, Ts, pre_q[sel], contact_q, **blade_kw)
        if dump.on:  # diagnostics: the blade depth with the pinched pieces looked up at the real contact (what the switch would do)
            deep_at = deep if blade_kw else finger_blades_clear(geom, hc, Ts, pre_q[sel], contact_q, contact_z=cz_real[sel].reshape(-1, 1))[1]
        recs = []
        for r, i in enumerate(sel):
            T_obj = np.linalg.inv(R) @ Ts[r]  # back to the canonical object frame
            ok_geo = bool(geo["ok"][r]) and bool(appr[r]) and bool(blades[r])
            reason = (geo["reason"][r] if not bool(geo["ok"][r]) else ("approach_hits_object" if not appr[r] else
                      ("finger_blade_in_object" if not blades[r] else "ok")))
            name = f"{pool}_{n_named.get(pool, 0)}"
            n_named[pool] = n_named.get(pool, 0) + 1
            cand = {
                "name": name, "mode": mode, "score": round(float(score[i]), 4),
                "T_object_eef": S.mat_pose(T_obj), "T_rest_eef": S.mat_pose(Ts[r]), "width_m": round(float(feats["width"][i]), 5),
                "preshape_q": round(float(pre_q[i]), 5), "contact_q": round(float(contact_q[r]), 5),
                "features": {k: round(float(v[i]), 5) for k, v in feats.items()},
                "geometry": {"ok": ok_geo, "reason": reason, "stage": geo["stage"][r], "finger_min_m": round(float(geo["finger_min"][r]), 5),
                             "hand_min_m": round(float(geo["hand_min"][r]), 5), "blade_object_depth_m": round(float(deep[r]), 5)},
            }
            if is_family:
                cand["family"] = pool
            out_cands.append(cand)
            if dump.on:
                recs.append({"name": name, "reason": reason, "stage": geo["stage"][r], "approach_object_clear": bool(appr[r]),
                             "blade_depth_mm": round(float(deep[r]) * 1000, 2), "blade_depth_at_real_contact_mm": round(float(deep_at[r]) * 1000, 2),
                             "shift_minus_real_mm": round(float(feats["shift_m"][i] - real[i]) * 1000, 2) if mode == "top" else 0.0,
                             "finger_min_mm": round(float(geo["finger_min"][r]) * 1000, 2), "hand_min_mm": round(float(geo["hand_min"][r]) * 1000, 2),
                             **{k: _js(v[0]) for k, v in _where(Ts[r:r + 1], cz[sel][r:r + 1]).items()}, "width_mm": round(float(feats["width"][i]) * 1000, 2)})
        if dump.on:
            dump.gate(pool, "geometry", len(sel), sum(1 for x in recs if x["reason"] == "ok"), summary=[x["reason"] for x in recs], records=recs)
        log[pool] = L
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
    if dumps is not None and dump.on:
        dumps.append({"n_candidates": cfg.n_candidates, "n_orientations": cfg.n_orientations, "cone_deg": cfg.cone_deg, "n_sampled": int(len(T_all)),
                      "object_top_z_mm": round(top_z * 1000, 2), "support_z_mm": round(z_min * 1000, 2), "plate_axis_xy_mm": [round(ax_x * 1000, 2), round(ax_y * 1000, 2)],
                      "log": {k: v for k, v in log.items() if isinstance(v, dict)}, "pools": dump.pools})
    out = {"schema": S.CANDIDATES, "pipeline": PIPELINE, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
           "dataset": job.get("dataset"), "constraint": constraint, "rest_hint_wxyz": job.get("rest_hint_wxyz"), "generation_rest_wxyz": job.get("generation_rest_wxyz"), "legacy_key": job.get("legacy_key"),
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


def _object_retreats(clear_fn, T, pre_q, shift, clear, ok, cfg: CandidateConfig) -> np.ndarray:
    """The object-aware retreat of top grasps, in place on ``T`` / ``shift`` / ``clear`` / ``ok``: a pose whose hand spheres are not clear
    of the object tries ``cfg.retreats_m`` (absolute retreats from its pose) while its contact stays within ``tip_max_m``. Returns the
    eef's real displacement along the approach (``shift`` on entry + the retreat that cleared). Without ``cfg.retreat_shift_exact`` (the
    behaviour before i7-switches) a failed try still adds to ``shift``; with it (owner decision, default on since i7-switches) the shift is
    the real displacement."""
    real = shift.copy()
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
        real[idx] += ds
        ok[idx] = True
        if not cfg.retreat_shift_exact:
            shift[bad[~good]] += ds
    return real


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
