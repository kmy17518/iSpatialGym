#!/usr/bin/env python3
"""Rotational-symmetry primitives shared by the symmetry survey (``survey_symmetry.py``).

* candidate rotations of the canonical object frame and their tags (``x2 y2 z2 x4 y4 z4 x3 y3 z3 x5 y5 z5 x6 y6 z6
  xinf yinf zinf full``), reduction of an accepted tag set to generators,
* geometric score of a rotation: symmetric Chamfer distance (mean nearest-neighbour distance in both directions,
  scipy cKDTree) between an area-weighted surface sample and its rotated copy, normalised by the bbox diagonal,
  evaluated about the bbox centre and about the surface centroid (the better one is kept),
* texture score of a rotation: flat-shaded (no lighting) renders of the textured model and of the rotated model from
  six fixed cameras, compared over the pixels fully covered by the object in both renders (mean absolute difference,
  95th and 99th percentile; the union-mask MAD and the mask IoU are recorded as well).

CPU only except ``Renderer`` (pyrender / EGL, ~0.3 GB of GPU memory).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
FOLD_ANGLE = {2: 180.0, 4: 90.0, 3: 120.0, 5: 72.0, 6: 60.0}
#: angles (deg) for the continuous ("inf") test: none is a multiple of 30 / 45 / 60 / 72 / 90, so a polygonal
#: 3-/4-/5-/6-fold body cannot pass as a body of revolution.
INF_ANGLES = [17.0, 39.0, 53.0, 101.0, 131.0, 163.0, 197.0, 251.0]
TAG_ORDER = ["x2", "y2", "z2", "x4", "y4", "z4", "x3", "y3", "z3", "x5", "y5", "z5", "x6", "y6", "z6", "xinf", "yinf", "zinf", "full"]
N_FULL = 8


def rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def random_rotations(n: int = N_FULL, seed: int = 12345) -> List[np.ndarray]:
    """Fixed pseudo-random rotations (uniform on SO(3), Shoemake) for the spherical test."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        u1, u2, u3 = rng.uniform(size=3)
        q = np.array([math.sqrt(1 - u1) * math.sin(2 * math.pi * u2), math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                      math.sqrt(u1) * math.sin(2 * math.pi * u3), math.sqrt(u1) * math.cos(2 * math.pi * u3)])
        x, y, z, w = q
        out.append(np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]))
    return out


def candidate_rotations() -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """tag -> [(rotation name, 3x3 matrix)]; a tag is accepted when every rotation of its list is accepted."""
    cands: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for ax in "xyz":
        for fold, deg in FOLD_ANGLE.items():
            cands[f"{ax}{fold}"] = [(f"{ax}{deg:g}", rot_axis(AXES[ax], deg))]
        cands[f"{ax}inf"] = [(f"{ax}{deg:g}", rot_axis(AXES[ax], deg)) for deg in INF_ANGLES]
    cands["full"] = [(f"rand{i}", R) for i, R in enumerate(random_rotations())]
    return cands


def reduce_tags(tags: Iterable[str]) -> List[str]:
    """Accepted tags -> generators: ``full`` absorbs everything; ``ainf`` absorbs every finite fold about ``a``; ``a6``
    absorbs ``a3`` / ``a2``; ``a4`` absorbs ``a2``; with ``ainf`` the two perpendicular half-turns are conjugate (one is
    kept, e.g. a plain cylinder -> ``zinf x2``). Finite folds about different axes are all kept (a box -> ``x2 y2 z2``)."""
    t = set(tags)
    if "full" in t:
        return ["full"]
    for a in "xyz":
        if f"{a}inf" in t:
            t -= {f"{a}{k}" for k in (2, 3, 4, 5, 6)}
        if f"{a}6" in t:
            t -= {f"{a}3", f"{a}2"}
        if f"{a}4" in t:
            t.discard(f"{a}2")
    for a, (p, q) in {"z": ("x", "y"), "x": ("y", "z"), "y": ("x", "z")}.items():
        if f"{a}inf" in t and f"{p}2" in t and f"{q}2" in t:
            t.discard(f"{q}2")
    return [x for x in TAG_ORDER if x in t]


def expand_tags(tags: Iterable[str]) -> List[str]:
    """Inverse of the generator reduction for reporting (which finite folds a tag set implies)."""
    t = set(tags)
    if "full" in t:
        return list(TAG_ORDER)
    for a in "xyz":
        if f"{a}inf" in t:
            t |= {f"{a}{k}" for k in (2, 3, 4, 5, 6)}
        if f"{a}6" in t:
            t |= {f"{a}3", f"{a}2"}
        if f"{a}4" in t:
            t.add(f"{a}2")
    for a, (p, q) in {"z": ("x", "y"), "x": ("y", "z"), "y": ("x", "z")}.items():
        if f"{a}inf" in t and (f"{p}2" in t or f"{q}2" in t):
            t |= {f"{p}2", f"{q}2"}
    return [x for x in TAG_ORDER if x in t]


# ----------------------------------------------------------------------------------------------- geometry

def sample_surface(mesh, n: int, seed: int) -> np.ndarray:
    """Area-weighted surface sample (trimesh) with a fixed seed."""
    import trimesh

    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def _nn_stats(d: np.ndarray, diag: float) -> Dict[str, float]:
    d = d / diag
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "frac2": float((d > 0.02).mean())}


#: distances are capped at this fraction of the bbox diagonal (KD-tree queries for points far from every surface point
#: are 20-30x slower than for points on the surface; anything beyond the cap is "not symmetric" anyway)
DIST_CAP = 0.10
#: coarse pre-test (2k points, one direction): a rotation whose coarse mean exceeds this is rejected without the full test
COARSE_POINTS = 2000
COARSE_REJECT = 0.03


def rotation_score(tree: cKDTree, P: np.ndarray, R: np.ndarray, centre: np.ndarray, diag: float,
                   both: bool = True) -> Dict[str, float]:
    """Symmetric Chamfer distance (mean nearest-neighbour distance of RP -> P and P -> RP, both against the dense
    tree) normalised by the bbox diagonal, plus the 95th percentile and the fraction of points farther than 2 %.
    ``both=False``: one direction only (coarse pre-test)."""
    cap = DIST_CAP * diag
    Q1 = (P - centre) @ R.T + centre  # R P
    d1, _ = tree.query(Q1, workers=1, distance_upper_bound=cap)
    if both:
        Q2 = (P - centre) @ R + centre  # R^-1 P  (dist(P, RP) = dist(R^-1 P, P))
        d2, _ = tree.query(Q2, workers=1, distance_upper_bound=cap)
        d1 = np.concatenate([d1, d2])
    return _nn_stats(np.minimum(d1, cap), diag)


def geometric_scores(mesh, n_points: int = 20000, n_dense: int = 400000, seed: int = 0) -> Dict[str, object]:
    """Per candidate tag: the worst (max-mean) rotation score of its rotation list about the better centre.

    Returns ``{"diag", "bbox_centre", "centroid", "floor": {mean, p95}, "tags": {tag: {mean, p95, frac2, centre,
    worst, coarse}}}`` where ``centre`` is ``bbox`` or ``centroid`` (the one with the smaller worst-case mean),
    ``coarse = 1`` marks a tag rejected by the 2k-point pre-test (its score is the coarse one-directional score) and
    ``floor`` is the score of an independent fresh sample against the dense tree (the sampling noise an exactly
    symmetric body scores)."""
    dense = sample_surface(mesh, n_dense, seed)
    P = dense[:n_points]
    Pc = dense[:COARSE_POINTS]
    fresh = sample_surface(mesh, n_points, seed + 1)
    lo, hi = dense.min(0), dense.max(0)
    diag = float(np.linalg.norm(hi - lo))
    centres = {"bbox": (lo + hi) / 2.0, "centroid": dense.mean(0)}
    tree = cKDTree(dense)
    dfl, _ = tree.query(fresh, workers=1)
    floor = _nn_stats(dfl, diag)
    out_tags = {}
    for tag, rots in candidate_rotations().items():
        # coarse pass per centre: worst rotation, early exit once a rotation is clearly not a symmetry
        coarse = {}
        for cname, c in centres.items():
            worst = None
            for rname, R in rots:
                s = rotation_score(tree, Pc, R, c, diag, both=False)
                if worst is None or s["mean"] > worst["mean"]:
                    worst = dict(s, worst=rname)
                if worst["mean"] > COARSE_REJECT:
                    break
            coarse[cname] = worst
        cbest = min(coarse, key=lambda k: coarse[k]["mean"])
        if coarse[cbest]["mean"] > COARSE_REJECT:
            out_tags[tag] = dict(coarse[cbest], centre=cbest, coarse=1)
            continue
        worst = None
        for rname, R in rots:
            s = rotation_score(tree, P, R, centres[cbest], diag, both=True)
            if worst is None or s["mean"] > worst["mean"]:
                worst = dict(s, worst=rname)
        out_tags[tag] = dict(worst, centre=cbest, coarse=0)
    return {"diag": diag, "bbox_centre": centres["bbox"].tolist(), "centroid": centres["centroid"].tolist(), "floor": floor,
            "tags": out_tags}


#: geometric thresholds (fractions of the bbox diagonal): mean symmetric Chamfer, the 95th percentile of the
#: nearest-neighbour distances and the fraction of points displaced by more than 2 % (``frac2``; a mean over the whole
#: surface dilutes small asymmetric parts such as a mug handle - 6 % of the surface, displaced by 20 % - while frac2
#: does not; a 16-gon bowl rotated by 17 deg has frac2 = 2 %).
GEOM_EXACT = {"mean": 0.005, "p95": 0.012, "frac2": 0.01}
GEOM_NEAR = {"mean": 0.015, "frac2": 0.03}


def classify_geom(tag_scores: Dict[str, dict], exact=GEOM_EXACT, near=GEOM_NEAR) -> Tuple[List[str], List[str]]:
    """(exact tags, near tags) - raw (un-reduced) tags; near excludes exact."""
    ex, nr = [], []
    for tag in TAG_ORDER:
        s = tag_scores.get(tag)
        if s is None:
            continue
        if s["mean"] <= exact["mean"] and s["p95"] <= exact["p95"] and s["frac2"] <= exact["frac2"]:
            ex.append(tag)
        elif s["mean"] <= near["mean"] and s["frac2"] <= near["frac2"]:
            nr.append(tag)
    return ex, nr


# ----------------------------------------------------------------------------------------------- texture

def look_at(eye, target, up) -> np.ndarray:
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def camera_directions(elev_deg: float = 25.0, azim_deg: float = 20.0) -> List[Tuple[str, np.ndarray]]:
    """Six unit vectors (from the object centre towards the camera): +-x, +-y (azimuth offset ``azim_deg``, elevation
    ``elev_deg``) and +-z (tilted ``elev_deg`` towards -y and twisted ``azim_deg``), so that no face is edge-on."""
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    out = []
    for name, base in (("+x", 0.0), ("+y", 90.0), ("-x", 180.0), ("-y", 270.0)):
        a = math.radians(base) + az
        out.append((name, np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)])))
    for name, sz in (("+z", 1.0), ("-z", -1.0)):
        out.append((name, np.array([math.sin(el) * math.sin(az), -math.sin(el) * math.cos(az), sz * math.cos(el)])))
    return out


class Renderer:
    """Flat-shaded (``RenderFlags.FLAT``: colour = albedo texture, no lighting) offscreen renders of a list of trimesh
    parts from the six fixed cameras, optionally with a rotation about ``centre`` applied to the object.

    Rendered at ``size * ss`` and box-downsampled to ``size`` (cheap anti-aliasing of the silhouettes)."""

    def __init__(self, size: int = 192, ss: int = 2, yfov_deg: float = 38.0, bg=(255, 255, 255)):
        import pyrender

        self.pyrender = pyrender
        self.base_size, self.size, self.ss = size, size, ss
        self.yfov = math.radians(yfov_deg)
        self.bg = np.array(bg, dtype=np.float64) / 255.0
        self.r = pyrender.OffscreenRenderer(size * ss, size * ss)
        self.dirs = camera_directions()

    def set_size(self, size: int):
        """Change the output resolution (elongated objects are rendered larger so that their thin side has pixels)."""
        if size != self.size:
            self.size = size
            self.r.viewport_width = size * self.ss
            self.r.viewport_height = size * self.ss

    @staticmethod
    def size_factor(bbox_extents) -> int:
        """1 or 2: a body whose longest extent is >= 4.5x its middle extent (pencil, chopstick, rod) is rendered at
        twice the base resolution (384 px, still 2x supersampled); at 192 px a 19 cm pencil is 4 px thick."""
        e = sorted(float(v) for v in bbox_extents)
        elong = e[2] / max(e[1], 1e-6)
        return 2 if elong >= 4.5 else 1

    def close(self):
        try:
            self.r.delete()
        except Exception:
            pass

    def _pyrender_meshes(self, meshes):
        import trimesh

        out = []
        for m in meshes:
            try:
                out.append(self.pyrender.Mesh.from_trimesh(m, smooth=False))
            except Exception:
                mm = m.copy()
                mm.visual = trimesh.visual.ColorVisuals(mm, face_colors=[190, 190, 200, 255])
                out.append(self.pyrender.Mesh.from_trimesh(mm, smooth=False))
        return out

    def prepare(self, meshes, centre: np.ndarray, diag: float):
        """Build the pyrender meshes once per model; ``render`` then only changes the object pose."""
        self._pmeshes = self._pyrender_meshes(meshes)
        self.centre = np.asarray(centre, dtype=np.float64)
        self.dist = 1.55 * diag + 0.03

    def render(self, R: Optional[np.ndarray] = None, centre: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """[(rgb uint8 (size, size, 3), mask bool (size, size))] per camera for the object rotated by ``R`` about
        ``centre`` (default: the bbox centre the cameras look at)."""
        pyrender = self.pyrender
        pose = np.eye(4)
        if R is not None:
            c = self.centre if centre is None else np.asarray(centre, dtype=np.float64)
            pose[:3, :3] = R
            pose[:3, 3] = c - R @ c
        out = []
        for _, d in self.dirs:
            scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[*self.bg, 1.0])
            for pm in self._pmeshes:
                scene.add(pm, pose=pose)
            eye = self.centre + d * self.dist
            up = [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [0.0, 1.0, 0.0]
            cam = pyrender.PerspectiveCamera(yfov=self.yfov, znear=0.005, zfar=50.0)
            scene.add(cam, pose=look_at(eye, self.centre, up))
            color, depth = self.r.render(scene, flags=pyrender.RenderFlags.FLAT)
            rgb = color[..., :3].astype(np.float64)
            cov = (depth > 0).astype(np.float64)
            if self.ss > 1:
                s = self.ss
                n = self.size
                rgb = rgb.reshape(n, s, n, s, 3).mean(axis=(1, 3))
                cov = cov.reshape(n, s, n, s).mean(axis=(1, 3))
            out.append((np.clip(rgb, 0, 255).astype(np.uint8), cov))
        return out


def image_diff(a: Sequence[Tuple[np.ndarray, np.ndarray]], b: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """Compare two render sets (same cameras). ``mad`` / ``p95`` / ``p99`` = mean, 95th and 99th percentile of the
    per-pixel channel-mean |diff| (fractions of 255) over the pixels fully covered by the object in both renders
    (partially covered silhouette pixels mix the background colour in and depend on the sub-pixel position of the
    edge, so they are excluded; for an exact geometric symmetry the two silhouettes coincide anyway); ``mad_union`` =
    the same mean over the union of the (any-coverage) object masks - the raw union-mask criterion, which also counts
    silhouette mismatches; ``iou`` = mask IoU (coverage >= 1/2)."""
    diffs, udiffs, inter, union = [], [], 0, 0
    for (ia, ca), (ib, cb) in zip(a, b):
        ma, mb = ca >= 0.5, cb >= 0.5
        u = ma | mb
        if not u.any():
            continue
        d = np.abs(ia.astype(np.float64) - ib.astype(np.float64)).mean(axis=2) / 255.0
        full = (ca >= 1.0) & (cb >= 1.0)
        if full.any():
            diffs.append(d[full])
        udiffs.append(d[u])
        inter += int((ma & mb).sum())
        union += int(u.sum())
    if not udiffs:
        return {"mad": 1.0, "p95": 1.0, "p99": 1.0, "mad_union": 1.0, "iou": 0.0}
    du = np.concatenate(udiffs)
    d = np.concatenate(diffs) if diffs else du
    return {"mad": float(d.mean()), "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
            "mad_union": float(du.mean()), "iou": inter / max(union, 1)}


#: texture thresholds (fractions of 255) on the fully-covered-pixel comparison. ``p99`` guards against small features
#: (die pips, letters, clock numerals, small logos) that cover 1-5 % of the object's pixels and so barely move the
#: mean / p95: >= 1 % of the pixels differing by more than 60/255 is a visible feature, never borderline. The
#: borderline band is closed by ``p95`` as well because a muted label on a bottle or a wrapper logo moves the mean by
#: only 7-12/255 while >= 5 % of the pixels differ by > 25/255 (tuned on a medicine bottle, a chocolate bar, a numeral
#: clock face, a magazine cover, a licence plate: broken; a tennis-ball seam, a pencil tip, a low-contrast letter
#: block: borderline; plain bowls / plates / caps / boxes / beakers: preserved).
TEX_PRESERVED = {"mad": 3.0 / 255.0, "p95": 20.0 / 255.0, "p99": 60.0 / 255.0}
TEX_BORDERLINE = {"mad": 12.0 / 255.0, "p95": 25.0 / 255.0, "p99": 60.0 / 255.0}
TEX_BORDERLINE_MAD = TEX_BORDERLINE["mad"]


def classify_tex(score: Dict[str, float], preserved=TEX_PRESERVED, borderline=TEX_BORDERLINE) -> str:
    """``preserved`` / ``borderline`` / ``broken`` for the worst rotation score of a tag."""
    p99 = score.get("p99", score["p95"])
    if score["mad"] < preserved["mad"] and score["p95"] < preserved["p95"] and p99 < preserved["p99"]:
        return "preserved"
    if score["mad"] <= borderline["mad"] and score["p95"] <= borderline["p95"] and p99 <= borderline["p99"]:
        return "borderline"
    return "broken"
