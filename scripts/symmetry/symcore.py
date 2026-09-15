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

    def __init__(self, size: int = 384, ss: int = 2, yfov_deg: float = 38.0, bg=(255, 255, 255)):
        import pyrender

        self.pyrender = pyrender
        self.base_size, self.size, self.ss = size, size, ss
        self.yfov = math.radians(yfov_deg)
        self.bg = np.array(bg, dtype=np.float64) / 255.0
        self.r = pyrender.OffscreenRenderer(size * ss, size * ss)
        self.dirs = camera_directions()
        self.dist = 0.0

    @property
    def view_names(self) -> List[str]:
        return [n for n, _ in self.dirs]

    @property
    def m_per_px(self) -> float:
        """Metres per output pixel at the object centre (the render scale the blob test converts millimetres with)."""
        return 2.0 * self.dist * math.tan(self.yfov / 2) / self.size

    def set_size(self, size: int):
        """Change the output resolution (elongated objects are rendered larger so that their thin side has pixels)."""
        if size != self.size:
            self.size = size
            self.r.viewport_width = size * self.ss
            self.r.viewport_height = size * self.ss

    @staticmethod
    def size_factor(bbox_extents) -> int:
        """1 or 2: a body whose longest extent is >= 4.5x its middle extent (pencil, chopstick, rod) is rendered at
        twice the base resolution (768 px, still 2x supersampled); at 384 px a 19 cm pencil is 8 px thick."""
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
            dep = depth.astype(np.float64)
            if self.ss > 1:
                s = self.ss
                n = self.size
                rgb = rgb.reshape(n, s, n, s, 3).mean(axis=(1, 3))
                cnt = cov.reshape(n, s, n, s).sum(axis=(1, 3))
                dep = dep.reshape(n, s, n, s).sum(axis=(1, 3)) / np.maximum(cnt, 1)
                cov = cnt / (s * s)
            out.append((np.clip(rgb, 0, 255).astype(np.uint8), cov, dep))
        return out


#: optional global contrast normalisation (per channel, the object's 2nd-98th percentile range over the fully covered
#: pixels of both renders stretched to 0-255, gain <= 255 / min_range). Off by default (``NORM_MIN_RANGE = None``):
#: tested with a gain cap of 4 and 2 it turned reviewer-preserved dark plain objects (a dark-grey 12-gon beaker with a
#: baked shading gradient, a mottled apple, a wooden bat, a jar of speckled seeds) into "broken" because it amplifies
#: smooth shading and stochastic texture exactly like a print. Low-contrast prints are caught by the structural blob
#: test below instead.
NORM_MIN_RANGE = None
#: perceptual blur before differencing (Gaussian sigma in millimetres at the render scale): a person at 0.5-1.5 m does
#: not resolve pixel-scale speckle, wood grain or mottle, only features of a few millimetres (1.5 mm smeared the 1 cm
#: letters of a low-contrast wrapping print below the blob threshold; 1.0 mm keeps them)
BLUR_MM = 1.0
BLUR_PX_RANGE = (0.6, 4.0)
#: local-feature (blob) test on D = 3x3 median of the channel-max |diff| of the blurred renders: a pixel is "hot" when
#: D > BLOB_T (a high-contrast feature: logo, glyph, label edge) or when the local standard deviation of D over a
#: BLOB_STD_MM window exceeds BLOB_STD_T (a low-contrast but structured feature: dark-on-dark text, a faint print -
#: a smooth baked-shading gradient has D of the same size but no local structure). Connected hot blobs are measured
#: in pixels; BLOB_MM is the feature size a blob must reach to break the symmetry (a person at 1 m sees a 6 x 6 mm
#: print), a blob of a quarter of that area (3 x 3 mm) already costs "preserved".
BLOB_T = 25.0 / 255.0
BLOB_STD_T = 6.0 / 255.0
BLOB_STD_MM = 5.0
BLOB_MM = 6.0
BLOB_MIN_PX = 9
#: morphological opening of the hot mask (structuring-element diameter in millimetres) before the blobs are measured:
#: removes structures thinner than ~2 mm - the baked shading edges between the facets of a low-poly body of revolution,
#: rim corners, grain streaks - while print strokes, logos and labels (>= 2-3 mm) survive. High-contrast pixels
#: (D > BLOB_T_STRONG: a black line drawing on a light face, a digit) are exempt from it up to a 1 mm opening, because
#: a thin high-contrast stroke is what a person reads first (the fish outline and the "6" of a letter block).
BLOB_OPEN_MM = 2.5
BLOB_T_STRONG = 60.0 / 255.0
BLOB_OPEN_STRONG_MM = 1.0
#: silhouette test: the XOR of the two coverage masks (a part of the object that is present in one render and absent
#: in the other - a pencil tip swapped with the flat end, a pump nozzle, a wick, a swing-top bail) opened with a disc
#: of SIL_OPEN_STRONG_MM (a shape difference at least that thick) resp. SIL_OPEN_WEAK_MM; the largest remaining blob
#: is compared with n6 like the texture blob (strong >= n6 -> broken, weak >= n6 -> borderline). Sub-pixel silhouette
#: shifts of an exact symmetry and the corner slivers of low-poly bodies are thinner than 3 mm and vanish (a 2 mm weak
#: opening still kept the 2.5 mm crescent of a slightly lopsided bar of soap). For thin bodies the openings shrink to
#: SIL_THIN_FRAC x the thinnest bbox extent (a pencil is 6-8 mm thick: its swapped tip is a wedge thinner than 4 mm).
SIL_OPEN_STRONG_MM = 4.0
SIL_OPEN_WEAK_MM = 3.0
SIL_THIN_FRAC = 0.4
SIL_OPEN_MIN_MM = 1.0


def normalise_pair(ia: np.ndarray, ib: np.ndarray, mask: np.ndarray, min_range: float = NORM_MIN_RANGE) -> Tuple[np.ndarray, np.ndarray, float]:
    """Contrast-normalise two renders with one common per-channel linear stretch (object 2nd-98th percentiles ->
    0-255, gain <= 255 / min_range) so that a dark bottle is judged like a bright one. Returns floats in 0-255 and the
    largest channel gain."""
    if not mask.any():
        return ia.astype(np.float64), ib.astype(np.float64), 1.0
    pix = np.concatenate([ia[mask], ib[mask]]).astype(np.float64)
    lo = np.percentile(pix, 2, axis=0)
    hi = np.percentile(pix, 98, axis=0)
    gain = 255.0 / np.maximum(hi - lo, min_range)
    na = np.clip((ia.astype(np.float64) - lo) * gain, 0, 255)
    nb = np.clip((ib.astype(np.float64) - lo) * gain, 0, 255)
    return na, nb, float(gain.max())


def _blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    from scipy import ndimage

    if sigma_px <= 0:
        return img.astype(np.float64)
    return ndimage.gaussian_filter(img.astype(np.float64), sigma=(sigma_px, sigma_px, 0), mode="nearest")


#: optional depth matching: pixels whose depth differs between the two renders by more than this fraction of the bbox
#: diagonal would be left out of the texture comparison (a polygon corner of a low-poly body, a lump). Off by default
#: (``None``): after a symmetry rotation the pixel legitimately shows another part of the object - a thin painting
#: flipped by ``y2`` presents its white back at a depth that differs by its thickness, and the mask hid the picture
#: (v2 trial: ``painting/fwuole`` y2 came out preserved). Polygon-corner artefacts are handled by the blur, the
#: opening and the graded blob rule instead (they land in ``borderline``, i.e. the review).
DEPTH_TOL_FRAC = None


def image_diff(a: Sequence[Tuple[np.ndarray, ...]], b: Sequence[Tuple[np.ndarray, ...]],
               m_per_px: Optional[float] = None, view_names: Optional[Sequence[str]] = None, diag: Optional[float] = None,
               blur_mm: float = BLUR_MM, min_range: Optional[float] = NORM_MIN_RANGE, blob_t: float = BLOB_T,
               std_t: float = BLOB_STD_T, std_mm: float = BLOB_STD_MM, open_mm: float = BLOB_OPEN_MM,
               depth_tol_frac: Optional[float] = DEPTH_TOL_FRAC, sil_strong_mm: float = SIL_OPEN_STRONG_MM,
               sil_weak_mm: float = SIL_OPEN_WEAK_MM, thin_mm: Optional[float] = None) -> Dict[str, float]:
    """Compare two render sets (same cameras; ``(rgb, coverage[, depth])`` per view): perceptual blur (``blur_mm``),
    optional contrast normalisation per view (``normalise_pair``, off when ``min_range`` is None), then the statistics
    below over the pixels fully covered by the object in both renders (optionally only those whose depth agrees within
    ``depth_tol_frac`` x ``diag``; off by default, see ``DEPTH_TOL_FRAC``).

    ``mad`` / ``p95`` / ``p99`` = mean, 95th and 99th percentile of the per-pixel channel-mean |diff| of the blurred
    renders (fractions of 255) over those pixels (partially covered silhouette pixels mix the background colour in and
    depend on the sub-pixel position of the edge, so they are excluded; for an exact geometric symmetry the two
    silhouettes coincide anyway); ``mad_raw`` = the same mean of the un-blurred renders; ``mad_union`` = raw mean over
    the union of the (coverage >= 1/2) masks - the union-mask criterion, which also counts silhouette mismatches;
    ``iou`` = mask IoU; ``depth_frac`` = share of the fully covered pixels excluded for a depth mismatch; ``gain`` = the
    largest normalisation gain applied (1 when off). Local-feature test on D = 3x3 median of the channel-max |diff|:
    hot pixels are D > ``blob_t`` or local-std(D, ``std_mm`` window) > ``std_t``, thinned by an opening of
    ``open_mm``; ``blob_px`` = largest connected hot blob (pixels), ``blob_frac`` = its share of that view's object
    pixels, ``blob_view`` = the camera it was seen from, ``blob_kind`` = ``abs`` / ``struct`` / ``both`` (which test
    lit its pixels), ``blob_d`` = mean D over the blob (/255 fraction), ``blob_mm`` = its side length in millimetres
    (``m_per_px`` = render scale), ``n6`` = the pixel area of a ``BLOB_MM`` x ``BLOB_MM`` feature. Silhouette test: the
    XOR of the coverage masks opened with ``sil_strong_mm`` / ``sil_weak_mm`` discs (shrunk for thin bodies, ``thin_mm``
    = thinnest bbox extent), largest blob ``sil_px`` / ``sil_px_weak``, ``sil_view``, ``sil_mm``."""
    from scipy import ndimage

    diffs, rdiffs, udiffs, inter, union, n_full, n_depth = [], [], [], 0, 0, 0, 0
    gain_max = 1.0
    blob = {"blob_px": 0, "blob_frac": 0.0, "blob_view": "", "blob_kind": "", "blob_d": 0.0}
    sil = {"sil_px": 0, "sil_px_weak": 0, "sil_view": ""}
    n6 = max(BLOB_MIN_PX, int(round((BLOB_MM / 1000.0 / m_per_px) ** 2))) if m_per_px else BLOB_MIN_PX
    sigma_px = min(BLUR_PX_RANGE[1], max(BLUR_PX_RANGE[0], blur_mm / 1000.0 / m_per_px)) if (m_per_px and blur_mm > 0) else 0.0
    win = int(min(25, max(3, round(std_mm / 1000.0 / m_per_px)))) if m_per_px else 5

    def disc(diam_mm: float):
        px = int(min(25, max(0, round(diam_mm / 1000.0 / m_per_px)))) if (m_per_px and diam_mm > 0) else 0
        if px < 2:
            return None
        yy, xx = np.mgrid[-px // 2: px // 2 + 1, -px // 2: px // 2 + 1]
        return (xx * xx + yy * yy) <= (px / 2.0) ** 2

    def largest(mask) -> int:
        if not mask.any():
            return 0
        lab, n = ndimage.label(mask)
        return int(ndimage.sum(mask, lab, index=np.arange(1, n + 1)).max())

    if thin_mm:  # thin bodies: a shape difference cannot be thicker than the body
        sil_strong_mm = max(SIL_OPEN_MIN_MM, min(sil_strong_mm, SIL_THIN_FRAC * thin_mm))
        sil_weak_mm = max(SIL_OPEN_MIN_MM, min(sil_weak_mm, SIL_THIN_FRAC * thin_mm))
    open_se, sil_strong_se, sil_weak_se, open_strong_se = disc(open_mm), disc(sil_strong_mm), disc(sil_weak_mm), disc(BLOB_OPEN_STRONG_MM)
    for vi, (ra, rb) in enumerate(zip(a, b)):
        ia, ca = ra[0], ra[1]
        ib, cb = rb[0], rb[1]
        ma, mb = ca >= 0.5, cb >= 0.5
        u = ma | mb
        if not u.any():
            continue
        full = (ca >= 1.0) & (cb >= 1.0)
        raw = np.abs(ia.astype(np.float64) - ib.astype(np.float64)).mean(axis=2) / 255.0
        udiffs.append(raw[u])
        inter += int((ma & mb).sum())
        union += int(u.sum())
        xor = ma ^ mb
        if xor.any():
            strong = largest(ndimage.binary_opening(xor, structure=sil_strong_se)) if sil_strong_se is not None else largest(xor)
            weak = largest(ndimage.binary_opening(xor, structure=sil_weak_se)) if sil_weak_se is not None else largest(xor)
            if strong > sil["sil_px"] or (strong == sil["sil_px"] and weak > sil["sil_px_weak"]):
                sil = {"sil_px": strong, "sil_px_weak": weak, "sil_view": view_names[vi] if view_names else str(vi)}
        if not full.any():
            continue
        n_full += int(full.sum())
        if depth_tol_frac and len(ra) > 2 and len(rb) > 2 and diag:
            same_depth = np.abs(ra[2] - rb[2]) <= depth_tol_frac * diag
            n_depth += int((full & ~same_depth).sum())
            full = full & same_depth
            if not full.any():
                continue
        # blur inside the object only: paint the background with the object's mean colour first so that the silhouette
        # does not bleed white into the object's border pixels
        fill = ia[full].mean(axis=0)
        ba, bb = ia.astype(np.float64).copy(), ib.astype(np.float64).copy()
        ba[~ma], bb[~mb] = fill, fill
        ba, bb = _blur(ba, sigma_px), _blur(bb, sigma_px)
        if min_range:
            ba, bb, g = normalise_pair(ba, bb, full, min_range)
            gain_max = max(gain_max, g)
        dn = np.abs(ba - bb)
        core = ndimage.binary_erosion(full, iterations=max(1, int(round(sigma_px)))) if full.sum() > 64 else full
        if not core.any():
            core = full
        diffs.append(dn.mean(axis=2)[core] / 255.0)
        rdiffs.append(raw[full])
        D = ndimage.median_filter(dn.max(axis=2) / 255.0, size=3)
        hot_abs = (D > blob_t) & core
        # structure: local standard deviation of D (window ~ std_mm); D outside the core is replaced by its core mean so
        # that the silhouette itself is not an edge, and the structural mask is kept off the outermost window/2 pixels
        Dm = np.where(core, D, float(D[core].mean()))
        m1 = ndimage.uniform_filter(Dm, size=win, mode="nearest")
        m2 = ndimage.uniform_filter(Dm * Dm, size=win, mode="nearest")
        lstd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
        inner = ndimage.binary_erosion(core, iterations=max(1, win // 2)) if core.sum() > 4 * win * win else core
        hot_struct = (lstd > std_t) & inner
        hot = hot_abs | hot_struct
        if open_se is not None and hot.any():
            hot = ndimage.binary_opening(hot, structure=open_se)
        hot_strong = (D > BLOB_T_STRONG) & core
        if hot_strong.any():
            if open_strong_se is not None:
                hot_strong = ndimage.binary_opening(hot_strong, structure=open_strong_se)
            hot = hot | hot_strong
        if hot.any():
            lab, n = ndimage.label(hot)
            sizes = ndimage.sum(hot, lab, index=np.arange(1, n + 1))
            k = int(np.argmax(sizes))
            if sizes[k] > blob["blob_px"]:
                part = lab == (k + 1)
                na_, ns_ = int((part & hot_abs).sum()), int((part & hot_struct).sum())
                blob = {"blob_px": int(sizes[k]), "blob_frac": float(sizes[k] / max(int(full.sum()), 1)),
                        "blob_view": view_names[vi] if view_names else str(vi),
                        "blob_kind": "both" if (na_ and ns_) else ("abs" if na_ else "struct"), "blob_d": float(D[part].mean())}
    mm = (m_per_px or 0.0) * 1000.0
    if not udiffs:
        return {"mad": 1.0, "p95": 1.0, "p99": 1.0, "mad_raw": 1.0, "mad_union": 1.0, "iou": 0.0, "depth_frac": 0.0, "gain": 1.0,
                "blob_px": 0, "blob_frac": 0.0, "blob_view": "", "blob_kind": "", "blob_d": 0.0, "blob_mm": 0.0, "n6": n6,
                "sil_px": 0, "sil_px_weak": 0, "sil_view": "", "sil_mm": 0.0}
    du = np.concatenate(udiffs)
    d = np.concatenate(diffs) if diffs else du
    dr = np.concatenate(rdiffs) if rdiffs else du
    return {"mad": float(d.mean()), "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
            "mad_raw": float(dr.mean()), "mad_union": float(du.mean()), "iou": inter / max(union, 1),
            "depth_frac": n_depth / max(n_full, 1), "gain": gain_max,
            **blob, "blob_mm": float(math.sqrt(blob["blob_px"]) * mm), "n6": n6,
            **sil, "sil_mm": float(math.sqrt(sil["sil_px"]) * mm)}


#: texture thresholds (fractions of 255) on the blurred, fully-covered, depth-matched pixel comparison (v2). The
#: percentile rules are those of v1 (``p99`` guards against small features that cover 1-5 % of the pixels; the
#: borderline band is closed by ``p95`` because a muted label moves the mean by only 7-12/255 while >= 5 % of the
#: pixels differ by > 25/255). The blob test adds what the percentiles miss: a connected patch of difference of
#: >= 4 x (6 x 6 mm) (``n6`` pixels at the render scale), or of >= 6 x 6 mm with a mean difference of >= 30/255,
#: breaks the symmetry outright (a logo, a glyph, a wrapping print, a label edge); a smaller / fainter patch of
#: >= 3 x 3 mm costs "preserved" (borderline -> review: facet shading of a low-poly body, a pale print, a seam).
TEX_PRESERVED = {"mad": 3.0 / 255.0, "p95": 20.0 / 255.0, "p99": 60.0 / 255.0, "blob_n6_frac": 0.25, "sil_weak_n6_frac": 1.0}
TEX_BORDERLINE = {"mad": 12.0 / 255.0, "p95": 25.0 / 255.0, "p99": 60.0 / 255.0, "blob_n6_frac": 4.0,
                  "blob_n6_frac_strong": 1.0, "blob_d_strong": 30.0 / 255.0, "sil_n6_frac": 1.0}
TEX_BORDERLINE_MAD = TEX_BORDERLINE["mad"]
#: the v1 thresholds (192 px, no blur, no blob test) - kept to reproduce the contact-sheet row order the first review
#: batches (rev_00..10) were issued with
TEX_PRESERVED_V1 = {"mad": 3.0 / 255.0, "p95": 20.0 / 255.0, "p99": 60.0 / 255.0}
TEX_BORDERLINE_V1 = {"mad": 12.0 / 255.0, "p95": 25.0 / 255.0, "p99": 60.0 / 255.0}


def classify_tex(score: Dict[str, float], preserved=TEX_PRESERVED, borderline=TEX_BORDERLINE) -> str:
    """``preserved`` / ``borderline`` / ``broken`` for the worst rotation score of a tag (v2: with the blob test; a
    v1 score without blob fields is classified on the percentiles alone)."""
    p99 = score.get("p99", score["p95"])
    n6 = max(score.get("n6", BLOB_MIN_PX), 1)
    blob_rel = score.get("blob_px", 0) / n6
    blob_d = score.get("blob_d", 0.0)
    sil_rel, sil_weak_rel = score.get("sil_px", 0) / n6, score.get("sil_px_weak", 0) / n6
    if (score["mad"] < preserved["mad"] and score["p95"] < preserved["p95"] and p99 < preserved["p99"]
            and blob_rel < preserved.get("blob_n6_frac", 1e9) and sil_weak_rel < preserved.get("sil_weak_n6_frac", 1e9)):
        return "preserved"
    strong = blob_rel >= borderline.get("blob_n6_frac_strong", 1e9) and blob_d >= borderline.get("blob_d_strong", 1e9)
    if (score["mad"] <= borderline["mad"] and score["p95"] <= borderline["p95"] and p99 <= borderline["p99"]
            and blob_rel < borderline.get("blob_n6_frac", 1e9) and not strong and sil_rel < borderline.get("sil_n6_frac", 1e9)):
        return "borderline"
    return "broken"


def classify_tex_v1(score: Dict[str, float]) -> str:
    """The first survey's texture class (before the contrast normalisation and the blob test)."""
    p99 = score.get("p99", score["p95"])
    if score["mad"] < TEX_PRESERVED_V1["mad"] and score["p95"] < TEX_PRESERVED_V1["p95"] and p99 < TEX_PRESERVED_V1["p99"]:
        return "preserved"
    if score["mad"] <= TEX_BORDERLINE_V1["mad"] and score["p95"] <= TEX_BORDERLINE_V1["p95"] and p99 <= TEX_BORDERLINE_V1["p99"]:
        return "borderline"
    return "broken"


def tex_severity(score: Dict[str, float]) -> Tuple[float, float]:
    """Ordering key for the worst rotation of a tag: the larger of the texture blob and the silhouette blob relative
    to n6 first, then the MAD."""
    n6 = max(score.get("n6", BLOB_MIN_PX), 1)
    return (max(score.get("blob_px", 0), score.get("sil_px", 0)) / n6, score["mad"])
