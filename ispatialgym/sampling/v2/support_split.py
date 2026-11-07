"""The SUPPORT SPLIT of the v2 sampler (i7-split 2026-11-07, the owner's rule "supports with protrusions").

Wherever the sampler ignores a support (the grasp chain's last pose, the approach tail, places, lifts, releases, the program check;
``IKChecker.ignoring``) it used to switch EVERY collision piece of the object off - the tray's rims and walls, the bookcase's walls and the
board above, the bed's rails and headboard went with the surface the object rests on. B1K collision meshes are convex pieces, one cuRobo
mesh entry each (``R1ProCuRobo._mesh_enable`` by prim path), so the support is split by height around its key plane instead:

* the pieces whose top lies within ``tol`` of the plane (or below it) are the SUPPORT - switched off as before and policed by the
  existing bounded finger rule (``ispatialgym.grasping.contact_policy``, unchanged);
* every piece whose top lies above ``plane + tol`` stays a NORMAL collision obstacle;
* a piece above ``plane + tol`` whose xy hull covers raster cells of the plane (a sloped piece reaching down to the surface) is a
  STRADDLER: its part above ``plane + tol`` is added to the cuRobo world as its own mesh while the piece is switched off
  (``trimesh.slice_plane`` + ``checker.data.meshes.add``; :data:`CLIP_SWITCH`), else the whole piece goes off as before (counted).

The plane is the one the sampler registered (:func:`register` / :func:`register_top`, next to ``IKChecker.support_plane_z``: a map hit's
measured top, a tray floor, an interior board, the floor), in the object's root frame (a tray moves with the focus). The tolerance per
support (:func:`tolerance_from_heights`) = clamp(q95 of |cell height - plane| + 2 mm, 3 mm, 10 mm) over the top raster's cells (cells
more than 10 mm above the plane are protrusions under any tolerance and do not count as spread); a support whose top spreads more than
10 mm, one without a raster, and every object without a registered plane (walls, handle objects) keep the whole-object ignore.

``ISPATIALGYM_SUPPORT_SPLIT=0``: the whole-object ignore everywhere (the former behaviour; registrations are kept, nothing reads them).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("ispatialgym.sampling.v2")

SWITCH = "ISPATIALGYM_SUPPORT_SPLIT"
#: straddling pieces: their part above ``plane + tol`` added to the world as its own mesh ("0": the whole piece off, as before)
CLIP_SWITCH = "ISPATIALGYM_SUPPORT_SPLIT_CLIP"
MIN_TOL_M = 0.003
MAX_TOL_M = 0.010
TOL_MARGIN_M = 0.002
TOL_QUANTILE = 0.95
#: fewer raster cells than this within the cap: no tolerance (the whole ignore)
MIN_CELLS = 5
#: a plane cell counts as covered by a piece's xy hull when it lies at least this far inside it (cells at a rim's foot stay out)
STRADDLE_INSET_M = 0.002
#: a support tilted more than this (deg) keeps the whole ignore (its plane is not horizontal in its root frame)
MAX_TILT_DEG = 2.0
#: version of the split rule in the cache keys (:func:`cache_identity`)
VERSION = 1
#: name suffix of a straddler's clipped upper part in the cuRobo mesh store
CLIP_SUFFIX = "__support_split_above"


def split_on() -> bool:
    """``ISPATIALGYM_SUPPORT_SPLIT`` (default on): split registered supports; "0" / off / false = the whole-object ignore."""
    return os.environ.get(SWITCH, "1").strip().lower() not in ("0", "off", "false", "no")


def clip_on() -> bool:
    return os.environ.get(CLIP_SWITCH, "1").strip().lower() not in ("0", "off", "false", "no")


def _count(ikc: Any, key: str, n: int = 1) -> None:
    fn = getattr(ikc, "_count", None)
    if callable(fn):
        fn(key, n)


# ----------------------------------------------------------------------------------------------------------------
# tolerance
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Tolerance:
    """A support's split tolerance: ``tol`` (m; None = the whole-object ignore), the spread it was derived from (``spread`` = q95 of
    |cell - plane| over the cells at most :data:`MAX_TOL_M` above the plane, ``spread_all`` = over every cell, logged), the number of
    cells and ``why`` the tolerance is None (``nonflat`` | ``no_raster`` | ``few_cells`` | ``no_object`` | ...)."""

    tol: Optional[float]
    spread: float = float("nan")
    spread_all: float = float("nan")
    n: int = 0
    why: str = ""

    @staticmethod
    def none(why: str) -> "Tolerance":
        return Tolerance(None, why=why)

    @staticmethod
    def fixed(tol: float, why: str) -> "Tolerance":
        """A tolerance without a raster (the floor): ``tol`` clamped to [MIN, MAX]; ``why`` says where it comes from."""
        return Tolerance(min(MAX_TOL_M, max(MIN_TOL_M, float(tol))), why=why)

    def text(self) -> str:
        if self.tol is None:
            extra = f", spread q95 {1000 * self.spread:.1f} mm (all cells {1000 * self.spread_all:.1f} mm) over {self.n} cells" if self.n else ""
            return f"whole ignore ({self.why}{extra})"
        src = f"spread q95 {1000 * self.spread:.1f} mm (all cells {1000 * self.spread_all:.1f} mm) over {self.n} cells" if self.n else self.why
        return f"tol {1000 * self.tol:.1f} mm ({src})"


def tolerance_from_heights(zs: Any, plane: float) -> Tolerance:
    """The owner's tolerance from a raster's cell hit heights ``zs`` (any shape, ``nan`` = no hit) about ``plane``: clamp(q95 of
    |cell - plane| + :data:`TOL_MARGIN_M`, :data:`MIN_TOL_M`, :data:`MAX_TOL_M`), the q95 over the cells at most :data:`MAX_TOL_M`
    above the plane (a rail / rim / lip cell higher than that is a protrusion piece under any tolerance, not spread of the top); a
    spread above :data:`MAX_TOL_M` = ``nonflat`` (the whole ignore)."""
    z = np.asarray(zs if not hasattr(zs, "detach") else zs.detach().cpu().numpy(), dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return Tolerance.none("no_raster")
    d = z - float(plane)
    spread_all = float(np.quantile(np.abs(d), TOL_QUANTILE))
    dk = np.abs(d[d <= MAX_TOL_M + 1e-9])
    if dk.size < MIN_CELLS:
        return Tolerance(None, float("nan"), spread_all, int(dk.size), "few_cells")
    spread = float(np.quantile(dk, TOL_QUANTILE))
    if spread > MAX_TOL_M:
        return Tolerance(None, spread, spread_all, int(dk.size), "nonflat")
    return Tolerance(min(MAX_TOL_M, max(MIN_TOL_M, spread + TOL_MARGIN_M)), spread, spread_all, int(dk.size), "")


def _own(top: Any) -> Any:
    return getattr(top, "own", None) or top  # a neighbours' union top (``neighbors.UnionTop``): the support's own raster


def top_heights(top: Any) -> Optional[np.ndarray]:
    """The hit heights of a top raster's cells (``SupportTop.zs`` under its ``mask``), None without them."""
    own = _own(top)
    zs = getattr(own, "zs", None)
    if zs is None:
        return None
    z = np.asarray(zs.detach().cpu().numpy() if hasattr(zs, "detach") else zs, dtype=np.float64)
    mask = getattr(own, "mask", None)
    if mask is not None:
        m = np.asarray(mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask, dtype=bool)
        if m.shape == z.shape:
            z = z[m]
    return z.ravel()


def tolerance_from_top(top: Any, plane: Optional[float] = None) -> Tolerance:
    """:func:`tolerance_from_heights` of a ``SupportTop`` / ``SupportInterior`` (its ``zs`` under ``mask``) about ``plane`` (default its
    ``z``); ``no_raster`` for a raster without heights."""
    z = top_heights(top)
    if z is None:
        return Tolerance.none("no_raster")
    return tolerance_from_heights(z, float(_own(top).z) if plane is None else float(plane))


def plane_cells_world(top: Any, plane: float, tol: float) -> np.ndarray:
    """(K, 2) world xy of the raster cells whose hit lies within ``tol`` of ``plane`` (the support surface itself)."""
    own = _own(top)
    zs = getattr(own, "zs", None)
    if zs is None:
        return np.zeros((0, 2))
    z = np.asarray(zs.detach().cpu().numpy() if hasattr(zs, "detach") else zs, dtype=np.float64)
    xs = np.asarray(own.xs.detach().cpu().numpy() if hasattr(own.xs, "detach") else own.xs, dtype=np.float64)
    ys = np.asarray(own.ys.detach().cpu().numpy() if hasattr(own.ys, "detach") else own.ys, dtype=np.float64)
    ok = np.isfinite(z) & (np.abs(z - float(plane)) <= float(tol) + 1e-9)
    ii, jj = np.nonzero(ok)
    return np.stack([xs[ii], ys[jj]], axis=1) if len(ii) else np.zeros((0, 2))


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers (numpy; xyzw quaternions)
# ----------------------------------------------------------------------------------------------------------------


def _np(x: Any) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64)


def quat_mat(q: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in _np(q).ravel()[:4])
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _root(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(p, R)`` of the object's root in the world, None when it cannot be read."""
    fn = getattr(obj, "get_position_orientation", None)
    if not callable(fn):
        return None
    try:
        p, q = fn()
    except Exception:  # noqa: BLE001
        return None
    return _np(p).ravel()[:3], quat_mat(q)


def to_local(root: Tuple[np.ndarray, np.ndarray], pts: np.ndarray) -> np.ndarray:
    p, R = root
    return (np.asarray(pts, dtype=np.float64) - p) @ R


def to_world(root: Tuple[np.ndarray, np.ndarray], pts: np.ndarray) -> np.ndarray:
    p, R = root
    return np.asarray(pts, dtype=np.float64) @ R.T + p


def hull2d(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of (N, 2) points; fewer than 3 distinct points give what there is."""
    P = np.unique(np.round(np.asarray(pts, dtype=np.float64)[:, :2], 9), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def inside_hull(hull: np.ndarray, pts: np.ndarray, inset: float = 0.0) -> np.ndarray:
    """(K,) bool: the points lie inside the CCW convex polygon ``hull`` at least ``inset`` from every edge."""
    q = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(hull) < 3 or len(q) == 0:
        return np.zeros(len(q), dtype=bool)
    a = hull
    b = np.roll(hull, -1, axis=0)
    e = b - a
    ln = np.linalg.norm(e, axis=1)
    keep = ln > 1e-12
    a, e, ln = a[keep], e[keep], ln[keep]
    # signed distance to the left of each edge (inside for a CCW polygon)
    d = (e[None, :, 0] * (q[:, None, 1] - a[None, :, 1]) - e[None, :, 1] * (q[:, None, 0] - a[None, :, 0])) / ln[None, :]
    return (d >= float(inset)).all(axis=1)


# ----------------------------------------------------------------------------------------------------------------
# piece classification
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """The pieces of one support split at ``dz`` (the plane's height above the object's root origin) with ``tol``: ``support`` (top
    at most ``dz + tol``: switched off), ``protrusion`` (above: obstacles), ``straddling`` (above, and their xy hull covers plane
    cells); piece ids as given (relative prim paths). ``tops_mm`` = each protrusion's / straddler's top above the plane; ``hulls`` =
    their (top above the plane m, (K, 2) xy hull in the object frame)."""

    support: Tuple[str, ...]
    protrusion: Tuple[str, ...]
    straddling: Tuple[str, ...]
    tops_mm: Dict[str, float] = field(default_factory=dict)
    hulls: Tuple[Tuple[str, float, np.ndarray], ...] = ()
    digest: str = ""
    #: support pieces whose top lies AT the plane (within [plane - MAX_TOL_M, plane + tol]): none = the plane does not match the object's
    #: geometry (a floor datum below the floor mesh, a top piece above its raster's median) - :func:`plan_for` keeps the whole ignore
    n_at_plane: int = 0

    @property
    def n_pieces(self) -> int:
        return len(self.support) + len(self.protrusion) + len(self.straddling)


def classify(pieces: Sequence[Tuple[str, np.ndarray]], dz: float, tol: float, cells: Optional[np.ndarray] = None,
             inset: float = STRADDLE_INSET_M) -> Classification:
    """Split ``pieces`` = [(id, (N, 3) points in the object frame, z up)] at the plane ``dz`` with ``tol`` (:class:`Classification`).
    ``cells`` = (K, 2) object-frame xy of the plane's raster cells (the straddler test; None = no test)."""
    lim = float(dz) + float(tol)
    n_at = 0
    sup: List[str] = []
    pro: List[str] = []
    stra: List[str] = []
    tops: Dict[str, float] = {}
    hulls: List[Tuple[str, float, np.ndarray]] = []
    cl = None if cells is None else np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    for pid, pts in pieces:
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if P.size == 0:
            continue
        top = float(P[:, 2].max())
        if top <= lim + 1e-9:
            sup.append(pid)
            n_at += int(top >= float(dz) - MAX_TOL_M)
            continue
        h = hull2d(P[:, :2])
        tops[pid] = round(1000.0 * (top - float(dz)), 1)
        hulls.append((pid, top - float(dz), h))
        if cl is not None and len(cl) and float(P[:, 2].min()) < lim and bool(inside_hull(h, cl, inset).any()):
            stra.append(pid)
        else:
            pro.append(pid)
    dig = hashlib.sha1(repr((sorted(sup), sorted(pro), sorted(stra))).encode()).hexdigest()[:12]
    return Classification(tuple(sup), tuple(pro), tuple(stra), tops, tuple(hulls), dig, n_at)


# ----------------------------------------------------------------------------------------------------------------
# registration (next to ``IKChecker.support_plane_z``)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Registration:
    """A support the split knows: the world ``plane_z`` it was registered at, ``dz`` = that plane's height above the object's root
    origin (object frame, upright), the :class:`Tolerance`, the plane cells in the object frame (straddler test) and ``what`` plane
    it is (``top`` | ``tray_floor`` | ``interior`` | ``floor`` | ...)."""

    name: str
    plane_z: float
    dz: Optional[float]
    tolerance: Tolerance
    cells: Optional[np.ndarray] = None
    what: str = "top"

    @property
    def tol(self) -> Optional[float]:
        return self.tolerance.tol


def _reg(ikc: Any) -> Dict[str, Registration]:
    d = getattr(ikc, "_support_split_reg", None)
    if not isinstance(d, dict):
        d = {}
        try:
            ikc._support_split_reg = d
        except Exception:  # noqa: BLE001
            pass
    return d


def registration(ikc: Any, name: Optional[str]) -> Optional[Registration]:
    return _reg(ikc).get(name) if name else None


def register(ikc: Any, obj: Any, plane_z: float, tolerance: Tolerance, *, dz: Optional[float] = None, cells_local: Optional[np.ndarray] = None,
             cells_world: Optional[np.ndarray] = None, what: str = "top", name: Optional[str] = None) -> Optional[Registration]:
    """Register ``obj``'s split plane: ``ikc.support_split_z[name]`` = the world plane, ``ikc.support_plane_tol[name]`` = the tolerance
    (None = the whole ignore), the plane's height above the root origin (``dz``; default from the object's current root z) and the plane
    cells (object frame, or ``cells_world`` converted at the current pose). One INFO line. None without an IK checker / name."""
    if ikc is None:
        return None
    name = name or getattr(obj, "name", None)
    if not name:
        return None
    root = _root(obj) if obj is not None else None
    if dz is None and root is not None and math.degrees(math.acos(max(-1.0, min(1.0, float(root[1][2, 2]))))) > MAX_TILT_DEG:
        tolerance = Tolerance.none("tilted")  # a world plane of a tilted object is not horizontal in its root frame (``plan_for`` re-checks)
    if dz is None:
        if root is None:
            tolerance = Tolerance.none("no_object")
        else:
            dz = float(plane_z) - float(root[0][2])
    cells = None
    if cells_local is not None:
        cells = _np(cells_local).reshape(-1, _np(cells_local).shape[-1])[:, :2]
    elif cells_world is not None and root is not None and len(cells_world):
        cw = _np(cells_world).reshape(-1, 2)
        cells = to_local(root, np.concatenate([cw, np.full((len(cw), 1), float(plane_z))], axis=1))[:, :2]
    rec = Registration(str(name), float(plane_z), None if dz is None else float(dz), tolerance, cells, what)
    prev = _reg(ikc).get(rec.name)
    if prev is not None and (round(prev.plane_z, 4), prev.dz is None or round(prev.dz, 4), prev.tol, prev.what, _cells_digest(prev.cells)) == \
            (round(rec.plane_z, 4), rec.dz is None or round(rec.dz, 4), rec.tol, rec.what, _cells_digest(rec.cells)):
        return prev  # the same registration again (a band lookup per edge / arm): kept, no second line
    _reg(ikc)[rec.name] = rec
    for attr, val in (("support_split_z", float(plane_z)), ("support_plane_tol", tolerance.tol)):
        d = getattr(ikc, attr, None)
        if not isinstance(d, dict):
            d = {}
            try:
                setattr(ikc, attr, d)
            except Exception:  # noqa: BLE001
                continue
        d[rec.name] = val
    log.info("SUPPORT_SPLIT register %s (%s plane z %.4f%s): %s", rec.name, what, float(plane_z),
             "" if rec.dz is None else f", {rec.dz:+.4f} m above its root", tolerance.text())
    if tolerance.tol is None:
        _count(ikc, f"support_split.registered_whole_{tolerance.why or 'none'}")
    return rec


def register_top(s: Any, obj: Any, top: Any, plane: Optional[float] = None, what: str = "top") -> Optional[Registration]:
    """:func:`register` from a top raster (``SupportTop`` / ``SupportInterior``): the tolerance from its cell heights about ``plane``
    (default ``top.z``), the plane cells for the straddler test. ``s`` = a sampler (its ``ikc`` and ``world``) or an IK checker;
    ``obj`` None = the world object of ``top.name``."""
    ikc = getattr(s, "ikc", None) if not hasattr(s, "support_policy_eval") else s
    if ikc is None or top is None:
        return None
    own = _own(top)
    if obj is None:
        world = getattr(s, "world", None)
        try:
            obj = world.obj(own.name) if world is not None else None
        except Exception:  # noqa: BLE001
            obj = None
    z = float(own.z) if plane is None else float(plane)
    tol = tolerance_from_top(own, z)
    cells = plane_cells_world(own, z, tol.tol) if tol.tol is not None else None
    return register(ikc, obj, z, tol, cells_world=cells, what=what, name=getattr(obj, "name", None) or getattr(own, "name", None))


def registered_names(ikc: Any) -> set:
    """The supports the sampler registered (split planes and the policy's ``support_plane_z``) while the split is on, else empty - the
    objects an approach does not count as 'approached' (``IKChecker.approached_objects``)."""
    if ikc is None or not split_on():
        return set()
    out = set(_reg(ikc))
    pz = getattr(ikc, "support_plane_z", None)
    if isinstance(pz, dict):
        out |= set(pz)
    return out


def drop_registered(ikc: Any, objs: Iterable[Any]) -> List[Any]:
    """``objs`` without the registered supports (:func:`registered_names`)."""
    names = registered_names(ikc)
    objs = list(objs)
    if not names:
        return objs
    return [o for o in objs if getattr(o, "name", None) not in names]


# ----------------------------------------------------------------------------------------------------------------
# plans: what one ignore context switches off
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Clip:
    """A straddler's part above the plane + tol: ``name`` in the mesh store, its source piece ``prim``, the (scaled) vertices / faces in
    the piece's own frame."""

    name: str
    prim: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class Plan:
    """What ignoring a registered support ``name`` switches off (``off``: full prim paths) and keeps (``kept`` = protrusion pieces +
    clipped straddlers left in the world), the straddlers' clipped parts to switch on (``clips`` with the source cm objects), the
    :class:`Classification`, and the object's root at classification (the draw's protrusion count)."""

    name: str
    off: frozenset
    kept: int
    clips: List[Tuple[Clip, Any]]
    cls: Classification
    reg: Registration
    root: Tuple[np.ndarray, np.ndarray]
    prefix: str = ""


#: (model, scale, dz mm, tol 0.1 mm, cells digest) -> Classification of relative piece ids (the same model / scale / plane anywhere)
_CLASS_CACHE: Dict[Tuple, Classification] = {}
#: (classification key, relative piece id) -> Clip geometry in the piece frame (None: the clip failed)
_CLIP_CACHE: Dict[Tuple, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
#: per backend: clip mesh names currently switched on by active contexts -> number of contexts
_ACTIVE_CLIPS: Dict[int, Dict[str, int]] = {}


def _pieces(obj: Any) -> List[Tuple[str, Any]]:
    out = []
    for link in getattr(obj, "links", {}).values():
        for cm in getattr(link, "collision_meshes", {}).values():
            out.append((str(cm.prim_path), cm))
    return out


def piece_geometry(backend: Any, cm: Any) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """``(scaled vertices in the piece frame, faces, (p, R) piece pose in the world)`` - the backend's cached mesh geometry
    (``R1ProCuRobo._mesh_geometry``) and the piece prim's pose (what the cuRobo world sync reads)."""
    verts, faces, scale = backend._mesh_geometry(cm)
    v = np.asarray(verts, dtype=np.float64) * np.asarray(scale, dtype=np.float64).reshape(1, -1)
    p, q = cm.get_position_orientation()
    return v, np.asarray(faces), (_np(p).ravel()[:3], quat_mat(q))


def _key_scale(obj: Any) -> Tuple:
    sc = getattr(obj, "scale", None)
    try:
        return tuple(round(float(v), 4) for v in _np(sc).ravel())
    except Exception:  # noqa: BLE001
        return ()


def _cells_digest(cells: Optional[np.ndarray]) -> str:
    if cells is None or len(cells) == 0:
        return ""
    return hashlib.sha1(np.round(np.asarray(cells, dtype=np.float64), 3).tobytes()).hexdigest()[:12]


def _clip_geometry(v: np.ndarray, faces: np.ndarray, pose_in_root: Tuple[np.ndarray, np.ndarray], lim: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The part of a piece (vertices ``v`` / ``faces`` in its frame, the frame's pose in the object root frame) above the object-frame plane
    z = ``lim``, in the piece's frame: ``trimesh.slice_plane`` (capped), else the convex hull of the vertices above and the edge
    crossings (the pieces are convex); None when nothing is left."""
    p, R = pose_in_root
    # the plane z_obj = lim in the piece frame: a point and the normal (R^T e_z)
    n = R.T @ np.array([0.0, 0.0, 1.0])
    o = R.T @ (np.array([0.0, 0.0, float(lim)]) - p)
    try:
        import trimesh

        m = trimesh.Trimesh(vertices=v, faces=np.asarray(faces).reshape(-1, 3), process=False)
        cut = m.slice_plane(o, n, cap=True)
        if cut is not None and len(cut.vertices) >= 4 and len(cut.faces) >= 4:
            return np.asarray(cut.vertices, dtype=np.float32), np.asarray(cut.faces, dtype=np.int64)
    except Exception as e:  # noqa: BLE001 - the convex construction below
        log.debug("support split: trimesh slice failed (%r); convex clip", e)
    s = (v - o) @ n
    above = v[s >= 0.0]
    cross = []
    f = np.asarray(faces).reshape(-1, 3)
    for a, b in np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]):
        sa, sb = s[a], s[b]
        if (sa < 0.0) != (sb < 0.0):
            t = sa / (sa - sb)
            cross.append(v[a] + t * (v[b] - v[a]))
    pts = np.concatenate([above, np.asarray(cross).reshape(-1, 3)]) if len(cross) else above
    if len(pts) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull

        h = ConvexHull(pts)
        return pts.astype(np.float32), np.asarray(h.simplices, dtype=np.int64)
    except Exception:  # noqa: BLE001 - degenerate (a sliver): nothing to keep
        return None


def plan_for(ikc: Any, obj: Any, count: bool = True) -> Optional[Plan]:
    """The split of ``obj`` for one ignore context (:class:`Plan`), None = the whole-object ignore (split off, not registered, no
    tolerance - counted ``support_split.whole_ignore_<why>`` -, no geometry, tilted). Counters: ``support_split.applied`` (a context left
    a protrusion / clipped part in the world), ``support_split.protrusion_pieces`` / ``.straddling_pieces`` (per support classified, with
    one INFO line)."""
    if not split_on() or obj is None:
        return None
    reg = registration(ikc, getattr(obj, "name", None))
    if reg is None:
        return None
    if reg.tol is None or reg.dz is None:
        if count:
            _count(ikc, f"support_split.whole_ignore_{reg.tolerance.why or 'none'}")
        return None
    backend = getattr(getattr(ikc, "kit", None), "backend", None)
    if backend is None or not hasattr(backend, "_mesh_geometry"):
        if count:
            _count(ikc, "support_split.whole_ignore_no_geometry")
        return None
    root = _root(obj)
    if root is None:
        return None
    up = root[1][:, 2]
    if math.degrees(math.acos(max(-1.0, min(1.0, float(up[2]))))) > MAX_TILT_DEG:
        if count:
            _count(ikc, "support_split.whole_ignore_tilted")
        return None
    # the last plan per support while it stays where it is (a tray moves with the focus; ``ignoring`` runs thousands of times per attempt)
    plans = getattr(ikc, "_support_split_plans", None)
    if not isinstance(plans, dict):
        plans = {}
        try:
            ikc._support_split_plans = plans
        except Exception:  # noqa: BLE001
            pass
    pk = (tuple(np.round(root[0], 4).tolist()), tuple(np.round(root[1], 4).ravel().tolist()), clip_on())
    last = plans.get(reg.name)
    if last is not None and last[0] == pk and last[1] is reg:
        if count and last[2].kept:
            _count(ikc, "support_split.applied")
        return last[2]
    prefix = str(getattr(obj, "prim_path", "") or "")
    rel = (lambda path: path[len(prefix):] if prefix and path.startswith(prefix) else path)  # noqa: E731
    pieces = _pieces(obj)
    model = getattr(obj, "model", None) or getattr(obj, "name", None)
    key = (str(model), _key_scale(obj), round(float(reg.dz), 3), round(float(reg.tol), 4), _cells_digest(reg.cells), len(pieces))
    cls = _CLASS_CACHE.get(key)
    geo: Dict[str, Any] = {}
    if cls is None:
        pts = []
        try:
            for path, cm in pieces:
                v, f, (pp, Rp) = piece_geometry(backend, cm)
                geo[path] = (v, f, (pp, Rp))
                pts.append((rel(path), to_local(root, v @ Rp.T + pp)))
        except Exception as e:  # noqa: BLE001 - a piece without geometry: the whole ignore
            log.info("SUPPORT_SPLIT %s: piece geometry unavailable (%r): whole ignore", reg.name, e)
            if count:
                _count(ikc, "support_split.whole_ignore_no_geometry")
            return None
        cls = _CLASS_CACHE[key] = classify(pts, reg.dz, reg.tol, reg.cells)
    if cls.n_at_plane == 0 and (cls.protrusion or cls.straddling):
        # no piece of the object reaches the plane: the registered plane is not this object's surface (a floor datum below the floor
        # mesh, a top piece above its raster) - splitting would keep the support itself in the world: the whole ignore, as before
        lk0 = (reg.name, key, "no_plane_piece")
        logged = getattr(ikc, "_support_split_logged", None)
        if not isinstance(logged, set) or lk0 not in logged:
            log.info("SUPPORT_SPLIT %s (%s plane z %.4f, %s): no piece's top at the plane (%d pieces, tops +%s mm above it): whole ignore",
                     reg.name, reg.what, reg.plane_z, reg.tolerance.text(), cls.n_pieces, sorted(cls.tops_mm.values())[:6])
            if isinstance(logged, set):
                logged.add(lk0)
        if count:
            _count(ikc, "support_split.whole_ignore_no_plane_piece")
        return None
    full = {rel(path): path for path, _ in pieces}
    off = {full[r] for r in cls.support if r in full}
    clips: List[Tuple[Clip, Any]] = []
    cm_of = dict(pieces)
    for r in cls.straddling:
        path = full.get(r)
        if path is None:
            continue
        ck = (key, r)
        g = _CLIP_CACHE.get(ck, False) if clip_on() else None
        if g is False:
            try:
                v, f, (pp, Rp) = geo.get(path) or piece_geometry(backend, cm_of[path])
                pose_in_root = ((pp - root[0]) @ root[1], root[1].T @ Rp)
                g = _clip_geometry(v, f, pose_in_root, float(reg.dz) + float(reg.tol))
            except Exception as e:  # noqa: BLE001
                log.info("SUPPORT_SPLIT %s: clip of %s failed (%r): the piece goes off whole", reg.name, r, e)
                g = None
            _CLIP_CACHE[ck] = g
        off.add(path)  # the straddler itself always goes off; its upper part (if any) comes back as its own mesh
        if g is not None:
            clips.append((Clip(path + CLIP_SUFFIX, path, g[0], g[1]), cm_of[path]))
    plan = Plan(reg.name, frozenset(off), len(cls.protrusion) + len(clips), clips, cls, reg, root, prefix)
    plans[reg.name] = (pk, reg, plan)
    st = getattr(ikc, "_support_split_logged", None)
    if not isinstance(st, set):
        st = set()
        try:
            ikc._support_split_logged = st
        except Exception:  # noqa: BLE001
            pass
    lk = (reg.name, key)
    if lk not in st:
        st.add(lk)
        _count(ikc, "support_split.classified")
        _count(ikc, "support_split.protrusion_pieces", len(cls.protrusion))
        if cls.straddling:
            _count(ikc, "support_split.straddling_pieces", len(cls.straddling))
            _count(ikc, "support_split.clipped" if clips else "support_split.clip_off", len(clips) if clips else len(cls.straddling))
        tops = sorted(cls.tops_mm.values())
        log.info("SUPPORT_SPLIT %s (%s plane z %.4f, %.4f m above its root, %s): %d pieces -> %d support (off), %d protrusion (obstacles%s), "
                 "%d straddling (%s)", reg.name, reg.what, reg.plane_z, reg.dz, reg.tolerance.text(), cls.n_pieces, len(cls.support), len(cls.protrusion),
                 f", tops +{tops[0]:.0f}..+{tops[-1]:.0f} mm" if tops else "", len(cls.straddling),
                 "clipped at the plane + tol" if clips else ("off whole" if cls.straddling else "-"))
    if count and plan.kept:
        _count(ikc, "support_split.applied")
    return plan


def cache_identity(ikc: Any, objs: Iterable[Any]) -> Dict[str, Any]:
    """The split's part of a cache key (exact IK, feasible sets): per ignored support whose split leaves pieces in the world, its name,
    plane (mm), tolerance (0.1 mm) and piece-set digest; {} with the split off or nothing split (the key as before)."""
    if not split_on():
        return {}
    ids = []
    for o in objs or ():
        try:
            p = plan_for(ikc, o, count=False)
        except Exception:  # noqa: BLE001
            p = None
        if p is not None and p.kept:
            ids.append([p.name, round(1000 * p.reg.plane_z, 1), round(10000 * float(p.reg.tol)), p.cls.digest, len(p.clips)])
    return {"support_split": [VERSION, sorted(ids)]} if ids else {}


# ----------------------------------------------------------------------------------------------------------------
# straddlers' clipped parts in the cuRobo mesh store
# ----------------------------------------------------------------------------------------------------------------


def enable_clips(backend: Any, plan: Plan) -> List[str]:
    """Switch the clipped upper parts of ``plan``'s straddlers on in the cuRobo world (added to the mesh store on first use, posed at the
    source piece's current root-relative pose; refcounted per backend). Returns the names switched on (for :func:`disable_clips`); a
    failure leaves that part out (the piece stays off: the former behaviour) and logs it."""
    out: List[str] = []
    if not plan.clips:
        return out
    active = _ACTIVE_CLIPS.setdefault(id(backend), {})
    known = backend.__dict__.setdefault("_support_split_clips", set()) if hasattr(backend, "__dict__") else set()
    in_store = getattr(backend, "_world_mesh_names", set())
    disabled = getattr(backend, "_disabled_obstacles", set())
    for clip, cm in plan.clips:
        if clip.prim not in in_store or clip.prim in disabled:
            continue  # the piece is not a mesh of the store (an sdf-grid layer) or out of the world for another reason: nothing to stand in for
        try:
            if active.get(clip.name, 0) == 0:
                from curobo.types import Pose as CuPose

                rp, rq = backend.root_pose()
                Rr = quat_mat(rq)
                p, q = cm.get_position_orientation()
                Rw = quat_mat(q)
                pos = (_np(p).ravel()[:3] - _np(rp).ravel()[:3]) @ Rr
                R = Rr.T @ Rw
                qw = _mat_quat_wxyz(R)
                dev = getattr(getattr(backend, "device_cfg", None), "device", "cpu")
                import torch as th

                if clip.name not in known:
                    from curobo.scene import Mesh

                    backend.checker.data.meshes.add(Mesh(name=clip.name, pose=list(pos) + list(qw), vertices=clip.vertices, faces=clip.faces,
                                                         scale=[1.0, 1.0, 1.0]))
                    known.add(clip.name)
                    if hasattr(backend, "_patch_mesh_query_dims"):
                        backend._patch_mesh_query_dims()
                else:
                    backend.checker.data.meshes.update_pose(clip.name, w_obj_pose=CuPose(position=th.tensor([list(pos)], dtype=th.float32, device=dev),
                                                                                         quaternion=th.tensor([list(qw)], dtype=th.float32, device=dev)))
                backend.checker.data.meshes.set_enabled(clip.name, True)
            active[clip.name] = active.get(clip.name, 0) + 1
            out.append(clip.name)
        except Exception as e:  # noqa: BLE001
            log.info("SUPPORT_SPLIT %s: the clipped part of %s could not be added (%r): the piece stays off whole", plan.name, clip.prim, e)
    return out


def disable_clips(backend: Any, names: Sequence[str]) -> None:
    active = _ACTIVE_CLIPS.setdefault(id(backend), {})
    for name in names:
        active[name] = active.get(name, 1) - 1
        if active[name] <= 0:
            active.pop(name, None)
            try:
                backend.checker.data.meshes.set_enabled(name, False)
            except Exception as e:  # noqa: BLE001
                log.info("SUPPORT_SPLIT: switching the clipped part %s off failed (%r)", name, e)


def _mat_quat_wxyz(R: np.ndarray) -> List[float]:
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)]


# ----------------------------------------------------------------------------------------------------------------
# the draw's protrusion count (measurement only)
# ----------------------------------------------------------------------------------------------------------------


def protrusion_hit(ikc: Any, supports: Iterable[Any], finger_pts_world: np.ndarray) -> Optional[str]:
    """Whether any of the (K, 3) world points (the pre-shaped finger pads / tips at a candidate grasp) lies below a protrusion piece's top
    and inside its xy hull, for the split supports among ``supports``: the protrusion's ``"support/piece"`` id, else None. A cheap 2-D
    test for ``support_split.protrusion_hits`` (the draw's waste the flat-slab maps cannot see); nothing is rejected on it."""
    P = np.asarray(finger_pts_world, dtype=np.float64).reshape(-1, 3)
    for o in supports or ():
        h = _hit_one(plan_for(ikc, o, count=False), P)
        if h:
            return h
    return None


def _hit_one(plan: Optional[Plan], P: np.ndarray) -> Optional[str]:
    if plan is None or not plan.kept:
        return None
    loc = to_local(plan.root, P)
    for pid, top, hull in plan.cls.hulls:
        below = loc[:, 2] < float(plan.reg.dz) + float(top)
        if bool(below.any()) and bool(inside_hull(hull, loc[below][:, :2]).any()):
            return f"{plan.name}/{pid.rsplit('/', 1)[-1]}"
    return None


def whole_ignoring(ikc: Any, objs: Sequence[Any]):
    """``ikc.ignoring(objs, split=False)``: every object whole (diagnostics asking "with these objects out of the world"); an IK checker
    stand-in without the ``split`` argument (tests) ignores as it does."""
    try:
        return ikc.ignoring(list(objs), split=False)
    except TypeError:
        return ikc.ignoring(list(objs))


def finger_points(eef_pose: Any, opening: Optional[float] = None) -> np.ndarray:
    """(4, 3) world points of the pre-shaped fingers at an eef pose: the pad centre and the fingertip, each shifted by ``opening`` (m per
    finger; default fully open) both ways along the closing axis (``conventions``)."""
    from ispatialgym.robot import conventions as C

    w = float(C.FINGER_OPEN if opening is None else opening)
    pose = (C.as_tensor(eef_pose[0]), C.as_tensor(eef_pose[1]))
    c = _np(C.closing_axis(pose[1])).ravel()[:3]
    out = []
    for p in (_np(C.pad_center(pose)).ravel()[:3], _np(C.fingertip(pose)).ravel()[:3]):
        out += [p + w * c, p - w * c]
    return np.stack(out)


def count_protrusion_hit(s: Any, eef_pose: Any, supports: Sequence[Any], opening: Optional[float] = None) -> Optional[str]:
    """The draw's measurement (no rejection): ``support_split.protrusion_hits`` += 1 when a candidate grasp's pre-shaped fingers
    (:func:`finger_points`) sit below a kept protrusion's top inside its xy hull (:func:`protrusion_hit`) - the waste of the flat-slab
    reach maps on a support with rims / walls. Returns the protrusion's id, None otherwise (or with the split off)."""
    ikc = getattr(s, "ikc", None)
    if ikc is None or not split_on() or not supports:
        return None
    if not any(registration(ikc, getattr(o, "name", None)) is not None for o in supports):
        return None
    try:
        hit = protrusion_hit(ikc, supports, finger_points(eef_pose, opening))
    except Exception as e:  # noqa: BLE001 - a measurement only
        log.debug("support split: protrusion count failed (%r)", e)
        return None
    if hit:
        _count(ikc, "support_split.protrusion_hits")
    return hit
