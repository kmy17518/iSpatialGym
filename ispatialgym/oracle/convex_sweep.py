"""The first touch of two sets of convex pieces along a direction (owner 2026-11-07, configuration_matching-attachment_04, option B).

The attach touchdown's hover (``execute.attach_hover_targets``, ``ISPATIALGYM_ATTACH_HOVER=swept``) starts from the height at which
the held part, coming down the seat axis, first meets its partner. Both are sets of convex pieces: PhysX collides the convex hull of
every ``convexHull``-approximated collision mesh (BEHAVIOR-1K's object collision meshes), so the first touch is the LARGEST ``t``
at which some moving piece translated by ``t * axis`` meets some fixed piece (the moving side comes down from ``t = +inf``).

For one pair of convex polytopes ``A`` (moving) and ``B`` (fixed) the ``t`` at which ``A + t axis`` meets ``B`` form an interval;
its top end is a touching configuration, and a translational contact of two convex polytopes is met at a vertex of one on a face
of the other or at an edge of each, so the top end is the largest of

1. every vertex ``a`` of ``A``: the top of the interval in which ``a + t axis`` lies inside ``B`` (its halfspaces);
2. every vertex ``b`` of ``B``: the top of the interval in which ``b - t axis`` lies inside ``A``;
3. every edge pair: ``a0 + s eA + t axis = b0 + r eB`` with ``s, r`` in [0, 1] (a 3 x 3 solve, Cramer's rule).

Each candidate is a ``t`` at which the two pieces share a point, so none exceeds the top end, and the top end is one of them: the
result is exact (to ``tol``). Per pair it is vectorized in numpy (float64): ``V_A x F_B + V_B x F_A`` ray / halfspace terms and
``E_A x E_B`` edge solves (about 35 k for two 64-vertex hulls, ~1 ms). Pairs whose footprints across the axis (their bounding boxes
in a plane normal to it) miss are skipped, and the rest are visited by their upper bound - the fixed piece's top minus the moving
piece's bottom along the axis, which no touch exceeds - so the search stops once no pair left can beat the best touch found.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: |n . axis| below this: the face is parallel to the axis (a containment constraint, not a bound)
PARALLEL_EPS = 1e-9


@dataclass
class ConvexPiece:
    """A convex hull in the world: its vertices, its faces as halfspaces ``normals @ x <= offsets`` and its edges ``e0 -> e1``."""

    verts: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    e0: np.ndarray
    e1: np.ndarray
    #: the two faces (indices into ``normals``) on either side of each edge
    edge_faces: np.ndarray

    def edges_facing(self, u: np.ndarray, sign: float) -> np.ndarray:
        """Mask of the edges with a face on their side facing ``sign * u`` (or across it): only those can carry a contact for
        a piece moving along ``-sign * u`` - an edge whose two faces both face away has its whole normal cone facing away."""
        k = sign * (self.normals @ u)
        return k[self.edge_faces].max(axis=1) >= -PARALLEL_EPS


def convex_piece(points: Any) -> Optional[ConvexPiece]:
    """The convex hull of ``points`` (``(N, 3)``; a torch tensor or an array) - None with fewer than 4 points or a hull qhull cannot
    build even joggled (a flat piece gets a joggled, paper-thin hull)."""
    from scipy.spatial import ConvexHull

    try:
        P = np.asarray(points.detach().cpu().numpy() if hasattr(points, "detach") else points, dtype=np.float64).reshape(-1, 3)
    except Exception:  # noqa: BLE001
        return None
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 4:
        return None
    hull = None
    for opts in (None, "QJ"):
        try:
            hull = ConvexHull(P, qhull_options=opts) if opts else ConvexHull(P)
            break
        except Exception:  # noqa: BLE001 - QhullError (flat / degenerate input): try joggled
            hull = None
    if hull is None:
        return None
    s = hull.simplices
    nf = len(s)
    edges, inv = np.unique(np.sort(np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]), axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    faces = np.tile(np.arange(nf), 3)
    ef = np.full((len(edges), 2), -1, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    first = np.ones(len(order), dtype=bool)
    first[1:] = inv[order][1:] != inv[order][:-1]
    ef[inv[order][first], 0] = faces[order][first]
    ef[inv[order][~first], 1] = faces[order][~first]
    ef[ef[:, 1] < 0, 1] = ef[ef[:, 1] < 0, 0]  # a joggled / open edge with one face: that face twice
    return ConvexPiece(verts=P[hull.vertices], normals=hull.equations[:, :3].copy(), offsets=-hull.equations[:, 3].copy(), e0=P[edges[:, 0]], e1=P[edges[:, 1]],
                       edge_faces=ef)


def _ray_intervals(points: np.ndarray, piece: ConvexPiece, u: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """For each point ``p``: ``(top, ok)`` - the top of the interval of ``t`` with ``p + t u`` inside ``piece``, and whether the
    interval is non-empty."""
    k = piece.normals @ u  # (F,)
    r = piece.offsets[None, :] - points @ piece.normals.T  # (V, F): inside iff t * k <= r for every face
    pos, neg = k > PARALLEL_EPS, k < -PARALLEL_EPS
    par = ~(pos | neg)
    q = r / np.where(pos | neg, k, 1.0)[None, :]
    hi = np.where(pos[None, :], q, np.inf).min(axis=1)
    lo = np.where(neg[None, :], q, -np.inf).max(axis=1)
    ok = np.isfinite(hi) & (lo <= hi + tol)
    if par.any():
        ok &= (r[:, par] >= -tol).all(axis=1)
    return hi, ok


def pair_first_touch(A: ConvexPiece, B: ConvexPiece, u: np.ndarray, tol: float = 1e-7) -> Optional[float]:
    """The largest ``t`` with ``(A + t u)`` meeting ``B`` (``u`` a unit vector), None when they never meet."""
    best = -math.inf
    hi, ok = _ray_intervals(A.verts, B, u, tol)  # (1) a vertex of A inside B
    if ok.any():
        best = max(best, float(hi[ok].max()))
    hi, ok = _ray_intervals(B.verts, A, -u, tol)  # (2) a vertex of B inside A: b - t u in A
    if ok.any():
        best = max(best, float(hi[ok].max()))
    # (3) a0 + s eA + t u = b0 + r eB, over A's edges facing down the motion (-u) and B's facing up (+u)
    mA, mB = A.edges_facing(u, -1.0), B.edges_facing(u, 1.0)
    a0, b0 = A.e0[mA], B.e0[mB]
    eA, eB = A.e1[mA] - a0, B.e1[mB] - b0
    if len(eA) and len(eB):
        w = np.cross(eB, u)  # (EB, 3)
        det = -(eA @ w.T)  # (EA, EB): det [eA, -eB, u]
        scale = np.linalg.norm(eA, axis=1)[:, None] * np.linalg.norm(eB, axis=1)[None, :]
        good = np.abs(det) > 1e-9 * np.maximum(scale, 1e-30)
        if good.any():
            d = np.where(good, det, 1.0)
            rr = b0[None, :, :] - a0[:, None, :]  # (EA, EB, 3)
            s = -np.einsum("ijk,jk->ij", rr, w) / d
            r = np.einsum("ijk,ik->ij", rr, np.cross(u, eA)) / d
            t = -np.einsum("ijk,ijk->ij", rr, np.cross(eA[:, None, :], eB[None, :, :])) / d
            eps = 1e-9
            hit = good & (s >= -eps) & (s <= 1 + eps) & (r >= -eps) & (r <= 1 + eps)
            if hit.any():
                best = max(best, float(t[hit].max()))
    return None if best == -math.inf else best


def _perp_basis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def first_touch(moving: Sequence[ConvexPiece], fixed: Sequence[ConvexPiece], axis: Any, tol: float = 1e-7) -> Dict[str, Any]:
    """The largest ``t`` at which some piece of ``moving`` translated by ``t * axis`` meets some piece of ``fixed`` (``t`` None when
    no pair ever meets - their footprints across the axis miss), the pair that sets it (indices into ``moving`` / ``fixed``) and the
    pair counts (total, overlapping across the axis, tested before the bound stopped the search)."""
    u = np.asarray(axis, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    e1, e2 = _perp_basis(u)
    mov, fix = list(moving), list(fixed)

    def box(p: ConvexPiece):
        x, y, z = p.verts @ e1, p.verts @ e2, p.verts @ u
        return x.min(), x.max(), y.min(), y.max(), z.min(), z.max()

    bm = [(i, box(p)) for i, p in enumerate(mov) if p is not None]
    bf = [(j, box(p)) for j, p in enumerate(fix) if p is not None]
    pairs: List[Tuple[float, int, int]] = []
    for i, a in bm:
        for j, b in bf:
            if a[0] > b[1] + tol or b[0] > a[1] + tol or a[2] > b[3] + tol or b[2] > a[3] + tol:
                continue
            pairs.append((b[5] - a[4], i, j))
    pairs.sort(key=lambda x: -x[0])
    best: Optional[float] = None
    arg: Optional[Tuple[int, int]] = None
    n = 0
    for ub, i, j in pairs:
        if best is not None and ub <= best + tol:
            break
        n += 1
        t = pair_first_touch(mov[i], fix[j], u, tol)
        if t is not None and (best is None or t > best):
            best, arg = t, (i, j)
    return {"t": best, "pair": arg, "pairs_total": len(bm) * len(bf), "pairs_overlapping": len(pairs), "pairs_tested": n,
            "pieces_moving": len(bm), "pieces_fixed": len(bf)}
