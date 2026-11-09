"""i7-t03 round 7 (LEAD DECISION #37, owner away; DEFAULT OFF): a collision-MESH check of one robot self-pair that the cuRobo sphere model misses.

``left_arm_link5`` (its wrist-end convex piece ``left_arm_link5_col_3`` reaches z -0.19 in the link frame, up to 57 mm outside link5's two
spheres, 10 mm outside every link5-7 sphere) touches ``torso_link2`` in the simulator while the spheres read +5 .. +38 mm clear: otwukr T7 / T8
(5 demos) and kmuacd arm d (2 demos) failed QA ``robot_touching_itself`` at the shelf tip release's retreat (DECISIONS "i7-t03 r6"). The sphere
model is unchanged; this module reads the simulator's own collision pieces (the convex hulls of ``r1pro.usda``, link-local) and the URDF's
kinematic chain (numpy, no simulator) and returns the pair's mesh clearance at a configuration.

``ISPATIALGYM_T03_PAIR_MESH=1``: the shelf tip release's checked retreat / clear targets (``IORSSmall.release_clear_why``, the 5p.1 place end,
the program check's retreat witnesses) refuse a configuration whose ``{arm}_arm_link5`` / ``torso_link2`` mesh clearance is <= ``PAIR_MESH_MARGIN_M`` (5 mm).
"""
from __future__ import annotations

import functools
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PAIR_MESH_ENV = "ISPATIALGYM_T03_PAIR_MESH"
#: a pair at or below this mesh clearance (m) is refused. Not 0: the executed arm comes closer than the checked witness - T8 300301 / 300303's
#: 15 cm witnesses read +2.1 / +0.1 mm and the executed retreat touched (0.0 mm at the step before QA's contact); the successful T7 retreat
#: witnesses read +16.8 .. +61.5 mm (i7-t03 r6 / r7 measurements)
PAIR_MESH_MARGIN_M = 0.005
#: the checked pair per arm (link a, link b)
PAIR_LINKS = {arm: (f"{arm}_arm_link5", "torso_link2") for arm in ("left", "right")}


def pair_mesh_on(s=None) -> bool:
    """The environment's explicit value, else on for a sampler whose customs list it (``alias_switches``: touching_03 only)."""
    from ispatialgym.sampling.v2 import alias_switches as AS

    return AS.switch_on(PAIR_MESH_ENV, s)


def _assets_dir() -> str:
    from ispatialgym.oracle import r1pro_model as M

    return M.robot_assets_dir()


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    k = axis / max(float(np.linalg.norm(axis)), 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


@functools.lru_cache(maxsize=2)
def urdf_chain(urdf_path: Optional[str] = None) -> Dict[str, Tuple[str, str, str, np.ndarray, np.ndarray]]:
    """child link -> (parent link, joint name, joint type, origin 4x4, axis) of the robot URDF (``r1pro.urdf``)."""
    path = urdf_path or os.path.join(_assets_dir(), "urdf", "r1pro.urdf")
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
        ax = j.find("axis")
        axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        out[j.find("child").get("link")] = (j.find("parent").get("link"), j.get("name"), j.get("type"), T, axis)
    return out


def link_transform(link: str, joints: Dict[str, float], chain: Optional[dict] = None) -> np.ndarray:
    """4x4 pose of ``link`` in the URDF root frame (``base_link``) at the named joint values (missing joints = 0)."""
    chain = chain or urdf_chain()
    Ts = []
    cur = link
    while cur in chain:
        parent, jn, jt, T0, axis = chain[cur]
        T = T0.copy()
        if jt in ("revolute", "continuous"):
            Tj = np.eye(4)
            Tj[:3, :3] = _axis_angle(axis, float(joints.get(jn, 0.0)))
            T = T @ Tj
        elif jt == "prismatic":
            Tj = np.eye(4)
            Tj[:3, 3] = axis * float(joints.get(jn, 0.0))
            T = T @ Tj
        Ts.append(T)
        cur = parent
    out = np.eye(4)
    for T in reversed(Ts):
        out = out @ T
    return out


def _quat_wxyz(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vec(s: str, key: str) -> Optional[np.ndarray]:
    m = re.search(re.escape(key) + r" = \(([^)]*)\)", s)
    return None if m is None else np.array([float(v) for v in m.group(1).split(",")])


def parse_usda_pieces(text: str, link: str) -> List[Tuple[str, np.ndarray]]:
    """The collision pieces (name, link-local vertices) of ``link`` in a robot ``.usda`` text: every ``Mesh`` with ``PhysicsCollisionAPI``
    inside the link's ``Xform`` block, its points scaled / rotated / translated by its own xform ops (the link Xform's frame = the URDF link
    frame at the zero configuration for r1pro: identity orientation, same origin - checked in i7-t03 r6)."""
    i = text.index(f'    def Xform "{link}" (')
    j = text.find('\n    def Xform "', i + 10)
    b = text[i:j if j > 0 else len(text)]
    out = []
    for m in re.finditer(r'def Mesh "([^"]+)" \(\s*apiSchemas = \["PhysicsCollisionAPI"', b):
        k = b.find("{", m.end())
        depth, e = 0, k
        while True:
            c = b[e]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            e += 1
        blk = b[k:e]
        pts = re.search(r"point3f\[\] points = \[(.*?)\]", blk, re.S).group(1)
        P = np.array([[float(v) for v in tup.split(",")] for tup in re.findall(r"\(([^)]*)\)", pts)])
        sc = _vec(blk, "double3 xformOp:scale")
        tr = _vec(blk, "double3 xformOp:translate")
        q = _vec(blk, "quatd xformOp:orient")
        P = P * (sc if sc is not None else 1.0)
        if q is not None:
            P = P @ _quat_wxyz(q).T
        if tr is not None:
            P = P + tr
        out.append((m.group(1), P))
    return out


@functools.lru_cache(maxsize=4)
def link_pieces(link: str, usda_path: Optional[str] = None) -> Tuple[Tuple[str, np.ndarray], ...]:
    """The simulator's collision pieces of ``link`` (read once per process from ``usd/r1pro.usda``)."""
    path = usda_path or os.path.join(_assets_dir(), "usd", "r1pro.usda")
    text = _usda_text(path)
    return tuple(parse_usda_pieces(text, link))


@functools.lru_cache(maxsize=1)
def _usda_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def hulls_overlap(A: np.ndarray, B: np.ndarray) -> bool:
    """True when conv(A) and conv(B) share a point (exact: the LP ``A^T l = B^T m``, ``l, m`` on the simplex, is feasible)."""
    from scipy.optimize import linprog

    na, nb = len(A), len(B)
    A_eq = np.zeros((5, na + nb))
    A_eq[:3, :na], A_eq[:3, na:] = A.T, -B.T
    A_eq[3, :na], A_eq[4, na:] = 1.0, 1.0
    b_eq = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    r = linprog(np.zeros(na + nb), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return bool(r.status == 0)


def _affine_min(S: np.ndarray) -> np.ndarray:
    """The weights (sum 1) of the min-norm point of the affine hull of the rows of ``S``."""
    n = len(S)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = S @ S.T
    K[:n, n], K[n, :n] = 1.0, 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    return np.linalg.lstsq(K, rhs, rcond=None)[0][:n]


def min_norm_point(A: np.ndarray, B: np.ndarray, eps: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    """Wolfe's min-norm-point algorithm on the Minkowski difference conv(A) - conv(B) (support oracle, corral of <= 4 points): the point
    of the difference nearest the origin - its norm is the distance between the hulls (exact up to round-off when they are apart)."""
    def support(d):
        return A[int(np.argmin(A @ d))] - B[int(np.argmax(B @ d))]

    S = np.array([A[0] - B[0]])
    w = np.array([1.0])
    x = S[0].copy()
    for _ in range(max_iter):
        p = support(x)
        if float(x @ x) - float(x @ p) <= eps * max(1.0, float(x @ x)) or float(x @ x) < 1e-18:
            break
        if any(np.allclose(p, s_, atol=1e-12) for s_ in S):
            break
        S = np.vstack([S, p])
        w = np.append(w, 0.0)
        for _minor in range(20):
            alpha = _affine_min(S)
            if np.all(alpha > 1e-12):
                w = alpha
                break
            neg = alpha <= 1e-12
            theta = min(1.0, float(np.min(w[neg] / np.maximum(w[neg] - alpha[neg], 1e-18))))
            w = theta * alpha + (1.0 - theta) * w
            keep = w > 1e-12
            S, w = S[keep], w[keep] / w[keep].sum()
        x = w @ S
    return x


def convex_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Signed distance (m) between conv(A) and conv(B). Overlap is decided exactly (:func:`hulls_overlap`) and returned as minus the deepest
    vertex of one inside the other's hull (0.0 for an edge-edge crossing with no vertex inside); a gap is the norm of Wolfe's min-norm point
    of the Minkowski difference (:func:`min_norm_point`)."""
    if hulls_overlap(A, B):
        return -_overlap_depth(A, B)
    return float(np.linalg.norm(min_norm_point(A, B)))


def _overlap_depth(A: np.ndarray, B: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    try:
        ha, hb = ConvexHull(A), ConvexHull(B)
    except Exception:  # noqa: BLE001
        return 0.0

    def depth(P, h):
        return -(P @ h.equations[:, :3].T + h.equations[:, 3]).max(axis=1)

    return float(max(0.0, depth(A[ha.vertices], hb).max(), depth(B[hb.vertices], ha).max()))


def joints_of(q: Sequence[float]) -> Dict[str, float]:
    """The named joint values of a full configuration in ``r1pro_model.ALL_JOINTS`` order (the sampler's / oracle's 28-vector)."""
    from ispatialgym.oracle import r1pro_model as M

    return dict(zip(M.ALL_JOINTS, [float(v) for v in q]))


def pair_clearance(q: Sequence[float], arm: str = "left", links: Optional[Tuple[str, str]] = None) -> Tuple[float, str]:
    """``({arm}_arm_link5 vs torso_link2 mesh clearance (m), the closest piece pair)`` at the full configuration ``q`` (ALL_JOINTS order)."""
    la, lb = links or PAIR_LINKS[arm]
    jv = joints_of(q)
    chain = urdf_chain()
    Ta, Tb = link_transform(la, jv, chain), link_transform(lb, jv, chain)
    best, which = math.inf, ""
    pb = [(n, P @ Tb[:3, :3].T + Tb[:3, 3]) for n, P in link_pieces(lb)]
    for na, P in link_pieces(la):
        W = P @ Ta[:3, :3].T + Ta[:3, 3]
        ca, ra = W.mean(0), float(np.linalg.norm(W - W.mean(0), axis=1).max())
        for nb, V in pb:
            cb, rb = V.mean(0), float(np.linalg.norm(V - V.mean(0), axis=1).max())
            if float(np.linalg.norm(ca - cb)) - ra - rb > min(best, 0.05):
                continue  # bounding spheres farther apart than the best so far (or 5 cm): this piece pair cannot be the closest touch
            d = convex_distance(W, V)
            if d < best:
                best, which = d, f"{na} x {nb}"
    return (best if math.isfinite(best) else 0.05), which


def pair_touch_why(q: Sequence[float], arm: str, margin: float = PAIR_MESH_MARGIN_M) -> Optional[str]:
    """None when the pair's mesh clearance at ``q`` is above ``margin``, else why (for the reject message)."""
    cl, which = pair_clearance(q, arm)
    if cl > margin:
        return None
    a, b = PAIR_LINKS[arm]
    return f"{a} touches {b} by collision mesh ({cl * 1000:+.1f} mm, {which}; the sphere model misses it)"
