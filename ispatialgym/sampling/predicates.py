"""Robust BDDL predicate evaluation shared by the sampler's goal check and the task evaluator (D-007, option A, owner
2026-09-16 21:17).

``under`` / ``ontop`` through OmniGibson alone are fragile on a physically correct placement: ``Under`` / ``OnTop`` shoot a
single vertical ray from the supporting object's own surface point into the resting object, and whether that ray
registers the object depends on sub-millimetre contact / penetration (teacup on the saucer well, 0916-172359 and
0916-192655: the cup 1.4 mm from its goal, 2.7 deg tilt, ``binary_success_ever`` False; the same geometry passed at 18:41).

``evaluate_predicate(predicate_cls, *entities)`` is a drop-in for ``omnigibson.utils.bddl_utils.evaluate_bddl_predicate``:

1. **omnigibson** — OmniGibson's verdict; True ends the evaluation.
2. **rays** (``under`` / ``ontop`` only) — several rays across the contact patch: the resting object's footprint centre and a
   ring at half its footprint radius; at each point the supporting object's surface under the resting object is found with
   a downward ray (the resting object's bodies ignored), the probe ray starts ``RAY_BELOW_SURFACE_M`` below that surface and
   goes up (the supporting object's bodies ignored); any hit on the resting object counts.
3. **geometry** — the resting object's AABB bottom within ``BOTTOM_GAP_MAX_M`` of the supporting surface under its footprint
   centre, the footprint centre inside the supporting object's top footprint, the resting object's up axis within
   ``UP_AXIS_MAX_DEG`` of vertical (a concave partner whose collision mesh lets the rays through).

``touching`` (owner 2026-09-19 00:49): OmniGibson's ``Touching`` is a PhysX contact report, which at rest between two
free-standing objects with zero normal force is a 0.1 mm lottery (touching 0918-200452 vs 0918-203047 / 224622: the same
final geometry to 0.1 mm, one read True for 260 steps, the others False). After an OmniGibson False the **geometry** branch
measures the separation of the two objects' convex hulls (the collision boundary points against the other hull's
half-spaces, exact for a face-to-face contact, a lower bound otherwise) and accepts a gap up to ``TOUCHING_GAP_MAX_M``
(3 mm; ``ISG_TOUCHING_GAP_MAX_M`` overrides).

``attached`` (owner 2026-09-19 17:10): OmniGibson's ``AttachedTo`` joins two parts only through its own contact-triggered
``_update`` (never between an ``attachment_targets`` pair — their collisions are filtered so an attached part does not
jitter) and only when the male / female meta-link frames agree within 5 cm **and 15° in full SO(3)**, an invisible yaw
constraint for a rotationally symmetric part (a round lid attaches in ~1/12 of its yaws). After an OmniGibson False the
**alignment** branch tests the meta-link pair **modulo the part's recorded symmetry** (the group / pose target's
``orientation_equivalences``, registered by the task through :func:`set_attachment_context`; ``zinf`` = yaw free) with the
same 5 cm / 15° and, when aligned, creates the BEHAVIOR attachment (``AttachedTo.set_value(True,
bypass_alignment_checking=True)``: the part snaps onto the female frame — a no-op for the shape of a symmetric part — and
gets the fixed joint), so the atom stays a real attachment joint and a policy can satisfy it at any equivalent yaw. The
oracle's attach at the place pose (``executor.SimState.attach``) is the same call.

Every evaluation appends ``{"predicate", "entities", "branch", "ok", "detail"}`` to the branch log (``reset_branches`` /
``branches``); the callers copy it into ``goal_check.binary_branch`` (sampler) and ``meta.binary_branch`` (evaluator) so the
ledger shows which branch decided. Other predicates pass straight through.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

RAY_BELOW_SURFACE_M = 0.003
RAY_LENGTH_M = 0.40
BOTTOM_GAP_MAX_M = 0.008
UP_AXIS_MAX_DEG = 15.0
RING_POINTS = 8
TOUCHING_GAP_MAX_M = float(os.environ.get("ISG_TOUCHING_GAP_MAX_M", "0.003"))
#: attachment alignment (OmniGibson ``AttachedTo``'s own thresholds: m.DEFAULT_POSITION_THRESHOLD / m.DEFAULT_ORIENTATION_THRESHOLD)
ATTACH_POS_MAX_M = 0.05
ATTACH_ORN_MAX_RAD = math.radians(15.0)

_BRANCHES: List[Dict[str, Any]] = []
#: (child object name, parent object name) -> {"child_link", "parent_link", "equivalences": [[x, y, z, w], ...]} — the task's
#: ``attachment_targets`` with the child part's recorded symmetry (``set_attachment_context``)
_ATTACHMENTS: Dict[Tuple[str, str], Dict[str, Any]] = {}


def reset_branches() -> None:
    _BRANCHES.clear()


def branches() -> List[Dict[str, Any]]:
    return list(_BRANCHES)


MAX_LOG = 200  # the evaluator calls every step: keep the latest entries only


def _log(predicate: str, entities: Sequence[Any], branch: str, ok: bool, detail: Optional[Dict[str, Any]] = None) -> None:
    _BRANCHES.append({"predicate": predicate, "entities": [getattr(e, "name", str(e)) for e in entities], "branch": branch, "ok": bool(ok),
                      "detail": detail or {}})
    if len(_BRANCHES) > MAX_LOG:
        del _BRANCHES[: len(_BRANCHES) - MAX_LOG]


def decided_by() -> Dict[str, str]:
    """``{"<predicate>(<a>, <b>)": branch}`` of the latest evaluation of every under / ontop atom in the log — what a caller
    writes as ``binary_branch``."""
    out: Dict[str, str] = {}
    for e in _BRANCHES:
        if e["predicate"] in ("under", "ontop", "touching", "attached"):
            out[f"{e['predicate']}({', '.join(e['entities'])})"] = e["branch"]
    return out


# ----------------------------------------------------------------------------------------------------------------
# attached: alignment modulo the part's symmetry, the BEHAVIOR joint created when aligned
# ----------------------------------------------------------------------------------------------------------------


def set_attachment_context(pairs: Sequence[Dict[str, Any]]) -> None:
    """Register the task's attachment pairs: ``{"child": <object name>, "parent": <object name>, "child_link", "parent_link",
    "equivalences": [[x, y, z, w], ...]}`` (object-frame rotations of the child that leave its shape unchanged; the identity
    is implied). Called by the task when the TRO targets are configured; ``[]`` clears."""
    _ATTACHMENTS.clear()
    for p in pairs or []:
        if p.get("child") and p.get("parent"):
            _ATTACHMENTS[(str(p["child"]), str(p["parent"]))] = {"child_link": p.get("child_link"), "parent_link": p.get("parent_link"),
                                                                 "equivalences": [[float(v) for v in q] for q in (p.get("equivalences") or [])]}


def attachment_context(child: Any, parent: Any) -> Optional[Dict[str, Any]]:
    return _ATTACHMENTS.get((getattr(child, "name", str(child)), getattr(parent, "name", str(parent))))


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return th.tensor([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
                      aw * bw - ax * bx - ay * by - az * bz], dtype=th.float64)


def _qinv(q):
    x, y, z, w = q
    return th.tensor([-x, -y, -z, w], dtype=th.float64)


def _qangle(a, b) -> float:
    d = abs(float(th.dot(a / th.linalg.norm(a), b / th.linalg.norm(b))))
    return 2.0 * math.acos(min(1.0, d))


def attachment_alignment(child: Any, parent: Any, equivalences: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, Any]:
    """Meta-link alignment of ``child``'s male link with ``parent``'s free female link, modulo the child's symmetry: for every
    object-frame equivalence ``q_e`` (identity included) the male frame of the equivalently rotated child is
    ``q_obj q_e q_obj^-1 q_male``; the smallest orientation difference to the female frame decides. Returns ``{"ok", "pos_m",
    "rot_rad", "rot_raw_rad", "equivalence", "child_link", "parent_link", "first_free_pair"}``; ``ok`` False with ``reason``
    when the pair has no attachment meta links or is already attached elsewhere."""
    from omnigibson.object_states import AttachedTo

    if AttachedTo not in getattr(child, "states", {}) or AttachedTo not in getattr(parent, "states", {}):
        return {"ok": False, "reason": "no AttachedTo state"}
    st_c, st_p = child.states[AttachedTo], parent.states[AttachedTo]
    if st_c.parent is not None:
        return {"ok": st_c.parent == parent, "reason": "already attached"}
    cands = st_c._get_parent_candidates(parent) or {}
    ctx = attachment_context(child, parent) or {}
    if equivalences is None:
        equivalences = ctx.get("equivalences") or []
    eqs = [th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float64)] + [th.tensor([float(v) for v in q], dtype=th.float64) for q in equivalences]
    op, oq = (x.double() for x in child.get_position_orientation())
    best: Optional[Dict[str, Any]] = None
    first_free: Optional[Tuple[str, str]] = None
    for child_name, parent_names in cands.items():
        for parent_name in sorted(parent_names):
            if st_p.children.get(parent_name) is not None:
                continue
            if first_free is None:
                first_free = (child_name, parent_name)
            if ctx.get("child_link") and ctx.get("parent_link") and (child_name, parent_name) != (ctx["child_link"], ctx["parent_link"]):
                continue
            cl, pl = st_c.links[child_name], st_p.links[parent_name]
            cp, cq = (x.double() for x in cl.get_position_orientation())
            pp, pq = (x.double() for x in pl.get_position_orientation())
            pos = float(th.linalg.norm(cp - pp))
            raw = _qangle(cq, pq)
            rot, which = raw, 0
            for k, qe in enumerate(eqs[1:], start=1):
                q_alt = _qmul(_qmul(_qmul(oq, qe), _qinv(oq)), cq)
                ang = _qangle(q_alt, pq)
                if ang < rot:
                    rot, which = ang, k
            rec = {"ok": pos <= ATTACH_POS_MAX_M and rot <= ATTACH_ORN_MAX_RAD, "pos_m": round(pos, 5), "rot_rad": round(rot, 5), "rot_raw_rad": round(raw, 5),
                   "equivalence": which, "n_equivalences": len(eqs) - 1, "child_link": child_name, "parent_link": parent_name}
            if best is None or (rec["ok"] and not best["ok"]) or (rec["ok"] == best["ok"] and rec["pos_m"] + rec["rot_rad"] < best["pos_m"] + best["rot_rad"]):
                best = rec
    if best is None:
        return {"ok": False, "reason": "no free attachment meta-link pair"}
    best["first_free_pair"] = list(first_free) if first_free else None
    return best


def attach_if_aligned(child: Any, parent: Any, equivalences: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, Any]:
    """The BEHAVIOR attachment of ``child`` to ``parent`` when :func:`attachment_alignment` passes: ``AttachedTo.set_value(True,
    bypass_alignment_checking=True)`` snaps the child onto the female frame (its own thresholds were met modulo the symmetry)
    and creates the fixed joint. Returns the alignment record with ``attached``."""
    from omnigibson.object_states import AttachedTo

    a = attachment_alignment(child, parent, equivalences)
    if a.get("reason") == "already attached":
        a["attached"] = bool(a["ok"])
        return a
    if not a.get("ok"):
        a["attached"] = False
        return a
    pair = (a["child_link"], a["parent_link"])
    if a.get("first_free_pair") is not None and tuple(a["first_free_pair"]) != pair:
        a.update({"attached": False, "reason": "the aligned pair is not OmniGibson's first free candidate pair"})
        return a
    a["attached"] = bool(child.states[AttachedTo].set_value(parent, True, bypass_alignment_checking=True))
    return a


def evaluate_predicate(predicate_cls, *entities) -> bool:
    """Drop-in for ``evaluate_bddl_predicate`` with the robust ``under`` / ``ontop`` / ``touching`` re-checks (module docstring)."""
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate

    name = str(getattr(predicate_cls, "__name__", predicate_cls)).lower()
    ok = bool(evaluate_bddl_predicate(predicate_cls, *entities))
    if name not in ("under", "ontop", "touching", "attached") or len(entities) != 2 or any(e is None or not _is_object(e) for e in entities):
        _log(name, entities, "omnigibson", ok)
        return ok
    if ok:
        _log(name, entities, "omnigibson", True)
        return True
    if name == "attached":
        # attached(child, parent): the BEHAVIOR joint does not exist yet — align modulo the child's symmetry and create it
        try:
            a = attach_if_aligned(entities[0], entities[1])
        except Exception as e:  # noqa: BLE001 - an alignment failure must not turn into a verdict
            a = {"ok": False, "attached": False, "error": repr(e)}
        got = bool(a.get("attached"))
        _log(name, entities, "alignment" if got else "none", got, a)
        return got
    if name == "touching":
        try:
            t = touching_geometry(entities[0], entities[1])
        except Exception as e:  # noqa: BLE001 - a geometry failure must not turn into a verdict
            t = {"ok": False, "error": repr(e)}
        _log(name, entities, "geometry" if t.get("ok") else "none", bool(t.get("ok")), t)
        return bool(t.get("ok"))
    # under(a, b): a is under b -> a supports b; ontop(a, b): a rests on b -> b supports a
    lower, upper = (entities[0], entities[1]) if name == "under" else (entities[1], entities[0])
    try:
        r = support_rays(lower, upper)
    except Exception as e:  # noqa: BLE001 - a raycast failure must not turn into a verdict
        r = {"ok": False, "error": repr(e)}
    if r.get("ok"):
        _log(name, entities, "rays", True, r)
        return True
    try:
        g = support_geometry(lower, upper)
    except Exception as e:  # noqa: BLE001
        g = {"ok": False, "error": repr(e)}
    if g.get("ok"):
        _log(name, entities, "geometry", True, g)
        return True
    _log(name, entities, "none", False, {"rays": r, "geometry": g})
    return False


# ----------------------------------------------------------------------------------------------------------------
# branches
# ----------------------------------------------------------------------------------------------------------------


def _is_object(e: Any) -> bool:
    return hasattr(e, "links") and hasattr(e, "get_position_orientation")


def _boundary_points(obj: Any) -> th.Tensor:
    """World collision boundary points (the convex-hull vertices of every link's collision geometry)."""
    try:
        import omnigibson as og

        og.sim.sync_physx_to_fabric()  # the boundary points read the render-side transforms (sim_utils.local_hull)
    except Exception:  # noqa: BLE001
        pass
    pts = [p for p in (link.collision_boundary_points_world for link in getattr(obj, "links", {}).values()) if p is not None and len(p)]
    return th.cat([th.as_tensor(p).float() for p in pts], dim=0) if pts else th.zeros(0, 3)


def hull_gap(points_a: th.Tensor, points_b: th.Tensor) -> float:
    """Separation of the convex hulls of two point sets: max over the hull-of-b half-spaces of the signed distance of a's
    points, minimised over a's points, and vice versa; the smaller of the two. Exact for a face-to-face contact (the
    contact face's vertices against the other face's plane), a lower bound for edge / corner contacts; negative when the
    hulls interpenetrate."""
    import numpy as np
    from scipy.spatial import ConvexHull

    hulls = {id(points_a): ConvexHull(points_a.numpy().astype(np.float64)), id(points_b): ConvexHull(points_b.numpy().astype(np.float64))}

    def surface_points(pts: th.Tensor) -> th.Tensor:
        # the hull vertices plus every hull triangle's centroid and edge midpoints: two faces overlapping only partially
        # (each overhanging the other) have no vertex inside the other's extent, and vertices alone would overestimate
        h = hulls[id(pts)]
        tri = pts[th.as_tensor(h.simplices, dtype=th.long)]  # (n_tri, 3, 3)
        extra = th.cat([tri.mean(dim=1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 0] + tri[:, 2]) / 2], dim=0)
        return th.cat([pts, extra], dim=0)

    def one_way(pts: th.Tensor, hull_pts: th.Tensor) -> float:
        eq = th.as_tensor(hulls[id(hull_pts)].equations).float()  # n·x + d <= 0 inside
        signed = surface_points(pts) @ eq[:, :3].T + eq[:, 3]  # (n_pts, n_faces)
        return float(signed.max(dim=1).values.min())

    return min(one_way(points_a, points_b), one_way(points_b, points_a))


def touching_geometry(a: Any, b: Any) -> Dict[str, Any]:
    """Geometric touching: the two objects' convex hulls within ``TOUCHING_GAP_MAX_M`` of each other (or interpenetrating)."""
    pa, pb = _boundary_points(a), _boundary_points(b)
    if len(pa) < 4 or len(pb) < 4:
        return {"ok": False, "why": "no collision boundary points"}
    gap = hull_gap(pa, pb)
    return {"ok": gap <= TOUCHING_GAP_MAX_M, "gap_m": round(gap, 5), "gap_max_m": TOUCHING_GAP_MAX_M}


def _link_paths(obj: Any) -> List[str]:
    return [link.prim_path for link in getattr(obj, "links", {}).values()]


def _aabb(obj: Any) -> Tuple[th.Tensor, th.Tensor]:
    lo, hi = obj.aabb
    return th.as_tensor(lo).float(), th.as_tensor(hi).float()


def _footprint_points(upper: Any) -> List[Tuple[float, float]]:
    """The resting object's footprint centre and a ring at half its footprint radius (world xy)."""
    lo, hi = _aabb(upper)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    r = 0.25 * float(min(hi[0] - lo[0], hi[1] - lo[1]))  # half the footprint radius
    pts = [(cx, cy)]
    for k in range(RING_POINTS):
        a = 2 * math.pi * k / RING_POINTS
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _surface_under(lower: Any, upper: Any, x: float, y: float) -> Optional[float]:
    """z of the supporting object's surface under world (x, y): a downward ray from above its AABB top with the resting
    object's bodies ignored; None when the first hit is not the supporting object."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_l, hi_l = _aabb(lower)
    start = [x, y, float(hi_l[2]) + 0.05]
    end = [x, y, float(lo_l[2]) - 0.02]
    hits = raytest_batch([start], [end], only_closest=True, ignore_bodies=_link_paths(upper) or None)
    h = hits[0] if hits else {}
    if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(tuple(_link_paths(lower))):
        return None
    return float(h["position"][2])


def support_rays(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 2: probe rays from just below the supporting surface up into the resting object across its contact patch."""
    from omnigibson.utils.sampling_utils import raytest_batch

    upper_paths = tuple(_link_paths(upper))
    lower_paths = _link_paths(lower)
    starts, ends = [], []
    n_surface = 0
    for x, y in _footprint_points(upper):
        z = _surface_under(lower, upper, x, y)
        if z is None:
            continue
        n_surface += 1
        starts.append([x, y, z - RAY_BELOW_SURFACE_M])
        ends.append([x, y, z - RAY_BELOW_SURFACE_M + RAY_LENGTH_M])
    if not starts:
        return {"ok": False, "n_rays": 0, "n_surface": 0, "why": "no supporting surface under the footprint"}
    hits = raytest_batch(starts, ends, only_closest=True, ignore_bodies=lower_paths or None)
    n_hit = sum(1 for h in hits if h.get("hit") and str(h.get("rigidBody", "")).startswith(upper_paths))
    return {"ok": n_hit > 0, "n_rays": len(starts), "n_surface": n_surface, "n_hits_on_upper": n_hit}


def support_geometry(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 3: bottom-to-surface gap under the footprint centre, footprint centre inside the supporting top, up axis."""
    lo_u, hi_u = _aabb(upper)
    lo_l, hi_l = _aabb(lower)
    cx, cy = float((lo_u[0] + hi_u[0]) / 2), float((lo_u[1] + hi_u[1]) / 2)
    z_surface = _surface_under(lower, upper, cx, cy)
    if z_surface is None:
        return {"ok": False, "why": "no supporting surface under the footprint centre"}
    gap = float(lo_u[2]) - z_surface
    inside = bool(lo_l[0] <= cx <= hi_l[0] and lo_l[1] <= cy <= hi_l[1])
    _, q = upper.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the rotated local z axis
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    ok = abs(gap) <= BOTTOM_GAP_MAX_M and inside and tilt <= UP_AXIS_MAX_DEG
    return {"ok": ok, "bottom_gap_mm": round(gap * 1000, 1), "centre_inside_support": inside, "tilt_deg": round(tilt, 1)}
