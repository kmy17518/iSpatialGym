"""Neighbouring supports (owner decision A4, 2026-10-12): a support paired with the support(s) it touches, at any height, is one
placement area for the target and the partner.

The pairing is declared on the support row of ``roles/representatives.csv`` (note key ``neighbors=<support>[,<support>]``, scene
objects of the row's scene), carried by ``ground_isg.support_surface`` into the variant's ``scenes.<scene>.bindings.support_surfaces``
entry (``"neighbors": [...]``) and read here:

* :func:`union_top` (``tabletop.support_surface``): the support top united with each neighbour's top, :class:`UnionTop`. Each top
  keeps its own height; a footprint lies on one top, or across a seam of tops whose heights differ by at most
  ``generation.neighbor_flush_tol_m`` (:data:`FLUSH_TOL_M`). The robot still stands at the support's own edges, each extended along
  its line over the neighbour cells beside it (:meth:`UnionTop.edges`).
* :class:`NeighborBands` (``tabletop.ensure_bands``): the support's reach band plus one band per neighbour, each read on its own
  top only (bands are per support: same object, grasp key, scale; ``ReachBands.load`` by the neighbour's name, computed in-process
  like the support's on a miss).
* the neighbours join ``support_objects()`` (ignored for the final grasp / contact pose), the sampler's ``:init`` check accepts
  ``ontop`` any of the tops (:func:`evaluate_on_tops`), and a target drawn on a neighbour records it as ``grasps[].grasp_from``
  (the oracle leaves it out of the collision world for the grasp poses, ``oracle.execute.contact_ignores``).

A support without neighbours keeps its own top (:func:`union_top` returns it) and every helper below is the identity for it.
Pure torch / numpy: no simulator import.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch as th

from ispatialgym.sampling import reach_bands as RB

log = logging.getLogger("ispatialgym.sampling.v2")

#: two tops whose heights differ by at most this are flush: a footprint may straddle their seam (``generation.neighbor_flush_tol_m``)
FLUSH_TOL_M = 0.01
#: a neighbour is left out unless this many of its free cells have free cells all around them ``MIN_MARGIN_M`` away (the support's
#: own ``support_small`` bar of 50 free cells, counted on the eroded top so that a strip too narrow to hold anything does not count;
#: ``generation.neighbor_min_free_cells`` / ``neighbor_min_margin_m``)
MIN_FREE_CELLS = 50
MIN_MARGIN_M = 0.04
#: neighbour cells that extend an edge of the support lie at most this far in front of its line / behind it (m;
#: ``generation.neighbor_edge_strip_m``)
EDGE_STRIP_M = (0.10, 0.30)
#: BDDL predicates whose support argument stands for every top of the area
NEIGHBOR_PREDICATES = ("ontop",)


def _xy(p) -> th.Tensor:
    return (p if isinstance(p, th.Tensor) else th.as_tensor(p, dtype=th.float32))[:2].float()


def usable_cells(top, margin: float) -> int:
    """Free cells of ``top`` whose 8 neighbours ``margin`` away (the points of ``SupportTop.contains``) are free cells too."""
    free = top.free
    k = int(round(float(margin) / float(top.resolution)))
    if k <= 0:
        return int(free.sum())
    nx, ny = free.shape
    out = free.clone()
    for dx, dy in ((k, 0), (-k, 0), (0, k), (0, -k), (k, k), (-k, -k), (k, -k), (-k, k)):
        shifted = th.zeros_like(free)
        shifted[max(0, -dx):min(nx, nx - dx), max(0, -dy):min(ny, ny - dy)] = free[max(0, dx):min(nx, nx + dx), max(0, dy):min(ny, ny + dy)]
        out &= shifted
    return int(out.sum())


def _probe_points(xy, margin: float) -> List[th.Tensor]:
    """The points ``SupportTop.contains`` tests: ``xy`` and, with a margin, its 8 neighbours at ±margin."""
    p = _xy(xy)
    pts = [p]
    if margin > 0:
        for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin), (margin, -margin), (-margin, margin)):
            pts.append(p + th.tensor([dx, dy]))
    return pts


@dataclass
class UnionTop:
    """The support top united with its neighbours' tops; duck-types ``sim_utils.SupportTop`` (``name``, ``z``, ``yaw`` are the
    support's; the grid is the support's grid grown over the neighbours). ``member[i, j]`` = index into ``members`` of the top a
    cell belongs to (-1 off every top; the support's own top first, so its cells are exactly its own raster's)."""

    name: str
    xs: th.Tensor
    ys: th.Tensor
    mask: th.Tensor
    occupied: th.Tensor
    z: float
    yaw: float
    occupants: List[str]
    resolution: float
    member: th.Tensor
    members: List[Any] = field(default_factory=list)
    flush_tol: float = FLUSH_TOL_M
    edge_strip: Tuple[float, float] = EDGE_STRIP_M

    @staticmethod
    def build(members: Sequence[Any], flush_tol: float = FLUSH_TOL_M, edge_strip: Sequence[float] = EDGE_STRIP_M) -> "UnionTop":
        """Resample ``members`` (``SupportTop``, the support's first) onto the support's grid grown to cover all of them; a cell
        belongs to the first member whose raster has it (nearest cell of that member's grid)."""
        own = members[0]
        res = float(own.resolution)
        x0, y0 = float(own.xs[0]), float(own.ys[0])
        lo_x, hi_x = min(float(m.xs[0]) for m in members), max(float(m.xs[-1]) for m in members)
        lo_y, hi_y = min(float(m.ys[0]) for m in members), max(float(m.ys[-1]) for m in members)
        xs = x0 + res * th.arange(int(round((lo_x - x0) / res)), int(round((hi_x - x0) / res)) + 1, dtype=th.float32)
        ys = y0 + res * th.arange(int(round((lo_y - y0) / res)), int(round((hi_y - y0) / res)) + 1, dtype=th.float32)
        nx, ny = len(xs), len(ys)
        mask = th.zeros(nx, ny, dtype=th.bool)
        occupied = th.zeros(nx, ny, dtype=th.bool)
        member = th.full((nx, ny), -1, dtype=th.long)
        for k, m in enumerate(members):
            ii = th.round((xs - float(m.xs[0])) / res).long()
            jj = th.round((ys - float(m.ys[0])) / res).long()
            ix = th.nonzero((ii >= 0) & (ii < len(m.xs))).flatten()
            jx = th.nonzero((jj >= 0) & (jj < len(m.ys))).flatten()
            m_mask = th.zeros(nx, ny, dtype=th.bool)
            m_occ = th.zeros(nx, ny, dtype=th.bool)
            if len(ix) and len(jx):
                m_mask[ix[:, None], jx[None, :]] = m.mask[ii[ix][:, None], jj[jx][None, :]]
                m_occ[ix[:, None], jx[None, :]] = m.occupied[ii[ix][:, None], jj[jx][None, :]]
            claim = m_mask & (member < 0)
            member[claim] = k
            mask |= claim
            occupied[claim] = m_occ[claim]
            occupied |= m_occ & (member < 0)
        names = {m.name for m in members}
        occupants = sorted({o for m in members for o in (m.occupants or []) if o not in names})
        return UnionTop(own.name, xs, ys, mask, occupied, float(own.z), float(own.yaw), occupants, res, member, list(members), float(flush_tol),
                        (float(edge_strip[0]), float(edge_strip[1])))

    # -- SupportTop interface ----------------------------------------------------------------------------------------
    @property
    def own(self):
        """The support's own top."""
        return self.members[0]

    def neighbors_only(self) -> "UnionTop":
        """The same union with the support's own cells taken out of ``mask`` (``member``, the members and their heights unchanged): a
        footprint it supports lies on neighbour tops only (``generation.start_region_on_neighbors``: the target starts on a neighbour)."""
        return replace(self, mask=self.mask & (self.member > 0))

    @property
    def free(self) -> th.Tensor:
        return self.mask & ~self.occupied

    def cells(self, free_only: bool = True) -> th.Tensor:
        m = self.free if free_only else self.mask
        idx = th.nonzero(m)
        return th.stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]]], dim=1)

    def local_axes(self) -> Tuple[th.Tensor, th.Tensor]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return th.tensor([c, s]), th.tensor([-s, c])

    def local_bounds(self):
        """Bounds of the united tops in the support's local (yaw) frame: (umin, umax, vmin, vmax)."""
        pts = self.cells(free_only=False)
        ax, ay = self.local_axes()
        u, v = pts @ ax, pts @ ay
        return float(u.min()), float(u.max()), float(v.min()), float(v.max())

    def members_at(self, pts) -> th.Tensor:
        """Index into ``members`` of the top under each world point of ``pts`` (N, 2+) (any exposed cell, free or not; -1 off every
        top)."""
        p = th.as_tensor(pts, dtype=th.float32).reshape(-1, th.as_tensor(pts).shape[-1])[:, :2]
        i = th.round((p[:, 0] - self.xs[0]) / self.resolution).long()
        j = th.round((p[:, 1] - self.ys[0]) / self.resolution).long()
        ok = (i >= 0) & (j >= 0) & (i < len(self.xs)) & (j < len(self.ys))
        out = th.full((len(p),), -1, dtype=th.long)
        out[ok] = self.member[i[ok], j[ok]]
        return out

    def member_at(self, xy) -> Optional[int]:
        """Index into ``members`` of the top under world ``xy``; None off every top."""
        k = int(self.members_at(_xy(xy)[None, :])[0])
        return None if k < 0 else k

    def _free_members(self, points: Iterable[th.Tensor]) -> Optional[Set[int]]:
        """The tops under ``points`` when every point lies on a free cell, else None."""
        out: Set[int] = set()
        for p in points:
            i = int(th.round((p[0] - self.xs[0]) / self.resolution))
            j = int(th.round((p[1] - self.ys[0]) / self.resolution))
            if i < 0 or j < 0 or i >= len(self.xs) or j >= len(self.ys) or not bool(self.free[i, j]):
                return None
            out.add(int(self.member[i, j]))
        return out

    def flush(self, members: Iterable[int]) -> bool:
        """Whether the tops ``members`` (indices) are flush: heights within ``flush_tol`` of each other."""
        zs = [float(self.members[k].z) for k in members]
        return not zs or max(zs) - min(zs) <= self.flush_tol + 1e-9

    def contains(self, xy, margin: float = 0.0) -> bool:
        """``SupportTop.contains`` over the united tops: ``xy`` (and its 8 neighbours at ``margin``) on free cells of one top or of
        flush tops."""
        ms = self._free_members(_probe_points(xy, margin))
        return ms is not None and self.flush(ms)

    def polygon_supported(self, poly_xy: th.Tensor, margin: float = 0.02) -> bool:
        """``SupportTop.polygon_supported`` over the united tops: every vertex (with ``margin``) and the centroid on free cells,
        all of them on one top or across a seam of flush tops (a footprint never straddles a step)."""
        pts = [q for p in poly_xy for q in _probe_points(p, margin)] + [_xy(th.as_tensor(poly_xy, dtype=th.float32).mean(dim=0))]
        ms = self._free_members(pts)
        return ms is not None and self.flush(ms)

    def edges(self):
        """The support's own four sides (``SupportTop.edges`` of its own top: name, midpoint, outward normal, length), each
        extended along its line over the united cells that lie at most ``edge_strip[0]`` in front of it or ``edge_strip[1]``
        behind it: the robot stands at the support's edges, the lateral room counts the neighbour beside it."""
        ax, ay = self.local_axes()
        umin, umax, vmin, vmax = self.own.local_bounds()
        pts = self.cells(free_only=False)
        u, v = pts @ ax, pts @ ay
        front, depth = self.edge_strip
        out = []
        for name, line, inward, along, lo, hi, normal in (("+u", umax, umax - u, v, vmin, vmax, ax), ("-u", umin, u - umin, v, vmin, vmax, -ax),
                                                         ("+v", vmax, vmax - v, u, umin, umax, ay), ("-v", vmin, v - vmin, u, umin, umax, -ay)):
            sel = (inward >= -front) & (inward <= depth)
            if bool(sel.any()):
                lo, hi = min(lo, float(along[sel].min())), max(hi, float(along[sel].max()))
            mid_uv = (line, 0.5 * (lo + hi)) if name in ("+u", "-u") else (0.5 * (lo + hi), line)
            out.append((name, mid_uv[0] * ax + mid_uv[1] * ay, normal, hi - lo))
        return out

    def to_json(self):
        umin, umax, vmin, vmax = self.local_bounds()
        return {"name": self.name, "z": self.z, "yaw": self.yaw, "local_bounds": [umin, umax, vmin, vmax], "n_free_cells": int(self.free.sum()),
                "occupants": self.occupants, "resolution": self.resolution, "kind": "union", "flush_tol_m": self.flush_tol,
                "members": [{"name": m.name, "z": float(m.z), "step_m": round(float(m.z) - self.z, 4), "flush": self.flush([0, k]),
                             "n_cells": int((self.member == k).sum()), "n_free_cells": int((self.free & (self.member == k)).sum())}
                            for k, m in enumerate(self.members)]}

    # -- heights -----------------------------------------------------------------------------------------------------
    def z_at(self, xy) -> float:
        """Height of the top under world ``xy`` (the support's own top height off every top)."""
        k = self.member_at(xy)
        return float(self.members[k].z) if k is not None else float(self.z)

    def z_under(self, points) -> float:
        """Height an object whose footprint has the vertices ``points`` rests at: the highest top under them (flush tops differ by
        at most ``flush_tol``); the support's own height when none is under them."""
        ks = sorted({int(k) for k in self.members_at(points).tolist() if k >= 0})
        return max(float(self.members[k].z) for k in ks) if ks else float(self.z)

    def tops_under(self, points) -> List[str]:
        """Names of the tops under ``points``, in ``members`` order."""
        return [self.members[k].name for k in sorted({int(k) for k in self.members_at(points).tolist() if k >= 0})]


class NeighborBands:
    """The reach bands of a :class:`UnionTop`: one band per member top (``members[0]`` = the support's own). The reads at a stance
    (:meth:`cells_world`, :meth:`union_cells`) keep each band's cells on its own top; everything else reads the support's own band
    (attribute access falls through to it: ``meta``, ``path``, ``arm``, ``band``, ``to_world`` ...)."""

    def __init__(self, top: UnionTop, members: Sequence[Any]):
        self.top = top
        self.members = list(members)
        self.own = self.members[0]

    def __getattr__(self, name):
        own = self.__dict__.get("own")
        if own is None:
            raise AttributeError(name)
        return getattr(own, name)

    def cells_world(self, reader: Callable[[Any], th.Tensor]) -> th.Tensor:
        """The world cells ``reader(band)`` (``tabletop.band_cells_world`` of a single band at a stance) gives for each member band,
        each kept on its own top."""
        out = []
        for k, b in enumerate(self.members):
            w = reader(b)
            if len(w):
                w = w[self.top.members_at(w) == k]
                if len(w):
                    out.append(w)
        return th.cat(out) if out else th.zeros(0, 2)

    def union_cells(self, stance, reader: Callable[[Any], th.Tensor]) -> th.Tensor:
        """Base-frame cells at ``stance`` of every member band (``reader`` = ``tabletop.band_union_cells`` of a single band), each kept
        where it falls on its own top."""
        out = []
        for k, b in enumerate(self.members):
            base = reader(b)
            if not len(base):
                continue
            keep = self.top.members_at(th.tensor(b.to_world(base.numpy(), stance.base_pose2d), dtype=th.float32)) == k
            if bool(keep.any()):
                out.append(base[keep])
        return th.cat(out) if out else th.zeros(0, 2)

    def member_for(self, xy):
        """The band of the top under world ``xy`` (None off every top)."""
        k = self.top.member_at(xy)
        return None if k is None else self.members[k]

    def for_arm(self, arm: str, reader: Callable[[Any], Any]) -> "NeighborBands":
        """The same per-top bands for ``arm`` (``reader`` = ``tabletop.arm_band`` of a single band)."""
        if self.own.arm == arm:
            return self
        return NeighborBands(self.top, [reader(b) for b in self.members])

    def facts(self) -> List[Dict[str, Any]]:
        return [{"support": m.name, "path": getattr(b, "path", None), "arm": getattr(b, "arm", None)} for m, b in zip(self.top.members, self.members)]


# ----------------------------------------------------------------------------------------------------------------
# the hooks of tabletop / core (each the identity for a support without neighbours)
# ----------------------------------------------------------------------------------------------------------------


def union_top(s, support, top, spec: Optional[Dict[str, Any]], ignore: Sequence[Any] = ()):
    """``tabletop.support_surface``: the raster it chose (``top``) or - when the selection's ``support_surfaces`` entry ``spec`` names
    ``neighbors`` and ``top`` is the support's top - the :class:`UnionTop` of ``top`` and each neighbour's top (``s.support_top``, the
    same raster call). A neighbour that is not in the scene, has no top raster or fewer than ``generation.neighbor_min_free_cells``
    :func:`usable_cells` is left out; an interior surface (a shelf) is never united. Facts per neighbour go to
    ``template_facts["support_surface"]["neighbors"]``; ``s.neighbors_of`` = ``{support: [neighbour objects used]}``
    (:func:`neighbor_objects`, :func:`evaluate_on_tops`), reset on every call that sets it before."""
    if getattr(s, "neighbors_of", None):
        s.neighbors_of = {}
    names = [n for n in dict.fromkeys((spec or {}).get("neighbors") or []) if isinstance(n, str) and n and n != top.name]
    if not names:
        return top
    tf = getattr(s, "template_facts", None)
    fact = tf.get("support_surface") if isinstance(tf, dict) and isinstance(tf.get("support_surface"), dict) else {}
    if getattr(top, "headroom", None) is not None:
        fact["neighbors"] = [{"name": n, "used": False, "why": "interior surface"} for n in names]
        return top
    gen = getattr(s, "gen", lambda k, d=None: d)
    tol = float(gen("neighbor_flush_tol_m", FLUSH_TOL_M))
    min_cells = int(gen("neighbor_min_free_cells", MIN_FREE_CELLS))
    min_margin = float(gen("neighbor_min_margin_m", MIN_MARGIN_M))
    floor_z = float(s.world.floor_z)
    members, objs, facts = [top], [], []
    for name in names:
        rec: Dict[str, Any] = {"name": name}
        try:
            obj = s.world.obj(name)
            t = s.support_top(obj, ignore=ignore)
        except (KeyError, RuntimeError, ValueError) as e:
            rec.update({"used": False, "why": f"no top raster: {e}"[:200]})
            facts.append(rec)
            continue
        n_usable = usable_cells(t, min_margin)
        rec.update({"z_above_floor": round(float(t.z) - floor_z, 4), "step_m": round(float(t.z) - float(top.z), 4),
                    "flush": abs(float(t.z) - float(top.z)) <= tol + 1e-9, "n_free_cells": int(t.free.sum()), "n_usable_cells": n_usable})
        if n_usable < min_cells:
            rec.update({"used": False, "why": f"{n_usable} free cells with free cells {min_margin:.2f} m around them < {min_cells}"})
        else:
            rec["used"] = True
            members.append(t)
            objs.append(obj)
        facts.append(rec)
    fact["neighbors"] = facts
    variant = getattr(getattr(s, "gp", None), "variant", "?")
    if len(members) == 1:
        log.info("%s: no usable neighbour top for %s (%s); its own top", variant, top.name, facts)
        return top
    u = UnionTop.build(members, flush_tol=tol, edge_strip=gen("neighbor_edge_strip_m", EDGE_STRIP_M))
    s.neighbors_of = {top.name: objs}
    log.info("%s: %s united with %s: %d free cells (own %d), heights above the floor %s, flush tolerance %.3f m", variant, top.name,
             [m.name for m in members[1:]], int(u.free.sum()), int(top.free.sum()), [round(float(m.z) - floor_z, 3) for m in members], tol)
    return u


def own_top(top):
    """The support's own top of a union (bands and rasters are per support), else ``top``."""
    return top.own if isinstance(top, UnionTop) else top


def z_at(top, xy) -> float:
    """Height of the top under world ``xy``: ``top.z`` for a single top."""
    return top.z_at(xy) if isinstance(top, UnionTop) else float(top.z)


def rest_on(top, pose, footprint, z_used: float):
    """``(pose, support z)`` of an object posed at the support height ``z_used`` whose footprint is ``footprint``: on a union the
    pose moved to the highest top under the footprint; a single top keeps both."""
    if not isinstance(top, UnionTop):
        return pose, float(z_used)
    z = top.z_under(footprint)
    if abs(z - float(z_used)) <= 1e-9:
        return pose, float(z_used)
    return (pose[0] + th.tensor([0.0, 0.0, z - float(z_used)], dtype=pose[0].dtype), pose[1]), z


def grasp_from(top, footprint) -> Optional[str]:
    """The neighbour top an object with ``footprint`` starts on (the first in ``members`` order; None on the support's own top
    alone or on a single top): what ``grasps[].grasp_from`` records for the oracle."""
    if not isinstance(top, UnionTop):
        return None
    others = [n for n in top.tops_under(footprint) if n != top.name]
    return others[0] if others else None


def note_grasp_from(top, footprint, rec) -> None:
    """Record :func:`grasp_from` on a grasp record's ``extras`` (nothing on a single top or the support's own top)."""
    name = grasp_from(top, footprint)
    if name is not None:
        rec.extras["grasp_from"] = name


def neighbor_objects(s) -> List[Any]:
    """The neighbour supports the sampler's union top uses (``support_objects()``); [] without one."""
    return [o for objs in (getattr(s, "neighbors_of", None) or {}).values() for o in objs]


def evaluate_on_tops(s, predicate_cls, entities: Sequence[Any], evaluate: Callable[..., bool]) -> bool:
    """``evaluate(predicate_cls, *entities)`` where a support with neighbours (``s.neighbors_of``) stands for any of its tops in the
    :data:`NEIGHBOR_PREDICATES` (``(ontop target support)`` holds on a neighbour): the entities as given first."""
    alts = getattr(s, "neighbors_of", None) or {}
    name = str(getattr(predicate_cls, "__name__", predicate_cls)).lower()
    if not alts or name not in NEIGHBOR_PREDICATES or not any(getattr(e, "name", None) in alts for e in entities):
        return evaluate(predicate_cls, *entities)
    options = [[e] + list(alts.get(getattr(e, "name", None), [])) for e in entities]
    return any(evaluate(predicate_cls, *combo) for combo in itertools.product(*options))


def with_neighbor_bands(s, bands, spec, top, compute: Callable[..., Any]):
    """``tabletop.ensure_bands``: ``bands`` (the support's own) or, on a union, :class:`NeighborBands` with each neighbour's band of
    the same object / grasp key / placed objects / scale (``ReachBands.load`` by the neighbour's name; ``compute(s, spec,
    neighbour_top, None)`` = ``tabletop.compute_bands`` on a miss)."""
    if not isinstance(top, UnionTop):
        return bands
    scene = s.world.scene_model
    gkey = "generic" if spec.generic_pinch else spec.grasp_key
    members = [bands]
    for m in top.members[1:]:
        try:
            b = RB.ReachBands.load(scene, m.name, spec.object, gkey, placed=spec.placed, scale=spec.scale)
            log.info("reach band cache hit for neighbour %s / %s (%s): %s", m.name, spec.object, gkey, b.path)
        except FileNotFoundError:
            log.info("reach band cache miss for neighbour %s / %s (%s): computing in-process", m.name, spec.object, gkey)
            b = compute(s, spec, m, None)
        members.append(b)
    nb = NeighborBands(top, members)
    tf = getattr(s, "template_facts", None)
    if isinstance(tf, dict):
        tf["neighbor_bands"] = nb.facts()
    return nb


__all__ = ["FLUSH_TOL_M", "MIN_FREE_CELLS", "MIN_MARGIN_M", "EDGE_STRIP_M", "NEIGHBOR_PREDICATES", "UnionTop", "NeighborBands", "usable_cells",
           "union_top", "own_top", "z_at", "rest_on", "grasp_from", "note_grasp_from", "neighbor_objects", "evaluate_on_tops", "with_neighbor_bands"]
