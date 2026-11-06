"""Band refinement (train-fast, owner design 2026-11-01, ``ISPATIALGYM_FAST_BAND_REFINE``; off = BAND_EVIDENCE as before): bands get finer
as sampling learns, and only verified reachability failures remove anything from them.

Keying: a stance's ACTUAL trunk ``(translate, tilt)`` rounded to 0.01 (:func:`trunk_key`), not the nearest grid trunk of the band
(``ReachBands.nearest_trunk``: a stance at 0.25 / -1.07 read 0.25 / -1.2's cells - the inside_02 failure). Phase 2's tilt scan is a
finite set per template, so the keys stay bounded.

A key without refined data starts from the band (:func:`init_cells`, ``ISPATIALGYM_FAST_BAND_REFINE_INIT``): ``nearest`` (default since
2026-11-01, owner after the refine A/B) a copy of the nearest grid trunk; ``bracket`` the cells both grid tilts around the key's tilt hold
(the TILT_SNAP bracket rule: no first failures paid at the far edge), the grid trunk's own cells on a grid tilt, the nearest grid trunk's
outside the grid's tilt range - on ontop_small_01 it dropped the far cells that succeed and wrote about half as many demos. The
refined cell set of a key (:func:`refined_cells`, what ``ReachBands.band`` returns when the switch is on) = that start - the blacklisted
cells + the added ones.

What counts (``tabletop.phase3_target``): a screen that comes back empty, or a drawn pair whose exact IK fails, is re-checked with the
task-spawned objects out of the collision world (the static scene the band was computed with stays); only if it still fails is it a
reachability failure and recorded. The exact IK's own rules count when they persist that way - ``exact IK failed``, ``ik_joint_limit``
(the joint-limit margin), ``ik_reach`` (the witness residual), ``approach_untrackable`` - all properties of the arm at that stance and
pose. Visibility, support, placement, repeats_eval and collisions with task objects never reach the store. A cell is blacklisted after
:data:`K` verified empty screens with no pair ever found; a pair after K verified failures (and more than twice its confirmations).
Additions: one attempt in :data:`ADD_P` screens a lattice neighbour of the focus cell missing from the key's set (task objects out); a
pair found adds the cell.

Storage: ``<band>.refine.json`` (the job's own records) next to the job's band copy, ``<band>.refine.base.json`` (fetched); shipped
band files are never written. Verified reachability does not depend on the world (crop / full, distractor layout), so every record is
under the one world key :data:`WORLD`; sharing between jobs is ``band_evidence.fetch`` / ``publish`` (per-job files, merged as a
union) behind ``ISPATIALGYM_FAST_BAND_SHARE`` (default off)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.sampling.v2 import band_evidence as EV

WORLD = "reach"
K = 2
ADD_P = 0.1
INIT = "nearest"


def _fv(name: str, default, cast):
    from ispatialgym.fastflags import fast_value

    return fast_value(name, default, cast)


def refine_on() -> bool:
    from ispatialgym.fastflags import fast_on

    return fast_on("BAND_REFINE")


def k() -> int:
    return int(_fv("BAND_REFINE_K", K, int))


def add_p() -> float:
    return float(_fv("BAND_REFINE_ADD_P", ADD_P, float))


def init_mode() -> str:
    return str(_fv("BAND_REFINE_INIT", INIT, str)).strip().lower()


def trunk_key(translate: float, tilt: float) -> Tuple[float, float]:
    return round(float(translate), 2), round(float(tilt), 2)


def cell_key(trunk: Tuple[float, float], cell_base: Sequence[float]) -> str:
    return EV.cell_key(trunk_key(*trunk), cell_base)


def _ck(c) -> Tuple[float, float]:
    return round(float(c[0]), 3), round(float(c[1]), 3)


class Refinement(EV.Evidence):
    """The refinement records of one band file (own + fetched base, see the module doc)."""

    own_suffix = ".refine.json"
    base_suffix = ".refine.base.json"

    def add_cell(self, cell: str, found: Sequence[Tuple[float, str]]) -> None:
        """A lattice cell found reachable at a key's trunk (an addition): the pairs the screen passed."""
        self.version += 1
        rec = self._own_rec(WORLD, cell)
        rec["added"] = True
        rec["screens"] += 1
        for yaw, g in found:
            rec["pairs"].setdefault(EV.pair_key(yaw, g), [0, 0, 0])[0] += 1
        self.save()

    def cells_of(self, trunk: Tuple[float, float]) -> Dict[Tuple[float, float], Dict[str, Any]]:
        """``{cell (base frame, mm-rounded): merged record}`` of a trunk key."""
        prefix = "%.3f,%.3f|" % trunk_key(*trunk)
        out: Dict[Tuple[float, float], Dict[str, Any]] = {}
        for table in (self.base, self.own):
            for ck, rec in (table.get(WORLD) or {}).items():
                if ck.startswith(prefix):
                    xy = tuple(float(v) for v in ck[len(prefix):].split(","))
                    out[_ck(xy)] = EV.merge_records(out.get(_ck(xy), EV.new_record()), rec)
        return out

    def seen(self, trunk: Tuple[float, float]) -> bool:
        return bool(self.cells_of(trunk))


def store_of(bands) -> Refinement:
    """The (cached) :class:`Refinement` of a loaded band object."""
    r = getattr(bands, "_refinement", None)
    if r is None:
        r = Refinement(getattr(bands, "path", None))
        try:
            bands._refinement = r
        except AttributeError:
            pass
    return r


def dead(rec: Optional[Dict[str, Any]]) -> bool:
    return EV.cell_dead(rec, m=k())


def init_cells(bands, translate: float, tilt: float, mode: Optional[str] = None) -> Optional[np.ndarray]:
    """The start cells of a key (module doc), base frame ``(N, 2)``; None when the band has no trunk at that translate."""
    mode = init_mode() if mode is None else mode
    grid = sorted((float(t), key) for key in bands.cells for t in [key[1]] if abs(float(key[0]) - float(translate)) < 1e-6)
    if not grid:
        return None
    tk = trunk_key(translate, tilt)
    on = [key for t, key in grid if abs(t - tk[1]) < 0.005]
    if on:
        return np.asarray(bands.cells[on[0]], dtype=float).reshape(-1, 2)
    nearest = min(grid, key=lambda g: abs(g[0] - tk[1]))[1]
    if mode != "bracket":
        return np.asarray(bands.cells[nearest], dtype=float).reshape(-1, 2)
    lo = [key for t, key in grid if t < tk[1]]
    hi = [key for t, key in grid if t > tk[1]]
    if not lo or not hi:
        return np.asarray(bands.cells[nearest], dtype=float).reshape(-1, 2)
    a, b = np.asarray(bands.cells[lo[-1]], dtype=float).reshape(-1, 2), np.asarray(bands.cells[hi[0]], dtype=float).reshape(-1, 2)
    keep_b = {_ck(c) for c in b}
    return np.asarray([c for c in a if _ck(c) in keep_b], dtype=float).reshape(-1, 2)


def refined_cells(bands, translate: float, tilt: float) -> Optional[np.ndarray]:
    """The refined cells of the key of ``(translate, tilt)``: :func:`init_cells` - blacklisted (:func:`dead`) + added; None without a
    trunk at that translate. Cached per key and store version."""
    store = store_of(bands)
    tk = trunk_key(translate, tilt)
    cache = bands.__dict__.setdefault("_refined_cache", {}) if hasattr(bands, "__dict__") else {}
    hit = cache.get(tk)
    if hit is not None and hit[0] == store.version:
        return hit[1]
    base = init_cells(bands, translate, tilt)
    if base is None:
        return None
    recs = store.cells_of(tk)
    out = [c for c in base if not dead(recs.get(_ck(c)))]
    have = {_ck(c) for c in out}
    for xy, rec in sorted(recs.items()):
        if rec.get("added") and not dead(rec) and xy not in have:
            out.append(np.asarray(xy, dtype=float))
            have.add(xy)
    arr = np.asarray(out, dtype=float).reshape(-1, 2)
    cache[tk] = (store.version, arr)
    return arr


def neighbours(cells: np.ndarray, cell: Sequence[float], lattice: float) -> List[Tuple[float, float]]:
    """The 4 lattice neighbours of ``cell`` that are not in ``cells``."""
    have = {_ck(c) for c in cells}
    out = []
    for dx, dy in ((lattice, 0.0), (-lattice, 0.0), (0.0, lattice), (0.0, -lattice)):
        c = _ck((float(cell[0]) + dx, float(cell[1]) + dy))
        if c not in have:
            out.append(c)
    return out
