"""Band evidence (train-fast 2026-11-01, ``ISPATIALGYM_FAST_BAND_EVIDENCE``, default on): what the samplers learned about a band's focus
cells, kept as counts instead of verdicts. Replaces ``BAND_FEASIBLE_REUSE`` (one stored feasible set per (trunk, cell), reused as is,
no fallback when it ran dry) and, while on, the base code's band write-back (``tabletop.store_feasible_set``: a cell deleted from the
band the first time its randomised screen came back empty).

Per world (:func:`world_key`: scene, loaded rooms, scene crop - a cropped and a full world never read each other's counts) and band
cell (trunk + the cell's lattice coordinates), a record holds

* ``screens`` / ``empty``: batched-IK screens of the cell and how many found nothing;
* ``pairs``: per ``"<yaw>|<grasp>"`` ``[screen_hits, exact_ok, exact_fail]``.

The feasible set of a cell is the **union** of every pair a screen ever passed (it grows with re-screens) minus the pairs the exact IK
demoted (:data:`PAIR_FAIL_K` failures and more than twice its confirmations); a cell is dead only after :data:`CELL_EMPTY_M` empty
screens with no pair ever found. Positives are kept freely; negatives only count with that much evidence.

Files: the job's own counts in ``<band>.evidence.json`` next to the job's band copy (atomic writes, only the job writes it); counts
fetched from the share in ``<band>.evidence.base.json`` (read only). The share (``ISPATIALGYM_FAST_BAND_SHARE``, **default off**:
the contamination of 2026-11-01, when concurrent jobs rewrote one committed band) is a directory of per-job files
``<band>.evidence.<job>.json``: ``publish`` copies a job's own counts there at its end, ``fetch`` merges them all into a job's base
file at its start (no file is ever written by two jobs, which NFS without locks needs).

    python -m ispatialgym.sampling.v2.band_evidence fetch <job band dir> <share dir>
    python -m ispatialgym.sampling.v2.band_evidence publish <job band dir> <share dir> <job id>
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("ispatialgym.sampling.v2")

#: a pair is demoted after this many exact-IK failures (and more failures than twice its confirmations)
PAIR_FAIL_K = 3
#: a cell is dead after this many empty screens with no pair ever found
CELL_EMPTY_M = 3
#: share of attempts that re-screen a cell with a known feasible set (borderline pairs get another chance; the set grows)
RESCREEN_P = 0.25

OWN_SUFFIX = ".evidence.json"
BASE_SUFFIX = ".evidence.base.json"


def _fast_value(name: str, default, cast):
    from ispatialgym.fastflags import fast_value

    return fast_value(name, default, cast)


def pair_fail_k() -> int:
    return int(_fast_value("BAND_PAIR_FAIL_K", PAIR_FAIL_K, int))


def cell_empty_m() -> int:
    return int(_fast_value("BAND_CELL_EMPTY_M", CELL_EMPTY_M, int))


def rescreen_p() -> float:
    return float(_fast_value("BAND_RESCREEN_P", RESCREEN_P, float))


# ----------------------------------------------------------------------------------------------------------------
# keys and records (pure)
# ----------------------------------------------------------------------------------------------------------------


def world_key(scene: str, rooms: Sequence[str], crop: Optional[Dict[str, Any]] = None) -> str:
    c = crop or {}
    mode = c.get("mode", "full")
    tag = mode if mode == "full" else f"{mode}:{c.get('center_object')}:{c.get('radius_m')}"
    return f"{scene}|{','.join(sorted(rooms or []))}|{tag}"


def cell_key(trunk: Tuple[float, float], cell_xy: Sequence[float]) -> str:
    return f"{float(trunk[0]):.3f},{float(trunk[1]):.3f}|{float(cell_xy[0]):.3f},{float(cell_xy[1]):.3f}"


def pair_key(yaw: float, grasp_name: str) -> str:
    return f"{float(yaw):.3f}|{grasp_name}"


def new_record() -> Dict[str, Any]:
    return {"screens": 0, "empty": 0, "pairs": {}}


def merge_records(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = {"screens": int(a.get("screens", 0)) + int(b.get("screens", 0)), "empty": int(a.get("empty", 0)) + int(b.get("empty", 0)), "pairs": {}}
    for flag in ("unreachable", "added"):
        if a.get(flag) or b.get(flag):
            out[flag] = True
    for src in (a, b):
        for k, v in (src.get("pairs") or {}).items():
            cur = out["pairs"].setdefault(k, [0, 0, 0])
            for j in range(3):
                cur[j] += int(v[j]) if j < len(v) else 0
    return out


def merge_tables(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{world: {cell: record}}`` tables summed."""
    out: Dict[str, Dict[str, Any]] = {}
    for src in (a, b):
        for w, cells in (src or {}).items():
            dst = out.setdefault(w, {})
            for c, rec in cells.items():
                dst[c] = merge_records(dst.get(c, new_record()), rec)
    return out


def demoted(counts: Sequence[int], k: Optional[int] = None) -> bool:
    hits, ok, fail = (list(counts) + [0, 0, 0])[:3]
    return fail >= (pair_fail_k() if k is None else int(k)) and fail > 2 * ok


def feasible_pairs(rec: Optional[Dict[str, Any]], k: Optional[int] = None) -> List[Tuple[float, str]]:
    """The cell's feasible set: every pair a screen passed, minus the demoted ones (``k`` failures, default :func:`pair_fail_k`),
    as ``(yaw, grasp name)``."""
    out = []
    for key, v in sorted(((rec or {}).get("pairs") or {}).items()):
        if int(v[0]) > 0 and not demoted(v, k):
            yaw, g = key.split("|", 1)
            out.append((float(yaw), g))
    return out


def cell_dead(rec: Optional[Dict[str, Any]], m: Optional[int] = None) -> bool:
    if not rec:
        return False
    if rec.get("unreachable"):
        return True  # TILT_SNAP: outside the band at a grid tilt next to the stance's tilt (geometry, not a screening miss)
    ever = any(int(v[0]) > 0 for v in (rec.get("pairs") or {}).values())
    return not ever and int(rec.get("empty", 0)) >= (cell_empty_m() if m is None else int(m))


# ----------------------------------------------------------------------------------------------------------------
# the job's files
# ----------------------------------------------------------------------------------------------------------------


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, sort_keys=True)
    os.replace(tmp, path)


class Evidence:
    """The evidence of one band file: the fetched base (read only) and this job's own counts (written back atomically)."""

    own_suffix = OWN_SUFFIX
    base_suffix = BASE_SUFFIX

    def __init__(self, band_path: Optional[str]):
        self.stem = band_path[:-4] if band_path and band_path.endswith(".npz") else band_path
        self.base = _read(self.stem + self.base_suffix) if self.stem else {}
        self.own = _read(self.stem + self.own_suffix) if self.stem else {}
        self.version = 0  # bumped on every change (caches of derived data compare it)

    def record(self, world: str, cell: str) -> Optional[Dict[str, Any]]:
        a = (self.base.get(world) or {}).get(cell)
        b = (self.own.get(world) or {}).get(cell)
        if a is None and b is None:
            return None
        return merge_records(a or new_record(), b or new_record())

    def _own_rec(self, world: str, cell: str) -> Dict[str, Any]:
        return self.own.setdefault(world, {}).setdefault(cell, new_record())

    def add_screen(self, world: str, cell: str, found: Sequence[Tuple[float, str]]) -> None:
        self.version += 1
        rec = self._own_rec(world, cell)
        rec["screens"] += 1
        if not found:
            rec["empty"] += 1
        for yaw, g in found:
            rec["pairs"].setdefault(pair_key(yaw, g), [0, 0, 0])[0] += 1
        self.save()

    def add_exact(self, world: str, cell: str, yaw: float, grasp: str, ok: bool) -> None:
        self.version += 1
        rec = self._own_rec(world, cell)
        rec["pairs"].setdefault(pair_key(yaw, grasp), [0, 0, 0])[1 if ok else 2] += 1
        self.save()

    def mark_unreachable(self, world: str, cell: str) -> None:
        """TILT_SNAP (2026-11-01): the cell is out of reach at this (stance) trunk - dead at once, never re-screened."""
        self.version += 1
        self._own_rec(world, cell)["unreachable"] = True
        self.save()

    def save(self) -> None:
        if not self.stem:
            return
        try:
            _write_atomic(self.stem + self.own_suffix, self.own)
        except OSError as e:  # noqa: BLE001 - a read-only band directory is not a sampling error
            log.debug("band evidence write skipped (%r)", e)


def evidence_of(bands) -> Evidence:
    """The (cached) :class:`Evidence` of a loaded band object."""
    ev = getattr(bands, "_evidence", None)
    if ev is None:
        ev = Evidence(getattr(bands, "path", None))
        try:
            bands._evidence = ev
        except AttributeError:
            pass
    return ev


# ----------------------------------------------------------------------------------------------------------------
# the share (default off)
# ----------------------------------------------------------------------------------------------------------------


#: the shared stores: (kind, own suffix, base suffix) - the band evidence and the band refinement (band_refine)
KINDS = (("evidence", OWN_SUFFIX, BASE_SUFFIX), ("refine", ".refine.json", ".refine.base.json"))


def fetch(job_dir: str, share_dir: str) -> int:
    """Merge every job's published counts in ``share_dir`` into ``<band>.<kind>.base.json`` of the bands in ``job_dir`` (a union: counts
    summed, flags or-ed - each job's file holds only its own counts, so nothing is counted twice)."""
    n = 0
    for npz in glob.glob(os.path.join(job_dir, "*.npz")):
        name = os.path.basename(npz)[:-4]
        for kind, _own, base in KINDS:
            table: Dict[str, Any] = {}
            for p in sorted(glob.glob(os.path.join(share_dir, f"{name}.{kind}.*.json"))):
                table = merge_tables(table, _read(p))
            if table:
                _write_atomic(os.path.join(job_dir, name + base), table)
                n += 1
    return n


def publish(job_dir: str, share_dir: str, job_id: str) -> int:
    """Copy each band's own counts of the job to ``share_dir/<band>.<kind>.<job>.json`` (the base is never re-published)."""
    os.makedirs(share_dir, exist_ok=True)
    n = 0
    for kind, own_sfx, _base in KINDS:
        for p in glob.glob(os.path.join(job_dir, "*" + own_sfx)):
            if own_sfx == OWN_SUFFIX and p.endswith(".refine.json"):
                continue
            own = _read(p)
            if not own:
                continue
            name = os.path.basename(p)[: -len(own_sfx)]
            _write_atomic(os.path.join(share_dir, f"{name}.{kind}.{job_id}.json"), own)
            n += 1
    return n


def main(argv: Sequence[str]) -> int:
    if len(argv) >= 3 and argv[0] == "fetch":
        print(f"BAND_EVIDENCE fetch: {fetch(argv[1], argv[2])} band(s) with shared counts")
        return 0
    if len(argv) >= 4 and argv[0] == "publish":
        print(f"BAND_EVIDENCE publish: {publish(argv[1], argv[2], argv[3])} band(s)")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
