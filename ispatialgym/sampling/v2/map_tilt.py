"""The expanding, self-correcting reach map (i7-plateqtf, owner decision 2026-11-07 23:40, item 3; instead of TILT_SNAP, which stays opt-in).

A reach map (:mod:`reach_maps`) holds cells per BUILT trunk tilt (``reach_bands.TILTS``: 0, -0.3, -0.6, -0.9) and per object yaw slot
(8 x 45 deg). Phase 2's stances take continuous tilts (``tilt_scan``), so a map view used to read the NEAREST built tilt's cells - a
stance at 0.25 / -0.53 read 0.25 / -0.6's cells, and a cell that is reachable at -0.6 but not at -0.53 came back as an empty feasible set
at every attempt (under_small_03: 33-37 / 60, kinematic).

**Inherited columns.** A stance tilt that is not a built tilt (more than :data:`EPS_TILT` from each) and lies between two built tilts of
the same translate gets a PROVISIONAL column, keyed by the tilt rounded to 0.01 rad: the UNION of the two bracketing built columns (a
cell is provisionally feasible if either neighbour holds it; its (yaw, grasp) pairs are the union too), marked ``inherited``. A tilt on
a built tilt, or outside the built range (one neighbour only), reads the map as before. The same rule for yaws: an object yaw relative
to the base between two of the map's yaw slots inherits the union of the two slots' grasps (:func:`yaw_union`); a slot yaw is unchanged.

**Corrections.** Only REACHABILITY failures count against an inherited entry: the batched screen of the focus comes back empty and stays
empty with every scene object out of the collision world, or an exact IK failure persists with the scene out and names IK / the reach
bound / a joint limit (:func:`reach_reason`). Scene collisions, the support policy and the camera frame never count (with the scene out
they cannot fail, or they are not IK). A cell (or yaw pair) is INVALIDATED for that column after :func:`k` such failures with no success
(IK seeds are random: one failure is not enough); one exact success CONFIRMS it, and a confirmed entry is never invalidated.

**Overlay, never the frozen map.** The map's ``.json`` / ``.npz`` are never written. Corrections are events in an overlay keyed by the
map identity (the map's ``key``):

* each process appends its events to its OWN shard file ``<shards dir>/<map key>/<host>_<pid>.jsonl`` (``ISPATIALGYM_MAP_OVERLAY_SHARDS``;
  the fast flow's job script sets ``$JOBDIR/map_overlay``) - one writer per file, so concurrent jobs never race;
* :func:`merge` (offline, ``python -m ispatialgym.sampling.v2.map_tilt merge``) folds shards into ``<overlay root>/<map key>/v<N>.json``,
  a NEW version file each time (written to a temp file and renamed; the shards it folded are listed in ``sources`` and skipped by the next
  merge);
* a job reads a merged version only when ``ISPATIALGYM_MAP_OVERLAY`` names the overlay root (latest version, or
  ``ISPATIALGYM_MAP_OVERLAY_VERSION``); the version then enters the view's identity (:func:`view_tag`: ``inherit+ov<N>``), which the
  start-cell caches key on. In-process corrections apply at once (Phase 2 drops invalidated cells; a cached candidate whose focus cell
  is invalidated leaves the draws).

No true column is ever built here (no GPU): the union prior, pruned during normal sampling, is the whole mechanism.

Switch: ``ISPATIALGYM_FAST_MAP_TILT_INHERIT`` (:func:`on`; default on since the i7-plateqtf default commit, ``=0`` off). ``ISPATIALGYM_FAST_MAP_TILT_K`` (default :data:`K`).
Kit counters (per attempt; plus ``map_tilt.by_map.<key>.<name>``): ``map_tilt.inherited_columns`` / ``.inherited_feasible`` (cells of
the new columns) / ``.cells_invalidated`` / ``.cells_confirmed`` / ``.empty_f_on_inherited`` / ``.reach_fail`` / ``.other_fail`` /
``.skipped_invalidated``; ``map_yaw.pairs_confirmed`` / ``.pairs_invalidated``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "isg-map-overlay/1"
EPS_TILT = 0.005  # rad: a stance tilt this close to a built tilt reads the built column
K = 2  # reachability failures (no success) before an inherited entry is invalidated
ENV_SHARDS = "ISPATIALGYM_MAP_OVERLAY_SHARDS"
ENV_ROOT = "ISPATIALGYM_MAP_OVERLAY"
ENV_VERSION = "ISPATIALGYM_MAP_OVERLAY_VERSION"
#: exact-IK failure reasons that are reachability (IK / reach bound / joint limit); anything else (collisions, the support policy, the
#: approach tracking against an obstacle, the camera frame) never counts against an inherited entry
REACH_REASONS = ("exact ik failed", "ik failed", "ik_reach", "reach", "joint limit", "joint_limit", "ik_joint_limit", "no ik")
NOT_REACH = ("support_policy", "collid", "collision", "frame", "camera", "touches the support", "occlu")


def on() -> bool:
    """i7-plateqtf 2026-11-07: default ON (the owner's mechanism instead of TILT_SNAP); ``ISPATIALGYM_FAST_MAP_TILT_INHERIT=0`` = the
    nearest built tilt as before."""
    from ispatialgym.fastflags import fast_on

    return fast_on("MAP_TILT_INHERIT", default=True)


def k() -> int:
    from ispatialgym.fastflags import fast_value

    return max(1, int(fast_value("MAP_TILT_K", K, int)))


def tilt_key(tilt: float) -> float:
    return round(float(tilt), 2)


def reach_reason(reason: str) -> bool:
    """Whether an exact-IK failure reason is a reachability failure (IK / reach bound / joint limit) and not a collision, the support
    policy or the camera frame."""
    r = str(reason or "").lower()
    if not r or any(w in r for w in NOT_REACH):
        return False
    return any(w in r for w in REACH_REASONS)


# ------------------------------------------------------------------------------------------------ inherited columns (pure)


@dataclass
class Column:
    """A column of a map view: the cells / pairs / classes one (stand-off, translate, tilt) query reads."""

    key: Tuple[float, float]
    cells: np.ndarray
    pairs: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    inherited: bool = False
    brackets: Tuple[Tuple[float, float], ...] = ()


def _ck(c) -> Tuple[float, float]:
    return round(float(c[0]), 3), round(float(c[1]), 3)


def bracket(keys: Iterable[Tuple[float, float]], translate: float, tilt: float, eps: float = EPS_TILT
            ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The two built trunks of ``translate`` whose tilts bracket ``tilt`` strictly (each more than ``eps`` away), else None (a built
    tilt, outside the built range, or no trunk of that translate)."""
    same = sorted((float(k[1]), k) for k in keys if abs(float(k[0]) - float(translate)) < 1e-6)
    if not same or any(abs(t - float(tilt)) <= eps for t, _ in same):
        return None
    lo = [kk for t, kk in same if t < float(tilt)]
    hi = [kk for t, kk in same if t > float(tilt)]
    if not lo or not hi:
        return None
    return lo[-1], hi[0]


def union_column(cells: Dict[Tuple[float, float], np.ndarray], pairs: Dict[Tuple[float, float], np.ndarray],
                 classes: Dict[Tuple[float, float], np.ndarray], translate: float, tilt: float, eps: float = EPS_TILT,
                 all_keys: Optional[Iterable[Tuple[float, float]]] = None) -> Optional[Column]:
    """The inherited column of ``(translate, tilt)``: the union of the two bracketing built columns (module doc), or None when the tilt
    is built / outside the range. ``all_keys``: every built trunk of the map at this stand-off (an ineligible trunk - its idle arm
    collides at this stand-off - is a bracket that holds no cells); default the keys of ``cells``."""
    br = bracket(all_keys if all_keys is not None else cells.keys(), translate, tilt, eps)
    if br is None:
        return None
    order: List[Tuple[float, float]] = []
    index: Dict[Tuple[float, float], int] = {}
    for b in br:
        for c in np.asarray(cells.get(b, np.zeros((0, 2))), dtype=float).reshape(-1, 2):
            ck = _ck(c)
            if ck not in index:
                index[ck] = len(order)
                order.append(ck)
    out_cells = np.asarray(order, dtype=float).reshape(-1, 2)
    out_pairs = None
    if any(b in pairs for b in br):
        seen = set()
        rows = []
        for b in br:
            src = np.asarray(cells.get(b, np.zeros((0, 2))), dtype=float).reshape(-1, 2)
            p = pairs.get(b)
            if p is None:
                continue
            for ci, yi, gi in np.asarray(p, dtype=np.int64).reshape(-1, 3).tolist():
                j = index[_ck(src[ci])]
                if (j, yi, gi) not in seen:
                    seen.add((j, yi, gi))
                    rows.append((j, yi, gi))
        out_pairs = np.asarray(sorted(rows), dtype=np.int32).reshape(-1, 3)
    out_classes = None
    if any(b in classes for b in br):
        bits = np.zeros(len(order), dtype=np.int64)
        for b in br:
            src = np.asarray(cells.get(b, np.zeros((0, 2))), dtype=float).reshape(-1, 2)
            m = classes.get(b)
            if m is None or len(m) != len(src):
                continue
            for c, v in zip(src, np.asarray(m).tolist()):
                bits[index[_ck(c)]] |= int(v)
        out_classes = bits
    return Column((round(float(translate), 2), tilt_key(tilt)), out_cells, out_pairs, out_classes, True, tuple(br))


def prune(col: Column, dead_cells: Iterable[Tuple[float, float]]) -> Column:
    """``col`` without the cells in ``dead_cells`` (mm-rounded base-frame xy): pairs / classes re-indexed."""
    dead = set(dead_cells)
    if not dead or not len(col.cells):
        return col
    keep = [i for i, c in enumerate(col.cells) if _ck(c) not in dead]
    remap = {old: new for new, old in enumerate(keep)}
    pairs = None
    if col.pairs is not None:
        pairs = np.asarray([(remap[c], y, g) for c, y, g in np.asarray(col.pairs).reshape(-1, 3).tolist() if c in remap],
                           dtype=np.int32).reshape(-1, 3)
    classes = None if col.classes is None else np.asarray(col.classes)[keep]
    return Column(col.key, col.cells[keep].reshape(-1, 2), pairs, classes, col.inherited, col.brackets)


def yaw_bracket(yaws: Sequence[float], psi: float, eps: float = 1e-3) -> Optional[Tuple[int, int]]:
    """The two yaw slots (indices into ``yaws``, a full turn) around the relative yaw ``psi``; None when ``psi`` is on a slot."""
    n = len(yaws)
    if n == 0:
        return None
    two_pi = 2 * math.pi
    rel = [((float(psi) - float(y)) % two_pi) for y in yaws]
    for r in rel:
        if min(r, two_pi - r) <= eps:
            return None
    lo = min(range(n), key=lambda i: rel[i])  # the slot just below psi (smallest positive offset)
    hi = min(range(n), key=lambda i: (two_pi - rel[i]) % two_pi)  # the slot just above
    return lo, hi


def yaw_union(status_cell: np.ndarray, yaws: Sequence[float], psi: float, feasible_code: int = 1) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """``(feasible per grasp, the bracketing slots)`` of a map cell's status ``[n_yaws, n_grasps]`` at the relative yaw ``psi``: on a slot
    that slot's row unchanged (slots None), between two slots the union of their rows (inherited)."""
    st = np.asarray(status_cell) == int(feasible_code)
    br = yaw_bracket(yaws, psi)
    if br is None:
        two_pi = 2 * math.pi
        i = min(range(len(yaws)), key=lambda j: min((float(psi) - float(yaws[j])) % two_pi, (float(yaws[j]) - float(psi)) % two_pi))
        return st[i].copy(), None
    return st[br[0]] | st[br[1]], br


# ------------------------------------------------------------------------------------------------ the overlay (corrections)


def col_id(stand_off: Optional[float], key: Tuple[float, float]) -> str:
    so = "-" if stand_off is None else f"{float(stand_off):.2f}"
    return f"so{so}|t{float(key[0]):.2f}|k{float(key[1]):+.2f}"


def cell_id(cell) -> str:
    c = _ck(cell)
    return f"{c[0]:.3f},{c[1]:.3f}"


def yaw_id(psi: float, grasp: str) -> str:
    return f"y{(float(psi) + math.pi) % (2 * math.pi) - math.pi:+.2f}|{grasp}"


def new_rec() -> Dict[str, int]:
    return {"reach_fail": 0, "other_fail": 0, "ok": 0}


def is_dead(rec: Optional[Dict[str, int]], kk: Optional[int] = None) -> bool:
    if not rec:
        return False
    return int(rec.get("ok", 0)) == 0 and int(rec.get("reach_fail", 0)) >= (k() if kk is None else int(kk))


def is_confirmed(rec: Optional[Dict[str, int]]) -> bool:
    return bool(rec) and int(rec.get("ok", 0)) > 0


@dataclass
class Overlay:
    """The corrections of one map identity: the merged base version (read-only) + this process's own events (appended to its shard)."""

    map_key: str
    base: Dict[str, Dict[str, int]] = field(default_factory=dict)
    base_version: int = 0
    own: Dict[str, Dict[str, int]] = field(default_factory=dict)
    shard_path: Optional[str] = None
    version: int = 0  # bumps on every own event (caches of pruned columns key on it)
    inherited_cols: set = field(default_factory=set)

    def rec(self, entry: str) -> Dict[str, int]:
        out = new_rec()
        for t in (self.base, self.own):
            r = t.get(entry)
            if r:
                for kk in out:
                    out[kk] += int(r.get(kk, 0))
        return out

    def dead(self, entry: str) -> bool:
        return is_dead(self.rec(entry))

    def record(self, entry: str, outcome: str, **facts) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Add one outcome (``ok`` / ``reach_fail`` / ``other_fail``) of ``entry`` (``<col id>#<cell id>[#<yaw id>]``): ``(before, after)``
        merged records. Appended to the shard (one JSON line) when a shard path is set."""
        assert outcome in ("ok", "reach_fail", "other_fail"), outcome
        before = self.rec(entry)
        r = self.own.setdefault(entry, new_rec())
        r[outcome] += 1
        self.version += 1
        if self.shard_path:
            line = {"schema": SCHEMA, "map_key": self.map_key, "base_version": self.base_version, "entry": entry, "outcome": outcome,
                    "t": round(time.time(), 3), **facts}
            try:
                os.makedirs(os.path.dirname(self.shard_path), exist_ok=True)
                with open(self.shard_path, "a") as f:  # this process's own file: one writer, no race
                    f.write(json.dumps(line, sort_keys=True) + "\n")
            except OSError:
                pass  # the overlay is advisory: a write failure never fails an attempt (the in-memory state still applies)
        return before, self.rec(entry)

    def dead_cells(self, cid: str) -> List[Tuple[float, float]]:
        """The invalidated cells (mm xy) of column ``cid``."""
        out = []
        prefix = cid + "#"
        for entry in set(self.base) | set(self.own):
            if entry.startswith(prefix) and entry.count("#") == 1 and self.dead(entry):
                x, y = entry[len(prefix):].split(",")
                out.append((round(float(x), 3), round(float(y), 3)))
        return out


_OVERLAYS: Dict[str, Overlay] = {}


def shard_path(map_key: str) -> Optional[str]:
    d = os.environ.get(ENV_SHARDS) or ""
    if not d:
        return None
    return os.path.join(d, str(map_key), f"{socket.gethostname()}_{os.getpid()}.jsonl")


def latest_version(root: str, map_key: str) -> int:
    vs = []
    for p in glob.glob(os.path.join(root, str(map_key), "v*.json")):
        b = os.path.basename(p)[1:-5]
        if b.isdigit():
            vs.append(int(b))
    return max(vs) if vs else 0


def load_merged(root: Optional[str], map_key: str, version: Optional[int] = None) -> Tuple[Dict[str, Dict[str, int]], int]:
    """``(entries, version)`` of a merged overlay (``<root>/<map key>/v<N>.json``; the latest unless ``version``); ``({}, 0)`` when none."""
    if not root:
        return {}, 0
    v = int(version) if version else latest_version(root, map_key)
    if v <= 0:
        return {}, 0
    p = os.path.join(root, str(map_key), f"v{v}.json")
    if not os.path.exists(p):
        return {}, 0
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA or d.get("map_key") != map_key:
        return {}, 0
    return {e: dict(r) for e, r in (d.get("entries") or {}).items()}, v


def overlay_of(map_key: str) -> Overlay:
    """The process's overlay of a map identity (created on first use: the merged base named by ``ISPATIALGYM_MAP_OVERLAY``, this
    process's shard)."""
    ov = _OVERLAYS.get(str(map_key))
    if ov is None:
        ver = os.environ.get(ENV_VERSION) or None
        base, v = load_merged(os.environ.get(ENV_ROOT) or None, str(map_key), int(ver) if ver else None)
        ov = _OVERLAYS[str(map_key)] = Overlay(str(map_key), base, v, shard_path=shard_path(str(map_key)))
    return ov


def view_tag(map_key: str) -> Optional[str]:
    """What a map view's identity gains from this mechanism (start-cell caches key on it): None when off; ``inherit+ov<N>`` with the
    merged overlay version read (0: none)."""
    if not on():
        return None
    return f"inherit_k{k()}+ov{overlay_of(map_key).base_version}"


def merge(shards: Sequence[str], root: str) -> Dict[str, Dict[str, Any]]:
    """Offline: fold shard files (``*.jsonl``) into a new merged version per map key under ``root`` (module doc). A shard already listed
    in the latest version's ``sources`` (by path + size) is skipped. Returns ``{map key: {version, entries, new_events}}``."""
    by_key: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = {}
    for p in sorted(set(shards)):
        lines = []
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue  # a torn last line of a killed job
                if d.get("schema") == SCHEMA and d.get("map_key"):
                    lines.append(d)
        for d in lines:
            by_key.setdefault(str(d["map_key"]), [])
        for mk in {str(d["map_key"]) for d in lines}:
            by_key[mk].append((p, [d for d in lines if str(d["map_key"]) == mk]))
    out: Dict[str, Dict[str, Any]] = {}
    for mk, items in by_key.items():
        entries, v = load_merged(root, mk)
        prev_sources = set()
        if v:
            with open(os.path.join(root, mk, f"v{v}.json")) as f:
                prev_sources = {(s["path"], int(s["size"])) for s in json.load(f).get("sources") or []}
        sources = [{"path": p, "size": int(os.path.getsize(p))} for p, _ in items]
        n_new = 0
        for (p, evs), src in zip(items, sources):
            if (src["path"], src["size"]) in prev_sources:
                continue
            for d in evs:
                r = entries.setdefault(str(d["entry"]), new_rec())
                if d.get("outcome") in r:
                    r[d["outcome"]] += 1
                    n_new += 1
        if not n_new:
            out[mk] = {"version": v, "entries": entries, "new_events": 0}
            continue
        nv = v + 1
        doc = {"schema": SCHEMA, "map_key": mk, "version": nv, "k": k(), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "sources": sorted(({"path": s["path"], "size": s["size"]} for s in sources), key=lambda s: s["path"])
               + [s for s in ({"path": a, "size": b} for a, b in prev_sources)],
               "entries": entries,
               "summary": {"entries": len(entries), "invalidated": sum(1 for r in entries.values() if is_dead(r)),
                           "confirmed": sum(1 for r in entries.values() if is_confirmed(r))}}
        os.makedirs(os.path.join(root, mk), exist_ok=True)
        tmp = os.path.join(root, mk, f".v{nv}.json.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        final = os.path.join(root, mk, f"v{nv}.json")
        if os.path.exists(final):
            os.remove(tmp)
            raise FileExistsError(final)  # another merge won: rerun (it reads the newer version)
        os.rename(tmp, final)
        out[mk] = {"version": nv, "entries": entries, "new_events": n_new}
    return out


# ------------------------------------------------------------------------------------------------ the map-view side (ReachBands)


def _count(s, name: str, n: int = 1, map_key: Optional[str] = None) -> None:
    st = getattr(getattr(s, "kit", None), "stats", None)
    if isinstance(st, dict) and n:
        st[name] = st.get(name, 0) + int(n)
        if map_key:
            kk = f"{name.split('.')[0]}.by_map.{str(map_key)[:8]}.{name.split('.', 1)[1]}"
            st[kk] = st.get(kk, 0) + int(n)


def _built_keys(bands, so: Optional[float]) -> List[Tuple[float, float]]:
    """The built trunks of a map view at stand-off ``so`` (``meta.trunks``: the eligible ones, one row per stand-off); the cell dict's
    keys when the meta lists none."""
    keys = [(float(t["translate"]), float(t["tilt"])) for t in (bands.meta.get("trunks") or [])
            if so is None or t.get("stand_off") is None or abs(float(t["stand_off"]) - float(so)) < 1e-6]
    return list(dict.fromkeys(keys)) or list(bands._maps(so)[0].keys())


def view_columns(bands, translate: float, tilt: float, stand_off: Optional[float]) -> Optional[Tuple[Column, Column, str]]:
    """``(pruned, full, column id)`` of the inherited column a map view (``meta.mode == "map_view"``) reads at ``(translate, tilt)``, or
    None (switch off, not a map view, a mirrored view, a built tilt, outside the built range). Cached per (stand-off, column) and overlay
    version; a new column is counted on the sampler the view carries (``bands._map_tilt_sampler``)."""
    meta = getattr(bands, "meta", None)
    if not isinstance(meta, dict) or meta.get("mode") != "map_view" or meta.get("mirrored_from") or not on():
        return None  # a mirrored view (another arm's cells) keeps the nearest tilt: its cells are not the map's own
    so = bands.stand_off_of(stand_off)
    keys = _built_keys(bands, so)
    if bracket(keys, translate, tilt) is None:
        return None
    ckey = (round(float(translate), 2), tilt_key(tilt))
    mk = str(meta.get("map_key"))
    ov = overlay_of(mk)
    cache = bands.__dict__.setdefault("_map_tilt_cols", {})
    hit = cache.get((so, ckey))
    if hit is not None and hit[0] == ov.version:
        return hit[1]
    full = hit[1][1] if hit is not None else union_column(*bands._maps(stand_off), translate, tilt, all_keys=keys)
    if full is None:
        return None
    cid = col_id(so, full.key)
    if cid not in ov.inherited_cols:
        ov.inherited_cols.add(cid)
        s = getattr(bands, "_map_tilt_sampler", None)
        _count(s, "map_tilt.inherited_columns", 1, mk)
        _count(s, "map_tilt.inherited_feasible", len(full.cells), mk)
    out = (prune(full, ov.dead_cells(cid)), full, cid)
    cache[(so, ckey)] = (ov.version, out)
    return out


def view_column(bands, translate: float, tilt: float, stand_off: Optional[float]) -> Optional[Column]:
    """The inherited column a map view reads at ``(translate, tilt)`` after pruning (:func:`view_columns`), or None."""
    r = view_columns(bands, translate, tilt, stand_off)
    return None if r is None else r[0]


@dataclass
class FocusEntry:
    """The inherited column cell a Phase-3 focus stands on (:func:`focus_entry`)."""

    overlay: Overlay
    col: str
    cell: str
    map_key: str
    base_yaw: float

    @property
    def entry(self) -> str:
        return f"{self.col}#{self.cell}"

    def yaw_entry(self, world_yaw: float, grasp: str) -> str:
        return f"{self.entry}#{yaw_id(float(world_yaw) - self.base_yaw, grasp)}"


def focus_entry(bands, stance, fx, stand_off: Optional[float]) -> Optional[FocusEntry]:
    """The inherited column cell of the focus ``fx`` (world xy) at ``stance`` when the view reads an inherited column there (the cell of
    the column nearest the focus within half a lattice step), else None."""
    r = view_columns(bands, float(stance.translate), float(stance.tilt), stand_off)
    if r is None:
        return None
    # the focus may stand on a cell invalidated meanwhile: the unpruned column decides whether the focus is on the column
    _, full, cid = r
    if not len(full.cells):
        return None
    cb = np.asarray(bands.to_base(np.asarray([float(fx[0]), float(fx[1])], dtype=float).reshape(1, 2), stance.base_pose2d), dtype=float)[0]
    lat = float(bands.meta.get("lattice") or 0.08)
    d = np.abs(full.cells - cb[None, :]).max(axis=1)
    i = int(np.argmin(d))
    if float(d[i]) > 0.5 * lat + 1e-6:
        return None
    mk = str(bands.meta.get("map_key"))
    return FocusEntry(overlay_of(mk), cid, cell_id(full.cells[i]), mk, float(stance.base_pose2d[2]))


def note(s, fe: FocusEntry, outcome: str, *, yaw: Optional[float] = None, grasp: Optional[str] = None, why: str = "") -> None:
    """Record one Phase-3 outcome on the focus's inherited cell (and, with ``yaw`` / ``grasp``, its yaw pair) + the kit counters."""
    before, after = fe.overlay.record(fe.entry, outcome, why=str(why)[:120])
    mk = fe.map_key
    if outcome == "ok" and not is_confirmed(before):
        _count(s, "map_tilt.cells_confirmed", 1, mk)
    if outcome == "reach_fail":
        _count(s, "map_tilt.reach_fail", 1, mk)
        if is_dead(after) and not is_dead(before):
            _count(s, "map_tilt.cells_invalidated", 1, mk)
    if outcome == "other_fail":
        _count(s, "map_tilt.other_fail", 1, mk)
    if yaw is not None and grasp:
        ye = fe.yaw_entry(yaw, grasp)
        yb, ya = fe.overlay.record(ye, outcome, why=str(why)[:120])
        if outcome == "ok" and not is_confirmed(yb):
            _count(s, "map_yaw.pairs_confirmed", 1, mk)
        if outcome == "reach_fail" and is_dead(ya) and not is_dead(yb):
            _count(s, "map_yaw.pairs_invalidated", 1, mk)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="fold shard files into a new merged overlay version per map key")
    m.add_argument("--root", required=True, help="the overlay root (<root>/<map key>/v<N>.json)")
    m.add_argument("shards", nargs="+", help="shard files or globs (e.g. $OFH/out/<tag>/*/g*/map_overlay/*/*.jsonl)")
    i = sub.add_parser("info", help="summary of the latest merged version of each map key under a root")
    i.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        paths = sorted({p for g in a.shards for p in (glob.glob(g) or [g]) if os.path.isfile(p)})
        res = merge(paths, a.root)
        for mk, r in sorted(res.items()):
            ent = r["entries"]
            print(f"{mk} v{r['version']} new_events {r['new_events']} entries {len(ent)} invalidated {sum(1 for x in ent.values() if is_dead(x))} "
                  f"confirmed {sum(1 for x in ent.values() if is_confirmed(x))}")
    else:
        for d in sorted(glob.glob(os.path.join(a.root, "*"))):
            mk = os.path.basename(d)
            ent, v = load_merged(a.root, mk)
            print(f"{mk} v{v} entries {len(ent)} invalidated {sum(1 for x in ent.values() if is_dead(x))} confirmed {sum(1 for x in ent.values() if is_confirmed(x))}")


if __name__ == "__main__":
    main()
