"""Tabletop reach bands: which cells of a support's near strip the right hand can grasp an object on, per trunk.

For one (scene support, grasped object) the script computes, in **base coordinates** (x ahead, lateral + = robot's
left) at one concrete stance (edge midpoint, stand‑off 0.45 m + AABB overhang when >= 5 cm, yaw = inward normal), for
every trunk of the grid (translate {0, 0.25, 0.5, 0.75, 0.95} x tilt {0, -0.3, -0.6, -0.9}; -1.2 added when the top is
below 0.6 m):

1. whole‑body collision check of the trunk with untucked arms and open grippers, no spawned objects; colliding trunks
   are skipped;
2. an 8 cm lattice over the near strip of the top at the object's grasp height; cells farther from the right shoulder
   than (arm reach + largest grasp offset + pre‑grasp) are dropped geometrically (no IK);
3. the library‑grasp mask on the remaining cells: object posed at 8 yaws x every library grasp, pre‑grasp 10 cm back
   along the grasp approach, 12‑seed batched IK with the locked joints of the untucked init configuration at that trunk
   (idle arm untucked), the support (and objects placed before, e.g. the source tray) ignored for the grasp pose only.
   (yaw, grasp) pairs are evaluated in rounds - best‑ranked grasps and robot‑facing yaws first - with early exit per
   cell (one feasible pair marks the cell). Objects without library grasps (``--generic-pinch``: blocks, push poses) use
   the generic top‑down pinch (2 wrist yaws x 2 approach pitches);
4. output per trunk: the reachable cells (x, lateral), the (yaw, grasp) pairs that marked each cell (the seed of the
   feasible set F of spec 4.6, stored in the ``.npz``) and, over all trunks, the union extent [lambda_min, lambda_max].

Two‑factor cache (``ISPATIALGYM_REACH_BANDS``, default ``~/.cache/ispatialgym/reach_bands``): the support factor
(collision‑free trunks, lattice, shoulder positions) is keyed by scene + support + top height + stance + trunk grid and
shared by every object; the object factor (the mask) adds the object model, spawn scale, grasp set, rest orientation and
the placed objects. Neither key contains the base pose: a different stand‑off only changes which cells fall on the top.

Modes::

    python -m ispatialgym.sampling.reach_bands --scene S --support T --object CATEGORY/MODEL [--grasp-key K] [--scale sx,sy,sz]
    python -m ispatialgym.sampling.reach_bands ... --placed tray/gsxbym            # disembedding (legacy): fixed tray, lattice = its floor
    python -m ispatialgym.sampling.reach_bands ... --placed 'tray/gsxbym*1.25@focus_centred_edge_aligned'
        # disembedding, stance-relative (owner item 9): the tray (spawn scale 1.25) is centred on every lattice cell
        # of the top with its long axis along the edge, the object stands on its floor at the cell; one band per
        # (support, object, tray spec) serves every template (see PlacedSpec / placed_spec())
    python -m ispatialgym.sampling.reach_bands ... --generic-pinch                 # blocks / pushes, no library grasps
    python -m ispatialgym.sampling.reach_bands --scene S --support floor --room R --object C/M [--under PARTNER]
    python -m ispatialgym.sampling.reach_bands --scene S --support T --target toolbox/redmtl --handle-key toolbox__redmtl \
        --joint j_lid --joint-values 0.0,1.48,2.1,2.97 --brace                      # small‑scale articulation

``--rooms`` restricts the loaded rooms (default: the support's rooms); chairs within 1 m of the support are removed
(scene variant) like the sampler does. ``--smoke`` uses a 16 cm lattice and two trunks ((0.5, 0) and (0.25, -0.3)),
verifies the shoulder FK against the simulated robot and prints a band sanity summary; ``--trunks 0.5,0 0.95,-0.9``
restricts the grid explicitly.

Worker mode - many bands in one process (the scene load + cuRobo setup was 41 % of a one-band process)::

    python -m ispatialgym.sampling.reach_bands --jobs runlogs/reach_bands_20260915/jobs.json --ids ID1 ID2 ... --out ROOT

runs the manifest jobs (``scripts/reach_bands/make_jobs.py`` format; ``--ids`` default = all) grouped by world key (scene,
loaded rooms, removed chairs) on one loaded world per key, writing ``ROOT/<id>/<band>.npz|.json`` + ``run.log`` and a
``ROOT/done.txt`` line per job like the pod runner (``collect_bands.py`` reads the tree unchanged). Bands are isolated (the
previous band's objects removed, the collision world re-synced, the cuRobo seed streams reset), so every band is
byte-identical to its one-process run; a failing job prints ``REACH_BANDS_FAILED <id>`` and the others continue.

Setup defaults since 2026-09-16 (single-band and worker mode alike; each verified byte-identical on three two-trunk
bands): the world is camera-less (``SimWorld(cameras=False)``), the arms' motion planners are not built (bands use the
kinematics entries and the batched IK solvers only) and the neighbouring rooms' objects that can never pass
``SimWorld.obstacle_filter`` are not loaded (``_slim_drop_list``: same obstacle set and cuRobo meshes, 421 -> 108
objects, scene load 117 s -> 54 s on house_single_floor). ``--cameras`` / ``--prebuild-planner`` / ``--full-load``
restore the old setup; none changes a band.

Reading a band (the sampler)::

    from ispatialgym.sampling.reach_bands import ReachBands
    rb = ReachBands.load("house_single_floor", "breakfast_table_rhjoby_0", "pack_of_kielbasa/dgodnn", "pack_of_kielbasa__dgodnn__t")
    rb.band(0.5, 0.0)                 # (N, 2) cells (x ahead, lateral) of the grid trunk (same translate, nearest tilt)
    rb.union_extent()                 # (lambda_min, lambda_max) over every trunk
    rb.feasible(0.5, 0.0, cell)       # [(yaw_offset, grasp_name), ...] recorded for that cell (yaw_world = base_yaw + yaw_offset)
    rb.to_world(cells, base_pose2d)   # base-frame cells -> world xy for a concrete stance

``ReachBands.load`` looks in ``ISPATIALGYM_REACH_BANDS`` (if set), then the committed ``runlogs/reach_bands_20260915/bands``,
then ``~/.cache/ispatialgym/reach_bands``; the match is by the meta fields (scene, support, object, grasp key, spawn
scale, placed objects, ``--under`` partner; brace: target, handle key, joint), not by the hashed file name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling.eef_reach_volume import TILTS, TRANSLATES

log = logging.getLogger("ispatialgym.sampling.reach_bands")

LATTICE = 0.08
STAND_OFF = 0.45
PRE_GRASP = 0.10
LOW_TOP_TILT = -1.2  # added to the grid when the top is below 0.6 m (floor, low tables)
LOW_TOP_Z = 0.60
YAWS = [k * math.pi / 4 for k in range(8)]
N_YAWS_GENERIC = 2
#: generic push / pinch approach pitches (deg, approach vector tilted from straight down toward the robot's forward
#: direction, i.e. the hand comes from the robot's side): owner decision 2026-09-16 A1 added 45 and 60 so that far cells
#: (0.9-1.0 m) keep an IK-feasible push arc. Class order = ``RobotKit.grasp_candidates_at`` (pitch-major, yaw-minor).
TILTS_GENERIC = (0.0, 30.0, 45.0, 60.0)
YAWS_GENERIC_DEG = (0, 90)
GENERIC_CLASSES = [f"p{int(t):02d}_y{y}" for t in TILTS_GENERIC for y in YAWS_GENERIC_DEG]
#: generic CONTACT pose set of the brace arm (owner decision 2026-09-16 A1): closed gripper, pads on a vertical face of
#: the object at ``CONTACT_HEIGHT_FRACTION`` of its height with a horizontal approach ``a`` (base frame: ``near`` = pads
#: on the face toward the robot pushing away, ``far`` = on the far face pulling toward the robot, ``left`` / ``right`` =
#: on the robot-left / robot-right face pushing across), closing axis ``c`` vertical (``_v``: fingers stacked) or
#: horizontal (``_h``: fingers side by side), plus the top press (``a`` = down, two wrist yaws) on the object's top.
#: eef origin = contact - CONTACT_CENTER_Z·a (pad centre on the point, conventions ``contact_eef``), pre-contact 10 cm
#: back along -a. One band per object height class (``HEIGHT_BIN``); the cell = the contact point (x ahead, lateral).
CONTACT_HEIGHT_FRACTION = 0.75
HEIGHT_BIN = 0.02
CONTACT_CLASSES: List[Dict[str, Any]] = [
    {"name": "near_v", "a": (1.0, 0.0, 0.0), "c": (0.0, 0.0, 1.0), "z": "side"},
    {"name": "near_h", "a": (1.0, 0.0, 0.0), "c": (0.0, 1.0, 0.0), "z": "side"},
    {"name": "far_v", "a": (-1.0, 0.0, 0.0), "c": (0.0, 0.0, 1.0), "z": "side"},
    {"name": "far_h", "a": (-1.0, 0.0, 0.0), "c": (0.0, 1.0, 0.0), "z": "side"},
    {"name": "left_v", "a": (0.0, -1.0, 0.0), "c": (0.0, 0.0, 1.0), "z": "side"},
    {"name": "left_h", "a": (0.0, -1.0, 0.0), "c": (1.0, 0.0, 0.0), "z": "side"},
    {"name": "right_v", "a": (0.0, 1.0, 0.0), "c": (0.0, 0.0, 1.0), "z": "side"},
    {"name": "right_h", "a": (0.0, 1.0, 0.0), "c": (1.0, 0.0, 0.0), "z": "side"},
    {"name": "top_y0", "a": (0.0, 0.0, -1.0), "c": (0.0, 1.0, 0.0), "z": "top"},
    {"name": "top_y90", "a": (0.0, 0.0, -1.0), "c": (1.0, 0.0, 0.0), "z": "top"},
]
CONTACT_CLASSES_NAMES = [c["name"] for c in CONTACT_CLASSES]
CONTACT_SET = "generic_contact"
#: pad centre offset along the eef approach axis (conventions.CONTACT_CENTER_Z; negative = behind the origin)
CONTACT_CENTER_Z = -0.01186
#: idle-arm postures tried in this order when a trunk is not collision-free (owner decision 2026-09-16 A6: the idle
#: arm may take any recorded collision-free posture; the band records the posture per trunk)
POSTURES = ("untucked", "ready", "tucked")
STRIP_MAX_DEPTH = 0.95  # m into the support from the edge; deeper cells are never reachable
STRIP_HALF_WIDTH = 0.80
CACHE_DEFAULT = os.path.expanduser("~/.cache/ispatialgym/reach_bands")
#: committed bands of the representative run (what the samplers read on every machine; the env var overrides)
REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runlogs", "reach_bands_20260915", "bands")
SMOKE_TRUNKS = [(0.5, 0.0), (0.25, -0.3)]
SMOKE_FK_TRUNKS = [(0.5, 0.0), (0.95, -0.9)]


def cache_root() -> str:
    """Where the script writes by default (``--out`` overrides)."""
    return os.environ.get("ISPATIALGYM_REACH_BANDS", CACHE_DEFAULT)


def default_dirs() -> List[str]:
    """Directories :meth:`ReachBands.load` searches, in order: ``ISPATIALGYM_REACH_BANDS``, the committed runlog, ``~/.cache``."""
    env = os.environ.get("ISPATIALGYM_REACH_BANDS")
    dirs = ([env] if env else []) + [REPO_DIR, CACHE_DEFAULT]
    out: List[str] = []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in out and os.path.isdir(d):
            out.append(d)
    return out


def _key(d: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def trunk_grid(top_z_above_floor: float) -> List[Tuple[float, float]]:
    tilts = list(TILTS) + ([LOW_TOP_TILT] if top_z_above_floor < LOW_TOP_Z else [])
    return [(tr, ti) for tr in TRANSLATES for ti in tilts]


def trunk_key(translate: float, tilt: float) -> str:
    return f"t{translate:.2f}_k{tilt:+.1f}"


def height_class(h: float, step: float = HEIGHT_BIN) -> float:
    """Object height class (bin centre) of ``h``: 2 cm bins, the class value is the rounded height (``round(h / step) * step``)."""
    return round(max(1, int(round(float(h) / step))) * step, 4)


def parse_scale(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    vals = [float(v) for v in str(s).replace("[", "").replace("]", "").split(",")]
    if len(vals) == 1:
        vals = vals * 3
    assert len(vals) == 3, f"scale needs 1 or 3 numbers, got {s!r}"
    return [round(v, 4) for v in vals]


# ----------------------------------------------------------------------------------------------- placed objects

#: ``--placed`` modes (owner decision 2026-09-16 item 9). ``edge``: the legacy placement - the object centred on the
#: stance line with its near edge 8 cm behind the front edge of the top, long axis along the edge; the lattice is its
#: inner floor (a band valid for that one tray pose only). ``focus_centred_edge_aligned``: **stance-relative** - for
#: every lattice cell of the top (= the focus the sampler's Phase 2 picks) the placed object is centred on the cell
#: (its inner-floor centroid on the cell) with its long axis along the edge, the grasped object stands on its floor at
#: the cell; the band therefore holds for every template and the sampler places the tray exactly so
#: (``sampling.v2.tasks.disembedding``). Spec string: ``CATEGORY/MODEL[*SCALE][@MODE]``, e.g.
#: ``tray/gsxbym*1.25@focus_centred_edge_aligned`` (``SCALE`` = one number or ``sx,sy,sz``; no ``@MODE`` = ``edge``).
PLACED_MODE_EDGE = "edge"
PLACED_MODE_FOCUS = "focus_centred_edge_aligned"
PLACED_MODES = (PLACED_MODE_EDGE, PLACED_MODE_FOCUS)
#: margin of the "placed object fits on the top" count recorded per band (§3.2, the sampler's ``SUPPORT_MARGIN``) - a
#: diagnostic only: the lattice is not filtered by it (the sampler's stance may stand closer than the band stance)
PLACED_FIT_MARGIN = 0.02
#: legacy ``edge`` placement: near edge of the placed object this far behind the front edge of the top
PLACED_EDGE_SETBACK = 0.08


@dataclass(frozen=True)
class PlacedSpec:
    """One ``--placed`` object: ``category/model``, optional spawn scale and the placement mode (:data:`PLACED_MODES`).

    ``PlacedSpec.parse`` accepts the spec string, a ``(object, scale, mode)`` tuple / list (the owner's
    ``("tray/gsxbym", 1.25, "focus_centred_edge_aligned")``), a dict with those keys or a PlacedSpec; ``canonical()`` is
    the normalised string the band metas store (``placed_specs``) and :meth:`ReachBands.find` compares - every form of
    the same placement maps to the same string (uniform scale as one number, scale 1 dropped, the mode always written).
    Unknown modes are kept verbatim so that bands written by another module round-trip through the loader."""

    category: str
    model: str
    scale: Optional[Tuple[float, float, float]] = None
    mode: str = PLACED_MODE_EDGE

    @property
    def object(self) -> str:
        return f"{self.category}/{self.model}"

    @staticmethod
    def parse(spec) -> "PlacedSpec":
        if isinstance(spec, PlacedSpec):
            return spec
        if isinstance(spec, dict):
            obj = spec.get("object") or f"{spec.get('category')}/{spec.get('model')}"
            return PlacedSpec.parse((obj, spec.get("scale"), spec.get("mode") or PLACED_MODE_EDGE))
        if isinstance(spec, (tuple, list)):
            obj = str(spec[0])
            scale = spec[1] if len(spec) > 1 else None
            mode = str(spec[2]) if len(spec) > 2 and spec[2] else PLACED_MODE_EDGE
            s = f"{obj}"
            if scale is not None:
                s += "*" + (",".join(str(float(v)) for v in scale) if isinstance(scale, (list, tuple)) else str(float(scale)))
            return PlacedSpec.parse(s + "@" + mode)
        s = str(spec).strip()
        mode = PLACED_MODE_EDGE
        if "@" in s:
            s, mode = s.split("@", 1)
            mode = mode.strip() or PLACED_MODE_EDGE
        scale = None
        if "*" in s:
            s, sc = s.split("*", 1)
            vals = parse_scale(sc)
            if vals and any(abs(v - 1.0) > 1e-6 for v in vals):
                scale = tuple(vals)
        if "/" not in s:
            raise ValueError(f"placed spec {spec!r}: expected CATEGORY/MODEL[*SCALE][@MODE]")
        cat, model = s.split("/", 1)
        if mode not in PLACED_MODES:
            log.warning("placed spec %r: unknown mode %r (known: %s); kept verbatim", spec, mode, list(PLACED_MODES))
        return PlacedSpec(cat.strip(), model.strip(), scale, mode)

    def scale_list(self) -> Optional[List[float]]:
        return list(self.scale) if self.scale is not None else None

    def canonical(self) -> str:
        s = self.object
        if self.scale is not None:
            sx, sy, sz = self.scale
            s += "*" + (f"{sx:g}" if abs(sx - sy) < 1e-6 and abs(sx - sz) < 1e-6 else f"{sx:g},{sy:g},{sz:g}")
        return s + "@" + self.mode

    def to_json(self) -> Dict[str, Any]:
        return {"category": self.category, "model": self.model, "scale": self.scale_list(), "mode": self.mode, "spec": self.canonical()}


def placed_spec(obj: str, scale=None, mode: str = PLACED_MODE_FOCUS) -> str:
    """Canonical ``--placed`` spec string of ``obj`` (``CATEGORY/MODEL``) at ``scale`` (number or ``[sx, sy, sz]``, None =
    canonical size) in ``mode`` - what the sampler passes as ``placed=[...]`` to :meth:`ReachBands.load`::

        placed_spec("tray/gsxbym", 1.25)   # 'tray/gsxbym*1.25@focus_centred_edge_aligned'
    """
    return PlacedSpec.parse((obj, scale, mode)).canonical()


def canonical_placed(specs) -> List[str]:
    """Sorted canonical spec strings of a ``placed`` request or of a band meta (``placed_specs``, else the ``placed``
    entries' category / model / scale / mode)."""
    if isinstance(specs, dict):  # a band meta
        items = specs.get("placed_specs")
        if items is None:
            items = [{"category": p.get("category"), "model": p.get("model"), "scale": p.get("scale"), "mode": p.get("mode")} for p in specs.get("placed") or []]
        specs = items
    return sorted(PlacedSpec.parse(s).canonical() for s in (specs or []))


# ----------------------------------------------------------------------------------------------- result container


_INDEX_CACHE: Dict[str, Tuple[float, List[Tuple[str, Dict[str, Any]]]]] = {}


def _index(d: str) -> List[Tuple[str, Dict[str, Any]]]:
    """(npz path, meta) of every band file in ``d`` (re-read when the directory changed)."""
    try:
        mtime = os.path.getmtime(d)
    except OSError:
        return []
    cached = _INDEX_CACHE.get(d)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        npz = os.path.join(d, fn[:-5] + ".npz")
        if not os.path.exists(npz):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and "trunks" in meta and "scene" in meta:
            out.append((npz, meta))
    _INDEX_CACHE[d] = (mtime, out)
    return out


def _same_scale(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return all(abs(float(x) - float(y)) < 1e-3 for x, y in zip(a, b)) and len(a) == len(b)


@dataclass
class ReachBands:
    """Reachable cells per trunk in base coordinates (x ahead, lateral + = robot's left) at the stance
    ``meta["stance"]["base_pose2d"]`` they were computed at, plus the (yaw, grasp) pairs that marked each cell.

    * ``cells[(translate, tilt)]`` -> ``(N, 2)`` float array of cell centres (lattice ``meta["lattice"]``);
    * ``pairs[(translate, tilt)]`` -> ``(M, 3)`` int array ``[cell index, yaw index, grasp index]`` (``yaw index`` into
      :data:`YAWS` = object yaw offset relative to the base yaw; ``grasp index`` into ``meta["grasp_names"]``, or, for the
      generic pinch, into the ``grasp_candidates_at`` order with ``yaw index = -1``);
    * ``classes[(translate, tilt)]`` -> ``(N,)`` int bitmask per cell (bit k = pose class ``meta["classes"][k]`` passed):
      generic pinch / push bands (:data:`GENERIC_CLASSES`, pitch x wrist yaw) and left-arm contact bands
      (:data:`CONTACT_CLASSES`); empty for library-grasp bands;
    * brace mode (``meta["mode"] == "brace"``): ``meta["trunks"][i]["feasible"]`` lists ``{cell, yaw_offset, q_feasible,
      brace_kinds, brace_q_prefix}``;
    * ``meta["trunks"][i]["posture"]`` = the idle-arm posture the trunk was computed with (``untucked`` / ``ready`` /
      ``tucked``, owner decision A6) and ``idle_arm_q`` its joint values per arm; ``meta["arm"]`` (default ``right``),
      ``meta["contact_set"]`` / ``meta["height_class"]`` for contact bands.
    """

    meta: Dict[str, Any]
    cells: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)  # (translate, tilt) -> (N, 2) reachable cells
    pairs: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)  # (translate, tilt) -> (M, 3) int [cell, yaw, grasp]
    path: Optional[str] = None
    classes: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)  # (translate, tilt) -> (N,) int class bitmask

    # -- loading --------------------------------------------------------------------------------------------------
    @staticmethod
    def from_files(path_npz: str) -> "ReachBands":
        d = np.load(path_npz, allow_pickle=False)
        with open(path_npz[:-4] + ".json") as f:
            meta = json.load(f)
        cells, pairs, classes = {}, {}, {}
        for t in meta["trunks"]:
            if t.get("collides"):
                continue
            k = trunk_key(t["translate"], t["tilt"])
            key = (float(t["translate"]), float(t["tilt"]))
            cells[key] = d[k] if k in d.files else np.zeros((0, 2))
            pairs[key] = d[k + "__pairs"] if (k + "__pairs") in d.files else np.zeros((0, 3), dtype=np.int32)
            if (k + "__classes") in d.files:
                classes[key] = d[k + "__classes"]
        return ReachBands(meta, cells, pairs, path_npz, classes)

    @staticmethod
    def find(scene: str, support: str, object: Optional[str] = None, grasp_key: Optional[str] = None, *, placed: Sequence[str] = (),
             under: Optional[str] = None, scale: Optional[Sequence[float]] = None, target: Optional[str] = None, joint: Optional[str] = None,
             mode: Optional[str] = None, arm: str = "right", contact_set: Optional[str] = None, height: Optional[float] = None,
             dirs: Optional[Sequence[str]] = None) -> List[str]:
        """Paths (``.npz``) of the cached bands matching the given factors, best match first.

        ``object`` = ``CATEGORY/MODEL`` (``None`` for object-less generic bands); ``grasp_key`` = the library key,
        ``"generic"`` for ``--generic-pinch`` bands, ``None`` = any grasp set; ``placed`` = the ``--placed`` objects
        (order-insensitive; spec strings ``CATEGORY/MODEL[*SCALE][@MODE]`` or ``(object, scale, mode)`` tuples, compared
        through :class:`PlacedSpec` - ``"tray/gsxbym"`` is the legacy ``edge`` placement, the disembedding sampler asks for
        :func:`placed_spec`\ ``("tray/gsxbym", 1.25)``); ``under`` = the IORS partner of ``--support floor --under``; ``scale`` = the spawn scale
        (``None`` = canonical); brace bands are selected with ``target`` (+ optional ``joint``); ``arm`` = the band's
        arm (bands without the field are right-arm bands); contact bands (``--generic-contact``) are selected with
        ``contact_set="generic_contact"`` and the object ``height`` (m) - the band whose height class (2 cm bins)
        contains it, ``height=None`` = any class (nearest to 0.10 m first)."""
        hits = []
        hcls = height_class(height) if height is not None else None
        placed_req = canonical_placed(placed)
        for d in (dirs if dirs is not None else default_dirs()):
            for npz, meta in _index(d):
                if meta.get("scene") != scene or meta.get("support") != support:
                    continue
                if mode is not None and meta.get("mode") != mode:
                    continue
                if meta.get("arm", "right") != arm:
                    continue
                if contact_set is not None:
                    if meta.get("contact_set") != contact_set:
                        continue
                    if hcls is not None and abs(float(meta.get("height_class", -1.0)) - hcls) > 1e-6:
                        continue
                elif meta.get("contact_set"):
                    continue
                elif target is not None:
                    if meta.get("target") != target or (joint is not None and meta.get("joint") != joint):
                        continue
                else:
                    if meta.get("mode") == "brace" or meta.get("object") != object:
                        continue
                    if grasp_key == "generic":
                        if not meta.get("generic_pinch"):
                            continue
                    elif grasp_key is not None and meta.get("grasp_key") != grasp_key:
                        continue
                    if not _same_scale(scale, meta.get("scale")):
                        continue
                if placed_req != canonical_placed(meta):
                    continue
                if (meta.get("under_partner") or None) != (under or None):
                    continue
                hits.append((npz, meta))
        # finest lattice first, then most trunks, then the newest (contact bands with height=None: the class nearest 0.10 m)
        hits.sort(key=lambda h: (abs(float(h[1].get("height_class", 0.1)) - 0.1) if (contact_set and hcls is None) else 0.0, float(h[1].get("lattice", 1.0)),
                                 -len(h[1].get("trunks", [])), -os.path.getmtime(h[0])))
        return [h[0] for h in hits]

    @staticmethod
    def load(scene: str, support: str, object: Optional[str] = None, grasp_key: Optional[str] = None, *, placed: Sequence[str] = (),
             under: Optional[str] = None, scale: Optional[Sequence[float]] = None, target: Optional[str] = None, joint: Optional[str] = None,
             mode: Optional[str] = None, arm: str = "right", contact_set: Optional[str] = None, height: Optional[float] = None,
             dirs: Optional[Sequence[str]] = None) -> "ReachBands":
        """Load the cached band for the given factors (see :meth:`find`); ``FileNotFoundError`` on a cache miss.

        Left-arm contact band of an object of height ``h`` on ``support``::

            ReachBands.load(scene, support, arm="left", contact_set="generic_contact", height=h)
        """
        paths = ReachBands.find(scene, support, object, grasp_key, placed=placed, under=under, scale=scale, target=target, joint=joint, mode=mode, arm=arm,
                                contact_set=contact_set, height=height, dirs=dirs)
        if not paths:
            raise FileNotFoundError(f"no reach band for scene={scene} support={support} object={object} grasp_key={grasp_key} placed={list(placed)} "
                                    f"under={under} scale={scale} target={target} joint={joint} arm={arm} contact_set={contact_set} height={height} "
                                    f"in {list(dirs) if dirs is not None else default_dirs()}")
        if len(paths) > 1:
            log.debug("%d reach bands match (%s ...); using %s", len(paths), object or target or contact_set, paths[0])
        return ReachBands.from_files(paths[0])

    # -- queries --------------------------------------------------------------------------------------------------
    def trunks(self) -> List[Tuple[float, float]]:
        return sorted(self.cells)

    def nearest_trunk(self, translate: float, tilt: float) -> Optional[Tuple[float, float]]:
        cands = [k for k in self.cells if abs(k[0] - translate) < 1e-6]
        if not cands:
            return None
        return min(cands, key=lambda k: abs(k[1] - tilt))

    def band(self, translate: float, tilt: float) -> np.ndarray:
        """Cells of the grid trunk with the same translate and the nearest tilt (empty array when that trunk collides)."""
        k = self.nearest_trunk(translate, tilt)
        return self.cells[k] if k is not None else np.zeros((0, 2))

    def union_extent(self) -> Tuple[float, float]:
        lat = np.concatenate([c[:, 1] for c in self.cells.values() if len(c)]) if self.cells else np.zeros(0)
        return (float(lat.min()), float(lat.max())) if len(lat) else (0.0, 0.0)

    def cell_index(self, translate: float, tilt: float, cell) -> Optional[int]:
        k = self.nearest_trunk(translate, tilt)
        if k is None or not len(self.cells[k]):
            return None
        d = np.linalg.norm(self.cells[k] - np.asarray(cell, dtype=float)[None, :2], axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= 0.51 * float(self.meta.get("lattice", LATTICE)) else None

    def feasible(self, translate: float, tilt: float, cell) -> List[Tuple[Any, Any]]:
        """Recorded feasible (yaw offset [rad, relative to the base yaw], grasp name) pairs of the band cell nearest to
        ``cell`` (generic pinch: ``("generic", candidate index)``); the seed of the feasible set F of spec 4.6."""
        k = self.nearest_trunk(translate, tilt)
        i = self.cell_index(translate, tilt, cell)
        if k is None or i is None:
            return []
        p = self.pairs.get(k)
        if p is None or not len(p):
            return []
        names = self.meta.get("grasp_names") or []
        out = []
        for ci, yi, gi in p[p[:, 0] == i].tolist():
            if yi < 0:
                out.append(("generic", int(gi)))
            else:
                out.append((YAWS[int(yi)], names[int(gi)] if int(gi) < len(names) else int(gi)))
        return out

    @property
    def class_names(self) -> List[str]:
        """Pose class names of the ``classes`` bitmask (generic pinch / push pitch x yaw classes, or the contact classes)."""
        return list(self.meta.get("classes") or [])

    def classes_at(self, translate: float, tilt: float, cell) -> List[str]:
        """Pose classes that passed at the band cell nearest to ``cell`` (generic / contact bands; ``[]`` for library-grasp
        bands or when the cell is not in the band). Placement prefers a class from this list (e.g. the shallowest pitch)."""
        k = self.nearest_trunk(translate, tilt)
        i = self.cell_index(translate, tilt, cell)
        m = self.classes.get(k) if k is not None else None
        if i is None or m is None or i >= len(m):
            return []
        bits = int(m[i])
        return [n for b, n in enumerate(self.class_names) if bits & (1 << b)]

    def band_for_class(self, translate: float, tilt: float, class_name: str) -> np.ndarray:
        """Cells of the trunk (same translate, nearest tilt) at which the given pose class passed."""
        k = self.nearest_trunk(translate, tilt)
        if k is None or k not in self.classes or class_name not in self.class_names:
            return np.zeros((0, 2))
        b = self.class_names.index(class_name)
        m = self.classes[k]
        return self.cells[k][(m & (1 << b)) != 0] if len(m) == len(self.cells[k]) else np.zeros((0, 2))

    def posture(self, translate: float, tilt: float) -> Optional[Dict[str, Any]]:
        """Idle-arm posture recorded for the trunk (same translate, nearest tilt): ``{"posture": name, "idle_arm_q": {arm: [7]}}``
        (``posture`` = ``untucked`` for bands computed before A6, which skipped colliding trunks)."""
        k = self.nearest_trunk(translate, tilt)
        if k is None:
            return None
        for t in self.meta.get("trunks", []):
            if abs(float(t["translate"]) - k[0]) < 1e-6 and abs(float(t["tilt"]) - k[1]) < 1e-6:
                return {"posture": t.get("posture", "untucked"), "idle_arm_q": t.get("idle_arm_q")}
        return None

    def brace_feasible(self, translate: float, tilt: float) -> List[Dict[str, Any]]:
        """Brace mode: ``[{cell, yaw_offset, brace_kinds}]`` of the trunk (same translate, nearest tilt)."""
        k = self.nearest_trunk(translate, tilt)
        if k is None:
            return []
        for t in self.meta.get("trunks", []):
            if abs(float(t["translate"]) - k[0]) < 1e-6 and abs(float(t["tilt"]) - k[1]) < 1e-6:
                return list(t.get("feasible", []))
        return []

    def to_world(self, cells_base: np.ndarray, base_pose2d: Sequence[float]) -> np.ndarray:
        x0, y0, yaw = base_pose2d
        c, s = math.cos(yaw), math.sin(yaw)
        cb = np.asarray(cells_base, dtype=float).reshape(-1, 2)
        return np.stack([x0 + c * cb[:, 0] - s * cb[:, 1], y0 + s * cb[:, 0] + c * cb[:, 1]], axis=1)

    def to_base(self, xy_world: np.ndarray, base_pose2d: Sequence[float]) -> np.ndarray:
        x0, y0, yaw = base_pose2d
        c, s = math.cos(yaw), math.sin(yaw)
        w = np.asarray(xy_world, dtype=float).reshape(-1, 2) - np.array([x0, y0])
        return np.stack([c * w[:, 0] + s * w[:, 1], -s * w[:, 0] + c * w[:, 1]], axis=1)


# ----------------------------------------------------------------------------------------------- computation


class ObstacleScope:
    """Collision world of one stance: the cuRobo meshes of scene objects whose AABB is farther than ``radius`` from the
    base centre are disabled while the scope is active (the robot's reach from the base centre is < 1.3 m; 400 meshes
    of the loaded rooms made every batched IK call ~4x slower than with the near obstacles only), and
    :meth:`ignoring` switches the meshes of given objects (the support for the grasp pose) off and on directly instead
    of re-syncing the whole world (``RobotKit.ignoring`` = two ``update_world`` passes per call).

    Nothing here changes what the checker considers an obstacle: ``update_world`` re-enables every mesh, so the scope
    must be re-entered after a ``kit.sync_world()``."""

    #: horizontal AABB distance from the base centre beyond which an object cannot touch the robot (arm span from the
    #: base centre < 1.3 m, plus the collision margin and the sphere radii)
    RADIUS = 1.6
    #: objects whose AABB bottom is this far above the floor (ceilings, room lights) are out of the robot's reach
    Z_ABOVE = 2.1

    def __init__(self, kit, world, base_xy, radius: float = RADIUS, drop_ground: bool = False):
        """``drop_ground``: also disable the ground-like meshes (floors, carpets, lawn - large meshes that dominate the
        collision cost); the cuRobo ground slab keeps modelling the floor. Only for tabletop bands (z_support >= 0.55 m),
        where the arm never comes near the floor; floor bands keep them."""
        self.kit, self.world, self.radius = kit, world, radius
        be = kit.backend
        names = set(be._world_mesh_names)
        self.names_of: Dict[str, List[str]] = {}
        for o in world.scene.objects:
            prefix = o.prim_path + "/"
            mine = [n for n in names if n.startswith(prefix)]
            if mine:
                self.names_of[o.name] = mine
        bx, by = float(base_xy[0]), float(base_xy[1])
        self.far: List[str] = []
        self.kept: List[str] = []
        ground = set(getattr(be, "_ground_mesh_names", set()))
        for o in world.scene.objects:
            if o is world.robot or o.name not in self.names_of:
                continue
            lo, hi = o.aabb
            dx = max(float(lo[0]) - bx, 0.0, bx - float(hi[0]))
            dy = max(float(lo[1]) - by, 0.0, by - float(hi[1]))
            mine = self.names_of[o.name]
            if math.hypot(dx, dy) > radius or float(lo[2]) > world.floor_z + self.Z_ABOVE or (drop_ground and all(n in ground for n in mine)):
                self.far.extend(mine)
            else:
                self.kept.append(o.name)
        self.n_near = sum(len(v) for v in self.names_of.values()) - len(self.far)

    def _enable(self, names: Sequence[str], enabled: bool):
        be = self.kit.backend
        for n in names:
            if n in be._world_mesh_names:
                be._mesh_enable(n, enabled)

    def __enter__(self):
        self._enable(self.far, False)
        log.info("collision world for the band: %d meshes of %d objects kept within %.1f m of the base (%s), %d far meshes disabled", self.n_near, len(self.kept),
                 self.radius, ", ".join(sorted(self.kept)), len(self.far))
        return self

    def __exit__(self, *exc):
        self._enable(self.far, True)

    def ignoring(self, objs):
        scope = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.names = [n for o in objs for n in scope.names_of.get(getattr(o, "name", o), [])]
                scope._enable(self_inner.names, False)

            def __exit__(self_inner, *exc):
                scope._enable(self_inner.names, True)

        return _Ctx()


class MovingPlaced:
    """A placed object (the disembedding source tray in the ``focus_centred_edge_aligned`` mode) whose cuRobo collision
    meshes follow the evaluated cell: hidden during the batched IK passes (the tray moves with the cell, so no single
    pose serves every cell), moved to the cell's tray pose for the per-cell pre-grasp verification (:meth:`place`) and
    hidden again (:meth:`hide`). Only the mesh poses in the collision checker change (``_mesh_update_pose``, no world
    re-sync); the simulated body stays where it was measured. ``pose_at(world_xy)`` = the tray pose whose inner-floor
    centroid is at the cell (the rule the sampler follows), ``rel`` = mesh poses in the tray's root frame."""

    def __init__(self, kit, world, obj, pose0, pose_at):
        from ispatialgym.sampling import geometry as G

        self.kit, self.world, self.obj, self.pose_at = kit, world, obj, pose_at
        self.G = G
        be = kit.backend
        prefix = obj.prim_path + "/"
        self.names = [n for n in be._world_mesh_names if n.startswith(prefix)]
        self.rel: Dict[str, Any] = {}
        p0 = (G.t(pose0[0]).float(), G.t(pose0[1]).float())
        for link in obj.links.values():
            for cm in link.collision_meshes.values():
                if cm.prim_path in self.names:
                    mp, mq = cm.get_position_orientation()
                    self.rel[cm.prim_path] = G.relative((mp.float(), mq.float()), p0)
        self.n_placed = 0
        self.visible = True

    def hide(self):
        be = self.kit.backend
        for n in self.names:
            be._mesh_enable(n, False)
        self.visible = False

    def show(self):
        be = self.kit.backend
        for n in self.names:
            be._mesh_enable(n, True)
        self.visible = True

    def place(self, world_xy):
        """Move the collision meshes to the tray pose of the cell and enable them."""
        import omnigibson.utils.transform_utils as T
        from curobo.types import Pose as CuPose
        from ispatialgym.oracle.poses import quat_xyzw_to_wxyz

        G, be = self.G, self.kit.backend
        pose = self.pose_at(world_xy)
        pose = (G.t(pose[0]).float(), G.t(pose[1]).float())
        rp, rq = be.root_pose()
        dev = be.device_cfg.device
        for n, rel in self.rel.items():
            mp, mq = G.compose(pose, rel)
            p, q = T.relative_pose_transform(mp, mq, rp, rq)
            be._mesh_update_pose(n, CuPose(position=p.reshape(1, 3).to(dev), quaternion=quat_xyzw_to_wxyz(q).reshape(1, 4).to(dev)))
        if not self.visible:
            self.show()
        self.n_placed += 1

    def self_check(self, cell_world) -> Dict[str, Any]:
        """Smoke check of the moving-mesh geometry: teleport the simulated body to ``pose_at(cell_world)``, read every
        collision mesh's world pose back from the simulator and compare it with the pose this class feeds the collision
        checker (``compose(pose_at(cell), rel)``); the body is put back afterwards. Returns the largest position /
        orientation error (m / rad); ``ok`` when both are within 5 mm / 0.5 deg (measured 3.0 mm / 0.02 deg on the
        1.25 tray: the render-side transform read lags the teleport by the settle; the checker margin is 1 cm)."""
        import omnigibson.utils.transform_utils as T

        G = self.G
        p_saved, q_saved = self.obj.get_position_orientation()
        pose = self.pose_at(cell_world)
        pose = (G.t(pose[0]).float(), G.t(pose[1]).float())
        err_p, err_r = 0.0, 0.0
        try:
            self.obj.set_position_orientation(pose[0], pose[1])
            self.world.sync_render_transforms()
            for link in self.obj.links.values():
                for cm in link.collision_meshes.values():
                    if cm.prim_path not in self.rel:
                        continue
                    sp, sq = cm.get_position_orientation()
                    mp, mq = G.compose(pose, self.rel[cm.prim_path])
                    err_p = max(err_p, float(th.linalg.norm(sp.float() - mp)))
                    dq = T.quat_distance(sq.float(), mq)
                    err_r = max(err_r, abs(float(T.quat2axisangle(dq).norm())))
        finally:
            self.obj.set_position_orientation(p_saved, q_saved)
            self.world.sync_render_transforms()
        rec = {"cell_world": [round(float(v), 3) for v in G.t(cell_world).tolist()], "n_meshes": len(self.rel), "max_pos_err_m": round(err_p, 5),
               "max_rot_err_rad": round(err_r, 5), "ok": err_p < 0.005 and err_r < math.radians(0.5)}
        (log.info if rec["ok"] else log.warning)("placed-object self check at cell %s: %d meshes, simulator vs moved-mesh pose error %.1f mm / %.2f deg -> %s",
                                                 rec["cell_world"], rec["n_meshes"], err_p * 1000, math.degrees(err_r), "ok" if rec["ok"] else "MISMATCH")
        return rec


class BandComputer:
    #: (yaw, grasp) pairs per round (one batched pre-grasp + grasp IK pass over the open cells)
    CHUNK = 8
    #: batched IK chunk (poses per cuRobo call)
    IK_BATCH = 128
    #: a cell none of whose first ``MAX_ROUNDS_WITHOUT_PRE * CHUNK`` pre-grasp poses is IK-feasible is dropped (out of
    #: reach for kinematic reasons; the exhaustive loop would spend the remaining rounds on it). Cells with a feasible
    #: pre-grasp but no feasible grasp yet (support / table collision at the grasp) run every round. Documented
    #: approximation of the exhaustive early-exit loop of spec 4.1 (measured 2026-09-15 on the smoke job: the 50 cells
    #: dropped after round 1 had no pre-feasible pose in round 2 either, so the band was identical; the approximation
    #: can lose cells at the far rim that only a lower-ranked grasp with another approach direction reaches); 0 = exhaustive.
    MAX_ROUNDS_WITHOUT_PRE = 1
    #: coarse-to-fine: when set and the lattice is finer, every trunk is first computed on a ``COARSE`` lattice (same
    #: cells snapped) and the fine lattice is evaluated only within ``COARSE * 1.5`` of a reachable coarse cell (the
    #: coarse cell and its 8 neighbours). Measured 2026-09-15 on the smoke job: 54 s vs 45 s exhaustive per trunk and one
    #: isolated cell lost - the IK cost is per pose, not per cell, so the coarse pass does not pay for itself; off (0).
    COARSE = 0.0

    def __init__(self, world, kit, arms: Sequence[str] = ("right",), lattice: float = LATTICE, seeds: int = 12):
        from ispatialgym.oracle import r1pro_model as M
        from ispatialgym.sampling import geometry as G

        self.M, self.G = M, G
        self.world, self.kit = world, kit
        self.arms = list(arms)
        self.lattice = lattice
        self.seeds = seeds
        self.scope: Optional[ObstacleScope] = None
        #: gripper opening of the locked finger joints (open for grasp bands, closed for the contact bands)
        self.fingers: float = 0.05
        #: contact-band mode: object height class (m) -> the generic CONTACT set replaces the generic pinch in
        #: :meth:`compute` (``grasp_sets=None``); ``None`` = pinch / library grasps
        self.contact_height: Optional[float] = None
        #: stance-relative placed object (:class:`MovingPlaced`, ``--placed ...@focus_centred_edge_aligned``): hidden
        #: during the batched passes, moved to every cell that passed for a per-cell pre-grasp check against it
        self.cell_obstacle: Optional[MovingPlaced] = None

    def ignoring(self, objs):
        return self.scope.ignoring(objs) if self.scope is not None else self.kit.ignoring(objs)

    # -- postures (owner decision A6) -----------------------------------------------------------------------------
    def trunk_config(self, translate: float, tilt: float, x0: float, y0: float, yaw0: float, postures: Sequence[str] = POSTURES):
        """Whole-body collision-free configuration at the trunk with the first posture of ``postures`` that is free
        (``untucked`` first; ``ready`` / ``tucked`` when the untucked arms self-collide, e.g. translate 0.95 / tilt -1.2):
        ``(q, posture)`` or ``(None, None)`` when every posture collides."""
        kit = self.kit
        for posture in postures:
            try:
                q = kit.q_at_base_pose(kit.q_config(arms=posture, translate=translate, tilt=tilt, fingers=self.fingers), x0, y0, yaw0)
            except (ValueError, KeyError):
                continue
            if bool(kit.collision_free(q, margin=0.01)[0]):
                return q, posture
        return None, None

    def idle_arm_q(self, q: th.Tensor) -> Dict[str, List[float]]:
        return {arm: [round(float(v), 4) for v in q[self.world.robot.arm_control_idx[arm]].tolist()] for arm in self.M.ARMS}

    # -- generic contact set (owner decision A1) --------------------------------------------------------------------
    def contact_candidates_at(self, xy_world, z_top: float, height: float, fwd, left) -> List[Tuple[int, List[Tuple[th.Tensor, th.Tensor]]]]:
        """[(class index, [pre-contact, contact])] of :data:`CONTACT_CLASSES` for a contact point at ``xy_world``: side
        classes at ``z_top + CONTACT_HEIGHT_FRACTION * height``, top presses at ``z_top + height``; approach / closing axes
        given in the base frame (``fwd``, ``left``, up); eef origin = point - CONTACT_CENTER_Z·a, pre = eef - PRE_GRASP·a."""
        import omnigibson.utils.transform_utils as T

        fx, fy = float(fwd[0]), float(fwd[1])
        lx, ly = float(left[0]), float(left[1])

        def world_vec(v):
            return th.tensor([v[0] * fx + v[1] * lx, v[0] * fy + v[1] * ly, v[2]], dtype=th.float32)

        out = []
        for k, cls in enumerate(CONTACT_CLASSES):
            a = world_vec(cls["a"])
            c = world_vec(cls["c"])
            a = a / th.linalg.norm(a)
            c = c - th.dot(c, a) * a
            c = c / th.linalg.norm(c)
            x = th.cross(c, a, dim=0)
            quat = T.mat2quat(th.stack([x, c, a], dim=1))
            z = z_top + (height if cls["z"] == "top" else CONTACT_HEIGHT_FRACTION * height)
            point = th.tensor([float(xy_world[0]), float(xy_world[1]), z], dtype=th.float32)
            eef = point - CONTACT_CENTER_Z * a
            out.append((k, [(eef - a * PRE_GRASP, quat), (eef, quat)]))
        return out

    #: batched IK solver sizes (one cuRobo solver per size, built once); the last chunk uses the smallest size that fits
    #: so that the padded problems (cuRobo solves full batches) stay few
    IK_SIZES = (16, 32, 64, 128)

    def ik_screen(self, poses, q: th.Tensor, arm: str) -> th.Tensor:
        """12-seed screening IK of ``poses`` from ``q`` with the arm embodiment, chunked without padding waste."""
        n = len(poses)
        if n == 0:
            return th.zeros(0, dtype=th.bool)
        out, i = [], 0
        while i < n:
            rem = n - i
            bs = self.IK_BATCH if rem >= self.IK_BATCH else next(b for b in self.IK_SIZES if b >= rem)
            out.append(self.kit.ik_batch(list(poses[i:i + bs]), q, arm=arm, batch_size=bs))
            i += bs
        return th.cat(out)

    # -- stance -------------------------------------------------------------------------------------------------
    def stance_at_edge(self, top, support, stand_off: float = STAND_OFF, edge_name: Optional[str] = None) -> Dict[str, Any]:
        """Base pose at the midpoint of one edge of the top (preferred: named edge; else the longest edge whose stance is on
        free floor and whole‑body collision free at trunk (0.5, 0))."""
        G, kit = self.G, self.kit
        lo, hi = support.aabb
        aabb_xy = G.aabb_corners(lo, hi)[:, :2]
        cands = []
        for name, mid, normal, length in top.edges():
            if edge_name is not None and name != edge_name:
                continue
            inward = -normal
            overhang = max(0.0, float((aabb_xy @ normal).max()) - float(th.dot(mid, normal)))
            if overhang < 0.05:
                overhang = 0.0
            base_xy = mid + normal * (stand_off + overhang)
            yaw = math.atan2(float(inward[1]), float(inward[0]))
            if not kit.trav_free(base_xy):
                continue
            q, posture = self.trunk_config(0.5, 0.0, float(base_xy[0]), float(base_xy[1]), yaw)
            free = q is not None
            cands.append((free, length, name, mid, inward, base_xy, yaw, overhang))
        if not cands:
            raise RuntimeError(f"no stance on free floor around {top.name}")
        cands.sort(key=lambda c: (not c[0], -c[1]))
        free, length, name, mid, inward, base_xy, yaw, overhang = cands[0]
        if not free:
            log.warning("every edge stance of %s collides at trunk (0.5, 0); using %s anyway", top.name, name)
        tangent = th.tensor([-float(inward[1]), float(inward[0])])
        return {"edge": name, "edge_mid": mid.tolist(), "inward": inward.tolist(), "tangent": tangent.tolist(), "length_m": float(length),
                "base_pose2d": [float(base_xy[0]), float(base_xy[1]), yaw], "stand_off": stand_off, "overhang": overhang,
                "edge_depth": float(th.dot(mid - base_xy, inward))}

    def stance_free_floor(self, room: Optional[str], facing: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """Floor mode: the room's floor centroid with the base facing +x (or ``facing``), nudged to a free cell."""
        kit, world = self.kit, self.world
        floor = world.floor_of_room(room) if room else None
        lo, hi = (floor.aabb if floor is not None else (th.tensor([-1.0, -1.0, 0.0]), th.tensor([1.0, 1.0, 0.0])))
        centre = th.tensor([float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2])
        yaw = math.atan2(float(facing[1]), float(facing[0])) if facing is not None else 0.0
        best = None
        for r in th.arange(0.0, 2.01, 0.25).tolist():
            for a in th.linspace(0.0, 2 * math.pi, 17)[:-1].tolist():
                xy = centre + th.tensor([r * math.cos(a), r * math.sin(a)])
                if not kit.trav_free(xy, eroded=True):
                    continue
                q, _ = self.trunk_config(0.5, 0.0, float(xy[0]), float(xy[1]), yaw)
                if q is not None:
                    best = xy
                    break
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"no collision-free floor stance in room {room}")
        inward = th.tensor([math.cos(yaw), math.sin(yaw)])
        return {"edge": None, "edge_mid": best.tolist(), "inward": inward.tolist(), "tangent": [-float(inward[1]), float(inward[0])],
                "base_pose2d": [float(best[0]), float(best[1]), yaw], "stand_off": 0.0, "overhang": 0.0, "edge_depth": 0.0}

    def stance_under(self, partner, stand_off: float = STAND_OFF) -> Dict[str, Any]:
        """IORS under: base square to the partner's near AABB edge (the edge with free floor in front), at ``stand_off``."""
        kit = self.kit
        lo, hi = partner.aabb
        cx, cy = float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2
        cands = []
        for normal, mid, length in ((th.tensor([1.0, 0.0]), th.tensor([float(hi[0]), cy]), float(hi[1] - lo[1])),
                                    (th.tensor([-1.0, 0.0]), th.tensor([float(lo[0]), cy]), float(hi[1] - lo[1])),
                                    (th.tensor([0.0, 1.0]), th.tensor([cx, float(hi[1])]), float(hi[0] - lo[0])),
                                    (th.tensor([0.0, -1.0]), th.tensor([cx, float(lo[1])]), float(hi[0] - lo[0]))):
            base_xy = mid + normal * stand_off
            if not kit.trav_free(base_xy):
                continue
            inward = -normal
            yaw = math.atan2(float(inward[1]), float(inward[0]))
            q, _ = self.trunk_config(0.5, 0.0, float(base_xy[0]), float(base_xy[1]), yaw)
            cands.append((q is not None, length, mid, inward, base_xy, yaw))
        if not cands:
            raise RuntimeError(f"no stance around {partner.name}")
        cands.sort(key=lambda c: (not c[0], -c[1]))
        _, length, mid, inward, base_xy, yaw = cands[0]
        return {"edge": "aabb", "edge_mid": mid.tolist(), "inward": inward.tolist(), "tangent": [-float(inward[1]), float(inward[0])], "length_m": length,
                "base_pose2d": [float(base_xy[0]), float(base_xy[1]), yaw], "stand_off": stand_off, "overhang": 0.0,
                "edge_depth": float(th.dot(mid - base_xy, inward))}

    # -- lattice ------------------------------------------------------------------------------------------------
    def base_frame(self, stance: Dict[str, Any]):
        x0, y0, yaw = stance["base_pose2d"]
        fwd = th.tensor([math.cos(yaw), math.sin(yaw)])
        left = th.tensor([-math.sin(yaw), math.cos(yaw)])
        return th.tensor([x0, y0]), fwd, left

    def lattice_on_cells(self, stance: Dict[str, Any], world_cells: th.Tensor, x_range: Tuple[float, float] = (0.0, STRIP_MAX_DEPTH + STAND_OFF + 0.2),
                         lat_half: float = STRIP_HALF_WIDTH) -> Tuple[th.Tensor, th.Tensor]:
        """Snap world cells (support top / tray floor / floor raster) to the base‑frame lattice: returns (base_xy (N,2), world_xy (N,2))."""
        origin, fwd, left = self.base_frame(stance)
        rel = world_cells - origin
        x = rel @ fwd
        lat = rel @ left
        keep = (x >= x_range[0]) & (x <= x_range[1]) & (lat.abs() <= lat_half)
        x, lat = x[keep], lat[keep]
        ix = th.round(x / self.lattice).long()
        il = th.round(lat / self.lattice).long()
        seen = {}
        for i in range(len(x)):
            k = (int(ix[i]), int(il[i]))
            if k not in seen:
                seen[k] = (float(ix[i]) * self.lattice, float(il[i]) * self.lattice)
        base = th.tensor(list(seen.values())) if seen else th.zeros(0, 2)
        world = origin + base[:, :1] * fwd + base[:, 1:] * left if len(base) else th.zeros(0, 2)
        return base, world

    # -- object poses / grasp sets ------------------------------------------------------------------------------
    def object_pose_at(self, world_xy, yaw: float, z_support: float, hull, rest_quat) -> Tuple[th.Tensor, th.Tensor]:
        G = self.G
        quat = G.qmul(G.quat_from_yaw(yaw), rest_quat)
        pos = th.tensor([float(world_xy[0]), float(world_xy[1]), z_support + self.world.bottom_offset(hull, quat) + 0.003])
        return pos, quat

    def grasp_eef_pair(self, obj_pose, grasp) -> List[Tuple[th.Tensor, th.Tensor]]:
        G = self.G
        eef = G.compose(obj_pose, grasp.pose)
        axis = G.rotate(eef[1], th.tensor([0.0, 0.0, 1.0]))
        return [(eef[0] - axis * PRE_GRASP, eef[1]), eef]

    def _shoulder_offset(self, arm: str) -> th.Tensor:
        """Shoulder (``<arm>_arm_link1`` origin) in the ``torso_link4`` frame - constant, measured once on the simulated
        robot (cuRobo's FK only knows the tool frames, the head camera, base_link and torso_link4)."""
        cache = getattr(self, "_sh_off", {})
        if arm not in cache:
            G = self.G
            t4 = self.world.robot.links["torso_link4"].get_position_orientation()
            sh = self.world.robot.links[f"{arm}_arm_link1"].get_position_orientation()
            cache[arm] = G.relative((sh[0].float(), sh[1].float()), (t4[0].float(), t4[1].float()))[0]
            self._sh_off = cache
        return cache[arm]

    def shoulder_and_reach(self, q: th.Tensor, arm: str) -> Tuple[th.Tensor, float]:
        """Shoulder position (world) at ``q`` and the arm's stretched length (shoulder -> eef with every arm joint at zero,
        the arm hanging straight)."""
        M, G = self.M, self.G
        t4 = self.kit.fk(["torso_link4"], q)["torso_link4"]
        sh = G.compose(t4, (self._shoulder_offset(arm), G.IDENTITY_QUAT))[0]
        if not hasattr(self, "_arm_len"):
            self._arm_len = {}
        if arm not in self._arm_len:
            qz = G.t(q).clone()
            qz[self.world.robot.arm_control_idx[arm]] = 0.0
            ee = self.kit.fk([M.EEF_LINK[arm]], qz)[M.EEF_LINK[arm]][0]
            self._arm_len[arm] = float(th.linalg.norm(ee - sh))
        return sh, self._arm_len[arm]

    # -- main loop ----------------------------------------------------------------------------------------------
    def compute(self, stance: Dict[str, Any], lattice_world: th.Tensor, z_support: float, pose_fn, grasp_sets, ignore_for_grasp, arm: str = "right",
                trunks: Optional[Sequence[Tuple[float, float]]] = None, max_grasp_offset: float = 0.0) -> Dict[str, Any]:
        """``pose_fn(world_xy, yaw) -> obj_pose`` and ``grasp_sets`` = list of grasps (with ``.pose``; rank order = library
        order) or ``None`` for the generic set (the generic pinch / push classes, or - with ``self.contact_height`` set -
        the generic CONTACT classes of the brace arm). Returns the per-trunk results: ``trunks`` (entries, incl. the
        idle-arm ``posture`` used), ``cells`` ((translate, tilt) -> (N, 2) reachable cells), ``pairs`` ((translate, tilt)
        -> (M, 3) int ``[cell index, yaw index into YAWS (-1 generic), grasp / class index]`` = the pairs found feasible
        for a cell in the round that marked it; the seed of the feasible set F) and ``classes`` ((translate, tilt) ->
        (N,) int bitmask of the generic / contact classes that passed per cell; zeros for library grasps)."""
        kit, G = self.kit, self.G
        base_cells, world_cells = self.lattice_on_cells(stance, lattice_world)
        trunks = list(trunks) if trunks is not None else trunk_grid(z_support - self.world.floor_z)
        x0, y0, yaw0 = stance["base_pose2d"]
        _, fwd, left = self.base_frame(stance)
        results = []
        # object yaw offsets relative to the base yaw, robot-facing first (canonical +x of the object toward the robot = offset pi)
        yaw_order = sorted(YAWS, key=lambda y: abs(G.wrap_angle(y - math.pi)))
        yaw_index = {y: k for k, y in enumerate(YAWS)}
        grasp_index = {g.name: k for k, g in enumerate(grasp_sets)} if grasp_sets is not None else {}
        common = (stance, z_support, pose_fn, grasp_sets, ignore_for_grasp, arm, max_grasp_offset, yaw_order, yaw_index, grasp_index, fwd, left)
        self.scope = ObstacleScope(kit, self.world, (x0, y0), drop_ground=(z_support - self.world.floor_z) >= 0.55).__enter__()
        try:
            if self.cell_obstacle is not None:
                self.cell_obstacle.hide()  # the stance-relative placed object enters per cell (_verify_with_placed)
            for tr, ti in trunks:
                results.append(self._trunk(tr, ti, base_cells, world_cells, *common))
        finally:
            self.scope.__exit__(None, None, None)
            self.scope = None
            if self.cell_obstacle is not None:
                self.cell_obstacle.show()
        return {"trunks": [e for e, _, _, _ in results], "cells": {(e["translate"], e["tilt"]): c for e, c, _, _ in results},
                "pairs": {(e["translate"], e["tilt"]): p for e, _, p, _ in results}, "classes": {(e["translate"], e["tilt"]): m for e, _, _, m in results}}

    def _verify_with_placed(self, hits: List[int], pairs, index, q: th.Tensor, arm: str, world_cells, entry: Dict[str, Any]) -> List[int]:
        """Per-cell pre-grasp verification against the stance-relative placed object (:attr:`cell_obstacle`): for every
        cell with a hit this round, the object is moved to the cell's pose and the hit pairs' pre-grasps are screened
        again with it enabled; pairs whose pre-grasp collides with it are dropped (``entry["n_placed_rejected_pairs"]``
        / ``n_placed_rejected_cells`` count them). Returns the surviving indices into ``pairs``."""
        by_cell: Dict[int, List[int]] = {}
        for k in hits:
            by_cell.setdefault(index[k], []).append(k)
        keep: List[int] = []
        t0 = time.time()
        try:
            for i, ks in by_cell.items():
                self.cell_obstacle.place(world_cells[i])
                ok_pre = self.ik_screen([pairs[k][0] for k in ks], q, arm)
                surv = [k for k, s in zip(ks, ok_pre.tolist()) if s]
                keep.extend(surv)
                entry["n_placed_rejected_pairs"] = entry.get("n_placed_rejected_pairs", 0) + (len(ks) - len(surv))
                if not surv:
                    entry["n_placed_rejected_cells"] = entry.get("n_placed_rejected_cells", 0) + 1
        finally:
            self.cell_obstacle.hide()
        entry["placed_verify_s"] = round(entry.get("placed_verify_s", 0.0) + time.time() - t0, 2)
        return keep

    def _trunk(self, tr, ti, base_cells, world_cells, stance, z_support, pose_fn, grasp_sets, ignore_for_grasp, arm, max_grasp_offset, yaw_order, yaw_index,
               grasp_index, fwd, left, tag: str = ""):
        """One trunk of :meth:`compute` over the given cells; returns ``(entry, cells (N, 2), pairs (M, 3), classes (N,))``."""
        kit = self.kit
        x0, y0, yaw0 = stance["base_pose2d"]
        empty = (np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int32))
        if True:
            t0 = time.time()
            entry = {"translate": tr, "tilt": ti, "collides": False, "n_lattice": int(len(base_cells)), "n_after_prune": 0, "n_reachable": 0, "rounds": 0,
                     "n_pairs": 0, "n_dropped_no_pre": 0, "n_pose_pruned": 0, "time_s": 0.0, "posture": None}
            q, posture = self.trunk_config(tr, ti, x0, y0, yaw0)
            if q is None:
                entry["collides"] = True
                entry["time_s"] = round(time.time() - t0, 2)
                log.info("%strunk (%.2f, %+.1f): collides with every idle-arm posture %s, skipped", tag, tr, ti, list(POSTURES))
                return (entry,) + empty
            entry["posture"] = posture
            entry["idle_arm_q"] = self.idle_arm_q(q)
            if posture != "untucked":
                log.info("%strunk (%.2f, %+.1f): untucked arms collide, using the %s posture", tag, tr, ti, posture)
            shoulder, arm_len = self.shoulder_and_reach(q, arm)
            reach = arm_len + max_grasp_offset + PRE_GRASP + 0.05
            entry["shoulder_world"] = [round(float(v), 4) for v in shoulder.tolist()]
            entry["arm_len"] = round(arm_len, 4)
            entry["reach"] = round(reach, 4)
            pts = th.cat([world_cells, th.full((len(world_cells), 1), z_support)], dim=1)
            near = th.linalg.norm(pts - shoulder, dim=1) <= reach
            todo = [i for i in range(len(base_cells)) if bool(near[i])]
            entry["n_after_prune"] = len(todo)
            ok = th.zeros(len(base_cells), dtype=th.bool)
            pre_seen = th.zeros(len(base_cells), dtype=th.bool)  # some pre-grasp pose of the cell was IK-feasible
            found: List[Tuple[int, int, int]] = []  # (cell, yaw index, grasp index)
            n_pose_pruned = 0
            # rounds: (grasp rank, yaw) pairs in chunks; every round is one batched pre-grasp + grasp IK call over the open cells
            if grasp_sets is None:
                pair_specs = [("generic", None, None)]
            else:
                pair_specs = [(g.name, g, y) for g in grasp_sets for y in yaw_order]
            chunk = self.CHUNK if grasp_sets is not None else 1
            pose_reach = arm_len + 0.03  # the eef origin can never be farther from the shoulder than the stretched arm
            for r0 in range(0, len(pair_specs), chunk):
                if not todo:
                    break
                specs = pair_specs[r0:r0 + chunk]
                pairs, index, spec_of = [], [], []
                for i in todo:
                    if grasp_sets is None and self.contact_height is not None:
                        # generic CONTACT set: the cell is the contact point; z_support = the top the object stands on
                        cands = [((-1, ci), pair) for ci, pair in self.contact_candidates_at(world_cells[i], z_support, self.contact_height, fwd, left)]
                    elif grasp_sets is None:
                        centre = th.tensor([float(world_cells[i][0]), float(world_cells[i][1]), z_support])
                        cands = [((-1, ci), pair) for ci, pair in enumerate(kit.grasp_candidates_at(centre, [yaw0, yaw0 + math.pi / 2][:N_YAWS_GENERIC], TILTS_GENERIC, fwd))]
                    else:
                        cands = [((yaw_index[y], grasp_index[name]), self.grasp_eef_pair(pose_fn(world_cells[i], y + yaw0), g)) for name, g, y in specs]
                    for spec, pair in cands:
                        if float(th.linalg.norm(pair[1][0] - shoulder)) > pose_reach or float(th.linalg.norm(pair[0][0] - shoulder)) > pose_reach:
                            n_pose_pruned += 1
                            continue
                        pairs.append(pair)
                        index.append(i)
                        spec_of.append(spec)
                if pairs:
                    pres = [p[0] for p in pairs]
                    ok_pre = self.ik_screen(pres, q, arm)
                    sel = [k for k, s in enumerate(ok_pre.tolist()) if s]
                    for k in sel:
                        pre_seen[index[k]] = True
                    hits: List[int] = []  # indices into ``pairs`` feasible at pre-grasp and grasp this round
                    if sel:
                        with self.ignoring(ignore_for_grasp):
                            ok_g = self.ik_screen([pairs[k][1] for k in sel], q, arm)
                        hits = [k for k, s in zip(sel, ok_g.tolist()) if s]
                    if hits and self.cell_obstacle is not None:
                        # stance-relative placed object: the batched passes ran without it; verify the pre-grasps of every
                        # passing cell with the object placed at that cell (the grasp pose ignores it, spec 3.6)
                        hits = self._verify_with_placed(hits, pairs, index, q, arm, world_cells, entry)
                    for k in hits:
                        ok[index[k]] = True
                        found.append((index[k], spec_of[k][0], spec_of[k][1]))
                entry["rounds"] += 1
                todo = [i for i in todo if not bool(ok[i])]
                if self.MAX_ROUNDS_WITHOUT_PRE and entry["rounds"] >= self.MAX_ROUNDS_WITHOUT_PRE:
                    n_before = len(todo)
                    todo = [i for i in todo if bool(pre_seen[i])]
                    entry["n_dropped_no_pre"] = entry.get("n_dropped_no_pre", 0) + (n_before - len(todo))
            entry["n_pose_pruned"] = n_pose_pruned
            sel_cells = [i for i in range(len(base_cells)) if bool(ok[i])]
            remap = {i: n for n, i in enumerate(sel_cells)}
            cells = base_cells[ok].numpy() if sel_cells else np.zeros((0, 2))
            pairs_arr = np.array([(remap[c], yi, gi) for c, yi, gi in found], dtype=np.int32).reshape(-1, 3)
            classes_arr = np.zeros(len(sel_cells), dtype=np.int32)
            if grasp_sets is None:  # generic pinch / contact classes: bit k = class k passed at the cell
                for c, yi, gi in found:
                    classes_arr[remap[c]] |= 1 << int(gi)
                entry["class_counts"] = {n: int(((classes_arr >> b) & 1).sum()) for b, n in enumerate(CONTACT_CLASSES_NAMES if self.contact_height is not None else GENERIC_CLASSES)}
            entry["n_reachable"] = len(sel_cells)
            entry["n_pairs"] = int(len(pairs_arr))
            entry["time_s"] = round(time.time() - t0, 2)
            placed_note = (f", placed object rejected {entry.get('n_placed_rejected_pairs', 0)} pairs / {entry.get('n_placed_rejected_cells', 0)} cells in "
                           f"{entry.get('placed_verify_s', 0.0):.1f}s" if self.cell_obstacle is not None else "")
            log.info("%strunk (%.2f, %+.1f): %d lattice, %d after prune, %d reachable (%d pairs), %d rounds (%d cells dropped without pre-grasp, %d poses pruned), %.1fs%s%s",
                     tag, tr, ti, len(base_cells), entry["n_after_prune"], entry["n_reachable"], entry["n_pairs"], entry["rounds"], entry["n_dropped_no_pre"],
                     entry["n_pose_pruned"], entry["time_s"], f" classes {entry['class_counts']}" if "class_counts" in entry else "", placed_note)
            return entry, cells, pairs_arr, classes_arr


# ----------------------------------------------------------------------------------------------- articulation (brace mode)


def handle_pose_at_q(joint_info: Dict[str, Any], handle: Dict[str, Any], q: float, obj_pose, G):
    """Handle contact / pinch / approach moved with the moving link to joint value ``q`` (closed configuration at
    ``lower``): revolute = rotation about (hinge_point_obj, axis_dir_obj), prismatic = translation along axis_dir_obj.
    Returns [pre-grasp, grasp] eef poses in the world."""
    import omnigibson.utils.transform_utils as T

    lower = float(joint_info.get("closed_q", joint_info.get("lower", 0.0)))
    contact = th.tensor(handle["contact"], dtype=th.float32)
    approach = th.tensor(handle["approach_dir"], dtype=th.float32)
    pinch = th.tensor(handle["pinch_axis"], dtype=th.float32)
    axis = th.tensor(joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
    axis = axis / th.linalg.norm(axis)
    if joint_info.get("type", "RevoluteJoint") == "RevoluteJoint":
        rot = G.quat_about_axis(axis, q - lower)
        hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
        contact = hinge + G.rotate(rot, contact - hinge)
        approach = G.rotate(rot, approach)
        pinch = G.rotate(rot, pinch)
    else:
        contact = contact + axis * (q - lower)
    # object frame -> world
    c_w = G.compose(obj_pose, (contact, G.IDENTITY_QUAT))[0]
    a_w = G.rotate(obj_pose[1], approach)
    p_w = G.rotate(obj_pose[1], pinch)
    z = a_w / th.linalg.norm(a_w)
    y = p_w - th.dot(p_w, z) * z
    y = y / th.linalg.norm(y)
    x = th.cross(y, z, dim=0)
    quat = T.mat2quat(th.stack([x, y, z], dim=1))
    eef = c_w - z * 0.012
    return [(eef - z * PRE_GRASP, quat), (eef, quat)]


class BraceCandidate(tuple):
    """``(kind, [pre, contact])`` (unpacks like the 2-tuple the v2 articulation module expects) with the number of leading
    joint values the contact is valid for as ``q_prefix``."""

    def __new__(cls, kind: str, pair, q_prefix: int):
        self = super().__new__(cls, (kind, pair))
        self.q_prefix = q_prefix
        return self


def brace_candidates(joint_info: Dict[str, Any], handle_data: Dict[str, Any], obj_pose, q_values: Sequence[float], brace_side: float, G):
    """Brace contacts on the base link (never the moving link): ``opposite_face`` (base-link face whose outward normal is
    closest to the push direction v, contact at the face centroid shifted 1/4 of the face width toward the brace arm's
    side, approach -v) and ``top_press`` (downward press on the exposed top of the base link nearest the brace arm's
    side). Contacts must stay >= 2 cm outside the moving link's swept AABB over the joint values; the swept AABB of a
    lid that flips all the way back covers the whole body, so the check is made for the longest prefix of ``q_values``
    (the waypoints the brace protects, in order) that leaves a candidate. Returns [(kind, [pre, contact], q_prefix)]
    in preference order, eef poses in the world; ``q_prefix`` = the number of leading joint values the contact is
    valid for (the sampler requires its direction's waypoints to lie within that prefix)."""
    import omnigibson.utils.transform_utils as T

    bl = handle_data.get("base_link", {})
    bmin, bmax = th.tensor(bl.get("bbox_min", [-0.1, -0.1, 0.0])), th.tensor(bl.get("bbox_max", [0.1, 0.1, 0.1]))
    # push direction: horizontal handle motion toward the first waypoint; if < 30 % of the motion is horizontal (lid lifted
    # straight up) use the direction from the hinge line to the handle contact
    h0 = handle_pose_at_q(joint_info, handle_data["handle"], q_values[0], (th.zeros(3), G.IDENTITY_QUAT), G)[1][0]
    h1 = handle_pose_at_q(joint_info, handle_data["handle"], q_values[min(1, len(q_values) - 1)], (th.zeros(3), G.IDENTITY_QUAT), G)[1][0]
    d = h1 - h0
    if float(th.linalg.norm(d)) < 1e-6 or float(th.linalg.norm(d[:2])) < 0.3 * float(th.linalg.norm(d)):
        hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
        d = th.tensor(handle_data["handle"]["contact"], dtype=th.float32) - hinge
    v = th.tensor([float(d[0]), float(d[1]), 0.0])
    v = v / max(float(th.linalg.norm(v)), 1e-6)
    # moving link swept AABB (object frame) over the joint values, from the annotated link bbox at the closed configuration
    lmin, lmax = th.tensor(joint_info.get("link_bbox_min", bmin.tolist())), th.tensor(joint_info.get("link_bbox_max", bmax.tolist()))
    corners = th.stack([th.tensor([x, y, z]) for x in (lmin[0], lmax[0]) for y in (lmin[1], lmax[1]) for z in (lmin[2], lmax[2])])
    axis = th.tensor(joint_info.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
    axis = axis / th.linalg.norm(axis)
    lower = float(joint_info.get("closed_q", joint_info.get("lower", 0.0)))
    per_q = [corners]
    for qv in q_values:
        if joint_info.get("type", "RevoluteJoint") == "RevoluteJoint":
            rot = G.quat_about_axis(axis, qv - lower)
            hinge = th.tensor(joint_info.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            per_q.append(hinge + th.stack([G.rotate(rot, c - hinge) for c in corners]))
        else:
            per_q.append(corners + axis * (qv - lower))
    faces = [(th.tensor([1.0, 0, 0]), 0, bmax[0]), (th.tensor([-1.0, 0, 0]), 0, bmin[0]), (th.tensor([0, 1.0, 0]), 1, bmax[1]), (th.tensor([0, -1.0, 0]), 1, bmin[1])]
    # opposite_face: outward normal closest to v (the face the handle motion points toward)
    n, ax, coord = max(faces, key=lambda f: float(th.dot(f[0], v)))
    other = 1 - ax
    centre = (bmin + bmax) / 2
    p = centre.clone()
    p[ax] = coord
    side_dir = th.tensor([-float(v[1]), float(v[0]), 0.0]) * brace_side  # brace arm's side along the face
    p = p + side_dir * (float(bmax[other] - bmin[other]) / 4)
    top = th.tensor([float(centre[0]), float(centre[1]), float(bmax[2])]) + side_dir * (float(bmax[other] - bmin[other]) / 3)
    out = []
    q_prefix = 0
    for k in range(len(q_values), 0, -1):  # longest prefix of the joint values whose swept AABB leaves a candidate
        swept = th.cat(per_q[:k + 1])
        smin, smax = swept.min(dim=0).values - 0.02, swept.max(dim=0).values + 0.02
        cands = []
        if not bool((p >= smin).all() and (p <= smax).all()):
            cands.append(("opposite_face", p, -n))
        if not bool((top >= smin).all() and (top <= smax).all()):
            cands.append(("top_press", top, th.tensor([0.0, 0.0, -1.0])))
        if cands:
            out, q_prefix = cands, k
            break
    poses = []
    for kind, p_obj, a_obj in out:
        c_w = G.compose(obj_pose, (p_obj, G.IDENTITY_QUAT))[0]
        a_w = G.rotate(obj_pose[1], a_obj)
        z = a_w / th.linalg.norm(a_w)
        y_ref = G.rotate(obj_pose[1], th.tensor([0.0, 1.0, 0.0]))  # closing axis = object +y
        y = y_ref - th.dot(y_ref, z) * z
        if float(th.linalg.norm(y)) < 1e-3:
            y_ref = G.rotate(obj_pose[1], th.tensor([1.0, 0.0, 0.0]))
            y = y_ref - th.dot(y_ref, z) * z
        y = y / th.linalg.norm(y)
        x = th.cross(y, z, dim=0)
        quat = T.mat2quat(th.stack([x, y, z], dim=1))
        eef = c_w - z * 0.012
        poses.append(BraceCandidate(kind, [(eef - z * PRE_GRASP, quat), (eef, quat)], q_prefix))
    return poses


# ----------------------------------------------------------------------------------------------- smoke checks


def verify_shoulder_fk(world, kit, bc: BandComputer, trunks: Sequence[Tuple[float, float]], arm: str = "right") -> List[Dict[str, Any]]:
    """Check the shoulder position the band computation uses (cuRobo FK of ``torso_link4`` + the measured constant offset)
    against the simulated robot's ``<arm>_arm_link1`` (and ``torso_link4``) world poses at the same joint state.

    The parked robot is put into the untucked configuration of every trunk (base untouched), one physics step, then
    both sides are read for the *same* full joint vector (``kit.q_current()`` includes the virtual base joints of the
    parked robot). Returns per trunk the errors in mm."""
    import omnigibson as og
    from ispatialgym.oracle import r1pro_model as M

    robot = world.robot
    q_saved = robot.get_joint_positions().clone()
    idx = th.cat([robot.trunk_control_idx] + [robot.arm_control_idx[a] for a in M.ARMS] + [robot.gripper_control_idx[a] for a in M.ARMS])
    out = []

    def settle(qs):
        robot.set_joint_positions(qs)
        robot.set_joint_velocities(th.zeros_like(qs))
        robot.keep_still()
        og.sim.step_physics()
        robot.set_joint_positions(qs)
        robot.keep_still()

    try:
        for tr, ti in trunks:
            q = kit.q_config(arms="untucked", translate=tr, tilt=ti)
            qs = q_saved.clone()
            qs[idx] = q[idx]
            settle(qs)
            sim_sh = robot.links[f"{arm}_arm_link1"].get_position_orientation()[0].float()
            sim_t4 = robot.links["torso_link4"].get_position_orientation()[0].float()
            sim_ee = robot.eef_links[arm].get_position_orientation()[0].float()
            q_now = kit.q_current()
            fk_sh, arm_len = bc.shoulder_and_reach(q_now, arm)
            fk_t4 = kit.fk(["torso_link4"], q_now)["torso_link4"][0]
            fk_ee = kit.eef_pose(q_now, arm)[0]
            e_sh = float(th.linalg.norm(fk_sh - sim_sh)) * 1000
            e_t4 = float(th.linalg.norm(fk_t4 - sim_t4)) * 1000
            e_ee = float(th.linalg.norm(fk_ee - sim_ee)) * 1000
            rec = {"translate": tr, "tilt": ti, "shoulder_err_mm": round(e_sh, 2), "torso_link4_err_mm": round(e_t4, 2), "eef_err_mm": round(e_ee, 2),
                   "shoulder_sim": [round(float(v), 4) for v in sim_sh.tolist()], "shoulder_fk": [round(float(v), 4) for v in fk_sh.tolist()],
                   "shoulder_above_floor_m": round(float(sim_sh[2]) - world.floor_z, 4), "arm_len_m": round(arm_len, 4)}
            log.info("shoulder FK check trunk (%.2f, %+.1f): shoulder err %.1f mm, torso_link4 err %.1f mm, eef err %.1f mm (shoulder %.3f m above the floor)",
                     tr, ti, e_sh, e_t4, e_ee, rec["shoulder_above_floor_m"])
            out.append(rec)
    finally:
        settle(q_saved)
    return out


def band_sanity(stance: Dict[str, Any], result: Dict[str, Any], z_support: float) -> Dict[str, Any]:
    """Geometric summary of a tabletop / floor result for the smoke report: extents of the reachable cells in base
    coordinates, whether any lies behind the base (x <= 0) and the shoulder -> cell distances per trunk."""
    per_trunk = []
    for t in result["trunks"]:
        cells = result["cells"][(t["translate"], t["tilt"])]
        rec = {"translate": t["translate"], "tilt": t["tilt"], "collides": t["collides"], "n_reachable": int(len(cells))}
        if len(cells):
            rec.update({"x_min": round(float(cells[:, 0].min()), 3), "x_max": round(float(cells[:, 0].max()), 3), "lat_min": round(float(cells[:, 1].min()), 3),
                        "lat_max": round(float(cells[:, 1].max()), 3), "n_behind_base": int((cells[:, 0] <= 0).sum())})
            if t.get("shoulder_world") is not None:
                x0, y0, yaw = stance["base_pose2d"]
                c, s = math.cos(yaw), math.sin(yaw)
                wx = x0 + c * cells[:, 0] - s * cells[:, 1]
                wy = y0 + s * cells[:, 0] + c * cells[:, 1]
                sh = np.asarray(t["shoulder_world"])
                d = np.sqrt((wx - sh[0]) ** 2 + (wy - sh[1]) ** 2 + (z_support - sh[2]) ** 2)
                rec.update({"shoulder_dist_min": round(float(d.min()), 3), "shoulder_dist_max": round(float(d.max()), 3)})
        pairs = result.get("pairs", {}).get((t["translate"], t["tilt"]))
        if pairs is not None and len(cells):
            rec["cells_without_pair"] = int(len(cells) - len(set(pairs[:, 0].tolist())))
        per_trunk.append(rec)
    return {"edge_depth": stance.get("edge_depth"), "per_trunk": per_trunk}


# ----------------------------------------------------------------------------------------------- driver


def _stable_scene(scene: str) -> Dict[str, Any]:
    """The stable scene json (``omnigibson`` package import only; no simulator launch)."""
    from omnigibson.utils.asset_utils import get_dataset_path

    with open(os.path.join(get_dataset_path("2026-challenge-task-instances"), "scenes", scene, "json", f"{scene}_stable.json")) as f:
        return json.load(f)


def _chairs_to_remove(scene: str, support_name: str, rooms: List[str], stable: Optional[Dict[str, Any]] = None) -> List[str]:
    """Chairs within 1 m of the support (removed for the band like the sampler does); sorted names, no file written."""
    from ispatialgym.sampling import scene_variants as SV

    stable = _stable_scene(scene) if stable is None else stable
    return SV.chairs_near_tables(stable, rooms, [support_name], distance=1.0, categories=SV.CHAIR_CATEGORIES)


#: cuRobo mesh store: capacity = loaded collision meshes + this headroom for the objects bands spawn (a target object is
#: 1-6 meshes, a placed tray a few more; the worker reloads the world when fewer than MESH_CACHE_RESERVE slots are left),
#: at least MESH_CACHE_MIN. The capacity, not the fill, sizes every collision kernel launch (R1ProCuRobo.__init__).
MESH_CACHE_HEADROOM = 256
MESH_CACHE_MIN = 512
MESH_CACHE_RESERVE = 64

#: slim load (``_slim_drop_list``): safety margin (m) added to every bounding box derived from the stable scene json before
#: the ``SimWorld.obstacle_filter`` test - the json knows the canonical (visual) bounding box, the filter measures the
#: collision meshes; articulated objects get their largest extent on top (an open door leaf / drawer reaches that far).
#: Calibrated 2026-09-16 on the 420 loaded objects of house_single_floor (dining room + neighbours): json box vs loaded
#: collision AABB mismatch median 0.0 m, 3 objects > 0.1 m, max 0.33 m (a rail fence); with 1.0 m every dropped object's
#: true AABB stayed >= 2.16 m outside the true filter bounds while 313 of 421 objects were dropped (0.5 m: 328, gap 1.2 m).
SLIM_MARGIN_M = 1.0


def _json_aabb(init_args: Dict[str, Any], state: Dict[str, Any], margin: float, articulated: bool) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """World AABB ``(lo, hi)`` of an object from the stable scene json alone: the scaled canonical bounding box
    (``knowledge.canonical_bbox`` + ``base_link_offset``, as ``scene_variants._bbox_footprint``) rotated by the full
    orientation, grown by ``margin`` (+ the largest extent for ``articulated`` objects); ``None`` when unknown."""
    from ispatialgym.robot.conventions import quat_to_mat
    from ispatialgym.sampling import knowledge as K
    from ispatialgym.sampling import scene_variants as SV

    model = init_args.get("model")
    bb = K.canonical_bbox(model) if model else None
    if bb is None:
        return None
    meta = SV._metadata(init_args.get("category"), model)
    scale = th.tensor(init_args.get("scale") or [1.0, 1.0, 1.0], dtype=th.float32)
    offset = th.tensor(meta.get("base_link_offset", [0.0, 0.0, 0.0]), dtype=th.float32)
    pos = th.tensor(state["root_link"]["pos"], dtype=th.float32)
    rot = quat_to_mat(state["root_link"]["ori"]).float()
    # bbox centre = base link origin + scale * base_link_offset (OmniGibson's ``scaled_bbox_center_in_base_frame`` is
    # the base link *in the bbox frame*, i.e. the negative; checked 2026-09-16 against the loaded collision AABBs of 420
    # objects: with this sign the median mismatch is 3 cm, with the other one whole windows / floors were displaced)
    centre = pos + rot @ (scale * offset)
    half = th.tensor(bb, dtype=th.float32) * scale / 2
    corners = th.tensor([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]) * half
    world = centre + corners @ rot.T
    grow = margin + (float(half.max()) * 2 if articulated else 0.0)
    return world.min(dim=0).values - grow, world.max(dim=0).values + grow


def _slim_drop_list(stable: Dict[str, Any], rooms: Sequence[str], loaded_rooms: Sequence[str], obstacle_margin: Optional[float] = None,
                    margin: float = SLIM_MARGIN_M) -> List[str]:
    """Objects of the loaded neighbouring rooms that cannot pass ``SimWorld.obstacle_filter`` and are therefore never
    collision obstacles of a band: their json AABB (:func:`_json_aabb`, grown by ``margin``) misses the task rooms' floor
    extent grown by the filter's ``ISG_OBSTACLE_MARGIN_M`` (and by ``margin`` again, the floor extent being a json
    estimate too).     Everything in the task rooms, everything with an unknown box and everything a kept non-structure
    object might rest on (xy overlap, top within 0.3 m of the kept object's bottom; closed under iteration) stays.
    Candidates are what OmniGibson's partial-room loading brings in: objects of the loaded rooms **plus every building
    structure and door of the whole scene** (``InteractiveTraversableScene._should_load_object`` loads walls, ceilings,
    roofs, fences and doors regardless of ``load_room_instances`` - 147 wall segments in house_single_floor), which is
    where most of the saving is. Conservative in the only direction that matters: an object kept needlessly is filtered
    out of the collision world at load time exactly as before; an object dropped wrongly would change the world.
    Sorted names."""
    from omnigibson.utils.constants import GROUND_CATEGORIES as OG_GROUND
    from omnigibson.utils.constants import STRUCTURE_CATEGORIES as OG_STRUCTURE

    from ispatialgym.sampling.knowledge import STRUCTURE_CATEGORIES

    always_loaded = (set(OG_STRUCTURE) - set(OG_GROUND)) | {"door", "sliding_door"}  # the loader's is_building_structure rule
    init = stable["objects_info"]["init_info"]
    reg = stable["state"]["registry"]["object_registry"]
    rooms_set, loaded_set = set(rooms), set(loaded_rooms)
    m_obst = float(os.environ.get("ISG_OBSTACLE_MARGIN_M", "1.0")) if obstacle_margin is None else float(obstacle_margin)
    los, his = [], []
    for name, info in init.items():
        a = info["args"]
        if a.get("category") == "floors" and set(a.get("in_rooms") or []) & rooms_set and name in reg:
            box = _json_aabb(a, reg[name], margin, False)
            if box is not None:
                los.append(box[0])
                his.append(box[1])
    if not los:
        return []
    blo = th.stack(los).min(dim=0).values - m_obst
    bhi = th.stack(his).max(dim=0).values + m_obst
    bhi[2] = 10.0  # everything above the floor (SimWorld.task_bounds)
    boxes: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
    keep, cand = set(), []
    for name, info in init.items():
        a = info["args"]
        in_rooms = set(a.get("in_rooms") or [])
        if name not in reg or not (in_rooms & loaded_set or a.get("category") in always_loaded):
            continue  # not loaded at all (partial-room loading) - nothing to drop
        box = _json_aabb(a, reg[name], margin, bool(reg[name].get("joint_pos")))
        if box is not None:
            boxes[name] = box
        if in_rooms & rooms_set or box is None or bool(((box[1] >= blo) & (box[0] <= bhi)).all()):
            keep.add(name)
        else:
            cand.append(name)
    # closure: keep what a kept non-structure object may rest on
    changed = True
    while changed and cand:
        changed = False
        supports = [(boxes[k][0], boxes[k][1]) for k in keep if k in boxes and init[k]["args"].get("category") not in STRUCTURE_CATEGORIES]
        for name in list(cand):
            lo, hi = boxes[name]
            for klo, khi in supports:  # the candidate's top at the kept object's bottom (both boxes carry ``margin``), footprints overlapping
                if bool((hi[:2] >= klo[:2]).all() and (lo[:2] <= khi[:2]).all()) and abs(float(hi[2]) - float(klo[2])) <= 0.3 + 2 * margin:
                    keep.add(name)
                    cand.remove(name)
                    changed = True
                    break
    return sorted(cand)


def _chairs_variant(scene: str, support_name: str, rooms: List[str], out_dir: str, removed: Optional[List[str]] = None, slim: bool = False):
    """Scene variant (``scene_variants.write_variant``) without the chairs around the support and, with ``slim``, without
    the neighbouring rooms' objects that can never be collision obstacles (:func:`_slim_drop_list`): ``(variant dict |
    None, removed chairs)``. Only the chairs' footprints are freed in the traversability map and recorded as removed."""
    from ispatialgym.sampling import scene_variants as SV
    from ispatialgym.sampling.sim_utils import adjacent_rooms, stable_scene_path

    stable = _stable_scene(scene)
    if removed is None:
        removed = _chairs_to_remove(scene, support_name, rooms, stable)
    dropped: List[str] = []
    if slim and os.environ.get("ISG_LOAD_ADJACENT", "1") == "1":
        loaded = list(rooms) + [r for r in adjacent_rooms(scene, rooms) if r not in rooms]
        dropped = [n for n in _slim_drop_list(stable, rooms, loaded) if n not in removed]
        log.info("slim load: %d objects of the neighbouring rooms %s can never pass the obstacle filter and are not loaded", len(dropped),
                 sorted(set(loaded) - set(rooms)))
    if not removed and not dropped:
        return None, []
    vpath, _ = SV.write_variant(stable_scene_path(scene), list(removed) + dropped, out_dir)
    variant = {"path": vpath, "removed_objects": list(removed), "freed_footprints": SV.freed_footprints(stable, removed) if removed else None}
    if dropped:
        variant["slim_dropped"] = dropped
    return variant, list(removed)


def _dataset_of(category: str, model: str, preferred: Optional[str] = None) -> str:
    """Dataset that holds ``category/model``: ``--dataset`` if given, else the first of behavior-1k-assets /
    ispatialgym-assets (the custom block ``toy_dice/isgblk`` lives in the latter) whose object directory exists."""
    from omnigibson.utils.asset_utils import get_dataset_path

    if preferred:
        return preferred
    for ds in ("behavior-1k-assets", "ispatialgym-assets"):
        try:
            if os.path.isdir(os.path.join(get_dataset_path(ds), "objects", category, model)):
                return ds
        except Exception:  # noqa: BLE001 - unknown dataset name
            continue
    return "behavior-1k-assets"


def _rooms_of(scene: str, name: str, stable: Optional[Dict[str, Any]] = None) -> List[str]:
    stable = _stable_scene(scene) if stable is None else stable
    info = stable["objects_info"]["init_info"].get(name)
    if info is None:
        raise SystemExit(f"{name} not in scene {scene}")
    rooms = info["args"].get("in_rooms") or []
    return list(rooms)


def band_rooms(args, stable: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """``(floor_mode, rooms)`` of one band's parsed arguments: the loaded rooms are ``--rooms``, else the floor mode's
    ``--room``, else the rooms of the support (``--under`` partner in the floor-under mode). Sets ``args.room`` for the
    floor-under mode (its lattice's floor object). No simulator."""
    floor_mode = args.support == "floor"
    if floor_mode and not (args.room or args.under or args.rooms):
        raise SystemExit("--support floor needs --room R (or --under PARTNER)")
    anchor_name = args.under if floor_mode and args.under else (None if floor_mode else args.support)
    rooms = args.rooms or ([args.room] if floor_mode and args.room else _rooms_of(args.scene, anchor_name, stable))
    if floor_mode and args.under and not args.room:
        args.room = rooms[0]
    return floor_mode, list(rooms)


def band_arms(args) -> Tuple[str, ...]:
    """Arms whose embodiments the band uses (both for ``--brace``)."""
    if args.generic_contact and args.brace:
        raise SystemExit("--generic-contact and --brace are exclusive")
    return ("right", "left") if args.brace else (args.arm,)


def band_world_key(args, stable: Optional[Dict[str, Any]] = None) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """The **world key** of a band job - ``(scene, loaded rooms, removed chairs)`` - what decides whether two jobs can
    share one loaded world in the ``--jobs`` worker (the sampler's ``sample_isg.world_key`` for aliases). Needs only the
    stable scene json (``stable``; loaded from the dataset when omitted)."""
    stable = _stable_scene(args.scene) if stable is None else stable
    floor_mode, rooms = band_rooms(args, stable)
    removed = [] if floor_mode else _chairs_to_remove(args.scene, args.support, rooms, stable)
    return args.scene, tuple(rooms), tuple(removed)


def select_jobs(manifest: Sequence[Dict[str, Any]], ids: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """Jobs of a ``make_jobs.py`` manifest by id (``ids=None`` = every job, manifest order). Unknown ids raise ``KeyError``."""
    if ids is None:
        return list(manifest)
    by_id = {j["id"]: j for j in manifest}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"job ids not in the manifest: {missing}")
    return [by_id[i] for i in ids]


def group_by_key(items: Sequence[Any], key_of) -> List[Any]:
    """``items`` reordered so that equal keys are contiguous: groups in the order of their first item, items in their
    original order within a group (what makes world reloads rare in the ``--jobs`` worker while keeping the given order
    where it does not matter)."""
    groups: Dict[Any, List[Any]] = {}
    for it in items:
        groups.setdefault(key_of(it), []).append(it)
    return [it for g in groups.values() for it in g]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="scene model (required unless --jobs)")
    ap.add_argument("--support", default=None, help="scene object name, or 'floor' (with --room); required unless --jobs")
    ap.add_argument("--room", default=None, help="floor mode: room instance")
    ap.add_argument("--rooms", nargs="*", default=None, help="rooms to load (default: the support's rooms)")
    ap.add_argument("--object", default=None, help="CATEGORY/MODEL of the grasped object")
    ap.add_argument("--grasp-key", default=None)
    ap.add_argument("--generic-pinch", action="store_true", help="no library grasps: generic top-down pinch / push classes (blocks, axial targets)")
    ap.add_argument("--arm", default="right", choices=["right", "left"], help="arm of the band (default right; the brace arm's contact bands use --arm left)")
    ap.add_argument("--generic-contact", action="store_true", help="generic CONTACT set (closed gripper, side faces at 3/4 height + top press) for --arm left; one band per object height class")
    ap.add_argument("--height", type=float, default=None, help="object height (m) of the contact band (else measured from --object); binned to the 2 cm height class")
    ap.add_argument("--placed", nargs="*", default=[], help="objects placed before the target, CATEGORY/MODEL[*SCALE][@MODE] (disembedding source tray; "
                    "MODE 'edge' (default, legacy: fixed tray at the stance line, lattice = its floor) or 'focus_centred_edge_aligned' (stance-relative: "
                    "the tray follows the cell, lattice = the top cells where it fits), e.g. tray/gsxbym*1.25@focus_centred_edge_aligned")
    ap.add_argument("--under", default=None, help="floor mode: partner scene object; lattice = floor under it")
    ap.add_argument("--target", default=None, help="brace mode: articulated CATEGORY/MODEL")
    ap.add_argument("--handle-key", default=None)
    ap.add_argument("--joint", default=None)
    ap.add_argument("--joint-values", default=None, help="comma-separated joint values whose handle poses are screened")
    ap.add_argument("--brace", action="store_true")
    ap.add_argument("--edge", default=None, help="edge name (+u/-u/+v/-v); default: longest collision-free edge")
    ap.add_argument("--stand-off", type=float, default=STAND_OFF)
    ap.add_argument("--lattice", type=float, default=None, help=f"cell size (default {LATTICE} m; --smoke: 0.16 m)")
    ap.add_argument("--scale", default=None, help="spawn scale of the grasped object, 'sx,sy,sz' (icl scale column); part of the cache key")
    ap.add_argument("--dataset", default=None, help="dataset of the grasped object (default: auto, behavior-1k-assets or ispatialgym-assets)")
    ap.add_argument("--trunks", nargs="*", default=None, help="restrict the trunk grid: 'translate,tilt' items, e.g. 0.5,0 0.95,-0.9")
    ap.add_argument("--out", default=None, help="directory for the .npz/.json (default: the cache dir)")
    ap.add_argument("--smoke", action="store_true", help="16 cm lattice, two trunks, shoulder-FK verification + band sanity summary")
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    # -- multi-job worker (one process, many bands; see run_jobs)
    ap.add_argument("--jobs", default=None, help="worker mode: a make_jobs.py manifest; every selected job's `args` is one band, written to <--out>/<id>/ "
                    "(+ <--out>/done.txt), the world (scene, rooms, removed chairs) is reused across jobs with the same key")
    ap.add_argument("--ids", nargs="*", default=None, help="worker mode: job ids of the manifest to run (default: every job), grouped by world key")
    ap.add_argument("--prebuild-planner", action="store_true", help="build the arms' cuRobo motion planners at start (the pre-2026-09-16 behaviour; bands "
                    "only use the kinematics and the batched IK solvers, which are independent of the planners, so this costs ~25 s per process for nothing)")
    ap.add_argument("--cameras", action="store_true", help="give the robot the sampler's 1080^2 rgb + seg_instance head camera (the pre-2026-09-16 behaviour); "
                    "bands never read pixels, so the default world is camera-less (SimWorld(cameras=False): faster launch, less GPU memory, same bands)")
    ap.add_argument("--full-load", action="store_true", help="load every object of the task + neighbouring rooms (the pre-2026-09-16 behaviour). Default: "
                    "the neighbouring rooms' (and the always-loaded structures') objects that can never pass SimWorld.obstacle_filter are left out - their "
                    "stable-scene bounding box, grown by SLIM_MARGIN_M (+ extent when articulated), misses the task rooms' floor extent + ISG_OBSTACLE_MARGIN_M; "
                    "the collision world (obstacle set, cuRobo mesh names, bands) is identical, the scene load 117 s -> 54 s, 421 -> 108 objects (house_single_floor)")
    ap.add_argument("--mesh-cache", default=None, help="SAME-ALGORITHM OPT-IN (default: cuRobo's 8192, unchanged): capacity of the cuRobo mesh store, a number "
                    "or 'auto' (= the loaded collision meshes + %d, at least %d). cuRobo launches spheres x capacity collision threads per evaluation, so "
                    "'auto' made every batched IK call ~2.5x faster (trunk 65 s -> 25 s), but the thread layout changes the order of its atomic float "
                    "accumulation and a marginal cell can differ from the 8192 result (1 pair of 88 on the contact test band); recorded as schedule.mesh_cache" % (MESH_CACHE_HEADROOM, MESH_CACHE_MIN))
    ap.add_argument("--ik-tail-sizes", default=None, help="SAME-ALGORITHM OPT-IN (default: the committed 16,32,64,128): batched IK solver sizes for the last "
                    "chunk of a screening call, e.g. 8,16,24,32,48,64,96,128 - fewer padded slots (73 real poses per 128-slot call in the pilot), but every "
                    "solver size draws its own cuRobo seed stream, so marginal cells can differ from the default schedule; recorded as schedule.ik_sizes")
    return ap


def _reset_kit_seeds(kit):
    """Put every cuRobo solver's random stream back where a fresh process starts it: the 11 extra IK seeds per pose come
    from a Halton buffer indexed by a ``torch.Generator`` seeded once per solver instance and advanced by every call, so a
    band computed after another one in the same process would draw different seeds (and marginal cells could flip)
    without this. ``IKSolver.reset_seed`` restores the generator state of the LM seed solver and the optimizer's."""
    be = kit.backend
    for e in (getattr(be, "_batch_ik", None) or {}).values():
        e.rsc.reset_seed()
    for e in be._entries.values():  # the planners' IK solvers (not used by bands; reset for completeness)
        if e.planner is not None and getattr(e.planner, "ik_solver", None) is not None:
            e.planner.ik_solver.reset_seed()


def load_world(args, rooms: List[str], variant: Optional[dict], removed: List[str], arms: Sequence[str]):
    """``(world, kit)`` of a band job: ``SimWorld`` + ``RobotKit`` + the first cuRobo world sync. The arms' motion planners
    are not built unless ``--prebuild-planner``: bands use FK / collision kinematics entries and the batched IK solvers
    only (measured 24.5-24.9 s per process for the right-arm planner build + warmup)."""
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import SimWorld

    import omnigibson as og

    t_load = time.time()
    world = SimWorld(args.scene, rooms, scene_variant=variant, cameras=bool(args.cameras))
    # cuRobo's mesh store capacity sizes every collision kernel launch (see R1ProCuRobo.__init__). SAME-ALGORITHM OPT-IN: a
    # snug capacity (``--mesh-cache auto`` = loaded meshes + MESH_CACHE_HEADROOM, or a number) made the batched IK 2.5x
    # faster, but the kernel's thread layout changes with the capacity and with it the order of cuRobo's atomic float
    # accumulation over the obstacles that touch one sphere: measured 2026-09-16, two of three two-trunk bands
    # byte-identical, the contact band gained one marginal (cell, class) pair. Default: the backend's 8192 (unchanged).
    n_meshes = RobotKit.count_collision_meshes(world)
    cache = None
    if args.mesh_cache:
        cache = max(MESH_CACHE_MIN, 64 * math.ceil((n_meshes + MESH_CACHE_HEADROOM) / 64)) if str(args.mesh_cache).lower() == "auto" else int(args.mesh_cache)
    kit = RobotKit(world, prebuild=tuple(f"{a}_arm" for a in arms) if args.prebuild_planner else (), mesh_cache_size=cache)
    kit.sync_world()
    log.info("cuRobo mesh cache %d for %d loaded collision meshes (%d after the sync)%s", kit.backend.mesh_cache_size, n_meshes, len(kit.backend._world_mesh_names),
             "" if cache is None else " [--mesh-cache: same-algorithm opt-in, marginal cells can differ from the 8192 default]")
    #: physics state a band starts from in a fresh process (robot parked, five physics steps, nothing spawned); restored
    #: before every further band of a worker (:func:`reset_world_between_bands`)
    world.parked_state = og.sim.dump_state()
    log.info("scene %s rooms %s loaded in %.0fs (chairs removed: %s)", args.scene, rooms, time.time() - t_load, removed)
    dump = os.environ.get("ISPATIALGYM_RB_DUMP_WORLD")
    if dump:  # diagnostic: the loaded objects, the obstacle-filter verdicts and the cuRobo mesh names (to diff a --slim-load world against a full one)
        rec = {"n_objects": len(world.scene.objects), "loaded_rooms": world.loaded_rooms, "n_curobo_meshes": len(kit.backend._world_mesh_names),
               "curobo_meshes": sorted(kit.backend._world_mesh_names), "obstacles": sorted(o.name for o in world.scene.objects if o is not world.robot and world.obstacle_filter(o)),
               "objects": {o.name: {"category": o.category, "in_rooms": list(o.in_rooms or []), "aabb": [[float(v) for v in b] for b in o.aabb],
                                    "obstacle": bool(world.obstacle_filter(o))} for o in world.scene.objects if o is not world.robot}}
        with open(dump, "w") as f:
            json.dump(rec, f, indent=1)
        log.info("world dump written to %s (%d objects, %d obstacles, %d cuRobo meshes)", dump, rec["n_objects"], len(rec["obstacles"]), rec["n_curobo_meshes"])
    return world, kit


def reset_world_between_bands(world, kit):
    """Isolation of consecutive bands in one process (``--jobs``), so that band k sees exactly what a fresh process shows
    a band: the previous band's spawned objects are removed; the render-side transforms are put back to the state after
    the environment's load (``SimWorld.initial_state`` + one PhysX -> Fabric sync: a fresh process's ``aabb`` / raster /
    first cuRobo sync read those, since only physics steps follow the load) and the physics side to the parked state
    (``parked_state``, loaded without a step: the fresh process's five park steps, nothing more - the objects a band
    spawns re-create PhysX from that state exactly as in a fresh process); the cuRobo world is re-synced (re-enables what
    an ``ObstacleScope`` / ``MovingPlaced`` of a failed band may have left disabled, disables the removed objects'
    meshes, re-poses every mesh from the restored transforms) and every solver's seed stream reset
    (:func:`_reset_kit_seeds`). Measured 2026-09-16 on three two-trunk bands: byte-identical to their one-band runs."""
    import omnigibson as og

    for o in list(world.spawned.values()):
        world.remove(o)
    og.sim.load_state(world.initial_state, serialized=False)
    world.sync_render_transforms()  # render side = the state after the load, as in a fresh process
    # the physics side alone goes to the parked state: RigidDynamicPrim.set_position_orientation (what load_state calls
    # per body while playing) writes the PhysX view and then syncs *every* body's PhysX pose to Fabric, which would move the
    # render side to the parked poses too (a fresh process never renders between the load and its band; measured
    # 2026-09-16: the support's AABB then differed by 2.4e-6 m and one pose of 1968 was pruned differently)
    og.sim.sync_physx_to_fabric = lambda: None  # shadows the bound method on the simulator instance
    try:
        og.sim.load_state(world.parked_state, serialized=False)
    finally:
        del og.sim.sync_physx_to_fabric
    kit.sync_world()
    _reset_kit_seeds(kit)


def compute_band(args, world, kit, out_dir: str, *, rooms: List[str], removed: List[str], floor_mode: bool, arms: Sequence[str], t_load: float,
                 setup_s: float, job_id: Optional[str] = None, spawn_suffix: str = "") -> Dict[str, Any]:
    """One band on an existing ``(world, kit)``: support geometry and stance, placed objects, target spawn, the band
    computation, the ``.npz`` / ``.json`` files in ``out_dir`` and the ``REACH_BANDS {...}`` stdout line. Returns the
    summary dict; raises on failure (the callers print ``REACH_BANDS_FAILED``). ``t_load`` = when the band's setup began
    (``time_s`` counts from there), ``setup_s`` = world load / reuse time (meta ``setup_s``), ``spawn_suffix`` = suffix of
    the spawned objects' names (the worker gives every band its own so that cuRobo, which caches mesh geometry by prim
    path, never reuses a previous band's geometry for a re-spawned model at another scale)."""
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling import grasp_library as GL

    lattice = args.lattice if args.lattice is not None else (0.16 if args.smoke else LATTICE)
    scale = parse_scale(args.scale)
    trunk_sel = [tuple(float(v) for v in t.split(",")) for t in args.trunks] if args.trunks else None
    band_arm = args.arm
    stats0 = kit.stats_snapshot()
    bc = BandComputer(world, kit, arms=arms, lattice=lattice)
    if args.generic_contact:
        bc.fingers = 0.0  # closed gripper
    meta: Dict[str, Any] = {"scene": args.scene, "rooms": rooms, "support": args.support, "removed_chairs": removed, "lattice": lattice,
                            "stand_off": args.stand_off, "arms": "untucked (ready / tucked per trunk when untucked collides, see trunks[].posture)",
                            "postures": list(POSTURES), "arm": band_arm, "fingers": bc.fingers,
                            "grid": {"translates": TRANSLATES, "tilts": TILTS, "low_top_tilt": LOW_TOP_TILT},
                            "seeds": 12, "pre_grasp": PRE_GRASP, "yaws": YAWS, "mode": "brace" if args.brace else ("floor" if floor_mode else "tabletop"),
                            "smoke": bool(args.smoke), "scale": scale, "placed_specs": [PlacedSpec.parse(s).canonical() for s in args.placed],
                            "schedule": {"chunk": BandComputer.CHUNK, "ik_batch": BandComputer.IK_BATCH, "max_rounds_without_pre": BandComputer.MAX_ROUNDS_WITHOUT_PRE, "coarse": BandComputer.COARSE,
                                         "pose_prune": "eef and pre-grasp origin within arm_len + 0.03 of the shoulder", "obstacle_radius_m": ObstacleScope.RADIUS,
                                         "obstacle_z_above_m": ObstacleScope.Z_ABOVE}}
    if trunk_sel:
        meta["trunk_selection"] = trunk_sel
    if job_id is not None:
        meta["job_id"] = job_id
    if args.ik_tail_sizes:
        sizes = tuple(sorted({int(v) for v in str(args.ik_tail_sizes).split(",") if v.strip()}))
        if not sizes or sizes[-1] != BandComputer.IK_BATCH or sizes[0] < 1:
            raise SystemExit(f"--ik-tail-sizes must be positive sizes ending with the batch size {BandComputer.IK_BATCH}, got {args.ik_tail_sizes!r}")
        if sizes != tuple(BandComputer.IK_SIZES):
            bc.IK_SIZES = sizes  # instance override; the class default stays the exact schedule
            meta["schedule"]["ik_sizes"] = list(sizes)
    if args.mesh_cache:  # same-algorithm opt-in (see load_world): record the capacity the bands were computed with
        meta["schedule"]["mesh_cache"] = int(kit.backend.mesh_cache_size)
    if args.smoke:
        t_fk = time.time()
        meta["shoulder_fk_check"] = verify_shoulder_fk(world, kit, bc, SMOKE_FK_TRUNKS, arm="right")
        log.info("shoulder FK check done in %.1fs", time.time() - t_fk)
        kit.sync_world()
    # ---- support geometry and stance
    placed_objs = []
    placed_specs = [PlacedSpec.parse(s) for s in args.placed]
    if floor_mode:
        z_support = world.floor_z
        if args.under:
            partner = world.obj(args.under)
            stance = bc.stance_under(partner, args.stand_off)
            lo, hi = partner.aabb
            res = 0.02
            xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
            ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
            gx, gy = th.meshgrid(xs, ys, indexing="ij")
            pts = th.stack([gx.flatten(), gy.flatten()], dim=1)
            starts = th.cat([pts, th.full((len(pts), 1), world.floor_z + 0.005)], dim=1)
            ends = starts.clone()
            ends[:, 2] = float(hi[2]) + 0.5
            hits = world.raycast(starts, ends)
            prims = {l.prim_path for l in partner.links.values()}
            clear = th.tensor([bool(h["hit"] and h["rigidBody"] in prims) for h in hits])
            clearance = th.tensor([float(h["position"][2]) - world.floor_z if h["hit"] else 0.0 for h in hits])
            lattice_world = pts[clear]
            meta["under_clearance_m"] = {"min": float(clearance[clear].min()) if int(clear.sum()) else None, "max": float(clearance[clear].max()) if int(clear.sum()) else None}
            meta["under_partner"] = args.under
            ignore_grasp = [world.obj(world.floor_of_room(rooms[0]).name)] if rooms else []
        else:
            stance = bc.stance_free_floor(args.room)
            origin, fwd, left = bc.base_frame(stance)
            xs = th.arange(0.2, STRIP_MAX_DEPTH + 0.5, 0.02)
            ls = th.arange(-STRIP_HALF_WIDTH, STRIP_HALF_WIDTH + 1e-6, 0.02)
            gx, gl = th.meshgrid(xs, ls, indexing="ij")
            lattice_world = origin + gx.flatten().unsqueeze(1) * fwd + gl.flatten().unsqueeze(1) * left
            lattice_world = th.stack([c for c in lattice_world if kit.trav_free(c)]) if len(lattice_world) else lattice_world
            ignore_grasp = [world.floor_of_room(args.room)] if args.room else []
        top_json = {"z": z_support, "kind": "floor"}
    else:
        support = world.obj(args.support)
        top = world.support_top(support)
        z_support = top.z
        stance = bc.stance_at_edge(top, support, args.stand_off, args.edge)
        lattice_world = top.cells(free_only=True)
        ignore_grasp = [support]
        top_json = top.to_json()
        # objects placed before the target (disembedding source tray), long axis along the edge. ``edge`` mode (legacy):
        # centred on the stance line, near edge PLACED_EDGE_SETBACK behind the front edge, the lattice = its inner
        # floor. ``focus_centred_edge_aligned`` (owner item 9, stance-relative): the tray is measured at that pose,
        # then follows the cell - inner-floor centroid on the cell -, the lattice = the top cells where it fits and
        # its collision meshes are checked per cell (MovingPlaced); the grasped object stands on its floor at the cell.
        moving_placed = []
        for ps in placed_specs:
            cat, model = ps.category, ps.model
            obj = world.spawn(f"placed_{cat}_{model}{spawn_suffix}", cat, model, dataset=_dataset_of(cat, model, None), scale=ps.scale_list())
            hull = world.local_hull(obj)
            ext = hull.max(dim=0).values - hull.min(dim=0).values
            tangent = th.tensor(stance["tangent"])
            inward = th.tensor(stance["inward"])
            tangent_yaw = math.atan2(float(tangent[1]), float(tangent[0]))
            yaw = tangent_yaw if float(ext[0]) >= float(ext[1]) else tangent_yaw - math.pi / 2
            quat = G.quat_from_yaw(yaw)
            fp0 = world.footprint(hull, (th.zeros(3), quat))
            half_depth = float((fp0 @ inward).abs().max())
            mid = th.tensor(stance["edge_mid"])
            xy = mid + inward * (PLACED_EDGE_SETBACK + half_depth)
            pos = th.tensor([float(xy[0]), float(xy[1]), top.z + world.bottom_offset(hull, quat) + 0.003])
            obj.set_position_orientation(pos, quat)
            world.activate(obj)
            world.step_physics(30, still=[world.robot])
            p0, q0 = obj.get_position_orientation()
            pose0 = (p0.float(), q0.float())
            placed_objs.append((obj, pose0))
            # inner floor raster of the placed object (lowest inner level hit by downward rays)
            lo, hi = obj.aabb
            res = 0.01
            xs = th.arange(float(lo[0]) + res / 2, float(hi[0]), res)
            ys = th.arange(float(lo[1]) + res / 2, float(hi[1]), res)
            gx, gy = th.meshgrid(xs, ys, indexing="ij")
            starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), float(hi[2]) + 0.10)], dim=1)
            ends = starts.clone()
            ends[:, 2] = float(lo[2]) - 0.02
            hits = world.raycast(starts, ends)
            prims = {l.prim_path for l in obj.links.values()}
            zs = th.tensor([float(h["position"][2]) if (h["hit"] and h["rigidBody"] in prims) else float("nan") for h in hits])
            valid = zs[~th.isnan(zs)]
            floor_z = float(th.quantile(valid, 0.10))
            keep = (~th.isnan(zs)) & ((zs - floor_z).abs() < 0.004)
            floor_cells = starts[keep][:, :2]
            z_support = floor_z
            c_w = floor_cells.mean(dim=0)
            c_local = G.relative((th.tensor([float(c_w[0]), float(c_w[1]), floor_z]), G.IDENTITY_QUAT), pose0)[0]
            rec = dict(ps.to_json(), pose=[pos.tolist(), quat.tolist()], pose_settled=[pose0[0].tolist(), pose0[1].tolist()], floor_z=floor_z,
                       n_floor_cells=int(keep.sum()), floor_centre_local=[round(float(v), 4) for v in c_local.tolist()], yaw=yaw,
                       hull_extent=[round(float(v), 4) for v in ext.tolist()],
                       yaw_rule="long footprint axis along the edge tangent: yaw = tangent_yaw if extent_x >= extent_y else tangent_yaw - pi/2")
            if ps.mode == PLACED_MODE_FOCUS:
                off = (c_w - pose0[0][:2]).clone()  # root -> inner-floor centroid, world xy at this yaw
                z0 = float(pose0[0][2])

                def pose_at(world_xy, _off=off, _z=z0, _q=pose0[1]):
                    return th.tensor([float(world_xy[0]) - float(_off[0]), float(world_xy[1]) - float(_off[1]), _z]), _q

                # the lattice is the whole near strip of the top (as for the plain bands): the band is relative to the base, the
                # sampler shifts it with its own stand-off and checks the tray's fit on the top / in the frame itself (a cell
                # near the front edge is the stance-relative geometry of a closer stance, so it is not filtered here)
                _, world_c = bc.lattice_on_cells(stance, top.cells(free_only=True))
                lattice_world = world_c
                fit = th.tensor([top.polygon_supported(fp0 + pose_at(c)[0][:2], margin=PLACED_FIT_MARGIN) for c in world_c], dtype=th.bool) if len(world_c) else th.zeros(0, dtype=th.bool)
                moving_placed.append((obj, pose0, pose_at))
                rec.update({"lattice": "near strip of the top (the tray follows every cell; its fit on the top is the sampler's check at its stance)",
                            "n_lattice_top": int(len(world_c)), "n_lattice_fit_at_band_stance": int(fit.sum()), "fit_margin": PLACED_FIT_MARGIN,
                            "pose_rule": "root xy = cell - (floor centroid - root) at the edge-aligned yaw, z = top + bottom_offset(hull, quat) + 0.003; "
                                         "the grasped object stands on the floor (z_support) at the cell; collision meshes checked per cell for the pre-grasp, "
                                         "ignored for the grasp pose"})
                log.info("placed %s (%s): floor z %.3f, %d floor cells, centroid offset from the root %s; %d top lattice cells (%d admit the tray at the band stance)",
                         ps.canonical(), ps.mode, floor_z, int(keep.sum()), [round(float(v), 3) for v in off.tolist()], len(world_c), int(fit.sum()))
            else:
                lattice_world = floor_cells
                ignore_grasp.append(obj)
            meta.setdefault("placed", []).append(rec)
        kit.sync_world()
        if len(moving_placed) > 1:
            raise SystemExit("at most one --placed object in the focus_centred_edge_aligned mode")
        for obj, pose0, pose_at in moving_placed:
            bc.cell_obstacle = MovingPlaced(kit, world, obj, pose0, pose_at)
            if not bc.cell_obstacle.names:
                raise RuntimeError(f"placed object {obj.name}: no collision meshes in the cuRobo world")
            if args.smoke:  # moved-mesh geometry against the simulator at a cell of the reachable strip (0.56 m ahead, 8 cm right)
                origin, fwd, left = bc.base_frame(stance)
                meta["placed"][-1]["self_check"] = bc.cell_obstacle.self_check(origin + 0.56 * fwd - 0.08 * left)
    meta.update({"stance": stance, "top": top_json, "z_support": z_support})

    def restore_placed():
        """``world.spawn`` (of the grasped object) re-parks every object spawned before it: put the placed objects back
        at their measured poses so that they are obstacles again (the ``edge`` mode's fixed tray; the focus mode moves
        its collision meshes itself but keeps the body where the meshes were measured)."""
        for po, p0 in placed_objs:
            po.set_position_orientation(p0[0], p0[1])
            world.activate(po)
        if placed_objs:
            world.step_physics(5, still=[world.robot])

    # ---- object / grasp set
    if args.brace:
        assert args.target and args.handle_key and args.joint and args.joint_values, "--brace needs --target --handle-key --joint --joint-values"
        cat, model = args.target.split("/", 1)
        from ispatialgym.sampling.grasp_library import library_root

        with open(os.path.join(library_root(), "artic", "handles", f"{args.handle_key}.json")) as f:
            hdata = json.load(f)
        jinfo = next(j for j in hdata["joints"] if j["joint"] == args.joint)
        handles = [h for h in jinfo.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([jinfo["fallback"]] if jinfo.get("fallback") else [])
        if not handles:
            raise SystemExit(f"no graspable handle for {args.handle_key}/{args.joint}")
        hdata["handle"] = handles[0]
        q_values = [float(v) for v in args.joint_values.split(",")]
        obj = world.spawn(f"target_{cat}_{model}{spawn_suffix}", cat, model, dataset=_dataset_of(cat, model, args.dataset), scale=scale)
        hull = world.local_hull(obj)
        world.park(obj, len(world.spawned) - 1)
        restore_placed()
        kit.sync_world()
        # facing direction: moving face outward normal (lids: from the hinge line to the centre) points at the robot
        outward = th.tensor(jinfo.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
        if abs(float(outward[2])) > 0.9:
            axis = th.tensor(jinfo.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32)
            hinge = th.tensor(jinfo.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            perp = th.cross(axis, th.tensor([0.0, 0.0, 1.0]), dim=0)
            centre = hull.mean(dim=0)
            if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
                perp = -perp
            outward = perp
        outward[2] = 0.0
        outward = outward / th.linalg.norm(outward)
        inward = th.tensor(stance["inward"])
        face_yaw = math.atan2(float(-inward[1]), float(-inward[0])) - math.atan2(float(outward[1]), float(outward[0]))
        # handle arm = the arm on the handle's side of the target centre (> 5 cm), else right; brace = the other
        hc = th.tensor(hdata["handle"]["contact"], dtype=th.float32)
        side_axis = th.tensor([-float(outward[1]), float(outward[0]), 0.0])  # robot's left when the face points at the robot
        lat = float(th.dot(hc - hull.mean(dim=0), side_axis))
        handle_arm = "left" if lat > 0.05 else "right"
        brace_arm = "right" if handle_arm == "left" else "left"
        brace_side = 1.0 if brace_arm == "left" else -1.0
        x0, y0, yaw0 = stance["base_pose2d"]
        face_off = G.wrap_angle(face_yaw - yaw0)  # object yaw offset (relative to the base yaw) with the moving face toward the robot
        meta.update({"target": args.target, "handle_key": args.handle_key, "joint": args.joint, "joint_values": q_values, "face_yaw_offset": face_off,
                     "handle_arm": handle_arm, "brace_arm": brace_arm, "handle": hdata["handle"]})
        base_cells, world_cells = bc.lattice_on_cells(stance, lattice_world)
        trunks = trunk_sel or (SMOKE_TRUNKS if args.smoke else trunk_grid(z_support - world.floor_z))
        per_trunk = []
        cells_out: Dict[Tuple[float, float], np.ndarray] = {}
        bc.scope = ObstacleScope(kit, world, (x0, y0), drop_ground=(z_support - world.floor_z) >= 0.55).__enter__()
        for tr, ti in trunks:
            t0 = time.time()
            entry = {"translate": tr, "tilt": ti, "collides": False, "n_lattice": int(len(base_cells)), "n_reachable": 0, "time_s": 0.0, "feasible": [], "posture": None}
            q, posture = bc.trunk_config(tr, ti, x0, y0, yaw0)
            if q is None:
                entry["collides"] = True
                per_trunk.append(entry)
                cells_out[(tr, ti)] = np.zeros((0, 2))
                log.info("brace trunk (%.2f, %+.1f): collides with every idle-arm posture, skipped", tr, ti)
                continue
            entry["posture"] = posture
            entry["idle_arm_q"] = bc.idle_arm_q(q)
            # 3 yaws about the facing yaw (offsets relative to the base yaw). Per (cell, yaw) the handle [pre-grasp, grasp]
            # is screened at every listed joint value (spec 5.4: the output is the SET of joint values whose handle poses
            # are feasible; the sampler requires its direction's waypoints to be a subset), then the brace [pre, contact]
            # candidates with the brace arm for the (cell, yaw) pairs with >= 1 feasible joint value. A cell is in the band
            # when some yaw has >= 1 feasible joint value and >= 1 feasible brace kind; the yaw with the most feasible
            # joint values (ties: the facing yaw first) is recorded. Batched per trunk: every handle pre-grasp of every
            # (cell, yaw, q) in one IK call, the grasps of the pre-feasible ones in a second (support ignored), the brace
            # pairs in a third.
            shoulder, arm_len = bc.shoulder_and_reach(q, handle_arm)
            pts = th.cat([world_cells, th.full((len(world_cells), 1), z_support)], dim=1)
            near = th.linalg.norm(pts - shoulder, dim=1) <= arm_len + float(th.linalg.norm(th.tensor(hdata["handle"]["contact"]))) + PRE_GRASP + 0.05
            yaw_offsets = (face_off, face_off - 0.1, face_off + 0.1)
            combos = [(i, dy) for i in range(len(base_cells)) if bool(near[i]) for dy in yaw_offsets]
            obj_poses = {c: bc.object_pose_at(world_cells[c[0]], yaw0 + c[1], z_support, hull, G.IDENTITY_QUAT) for c in combos}
            handle_pairs = {c: [handle_pose_at_q(jinfo, hdata["handle"], qv, obj_poses[c], G) for qv in q_values] for c in combos}
            flat = [(c, k) for c in combos for k in range(len(q_values))]
            ok_pre = bc.ik_screen([handle_pairs[c][k][0] for c, k in flat], q, handle_arm)
            pre_ok = [(c, k) for n, (c, k) in enumerate(flat) if bool(ok_pre[n])]
            with bc.ignoring(ignore_grasp):
                ok_g = bc.ik_screen([handle_pairs[c][k][1] for c, k in pre_ok], q, handle_arm)
            handle_ok: Dict[Tuple[int, float], List[int]] = {}
            for n, (c, k) in enumerate(pre_ok):
                if bool(ok_g[n]):
                    handle_ok.setdefault(c, []).append(k)
            surv = [c for c in combos if c in handle_ok]
            braces = {c: brace_candidates(jinfo, hdata, obj_poses[c], q_values, brace_side, G) for c in surv}
            bflat = [(c, kind) for c in surv for kind, _ in braces[c]]
            bposes = [pose for c in surv for _, pair in braces[c] for pose in pair]
            brace_prefix = {c: (braces[c][0].q_prefix if braces[c] else 0) for c in surv}
            ok_b = bc.ik_screen(bposes, q, brace_arm)
            brace_ok: Dict[Tuple[int, float], List[str]] = {}
            for n, (c, kind) in enumerate(bflat):
                if bool(ok_b[2 * n]) and bool(ok_b[2 * n + 1]):
                    brace_ok.setdefault(c, []).append(kind)
            feasible_cells = []
            n_handle_only = 0
            for i in range(len(base_cells)):
                best = None
                for dy in yaw_offsets:
                    c = (i, dy)
                    if c not in handle_ok:
                        continue
                    if c not in brace_ok:
                        n_handle_only += 1
                        continue
                    if best is None or len(handle_ok[c]) > len(handle_ok[best]):
                        best = c
                if best is not None:
                    feasible_cells.append((i, best[1], sorted(handle_ok[best]), brace_ok[best], brace_prefix[best]))
            entry["n_after_prune"] = int(near.sum())
            entry["n_reachable"] = len(feasible_cells)
            entry["n_handle_only_pairs"] = n_handle_only
            entry["n_all_q"] = sum(1 for _, _, ks, _, _ in feasible_cells if len(ks) == len(q_values))
            entry["feasible"] = [{"cell": base_cells[i].tolist(), "yaw_offset": yaw, "q_indices": ks, "q_feasible": [q_values[k] for k in ks], "brace_kinds": kinds,
                                  "brace_q_prefix": kp}
                                 for i, yaw, ks, kinds, kp in feasible_cells]
            entry["time_s"] = round(time.time() - t0, 2)
            per_trunk.append(entry)
            cells_out[(tr, ti)] = base_cells[[i for i, _, _, _, _ in feasible_cells]].numpy() if feasible_cells else np.zeros((0, 2))
            log.info("brace trunk (%.2f, %+.1f): %d/%d cells feasible (%d with every joint value; %d (cell, yaw) handle-only), %.1fs", tr, ti,
                     len(feasible_cells), len(base_cells), entry["n_all_q"], n_handle_only, entry["time_s"])
        bc.scope.__exit__(None, None, None)
        bc.scope = None
        result = {"trunks": per_trunk, "cells": cells_out, "pairs": {}}
        key = _key({"scene": args.scene, "support": args.support, "z": round(z_support, 2), "stance": stance["edge"], "s": args.stand_off, "target": args.target,
                    "scale": scale, "handle_key": args.handle_key, "joint": args.joint, "q": q_values, "brace": True, "grid": meta["grid"], "lattice": lattice, "arms": "untucked"})
        name = f"brace__{args.scene}__{args.support}__{cat}-{model}__{key}"
    else:
        grasps, rest_quat, max_off = None, G.IDENTITY_QUAT.clone(), 0.0
        hull = None
        if args.object:
            cat, model = args.object.split("/", 1)
            obj = world.spawn(f"target_{cat}_{model}{spawn_suffix}", cat, model, dataset=_dataset_of(cat, model, args.dataset), scale=scale)
            hull = world.local_hull(obj)
            world.park(obj, len(world.spawned) - 1)
            ext = (hull.max(dim=0).values - hull.min(dim=0).values).tolist() if hull is not None and len(hull) else None
            meta["object_extent"] = [round(float(v), 4) for v in ext] if ext else None
            if not args.generic_pinch:
                key = GL.resolve_key(cat, model, args.grasp_key, ["tabletop", "insert", "floating"])
                grasps = GL.load_grasps(key)
                rq = GL.rest_quat_xyzw(key)
                rest_quat = rq if rq is not None else rest_quat
                max_off = max(float(th.linalg.norm(g.position)) for g in grasps) if grasps else 0.0
                meta.update({"object": args.object, "grasp_key": key, "n_grasps": len(grasps), "grasp_names": [g.name for g in grasps],
                             "rest_quat_xyzw": rest_quat.tolist(), "generic_pinch": False})
            else:
                meta.update({"object": args.object, "grasp_key": None, "generic_pinch": True, "grasp_names": []})
            h = float(hull[:, 2].max() - hull[:, 2].min()) if hull is not None and len(hull) else 0.05
        else:
            meta.update({"object": None, "grasp_key": None, "generic_pinch": True, "grasp_names": []})
            h = 0.05
        if args.generic_contact:
            # left-arm (brace) contact band: one per object height class; the cell is the contact point on the object's
            # side face at 3/4 of the class height (top press at the class height); z_support = the top the object stands on
            h_obj = float(args.height) if args.height is not None else h
            hcls = height_class(h_obj)
            bc.contact_height = hcls
            pose_fn, grasps, max_off = None, None, 0.0
            meta.update({"object": None if args.height is not None else args.object, "grasp_key": None, "generic_pinch": False, "grasp_names": [],
                         "contact_set": CONTACT_SET, "height_class": hcls, "height_measured": round(h_obj, 4), "classes": CONTACT_CLASSES_NAMES,
                         "contact_spec": {"classes": CONTACT_CLASSES, "height_fraction": CONTACT_HEIGHT_FRACTION, "height_bin": HEIGHT_BIN,
                                          "eef_origin": "point - CONTACT_CENTER_Z * a", "pre_contact": PRE_GRASP, "fingers": bc.fingers,
                                          "support_ignored_for": "contact pose only"}})
            z_eval = z_support
        elif args.generic_pinch or grasps is None:
            pose_fn = None
            meta["generic_pinch_spec"] = {"n_yaws": N_YAWS_GENERIC, "tilts_deg": list(TILTS_GENERIC), "z_above_support": round(max(0.6 * h, 0.03), 4),
                                         "order": "tilt-major, yaw-minor (RobotKit.grasp_candidates_at)", "classes": GENERIC_CLASSES}
            meta["classes"] = GENERIC_CLASSES
            z_eval = z_support + max(0.6 * h, 0.03)
        else:
            def pose_fn(world_xy, yaw, _hull=hull, _rq=rest_quat):
                return bc.object_pose_at(world_xy, yaw, z_support, _hull, _rq)
            z_eval = z_support
        restore_placed()
        kit.sync_world()
        trunks = trunk_sel or (SMOKE_TRUNKS if args.smoke else trunk_grid(z_support - world.floor_z))
        result = bc.compute(stance, lattice_world, z_eval, pose_fn, grasps if not (args.generic_pinch or args.generic_contact) else None, ignore_grasp,
                            arm=band_arm, trunks=trunks, max_grasp_offset=max_off)
        gset = [(g.key, g.name, g.position.tolist(), g.quat.tolist()) for g in grasps] if grasps else ("contact:" + ",".join(CONTACT_CLASSES_NAMES) if args.generic_contact else "generic:" + ",".join(GENERIC_CLASSES))
        key = _key({"scene": args.scene, "support": args.support, "z": round(z_support, 2), "stance": stance["edge"], "s": args.stand_off, "object": meta.get("object"),
                    "scale": scale, "grasps": gset, "rest": rest_quat.tolist(), "placed": meta["placed_specs"], "under": args.under, "grid": meta["grid"], "lattice": lattice,
                    "arms": "untucked", "seeds": 12, "arm": band_arm, "height_class": meta.get("height_class"), "postures": list(POSTURES)})
        oname = f"h{meta['height_class']:.2f}" if args.generic_contact else (args.object.replace("/", "-") if args.object else "generic")
        for ps in placed_specs:
            oname += f"__placed-{ps.category}-{ps.model}" + (f"-s{ps.scale[0]:g}" if ps.scale else "") + ("-focus" if ps.mode == PLACED_MODE_FOCUS else "")
        mode_name = "contact" if args.generic_contact else meta["mode"]
        name = f"{mode_name}__{args.scene}__{args.support}{'-under-' + args.under if args.under else ''}__{band_arm if band_arm != 'right' else ''}{'_' if band_arm != 'right' else ''}{oname}__{key}"
        if args.smoke:
            meta["sanity"] = band_sanity(stance, result, z_eval)

    # ---- write
    lat_all = np.concatenate([c[:, 1] for c in result["cells"].values() if len(c)]) if any(len(c) for c in result["cells"].values()) else np.zeros(0)
    # kit_stats = this band's share of the kit counters (the worker's kit serves several bands; a fresh process's
    # counters start at zero, so the single-band value is unchanged)
    stats1 = kit.stats_snapshot()
    meta.update({"trunks": result["trunks"], "union_extent": [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None,
                 "n_reachable_total": int(sum(len(c) for c in result["cells"].values())), "key": key, "time_s": round(time.time() - t_load, 1),
                 "setup_s": round(setup_s, 1), "kit_stats": {k: stats1[k] - stats0.get(k, 0) for k in stats1}})
    arrays = {trunk_key(tr, ti): np.asarray(c, dtype=np.float32).reshape(-1, 2) for (tr, ti), c in result["cells"].items()}
    for (tr, ti), p in result.get("pairs", {}).items():
        arrays[trunk_key(tr, ti) + "__pairs"] = np.asarray(p, dtype=np.int32).reshape(-1, 3)
    for (tr, ti), m in result.get("classes", {}).items():
        if meta.get("classes"):
            arrays[trunk_key(tr, ti) + "__classes"] = np.asarray(m, dtype=np.int32).reshape(-1)
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    summary = {"file": os.path.join(out_dir, name + ".npz"), "union_extent": meta["union_extent"], "n_reachable_total": meta["n_reachable_total"],
               "trunks": [(t["translate"], t["tilt"], t["n_reachable"], t["collides"], t.get("time_s"), t.get("posture")) for t in result["trunks"]],
               "time_s": meta["time_s"]}
    if args.smoke:
        summary["shoulder_fk_check"] = [(r["translate"], r["tilt"], r["shoulder_err_mm"]) for r in meta.get("shoulder_fk_check", [])]
        summary["sanity"] = meta.get("sanity")
    if job_id is not None:
        summary["job_id"] = job_id
    line = f"REACH_BANDS {json.dumps(summary, default=str)}"
    print(line, flush=True)
    if job_id is not None:  # the worker's per-job run.log (what collect_bands.py reads the summary from)
        with open(os.path.join(out_dir, "run.log"), "a") as f:
            f.write(line + "\n")
    return summary


def run_single(args):
    """One band per process (the original CLI): world load, band, ``og.shutdown()``."""
    import omnigibson as og

    out_dir = args.out or cache_root()
    os.makedirs(out_dir, exist_ok=True)
    floor_mode, rooms = band_rooms(args)
    arms = band_arms(args)
    variant, removed = (None, [])
    if not floor_mode or not args.full_load:
        variant, removed = _chairs_variant(args.scene, args.support, rooms, os.path.join(out_dir, "scene_variants"), removed=[] if floor_mode else None,
                                           slim=not args.full_load)
    t_load = time.time()
    world, kit = load_world(args, rooms, variant, removed, arms)
    setup_s = time.time() - t_load
    try:
        compute_band(args, world, kit, out_dir, rooms=rooms, removed=removed, floor_mode=floor_mode, arms=arms, t_load=t_load, setup_s=setup_s)
    except BaseException:
        import traceback

        traceback.print_exc()
        print("REACH_BANDS_FAILED", flush=True)
        raise
    finally:
        og.shutdown()


def run_jobs(args):
    """Worker mode (``--jobs MANIFEST [--ids ID ...] --out ROOT``): many bands in one process.

    Every selected job's ``args`` (the ``make_jobs.py`` manifest format) is parsed like a single-band command line and
    computed on a shared ``(world, kit)``; jobs are grouped by :func:`band_world_key` (scene, loaded rooms, removed
    chairs) and the world is reloaded in place (``og.clear()`` + a new ``SimWorld`` / ``RobotKit``, as
    ``sample_isg.WorldPool`` does) only when the key changes. Between bands of one world
    :func:`reset_world_between_bands` removes the previous band's objects, re-syncs the collision world and resets the
    cuRobo seed streams, so band k of a process is computed exactly like band k alone (verified 2026-09-16 on three
    two-trunk jobs: byte-identical arrays, metas equal up to the timing fields). Per job: the band files in
    ``ROOT/<id>/`` (same names as a single run), ``ROOT/<id>/run.log`` (this job's log records + the ``REACH_BANDS``
    line), a ``REACH_BANDS_JOB <id>`` stdout line before the usual ``REACH_BANDS {...}`` (which also carries
    ``job_id``) and a ``<id> rc=<rc> wall=<s> n_written=-1`` line appended to ``ROOT/done.txt`` - the layout of the pod
    runner / ``run_local.sh``, so ``collect_bands.py`` reads the tree unchanged. A failing job logs its traceback, prints
    ``REACH_BANDS_FAILED <id>`` and does not stop the others."""
    import omnigibson as og

    with open(args.jobs) as f:
        manifest = json.load(f)
    jobs = select_jobs(manifest, args.ids)
    root = args.out or cache_root()
    os.makedirs(root, exist_ok=True)
    parser = build_parser()
    prepared = []  # (job, job args, floor_mode, rooms, removed, world key) or (job, error)
    for job in jobs:
        try:
            jargs = parser.parse_args(list(job["args"]))
            if jargs.jobs:
                raise ValueError("a job's args cannot contain --jobs")
            jargs.out = os.path.join(root, job["id"])
            jargs.log_level, jargs.prebuild_planner = args.log_level, args.prebuild_planner
            jargs.cameras, jargs.full_load = args.cameras or jargs.cameras, args.full_load or jargs.full_load
            jargs.ik_tail_sizes, jargs.mesh_cache = args.ik_tail_sizes or jargs.ik_tail_sizes, args.mesh_cache or jargs.mesh_cache
            key = band_world_key(jargs)
            floor_mode, rooms = band_rooms(jargs)
            prepared.append((job, jargs, floor_mode, rooms, list(key[2]), key))
        except (SystemExit, Exception) as e:  # noqa: BLE001 - a malformed job must not stop the others
            prepared.append((job, e))
    ordered = group_by_key(prepared, lambda p: p[5] if len(p) > 2 else ("__invalid__", p[0]["id"]))
    n_keys = len({p[5] for p in prepared if len(p) > 2})
    log.info("worker: %d jobs, %d world keys (%d reloads), order: %s", len(ordered), n_keys, max(n_keys - 1, 0), [p[0]["id"] for p in ordered])
    world = kit = None
    cur_key = None
    n_bands = 0
    for p in ordered:
        job = p[0]
        jid = job["id"]
        t_job = time.time()
        job_dir = os.path.join(root, jid)
        os.makedirs(job_dir, exist_ok=True)
        # this job's log records (the ``ispatialgym`` logger has its own stdout handler and does not propagate to the root)
        handler = logging.FileHandler(os.path.join(job_dir, "run.log"))
        handler.setFormatter(logging.Formatter("[%(levelname)s %(name)s] %(message)s"))
        handler.setLevel(logging.INFO)
        logging.getLogger("ispatialgym").addHandler(handler)
        print(f"REACH_BANDS_JOB {jid}", flush=True)
        rc = 0
        try:
            if len(p) == 2:
                raise p[1]
            _, jargs, floor_mode, rooms, removed, key = p
            arms = band_arms(jargs)
            if world is not None and key == cur_key and len(kit.backend._world_mesh_names) + MESH_CACHE_RESERVE > kit.backend.mesh_cache_size:
                log.info("worker: cuRobo mesh store nearly full (%d of %d after %d bands); reloading the scene", len(kit.backend._world_mesh_names),
                         kit.backend.mesh_cache_size, n_bands)
                cur_key = None  # forces the reload below
            if world is None or key != cur_key:
                if world is not None:
                    log.info("worker: world key changed (%s -> %s); reloading the scene", cur_key, key)
                    og.clear()
                    world = kit = None
                    cur_key = None
                variant = None
                if removed or not jargs.full_load:
                    variant, _ = _chairs_variant(jargs.scene, jargs.support, rooms, os.path.join(root, "scene_variants"), removed, slim=not jargs.full_load)
                world, kit = load_world(jargs, rooms, variant, removed, arms)
                cur_key = key
                n_bands = 0
            else:
                reset_world_between_bands(world, kit)
                log.info("worker: world reused for %s (%s)", jid, key)
            setup_s = time.time() - t_job
            compute_band(jargs, world, kit, job_dir, rooms=rooms, removed=removed, floor_mode=floor_mode, arms=arms, t_load=t_job, setup_s=setup_s, job_id=jid,
                         spawn_suffix=f"_b{n_bands}" if n_bands else "")
            n_bands += 1
        except KeyboardInterrupt:
            raise
        except BaseException:  # noqa: BLE001 - SystemExit from argument checks included; one job must not abort the others
            rc = 1
            log.error("worker: job %s failed", jid, exc_info=True)
            print(f"REACH_BANDS_FAILED {jid}", flush=True)
            if world is not None:
                n_bands += 1  # the failed band may have spawned objects: keep the name suffixes unique
                try:
                    reset_world_between_bands(world, kit)
                except Exception:  # noqa: BLE001 - the world is unusable: reload it for the next job
                    log.error("worker: could not recover the world after job %s; it is reloaded for the next job", jid, exc_info=True)
                    try:
                        og.clear()
                    except Exception:  # noqa: BLE001
                        log.error("worker: og.clear() failed", exc_info=True)
                    world = kit = None
                    cur_key = None
        finally:
            with open(os.path.join(root, "done.txt"), "a") as f:
                f.write(f"{jid} rc={rc} wall={int(time.time() - t_job)} n_written=-1\n")
            logging.getLogger("ispatialgym").removeHandler(handler)
            handler.close()
    og.shutdown()


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", force=True)
    # import order matters: the cuRobo backend (through robot_utils) pins pip-warp before omnigibson starts Isaac Sim
    from ispatialgym.sampling import robot_utils  # noqa: F401
    from ispatialgym.sampling import sim_utils  # noqa: F401

    if args.jobs:
        return run_jobs(args)
    if args.scene is None or args.support is None:
        raise SystemExit("--scene and --support are required (or --jobs MANIFEST)")
    return run_single(args)


if __name__ == "__main__":
    main()
