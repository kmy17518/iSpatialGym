"""Scene-independent reach maps: where, relative to the robot, the right hand can grasp an object model resting on a table edge.

Replaces the per-scene / per-support bands of :mod:`reach_bands` (kept, explicitly separate: a map is never returned for a band
query or vice versa). A map is computed in a minimal synthetic world - the R1Pro alone on the floor (cuRobo: the floor cuboid and
self-collision; no household scene is loaded) - for one (object library entry, rest, exact support-height anchor, arm), and holds:

* the **frame**: base_link projected on the floor, +x ahead, +y the robot's left, z up from the floor;
* the **support**: a synthetic slab whose top is at the anchor height ``h`` above the floor, its near edge ``d_e`` ahead of the base
  (``d_e`` = the actual base-to-edge distance: the sampler's stand-off plus the support's overhang), ``slab_depth_m`` deep, ``slab_long_m``
  wide (>= the longest support edge of the inventory + margin), ``slab_thickness_m`` thick (a slab, never a block to the floor);
* per **edge distance** ``d_e`` x **trunk** (``reach_bands.trunk_grid(h)``; eligibility: the idle-arm posture free of the slab and of
  itself) x **cell** (an ``lattice_m`` lattice of object-root xy) x **object yaw** (``n_yaws``) x **grasp** (validated grasps, their
  finger-swapped twins unless stored, members of non-yaw-equivalent symmetry families): a status code (:data:`STATUS`) - feasible, or why
  not, or ``skipped`` (unknown: the accelerated mode's early cut; never labelled exhaustive).

Per (trunk, cell, yaw, grasp): ``T_world_eef = place_object(rest, yaw, cell, h) @ T_object_eef`` (``schema.place_object``: measured
``h_root``, no offset), the demo pre-grasp (``conventions.pre_pose``, 0.10 m back along +z); the hand's exact hulls against the slab
under the contact policy at the pre-grasp, along the straight approach and through the closing sweep (IK-independent, per ``d_e``);
batched IK of the pre-grasp (``seeds`` seeds) and of the grasp seeded from it (same branch: max joint step <= ``branch_max_rad``);
then, per ``d_e``, every non-finger sphere of the whole robot at the pre-grasp, the grasp and ``approach_configs`` joint-interpolated
configurations clear of the slab (``contact_policy.sphere_support_ok``), the hand hulls at the FK eef poses within the finger
allowance, and the arm's spheres (not the hand's) clear of the object at the cell. The IK itself never sees the slab - that is
what lets one IK pass serve every ``d_e`` (all seeds' solutions are kept and the first passing one counts).

Maps are proposal / filter data: the sampler transforms cells into a real stance on a real edge and re-checks everything exactly
(scene geometry, clutter, legs, approach) - a map's feasible cell is not an executable trajectory.

**Floor maps** (``kind="floor"``, i7-floor 2026-11-07; the legacy floor band's successor): the object on the floor the robot stands on -
height 0, the support the floor plane (no edge: one edge distance 0.0, cells from ``FLOOR_X_MIN_M`` = 0.20 m ahead to ``x_max_m``, laterals
+-``FLOOR_Y_HALF_M`` = 0.80 m: :func:`floor_config`), the trunks of ``reach_bands.trunk_grid(0)`` (incl. the low-top tilt and the ground
trunks), the contact policy's floor mode (rule 5: the fingers' floor allowance, the robot's ground-contact links scoped out of the sphere
check); meta ``plane: "floor"`` (in the key; slab maps have no such field). Looked up with ``ReachMap.find(kind="floor", height=0.0,
anchors=None)`` (``tabletop.floor_map_bands``). i7-floorfix 2026-11-07 (config ``floor_fingers``, :data:`FLOOR_FINGERS` "policy"): the
floor IS the support there, so the pre-grasp / grasp IK run without the cuRobo ground slab - a padded finger sphere touching the floor
failed the IK before the 2 mm finger rule was read (probe 2026-11-07, umbrella top_2 at fix1's configuration, left arm, t2.00/k0: 0 / 8 grasp
seeds with the slab, 8 / 8 without it; the lower finger's 7 mm sphere 5.2 mm into the slab, its hull 3.5 mm above the floor) - and the
fingers are judged by the policy's floor allowance on their hulls alone; every other link but the ground-contact ones is checked against the floor
plane by its spheres at the pre-grasp, the grasp and the approach configurations (the slab maps' treatment of their slab).

    python -m ispatialgym.sampling.reach_maps bench
    python -m ispatialgym.sampling.reach_maps compute --jobs M.json --glib ROOT --anchors anchors.json [--heights 0.7351 ...] --out MAPS
    python -m ispatialgym.sampling.reach_maps compute --kind floor --jobs M.json --glib ROOT --out MAPS [--arm left]
    python -m ispatialgym.sampling.reach_maps info MAP.json
"""

from __future__ import annotations

import argparse
import glob
import functools
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import hand_geometry as H
from ispatialgym.grasping import schema as S

SCHEMA = "isg-reach-map/1"
STATUS = {"unevaluated": 0, "feasible": 1, "off_support": 2, "reach_bound": 3, "hand_policy": 4, "ik_pre": 5, "ik_grasp": 6,
          "slab_spheres": 7, "object_arm": 8, "skipped": 9, "trunk_ineligible": 10, "branch_jump": 11}
STATUS_NAME = {v: k for k, v in STATUS.items()}


@dataclass
class MapConfig:
    arm: str = "right"
    lattice_m: float = 0.08
    x_max_m: float = 1.20
    y_half_m: float = 0.88
    edge_distances: Tuple[float, ...] = (0.35, 0.45, 0.55)
    n_yaws: int = 8
    seeds: int = 8
    batch: int = 2048
    branch_max_rad: float = 1.0
    approach_configs: int = 3
    family_samples: int = 4
    twins: bool = True
    mode: str = "full"  # "accelerated": the early cut (measured 2026-11-02: 1-4 % fewer IK problems, up to 6.5 % fewer feasible entries)
    slab_depth_m: float = 1.2
    slab_long_m: float = 3.2
    slab_thickness_m: float = 0.04
    hand_spacing_m: float = 0.006
    reach_margin_m: float = 0.03
    postures: Tuple[str, ...] = ("untucked", "ready", "tucked")
    # of-v2maps 2026-11-07 (prompt I item 3): the fingers' opening of the hand-vs-slab check - "grasp": each grasp's own pre-shape / contact
    # opening (the oracle's), "max": the batch rule of the maps before (max pre-shape, min contact for every grasp). A map without the
    # field was computed with "max" (CONFIG_LEGACY_DEFAULTS)
    preshape: str = "grasp"
    # i7-floor 2026-11-07: the lattice's first x (m ahead of the base); None = min(edge_distances) (a slab map: no cell before the edge).
    # A floor map has no edge (edge_distances (0.0,)) and starts at FLOOR_X_MIN_M. Left out of key() while None (the keys of the maps
    # computed before the field existed stay as they were)
    x_min_m: Optional[float] = None
    # i7-floorfix 2026-11-07: a floor map's finger rule against the floor it stands on (FLOOR_FINGER_RULES): "policy" = the pre-grasp / grasp
    # IK without the cuRobo ground slab (the support itself), the fingers judged only by the contact policy's floor allowance on their hulls,
    # every other link but the ground-contact ones by its spheres against the floor plane; "spheres" = the former rule (the padded finger
    # spheres in the IK against the slab). None for a slab map (left out of key(): their keys stay); a floor map without the field
    # was computed with "spheres" (CONFIG_LEGACY_DEFAULTS)
    floor_fingers: Optional[str] = None
    # i7-unblock 2026-11-07 (owner decision: the former rule was a bug, nextto_04): the trunk eligibility rule (TRUNK_ELIGIBILITY_RULES,
    # :func:`trunk_posture_groups`) - "per_d_e": a trunk is ineligible only at the edge distances where every idle-arm posture collides
    # (self or the slab); "all_d_e" / None: the former rule (one posture free of the slab at EVERY d_e, else the whole trunk is dropped).
    # None is left out of key() (the maps computed before the field existed keep their keys; CONFIG_LEGACY_DEFAULTS reads them as "all_d_e").
    # i7-unblock 2026-11-07 (owner decision): "per_d_e" is the default of new builds; a floor map (one edge distance: both rules give the
    # same map) stays None (floor_config), so its key does not move
    trunk_eligibility: Optional[str] = "per_d_e"  # = TRUNK_ELIGIBILITY_DEFAULT

    def key(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("batch")  # does not change results
        for k in ("x_min_m", "floor_fingers", "trunk_eligibility"):
            if d.get(k) is None:
                d.pop(k)
        return d


#: the value a config field had before it existed (maps written earlier carry no such field)
CONFIG_LEGACY_DEFAULTS: Dict[str, Any] = {"preshape": "max", "x_min_m": None, "floor_fingers": "spheres", "trunk_eligibility": "all_d_e"}
#: i7-unblock 2026-11-07: the trunk eligibility rules (``MapConfig.trunk_eligibility``; None = "all_d_e", the former rule)
TRUNK_ELIGIBILITY_RULES: Tuple[str, ...] = ("all_d_e", "per_d_e")
#: the rule of new builds (MapConfig's default; the --skip-existing matchers' default)
TRUNK_ELIGIBILITY_DEFAULT = "per_d_e"


def trunk_rule(cfg: "MapConfig") -> str:
    """The trunk eligibility rule of ``cfg`` (``MapConfig.trunk_eligibility``; None = "all_d_e")."""
    rule = cfg.trunk_eligibility or "all_d_e"
    if rule not in TRUNK_ELIGIBILITY_RULES:
        raise ValueError(f"trunk_eligibility {rule!r} is not one of {TRUNK_ELIGIBILITY_RULES}")
    return rule


def trunk_posture_groups(postures: Sequence[str], n_de: int, free_of, rule: str = "all_d_e", obstacle_free_of=None
                         ) -> Tuple[List[Tuple[str, List[int]]], Dict[str, List[bool]]]:
    """A trunk's idle-arm posture per edge distance (pure; i7-unblock 2026-11-07). ``free_of(posture)`` -> None when the posture
    self-collides, else its slab (support) clearance per edge distance (``n_de`` bools, the map's ``edge_distances`` order). Returns
    ``(groups, collide)``: ``groups`` = ``[(posture, [d_e index, ...]), ...]`` in posture order (no group: the trunk is ineligible
    everywhere), ``collide`` = the ``slab_collide_by_posture`` record (the self-free postures that collide with the slab at some d_e).

    Both rules first take a posture free of the slab at EVERY d_e - the first one (the postures after it are not evaluated), one group over
    all d_e: the former choice, so a trunk eligible under the former rule keeps its posture and its cells. Without such a posture,
    "all_d_e" (the former rule) drops the trunk; "per_d_e" gives each d_e the first posture free of the slab there, and a d_e no posture
    clears stays ineligible.

    ``obstacle_free_of(posture)`` (optional; i7-lean b4230177's interior rule for a FIXED compartment): its clearance of the compartment's
    obstacles per d_e. Among the postures free of the slab at every d_e the first free of the obstacles at the most d_e is taken (the
    search stops at one free of them everywhere); in the "per_d_e" fallback each d_e takes the first slab-free posture there that is also
    obstacle-free there, else the first slab-free one."""
    collide: Dict[str, List[bool]] = {}
    evaluated: List[Tuple[str, List[bool]]] = []
    best: Optional[Tuple[int, str]] = None
    obs: Dict[str, List[bool]] = {}
    for p in postures:
        f = free_of(p)
        if f is None:
            continue
        f = [bool(v) for v in f]
        if len(f) != n_de:
            raise ValueError(f"posture {p}: {len(f)} slab verdicts for {n_de} edge distances")
        if not all(f):
            collide[p] = f
            evaluated.append((p, f))
            continue
        if obstacle_free_of is None:
            return [(p, list(range(n_de)))], collide
        o = [bool(v) for v in obstacle_free_of(p)]
        obs[p] = o
        if best is None or sum(o) > best[0]:
            best = (sum(o), p)
        if all(o):
            break
    if best is not None:
        return [(best[1], list(range(n_de)))], collide
    if rule == "all_d_e":
        return [], collide
    if rule != "per_d_e":
        raise ValueError(f"trunk_eligibility {rule!r} is not one of {TRUNK_ELIGIBILITY_RULES}")
    if obstacle_free_of is not None:
        for p, _ in evaluated:
            obs[p] = [bool(v) for v in obstacle_free_of(p)]
    choice: Dict[int, Optional[str]] = {}
    for k in range(n_de):
        free_here = [p for p, f in evaluated if f[k]]
        clear_here = [p for p in free_here if obs.get(p, [True] * n_de)[k]]
        choice[k] = (clear_here or free_here or [None])[0]
    return [(p, [k for k in range(n_de) if choice[k] == p]) for p, _ in evaluated if any(choice[k] == p for k in range(n_de))], collide


def mark_partial(rec: Dict[str, Any], groups: Sequence[Tuple[str, Sequence[int]]], edge_distances: Sequence[float], status: Dict[float, Any],
                 arrays: Dict[str, np.ndarray], key: str, shape: Tuple[int, ...]) -> None:
    """A trunk eligible at some edge distances only ("per_d_e"): ``rec`` records the posture per eligible d_e (``posture_by_d_e``) and the
    ineligible ones (``ineligible_d_e``), whose status arrays are ``trunk_ineligible``; a trunk with one posture over every d_e is left as the
    former rule records it (no new fields)."""
    covered = {k for _, ks in groups for k in ks}
    if len(groups) == 1 and len(covered) == len(edge_distances):
        return
    rec["posture_by_d_e"] = {f"{edge_distances[k]:.2f}": p for p, ks in groups for k in ks}
    rec["ineligible_d_e"] = [float(edge_distances[k]) for k in range(len(edge_distances)) if k not in covered]
    rec["eligibility"] = "per_d_e"
    for k in range(len(edge_distances)):
        if k in covered:
            continue
        d_e = edge_distances[k]
        if status.get(d_e) is not None:
            status[d_e].fill_(STATUS["trunk_ineligible"])
        arrays[f"s{d_e:.2f}_{key}"] = np.full(shape, STATUS["trunk_ineligible"], dtype=np.uint8)

#: the map kinds (``--kind``): ``tabletop`` (library grasps of an object on a slab edge), ``brace`` (contact classes on a slab edge),
#: ``floor`` (i7-floor 2026-11-07: library grasps of an object on the floor plane the robot stands on - h = 0, no edge)
KINDS = ("tabletop", "brace", "floor")
#: the support plane a map's cells stand on (meta ``plane``; a map without the field is a slab map)
PLANE_OF_KIND = {"tabletop": "slab", "brace": "slab", "floor": "floor"}
#: the floor kind (the legacy floor band's raster, ``reach_bands`` floor mode: cells 0.2 m ahead of the base, lateral +-STRIP_HALF_WIDTH):
#: one "edge distance" 0.0 (no edge: the floor is everywhere), the lattice from FLOOR_X_MIN_M, laterals +-FLOOR_Y_HALF_M, height 0
FLOOR_EDGE_DISTANCES: Tuple[float, ...] = (0.0,)
FLOOR_X_MIN_M = 0.20
FLOOR_Y_HALF_M = 0.80
FLOOR_HEIGHT = 0.0
#: the synthetic anchor of a floor map (the anchor set has no h = 0: a floor lookup passes ``anchors=None`` and height 0.0)
FLOOR_ANCHOR: Dict[str, Any] = {"key": "floor", "height": FLOOR_HEIGHT, "surfaces": ["floor"]}
#: i7-floorfix 2026-11-07: a floor map's finger rules (``MapConfig.floor_fingers``) and the one the sampler's exact IK applies at a floor grasp
#: (``ik.IKChecker.exact``: the registered floor out of the world for the whole grasp chain, the fingers by the policy's floor allowance) -
#: ``tabletop.floor_map_bands`` requires it, so a floor map of the former rule ("spheres") never serves
FLOOR_FINGER_RULES: Tuple[str, ...] = ("policy", "spheres")
FLOOR_FINGERS = "policy"


def floor_config(**kw) -> MapConfig:
    """The :class:`MapConfig` of a floor map (``kind="floor"``): no edge (one distance 0.0), cells from FLOOR_X_MIN_M, laterals
    +-FLOOR_Y_HALF_M, the finger rule :data:`FLOOR_FINGERS`; ``kw`` sets the other fields (arm, lattice, seeds, ...)."""
    kw = {"edge_distances": FLOOR_EDGE_DISTANCES, "x_min_m": FLOOR_X_MIN_M, "y_half_m": FLOOR_Y_HALF_M, "floor_fingers": FLOOR_FINGERS,
          "trunk_eligibility": None, **kw}  # one edge distance: the trunk eligibility rules coincide (i7-unblock: the floor keys stay)
    return MapConfig(**kw)


def config_value(meta: Dict[str, Any], name: str) -> Any:
    cfg = meta.get("config") or {}
    return cfg[name] if name in cfg else CONFIG_LEGACY_DEFAULTS.get(name)


def map_plane(meta: Dict[str, Any]) -> str:
    """The support plane of a map (meta ``plane``; maps written before the floor kind are slab maps)."""
    return str(meta.get("plane") or PLANE_OF_KIND.get(meta.get("kind", "tabletop"), "slab"))


def lattice(cfg: MapConfig) -> np.ndarray:
    x0 = min(cfg.edge_distances) if cfg.x_min_m is None else float(cfg.x_min_m)
    xs = np.arange(x0, cfg.x_max_m + 1e-9, cfg.lattice_m)
    ys = np.arange(-cfg.y_half_m, cfg.y_half_m + 1e-9, cfg.lattice_m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def support_mask(cells: np.ndarray, reach_back: np.ndarray, edge_distance: float, kind: str = "tabletop") -> np.ndarray:
    """(C, Y) bool: the object (root at each cell, footprint reaching ``reach_back`` (Y,) toward the robot at each yaw) stands on the
    support. A slab map: the footprint behind the near edge ``edge_distance`` (cell + reach_back >= d_e). A floor map: everywhere (the
    floor has no edge; the lattice starts FLOOR_X_MIN_M ahead of the base)."""
    if kind == "floor":
        return np.ones((len(cells), len(reach_back)), dtype=bool)
    return (cells[:, None, 0] + reach_back[None, :]) >= edge_distance


# ------------------------------------------------------------------------------------------------ grasp set


def grasp_set(lib: S.ModelLibrary, rest_id: str, cfg: MapConfig) -> List[Dict[str, Any]]:
    """The map's grasp axis: every validated grasp of the rest, its twin (eef turned pi about +z) unless an equivalent row is stored,
    members of its family when the family is NOT vertical-yaw-equivalent (``family_samples`` over the interval incl. endpoints; the
    yaw-equivalent ones are covered by the yaw axis and keep the representative with ``yaw_equivalent``)."""
    from ispatialgym.grasping import symmetry as SY

    out: List[Dict[str, Any]] = []
    fams = lib.families()
    rows = lib.grasps(rest_id)
    Ts = [S.pose_of(g["T_object_eef"]) for g in rows]
    for g, T in zip(rows, Ts):
        fam = fams.get(g.get("family_id") or "", {"type": "none"})
        params = [0.0]
        if fam.get("type") in ("rotational", "translational") and not fam.get("vertical_yaw_equivalent"):
            # the representative (the physics-tested pose) always, plus samples over the interval incl. its endpoints
            params = sorted({0.0, *[round(float(v), 6) for v in S.family_samples(fam, cfg.family_samples)]})
        for p in params:
            Tm = S.family_member_T(fam, T, p)
            base = {"base": g["name"], "family_id": g.get("family_id"), "param": round(float(p), 5), "preshape_q": g.get("preshape_q", 0.05),
                    "contact_q": 0.5 * float(g.get("width_m", 0.04)) + 0.0009, "physics": "tested" if abs(p) < 1e-9 else "family_member",
                    "yaw_equivalent": bool(fam.get("vertical_yaw_equivalent"))}
            out.append({**base, "name": g["name"] if abs(p) < 1e-9 else f"{g['name']}@{p:+.3f}", "twin": False, "T": Tm})
            if cfg.twins:
                Tt = SY.twin_T(Tm)
                if not any(SY._pose_close(Tt, T2, 0.015, 20.0) for T2 in Ts):
                    out.append({**base, "name": f"{out[-1]['name']}~r180", "twin": True, "T": Tt})
    return out


# ------------------------------------------------------------------------------------------------ brace / contact poses (prompt 2 §10)

BRACE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
BRACE_SIDES = {"near": (1.0, 0.0), "far": (-1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}  # horizontal approach in the map frame
BRACE_PITCHES_DEG = (0.0, 45.0)
#: the oracle's brace stop: the fingertip plane this far before the face along the approach (``conventions.fingertip_eef``, spec 5.5)
BRACE_CLEARANCE_M = 0.005


def brace_classes(fractions: Sequence[float] = BRACE_FRACTIONS) -> List[Dict[str, Any]]:
    """The contact classes of a brace map (``reach_bands.CONTACT_CLASSES`` generalised): side ``near`` (pads on the face toward the robot,
    pushing away: approach +x of the map frame), ``far`` (-x, pulling), ``left`` / ``right`` (-y / +y, across); approach horizontal or
    pitched 45 deg DOWN toward the face; closing axis ``v`` (vertical-most axis perpendicular to the approach: fingers stacked) or ``h``
    (horizontal, perpendicular to the approach: fingers side by side); at each height fraction of the object's settled height; plus the
    top press (approach straight down, two wrist yaws) at the top. Closed fingers, the fingertip plane ``BRACE_CLEARANCE_M`` before the
    contact point (``conventions.fingertip_eef``: the oracle's brace stop)."""
    out = []
    for side in BRACE_SIDES:
        for pitch in BRACE_PITCHES_DEG:
            for ax in ("v", "h"):
                for f in fractions:
                    out.append({"name": f"{side}_{ax}_p{int(pitch)}_f{f:.1f}", "base": f"{side}_{ax}_p{int(pitch)}", "side": side, "pitch_deg": pitch,
                                "closing": ax, "fraction": f, "twin": False, "family_id": None, "param": None, "preshape_q": 0.0, "contact_q": 0.0,
                                "physics": "geometric_contact", "yaw_equivalent": False})
    for wy in (0, 90):
        out.append({"name": f"top_press_y{wy}", "base": f"top_press_y{wy}", "side": "top", "pitch_deg": 90.0, "closing": f"y{wy}", "fraction": 1.0,
                    "twin": False, "family_id": None, "param": None, "preshape_q": 0.0, "contact_q": 0.0, "physics": "geometric_contact",
                    "yaw_equivalent": False})
    return out


def brace_targets(V_rest: np.ndarray, yaws: Sequence[float], height: float, h_root: float, classes: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """(Y, G, 4, 4) contact eef poses with the object root at the map origin xy, and (Y, G, 3) contact points: per yaw the object's
    hull (``V_rest``: collision vertices in the rest frame) is placed, each class's contact point is the hull's extreme point against the
    approach at the class's height (cross-section +-1 cm; the top press: the top centre), ``fingertip_eef`` stops the fingertips 5 mm before it."""
    from ispatialgym.robot import conventions as C

    nY, nG = len(yaws), len(classes)
    TE = np.tile(np.eye(4), (nY, nG, 1, 1))
    P = np.zeros((nY, nG, 3))
    for yi, yaw in enumerate(yaws):
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        W = V_rest @ R.T
        W[:, 2] += height + h_root
        zlo, zhi = float(W[:, 2].min()), float(W[:, 2].max())
        H = zhi - height
        for gi, c in enumerate(classes):
            if c["side"] == "top":
                top = W[W[:, 2] >= zhi - 0.01]
                p = np.array([top[:, 0].mean(), top[:, 1].mean(), zhi])
                a = np.array([0.0, 0.0, -1.0])
                cl = np.array([1.0, 0.0, 0.0]) if c["closing"] == "y0" else np.array([0.0, 1.0, 0.0])
            else:
                ah = np.array([*BRACE_SIDES[c["side"]], 0.0])
                pr = math.radians(c["pitch_deg"])
                a = ah * math.cos(pr) + np.array([0, 0, -1.0]) * math.sin(pr)
                z = height + c["fraction"] * H
                sl = W[np.abs(W[:, 2] - z) <= 0.01]
                if len(sl) == 0:
                    sl = W
                ctr = sl[:, :2].mean(0)
                e = float(((sl[:, :2] - ctr) @ (-ah[:2])).max())
                p = np.array([ctr[0] - ah[0] * e, ctr[1] - ah[1] * e, z])
                hz = np.array([0.0, 0.0, 1.0])
                cl = hz - (hz @ a) * a if c["closing"] == "v" else np.cross(hz, ah)
                cl = cl / np.linalg.norm(cl)
            pos, quat = C.fingertip_eef(th.as_tensor(p, dtype=th.float32), th.as_tensor(a, dtype=th.float32), th.as_tensor(cl, dtype=th.float32), clearance=BRACE_CLEARANCE_M)
            TE[yi, gi] = S.pose_mat(pos.numpy(), quat.numpy())
            P[yi, gi] = p
    return TE, P


# ------------------------------------------------------------------------------------------------ the synthetic world


class SynthWorld:
    """The R1Pro on an empty floor with its cuRobo backend (floor cuboid only): kinematics, batched IK, sphere FK. One per process,
    reused across heights / models (``reset`` between maps: the IK solvers' seed streams)."""

    def __init__(self, cfg: MapConfig):
        from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # warp pinning before omnigibson
        import omnigibson as og

        from ispatialgym.oracle import r1pro_model as M

        t0 = time.time()
        self.cfg, self.og, self.M = cfg, og, M
        self.env = og.Environment(configs={"env": {"action_frequency": 30, "physics_frequency": 120}, "scene": {"type": "Scene"},
                                           "robots": [M.joylo_robot_config("robot", obs_modalities=["proprio"], position=[0.0, 0.0, 0.0])]})
        self.robot = self.env.scene.robots[0]
        for _ in range(3):
            og.sim.step()
        self.backend = R1ProCuRobo(self.robot, prebuild=(), mesh_cache_size=8, cuboid_cache_size=8)
        self.backend.update_world()
        rp, rq = self.backend.root_pose()
        assert float(th.linalg.norm(rp[:2])) < 1e-4 and abs(float(rq[3])) > 0.99999, f"the synthetic world needs the root at the origin ({rp}, {rq})"
        self.root_z = float(rp[2])
        self.jidx = self.backend._jidx
        self.trunk_limits = (self.robot.joint_lower_limits[self.robot.trunk_control_idx].clone(), self.robot.joint_upper_limits[self.robot.trunk_control_idx].clone())
        self.dev = self.backend.device_cfg.device
        self.setup_s = time.time() - t0
        self._shoulder: Dict[Tuple[float, float], Tuple[np.ndarray, float]] = {}

    def q_config(self, translate: float, tilt: float, posture: str = "untucked", fingers: float = 0.05) -> th.Tensor:
        from ispatialgym.robot import conventions as C

        q = th.zeros(28)
        q[self.jidx["base_footprint_z_joint"]] = C.NOMINAL_BASE_Z - self.root_z
        q[self.robot.trunk_control_idx] = C.torso_q(translate, tilt, self.trunk_limits)
        for arm in ("left", "right"):
            if posture == "untucked":
                q[self.robot.arm_control_idx[arm]] = self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[arm]].clone()
            elif posture == "ready":
                q[self.robot.arm_control_idx[arm]] = th.tensor(self.M.ARM_READY_Q[arm], dtype=th.float32)
            else:
                q[self.robot.arm_control_idx[arm]] = 0.0
            q[self.robot.gripper_control_idx[arm]] = fingers
        return q

    def entry(self):
        return self.backend._batch_ik_entry(f"{self.cfg.arm}_arm", self.cfg.batch, seeds=self.cfg.seeds)

    def full_q(self, entry, q_active: th.Tensor, q_lock: th.Tensor) -> th.Tensor:
        """(N, 28) configurations (GPU): ``q_lock`` with the IK entry's active joints from ``q_active``."""
        names = list(entry.rsc.joint_names)
        idx = th.tensor([self.jidx[n] for n in names], device=self.dev)
        q = q_lock.to(self.dev).unsqueeze(0).repeat(q_active.shape[0], 1)
        q[:, idx] = q_active.to(self.dev)
        return q

    def _wb(self):
        return self.backend.kin_entry("whole_body")

    def _fk(self, q_full: th.Tensor, chunk: int = 8192):
        wb = self._wb()
        self.backend.refresh_locked_joints("whole_body", q_full[0].cpu())
        for s0 in range(0, q_full.shape[0], chunk):
            js = self.backend._active_js(wb, q_full[s0:s0 + chunk].contiguous())
            with th.no_grad():
                yield wb.fk.compute_kinematics(js)

    def spheres(self, q_full: th.Tensor) -> th.Tensor:
        """(N, S, 4) whole-body spheres (root = world frame; radii with the model's buffer) of full configurations (any batch)."""
        n = q_full.shape[0]
        return th.cat([st.robot_spheres.reshape(-1, st.robot_spheres.shape[-2], 4) for st in self._fk(q_full)]).reshape(n, -1, 4)

    def eef(self, q_full: th.Tensor) -> th.Tensor:
        """(N, 4, 4) eef poses (world) of full configurations."""
        out = []
        for st in self._fk(q_full):
            p = st.tool_poses.get_link_pose(self.M.EEF_LINK[self.cfg.arm])
            out.append(_mat(p.position.reshape(-1, 3), p.quaternion.reshape(-1, 4)))
        return th.cat(out)

    def link_of_sphere(self) -> List[str]:
        return list(self.backend._collision_costs(self._wb())["link_of_sphere"])

    def shoulder(self, translate: float, tilt: float) -> Tuple[np.ndarray, float]:
        """(shoulder xyz, arm reach) of the trunk (``eef_reach_volume``'s geometric bound: link1 position, straight-arm length)."""
        k = (round(translate, 4), round(tilt, 4))
        if k not in self._shoulder:
            arm = self.cfg.arm
            q = self.q_config(translate, tilt)
            sh = self._link_pos(q, f"{arm}_arm_link1")
            q0 = q.clone()
            q0[self.robot.arm_control_idx[arm]] = 0.0
            L = float(np.linalg.norm(self._link_pos(q0, self.M.EEF_LINK[arm]) - self._link_pos(q0, f"{arm}_arm_link1")))
            self._shoulder[k] = (sh, L)
        return self._shoulder[k]

    def _link_pos(self, q: th.Tensor, link: str) -> np.ndarray:
        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        self.og.sim.step_physics()
        self.robot.set_joint_positions(q)
        lk = self.robot.eef_links[self.cfg.arm] if link == self.M.EEF_LINK[self.cfg.arm] else self.robot.links[link]
        return lk.get_position_orientation()[0].numpy().astype(float)


def _mat(pos: th.Tensor, quat_wxyz: th.Tensor) -> th.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    R = th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    T = th.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T


def _quat_wxyz(R: th.Tensor) -> th.Tensor:
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4) wxyz (robust branch per row)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = th.zeros(R.shape[0], 4, device=R.device)
    c0 = t > 0
    s = th.sqrt(th.clamp(t + 1.0, min=1e-12)) * 2
    q[c0] = th.stack([0.25 * s, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)[c0]
    c1 = (~c0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = th.sqrt(th.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2
    q[c1] = th.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, 0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)[c1]
    c2 = (~c0) & (~c1) & (m[:, 1, 1] > m[:, 2, 2])
    s = th.sqrt(th.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-12)) * 2
    q[c2] = th.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)[c2]
    c3 = (~c0) & (~c1) & (~c2)
    s = th.sqrt(th.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-12)) * 2
    q[c3] = th.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s], 1)[c3]
    return q / th.linalg.norm(q, dim=1, keepdim=True)


def _solve(world: SynthWorld, entry, T: th.Tensor, q_lock: th.Tensor, seed_active: Optional[th.Tensor] = None):
    """Batched IK of eef poses ``T`` (N, 4, 4; world = root frame) with the locked joints of ``q_lock``: ``(success (N, S), q (N, S, dof))``
    - every seed's solution (the caller keeps the first that passes its own checks). ``seed_active`` (N, dof): one seed per problem (the
    pre-grasp solution of a grasp), the rest of the ``seeds`` random. Poses go to cuRobo as tensors (no per-pose Python), in chunks of the
    entry's batch size, the last one padded."""
    be = world.backend
    ik = entry.rsc
    B = world.cfg.batch
    be._refresh_locked_entry(entry, q_lock)
    frames = list(ik.kinematics.tool_frames)
    dof = len(ik.joint_names)
    n = T.shape[0]
    S_ = world.cfg.seeds
    succ = th.zeros(n, S_, dtype=th.bool, device=world.dev)
    qs = th.zeros(n, S_, dof, device=world.dev)
    cur_full = be._js_full(q_lock.unsqueeze(0).repeat(B, 1))
    cur = entry.fk.get_active_js(cur_full)
    from ispatialgym.oracle.curobo_planner import CuPose, GoalToolPose

    for s0 in range(0, n, B):
        Tc = T[s0:s0 + B].to(world.dev)
        b = Tc.shape[0]
        if b < B:
            Tc = th.cat([Tc, Tc[-1:].repeat(B - b, 1, 1)])
        pose = CuPose(position=Tc[:, :3, 3].contiguous(), quaternion=_quat_wxyz(Tc[:, :3, :3]).contiguous())
        goal = GoalToolPose.from_poses({frames[0]: pose}, ordered_tool_frames=frames, num_goalset=1)
        if seed_active is not None:
            sd = seed_active[s0:s0 + B].to(world.dev)
            if sd.shape[0] < B:
                sd = th.cat([sd, sd[-1:].repeat(B - sd.shape[0], 1)])
            seed = sd.view(B, 1, dof).clone()
        else:
            seed = cur.position.view(B, 1, -1).clone()
        res = ik.solve_pose(goal, current_state=cur, seed_config=seed, return_seeds=S_)
        succ[s0:s0 + b] = res.success.reshape(B, -1)[:b, :S_]
        qs[s0:s0 + b] = res.solution.reshape(B, -1, dof)[:b, :S_]
    return succ, qs


# ------------------------------------------------------------------------------------------------ the computation


def slab(h: float, d_e: float, cfg: MapConfig) -> CP.Support:
    return CP.Support.box([d_e + cfg.slab_depth_m / 2, 0.0, h - cfg.slab_thickness_m / 2],
                          [cfg.slab_depth_m / 2, cfg.slab_long_m / 2, cfg.slab_thickness_m / 2], name="synthetic_slab")


def floor_plane() -> CP.Support:
    """The support of a floor map: the floor the robot stands on (solid below z = 0 of the map frame; no edge, unbounded)."""
    return CP.Support.plane(FLOOR_HEIGHT, name="floor")


def map_support(kind: str, h: float, d_e: float, cfg: MapConfig) -> CP.Support:
    """The support of a map kind at (height, edge distance): the floor plane (``floor``) or the synthetic slab."""
    return floor_plane() if kind == "floor" else slab(h, d_e, cfg)


@contextmanager
def ground_slab_off(world):
    """i7-floorfix 2026-11-07: the cuRobo ground slab (``R1ProCuRobo.GROUND_SLAB_NAME``) out of the collision world for the block - the IK of a
    floor map under ``floor_fingers`` "policy" (the slab is the support there; the backend has no per-pair tolerance, so the support leaves the
    IK and its contacts are checked explicitly, as a slab map's slab never enters it). Yields whether a slab was switched off."""
    be = world.backend
    cub = getattr(getattr(getattr(be, "checker", None), "data", None), "cuboids", None)
    name = getattr(be, "GROUND_SLAB_NAME", "__ground__")
    off = False
    try:
        if cub is not None and name in cub.names[0]:
            cub.set_enabled(name, False)
            off = True
        yield off
    finally:
        if off:
            cub.set_enabled(name, True)


def _yaw_R(yaws: Sequence[float]) -> np.ndarray:
    return np.stack([np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]) for a in yaws])


def compute_map(world: SynthWorld, lib: S.ModelLibrary, rest_id: str, height: float, anchor: Dict[str, Any], geom=None,
                cfg: Optional[MapConfig] = None, policy: CP.ContactPolicy = CP.DEFAULT, kind: str = "tabletop") -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One map: (meta, arrays). ``geom`` (the object's collision pieces, :mod:`object_geometry`) for the arm-vs-object check.
    ``kind="brace"``: the contact classes of :func:`brace_classes` instead of the library grasps (closed fingers, the pads on the
    object's faces; single-arm proposals only - simultaneous brace / manipulation and the arms' mutual collisions are checked later).
    ``kind="floor"`` (i7-floor 2026-11-07): the library grasps of the object on the FLOOR the robot stands on - ``height`` 0, the support
    the floor plane (:func:`floor_plane`, no edge: ``cfg`` from :func:`floor_config`, one edge distance 0.0, cells from FLOOR_X_MIN_M),
    the trunks of ``trunk_grid(0)`` (the low-top tilt and the ground trunks), the policy's floor mode (contact_policy rule 5): the fingers'
    floor allowance and the robot's ground-contact links (wheels, base) scoped out of the sphere check. i7-floorfix 2026-11-07
    (``cfg.floor_fingers``): "policy" - the pre-grasp and grasp IK without the cuRobo ground slab (:func:`ground_slab_off`; the fingers are
    then judged by their hulls' floor allowance only, every other link but the ground-contact ones by the sphere check against the floor
    plane at the pre-grasp, the grasp and the approach configurations; a problem's pre-grasp is its first IK seed that passes that check);
    "spheres" - the former rule (the slab in the IK for every sphere: a padded finger sphere touching it failed the IK)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    rest = lib.rests[rest_id]
    if kind not in KINDS:
        raise ValueError(f"unknown map kind {kind!r} (one of {KINDS})")
    floor = kind == "floor"
    if floor and (abs(float(height) - FLOOR_HEIGHT) > 1e-9 or tuple(float(d) for d in cfg.edge_distances) != FLOOR_EDGE_DISTANCES or cfg.x_min_m is None):
        raise ValueError(f"a floor map is computed at height {FLOOR_HEIGHT} with floor_config() (got height {height}, edge distances "
                         f"{cfg.edge_distances}, x_min_m {cfg.x_min_m})")
    if floor and cfg.floor_fingers not in FLOOR_FINGER_RULES:
        raise ValueError(f"a floor map's floor_fingers is one of {FLOOR_FINGER_RULES} (got {cfg.floor_fingers!r})")
    if not floor and cfg.floor_fingers is not None:
        raise ValueError(f"floor_fingers is a floor map's field (kind {kind!r}, got {cfg.floor_fingers!r})")
    # i7-floorfix 2026-11-07: the support (the floor) out of the pre-grasp / grasp IK - the fingers then meet it only through the hull rule
    ik_without_ground = floor and cfg.floor_fingers == "policy"
    # the robot's ground-contact links (wheels, base body): they stand on the floor - not a grasp contact (rule 5; floor maps only)
    ground_links = sorted(getattr(world.backend, "ground_contact_links", ()) or ()) if floor else []
    grasps = brace_classes() if kind == "brace" else grasp_set(lib, rest_id, cfg)
    if kind == "brace" and geom is None:
        raise ValueError("a brace map needs the object's collision geometry")
    cells = lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY, nG = len(cells), len(yaws), len(grasps)
    trunks = trunk_grid(height)
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32)}
    meta_trunks: List[Dict[str, Any]] = []
    if nG == 0:
        raise ValueError(f"{lib.path}: no validated grasps on rest {rest_id}")
    # -- object poses and eef targets (cells x yaws x grasps), world frame
    t0 = time.time()
    Ry = _yaw_R(yaws)
    Rr = S.quat_to_mat(rest.quat_xyzw)
    To = np.zeros((nY, 4, 4))
    To[:, :3, :3] = Ry @ Rr
    To[:, 3, 3] = 1.0
    To[:, 2, 3] = height + rest.h_root
    if kind == "brace":
        TE, contact_pts = brace_targets(np.asarray(geom.union.vertices) @ Rr.T, yaws, height, rest.h_root, grasps)
    else:
        T_og = np.stack([g["T"] for g in grasps])  # (G, 4, 4)
        TE = np.einsum("yij,gjk->ygik", To, T_og)  # object at the origin xy: (Y, G, 4, 4)
    TE_t = th.as_tensor(TE, dtype=th.float32, device=dev).reshape(1, nY, nG, 4, 4).repeat(nC, 1, 1, 1, 1)
    TE_t[..., 0, 3] += th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1)
    TE_t[..., 1, 3] += th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1)
    TE_flat = TE_t.reshape(-1, 4, 4)
    TP_flat = TE_flat.clone()
    TP_flat[:, :3, 3] -= C.PRE_DIST * TP_flat[:, :3, 2]
    N = TE_flat.shape[0]
    # footprint of the rest at each yaw: the object's xy reach toward the robot (support fit at d_e)
    if geom is not None:
        V = np.asarray(geom.union.vertices) @ Rr.T
        reach_back = np.array([float((V @ R.T)[:, 0].min()) for R in Ry])  # most negative x of the footprint at each yaw
    else:
        reach_back = np.zeros(nY)
    tim["targets_s"] = round(time.time() - t0, 3)
    # -- the hand vs the slab per d_e (IK-independent): pre-grasp, approach, grasp, closing sweep
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)
    hand_ok: Dict[float, th.Tensor] = {}
    path = CP.approach_path(TP_flat, TE_flat, policy.approach_step_m * 2)
    # of-v2maps 2026-11-07 (prompt I item 3; the isg1020 spec P1 §2): each grasp's fingers at ITS pre-shape on the approach and swept to
    # ITS contact opening (cfg.preshape "grasp"); "max" = the former batch rule (max pre-shape, min contact) for every grasp of the entry
    pq_g = [float(g["preshape_q"]) for g in grasps]
    cq_g = [float(g["contact_q"]) for g in grasps]
    if cfg.preshape == "max":
        pq_g, cq_g = [max(pq_g)] * nG, [min(cq_g)] * nG
    g_of = th.arange(N, device=dev) % nG  # problems are (cell, yaw, grasp) row-major
    pq_dev = th.as_tensor(pq_g, dtype=th.float32, device=dev)
    cq_dev = th.as_tensor(cq_g, dtype=th.float32, device=dev)

    def hand_check_at(T, sb_, fq_t):
        """``hc.check`` with one finger opening per pose (grouped by value)."""
        vals = th.round(fq_t * 1e5) / 1e5
        out = th.ones(T.shape[0], dtype=th.bool, device=dev)
        for v in th.unique(vals).tolist():
            m = vals == v
            out[m] = hc.check(T[m], sb_, (float(v), float(v)), floor=floor)[0]
        return out

    def hand_check(T, q_of_grasp):
        groups: Dict[float, List[int]] = {}
        for gi, v in enumerate(q_of_grasp):
            groups.setdefault(round(v, 5), []).append(gi)
        if len(groups) == 1:
            v = next(iter(groups))
            return hc.check(T, sb, (v, v), floor=floor)[0]
        o_all = th.ones(N, dtype=th.bool, device=dev)
        for v, gis in groups.items():
            sel = th.isin(g_of, th.as_tensor(gis, device=dev))
            o_all[sel] = hc.check(T[sel], sb, (v, v), floor=floor)[0]
        return o_all

    def spheres_ok(sph, ln, sb_):
        """``contact_policy.sphere_support_ok`` against the map's support (floor: the ground-contact links scoped out)."""
        return CP.sphere_support_ok(sph, ln, sb_, cfg.arm, policy, ground_contact_links=ground_links, floor_support=floor)

    for d_e in cfg.edge_distances:
        sb = map_support(kind, height, d_e, cfg)
        ok = th.ones(N, dtype=th.bool, device=dev)
        for k in range(path.shape[0]):
            ok &= hand_check(path[k], pq_g)
        for j in range(1, 3):
            ok &= hand_check(TE_flat, [p + (c - p) * j / 2 for p, c in zip(pq_g, cq_g)])
        hand_ok[d_e] = ok
    del path
    if th.cuda.is_available():
        th.cuda.empty_cache()
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    # off-support per (d_e, cell, yaw)
    onsup = {}
    for d_e in cfg.edge_distances:
        m = support_mask(cells, reach_back, d_e, kind)  # (C, Y)
        onsup[d_e] = th.as_tensor(np.repeat(m[:, :, None], nG, axis=2).reshape(-1), device=dev)
    # -- arm-vs-object planes (object at each cell / yaw): world-frame planes per (cell, yaw) are built on the fly
    pieces = []
    if geom is not None:
        for pc in geom.pieces:
            pieces.append((th.as_tensor(pc.face_normals, dtype=th.float32, device=dev), th.as_tensor(np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0]), dtype=th.float32, device=dev)))
    hand_links = set(policy.finger_links(cfg.arm)) | set(policy.hand_rigid_links(cfg.arm))
    entry = world.entry()
    links = world.link_of_sphere()
    arm_mask = th.tensor([(l not in hand_links) and bool(l) and (l.startswith(f"{cfg.arm}_arm")) for l in links], device=dev)
    # object frame of each problem (for the arm-vs-object test)
    To_t = th.as_tensor(To, dtype=th.float32, device=dev)
    obj_idx_y = th.arange(nY, device=dev).view(1, nY, 1).repeat(nC, 1, nG).reshape(-1)
    obj_xy = th.as_tensor(cells, dtype=th.float32, device=dev).view(nC, 1, 1, 2).repeat(1, nY, nG, 1).reshape(-1, 2)
    t_ik = t_chk = 0.0
    n_ik = 0
    def solve_group(rec: Dict[str, Any], key: str, status: Dict[float, th.Tensor], q_lock: th.Tensor, des: Sequence[float], in_reach: th.Tensor) -> None:
        """The trunk's IK pass and per-d_e checks with the idle-arm posture ``q_lock`` for the edge distances ``des`` (i7-unblock 2026-11-07:
        a trunk whose postures are free of the slab at different d_e (``trunk_eligibility`` "per_d_e") runs one pass per posture group; a
        trunk with one posture free at every d_e runs exactly the former single pass over all of them)."""
        nonlocal t_ik, t_chk, n_ik
        any_hand = th.zeros(N, dtype=th.bool, device=dev)
        for d_e in des:
            any_hand |= hand_ok[d_e] & onsup[d_e]
        cand = in_reach & any_hand
        idx = th.nonzero(cand).flatten()
        rec["n_problems"] = rec.get("n_problems", 0) + int(idx.numel())
        # accelerated mode: one round of the robot-facing yaws first; cells with no pre-grasp in it skip the rest
        if cfg.mode == "accelerated" and idx.numel() > 0:
            yaw_of = obj_idx_y[idx]
            first = (yaw_of == (nY // 2)) | (yaw_of == (nY // 2 - 1)) | (yaw_of == (nY // 2 + 1) % nY)
            order = th.cat([idx[first], idx[~first]])
        else:
            order = idx
        ta = time.time()
        with (ground_slab_off(world) if ik_without_ground else nullcontext()):
            succ_pre, q_pre = _solve(world, entry, TP_flat[order], q_lock)
        n_ik += int(order.numel())
        t_ik += time.time() - ta
        pre_any = succ_pre.any(dim=1)
        if cfg.mode == "accelerated" and order.numel() > 0:
            cell_of = order // (nY * nG)
            first_n = int(first.sum())
            hit = th.zeros(nC, dtype=th.bool, device=dev)
            hit.index_put_((cell_of[:first_n][pre_any[:first_n]],), th.ones(int(pre_any[:first_n].sum()), dtype=th.bool, device=dev))
            skip = (~hit[cell_of]) & (th.arange(order.numel(), device=dev) >= first_n)
        else:
            skip = th.zeros(order.numel(), dtype=th.bool, device=dev)
        # the grasp IK seeded from the first successful pre-grasp seed of each problem
        first_seed = th.argmax(succ_pre.int(), dim=1)
        if ik_without_ground and order.numel() > 0:
            # i7-floorfix: the IK never saw the floor - the pre-grasp seed of a problem is its first successful one whose spheres (every
            # link but the fingers and the ground-contact ones) clear the floor plane; the first successful one when none does (it then
            # fails the sphere check below as slab_spheres)
            tb = time.time()
            sb0 = map_support(kind, height, cfg.edge_distances[0], cfg)
            clear = th.zeros_like(succ_pre)
            for s in range(succ_pre.shape[1]):
                rows = th.nonzero(succ_pre[:, s]).flatten()
                for c0 in range(0, rows.numel(), 8192):
                    rc = rows[c0:c0 + 8192]
                    o, _ = spheres_ok(world.spheres(world.full_q(entry, q_pre[rc, s], q_lock)), links, sb0)
                    clear[rc, s] = o.to(dev)
            good_pre = succ_pre & clear
            first_seed = th.where(good_pre.any(dim=1), th.argmax(good_pre.int(), dim=1), first_seed)
            t_chk += time.time() - tb
        q_pre1 = q_pre[th.arange(order.numel(), device=dev), first_seed]
        sel = th.nonzero(pre_any & ~skip).flatten()
        ta = time.time()
        with (ground_slab_off(world) if ik_without_ground else nullcontext()):
            succ_g, q_g = _solve(world, entry, TE_flat[order[sel]], q_lock, seed_active=q_pre1[sel]) if sel.numel() else (th.zeros(0, cfg.seeds, dtype=th.bool, device=dev), None)
        n_ik += int(sel.numel())
        t_ik += time.time() - ta
        tb = time.time()
        # per d_e post checks over the seeds of each grasp solution (first passing seed counts)
        feas = {d_e: th.zeros(sel.numel(), dtype=th.bool, device=dev) for d_e in des}
        why = {d_e: th.full((sel.numel(),), STATUS["ik_grasp"], dtype=th.uint8, device=dev) for d_e in des}
        if sel.numel():
            qp = q_pre1[sel]
            for s in range(cfg.seeds):
                ok_s = succ_g[:, s]
                if not bool(ok_s.any()):
                    continue
                qg = q_g[:, s]
                jump = (qg - qp).abs().max(dim=1).values > cfg.branch_max_rad
                cand_s = ok_s & ~jump
                for d_e in des:
                    cur = cand_s & ~feas[d_e]
                    why[d_e] = th.where(ok_s & jump & ~feas[d_e], th.full_like(why[d_e], STATUS["branch_jump"]), why[d_e])
                    if not bool(cur.any()):
                        continue
                    ii = th.nonzero(cur).flatten()
                    good = th.ones(ii.numel(), dtype=th.bool, device=dev)
                    sb = map_support(kind, height, d_e, cfg)
                    for t in [0.0] + [k / (cfg.approach_configs + 1) for k in range(1, cfg.approach_configs + 1)] + [1.0]:
                        for c0 in range(0, ii.numel(), 8192):
                            iic = ii[c0:c0 + 8192]
                            qq = qp[iic] * (1 - t) + qg[iic] * t
                            sp = world.spheres(world.full_q(entry, qq, q_lock))
                            o, _ = spheres_ok(sp, links, sb)
                            good[c0:c0 + 8192] &= o.to(dev)
                    w_slab = ~good
                    # the hand hulls at the FK eef of the grasp and the pre-grasp (IK tolerance: 5 mm / 0.05 rad)
                    # ... each problem's fingers at ITS grasp's openings (cfg.preshape; "max": the batch values as before)
                    gq = g_of[order[sel[ii]]]
                    pq_t, cq_t = pq_dev[gq], cq_dev[gq]
                    for qq, fq_t in ((qg[ii], 0.5 * (pq_t + cq_t)), (qp[ii], pq_t)):
                        good &= hand_check_at(world.eef(world.full_q(entry, qq, q_lock)), sb, fq_t)
                    w_hand = ~good & ~w_slab
                    # the arm (not the hand) clear of the object at the cell
                    if pieces:
                        # chunked: (problems x arm spheres x faces of a piece) does not fit a 24 GB card for large grasp sets in one go
                        o = th.ones(ii.numel(), dtype=th.bool, device=dev)
                        for c0 in range(0, ii.numel(), 2048):
                            iic = ii[c0:c0 + 2048]
                            sp = world.spheres(world.full_q(entry, qg[iic], q_lock))[:, arm_mask]
                            oi = order[sel[iic]]
                            Ro = To_t[obj_idx_y[oi], :3, :3]
                            to = th.cat([obj_xy[oi], To_t[obj_idx_y[oi], 2, 3:4]], 1)
                            Po = th.einsum("nji,nsj->nsi", Ro, sp[..., :3] - to[:, None])
                            deep = th.full(Po.shape[:2], float("inf"), device=dev)
                            for nrm, dd in pieces:
                                deep = th.minimum(deep, (Po @ nrm.T - dd).max(dim=-1).values)
                            o[c0:c0 + 2048] = (((deep - sp[..., 3]) >= 0.0) | (sp[..., 3] <= 0)).all(dim=1)
                        w_obj = good & ~o
                        good &= o
                    else:
                        w_obj = th.zeros_like(good)
                    feas[d_e][ii] |= good
                    w = why[d_e][ii]
                    w = th.where(w_slab, th.full_like(w, STATUS["slab_spheres"]), w)
                    w = th.where(w_hand, th.full_like(w, STATUS["hand_policy"]), w)
                    w = th.where(w_obj, th.full_like(w, STATUS["object_arm"]), w)
                    why[d_e][ii] = w
        t_chk += time.time() - tb
        for d_e in des:
            st = status[d_e]
            st.fill_(STATUS["unevaluated"])
            st[~in_reach] = STATUS["reach_bound"]
            st[in_reach & ~hand_ok[d_e]] = STATUS["hand_policy"]
            st[~onsup[d_e]] = STATUS["off_support"]
            ev = order
            vals = th.full((ev.numel(),), STATUS["ik_pre"], dtype=th.uint8, device=dev)
            vals[skip] = STATUS["skipped"]
            if sel.numel():
                v2 = th.where(feas[d_e], th.full_like(why[d_e], STATUS["feasible"]), why[d_e])
                vals[sel] = v2
            ok_here = hand_ok[d_e][ev] & onsup[d_e][ev]
            st[ev] = th.where(ok_here, vals, st[ev])
            arrays[f"s{d_e:.2f}_{key}"] = st.view(nC, nY, nG).cpu().numpy()
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int((st == STATUS["feasible"]).sum())
            rec.setdefault("n_skipped", {})[f"{d_e:.2f}"] = int((st == STATUS["skipped"]).sum())

    for tr, ti in trunks:
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        status = {d_e: th.zeros(N, dtype=th.uint8, device=dev) for d_e in cfg.edge_distances}
        # eligibility (i7-unblock 2026-11-07, owner decision: the former rule was a bug): the first idle-arm posture free of self-collision and of
        # the slab at every d_e (support factor, shared by objects); "per_d_e" (MapConfig.trunk_eligibility, the default of new builds) keeps
        # that choice when it exists and otherwise judges each d_e alone - the first posture free there - so the trunk is ineligible only at
        # the d_e where every posture collides (nextto_04: t0.95 collided at d_e 0.35 only, and the whole trunk was dropped)
        q_of: Dict[str, th.Tensor] = {}

        def free_of(p: str, tr=tr, ti=ti) -> Optional[List[bool]]:
            q = world.q_config(tr, ti, p)
            if bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                return None
            q_of[p] = q
            sph, ln = world.backend.robot_spheres_world(q.unsqueeze(0))
            return [bool(spheres_ok(sph, ln, map_support(kind, height, d_e, cfg))[0][0]) for d_e in cfg.edge_distances]

        groups, collide = trunk_posture_groups(cfg.postures, len(cfg.edge_distances), free_of, trunk_rule(cfg))
        if collide:
            rec["slab_collide_by_posture"] = collide
        rec["posture"] = groups[0][0] if groups else None
        if not groups:
            rec["eligible"] = False
            rec["reason"] = "every idle-arm posture collides (self or the slab at some d_e)" if trunk_rule(cfg) == "all_d_e" else "every idle-arm posture collides (self or the slab) at every d_e"
            for d_e in cfg.edge_distances:
                status[d_e].fill_(STATUS["trunk_ineligible"])
                arrays[f"s{d_e:.2f}_{key}"] = status[d_e].view(nC, nY, nG).cpu().numpy()
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        mark_partial(rec, groups, cfg.edge_distances, status, arrays, key, (nC, nY, nG))
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        reach = L + cfg.reach_margin_m
        in_reach = (th.linalg.norm(TE_flat[:, :3, 3] - sh_t, dim=1) <= reach) & (th.linalg.norm(TP_flat[:, :3, 3] - sh_t, dim=1) <= reach)
        rec.update({"shoulder": [round(float(v), 4) for v in sh], "arm_len": round(L, 4)})
        for p, ks in groups:
            solve_group(rec, key, status, q_of[p], [cfg.edge_distances[k] for k in ks], in_reach)
        meta_trunks.append(rec)
    tim.update({"ik_s": round(t_ik, 2), "checks_s": round(t_chk, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)})
    meta = {"schema": SCHEMA, "kind": kind, "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": lib.identity(), "object_key": lib.identity_key(), "rest_id": rest_id,
            "height": {"value": height, "anchor_key": anchor.get("key"), "anchor_surfaces": anchor.get("surfaces", [])[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "policy": policy.meta(cfg.arm),
            "slab": None if floor else {"depth_m": cfg.slab_depth_m, "long_m": cfg.slab_long_m, "thickness_m": cfg.slab_thickness_m},
            "ik": {"seeds": cfg.seeds, "position_tolerance": world.backend.position_tolerance, "orientation_tolerance": world.backend.orientation_tolerance,
                   "collision_world": ("self-collision only: the floor cuboid (the support itself) out of the pre-grasp / grasp IK (floor_fingers "
                                       "'policy': the fingers judged by the floor allowance on their hulls, every other link but the ground-contact ones "
                                       "by its spheres against the floor plane; the object checked explicitly)" if ik_without_ground else
                                       "floor cuboid (the support itself: kept for the IK, floor_fingers 'spheres') + self-collision (the floor plane's "
                                       "contact policy and the object checked explicitly)" if floor else
                                       "floor cuboid + self-collision (slab and object checked explicitly)"), "pre_dist": C.PRE_DIST},
            "approach_check": f"hand hulls at the pre-grasp, every {policy.approach_step_m * 2:.3f} m of the straight approach and 2 sweep openings; "
                              f"whole-robot spheres at {cfg.approach_configs} joint-interpolated configurations",
            "screening": cfg.mode, "exhaustive": cfg.mode == "full",
            "grasps": ([{k: v for k, v in g.items() if k != "T"} | {"T_object_eef": S.mat_pose(g["T"])} for g in grasps] if kind != "brace" else
                       [dict(g, contact_point_world_yaw0=[round(float(v), 5) for v in contact_pts[0, i]],
                             contact_to_eef_yaw0=S.mat_pose(np.linalg.inv(S.pose_mat(contact_pts[0, i], [0, 0, 0, 1.0])) @ TE[0, i])) for i, g in enumerate(grasps)]),
            "brace": ({"fractions": list(BRACE_FRACTIONS), "pitches_deg": list(BRACE_PITCHES_DEG), "sides": list(BRACE_SIDES), "fingers": "closed",
                       "frames": "approach / closing / contact point in the map frame (base on the floor); fingertip_eef: fingertip plane 5 mm before the point",
                       "note": "single-arm feasibility proposal; simultaneous brace + manipulation and inter-arm collisions are validated later"}
                      if kind == "brace" else None),
            "trunks": meta_trunks, "status_codes": STATUS, "robot": robot_digest(), "timing": tim, "created": S.now()}
    if floor:
        # i7-floor 2026-11-07: the support plane is part of the identity (map_key, ReachMap.find); slab maps carry no field (their keys stay)
        meta["plane"] = PLANE_OF_KIND["floor"]
        meta["floor"] = {"support": "the floor plane the robot stands on (z = 0 of the map frame, unbounded: no edge)", "edge_distance": 0.0,
                         "x_min_m": cfg.x_min_m, "y_half_m": cfg.y_half_m, "x_max_m": cfg.x_max_m, "ground_contact_links": ground_links,
                         "finger_allowance_m": policy.floor_finger_max_penetration_m, "fingers": cfg.floor_fingers,
                         "trunks": "reach_bands.trunk_grid(0.0): TRANSLATES x (TILTS + LOW_TOP_TILT) + GROUND_TRUNKS"}
    meta["key"] = map_key(meta)
    return meta, arrays


@functools.lru_cache(maxsize=1)
def robot_digest() -> str:
    from ispatialgym.oracle import r1pro_model as M

    h = hashlib.sha1()
    for p in (M.source_urdf_path(), M.source_curobo_yaml_path()):
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(json.dumps(M.EXTRA_SELF_COLLISION_IGNORE_PAIRS).encode())
    return h.hexdigest()[:12]


def map_key(meta: Dict[str, Any]) -> str:
    ident = {k: meta[k] for k in ("schema", "kind", "arm", "object_key", "rest_id", "edge_distances", "config", "policy_key", "robot")}
    ident["brace"] = meta.get("brace")
    ident["height"] = meta["height"]["value"]
    if meta.get("plane") is not None:  # i7-floor 2026-11-07: a floor map's plane (slab maps keep their keys)
        ident["plane"] = meta["plane"]
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def save(out_dir: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"map__{meta['object']['category']}__{meta['object']['model']}__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    S.save_json(base + ".json", meta)
    return base + ".json"


# ------------------------------------------------------------------------------------------------ reading maps


class ReachMap:
    """One map (``.json`` + ``.npz``). Status queries per (edge distance, trunk, cell); cell weights for sampling."""

    def __init__(self, path_json: str):
        self.path = path_json
        with open(path_json) as f:
            self.meta = json.load(f)
        if self.meta.get("schema") != SCHEMA:
            raise S.IncompatibleRecord(f"{path_json}: schema {self.meta.get('schema')!r} is not {SCHEMA}")
        z = np.load(path_json[:-5] + ".npz")
        self.arrays = {k: z[k] for k in z.files}
        self.cells = self.arrays["cells"]
        self.yaws = self.arrays["yaws"]

    @staticmethod
    def find(maps_root: str, lib: S.ModelLibrary, height: float, *, rest_id: str, arm: str = "right", policy_key: Optional[str] = None,
             anchors=None, kind: str = "tabletop", robot: Optional[str] = None, config_require: Optional[Dict[str, Any]] = None) -> Optional["ReachMap"]:
        """The map of this library entry / rest / arm at the anchor that covers ``height`` (``height_anchors.AnchorSet.lookup``: exact,
        or within the anchor's clipped range); None when uncovered or when no compatible map exists - never a band, never a map of other
        metadata (object identity, rest, policy, schema are all checked). ``kind="floor"`` (i7-floor 2026-11-07): ``height`` 0.0 and
        ``anchors=None`` (the anchor set has no floor height); the map's kind AND support plane (meta ``plane``: "floor"; a slab map's
        is "slab") must be the kind's - a tabletop map at h 0 never serves a floor lookup, nor a floor map a tabletop one."""
        want_plane = PLANE_OF_KIND.get(kind, "slab")
        target = height
        if anchors is not None:
            # of-v2maps 2026-11-06: the lookup's kind (exact / approx) used to overwrite the MAP kind argument ("tabletop"), so with an anchor
            # file every map failed the kind test below - no sampler map lookup could hit (v2smoke1: "reach map miss" at an existing map's height)
            a, _how = anchors.lookup(height)
            if a is None:
                return None
            target = a["height"]
        want = lib.identity_key()
        fallback: Optional[str] = None
        for fn in sorted(os.listdir(maps_root)) if os.path.isdir(maps_root) else []:
            if not fn.endswith(".json") or not fn.startswith("map__"):
                continue
            p = os.path.join(maps_root, fn)
            try:
                with open(p) as f:
                    m = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if m.get("schema") != SCHEMA or m.get("kind", "tabletop") != kind or m.get("object_key") != want or m.get("rest_id") != rest_id or m.get("arm") != arm:
                continue
            if map_plane(m) != want_plane:  # i7-floor 2026-11-07: the support plane the map's cells stand on
                continue
            if policy_key is not None and m.get("policy_key") != policy_key:
                continue
            if robot is not None and m.get("robot") != robot:  # of-v2maps 2026-11-07 (P2 §11): a map of another robot model does not serve
                continue
            if config_require and any(config_value(m, k) != v for k, v in config_require.items()):
                continue  # ... nor one computed under another configuration (e.g. preshape "max")
            if abs(float(m["height"]["value"]) - target) > 1e-6:
                continue
            if config_value(m, "trunk_eligibility") != "per_d_e" and not (config_require and "trunk_eligibility" in config_require):
                # i7-unblock 2026-11-07: both trunk eligibility rules serve; a root holding a map of each for this lookup serves the per_d_e
                # one (the corrected rule) - the first former-rule match is kept in case none follows
                fallback = fallback or p
                continue
            return ReachMap(p)
        return ReachMap(fallback) if fallback else None

    def trunks(self) -> List[Tuple[float, float]]:
        return [(t["translate"], t["tilt"]) for t in self.meta["trunks"] if t.get("eligible")]

    def status(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        d = min(self.meta["edge_distances"], key=lambda v: abs(v - edge_distance))
        if abs(d - edge_distance) > 1e-6:
            raise KeyError(f"edge distance {edge_distance} is not in the map ({self.meta['edge_distances']})")
        return self.arrays[f"s{d:.2f}_t{translate:.2f}_k{tilt:+.1f}"]

    def feasible_cells(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        st = self.status(edge_distance, translate, tilt)
        return self.cells[(st == STATUS["feasible"]).any(axis=(1, 2))]

    def pairs(self, edge_distance: float, translate: float, tilt: float, cell_index: int) -> List[Tuple[int, int]]:
        st = self.status(edge_distance, translate, tilt)[cell_index]
        return [tuple(map(int, v)) for v in np.argwhere(st == STATUS["feasible"])]

    def cell_weights(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        """Plausibility per cell, normalised for discretisation: a feasible (yaw, grasp) adds 1 / n_yaws of the yaw axis times the grasp's
        share of its family (a family sampled at k points contributes its measure once, whatever k), twins as their own wrist choice."""
        st = self.status(edge_distance, translate, tilt) == STATUS["feasible"]
        g = self.meta["grasps"]
        fam_n: Dict[Tuple[str, bool], int] = {}
        for x in g:
            k = (x["base"], x["twin"])
            fam_n[k] = fam_n.get(k, 0) + 1
        w_g = np.array([1.0 / fam_n[(x["base"], x["twin"])] for x in g])
        return (st * w_g[None, None, :]).sum(axis=(1, 2)) / len(self.yaws)


# ------------------------------------------------------------------------------------------------ generic pinch maps (prompt I item 7)
#
# i7-generic 2026-11-07: the GENERIC-PINCH kind - the scene-independent replacement of the legacy per-scene ``--generic-pinch`` bands
# (``reach_bands.BandComputer`` with ``grasp_sets=None``: blocks, axial push targets; alignment-axial_01..05, block_design_01). The legacy
# band's pose set does not depend on the object, so a generic map has no library entry: one map per (height, arm, pose set, policy, robot,
# config), written under ``<maps root>/_generic/``. Kept apart from :func:`compute_map` (its own functions, its own key / save / find).


GENERIC_KIND = "generic"
#: the maps-root directory of the generic pseudo-entry (the maps of one root share it; the pose set is in each map's identity)
GENERIC_DIR = "_generic"
#: eef height above the support (m): the legacy ``max(0.6 h, 0.03)`` (``reach_bands`` generic pinch, ``tabletop.compute_bands``) at the
#: objects of the six aliases (h <= 5 cm: the axial targets, the 2.5 cm blocks)
GENERIC_DZ_M = 0.03
#: the finger opening rule of a generic pinch (``fingers_q`` default): the support policy measures a grasp without a pinch width at the
#: configuration's own opening (``tabletop.policy_opening_of`` -> None; ``RobotKit.q_config`` locks the fingers at ``FINGER_OPEN``), which
#: is also the legacy band's locked opening (``BandComputer.fingers`` 0.05) - fully open, the widest (worst-case) hand against the slab
GENERIC_FINGERS_RULE = "configuration's own opening (no pinch width: tabletop.policy_opening_of -> None; RobotKit.q_config FINGER_OPEN)"


def generic_dz(object_height: float) -> float:
    """The legacy eef height above the support of a generic pinch for an object ``object_height`` tall: ``max(0.6 h, 0.03)``."""
    return round(max(0.6 * float(object_height), GENERIC_DZ_M), 4)


def generic_pose_set(dz_m: float = GENERIC_DZ_M, fingers_q: Optional[float] = None) -> Dict[str, Any]:
    """The pose set of a generic map (part of its identity): the legacy generic classes (``reach_bands.GENERIC_CLASSES``: approach pitched
    ``TILTS_GENERIC`` deg from straight down toward the robot's forward = +x of the map frame, wrist yaw ``YAWS_GENERIC_DEG`` in the base
    frame; tilt-major, yaw-minor = ``RobotKit.grasp_candidates_at``'s order = the bit order of the class mask), the eef at the cell, ``dz_m``
    above the slab top, the pre-grasp ``pre_dist_m`` back along the approach (eef +z), the fingers at ``fingers_q`` per finger (default
    :data:`GENERIC_FINGERS_RULE`) through the whole approach - no closing sweep (no pinch width: the exact check closes on the real
    object). The support is NOT ignored at the grasp (the legacy band ignored it): the contact policy applies against the slab."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import reach_bands as RB

    fq = float(C.FINGER_OPEN if fingers_q is None else fingers_q)
    return {"name": "generic_pinch", "classes": list(RB.GENERIC_CLASSES), "tilts_deg": [float(t) for t in RB.TILTS_GENERIC],
            "wrist_yaws_deg": [float(y) for y in RB.YAWS_GENERIC_DEG], "order": "tilt-major, wrist-yaw-minor (RobotKit.grasp_candidates_at)",
            "toward": [1.0, 0.0], "dz_m": round(float(dz_m), 4), "fingers_q": round(fq, 4),
            "fingers_rule": GENERIC_FINGERS_RULE if fingers_q is None else "explicit", "pre_dist_m": float(C.PRE_DIST),
            "closing_sweep": False, "support_at_grasp": "contact policy (not ignored)", "on_support": "cell x >= d_e (the eef point over the slab)"}


def generic_rotations(pose_set: Dict[str, Any]) -> np.ndarray:
    """(G, 3, 3) eef orientations of the pose set's classes in the map frame, by the legacy band's own pose function
    (``RobotKit.top_down_quat``: base yaw 0, forward +x)."""
    from ispatialgym.sampling.robot_utils import RobotKit

    out = [S.quat_to_mat([float(v) for v in RobotKit.top_down_quat(math.radians(float(y)), float(t), pose_set["toward"])])
           for t in pose_set["tilts_deg"] for y in pose_set["wrist_yaws_deg"]]
    if len(out) != len(pose_set["classes"]):
        raise ValueError(f"generic pose set: {len(out)} poses for {len(pose_set['classes'])} classes")
    return np.stack(out)


def generic_class_rows(pose_set: Dict[str, Any], R: np.ndarray) -> List[Dict[str, Any]]:
    """The map's class axis (``meta["grasps"]``, the rows ``ReachMap.cell_weights`` / ``pairs`` index): one row per class, each its own
    measure (``base`` = the class, no twins / families), the eef pose relative to the cell at the slab top (``T_cell_eef``)."""
    rows = []
    names = iter(pose_set["classes"])
    for t in pose_set["tilts_deg"]:
        for y in pose_set["wrist_yaws_deg"]:
            T = np.eye(4)
            T[:3, :3] = R[len(rows)]
            T[2, 3] = float(pose_set["dz_m"])
            n = next(names)
            rows.append({"name": n, "base": n, "twin": False, "family_id": None, "param": None, "tilt_deg": float(t), "wrist_yaw_deg": float(y),
                         "preshape_q": float(pose_set["fingers_q"]), "contact_q": float(pose_set["fingers_q"]), "physics": "generic_pinch",
                         "yaw_equivalent": False, "T_cell_eef": S.mat_pose(T)})
    return rows


def generic_map_key(meta: Dict[str, Any]) -> str:
    """The identity of a generic map: schema, kind, arm, edge distances, config, policy key, robot, height and the pose set."""
    ident = {k: meta.get(k) for k in ("schema", "kind", "arm", "edge_distances", "config", "policy_key", "robot", "pose_set")}
    ident["height"] = meta["height"]["value"]
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def compute_generic_map(world: SynthWorld, height: float, anchor: Dict[str, Any], cfg: Optional[MapConfig] = None,
                        policy: CP.ContactPolicy = CP.DEFAULT, pose_set: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One generic map: (meta, arrays), status per (edge distance, trunk, cell, class) as ``(cells, 1, classes)`` arrays (one "yaw": the
    classes carry the orientation). The checks of :func:`compute_map` without an object: the hand hulls against the slab at the pose set's
    fixed opening at the pre-grasp and along the straight approach to the grasp (per ``d_e``), the reach bound, batched IK of the pre-grasp
    and of the grasp seeded from it (branch jump), then per ``d_e`` every non-finger sphere at the pre-grasp, the grasp and the
    joint-interpolated configurations clear of the slab and the hand hulls at the FK eef poses. Exhaustive (``cfg.mode`` is not used)."""
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    ps = dict(pose_set or generic_pose_set())
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    cells = lattice(cfg)
    R = generic_rotations(ps)
    rows = generic_class_rows(ps, R)
    nC, nG = len(cells), len(rows)
    fq = float(ps["fingers_q"])
    trunks = trunk_grid(height)
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.zeros(1, dtype=np.float32)}
    meta_trunks: List[Dict[str, Any]] = []
    # -- eef targets (cells x classes, row-major), world = map frame
    t0 = time.time()
    TE_t = th.eye(4, device=dev).repeat(nC, nG, 1, 1)
    TE_t[:, :, :3, :3] = th.as_tensor(R, dtype=th.float32, device=dev).unsqueeze(0)
    TE_t[:, :, 0, 3] = th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1)
    TE_t[:, :, 1, 3] = th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1)
    TE_t[:, :, 2, 3] = float(height) + float(ps["dz_m"])
    TE_flat = TE_t.reshape(-1, 4, 4)
    TP_flat = TE_flat.clone()
    TP_flat[:, :3, 3] -= float(ps["pre_dist_m"]) * TP_flat[:, :3, 2]
    N = TE_flat.shape[0]
    tim["targets_s"] = round(time.time() - t0, 3)
    # -- the hand vs the slab per d_e (IK-independent): the fixed opening from the pre-grasp along the approach to the grasp (path ends incl.)
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)
    path = CP.approach_path(TP_flat, TE_flat, policy.approach_step_m * 2)
    hand_ok: Dict[float, th.Tensor] = {}
    for d_e in cfg.edge_distances:
        sb = slab(height, d_e, cfg)
        ok = th.ones(N, dtype=th.bool, device=dev)
        for k in range(path.shape[0]):
            ok &= hc.check(path[k], sb, (fq, fq))[0]
        hand_ok[d_e] = ok
    del path
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    # the eef point over the slab (object-independent: the sampler's erosion keeps the object's footprint on the real top)
    onsup = {d_e: th.as_tensor(np.repeat((cells[:, 0] >= d_e - 1e-9)[:, None], nG, axis=1).reshape(-1), device=dev) for d_e in cfg.edge_distances}
    entry = world.entry()
    links = world.link_of_sphere()
    t_ik = t_chk = 0.0
    n_ik = 0
    def solve_group(rec: Dict[str, Any], key: str, status: Dict[float, th.Tensor], q_lock: th.Tensor, des: Sequence[float], in_reach: th.Tensor) -> None:
        """:func:`compute_map`'s ``solve_group`` for the generic pose set: one IK pass with ``q_lock`` over the edge distances ``des``."""
        nonlocal t_ik, t_chk, n_ik
        any_hand = th.zeros(N, dtype=th.bool, device=dev)
        for d_e in des:
            any_hand |= hand_ok[d_e] & onsup[d_e]
        order = th.nonzero(in_reach & any_hand).flatten()
        rec["n_problems"] = rec.get("n_problems", 0) + int(order.numel())
        ta = time.time()
        succ_pre, q_pre = _solve(world, entry, TP_flat[order], q_lock)
        n_ik += int(order.numel())
        t_ik += time.time() - ta
        pre_any = succ_pre.any(dim=1)
        first_seed = th.argmax(succ_pre.int(), dim=1)
        q_pre1 = q_pre[th.arange(order.numel(), device=dev), first_seed]
        sel = th.nonzero(pre_any).flatten()
        ta = time.time()
        succ_g, q_g = _solve(world, entry, TE_flat[order[sel]], q_lock, seed_active=q_pre1[sel]) if sel.numel() else (None, None)
        n_ik += int(sel.numel())
        t_ik += time.time() - ta
        tb = time.time()
        feas = {d_e: th.zeros(sel.numel(), dtype=th.bool, device=dev) for d_e in des}
        why = {d_e: th.full((sel.numel(),), STATUS["ik_grasp"], dtype=th.uint8, device=dev) for d_e in des}
        if sel.numel():
            qp = q_pre1[sel]
            for s in range(cfg.seeds):
                ok_s = succ_g[:, s]
                if not bool(ok_s.any()):
                    continue
                qg = q_g[:, s]
                jump = (qg - qp).abs().max(dim=1).values > cfg.branch_max_rad
                cand_s = ok_s & ~jump
                for d_e in des:
                    cur = cand_s & ~feas[d_e]
                    why[d_e] = th.where(ok_s & jump & ~feas[d_e], th.full_like(why[d_e], STATUS["branch_jump"]), why[d_e])
                    if not bool(cur.any()):
                        continue
                    ii = th.nonzero(cur).flatten()
                    good = th.ones(ii.numel(), dtype=th.bool, device=dev)
                    sb = slab(height, d_e, cfg)
                    for t in [0.0] + [k / (cfg.approach_configs + 1) for k in range(1, cfg.approach_configs + 1)] + [1.0]:
                        for c0 in range(0, ii.numel(), 8192):
                            iic = ii[c0:c0 + 8192]
                            sp = world.spheres(world.full_q(entry, qp[iic] * (1 - t) + qg[iic] * t, q_lock))
                            o, _ = CP.sphere_support_ok(sp, links, sb, cfg.arm, policy)
                            good[c0:c0 + 8192] &= o.to(dev)
                    w_slab = ~good
                    # the hand hulls at the FK eef of the grasp and the pre-grasp (IK tolerance), the pose set's opening
                    for qq in (qg[ii], qp[ii]):
                        good &= hc.check(world.eef(world.full_q(entry, qq, q_lock)), sb, (fq, fq))[0]
                    w_hand = ~good & ~w_slab
                    feas[d_e][ii] |= good
                    w = why[d_e][ii]
                    w = th.where(w_slab, th.full_like(w, STATUS["slab_spheres"]), w)
                    w = th.where(w_hand, th.full_like(w, STATUS["hand_policy"]), w)
                    why[d_e][ii] = w
        t_chk += time.time() - tb
        for d_e in des:
            st = status[d_e]
            st.fill_(STATUS["unevaluated"])
            st[~in_reach] = STATUS["reach_bound"]
            st[in_reach & ~hand_ok[d_e]] = STATUS["hand_policy"]
            st[~onsup[d_e]] = STATUS["off_support"]
            vals = th.full((order.numel(),), STATUS["ik_pre"], dtype=th.uint8, device=dev)
            if sel.numel():
                vals[sel] = th.where(feas[d_e], th.full_like(why[d_e], STATUS["feasible"]), why[d_e])
            ok_here = hand_ok[d_e][order] & onsup[d_e][order]
            st[order] = th.where(ok_here, vals, st[order])
            a = st.view(nC, 1, nG).cpu().numpy()
            arrays[f"s{d_e:.2f}_{key}"] = a
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int((a == STATUS["feasible"]).sum())
            rec.setdefault("class_counts", {})[f"{d_e:.2f}"] = {r["name"]: int((a[:, 0, g] == STATUS["feasible"]).sum()) for g, r in enumerate(rows)}

    for tr, ti in trunks:
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        status = {d_e: th.zeros(N, dtype=th.uint8, device=dev) for d_e in cfg.edge_distances}
        q_of: Dict[str, th.Tensor] = {}

        def free_of(p: str, tr=tr, ti=ti) -> Optional[List[bool]]:
            q = world.q_config(tr, ti, p, fingers=fq)
            if bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                return None
            q_of[p] = q
            sph, ln = world.backend.robot_spheres_world(q.unsqueeze(0))
            return [bool(CP.sphere_support_ok(sph, ln, slab(height, d_e, cfg), cfg.arm, policy)[0][0]) for d_e in cfg.edge_distances]

        # i7-unblock 2026-11-07: the eligibility rule of compute_map (trunk_posture_groups, MapConfig.trunk_eligibility)
        groups, collide = trunk_posture_groups(cfg.postures, len(cfg.edge_distances), free_of, trunk_rule(cfg))
        if collide:
            rec["slab_collide_by_posture"] = collide
        rec["posture"] = groups[0][0] if groups else None
        if not groups:
            rec.update(eligible=False, reason="every idle-arm posture collides (self or the slab at some d_e)" if trunk_rule(cfg) == "all_d_e"
                       else "every idle-arm posture collides (self or the slab) at every d_e")
            for d_e in cfg.edge_distances:
                arrays[f"s{d_e:.2f}_{key}"] = np.full((nC, 1, nG), STATUS["trunk_ineligible"], dtype=np.uint8)
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        mark_partial(rec, groups, cfg.edge_distances, status, arrays, key, (nC, 1, nG))
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        reach = L + cfg.reach_margin_m
        in_reach = (th.linalg.norm(TE_flat[:, :3, 3] - sh_t, dim=1) <= reach) & (th.linalg.norm(TP_flat[:, :3, 3] - sh_t, dim=1) <= reach)
        rec.update({"shoulder": [round(float(v), 4) for v in sh], "arm_len": round(L, 4)})
        for p, ks in groups:
            solve_group(rec, key, status, q_of[p], [cfg.edge_distances[k] for k in ks], in_reach)
        meta_trunks.append(rec)
    tim.update({"ik_s": round(t_ik, 2), "checks_s": round(t_chk, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)})
    meta = {"schema": SCHEMA, "kind": GENERIC_KIND, "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": None, "object_key": GENERIC_KIND, "rest_id": None,
            "height": {"value": height, "anchor_key": anchor.get("key"), "anchor_surfaces": anchor.get("surfaces", [])[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "policy": policy.meta(cfg.arm),
            "slab": {"depth_m": cfg.slab_depth_m, "long_m": cfg.slab_long_m, "thickness_m": cfg.slab_thickness_m},
            "ik": {"seeds": cfg.seeds, "position_tolerance": world.backend.position_tolerance, "orientation_tolerance": world.backend.orientation_tolerance,
                   "collision_world": "floor cuboid + self-collision (slab checked explicitly)", "pre_dist": float(ps["pre_dist_m"])},
            "approach_check": f"hand hulls at {fq:.4f} m per finger at the pre-grasp and every {policy.approach_step_m * 2:.3f} m of the straight approach "
                              f"to the grasp (no closing sweep); whole-robot spheres at {cfg.approach_configs} joint-interpolated configurations",
            "screening": "full", "exhaustive": True, "pose_set": ps, "classes": list(ps["classes"]), "grasps": rows,
            "trunks": meta_trunks, "status_codes": STATUS, "robot": robot_digest(), "timing": tim, "created": S.now()}
    meta["key"] = generic_map_key(meta)
    return meta, arrays


def save_generic(out_dir: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    """``<out_dir>/map__generic__h<h>__<key>.{json,npz}`` (``out_dir`` = ``<maps root>/_generic``)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"map__{GENERIC_KIND}__h{float(meta['height']['value']):.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    S.save_json(base + ".json", meta)
    return base + ".json"


def find_generic(maps_root: str, height: float, *, arm: str = "right", pose_set: Optional[Dict[str, Any]] = None, policy_key: Optional[str] = None,
                 robot: Optional[str] = None, config_require: Optional[Dict[str, Any]] = None, anchors=None) -> Optional[ReachMap]:
    """The generic map at the anchor covering ``height`` (``AnchorSet.lookup``: exact, or within the anchor's clipped range; without
    ``anchors`` the height itself) in ``maps_root`` (the ``_generic`` directory), or None. Identity enforced: schema, kind ``generic``, arm,
    the pose set (dz, fingers, classes, ...: equal as recorded), the policy key, the robot digest, the required config fields, the height,
    and the map's own key over those (a hand-edited identity does not serve). The hit carries ``rm.lookup`` = ``{how, query_h, map_h}``."""
    target, how = float(height), "height"
    if anchors is not None:
        a, how = anchors.lookup(height)
        if a is None:
            return None
        target = float(a["height"])
    want_ps = json.dumps(pose_set, sort_keys=True) if pose_set is not None else None
    fallback: Optional[str] = None
    for fn in sorted(os.listdir(maps_root)) if os.path.isdir(maps_root) else []:
        if not fn.endswith(".json") or not fn.startswith("map__"):
            continue
        p = os.path.join(maps_root, fn)
        try:
            with open(p) as f:
                m = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if m.get("schema") != SCHEMA or m.get("kind") != GENERIC_KIND or m.get("arm") != arm:
            continue
        if policy_key is not None and m.get("policy_key") != policy_key:
            continue
        if robot is not None and m.get("robot") != robot:
            continue
        if config_require and any(config_value(m, k) != v for k, v in config_require.items()):
            continue
        if want_ps is not None and json.dumps(m.get("pose_set"), sort_keys=True) != want_ps:
            continue
        if abs(float(m["height"]["value"]) - target) > 1e-6 or m.get("key") != generic_map_key(m):
            continue
        if config_value(m, "trunk_eligibility") != "per_d_e" and not (config_require and "trunk_eligibility" in config_require):
            fallback = fallback or p  # i7-unblock 2026-11-07: a per_d_e map of the same lookup serves first (ReachMap.find)
            continue
        fallback = p
        break
    if fallback is None:
        return None
    rm = ReachMap(fallback)
    rm.lookup = {"how": how, "query_h": round(float(height), 4), "map_h": float(rm.meta["height"]["value"])}
    return rm


def _generic_meta_matches(path: str, arm: str, pose_set: Dict[str, Any], trunk_eligibility: Optional[str] = TRUNK_ELIGIBILITY_DEFAULT) -> bool:
    """``--skip-existing`` of a generic job: the map ``path`` is this run's arm, the generic kind, the job's pose set and its trunk eligibility
    rule (i7-unblock 2026-11-07; None = "all_d_e")."""
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    if config_value(m, "trunk_eligibility") != (trunk_eligibility or "all_d_e"):
        return False
    return m.get("arm") == arm and m.get("kind") == GENERIC_KIND and json.dumps(m.get("pose_set"), sort_keys=True) == json.dumps(pose_set, sort_keys=True)


def _compute_generic_cli(a) -> None:
    """``compute --kind generic``: the manifest's generic pseudo-entries (``{"id": "_generic", "kind": "generic"}``, optional ``dz_m``,
    ``fingers_q``, ``dir`` (default ``_generic``)) at the anchors (``--heights`` / ``--heights-by-id`` by the entry's id) for ``--arm``,
    written to ``<--out>/<dir>/``; every other manifest entry (and ``--include-derived`` / ``--glib``) is ignored. Sharded by the index of
    the generic entries (``i/n``): several entries with distinct ids and ``--heights-by-id`` split the heights over the shards."""
    from ispatialgym.sampling.height_anchors import AnchorSet

    with open(a.jobs) as f:
        jobs = [j for j in json.load(f) if j.get("kind") == GENERIC_KIND]
    if a.only:
        jobs = [j for j in jobs if j["id"] in set(a.only)]
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    with open(a.anchors) as f:
        an = json.load(f)
    aset = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
    want = set(a.heights) if a.heights else None
    anchors = [x for x in aset.anchors if want is None or x["key"] in want]
    by_id = json.load(open(a.heights_by_id)) if a.heights_by_id else {}
    if not jobs:
        print(f"GB_MAP {json.dumps({'kind': GENERIC_KIND, 'ok': True, 'reason': 'no generic entry in this shard'})}", flush=True)
        print("GB_MAP_DONE", flush=True)
        raise SystemExit(0)  # no synthetic world was started: nothing to shut down
    cfg = MapConfig(mode="full", lattice_m=a.lattice, seeds=a.seeds, batch=a.batch, arm=a.arm, twins=False, **_trunk_kw(a))
    w = SynthWorld(cfg)
    known = {x["key"] for x in aset.anchors}
    for j in jobs:
        anchors_j = [x for x in aset.anchors if x["key"] in set(by_id[j["id"]])] if j["id"] in by_id else anchors
        for k in sorted(set(by_id.get(j["id"], a.heights or [])) - known):  # a height the anchor file lacks is said, not skipped silently
            print(f"GB_MAP {json.dumps({'id': j['id'], 'kind': GENERIC_KIND, 'ok': False, 'h': k, 'arm': cfg.arm, 'reason': 'not an anchor key of --anchors'})}", flush=True)
        ps = generic_pose_set(float(j.get("dz_m", GENERIC_DZ_M)), j.get("fingers_q"))
        out = os.path.join(a.out, j.get("dir") or GENERIC_DIR)
        for anc in anchors_j:
            h = float(anc["height"])
            if a.skip_existing and any(_generic_meta_matches(p, cfg.arm, ps, cfg.trunk_eligibility) for p in glob.glob(os.path.join(out, f"map__{GENERIC_KIND}__h{h:.4f}__*.json"))):
                continue
            try:
                meta, arrays = compute_generic_map(w, h, anc, cfg=cfg, pose_set=ps)
                p = save_generic(out, meta, arrays)
                nf = {t["key"]: t.get("n_feasible") for t in meta["trunks"] if t.get("eligible")}
                print(f"GB_MAP {json.dumps({'id': j['id'], 'kind': GENERIC_KIND, 'ok': True, 'h': anc['key'], 'arm': cfg.arm, 'path': p, 'timing': meta['timing'], 'n_classes': len(meta['grasps']), 'dz_m': ps['dz_m'], 'fingers_q': ps['fingers_q'], 'feasible': nf})}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                print(f"GB_MAP {json.dumps({'id': j['id'], 'kind': GENERIC_KIND, 'ok': False, 'h': anc['key'], 'arm': cfg.arm, 'error': repr(e), 'tb': traceback.format_exc()[-1200:]})}", flush=True)
    print("GB_MAP_DONE", flush=True)


# ------------------------------------------------------------------------------------------------ articulation opening prefixes (prompt 2 §7)


def compute_prefix_map(world: SynthWorld, alib: Dict[str, Any], height: float, anchor: Dict[str, Any], cfg: Optional[MapConfig] = None,
                       policy: CP.ContactPolicy = CP.DEFAULT, min_open_rad: float = 0.2, subsample: int = 2, step_max_rad: float = 0.5
                       ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Reachable opening prefixes of an articulated tabletop object (``validate_artic``'s ``artic_library.json``: the measured rest and,
    per tested handle, the FIRST fixed-stance pull segment as object-relative eef waypoints with the joint value). Per (edge distance,
    trunk, cell, yaw, segment): the pre-grasp and the waypoints in order by batched IK, each seeded from the previous waypoint's
    solution; a configuration counts only while every joint stays within ``step_max_rad`` of the previous one (no branch jumps between
    disconnected successes) and the whole robot keeps the contact policy against the slab (spheres; the hand hulls at the FK eef).
    Stored: the contiguous prefix (rad of joint travel from the initial state) - ``accepted`` where it reaches ``min_open_rad`` (the full
    validated range is not required). Hybrid demonstrations contribute their first fixed-stance segment only; the achieved range is that
    of the tested grasp / stance / controller, not the object's limit. The moving link's geometry is not posed per waypoint: the arm-vs-
    object check is not run here (the exact sampler checks the articulated target)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    t_all = time.time()
    dev = world.dev
    rest = S.Rest.from_record(alib["rests"]["r0"])
    cells = lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY = len(cells), len(yaws)
    To = np.stack([S.place_object(rest, y, (0.0, 0.0), height) for y in yaws])  # (Y, 4, 4) at the origin cell
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32)}
    entry = world.entry()
    links = world.link_of_sphere()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)
    segs_meta = []
    t_ik = 0.0
    n_ik = 0
    for si, seg in enumerate(alib.get("segments") or []):
        wps = seg["waypoints"][::max(1, subsample)]
        if len(wps) < 2:
            continue
        q0 = float(wps[0]["q"])
        travel = np.array([abs(float(w["q"]) - q0) for w in wps])
        W = np.stack([S.pose_of(w["eef_in_object"]) for w in wps])  # (K, 4, 4)
        TW = np.einsum("yij,kjl->ykil", To, W)  # (Y, K, 4, 4)
        T = th.as_tensor(TW, dtype=th.float32, device=dev).unsqueeze(0).repeat(nC, 1, 1, 1, 1)
        T[..., 0, 3] += th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1)
        T[..., 1, 3] += th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1)
        T = T.reshape(nC * nY, len(wps), 4, 4)
        segs_meta.append({"index": si, "candidate": seg.get("candidate"), "joint": seg.get("joint"), "n_waypoints": len(wps),
                          "validated_travel_rad": round(float(travel.max()), 4), "trial_stop": seg.get("stop"), "profile": seg.get("profile")})
        for tr, ti in trunk_grid(height):
            key = f"t{tr:.2f}_k{ti:+.1f}"
            q_lock = world.q_config(tr, ti, "untucked")
            if bool(world.backend.check_collision(q_lock.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                for d_e in cfg.edge_distances:
                    arrays[f"s{d_e:.2f}_{key}__seg{si}"] = np.full((nC, nY), -1.0, dtype=np.float32)
                continue
            sh, L = world.shoulder(tr, ti)
            sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
            in_reach = (th.linalg.norm(T[:, 0, :3, 3] - sh_t, dim=1) <= L + cfg.reach_margin_m)
            prefix = {d_e: th.full((nC * nY,), -1.0, device=dev) for d_e in cfg.edge_distances}
            alive = {d_e: in_reach.clone() for d_e in cfg.edge_distances}
            # the pre-grasp of the first waypoint, then the chain
            Tp = T[:, 0].clone()
            Tp[:, :3, 3] -= C.PRE_DIST * Tp[:, :3, 2]
            idx = th.nonzero(in_reach).flatten()
            if idx.numel() == 0:
                for d_e in cfg.edge_distances:
                    arrays[f"s{d_e:.2f}_{key}__seg{si}"] = prefix[d_e].view(nC, nY).cpu().numpy()
                continue
            ta = time.time()
            sp_ok, qp = _solve(world, entry, Tp[idx], q_lock)
            t_ik += time.time() - ta
            n_ik += int(idx.numel())
            ok_any = sp_ok.any(dim=1)
            prev = th.zeros(nC * nY, qp.shape[-1], device=dev)
            prev[idx] = qp[th.arange(idx.numel(), device=dev), th.argmax(sp_ok.int(), dim=1)]
            for d_e in cfg.edge_distances:
                a = alive[d_e]
                a[idx[~ok_any]] = False
            for k in range(len(wps)):
                live = alive[cfg.edge_distances[0]].clone()
                for d_e in cfg.edge_distances[1:]:
                    live |= alive[d_e]
                ii = th.nonzero(live).flatten()
                if ii.numel() == 0:
                    break
                ta = time.time()
                sk, qk = _solve(world, entry, T[ii, k], q_lock, seed_active=prev[ii])
                t_ik += time.time() - ta
                n_ik += int(ii.numel())
                best = th.argmax(sk.int(), dim=1)
                qsel = qk[th.arange(ii.numel(), device=dev), best]
                okk = sk.any(dim=1) & ((qsel - prev[ii]).abs().max(dim=1).values <= step_max_rad)
                qf = world.full_q(entry, qsel, q_lock)
                sp = world.spheres(qf)
                eef = world.eef(qf)
                for d_e in cfg.edge_distances:
                    sb = slab(height, d_e, cfg)
                    o_s, _ = CP.sphere_support_ok(sp, links, sb, cfg.arm, policy)
                    o_h, _, _ = hc.check(eef, sb, (0.0, 0.0))
                    good = okk & o_s.to(dev) & o_h & alive[d_e][ii]
                    alive[d_e][ii] = good
                    prefix[d_e][ii[good]] = float(travel[k])
                prev[ii] = qsel
            for d_e in cfg.edge_distances:
                arrays[f"s{d_e:.2f}_{key}__seg{si}"] = prefix[d_e].view(nC, nY).cpu().numpy()
    meta = {"schema": SCHEMA, "kind": "articulation_prefix", "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": {"category": alib.get("category"), "model": alib.get("model"), "scale": alib.get("scale"), "rests": {"r0": {k: alib["rests"]["r0"][k] for k in ("quat_xyzw", "h_root")}}},
            "object_key": S.digest({"artic": alib.get("category"), "m": alib.get("model"), "rest": alib["rests"]["r0"]["quat_xyzw"], "segs": [s_["candidate"] for s_ in segs_meta]}, 16),
            "rest_id": "r0", "height": {"value": height, "anchor_key": anchor.get("key")}, "edge_distances": list(cfg.edge_distances), "config": cfg.key(),
            "policy_key": policy.key(cfg.arm), "segments": segs_meta, "min_open_rad": min_open_rad, "step_max_rad": step_max_rad,
            "value": "contiguous joint travel (rad) from the initial state reachable from the cell / yaw / trunk; -1 = not even the grasp",
            "scope": "the tested grasp / stance / controller's first fixed-stance segment - not the object's global opening limit",
            "robot": robot_digest(), "timing": {"ik_s": round(t_ik, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)}, "created": S.now()}
    meta["key"] = S.digest({k: meta[k] for k in ("schema", "kind", "arm", "object_key", "edge_distances", "config", "policy_key", "robot")} | {"h": height}, 16)
    return meta, arrays


# ------------------------------------------------------------------------------------------------ CLI


def _bench(args):
    cfg = MapConfig(batch=args.batch, seeds=args.seeds)
    w = SynthWorld(cfg)
    entry = w.entry()
    q = w.q_config(0.5, 0.0)
    ai = w.robot.arm_control_idx[cfg.arm]
    g = th.Generator().manual_seed(0)
    n = args.n
    Q = q.unsqueeze(0).repeat(n, 1)
    Q[:, ai] += 0.6 * (th.rand(n, len(ai), generator=g) - 0.5)
    T = w.eef(Q.to(w.dev))
    _solve(w, entry, T[: cfg.batch], q)  # warm (graph capture)
    th.cuda.synchronize()
    t0 = time.time()
    succ, qs = _solve(w, entry, T, q)
    th.cuda.synchronize()
    dt = time.time() - t0
    t1 = time.time()
    sp = w.spheres(w.full_q(entry, qs[:, 0], q))
    th.cuda.synchronize()
    print(f"BENCH seeds {cfg.seeds} batch {cfg.batch}: {n} IK problems in {dt:.2f}s = {n / dt:.0f}/s, any-seed success {float(succ.any(1).float().mean()):.3f}; "
          f"sphere FK {n} in {time.time() - t1:.3f}s; setup {w.setup_s:.1f}s; peak {th.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def _map_meta_matches(path: str, arm: str, kind: str, floor_fingers: Optional[str] = None, trunk_eligibility: Optional[str] = TRUNK_ELIGIBILITY_DEFAULT) -> bool:
    """``--skip-existing``: the map ``path`` (``.json``) is this run's arm and kind (an unreadable file does not count); a floor run's map also
    its finger rule (``floor_fingers``; i7-floorfix 2026-11-07: a floor map of the former rule does not stand in for the new one), and its
    trunk eligibility rule (i7-unblock 2026-11-07: ``trunk_eligibility``, None = "all_d_e"; a map of the former rule does not stand in for
    a "per_d_e" one)."""
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    if kind == "floor" and floor_fingers is not None and config_value(m, "floor_fingers") != floor_fingers:
        return False
    if config_value(m, "trunk_eligibility") != (trunk_eligibility or "all_d_e"):
        return False
    return m.get("arm", "right") == arm and m.get("kind", "tabletop") == kind


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=16384)
    b.add_argument("--batch", type=int, default=2048)
    b.add_argument("--seeds", type=int, default=8)
    c = sub.add_parser("compute")
    c.add_argument("--jobs", required=True)
    c.add_argument("--glib", default=None)
    c.add_argument("--anchors", default=None, help="the height anchors (required unless --kind floor)")
    c.add_argument("--heights", nargs="*", default=None, help="anchor keys (default: every anchor; --kind floor: ignored, h 0)")
    c.add_argument("--heights-by-id", default=None, help="json {job id: [anchor keys]} (overrides --heights per job)")
    c.add_argument("--only", nargs="*", default=None)
    c.add_argument("--shard", default=None, help="i/n")
    c.add_argument("--out", required=True)
    c.add_argument("--mode", default="full", choices=["accelerated", "full"])
    c.add_argument("--lattice", type=float, default=0.08)
    c.add_argument("--seeds", type=int, default=8)
    c.add_argument("--batch", type=int, default=2048)
    c.add_argument("--kind", default="tabletop", choices=list(KINDS) + ["generic"],
                   help="floor: the object on the floor plane (h 0, no edge, floor_config); generic: the manifest's generic pseudo-entries ({\"id\": \"_generic\", \"kind\": \"generic\"}) -> <out>/_generic/")
    c.add_argument("--skip-existing", action="store_true", help="skip (job, height) pairs whose map file exists already")
    c.add_argument("--include-derived", action="store_true", help="also every library entry of --glib that is not a manifest job (the measured-rest retries)")
    c.add_argument("--arm", default="right", choices=["right", "left"])
    c.add_argument("--trunk-eligibility", default=None, choices=list(TRUNK_ELIGIBILITY_RULES),
                   help="the trunk eligibility rule (MapConfig.trunk_eligibility; default: MapConfig's, per_d_e). all_d_e = the former rule (the field is left out "
                        "of the map's config and key); per_d_e = a trunk is ineligible only at the edge distances where every idle-arm posture collides")
    pf = sub.add_parser("prefix")
    pf.add_argument("--artic-lib", nargs="+", required=True, help="artic_library.json files (validate_artic)")
    pf.add_argument("--anchors", required=True)
    pf.add_argument("--heights", nargs="+", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--min-open-rad", type=float, default=0.2)
    i = sub.add_parser("info")
    i.add_argument("map")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        _bench(a)
    elif a.cmd == "prefix":
        an = json.load(open(a.anchors))
        anchors = {x["key"]: x for x in an["anchors"]}
        w = SynthWorld(MapConfig())
        for p in a.artic_lib:
            alib = json.load(open(p))
            for hk in a.heights:
                try:
                    meta, arrays = compute_prefix_map(w, alib, float(anchors[hk]["height"]), anchors[hk], min_open_rad=a.min_open_rad)
                    base = os.path.join(a.out, f"prefix__{alib['category']}__{alib['model']}__h{float(hk):.4f}__{meta['key']}")
                    os.makedirs(a.out, exist_ok=True)
                    np.savez_compressed(base + ".npz", **arrays)
                    S.save_json(base + ".json", meta)
                    acc = {k: int((v >= a.min_open_rad).sum()) for k, v in arrays.items() if k.startswith("s0.45") and k.endswith("__seg0")}
                    print("GB_PREFIX " + json.dumps({"lib": p, "h": hk, "timing": meta["timing"], "segments": meta["segments"], "accepted_cells_x_yaws@0.45": acc,
                                                     "max_prefix_rad": max(float(v.max()) for k, v in arrays.items() if "__seg" in k)}), flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print("GB_PREFIX " + json.dumps({"lib": p, "h": hk, "error": repr(e), "tb": traceback.format_exc()[-1500:]}), flush=True)
    elif a.cmd == "info":
        m = ReachMap(a.map)
        print(json.dumps({k: m.meta[k] for k in ("key", "rest_id", "height", "edge_distances", "screening", "timing")}, indent=1))
        for t in m.meta["trunks"]:
            print(t["key"], t.get("eligible"), t.get("posture"), t.get("n_problems"), t.get("n_feasible"), t.get("n_skipped"))
    elif a.kind == GENERIC_KIND:
        _compute_generic_cli(a)
    else:
        _compute_cli(a)
    import omnigibson as og

    if a.cmd != "info":
        og.shutdown()


def _trunk_kw(a) -> Dict[str, Any]:
    """``--trunk-eligibility`` as MapConfig keyword(s): {} when not given (MapConfig's default), "all_d_e" -> None (the former rule, left out
    of the map's config and key: such a build reproduces the earlier maps and their keys), "per_d_e" as given."""
    rule = getattr(a, "trunk_eligibility", None)
    if rule is None:
        return {}
    return {"trunk_eligibility": None if rule == "all_d_e" else rule}


def _compute_cli(a):
    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.sampling.height_anchors import AnchorSet

    with open(a.jobs) as f:
        jobs = json.load(f)
    if a.only:
        jobs = [j for j in jobs if j["id"] in set(a.only) or j.get("legacy_key") in set(a.only)]
    derived = []
    if a.include_derived and a.glib:
        known = {j["id"] for j in jobs}
        for p in sorted(glob.glob(os.path.join(S.library_root(a.glib), "models", "*", "library.json"))):
            eid = os.path.basename(os.path.dirname(p))
            if eid in known:
                continue
            lib = json.load(open(p))
            cand = S.load_json(os.path.join(os.path.dirname(p), "candidates.json")) or {}
            parent = (S.load_json(os.path.join(os.path.dirname(p), "job.json")) or {}).get("derived_from")
            derived.append({"id": eid, "category": lib["category"], "model": lib["model"], "scale": lib.get("scale"), "constraint": lib.get("constraint") or "tabletop",
                            "rest_hint_wxyz": cand.get("rest_hint_wxyz"), "derived": True, "derived_from": parent})
    if a.shard:
        # a derived entry goes to its parent job's shard - the shard whose glib stage made it, so it exists when this stage lists the
        # library; sharding the appended list by index instead raced the other shards' retries (each saw a different list). The parent
        # is job.json's derived_from; entries made before it was written fall back to the first job with the same id minus the rest digest
        i, n = (int(v) for v in a.shard.split("/"))
        base = lambda eid: re.sub(r"__r[0-9a-f]{6}$", "", eid)  # noqa: E731 - the entry without its designated-rest digest
        index, first = {}, {}
        for k, j in enumerate(jobs):
            index[j["id"]] = k
            first.setdefault(base(j["id"]), k)
        home = lambda d: index[d["derived_from"]] if d.get("derived_from") in index else first.get(base(d["id"]), 0)  # noqa: E731
        jobs = [j for k, j in enumerate(jobs) if k % n == i] + [d for d in derived if home(d) % n == i]
    else:
        jobs += derived
    floor = a.kind == "floor"
    kw = dict(mode=a.mode, lattice_m=a.lattice, seeds=a.seeds, batch=a.batch, arm=a.arm, twins=a.kind != "brace", **_trunk_kw(a))
    if floor:
        # i7-floor 2026-11-07: one map per (entry, stable rest, arm) at the floor (h 0, the synthetic FLOOR_ANCHOR); no anchor file, no heights
        if a.heights or a.heights_by_id:
            print(f"GB_MAP_NOTE {json.dumps({'kind': 'floor', 'note': '--heights / --heights-by-id ignored: a floor map is at h 0'})}", flush=True)
        aset, anchors, by_id = None, [dict(FLOOR_ANCHOR)], {}
        cfg = floor_config(**kw)
    else:
        if not a.anchors:
            raise SystemExit(f"--anchors is required for --kind {a.kind}")
        with open(a.anchors) as f:
            an = json.load(f)
        aset = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
        want = set(a.heights) if a.heights else None
        anchors = [x for x in aset.anchors if want is None or x["key"] in want]
        cfg = MapConfig(**kw)
        by_id = json.load(open(a.heights_by_id)) if a.heights_by_id else {}
    w = SynthWorld(cfg)
    for j in jobs:
        anchors_j = [x for x in aset.anchors if x["key"] in set(by_id[j["id"]])] if (aset is not None and j["id"] in by_id) else anchors
        lib = S.load_model(j["category"], j["model"], j.get("scale"), a.glib, constraint=j.get("constraint") or "tabletop", rest_hint_wxyz=j.get("rest_hint_wxyz"))
        if lib is None or not lib.grasps():
            print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': False, 'reason': 'no v2 library entry with validated grasps'})}", flush=True)
            continue
        geom = OG.load(os.path.dirname(lib.path))
        for rid, r in lib.rests.items():
            if not r.stable or not lib.grasps(rid):
                continue
            for anc in anchors_j:
                # of-v2maps 2026-11-06: an existing map counts only for this run's arm and kind (the file name's key hashes the arm, the glob
                # did not: a left-arm run after the right-arm one into the same root computed nothing)
                if a.skip_existing and any(_map_meta_matches(p, cfg.arm, a.kind, cfg.floor_fingers, cfg.trunk_eligibility)
                                           for p in glob.glob(os.path.join(a.out, j["id"], f"map__*__h{float(anc['height']):.4f}__*.json"))):
                    continue
                try:
                    meta, arrays = compute_map(w, lib, rid, float(anc["height"]), anc, geom=geom, cfg=cfg, kind=a.kind)
                    p = save(os.path.join(a.out, j["id"]), meta, arrays)
                    nf = {t["key"]: t.get("n_feasible") for t in meta["trunks"] if t.get("eligible")}
                    print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': True, 'rest': rid, 'h': anc['key'], 'path': p, 'timing': meta['timing'], 'n_grasps': len(meta['grasps']), 'feasible': nf})}", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': False, 'rest': rid, 'h': anc['key'], 'error': repr(e), 'tb': traceback.format_exc()[-1200:]})}", flush=True)
    print("GB_MAP_DONE", flush=True)


if __name__ == "__main__":
    main()
