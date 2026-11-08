"""Contact reach maps from the PUSH / BRACE contact libraries (i7-contact 2026-11-07, step 3; :mod:`ispatialgym.grasping.contact_library`).

A contact map is the reach map of a library's entries - the contacts the oracle executes (``contact_poses.contact_entry``: the contact eef
pose, the EXECUTED pre-contact - a straight push's lowered 0.04 m pre, owner decision 6 -, the fingers' opening, the push stroke / arc,
the partner gap and the task goal) - over a lattice of object positions in the robot's frame, computed in the synthetic world of
:mod:`reach_maps` (the R1Pro alone, cuRobo; the support a synthetic slab at the anchor height, or the floor plane). Kinds
(:data:`KINDS`): ``touch_push`` (right arm, the touching targets), ``touch_brace`` (left arm, the touching partners), ``push_blade``
(right arm, the axial blade push), ``axial_brace`` (left arm, the axial stop brace: the object ROOT at its GOAL pose), ``floor_push``
(right arm, on the floor plane: the push-under blade families when their library exists), ``wall_brace`` (i7-brace, owner 3.4: the world_vertical
brace in the WALL frame - per stand-off d_e the wall plane at x = d_e, the hung target's root row in front of it, the hinge at the map height;
each arm's entries their own map). No maps for the base push or articulation_large.

Axes: per edge distance x trunk (``reach_bands.trunk_grid(h)``) a status (:data:`STATUS`) per (cell = the object ROOT xy in the map frame -
the base on the floor, +x ahead -, yaw slot = the library's object yaw relative to the stance (``yaw_deg``: the start yaw, the goal yaw of
an axial brace), class = the library entry's class label). A (yaw, class) without an entry is ``no_entry``. Per problem (the entry at the
object pose ``T_rule_object`` with its root xy at the cell and the support at the anchor height) the CHAIN the oracle runs -
the executed pre-contact, the contact, then a straight push's end (contact + stroke x direction) or an arc's mid and end - is checked
as :func:`reach_maps.compute_map` checks a grasp: the shoulder's reach, the hand's hulls against the support (contact policy, the
entry's own finger opening) at every chain pose and along the approach / the stroke, batched IK of the pre-contact and of each next pose
seeded from the previous one (same branch: ``branch_max_rad``), the whole robot's spheres clear of the support at the chain's configurations
and ``approach_configs`` interpolated ones, the arm (not the hand) clear of the object at the contact. A map is a proposal filter: the
sampler's exact checks stay exact.

Identity (:func:`map_key`): the kind, the arm, the library object (entry, category / model / scale, rest, geometry digest), the height,
the edge distances, the config, the contact policy key, the robot digest, and the POSE SET (:func:`pose_set`: per (yaw, class) the
entry's pose content digest - the contact, the executed pre and its rule, the fingers' opening, the stroke / arc / partner gap / goal, the
brace stop - and the verdict filter). Files ``<out>/<entry>/cmap__<kind>__<category>__<model>__h<h>__<key>.{json,npz}``.

    python -m ispatialgym.sampling.contact_maps compute --root LIBROOT --kind touch_push --out MAPS [--heights 0.8126 ...] [--shard i/n]
    python -m ispatialgym.sampling.contact_maps info MAP.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import reach_maps as RM

SCHEMA = "isg-contact-map/1"
#: kind -> the library kind and families it reads, the arm, the support plane, the object frame of the cell
KINDS: Dict[str, Dict[str, Any]] = {
    "touch_push": {"lib_kind": "push", "families": ("touching_push",), "arm": "right", "plane": "slab", "frame": "start"},
    "touch_brace": {"lib_kind": "brace", "families": ("touching_brace",), "arm": "left", "plane": "slab", "frame": "start"},
    "push_blade": {"lib_kind": "push", "families": ("axial_push",), "arm": "right", "plane": "slab", "frame": "start"},
    "axial_brace": {"lib_kind": "brace", "families": ("axial_brace",), "arm": "left", "plane": "slab", "frame": "goal"},
    "floor_push": {"lib_kind": "push", "families": ("floor_blade", "floor_flat", "floor_pitched"), "arm": "right", "plane": "floor", "frame": "start"},
    # i7-brace (owner 3.4): the world_vertical brace in the WALL frame - the base square to the wall at the stand-off d_e (the wall plane at
    # x = d_e facing -x), the cell = the hung target's ROOT at its level (goal) pose (x = d_e + the rule frame's root x, y lateral), the hinge
    # at the map height above the floor; arm None = per entry (``wall.brace_arm``: one map per arm, :func:`select_entries`)
    "wall_brace": {"lib_kind": "brace", "families": ("wall_brace",), "arm": None, "plane": "wall", "frame": "goal"},
}
#: i7-brace: the wall map's stand-offs (prompt0's Phase 2 d lattice: D_RANGE_DEFAULT 0.35-0.90 every D_STEP 0.05), the lateral lattice
#: (LAMBDA_STEP 0.05 over the task's lateral_range +-0.40 plus the root's offset from the hinge), the idle arm's posture (the task's A6
#: ``stance_arm_posture``: tucked), the wall box (thickness, width, height from the floor)
WALL_EDGE_DISTANCES = tuple(round(0.35 + 0.05 * k, 2) for k in range(12))
WALL_LATTICE_M = 0.05
WALL_Y_HALF_M = 0.60
WALL_POSTURES = ("tucked",)
WALL_BOX_M = (0.10, 3.2, 2.6)
#: the reach-map status codes plus a (yaw, class) the library has no entry for
STATUS = dict(RM.STATUS, no_entry=12)
STATUS_NAME = {v: k for k, v in STATUS.items()}
#: the entry fields that are not its pose content (the validator's verdict, the screen, the stored digest)
NON_POSE_FIELDS = ("validation", "digest", "sampler_screen")
DEFAULT_VERDICTS = ("valid",)


# ------------------------------------------------------------------------------------------------ the pose set (pure)


def entry_pose_digest(entry: Dict[str, Any]) -> str:
    """The digest of an entry's pose content (everything but :data:`NON_POSE_FIELDS`): the contact, the executed pre and its rule, the
    fingers, the stroke / arc / partner gap / goal target, the stop rule, the yaw slot and the object frame."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    return CPZ.digest({k: v for k, v in entry.items() if k not in NON_POSE_FIELDS})


def select_entries(lib_k: Dict[str, Any], kind: str, verdicts: Optional[Sequence[str]] = DEFAULT_VERDICTS, arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """The library entries a ``kind`` map is built from: the kind's families, with a validation verdict in ``verdicts`` (None: every entry,
    validated or not); a per-entry-arm kind (``arm`` None in :data:`KINDS`: wall_brace) the entries of ``arm``'s brace only (``wall.brace_arm``)."""
    spec = KINDS[kind]
    if lib_k.get("kind") != spec["lib_kind"]:
        raise ValueError(f"a {kind} map reads a {spec['lib_kind']} library, not {lib_k.get('kind')!r}")
    out = []
    for e in lib_k.get("entries") or []:
        if e.get("family") not in spec["families"]:
            continue
        if verdicts is not None and (e.get("validation") or {}).get("verdict") not in verdicts:
            continue
        if spec["arm"] is None and arm is not None and str((e.get("wall") or {}).get("brace_arm")) != arm:
            continue
        out.append(e)
    return out


def axes(entries: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[str], Dict[Tuple[int, int], Dict[str, Any]]]:
    """``(yaws_deg, classes, grid)``: the sorted yaw slots and class labels of the entries and ``grid[(yaw index, class index)]`` = the entry
    (one per pair: a duplicate raises)."""
    yaws = sorted({round(float(e.get("yaw_deg", 0.0)), 4) for e in entries})
    classes = sorted({str(e["class"]) for e in entries})
    yi = {y: i for i, y in enumerate(yaws)}
    gi = {c: i for i, c in enumerate(classes)}
    grid: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for e in entries:
        k = (yi[round(float(e.get("yaw_deg", 0.0)), 4)], gi[str(e["class"])])
        if k in grid:
            raise ValueError(f"two entries of class {e['class']} at yaw {e.get('yaw_deg')}")
        grid[k] = e
    return yaws, classes, grid


def class_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """One (yaw, class) of the map's meta: what the oracle executes there - the fingers' opening, the pre-pose rule (owner decision 6) and the
    executed pre distance, the push (direction, stroke, partner gap, arc turn), the task goal, the brace stop, the axial goal facts - and the
    entry's pose digest."""
    push = entry.get("push") or None
    row = {"class": entry["class"], "family": entry.get("family"), "yaw_deg": entry.get("yaw_deg"), "fingers": entry.get("fingers"),
           "pre_pose_rule": entry.get("pre_pose_rule"), "pre_executed_m": (entry.get("pre") or {}).get("executed_m"),
           "push": None if push is None else {k: push.get(k) for k in ("direction", "stroke_m", "arc", "stop_gap_m")},
           "arc_turn_rad": (entry.get("arc") or {}).get("turn_rad"), "goal_target": entry.get("goal_target"), "stop": entry.get("stop"),
           "pose_digest": entry_pose_digest(entry), "verdict": (entry.get("validation") or {}).get("verdict")}
    if entry.get("goal"):
        row["goal"] = {k: entry["goal"].get(k) for k in ("yaw_goal_deg", "turn_rad", "end_sign", "spin")}
        for k in ("clear_turn_max_deg", "opposite_end", "tilt_sign", "clear_delta_max_deg"):  # i7-brace (owner 3.4): only when the entry has them
            if entry["goal"].get(k) is not None:
                row["goal"][k] = entry["goal"][k]
    if entry.get("wall"):  # i7-brace: the wall brace's kind / wrist / arm (the sampler's brace identity)
        row["wall"] = {"kind": (entry.get("extras") or {}).get("kind"), "wrist": (entry.get("extras") or {}).get("wrist"),
                       "brace_arm": entry["wall"].get("brace_arm")}
    return row


def pose_set(kind: str, entries: Sequence[Dict[str, Any]], verdicts: Optional[Sequence[str]]) -> Dict[str, Any]:
    """The map's pose set (part of its identity): the families, the verdict filter, the yaw slots, the classes and the digest of every
    (yaw, class) entry's pose content."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    yaws, classes, grid = axes(entries)
    cells = sorted((yaws[y], classes[g], entry_pose_digest(e)) for (y, g), e in grid.items())
    return {"kind": kind, "families": list(KINDS[kind]["families"]), "verdicts": None if verdicts is None else sorted(verdicts),
            "yaws_deg": yaws, "classes": classes, "n_entries": len(grid), "frame": KINDS[kind]["frame"], "digest": CPZ.digest(cells)}


def chain_object(entry: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """The eef chain the oracle runs, in the object frame: the EXECUTED pre-contact, the contact, then a straight push's end (contact +
    stroke x direction, the entry's object-frame direction) or an arc's mid and end (``arc.mid_object`` / ``goal_object``)."""
    Tc = S.pose_of(entry["T_object_eef"])
    out = [("pre", S.pose_of(entry["pre"]["executed"])), ("contact", Tc)]
    push = entry.get("push") or {}
    if entry.get("arc"):
        out += [("mid", S.pose_of(entry["arc"]["mid_object"])), ("goal", S.pose_of(entry["arc"]["goal_object"]))]
    elif push.get("stroke_m") and push.get("direction") is not None:
        Tg = Tc.copy()
        Tg[:3, 3] = Tc[:3, 3] + float(push["stroke_m"]) * np.asarray(push["direction"], dtype=float)
        out.append(("goal", Tg))
    return out


def object_pose(entry: Dict[str, Any], cell_xy, height: float) -> np.ndarray:
    """The object's pose of a problem: the entry's rule-frame pose (``T_rule_object``: its rest at the yaw slot - an axial brace's at the GOAL
    pose -, support z 0) with its ROOT xy at the cell and the support at ``height``."""
    T = S.pose_of(entry["T_rule_object"]).copy()
    T[0, 3], T[1, 3] = float(cell_xy[0]), float(cell_xy[1])
    T[2, 3] += float(height)
    return T


def chain_world(entry: Dict[str, Any], cell_xy, height: float) -> List[Tuple[str, np.ndarray]]:
    To = object_pose(entry, cell_xy, height)
    return [(n, To @ T) for n, T in chain_object(entry)]


def problems(grid: Dict[Tuple[int, int], Dict[str, Any]], cells: np.ndarray, height: float) -> Dict[str, Any]:
    """Every (cell, yaw, class) with an entry: ``{"index": (N, 3) (cell, yaw, class), "chain": (N, K, 4, 4) world eef poses (pre, contact,
    ... padded with the last), "n_chain": (N,), "fingers": (N,) the opening per finger (m; 0 closed), "To": (N, 4, 4)}``."""
    keys = sorted(grid)
    K = max((len(chain_object(grid[k])) for k in keys), default=2)
    rows, chains, nch, fq, tos = [], [], [], [], []
    for ci, xy in enumerate(cells):
        for k in keys:
            e = grid[k]
            To = object_pose(e, xy, height)
            ch = [To @ T for _n, T in chain_object(e)]
            nch.append(len(ch))
            ch = ch + [ch[-1]] * (K - len(ch))
            rows.append((ci, k[0], k[1]))
            chains.append(np.stack(ch))
            fq.append(float((e.get("fingers") or {}).get("opening_m") or 0.0))
            tos.append(To)
    return {"index": np.asarray(rows, dtype=np.int64).reshape(-1, 3), "chain": np.asarray(chains, dtype=float).reshape(-1, K, 4, 4),
            "n_chain": np.asarray(nch, dtype=np.int64), "fingers": np.asarray(fq, dtype=float), "To": np.asarray(tos, dtype=float).reshape(-1, 4, 4), "K": K}


def axial_end_spin(row: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """``(end_sign, spin)`` of an axial class row (``push_blade`` / ``axial_brace``): the brace's ``goal`` record, else the class label's
    ``[+-]end[+-]spin`` prefix (``contact_library.build_axial_push`` / ``build_axial_brace``: end +1 = the ``long_axis`` side); ``(None, None)``
    for a row that names neither."""
    if not row:
        return None, None
    g = row.get("goal") or {}
    if g.get("end_sign") is not None and g.get("spin") is not None:
        return float(g["end_sign"]), float(g["spin"])
    c = str(row.get("class", ""))
    if len(c) >= 9 and c[0] in "+-" and c[1:4] == "end" and c[4] in "+-" and c[5:9] == "spin":
        return (1.0 if c[0] == "+" else -1.0), (1.0 if c[4] == "+" else -1.0)
    return None, None


def entry_rest_quat(entry: Dict[str, Any]) -> List[float]:
    """The rest the entry's object stands on (xyzw): ``Rz(-yaw_deg) R(T_rule_object)`` - the library's v2 rest, an axial entry's identity
    rest (``contact_library.axial_rest``); a sampler's object yaw is read over it (:func:`yaw_of`)."""
    a = -math.radians(float(entry.get("yaw_deg", 0.0)))
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    q = S.mat_to_quat(Rz @ S.pose_of(entry["T_rule_object"])[:3, :3])
    q = np.asarray(q, dtype=float)
    return [round(float(v), 7) for v in (q if q[3] >= 0 else -q)]


def reach_back(entry: Dict[str, Any], pts_obj: np.ndarray) -> float:
    """The object's footprint reach toward the robot (most negative x, map frame) from its root at the entry's pose (the support fit at d_e)."""
    T = S.pose_of(entry["T_rule_object"])
    return float((pts_obj @ T[:3, :3].T)[:, 0].min())


# ------------------------------------------------------------------------------------------------ the computation (Isaac Sim / cuRobo)


def contact_config(kind: str, **kw) -> RM.MapConfig:
    """The map configuration of a kind: the arm of the kind, the floor plane's lattice for ``floor_push`` (``reach_maps.floor_config``), the
    wall map's stand-offs / lateral lattice / idle posture for ``wall_brace`` (:func:`wall_config`)."""
    if KINDS[kind]["arm"] is not None:
        kw.setdefault("arm", KINDS[kind]["arm"])
    if KINDS[kind]["plane"] == "wall":
        return wall_config(**kw)
    return RM.floor_config(**kw) if KINDS[kind]["plane"] == "floor" else RM.MapConfig(**kw)


def wall_config(**kw) -> RM.MapConfig:
    """i7-brace: the wall map's configuration (:data:`WALL_EDGE_DISTANCES`, :data:`WALL_LATTICE_M`, :data:`WALL_Y_HALF_M`,
    :data:`WALL_POSTURES`; one yaw slot); ``lattice_m`` / ``seeds`` / ``arm`` from the caller."""
    kw = dict(kw)
    kw.pop("lattice_m", None)  # the lateral lattice is the stance lattice's
    cfg = {"edge_distances": WALL_EDGE_DISTANCES, "lattice_m": WALL_LATTICE_M, "y_half_m": WALL_Y_HALF_M, "postures": WALL_POSTURES, "n_yaws": 1,
           "x_min_m": 0.0, "x_max_m": max(WALL_EDGE_DISTANCES), **kw}
    return RM.MapConfig(**cfg)


def wall_root_x(entry: Dict[str, Any]) -> float:
    """The hung target's root x relative to the wall plane (rule frame: the wall at x = 0, the root in front of it, x < 0)."""
    return float(S.pose_of(entry["T_rule_object"])[0, 3])


def wall_cells(cfg: RM.MapConfig, root_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(cells (C, 2), d_e index (C,))`` of a wall map: per stand-off d_e the root at x = d_e + ``root_x`` (the wall at x = d_e), laterals
    ``-y_half_m .. y_half_m`` every ``lattice_m``."""
    ys = np.round(np.arange(-cfg.y_half_m, cfg.y_half_m + 1e-9, cfg.lattice_m), 6)
    cells = np.concatenate([np.stack([np.full(len(ys), float(d) + float(root_x)), ys], 1) for d in cfg.edge_distances])
    idx = np.concatenate([np.full(len(ys), k, dtype=np.int64) for k in range(len(cfg.edge_distances))])
    return cells, idx


def wall_support(d_e: float) -> CP.Support:
    """The wall of a wall map at stand-off ``d_e``: a box :data:`WALL_BOX_M` behind the plane x = d_e (the base at the origin facing it), from the
    floor up."""
    sx, sy, sz = WALL_BOX_M
    return CP.Support.box([float(d_e) + 0.5 * sx, 0.0, 0.5 * sz], [0.5 * sx, 0.5 * sy, 0.5 * sz], name="wall")


def identity_meta(lib_k: Dict[str, Any], kind: str, height: float, cfg: RM.MapConfig, policy: CP.ContactPolicy, entries: Sequence[Dict[str, Any]],
                  verdicts: Optional[Sequence[str]], robot: Optional[str] = None) -> Dict[str, Any]:
    """The fields :func:`map_key` reads, known before the map is computed (the overlay's LINEAGE key is the key of the map built from
    ``entries`` alone)."""
    spec = KINDS[kind]
    return {"schema": SCHEMA, "kind": kind, "arm": cfg.arm, "plane": spec["plane"], "edge_distances": list(cfg.edge_distances), "config": cfg.key(),
            "policy_key": policy.key(cfg.arm), "robot": robot if robot is not None else RM.robot_digest(), "object": dict(lib_k.get("object") or {}),
            "height": {"value": float(height)}, "pose_set": pose_set(kind, entries, verdicts)}


def overlay_entries(lib_k: Dict[str, Any], kind: str, height: float, cfg: RM.MapConfig, policy: CP.ContactPolicy, entries: List[Dict[str, Any]],
                    verdicts: Optional[Sequence[str]], overlay: Optional[Dict[str, Any]], robot: Optional[str] = None
                    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Owner A6: the map's entries with the learned overlay (:mod:`contact_overlay`): ``overlay = {"root", "version"?}`` names the merged
    overlay of the map's LINEAGE (the key of the map of ``entries`` alone); every library entry it confirmed (a real oracle success)
    joins. Returns ``(entries, overlay meta or None)`` - None when no merged version exists (the map is the lineage's frozen map)."""
    from ispatialgym.sampling import contact_overlay as CO

    if not overlay or not overlay.get("root"):
        return entries, None
    lin = map_key(identity_meta(lib_k, kind, height, cfg, policy, entries, verdicts, robot=robot))
    conf, v = CO.confirmed(overlay["root"], lin, overlay.get("version"))
    if v <= 0:
        return entries, None
    out, joined = CO.join_entries(lib_k.get("entries") or [], entries, conf, KINDS[kind]["families"])
    return out, {"lineage": lin, "version": int(v), "root": overlay["root"], "joined": joined, "n_confirmed": len(conf)}


def compute_contact_map(world: RM.SynthWorld, lib_k: Dict[str, Any], kind: str, height: float, anchor: Dict[str, Any], geom, *,
                        cfg: Optional[RM.MapConfig] = None, policy: CP.ContactPolicy = CP.DEFAULT,
                        verdicts: Optional[Sequence[str]] = DEFAULT_VERDICTS, overlay: Optional[Dict[str, Any]] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One contact map: ``(meta, arrays)`` (module docstring). ``geom``: the object's collision pieces (the support fit, the arm-vs-object
    check). Raises ``ValueError`` when the library has no entry of the kind under ``verdicts``. ``overlay`` (owner A6): the entries a merged
    overlay version confirmed join (:func:`overlay_entries`); the version enters the identity."""
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    spec = KINDS[kind]
    floor = spec["plane"] == "floor"
    wall = spec["plane"] == "wall"
    plane_kind = "floor" if floor else "tabletop"
    if spec["arm"] is not None and cfg.arm != spec["arm"]:
        raise ValueError(f"a {kind} map is the {spec['arm']} arm's (config arm {cfg.arm})")
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    entries = select_entries(lib_k, kind, verdicts, arm=cfg.arm)
    if not entries:
        raise ValueError(f"{(lib_k.get('object') or {}).get('entry')}: no {kind} entry with a verdict in {verdicts}")
    entries, ov_meta = overlay_entries(lib_k, kind, height, cfg, policy, entries, verdicts, overlay)
    yaws, classes, grid = axes(entries)
    pset = pose_set(kind, entries, verdicts)
    if wall:  # i7-brace: per stand-off the root's row in front of the wall (one target pose per library: its level pose)
        rxs = {round(wall_root_x(e), 6) for e in entries}
        if len(rxs) != 1:
            raise ValueError(f"a wall_brace map's entries hang at one level pose (root x {sorted(rxs)})")
        cells, cell_de = wall_cells(cfg, rxs.pop())
    else:
        cells, cell_de = RM.lattice(cfg), None
    nC, nY, nG = len(cells), len(yaws), len(classes)
    pr = problems(grid, cells, height)
    N, K = len(pr["index"]), pr["K"]
    pts = np.asarray(geom.union.vertices, dtype=float)
    ground_links = sorted(getattr(world.backend, "ground_contact_links", ()) or ()) if floor else []
    chain = th.as_tensor(pr["chain"], dtype=th.float32, device=dev)  # (N, K, 4, 4)
    n_chain = th.as_tensor(pr["n_chain"], device=dev)
    fq = th.as_tensor(pr["fingers"], dtype=th.float32, device=dev)
    ci, yi, gi = (th.as_tensor(pr["index"][:, j], device=dev) for j in range(3))
    rb = {k: reach_back(e, pts) for k, e in grid.items()}
    rb_n = th.as_tensor([rb[(int(y), int(g))] for y, g in pr["index"][:, 1:]], dtype=th.float32, device=dev)
    cx = th.as_tensor(cells[:, 0], dtype=th.float32, device=dev)[ci]
    # -- the hand vs the support per d_e (IK-independent): the approach pre -> contact, every chain pose, a straight push's stroke
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)

    def hand_ok_at(T, sb):
        out = th.ones(T.shape[0], dtype=th.bool, device=dev)
        vals = th.round(fq * 1e5) / 1e5
        for v in th.unique(vals).tolist():
            m = vals == v
            out[m] = hc.check(T[m], sb, (float(v), float(v)), floor=floor)[0]
        return out

    paths = [CP.approach_path(chain[:, 0], chain[:, 1], policy.approach_step_m * 2)]
    has_stroke = (n_chain == 3) & th.as_tensor([not bool(grid[(int(y), int(g))].get("arc")) for y, g in pr["index"][:, 1:]], device=dev)
    if K >= 3:
        paths.append(CP.approach_path(chain[:, 1], chain[:, 2], policy.approach_step_m * 2))
    hand_ok: Dict[float, th.Tensor] = {}
    onsup: Dict[float, th.Tensor] = {}
    def support_at(d_e):
        return wall_support(d_e) if wall else RM.map_support(plane_kind, height, d_e, cfg)

    cde = None if cell_de is None else th.as_tensor(cell_de, device=dev)[ci]
    for k_de, d_e in enumerate(cfg.edge_distances):
        sb = support_at(d_e)
        ok = th.ones(N, dtype=th.bool, device=dev)
        for k in range(K):
            ok &= hand_ok_at(chain[:, k], sb)
        for p in paths[0]:
            ok &= hand_ok_at(p, sb)
        if len(paths) > 1:
            for p in paths[1]:
                ok &= hand_ok_at(p, sb) | ~has_stroke
        hand_ok[d_e] = ok
        # a wall map: the problems of this stand-off's row only (the target hangs against the wall at d_e)
        onsup[d_e] = (cde == k_de) if wall else (th.ones(N, dtype=th.bool, device=dev) if floor else (cx + rb_n) >= d_e)
    del paths
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    # -- the arm-vs-object planes of the object at each problem's pose
    pieces = [(th.as_tensor(pc.face_normals, dtype=th.float32, device=dev), th.as_tensor(np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0]),
                                                                                         dtype=th.float32, device=dev)) for pc in geom.pieces]
    To_t = th.as_tensor(pr["To"], dtype=th.float32, device=dev)
    hand_links = set(policy.finger_links(cfg.arm)) | set(policy.hand_rigid_links(cfg.arm))
    entry_ik = world.entry()
    links = world.link_of_sphere()
    arm_mask = th.tensor([(l not in hand_links) and bool(l) and l.startswith(f"{cfg.arm}_arm") for l in links], device=dev)
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.radians(np.asarray(yaws, dtype=np.float32)),
                                     "present": np.zeros((nY, nG), dtype=bool)}
    if cell_de is not None:
        arrays["cell_de"] = cell_de.astype(np.int16)
    for (y, g) in grid:
        arrays["present"][y, g] = True
    meta_trunks: List[Dict[str, Any]] = []
    t_ik = 0.0
    n_ik = 0

    def spheres_ok(sph, ln, sb):
        return CP.sphere_support_ok(sph, ln, sb, cfg.arm, policy, ground_contact_links=ground_links, floor_support=floor)

    def put(vals):
        st = np.full((nC, nY, nG), STATUS["no_entry"], dtype=np.uint8)
        v = vals.cpu().numpy()
        st[pr["index"][:, 0], pr["index"][:, 1], pr["index"][:, 2]] = v
        return st

    for tr, ti in trunk_grid(height):
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        posture, q_lock = None, None
        de_free: Dict[float, bool] = {}
        for p in cfg.postures:
            q = world.q_config(tr, ti, p)
            if bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                continue
            sph, ln = world.backend.robot_spheres_world(q.unsqueeze(0))
            free = {d_e: bool(spheres_ok(sph, ln, support_at(d_e))[0][0]) for d_e in cfg.edge_distances}
            if wall and any(free.values()):
                # i7-brace: per stand-off (the wall close ahead blocks a posture at the small d_e only): ineligible where it collides
                posture, q_lock, de_free = p, q, free
                break
            if all(free.values()):
                posture, q_lock = p, q
                break
        rec["posture"] = posture
        if q_lock is None:
            rec.update(eligible=False, reason="every idle-arm posture collides (self or the support at some d_e)")
            for d_e in cfg.edge_distances:
                arrays[f"s{d_e:.2f}_{key}"] = put(th.full((N,), STATUS["trunk_ineligible"], dtype=th.uint8))
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        in_reach = (th.linalg.norm(chain[:, :, :3, 3] - sh_t, dim=2) <= L + cfg.reach_margin_m).all(dim=1)
        rec.update(shoulder=[round(float(v), 4) for v in sh], arm_len=round(L, 4))
        any_ok = th.zeros(N, dtype=th.bool, device=dev)
        for d_e in cfg.edge_distances:
            any_ok |= hand_ok[d_e] & onsup[d_e]
        idx = th.nonzero(in_reach & any_ok).flatten()
        rec["n_problems"] = int(idx.numel())
        # the chain's IK: the pre-contact with random seeds, then each next pose seeded from the previous one (same branch)
        ta = time.time()
        qs: List[th.Tensor] = []
        alive = th.ones(idx.numel(), dtype=th.bool, device=dev)
        why = th.full((idx.numel(),), STATUS["ik_pre"], dtype=th.uint8, device=dev)
        succ, q0 = RM._solve(world, entry_ik, chain[idx, 0], q_lock)
        n_ik += int(idx.numel())
        first = th.argmax(succ.int(), dim=1)
        alive &= succ.any(dim=1)
        qs.append(q0[th.arange(idx.numel(), device=dev), first])
        for k in range(1, K):
            sel = th.nonzero(alive).flatten()
            qk = qs[-1].clone()
            if sel.numel():
                need = n_chain[idx[sel]] > k
                s2, q2 = RM._solve(world, entry_ik, chain[idx[sel], k], q_lock, seed_active=qs[-1][sel])
                n_ik += int(sel.numel())
                f2 = th.argmax(s2.int(), dim=1)
                q2 = q2[th.arange(sel.numel(), device=dev), f2]
                ok2 = s2.any(dim=1)
                jump = (q2 - qs[-1][sel]).abs().max(dim=1).values > cfg.branch_max_rad
                fail = need & (~ok2 | jump)
                why[sel[fail & ~ok2]] = STATUS["ik_grasp"]
                why[sel[fail & ok2 & jump]] = STATUS["branch_jump"]
                alive[sel[fail]] = False
                qk[sel] = th.where(need.unsqueeze(1), q2, qs[-1][sel])
            qs.append(qk)
        t_ik += time.time() - ta
        # per d_e: the whole robot's spheres clear of the support at the chain's configurations and the interpolated approach ones; the arm
        # (not the hand) clear of the object at the contact
        sel = th.nonzero(alive).flatten()
        for d_e in cfg.edge_distances:
            sb = support_at(d_e)
            good = th.ones(sel.numel(), dtype=th.bool, device=dev)
            w = th.where(alive, th.full_like(why, STATUS["feasible"]), why)
            if sel.numel():
                confs = [qs[0][sel] * (1 - t) + qs[1][sel] * t for t in [k / (cfg.approach_configs + 1) for k in range(1, cfg.approach_configs + 1)]]
                confs += [q[sel] for q in qs]
                for qq in confs:
                    for c0 in range(0, sel.numel(), 8192):
                        o, _ = spheres_ok(world.spheres(world.full_q(entry_ik, qq[c0:c0 + 8192], q_lock)), links, sb)
                        good[c0:c0 + 8192] &= o.to(dev)
                w_slab = ~good
                o = th.ones(sel.numel(), dtype=th.bool, device=dev)
                for c0 in range(0, sel.numel(), 2048):
                    sp = world.spheres(world.full_q(entry_ik, qs[1][sel[c0:c0 + 2048]], q_lock))[:, arm_mask]
                    Tn = To_t[idx[sel[c0:c0 + 2048]]]
                    Po = th.einsum("nji,nsj->nsi", Tn[:, :3, :3], sp[..., :3] - Tn[:, None, :3, 3])
                    deep = th.full(Po.shape[:2], float("inf"), device=dev)
                    for nrm, dd in pieces:
                        deep = th.minimum(deep, (Po @ nrm.T - dd).max(dim=-1).values)
                    o[c0:c0 + 2048] = (((deep - sp[..., 3]) >= 0.0) | (sp[..., 3] <= 0)).all(dim=1)
                w_sel = th.full((sel.numel(),), STATUS["feasible"], dtype=th.uint8, device=dev)
                w_sel = th.where(~o, th.full_like(w_sel, STATUS["object_arm"]), w_sel)
                w_sel = th.where(w_slab, th.full_like(w_sel, STATUS["slab_spheres"]), w_sel)
                w[sel] = w_sel
            vals = th.full((N,), STATUS["unevaluated"], dtype=th.uint8, device=dev)
            vals[~in_reach] = STATUS["reach_bound"]
            vals[in_reach & ~hand_ok[d_e]] = STATUS["hand_policy"]
            vals[~onsup[d_e]] = STATUS["off_support"]
            here = hand_ok[d_e][idx] & onsup[d_e][idx]
            vals[idx] = th.where(here, w, vals[idx])
            if de_free and not de_free.get(d_e, True):
                vals[onsup[d_e]] = STATUS["trunk_ineligible"]
            arrays[f"s{d_e:.2f}_{key}"] = put(vals)
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int((vals == STATUS["feasible"]).sum())
        meta_trunks.append(rec)
    tim.update(ik_s=round(t_ik, 2), ik_problems=n_ik, total_s=round(time.time() - t_all, 2))
    obj = dict(lib_k.get("object") or {})
    meta = {"schema": SCHEMA, "kind": kind, "arm": cfg.arm, "plane": spec["plane"], "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "cell": f"the object ROOT xy at its {spec['frame']} pose" + (" hung at the wall x = d_e (its d_e row: arrays cell_de), the hinge at the height" if wall else ""),
            "object": obj, "rest_id": obj.get("rest_id"),
            "entry_rest_quat_xyzw": entry_rest_quat(entries[0]),
            "library": {"digest": lib_k.get("digest"), "validate": lib_k.get("validate"), "families": sorted(lib_k.get("families") or {}),
                        # the builders' params (an axial library's long_axis: the side its +end classes push) - the consumer checks its own axis
                        "params": {f: (v or {}).get("params") for f, v in (lib_k.get("families") or {}).items() if f in spec["families"]},
                        # the pose rule's knobs as the library resolved them (i7-contactbrace 2026-11-08: a touch_brace map's brace_full_open -
                        # the consumer keeps a map built under the opening rule it runs, :func:`library_knob`); not in the map key
                        "knobs": {f: (v or {}).get("knobs") for f, v in (lib_k.get("families") or {}).items() if f in spec["families"]}},
            "height": {"value": float(height), "anchor_key": anchor.get("key"), "anchor_surfaces": list(anchor.get("surfaces", []))[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "robot": RM.robot_digest(),
            "pose_set": pset, "classes": [class_row(grid[(y, g)]) if (y, g) in grid else None for y in range(nY) for g in range(nG)],
            "chain": "the executed pre -> contact (+ a straight push's end / an arc's mid and end): IK seeded along the chain, branch <= branch_max_rad",
            "trunks": meta_trunks, "status_codes": STATUS, "timing": tim, "created": S.now()}
    if ov_meta is not None:
        meta["overlay"] = ov_meta
    meta["key"] = map_key(meta)
    return meta, arrays


# ------------------------------------------------------------------------------------------------ identity, files, reading


def object_key(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: obj.get(k) for k in ("entry", "category", "model", "scale", "rest_id", "geometry")}


def library_knob(meta: Dict[str, Any], key: str, default: Any = None) -> Any:
    """A pose-rule knob of the library a map was built from (``meta["library"]["knobs"][family][key]``, the first of the map's families
    that records it); ``default`` for a map built before the knobs were recorded (2026-11-08) - those libraries ran every rule's default."""
    for fam in sorted(((meta.get("library") or {}).get("knobs") or {}).items()):
        kn = fam[1] or {}
        if key in kn:
            return kn[key]
    return default


#: kind -> the library knobs a consumer requires to match the rule it runs, with the value a map that does not record them was built under
#: (i7-contactbrace 2026-11-08: a touch_brace map's entries carry the wide stop's opening - ``ISPATIALGYM_BRACE_FULL_OPEN`` changes it, so a
#: map built under the other rule proposes braces the oracle does not execute)
CONSUMER_KNOBS: Dict[str, Dict[str, Any]] = {"touch_brace": {"brace_full_open": False}}


def map_key(meta: Dict[str, Any]) -> str:
    """The map identity (module docstring): kind, arm, plane, object, height, edge distances, config, policy, robot, pose set."""
    from ispatialgym.sampling.v2 import contact_poses as CPZ

    ident = {k: meta[k] for k in ("schema", "kind", "arm", "plane", "edge_distances", "config", "policy_key", "robot")}
    ident.update(object=object_key(meta["object"]), height=meta["height"]["value"], pose_set=meta["pose_set"]["digest"])
    if meta.get("overlay"):  # owner A6: a learned map's lineage + merged overlay version (a frozen map's key is unchanged)
        ident["overlay"] = {"lineage": meta["overlay"]["lineage"], "version": int(meta["overlay"]["version"])}
    return CPZ.digest(ident)


def save(out_dir: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    d = os.path.join(out_dir, str(meta["object"].get("entry")))
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"cmap__{meta['kind']}__{meta['object'].get('category')}__{meta['object'].get('model')}__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    S.save_json(base + ".json", meta)
    return base + ".json"


def yaw_of(quat_xyzw, rest_quat_xyzw) -> float:
    """The object's yaw about z: ``R = Rz(yaw) R_rest`` (the library's yaw slot convention, ``contact_library.object_frame``)."""
    R = S.quat_to_mat(np.asarray(quat_xyzw, dtype=float)) @ S.quat_to_mat(np.asarray(rest_quat_xyzw, dtype=float)).T
    return math.atan2(R[1, 0], R[0, 0])


class ContactMap:
    """One contact map (``.json`` + ``.npz``): status per (edge distance, trunk) of (cell, yaw slot, class); :meth:`lookup` at a stance."""

    def __init__(self, path_json: str):
        self.path = path_json
        with open(path_json) as f:
            self.meta = json.load(f)
        if self.meta.get("schema") != SCHEMA:
            raise S.IncompatibleRecord(f"{path_json}: schema {self.meta.get('schema')!r} is not {SCHEMA}")
        z = np.load(path_json[:-5] + ".npz")
        self.arrays = {k: z[k] for k in z.files}
        self.cells, self.yaws = self.arrays["cells"], self.arrays["yaws"]
        self.classes = list(self.meta["pose_set"]["classes"])
        self.lookup_how: Optional[Dict[str, Any]] = None

    @staticmethod
    def find(maps_root: str, kind: str, entry: str, height: float, *, rest_id: Optional[str] = None, arm: Optional[str] = None,
             policy_key: Optional[str] = None, robot: Optional[str] = None, anchors=None, config_require: Optional[Dict[str, Any]] = None,
             pose_set_digest: Optional[str] = None, knobs_require: Optional[Dict[str, Any]] = None) -> Optional["ContactMap"]:
        """The ``kind`` map of the library entry ``entry`` (the v2 model directory id) at the anchor covering ``height`` (``anchors``: exact or
        within the anchor's range; None: the map's own height within 1 mm), the arm (default the kind's), the rest, the contact policy, the
        robot model, the required config and pose set, and ``knobs_require`` (knob -> value: :func:`library_knob`, a map without the record
        read at :data:`CONSUMER_KNOBS`' value of the kind); None when none matches."""
        arm = arm or KINDS[kind]["arm"]
        target, how = float(height), "height"
        if anchors is not None:
            a, how = anchors.lookup(height)
            if a is None:
                return None
            target = float(a["height"])
        d = os.path.join(maps_root, entry)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not (fn.startswith(f"cmap__{kind}__") and fn.endswith(".json")):
                continue
            p = os.path.join(d, fn)
            try:
                with open(p) as f:
                    m = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if m.get("schema") != SCHEMA or m.get("kind") != kind or m.get("arm") != arm or (m.get("object") or {}).get("entry") != entry:
                continue
            if rest_id is not None and m.get("rest_id") != rest_id:
                continue
            if policy_key is not None and m.get("policy_key") != policy_key:
                continue
            if robot is not None and m.get("robot") != robot:
                continue
            if config_require and any(RM.config_value(m, k) != v for k, v in config_require.items()):
                continue
            if pose_set_digest is not None and (m.get("pose_set") or {}).get("digest") != pose_set_digest:
                continue
            if knobs_require and any(library_knob(m, k, CONSUMER_KNOBS.get(kind, {}).get(k)) != v for k, v in knobs_require.items()):
                continue
            if abs(float(m["height"]["value"]) - target) > (1e-6 if anchors is not None else 1e-3):
                continue
            cm = ContactMap(p)
            cm.lookup_how = {"how": how, "asked_h": round(float(height), 4), "map_h": float(m["height"]["value"])}
            return cm
        return None

    @staticmethod
    def find_near(maps_root: str, kind: str, entry: str, height: float, tol: float, **kw) -> Optional["ContactMap"]:
        """i7-brace: :meth:`find` at the ``kind`` map heights of ``entry`` within ``tol`` of ``height``, the nearest first (a wall map's height
        is the hinge's above the floor - no support anchor covers it)."""
        d = os.path.join(maps_root, entry)
        hs = []
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.startswith(f"cmap__{kind}__") and fn.endswith(".json"):
                try:
                    hs.append(float(fn.split("__h", 1)[1].split("__", 1)[0]))
                except (IndexError, ValueError):
                    continue
        for h in sorted(set(hs), key=lambda v: abs(v - float(height))):
            if abs(h - float(height)) > tol:
                break
            cm = ContactMap.find(maps_root, kind, entry, h, **kw)
            if cm is not None:
                cm.lookup_how = {"how": "near", "asked_h": round(float(height), 4), "map_h": h, "tol": tol}
                return cm
        return None

    def trunks(self) -> List[Tuple[float, float]]:
        return [(t["translate"], t["tilt"]) for t in self.meta["trunks"] if t.get("eligible")]

    def nearest_trunk(self, translate: float, tilt: float) -> Optional[Tuple[float, float]]:
        """The map trunk of the stance's: the same translate (2 cm), the nearest tilt."""
        ts = [t for t in self.trunks() if abs(t[0] - float(translate)) <= 0.02]
        return min(ts, key=lambda t: abs(t[1] - float(tilt))) if ts else None

    def status(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        d = min(self.meta["edge_distances"], key=lambda v: abs(v - edge_distance))
        return self.arrays[f"s{d:.2f}_t{translate:.2f}_k{tilt:+.1f}"]

    def class_rows(self, yaw_index: int) -> List[Optional[Dict[str, Any]]]:
        nG = len(self.classes)
        return self.meta["classes"][yaw_index * nG:(yaw_index + 1) * nG]

    def lookup(self, stand_off: float, translate: float, tilt: float, root_xy_base, yaw_rel: float, *, keep=None, tol: float = 0.06,
               yaw_tol: Optional[float] = None) -> Dict[str, Any]:
        """The map at a stance (``stand_off`` = the nearest edge distance, the trunk: :meth:`nearest_trunk`) for the object's root at
        ``root_xy_base`` (base frame) turned ``yaw_rel`` from the stance: the nearest cell (within ``tol``), the nearest yaw slot (within
        ``yaw_tol``, default half the slot spacing), the classes there (``keep(row)`` filters them: e.g. the spin of an axial push) that are
        feasible. ``{"ok", "why", "cell", "yaw_slot_deg", "yaw_err_deg", "feasible": [classes], "trunk", "edge_distance"}``."""
        out: Dict[str, Any] = {"ok": False, "map": self.path, "kind": self.meta["kind"]}
        tk = self.nearest_trunk(translate, tilt)
        if tk is None:
            return dict(out, why=f"no eligible map trunk at translate {translate:.2f}")
        xy = np.asarray(root_xy_base, dtype=float)[:2]
        dist = np.linalg.norm(self.cells.astype(float) - xy[None], axis=1)
        if "cell_de" in self.arrays:  # i7-brace: a wall map - the cells of the stand-off's own row (the wall at that d_e)
            k_de = int(np.argmin([abs(v - float(stand_off)) for v in self.meta["edge_distances"]]))
            dist = np.where(self.arrays["cell_de"].astype(int) == k_de, dist, np.inf)
        c = int(np.argmin(dist))
        if dist[c] > tol:
            return dict(out, why=f"root {np.round(xy, 3).tolist()} off the lattice (nearest cell {dist[c]:.3f} m)", trunk=list(tk))
        yaws = self.yaws.astype(float)
        dyaw = np.abs(np.arctan2(np.sin(yaws - yaw_rel), np.cos(yaws - yaw_rel)))
        y = int(np.argmin(dyaw))
        ytol = yaw_tol if yaw_tol is not None else (math.pi / max(len(yaws), 1) if len(yaws) > 1 else math.pi)
        out.update(cell=[round(float(v), 3) for v in self.cells[c]], yaw_slot_deg=round(math.degrees(float(yaws[y])), 3),
                   yaw_err_deg=round(math.degrees(float(dyaw[y])), 3), trunk=list(tk))
        if dyaw[y] > ytol + 1e-9:
            return dict(out, why=f"yaw {math.degrees(yaw_rel):.1f} deg is {math.degrees(float(dyaw[y])):.1f} deg from the nearest slot")
        d_e = min(self.meta["edge_distances"], key=lambda v: abs(v - float(stand_off)))
        st = self.status(d_e, tk[0], tk[1])[c, y]
        rows = self.class_rows(y)
        feas = [self.classes[g] for g in range(len(self.classes)) if st[g] == STATUS["feasible"] and rows[g] is not None and (keep is None or keep(rows[g]))]
        out.update(edge_distance=d_e, feasible=feas)
        if not feas:
            return dict(out, why=f"no feasible class at the cell / yaw slot ({int((st == STATUS['feasible']).sum())} before the class filter)")
        return dict(out, ok=True, why=None)


# ------------------------------------------------------------------------------------------------ CLI


def _shard_ids(ids: List[str], shard: Optional[str]) -> List[str]:
    if not shard:
        return ids
    i, n = (int(v) for v in shard.split("/"))
    return [x for k, x in enumerate(ids) if k % n == i]


def library_ids(root: str, lib_kind: str) -> List[str]:
    base = os.path.join(root, lib_kind, "models")
    return sorted(d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, "library.json"))) if os.path.isdir(base) else []


def _compute_cli(a) -> None:
    from ispatialgym.grasping import contact_library as CL
    from ispatialgym.grasping import object_geometry as OG

    spec = KINDS[a.kind]
    ids = _shard_ids([i for i in library_ids(a.root, spec["lib_kind"]) if not a.only or i in a.only], a.shard)
    verdicts = None if a.verdicts == "all" else tuple(a.verdicts.split(","))
    log_path = os.path.join(a.out, "logs", f"maps_{a.kind}_{(a.shard or '0/1').replace('/', 'of')}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(rec, t=round(time.time(), 1))) + "\n")

    todo = []
    arms = [None] if spec["arm"] is not None else ["left", "right"]  # i7-brace: a per-entry-arm kind maps each arm's entries
    for eid in ids:
        lib_k = CL.load_library(CL.library_path(a.root, spec["lib_kind"], eid))
        if not any(select_entries(lib_k, a.kind, verdicts, arm=arm) for arm in arms):
            log({"id": eid, "kind": a.kind, "ok": True, "skipped": f"no {a.kind} entry with a verdict in {verdicts}"})
            continue
        hs = [float(h) for h in a.heights] if a.heights else ([RM.FLOOR_HEIGHT] if spec["plane"] == "floor" else
                                                             [float((lib_k.get("validate") or {}).get("support_z"))] if (lib_k.get("validate") or {}).get("support_z") is not None else [])
        if not hs:
            log({"id": eid, "kind": a.kind, "ok": False, "reason": "no height: pass --heights or validate.support_z in the library"})
            continue
        todo.append((eid, lib_k, hs))
    if not todo:
        print("GB_CONTACT_MAPS_DONE", flush=True)
        return
    cfg0 = contact_config(a.kind, lattice_m=a.lattice, seeds=a.seeds, **({} if spec["arm"] is not None else {"arm": "left"}))
    world = RM.SynthWorld(cfg0)
    import dataclasses

    for eid, lib_k, hs, arm in [(e, l, h, arm) for arm in arms for e, l, h in todo]:
        cfg = cfg0 if arm is None else dataclasses.replace(cfg0, arm=arm)
        if arm is not None and world.cfg.arm != arm:  # the synthetic world's IK entry / shoulder cache are the config arm's
            world.cfg = cfg
            world._shoulder = {}
        if arm is not None and not select_entries(lib_k, a.kind, verdicts, arm=arm):
            log({"id": eid, "kind": a.kind, "arm": arm, "ok": True, "skipped": f"no {a.kind} entry of the {arm} arm with a verdict in {verdicts}"})
            continue
        geom = OG.load(lib_k["object"]["source"])
        for h in hs:
            t0 = time.time()
            try:
                anchor = RM.FLOOR_ANCHOR if spec["plane"] == "floor" else {"key": f"h{h:.4f}", "height": h, "surfaces": [f"library validate.support_z ({eid})"]}
                meta, arrays = compute_contact_map(world, lib_k, a.kind, h, anchor, geom, cfg=cfg, verdicts=verdicts,
                                                   overlay={"root": a.overlay_root, "version": a.overlay_version} if a.overlay_root else None)
                p = save(a.out, meta, arrays)
                log({"id": eid, "kind": a.kind, "arm": cfg.arm, "h": h, "ok": True, "path": p, "s": round(time.time() - t0, 1), "n_entries": meta["pose_set"]["n_entries"],
                     "overlay": {k: (meta.get("overlay") or {}).get(k) for k in ("lineage", "version", "n_confirmed")} if meta.get("overlay") else None,
                     "n_joined": len((meta.get("overlay") or {}).get("joined") or []),
                     "n_feasible": sum(sum(t.get("n_feasible", {}).values()) for t in meta["trunks"])})
            except Exception as e:  # noqa: BLE001 - the next model goes on; the log names the failure
                log({"id": eid, "kind": a.kind, "h": h, "ok": False, "error": repr(e)[:500]})
    print("GB_CONTACT_MAPS_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ispatialgym.sampling.contact_maps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="the contact maps of a library root's objects")
    c.add_argument("--root", required=True, help="the contact library root (contact_library build / validate)")
    c.add_argument("--kind", required=True, choices=sorted(KINDS))
    c.add_argument("--out", required=True, help="the maps root (<out>/<entry>/cmap__*)")
    c.add_argument("--heights", nargs="*", type=float, default=None, help="default: the library's validate.support_z (floor_push: 0)")
    c.add_argument("--verdicts", default="valid", help="comma list of validation verdicts the entries need, or 'all'")
    c.add_argument("--shard", default=None)
    c.add_argument("--only", nargs="*", default=None)
    c.add_argument("--lattice", type=float, default=0.08)
    c.add_argument("--seeds", type=int, default=8)
    c.add_argument("--overlay-root", default=None, help="owner A6: the merged contact overlay root (<root>/<lineage key>/v<N>.json): the entries a "
                   "real oracle success confirmed join the validated ones; the version enters the map identity")
    c.add_argument("--overlay-version", type=int, default=None, help="the overlay version (default the latest)")
    i = sub.add_parser("info", help="a contact map's summary")
    i.add_argument("map")
    a = ap.parse_args(argv)
    if a.cmd == "compute":
        _compute_cli(a)
    else:
        m = ContactMap(a.map)
        print(json.dumps({k: m.meta[k] for k in ("kind", "arm", "plane", "object", "height", "edge_distances", "pose_set", "key")}, indent=1))
        print(json.dumps([{"trunk": t["key"], "n_feasible": t.get("n_feasible")} for t in m.meta["trunks"] if t.get("eligible")]))


if __name__ == "__main__":
    main()
