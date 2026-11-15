"""Replay-time scene perturbation for the kinematic batched replayer (:mod:`ispatialgym.multienv.replay`).

Owner request 2026-11-14 (paper deadline): the training set's scene perturbation (plan "perturbations / 1. scene perturbation":
remove non-structural scene objects, never task-relevant ones) is applied when the accepted demos are rendered by the batched
replay, for every class except camera relocalization. Demos are collected from canonical eval templates with lighting only.

Mechanism. The replay is kinematic: every tick loads the recorded state of every object into the slot's scene and renders; no
physics step runs. A removed object is therefore hidden (USD visibility of its prim: invisible) instead of parked: its recorded
state still loads every tick, so the scene state - and every quantity the task's success is judged on - is the recorded one
bit for bit, and only the rendered pixels change. Visibility is per prim, so each slot (one scene per env) is independent. The
next demo started in a slot first makes the slot's hidden objects visible again.

What may be removed (``plan``):
* candidates: the slot's scene objects that are not robots, not of a structural category (``training_scene_policy.STRUCTURAL``:
  walls, floors, ceilings, doors, windows, lights, ...), hold no USD light (a lamp: lighting is its own perturbation), and are
  not parked by the campaign's sampling-time scene removal;
* protected (never removed): the task's objects - every TRO ``object_bindings`` scene object (target, partner, support,
  distractors, articulated target, source / destination), the demo's ``tracked_objects`` and ``protected_objects`` -; every
  object whose recorded state changes during the demo (anything the robot or a task object moved); every object whose AABB
  (+ ``contact_margin``) touches, at any of ``contact_frames`` frames spread over the demo, the AABB of a non-structural task or
  moved object or of a robot link (the demo's supports, contacts and anything the robot brushes; the demo hdf5 records no
  contact reports, so contact is the conservative AABB proxy); and an object whose AABB is unknown;
* removal units: candidates whose frame-0 AABBs touch (``touch_margin``) form one unit (connected component) that is removed
  whole or not at all - nothing is left floating (a table goes with what stands on it) - and a unit with any protected member
  is protected whole.

The draw (``draw``) is deterministic in (seed, demo name, replay index): with probability ``p_episode`` the episode is
perturbed,
then every removable unit is removed with probability ``p_unit`` (the sampling-time ``removal_candidates`` scheme). Each
replayed episode's record (``REPLAY_SCENE_PERT``: removed names, units, protected set and why, key) is emitted with the demo
and can be replayed exactly with ``--scene-pert-from``.

No simulator import at module load (the pure parts are unit tested on CPU).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

SCHEMA = "isg-replay-scene-pert-v1"

# training_scene_policy.STRUCTURAL (the sampling-time removal's structural categories), copied so this module needs no scripts/ import
STRUCTURAL = frozenset((
    'walls', 'floors', 'ceilings', 'roof', 'baseboard', 'downlight',
    'room_light', 'electric_switch', 'door', 'sliding_door', 'garage_door',
    'fixed_window', 'window', 'skylight', 'stairs', 'staircase', 'railing',
))
#: building parts the replay also never hides (not in the sampling-time list; hiding them opens holes in the building)
EXTRA_STRUCTURAL = frozenset(('elevator_door', 'lintel', 'trim', 'molding', 'ceiling_light', 'wall_mounted_light'))

EXEMPT_FAMILIES = ("camera_relocalization",)

DEFAULTS = {"p_episode": 0.5, "p_unit": 0.5, "contact_margin": 0.02, "touch_margin": 0.01, "contact_frames": 40,
            "move_tol": 1e-4, "move_frames": 300, "offstage_m": 50.0}


def is_structural(category: Optional[str]) -> bool:
    return (category or "") in STRUCTURAL or (category or "") in EXTRA_STRUCTURAL


def exempt(meta: Mapping[str, Any]) -> bool:
    """Camera relocalization keeps its scene (owner: every class except camera relocalization)."""
    text = " ".join(str(meta.get(k) or "") for k in ("task_family", "variant", "task", "subtask", "activity"))
    return any(f in text for f in EXEMPT_FAMILIES)


def episode_key(seed: int, demo_name: str, replay_index: int) -> int:
    digest = hashlib.sha256(f"{SCHEMA}|{int(seed)}|{demo_name}|{int(replay_index)}".encode()).hexdigest()
    return int(digest[:16], 16)


# ----------------------------------------------------------------------------------------------------------------------------
# pure geometry / sets
# ----------------------------------------------------------------------------------------------------------------------------

def aabbs_touch(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray], margin: float) -> bool:
    (alo, ahi), (blo, bhi) = a, b
    return bool(np.all(np.asarray(alo) - margin <= np.asarray(bhi)) and np.all(np.asarray(blo) - margin <= np.asarray(ahi)))


def touching_any(box: Tuple[np.ndarray, np.ndarray], others: Sequence[Tuple[np.ndarray, np.ndarray]], margin: float) -> bool:
    if not len(others):
        return False
    lo = np.asarray([o[0] for o in others], dtype=float)
    hi = np.asarray([o[1] for o in others], dtype=float)
    blo, bhi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    hit = np.all(blo - margin <= hi, axis=1) & np.all(lo - margin <= bhi, axis=1)
    return bool(hit.any())


def aabb_gap(box: Tuple[np.ndarray, np.ndarray], others: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[float]:
    """Smallest Euclidean gap (m) between ``box`` and any of ``others`` (0 when overlapping); None without others."""
    if not len(others):
        return None
    lo = np.asarray([o[0] for o in others], dtype=float)
    hi = np.asarray([o[1] for o in others], dtype=float)
    blo, bhi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    sep = np.maximum(0.0, np.maximum(lo - bhi, blo - hi))
    return float(np.min(np.linalg.norm(sep, axis=1)))


def components(names: Sequence[str], boxes: Mapping[str, Tuple[np.ndarray, np.ndarray]], margin: float) -> List[List[str]]:
    """Connected components of ``names`` under AABB touch (``margin``); sorted members, components sorted by first member."""
    names = sorted(names)
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if aabbs_touch(boxes[a], boxes[b], margin):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def removable_units(candidates: Iterable[str], boxes: Mapping[str, Optional[Tuple[np.ndarray, np.ndarray]]],
                    protected: Mapping[str, str], margin: float) -> Tuple[List[List[str]], Dict[str, str]]:
    """(units, protected_with_reason): units = touch components of the candidates with a known AABB, a component with a protected
    member is protected whole (its unprotected members get reason ``unit:<protected member>``)."""
    protected = dict(protected)
    known = [n for n in candidates if boxes.get(n) is not None]
    for n in candidates:
        if boxes.get(n) is None and n not in protected:
            protected[n] = "aabb unknown"
    units = []
    for comp in components(known, boxes, margin):
        hit = next((n for n in comp if n in protected), None)
        if hit is None:
            units.append(comp)
        else:
            for n in comp:
                protected.setdefault(n, f"unit:{hit}")
    return units, protected


def draw(units: Sequence[Sequence[str]], key: int, p_episode: float, p_unit: float) -> Dict[str, Any]:
    """Deterministic removal draw for one replayed episode."""
    rng = random.Random(int(key))
    perturbed = rng.random() < float(p_episode)
    removed: List[str] = []
    chosen: List[int] = []
    if perturbed:
        for i, unit in enumerate(units):
            if rng.random() < float(p_unit):
                chosen.append(i)
                removed.extend(unit)
    return {"perturbed": bool(perturbed), "units_removed": chosen, "removed": sorted(removed)}


def flatten_state(value) -> np.ndarray:
    """One object's (nested dict of tensors / arrays / scalars) state as one float vector, keys in sorted order."""
    out: List[np.ndarray] = []

    def walk(v):
        if isinstance(v, Mapping):
            for k in sorted(v):
                walk(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        else:
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            try:
                out.append(np.asarray(v, dtype=np.float64).reshape(-1))
            except (TypeError, ValueError):
                pass

    walk(value)
    return np.concatenate(out) if out else np.zeros(0)


def moved_objects(frames: Sequence[Mapping[str, Any]], tol: float) -> Set[str]:
    """Objects whose flattened state differs (> ``tol`` max abs, or changes size / appears) from the first frame in any frame."""
    if not frames:
        return set()
    ref = {n: flatten_state(s) for n, s in frames[0].items()}
    moved: Set[str] = set()
    for fr in frames[1:]:
        for n, s in fr.items():
            if n in moved:
                continue
            v = flatten_state(s)
            r = ref.get(n)
            if r is None or r.shape != v.shape or (v.size and float(np.max(np.abs(v - r))) > tol):
                moved.add(n)
    return moved


def task_objects(meta: Mapping[str, Any], tro_bindings: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """{scene object name: reason} of the task's objects (TRO bindings, meta tracked / protected objects)."""
    out: Dict[str, str] = {}
    for inst, b in (tro_bindings or {}).items():
        name = b.get("scene_object_name") if isinstance(b, Mapping) else b
        if isinstance(name, str) and name:
            out.setdefault(name, f"tro_binding:{inst}")
    for field in ("tracked_objects", "protected_objects"):
        for inst, name in (meta.get(field) or {}).items():
            if isinstance(name, str) and name:
                out.setdefault(name, f"{field}:{inst}")
    return out


def sample_frames(T: int, n: int) -> List[int]:
    if T <= 0:
        return []
    n = max(2, min(int(n), T))
    return sorted({int(round(x)) for x in np.linspace(0, T - 1, n)})


def load_from_records(path: str) -> Dict[Tuple[str, int], List[str]]:
    """``--scene-pert-from``: (demo name, replay index) -> removed names of a previous run's REPLAY_SCENE_PERT lines."""
    out: Dict[Tuple[str, int], List[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("REPLAY_SCENE_PERT "):
                line = line[len("REPLAY_SCENE_PERT "):]
            rec = json.loads(line)
            if rec.get("tag", "REPLAY_SCENE_PERT") != "REPLAY_SCENE_PERT" or rec.get("schema") != SCHEMA:
                continue
            out[(rec["demo_name"], int(rec["replay_index"]))] = list(rec["removed"])
    return out


# ----------------------------------------------------------------------------------------------------------------------------
# simulator side
# ----------------------------------------------------------------------------------------------------------------------------

def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def object_aabb(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """World AABB from the object's collision boundary points (OmniGibson ``EntityPrim.aabb``); None when unknown."""
    try:
        lo, hi = obj.aabb
        lo, hi = _np(lo), _np(hi)
        if lo.shape == (3,) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi >= lo):
            return lo, hi
    except Exception:  # noqa: BLE001
        pass
    return None


def link_boxes(robot, radius: float = 0.12) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-link AABBs of the robot (collision points; a link without them: its origin +- ``radius``)."""
    boxes = []
    for link in robot.links.values():
        box = None
        try:
            pts = link.collision_boundary_points_world
            if pts is not None and len(pts):
                p = _np(pts)
                box = (p.min(axis=0), p.max(axis=0))
        except Exception:  # noqa: BLE001
            box = None
        if box is None:
            try:
                c = _np(link.get_position_orientation()[0])
                box = (c - radius, c + radius)
            except Exception:  # noqa: BLE001
                continue
        boxes.append(box)
    return boxes


def holds_light(obj) -> bool:
    stack = [obj.prim]
    while stack:
        prim = stack.pop()
        if "Light" in prim.GetTypeName():
            return True
        stack.extend(prim.GetChildren())
    return False


def _read_json(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    return {}


def demo_task_context(demo) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """(meta.json, TRO object_bindings, campaign parked objects) of a replay ``Demo``."""
    meta = _read_json(os.path.join(demo.dir, "meta.json"))
    tro = _read_json(getattr(demo, "tro_path", None) or meta.get("tro_path")).get("ispatialgym") or {}
    cmeta = getattr(demo, "campaign_meta", None) or {}
    bindings = tro.get("object_bindings") or cmeta.get("object_bindings") or {}
    parked = list(((cmeta.get("campaign") or tro.get("campaign") or {}).get("perturbations") or {}).get("removed_objects") or [])
    return meta, bindings, parked


def plan(scene, demo, replay_index: int, seed: int, cfg: Optional[Mapping[str, Any]] = None,
         forced: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Decide this replayed episode's removals in ``scene`` (a slot's scene). Loads sampled recorded states into the scene (the
    caller reloads frame 0 afterwards). ``forced``: the removed names of a previous record (``--scene-pert-from``), checked
    against this run's protection."""
    cfg = {**DEFAULTS, **(cfg or {})}
    meta, bindings, parked = demo_task_context(demo)
    rec: Dict[str, Any] = {"schema": SCHEMA, "demo": demo.dir, "demo_name": demo.name, "replay_index": int(replay_index),
                           "seed": int(seed), "cfg": cfg, "removed": [], "units": [], "perturbed": False}
    if exempt(meta):
        rec["exempt"] = "camera_relocalization"
        return rec
    robots = {r.name for r in scene.robots}
    objs = {o.name: o for o in scene.objects if o.name not in robots}
    protected: Dict[str, str] = {n: r for n, r in task_objects(meta, bindings).items() if n in objs}
    # recorded motion: every object whose recorded state changes during the demo
    step = max(1, demo.T // max(1, int(cfg["move_frames"])))
    idx = sorted(set(range(0, demo.T, step)) | {demo.T - 1})
    frames = []
    for i in idx:
        st, _ = scene.deserialize(demo.state_at(i))
        frames.append({n: s for n, s in st["registry"]["object_registry"].items() if n not in robots})
    for n in sorted(moved_objects(frames, float(cfg["move_tol"]))):
        if n in objs:
            protected.setdefault(n, "moved")
    candidates = []
    for n, o in sorted(objs.items()):
        if n in parked or is_structural(getattr(o, "category", None)):
            continue
        if holds_light(o):
            protected.setdefault(n, "holds light")
            continue
        candidates.append(n)
    # frame-0 boxes of the candidates (static unless moved; moved ones are protected)
    scene.load_state(demo.state_at(0), serialized=True)
    boxes = {n: object_aabb(objs[n]) for n in candidates}
    static = [n for n in candidates if n not in protected and boxes[n] is not None]
    dynamic = [n for n in protected if protected[n] != "holds light" and n in objs
               and not is_structural(getattr(objs[n], "category", None))]
    contact_frames = sample_frames(demo.T, int(cfg["contact_frames"]))
    margin = float(cfg["contact_margin"])
    hits: Dict[str, str] = {}
    gaps: Dict[str, List[Optional[float]]] = {n: [None, None] for n in static}  # min gap to the robot links / task objects
    n_robot_boxes: List[int] = []

    def _min(a, b):
        return b if a is None else (a if b is None else min(a, b))

    for i in contact_frames:
        if i != 0:
            scene.load_state(demo.state_at(i), serialized=True)
        dyn = [(n, object_aabb(objs[n])) for n in dynamic]
        dyn_boxes = [b for _, b in dyn if b is not None]
        rob_boxes = [b for r in scene.robots for b in link_boxes(r)]
        n_robot_boxes.append(len(rob_boxes))
        for n in static:
            gaps[n][0] = _min(gaps[n][0], aabb_gap(boxes[n], rob_boxes))
            gaps[n][1] = _min(gaps[n][1], aabb_gap(boxes[n], dyn_boxes))
            if n in hits:
                continue
            if touching_any(boxes[n], rob_boxes, margin):
                hits[n] = f"robot contact@{i}"
            elif touching_any(boxes[n], dyn_boxes, margin):
                hits[n] = f"task/moved contact@{i}"
    for n, why in hits.items():
        protected.setdefault(n, why)
    # offstage: an object farther than offstage_m from the robot at every sampled frame (a template's own far parking, e.g. chairs
    # parked ~200 m away) - hiding it changes no image, so it is neither a unit nor counted as a removal
    offstage = sorted(n for n in static if n not in protected and gaps[n][0] is not None and gaps[n][0] > float(cfg["offstage_m"]))
    candidates = [n for n in candidates if n not in set(offstage)]
    if not n_robot_boxes or min(n_robot_boxes) == 0:
        raise RuntimeError(f"scene perturbation: no robot link boxes for {demo.name} (robot contact cannot be protected)")
    units, protected = removable_units(candidates, boxes, protected, float(cfg["touch_margin"]))
    if forced is not None:
        bad = sorted(set(forced) - {n for u in units for n in u})
        if bad:
            raise ValueError(f"--scene-pert-from: {demo.name} r{replay_index}: recorded removals now protected or unknown: {bad}")
        d = {"perturbed": bool(forced), "units_removed": None, "removed": sorted(forced), "forced": True}
    else:
        d = draw(units, episode_key(seed, demo.name, replay_index), float(cfg["p_episode"]), float(cfg["p_unit"]))
    rec.update(d)
    rec.update({"units": units, "n_candidates": len(candidates), "n_objects": len(objs),
                "protected": dict(sorted(protected.items())), "contact_frames": len(contact_frames), "move_frames": len(idx),
                "parked_by_campaign": sorted(parked), "offstage": offstage, "robot_boxes_min": min(n_robot_boxes),
                "removed_min_gap_m": {n: {"robot": None if gaps.get(n, [None])[0] is None else round(gaps[n][0], 4),
                                          "task": None if gaps.get(n, [None, None])[1] is None else round(gaps[n][1], 4)}
                                      for n in rec.get("removed", [])}})
    return rec


def apply(scene, names: Sequence[str]) -> List[str]:
    """Hide ``names`` in ``scene``; returns the names hidden (a name the scene lacks raises)."""
    done = []
    for n in names:
        obj = scene.object_registry("name", n)
        if obj is None:
            raise ValueError(f"scene perturbation: {n} not in the slot's scene")
        obj.visible = False
        done.append(n)
    return done


def clear(scene, names: Sequence[str]) -> None:
    """Make the previous episode's hidden objects visible again (an object the scene no longer holds is skipped)."""
    for n in names:
        obj = scene.object_registry("name", n)
        if obj is not None:
            obj.visible = True


def visibility_matrix(scenes, names_per_slot: Sequence[Sequence[str]]) -> Dict[str, Any]:
    """Per slot k and every name removed in any slot: is it visible in slot k? Proof of independence: a name is invisible exactly
    in the slots that removed it."""
    union = sorted({n for names in names_per_slot for n in names})
    out = {"names": union, "visible": [], "independent": True}
    for k, scene in enumerate(scenes):
        row = []
        for n in union:
            obj = scene.object_registry("name", n)
            vis = None if obj is None else bool(obj.visible)
            row.append(vis)
            if obj is not None and vis == (n in names_per_slot[k]):
                out["independent"] = False
        out["visible"].append(row)
    return out


def pose_errors(scene, demo, i: int, tracked: Mapping[str, str]) -> Dict[str, float]:
    """Max position (m) / orientation (rad) error between the slot's tracked objects and the recorded ``isg/object_pose`` row i."""
    poses = getattr(demo, "_rsp_object_pose", None)
    if poses is None:
        import h5py

        poses = {}
        with h5py.File(os.path.join(demo.dir, "demo.hdf5"), "r") as f:
            g = f["data/demo_0"]
            if "isg/object_pose" in g:
                for inst in g["isg/object_pose"]:
                    poses[inst] = g["isg/object_pose"][inst][()]
        demo._rsp_object_pose = poses
    pe = oe = 0.0
    n = 0
    for inst, name in tracked.items():
        if inst not in poses:
            continue
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        p, q = obj.get_position_orientation(frame="scene")
        p, q = _np(p), _np(q)
        rp, rq = poses[inst][i, :3].astype(float), poses[inst][i, 3:7].astype(float)
        pe = max(pe, float(np.linalg.norm(p - rp)))
        dot = min(1.0, abs(float(np.dot(q / np.linalg.norm(q), rq / max(1e-12, np.linalg.norm(rq))))))
        oe = max(oe, 2.0 * math.acos(dot))
        n += 1
    return {"pos_m": pe, "ori_rad": oe, "n": n}


def trajectory_update(check: Dict[str, Any], scene, tracked: Mapping[str, str]) -> None:
    """--scene-pert-verify: fold this frame's tracked-object poses (scene frame) and the robots' joint positions into
    ``check["_traj"]`` (sha256): equal digests of a perturbed and an unperturbed replay of a demo = the same executed trajectory."""
    h = check.get("_traj")
    if h is None:
        h = check["_traj"] = hashlib.sha256()
    for inst in sorted(tracked):
        obj = scene.object_registry("name", tracked[inst])
        if obj is None:
            continue
        p, q = obj.get_position_orientation(frame="scene")
        h.update(np.round(np.concatenate([_np(p), _np(q)]), 7).astype(np.float64).tobytes())
    for r in scene.robots:
        h.update(np.round(_np(r.get_joint_positions()), 7).astype(np.float64).tobytes())


def probe_targets(names_per_slot: Sequence[Sequence[str]], limit: int = 3) -> List[str]:
    """Objects removed in some slots and kept in others (the ones a probe view can show as absent only where removed)."""
    k = len(names_per_slot)
    count: Dict[str, int] = {}
    for names in names_per_slot:
        for n in set(names):
            count[n] = count.get(n, 0) + 1
    both = sorted((n for n, c in count.items() if 0 < c < k), key=lambda n: (-min(count[n], k - count[n]), n))
    return both[:limit]


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """xyzw quaternion of a USD camera (looks along its -z, +y up) at ``eye`` looking at ``target`` (world z up)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= max(1e-9, np.linalg.norm(f))
    z = -f
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(m).as_quat()


def probe_views(scenes, hidden_per_slot: Sequence[Sequence[str]], render, out_dir: str, limit: int = 3,
                renders: int = 30) -> Dict[str, Any]:
    """--scene-pert-probe: for up to ``limit`` objects removed in some slots and kept in others, point every slot's head camera at
    the object (in that slot's scene), render, save the rgb per slot and measure how much the image differs between slots that
    removed it and slots that kept it. The cameras' parent-frame poses are restored afterwards (the replay's states are untouched)."""
    import torch as th
    from PIL import Image

    from ispatialgym.oracle.r1pro_model import HEAD_CAMERA_SENSOR_SUFFIX

    os.makedirs(out_dir, exist_ok=True)
    report: Dict[str, Any] = {"targets": [], "out_dir": out_dir}
    sensors = []
    for scene in scenes:
        robot = scene.robots[0]
        key = next(k for k in robot.sensors if k.endswith(HEAD_CAMERA_SENSOR_SUFFIX))
        sensors.append(robot.sensors[key])
    saved = [s.get_position_orientation(frame="parent") for s in sensors]
    try:
        for name in probe_targets(hidden_per_slot, limit):
            imgs = []
            for k, (scene, sensor) in enumerate(zip(scenes, sensors)):
                obj = scene.object_registry("name", name)
                lo, hi = object_aabb(obj) or (None, None)
                if lo is None:
                    p = _np(obj.get_position_orientation()[0])
                    lo, hi = p - 0.3, p + 0.3
                c = (lo + hi) / 2
                ext = float(np.max(hi - lo))
                rp = _np(scene.robots[0].get_position_orientation()[0])
                d = rp[:2] - c[:2]
                d = d / max(1e-6, np.linalg.norm(d))
                dist = max(1.5, 1.6 * ext)
                eye = np.array([c[0] + d[0] * dist, c[1] + d[1] * dist, c[2] + 0.6 + 0.3 * ext])
                q = look_at_quat(eye, c)
                sensor.set_position_orientation(th.as_tensor(eye, dtype=th.float32), th.as_tensor(q, dtype=th.float32))
            for _ in range(renders):
                render()
            for k, sensor in enumerate(sensors):
                obs = sensor.get_obs()
                obs = obs[0] if isinstance(obs, tuple) else obs
                rgb = obs["rgb"]
                a = (rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb))[..., :3].astype(np.uint8)
                removed = name in hidden_per_slot[k]
                Image.fromarray(a).save(os.path.join(out_dir, f"probe_{name}_slot{k}_{'REMOVED' if removed else 'kept'}.png"))
                imgs.append((k, removed, a.astype(np.int16)))
            kept = [a for _, r, a in imgs if not r]
            rows = []
            for k, removed, a in imgs:
                ref = next((b for kk, r, b in imgs if not r and kk != k), None) if kept else None
                frac = None if ref is None else float((np.abs(a - ref).max(axis=2) > 25).mean())
                rows.append({"slot": k, "removed": removed, "changed_frac_vs_a_kept_slot": frac})
            report["targets"].append({"name": name, "slots": rows})
    finally:
        for s, (p, q) in zip(sensors, saved):
            s.set_position_orientation(p, q, frame="parent")
        for _ in range(3):
            render()
    return report
