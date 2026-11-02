"""Distinct eval instances and the train-repeat rule (owner 2026-10-11). Pure: reads the TRO / sidecar / summary /
template_meta files of sampled instances, no simulator.

Instance record (:func:`read_instance`, :func:`record_from_dicts`): task (TRO ``activity``), alias (``task_variant``),
scene (``scene.model``), room (``scene.task_rooms``), the robot's start base (TRO ``robot_poses.robot[0]``: x, y and the yaw
of its quaternion; sidecar ``stances.init.base_pose2d`` when the TRO has none), the goal base (camera relocalization:
sidecar ``goal.base_pose2d`` / ``stances.goal.base_pose2d``, else summary ``goal_base_pose2d``; the held-object tasks never
move the base: the start base), and the poses the train-repeat rule compares:

* camera relocalization: start = the start base (x, y, yaw); goal = the goal camera pose (TRO ``pose_targets`` of the agent,
  world frame), else the goal base;
* object scaling / mental rotation: start = the start base and the held object's start pose (its TRO state ``root_link``);
  goal = the held object's goal pose (sidecar ``goal.eef`` composed with TRO ``grasps[0].object_pose_in_eef_frame``, else
  summary ``goal_realised``).

Eval separation (:func:`separation_offenders`): within one task and one (scene, room) every pair of instances whose start
bases are less than ``min_m`` (0.5 m) apart - for camera relocalization also every pair whose goal bases are - is an offender;
:func:`resample_choice` walks the aliases in order (an alias list, else sorted) and re-samples an alias that conflicts with
one kept before it (the later alias of a pair), keeping the rest. The checker CLI (:func:`main`, ``$E/separation_check.py``)
prints the pairs and the aliases to re-sample with their next seed index (``sample_isg --seed-split eval --seed-index``), and
exits 1 when there is any.

Train repeat (:func:`find_repeat`, :func:`repeat_filter`; ``multiply_isg --split train --eval-run``): a train instance repeats an
eval instance of the same task and scene when every compared start pose and every compared goal pose are within ``pos_m``
(0.25 m) and ``ang_deg`` (15 deg) of the eval instance's; the train sampler then rejects the attempt (``repeats_eval``).
Similar instances are kept; there is no distance rule among train instances.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CAMERA = "camera_relocalization"
HELD_TASKS = ("object_scaling", "mental_rotation")
RULE_TASKS = (CAMERA,) + HELD_TASKS
AGENT = "agent.n.01_1"
MIN_SEPARATION_M = 0.5
REPEAT_POS_M = 0.25
REPEAT_ANG_DEG = 15.0
INDEX_SCHEMA = "isg-instance-index-v1"
TRO_SUFFIX = "-tro_state.json"

Quat = Tuple[float, float, float, float]


# ----------------------------------------------------------------------------------------------------------------
# geometry (xyzw quaternions)
# ----------------------------------------------------------------------------------------------------------------


def yaw_of(q: Sequence[float]) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_gap_deg(a: float, b: float) -> float:
    return abs(math.degrees((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi))


def quat_gap_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(float(v) ** 2 for v in a))
    nb = math.sqrt(sum(float(v) ** 2 for v in b))
    d = abs(sum(float(x) * float(y) for x, y in zip(a, b))) / max(na * nb, 1e-12)
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def _qmul(a: Sequence[float], b: Sequence[float]) -> Quat:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qrot(q: Sequence[float], v: Sequence[float]) -> List[float]:
    x, y, z, w = (float(c) for c in q)
    vx, vy, vz = (float(c) for c in v)
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]


def compose(p1: Sequence[float], q1: Sequence[float], p2: Sequence[float], q2: Sequence[float]) -> Tuple[List[float], Quat]:
    """``(p1, q1) o (p2, q2)``: the pose ``(p2, q2)`` given in the frame ``(p1, q1)``, in that frame's parent."""
    r = _qrot(q1, p2)
    return [float(p1[i]) + r[i] for i in range(3)], _qmul(q1, q2)


# ----------------------------------------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Pose:
    """A compared pose: ``p`` (2-D base position or 3-D position) with ``yaw`` (rad, base poses) or ``q`` (xyzw)."""

    name: str
    p: List[float]
    yaw: Optional[float] = None
    q: Optional[List[float]] = None

    def gap(self, other: "Pose") -> Tuple[float, float]:
        """(position distance m, angle deg) to ``other`` (angle 0 when either has no orientation)."""
        n = min(len(self.p), len(other.p))
        d = math.dist(self.p[:n], other.p[:n])
        if self.yaw is not None and other.yaw is not None:
            return d, yaw_gap_deg(self.yaw, other.yaw)
        if self.q is not None and other.q is not None:
            return d, quat_gap_deg(self.q, other.q)
        return d, 0.0

    def to_json(self) -> Dict[str, Any]:
        return {k: v for k, v in (("name", self.name), ("p", self.p), ("yaw", self.yaw), ("q", self.q)) if v is not None}


@dataclass
class Measured:
    """train-fast 2026-11-01 (REPEAT_MEASURED): one thing the task measures, at the start or the goal, with the task's own metric and
    tolerance (TRO ``pose_targets`` / ``articulation_targets``). ``kind`` "pose": ``p`` (``pos_metric`` "full" -> distance, "depth" ->
    |dz| in the camera ``frame``, None -> not compared), ``q`` (``tol_rad``, the smallest angle over ``equiv``, the object-frame symmetry
    quaternions as ``eval.geometry.orientation_error``); kind "joint": ``p = [position]``, ``tol_m`` its tolerance (rad or m)."""

    name: str
    kind: str = "pose"
    frame: str = "world"
    p: Optional[List[float]] = None
    q: Optional[List[float]] = None
    pos_metric: Optional[str] = None
    tol_m: Optional[float] = None
    tol_rad: Optional[float] = None
    equiv: List[List[float]] = field(default_factory=list)
    #: entries with the same ``match`` tag form a set compared without regard to names (an exchangeable group's members / slots)
    match: Optional[str] = None

    def gap(self, other: "Measured") -> Optional[Dict[str, float]]:
        """``{"pos": err, "ang": err}`` of the components ``self`` measures, or None when the two are not comparable (kind / frame)."""
        if self.kind != other.kind or self.frame != other.frame:
            return None
        out: Dict[str, float] = {}
        if self.p is not None and other.p is not None and (self.kind == "joint" or self.pos_metric):
            if self.kind == "joint":
                out["pos"] = abs(float(self.p[0]) - float(other.p[0]))
            elif self.pos_metric == "depth":
                out["pos"] = abs(float(self.p[2]) - float(other.p[2]))
            else:
                out["pos"] = math.dist(self.p[:3], other.p[:3])
        if self.q is not None and other.q is not None and self.tol_rad is not None:
            eqs = [(0.0, 0.0, 0.0, 1.0)] + [tuple(e) for e in self.equiv]
            out["ang"] = min(math.radians(quat_gap_deg(_qmul(other.q, e), self.q)) for e in eqs)
        return out

    def within(self, other: "Measured") -> Optional[Dict[str, float]]:
        """The gaps when ``other`` lies within ``self``'s tolerances on every measured component, else None."""
        g = self.gap(other)
        if g is None or not g:
            return None
        if "pos" in g and self.tol_m is not None and g["pos"] > float(self.tol_m):
            return None
        if "ang" in g and g["ang"] > float(self.tol_rad):
            return None
        return g

    def to_json(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class InstanceRecord:
    path: str
    task: str
    alias: str
    instance_id: str
    scene: str
    room: str
    start_base: List[float]
    goal_base: Optional[List[float]]
    start: List[Pose]
    goal: List[Pose]
    split: Dict[str, Any] = field(default_factory=dict)
    #: REPEAT_MEASURED (train-fast 2026-11-01): what the task measures (TRO pose_targets / articulation_targets) at the start and the goal
    measured_start: List[Measured] = field(default_factory=list)
    measured_goal: List[Measured] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"path": self.path, "task": self.task, "alias": self.alias, "instance_id": self.instance_id, "scene": self.scene, "room": self.room,
                "start_base": self.start_base, "goal_base": self.goal_base, "start": [p.to_json() for p in self.start],
                "goal": [p.to_json() for p in self.goal], "split": self.split,
                "measured_start": [m.to_json() for m in self.measured_start], "measured_goal": [m.to_json() for m in self.measured_goal]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "InstanceRecord":
        return InstanceRecord(d.get("path", ""), d["task"], d["alias"], d.get("instance_id", ""), d["scene"], d.get("room", ""),
                              list(d["start_base"]), None if d.get("goal_base") is None else list(d["goal_base"]),
                              [Pose(**p) for p in d.get("start") or []], [Pose(**p) for p in d.get("goal") or []], dict(d.get("split") or {}),
                              [Measured(**m) for m in d.get("measured_start") or []], [Measured(**m) for m in d.get("measured_goal") or []])


def _base2d(v: Any) -> Optional[List[float]]:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return [float(v[0]), float(v[1]), float(v[2])]
    return None


def _pose_of(d: Any) -> Optional[Tuple[List[float], List[float]]]:
    """``{"position": [3], "orientation": [4]}`` (or ``pos`` / ``ori``) -> (p, q)."""
    if not isinstance(d, dict):
        return None
    p, q = d.get("position", d.get("pos")), d.get("orientation", d.get("ori"))
    if p is None or q is None or len(p) < 3 or len(q) != 4:
        return None
    return [float(v) for v in p[:3]], [float(v) for v in q]


def start_base_of(tro: Dict[str, Any], sidecar: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None) -> List[float]:
    """The robot's start base ``[x, y, yaw]``: TRO ``robot_poses.robot[0]``, else sidecar ``stances.init``, else summary."""
    robots = ((tro.get("robot_poses") or {}).get("robot") or [])
    if robots and _pose_of(robots[0]) is not None:
        p, q = _pose_of(robots[0])
        return [p[0], p[1], yaw_of(q)]
    for v in (((sidecar or {}).get("stances") or {}).get("init") or {}).get("base_pose2d"), (summary or {}).get("base_pose2d"):
        b = _base2d(v)
        if b is not None:
            return b
    raise ValueError("no start base pose (TRO robot_poses, sidecar stances.init, summary base_pose2d)")


def goal_base_of(sidecar: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None) -> Optional[List[float]]:
    """The camera-relocalization goal base ``[x, y, yaw]``: sidecar ``goal`` / ``stances.goal``, else summary ``goal_base_pose2d``."""
    sc = sidecar or {}
    for v in ((sc.get("goal") or {}).get("base_pose2d"), ((sc.get("stances") or {}).get("goal") or {}).get("base_pose2d"),
              (summary or {}).get("goal_base_pose2d")):
        b = _base2d(v)
        if b is not None:
            return b
    return None


def held_object_poses(tro: Dict[str, Any], sidecar: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None
                      ) -> Tuple[Optional[Tuple[List[float], List[float]]], Optional[Tuple[List[float], List[float]]]]:
    """(start, goal) world pose of the held object: its TRO state at the start; at the goal the sidecar goal eef composed with the
    TRO grasp's object-in-eef pose, else the summary's ``goal_realised``."""
    isg = tro.get("ispatialgym") or {}
    grasps = isg.get("grasps") or []
    obj = grasps[0].get("object") if grasps else None
    if obj is None:
        obj = next((k for k, v in tro.items() if k not in ("robot_poses", "ispatialgym") and isinstance(v, dict) and "root_link" in v
                    and not k.startswith("floor.")), None)
    start = _pose_of((tro.get(obj) or {}).get("root_link")) if obj else None
    goal = None
    eef = _pose_of(((sidecar or {}).get("goal") or {}).get("eef"))
    rel = _pose_of(grasps[0].get("object_pose_in_eef_frame")) if grasps else None
    if eef is not None and rel is not None:
        p, q = compose(eef[0], eef[1], rel[0], rel[1])
        goal = (p, list(q))
    elif _pose_of((summary or {}).get("goal_realised")) is not None:
        goal = _pose_of(summary["goal_realised"])
    return start, goal


def measured_of(tro: Dict[str, Any]) -> Tuple[List[Measured], List[Measured]]:
    """REPEAT_MEASURED: ``(start, goal)`` of what the TRO's task measures - every ``pose_targets`` entry (goal: the target in its frame,
    with its metrics, tolerances and orientation equivalences; start: the object's TRO ``root_link`` state, world frame, the same
    components and tolerances - none for the agent, whose start is the start base) and every ``articulation_targets`` joint (goal: the
    target position; start: the object's ``joint_pos`` when it has that one joint). Distractors and other objects are not measured."""
    isg = tro.get("ispatialgym") or {}
    start: List[Measured] = []
    goal: List[Measured] = []
    for name, t in sorted((isg.get("pose_targets") or {}).items()):
        if not isinstance(t, dict):
            continue
        pm, om = t.get("position_metric"), t.get("orientation_metric")
        p = [float(v) for v in t["position"]] if (pm and t.get("position") is not None) else None
        q = [float(v) for v in t["orientation"]] if (om and t.get("orientation") is not None) else None
        if p is None and q is None:
            continue
        tol_m = float(t["position_tolerance"]) if (p is not None and t.get("position_tolerance") is not None) else None
        tol_rad = float(t["orientation_tolerance"]) if (q is not None and t.get("orientation_tolerance") is not None) else None
        if (p is not None and tol_m is None) or (q is not None and tol_rad is None):
            continue  # a measured component without a tolerance cannot be compared at the task's precision
        eqv = [[float(v) for v in e] for e in (t.get("orientation_equivalences") or [])]
        goal.append(Measured(f"goal:{name}", "pose", str(t.get("frame") or "world"), p, q, pm if p is not None else None, tol_m, tol_rad, eqv))
        if name == AGENT:
            continue
        s = _pose_of((tro.get(name) or {}).get("root_link"))
        if s is not None:
            start.append(Measured(f"start:{name}", "pose", "world", s[0] if p is not None else None, s[1] if q is not None else None,
                                  "full" if p is not None else None, tol_m, tol_rad, eqv))
    for gname, gspec in sorted((isg.get("groups") or {}).items()):
        # groups (attachment's assembly, block design's slots): world-frame target slots at the group's pose tolerances; members of an
        # ``exchangeable`` group (any member in any slot) and their slots compare as sets
        if not isinstance(gspec, dict) or not gspec.get("target_slots"):
            continue
        tol = gspec.get("pose_tolerances") or {}
        tol_m = float(tol["position"]) if tol.get("position") is not None else None
        tol_rad = float(tol["orientation"]) if tol.get("orientation") is not None else None
        if tol_m is None and tol_rad is None:
            continue
        eqv = [[float(v) for v in e] for e in (gspec.get("orientation_equivalences") or [])]
        exch = ((gspec.get("assignment") or {}).get("mode") or "fixed") == "exchangeable"
        for slot, sp in sorted(gspec["target_slots"].items()):
            pose = _pose_of(sp)
            if pose is None:
                continue
            goal.append(Measured(f"goal:{gname}:{slot}", "pose", "world", pose[0] if tol_m is not None else None, pose[1] if tol_rad is not None else None,
                                 "full" if tol_m is not None else None, tol_m, tol_rad, eqv, match=f"{gname}:slots" if exch else None))
        for member in gspec.get("members") or []:
            s0 = _pose_of((tro.get(member) or {}).get("root_link"))
            if s0 is not None:
                start.append(Measured(f"start:{member}", "pose", "world", s0[0] if tol_m is not None else None, s0[1] if tol_rad is not None else None,
                                      "full" if tol_m is not None else None, tol_m, tol_rad, eqv, match=f"{gname}:members" if exch else None))
    for obj, spec in sorted((isg.get("articulation_targets") or {}).items()):
        joints = (spec or {}).get("joints") or {}
        for jn, j in sorted(joints.items()):
            if not isinstance(j, dict) or j.get("position") is None or j.get("tolerance") is None:
                continue
            goal.append(Measured(f"goal:{obj}:{jn}", "joint", "joint", [float(j["position"])], tol_m=float(j["tolerance"])))
            jp = (tro.get(obj) or {}).get("joint_pos")
            if isinstance(jp, list) and len(jp) == 1 and len(joints) == 1:
                start.append(Measured(f"start:{obj}:{jn}", "joint", "joint", [float(jp[0])], tol_m=float(j["tolerance"])))
    return start, goal


def record_from_dicts(tro: Dict[str, Any], sidecar: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None,
                      meta: Optional[Dict[str, Any]] = None, path: str = "") -> InstanceRecord:
    """The :class:`InstanceRecord` of one instance from its TRO (and, when given, sidecar / summary / template_meta dicts)."""
    isg = tro.get("ispatialgym") or {}
    meta = meta or {}
    task = str(isg.get("activity") or meta.get("problem", "").split("-")[0])
    alias = str(isg.get("task_variant") or (sidecar or {}).get("alias") or meta.get("alias") or "")
    sb = isg.get("scene") or {}
    scene = str(sb.get("model") or meta.get("scene") or "")
    room = "+".join(sorted(sb.get("task_rooms") or meta.get("task_rooms") or meta.get("rooms") or []))
    start_base = start_base_of(tro, sidecar, summary)
    start = [Pose("start_base", start_base[:2], yaw=start_base[2])]
    goal: List[Pose] = []
    if task in HELD_TASKS:
        goal_base: Optional[List[float]] = list(start_base)
        obj0, obj1 = held_object_poses(tro, sidecar, summary)
        if obj0 is not None:
            start.append(Pose("object_start", obj0[0], q=obj0[1]))
        if obj1 is not None:
            goal.append(Pose("object_goal", obj1[0], q=obj1[1]))
    else:
        goal_base = goal_base_of(sidecar, summary)
        cam = (isg.get("pose_targets") or {}).get(AGENT) or {}
        if task == CAMERA and cam.get("frame", "world") == "world" and cam.get("position") is not None and cam.get("orientation") is not None:
            goal.append(Pose("goal_camera", [float(v) for v in cam["position"]], q=[float(v) for v in cam["orientation"]]))
        elif goal_base is not None:
            goal.append(Pose("goal_base", goal_base[:2], yaw=goal_base[2]))
    m_start, m_goal = measured_of(tro)
    return InstanceRecord(path, task, alias, str(isg.get("instance_id") or (sidecar or {}).get("instance_id") or ""), scene, room, start_base,
                          goal_base, start, goal, dict(meta.get("split") or {}), m_start, m_goal)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_instance(tro_path: str) -> InstanceRecord:
    """:func:`record_from_dicts` of ``<iid>-tro_state.json`` with its ``.oracle.json`` / ``.summary.json`` and the template's
    ``template_meta.json`` (each optional but the TRO)."""
    with open(tro_path) as f:
        tro = json.load(f)
    stem = tro_path[: -len(TRO_SUFFIX)] if tro_path.endswith(TRO_SUFFIX) else tro_path
    meta = _read_json(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tro_path))), "template_meta.json"))
    return record_from_dicts(tro, _read_json(stem + ".oracle.json"), _read_json(stem + ".summary.json"), meta, path=tro_path)


# ----------------------------------------------------------------------------------------------------------------
# loading (run directories, TRO files, index files)
# ----------------------------------------------------------------------------------------------------------------


def renamed_run_dir(tro_path: str, alias: str) -> bool:
    """True for an instance under ``<run>/<dir>/inst/<alias>/<template>/instances/`` whose ``<dir>`` is not the alias (a run
    directory set aside, e.g. ``<alias>.hung_<job>``)."""
    d = os.path.abspath(tro_path)
    for _ in range(4):
        d = os.path.dirname(d)
    return os.path.basename(d) == "inst" and os.path.basename(os.path.dirname(d)) != alias


#: directory layouts, first match wins: a run tag (``<tag>/<alias>/inst/<alias>/<template>/instances``; the job's other copies -
#: ``replay_inst/`` - are not instances of the run), one run of an alias, a sampler out root, a template directory
_LAYOUTS = (("*", "inst", "*", "*", "instances"), ("inst", "*", "*", "instances"), ("*", "*", "instances"), ("instances",))


def tro_files(source: str) -> List[str]:
    """The TRO files of ``source``: a TRO file, or a directory in one of the ``_LAYOUTS`` (else searched recursively)."""
    if os.path.isfile(source):
        return [source] if source.endswith(TRO_SUFFIX) else []
    for layout in _LAYOUTS:
        hits = sorted(glob.glob(os.path.join(source, *layout, "*" + TRO_SUFFIX)))
        if hits:
            return hits
    return sorted(glob.glob(os.path.join(source, "**", "instances", "*" + TRO_SUFFIX), recursive=True))


def write_index(path: str, records: Iterable[InstanceRecord]) -> str:
    recs = [r.to_json() for r in records]
    with open(path, "w") as f:
        json.dump({"schema": INDEX_SCHEMA, "n": len(recs), "instances": recs}, f, indent=1)
    return path


def load_instances(sources: Iterable[str], tasks: Optional[Sequence[str]] = None, notes: Optional[List[str]] = None) -> List[InstanceRecord]:
    """Records of every instance under ``sources`` (directories scanned recursively for ``instances/*-tro_state.json``, TRO
    files, index JSON files from :func:`write_index`), of ``tasks`` when given; run directories set aside are skipped. ``notes``
    collects what was skipped and why."""
    out: List[InstanceRecord] = []
    seen = set()
    notes = notes if notes is not None else []
    for src in sources:
        if os.path.isfile(src) and src.endswith(".json") and not src.endswith(TRO_SUFFIX):
            d = _read_json(src) or {}
            if d.get("schema") != INDEX_SCHEMA:
                raise ValueError(f"{src}: not a TRO or an instance index ({INDEX_SCHEMA})")
            recs = [InstanceRecord.from_json(r) for r in d.get("instances") or []]
        else:
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            recs = []
            for p in tro_files(src):
                rp = os.path.realpath(p)
                if rp in seen:
                    continue
                seen.add(rp)
                try:
                    rec = read_instance(p)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    notes.append(f"unreadable {p}: {e!r}")
                    continue
                if renamed_run_dir(p, rec.alias):
                    notes.append(f"skipped (run directory set aside): {p}")
                    continue
                recs.append(rec)
        out += [r for r in recs if tasks is None or r.task in tasks]
    return out


def latest_by_alias(record_lists: Sequence[List[InstanceRecord]]) -> List[InstanceRecord]:
    """One list from several runs: a later run's instances replace every instance of the same alias from an earlier run."""
    by_alias: Dict[str, List[InstanceRecord]] = {}
    for recs in record_lists:
        fresh: Dict[str, List[InstanceRecord]] = {}
        for r in recs:
            fresh.setdefault(r.alias, []).append(r)
        by_alias.update(fresh)
    return [r for a in sorted(by_alias) for r in by_alias[a]]


# ----------------------------------------------------------------------------------------------------------------
# eval separation
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Offense:
    a: InstanceRecord
    b: InstanceRecord
    kind: str  # "start" | "goal"
    distance_m: float


def separation_offenders(records: Sequence[InstanceRecord], min_m: float = MIN_SEPARATION_M) -> List[Offense]:
    """Pairs of instances of one task in one (scene, room) whose start bases are closer than ``min_m`` (camera relocalization:
    or whose goal bases are)."""
    groups: Dict[Tuple[str, str, str], List[InstanceRecord]] = {}
    for r in records:
        groups.setdefault((r.task, r.scene, r.room), []).append(r)
    out: List[Offense] = []
    for key in sorted(groups):
        rs = sorted(groups[key], key=lambda r: (r.alias, r.instance_id))
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                d = math.dist(a.start_base[:2], b.start_base[:2])
                if d < min_m:
                    out.append(Offense(a, b, "start", d))
                if a.task == CAMERA and a.goal_base is not None and b.goal_base is not None:
                    dg = math.dist(a.goal_base[:2], b.goal_base[:2])
                    if dg < min_m:
                        out.append(Offense(a, b, "goal", dg))
    return out


def resample_choice(offenses: Sequence[Offense], order: Optional[Sequence[str]] = None) -> Dict[str, List[Offense]]:
    """Alias -> its offenses with the aliases kept, for every alias to re-sample: in ``order`` (unlisted aliases after it, sorted),
    an alias that conflicts with an alias kept before it or with itself is re-sampled, every other alias is kept."""
    rank = {a: i for i, a in enumerate(order or [])}
    aliases = sorted({o.a.alias for o in offenses} | {o.b.alias for o in offenses}, key=lambda a: (rank.get(a, len(rank)), a))
    kept: set = set()
    out: Dict[str, List[Offense]] = {}
    for al in aliases:
        conf = [o for o in offenses if (o.a.alias == al and (o.b.alias == al or o.b.alias in kept)) or (o.b.alias == al and o.a.alias in kept)]
        if conf:
            out[al] = conf
        else:
            kept.add(al)
    return out


def next_seed_index(records: Sequence[InstanceRecord], alias: str) -> Tuple[int, str]:
    """The seed index to re-sample ``alias`` with and why: one past the split seed index its instances were sampled with, 0 for an
    instance sampled without a split seed (``--seed``)."""
    ks = [r.split.get("index") for r in records if r.alias == alias and r.split.get("seed_source") == "split" and r.split.get("index") is not None]
    if ks:
        return max(int(k) for k in ks) + 1, f"sampled with split seed index {max(int(k) for k in ks)}"
    seeds = sorted({str(r.split.get("seed")) for r in records if r.alias == alias and r.split.get("seed") is not None})
    return 0, ("sampled with --seed " + ",".join(seeds) if seeds else "sampled without a recorded split seed")


# ----------------------------------------------------------------------------------------------------------------
# train repeat
# ----------------------------------------------------------------------------------------------------------------


def _all_within(a: Sequence[Pose], b: Sequence[Pose], pos_m: float, ang_deg: float) -> Optional[Dict[str, Tuple[float, float]]]:
    """Gap per pose name when every pose of ``a`` with a namesake in ``b`` is within the tolerances (at least one compared)."""
    other = {p.name: p for p in b}
    gaps: Dict[str, Tuple[float, float]] = {}
    for p in a:
        if p.name not in other:
            continue
        d, ang = p.gap(other[p.name])
        if d > pos_m or ang > ang_deg:
            return None
        gaps[p.name] = (round(d, 3), round(ang, 1))
    return gaps or None


def find_repeat(rec: InstanceRecord, eval_records: Sequence[InstanceRecord], pos_m: float = REPEAT_POS_M,
                ang_deg: float = REPEAT_ANG_DEG) -> Optional[Tuple[InstanceRecord, Dict[str, Tuple[float, float]]]]:
    """The first eval instance of ``rec``'s task and scene whose start and goal poses are all within ``pos_m`` / ``ang_deg`` of
    ``rec``'s, with the gaps (m, deg) per pose; None when ``rec`` repeats none."""
    measured = measured_rule_on()
    for e in eval_records:
        if e.task != rec.task or e.scene != rec.scene:
            continue
        if measured and rec.measured_goal and e.measured_goal:
            gaps = measured_repeat(rec, e, pos_m, ang_deg)
            if gaps is not None:
                return e, gaps
            continue
        g0 = _all_within(rec.start, e.start, pos_m, ang_deg)
        if g0 is None:
            continue
        g1 = _all_within(rec.goal, e.goal, pos_m, ang_deg)
        if g1 is None:
            continue
        return e, {**g0, **g1}
    return None


def expected_pose_names(task: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(start pose names, goal pose names) a complete record of ``task`` compares (:func:`record_from_dicts`): the held-object
    tasks the start base and the held object's start / goal pose, camera relocalization the start base and the goal camera, every
    other task the start base and the goal base. A record with fewer poses compares fewer constraints (``_all_within`` skips a
    name the other side lacks), so an *early* record (train-fast 2026-10-31: built from the draft before the settle) must carry
    all of them before it may reject - see :func:`record_complete`."""
    if task in HELD_TASKS:
        return ("start_base", "object_start"), ("object_goal",)
    if task == CAMERA:
        return ("start_base",), ("goal_camera",)
    return ("start_base",), ("goal_base",)


def measured_rule_on() -> bool:
    """``ISPATIALGYM_FAST_REPEAT_MEASURED`` (train-fast 2026-11-01, default on; owner 2026-11-01): the train-repeat rule compares what the
    task measures (:func:`measured_repeat`) instead of the robot base (+ the held object) of the 2026-10-11 rule."""
    from ispatialgym.fastflags import fast_on

    return fast_on("REPEAT_MEASURED")


def measured_repeat(rec: InstanceRecord, e: InstanceRecord, pos_m: float = REPEAT_POS_M, ang_deg: float = REPEAT_ANG_DEG
                    ) -> Optional[Dict[str, Tuple[float, float]]]:
    """REPEAT_MEASURED (owner 2026-11-01): ``rec`` repeats ``e`` when the robot's start base is within ``pos_m`` / ``ang_deg`` and every
    measured entry (:func:`measured_of`: each measured object's start state and goal target, each joint target) is within that entry's
    own task tolerance (its metric, its orientation equivalences); entries are matched by name and both records must measure the same
    things (else they are not the same task instance). Returns the gaps per entry (m or rad as m, deg), or None."""
    sb_a = [p for p in rec.start if p.name == "start_base"]
    sb_b = [p for p in e.start if p.name == "start_base"]
    g0 = _all_within(sb_a, sb_b, pos_m, ang_deg)
    if g0 is None:
        return None
    gaps: Dict[str, Tuple[float, float]] = dict(g0)
    mine_all, theirs_all = rec.measured_start + rec.measured_goal, e.measured_start + e.measured_goal
    mine = {m.name: m for m in mine_all if m.match is None}
    theirs = {m.name: m for m in theirs_all if m.match is None}
    if set(mine) != set(theirs):
        return None
    for name, m in sorted(theirs.items()):
        g = m.within(mine[name])
        if g is None:
            return None
        gaps[name] = (float(g.get("pos", 0.0)), math.degrees(float(g.get("ang", 0.0))))
    sets_a: Dict[str, List[Measured]] = {}
    sets_b: Dict[str, List[Measured]] = {}
    for m in mine_all:
        if m.match is not None:
            sets_a.setdefault(m.match, []).append(m)
    for m in theirs_all:
        if m.match is not None:
            sets_b.setdefault(m.match, []).append(m)
    if set(sets_a) != set(sets_b):
        return None
    for tag in sorted(sets_b):
        pairing = _match_sets(sets_b[tag], sets_a[tag])
        if pairing is None:
            return None
        for name, g in pairing.items():
            gaps[name] = (float(g.get("pos", 0.0)), math.degrees(float(g.get("ang", 0.0))))
    return gaps


def _match_sets(theirs: Sequence[Measured], mine: Sequence[Measured]) -> Optional[Dict[str, Dict[str, float]]]:
    """An exchangeable set: each of ``theirs`` paired with a distinct one of ``mine`` within its tolerances (every permutation up to 7
    entries, else greedy), the gaps per entry of ``theirs``; None when the sizes differ or no pairing fits."""
    if len(theirs) != len(mine):
        return None
    ok = [[t.within(m) for m in mine] for t in theirs]
    n = len(theirs)
    if n <= 7:
        import itertools

        for perm in itertools.permutations(range(n)):
            if all(ok[i][perm[i]] is not None for i in range(n)):
                return {theirs[i].name: ok[i][perm[i]] for i in range(n)}
        return None
    used, out = set(), {}
    for i in range(n):
        j = next((j for j in range(n) if j not in used and ok[i][j] is not None), None)
        if j is None:
            return None
        used.add(j)
        out[theirs[i].name] = ok[i][j]
    return out


def record_complete(rec: InstanceRecord) -> Optional[str]:
    """None when ``rec`` carries every pose :func:`expected_pose_names` lists for its task, else the missing names (a reason to
    leave the decision to the post-write check). REPEAT_MEASURED: the start base and, for every measured pose target but the agent's,
    the object's start state (an early record built from the draft may lack one)."""
    if measured_rule_on() and rec.measured_goal:
        have = {m.name for m in rec.measured_start}
        missing = [] if any(p.name == "start_base" for p in rec.start) else ["start_base"]
        # pose targets only (``goal:<object>``; a group's slot ``goal:<group>:<slot>`` has no start of its own)
        missing += ["start:" + m.name[len("goal:"):] for m in rec.measured_goal
                    if m.kind == "pose" and m.match is None and m.name.count(":") == 1 and m.name != f"goal:{AGENT}"
                    and "start:" + m.name[len("goal:"):] not in have]
        return None if not missing else "missing " + ", ".join(missing)
    want_start, want_goal = expected_pose_names(rec.task)
    missing = [n for n in want_start if n not in {p.name for p in rec.start}] + [n for n in want_goal if n not in {p.name for p in rec.goal}]
    return None if not missing else "missing " + ", ".join(missing)


def repeat_filter(eval_records: Sequence[InstanceRecord], task: Optional[str] = None, pos_m: float = REPEAT_POS_M,
                  ang_deg: float = REPEAT_ANG_DEG) -> Callable[[str], Optional[str]]:
    """``SamplerV2.instance_filter`` of a train run: ``(tro_path) -> None | why`` - why the instance just written repeats an eval
    instance (``task``: only the eval instances of that task are kept). The returned callable also carries ``check_record``
    (``(InstanceRecord) -> None | why``, the same decision on a record built in memory: ``SamplerV2.instance_record`` runs it on
    the draft right after the sample, train-fast ``EARLY_REPEAT_FILTER`` 2026-10-31) and ``pool`` (the eval records compared)."""
    pool = [e for e in eval_records if task is None or e.task == task]

    def check_record(rec: InstanceRecord) -> Optional[str]:
        hit = find_repeat(rec, pool, pos_m, ang_deg)
        if hit is None:
            return None
        e, gaps = hit
        return f"repeats eval instance {e.instance_id or e.path} ({e.alias}): " + ", ".join(f"{k} {d:.2f} m / {a:.0f} deg" for k, (d, a) in gaps.items())

    def check(tro_path: str) -> Optional[str]:
        return check_record(read_instance(tro_path))

    check.check_record = check_record  # type: ignore[attr-defined]
    check.pool = pool  # type: ignore[attr-defined]
    return check


# ----------------------------------------------------------------------------------------------------------------
# checker CLI ($E/separation_check.py)
# ----------------------------------------------------------------------------------------------------------------


def run_tag_records(source: str, runs_root: Optional[str], tasks: Sequence[str], notes: List[str]) -> List[InstanceRecord]:
    """The instances of one run: ``source`` is a directory / index file, or a tag under ``runs_root``
    (``<runs_root>/<tag>/<alias>/inst/...``)."""
    path = source if (os.path.exists(source) or runs_root is None) else os.path.join(runs_root, source)
    return load_instances([path], tasks=tasks, notes=notes)


def main(argv: Optional[Sequence[str]] = None, runs_root: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Eval separation check: instances of one task in one (scene, room) whose start bases (camera "
                                                 "relocalization: or goal bases) are closer than --min. Several runs: a later run's instances replace "
                                                 "the alias's instances of the earlier ones. Exit 1 when there is an offender.")
    parser.add_argument("runs", nargs="+", help="run tags (under --runs-root) or directories / TRO files / index files")
    parser.add_argument("--runs-root", default=runs_root, help="where the run tags live (default: %(default)s)")
    parser.add_argument("--min", type=float, default=MIN_SEPARATION_M, help="minimum base distance in m (default: %(default)s)")
    parser.add_argument("--tasks", default=",".join(RULE_TASKS), help="comma-separated tasks (TRO activity; default: %(default)s)")
    parser.add_argument("--aliases", default=None, help="alias list file: its order decides which alias of a pair is re-sampled (the later); default sorted")
    parser.add_argument("--write-index", default=None, help="write the instances checked as an index file (multiply_isg --eval-run)")
    parser.add_argument("--json", default=None, help="write the report as JSON")
    args = parser.parse_args(argv)
    tasks = [t for t in args.tasks.split(",") if t]
    notes: List[str] = []
    records = latest_by_alias([run_tag_records(s, args.runs_root, tasks, notes) for s in args.runs])
    order = None
    if args.aliases:
        with open(args.aliases) as f:
            order = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    offenses = separation_offenders(records, args.min)
    resample = resample_choice(offenses, order)
    for n in notes:
        print(f"NOTE {n}")
    groups: Dict[Tuple[str, str, str], int] = {}
    for r in records:
        groups[(r.task, r.scene, r.room)] = groups.get((r.task, r.scene, r.room), 0) + 1
    print(f"{len(records)} instances of {len({r.alias for r in records})} aliases in {len(groups)} (task, scene, room) groups; minimum {args.min:.2f} m")
    for (task, scene, room), n in sorted(groups.items()):
        print(f"  {task:24s} {scene:26s} {room:22s} {n}")
    report: Dict[str, Any] = {"min_m": args.min, "n_instances": len(records), "offenses": [], "resample": {}}
    pairs: Dict[Tuple[str, str], List[Offense]] = {}
    for o in offenses:
        pairs.setdefault((o.a.path, o.b.path), []).append(o)
    for os_ in pairs.values():
        a, b = os_[0].a, os_[0].b
        who = b.alias if b.alias in resample else (a.alias if a.alias in resample else None)
        gaps = ", ".join(f"{o.kind} {o.distance_m:.2f} m" for o in os_)
        print(f"OFFENDER {a.task} {a.scene}/{a.room}: {a.alias} ({a.instance_id}) <-> {b.alias} ({b.instance_id}): {gaps}; start bases "
              f"{[round(v, 2) for v in a.start_base[:2]]} / {[round(v, 2) for v in b.start_base[:2]]} -> re-sample {who}")
        for o in os_:
            report["offenses"].append({"task": a.task, "scene": a.scene, "room": a.room, "a": a.alias, "b": b.alias, "kind": o.kind,
                                       "distance_m": round(o.distance_m, 3), "a_instance": a.path, "b_instance": b.path, "resample": who})
    for alias in resample:
        k, why = next_seed_index(records, alias)
        print(f"RESAMPLE {alias} --seed-split eval --seed-index {k}  ({why})")
        report["resample"][alias] = {"seed_index": k, "why": why}
    if args.write_index:
        write_index(args.write_index, records)
        print(f"index of {len(records)} instances -> {args.write_index}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    print(f"SEPARATION {'FAIL' if offenses else 'OK'} pairs={len(pairs)} resample={len(resample)}")
    return 1 if offenses else 0


if __name__ == "__main__":
    sys.exit(main())
