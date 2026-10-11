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
bases are less than ``min_m`` (1.0 m) apart - for camera relocalization also every pair whose goal bases are - is an offender;
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
MIN_SEPARATION_M = 1.0
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

    def to_json(self) -> Dict[str, Any]:
        return {"path": self.path, "task": self.task, "alias": self.alias, "instance_id": self.instance_id, "scene": self.scene, "room": self.room,
                "start_base": self.start_base, "goal_base": self.goal_base, "start": [p.to_json() for p in self.start],
                "goal": [p.to_json() for p in self.goal], "split": self.split}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "InstanceRecord":
        return InstanceRecord(d.get("path", ""), d["task"], d["alias"], d.get("instance_id", ""), d["scene"], d.get("room", ""),
                              list(d["start_base"]), None if d.get("goal_base") is None else list(d["goal_base"]),
                              [Pose(**p) for p in d.get("start") or []], [Pose(**p) for p in d.get("goal") or []], dict(d.get("split") or {}))


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
    return InstanceRecord(path, task, alias, str(isg.get("instance_id") or (sidecar or {}).get("instance_id") or ""), scene, room, start_base,
                          goal_base, start, goal, dict(meta.get("split") or {}))


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
    for e in eval_records:
        if e.task != rec.task or e.scene != rec.scene:
            continue
        g0 = _all_within(rec.start, e.start, pos_m, ang_deg)
        if g0 is None:
            continue
        g1 = _all_within(rec.goal, e.goal, pos_m, ang_deg)
        if g1 is None:
            continue
        return e, {**g0, **g1}
    return None


def repeat_filter(eval_records: Sequence[InstanceRecord], task: Optional[str] = None, pos_m: float = REPEAT_POS_M,
                  ang_deg: float = REPEAT_ANG_DEG) -> Callable[[str], Optional[str]]:
    """``SamplerV2.instance_filter`` of a train run: ``(tro_path) -> None | why`` - why the instance just written repeats an eval
    instance (``task``: only the eval instances of that task are kept)."""
    pool = [e for e in eval_records if task is None or e.task == task]

    def check(tro_path: str) -> Optional[str]:
        hit = find_repeat(read_instance(tro_path), pool, pos_m, ang_deg)
        if hit is None:
            return None
        e, gaps = hit
        return f"repeats eval instance {e.instance_id or e.path} ({e.alias}): " + ", ".join(f"{k} {d:.2f} m / {a:.0f} deg" for k, (d, a) in gaps.items())

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
