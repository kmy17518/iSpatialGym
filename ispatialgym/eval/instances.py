"""The eval instance dataset: task family -> task -> subtask -> instances, laid out like BEHAVIOR-1K's
2026-challenge-task-instances so that an instance is found from a name and an index.

It sits next to the challenge instances in OmniGibson's data folder: ``<root>`` = ``$ISPATIALGYM_INSTANCES_PATH``, default
``$OMNIGIBSON_DATA_PATH/ispatialgym-eval-instances`` (``<repo>/Behavior-1K/datasets/`` when unset); the demos replays use
are the sibling dataset ``<demos>`` = ``$ISPATIALGYM_DEMOS_PATH``, default ``<root>/../ispatialgym-eval-demos``::

    <root>/metadata/instances.jsonl        one line per instance (the index this module reads)
    <root>/metadata/episodes.jsonl         one line per instance's demo: its length (recorded steps) and its file in <demos>
    <root>/metadata/tasks.yaml             task family -> task -> [subtask, ...]
    <root>/metadata/available_tasks.yaml   subtask -> {0: scene_model, robot start pose, task, task family}
    <root>/metadata/task.jsonl             one line per subtask (task_index, task_name = subtask, length, ...)
    <root>/scenes/<scene>/json/<scene>_task_<subtask>_0_0_template.json          the scene with the task objects
    <root>/scenes/<scene>/json/<scene>_task_<subtask>_0_0_template-meta.json     the sampler's template facts
    <root>/scenes/<scene>/json/<scene>_task_<subtask>_instances/
        <scene>_task_<subtask>_0_<i>_template-tro_state.json   instance i (run_isg --instance takes this file)
        <scene>_task_<subtask>_0_<i>_template.oracle.json      its oracle sidecar
        <scene>_task_<subtask>_0_<i>_template-goal_rgb.png     its goal image
    <demos>/scenes/<scene>/json/<scene>_task_<subtask>_instances/
        <scene>_task_<subtask>_0_<i>_template-demo.hdf5        instance i's demo (run_isg --task replays it)

As in BEHAVIOR, the BDDL problems are not in the dataset: an instance names its problem (``<activity>-<id>``: ``problem`` in
the index and the template metadata, ``activity_definition_id`` in the instance file), loaded from the repository's
``ispatialgym/assets/bddl/activity_definitions`` (BEHAVIOR's bddl package holds its problems).

``run_isg --task`` gives an instance ``max(300, 2 x`` its demo's length ``)`` steps by default (``step_budget``).

The goal image is the head camera's last frame (``final_rgb.png``) of a successful replay of the instance's demo: the eval
run's replay re-run with the current code, else its original replay (``set_goal``). A replay renders the whole episode, and
the renderer's bounced light depends on what it rendered before, so a frame rendered straight at the demo's final state
comes out darker than the end of the episode. The goal image is registered in its instance file
(``ispatialgym.goal_observation``: image path, sha256, size, camera; the block ``run_isg --goal-image`` writes), so whatever
reads the goal image from an instance file finds it. Where a goal image came from (the replay, its success) is an internal
record outside the dataset (``goal_record_path``: ``<records>/goal_images/<instance id>-goal.json``, ``<records>`` =
``$ISPATIALGYM_RELEASE_RECORDS``, default ``<repo>/datasets/release_records``). For another image size, replay the instance at that size
(``run_isg --task <name> --instance-indices <i> --mode replay --image-size N --record video --out <dir>``; a replay only
logs its ``--qa`` metrics, so the demo's own contacts never stop it) and ``set-goal`` it with ``--run <dir>``; the instance
keeps one goal image, and a re-export keeps a goal image set from such a run.

A name resolves at the most specific level that has it (subtask, then task, then task family); the instance indices count
that level's instances in the index order (a task's instances are its subtasks' instances, subtask by subtask).

    python -m ispatialgym.eval.instances [--root R] list [--level task_family|task|subtask]
    python -m ispatialgym.eval.instances [--root R] show <name> [<index> ...]
    python -m ispatialgym.eval.instances [--root R] set-goal <name> <index> [--run <run_isg out dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_ENV = "ISPATIALGYM_INSTANCES_PATH"
DEMOS_ENV = "ISPATIALGYM_DEMOS_PATH"
RECORDS_ENV = "ISPATIALGYM_RELEASE_RECORDS"
DATASET_NAME = "ispatialgym-eval-instances"
DEMOS_NAME = "ispatialgym-eval-demos"
TRO_SUFFIX = "-tro_state.json"
DEMO_SUFFIX = "-demo.hdf5"
LEVELS = ("subtask", "task", "task_family")
EVALUATION_PROFILE = "isg-v1"  # run_isg.EVALUATION_PROFILE (run_isg imports the simulator)
#: step budget of run_isg --task: max(MIN_MAX_STEPS, DEMO_STEP_FACTOR x the demo's length) steps, a settle window of
#: SETTLE_FRACTION of them, WALL_S_PER_STEP seconds per step; DEFAULT_MAX_STEPS without a demo
MIN_MAX_STEPS = 300
DEMO_STEP_FACTOR = 2
SETTLE_FRACTION = 0.2
WALL_S_PER_STEP = 1.0
DEFAULT_MAX_STEPS = 3000
_INSTANCES_DIR = re.compile(r"^(?P<prefix>.+_task_.+)_instances$")


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: the release's BDDL problems (ISpatialGymTask's default activity definitions root)
ACTIVITY_DEFINITIONS = os.path.join(REPO, "ispatialgym", "assets", "bddl", "activity_definitions")


def data_root() -> str:
    """OmniGibson's data folder (``$OMNIGIBSON_DATA_PATH``, else ``<repo>/Behavior-1K/datasets``)."""
    return os.environ.get("OMNIGIBSON_DATA_PATH") or os.path.join(REPO, "Behavior-1K", "datasets")


def records_root() -> str:
    """Internal records of the release, outside the datasets (``$ISPATIALGYM_RELEASE_RECORDS``, else ``<repo>/datasets/release_records``)."""
    return os.environ.get(RECORDS_ENV) or os.path.join(REPO, "datasets", "release_records")


def default_root() -> str:
    return os.environ.get(ROOT_ENV) or os.path.join(data_root(), DATASET_NAME)


def demos_root(root: Optional[str] = None) -> str:
    return os.environ.get(DEMOS_ENV) or os.path.join(os.path.dirname(os.path.abspath(root or default_root())), DEMOS_NAME)


def step_budget(length: Optional[int], max_steps: Optional[int] = None, settle_steps: Optional[int] = None,
                wallclock_s: Optional[float] = None) -> Dict[str, Any]:
    """``run_isg``'s step budget: ``max_steps`` (the actions: policy steps, a replay, an oracle program) given, else
    ``max(300, 2 x length)`` for a demo of ``length`` action steps, else 3000; ``settle_steps`` (the no-op window after
    them) given, else 0.2 x ``max_steps``; ``wallclock_s`` (seconds the actions may take) given, else 1 s x ``max_steps``.
    The episode is cut at ``max_steps + settle_steps`` steps."""
    if max_steps is None:
        max_steps = max(MIN_MAX_STEPS, DEMO_STEP_FACTOR * int(length)) if length else DEFAULT_MAX_STEPS
    if settle_steps is None:
        settle_steps = int(round(SETTLE_FRACTION * max_steps))
    if wallclock_s is None:
        wallclock_s = WALL_S_PER_STEP * max_steps
    return {"max_steps": int(max_steps), "settle_steps": int(settle_steps), "wallclock_s": float(wallclock_s), "demo_length": length,
            "episode_cap": int(max_steps) + int(settle_steps)}


@dataclass
class InstanceRecord:
    """One line of ``metadata/instances.jsonl``; ``files`` are relative to the dataset root. ``episode`` is the instance's
    line of ``metadata/episodes.jsonl`` (its demo)."""

    task_family: str
    task: str
    subtask: str
    instance: int
    task_instance: int
    scene: str
    files: Dict[str, str]
    source: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    root: str = ""
    episode: Dict[str, Any] = field(default_factory=dict)

    def path(self, kind: str) -> str:
        return os.path.join(self.root, self.files[kind])

    def demo_path(self) -> Optional[str]:
        """The instance's demo (``<demos>/<episode["demo"]>``), None when the dataset records none."""
        return os.path.join(demos_root(self.root), self.episode["demo"]) if self.episode.get("demo") else None


def load_index(root: Optional[str] = None) -> List[InstanceRecord]:
    root = os.path.abspath(root or default_root())
    path = os.path.join(root, "metadata", "instances.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no instance index at {path} (set {ROOT_ENV} or --instances-root)")
    episodes = {}
    ep_path = os.path.join(root, "metadata", "episodes.jsonl")
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    episodes[(e["subtask"], int(e["instance"]))] = e
    known = {"task_family", "task", "subtask", "instance", "task_instance", "scene", "files", "source"}
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                recs.append(InstanceRecord(**{k: d[k] for k in known if k in d}, extra={k: v for k, v in d.items() if k not in known}, root=root,
                                           episode=episodes.get((d["subtask"], int(d["instance"])), {})))
    return recs


def by_level(index: Sequence[InstanceRecord]) -> Dict[str, Dict[str, List[InstanceRecord]]]:
    """{level: {name: [records in index order]}} for subtask, task and task_family."""
    out: Dict[str, Dict[str, List[InstanceRecord]]] = {lvl: {} for lvl in LEVELS}
    for rec in index:
        for lvl in LEVELS:
            out[lvl].setdefault(getattr(rec, lvl), []).append(rec)
    return out


def resolve(name: str, indices: Sequence[int] = (0,), root: Optional[str] = None) -> List[InstanceRecord]:
    """The instances ``indices`` of ``name`` (a subtask, task or task family name)."""
    levels = by_level(load_index(root))
    for lvl in LEVELS:
        if name in levels[lvl]:
            recs = levels[lvl][name]
            bad = [i for i in indices if not 0 <= int(i) < len(recs)]
            if bad:
                raise IndexError(f"{lvl} {name} has {len(recs)} instance(s) (indices 0-{len(recs) - 1}); got {bad}")
            return [recs[int(i)] for i in indices]
    raise KeyError(f"{name!r} is not a subtask, task or task family of the dataset; see `python -m ispatialgym.eval.instances list`")


def template_files(tro_path: str) -> Optional[Dict[str, str]]:
    """For an instance file of the dataset layout, its template-level files: ``template`` (scene file), ``template_meta`` and
    ``activity_definitions`` (the BDDL root: the repository's); None for any other layout (a sampler output
    ``<template>/instances/``)."""
    tro_path = os.path.abspath(tro_path)
    inst_dir = os.path.dirname(tro_path)
    m = _INSTANCES_DIR.match(os.path.basename(inst_dir))
    if not m or not tro_path.endswith(TRO_SUFFIX):
        return None
    json_dir = os.path.dirname(inst_dir)
    prefix = os.path.join(json_dir, m.group("prefix"))
    meta = f"{prefix}_0_0_template-meta.json"
    if not os.path.exists(meta):
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(json_dir)))  # <root>/scenes/<scene>/json
    return {"template": f"{prefix}_0_0_template.json", "template_meta": meta, "activity_definitions": ACTIVITY_DEFINITIONS, "root": root}


def goal_record_path(rec: "InstanceRecord") -> str:
    """The internal record of where the instance's goal image came from (outside the dataset; ``set_goal`` writes it)."""
    return os.path.join(records_root(), "goal_images", instance_id(rec) + "-goal.json")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GoalImageError(RuntimeError):
    """No successful replay with a final frame for an instance's goal image."""


def instance_id(rec: InstanceRecord) -> str:
    """The instance's id in the dataset (its instance file's name; ``run_isg`` writes a run to ``<out>/<id>/``)."""
    return os.path.basename(rec.files["tro_state"])[: -len(TRO_SUFFIX)]


def replay_dirs(rec: InstanceRecord) -> List[Tuple[str, str]]:
    """``[(kind, replay dir)]`` the goal image is taken from, in order: ``rerun``, the eval run's replay re-run with the
    current code (``<eval root>/review_videos/replays/<run dir under runs/>/replay_log/<demo id>``), then ``original``, the
    run's own replay (``<run dir>/replay_log/<demo id>``)."""
    run_dir = rec.source["run_dir"]
    demo_id = os.path.basename(os.path.dirname(rec.source["demo_hdf5"]))
    runs_root = run_dir[: run_dir.index(os.sep + "runs" + os.sep) + len(os.sep + "runs")]
    rerun = os.path.join(os.path.dirname(runs_root), "review_videos", "replays", os.path.relpath(run_dir, runs_root), "replay_log", demo_id)
    return [("rerun", rerun), ("original", os.path.join(run_dir, "replay_log", demo_id))]


def _usable_replay(rec: InstanceRecord, replay_dir: str) -> Tuple[Optional[dict], str]:
    """``(meta, "")`` when ``replay_dir`` holds a successful replay of this instance with its final frame, else ``(None, why)``."""
    meta_path = os.path.join(replay_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None, "no meta.json"
    if not os.path.exists(os.path.join(replay_dir, "final_rgb.png")):
        return None, "no final_rgb.png"
    with open(meta_path) as f:
        meta = json.load(f)
    demo_id = os.path.basename(os.path.dirname(rec.source["demo_hdf5"]))
    if meta.get("instance_id") not in (instance_id(rec), demo_id):
        return None, f"a run of {meta.get('instance_id')!r}"
    if meta.get("mode") != "replay":
        return None, f"mode {meta.get('mode')!r}, not a replay"
    if meta.get("success") is not True:
        return None, f"success = {meta.get('success')!r}"
    return meta, ""


def set_goal(rec: InstanceRecord, run: Optional[str] = None) -> Dict[str, Any]:
    """Make the instance's goal image the head camera's last frame (``final_rgb.png``) of a successful replay of its demo:
    of ``run`` (a ``run_isg --out`` dir of a replay of this instance, or the ``<instance id>`` dir in it) when given, else of
    the first usable of ``replay_dirs``. Copies the frame to ``-goal_rgb.png``, writes the internal record
    (``goal_record_path``: the source, its success, the frame's sha256, the image size and the camera: the replay's head
    camera key and intrinsics, the pose of the demo's last ``isg/head_camera_pose`` row, which the replay holds through its
    settle window) and registers the image in the instance file (``register_goal_image``). Returns the record; raises
    :class:`GoalImageError` when no replay qualifies (the demo's own frame is never used)."""
    import h5py
    from PIL import Image

    if run is not None:
        run = os.path.abspath(run)
        candidates = [("run", run if os.path.exists(os.path.join(run, "meta.json")) else os.path.join(run, instance_id(rec)))]
    else:
        candidates = replay_dirs(rec)
    tried = []
    for kind, replay_dir in candidates:
        meta, why = _usable_replay(rec, replay_dir)
        if meta is not None:
            break
        tried.append(f"{kind} {replay_dir}: {why}")
    else:
        raise GoalImageError(f"{rec.subtask} instance {rec.instance}: no usable replay ({'; '.join(tried)})")
    frame = os.path.join(replay_dir, "final_rgb.png")
    demo = rec.demo_path() if rec.demo_path() and os.path.exists(rec.demo_path()) else rec.source["demo_hdf5"]
    with h5py.File(demo, "r") as f:
        pose = [float(v) for v in f["data"]["demo_0"]["isg"]["head_camera_pose"][-1]]
    with Image.open(frame) as im:
        size = [int(im.width), int(im.height)]
    os.makedirs(os.path.dirname(rec.path("goal_rgb")), exist_ok=True)
    shutil.copyfile(frame, rec.path("goal_rgb"))
    info = {
        "source": {"replay": kind, "run_dir": run if kind == "run" else rec.source["run_dir"], "replay_dir": replay_dir, "success": True,
                   "frame": frame},
        "sha256": sha256_of(rec.path("goal_rgb")),
        "image_size": size,
        "camera": {"sensor_key": meta["loader"]["head_camera_key"], "intrinsics": (meta.get("monitors") or meta["qa"])["camera"], "position": pose[:3],
                   "orientation_xyzw": pose[3:7], "pose_from": "demo.hdf5 data/demo_0/isg/head_camera_pose[-1]"},
    }
    os.makedirs(os.path.dirname(goal_record_path(rec)), exist_ok=True)
    with open(goal_record_path(rec), "w") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    register_goal_image(rec)
    return info


def register_goal_image(rec: InstanceRecord) -> bool:
    """Write the instance's goal image into its instance file as ``run_isg --goal-image`` does (``ispatialgym.goal_observation``:
    image path, sha256, size, camera, from the internal record) and refresh its sidecar's ``tro_sha256`` (the sidecar is
    rejected when it does not match). False when the instance has no goal image yet."""
    png, info_path, tro = rec.path("goal_rgb"), goal_record_path(rec), rec.path("tro_state")
    if not (os.path.exists(png) and os.path.exists(info_path)):
        return False
    with open(info_path) as f:
        info = json.load(f)
    cam = info["camera"]
    with open(tro) as f:
        raw = json.load(f)
    raw["ispatialgym"]["goal_observation"] = {
        "sensor_key": cam["sensor_key"],
        "rgb_path": os.path.basename(png),
        "rgb_sha256": sha256_of(png),
        "resolution": info["image_size"],
        "intrinsics": cam.get("intrinsics"),
        "camera_pose_world": {"position": cam["position"], "orientation": cam["orientation_xyzw"]},
        "evaluation_profile": EVALUATION_PROFILE,
        "n_render_iterations": 1,
    }
    with open(tro, "w") as f:
        json.dump(raw, f, indent=2)
    sidecar = rec.path("oracle") if "oracle" in rec.files else None
    if sidecar and os.path.exists(sidecar):
        with open(sidecar) as f:
            sc = json.load(f)
        sc["tro_sha256"] = sha256_of(tro)
        with open(sidecar, "w") as f:
            json.dump(sc, f, indent=2)
            f.write("\n")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List / resolve the eval instances (task family -> task -> subtask -> instances).")
    ap.add_argument("--root", default=None, help=f"dataset root (default ${ROOT_ENV} or <repo>/datasets/{DATASET_NAME})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--level", choices=("task_family", "task", "subtask"), default="task")
    sh = sub.add_parser("show")
    sh.add_argument("name")
    sh.add_argument("indices", type=int, nargs="*", default=[0])
    sg = sub.add_parser("set-goal", help="make an instance's goal image the final frame of a successful replay")
    sg.add_argument("name")
    sg.add_argument("index", type=int)
    sg.add_argument("--run", default=None, help="a run_isg --out dir of a replay of this instance (e.g. at another --image-size); "
                                                "default: the eval run's re-run replay, else its original replay")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for name, recs in by_level(load_index(a.root))[a.level].items():
            print(f"{name}\t{len(recs)}")
        return 0
    if a.cmd == "set-goal":
        (rec,) = resolve(a.name, [a.index], a.root)
        try:
            info = set_goal(rec, a.run)
        except GoalImageError as e:
            print(f"set-goal: {e}", file=sys.stderr)
            return 1
        print(json.dumps({"subtask": rec.subtask, "instance": rec.instance, "goal_rgb": rec.path("goal_rgb"), "image_size": info["image_size"],
                          "replay": info["source"]["replay"], "replay_dir": info["source"]["replay_dir"]}))
        return 0
    for i, rec in zip(a.indices, resolve(a.name, a.indices, a.root)):
        print(json.dumps({"index": i, "task_family": rec.task_family, "task": rec.task, "subtask": rec.subtask, "instance": rec.instance,
                          "scene": rec.scene, "tro_state": rec.path("tro_state"), "goal_rgb": rec.path("goal_rgb"), "demo": rec.demo_path(),
                          "demo_length": rec.episode.get("length")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
