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
``ispatialgym/assets/bddl/eval_release/activity_definitions`` (BEHAVIOR's bddl package holds its problems): the release's problems,
numbered from 0 per activity, as the dataset names them. The sampler's ``ispatialgym/assets/bddl/activity_definitions`` keeps this
repository's numbering (the train templates and TROs name those ids; their runs carry their own BDDL copies).

``run_isg --task`` gives an instance ``max(300, round(1.5 x`` its demo's length ``))`` steps by default and a ``--settle-steps`` window of 45
steps (oracle, policy) or its demo's settle rows (replay; ``step_budget``).

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
keeps one goal image, and a re-export keeps a goal image set from such a run. ``--measured-targets`` also sets the pose targets
of the instance's ``(:measure ...)``-only atoms to where that replay left them (its ``meta.json`` ``final_target_poses``), so the
goal image and the measured targets show the same scene; the internal record keeps the values replaced.

A name resolves at the most specific level that has it (subtask, then task, then task family); the instance indices count
that level's instances in the index order (a task's instances are its subtasks' instances, subtask by subtask).

    python -m ispatialgym.eval.instances [--root R] list [--level task_family|task|subtask]
    python -m ispatialgym.eval.instances [--root R] show <name> [<index> ...]
    python -m ispatialgym.eval.instances [--root R] set-goal <name> <index> [--run <run_isg out dir> [--measured-targets]]
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
HF_REPO_ID = "kmy17518/ispatialgym-eval-data"
HF_REVISION = "v1.0"
DATASET_NAME = "ispatialgym-eval-instances"
DEMOS_NAME = "ispatialgym-eval-demos"
TRO_SUFFIX = "-tro_state.json"
DEMO_SUFFIX = "-demo.hdf5"
LEVELS = ("subtask", "task", "task_family")
EVALUATION_PROFILE = "isg-v1"  # run_isg.EVALUATION_PROFILE (run_isg imports the simulator)
#: step budget of run_isg --task: max(MIN_MAX_STEPS, DEMO_STEP_FACTOR x the demo's length) steps, WALL_S_PER_STEP seconds per
#: step; DEFAULT_MAX_STEPS without a demo; a settle window (run_isg --settle-steps) of DEFAULT_SETTLE_STEPS (oracle, policy)
MIN_MAX_STEPS = 300
DEMO_STEP_FACTOR = 1.5  # 2 until 2026-10-23 11:50 (the evaluation plan's change: 1.5 x the demo's length)
WALL_S_PER_STEP = 1.0
DEFAULT_MAX_STEPS = 3000
DEFAULT_SETTLE_STEPS = 45
MODES = ("oracle", "replay", "policy")
_INSTANCES_DIR = re.compile(r"^(?P<prefix>.+_task_.+)_instances$")


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: the release's BDDL problems, the eval dataset's numbering (95b03b3b's activity_definitions; the sampler's own problems keep this
#: repository's numbering under ``assets/bddl/activity_definitions``)
ACTIVITY_DEFINITIONS = os.path.join(REPO, "ispatialgym", "assets", "bddl", "eval_release", "activity_definitions")


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


def download(
    local_dir: Optional[str] = None,
    revision: str = HF_REVISION,
    repo_id: str = HF_REPO_ID,
    demos: bool = False,
) -> str:
    """Download the dataset from the Hugging Face Hub into ``local_dir`` (default :func:`data_root`)."""
    from huggingface_hub import snapshot_download

    local_dir = local_dir or data_root()
    patterns = [f"{DATASET_NAME}/*"] + ([f"{DEMOS_NAME}/*"] if demos else [])
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=patterns,
    )
    return os.path.join(local_dir, DATASET_NAME)


def step_budget(length: Optional[int], max_steps: Optional[int] = None, settle_steps: Optional[int] = None,
                wallclock_s: Optional[float] = None, mode: str = "oracle", demo_settle_steps: Optional[int] = None,
                instance_max_steps: Optional[int] = None) -> Dict[str, Any]:
    """``run_isg``'s step budget. ``max_steps`` (the actions: policy steps, a replay, an oracle program) given, else
    ``instance_max_steps`` when given, else ``max(300, round(1.5 x length))`` for a demo of ``length`` steps, else 3000;
    ``wallclock_s`` (seconds the actions may take) given, else 1 s x ``max_steps``. ``settle_steps`` - the
    ``pose_settled`` and settle-eval window, in every mode - given, else 45 (oracle, policy) or the demo's settle rows
    (replay: ``demo_settle_steps``). An oracle program's epilogue is exactly ``settle_steps`` long and reserved: its
    episode is cut at ``max_steps + settle_steps``; a replay streams its demo's action rows and the first
    ``settle_steps`` settle rows, a policy acts until ``max_steps``: theirs at ``max_steps``. Raises ValueError when the
    window cannot fit: below one step, a replay without settle rows or with fewer than ``settle_steps``, a policy whose
    ``max_steps`` is shorter than it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if max_steps is None:
        if instance_max_steps is not None:
            max_steps = int(instance_max_steps)
        else:
            max_steps = max(MIN_MAX_STEPS, int(round(DEMO_STEP_FACTOR * int(length)))) if length else DEFAULT_MAX_STEPS
    if mode == "replay" and demo_settle_steps is None:
        raise ValueError("--mode replay: the demo's settle rows are unknown (isg/phase, or the eval dataset's settle_steps of a LeRobot "
                         "episode), so the --settle-steps window cannot be placed")
    if settle_steps is None:
        settle_steps = int(demo_settle_steps) if mode == "replay" else DEFAULT_SETTLE_STEPS
    settle_steps = int(settle_steps)
    if settle_steps < 1:
        raise ValueError(f"--settle-steps must be at least 1 (the pose_settled window), got {settle_steps}")
    if mode == "replay" and settle_steps > int(demo_settle_steps):
        raise ValueError(f"--settle-steps {settle_steps} exceeds the demo's {int(demo_settle_steps)} settle rows: a replay streams only "
                         "recorded rows, so the window cannot fit")
    if mode == "policy" and settle_steps > int(max_steps):
        raise ValueError(f"--settle-steps {settle_steps} exceeds --max-steps {int(max_steps)}: the window cannot fit in the episode")
    if wallclock_s is None:
        wallclock_s = WALL_S_PER_STEP * max_steps
    return {"mode": mode, "max_steps": int(max_steps), "settle_steps": settle_steps, "wallclock_s": float(wallclock_s), "demo_length": length,
            "demo_settle_steps": demo_settle_steps, "episode_cap": int(max_steps) + (settle_steps if mode == "oracle" else 0)}


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

    def demo_path(self, kind: str = "demo") -> Optional[str]:
        """The instance's demo (``<demos>/<episode[kind]>``), None when the dataset records none."""
        rel = self.episode.get(kind)
        return os.path.join(demos_root(self.root), rel) if rel else None

    @property
    def instance_id(self) -> str:
        """The instance's id in the dataset (its instance file's name; ``run_isg`` writes a run to ``<out>/<id>/``)."""
        return os.path.basename(self.files["tro_state"])[: -len(TRO_SUFFIX)]

    @property
    def max_steps(self) -> Optional[int]:
        """The instance's ``max_steps`` from ``instances.jsonl``, else ``max(300, round(1.5 * length))`` when a demo length is known."""
        if self.extra.get("max_steps") is not None:
            return int(self.extra["max_steps"])
        length = self.extra.get("length") or self.episode.get("length")
        return max(MIN_MAX_STEPS, int(round(DEMO_STEP_FACTOR * int(length)))) if length else None


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


def record_of(tro_path: str, root: Optional[str] = None) -> Optional[InstanceRecord]:
    """The index record of an instance file, None when it is not in the index."""
    tro_path = os.path.abspath(tro_path)
    return next((r for r in load_index(root) if os.path.abspath(r.path("tro_state")) == tro_path), None)


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
    root = os.path.dirname(os.path.dirname(os.path.dirname(json_dir)))  # <root>/scenes/<scene>/json
    # policy eval 2026-11-15: a dataset with several templates per subtask (scripts/batched_replay/export_eval.py writes
    # <prefix>_<t>_<i>_template-*: t = the subtask's t-th distinct template) names each instance's own template in its index record;
    # the first-template convention below would load another instance's template for t > 0
    rec = _index_record(root, tro_path)
    if rec is not None and rec.files.get("template") and rec.files.get("template_meta"):
        template, meta = rec.path("template"), rec.path("template_meta")
    else:
        template, meta = f"{prefix}_0_0_template.json", f"{prefix}_0_0_template-meta.json"
    if not os.path.exists(meta):
        return None
    # the dataset's own copy of the instance's BDDL problem (re-export 2026-11-16: files.bddl_root), else the overlay / release tree
    bddl = rec.path("bddl_root") if rec is not None and rec.files.get("bddl_root") else None
    if bddl is None or not os.path.isdir(bddl):
        bddl = instance_bddl_root(tro_path)
    return {"template": template, "template_meta": meta, "activity_definitions": bddl, "root": root}


#: policy eval 2026-11-15: per-instance BDDL roots (<root>/<instance id>/<activity>/problem<id>.bddl, the exact problem each instance was
#: collected with; scripts/policy_eval/build_bddl_overlay.py) for a dataset whose problems are not the release tree's (the v2 batched-replay
#: export names each template's own problem id)
BDDL_OVERLAY_ENV = "ISPATIALGYM_EVAL_BDDL_OVERLAY"


def instance_bddl_root(tro_path: str) -> str:
    """The BDDL root an instance's problem is loaded from: ``$ISPATIALGYM_EVAL_BDDL_OVERLAY/<instance id>`` when it exists, else the
    repository's release tree (:data:`ACTIVITY_DEFINITIONS`)."""
    overlay = os.environ.get(BDDL_OVERLAY_ENV)
    if overlay:
        d = os.path.join(overlay, os.path.basename(tro_path)[: -len(TRO_SUFFIX)])
        if os.path.isdir(d):
            return d
    return ACTIVITY_DEFINITIONS


_INDEX_BY_TRO: Dict[str, Dict[str, InstanceRecord]] = {}


def _index_record(root: str, tro_path: str) -> Optional[InstanceRecord]:
    """The index record of ``tro_path`` in the dataset at ``root`` (cached per root; None without an index)."""
    if root not in _INDEX_BY_TRO:
        try:
            _INDEX_BY_TRO[root] = {os.path.abspath(r.path("tro_state")): r for r in load_index(root)}
        except FileNotFoundError:
            _INDEX_BY_TRO[root] = {}
    return _INDEX_BY_TRO[root].get(os.path.abspath(tro_path))


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


def _usable_replay(rec: InstanceRecord, replay_dir: str, measured_targets: bool = False) -> Tuple[Optional[dict], str]:
    """``(meta, "")`` when ``replay_dir`` holds a successful replay of this instance with its final frame, else ``(None, why)``.
    ``measured_targets``: the replay re-pins the measured targets, so its symbolic success suffices (its geometric success
    failed only on the targets being re-pinned) and it must record their final poses (``final_target_poses``)."""
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
    if measured_targets:
        if meta.get("symbolic_success") is not True:
            return None, f"symbolic_success = {meta.get('symbolic_success')!r}"
        if not meta.get("final_target_poses"):
            return None, "no final_target_poses (a replay before 2026-10-14)"
    elif meta.get("success") is not True:
        return None, f"success = {meta.get('success')!r}"
    return meta, ""


def repin_measured_targets(rec: InstanceRecord, final_poses: Dict[str, Any]) -> Dict[str, Any]:
    """Set the pose targets of the instance's ``(:measure ...)``-only ``pose_reached`` atoms (not in ``:goal``) in its instance file
    to ``final_poses`` (a replay's ``final_target_poses``: each target's pose in its own frame and link); a component the target
    does not compare stays None. Returns ``{inst: {"position_was", "position", "orientation_was", "orientation"}}``; raises
    :class:`GoalImageError` when a target has no final pose or another frame / link."""
    from ispatialgym.activity.problem import load_problem

    tro = rec.path("tro_state")
    with open(tro) as f:
        raw = json.load(f)
    block = raw["ispatialgym"]
    compiled = load_problem(block["activity"], int(block["activity_definition_id"])).compile()
    changed: Dict[str, Any] = {}
    for inst in compiled.measure_pose_reached_args:
        if compiled.is_gating("pose_reached", inst) or inst not in block.get("pose_targets", {}):
            continue
        tgt, fin = block["pose_targets"][inst], final_poses.get(inst) or {}
        if "position" not in fin or "orientation" not in fin:
            raise GoalImageError(f"{rec.subtask}: no final pose of the measured target {inst} ({fin.get('error', 'not recorded')})")
        if fin.get("frame") != tgt.get("frame") or fin.get("link") != tgt.get("link"):
            raise GoalImageError(f"{rec.subtask}: the final pose of {inst} is in frame {fin.get('frame')!r} / link {fin.get('link')!r}, the target's "
                                 f"{tgt.get('frame')!r} / {tgt.get('link')!r}")
        rec_i: Dict[str, Any] = {}
        for key in ("position", "orientation"):
            if tgt.get(key) is not None:
                rec_i.update({f"{key}_was": tgt[key], key: [float(v) for v in fin[key]]})
                tgt[key] = rec_i[key]
        changed[inst] = rec_i
    with open(tro, "w") as f:
        json.dump(raw, f, indent=2)
    return changed


def set_goal(rec: InstanceRecord, run: Optional[str] = None, measured_targets: bool = False) -> Dict[str, Any]:
    """Make the instance's goal image the head camera's last frame (``final_rgb.png``) of a successful replay of its demo:
    of ``run`` (a ``run_isg --out`` dir of a replay of this instance, or the ``<instance id>`` dir in it) when given, else of
    the first usable of ``replay_dirs``. Copies the frame to ``-goal_rgb.png``, writes the internal record
    (``goal_record_path``: the source, its success, the frame's sha256, the image size and the camera: the replay's head
    camera key and intrinsics, the pose of the demo's last ``isg/head_camera_pose`` row, which the replay holds through its
    settle window) and registers the image in the instance file (``register_goal_image``). ``measured_targets`` (needs
    ``run``): the same replay also re-pins the measured targets (:func:`repin_measured_targets`; the replay's symbolic success
    suffices), recorded in the internal record. Returns the record; raises :class:`GoalImageError` when no replay qualifies
    (the demo's own frame is never used)."""
    import h5py
    from PIL import Image

    if measured_targets and run is None:
        raise GoalImageError(f"{rec.subtask} instance {rec.instance}: --measured-targets re-pins from one replay: give --run")
    if run is not None:
        run = os.path.abspath(run)
        candidates = [("run", run if os.path.exists(os.path.join(run, "meta.json")) else os.path.join(run, instance_id(rec)))]
    else:
        candidates = replay_dirs(rec)
    tried = []
    for kind, replay_dir in candidates:
        meta, why = _usable_replay(rec, replay_dir, measured_targets)
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
        "source": {"replay": kind, "run_dir": run if kind == "run" else rec.source["run_dir"], "replay_dir": replay_dir, "success": meta.get("success") is True,
                   "frame": frame},
        "sha256": sha256_of(rec.path("goal_rgb")),
        "image_size": size,
        "camera": {"sensor_key": meta["loader"]["head_camera_key"], "intrinsics": (meta.get("monitors") or meta["qa"])["camera"], "position": pose[:3],
                   "orientation_xyzw": pose[3:7], "pose_from": "demo.hdf5 data/demo_0/isg/head_camera_pose[-1]"},
    }
    if measured_targets:
        info["source"]["measured_targets"] = True
        info["measured_targets"] = repin_measured_targets(rec, meta["final_target_poses"])
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
    ap = argparse.ArgumentParser(description="Download / list / resolve the eval instances (task family -> task -> subtask -> instances).")
    ap.add_argument("--root", default=None, help=f"dataset root (default ${ROOT_ENV} or <repo>/datasets/{DATASET_NAME})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download the dataset from the Hugging Face Hub into the data path")
    dl.add_argument("--revision", default=HF_REVISION)
    dl.add_argument("--local-dir", default=None, help="download target (default: OmniGibson's data path)")
    dl.add_argument("--demos", action="store_true", help=f"also download {DEMOS_NAME} (HDF5 demos and rollout videos)")
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
    sg.add_argument("--measured-targets", action="store_true",
                    help="with --run: also set the pose targets of the (:measure ...)-only atoms to where that replay left them "
                         "(meta.json final_target_poses; its symbolic success suffices), so the goal image and the measured targets agree")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        print(download(a.local_dir, revision=a.revision, demos=a.demos))
        return 0
    if a.cmd == "list":
        for name, recs in by_level(load_index(a.root))[a.level].items():
            print(f"{name}\t{len(recs)}")
        return 0
    if a.cmd == "set-goal":
        (rec,) = resolve(a.name, [a.index], a.root)
        try:
            info = set_goal(rec, a.run, measured_targets=a.measured_targets)
        except GoalImageError as e:
            print(f"set-goal: {e}", file=sys.stderr)
            return 1
        print(json.dumps({"subtask": rec.subtask, "instance": rec.instance, "goal_rgb": rec.path("goal_rgb"), "image_size": info["image_size"],
                          "replay": info["source"]["replay"], "replay_dir": info["source"]["replay_dir"],
                          "measured_targets": info.get("measured_targets")}))
        return 0
    for i, rec in zip(a.indices, resolve(a.name, a.indices, a.root)):
        print(json.dumps({"index": i, "task_family": rec.task_family, "task": rec.task, "subtask": rec.subtask, "instance": rec.instance,
                          "scene": rec.scene, "tro_state": rec.path("tro_state"), "goal_rgb": rec.path("goal_rgb"), "demo": rec.demo_path(),
                          "demo_length": rec.episode.get("length")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
