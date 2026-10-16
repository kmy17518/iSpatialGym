"""The iSpatialGym eval instances: download, index and name resolution.

The dataset ``ispatialgym-eval-instances`` sits in OmniGibson's data folder (``$OMNIGIBSON_DATA_PATH``, default
``<repo>/Behavior-1K/datasets``), with optional reference demonstrations and rollout videos in its sibling folder
``ispatialgym-eval-demos``::

    metadata/instances.jsonl    one line per instance: names, scene, problem, max_steps, files
    metadata/episodes.jsonl     one line per instance's demo: length, demo (.hdf5), head_video, external_video
    metadata/tasks.yaml         task family -> task -> [subtask, ...]
    scenes/<scene>/json/<scene>_task_<subtask>_0_0_template.json         the scene with the task objects
    scenes/<scene>/json/<scene>_task_<subtask>_0_0_template-meta.json    the template facts the loader reads
    scenes/<scene>/json/<scene>_task_<subtask>_instances/
        <scene>_task_<subtask>_0_<i>_template-tro_state.json   instance i
        <scene>_task_<subtask>_0_<i>_template-goal_rgb.png     its goal image
    ../ispatialgym-eval-demos/scenes/<scene>/json/<scene>_task_<subtask>_instances/
        <scene>_task_<subtask>_0_<i>_template-demo.hdf5        reference demonstration trajectory
        <scene>_task_<subtask>_0_<i>_template-head_rgb.mp4     reference rollout (head camera)
        <scene>_task_<subtask>_0_<i>_template-external_rgb.mp4 reference rollout (external camera)

An instance names its BDDL problem (``<activity>-<id>``), which ships with this package. A name resolves at the most
specific level that has it (subtask, task, task family); indices count that level's instances in index order.

    python -m ispatialgym.eval.instances download [--demos] [--revision REV]
    python -m ispatialgym.eval.instances list [--level task_family|task|subtask]
    python -m ispatialgym.eval.instances show NAME [INDEX ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

HF_REPO_ID = "kmy17518/ispatialgym-eval-data"
HF_REVISION = "v1.0"
DATASET_NAME = "ispatialgym-eval-instances"
DEMOS_NAME = "ispatialgym-eval-demos"
ROOT_ENV = "ISPATIALGYM_INSTANCES_PATH"
DEMOS_ENV = "ISPATIALGYM_DEMOS_PATH"
TRO_SUFFIX = "-tro_state.json"
LEVELS = ("subtask", "task", "task_family")

DEFAULT_MAX_STEPS = 3000
DEFAULT_SETTLE_STEPS = 45
WALL_S_PER_STEP = 1.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INSTANCES_DIR = re.compile(r"^(?P<prefix>.+_task_.+)_instances$")


def data_root() -> str:
    """OmniGibson's data folder: ``$OMNIGIBSON_DATA_PATH``, else ``<repo>/Behavior-1K/datasets``."""
    return os.environ.get("OMNIGIBSON_DATA_PATH") or os.path.join(_REPO_ROOT, "Behavior-1K", "datasets")


def default_root() -> str:
    return os.environ.get(ROOT_ENV) or os.path.join(data_root(), DATASET_NAME)


def demos_root(root: str | None = None) -> str:
    return os.environ.get(DEMOS_ENV) or os.path.join(os.path.dirname(os.path.abspath(root or default_root())), DEMOS_NAME)


def download(
    local_dir: str | None = None,
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


@dataclass
class InstanceRecord:
    """One line of ``metadata/instances.jsonl``; ``files`` are relative to the dataset root."""

    task_family: str
    task: str
    subtask: str
    instance: int
    task_instance: int
    scene: str
    problem: str
    max_steps: int | None
    files: dict[str, str]
    root: str = ""
    episode: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def path(self, kind: str) -> str:
        return os.path.join(self.root, self.files[kind])

    def demo_path(self, kind: str = "demo") -> str | None:
        rel = self.episode.get(kind)
        return os.path.join(demos_root(self.root), rel) if rel else None

    @property
    def instance_id(self) -> str:
        """The instance file's name without its suffix; ``run_isg`` writes a run to ``<out>/<instance id>/``."""
        return os.path.basename(self.files["tro_state"])[: -len(TRO_SUFFIX)]


def load_index(root: str | None = None) -> list[InstanceRecord]:
    root = os.path.abspath(root or default_root())
    path = os.path.join(root, "metadata", "instances.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no instance index at {path}; run `python -m ispatialgym.eval.instances download` or set {ROOT_ENV}"
        )
    episodes: dict[tuple[str, int], dict[str, Any]] = {}
    ep_path = os.path.join(root, "metadata", "episodes.jsonl")
    if os.path.exists(ep_path):
        with open(ep_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    episodes[(e["subtask"], int(e["instance"]))] = e
    known = {f for f in InstanceRecord.__dataclass_fields__} - {"root", "episode", "extra"}
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                fields = {k: d.get(k) for k in known}
                records.append(
                    InstanceRecord(
                        **fields,
                        root=root,
                        episode=episodes.get((d["subtask"], int(d["instance"])), {}),
                        extra={k: v for k, v in d.items() if k not in known},
                    )
                )
    return records


def by_level(index: Sequence[InstanceRecord]) -> dict[str, dict[str, list[InstanceRecord]]]:
    """``{level: {name: [records in index order]}}`` for subtask, task and task family."""
    out: dict[str, dict[str, list[InstanceRecord]]] = {level: {} for level in LEVELS}
    for rec in index:
        for level in LEVELS:
            out[level].setdefault(getattr(rec, level), []).append(rec)
    return out


def resolve(name: str, indices: Sequence[int] = (0,), root: str | None = None) -> list[InstanceRecord]:
    """The instances ``indices`` of ``name`` (a subtask, task or task family)."""
    levels = by_level(load_index(root))
    for level in LEVELS:
        if name in levels[level]:
            records = levels[level][name]
            bad = [i for i in indices if not 0 <= int(i) < len(records)]
            if bad:
                raise IndexError(
                    f"{level} {name} has {len(records)} instance(s) (indices 0-{len(records) - 1}); got {bad}"
                )
            return [records[int(i)] for i in indices]
    raise KeyError(
        f"{name!r} is not a subtask, task or task family of the dataset; see `python -m ispatialgym.eval.instances list`"
    )


def record_of(tro_path: str, root: str | None = None) -> InstanceRecord | None:
    """The index record of an instance file, None when it is not in the index."""
    tro_path = os.path.abspath(tro_path)
    return next((r for r in load_index(root) if os.path.abspath(r.path("tro_state")) == tro_path), None)


def template_files(tro_path: str) -> dict[str, str]:
    """``template`` (scene file), ``template_meta`` and dataset ``root`` of an instance file of the dataset layout."""
    tro_path = os.path.abspath(tro_path)
    inst_dir = os.path.dirname(tro_path)
    m = _INSTANCES_DIR.match(os.path.basename(inst_dir))
    if not m or not tro_path.endswith(TRO_SUFFIX):
        raise ValueError(f"{tro_path} is not an instance file of the dataset layout (<...>_instances/<id>{TRO_SUFFIX})")
    json_dir = os.path.dirname(inst_dir)
    prefix = os.path.join(json_dir, m.group("prefix"))
    files = {
        "template": f"{prefix}_0_0_template.json",
        "template_meta": f"{prefix}_0_0_template-meta.json",
        "root": os.path.dirname(os.path.dirname(os.path.dirname(json_dir))),
    }
    missing = [p for k, p in files.items() if k != "root" and not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"template files of {tro_path} not found: {missing}")
    return files


def step_budget(
    instance_max_steps: int | None,
    max_steps: int | None = None,
    settle_steps: int | None = None,
    wallclock_s: float | None = None,
    settle_eval: bool = False,
) -> dict[str, Any]:
    """The episode budget: ``max_steps`` (default the instance's), the settle window (default 45 steps) and the wall
    clock the policy's steps may take (default 1 s per step). With settle eval on, the window must fit in the episode."""
    if max_steps is None:
        max_steps = instance_max_steps if instance_max_steps is not None else DEFAULT_MAX_STEPS
    if settle_steps is None:
        settle_steps = DEFAULT_SETTLE_STEPS
    max_steps, settle_steps = int(max_steps), int(settle_steps)
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    if settle_steps < 1:
        raise ValueError(f"settle_steps must be at least 1, got {settle_steps}")
    if settle_eval and settle_steps > max_steps:
        raise ValueError(
            f"settle_steps {settle_steps} exceeds max_steps {max_steps}: the settle window cannot fit in the episode"
        )
    wallclock_s = WALL_S_PER_STEP * max_steps if wallclock_s is None else float(wallclock_s)
    return {
        "max_steps": max_steps,
        "settle_steps": settle_steps,
        "wallclock_s": float(wallclock_s),
        "instance_max_steps": instance_max_steps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download and browse the iSpatialGym eval instances.")
    parser.add_argument(
        "--root", default=None, help=f"dataset root (default ${ROOT_ENV} or <data path>/{DATASET_NAME})"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download the dataset from the Hugging Face Hub into the data path")
    dl.add_argument("--revision", default=HF_REVISION)
    dl.add_argument("--local-dir", default=None, help="download target (default: OmniGibson's data path)")
    dl.add_argument("--demos", action="store_true", help=f"also download {DEMOS_NAME} (HDF5 demos and rollout videos)")
    ls = sub.add_parser("list", help="names and instance counts")
    ls.add_argument("--level", choices=LEVELS[::-1], default="task")
    sh = sub.add_parser("show", help="what instance indices resolve to")
    sh.add_argument("name")
    sh.add_argument("indices", type=int, nargs="*", default=[0])
    args = parser.parse_args(argv)

    if args.cmd == "download":
        print(download(args.local_dir, revision=args.revision, demos=args.demos))
    elif args.cmd == "list":
        for name, records in by_level(load_index(args.root))[args.level].items():
            print(f"{name}\t{len(records)}")
    else:
        for i, rec in zip(args.indices, resolve(args.name, args.indices, args.root), strict=True):
            info = {
                "index": i,
                "task_family": rec.task_family,
                "task": rec.task,
                "subtask": rec.subtask,
                "instance": rec.instance,
                "scene": rec.scene,
                "max_steps": rec.max_steps,
                "tro_state": rec.path("tro_state"),
                "goal_rgb": rec.path("goal_rgb"),
            }
            if rec.demo_path():
                info["demo"] = rec.demo_path()
            print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
