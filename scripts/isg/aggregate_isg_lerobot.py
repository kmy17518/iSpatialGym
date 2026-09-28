#!/usr/bin/env python3
"""Per-task and multi-task LeRobot v3 datasets from the production shards, as BEHAVIOR's aggregate_lerobot_task_shards.py
builds them (OmniGibson/scripts/learning): lerobot's ``aggregate_datasets`` packs the episodes of every shard into shared
data / video files, then BEHAVIOR's ``verify_aggregate`` checks the result.

    python scripts/isg/aggregate_isg_lerobot.py --sweep-root <prod root>/<sweep> --output-root <out> --task <alias> [--task ...] [--multitask]

Per task the inputs are the collection shards (``<slice>/lerobot/shard_*``: successful demos without a mid-episode retry,
written while recording) and the replay shards (``<task dir>/replay/lerobot/*``: replay_obs_isg.py of retried demos and of
demos recorded before the BEHAVIOR output). Output ``<out>/isg/<task>`` (rebuilt) and ``<out>/isg/<task>.episodes.jsonl``
(per episode index: the shard, its episode index there and the instance / demo it came from). ``--multitask``: every task
with a per-task dataset listed in ``<out>/isg_multitask.tasks.json`` plus this run's tasks -> ``<out>/isg_multitask``, rebuilt
from the per-task datasets so a task is never included twice. Only complete shards (``meta/info.json`` present, finalized)
are read; verify_aggregate runs on every output with the expected episode count.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import omnigibson as og  # noqa: F401  (resolves BEHAVIOR-1K for verify_aggregate)

from ispatialgym.oracle import lerobot_compat

lerobot_compat.install()
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "aggregate_lerobot_task_shards",
    os.path.join(os.path.dirname(os.path.dirname(og.__file__)), "scripts", "learning", "aggregate_lerobot_task_shards.py"))
B1K_AGG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B1K_AGG)
verify_aggregate = B1K_AGG.verify_aggregate


def dataset_episodes(root: str) -> int:
    with open(os.path.join(root, "meta", "info.json")) as f:
        return int(json.load(f).get("total_episodes", 0))


def complete(root: str) -> bool:
    return os.path.exists(os.path.join(root, "meta", "info.json")) and bool(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))


def episode_sources(sweep_root: str, task: str) -> dict:
    """``{shard root: {episode index: {instance, demo, job_dir}}}`` from the results rows (collection) and the replay manifests."""
    src = {}
    for rp in glob.glob(os.path.join(sweep_root, task, "i*", "results.jsonl")):
        if ".empty-" in rp:
            continue
        rows = {}
        for line in open(rp):
            if line.strip():
                r = json.loads(line)
                rows[r["instance"]] = r
        for iid, r in rows.items():
            b = r.get("behavior_lerobot") or {}
            if b.get("shard") and b.get("episode_index") is not None:
                src.setdefault(b["shard"], {})[int(b["episode_index"])] = {"instance": iid, "demo": r.get("out"), "job_dir": os.path.dirname(rp),
                                                                          "source": "collection"}
    for mp in glob.glob(os.path.join(sweep_root, task, "replay", "lerobot", "*.episodes.json")):
        shard = mp[: -len(".episodes.json")]
        for k, e in enumerate(json.load(open(mp))):
            src.setdefault(shard, {})[k] = {**e, "source": "replay"}
    return src


def aggregate(inputs, repo_id: str, out_root: str, data_mb, video_mb) -> int:
    tmp = out_root + ".building"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    expected = sum(dataset_episodes(r) for r in inputs)
    aggregate_datasets(repo_ids=[os.path.relpath(r, os.path.dirname(r)) for r in inputs], roots=[Path(r) for r in inputs],
                       aggr_repo_id=repo_id, aggr_root=Path(tmp), data_files_size_in_mb=data_mb, video_files_size_in_mb=video_mb)
    verify_aggregate(Path(tmp), expected, require_packed=expected > 1)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.replace(tmp, out_root)
    return expected


def aggregate_task(args, task: str) -> dict:
    task_dir = os.path.join(args.sweep_root, task)
    shards = sorted(r for r in glob.glob(os.path.join(task_dir, "i*", "lerobot", "shard_*")) + glob.glob(os.path.join(task_dir, "replay", "lerobot", "*"))
                    if os.path.isdir(r) and ".empty-" not in r)
    ready = [r for r in shards if complete(r) and dataset_episodes(r) > 0]
    skipped = sorted(set(shards) - set(ready))
    out_root = os.path.join(args.output_root, "isg", task)
    n = aggregate(ready, f"isg/{task}", out_root, args.data_files_size_in_mb, args.video_files_size_in_mb) if ready else 0
    src = episode_sources(args.sweep_root, task)
    with open(out_root + ".episodes.jsonl", "w") as f:
        k = 0
        for r in ready:
            for e in range(dataset_episodes(r)):
                f.write(json.dumps({"episode_index": k, "shard": r, "shard_episode_index": e, **src.get(os.path.abspath(r), {}).get(e, {})}) + "\n")
                k += 1
    rec = {"task": task, "dataset": out_root, "episodes": n, "shards": len(ready), "skipped_shards": skipped}
    print("AGGREGATE_TASK " + json.dumps(rec), flush=True)
    return rec


def aggregate_multitask(args, tasks) -> dict:
    listed = os.path.join(args.output_root, "isg_multitask.tasks.json")
    included = json.load(open(listed)) if os.path.exists(listed) else []
    for t in tasks:
        if t not in included:
            included.append(t)
    roots = [os.path.join(args.output_root, "isg", t) for t in included if complete(os.path.join(args.output_root, "isg", t))]
    out_root = os.path.join(args.output_root, "isg_multitask")
    n = aggregate(roots, "isg_multitask", out_root, args.data_files_size_in_mb, args.video_files_size_in_mb)
    with open(listed, "w") as f:
        json.dump(included, f, indent=1)
    rec = {"dataset": out_root, "episodes": n, "tasks": included}
    print("AGGREGATE_MULTITASK " + json.dumps(rec), flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--task", action="append", default=[])
    ap.add_argument("--multitask", action="store_true")
    ap.add_argument("--data-files-size-in-mb", type=int, default=None)
    ap.add_argument("--video-files-size-in-mb", type=int, default=None)
    args = ap.parse_args(argv)
    for t in args.task:
        aggregate_task(args, t)
    if args.multitask:
        aggregate_multitask(args, args.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
