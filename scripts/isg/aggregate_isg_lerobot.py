#!/usr/bin/env python3
"""Per-task and multi-task LeRobot v3 datasets from the production shards, as BEHAVIOR's aggregate_lerobot_task_shards.py
builds them (OmniGibson/scripts/learning): lerobot's ``aggregate_datasets`` packs the episodes of every shard into shared
data / video files, then BEHAVIOR's ``verify_aggregate`` checks the result.

    python scripts/isg/aggregate_isg_lerobot.py --sweep-root <prod root>/<sweep> --output-root <out> --task <variant> [--task ...] [--multitask]

Per task the inputs are the collection shards (``<slice>/lerobot/shard_*``: successful demos without a mid-episode retry,
written while recording) and the replay shards (``<task dir>/replay/lerobot/*``: replay_obs_isg.py of retried demos and of
demos recorded before the BEHAVIOR output). Output ``<out>/isg/<task>`` (rebuilt) and ``<out>/isg/<task>.episodes.jsonl``
(per episode index: the shard, its episode index there and the instance / demo it came from). ``--multitask``: every task
with a per-task dataset listed in ``<out>/isg_multitask.tasks.json`` plus this run's tasks -> ``<out>/isg_multitask``, rebuilt
from the per-task datasets so a task is never included twice, in BEHAVIOR's multi-task layout (2026-challenge-demos, read by
omnigibson.eval.utils.score_utils): the k-th listed task has task index k and is chunk k of the data, the videos and the
episode metadata. Only complete shards (``meta/info.json`` present, finalized) are read; every output is verified with the
expected episode count.
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
from lerobot.datasets.aggregate import (  # noqa: E402
    aggregate_data,
    aggregate_datasets,
    aggregate_metadata,
    aggregate_videos,
    finalize_aggregation,
    merge_video_feature_info_for_aggregate,
    validate_all_metadata,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import DEFAULT_CHUNK_SIZE, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB  # noqa: E402

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
                src.setdefault(os.path.realpath(b["shard"]), {})[int(b["episode_index"])] = {
                    "instance": iid, "demo": r.get("out"), "job_dir": os.path.dirname(rp), "source": "collection"}
    for mp in glob.glob(os.path.join(sweep_root, task, "replay", "lerobot", "*.episodes.json")):
        shard = os.path.realpath(mp[: -len(".episodes.json")])
        for k, e in enumerate(json.load(open(mp))):
            src.setdefault(shard, {})[k] = {**e, "source": "replay"}
    return src


def fix_episode_meta_locations(root: str) -> int:
    """Point every episode row's ``meta/episodes/{chunk,file}_index`` at the metadata file it sits in. A shard finalized after
    each episode and resumed has one metadata file per episode, and aggregate_datasets copies the source indices into the
    single destination file, so a later reader (another aggregation) looks for files that do not exist."""
    import pandas as pd

    fixed = 0
    for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        chunk = int(os.path.basename(os.path.dirname(p)).split("-")[1])
        file = int(os.path.basename(p).split("-")[1].split(".")[0])
        e = pd.read_parquet(p)
        if "meta/episodes/file_index" in e.columns and ((e["meta/episodes/file_index"] != file) | (e["meta/episodes/chunk_index"] != chunk)).any():
            fixed += int(((e["meta/episodes/file_index"] != file) | (e["meta/episodes/chunk_index"] != chunk)).sum())
            e["meta/episodes/chunk_index"] = chunk
            e["meta/episodes/file_index"] = file
            e.to_parquet(p)
    return fixed


def readable_dataset(root: str) -> None:
    """lerobot opens the dataset and decodes the first and last frame."""
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset(repo_id="check", root=root)
    for i in (0, len(ds) - 1):
        _ = ds[i]


def aggregate(inputs, repo_id: str, out_root: str, data_mb, video_mb) -> int:
    tmp = out_root + ".building"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    expected = sum(dataset_episodes(r) for r in inputs)
    aggregate_datasets(repo_ids=[os.path.relpath(r, os.path.dirname(r)) for r in inputs], roots=[Path(r) for r in inputs],
                       aggr_repo_id=repo_id, aggr_root=Path(tmp), data_files_size_in_mb=data_mb, video_files_size_in_mb=video_mb)
    fix_episode_meta_locations(tmp)
    verify_aggregate(Path(tmp), expected, require_packed=expected > 1)
    readable_dataset(tmp)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.replace(tmp, out_root)
    return expected


def verify_one_chunk_per_task(root: str, expected: int, tasks) -> None:
    """verify_aggregate's checks over every chunk (it reads the first episode-metadata file only) plus the layout: the
    episodes of task k, their rows and their video files all in chunk k, and no data file holding another task's rows."""
    import pandas as pd

    info = json.load(open(os.path.join(root, "meta", "info.json")))
    eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))],
                    ignore_index=True)
    if info["total_episodes"] != expected or sorted(eps["episode_index"]) != list(range(expected)):
        raise RuntimeError(f"expected episodes 0..{expected - 1}: info.json has {info['total_episodes']}, the metadata {len(eps)}")
    if int(eps["length"].sum()) != info["total_frames"]:
        raise RuntimeError(f"episode lengths sum to {int(eps['length'].sum())}, info.json has {info['total_frames']} frames")
    listed = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
    if list(listed.index) != list(tasks) or list(listed["task_index"]) != list(range(len(tasks))):
        raise RuntimeError(f"meta/tasks.parquet does not list the tasks in order: {list(listed.index)}")
    chunk = eps["tasks"].map(lambda t: list(tasks).index(t[0]))
    video_keys = [c[len("videos/"):-len("/chunk_index")] for c in eps.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    for col in ["data/chunk_index", "meta/episodes/chunk_index"] + [f"videos/{v}/chunk_index" for v in video_keys]:
        if (eps[col] != chunk).any():
            raise RuntimeError(f"{int((eps[col] != chunk).sum())} episodes have {col} outside their task's chunk")
    missing = []
    for (c, f), g in eps.groupby(["data/chunk_index", "data/file_index"]):
        p = os.path.join(root, info["data_path"].format(chunk_index=c, file_index=f))
        if not os.path.exists(p):
            missing.append(p)
            continue
        rows = pd.read_parquet(p, columns=["episode_index", "task_index"])
        if set(rows["task_index"]) != {c} or len(rows) != int(g["length"].sum()) or set(rows["episode_index"]) != set(g["episode_index"]):
            raise RuntimeError(f"{p}: its rows are not exactly the frames of the task-{c} episodes that point to it")
    for v in video_keys:
        for c, f in eps[[f"videos/{v}/chunk_index", f"videos/{v}/file_index"]].drop_duplicates().itertuples(index=False):
            p = os.path.join(root, info["video_path"].format(video_key=v, chunk_index=c, file_index=f))
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        raise RuntimeError(f"{len(missing)} referenced files missing, e.g. {missing[:3]}")


def aggregate_one_chunk_per_task(inputs, repo_id: str, out_root: str, data_mb, video_mb) -> int:
    """lerobot's aggregate_datasets with each input (a per-task dataset) starting a new chunk: input k becomes chunk k of the
    data, the videos and the episode metadata, with task index k; within a chunk the files are packed by size as usual."""
    import pandas as pd

    tmp = out_root + ".building"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    metas = [LeRobotDatasetMetadata(os.path.relpath(r, os.path.dirname(r)), root=Path(r)) for r in inputs]
    for r, m in zip(inputs, metas):
        if len(m.tasks) != 1:
            raise ValueError(f"{r}: a per-task dataset has one task, found {len(m.tasks)}")
    tasks = [m.tasks.index[0] for m in metas]
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"a task comes from two inputs: {tasks}")
    fps, robot_type, _ = validate_all_metadata(metas)
    features = merge_video_feature_info_for_aggregate(metas)
    video_keys = [k for k in features if features[k]["dtype"] == "video"]
    data_mb, video_mb = data_mb or DEFAULT_DATA_FILE_SIZE_IN_MB, video_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
    dst = LeRobotDatasetMetadata.create(repo_id=repo_id, fps=fps, robot_type=robot_type, features=features, root=Path(tmp),
                                        use_videos=bool(video_keys), chunks_size=DEFAULT_CHUNK_SIZE,
                                        data_files_size_in_mb=data_mb, video_files_size_in_mb=video_mb)
    dst.tasks = pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task"))
    dst.episodes = {}
    videos_idx = {k: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for k in video_keys}
    for k, src in enumerate(metas):
        for v in videos_idx.values():
            v.update(chunk=k, file=0)
        videos_idx = aggregate_videos(src, dst, videos_idx, video_mb, DEFAULT_CHUNK_SIZE)
        data_idx = aggregate_data(src, dst, {"chunk": k, "file": 0}, data_mb, DEFAULT_CHUNK_SIZE)
        aggregate_metadata(src, dst, {"chunk": k, "file": 0}, data_idx, videos_idx)
        dst.info.total_episodes += src.total_episodes
        dst.info.total_frames += src.total_frames
    finalize_aggregation(dst, metas)
    fix_episode_meta_locations(tmp)
    expected = sum(m.total_episodes for m in metas)
    verify_one_chunk_per_task(tmp, expected, tasks)
    readable_dataset(tmp)
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
                f.write(json.dumps({"episode_index": k, "shard": r, "shard_episode_index": e, **src.get(os.path.realpath(r), {}).get(e, {})}) + "\n")
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
    n = aggregate_one_chunk_per_task(roots, "isg_multitask", out_root, args.data_files_size_in_mb, args.video_files_size_in_mb)
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
