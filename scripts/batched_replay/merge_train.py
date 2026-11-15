#!/usr/bin/env python3
"""Batched replay -> one LeRobot v3 dataset (the training format; also usable for eval episodes with --purpose eval).

    merge_train.py --runs <run>[,<run>...] --out <dataset dir> [--purpose train] [--name <dataset name>] [--load-test] [--copy]

Reads the units pulled by sync.py ($RV/runs/<run>/out/<unit>/: the unit's LeRobot v3 dataset <unit>/, goal_images/,
episodes_isg.jsonl, instances/ = replayed TROs, done.json). No re-encoding: every unit's data file and video files are linked
(hard links; --copy copies) under contiguous file indices, episode_index / index / task_index are remapped, per-episode stats kept
(remapped columns recomputed) and aggregated into meta/stats.json. Tasks = the eval class ids (e.g. disembedding_02).

Beside the LeRobot files, isg_meta/ (read by ACT / Diffusion Policy / openpi loaders that know it, ignored by plain LeRobot):
  episodes.jsonl        one line per episode: episode_index, task, unit/run/source episode, length + settle (action_steps =
                        program_length, settle_steps, recorded lengths), perturbation record (scene removal, lighting draw, goal
                        group), goal image file, and the instance linkage: demo dir + sha256, instance_id, variant, subtask,
                        template dir / template.json / template_meta.json, TRO path + sha256, oracle sidecar, eval ids
  goal_images/episode_<i>.png   the episode's goal image: its last rendered head-camera frame (224 x 224), the same image as the
                        episode's last frame of observation.rgb.zed_link_camera_0 before video compression
  tro/episode_<i>-tro_state.json   the instance's TRO (the replayed TRO with campaign.replay_perturbations when the replay perturbed
                        it, else the source TRO), so an episode can be reloaded as an instance
  templates/<sha12>/template.json, template_meta.json   the episode's template (episodes.jsonl names its <sha12>)
  settle_windows.json   isg-settle-windows/v1: program_length (action steps) and length of every episode (--settle-steps of ACT)
  task_groups.json      isg-task-groups/v1: task family -> class ids
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))
FAMILIES = ("alignment", "camera_relocalization", "configuration_matching", "disembedding", "inter_object_relation_satisfaction",
            "mental_rotation", "object_scaling")
REMAPPED = ("episode_index", "index", "task_index")
CHUNK = 1000


def family_of(task: str) -> str:
    return next((f for f in FAMILIES if task.startswith(f)), "other")


def sha256_file(p, bs=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def link(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def units_of(runs, purpose):
    out = []
    for run in runs:
        spec = json.load(open(RV / "runs" / run / "units.json"))
        for u in spec["units"]:
            d = RV / "runs" / run / "out" / u["unit"]
            if u["purpose"] != purpose:
                continue
            if not (d / ".synced").exists():
                print(f"skip {run}/{u['unit']}: not pulled (sync.py pull)")
                continue
            out.append((run, u, d))
    out.sort(key=lambda x: (FAMILIES.index(family_of(x[1]["task"])) if family_of(x[1]["task"]) in FAMILIES else 99, x[1]["task"], x[0],
                            x[1]["unit"]))
    return out


def merge(a):
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats

    t0 = time.time()
    runs = a.runs.split(",")
    units = units_of(runs, a.purpose)
    if not units:
        sys.exit("no pulled units")
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"{out} exists")
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tasks = sorted({u["task"] for _, u, _ in units}, key=lambda t: (FAMILIES.index(family_of(t)) if family_of(t) in FAMILIES else 99, t))
    task_index = {t: i for i, t in enumerate(tasks)}
    info0 = None
    video_keys = None
    ep_rows, isg_eps, settle = [], [], {}
    n_data = 0
    n_vid = collections.Counter()
    gidx, gframe = 0, 0
    templates = {}
    for run, u, d in units:
        uid = u["unit"]
        ds = d / uid
        info = json.load(open(ds / "meta/info.json"))
        if info0 is None:
            info0 = info
            video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        elif info["features"] != info0["features"] or info.get("cam_intrinsics") != info0.get("cam_intrinsics") or info["fps"] != info0["fps"]:
            raise ValueError(f"{run}/{uid}: features / intrinsics / fps differ from the first unit's")
        rows = sorted((r for p in sorted((ds / "meta/episodes").glob("*/*.parquet")) for r in pq.read_table(p).to_pylist()),
                      key=lambda r: r["episode_index"])
        recs = {e["episode_index"]: e for e in (json.loads(x) for x in (d / "episodes_isg.jsonl").read_text().splitlines() if x.strip())
                if e.get("kept")}
        if sorted(recs) != [r["episode_index"] for r in rows]:
            raise ValueError(f"{run}/{uid}: episodes {[r['episode_index'] for r in rows]} vs kept records {sorted(recs)}")
        # data: one file per source data file, rows remapped
        dmap, vmap = {}, {}
        tables = {}
        for r in rows:
            key = (r["data/chunk_index"], r["data/file_index"])
            if key not in tables:
                tables[key] = pq.read_table(ds / info["data_path"].format(chunk_index=key[0], file_index=key[1]))
            for k in video_keys:
                vk = (k, r[f"videos/{k}/chunk_index"], r[f"videos/{k}/file_index"])
                if vk not in vmap:
                    n = n_vid[k]
                    n_vid[k] += 1
                    link(ds / info["video_path"].format(video_key=k, chunk_index=vk[1], file_index=vk[2]),
                         tmp / info["video_path"].format(video_key=k, chunk_index=n // CHUNK, file_index=n % CHUNK), a.copy)
                    vmap[vk] = (n // CHUNK, n % CHUNK)
        new_tables = collections.defaultdict(list)
        for r in rows:
            key = (r["data/chunk_index"], r["data/file_index"])
            t = tables[key]
            import pyarrow.compute as pc

            sel = t.filter(pc.equal(t["episode_index"], r["episode_index"])).sort_by("frame_index")
            L = int(r["length"])
            if sel.num_rows != L or sel["frame_index"].to_pylist() != list(range(L)):
                raise ValueError(f"{run}/{uid} episode {r['episode_index']}: {sel.num_rows} rows, length {L}")
            rec = recs[r["episode_index"]]
            new = {"episode_index": np.full(L, gidx, np.int64), "index": np.arange(gframe, gframe + L, dtype=np.int64),
                   "task_index": np.full(L, task_index[u["task"]], np.int64)}
            cols = [pa.array(new[n]) if n in REMAPPED else sel[n] for n in sel.column_names]
            new_tables[key].append(pa.Table.from_arrays(cols, names=sel.column_names))
            stats = {f: {col.split("/")[-1]: v for col, v in r.items() if col.startswith(f"stats/{f}/")} for f in info["features"]
                     if f not in REMAPPED}
            stats = {f: s for f, s in stats.items() if s}
            stats.update({f: {k: np.asarray(v).tolist() for k, v in get_feature_stats(new[f], axis=0, keepdims=True).items()}
                          for f in REMAPPED})
            row = {"episode_index": gidx, "tasks": [u["task"]], "length": L, "_dkey": key,
                   "dataset_from_index": gframe, "dataset_to_index": gframe + L, "_stats": stats}
            for k in video_keys:
                ci, fi = vmap[(k, r[f"videos/{k}/chunk_index"], r[f"videos/{k}/file_index"])]
                row.update({f"videos/{k}/chunk_index": ci, f"videos/{k}/file_index": fi,
                            f"videos/{k}/from_timestamp": r[f"videos/{k}/from_timestamp"], f"videos/{k}/to_timestamp": r[f"videos/{k}/to_timestamp"]})
            ep_rows.append(row)
            # isg_meta
            inst = rec.get("instance") or {}
            gname = f"episode_{gidx:06d}"
            goal_rel = None
            if rec.get("goal_image"):
                link(d / rec["goal_image"]["rgb_path"], tmp / "isg_meta/goal_images" / f"{gname}.png", True)
                goal_rel = f"isg_meta/goal_images/{gname}.png"
            tro_src = None
            if rec.get("replayed_tro"):
                cand = d / "instances" / os.path.basename(rec["replayed_tro"])
                tro_src = cand if cand.exists() else None
            tro_src = tro_src or (Path(inst["tro_path"]) if inst.get("tro_path") and os.path.exists(inst["tro_path"]) else None)
            tro_rel = None
            if tro_src is not None:
                link(tro_src, tmp / "isg_meta/tro" / f"{gname}-tro_state.json", True)
                tro_rel = f"isg_meta/tro/{gname}-tro_state.json"
            tkey = None
            tj = inst.get("template_json")
            if tj and os.path.exists(tj):
                sha = sha256_file(tj)
                tkey = sha[:12]
                if tkey not in templates:
                    templates[tkey] = {"template_json": tj, "sha256": sha, "template_meta": inst.get("template_meta"),
                                       "template_dir": inst.get("template_dir")}
                    link(Path(tj), tmp / "isg_meta/templates" / tkey / "template.json", True)
                    tm = inst.get("template_meta")
                    if tm and os.path.exists(tm):
                        link(Path(tm), tmp / "isg_meta/templates" / tkey / "template_meta.json", True)
            st = rec.get("settle") or {}
            settle[str(gidx)] = {"length": L, "program_length": int(st.get("action_steps") or L)}
            isg_eps.append({"episode_index": gidx, "task": u["task"], "task_family": family_of(u["task"]), "purpose": u["purpose"],
                            "run": run, "unit": uid, "unit_episode_index": r["episode_index"], "length": L, "settle": st,
                            "image_size": rec.get("image_size"), "scene_mode": rec.get("scene_mode"),
                            "perturbation": rec.get("perturbation"), "goal_image": goal_rel,
                            "goal_image_record": rec.get("goal_image"), "tro": tro_rel, "template": tkey,
                            "replayed_tro_source": rec.get("replayed_tro"), "instance": inst})
            gidx += 1
            gframe += L
        for key, parts in new_tables.items():
            n = n_data
            n_data += 1
            p = tmp / info0["data_path"].format(chunk_index=n // CHUNK, file_index=n % CHUNK)
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(parts), p, compression="snappy")
            for row in ep_rows:
                if row.get("_dkey") == key and "data/chunk_index" not in row:
                    row["data/chunk_index"], row["data/file_index"] = n // CHUNK, n % CHUNK
        for row in ep_rows:
            row.pop("_dkey", None)
        print(f"[{time.time() - t0:6.1f}s] {run}/{uid}: {len(rows)} episodes", flush=True)
    # meta/episodes
    cols = {}
    for k in [k for k in ep_rows[0] if k != "_stats"]:
        cols[k] = pa.array([r[k] for r in ep_rows], pa.list_(pa.string())) if k == "tasks" else pa.array([r[k] for r in ep_rows])
    for f in ep_rows[0]["_stats"]:
        for st in ep_rows[0]["_stats"][f]:
            vals = [np.asarray(r["_stats"][f][st]) for r in ep_rows]
            ints = vals[0].dtype.kind in "iub"
            cols[f"stats/{f}/{st}"] = pa.array([v.reshape(-1).astype(np.int64 if ints else np.float64).tolist() for v in vals],
                                               pa.list_(pa.int64() if ints else pa.float64()))
    cols["meta/episodes/chunk_index"] = pa.array([0] * len(ep_rows))
    cols["meta/episodes/file_index"] = pa.array([0] * len(ep_rows))
    p = tmp / "meta/episodes/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), p)
    # tasks.parquet the way lerobot writes it (pandas index "task")
    import pandas as pd

    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(tmp / "meta/tasks.parquet")
    n_ep, n_frames = len(ep_rows), sum(r["length"] for r in ep_rows)
    info = {k: v for k, v in info0.items()}
    info.update({"total_episodes": n_ep, "total_frames": n_frames, "total_tasks": len(tasks), "chunks_size": CHUNK,
                 "splits": {"train": f"0:{n_ep}"}})
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    agg = aggregate_stats([{f: {k: np.asarray(v) for k, v in s.items()} for f, s in r["_stats"].items()} for r in ep_rows])
    ref = json.load(open(units[0][2] / units[0][1]["unit"] / "meta/stats.json"))
    stats = {}
    for f, st in agg.items():
        stats[f] = {}
        for k, v in st.items():
            arr = np.asarray(v)
            r0 = (ref.get(f) or {}).get(k)
            if r0 is not None and np.asarray(r0).size == arr.size:
                arr = arr.reshape(np.asarray(r0).shape)
            stats[f][k] = arr.tolist()
    (tmp / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    # isg_meta
    im = tmp / "isg_meta"
    im.mkdir(exist_ok=True)
    with open(im / "episodes.jsonl", "w") as f:
        for e in isg_eps:
            f.write(json.dumps(e) + "\n")
    (im / "settle_windows.json").write_text(json.dumps({
        "format": "isg-settle-windows/v1",
        "description": "Every episode is its oracle program (program_length = the demo's action steps) followed by the batched replay's "
                       "settle window: max(ceil(0.2 x program_length), 50) frames of the settled final state, replacing the 45 recorded "
                       "settle steps (rows past the recorded end repeat the last recorded row). ACT, Diffusion Policy and openpi read this "
                       "file for --settle-steps.",
        "dataset": {"name": a.name, "total_episodes": n_ep, "total_frames": n_frames}, "episodes": settle}, indent=1))
    groups = collections.OrderedDict()
    for t in tasks:
        groups.setdefault(family_of(t), []).append(t)
    (im / "task_groups.json").write_text(json.dumps({"format": "isg-task-groups/v1",
                                                     "description": "task family -> the eval class ids (meta/tasks.parquet) of this dataset",
                                                     "groups": groups}, indent=2))
    (im / "templates.json").write_text(json.dumps(templates, indent=1))
    (tmp / "README.md").write_text(readme(a.name, n_ep, n_frames, tasks, groups, video_keys, info0, runs, a.purpose))
    tmp.rename(out)
    print(f"[{time.time() - t0:6.1f}s] DONE {out}: {n_ep} episodes, {n_frames} frames, {len(tasks)} tasks, {len(templates)} templates")
    return out


def readme(name, n_ep, n_frames, tasks, groups, video_keys, info0, runs, purpose):
    fam = "\n".join(f"| {f} | {len(ts)} | {', '.join(ts[:6])}{' ...' if len(ts) > 6 else ''} |" for f, ts in groups.items())
    size = info0["features"][video_keys[0]]["shape"][:2]
    return f"""# {name}

iSpatialGym oracle demos (R1Pro), {"training instances" if purpose == "train" else "eval instances"}: {n_ep} episodes, {n_frames}
frames at {info0['fps']} fps, LeRobot v3.0. Replayed kinematically from each demo's demo.hdf5 by the batched replay of 2026-11-15
(`python -m ispatialgym.multienv.replay --slots 1 --batched-usd-edits --async-flush --proprio loaded --warmup-s 15 --scene full
--image-size 224 --settle-pad 0.2 --settle-pad-min 50 --goal-images on`, ISPATIALGYM_SHARD_FORMAT=legacy, Behavior-1K 6a6194600 +
p59fc36c6 patches, Isaac Sim 5.1.0) on Vast.ai GPUs; runs {', '.join(runs)}.

- Cameras: {', '.join(f'`{k}`' for k in video_keys)}, all {size[0]} x {size[1]} (BEHAVIOR's encoders: RGB HEVC x265 CRF 30, depth
  12-bit HEVC log depth), with `observation.robot2cam_pose.*`. `observation.state`: the replayed robot's 61-D proprioception;
  `action`: 23-D. Frame i pairs action i with the observation of recorded state i.
- Scene: the demo's recorded (cropped) world merged back into its template's task rooms + adjacent rooms (`--scene full`).
- {"Training perturbations, per goal group with p = 0.5 each: replay-time lighting (fixture intensity x U[0.5, 1.5] + colour tint; the skybox per replay process; the collection-time lighting is ignored) and replay-time scene perturbation (one object outside the recorded crop removed with what rests on it; never for camera relocalization, whose removals were made at collection). The draws are in isg_meta/episodes.jsonl `perturbation` and in the episode's TRO (`campaign.replay_perturbations`)." if purpose == "train" else "No replay-time perturbation (eval instances)."}
- Settle: every episode is its demo's action steps A followed by max(ceil(0.2 A), 50) settle frames (isg_meta/settle_windows.json).
- Goal image: isg_meta/goal_images/episode_<i>.png, the episode's last rendered head-camera frame (= the last frame of
  `observation.rgb.zed_link_camera_0` before compression; ACT's goal_source=episode_last reads the same frame from the video).
- Instance linkage: isg_meta/episodes.jsonl (demo, TRO, template, eval ids, sha256), isg_meta/tro/, isg_meta/templates/.
- Tasks ({len(tasks)}, `meta/tasks.parquet`), grouped by task family in isg_meta/task_groups.json:

| task family | tasks | |
|---|---|---|
{fam}
"""


def load_test(root, n=4):
    """LeRobot's own loader on the merged dataset (videos decoded), then a few random items."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=Path(root).name, root=root)
    rng = np.random.default_rng(0)
    for i in rng.integers(len(ds), size=n):
        it = ds[int(i)]
        shapes = {k: tuple(v.shape) for k, v in it.items() if hasattr(v, "shape")}
        print(f"item {int(i)}: episode {int(it['episode_index'])} task {it.get('task')} {shapes}")
    print(f"LeRobotDataset OK: {ds.num_episodes} episodes, {ds.num_frames} frames")
    eps = [json.loads(x) for x in open(Path(root) / "isg_meta/episodes.jsonl")]
    missing = [e["episode_index"] for e in eps if not (e["goal_image"] and (Path(root) / e["goal_image"]).is_file())]
    print(f"goal images: {len(eps) - len(missing)}/{len(eps)} present; TROs {sum(1 for e in eps if e['tro'])}/{len(eps)}; "
          f"templates {len({e['template'] for e in eps if e['template']})}")
    if missing:
        raise SystemExit(f"episodes without a goal image: {missing[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--purpose", default="train", choices=("train", "eval"))
    ap.add_argument("--name", default="isg-train")
    ap.add_argument("--copy", action="store_true", help="copy unit files instead of hard-linking them")
    ap.add_argument("--load-test", action="store_true")
    a = ap.parse_args()
    out = merge(a)
    if a.load_test:
        load_test(out)


if __name__ == "__main__":
    main()
