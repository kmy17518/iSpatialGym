#!/usr/bin/env python3
"""Episode -> instance map of the test set ``kmy17518/isg-init-eval`` with first-frame verification.

For every task: the first head-camera frame (``observation.rgb.zed_link_camera_0``, 720 x 720) and the first
``observation.state`` of each episode are read from the dataset; the regenerated instances' first frames come from
``lab_tools/first_frames.py``. Relocalization episodes are mapped by the dataset's own CSV
(``isg_meta/camera_relocalization_distances.csv``) and only verified; the other tasks are assigned one-to-one by the
Hungarian assignment on the mean absolute pixel difference (0-255) and verified with the same metric (machine 1's
criterion: <= 20). Writes ``lab/instances.json`` as ``{episode_index: {task, instance, method, first_frame_mean_abs_diff, ...}}``
plus a per-task report under ``lab/instances_report.json``.

    python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json
"""

from __future__ import annotations

import argparse
import csv
import fractions
import glob
import json
import os
import sys
import time

from typing import Optional

import numpy as np

TASKS = [
    "camera_relocalization-standard",
    "configuration_matching-articulation_open_large_scale-dishwasher",
    "configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table",
    "alignment-axial-board_game-breakfast_table",
]
HEAD_KEY = "observation.rgb.zed_link_camera_0"


def load_png(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def dataset_episodes(root: str, chunks=None):
    """{episode_index: {task, task_index, chunk, file, from_ts, length, state0}} from meta/episodes + data parquet.
    ``chunks``: restrict to these meta/episodes chunk indices (one chunk per task; the training set has 24)."""
    import pandas as pd

    tasks = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
    task_of_index = {int(v): str(k) for k, v in tasks["task_index"].items()}
    eps = {}
    paths = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if chunks is not None:
        paths = [p for p in paths if int(os.path.basename(os.path.dirname(p)).split("-")[1]) in set(chunks)]
    for p in paths:
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            e = int(r["episode_index"])
            eps[e] = {
                "task": str(r["tasks"][0]) if hasattr(r["tasks"], "__len__") else str(r["tasks"]),
                "length": int(r["length"]),
                "data_chunk": int(r["data/chunk_index"]), "data_file": int(r["data/file_index"]),
                "from_index": int(r["dataset_from_index"]),
                "video_chunk": int(r[f"videos/{HEAD_KEY}/chunk_index"]), "video_file": int(r[f"videos/{HEAD_KEY}/file_index"]),
                "from_ts": float(r[f"videos/{HEAD_KEY}/from_timestamp"]),
            }
    # first observation.state of every episode (row at dataset_from_index of its data file)
    by_file = {}
    for e, d in eps.items():
        by_file.setdefault((d["data_chunk"], d["data_file"]), []).append(e)
    for (c, f), es in by_file.items():
        df = pd.read_parquet(os.path.join(root, "data", f"chunk-{c:03d}", f"file-{f:03d}.parquet"), columns=["index", "episode_index", "frame_index", "observation.state", "task_index"])
        first = df[df["frame_index"] == 0].set_index("episode_index")
        for e in es:
            row = first.loc[e]
            eps[e]["state0"] = np.asarray(row["observation.state"], dtype=np.float32)
            eps[e]["task_index"] = int(row["task_index"])
            assert task_of_index[eps[e]["task_index"]] == eps[e]["task"], (e, eps[e]["task"], task_of_index[eps[e]["task_index"]])
    return eps


def first_frame(root: str, ep: dict) -> np.ndarray:
    """Decode the episode's first head frame from the chunk video (timestamp from_ts) with PyAV."""
    import av

    path = os.path.join(root, "videos", HEAD_KEY, f"chunk-{ep['video_chunk']:03d}", f"file-{ep['video_file']:03d}.mp4")
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        target = ep["from_ts"]
        container.seek(int(max(0.0, target - 0.5) / float(tb)), stream=stream, backward=True, any_frame=False)
        best = None
        for frame in container.decode(stream):
            ts = float(frame.pts * tb) if frame.pts is not None else None
            if ts is None:
                continue
            if ts >= target - 1e-3:
                best = frame
                break
        if best is None:
            raise RuntimeError(f"no frame at {target}s in {path}")
        return best.to_ndarray(format="rgb24")


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


DEPTH_KEY = "observation.depth_linear.zed_link_camera_0"


def first_depth(root: str, ep: dict) -> Optional[np.ndarray]:
    """The episode's first head depth frame in metres (lerobot's 12-bit log quantization), (H, W) float32; None if absent."""
    import av

    from lerobot.datasets.depth_utils import dequantize_depth

    path = os.path.join(root, "videos", DEPTH_KEY, f"chunk-{ep['video_chunk']:03d}", f"file-{ep['video_file']:03d}.mp4")
    if not os.path.exists(path):
        return None
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        target = ep["from_ts"]
        container.seek(int(max(0.0, target - 0.5) / float(tb)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts * tb) >= target - 1e-3:
                return np.asarray(dequantize_depth(frame, output_unit="m"), dtype=np.float32).reshape(frame.height, frame.width)
    return None


def quantized_like_dataset(depth_m: np.ndarray) -> np.ndarray:
    """A rendered depth map through the dataset's quantization (so both sides carry the same quantization error)."""
    from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth

    q = quantize_depth(np.asarray(depth_m, dtype=np.float32), input_unit="m")
    return np.asarray(dequantize_depth(q, output_unit="m"), dtype=np.float32).reshape(depth_m.shape[0], depth_m.shape[1])


def depth_mismatch(a: np.ndarray, b: np.ndarray, tol_m: float = 0.05) -> float:
    """Fraction of pixels whose depths differ by more than ``tol_m`` (both clipped to the dataset range)."""
    a = np.clip(a, 0.01, 10.0)
    b = np.clip(b, 0.01, 10.0)
    return float(np.mean(np.abs(a - b) > tol_m))


def hungarian(cost: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:  # noqa: BLE001  greedy fallback
        pairs, used = [], set()
        for i in np.argsort(cost.min(axis=1)):
            order = np.argsort(cost[i])
            for j in order:
                if j not in used:
                    used.add(j)
                    pairs.append((int(i), int(j)))
                    break
        return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", required=True, help="root of lab_tools/first_frames.py outputs: <frames>/<task>/<iid>.png")
    ap.add_argument("--out", required=True, help="lab/instances.json")
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--merge", action="store_true", help="keep entries of tasks not processed now from the existing --out")
    ap.add_argument("--strict-tasks", nargs="*", default=["configuration_matching-articulation_open_large_scale-dishwasher"],
                    help="tasks whose goal differs per instance even when the first frames coincide (a target angle): an assignment counts as "
                         "verified only when the depth match is exact (<= 0.0005) and no other episode matches the same instance exactly")
    ap.add_argument("--dataset-repo", default=None, help="recorded in the output (default: kmy17518/<dataset dir name>)")
    ap.add_argument("--method-note", default="", help="appended to every row's method string (e.g. '+reloc_recover(seed 7)')")
    args = ap.parse_args(argv)

    import pandas as pd

    tasks_df = pd.read_parquet(os.path.join(args.dataset, "meta", "tasks.parquet"))
    chunks = sorted(int(tasks_df.loc[t, "task_index"]) for t in args.tasks if t in tasks_df.index)  # one chunk per task in both datasets
    eps = dataset_episodes(args.dataset, chunks=chunks or None)
    goals = json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]
    train_idx = {g["task"]: g["train_task_index"] for g in goals}
    csv_map = {}
    with open(os.path.join(args.dataset, "isg_meta", "camera_relocalization_distances.csv")) as f:
        for r in csv.DictReader(f):
            csv_map[int(r["episode_index"])] = r
    existing = {}
    if args.merge and os.path.exists(args.out):
        existing = json.load(open(args.out))
        existing = existing.get("episodes", existing)
    result = dict(existing)
    report = {}
    for task in args.tasks:
        t0 = time.time()
        task_eps = sorted(e for e, d in eps.items() if d["task"] == task)
        frames_dir = os.path.join(args.frames, task)
        index = json.load(open(os.path.join(frames_dir, "index.json"))) if os.path.exists(os.path.join(frames_dir, "index.json")) else {}
        iids = sorted(i for i, rec in index.items() if rec.get("ok") and os.path.exists(os.path.join(frames_dir, i + ".png")))
        if not task_eps or not iids:
            report[task] = {"episodes": len(task_eps), "instances": len(iids), "skipped": True}
            print(f"{task}: episodes={len(task_eps)} instances={len(iids)} -> skipped", flush=True)
            continue
        inst_img = {i: load_png(os.path.join(frames_dir, i + ".png")) for i in iids}
        inst_prop = {}
        for i in iids:
            p = os.path.join(frames_dir, i + ".proprio.npy")
            if os.path.exists(p):
                inst_prop[i] = np.load(p)
        ep_img = {e: first_frame(args.dataset, eps[e]) for e in task_eps}
        cost = np.zeros((len(task_eps), len(iids)), dtype=np.float64)  # plain RGB mean abs diff (the reported criterion)
        for a, e in enumerate(task_eps):
            for b, i in enumerate(iids):
                cost[a, b] = mean_abs_diff(ep_img[e], inst_img[i])
        # assignment cost: depth mismatch (geometry only, robust to rendering / codec differences) when every render has a
        # depth map, else the RGB diff restricted to the pixels that vary across the renders (the task objects)
        candidates_cost = {}
        stack = np.stack([inst_img[i].astype(np.float32) for i in iids])
        mask = stack.std(0).mean(-1) > 8.0 if len(iids) > 1 else np.ones(stack.shape[1:3], dtype=bool)
        if mask.any():
            m = np.zeros_like(cost)
            for a, e in enumerate(task_eps):
                for b, i in enumerate(iids):
                    m[a, b] = float(np.abs(ep_img[e].astype(np.float32) - inst_img[i].astype(np.float32)).mean(-1)[mask].mean())
            candidates_cost[f"rgb_masked_diff({int(mask.sum())}px)"] = m
        inst_depth = {}
        for i in iids:
            p = os.path.join(frames_dir, i + ".depth.npy")
            if os.path.exists(p):
                inst_depth[i] = quantized_like_dataset(np.load(p).astype(np.float32))
        if len(inst_depth) == len(iids):
            ep_depth = {e: first_depth(args.dataset, eps[e]) for e in task_eps}
            if all(d is not None for d in ep_depth.values()):
                m = np.zeros_like(cost)
                for a, e in enumerate(task_eps):
                    for b, i in enumerate(iids):
                        m[a, b] = depth_mismatch(ep_depth[e], inst_depth[i])
                candidates_cost["depth_mismatch_frac(>5cm)"] = m

        ids_all = sorted(set(k.split("@", 1)[0] for k in iids))
        cols_all = {i: [b for b, k in enumerate(iids) if k.split("@", 1)[0] == i] for i in ids_all}

        def separability(m):
            # on the per-instance matrix (the candidates of one index collapse to their best): the margin to the best *other* instance
            mc = np.array([[m[a, cols_all[i]].min() for i in ids_all] for a in range(m.shape[0])])
            srt = np.sort(mc, axis=1)
            if srt.shape[1] < 2:
                return 0.0
            return float(np.min((srt[:, 1] - srt[:, 0]) / (srt[:, 0] + 1e-6)))

        if candidates_cost:
            assign_metric = max(candidates_cost, key=lambda k: separability(candidates_cost[k]))
            assign_cost = candidates_cost[assign_metric]
        else:
            assign_metric, assign_cost = "rgb_mean_abs_diff", cost
        sep_report = {k: round(separability(v), 3) for k, v in candidates_cost.items()}
        base_of = {k: k.split("@", 1)[0] for k in iids}  # candidate keys <iid>@a<k> -> instance id
        pairs = {}
        if task == TASKS[0]:
            method = "dataset_csv+first_frame_verify"
            for a, e in enumerate(task_eps):
                inst = csv_map[e]["instance"]
                keys = [k for k in iids if base_of[k] == inst]
                if keys:
                    pairs[e] = min(keys, key=lambda k: assign_cost[a, iids.index(k)])
        else:
            method = "regenerated_sampler_seed1003+first_frame_match"
            # one column per instance id (the candidates of one index collapse to their best-matching attempt)
            ids = sorted(set(base_of.values()))
            cols = {i: [b for b, k in enumerate(iids) if base_of[k] == i] for i in ids}
            collapsed = np.array([[assign_cost[a, cols[i]].min() for i in ids] for a in range(len(task_eps))])
            for a, j in hungarian(collapsed):
                i = ids[j]
                b = min(cols[i], key=lambda bb: assign_cost[a, bb])
                pairs[task_eps[a]] = iids[b]
        n_ok = 0
        per_ep = {}
        for a, e in enumerate(task_eps):
            key = pairs.get(e)
            inst = base_of[key] if key is not None else None
            row = {"task": task, "instance": inst, "method": method + ("+lenient_candidate" if key and "@" in key else "") + args.method_note,
                   "first_frame_mean_abs_diff": None, "verified": False, "train_task_index": train_idx.get(task), "task_index": eps[e]["task_index"]}
            if key is not None:
                b = iids.index(key)
                d = float(cost[a, b])
                others = [cost[a, j] for j in range(len(iids)) if base_of[iids[j]] != inst]
                others_assign = [assign_cost[a, j] for j in range(len(iids)) if base_of[iids[j]] != inst]
                row.update({
                    "first_frame_mean_abs_diff": round(d, 3),
                    "verified": bool(d <= args.threshold),
                    "second_best_diff": round(float(min(others)), 3) if others else None,
                    "best_instance_by_frame": base_of[iids[int(np.argmin(cost[a]))]],
                    "assignment_metric": assign_metric,
                    "assignment_cost": round(float(assign_cost[a, b]), 4),
                    "assignment_second_best": round(float(min(others_assign)), 4) if others_assign else None,
                    "metrics": {k2: round(float(v[a, b]), 4) for k2, v in candidates_cost.items()},
                    "instance_index": int(inst.rsplit("__", 1)[1]) if "__" in inst else None,
                    "tro": index[key]["tro"],
                    "candidate": key if "@" in key else None,
                })
                if key in inst_prop and eps[e].get("state0") is not None and inst_prop[key].shape == eps[e]["state0"].shape:
                    row["first_state_max_abs_diff"] = round(float(np.max(np.abs(inst_prop[key] - eps[e]["state0"]))), 4)
                n_ok += int(row["verified"])
            per_ep[e] = row
            result[str(e)] = row
        # tasks with a per-instance goal that the first frame does not show (dishwasher: the target angle): only an exact, unique
        # depth match pins the instance; the others keep their assignment but are not counted as verified
        depth_cost = candidates_cost.get("depth_mismatch_frac(>5cm)")
        if task in (args.strict_tasks or []) and depth_cost is not None:
            exact_of = {}
            for a2, e2 in enumerate(task_eps):
                for j in range(len(iids)):
                    if depth_cost[a2, j] <= 0.0005:
                        exact_of.setdefault(base_of[iids[j]], set()).add(e2)
            for e2 in task_eps:
                row = per_ep[e2]
                if row.get("instance") is None:
                    continue
                a2 = task_eps.index(e2)
                key2 = pairs.get(e2)
                dc = float(depth_cost[a2, iids.index(key2)]) if key2 is not None else None
                row["depth_mismatch_frac"] = None if dc is None else round(dc, 4)
                exact = dc is not None and dc <= 0.0005
                unique = len(exact_of.get(row["instance"], set()) - {e2}) == 0
                row["assignment_confident"] = bool(exact and unique)
                if not row["assignment_confident"]:
                    why = "no exact first-frame match" if not exact else "another episode matches this instance exactly"
                    row["ambiguous"] = why + "; the target angle cannot be confirmed from the first frame"
                    if row["verified"]:
                        row["verified"] = False
                        row["verified_by_first_frame_only"] = True
                        n_ok -= 1
        # instances the assignment did not use (surplus / missing) and duplicate checks
        used = {base_of[k] for k in pairs.values()}
        report[task] = {
            "episodes": len(task_eps), "instances": len(set(base_of.values())), "candidates": sum(1 for k in iids if "@" in k), "verified": n_ok, "method": method,
            "unused_instances": sorted(set(base_of.values()) - used), "unmapped_episodes": [e for e in task_eps if e not in pairs],
            "diffs": sorted(round(float(per_ep[e]["first_frame_mean_abs_diff"]), 2) for e in per_ep if per_ep[e]["first_frame_mean_abs_diff"] is not None),
            "assignment_metric": assign_metric, "metric_separability": sep_report, "match_min_margin": None, "assignment_min_margin": None, "assignment_margins": None,
            "wall_s": round(time.time() - t0, 1),
        }
        margins = [per_ep[e]["second_best_diff"] - per_ep[e]["first_frame_mean_abs_diff"] for e in per_ep
                   if per_ep[e].get("second_best_diff") is not None and per_ep[e]["first_frame_mean_abs_diff"] is not None]
        if margins:
            report[task]["match_min_margin"] = round(float(min(margins)), 3)
        amargins = [per_ep[e]["assignment_second_best"] - per_ep[e]["assignment_cost"] for e in per_ep
                    if per_ep[e].get("assignment_second_best") is not None and per_ep[e].get("assignment_cost") is not None]
        if amargins:
            report[task]["assignment_min_margin"] = round(float(min(amargins)), 4)
            report[task]["assignment_margins"] = sorted(round(float(m), 4) for m in amargins)
        print(f"{task}: episodes={len(task_eps)} instances={len(set(base_of.values()))} verified={n_ok} diffs(min/median/max)="
              f"{min(report[task]['diffs']):.2f}/{np.median(report[task]['diffs']):.2f}/{max(report[task]['diffs']):.2f} "
              f"rgb_min_margin={report[task]['match_min_margin']} assignment={assign_metric} sep={sep_report} assign_min_margin={report[task]['assignment_min_margin']} "
              f"unused={len(report[task]['unused_instances'])} unmapped={len(report[task]['unmapped_episodes'])}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = {str(k): v for k, v in sorted(result.items(), key=lambda kv: int(kv[0]))}
    doc = {"format": "isg-lab-instances/v1", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dataset": args.dataset_repo or ("kmy17518/" + os.path.basename(os.path.normpath(args.dataset))), "threshold_mean_abs_diff": args.threshold,
           "verified": sum(1 for v in out.values() if v.get("verified")), "total": len(out), "episodes": out}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    stem = os.path.abspath(args.out)[: -len(".json")] if args.out.endswith(".json") else os.path.abspath(args.out)
    with open(stem + "_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"instances.json: verified {doc['verified']}/{doc['total']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
