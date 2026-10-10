#!/usr/bin/env python3
"""Goal images of the wrist cameras (and a zed self-check) for goal-conditioned checkpoints with ``goal_views`` beyond ``zed_link``.

The ACT trainers' goal source ``episode_last`` is the episode's last recorded frame of each goal camera's own video stream
(``b1k_dataset.ISGDataset._goal_frame``: the recorded episode's last row timestamp plus the stream's ``from_timestamp``,
decoded with PyAV - seek backward, nearest pts, stop at the first frame at or after the target - at native size; the server
resizes it like an observation). The datasets ship that image for ``zed_link`` only (``isg_meta/goals/``); this tool decodes
it the same way for ``left_realsense_link`` / ``right_realsense_link`` (480 x 480) and writes

    <out>/<camera>/episode_XXXXXX.png  +  <out>/index.json   {episodes: [{episode_index, camera, png, frame, timestamp, rgb_sha256, ...}]}

The zed frame is decoded too and its rgb sha256 compared with the dataset's ``isg_meta/goals/index.json`` entry (``zed_check``),
which validates the decode path against the trainer's.

    python lab_tools/wrist_goals.py --dataset datasets/isg-init --out datasets/goals/isg-init --episodes 98 99 ...
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
VIDEO_KEY = {c: f"observation.rgb.{c}_camera_0" for c in CAMERAS}
TOL = 1e-4


def decode(path: str, timestamp: float):
    """The trainer's decode (b1k_dataset._decode): nearest frame to ``timestamp``, scanning from the preceding keyframe."""
    import av

    av.logging.set_level(None)
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(max(0, int(timestamp / float(stream.time_base))), stream=stream, backward=True)
        best, distance = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            d = abs(t - timestamp)
            if d < distance:
                best, distance = frame, d
            if t >= timestamp:
                break
        if best is None or distance > TOL:
            raise ValueError(f"{path}: no frame within {TOL}s of {timestamp:.6f} (nearest {distance:.6f})")
        return best.to_ndarray(format="rgb24"), float(best.pts * stream.time_base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: every episode of the dataset's isg_meta/goals/index.json")
    ap.add_argument("--cameras", nargs="*", default=["left_realsense_link", "right_realsense_link"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    from PIL import Image

    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    episodes = sorted(args.episodes) if args.episodes else sorted(goals)
    meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(args.dataset, "meta", "episodes", "chunk-*", "file-*.parquet")))])
    meta = meta.set_index("episode_index")
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"format": "isg-goal-images-wrist/v1", "dataset": os.path.basename(os.path.normpath(args.dataset)),
                                                                             "source": "episode_last (b1k_dataset._goal_frame / _decode), native size", "episodes": []}
    have = {(e["episode_index"], e["camera"]) for e in index["episodes"]}
    t0 = time.time()
    n_new = 0
    checks = []
    for e in episodes:
        if e not in meta.index:
            print(f"episode {e}: not in meta/episodes", flush=True)
            continue
        row = meta.loc[e]
        length = int(row["length"])
        data = pd.read_parquet(os.path.join(args.dataset, "data", f"chunk-{int(row['data/chunk_index']):03d}", f"file-{int(row['data/file_index']):03d}.parquet"),
                               columns=["episode_index", "frame_index", "timestamp"])
        last = data[(data["episode_index"] == e) & (data["frame_index"] == length - 1)]
        if len(last) != 1:
            print(f"episode {e}: last row not found", flush=True)
            continue
        ts = float(last["timestamp"].iloc[0])
        for cam in ["zed_link", *args.cameras]:
            key = VIDEO_KEY[cam]
            offset = float(row[f"videos/{key}/from_timestamp"])
            target = offset + ts
            path = os.path.join(args.dataset, "videos", key, f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}", f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
            if not os.path.exists(path):
                print(f"episode {e} {cam}: video missing ({os.path.relpath(path, args.dataset)})", flush=True)
                continue
            if cam == "zed_link":
                # self-check of the decode path against the dataset's own goal PNG: same frame (mean abs diff well below 1/255 x
                # a few: the PNGs were decoded on another machine whose yuv -> rgb conversion differs by ~0.6/255 on average)
                rgb, t_dec = decode(path, target)
                g = goals.get(e, {})
                mad = None
                if g.get("png") and os.path.exists(os.path.join(args.dataset, g["png"])):
                    png = np.asarray(Image.open(os.path.join(args.dataset, g["png"])).convert("RGB"), dtype=np.float32)
                    mad = float(np.abs(rgb.astype(np.float32) - png).mean()) if png.shape == rgb.shape else None
                ok = mad is not None and mad < 2.0
                checks.append({"episode_index": e, "mean_abs_diff_vs_dataset_png": None if mad is None else round(mad, 4), "ok": ok,
                               "sha_equal": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() == g.get("rgb_sha256")})
                if not ok:
                    print(f"episode {e}: zed self-check FAILED (mean abs diff {mad} vs the dataset PNG, frame {length - 1}, t {target:.4f})", flush=True)
                continue
            if (e, cam) in have and not args.overwrite:
                continue
            rgb, t_dec = decode(path, target)
            png_rel = os.path.join(cam, f"episode_{e:06d}.png")
            os.makedirs(os.path.join(args.out, cam), exist_ok=True)
            Image.fromarray(rgb).save(os.path.join(args.out, png_rel))
            index["episodes"] = [x for x in index["episodes"] if not (x["episode_index"] == e and x["camera"] == cam)]
            index["episodes"].append({"episode_index": e, "task": goals.get(e, {}).get("task"), "camera": cam, "png": png_rel, "frame": length - 1, "timestamp": target,
                                      "decoded_pts_s": t_dec, "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
                                      "rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                                      "png_sha256": hashlib.sha256(open(os.path.join(args.out, png_rel), "rb").read()).hexdigest()})
            n_new += 1
    index["episodes"].sort(key=lambda x: (x["episode_index"], x["camera"]))
    prev = {c["episode_index"]: c for c in (index.get("zed_self_check") or {}).get("per_episode", [])}
    for c in checks:
        prev[c["episode_index"]] = c
    mads = [c["mean_abs_diff_vs_dataset_png"] for c in prev.values() if c.get("mean_abs_diff_vs_dataset_png") is not None]
    index["zed_self_check"] = {"n": len(prev), "ok": int(sum(1 for c in prev.values() if c["ok"])), "mean_abs_diff_max": max(mads) if mads else None,
                               "note": "the zed frame decoded here vs the dataset's isg_meta/goals PNG (same frame; sha differs: another machine's yuv->rgb conversion)",
                               "per_episode": sorted(prev.values(), key=lambda c: c["episode_index"])}
    index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(index_path, "w") as f:
        json.dump(index, f, indent=1)
    n_ok = sum(1 for c in checks if c["ok"])
    print(f"WRIST_GOALS dataset={args.dataset} episodes={len(episodes)} new_pngs={n_new} zed_self_check={n_ok}/{len(checks)} (max mad {max(mads) if mads else None}) wall={time.time() - t0:.1f}s -> {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
