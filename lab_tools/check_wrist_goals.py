#!/usr/bin/env python3
"""Bit-for-bit check of the lab's goal PNGs against the ACT trainer's own goal images (user, 2026-10-07 18:30Z).

For every episode and goal view: ``quantize_image(preprocess_image(PNG, image_size))`` must equal ``B1KDataset.goal_for(position)``
of the ACT repo's dataset built with ``goal_views`` = the three cameras and ``goal_source='episode_last'`` - the exact goal tensor
the "wrist" checkpoints trained on. The zed PNGs are the dataset's own (``isg_meta/goals``), the wrist PNGs the lab's
(``lab_tools/wrist_goals.py``). Runs in the ACT venv (``act-isg/.venv``):

    act-isg/.venv/bin/python lab_tools/check_wrist_goals.py --dataset datasets/isg-init --goals datasets/goals/isg-init --episodes 98 99 ... [--image-size 240 240]

Writes ``<goals>/bit_exact_check.json`` and prints one line per mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT = os.path.join(WT, "act-isg")
TASK = "camera_relocalization-standard"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--goals", required=True, help="lab_tools/wrist_goals.py output dir (index.json)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--image-size", type=int, nargs=2, default=None, help="training image size (default: the checkpoint adapter's, else 240 240)")
    ap.add_argument("--checkpoint", default=None, help="a .pt whose adapter_config.image_size and goal_views are used")
    args = ap.parse_args(argv)

    sys.path.insert(0, ACT)
    import numpy as np
    import torch
    from PIL import Image

    from b1k_dataset import B1KDataset, CAMERAS, preprocess_image, quantize_image

    image_size = tuple(args.image_size) if args.image_size else (240, 240)
    goal_views = list(CAMERAS)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        image_size = tuple(ck["adapter_config"]["image_size"])
        goal_views = list(ck["model_config"].get("goal_views") or goal_views)
    wrist_index = json.load(open(os.path.join(args.goals, "index.json")))
    wrist = {}
    for g in wrist_index["episodes"]:
        wrist.setdefault(int(g["episode_index"]), {})[g["camera"]] = os.path.join(args.goals, g["png"])
    zed = {int(g["episode_index"]): os.path.join(args.dataset, g["png"]) for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    episodes = sorted(args.episodes) if args.episodes else sorted(e for e in wrist if e in zed)
    t0 = time.time()
    ds = B1KDataset(args.dataset, task_names=[TASK], image_size=image_size, goal_views=goal_views, goal_source="episode_last", episodes=episodes)
    results = []
    n_ok = 0
    for e in episodes:
        pos = int(ds.positions[e])
        table = ds.goal_for(pos).numpy()  # (views, H, W, 3) uint8
        for vi, view in enumerate(goal_views):
            png = zed.get(e) if view == "zed_link" else wrist.get(e, {}).get(view)
            rec = {"episode_index": e, "view": view, "png": None if png is None else os.path.relpath(png, WT), "equal": False}
            if png is None or not os.path.exists(png):
                rec["error"] = "no PNG"
            else:
                img = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
                mine = quantize_image(preprocess_image(img, image_size))
                theirs = table[vi]
                rec["equal"] = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
                if not rec["equal"]:
                    d = np.abs(mine.astype(np.int16) - theirs.astype(np.int16)) if mine.shape == theirs.shape else None
                    rec["max_abs_diff"] = None if d is None else int(d.max())
                    rec["mean_abs_diff"] = None if d is None else float(d.mean())
                    rec["n_pixels_differ"] = None if d is None else int((d.max(-1) > 0).sum())
                    print(f"episode {e} {view}: NOT equal (max {rec['max_abs_diff']}, mean {rec['mean_abs_diff']}, {rec['n_pixels_differ']} px)", flush=True)
            n_ok += int(rec["equal"])
            results.append(rec)
    out = {"format": "isg-goal-bit-exact-check/v1", "dataset": os.path.basename(os.path.normpath(args.dataset)), "image_size": list(image_size), "goal_views": goal_views,
           "method": "quantize_image(preprocess_image(PNG)) == B1KDataset(goal_views, goal_source=episode_last).goal_for(position)[view]",
           "n_checked": len(results), "n_equal": n_ok, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    with open(os.path.join(args.goals, "bit_exact_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"BIT_EXACT dataset={args.dataset} episodes={len(episodes)} views={goal_views} image_size={image_size} equal={n_ok}/{len(results)} wall={time.time() - t0:.1f}s", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
