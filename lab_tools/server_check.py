#!/usr/bin/env python3
"""Handshake + latency check of the ACT server with a real test-set observation (frame 0 of one episode, its goal PNG).

    python lab_tools/server_check.py --dataset datasets/isg-init-eval --episode 0 [--server ws://127.0.0.1:8000] [--n 20]

Prints the server metadata, the request latency, the first action and whether another episode's goal changes the
action (goal plumbing at the server level).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab_tools import act_policy as AP  # noqa: E402
from lab_tools.match_instances import dataset_episodes, first_frame  # noqa: E402


def frame_of(root, ep, key):
    import av

    path = os.path.join(root, "videos", key, f"chunk-{ep['video_chunk']:03d}", f"file-{ep['video_file']:03d}.mp4")
    with av.open(path) as c:
        s = c.streams.video[0]
        tb = s.time_base
        c.seek(int(max(0.0, ep["from_ts"] - 0.5) / float(tb)), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            if fr.pts is not None and float(fr.pts * tb) >= ep["from_ts"] - 1e-3:
                return fr.to_ndarray(format="rgb24")
    raise RuntimeError("no frame")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--other-episode", type=int, default=None)
    ap.add_argument("--server", default="ws://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)
    from PIL import Image

    eps = dataset_episodes(args.dataset)
    ep = eps[args.episode]
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    g = goals[args.episode]
    other = args.other_episode
    if other is None:
        other = next(e for e in sorted(eps) if eps[e]["task"] == ep["task"] and e != args.episode)
    imgs = {cam: frame_of(args.dataset, ep, f"observation.rgb.{cam}_camera_0") for cam in AP.CAMERAS}
    goal = np.asarray(Image.open(os.path.join(args.dataset, g["png"])).convert("RGB"), dtype=np.uint8)
    goal2 = np.asarray(Image.open(os.path.join(args.dataset, goals[other]["png"])).convert("RGB"), dtype=np.uint8)
    state = ep["state0"].astype(np.float32)
    print("task:", ep["task"], "train_task_index:", g["train_task_index"], "images:", {k: v.shape for k, v in imgs.items()}, "goal:", goal.shape, "state:", state.shape)

    from websockets.sync.client import connect

    def session(goal_img):
        ws = connect(args.server, max_size=64 * 1024 * 1024, compression=None, open_timeout=60)
        meta = AP.unpackb(ws.recv(timeout=60))
        acts, lat = [], []
        for _ in range(args.n):
            msg = {AP.PROPRIO_KEY: state[None], "task_id": np.int64(g["train_task_index"]), AP.GOAL_KEY: goal_img[None]}
            for cam in AP.CAMERAS:
                msg[AP.OBS_KEYS[cam]] = imgs[cam][None]
            t0 = time.time()
            ws.send(AP.packb(msg))
            a = AP.unpackb(ws.recv(timeout=120))["action"]
            lat.append(time.time() - t0)
            acts.append(np.asarray(a)[0])
        ws.close()
        return meta, np.stack(acts), np.asarray(lat)

    meta, acts, lat = session(goal)
    print("metadata:", json.dumps({k: v for k, v in meta.items() if k not in ("observation_keys",)}, default=str)[:900])
    print(f"latency: first {lat[0]*1000:.0f} ms, replan steps mean {np.mean(lat[::meta['action_horizon']])*1000:.0f} ms, cached steps mean {np.mean([l for i, l in enumerate(lat) if i % meta['action_horizon']])*1000:.1f} ms")
    print("action[0]:", np.round(acts[0], 3).tolist())
    print("action std over steps:", np.round(acts.std(0), 3).tolist())
    _, acts2, _ = session(goal2)
    print(f"other goal (episode {other}): mean |delta action| = {np.abs(acts2 - acts).mean():.4f}, max = {np.abs(acts2 - acts).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
