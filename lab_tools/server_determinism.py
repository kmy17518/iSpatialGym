"""Item 1c: is the ACT policy server deterministic? The same observation sequence - a recorded episode's own frames (the three
cameras decoded from the dataset videos) and its raw 61-D ``observation.state`` - is sent through two fresh websocket sessions
(and optionally through the same session twice after a reset), with the episode's goal image and task id, exactly as the
evaluator's bridge sends it; the returned actions are compared step by step.

    python lab_tools/server_determinism.py --dataset datasets/isg-init --episode 105 --steps 64 --server ws://127.0.0.1:8002 --out lab/diagnostics/closedloop/server_determinism.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools.act_policy import CAMERAS, OBS_KEYS, PROPRIO_KEY, ActPolicy, packb, unpackb  # noqa: E402
from lab_tools.eval_episodes import load_goal  # noqa: E402
from lab_tools.match_instances import dataset_episodes  # noqa: E402
from lab_tools.replay_episodes import load_episode_arrays  # noqa: E402

VIDEO_KEYS = {cam: f"observation.rgb.{cam}_camera_0" for cam in CAMERAS}


def episode_frames(root: str, ep: dict, cam: str, n: int):
    """The first ``n`` frames of one camera stream of the episode."""
    import av
    import pandas as pd

    key = VIDEO_KEYS[cam]
    # the chunk / file / timestamp of this camera come from the episode metadata parquet
    meta = None
    import glob

    for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet"))):
        df = pd.read_parquet(p)
        hit = df[df["episode_index"] == ep["episode_index"]]
        if len(hit):
            meta = hit.iloc[0]
            break
    path = os.path.join(root, "videos", key, f"chunk-{int(meta[f'videos/{key}/chunk_index']):03d}", f"file-{int(meta[f'videos/{key}/file_index']):03d}.mp4")
    target = float(meta[f"videos/{key}/from_timestamp"])
    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        container.seek(int(max(0.0, target - 0.5) / float(tb)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            ts = float(frame.pts * tb) if frame.pts is not None else None
            if ts is None or ts < target - 1e-3:
                continue
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= n:
                break
    return out


def run_session(server: str, goal_msg: dict, task_id: int, frames: dict, states: np.ndarray, n: int, reset_between: bool = False):
    from websockets.sync.client import connect

    ws = connect(server, max_size=64 * 1024 * 1024, compression=None, open_timeout=60, close_timeout=5)
    meta = unpackb(ws.recv(timeout=60))
    actions = []
    t0 = time.time()
    for t in range(n):
        msg = {PROPRIO_KEY: states[t][None].astype(np.float32), "task_id": np.int64(task_id), **goal_msg}
        for cam in CAMERAS:
            msg[OBS_KEYS[cam]] = np.ascontiguousarray(frames[cam][t])[None, ..., :3]
        ws.send(packb(msg))
        reply = unpackb(ws.recv(timeout=120))
        a = np.asarray(reply["action"], dtype=np.float32)
        actions.append(a[0] if a.ndim == 2 else a)
    ws.close()
    return np.stack(actions), meta, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--server", default="ws://127.0.0.1:8002")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    eps = dataset_episodes(args.dataset)
    ep = dict(eps[args.episode], episode_index=args.episode)
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    g = goals[args.episode]
    goal = load_goal(os.path.join(args.dataset, g["png"]))
    actions_demo, states, length = load_episode_arrays(args.dataset, args.episode)
    n = min(args.steps, length)
    frames = {cam: episode_frames(args.dataset, ep, cam, n) for cam in CAMERAS}
    for cam in CAMERAS:
        assert len(frames[cam]) == n, (cam, len(frames[cam]), n)
    # the goal message as the bridge builds it (zed only for a single-view checkpoint; wrist goal PNGs when the server wants them)
    policy = ActPolicy(ctx={})
    from websockets.sync.client import connect

    ws = connect(args.server, max_size=64 * 1024 * 1024, compression=None, open_timeout=60, close_timeout=5)
    meta = unpackb(ws.recv(timeout=60))
    ws.close()
    views = list(meta.get("goal_views") or ["zed_link"])
    keys = meta.get("goal_observation_keys") or [f"goal::robot_r1::robot_r1:{v}:Camera:0::rgb" for v in views]
    rgbs = {"zed_link": goal}
    goals_root = os.path.join(WT, "datasets", "goals", os.path.basename(os.path.normpath(args.dataset)))
    if os.path.exists(os.path.join(goals_root, "index.json")):
        for w in json.load(open(os.path.join(goals_root, "index.json")))["episodes"]:
            if int(w["episode_index"]) == args.episode:
                rgbs[w["camera"]] = load_goal(os.path.join(goals_root, w["png"]))
    goal_msg = {k: np.ascontiguousarray(rgbs[v])[None, ..., :3] for k, v in zip(keys, views)}
    runs = []
    for s in range(args.sessions):
        acts, meta_s, dt = run_session(args.server, goal_msg, int(g["train_task_index"]), frames, states, n)
        runs.append(acts)
        print(f"session {s}: {n} steps in {dt:.1f}s; first action {acts[0][:4]}", flush=True)
    out = {"format": "isg-lab-server-determinism/v1", "server": args.server, "metadata": {k: v for k, v in meta.items() if k in ("policy", "action_horizon", "chunk_size", "execution_mode", "checkpoint_step", "goal_views", "temporal_agg")},
           "episode": args.episode, "steps": n, "sessions": args.sessions, "pairs": []}
    for i in range(1, len(runs)):
        d = np.abs(runs[i] - runs[0])
        per_step = d.max(axis=1)
        first = int(np.argmax(per_step > 0)) if (per_step > 0).any() else None
        out["pairs"].append({"sessions": [0, i], "bit_identical": bool(per_step.max() == 0.0), "first_differing_step": first, "max_abs_diff": float(per_step.max()),
                             "n_steps_differing": int((per_step > 0).sum()), "max_abs_diff_per_dim": [float(v) for v in d.max(axis=0)]})
    # the demo's own actions vs the policy's (how far the policy is from the demo on its own observations; information only)
    out["policy_vs_demo_actions_mean_abs"] = float(np.abs(runs[0] - actions_demo[:n]).mean())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    np.save(args.out.replace(".json", "_actions.npy"), np.stack(runs))
    for p in out["pairs"]:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
