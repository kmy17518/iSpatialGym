#!/usr/bin/env python3
"""Recompute the base-trajectory block of replay records from the arrays the driver saved next to the video
(``replay_base_xyyaw.npy``, ``replay_base_qvel.npy``, ``demo_base_qvel.npy``), with :func:`replay_episodes.trajectory_block`.
Used once for records written before the calibrated dead reckoning existed; no simulator needed.

    python lab_tools/replay_recompute.py lab/diagnostics/replay/eval50/episode_*.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
from lab_tools.replay_episodes import end_vs_goal, goal_base_pose, headline_deviation, load_episode_arrays, trajectory_block  # noqa: E402


def main(paths) -> int:
    for p in paths:
        r = json.load(open(p))
        raw_dir = os.path.join(WT, r["raw_dir"]) if r.get("raw_dir") else None
        if not raw_dir or not os.path.exists(os.path.join(raw_dir, "replay_base_xyyaw.npy")):
            print(f"{p}: no arrays; skipped")
            continue
        rep = np.load(os.path.join(raw_dir, "replay_base_xyyaw.npy")).astype(np.float64)
        qv = os.path.join(raw_dir, "replay_base_qvel.npy")
        replay_qvel = np.load(qv).astype(np.float64) if os.path.exists(qv) else np.zeros((0, 3))
        dq = os.path.join(raw_dir, "demo_base_qvel.npy")
        root = os.path.join(WT, "datasets", "isg-init-eval" if r.get("replay_split", "eval50") == "eval50" else "isg-init")
        actions, states, _ = load_episode_arrays(root, int(r["episode_index"]))
        demo_qvel = np.load(dq).astype(np.float64) if os.path.exists(dq) else states[:, 0:3]
        x0, y0, yaw0 = rep[0]
        traj = trajectory_block(rep, replay_qvel, demo_qvel, x0, y0, yaw0, actions=actions, streamed=int(r.get("actions_streamed") or 0))
        inst_map = json.load(open(os.path.join(WT, "lab", "instances.json" if r.get("replay_split", "eval50") == "eval50" else "instances-train.json")))
        inst_map = inst_map.get("episodes", inst_map)
        tro = (inst_map.get(str(r["episode_index"])) or {}).get("tro")
        traj["end_vs_goal_base"] = end_vs_goal(rep, goal_base_pose(os.path.join(WT, tro))) if tro else None
        traj.pop("demo_dead_reckoned_raw", None)
        cal = traj.pop("demo_dead_reckoned_calibrated", None)
        if cal is not None:
            np.save(os.path.join(raw_dir, "demo_base_xyyaw_dead_reckoned_calibrated.npy"), np.asarray(cal, dtype=np.float32))
        r.setdefault("score", {}).setdefault("replay", {})["trajectory"] = traj
        for k in ("init_xyyaw", "demo_dead_reckoned_end_xyyaw", "replay_end_xyyaw"):
            r["score"]["replay"].pop(k, None)
        r.update(headline_deviation(traj))
        r["trajectory_recomputed_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
        with open(p + ".tmp", "w") as f:
            json.dump(r, f, indent=1, default=str)
        os.replace(p + ".tmp", p)
        h = headline_deviation(traj)
        print(f"{os.path.basename(p)}: max dev {h['base_trajectory_max_deviation_m']} m ({h['base_trajectory_deviation_method']}, floor {h['base_trajectory_method_floor_m']}, raw {h['base_trajectory_max_deviation_raw_m']}), "
              f"yaw-rate scale {traj.get('yaw_rate_scale_measured_over_true')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
