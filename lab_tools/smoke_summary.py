#!/usr/bin/env python3
"""lab/smoke/summary.json from the smoke episodes: per checkpoint and task the outcomes, timing (minutes per episode, simulator
steps/s, world load), the goal-swap plumbing check (action-sequence and outcome difference between an episode run with its own
goal PNG and with another instance's), and the two-evaluator memory test (peak GPU memory from lab_tools/smoke_gpu_mem.log)."""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    out = {"format": "isg-lab-smoke/v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}, "goal_swap": None,
           "timing": {}, "two_evaluators": {}}
    walls, rates, loads = [], [], []
    for p in sorted(glob.glob(os.path.join(WT, "lab", "smoke", "*", "*", "episode_*.json"))):
        r = json.load(open(p))
        run, task = p.split(os.sep)[-3], p.split(os.sep)[-2]
        swap = ".goalswap_" in os.path.basename(p)
        ent = out["runs"].setdefault(run, {}).setdefault(task, {"episodes": []})
        t = r.get("timings") or {}
        ent["episodes"].append({"episode_index": r["episode_index"], "instance": r["instance"], "goal_swap": swap, "goal_from_episode": r.get("goal_from_episode"),
                                "success": r["success"], "reason": r.get("reason"), "steps": r["steps"], "wall_s": r["wall_s"],
                                "score": {k: v for k, v in (r.get("score") or {}).items() if isinstance(v, (int, float))},
                                "server_request_ms": None if r.get("server_request_s_mean") is None else round(1000 * r["server_request_s_mean"], 1),
                                "error": r.get("error")})
        if not swap and not r.get("error"):
            walls.append(r["wall_s"])
            if t.get("executor_actions_s") and r.get("steps"):
                rates.append(r["steps"] / t["executor_actions_s"])
            if t.get("load_s", 0) > 30:
                loads.append(t["load_s"] + t.get("planner_build_s", 0))
    if walls:
        out["timing"] = {"episodes": len(walls), "minutes_per_episode_mean": round(float(np.mean(walls)) / 60, 2), "minutes_per_episode_median": round(float(np.median(walls)) / 60, 2),
                         "sim_steps_per_s_mean": round(float(np.mean(rates)), 1) if rates else None, "world_load_s_mean": round(float(np.mean(loads)), 0) if loads else None,
                         "episodes_per_hour_per_evaluator_at_horizon": round(3600 / float(np.mean(walls)), 1),
                         "note": "wall_s includes the per-episode restore; the first episode of a task process also includes the world load (world_load_s_mean) which the mean excludes"}
    # goal swap: the normal and the swapped run of the same episode
    for p in glob.glob(os.path.join(WT, "lab", "smoke", "*", "*", "episode_*.goalswap_*.json")):
        sw = json.load(open(p))
        base = os.path.join(os.path.dirname(p), f"episode_{sw['episode_index']:06d}.json")
        if not os.path.exists(base):
            continue
        nr = json.load(open(base))
        rec = {"run": p.split(os.sep)[-3], "task": p.split(os.sep)[-2], "episode_index": sw["episode_index"], "goal_from_episode": sw.get("goal_from_episode"),
               "own_goal": {"success": nr["success"], "steps": nr["steps"], "score": {k: v for k, v in nr["score"].items() if isinstance(v, (int, float))}},
               "swapped_goal": {"success": sw["success"], "steps": sw["steps"], "score": {k: v for k, v in sw["score"].items() if isinstance(v, (int, float))}}}
        try:
            a = np.load(os.path.join(WT, nr["actions_npy"])); b = np.load(os.path.join(WT, sw["actions_npy"]))
            n = min(len(a), len(b))
            d = np.abs(a[:n] - b[:n])
            rec["actions"] = {"compared_steps": int(n), "mean_abs_diff": float(d.mean()), "mean_abs_diff_first_16_steps": float(d[:16].mean()),
                              "mean_abs_diff_after_16": float(d[16:].mean()) if n > 16 else None, "max_abs_diff": float(d.max()),
                              "identical_first_action": bool(np.allclose(a[0], b[0]))}
            rec["behavior_changes"] = bool(d[16:].mean() > 1e-3) if n > 16 else None
        except Exception as e:  # noqa: BLE001
            rec["actions_error"] = repr(e)
        out["goal_swap"] = rec
    mem = os.path.join(WT, "lab_tools", "smoke_gpu_mem.log")
    if os.path.exists(mem):
        vals = [int(line.split()[1]) for line in open(mem) if len(line.split()) == 2 and line.split()[1].isdigit()]
        if vals:
            out["two_evaluators"] = {"gpu_memory_peak_mib": max(vals), "gpu_memory_total_mib": 24564, "samples": len(vals),
                                     "fits": max(vals) < 23500, "note": "two eval_episodes.py processes (dishwasher + blender, one episode each) plus the ACT server; sampled every 10 s"}
    os.makedirs(os.path.join(WT, "lab", "smoke"), exist_ok=True)
    with open(os.path.join(WT, "lab", "smoke", "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("timing", "goal_swap", "two_evaluators")}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
