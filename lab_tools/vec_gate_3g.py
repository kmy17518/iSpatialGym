#!/usr/bin/env python
"""Gate 3g: throughput of the vectorized evaluator vs single-env processes, from the per-episode records.

Each record carries wall_s (the episode's share of its batch) and batch_wall_s (the whole batch, settle + stream + epilogue for all slots);
episodes/hour of a set = num_envs * 3600 / median(batch_wall_s). For concurrent single-env processes (one set per process) the throughputs add.

    python lab_tools/vec_gate_3g.py --out lab/diagnostics/vec-eval/3g/summary.json \
        --set n2x1=lab/diagnostics/vec-eval/3g/n2x1/camera_relocalization-standard \
        --concurrent n1x2=lab/diagnostics/vec-eval/3g/n1x2-a/camera_relocalization-standard,lab/diagnostics/vec-eval/3g/n1x2-b/camera_relocalization-standard \
        --set 3e-default-n4=lab/diagnostics/vec-eval/3e/default-n4/camera_relocalization-standard ...
"""
import argparse
import glob
import json
import os
import statistics as S


SINCE = None


def load(d, since=None):
    """Records of one directory; `dir@ISO` or --since drop records finished before the timestamp (leftovers of an aborted run)."""
    if "@" in d:
        d, since = d.split("@", 1)
    since = since or SINCE
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "episode_*.json")))]
    return [r for r in recs if not r.get("error") and (since is None or (r.get("finished_at") or "") >= since)]


def stats(recs):
    if not recs:
        return {"n": 0}
    bw = [r["batch_wall_s"] for r in recs if r.get("batch_wall_s")]
    w = [r["wall_s"] for r in recs if r.get("wall_s")]
    st = [r["steps"] for r in recs if r.get("steps")]
    n_env = int(recs[0].get("num_envs") or 1)
    mem = [(r.get("memory") or {}).get("gpu_mib") or r.get("memory_gpu_mib") for r in recs]
    mem = [m for m in mem if m]
    bs = [r["batch_steps"] for r in recs if r.get("batch_steps")]
    out = {"n": len(recs), "num_envs": n_env, "steps_median": S.median(st) if st else None, "steps_mean": round(S.mean(st), 1) if st else None,
           "batch_wall_s_median": round(S.median(bw), 1) if bw else None, "wall_per_episode_s_median": round(S.median(w), 1) if w else None,
           "memory_gpu_mib_max": max(mem) if mem else None}
    if bw and st:
        out["episodes_per_hour"] = round(n_env * 3600.0 / S.median(bw), 2)
        # env-steps per second: every batched step advances all N environments (batch_steps = batched steps of the batch incl. settle/epilogue)
        if bs:
            out["env_steps_per_second"] = round(n_env * S.mean(bs) / S.mean(bw), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="name=dir[@since] (one process)")
    ap.add_argument("--concurrent", action="append", default=[], help="name=dir1,dir2 (processes that ran at the same time; throughputs add)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default=None, help="ignore records finished before this UTC timestamp (e.g. leftovers of an aborted run)")
    args = ap.parse_args()
    global SINCE
    SINCE = args.since
    summary = {"format": "isg-lab-vec-gate-3g/v1", "since": args.since, "sets": {}}
    for item in args.set:
        name, d = item.split("=", 1)
        summary["sets"][name] = dict(stats(load(d)), dir=d)
    for item in args.concurrent:
        name, ds = item.split("=", 1)
        parts = [stats(load(d)) for d in ds.split(",")]
        eph = sum(p.get("episodes_per_hour") or 0 for p in parts)
        sps = sum(p.get("env_steps_per_second") or 0 for p in parts)
        summary["sets"][name] = {"concurrent": parts, "episodes_per_hour": round(eph, 2), "env_steps_per_second": round(sps, 2),
                                 "memory_gpu_mib_max_sum": sum(p.get("memory_gpu_mib_max") or 0 for p in parts), "dirs": ds.split(",")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    for k, v in summary["sets"].items():
        print(f"{k}: episodes/hour {v.get('episodes_per_hour')}  env-steps/s {v.get('env_steps_per_second')}  n={v.get('n', sum(p.get('n', 0) for p in v.get('concurrent', [])))}"
              f"  batch wall median {v.get('batch_wall_s_median')}  gpu MiB {v.get('memory_gpu_mib_max') or v.get('memory_gpu_mib_max_sum')}")


if __name__ == "__main__":
    main()
