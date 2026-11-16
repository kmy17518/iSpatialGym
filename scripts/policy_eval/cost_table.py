#!/usr/bin/env python3
"""Cost of a policy-eval run per checkpoint from its done records: instances, unit wall time (process-hours: each unit is one
simulator process), GPU-hours (process time / processes per GPU), mean steps/s per process, the share of process time spent loading
worlds, and the GPUs needed to evaluate 1000 instances of that checkpoint in a target wall time (default 1.5 h minus a boot
allowance) at the measured cost per instance.

    cost_table.py <run dir> [--target-h 1.5] [--boot-min 25]
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--target-h", type=float, default=1.5)
    ap.add_argument("--boot-min", type=float, default=25.0, help="boot + checkpoint download before the first unit starts")
    a = ap.parse_args()
    agg = collections.defaultdict(lambda: collections.Counter())
    for f in glob.glob(os.path.join(a.run_dir, "done", "*.json")):
        d = json.load(open(f))
        c = agg[d["ckpt"]]
        P = max(1, int(d.get("procs_per_gpu") or 1))
        c["units"] += 1
        c["proc_s"] += d["seconds"]
        c["gpu_s"] += d["seconds"] / P
        for x in d.get("instances") or []:
            t = (x.get("meta") or {}).get("timings") or {}
            c["instances"] += 1
            c["steps"] += (x.get("result") or {}).get("n_steps") or 0
            c["episode_s"] += t.get("episode_s") or 0
            c["load_s"] += t.get("load_s") or 0
    avail_h = a.target_h - a.boot_min / 60
    print(f"{'ckpt':24s} inst units  proc-h  GPU-h  steps/s/proc  load share  GPU-h/1000 inst  GPUs for 1000 in {a.target_h} h (boot {a.boot_min:.0f} min)")
    for ck, c in sorted(agg.items()):
        gpu_h_1000 = c["gpu_s"] / 3600 / c["instances"] * 1000
        print(f"{ck:24s} {c['instances']:4d} {c['units']:5d} {c['proc_s'] / 3600:7.1f} {c['gpu_s'] / 3600:6.1f} {c['steps'] / max(1, c['episode_s']):10.1f}"
              f"  {100 * c['load_s'] / max(1, c['proc_s']):8.0f}%  {gpu_h_1000:13.1f}  {gpu_h_1000 / avail_h:8.0f}")


if __name__ == "__main__":
    main()
