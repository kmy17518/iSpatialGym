#!/usr/bin/env python3
"""Throughput of policy-eval runs from their done records (``$EV/runs/<run>/done/*.json``): per checkpoint tag (one host layout each in
the speed sweeps) the episode steps/s per process (instances' steps / episode seconds), the world load per instance (first in a world
vs reused), the policy request latency, and the per-GPU throughput = processes per GPU x steps/s.

    speed_table.py <run dir> [<run dir> ...]
"""
import collections
import glob
import json
import os
import statistics
import sys


def main():
    rows = collections.defaultdict(list)
    for run in sys.argv[1:]:
        for f in glob.glob(os.path.join(run, "done", "*.json")):
            d = json.load(open(f))
            for x in d.get("instances") or []:
                m, r = x.get("meta") or {}, x.get("result") or {}
                t = m.get("timings") or {}
                pc = m.get("policy_client") or {}
                if not t.get("episode_s"):
                    continue
                rows[(os.path.basename(run.rstrip("/")), d["ckpt"])].append({
                    "gpu": d.get("gpu_name"), "cpus": d.get("cpus"), "P": d.get("procs_per_gpu"), "steps": r.get("n_steps") or 0,
                    "episode_s": t["episode_s"], "load_s": t.get("load_s") or 0, "reused": bool(x.get("reused_world")),
                    "req_ms": 1000 * (pc.get("request_s_mean") or 0), "qa": (m.get("qa") or {}).get("enabled", True)})
    print(f"{'run/ckpt':32s} {'gpu':14s} cpus P   n  steps/s/proc  per GPU  load fresh/reused s  request ms")
    for (run, ck), rs in sorted(rows.items()):
        sps = sum(x["steps"] for x in rs) / sum(x["episode_s"] for x in rs)
        fresh = [x["load_s"] for x in rs if not x["reused"]]
        reused = [x["load_s"] for x in rs if x["reused"]]
        P = rs[0]["P"] or 1
        print(f"{run + '/' + ck:32s} {str(rs[0]['gpu'])[-12:]:14s} {rs[0]['cpus']:>4} {P} {len(rs):3d}  {sps:10.1f}  {P * sps:7.1f}  "
              f"{statistics.fmean(fresh) if fresh else 0:7.0f} / {statistics.fmean(reused) if reused else 0:5.0f}  "
              f"{statistics.fmean(x['req_ms'] for x in rs):8.0f}")


if __name__ == "__main__":
    main()
