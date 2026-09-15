#!/usr/bin/env python3
"""Collect /root/bench/result.json from the benchmark pods and print the throughput / cost comparison.

    python scripts/runpod/bench_report.py pull      # rsync result.json (+ gpu csv, job logs' tails) into runlogs/runpod/bench/<name>/
    python scripts/runpod/bench_report.py report    # table from what is in runlogs/runpod/bench/

Throughput unit: sampler jobs per hour (one job = one `sample_instances` run with 2 instances, IK cache off). The
concurrent phase gives the *aggregate* jobs/h of one card running K processes; dividing the card's $/h by it gives $/job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402

OUT = os.path.join(rp.STATE_DIR, "bench")


def pull():
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    fleet = rp.load_fleet()
    for pid, info in fleet["pods"].items():
        if info.get("purpose") != "bench" or info.get("terminated_at"):
            continue
        ep = rp.pod_ssh(api, pid)
        if ep is None:
            print(f"{info['name']}: no ssh endpoint (terminated?)")
            continue
        dst = os.path.join(OUT, info["name"])
        os.makedirs(dst, exist_ok=True)
        r = rp.ssh_run(*ep, "test -f /root/bench/result.json && echo ready", capture=True, timeout=60)
        if "ready" not in (r.stdout or ""):
            r2 = rp.ssh_run(*ep, "cat /root/bench/jobs.txt 2>/dev/null; tail -2 /root/isg_bench.log 2>/dev/null", capture=True, timeout=60)
            print(f"{info['name']}: not finished. progress:\n{(r2.stdout or '').strip()}")
            continue
        for f in ("result.json", "jobs.txt", "gpu_single.csv", "gpu_conc.csv", "cpu.txt"):
            try:
                rp.rsync(f"/root/bench/{f}", dst + "/", *ep, to_pod=False)
            except subprocess.CalledProcessError:
                pass
        # sampling reports of every job (n_written per template + the sampler's own template time); the early bench.sh
        # parser wrote -1 for n_written, so recompute it here
        r = rp.ssh_run(*ep, "cd /root/bench && for d in warm_* single_* conc_*; do [ -d $d ] || continue; f=$(ls $d/sampling_report_*.json 2>/dev/null | tail -1); "
                            "[ -n \"$f\" ] && echo \"$d $(python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); ts=[t for p in d['problems'].values() for t in p.values()]; "
                            "print(sum(int(t.get('n_written',0)) for t in ts), round(sum(float(t.get('time_s',0)) for t in ts),1))\" $f)\"; done", capture=True, timeout=120)
        per_job = {}
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                per_job[parts[0]] = (int(parts[1]), float(parts[2]))
        res_path = os.path.join(dst, "result.json")
        if os.path.exists(res_path) and per_job:
            res = json.load(open(res_path))
            for j in res["jobs"]:
                key = f"{j['phase']}_{j['idx']}"
                if key in per_job:
                    j["n_written"], j["sampler_time_s"] = per_job[key]
            json.dump(res, open(res_path, "w"), indent=2)
        with open(os.path.join(dst, "pod.json"), "w") as f:
            json.dump({"pod": pid, **info}, f, indent=2)
        print(f"{info['name']}: pulled ({len(per_job)} job reports)")


def report():
    rows = []
    for name in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        p = os.path.join(OUT, name, "result.json")
        if not os.path.exists(p):
            continue
        res = json.load(open(p))
        pod = json.load(open(os.path.join(OUT, name, "pod.json")))
        price = float(pod.get("price_per_h") or 0)
        single = {j["problem"].split("-")[0][:8]: j["wall_s"] for j in res["jobs"] if j["phase"] == "single"}
        conc = [j for j in res["jobs"] if j["phase"] == "conc"]
        ok = all(j["n_written"] >= 1 for j in res["jobs"] if j["phase"] != "warm")
        rows.append({
            "name": name, "gpu": res["gpu"].split(",")[0], "K": res["K"], "price": price,
            "single_iors_s": single.get("inter_ob"), "single_cam_s": single.get("camera_r"),
            "conc_total_s": res.get("conc_total_wall_s"), "slowdown": res.get("conc_slowdown_mean"),
            "single_jobs_h": res.get("single_jobs_per_h"), "conc_jobs_h": res.get("conc_jobs_per_h"),
            "usd_per_job": (price / res["conc_jobs_per_h"]) if res.get("conc_jobs_per_h") else None,
            "gpu_util_conc": (res.get("gpu_stats", {}).get("conc") or {}).get("util_mean"),
            "mem_peak_gb": round(((res.get("gpu_stats", {}).get("conc") or {}).get("mem_peak_mb") or 0) / 1024, 1),
            "all_ok": ok,
        })
    hdr = f"{'pod':14s} {'GPU':26s} {'K':>2} {'$/h':>5} {'iors1':>6} {'cam1':>6} {'concK':>6} {'slow':>5} {'jobs/h K':>8} {'$/job':>6} {'util%':>5} {'peakGB':>6} ok"
    print(hdr)
    for r in rows:
        f = lambda v, w=6, d=0: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v is not None else f"{str(v):>{w}s}")
        print(f"{r['name']:14s} {r['gpu'][:26]:26s} {r['K']:>2} {f(r['price'],5,2)} {f(r['single_iors_s'])} {f(r['single_cam_s'])} {f(r['conc_total_s'])} "
              f"{f(r['slowdown'],5,2)} {f(r['conc_jobs_h'],8,1)} {f(r['usd_per_job'],6,3)} {f(r['gpu_util_conc'],5,0)} {f(r['mem_peak_gb'],6,1)} {r['all_ok']}")
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    {"pull": pull, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
