#!/usr/bin/env python3
"""Live summary of the ``samp`` fleet: pulls ``done.txt`` + the sampling reports (small JSON) of every pod into
``runlogs/runpod/samp/live/<pod name>/`` and prints per-subtask counts (jobs done, instances written, rejections).

    python scripts/runpod/samp_summary.py [--no-pull] [--md runlogs/runpod/samp/summary.md]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402

LIVE = os.path.join(rp.REPO, "runlogs", "runpod", "samp", "live")
JOBS = os.path.join(rp.REPO, "ispatialgym", "assets", "task_specs", "isg_reps", "jobs.json")


def pull(api, pid: str, name: str) -> None:
    ep = rp.pod_ssh(api, pid)
    if ep is None:
        return
    ip, port = ep
    dst = os.path.join(LIVE, name)
    os.makedirs(dst, exist_ok=True)
    ssh_cmd = " ".join(shlex.quote(x) for x in rp.ssh_base(ip, port)[:-1])
    subprocess.run(["rsync", "-az", "-e", ssh_cmd, "--include", "*/", "--include", "done.txt", "--include", "current.txt",
                    "--include", "sampling_report_*.json", "--include", "run.log", "--exclude", "*", "--prune-empty-dirs",
                    f"root@{ip}:/root/samp/", dst + "/"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--purpose", default="samp")
    ap.add_argument("--jobs", default=None, help="job manifest (default: the isg_reps jobs.json)")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    global LIVE, JOBS
    LIVE = os.path.join(rp.REPO, "runlogs", "runpod", args.purpose, "live")
    if args.jobs:
        JOBS = args.jobs
    if args.md is None:
        args.md = os.path.join(rp.REPO, "runlogs", "runpod", args.purpose, "summary.md")
    fleet = rp.load_fleet()
    pods = {pid: info for pid, info in fleet["pods"].items() if info.get("purpose") == args.purpose}
    if not args.no_pull:
        api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
        live = {p["id"] for p in api.myself()["pods"]}
        for pid, info in pods.items():
            if pid in live:
                pull(api, pid, info["name"])
    jobs = {j["id"]: j for j in json.load(open(JOBS))}
    for extra in glob.glob(os.path.join(os.path.dirname(JOBS), "jobs_*.json")):
        for j in json.load(open(extra)):
            jobs.setdefault(j["id"], j)
    done = {}
    rejections = collections.defaultdict(collections.Counter)
    current = {}
    for pod_dir in sorted(glob.glob(os.path.join(LIVE, "*"))):
        name = os.path.basename(pod_dir)
        cur = os.path.join(pod_dir, "current.txt")
        if os.path.exists(cur):
            current[name] = open(cur).read().strip()
        dpath = os.path.join(pod_dir, "done.txt")
        if not os.path.exists(dpath):
            continue
        for line in open(dpath):
            m = re.match(r"(\S+) rc=(-?\d+) wall=(\d+) n_written=(-?\d+)", line.strip())
            if not m:
                continue
            jid, rc, wall, nw = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            prev = done.get(jid)
            if prev is None or nw > prev["n_written"]:
                done[jid] = {"rc": rc, "wall": wall, "n_written": nw, "pod": name}
            for rep in glob.glob(os.path.join(pod_dir, jid, "sampling_report_*.json")):
                try:
                    d = json.load(open(rep))
                except Exception:
                    continue
                for prob in d.get("problems", {}).values():
                    for t in prob.values():
                        if isinstance(t, dict):
                            for k, v in (t.get("rejections") or {}).items():
                                rejections[jobs.get(jid, {}).get("subtask", "?")][k] += int(v)
                            if t.get("error"):
                                rejections[jobs.get(jid, {}).get("subtask", "?")]["ERROR:" + str(t["error"])[:60]] += 1
    by = collections.defaultdict(lambda: {"jobs": 0, "done": 0, "ok": 0, "instances": 0, "segv": 0, "wall": 0})
    for jid, j in jobs.items():
        st = j.get("subtask", "?")
        by[st]["jobs"] += 1
        if jid in done:
            d = done[jid]
            by[st]["done"] += 1
            by[st]["ok"] += 1 if d["n_written"] > 0 else 0
            by[st]["instances"] += max(0, d["n_written"])
            by[st]["segv"] += 1 if d["rc"] == 139 else 0
            by[st]["wall"] += d["wall"]
    lines = ["| subtask | jobs | done | with >=1 instance | instances | segfaults | mean wall s | top rejections |", "|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for st in sorted(by):
        b = by[st]
        top = ", ".join(f"{k} {v}" for k, v in rejections[st].most_common(4))
        lines.append(f"| {st} | {b['jobs']} | {b['done']} | {b['ok']} | {b['instances']} | {b['segv']} | {b['wall'] // max(1, b['done'])} | {top} |")
        for k in ("jobs", "done", "ok", "instances", "segv"):
            tot[k] += b[k]
    lines.append(f"| **total** | {tot['jobs']} | {tot['done']} | {tot['ok']} | {tot['instances']} | {tot['segv']} | | |")
    lines.append("")
    lines.append("Running now: " + "; ".join(f"{k}: {v}" for k, v in sorted(current.items())))
    text = "\n".join(lines)
    print(text)
    if args.md:
        os.makedirs(os.path.dirname(args.md), exist_ok=True)
        with open(args.md, "w") as f:
            f.write("# samp fleet summary\n\n" + text + "\n")
    with open(os.path.join(os.path.dirname(args.md), "done.json"), "w") as f:
        json.dump(done, f, indent=1)


if __name__ == "__main__":
    main()
