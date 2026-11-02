#!/usr/bin/env python3
"""Queue ``run_isg`` demo jobs on the pods that hold the sampled instances (one CMD job per finished sampling job).

    python scripts/runpod/samp_summary.py            # refresh runlogs/runpod/samp/done.json (job id -> pod)
    python scripts/runpod/make_demo_jobs.py [--subtasks camera_relocalization-standard ...] [--dispatch] [--wave 5]

Writes ``ispatialgym/assets/task_specs/isg_reps/jobs_demo_<pod>.json`` per pod and, with ``--dispatch``, calls
``dispatch_samp.py --only <pod> --no-sync`` for each (run ``dispatch_samp.py --sync-only`` once before so that
``ispatialgym/oracle/run_isg.py`` is on the pods). Demo jobs already queued or done (``demo__<jid>``) are skipped.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402

REPS = os.path.join(rp.REPO, "ispatialgym", "assets", "task_specs", "isg_reps")
LIVE = os.path.join(rp.REPO, "runlogs", "runpod", "samp", "live")
ORACLE_SUBTASKS = ("camera_relocalization-standard", "camera_relocalization-obstructed", "object_scaling", "mental_rotation")
CMD = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
       "for tro in /root/samp/{jid}/*/*/instances/*-tro_state.json; do "
       "OMNIGIBSON_HEADLESS=1 timeout 1500 python -m ispatialgym.oracle.run_isg --instance \"$tro\" --out \"$OUT\" --video --video-size 540; "
       "echo \"DEMO_EXIT $? $tro\"; done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtasks", nargs="*", default=list(ORACLE_SUBTASKS))
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--wave", type=int, default=5)
    ap.add_argument("--all-activities", action="store_true", help="also queue no-oracle activities (loads + init render only)")
    args = ap.parse_args()
    done = json.load(open(os.path.join(rp.REPO, "runlogs", "runpod", "samp", "done.json")))
    jobs = {j["id"]: j for j in json.load(open(os.path.join(REPS, "jobs.json")))}
    fleet = rp.load_fleet()
    name_to_pid = {info["name"]: pid for pid, info in fleet["pods"].items()}
    # demo jobs already recorded on the pods (done.txt) or queued
    known = set()
    for pod_dir in glob.glob(os.path.join(LIVE, "*")):
        dpath = os.path.join(pod_dir, "done.txt")
        if os.path.exists(dpath):
            for line in open(dpath):
                if line.startswith("demo__"):
                    m = re.search(r"rc=(-?\d+) wall=(\d+)", line)
                    if m and int(m.group(1)) != 0 and int(m.group(2)) < 5:
                        continue  # runner without CMD support / instant failure: re-queue
                    known.add(line.split()[0])
    per_pod = collections.defaultdict(list)
    for jid, d in done.items():
        j = jobs.get(jid)
        if j is None or d["n_written"] <= 0:
            continue
        if not args.all_activities and j["subtask"] not in args.subtasks:
            continue
        did = f"demo__{jid}"
        if did in known:
            continue
        per_pod[d["pod"]].append({"id": did, "cmd": CMD.format(jid=jid), "weight": 1.0, "subtask": j["subtask"], "slot": j["slot"], "variant": j["variant"],
                                  "source_job": jid})
    total = 0
    for pod, js in per_pod.items():
        path = os.path.join(REPS, f"jobs_demo_{pod}.json")
        with open(path, "w") as f:
            json.dump(js, f, indent=1)
        total += len(js)
        print(f"{pod}: {len(js)} demo jobs -> {path}")
        if args.dispatch and pod in name_to_pid:
            subprocess.run([sys.executable, os.path.join(HERE, "dispatch_samp.py"), "--manifest", path, "--only", name_to_pid[pod], "--wave", str(args.wave),
                            "--no-sync"], check=False)
    print(f"{total} demo jobs on {len(per_pod)} pods")


if __name__ == "__main__":
    main()
