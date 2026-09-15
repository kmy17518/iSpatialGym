#!/usr/bin/env python3
"""Dispatch sampling jobs (one spec file + problem key each) to the bootstrapped ``samp`` pods and run them detached.

    python scripts/runpod/dispatch_samp.py --manifest <dir>/jobs.json [--ref survey] [--pods-file runlogs/runpod/samp/pods.txt]
    python scripts/runpod/dispatch_samp.py --status            # per pod: jobs done / running / failed

``jobs.json`` = ``[{"id": ..., "spec": "<repo-relative spec json>", "problem": "<activity>-<def_id>", "scene": ...,
"n_instances": 2, "max_templates": 1, "args": [...]}, ...]`` (written by ``ispatialgym.sampling.ground_isg``).
Jobs are assigned round-robin (a job list per pod, ``/root/samp/jobs.txt``); the pod runs them sequentially with
``sample_instances.py`` (one process per pod, see TODO.md pitfalls), writes ``/root/samp/<job id>/`` per job and
``/root/samp/done.txt`` lines ``<id> rc=<rc> wall=<s> n_written=<n>``, then ``/root/JOB_DONE`` so ``collect.py
--purpose samp --remote-dir /root/samp`` pulls the results and terminates the pod.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402

RUNNER = r'''#!/bin/bash
# pod-side job runner (written by dispatch_samp.py)
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym
cd /root/ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ISPATIALGYM_LOG_LEVEL=INFO
mkdir -p /root/samp
: > /root/samp/done.txt
while IFS=$'\t' read -r jid spec problem extra; do
  [ -z "$jid" ] && continue
  out=/root/samp/$jid; mkdir -p "$out"
  echo "$jid" > /root/samp/current.txt
  t0=$(date +%s)
  timeout 5400 python -m ispatialgym.sampling.sample_instances --spec "$spec" --problem "$problem" --out "$out" --overwrite $extra > "$out/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  nw=$(python - "$out" <<'EOF'
import glob, json, sys
reps = sorted(glob.glob(sys.argv[1] + "/sampling_report_*.json"))
if not reps:
    print(-1)
else:
    d = json.load(open(reps[-1]))
    print(sum(int(t.get("n_written", 0)) for prob in d.get("problems", {}).values() for t in prob.values() if isinstance(t, dict)))
EOF
)
  echo "$jid rc=$rc wall=$((t1-t0)) n_written=$nw" >> /root/samp/done.txt
  # keep the logs small: drop Isaac Sim's cache dumps, keep run.log + outputs
done < /root/samp/jobs.txt
rm -f /root/samp/current.txt
touch /root/JOB_DONE
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--pods-file", default=os.path.join(rp.REPO, "runlogs", "runpod", "samp", "pods.txt"))
    ap.add_argument("--ref", default="survey")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these pod ids")
    args = ap.parse_args()
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    fleet = rp.load_fleet()
    live = {p["id"] for p in api.myself()["pods"]}
    pods = [l.strip() for l in open(args.pods_file) if l.strip()]
    pods = [p for p in pods if p in live and not fleet["pods"].get(p, {}).get("terminated_at")]
    if args.only:
        pods = [p for p in pods if p in args.only]
    if args.status:
        for pid in pods:
            ep = rp.pod_ssh(api, pid)
            if ep is None:
                print(f"{pid} {fleet['pods'][pid]['name']}: no ssh")
                continue
            try:
                r = rp.ssh_run(*ep, "test -f /root/BOOTSTRAP_DONE && echo BOOT_OK; test -f /root/BOOTSTRAP_FAILED && echo BOOT_FAILED; "
                               "test -f /root/JOB_DONE && echo JOB_DONE; cat /root/samp/current.txt 2>/dev/null; wc -l < /root/samp/jobs.txt 2>/dev/null; "
                               "cat /root/samp/done.txt 2>/dev/null | tail -20", capture=True, timeout=45)
                print(f"== {pid} {fleet['pods'][pid]['name']}\n{(r.stdout or '').strip()}")
            except SystemExit:
                print(f"{pid}: ssh failed")
        return
    with open(args.manifest) as f:
        jobs = json.load(f)
    ready = []
    for pid in pods:
        ep = rp.pod_ssh(api, pid)
        if ep is None:
            print(f"{pid}: no ssh yet, skipped")
            continue
        try:
            r = rp.ssh_run(*ep, "test -f /root/BOOTSTRAP_DONE && echo OK", capture=True, timeout=45)
        except SystemExit:
            continue
        if "OK" in (r.stdout or ""):
            ready.append((pid, ep))
        else:
            print(f"{pid} {fleet['pods'][pid]['name']}: bootstrap not done, skipped")
    if not ready:
        sys.exit("no bootstrapped pod")
    per_pod = {pid: [] for pid, _ in ready}
    # longest jobs first, round-robin
    for i, job in enumerate(sorted(jobs, key=lambda j: -float(j.get("weight", 1.0)))):
        per_pod[ready[i % len(ready)][0]].append(job)
    for pid, ep in ready:
        lines = []
        for j in per_pod[pid]:
            extra = " ".join(shlex.quote(a) for a in j.get("args", []))
            if j.get("n_instances") is not None:
                extra += f" --n-instances {int(j['n_instances'])}"
            if j.get("max_templates") is not None:
                extra += f" --max-templates {int(j['max_templates'])}"
            if j.get("scene"):
                extra += f" --scene {j['scene']}"
            lines.append("\t".join([j["id"], j["spec"], j["problem"], extra.strip()]))
        jobs_txt = "\n".join(lines) + "\n"
        rp.ssh_run(*ep, "mkdir -p /root/samp && cat > /root/samp/jobs.txt", timeout=60, stdin_text=jobs_txt)
        rp.ssh_run(*ep, "cat > /root/samp/run_jobs.sh && chmod +x /root/samp/run_jobs.sh", timeout=60, stdin_text=RUNNER)
        pull = f"cd /root/ispatialgym && git fetch -q --depth 1 origin {shlex.quote(args.ref)} && git checkout -q FETCH_HEAD && git log --oneline -1"
        r = rp.ssh_run(*ep, pull, capture=True, timeout=180)
        print(f"{pid} {fleet['pods'][pid]['name']}: {len(per_pod[pid])} jobs; repo at {(r.stdout or '').strip()}")
        remote = "cd /root && rm -f /root/JOB_DONE && setsid nohup /root/samp/run_jobs.sh > /root/samp/runner.log 2>&1 < /dev/null & disown"
        try:
            subprocess.run(rp.ssh_base(*ep) + [remote], text=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except subprocess.TimeoutExpired:
            pass
    with open(os.path.join(os.path.dirname(args.manifest), "assignment.json"), "w") as f:
        json.dump({pid: [j["id"] for j in js] for pid, js in per_pod.items()}, f, indent=1)
    print(f"dispatched {len(jobs)} jobs to {len(ready)} pods")


if __name__ == "__main__":
    main()
