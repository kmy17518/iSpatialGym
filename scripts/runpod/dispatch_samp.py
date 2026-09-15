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
# pod-side job runner (written by dispatch_samp.py): consumes /root/samp/queue/*.job (one tab-separated line each:
# id, spec, problem, extra args) in name order; waits for more jobs until /root/samp/CLOSE exists and the queue is empty.
set -uo pipefail
source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym
cd /root/ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ISPATIALGYM_LOG_LEVEL=INFO
mkdir -p /root/samp/queue /root/samp/running /root/samp/finished
touch /root/samp/done.txt
echo $$ > /root/samp/runner.pid
while true; do
  next=$(ls /root/samp/queue/*.job 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then
    if [ -f /root/samp/CLOSE ]; then break; fi
    sleep 30; continue
  fi
  mv "$next" /root/samp/running/ || continue
  jf=/root/samp/running/$(basename "$next")
  IFS=$'\t' read -r jid spec problem extra < "$jf"
  out=/root/samp/$jid; mkdir -p "$out"
  echo "$jid" > /root/samp/current.txt
  t0=$(date +%s)
  if [ "$spec" = "CMD" ]; then
    # generic job: <problem> is a shell command run from the repo root with $out as OUT
    OUT="$out" bash -c "$problem $extra" > "$out/run.log" 2>&1
  else
    timeout 5400 python -m ispatialgym.sampling.sample_instances --spec "$spec" --problem "$problem" --out "$out" --overwrite $extra > "$out/run.log" 2>&1
  fi
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
  mv "$jf" /root/samp/finished/
done
rm -f /root/samp/current.txt
touch /root/JOB_DONE
'''


SYNC_DIRS = ["ispatialgym/sampling", "ispatialgym/oracle", "ispatialgym/activity", "ispatialgym/eval", "ispatialgym/tasks",
             "ispatialgym/assets/task_specs", "ispatialgym/assets/icl/roles", "ispatialgym/assets/iddl", "scripts"]


EXPORT_DIR = "/tmp/isg_dispatch_export"


def export_head() -> str:
    """Export the committed HEAD (not the working tree: other agents' uncommitted edits must not reach the fleet)."""
    if os.path.exists(EXPORT_DIR):
        subprocess.run(["rm", "-rf", EXPORT_DIR], check=True)
    os.makedirs(EXPORT_DIR)
    tar = subprocess.Popen(["git", "-C", rp.REPO, "archive", "HEAD"] + SYNC_DIRS, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", EXPORT_DIR], stdin=tar.stdout, check=True)
    tar.wait()
    return EXPORT_DIR


def sync_tree(ep) -> None:
    """rsync the exported HEAD (code + specs) to /root/ispatialgym on the pod (no git: the repo is private)."""
    ip, port = ep
    root = export_head() if not os.path.exists(os.path.join(EXPORT_DIR, ".exported")) else EXPORT_DIR
    open(os.path.join(EXPORT_DIR, ".exported"), "w").close()
    ssh_cmd = " ".join(shlex.quote(x) for x in rp.ssh_base(ip, port)[:-1])
    for d in SYNC_DIRS:
        src = os.path.join(root, d) + "/"
        subprocess.run(["rsync", "-az", "--delete", "--exclude", "__pycache__", "--exclude", "*.pyc", "-e", ssh_cmd, src,
                        f"root@{ip}:/root/ispatialgym/{d}/"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--pods-file", default=os.path.join(rp.REPO, "runlogs", "runpod", "samp", "pods.txt"))
    ap.add_argument("--ref", default="survey")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these pod ids")
    ap.add_argument("--wave", type=int, default=1, help="dispatch wave (queue order prefix)")
    ap.add_argument("--close", action="store_true", help="close the queues: runners exit when empty -> JOB_DONE -> collect.py")
    ap.add_argument("--subtasks", nargs="*", default=None, help="only jobs of these subtasks")
    ap.add_argument("--reset-queue", action="store_true", help="drop queued (not yet started) jobs before adding")
    ap.add_argument("--sync-only", action="store_true", help="only rsync the working tree to the pods")
    ap.add_argument("--rebalance", action="store_true", help="drop every queued job on the ready pods and re-dispatch the manifest jobs that are neither done nor running")
    ap.add_argument("--restart-runner", action="store_true", help="on idle pods (no current job) restart the runner with the current RUNNER script")
    ap.add_argument("--exclude-subtasks", nargs="*", default=None)
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
    jobs = []
    if args.manifest:
        with open(args.manifest) as f:
            jobs = json.load(f)
        if args.subtasks:
            jobs = [j for j in jobs if j.get("subtask") in args.subtasks]
        if args.exclude_subtasks:
            jobs = [j for j in jobs if j.get("subtask") not in args.exclude_subtasks]
    if os.path.exists(EXPORT_DIR):
        subprocess.run(["rm", "-rf", EXPORT_DIR], check=True)
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
    if args.restart_runner:
        for pid, ep in ready:
            r = rp.ssh_run(*ep, "test -f /root/samp/current.txt && echo BUSY", capture=True, timeout=45)
            if "BUSY" in (r.stdout or ""):
                print(f"{pid} {fleet['pods'][pid]['name']}: busy, not restarted")
                continue
            rp.ssh_run(*ep, "test -f /root/samp/runner.pid && kill $(cat /root/samp/runner.pid) 2>/dev/null; sleep 1; cat > /root/samp/run_jobs.sh && chmod +x /root/samp/run_jobs.sh",
                       timeout=60, stdin_text=RUNNER)
            remote = "cd /root && rm -f /root/JOB_DONE /root/samp/CLOSE && setsid nohup /root/samp/run_jobs.sh > /root/samp/runner.log 2>&1 < /dev/null & disown"
            try:
                subprocess.run(rp.ssh_base(*ep) + [remote], text=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except subprocess.TimeoutExpired:
                pass
            print(f"{pid} {fleet['pods'][pid]['name']}: runner restarted")
        return
    if args.sync_only:
        for pid, ep in ready:
            sync_tree(ep)
            print(f"{pid}: synced")
        return
    if args.close:
        for pid, ep in ready:
            rp.ssh_run(*ep, "touch /root/samp/CLOSE", timeout=60)
        print(f"queues closed on {len(ready)} pods (runner exits when its queue is empty -> JOB_DONE)")
        return
    if args.rebalance:
        finished, running = set(), set()
        for pid, ep in ready:
            r = rp.ssh_run(*ep, "cat /root/samp/done.txt 2>/dev/null; echo ---; cat /root/samp/current.txt 2>/dev/null", capture=True, timeout=45)
            out = (r.stdout or "").split("---")
            for line in out[0].splitlines():
                if " rc=" in line:
                    jid, rest = line.split(" rc=", 1)
                    if "n_written=-1" not in rest or "rc=139" not in line:  # a segfault is retried, a real run (even 0 instances) is not
                        finished.add(jid.strip())
            if len(out) > 1 and out[1].strip():
                running.add(out[1].strip())
        before = len(jobs)
        jobs = [j for j in jobs if j["id"] not in finished and j["id"] not in running]
        args.reset_queue = True
        print(f"rebalance: {before} manifest jobs, {len(finished)} finished, {len(running)} running -> {len(jobs)} re-queued")
    per_pod = {pid: [] for pid, _ in ready}
    # heaviest jobs first, round-robin; a wave number keeps later dispatches behind earlier ones in the queue order
    for i, job in enumerate(sorted(jobs, key=lambda j: -float(j.get("weight", 1.0)))):
        per_pod[ready[i % len(ready)][0]].append(job)
    for pid, ep in ready:
        script = ["mkdir -p /root/samp/queue"] + (["rm -f /root/samp/queue/*.job"] if args.reset_queue else [])
        for k, j in enumerate(per_pod[pid]):
            extra = " ".join(shlex.quote(a) for a in j.get("args", []))
            if j.get("n_instances") is not None:
                extra += f" --n-instances {int(j['n_instances'])}"
            if j.get("max_templates") is not None:
                extra += f" --max-templates {int(j['max_templates'])}"
            if j.get("scene"):
                extra += f" --scene {j['scene']}"
            if j.get("cmd"):
                line = "\t".join([j["id"], "CMD", j["cmd"], ""])
            else:
                line = "\t".join([j["id"], j["spec"], j["problem"], extra.strip()])
            script.append(f"printf '%s\\n' {shlex.quote(line)} > /root/samp/queue/w{args.wave:02d}_{k:03d}_{j['id'][:40]}.job")
        script.append(f"printf '%s' {shlex.quote(args.ref)} > /root/samp/ref.txt")
        rp.ssh_run(*ep, "bash -s", timeout=120, stdin_text="\n".join(script) + "\n")
        rp.ssh_run(*ep, "cat > /root/samp/run_jobs.sh && chmod +x /root/samp/run_jobs.sh", timeout=60, stdin_text=RUNNER)
        # the pods cannot pull the private repo after the bootstrap (token shredded): sync the working tree over rsync
        sync_tree(ep)
        running = rp.ssh_run(*ep, "test -f /root/samp/runner.pid && kill -0 $(cat /root/samp/runner.pid) 2>/dev/null && echo RUNNING", capture=True, timeout=45)
        print(f"{pid} {fleet['pods'][pid]['name']}: +{len(per_pod[pid])} jobs; runner {'already running' if 'RUNNING' in (running.stdout or '') else 'started'}")
        if "RUNNING" not in (running.stdout or ""):
            remote = "cd /root && rm -f /root/JOB_DONE /root/samp/CLOSE && setsid nohup /root/samp/run_jobs.sh > /root/samp/runner.log 2>&1 < /dev/null & disown"
            try:
                subprocess.run(rp.ssh_base(*ep) + [remote], text=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except subprocess.TimeoutExpired:
                pass
    with open(os.path.join(os.path.dirname(args.manifest), f"assignment_w{args.wave:02d}.json"), "w") as f:
        json.dump({pid: [j["id"] for j in js] for pid, js in per_pod.items()}, f, indent=1)
    print(f"dispatched {len(jobs)} jobs to {len(ready)} pods (wave {args.wave})")


if __name__ == "__main__":
    main()
