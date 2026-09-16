#!/usr/bin/env python3
"""Dispatch sampling jobs (one spec file + problem key each) to the bootstrapped ``samp`` pods and run them detached.

    python scripts/runpod/dispatch_samp.py --manifest <dir>/jobs.json [--ref survey] [--pods-file runlogs/runpod/samp/pods.txt]
    python scripts/runpod/dispatch_samp.py --status            # per pod: jobs done / running / failed

``jobs.json`` = ``[{"id": ..., "spec": "<repo-relative spec json>", "problem": "<activity>-<def_id>", "scene": ...,
"n_instances": 2, "max_templates": 1, "args": [...]}, ...]`` (written by ``ispatialgym.sampling.v1_bridge``).
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
reps = sorted(glob.glob(sys.argv[1] + "/sampling_report_*.json")) or ([sys.argv[1] + "/summary.json"] if glob.glob(sys.argv[1] + "/summary.json") else [])
if not reps:
    print(-1)  # no report at all: the run crashed before writing one (retried by --rebalance)
else:
    d = json.load(open(reps[-1]))
    n = sum(int(t.get("n_written", 0)) for prob in d.get("problems", {}).values() for t in prob.values() if isinstance(t, dict))
    errs = [t.get("error") for prob in d.get("problems", {}).values() for t in prob.values() if isinstance(t, dict) and t.get("error")]
    print(-1 if (n == 0 and (errs or d.get("errors"))) else n)  # a code error (not a sampling outcome) counts as no report
EOF
)
  echo "$jid rc=$rc wall=$((t1-t0)) n_written=$nw" >> /root/samp/done.txt
  mv "$jf" /root/samp/finished/
done
rm -f /root/samp/current.txt
touch /root/JOB_DONE
'''


#: the whole python package (every sub-package a pod may import: sampling, oracle, robot, activity, eval, tasks, ...) plus
#: the assets the samplers read and the scripts; the 700 MB website checkout and __pycache__ are excluded in the rsync
SYNC_DIRS = ["ispatialgym", "scripts"]
SYNC_EXCLUDES = ["__pycache__", "*.pyc", "behavior-vision-suite.github.io", "data_collection", ".pytest_cache"]
#: committed runlogs the samplers read at run time (reach volume / bands, symmetry, scene inventory); synced when present
SYNC_RUNLOGS = ["runlogs/reach_volume_20260915", "runlogs/reach_bands_20260915", "runlogs/scene_inventory_20260915", "runlogs/symmetry_20260915",
                "runlogs/held_visibility_20260915"]


EXPORT_DIR = "/tmp/isg_dispatch_export"


CODE_REF = "HEAD"


def export_head() -> str:
    """Export the committed ``CODE_REF`` (not the working tree: other agents' uncommitted edits must not reach the fleet)."""
    if os.path.exists(EXPORT_DIR):
        subprocess.run(["rm", "-rf", EXPORT_DIR], check=True)
    os.makedirs(EXPORT_DIR)
    tracked = subprocess.run(["git", "-C", rp.REPO, "ls-tree", "-d", "--name-only", CODE_REF] + SYNC_RUNLOGS, capture_output=True, text=True)
    dirs = SYNC_DIRS + [d for d in tracked.stdout.split() if d]
    tar = subprocess.Popen(["git", "-C", rp.REPO, "archive", CODE_REF] + dirs, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", EXPORT_DIR], stdin=tar.stdout, check=True)
    tar.wait()
    return EXPORT_DIR


def sync_tree(ep) -> None:
    """rsync the exported HEAD (code + specs) to /root/ispatialgym on the pod (no git: the repo is private)."""
    ip, port = ep
    root = export_head() if not os.path.exists(os.path.join(EXPORT_DIR, ".exported")) else EXPORT_DIR
    open(os.path.join(EXPORT_DIR, ".exported"), "w").close()
    ssh_cmd = " ".join(shlex.quote(x) for x in rp.ssh_base(ip, port)[:-1])
    excludes = [x for e in SYNC_EXCLUDES for x in ("--exclude", e)]
    for d in SYNC_DIRS + [d for d in SYNC_RUNLOGS if os.path.isdir(os.path.join(root, d))]:
        src = os.path.join(root, d) + "/"
        # no --delete for the package root: the pods keep their local caches / outputs under the repo untouched
        subprocess.run(["rsync", "-az"] + excludes + ["-e", ssh_cmd, src, f"root@{ip}:/root/ispatialgym/{d}/"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)


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
    ap.add_argument("--no-sync", action="store_true", help="do not rsync the code to the pods (a --sync-only pass did it)")
    ap.add_argument("--code-ref", default="HEAD", help="git ref whose committed tree is synced to the pods")
    ap.add_argument("--rebalance", action="store_true", help="drop every queued job on the ready pods and re-dispatch the manifest jobs that are neither done nor running")
    ap.add_argument("--force", action="store_true", help="with --restart-runner: also on busy pods (kills the current job and re-queues it)")
    ap.add_argument("--restart-runner", action="store_true", help="on idle pods (no current job) restart the runner with the current RUNNER script")
    ap.add_argument("--exclude-subtasks", nargs="*", default=None)
    args = ap.parse_args()
    global CODE_REF
    CODE_REF = args.code_ref
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
            busy = "BUSY" in (r.stdout or "")
            if busy and not args.force:
                print(f"{pid} {fleet['pods'][pid]['name']}: busy, not restarted")
                continue
            stop = "test -f /root/samp/runner.pid && kill $(cat /root/samp/runner.pid) 2>/dev/null; sleep 1; "
            if busy:  # kill the current job's process tree and put its job file back into the queue (re-run from scratch)
                stop += ("pkill -f 'ispatialgym.sampling.sample_instances' ; pkill -f 'ispatialgym.sampling.sample_isg' ; pkill -f 'ispatialgym.sampling.multiply_isg' ; "
                         "pkill -f 'ispatialgym.oracle.run_isg' ; sleep 3; "
                         "for f in /root/samp/running/*.job; do [ -f \"$f\" ] && mv \"$f\" /root/samp/queue/; done; rm -f /root/samp/current.txt; ")
            stop += ("rm -f /tmp/seen_jids; for f in $(ls /root/samp/queue/*.job 2>/dev/null | sort); do jid=$(cut -f1 \"$f\"); "
                     "if grep -qsx \"$jid\" /tmp/seen_jids; then rm -f \"$f\"; else echo \"$jid\" >> /tmp/seen_jids; fi; done; rm -f /tmp/seen_jids; ")
            rp.ssh_run(*ep, stop + "cat > /root/samp/run_jobs.sh && chmod +x /root/samp/run_jobs.sh", timeout=120, stdin_text=RUNNER)
            remote = "cd /root && rm -f /root/JOB_DONE /root/samp/CLOSE && setsid nohup /root/samp/run_jobs.sh > /root/samp/runner.log 2>&1 < /dev/null & disown"
            try:
                subprocess.run(rp.ssh_base(*ep) + [remote], text=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            except subprocess.TimeoutExpired:
                pass
            print(f"{pid} {fleet['pods'][pid]['name']}: runner restarted{' (job re-queued)' if busy else ''}")
        return
    if args.sync_only:
        export_head()
        open(os.path.join(EXPORT_DIR, ".exported"), "w").close()
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda pe: sync_tree(pe[1]), ready))
        print(f"synced {len(ready)} pods")
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
                    if "n_written=-1" not in rest:  # segfaults / crashes (no report or a code error) are retried, a real run (even 0 instances) is not
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
            # deterministic file name per job id: re-dispatching the same job overwrites instead of duplicating
            script.append(f"printf '%s\\n' {shlex.quote(line)} > /root/samp/queue/w{args.wave:02d}_{j['id'][:120]}.job")
        script.append(f"printf '%s' {shlex.quote(args.ref)} > /root/samp/ref.txt")
        rp.ssh_run(*ep, "bash -s", timeout=120, stdin_text="\n".join(script) + "\n")
        rp.ssh_run(*ep, "cat > /root/samp/run_jobs.sh && chmod +x /root/samp/run_jobs.sh", timeout=60, stdin_text=RUNNER)
        # the pods cannot pull the private repo after the bootstrap (token shredded): sync the working tree over rsync
        if not args.no_sync:
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
