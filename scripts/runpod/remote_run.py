#!/usr/bin/env python3
"""Run one command on a pod of the shared **smoke-test pool** (purpose ``smoke``) instead of the contended local GPU.

    python scripts/runpod/remote_run.py [--sync-tree] [--pull REMOTE LOCAL] [--timeout S] -- '<shell command>'

* Picks the first pool pod (``runlogs/runpod/smoke/pods.txt``) whose lock file is free (one job per pod; waits up to
  ``--wait`` seconds for a free pod), holds the lock while the command runs.
* ``--sync-tree`` (default on) rsyncs the local **working tree** (``ispatialgym/``, ``scripts/`` and the runtime
  runlogs — uncommitted edits included, exactly what a local run would see) to ``/root/ispatialgym`` on the pod first.
  Other agents' uncommitted edits in the shared tree come along, as they would locally.
* The command runs on the pod as ``cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda
  activate ispatialgym && export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES && <command>`` with its stdout/stderr
  streamed here; the exit code is the command's. ``$SMOKE_OUT`` (= ``/root/smoke/<run id>``) is exported for outputs.
* ``--pull REMOTE LOCAL`` rsyncs a pod path back after the run (e.g. ``--pull '$SMOKE_OUT' /tmp/my_smoke``).

Pods have the datasets, the conda env and the grasp library (bootstrapped from the repo); they do NOT have the
local ``~/.cache/ispatialgym`` caches or anything under ``/tmp``. Inputs a run needs (an instance directory, a spec)
go under ``runlogs/smoke_inputs/`` (git-ignored, synced) and are referenced by that repo-relative path. GPU per pod: L40S 48 GB (two sampler processes
would fit, but the lock keeps it to one).
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402

POOL = os.path.join(rp.REPO, "runlogs", "runpod", "smoke")
PODS_FILE = os.path.join(POOL, "pods.txt")
LOCKS = os.path.join(POOL, "locks")
SYNC_DIRS = ["ispatialgym", "scripts", "runlogs/smoke_inputs", "runlogs/reach_volume_20260915", "runlogs/reach_bands_20260915", "runlogs/scene_inventory_20260915",
             "runlogs/symmetry_20260915", "runlogs/held_visibility_20260915"]
EXCLUDES = ["__pycache__", "*.pyc", "behavior-vision-suite.github.io", "data_collection", ".pytest_cache", "renders", "strips", "*.png", "*.jpg", "*.mp4"]


def sync_tree(ep) -> None:
    ip, port = ep
    ssh_cmd = " ".join(shlex.quote(x) for x in rp.ssh_base(ip, port)[:-1])
    ex = [x for e in EXCLUDES for x in ("--exclude", e)]
    for d in SYNC_DIRS:
        src = os.path.join(rp.REPO, d)
        if not os.path.isdir(src):
            continue
        subprocess.run(["rsync", "-az"] + ex + ["-e", ssh_cmd, src + "/", f"root@{ip}:/root/ispatialgym/{d}/"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)


def acquire_pod(api, wait_s: int):
    os.makedirs(LOCKS, exist_ok=True)
    pods = [l.strip() for l in open(PODS_FILE) if l.strip()]
    t0 = time.time()
    while time.time() - t0 < wait_s:
        for pid in pods:
            fh = open(os.path.join(LOCKS, pid + ".lock"), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            ep = rp.pod_ssh(api, pid)
            if ep is None:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
                continue
            try:
                r = rp.ssh_run(*ep, "test -f /root/BOOTSTRAP_DONE && echo OK", capture=True, timeout=45)
            except SystemExit:
                r = None
            if r is None or "OK" not in (r.stdout or ""):
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
                continue
            return pid, ep, fh
        print(f"remote_run: all {len(pods)} pool pods busy or not ready; waiting...", file=sys.stderr, flush=True)
        time.sleep(30)
    sys.exit("remote_run: no pool pod became free in time")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync-tree", dest="sync", action="store_true", default=True)
    ap.add_argument("--no-sync", dest="sync", action="store_false")
    ap.add_argument("--pull", nargs=2, metavar=("REMOTE", "LOCAL"), default=None)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds for the remote command")
    ap.add_argument("--wait", type=int, default=3600, help="seconds to wait for a free pool pod")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = " ".join(args.command).lstrip("- ").strip() if args.command and args.command[0] == "--" else " ".join(args.command)
    if not cmd:
        sys.exit("usage: remote_run.py [options] -- '<shell command>'")
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    pid, ep, lock_fh = acquire_pod(api, args.wait)
    run_id = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
    print(f"remote_run: pod {pid} ({ep[0]}:{ep[1]}), run id {run_id}", file=sys.stderr, flush=True)
    try:
        if args.sync:
            sync_tree(ep)
        remote = (f"mkdir -p /root/smoke/{run_id} && cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && "
                  f"conda activate ispatialgym && export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES SMOKE_OUT=/root/smoke/{run_id} && "
                  f"timeout {int(args.timeout)} bash -c {shlex.quote(cmd)}")
        r = subprocess.run(rp.ssh_base(*ep) + [remote], stdin=subprocess.DEVNULL)
        rc = r.returncode
        if args.pull:
            remote_path = args.pull[0].replace("$SMOKE_OUT", f"/root/smoke/{run_id}")
            os.makedirs(args.pull[1], exist_ok=True)
            rp.rsync(remote_path.rstrip("/") + "/", args.pull[1].rstrip("/") + "/", *ep, to_pod=False)
        print(f"remote_run: exit {rc} (pod {pid}, run {run_id})", file=sys.stderr, flush=True)
        sys.exit(rc)
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == "__main__":
    main()
