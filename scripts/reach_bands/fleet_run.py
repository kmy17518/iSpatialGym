#!/usr/bin/env python3
"""Run one shell command on every live pod of a pods file (band fleet housekeeping), printing each pod's output.

    python scripts/reach_bands/fleet_run.py --pods-file runlogs/runpod/pilot/pods_bands.txt "cat /root/samp/current.txt; ls /root/samp/queue | wc -l"
    python scripts/reach_bands/fleet_run.py --pods-file ... --only <pod id> ... "pkill -f 'ispatialgym.sampling.reach_band[s]'"

Used after ``dispatch_samp.py --restart-runner --force`` (which kills the runner and the *sampling* processes but not a
running ``reach_bands`` job) to stop the orphaned band processes before the queue is rebuilt, and to inspect queues.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "runpod"))
import rp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--pods-file", default=os.path.join(REPO, "runlogs", "runpod", "pilot", "pods_bands.txt"))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these pod ids")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    fleet = rp.load_fleet()
    live = {p["id"] for p in api.myself()["pods"]}
    pods = [l.strip() for l in open(args.pods_file) if l.strip()]
    pods = [p for p in pods if p in live and not fleet["pods"].get(p, {}).get("terminated_at")]
    if args.only:
        pods = [p for p in pods if p in args.only]
    for pid in pods:
        name = fleet["pods"].get(pid, {}).get("name", pid)
        ep = rp.pod_ssh(api, pid)
        if ep is None:
            print(f"== {pid} {name}: no ssh")
            continue
        try:
            r = rp.ssh_run(*ep, args.command, capture=True, timeout=args.timeout)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
        except SystemExit as e:
            out = f"ssh failed: {e}"
        print(f"== {pid} {name}\n{out}")


if __name__ == "__main__":
    main()
