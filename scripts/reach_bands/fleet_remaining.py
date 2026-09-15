#!/usr/bin/env python3
"""Which manifest jobs still have to run on the ``rb`` fleet (for a re-dispatch with ``dispatch_samp.py --reset-queue``).

    python scripts/reach_bands/fleet_remaining.py --manifest runlogs/reach_bands_20260915/jobs.json [--manifest ...] \
        --pods-file runlogs/runpod/rb/pods.txt --out runlogs/reach_bands_20260915/jobs_remaining.json

A job counts as done when its output directory on some pod holds a ``.npz`` band file (``/root/samp/<id>/*.npz``);
``done.txt`` return codes are not used (the stale-code ImportError runs of 2026-09-15 ended with rc=1 after 15 s, the
OmniGibson shutdown masks the exit code of real failures with rc=0). Jobs currently running (``current.txt``) are kept
out of the remaining set too. ``--local <dir>`` adds a local runner directory (``run_local.sh`` layout).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "runpod"))
import rp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--pods-file", default=os.path.join(REPO, "runlogs", "runpod", "rb", "pods.txt"))
    ap.add_argument("--local", nargs="*", default=[], help="local runner output roots (jobs with a .npz there count as done)")
    ap.add_argument("--pulled", nargs="*", default=[], help="pulled fleet roots (<root>/<pod>/<job>/*.npz), for pods that are gone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-kinds", nargs="*", default=None)
    args = ap.parse_args()
    jobs = []
    seen = set()
    for m in args.manifest:
        for j in json.load(open(m)):
            if j["id"] not in seen:
                seen.add(j["id"])
                jobs.append(j)
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    fleet = rp.load_fleet()
    live = {p["id"] for p in api.myself()["pods"]}
    done, running, per_pod = set(), set(), {}
    for pid in [l.strip() for l in open(args.pods_file) if l.strip()]:
        if pid not in live or fleet["pods"].get(pid, {}).get("terminated_at"):
            continue
        ep = rp.pod_ssh(api, pid)
        if ep is None:
            print(f"{pid}: no ssh", file=sys.stderr)
            continue
        try:
            r = rp.ssh_run(*ep, "ls /root/samp/*/*.npz 2>/dev/null; echo ---; cat /root/samp/current.txt 2>/dev/null; echo ---; ls /root/samp/queue 2>/dev/null | wc -l",
                           capture=True, timeout=60)
        except SystemExit:
            print(f"{pid}: ssh failed", file=sys.stderr)
            continue
        parts = (r.stdout or "").split("---")
        d = {os.path.basename(os.path.dirname(l.strip())) for l in parts[0].splitlines() if l.strip()}
        cur = parts[1].strip() if len(parts) > 1 else ""
        per_pod[pid] = {"name": fleet["pods"].get(pid, {}).get("name"), "done": sorted(d), "running": cur, "queued": int(parts[2].strip() or 0) if len(parts) > 2 else None}
        done |= d
        if cur:
            running.add(cur)
    for root in args.local:
        for f in glob.glob(os.path.join(root, "*", "*.npz")):
            done.add(os.path.basename(os.path.dirname(f)))
    for root in args.pulled:  # collect.py / pull_loop.sh layout: <root>/<pod name>/<job id>/*.npz (pods that are gone)
        for f in glob.glob(os.path.join(root, "*", "*", "*.npz")):
            done.add(os.path.basename(os.path.dirname(f)))
    remaining = [j for j in jobs if j["id"] not in done and j["id"] not in running and (not args.only_kinds or j["kind"] in args.only_kinds)]
    with open(args.out, "w") as f:
        json.dump(remaining, f, indent=1)
    with open(os.path.splitext(args.out)[0] + "_fleet_state.json", "w") as f:
        json.dump({"done": sorted(done & seen), "running": sorted(running), "per_pod": per_pod}, f, indent=1)
    print(f"{len(jobs)} manifest jobs: {len(done & seen)} done (band files), {len(running & seen)} running, {len(remaining)} remaining -> {args.out}")
    for pid, st in per_pod.items():
        print(f"  {pid} {st['name']}: {len(st['done'])} done, running {st['running'] or '-'}, {st['queued']} queued")


if __name__ == "__main__":
    main()
