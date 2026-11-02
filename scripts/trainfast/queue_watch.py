#!/usr/bin/env python3
"""Queue helpers for job_train.sh (no simulator).

    queue_watch.py <job dir> left                    -> number of ready instances without a claim (run_isg queue_ready's rule)
    queue_watch.py <job dir> over <pid> <limit_s>    -> the instance id the queue process <pid> has run for more than limit_s, else ""
"""
import glob
import json
import os
import sys
import time

SUMMARY = ".summary.json"
TRO = "-tro_state.json"


def ready(job):
    out = []
    for s in sorted(glob.glob(os.path.join(job, "inst", "*", "*", "instances", "*" + SUMMARY))):
        iid = os.path.basename(s)[: -len(SUMMARY)]
        d = os.path.dirname(s)
        if os.path.exists(os.path.join(d, iid + TRO)) and os.path.exists(os.path.join(os.path.dirname(d), "template_meta.json")):
            out.append(iid)
    return out


def finished(job):
    done = set()
    try:
        for line in open(os.path.join(job, "results.jsonl")):
            try:
                done.add(json.loads(line)["instance"])
            except (ValueError, KeyError):
                pass
    except OSError:
        pass
    return done


def main():
    job, cmd = sys.argv[1], sys.argv[2]
    claims = os.path.join(job, "claims")
    if cmd == "left":
        print(sum(1 for i in ready(job) if not os.path.exists(os.path.join(claims, i))))
        return 0
    if cmd == "over":
        pid, limit = sys.argv[3], float(sys.argv[4])
        done = finished(job)
        now = time.time()
        for c in glob.glob(os.path.join(claims, "*")):
            try:
                parts = open(c).read().split()
            except OSError:
                continue
            if len(parts) >= 2 and parts[1] == pid and os.path.basename(c) not in done and now - os.path.getmtime(c) > limit:
                print(os.path.basename(c))
                return 0
        print("")
        return 0
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())
