#!/usr/bin/env python3
"""trainfast: aggregate the jobs of one or more tags (out/<tag>/<key>/g*/) into successes per GPU-hour and the cost breakdown.

    fleet_summary.py [--tag base --tag fast ...] [--md] [--root DIR]   (--root: alias validation under $OFH, one trunk 2026-11-05)

GPU-hours = job wall (ISGF_END); for a job cancelled or killed before it, slurm's Elapsed (sacct); for a running job, now - start
(ISGF_SAMPLE line). A job's statistics come from job_stats.json
(fast_check.py); for a running job they are recomputed from its logs on the fly.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

F = "/vision/u/minyeong/projects/isg_home/trainfast"


def job_wall(job):
    p = os.path.join(job, "job.txt")
    if not os.path.exists(p):
        return None, "pending"
    start = end = None
    for line in open(p):
        if line.startswith(("ISGF_SAMPLE", "ISGT_SAMPLE")):
            m = re.search(r"start=(\S+)", line)
            if m:
                start = m.group(1)
        if line.startswith("ISGF_END"):
            m = re.search(r"wall_s=(\d+)", line)
            if m:
                end = int(m.group(1))
    if end is not None:
        return end, "done"
    # 2026-11-01: a job cancelled or killed before its ISGF_END line ran for slurm's Elapsed, not until now
    ended = _slurm_ended(job)
    if ended is not None:
        return ended, "ended"
    if start:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(start).timestamp()
            return time.time() - t0, "running"
        except ValueError:
            pass
    return None, "unknown"


_SACCT = {}


def _slurm_ended(job):
    """Elapsed seconds of the job's slurm job (``<job>/jobid``) when it is no longer pending or running, else None."""
    try:
        jid = open(os.path.join(job, "jobid")).read().split()[0]
    except (OSError, IndexError):
        return None
    if jid not in _SACCT:
        out = subprocess.run(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "State,ElapsedRaw"], capture_output=True, text=True).stdout.split("\n")
        _SACCT[jid] = out[0].split("|") if out and out[0] else None
    rec = _SACCT[jid]
    if not rec or rec[0].split()[0] in ("RUNNING", "PENDING", "REQUEUED", "SUSPENDED"):
        return None
    return int(rec[1])


def stats_of(job):
    p = os.path.join(job, "job_stats.json")
    wall, state = job_wall(job)
    if state != "done" or not os.path.exists(p):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_check.py"), job], capture_output=True)
    try:
        st = json.load(open(p))
    except Exception:
        return None
    st["_wall_s"], st["_state"], st["_job"] = wall, state, job
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=None)
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--root", default=F, help="the root whose out/<tag>/<key>/g*/ jobs are summarized (default: trainfast)")
    a = ap.parse_args()
    root = a.root
    tags = a.tag or sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "out", "*")))
    rows = []
    for tag in tags:
        for job in sorted(glob.glob(os.path.join(root, "out", tag, "*", "g*"))):
            st = stats_of(job)
            if st is None:
                continue
            key = os.path.basename(os.path.dirname(job))
            rows.append((tag, key, st))
    def steady(job):
        """(successes after the first finished demo, hours between the first and the last finished demo) from the demos' meta.json."""
        from datetime import datetime
        ends = []
        for m in glob.glob(os.path.join(job, "demo", "*", "meta.json")):
            try:
                d = json.load(open(m))
                t0 = datetime.fromisoformat(d["started_utc"].replace("Z", "+00:00")).timestamp()
                ends.append((t0 + float(d.get("wall_time_s") or 0), bool(d.get("success"))))
            except Exception:
                pass
        if len(ends) < 2:
            return 0, 0.0
        ends.sort()
        return sum(1 for _, ok in ends[1:] if ok), (ends[-1][0] - ends[0][0]) / 3600
    def agg(sel):
        st_succ = sum(steady(j)[0] for j in [r[2]["_job"] for r in sel])
        st_h = sum(steady(j)[1] for j in [r[2]["_job"] for r in sel])
        succ = sum(s["successes"] for _, _, s in sel)
        demos = sum(s["demos"] for _, _, s in sel)
        att = sum(s["attempts"] for _, _, s in sel)
        att_s = sum(s["attempts_s"] for _, _, s in sel)
        wr = sum(s["written"] for _, _, s in sel)
        wall = sum((s["_wall_s"] or 0) for _, _, s in sel)
        dw = [w for _, _, s in sel for w in s["demo_wall_s"] if w]
        sw = [s["sample_wall_s"] for _, _, s in sel if s["sample_wall_s"]]
        rej = {}
        for _, _, s in sel:
            for k, v in s["rejections"].items():
                rej[k] = rej.get(k, 0) + v
        return {"jobs": len(sel), "done": sum(1 for _, _, s in sel if s["_state"] in ("done", "ended")), "attempts": att, "written": wr,
                "s_per_attempt": att_s / att if att else None, "attempts_per_written": att / wr if wr else None,
                "demos": demos, "successes": succ, "s_per_demo": sum(dw) / len(dw) if dw else None,
                "gpu_h": wall / 3600, "succ_per_gpu_h": succ / (wall / 3600) if wall else None,
                "sample_wall_mean_s": sum(sw) / len(sw) if sw else None, "rejections": dict(sorted(rej.items(), key=lambda kv: -kv[1])[:6]),
                "steady_succ_per_gpu_h": (st_succ / st_h) if st_h > 0 else None}
    def fmt(v, d=1):
        return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    hdr = ["tag", "key", "jobs(done)", "attempts", "written", "s/attempt", "att/written", "demos", "succ", "s/demo", "GPU-h", "succ/GPU-h", "steady succ/GPU-h"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for tag in tags:
        sel = [r for r in rows if r[0] == tag]
        if not sel:
            continue
        for key in sorted({k for _, k, _ in sel}):
            g = agg([r for r in sel if r[1] == key])
            out.append("| " + " | ".join([tag, key[:48], f"{g['jobs']}({g['done']})", str(g["attempts"]), str(g["written"]), fmt(g["s_per_attempt"], 0), fmt(g["attempts_per_written"]),
                                           str(g["demos"]), str(g["successes"]), fmt(g["s_per_demo"], 0), fmt(g["gpu_h"], 2), fmt(g["succ_per_gpu_h"]), fmt(g["steady_succ_per_gpu_h"])]) + " |")
        g = agg(sel)
        out.append("| " + " | ".join([f"**{tag}**", "**all**", f"{g['jobs']}({g['done']})", str(g["attempts"]), str(g["written"]), fmt(g["s_per_attempt"], 0), fmt(g["attempts_per_written"]),
                                       str(g["demos"]), str(g["successes"]), fmt(g["s_per_demo"], 0), fmt(g["gpu_h"], 2), fmt(g["succ_per_gpu_h"]), fmt(g["steady_succ_per_gpu_h"])]) + " |")
        out.append(f"| | {tag} rejections | {g['rejections']} |")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
