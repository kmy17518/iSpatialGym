#!/usr/bin/env python3
"""trainfast: where the time goes, per tag (base / flags / fast ...), from the job logs and the written instances' summaries.

    stage_summary.py [--tag base --tag flags --tag fast] [--key <substring>]

Sampler: attempts (rejected: "rejected in <stage> after <s>s", written: "written (attempt K, <s>s)"), the written instances'
``phase_s`` and ``kit`` counters (.summary.json, both code versions), ``start_per_attempt.draw_s`` and the SPA cache facts, and on the
train-fast code the per-attempt "stages:" / "kit:" lines. Collector: results.jsonl wall_s / n_steps, meta.json timings and the
``fast`` block (straight-arm counters), the GPU memory peak from gpu.log.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX_REJ = re.compile(r"\] (\S+): instance (\d+) attempt (\d+) rejected in (\S+) after ([\d.]+)s: \[?([^\]\n]*)")
RX_WRITTEN = re.compile(r"\] (\S+): instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")
RX_STAGES = re.compile(r"instance (\d+) attempt (\d+) stages: (.*) total=([\d.]+)s")
RX_KIT = re.compile(r"instance (\d+) attempt (\d+) kit: (.*)$")
RX_SPA = re.compile(r"start draw cache (hit|miss|memo)[^\n]* in ([\d.]+)s")
RX_WORLD = re.compile(r"world loaded for \S+ in ([\d.]+)s")
RX_INSTLOAD = re.compile(r"instance loaded in ([\d.]+)s")
RX_PLAN = re.compile(r"planned in ([\d.]+)s")
RX_STRAIGHT = re.compile(r"straight ramp of (\d+) steps .* checked in ([\d.]+)s")
RX_STRAIGHT_NO = re.compile(r"straight ramp not clear \((\w+)\)")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt(v, d=1):
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def scan_job(job):
    out = {"attempts": [], "rejected": [], "written": [], "stages": [], "kit_lines": [], "spa": [], "world": [], "plans": [], "straight_ok": [], "straight_no": {},
           "inst_load": [], "demos": [], "phase_s": [], "kit": [], "spa_draw": [], "gpu_peak": None}
    p = os.path.join(job, "sample.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            m = RX_REJ.search(line)
            if m:
                out["attempts"].append(float(m.group(5)))
                out["rejected"].append((m.group(4), m.group(6).split(",")[0].split(":")[0].strip(), float(m.group(5))))
                continue
            m = RX_WRITTEN.search(line)
            if m:
                out["attempts"].append(float(m.group(4)))
                out["written"].append(float(m.group(4)))
                continue
            m = RX_STAGES.search(line)
            if m:
                d = {}
                for part in m.group(3).split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            d[k] = float(v.rstrip("s"))
                        except ValueError:
                            pass
                out["stages"].append(d)
                continue
            m = RX_KIT.search(line)
            if m:
                out["kit_lines"].append(m.group(3))
                continue
            m = RX_SPA.search(line)
            if m:
                out["spa"].append((m.group(1), float(m.group(2))))
                continue
            m = RX_WORLD.search(line)
            if m:
                out["world"].append(float(m.group(1)))
    for s in glob.glob(os.path.join(job, "inst", "*", "*", "instances", "*.summary.json")):
        try:
            d = json.load(open(s))
        except Exception:
            continue
        if isinstance(d.get("phase_s"), dict):
            out["phase_s"].append(d["phase_s"])
        if isinstance(d.get("kit"), dict):
            out["kit"].append(d["kit"])
        spa = d.get("start_per_attempt") or {}
        if isinstance(spa, dict) and spa.get("draw_s") is not None:
            out["spa_draw"].append(float(spa["draw_s"]))
    for c in glob.glob(os.path.join(job, "collect_c*_q*.log")) + glob.glob(os.path.join(job, "demo_q*.log")):
        for line in open(c, errors="replace"):
            m = RX_PLAN.search(line)
            if m:
                out["plans"].append(float(m.group(1)))
            m = RX_STRAIGHT.search(line)
            if m:
                out["straight_ok"].append(float(m.group(2)))
            m = RX_STRAIGHT_NO.search(line)
            if m:
                out["straight_no"][m.group(1)] = out["straight_no"].get(m.group(1), 0) + 1
            m = RX_INSTLOAD.search(line)
            if m:
                out["inst_load"].append(float(m.group(1)))
    p = os.path.join(job, "results.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out["demos"].append({"success": bool(r.get("success")), "wall_s": r.get("wall_s"), "n_steps": r.get("n_steps")})
    p = os.path.join(job, "gpu.log")
    if os.path.exists(p):
        peak = 0
        for line in open(p):
            m = re.search(r"^\d+ (\d+)MiB", line)
            if m:
                peak = max(peak, int(m.group(1)))
        out["gpu_peak"] = peak or None
    return out


def agg(jobs):
    A = {k: [] for k in ("attempts", "written", "world", "plans", "straight_ok", "inst_load", "spa_draw")}
    rej_stage, rej_reason, straight_no, demos, phases, kits, stages, spa = {}, {}, {}, [], [], [], [], {"hit": [], "miss": [], "memo": []}
    for j in jobs:
        for k in A:
            A[k] += j[k]
        for stg, why, s in j["rejected"]:
            rej_stage[stg] = rej_stage.get(stg, 0) + 1
            rej_reason[why] = rej_reason.get(why, 0) + 1
        for k, v in j["straight_no"].items():
            straight_no[k] = straight_no.get(k, 0) + v
        demos += j["demos"]
        phases += j["phase_s"]
        kits += j["kit"]
        stages += j["stages"]
        for kind, s in j["spa"]:
            spa.setdefault(kind, []).append(s)
    succ = [d for d in demos if d["success"]]
    phase_mean = {}
    for ph in phases:
        for k, v in ph.items():
            if isinstance(v, (int, float)):
                phase_mean.setdefault(k, []).append(float(v))
    kit_mean = {}
    for kt in kits:
        for k, v in kt.items():
            if isinstance(v, (int, float)) and (k.endswith("_s") or k.endswith("_calls")):
                kit_mean.setdefault(k, []).append(float(v))
    stage_mean = {}
    for sg in stages:
        for k, v in sg.items():
            stage_mean.setdefault(k, []).append(v)
    return {
        "jobs": len(jobs), "attempts": len(A["attempts"]), "s_per_attempt": mean(A["attempts"]), "written": len(A["written"]),
        "att_per_written": (len(A["attempts"]) / len(A["written"])) if A["written"] else None, "s_per_written_attempt": mean(A["written"]),
        "world_load_s": mean(A["world"]), "rej_stage": dict(sorted(rej_stage.items(), key=lambda kv: -kv[1])),
        "rej_reason": dict(sorted(rej_reason.items(), key=lambda kv: -kv[1])[:8]),
        "phase_mean": {k: round(mean(v), 1) for k, v in phase_mean.items()}, "kit_mean": {k: round(mean(v), 2) for k, v in sorted(kit_mean.items())},
        "stage_mean": {k: round(mean(v), 1) for k, v in stage_mean.items()}, "n_stage_lines": len(stages),
        "spa_draw_s": mean(A["spa_draw"]), "spa": {k: (len(v), round(mean(v), 2)) for k, v in spa.items() if v},
        "demos": len(demos), "successes": len(succ), "demo_wall_s": mean([d["wall_s"] for d in succ]), "demo_steps": mean([d["n_steps"] for d in succ]),
        "inst_load_s": mean(A["inst_load"]), "plan_s": mean(A["plans"]), "n_plans": len(A["plans"]),
        "straight_ok": len(A["straight_ok"]), "straight_no": straight_no,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=None)
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    tags = a.tag or sorted(os.path.basename(d) for d in glob.glob(os.path.join(F, "out", "*")))
    for tag in tags:
        jobs = []
        for job in sorted(glob.glob(os.path.join(F, "out", tag, "*", "g*"))):
            if a.key and a.key not in job:
                continue
            jobs.append(scan_job(job))
        if not jobs:
            continue
        g = agg(jobs)
        print(f"\n## {tag}: {g['jobs']} jobs, {g['attempts']} attempts ({fmt(g['s_per_attempt'], 0)} s each), {g['written']} written "
              f"({fmt(g['att_per_written'])} attempts per written, written attempt {fmt(g['s_per_written_attempt'], 0)} s), world load {fmt(g['world_load_s'], 0)} s")
        print(f"  rejections by stage {g['rej_stage']}")
        print(f"  by reason {g['rej_reason']}")
        print(f"  written instances phase_s mean {g['phase_mean']}")
        print(f"  written instances kit mean {g['kit_mean']}")
        if g["n_stage_lines"]:
            print(f"  per-attempt stages mean over {g['n_stage_lines']} attempts {g['stage_mean']}")
        print(f"  SPA draw_s mean {fmt(g['spa_draw_s'])}; cache {g['spa']}")
        print(f"  demos {g['demos']} ({g['successes']} success), wall {fmt(g['demo_wall_s'], 0)} s, steps {fmt(g['demo_steps'], 0)}, "
              f"collector instance load {fmt(g['inst_load_s'], 0)} s, trajopt plans {g['n_plans']} x {fmt(g['plan_s'], 1)} s, straight ramps ok {g['straight_ok']} / not clear {g['straight_no']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
