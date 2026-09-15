#!/usr/bin/env python3
"""Assemble the representative run (G0): sampled instances + run_isg demos pulled from the ``samp`` pods.

    python scripts/isg_reps/assemble_results.py [--src runlogs/runpod/samp] [--out runlogs/isg_reps_20260915]

Reads every collected pod directory (``<src>/<pod name>/<job id>/...`` from ``collect.py``; falls back to
``<src>/live/<pod>`` for jobs not collected yet) and writes into ``<out>``:

* ``instances.csv`` - one row per written instance: subtask, alias, pick, slot, job id, pod, instance id, template id,
  scene, path (relative to ``<src>``), demo status / reason / errors when a run_isg run exists;
* ``jobs.csv`` - one row per job: subtask, alias, job id, pod, rc, wall s, n_written, rejections (JSON), error;
* ``demos.csv`` - one row per run_isg run (meta.json): instance, status, success, reason, steps, wall, video path;
* ``summary.md`` - per-subtask table (jobs, jobs with instances, instances, demos success / QA-failed / no_oracle,
  top rejections) + the list of picks without any instance (with the rejection reasons).
The heavy artefacts (TROs, hdf5, mp4) stay under ``<src>`` (git-ignored); this directory holds the index.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JOBS = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_reps", "jobs.json")


def load_jobs():
    jobs = {j["id"]: j for j in json.load(open(JOBS))}
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "runlogs", "runpod", "samp"))
    ap.add_argument("--out", default=os.path.join(REPO, "runlogs", "isg_reps_20260915"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = load_jobs()
    pod_dirs = [d for d in sorted(glob.glob(os.path.join(args.src, "*"))) if os.path.isdir(d) and os.path.basename(d) not in ("live",)]
    pod_dirs += [d for d in sorted(glob.glob(os.path.join(args.src, "live", "*"))) if os.path.isdir(d)]
    job_rows, inst_rows, demo_rows = {}, {}, {}
    for pod_dir in pod_dirs:
        pod = os.path.basename(pod_dir)
        dpath = os.path.join(pod_dir, "done.txt")
        done_lines = open(dpath).read().splitlines() if os.path.exists(dpath) else []
        for line in done_lines:
            m = re.match(r"(\S+) rc=(-?\d+) wall=(\d+) n_written=(-?\d+)", line.strip())
            if not m or m.group(1).startswith("demo__"):
                continue
            jid, rc, wall, nw = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            j = jobs.get(jid, {})
            rej, err = {}, ""
            for rep in glob.glob(os.path.join(pod_dir, jid, "sampling_report_*.json")):
                try:
                    d = json.load(open(rep))
                except Exception:
                    continue
                for prob in d.get("problems", {}).values():
                    for t in prob.values():
                        if isinstance(t, dict):
                            for k, v in (t.get("rejections") or {}).items():
                                rej[k] = rej.get(k, 0) + int(v)
                            if t.get("error"):
                                err = str(t["error"])[:300]
            prev = job_rows.get(jid)
            if prev is None or nw > int(prev["n_written"]):
                job_rows[jid] = {"subtask": j.get("subtask", ""), "alias": j.get("alias", ""), "job": jid, "pod": pod, "rc": rc, "wall_s": wall,
                                 "n_written": nw, "rejections": json.dumps(rej, sort_keys=True), "error": err}
            for tro in glob.glob(os.path.join(pod_dir, jid, "*", "*", "instances", "*-tro_state.json")):
                iid = os.path.basename(tro)[: -len("-tro_state.json")]
                tdir = os.path.dirname(os.path.dirname(tro))
                inst_rows[iid] = {"subtask": j.get("subtask", ""), "alias": j.get("alias", ""), "pick": "", "slot": j.get("slot", ""), "job": jid, "pod": pod,
                                  "instance": iid, "template": os.path.basename(tdir), "scene": os.path.basename(tdir).split("__")[0],
                                  "path": os.path.relpath(tro, args.src), "demo_status": "", "demo_reason": "", "demo_success": "", "demo_errors": ""}
        for meta in glob.glob(os.path.join(pod_dir, "demo__*", "*", "meta.json")) + glob.glob(os.path.join(pod_dir, "*", "*", "*", "runs", "*", "meta.json")):
            try:
                m = json.load(open(meta))
            except Exception:
                continue
            iid = m.get("instance_id") or m.get("instance") or os.path.basename(os.path.dirname(meta))
            video = os.path.join(os.path.dirname(meta), "head_rgb.mp4")
            demo_rows[iid] = {"instance": iid, "status": m.get("status", ""), "success": m.get("success", ""), "reason": m.get("reason") or m.get("qa_reason", ""),
                              "n_steps": m.get("n_steps", ""), "wall_s": m.get("wall_time_s", ""), "oracle": m.get("oracle", ""),
                              "video": os.path.relpath(video, args.src) if os.path.exists(video) else "", "meta": os.path.relpath(meta, args.src)}
            if iid in inst_rows:
                inst_rows[iid].update({"demo_status": m.get("status", ""), "demo_reason": m.get("reason") or "", "demo_success": m.get("success", ""),
                                       "demo_errors": json.dumps({k: v for k, v in m.items() if k.startswith("max_") or k.startswith("final_step")})})
    # picks
    problems = {p["id"]: p for p in json.load(open(os.path.join(os.path.dirname(JOBS), "problems.json")))}
    for r in inst_rows.values():
        r["pick"] = problems.get(r["job"], {}).get("pick", "")

    def write(name, rows, cols):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    write("jobs.csv", sorted(job_rows.values(), key=lambda r: (r["subtask"], r["job"])), ["subtask", "alias", "job", "pod", "rc", "wall_s", "n_written", "rejections", "error"])
    write("instances.csv", sorted(inst_rows.values(), key=lambda r: (r["subtask"], r["instance"])),
          ["subtask", "alias", "pick", "slot", "job", "pod", "instance", "template", "scene", "path", "demo_status", "demo_success", "demo_reason", "demo_errors"])
    write("demos.csv", sorted(demo_rows.values(), key=lambda r: r["instance"]), ["instance", "status", "success", "reason", "n_steps", "wall_s", "oracle", "video", "meta"])
    # summary
    by = collections.defaultdict(lambda: collections.Counter())
    rej_by = collections.defaultdict(collections.Counter)
    for jid, j in jobs.items():
        by[j["subtask"]]["jobs"] += 1
    for r in job_rows.values():
        st = r["subtask"]
        by[st]["done"] += 1
        by[st]["with_instances"] += 1 if r["n_written"] > 0 else 0
        by[st]["instances"] += max(0, r["n_written"])
        for k, v in json.loads(r["rejections"]).items():
            rej_by[st][k] += v
        if r["error"]:
            rej_by[st]["ERROR " + r["error"][:50]] += 1
    for r in inst_rows.values():
        st = r["subtask"]
        if r["demo_status"] == "success":
            by[st]["demo_ok"] += 1
        elif r["demo_status"] == "oracle_failed":
            by[st]["demo_qa_fail"] += 1
        elif r["demo_status"] == "no_oracle":
            by[st]["demo_no_oracle"] += 1
        elif r["demo_status"]:
            by[st]["demo_error"] += 1
    lines = ["# Representative run (G0) - results index", "",
             f"Source: `{os.path.relpath(args.src, REPO)}` (collected pods + live pulls). {len(job_rows)} jobs done of {len(jobs)}, "
             f"{len(inst_rows)} instances, {len(demo_rows)} demo runs.", "",
             "| subtask | jobs | done | with >= 1 instance | instances | demos ok | demos QA-failed (binary goal may still hold) | demo error | top rejections |",
             "|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for st in sorted(by):
        b = by[st]
        top = ", ".join(f"{k} {v}" for k, v in rej_by[st].most_common(4))
        lines.append(f"| {st} | {b['jobs']} | {b['done']} | {b['with_instances']} | {b['instances']} | {b['demo_ok']} | {b['demo_qa_fail']} | {b['demo_error']} | {top} |")
        tot.update(b)
    lines.append(f"| **total** | {tot['jobs']} | {tot['done']} | {tot['with_instances']} | {tot['instances']} | {tot['demo_ok']} | {tot['demo_qa_fail']} | {tot['demo_error']} | |")
    lines += ["", "## Picks without an instance", ""]
    for r in sorted(job_rows.values(), key=lambda r: (r["subtask"], r["job"])):
        if r["n_written"] <= 0:
            lines.append(f"- {r['subtask']} `{r['job']}`: rc {r['rc']}, rejections {r['rejections']} {('error: ' + r['error']) if r['error'] else ''}")
    missing = [jid for jid in jobs if jid not in job_rows]
    if missing:
        lines += ["", f"## Jobs not run ({len(missing)})", ""] + [f"- {jobs[j]['subtask']} `{j}`" for j in sorted(missing)]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
