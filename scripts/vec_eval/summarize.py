"""Aggregate the fleet's results: ``<root>/out/<unit>/<tag>/results.jsonl`` (a local mirror of the run prefix) into per-run,
per-checkpoint success rates on the val instances and on the eval-dataset instances, per task and overall.

    python scripts/vec_eval/summarize.py --root /vision/u/minyeong/projects/isg_home/vec_eval85/mirror --out /vision/u/minyeong/projects/isg_home/vec_eval85/summary
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", default="/vision/u/minyeong/projects/isg_home/vec_eval85/instances/all",
                    help="per-task instance lists (demo lengths): every verdict is normalized to the 1.5x step budget")
    ap.add_argument("--budget-factor", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # demo lengths: the plan's budget is max(300, round(1.5 x length)); an episode run under the earlier 2x budget counts as a
    # success only if its (settled) success step lies within the 1.5x budget -- the trajectory up to that step is the same
    lengths = {}
    for f in glob.glob(os.path.join(args.instances, "*.json")):
        for spec in json.load(open(f)).get("instances", []):
            if spec.get("name") and spec.get("length"):
                lengths[spec["name"]] = int(spec["length"])
    normalized = {"same": 0, "flipped": 0, "unknown_length": 0}

    def normalize(r):
        L = lengths.get(r.get("instance"))
        if L is None:
            normalized["unknown_length"] += 1
            return bool(r.get("success"))
        b = max(300, int(round(args.budget_factor * L)))
        r["budget_1p5"] = b
        if not r.get("success") or (r.get("max_steps") or 0) <= b:
            normalized["same"] += 1
            return bool(r.get("success"))
        step = r.get("geometric_success_step") if r.get("success_metric") == "geometric" else r.get("symbolic_success_step")
        ok = step is not None and int(step) <= b
        normalized["same" if ok else "flipped"] += 1
        return ok

    # the episodes come from their own meta.json files (authoritative: a results.jsonl can lose rows to a resume that re-ran
    # part of a tag, an error episode may be missing from it, two workers may both have appended rows); results.jsonl rows
    # fill in the fields a meta.json lacks
    rows = []
    for tag_dir in sorted(glob.glob(os.path.join(args.root, "out", "*", "step-*"))):
        unit, tag = tag_dir.split(os.sep)[-2], tag_dir.split(os.sep)[-1]
        if "__" not in unit:
            continue
        run, task = unit.split("__", 1)
        step = int(tag.split("-")[-1])
        by_inst = {}
        rp = os.path.join(tag_dir, "results.jsonl")
        if os.path.exists(rp):
            for line in open(rp):
                if line.strip():
                    row = json.loads(line)
                    by_inst[row.get("instance")] = row
        for mp in glob.glob(os.path.join(tag_dir, "*", "meta.json")):
            name = mp.split(os.sep)[-2]
            if name == "_batches":
                continue
            try:
                m = json.load(open(mp))
            except Exception:  # noqa: BLE001
                m = {"status": "error", "unreadable": True}
            budget = m.get("budget") if isinstance(m.get("budget"), dict) else {}
            r = dict(by_inst.get(name) or {})
            r.update({"instance": name, "episode_index": m.get("episode_index"), "status": m.get("status"), "success": m.get("success"),
                      "reason": m.get("reason"), "success_metric": m.get("success_metric", r.get("success_metric")),
                      "symbolic_success_step": m.get("symbolic_success_step"), "geometric_success_step": m.get("geometric_success_step"),
                      "symbolic_first_step": m.get("symbolic_first_step"), "geometric_first_step": m.get("geometric_first_step"),
                      "n_steps": m.get("n_steps", r.get("n_steps")), "max_steps": budget.get("max_steps", r.get("max_steps")),
                      "wall_time_s": m.get("wall_time_s", r.get("wall_time_s")), "task_id": m.get("task_id", r.get("task_id"))})
            if r.get("n_steps") is not None and r.get("max_steps"):
                r["wall_clock_exceeded"] = bool(r["status"] != "error" and not r["success"] and int(r["n_steps"]) < int(r["max_steps"]))
            r.update({"unit": unit, "run": run, "task_name": task, "step": step, "split": "eval" if r.get("episode_index") is None else "val"})
            r["success_raw"] = r.get("success")
            r["success"] = normalize(r)
            rows.append(r)
    print(f"{len(rows)} episode rows from {len({r['unit'] for r in rows})} units; budget normalization to {args.budget_factor}x: {normalized}")
    # per run x step x split
    agg = defaultdict(lambda: {"n": 0, "succ": 0, "err": 0, "steps": 0, "geo_steps": [], "truncated": 0, "n_complete": 0, "succ_complete": 0})
    per_task = defaultdict(lambda: {"n": 0, "succ": 0, "err": 0, "truncated": 0})
    for r in rows:
        k = (r["run"], r["step"], r["split"])
        a = agg[k]
        # an episode the vectorized driver cut at run_isg's 1 s/step wall clock before its step budget (a shared GPU takes 1-2 s
        # per batch step): a failure verdict on fewer steps than the budget; rates are also given over the complete episodes
        truncated = bool(r.get("wall_clock_exceeded")) and not r.get("success") and (r.get("n_steps") or 0) < (r.get("max_steps") or 0)
        r["truncated"] = truncated
        a["n"] += 1
        a["succ"] += 1 if r.get("success") else 0
        a["err"] += 1 if r.get("status") == "error" else 0
        a["truncated"] += 1 if truncated else 0
        a["n_complete"] += 0 if truncated else 1
        a["succ_complete"] += 1 if (r.get("success") and not truncated) else 0
        a["steps"] += int(r.get("n_steps") or 0)
        if r.get("success") and r.get("geometric_success_step") is not None:
            a["geo_steps"].append(int(r["geometric_success_step"]))
        t = per_task[(r["run"], r["step"], r["split"], r["task_name"])]
        t["n"] += 1
        t["succ"] += 1 if r.get("success") else 0
        t["err"] += 1 if r.get("status") == "error" else 0
        t["truncated"] += 1 if truncated else 0
    with open(os.path.join(args.out, "by_run_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "split", "episodes", "successes", "success_rate", "errors", "mean_steps", "median_success_step",
                    "truncated", "complete_episodes", "complete_successes", "complete_success_rate"])
        for (run, step, split), a in sorted(agg.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0), kv[0][2])):
            gs = sorted(a["geo_steps"])
            w.writerow([run, step, split, a["n"], a["succ"], round(a["succ"] / a["n"], 4) if a["n"] else None, a["err"], round(a["steps"] / a["n"], 1) if a["n"] else None,
                        gs[len(gs) // 2] if gs else None, a["truncated"], a["n_complete"], a["succ_complete"],
                        round(a["succ_complete"] / a["n_complete"], 4) if a["n_complete"] else None])
    with open(os.path.join(args.out, "by_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "split", "task", "episodes", "successes", "errors", "truncated"])
        for (run, step, split, task), t in sorted(per_task.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0), kv[0][2], kv[0][3])):
            w.writerow([run, step, split, task, t["n"], t["succ"], t["err"], t["truncated"]])
    with open(os.path.join(args.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    # a compact text table: run x step -> val rate / eval rate
    runs = sorted({r["run"] for r in rows})
    lines = []
    for run in runs:
        steps = sorted({r["step"] for r in rows if r["run"] == run and r["step"] is not None}, reverse=True)
        lines.append(f"== {run}")
        for st in steps:
            v, e = agg.get((run, st, "val")), agg.get((run, st, "eval"))
            fv = f"val {v['succ']}/{v['n']} ({100 * v['succ'] / v['n']:.1f}%)" if v and v["n"] else "val -"
            fe = f"eval {e['succ']}/{e['n']} ({100 * e['succ'] / e['n']:.1f}%)" if e and e["n"] else "eval -"
            errs = (v or {}).get("err", 0) + (e or {}).get("err", 0)
            trunc = (v or {}).get("truncated", 0) + (e or {}).get("truncated", 0)
            lines.append(f"  step {st:>7}: {fv:<24} {fe:<22}" + (f" errors {errs}" if errs else "") + (f" truncated {trunc}" if trunc else ""))
    total_trunc = sum(a["truncated"] for a in agg.values())
    lines.append(f"(truncated = cut by the 1 s/step wall clock before the step budget, counted as failures: {total_trunc} of {len(rows)} episodes; "
                 f"by_run_step.csv also gives the rates over the complete episodes)")
    lines.append(f"(success = settled verdict within the {args.budget_factor}x-demo-length budget; episodes run under the earlier 2x budget "
                 f"re-judged by their success step: {normalized['flipped']} flipped, {normalized['same']} unchanged, {normalized['unknown_length']} without a known length)")
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
