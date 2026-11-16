#!/usr/bin/env python3
"""Summary of a policy-eval run: per checkpoint, success and the soft QA counts per instance, per task class (the training alias),
per task, per task family and in total (micro: every instance counts once; macro over classes / families reported alongside).

    summarize.py --run-dir $EV/runs/<run> [--done <dir of done/*.json>] [--out <dir>]

Reads ``spec.json`` (the units) and the workers' ``done/<unit>.json`` records (``eval_worker.py``; pulled from R2 into
``<run-dir>/done/`` by default). Per instance: success, evaluated step / kind / settled flag, steps executed, the error metric at the
evaluated step, the closest approach, and for every QA metric of the stream (``meta.qa.metrics``: logged, never failing a policy
episode) its violation steps and its runs (separate violation episodes) and whether it triggered. Aggregates: success rate; per QA
metric the mean violation steps, the mean runs and the share of instances where it triggered. Writes ``summary_<ckpt>.json``
(instances + every level), ``summary.md`` (tables) and ``instances.csv``; missing / failed units are listed.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics


#: QA metrics counted by their violating steps (``steps``; ``runs`` = separate stretches of them)
STEP_METRICS = ("robot_touching_objects", "robot_touching_itself", "arm_hand_contacts", "held_object_drift", "stalled")


def violations(name, m):
    """(violation count, separate runs) of one ``meta.qa.metrics`` record (run_isg --qa: logged, never failing a policy episode).
    Step metrics: violating steps; ``pause`` / ``goal_occlusion``: 1 when it fired; ``jerky_motion``: limits exceeded; ``ghost_hand`` /
    ``bad_release`` / ``arm_stutter``: events; ``head_camera_upright``: tilted steps while driving; ``robot_wobble``: steps over the
    limit; others: 1 when triggered."""
    if name in STEP_METRICS:
        return int(m.get("steps") or 0), len(m.get("runs") or [])
    if name == "jerky_motion":
        return len(m.get("failed") or []), 0
    if name == "ghost_hand":
        return len(m.get("events") or []), len(m.get("events") or [])
    if name == "bad_release":
        return len(m.get("violations") or []), len(m.get("violations") or [])
    if name == "arm_stutter":
        return len(m.get("stutters") or []), len(m.get("stutters") or [])
    if name == "head_camera_upright":
        return int(m.get("steps_tilted_while_navigating") or 0), 0
    if name == "robot_wobble":
        o = m.get("over_limit") or {}
        return int(o.get("steps") or 0), len(o.get("runs") or [])
    if name == "goal_occlusion":
        return int(bool((m.get("report") or {}).get("occluded"))), 0
    return int(bool(m.get("triggered"))), 0


def qa_counts(meta):
    out = {}
    for name, m in ((meta.get("qa") or {}).get("metrics") or {}).items():
        if not isinstance(m, dict) or m.get("measured") is False:  # oracle-only rules (gripper_in_view, failed_grasp) in a stream
            continue
        v, runs = violations(name, m)
        out[name] = {"steps": v, "runs": runs, "triggered": bool(m.get("triggered"))}
    return out


def instance_row(u, d):
    r, m = d.get("result") or {}, d.get("meta") or {}
    em = m.get("error_metric") or {}
    pc = m.get("policy_client") or {}
    return {"unit": u["unit"], "index": u["index"], "class_id": u["class_id"], "task_family": u["task_family"], "subtask": u["subtask"],
            "task": m.get("task") or r.get("task"), "success": bool(r.get("success")), "status": r.get("status"), "reason": r.get("reason"),
            "n_steps": r.get("n_steps") or m.get("n_steps"), "budget": u["budget"], "evaluation_step": m.get("evaluation_step"),
            "evaluation_kind": m.get("evaluation_kind"), "evaluation_settled": m.get("evaluation_settled"),
            "position_error": em.get("position_error"), "orientation_error": em.get("orientation_error"),
            "closest_score": (m.get("closest_approach") or {}).get("score"), "goal_kind": pc.get("goal_kind"),
            "cross_episode_goal": pc.get("cross_episode_goal"), "qa": qa_counts(m), "seconds": d.get("seconds"),
            "load_s": (m.get("timings") or {}).get("load_s"), "episode_s": (m.get("timings") or {}).get("episode_s"),
            "gpu": d.get("gpu_name"), "request_s_mean": pc.get("request_s_mean")}


def aggregate(rows, metrics):
    n = len(rows)
    out = {"n": n, "success": sum(r["success"] for r in rows), "success_rate": (sum(r["success"] for r in rows) / n) if n else None}
    out["qa"] = {}
    for q in metrics:
        vals = [r["qa"].get(q, {"steps": 0, "runs": 0, "triggered": False}) for r in rows]
        out["qa"][q] = {"mean_steps": statistics.fmean(v["steps"] for v in vals) if vals else None,
                        "mean_runs": statistics.fmean(v["runs"] for v in vals) if vals else None,
                        "triggered_rate": statistics.fmean(v["triggered"] for v in vals) if vals else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--done", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(a.run_dir, "spec.json")))
    done_dir = a.done or os.path.join(a.run_dir, "done")
    done = {}  # (ckpt, dataset index) -> {"result", "meta", worker fields}; a unit record holds one instance or a world chunk's list
    for p in glob.glob(os.path.join(done_dir, "*.json")):
        d = json.load(open(p))
        for x in (d.get("instances") or [{"index": d.get("index"), "result": d.get("result"), "meta": d.get("meta")}]):
            done[(d["ckpt"], int(x["index"]))] = {**{k: v for k, v in d.items() if k != "instances"}, **x}
    out_dir = a.out or a.run_dir
    by_ckpt = collections.defaultdict(list)  # ckpt -> its instances (a unit's instances, or the unit itself)
    for u in spec["units"]:
        for x in (u.get("instances") or [u]):
            by_ckpt[u["ckpt"]].append({**x, "unit": f"{u['ckpt']}__{int(x['index']):04d}", "ckpt": u["ckpt"]})
    md = [f"# Policy eval {spec['run']}\n", f"run_isg args: `{' '.join(spec['run_isg_args'])}`\n"]
    all_rows = []
    for tag, units in by_ckpt.items():
        rows, missing = [], []
        for u in units:
            if (tag, int(u["index"])) in done:
                rows.append(instance_row(u, done[(tag, int(u["index"]))]))
            else:
                missing.append(u["unit"])
        metrics = sorted({q for r in rows for q in r["qa"]})
        levels = {}
        for level in ("class_id", "task", "task_family"):
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r[level]].append(r)
            levels[level] = {k: aggregate(v, metrics) for k, v in sorted(groups.items())}
        total = aggregate(rows, metrics)
        macro = {lvl: (statistics.fmean(g["success_rate"] for g in levels[lvl].values()) if levels[lvl] else None) for lvl in ("class_id", "task_family")}
        summ = {"run": spec["run"], "ckpt": tag, "ckpt_spec": spec["ckpts"][tag], "units": len(units), "done": len(rows), "missing": missing,
                "total": total, "macro_success_rate": macro, "levels": levels, "instances": rows}
        json.dump(summ, open(os.path.join(out_dir, f"summary_{tag}.json"), "w"), indent=1)
        all_rows += [{**r, "ckpt": tag} for r in rows]
        sr = total["success_rate"]
        head = f"## {tag}\n\n{len(rows)}/{len(units)} instances evaluated" + (f" ({len(missing)} missing)" if missing else "")
        if sr is not None:
            head += (f"; success {total['success']}/{total['n']} = {100 * sr:.1f}% (micro); macro over classes {100 * macro['class_id']:.1f}%, "
                     f"over families {100 * macro['task_family']:.1f}%")
        md.append(head + "\n")
        if not rows:
            continue
        md.append("| family | n | success | " + " | ".join(f"{q} (violations / runs / fired)" for q in metrics) + " |")
        md.append("|---|---|---|" + "---|" * len(metrics))
        for fam, g in list(levels["task_family"].items()) + [("**total**", total)]:
            md.append(f"| {fam} | {g['n']} | {100 * g['success_rate']:.1f}% | "
                      + " | ".join(f"{g['qa'][q]['mean_steps']:.1f} / {g['qa'][q]['mean_runs']:.2f} / {100 * g['qa'][q]['triggered_rate']:.0f}%" for q in metrics) + " |")
        kinds = collections.Counter((r["goal_kind"], r["cross_episode_goal"]) for r in rows)
        if any(k[0] and k[0] != "own" for k in kinds):  # run_isg --cross-goal: success per goal kind / difficulty
            md.append("\n| goal image | n | success |\n|---|---|---|")
            for (kind, diff), n in sorted(kinds.items(), key=lambda kv: str(kv[0])):
                g = [r for r in rows if (r["goal_kind"], r["cross_episode_goal"]) == (kind, diff)]
                md.append(f"| {kind}{' / ' + diff if diff else ''} | {n} | {100 * sum(r['success'] for r in g) / n:.1f}% |")
            summ["by_goal_kind"] = {f"{k}/{d}": {"n": n, "success": sum(r["success"] for r in rows if (r["goal_kind"], r["cross_episode_goal"]) == (k, d))}
                                    for (k, d), n in kinds.items()}
            json.dump(summ, open(os.path.join(out_dir, f"summary_{tag}.json"), "w"), indent=1)
        md.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    if all_rows:
        keys = [k for k in all_rows[0] if k != "qa"]
        qkeys = sorted({q for r in all_rows for q in r["qa"]})
        with open(os.path.join(out_dir, "instances.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ckpt"] + [k for k in keys if k != "ckpt"] + [f"qa_{q}_steps" for q in qkeys] + [f"qa_{q}_runs" for q in qkeys])
            for r in all_rows:
                w.writerow([r["ckpt"]] + [r.get(k) for k in keys if k != "ckpt"] + [r["qa"].get(q, {}).get("steps", 0) for q in qkeys]
                           + [r["qa"].get(q, {}).get("runs", 0) for q in qkeys])
    print("\n".join(md))


if __name__ == "__main__":
    main()
