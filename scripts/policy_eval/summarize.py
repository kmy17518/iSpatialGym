#!/usr/bin/env python3
"""Summary of a policy-eval run: per checkpoint, the success verdicts, errors and soft QA counts per instance, per task class (the
training alias), per task, per task family and in total (micro: every instance counts once; macro over classes / families alongside).

    summarize.py --run-dir $EV/runs/<run> [--done <dir of done/*.json>] [--out <dir>]

Reads ``spec.json`` (the units), the workers' ``done/<unit>.json`` records (``eval_worker.py``; pulled into ``<run-dir>/done/``) and,
when present, the runs' full ``meta.json`` files (``collect.sh --outputs``: ``<run-dir>/metas/<ckpt>/<instance id>/meta.json``), which
take precedence over the records' compact copies.

Per instance: the run's verdict (``success``: settled geometric with ``--settle-eval on``), the four verdicts settled / unsettled x
geometric / symbolic (run_isg evaluates in its ``--settle-eval`` mode and reports the other mode alongside, ``other_mode``), each
settle mode's position / orientation error at its evaluated step (the success step, else the last step; geometric and symbolic share
it: the symbolic check has no pose error of its own), steps, the closest approach, the goal image kind, and for every QA metric of the
stream (``meta.qa.metrics``: logged, never failing a policy episode) its violations, separate runs and whether it fired.
Aggregates per level: the four success rates; per settle mode the mean position / orientation error over all instances and over
each verdict's successes; per QA metric the mean violations, mean runs and the share of instances where it fired.
Writes ``summary_<ckpt>.json`` (instances + every level), ``summary.md`` (tables) and ``instances.csv``; missing units are listed.
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
#: the four verdicts (settle mode, success kind)
VERDICTS = (("settled", "geometric"), ("settled", "symbolic"), ("unsettled", "geometric"), ("unsettled", "symbolic"))
ERRORS = ("position_error", "orientation_error")


def violations(name, m):
    """(violation count, separate runs) of one ``meta.qa.metrics`` record. Step metrics: violating steps; ``pause`` /
    ``goal_occlusion``: 1 when it fired; ``jerky_motion``: limits exceeded; ``ghost_hand`` / ``bad_release`` / ``arm_stutter``: events;
    ``head_camera_upright``: tilted steps while driving; ``robot_wobble``: steps over the limit; others: 1 when triggered."""
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


def verdicts(m):
    """The four verdicts and the two settle modes' errors of a run's meta.json (None where not recorded, e.g. a compact record
    without ``other_mode``)."""
    main = {k: m.get(k) for k in ("geometric_success", "symbolic_success", "error_metric")}
    other = m.get("other_mode") or {}
    settled_main = bool(m.get("settle_eval", True))
    modes = {"settled": main if settled_main else other, "unsettled": other if settled_main else main}
    out = {}
    for mode, rec in modes.items():
        for kind in ("geometric", "symbolic"):
            v = rec.get(f"{kind}_success")
            out[f"{mode}_{kind}"] = None if v is None else bool(v)
        em = rec.get("error_metric") or {}
        for e in ERRORS:
            out[f"{mode}_{e}"] = em.get(e)
    return out


def instance_row(u, d):
    r, m = d.get("result") or {}, d.get("meta") or {}
    pc = m.get("policy_client") or {}
    return {"unit": u["unit"], "index": u["index"], "class_id": u["class_id"], "task_family": u["task_family"], "subtask": u["subtask"],
            "task": m.get("task") or r.get("task"), "success": bool(r.get("success")), "status": r.get("status"), "reason": r.get("reason"),
            "n_steps": r.get("n_steps") or m.get("n_steps"), "budget": u["budget"], "evaluation_step": m.get("evaluation_step"),
            "evaluation_kind": m.get("evaluation_kind"), "evaluation_settled": m.get("evaluation_settled"), **verdicts(m),
            "closest_score": (m.get("closest_approach") or {}).get("score"), "goal_kind": pc.get("goal_kind"),
            "cross_episode_goal": pc.get("cross_episode_goal"), "qa": qa_counts(m), "seconds": d.get("seconds"),
            "load_s": (m.get("timings") or {}).get("load_s"), "episode_s": (m.get("timings") or {}).get("episode_s"),
            "gpu": d.get("gpu_name"), "request_s_mean": pc.get("request_s_mean")}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def aggregate(rows, metrics):
    n = len(rows)
    out = {"n": n, "success": sum(r["success"] for r in rows), "success_rate": (sum(r["success"] for r in rows) / n) if n else None}
    for mode, kind in VERDICTS:
        vals = [r[f"{mode}_{kind}"] for r in rows if r.get(f"{mode}_{kind}") is not None]
        out[f"{mode}_{kind}_rate"] = statistics.fmean(vals) if vals else None
        out[f"{mode}_{kind}_n"] = len(vals)
        for e in ERRORS:
            out[f"{mode}_{kind}_{e}_successes"] = mean(r.get(f"{mode}_{e}") for r in rows if r.get(f"{mode}_{kind}"))
    for mode in ("settled", "unsettled"):
        for e in ERRORS:
            out[f"{mode}_{e}_all"] = mean(r.get(f"{mode}_{e}") for r in rows)
    out["qa"] = {}
    for q in metrics:
        vals = [r["qa"].get(q, {"steps": 0, "runs": 0, "triggered": False}) for r in rows]
        out["qa"][q] = {"mean_steps": statistics.fmean(v["steps"] for v in vals) if vals else None,
                        "mean_runs": statistics.fmean(v["runs"] for v in vals) if vals else None,
                        "triggered_rate": statistics.fmean(v["triggered"] for v in vals) if vals else None}
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def num(x):
    return "-" if x is None else f"{x:.3f}"


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
    full_meta = {}  # collect.sh --outputs: the runs' full meta.json files
    for p in glob.glob(os.path.join(a.run_dir, "metas", "*", "*", "meta.json")):
        parts = p.split(os.sep)
        full_meta[(parts[-3], parts[-2])] = p
    out_dir = a.out or a.run_dir
    by_ckpt = collections.defaultdict(list)  # ckpt -> its instances (a unit's instances, or the unit itself)
    for u in spec["units"]:
        for x in (u.get("instances") or [u]):
            by_ckpt[u["ckpt"]].append({**x, "unit": f"{u['ckpt']}__{int(x['index']):04d}", "ckpt": u["ckpt"]})
    md = [f"# Policy eval {spec['run']}\n", f"run_isg args: `{' '.join(spec['run_isg_args'])}`\n",
          "Verdicts: settled = the success held over the 45-step settle window (the eval verdict with --settle-eval on); unsettled = the "
          "success reached at all (its lowest-error step of the window after the first success). Errors: each settle mode's error at its "
          "evaluated step (the success step, else the last step), mean over all instances / over that verdict's successes; position m, "
          "orientation rad (only tasks that measure it).\n"]
    all_rows = []
    for tag, units in by_ckpt.items():
        rows, missing = [], []
        for u in units:
            key = (tag, int(u["index"]))
            if key not in done:
                missing.append(u["unit"])
                continue
            d = done[key]
            iid = os.path.basename(u["tro"])[: -len("-tro_state.json")]
            if (tag, iid) in full_meta:
                d = {**d, "meta": json.load(open(full_meta[(tag, iid)]))}
            rows.append(instance_row(u, d))
        metrics = sorted({q for r in rows for q in r["qa"]})
        levels = {}
        for level in ("class_id", "task", "task_family"):
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r[level]].append(r)
            levels[level] = {k: aggregate(v, metrics) for k, v in sorted(groups.items())}
        total = aggregate(rows, metrics)
        macro = {lvl: {f"{m}_{k}": mean(g[f"{m}_{k}_rate"] for g in levels[lvl].values()) for m, k in VERDICTS} for lvl in ("class_id", "task_family")}
        summ = {"run": spec["run"], "ckpt": tag, "ckpt_spec": spec["ckpts"][tag], "units": len(units), "done": len(rows), "missing": missing,
                "total": total, "macro_success_rate": macro, "levels": levels, "instances": rows}
        all_rows += [{**r, "ckpt": tag} for r in rows]
        md.append(f"## {tag}\n\n{len(rows)}/{len(units)} instances evaluated" + (f" ({len(missing)} missing)" if missing else "")
                  + (f"; macro over classes: settled geometric {pct(macro['class_id']['settled_geometric'])}, unsettled geometric "
                     f"{pct(macro['class_id']['unsettled_geometric'])}" if rows else "") + "\n")
        if rows:
            md.append("| family | n | settled geom | settled symb | unsettled geom | unsettled symb | pos err settled: all / geom succ / symb succ | "
                      "pos err unsettled: all / geom succ / symb succ | ori err settled: all / geom succ / symb succ | "
                      "ori err unsettled: all / geom succ / symb succ |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for fam, g in list(levels["task_family"].items()) + [("**total**", total)]:
                errs = [" / ".join([num(g[f"{m}_{e}_all"]), num(g[f"{m}_geometric_{e}_successes"]), num(g[f"{m}_symbolic_{e}_successes"])])
                        for e in ERRORS for m in ("settled", "unsettled")]
                md.append(f"| {fam} | {g['n']} | " + " | ".join(pct(g[f"{m}_{k}_rate"]) for m, k in VERDICTS) + " | " + " | ".join(errs) + " |")
            md.append("\nSoft QA (logged, never failing an episode): mean violations / mean separate runs / share of instances where it fired\n")
            md.append("| family | n | " + " | ".join(metrics) + " |")
            md.append("|---|---|" + "---|" * len(metrics))
            for fam, g in list(levels["task_family"].items()) + [("**total**", total)]:
                md.append(f"| {fam} | {g['n']} | " + " | ".join(
                    f"{g['qa'][q]['mean_steps']:.1f} / {g['qa'][q]['mean_runs']:.2f} / {100 * g['qa'][q]['triggered_rate']:.0f}%" for q in metrics) + " |")
            kinds = collections.Counter((r["goal_kind"], r["cross_episode_goal"]) for r in rows)
            if any(k[0] and k[0] != "own" for k in kinds):  # run_isg --cross-goal: the verdicts per goal kind / difficulty
                md.append("\n| goal image | n | settled geom | unsettled geom |\n|---|---|---|---|")
                summ["by_goal_kind"] = {}
                for (kind, diff), n in sorted(kinds.items(), key=lambda kv: str(kv[0])):
                    g = aggregate([r for r in rows if (r["goal_kind"], r["cross_episode_goal"]) == (kind, diff)], [])
                    md.append(f"| {kind}{' / ' + diff if diff else ''} | {n} | {pct(g['settled_geometric_rate'])} | {pct(g['unsettled_geometric_rate'])} |")
                    summ["by_goal_kind"][f"{kind}/{diff}"] = {k: v for k, v in g.items() if k != "qa"}
        json.dump(summ, open(os.path.join(out_dir, f"summary_{tag}.json"), "w"), indent=1)
        md.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    if all_rows:
        keys = [k for k in all_rows[0] if k != "qa"]
        qkeys = sorted({q for r in all_rows for q in r["qa"]})
        with open(os.path.join(out_dir, "instances.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ckpt"] + [k for k in keys if k != "ckpt"] + [f"qa_{q}_violations" for q in qkeys] + [f"qa_{q}_runs" for q in qkeys])
            for r in all_rows:
                w.writerow([r["ckpt"]] + [r.get(k) for k in keys if k != "ckpt"] + [r["qa"].get(q, {}).get("steps", 0) for q in qkeys]
                           + [r["qa"].get(q, {}).get("runs", 0) for q in qkeys])
    print("\n".join(md))


if __name__ == "__main__":
    main()
