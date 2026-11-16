#!/usr/bin/env python3
"""Metrics of a policy-eval results dir: per checkpoint, in total, per task family, per task class and per task.

    summarize.py <results dir> [--out <dir>] [--ckpts a,b] [--instances <eval-instances root>] [--quiet]

A results dir holds ``<ckpt>/<instance id>/`` per evaluated instance (every artifact of its episode: ``meta.json``, ``result.json``,
videos, images; ``vast/collect.sh`` gathers a Vast run into one, a cluster job writes one with ``OUT=<results>/<ckpt>``), and
optionally ``<ckpt>/ckpt.json`` (``collect_results.py``: the checkpoint's spec and the instances each run expected -> the missing ones).

Levels: task family (7), task class (the 23 classes: ``TASK_CLASSES``, = the dataset's ``task``), task (the dataset's
``source.class_id``: one task alias, 10 instances each in the v2 eval set). An instance's identity comes from its ``meta.eval_instance``
(run_isg since 2026-11-16), else ``ckpt.json``'s expected entry, else the dataset index (``--instances``).

Per instance: the run's verdict (``success``: settled geometric with ``--settle-eval on``), the four verdicts settled / unsettled x
geometric / symbolic (run_isg evaluates in its ``--settle-eval`` mode and reports the other mode alongside, ``other_mode``), each settle
mode's position / orientation error at its evaluated step (the success step, else the last step; geometric and symbolic share it: the
symbolic check has no pose error of its own), steps, the goal image kind, and per QA metric of the stream (``meta.qa.metrics``: logged,
never failing a policy episode) its violations, separate runs and whether it fired.
Per group: the four success rates and success counts; per settle mode the mean position / orientation error over all instances and
over each verdict's successes; per QA metric the mean violations, mean runs and fired share over all rollouts and over each verdict's
successful rollouts (with their count).
Writes ``summary.md``, ``summary_<ckpt>.json`` (instances + every group), ``instances.csv``, ``groups.csv`` (rates / counts / errors per
group) and ``qa.csv`` (long: ckpt, level, group, rollouts = all | a verdict's successes, n, metric, violations, runs, fired).
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

#: the task classes (display name per dataset ``task``), in report order; the family of each is its dataset ``task_family``
TASK_CLASSES = {
    "camera_relocalization_standard": "camera relocalization - standard",
    "camera_relocalization_obstructed": "camera relocalization - obstructed",
    "object_scaling": "object scaling",
    "mental_rotation": "mental rotation",
    "disembedding": "disembedding",
    "alignment_vertical": "alignment - world vertical",
    "alignment_axial": "alignment - axial",
    "configuration_matching_block_design": "configuration matching - block design",
    "configuration_matching_attachment": "configuration matching - attachment",
    "configuration_matching_articulation_close_small_scale": "configuration matching - articulation close small scale",
    "configuration_matching_articulation_open_small_scale": "configuration matching - articulation open small scale",
    "configuration_matching_articulation_close_large_scale": "configuration matching - articulation close large scale",
    "configuration_matching_articulation_open_large_scale": "configuration matching - articulation open large scale",
    **{f"inter_object_relation_satisfaction_{rel}_{scale}_scale": f"inter object relation satisfaction - {rel} {scale} scale"
       for scale in ("small", "large") for rel in ("ontop", "under", "inside", "nextto", "touching")},
}
FAMILIES = ("camera_relocalization", "object_scaling", "mental_rotation", "disembedding", "alignment", "configuration_matching",
            "inter_object_relation_satisfaction")
LEVELS = (("task_family", "family"), ("task_class", "task class"), ("class_id", "task"))
#: QA metrics counted by their violating steps (``steps``; ``runs`` = separate stretches of them)
STEP_METRICS = ("robot_touching_objects", "robot_touching_itself", "arm_hand_contacts", "held_object_drift", "stalled")
#: the four verdicts (settle mode, success kind)
VERDICTS = (("settled", "geometric"), ("settled", "symbolic"), ("unsettled", "geometric"), ("unsettled", "symbolic"))
VKEYS = [f"{m}_{k}" for m, k in VERDICTS]
VSHORT = {"settled_geometric": "settled geom", "settled_symbolic": "settled symb", "unsettled_geometric": "unsettled geom",
          "unsettled_symbolic": "unsettled symb"}
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
    """The four verdicts and the two settle modes' errors of a run's meta.json (None where not recorded)."""
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


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Identity:
    """An instance's dataset identity: meta.eval_instance, else ckpt.json's expected entries, else the dataset index (lazy)."""

    def __init__(self, instances_root):
        self.root, self._index = instances_root, None

    def index(self):
        if self._index is None:
            self._index = {}
            p = os.path.join(self.root or "", "metadata", "instances.jsonl")
            if self.root and os.path.exists(p):
                for i, line in enumerate(open(p)):
                    if line.strip():
                        r = json.loads(line)
                        iid = os.path.basename(r["files"]["tro_state"])[: -len("-tro_state.json")]
                        self._index[iid] = {"index": i, "instance_id": iid, "class_id": r["source"].get("class_id"), "task_family": r["task_family"],
                                            "task": r["task"], "subtask": r["subtask"]}
        return self._index

    def of(self, iid, meta, expected):
        ident = dict(meta.get("eval_instance") or {})
        if not ident.get("class_id"):
            ident = {**(expected.get(iid) or {}), **ident}
        if not ident.get("class_id"):
            ident = {**(self.index().get(iid) or {}), **ident}
        ident.setdefault("task_family", meta.get("task_family"))
        ident.setdefault("task", meta.get("task"))
        ident.setdefault("subtask", meta.get("subtask"))
        ident.setdefault("index", None)
        ident.setdefault("class_id", None)
        return ident


def instance_row(iid, meta, result, worker, ident):
    pc = meta.get("policy_client") or {}
    task = ident.get("task") or meta.get("task")
    success = (result or {}).get("success", meta.get("success"))
    return {"index": ident.get("index"), "instance_id": iid, "task_family": ident.get("task_family"), "task_class": TASK_CLASSES.get(task, task),
            "class_id": ident.get("class_id"), "subtask": ident.get("subtask"), "success": bool(success),
            "status": (result or {}).get("status", meta.get("status")), "reason": (result or {}).get("reason", meta.get("reason")),
            "n_steps": meta.get("n_steps"), "budget": (meta.get("budget") or {}).get("max_steps"), "evaluation_step": meta.get("evaluation_step"),
            "evaluation_kind": meta.get("evaluation_kind"), **verdicts(meta), "closest_score": (meta.get("closest_approach") or {}).get("score"),
            "goal_kind": pc.get("goal_kind"), "cross_episode_goal": pc.get("cross_episode_goal"), "qa": qa_counts(meta),
            "load_s": (meta.get("timings") or {}).get("load_s"), "episode_s": (meta.get("timings") or {}).get("episode_s"),
            "gpu": (worker or {}).get("gpu_name"), "unit": (worker or {}).get("unit"), "run": (worker or {}).get("run"),
            "request_s_mean": pc.get("request_s_mean")}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def qa_stats(rows, metrics):
    out = {}
    for q in metrics:
        vals = [r["qa"].get(q, {"steps": 0, "runs": 0, "triggered": False}) for r in rows]
        out[q] = {"mean_violations": statistics.fmean(v["steps"] for v in vals) if vals else None,
                  "mean_runs": statistics.fmean(v["runs"] for v in vals) if vals else None,
                  "fired_rate": statistics.fmean(v["triggered"] for v in vals) if vals else None}
    return out


def aggregate(rows, metrics):
    n = len(rows)
    out = {"n": n}
    for v in VKEYS:
        vals = [r[v] for r in rows if r.get(v) is not None]
        out[f"{v}_rate"] = statistics.fmean(vals) if vals else None
        out[f"{v}_successes"] = sum(vals)
        mode = v.split("_")[0]
        for e in ERRORS:
            out[f"{v}_{e}_successes"] = mean(r.get(f"{mode}_{e}") for r in rows if r.get(v))
    for mode in ("settled", "unsettled"):
        for e in ERRORS:
            out[f"{mode}_{e}_all"] = mean(r.get(f"{mode}_{e}") for r in rows)
    out["qa"] = {"all": {"n": n, "metrics": qa_stats(rows, metrics)}}
    for v in VKEYS:
        succ = [r for r in rows if r.get(v)]
        out["qa"][v] = {"n": len(succ), "metrics": qa_stats(succ, metrics)}
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def num(x):
    return "-" if x is None else f"{x:.3f}"


def qa_cell(s):
    if s["mean_violations"] is None:
        return "-"
    return f"{s['mean_violations']:.1f} / {s['mean_runs']:.2f} / {100 * s['fired_rate']:.0f}%"


def group_order(level, names):
    if level == "task_family":
        rank = {f: i for i, f in enumerate(FAMILIES)}
    elif level == "task_class":
        rank = {c: i for i, c in enumerate(TASK_CLASSES.values())}
    else:
        rank = {}
    return sorted(names, key=lambda k: (rank.get(k, len(rank)), str(k)))


def label(level, name):
    return name.replace("_", " ") if level == "task_family" and name else str(name)


def success_table(md, level, title, groups, total):
    md.append(f"\n### Success and errors per {title}\n")
    md.append(f"| {title} | n | settled geom | settled symb | unsettled geom | unsettled symb | pos err settled: all / geom succ / symb succ | "
              "pos err unsettled: all / geom succ / symb succ | ori err settled: all / geom succ / symb succ | "
              "ori err unsettled: all / geom succ / symb succ |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, g in [(label(level, k), groups[k]) for k in group_order(level, groups)] + [("**total**", total)]:
        errs = [" / ".join([num(g[f"{m}_{e}_all"]), num(g[f"{m}_geometric_{e}_successes"]), num(g[f"{m}_symbolic_{e}_successes"])])
                for e in ERRORS for m in ("settled", "unsettled")]
        md.append(f"| {name} | {g['n']} | " + " | ".join(f"{pct(g[f'{v}_rate'])} ({g[f'{v}_successes']})" for v in VKEYS) + " | "
                  + " | ".join(errs) + " |")


def qa_table(md, level, title, groups, total, metrics, rollouts):
    md.append("| " + title + " | rollouts | n | " + " | ".join(metrics) + " |")
    md.append("|---|---|---|" + "---|" * len(metrics))
    items = [(label(level, k), groups[k]) for k in group_order(level, groups)] if groups else []
    for name, g in items + [("**total**", total)]:
        for ro in rollouts:
            q = g["qa"][ro]
            md.append(f"| {name} | {VSHORT.get(ro, ro)} | {q['n']} | " + " | ".join(qa_cell(q["metrics"][m]) for m in metrics) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results dir: <ckpt>/<instance id>/meta.json")
    ap.add_argument("--out", default=None, help="where the summary files go (default: the results dir)")
    ap.add_argument("--ckpts", default=None, help="comma list (default: every checkpoint dir)")
    ap.add_argument("--instances", default=os.environ.get("ISPATIALGYM_INSTANCES_PATH"),
                    help="eval-instances root: the identity of instances whose meta.json predates meta.eval_instance and that no ckpt.json lists")
    ap.add_argument("--quiet", action="store_true", help="print only the overview")
    a = ap.parse_args()
    root = os.path.abspath(a.results)
    out_dir = a.out or root
    os.makedirs(out_dir, exist_ok=True)
    ident = Identity(a.instances)
    ckpts = [c for c in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, c)) and not c.startswith((".", "_"))
             and (os.path.exists(os.path.join(root, c, "ckpt.json")) or any(os.path.exists(os.path.join(root, c, d, "meta.json"))
                                                                              for d in os.listdir(os.path.join(root, c))))]
    if a.ckpts:
        ckpts = [c for c in ckpts if c in a.ckpts.split(",")]
    overview = ["| ckpt | evaluated / expected | settled geom | settled symb | unsettled geom | unsettled symb |", "|---|---|---|---|---|---|"]
    sections, all_rows, group_rows, qa_rows = [], [], [], []
    for tag in ckpts:
        cdir = os.path.join(root, tag)
        cj = load_json(os.path.join(cdir, "ckpt.json")) or {}
        expected = {}
        for run in (cj.get("runs") or {}).values():
            for x in run.get("expected") or []:
                expected[x["instance_id"]] = x
        rows, errored = [], []
        for iid in sorted(os.listdir(cdir)):
            d = os.path.join(cdir, iid)
            if iid.startswith(("_", ".")) or not os.path.isdir(d):
                continue
            meta = load_json(os.path.join(d, "meta.json"))
            if meta is None:
                errored.append(iid)
                continue
            result, worker = load_json(os.path.join(d, "result.json")), load_json(os.path.join(d, "worker.json"))
            rows.append(instance_row(iid, meta, result, worker, ident.of(iid, meta, expected)))
        present = {r["instance_id"] for r in rows}
        missing = sorted(set(expected) - present)
        metrics = sorted({q for r in rows for q in r["qa"]})
        levels = {}
        for level, _ in LEVELS:
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r[level]].append(r)
            levels[level] = {k: aggregate(v, metrics) for k, v in groups.items()}
        total = aggregate(rows, metrics)
        macro = {lvl: {v: mean(g[f"{v}_rate"] for g in levels[lvl].values()) for v in VKEYS} for lvl, _ in LEVELS}
        summ = {"ckpt": tag, "results_dir": root, "ckpt_spec": cj.get("spec"), "runs": {k: {kk: vv for kk, vv in v.items() if kk != "expected"}
                                                                                         for k, v in (cj.get("runs") or {}).items()},
                "evaluated": len(rows), "expected": len(expected) or None, "missing": missing, "errored": errored, "total": total,
                "macro_success_rate": macro, "levels": levels, "instances": rows}
        n_exp = len(expected) or len(rows)
        overview.append(f"| {tag} | {len(rows)} / {n_exp} | " + " | ".join(f"{pct(total[f'{v}_rate'])} ({total[f'{v}_successes']})" for v in VKEYS) + " |")
        md = [f"## {tag}\n", f"{len(rows)}/{n_exp} instances evaluated" + (f"; {len(missing)} missing: {', '.join(missing[:10])}"
                                                                           + (" ..." if len(missing) > 10 else "") if missing else "")
              + (f"; {len(errored)} without meta.json: {', '.join(errored[:5])}" if errored else "")]
        if rows:
            md.append("\nMacro averages (each group counts once): " + "; ".join(
                f"over {'families' if lvl == 'task_family' else 'task classes'} " + ", ".join(f"{VSHORT[v]} {pct(macro[lvl][v])}" for v in VKEYS) for lvl, title in LEVELS[:2]))
            for level, title in LEVELS:
                success_table(md, level, title, levels[level], total)
            md.append("\n### Soft QA, all rollouts (logged, never failing an episode): mean violations / mean separate runs / share of rollouts where it fired\n")
            qa_table(md, "task_family", "family", levels["task_family"], total, metrics, ["all"])
            md.append("\n### Soft QA, successful rollouts only (per success type: n = its successful rollouts)\n")
            qa_table(md, "task_family", "family", levels["task_family"], total, metrics, VKEYS)
            md.append("\n(per task class and per task: summary_<ckpt>.json, qa.csv)")
            kinds = collections.Counter((r["goal_kind"], r["cross_episode_goal"]) for r in rows)
            if any(k[0] and not str(k[0]).startswith("own") for k in kinds):  # run_isg --cross-goal: the verdicts per goal kind / difficulty
                md.append("\n### Per goal image kind\n\n| goal image | n | settled geom | unsettled geom |\n|---|---|---|---|")
                summ["by_goal_kind"] = {}
                for (kind, diff), n in sorted(kinds.items(), key=lambda kv: str(kv[0])):
                    g = aggregate([r for r in rows if (r["goal_kind"], r["cross_episode_goal"]) == (kind, diff)], [])
                    md.append(f"| {kind}{' / ' + diff if diff else ''} | {n} | {pct(g['settled_geometric_rate'])} | {pct(g['unsettled_geometric_rate'])} |")
                    summ["by_goal_kind"][f"{kind}/{diff}"] = {k: v for k, v in g.items() if k != "qa"}
        sections += md + [""]
        with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as f:
            json.dump(summ, f, indent=1)
        all_rows += [{**r, "ckpt": tag} for r in rows]
        for level, _ in LEVELS:
            for name, g in list(levels[level].items()) + ([("total", total)] if level == "task_family" else []):
                lvl = "total" if name == "total" and g is total else level
                group_rows.append({"ckpt": tag, "level": lvl, "group": name, **{k: v for k, v in g.items() if k != "qa"}})
                for ro, q in g["qa"].items():
                    for m, s in q["metrics"].items():
                        qa_rows.append({"ckpt": tag, "level": lvl, "group": name, "rollouts": ro, "n": q["n"], "metric": m, **s})
    head = [f"# Policy eval results: {root}\n",
            "Verdicts: settled = the success held over the settle window (the eval verdict with --settle-eval on); unsettled = the success "
            "reached at all (its lowest-error step of the window after the first success); symbolic = the BDDL goal, geometric = the goal "
            "and the measured atoms within tolerance. Rates with success counts in parentheses. Errors: each settle mode's error at its "
            "evaluated step (the success step, else the last step), mean over all instances / over that verdict's successes; position m, "
            "orientation rad (only tasks that measure it). Groups: family; task class (23); task = the dataset's class_id.\n",
            "## Overview\n", *overview, ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(head + sections) + "\n")
    if all_rows:
        keys = [k for k in all_rows[0] if k not in ("qa", "ckpt")]
        qkeys = sorted({q for r in all_rows for q in r["qa"]})
        with open(os.path.join(out_dir, "instances.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ckpt"] + keys + [f"qa_{q}_violations" for q in qkeys] + [f"qa_{q}_runs" for q in qkeys])
            for r in all_rows:
                w.writerow([r["ckpt"]] + [r.get(k) for k in keys] + [r["qa"].get(q, {}).get("steps", 0) for q in qkeys]
                           + [r["qa"].get(q, {}).get("runs", 0) for q in qkeys])
    for name, rows in (("groups.csv", group_rows), ("qa.csv", qa_rows)):
        if rows:
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("\n".join(head[-len(overview) - 1:] if a.quiet else head + sections))
    print(f"\nwrote {out_dir}/summary.md, summary_<ckpt>.json, instances.csv, groups.csv, qa.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
