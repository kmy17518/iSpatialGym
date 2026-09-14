#!/usr/bin/env python3
"""Aggregate ``results/*.json`` of validate_tables.py into

* ``results_summary.csv``  - one row per (scene, table[, partner], problem): status, instances, rejections, stance, ...
* ``candidates_by_task.json`` - per problem -> per scene -> viable supports (with evidence) and rejected supports (with reason)
* ``candidates_by_task.md``   - the same, readable.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")

SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]
PROBLEMS = ["disembedding-0", "configuration_matching-0", "configuration_matching-1", "configuration_matching-2", "configuration_matching-3",
            "world_alignment-0", "inter_object_relation_satisfaction-0", "inter_object_relation_satisfaction-1",
            "inter_object_relation_satisfaction-2", "inter_object_relation_satisfaction-3", "inter_object_relation_satisfaction-4"]


def option_tags(err: str) -> set:
    """Innermost ``[reason]`` tag of every option tried in a sampler Reject message (options are joined with '; ')."""
    tags = set()
    for part in err.split("; "):
        found = [part[i + 1:part.index("]", i)] for i in range(len(part)) if part[i] == "[" and "]" in part[i:]]
        if found:
            tags.add(found[-1])
    return tags


def failure_class(rec: dict) -> str:
    """Coarse failure class for the sheet: no stance / unreachable / no fit on top / low yield / error / skipped."""
    st = rec.get("status")
    if st in ("ok", "ok_other_support"):
        return ""
    if st == "skipped":
        return "no stance or reach (skipped after IORS ontop)" if "no reachable" in str(rec.get("reason", "")) or "no stance" in str(rec.get("reason", "")) else "skipped"
    if st == "no_instances":
        rej = {k for k in (rec.get("rejections") or {}) if k != "scene_reset"}
        if rej == {"under_goal"}:
            return "no goal spot under the partner (clearance)"
        return "template ok, no instance (low yield)"
    if st in ("error", "crashed"):
        return "error"
    tags = option_tags(str(rec.get("error", "")))
    if tags and tags <= {"no_stance", "stance_collision"}:
        return "no stance"
    if tags and tags <= {"no_stance", "stance_collision", "partner_not_visible", "partner_far"}:
        return "partner not visible / too far"
    if "unreachable" in tags:
        return "unreachable"
    if "partner_not_visible" in tags or "partner_far" in tags:
        return "partner not visible / too far"
    if "support" in tags or "spec" in tags:
        return "no fit on the top (region / trays / view)"
    return rec.get("reject_reason") or st or ""


def short_reason(rec: dict) -> str:
    st = rec.get("status")
    if st in ("ok", "ok_other_support"):
        return ""
    if st == "skipped":
        return rec.get("reason", "skipped")
    if st == "crashed":
        return "simulator process died during this run"
    if st == "no_instances":
        rej = {k: v for k, v in (rec.get("rejections") or {}).items() if k != "scene_reset"}  # scene_reset counts resets, not attempts
        return "template ok, no instance in " + str(sum(rej.values())) + " attempts: " + ", ".join(f"{k} x{v}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1]))
    err = str(rec.get("error", ""))
    # the samplers join every tried option with ';' - keep the leading reason and the distinct bracketed tags
    tags = []
    for part in err.replace("[", "\n[").split("\n"):
        part = part.strip()
        if part.startswith("[") and "]" in part:
            tag = part[1:part.index("]")]
            if tag not in tags:
                tags.append(tag)
    head = err.split(";")[0].strip()
    return (rec.get("reject_reason") or st or "") + ": " + head[:220] + (f" (tags: {', '.join(tags[:6])})" if len(tags) > 1 else "")


def stance_txt(rec: dict) -> str:
    tm = rec.get("template_meta") or {}
    st = tm.get("stance") or {}
    sup = tm.get("support") or {}
    parts = []
    if sup:
        parts.append(f"top z {sup.get('z', 0):.2f} m, {sup.get('n_free_cells', '?')} free cells")
        if sup.get("occupants"):
            parts.append(f"occupants {sup['occupants']}")
    if st:
        v = st.get("view") or {}
        parts.append(f"stance edge {st.get('edge')} base ({st['base_pose2d'][0]:.2f}, {st['base_pose2d'][1]:.2f}, yaw {st['base_pose2d'][2]:.2f})")
        if v:
            parts.append(f"trunk translate {v.get('translate', 0):.2f} tilt {v.get('tilt', 0):.2f}")
    if tm.get("view_arms") and tm.get("view_arms") != "untucked":
        parts.append(f"arms {tm['view_arms']}")
    if tm.get("yaw_adjust_deg"):
        parts.append(f"base yaw turned {tm['yaw_adjust_deg']:.1f} deg (destination barely visible fallback)")
    if "stand_off" in tm and tm["stand_off"] != 0.45:
        parts.append(f"stand-off {tm['stand_off']}")
    if tm.get("approach_forward"):
        parts.append(f"drive forward {tm['approach_forward']}")
    for k in ("n_reachable_cells", "n_grasp_cells"):
        if k in tm:
            parts.append(f"{k} {tm[k]}")
    return "; ".join(parts)


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        if p.endswith("__precheck.json"):
            continue
        with open(p) as f:
            recs.append(json.load(f))
    rows = []
    for r in recs:
        inst = r.get("instances") or []
        rows.append(OrderedDict([
            ("scene", r["scene"]), ("problem", r["problem"]), ("table", r["table"]), ("category", r.get("category")), ("model", r.get("model")),
            ("partner", r.get("partner") or ""), ("rooms", ";".join(r.get("rooms") or [])), ("status", r.get("status")),
            ("n_written", r.get("n_written", 0)), ("n_requested", r.get("n_requested", "")),
            ("attempts_total", sum(v for k, v in (r.get("rejections") or {}).items() if k != "scene_reset") + r.get("n_written", 0) if r.get("status") not in ("skipped", "crashed") else ""),
            ("template_seeds", len(r.get("attempts") or [])),
            ("rejections", json.dumps({k: v for k, v in (r.get("rejections") or {}).items() if k != "scene_reset"})), ("reason", short_reason(r)),
            ("failure_class", failure_class(r)),
            ("inventory_top_z", (r.get("inventory") or {}).get("top_z")), ("support_top_z", ((r.get("template_meta") or {}).get("support") or {}).get("z")),
            ("stance", stance_txt(r)),
            ("init_true", all(i.get("init_conditions_all_true") for i in inst) if inst else ""),
            ("goal_false_at_init", all(i.get("goal_false_at_init") for i in inst) if inst else ""),
            ("goal_true_at_goal", all(i.get("goal_true_at_goal") for i in inst) if inst else ""),
            ("setup_s", r.get("setup_s", "")), ("run_s", r.get("run_s", "")), ("job_s", r.get("job_s", "")),
            ("work_dir", r.get("work_dir", "")),
        ]))
    rows.sort(key=lambda x: (SCENES.index(x["scene"]) if x["scene"] in SCENES else 99, PROBLEMS.index(x["problem"]) if x["problem"] in PROBLEMS else 99,
                             x["status"] not in ("ok", "ok_other_support"), x["table"], x["partner"]))
    with open(os.path.join(ROOT, "results_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scene"])
        w.writeheader()
        for x in rows:
            w.writerow(x)

    by_task = OrderedDict()
    for prob in PROBLEMS:
        by_task[prob] = OrderedDict()
        for scene in SCENES:
            sel = [x for x in rows if x["problem"] == prob and x["scene"] == scene]
            if not sel:
                continue
            viable = [x for x in sel if x["status"] == "ok"]
            fallback = [x for x in sel if x["status"] == "ok_other_support"]
            rejected = [x for x in sel if x["status"] not in ("ok", "ok_other_support")]
            by_task[prob][scene] = {
                "viable": [{"table": x["table"], "category": x["category"], "model": x["model"], "partner": x["partner"] or None, "rooms": x["rooms"],
                            "n_written": x["n_written"], "attempts_total": x["attempts_total"], "rejections": json.loads(x["rejections"]),
                            "support_top_z": x["support_top_z"], "stance": x["stance"]} for x in viable],
                "viable_other_support": [{"table": x["table"], "partner": x["partner"], "note": "instances were written but the sampler bound another support (see results json)"} for x in fallback],
                "rejected": [{"table": x["table"], "category": x["category"], "partner": x["partner"] or None, "status": x["status"], "reason": x["reason"],
                              "failure_class": x["failure_class"]} for x in rejected],
            }
    with open(os.path.join(ROOT, "candidates_by_task.json"), "w") as f:
        json.dump(by_task, f, indent=1)

    lines = ["# Table / work-surface candidates per task (sampler-validated)", "",
             "Status `ok` = `TaskSampler.setup()` succeeded on that support and `TaskSampler.run()` wrote >= 1 instance "
             "(init settled and BEHAVIOR-validated, `:init` true, goal false at init, goal feasible by teleport + settle). "
             "Details: `results_summary.csv`, `results/<scene>__<table>__<problem>.json`.", ""]
    for prob, scenes in by_task.items():
        lines.append(f"## {prob}")
        lines.append("")
        for scene, d in scenes.items():
            lines.append(f"### {scene}")
            if d["viable"]:
                for v in d["viable"]:
                    p_txt = f" (partner {v['partner']})" if v["partner"] else ""
                    lines.append(f"- **{v['table']}**{p_txt} [{v['category']}/{v['model']}, rooms {v['rooms']}]: {v['n_written']} instance(s) in "
                                 f"{v['attempts_total']} attempt(s), rejections {v['rejections'] or 'none'}; {v['stance']}")
            else:
                lines.append("- (no viable support)")
            for v in d["viable_other_support"]:
                lines.append(f"- {v['table']} (partner {v['partner']}): {v['note']}")
            if d["rejected"]:
                lines.append("")
                lines.append("  rejected:")
                for x in d["rejected"]:
                    p_txt = f" (partner {x['partner']})" if x["partner"] else ""
                    lines.append(f"  - {x['table']}{p_txt} [{x['category']}]: {x['status']} - {x['reason']}")
            lines.append("")
    with open(os.path.join(ROOT, "candidates_by_task.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    n_ok = sum(1 for x in rows if x["status"] == "ok")
    print(f"{len(rows)} runs, {n_ok} ok; wrote results_summary.csv, candidates_by_task.json, candidates_by_task.md")
    stat = defaultdict(int)
    for x in rows:
        stat[(x["scene"], x["status"])] += 1
    for k in sorted(stat):
        print(f"  {k[0]:26s} {k[1]:18s} {stat[k]}")


if __name__ == "__main__":
    main()
