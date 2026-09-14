#!/usr/bin/env python3
"""Record the sampler-validated table / work-surface candidates in
``runlogs/grasp_library_20260905/select/task_object_selection.csv`` (and regenerate the .xlsx / .html twins).

For every task with a fixed ``table`` role the header row ``<task>,table,...`` gets

* category / model cells = number of distinct support categories / distinct scene objects that passed,
* a note summarising the validation (spec-pinned support, viable supports per scene, rejected supports with the
  sampler's reason), and
* one sub-row per viable support: ``category`` = support category, ``model`` = scene object name,
  ``selection`` = ``viable`` (``spec`` when it is the support pinned in task_custom_lists.json), ``pair`` =
  ``<scene>: <room(s)>`` (for IORS under also the partner), ``note`` = evidence (instances written / attempts,
  top height, stance, trunk, ...).

Idempotent: previously inserted table sub-rows (rows between the table header and the next header row) are replaced.
Input: ``candidates_by_task.json`` from aggregate_results.py.
"""
from __future__ import annotations

import argparse
import csv
import html as _html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
XLSX_PATH = os.path.join(SEL, "task_object_selection.xlsx")
HTML_PATH = os.path.join(SEL, "task_object_selection.html")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
RUN = "runlogs/table_selection_20260911"

SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]

#: CSV task label -> problems whose results feed the row (articulation shares one row for -1 / -2)
TASK_ROWS = {
    "disembedding-0": ["disembedding-0"],
    "configuration_matching-0 (block_design)": ["configuration_matching-0"],
    "configuration_matching-1/-2 (articulation; -1 closed->open, -2 open->closed)": ["configuration_matching-1", "configuration_matching-2"],
    "configuration_matching-3 (attachment)": ["configuration_matching-3"],
    "world_alignment-0 (table_surface)": ["world_alignment-0"],
    "inter_object_relation_satisfaction-0 (ontop)": ["inter_object_relation_satisfaction-0"],
    "inter_object_relation_satisfaction-1 (under)": ["inter_object_relation_satisfaction-1"],
    "inter_object_relation_satisfaction-2 (inside)": ["inter_object_relation_satisfaction-2"],
    "inter_object_relation_satisfaction-3 (nextto)": ["inter_object_relation_satisfaction-3"],
    "inter_object_relation_satisfaction-4 (touching)": ["inter_object_relation_satisfaction-4"],
}


def pinned_tables(spec):
    out = {}
    for prob, body in spec.items():
        if prob.startswith("_"):
            continue
        for scene, cfg in (body.get("scenes") or {}).items():
            t = (cfg.get("fixed") or {}).get("table")
            if isinstance(t, list):
                for n in t:
                    out.setdefault(prob, set()).add((scene, n))
    return out


def fmt_rej(rej: dict) -> str:
    rej = {k: v for k, v in (rej or {}).items() if k != "scene_reset"}
    if not rej:
        return "no rejected attempt"
    return "rejected attempts " + ", ".join(f"{k} x{v}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1]))


def build_rows(cands, problems, pinned, spec_pinned_note):
    """(header category count, header model count, header note tail, sub-rows) for one CSV task row."""
    viable = {}  # (scene, table, partner) -> merged record
    rejected = {}  # scene -> list of (table, partner, status, reason)
    for prob in problems:
        for scene, d in (cands.get(prob) or {}).items():
            for v in d.get("viable", []):
                key = (scene, v["table"], v.get("partner"))
                rec = viable.setdefault(key, {"problems": [], **v, "scene": scene})
                rec["problems"].append(prob)
                rec.setdefault("per_problem", {})[prob] = (v["n_written"], v["attempts_total"], v["rejections"])
            for x in d.get("rejected", []):
                rejected.setdefault(scene, []).append((x["table"], x.get("partner"), x["status"], x["reason"], prob, x.get("failure_class", "")))
    # a support counts as rejected for the task only if no problem of the task passed on it
    sub_rows = []
    for scene in SCENES:
        for key in sorted(k for k in viable if k[0] == scene):
            v = viable[key]
            is_pinned = any((scene, v["table"]) in pinned.get(p, set()) for p in problems)
            parts = []
            if is_pinned:
                parts.append("PINNED in task_custom_lists.json (role table) for the test round")
            ev = "; ".join(f"{p.rsplit('-', 1)[1] if len(problems) > 1 else 'sampler'}: {n} instance(s) in {a} attempt(s) ({fmt_rej(r)})"
                           for p, (n, a, r) in v["per_problem"].items())
            if len(problems) > 1:
                ev = "problems " + ", ".join(p.rsplit("-", 1)[1] for p in v["problems"]) + " -> " + ev
            parts.append(ev)
            if v.get("stance"):
                parts.append(v["stance"])
            elif v.get("support_top_z") is not None:
                parts.append(f"support top z {v['support_top_z']:.2f} m")
            parts.append(f"evidence {RUN}/results/{scene}__{v['table']}" + (f"__under_{v['partner']}" if v.get("partner") else "") + "__<problem>.json")
            pair = f"{scene}: {v['rooms'].replace(';', ', ')}" + (f"; partner {v['partner']}" if v.get("partner") else "")
            sub_rows.append({"task": "", "role": "", "category": v["category"], "model": v["table"], "selection": "spec" if is_pinned else "viable",
                             "pair": pair, "duplicates": "", "note": "; ".join(parts)})
    # header
    cats = sorted({r["category"] for r in sub_rows})
    n_models = len({(r["pair"], r["model"]) for r in sub_rows})  # (scene, support) pairs: bottom_cabinet_jhymlr_0 exists in two scenes
    per_scene = []
    for scene in SCENES:
        ok = [k[1] + (f" (partner {k[2]})" if k[2] else "") for k in sorted(viable) if k[0] == scene]
        groups = {}
        seen = set()
        for table, partner, status, reason, prob, fclass in rejected.get(scene, []):
            if (scene, table, partner) in viable or (table, partner) in seen:
                continue
            seen.add((table, partner))
            groups.setdefault(fclass or status, []).append(f"{table}{' (partner ' + partner + ')' if partner else ''}")
        rej = [f"{k}: {', '.join(v)}" for k, v in groups.items()]
        if ok or rej:
            per_scene.append(f"{scene}: VIABLE {', '.join(ok) if ok else 'none'}" + (f" / rejected - {'; '.join(rej)}" if rej else ""))
    note = (f"fixed scene object (table role). {spec_pinned_note} Candidates in the 7 scenes were pre-selected geometrically "
            f"({RUN}/inventory.csv) and validated with the instance sampler exactly as sample_instances.py generates instances "
            f"(TaskSampler.setup: support raster, stance, view, reachability; TaskSampler.run: init settled + BEHAVIOR-validated, "
            f":init true, goal false at init, goal feasible by teleport+settle); viable = >= 1 instance written. Rows below: one per viable "
            f"support (model = scene object name, pair = scene: room(s)); the rejected supports and the sampler's reason are listed here. "
            + " | ".join(per_scene))
    return len(cats), n_models, note, sub_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=os.path.join(ROOT, "candidates_by_task.json"))
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--no-render", action="store_true", help="do not regenerate the xlsx / html")
    args = ap.parse_args()
    with open(args.candidates) as f:
        cands = json.load(f)
    with open(SPEC) as f:
        spec = json.load(f)
    pinned = pinned_tables(spec)
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    i = 0
    n_updated = 0
    while i < len(rows):
        r = rows[i]
        task_label = r["task"]
        if task_label in TASK_ROWS and r["role"] == "table":
            problems = TASK_ROWS[task_label]
            # drop previously inserted sub-rows (until the next header row)
            j = i + 1
            while j < len(rows) and not rows[j]["task"]:
                j += 1
            pin_txt = "; ".join(sorted({f"{s}: {n}" for p in problems for (s, n) in pinned.get(p, set())}))
            spec_note = f"Test spec pins {pin_txt}." if pin_txt else "Test spec: table chosen automatically among scene categories."
            if task_label.startswith("inter_object_relation_satisfaction-1"):
                spec_note = ("Init support of the target next to the fixed partner (bed / sofa); test spec: auto among scene desk / nightstand / "
                             "breakfast_table in house_single_floor childs_room_0 (partner bed_rrcvaq_0). Candidates below are (partner, table) pairs. "
                             "With the frozen sampler no pair passed with the table as init support; the sampler's fallback (target initialised on the "
                             "partner's own top, spec option (b)) passed for house_single_floor desk_xhjsub_0 / bed_rrcvaq_0 (the test pair), "
                             "desk_bhkhxo_0 / bed_rrcvaq_1 and bottom_cabinet_xqmipa_0 / bed_rrcvaq_1 - recorded as 'ok_other_support', not as viable tables.")
            n_cat, n_mod, note, subs = build_rows(cands, problems, pinned, spec_note)
            hdr = dict(r)
            hdr["category"] = str(n_cat)
            hdr["model"] = str(n_mod)
            hdr["note"] = note
            out.append(hdr)
            out.extend(subs)
            n_updated += 1
            i = j
            continue
        if r["role"] == "floor" and task_label in TASK_ROWS:
            r = dict(r)
            base = re.sub(r";? ?other scenes:.*$", "", r["note"])
            r["note"] = base + "; other scenes: floor of the room of the bound table (see the table row)"
        out.append(r)
        i += 1
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"updated {n_updated} table rows; {len(out)} rows -> {args.csv}")
    if not args.no_render:
        render(out)


# ----------------------------------------------------------------------------------------------------------------
# xlsx / html twins (same styling as the generator of the sheet, /tmp/build_task_object_csv.py of 2026-09-11)
# ----------------------------------------------------------------------------------------------------------------


def render(rows, extra_legend=()):
    """Write the xlsx / html twins of ``rows``. ``extra_legend``: extra (item, meaning) pairs appended to the legend
    sheet (used by later selection runs that add rows, e.g. runlogs/world_alignment_selection_20260911)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    FILL_HEADER = PatternFill("solid", fgColor="D9E1F2")
    FILL_PICK1 = PatternFill("solid", fgColor="C6EFCE")
    FILL_PICK2 = PatternFill("solid", fgColor="FFF2CC")
    FILL_VIABLE = PatternFill("solid", fgColor="E2F0D9")
    FILL_EXACT = PatternFill("solid", fgColor="F8CBAD")
    FILL_TEXVAR = PatternFill("solid", fgColor="DDEBF7")
    FONT_BOLD = Font(bold=True)
    FONT_PINNED = Font(bold=True, color="1F4E79")
    WIDTHS = {"task": 46, "role": 20, "category": 28, "model": 14, "selection": 12, "pair": 62, "duplicates": 58, "note": 160}

    wb = Workbook()
    ws = wb.active
    ws.title = "objects per task"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = FONT_BOLD
        c.fill = FILL_HEADER
    for r in rows:
        ws.append([r[c] for c in COLUMNS])
        cells = ws[ws.max_row]
        if r["task"]:
            for c in cells:
                c.fill = FILL_HEADER
                c.font = FONT_BOLD
            continue
        sel = r["selection"]
        fill = {"pick 1": FILL_PICK1, "pick 2": FILL_PICK2, "viable": FILL_VIABLE}.get(sel)
        if fill:
            for c in cells:
                c.fill = fill
        if sel == "pick 1":
            for c in cells:
                c.font = FONT_BOLD
        if "PINNED" in r["note"]:
            cells[COLUMNS.index("model")].font = FONT_PINNED
        d = r["duplicates"]
        if d.startswith("exact duplicate") or d.startswith("LOOK-ALIKE"):
            cells[COLUMNS.index("duplicates")].fill = FILL_EXACT
        elif d.startswith("same geometry"):
            cells[COLUMNS.index("duplicates")].fill = FILL_TEXVAR
    for i, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = WIDTHS[col]
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    lg = wb.create_sheet("legend")
    legend = [
        ("Layout", "A task / role header row (blue-grey, bold) carries the BDDL object list and role requirements in 'note'; "
                   "its category / model cells hold the number of distinct categories / models listed under the role "
                   "(models as 'M (N)', see below). "
                   "The rows below it (task / role blank) are the objects for that role: category, model, selection, pair, note."),
        ("category column", "Within a role, category blocks are ordered by their number of distinct geometries (descending), then "
                            "distinct models, then name. A category's name appears on the first row of its block only; the last row of "
                            "the block shows 'M (N)': M = distinct geometries of that category (exact duplicates AND texture/colour "
                            "variants / the engraved alphabet blocks counted once), N = distinct models (only exact duplicates counted "
                            "once). Single-row categories just show the name."),
        ("model count in header rows", "'M (N)' with the same meaning as above, over all models listed under the role."),
        ("selection = pick 1", "Test-instance pick 1 for the (task, category) group (green row, bold)."),
        ("selection = pick 2", "Test-instance pick 2 (yellow row)."),
        ("selection = alternate", "3rd choice of the selection reviewers, not selected (no fill; a training candidate like the rest)."),
        ("selection = candidate", "Any other model of the category with >= 1 AG-valid grasp in the grasp-library run for this task "
                                  "(training candidate). Sorted by selection score. Models with no AG-valid grasp are not listed; "
                                  "categories where no model passed are named in the role header note."),
        ("selection = viable", "Articulated object whose joint the robot actually opened in the articulation pass; or (table rows, pale green) "
                               "a scene table / work surface on which the instance sampler generated >= 1 validated instance of the task "
                               "(runlogs/table_selection_20260911). For table rows: model = scene object name, pair = scene: room(s)."),
        ("selection = spec", "Object pinned in ispatialgym/assets/task_specs/task_custom_lists.json for a role without a grasp test "
                             "(trays, blocks, keyboard, painting, partners, the test-round table); for the disembedding pool rows: one row "
                             "per target category of the spec's distractors section."),
        ("disembedding pools", "disembedding-0 rows same_category_pool / distractor_pool: one row per target category (model cell = pool "
                               "size, note = models / categories). Spec: disembedding-0.distractors (pools.<category>.same_category, "
                               "pools.<category>.other_categories, lookalike_groups, families); full record in "
                               "select/disembedding_distractor_pools.json, builder runlogs/grasp_library_20260905/scripts/build_disembedding_pools.py. "
                               "The pair column of every disembedding target row gives the pool sizes for that model as target."),
        ("duplicates = LOOK-ALIKE with ...", "Disembedding target rows: the model cannot be told apart from the listed models in the robot's "
                                             "initial view (exact duplicate, or same shape with only a small print / label / shade difference; "
                                             "reason in brackets). Members of a look-alike group are never drawn as distractors of each other. "
                                             "Red cell; the original duplicate note follows after ' | '."),
        ("selection = partner", "Partner category / model named by the selection (not pinned in the spec)."),
        ("partner model rows (IORS-0..3)", "The partner categories of the IORS ontop / under / inside / nextto problems list their ready models "
                                           "with pick 1 / pick 2 / alternate / candidate: every listed model fits the sampler-validated IORS "
                                           "supports (footprint) and the relation geometry against the listed targets (wide enough to support "
                                           "the target; target passes the fillable-volume opening and its centre stays inside; NextTo bound vs "
                                           "the 2 cm goal gap); ranking and per-model evidence in select/partners/ (select_partners.py, "
                                           "partner_fit.json, partner_geometry.json)."),
        ("model in bold blue", "Model is pinned in task_custom_lists.json for this problem (the note says in which role)."),
        ("duplicates = exact duplicate of ...", "Visually the exact same asset as the listed model(s) (same geometry and texture; may be stored "
                                                "rotated or re-tessellated). Counted ONCE in the header / category counts. Red cell. "
                                                "See select/duplicates.md and duplicates/clusters/*.png."),
        ("duplicates = same geometry as ...", "Same mesh as the listed model(s) but a different texture / colour / label (kept as a separate "
                                              "model, counted separately). Blue cell."),
        ("duplicates (header row)", "How many listed model ids collapse into exact duplicates and how many are texture variants for that role."),
        ("note fields", "round; score (G grasp 0-5, T task fit 0-3, V visual 0-5, R robustness 0-2; score = 2G+2V+T+R); grasp key "
                        "(runlogs/grasp_library_20260905/cands|results|strips/<key>); AG-valid grasps out of 5 and reviewer-clean grasps "
                        "(one_finger ignored); reviewer flags other than one_finger (end_grasp, gap, penetration, fragile, task_violation, "
                        "unsure); scale of scaled variants; cross-task reuse; reviewer note on the grasps and the asset."),
        ("pair", "attachment: '-> female category/model [attachment type]'; IORS: '-> partner category' / '<- target categories'; "
                 "articulation: joint name; disembedding target rows: sizes of the two distractor pools for that model as target; "
                 "disembedding pool rows: '<same|other>-category distractors for <category> targets'; table rows: '<scene>: <room(s)>' "
                 "(+ partner for IORS under)."),
        ("table rows", "Header note: test-spec table, per-scene viable / rejected supports with the sampler's rejection reason. Sub-row note: "
                       "instances written / attempts, support top height, stance edge and base pose, trunk (translate, tilt), arm pose, "
                       "path of the per-run evidence json (template_meta digest, instance summaries)."),
        ("sources", "select/test_instances.json, select/groups/*.json, select/out/*.json, artic/artic_results.csv, task_custom_lists.json, "
                    "runlogs/table_selection_20260911/{inventory.csv,results_summary.csv,candidates_by_task.md}; "
                    "see runlogs/grasp_library_20260905/README.md and select/select_contract.md."),
    ]
    legend.extend(list(extra_legend))
    extra_path = os.path.join(SEL, "legend_extra.json")  # legend items of later selection runs (kept across re-renders)
    if os.path.exists(extra_path):
        with open(extra_path) as f:
            legend.extend(tuple(x) for x in json.load(f) if tuple(x) not in legend)
    lg.append(["item", "meaning"])
    for c in lg[1]:
        c.font = FONT_BOLD
        c.fill = FILL_HEADER
    for k, v in legend:
        lg.append([k, v])
        if k == "selection = pick 1":
            for c in lg[lg.max_row]:
                c.fill = FILL_PICK1
                c.font = FONT_BOLD
        elif k == "selection = pick 2":
            for c in lg[lg.max_row]:
                c.fill = FILL_PICK2
        elif k == "selection = viable":
            for c in lg[lg.max_row]:
                c.fill = FILL_VIABLE
        elif k == "model in bold blue":
            lg[lg.max_row][0].font = FONT_PINNED
        elif k.startswith("duplicates = exact") or k.startswith("duplicates = LOOK-ALIKE"):
            lg[lg.max_row][0].fill = FILL_EXACT
        elif k.startswith("duplicates = same"):
            lg[lg.max_row][0].fill = FILL_TEXVAR
    lg.column_dimensions["A"].width = 26
    lg.column_dimensions["B"].width = 150
    wb.save(XLSX_PATH)

    CSS = """
body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;font-size:12.5px;margin:0;padding:12px;background:#fff;color:#111}
table{border-collapse:collapse;white-space:nowrap}
th,td{border:1px solid #d0d0d0;padding:3px 7px;vertical-align:top;text-align:left}
thead th{position:sticky;top:0;background:#D9E1F2;font-weight:700;z-index:2}
tr.hdr td{background:#D9E1F2;font-weight:700;white-space:normal;min-width:120px}
tr.hdr td.note{max-width:1100px}
tr.pick1 td{background:#C6EFCE;font-weight:700}
tr.pick2 td{background:#FFF2CC}
tr.viable td{background:#E2F0D9}
td.pinned{color:#1F4E79;font-weight:700}
td.exact{background:#F8CBAD !important}
td.texvar{background:#DDEBF7 !important}
td.num{text-align:right;color:#444}
td.note{white-space:normal;max-width:900px}
.legend span{display:inline-block;padding:2px 8px;margin:0 6px 8px 0;border:1px solid #bbb}
"""
    LEGEND_HTML = (
        '<div class="legend">'
        '<span style="background:#D9E1F2;font-weight:700">task / role header (C,D = distinct categories / models)</span>'
        '<span style="background:#C6EFCE;font-weight:700">pick 1</span><span style="background:#FFF2CC">pick 2</span>'
        '<span style="background:#E2F0D9">viable scene table / work surface (sampler-validated, runlogs/table_selection_20260911)</span>'
        '<span style="color:#1F4E79;font-weight:700">model pinned in task_custom_lists.json</span>'
        '<span style="background:#F8CBAD">exact duplicate of another listed model (counted once) / LOOK-ALIKE from the init view (disembedding: never each other\'s distractor)</span>'
        '<span style="background:#DDEBF7">same geometry, texture/colour variant</span>'
        '<span>see the legend sheet of the xlsx and select/duplicates.md</span></div>')
    with open(HTML_PATH, "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>task_object_selection</title>"
                f"<style>{CSS}</style></head><body>{LEGEND_HTML}<table><thead><tr>"
                + "".join(f"<th>{c}</th>" for c in COLUMNS) + "</tr></thead><tbody>")
        for r in rows:
            cls = "hdr" if r["task"] else {"pick 1": "pick1", "pick 2": "pick2", "viable": "viable"}.get(r["selection"], "")
            f.write(f'<tr class="{cls}">')
            for c in COLUMNS:
                v = r[c]
                tdc = []
                if c == "note":
                    tdc.append("note")
                if isinstance(v, int) or (c == "category" and str(v).isdigit()):
                    tdc.append("num")
                if c == "model" and "PINNED" in r["note"]:
                    tdc.append("pinned")
                if c == "duplicates" and (str(v).startswith("exact duplicate") or str(v).startswith("LOOK-ALIKE")):
                    tdc.append("exact")
                elif c == "duplicates" and str(v).startswith("same geometry"):
                    tdc.append("texvar")
                f.write(f'<td class="{" ".join(tdc)}">{_html.escape(str(v))}</td>')
            f.write("</tr>\n")
        f.write("</tbody></table></body></html>")
    print(f"wrote {XLSX_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
