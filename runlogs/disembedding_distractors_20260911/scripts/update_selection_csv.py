#!/usr/bin/env python3
"""Write the validated distractor pools into ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` and
regenerate the ``.xlsx`` / ``.html`` twins (same styling as the sheet's other generators).

Rows touched (everything else is preserved):

* ``disembedding-0 / same_category_pool`` header + one sub-row per pool model, one block per target category
  (``category`` = the target category, ``model`` = the model, ``selection`` = ``pool`` / ``reserve``, ``pair`` = the
  look-alike group, ``duplicates`` = ``LOOK-ALIKE (init view) with ...`` + the geometry note, ``note`` = evidence).
* ``disembedding-0 / distractor_pool`` header + one sub-row per pool model, one block per target category
  (``category`` = the TARGET category, ``model`` = ``<distractor category>/<model>``).
* ``disembedding-0 / target`` sub-rows: ``pair`` = pool sizes for this model as the target (look-alikes excluded),
  ``duplicates`` = ``LOOK-ALIKE (init view) with ...`` prefix refreshed.

Idempotent: previous pool blocks (from any generator) are replaced.
"""
from __future__ import annotations

import argparse
import csv
import html as _html
import json
import os
import re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
XLSX_PATH = os.path.join(SEL, "task_object_selection.xlsx")
HTML_PATH = os.path.join(SEL, "task_object_selection.html")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
RUN = "runlogs/disembedding_distractors_20260911"
TASK = "disembedding-0"


def strip_lookalike(dup: str) -> str:
    """Remove a previous look-alike annotation, keep the geometry-duplicate note."""
    parts = [p.strip() for p in re.split(r"\s*\|\s*", dup or "") if p.strip()]
    parts = [p for p in parts if not p.startswith("LOOK-ALIKE")]
    return "; ".join(parts) if parts else ""


def fmt_group(members, me=None):
    return ", ".join(m for m in members if m != me)


def evidence(r):
    parts = []
    if r.get("validation_mode") == "prelim":
        parts.append("sampler validation IN PROGRESS (preliminary pool from the render check + review)")
    elif r.get("validation_mode") == "render_only":
        parts.append("NOT sampler-validated (the category's target template failed to set up on the test table; render check only)")
    elif r.get("validation_status") == "ok":
        parts.append("sampler-validated: " + r["validation"])
    elif r.get("validation_status") == "init_ok":
        parts.append("sampler: " + r["validation"])
    else:
        parts.append(f"sampler: {r.get('validation')}")
    parts.append("head-camera render: " + str(r.get("render")))
    sel = r.get("selection")
    if sel in ("pick 1", "pick 2", "alternate", "candidate"):
        parts.append(f"disembedding grasp round: {sel}" + (f", score {r['score']}" if r.get("score") is not None else "") + (f", AG-valid {r['ag_valid']}/5" if r.get("ag_valid") is not None else ""))
    elif sel == "untested":
        parts.append("ready model, not in the grasp-library round (distractors are not grasped)")
    elif sel == "excluded":
        parts.append("grasp round: no AG-valid grasp (irrelevant for a distractor)")
    if r.get("bbox"):
        bb = r["bbox"]
        parts.append(f"bbox {bb[0] * 100:.0f} x {bb[1] * 100:.0f} x {bb[2] * 100:.0f} cm")
    if r.get("reviewer"):
        parts.append("reviewer: " + r["reviewer"])
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", default=os.path.join(ROOT, "distractor_pools.json"))
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    pools = json.load(open(args.pools))
    cats = pools["categories"]
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # ---- locate the disembedding blocks
    def is_header(r):
        return bool(r["task"])

    idx_target = idx_same = idx_diff = None
    for i, r in enumerate(rows):
        if r["task"] == TASK and r["role"] == "target":
            idx_target = i
        if r["task"] == TASK and r["role"] == "same_category_pool":
            idx_same = i
        if r["task"] == TASK and r["role"] == "distractor_pool":
            idx_diff = i
    assert idx_target is not None, "no disembedding target block"

    def block_end(i):
        j = i + 1
        while j < len(rows) and not is_header(rows[j]):
            j += 1
        return j

    # ---- 1. target rows: pair / duplicates annotations
    t_end = block_end(idx_target)
    cur_cat = None
    order_of_cats = []
    for r in rows[idx_target + 1:t_end]:
        if r["category"] and not re.match(r"^\d+ \(\d+\)$", r["category"]):
            cur_cat = r["category"]
            if cur_cat not in order_of_cats:
                order_of_cats.append(cur_cat)
        d = cats.get(cur_cat)
        if d is None:
            continue
        m = r["model"]
        same_rows = d["same_category"]
        grp = next((g for g in d["lookalike_groups"] if m in g["members"]), None)
        n_same_pool = sum(1 for x in same_rows if x.get("pool") == "pool")
        excluded = [x["model"] for x in same_rows if x.get("pool") == "pool" and grp and x["model"] in grp["members"] and x["model"] != m]
        in_pool = any(x["model"] == m and x.get("pool") == "pool" for x in same_rows)
        n_groups = d["n_same_groups_selected"] - (1 if (excluded or in_pool) else 0)
        n_avail = n_same_pool - len(excluded) - (1 if in_pool else 0)
        r["pair"] = (f"as target: same-category pool {max(n_avail, 0)} models / {max(n_groups, 0)} look-alike groups"
                     + (f" (excluded: itself + look-alikes {fmt_group(excluded, m)})" if excluded else " (excluded: itself)")
                     + f"; other-category pool {d['n_other_selected']} models; see the pool rows below"
                     + ("; NOTE: pools render-checked only (template setup failed)" if d["template_setup_failed"] else ""))
        dup = strip_lookalike(r["duplicates"])
        if grp:
            others = fmt_group(grp["members"], m)
            r["duplicates"] = f"LOOK-ALIKE (init view) with {others}" + (f": {grp['reason']}" if grp.get("reason") else "") + (" | " + dup if dup else "")
        else:
            r["duplicates"] = dup

    # ---- 2. rebuild the two pool blocks
    def drop_block(i):
        if i is None:
            return
        j = block_end(i)
        del rows[i:j]

    # remove old blocks (from the highest index down so indices stay valid)
    for i in sorted([x for x in (idx_same, idx_diff) if x is not None], reverse=True):
        drop_block(i)
    # insert position: right after the target block
    ins = block_end(idx_target)
    ordered_cats = [c for c in order_of_cats if c in cats] + [c for c in cats if c not in order_of_cats]

    n_same_models = sum(d["n_same_models_selected"] for d in cats.values())
    n_same_groups = sum(d["n_same_groups_selected"] for d in cats.values())
    n_other = sum(d["n_other_selected"] for d in cats.values())
    n_la = sum(len(d["lookalike_groups"]) for d in cats.values())
    short = [c for c, d in cats.items() if d["short"]]
    failed = [c for c, d in cats.items() if d["template_setup_failed"]]
    method = (f"Method ({RUN}): candidates = every ready model of the category (same pool) and ready models of hand-curated related categories "
              f"(other pool, closest first: e.g. forks / knives for spoons, rice / sugar boxes for cereal boxes), size-filtered for the source tray; "
              f"each candidate was rendered alone on the source tray from the robot's head camera at the disembedding init stance (house_single_floor "
              f"dining table, test trays; renders/) and settled for 60 physics steps; reviewer agents grouped same-category models that cannot be told "
              f"apart in that view (review_contract.md, groups/; merged with the exact-duplicate groups of select/duplicates.md); every candidate was then "
              f"validated with the instance sampler (validate_distractors.py: DisembeddingSampler.setup with the category's test pick as target + "
              f"TaskSampler.run(1) with the candidate as the only distractor - sample, settle, BEHAVIOR checks, :init true, goal false, occlusion "
              f"measurement, goal feasibility; results/). Pool = {pools['pool_size']} candidates per target category: same-category look-alike groups "
              f"first (a group counts as ONE candidate, at most {pools['same_cap']} groups; all validated members of a selected group are listed so the "
              f"group exclusion works whichever member is the target), related-category models fill the rest (at most {pools['per_category']} per "
              f"category). selection = pool (in the 20) / reserve (validated, beyond the 20). Spec form: {RUN}/distractors_section.json "
              f"(disembedding-0.distractors: pools.<category>.same_category / other_models, lookalike_groups).")
    hdr_same = {"task": TASK, "role": "same_category_pool", "category": str(len(cats)), "model": f"{n_same_groups} ({n_same_models})", "selection": "", "pair": "",
                "duplicates": f"{n_la} look-alike groups (init view) across the same-category pools",
                "note": ("Same-category distractor pool per target category (blocks below: category = target category; model count = look-alike "
                         "groups (models)). At sampling time the target itself and every model in its LOOK-ALIKE group are removed (spec lookalike_groups). "
                         + method
                         + (f" SHORT of 20: {', '.join(short)}." if short else "")
                         + (f" Target template failed on the test table (pools render-checked only): {', '.join(failed)}." if failed else ""))}
    hdr_diff = {"task": TASK, "role": "distractor_pool", "category": str(len(cats)), "model": f"{n_other} ({n_other})", "selection": "", "pair": "", "duplicates": "",
                "note": ("Other-category distractor pool per target category (blocks below: category = TARGET category, model = <distractor category>/<model>, "
                         "pair = why the category is related, tier 1 = closest). Drawn when generation.same_category_prob does not select the same-category pool. "
                         + method)}
    new_rows = [hdr_same]
    for cat in ordered_cats:
        d = cats[cat]
        sel_rows = [r for r in d["same_category"] if r.get("pool") in ("pool", "reserve")]
        sel_rows.sort(key=lambda r: (0 if r["pool"] == "pool" else 1, int(r["group"].split("__g")[1]), r["model"]))
        block = []
        for r in sel_rows:
            grp = next((g for g in d["lookalike_groups"] if r["model"] in g["members"]), None)
            gid = r["group"].split("__")[1]
            pair = f"same-category distractor for {cat} targets; look-alike group {gid}" + (f" = {{{', '.join(grp['members'])}}}" if grp else " (distinct from every other model at the init view)")
            dup = ""
            if grp:
                dup = f"LOOK-ALIKE (init view) with {fmt_group(grp['members'], r['model'])}" + (f": {grp['reason']}" if grp.get("reason") else "")
            geo = []
            if r.get("exact_group"):
                geo.append("exact duplicate of " + ", ".join(m for m in r["exact_group"] if m != r["model"]))
            elif r.get("geom_cluster"):
                geo.append(f"same geometry cluster {r['geom_cluster']} (texture/colour variant)")
            if geo:
                dup = (dup + " | " if dup else "") + "; ".join(geo)
            block.append({"task": "", "role": "", "category": cat, "model": r["model"], "selection": r["pool"], "pair": pair, "duplicates": dup, "note": evidence(r)})
        if not block:
            block.append({"task": "", "role": "", "category": cat, "model": "(none)", "selection": "", "pair": f"no validated same-category distractor for {cat}", "duplicates": "",
                          "note": "; ".join(f"{r['model']}: {r.get('unusable') or r.get('validation')}" for r in d["same_category"])[:900]})
        for r in block[1:-1]:
            r["category"] = ""
        if len(block) > 1:
            block[-1]["category"] = f"{d['n_same_groups_selected']} ({d['n_same_models_selected']})"
        new_rows.extend(block)
    new_rows.append(hdr_diff)
    for cat in ordered_cats:
        d = cats[cat]
        sel_rows = [r for r in d["other_category"] if r.get("pool") in ("pool", "reserve")]
        sel_rows.sort(key=lambda r: (0 if r["pool"] == "pool" else 1, r.get("tier") or 9, r["category"], r["model"]))
        block = []
        for r in sel_rows:
            pair = f"other-category distractor for {cat} targets; tier {r.get('tier')} related category ({r['category']})"
            dup = ""
            if r.get("exact_group"):
                dup = "exact duplicate of " + ", ".join(m for m in r["exact_group"] if m != r["model"])
            elif r.get("geom_cluster"):
                dup = f"same geometry cluster {r['geom_cluster']} (texture/colour variant)"
            block.append({"task": "", "role": "", "category": cat, "model": f"{r['category']}/{r['model']}", "selection": r["pool"], "pair": pair, "duplicates": dup, "note": evidence(r)})
        if not block:
            block.append({"task": "", "role": "", "category": cat, "model": "(none)", "selection": "", "pair": f"no validated other-category distractor for {cat}", "duplicates": "", "note": ""})
        for r in block[1:-1]:
            r["category"] = ""
        if len(block) > 1:
            block[-1]["category"] = f"{d['n_other_selected']} ({d['n_other_selected']})"
        new_rows.extend(block)
    rows[ins:ins] = new_rows
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows ({len(new_rows)} pool rows) -> {args.csv}")
    if not args.no_render:
        render(rows)


# ----------------------------------------------------------------------------------------------------------------
# xlsx / html twins (styling of the sheet's generators; pool rows pale green, reserve rows plain)
# ----------------------------------------------------------------------------------------------------------------


def render(rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    FILL_HEADER = PatternFill("solid", fgColor="D9E1F2")
    FILL_PICK1 = PatternFill("solid", fgColor="C6EFCE")
    FILL_PICK2 = PatternFill("solid", fgColor="FFF2CC")
    FILL_VIABLE = PatternFill("solid", fgColor="E2F0D9")
    FILL_EXACT = PatternFill("solid", fgColor="F8CBAD")
    FILL_TEXVAR = PatternFill("solid", fgColor="DDEBF7")
    FILL_LOOKALIKE = PatternFill("solid", fgColor="FCE4D6")
    FONT_BOLD = Font(bold=True)
    FONT_PINNED = Font(bold=True, color="1F4E79")
    WIDTHS = {"task": 46, "role": 20, "category": 28, "model": 22, "selection": 12, "pair": 62, "duplicates": 58, "note": 160}

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
        fill = {"pick 1": FILL_PICK1, "pick 2": FILL_PICK2, "viable": FILL_VIABLE, "pool": FILL_VIABLE}.get(sel)
        if fill:
            for c in cells:
                c.fill = fill
        if sel == "pick 1":
            for c in cells:
                c.font = FONT_BOLD
        if "PINNED" in r["note"]:
            cells[COLUMNS.index("model")].font = FONT_PINNED
        d = r["duplicates"]
        if d.startswith("LOOK-ALIKE"):
            cells[COLUMNS.index("duplicates")].fill = FILL_LOOKALIKE
        elif d.startswith("exact duplicate"):
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
        ("Layout", "A task / role header row (blue-grey, bold) carries the BDDL object list and role requirements in 'note'; its category / model cells hold the "
                   "number of distinct categories / models listed under the role (models as 'M (N)', see below). The rows below it (task / role blank) are the "
                   "objects for that role: category, model, selection, pair, note."),
        ("category column", "Within a role, category blocks are ordered by their number of distinct geometries (descending), then distinct models, then name. A "
                            "category's name appears on the first row of its block only; the last row of the block shows 'M (N)': M = distinct geometries of "
                            "that category (exact duplicates AND texture/colour variants / the engraved alphabet blocks counted once), N = distinct models "
                            "(only exact duplicates counted once). Single-row categories just show the name. Disembedding pool blocks: category = TARGET "
                            "category, last row 'G (N)' = look-alike groups (models) in the pool."),
        ("model count in header rows", "'M (N)' with the same meaning as above, over all models listed under the role."),
        ("selection = pick 1", "Test-instance pick 1 for the (task, category) group (green row, bold)."),
        ("selection = pick 2", "Test-instance pick 2 (yellow row)."),
        ("selection = alternate", "3rd choice of the selection reviewers, not selected (no fill; a training candidate like the rest)."),
        ("selection = candidate", "Any other model of the category with >= 1 AG-valid grasp in the grasp-library run for this task (training candidate). Sorted by "
                                  "selection score. Models with no AG-valid grasp are not listed; categories where no model passed are named in the role header note."),
        ("selection = viable", "Articulated object whose joint the robot actually opened in the articulation pass; or (table rows, pale green) a scene table / work "
                               "surface on which the instance sampler generated >= 1 validated instance of the task (runlogs/table_selection_20260911)."),
        ("selection = pool", "Disembedding distractor pool member (pale green): one of the 20 candidates per target category, validated by the instance sampler "
                             "(one instance with it as the distractor of the category's test target) and by the head-camera render check "
                             "(runlogs/disembedding_distractors_20260911)."),
        ("selection = reserve", "Validated disembedding distractor beyond the 20 of the pool (look-alike groups beyond the first 10, or related-category models "
                                "beyond the fill); a drop-in replacement."),
        ("selection = spec", "Object pinned in ispatialgym/assets/task_specs/task_custom_lists.json for a role without a grasp test (trays, blocks, keyboard, painting, "
                             "partners, the test-round table)."),
        ("selection = partner", "Partner category / model named by the selection (not pinned in the spec)."),
        ("model in bold blue", "Model is pinned in task_custom_lists.json for this problem (the note says in which role)."),
        ("duplicates = LOOK-ALIKE (init view) with ...", "Models of the same category that cannot be told apart from the robot's head camera at the disembedding init "
                                                        "stance (reviewer agents on renders from that view; runlogs/disembedding_distractors_20260911/grids/same, groups/). "
                                                        "Never each other's distractor: when one is the target the others are removed from the pools (spec lookalike_groups). "
                                                        "Orange cell. A look-alike group counts as ONE candidate of the 20."),
        ("duplicates = exact duplicate of ...", "Visually the exact same asset as the listed model(s) (same geometry and texture; may be stored rotated or re-tessellated). "
                                                "Counted ONCE in the header / category counts. Red cell. See select/duplicates.md and duplicates/clusters/*.png."),
        ("duplicates = same geometry as ...", "Same mesh as the listed model(s) but a different texture / colour / label (kept as a separate model, counted separately). Blue cell."),
        ("duplicates (header row)", "How many listed model ids collapse into exact duplicates and how many are texture variants for that role."),
        ("note fields", "round; score (G grasp 0-5, T task fit 0-3, V visual 0-5, R robustness 0-2; score = 2G+2V+T+R); grasp key "
                        "(runlogs/grasp_library_20260905/cands|results|strips/<key>); AG-valid grasps out of 5 and reviewer-clean grasps (one_finger ignored); reviewer "
                        "flags other than one_finger; scale of scaled variants; cross-task reuse; reviewer note on the grasps and the asset. Pool rows: sampler "
                        "validation (instance written / rejections, occluded or unoccluded variant, occlusion ratio, footprint fraction of the tray), head-camera "
                        "render (segmentation pixels, settled tilt), grasp-round status, bbox."),
        ("pair", "attachment: '-> female category/model [attachment type]'; IORS: '-> partner category' / '<- target categories'; articulation: joint name; "
                 "disembedding target rows: pool sizes when this model is the target (look-alikes excluded); disembedding pool rows: which targets they distract, "
                 "look-alike group / related-category tier; table rows: '<scene>: <room(s)>' (+ partner for IORS under)."),
        ("sources", "select/test_instances.json, select/groups/*.json, select/out/*.json, artic/artic_results.csv, task_custom_lists.json, "
                    "runlogs/table_selection_20260911/{inventory.csv,results_summary.csv,candidates_by_task.md}, "
                    "runlogs/disembedding_distractors_20260911/{candidates.json,distractor_pools.json,distractors_section.json,results/,renders/,grids/,groups/}; "
                    "see runlogs/grasp_library_20260905/README.md and select/select_contract.md."),
    ]
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
        elif k in ("selection = viable", "selection = pool"):
            for c in lg[lg.max_row]:
                c.fill = FILL_VIABLE
        elif k == "model in bold blue":
            lg[lg.max_row][0].font = FONT_PINNED
        elif k.startswith("duplicates = LOOK-ALIKE"):
            lg[lg.max_row][0].fill = FILL_LOOKALIKE
        elif k.startswith("duplicates = exact"):
            lg[lg.max_row][0].fill = FILL_EXACT
        elif k.startswith("duplicates = same"):
            lg[lg.max_row][0].fill = FILL_TEXVAR
    lg.column_dimensions["A"].width = 30
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
tr.pool td{background:#E2F0D9}
td.pinned{color:#1F4E79;font-weight:700}
td.exact{background:#F8CBAD !important}
td.texvar{background:#DDEBF7 !important}
td.lookalike{background:#FCE4D6 !important}
td.num{text-align:right;color:#444}
td.note{white-space:normal;max-width:900px}
.legend span{display:inline-block;padding:2px 8px;margin:0 6px 8px 0;border:1px solid #bbb}
"""
    LEGEND_HTML = (
        '<div class="legend">'
        '<span style="background:#D9E1F2;font-weight:700">task / role header (C,D = distinct categories / models)</span>'
        '<span style="background:#C6EFCE;font-weight:700">pick 1</span><span style="background:#FFF2CC">pick 2</span>'
        '<span style="background:#E2F0D9">viable scene table (runlogs/table_selection_20260911) / disembedding distractor pool member (runlogs/disembedding_distractors_20260911)</span>'
        '<span style="color:#1F4E79;font-weight:700">model pinned in task_custom_lists.json</span>'
        '<span style="background:#FCE4D6">LOOK-ALIKE from the init view (never each other\'s distractor)</span>'
        '<span style="background:#F8CBAD">exact duplicate of another listed model (counted once)</span>'
        '<span style="background:#DDEBF7">same geometry, texture/colour variant</span>'
        '<span>see the legend sheet of the xlsx and select/duplicates.md</span></div>')
    with open(HTML_PATH, "w") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>task_object_selection</title>"
                f"<style>{CSS}</style></head><body>{LEGEND_HTML}<table><thead><tr>"
                + "".join(f"<th>{c}</th>" for c in COLUMNS) + "</tr></thead><tbody>")
        for r in rows:
            cls = "hdr" if r["task"] else {"pick 1": "pick1", "pick 2": "pick2", "viable": "viable", "pool": "pool"}.get(r["selection"], "")
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
                if c == "duplicates" and str(v).startswith("LOOK-ALIKE"):
                    tdc.append("lookalike")
                elif c == "duplicates" and str(v).startswith("exact duplicate"):
                    tdc.append("exact")
                elif c == "duplicates" and str(v).startswith("same geometry"):
                    tdc.append("texvar")
                f.write(f'<td class="{" ".join(tdc)}">{_html.escape(str(v))}</td>')
            f.write("</tr>\n")
        f.write("</tbody></table></body></html>")
    print(f"wrote {XLSX_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
