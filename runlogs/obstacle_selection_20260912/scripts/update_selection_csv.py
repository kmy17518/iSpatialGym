#!/usr/bin/env python3
"""Write the obstacle selection into ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` (and regenerate
the .xlsx / .html twins with the shared renderer of runlogs/table_selection_20260911).

Rewrites the ``camera_relocalization-0`` block (idempotent: every existing ``camera_relocalization-0`` header row and
its sub-rows - including the original single ``(no task objects)`` row - is replaced; the block stays at the top of
the sheet, where the problem has always been):

* ``camera_relocalization-0`` / ``obstacle`` - rounds 1-3 categories and every valid model (``selection`` = pick 1 /
  pick 2 / alternate / candidate, ``pair`` = the scenes the category is natural in (``*`` = the category already stands
  in that scene's BEHAVIOR files), ``note`` = round, score (V visual, N naturalness as a mid-path floor obstacle, S size
  fit; score = 2V + 2N + S), bbox, reviewer note; the first row of a category carries the category verdict, the
  ready / reviewed / valid counts, the in-scene rooms and the other tasks using the category);
* ``camera_relocalization-0`` / ``floor`` - the fixed floor of the test spec.

Inputs: ``selection.json`` (select_obstacles.py), ``duplicates.json`` (wa_duplicates.py of the world-alignment run on
the valid models), ``candidate_categories.json`` (bboxes, counts), the task spec (pinned models, floor).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
TABLE_RENDERER = os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts", "update_selection_csv.py")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
RUN = "runlogs/obstacle_selection_20260912"
SCENES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "restaurant_diner",
          "hotel_suite_large", "office_cubicles_right"]

LABEL = "camera_relocalization-0"
PROBLEM = "camera_relocalization-0"

LEGEND_EXTRA = [
    ["obstacle rows (camera_relocalization-0)",
     f"Rounds 1-3 of {RUN} (no grasp test: the obstacle is never touched; it is an optional extra object dropped on the floor "
     "midway along the robot's path in some instances). Every valid model is listed: valid = reviewer scores V >= 3 (visual), "
     "N >= 2 (naturalness as a mid-path floor obstacle in at least one of the 7 candidate scenes), S >= 1 (size fit: 2 = footprint "
     "0.20-0.55 m and height 0.20-1.10 m, 1 = borderline, 0 = blocks a corridor / too small) and rest_ok (stands stably in its "
     "canonical pose, +z up); score = 2V + 2N + S, max 18; picks = the two best valid models per category, alternate = the next one. "
     "Rounds are ranking tiers (round 1 = the 10 best categories with >= 2 valid models, round 2 = the next 5, round 3 = the next 5 "
     "with >= 1 valid model); no overlap rule with other tasks' categories (overlaps are noted). pair = the scenes the category is "
     "natural in ('*' = the category already stands in that scene's BEHAVIOR files; rooms in the first note of the category). "
     "Category blocks ordered by round, then category score. Renders: renders/grids; ranking of the categories not selected: selection.md."],
]


def load_renderer():
    spec = importlib.util.spec_from_file_location("table_update_selection_csv", TABLE_RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bbox_txt(b):
    return f"bbox {b[0]:.2f} x {b[1]:.2f} x {b[2]:.2f} m" if b else ""


def target_rows(sel, dups, pinned, info):
    rows, listed = [], []
    for rnd in ("1", "2", "3"):
        for rec in sel["rounds"][rnd]:
            cat = rec["category"]
            by = {m["model"]: m for m in rec["models"]}
            order = {m: i for i, m in enumerate(rec["picks"])}
            if rec.get("alternate"):
                order[rec["alternate"]] = 2
            valid = sorted(rec["valid_models"], key=lambda m: (order.get(m, 3), -by[m]["score"], m))
            pair = "natural in: " + ", ".join(s + ("*" if s in rec["in_scene"] else "") for s in rec["natural_scenes"])
            first = True
            for model in valid:
                m = by[model]
                selname = {0: "pick 1", 1: "pick 2", 2: "alternate"}.get(order.get(model, 3), "candidate")
                parts = [f"round {rnd}", f"score {m['score']} (V{m['V']} N{m['N']} S{m['S']})"]
                if model in pinned.get(cat, set()):
                    parts.append("PINNED in task_custom_lists.json (role obstacle)")
                bb = bbox_txt(m.get("bbox"))
                if bb:
                    parts.append(bb)
                if first:
                    ci = info.get(cat, {})
                    n_not = len(ci.get("not_rendered", []))
                    parts.append(f"category {rec['verdict']}: {rec['category_note'].strip()} [{rec['n_ready']} ready models, {rec['n_reviewed']} reviewed, "
                                 f"{rec['n_valid']} valid" + (f", {n_not} ready models not rendered (cap)" if n_not else "") + "]")
                    ins = rec.get("in_scene") or {}
                    if ins:
                        parts.append("already in scene files: " + "; ".join(f"{s} ({', '.join(ins[s])})" for s in SCENES if s in ins))
                    u = sorted({f"{x['task']} r{x['round']}" for x in rec["used_by_other_tasks"]})
                    if u:
                        parts.append("category also used by " + ", ".join(u))
                    if rec.get("override"):
                        parts.append("override: " + rec["override"])
                    first = False
                if m.get("note"):
                    parts.append("reviewer: " + m["note"].strip())
                d = dups.get(f"{cat}/{model}", {})
                dtxt = []
                if d.get("exact"):
                    dtxt.append("exact duplicate of " + ", ".join(d["exact"]))
                if d.get("same_geometry"):
                    dtxt.append("same geometry as " + ", ".join(d["same_geometry"]) + " (texture/colour variant; geometry check only)")
                rows.append({"task": "", "role": "", "category": cat, "model": model, "selection": selname, "pair": pair,
                             "duplicates": "; ".join(dtxt), "note": "; ".join(parts)})
                listed.append((cat, model))

    def canon(cat, model, key):
        d = dups.get(f"{cat}/{model}", {})
        others = d.get(key, [])
        return min([model] + others) if others else model

    unique = {(c, canon(c, m, "exact")) for c, m in listed}
    geoms = {(c, dups.get(f"{c}/{m}", {}).get("cluster", canon(c, m, "exact"))) for c, m in listed}
    n_tv = sum(1 for c, m in listed if dups.get(f"{c}/{m}", {}).get("same_geometry"))
    dsum = ""
    if len(unique) != len(listed) or len(geoms) != len(unique):
        dsum = (f"{len(listed)} model ids listed; {len(listed) - len(unique)} exact duplicates counted once -> {len(unique)} models (bracket); "
                f"{n_tv} models are texture/colour variants of another listed model -> {len(geoms)} distinct geometries")
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["category"] == rows[i]["category"]:
            j += 1
        if j > i:
            cat = rows[i]["category"]
            block = [(cat, r["model"]) for r in rows[i:j + 1]]
            n_u = len({canon(c, m, "exact") for c, m in block})
            n_g = len({dups.get(f"{c}/{m}", {}).get("cluster", canon(c, m, "exact")) for c, m in block})
            for r in rows[i + 1:j]:
                r["category"] = ""
            rows[j]["category"] = f"{n_g} ({n_u})"
        i = j + 1
    n_cats = len({c for c, _ in listed})
    dropped = [f"{r['category']} ({r['verdict']}, {r['n_valid']} valid of {r['n_reviewed']} reviewed)" for r in sel["not_selected"]]
    return rows, n_cats, f"{len(geoms)} ({len(unique)})", dsum, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--selection", default=os.path.join(ROOT, "selection.json"))
    ap.add_argument("--duplicates", default=os.path.join(ROOT, "duplicates.json"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    with open(args.selection) as f:
        sel = json.load(f)
    dups = {}
    if os.path.exists(args.duplicates):
        with open(args.duplicates) as f:
            dups = json.load(f).get("flags", {})
    with open(os.path.join(ROOT, "candidate_categories.json")) as f:
        info = json.load(f)
    with open(SPEC) as f:
        spec = json.load(f)
    prob = spec.get(PROBLEM, {})
    pinned = defaultdict(set)
    for cat, ms in (prob.get("roles", {}).get("obstacle", {}).get("models") or {}).items():
        pinned[cat] |= set(ms)
    pin_txt = ", ".join(f"{c}/{m}" for c in sorted(pinned) for m in sorted(pinned[c])) or "none yet (no obstacle role in the spec)"
    floor_txt = "; ".join(f"{s}: floor of {(cfg.get('fixed') or {}).get('floor', {}).get('room', '?')}" for s, cfg in (prob.get("scenes") or {}).items())
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    subs, n_cats, mn, dsum, dropped = target_rows(sel, dups, pinned, info)
    n_cand_cats = len(info)
    n_cand_models = sum(len(v["rendered"]) for v in info.values())
    n_selected_valid = len(subs)
    hdr = {"task": LABEL, "role": "obstacle", "category": str(n_cats), "model": mn, "selection": "", "pair": "", "duplicates": dsum,
           "note": ("BDDL objects: floor.n.01_1, agent.n.01_1 only; goal = head-camera pose (no manipulated object). OBSTACLE = optional extra "
                    "object for some instances: dropped on a free floor cell roughly midway along the init -> goal path so that the robot has "
                    "to steer around it; never touched. Not in problem0.bddl and not placed by the sampler yet (sampler support pending; a "
                    f"spec block is ready in {RUN}/spec_fragments.json: roles.obstacle.models = the picks, _pool = every valid model, "
                    f"_natural_scenes = where each category may be drawn). Test spec pins {pin_txt}. Requirements: natural on the floor of at "
                    "least one of the 7 candidate scenes (house_single_floor, house_double_floor_upper / _lower, Rs_int, restaurant_diner, "
                    "hotel_suite_large, office_cubicles_right) and unremarkable in the middle of a room or corridor (bins, potted plants, "
                    "cardboard boxes, laundry hampers, buckets, luggage, toys, stools, vacuum cleaners, carts), small enough to be dropped on a "
                    "free floor cell without blocking a ~1 m corridor (R1Pro chassis half-diagonal 0.33 m) yet tall enough to be a real obstacle "
                    "for the base and visible to the head camera, and stable in its canonical pose (placed as is, +z up). Selection = "
                    f"{RUN} (no grasp test): {n_cand_cats} candidate categories hand-picked from a geometry scan of every BEHAVIOR category with "
                    "ready models (median footprint 0.12-0.9 m, height 0.10-1.7 m; candidate_scan.json) and the object inventory of the 7 scenes "
                    f"(scene_presence.json); {n_cand_models} ready models (<= 12-16 per category) rendered in their canonical frame (renders/grids) "
                    "and scored by 8 reviewers with review_contract.md: V visual 0-5, N naturalness as a mid-path floor obstacle 0-3 (3 = commonly "
                    "stands in walkable space and is unremarkable mid-path, 2 = plausible but normally against a wall / small clutter, 1 = odd, "
                    "0 = not floor-standing / not in these scenes), S size fit 0-2 (2 = footprint 0.20-0.55 m and height 0.20-1.10 m, 1 = "
                    "borderline, 0 = > 0.80 m footprint or < 0.12 m / < 0.10 m tall / > 1.70 m), rest_ok (stands stably, +z up; balls and "
                    "lying / hanging models fail); score = 2V + 2N + S (max 18); valid = V >= 3, N >= 2, S >= 1, rest_ok. Round 1 = the 10 "
                    "best categories (mean score of the two best valid models; >= 2 valid), round 2 = the next 5 (>= 2 valid), round 3 = the "
                    "next 5 (>= 1 valid); rounds are ranking tiers - the obstacle is never manipulated, so no overlap rule with other tasks' "
                    f"categories (overlaps noted per category). EVERY valid model of the {n_cats} selected categories is listed "
                    f"({n_selected_valid} rows: pick 1 / pick 2 / alternate / candidate, sorted by score); invalid models are not listed. "
                    "pair = the scenes the category is natural in (* = already present in that scene's files). Duplicate check: geometry "
                    "signatures (duplicates.json)." + (" Reviewed but not selected (ranked in selection.md): " + "; ".join(dropped) + "." if dropped else ""))}
    floor_row = {"task": LABEL, "role": "floor", "category": "0", "model": "0", "selection": "", "pair": "", "duplicates": "",
                 "note": f"fixed scene object: {floor_txt or 'house_single_floor: floor of living_room_0'}; other scenes: the floor of the room the instance is sampled in "
                         "(init and goal in one room; an obstacle is dropped on a free cell of that floor)"}
    block = [hdr] + subs + [floor_row]

    # remove the existing block(s) of the problem, remember where the first one was
    out, i, insert_at = [], 0, None
    while i < len(rows):
        r = rows[i]
        if r["task"] == LABEL:
            j = i + 1
            while j < len(rows) and not rows[j]["task"]:
                j += 1
            if insert_at is None:
                insert_at = len(out)
            i = j
            continue
        out.append(r)
        i += 1
    if insert_at is None:
        insert_at = 0  # the problem has always been the first block of the sheet
    out[insert_at:insert_at] = block
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    legend_path = os.path.join(SEL, "legend_extra.json")
    legend = []
    if os.path.exists(legend_path):
        with open(legend_path) as f:
            legend = [list(x) for x in json.load(f)]
    keys = {x[0] for x in LEGEND_EXTRA}
    legend = [x for x in legend if x[0] not in keys] + LEGEND_EXTRA
    with open(legend_path, "w") as f:
        json.dump(legend, f, indent=1)
    print(f"{LABEL}: {len(subs)} obstacle rows ({n_cats} categories, {mn} models); {len(out)} rows -> {args.csv}")
    if not args.no_render:
        load_renderer().render(out)


if __name__ == "__main__":
    main()
