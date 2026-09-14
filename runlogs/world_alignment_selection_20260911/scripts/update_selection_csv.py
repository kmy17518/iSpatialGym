#!/usr/bin/env python3
"""Write the world-alignment selection into ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` (and
regenerate the .xlsx / .html twins with the shared renderer of runlogs/table_selection_20260911).

Rewrites three blocks (idempotent: the rows between a header row and the next header row are replaced):

* ``world_alignment-0 (table_surface)`` / ``target`` - rounds 1-3 categories and every valid model (``selection`` =
  pick 1 / pick 2 / alternate / candidate, ``pair`` = canonical frame ``long axis .., front ..``, ``note`` = round, score
  (V visual, O orientation readability, F table fit; score = 2V + 2O + F), reviewer note; the first row of a category
  carries the category verdict and the ready / reviewed / valid counts);
* ``world_alignment-1 (world_vertical)`` / ``target`` - the same with U (up readability) / H (hangability) and ``pair``
  = ``front .., up .., thin ..``;
* ``world_alignment-1 (world_vertical)`` / ``wall`` - one ``viable`` row per validated wall region of the wall survey
  (``model`` = wall object name, ``pair`` = ``<scene>: <room>``, ``note`` = region geometry + instance evidence).

Inputs: ``selection.json`` (select_wa_objects.py), ``duplicates.json`` (wa_duplicates.py), ``wall_candidates.json``
(aggregate_walls.py), the task spec (pinned models).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
TABLE_RENDERER = os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts", "update_selection_csv.py")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
RUN = "runlogs/world_alignment_selection_20260911"
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]

TASK_LABEL = {"wa0": "world_alignment-0 (table_surface)", "wa1": "world_alignment-1 (world_vertical)"}
PROBLEM = {"wa0": "world_alignment-0", "wa1": "world_alignment-1"}

LEGEND_EXTRA = [
    ["world-alignment target rows", "Rounds 1-3 of runlogs/world_alignment_selection_20260911 (no grasp test: the objects are pushed). Every valid model is "
                                    "listed: valid = reviewer scores V >= 3 (visual), O/U >= 2 (orientation / up readability), F/H >= 1 (table fit / hangability); "
                                    "score = 2V + 2O + F (table surface) or 2V + 2U + H (world vertical), max 18; picks = the two best valid models per category, "
                                    "alternate = the next one. pair = canonical frame read from the renders: table surface 'long axis x|y, front +x|-x|+y|-y' "
                                    "(side that faces the robot when aligned); world vertical 'front <axis out of the wall>, up <axis>, thin <axis>'. "
                                    "Category blocks are ordered by round, then category score. Renders: renders/grids (pyrender, canonical frame + axis gizmo)."],
    ["selection = viable (wall rows)", "world_alignment-1 wall row: a wall region where the sampler generated a validated instance (survey of every indoor room of "
                                       "the 7 scenes, runlogs/world_alignment_selection_20260911/wall_candidates.{json,md} and maps/<scene>_walls.png). "
                                       "model = wall object, pair = scene: room, note = region segment p0 -> p1 (world xy at hang height), outward normal, "
                                       "length, largest object that fits, arc bases / pushing-capable bases, the instance's robot base, and the evidence path."],
]


def load_renderer():
    spec = importlib.util.spec_from_file_location("table_update_selection_csv", TABLE_RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_frame(task, m):
    if task == "wa0":
        s = f"long axis {m.get('long_axis', '?')}, front {m.get('front_axis', '?')}"
        if m.get("rest_ok") is False:
            s += " (canonical pose is not the resting pose)"
        return s
    return f"front {m.get('front_axis', '?')}, up {m.get('up_axis', '?')}, thin {m.get('thin_axis', '?')}"


def target_rows(task, sel, dups, pinned, used):
    """(sub-rows, n_categories, 'M (N)', duplicates summary, dropped text) for one task."""
    t = sel["tasks"][task]
    a, b = ("O", "F") if task == "wa0" else ("U", "H")
    rows = []
    listed = []
    for rnd in ("1", "2", "3"):
        for rec in t["rounds"][rnd]:
            cat = rec["category"]
            by = {m["model"]: m for m in rec["models"]}
            order = {m: i for i, m in enumerate(rec["picks"])}
            if rec.get("alternate"):
                order[rec["alternate"]] = 2
            valid = sorted(rec["valid_models"], key=lambda m: (order.get(m, 3), -by[m]["score"], m))
            first = True
            for model in valid:
                m = by[model]
                selname = {0: "pick 1", 1: "pick 2", 2: "alternate"}.get(order.get(model, 3), "candidate")
                parts = [f"round {rnd}", f"score {m['score']} (V{m['V']} {a}{m[a]} {b}{m[b]})"]
                if model in pinned.get((task, cat), set()):
                    parts.append("PINNED in task_custom_lists.json (role target)")
                if first:
                    parts.append(f"category {rec['verdict']}: {rec['category_note'].strip()} [{rec['n_ready']} ready models, {rec['n_reviewed']} reviewed, "
                                 f"{rec['n_valid']} valid" + (f", {rec['n_unreviewed']} ready models not reviewed" if rec["n_unreviewed"] else "") + "]")
                    u = sorted({f"{x['task']} r{x['round']}" for x in rec["used_by_other_tasks"]})
                    if u:
                        parts.append("category also used by " + ", ".join(u))
                    first = False
                if m.get("note"):
                    parts.append("reviewer: " + m["note"].strip())
                d = dups.get(f"{cat}/{model}", {})
                dtxt = []
                if d.get("exact"):
                    dtxt.append("exact duplicate of " + ", ".join(d["exact"]))
                if d.get("same_geometry"):
                    dtxt.append("same geometry as " + ", ".join(d["same_geometry"]) + " (texture/colour variant; geometry check only)")
                rows.append({"task": "", "role": "", "category": cat, "model": model, "selection": selname, "pair": fmt_frame(task, m),
                             "duplicates": "; ".join(dtxt), "note": "; ".join(parts)})
                listed.append((cat, model))
    # counts (exact duplicates counted once; texture variants counted once for the geometry count)
    def canon(cat, model, key):
        d = dups.get(f"{cat}/{model}", {})
        others = d.get(key, [])
        return min([model] + others) if others else model

    unique = {(c, canon(c, m, "exact")) for c, m in listed}
    geoms = set()
    for c, m in listed:
        d = dups.get(f"{c}/{m}", {})
        geoms.add((c, d.get("cluster", canon(c, m, "exact"))))
    n_tv = sum(1 for c, m in listed if dups.get(f"{c}/{m}", {}).get("same_geometry"))
    dsum = ""
    if len(unique) != len(listed) or len(geoms) != len(unique):
        dsum = (f"{len(listed)} model ids listed; {len(listed) - len(unique)} exact duplicates counted once -> {len(unique)} models (bracket); "
                f"{n_tv} models are texture/colour variants of another listed model -> {len(geoms)} distinct geometries")
    # category blocks: name on the first row, 'M (N)' on the last row
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
    dropped = [f"{r['category']} ({r['verdict']}, {r['n_valid']} valid of {r['n_reviewed']} reviewed)" for r in t["not_selected"]]
    return rows, n_cats, f"{len(geoms)} ({len(unique)})", dsum, dropped


def wall_rows(walls):
    rows = []
    per_scene = []
    n_objects = set()
    for scene in SCENES:
        sc = walls["scenes"].get(scene)
        if not sc:
            per_scene.append(f"{scene}: not surveyed")
            continue
        ok_rooms, bad_rooms = [], []
        for room, rr in sorted(sc["rooms"].items()):
            regs = rr.get("regions") or []
            val = [r for r in regs if r.get("validated")]
            if val:
                ok_rooms.append(f"{room} ({len(val)} region{'s' if len(val) > 1 else ''}, {sum(r['length_m'] for r in val):.1f} m)")
            else:
                if rr.get("status") != "ok":
                    why = rr.get("status")
                elif not regs:
                    why = f"no viable spot ({rr.get('n_spots', 0)} spots, {rr.get('n_target_ok', 0)} flat+clear, {rr.get('n_viable', 0)} with a base)"
                else:
                    rj = set()
                    for r in regs:
                        for i in r.get("instances", []):
                            rj |= set((i.get("rejections") or {}).keys())
                    why = f"{len(regs)} region(s), no instance ({', '.join(sorted(rj - {'scene_reset'})) or 'not tested'})"
                bad_rooms.append(f"{room} [{why}]")
            for r in val:
                inst = next((i for i in r.get("instances", []) if i.get("n_written")), None)
                ii = (inst or {}).get("instance") or {}
                parts = [f"region {r['id']}: ({r['p0'][0]:.2f}, {r['p0'][1]:.2f}) -> ({r['p1'][0]:.2f}, {r['p1'][1]:.2f}) at z {r['hang_z']:.2f} m, "
                         f"normal ({r['normal'][0]:.2f}, {r['normal'][1]:.2f}), length {r['length_m']:.2f} m",
                         f"objects up to {max(0.0, r['max_patch_m'] - 0.10):.1f} m fit (flat + unobstructed patch {r['max_patch_m']:.1f} m)",
                         f"{r['n_spots']} spots, {r['n_bases']} collision-free arc bases, {r['n_reach']} with pushing reach"]
                if ii:
                    bp = ii.get("base_pose2d") or [0, 0, 0]
                    parts.append(f"instance: robot base ({bp[0]:.2f}, {bp[1]:.2f}, yaw {bp[2]:.2f}), arc radius {(ii.get('arc') or {}).get('radius', 0):.2f} m, "
                                 f"trunk {tuple(round(v, 2) for v in (ii.get('trunk') or []))}, deviation {ii.get('deviation_deg', 0):.0f} deg, "
                                 f"{ii.get('n_visible_objects', '?')} objects visible")
                parts.append(f"evidence {RUN}/results/{scene}__{room}.json" + (f" ({inst.get('work_dir')})" if inst and inst.get("work_dir") else ""))
                rows.append({"task": "", "role": "", "category": "walls", "model": r["wall_name"], "selection": "viable",
                             "pair": f"{scene}: {room}", "duplicates": "", "note": "; ".join(parts)})
                n_objects.add((scene, r["wall_name"]))
        per_scene.append(f"{scene}: viable {', '.join(ok_rooms) if ok_rooms else 'NONE'}" + (f"; rejected {', '.join(bad_rooms)}" if bad_rooms else ""))
    return rows, len(n_objects), per_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--selection", default=os.path.join(ROOT, "selection.json"))
    ap.add_argument("--duplicates", default=os.path.join(ROOT, "duplicates.json"))
    ap.add_argument("--walls", default=os.path.join(ROOT, "wall_candidates.json"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    with open(args.selection) as f:
        sel = json.load(f)
    dups = {}
    if os.path.exists(args.duplicates):
        with open(args.duplicates) as f:
            dups = json.load(f).get("flags", {})
    walls = None
    if os.path.exists(args.walls):
        with open(args.walls) as f:
            walls = json.load(f)
    with open(SPEC) as f:
        spec = json.load(f)
    pinned = defaultdict(set)
    for task, prob in PROBLEM.items():
        for cat, ms in (spec.get(prob, {}).get("roles", {}).get("target", {}).get("models") or {}).items():
            pinned[(task, cat)] |= set(ms)
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    method = (f"Selection = {RUN} (no grasp test: the object is pushed). Every ready model of the candidate categories was rendered in its canonical "
              f"frame (renders/grids, up to 16 per category) and scored by 8 reviewers with review_contract.md: V visual 0-5, "
              "{ab} 0-3, {cd} 0-2; score = 2V + 2{a} + {c} (max 18); valid = V >= 3, {a} >= 2, {c} >= 1. Round 1 = 10 categories used by no "
              "other task (>= 2 valid models), round 2 = 5 more (may overlap other tasks' round-1 categories, >= 2 valid), round 3 = 5 more (any "
              "overlap, 1 valid model enough). EVERY valid model is listed (pick 1 / pick 2 / alternate / candidate, sorted by score; picks = the two "
              "best, pair rule prefers a visually different second model); invalid models are not listed. pair = canonical frame read from the "
              "renders ({frame}). Blocks ordered by round, then category score (mean of the two best valid models). Duplicate check: geometry "
              "signatures of the listed models (duplicates.json; exact = identical geometry and identical diffuse textures).")
    notes = {
        "wa0": ("BDDL objects: 1 target ontop the table (goal = orientation only, released), table, floor, agent. The object rests on the table with a "
                "vertical hinge through its CoM (TRO revolute constraint) and is pushed from a deviation of 20-70 deg back to the aligned yaw (long axis "
                "parallel to the table's front edge, front toward the robot); it must therefore have ONE readable yaw (asymmetric print / keys / handle), "
                "rest flat and stable, and fit on a table (0.12-0.50 m long, <= 0.10 m high). Test spec pins keyboard/hgtbot and keyboard/injrgc "
                "(front_axis +x, long_axis y). The frames below differ per category (and sometimes per model): sampler >= 1.4.1 takes "
                "generation.front_axis / long_axis as dicts keyed by '<category>/<model>' (ready-made in spec_fragments.json). "
                + method.format(ab="O orientation readability", cd="F table fit", a="O", c="F", frame="long axis x|y, front = side facing the robot when aligned")),
        "wa1": ("BDDL objects: 1 target hinged on a wall through its CoM (TRO revolute constraint, axis = wall normal; goal = straight orientation, "
                "released), wall, floor, agent. Hung with its centre at 1.45 m, started 15-45 deg off in the wall plane and pushed back by a lower corner "
                "from an arc 0.6-0.95 m in front; it must therefore have ONE readable upright orientation, be flat-backed and thin, 0.25-0.9 m across "
                "and natural on a wall. Test spec pins painting/dacihj and painting/ljabhq (front_axis +x, up +z, thin x = the sampler's assumed frame). "
                "Categories whose canonical frame lies flat (thin z: picture, license_plate, guitar, hanger, dartboard, ...) are hung with sampler "
                ">= 1.4.1's generation.up_axis / front_axis dicts keyed by '<category>/<model>' (ready-made in spec_fragments.json); the frames are in pair. "
                + method.format(ab="U up readability", cd="H hangability", a="U", c="H", frame="front = axis out of the wall, up = axis that points up when hung, thin = thinnest bbox axis")),
    }

    out = []
    i = 0
    n_done = 0
    while i < len(rows):
        r = rows[i]
        label, role = r["task"], r["role"]
        task = next((t for t, lab in TASK_LABEL.items() if lab == label), None)
        if task and role == "target":
            j = i + 1
            while j < len(rows) and not rows[j]["task"]:
                j += 1
            subs, n_cats, mn, dsum, dropped = target_rows(task, sel, dups, pinned, None)
            hdr = dict(r)
            hdr.update({"category": str(n_cats), "model": mn, "duplicates": dsum,
                        "note": notes[task] + (" Reviewed but not selected: " + "; ".join(dropped) + "." if dropped else "")})
            out.append(hdr)
            out.extend(subs)
            i = j
            n_done += 1
            continue
        if task == "wa1" and role == "wall" and walls is not None:
            j = i + 1
            while j < len(rows) and not rows[j]["task"]:
                j += 1
            subs, n_obj, per_scene = wall_rows(walls)
            hdr = dict(r)
            hdr.update({"category": "1" if subs else "0", "model": str(n_obj),
                        "note": ("fixed scene wall (wall.n.01_1, constraint parent). Test spec: house_single_floor living_room_0, wall found by the sampler. "
                                 f"Wall survey ({RUN}/scripts/survey_walls.py): every indoor room of the 7 scenes, wall spots every 0.25 m at 1.45 m height "
                                 "checked with the sampler's own tests (flat patch, 0.35 m front clearance, traversable collision-free arc bases at 0.6-0.95 m, "
                                 "right-arm pushing reach of the aligned target's lower corners; reference target painting/dacihj 0.50 x 0.42 m), merged into "
                                 "regions; then one full sampler instance per region (init settled + BEHAVIOR-validated, :init true, goal false, >= 5 objects "
                                 "visible, hinge drift, goal feasible). Rows below: one per validated region (selection viable, model = wall object, pair = "
                                 f"scene: room; full tables and maps in {RUN}/wall_candidates.md, spec-ready blocks in {RUN}/wall_candidates.json). "
                                 + " | ".join(per_scene))})
            out.append(hdr)
            out.extend(subs)
            i = j
            n_done += 1
            continue
        if task == "wa1" and role == "floor" and walls is not None:
            r = dict(r)
            base = re.sub(r";? ?other scenes:.*$", "", r["note"])
            r["note"] = base + "; other scenes: floor of the room of the validated wall region (see the wall row)"
        if task == "wa0" and role == "floor":
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
    with open(os.path.join(SEL, "legend_extra.json"), "w") as f:
        json.dump(LEGEND_EXTRA, f, indent=1)
    print(f"updated {n_done} blocks; {len(out)} rows -> {args.csv}")
    if not args.no_render:
        load_renderer().render(out)


if __name__ == "__main__":
    main()
