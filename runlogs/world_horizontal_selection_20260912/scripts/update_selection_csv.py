#!/usr/bin/env python3
"""Write the world-horizontal selection into ``runlogs/grasp_library_20260905/select/task_object_selection.csv`` (and
regenerate the .xlsx / .html twins with the shared renderer of runlogs/table_selection_20260911).

Adds / rewrites three blocks (idempotent: an existing ``world_alignment-2 (world_horizontal)`` block is replaced, a
missing one is inserted right after the ``world_alignment-1 (world_vertical)`` blocks):

* ``world_alignment-2 (world_horizontal)`` / ``target`` - rounds 1-3 categories and every valid model (``selection`` =
  pick 1 / pick 2 / alternate / candidate, ``pair`` = canonical frame ``front .., level .., up .., thin ..`` plus
  ``symmetric 180`` for models that look the same upside down, ``note`` = round, score (V visual, L level readability,
  H hangability; score = 2V + 2L + H), reviewer note; the first row of a category carries the category verdict and
  the ready / reviewed / valid counts);
* ``world_alignment-2 (world_horizontal)`` / ``wall`` - the 78 validated wall regions of the world-vertical survey
  (``runlogs/world_alignment_selection_20260911/wall_candidates.json``; ``model`` = wall object name, ``pair`` =
  ``<scene>: <room>``) annotated with the horizontal picks whose swept extent fits the region;
* ``world_alignment-2 (world_horizontal)`` / ``floor``.

Inputs: ``selection.json`` (select_wh_objects.py), ``duplicates.json`` (wa_duplicates.py of the world-alignment run),
``renders/render_meta.json`` (bboxes), the wall survey, the task spec (pinned models).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
TABLE_RENDERER = os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts", "update_selection_csv.py")
WA_RUN = "runlogs/world_alignment_selection_20260911"
WALLS = os.path.join(REPO, WA_RUN, "wall_candidates.json")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
RUN = "runlogs/world_horizontal_selection_20260912"
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]

LABEL = "world_alignment-2 (world_horizontal)"
PREV_LABEL = "world_alignment-1 (world_vertical)"
PROBLEM = "world_alignment-2"
MAX_DEV = math.radians(45.0)
PATCH_MARGIN = 0.10

LEGEND_EXTRA = [
    ["world-horizontal target rows", f"Rounds 1-3 of {RUN} (no grasp test: the object is pushed). Every valid model is listed: valid = reviewer "
                                     "scores V >= 3 (visual), L >= 2 (level readability: one dominant axis / edge that must be horizontal), "
                                     "H >= 1 (hangability); score = 2V + 2L + H, max 18; picks = the two best valid models per category, "
                                     "alternate = the next one. pair = canonical frame read from the renders: 'front <axis out of the wall>, "
                                     "level <axis along the wall when aligned>, up <axis pointing up when hung>, thin <axis>' plus "
                                     "'symmetric 180' when the model looks the same upside down (the sampler then accepts the half turn about "
                                     "the front axis as an orientation equivalence of the level goal). Category blocks ordered by round, then "
                                     "category score. Renders: renders/grids."],
    ["world-horizontal wall rows", "world_alignment-2 wall rows are the 78 validated regions of the world-vertical survey (same wall, same "
                                   "instance evidence); the note ends with the horizontal pick-1 models whose swept extent (footprints "
                                   "between the largest init tilt of 45 deg and level, + 0.10 m margin) fits the region's flat / clear patch."],
]


def load_renderer():
    spec = importlib.util.spec_from_file_location("table_update_selection_csv", TABLE_RENDERER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt_frame(m):
    s = f"front {m.get('front_axis', '?')}, level {m.get('level_axis', '?')}, up {m.get('up_axis', '?')}, thin {m.get('thin_axis', '?')}"
    if m.get("symmetric_180"):
        s += ", symmetric 180"
    return s


def swept_extent(p, q, d):
    if math.atan2(q, p) <= d:
        return math.hypot(p, q)
    return p * math.cos(d) + q * math.sin(d)


def swept_patch(bbox, m):
    """Side of the square wall patch (m) the model needs: max swept width / height between a 45 deg tilt and level + margin."""
    try:
        lv = "xyz".index(str(m["level_axis"]).lstrip("+-"))
        up = "xyz".index(str(m["up_axis"]).lstrip("+-"))
    except (KeyError, ValueError):
        return None
    w, h = float(bbox[lv]), float(bbox[up])
    return max(swept_extent(w, h, MAX_DEV), swept_extent(h, w, MAX_DEV)) + PATCH_MARGIN


def target_rows(sel, dups, pinned, meta):
    rows, listed, picks1 = [], [], []
    for rnd in ("1", "2", "3"):
        for rec in sel["rounds"][rnd]:
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
                parts = [f"round {rnd}", f"score {m['score']} (V{m['V']} L{m['L']} H{m['H']})"]
                if model in pinned.get(cat, set()):
                    parts.append("PINNED in task_custom_lists.json (role target)")
                if first:
                    parts.append(f"category {rec['verdict']}: {rec['category_note'].strip()} [{rec['n_ready']} ready models, {rec['n_reviewed']} reviewed, "
                                 f"{rec['n_valid']} valid" + (f", {rec['n_unreviewed']} ready models not reviewed" if rec["n_unreviewed"] else "") + "]")
                    u = sorted({f"{x['task']} r{x['round']}" for x in rec["used_by_other_tasks"]})
                    if u:
                        parts.append("category also used by " + ", ".join(u))
                    first = False
                bb = (meta.get(f"{cat}__{model}") or {}).get("bbox")
                need = swept_patch(bb, m) if bb else None
                if need is not None:
                    parts.append(f"bbox {bb[0]:.2f} x {bb[1]:.2f} x {bb[2]:.2f} m, needs a {need:.2f} m flat wall patch (swept extent at 45 deg + 0.10 m)")
                if m.get("note"):
                    parts.append("reviewer: " + m["note"].strip())
                d = dups.get(f"{cat}/{model}", {})
                dtxt = []
                if d.get("exact"):
                    dtxt.append("exact duplicate of " + ", ".join(d["exact"]))
                if d.get("same_geometry"):
                    dtxt.append("same geometry as " + ", ".join(d["same_geometry"]) + " (texture/colour variant; geometry check only)")
                rows.append({"task": "", "role": "", "category": cat, "model": model, "selection": selname, "pair": fmt_frame(m),
                             "duplicates": "; ".join(dtxt), "note": "; ".join(parts)})
                listed.append((cat, model))
                if selname == "pick 1":
                    picks1.append((cat, model, need))

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
    return rows, n_cats, f"{len(geoms)} ({len(unique)})", dsum, dropped, picks1


def wall_rows(walls, picks1):
    rows, per_scene, n_objects = [], [], set()
    for scene in SCENES:
        sc = walls["scenes"].get(scene)
        if not sc:
            per_scene.append(f"{scene}: not surveyed")
            continue
        ok_rooms = []
        for room, rr in sorted(sc["rooms"].items()):
            val = [r for r in (rr.get("regions") or []) if r.get("validated")]
            if val:
                ok_rooms.append(f"{room} ({len(val)} region{'s' if len(val) > 1 else ''}, {sum(r['length_m'] for r in val):.1f} m)")
            for r in val:
                inst = next((i for i in r.get("instances", []) if i.get("n_written")), None)
                ii = (inst or {}).get("instance") or {}
                patch = float(r["max_patch_m"])
                fit = [f"{c}/{m}" for c, m, need in picks1 if need is not None and need <= patch + 1e-6]
                nofit = [f"{c} ({need:.2f} m)" for c, m, need in picks1 if need is not None and need > patch + 1e-6]
                parts = [f"region {r['id']}: ({r['p0'][0]:.2f}, {r['p0'][1]:.2f}) -> ({r['p1'][0]:.2f}, {r['p1'][1]:.2f}) at z {r['hang_z']:.2f} m, "
                         f"normal ({r['normal'][0]:.2f}, {r['normal'][1]:.2f}), length {r['length_m']:.2f} m",
                         f"flat + unobstructed patch {patch:.1f} m (objects up to {max(0.0, patch - 0.10):.1f} m fit when level)",
                         f"{r['n_spots']} spots, {r['n_bases']} collision-free arc bases, {r['n_reach']} with pushing reach (painting reference target)"]
                if ii:
                    bp = ii.get("base_pose2d") or [0, 0, 0]
                    parts.append(f"world-vertical instance: robot base ({bp[0]:.2f}, {bp[1]:.2f}, yaw {bp[2]:.2f}), arc radius {(ii.get('arc') or {}).get('radius', 0):.2f} m, "
                                 f"deviation {ii.get('deviation_deg', 0):.0f} deg, {ii.get('n_visible_objects', '?')} objects visible")
                parts.append(f"evidence {WA_RUN}/results/{scene}__{room}.json")
                parts.append(f"horizontal pick-1 models that fit (swept extent at 45 deg + 0.10 m <= patch): {len(fit)}/{len(picks1)}"
                             + (": " + ", ".join(fit) if fit else "") + ("; too large: " + ", ".join(nofit) if nofit else ""))
                rows.append({"task": "", "role": "", "category": "walls", "model": r["wall_name"], "selection": "viable",
                             "pair": f"{scene}: {room}", "duplicates": "", "note": "; ".join(parts)})
                n_objects.add((scene, r["wall_name"]))
        per_scene.append(f"{scene}: {', '.join(ok_rooms) if ok_rooms else 'NONE'}")
    return rows, len(n_objects), per_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--selection", default=os.path.join(ROOT, "selection.json"))
    ap.add_argument("--duplicates", default=os.path.join(ROOT, "duplicates.json"))
    ap.add_argument("--walls", default=WALLS)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    with open(args.selection) as f:
        sel = json.load(f)
    dups = {}
    if os.path.exists(args.duplicates):
        with open(args.duplicates) as f:
            dups = json.load(f).get("flags", {})
    with open(args.walls) as f:
        walls = json.load(f)
    with open(os.path.join(ROOT, "renders", "render_meta.json")) as f:
        meta = json.load(f)
    with open(SPEC) as f:
        spec = json.load(f)
    pinned = defaultdict(set)
    for cat, ms in (spec.get(PROBLEM, {}).get("roles", {}).get("target", {}).get("models") or {}).items():
        pinned[cat] |= set(ms)
    pin_txt = ", ".join(f"{c}/{m}" for c in sorted(pinned) for m in sorted(pinned[c])) or "none yet"
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    subs, n_cats, mn, dsum, dropped, picks1 = target_rows(sel, dups, pinned, meta)
    wsubs, n_wall_obj, per_scene = wall_rows(walls, picks1)
    target_hdr = {"task": LABEL, "role": "target", "category": str(n_cats), "model": mn, "selection": "", "pair": "", "duplicates": dsum,
                  "note": ("BDDL objects: 1 target hinged flat on a wall through its CoM (TRO revolute constraint, axis = wall normal; goal = LEVEL "
                           "orientation, released), wall, floor, agent. Sibling of world_alignment-1 with the same hinge, 1.45 m hanging height, "
                           "15-45 deg init tilt in the wall plane and pushing from an arc 0.6-0.95 m in front; the object must therefore have ONE "
                           "readable horizontal direction (a dominant long axis or reference edge: bars, rails, shelves, wide boards / signs, rods on "
                           "display hooks), be flat against the wall and 0.25-0.9 m along that axis. A 180-deg ambiguity is fine (a plain bar is level "
                           "either way; such models carry 'symmetric 180' in pair and the sampler adds the half turn about the front axis to the goal's "
                           f"orientation equivalences), a 90-deg one (square) is not. Test spec pins {pin_txt}. Frames per model in pair "
                           "(sampler dict form ready-made in spec_fragments.json: generation.front_axis / up_axis / level_axis / symmetric_180). "
                           f"Selection = {RUN} (no grasp test). 135 ready models of 40 elongated wall / rod categories (candidate_categories.json: "
                           "geometry scan of every BEHAVIOR category + wall / rod vocabulary) rendered in their canonical frame (renders/grids) and "
                           "scored by 8 reviewers with review_contract.md: V visual 0-5, L level readability 0-3, H hangability 0-2; score = 2V + 2L + H "
                           "(max 18); valid = V >= 3, L >= 2, H >= 1. Round 1 = up to 10 categories used by no other task (>= 2 valid models), round 2 = "
                           "up to 5 more (may overlap other tasks' round-1 categories, incl. world_alignment-1's; >= 2 valid), round 3 = up to 5 more "
                           "(any overlap, 1 valid model enough). EVERY valid model is listed (pick 1 / pick 2 / alternate / candidate, sorted by score); "
                           "invalid models are not listed. Each row notes the flat wall patch the model needs (its extents swept between a 45 deg tilt "
                           "and level + 0.10 m); compare with the wall rows' patch sizes. Duplicate check: geometry signatures (duplicates.json)."
                           + (" Reviewed but not selected: " + "; ".join(dropped) + "." if dropped else ""))}
    wall_hdr = {"task": LABEL, "role": "wall", "category": "1" if wsubs else "0", "model": str(n_wall_obj), "selection": "", "pair": "", "duplicates": "",
                "note": ("fixed scene wall (wall.n.01_1, constraint parent). Same walls as world_alignment-1: the 78 regions validated by the wall survey "
                         f"({WA_RUN}/scripts/survey_walls.py, sampler tests at 1.45 m with the painting/dacihj reference target 0.50 x 0.42 m, one full "
                         "instance per region) are reused; the horizontal sampler seeds its spots along the same fixed.wall_regions and re-checks flat patch + "
                         "0.35 m front clearance for the target's SWEPT extents (footprints between the largest init tilt and level, spec section 5.9), so a "
                         "region admits a horizontal target only when its patch covers that swept extent (each row lists the pick-1 models that fit). "
                         "Test spec: house_single_floor living_room_0, wall found by the sampler. Rows below: one per validated region (selection viable, "
                         f"model = wall object, pair = scene: room; tables and maps in {WA_RUN}/wall_candidates.md, spec-ready blocks in "
                         f"{WA_RUN}/wall_candidates.json). Validated rooms: " + " | ".join(per_scene))}
    floor_row = {"task": LABEL, "role": "floor", "category": "0", "model": "0", "selection": "", "pair": "", "duplicates": "",
                 "note": "fixed scene object: house_single_floor: floor of living_room_0; other scenes: floor of the room of the validated wall region (see the wall row)"}
    block = [target_hdr] + subs + [wall_hdr] + wsubs + [floor_row]

    # remove an existing block, remember where the previous problem ends
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
        last_prev = max((k for k, r in enumerate(out) if r["task"] == PREV_LABEL), default=None)
        if last_prev is None:
            raise SystemExit(f"{PREV_LABEL} block not found in {args.csv}")
        insert_at = last_prev + 1
        while insert_at < len(out) and not out[insert_at]["task"]:
            insert_at += 1
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
    print(f"{LABEL}: {len(subs)} target rows ({n_cats} categories, {mn} models), {len(wsubs)} wall rows; {len(out)} rows -> {args.csv}")
    if not args.no_render:
        load_renderer().render(out)


if __name__ == "__main__":
    main()
