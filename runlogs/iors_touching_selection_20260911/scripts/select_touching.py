#!/usr/bin/env python3
"""IORS-4 (touching) target / partner category and model selection -> ``select/task_object_selection.csv`` (+ xlsx / html).

Rules (same as IORS-3 nextto, see the header notes written into the sheet):

* target = ready tabletop object that is pinchable in its rest pose (smallest horizontal extent <= 8.9 cm, the antipodal
  pipeline's width limit) and stays upright when pushed at its near face (flat base, height <= 3 x smallest horizontal
  extent, no rolling body); 2 picks per category. Categories that already have a tabletop grasp test reuse that test
  (result key, AG-valid / clean counts, selection score and reviewer note); fresh categories have no grasp test (the
  touching oracle pushes) and were checked from textured renders (renders/grids/*.png).
* partner = ready model that fits every scene support validated for IORS by the table-selection run plus the test table
  (short side <= min depth - 6 cm, long side <= min length / 2), height <= 35 cm and <= 2.5 x its smallest horizontal
  extent, flat base (>= 25 % of the footprint in contact), does not roll. Not grasped.
* round 1 = 4 pairs of categories unused by any other task, >= 2 valid models each; round 2 = 2 pairs whose categories
  are round-1 categories of another task (not reused yet), >= 2 valid models; round 3 = 2 pairs with any overlap,
  single-model categories allowed.

Inputs: candidate_pool.json, model_geometry.json, used_categories.json (this run), the grasp-library selection
(select/out/*.json, selection_manifest.csv), duplicates/verdicts.json, task_custom_lists.json, the current sheet.
Outputs: touching_selection.json (evidence) and the rewritten IORS-4 target / partner blocks of the sheet. Idempotent.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
LIB = os.path.join(REPO, "runlogs", "grasp_library_20260905")
SEL = os.path.join(LIB, "select")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
TABLE_SCRIPTS = os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts")
RUN_REL = "runlogs/iors_touching_selection_20260911"
OUT_JSON = os.path.join(ROOT, "touching_selection.json")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]
TASK_LABEL = "inter_object_relation_satisfaction-4 (touching)"
PROB = "inter_object_relation_satisfaction-4"

PINCH_MAX, T_LONG_MAX, T_H_MAX, T_ASPECT, T_BASE = 0.089, 0.30, 0.30, 3.0, 0.25
P_H_MAX, P_ASPECT, P_BASE = 0.35, 2.5, 0.25

# ---------------------------------------------------------------------------------------------------- curated pairs
PAIRS = [
    {"round": 1, "target": "can_of_baking_mix", "partner": "board_game",
     "theme": "test-round spec pair: baking-mix can pushed against a flat board-game box",
     "why": "pinned in task_custom_lists.json and validated by the sampler on every IORS-4 support of the table-selection run; both categories unused elsewhere"},
    {"round": 1, "target": "box_of_baking_powder", "partner": "kitchen_analog_scale",
     "theme": "baking: baking-powder tin pushed against the kitchen scale",
     "why": "fresh categories; 8.4 cm tins, heavy stable scales"},
    {"round": 1, "target": "box_of_coffee", "partner": "jar_of_grains",
     "theme": "pantry: coffee tin / box pushed against the grain jar",
     "why": "fresh categories; two different coffee containers, 12 texture variants of one heavy 22 cm jar"},
    {"round": 1, "target": "box_of_vegetable_juice", "partner": "pop_case",
     "theme": "beverages: juice carton pushed against the soda case",
     "why": "fresh categories; two small cartons, heavy flat soda cases"},
    {"round": 2, "target": "box_of_sanitary_napkins", "partner": "box_of_tissues",
     "theme": "bathroom paper goods: napkin pack pushed against the tissue pack",
     "why": "both are disembedding round-1 targets not reused by another task; tabletop grasp tests exist"},
    {"round": 2, "target": "carton_of_milk", "partner": "box_of_cookies",
     "theme": "milk and cookies: milk carton pushed against the cookie box",
     "why": "both are disembedding round-1 targets not reused by another task; tabletop grasp tests exist"},
    {"round": 3, "target": "teacup", "partner": "teapot",
     "theme": "tea set: cup pushed against the teapot",
     "why": "teacup is mental_rotation r1 / IORS-1 r2, teapot a configuration_matching-3 female partner; 3 plain grey teapots"},
    {"round": 3, "target": "bottle_of_medicine", "partner": "first_aid_kit",
     "theme": "medicine: pill bottle pushed against the first-aid case",
     "why": "bottle_of_medicine is object_scaling r1 / disembedding r2; first_aid_kit has a single ready model (round 3 allows it)"},
]

#: grasp-library tabletop tests reused for the targets of rounds 2-3: category -> (select/out group, key suffix)
GRASP_REUSE = {
    "box_of_sanitary_napkins": ("disembedding__box_of_sanitary_napkins__r1", ""),
    "carton_of_milk": ("disembedding__carton_of_milk__r1", ""),
    "teacup": ("IORS__teacup__r2", "__t"),
    "bottle_of_medicine": ("disembedding__bottle_of_medicine__r2", "__t"),
}

#: visual review of the fresh categories from renders/grids (V 0-5 as in select_contract.md; note <= 25 words)
VISUAL = {
    "can_of_baking_mix/blrqqz": (4, "green cupcake-mix can 9.4 x 11 cm, sharp label all round"),
    "can_of_baking_mix/fpbxfp": (4, "green / yellow cupcake-mix can, same can as blrqqz"),
    "can_of_baking_mix/ohtiap": (4, "blue cupcake-mix can, same can as blrqqz"),
    "can_of_baking_mix/orwvfx": (4, "cookie-mix tub 10.4 x 9.7 cm with cookie photo on the lid"),
    "can_of_baking_mix/rxorlp": (4, "dark cookie-mix tub, same tub as orwvfx"),
    "can_of_baking_mix/xefopo": (4, "light-blue cookie-mix tub, same tub as orwvfx"),
    "board_game/eoywpy": (4, "purple 13 x 22 x 3 cm printed game box"),
    "board_game/frgdqh": (4, "blue Chutes-and-Ladders style box 13 x 20 x 3 cm"),
    "board_game/fzyuip": (4, "dark 13 x 16 x 4 cm box with block-game art"),
    "board_game/gcfkqa": (4, "colourful 13 x 20 x 4 cm box (Don't Wake Daddy style)"),
    "board_game/gumkqh": (4, "Clue-style dark box 13 x 20 x 3 cm"),
    "board_game/ibzhzu": (4, "teal 13 x 20 x 2.7 cm box"),
    "board_game/iqfvfl": (4, "orange 13 cm cube box (Mad Gab); the only tall box of the category"),
    "board_game/lzslbu": (4, "Game-of-Life style 13 x 20 x 3 cm box"),
    "board_game/nfewft": (4, "small 13 x 10 x 3 cm box"),
    "board_game/ovttum": (4, "Monopoly-style 13 x 20 x 2.7 cm box"),
    "board_game/phwwnx": (4, "small 13 x 7 x 7 cm box (Cootie)"),
    "board_game/qdkyzl": (4, "small 13 x 10 x 2.7 cm box"),
    "board_game/qgvsnb": (4, "long Clue box 13 x 25.5 x 2.5 cm"),
    "board_game/rgvkmy": (4, "small blue 13 x 10 x 3 cm box"),
    "board_game/twkglv": (4, "small red-striped 13 x 10 x 3 cm box"),
    "board_game/vbamua": (4, "long 29 x 14 x 3.5 cm box"),
    "box_of_baking_powder/vzgrlv": (4, "red Royal baking-powder tin, printed front and sides, plain red back"),
    "box_of_baking_powder/zevydc": (4, "cream Royal tin with vintage figure print, plain back"),
    "kitchen_analog_scale/zpmpoq": (4, "red pedestal scale with dial and top tray, 23 x 23 x 30 cm, flat 19 x 18 cm base"),
    "kitchen_analog_scale/lpjpez": (3, "brown bowl scale on a domed housing, 23 x 23 x 20 cm, 18 cm base ring, dull texture"),
    "box_of_coffee/cjtadw": (4, "red Desirable roasted-coffee tin 8.5 x 14 x 19 cm, printed front and side"),
    "box_of_coffee/mreyla": (4, "slim dark coffee box 5.5 x 21 x 13 cm, printed all round"),
    "box_of_vegetable_juice/jsnnlv": (4, "hexagonal green vegetable-juice cup 6 x 6.5 x 11 cm, tapered 4.5 cm base"),
    "box_of_vegetable_juice/qgjdbn": (4, "rectangular vegetable-juice carton 4.6 x 7.4 x 13 cm, printed all round"),
    "pop_case/gwgaqb": (4, "grey Tribal Cola 24-can case 24 x 36 x 11.5 cm"),
    "pop_case/pldtgy": (4, "red Tribal Cola 24-can case 33 x 22 x 10.5 cm"),
    "pop_case/qcfxec": (4, "green shrink-wrapped 24-can pack 22 x 32.5 x 10 cm"),
    "pop_case/rzhvvk": (4, "red shrink-wrapped 24-can pack 20 x 29.5 x 9 cm"),
    "pop_case/unmexe": (4, "black Super Cola shrink-wrapped 24-pack 18 x 27 x 8.4 cm"),
    "pop_case/zfyatg": (4, "black Tribal Cola no-sugar 24-pack 15 x 22 x 7 cm"),
    "pop_case/dufbnv": (4, "orange Tango 12-can fridge pack 13 x 39 x 13 cm"),
    "pop_case/noiyxv": (4, "green citrus fridge pack 13 x 39 x 13 cm"),
    "pop_case/oarbks": (4, "pink citrus fridge pack 13 x 39 x 13 cm"),
    "pop_case/qwtbon": (4, "dark red fridge pack 13 x 39 x 13 cm"),
    "pop_case/seiatm": (4, "grey ginger-ale fridge pack 13 x 39 x 13 cm"),
    "pop_case/sqiatv": (4, "red cherry-cola fridge pack 13 x 39 x 13 cm"),
    "pop_case/zxguap": (4, "blue iced-tea fridge pack 13 x 39 x 13 cm"),
    "teapot/foaehs": (2, "grey untextured squat teapot with lid, 26 x 18 x 17 cm"),
    "teapot/jlalfc": (2, "grey untextured round teapot, 29 x 16.5 x 12 cm"),
    "teapot/mvrhya": (2, "grey untextured teapot body without lid (the configuration_matching-3 lid partner), 23.5 x 15 x 9 cm"),
    "first_aid_kit/flctdz": (4, "white plastic first-aid case 16 x 20 x 7.4 cm, red / blue FIRST AID label"),
    "box_of_cookies/qkzrdd": (3, "hexagonal prism cookie pack 17 x 7 x 6.5 cm lying on a face (no AG-valid grasp in the library)"),
    "box_of_cookies/swilxz": (3, "octagonal ice-cream-style cookie tub 16 x 17 x 10.5 cm (not in the grasp-library run)"),
}
JAR_COLOURS = {"fyfahz": "tan grains", "irqkdw": "dark grains", "jbhdeu": "grey-tan grains", "lnxzuy": "brown grains", "piemou": "grey grains",
               "pluujx": "dark brown grains", "uydvnv": "yellow-green grains", "vuolos": "red beans", "xebmcy": "grains", "xrirkq": "grains",
               "ykrdzg": "grains", "zqoboc": "grains"}
for _m, _c in JAR_COLOURS.items():
    VISUAL[f"jar_of_grains/{_m}"] = (3, f"same 22 x 22 x 26.5 cm bulbous lidded jar filled with {_c}; 12 cm base")

#: same-geometry groups judged from the renders (texture / colour variants; the fresh categories are not in duplicates.md)
MANUAL_SAME_GEOM = {
    "can_of_baking_mix": [["blrqqz", "fpbxfp", "ohtiap"], ["orwvfx", "rxorlp", "xefopo"]],
    "board_game": [["frgdqh", "gumkqh"], ["nfewft", "twkglv"]],
    "box_of_baking_powder": [["vzgrlv", "zevydc"]],
    "jar_of_grains": [list(JAR_COLOURS)],
    "pop_case": [["dufbnv", "noiyxv", "oarbks", "qwtbon", "seiatm", "sqiatv", "zxguap"]],
}
MANUAL_EXACT = {}

#: fresh-category picks (after the geometric ranking, the tie-breaks are the visual review); None = automatic
FORCED_PICKS = {
    "can_of_baking_mix": ["blrqqz", "orwvfx"],  # pinned in the spec, one per geometry
    "board_game": ["eoywpy", "gcfkqa"],  # pinned in the spec
    "box_of_baking_powder": ["vzgrlv", "zevydc"],
    "kitchen_analog_scale": ["zpmpoq", "lpjpez"],
    "box_of_coffee": ["cjtadw", "mreyla"],
    "jar_of_grains": ["vuolos", "uydvnv"],  # the two most distinct grain colours
    "box_of_vegetable_juice": ["jsnnlv", "qgjdbn"],
    "pop_case": ["rzhvvk", "gwgaqb"],  # mid-size red shrink pack, then a different (boxed grey) case
    "box_of_tissues": ["uglbjc", "zutrxn"],  # the disembedding picks (same tabletop pipeline)
    "box_of_cookies": ["dlhqft", "apjokz"],  # flat-lying disembedding picks, dlhqft lies flat in the canonical pose
    "teapot": ["foaehs", "jlalfc"],
    "first_aid_kit": ["flctdz"],
}

#: categories looked at and not taken (named in the header note)
REJECTED_CATEGORIES = [
    "paintbrush -> bucket_of_paint (brush 0.9-1.3 cm thick: not pushable / not AG-graspable)",
    "toilet_paper -> tissue_dispenser (2 of 4 rolls untextured grey, dispensers plain dark boxes)",
    "can_of_beans (its two models are the same asset)",
    "bottle_of_whiskey, bottle_of_soup, round_bottom_flask, erlenmeyer_flask (glass / breakable or round base / > 2.8 x aspect)",
    "eggplant, zucchini, gourd, sweet_corn, firewood, pencil_case, screwdriver (roll when pushed)",
    "carving_knife (blade a few mm thick: no push face), post_it, tag, table_knife, hard_drive (< 2.5 cm thick)",
    "test_tube_rack, shoe_box, storage_box, gift_box, desk_phone, loudspeaker, plant_pot, bucket_of_paint, hiking_boot (usable partners but no fresh pinchable target category pairs with them; kept as alternatives)",
    "kettle (alternative tea partner: 2 ready dark stovetop kettles, 18 x 16 x 17 cm)",
]


# ---------------------------------------------------------------------------------------------------- helpers
def fmt_cm(v):
    return f"{v * 100:.1f}"


def bbox_txt(b):
    return " x ".join(fmt_cm(v) for v in b) + " cm"


def load_inputs():
    pool = json.load(open(os.path.join(ROOT, "candidate_pool.json")))
    geom = json.load(open(os.path.join(ROOT, "model_geometry.json")))
    used = json.load(open(os.path.join(ROOT, "used_categories.json")))
    ct = json.load(open(os.path.join(REPO, "runlogs", "grasp_exp_20260904", "category_table.json")))["objects"]
    spec = json.load(open(SPEC))
    verdicts = json.load(open(os.path.join(SEL, "duplicates", "verdicts.json")))
    manifest = {}
    with open(os.path.join(LIB, "selection_manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            manifest[r["result_key"]] = r
    return pool, geom, used, ct, spec, verdicts, manifest


def same_geom_groups(cat, models, geom, verdicts):
    """model -> (exact group members, same-geometry members) from the sheet's verdicts (reviewed categories) or from the
    vertex hashes + the manual render-based groups (fresh categories)."""
    exact = defaultdict(set)
    same = defaultdict(set)
    for c in verdicts:
        if c["category"] != cat:
            continue
        for g in c.get("exact_groups", []):
            for m in g:
                exact[m].update(x for x in g if x != m)
        if c["verdict"] in ("exact_same", "texture_variants"):
            for m in c["models"]:
                same[m].update(x for x in c["models"] if x != m)
    by_hash = defaultdict(list)
    for m in models:
        g = geom.get(f"{cat}/{m}")
        if g and "vertex_hash" in g:
            by_hash[g["vertex_hash"]].append(m)
    for ms in by_hash.values():
        for m in ms:
            for x in ms:
                if x == m:
                    continue
                same[m].add(x)
                if geom[f"{cat}/{m}"]["texture_md5"] == geom[f"{cat}/{x}"]["texture_md5"]:
                    exact[m].add(x)
    for grp in MANUAL_SAME_GEOM.get(cat, []):
        for m in grp:
            same[m].update(x for x in grp if x != m)
    for grp in MANUAL_EXACT.get(cat, []):
        for m in grp:
            exact[m].update(x for x in grp if x != m)
    for m in list(exact):
        same[m].update(exact[m])
    return exact, same


def count_geoms(models, exact, same):
    """(M distinct geometries, N distinct models after folding exact duplicates)."""
    def components(rel):
        seen, n = set(), 0
        for m in models:
            if m in seen:
                continue
            n += 1
            stack = [m]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(y for y in rel.get(x, ()) if y in models)
        return n
    return components(same), components(exact)


def dup_text(m, exact, same):
    if exact.get(m):
        return "exact duplicate of " + ", ".join(sorted(exact[m]))
    if same.get(m):
        return "same geometry as " + ", ".join(sorted(same[m])) + " (texture/colour variant)"
    return ""


# ---------------------------------------------------------------------------------------------------- validity
def base_ok(g, min_frac, min_rel):
    """Flat base: the convex hull of the vertices within 5 mm of the bottom covers >= min_frac of the footprint, or its
    smaller side spans >= min_rel of the object's smallest horizontal extent (teapots / jars: a foot ring under a body
    whose footprint the spout and handle inflate)."""
    x, y, z = g["bbox"]
    w = min(x, y)
    frac = g.get("base_contact_fraction") or 0.0
    bmin = min(g.get("base_hull_width") or 0.0, g.get("base_hull_depth") or 0.0)
    return frac >= min_frac or bmin >= min_rel * w, frac, bmin


def target_check(g):
    x, y, z = g["bbox"]
    w, L = sorted((x, y))
    reasons = []
    if w > PINCH_MAX:
        reasons.append(f"smallest horizontal extent {fmt_cm(w)} cm > {fmt_cm(PINCH_MAX)} cm gripper opening (push-only)")
    if L > T_LONG_MAX or z > T_H_MAX:
        reasons.append(f"too large ({bbox_txt(g['bbox'])})")
    if z / max(w, 1e-6) > T_ASPECT:
        reasons.append(f"height {fmt_cm(z)} cm > 3 x smallest extent {fmt_cm(w)} cm (tips when pushed)")
    ok, frac, bmin = base_ok(g, T_BASE, 0.6)
    if not ok:
        reasons.append(f"base {fmt_cm(bmin)} cm wide under a {fmt_cm(w)} cm body, contact {frac:.2f} of the footprint (rounded base, rocks when pushed)")
    return reasons


def partner_check(g, limits, supports):
    x, y, z = g["bbox"]
    w, L = sorted((x, y))
    reasons = []
    misses = [s["name"] for s in supports.values() if w > s["depth"] - 0.06 or L > s["length"] / 2]
    if misses:
        reasons.append("does not fit " + ", ".join(misses) + f" (footprint {fmt_cm(w)} x {fmt_cm(L)} cm)")
    if z > P_H_MAX:
        reasons.append(f"tall ({fmt_cm(z)} cm)")
    if z / max(w, 1e-6) > P_ASPECT:
        reasons.append(f"height {fmt_cm(z)} cm > 2.5 x smallest extent {fmt_cm(w)} cm (topples when bumped)")
    ok, frac, bmin = base_ok(g, P_BASE, 0.45)
    if not ok:
        reasons.append(f"base {fmt_cm(bmin)} cm wide under a {fmt_cm(w)} cm body, contact {frac:.2f} of the footprint (rounded base)")
    return reasons


# ---------------------------------------------------------------------------------------------------- per category
#: disembedding r1 reviews of partner categories that were disembedding targets (visual score reused for the ranking)
partner_review = {}
for _cat in ("box_of_tissues", "box_of_cookies"):
    _p = os.path.join(SEL, "out", f"disembedding__{_cat}__r1.json")
    if os.path.exists(_p):
        partner_review[_cat] = {c["model"]: c for c in json.load(open(_p))["candidates"]}


def build_category(cat, role, rnd, pair_cat, pool, geom, ct, spec, verdicts, manifest, used, limits, supports):
    pc = pool["categories"][cat]
    models = [m["model"] for m in pc["models"]]  # ready, single-link, on disk
    not_ready = [o["model"] for o in ct.values() if o["category"] == cat and not o["ready"]]
    exact, same = same_geom_groups(cat, models, geom, verdicts)
    sizes = [max(geom[f"{cat}/{m}"]["bbox"][:2]) for m in models]
    median = statistics.median(sizes)
    pinned = set()
    role_cfg = (spec.get(PROB, {}).get("roles") or {}).get(role, {})
    for m in (role_cfg.get("models") or {}).get(cat, []):
        pinned.add(m)
    reuse = GRASP_REUSE.get(cat) if role == "target" else None
    sel_out, sel_by_model = None, {}
    if reuse:
        sel_out = json.load(open(os.path.join(SEL, "out", reuse[0] + ".json")))
        sel_by_model = {c["model"]: c for c in sel_out["candidates"]}
        labels_lib = {s["model"]: f"pick {s['rank']}" for s in sel_out["selected"]}
        if sel_out.get("alternate"):
            labels_lib[sel_out["alternate"]["model"]] = "alternate"
    recs = []
    for m in models:
        g = geom[f"{cat}/{m}"]
        x, y, z = g["bbox"]
        w, L = sorted((x, y))
        reasons = target_check(g) if role == "target" else partner_check(g, limits, supports)
        push_only = role == "target" and any("push-only" in r for r in reasons)
        hard = [r for r in reasons if "push-only" not in r]
        rec = {"model": m, "bbox": g["bbox"], "short_side": round(w, 4), "long_side": round(L, 4), "height": round(z, 4),
               "aspect": round(z / max(w, 1e-6), 2), "base_contact_fraction": g.get("base_contact_fraction") or 0.0,
               "base_min": min(g.get("base_hull_width") or 0.0, g.get("base_hull_depth") or 0.0),
               "pinchable": w <= PINCH_MAX, "push_only": push_only, "reasons": hard, "valid": not hard or (m in pinned),
               "pinned": m in pinned, "size_dev": abs(max(x, y) - median), "meta_links": g.get("meta_link_types", []),
               "exact_dup": sorted(exact.get(m, ())), "same_geom": sorted(same.get(m, ()))}
        if m in pinned and hard:
            rec["reasons_pinned"] = hard
        if reuse:
            key = f"{cat}__{m}{reuse[1]}"
            mr = manifest.get(key)
            c = sel_by_model.get(m)
            rec["grasp_key"] = key
            rec["ag_valid"] = int(mr["n_valid"]) if mr and mr["n_valid"] else 0
            rec["ag_tested"] = int(mr["tested"]) if mr and mr["tested"] else 0
            rec["clean"] = int(mr["n_clean"]) if mr and mr["n_clean"] else 0
            rec["flags"] = [f for f in (mr["flags"].split(";") if mr and mr["flags"] else []) if f and f != "one_finger"]
            if c:
                rec["score"] = c["score"]
                rec["scores"] = f"G{c['G']} T{c['T']} V{c['V']} R{c['R']}"
                rec["V"] = c["V"]
                rec["reviewer"] = c["note"]
                rec["lib_label"] = labels_lib.get(m, "candidate")
            else:
                rec["score"] = None
                rec["lib_label"] = None
            if rec["ag_valid"] == 0 and m not in pinned:
                rec["valid"] = False
                rec["reasons"].append("no AG-valid grasp in the grasp-library run" if mr else "not in the grasp-library run")
        vis = VISUAL.get(f"{cat}/{m}")
        if vis:
            rec["V"] = vis[0]
            rec["visual"] = vis[1]
        elif role == "partner" and m in partner_review.get(cat, {}):
            rec["V"] = partner_review[cat][m]["V"]
        recs.append(rec)
    # ranking
    def rank_key(r):
        lib = r.get("score")
        return (0 if r["valid"] else 1,
                0 if r["pinned"] else 1,
                -(lib if lib is not None else 0),
                0 if r["pinchable"] else 1,
                r["aspect"] if role == "target" else r["aspect"],
                -(r.get("V") or 0),
                -(r.get("base_contact_fraction") or 0),
                r["size_dev"], r["model"])
    recs.sort(key=rank_key)
    listed = [r for r in recs if r["valid"]]
    rejected = [r for r in recs if not r["valid"]]
    # picks: pinned first, then forced, then the library labels, then the ranking (pick 2 prefers another geometry)
    labels = {}
    forced = FORCED_PICKS.get(cat)
    order = []
    if forced:
        order = [m for m in forced if m in {r["model"] for r in listed}]
    elif reuse:
        order = [r["model"] for r in listed if r.get("lib_label") == "pick 1"] + [r["model"] for r in listed if r.get("lib_label") == "pick 2"]
    if len(order) < 2:
        seen_exact = set()
        for r in listed:
            if r["model"] in order:
                continue
            sig = frozenset([r["model"], *r["exact_dup"]])
            if any(sig & s for s in seen_exact):
                continue
            if len(order) == 1:
                p1 = next(x for x in listed if x["model"] == order[0])
                if r["model"] in p1["same_geom"] and any(x["model"] not in p1["same_geom"] and x["model"] not in order and x["valid"] for x in listed):
                    continue
            order.append(r["model"])
            seen_exact.add(sig)
            if len(order) == 2:
                break
    for i, m in enumerate(order[:2]):
        labels[m] = f"pick {i + 1}"
    alt = next((r["model"] for r in listed if r["model"] not in labels and not (set(r["exact_dup"]) & set(labels))), None)
    if alt:
        labels[alt] = "alternate"
    for r in recs:
        r["selection"] = labels.get(r["model"], "candidate" if r["valid"] else "rejected")
    M, N = count_geoms([r["model"] for r in listed], exact, same)
    return {"category": cat, "role": role, "round": rnd, "pair": pair_cat, "n_ready": len(models), "not_ready": not_ready,
            "n_valid": len(listed), "n_geometries": M, "n_models_folded": N, "median_long_side": round(median, 4),
            "used_by": [f"{u['task']} ({u['role']}, r{u['round']})" for u in used.get(cat, [])],
            "grasp_reuse": reuse[0] if reuse else None, "models": recs}


# ---------------------------------------------------------------------------------------------------- sheet rows
def target_note(block, r, pair, first, theme, why):
    parts = [f"round {block['round']}"]
    if r["pinned"]:
        parts.append("PINNED in task_custom_lists.json (role target)")
    if r.get("grasp_key"):
        if r.get("score") is not None:
            parts.append(f"score {r['score']} ({r['scores']})")
        parts.append(f"grasp key {r['grasp_key']}")
        parts.append(f"AG-valid {r['ag_valid']}/{r['ag_tested']}, reviewer-clean {r['clean']}")
        if r["flags"]:
            parts.append("flags: " + ", ".join(r["flags"]))
        parts.append(f"tabletop grasp test reused from {block['grasp_reuse'].split('__')[0]} (selection {block['grasp_reuse']})")
    else:
        parts.append("no grasp-library test (fresh category; the touching target is pushed, a pick-and-place is optional)")
    geo = (f"touching check: bbox {bbox_txt(r['bbox'])}; smallest horizontal extent {fmt_cm(r['short_side'])} cm "
           f"({'pinchable' if r['pinchable'] else 'NOT pinchable, push-only'}); height / smallest extent {r['aspect']}; "
           f"flat base {fmt_cm(r['base_min'])} cm wide ({r['base_contact_fraction']:.2f} of the footprint)")
    parts.append(geo)
    if r.get("reasons_pinned"):
        parts.append("kept because pinned although: " + "; ".join(r["reasons_pinned"]))
    if r.get("visual"):
        parts.append(f"render check (V{r['V']}): {r['visual']}")
    if r.get("reviewer"):
        parts.append("reviewer: " + r["reviewer"])
    if first:
        parts.append(f"pair: {theme} ({why})")
        rej = [x for x in block["models"] if not x["valid"]]
        if rej:
            parts.append(f"not listed: {len(rej)} ready model(s) rejected: " + "; ".join(f"{x['model']} ({'; '.join(x['reasons'])})" for x in rej))
        if block["not_ready"]:
            parts.append("not ready (unprocessed complaint): " + ", ".join(block["not_ready"]))
        if block["used_by"]:
            parts.append("category also used by: " + "; ".join(block["used_by"]))
    return "; ".join(parts)


def partner_note(block, r, first, theme, n_sup):
    parts = [f"round {block['round']}"]
    if r["pinned"]:
        parts.append("PINNED in task_custom_lists.json (role partner)")
    parts.append("ready")
    parts.append(f"bbox {bbox_txt(r['bbox'])}")
    parts.append(f"table fit: footprint {fmt_cm(r['short_side'])} x {fmt_cm(r['long_side'])} cm fits all {n_sup} validated IORS supports")
    parts.append(f"stability: height / smallest extent {r['aspect']}, flat base {fmt_cm(r['base_min'])} cm wide ({r['base_contact_fraction']:.2f} of the footprint)")
    if r.get("visual"):
        parts.append(f"render check (V{r['V']}): {r['visual']}")
    if r.get("grasp_note"):
        parts.append(r["grasp_note"])
    if first:
        parts.append(f"pair: {theme}")
        rej = [x for x in block["models"] if not x["valid"]]
        if rej:
            parts.append(f"not listed: {len(rej)} ready model(s) rejected: " + "; ".join(f"{x['model']} ({'; '.join(x['reasons'])})" for x in rej))
        if block["not_ready"]:
            parts.append("not ready (unprocessed complaint): " + ", ".join(block["not_ready"]))
        if block["used_by"]:
            parts.append("category also used by: " + "; ".join(block["used_by"]))
    return "; ".join(parts)


SEL_ORDER = {"pick 1": 0, "pick 2": 1, "alternate": 2, "candidate": 3}


def block_rows(block, role, pair_prefix, theme, why, n_sup, partner_lib_notes):
    rows = []
    listed = [r for r in block["models"] if r["valid"]]
    listed.sort(key=lambda r: SEL_ORDER.get(r["selection"], 9))  # stable: keeps the ranking order inside each label
    exact = {r["model"]: set(r["exact_dup"]) for r in listed}
    same = {r["model"]: set(r["same_geom"]) for r in listed}
    for i, r in enumerate(listed):
        cat_cell = block["category"] if i == 0 else ""
        if i == len(listed) - 1 and len(listed) > 1:
            cat_cell = f"{block['n_geometries']} ({block['n_models_folded']})"
        if role == "partner" and r["model"] in partner_lib_notes:
            r["grasp_note"] = partner_lib_notes[r["model"]]
        note = target_note(block, r, block["pair"], i == 0, theme, why) if role == "target" else partner_note(block, r, i == 0, theme, n_sup)
        rows.append({"task": "", "role": "", "category": cat_cell, "model": r["model"], "selection": r["selection"],
                     "pair": f"{pair_prefix} {block['pair']}", "duplicates": dup_text(r["model"], exact, same), "note": note})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--dry-run", action="store_true", help="write touching_selection.json only")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    pool, geom, used, ct, spec, verdicts, manifest = load_inputs()
    supports, limits = pool["supports"], pool["limits"]
    n_sup = len(supports)

    # grasp-library notes for partner models that were disembedding targets (informative only: partners are not grasped)
    partner_lib_notes = {}
    for cat in ("box_of_tissues", "box_of_cookies"):
        p = os.path.join(SEL, "out", f"disembedding__{cat}__r1.json")
        if os.path.exists(p):
            d = json.load(open(p))
            lab = {s["model"]: f"pick {s['rank']}" for s in d["selected"]}
            for c in d["candidates"]:
                mr = manifest.get(f"{cat}__{c['model']}")
                partner_lib_notes[c["model"]] = (f"disembedding r1 tabletop grasp test (informative, the partner is not grasped): "
                                                 f"score {c['score']} (G{c['G']} T{c['T']} V{c['V']} R{c['R']}), {lab.get(c['model'], 'candidate')}, "
                                                 f"AG-valid {mr['n_valid'] if mr else '?'}/5; reviewer: {c['note']}")

    blocks = {"target": [], "partner": []}
    for p in PAIRS:
        blocks["target"].append(build_category(p["target"], "target", p["round"], p["partner"], pool, geom, ct, spec, verdicts, manifest, used, limits, supports))
        blocks["partner"].append(build_category(p["partner"], "partner", p["round"], p["target"], pool, geom, ct, spec, verdicts, manifest, used, limits, supports))
    # checks against the round rules
    problems = []
    for role in ("target", "partner"):
        for b in blocks[role]:
            other_r1 = {u["round"] for u in used.get(b["category"], [])}
            if b["round"] == 1 and 1 in other_r1 and not (b["category"] == "board_game"):
                problems.append(f"{b['category']} chosen for round 1 but used in round 1 by: {b['used_by']}")
            if b["round"] in (1, 2) and b["n_models_folded"] < 2:
                problems.append(f"{b['category']} round {b['round']} has only {b['n_models_folded']} distinct valid model(s)")
            if b["n_valid"] < 1:
                problems.append(f"{b['category']} has no valid model")
    for pr in problems:
        print("WARNING:", pr)
    evidence = {"date": datetime.date.today().isoformat(), "rules": {"target": {"pinch_max": PINCH_MAX, "long_max": T_LONG_MAX, "height_max": T_H_MAX,
                                                                                  "aspect_max": T_ASPECT, "base_contact_min": T_BASE},
                                                                       "partner": {"short_max": limits["partner"]["short_max"], "long_max": limits["partner"]["long_max"],
                                                                                   "height_max": P_H_MAX, "aspect_max": P_ASPECT, "base_contact_min": P_BASE}},
                "supports": supports, "pairs": PAIRS, "rejected_categories": REJECTED_CATEGORIES, "warnings": problems, "blocks": blocks}
    json.dump(evidence, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")
    for role in ("target", "partner"):
        for b in blocks[role]:
            picks = [f"{r['model']} ({r['selection']})" for r in b["models"] if r["selection"].startswith("pick")]
            print(f"  r{b['round']} {role:7s} {b['category']:26s} valid {b['n_valid']:2d}/{b['n_ready']:2d} ready, {b['n_geometries']} ({b['n_models_folded']}): {', '.join(picks)}")
    if args.dry_run:
        return

    # ---------------------------------------------------------------------------------------------- CSV blocks
    def ordered(role):
        bl = blocks[role]
        return sorted(bl, key=lambda b: (b["round"], -b["n_geometries"], -b["n_models_folded"], b["category"]))

    pinned_t = ", ".join(f"{c}/{{{', '.join(ms)}}}" for c, ms in (spec[PROB]["roles"]["target"]["models"]).items())
    pinned_p = ", ".join(f"{c}/{{{', '.join(ms)}}}" for c, ms in (spec[PROB]["roles"]["partner"]["models"]).items())
    pairs_txt = "; ".join(f"r{p['round']}: {p['target']} -> {p['partner']}" for p in PAIRS)
    sup_txt = ", ".join(f"{s['scene']}:{s['name']} ({s['depth']:.2f} x {s['length']:.2f} m)" for s in supports.values())
    target_hdr_note = (
        "BDDL objects: target (pushed - or grasped and set down - until rigid contact (Touching) with the partner, released), partner "
        "(must not move > 5 mm at the goal settle), table, floor, agent; both poses measured only. The sampler puts the partner on the "
        "robot's left with a 10-25 cm gap and the goal translates the target sideways until the hulls touch (ispatialgym/sampling/tasks/iors.py). "
        f"Test spec pins {pinned_t} -> {pinned_p}. "
        f"Selection (2026-09-11, {RUN_REL}): same rules as nextto - target = ready tabletop object that is pinchable in its rest pose "
        f"(smallest horizontal extent <= {fmt_cm(PINCH_MAX)} cm, the antipodal pipeline's width limit) and stays upright when pushed at its near face "
        f"(flat base >= {T_BASE:.0%} of the footprint, height <= {T_ASPECT:.0f} x its smallest horizontal extent, no rolling body), long side <= {fmt_cm(T_LONG_MAX)} cm, "
        "semantically paired with a partner category; 2 picks per category. Round 1 = 4 categories unused by any other task (incl. the parallel "
        "world-alignment candidate list), >= 2 distinct valid models; round 2 = 2 categories that another task uses in its round 1 (disembedding "
        "r1 targets not reused elsewhere), >= 2 valid models; round 3 = 2 categories with any overlap, single-model categories allowed. "
        f"Pairs: {pairs_txt}. Rounds 2-3 reuse the tabletop grasp test of that category (result key, AG-valid / clean, selection score, reviewer note); "
        "round-1 categories have no grasp test (the touching oracle pushes) and were checked on textured renders "
        f"({RUN_REL}/renders/grids). EVERY ready single-link model of a chosen category is listed (selection = pick 1 / pick 2 / alternate / candidate); "
        "models failing the touching check are named with the reason in the first row of the category. Categories are ordered by round, then by "
        "their number of distinct geometries. Duplicate check: reviewed categories use select/duplicates.md, fresh categories the vertex hash / "
        "texture md5 of the USD plus the renders (same geometry, texture/colour variant). Considered and not taken: "
        + "; ".join(REJECTED_CATEGORIES) + f". Method and per-model evidence: {RUN_REL}/scripts/select_touching.py, touching_selection.json, "
        "candidate_pool.json, model_geometry.json, used_categories.json.")
    partner_hdr_note = (
        "Partner object of the relation (not grasped; the target is pushed against it, so it must stand still when bumped and the sampler rejects "
        "goals where it moves > 5 mm). Rules: ready, single link; footprint fits every scene support validated for IORS by the table-selection run "
        f"plus the test-round table ({n_sup} supports: {sup_txt}; short side <= min depth - 6 cm = {fmt_cm(limits['partner']['short_max'])} cm, long side <= "
        f"min length / 2 = {fmt_cm(limits['partner']['long_max'])} cm, the sampler's cell + footprint margins as in select/partners/select_partners.py); "
        f"height <= {fmt_cm(P_H_MAX)} cm and <= {P_ASPECT} x its smallest horizontal extent; flat base (>= {P_BASE:.0%} of the footprint in contact); "
        "does not roll. Not required to be graspable. Ranking: pinned > lower aspect / larger base > render check > size closest to the category median; "
        f"pick 2 prefers another geometry. Test spec pins {pinned_p}. Pairs: {pairs_txt}. Rejected ready models and not-ready models are named in the "
        f"first row of each category. Evidence: {RUN_REL}/touching_selection.json (bbox, base contact, table fit per model), renders/grids.")

    def header(role, note):
        bl = blocks[role]
        n_listed = sum(b["n_valid"] for b in bl)
        n_models = sum(b["n_models_folded"] for b in bl)
        n_geom = sum(b["n_geometries"] for b in bl)
        n_exact = n_listed - n_models
        n_tex = n_models - n_geom
        dup = (f"{n_listed} model ids listed; {n_exact} exact duplicates counted once -> {n_models} models (bracket); "
               f"{n_tex} models are texture/colour variants of another listed model -> {n_geom} distinct geometries")
        return {"task": TASK_LABEL, "role": role, "category": str(len(bl)), "model": f"{n_geom} ({n_models})", "selection": "", "pair": "",
                "duplicates": dup, "note": note}

    new_rows = {"target": [header("target", target_hdr_note)], "partner": [header("partner", partner_hdr_note)]}
    for b in ordered("target"):
        p = next(x for x in PAIRS if x["target"] == b["category"])
        new_rows["target"].extend(block_rows(b, "target", "->", p["theme"], p["why"], n_sup, {}))
    for b in ordered("partner"):
        p = next(x for x in PAIRS if x["partner"] == b["category"])
        new_rows["partner"].extend(block_rows(b, "partner", "<-", p["theme"], p["why"], n_sup, partner_lib_notes))

    # ---------------------------------------------------------------------------------------------- rewrite the sheet
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    out, i, replaced = [], 0, set()
    while i < len(rows):
        r = rows[i]
        if r["task"].split(" ")[0] == PROB and r["role"] in ("target", "partner"):
            j = i + 1
            while j < len(rows) and not rows[j]["task"]:
                j += 1
            out.extend(new_rows[r["role"]])
            replaced.add(r["role"])
            i = j
            continue
        out.append(r)
        i += 1
    if replaced != {"target", "partner"}:
        sys.exit(f"could not find the IORS-4 target / partner header rows (found {replaced})")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"rewrote the IORS-4 target ({len(new_rows['target']) - 1} rows) and partner ({len(new_rows['partner']) - 1} rows) blocks; {len(out)} rows -> {args.csv}")
    if not args.no_render:
        sys.path.insert(0, TABLE_SCRIPTS)
        from update_selection_csv import render  # same styling as the rest of the sheet

        render(out)


if __name__ == "__main__":
    main()
