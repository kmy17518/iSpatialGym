#!/usr/bin/env python3
"""Candidate categories / models for ``world_alignment-2`` (world horizontal): elongated wall items whose long axis
(or reference edge) is level when they hang correctly - bars, rails, shelves, wide boards / screens, text signs, and
rod-like objects that are displayed horizontally on wall hooks (bats, rolling pins, tools).

Picked by hand from the BEHAVIOR category list after a geometry scan of every category with ready models
(``scan`` below: median ready-model bbox with long extent 0.2-1.5 m, long / mid >= 1.4, thin <= 0.30 m) plus a name
search for wall / rod vocabulary. Per category only the ready models whose canonical bbox can hang on a wall in front
of the head camera are kept (``MODEL_FILTER``: long extent <= ~1.4 m, tall vertical radiators / shelving units and
mini rulers excluded). Categories already reviewed for world vertical (towel_rack, hook_rack, signpost, ...) are
re-rendered and re-reviewed here with the horizontal rubric.

Usage (ispatialgym env, needs bddl):  python build_candidates.py [--scan]
Output: ``candidate_models.json`` ({category: [models]}) and ``candidate_categories.json`` (counts / notes).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
sys.path.insert(0, REPO)

from ispatialgym.sampling import knowledge as K  # noqa: E402

#: category -> (group, why it is a candidate)
CATEGORIES = {
    # wall-native items with a level long axis / edge
    "towel_rack": ("wall", "horizontal bar on brackets; world-vertical round-1 category (weak there: no readable up)"),
    "hook_rack": ("wall", "0.96 m rail with hooks"),
    "shelf": ("wall", "floating / bracket wall shelves (models <= 1.5 m long; shelving units excluded)"),
    "bookcase_shelf": ("wall", "single shelf plank 0.74 x 0.26 m"),
    "wall_mounted_tv": ("wall", "landscape screens 0.56-1.32 m wide (world-vertical drop: 180-deg ambiguity does not matter for level)"),
    "whiteboard": ("wall", "landscape boards with a marker tray"),
    "bulletin_board": ("wall", "landscape cork boards ~1.08 x 0.75 m"),
    "address": ("wall", "house address plaque 0.58 x 0.16 m"),
    "decorative_sign": ("wall", "cut-out word signs (LOVE, Welcome HOME): text baseline; world-vertical round 1"),
    "signpost": ("wall", "wide indoor text / pictogram plates; world-vertical round 1"),
    "license_plate": ("wall", "0.31 x 0.16 m plates with text (aspect 2); world-vertical round 2"),
    "hanger": ("wall", "clothes hangers: level bar under a hook; world-vertical round 1"),
    "air_conditioner": ("wall", "wall split unit 0.88 x 0.30 x 0.21 m"),
    "wall_mounted_light": ("wall", "vanity light bars 0.62 m and sconces"),
    "radiator": ("wall", "wall panel radiators (tall vertical models excluded)"),
    "range_hood": ("wall", "wall hoods 0.72-1.05 m wide (thick; likely H <= 1)"),
    "frame": ("wall", "plain frames 1.46-1.76 m (over the size limit; kept for the record)"),
    "board": ("wall", "plywood board 1.12 x 0.74 m"),
    "picture": ("wall", "landscape framed prints (aspect >= 1.3 subset); world-vertical round 1"),
    "painting": ("wall", "landscape paintings (aspect >= 1.3 subset); world-vertical round 1 (pinned)"),
    "poster": ("wall", "portrait movie poster 0.62 x 0.89 m (horizontal = short edge / title baseline)"),
    # rod-like objects displayed horizontally on wall hooks / racks
    "baseball_bat": ("rod", "0.75-0.81 m bats (wall display racks)"),
    "golf_club": ("rod", "1.0 m club"),
    "fishing_rod": ("rod", "1.33 m rod (wall rod racks; over the 1.3 m limit)"),
    "skateboard_deck": ("rod", "1.04 m deck with trucks (horizontal wall display); configuration_matching-3 category"),
    "longboard": ("rod", "1.04 m longboard"),
    "guitar": ("rod", "1.04 m guitars (horizontal wall hangers exist); world-vertical round 1"),
    "trumpet": ("rod", "0.91 m trumpet (wall mount)"),
    "violin": ("rod", "0.60 m violin; world-vertical not selected"),
    "tennis_racket": ("rod", "0.82 m racket; world-vertical not selected"),
    "rolling_pin": ("rod", "0.40-0.42 m pins (kitchen wall hooks)"),
    "ruler": ("rod", "0.33 m rulers (workshop wall hook); 0.12-0.17 m models excluded"),
    "drumstick": ("rod", "0.38-0.41 m sticks, 1 cm thick (display)"),
    "poker": ("rod", "0.66 m fireplace poker"),
    "axe": ("rod", "0.53 m axe (wall display)"),
    "umbrella": ("rod", "0.5-1.0 m closed umbrellas; world-vertical round 2"),
    "handle": ("rod", "0.62 x 0.13 m bar handle (grab bar?)"),
    "rake": ("rod", "garden rake 1.02 m (garage wall)"),
    "shovel": ("rod", "0.74 m shovel (garage wall)"),
    "wall_mounted_loudspeaker": ("wall", "0.31 m speaker / soundbar"),
}

#: explicit model subsets where the ready list contains unusable sizes (None = every ready model)
MODEL_FILTER = {
    "shelf": ["adgrkb", "apiaid", "cyckht", "hrfwyd", "ibrbnl", "iruewm", "kxvbui", "notszn", "qrmiur", "wiiyfk", "wvkefp", "xogqqm", "xrnzjv", "zadqqv"],
    "radiator": ["arfdpk", "arsjbe", "cbjgea", "dwmloz", "jlhyiq", "jxrlul", "lnvush", "wntwxx", "xnoiwp", "zgjthd", "znmvtw"],
    "range_hood": ["beanpv", "duqgcu", "iqbpie", "lejkwg", "uokzpw", "ylsdmp"],
    "wall_mounted_tv": ["bfehsl", "cwdngq", "dxyfxf", "gffrzl", "ggfznh", "leeqgt", "ovzajm", "sodbbr", "ylvjhb"],
    "picture": ["fhkzlm", "ytxhyl", "chfljy", "watgtj", "hffaxw", "szpupz", "etixod", "qavxsz", "cltbty", "hhxttu", "jpfyrq", "wfdvzv"],
    "painting": ["ejhanr", "fwuole", "tpmcuj", "xlzxry", "qvmegb", "gardkz"],
    "ruler": ["rutqab", "ujizkx", "vmnaih"],
}


def scan():
    m2c = K._model_to_category()
    cats = {}
    for m, c in m2c.items():
        cats.setdefault(c, []).append(m)
    rows = []
    for c, ms in cats.items():
        ready = [m for m in ms if K.is_ready(m)]
        bbs = [sorted(b) for b in (K.canonical_bbox(m) for m in ready) if b]
        if not bbs:
            continue
        med = [statistics.median(x[i] for x in bbs) for i in range(3)]
        if 0.2 <= med[2] <= 1.5 and med[2] / max(med[1], 1e-6) >= 1.4 and med[0] <= 0.30:
            rows.append({"category": c, "n_total": len(ms), "n_ready": len(ready), "median_sorted_bbox": [round(v, 3) for v in med],
                         "aspect": round(med[2] / max(med[1], 1e-6), 2)})
    rows.sort(key=lambda r: r["category"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="also write candidate_scan.json (every elongated category)")
    args = ap.parse_args()
    if args.scan:
        with open(os.path.join(ROOT, "candidate_scan.json"), "w") as f:
            json.dump(scan(), f, indent=1)
    models, info = {}, {}
    for cat, (group, why) in CATEGORIES.items():
        all_models = K.models_of_category(cat, ready_only=False)
        ready = [m for m in all_models if K.is_ready(m)]
        keep = MODEL_FILTER.get(cat)
        chosen = [m for m in ready if keep is None or m in keep]
        if keep is not None:
            missing = sorted(set(keep) - set(ready))
            if missing:
                print(f"WARNING {cat}: filtered models not ready / unknown: {missing}")
        models[cat] = chosen
        info[cat] = {"group": group, "why": why, "n_total": len(all_models), "n_ready": len(ready), "rendered": chosen,
                     "bbox": {m: [round(v, 3) for v in (K.canonical_bbox(m) or [])] for m in chosen}}
    with open(os.path.join(ROOT, "candidate_models.json"), "w") as f:
        json.dump(models, f, indent=1)
    with open(os.path.join(ROOT, "candidate_categories.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"{len(models)} categories, {sum(len(v) for v in models.values())} models -> candidate_models.json")


if __name__ == "__main__":
    main()
