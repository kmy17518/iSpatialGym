#!/usr/bin/env python3
"""Split the candidate categories into reviewer assignments (``assign/rev_XX.json``): whole categories, balanced by
model count (greedy, largest first), each with its grid paths and the scene hints of ``candidate_categories.json``."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "candidate_categories.json")) as f:
        info = json.load(f)
    cats = sorted(info, key=lambda c: (-len(info[c]["rendered"]), c))
    buckets = [{"reviewer": f"rev_{i:02d}", "task": "obst", "categories": [], "grids": [], "hints": {}, "n_models": 0} for i in range(args.n)]
    for c in cats:
        b = min(buckets, key=lambda b: (b["n_models"], len(b["categories"])))
        # exact category match: "plant_pot_*" must not pick up plant_pot_stand_00.png, "bucket_*" not bucket_of_paint_00.png
        grids = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "renders", "grids", f"{c}_*.png"))
                       if re.fullmatch(rf"{re.escape(c)}_\d+\.png", os.path.basename(p)))
        b["categories"].append(c)
        b["grids"].extend(grids)
        b["hints"][c] = {"why": info[c]["why"], "natural_scenes": info[c]["natural_scenes"], "in_scene": info[c]["in_scene"],
                         "n_models": len(info[c]["rendered"]), "bbox_m": info[c]["bbox"]}
        b["n_models"] += len(info[c]["rendered"])
    os.makedirs(os.path.join(ROOT, "assign"), exist_ok=True)
    for b in buckets:
        b["categories"].sort()
        b["grids"].sort()
        with open(os.path.join(ROOT, "assign", b["reviewer"] + ".json"), "w") as f:
            json.dump(b, f, indent=1)
        print(f"{b['reviewer']}: {len(b['categories'])} categories, {b['n_models']} models, {len(b['grids'])} grids: {', '.join(b['categories'])}")


if __name__ == "__main__":
    main()
