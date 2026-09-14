#!/usr/bin/env python3
"""Contact sheets of the selected other-category distractors (pool + reserve) per target category, for the semantic /
quality check of the related-category pool: ``grids/pool_other/<category>__p<k>.png`` (5 x 3 tiles per page, yaw 0
head-camera crops, label = category/model, tier, pool/reserve). Reads ``distractor_pools.json``."""
from __future__ import annotations

import argparse
import json
import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("cg", os.path.join(HERE, "compose_grids.py"))
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

COLS, ROWS, W = 5, 3, 300


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", default=os.path.join(ROOT, "distractor_pools.json"))
    ap.add_argument("--renders", default=os.path.join(ROOT, "renders"))
    ap.add_argument("--out", default=os.path.join(ROOT, "grids", "pool_other"))
    args = ap.parse_args()
    pools = json.load(open(args.pools))
    meta = json.load(open(os.path.join(args.renders, "render_meta.json")))
    proj = cg.Projector(meta)
    f, fb = cg.font(15), cg.font(18)
    os.makedirs(args.out, exist_ok=True)
    index = {}
    for cat, d in pools["categories"].items():
        rows = [r for r in d["other_category"] if r.get("pool") in ("pool", "reserve")]
        rows.sort(key=lambda r: (0 if r["pool"] == "pool" else 1, r.get("tier") or 9))
        pages = []
        per = COLS * ROWS
        for pi in range(math.ceil(len(rows) / per)):
            chunk = rows[pi * per:(pi + 1) * per]
            n_rows = math.ceil(len(chunk) / COLS)
            cell_h = W + 58
            img = Image.new("RGB", (COLS * (W + 6) + 6, 30 + n_rows * (cell_h + 6)), (255, 255, 255))
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), f"{cat}: selected other-category distractors, page {pi + 1} (yaw 0, head camera at the init stance)", fill=(0, 0, 0), font=fb)
            for i, r in enumerate(chunk):
                rr, c = divmod(i, COLS)
                x, y = 6 + c * (W + 6), 30 + rr * (cell_h + 6)
                png = os.path.join(args.renders, f"{r['category']}__{r['model']}__y0.png")
                if os.path.exists(png) and r.get("bbox"):
                    crop = cg.tight_crop(png, proj.object_box(r["bbox"], 0.0))
                    s = min(W / crop.width, W / crop.height)
                    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
                    img.paste(crop, (x + (W - crop.width) // 2, y + (W - crop.height) // 2))
                else:
                    dr.rectangle([x, y, x + W, y + W], outline=(200, 0, 0))
                lab = [f"{r['category']}/{r['model']}", f"tier {r.get('tier')}  {r['pool']}  {r.get('selection', '')}", (r.get("render") or "")[:34]]
                for j, ln in enumerate(lab):
                    dr.text((x + 2, y + W + 2 + 18 * j), ln[:34], fill=(0, 0, 0) if j else (20, 20, 160), font=f)
            path = os.path.join(args.out, f"{cat}__p{pi + 1}.png")
            img.save(path)
            pages.append(os.path.relpath(path, ROOT))
        index[cat] = pages
        print(f"{cat}: {len(rows)} selected other-category models on {len(pages)} page(s)")
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)


if __name__ == "__main__":
    main()
