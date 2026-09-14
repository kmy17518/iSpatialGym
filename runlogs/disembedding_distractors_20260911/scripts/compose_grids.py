#!/usr/bin/env python3
"""Review grids from the head-camera renders (render phase of validate_distractors.py).

* ``grids/same/<category>__p<k>.png``: the same-category candidates of a target category, one row per model
  (label + the four yaw crops, tight around the object, native head-camera resolution scaled to a common row
  height), <= ROWS_PER_PAGE rows per page. Input of the look-alike review (review_contract.md).
* ``grids/diff/<category>.png``: contact sheet of the related-category candidates (yaw 0), for the semantic check.
* ``grids/same_contact/<category>.png``: contact sheet of the same-category candidates (yaw 0), for the README.

Crops: the render phase saved the source-tray region of the 1080x1080 head image; the object stands at the tray
centre, so its image box follows from the camera pose, the tray centre and the model's canonical bbox (rotated by
the render yaw); a margin of 12 % is added.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

ROWS_PER_PAGE = 4
ROW_H = 300
LABEL_W = 300
CONTACT_COLS = 5
CONTACT_ROWS = 3
CONTACT_W = 300


def font(size=16):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def quat_rotate_z(yaw_deg, pts):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pts @ R.T


def quat_to_mat(q):  # x, y, z, w
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Projector:
    def __init__(self, meta):
        self.pos = np.array(meta["camera_pose"][0])
        self.R = quat_to_mat(meta["camera_pose"][1])
        cm = meta["camera_model"]
        self.fx, self.fy, self.cx, self.cy = cm["fx"], cm["fy"], cm["cx"], cm["cy"]
        self.crop = meta["crop_uv"]
        self.centre = np.array(meta["centre_xy"])
        self.floor_z = meta["floor_z"]

    def project(self, pts):
        p = (pts - self.pos) @ self.R  # world -> camera
        depth = -p[:, 2]
        u = self.cx + self.fx * p[:, 0] / np.maximum(depth, 1e-6)
        v = self.cy - self.fy * p[:, 1] / np.maximum(depth, 1e-6)
        return np.stack([u, v], axis=1)

    def object_box(self, bbox, yaw_deg, settled_z=None, margin=0.12):
        """Pixel box (in crop coordinates) of the canonical bbox standing at the tray centre, rotated by ``yaw``."""
        hx, hy, hz = bbox[0] / 2, bbox[1] / 2, bbox[2]
        corners = np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, hz)])
        corners = quat_rotate_z(yaw_deg, corners)
        corners[:, 0] += self.centre[0]
        corners[:, 1] += self.centre[1]
        corners[:, 2] += self.floor_z
        uv = self.project(corners)
        u0, v0 = uv[:, 0].min(), uv[:, 1].min()
        u1, v1 = uv[:, 0].max(), uv[:, 1].max()
        w, h = u1 - u0, v1 - v0
        m = margin * max(w, h) + 12
        u0, v0, u1, v1 = u0 - m, v0 - m, u1 + m, v1 + m
        cu0, cv0, cu1, cv1 = self.crop
        return [int(max(cu0, u0) - cu0), int(max(cv0, v0) - cv0), int(min(cu1, u1) - cu0), int(min(cv1, v1) - cv0)]


def tight_crop(png, box):
    img = Image.open(png).convert("RGB")
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, max(x1, x0 + 8)), min(img.height, max(y1, y0 + 8))
    return img.crop((x0, y0, x1, y1))


def scaled(img, height):
    w = max(1, int(round(img.width * height / img.height)))
    return img.resize((w, height), Image.LANCZOS)


def label_lines(e, meta_rec):
    lines = [f"{e['model']}", f"{e.get('selection', '')}" + (f"  score {e['score']}" if e.get("score") is not None else "")]
    bb = e.get("bbox") or [0, 0, 0]
    lines.append(f"bbox {bb[0] * 100:.0f} x {bb[1] * 100:.0f} x {bb[2] * 100:.0f} cm")
    if e.get("exact_group"):
        lines.append("exact dup: " + ", ".join(m for m in e["exact_group"] if m != e["model"]))
    elif e.get("geom_cluster"):
        lines.append(f"geom cluster {e['geom_cluster']}")
    if meta_rec:
        r0 = (meta_rec.get("renders") or {}).get("y0") or {}
        if r0:
            lines.append(f"{r0.get('seg_pixels', '?')} px  tilt {r0.get('tilt_deg', '?')}  {'stable' if meta_rec.get('stable') else 'UNSTABLE'}")
        if meta_rec.get("status") != "ok":
            lines.append(f"render {meta_rec.get('status')}")
    return lines


def draw_label(draw, x, y, lines, f, width):
    yy = y + 6
    for i, ln in enumerate(lines):
        draw.text((x + 6, yy), ln, fill=(0, 0, 0) if i else (20, 20, 160), font=f)
        yy += 19


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=os.path.join(ROOT, "candidates.json"))
    ap.add_argument("--renders", default=os.path.join(ROOT, "renders"))
    ap.add_argument("--out", default=os.path.join(ROOT, "grids"))
    ap.add_argument("--categories", default=None)
    args = ap.parse_args()
    cands = json.load(open(args.candidates))
    meta = json.load(open(os.path.join(args.renders, "render_meta.json")))
    proj = Projector(meta)
    f, fb = font(15), font(18)
    cats = list(cands["targets"]) if not args.categories else args.categories.split(",")
    for sub in ("same", "diff", "same_contact"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    def render_meta(cat, model):
        p = os.path.join(args.renders, "meta", f"{cat}__{model}.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def crops_for(e, yaws):
        out = []
        for k, yaw in enumerate(yaws):
            png = os.path.join(args.renders, f"{e['category']}__{e['model']}__y{k}.png")
            if not os.path.exists(png):
                out.append(None)
                continue
            out.append(tight_crop(png, proj.object_box(e["bbox"], yaw)))
        return out

    index = OrderedDict()
    for cat in cats:
        same = [e for e in cands["same"][cat] if e["size_ok"] and e["ready"]]
        pages = []
        for p0 in range(0, len(same), ROWS_PER_PAGE):
            rows = same[p0:p0 + ROWS_PER_PAGE]
            tiles = []
            for e in rows:
                crops = crops_for(e, meta["yaws_same"])
                tiles.append((e, [scaled(c, ROW_H) if c is not None else None for c in crops]))
            col_w = [max([t[1][k].width for t in tiles if t[1][k] is not None] or [ROW_H]) for k in range(len(meta["yaws_same"]))]
            W = LABEL_W + sum(col_w) + 8 * (len(col_w) + 1)
            H = 40 + len(rows) * (ROW_H + 8)
            img = Image.new("RGB", (W, H), (255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((8, 8), f"{cat}: same-category candidates, page {len(pages) + 1} of {math.ceil(len(same) / ROWS_PER_PAGE)} "
                           f"(head camera at the disembedding init stance; yaws {meta['yaws_same']})", fill=(0, 0, 0), font=fb)
            y = 40
            for e, crops in tiles:
                draw_label(d, 0, y, label_lines(e, render_meta(cat, e["model"])), f, LABEL_W)
                x = LABEL_W + 8
                for k, c in enumerate(crops):
                    if c is not None:
                        img.paste(c, (x, y))
                    else:
                        d.rectangle([x, y, x + col_w[k], y + ROW_H], outline=(200, 0, 0))
                        d.text((x + 8, y + 8), "no render", fill=(200, 0, 0), font=f)
                    x += col_w[k] + 8
                d.line([0, y + ROW_H + 4, W, y + ROW_H + 4], fill=(180, 180, 180))
                y += ROW_H + 8
            path = os.path.join(args.out, "same", f"{cat}__p{len(pages) + 1}.png")
            img.save(path)
            pages.append(os.path.relpath(path, ROOT))
        # contact sheets (paginated: CONTACT_COLS x CONTACT_ROWS tiles per page keep the tiles legible)
        contact_pages = {"same_contact": [], "diff": []}
        for kind, lst, sub in (("same", same, "same_contact"), ("diff", [e for e in cands["diff"][cat] if e["size_ok"] and e["ready"]], "diff")):
            if not lst:
                continue
            n_cols = CONTACT_COLS
            per_page = CONTACT_COLS * CONTACT_ROWS
            n_pages = math.ceil(len(lst) / per_page)
            for pi in range(n_pages):
                chunk = lst[pi * per_page:(pi + 1) * per_page]
                n_rows = math.ceil(len(chunk) / n_cols)
                cell_h = CONTACT_W + 58
                img = Image.new("RGB", (n_cols * (CONTACT_W + 6) + 6, 30 + n_rows * (cell_h + 6)), (255, 255, 255))
                d = ImageDraw.Draw(img)
                d.text((8, 6), f"{cat}: {'same' if kind == 'same' else 'related'}-category candidates, page {pi + 1} of {n_pages} "
                               f"(yaw 0, head camera at the init stance)", fill=(0, 0, 0), font=fb)
                for i, e in enumerate(chunk):
                    r, c = divmod(i, n_cols)
                    x, y = 6 + c * (CONTACT_W + 6), 30 + r * (cell_h + 6)
                    png = os.path.join(args.renders, f"{e['category']}__{e['model']}__y0.png")
                    if os.path.exists(png):
                        crop = tight_crop(png, proj.object_box(e["bbox"], 0.0))
                        s = min(CONTACT_W / crop.width, CONTACT_W / crop.height)
                        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
                        img.paste(crop, (x + (CONTACT_W - crop.width) // 2, y + (CONTACT_W - crop.height) // 2))
                    else:
                        d.rectangle([x, y, x + CONTACT_W, y + CONTACT_W], outline=(200, 0, 0))
                    mr = render_meta(e["category"], e["model"])
                    r0 = ((mr or {}).get("renders") or {}).get("y0") or {}
                    lab = [f"{e['category']}/{e['model']}" if kind == "diff" else e["model"],
                           (f"tier {e.get('tier')}  " if kind == "diff" else "") + f"{e.get('selection', '')}",
                           f"{r0.get('seg_pixels', '?')} px  {'stable' if (mr or {}).get('stable') else 'UNSTABLE'}" if mr else "not rendered"]
                    for j, ln in enumerate(lab):
                        d.text((x + 2, y + CONTACT_W + 2 + 18 * j), ln[:34], fill=(0, 0, 0) if j else (20, 20, 160), font=f)
                path = os.path.join(args.out, sub, f"{cat}__p{pi + 1}.png")
                img.save(path)
                contact_pages[sub].append(os.path.relpath(path, ROOT))
        index[cat] = {"same_pages": pages, "n_same": len(same), "same_contact": contact_pages["same_contact"], "diff_contact": contact_pages["diff"]}
        print(f"{cat}: {len(same)} same-category models on {len(pages)} page(s)")
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)


if __name__ == "__main__":
    main()
