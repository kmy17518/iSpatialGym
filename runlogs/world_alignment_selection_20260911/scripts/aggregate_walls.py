#!/usr/bin/env python3
"""Aggregate the wall survey (``results/<scene>__<room>.json`` from survey_walls.py) into

* ``wall_candidates.json`` - the wall candidate specification: per scene / room / wall object the viable regions
  (world xy segment ``p0``-``p1`` at hanging height ``hang_z``, outward normal, length, largest flat + unobstructed
  square patch ``max_patch_m`` (objects up to ``max_patch_m - 0.10`` fit), number of survey spots and collision-free
  arc bases / pushable bases behind it, whether the sampler wrote a full instance there) plus, per scene, a
  ``task_spec`` block in the shape of ``task_custom_lists.json`` -> ``scenes`` (rooms + ``fixed.floor`` + ``fixed.wall``
  + ``fixed.wall_regions``) restricted to rooms with a validated region;
* ``wall_candidates.md`` - the same as tables, with the sampler's rejection reasons for rooms without a region;
* ``maps/<scene>_walls.png`` - floor plan (room instance map) with every survey spot (green = viable, orange = flat
  and unobstructed but no collision-free / pushing-capable base, red = not flat or obstructed), the merged regions
  (thick green) and the spots where a full instance was generated (blue).

No simulator needed (map PNGs + json).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets")
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]
RUN = "runlogs/world_alignment_selection_20260911"


def room_map(scene: str, resolution: float = 0.1):
    """(instance map at ``resolution``, ins_id -> room instance name, map_size) exactly like OmniGibson's SegmentationMap."""
    layout = os.path.join(ASSETS, "scenes", scene, "layout")
    ins = cv2.imread(os.path.join(layout, "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(layout, "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    size = int(ins.shape[0] * 0.01 / resolution)
    ins = cv2.resize(ins, (size, size), interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(sem, (size, size), interpolation=cv2.INTER_NEAREST)
    with open(os.path.join(ASSETS, "metadata", "room_categories.txt")) as f:
        cats = [l.rstrip() for l in f]
    sem_to_ins = defaultdict(list)
    for ins_id in np.unique(ins):
        if ins_id == 0:
            continue
        r, c = np.argwhere(ins == ins_id)[0]
        sem_to_ins[int(sem[r, c])].append(int(ins_id))
    names = {}
    for sem_id, ids in sem_to_ins.items():
        for i, ins_id in enumerate(ids):
            names[ins_id] = f"{cats[sem_id - 1]}_{i}"
    return ins, names, size


def world_to_px(xy, size, resolution):
    """world (x, y) -> image (col, row): row = y / res + size / 2, col = x / res + size / 2 (OmniGibson maps)."""
    return float(xy[0]) / resolution + size / 2.0, float(xy[1]) / resolution + size / 2.0


def font(size=12):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_map(scene: str, rooms: dict, out_path: str, resolution: float = 0.05, target_width: int = 1400):
    ins, names, size = room_map(scene, resolution)
    rng = np.random.default_rng(3)
    img = np.full((size, size, 3), 255, np.uint8)
    outdoor = ("garden", "patio", "backyard", "yard", "pool", "balcony", "terrace", "porch")
    indoor_mask = np.zeros_like(ins, dtype=bool)
    for ins_id, name in names.items():
        col = (200 + rng.integers(0, 45, 3)).astype(np.uint8)
        if any(s in name for s in outdoor):
            col = np.array([235, 235, 235], np.uint8)  # outdoor rooms were not surveyed
        else:
            indoor_mask |= ins == ins_id
        img[ins == ins_id] = col
    # crop to the indoor rooms (+ margin) and upscale so that small scenes stay readable
    rr, cc = np.nonzero(indoor_mask if indoor_mask.any() else ins > 0)
    m = int(1.0 / resolution)
    r0, r1 = max(0, rr.min() - m), min(size, rr.max() + m)
    c0, c1 = max(0, cc.min() - m), min(size, cc.max() + m)
    crop = img[r0:r1, c0:c1]
    scale = max(1, int(round(target_width / max(1, crop.shape[1]))))
    scale = min(scale, 12)
    im = Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    f = font(11)

    def px(xy):
        c, r = world_to_px(xy, size, resolution)
        return (c - c0) * scale, (r - r0) * scale

    # room labels at the centroid
    for ins_id, name in names.items():
        rr, cc = np.nonzero(ins == ins_id)
        if len(rr) < 4 or any(s in name for s in outdoor):
            continue
        dr.text(((cc.mean() - c0) * scale - 3 * len(name), (rr.mean() - r0) * scale - 6), name, fill=(70, 70, 70), font=f)
    n_v = n_o = n_r = 0
    for room, rec in rooms.items():
        for s in rec.get("spots", []):
            c, r = px(s["point"][:2])
            n = s["normal"]
            # a short tick along the normal (into the room)
            c2, r2 = px((s["point"][0] + n[0] * 0.15, s["point"][1] + n[1] * 0.15))
            if s.get("viable"):
                col, n_v = (30, 160, 60), n_v + 1
            elif s.get("skipped"):
                col = (170, 170, 170)
            elif s.get("target_ok"):
                col, n_o = (240, 150, 20), n_o + 1
            else:
                col, n_r = (210, 60, 60), n_r + 1
            dr.line([(c, r), (c2, r2)], fill=col, width=max(1, scale // 3))
            rad = max(1.5, scale * 0.4)
            dr.ellipse([c - rad, r - rad, c + rad, r + rad], fill=col)
        for reg in rec.get("regions", []):
            n = reg["normal"]
            p0 = (reg["p0"][0] + n[0] * 0.08, reg["p0"][1] + n[1] * 0.08)
            p1 = (reg["p1"][0] + n[0] * 0.08, reg["p1"][1] + n[1] * 0.08)
            col = (0, 120, 40) if reg.get("validated") else ((30, 160, 60) if not reg.get("tested") else (200, 120, 0))
            dr.line([px(p0), px(p1)], fill=col, width=max(3, scale // 2))
            mid = ((p0[0] + p1[0]) / 2 + n[0] * 0.3, (p0[1] + p1[1]) / 2 + n[1] * 0.3)
            dr.text(px(mid), reg["id"], fill=col, font=font(11))
        for inst in rec.get("instances", []):
            if inst.get("n_written"):
                c, r = px(inst["point"][:2])
                rad = max(4, scale)
                dr.ellipse([c - rad, r - rad, c + rad, r + rad], outline=(20, 60, 220), width=2)
                b = (inst.get("instance") or {}).get("base_pose2d")
                if b:
                    bc, br = px(b[:2])
                    dr.ellipse([bc - rad * 0.7, br - rad * 0.7, bc + rad * 0.7, br + rad * 0.7], fill=(20, 60, 220))
                    dr.line([(bc, br), (c, r)], fill=(20, 60, 220), width=1)
    legend = [f"{scene}: wall survey for world_alignment-1 (hang height 1.45 m; reference target painting 0.50 x 0.42 m)",
              f"green spot = viable (flat, unobstructed, collision-free arc base with pushing reach) [{n_v}]; orange = flat + unobstructed only [{n_o}]; red = not flat / obstructed [{n_r}]; grey = flat + unobstructed, not tested (strided)",
              "thick green = merged viable region (dark green = a full instance was generated there; orange = tested, no instance); blue ring = instance spot, blue dot = robot base of that instance",
              "image axes: x = right, y = down (OmniGibson map convention); 1 px = %.3f m" % (resolution / scale)]
    W = im.size[0]
    pad = Image.new("RGB", (max(W, 900), im.size[1] + 16 * len(legend) + 8), "white")
    pad.paste(im, (0, 16 * len(legend) + 8))
    d2 = ImageDraw.Draw(pad)
    for i, t in enumerate(legend):
        d2.text((6, 4 + 16 * i), t, fill=(0, 0, 0), font=f)
    pad.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--min-length", type=float, default=0.5, help="regions shorter than this (m) are listed but not put in the task_spec block")
    args = ap.parse_args()
    by_scene = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        with open(p) as f:
            rec = json.load(f)
        by_scene[rec["scene"]][rec["room"]] = rec
    os.makedirs(os.path.join(args.out, "maps"), exist_ok=True)
    spec = {"_comment": ("Wall candidates for world_alignment-1 (world vertical), surveyed 2026-09-11 with the instance sampler "
                         "(runlogs/world_alignment_selection_20260911/scripts/survey_walls.py). A region is a run of wall where the sampler found flat, "
                         "unobstructed patches at hanging height 1.45 m with collision-free base poses on the 0.6-0.95 m arc from which the right arm "
                         "reaches the lower corners of the aligned target (reference target painting/dacihj, 0.50 x 0.42 m); 'validated' = a complete "
                         "instance (init settled, BEHAVIOR-validated, :init true, goal false, >= 5 objects visible, goal feasible) was written at the "
                         "region's middle spot. Coordinates are world xy in metres, normal = unit vector out of the wall into the room, max_patch_m = "
                         "largest square patch that is flat and unobstructed (objects up to max_patch_m - 0.10 m fit). task_spec = the scenes block "
                         "for task_custom_lists.json['world_alignment-1'] (rooms with a validated region; fixed.wall / wall_regions are informational "
                         "for the current sampler, which searches the room's walls itself)."),
            "reference_target": "painting/dacihj", "hang_height_m": 1.45, "scenes": {}}
    md = ["# Wall candidates for world_alignment-1 (world vertical)", "",
          "Survey: `scripts/survey_walls.py` (dense wall spots every 0.25 m at 1.45 m height, the sampler's own flat-patch / front-clearance / "
          "arc-base / pushing-reach checks, then one full sampler instance per region). Reference target painting/dacihj (0.50 x 0.42 m). "
          "`max patch` = largest flat + unobstructed square (m); `bases` = collision-free arc bases over the region's spots, `reach` = those with "
          "pushing reach of a lower corner; `instance` = full instance written by the sampler at the region's middle spot (rejections otherwise).", ""]
    summary = {}
    for scene in SCENES:
        rooms = by_scene.get(scene)
        if not rooms:
            md.append(f"## {scene}\n\n(not surveyed)\n")
            continue
        sc = {"rooms": {}, "task_spec": {"rooms": [], "fixed": {"floor": {}, "wall": [], "wall_regions": []}}}
        md.append(f"## {scene}\n")
        md.append("| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        n_val = 0
        rooms_ok = []
        for room in sorted(rooms):
            rec = rooms[room]
            rr = {"status": rec.get("status"), "floor": rec.get("floor"), "walls": rec.get("walls"), "n_spots": rec.get("n_spots"),
                  "n_target_ok": rec.get("n_target_ok"), "n_viable": rec.get("n_viable"), "regions": [], "error": rec.get("error"),
                  "sim_error": rec.get("sim_error")}
            inst_by_region = defaultdict(list)
            for i in rec.get("instances", []):
                inst_by_region[i["region"]].append(i)
            if rec.get("status") != "ok":
                md.append(f"| {room} | - | - | - | - | - | - | - | - | {rec.get('status')}: {str(rec.get('error', ''))[:120]} |")
            elif not rec.get("regions"):
                why = f"{rec.get('n_spots', 0)} spots, {rec.get('n_target_ok', 0)} flat + clear, {rec.get('n_viable', 0)} viable"
                md.append(f"| {room} | {', '.join(rec.get('walls') or [])} | - | - | - | - | - | {why} | - | no viable region |")
            for reg in rec.get("regions", []):
                insts = inst_by_region.get(reg["id"], [])
                ok = any(i.get("n_written") for i in insts)
                if ok:
                    n_val += 1
                rj = "; ".join(f"{i.get('rejections')} {str(i.get('error', ''))[:80]}".strip() for i in insts if not i.get("n_written"))
                inst_txt = "yes" if ok else ("not tested" if not insts else f"no ({rj})")
                r = dict(reg)
                r["validated"] = bool(ok)
                r["instances"] = [{k: i.get(k) for k in ("spot", "point", "n_written", "rejections", "instance", "work_dir", "error") if k in i} for i in insts]
                rr["regions"].append(r)
                md.append(f"| {room} | {reg['wall_name']} | {reg['id']} | ({reg['p0'][0]:.2f}, {reg['p0'][1]:.2f}) -> ({reg['p1'][0]:.2f}, {reg['p1'][1]:.2f}) | "
                          f"({reg['normal'][0]:.2f}, {reg['normal'][1]:.2f}) | {reg['length_m']:.2f} | {reg['max_patch_m']:.1f} | {reg['n_spots']} | "
                          f"{reg['n_bases']} / {reg['n_reach']} | {inst_txt} |")
                if ok and reg["length_m"] >= args.min_length:
                    ts = sc["task_spec"]
                    if room not in ts["rooms"]:
                        ts["rooms"].append(room)
                        rooms_ok.append(room)
                    ts["fixed"]["floor"].setdefault("room", room)
                    if reg["wall_name"] not in ts["fixed"]["wall"]:
                        ts["fixed"]["wall"].append(reg["wall_name"])
                    ts["fixed"]["wall_regions"].append({"room": room, "wall": reg["wall_name"], "id": f"{room}/{reg['wall_name']}/{reg['id']}",
                                                        "p0": reg["p0"], "p1": reg["p1"], "normal": reg["normal"], "hang_z": reg["hang_z"],
                                                        "length_m": reg["length_m"], "max_object_m": round(max(0.0, reg["max_patch_m"] - 0.10), 2)})
            sc["rooms"][room] = rr
        sc["n_validated_regions"] = n_val
        sc["rooms_with_validated_region"] = rooms_ok
        spec["scenes"][scene] = sc
        summary[scene] = {"rooms_surveyed": len(rooms), "rooms_ok": rooms_ok, "n_regions": sum(len(r.get("regions", [])) for r in rooms.values()),
                          "n_validated": n_val,
                          "length_validated_m": round(sum(reg["length_m"] for r in rooms.values() for reg in r.get("regions", []) if reg.get("validated")), 1)}
        md.append("")
        md.append(f"Rooms with a validated region: {', '.join(rooms_ok) if rooms_ok else 'none'}. Map: `maps/{scene}_walls.png`.\n")
        try:
            draw_map(scene, rooms, os.path.join(args.out, "maps", f"{scene}_walls.png"))
        except Exception as e:  # maps are a convenience
            md.append(f"(map failed: {e!r})\n")
    spec["summary"] = summary
    with open(os.path.join(args.out, "wall_candidates.json"), "w") as f:
        json.dump(spec, f, indent=1)
    with open(os.path.join(args.out, "wall_candidates.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
