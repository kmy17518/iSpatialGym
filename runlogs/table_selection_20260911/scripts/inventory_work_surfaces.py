#!/usr/bin/env python3
"""Geometric first pass (no simulator): list every table / work-surface object of the candidate scenes.

Reads the 2026-challenge stable scene jsons and the asset metadata (bbox_size, base_link_offset) and writes
``inventory.json`` / ``inventory.csv`` with, per object: scene, name, category, model, rooms, world footprint
(yaw-aligned bbox extents), estimated top height, and a first-pass verdict:

* ``keep``    - plausible work surface, goes to the sampler validation,
* ``low`` / ``high`` - top height outside [0.55, 1.20] m (kept for the sampler only when ``--keep-all``),
* ``small``   - footprint < 0.30 m on a side or < 0.12 m^2,
* ``no_room`` - no ``in_rooms`` (partial-room loading would skip the object).

Also lists the ``under`` partners (bed / sofa / couch) so that IORS-1 pairs can be built.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCENES_DIR = os.path.join(REPO, "Behavior-1K", "datasets", "2026-challenge-task-instances", "scenes")
ASSETS_DIR = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")

SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]

#: sampler default (ispatialgym.sampling.tasks.base.TABLE_CATEGORIES) plus work surfaces present in the scenes
TABLE_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar",
                    "nightstand", "console_table", "side_table")
EXTRA_SURFACES = ("commercial_kitchen_table", "cubicle", "booth", "bottom_cabinet", "metal_bottom_cabinet", "serving_cart",
                  "stand", "shelf", "bench", "ottoman", "bottom_cabinet_no_top")
UNDER_PARTNERS = ("bed", "sofa", "couch")

TOP_RANGE = (0.55, 1.20)
MIN_SIDE, MIN_AREA = 0.30, 0.12


def quat_to_yaw_and_tilt(q):
    x, y, z, w = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # tilt of the local z axis from the world z axis
    zz = 1 - 2 * (x * x + y * y)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))
    return yaw, tilt


def rot_z(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def load_metadata(category, model):
    p = os.path.join(ASSETS_DIR, category, model, "misc", "metadata.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    args = ap.parse_args()
    rows = []
    for scene in args.scenes:
        path = os.path.join(SCENES_DIR, scene, "json", f"{scene}_stable.json")
        with open(path) as f:
            d = json.load(f)
        init_info = d["objects_info"]["init_info"]
        states = d["state"]["registry"]["object_registry"]
        for name, info in init_info.items():
            a = info["args"]
            cat = a["category"]
            kind = "table" if cat in TABLE_CATEGORIES else "surface" if cat in EXTRA_SURFACES else "under_partner" if cat in UNDER_PARTNERS else None
            if kind is None:
                continue
            meta = load_metadata(cat, a["model"])
            st = states.get(name, {}).get("root_link", {})
            pos, ori = st.get("pos", [0, 0, 0]), st.get("ori", [0, 0, 0, 1])
            scale = a.get("scale", [1, 1, 1])
            yaw, tilt = quat_to_yaw_and_tilt(ori)
            row = {"scene": scene, "name": name, "category": cat, "model": a["model"], "kind": kind,
                   "rooms": ";".join(a.get("in_rooms") or []), "fixed_base": a.get("fixed_base", False),
                   "pos_x": round(pos[0], 3), "pos_y": round(pos[1], 3), "pos_z": round(pos[2], 3), "yaw_deg": round(math.degrees(yaw), 1),
                   "tilt_deg": round(tilt, 1)}
            if meta is not None:
                bb = [meta["bbox_size"][i] * scale[i] for i in range(3)]
                off = [meta["base_link_offset"][i] * scale[i] for i in range(3)]
                # bbox centre in world = root + R * (base_link_offset * scale)  (verified on breakfast_table_rhjoby_0:
                # root z 0.765, offset -0.359, bbox_z 0.781 -> top 0.797 vs 0.813 measured by raycast)
                centre = [pos[i] + rot_z(off, yaw)[i] for i in range(3)]
                ext_w = [abs(bb[0] * math.cos(yaw)) + abs(bb[1] * math.sin(yaw)), abs(bb[0] * math.sin(yaw)) + abs(bb[1] * math.cos(yaw))]
                row.update({"bbox_x": round(bb[0], 3), "bbox_y": round(bb[1], 3), "bbox_z": round(bb[2], 3),
                            "world_ext_x": round(ext_w[0], 3), "world_ext_y": round(ext_w[1], 3),
                            "bottom_z": round(centre[2] - bb[2] / 2, 3), "top_z": round(centre[2] + bb[2] / 2, 3),
                            "area_m2": round(bb[0] * bb[1], 3)})
                reasons = []
                if not a.get("in_rooms"):
                    reasons.append("no_room")
                if kind != "under_partner":
                    if min(bb[0], bb[1]) < MIN_SIDE or bb[0] * bb[1] < MIN_AREA:
                        reasons.append("small")
                    if row["top_z"] < TOP_RANGE[0]:
                        reasons.append("low")
                    elif row["top_z"] > TOP_RANGE[1]:
                        reasons.append("high")
                row["verdict"] = "keep" if not reasons else "+".join(reasons)
            else:
                row["verdict"] = "no_metadata"
            rows.append(row)
    rows.sort(key=lambda r: (SCENES.index(r["scene"]), r["kind"], r["category"], r["name"]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "inventory.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = ["scene", "kind", "category", "name", "model", "rooms", "verdict", "top_z", "bottom_z", "bbox_x", "bbox_y", "bbox_z", "area_m2",
            "world_ext_x", "world_ext_y", "pos_x", "pos_y", "pos_z", "yaw_deg", "tilt_deg", "fixed_base"]
    with open(os.path.join(args.out, "inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # console summary
    for scene in args.scenes:
        sub = [r for r in rows if r["scene"] == scene]
        print(f"== {scene}: {len(sub)} table-like objects")
        for r in sub:
            print(f"  {r['kind']:13s} {r['category']:26s} {r['name']:38s} rooms={r['rooms']:28s} top={r.get('top_z', '?')!s:6s} "
                  f"bbox={r.get('bbox_x', '?')}x{r.get('bbox_y', '?')}x{r.get('bbox_z', '?')} verdict={r['verdict']}")


if __name__ == "__main__":
    main()
