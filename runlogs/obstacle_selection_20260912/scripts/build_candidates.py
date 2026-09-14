#!/usr/bin/env python3
"""Candidate obstacle categories / models for ``camera_relocalization-0``: objects that a robot naturally has to steer
around when it drives from its init pose to the goal pose (bins, potted plants, boxes, laundry hampers, buckets,
luggage, toys, stools, vacuum cleaners, ...), that stand on the floor in their canonical pose and are small enough to be
dropped on a free floor cell of the seven candidate scenes without blocking a corridor (R1Pro chassis half-diagonal
0.33 m; corridors ~1 m), yet tall enough to be a real obstacle for the base and visible to the head camera.

Picked by hand from the BEHAVIOR category list after (a) a scan of every category with ready models (``--scan``:
median ready-model canonical bbox with largest footprint side 0.12-0.9 m and height 0.10-1.7 m -> ``candidate_scan.json``)
and (b) the object inventory of the seven scenes (every ``<scene>_*.json`` variant: which categories already stand in
which rooms -> ``scene_presence.json``). Every category records the scenes it is natural in (hand annotation,
``natural_scenes`` below; ``in_scene`` = the scenes whose JSON already contains the category). Per category up to
``CAP`` ready models are rendered (seeded random subset when the category is larger; the flagship categories keep 16).

Usage (ispatialgym env, needs bddl):  python build_candidates.py [--scan]
Output: ``candidate_models.json`` ({category: [models]}), ``candidate_categories.json`` (group, why, natural scenes,
in-scene rooms, bboxes), ``scene_presence.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
sys.path.insert(0, REPO)

from ispatialgym.sampling import knowledge as K  # noqa: E402

SCENES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "restaurant_diner",
          "hotel_suite_large", "office_cubicles_right"]
SCENE_DIR = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "scenes")

HOUSES = ["house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int"]
ALL = SCENES
HOUSE_HOTEL = HOUSES + ["hotel_suite_large"]
HOUSE_OFFICE = HOUSES + ["office_cubicles_right"]
GARAGE = ["house_single_floor", "house_double_floor_lower"]  # the two scenes with a garage
KITCHENS = ["house_single_floor", "house_double_floor_lower", "Rs_int", "restaurant_diner"]

#: category -> (group, why it is a candidate, scenes it is natural in)
CATEGORIES = {
    # --- waste bins -------------------------------------------------------------------------------------------------
    "trash_can": ("bins", "household waste bins ~0.35 x 0.36 x 0.41 m (15 ready): kitchens, bathrooms, bedrooms, offices", ALL),
    "public_trash_can": ("bins", "larger lidded / pedal bins ~0.43 x 0.48 x 0.71 m; already stands in 4 of the 7 scenes", ALL),
    "recycling_bin": ("bins", "0.57 x 0.49 x 0.81 m recycling containers: copy room, garage, kitchen", ["office_cubicles_right", "house_single_floor", "house_double_floor_lower", "restaurant_diner"]),
    "compost_bin": ("bins", "0.42 x 0.28 x 0.82 m kitchen compost bin", KITCHENS),
    # --- potted plants ----------------------------------------------------------------------------------------------
    "pot_plant": ("plants", "potted plants, median 0.25 x 0.28 x 0.37 m (27 ready of 62); already stands in all 7 scenes", ALL),
    "plant_pot": ("plants", "empty / soil-filled pots ~0.22 m: garden door, garage, corridor", ["house_single_floor", "house_double_floor_lower", "Rs_int"]),
    "poinsettia": ("plants", "potted poinsettias ~0.27 m: living room, lobby, dining room", ALL),
    "marigold": ("plants", "small potted marigolds 0.20 x 0.19 x 0.14 m (low; size borderline)", HOUSES + ["restaurant_diner"]),
    "pottable_dahlia": ("plants", "dahlia plants 0.34 x 0.17 x 0.16 m (check whether the model has a pot)", HOUSES),
    "plant_pot_stand": ("plants", "0.26 x 0.56 x 0.83 m tiered stand with pots: living room, corridor", HOUSES + ["restaurant_diner", "office_cubicles_right"]),
    # --- boxes, baskets, bags, luggage ------------------------------------------------------------------------------
    "hamper": ("containers", "laundry hampers / baskets ~0.31 x 0.45 x 0.31 m (16 ready): bedroom, bathroom, corridor, hotel", HOUSE_HOTEL),
    "wicker_basket": ("containers", "wicker baskets ~0.31 x 0.41 x 0.28 m: living room, bedroom, hotel", HOUSE_HOTEL),
    "shopping_basket": ("containers", "0.49 x 0.32 x 0.26 m shopping baskets: kitchen, entryway, diner", KITCHENS),
    "carton": ("containers", "cardboard boxes ~0.43 x 0.48 x 0.34 m (6 ready): deliveries in entryways / corridors, garage, copy room", ALL),
    "packing_box": ("containers", "0.38 x 0.47 x 0.22 m moving box: garage, office, entryway", HOUSE_OFFICE),
    "storage_box": ("containers", "0.26 x 0.28 x 0.11 m plastic storage boxes (low; size borderline)", HOUSE_OFFICE),
    "box_of_fruit": ("containers", "0.39 x 0.56 x 0.15 m produce crate: diner kitchen, kitchen, garage", KITCHENS),
    "pop_case": ("containers", "0.13 x 0.39 x 0.13 m cases of cans (13 ready; low): diner, garage, kitchen", KITCHENS),
    "cedar_chest": ("containers", "0.40 x 0.36 x 0.20 m small chests: bedroom, living room", HOUSE_HOTEL),
    "toolbox": ("containers", "0.26 x 0.61 x 0.35 m toolbox: garage, utility room", GARAGE),
    "tackle_box": ("containers", "0.23 x 0.60 x 0.24 m tackle box: garage", GARAGE),
    "suitcase": ("luggage", "0.43 x 0.62 x 0.20 m suitcase (1 ready of 4): hotel, bedroom, entryway, corridor", HOUSE_HOTEL),
    "backpack": ("luggage", "0.49 x 0.40 x 0.47 m backpack: child's room, entryway, office, hotel", ALL),
    "satchel": ("luggage", "0.46 x 0.54 x 0.24 m satchel: office, entryway, hotel", HOUSE_HOTEL + ["office_cubicles_right"]),
    "tote": ("luggage", "0.34 x 0.24 x 0.47 m tote bag: entryway, office, hotel", ALL),
    "shoulder_bag": ("luggage", "shoulder bags ~0.10 x 0.23 x 0.28 m (2 ready of 11)", ALL),
    "briefcase": ("luggage", "0.28 x 0.33 x 0.10 m briefcases lying flat (7 ready; low)", ["office_cubicles_right", "hotel_suite_large"] + HOUSES),
    "violin_case": ("luggage", "0.30 x 0.65 x 0.13 m case lying flat (low)", HOUSES),
    "paper_bag": ("containers", "grocery paper bags ~0.12 x 0.20 x 0.24 m: kitchen, entryway", KITCHENS),
    "plastic_bag": ("containers", "0.17 x 0.31 x 0.39 m filled plastic bag: kitchen, entryway", KITCHENS),
    "gift_box": ("containers", "0.20 x 0.18 x 0.18 m gift boxes: living room, bedroom, lobby", HOUSE_HOTEL + ["office_cubicles_right"]),
    "burlap_bag": ("containers", "0.27 x 0.37 x 0.24 m sack: diner kitchen, garage, utility room", ["restaurant_diner"] + GARAGE),
    "flour_sack": ("containers", "0.27 x 0.37 x 0.24 m flour sack: diner kitchen", ["restaurant_diner", "house_single_floor"]),
    "petfood_bag": ("containers", "0.27 x 0.37 x 0.24 m pet-food bag: kitchen, utility room, garage", HOUSES),
    "soil_bag": ("containers", "0.27 x 0.37 x 0.24 m potting-soil bag: garage, garden door", GARAGE),
    "bag_of_mulch": ("containers", "0.33 x 0.30 x 0.59 m mulch bag: garage", GARAGE),
    "cooler": ("containers", "0.51 x 1.03 x 0.52 m picnic cooler (over the footprint limit; kept for the record)", GARAGE + ["restaurant_diner"]),
    # --- cleaning / utility -----------------------------------------------------------------------------------------
    "bucket": ("cleaning", "buckets ~0.25 x 0.30 x 0.19 m: kitchen, bathroom, garage, utility room", ALL),
    "mop_bucket": ("cleaning", "0.35 x 0.59 x 0.43 m janitor mop bucket: office corridor, diner kitchen, garage", ALL),
    "bucket_of_paint": ("cleaning", "paint buckets ~0.16 x 0.15 x 0.19 m (small): garage, utility room", GARAGE + ["office_cubicles_right"]),
    "ice_bucket": ("cleaning", "0.34 x 0.41 x 0.39 m ice bucket: bar, hotel", ["restaurant_diner", "hotel_suite_large"]),
    "vacuum": ("cleaning", "upright vacuum cleaners ~0.37 x 0.30 x 0.85 m: house, hotel, office", ALL),
    "carpet_sweeper": ("cleaning", "0.25 x 0.28 x 0.83 m carpet sweeper: house, hotel, diner", HOUSE_HOTEL + ["restaurant_diner"]),
    "fuel_can": ("cleaning", "0.23 x 0.11 x 0.31 m jerrycan: garage", GARAGE),
    # --- stools and small seats -------------------------------------------------------------------------------------
    "taboret": ("seats", "stools ~0.37 x 0.36 x 0.58 m (6 ready); already stands in 3 of the 7 scenes", ALL),
    "footstool": ("seats", "footstools ~0.41 x 0.36 x 0.35 m: living room, bedroom, hotel", HOUSE_HOTEL),
    "ottoman": ("seats", "ottomans ~0.57 x 0.66 x 0.33 m (7 ready); already stands in 4 of the 7 scenes", HOUSE_HOTEL),
    "music_stool": ("seats", "0.41 x 0.42 x 0.81 m stools: living room, bar", HOUSES + ["restaurant_diner"]),
    "bar_stool": ("seats", "0.52 x 0.59 x 0.77 m bar stool: diner bar, kitchen island", ["restaurant_diner", "house_single_floor", "house_double_floor_lower"]),
    "folding_chair": ("seats", "folding chairs ~0.88 x 0.48 x 0.59 m (check the canonical pose): garage, office, conference hall", GARAGE + ["office_cubicles_right"]),
    "straight_chair": ("seats", "dining / desk chairs ~0.50 x 0.46 x 0.78 m (51 ready; a pulled-out chair mid-path); already stands in 4 of the 7 scenes", ALL),
    "swivel_chair": ("seats", "office chairs ~0.62 x 0.64 x 0.97 m (56 ready; a chair rolled into the corridor); already stands in 4 of the 7 scenes", HOUSE_OFFICE + ["hotel_suite_large"]),
    "highchair": ("seats", "0.33 x 0.30 x 0.78 m high chairs: diner, kitchen", ["restaurant_diner", "house_single_floor", "house_double_floor_lower"]),
    "rocking_chair": ("seats", "0.66 x 0.45 x 0.70 m rocking chairs: living room, bedroom (footprint borderline)", HOUSES),
    "stand": ("seats", "0.56 x 0.63 x 0.46 m stands (luggage rack?); already stands in the hotel suite", ["hotel_suite_large"] + HOUSES),
    # --- toys, sports gear, pets ------------------------------------------------------------------------------------
    "toy_car": ("toys", "ride-on / push toy cars ~0.57 x 0.26 x 0.28 m: child's room, living room, corridor", HOUSES),
    "teddy_bear": ("toys", "teddy bears ~0.26 x 0.27 x 0.19 m left on the floor: child's room, bedroom", HOUSE_HOTEL),
    "doll": ("toys", "0.45 x 0.37 x 0.35 m doll: child's room", HOUSES),
    "soccer_ball": ("toys", "0.25 m soccer balls (rolling risk when settled): child's room, garage, garden door", HOUSES),
    "volleyball": ("toys", "0.24 m volleyball (rolling risk): garage, child's room", HOUSES),
    "bowling_ball": ("toys", "0.22 m bowling ball (rolling risk): garage, closet", GARAGE),
    "alphabet_abacus": ("toys", "0.15 x 0.32 x 0.39 m toddler abacus: child's room", ["house_single_floor"] + HOUSES),
    "free_weight": ("toys", "0.29 m dumbbells / kettlebells on the floor: bedroom, living room, hotel", HOUSE_HOTEL),
    "boxing_gloves": ("toys", "0.37 x 0.25 x 0.18 m gloves on the floor: bedroom, garage", HOUSES),
    "skates": ("toys", "0.34 x 0.37 x 0.14 m pair of skates: entryway, child's room (low)", HOUSES),
    "rolled_yoga_mat": ("toys", "0.13 x 0.13 x 0.61 m rolled mat (standing? check pose): bedroom, living room, hotel", HOUSE_HOTEL),
    "pet_bed": ("toys", "0.66 x 0.88 x 0.23 m pet bed (footprint borderline): living room, kitchen, corridor", HOUSES),
    "litter_box": ("toys", "0.53 x 0.36 x 0.52 m litter box: bathroom, utility room, corridor", HOUSES),
    "birdcage": ("toys", "0.36 x 0.31 x 0.50 m birdcage on the floor: living room", HOUSES),
    "terrarium": ("toys", "0.39 x 0.43 x 0.84 m floor terrarium: living room, lobby", HOUSES + ["office_cubicles_right"]),
    "balloon": ("toys", "0.41 x 0.33 x 0.33 m balloon (weighted? check pose): child's room, dining room", HOUSES + ["restaurant_diner"]),
    # --- floor-standing appliances, electronics, small furniture ----------------------------------------------------
    "floor_lamp": ("appliances", "floor lamps ~0.46 x 0.38 x 1.35 m (15 ready; tall but slim); already stands in 3 of the 7 scenes", HOUSE_HOTEL),
    "standing_loudspeaker": ("appliances", "0.26 x 0.26 x 0.44 m floor speakers: living room, bedroom", HOUSES),
    "electric_fan": ("appliances", "0.30 x 0.41 x 0.53 m floor fan: house, office, diner kitchen", ALL),
    "space_heater": ("appliances", "0.26 x 0.28 x 0.39 m space heater: bedroom, office, living room", HOUSE_OFFICE + ["hotel_suite_large"]),
    "dehumidifier": ("appliances", "0.30 x 0.39 x 0.59 m dehumidifier: bathroom, garage, office", HOUSE_OFFICE),
    "water_cooler": ("appliances", "0.39 x 0.39 x 1.02 m water cooler: office corridor / lobby, diner", ["office_cubicles_right", "restaurant_diner"]),
    "water_dispenser": ("appliances", "0.26 x 0.25 x 0.60 m water dispenser: office, diner", ["office_cubicles_right", "restaurant_diner"]),
    "shredder": ("appliances", "0.25 x 0.34 x 0.47 m paper shredder: office", ["office_cubicles_right"]),
    "printer": ("appliances", "printers ~0.46 x 0.53 x 0.35 m on the floor of an office / copy room", ["office_cubicles_right"]),
    "coatrack": ("appliances", "coat racks ~0.52 x 0.48 x 1.66 m (tall, slim; height borderline): entryway, office, diner", ALL),
    "shoe_rack": ("appliances", "0.31 x 0.68 x 0.81 m shoe rack: entryway, corridor, hotel", HOUSE_HOTEL),
    "magazine_rack": ("appliances", "0.24 x 0.08 x 0.30 m magazine racks (thin; size borderline): living room, lobby", HOUSES + ["office_cubicles_right"]),
    "camera_tripod": ("appliances", "0.72 x 0.83 x 1.28 m tripod (footprint borderline): office, living room", HOUSE_OFFICE),
    "easel": ("appliances", "0.54 x 0.77 x 1.66 m easel (height / footprint borderline): living room, meeting room", HOUSES + ["office_cubicles_right"]),
    "serving_cart": ("appliances", "serving carts ~0.48 x 0.75 x 0.85 m; already stands in 3 of the 7 scenes", ["restaurant_diner", "hotel_suite_large", "office_cubicles_right", "house_double_floor_upper"]),
    "stroller": ("appliances", "0.85 x 0.51 x 0.93 m stroller (footprint borderline): entryway, corridor, diner", HOUSES + ["restaurant_diner"]),
    "hose_cart": ("appliances", "0.67 x 0.59 x 1.00 m hose reel cart: garage", GARAGE),
    "power_washer": ("appliances", "0.25 x 0.66 x 0.82 m pressure washer: garage", GARAGE),
    "car_wheel": ("appliances", "0.58 m tyres lying flat: garage", GARAGE),
    "beer_keg": ("appliances", "0.42 x 0.42 x 0.60 m keg: diner bar / kitchen", ["restaurant_diner"]),
    "snack_rack": ("appliances", "0.44 x 0.80 x 0.48 m display rack (footprint borderline): diner", ["restaurant_diner"]),
    # --- footwear and soft items left on the floor ------------------------------------------------------------------
    "gym_shoe": ("clutter", "sneakers ~0.31 x 0.11 x 0.14 m (22 ready; low, size borderline): entryway, bedroom, hotel", HOUSE_HOTEL),
    "rubber_boot": ("clutter", "0.33 x 0.12 x 0.27 m boots: entryway, garage", HOUSES),
    "hiking_boot": ("clutter", "0.27 x 0.10 x 0.15 m boots (low): entryway", HOUSES),
    "pillow": ("clutter", "pillows ~0.45 x 0.55 x 0.19 m on the floor (127 ready; low): bedroom, living room, hotel", HOUSE_HOTEL),
}

CAP_DEFAULT = 12
CAP = {"pot_plant": 16, "hamper": 16, "trash_can": 16, "floor_lamp": 16, "straight_chair": 8, "swivel_chair": 8, "gym_shoe": 6,
       "pillow": 6, "marigold": 6, "pop_case": 6}
SEED = 0


def scene_presence():
    """category -> scene -> sorted room types (from every scene JSON variant that has objects_info)."""
    pres = defaultdict(lambda: defaultdict(set))
    models_in_scene = defaultdict(set)
    for s in SCENES:
        for p in sorted(glob.glob(os.path.join(SCENE_DIR, s, "json", f"{s}_*.json"))):
            with open(p) as f:
                d = json.load(f)
            if "objects_info" not in d:
                continue
            for o in d["objects_info"]["init_info"].values():
                a = o.get("args", {})
                if "category" not in a:
                    continue
                for r in a.get("in_rooms") or ["?"]:
                    pres[a["category"]][s].add(r.rsplit("_", 1)[0])
                models_in_scene[a["category"]].add(a["model"])
    return {c: {s: sorted(r) for s, r in sc.items()} for c, sc in pres.items()}, {c: sorted(v) for c, v in models_in_scene.items()}


def scan(pres):
    m2c = K._model_to_category()
    cats = defaultdict(list)
    for m, c in m2c.items():
        cats[c].append(m)
    rows = []
    for c, ms in cats.items():
        ready = [m for m in ms if K.is_ready(m)]
        bbs = [b for b in (K.canonical_bbox(m) for m in ready) if b]
        if not bbs:
            continue
        med = [statistics.median(b[i] for b in bbs) for i in range(3)]
        foot, h = max(med[0], med[1]), med[2]
        if 0.12 <= foot <= 0.9 and 0.10 <= h <= 1.7:
            rows.append({"category": c, "n_total": len(ms), "n_ready": len(ready), "median_bbox": [round(v, 3) for v in med],
                         "in_scene": pres.get(c, {}), "candidate": c in CATEGORIES})
    rows.sort(key=lambda r: r["category"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="also write candidate_scan.json (every obstacle-sized category)")
    args = ap.parse_args()
    pres, models_in_scene = scene_presence()
    with open(os.path.join(ROOT, "scene_presence.json"), "w") as f:
        json.dump({"scenes": SCENES, "categories": pres, "models_in_scene": models_in_scene}, f, indent=1, sort_keys=True)
    if args.scan:
        with open(os.path.join(ROOT, "candidate_scan.json"), "w") as f:
            json.dump(scan(pres), f, indent=1)
    models, info = {}, {}
    rng = random.Random(SEED)
    for cat, (group, why, natural) in CATEGORIES.items():
        all_models = K.models_of_category(cat, ready_only=False)
        ready = [m for m in all_models if K.is_ready(m)]
        cap = CAP.get(cat, CAP_DEFAULT)
        chosen = sorted(ready)
        if len(chosen) > cap:
            # keep the models that already stand in one of the 7 scenes, fill up with a seeded random subset
            keep = [m for m in chosen if m in models_in_scene.get(cat, [])][:cap]
            rest = [m for m in chosen if m not in keep]
            rng.shuffle(rest)
            chosen = sorted(keep + rest[:cap - len(keep)])
        models[cat] = chosen
        info[cat] = {"group": group, "why": why, "natural_scenes": list(natural), "in_scene": pres.get(cat, {}),
                     "n_total": len(all_models), "n_ready": len(ready), "n_rendered": len(chosen), "rendered": chosen,
                     "not_rendered": sorted(set(ready) - set(chosen)),
                     "bbox": {m: [round(v, 3) for v in (K.canonical_bbox(m) or [])] for m in chosen}}
        if not ready:
            print(f"WARNING {cat}: no ready model")
    with open(os.path.join(ROOT, "candidate_models.json"), "w") as f:
        json.dump(models, f, indent=1)
    with open(os.path.join(ROOT, "candidate_categories.json"), "w") as f:
        json.dump(info, f, indent=1)
    n = sum(len(v) for v in models.values())
    print(f"{len(models)} categories, {n} models (of {sum(i['n_ready'] for i in info.values())} ready) -> candidate_models.json")
    by_group = defaultdict(int)
    for cat, i in info.items():
        by_group[i["group"]] += len(i["rendered"])
    print("per group: " + ", ".join(f"{g} {k}" for g, k in sorted(by_group.items())))


if __name__ == "__main__":
    main()
