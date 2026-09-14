"""Curated task -> category plan (hand-picked from the ranked pools), with counts pulled from the data."""
import json
OUT = "/tmp/grasp_exp"
d = json.load(open(f"{OUT}/task_category_candidates.json")); P = d["pool"]
C = json.load(open(f"{OUT}/category_table.json"))["categories"]

PLAN = {
 "object_scaling (20) - prefer symmetric": {
   "main": ["bottle_of_pop", "bottle_of_wine", "bottle_of_black_pepper", "bottle_of_beer", "wineglass", "lemon", "bottle_of_water", "beaker", "can_of_soda", "ice_cube",
            "bottle_of_medicine", "apple", "canned_food", "jar_of_spaghetti_sauce", "beeswax_candle", "bottle_of_essential_oil", "bottle_of_catsup", "bell_pepper", "potato", "soda_cup"],
   "mode": {"beaker": "V", "wineglass": "V", "soda_cup": "V", "beeswax_candle": "C (override: taxonomy says device)"},
   "alternates": ["round_bottom_flask", "toy_dice", "pillar_candle", "bottle_of_sake", "bottle_of_olive_oil", "beefsteak_tomato", "half_lime", "pine_cone", "spray_can", "salt_shaker", "pepper_shaker", "beer_glass", "pear", "eggplant", "zucchini", "shiitake", "vidalia_onion", "bottle_of_salsa", "pencil", "crayon", "marker"],
   "note": "GraspGen pipeline (object starts in hand). Scoring mode per category from the BDDL taxonomy: C = confidence gate then CoM (default here: bottles, cans, produce, plaything), V = vessel (wineglass, beaker, soda_cup: default CoM), H = confidence-first (none needed; beeswax_candle is tagged device.n.01 by the taxonomy -> override to C). Rotationally symmetric bodies; vase (32) and pillar_candle are too wide for the gripper; cap/lid reserved for attachment; toy_dice moved to disembedding."},
 "mental_rotation (19) - must NOT be symmetric": {
   "main": ["toy_figure", "carving_knife", "walker", "coffee_cup", "mug", "statue", "spatula", "teaspoon", "mouse", "teacup", "tablefork", "soup_ladle", "hook", "crab",
            "cellulose_tape_dispenser", "tongs", "vegetable_peeler", "hammer", "banana"],
   "mode": {"spatula": "H", "teaspoon": "H", "tablefork": "H", "hammer": "H", "tongs": "H", "vegetable_peeler": "H", "hook": "H", "carving_knife": "H", "mouse": "H (device; harmless)", "soup_ladle": "H (override: taxonomy says vessel)",
            "coffee_cup": "V", "mug": "V", "teacup": "V", "cellulose_tape_dispenser": "V", "toy_figure": "C", "walker": "C", "statue": "C", "crab": "C", "banana": "C"},
   "alternates": ["sunflower", "sandal", "paintbrush", "high_heel", "baby_shoe", "cream_pitcher", "toothbrush", "gaming_controller", "stapler", "plier", "drill", "corkscrew", "mallet", "trowel", "hairbrush", "scrub_brush", "watch", "whistle", "pruner", "open_end_wrench", "shears", "dustpan"],
   "note": "GraspGen pipeline (object starts in hand). Modes: H = confidence-first (handled implements: fork, spoon, spatula, hammer, tongs, peeler, ladle), V = vessels (mug, cups, ladle-like containers; default CoM so the body stays at the wrist), C = confidence gate then CoM. 3-D shapes with no rotational symmetry; bounding-box heuristics cannot detect this reliably, so the list is curated by semantics. sandal moved to IORS-under and sunflower to IORS-inside; statue (10 ready, 4 pinchable, 23 cm) and carving_knife (5 ready, blade+handle, mirror-symmetric only) take their slots. toy_figure needs per-model curation (35 ready, 23 pinchable). Only 8 categories have >= 9 ready models."},
 "disembedding (19) - settles on a flat surface": {
   "main": ["toy_dice", "pack_of_cigarettes", "video_game", "box_of_cereal", "box_of_sanitary_napkins", "box_of_baking_mix", "notebook", "tortilla_chips", "box_of_cookies", "box_of_yogurt",
            "bar_soap", "box_of_shampoo", "box_of_cane_sugar", "box_of_tissues", "carton_of_milk", "chili", "pencil_case", "cookie_cutter", "cellulose_tape"],
   "alternates": ["soap_dispenser", "bottle_of_milk", "post_it", "wall_socket", "seashell", "sliced_lemon", "dice", "box_of_rice", "box_of_granola_bars", "box_of_oatmeal", "boxed_ink_cartridge", "half_lime"],
   "note": "A1 pipeline (object rests on a table/tray), so every entry must be pinchable IN ITS RESTING POSE: lying flat with the second-smallest extent <= 9.5 cm (dice, cigarette pack, game cartridge, soap, tape, cookie cutter, chili, the 6 small notebooks) or a box that stands stably on its narrow face with that face <= 9 cm wide (cereal 5.2, baking mix 5.0, sanitary napkins 7.0, cane sugar 8.0, tissues 7.5, milk carton 8.4, shampoo box 5.6, pencil case 6.3) or a small box (box_of_yogurt 6.5 x 6.4 x 5.4). Dropped from the previous list because a flat book/board cannot be pinched from above (needs a finger under it): hardback, paperback_book, letter, board_game, chopping_board, magazine, comic_book, notepad, chocolate_bar. Counts are 'ready models that pass this pose test', see disembedding_table_graspable.json. Only 5 categories have >= 9 such models."},
 "configuration_matching - articulation (9)": {
   "main": ["bottom_cabinet", "top_cabinet", "bottom_cabinet_no_top", "fridge", "dishwasher", "carton", "oven", "clothes_dryer", "hinged_jar"],
   "alternates": ["washer", "recycling_bin", "laptop", "toolbox", "microwave"],
   "note": "openable + articulated. Scene objects have a ready model in the 7 challenge scenes: bottom_cabinet (house_single_floor xqmipa, house_double_floor_lower qacthv, house_double_floor_upper nddvba, Rs_int bamfsz/dajebq/slgzfc, restaurant_diner olgoza, office immwzb/rvunhj), top_cabinet (Rs_int fqhdne/lsyzkh, office fqitbw), bottom_cabinet_no_top (house_double_floor_upper ncgmqr), fridge (house_single_floor dszchb, Rs_int xyejdx, restaurant jtqazu/juwaoh), dishwasher (ynrhuk single/lower, Rs_int xlmier), oven (ffitak single/lower), clothes_dryer (single xsuyua). carton and hinged_jar are non-scene openables. washer/microwave models in the scenes are NOT ready; laptop has one ready model (Rs_int nvulcs)."},
 "configuration_matching - attachment (9)": {
   "main": ["lid", "cap", "license_plate", "dip_candle", "cork", "light_bulb", "wreath", "dartboard", "webcam"],
   "alternates": ["bow", "holly_decoration", "fire_alarm", "antlers", "menu", "address", "poster", "skateboard_wheel_assembly", "sticker"],
   "note": "Table-graspability in the resting pose: cap, cork, dip_candle, light_bulb, webcam and the wreath (ring cross-section) are fine; a lid lying knob-up must be grasped by its knob (check per model that the knob is in the collision mesh, otherwise rest it rim-up); license_plate (30 x 15 x 0.4 cm) and dartboard (45 cm disc) cannot be pinched off a table and would have to start leaning against something or be handed over. graspable (male) part -> partner: lid -> hingeless_jar x4, storage_box x3, teapot x2, kettle x2, canister, tupperware, stockpot, pressure_cooker, crock_pot, steamer_basket, compost_bin, toy_box, trash_can; cap -> bottle_of_whiskey x4, bottle_of_water, bottle_of_vodka, oil_bottle, jar_of_clove, jar_of_tumeric, soda_cup, paper_cup, bird_feeder, deodorant_stick, round_bottom_flask (+ one generic cap for 19 bottle categories and one for 9 cartons); license_plate -> car (2 ready cars, present in the house scenes); dip_candle -> candle_holder (5 ready); cork -> wine_bottle / rum_bottle / vinegar_bottle; light_bulb -> table_lamp; wreath / dartboard / bow / holly_decoration / fire_alarm / antlers / menu / address / poster -> wall_nail (scene object, 1 ready); webcam -> monitor; skateboard_wheel_assembly -> skateboard_deck; sticker -> license_plate. Only lid and cap have >= 9 ready models; everything else is 1-5."},
 "IORS - ontop (4)": {
   "main": ["pen", "shampoo_dispenser", "cherry_tomato", "pickle"],
   "alternates": ["french_fries", "sliced_lemon", "garlic_clove", "bottle_of_milk", "pack_of_kielbasa"],
   "note": "pen -> desk (scene), shampoo_dispenser -> countertop (scene), cherry_tomato -> plate (46), pickle -> cutting_board (5)."},
 "IORS - under (4)": {
   "main": ["french_fries", "beer_glass", "sandal", "briefcase"],
   "alternates": ["toy_figure (small under furniture, disjoint models from mental_rotation)", "walker", "tennis_ball + baseball + pool_ball (1 model each)", "toy_dice", "sliced_lemon", "tablespoon"],
   "note": "two small-under-small (the TOP object is grasped and placed on the bottom one): platter (6) under french_fries (11); coaster (2) or saucer (4) under beer_glass (3). Two small-under-big (the small object is grasped and placed under furniture with enough clearance: desk 0.75 m, breakfast_table 0.70 m, pedestal_table 0.69 m, bed; not coffee_table 0.41 m): sandal (9 ready, 5 fit by width, 7 pinchable) under bed / desk; briefcase (7 ready) under desk - its body is 9.7 cm thick so only 3 models pinch by thickness, the rest need a handle grasp (verify the handle is in the collision mesh). sandal therefore leaves mental_rotation (walker keeps footwear covered there; statue takes the slot)."},
 "IORS - inside (4)": {
   "main": ["garlic_clove", "sunflower", "quartz", "hard_candy"],
   "alternates": ["lily (10 models, not 'ready' only because of an automatic non-VRay material flag; renders fine)", "dahlia_flower (2, same flag, renders fine)", "pen", "plant_stem", "cinnamon_stick", "chopstick", "sliced_lemon", "french_fries", "spinach"],
   "note": "garlic_clove -> bowl (39), sunflower -> vase (32, fillable), quartz -> saucepot (12), hard_candy -> hingeless_jar (8). sunflower therefore leaves mental_rotation (carving_knife takes the slot).\n\n"
           "Flowers: there is no tidy shared synset. In the BDDL/WordNet taxonomy sunflower.n.01 and dahlia.n.01 sit under flower.n.01 (together with marigold, poinsettia, rose), while lily.n.01 and tulip.n.01 sit under liliaceous_plant.n.01 / bulbous_plant.n.01; the lowest common ancestor of all three is vascular_plant.n.01, which also covers trees and grass, so it cannot be used as the task synset. Treat 'flower' as a benchmark-level group {sunflower 18, lily 10 (flagged), dahlia_flower 2 (flagged)} instead; marigold (10 ready) and poinsettia are bushy 12-25 cm bodies with no pinchable stem, so they do not help.\n\n"
           "Vase heights (TODO when building instances): OmniGibson's Inside tests the flower's AABB centre against the vase's fillable volume, so the centre must be below the rim: flower length <= ~2 x vase interior depth. Plan: pick the 9 tallest TABLE-scale vases (heights 0.27-0.47 m: aegxpb 0.269, meetii 0.301, hliauj 0.315, saenda 0.417, tjrbxv 0.473, ...; skip the four floor vases of 0.77-1.5 m, 37-53 cm wide) and rescale each flower so that its centre ends up inside the vase (e.g. sunflower 0.11-0.24 m fits as is; lily 0.42-0.47 m needs ~0.6-0.9 x scale for a 0.3 m vase, or a 0.47 m vase). Keep stems >= 6 mm thick after scaling so they remain pinchable."},
 "IORS - nextto (4)": {
   "main": ["marker", "pencil", "cinnamon_stick", "parsnip"],
   "alternates": ["ruler", "colored_pencil", "star_anise", "carrot", "whisk", "bottle_of_milk"],
   "note": "marker <-> colored_pencil (20/20), pencil <-> ruler (69/5), cinnamon_stick <-> star_anise (8/3), parsnip <-> carrot (6/3). Rods roll a little on a table; the vegetable pair settles best."},
}

used = {}
lines = ["# Curated category plan (2026-09-04)", "",
         "ready = no unprocessed QA complaint (BEHAVIOR knowledge-base definition); pinch = thinnest extent <= 8.5 cm; width-fit = two extents <= 9.5 cm (top-down body pinch); bbox = median over ready models [m], canonical frame.", ""]
for task, spec in PLAN.items():
    modes = spec.get("mode", {})
    lines += [f"## {task}", "", "| # | category | ready | pinch | width-fit | median bbox | GraspGen mode |" if modes else "| # | category | ready | pinch | width-fit | median bbox |", "|---|---|---|---|---|---|---|" if modes else "|---|---|---|---|---|---|"]
    for i, c in enumerate(spec["main"], 1):
        key = c.split(" (")[0].split(" +")[0]
        s = P.get(key); m = f" {modes.get(key, 'C')} |" if modes else ""
        if s is None:
            r = C.get(key, {}); lines.append(f"| {i} | {c} | {r.get('n_ready')} | - | - | {r.get('median_bbox_ready')} |{m}")
        else:
            lines.append(f"| {i} | {c} | {s['n_ready']} | {s['graspable_pinch']} | {s['graspable_width']} | {s['median_bbox']} |{m}")
        used.setdefault(key, []).append(task.split(" (")[0])
    n9 = sum(1 for c in spec["main"] if (P.get(c.split(" (")[0].split(" +")[0]) or C.get(c.split(" (")[0].split(" +")[0], {})).get("n_ready", 0) >= 9)
    lines += ["", f"{n9}/{len(spec['main'])} with >= 9 ready models. Alternates: " + ", ".join(f"{c} ({(P.get(c) or C.get(c, {})).get('n_ready')})" for c in spec["alternates"]), "", spec["note"], ""]
dups = {c: t for c, t in used.items() if len(t) > 1}
lines += ["## Overlaps between tasks", "", ("none" if not dups else "\n".join(f"- {c}: {t}" for c, t in dups.items())), ""]
open(f"{OUT}/curated_category_plan.md", "w").write("\n".join(lines))
missing = [c for spec in PLAN.values() for c in spec["main"] + spec["alternates"] if c.split(" (")[0].split(" +")[0] not in P and c.split(" (")[0].split(" +")[0] not in C]
print("missing categories:", missing); print("overlaps:", dups)
print(f"total main categories: {sum(len(s['main']) for s in PLAN.values())}")
