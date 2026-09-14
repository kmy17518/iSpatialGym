"""Rounds 2 and 3 of the category selection + the job list for the grasp tests.

round 1 = the 582 models already validated (models.json; pencil / garlic_clove at their >= 2 cm scale),
round 2 = 10 more categories per task taken from categories already used by ANOTHER task (task requirements still apply;
          models are chosen so that per-model selection counts stay as even as possible; a grasp test is only needed when
          the grasp method (pipeline + constraint + scale) differs from what was already validated),
round 3 = 10 more categories per task that no task has used yet, single-model categories allowed.

Outputs: manifest.json (every (task, category, model) selection with its round and result key), jobs_round23.json (new
grasp-test jobs), rounds_summary.md."""
import json, collections
OUT = "/tmp/glib"; EXP = "/tmp/grasp_exp"; CAP = 12
P = json.load(open(f"{EXP}/task_category_candidates.json"))["pool"]
CT = json.load(open(f"{EXP}/category_table.json")); O = CT["objects"]; CATS = CT["categories"]
r1 = json.load(open(f"{OUT}/models.json")); scaled = {j["base_key"]: j for j in json.load(open(f"{OUT}/models_scaled.json"))}
best = json.load(open(f"{OUT}/best_per_model.json"))                      # which variant (scaled or not) is the round-1 result
FLAGGED_OK = {"lily", "dahlia_flower"}
TAG = {"floating": "f", "tabletop": "t", "tabletop_insert": "i", "flower_stem_top": "fl", "attachment_free": "a", "handle_half": "h"}

# ----------------------------------------------------------------------------------------------------------------------
# selections. (category, relation-or-None, partner/note)
ROUND2 = {
 "object_scaling": [("lid", None, "round disc"), ("cap", None, "round"), ("toy_dice", None, "cube"), ("marker", None, "rod"),
                    ("pack_of_cigarettes", None, "box"), ("box_of_cereal", None, "box"), ("shampoo_dispenser", None, "round body"),
                    ("hard_candy", None, "wrapped, mirror-symmetric"), ("cherry_tomato", None, "round"), ("bar_soap", None, "rounded bar")],
 "mental_rotation": [("sandal", None, ""), ("briefcase", None, "handle"), ("sunflower", None, ""), ("garlic_clove", None, "scaled x2.2-3.8"),
                     ("quartz", None, ""), ("pickle", None, ""), ("webcam", None, ""), ("cookie_cutter", None, "shaped"),
                     ("bell_pepper", None, "same floating grasps as object scaling"), ("potato", None, "same floating grasps as object scaling")],
 "disembedding": [("bottle_of_pop", None, "stands"), ("bottle_of_wine", None, "stands"), ("bottle_of_beer", None, "stands"), ("can_of_soda", None, "stands"),
                  ("bottle_of_medicine", None, "stands"), ("mug", None, "stands"), ("mouse", None, "flat"), ("teaspoon", None, "lies flat"),
                  ("tablefork", None, "lies flat"), ("beaker", None, "stands")],
 "configuration_matching": [("bottle_of_water", None, "F part: partner cap must be fixed"), ("soda_cup", None, "F part: partner cap must be fixed")],
 "articulation": [],
 "IORS": [("apple", "ontop", "plate"), ("coffee_cup", "ontop", "saucer"), ("canned_food", "ontop", "countertop (scene)"),
          ("teacup", "under", "saucer under teacup (teacup grasped)"), ("toy_figure", "under", "under bed (small under big)"),
          ("lemon", "inside", "bowl"), ("ice_cube", "inside", "beer_glass"), ("chili", "inside", "bowl"),
          ("spatula", "nextto", "soup_ladle"), ("jar_of_spaghetti_sauce", "nextto", "canned_food")],
}
ROUND3 = {
 "object_scaling": [("spray_can", None, ""), ("bottle_of_salsa", None, ""), ("bottle_of_olive_oil", None, ""), ("bottle_of_sake", None, ""),
                    ("beefsteak_tomato", None, ""), ("pine_cone", None, ""), ("pear", None, ""), ("orange", None, ""), ("vidalia_onion", None, ""), ("shiitake", None, "")],
 "mental_rotation": [("toothbrush", None, "H"), ("tablespoon", None, "H"), ("high_heel", None, ""), ("baby_shoe", None, ""), ("ginger_root", None, ""),
                     ("gaming_controller", None, ""), ("stapler", None, ""), ("drill", None, "H"), ("mallet", None, "H"), ("cleaver", None, "handle half")],
 "disembedding": [("soap_dispenser", None, ""), ("bottle_of_milk", None, ""), ("box_of_rice", None, "stands on narrow face"), ("box_of_granola_bars", None, ""),
                  ("box_of_oatmeal", None, ""), ("pencil_holder", None, ""), ("bottle_of_shampoo", None, ""), ("conditioner_dispenser", None, ""),
                  ("jar_of_bath_salt", None, ""), ("box_of_butter", None, "")],
 "configuration_matching": [("blackberry", None, "M"), ("holly_decoration", None, "M -> wall_nail"), ("trout", None, "M -> fishing_rod"), ("antlers", None, "M -> wall_nail"),
                            ("skateboard_wheel_assembly", None, "M -> skateboard_deck"), ("fire_alarm", None, "M"), ("bow", None, "M"), ("cabinet_door", None, "M -> cabinet_base"),
                            ("bookcase_baseboard", None, "M -> bookcase_side"), ("menu", None, "M -> displaywall; 1 mm sheet, expected to fail like license_plate")],
 "articulation": [("door", None, "scene; all 7 challenge scenes"), ("openable_window", None, "scene; house_single_floor"), ("sliding_door", None, "scene; house_single_floor"),
                  ("recycling_bin", None, ""), ("blender", None, ""), ("cedar_chest", None, ""), ("toolbox", None, ""), ("cooler", None, ""), ("violin_case", None, ""),
                  ("laptop", None, "Rs_int nvulcs")],
 "IORS": [("pack_of_kielbasa", "ontop", "platter"), ("half_bagel", "ontop", "plate"), ("baguette", "ontop", "cutting_board"),
          ("bottle_of_supplements", "under", "tray under bottle (bottle grasped)"), ("tennis_ball", "under", "under couch / bed"),
          ("lily", "inside", "vase (flower_stem_top)"), ("half_lime", "inside", "bowl"), ("brussels_sprouts", "inside", "bowl"),
          ("salt_shaker", "nextto", "pepper_shaker"), ("pepper_shaker", "nextto", "salt_shaker")],
}
MODES = {"briefcase": "H", "toothbrush": "H", "tablespoon": "H", "drill": "H", "mallet": "H", "cleaver": "H"}
HANDLE_HALF = {"carving_knife", "cleaver"}


def method_for(task, relation, cat):
    if task in ("object_scaling", "mental_rotation"):
        return "graspgen", ("handle_half" if cat in HANDLE_HALF else "floating"), MODES.get(cat, "C")
    if task == "configuration_matching":
        return "a1", "attachment_free", None
    if task == "IORS" and relation == "inside":
        return "a1", ("flower_stem_top" if cat in ("sunflower", "lily", "dahlia_flower") else "tabletop_insert"), None
    return "a1", "tabletop", None


def friendly_key(o):
    a, b, c = sorted(o["bbox"]); return (0 if b <= 0.095 else 1, 0 if a <= 0.085 else 1, a)


def ready_models(cat, task):
    ms = [o for o in O.values() if o["category"] == cat and o["bbox"] and (o["ready"] or cat in FLAGGED_OK)]
    if task == "configuration_matching":                                        # only models that carry an attachment meta-link
        ms = [o for o in ms if o["attachment"]]
    return ms


# ----------------------------------------------------------------------------------------------------------------------
manifest = []; usage = collections.Counter(); existing = {}                      # (cat, model, pipeline, constraint, scale) -> key
for j in r1:                                                                     # round 1: the validated variant per model
    b = best.get(f"{j['category']}__{j['model']}", {}); sc = b.get("scale")
    key = f"{j['key']}__s" if sc else j["key"]
    manifest.append({"round": 1, "task": j["task"], "relation": None, "category": j["category"], "model": j["model"], "key": key, "pipeline": j["pipeline"],
                     "mode": j["mode"], "constraint": j["constraint"], "scale": sc, "reuse_of": None, "grasp_test": True, "partner": None})
    usage[(j["category"], j["model"])] += 1
    existing[(j["category"], j["model"], j["pipeline"], j["constraint"], json.dumps(sc))] = key
    if j["category"] == "carving_knife": existing[(j["category"], j["model"], "graspgen", "handle_half", "null")] = j["key"]
ns = {}; exec(open(f"{EXP}/curated_plan.py").read().split("used = {}")[0], ns)
for c in ns["PLAN"]["configuration_matching - articulation (9)"]["main"]:
    for o in [o for o in O.values() if o["category"] == c and o["ready"] and o["n_links"] > 1][:CAP]:
        manifest.append({"round": 1, "task": "articulation", "relation": None, "category": c, "model": o["model"], "key": None, "pipeline": None, "mode": None,
                         "constraint": None, "scale": None, "reuse_of": None, "grasp_test": False, "partner": None}); usage[(c, o["model"])] += 1

new_jobs = []
def add_round(rnd, plan):
    for task, entries in plan.items():
        for cat, rel, note in entries:
            if task == "articulation":
                ms = [o for o in O.values() if o["category"] == cat and o["ready"] and o["n_links"] > 1]
                for o in sorted(ms, key=lambda o: o["model"])[:CAP]:
                    manifest.append({"round": rnd, "task": task, "relation": None, "category": cat, "model": o["model"], "key": None, "pipeline": None, "mode": None,
                                     "constraint": None, "scale": None, "reuse_of": None, "grasp_test": False, "partner": note}); usage[(cat, o["model"])] += 1
                continue
            pipeline, constraint, mode = method_for(task, rel, cat)
            ms = ready_models(cat, task)
            ms.sort(key=lambda o: (usage[(cat, o["model"])], friendly_key(o)))       # least-used models first, then gripper-friendly
            for o in ms[:CAP]:
                sc = scaled.get(f"{cat}__{o['model']}", {}).get("scale")            # keep the >= 2 cm scale for the thin categories
                ek = (cat, o["model"], pipeline, constraint, json.dumps(sc)); reuse = existing.get(ek)
                key = reuse or f"{cat}__{o['model']}__{'s' if sc else ''}{TAG[constraint]}"
                manifest.append({"round": rnd, "task": task, "relation": rel, "category": cat, "model": o["model"], "key": key, "pipeline": pipeline, "mode": mode,
                                 "constraint": constraint, "scale": sc, "reuse_of": reuse, "grasp_test": reuse is None, "partner": note})
                usage[(cat, o["model"])] += 1
                if reuse is None:
                    existing[ek] = key
                    job = {"task": task, "relation": rel, "category": cat, "model": o["model"], "key": key, "pipeline": pipeline, "mode": mode, "constraint": constraint,
                           "bbox": o["bbox"], "ready": o["ready"], "round": rnd}
                    if sc: job["scale"] = sc; job["scale_rule"] = scaled[f"{cat}__{o['model']}"]["scale_rule"]
                    new_jobs.append(job)
add_round(2, ROUND2); add_round(3, ROUND3)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
json.dump(new_jobs, open(f"{OUT}/jobs_round23.json", "w"), indent=1)

# ----------------------------------------------------------------------------------------------------------------------
by = collections.defaultdict(lambda: collections.Counter())
for m in manifest: by[(m["round"], m["task"])]["models"] += 1; by[(m["round"], m["task"])]["tests"] += m["grasp_test"] and m["reuse_of"] is None and m["round"] > 1
print(f"{'round':5s} {'task':24s} categories models new_tests")
for (rnd, task), c in sorted(by.items()):
    ncat = len({m["category"] for m in manifest if m["round"] == rnd and m["task"] == task})
    print(f"{rnd:5d} {task:24s} {ncat:10d} {c['models']:6d} {c['tests']:9d}")
print(len(new_jobs), "new grasp-test jobs:", dict(collections.Counter(j["pipeline"] for j in new_jobs)), dict(collections.Counter(j["constraint"] for j in new_jobs)))
sel = collections.Counter(usage.values()); print("per-model selection counts (all rounds):", dict(sorted(sel.items())))
print("categories used in >= 3 tasks:", sorted({m["category"] for m in manifest if len({x["task"] for x in manifest if x["category"] == m["category"]}) >= 3}))
