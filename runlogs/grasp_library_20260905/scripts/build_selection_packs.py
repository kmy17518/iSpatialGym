"""Per-(task, category, round) input packs for the test-instance selection + reviewer assignments (<= MAX_STRIPS strips each)."""
import json, glob, os, collections
OUT = "/tmp/glib"; SEL = f"{OUT}/select"; MAX_STRIPS = 34
os.makedirs(f"{SEL}/groups", exist_ok=True); os.makedirs(f"{SEL}/out", exist_ok=True); os.makedirs(f"{SEL}/assign", exist_ok=True)
rows = json.load(open(f"{OUT}/selection_manifest.json"))
review = {}
for p in sorted(glob.glob(f"{OUT}/review/wave*_*.json")):
    for m in json.load(open(p))["models"]: review[m["key"]] = m
REQ = {
 "object_scaling": "In-hand task: the object starts in the gripper (no table). Several different object-in-hand transforms are needed per model, so a model should offer at least 3 visibly different clean grasps. Rotationally symmetric bodies are preferred for this task.",
 "mental_rotation": "In-hand task: the object starts in the gripper (no table). Several different object-in-hand transforms are needed per model, so a model should offer at least 3 visibly different clean grasps. The object must NOT be rotationally symmetric (its orientation must be readable).",
 "disembedding": "Tabletop task: the object rests among other objects on a table / tray and is picked from its resting pose. It must settle stably on a flat surface (it must not roll or topple). The best single grasp matters most.",
 "configuration_matching": "Attachment task: the object is picked from a table and attached to its partner. The pads must leave the mating side of the object completely free.",
 "IORS": "Inter-object relation task: the object is picked from a table and placed so that a spatial relation with a partner object holds (see relation / partner). The grasp must leave the placing side free; for 'inside' the pinch must come from above so the body can be lowered into the container; flowers are held by the stem.",
}
groups = collections.defaultdict(list)
for r in rows:
    if r["task"] == "articulation": continue
    groups[(r["task"], r["category"], r["round"])].append(r)
R1_IORS = {"pen": ("ontop", "desk (scene)"), "shampoo_dispenser": ("ontop", "countertop (scene)"), "cherry_tomato": ("ontop", "plate"), "pickle": ("ontop", "cutting_board"),
           "french_fries": ("under", "platter under fries (fries grasped)"), "beer_glass": ("under", "coaster / saucer under glass (glass grasped)"), "sandal": ("under", "under bed / desk"), "briefcase": ("under", "under desk"),
           "garlic_clove": ("inside", "bowl"), "sunflower": ("inside", "vase (flower_stem_top)"), "quartz": ("inside", "saucepot"), "hard_candy": ("inside", "hingeless_jar"),
           "marker": ("nextto", "colored_pencil"), "pencil": ("nextto", "ruler"), "cinnamon_stick": ("nextto", "star_anise"), "parsnip": ("nextto", "carrot")}
packs = []
for (task, cat, rnd), rs in sorted(groups.items()):
    gid = f"{task}__{cat}__r{rnd}"
    cands, excluded = [], []
    for r in rs:
        key = r["result_key"]; rv = review.get(key, {})
        entry = {"model": r["model"], "key": key, "strip": f"{OUT}/strips/{key}.png", "n_valid": r["n_valid"], "n_clean": r["n_clean"], "n_clean_strict": r["n_clean_strict"],
                 "flags": r["flags"], "grasp_notes": [{"rank": g.get("rank"), "verdict": g.get("verdict"), "flags": g.get("flags", []), "note": g.get("note")} for g in rv.get("grasps", [])],
                 "object_note": rv.get("object_note"), "bbox_mm": None, "scale": r["scale"], "ready": None, "pipeline": r["pipeline"], "constraint": r["constraint"]}
        if (r["n_valid"] or 0) >= 1 and os.path.exists(entry["strip"]): cands.append(entry)
        else: excluded.append({"model": r["model"], "key": key, "reason": r["error"] or "no AG-valid grasp"})
    rel, partner = rs[0]["relation"], rs[0]["partner"]
    if task == "IORS" and rnd == 1 and cat in R1_IORS: rel, partner = R1_IORS[cat]
    packs.append({"group": gid, "task": task, "category": cat, "round": rnd, "relation": rel, "partner": partner, "requirements": REQ[task], "candidates": cands, "excluded": excluded})
# bbox / ready from the category table
O = json.load(open("/tmp/grasp_exp/category_table.json"))["objects"]
for p in packs:
    for c in p["candidates"]:
        o = next((o for o in O.values() if o["category"] == p["category"] and o["model"] == c["model"]), None)
        if o: c["bbox_mm"] = [round(x * 1000) for x in o["bbox"]]; c["ready"] = o["ready"]
for p in packs: json.dump(p, open(f"{SEL}/groups/{p['group']}.json", "w"), indent=1)
# groups without any valid candidate: write their (empty) selection directly
for p in [p for p in packs if not p["candidates"]]:
    json.dump({"group": p["group"], "task": p["task"], "category": p["category"], "round": p["round"], "relation": p["relation"], "n_candidates": 0, "n_opened": 0, "candidates": [], "selected": [], "alternate": None,
               "short": True, "warnings": ["no model with an AG-valid grasp: " + "; ".join(f"{e['model']}: {e['reason']}" for e in p["excluded"])], "confidence": "high"}, open(f"{SEL}/out/{p['group']}.json", "w"), indent=1)
packs = [p for p in packs if p["candidates"]]
# assignments: greedy packing by strip count, largest groups first
packs.sort(key=lambda p: -len(p["candidates"]))
bins = []
for p in packs:
    n = len(p["candidates"])
    for b in bins:
        if b["n"] + n <= MAX_STRIPS: b["groups"].append(p["group"]); b["n"] += n; break
    else: bins.append({"groups": [p["group"]], "n": n})
for i, b in enumerate(bins):
    json.dump({"assignment": f"sel_{i:02d}", "groups": [f"{SEL}/groups/{g}.json" for g in b["groups"]], "n_strips": b["n"]}, open(f"{SEL}/assign/sel_{i:02d}.json", "w"), indent=1)
print(len(packs), "groups,", sum(len(p["candidates"]) for p in packs), "candidate strips,", len(bins), "assignments; strips per assignment:", [b["n"] for b in bins])
print("groups with 0 candidates:", [p["group"] for p in packs if not p["candidates"]])
