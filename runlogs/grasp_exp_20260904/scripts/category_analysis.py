"""Build a per-category table from the BDDL generated data + asset metadata + challenge scenes."""
import csv, glob, json, os, collections, statistics
import numpy as np

B1K = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K"
GD = f"{B1K}/bddl3/bddl/generated_data"
ASSETS = f"{B1K}/datasets/behavior-1k-assets/objects"
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "restaurant_diner", "office_cubicles_right", "hotel_suite_large"]

inv = json.load(open(f"{GD}/object_inventory.json"))
deletion = {r["Object"].strip().split("-")[1] for r in csv.DictReader(open(f"{GD}/deletion_queue.csv"))}
renames = {}
for r in csv.DictReader(open(f"{GD}/object_renames.csv")):
    frm, to = r["Object name"].strip(), r["New Category"].strip()
    if frm and to and "-" in frm and to != "#N/A":
        renames[frm.split("-")[1]] = (frm, to)
complaints = json.load(open(f"{GD}/complaints.json"))
unprocessed = collections.defaultdict(list)
for c in complaints:
    if str(c.get("processed")).lower() != "true":
        unprocessed[c["object"].split("-")[1]].append(c["type"])
cat2syn = {}
for r in csv.DictReader(open(f"{GD}/category_mapping.csv")):
    if r.get("category") and r.get("synset"):
        cat2syn[r["category"].strip()] = r["synset"].strip()
syn_props = json.load(open(f"{GD}/propagated_annots_params.json"))

# scene inventories (challenge scenes, stable state)
scene_models = collections.defaultdict(set)
for sc in SCENES:
    f = f"{B1K}/datasets/2026-challenge-task-instances/scenes/{sc}/json/{sc}_stable.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for name, info in d["objects_info"]["init_info"].items():
        a = info["args"]
        if "category" in a and "model" in a:
            scene_models[(a["category"], a["model"])].add(sc)

objects = {}
for orig_name, provider in inv["providers"].items():
    cat, mid = orig_name.split("-")
    if mid in deletion:
        continue
    if mid in renames:
        cat = renames[mid][1]
    bbox = inv["bounding_box_sizes"].get(mid) or inv["bounding_box_sizes"].get(orig_name)
    meta = f"{ASSETS}/{cat}/{mid}/misc/metadata.json"
    n_links = None; meta_links = []
    if os.path.exists(meta):
        md = json.load(open(meta)); n_links = len(md.get("link_bounding_boxes") or {"base_link": 0}); meta_links = sorted((md.get("meta_links") or {}).keys())
        if bbox is None: bbox = md.get("bbox_size")
    objects[mid] = {"category": cat, "model": mid, "ready": mid not in unprocessed, "complaints": sorted(set(unprocessed.get(mid, []))), "bbox": bbox,
                    "n_links": n_links, "meta_links": inv["meta_links"].get(orig_name, []), "attachment": inv["attachment_pairs"].get(orig_name, {}),
                    "scenes": sorted(scene_models.get((cat, mid), [])), "on_disk": os.path.exists(meta)}

cats = collections.defaultdict(list)
for o in objects.values():
    cats[o["category"]].append(o)
rows = {}
for cat, objs in cats.items():
    syn = cat2syn.get(cat); props = syn_props.get(syn, {}) if syn else {}
    ready = [o for o in objs if o["ready"]]
    bb = [o["bbox"] for o in ready if o["bbox"]] or [o["bbox"] for o in objs if o["bbox"]]
    med = [round(statistics.median(b[i] for b in bb), 3) for i in range(3)] if bb else None
    nl = [o["n_links"] for o in objs if o["n_links"]]
    rows[cat] = {"synset": syn, "n_models": len(objs), "n_ready": len(ready), "n_on_disk": sum(o["on_disk"] for o in objs),
                 "props": sorted(k for k in props if k in ("rigidBody", "cloth", "softBody", "liquid", "rope", "openable", "attachable", "assembleable", "fillable", "sceneObject", "toggleable", "sliceable", "breakable", "deformable", "physicalSubstance", "visualSubstance", "macroPhysicalSubstance", "microPhysicalSubstance", "needsOrientation")),
                 "median_bbox_ready": med, "max_links": max(nl) if nl else None, "n_articulated": sum(1 for o in objs if (o["n_links"] or 1) > 1),
                 "ready_models": sorted(o["model"] for o in ready), "scenes_ready": sorted({s for o in ready for s in o["scenes"]}),
                 "scene_models_ready": {o["model"]: o["scenes"] for o in ready if o["scenes"]},
                 "attachment": {o["model"]: o["attachment"] for o in objs if o["attachment"]}}
json.dump({"categories": rows, "objects": objects}, open("/tmp/grasp_exp/category_table.json", "w"), indent=1)
print(len(objects), "objects,", len(cats), "categories;", sum(o["ready"] for o in objects.values()), "ready objects;",
      sum(1 for r in rows.values() if r["n_ready"] >= 9), "categories with >= 9 ready models")
EOF_MARK = None
