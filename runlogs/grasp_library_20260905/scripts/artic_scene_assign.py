"""Assign every scene-object articulation model under test to a host instance in one of the 7 challenge scenes.
Prefers a scene that already contains the same model (no swap); otherwise the same-category instance whose AABB aspect is closest
(the test model is then swapped in and scaled to the host's AABB). Output: artic/scene_assign.json"""
import json, glob, os, collections, numpy as np
ASSETS = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets/behavior-1k-assets"
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "restaurant_diner", "office_cubicles_right", "hotel_suite_large"]
SCENE_CATS = {"bottom_cabinet", "top_cabinet", "bottom_cabinet_no_top", "fridge", "dishwasher", "oven", "clothes_dryer", "door", "openable_window", "sliding_door"}
inv = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob("/tmp/glib/artic/inv/*.json")}
tests = sorted((v["category"], v["model"]) for v in inv.values() if v["category"] in SCENE_CATS)
# host instances per scene
hosts = []
for s in SCENES:
    d = json.load(open(f"{ASSETS}/scenes/{s}/json/{s}_best.json")); objs = d["objects_info"]["init_info"]; pos = d["state"]["pos"] if "pos" in d["state"] else {}
    reg = d["state"].get("registry", {}).get("object_registry", {})
    for name, o in objs.items():
        a = o["args"]
        if a.get("category") not in SCENE_CATS: continue
        st = reg.get(name, {}); root = (st.get("root_link") or {})
        p = root.get("pos") or st.get("pos"); q = root.get("ori") or st.get("ori")
        mdp = f"{ASSETS}/objects/{a['category']}/{a['model']}/misc/metadata.json"; md = json.load(open(mdp)) if os.path.exists(mdp) else {}
        bbox = np.array(md.get("bbox_size", [0, 0, 0])) * np.array(a.get("scale", [1, 1, 1]))
        hosts.append({"scene": s, "name": name, "category": a["category"], "model": a["model"], "scale": a.get("scale", [1, 1, 1]), "aabb_extent": bbox.round(4).tolist(), "pos": p, "ori": q, "rooms": a.get("in_rooms")})
print(len(hosts), "host instances in the 7 scenes; per scene:", dict(collections.Counter(h["scene"] for h in hosts)))
assign = []
for cat, model in tests:
    ext_t = np.array(inv[f"{cat}__{model}"]["aabb_extent"])
    same = [h for h in hosts if h["category"] == cat and h["model"] == model]
    if same:
        h = same[0]; alts = [x["name"] for x in hosts if x["category"] == cat and x["scene"] == h["scene"] and x["name"] != h["name"]][:2]
        assign.append({"category": cat, "model": model, "scene": h["scene"], "host": h["name"], "host_model": h["model"], "swap": False, "host_aabb": h["aabb_extent"], "scale_factors": [round(float(x), 3) for x in np.array(h["aabb_extent"]) / np.maximum(ext_t, 1e-6)], "host_pos": h["pos"], "host_ori": h["ori"], "host_alternatives": alts})
        continue
    cands = [h for h in hosts if h["category"] == cat]
    if not cands: assign.append({"category": cat, "model": model, "scene": None, "host": None, "swap": None, "note": "no instance of this category in the 7 scenes"}); continue
    def aspect_dist(h):
        e = np.array(h["aabb_extent"]); f = e / np.maximum(ext_t, 1e-6); f = f / f.mean(); return float(np.abs(np.log(f)).sum())
    h = min(cands, key=aspect_dist)
    alts = [x["name"] for x in sorted([c for c in cands if c["scene"] == h["scene"] and c["name"] != h["name"]], key=aspect_dist)[:2]]
    assign.append({"category": cat, "model": model, "scene": h["scene"], "host": h["name"], "host_model": h["model"], "swap": True, "host_aabb": h["aabb_extent"], "scale_factors": [round(float(x), 3) for x in np.array(h["aabb_extent"]) / np.maximum(ext_t, 1e-6)], "host_pos": h["pos"], "host_ori": h["ori"], "aspect_dist": round(aspect_dist(h), 3), "host_alternatives": alts})
json.dump(assign, open("/tmp/glib/artic/scene_assign.json", "w"), indent=1)
print(len(assign), "tests:", dict(collections.Counter(a["scene"] for a in assign)), "| swaps:", sum(1 for a in assign if a["swap"]), "| no host:", sum(1 for a in assign if a["scene"] is None))
for a in assign[:6]: print("  ", a["category"], a["model"], "->", a["scene"], a["host"], "swap" if a["swap"] else "same", a.get("scale_factors"))
print("pose available for", sum(1 for a in assign if a.get("host_pos")), "of", len(assign))
