"""Scaled variants for objects that slipped under assisted grasping because they are thinner than ~1 cm.

Rule (min thickness 2 cm):
  * compact objects (longest / thinnest extent <= 3): uniform scale so the thinnest extent becomes 2 cm (garlic clove ->
    a large clove; the shape is preserved);
  * elongated objects: scale only the two cross-section axes so each is >= 2 cm and keep the length (a pencil becomes a
    chunky 2 cm marker instead of a 50 cm giant pencil; fries become steak fries);
  * flat discs (caps): scale only the thickness axis to 2 cm.
Scales are axis-aligned in the object's canonical frame (OmniGibson DatasetObject accepts a per-axis scale)."""
import json, collections, numpy as np
OUT = "/tmp/glib"; MIN_T = 0.02
jobs = json.load(open(f"{OUT}/models.json"))
results = {j["key"]: json.load(open(f"{OUT}/results/{j['key']}.json")) for j in jobs}
TARGET_CATS = {"pencil", "french_fries", "garlic_clove", "cinnamon_stick", "cap"}
scaled = []
for j in jobs:
    if j["category"] not in TARGET_CATS: continue
    r = results[j["key"]]
    if r["n_valid"] >= 5 and j["category"] == "cap": continue                 # the caps that already hold are left alone
    b = np.array(j["bbox"], float); order = np.argsort(b); a, m, x = b[order]
    s = np.ones(3)
    if j["category"] == "cap":                                                 # flat disc: thickness only
        s[order[0]] = max(1.0, MIN_T / a); rule = "thickness_only"
    elif x / a <= 3.0:                                                         # compact: uniform
        s[:] = max(1.0, MIN_T / a); rule = "uniform"
    else:                                                                      # elongated: cross-section only
        s[order[0]] = max(1.0, MIN_T / a); s[order[1]] = max(1.0, MIN_T / m); rule = "cross_section"
    new = dict(j); new["base_key"] = j["key"]; new["key"] = f"{j['key']}__s"; new["scale"] = [round(float(v), 3) for v in s]; new["scale_rule"] = rule
    new["bbox_scaled"] = [round(float(v), 4) for v in b * s]; new["orig_n_valid"] = r["n_valid"]
    scaled.append(new)
json.dump(scaled, open(f"{OUT}/models_scaled.json", "w"), indent=1)
by = collections.defaultdict(list)
for s in scaled: by[s["category"]].append(s)
for c, ss in by.items():
    print(f"{c:16s} n={len(ss):2d} rule={ss[0]['scale_rule']:14s} example bbox {np.round(np.array(ss[0]['bbox'])*1000).astype(int).tolist()} mm x scale {ss[0]['scale']} -> {np.round(np.array(ss[0]['bbox_scaled'])*1000).astype(int).tolist()} mm")
print(len(scaled), "scaled jobs")
