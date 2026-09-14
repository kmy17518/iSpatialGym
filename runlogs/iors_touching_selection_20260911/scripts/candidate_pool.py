#!/usr/bin/env python3
"""Candidate pool for the IORS-4 (touching) target and partner categories, from the knowledge-base category table
(``runlogs/grasp_exp_20260904/category_table.json``: ready flag, canonical bbox, meta links, taxonomy props per model).

Geometric pre-filters (the semantic curation is done by hand on this list):

* common: taxonomy ``rigidBody``, not ``sceneObject`` / ``cloth`` / ``softBody`` / ``deformable`` / ``rope`` / substances,
  not ``openable`` (articulated -> configuration matching), model has one link (no joints), on disk;
* target (grasped or pushed toward the partner, same rules as IORS nextto): pinchable in the rest pose - smallest
  horizontal extent <= 8.5 cm (gripper opening 10 cm) and >= 2 cm -, long side <= 30 cm, height <= 30 cm, height /
  smallest horizontal extent <= 3 (does not topple when pushed at its near face);
* partner (not grasped; must not move when touched): footprint fits every scene support validated for IORS by the
  table-selection run plus the test table (short side <= min support depth - 6 cm, long side <= min support length / 2,
  the sampler's margins, same rule as select/partners/select_partners.py), height <= 35 cm, height / smallest
  horizontal extent <= 2.5 (stands when bumped).

Output: ``candidate_pool.json`` (per category: counts, median bbox, per-model bbox, flags) and a table on stdout.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
CT = os.path.join(REPO, "runlogs", "grasp_exp_20260904", "category_table.json")
TABLE_RUN = os.path.join(REPO, "runlogs", "table_selection_20260911")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
USED = os.path.join(ROOT, "used_categories.json")
OUT = os.path.join(ROOT, "candidate_pool.json")

EXCLUDE_PROPS = {"sceneObject", "cloth", "softBody", "deformable", "rope", "physicalSubstance", "macroPhysicalSubstance",
                 "microPhysicalSubstance", "visualSubstance", "liquid", "openable"}

T_PINCH_MAX, T_PINCH_MIN, T_LONG_MAX, T_H_MAX, T_ASPECT = 0.085, 0.02, 0.30, 0.30, 3.0
P_H_MAX, P_ASPECT = 0.35, 2.5


def validated_supports():
    """scene supports with >= 1 IORS instance in the table-selection run + the spec-pinned test tables."""
    inv = {}
    with open(os.path.join(TABLE_RUN, "inventory.csv"), newline="") as f:
        for r in csv.DictReader(f):
            inv[(r["scene"], r["name"])] = r
    sup = {}
    for p in sorted(glob.glob(os.path.join(TABLE_RUN, "results", "*__inter_object_relation_satisfaction-*.json"))):
        d = json.load(open(p))
        if d.get("status") != "ok" or d.get("partner"):
            continue
        sup.setdefault((d["scene"], d["table"]), set()).add(d["problem"].rsplit("-", 1)[1])
    spec = json.load(open(SPEC))
    for prob, body in spec.items():
        if not prob.startswith("inter_object_relation_satisfaction"):
            continue
        for scene, cfg in (body.get("scenes") or {}).items():
            t = (cfg.get("fixed") or {}).get("table")
            if isinstance(t, list):
                for n in t:
                    sup.setdefault((scene, n), set()).add("spec")
    out = {}
    for (scene, name), probs in sorted(sup.items()):
        r = inv.get((scene, name))
        if r is None:
            continue
        bx, by = float(r["world_ext_x"]), float(r["world_ext_y"])
        out[f"{scene}:{name}"] = {"scene": scene, "name": name, "depth": round(min(bx, by), 3), "length": round(max(bx, by), 3),
                                  "top_z": float(r["top_z"]), "problems": sorted(probs)}
    return out


def main():
    ct = json.load(open(CT))
    cats, objs = ct["categories"], ct["objects"]
    used = json.load(open(USED)) if os.path.exists(USED) else {}
    supports = validated_supports()
    min_depth = min(s["depth"] for s in supports.values())
    min_length = min(s["length"] for s in supports.values())
    p_short_max, p_long_max = round(min_depth - 0.06, 3), round(min_length / 2, 3)
    print(f"{len(supports)} validated IORS supports; tightest depth {min_depth:.2f} m, shortest length {min_length:.2f} m -> "
          f"partner short side <= {p_short_max:.3f} m, long side <= {p_long_max:.3f} m")

    by_cat = defaultdict(list)
    for o in objs.values():
        if o["on_disk"]:
            by_cat[o["category"]].append(o)
    pool = {}
    for cat, cinfo in cats.items():
        props = set(cinfo["props"])
        if "rigidBody" not in props or props & EXCLUDE_PROPS:
            continue
        models = by_cat.get(cat, [])
        ready = [o for o in models if o["ready"] and o["n_links"] == 1]
        if not ready:
            continue
        recs = []
        n_t = n_p = 0
        for o in ready:
            x, y, z = o["bbox"]
            w, L = sorted((x, y))
            t_ok = (T_PINCH_MIN <= w <= T_PINCH_MAX) and L <= T_LONG_MAX and z <= T_H_MAX and z / max(w, 1e-6) <= T_ASPECT
            p_ok = w <= p_short_max and L <= p_long_max and z <= P_H_MAX and z / max(w, 1e-6) <= P_ASPECT
            n_t += t_ok
            n_p += p_ok
            recs.append({"model": o["model"], "bbox": [round(v, 4) for v in o["bbox"]], "target_ok": t_ok, "partner_ok": p_ok,
                         "meta_links": o["meta_links"], "scenes": o["scenes"]})
        med = [round(statistics.median(o["bbox"][i] for o in ready), 3) for i in range(3)]
        uses = used.get(cat, [])
        rounds = sorted({u["round"] for u in uses})
        pool[cat] = {"synset": cinfo["synset"], "props": sorted(props), "n_models": len(models), "n_ready": len(ready),
                     "n_not_ready": len(models) - len(ready), "median_bbox_ready": med, "n_target_ok": n_t, "n_partner_ok": n_p,
                     "used_rounds": rounds, "used_by": [f"{u['task']}({u['role']},r{u['round']})" for u in uses],
                     "has_fillable": any("fillable" in o["meta_links"] or "openfillable" in o["meta_links"] for o in ready),
                     "models": sorted(recs, key=lambda r: r["model"])}
    json.dump({"supports": supports, "limits": {"target": {"pinch_max": T_PINCH_MAX, "pinch_min": T_PINCH_MIN, "long_max": T_LONG_MAX,
                                                            "height_max": T_H_MAX, "aspect_max": T_ASPECT},
                                                 "partner": {"short_max": p_short_max, "long_max": p_long_max, "height_max": P_H_MAX,
                                                             "aspect_max": P_ASPECT}},
               "categories": pool}, open(OUT, "w"), indent=1)
    rows = sorted(pool.items(), key=lambda kv: (-kv[1]["n_ready"], kv[0]))
    print(f"{len(pool)} rigid non-scene categories with >= 1 ready single-link model -> {OUT}")
    print(f"{'category':32s} {'ready':>5s} {'T_ok':>4s} {'P_ok':>4s} {'median bbox (cm)':>20s}  rounds  props")
    for cat, d in rows:
        if d["n_target_ok"] == 0 and d["n_partner_ok"] == 0:
            continue
        bb = " x ".join(f"{v * 100:5.1f}" for v in d["median_bbox_ready"])
        pr = ",".join(p for p in d["props"] if p not in ("rigidBody",))
        print(f"{cat:32s} {d['n_ready']:5d} {d['n_target_ok']:4d} {d['n_partner_ok']:4d} {bb:>20s}  {str(d['used_rounds']):7s} {pr}"
              + ("  FILL" if d["has_fillable"] else ""))


if __name__ == "__main__":
    main()
