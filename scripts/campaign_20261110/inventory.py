#!/usr/bin/env python3
"""Build the campaign's auditable candidate inventory without importing simulation.

This is a candidate list, never a claim that a template passed demo/replay gates.
Template bindings override the older representatives table. Structural floor is
excluded; walls remain explicit until the owner decides the structural exception.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from readiness_catalog import ReadinessCatalog

SCENES = ("house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int",
          "hotel_suite_large", "restaurant_diner", "office_cubicles_right")
VARIANT_SUFFIXES = ("_sinkvekaaf", "_sinkcounter080", "_drawn", "_t7otwukr", "_t8otwukr")
PENDING_KEYS = {"inter_object_relation_satisfaction-" + key for key in
                ("touching_03", "under_large_scale_01", "under_large_scale_02", "under_large_scale_03", "nextto_04")}
ROLE_NORMALIZATION = {"targets": "target", "obstacles": "target", "source": "support", "destination": "support", "board": "support"}
SUPPORT_UTILITIES = {
    "table_surface": {"desk", "breakfast_table", "commercial_kitchen_table", "conference_table", "countertop", "console_table", "lab_table"},
    "low_table_surface": {"coffee_table", "garden_coffee_table"},
    "seat_surface": {"bench", "booth", "sofa", "armchair", "footstool", "ottoman"},
    "bed_surface": {"bed", "bunk_bed", "hospital_bed"},
    "wall": {"walls"},
    "tray_surface": {"tray", "baking_sheet", "roasting_pan"},
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def model_record(root, category, model, readiness):
    p = root / "objects" / category / model
    try:
        meta = json.loads((p / "misc/metadata.json").read_text())
    except (OSError, ValueError):
        meta = {}
    return {"category": category, "model": model, "dataset": root.name,
            "usd_present": any((p / "usd").glob("*.usd")), "bbox_m": meta.get("bbox_size"),
            "link_count": len(meta.get("meta_links", {})), "metadata_path": str(p / "misc/metadata.json"),
            **readiness.status(category, model)}


def canonical_task(subtask):
    if subtask in {"inter_object_relation_satisfaction-inside", "inter_object_relation_satisfaction-nextto", "inter_object_relation_satisfaction-touching"}:
        return subtask + "_small_scale"
    return subtask


def build(source, asset_root, representatives, output):
    raw = json.loads(source.read_text())
    readiness = ReadinessCatalog(asset_root.parent.parent / "bddl3/bddl/generated_data")
    records = raw["records"]
    missing = [r["key"] for r in records if "metadata" not in r]
    if missing:
        raise ValueError("Missing template metadata: " + ", ".join(missing))
    rep_rows = list(csv.DictReader(representatives.open()))
    standard, variants = [], []
    for record in records:
        m, row = record["metadata"], record["row"]
        roles = []
        for binding in m.get("bindings", []):
            args = record.get("object_init_args", {}).get(binding["scene_object_name"], {})
            roles.append({**binding, "canonical_role": ROLE_NORMALIZATION.get(binding["role"], binding["role"]),
                          "scale": args.get("scale"), "structural_holdout_exempt": binding["category"] in {"walls", "floors"},
                          **readiness.status(binding["category"], binding["model"])})
        entry = {"key": record["key"], "task": canonical_task(m.get("subtask") or row["subtask"]),
                 "alias": row["alias"], "pick": row.get("pick"), "template_id": m["template_id"], "scene": m["scene"],
                 "rooms": m.get("task_rooms") or m.get("rooms"), "roles": roles, "choices": m.get("choices"),
                 "template_path": str(Path(record["metadata_path"]).parent), "template_sha256": m.get("template_sha256"),
                 "source_metadata": record["metadata_path"], "fresh_validation": "not_started",
                 "status": "awaiting_owner_update" if record["key"] in PENDING_KEYS else "baseline_available"}
        (variants if record["key"].endswith(VARIANT_SUFFIXES) else standard).append(entry)
    if len(standard) != 100 or len({r["task"] for r in standard}) != 23:
        raise ValueError("Expected 100 standard classes across 23 tasks")
    used_models = {(b["category"], b["model"]) for r in standard for b in r["roles"] if not b["structural_holdout_exempt"]}
    used_categories = {c for c, _ in used_models}
    used_rooms = {(r["scene"], room) for r in standard for room in r["rooms"]}
    # Global cross-role exclusion, not a per-category/per-task blacklist.
    by_category = {}
    for category_dir in sorted((asset_root / "objects").iterdir()):
        if category_dir.is_dir():
            by_category[category_dir.name] = [model_record(asset_root, category_dir.name, p.name, readiness)
                                            for p in sorted(category_dir.iterdir()) if p.is_dir()]
    model_candidates = []
    for record in standard:
        seen_roles = set()
        for binding in record["roles"]:
            if binding["structural_holdout_exempt"]:
                continue
            category, model, role = binding["category"], binding["model"], binding["role"]
            if (role, category, model) in seen_roles:
                continue
            seen_roles.add((role, category, model))
            preferred = {r["model"] for r in rep_rows if r["category"] == category and r["slot"] == "heldout"}
            candidates = [{**m, "preferred_by_representatives": m["model"] in preferred}
                          for m in by_category.get(category, []) if (category, m["model"]) not in used_models and m["usd_present"] and m["ready"]]
            candidates.sort(key=lambda c: (not c["preferred_by_representatives"], c["model"]))
            model_candidates.append({"key": record["key"], "role": role, "category": category, "source_model": model,
                                     "candidates": candidates, "status": "candidates_only" if candidates else "no_same_category_candidate"})
    room_inventory = []
    for scene in SCENES:
        scene_file = asset_root / "scenes" / scene / "json" / f"{scene}_best.json"
        data = json.loads(scene_file.read_text())
        rooms = collections.defaultdict(list)
        for name, obj in data["objects_info"]["init_info"].items():
            a = obj.get("args", {})
            if not a.get("category") or a["category"] in {"walls", "floors", "ceilings"}:
                continue
            for room in a.get("in_rooms", []) or []:
                rooms[room].append({"name": name, "category": a["category"], "model": a.get("model"), "scale": a.get("scale"),
                                    "category_unused_in_standard_roles": a["category"] not in used_categories})
        for room, objects in sorted(rooms.items()):
            room_inventory.append({"scene": scene, "room": room, "heldout_eligible_by_task_room": (scene, room) not in used_rooms,
                                   "scene_file": str(scene_file), "objects": objects})
    collisions = [r for r in rep_rows if r["slot"] == "heldout" and (r["category"], r["model"]) in used_models]
    summary = {"source_code": raw["source_code"], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "standard_classes": len(standard), "tasks": dict(sorted(collections.Counter(r["task"] for r in standard).items())),
               "registered_variants": len(variants), "standard_role_models": len(used_models), "standard_role_categories": len(used_categories),
               "standard_task_rooms": len(used_rooms), "eligible_unused_task_rooms": sum(r["heldout_eligible_by_task_room"] for r in room_inventory),
               "readiness_provenance": readiness.provenance,
               "nonready_standard_roles": [{"key":r["key"], **b} for r in standard for b in r["roles"] if not b["ready"]],
               "representatives_holdout_rows_leaking_into_standard": collisions,
               "candidate_roles_without_same_category_model": [{k:v for k,v in r.items() if k!="candidates"} for r in model_candidates if not r["candidates"]],
               "exclusions": ["structural floors and walls exempt from model/category holdouts; trays/boards only category-exempt", "background objects do not count as task roles",
                              "candidates need geometry/semantic review, maps, a fresh demo and open-loop replay", "no old instances copied"]}
    write_json(output / "standard_templates.json", {"templates": standard})
    write_json(output / "registered_variants.json", {"templates": variants})
    write_json(output / "model_holdout_candidates.json", {"roles": model_candidates})
    write_json(output / "room_inventory.json", {"rooms": room_inventory})
    write_json(output / "category_inventory.json", {"categories": by_category, "used_standard_categories": sorted(used_categories)})
    write_json(output / "inventory_summary.json", summary)
    lines = ["# Campaign template inventory", "", f"100 standard classes across 23 tasks; {len(variants)} registered variants.",
             f"{len(used_models)} task-role models, {len(used_categories)} task-role categories, {len(used_rooms)} task rooms are already in standard.",
             f"{summary['eligible_unused_task_rooms']} remaining task rooms are candidates, subject to navigability and visibility checks.", "",
             "No new template is validated yet. Existing instances were not copied.", "", "| Task | Standard classes |", "|---|---:|"]
    lines += [f"| {task} | {count} |" for task, count in summary["tasks"].items()]
    lines += ["", "## Existing held-out row conflicts", "", "These older representative rows cannot be used as model holdouts without changing the globally used models.", ""]
    lines += [f"- {r['pick']} / {r['role']}: {r['category']}/{r['model']}" for r in collisions] or ["None."]
    (output / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"candidate_roles_without_same_category_model", "representatives_holdout_rows_leaking_into_standard"}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.templates, args.assets, args.representatives, args.out)


if __name__ == "__main__":
    main()
