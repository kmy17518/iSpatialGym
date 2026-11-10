#!/usr/bin/env python3
"""Reserve ready holdout resources globally before train expansion.

Reservations are tentative geometric candidates and hard training exclusions.
They do not certify a template or claim newly planned custom models are ready.
"""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "tmp/campaign_20261110/inventory"
    standards = json.loads((out / "standard_templates.json").read_text())["templates"]
    categories = json.loads((out / "category_inventory.json").read_text())["categories"]
    pools = json.loads((out / "model_holdout_candidates.json").read_text())["roles"]
    proposals = json.loads((out / "category_holdout_proposal.json").read_text())
    index = {(v["category"], v["model"]): v for values in categories.values() for v in values}
    used = {(b["category"], b["model"]) for t in standards for b in t["roles"] if not b["structural_holdout_exempt"]}
    mandatory = [
        {"key": "configuration_matching-articulation_open_small_scale_01", "action": "replace_target", "from": "blender/eyedvd", "to": "blender/cwkvib", "reason": "Use same ready model for standard open/close; reserve other ready model for both model-heldout tasks."},
        {"key": "disembedding_04", "action": "restrict_distractor_models", "to": ["bottle_of_milk/debqen", "carton_of_milk/kklgxk"], "remove": ["bottle_of_soy_milk/mcjlhs"], "reason": "Declared category pool unchanged; reserve soy model globally for model holdout."},
        {"key": "disembedding_06", "action": "restrict_distractor_models", "to": ["bottle_of_coconut_milk/idenxg", "carton_of_soy_milk/orgyvw"], "remove": ["bottle_of_soy_milk/mcjlhs"], "reason": "Owner allows templates to select subset of category pool; repeat selected models as needed for crowding."},
    ]
    # These exclusions are valid only after all mandatory rebindings are built.
    heldin_after = used - {("blender", "eyedvd"), ("bottle_of_soy_milk", "mcjlhs")}
    category_models = {}
    for t in proposals["templates"]:
        for value in t.values():
            if isinstance(value, dict) and "category" in value:
                category_models[(value["category"], value["model"])] = value
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict) and "category" in v:
                        category_models[(v["category"], v["model"])] = v
    category_set = {c for c, _ in category_models} | {"dice"}
    attachment = {
        "configuration_matching-attachment_01": {"target": ("lid", "pwgbsb"), "partner": ("teapot", "jlalfc")},
        "configuration_matching-attachment_02": {"target": ("cap", "iqeyba"), "partner": ("paper_cup", "guobeq")},
        "configuration_matching-attachment_03": {"target": ("cork", "uyceta"), "partner": ("vinegar_bottle", "hbsbwt")},
        "configuration_matching-attachment_04": {"target": ("lid", "ppkjuy"), "partner": ("steamer_basket", "lageli")},
    }
    model_refs, heldout_templates, unresolved = {}, [], []
    # Flat source/destination candidates, KB-ready. Geometry / scaling still needs validation.
    model_trays = {"source": ("tray", "gsxbym"), "destination": ("tray", "mhhoga")}
    category_trays = {"source": ("tray", "hjxczh"), "destination": ("tray", "thkphg")}
    def reserve(pair, pool):
        v = index[pair]
        if not v["ready"] or not v["usd_present"]:
            raise ValueError(f"Cannot reserve unready asset {pair}: {v['readiness_details']}")
        if pair in heldin_after:
            raise ValueError(f"Model leakage {pair}")
        if pool == "model" and pair[0] in category_set and pair[0] not in {"tray"}:
            raise ValueError(f"Category leakage {pair}")
        (model_refs if pool == "model" else category_models)[pair] = v
        return {"category": pair[0], "model": pair[1], "ready": v["ready"], "readiness_source": v["readiness_source"], "geometry_validated": False}
    for pair in category_trays.values():
        reserve(pair, "category")
    for standard in standards:
        key = standard["key"]
        if key == "camera_relocalization-standard":
            continue
        assignments = []
        candidates = [p for p in pools if p["key"] == key]
        for p in candidates:
            role, category = p["role"], p["category"]
            selected = None
            if key in attachment and role in attachment[key]:
                selected = attachment[key][role]
            elif category == "blender" and role == "target":
                selected = ("blender", "eyedvd")
            elif role in model_trays:
                selected = model_trays[role]
            elif key in {"disembedding_04", "disembedding_06"} and role == "distractors":
                selected = ("bottle_of_soy_milk", "mcjlhs")
            elif role == "board" or (key == "configuration_matching-block_design_01" and role == "targets") or key == "disembedding_01" and role in {"target", "distractors"}:
                assignments.append({"role": role, "source_category": category, "status": "custom_asset_build_required"})
                continue
            elif role == "support" and category == "booth":
                selected = ("bench", "sntjne")
            else:
                possible = [(v["category"], v["model"]) for v in p["candidates"]
                            if (v["category"], v["model"]) not in heldin_after and v["category"] not in category_set]
                if possible:
                    selected = possible[0]
            if selected is None:
                unresolved.append({"key": key, "role": role, "category": category})
                assignments.append({"role": role, "source_category": category, "status": "unresolved"})
            else:
                assignments.append({"role": role, "source_category": category, **reserve(selected, "model")})
        # Declared distractor categories are a pool; repeated selected model entries are collapsed.
        unique = {json.dumps(a, sort_keys=True): a for a in assignments}
        heldout_templates.append({"class_id": key, "task": standard["task"], "status": "reserved_candidates_not_validated", "roles": list(unique.values())})
    data = {"status": "reservations_require_standard_rebindings_and_geometry_validation", "mandatory_standard_rebindings": mandatory,
            "model_holdout_templates": heldout_templates, "model_holdout_models": [f"{c}/{m}" for c,m in sorted(model_refs)],
            "category_holdout_models": [f"{c}/{m}" for c,m in sorted(category_models)], "category_holdout_categories": sorted(category_set),
            "category_holdout_rooms": [{"scene": "office_cubicles_right", "room": "conference_hall_0"}],
            "train_exclude_models": [f"{c}/{m}" for c,m in sorted(model_refs.keys() | category_models.keys())],
            "train_exclude_categories": sorted(category_set), "structural_model_category_exempt": ["walls", "floors"],
            "category_only_exempt_semantic_kinds": ["tray", "board"],
            "custom_development_queue_not_ready_candidates": [
                {"category": "toy_dice", "model": "isgk10", "split": "eval_model", "description": "Unseen Kohs face permutation;9instances"},
                {"category": "dice", "model": "isgwch", "split": "eval_category", "description": "Wechsler block;1instance"},
                {"category": "tray", "model": "isgbdm", "split": "eval_model", "description": "New compatible model-heldout board"},
                {"category": "tray", "model": "isgbdw", "split": "eval_category", "description": "New compatible Wechsler board"},
                {"category": "jigsaw_puzzle_piece", "model": "pending_Leuven_variant", "split": "eval_model", "description": "New target/context fixture; all component models held out"}],
            "unresolved": unresolved, "readiness_evidence": {f"{c}/{m}": v["readiness_source"] for (c,m),v in {**model_refs,**category_models}.items()}}
    (out / "split_reservations.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_holdout_classes": len(heldout_templates), "reserved_model_holdout_models":len(model_refs),
                      "category_holdout_models":len(category_models), "category_holdout_categories":len(category_set), "unresolved":unresolved}))


if __name__ == "__main__":
    main()
