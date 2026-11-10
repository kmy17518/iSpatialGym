#!/usr/bin/env python3
"""Resolve exact baseline scales and source USD hashes into campaign contracts.

Unready or hash-mismatched templates remain in the blocked manifest; no fallback
scale or implicit readiness is permitted. These records still require fresh
campaign adaptation and successful demo/replay validation before production.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from contracts import ObjectRef, Template, digest, file_digest, task_for_alias
from inventory import VARIANT_SUFFIXES, PENDING_KEYS
from readiness_catalog import ReadinessCatalog

ROLES = {"target": "target", "targets": "target", "obstacles": "target", "partner": "partner", "support": "support",
         "source": "support", "destination": "support", "board": "support", "distractors": "distractor", "floor": "support"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--b1k", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    catalog = ReadinessCatalog(args.b1k / "bddl3/bddl/generated_data")
    source = json.loads(args.source.read_text())
    project_path = args.out / "project_readiness.json"
    project_ready = json.loads(project_path.read_text())["objects"] if project_path.is_file() else {}
    customs = json.loads((args.repo / "ispatialgym/assets/task_specs/isg_customs.json").read_text())
    hashes, ready, blocked, asset_provenance = {}, [], [], {}
    for record in source["records"]:
        if record["key"].endswith(VARIANT_SUFFIXES):
            continue
        meta = record["metadata"]
        objects, problems = [], []
        for binding in meta["bindings"]:
            name, role = binding["scene_object_name"], binding["role"]
            init = record["object_init_args"].get(name)
            if init is None or not isinstance(init.get("scale"), list) or len(init["scale"]) != 3:
                problems.append(f"missing explicit scale {name}")
                continue
            category, model = init["category"], init["model"]
            dataset = init.get("dataset_name") or binding.get("dataset") or "behavior-1k-assets"
            root = args.repo / "ispatialgym/assets/datasets" / dataset if dataset == "ispatialgym-assets" else args.b1k / "datasets" / dataset
            usd = root / "objects" / category / model / "usd" / (model + (".encrypted.usd" if dataset == "behavior-1k-assets" else ".usd"))
            if not usd.is_file():
                problems.append(f"asset absent {usd}")
                continue
            if str(usd) not in hashes:
                hashes[str(usd)] = (file_digest(usd), hashlib.md5(usd.read_bytes()).hexdigest())
            sha, md5 = hashes[str(usd)]
            expected = init.get("expected_file_hash")
            runtime = source.get("runtime_asset_hashes", {}).get(f"{dataset}/{category}/{model}", {})
            if sha != runtime.get("sha256"):
                problems.append(f"local asset does not match frozen cluster runtime {category}/{model}")
            rs = catalog.status(category, model)
            custom = project_ready.get(f"{category}/{model}")
            if dataset == "ispatialgym-assets" and custom and custom["asset_sha256"] == sha:
                rs = {**rs, "ready": custom["ready"], "readiness_source": custom["readiness_source"],
                      "readiness_details": {"source": "established_project_asset", "registry": str(project_path)}}
            semantic = "wall" if category == "walls" else "floor" if category == "floors" else "board" if role == "board" else "tray" if category == "tray" else "object"
            joint = None
            if "articulation" in record["key"] and role == "target":
                facts = meta.get("template_facts", {})
                joint = facts.get("joint") or facts.get("handle", {}).get("joint")
                if not joint:
                    problems.append(f"missing articulation joint {name}")
            obj = ObjectRef(instance=binding["name"], role=ROLES[role], category=category, model=model, asset_sha256=sha,
                            default_scale=tuple(init["scale"]), joint=joint, semantic_kind=semantic,
                            ready=rs["ready"], readiness_source=rs["readiness_source"])
            objects.append(obj)
            asset_provenance[f"{category}/{model}"] = {"dataset": dataset, "usd": str(usd), "usd_sha256": sha,
                "expected_md5": expected, "actual_md5": md5, "stored_metadata_hash_matches_current": expected == md5,
                "verified_cluster_runtime": runtime, **rs}
            if not rs["ready"]:
                problems.append(f"not ready {category}/{model}: {rs['readiness_details']}")
        scene_setup = {"task_rooms": meta.get("task_rooms") or meta["rooms"], "source_template": str(Path(record["metadata_path"]).parent),
                       "source_template_sha256": meta.get("template_sha256"), "source_bddl_sha256": meta.get("bddl_sha256"),
                       "source_alias": record["row"]["alias"], "campaign_adaptation": "pending", "fresh_validation": "not_started",
                       "runtime_template_id": meta["template_id"], "inst_to_name": meta.get("inst_to_name", {}),
                       "awaiting_owner_update": record["key"] in PENDING_KEYS,
                       "scene_variant": meta.get("scene_variant", {})}
        customs_alias = record["row"]["alias"]
        if customs_alias not in customs:
            for relation in ("inside", "nextto", "touching"):
                old = "inter_object_relation_satisfaction-" + relation + "-"
                if customs_alias.startswith(old):
                    customs_alias = customs_alias.replace(old, old[:-1] + "_small_scale-", 1)
                    break
        definition = {"template_sha256": meta.get("template_sha256"), "customs": customs[customs_alias],
                      "object_init_args": record["object_init_args"], "bddl_sha256": meta.get("bddl_sha256")}
        raw = {"template_id": "standard__" + record["key"], "task": task_for_alias(record["key"]), "class_id": record["key"],
               "split": "eval_standard", "scene": meta["scene"], "room": scene_setup["task_rooms"][0],
               "objects": tuple(objects), "definition_sha256": digest(definition), "scene_setup": scene_setup}
        if problems:
            blocked.append({**raw, "objects": [asdict(o) for o in objects], "blockers": problems})
        else:
            template = Template(**raw)
            ready.append({**asdict(template), "fingerprint": template.fingerprint})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("standard_contract_ready.json", {"templates": ready}), ("standard_contract_blocked.json", {"templates": blocked}),
                       ("asset_provenance.json", {"assets": asset_provenance, "readiness_files_sha256": catalog.provenance})):
        (args.out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ready_baseline_templates": len(ready), "blocked_baseline_templates": len(blocked),
                      "blocked_keys": [r["class_id"] for r in blocked], "fresh_demo_replay_passes": 0}))


if __name__ == "__main__":
    main()
