"""Materialize exact zero-credit camera room and obstacle feasibility queries."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
import time


def main(args):
    root, out, runtime = map(Path, (args.root, args.output, args.runtime))
    sys.path[:0] = [str(runtime / "Behavior-1K/bddl3"), str(runtime)]
    from scripts.campaign_20261110.reservation_policy import active_policy
    from ispatialgym.sampling.knowledge import is_ready
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices

    cache = {}
    def sha(p):
        p = str(p)
        if p not in cache:
            cache[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return cache[p]
    def ref(p):
        return {"path": str(p), "sha256": sha(p)}
    def read(p):
        return json.loads(Path(p).read_text())
    def write(p, d):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2) + "\n")
    assert not out.exists()
    out.mkdir(parents=True)
    inventory = root / "training_preparation/model_variant_inventory_r2"
    scale_path = inventory / "target_profile_rekey_r2/scale_expansion.json"
    canonical_path = root / "infra/standard_recovery/canonical_validation.json"
    canonical_origin = ref(canonical_path)
    canonical_copy = out / "sources/canonical_validation.json"
    canonical_copy.parent.mkdir()
    canonical_copy.write_bytes(canonical_path.read_bytes())
    assert sha(canonical_copy) == canonical_origin["sha256"]
    canonical = {r["class_id"]: r for r in read(canonical_copy)["classes"]}
    pointer, policy, assets = active_policy(root)
    reserved_rooms = {(r["scene"], r["room"]) for r in policy["category_holdout_rooms"]}
    scenes, jobs, blocked = {}, {}, []
    for c in read(scale_path)["rows"]:
        if not c["class_id"].startswith("camera_relocalization-") or c.get("is_canonical_eval_template"):
            continue
        try:
            assert c["exact_scales_resolved"] and len(c["rooms"]) == 1
            source = canonical[c["class_id"]]
            assert source["validated"]
            contract = source["configuration"]
            source_dir = Path(contract["scene_setup"]["source_template"])
            metadata = source_dir / "template_meta.json"
            meta = read(metadata)
            customs = Path(meta["customs"])
            alias = meta.get("alias", meta.get("variant"))
            assert alias == c["class_id"]
            entry = deepcopy(read(customs)[alias])
            obstructed = "obstacles" in entry["roles"]
            assert len(c["roles"]) == int(obstructed)
            scene, room = c["scene"], c["rooms"][0]
            assert (scene, room) not in reserved_rooms
            scene_file = runtime / "Behavior-1K/datasets/2026-challenge-task-instances/scenes" / scene / "json" / (scene + "_stable.json")
            if scene not in scenes:
                scenes[scene] = read(scene_file)["objects_info"]["init_info"]
            floors = [(n, v["args"]) for n, v in scenes[scene].items() if v["args"].get("category") == "floors" and room in (v["args"].get("in_rooms") or [])]
            assert floors
            floor_name, floor = floors[0]
            floor_asset = runtime / "Behavior-1K/datasets/behavior-1k-assets/objects/floors" / floor["model"] / "usd" / (floor["model"] + ".encrypted.usd")
            assert is_ready(floor["model"])
            source_scene = contract["scene"]
            if source_scene not in entry["scenes"]:
                assert len(entry["scenes"]) == 1
                source_scene = next(iter(entry["scenes"]))
            block = deepcopy(entry["scenes"][source_scene])
            assert set(block["bindings"]) == {"floor"}
            block.update(rooms=[room], bindings={"floor": {"room": room}}, blacklist={}, whitelist={})
            target, target_asset = None, None
            if obstructed:
                target = deepcopy(c["roles"][0])
                category, model = target["model_key"].split("/")
                assert target["role"] == "target" and is_ready(model)
                assert target["model_key"] not in policy["train_exclude_models"] and category not in policy["train_exclude_categories"] and target["asset_sha256"] not in assets["reserved_asset_hashes"]
                dataset = entry["roles"]["obstacles"].get("dataset", "behavior-1k-assets")
                target_asset = runtime / "Behavior-1K/datasets" / dataset / "objects" / category / model / "usd" / (model + ".encrypted.usd")
                assert sha(target_asset) == target["asset_sha256"]
                assert entry["roles"]["obstacles"]["category"] == category
                assert entry["generation"]["groups"]["obstacles"]["size"] == 1
                block["whitelist"] = {"obstacles": {category: [model]}}
                entry["roles"]["obstacles"].update(representative=model, scale={target["model_key"]: target["applied_scale"]})
                adaptation = contract["scene_setup"]["runtime_bddl_adaptation"]
                bddl_source = Path(adaptation["authored_bddl_path"])
                assert sha(bddl_source) == adaptation["runtime_bddl_sha256"]
            else:
                matches = [p for base in (source_dir / "activity_definitions", source_dir.parent / "activity_definitions") for p in base.rglob("*.bddl") if sha(p) == meta["bddl_sha256"]]
                assert len(set(matches)) == 1
                bddl_source = matches[0]
            entry["scenes"] = {scene: block}
            entry["generation"].update(templates_max=1, instances_per_template=1)
            physical = {"class_id": c["class_id"], "scene": scene, "room": room, "target": target, "customs": entry, "bddl_sha256": sha(bddl_source), "runtime_sha256": sha(runtime / ".snapshot_commit"), "stationary_obstacle_goal": obstructed}
            key = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if key in jobs:
                jobs[key]["configuration_ids"].append(c["configuration_id"])
                continue
            job_id = "camera_query_" + key[:20]
            directory = out / "queries" / job_id
            custom = directory / "customs.json"
            write(custom, {alias: entry})
            bddl = directory / "bddl/camera_relocalization" / ("problem" + str(entry["definition_id"]) + ".bddl")
            bddl.parent.mkdir(parents=True)
            bddl.write_bytes(bddl_source.read_bytes())
            grounded = GroundedProblem.load(alias, str(custom), str(directory / "bddl"))
            choices = list(iter_template_choices(grounded, scene, np.random.default_rng(2026111214), max_templates=1))
            assert len(choices) == 1 and choices[0].rooms == [room]
            if obstructed:
                picked = choices[0].groups["obstacles"]
                assert len(picked) == 1 and picked[0].category == category and picked[0].model == model and list(picked[0].scale) == target["applied_scale"]
            inputs = [ref(p) for p in (custom, bddl, metadata, customs, scene_file, floor_asset, canonical_copy)]
            if target_asset:
                inputs.append(ref(target_asset))
            job = {"id": job_id, "physical_dedup_key": key, "configuration_ids": [c["configuration_id"]], "class_id": c["class_id"], "scene": scene, "room": room, "target": target, "expected_native_floor": {"name": floor_name, "args": floor, "asset": ref(floor_asset)}, "native_runtime": str(runtime), "environment": {"ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL": "1"}, "sample_argv": ["python", "-u", "-m", "ispatialgym.sampling.sample_isg", "--variant", alias, "--customs", str(custom), "--bddl-root", str(directory / "bddl"), "--scene", scene, "--room", room, "--templates", "1", "--instances-per-template", "1", "--max-attempts", "3", "--seed", "2026111214", "--no-write-back", "--out", "${NATIVE_OUTPUT}"], "input_refs": inputs, "native_time_seconds": 1800, "hard_time_seconds": 2400, "benchmark_credit": 0, "production_authorized": False, "query_scope": "Native obstacle geometry admissibility at the exact selected scale, settled placement, camera visibility, navigation and stationary-obstacle goal. Standard camera omits obstacles. A bounded positive proves this configuration can instantiate; no full room/path coverage claim.", "followup": "Verify native bindings, exact object scale, floor, room, stationary obstacle goal/camera-only measure, fresh path/visibility and native full-program evidence. Continue failed physical configurations as owned diagnosis without unchanged retries."}
            request = directory / "request.json"
            write(request, job)
            job["request"] = ref(request)
            jobs[key] = job
        except Exception as exc:
            blocked.append({"configuration_id": c["configuration_id"], "class_id": c["class_id"], "reason": repr(exc)})
    report = {"schema": "isg-exact-native-feasibility-requests-v1", "observed_at": time.time(), "campaign_id": root.name, "owner": "/root/templates_assets", "dispatch_owner": "/root", "sources": [ref(scale_path), ref(canonical_copy), ref(runtime / ".snapshot_commit")], "canonical_origin": canonical_origin, "reservation_policy": pointer, "jobs": list(jobs.values()), "blocked": blocked, "summary": {"ready_physical_queries": len(jobs), "blocked": len(blocked), "by_class": dict(Counter(j["class_id"] for j in jobs.values()))}, "benchmark_credit": 0, "production_authorized": False}
    write(out / "requests.json", report)
    print(json.dumps({"manifest": ref(out / "requests.json"), "summary": report["summary"], "blocked_examples": blocked[:3]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output", "runtime"):
        p.add_argument("--" + name, required=True)
    main(p.parse_args())
