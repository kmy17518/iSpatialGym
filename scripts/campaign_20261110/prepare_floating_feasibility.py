"""Materialize bounded, zero-credit native room/stage queries for exact held targets.

This author never submits work or relaxes physical prerequisites. Each query uses
one explicitly validated arm, the selected scale and room, and the canonical
task's unchanged BDDL and generation constraints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def main(args):
    root, out, runtime = map(Path, (args.root, args.output, args.runtime))
    sys.path[:0] = [str(runtime / "Behavior-1K/bddl3"), str(runtime)]
    from scripts.campaign_20261110.reservation_policy import active_policy
    from ispatialgym.sampling.knowledge import is_ready
    from ispatialgym.sampling.v2.problem import GroundedProblem

    cache = {}
    def sha(path):
        key = str(path)
        if key not in cache:
            cache[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return cache[key]
    def ref(path):
        return {"path": str(path), "sha256": sha(path)}
    def read(path):
        return json.loads(Path(path).read_text())
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
    def key(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert not out.exists(), "Immutable preparation output already exists"
    out.mkdir(parents=True)
    inventory = root / "training_preparation/model_variant_inventory_r2"
    scales_path = inventory / "target_profile_rekey_r2/scale_expansion.json"
    graph_path = inventory / "broad_map_inputs_r1/selected_manifest.json"
    graph = read(graph_path)
    deps = {r["id"]: r for r in graph["artifact_requests"]}
    canonical_path = root / "infra/standard_recovery/canonical_validation.json"
    canonical_bytes = canonical_path.read_bytes()
    canonical_origin = {"path": str(canonical_path), "sha256": hashlib.sha256(canonical_bytes).hexdigest()}
    canonical_path = out / "sources/canonical_validation.json"
    canonical_path.parent.mkdir(mode=0o700)
    canonical_path.write_bytes(canonical_bytes)
    canonical = {r["class_id"]: r for r in json.loads(canonical_bytes)["classes"]}
    previous_keys = set()
    for path in args.existing_requests:
        previous_keys.update(j["physical_dedup_key"] for j in read(path)["jobs"])
    pointer, policy, assets = active_policy(root)
    excluded_rooms = {(r["scene"], r["room"]) for r in policy["category_holdout_rooms"]}
    old = {}
    for p in (root / "training_pipeline_canaries").glob("*/selected_slot.json"):
        d = read(p)
        cid = d.get("configuration", {}).get("id")
        if not cid:
            continue
        prep = p.parent / "preparation.json"
        if prep.exists():
            for arm in read(prep).get("offered_arms", []):
                old[(cid, arm)] = ref(p)
    scene_cache, releases, jobs, blocked, existing = {}, {}, {}, [], []
    original_configs = read(scales_path)["rows"]
    for config in original_configs:
        if not config["class_id"].startswith("object_scaling-") or config.get("is_canonical_eval_template"):
            continue
        cid = config["configuration_id"]
        if not config["exact_scales_resolved"]:
            blocked.append({"configuration_id": cid, "reason": "Exact scale unresolved"})
            continue
        try:
            required = [deps[i] for i in config["dependencies"]]
            assert len(required) == 4 and {x["kind"] for x in required} == {"floating_grasp_library", "floating_native_motion_validation", "native_asset_default_geometry"}
            assert len(config["rooms"]) == 1 and len(config["roles"]) == 1
            source = canonical[config["class_id"]]
            assert source["validated"] is True
            contract = source["configuration"]
            source_dir = Path(contract["scene_setup"]["source_template"])
            metadata_path = source_dir / "template_meta.json"
            meta = read(metadata_path)
            customs_path = Path(meta["customs"])
            alias = meta.get("alias", meta.get("variant"))
            assert alias == config["class_id"]
            original = read(customs_path)[alias]
            matches = [p for p in (source_dir / "activity_definitions").rglob("*.bddl") if sha(p) == meta["bddl_sha256"]]
            assert len(matches) == 1
            scene, room = config["scene"], config["rooms"][0]
            assert (scene, room) not in excluded_rooms
            scene_path = runtime / "Behavior-1K/datasets/2026-challenge-task-instances/scenes" / scene / "json" / (scene + "_stable.json")
            if scene not in scene_cache:
                scene_cache[scene] = read(scene_path)["objects_info"]["init_info"]
            floors = [(name, value["args"]) for name, value in scene_cache[scene].items() if value["args"].get("category") == "floors" and room in (value["args"].get("in_rooms") or [])]
            assert floors
            floor_name, floor = floors[0]
            assert is_ready(floor["model"]) and len(floor["scale"]) == 3
            floor_asset = runtime / "Behavior-1K/datasets/behavior-1k-assets/objects/floors" / floor["model"] / "usd" / (floor["model"] + ".encrypted.usd")
            target = next(x["inputs"]["target"] for x in required if x["kind"] == "floating_grasp_library")
            assert target["role"] == "target" and target["ready"] and target["readiness_source"] and is_ready(target["model"])
            assert target["model_key"] not in policy["train_exclude_models"] and target["category"] not in policy["train_exclude_categories"] and target["asset_sha256"] not in assets["reserved_asset_hashes"]
            target_asset = runtime / "Behavior-1K/datasets" / target["dataset"] / "objects" / target["category"] / target["model"] / "usd" / (target["model"] + ".encrypted.usd")
            assert sha(target_asset) == target["asset_sha256"]
            assert config["roles"][0]["applied_scale"] == target["applied_scale"]
            ratios = [a / b for a, b in zip(target["applied_scale"], target["original_default_scale"])]
            assert all(.8 - 1e-4 <= x <= 1.2 + 1e-4 for x in ratios) and max(ratios) - min(ratios) < 2e-4
        except Exception as exc:
            blocked.append({"configuration_id": cid, "reason": repr(exc), "stage": "source_materialization"})
            continue
        for arm in ("left", "right") if args.arm == "both" else (args.arm,):
            if (cid, arm) in old:
                existing.append({"configuration_id": cid, "arm": arm, "owner": old[(cid, arm)]})
                continue
            try:
                motion = next(x for x in required if x["kind"] == "floating_native_motion_validation" and x["inputs"]["arm"] == arm)
                assert motion["status"] == "verified", "Exact arm motion is not verified"
                library_ref = motion["artifact"]
                lp = Path(library_ref["path"])
                assert sha(lp) == library_ref["sha256"]
                lib = read(lp)
                assert lib["arm"] == arm and lib["constraint"] == "floating" and lib["scale"] == target["applied_scale"] and lib["grasps"]
                assert all(lib[k] == target[k] for k in ("category", "model", "dataset"))
                for evidence in motion["evidence"]:
                    assert sha(evidence["path"]) == evidence["sha256"]
                release_key = key({"library": library_ref["sha256"], "arm": arm})
                if release_key not in releases:
                    release = out / "dependencies" / release_key
                    dest = release / "glib/models" / lp.parent.name
                    shutil.copytree(lp.parent, dest)
                    routed = read(dest / "library.json")
                    routed["legacy_key"] = target["category"] + "__" + target["model"]
                    assert {k: v for k, v in routed.items() if k != "legacy_key"} == {k: v for k, v in lib.items() if k != "legacy_key"}
                    write(dest / "library.json", routed)
                    (release / "maps").mkdir()
                    write(release / "anchors.json", {})
                    proof = {"source": library_ref, "routed": ref(dest / "library.json"), "alias_only": True, "arm": arm, "native_motion_dependency": motion, "files": [ref(p) for p in release.rglob("*") if p.is_file()], "benchmark_credit": 0}
                    write(release / "release.json", proof)
                    releases[release_key] = release
                release = releases[release_key]
                entry = deepcopy(original)
                declared = entry["scenes"]
                # Some canonical instances were created through the native room
                # override from a declaration containing one floor-only scene.
                # Preserve that declaration's constraints, then bind the exact
                # selected floor/room using the same established override rule.
                source_scene = contract["scene"]
                if source_scene not in declared:
                    assert len(declared) == 1, "Ambiguous canonical scene declaration"
                    source_scene = next(iter(declared))
                block = deepcopy(declared[source_scene])
                assert set(block.get("bindings", {})) == {"floor"}
                block.update(rooms=[room], bindings={"floor": {"room": room}}, blacklist={}, whitelist={"target": {target["category"]: [target["model"]]}})
                entry["scenes"] = {scene: block}
                entry["roles"]["target"].setdefault("scale", {})[target["model_key"]] = target["applied_scale"]
                entry["roles"]["target"]["representative"] = target["model"]
                entry["generation"].update(templates_max=1, instances_per_template=1, arm=arm, other_arm=False)
                for name in list(entry):
                    if name.startswith("fallback") or name == "superseded_by":
                        entry.pop(name)
                physical = {"task": config["class_id"], "scene": scene, "room": room, "target": target["model_key"], "scale": target["applied_scale"], "arm": arm, "library_sha256": library_ref["sha256"], "customs": entry, "bddl_sha256": meta["bddl_sha256"], "runtime_sha256": sha(runtime / ".snapshot_commit")}
                physical_key = key(physical)
                if physical_key in previous_keys:
                    existing.append({"configuration_id": cid, "arm": arm, "physical_dedup_key": physical_key,
                                     "owner": "Previously materialized immutable feasibility request"})
                    continue
                if physical_key in jobs:
                    jobs[physical_key]["configuration_ids"].append(cid)
                    continue
                job_id = "floating_query_" + physical_key[:20]
                directory = out / "queries" / job_id
                directory.mkdir(parents=True)
                custom = directory / "customs.json"
                write(custom, {alias: entry})
                bddl = directory / "bddl" / matches[0].parent.name / matches[0].name
                bddl.parent.mkdir(parents=True)
                bddl.write_bytes(matches[0].read_bytes())
                grounded = GroundedProblem.load(alias, str(custom), str(directory / "bddl"))
                assert grounded.rooms(scene) == [room]
                env = {"ISPATIALGYM_GRASP_LIBRARY_V2": str(release / "glib"), "ISPATIALGYM_REACH_MAPS": str(release / "maps"), "ISPATIALGYM_REACH_MAP_ANCHORS": str(release / "anchors.json"), "ISPATIALGYM_V2_ONLY": "1"}
                query = {"id": job_id, "physical_dedup_key": physical_key, "configuration_ids": [cid], "class_id": config["class_id"], "scene": scene, "room": room, "arm": arm, "target": target, "expected_native_floor": {"name": floor.get("name", floor_name), "args": floor, "asset": ref(floor_asset)}, "native_runtime": str(runtime), "environment": env, "sample_argv": ["python", "-u", "-m", "ispatialgym.sampling.sample_isg", "--variant", alias, "--customs", str(custom), "--bddl-root", str(directory / "bddl"), "--scene", scene, "--room", room, "--templates", "1", "--instances-per-template", "1", "--max-attempts", "3", "--seed", "2026111214", "--no-write-back", "--out", "${NATIVE_OUTPUT}"], "input_refs": [ref(custom), ref(bddl), ref(metadata_path), ref(customs_path), ref(scene_path), ref(target_asset), ref(floor_asset), ref(release / "release.json")], "dependency_release": str(release), "native_time_seconds": 1800, "hard_time_seconds": 2400, "benchmark_credit": 0, "production_authorized": False, "query_scope": "Exact held-target scene collision/navigation and native task-stage feasibility; one explicitly validated arm. No tabletop packing requirement.", "followup": "Verify fresh native target model/scale, floor binding, scene/room, arm and full task-program result; failures remain owned diagnostics; no automatic replay, acceptance or unchanged retry."}
                write(directory / "request.json", query)
                query["request"] = ref(directory / "request.json")
                jobs[physical_key] = query
            except Exception as exc:
                blocked.append({"configuration_id": cid, "arm": arm, "reason": repr(exc), "stage": "arm_or_native_materialization"})
    report = {"schema": "isg-exact-native-feasibility-requests-v1", "observed_at": time.time(), "campaign_id": root.name, "owner": "/root/templates_assets", "dispatch_owner": "/root", "sources": [ref(scales_path), ref(graph_path), ref(canonical_path), ref(runtime / ".snapshot_commit")], "canonical_origin": canonical_origin, "reservation_policy": pointer, "jobs": list(jobs.values()), "blocked": blocked, "existing_owned": existing, "summary": {"ready_physical_queries": len(jobs), "ready_configuration_arm_links": sum(len(j["configuration_ids"]) for j in jobs.values()), "blocked": len(blocked), "existing_owned": len(existing), "by_class": dict(Counter(j["class_id"] for j in jobs.values()))}, "production_authorized": False, "benchmark_credit": 0}
    write(out / "requests.json", report)
    print(json.dumps({"manifest": ref(out / "requests.json"), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--existing-requests", action="append", default=[])
    main(parser.parse_args())
