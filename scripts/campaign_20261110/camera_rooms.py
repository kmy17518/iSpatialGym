"""Bind camera room candidates to fresh native template and stable-floor bytes."""
from dataclasses import replace
from pathlib import Path

from scripts.campaign_20261110.contracts import digest, file_digest, require
from scripts.campaign_20261110.goal_groups import read_json


def bind_camera_room_candidate(candidate, directory, *, code, root, started_at):
    directory, code, root = (Path(p).resolve() for p in (directory,code,root))
    require(directory.is_relative_to(root) and code.is_relative_to(root/"code"), "camera binding outside owned sandbox")
    require(candidate.task == "camera_relocalization-standard" and candidate.split in ("train","eval_category"),
            "camera room binder requires a training/category camera template")
    require(all(o.semantic_kind == "floor" for o in candidate.objects), "camera-standard room candidate unexpectedly contains objects")
    metadata, state = directory/"template_meta.json", directory/"template.json"
    require(all(p.is_file() and p.stat().st_mtime >= started_at for p in (metadata,state)), "camera template is not fresh")
    meta, full = read_json(metadata), read_json(state)
    require(meta.get("n_written",0) >= 1, "camera bootstrap has no successful fresh instance")
    require(meta["scene"] == candidate.scene and meta["task_rooms"] == [candidate.room], "fresh camera room differs from candidate")
    require(meta["template_sha256"] == file_digest(state), "fresh camera template bytes changed")
    definition = (directory/meta["activity_definition"]).resolve()
    require(definition.is_relative_to(root) and file_digest(definition) == meta["bddl_sha256"]
            == candidate.scene_setup["runtime_bddl_sha256"], "fresh camera definition differs from approved candidate")
    stable = code/"Behavior-1K/datasets/2026-challenge-task-instances/scenes"/candidate.scene/"json"/(candidate.scene+"_stable.json")
    stable_info = read_json(stable)["objects_info"]["init_info"]
    expected = {o.instance:o for o in candidate.objects}
    bindings, refs, resolutions, actual_init = {}, [], [], {}
    for row in meta["bindings"]:
        name = row["name"]
        require(name in expected and name not in bindings, "unexpected or duplicated camera floor binding")
        ref = expected[name]
        args = full["objects_info"]["init_info"][row["scene_object_name"]]["args"]
        require((args["category"],args["model"]) == (ref.category,ref.model), "fresh camera floor model differs from candidate")
        scale = args.get("scale")
        require(isinstance(scale,list) and len(scale)==3, "fresh camera floor scale missing")
        if max(abs(a-b) for a,b in zip(scale,ref.default_scale)) > 1e-6:
            actual = stable_info.get(row["scene_object_name"],{}).get("args",{})
            require((actual.get("category"),actual.get("model")) == (ref.category,ref.model)
                    and isinstance(actual.get("scale"),list) and len(actual["scale"])==3
                    and max(abs(a-b) for a,b in zip(scale,actual.get("scale",[]))) <= 1e-6,
                    "fresh floor scale differs from both candidate and untouched stable scene")
            resolutions.append({"instance":name,"candidate_scale":list(ref.default_scale),"actual_default_scale":scale,
                "source":str(stable),"source_sha256":file_digest(stable),"reason":"Actual untouched stable-scene default supersedes best-scene proposal."})
        asset = code/"Behavior-1K/datasets/behavior-1k-assets/objects"/ref.category/ref.model/"usd"/(ref.model+".encrypted.usd")
        require(file_digest(asset) == ref.asset_sha256, "fresh floor asset bytes differ from readiness contract")
        refs.append(replace(ref,default_scale=tuple(scale)))
        bindings[name] = row["scene_object_name"]
        actual_init[name] = args
    require(set(bindings) == set(expected), "camera bootstrap omitted a required floor")
    setup = {**candidate.scene_setup,"source_template":str(directory),"source_template_sha256":file_digest(state),
        "runtime_template_id":meta["template_id"],"source_bddl_sha256":meta["bddl_sha256"],"source_alias":meta["variant"],
        "inst_to_name":bindings,"task_rooms":meta["task_rooms"],"fresh_metadata_binding_check":"passed",
        "fresh_metadata_sha256":file_digest(metadata),"actual_init_args":actual_init,"default_scale_resolution":resolutions}
    return replace(candidate,objects=tuple(refs),definition_sha256=digest({"candidate_definition":candidate.definition_sha256,
        "template_sha256":file_digest(state),"bddl_sha256":meta["bddl_sha256"],"actual_init_args":actual_init}),scene_setup=setup)
