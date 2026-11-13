"""Consume exact prepared handle rows through the existing native sampler.

Joining a declaration is CPU work. Only the sampler's actual selected actor
can produce a lookup observation; neither operation proves motion readiness.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scripts.campaign_20261110.contracts import digest, file_digest, require


KEY = "training_prepared_handle"
OPTIONS = ("handle_file", "handle_key", "handle_row", "joint")


def checked(reference):
    path = Path(reference["path"])
    require(path.is_absolute() and file_digest(path) == reference["sha256"],
            "prepared handle input changed: " + str(path))
    return json.loads(path.read_text())


def binding(metadata_ref, selection_ref, configuration_id, variant):
    metadata, selection = checked(metadata_ref), checked(selection_ref)
    require(metadata.get("schema") == "isg-exact-articulation-handle-binding-v1"
            and metadata["source"] == selection_ref, "handle selection source differs")
    matches = [r for r in metadata["rows"] if r["configuration_id"] == configuration_id]
    choices = [r for r in selection["rows"] if r["configuration_id"] == configuration_id]
    require(len(matches) == len(choices) == 1, "handle configuration missing or ambiguous")
    row, choice = matches[0], choices[0]
    require(row["status"] == "consumed_handle_metadata_resolved"
            and row.get("annotation_lookup_passed") is True,
            "prepared handle annotation is blocked: " + str(row.get("reason", row["status"])))
    require(digest(choice) == row["scale_row_sha256"]
            and choice["class_id"] == row["class_id"], "selected handle row identity changed")
    targets = [r for r in choice["roles"] if r["role"] == "target"]
    require(len(targets) == 1, "handle selection requires one exact target")
    target = targets[0]
    require(all(target[k] == row[k] for k in ("model_key", "applied_scale", "asset_sha256"))
            and target["joint"] == row["selected_joint"], "prepared target identity differs")
    options = row["role_options"]
    require(set(options) == set(OPTIONS)
            and options["handle_file"] == row["handle_record"]["path"]
            and options["handle_key"] == row["handle_key"]
            and options["joint"] == row["consumer_joint"]
            and options["handle_row"] == row["native_selected_handle"]["source"],
            "prepared native role options differ")
    scene_actor(row, choice)
    value = {"schema": "isg-prepared-articulation-consumer-v1",
             "metadata": metadata_ref, "selection": selection_ref,
             "configuration_id": configuration_id, "class_id": row["class_id"],
             "variant": variant, "selected_row_sha256": digest(choice),
             "metadata_row_sha256": digest(row), "role_options": copy.deepcopy(options)}
    return value, row, choice


def scene_actor(row, choice):
    """Instance annotations belong to one actor in one exact source scene."""
    frame = row.get("handle_frame", "canonical")
    require(frame in ("canonical", "instance"), "unsupported prepared annotation frame")
    if frame == "canonical":
        require(not row.get("scene_actor_binding"), "canonical handle has an instance-only binding")
        return None
    from scripts.campaign_20261110.bind_fresh_candidates import native_scale_equal

    actor = row.get("scene_actor_binding")
    target = next(role for role in choice["roles"] if role["role"] == "target")
    require(isinstance(actor, dict) and actor.get("scene") == choice["scene"]
            and actor.get("name") in target.get("scene_matching_instances", [])
            and actor.get("source") == target.get("scene_scale_evidence"),
            "instance handle requires its exact selected scene actor")
    name = actor["name"]
    require(actor.get("source_pointer") == "/objects_info/init_info/" + name + "/args",
            "instance handle source pointer differs")
    scene = checked(actor["source"])
    args = scene.get("objects_info", {}).get("init_info", {}).get(name, {}).get("args", {})
    require(args.get("category", "") + "/" + args.get("model", "") == row["model_key"]
            and native_scale_equal(row["applied_scale"], args.get("scale")),
            "instance handle scene actor model or scale differs")
    annotation = checked(row["handle_record"])
    require(annotation.get("scene") == actor["scene"] and annotation.get("name") == name
            and annotation.get("category", "") + "/" + annotation.get("model", "") == row["model_key"],
            "instance handle annotation belongs to another scene actor")
    return actor


def bind_generation(entry, *, metadata_ref, selection_ref, configuration_id, variant):
    """Return a private declaration with exact target options; retain geometry.

    This joins metadata to an already selected generation declaration. It does
    not create scene actors, substitute supports, or mark dependencies ready.
    """
    value, row, choice = binding(metadata_ref, selection_ref, configuration_id, variant)
    require("target" in entry.get("roles", {}), "generation declaration has no target role")
    require(choice["scene"] in entry.get("scenes", {}), "selected scene missing from declaration")
    actor = scene_actor(row, choice)
    if actor is not None:
        require(entry["scenes"][choice["scene"]].get("bindings", {}).get("target") == [actor["name"]],
                "instance handle generation declaration names another actor")
    result = copy.deepcopy(entry)
    target = result["roles"]["target"]
    require(KEY not in target, "generation declaration already has a prepared handle binding")
    target.update(copy.deepcopy(row["role_options"]))
    target[KEY] = value
    return result


def _close(actual, expected, tolerance=1e-6):
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_close(a, b, tolerance) for a, b in zip(actual, expected))
    if isinstance(expected, (float, int)):
        return math.isfinite(float(actual)) and abs(float(actual) - expected) <= tolerance
    return actual == expected


def verify_inputs(sampler, obj, *, handles_csv):
    """Check exact files and the real selected actor before any fallback lookup."""
    raw = (sampler.gp.entry.get("roles") or {}).get(sampler.target_role) or {}
    declared = raw.get(KEY)
    if declared is None:
        return None
    require(sampler.target_role == "target", "prepared handle belongs to the target role")
    value, row, choice = binding(declared["metadata"], declared["selection"],
                                 declared["configuration_id"], sampler.gp.variant)
    require(declared == value, "generation handle binding changed")
    require(all(sampler.role_option(k) == row["role_options"][k] for k in OPTIONS),
            "native sampler resolved different handle options")
    require(sampler.scene == choice["scene"] and sampler.target is obj,
            "prepared handle is bound to a different scene or actor")
    actor = scene_actor(row, choice)
    if actor is not None:
        require(obj.name == actor["name"]
                and file_digest(sampler.world.stable_path) == actor["source"]["sha256"],
                "native instance handle actor or source scene differs")
    require(f"{obj.category}/{obj.model}" == row["model_key"]
            and _close(list(obj.scale), row["applied_scale"]),
            "native selected actor model or applied scale differs")
    joint_name = row["consumer_joint"]
    require(joint_name in obj.joints, "native selected actor lacks the prepared joint")
    joint = obj.joints[joint_name]
    selected = row["selected_joint"]
    require(("RevoluteJoint" if joint.is_revolute else "PrismaticJoint") == selected["type"]
            and _close(float(joint.lower_limit), selected["lower_limit"])
            and _close(float(joint.upper_limit), selected["upper_limit"]),
            "native selected joint type or limits differ")
    checked(row["handle_record"])
    require(file_digest(handles_csv) == row["handle_csv"]["sha256"], "native handle CSV changed")
    return {"binding": value, "row": row, "actor": {"name": obj.name,
            "model_key": row["model_key"], "applied_scale": [float(v) for v in obj.scale],
            "joint": joint_name, "scene": sampler.scene}}


def verify_selected(prepared, *, handle, source, path, frame, joint_name, width_range):
    """Observe unmodified native row selection before optional approach repair."""
    row = prepared["row"]
    require(Path(path).resolve() == Path(row["handle_record"]["path"]).resolve()
            and frame == row.get("handle_frame", "canonical") and joint_name == row["consumer_joint"],
            "native handle lookup used another annotation or joint")
    expected = row["native_selected_handle"]
    require(source == expected["source"] and _close(list(width_range), row["width_range"]),
            "native handle selection or width policy differs")
    require(all(k in handle and _close(handle[k], v) for k, v in expected.items() if k != "source"),
            "native selected handle differs from prepared metadata")
    return {"schema": "isg-native-prepared-handle-lookup-v1", "binding": prepared["binding"],
            "actor": prepared["actor"], "handle_record": row["handle_record"],
            "handle_csv": row["handle_csv"], "selected_handle": copy.deepcopy(expected),
            "generation_role_options_binding": True, "native_selected_actor_handle_lookup": True,
            "motion_readiness_proven": False, "production_released": False}
