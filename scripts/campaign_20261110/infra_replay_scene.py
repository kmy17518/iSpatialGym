"""Pin true action replay to the exact scene used to collect its source demo."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def scene_mode(report):
    return (report.get("scene_crop") or {}).get("crop", "full")


def scene_file(report):
    return Path((report.get("scene_crop") or {}).get("loaded_scene_file") or
                str(Path(report["template_dir"]) / "template.json")).resolve()


def room_names(report):
    return sorted((report.get("load_rooms") or {}).get("rooms") or report.get("load_room_instances") or [])


def physical_objects(scene):
    fields = ("name", "category", "model", "scale", "fixed_base", "visual_only", "expected_file_hash")
    return {name: {"class_module": row.get("class_module"), "class_name": row.get("class_name"),
                   "args": {key: row.get("args", {}).get(key) for key in fields}}
            for name, row in scene["objects_info"]["init_info"].items()}


def object_digest(scene):
    return hashlib.sha256(json.dumps(physical_objects(scene), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_room_projection(source, recorded, report):
    """Apply the audited native room filter, keeping the configured robot.

    Full template JSON may contain rooms that the collector deliberately does
    not load. The HDF5 is the resulting scene, not a copy of that whole JSON.
    Unknown filter implementations or category filters remain fail-closed.
    """
    runtime = Path(report["provenance"]["isg_root"])
    og = runtime / "Behavior-1K/OmniGibson/omnigibson"
    path = og / "scenes/interactive_traversable_scene.py"
    module = ast.parse(path.read_text())
    method = next(node for cls in module.body if isinstance(cls, ast.ClassDef) and cls.name == "InteractiveTraversableScene"
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_should_load_object")
    try:
        canonical_method = ast.dump(method, include_attributes=False, show_empty=True)
    except TypeError:  # Python <=3.12 always includes empty list fields.
        canonical_method = ast.dump(method, include_attributes=False)
    canonical_method = canonical_method.replace(", type_params=[]", "")
    method_sha = hashlib.sha256(canonical_method.encode()).hexdigest()
    require(method_sha == "2b1664c4c71fef3188542fa4eda4b5c6f577d2961f62b48079e73bee567accf7",
            "native room-filter implementation differs from audited policy")
    constants_path = og / "utils/constants.py"
    constants = {}
    for node in ast.parse(constants_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("STRUCTURE_CATEGORIES", "GROUND_CATEGORIES"):
                require(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "frozenset", "unrecognized native room-filter constants")
                constants[name] = set(ast.literal_eval(node.value.args[0]))
    structures = (constants["STRUCTURE_CATEGORIES"] - constants["GROUND_CATEGORIES"]) | {"door", "sliding_door"}
    init = recorded.get("init_info", {})
    require(init.get("class_name") == "InteractiveTraversableScene", "recorded room filter has unknown scene type")
    args = init["args"]
    require(not args.get("load_task_relevant_only") and args.get("load_object_categories") is None
            and args.get("not_load_object_categories") is None, "unsupported recorded category filter")
    rooms = args.get("load_room_instances")
    require(rooms is None or isinstance(rooms, list), "invalid recorded native room list")
    require(sorted(rooms or []) == room_names(report), "HDF5/report native room filters differ")
    selected = []
    for name, row in source["objects_info"]["init_info"].items():
        obj = row["args"]
        in_rooms = obj.get("in_rooms", [])
        in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms
        # run_isg supplies its robot config separately after scene loading.
        robot = row.get("class_module", "").startswith("omnigibson.robots.")
        if robot or obj.get("category", "object") in structures or rooms is None or set(rooms) & set(in_rooms):
            selected.append(name)
    return sorted(selected), {"policy": "native_room_filter_plus_configured_robot", "method_sha256": method_sha,
        "source": str(path.resolve()), "source_sha256": sha(path),
        "constants": str(constants_path.resolve()), "constants_sha256": sha(constants_path),
        "rooms": rooms, "always_loaded_categories": sorted(structures)}


def prepare(demo, job):
    demo, job = Path(demo).resolve(), Path(job).resolve()
    require(demo.is_relative_to(job / "demo"), "source demo is outside owned job")
    report_path = demo.parent / "meta.json"
    report = read(report_path)
    source = scene_file(report)
    require(source.is_relative_to(job / "inst") and source.is_file(), "recorded scene is missing or outside source job")
    template_meta = Path(report["template_dir"]) / "template_meta.json"
    tmeta = read(template_meta)
    mode = scene_mode(report)
    partial = mode != "full"
    if partial:
        require((tmeta.get("scene_crop") or {}).get("mode") == mode, "source report/template crop differs")
        expected = (Path(report["template_dir"]) / tmeta["scene_crop"]["loaded_scene_file"]).resolve()
        require(expected == source, "source report/template scene file differs")
    with h5py.File(demo, "r") as stream:
        attrs = stream["data"].attrs
        crop = json.loads(attrs.get("isg_scene_crop", "null"))
        require((crop or {}).get("mode", "full") == mode, "HDF5/report crop mode differs")
        if partial:
            require(sorted(crop["loaded_rooms"]) == room_names(report), "HDF5/report room sets differ")
        encoded_scene = attrs["scene_file"]
        if isinstance(encoded_scene, bytes):
            encoded_scene = encoded_scene.decode()
        recorded_scene = json.loads(encoded_scene)
    actual_scene = read(source)
    source_objects = sorted(actual_scene["objects_info"]["init_info"])
    objects = sorted(recorded_scene["objects_info"]["init_info"])
    projection = None
    if source_objects != objects:
        require(set(objects) <= set(source_objects), "HDF5/source scene object sets differ: unexpected recorded object")
        require((report.get("provenance") or {}).get("isg_root"), "HDF5/source scene object sets differ: no native runtime proof")
        expected, projection = native_room_projection(actual_scene, recorded_scene, report)
        require(expected == objects, "HDF5/source scene object sets differ after exact native room filtering")
    actual_physical, recorded_physical = physical_objects(actual_scene), physical_objects(recorded_scene)
    require(all(actual_physical[name] == recorded_physical[name] for name in objects),
            "HDF5/source object physical identities differ")
    return {"schema": "isg-recorded-action-replay-scene-v1", "scene_mode": mode,
            "partial_scene": partial, "scene_file": str(source), "scene_sha256": sha(source),
            "source_report": str(report_path), "source_report_sha256": sha(report_path),
            "template_meta": str(template_meta), "template_meta_sha256": sha(template_meta),
            "recorded_hdf5_scene_sha256": hashlib.sha256(encoded_scene.encode()).hexdigest(),
            "rooms": room_names(report), "objects": objects, "source_objects": source_objects,
            "room_projection": projection, "recorded_object_identity_sha256": object_digest(recorded_scene),
            "verified_replay": False}


def verify(identity, report_path):
    report = read(report_path)
    require(scene_mode(report) == identity["scene_mode"], "replay used a different scene mode")
    require(str(scene_file(report)) == identity["scene_file"], "replay used another scene file")
    require(room_names(report) == identity["rooms"], "replay used different loaded rooms")
    require(sha(identity["scene_file"]) == identity["scene_sha256"], "source scene bytes changed during replay")
    require(sha(identity["source_report"]) == identity["source_report_sha256"], "source demo report changed")
    require(sha(identity["template_meta"]) == identity["template_meta_sha256"], "source template metadata changed")
    if identity.get("room_projection"):
        projection = identity["room_projection"]
        require(sha(projection["source"]) == projection["source_sha256"]
                and sha(projection["constants"]) == projection["constants_sha256"], "native room-filter source changed")
    if identity.get("recorded_object_identity_sha256"):
        replay = Path(report_path).parent / "demo.hdf5"
        with h5py.File(replay, "r") as stream:
            encoded = stream["data"].attrs["scene_file"]
            recorded = json.loads(encoded.decode() if isinstance(encoded, bytes) else encoded)
        require(object_digest(recorded) == identity["recorded_object_identity_sha256"],
                "actual replay object set or physical identity differs from source demo")
    return {**identity, "verified_replay": True, "replay_report": str(report_path), "replay_report_sha256": sha(report_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo")
    parser.add_argument("--job")
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--replay-report", type=Path)
    args = parser.parse_args()
    result = verify(read(args.identity), args.replay_report) if args.replay_report else prepare(args.demo, args.job)
    args.identity.write_text(json.dumps(result, indent=2) + "\n")
    print("1" if result["partial_scene"] else "0")


if __name__ == "__main__":
    main()
