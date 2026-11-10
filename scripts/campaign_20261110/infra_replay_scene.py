"""Pin true action replay to the exact scene used to collect its source demo."""
import argparse
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
    objects = sorted(actual_scene["objects_info"]["init_info"])
    require(objects == sorted(recorded_scene["objects_info"]["init_info"]), "HDF5/source scene object sets differ")
    return {"schema": "isg-recorded-action-replay-scene-v1", "scene_mode": mode,
            "partial_scene": partial, "scene_file": str(source), "scene_sha256": sha(source),
            "source_report": str(report_path), "source_report_sha256": sha(report_path),
            "template_meta": str(template_meta), "template_meta_sha256": sha(template_meta),
            "recorded_hdf5_scene_sha256": hashlib.sha256(encoded_scene.encode()).hexdigest(),
            "rooms": room_names(report), "objects": objects, "verified_replay": False}


def verify(identity, report_path):
    report = read(report_path)
    require(scene_mode(report) == identity["scene_mode"], "replay used a different scene mode")
    require(str(scene_file(report)) == identity["scene_file"], "replay used another scene file")
    require(room_names(report) == identity["rooms"], "replay used different loaded rooms")
    require(sha(identity["scene_file"]) == identity["scene_sha256"], "source scene bytes changed during replay")
    require(sha(identity["source_report"]) == identity["source_report_sha256"], "source demo report changed")
    require(sha(identity["template_meta"]) == identity["template_meta_sha256"], "source template metadata changed")
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
