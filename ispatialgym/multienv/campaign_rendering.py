"""Manifest-bound restoration for kinematic campaign observation rendering."""
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, validate_campaign_block


def selected_records(manifest, demo_dirs):
    """Select accepted episodes only, with one exact rendering configuration."""
    records = {}
    for line in Path(manifest).read_text().splitlines():
        row = json.loads(line)
        directory = str(Path(row["artifacts"]["demo"]["path"]).resolve().parent)
        require(directory not in records, "duplicate demo in accepted manifest")
        records[directory] = row
    wanted = [str(Path(path).resolve()) for path in demo_dirs]
    require(len(set(wanted)) == len(wanted) and wanted, "empty or duplicated campaign render input")
    require(all(path in records for path in wanted), "render input is not in accepted manifest")
    selected = {path: records[path] for path in wanted}
    require(len({row["render_compatibility_sha256"] for row in selected.values()}) == 1,
            "campaign rendering batch mixes scene, scale, light or validation configuration")
    return selected


def checked_metadata(record, demo_dir):
    demo_dir = Path(demo_dir).resolve()
    require(Path(record["artifacts"]["demo"]["path"]).resolve() == demo_dir / "demo.hdf5", "manifest demo path mismatch")
    for kind in ("tro", "demo", "report", "sidecar"):
        item = record["artifacts"][kind]
        require(file_digest(item["path"]) == item["sha256"], f"render input {kind} changed after export")
    with open(record["artifacts"]["tro"]["path"]) as stream:
        meta = json.load(stream)["ispatialgym"]
    require(meta["instance_id"] == record["instance_id"] == demo_dir.name, "render instance identity mismatch")
    validate_campaign_block(meta["campaign"], object_bindings=meta["object_bindings"])
    require(meta["campaign"]["validation_key"] == record["validation_key"], "render validation key mismatch")
    require(meta["campaign"]["perturbations"] == record["perturbations"], "render perturbations differ from accepted manifest")
    return meta


def restore_campaign_scene(scene, sim, meta):
    """Call after loading the first recorded state, before rendering any frame."""
    from ispatialgym.sampling.campaign_runtime import verify_loaded

    physical = {}
    for instance in meta["campaign"]["perturbations"]["scales"]:
        binding = meta["object_bindings"][instance]
        name = binding.get("scene_object_name") if isinstance(binding, dict) else binding
        require(bool(name), f"missing campaign render binding: {instance}")
        obj = scene.object_registry("name", name)
        require(obj is not None, f"campaign render object missing: {name}")
        if isinstance(binding, dict):
            for field in ("category", "model"):
                if binding.get(field):
                    require(getattr(obj, field) == binding[field], f"campaign render {field} differs: {instance}")
        physical[instance] = obj
    return verify_loaded(meta["campaign"], physical, scene=scene, sim=sim)
