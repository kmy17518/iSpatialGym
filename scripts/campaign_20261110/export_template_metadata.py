#!/usr/bin/env python3
"""Read-only export of the frozen cluster alias registry and template metadata.

Run on the cluster; stdout is JSON. No simulator import or instance state is read.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    args = parser.parse_args()
    code = Path(args.code)
    spec = importlib.util.spec_from_file_location("campaign_submit_fast", code / "scripts/trainfast/submit_fast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    aliases = module.aliases(SimpleNamespace(code=str(code), registry=None))
    records = []
    asset_hashes = {}
    for key, row in sorted(aliases.items()):
        path = Path(row.get("template") or row.get("template_dir") or "")
        if path.is_file():
            path = path.parent
        meta_path = path / "template_meta.json"
        record = {"key": key, "row": row, "metadata_path": str(meta_path)}
        try:
            meta = json.loads(meta_path.read_text())
            record["metadata"] = {k: meta[k] for k in (
                "template_id", "scene", "variant", "subtask", "choice", "rooms", "bindings", "template_sha256",
                "task_rooms", "load_room_instances", "choices", "scene_variant", "bddl_sha256", "problem",
                "template_facts", "inst_to_name") if k in meta}
            template = json.loads((path / "template.json").read_text())
            init_info = template.get("objects_info", {}).get("init_info", {})
            names = {b.get("scene_object_name") for b in meta.get("bindings", [])}
            record["object_init_args"] = {name: value.get("args", {}) for name, value in init_info.items() if name in names}
            for name, init in record["object_init_args"].items():
                dataset = init.get("dataset_name") or "behavior-1k-assets"
                category, model = init["category"], init["model"]
                root = code / "ispatialgym/assets/datasets" / dataset if dataset == "ispatialgym-assets" else code / "Behavior-1K/datasets" / dataset
                usd = root / "objects" / category / model / "usd" / (model + (".encrypted.usd" if dataset == "behavior-1k-assets" else ".usd"))
                key = f"{dataset}/{category}/{model}"
                if key not in asset_hashes:
                    if usd.is_file():
                        data = usd.read_bytes()
                        asset_hashes[key] = {"path": str(usd.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "md5": hashlib.md5(data).hexdigest()}
                    else:
                        asset_hashes[key] = {"path": str(usd), "error": "not found"}
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
        records.append(record)
    print(json.dumps({"source_code": str(code), "records": records, "runtime_asset_hashes": asset_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
