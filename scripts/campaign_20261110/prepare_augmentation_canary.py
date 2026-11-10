#!/usr/bin/env python3
"""Prepare a separate, fresh, ready-asset augmentation validation configuration."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ispatialgym.sampling.campaign_augmentation import lighting_draw, removal_candidates
from scripts.campaign_20261110.contracts import (
    ObjectRef, Perturbations, ScaleState, Template, ValidationKey, file_digest, require, write_json_atomic,
)


def prepare(base_group, snapshot, stable_root, output, seed=1, mode="both"):
    base_group, snapshot, output = Path(base_group), Path(snapshot), Path(output)
    require(not output.exists(), "augmentation canary output already exists")
    config = json.loads((base_group / "campaign_config.json").read_text())
    source = json.loads((base_group / "source.json").read_text())
    raw = config["template"]
    raw["objects"] = tuple(ObjectRef(**obj) for obj in raw["objects"])
    original = Template(**raw)
    template = replace(original, split="train", template_id=f"train__{original.class_id}__augmentation_{mode}_{seed}")
    path = Path(stable_root) / template.scene / "json" / f"{template.scene}_stable.json"
    stable = json.loads(path.read_text())
    removed = (removal_candidates(stable, rooms=template.scene_setup.get("task_rooms", [template.room]),
                                 protected=template.scene_setup.get("inst_to_name", {}).values(), seed=seed)
               if mode in ("scene", "both") else [])
    lighting = lighting_draw(seed) if mode in ("lighting", "both") else {}
    require(bool(removed) or mode == "lighting", "seed selected no scene perturbation; choose another canary seed")
    require(bool(lighting) or mode == "scene", "seed selected no lighting perturbation; choose another canary seed")
    p = Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, "fixed") for obj in template.objects},
                      tuple(removed), lighting)
    p.validate("train", [obj.instance for obj in template.objects])
    key = ValidationKey(template.fingerprint, file_digest(snapshot / ".snapshot_commit"),
                        config["validation_key"]["physics_sha256"], config["validation_key"]["data_sha256"], p.scale_fingerprint)
    config.update(split="train", purpose="validation", template=asdict(template), perturbations=asdict(p), validation_key=asdict(key))
    output.mkdir(parents=True)
    write_json_atomic(output / "campaign_config.json", config)
    write_json_atomic(output / "source.json", {**source, "purpose": "augmentation_canary", "source_group": str(base_group),
                                               "stable_scene_sha256": file_digest(path), "mode": mode, "seed": seed})
    write_json_atomic(output / "empty_eval_index.json", {"schema": "isg-instance-index-v1", "n": 0, "instances": []})
    return {"output": str(output), "removed": list(removed), "lighting": lighting, "template": source["template_dir"],
            "validation_key": asdict(key), "release_status": "requires fresh demo, replay and rendered lighting review"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-group", "snapshot", "stable-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", choices=("scene", "lighting", "both"), default="both")
    args = p.parse_args()
    print(json.dumps(prepare(args.base_group, args.snapshot, args.stable_root, args.output, args.seed, args.mode), indent=2))


if __name__ == "__main__":
    main()
