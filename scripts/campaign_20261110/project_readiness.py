#!/usr/bin/env python3
"""Pin readiness of the seven established custom models to frozen assets/evidence.

This explicit registry does not grant readiness to newly generated assets.
"""
import json
from pathlib import Path
from contracts import file_digest

MODELS = {"toy_dice/isgblk": "kohs_block", "tray/isgboard": "block_board",
          "jigsaw_puzzle_piece/isgefa": "embedded_figure", "jigsaw_puzzle_piece/isgefb": "embedded_figure",
          "jigsaw_puzzle_piece/isgefc": "embedded_figure", "jigsaw_puzzle_piece/isgefd": "embedded_figure",
          "classroom_mock_up/isgsma": "mental_rotation_block"}


def build(root, source_checkout, frozen_export, destination):
    runtime = json.loads(frozen_export.read_text())["runtime_asset_hashes"]
    records = {}
    for key, family in MODELS.items():
        category, model = key.split("/")
        usd = root / "ispatialgym/assets/datasets/ispatialgym-assets/objects" / category / model / "usd" / (model + ".usd")
        conversion = root / "ispatialgym/assets/objects" / family / "model" / model / "conversion.json"
        if not conversion.is_file():
            conversion = source_checkout / "ispatialgym/assets/objects" / family / "model" / model / "conversion.json"
        report = json.loads(conversion.read_text())
        if report.get("status") != "passed" or report.get("model") != model:
            raise ValueError(f"No passed conversion evidence for {key}")
        sha = file_digest(usd)
        observed = runtime["ispatialgym-assets/" + key]
        if sha != observed["sha256"]:
            raise ValueError(f"Local custom asset differs from established frozen runtime: {key}")
        evidence = {str(conversion): file_digest(conversion)}
        smoke = root / "ispatialgym/assets/objects/embedded_figure/previews/omnigibson/report.json"
        if family in {"embedded_figure", "mental_rotation_block"} and smoke.is_file():
            evidence[str(smoke)] = file_digest(smoke)
        records[key] = {"ready": True, "asset_sha256": sha,
                        "readiness_source": json.dumps({"type": "established_project_asset", "asset_sha256": sha,
                            "conversion_evidence_sha256": evidence, "frozen_runtime_source": observed["path"],
                            "scope": "baseline custom asset readiness; fresh campaign demo/replay still required"}, sort_keys=True),
                        "model_category_note": "Original conversion calls the same isgsma asset toy_figure; baseline registers classroom_mock_up." if model == "isgsma" else ""}
    destination.write_text(json.dumps({"objects": records}, indent=2, sort_keys=True) + "\n")
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build(root, Path("/home/minyeongk/Desktop/projects/ispatialgym"),
          root / "tmp/campaign_20261110/inventory/cluster_templates_with_scales.json",
          root / "tmp/campaign_20261110/inventory/project_readiness.json")
