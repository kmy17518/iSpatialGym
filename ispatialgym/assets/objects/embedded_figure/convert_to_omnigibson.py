"""Convert the embedded figure's URDFs (``model/isgefa`` ... ``isgefd``) into OmniGibson USD assets.

Mirrors ``../block_board/convert_to_omnigibson.py`` (the ARAT ``convert_to_omnigibson.py`` recipe), for the four parts in
one OmniGibson session. Each asset is written to ``<OMNIGIBSON_DATA_PATH>/ispatialgym-assets/objects/<category>/<model>``
so that e.g. ``DatasetObject(category="jigsaw_puzzle_piece", model="isgefa", dataset_name="ispatialgym-assets")`` (the
target) loads it, and copied into the repo's pre-converted dataset ``ispatialgym/assets/datasets/ispatialgym-assets``
(what ``setup.sh`` installs on the pods). Each model's build manifest plus ``model/assembly.json`` (where the parts lie
relative to the figure's centre) go to ``misc/isg_embedded_figure.json`` (``import_obj_metadata`` turns every entry of
``metadata.json`` into a USD attribute and rejects nested lists).

Run with an OmniGibson environment::

    OMNIGIBSON_HEADLESS=1 python convert_to_omnigibson.py
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import omnigibson as og  # noqa: E402
from omnigibson.utils.asset_conversion_utils import (  # noqa: E402
    convert_urdf_to_usd,
    import_obj_metadata,
    record_obj_metadata_from_urdf,
)
from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET_NAME = "ispatialgym-assets"
REPO_DATASET = HERE.parents[1] / "datasets" / DATASET_NAME  # ispatialgym/assets/datasets/ispatialgym-assets
MODELS = ("isgefa", "isgefb", "isgefc", "isgefd")


def main() -> None:
    assembly = json.loads((HERE / "model" / "assembly.json").read_text())
    dataset_root = Path(get_dataset_path(DATASET_NAME)).resolve()
    staged = []
    for model in MODELS:
        manifest = json.loads((HERE / "model" / model / "build_manifest.json").read_text())
        model_dir = dataset_root / "objects" / manifest["category"] / model
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        source_urdf = HERE / "model" / model / f"{model}.urdf"
        record_obj_metadata_from_urdf(urdf_path=str(source_urdf), obj_dir=str(model_dir), joint_setting="zero", overwrite=True)
        info = {"model": {k: v for k, v in manifest.items() if k != "urdf"}, "assembly": assembly}
        (model_dir / "misc" / "isg_embedded_figure.json").write_text(json.dumps(info, indent=2) + "\n")
        staged.append((manifest["category"], model, source_urdf, model_dir))

    og.launch()
    try:
        for category, model, source_urdf, model_dir in staged:
            try:
                processed_urdf, usd_path = convert_urdf_to_usd(
                    urdf_path=str(source_urdf),
                    obj_category=category,
                    obj_model=model,
                    dataset_root=str(dataset_root),
                    use_omni_convex_decomp=False,
                    use_usda=False,
                    merge_fixed_joints=False,
                    import_inertia_tensor=True,
                )
                import_obj_metadata(usd_path=usd_path, obj_category=category, obj_model=model, dataset_root=str(dataset_root))
                report = {
                    "status": "passed",
                    "dataset_name": DATASET_NAME,
                    "category": category,
                    "model": model,
                    "source_urdf": str(source_urdf),
                    "processed_urdf": str(processed_urdf),
                    "usd": str(usd_path),
                    "usd_size_bytes": Path(usd_path).stat().st_size,
                }
                (HERE / "model" / model / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
                dst = REPO_DATASET / "objects" / category / model
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(model_dir, dst)
                print(f"EMBEDDED_FIGURE_CONVERSION_PASSED {category}/{model} usd={usd_path} repo_copy={dst}", flush=True)
            except BaseException as exc:
                traceback.print_exc()
                (HERE / "model" / model / "conversion_error.json").write_text(
                    json.dumps({"exception": type(exc).__name__, "message": str(exc)}, indent=2) + "\n"
                )
                raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
