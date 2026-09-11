"""Convert the Kohs block URDF (``model/isgblk``) into an OmniGibson USD asset.

Mirrors ``Behavior-1K-arat/asset_pipeline/arat_v1/components/source/convert_to_omnigibson.py``. The asset
is written to ``<OMNIGIBSON_DATA_PATH>/ispatialgym-assets/objects/toy_dice/isgblk`` so that
``DatasetObject(category="toy_dice", model="isgblk", dataset_name="ispatialgym-assets")`` loads it.

Run with an OmniGibson environment (``behavior_dex`` as in the ARAT pipeline, or ``ispatialgym``)::

    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_DATA_PATH=<repo>/Behavior-1K/datasets \
        conda run -n behavior_dex python convert_to_omnigibson.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
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


def main() -> None:
    manifest = json.loads((HERE / "model" / "isgblk" / "build_manifest.json").read_text())
    category, model = manifest["category"], manifest["model"]
    source_urdf = HERE / "model" / model / f"{model}.urdf"
    dataset_root = Path(get_dataset_path(DATASET_NAME)).resolve()
    model_dir = dataset_root / "objects" / category / model
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    record_obj_metadata_from_urdf(urdf_path=str(source_urdf), obj_dir=str(model_dir), joint_setting="zero", overwrite=True)
    # Face / mass information is kept in a separate file: ``import_obj_metadata`` converts every entry of
    # metadata.json into USD attributes and rejects nested lists.
    (model_dir / "misc" / "isg_block_faces.json").write_text(
        json.dumps(
            {
                "faces": manifest["faces"],
                "opposite_pairs": manifest["opposite_pairs"],
                "tile_up_axis": manifest["tile_up_axis"],
                "mass_kg": manifest["mass_kg"],
                "edge_m": manifest["edge_m"],
                "source_urdf": str(source_urdf),
            },
            indent=2,
        )
        + "\n"
    )

    og.launch()
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
        print(f"KOHS_BLOCK_CONVERSION_PASSED usd={usd_path}", flush=True)
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
