"""Convert the block board URDF (``model/isgboard``) into an OmniGibson USD asset.

Mirrors ``../kohs_block/convert_to_omnigibson.py`` (the ARAT ``convert_to_omnigibson.py`` recipe). The asset is written to
``<OMNIGIBSON_DATA_PATH>/ispatialgym-assets/objects/tray/isgboard`` so that
``DatasetObject(category="tray", model="isgboard", dataset_name="ispatialgym-assets")`` loads it, and copied into the
repo's pre-converted dataset ``ispatialgym/assets/datasets/ispatialgym-assets`` (what ``setup.sh`` installs on the pods).

Run with an OmniGibson environment::

    OMNIGIBSON_HEADLESS=1 python convert_to_omnigibson.py
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
REPO_DATASET = HERE.parents[1] / "datasets" / DATASET_NAME  # ispatialgym/assets/datasets/ispatialgym-assets
#: physics material baked into the USD (a lacquered hardwood board): a cube edge that lands on a chamfer must slide down into
#: the seat instead of leaning on it (drop test 2026-09-20 23:xx: with PhysX's default friction a cube rested with one bottom
#: edge on the 56-degree chamfer and the other on the floor, tilted 17-20 degrees, in a quarter of the asymmetric landings)
BOARD_FRICTION = float(os.environ.get("BOARD_FRICTION", "0.25"))
BOARD_RESTITUTION = 0.0


def bake_physics_material(usd_path: str, friction: float, restitution: float) -> int:
    """Bind a ``UsdPhysics.MaterialAPI`` material (static / dynamic friction ``friction``) to every collision prim of the
    asset, in the USD file itself so the sampler's world and the demo's scene both get it. Returns the number of prims bound."""
    import omnigibson.lazy as lazy

    Usd, UsdShade, UsdPhysics, Sdf = lazy.pxr.Usd, lazy.pxr.UsdShade, lazy.pxr.UsdPhysics, lazy.pxr.Sdf
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    looks = stage.DefinePrim(root.GetPath().AppendChild("Looks"), "Scope")
    mat = UsdShade.Material.Define(stage, looks.GetPath().AppendChild("board_physics_mat"))
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(friction))
    api.CreateDynamicFrictionAttr().Set(float(friction))
    api.CreateRestitutionAttr().Set(float(restitution))
    # PhysX combines the two touching materials' frictions by AVERAGE by default, so a slippery board against the cubes'
    # default material would still be sticky; "min" makes the board's friction the effective one (the cubes are untouched)
    px = lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set("min")
    px.CreateRestitutionCombineModeAttr().Set("min")
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
            n += 1
    stage.GetRootLayer().Save()
    return n


def main() -> None:
    manifest = json.loads((HERE / "model" / "isgboard" / "build_manifest.json").read_text())
    category, model = manifest["category"], manifest["model"]
    source_urdf = HERE / "model" / model / f"{model}.urdf"
    dataset_root = Path(get_dataset_path(DATASET_NAME)).resolve()
    model_dir = dataset_root / "objects" / category / model
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    record_obj_metadata_from_urdf(urdf_path=str(source_urdf), obj_dir=str(model_dir), joint_setting="zero", overwrite=True)
    (model_dir / "misc" / "isg_board.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "urdf"}, indent=2) + "\n")

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
        n_bound = bake_physics_material(usd_path, BOARD_FRICTION, BOARD_RESTITUTION)
        print(f"physics material friction {BOARD_FRICTION} bound to {n_bound} collision prims", flush=True)
        report = {
            "physics_material": {"friction": BOARD_FRICTION, "restitution": BOARD_RESTITUTION, "collision_prims": n_bound},
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
        # the pre-converted copy the pods install (setup.sh ISG_ASSETS_SRC)
        dst = REPO_DATASET / "objects" / category / model
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(model_dir, dst)
        print(f"BLOCK_BOARD_CONVERSION_PASSED usd={usd_path} repo_copy={dst}", flush=True)
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
