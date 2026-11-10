#!/usr/bin/env python3
"""Convert newly built campaign URDFs into a new private dataset, then smoke-load.

Never deletes or replaces model directories. The dataset output must be a new
directory beneath --campaign-root. Run in the established OmniGibson environment.
Passing --models permits independent model batches in distinct dataset roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-index", type=Path, required=True)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    campaign = args.campaign_root.resolve()
    dataset = args.dataset_root.resolve()
    if not dataset.is_relative_to(campaign) or dataset == campaign:
        raise ValueError("dataset output must be inside the private campaign root")
    if args.dataset_root.exists() or args.dataset_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing dataset {dataset}")
    index_path = args.build_index.resolve()
    index = json.loads(index_path.read_text())
    wanted = set(filter(None, args.models.split(",")))
    rows = [r for r in index["models"] if not wanted or r["model"] in wanted]
    if not rows or wanted - {r["model"] for r in rows}:
        raise ValueError("unknown model or empty selection")
    # Relative location remains portable after staging the whole build directory.
    for row in rows:
        source = index_path.parent / row["model"] / "model" / row["model"]
        for rel, digest in row["files"].items():
            path = source / rel
            if not path.is_relative_to(source) or sha256(path) != digest:
                raise ValueError(f"source hash mismatch {path}")
        if row.get("assembly_file"):
            assembly = index_path.parent / row["assembly_file"]
            if sha256(assembly) != row["assembly_sha256"]:
                raise ValueError(f"assembly hash mismatch {assembly}")
        row["source"] = source
    dataset.mkdir(parents=True)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    import omnigibson as og
    from omnigibson.utils.asset_conversion_utils import convert_urdf_to_usd, import_obj_metadata, record_obj_metadata_from_urdf
    import omnigibson.lazy as lazy

    reports = []
    # Conversion launches the simulator before Environment is constructed.
    # Pin all clocks here as well; Environment rejects a prelaunched 60 Hz sim.
    og.launch(physics_dt=1.0/120, rendering_dt=1.0/30, sim_step_dt=1.0/30)
    try:
        for row in rows:
            category, model = row["category"], row["model"]
            source = row["source"]
            manifest = json.loads((source / "build_manifest.json").read_text())
            urdf = source / f"{model}.urdf"
            dest = dataset / "objects" / category / model
            dest.mkdir(parents=True)
            record_obj_metadata_from_urdf(urdf_path=str(urdf), obj_dir=str(dest), joint_setting="zero", overwrite=False)
            if row["family"] in {"kohs_face_permutation", "wechsler_red_white"}:
                (dest / "misc/isg_block_faces.json").write_text(json.dumps({k: manifest[k] for k in
                    ("faces", "opposite_pairs", "tile_up_axis", "mass_kg", "edge_m")}, indent=2) + "\n")
            elif row["family"] == "block_board":
                (dest / "misc/isg_board.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "urdf"}, indent=2) + "\n")
            elif row["family"] == "mental_rotation":
                (dest / "misc/isg_mental_rotation_block.json").write_text(json.dumps({k:v for k,v in manifest.items() if k!="urdf"},indent=2)+"\n")
            elif row["family"] == "embedded_figure":
                assembly = json.loads((index_path.parent / row["assembly_file"]).read_text())
                (dest / "misc/isg_embedded_figure.json").write_text(json.dumps({"model":{k:v for k,v in manifest.items() if k!="urdf"},"assembly":assembly},indent=2)+"\n")
            processed, usd = convert_urdf_to_usd(urdf_path=str(urdf), obj_category=category, obj_model=model,
                dataset_root=str(dataset), use_omni_convex_decomp=False, use_usda=False,
                merge_fixed_joints=False, import_inertia_tensor=True)
            import_obj_metadata(usd_path=usd, obj_category=category, obj_model=model, dataset_root=str(dataset))
            stage = lazy.pxr.Usd.Stage.Open(str(usd))
            collisions = [p for p in stage.Traverse() if p.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)]
            if not collisions:
                raise ValueError(f"no collision prims for {model}")
            if row["family"] == "block_board":
                UsdShade, UsdPhysics, PhysxSchema = lazy.pxr.UsdShade, lazy.pxr.UsdPhysics, lazy.pxr.PhysxSchema
                looks = stage.DefinePrim(stage.GetDefaultPrim().GetPath().AppendChild("Looks"), "Scope")
                material = UsdShade.Material.Define(stage, looks.GetPath().AppendChild("board_physics_mat"))
                api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                api.CreateStaticFrictionAttr().Set(0.25)
                api.CreateDynamicFrictionAttr().Set(0.25)
                api.CreateRestitutionAttr().Set(0.0)
                physics = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
                physics.CreateFrictionCombineModeAttr().Set("min")
                physics.CreateRestitutionCombineModeAttr().Set("min")
                for prim in collisions:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
                stage.GetRootLayer().Save()
            report = {"category": category, "model": model, "family": row["family"],
                      "split_reservation": row["split_reservation"], "conversion": "passed",
                      "collision_prims": len(collisions), "usd": str(usd), "asset_sha256": sha256(usd),
                      "source_urdf_sha256": sha256(urdf), "build_manifest_sha256": row["build_manifest_sha256"],
                      "processed_urdf": str(processed), "ready": False}
            (dest / "campaign_conversion.json").write_text(json.dumps(report, indent=2) + "\n")
            reports.append(report)
        # Use the actual DatasetObject path with an explicit absolute private
        # dataset name; get_dataset_path preserves absolute dataset roots.
        import torch as th
        from omnigibson.objects import DatasetObject
        # URDFParseAndImportFile opens a stage with its own physics clock.
        # Reset after the final import, so its 60 Hz stage cannot leak into the
        # independent 120 Hz physical smoke scene.
        og.clear(physics_dt=1.0/120, rendering_dt=1.0/30, sim_step_dt=1.0/30)
        env = og.Environment(configs={"env": {"action_frequency": 30, "physics_frequency": 120, "rendering_frequency": 30},
                                      "scene": {"type": "Scene"}, "robots": []})
        objects = []
        og.sim.stop()
        for i, report in enumerate(reports):
            obj = DatasetObject(name="campaign_" + report["model"], category=report["category"],
                                model=report["model"], dataset_name=str(dataset), scale=[1.0, 1.0, 1.0])
            env.scene.add_object(obj)
            objects.append(obj)
        og.sim.play()
        for i, obj in enumerate(objects):
            obj.set_position_orientation(th.tensor([(i % 5) * 0.3, (i // 5) * 0.3, 0.06]), th.tensor([0., 0., 0., 1.]))
            obj.keep_still()
            obj.wake()
        for _ in range(90):
            with og.sim.render_on_step(True):
                og.sim.step()
        for report, obj in zip(reports, objects):
            lo, hi = obj.aabb
            position, orientation = obj.get_position_orientation()
            if not th.isfinite(lo).all() or not th.isfinite(hi).all() or not th.all(hi > lo):
                raise ValueError(f"invalid simulator bounds for {report['model']}")
            if float(lo[2]) < -0.005 or float(lo[2]) > 0.015:
                raise ValueError(f"object did not settle on the ground: {report['model']} {lo.tolist()}")
            if float(th.linalg.norm(obj.get_linear_velocity())) > 0.05:
                raise ValueError(f"object still moving after settle: {report['model']}")
            report["smoke"] = {"status": "passed", "steps": 90, "aabb_lower": lo.tolist(),
                               "aabb_upper": hi.tolist(), "position": position.tolist(), "orientation": orientation.tolist()}
            report["ready"] = True
            report["readiness_source"] = "project:exact converted USD SHA256 + fresh DatasetObject collision/bounds/settling smoke"
            report["task_validation"] = "pending grasps, geometry, successful demo and open-loop replay"
        # Save native RGB evidence for visual review of material assignment and
        # geometry. Physics readiness remains distinct from task acceptance.
        import numpy as np
        from PIL import Image
        from omnigibson.utils.transform_utils import mat2quat
        camera = og.sim.viewer_camera
        if camera is None:
            raise ValueError("native viewer camera unavailable for visual smoke evidence")
        centre = np.array([0.6, ((len(objects)-1)//5)*0.15, 0.03])
        span = max(1.5, ((len(objects)+4)//5)*0.3)
        images = []
        for name, offset in (("top", [0.,0.,span]), ("oblique", [span*.5,-span*.7,span])):
            eye = centre + np.array(offset)
            back = (eye-centre)/np.linalg.norm(eye-centre)
            right = np.cross([0.,1.,0.], back)
            right /= np.linalg.norm(right)
            up = np.cross(back,right)
            rotation = np.column_stack([right,up,back])
            camera.set_position_orientation(th.tensor(eye,dtype=th.float32),mat2quat(th.tensor(rotation,dtype=th.float32)))
            for _ in range(6):
                og.sim.render()
            obs, _ = camera.get_obs()
            rgb = obs["rgb"].cpu().numpy()[...,:3]
            if not np.isfinite(rgb).all() or float(rgb.std()) < 2.:
                raise ValueError("native visual smoke image is empty or invalid")
            path = dataset / f"smoke_{name}.png"
            Image.fromarray(rgb.astype(np.uint8)).save(path)
            images.append({"path":str(path),"sha256":sha256(path),"view":name})
        (dataset / "campaign_readiness.json").write_text(json.dumps({"schema": "isg-project-readiness-v1",
            "build_index_sha256": sha256(index_path), "objects": reports, "visual_smoke_images":images,
            "visual_review":"pending image inspection; task acceptance remains blocked"}, indent=2, sort_keys=True) + "\n")
        print(f"CAMPAIGN_CUSTOM_READY models={len(reports)} dataset={dataset}", flush=True)
    except BaseException as exc:
        failure = {"status":"failed","exception":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc(),"completed_conversion_reports":reports}
        (dataset / "campaign_conversion_failure.json").write_text(json.dumps(failure,indent=2,sort_keys=True)+"\n")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        # OmniGibson shutdown can exit the process; the durable success/failure
        # record above is authoritative, never shell return code alone.
        og.shutdown()


if __name__ == "__main__":
    main()
