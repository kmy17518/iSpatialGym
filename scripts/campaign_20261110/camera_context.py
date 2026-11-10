"""Hash-bound private camera landmarks, separate from active BDDL bindings."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import ObjectRef, file_digest, require

SCHEMA = "isg-camera-context-v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _checked(reference, root=None):
    path = Path(reference["path"])
    require(path.is_absolute(), "camera context source must be absolute")
    if root is not None:
        require(path.resolve().is_relative_to(Path(root).resolve()), "camera context source outside campaign")
    require(file_digest(path) == reference["sha256"], "camera context source bytes changed")
    return path


def _same_scale(a, b):
    return len(a) == len(b) == 3 and all(abs(float(x)-float(y)) <= 1e-6 for x,y in zip(a,b))


def load_camera_context(template, *, root=None):
    context = template.scene_setup.get("camera_context")
    if context is None:
        return ()
    require(context.get("schema") == SCHEMA and template.task.startswith("camera_relocalization-"),
            "invalid camera context contract")
    require(template.split == "eval_category", "private camera context is reserved for category holdout")
    scene_path = _checked(context["scene"], root)
    scene = _read(scene_path)
    proof_path = _checked(context["geometry_proof"], root)
    proof = _read(proof_path)
    support_path = _checked(context["support_geometry_proof"], root)
    support = _read(support_path)
    visual = _read(_checked(context["visual_proof"], root))
    require(proof.get("status") == "private_context_geometry_and_reload_passed"
            and proof.get("original_object_values_unchanged") is True, "camera context geometry has not passed")
    require(support.get("status") == "private_support_geometry_and_reload_passed"
            and support.get("original_object_values_unchanged") is True, "camera context support geometry has not passed")
    require((proof.get("scene"),proof.get("room")) == (template.scene,template.room)
            == (support.get("scene"),support.get("room")), "camera context scene/room differs")
    require(proof["native_scene_sha256"] == context["scene"]["sha256"]
            and Path(proof["native_scene_path"]).resolve() == scene_path.resolve(), "camera context scene differs from geometry proof")
    source = _checked({"path": proof["source_scene"], "sha256": proof["source_scene_sha256"]}, root)
    require(support["native_scene_sha256"] == proof["source_scene_sha256"]
            and Path(support["native_scene_path"]).resolve() == source.resolve(), "camera context source lacks support geometry proof")
    require(visual.get("status") == "passed" and visual.get("context_probe_sha256") == file_digest(proof_path)
            and bool(visual.get("reviewer")), "camera context visual review is missing or unbound")
    images = visual.get("images", [])
    require(isinstance(images,list) and bool(images), "camera context visual review has no images")
    measured_images = {(row["path"],row["sha256"]) for row in proof["images"]}
    for image in images:
        require((image["path"],image["sha256"]) in measured_images, "visual review uses an unrelated context image")
        _checked(image, root)
    metadata = scene.get("metadata",{}).get("ispatialgym",{})
    rows = [metadata.get("campaign_added_support",{})] + metadata.get("campaign_added_context_objects",[])
    require(all(row.get("name") for row in rows) and len({row["name"] for row in rows}) == len(rows), "invalid added context identities")
    expected = {row["name"]:row for row in proof["objects"]}
    require(support["object_name"] not in expected, "context support was duplicated")
    expected[support["object_name"]] = {**support,"name":support["object_name"]}
    require(set(expected) == {row["name"] for row in rows}, "private scene has undeclared context objects")
    objects = tuple(ObjectRef(**row) for row in context["objects"])
    require(len(objects) == len(rows) and {obj.instance for obj in objects} == set(expected), "camera context object contract is incomplete")
    require(not {obj.instance for obj in objects} & {obj.instance for obj in template.objects}, "context name collides with active BDDL binding")
    actual = scene["objects_info"]["init_info"]
    by_name = {row["name"]:row for row in rows}
    require(set(context["asset_files"]) == set(expected), "camera context asset audit is incomplete")
    for obj in objects:
        native, meta = expected[obj.instance], by_name[obj.instance]
        require(obj.ready and obj.readiness_source and obj.role == "support" and obj.semantic_kind == "object", "context landmark is not ready")
        require(obj.model_key == meta["model_key"] == native["model_key"]
                and obj.asset_sha256 == meta["asset_sha256"] == native["asset_sha256"], "context model/hash differs from native proof")
        require(_same_scale(obj.default_scale,meta["default_scale"]) and _same_scale(obj.default_scale,native["default_scale"]), "context native default scale changed")
        args = actual[obj.instance]["args"]
        require((args.get("category"),args.get("model")) == (obj.category,obj.model)
                and _same_scale(args.get("scale",[]),obj.default_scale), "private context scene identity changed")
        asset = context["asset_files"][obj.instance]
        require(asset["sha256"] == obj.asset_sha256, "context asset reference differs from contract")
        # Assets may be read-only symlink targets outside campaign root.
        _checked(asset)
    return objects


def bind_camera_context(template, *, scene_path, geometry_proof, support_geometry_proof,
                        visual_proof, ready_assets, asset_root, root):
    """Author only after actual geometry, image review and model readiness pass."""
    scene = _read(scene_path)
    meta = scene.get("metadata",{}).get("ispatialgym",{})
    rows = [meta.get("campaign_added_support",{})] + meta.get("campaign_added_context_objects",[])
    objects, assets = [], {}
    for row in rows:
        key = row["model_key"]
        ready = ready_assets.get(key,{})
        require(ready.get("ready") is True and ready.get("asset_sha256") == row["asset_sha256"], "context model lacks matching ready admission")
        category,model = key.split("/",1)
        asset = Path(asset_root)/"objects"/category/model/"usd"/(model+".encrypted.usd")
        assets[row["name"]] = _reference(asset)
        require(assets[row["name"]]["sha256"] == row["asset_sha256"], "context source USD differs from readiness proof")
        objects.append(ObjectRef(row["name"],"support",category,model,row["asset_sha256"],tuple(row["default_scale"]),
            ready=True,readiness_source=json.dumps({"ready_asset":ready,"geometry_proof":_reference(geometry_proof)},sort_keys=True)))
    context = {"schema":SCHEMA,"scene":_reference(scene_path),"geometry_proof":_reference(geometry_proof),
        "support_geometry_proof":_reference(support_geometry_proof),"visual_proof":_reference(visual_proof),
        "objects":[asdict(obj) for obj in objects],"asset_files":assets}
    bound = replace(template,scene_setup={**template.scene_setup,"camera_context":context})
    load_camera_context(bound,root=root)
    return bound


def verify_live_context(template, scene):
    """Existing native footprint/stability checks still govern physical validity."""
    observed = {}
    for ref in load_camera_context(template):
        obj = scene.object_registry("name",ref.instance)
        require(obj is not None and (obj.category,obj.model) == (ref.category,ref.model), "camera context object missing at runtime")
        actual = obj.scale.tolist() if hasattr(obj.scale,"tolist") else list(obj.scale)
        require(_same_scale(actual,ref.default_scale), "camera context scale differs at runtime")
        observed[ref.instance] = {"category":ref.category,"model":ref.model,"scale":actual}
    return observed
