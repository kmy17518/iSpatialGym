from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import contracts as c, camera_context as context


def fixture(root):
    area=root/"proofs";area.mkdir()
    asset_root=root/"assets"
    rows=[];ready={};init={}
    for name,category,model in [("table","pedestal_table","table1"),("chair","armchair","chair1"),("footstool","footstool","foot1")]:
        asset=asset_root/"objects"/category/model/"usd"/(model+".encrypted.usd")
        asset.parent.mkdir(parents=True);asset.write_bytes(name.encode())
        sha=c.file_digest(asset)
        row={"name":name,"model_key":category+"/"+model,"asset_sha256":sha,"default_scale":[1,1,1]}
        rows.append(row);ready[row["model_key"]]={"ready":True,"asset_sha256":sha,"provenance":"native-ready-audit"}
        init[name]={"args":{"category":category,"model":model,"scale":[1,1,1]}}
    source=area/"table_scene.json"
    c.write_json_atomic(source,{"objects_info":{"init_info":{"table":init["table"]}},"metadata":{"ispatialgym":{"campaign_added_support":rows[0]}}})
    support=area/"support_probe.json"
    c.write_json_atomic(support,{**rows[0],"object_name":"table","scene":"office_cubicles_right","room":"conference_hall_0",
        "status":"private_support_geometry_and_reload_passed","original_object_values_unchanged":True,
        "native_scene_path":str(source),"native_scene_sha256":c.file_digest(source)})
    scene=area/"full_scene.json"
    c.write_json_atomic(scene,{"objects_info":{"init_info":init},"metadata":{"ispatialgym":{
        "campaign_added_support":rows[0],"campaign_added_context_objects":rows[1:]}}})
    image=area/"view.png";image.write_bytes(b"native-render-image")
    images=[{"path":str(image),"sha256":c.file_digest(image)}]
    geometry=area/"context_probe.json"
    c.write_json_atomic(geometry,{"status":"private_context_geometry_and_reload_passed","original_object_values_unchanged":True,
        "scene":"office_cubicles_right","room":"conference_hall_0","native_scene_path":str(scene),"native_scene_sha256":c.file_digest(scene),
        "source_scene":str(source),"source_scene_sha256":c.file_digest(source),"objects":rows[1:],"images":images})
    visual=area/"visual.json"
    c.write_json_atomic(visual,{"status":"passed","context_probe_sha256":c.file_digest(geometry),"reviewer":"direct native image review","images":images})
    floor=c.ObjectRef("floor.n.01_1","support","floors","floor",c.digest("floor"),semantic_kind="floor",ready=True,readiness_source="native")
    template=c.Template("camera-category","camera_relocalization-standard","camera_relocalization-standard","eval_category",
        "office_cubicles_right","conference_hall_0",(floor,),c.digest("definition"))
    args=dict(scene_path=scene,geometry_proof=geometry,support_geometry_proof=support,visual_proof=visual,
              ready_assets=ready,asset_root=asset_root,root=root)
    return template,args


def test_added_landmarks_have_full_native_visual_readiness_provenance_without_new_bddl_bindings(tmp_path):
    template,args=fixture(tmp_path)
    bound=context.bind_camera_context(template,**args)
    assert bound.objects == template.objects
    assert bound.fingerprint != template.fingerprint
    refs=context.load_camera_context(bound,root=tmp_path)
    assert {obj.instance for obj in refs} == {"table","chair","footstool"}
    assert all(obj.ready and obj.role == "support" for obj in refs)


def test_context_requires_ready_bytes_and_bound_visual_review(tmp_path):
    template,args=fixture(tmp_path)
    args["ready_assets"]["armchair/chair1"]["ready"]=False
    with pytest.raises(c.ContractError,match="ready admission"):
        context.bind_camera_context(template,**args)
    args["ready_assets"]["armchair/chair1"]["ready"]=True
    visual=context._read(args["visual_proof"]);visual["context_probe_sha256"]=c.digest("another proof")
    c.write_json_atomic(args["visual_proof"],visual)
    with pytest.raises(c.ContractError,match="visual review"):
        context.bind_camera_context(template,**args)


@pytest.mark.parametrize("which",["scene_path","geometry_proof","support_geometry_proof","visual_proof"])
def test_proof_files_are_immutable(tmp_path,which):
    template,args=fixture(tmp_path);bound=context.bind_camera_context(template,**args)
    path=args[which];path.write_text(path.read_text()+"\n")
    with pytest.raises(c.ContractError,match="source bytes changed"):
        context.load_camera_context(bound,root=tmp_path)


def test_context_cannot_migrate_to_training_or_another_room(tmp_path):
    template,args=fixture(tmp_path);bound=context.bind_camera_context(template,**args)
    with pytest.raises(c.ContractError,match="category holdout"):
        context.load_camera_context(replace(bound,split="train"))
    with pytest.raises(c.ContractError,match="scene/room"):
        context.load_camera_context(replace(bound,room="seen_room"))


def test_runtime_cannot_drop_added_landmarks_or_change_their_scale(tmp_path):
    template,args=fixture(tmp_path);bound=context.bind_camera_context(template,**args)
    objects={obj.instance:SimpleNamespace(category=obj.category,model=obj.model,scale=list(obj.default_scale))
             for obj in context.load_camera_context(bound)}
    scene=SimpleNamespace(object_registry=lambda kind,name:objects.get(name))
    assert set(context.verify_live_context(bound,scene)) == set(objects)
    objects["chair"].scale=[1.1,1.1,1.1]
    with pytest.raises(c.ContractError,match="scale differs at runtime"):
        context.verify_live_context(bound,scene)
    objects.pop("chair")
    with pytest.raises(c.ContractError,match="missing at runtime"):
        context.verify_live_context(bound,scene)


def test_explicit_added_landmarks_participate_in_global_holdout_and_physical_identity(tmp_path):
    from scripts.campaign_20261110.binding_pools import audit_objects
    from scripts.campaign_20261110.group_lane import check_reservations, physical_template
    template,args=fixture(tmp_path);bound=context.bind_camera_context(template,**args)
    assert len(audit_objects(bound)) == 4
    assert physical_template(bound) != physical_template(template)
    keys=list(args["ready_assets"])
    reserved={"model_holdout_models":[],"category_holdout_models":keys,
        "category_holdout_categories":[key.split('/')[0] for key in keys],
        "category_holdout_rooms":[{"scene":template.scene,"room":template.room}],
        "train_exclude_models":keys,"pending_custom_hard_reservations_not_ready":[]}
    hashes={"schema":"campaign-reserved-asset-hashes-v1","reserved_model_keys":keys,
        "ready_assets":args["ready_assets"],"pending_unready_model_keys":[],
        "reserved_asset_hashes":[row["asset_sha256"] for row in args["ready_assets"].values()]}
    check_reservations(bound,reserved,hashes)
    reserved["category_holdout_models"]=[key for key in keys if key != "armchair/chair1"]
    with pytest.raises(c.ContractError,match="unreserved model in category holdout"):
        check_reservations(bound,reserved,hashes)


def test_actual_runtime_replay_verifies_context_without_campaign_environment(tmp_path, monkeypatch):
    from dataclasses import asdict
    from ispatialgym.sampling import campaign_runtime as runtime
    template,args=fixture(tmp_path);bound=context.bind_camera_context(template,**args)
    scales={obj.instance:c.ScaleState(obj.default_scale,obj.default_scale,"fixed") for obj in bound.objects}
    perturbations=c.Perturbations(scales)
    key=c.ValidationKey(bound.fingerprint,c.digest("code"),c.digest("physics"),c.digest("data"),perturbations.scale_fingerprint)
    config={"template":asdict(bound),"perturbations":asdict(perturbations),"validation_key":asdict(key),"split":"eval_category"}
    path=tmp_path/"config.json";c.write_json_atomic(path,config)
    monkeypatch.setenv("ISGF_ROOT",str(tmp_path))
    assert runtime._load(str(path))[1].fingerprint == bound.fingerprint
    tro=c.attach_tro_metadata({"ispatialgym":{}},campaign_id="fresh",split="eval_category",template=bound,
        created_at=1,seed=1,perturbations=perturbations,validation_key=key)
    meta=tro["ispatialgym"]["campaign"]
    landmarks={obj.instance:SimpleNamespace(category=obj.category,model=obj.model,scale=list(obj.default_scale))
        for obj in context.load_camera_context(bound)}
    scene=SimpleNamespace(object_registry=lambda kind,name:landmarks.get(name))
    physical={obj.instance:SimpleNamespace(scale=list(obj.default_scale)) for obj in bound.objects}
    with pytest.raises(c.ContractError,match="replay evidence missing"):
        runtime.verify_loaded(meta,physical,scene=scene)
    meta["runtime"]={"camera_context":{"scene":bound.scene,"room":bound.room,
        "scene_sha256":bound.scene_setup["camera_context"]["scene"]["sha256"],
        "observed_objects":context.verify_live_context(bound,scene)}}
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG",raising=False)
    assert set(runtime.verify_loaded(meta,physical,scene=scene)["camera_context"]) == set(landmarks)
    landmarks["chair"].model="unapproved"
    with pytest.raises(c.ContractError,match="missing at runtime"):
        runtime.verify_loaded(meta,physical,scene=scene)


def test_separate_rerender_can_replace_bad_framing_only_for_exact_proven_scene(tmp_path):
    template,args=fixture(tmp_path)
    geometry_sha=c.file_digest(args['geometry_proof'])
    image=tmp_path/'proofs/inside_room.png';image.write_bytes(b'exact-private-scene-new-camera')
    images=[{'path':str(image),'sha256':c.file_digest(image)}]
    rendered=tmp_path/'proofs/render_report.json'
    report={'status':'exact_private_scene_rendered','context_probe_sha256':geometry_sha,
        'native_scene_sha256':c.file_digest(args['scene_path']),'native_scene_path':str(args['scene_path']),'images':images}
    c.write_json_atomic(rendered,report)
    visual=context._read(args['visual_proof'])
    visual.update(images=images,render_proof={'path':str(rendered),'sha256':c.file_digest(rendered)})
    c.write_json_atomic(args['visual_proof'],visual)
    bound=context.bind_camera_context(template,**args)
    assert len(context.load_camera_context(bound,root=tmp_path))==3
    assert c.file_digest(args['geometry_proof'])==geometry_sha
    report['native_scene_sha256']=c.digest('changed-scene');c.write_json_atomic(rendered,report)
    visual['render_proof']['sha256']=c.file_digest(rendered);c.write_json_atomic(args['visual_proof'],visual)
    with pytest.raises(c.ContractError,match='another private scene'):
        context.bind_camera_context(template,**args)
