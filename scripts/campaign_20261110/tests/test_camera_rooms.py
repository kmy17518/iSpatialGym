import time
from dataclasses import replace

import pytest

from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.camera_rooms import bind_camera_room_candidate


def fixture(root):
    code=root/"code/snapshot";directory=root/"materialized_runs/room/template"
    directory.mkdir(parents=True)
    dataset=code/"Behavior-1K/datasets"
    asset=dataset/"behavior-1k-assets/objects/floors/abc/usd/abc.encrypted.usd"
    asset.parent.mkdir(parents=True);asset.write_bytes(b"floor-usd")
    definition=directory/"problem.bddl";definition.write_text("camera-only definition")
    args={"category":"floors","model":"abc","scale":[1.01,1,1],"name":"floor_obj"}
    state={"objects_info":{"init_info":{"floor_obj":{"args":args}}}}
    stable=dataset/"2026-challenge-task-instances/scenes/house_single_floor/json/house_single_floor_stable.json"
    c.write_json_atomic(stable,state);c.write_json_atomic(directory/"template.json",state)
    meta={"n_written":1,"scene":"house_single_floor","task_rooms":["living_room_0"],"template_id":"room",
        "variant":"camera_relocalization-standard","activity_definition":"problem.bddl","bddl_sha256":c.file_digest(definition),
        "template_sha256":c.file_digest(directory/"template.json"),"bindings":[{"name":"floor.n.01_1","scene_object_name":"floor_obj"}]}
    c.write_json_atomic(directory/"template_meta.json",meta)
    obj=c.ObjectRef("floor.n.01_1","support","floors","abc",c.file_digest(asset),semantic_kind="floor",ready=True,readiness_source="test")
    template=c.Template("room-candidate","camera_relocalization-standard","camera_relocalization-standard","train",
        "house_single_floor","living_room_0",(obj,),c.digest("definition"),{"runtime_bddl_sha256":c.file_digest(definition)})
    return template,directory,code,stable


def test_fresh_camera_room_binding_records_actual_untouched_default(tmp_path):
    template,directory,code,stable=fixture(tmp_path)
    result=bind_camera_room_candidate(template,directory,code=code,root=tmp_path,started_at=time.time()-60)
    assert result.objects[0].default_scale==(1.01,1,1)
    assert result.scene_setup["fresh_metadata_binding_check"]=="passed"
    assert result.scene_setup["default_scale_resolution"][0]["source_sha256"]==c.file_digest(stable)
    assert result.fingerprint != template.fingerprint
    with pytest.raises(c.ContractError,match="room differs"):
        bind_camera_room_candidate(replace(template,room="wrong"),directory,code=code,root=tmp_path,started_at=time.time()-60)


def test_unproved_default_and_changed_asset_fail_closed(tmp_path):
    template,directory,code,stable=fixture(tmp_path)
    c.write_json_atomic(stable,{"objects_info":{"init_info":{}}})
    with pytest.raises(c.ContractError,match="untouched stable"):
        bind_camera_room_candidate(template,directory,code=code,root=tmp_path,started_at=time.time()-60)
    template=replace(template,objects=(replace(template.objects[0],default_scale=(1.01,1,1)),))
    asset=code/"Behavior-1K/datasets/behavior-1k-assets/objects/floors/abc/usd/abc.encrypted.usd"
    asset.write_bytes(b"changed")
    with pytest.raises(c.ContractError,match="asset bytes differ"):
        bind_camera_room_candidate(template,directory,code=code,root=tmp_path,started_at=time.time()-60)
