import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import vast_prebuild as v


def fixture(root):
    native=root/'code/native'; helper=root/'code/helper'
    for p in [native/'.snapshot_commit',native/'scripts/grasp_bands/job_glib.sh',
              helper/'scripts/campaign_20261110/vast_prebuild.py',helper/'scripts/campaign_20261110/job_prebuild.sh']:
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text('exact runtime')
    asset=native/'Behavior-1K/datasets/objects/model/usd/model.usd'; asset.parent.mkdir(parents=True); asset.write_text('asset')
    proof=root/'training_preparation/default.json'; v.write(proof,{'measured':True})
    row={'id':'model__s0.9','ready':True,'kind':'tabletop','constraint':'tabletop','scale':[.9]*3,
         'campaign_classes':['disembedding_07'],'asset_path':str(asset),'asset_sha256':v.sha(asset),
         'native_default_evidence':{'path':str(proof),'sha256':v.sha(proof)}}
    manifest=root/'training_preparation/jobs.json'; v.write(manifest,[row])
    request=root/'training_preparation/request.json'
    v.write(request,{'batches':[{'kind':'tabletop','jobs':str(manifest),'jobs_sha256':v.sha(manifest),'stages':'glib'}]})
    return request,native,helper,asset,manifest


def test_exact_recipe_copies_row_and_idempotently_enqueues_without_activation(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    before=manifest.read_bytes()
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    value=v.verify(job,tmp_path)
    copied=json.loads(Path(value['selected_manifest']).read_text())
    copied[0]['native_default_evidence']['path']=value['proof_relocation']['original']
    assert copied==json.loads(before)
    assert manifest.read_bytes()==before and value['new_native_instances']==0
    jid=v.enqueue(job,tmp_path); assert v.enqueue(job,tmp_path)==jid
    assert not (tmp_path/'vast/config.json').exists()
    with pytest.raises(ValueError,match='already has evidence'):v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')


def test_changed_native_asset_fails_before_execution(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    asset.write_text('other model')
    with pytest.raises(ValueError,match='asset changed'):v.verify(job,tmp_path)
    assert not (job/'result').exists()


def test_unproved_floating_and_block_design_do_not_allocate(tmp_path):
    request,native,helper,asset,manifest=fixture(tmp_path)
    with pytest.raises(ValueError,match='GraspGen'):v.prepare(tmp_path,request,'floating',0,native,helper,'one')
    rows=json.loads(manifest.read_text());rows[0]['campaign_classes']=['configuration_matching-block_design_01'];v.write(manifest,rows)
    req=json.loads(request.read_text());req['batches'][0]['jobs_sha256']=v.sha(manifest);v.write(request,req)
    with pytest.raises(ValueError,match='selection remains'):v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    assert not (tmp_path/'out/vast_prebuild/one').exists()


def test_native_zero_exit_without_validation_output_is_failure(tmp_path,monkeypatch):
    request,native,helper,asset,manifest=fixture(tmp_path)
    job=v.prepare(tmp_path,request,'tabletop',0,native,helper,'one')
    class Result:returncode=0
    monkeypatch.setattr(v.subprocess,'run',lambda *a,**k:Result())
    assert v.run(job)==2
    report=json.loads((job/'result/prebuild_result.json').read_text())
    assert not report['native_validation_done'] and report['positive_native_reader_required']
    assert report['new_native_instances']==0
