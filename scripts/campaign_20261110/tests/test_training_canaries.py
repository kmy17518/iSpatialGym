from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import time
import pytest
from scripts.campaign_20261110 import contracts as c, training_canaries as canary, controller_admission as admission
from scripts.campaign_20261110.tests.test_controller_admission import anchor_fixture
from scripts.campaign_20261110.tests.test_group_lane import reservations


def setup(root,monkeypatch,task='inter_object_relation_satisfaction-ontop_small_scale'):
    f,template,config_path=anchor_fixture(root,monkeypatch,split='train',task=task)
    config=json.loads(config_path.read_text());config['campaign_id']='campaign_20261110'
    config['perturbations']['scales']['apple.n.01_1'].update(applied=[1.1]*3,mode='uniform')
    perturb=c.Perturbations({k:c.ScaleState(**v) for k,v in config['perturbations']['scales'].items()})
    config['validation_key']['scale_sha256']=perturb.scale_fingerprint;c.write_json_atomic(config_path,config)
    # Give the real evidence fixture normal campaign index identities.
    for path in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if path.is_file() and path.suffix in ('.json','.jsonl'):
            text=path.read_text().replace('fresh_0','canary__50000').replace('fresh_1','canary__50001')
            path.write_text(text)
        if 'fresh_' in path.name:path.rename(path.with_name(path.name.replace('fresh_0','canary__50000').replace('fresh_1','canary__50001')))
    for path in f.job.rglob('*-tro_state.json'):
        raw=json.loads(path.read_text());block=raw['ispatialgym']['campaign']
        block.update(campaign_id='campaign_20261110',perturbations=config['perturbations'],validation_key=config['validation_key'])
        block['runtime']['observed_scales']={'apple.n.01_1':[1.1]*3};block['runtime']['augmentation']={'moved_objects':{},'lighting':{}}
        raw['ispatialgym']['scene']={'model':template.scene,'task_rooms':[template.room]}
        c.write_json_atomic(path,raw);iid=raw['ispatialgym']['instance_id']
        c.write_json_atomic(path.with_name(iid+'.oracle.json'),{'tro_sha256':c.file_digest(path)})
    for path in root.rglob('meta.json'):
        value=json.loads(path.read_text());value['loader']['campaign_restoration'].update(
            observed_scales={'apple.n.01_1':[1.1]*3},scene_augmentation=False,lighting_augmentation=False)
        c.write_json_atomic(path,value)
    gate=json.loads(f.gate.read_text());gate['tro_sha256']=c.file_digest(gate['instance']);c.write_json_atomic(f.gate,gate)
    with (f.job/'job.env').open('a') as stream:stream.write('START=50000\nN=2\n')
    auth=root/'authorization.json';proof={'purpose':'training_pipeline_canary','config_sha256':c.file_digest(config_path),'start':50000,'count':2}
    c.write_json_atomic(auth,proof);os.utime(auth,(time.time()-50,time.time()-50))
    entry={**proof,'max_instances':2,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=6)).isoformat(),
        'authorization_path':str(auth),'authorization_sha256':c.file_digest(auth)}
    c.write_json_atomic(root/'controllers/training_production_gate.json',{'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','mode':'blocked','allowed_canaries':[entry]})
    reserved=reservations();hashes={'schema':'campaign-reserved-asset-hashes-v1','reserved_model_keys':[],
        'ready_assets':{},'reserved_asset_hashes':[],'pending_unready_model_keys':[]}
    active=({'policy_sha256':'a'*64},reserved,hashes)
    monkeypatch.setattr(admission,'active_policy',lambda root:active)
    monkeypatch.setattr(canary,'active_policy',lambda root:active)
    return f,config_path


def test_actual_artifact_export_preserves_scales_without_acceptance_or_quota(tmp_path,monkeypatch):
    f,_=setup(tmp_path,monkeypatch)
    report=canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    assert report['validation_passed'] and not report['accepted'] and not report['eligible_for_benchmark']
    assert report['quota_consumed']==0
    assert report['scale_provenance']['apple.n.01_1']['default']==[1.,1.,1.]
    assert report['scale_provenance']['apple.n.01_1']['applied']==[1.1]*3
    assert report['export_record']['perturbations']['scales']['apple.n.01_1']['default']==[1.,1.,1.]
    first=canary.publish_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    assert canary.publish_canary(f.job,f.gate,root=tmp_path,started_at=f.since)==first
    ledger=json.loads((tmp_path/'validation/training_canaries/ledger.json').read_text())
    assert ledger['validated_instances']==1 and ledger['eligible_accepted']==ledger['quota_consumed']==0
    assert not (tmp_path/'dataset').exists()
    from scripts.campaign_20261110.training_eligibility import canary_projection
    projection=canary_projection(tmp_path)
    assert projection['validated_instances']==1 and projection['eligible_accepted']==projection['quota_consumed']==0
    assert projection['coverage']['measures']['instances_with_nondefault_scale']==1
    assert 'excluded from benchmark' in projection['coverage']['basis']


@pytest.mark.parametrize('task',['object_scaling','mental_rotation'])
def test_authorized_native_held_canary_does_not_require_generic_queue_adapter(tmp_path,monkeypatch,task):
    f,_=setup(tmp_path,monkeypatch,task=task)
    with pytest.raises(c.ContractError,match='no generic queue adapter'):
        admission.source_proof(f.gate,root=tmp_path,started_at=f.since,campaign_id='campaign_20261110')
    review=canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    assert review['task']==task and review['validation_passed']
    assert not review['accepted'] and review['quota_consumed']==0
    replay=next(f.gate.parent.glob('replay/*/meta.json'))
    value=json.loads(replay.read_text());value['qa']['hard_triggered']=['unsafe']
    c.write_json_atomic(replay,value)
    with pytest.raises(c.ContractError,match='QA'):
        canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)


def test_canary_rejects_changed_authorization_and_missing_restored_scale(tmp_path,monkeypatch):
    f,_=setup(tmp_path,monkeypatch)
    gate=json.loads(f.gate.read_text());report=Path(gate['demo']).parent/'meta.json'
    value=json.loads(report.read_text());value['loader']['campaign_restoration']['observed_scales']['apple.n.01_1']=[1,1,1]
    c.write_json_atomic(report,value)
    with pytest.raises(c.ContractError,match='observed scale restoration'):
        canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    (tmp_path/'authorization.json').write_text('{}')
    with pytest.raises(c.ContractError,match='authorization changed'):
        canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)


def test_canary_cannot_borrow_another_job_replay_or_ignore_hard_qa(tmp_path,monkeypatch):
    f,_=setup(tmp_path,monkeypatch)
    second=tmp_path/'out/other';second.mkdir();(second/'job.env').write_text((f.job/'job.env').read_text())
    with pytest.raises(c.ContractError,match='own exact-job'):
        canary.audit_canary(second,f.gate,root=tmp_path,started_at=f.since)
    replay=next(f.gate.parent.glob('replay/*/meta.json'));value=json.loads(replay.read_text());value['qa']['hard_triggered']=['unsafe']
    c.write_json_atomic(replay,value)
    with pytest.raises(c.ContractError,match='QA'):
        canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)


def test_existing_collector_requires_its_exact_scope_with_active_native_approval(tmp_path,monkeypatch):
    f,_=setup(tmp_path,monkeypatch)
    policy_path=tmp_path/'controllers/training_production_gate.json'
    policy=json.loads(policy_path.read_text())
    entry={**policy['allowed_canaries'][0], 'job_dir':str(f.job),
           'review_manifest':{'path':str(tmp_path/'exact_review.json'),'sha256':'a'*64}}
    policy['allowed_existing_canary_reviews']=[entry]
    c.write_json_atomic(policy_path,policy)
    from scripts.campaign_20261110 import training_existing_canaries as existing
    checked=[]
    def validate(root,job,env,actual,*,kind):
        checked.append((job,kind))
        assert actual['allowed_existing_canary_reviews']==[entry]
        return {'allowed':True}
    monkeypatch.setattr(existing,'check_existing_review',validate)
    report=canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    assert report['authorization']==entry and checked==[(f.job.resolve(),'source_validation')]
    assert report['quota_consumed']==0 and not report['accepted']
    # A malformed exact review must not fall back to the generic sampler entry.
    entry['config_sha256']='b'*64
    c.write_json_atomic(policy_path,policy)
    with pytest.raises(c.ContractError,match='one exact explicit authorization'):
        canary.audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)


def test_reobservation_reaudits_but_does_not_contend_with_unrelated_writer(tmp_path,monkeypatch):
    f,_=setup(tmp_path,monkeypatch)
    first=canary.publish_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    ledger=tmp_path/'validation/training_canaries/ledger.json'
    original=ledger.read_bytes();stat=ledger.stat()
    (ledger.parent/'ledger.lockdir').mkdir()
    assert canary.publish_canary(f.job,f.gate,root=tmp_path,started_at=f.since)==first
    assert ledger.read_bytes()==original and ledger.stat().st_mtime_ns==stat.st_mtime_ns
    replay=next(f.gate.parent.glob('replay/*/meta.json'));value=json.loads(replay.read_text())
    value['qa']['hard_triggered']=['unsafe'];c.write_json_atomic(replay,value)
    with pytest.raises(c.ContractError,match='QA'):
        canary.publish_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
