import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.production_gate import check_job,check_environment,read_environment
from scripts.campaign_20261110.training_canaries import audit_canary
from scripts.campaign_20261110.tests.test_training_canaries import setup


def prepared(root,monkeypatch):
    f,config=setup(root,monkeypatch)
    ref=lambda p:{'path':str(p),'sha256':c.file_digest(p)}
    script=root/'code/snapshot/scripts/trainfast/job_recollect.sh';script.parent.mkdir(parents=True);script.write_text('# collector-only fixture\n')
    receipt=f.job/'recollect.json';receipt.write_text('{}');(f.job/'sampling.done').touch()
    files=[]
    for p in (f.job/'inst').rglob('*-tro_state.json'):
        source=root/'original_native'/p.name;source.parent.mkdir(exist_ok=True);source.write_bytes(p.read_bytes())
        iid=json.loads(p.read_text())['ispatialgym']['instance_id'];side=p.with_name(iid+'.oracle.json')
        ss=source.with_name(side.name);ss.write_bytes(side.read_bytes())
        files.append({'source_tro':ref(source),'copied_tro':ref(p),'source_sidecar':ref(ss),'copied_sidecar':ref(side)})
    manifest=root/'review_manifest.json';c.write_json_atomic(manifest,{
        'schema':'isg-existing-training-canary-review-v1','campaign_id':'campaign_20261110','job_dir':str(f.job),
        'new_native_instances':0,'configuration':ref(config),'job_environment':ref(f.job/'job.env'),
        'recollect_receipt':ref(receipt),'submission_script':ref(script),'native_runtime':str(root/'code/snapshot'),
        'native_files':files})
    policy_path=root/'controllers/training_production_gate.json';policy=json.loads(policy_path.read_text())
    entry=policy['allowed_canaries'][0];policy['allowed_canaries']=[]
    policy['allowed_existing_canary_reviews']=[{**entry,'job_dir':str(f.job),'review_manifest':ref(manifest)}]
    c.write_json_atomic(policy_path,policy)
    return f,script,manifest


def test_only_exact_existing_collector_and_own_replay_are_authorized(tmp_path,monkeypatch):
    f,script,manifest=prepared(tmp_path,monkeypatch)
    assert check_job(tmp_path,f.job,submission_script=script)['allowed']
    assert not check_job(tmp_path,f.job)['allowed']
    assert not check_job(tmp_path,f.job,submission_script=script.with_name('job_fast.sh'))['allowed']
    env=read_environment(f.job/'job.env')
    assert not check_environment(tmp_path,env,start=50000,count=2)['allowed']
    replay=tmp_path/'replay';replay.mkdir();(replay/'replay_inputs.json').write_text(json.dumps({'source_job':str(f.job)}))
    assert check_job(tmp_path,replay,kind='physics_replay')['allowed']
    report=audit_canary(f.job,f.gate,root=tmp_path,started_at=f.since)
    assert report['validation_passed'] and not report['accepted'] and report['quota_consumed']==0
    (f.job/'job.env').write_text((f.job/'job.env').read_text()+'UNDECLARED=1\n')
    assert not check_job(tmp_path,f.job,submission_script=script)['allowed']


def test_extra_native_or_changed_original_cannot_join_existing_review(tmp_path,monkeypatch):
    f,script,manifest=prepared(tmp_path,monkeypatch)
    p=next((f.job/'inst').rglob('*-tro_state.json'));extra=p.with_name('extra-tro_state.json');extra.write_bytes(p.read_bytes())
    assert not check_job(tmp_path,f.job,submission_script=script)['allowed']
    extra.unlink()
    source=Path(json.loads(manifest.read_text())['native_files'][0]['source_tro']['path']);source.write_text('{}')
    assert not check_job(tmp_path,f.job,submission_script=script)['allowed']
