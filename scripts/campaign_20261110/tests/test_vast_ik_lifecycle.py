import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.infra_vast.worker.ik_lifecycle import ExactIKLifecycle
from scripts.campaign_20261110.infra_vast import ik_transport


def fixture(tmp_path):
    root=tmp_path/'campaign_20261110';root.mkdir();manifest=root/'manifest.json';manifest.write_text('{}')
    job=root/'out'/'group';job.mkdir(parents=True);(job/'job.env').write_text('ISGE_CODE=/immutable/native\nMAX_ATTEMPTS=3\n')
    spec={'id':'900001','job_dir':str(job),'exact_ik_exchange':{'campaign_id':'campaign_20261110','compatibility_id':'a'*64,
          'manifest_path':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'helper_code':str(root/'code'),'seed':None}}
    return root,spec


def test_old_job_has_no_exchange_side_effects(tmp_path):
    calls=[];c=ExactIKLifecycle({'id':'9'},lambda *a,**k:calls.append(a),'s3://private/p','3',local_root=tmp_path/'campaign_20261110')
    assert c.prepare()=={};c.flush();assert not calls and not c.enabled


def test_new_cache_environment_preserves_native_and_sampling_inputs(tmp_path):
    root,spec=fixture(tmp_path);c=ExactIKLifecycle(spec,lambda *a,**k:None,'s3://private/p','3',local_root=root/'local')
    env=c.prepare();text=(Path(spec['job_dir'])/'job.env').read_text()
    assert c.enabled and env['ISPATIALGYM_EXACT_IK_LOCAL_CACHE']==str(root/'local/cache')
    assert 'ISGE_CODE=/immutable/native\nMAX_ATTEMPTS=3\n' in text
    assert (Path(spec['job_dir'])/'job.env.before_exact_ik').read_text()=='ISGE_CODE=/immutable/native\nMAX_ATTEMPTS=3\n'


def test_checksum_failure_falls_back_without_import(tmp_path):
    root,spec=fixture(tmp_path);spec['exact_ik_exchange']['manifest_sha256']='b'*64
    calls=[];c=ExactIKLifecycle(spec,lambda *a,**k:calls.append(a),'s3://private/p','3',local_root=root/'local')
    assert c.prepare()=={} and not c.enabled and not calls


def test_failed_descriptor_upload_does_not_advance_cursor(tmp_path):
    root,spec=fixture(tmp_path)
    def transfer(*a,**k):
        if a[0]=='pipe':raise TimeoutError('network unavailable')
    c=ExactIKLifecycle(spec,transfer,'s3://private/p','3',local_root=root/'local');c.prepare()
    c.call=lambda *a,**k:{'entry_count':1,'branch_count':2,'sha256':'c'*64,'bytes':100,'compatibility_id':'a'*64,'elapsed_seconds':1,'cursor':{'key':'enriched'}}
    c.flush('postjob');assert not(c.job_root/'cursor.json').exists()
    assert c.report['events'][-1]['stage']=='export_failed'


def test_durable_payload_and_descriptor_commit_enriched_cursor(tmp_path):
    root,spec=fixture(tmp_path);calls=[]
    c=ExactIKLifecycle(spec,lambda *a,**k:calls.append((a,k)),'s3://private/p','3',local_root=root/'local');c.prepare()
    c.call=lambda *a,**k:{'entry_count':1,'branch_count':3,'sha256':'c'*64,'bytes':100,'compatibility_id':'a'*64,'elapsed_seconds':1,'cursor':{'existing_key':'new_branch_digest'}}
    c.flush('normal_drain');assert json.loads((c.job_root/'cursor.json').read_text())=={'existing_key':'new_branch_digest'}
    assert calls[0][0][0]=='cp' and calls[1][0][0]=='pipe'
    descriptor=json.loads(calls[1][1]['inp']);assert descriptor['branch_count']==3 and descriptor['reason']=='normal_drain'


def test_no_optin_does_not_read_or_upload_cache(tmp_path):
    assert ik_transport.prepare(tmp_path,tmp_path,{},None,None,None,tmp_path) is None


def test_foreign_job_input_rejected_before_remote_access(tmp_path):
    root=tmp_path/'campaign_20261110';root.mkdir()
    with pytest.raises(ValueError,match='foreign'):
        ik_transport.prepare(root,tmp_path/'previous_campaign',{'ISPATIALGYM_EXACT_IK_MANIFEST':str(root/'m')},None,None,None,root)


def test_optin_never_assigns_to_old_worker_but_normal_work_continues():
    from scripts.campaign_20261110.infra_vast.campaign_inputs import assignment_plan,uncovered_jobs
    jobs=[{'id':'1','time_s':100,'exact_ik_exchange_required':True},{'id':'2','time_s':100}]
    live={'3':{'gpu_name':'RTX 4090'}};hb={'3':{'phase':'ready','slots':1}}
    plan,remaining=assignment_plan(jobs,live,hb,{}, {},10)
    assert [x[2]['id']for x in plan]==['2'] and remaining[0]['id']=='1'
    assert uncovered_jobs([jobs[0]],live,hb,{}, {},10)==[jobs[0]]
    hb['3']['exact_ik_exchange_version']=1
    plan,_=assignment_plan([jobs[0]],live,hb,{}, {},10)
    assert plan[0][2]['id']=='1'


def test_staging_verifies_manifest_in_native_interpreter_without_rewriting(tmp_path,monkeypatch):
    root,spec=fixture(tmp_path);job=Path(spec['job_dir']);manifest=Path(spec['exact_ik_exchange']['manifest_path']);native=tmp_path/'native-python';native.touch();monkeypatch.setattr(ik_transport,'NATIVE_PYTHON',native)
    env={'ISGE_CODE':'/native','ISPATIALGYM_CAMPAIGN_CONFIG':'/config'};seen=[]
    def run(command,**kwargs):
        seen.append((command,json.loads(kwargs['input'])))
        return SimpleNamespace(returncode=0,stdout=json.dumps({'verified':True,'manifest_sha256':ik_transport.sha(manifest),'native_interpreter':str(native)}))
    monkeypatch.setattr(ik_transport.subprocess,'run',run);before=manifest.read_bytes()
    ik_transport.verify_native_manifest(job,manifest,env,root/'helper')
    assert seen[0][0][0]==str(native) and seen[0][1]['environment']==env
    assert manifest.read_bytes()==before
    assert (job/'exact_ik_native_verification.json').exists()


def test_staging_native_mismatch_cannot_become_a_receipt(tmp_path,monkeypatch):
    root,spec=fixture(tmp_path);job=Path(spec['job_dir']);manifest=Path(spec['exact_ik_exchange']['manifest_path']);native=tmp_path/'native-python';native.touch();monkeypatch.setattr(ik_transport,'NATIVE_PYTHON',native)
    monkeypatch.setattr(ik_transport.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='semantic modules differ'))
    with pytest.raises(ValueError,match='semantic modules differ'):
        ik_transport.verify_native_manifest(job,manifest,{'ISGE_CODE':'/native','ISPATIALGYM_CAMPAIGN_CONFIG':'/config'},root/'helper')
    assert not(job/'exact_ik_native_verification.json').exists()
