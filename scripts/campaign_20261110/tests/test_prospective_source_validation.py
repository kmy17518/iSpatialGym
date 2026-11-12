import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import prospective_source_validation as p
from scripts.campaign_20261110.contracts import file_digest, ContractError


def fixture(root, monkeypatch):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': file_digest(path)}
    native=root/'code/native'; stamp=put(native/'.snapshot_commit','frozen')
    script=put(native/'scripts/trainfast/job_recollect.sh','collector')
    job=root/'job'; source=root/'source'; iid='exact__50304500'
    key={'code_sha256':stamp['sha256']}
    cfg=put(root/'config.json',{'purpose':'validation','split':'eval_standard','template':{'split':'eval_standard','class_id':'class'},'validation_key':key})
    env={'ISPATIALGYM_CAMPAIGN_CONFIG':cfg['path'],'START':'50304500','N':'1'}
    envref=put(job/'job.env',env); copy=put(job/'recollect.json',{});(job/'sampling.done').touch()
    request=put(root/'request.json',{'campaign_id':'campaign_20261110','purpose':'canonical_recovery_bounded_canary','benchmark_credit':0,'current100_r1_credit':0,'production_released':False,'config':cfg['path'],'class_id':'class','start_index':50304500,'count':1,'native_runtime':str(native)})
    tro={'ispatialgym':{'instance_id':iid,'campaign':{'validation_key':key,'split':'eval_standard'},'object_bindings':{}}}
    s=put(source/(iid+'-tro_state.json'),tro); t=put(job/'inst'/(iid+'-tro_state.json'),tro)
    artifacts=[s]
    for suffix,data in [('.oracle.json',{'tro_sha256':s['sha256']}),('.summary.json',{})]:
        artifacts.append(put(source/(iid+suffix),data));put(job/'inst'/(iid+suffix),data)
    review=put(root/'review.json',{'request':request,'returncode':0,'native_artifacts_complete':True,'benchmark_credit':0,'production_released':False,'native_runtime':str(native),'artifacts':artifacts})
    prep=put(root/'prepared.json',{'configuration':cfg,'source_request':request,'source_review':review,'runtime':str(native),'submission_script':script,'start':50304500,'job_environment':envref,'copy_proof':copy,'source_environment':envref,'source_tro':s,'copied_tro':t})
    manifest=put(root/'manifest.json',{'schema':'isg-prospective-source-review-v1','campaign_id':'campaign_20261110','job_dir':str(job),'instance_id':iid,'new_native_draws':0,'benchmark_credit':0,'quota_credit':0,'current100_r1_credit':0,'preparation':prep})
    entry={'job_dir':str(job),'instance_id':iid,'manifest':manifest}
    policy={'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110','allowed_prospective_source_reviews':[entry]}
    put(root/'controllers/training_production_gate.json',policy)
    monkeypatch.setattr(p,'validate_campaign_block',lambda *a,**k:None)
    return job,env,policy,script['path'],iid


def test_exact_copy_and_source_replay_only_and_permanent_zero_credit(tmp_path,monkeypatch):
    job,env,policy,script,iid=fixture(tmp_path,monkeypatch)
    assert p.check_review(tmp_path,job,env,policy,kind='collection',submission_script=script)['allowed']
    assert p.check_review(tmp_path,job,env,policy,kind='source_validation')['quota_credit']==0
    for kind,s in [('collection',None),('collection',str(Path(script).with_name('job_fast.sh'))),('native_sampling',script)]:
        with pytest.raises(ContractError,match='collector-only'):p.check_review(tmp_path,job,env,policy,kind=kind,submission_script=s)
    with pytest.raises(ContractError,match='cannot consume'):p.require_no_quota(tmp_path,job,[iid])
    with pytest.raises(ContractError,match='cannot consume'):p.require_no_quota(tmp_path,tmp_path/'different',[iid])


@pytest.mark.parametrize('change',['config','range','runtime','sidecar','extra_tro','manifest','source'])
def test_changed_exact_identity_fails_closed(tmp_path,monkeypatch,change):
    job,env,policy,script,iid=fixture(tmp_path,monkeypatch)
    if change=='range':env['N']='2'
    elif change=='config':env['ISPATIALGYM_CAMPAIGN_CONFIG']=str(tmp_path/'other.json')
    else:
        paths={'runtime':Path(script).parents[2]/'.snapshot_commit','sidecar':job/'inst'/(iid+'.oracle.json'),
          'extra_tro':job/'inst'/'extra-tro_state.json','manifest':tmp_path/'manifest.json','source':tmp_path/'source'/(iid+'-tro_state.json')}
        paths[change].write_text('{}')
    with pytest.raises((ContractError,KeyError)):
        p.check_review(tmp_path,job,env,policy,kind='collection',submission_script=script)
