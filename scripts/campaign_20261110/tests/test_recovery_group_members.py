import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import prepare_recovery_group_members as m
from scripts.campaign_20261110 import prospective_source_validation as p
from scripts.campaign_20261110.contracts import ContractError


def fixture(root):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return m.reference(path)
    native=root/'code/frozen'; stamp=put(native/'.snapshot_commit','immutable')
    put(native/'scripts/trainfast/job_recollect.sh','collector')
    key={'code_sha256':stamp['sha256']}
    cfg=put(root/'config.json',{'purpose':'validation','split':'eval_standard','template':{'class_id':'class','split':'eval_standard','task':'disembedding'},'validation_key':key})
    parent=root/'diagnostics/group/request.json'; review=parent.with_name('native_review.json')
    qr=put(parent,{'schema':'isg-bounded-canonical-group-request-v1','campaign_id':'campaign_20261110','purpose':'canonical_recovery_bounded_canary',
        'class_id':'class','config':cfg['path'],'native_runtime':str(native),'start_index':101,'count':3,'benchmark_credit':0,'quota_credit':0,'current100_r1_credit':0,
        'production_released':False,'inputs':{cfg['path']:cfg['sha256']}})
    parent.with_name('exact.env').write_text('export ISPATIALGYM_CAMPAIGN_CONFIG='+cfg['path']+'\n')
    artifacts=[]; members=[]
    for i in [101,102,103]:
        iid='member__'+str(i); base=parent.parent/'native/variant/template/instances'
        tro=put(base/(iid+'-tro_state.json'),{'ispatialgym':{'instance_id':iid,'object_bindings':{},'campaign':{'validation_key':key,'split':'eval_standard'}}})
        artifacts.extend([tro,put(base/(iid+'.oracle.json'),{'tro_sha256':tro['sha256']}),put(base/(iid+'.summary.json'),{'start':i})]);members.append({'instance_id':iid,'index':i})
    put(review,{'request':qr,'returncode':0,'native_artifacts_complete':True,'native_runtime':str(native),'benchmark_credit':0,'production_released':False,'artifacts':artifacts,'members':members})
    return qr,review


@pytest.mark.parametrize('timeout',[False,True])
@pytest.mark.parametrize('legacy_credit_omitted',[False,True])
def test_member_projection_preserves_parent_bytes_and_passes_existing_singleton_validator(tmp_path,monkeypatch,timeout,legacy_credit_omitted):
    qr,review=fixture(tmp_path); parent_hash=m.reference(review); original=json.loads(review.read_text())
    if legacy_credit_omitted:
        parent=Path(qr['path']);q=json.loads(parent.read_text());q.pop('current100_r1_credit')
        parent.write_text(json.dumps(q));qr=m.reference(parent);original['request']=qr
        review.write_text(json.dumps(original));parent_hash=m.reference(review)
    if timeout:
        original.update(returncode=124,native_artifacts_complete=False,requested_count=3,written_count=2,
                        members=original['members'][:2],artifacts=original['artifacts'][:6])
        review.write_text(json.dumps(original));parent_hash=m.reference(review)
    prep=m.prepare(tmp_path,qr,102); value=json.loads(prep.read_text())
    assert value['native_generation_request']==qr and value['native_generation_review']==parent_hash
    projection=json.loads(Path(value['source_request']['path']).read_text())
    assert projection['count']==1 and projection['start_index']==102 and projection['projection_only']
    assert projection['native_generation_request']==qr and projection['native_runtime']==original['native_runtime']
    assert projection['current100_r1_credit']==0 and m.reference(Path(qr['path']))==qr
    assert m.reference(review)==parent_hash
    projected=json.loads(Path(value['source_review']['path']).read_text())
    assert projected['returncode']==original['returncode']
    assert projected['native_artifacts_complete']==original['native_artifacts_complete']
    assert projected['member_native_artifacts_complete'] is True
    job=Path(value['job_dir']); assert len(list((job/'inst').rglob('*-tro_state.json')))==1
    manifest=prep.with_name('manifest.json'); manifest.write_text(json.dumps({'schema':'isg-prospective-source-review-v1','campaign_id':'campaign_20261110',
        'job_dir':str(job),'instance_id':'member__102','preparation':m.reference(prep),'new_native_draws':0,'benchmark_credit':0,'quota_credit':0,'current100_r1_credit':0}))
    policy={'allowed_prospective_source_reviews':[{'job_dir':str(job),'instance_id':'member__102','manifest':m.reference(manifest)}]}
    from scripts.campaign_20261110.production_gate import read_environment
    monkeypatch.setattr(p,'validate_campaign_block',lambda *a,**k:None)
    result=p.check_review(tmp_path,job,read_environment(job/'job.env'),policy,kind='collection',submission_script=value['submission_script']['path'])
    assert result['allowed'] and result['quota_credit']==result['new_native_draws']==0
    assert m.prepare(tmp_path,qr,102)==prep


@pytest.mark.parametrize('corruption',['missing_summary','missing_member','source_hash','parent_rc','count','identity'])
def test_timeout_does_not_admit_missing_or_changed_native_evidence(tmp_path,corruption):
    qr,review=fixture(tmp_path);v=json.loads(review.read_text())
    v.update(returncode=124,native_artifacts_complete=False,requested_count=3,written_count=2,
             members=v['members'][:2],artifacts=v['artifacts'][:6])
    if corruption=='missing_summary':v['artifacts'].pop()
    if corruption=='missing_member':v['members'].pop()
    if corruption=='source_hash':Path(v['artifacts'][3]['path']).write_text('{}')
    if corruption=='parent_rc':v['returncode']=1
    review.write_text(json.dumps(v))
    if corruption in ['count','identity']:
        m.member_projection(tmp_path,qr,review,102,tmp_path/'projection')
        q=json.loads((tmp_path/'projection/request.json').read_text());r=json.loads((tmp_path/'projection/native_review.json').read_text())
        if corruption=='count':q['count']=2
        else:q['native_runtime']='different'
        with pytest.raises(ContractError):m.validate_timed_out_member(tmp_path,q,r)
    else:
        with pytest.raises(ContractError):m.member_projection(tmp_path,qr,review,102,tmp_path/'projection')


@pytest.mark.parametrize('change',['range','source','runtime','partial'])
def test_member_projection_rejects_unbound_or_incomplete_evidence(tmp_path,change):
    qr,review=fixture(tmp_path);v=json.loads(review.read_text());index=102
    if change=='range':index=104
    elif change=='source':Path(v['artifacts'][3]['path']).write_text('{}')
    elif change=='runtime':v['native_runtime']='another';review.write_text(json.dumps(v))
    else:v['returncode']=124;review.write_text(json.dumps(v))
    with pytest.raises(ContractError):m.member_projection(tmp_path,qr,review,index,tmp_path/'projection')


@pytest.mark.parametrize('key,value',[('current100_r1_credit',1),('quota_credit',1),('benchmark_credit',1),('production_released',True)])
def test_missing_legacy_label_cannot_authorize_credit_or_production(tmp_path,key,value):
    qr,review=fixture(tmp_path);p=Path(qr['path']);q=json.loads(p.read_text());q.pop('current100_r1_credit');q[key]=value
    p.write_text(json.dumps(q));qr=m.reference(p);v=json.loads(review.read_text());v['request']=qr;review.write_text(json.dumps(v))
    with pytest.raises(ContractError):m.member_projection(tmp_path,qr,review,102,tmp_path/'projection')


@pytest.mark.parametrize('schema',['isg-canonical-six-group-batch-v1','isg-canonical-group-batch-v1'])
def test_exact_parent_batches_project_complete_members_without_rewriting(tmp_path,schema):
    qr,review=fixture(tmp_path);original=m.reference(review)
    batch=tmp_path/'batch.json';batch.write_text(json.dumps({'schema':schema,'benchmark_credit':0,
        'requests':[{'class_id':'class','request':qr}]}));br=m.reference(batch)
    result=m.run(tmp_path,br)
    assert len(result['prepared'])==3 and result['submitted_jobs']==[]
    assert m.reference(batch)==br and m.reference(review)==original
    Path(qr['path']).write_text('{}')
    with pytest.raises(ContractError):m.run(tmp_path,br)


@pytest.mark.parametrize('schema,credit',[('unknown',0),('isg-canonical-group-batch-v1',1)])
def test_batch_admission_does_not_authorize_unknown_schema_or_credit(tmp_path,schema,credit):
    batch=tmp_path/'batch.json';batch.write_text(json.dumps({'schema':schema,'benchmark_credit':credit,'requests':[]}))
    with pytest.raises(ContractError):m.run(tmp_path,m.reference(batch))
