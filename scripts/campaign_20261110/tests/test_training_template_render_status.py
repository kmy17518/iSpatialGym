import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import training_pipeline_status as status
from scripts.campaign_20261110.contracts import write_json_atomic


def fixture(root):
    directory=root/'training_pipeline_canaries/pool/template_rendering_r1/candidate'
    directory.mkdir(parents=True)
    record={'candidate_id':'candidate','start':12,'instance_id':'source__12',
            'native_runtime':str(root/'code/native'),'configuration_sha256':None,
            'results':str(directory/'results.jsonl'),'output':str(directory/'output')}
    for key in ('configuration','canary_review','source_environment','native_runtime_identity','manifest','demos'):
        path=directory/(key+'.json');write_json_atomic(path,{'identity':key});record[key]=status.ref(path)
    record['configuration_sha256']=record['configuration']['sha256']
    request=directory.parent/'request.json'
    write_json_atomic(request,{'schema':'isg-training-template-render-request-v1',
        'campaign_id':'campaign_20261110','production_released':False,'benchmark_credit':0,'records':[record]})
    launch={'instances':['source__12'],'request':status.ref(request),
            'configuration':record['configuration'],'native_runtime_identity':record['native_runtime_identity']}
    write_json_atomic(directory/'launch.json',launch)
    return directory,request,record


def result(directory, *, wrong_member=False):
    dataset=directory/'output/dataset';dataset.mkdir(parents=True)
    (dataset/'observations.parquet').write_bytes(b'executed observation payload')
    (dataset/'camera.mp4').write_bytes(b'executed camera payload')
    iid='foreign__13' if wrong_member else 'source__12'
    value={'schema':'isg-render-result-v1','launch_sha256':status.file_digest(directory/'launch.json'),
           'success':True,'returncode':0,'expected_instances':[iid],'rendered_instances':[iid],
           'kept_instances':[iid],'episodes':[{'demo':'/source/'+iid,'kept':True,
               'episode_index':0,'dataset':str(dataset),'campaign_restoration':{'scale_verified':True}}]}
    write_json_atomic(directory/'result.json',value)
    return dataset


def test_submission_only_never_claims_execution_or_success(tmp_path):
    d,p,r=fixture(tmp_path)
    write_json_atomic(d/'submission.json',{'request':status.ref(p),'candidate_id':'candidate',
        'start':12,'job_id':'123','native_runtime':r['native_runtime'],'launch':status.ref(d/'launch.json'),
        'source_environment':r['source_environment']})
    observed=status.template_render_observations(tmp_path,p)[('candidate',12)]
    assert observed['status']=='submitted' and observed['scheduler_state']=='unavailable'
    assert 'result' not in observed and observed['benchmark_credit']==0


def test_exact_single_episode_saved_artifacts_are_required(tmp_path):
    d,p,r=fixture(tmp_path);dataset=result(d)
    observed=status.template_render_observations(tmp_path,p)[('candidate',12)]
    assert observed['status']=='verified' and len(observed['artifacts'])==2
    assert observed['configuration_sha256']==r['configuration_sha256']
    (dataset/'camera.mp4').unlink()
    with pytest.raises(ValueError,match='saved observation data/video'):
        status.template_render_observations(tmp_path,p)


def test_success_boolean_cannot_promote_foreign_episode(tmp_path):
    d,p,_=fixture(tmp_path);result(d,wrong_member=True)
    observed=status.template_render_observations(tmp_path,p)[('candidate',12)]
    assert observed['status']=='failed_validation' and 'artifacts' not in observed


def test_mutated_source_or_launch_is_not_accepted(tmp_path):
    d,p,r=fixture(tmp_path);result(d)
    launch=json.loads((d/'launch.json').read_text());launch['configuration']={'path':'/foreign','sha256':'a'*64}
    write_json_atomic(d/'launch.json',launch)
    with pytest.raises(ValueError,match='launch differs'):
        status.template_render_observations(tmp_path,p)


def policy_review_fixture(root):
    policies=root/'infra/training_production_gate/current_evaluations_r1'
    old_policy={'schema':'isg-training-production-gate-v1','campaign_id':'campaign_20261110',
                'mode':'blocked','standalone_training_allowed':False,'allowed_canaries':[{'id':'exact'}],
                'canonical_eval_required':100,'observed_at':'before'}
    write_json_atomic(policies/'before_previous_policy.json',old_policy)
    write_json_atomic(policies/'observation.json',{'status':'blocked','canonical':99})
    new_policy={**old_policy,'observed_at':'after','current_evaluation':status.ref(policies/'observation.json'),
                'evaluation_policy_history':{'authorization_change':False}}
    write_json_atomic(policies/'after_previous_policy.json',new_policy)
    review={'schema':'isg-training-canary-review-v1','campaign_id':'campaign_20261110',
            'validation_passed':True,'accepted':False,'eligible_for_benchmark':False,'quota_consumed':0,
            'instance_id':'exact__12','source_proof':{'config':{'sha256':'a'*64}},
            'export_record':{'artifacts':{'tro':{'sha256':'b'*64},'demo':{'sha256':'c'*64}}},
            'gate_policy_sha256':status.file_digest(policies/'before_previous_policy.json')}
    reviews=root/'validation/training_canaries/reviews'
    write_json_atomic(reviews/'old.json',review)
    write_json_atomic(reviews/'new.json',{**review,'gate_policy_sha256':status.file_digest(policies/'after_previous_policy.json')})
    row={'configuration_sha256':'a'*64,'native_runtime':'/native','validated_evidence':{'review':status.ref(reviews/'new.json')}}
    rendering={'configuration_sha256':'a'*64,'native_runtime':'/native','instance_id':'exact__12','review':status.ref(reviews/'old.json')}
    return row,rendering,policies,reviews


def test_policy_observation_only_revision_keeps_both_exact_reviews(tmp_path):
    row,rendering,policies,reviews=policy_review_fixture(tmp_path)
    result=status.render_review_association(tmp_path,row,rendering)
    assert result['status']=='identical_evidence_policy_observation_revision'
    assert result['render_review']==status.ref(reviews/'old.json')
    assert result['current_review']==status.ref(reviews/'new.json')
    assert result['excluded_policy_metadata']==['current_evaluation','evaluation_policy_history','observed_at']
    assert result['render_policy']['sha256']!=result['current_policy']['sha256']


@pytest.mark.parametrize('field,value',[('allowed_canaries',[]),('standalone_training_allowed',True),
                                        ('canonical_eval_required',99),('unrecognized_policy_field',True)])
def test_semantic_policy_change_never_reconciles_render(tmp_path,field,value):
    row,rendering,policies,reviews=policy_review_fixture(tmp_path)
    policy=status.read(policies/'after_previous_policy.json');policy[field]=value
    write_json_atomic(policies/'after_previous_policy.json',policy)
    proof=status.read(reviews/'new.json');proof['gate_policy_sha256']=status.file_digest(policies/'after_previous_policy.json')
    write_json_atomic(reviews/'new.json',proof);row['validated_evidence']['review']=status.ref(reviews/'new.json')
    with pytest.raises(ValueError,match='authorization or release semantics'):
        status.render_review_association(tmp_path,row,rendering)


def test_missing_historical_policy_or_changed_trajectory_is_not_inferred(tmp_path):
    row,rendering,policies,reviews=policy_review_fixture(tmp_path)
    original=(policies/'before_previous_policy.json').read_bytes();(policies/'before_previous_policy.json').unlink()
    with pytest.raises(ValueError,match='historical gate policy bytes unavailable'):
        status.render_review_association(tmp_path,row,rendering)
    (policies/'before_previous_policy.json').write_bytes(original)
    proof=status.read(reviews/'new.json');proof['export_record']['artifacts']['tro']['sha256']='d'*64
    write_json_atomic(reviews/'new.json',proof);row['validated_evidence']['review']=status.ref(reviews/'new.json')
    with pytest.raises(ValueError,match='source or validation evidence'):
        status.render_review_association(tmp_path,row,rendering)


def test_bad_render_record_does_not_hide_other_exact_templates(tmp_path):
    d,p,r=fixture(tmp_path);result(d)
    request=status.read(p);request['records'].insert(0,{**r,'candidate_id':'bad','configuration':{'path':str(d/'missing'),'sha256':'a'*64}})
    write_json_atomic(p,request)
    launch=status.read(d/'launch.json');launch['request']=status.ref(p);write_json_atomic(d/'launch.json',launch)
    saved=status.read(d/'result.json');saved['launch_sha256']=status.file_digest(d/'launch.json');write_json_atomic(d/'result.json',saved)
    errors=[];rows=status.template_render_observations(tmp_path,p,errors=errors)
    assert list(rows)==[('candidate',12)] and rows[('candidate',12)]['status']=='verified'
    assert len(errors)==1 and errors[0]['candidate_id']=='bad'
