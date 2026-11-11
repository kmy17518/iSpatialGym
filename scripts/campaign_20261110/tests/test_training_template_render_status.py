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
