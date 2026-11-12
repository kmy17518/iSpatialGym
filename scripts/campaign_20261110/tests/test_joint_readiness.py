import hashlib
import json
from scripts.campaign_20261110.joint_readiness import forecast, project


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def setup(root):
    manifest=put(root/'wave/tickets.json', {'jobs':[{'physical_dedup_key':'a'},{'physical_dedup_key':'b'}]})
    index=put(root/'wave/index.json', {'chunks':[{'manifest':manifest}]})
    put(root/'wave/readers/first/status.json', {'rows':[{'status':'verified_positive','builder_elapsed_seconds':120},
                                                    {'status':'verified_positive','builder_elapsed_seconds':240}]})
    streams=[{'id':s,'finish_range':None,'engineering_blockers':['exact input or integration still unresolved']}
             for s in ['preparation','pipeline','eval100']]
    spec={'observed_at':990,'streams':streams,'desired_preparation_gpus':200,'resource_actions':['use admitted secondary route'],
          'preparation_waves':[{'index':index,'reader_directory':str(root/'wave/readers')}]}
    put(root/'infra/coordination/joint_schedule_inputs.json',spec)
    put(root/'vast/metrics.json',{'observed_at_unix':990,'instances':[],'running_gpus':0,
                                'spend_estimate_usd':1000,'future_compute_commitment_usd':100,'transfer_reserve_usd':300})
    flow={'items':[{'physical_dedup_key':'a','verified_at':900,'backend':'cluster','execution_state':'COMPLETED'},
                   {'physical_dedup_key':'b','backend':'cluster','execution_state':'RUNNING'}]}
    return spec,flow


def test_known_compute_never_becomes_full_readiness_forecast(tmp_path):
    _,flow=setup(tmp_path);value=project(tmp_path,flow,1000)
    assert value['common_finish_range'] is None and value['preparation_extra_delay_seconds'] is None
    assert value['known_wave']['remaining_gpu_jobs']==1
    assert value['capacity']['current_executing_wave_jobs']==1
    assert value['known_wave']['conditional_finish_ranges'][0]['known_wave_finish_range']
    assert not value['full_preparation_forecast_known']


def test_tail_and_stale_engineering_review_are_alerted(tmp_path):
    spec,flow=setup(tmp_path)
    for stream,times in zip(spec['streams'],([2000,2500],[1400,1600],[1500,1800])):stream['finish_range']=times
    put(tmp_path/'infra/coordination/joint_schedule_inputs.json',spec)
    value=project(tmp_path,flow,1400)
    assert value['preparation_target_range']==[1500,1800]
    assert value['preparation_extra_delay_seconds']==[200,1000]
    assert value['common_finish_range']==[2000,2500]
    kinds={e['kind']for e in value['exceptions']}
    assert {'joint_engineering_forecast_review_due','preparation_forecast_tail'}<=kinds


def test_more_useful_concurrency_reduces_only_known_compute_window():
    assert forecast(100,[120,240],100,1000)[1] < forecast(100,[120,240],10,1000)[1]
    assert forecast(100,[],100,1000) is None
    assert forecast(100,[120],0,1000) is None


def test_other_required_work_counts_but_does_not_inflate_map_rate(tmp_path):
    spec,flow=setup(tmp_path)
    flow['items'].append({'physical_dedup_key':'library','backend':'cluster','execution_state':'RUNNING'})
    spec['desired_preparation_gpus']=2
    put(tmp_path/'infra/coordination/joint_schedule_inputs.json',spec)
    value=project(tmp_path,flow,1000)
    assert value['capacity']['current_executing_preparation_jobs']==2
    assert value['capacity']['current_executing_wave_jobs']==1
    assert value['capacity']['desired_known_wave_gpus_after_other_work']==1
    assert value['capacity']['additional_useful_gpus_needed']==0


def test_incremental_manifest_and_stale_vast_are_explicit(tmp_path):
    spec,flow=setup(tmp_path)
    extra=put(tmp_path/'extra.json',{'jobs':[{'physical_dedup_key':'c'}]})
    spec['preparation_waves'].append({'manifest':extra,'reader_directory':str(tmp_path/'extra_readers')})
    put(tmp_path/'infra/coordination/joint_schedule_inputs.json',spec)
    flow['items'].append({'physical_dedup_key':'c','backend':'vast','job_id':'10','execution_state':'RUNNING'})
    value=project(tmp_path,flow,1500)
    assert value['known_wave']['physical_jobs']==3
    assert value['capacity']['vast_observation_current'] is False
    assert value['capacity']['vast_provider_gpus'] is None
    assert value['capacity']['current_executing_preparation_jobs']==1
