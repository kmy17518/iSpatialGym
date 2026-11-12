import json
from scripts.campaign_20261110.monitor_projection import freshness, project


def fixture(tmp_path, registry):
    (tmp_path/'infra/coordination').mkdir(parents=True)
    (tmp_path/'infra/coordination/ownership.json').write_text(json.dumps(registry))
    (tmp_path/'validation').mkdir()
    (tmp_path/'validation/controller.json').write_text('{"gates":{}}')
    return tmp_path


def test_unknown_job_never_uses_name(tmp_path):
    root=fixture(tmp_path,{})
    result=project(root,[{'id':'1','state':'RUNNING','name':'isg261110_canary_geometry_held'}],{},now=1000,accountant=lambda ids:[])
    row=result['jobs'][0]
    assert row['mapping_status']=='missing' and row['mapping_remediation_owner']=='/root'
    assert row.get('workstream') is None and row['execution_state']=='RUNNING'
    assert row['utilization_percent'] is None


def test_explicit_submission_and_completed_handoff(tmp_path):
    p=tmp_path/'submitted.json';p.write_text('{"job_id":"2"}')
    root=fixture(tmp_path,{'job_sources':[{'path':str(p),'stage':'native','owner':'engineer','workstream':'pipeline'}],
        'backlog':[{'requirement':'B reset','owner':'engineer','jobs':['2'],'last_verified_event_at':995}]})
    result=project(root,[],{},now=1000,accountant=lambda ids:[{'id':'2','state':'COMPLETED'}])
    assert result['jobs'][0]['mapping_status']=='mapped'
    assert any(e['kind']=='completed_not_advanced' for e in result['exceptions'])


def test_stale_event_does_not_change_execution(tmp_path):
    root=fixture(tmp_path,{'jobs':[{'job_id':'3','stage':'native','owner':'e','workstream':'canonical recovery'}],
        'backlog':[{'requirement':'x','owner':'e','jobs':['3'],'last_verified_event_at':1}]})
    result=project(root,[{'id':'3','state':'RUNNING'}],{},now=1000,accountant=lambda ids:[])
    assert result['jobs'][0]['execution_state']=='RUNNING'
    assert result['backlog'][0]['observation_status']=='stale'


def test_observation_failure_is_not_execution_failure(tmp_path):
    root=fixture(tmp_path,{'jobs':[{'job_id':'4'}]})
    def fail(ids):raise OSError('unreachable')
    result=project(root,[],{},now=1000,accountant=fail)
    assert result['jobs'][0]['execution_state']=='UNKNOWN'
    assert any(e['kind']=='observation_failure' for e in result['exceptions'])


def test_ready_clock_and_controller_liveness_separate(tmp_path):
    p=tmp_path/'report.json';p.write_text('{"observed_at":10}')
    root=fixture(tmp_path,{'controllers':[{'name':'replay','job_id':'5','report':str(p)}],
        'backlog':[{'requirement':'ready','owner':'e','stage':'ready_for_dispatch','ready_since':950,'last_verified_event_at':950}]})
    result=project(root,[{'id':'5','state':'RUNNING'}],{},now=1000,accountant=lambda ids:[])
    assert result['workers'][0]['process_state']=='UNKNOWN'
    assert result['workers'][0]['allocation']['state']=='RUNNING'
    assert result['workers'][0]['observation_status']=='stale'
    assert result['workers'][0]['last_useful_progress_at'] is None
    assert next(e for e in result['exceptions'] if e['kind']=='ready_not_submitted')['waiting_seconds']==50


def test_freshness_missing():
    assert freshness(None,1000)=='missing'
    assert freshness(1,1000)=='stale'
