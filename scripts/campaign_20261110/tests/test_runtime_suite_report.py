import json
from unittest.mock import patch
from scripts.campaign_20261110 import runtime_suite_report as r


def fixture(root):
    base=root/'runtime_validation/run';base.mkdir(parents=True)
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    put(base/'release.json',{'run_id':'run','frozen_runtime':{'commit':'frozen'}})
    put(base/'manifest.json',{'classes':[{'class_id':'class0'}],'summary':{}})
    put(root/'runtime_validation/current.json',{'release':{'path':str(base/'release.json')},'manifest':{'path':str(base/'manifest.json')}})
    put(base/'allocations/class0.json',{'allocation':{'jobid':'1'},'execution_state':'COMPLETED'})
    return base,put


def test_completed_collection_is_not_validation_pass(tmp_path):
    fixture(tmp_path)
    v=r.report(tmp_path)
    assert v['validation_passed']==0
    assert v['rows'][0]['state']=='collection_terminal_awaiting_checks'


def test_failed_demo_and_stale_predecessor_are_distinct(tmp_path):
    base,put=fixture(tmp_path)
    put(base/'old.json',{'allocation':{'jobid':'0'}})
    audit={'allocation':{'path':str(base/'old.json')},'status':'insufficient_successful_demos','failed_demo_report_paths':['failure.json']}
    put(base/'acceptance_observer.json',{'classes':{'class0':audit}})
    assert r.report(tmp_path)['rows'][0]['state']=='collection_terminal_awaiting_checks'
    audit['allocation']['path']=str(base/'allocations/class0.json')
    put(base/'acceptance_observer.json',{'classes':{'class0':audit}})
    row=r.report(tmp_path)['rows'][0]
    assert row['state']=='insufficient_successful_demos' and row['blocker']==['failure.json']
    assert row['owner']=='/root'


def test_acceptance_lock_reports_acceptance_owner(tmp_path):
    base,put=fixture(tmp_path)
    put(base/'acceptance_observer.json',{'classes':{'class0':{'allocation':{'path':str(base/'allocations/class0.json')},'status':'awaiting_acceptance_lock'}}})
    row=r.report(tmp_path)['rows'][0]
    assert row['state']=='awaiting_acceptance_lock' and row['owner']=='/root/dataset_contract'


def test_registered_retry_does_not_inherit_predecessor_job_or_acceptance(tmp_path):
    base,put=fixture(tmp_path)
    put(base/'rows/class0.json',{'fresh_start':1,'fresh_count':3})
    new=base/'attempts/class0/attempt_1/row.json'
    put(new,{'attempt_number':1,'fresh_start':4,'fresh_count':3})
    put(base/'acceptance/class0.json',{'validation_passed':False,'status':'failed','row':{'path':'old'}})
    refs=[{'path':str(base/'rows/class0.json')},{'path':str(new)}]
    with patch.object(r,'attempt_scope',return_value=(refs[-1],refs,None)):
        row=r.report(tmp_path)['rows'][0]
    assert row['fresh_indices']==[4,3] and row['state']=='ready_unsubmitted'
    assert row['jobs']==[] and row['acceptance'] is None
    assert row['historical_acceptance']['status']=='failed'
    assert row['attempt_rows']==refs


def test_unreadable_registry_is_explicit_not_original_attempt_fallback(tmp_path):
    base,put=fixture(tmp_path)
    with patch.object(r,'attempt_scope',return_value=(None,[],'changed registry bytes')):
        row=r.report(tmp_path)['rows'][0]
    assert row['state']=='attempt_registry_unavailable'
    assert row['blocker']==['changed registry bytes']


def test_owned_diagnosis_is_visible_without_changing_execution_stage(tmp_path):
    base,put=fixture(tmp_path)
    put(base/'diagnoses/class0.json',{'owner':'/root/templates_assets','blocker':'exact fixed geometry excludes requested bucket','next_action':'Verify corrected input after old allocation terminates','observed_at':123})
    row=r.report(tmp_path)['rows'][0]
    assert row['state']=='collection_terminal_awaiting_checks'
    assert row['owner']=='/root/templates_assets' and row['diagnosis']['observed_at']==123
    assert row['blocker']==['exact fixed geometry excludes requested bucket']
