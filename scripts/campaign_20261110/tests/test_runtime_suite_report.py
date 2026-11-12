import json
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
