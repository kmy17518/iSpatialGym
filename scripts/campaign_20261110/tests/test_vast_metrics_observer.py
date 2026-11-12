"""Metrics publication cannot invoke resource authorization or delay assignment."""
import ast
import json
import os
from pathlib import Path


def publisher(root):
    path=Path(__file__).parents[1]/'infra_vast/vastctl.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='publish_metrics')
    namespace={'__name__':'scripts.campaign_20261110.infra_vast.metrics_probe',
               '__package__':'scripts.campaign_20261110.infra_vast','os':os,'json':json,
               'VJ':str(root),'F':str(root),'STAGER':None,'now':lambda:20,
               'preserved_instances':lambda _:set(),'SHORT':{},'log':lambda *args:None,
               'dispatch_order':lambda *_:(_ for _ in ()).throw(AssertionError('metrics must not authorize jobs'))}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['publish_metrics']


def test_early_metrics_eligibility_is_unavailable_and_does_not_rescan(tmp_path):
    publish=publisher(tmp_path)
    publish({'max_instances':100,'budget_usd':3000},{'1':{'id':'1','status':'queued'}},{},{},{},10,True,1,2,3,'dispatching')
    value=json.loads((tmp_path/'metrics.json').read_text())
    assert value['dispatch_eligible_queued'] is None
    assert value['dispatch_excluded_queued'] is None
    assert value['observed_at_unix']==10 and value['assignments_observed_at_unix']==20


def test_metrics_filter_checked_ids_by_current_state_without_reauthorization(tmp_path):
    states={str(i):{'id':str(i),'status':status}for i,status in [(1,'assigned'),(2,'queued'),(3,'queued')]}
    publisher(tmp_path)({'max_instances':100,'budget_usd':3000},states,{},{},{},10,True,1,2,3,'complete',
                        {'job_ids':{'1','2'},'observed_at':15})
    value=json.loads((tmp_path/'metrics.json').read_text())
    assert value['dispatch_eligible_queued']==1 and value['dispatch_excluded_queued']==1
    assert value['dispatch_eligibility_observed_at_unix']==15
