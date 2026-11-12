"""Metrics publication cannot invoke resource authorization or delay assignment."""
import ast
import json
import os
from pathlib import Path


def publisher(root, *, preserved=(), stager=None):
    path=Path(__file__).parents[1]/'infra_vast/vastctl.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='publish_metrics')
    namespace={'__name__':'scripts.campaign_20261110.infra_vast.metrics_probe',
               '__package__':'scripts.campaign_20261110.infra_vast','os':os,'json':json,
               'VJ':str(root),'F':str(root),'STAGER':stager,'now':lambda:20,
               'preserved_instances':lambda _:set(preserved),'SHORT':{},'log':lambda *args:None,
               'staging_compression_threads':lambda _:1,
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


def test_stopped_providers_retain_recovery_evidence_but_no_worker_capacity(tmp_path):
    live={str(i):{'id':i,'actual_status':status,'num_gpus':1} for i,status in
          [(1,'exited'),(2,'stopped'),(3,'exited')]}
    heartbeats={'1':{'phase':'run','time':999,'running':{'job1':{}}},
                '2':{'phase':'ready','time':999},'3':{'phase':'boot','time':999}}
    publisher(tmp_path,preserved=live)({'max_instances':100,'budget_usd':3000},
        {},live,heartbeats,{'1':['job1']},1000,True,1,2,3,'complete')
    value=json.loads((tmp_path/'metrics.json').read_text())
    for field in ('running_gpus','worker_booting_gpus','worker_ready_gpus','worker_running_gpus',
                  'worker_idle_ready_gpus','worker_staging_gpus','worker_unknown_gpus','rental_pending_gpus'):
        assert value[field]==0,field
    assert value['preserved_stopped_instances']==['1','2','3']
    assert value['instances'][0]['assigned_jobs']==['job1']
    assert value['instances'][0]['worker_phase']=='run'
    assert value['instances'][0]['heartbeat_observation_status']=='fresh'
    assert not value['instances'][0]['worker_observation_current']


def test_live_workers_require_original_fresh_nonfuture_heartbeat(tmp_path):
    live={str(i):{'id':i,'actual_status':'running','num_gpus':1} for i in range(1,8)}
    heartbeats={'1':{'phase':'ready','time':700},  # inclusive 300-second boundary
                '2':{'phase':'ready','time':699},
                '3':{'phase':'run','time':1001,'running':{'j3':{}}},
                '4':{'phase':'boot'},'5':{'phase':'ready','time':'999'},
                '6':{'phase':'run','time':999,'running':{'j6':{}}},
                '7':{'phase':'boot','time':999}}
    publisher(tmp_path)({'max_instances':100,'budget_usd':3000},{},live,heartbeats,
                       {'2':['j2'],'3':['j3'],'6':['j6']},1000,True,1,2,3,'complete')
    value=json.loads((tmp_path/'metrics.json').read_text())
    assert value['running_gpus']==7
    assert value['worker_ready_gpus']==2 and value['worker_running_gpus']==1
    assert value['worker_idle_ready_gpus']==1 and value['worker_booting_gpus']==1
    assert value['worker_unknown_gpus']==4
    assert value['worker_heartbeat_max_age_seconds']==300
    assert [row['heartbeat_observation_status'] for row in value['instances']]==[
        'fresh','stale','future','missing','missing','fresh','fresh']


def test_staging_and_idle_counts_exclude_stale_and_retained_provider_records(tmp_path):
    class Stager:
        def reservations(self):return {'1':['stage1'],'2':['stage2'],'3':['stage3']}
        def metrics(self,_):return {'in_flight':3}
    live={str(i):{'id':i,'actual_status':'exited' if i==3 else 'running','num_gpus':2}
          for i in range(1,5)}
    heartbeats={str(i):{'phase':'ready','time':699 if i==2 else 999} for i in range(1,5)}
    heartbeats['4']['running']={'worker_job':{}}
    publisher(tmp_path,stager=Stager())({'max_instances':100,'budget_usd':3000},
        {},live,heartbeats,{},1000,True,1,2,3,'complete')
    value=json.loads((tmp_path/'metrics.json').read_text())
    assert value['worker_staging_gpus']==2 and value['worker_idle_ready_gpus']==0
    assert value['worker_ready_gpus']==4 and value['worker_unknown_gpus']==2
    assert value['staging_scheduler']['in_flight']==3  # preserve reservation evidence


def test_incomplete_provider_listing_keeps_capacity_unknown(tmp_path):
    publisher(tmp_path,preserved=['stopped'])({'max_instances':100,'budget_usd':3000},{},
        {'1':{'id':1,'actual_status':'running','num_gpus':1}},
        {'1':{'phase':'ready','time':999}}, {},1000,False,1,2,3,'complete')
    value=json.loads((tmp_path/'metrics.json').read_text())
    for field in ('running_gpus','worker_booting_gpus','worker_ready_gpus','worker_running_gpus',
                  'worker_idle_ready_gpus','worker_staging_gpus','worker_unknown_gpus','rental_pending_gpus'):
        assert value[field] is None,field
    assert value['preserved_stopped_instances']==['stopped']
