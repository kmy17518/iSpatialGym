import json
import hashlib
import pytest

from scripts.campaign_20261110.infra_vast import fleet_accounting as fa


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def attempt(root, job, instance, start, end, success):
    path=root/'vast/state/attempts'/job/'1.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'instance':instance,'done':{'fetch_started_unix':start,
        'upload_finished_unix':end,'gpu_count':1,'successes':success}}))


def test_all_rental_time_keeps_startup_failed_work_and_idle_unattributed(tmp_path):
    write_rows(tmp_path/'vast/ledger/events.jsonl',[
        {'ev':'create','instance':'1','time':10,'num_gpus':1},
        {'ev':'create','instance':'2','time':35,'num_gpus':1},
        {'ev':'destroy','instance':'1','time':80,'ok':True}])
    fa.observe(tmp_path/'vast',[{'id':'1','num_gpus':1,'actual_status':'running','api_key':'must_not_appear'}],True,19,20)
    fa.observe(tmp_path/'vast',[{'id':'2','num_gpus':1,'actual_status':'running'}],True,99,100)
    attempt(tmp_path,'failed','1',30,40,0);attempt(tmp_path,'retry','1',45,60,1)
    attempt(tmp_path,'replay','2',50,75,1)
    value=fa.snapshot(tmp_path,100)
    assert value['fleet_window_estimated_gpu_seconds']==125  # 60 + 65 rental seconds.
    assert value['fleet_window_measured_attributed_worker_gpu_seconds']==50
    assert value['fleet_window_unattributed_gpu_seconds']==75
    assert value['recorded_fleet_estimated_gpu_seconds']==135
    assert value['recorded_fleet_unattributed_gpu_seconds']==85
    assert 'api_key' not in (tmp_path/'vast/ledger/rental_presence.jsonl').read_text()


def test_legacy_missing_gpu_count_remains_unavailable_without_guessing(tmp_path):
    write_rows(tmp_path/'vast/ledger/events.jsonl',[{'ev':'create','instance':'old','time':1,'slots':1}])
    value=fa.snapshot(tmp_path,100)
    assert value['fleet_window_estimated_gpu_seconds'] is None
    assert value['historical_unknown_gpu_count_instances']==['old']
    assert value['recorded_fleet_estimated_gpu_seconds'] is None
    fa.observe(tmp_path/'vast',[{'id':'old','num_gpus':1}],True,19,20)
    fa.observe(tmp_path/'vast',[{'id':'old','num_gpus':1}],True,99,100)
    value=fa.snapshot(tmp_path,100)
    assert value['fleet_window_estimated_gpu_seconds']==80
    assert value['fleet_window_measured_attributed_worker_gpu_seconds']==0


def test_overlapping_retained_intervals_do_not_double_count_rental_gpu_time(tmp_path):
    write_rows(tmp_path/'vast/ledger/events.jsonl',[{'ev':'create','instance':'1','time':1,'num_gpus':1}])
    fa.observe(tmp_path/'vast',[{'id':'1','num_gpus':1}],True,9,10)
    fa.observe(tmp_path/'vast',[{'id':'1','num_gpus':1}],True,99,100)
    attempt(tmp_path,'a','1',20,60,0);attempt(tmp_path,'b','1',50,80,1)
    value=fa.snapshot(tmp_path,100)
    assert value['fleet_window_measured_attributed_worker_gpu_seconds']==60
    assert value['fleet_window_unattributed_gpu_seconds']==30


def test_exact_historical_offer_recovers_count_but_not_unknown_timing_or_attribution(tmp_path):
    write_rows(tmp_path/'vast/ledger/events.jsonl',[
        {'ev':'create','instance':'old','time':10,'offer':7,'machine_id':8},
        {'ev':'destroy','instance':'old','time':80,'ok':True}])
    fa.observe(tmp_path/'vast',[],True,99,100)
    market=tmp_path/'vast/market/old.json';market.parent.mkdir()
    market.write_text(json.dumps({'time':9,'offers':[{'id':7,'machine_id':8,'num_gpus':2}]}))
    receipt=tmp_path/'accounting/vast_rental_gpu_count_sources.json';receipt.parent.mkdir()
    receipt.write_text(json.dumps({'schema':'isg-vast-exact-historical-gpu-count-v1','campaign_id':'campaign_20261110',
        'records':[{'instance':'old','offer':7,'machine_id':8,'num_gpus':2,'source_path':str(market),
                    'source_sha256':hashlib.sha256(market.read_bytes()).hexdigest()}]}))
    value=fa.snapshot(tmp_path,200)
    assert value['historical_unknown_gpu_count_instances']==[]
    assert value['recorded_fleet_estimated_gpu_seconds']==140
    assert value['recorded_fleet_measured_attributed_worker_gpu_seconds']==0
    assert value['recorded_fleet_unattributed_gpu_seconds']==140
    assert value['rentals'][0]['creation_request_started_unix'] is None
    market.write_text(market.read_text()+' ')
    with pytest.raises(ValueError,match='source changed'):fa.snapshot(tmp_path,200)
