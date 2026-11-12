import hashlib
import json
from scripts.campaign_20261110.preparation_watchdog import project


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root, submitted=False):
    d=root/'infra/coordinator_dispatch'; source=root/'source.json'; key='a'*64
    receipt=root/'receipts'/f'{key}.json'; claim=root/'claims'/f'{key}.json'
    request=put(root/'prepared.json', {'observed_at': 600})
    ticket={'id':'map', 'physical_dedup_key':key, 'owner':'engineer', 'expected_account':'minyeong',
            'submission_path':str(receipt), 'claim_path':str(claim), 'request':request, 'ready_since':700}
    manifest=put(root/'tickets.json', {'tickets':[ticket]})
    source_ref=put(source, {'jobs':[ticket]})
    contract=put(root/'scope.json', {'approved_manifests':[{'manifest':source_ref,'required_for_readiness':True,'consumer_kind':'tabletop'}]})
    put(d/'policy.json', {'prepared_manifests':[manifest], 'preparation_scope':{
        'policy_ref':contract,'normalized_manifests':[{'manifest':manifest,'source_manifest':source_ref}]}})
    for account in ['minyeong','shiyuc']:
        put(d/f'status_{account}.json', {'last_reconciliation_at':990,'errors':[], 'requests':[],
            'capacity':{'status':'observed','observed_at':990,'account_slots':{'minyeong':10,'shiyuc':10},'compatible_available_slots':20}})
    if submitted:put(receipt, {'job_id':'123','submitted_at':710})
    return d, key, receipt


def test_ready_with_capacity_alert_has_immutable_clock(tmp_path):
    fixture(tmp_path)
    value=project(tmp_path,[],1000)
    assert value['ready_count']==1 and value['oldest_ready_age_seconds']==300
    alert=next(e for e in value['exceptions'] if e['kind']=='preparation_dispatch_stall')
    second=project(tmp_path,[],1010,value)
    next_alert=next(e for e in second['exceptions'] if e['kind']=='preparation_dispatch_stall')
    assert next_alert['first_raised_at']==alert['first_raised_at']==1000
    assert second['items'][0]['artifact_ready_at']==600


def test_stale_dispatch_never_reports_empty_queue(tmp_path):
    d,_,_=fixture(tmp_path,submitted=True)
    p=d/'status_minyeong.json';s=json.loads(p.read_text());s['last_reconciliation_at']=900;p.write_text(json.dumps(s))
    value=project(tmp_path,[],1000)
    assert value['ready_count'] is None and value['known_ready_count']==0
    assert any(e['kind']=='preparation_dispatcher_missed_two_cycles' for e in value['exceptions'])


def test_delayed_verification_and_documented_pending(tmp_path):
    d,key,_=fixture(tmp_path,submitted=True)
    jobs=[{'job_id':'123','execution_state':'COMPLETED','ended_at':650}]
    value=project(tmp_path,jobs,1000)
    assert value['completed_awaiting_verification']==1
    assert any(e['kind']=='preparation_terminal_unreviewed' for e in value['exceptions'])
    put(d/'preparation_verification'/f'{key}.json', {'status':'pending','owner':'engineer',
        'blocker':'Exact native reader scheduled as CPU callback','next_check_at':1020})
    value=project(tmp_path,jobs,1000)
    assert value['items'][0]['verification_started_at'] is None
    assert not any(e['kind']=='preparation_terminal_unreviewed' for e in value['exceptions'])


def test_zero_grasps_do_not_unlock_downstream(tmp_path):
    d,key,_=fixture(tmp_path,submitted=True)
    put(d/'preparation_verification'/f'{key}.json',{'status':'verified_zero','owner':'engineer',
        'verified_at':800,'downstream_required':False})
    value=project(tmp_path,[{'job_id':'123','execution_state':'COMPLETED','ended_at':750}],1000)
    assert value['completed_awaiting_verification']==0 and value['completed_awaiting_downstream']==0


def test_library_downstream_overdue_then_real_receipt(tmp_path):
    d,key,_=fixture(tmp_path,submitted=True);p=d/'preparation_verification'/f'{key}.json'
    disposition={'status':'verified_positive','owner':'engineer','verified_at':800,'downstream_required':True}
    put(p,disposition);value=project(tmp_path,[],1000)
    assert value['completed_awaiting_downstream']==1
    assert any(e['kind']=='preparation_downstream_stall' for e in value['exceptions'])
    put(p,{**disposition,'downstream_submitted_at':950,'downstream_ids':['456']})
    assert project(tmp_path,[],1000)['completed_awaiting_downstream']==0


def test_ambiguous_submission_never_ready_or_retried(tmp_path):
    _,_,receipt=fixture(tmp_path);put(receipt,{'status':'submitting','observed_at':800})
    value=project(tmp_path,[],1000)
    assert value['ready_count']==0 and value['items'][0]['submitted_at'] is None
    assert any(e['kind']=='preparation_submission_requires_reconciliation' for e in value['exceptions'])


def test_report_published_during_monitor_read_is_not_a_missed_cycle(tmp_path):
    d,_,_=fixture(tmp_path,submitted=True)
    p=d/'status_minyeong.json';s=json.loads(p.read_text());s['last_reconciliation_at']=1001;p.write_text(json.dumps(s))
    value=project(tmp_path,[],1000)
    assert not any(e['kind']=='preparation_dispatcher_missed_two_cycles' for e in value['exceptions'])


def test_optional_historical_work_does_not_create_required_verification_gate(tmp_path):
    d,_,_=fixture(tmp_path,submitted=True)
    p=tmp_path/'scope.json';s=json.loads(p.read_text());s['approved_manifests'][0]['required_for_readiness']=False
    new_ref=put(p,s);policy=json.loads((d/'policy.json').read_text());policy['preparation_scope']['policy_ref']=new_ref;put(d/'policy.json',policy)
    value=project(tmp_path,[{'job_id':'123','execution_state':'COMPLETED','ended_at':600}],1000)
    assert value['completed_awaiting_verification']==0 and value['items']==[]
    assert not any(e['kind']=='preparation_terminal_unreviewed'for e in value['exceptions'])


def test_secondary_capacity_requires_admitted_route_and_preserves_ready_clock(tmp_path):
    d,_,_=fixture(tmp_path)
    for account in ['minyeong','shiyuc']:
        p=d/f'status_{account}.json';v=json.loads(p.read_text())
        v['capacity']['account_slots']={'minyeong':0,'shiyuc':8};put(p,v)
    first=project(tmp_path,[],1000)
    assert not first['items'][0]['compatible_capacity_available']
    ticket=json.loads((tmp_path/'tickets.json').read_text())['tickets'][0]
    secondary={**ticket,'expected_account':'shiyuc','ready_since':990}
    manifest=put(tmp_path/'secondary.json',{'tickets':[secondary]})
    source=put(tmp_path/'secondary_source.json',{'jobs':[secondary]})
    contract=json.loads((tmp_path/'scope.json').read_text())
    contract['approved_manifests'].append({'manifest':source,'required_for_readiness':True,'consumer_kind':'tabletop'})
    policy=json.loads((d/'policy.json').read_text());policy['prepared_manifests'].append(manifest)
    policy['preparation_scope']['policy_ref']=put(tmp_path/'scope.json',contract)
    policy['preparation_scope']['normalized_manifests'].append({'manifest':manifest,'source_manifest':source})
    put(d/'policy.json',policy)
    result=project(tmp_path,[],1000)
    assert result['ready_count']==1 and result['oldest_ready_age_seconds']==300
    assert result['items'][0]['account']=='shiyuc'
    assert result['items'][0]['compatible_capacity_available']
    assert result['items'][0]['admitted_accounts']==['minyeong','shiyuc']
    assert any(e['kind']=='preparation_dispatch_stall' for e in result['exceptions'])
