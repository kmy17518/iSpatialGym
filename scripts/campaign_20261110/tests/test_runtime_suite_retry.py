import copy
import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import runtime_suite_retry as r, runtime_suite as s, production_gate as g
from scripts.campaign_20261110.contracts import ContractError, file_digest
from scripts.campaign_20261110.tests.test_runtime_suite import put
from scripts.campaign_20261110.tests.test_runtime_suite_acceptance import bound


def proposed(root):
    base,prior,allocation,gate=bound(root)
    row=json.loads(Path(prior['path']).read_text());directory=base/'attempts/class0/attempt_1'
    put(root/'manifests/family_index_reservations.json',{'reservations':{
        'original':{'start':1000,'n':300},'retry':{'start':2000,'n':3}}})
    row.update(attempt_number=1,previous_row=prior,fresh_start=2000,group_id='retry1',
               goal_plan=put(directory/'goal.json',{'group_id':'retry1'}))
    rowref=put(directory/'row.json',row)
    diagnosis=put(directory/'diagnosis.json',{'schema':'isg-runtime-suite-retry-diagnosis-v1','owner':'/root',
        'previous_row':prior,'allocation':allocation,'failed_gates':[],
        'changed_reason':'Scheduler infrastructure failed before native initialization.',
        'next_action':'Use the corrected bootstrap with the identical immutable native configuration.',
        'physical_runtime_change':False,'configuration_change':False,
        'evidence':[json.loads(Path(allocation['path']).read_text())['terminal_evidence']]})
    proposal=put(directory/'proposal.json',{'schema':'isg-runtime-suite-retry-proposal-v1','row':rowref,
        'previous_row':prior,'allocation':allocation,'diagnosis':diagnosis,'max_attempts':2,
        'reservation':{'token':'retry','start':2000,'reserved_count':3}})
    return base,prior,allocation,rowref,proposal


def approve(root,prepared):
    p=Path(prepared['path']);value=json.loads(p.read_text())
    return put(p.parent/'approval.json',{'schema':'isg-runtime-suite-retry-approval-v1','owner':'/root',
        'approved':True,'prepared':prepared,'diagnosis':value['diagnosis'],'automatic_retry':False})


def environment(rowref):
    row=json.loads(Path(rowref['path']).read_text())
    return {'ISPATIALGYM_CAMPAIGN_CONFIG':row['campaign_config']['path'],
        'ISPATIALGYM_CAMPAIGN_GOAL_PLAN':row['goal_plan']['path'],
        'ISPATIALGYM_RUNTIME_SUITE_RUN':row['run_id'],'ISPATIALGYM_RUNTIME_SUITE_ROW':rowref['path'],
        'ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256':rowref['sha256'],'ISGE_CODE':row['frozen_runtime']['path']}


def test_prepare_does_not_authorize_and_registration_preserves_original(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    oldrow=file_digest(prior['path']);reservation=file_digest(base/'index_reservation.json')
    prepared=r.prepare(tmp_path,proposal)
    assert not g.check_environment(tmp_path,environment(rowref),start=2000,count=3)['allowed']
    assert g.check_environment(tmp_path,environment(prior),start=1000,count=3)['allowed']
    result=r.register(tmp_path,prepared,approve(tmp_path,prepared))
    assert not result['dispatch_performed']
    assert g.check_environment(tmp_path,environment(rowref),start=2000,count=3)['allowed']
    assert not g.check_environment(tmp_path,environment(prior),start=1000,count=3)['allowed']
    assert g.check_environment(tmp_path,environment(prior),start=1000,count=3,kind='source_validation')['allowed']
    assert file_digest(prior['path'])==oldrow and file_digest(base/'index_reservation.json')==reservation
    assert not (tmp_path/'dataset').exists()


@pytest.mark.parametrize('field,value', [('fresh_start',1000),('fresh_count',4),('frozen_runtime',{}),
                                        ('campaign_config',{}),('group_id','suite0')])
def test_retry_cannot_change_identity_or_reuse_inputs(tmp_path,field,value):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    row=json.loads(Path(rowref['path']).read_text());row[field]=value
    body=json.loads(Path(proposal['path']).read_text());body['row']=put(Path(rowref['path']),row)
    proposal=put(Path(proposal['path']),body)
    with pytest.raises((ContractError,KeyError)):r.prepare(tmp_path,proposal)


def test_goal_semantics_cannot_be_changed_under_retry_authorization(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    row=json.loads(Path(rowref['path']).read_text());row['goal_plan']=put(Path(row['goal_plan']['path']),{'group_id':'retry1','goal_pose':[9,9,9]})
    body=json.loads(Path(proposal['path']).read_text());body['row']=put(Path(rowref['path']),row)
    with pytest.raises(ContractError,match='physical goal'):r.prepare(tmp_path,put(Path(proposal['path']),body))


def test_pending_job_is_not_retryable(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    raw=json.loads((base/'raw.json').read_text());raw['squeue_stdout']='123|RUNNING'
    terminal=json.loads((base/'terminal.json').read_text());terminal['jobs'][0]['raw']=put(base/'raw.json',raw)
    wrapper=json.loads(Path(allocation['path']).read_text());wrapper['terminal_evidence']=put(base/'terminal.json',terminal)
    ar=put(Path(allocation['path']),wrapper);body=json.loads(Path(proposal['path']).read_text());body['allocation']=ar
    diagnosis=json.loads(Path(body['diagnosis']['path']).read_text());diagnosis['allocation']=ar;diagnosis['evidence']=[wrapper['terminal_evidence']]
    body['diagnosis']=put(Path(body['diagnosis']['path']),diagnosis)
    with pytest.raises(ContractError,match='active'):r.prepare(tmp_path,put(Path(proposal['path']),body))


def test_missing_approval_and_exhausted_budget_fail_closed(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path);prepared=r.prepare(tmp_path,proposal)
    approval=approve(tmp_path,prepared);body=json.loads(Path(approval['path']).read_text());body['approved']=False
    with pytest.raises(ContractError,match='explicit exact approval'):r.register(tmp_path,prepared,put(Path(approval['path']),body))
    body=json.loads(Path(proposal['path']).read_text());body['max_attempts']=1
    with pytest.raises(ContractError,match='budget exhausted'):r.prepare(tmp_path,put(Path(proposal['path']),body))


def test_new_overlapping_reservation_revokes_retry_dispatch(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path);prepared=r.prepare(tmp_path,proposal)
    r.register(tmp_path,prepared,approve(tmp_path,prepared))
    path=tmp_path/'manifests/family_index_reservations.json';value=json.loads(path.read_text())
    value['reservations']['forged']={'start':2001,'n':3};put(path,value)
    result=g.check_environment(tmp_path,environment(rowref),start=2000,count=3)
    assert not result['allowed'] and 'overlaps another' in result['reason']


def test_successful_class_cannot_be_registered_for_unchanged_retry(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    put(base/'acceptance/class0.json',{'validation_passed':True,'row':prior})
    with pytest.raises(ContractError,match='validated class'):r.prepare(tmp_path,proposal)
