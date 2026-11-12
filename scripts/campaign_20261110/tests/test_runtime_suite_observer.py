import json
from pathlib import Path
from scripts.campaign_20261110 import runtime_suite_observer as o
from scripts.campaign_20261110.tests.test_runtime_suite import put
from scripts.campaign_20261110.tests.test_runtime_suite_acceptance import bound


def test_first_cycle_records_missing_inputs_and_awaits_two_member_gates(tmp_path,monkeypatch):
    base,row,allocation,gate=bound(tmp_path)
    wrapper=json.loads(Path(allocation['path']).read_text());wrapper['gate']=gate
    put(Path(allocation['path']),wrapper)
    monkeypatch.setattr(o,'reconcile',lambda *args:(_ for _ in ()).throw(AssertionError('must not claim before own member gates')))
    monkeypatch.setattr(o,'source_stage',lambda *args:{'status':'awaiting_own_replay','successful_demo_reports':[{},{}]})
    state=o.observe(tmp_path,base)
    assert len(state['classes'])==100
    assert state['classes']['class0']['status']=='awaiting_member_replay'
    assert state['classes']['class1']['status']=='awaiting_authored_row'
    assert not (base/'acceptance/class0.json').exists()
    assert not (base/'acceptance_observer.lockdir').exists()


def test_one_passed_gate_cannot_hide_terminal_insufficient_demos(tmp_path,monkeypatch):
    base,row,allocation,gate=bound(tmp_path)
    wrapper=json.loads(Path(allocation['path']).read_text());wrapper['gate']=gate
    put(Path(allocation['path']),wrapper)
    monkeypatch.setattr(o,'source_stage',lambda *args:{'status':'insufficient_successful_demos',
        'successful_demo_reports':[{'path':'exact-positive-demo'}],'source_terminal_verified':True})
    monkeypatch.setattr(o,'reconcile',lambda *args:(_ for _ in ()).throw(AssertionError('no claim')))
    result=o.observe(tmp_path,base)['classes']['class0']
    assert result['status']=='insufficient_successful_demos'
    assert result['source_terminal_verified'] and result['validation_passed'] is False


def test_failed_gate_remains_failure_without_resubmission(tmp_path,monkeypatch):
    base,row,allocation,gate=bound(tmp_path)
    body=json.loads(Path(gate['path']).read_text());body['success']=False
    gate=put(Path(gate['path']),body)
    wrapper=json.loads(Path(allocation['path']).read_text());wrapper['gate']=gate
    put(Path(allocation['path']),wrapper)
    state=o.observe(tmp_path,base)
    assert state['classes']['class0']['status']=='replay_failed'
    assert state['classes']['class0']['validation_passed'] is False


def test_allocated_source_without_execution_is_not_awaiting_replay(tmp_path):
    base,row_ref,allocation_ref,gate=bound(tmp_path)
    wrapper=json.loads(Path(allocation_ref['path']).read_text());wrapper.pop('terminal_evidence')
    row=json.loads(Path(row_ref['path']).read_text())
    result=o.source_stage(tmp_path,row,wrapper)
    assert result['status']=='awaiting_worker' and not result['successful_demo_reports']
    job=Path(wrapper['allocation']['job_dir']);(job/'sample.log').write_text('native process started')
    assert o.source_stage(tmp_path,row,wrapper)['status']=='native_started'


def test_terminal_no_demo_is_not_reported_as_replay_wait(tmp_path):
    base,row_ref,allocation_ref,gate=bound(tmp_path)
    wrapper=json.loads(Path(allocation_ref['path']).read_text());row=json.loads(Path(row_ref['path']).read_text())
    job=Path(wrapper['allocation']['job_dir']);(job/'sample.log').write_text('native rejected all draws')
    assert o.source_stage(tmp_path,row,wrapper)['status']=='native_failed'
    (job/'inst').mkdir();(job/'inst/member__1000-tro_state.json').write_text('{}')
    assert o.source_stage(tmp_path,row,wrapper)['status']=='demo_failed'
