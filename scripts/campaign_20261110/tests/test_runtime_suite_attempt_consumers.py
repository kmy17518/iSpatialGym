from pathlib import Path
import json
import pytest
from scripts.campaign_20261110 import runtime_suite_paths as P, runtime_suite_retry as Retry, runtime_validation_metrics as M
from scripts.campaign_20261110.tests.test_runtime_suite_retry import proposed,approve


def test_prepared_unregistered_attempt_cannot_enter_observers(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    assert P.class_rows(tmp_path,base,Path(prior['path']))==[Path(prior['path'])]
    with pytest.raises(Exception,match='unregistered'):
        P.paths(base,rowref['path'])


def test_registered_attempt_paths_preserve_original_and_bind_registry(tmp_path):
    base,prior,allocation,rowref,proposal=proposed(tmp_path)
    prepared=Retry.prepare(tmp_path,proposal);Retry.register(tmp_path,prepared,approve(tmp_path,prepared))
    assert P.class_rows(tmp_path,base,Path(prior['path']))==[Path(prior['path']),Path(rowref['path'])]
    assert P.paths(base,prior['path'])['allocation']==base/'allocations/class0.json'
    assert P.paths(base,rowref['path'])['allocation']==Path(rowref['path']).parent/'allocation.json'
    assert P.paths(base,rowref['path'])['members']==Path(rowref['path']).parent/'member_replays.json'
    assert P.paths(base,rowref['path'])['registration']==Path(rowref['path']).parent/'measurement_registration.json'
    p=Path(rowref['path']);v=json.loads(p.read_text());v['fresh_start']+=1;p.write_text(json.dumps(v))
    with pytest.raises(Exception,match='changed'):
        P.class_rows(tmp_path,base,Path(prior['path']))


def record(rowid,jobs):
    return {'class_id':'c','row':{'path':rowid},'jobs':jobs,'native_attempts':{'native_draws':1,'attempted_instance_slots':1},
            'distinct_demo_instance_ids':[rowid],'strict_outcomes':None,'errors':[]}


def test_cost_rollup_keeps_failed_original_and_optional_replay():
    old=[{'backend':'cluster','job_id':'1','allocated_gpu_seconds':120,'terminal':True},
         {'backend':'cluster','job_id':'2','allocated_gpu_seconds':8,'terminal':True}]
    new=[{'backend':'vast','job_id':'3','allocated_gpu_seconds':200,'terminal':True}]
    out=M.rollup_attempts([record('old',old),record('new',new)])[0]
    assert len(out['attempt_history'])==2 and out['row']['path']=='new'
    assert out['cost']['observed_allocated_gpu_hours']==328/3600
    assert out['native_attempts']['native_draws']==2
    assert len(out['jobs'])==3


def test_shared_allocation_cannot_be_double_charged_or_conflict():
    job={'backend':'cluster','job_id':'1','allocated_gpu_seconds':120,'terminal':True}
    out=M.rollup_attempts([record('old',[job]),record('new',[job])])[0]
    assert out['cost']['unique_allocations']==1
    with pytest.raises(ValueError,match='conflicting'):
        M.rollup_attempts([record('old',[job]),record('new',[{**job,'allocated_gpu_seconds':121}])])
