import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import training_loading_boundary_probe as probe
from scripts.campaign_20261110 import training_loading_boundary_contract as contract
from scripts.campaign_20261110.tests.test_training_loading_boundary_contract import completed


def test_supervisor_only_advances_after_exact_native_restart_receipt(tmp_path,monkeypatch):
    monkeypatch.setenv('ISG_A_ONLY', 'parent_wrapper_A')
    q,expected,children=completed();request=tmp_path/'request.json';request.write_text(json.dumps(q))
    plan=contract.segment_plan(q);monkeypatch.setattr(probe,'inputs',lambda *a,**k:(q,[{'room':'A','environment':{'ISG_A_ONLY':'1'}},{'room':'B','environment':{'ISG_B_ONLY':'1'}}],plan))
    calls=[]
    def runner(command,**kwargs):
        ordinal=int(command[-1]);destination=Path(command[command.index('--output')+1]);destination.mkdir()
        (destination/'probe.json').write_text(json.dumps(children[ordinal]));segment=expected['segments'][ordinal]
        (destination/'loading_boundary.json').write_text(json.dumps({**segment,'request':contract.ref(request),
            'native_runtime':q['native_runtime'],'probe':contract.ref(destination/'probe.json')}))
        assert 0<kwargs['timeout']<=1800 and kwargs['env']['HDF5_USE_FILE_LOCKING']=='FALSE'
        if ordinal==1:assert 'ISG_A_ONLY' not in kwargs['env'] and kwargs['env']['ISG_B_ONLY']=='1'
        else:assert 'ISG_B_ONLY' not in kwargs['env'] and kwargs['env']['ISG_A_ONLY']=='1'
        calls.append(command);return SimpleNamespace(returncode=segment['returncode'])
    result=probe.run(tmp_path,request,tmp_path/'output',runner=runner)
    assert len(calls)==3 and result['world_loads']==3 and result['status']=='completed'


def test_unexpected_child_exit_preserves_failure_without_starting_next_world(tmp_path,monkeypatch):
    q,expected,children=completed();request=tmp_path/'request.json';request.write_text(json.dumps(q))
    monkeypatch.setattr(probe,'inputs',lambda *a,**k:(q,[{'environment':{}},{'environment':{}}],contract.segment_plan(q)));calls=[]
    def runner(command,**kwargs):
        destination=Path(command[command.index('--output')+1]);destination.mkdir()
        (destination/'probe.json').write_text(json.dumps(children[0]))
        (destination/'loading_boundary.json').write_text(json.dumps({**expected['segments'][0],
            'request':contract.ref(request),'native_runtime':q['native_runtime'],
            'probe':contract.ref(destination/'probe.json')}));calls.append(command)
        return SimpleNamespace(returncode=1)
    with pytest.raises(ValueError,match='exit or identity'):probe.run(tmp_path,request,tmp_path/'output',runner=runner)
    saved=json.loads((tmp_path/'output/probe.json').read_text())
    assert len(calls)==1 and saved['status']=='failed' and saved['world_loads']==0
    assert json.loads((tmp_path/'output/segment_0_exit.json').read_text())['returncode']==1


@pytest.mark.parametrize('code', [0, 1, 75])
@pytest.mark.parametrize('teardown_exit', [False, True])
def test_native_teardown_cannot_replace_declared_exit_status(code, teardown_exit):
    calls=[]
    def shutdown():
        calls.append('shutdown')
        if teardown_exit: raise SystemExit(0)
    probe.shutdown_and_exit(code, SimpleNamespace(shutdown=shutdown), exit_process=calls.append)
    assert calls==['shutdown', code]


@pytest.mark.parametrize('error', [RuntimeError('teardown failed'), SystemExit(3), KeyboardInterrupt()])
@pytest.mark.parametrize('code', [0, 75])
def test_unexpected_teardown_failure_cannot_be_reported_as_success(code, error):
    exits=[]
    def shutdown(): raise error
    probe.shutdown_and_exit(code, SimpleNamespace(shutdown=shutdown), exit_process=exits.append)
    assert exits==[1]


@pytest.mark.parametrize('corruption', [None, 'epoch', 'target', 'incomplete', 'nonzero_exit'])
def test_clean_sdk_exit_advances_only_with_exact_completed_handoff(tmp_path,monkeypatch,corruption):
    from copy import deepcopy
    monkeypatch.setenv('SLURM_JOB_ID','123')
    q,expected,children=completed();q['child_boundary_protocol']=contract.HANDOFF_PROTOCOL
    request=tmp_path/'request.json';request.write_text(json.dumps(q));plan=contract.segment_plan(q)
    monkeypatch.setattr(probe,'inputs',lambda *a,**k:(q,[{'environment':{}},{'environment':{}}],plan));calls=[]
    def runner(command,**kwargs):
        ordinal=int(command[command.index('--segment')+1]);epoch=command[command.index('--invocation-epoch')+1]
        destination=Path(command[command.index('--output')+1]);destination.mkdir();calls.append(ordinal)
        child=deepcopy(children[ordinal]);segment=deepcopy(expected['segments'][ordinal])
        next_spec=plan[ordinal+1]if ordinal+1<len(plan)else None
        segment.update(returncode=0,invocation_epoch=epoch,scheduler_job_id='123',request=contract.ref(request),
                       native_runtime=q['native_runtime'],handoff={'queue_restart_code':75,
                       'next_source':next_spec['source'],'next_request':next_spec['request']}if next_spec else None)
        if ordinal==0:
            if corruption=='epoch':segment['invocation_epoch']='0'*32
            if corruption=='target':segment['handoff']['next_request']=q['sources'][0]
            if corruption=='incomplete':child['status']='failed'
        (destination/'probe.json').write_text(json.dumps(child));segment['probe']=contract.ref(destination/'probe.json')
        (destination/'loading_boundary.json').write_text(json.dumps(segment))
        return SimpleNamespace(returncode=2 if corruption=='nonzero_exit'and ordinal==0 else 0)
    if corruption:
        with pytest.raises(ValueError):probe.run(tmp_path,request,tmp_path/'output',runner=runner)
        assert calls==[0]
    else:
        result=probe.run(tmp_path,request,tmp_path/'output',runner=runner)
        assert result['status']=='completed'and calls==[0,1,2]and result['world_loads']==3
        assert result['segments'][0]['returncode']==0
        assert result['segments'][0]['handoff']['queue_restart_code']==75
        assert json.loads((tmp_path/'output/segment_0_exit.json').read_text())['returncode']==0
