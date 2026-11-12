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
