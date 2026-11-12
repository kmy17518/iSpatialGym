import json
import pytest
from scripts.campaign_20261110.training_batch_probe import episode_order,verify_runtime_compatibility
from scripts.campaign_20261110.contracts import ContractError,file_digest


def test_render_fault_scope_cannot_authorize_physics():
    r={'diagnostic_control':'RENDER_FAILURE_A','render_cleanup_only':True,'sequential_augmentation':True,'diagnostic_only':True,'production_released':False}
    assert episode_order(r,'kinematic_observation_rendering')==[0,0]
    with pytest.raises(ContractError):episode_order(r,'physics_action_open_loop')
    with pytest.raises(ContractError):episode_order({**r,'render_cleanup_only':False},'kinematic_observation_rendering')


def test_render_compatibility_rejects_any_physics_change(tmp_path):
    a=tmp_path/'code/native';b=tmp_path/'code/render'
    for directory,stamp in ((a,'native'),(b,'render')):
        (directory/'ispatialgym/oracle').mkdir(parents=True);(directory/'ispatialgym/multienv').mkdir()
        (directory/'.snapshot_commit').write_text(stamp)
        (directory/'ispatialgym/oracle/run_isg.py').write_text('physical loader')
        (directory/'ispatialgym/multienv/replay.py').write_text(stamp)
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    proof={'schema':'isg-render-cleanup-runtime-compatibility-v1','native_runtime':ref(a/'.snapshot_commit'),'cleanup_runtime':ref(b/'.snapshot_commit'),'physical_code_changes':0,'changed_payloads':{'ispatialgym/multienv/replay.py':file_digest(b/'ispatialgym/multienv/replay.py')}}
    p=tmp_path/'proof.json';p.write_text(json.dumps(proof))
    r={'runtime_compatibility':ref(p),'render_cleanup_only':True,'diagnostic_control':'RENDER_FAILURE_A'}
    rows=[{'validation_key':{'code_sha256':file_digest(a/'.snapshot_commit')}}]
    assert verify_runtime_compatibility(tmp_path,r,rows,b)==ref(p)
    (b/'ispatialgym/oracle/run_isg.py').write_text('changed loader')
    with pytest.raises(ContractError,match='native physics'):
        verify_runtime_compatibility(tmp_path,r,rows,b)
