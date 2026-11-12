import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.contracts import ContractError,file_digest
from scripts.campaign_20261110.training_batch_probe import verify_runtime_compatibility


def test_checker_derivative_rejects_physical_changes_and_wrong_scope(tmp_path):
    base=tmp_path/'code/base';runtime=tmp_path/'code/checker'
    for directory,stamp in ((base,'base'),(runtime,'checker')):
        (directory/'ispatialgym/oracle').mkdir(parents=True)
        (directory/'.snapshot_commit').write_text(stamp)
    path='ispatialgym/oracle/campaign_batch.py'
    (base/path).write_text('def restore_parking(x):\n return x\ndef saved_world_scope(x):\n return x\n')
    (runtime/path).write_text('def restore_parking(x):\n return x\ndef checked_nested_parking_provenance(x):\n return x\ndef saved_world_scope(x):\n return checked_nested_parking_provenance(x)\n')
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    proof={'schema':'isg-sequential-checker-runtime-compatibility-v1','native_runtime':ref(base/'.snapshot_commit'),
           'checker_runtime':ref(runtime/'.snapshot_commit'),'physical_code_changes':0,
           'permitted_execution_kinds':['physics_action_open_loop'],'changed_payloads':{path:file_digest(runtime/path)},
           'saved_scene_checker':{'base_sha256':file_digest(base/path),'runtime_sha256':file_digest(runtime/path)}}
    p=tmp_path/'proof.json'
    def request():
        p.write_text(json.dumps(proof));return {'runtime_compatibility':ref(p),'sequential_checker_only':True,'sequential_augmentation':True,'production_released':False}
    records=[{'validation_key':{'code_sha256':file_digest(base/'.snapshot_commit')}}]
    assert verify_runtime_compatibility(tmp_path,request(),records,runtime)==ref(p)
    with pytest.raises(ContractError,match='bounded physics'):verify_runtime_compatibility(tmp_path,{**request(),'production_released':True},records,runtime)
    (runtime/path).write_text((runtime/path).read_text().replace('def restore_parking(x):\n return x','def restore_parking(x):\n return x+1'))
    proof['changed_payloads'][path]=proof['saved_scene_checker']['runtime_sha256']=file_digest(runtime/path)
    with pytest.raises(ContractError,match='restoration or other physical'):verify_runtime_compatibility(tmp_path,request(),records,runtime)
    (runtime/'ispatialgym/oracle/run_isg.py').write_text('changed physical loader')
    proof['changed_payloads']['ispatialgym/oracle/run_isg.py']=file_digest(runtime/'ispatialgym/oracle/run_isg.py')
    with pytest.raises(ContractError,match='native physics'):verify_runtime_compatibility(tmp_path,request(),records,runtime)
