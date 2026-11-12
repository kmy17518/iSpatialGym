import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.contracts import ContractError,file_digest
from scripts.campaign_20261110.training_batch_probe import verify_runtime_compatibility


def fixture(tmp_path):
    runtime=tmp_path/'code/trunk_frozen';runtime.mkdir(parents=True)
    (runtime/'.snapshot_commit').write_text('f'*40+'\n')
    for name in ('ispatialgym/oracle/run_isg.py','scripts/campaign_20261110/training_batch_probe.py'):
        p=runtime/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unchanged = True\n')
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    proof={'schema':'isg-exact-native-runtime-compatibility-v1','native_runtime':ref(runtime/'.snapshot_commit'),
           'physical_code_changes':0,'python_payloads':{str(p.relative_to(runtime)):file_digest(p)for p in runtime.rglob('*.py')}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    return runtime,{'runtime_compatibility':ref(path)},[{'validation_key':{'code_sha256':file_digest(runtime/'.snapshot_commit')}}]


def test_same_frozen_native_executes_observations_without_a_modified_runtime(tmp_path):
    runtime,request,records=fixture(tmp_path)
    assert verify_runtime_compatibility(tmp_path,request,records,runtime)==request['runtime_compatibility']


def test_another_source_runtime_cannot_use_exact_identity_proof(tmp_path):
    runtime,request,records=fixture(tmp_path);records[0]['validation_key']['code_sha256']='0'*64
    with pytest.raises(ContractError,match='differs from native source'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)


def test_changed_payload_with_unchanged_commit_stamp_is_rejected(tmp_path):
    runtime,request,records=fixture(tmp_path)
    (runtime/'ispatialgym/oracle/run_isg.py').write_text('changed = True\n')
    with pytest.raises(ContractError,match='payload changed'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)


def test_added_or_missing_code_cannot_hide_outside_the_manifest(tmp_path):
    runtime,request,records=fixture(tmp_path);extra=runtime/'ispatialgym/new_physics.py';extra.write_text('new = True\n')
    with pytest.raises(ContractError,match='inventory differs'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)
    extra.unlink();(runtime/'ispatialgym/oracle/run_isg.py').unlink()
    with pytest.raises(ContractError,match='inventory differs'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)


def test_symlink_to_another_runtime_cannot_satisfy_exact_payloads(tmp_path):
    runtime,request,records=fixture(tmp_path);target=runtime/'ispatialgym/oracle/run_isg.py'
    external=tmp_path/'outside.py';external.write_bytes(target.read_bytes());target.unlink();target.symlink_to(external)
    with pytest.raises(ContractError,match='payload changed'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)


def test_driver_only_observation_does_not_require_changing_physical_components(tmp_path):
    import shutil
    base=tmp_path/'code/trunk_aaaaaaaaaaaa';base.mkdir(parents=True)
    (base/'.snapshot_commit').write_text('a'*40+'\n')
    runtime=tmp_path/'code/trunk_bbbbbbbbbbbb'
    for name in ('ispatialgym/oracle/run_isg.py','ispatialgym/sampling/campaign_runtime.py',
                 'ispatialgym/multienv/campaign_rendering.py','scripts/campaign_20261110/training_batch_probe.py'):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('physical = True\n')
    shutil.copytree(base,runtime);(runtime/'.snapshot_commit').write_text('b'*40+'\n')
    relative='scripts/campaign_20261110/training_batch_probe.py';(runtime/relative).write_text('driver = True\n')
    proof={'schema':'isg-observational-runtime-compatibility-v1','base_commit':'a'*40,'observational_commit':'b'*40,
           'changed_files':[relative],'changed_payloads':{relative:file_digest(runtime/relative)}}
    p=tmp_path/'observation.json';p.write_text(json.dumps(proof));request={'runtime_compatibility':{'path':str(p),'sha256':file_digest(p)}}
    records=[{'validation_key':{'code_sha256':file_digest(base/'.snapshot_commit')}}]
    assert verify_runtime_compatibility(tmp_path,request,records,runtime)==request['runtime_compatibility']
    (runtime/'ispatialgym/oracle/run_isg.py').write_text('physical_changed = True\n')
    with pytest.raises(ContractError,match='outside observation-only'):
        verify_runtime_compatibility(tmp_path,request,records,runtime)
