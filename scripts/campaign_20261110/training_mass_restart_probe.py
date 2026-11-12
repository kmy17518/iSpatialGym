"""Exact prospective mass-before-restart repair; preserve the prior anchor proof."""
import json
from pathlib import Path
from .contracts import file_digest, require

PARENT_COMMIT='2bbeebe84aaa4c1f86cb1da5ed7082d198d07314'
REVIEWED_COMMIT='924d594da42e5f5773f59b5d5e30ba21f0b5648b'
PHYSICAL_PAYLOADS={
 'ispatialgym/oracle/run_isg.py':'6693ae2667586ca75423d34635b0ad4f6d99297e44a5368c09ed6aa4f72cf354'}
ALLOWED=set(PHYSICAL_PAYLOADS)|{
 'scripts/campaign_20261110/training_mass_restart_probe.py',
 'scripts/campaign_20261110/tests/test_training_mass_restart_probe.py',
 'scripts/campaign_20261110/training_batch_probe.py'}
DELTA={'name':'sequential_pristine_mass_before_restart_v1','reviewed_commit':REVIEWED_COMMIT,
 'boundary':'before_existing_native_stop_play_using_original_pristine_capture',
 'cold_physical_operations_changed':False,'warm_physical_operations_changed':True,
 'additional_physics_steps':0,'task_inputs_changed':False,'thresholds_changed':False}

def reviewed_payloads(runtime):
    require(all((runtime/name).is_file()and file_digest(runtime/name)==sha
                for name,sha in PHYSICAL_PAYLOADS.items()),'unreviewed mass-restart physical payload')

def verify_mass_restart_runtime(root,request,records,runtime,verify_parent):
    require(request.get('sequential_mass_restart')is True and request.get('sequential_anchor_rebase')is True and request.get('sequential_augmentation')is True and
            not request.get('sequential_checker_only')and not request.get('physical_candidate')and
            request.get('production_released')is False and request.get('benchmark_credit')==0 and
            request.get('diagnostic_control')in(None,'FRESH_B'),
            'mass-restart repair requires bounded prospective physics scope')
    reference=request['runtime_compatibility'];path=Path(reference['path']).resolve()
    require(path.is_relative_to(root)and file_digest(path)==reference['sha256'],'mass-restart runtime proof changed')
    proof=json.loads(path.read_text())
    require(proof.get('schema')=='isg-sequential-mass-restart-runtime-compatibility-v1'and
            proof.get('physical_delta')==DELTA and proof.get('physical_code_changes')==1 and
            proof.get('permitted_execution_kinds')==['physics_action_open_loop'],
            'mass-restart repair must declare exact prospective physical delta')
    parent_ref=proof['parent_runtime'];parent=Path(parent_ref['path']).resolve().parent
    require(parent.is_relative_to(root/'code')and runtime.is_relative_to(root/'code')and
            parent_ref=={'path':str(parent/'.snapshot_commit'),'sha256':file_digest(parent/'.snapshot_commit')}and
            (parent/'.snapshot_commit').read_text().strip()==PARENT_COMMIT and
            proof['candidate_runtime']=={'path':str(runtime/'.snapshot_commit'),'sha256':file_digest(runtime/'.snapshot_commit')},
            'mass-restart parent or candidate runtime differs')
    inherited={**request,'runtime_compatibility':proof['parent_compatibility'],
               'probe_runtime_sha256':parent_ref['sha256']}
    inherited.pop('sequential_mass_restart')
    verify_parent(root,inherited,records,parent)
    require(all(row['validation_key']['code_sha256']==proof['native_runtime']['sha256']for row in records),
            'mass-restart immutable native source differs')
    parent_proof=json.loads(Path(proof['parent_compatibility']['path']).read_text())
    require(parent_proof['native_runtime']==proof['native_runtime'],'mass-restart source provenance differs')
    reviewed_payloads(runtime)
    paths={str(p.relative_to(tree))for tree in(parent,runtime)for folder in('ispatialgym','scripts')
           for p in(tree/folder).rglob('*.py')if p.is_file()}
    changed={name for name in paths if not(parent/name).is_file()or not(runtime/name).is_file()or
             file_digest(parent/name)!=file_digest(runtime/name)}
    require(changed==set(proof['changed_payloads'])and changed<=ALLOWED,
            'unapproved payload changed alongside mass-restart repair')
    require(all((runtime/name).resolve().is_relative_to(runtime)and file_digest(runtime/name)==sha
                for name,sha in proof['changed_payloads'].items()),'mass-restart payload changed')
    return reference
