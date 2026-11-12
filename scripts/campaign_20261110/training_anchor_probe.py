"""Exact prospective warm-anchor repair; never an observation-only equivalence."""
import json
from pathlib import Path
from .contracts import file_digest, require

PARENT_COMMIT='7416873efaffc87cc153cf25265d0a6e7534527c'
REVIEWED_COMMIT='28a1db54066a4d2716221361fd210b19715f9f0e'
PHYSICAL_PAYLOADS={
 'ispatialgym/oracle/campaign_batch.py':'57e2a5c1b0212394606fe9a172d9e3ac9cd46aa0f1620935bdb75d00549ec52b',
 'ispatialgym/oracle/run_isg.py':'57b1878a36990ea015d00a39e9da623617e35d3fe7c86c267d047665d4a9ac49',
 'ispatialgym/oracle/campaign_restore.py':'8b5ca6ab59eadeb954fc1c5069d746fdc31de084441b1e7561c7d49208bce26b'}
ALLOWED=set(PHYSICAL_PAYLOADS)|{
 'ispatialgym/tests/test_campaign_robot_anchor.py',
 'ispatialgym/tests/test_campaign_restore_reuse_native_lifecycle.py',
 'scripts/campaign_20261110/training_anchor_probe.py',
 'scripts/campaign_20261110/tests/test_training_anchor_probe.py',
 'scripts/campaign_20261110/training_batch_probe.py',
 'scripts/campaign_20261110/training_canary_scope.py',
 'scripts/campaign_20261110/tests/test_training_canary_scope.py'}
DELTA={'name':'sequential_native_robot_anchor_v1','reviewed_commit':REVIEWED_COMMIT,
 'boundary':'existing_native_stop_play_before_saved_scene_restore',
 'cold_physical_operations_changed':False,'warm_physical_operations_changed':True,
 'additional_physics_steps':0,'task_inputs_changed':False,'thresholds_changed':False}

def reviewed_payloads(runtime):
    require(all((runtime/name).is_file()and file_digest(runtime/name)==sha
                for name,sha in PHYSICAL_PAYLOADS.items()),'unreviewed warm-anchor physical payload')

def verify_anchor_runtime(root,request,records,runtime,verify_parent):
    require(request.get('sequential_anchor_rebase')is True and request.get('sequential_augmentation')is True and
            not request.get('sequential_checker_only')and not request.get('physical_candidate')and
            request.get('production_released')is False and request.get('benchmark_credit')==0 and
            request.get('diagnostic_control')in(None,'FRESH_A','FRESH_B','A_A','A_FAILURE_A','A_INTERRUPT_A'),
            'warm-anchor repair requires bounded prospective physics scope')
    reference=request['runtime_compatibility'];path=Path(reference['path']).resolve()
    require(path.is_relative_to(root)and file_digest(path)==reference['sha256'],'anchor runtime proof changed')
    proof=json.loads(path.read_text())
    require(proof.get('schema')=='isg-sequential-anchor-runtime-compatibility-v1'and
            proof.get('physical_delta')==DELTA and proof.get('physical_code_changes')==1 and
            proof.get('permitted_execution_kinds')==['physics_action_open_loop'],
            'anchor repair must declare exact prospective physical delta')
    parent_ref=proof['parent_runtime'];parent=Path(parent_ref['path']).resolve().parent
    require(parent.is_relative_to(root/'code')and runtime.is_relative_to(root/'code')and
            parent_ref=={'path':str(parent/'.snapshot_commit'),'sha256':file_digest(parent/'.snapshot_commit')}and
            (parent/'.snapshot_commit').read_text().strip()==PARENT_COMMIT and
            proof['candidate_runtime']=={'path':str(runtime/'.snapshot_commit'),'sha256':file_digest(runtime/'.snapshot_commit')},
            'warm-anchor parent or candidate runtime differs')
    inherited={**request,'runtime_compatibility':proof['parent_compatibility'],
               'probe_runtime_sha256':parent_ref['sha256'],'sequential_checker_only':True}
    inherited.pop('sequential_anchor_rebase')
    verify_parent(root,inherited,records,parent)
    require(all(row['validation_key']['code_sha256']==proof['native_runtime']['sha256']for row in records),
            'warm-anchor immutable native source differs')
    parent_proof=json.loads(Path(proof['parent_compatibility']['path']).read_text())
    require(parent_proof['native_runtime']==proof['native_runtime'],'warm-anchor source provenance differs')
    reviewed_payloads(runtime)
    paths={str(p.relative_to(tree))for tree in(parent,runtime)for folder in('ispatialgym','scripts')
           for p in(tree/folder).rglob('*.py')if p.is_file()}
    changed={name for name in paths if not(parent/name).is_file()or not(runtime/name).is_file()or
             file_digest(parent/name)!=file_digest(runtime/name)}
    require(changed==set(proof['changed_payloads'])and changed<=ALLOWED,
            'unapproved payload changed alongside warm-anchor repair')
    require(all((runtime/name).resolve().is_relative_to(runtime)and file_digest(runtime/name)==sha
                for name,sha in proof['changed_payloads'].items()),'warm-anchor payload changed')
    return reference
