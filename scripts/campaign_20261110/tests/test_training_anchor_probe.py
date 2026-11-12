from pathlib import Path
import shutil
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_anchor_probe import reviewed_payloads,verify_anchor_runtime,PHYSICAL_PAYLOADS,DELTA

def test_reviewed_physical_payloads_are_exact_and_mutation_fails(tmp_path):
    source=Path(__file__).resolve().parents[3]
    for name in PHYSICAL_PAYLOADS:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
    reviewed_payloads(tmp_path)
    target=tmp_path/'ispatialgym/oracle/run_isg.py';target.write_text(target.read_text()+'\n# changed\n')
    with pytest.raises(ContractError,match='unreviewed warm-anchor'):reviewed_payloads(tmp_path)

@pytest.mark.parametrize('key,value',[('sequential_anchor_rebase',False),('sequential_augmentation',False),
 ('sequential_checker_only',True),('physical_candidate',{'override':True}),('production_released',True),
 ('benchmark_credit',1),('diagnostic_control','RENDER_FAILURE_A')])
def test_prospective_physical_scope_cannot_be_reused_for_release_or_renderer(tmp_path,key,value):
    request={'sequential_anchor_rebase':True,'sequential_augmentation':True,'production_released':False,'benchmark_credit':0}
    request[key]=value
    with pytest.raises(ContractError,match='bounded prospective physics'):
        verify_anchor_runtime(tmp_path,request,[],tmp_path,None)

def test_delta_is_explicitly_physical_warm_only():
    assert DELTA['warm_physical_operations_changed']is True
    assert DELTA['cold_physical_operations_changed']is False
    assert DELTA['additional_physics_steps']==0

def test_exact_parent_chain_source_inventory_and_candidate_payloads(tmp_path,monkeypatch):
    import json
    from scripts.campaign_20261110 import training_anchor_probe as module
    from scripts.campaign_20261110.contracts import file_digest
    parent=tmp_path/'code/parent';runtime=tmp_path/'code/candidate'
    for folder,stamp in ((parent,module.PARENT_COMMIT),(runtime,'candidate')):
        (folder/'ispatialgym/oracle').mkdir(parents=True);(folder/'.snapshot_commit').write_text(stamp)
    rel='ispatialgym/oracle/run_isg.py';(parent/rel).write_text('original cold and warm')
    (runtime/rel).write_text('same cold; reviewed warm rebase')
    monkeypatch.setattr(module,'PHYSICAL_PAYLOADS',{rel:file_digest(runtime/rel)})
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    native={'path':str(tmp_path/'code/native/.snapshot_commit'),'sha256':'original_source'}
    inherited=tmp_path/'parent_proof.json';inherited.write_text(json.dumps({'native_runtime':native}))
    proof={'schema':'isg-sequential-anchor-runtime-compatibility-v1','physical_delta':module.DELTA,
      'physical_code_changes':1,'permitted_execution_kinds':['physics_action_open_loop'],
      'parent_runtime':ref(parent/'.snapshot_commit'),'candidate_runtime':ref(runtime/'.snapshot_commit'),
      'parent_compatibility':ref(inherited),'native_runtime':native,'changed_payloads':{rel:file_digest(runtime/rel)}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    request={'runtime_compatibility':ref(path),'sequential_anchor_rebase':True,'sequential_augmentation':True,
             'production_released':False,'benchmark_credit':0}
    records=[{'validation_key':{'code_sha256':'original_source'}}];calls=[]
    def check_parent(root,req,rows,tree):
        assert req['sequential_checker_only']and'sequential_anchor_rebase'not in req
        assert tree==parent and req['runtime_compatibility']==ref(inherited);calls.append(tree)
    assert module.verify_anchor_runtime(tmp_path,request,records,runtime,check_parent)==ref(path)
    assert calls==[parent]
    records[0]['validation_key']['code_sha256']='different_source'
    with pytest.raises(ContractError,match='immutable native source'):
        module.verify_anchor_runtime(tmp_path,request,records,runtime,check_parent)
    records[0]['validation_key']['code_sha256']='original_source'
    extra=runtime/'ispatialgym/oracle/unreviewed.py';extra.write_text('extra native operation')
    with pytest.raises(ContractError,match='unapproved payload'):
        module.verify_anchor_runtime(tmp_path,request,records,runtime,check_parent)
    extra.unlink();proof['physical_code_changes']=0;path.write_text(json.dumps(proof));request['runtime_compatibility']=ref(path)
    with pytest.raises(ContractError,match='prospective physical delta'):
        module.verify_anchor_runtime(tmp_path,request,records,runtime,check_parent)
