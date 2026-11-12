import copy
from types import SimpleNamespace as NS
import pytest
from scripts.campaign_20261110.training_pristine_probe import install_override
from scripts.campaign_20261110.contracts import ContractError,digest


def test_override_keeps_source_bytes_key_and_physical_state_unchanged(tmp_path):
    path=tmp_path/'input.json';path.write_bytes(b'original immutable source')
    campaign={'campaign_id':'campaign_20261110','split':'train','task':'disembedding','class_id':'disembedding_07',
              'runtime':{'purpose':'training_pipeline_canary'},'validation_key':{'code_sha256':'old'},
              'scene_setup':{'restoration_options':{'serialized_root_motion_v1':True}}}
    original=copy.deepcopy(campaign)
    class Loader:
        def __init__(self):
            self.tro_path=str(path);self.report={}
            self.tro=NS(meta=NS(raw={'campaign':copy.deepcopy(campaign)}),object_states={'tray':{'position':[1,2,3]}})
    initialize=Loader.__init__;install_override(NS(InstanceLoader=Loader));a=Loader()
    assert path.read_bytes()==b'original immutable source' and campaign==original
    assert a.tro.meta.raw['campaign']['validation_key']==original['validation_key']
    assert a.tro.object_states=={'tray':{'position':[1,2,3]}}
    assert a.report['physical_diagnostic_override']['source_campaign_sha256']==digest(original)
    assert a.tro.meta.raw['campaign']['scene_setup']['restoration_options']['pristine_mass_before_reset_v1']
    campaign['scene_setup']['restoration_options']['serialized_mass_before_reset_v1']=True
    with pytest.raises(ContractError,match='original fde'):Loader()
    Loader.__init__=initialize


def diagnostic_runtime(tmp_path):
    import json
    from scripts.campaign_20261110.training_pristine_probe import ALLOWED, RESTORE_OPTIONS
    from scripts.campaign_20261110.contracts import file_digest
    root=tmp_path/'campaign';base=root/'code/trunk_aaaaaaaaaaaa';runtime=root/'code/trunk_bbbbbbbbbbbb';production=root/'code/trunk_cccccccccccc'
    for p,c in [(base,'a'),(runtime,'b'),(production,'c')]:
        p.mkdir(parents=True);(p/'.snapshot_commit').write_text(c*40+'\n')
    paths=ALLOWED|{'scripts/campaign_20261110/training_canary_scope.py'}
    for name in paths:
        for p in (base,runtime,production):(p/name).parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('campaign_runtime.py'):
            (base/name).write_text('def verify_loaded(): return 0\nphysical = 1\n')
            (runtime/name).write_text('def verify_loaded(): return 1\nphysical = 1\n')
        else:
            if not name.endswith('training_canary_scope.py'):(base/name).write_text('original = True\n')
            (runtime/name).write_text('integrated = True\n')
        (production/name).write_bytes((runtime/name).read_bytes())
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    proof={'schema':'isg-pristine-mass-diagnostic-runtime-v2','source_commit':'a'*40,'candidate_commit':'b'*40,
           'production_released':False,'source_files_unchanged':True,'changed_payloads':{name:file_digest(runtime/name)for name in paths},
           'production_restoration_runtime':ref(production/'.snapshot_commit'),'diagnostic_restore_options':RESTORE_OPTIONS}
    path=root/'proof.json';path.write_text(json.dumps(proof));request={'physical_candidate':ref(path),'diagnostic_only':True,
        'production_released':False,'diagnostic_restore_options':dict(RESTORE_OPTIONS)}
    return root,base,runtime,production,request,[{'validation_key':{'code_sha256':file_digest(base/'.snapshot_commit')}}]


def test_bound_v2_diagnostic_keeps_every_other_original_payload(tmp_path):
    from scripts.campaign_20261110.training_pristine_probe import verify_candidate
    root,base,runtime,production,request,records=diagnostic_runtime(tmp_path)
    assert verify_candidate(root,request,records,runtime)['production_released'] is False
    p=runtime/'ispatialgym/unrelated.py';p.write_text('altered_physics = True\n')
    with pytest.raises(ContractError,match='unrelated implementation'):
        verify_candidate(root,request,records,runtime)


def test_production_restore_bytes_cannot_be_substituted(tmp_path):
    from scripts.campaign_20261110.training_pristine_probe import verify_candidate
    root,base,runtime,production,request,records=diagnostic_runtime(tmp_path)
    (production/'ispatialgym/oracle/campaign_restore.py').write_text('different = True\n')
    with pytest.raises(ContractError,match='exact production restoration'):
        verify_candidate(root,request,records,runtime)


def test_diagnostic_override_options_are_exact_and_not_release_authority(tmp_path):
    from scripts.campaign_20261110.training_pristine_probe import verify_candidate,RESTORE_OPTIONS
    root,base,runtime,production,request,records=diagnostic_runtime(tmp_path)
    request['diagnostic_restore_options']={**RESTORE_OPTIONS,'serialized_mass_before_reset_v1':True}
    with pytest.raises(ContractError,match='options differ'):
        verify_candidate(root,request,records,runtime)
    request['diagnostic_restore_options']=dict(RESTORE_OPTIONS);request['production_released']=True
    with pytest.raises(ContractError,match='cannot become'):
        verify_candidate(root,request,records,runtime)
    with pytest.raises(ContractError,match='unsupported diagnostic'):
        install_override(NS(InstanceLoader=None),{'diagnostic_restore_options':{'unrecognized':True}})
