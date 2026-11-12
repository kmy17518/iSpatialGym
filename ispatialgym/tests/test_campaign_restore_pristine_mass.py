import copy
from types import SimpleNamespace as NS
import pytest
from scripts.campaign_20261110.contracts import ContractError
from ispatialgym.oracle.campaign_restore import capture_pristine_mass, restore_pristine_mass


def inputs():
    meta={'campaign_id':'campaign_20261110','split':'train','class_id':'disembedding_07','task':'disembedding',
          'runtime':{'purpose':'training_pipeline_canary'},'scene_setup':{'restoration_options':{'pristine_mass_before_reset_v1':True}}}
    link=NS(prim_path='/World/tray/base')
    obj=NS(name='tray',links={'base':link},kinematic_only=False)
    state={'mass_kg':.3876259029,'inertia':[[.01,0,0,0,.02,0,0,0,.03]],'usd_mass_authored':False,'usd_mass':None}
    values={link.prim_path:copy.deepcopy(state)};writes=[]
    def read(link):return copy.deepcopy(values[link.prim_path])
    def write(link,state):writes.append(copy.deepcopy(state));values[link.prim_path]=copy.deepcopy(state)
    return meta,obj,values,writes,read,write


def test_cold_actor_unchanged_and_reused_inertia_exactly_restored():
    meta,obj,values,writes,read,write=inputs()
    baseline=capture_pristine_mass(meta,[obj],read_state=read);immutable=copy.deepcopy(baseline)
    result=restore_pristine_mass(meta,[obj],baseline,read_state=read,write_state=write)
    assert writes==[] and result['actors_restored']==[]
    # Simulate either a successful or aborted episode after its normal mass constraint.
    for _ in range(2):
        values[obj.links['base'].prim_path]={'mass_kg':2.,'inertia':[[.2,0,0,0,.3,0,0,0,.4]],'usd_mass_authored':True,'usd_mass':2.}
        result=restore_pristine_mass(meta,[obj],baseline,read_state=read,write_state=write)
        assert read(obj.links['base'])=={k:v for k,v in baseline['actors'][obj.links['base'].prim_path].items()if k!='object'}
        assert result['physics_steps_added']==0 and result['later_serialized_mass_application_unchanged']
    assert baseline==immutable and len(writes)==2


def test_no_baseline_recapture_after_failed_episode():
    meta,obj,values,writes,read,write=inputs()
    with pytest.raises(ContractError,match='cannot recapture'):
        restore_pristine_mass(meta,[obj],None,read_state=read,write_state=write)
    assert writes==[]


def test_all_bindings_checked_before_any_mutation():
    meta,obj,values,writes,read,write=inputs();baseline=capture_pristine_mass(meta,[obj],read_state=read)
    values[obj.links['base'].prim_path]['mass_kg']=2
    other=NS(name='different',links={'base':obj.links['base']},kinematic_only=False)
    with pytest.raises(ContractError,match='identity differs'):
        restore_pristine_mass(meta,[other],baseline,read_state=read,write_state=write)
    assert writes==[]


def test_default_unchanged_and_conflicting_physical_options_rejected():
    meta,obj,values,writes,read,write=inputs()
    assert capture_pristine_mass({},[obj],read_state=lambda _:pytest.fail()) is None
    assert restore_pristine_mass({},[obj],None,read_state=lambda _:pytest.fail()) is None
    meta['scene_setup']['restoration_options']['serialized_mass_before_reset_v1']=True
    with pytest.raises(ContractError,match='exact bounded'):
        capture_pristine_mass(meta,[obj],read_state=read)


def test_pristine_capture_lifecycle_precedes_constraint_mutation():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'oracle/run_isg.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='InstanceLoader')
    functions={x.name:x for x in cls.body if isinstance(x,ast.FunctionDef)}
    source=lambda name:ast.unparse(functions[name])
    assert 'capture_pristine_mass' in source('build_env')
    assert 'capture_pristine_mass' not in source('attach')+source('restore')
    assert "shared.get('pristine_mass0')" in source('attach')
    restore=source('restore')
    assert restore.index('restore_pristine_mass(')<restore.index('task.load_tro_instance(')<restore.index('self._restore_constraints(')


def test_production_purpose_cannot_enable_restore_without_bound_execution():
    from scripts.campaign_20261110.contracts import digest
    meta,obj,values,writes,read,write=inputs();meta['runtime']['purpose']='training_production'
    with pytest.raises(ContractError,match='exact bounded'):
        capture_pristine_mass(meta,[obj],read_state=read)
    execution={'schema':'isg-training-restoration-execution-v1','authorization_checked':True,
        'category':'validation_canary','metadata_sha256':digest(meta)}
    baseline=capture_pristine_mass(meta,[obj],read_state=read,execution=execution)
    assert restore_pristine_mass(meta,[obj],baseline,read_state=read,write_state=write,execution=execution)['actors_restored']==[]
    altered=copy.deepcopy(meta);altered['scene_setup']['unexpected']=True
    with pytest.raises(ContractError,match='exact bounded'):
        capture_pristine_mass(altered,[obj],read_state=read,execution=execution)
    assert writes==[]
