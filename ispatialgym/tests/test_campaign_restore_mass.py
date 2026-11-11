from types import SimpleNamespace
import ast
from pathlib import Path
import pytest
from ispatialgym.oracle.campaign_restore import restore_serialized_mass_before_reset as restore


def inputs():
    meta={'campaign_id':'campaign_20261110','split':'train','class_id':'disembedding_07','task':'disembedding',
          'runtime':{'purpose':'training_pipeline_canary'},'scene_setup':{'restoration_options':{'serialized_mass_before_reset_v1':True}}}
    objects={i:SimpleNamespace(name=n,kinematic_only=False,links={},mass=m)for i,n,m in [('source','source_tray',.3876),('destination','destination_tray',.2973)]}
    bindings={i:{'scene_object_name':o.name}for i,o in objects.items()}
    records=[{'type':'mass','object':i,'mass_kg':2.}for i in objects]
    return meta,objects,bindings,records


def test_recorded_mass_applies_identically_on_first_load_and_reuse_without_steps():
    args=inputs();calls=[]
    def apply(obj,mass,link):
        before=obj.mass;obj.mass=mass;calls.append((obj.name,mass,link))
        return {'before_kg':before,'after_kg':obj.mass,'link':'base_link'}
    a=restore(*args,apply_mass_fn=apply);b=restore(*args,apply_mass_fn=apply)
    assert [r['before_kg']for r in a['objects']]==[.3876,.2973]
    assert [r['before_kg']for r in b['objects']]==[2.,2.]
    assert a['physics_steps_added']==b['physics_steps_added']==0 and len(calls)==4
    assert a['later_constraint_restore_unchanged'] and not a['pose_or_goal_changes']


@pytest.mark.parametrize('mutation',['wrong_scope','wrong_binding','wrong_model','missing_binding','nonfinite','duplicate','kinematic','missing_link'])
def test_invalid_constraint_set_never_partially_mutates(mutation):
    meta,objects,bindings,records=inputs();calls=[]
    if mutation=='wrong_scope':meta['split']='eval_standard'
    if mutation=='wrong_binding':bindings['destination']['scene_object_name']='other'
    if mutation=='wrong_model':bindings['destination']['model']='different'
    if mutation=='missing_binding':bindings.pop('destination')
    if mutation=='nonfinite':records[1]['mass_kg']=float('nan')
    if mutation=='duplicate':records.append(records[-1])
    if mutation=='kinematic':objects['destination'].kinematic_only=True
    if mutation=='missing_link':records[1]['link']='missing'
    with pytest.raises(ValueError):restore(meta,objects,bindings,records,apply_mass_fn=lambda *a:calls.append(a))
    assert calls==[]


def test_default_does_not_read_or_change_scene():
    assert restore({},None,None,None) is None
    meta,*_=inputs();meta['scene_setup']['restoration_options']['serialized_mass_before_reset_v1']=False
    assert restore(meta,None,None,None) is None


def test_readback_must_match_serialized_mass():
    with pytest.raises(ValueError,match='readback'):restore(*inputs(),apply_mass_fn=lambda *a:{'after_kg':1.})


def test_loader_applies_mass_before_native_reset_and_retains_later_constraints():
    tree=ast.parse(Path('ispatialgym/oracle/run_isg.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='InstanceLoader')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='restore')
    calls={getattr(n.func,'id',getattr(n.func,'attr','')):n.lineno for n in ast.walk(method)if isinstance(n,ast.Call)}
    assert calls['restore_serialized_mass_before_reset']<calls['load_tro_instance']<calls['_restore_constraints']
