import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ispatialgym.oracle import campaign_support_transit as P


def test_exact_opt_in_and_canonical_scope(monkeypatch):
    from ispatialgym.sampling import campaign_runtime as CR
    from scripts.campaign_20261110.contracts import ContractError
    objects = [NS(role='target', category='box_of_sanitary_napkins', model='humkqw', default_scale=(1.,)*3),
               NS(role='partner', category='box_of_tissues', model='ntbrtz', default_scale=(1.,)*3),
               NS(role='support', category='breakfast_table', model='rhjoby', default_scale=(1.,)*3)]
    template = NS(class_id='inter_object_relation_satisfaction-touching_01',
                  task='inter_object_relation_satisfaction-touching_small_scale', split='eval_standard', objects=objects)
    monkeypatch.setattr(CR, 'active', lambda: True)
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    context = {'leg':'pre_grasp', 'action':'grasp'}
    monkeypatch.delenv('ISPATIALGYM_TOUCHING_PREGRASP_SUPPORT_PRESENT', raising=False)
    assert not P.required(context)
    monkeypatch.setenv('ISPATIALGYM_TOUCHING_PREGRASP_SUPPORT_PRESENT', '1')
    assert P.required(context)
    assert not P.required({'leg':'pre_contact', 'action':'push'})
    for obj, key, value in [(template,'class_id','other'),(template,'split','train'),
                             (objects[0],'model','other'),(objects[1],'default_scale',(.8,)*3),
                             (objects[2],'model','other')]:
        old=getattr(obj,key);setattr(obj,key,value)
        with pytest.raises(ContractError): P.required(context)
        setattr(obj,key,old)


class FailedMotion(Exception):
    def __init__(self):
        self.reason='plan_failed';self.details={'stage':'planner'};self.message='no checked path'


def native_functions():
    # Run the actual generator branches without importing the native simulator.
    source=Path(P.__file__).with_name('execute.py')
    nodes=[n for n in ast.parse(source.read_text()).body
           if isinstance(n,ast.FunctionDef) and n.name in ('reach_pre_pose','_grasp_transit','_counted')]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    def no_campaign(*a):
        yield from ()
        return False
    scope={'__package__':'ispatialgym.oracle','SkillError':FailedMotion,
           'escape_direction':lambda *a:None, 'campaign_pregrasp_transit':no_campaign,
           '_grasp_extra_float':lambda *a:None, '_witness':lambda *a:None,
           '_cfg':lambda *a:NS(), 'nullcontext':nullcontext,
           'cartesian_before_ignoring':lambda *a:pytest.fail('must not fall back after scoped support-present failure')}
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),scope)
    return scope


@pytest.mark.parametrize('emitted_before_failure',[0,1])
def test_failed_checked_motion_never_falls_back_to_ignored_table(monkeypatch,emitted_before_failure):
    monkeypatch.setattr(P,'required',lambda context:True)
    calls=[]
    def arm_to(*args,**kwargs):
        calls.append(kwargs)
        yield from range(emitted_before_failure)
        raise FailedMotion()
    scope=native_functions();ctx={};motions=NS(arm_to=arm_to,_protect={'leg':'pre_grasp','action':'grasp'})
    with pytest.raises(FailedMotion):
        list(scope['_grasp_transit'](NS(),motions,None,{'pre_grasp':'unchanged_pose'},ctx,'right',[NS(name='table')],[]))
    assert len(calls)==1 and calls[0]['ignore_objects']==[]
    assert calls[0]['pose']=='unchanged_pose'
    assert ctx['far_pregrasp_support_present']['emitted_steps']==emitted_before_failure
    assert ctx['far_pregrasp_support_present']['ignored_support_fallback'] is False


@pytest.mark.parametrize('enabled',[False,True])
def test_far_endpoint_keeps_only_reviewed_supports_in_planner(monkeypatch,enabled):
    monkeypatch.setattr(P,'required',lambda context:enabled)
    calls=[]
    def arm_to(*args,**kwargs):
        calls.append(kwargs);yield 'checked_step'
    scope=native_functions();table=NS(name='table');through=NS(name='intentional_target');ctx={}
    result=list(scope['reach_pre_pose'](NS(arm_to=arm_to,_protect={}),None,'right','pre',[table],[through],None,ctx,NS()))
    assert result==['checked_step']
    assert calls[0]['ignore_objects']==([through] if enabled else [table,through])
    if enabled: assert ctx['far_pregrasp_support_present']['status']=='completed'
